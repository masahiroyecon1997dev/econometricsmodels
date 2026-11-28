//! REの分散成分（Swamy-Arora）の推定と、その推定に使う内部のFE推定・エンティティ平均・between回帰。
//!
//! テストは`variance/tests.rs`と`variance/tests/`にテーマ別で置く。

use faer::Mat;

use super::input::{ReEffects, ReInput};
use crate::linear::common::LeastSquaresError;
use crate::panel::common::{
    BetweenMeans, PanelError, ensure_no_singleton_groups, time_varying_regressor_indices,
    two_way_time_varying_regressor_indices, validate_balanced_panel,
};
use crate::panel::fe::{FeCovType, FeEffects, FeEstimator, FeInput};
use crate::shared::error::{CommonError, ErrorInfo};
use crate::shared::error_catalog::Stage;
use crate::shared::group_codes::GroupCodes;
use crate::shared::least_squares::residual_sum_of_squares;
use crate::shared::linear_algebra::{COLLINEARITY_TOLERANCE, independent_columns_in_order};
use crate::shared::orthonormal::OrthonormalDesign;

/// エンティティ平均（between回帰・between R²用）。`y`/各`x`列のエンティティごとの単純平均と、
/// 各エンティティの観測数`T_i`（1-wayのbetween回帰とbetween R²の重み、`re-spec.md`3.1節・3.6節）。
///
/// 各`Vec`はエンティティのユニークID辞書順（コード順）で揃っている。entity IDの文字列自体は
/// 持たない（between回帰・between R²のいずれも数値だけで足りるため）。
/// `swamy_arora_variance_components`が一度だけ計算して返し、between R²が再利用する。
#[derive(Debug)]
pub(super) struct EntityMeans {
    pub(super) y: Vec<f64>,
    pub(super) x: Vec<Vec<f64>>,
    pub(super) t: Vec<f64>,
}

/// エンティティの整数コード（`GroupCodes::group_indices`）で集計して`EntityMeans`を作る。
pub(super) fn entity_means(y: &[f64], x: &[Vec<f64>], entity: &GroupCodes) -> EntityMeans {
    let groups = entity.group_indices();
    let n_groups = entity.n_groups();
    let k = x.len();
    let mut y_means = Vec::with_capacity(n_groups);
    let mut x_means: Vec<Vec<f64>> = vec![Vec::with_capacity(n_groups); k];
    let mut t = Vec::with_capacity(n_groups);
    for indices in groups.iter() {
        let t_i = indices.len() as f64;
        y_means.push(indices.iter().map(|&i| y[i]).sum::<f64>() / t_i);
        for (j, x_means_j) in x_means.iter_mut().enumerate() {
            x_means_j.push(indices.iter().map(|&i| x[j][i]).sum::<f64>() / t_i);
        }
        t.push(t_i);
    }
    EntityMeans {
        y: y_means,
        x: x_means,
        t,
    }
}

/// `re_r_squared_within`/`re_hausman_test`が内部FE推定のwithin変換済みデータを再利用する
/// 前提（`fe`はRE本体と同じ個体効果構造——1-wayは`time`なし、2-wayは`time`あり——で、
/// 切片なしの`OlsEstimator`への入力が`within_transform_one_way`/`within_transform_two_way`の
/// 全列そのもの）をdebugビルドで確認する。`swamy_arora_variance_components`を変更して
/// （重み付け・共線列の自動除外等）この前提が崩れた場合に、ビット単位で同値という
/// 再利用の根拠が黙って壊れないようにする。
pub(super) fn debug_assert_within_reuse_contract(fe: &FeEstimator, k_time_varying: usize) {
    match fe.effects() {
        FeEffects::OneWay => debug_assert!(fe.input().time_codes().is_none()),
        FeEffects::TwoWay => debug_assert!(fe.input().time_codes().is_some()),
        // REは3次元以上を作らないので、契約として確かめることが無い。
        FeEffects::MultiWay => {}
    }
    debug_assert!(!fe.within_input().has_intercept());
    debug_assert_eq!(fe.within_input().k(), k_time_varying);
    debug_assert_eq!(fe.input().x().len(), k_time_varying);
}

/// `swamy_arora_variance_components`の戻り値。
#[derive(Debug)]
pub(super) struct VarianceComponents {
    /// σ_ε²（idiosyncratic variance）。
    pub(super) sigma2_eps: f64,
    /// σ_u²（individual variance、負なら0に切り上げ済み）。
    pub(super) sigma2_u: f64,
    /// σ_λ²（time variance、負なら0に切り上げ済み）。1-wayでは常に0。
    pub(super) sigma2_lambda: f64,
    /// σ_ε²を求めた内部FE推定量（1-wayはentity、2-wayはentity + time）。
    /// `time_varying`の列だけを説明変数に持つ。
    pub(super) fe: FeEstimator,
    /// エンティティごとの平均と観測数。
    pub(super) means: EntityMeans,
    /// `input.x()`のうち時間変化のある列の添字（昇順）。`fe`の`k`個の説明変数は
    /// この順で`input.x()`の列に対応する。時間不変の列（within変換後の分散がゼロ。2-wayでは
    /// 時点間でのみ・個体間でのみ変わる列も）は含まない。
    pub(super) time_varying: Vec<usize>,
}

/// Swamy-Arora法で分散成分（σ_ε²・σ_u²）を推定する（`re-spec.md`3.1節）。
///
/// - **σ_ε²（idiosyncratic variance）**: 内部でFE推定
///   （`FeEstimator::fit`、`FeCovType::Classical`固定——`cov_type`は残差そのものには
///   影響しないため）を呼び、そのwithin回帰残差平方和とFE自身の`df_resid()`
///   （`n - k_tv - n_entities`、`k_tv`は時間変化のある列の数）から`SSR / df_resid`として求める
///   （`re-spec.md`3.2節「σ_ε²の推定はFEのwithin回帰の残差分散をそのまま利用する」、
///   RE→FE→`OlsEstimator`の委譲チェーン）。FEは分散ゼロの列を拒否するため、時間不変の列は
///   内部FEに渡さない（`time_varying_regressor_indices`。REは時間不変の列を許容する、
///   `re-spec.md`5章。`plm`が定数列を除いてwithin回帰の自由度を数えるのと同じ）。
/// - **σ_u²（individual variance）**: 1-wayは`plm`の`swar`のトレース形（`dfcor = 3`）。
///   観測数`T_i`で重み付けしたbetween回帰（`between_regression`。エンティティ平均への
///   `OrthonormalDesign`のQRによる最小二乗。REは切片を持つためFEと異なりbetween回帰にも切片が要る）の
///   重み付き残差平方和`Q_id`・列数`p`・`Σ T_i h_i`から`one_way_sigma2_u`で求める。
///   バランスパネルでは`ssr/(N - p) - σ_ε²/T`（`linearmodels`と同じ）に一致する。
///   2-way（バランスのみ）は`two_way_variance_components`のdoc参照。
///
/// **自由度は委譲先の値を使う**: σ_ε²の分母は内部FEの`df_resid()`（`n - k_tv - N`）、
/// σ_u²の`N - p`はbetween回帰が実際に残した列の数から決める。`n`・`N`・`k`から式を
/// 組み立て直さない（時間不変の列・落とした列の数え方を委譲先と一致させるため）。
///
/// **戻り値に内部で計算した`EntityMeans`も含める**: between R²（`re_r_squared_between`）が
/// 同じエンティティ平均を使うため、呼び出し側（`ReEstimator::fit`）が再利用する
/// （同じ集計を二度計算しない）。
///
/// **戻り値に内部で構築した`FeEstimator`（σ_ε²用）も含める**:
/// ハウスマン検定（`re_hausman_test`、`re-spec.md`3.7節）のwithin変換済み`X̃`は
/// このFE推定量（1-wayはentity、2-wayはentity+time）の入力から得られるため、
/// 呼び出し側（`ReEstimator::fit`）がそのまま再利用する（同じFE推定を2回計算する
/// 無駄を避ける）。
///
/// # Errors
/// - 内部FE推定が失敗した場合（singleton・自由度不足等）は、その
///   `PanelError`（FE用バリアント）をそのまま伝播する。
/// - between回帰が失敗した場合（エンティティ数が説明変数の数以下等）は
///   `PanelError::BetweenRegressionFailed`。
///
pub(super) fn swamy_arora_variance_components(
    input: &ReInput,
    effects: ReEffects,
    confidence_level: f64,
) -> Result<VarianceComponents, PanelError> {
    // faerのグローバル並列度をPar::Seqに固定する（`crate::shared::parallelism`。
    // 委譲先の`FeEstimator::fit`/`OlsEstimator::fit`自身も呼ぶが、`cargo test -p engine`で
    // この関数を直接叩く経路との統一のためここでも呼ぶ、
    // `.claude/rules/rust-style.md`「パフォーマンス」参照）。
    crate::shared::parallelism::ensure_serial();

    match effects {
        ReEffects::OneWay => one_way_variance_components(input, confidence_level),
        ReEffects::TwoWay => two_way_variance_components(input, confidence_level),
    }
}

/// 2-wayのSwamy-Arora分散成分（`re-spec.md`5章）。バランスパネル専用。
///
/// `plm::ercomp(effect = "twoways", method = "swar")`のバランスパネルの式で、3つの成分が
/// 独立な3本の式で決まる:
/// - **σ_ε²**: 内部の2-way FE推定（時間変化のある列、`k_tv`個）の残差平方和を`fe.df_resid()`
///   （`n - N - T + 1 - k_tv`）で割る。
/// - **σ_u²**: entity平均へのbetween回帰（`[1, x̄ᵢ.]`）の残差平方和を`N - K_b - 1`で割った値から
///   `σ_ε²/T`を引く（`max(0, ·)`）。
/// - **σ_λ²**: time平均へのbetween回帰（`[1, x̄.ₜ]`）の残差平方和を`T - K_b - 1`で割った値から
///   `σ_ε²/N`を引く（`max(0, ·)`）。
///
/// between回帰では、平均が単位間で変わらない列（個体間のbetween回帰における時点間でのみ
/// 変わる列、時点間のbetween回帰における個体間でのみ変わる列）を除き、`K_b`にも数えない
/// （`plm`と同じ。切片と共線になるため）。
///
/// 検証の順序は2-way FEと同じ（singleton検出→バランスパネルの検証）。
fn two_way_variance_components(
    input: &ReInput,
    confidence_level: f64,
) -> Result<VarianceComponents, PanelError> {
    let entity = input.entity_codes();
    let time = input.time_codes().ok_or(PanelError::TwoWayRequiresTime)?;
    let (fe, time_varying) = internal_fe(input, ReEffects::TwoWay, confidence_level)?;
    let ssr_within = residual_sum_of_squares(fe.residuals());
    let sigma2_eps = ssr_within / fe.df_resid() as f64;

    let entity_mean_stats = entity_means(input.y(), input.x(), entity);
    let time_means = entity_means(input.y(), input.x(), time);
    let x_scales = column_max_abs(input.x());
    let entity_fit = between_regression(
        &entity_mean_stats,
        &x_scales,
        input.x_names(),
        BetweenMeans::Entity,
        None,
    )?;
    let time_fit = between_regression(
        &time_means,
        &x_scales,
        input.x_names(),
        BetweenMeans::Time,
        None,
    )?;
    let n_periods = time.n_groups() as f64;
    let n_entities = entity.n_groups() as f64;
    let sigma2_u = (entity_fit.ssr / entity_fit.df_resid as f64 - sigma2_eps / n_periods).max(0.0);
    let sigma2_lambda =
        (time_fit.ssr / time_fit.df_resid as f64 - sigma2_eps / n_entities).max(0.0);

    Ok(VarianceComponents {
        sigma2_eps,
        sigma2_u,
        sigma2_lambda,
        fe,
        means: entity_mean_stats,
        time_varying,
    })
}

/// between回帰（`between_regression`）の結果。
#[derive(Debug)]
struct BetweenFit {
    /// 残差平方和。重み付きなら`Σ w_i ê_i²`（1-wayは`Q_id = Σ T_i ē_i²`）。
    ssr: f64,
    /// 残差自由度`n_units - p`（`p = K_b + 1`は切片込みの列数、落とした列は数えない）。
    df_resid: usize,
    /// `Σ w_i h_i`（`h_i`は重み付き設計行列のレバレッジ）。重みが`T_i`のとき
    /// `tr(B⁻¹S)`（`B = Σ T_i z̄_i z̄_i'`、`S = Σ T_i² z̄_i z̄_i'`）に等しく、不均衡1-wayの
    /// Swamy-Aroraのトレース形の分母`O - tr(B⁻¹S)`に使う。重みなしなら`p`。
    weighted_leverage: f64,
    /// 重みの和`Σ w_i`（1-wayは総観測数`O`）。重みなしなら単位数。
    total_weight: f64,
}

/// between回帰（個体平均または時点平均の`[1, x̄]`への最小二乗）。1-way・2-wayで共有する。
///
/// `weights`を渡すと各単位を`w_i`で重み付けした最小二乗（行を`√w_i`倍した設計行列のQR）にする。
/// 1-wayは`T_i`を渡し、個体平均を観測レベルに展開したプールドOLSと同じ当てはめになる
/// （`plm`の`swar`の`Between`モデル、`re-spec.md`3.1節）。2-way（バランスパネル）は重みなし。
///
/// 平均が単位間でほぼ一定の列（バランスパネルで個体平均が一定になる、時点間でのみ変わる列等）は
/// 切片と共線になるため落とし、`K_b`にも数えない（`plm`と同じ、`two_way_variance_components`のdoc参照）。
/// 最終回帰には全列が残る。切片そのもの（全観測で一定の列）と共線な列は、ここでは落ちるが
/// 最終回帰の特異性（`QuasiDemeanedRegressionFailed`）で検出される。一定かどうかの判定は`column_is_zero_variance`
/// （`fe/within.rs`の分散ゼロ検査）と同じ発想の相対閾値（平均の標準偏差が`n_units·ε·max|x̄|`以下）。
///
/// 一定の列を落とした後、残りの列の線形結合になっている列も落とす（`plm`の`lm`と同じ、
/// `independent_columns_in_order`の規則: 列順で前の列に従属する列、許容誤差`1e-7`）。個体平均を
/// 説明変数に加える相関ランダム効果（Mundlak）の指定では、`x`と`x̄`の個体平均が同じ列になる。
/// どちらを落としても残差平方和・ランク・レバレッジは変わらないため`σ_u²`は同じで、後ろの列を落とす。
/// 列の選別は重みによらない（正の重みは列のランクを変えない）。
///
/// # Errors
/// 単位数が`K_b + 1`以下なら`PanelError::BetweenRegressionFailed`（`kind`が個体平均・時点平均の
/// どちらの回帰かを持つ）。別名の列を落とした後の設計行列が特異になることは通常無い（相対`1e-7`で落とすので、
/// `OrthonormalDesign::new_checked`のランク判定（等化した`R`に対する相対`k·ε`）より手前で落とす）が、非有限値などで失敗したときも同じバリアントで返す。
fn between_regression(
    means: &EntityMeans,
    x_scales: &[f64],
    x_names: &[String],
    kind: BetweenMeans,
    weights: Option<&[f64]>,
) -> Result<BetweenFit, PanelError> {
    let n_units = means.y.len();
    let max_terms = means.t.iter().fold(0.0_f64, |acc, &t| acc.max(t));
    let non_constant: Vec<usize> = (0..means.x.len())
        .filter(|&j| !between_column_is_constant(&means.x[j], x_scales[j], max_terms))
        .collect();
    let ones = vec![1.0; n_units];
    let mut candidates: Vec<&[f64]> = vec![&ones];
    candidates.extend(non_constant.iter().map(|&j| means.x[j].as_slice()));
    let independent = independent_columns_in_order(&candidates, COLLINEARITY_TOLERANCE);
    let kept: Vec<usize> = non_constant
        .iter()
        .zip(&independent[1..])
        .filter_map(|(&j, &keep)| keep.then_some(j))
        .collect();
    let n_params = kept.len() + 1;
    if n_units <= n_params {
        return Err(PanelError::BetweenRegressionFailed {
            means: kind,
            source: LeastSquaresError::Common(CommonError::InsufficientObservations {
                n: n_units,
                k: n_params,
            }),
        });
    }
    let root_weights: Vec<f64> = match weights {
        Some(w) => w.iter().map(|v| v.sqrt()).collect(),
        None => vec![1.0; n_units],
    };
    let design_matrix = Mat::from_fn(n_units, n_params, |i, j| {
        let value = if j == 0 { 1.0 } else { means.x[kept[j - 1]][i] };
        value * root_weights[i]
    });
    let y = Mat::from_fn(n_units, 1, |i, _| means.y[i] * root_weights[i]);
    // 別名の列は上で落としてあり、残りは前の列に対する
    // 相対残差が`1e-7`より大きいので、ここで特異と判定される入力は通常無い（防御的なエラー）。
    let design = OrthonormalDesign::new_checked(&design_matrix).map_err(|err| {
        let names: Vec<String> = std::iter::once("const".to_string())
            .chain(kept.iter().map(|&j| x_names[j].clone()))
            .collect();
        PanelError::BetweenRegressionFailed {
            means: kind,
            source: LeastSquaresError::from_rank_deficient(err, &design_matrix, &names),
        }
    })?;
    let residuals = design.residuals(&y, &design.coordinates(&y));
    let weighted_leverage = match weights {
        Some(w) => design
            .row_leverages()
            .iter()
            .zip(w)
            .map(|(h, w)| h * w)
            .sum(),
        None => n_params as f64,
    };
    let total_weight = weights.map_or(n_units as f64, |w| w.iter().sum());
    Ok(BetweenFit {
        ssr: residual_sum_of_squares(&residuals),
        df_resid: n_units - n_params,
        weighted_leverage,
        total_weight,
    })
}

/// 各説明変数の元の列（単位平均ではなく観測値）の最大絶対値。between回帰の一定判定
/// （`between_column_is_constant`）のスケールに使う。
fn column_max_abs(x: &[Vec<f64>]) -> Vec<f64> {
    x.iter()
        .map(|col| col.iter().fold(0.0_f64, |acc, v| acc.max(v.abs())))
        .collect()
}

/// 単位平均の1列が単位間で一定（切片と共線）か。
///
/// 平均の丸め誤差は、1つの平均に足し合わせた項数（`max_terms`、最大の単位の観測数）と元の列の
/// スケール（`raw_scale = max|x|`）に比例する（`column_is_zero_variance`（`fe/within.rs`の分散ゼロ検査）と同じ発想の
/// 相対閾値）。平均の列自身のスケールや単位数を基準にすると、中心化・標準化した列（平均が
/// 丸め誤差程度）や、単位数が多く項数が少ない時点平均の回帰で一定の列を見逃す。
fn between_column_is_constant(column: &[f64], raw_scale: f64, max_terms: f64) -> bool {
    let n = column.len() as f64;
    let mean = column.iter().sum::<f64>() / n;
    let std_dev = (column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    std_dev <= max_terms * f64::EPSILON * raw_scale
}

/// 分散成分の推定に使う内部のFE推定量と、それに渡した説明変数（within変動のある列）の添字を作る。
/// 全ての`variance_method`で共通（ハウスマン検定・within R²・σ_ε²がこのFEを再利用する）。
///
/// 2-wayは検証の順序を2-way FEと同じにする（singleton検出→バランスパネルの検証）。
///
/// # Errors
/// 内部FE推定の失敗（singleton・自由度不足・不均衡な2-way等）は`PanelError`（FE用バリアント）
/// のまま伝播する。2-wayで`time`が無ければ`PanelError::TwoWayRequiresTime`。
pub(super) fn internal_fe(
    input: &ReInput,
    effects: ReEffects,
    confidence_level: f64,
) -> Result<(FeEstimator, Vec<usize>), PanelError> {
    match effects {
        ReEffects::OneWay => {
            let time_varying = time_varying_regressor_indices(input.x(), input.entity_codes());
            let fe = FeEstimator::fit(
                FeInput::from_re_input(input, &time_varying, ReEffects::OneWay),
                FeEffects::OneWay,
                FeCovType::Classical,
                confidence_level,
            )
            .map_err(within_for_variance_component)?;
            Ok((fe, time_varying))
        }
        ReEffects::TwoWay => {
            let entity = input.entity_codes();
            let time = input.time_codes().ok_or(PanelError::TwoWayRequiresTime)?;
            let names = input.effect_names();
            ensure_no_singleton_groups(names, &[entity, time])?;
            validate_balanced_panel(entity, time, names)?;
            let time_varying = two_way_time_varying_regressor_indices(input.x(), entity, time);
            let fe = FeEstimator::fit(
                FeInput::from_re_input(input, &time_varying, ReEffects::TwoWay),
                FeEffects::TwoWay,
                FeCovType::Classical,
                confidence_level,
            )
            .map_err(within_for_variance_component)?;
            Ok((fe, time_varying))
        }
    }
}

/// 内部FE推定の失敗のうち、REの利用者に「何の計算か」が伝わらないもの（自由度不足と、
/// 内部の最小二乗・F検定の失敗）に、分散成分を推定するwithin回帰の文脈を付ける。
/// singletonや不均衡な2-wayは、REにも通じるデータの事実を述べるのでそのまま返す。
fn within_for_variance_component(err: PanelError) -> PanelError {
    let wrappable = matches!(
        err,
        PanelError::InsufficientDegreesOfFreedom { .. }
            | PanelError::WithinRegressionFailed { .. }
            | PanelError::FTestFailed { .. }
    );
    // `confidence_level`の範囲違反のように、どの計算で起きても同じ入力の誤りは包まない。
    if wrappable && err.report().code().spec().stage_allowed {
        err.in_stage(Stage::VarianceComponentWithinRegression)
    } else {
        err
    }
}

/// 1-wayのSwamy-Arora分散成分（`swamy_arora_variance_components`のdoc参照）。
fn one_way_variance_components(
    input: &ReInput,
    confidence_level: f64,
) -> Result<VarianceComponents, PanelError> {
    // σ_ε²: 内部1-way FE推定のwithin回帰残差を再利用する（`re-spec.md`3.2節）。
    let (fe, time_varying) = internal_fe(input, ReEffects::OneWay, confidence_level)?;

    let fe_residuals = fe.residuals();
    let ssr_within: f64 = (0..fe_residuals.nrows())
        .map(|i| {
            let r = *fe_residuals.get(i, 0);
            r * r
        })
        .sum();
    let sigma2_eps = ssr_within / fe.df_resid() as f64;

    // σ_u²: 観測数`T_i`で重み付けしたbetween回帰（エンティティ平均、切片あり）とトレース形
    // （`plm`の`swar`、`dfcor = 3`）。平均が単位間で一定の列（バランスパネルで時点間でのみ
    // 変わる列等）は切片と共線になるため、2-wayと同じく落として`K_b`にも数えない
    // （`between_regression`、`plm`と同じ）。
    let means = entity_means(input.y(), input.x(), input.entity_codes());
    let between = between_regression(
        &means,
        &column_max_abs(input.x()),
        input.x_names(),
        BetweenMeans::Entity,
        Some(&means.t),
    )?;
    let sigma2_u = one_way_sigma2_u(&between, sigma2_eps);

    Ok(VarianceComponents {
        sigma2_eps,
        sigma2_u,
        sigma2_lambda: 0.0,
        fe,
        means,
        time_varying,
    })
}

/// 1-wayのσ_u²（`plm`の`swar`のトレース形、`re-spec.md`3.1節）。
///
/// `M·(σ_ε², σ_u²)ᵀ = (Q_w, Q_id)ᵀ`の2行目
/// `(N - p)·σ_ε² + (O - tr(B⁻¹S))·σ_u² = Q_id`を、1行目で求めた`σ_ε²`で解き、負なら0にする。
/// `Q_id`は`T_i`重み付きbetween回帰の残差平方和、`tr(B⁻¹S) = Σ T_i h_i`、`O = Σ T_i`
/// （どれも`between`が同じ重みから求めた値）。
///
/// 全個体の観測数が等しい（`T_i = T`）と`O - tr(B⁻¹S) = T(N - p)`・`Q_id = T·SSR_b`で、
/// バランスパネルの簡約形`SSR_b/(N - p) - σ_ε²/T`（`linearmodels`・`plm`の`dfcor = 2`）に
/// 代数的に一致する。そのためバランス・不均衡で式を切り替えない。分母は`N > p`なら正
/// （`h_i ≤ 1`・`Σ h_i = p`から、`Σ T_i h_i`は大きい方から`p`個の`T_i`の和を超えない）。
fn one_way_sigma2_u(between: &BetweenFit, sigma2_eps: f64) -> f64 {
    let numerator = between.ssr - sigma2_eps * between.df_resid as f64;
    let denominator = between.total_weight - between.weighted_leverage;
    debug_assert!(denominator > 0.0, "O - Σ T_i h_i = {denominator}");
    (numerator / denominator).max(0.0)
}

#[cfg(test)]
mod tests;
