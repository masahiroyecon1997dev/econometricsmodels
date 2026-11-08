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
use crate::shared::least_squares::{least_squares, residual_sum_of_squares};
use crate::shared::linear_algebra::{COLLINEARITY_TOLERANCE, independent_columns_in_order};

/// エンティティ平均（between回帰・between R²用）。`y`/各`x`列のエンティティごとの単純平均と、
/// 各エンティティの観測数`T_i`（`re-spec.md`3.1節の調和平均`t_bar`計算に使う）。
///
/// 各`Vec`はエンティティのユニークID辞書順（コード順）で揃っている。entity IDの文字列自体は
/// 持たない（between回帰・`t_bar`計算・between R²のいずれも数値だけで足りるため）。
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
/// - **σ_u²（individual variance）**: between回帰（`between_ssr_and_df`。エンティティ平均への
///   `least_squares`。REは切片を持つためFEと異なりbetween回帰にも切片が要る）のSSRと、
///   その自由度（エンティティ数 - 切片込みの列数）から、
///   調和平均`t_bar = n_entities / Σ(1/T_i)`を使う標準式
///   `max(0, ssr/df_resid - σ_ε²/t_bar)`で求める。
///
/// **`k`規約についての注記**: `panel-common.md`
/// `re-spec.md`3.1節に書かれている式（σ_ε²分母`n-k-n_entities+1`、σ_u²分母`n_entities-k`）は、
/// `linearmodels`ソースの`nvar`（切片を含む列数）表記をそのまま転記したものである。
/// このプロジェクトのFE/OLSの`k`規約（傾き係数のみ、切片を含まない）では、分母の
/// 「+1」「-1」を式に手で足し引きする必要はない——FE/OLS双方の委譲先が返す実際の
/// `df_resid`相当の値（`FeEstimator::df_resid()`・`OlsInput::nobs()-k()`）をそのまま
/// 使えば自動的に一致する（`linearmodels`との数値完全一致を複数の乱数・手動データで
/// 実地検証済み）。このためFE/OLSへの委譲を経ず`n`・`n_entities`・`k`から直接式を
/// 組み立てる実装はしない（委譲先の状態を信頼できるソースとして再利用する）。
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
    // この関数を直接叩く経路との統一のためここでも呼ぶ、`engine/src/panel/CLAUDE.md`
    // 「faerのグローバル並列度」参照）。
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
    let (ssr_entity, df_entity) = between_ssr_and_df(
        &entity_mean_stats,
        &x_scales,
        input.x_names(),
        BetweenMeans::Entity,
    )?;
    let (ssr_time, df_time) =
        between_ssr_and_df(&time_means, &x_scales, input.x_names(), BetweenMeans::Time)?;
    let n_periods = time.n_groups() as f64;
    let n_entities = entity.n_groups() as f64;
    let sigma2_u = (ssr_entity / df_entity as f64 - sigma2_eps / n_periods).max(0.0);
    let sigma2_lambda = (ssr_time / df_time as f64 - sigma2_eps / n_entities).max(0.0);

    Ok(VarianceComponents {
        sigma2_eps,
        sigma2_u,
        sigma2_lambda,
        fe,
        means: entity_mean_stats,
        time_varying,
    })
}

/// between回帰（個体平均または時点平均の`[1, x̄]`への最小二乗）の残差平方和と
/// 残差自由度`n_units - K_b - 1`。1-way・2-wayで共有する。
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
/// どちらを落としても残差平方和とランクは変わらないため`σ_u²`は同じで、後ろの列を落とす。
///
/// # Errors
/// 単位数が`K_b + 1`以下なら`PanelError::BetweenRegressionFailed`（`kind`が個体平均・時点平均の
/// どちらの回帰かを持つ）。別名の列を落とした後の設計行列が特異になることは通常無い（相対`1e-7`で落とすので、
/// `least_squares`の判定（約`1e-15`）より厳しい）が、非有限値などで失敗したときも同じバリアントで返す。
fn between_ssr_and_df(
    means: &EntityMeans,
    x_scales: &[f64],
    x_names: &[String],
    kind: BetweenMeans,
) -> Result<(f64, usize), PanelError> {
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
    let design = Mat::from_fn(n_units, n_params, |i, j| {
        if j == 0 { 1.0 } else { means.x[kept[j - 1]][i] }
    });
    let y = Mat::from_fn(n_units, 1, |i, _| means.y[i]);
    // 別名の列は上で落としてあり、残りは前の列に対する
    // 相対残差が`1e-7`より大きいので、ここで特異と判定される入力は通常無い（防御的なエラー）。
    let fit = least_squares(&design, &y).map_err(|err| {
        let names: Vec<String> = std::iter::once("const".to_string())
            .chain(kept.iter().map(|&j| x_names[j].clone()))
            .collect();
        PanelError::BetweenRegressionFailed {
            means: kind,
            source: LeastSquaresError::from_rank_deficient(err, &design, &names),
        }
    })?;
    Ok((residual_sum_of_squares(&fit.residuals), n_units - n_params))
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

    // σ_u²: between回帰（エンティティ平均、切片あり）。平均が単位間で一定の列（バランスパネルで
    // 時点間でのみ変わる列等）は切片と共線になるため、2-wayと同じく落として`K_b`にも数えない
    // （`between_ssr_and_df`、`plm`と同じ）。
    let means = entity_means(input.y(), input.x(), input.entity_codes());
    let n_entities = means.y.len();
    let (ssr_between, df_resid_between) = between_ssr_and_df(
        &means,
        &column_max_abs(input.x()),
        input.x_names(),
        BetweenMeans::Entity,
    )?;

    let t_bar = n_entities as f64 / means.t.iter().map(|t_i| 1.0 / t_i).sum::<f64>();
    let sigma2_u = (ssr_between / df_resid_between as f64 - sigma2_eps / t_bar).max(0.0);

    Ok(VarianceComponents {
        sigma2_eps,
        sigma2_u,
        sigma2_lambda: 0.0,
        fe,
        means,
        time_varying,
    })
}

#[cfg(test)]
mod tests;
