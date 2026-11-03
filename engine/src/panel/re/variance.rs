//! REの分散成分（Swamy-Arora）の推定と、その推定に使う内部のFE推定・エンティティ平均・between回帰。

use faer::Mat;

use super::input::{ReEffects, ReInput};
use crate::linear::common::LeastSquaresError;
use crate::panel::common::{
    BetweenMeans, PanelDimension, PanelError, reject_singleton_group,
    time_varying_regressor_indices, two_way_time_varying_regressor_indices,
    validate_balanced_panel,
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
        FeEffects::OneWay => debug_assert!(fe.input().time_keys().is_none()),
        FeEffects::TwoWay => debug_assert!(fe.input().time_keys().is_some()),
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
            reject_singleton_group(
                PanelDimension::Entity,
                names.get(PanelDimension::Entity),
                entity,
            )?;
            reject_singleton_group(PanelDimension::Time, names.get(PanelDimension::Time), time)?;
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
mod tests {
    use super::super::test_support::*;
    use super::super::transform::compute_two_way_theta;
    use super::*;

    use crate::panel::re::ReCovType;
    use crate::panel::re::ReEstimator;

    // ── swamy_arora_variance_components ─────────────────────────────────────

    #[test]
    fn swamy_arora_variance_components_matches_linearmodels_reference() {
        // 手動データ（不均衡パネル、entity a: T=3, b: T=2, c: T=2）。
        // `linearmodels.RandomEffects`（Python）で実地検証済みの値と比較する
        // （`RandomEffects(y, [const, x1]).fit().variance_decomposition`）。
        // between回帰は切片+傾き1個の2パラメータのため、`n_entities > k+1`（3章参照）を
        // 満たすには最低3エンティティが必要（2エンティティだと`neffects-nvar=0`で
        // 除算不能になることを`linearmodels`自身でも実地確認済み）。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let VarianceComponents {
            sigma2_eps,
            sigma2_u,
            ..
        } = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();

        assert!(
            (sigma2_eps - 1.132_352_941_176_471_5).abs() < 1e-9,
            "sigma2_eps = {sigma2_eps}"
        );
        assert!(
            (sigma2_u - 6.537_912_784_161_284).abs() < 1e-9,
            "sigma2_u = {sigma2_u}"
        );
    }

    #[test]
    fn swamy_arora_variance_components_clips_negative_sigma2_u_to_zero() {
        // entity間のy平均のばらつきが、within回帰から推定したσ_ε²/t_barに対して
        // 十分小さいため、素朴な式ではσ_u²が負になるが`max(0, ...)`で0にクリップされる
        // （`linearmodels`と同じ挙動、`re-spec.md`3.7節のハウスマン統計量の負値と同型の
        // 「有限標本でのPSD仮定崩れ」）。エンティティ間のx1平均はわずかに異なる値にし、
        // between回帰の設計行列が特異にならないようにする。`linearmodels`で実地検証済み
        // （`variance_decomposition["Effects"] == 0.0`）。
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c"]);
        let x1 = vec![1.0, 2.0, 3.0, 1.2, 2.1, 2.9, 0.9, 2.2, 3.1];
        let y = [2.0, 4.0, 6.0, 2.3, 4.1, 5.9, 1.8, 4.3, 6.2];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let VarianceComponents {
            sigma2_eps,
            sigma2_u,
            ..
        } = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();

        assert!(
            (sigma2_eps - 0.005_868_778_280_543_04).abs() < 1e-9,
            "sigma2_eps = {sigma2_eps}"
        );
        assert_eq!(sigma2_u, 0.0);
    }

    #[test]
    fn swamy_arora_variance_components_does_not_panic_when_entity_means_are_all_zero() {
        // 全エンティティの`ȳ_i.`が完全に一致し、かつその共通値がちょうど0のデータ。between回帰は
        // 「切片=0・傾き=0」という完全な当てはめ（`SSR_between=0`）になる。between回帰が
        // 係数の標準誤差・t統計量を計算していた頃は、切片・傾きの標準誤差が0で
        // `stat=0.0/0.0=NaN`となり`StudentsT::cdf(NaN)`がパニックしていた（`shared/inference.rs`の
        // `compute_inference_stat_returns_nan_p_value_without_panicking_when_coef_and_se_are_both_zero`
        // で別途固定）。現在のbetween回帰は残差平方和しか使わないため推論統計量を計算せず、
        // `σ_u² = max(0, 0 - σ_ε²/t_bar) = 0`で成功する。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let x = vec![1.0, 3.0, 2.0, 6.0, 1.0, 4.0];
        let y = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
        let input =
            ReInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let vc = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();

        assert_eq!(vc.sigma2_u, 0.0);
    }

    #[test]
    fn swamy_arora_variance_components_propagates_fe_singleton_error() {
        // entity "c"は1観測のみ（singleton）。内部FE推定（1-way）の
        // `PanelError::SingletonGroup`がそのまま伝播することを確認する。
        let entity = strings(&["a", "a", "c"]);
        let x1 = vec![1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 3.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                column: "entity".to_string(),
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn swamy_arora_variance_components_returns_between_regression_failed_when_entities_are_insufficient()
     {
        // between回帰は切片+傾き1個で2パラメータ。エンティティ数が2つだけだと
        // `n_entities <= k`（`OlsEstimator::fit`自身の`n<=k`検証）で失敗する。
        // singletonにならないよう各エンティティは2観測以上にする。
        let entity = strings(&["a", "a", "b", "b"]);
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 2.0, 3.0, 5.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95);

        assert!(matches!(
            result,
            Err(PanelError::BetweenRegressionFailed { .. })
        ));
    }

    #[test]
    fn swamy_arora_variance_components_passes_only_time_varying_columns_to_internal_fe() {
        // 時間不変の`z`は内部FEに渡さない（FEは分散ゼロの列を拒否する）。σ_ε²は
        // `z`を含めても含めなくても同じ（within変動のある列だけで決まる）で、
        // 時間変化のある列の添字は列順どおり。
        let with_z = swamy_arora_variance_components(
            &time_invariant_input(&["z", "x1", "x2"]),
            ReEffects::OneWay,
            0.95,
        )
        .unwrap();
        let without_z = swamy_arora_variance_components(
            &time_invariant_input(&["x1", "x2"]),
            ReEffects::OneWay,
            0.95,
        )
        .unwrap();

        assert_eq!(with_z.time_varying, vec![1, 2]);
        assert_eq!(without_z.time_varying, vec![0, 1]);
        assert_eq!(with_z.fe.input().x_names(), &strings(&["x1", "x2"]));
        assert_eq!(with_z.sigma2_eps, without_z.sigma2_eps);
        // plm: ercomp(m)$sigma2["idios"]（`z`を含むモデル）。自由度は`n - N - k_tv`。
        assert!((with_z.sigma2_eps - 0.201_246_585_026_864).abs() < 1e-12);
    }

    #[test]
    fn between_regression_drops_a_column_that_repeats_an_earlier_one() {
        // 個体平均の回帰で`x1bar`の個体平均は`x1`のそれと同じ列になる（`plm`の`lm`は後ろの列を
        // 落とし、残差平方和・ランクは変わらない）。落とした列を数えず、`σ_u²`の自由度が変わらない。
        let with_bar = mundlak_input(&["x1", "x2", "x1bar"], false);
        let without = mundlak_input(&["x1", "x2"], false);
        let stats = |input: &ReInput| {
            let means = entity_means(input.y(), input.x(), input.entity_codes());
            between_ssr_and_df(
                &means,
                &column_max_abs(input.x()),
                input.x_names(),
                BetweenMeans::Entity,
            )
            .unwrap()
        };
        let (ssr_bar, df_bar) = stats(&with_bar);
        let (ssr, df) = stats(&without);
        assert_eq!(df_bar, df);
        assert_close(ssr_bar, ssr, 1e-12, "ssr");
        assert!(ssr > 0.0);
        // 列順を入れ替えても（落ちる列が`x1`に変わっても）同じ。
        let reordered = mundlak_input(&["x1bar", "x1", "x2"], false);
        let (ssr_re, df_re) = stats(&reordered);
        assert_eq!(df_re, df);
        assert_close(ssr_re, ssr, 1e-12, "ssr reordered");
    }

    #[test]
    fn one_way_balanced_panel_drops_time_only_column_from_between_regression_like_plm() {
        // plm: `plm(y ~ x1 + xt, model = "random", effect = "individual")`。`xt`は時点間でのみ
        // 変わる列で、バランスパネルでは個体平均が全個体で一致し切片と共線になる。between回帰から
        // 落とし`K_b`にも数えない（plmと同じ）。最終回帰には`xt`も残り、係数が推定される。
        let c = two_way_columns();
        let input = ReInput::from_columns(
            &c.y,
            &[c.x1.clone(), c.xt.clone()],
            strings(&["x1", "xt"]),
            &c.entity,
            None,
            "y".into(),
        )
        .unwrap();

        let vc = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();
        assert_close(vc.sigma2_eps, 5.558_492_530_031_27, 1e-9, "sigma2_eps");
        assert_close(vc.sigma2_u, 2.781_216_088_364_69, 1e-9, "sigma2_u");

        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
        let params = [
            0.117_907_102_600_895,
            0.400_160_538_463_792,
            -1.029_152_589_790_345,
        ];
        let std_errors = [
            0.905_185_788_075_077,
            0.851_443_915_446_071,
            0.510_706_679_841_924,
        ];
        for j in 0..3 {
            assert_close(*re.estimator().params().get(j, 0), params[j], 1e-9, "param");
            assert_close(*re.std_errors().get(j, 0), std_errors[j], 1e-9, "std_err");
        }
        assert_eq!(re.df_resid(), 21);
        // ハウスマン検定は`xt`の`x̃t`が定数項と共線で識別できないため外し、`x1`だけを検定する
        // （補助回帰`y* ~ 1 + x1* + xt* + x̃1`のWald統計量をnumpyで独立に計算した値）。
        assert_eq!(re.hausman_df(), Some(1));
        assert_close(
            re.hausman_statistic().unwrap(),
            0.293_444_752_111_572_3,
            1e-9,
            "hausman",
        );
    }

    #[test]
    fn one_way_balanced_panel_keeps_entity_only_column_in_between_regression() {
        // plm: `y ~ x1 + xi + xt`。`xi`（個体間でのみ変わる列）は個体平均でも変動するため
        // between回帰に残り、`xt`だけが落ちる。
        let c = two_way_columns();
        let input = ReInput::from_columns(
            &c.y,
            &[c.x1.clone(), c.xi.clone(), c.xt.clone()],
            strings(&["x1", "xi", "xt"]),
            &c.entity,
            None,
            "y".into(),
        )
        .unwrap();

        let vc = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();
        assert_close(vc.sigma2_u, 1.808_007_274_071_92, 1e-9, "sigma2_u");

        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
        let params = [
            0.139_211_658_956_512_4,
            0.455_007_744_295_883_3,
            -0.053_752_290_510_623_2,
            -1.036_241_827_179_690_7,
        ];
        for (j, expected) in params.iter().enumerate() {
            assert_close(*re.estimator().params().get(j, 0), *expected, 1e-9, "param");
        }
        assert_eq!(re.df_resid(), 20);
        assert_eq!(re.hausman_df(), Some(1));
        assert_close(
            re.hausman_statistic().unwrap(),
            2.587_055_364_739_007,
            1e-9,
            "hausman",
        );
    }

    #[test]
    fn two_way_variance_components_and_theta_match_plm() {
        // plm: ercomp(m)$sigma2（idios, id, time）・ercomp(m)$theta（id, time, total）。
        // `y ~ x1`（両次元で変動する列だけ）と、個体間でのみ変わる`xi`・時点間でのみ変わる`xt`を
        // 加えた`y ~ x1 + xi + xt`。後者はbetween回帰が変動のない列を落とす規則
        // （個体平均の回帰から`xt`、時点平均の回帰から`xi`）の確認でもある。
        for (names, sigma2, theta) in [
            (
                vec!["x1"],
                [
                    0.710_563_608_206_315,
                    3.993_198_318_827_93,
                    1.831_135_868_289_6,
                ],
                [
                    0.793_623_729_209_202,
                    0.753_533_894_195_726,
                    0.707_406_705_465_165,
                ],
            ),
            (
                vec!["x1", "xi", "xt"],
                [
                    0.710_563_608_206_315,
                    3.019_989_504_528_157,
                    2.596_736_326_025_116_6,
                ],
                [
                    0.764_301_210_325_029,
                    0.791_153_185_965_249,
                    0.713_712_012_990_144,
                ],
            ),
        ] {
            let input = two_way_input(&names);
            let vc = swamy_arora_variance_components(&input, ReEffects::TwoWay, 0.95).unwrap();
            let label = format!("{names:?}");
            assert_close(
                vc.sigma2_eps,
                sigma2[0],
                1e-9,
                &format!("{label} sigma2_eps"),
            );
            assert_rel(vc.sigma2_u, sigma2[1], 1e-9, &format!("{label} sigma2_u"));
            assert_rel(
                vc.sigma2_lambda,
                sigma2[2],
                1e-9,
                &format!("{label} sigma2_lambda"),
            );
            let t = compute_two_way_theta(vc.sigma2_eps, vc.sigma2_u, vc.sigma2_lambda, 6, 4);
            assert_close(t.entity, theta[0], 1e-9, &format!("{label} theta_entity"));
            assert_close(t.time, theta[1], 1e-9, &format!("{label} theta_time"));
            assert_close(t.total, theta[2], 1e-9, &format!("{label} theta_total"));
        }
    }

    #[test]
    fn one_way_exposed_variance_components_are_clipped_at_zero() {
        // 個体効果の無いDGPでは負の推定が0に切り上げられ、公開される`σ_u²`は0、`θ_i`はすべて0
        // （変換なし、pooled OLSに一致）。
        let c = two_way_columns();
        let y: Vec<f64> =
            c.x1.iter()
                .enumerate()
                .map(|(r, v)| 2.0 * v + 0.1 * ((r * 7) % 5) as f64)
                .collect();
        let input = ReInput::from_columns(
            &y,
            std::slice::from_ref(&c.x1),
            strings(&["x1"]),
            &c.entity,
            None,
            "y".into(),
        )
        .unwrap();
        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
        assert_eq!(re.sigma2_u(), 0.0);
        assert!(re.theta_by_entity().unwrap().iter().all(|&t| t == 0.0));
    }

    #[test]
    fn exposed_variance_components_are_clipped_at_zero() {
        // 負の推定値は0に切り上げた値（θの計算に使った値）を返す。グループ効果が無いDGPでは
        // 両方0になり、θもすべて0（変換なし）。
        let c = two_way_columns();
        let y: Vec<f64> =
            c.x1.iter()
                .enumerate()
                .map(|(r, v)| 2.0 * v + 0.1 * ((r * 7) % 5) as f64)
                .collect();
        let input = ReInput::from_columns(
            &y,
            std::slice::from_ref(&c.x1),
            strings(&["x1"]),
            &c.entity,
            Some(&c.time),
            "y".into(),
        )
        .unwrap();
        let re = ReEstimator::fit(input, ReEffects::TwoWay, ReCovType::Classical, 0.95).unwrap();
        assert_eq!(re.sigma2_u(), 0.0);
        assert_eq!(re.sigma2_lambda(), Some(0.0));
        assert_eq!(re.two_way_theta(), Some((0.0, 0.0, 0.0)));
    }

    #[test]
    fn between_column_is_constant_follows_the_raw_scale_and_the_number_of_terms() {
        // 平均が丸め誤差程度に揺れる（元の列は一定）場合、元の列のスケールと項数が基準なら一定と判定する。
        let noise = [1.2e-17, -3.4e-17, 0.0, 2.2e-17];
        assert!(between_column_is_constant(&noise, 1.0, 5.0));
        // 平均の列自身のスケール（~1e-17）を基準にすると見逃す。
        assert!(!between_column_is_constant(&noise, 3.4e-17, 1.0));
        // 本物の変動（元のスケールの1e-3）は一定とみなさない。
        let varying = [1.0, 1.001, 0.999, 1.0];
        assert!(!between_column_is_constant(&varying, 1.0, 5.0));
        // 完全に一定。
        assert!(between_column_is_constant(&[0.3; 7], 0.3, 1.0));
    }

    #[test]
    fn one_way_unbalanced_panel_keeps_time_only_column_in_between_regression() {
        // 1行欠けて不均衡になると個体平均が一定でなくなり、`xt`はbetween回帰に残る
        // （`K_b = 2`、自由度`6 - 3 = 3`。バランスなら`xt`が落ちて自由度は`6 - 2 = 4`）。
        let rows: Vec<usize> = (0..24).filter(|&r| r != 5).collect();
        let unbalanced = subset_input(&rows, &["x1", "xt"], false);
        let balanced = subset_input(&(0..24).collect::<Vec<_>>(), &["x1", "xt"], false);
        let df = |input: &ReInput| {
            let means = entity_means(input.y(), input.x(), input.entity_codes());
            between_ssr_and_df(
                &means,
                &column_max_abs(input.x()),
                input.x_names(),
                BetweenMeans::Entity,
            )
            .unwrap()
            .1
        };
        assert_eq!(df(&balanced), 4);
        assert_eq!(df(&unbalanced), 3);

        // ハウスマン検定でも`x̃t`が識別できる（θが個体ごとに異なる）ため両方検定する。
        let re =
            ReEstimator::fit(unbalanced, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
        assert_eq!(re.hausman_df(), Some(2));
    }

    #[test]
    fn one_way_between_regression_counts_only_columns_that_still_vary() {
        // 3エンティティ×3時点。`x1 + xt`は`K_b = 1`（`xt`は落ちる）で`n_entities = 3 > 2`のため成功し、
        // 2エンティティなら`n_entities <= K_b + 1`で失敗する。
        // 落とさない場合は`n_entities <= k + 1 = 3`で失敗する。
        let build = |n_entities: usize| {
            let rows: Vec<usize> = (0..n_entities * 4).collect();
            // 4時点のうち先頭3時点だけ使う（`r % 4 < 3`）。
            let rows: Vec<usize> = rows.into_iter().filter(|r| r % 4 < 3).collect();
            subset_input(&rows, &["x1", "xt"], false)
        };
        assert!(swamy_arora_variance_components(&build(3), ReEffects::OneWay, 0.95).is_ok());
        assert!(matches!(
            swamy_arora_variance_components(&build(2), ReEffects::OneWay, 0.95),
            Err(PanelError::BetweenRegressionFailed { .. })
        ));
    }

    #[test]
    fn re_estimator_fit_propagates_between_regression_failed_error() {
        // `swamy_arora_variance_components_returns_between_regression_failed_when_
        // entities_are_insufficient`と同じデータ（n_entities=2, k=1で between回帰が
        // 特異になる）。`ReEstimator::fit`が分散成分推定の失敗をそのまま伝播することを
        // 確認する。
        let entity = strings(&["a", "a", "b", "b"]);
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 2.0, 3.0, 5.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95);

        assert!(matches!(
            result,
            Err(PanelError::BetweenRegressionFailed { .. })
        ));
    }

    /// 内部FEの失敗のうち、REの利用者に文脈が伝わらない3種だけが「分散成分を推定するwithin回帰」の
    /// `stage`で包まれる。データの事実を述べるもの、`stage`を付けてはいけない`code`は包まない。
    #[test]
    fn within_for_variance_component_wraps_only_stage_capable_internal_fe_errors() {
        use crate::linear::common::LeastSquaresError;

        let stage = Stage::VarianceComponentWithinRegression;
        let wrapped: [fn() -> PanelError; 3] = [
            || PanelError::InsufficientDegreesOfFreedom {
                n_obs: 6,
                n_entities: 3,
                n_periods: None,
                k: 3,
            },
            || PanelError::WithinRegressionFailed {
                source: LeastSquaresError::SingularMatrix {
                    columns: vec!["const".to_string()],
                },
            },
            || PanelError::FTestFailed {
                source: LeastSquaresError::Common(CommonError::NearSingularMatrix {
                    matrix: crate::shared::error::MatrixKind::CoefficientCovariance,
                }),
            },
        ];
        for make in wrapped {
            let result = within_for_variance_component(make());
            assert_eq!(result, make().in_stage(stage.clone()));
            assert_eq!(result.report().stage(), Some(&stage));
        }
        let untouched: [fn() -> PanelError; 3] = [
            || PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                column: "entity".to_string(),
                group_id: "a".to_string(),
            },
            || PanelError::UnbalancedPanelForTwoWay {
                n_obs: 5,
                entity_column: "entity".to_string(),
                n_entities: 2,
                time_column: "time".to_string(),
                n_periods: 3,
            },
            // どの計算で起きても同じ入力の誤り（`stage`を付けない`code`）。
            || PanelError::WithinRegressionFailed {
                source: LeastSquaresError::Common(CommonError::InvalidConfidenceLevel {
                    confidence_level: 1.5,
                }),
            },
        ];
        for make in untouched {
            assert_eq!(within_for_variance_component(make()), make());
        }
    }
}
