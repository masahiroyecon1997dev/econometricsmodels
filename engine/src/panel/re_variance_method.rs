//! REの分散成分の推定法（`VarianceMethod`、`re-spec.md`3.1.2節）のうち、`plm::ercomp`の
//! `walhus`（Wallace-Hussain）と`amemiya`の実装。Swamy-Arora（既定）は`re.rs`の
//! `swamy_arora_variance_components`で、`variance_components`が3法を振り分ける。
//!
//! 3法は同じ枠組みで、1段目の回帰の残差`e`から2つ（2-wayは3つ）の二乗和を作り、
//! 期待値の行列`M`と連立させて分散成分を解く。違いは1段目の回帰だけである。
//!
//! - **Wallace-Hussain**: 切片と全説明変数へのプールドOLS。
//! - **Amemiya**: within推定（内部FEの係数）。切片は`ȳ - x̄'β̂_within`。
//!
//! 二乗和は、`Q_w = Σ (within変換した e)²`、`Q_id = Σ_i T_i ē_i²`、2-wayは
//! `Q_ts = Σ_t N_t ē_t²`。
//!
//! - **バランス**（1-wayは全個体の観測数が等しい、2-wayは常に）: `plm`の`dfcor = 1`の簡約形。
//!   `σ_ε² = Q_w/(O - N - T + 1)`（1-wayは`O - N`）、`σ_u² = Q_id/(T·N) - σ_ε²/T`、
//!   `σ_λ² = Q_ts/(N·T) - σ_ε²/N`。負の`σ_u²`・`σ_λ²`は0に切り上げる。
//! - **不均衡**（1-wayのみ）: `plm`の`dfcor = 3`のトレース項の式。
//!   `M·(σ_ε², σ_u²)ᵀ = (Q_w, Q_id)ᵀ`を2×2の連立として解く。
//!
//! 簡約形とトレース形はバランスパネルでも一致しない（`plm`が両者を切り替えている）。1-wayは
//! 時点列を持たないため、`plm`の`is.pbalanced`（同じ時点集合も要求する）ではなく
//! 「全個体の観測数が等しい」でバランスを判定する。観測数が等しく時点がずれたパネルでは
//! `plm`と値が異なる（`plm`はトレース形を使う）。

use faer::Mat;

use crate::linear::common::LeastSquaresError;
use crate::panel::common::PanelError;
use crate::panel::re::{
    EntityMeans, ReEffects, ReInput, VarianceComponents, VarianceMethod, entity_means, internal_fe,
    swamy_arora_variance_components,
};
use crate::shared::group_codes::GroupCodes;
use crate::shared::least_squares::{LeastSquaresFit, least_squares};

/// `method`に応じて分散成分を推定する。`ReEstimator::fit_with_variance_method`の入り口。
///
/// # Errors
/// - 内部FE推定の失敗（singleton・自由度不足・不均衡な2-way等）は`PanelError`のまま伝播する。
/// - `SwamyArora`のbetween回帰の失敗は`PanelError::BetweenRegressionFailed`。
/// - `WallaceHussain`の1段目（プールドOLS）が特異なら`PanelError::PooledRegressionFailed`。
/// - `Amemiya`でwithin変動の無い列があれば`PanelError::AmemiyaRequiresWithinVariation`。
/// - `WallaceHussain`・`Amemiya`で`σ_ε²`が0以下なら
///   `PanelError::NonPositiveIdiosyncraticVariance`、トレース形の連立が特異なら
///   `PanelError::VarianceComponentsMomentsSingular`。
pub(crate) fn variance_components(
    input: &ReInput,
    effects: ReEffects,
    method: VarianceMethod,
    confidence_level: f64,
) -> Result<VarianceComponents, PanelError> {
    match method {
        VarianceMethod::SwamyArora => {
            swamy_arora_variance_components(input, effects, confidence_level)
        }
        VarianceMethod::WallaceHussain | VarianceMethod::Amemiya => {
            first_stage_variance_components(input, effects, method, confidence_level)
        }
    }
}

fn first_stage_variance_components(
    input: &ReInput,
    effects: ReEffects,
    method: VarianceMethod,
    confidence_level: f64,
) -> Result<VarianceComponents, PanelError> {
    crate::shared::parallelism::ensure_serial();

    let (fe, time_varying) = internal_fe(input, effects, confidence_level)?;
    let k = input.x().len();
    if method == VarianceMethod::Amemiya && time_varying.len() != k {
        let columns = (0..k)
            .filter(|j| !time_varying.contains(j))
            .map(|j| input.x_names()[j].clone())
            .collect::<Vec<_>>();
        return Err(PanelError::AmemiyaRequiresWithinVariation { columns });
    }

    let n = input.nobs();
    let entity = input.entity_codes();
    let pooled = match method {
        VarianceMethod::WallaceHussain => Some(pooled_fit(input)?),
        _ => None,
    };
    let residuals: Vec<f64> = match &pooled {
        Some(fit) => (0..n).map(|i| *fit.residuals.get(i, 0)).collect(),
        None => within_fit_residuals(input, &fe),
    };

    let entity_mean_e = group_means(&residuals, entity);
    let q_id: f64 = entity
        .counts()
        .iter()
        .zip(&entity_mean_e)
        .map(|(&t_i, m)| t_i as f64 * m * m)
        .sum();
    let n_entities = entity.n_groups();
    let observations = n as f64;
    let balanced = entity.counts().iter().all(|&c| c == entity.counts()[0]);

    let (sigma2_eps, sigma2_u, sigma2_lambda) = match effects {
        ReEffects::OneWay => {
            let q_w: f64 = (0..n)
                .map(|i| {
                    let r = residuals[i] - entity_mean_e[entity.codes()[i]];
                    r * r
                })
                .sum();
            if balanced {
                let t = entity.counts()[0] as f64;
                let sigma2_eps = q_w / (observations - n_entities as f64);
                check_positive(sigma2_eps, method)?;
                let sigma2_u = (q_id / (t * n_entities as f64) - sigma2_eps / t).max(0.0);
                (sigma2_eps, sigma2_u, 0.0)
            } else {
                let moments = match &pooled {
                    Some(fit) => pooled_trace_moments(input, entity, fit),
                    None => within_trace_moments(input, entity, &fe)?,
                };
                let (sigma2_eps, sigma2_u) = solve_two_by_two(moments, (q_w, q_id), method)?;
                check_positive(sigma2_eps, method)?;
                (sigma2_eps, sigma2_u.max(0.0), 0.0)
            }
        }
        ReEffects::TwoWay => {
            // `internal_fe`が`time`の存在とバランスパネルを検証済み。
            let time = input.time_codes().ok_or(PanelError::TwoWayRequiresTime)?;
            let time_mean_e = group_means(&residuals, time);
            let n_periods = time.n_groups();
            let grand_mean = residuals.iter().sum::<f64>() / observations;
            let q_w: f64 = (0..n)
                .map(|i| {
                    let r = residuals[i]
                        - entity_mean_e[entity.codes()[i]]
                        - time_mean_e[time.codes()[i]]
                        + grand_mean;
                    r * r
                })
                .sum();
            let q_ts: f64 = time
                .counts()
                .iter()
                .zip(&time_mean_e)
                .map(|(&n_t, m)| n_t as f64 * m * m)
                .sum();
            let (big_n, big_t) = (n_entities as f64, n_periods as f64);
            let sigma2_eps = q_w / (observations - big_n - big_t + 1.0);
            check_positive(sigma2_eps, method)?;
            let sigma2_u = (q_id / (big_t * big_n) - sigma2_eps / big_t).max(0.0);
            let sigma2_lambda = (q_ts / (big_n * big_t) - sigma2_eps / big_n).max(0.0);
            (sigma2_eps, sigma2_u, sigma2_lambda)
        }
    };

    Ok(VarianceComponents {
        sigma2_eps,
        sigma2_u,
        sigma2_lambda,
        fe,
        means: entity_means(input.y(), input.x(), entity),
        time_varying,
    })
}

/// σ_ε²が正でなければθが定まらない。
fn check_positive(sigma2_eps: f64, method: VarianceMethod) -> Result<(), PanelError> {
    if sigma2_eps > 0.0 && sigma2_eps.is_finite() {
        Ok(())
    } else {
        Err(PanelError::NonPositiveIdiosyncraticVariance {
            method: method.as_str(),
            sigma2_eps,
        })
    }
}

/// グループ（コード）ごとの平均。
fn group_means(values: &[f64], groups: &GroupCodes) -> Vec<f64> {
    let mut sums = vec![0.0; groups.n_groups()];
    for (&v, &g) in values.iter().zip(groups.codes()) {
        sums[g] += v;
    }
    sums.iter()
        .zip(groups.counts())
        .map(|(s, &c)| s / c as f64)
        .collect()
}

/// 切片と全説明変数へのプールドOLS（Wallace-Hussainの1段目）。
fn pooled_fit(input: &ReInput) -> Result<LeastSquaresFit, PanelError> {
    let n = input.nobs();
    let k = input.x().len();
    let design = Mat::from_fn(
        n,
        k + 1,
        |i, j| {
            if j == 0 { 1.0 } else { input.x()[j - 1][i] }
        },
    );
    let y = Mat::from_fn(n, 1, |i, _| input.y()[i]);
    least_squares(&design, &y).map_err(|err| {
        let names: Vec<String> = std::iter::once("const".to_string())
            .chain(input.x_names().iter().cloned())
            .collect();
        PanelError::PooledRegressionFailed {
            source: LeastSquaresError::from_rank_deficient(err, &names),
        }
    })
}

/// Amemiyaの1段目の残差: `y - x'β̂_within - (ȳ - x̄'β̂_within)`。内部FEの説明変数は全列
/// （within変動の無い列があれば呼び出し前に拒否済み）で、`fe.params()`が入力の列順に並ぶ。
fn within_fit_residuals(input: &ReInput, fe: &crate::panel::fe::FeEstimator) -> Vec<f64> {
    let n = input.nobs() as f64;
    let beta = fe.params();
    let k = input.x().len();
    let mean_y = input.y().iter().sum::<f64>() / n;
    let mean_fit: f64 = (0..k)
        .map(|j| *beta.get(j, 0) * input.x()[j].iter().sum::<f64>() / n)
        .sum();
    (0..input.nobs())
        .map(|i| {
            let fitted: f64 = (0..k).map(|j| *beta.get(j, 0) * input.x()[j][i]).sum();
            input.y()[i] - fitted - (mean_y - mean_fit)
        })
        .collect()
}

/// `a`・`b`が`p×p`の対称行列でなくてもよい一般の積の対角和`tr(A·B)`。
fn trace_of_product(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
    let p = a.nrows();
    let mut sum = 0.0;
    for i in 0..p {
        for j in 0..p {
            sum += *a.get(i, j) * *b.get(j, i);
        }
    }
    sum
}

/// `Σ_i weight_i · v_i v_i'`（`p×p`）。
fn weighted_outer_sum(vectors: &[Vec<f64>], weights: &[f64], p: usize) -> Mat<f64> {
    let mut acc = vec![0.0; p * p];
    for (v, &w) in vectors.iter().zip(weights) {
        for r in 0..p {
            for c in 0..p {
                acc[r * p + c] += w * v[r] * v[c];
            }
        }
    }
    Mat::from_fn(p, p, |r, c| acc[r * p + c])
}

/// エンティティ平均のベクトル（Wallace-Hussainは`[1, x̄_i]`、`with_intercept = false`なら`x̄_i`）。
fn mean_vectors(means: &EntityMeans, with_intercept: bool) -> Vec<Vec<f64>> {
    (0..means.y.len())
        .map(|i| {
            let mut v = Vec::with_capacity(means.x.len() + 1);
            if with_intercept {
                v.push(1.0);
            }
            v.extend(means.x.iter().map(|col| col[i]));
            v
        })
        .collect()
}

/// 期待値の行列の4要素`(M_w,ν, M_w,η, M_id,ν, M_id,η)`。
type Moments = (f64, f64, f64, f64);

/// Wallace-Hussainの不均衡パネルのトレース形（`plm`の`dfcor = 3`、`models = pooling`）。
///
/// `Z = [1, X]`、`(Z'Z)⁻¹`、`B = Σ_i T_i z̄_i z̄_i'`（`Z`の`Between`の外積）、
/// `S = Σ_i T_i² z̄_i z̄_i'`（`Z`の個体和と`Z`の外積）、`W = Z'Z - B`（`Z`のwithin変換の外積）として:
/// - `M_w,ν = O - N - tr((Z'Z)⁻¹ W)`
/// - `M_w,η = tr((Z'Z)⁻¹ W (Z'Z)⁻¹ S)`
/// - `M_id,ν = N - tr((Z'Z)⁻¹ B)`
/// - `M_id,η = O - 2 tr((Z'Z)⁻¹ S) + tr((Z'Z)⁻¹ B (Z'Z)⁻¹ S)`
fn pooled_trace_moments(input: &ReInput, entity: &GroupCodes, fit: &LeastSquaresFit) -> Moments {
    let n = input.nobs();
    let k = input.x().len();
    let p = k + 1;
    let means = entity_means(input.y(), input.x(), entity);
    let z_bar = mean_vectors(&means, true);
    let t_squared: Vec<f64> = means.t.iter().map(|v| v * v).collect();
    let b = weighted_outer_sum(&z_bar, &means.t, p);
    let s = weighted_outer_sum(&z_bar, &t_squared, p);

    // Z'Z（観測ごとの外積の和）。
    let mut ztz = vec![0.0; p * p];
    for i in 0..n {
        let z: Vec<f64> = (0..p)
            .map(|j| if j == 0 { 1.0 } else { input.x()[j - 1][i] })
            .collect();
        for r in 0..p {
            for c in 0..p {
                ztz[r * p + c] += z[r] * z[c];
            }
        }
    }
    let w = Mat::from_fn(p, p, |r, c| ztz[r * p + c] - *b.get(r, c));

    let inv = &fit.xtx_inv;
    let inv_w = inv * &w;
    let inv_b = inv * &b;
    let inv_s = inv * &s;
    let (big_o, big_n) = (n as f64, entity.n_groups() as f64);
    let m_w_nu = big_o - big_n - trace_of_product(inv, &w);
    let m_w_eta = trace_of_product(&inv_w, &inv_s);
    let m_id_nu = big_n - trace_of_product(inv, &b);
    let m_id_eta = big_o - 2.0 * trace_of_product(inv, &s) + trace_of_product(&inv_b, &inv_s);
    (m_w_nu, m_w_eta, m_id_nu, m_id_eta)
}

/// Amemiyaの不均衡パネルのトレース形（`plm`の`dfcor = 3`、`models = within`）。
///
/// `(X̃'X̃)⁻¹`（内部FEのwithin変換後の設計行列）と、`C = Σ_i T_i (x̄_i - x̄)(x̄_i - x̄)'`
/// （`x̄`は全観測の平均）として:
/// - `M_w,ν = O - N - k`、`M_w,η = 0`
/// - `M_id,ν = N - 1 + tr((X̃'X̃)⁻¹ C)`
/// - `M_id,η = O - Σ_i T_i² / O`
fn within_trace_moments(
    input: &ReInput,
    entity: &GroupCodes,
    fe: &crate::panel::fe::FeEstimator,
) -> Result<Moments, PanelError> {
    let n = input.nobs() as f64;
    let k = input.x().len();
    let n_entities = entity.n_groups() as f64;
    let counts: Vec<f64> = entity.counts().iter().map(|&c| c as f64).collect();
    let trace = if k == 0 {
        0.0
    } else {
        let means = entity_means(input.y(), input.x(), entity);
        let overall: Vec<f64> = (0..k)
            .map(|j| input.x()[j].iter().sum::<f64>() / n)
            .collect();
        let centred: Vec<Vec<f64>> = mean_vectors(&means, false)
            .into_iter()
            .map(|v| v.iter().zip(&overall).map(|(a, b)| a - b).collect())
            .collect();
        let c = weighted_outer_sum(&centred, &counts, k);
        let inv = crate::panel::common::xtx_inverse(fe.within_input().x())?;
        trace_of_product(&inv, &c)
    };
    let sum_t_squared: f64 = counts.iter().map(|t| t * t).sum();
    Ok((
        n - n_entities - k as f64,
        0.0,
        n_entities - 1.0 + trace,
        n - sum_t_squared / n,
    ))
}

/// `[[a, b], [c, d]]·(σ_ε², σ_u²)ᵀ = (q_w, q_id)ᵀ`を解く。
fn solve_two_by_two(
    (a, b, c, d): Moments,
    (q_w, q_id): (f64, f64),
    method: VarianceMethod,
) -> Result<(f64, f64), PanelError> {
    let det = a * d - b * c;
    if !det.is_finite() || det.abs() <= f64::EPSILON * (a * d).abs().max((b * c).abs()) {
        return Err(PanelError::VarianceComponentsMomentsSingular {
            method: method.as_str(),
        });
    }
    Ok(((q_w * d - b * q_id) / det, (a * q_id - c * q_w) / det))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panel::re::{ReCovType, ReEstimator};

    const Y: [f64; 24] = [
        2.973, 0.102, -4.005, -1.69, 4.512, 2.697, 1.044, 0.361, 0.457, -1.063, -3.188, -3.244,
        2.213, 0.875, -0.479, -2.389, 5.936, 5.314, 2.163, -0.585, 2.081, 0.631, -1.679, -4.541,
    ];
    const X1: [f64; 24] = [
        -0.828, -0.348, -1.538, -0.256, -1.15, 0.012, -0.223, 0.888, -0.592, -0.656, -0.683,
        -0.016, -0.443, 0.353, 0.073, 0.007, -0.188, -0.766, -0.221, -0.984, -1.104, -0.938, 0.679,
        -1.577,
    ];
    const XI: [f64; 6] = [-0.87, 0.485, -0.186, 1.546, -0.611, -0.348];
    const XT: [f64; 4] = [-1.637, 0.02, 0.892, -0.873];

    /// 6エンティティ×4時点の手作りパネル（`plm`の参照値はこの値から作った）。`rows`は使う行
    /// （0始まり）、`time`なら2-wayの時点も付ける。列は`names`（`x1`・`xi`・`xt`）。
    fn input(rows: &[usize], names: &[&str], two_way: bool) -> ReInput {
        let pick = |f: &dyn Fn(usize) -> f64| -> Vec<f64> { rows.iter().map(|&r| f(r)).collect() };
        let columns: Vec<Vec<f64>> = names
            .iter()
            .map(|n| match *n {
                "x1" => pick(&|r| X1[r]),
                "xi" => pick(&|r| XI[r / 4]),
                "xt" => pick(&|r| XT[r % 4]),
                other => panic!("unknown column {other}"),
            })
            .collect();
        let entity: Vec<String> = rows.iter().map(|r| format!("e{}", r / 4)).collect();
        let time: Vec<String> = rows.iter().map(|r| format!("t{}", r % 4)).collect();
        ReInput::from_columns(
            &pick(&|r| Y[r]),
            &columns,
            names.iter().map(|n| n.to_string()).collect(),
            &entity,
            two_way.then_some(time.as_slice()),
            "y".into(),
        )
        .unwrap()
    }

    fn all_rows() -> Vec<usize> {
        (0..24).collect()
    }

    fn assert_rel(actual: f64, expected: f64, label: &str) {
        assert!(
            (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0),
            "{label}: actual={actual}, expected={expected}"
        );
    }

    fn components(
        input: &ReInput,
        effects: ReEffects,
        method: VarianceMethod,
    ) -> Result<VarianceComponents, PanelError> {
        variance_components(input, effects, method, 0.95)
    }

    #[test]
    fn balanced_one_way_matches_plm_closed_forms() {
        // plm: ercomp(y ~ x1, d, effect = "individual", method = "walhus" | "amemiya")
        // （バランス、`dfcor = 1`の閉形式）。
        let inp = input(&all_rows(), &["x1"], false);
        for (method, eps, u) in [
            (
                VarianceMethod::WallaceHussain,
                6.180_679_388_810_379,
                1.446_976_393_641_097,
            ),
            (
                VarianceMethod::Amemiya,
                6.103_755_807_867_632,
                1.597_323_540_521_647,
            ),
        ] {
            let vc = components(&inp, ReEffects::OneWay, method).unwrap();
            assert_rel(vc.sigma2_eps, eps, "sigma2_eps");
            assert_rel(vc.sigma2_u, u, "sigma2_u");
            assert_eq!(vc.sigma2_lambda, 0.0);
        }
    }

    #[test]
    fn balanced_one_way_with_a_time_only_regressor_matches_plm() {
        // plm: y ~ x1 + xt（`xt`は時点間でのみ変わる）。WHはプールドOLSの残差、Amemiyaはwithin
        // の係数（`xt`もwithin変動を持つ）を使う。
        let inp = input(&all_rows(), &["x1", "xt"], false);
        for (method, eps, u) in [
            (
                VarianceMethod::WallaceHussain,
                4.992_539_566_332_841,
                1.667_316_188_181_227,
            ),
            (
                VarianceMethod::Amemiya,
                4.940_882_248_916_686,
                1.768_844_088_951_385,
            ),
        ] {
            let vc = components(&inp, ReEffects::OneWay, method).unwrap();
            assert_rel(vc.sigma2_eps, eps, "sigma2_eps");
            assert_rel(vc.sigma2_u, u, "sigma2_u");
        }
    }

    #[test]
    fn unbalanced_one_way_matches_plm_trace_forms() {
        // 4行（0始まり5・10・11・19）を落とした不均衡パネル（観測数 4,3,2,4,3,4）。
        // plm: 同じデータの`ercomp`（不均衡は`dfcor = 3`のトレース項の式）。
        let rows: Vec<usize> = (0..24).filter(|r| ![5, 10, 11, 19].contains(r)).collect();
        let inp = input(&rows, &["x1"], false);
        for (method, eps, u) in [
            (
                VarianceMethod::WallaceHussain,
                6.190_506_235_160_872,
                2.457_088_808_540_195,
            ),
            (
                VarianceMethod::Amemiya,
                6.229_221_018_750_94,
                2.375_376_415_113_585,
            ),
        ] {
            let vc = components(&inp, ReEffects::OneWay, method).unwrap();
            assert_rel(vc.sigma2_eps, eps, "sigma2_eps");
            assert_rel(vc.sigma2_u, u, "sigma2_u");
        }
    }

    #[test]
    fn balanced_two_way_matches_plm_closed_forms() {
        // plm: ercomp(y ~ x1, d, effect = "twoways", method = ...)。
        let inp = input(&all_rows(), &["x1"], true);
        for (method, eps, u, lambda) in [
            (
                VarianceMethod::WallaceHussain,
                0.816_015_873_070_809_3,
                2.788_142_272_575_989,
                3.989_496_975_426_726_4,
            ),
            (
                VarianceMethod::Amemiya,
                0.663_192_700_992_561,
                2.699_290_005_810_702,
                4.371_238_538_448_286,
            ),
        ] {
            let vc = components(&inp, ReEffects::TwoWay, method).unwrap();
            assert_rel(vc.sigma2_eps, eps, "sigma2_eps");
            assert_rel(vc.sigma2_u, u, "sigma2_u");
            assert_rel(vc.sigma2_lambda, lambda, "sigma2_lambda");
        }
    }

    #[test]
    fn swamy_arora_dispatch_is_the_existing_estimator() {
        let inp = input(&all_rows(), &["x1"], false);
        let direct = swamy_arora_variance_components(&inp, ReEffects::OneWay, 0.95).unwrap();
        let via = components(&inp, ReEffects::OneWay, VarianceMethod::SwamyArora).unwrap();
        assert_eq!(direct.sigma2_eps, via.sigma2_eps);
        assert_eq!(direct.sigma2_u, via.sigma2_u);
    }

    #[test]
    fn amemiya_rejects_regressors_without_within_variation() {
        // 1-wayの`xi`（個体間でのみ変わる）と、2-wayの`xi`・`xt`（一次元でしか変わらない）。
        let one_way = input(&all_rows(), &["x1", "xi"], false);
        let err = components(&one_way, ReEffects::OneWay, VarianceMethod::Amemiya).unwrap_err();
        assert_eq!(
            err,
            PanelError::AmemiyaRequiresWithinVariation {
                columns: vec!["xi".to_string()]
            }
        );
        let two_way = input(&all_rows(), &["x1", "xi", "xt"], true);
        let err = components(&two_way, ReEffects::TwoWay, VarianceMethod::Amemiya).unwrap_err();
        assert_eq!(
            err,
            PanelError::AmemiyaRequiresWithinVariation {
                columns: vec!["xi".to_string(), "xt".to_string()]
            }
        );
        // 不均衡でも同じ。Wallace-Hussainは受け付ける。
        let rows: Vec<usize> = (0..24).filter(|r| *r != 5).collect();
        let unbalanced = input(&rows, &["x1", "xi"], false);
        assert!(matches!(
            components(&unbalanced, ReEffects::OneWay, VarianceMethod::Amemiya),
            Err(PanelError::AmemiyaRequiresWithinVariation { .. })
        ));
        assert!(
            components(
                &unbalanced,
                ReEffects::OneWay,
                VarianceMethod::WallaceHussain
            )
            .is_ok()
        );
    }

    #[test]
    fn wallace_hussain_accepts_every_regressor_without_within_variation() {
        // 全列が個体間でのみ変わる。SwamyArora・Amemiyaは使えないがWHは推定できる。
        let inp = input(&all_rows(), &["xi"], false);
        let vc = components(&inp, ReEffects::OneWay, VarianceMethod::WallaceHussain).unwrap();
        assert!(vc.sigma2_eps > 0.0 && vc.time_varying.is_empty());
    }

    #[test]
    fn pooled_regression_failure_is_reported_for_wallace_hussain() {
        // `xi`とその2倍は内部FEには渡らない（within変動なし）が、プールドOLSは特異になる。
        let base = input(&all_rows(), &["x1", "xi"], false);
        let doubled: Vec<f64> = base.x()[1].iter().map(|v| 2.0 * v).collect();
        let columns = vec![base.x()[0].clone(), base.x()[1].clone(), doubled];
        let entity: Vec<String> = (0..24).map(|r| format!("e{}", r / 4)).collect();
        let inp = ReInput::from_columns(
            base.y(),
            &columns,
            vec!["x1".into(), "xi".into(), "xi2".into()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();
        assert!(matches!(
            components(&inp, ReEffects::OneWay, VarianceMethod::WallaceHussain),
            Err(PanelError::PooledRegressionFailed { .. })
        ));
    }

    #[test]
    fn non_positive_idiosyncratic_variance_is_rejected() {
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                check_positive(value, VarianceMethod::Amemiya).is_err(),
                "{value}"
            );
        }
        assert!(check_positive(1e-300, VarianceMethod::Amemiya).is_ok());
        let err = check_positive(0.0, VarianceMethod::WallaceHussain).unwrap_err();
        assert!(err.to_string().contains("wallace_hussain"));
    }

    #[test]
    fn negative_sigma2_eps_from_the_trace_form_is_rejected_through_fit() {
        // 観測数2・6・2の不均衡パネル。y は個体間でほぼ決まり、プールドOLSの残差から作る
        // トレース形の連立が負の`σ_ε²`（約-1.09）を返す。`plm`は0に切り上げてθがNaNになるが、
        // 本実装は拒否する。閉形式（バランス）とAmemiya（`Q_w ≥ 0`を正の自由度で割る）は
        // 負にならず、0になるのは完全適合のときだけ。
        let y = [0.0, 1.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 6.0, 7.0];
        let x1 = [2.0, 1.0, 0.0, -2.0, -1.0, -3.0, -3.0, -3.0, -2.0, 2.0];
        let entity: Vec<String> = [0, 0, 1, 1, 1, 1, 1, 1, 2, 2]
            .iter()
            .map(|i| format!("e{i}"))
            .collect();
        let build = || {
            ReInput::from_columns(
                &y,
                &[x1.to_vec()],
                vec!["x1".into()],
                &entity,
                None,
                "y".into(),
            )
            .unwrap()
        };
        let err = ReEstimator::fit_with_variance_method(
            build(),
            ReEffects::OneWay,
            ReCovType::Classical,
            VarianceMethod::WallaceHussain,
            0.95,
        )
        .unwrap_err();
        match err {
            PanelError::NonPositiveIdiosyncraticVariance { method, sigma2_eps } => {
                assert_eq!(method, "wallace_hussain");
                assert!(sigma2_eps < 0.0, "{sigma2_eps}");
            }
            other => panic!("unexpected error: {other:?}"),
        }
        // 同じデータでもAmemiyaは推定できる。
        assert!(
            ReEstimator::fit_with_variance_method(
                build(),
                ReEffects::OneWay,
                ReCovType::Classical,
                VarianceMethod::Amemiya,
                0.95,
            )
            .is_ok()
        );
    }

    #[test]
    fn singular_moment_system_is_a_computation_failure() {
        // 実入力でこの連立が特異になる例は見つかっていない（Amemiyaは`M_w,η = 0`で行列式が
        // `(O - N - K)(O - ΣT_i²/O) > 0`、Wallace-Hussainも一般の設計では非特異）。到達不能と
        // 断定せず、数値的に特異な連立を黙って解かないための防御として残す。
        // 2行が比例する連立（トレース形の`M`が特異）。
        let result = solve_two_by_two((1.0, 2.0, 2.0, 4.0), (1.0, 1.0), VarianceMethod::Amemiya);
        assert!(matches!(
            result,
            Err(PanelError::VarianceComponentsMomentsSingular { .. })
        ));
        let (eps, u) =
            solve_two_by_two((2.0, 0.0, 0.0, 3.0), (4.0, 9.0), VarianceMethod::Amemiya).unwrap();
        assert_eq!((eps, u), (2.0, 3.0));
    }

    #[test]
    fn fit_records_the_variance_method() {
        let inp = || input(&all_rows(), &["x1"], false);
        let default =
            ReEstimator::fit(inp(), ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
        assert_eq!(default.variance_method(), VarianceMethod::SwamyArora);
        let walhus = ReEstimator::fit_with_variance_method(
            inp(),
            ReEffects::OneWay,
            ReCovType::Classical,
            VarianceMethod::WallaceHussain,
            0.95,
        )
        .unwrap();
        assert_eq!(walhus.variance_method(), VarianceMethod::WallaceHussain);
        assert_rel(walhus.sigma2_eps(), 6.180_679_388_810_379, "sigma2_eps");
        assert_eq!(VarianceMethod::default(), VarianceMethod::SwamyArora);
        assert_eq!(VarianceMethod::Amemiya.as_str(), "amemiya");
    }

    /// property-basedテスト。固定シナリオでは扱えない不均衡パネル・サイズの組み合わせで、分散成分の
    /// 不変条件（`y`のスケール・シフト、説明変数のスケール、行順序・ラベルへの不変性）と、
    /// 報告したθで手で準偏差変換したOLSとの一致を検証する。トレース項の式の書き間違い
    /// （転置・取り違え）は、これらの不変条件を壊す。
    mod proptests {
        use super::*;
        use crate::linear::ols::{CovType, OlsEstimator, OlsInput};
        use proptest::collection;
        use proptest::prelude::*;

        #[derive(Debug, Clone)]
        struct Case {
            entity_idx: Vec<usize>,
            n_entities: usize,
            y: Vec<f64>,
            x: Vec<Vec<f64>>,
            keys: Vec<u64>,
        }

        /// 1-wayのパネル。`balanced`なら全個体の観測数が等しい（閉形式）、そうでなければ
        /// 2〜6にばらつく（トレース項の式）。`has_effect`が偽なら`σ_u²`が0に切り上げられる
        /// ケースが増える。
        fn one_way_case() -> impl Strategy<Value = Case> {
            (
                6..=9usize,
                1..=2usize,
                any::<bool>(),
                any::<bool>(),
                3..=5usize,
            )
                .prop_flat_map(|(n_entities, k, has_effect, balanced, t)| {
                    let sizes: BoxedStrategy<Vec<usize>> = if balanced {
                        Just(vec![t; n_entities]).boxed()
                    } else {
                        collection::vec(2..=6usize, n_entities).boxed()
                    };
                    (Just((n_entities, k, has_effect)), sizes)
                })
                .prop_flat_map(|((n_entities, k, has_effect), sizes)| {
                    let n: usize = sizes.iter().sum();
                    (
                        Just((n_entities, has_effect)),
                        Just(sizes),
                        collection::vec(collection::vec(-10.0f64..10.0, n), k),
                        collection::vec(-5.0f64..5.0, n),
                        collection::vec(-20.0f64..20.0, n_entities),
                        collection::vec(any::<u64>(), n),
                    )
                })
                .prop_map(
                    |((n_entities, has_effect), sizes, x, noise, effect, keys)| {
                        let mut entity_idx = Vec::new();
                        for (i, &size) in sizes.iter().enumerate() {
                            entity_idx.extend(std::iter::repeat_n(i, size));
                        }
                        let y: Vec<f64> = (0..noise.len())
                            .map(|r| {
                                let u = if has_effect {
                                    effect[entity_idx[r]]
                                } else {
                                    0.0
                                };
                                x.iter().map(|c| c[r]).sum::<f64>() + u + noise[r]
                            })
                            .collect();
                        Case {
                            entity_idx,
                            n_entities,
                            y,
                            x,
                            keys,
                        }
                    },
                )
        }

        fn method_strategy() -> impl Strategy<Value = VarianceMethod> {
            prop_oneof![
                Just(VarianceMethod::WallaceHussain),
                Just(VarianceMethod::Amemiya)
            ]
        }

        fn build(case: &Case, y: &[f64], x: &[Vec<f64>], entity: &[String]) -> ReInput {
            let names: Vec<String> = (0..x.len()).map(|j| format!("x{j}")).collect();
            let _ = case;
            ReInput::from_columns(y, x, names, entity, None, "y".into()).unwrap()
        }

        fn labels(case: &Case) -> Vec<String> {
            case.entity_idx.iter().map(|i| format!("e{i}")).collect()
        }

        fn sigma(input: &ReInput, method: VarianceMethod) -> Option<(f64, f64)> {
            variance_components(input, ReEffects::OneWay, method, 0.95)
                .ok()
                .map(|vc| (vc.sigma2_eps, vc.sigma2_u))
        }

        /// `plm::ercomp`（`dfcor = 3`）のソースどおりに、`Z`の`Between`・`Sum`・within変換を
        /// 観測ごとの行列に展開して`M`を組んだ参照実装（`variance_components`が使う
        /// エンティティ単位の外積による実装とは独立）。不均衡の1-way専用。
        fn observation_level_trace_form(case: &Case, method: VarianceMethod) -> (f64, f64) {
            let n = case.y.len();
            let n_ent = case.n_entities;
            let idx = &case.entity_idx;
            let mut counts = vec![0.0; n_ent];
            for &i in idx {
                counts[i] += 1.0;
            }
            let k = case.x.len();
            let p = k + 1;
            let z = Mat::from_fn(n, p, |r, j| if j == 0 { 1.0 } else { case.x[j - 1][r] });
            let entity_mean = |col: &dyn Fn(usize) -> f64| -> Vec<f64> {
                let mut sums = vec![0.0; n_ent];
                for r in 0..n {
                    sums[idx[r]] += col(r);
                }
                (0..n_ent).map(|i| sums[i] / counts[i]).collect()
            };
            let z_means: Vec<Vec<f64>> = (0..p).map(|j| entity_mean(&|r| *z.get(r, j))).collect();
            let z_between = Mat::from_fn(n, p, |r, j| z_means[j][idx[r]]);
            let z_sum = Mat::from_fn(n, p, |r, j| z_means[j][idx[r]] * counts[idx[r]]);
            let z_within = Mat::from_fn(n, p, |r, j| *z.get(r, j) - z_means[j][idx[r]]);
            let gram = |a: &Mat<f64>, b: &Mat<f64>| a.transpose() * b;

            // 1段目の残差。
            let y_means = entity_mean(&|r| case.y[r]);
            let e: Vec<f64> = match method {
                VarianceMethod::WallaceHussain => {
                    let inv = crate::shared::covariance::xtx_inverse(&z).unwrap();
                    let zy = Mat::from_fn(p, 1, |j, _| {
                        (0..n).map(|r| *z.get(r, j) * case.y[r]).sum::<f64>()
                    });
                    let beta = &inv * &zy;
                    (0..n)
                        .map(|r| {
                            case.y[r] - (0..p).map(|j| *z.get(r, j) * *beta.get(j, 0)).sum::<f64>()
                        })
                        .collect()
                }
                _ => {
                    let wx = Mat::from_fn(n, k, |r, j| *z_within.get(r, j + 1));
                    let wy: Vec<f64> = (0..n).map(|r| case.y[r] - y_means[idx[r]]).collect();
                    let inv = crate::shared::covariance::xtx_inverse(&wx).unwrap();
                    let wxy = Mat::from_fn(k, 1, |j, _| {
                        (0..n).map(|r| *wx.get(r, j) * wy[r]).sum::<f64>()
                    });
                    let beta = &inv * &wxy;
                    let mean_y = case.y.iter().sum::<f64>() / n as f64;
                    let mean_fit: f64 = (0..k)
                        .map(|j| *beta.get(j, 0) * case.x[j].iter().sum::<f64>() / n as f64)
                        .sum();
                    (0..n)
                        .map(|r| {
                            let fitted: f64 = (0..k).map(|j| *beta.get(j, 0) * case.x[j][r]).sum();
                            case.y[r] - fitted - (mean_y - mean_fit)
                        })
                        .collect()
                }
            };
            let e_means = entity_mean(&|r| e[r]);
            let q_w: f64 = (0..n).map(|r| (e[r] - e_means[idx[r]]).powi(2)).sum();
            let q_id: f64 = (0..n).map(|r| e_means[idx[r]].powi(2)).sum();

            let tr = |m: &Mat<f64>| -> f64 { (0..m.nrows()).map(|i| *m.get(i, i)).sum() };
            let (big_o, big_n) = (n as f64, n_ent as f64);
            let (a, b, c, d) = match method {
                VarianceMethod::WallaceHussain => {
                    let inv = crate::shared::covariance::xtx_inverse(&z).unwrap();
                    let cp_w = gram(&z_within, &z_within);
                    let cp_b = gram(&z_between, &z_between);
                    let cp_s = gram(&z_sum, &z);
                    (
                        big_o - big_n - tr(&(&inv * &cp_w)),
                        tr(&(&(&(&inv * &cp_w) * &inv) * &cp_s)),
                        big_n - tr(&(&inv * &cp_b)),
                        big_o - 2.0 * tr(&(&inv * &cp_s))
                            + tr(&(&(&(&inv * &cp_b) * &inv) * &cp_s)),
                    )
                }
                _ => {
                    let wx = Mat::from_fn(n, k, |r, j| *z_within.get(r, j + 1));
                    let xb_mean: Vec<f64> = (0..k)
                        .map(|j| (0..n).map(|r| z_means[j + 1][idx[r]]).sum::<f64>() / big_o)
                        .collect();
                    let xb = Mat::from_fn(n, k, |r, j| z_means[j + 1][idx[r]] - xb_mean[j]);
                    let inv = crate::shared::covariance::xtx_inverse(&wx).unwrap();
                    (
                        big_o - big_n - k as f64,
                        0.0,
                        big_n - 1.0 + tr(&(&inv * &gram(&xb, &xb))),
                        big_o - counts.iter().map(|t| t * t).sum::<f64>() / big_o,
                    )
                }
            };
            let det = a * d - b * c;
            ((q_w * d - b * q_id) / det, (a * q_id - c * q_w) / det)
        }

        fn close(actual: f64, expected: f64, tol: f64) -> bool {
            (actual - expected).abs() <= tol * expected.abs().max(1.0)
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            /// 不均衡の1-wayでは、トレース項の式（エンティティ単位の外積で実装）が、`plm`のソースどおりに
            /// 観測ごとの行列へ展開した参照実装と一致する。
            ///
            /// 有効性の確認（`pooled_trace_moments`・`within_trace_moments`へのバグ注入、確認後に
            /// 元へ戻した）: `W`と`S`の取り違え・`M_id,η`の係数のずれ・`M_id,ν`の`-1`の欠落・
            /// `Σ T_i²`の割る数のずれ・`M_w,ν`のトレース項の取り違えのすべてを本テストが検出する
            /// （不変性のテストは、式が一貫して誤っていても通るため検出できない）。
            #[test]
            fn trace_forms_match_the_observation_level_reference(
                case in one_way_case(),
                method in method_strategy(),
            ) {
                let mut counts = vec![0usize; case.n_entities];
                for &i in &case.entity_idx {
                    counts[i] += 1;
                }
                prop_assume!(counts.iter().any(|&c| c != counts[0]));
                let entity = labels(&case);
                let ours = variance_components(
                    &build(&case, &case.y, &case.x, &entity),
                    ReEffects::OneWay,
                    method,
                    0.95,
                );
                prop_assume!(ours.is_ok());
                let ours = ours.unwrap();
                let (eps, u) = observation_level_trace_form(&case, method);
                prop_assert!(close(ours.sigma2_eps, eps, 1e-8), "{} vs {eps}", ours.sigma2_eps);
                prop_assert!(close(ours.sigma2_u, u.max(0.0), 1e-8), "{} vs {u}", ours.sigma2_u);
            }

            /// `y`を`c`倍すると`σ_ε²`・`σ_u²`は`c²`倍、シフトしても変わらない。
            #[test]
            fn variance_components_scale_with_y_and_ignore_a_shift(
                case in one_way_case(),
                method in method_strategy(),
                scale in 0.5f64..4.0,
                shift in -50.0f64..50.0,
            ) {
                let entity = labels(&case);
                let base = sigma(&build(&case, &case.y, &case.x, &entity), method);
                prop_assume!(base.is_some());
                let (eps, u) = base.unwrap();
                let scaled_y: Vec<f64> = case.y.iter().map(|v| v * scale).collect();
                let scaled = sigma(&build(&case, &scaled_y, &case.x, &entity), method);
                prop_assume!(scaled.is_some());
                let (eps_s, u_s) = scaled.unwrap();
                prop_assert!(close(eps_s, eps * scale * scale, 1e-7), "{eps_s} vs {eps}");
                prop_assert!(close(u_s, u * scale * scale, 1e-7), "{u_s} vs {u}");
                let shifted_y: Vec<f64> = case.y.iter().map(|v| v + shift).collect();
                let shifted = sigma(&build(&case, &shifted_y, &case.x, &entity), method);
                prop_assume!(shifted.is_some());
                let (eps_h, u_h) = shifted.unwrap();
                prop_assert!(close(eps_h, eps, 1e-7), "{eps_h} vs {eps}");
                prop_assert!(close(u_h, u, 1e-7), "{u_h} vs {u}");
            }

            /// 説明変数の列をスケールしても（係数が逆数倍になるだけで）分散成分は変わらない。
            #[test]
            fn variance_components_ignore_the_scale_of_a_regressor(
                case in one_way_case(),
                method in method_strategy(),
                scale in 0.2f64..5.0,
            ) {
                let entity = labels(&case);
                let base = sigma(&build(&case, &case.y, &case.x, &entity), method);
                prop_assume!(base.is_some());
                let (eps, u) = base.unwrap();
                let mut x = case.x.clone();
                for v in x[0].iter_mut() {
                    *v *= scale;
                }
                let scaled = sigma(&build(&case, &case.y, &x, &entity), method);
                prop_assume!(scaled.is_some());
                let (eps_s, u_s) = scaled.unwrap();
                prop_assert!(close(eps_s, eps, 1e-7), "{eps_s} vs {eps}");
                prop_assert!(close(u_s, u, 1e-7), "{u_s} vs {u}");
            }

            /// 行の並べ替えとエンティティラベルの付け替えで分散成分は変わらない。
            #[test]
            fn variance_components_are_invariant_to_row_order_and_relabeling(
                case in one_way_case(),
                method in method_strategy(),
            ) {
                let entity = labels(&case);
                let base = sigma(&build(&case, &case.y, &case.x, &entity), method);
                prop_assume!(base.is_some());
                let (eps, u) = base.unwrap();
                let n = case.y.len();
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by_key(|&r| case.keys[r]);
                let y: Vec<f64> = order.iter().map(|&r| case.y[r]).collect();
                let x: Vec<Vec<f64>> = case
                    .x
                    .iter()
                    .map(|c| order.iter().map(|&r| c[r]).collect())
                    .collect();
                let entity2: Vec<String> = order
                    .iter()
                    .map(|&r| format!("z{}", case.n_entities - 1 - case.entity_idx[r]))
                    .collect();
                let permuted = sigma(&build(&case, &y, &x, &entity2), method);
                prop_assume!(permuted.is_some());
                let (eps_p, u_p) = permuted.unwrap();
                prop_assert!(close(eps_p, eps, 1e-7), "{eps_p} vs {eps}");
                prop_assert!(close(u_p, u, 1e-7), "{u_p} vs {u}");
            }

            /// どの推定法でも、報告したθ_iで手で準偏差変換した`y`・`X`（切片列は`1 - θ_i`倍）への
            /// OLSが、REの係数・Classical標準誤差と一致する（不均衡パネルを含む）。
            #[test]
            fn fit_equals_ols_on_the_data_quasi_demeaned_with_the_reported_theta(
                case in one_way_case(),
                method in prop_oneof![
                    Just(VarianceMethod::SwamyArora),
                    Just(VarianceMethod::WallaceHussain),
                    Just(VarianceMethod::Amemiya)
                ],
            ) {
                let entity = labels(&case);
                let input = build(&case, &case.y, &case.x, &entity);
                let est = ReEstimator::fit_with_variance_method(
                    input,
                    ReEffects::OneWay,
                    ReCovType::Classical,
                    method,
                    0.95,
                );
                prop_assume!(est.is_ok());
                let est = est.unwrap();
                let theta = est.theta_by_entity().unwrap();
                let n = case.y.len();
                let quasi = |v: &[f64]| -> Vec<f64> {
                    let mut sums = vec![0.0; case.n_entities];
                    let mut counts = vec![0.0; case.n_entities];
                    for (r, &value) in v.iter().enumerate() {
                        sums[case.entity_idx[r]] += value;
                        counts[case.entity_idx[r]] += 1.0;
                    }
                    (0..n)
                        .map(|r| {
                            let i = case.entity_idx[r];
                            v[r] - theta[i] * sums[i] / counts[i]
                        })
                        .collect()
                };
                let y_star = quasi(&case.y);
                let mut columns =
                    vec![(0..n).map(|r| 1.0 - theta[case.entity_idx[r]]).collect::<Vec<f64>>()];
                columns.extend(case.x.iter().map(|c| quasi(c)));
                let names: Vec<String> = (0..columns.len()).map(|j| format!("c{j}")).collect();
                let ols_input =
                    OlsInput::from_columns(&y_star, &columns, names, false, "y".to_string())
                        .unwrap();
                let ols = OlsEstimator::fit(ols_input, CovType::Classical, 0.95);
                prop_assume!(ols.is_ok());
                let ols = ols.unwrap();
                for j in 0..est.estimator().params().nrows() {
                    let (a, b) = (*est.estimator().params().get(j, 0), *ols.params().get(j, 0));
                    prop_assert!(close(a, b, 1e-6), "param[{j}] {a} vs {b}");
                    let (a, b) = (*est.std_errors().get(j, 0), *ols.std_errors().get(j, 0));
                    prop_assert!(close(a, b, 1e-6), "se[{j}] {a} vs {b}");
                }
            }
        }
    }
}
