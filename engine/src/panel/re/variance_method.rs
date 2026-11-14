//! REの分散成分の推定法（`VarianceMethod`、`re-spec.md`3.1.2節）のうち、`plm::ercomp`の
//! `walhus`（Wallace-Hussain）と`amemiya`の実装。Swamy-Arora（既定）は`re/variance.rs`の
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
//!
//! テストは`variance_method/tests.rs`と`variance_method/tests/`にテーマ別で置く。

use faer::Mat;

use super::input::{ReEffects, ReInput};
use super::variance::{
    EntityMeans, VarianceComponents, entity_means, internal_fe, swamy_arora_variance_components,
};
use crate::linear::common::LeastSquaresError;
use crate::panel::common::PanelError;
use crate::shared::group_codes::GroupCodes;
use crate::shared::least_squares::{LeastSquaresFit, least_squares};

/// RE（GLS）の分散成分（σ_ε²・σ_u²・σ_λ²）の推定法（`re-spec.md`3.1節・3.1.2節）。
/// `plm::ercomp`の`method`（`swar`・`walhus`・`amemiya`）に対応する。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VarianceMethod {
    /// Swamy-Arora。1段目はwithin回帰とbetween回帰。1-wayの不均衡パネルは
    /// `linearmodels`と同じ式（`plm`のトレース項の式とは異なる、`re-spec.md`3.1節）。
    #[default]
    SwamyArora,
    /// Wallace-Hussain。1段目は切片と全説明変数へのプールドOLSの残差。参照実装は`plm`
    /// （`random.method = "walhus"`）のみ。
    WallaceHussain,
    /// Amemiya。1段目はwithin推定の残差（切片は`ȳ - x̄'β̂_within`）。within変動の無い列が
    /// あると推定できない（`PanelError::AmemiyaRequiresWithinVariation`）。参照実装は`plm`
    /// （`random.method = "amemiya"`）のみ。
    Amemiya,
}

impl VarianceMethod {
    /// `variance_method`オプションの正規化した値（小文字のスネークケース）。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SwamyArora => "swamy_arora",
            Self::WallaceHussain => "wallace_hussain",
            Self::Amemiya => "amemiya",
        }
    }
}

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
pub(super) fn variance_components(
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
            source: LeastSquaresError::from_rank_deficient(err, &design, &names),
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
mod tests;
