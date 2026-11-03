//! 限界効果（`at="overall"`・`"mean"`・`"median"`）の計算。logit/probitの`model.rs`の
//! `overall_w_and_s`・`at_point_w_and_s`に相当する部分と、それを使ってデルタ法で限界効果を組み立てる関数。

use super::target::{MarginalEffectsTarget, target_w_and_s};
use crate::nonlinear::common::{MarginalEffects, MleError, validate_confidence_level};
use crate::shared::error::CommonError;
use crate::shared::inference;
use faer::Mat;
use statrs::distribution::Normal;

/// `target_w_and_s`の`at="overall"`（AME）版。全観測を1回走査し、各観測の`(w,s_beta,
/// s_sigma)`を平均する（`w`/`s`は`β`について線形に`dydx_j=w*βⱼ`へ寄与するため、平均を
/// 先に取ってから`dydx_j`を計算しても、観測ごとに`dydx_j`を計算してから平均しても
/// 同じ結果になる。Logitの`overall_w_and_s`と同じ考え方）。
pub(super) fn overall_target_w_and_s(
    x: &Mat<f64>,
    beta: &[f64],
    sigma: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    target: MarginalEffectsTarget,
    normal: &Normal,
) -> (f64, Vec<f64>, f64) {
    let n = x.nrows();
    let k = x.ncols();
    let mut w_sum = 0.0;
    let mut s_beta_sum = vec![0.0; k];
    let mut s_sigma_sum = 0.0;
    // `x_row`はループの外で1回だけ確保し、観測ごとに上書きして使い回す
    // （`target_w_and_s`が`&[f64]`を要求する都合上コピー自体は避けられないが、
    // 観測数`n`回分のヒープ確保は不要）。
    let mut x_row = vec![0.0; k];
    for i in 0..n {
        for (j, v) in x_row.iter_mut().enumerate() {
            *v = *x.get(i, j);
        }
        let (w, s_beta, s_sigma) =
            target_w_and_s(target, &x_row, beta, sigma, lower, upper, normal);
        w_sum += w;
        for (m, s_m) in s_beta_sum.iter_mut().enumerate() {
            *s_m += s_beta[m];
        }
        s_sigma_sum += s_sigma;
    }
    let n_f = n as f64;
    for s_m in s_beta_sum.iter_mut() {
        *s_m /= n_f;
    }
    (w_sum / n_f, s_beta_sum, s_sigma_sum / n_f)
}

/// `target_w_and_s`の`at="mean"`/`"median"`版。代表点`x_bar`（各説明変数の標本平均または
/// 中央値）で1回だけ評価する（Logitの`at_point_w_and_s`と同じ考え方、`target_w_and_s`
/// 自体がこの形なので薄いラッパー）。
pub(super) fn at_point_target_w_and_s(
    x_bar: &[f64],
    beta: &[f64],
    sigma: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    target: MarginalEffectsTarget,
    normal: &Normal,
) -> (f64, Vec<f64>, f64) {
    target_w_and_s(target, x_bar, beta, sigma, lower, upper, normal)
}

/// `marginal_effects_from_tobit_w_s`の呼び出しに必要な、フィット済みモデルの情報を束ねる
/// （`clippy::too_many_arguments`を避けるため。`nonlinear/common/marginal_effects.rs`の
/// `FittedModelForMarginalEffects`と同じ理由・同じパターンだが、Tobitは`cov_params`が
/// `(k+1)×(k+1)`（`σ`を含む）である点が異なるため独立の型として定義する）。
pub(super) struct TobitMarginalEffectsInputs<'a> {
    /// 説明変数名（定数項を含む）
    pub(super) param_names: &'a [String],
    /// 定数項を含むか
    pub(super) has_intercept: bool,
    /// 説明変数の数（定数項を含む、`σ`は含まない）
    pub(super) k: usize,
    /// 係数（元のスケール、`β`部分のみ）
    pub(super) beta: &'a [f64],
    /// `(β,σ)`の分散共分散行列（元のスケール、`(k+1)×(k+1)`）
    pub(super) cov_params: &'a Mat<f64>,
}

/// `target_w_and_s`系が計算する`(w, s_beta, s_sigma)`から、限界効果`dydx_j=w*βⱼ`と
/// そのヤコビアン（`β`の`k`列+`σ`の1列、計`k+1`列）を組み立て、デルタ法による標準誤差・
/// z値・p値・信頼区間を計算し、定数項を除いた`MarginalEffects`を返す。
///
/// `nonlinear/common/marginal_effects.rs`の`marginal_effects_from_w_s`と数式の骨格
/// （`∂dydx_j/∂θₘ=βⱼ*s_m+[j==m]*w`）は同じだが、パラメータ次元が`k`ではなく`k+1`
/// （`σ`を含む）である点が異なるため独立実装とする。
///
/// # Errors
/// `confidence_level`が`(0, 1)`の範囲外: `CommonError::InvalidConfidenceLevel`
pub(super) fn marginal_effects_from_tobit_w_s(
    model: TobitMarginalEffectsInputs,
    w: f64,
    s_beta: &[f64],
    s_sigma: f64,
    confidence_level: f64,
) -> Result<MarginalEffects, MleError> {
    validate_confidence_level(confidence_level)?;
    let TobitMarginalEffectsInputs {
        param_names,
        has_intercept,
        k,
        beta,
        cov_params,
    } = model;

    let dydx: Vec<f64> = (0..k).map(|j| w * beta[j]).collect();
    // ヤコビアンの各行は長さ`k+1`（`β`のk列 + `σ`の1列）
    let jacobian: Vec<Vec<f64>> = (0..k)
        .map(|j| {
            let mut row: Vec<f64> = (0..k)
                .map(|m| beta[j] * s_beta[m] + if j == m { w } else { 0.0 })
                .collect();
            row.push(beta[j] * s_sigma);
            row
        })
        .collect();

    // `Normal::new(0.0, 1.0)`は標準正規分布であり、標準偏差が正であることを要求する
    // statrsの検証を常に満たすため、この`map_err`分岐は理論上到達不能
    // （`.claude/rules/rust-style.md`「テスト」のカバレッジ方針参照、`marginal_effects_from_w_s`
    // と同じ扱い）。
    let normal = Normal::new(0.0, 1.0).map_err(|e| CommonError::internal(e.to_string()))?;
    let z_crit = inference::critical_value(&normal, confidence_level);

    let k_constant = usize::from(has_intercept);
    let k_plus_1 = k + 1;
    let mut out_param_names = Vec::with_capacity(k - k_constant);
    let mut out_dydx = Vec::with_capacity(k - k_constant);
    let mut std_errors = Vec::with_capacity(k - k_constant);
    let mut test_stats = Vec::with_capacity(k - k_constant);
    let mut p_values = Vec::with_capacity(k - k_constant);
    let mut conf_lower = Vec::with_capacity(k - k_constant);
    let mut conf_upper = Vec::with_capacity(k - k_constant);

    for j in k_constant..k {
        let jac_row = &jacobian[j];
        let mut var_j = 0.0;
        for a in 0..k_plus_1 {
            for b in 0..k_plus_1 {
                var_j += jac_row[a] * (*cov_params.get(a, b)) * jac_row[b];
            }
        }
        let se = var_j.sqrt();
        let stat = inference::compute_inference_stat(&normal, dydx[j], se, z_crit);

        out_param_names.push(param_names[j].clone());
        out_dydx.push(dydx[j]);
        std_errors.push(se);
        test_stats.push(stat.stat);
        p_values.push(stat.p_value);
        conf_lower.push(stat.conf_low);
        conf_upper.push(stat.conf_high);
    }

    Ok(MarginalEffects::from_parts(
        out_param_names,
        out_dydx,
        std_errors,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
    ))
}
