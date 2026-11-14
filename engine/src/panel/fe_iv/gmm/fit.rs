//! `FeGmmEstimator::fit`の本体。パイプラインの契約はモジュールdoc（`gmm.rs`）が正本。
//!
//! テストは`fit/tests.rs`と`fit/tests/`にテーマ別で置く。

use super::hansen_j::fe_hansen_j;
use super::{FeGmmEstimator, FeGmmType};
use crate::panel::common::{PanelError, PanelMoment, PanelMomentInput, panel_moment_covariance};
use crate::panel::fe::{
    FeAbsorption, FeCovType, FeEffects, fe_r_squared_between, fe_r_squared_overall,
};
use crate::panel::fe_iv::input::FeIvInput;
use crate::panel::fe_iv::prepare::{
    WithinPrepared, mat_from_columns, prepare_within, structural_fit_is_perfect,
};
use crate::shared::error::CommonError;
use crate::shared::gmm::gmm_sandwich_covariance;
use crate::shared::gmm_iteration::{GmmInitialStep, GmmIteration, gmm_initial_step, gmm_iterate};
use crate::shared::gmm_weight::validate_weight_type;
use crate::shared::inference::{InferenceTable, inference_table};
use crate::shared::wald::wald_chi2_test;
use faer::Mat;
use statrs::distribution::Normal;

/// [`FeGmmEstimator::fit_with_absorption`]の本体。契約（エラー条件を含む）は`fit`のdocコメント参照。
pub(super) fn fit_fe_gmm(
    input: FeIvInput,
    effects: FeEffects,
    gmm_type: FeGmmType,
    raise_on_non_convergence: bool,
    cov_type: FeCovType,
    confidence_level: f64,
    absorption: &FeAbsorption,
) -> Result<FeGmmEstimator, PanelError> {
    // faerのグローバル並列度を`Par::Seq`に固定する（`FeEstimator::fit`と同じ理由）。
    crate::shared::parallelism::ensure_serial();

    // 共有の反復コアが受け取る型に直して検証する。`FeGmmType`は`Hac`の重みを持たない。
    let shared_type = gmm_type.to_gmm_type();
    shared_type.validate()?;
    let weight_type = shared_type.weight_type();

    let n = input.nobs();
    let (k_exog, k_instruments) = (input.k_exog(), input.k_instruments());
    let k = input.k();
    let l = k_exog + k_instruments;

    let WithinPrepared {
        levels,
        tilde,
        absorbed: _,
        noise,
        n_groups,
        neffects,
        df_model,
        exog_range,
        endog_range,
        instrument_range,
    } = prepare_within(
        &input,
        effects,
        absorption,
        Some(confidence_level),
        false,
        false,
    )?;

    // `weight_type=Cluster`のグループ（検証して返す）。丁度識別の閾値の違いは`validate_weight_type`が持つ。
    let weight_cluster_codes = match weight_type {
        Some(weight_type) => validate_weight_type(weight_type, n, k, l)?,
        None => None,
    };

    // Z̃ = [x̃_exog, z̃]（全操作変数）、X̃ = [x̃_exog, x̃_endog]（実際の内生変数）、ỹ。
    let column = |range: &std::ops::Range<usize>| -> Vec<&[f64]> {
        tilde[range.clone()].iter().map(Vec::as_slice).collect()
    };
    let (exog, endog, instruments) = (
        column(&exog_range),
        column(&endog_range),
        column(&instrument_range),
    );
    let z = mat_from_columns(n, &[exog.as_slice(), instruments.as_slice()].concat());
    let x = mat_from_columns(n, &[exog.as_slice(), endog.as_slice()].concat());
    let y = mat_from_columns(n, &[tilde[0].as_slice()]);

    let mut param_names = input.x_exog_names().to_vec();
    param_names.extend(input.x_endog_names().iter().cloned());

    // 初期推定: W₀ = (Z̃'Z̃)⁻¹（2SLSと同じ重み）。
    let GmmInitialStep {
        ztz,
        beta: beta0,
        residuals: residuals0,
        classical_s,
    } = gmm_initial_step(&z, &x, &y)?;

    // 初期推定の残差が丸め誤差の範囲で0なら完全適合。重み行列`S`が0になって後段の逆行列が
    // 特異になる前に、原因が分かるエラーで止める。
    let sst_within: f64 = tilde[0].iter().map(|v| v * v).sum();
    let ssr0: f64 = (0..n).map(|i| (*residuals0.get(i, 0)).powi(2)).sum();
    if structural_fit_is_perfect(&input, &beta0, ssr0, sst_within, noise) {
        // GMMに第二段階は無いので、`stage`を付けずに返す（`iv::gmm`と同じ）。
        return Err(CommonError::PerfectFit.into());
    }

    // 方式（`gmm_type`）に応じた回数の再推定。
    let GmmIteration {
        beta,
        residuals,
        s_used,
        n_iter,
        converged,
        hac_lags_used: _,
    } = gmm_iterate(
        &shared_type,
        weight_type,
        weight_cluster_codes,
        &z,
        &x,
        &y,
        &ztz,
        beta0,
        residuals0,
        classical_s,
    )?;

    if !converged && raise_on_non_convergence {
        return Err(CommonError::GmmNonConvergence {
            n_iter: n_iter as usize,
        }
        .into());
    }

    // Hansen J過剰識別検定（内生変数が無い・丁度識別のときは検定しない）。`Classical`・`OneStep`の`S`は
    // `σ̂²Z̃'Z̃`なので、`σ̂²`の分母を`n − neffects`に直す（`hansen_j.rs`）。
    let (k_endog, k_instruments) = (input.k_endog(), input.k_instruments());
    let hansen_j = if k_endog > 0 && k_instruments > k_endog {
        let homoskedastic_weight = gmm_type.has_homoskedastic_weight();
        Some(fe_hansen_j(
            &z,
            &residuals,
            &s_used,
            homoskedastic_weight,
            neffects,
            k_instruments - k_endog,
        )?)
    } else {
        None
    };

    // 標準誤差。`Ω̂`は`cov_type`ごとの小標本補正込み（`FeIvEstimator`と同じ`df_resid`・`K`）。
    let df_resid = n - df_model;
    let residual_values: Vec<f64> = (0..n).map(|i| *residuals.get(i, 0)).collect();
    let groups = input.index().effect_codes();
    let PanelMoment {
        omega,
        dk_bandwidth_used,
    } = panel_moment_covariance(
        &cov_type,
        &PanelMomentInput {
            z: &z,
            ztz: &ztz,
            residuals: &residual_values,
            df_resid,
            df_model,
            n_slopes: k,
            entity: &groups[0],
            levels: &levels,
        },
    )?;
    let cov_params = symmetrized(gmm_sandwich_covariance(&z, &x, &s_used, &omega)?);

    let normal = Normal::new(0.0, 1.0).map_err(|e| CommonError::internal(e.to_string()))?;
    let InferenceTable {
        std_errors,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
    } = inference_table(&beta, &cov_params, &normal, confidence_level);
    // `k = 0`（`x_exog`も`x_endog`も無い）はバインディングが拒否するので、通常は`Some`。
    let wald = if k == 0 {
        None
    } else {
        Some(wald_chi2_test(&beta, &cov_params, 0, k).map_err(|source| {
            PanelError::FTestFailed {
                source: source.into(),
            }
        })?)
    };

    // R²は構造残差ベース。between・overallは変換前の`y`と実際の`X`で`FeEstimator`と同じ式。
    let ssr: f64 = residual_values.iter().map(|r| r * r).sum();
    let mut structural_x: Vec<Vec<f64>> = input.x_exog().to_vec();
    structural_x.extend(input.x_endog().iter().cloned());
    let r_squared_within = 1.0 - ssr / sst_within;
    let r_squared_between = fe_r_squared_between(input.y(), &structural_x, &beta, &groups[0]);
    let r_squared_overall = fe_r_squared_overall(input.y(), &structural_x, &beta);

    Ok(FeGmmEstimator {
        input,
        effects,
        gmm_type,
        param_names,
        params: beta,
        residuals,
        s_used,
        n_iter,
        converged,
        df_model,
        df_resid,
        n_groups,
        cov_type,
        dk_bandwidth_used,
        std_errors,
        cov_params,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
        wald,
        r_squared_within,
        r_squared_between,
        r_squared_overall,
        hansen_j,
    })
}

/// 丸めで厳密には非対称な`cov_params`を対称化する（`gmm_sandwich_covariance`は対称化しない）。
fn symmetrized(cov: Mat<f64>) -> Mat<f64> {
    Mat::from_fn(cov.nrows(), cov.ncols(), |i, j| {
        0.5 * (*cov.get(i, j) + *cov.get(j, i))
    })
}

#[cfg(test)]
mod tests;
