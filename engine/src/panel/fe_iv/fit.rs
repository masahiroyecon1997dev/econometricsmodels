//! `FeIvEstimator::fit`の本体。操作変数の数・singleton・定数列の検査、全列のwithin変換（1回）、
//! 自由度の検証、第一段階、2SLSの点推定、`cov_type`別の共分散・統計量をつなぐオーケストレーション。
//! パイプラインの契約はモジュールdoc（`fe_iv.rs`）が正本。

use statrs::distribution::StudentsT;

use super::FeIvEstimator;
use super::input::FeIvInput;
use super::prepare::{
    WithinPrepared, in_first_stage, mat_from_columns, prepare_within, structural_fit_is_perfect,
};
use super::sargan::fe_sargan_test;
use super::weak_instruments::first_stage_partial_f;
use super::wu_hausman::{WuHausmanContext, fe_wu_hausman_test};
use crate::linear::common::LeastSquaresError;
use crate::panel::common::{
    PanelCov, PanelCovInput, PanelDimension, PanelError, PanelStructure, panel_cov_params,
};
use crate::panel::fe::{
    AbsorbedEffects, FeAbsorption, FeCovType, FeEffects, FeEstimator, FeInput, WithinOutput,
    fe_r_squared_between, fe_r_squared_overall,
};
use crate::shared::error::CommonError;
use crate::shared::inference::{InferenceTable, inference_table};
use crate::shared::instrumental_variables::two_sls_point_estimate;
use crate::shared::wald::wald_f_test;

/// 列の位置で選んだ列の吸収した効果（`absorbed[列][次元][水準]`から、`y`の列と`x`の列を取り出す）。
fn pick_absorbed(
    absorbed: &Option<Vec<Vec<Vec<f64>>>>,
    y_column: usize,
    x_columns: impl Iterator<Item = usize>,
) -> Option<AbsorbedEffects> {
    absorbed.as_ref().map(|per_column| AbsorbedEffects {
        y: per_column[y_column].clone(),
        x: x_columns.map(|j| per_column[j].clone()).collect(),
    })
}

/// [`FeIvEstimator::fit_with_absorption`]の本体。契約（エラー条件を含む）は`fit`のdocコメント参照。
pub(super) fn fit_fe_iv(
    input: FeIvInput,
    effects: FeEffects,
    cov_type: FeCovType,
    confidence_level: f64,
    absorption: &FeAbsorption,
) -> Result<FeIvEstimator, PanelError> {
    // faerのグローバル並列度を`Par::Seq`に固定する（`FeEstimator::fit`と同じ理由）。
    crate::shared::parallelism::ensure_serial();

    let groups = input.index().effect_codes();
    let n = input.nobs();
    let (k_exog, k_endog, k_instruments) = (input.k_exog(), input.k_endog(), input.k_instruments());
    let k = input.k();
    let WithinPrepared {
        levels,
        tilde,
        absorbed,
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
        matches!(cov_type, FeCovType::Hc2 | FeCovType::Hc3),
        true,
    )?;
    let y_tilde = &tilde[0];
    let df_resid = n - df_model;

    // 第一段階: `x̃_endog[j] ~ x̃_exog + 操作変数`。変換済みの列を使い回す。
    let effect_names: Vec<String> = (0..groups.len())
        .map(|d| {
            input
                .index()
                .effect_name(PanelDimension::new(d))
                .to_string()
        })
        .collect();
    let mut first_stage_names = input.x_exog_names().to_vec();
    first_stage_names.extend(input.instrument_names().iter().cloned());
    let mut first_stage_raw: Vec<Vec<f64>> = input.x_exog().to_vec();
    first_stage_raw.extend(input.instruments().iter().cloned());

    let mut first_stages = Vec::with_capacity(k_endog);
    let mut fitted_endog: Vec<Vec<f64>> = Vec::with_capacity(k_endog);
    for j in 0..k_endog {
        let endog_name = &input.x_endog_names()[j];
        let stage_input = FeInput::from_effect_groups(
            &input.x_endog()[j],
            &first_stage_raw,
            first_stage_names.clone(),
            groups.to_vec(),
            endog_name.clone(),
        )
        .map_err(|err| in_first_stage(err, endog_name))?
        .with_effect_names(effect_names.clone());

        let endog_column = 1 + k_exog + j;
        let x_columns = exog_range.clone().chain(instrument_range.clone());
        let within = WithinOutput {
            y: tilde[endog_column].clone(),
            x: x_columns.clone().map(|c| tilde[c].clone()).collect(),
            absorbed: pick_absorbed(&absorbed, endog_column, x_columns),
            noise,
        };
        let stage = FeEstimator::fit_from_within(
            stage_input,
            effects,
            cov_type.clone(),
            confidence_level,
            within,
            false,
        )
        .map_err(|err| in_first_stage(err, endog_name))?;

        // 当てはめ値 = 変換後の内生変数 − 第一段階の残差。
        let fitted: Vec<f64> = (0..n)
            .map(|i| tilde[endog_column][i] - stage.residuals().get(i, 0))
            .collect();
        fitted_endog.push(fitted);
        first_stages.push(stage);
    }

    // 弱操作変数の部分F統計量（診断。内生変数ごと）。
    let mut weak_instrument_f = Vec::with_capacity(k_endog);
    for (j, stage) in first_stages.iter().enumerate() {
        let name = &input.x_endog_names()[j];
        let f = first_stage_partial_f(stage, k_exog, k_instruments, neffects)
            .map_err(|err| in_first_stage(err.into(), name))?;
        weak_instrument_f.push((name.clone(), f));
    }

    // 点推定。`X̃`は実際の内生変数、`X̂`は内生変数を第一段階の当てはめ値に置き換えたもの。
    let exog_columns: Vec<&[f64]> = tilde[exog_range.clone()]
        .iter()
        .map(Vec::as_slice)
        .collect();
    let endog_columns: Vec<&[f64]> = tilde[endog_range].iter().map(Vec::as_slice).collect();
    let fitted_columns: Vec<&[f64]> = fitted_endog.iter().map(Vec::as_slice).collect();
    let x_tilde = mat_from_columns(
        n,
        &[exog_columns.as_slice(), endog_columns.as_slice()].concat(),
    );
    let x_hat = mat_from_columns(
        n,
        &[exog_columns.as_slice(), fitted_columns.as_slice()].concat(),
    );
    let y_mat = mat_from_columns(n, &[y_tilde.as_slice()]);
    let mut param_names = input.x_exog_names().to_vec();
    param_names.extend(input.x_endog_names().iter().cloned());

    let estimate = two_sls_point_estimate(&x_tilde, &x_hat, &y_mat).map_err(|err| {
        PanelError::SecondStageFailed {
            source: LeastSquaresError::from_rank_deficient(err, &x_hat, &param_names),
        }
    })?;

    // 構造残差が丸め誤差の範囲で0なら完全適合。標準誤差とF検定が定義できない。
    let sst_within: f64 = y_tilde.iter().map(|v| v * v).sum();
    if structural_fit_is_perfect(&input, &estimate.params, estimate.ssr, sst_within, noise) {
        return Err(PanelError::SecondStageFailed {
            source: CommonError::PerfectFit.into(),
        });
    }

    // Sargan過剰識別検定。`Z̃ = [x̃_exog, z̃]`は第一段階の設計行列と同じ列。内生変数が無いときは検定しない。
    let sargan = if k_endog > 0 && k_instruments > k_endog {
        let z_columns: Vec<&[f64]> = exog_range
            .clone()
            .chain(instrument_range.clone())
            .map(|c| tilde[c].as_slice())
            .collect();
        Some(fe_sargan_test(
            &mat_from_columns(n, &z_columns),
            &estimate.residuals,
            estimate.ssr,
            neffects,
            k_instruments - k_endog,
        )?)
    } else {
        None
    };

    let residuals: Vec<f64> = (0..n).map(|i| *estimate.residuals.get(i, 0)).collect();
    let PanelCov {
        cov_params,
        df_inference,
        dk_bandwidth_used,
    } = panel_cov_params(
        &cov_type,
        &PanelCovInput {
            x: &x_hat,
            residuals: &residuals,
            xtx_inv: &estimate.xtx_hat_inv,
            df_resid,
            df_model,
            n_slopes: k,
            tests_slopes: true,
            entity: &groups[0],
            structure: PanelStructure::FixedEffects { levels: &levels },
        },
    )?;

    let t_dist = StudentsT::new(0.0, 1.0, df_inference as f64)
        .map_err(|e| CommonError::internal(e.to_string()))?;
    let InferenceTable {
        std_errors,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
    } = inference_table(&estimate.params, &cov_params, &t_dist, confidence_level);

    let f_test = if k == 0 {
        None
    } else {
        Some(
            wald_f_test(&estimate.params, &cov_params, 0, k, df_inference).map_err(|source| {
                PanelError::FTestFailed {
                    source: source.into(),
                }
            })?,
        )
    };

    // 構造式の共分散・F検定が計算できた後に、補助回帰を回す（構造式のエラーを先に返す）。
    let wu_hausman = fe_wu_hausman_test(&WuHausmanContext {
        input: &input,
        effects,
        cov_type: &cov_type,
        confidence_level,
        groups,
        effect_names: &effect_names,
        tilde: &tilde,
        first_stages: &first_stages,
        noise,
    })?;

    // 構造式の`y`・実際の`X`（変換前）。R²のbetween・overallと`fixed_effects()`が使う。
    let mut structural_x: Vec<Vec<f64>> = input.x_exog().to_vec();
    structural_x.extend(input.x_endog().iter().cloned());
    let r_squared_within = 1.0 - estimate.ssr / sst_within;
    let r_squared_between =
        fe_r_squared_between(input.y(), &structural_x, &estimate.params, &groups[0]);
    let r_squared_overall = fe_r_squared_overall(input.y(), &structural_x, &estimate.params);

    let structural = FeInput::from_effect_groups(
        input.y(),
        &structural_x,
        param_names.clone(),
        groups.to_vec(),
        input.dep_var_name().to_string(),
    )?
    .with_effect_names(effect_names);
    let structural_absorbed = pick_absorbed(&absorbed, 0, 1..1 + k);

    Ok(FeIvEstimator {
        input,
        effects,
        cov_type,
        param_names,
        params: estimate.params,
        residuals: estimate.residuals,
        df_model,
        df_resid,
        df_inference,
        dk_bandwidth_used,
        n_groups,
        std_errors,
        cov_params,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
        r_squared_within,
        r_squared_between,
        r_squared_overall,
        f_test,
        first_stages,
        weak_instrument_f,
        sargan,
        wu_hausman,
        structural,
        absorbed: structural_absorbed,
    })
}

#[cfg(test)]
mod tests;
