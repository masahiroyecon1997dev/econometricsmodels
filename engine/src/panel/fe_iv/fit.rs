//! `FeIvEstimator::fit`の本体。操作変数の数・singleton・定数列の検査、全列のwithin変換（1回）、
//! 自由度の検証、第一段階、2SLSの点推定、`cov_type`別の共分散・統計量をつなぐオーケストレーション。
//! パイプラインの契約はモジュールdoc（`fe_iv.rs`）が正本。

use faer::Mat;
use statrs::distribution::StudentsT;

use super::FeIvEstimator;
use super::input::FeIvInput;
use crate::linear::common::LeastSquaresError;
use crate::panel::common::{
    ColumnRole, PanelCov, PanelCovInput, PanelDimension, PanelError, PanelStructure,
    absorbed_parameters, ensure_no_singleton_groups, ensure_no_zero_variance_columns,
    panel_cov_params,
};
use crate::panel::fe::{
    AbsorbedEffects, FeAbsorption, FeCovType, FeEffects, FeEstimator, FeInput, WithinColumns,
    WithinOutput, fe_r_squared_between, fe_r_squared_overall, within_transform_columns,
};
use crate::shared::error::{CommonError, ErrorInfo};
use crate::shared::error_catalog::Stage;
use crate::shared::goodness_of_fit::is_perfect_fit;
use crate::shared::inference::{InferenceTable, inference_table};
use crate::shared::instrumental_variables::two_sls_point_estimate;
use crate::shared::validation::{column_is_zero_variance_with_noise, has_no_variation};
use crate::shared::wald::wald_f_test;

/// 第一段階の失敗に`stage=FIRST_STAGE`を付ける。`stage`を付けられない`code`（`confidence_level`の
/// 範囲違反など）は、そのまま返す。
fn in_first_stage(err: PanelError, endog_name: &str) -> PanelError {
    if err.report().code().spec().stage_allowed {
        err.in_stage(Stage::FirstStage {
            endog_name: endog_name.to_string(),
        })
    } else {
        err
    }
}

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

/// 列`columns`（各列は長さ`n`）を並べた`(n, 列数)`の行列。
fn mat_from_columns(n: usize, columns: &[&[f64]]) -> Mat<f64> {
    Mat::from_fn(n, columns.len(), |i, j| columns[j][i])
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
    // 吸収する次元。`time`の無い2-wayや次元が足りない`MultiWay`はここで弾く。
    let levels = effects.levels(groups)?;
    absorption.validate()?;
    if !(confidence_level > 0.0 && confidence_level < 1.0) {
        return Err(PanelError::WithinRegressionFailed {
            source: CommonError::InvalidConfidenceLevel { confidence_level }.into(),
        });
    }

    let n = input.nobs();
    let (k_exog, k_endog, k_instruments) = (input.k_exog(), input.k_endog(), input.k_instruments());
    let k = input.k();
    let l = k_exog + k_instruments;

    // 識別の順序条件（`IvInput`の検査と同じ`code`）。
    if k_instruments < k_endog {
        return Err(PanelError::InsufficientInstruments {
            n_instruments: k_instruments,
            n_endog: k_endog,
        });
    }

    ensure_no_singleton_groups(input.index().effect_names(), &levels)?;

    // 生の値が全ての行で同じ列は、固定効果を除く前に分かる。固定効果を除いた後で分散がなくなる場合
    // （時間不変の列など）とは、エラーを分ける。重い変換より先に調べる。
    let roles = [
        (ColumnRole::XExog, input.x_exog_names(), input.x_exog()),
        (ColumnRole::XEndog, input.x_endog_names(), input.x_endog()),
        (
            ColumnRole::Instruments,
            input.instrument_names(),
            input.instruments(),
        ),
    ];
    for (role, names, columns) in roles {
        for (name, column) in names.iter().zip(columns) {
            if has_no_variation(column) {
                return Err(PanelError::RegressorConstant {
                    role,
                    column: name.clone(),
                });
            }
        }
    }
    if has_no_variation(input.y()) {
        return Err(CommonError::OutcomeConstant.into());
    }

    // 全ての列を1回だけ変換する。列の順は`y`、`x_exog`、`x_endog`、`instruments`。
    let leverage_needed = matches!(cov_type, FeCovType::Hc2 | FeCovType::Hc3);
    let mut columns: Vec<&[f64]> = Vec::with_capacity(1 + k_exog + k_endog + k_instruments);
    columns.push(input.y());
    columns.extend(input.x_exog().iter().map(Vec::as_slice));
    columns.extend(input.x_endog().iter().map(Vec::as_slice));
    columns.extend(input.instruments().iter().map(Vec::as_slice));
    let WithinColumns {
        columns: tilde,
        absorbed,
        noise,
    } = within_transform_columns(&levels, &columns, absorption, leverage_needed)?;
    let (exog_range, endog_range, instrument_range) = (
        1..1 + k_exog,
        1 + k_exog..1 + k,
        1 + k..1 + k + k_instruments,
    );
    let y_tilde = &tilde[0];

    // 自由度。構造式を先に、次に第一段階を検証する。
    let n_groups: Vec<usize> = levels.iter().map(|level| level.n_groups()).collect();
    let neffects = absorbed_parameters(&n_groups);
    let df_model = k + neffects;
    if n <= df_model {
        return Err(PanelError::InsufficientDegreesOfFreedom {
            n_obs: n,
            n_groups,
            k,
        });
    }
    // 第一段階が無い（内生変数が無い）ときは検証しない。
    if k_endog > 0 && n <= l + neffects {
        let first_name = input.x_endog_names().first().map_or("", String::as_str);
        return Err(in_first_stage(
            PanelError::InsufficientDegreesOfFreedom {
                n_obs: n,
                n_groups,
                k: l,
            },
            first_name,
        ));
    }
    let df_resid = n - df_model;

    // 固定効果を除いた後に分散が残らない列（役割ごとに報告）と被説明変数。
    let transformed_roles = [
        (
            ColumnRole::XExog,
            input.x_exog_names(),
            input.x_exog(),
            &tilde[exog_range.clone()],
        ),
        (
            ColumnRole::XEndog,
            input.x_endog_names(),
            input.x_endog(),
            &tilde[endog_range.clone()],
        ),
        (
            ColumnRole::Instruments,
            input.instrument_names(),
            input.instruments(),
            &tilde[instrument_range.clone()],
        ),
    ];
    for (role, names, raw, transformed) in transformed_roles {
        ensure_no_zero_variance_columns(role, names, raw, transformed, noise)?;
    }
    if column_is_zero_variance_with_noise(input.y(), y_tilde, noise) {
        return Err(PanelError::OutcomeConstantWithinEntity);
    }

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
    // 交互射影が残す誤差は、構造残差`ỹ − X̃β̂`では`y`だけでなく`|β̂_j|`倍した各`x`の分にも乗る。
    let abs_max = |column: &[f64]| column.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let residual_scale = abs_max(input.y())
        + input
            .x_exog()
            .iter()
            .chain(input.x_endog())
            .enumerate()
            .map(|(j, column)| estimate.params.get(j, 0).abs() * abs_max(column))
            .sum::<f64>();
    if is_perfect_fit(estimate.ssr, sst_within, k)
        || estimate.ssr <= n as f64 * (noise * residual_scale).powi(2)
    {
        return Err(PanelError::SecondStageFailed {
            source: CommonError::PerfectFit.into(),
        });
    }

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
        structural,
        absorbed: structural_absorbed,
    })
}

#[cfg(test)]
mod tests;
