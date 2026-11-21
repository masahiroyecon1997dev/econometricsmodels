//! `FeTwoSlsEstimator::fit`の本体。操作変数の数・singleton・定数列の検査、全列のwithin変換（1回）、
//! 自由度の検証、第一段階、2SLSの点推定、`cov_type`別の共分散・統計量をつなぐオーケストレーション。
//! パイプラインの契約はモジュールdoc（`two_sls.rs`）が正本。

use statrs::distribution::StudentsT;

use super::super::common::{
    FeIvInput, WithinPrepared, in_first_stage, mat_from_columns, prepare_within,
    structural_fit_is_perfect,
};
use super::sargan::fe_sargan_test;
use super::weak_instruments::first_stage_partial_f;
use super::wu_hausman::{FirstStageSummary, WuHausmanContext, fe_wu_hausman_test};
use super::{FeTwoSlsEstimator, FirstStageWithin};
use crate::linear::common::LeastSquaresError;
use crate::panel::common::{
    PanelCov, PanelCovInput, PanelDimension, PanelError, PanelStructure, overall_f_test,
    panel_cov_params,
};
use crate::panel::fe::{
    AbsorbedEffects, FeAbsorption, FeCovType, FeEffects, FeEstimator, FeInput, WithinOutput,
    fe_r_squared_between, fe_r_squared_overall,
};
use crate::shared::error::CommonError;
use crate::shared::inference::{InferenceTable, inference_table};
use crate::shared::instrumental_variables::two_sls_point_estimate;
use crate::shared::orthonormal::OrthonormalDesign;

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

/// [`FeTwoSlsEstimator::fit_with_first_stages`]の本体。契約（エラー条件を含む）は`fit`のdocコメント参照。
///
/// 第一段階の`FeEstimator`（生の列・within行列・残差を持つ）は、部分F統計量と、第二段階・Wu-Hausmanに
/// 要る量（当てはめ値・残差・係数）を取り出した直後に`consume`へ渡して手放す。内生変数の数だけ
/// `fit`の終わりまで持つと、ピークメモリがその数に比例して増える。
pub(super) fn fit_fe_two_sls<T>(
    input: FeIvInput,
    effects: FeEffects,
    cov_type: FeCovType,
    confidence_level: f64,
    absorption: &FeAbsorption,
    mut consume: impl FnMut(&str, FeEstimator) -> T,
) -> Result<(FeTwoSlsEstimator, Vec<T>), PanelError> {
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

    let mut consumed: Vec<T> = Vec::with_capacity(k_endog);
    let mut first_stage_summaries: Vec<FirstStageSummary> = Vec::with_capacity(k_endog);
    let mut weak_instrument_f = Vec::with_capacity(k_endog);
    let mut fitted_endog: Vec<Vec<f64>> = Vec::with_capacity(k_endog);
    for j in 0..k_endog {
        let endog_name = &input.x_endog_names()[j];
        let stage_x: Vec<Vec<f64>> = input
            .x_exog()
            .iter()
            .chain(input.instruments())
            .cloned()
            .collect();
        let stage_input = FeInput::from_effect_groups_owned(
            input.x_endog()[j].clone(),
            stage_x,
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

        // 弱操作変数の部分F統計量（診断）は第一段階の残差平方和と係数の数だけを使うので、手放す前に求める。
        let f = first_stage_partial_f(&stage, k_exog, k_instruments, neffects)
            .map_err(|err| in_first_stage(err.into(), endog_name))?;
        weak_instrument_f.push((endog_name.clone(), f));
        first_stage_summaries.push(FirstStageSummary {
            params: (0..stage.params().nrows())
                .map(|l| *stage.params().get(l, 0))
                .collect(),
            residuals: (0..n).map(|i| *stage.residuals().get(i, 0)).collect(),
        });
        consumed.push(consume(endog_name, stage));
    }

    // 点推定。`X̃`は実際の内生変数、`X̂`は内生変数を第一段階の当てはめ値に置き換えたもの。
    // `X̃`・`ỹ`・当てはめ値は点推定の後に使わないので、ブロックを抜けるときに解放する
    // （共分散・Wu-Hausmanの間も保持するとピークメモリが列数の数倍になる）。
    let mut param_names = input.x_exog_names().to_vec();
    param_names.extend(input.x_endog_names().iter().cloned());
    let (estimate, design) = {
        let exog_columns: Vec<&[f64]> = tilde[exog_range.clone()]
            .iter()
            .map(Vec::as_slice)
            .collect();
        let endog_columns: Vec<&[f64]> = tilde[endog_range.clone()]
            .iter()
            .map(Vec::as_slice)
            .collect();
        let fitted_columns: Vec<&[f64]> = fitted_endog.iter().map(Vec::as_slice).collect();
        // `X̂ = QR`（ピボットなしQR）1本から、係数・構造残差・共分散・全体Fを求める。ランク落ちと
        // 悪条件は`new_checked`が判定する。`X̂`は`Q`を作った直後に解放し、`X̃`はその後に作る
        // （`X̂`・QRの作業領域・`Q`が重なる分より、行列を同時に持たない）。
        let x_hat = mat_from_columns(
            n,
            &[exog_columns.as_slice(), fitted_columns.as_slice()].concat(),
        );
        let design = OrthonormalDesign::new_checked(&x_hat).map_err(|err| {
            PanelError::SecondStageFailed {
                source: LeastSquaresError::from_rank_deficient(err, &x_hat, &param_names),
            }
        })?;
        drop(x_hat);
        let x_tilde = mat_from_columns(
            n,
            &[exog_columns.as_slice(), endog_columns.as_slice()].concat(),
        );
        let y_mat = mat_from_columns(n, &[y_tilde.as_slice()]);
        (two_sls_point_estimate(&design, &x_tilde, &y_mat), design)
    };
    drop(fitted_endog);

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

    // 係数の共分散と全体F検定は、上のQRの直交座標（`b̃ = Q'ỹ`、構造残差のミート`M̃`）で求める。
    // `V = B M B`（`B = (X̂'X̂)⁻¹`）を作ってから反転すると、傾きの相関が極端に高いときに`B`の
    // 明示的な逆行列で桁落ちする。`M̃`は`panel_cov_params`に`x = Q`・`xtx_inv = 単位行列`を
    // 渡して得て、`V = R⁻¹ M̃ R⁻ᵀ`と統計量の両方に使う（`shared::orthonormal`）。`β = (X̂'X̂)⁻¹X̂'ỹ`
    // なので`Rβ = Q'ỹ`（点推定が返す`coordinates`）。
    let residuals: Vec<f64> = (0..n).map(|i| *estimate.residuals.get(i, 0)).collect();
    let coordinates = estimate.coordinates.clone();
    let PanelCov {
        cov_params: meat,
        df_inference,
        dk_bandwidth_used,
    } = panel_cov_params(
        &cov_type,
        &PanelCovInput {
            x: design.q(),
            residuals: &residuals,
            xtx_inv: &design.identity(),
            df_resid,
            df_model,
            n_slopes: k,
            tests_slopes: true,
            entity: &groups[0],
            structure: PanelStructure::FixedEffects { levels: &levels },
        },
    )?;
    let cov_params = design.coefficient_covariance(&meat);
    drop(design);
    drop(residuals);
    let wald = (k > 0).then_some((coordinates, meat));

    let t_dist = StudentsT::new(0.0, 1.0, df_inference as f64)
        .map_err(|e| CommonError::internal(e.to_string()))?;
    let InferenceTable {
        std_errors,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
    } = inference_table(&estimate.params, &cov_params, &t_dist, confidence_level);

    let f_test = match &wald {
        None => None,
        Some((coordinates, meat)) => Some(overall_f_test(coordinates, meat, k, df_inference)?),
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
        first_stages: &first_stage_summaries,
        noise,
    })?;

    // 弱操作変数診断（Cragg-Donald・Kleibergen-Paap・有効F）が使う、第一段階のwithin変換済みの行列。
    // `[x̃_exog, z̃]`は全ての第一段階で同じ列なので1つだけ持つ（第一段階ごとに持たない）。持ったら、
    // within変換した列は解放する。
    let first_stage_within = (k_endog > 0).then(|| {
        let first_stage_x: Vec<&[f64]> = exog_range
            .clone()
            .chain(instrument_range.clone())
            .map(|c| tilde[c].as_slice())
            .collect();
        let endog_columns: Vec<&[f64]> = endog_range.clone().map(|c| tilde[c].as_slice()).collect();
        FirstStageWithin {
            x: mat_from_columns(n, &first_stage_x),
            y_endog: mat_from_columns(n, &endog_columns),
        }
    });
    drop(tilde);

    // 構造式の`y`・実際の`X`（変換前）。R²のbetween・overallと`fixed_effects()`が使う。
    let structural_x: Vec<Vec<f64>> = input
        .x_exog()
        .iter()
        .chain(input.x_endog())
        .cloned()
        .collect();
    let structural = FeInput::from_effect_groups_owned(
        input.y().to_vec(),
        structural_x,
        param_names.clone(),
        groups.to_vec(),
        input.dep_var_name().to_string(),
    )?
    .with_effect_names(effect_names);
    let r_squared_within = 1.0 - estimate.ssr / sst_within;
    let r_squared_between =
        fe_r_squared_between(structural.y(), structural.x(), &estimate.params, &groups[0]);
    let r_squared_overall = fe_r_squared_overall(structural.y(), structural.x(), &estimate.params);
    let structural_absorbed = pick_absorbed(&absorbed, 0, 1..1 + k);

    let estimator = FeTwoSlsEstimator {
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
        first_stage_within,
        weak_instrument_f,
        sargan,
        wu_hausman,
        structural,
        absorbed: structural_absorbed,
    };
    Ok((estimator, consumed))
}

#[cfg(test)]
mod tests;
