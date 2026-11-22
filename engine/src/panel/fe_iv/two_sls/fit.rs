//! `FeTwoSlsEstimator::fit`の本体。操作変数の数・singleton・定数列の検査、全列のwithin変換（1回）、
//! 自由度の検証、第一段階、2SLSの点推定、`cov_type`別の共分散・統計量をつなぐオーケストレーション。
//! パイプラインの契約はモジュールdoc（`two_sls.rs`）が正本。

use statrs::distribution::StudentsT;

use super::super::common::{
    FeIvInput, FirstStageArgs, FirstStageWithin, FirstStages, WithinPrepared, WuHausmanContext,
    effect_names, fe_wu_hausman_test, mat_from_columns, pick_absorbed, prepare_within,
    run_first_stages, structural_fit_is_perfect,
};
use super::FeTwoSlsEstimator;
use super::sargan::fe_sargan_test;
use crate::linear::common::LeastSquaresError;
use crate::panel::common::{
    PanelCov, PanelCovInput, PanelError, PanelStructure, leverage_full, overall_f_test,
    panel_cov_params,
};
use crate::panel::fe::{
    FeAbsorption, FeCovType, FeEffects, FeEstimator, FeInput, fe_r_squared_between,
    fe_r_squared_overall,
};
use crate::shared::error::CommonError;
use crate::shared::inference::{InferenceTable, inference_table};
use crate::shared::instrumental_variables::two_sls_point_estimate;
use crate::shared::orthonormal::OrthonormalDesign;

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
    consume: impl FnMut(&str, FeEstimator) -> T,
) -> Result<(FeTwoSlsEstimator, Vec<T>), PanelError> {
    // faerのグローバル並列度を`Par::Seq`に固定する（`FeEstimator::fit`と同じ理由）。
    crate::shared::parallelism::ensure_serial();

    let groups = input.index().effect_codes();
    let n = input.nobs();
    let k_endog = input.k_endog();
    let k_instruments = input.k_instruments();
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

    // 第一段階: `x̃_endog[j] ~ x̃_exog + 操作変数`。変換済みの列を使い回す（固定効果つきGMMと共有）。
    let effect_names = effect_names(&input);
    let FirstStages {
        consumed,
        weak_instrument_f,
        summaries: mut first_stage_summaries,
    } = run_first_stages(
        &FirstStageArgs {
            input: &input,
            effects,
            cov_type: &cov_type,
            confidence_level,
            effect_names: &effect_names,
            tilde: &tilde,
            absorbed: &absorbed,
            noise,
            neffects,
            exog_range: &exog_range,
            instrument_range: &instrument_range,
        },
        consume,
    )?;
    // 当てはめ値 = 変換後の内生変数 − 第一段階の残差。
    let fitted_endog: Vec<Vec<f64>> = (0..k_endog)
        .map(|j| {
            tilde[endog_range.start + j]
                .iter()
                .zip(&first_stage_summaries[j].residuals)
                .map(|(endog, residual)| endog - residual)
                .collect()
        })
        .collect();

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

    // 全体F検定。傾きの共分散が特異で失敗したときだけ、原因がレバレッジ1の行かを調べる（HC2/HC3と同じ、
    // `X̂`の`Q`と吸収した効果込みのレバレッジ）ので、`Q`を手放す前に試す。成功する推定は求めない。
    let f_test = if k > 0 {
        Some(overall_f_test(
            &coordinates,
            &meat,
            k,
            df_inference,
            || leverage_full(&design.row_leverages(), &levels, n).ok(),
        )?)
    } else {
        None
    };
    drop(design);
    drop(residuals);
    drop((coordinates, meat));

    let t_dist = StudentsT::new(0.0, 1.0, df_inference as f64)
        .map_err(|e| CommonError::internal(e.to_string()))?;
    let InferenceTable {
        std_errors,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
    } = inference_table(&estimate.params, &cov_params, &t_dist, confidence_level);

    // 構造式の共分散・F検定が計算できた後に、補助回帰を回す（構造式のエラーを先に返す）。
    let wu_hausman = fe_wu_hausman_test(WuHausmanContext {
        input: &input,
        effects,
        cov_type: &cov_type,
        confidence_level,
        groups,
        effect_names: &effect_names,
        tilde: &tilde,
        first_stages: &mut first_stage_summaries,
        noise,
    })?;

    // 弱操作変数診断（Cragg-Donald・Kleibergen-Paap・有効F）が使う、第一段階のwithin変換済みの行列。
    // `[x̃_exog, z̃]`は全ての第一段階で同じ列なので1つだけ持つ（第一段階ごとに持たない）。持ったら、
    // within変換した列は解放する。
    let first_stage_within =
        FirstStageWithin::from_tilde(n, &tilde, &exog_range, &endog_range, &instrument_range);
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
