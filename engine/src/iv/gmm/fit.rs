//! `GmmEstimator::fit`の本体（検証・反復・Hansen J・共分散・検定をつなぐオーケストレーション）。
//!
//! テストは`fit/tests.rs`と`fit/tests/`にテーマ別で置く（`basic`・`reference`・`cov_type`・`validation`・`proptests`等）。

use super::GmmEstimator;
use super::GmmType;
use super::hansen_j::hansen_j_test;
use super::sandwich::{GmmCovariance, sandwich_covariance};
use crate::iv::common::{
    IvError, IvInput, compute_first_stage_with, hcat, mat_column_to_vec, validate_iv_fit_inputs,
    wu_hausman_test,
};
use crate::linear::ols::CovType;
use crate::shared::error::CommonError;
use crate::shared::gmm_iteration::{GmmInitialStep, GmmIteration, gmm_initial_step, gmm_iterate};
use crate::shared::gmm_weight::validate_weight_type;
use crate::shared::goodness_of_fit::{is_perfect_fit, total_sum_of_squares};
use crate::shared::inference::{InferenceTable, inference_table};
use crate::shared::normal::StandardNormal;
use crate::shared::wald::wald_chi2_test;

/// [`GmmEstimator::fit`]・[`GmmEstimator::fit_with_first_stage_residuals`]の本体。契約（エラー条件を
/// 含む）は`fit`のdocコメント参照。
///
/// `first_stage_residuals`は内生変数ごとの第一段階のOLS残差（`x_endog`の順、各`n`個、借用）。`None`なら
/// Wu-Hausmanの補助回帰の直前にここで第一段階を回して求める（`Some`なら呼び出し側が回した第一段階の
/// 残差をそのまま使う。形が合わないのは呼び出し側の誤りで、`INTERNAL_ERROR`）。
pub(super) fn fit_gmm(
    input: IvInput,
    gmm_type: GmmType,
    raise_on_non_convergence: bool,
    cov_type: CovType,
    confidence_level: f64,
    first_stage_residuals: Option<&[&[f64]]>,
) -> Result<GmmEstimator, IvError> {
    // faer のグローバル並列度を Par::Seq に固定する（`crate::shared::parallelism`）。
    crate::shared::parallelism::ensure_serial();

    let n = input.nobs();
    // `confidence_level`の範囲・反復設定（`gmm_max_iter`・`gmm_tol`）・識別の順序条件・観測数・
    // `cov_type=Cluster`のクラスター数（構造方程式の`G <= q`）・被説明変数の定数を、点推定・SE計算より前に
    // 弾く（`TwoSlsEstimator::fit`・`engine_pybind`と同じ`validate_iv_fit_inputs`）。観測数の検査は、
    // `n <= k`がサンドイッチSE計算のNaN/Infinityを、`n <= l`が`Z'Z`の特異または
    // `W₀ = (Z'Z)⁻¹`のGMMのOLSへの縮退を防ぐ。
    // `weight_type=Cluster`の重み行列`S`（l×l）の`G`と`l`の関係は別軸のため
    // `validate_weight_type`側で検証する。
    validate_iv_fit_inputs(&input, &cov_type, confidence_level, Some(&gmm_type))?;
    if let Some(columns) = first_stage_residuals
        && (columns.len() != input.k_endog() || columns.iter().any(|column| column.len() != n))
    {
        return Err(CommonError::internal(
            "first-stage residuals do not match the endogenous variables and observations",
        )
        .into());
    }
    let k = input.k_exog() + input.k_endog();

    // 1段階GMMは`weight_type`を持たない（点推定・Hansen Jとも`(Z'Z)⁻¹`のみで計算する）
    // ため検証もしない。2段階・反復では点推定に使う重みの妥当性を検証する。
    let weight_type = gmm_type.weight_type();
    let l_instruments = input.k_exog() + input.instruments().ncols();
    // `weight_type=Cluster`のグループ（`validate_weight_type`が検証して返す）。`cov_type=Cluster`
    // のグループとは別の入力で、同じ列を指していても別々に渡される。
    let weight_cluster_codes = match weight_type {
        Some(weight_type) => validate_weight_type(weight_type, n, k, l_instruments)?,
        None => None,
    };

    // Z = x_exog ++ instruments（全操作変数、docs/spec/iv-spec.md 1.1節）。
    let z = hcat(input.x_exog(), input.instruments());

    // X = x_exog ++ x_endog（実際の内生変数。2SLSの第二段階のようなX̂への
    // 置き換えはしない、`gmm.rs`のモジュールdoc参照）。
    let x = hcat(input.x_exog(), input.x_endog());

    let mut param_names = input.x_exog_names().to_vec();
    param_names.extend(input.x_endog_names().iter().cloned());

    let y = input.y();

    // 初期推定: W₀ = (Z'Z)⁻¹（weight_typeによらず共通、2SLSと同じ重み）。`classical_s`は
    // weight_type=ClassicalのHansen J検定専用の重み行列`σ̂²₀・Z'Z`（`gmm.rs`のモジュールdoc
    // 「Hansen J過剰識別検定」参照。`σ̂²`によるスケーリングは点推定には不要だがHansen Jには必須。
    // `OneStep`ではstep-0残差＝最終残差のため、この一箇所の計算で両方のケースをカバーできる）。
    let GmmInitialStep {
        ztz,
        solution: solution0,
        beta: beta0,
        residuals: residuals0,
        classical_s,
    } = gmm_initial_step(&z, &x, y)?;

    // 初期推定の残差が丸め誤差の範囲で0なら完全フィット。重み行列`S`が0になって後段の逆行列が
    // 特異になる前に、原因が分かるエラーで止める（`TwoSlsEstimator::fit`と同じ判定）。
    let sst = total_sum_of_squares(&mat_column_to_vec(y, 0), None, input.has_intercept());
    let ssr0: f64 = (0..n).map(|i| (*residuals0.get(i, 0)).powi(2)).sum();
    if is_perfect_fit(ssr0, sst, k) {
        return Err(CommonError::PerfectFit.into());
    }

    // 方式（`gmm_type`）に応じた回数の再推定（詳細は`gmm_iterate`のdocコメント）。
    let GmmIteration {
        beta,
        solution,
        residuals,
        s_used: _,
        n_iter,
        converged,
        hac_lags_used,
    } = gmm_iterate(
        &gmm_type,
        weight_type,
        weight_cluster_codes,
        &z,
        &x,
        y,
        &ztz,
        solution0,
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

    let hansen_j = hansen_j_test(&input, &solution, &residuals)?;

    // ── ここからサンドイッチ型SE計算（`gmm.rs`のモジュールdoc
    // 「標準誤差・検定統計量（cov_type対応）」参照）──
    let k_constant = usize::from(input.has_intercept());
    let df_resid = n - k;
    let df_model = k - k_constant;
    let ssr: f64 = (0..n).map(|i| (*residuals.get(i, 0)).powi(2)).sum();

    let GmmCovariance {
        cov_params,
        wald,
        hac_lags_used: cov_hac_lags,
    } = sandwich_covariance(&cov_type, &x, &residuals, &solution, df_resid)?;
    // `cov_type=Hac`のラグ数は`weight_type=Hac`のラグ数より優先する。
    let hac_lags_used = cov_hac_lags.or(hac_lags_used);

    let normal_dist = StandardNormal;
    let InferenceTable {
        std_errors,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
    } = inference_table(&beta, &cov_params, &normal_dist, confidence_level);

    let r_squared = 1.0 - ssr / sst;
    let adj_r_squared = 1.0 - ((n - k_constant) as f64 / df_resid as f64) * (1.0 - r_squared);

    let (wald_statistic, wald_p_value) = if df_model == 0 {
        // 説明変数が定数項のみ（傾き係数が無い）モデル。検定対象が存在しないため
        // 2SLSと同様NaNを返す（0除算を避ける）。
        (f64::NAN, f64::NAN)
    } else {
        wald_chi2_test(&wald.coordinates, &wald.meat, k_constant, df_model)?
    };

    // Wu-Hausman内生性検定（`common/wu_hausman.rs`。2SLSと共有）。補助回帰は第一段階の残差を足した最小二乗で、
    // 点推定・重み・共分散を使わないので、構造式のエラーを先に返してから回す。点推定に使った行列と重みは
    // 補助回帰の前に手放す（第一段階の残差を同時に持つ時間を短くする）。
    drop((x, z, solution));
    let computed: Vec<Vec<f64>>;
    let first_stage_residuals: Vec<&[f64]> = match first_stage_residuals {
        Some(columns) => columns.to_vec(),
        None => {
            computed =
                compute_first_stage_with(&input, &cov_type, confidence_level, |_, estimator| {
                    mat_column_to_vec(estimator.residuals(), 0)
                })?
                .0
                .into_iter()
                .map(|(_, residuals)| residuals)
                .collect();
            computed.iter().map(Vec::as_slice).collect()
        }
    };
    let wu_hausman = wu_hausman_test(&input, &first_stage_residuals, &cov_type, confidence_level)?;

    Ok(GmmEstimator {
        params: beta,
        param_names,
        dep_var_name: input.dep_var_name().to_string(),
        residuals,
        gmm_type,
        n_iter,
        converged,
        nobs: n,
        k,
        cov_type,
        hac_lags_used,
        std_errors,
        cov_params,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
        df_resid,
        df_model,
        r_squared,
        adj_r_squared,
        wald_statistic,
        wald_p_value,
        hansen_j_statistic: hansen_j.as_ref().map(|t| t.statistic),
        hansen_j_p_value: hansen_j.as_ref().map(|t| t.p_value),
        hansen_j_df: hansen_j.as_ref().map(|t| t.df),
        wu_hausman_statistic: wu_hausman.map(|t| t.statistic),
        wu_hausman_p_value: wu_hausman.map(|t| t.p_value),
        wu_hausman_df: wu_hausman.map(|t| t.df),
    })
}

#[cfg(test)]
mod tests;
