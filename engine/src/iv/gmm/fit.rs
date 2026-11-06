//! `GmmEstimator::fit`の本体（検証・反復・Hansen J・共分散・検定をつなぐオーケストレーション）。
//!
//! テストは`fit/tests.rs`と`fit/tests/`にテーマ別で置く（`basic`・`reference`・`cov_type`・`validation`・`proptests`等）。

use faer::Mat;
use statrs::distribution::Normal;

use super::GmmEstimator;
use super::hansen_j::hansen_j_test;
use super::iterate::{GmmIteration, GmmType, gmm_iterate, gmm_point_estimate};
use super::sandwich::{GmmCovariance, gmm_wald_chi2_test, sandwich_covariance};
use super::weight::{WeightType, validate_weight_type};
use crate::iv::common::{IvError, IvInput, hcat, mat_column_to_vec, validate_iv_fit_inputs};
use crate::linear::ols::CovType;
use crate::shared::error::CommonError;
use crate::shared::goodness_of_fit::{is_perfect_fit, total_sum_of_squares};
use crate::shared::inference::{InferenceTable, inference_table};
use crate::shared::validation::{GMM_MAX_ITER_MIN, MAX_ITER_LIMIT};

/// [`GmmEstimator::fit`]の本体。契約（エラー条件を含む）は`fit`のdocコメント参照。
pub(super) fn fit_gmm(
    input: IvInput,
    gmm_type: GmmType,
    raise_on_non_convergence: bool,
    cov_type: CovType,
    confidence_level: f64,
) -> Result<GmmEstimator, IvError> {
    // faer のグローバル並列度を Par::Seq に固定する（`crate::shared::parallelism`）。
    crate::shared::parallelism::ensure_serial();

    if !(confidence_level > 0.0 && confidence_level < 1.0) {
        return Err(CommonError::InvalidConfidenceLevel { confidence_level }.into());
    }
    if let GmmType::Iterated { max_iter, tol, .. } = &gmm_type {
        if *max_iter < GMM_MAX_ITER_MIN as usize || *max_iter as u64 > MAX_ITER_LIMIT as u64 {
            return Err(IvError::InvalidGmmMaxIter {
                max_iter: *max_iter as i64,
            });
        }
        if !(tol.is_finite() && *tol > 0.0) {
            return Err(IvError::InvalidGmmTol { gmm_tol: *tol });
        }
    }
    let n = input.nobs();
    // 観測数（`n <= k`はサンドイッチSE計算のNaN/Infinity、`n <= l`は`Z'Z`の特異または
    // `W₀ = (Z'Z)⁻¹`のGMMがOLSへ縮退するのを防ぐ）・`cov_type=Cluster`のクラスター数
    // （構造方程式の`G <= q`）・被説明変数の定数を、点推定・SE計算より前に弾く
    // （`TwoSlsEstimator::fit`・`engine_pybind`と同じ`validate_iv_fit_inputs`）。
    // `weight_type=Cluster`の重み行列`S`（l×l）の`G`と`l`の関係は別軸のため
    // `validate_weight_type`側で検証する。
    validate_iv_fit_inputs(&input, &cov_type)?;
    let k = input.k_exog() + input.k_endog();

    // 1段階GMMは`weight_type`を持たない（点推定・Hansen Jとも`(Z'Z)⁻¹`のみで計算する）
    // ため検証もしない。2段階・反復では点推定に使う重みの妥当性を検証する。
    let weight_type: Option<&WeightType> = match &gmm_type {
        GmmType::OneStep => None,
        GmmType::TwoStep { weight } | GmmType::Iterated { weight, .. } => Some(weight),
    };
    let l_instruments = input.k_exog() + input.instruments().ncols();
    // `weight_type=Cluster`のグループ（`validate_weight_type`が検証して返す）。`cov_type=Cluster`
    // のグループとは別の入力で、同じ列を指していても別々に渡される。
    let weight_cluster_codes = match weight_type {
        Some(weight_type) => validate_weight_type(weight_type, n, k, l_instruments)?,
        None => None,
    };

    // Z = x_exog ++ instruments（全操作変数、docs/spec/iv-spec.md 1.1節）。
    let z = hcat(input.x_exog(), input.instruments());
    let l = z.ncols();

    // X = x_exog ++ x_endog（実際の内生変数。2SLSの第二段階のようなX̂への
    // 置き換えはしない、`gmm.rs`のモジュールdoc参照）。
    let x = hcat(input.x_exog(), input.x_endog());

    let mut param_names = input.x_exog_names().to_vec();
    param_names.extend(input.x_endog_names().iter().cloned());

    let y = input.y();

    // 初期推定: W₀ = (Z'Z)⁻¹（weight_typeによらず共通、2SLSと同じ重み）。
    let ztz = z.transpose() * &z;
    let beta0 = gmm_point_estimate(&z, &x, y, &ztz)?;
    let residuals0 = y - &x * &beta0;

    // 初期推定の残差が丸め誤差の範囲で0なら完全フィット。重み行列`S`が0になって後段の逆行列が
    // 特異になる前に、原因が分かるエラーで止める（`TwoSlsEstimator::fit`と同じ判定）。
    let sst = total_sum_of_squares(&mat_column_to_vec(y, 0), None, input.has_intercept());
    let ssr0: f64 = (0..n).map(|i| (*residuals0.get(i, 0)).powi(2)).sum();
    if is_perfect_fit(ssr0, sst, k) {
        return Err(CommonError::PerfectFit.into());
    }

    // weight_type=ClassicalのHansen J検定専用の重み行列`σ̂²₀・Z'Z`（`gmm.rs`の
    // モジュールdoc「Hansen J過剰識別検定」参照。`σ̂²`によるスケーリングは点推定には
    // 不要だがHansen Jには必須）。`σ̂²₀`はstep-0残差から計算する
    // （`OneStep`ではstep-0残差＝最終残差のため、この一箇所の計算で両方の
    // ケースをカバーできる）。
    let sigma2_0: f64 = (0..n).map(|i| (*residuals0.get(i, 0)).powi(2)).sum::<f64>() / (n as f64);
    let classical_s = Mat::from_fn(l, l, |i, j| sigma2_0 * (*ztz.get(i, j)));

    // 方式（`gmm_type`）に応じた回数の再推定（詳細は`gmm_iterate`のdocコメント）。
    let GmmIteration {
        beta,
        residuals,
        s_used,
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
        beta0,
        residuals0,
        classical_s,
    )?;

    if !converged && raise_on_non_convergence {
        return Err(IvError::GmmNonConvergence {
            n_iter: n_iter as usize,
        });
    }

    let hansen_j = hansen_j_test(&input, &z, &residuals, &s_used)?;

    // ── ここからサンドイッチ型SE計算（`gmm.rs`のモジュールdoc
    // 「標準誤差・検定統計量（cov_type対応）」参照）──
    let k_constant = usize::from(input.has_intercept());
    let df_resid = n - k;
    let df_model = k - k_constant;
    let ssr: f64 = (0..n).map(|i| (*residuals.get(i, 0)).powi(2)).sum();

    let GmmCovariance {
        cov_params,
        hac_lags_used: cov_hac_lags,
    } = sandwich_covariance(&cov_type, &z, &x, &residuals, &ztz, &s_used, df_resid)?;
    // `cov_type=Hac`のラグ数は`weight_type=Hac`のラグ数より優先する。
    let hac_lags_used = cov_hac_lags.or(hac_lags_used);

    let normal_dist = Normal::new(0.0, 1.0).map_err(|e| CommonError::internal(e.to_string()))?;
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
        gmm_wald_chi2_test(&beta, &cov_params, k_constant, df_model)?
    };

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
    })
}

#[cfg(test)]
mod tests;
