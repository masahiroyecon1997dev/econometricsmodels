//! Python境界の`fit`: `build_iv_input`の後にエンジンを呼び、`IVResult`を組み立てる。

use crate::linear::ols::{OLSResult, ols_estimator_to_result};
use crate::shared::column_extraction::SelectedData;
use crate::shared::mat_conversion::{mat_to_rows, mat_to_vec};
use engine::iv::common::{
    IvError, compute_first_stage, validate_observation_counts, validate_outcome_not_constant,
    validate_structural_cluster_count,
};
use engine::iv::gmm::GmmEstimator;
use engine::iv::two_sls::TwoSlsEstimator;
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use super::error::iv_error_to_pyerr;
use super::input::build_iv_input;
use super::options::IVOptions;
use super::result::IVResult;

/// Pythonから渡された `data` / `y` / `x_exog` / `x_endog` / `instruments` / `options` を
/// 検証し、`build_iv_input`で構築した`IvInput`に対して`estimator`に応じた推定
/// （`TwoSlsEstimator::fit`または`GmmEstimator::fit`）を呼び出し、`IVResult`として返す。
///
/// `first_stage`/`weak_instrument_f_statistics`は`estimator`によらず`engine::iv::common::
/// compute_first_stage`（`IVResult`のdocコメント・モジュールdocコメント「`first_stage()`/
/// `weak_instrument_f_statistics`は`estimator`に依存しない共通ロジック」参照）から構築する。
/// `overid_statistic`/`overid_p_value`は`estimator="2sls"`では`TwoSlsEstimator::
/// sargan_statistic()`/`sargan_p_value()`（Sargan検定）、`estimator="gmm"`では
/// `GmmEstimator::hansen_j_statistic()`/`hansen_j_p_value()`（Hansen J検定）から構築する。
/// `wu_hausman_statistic`/`wu_hausman_p_value`は`estimator="2sls"`では`TwoSlsEstimator::
/// wu_hausman_statistic()`/`wu_hausman_p_value()`、`estimator="gmm"`では
/// 常に`None`（`GmmEstimator`は実装しない、モジュールdocコメント参照）。
///
/// # Errors
/// - 観測数が構造方程式の係数の数以下、または第一段階の回帰変数の数以下:
///   `ValidationError`（`validate_observation_counts`、`compute_first_stage`より前）
/// - `cov_type="cluster"`でクラスター数が構造方程式の傾き係数の数以下（または2未満）:
///   `ValidationError`（`validate_structural_cluster_count`、`compute_first_stage`より前）
/// - 切片があり`y`が全行で同じ値: `ValidationError`（`validate_outcome_not_constant`、
///   `compute_first_stage`より前）
/// - `build_iv_input`が返すエラー（列抽出・y/x_exog/x_endog/instrumentsの重複・
///   `"const"`列衝突・`estimator`/`cov_type`文字列の検証等）は`ValidationError`
/// - `gmm_type`/`gmm_weight_type`の文字列が不正（使われないオプションでも）、または使われない
///   オプションの指定: `ValidationError`（`build_iv_input`が`parse_iv_option_values`/
///   `check_iv_options`で報告する）
/// - `TwoSlsEstimator::fit`/`GmmEstimator::fit`/`compute_first_stage`が返す
///   `engine::iv::common::IvError`（識別の順序条件・第一段階回帰の失敗・`cov_type`起因の
///   エラー・GMM固有のエラー等）は`iv_error_to_pyerr`で変換
pub(crate) fn fit(
    py: Python<'_>,
    data: SelectedData,
    y: String,
    x_exog: Vec<String>,
    x_endog: Vec<String>,
    instruments: Vec<String>,
    options: &IVOptions,
) -> PyResult<IVResult> {
    // 取り込み済みの`frame`以降はPythonオブジェクトに触れないので、GILを手放して走らせる
    // （他のPythonスレッドを止めず、複数のfitをスレッドで並行できる）。`source`は使わず、
    // 関数の末尾（`detach`から戻ってGILを持った状態）で`data`ごとdropされる。
    let SelectedData { frame, .. } = data;
    py.detach(|| fit_frame(&frame, y, x_exog, x_endog, instruments, options))
}

/// [`fit`]の純Rust部分（GILを手放した状態で走る）。`PyErr`は遅延生成でGILを要求しない
/// ので、ここで`?`により返してよい。
fn fit_frame(
    df: &DataFrame,
    y: String,
    x_exog: Vec<String>,
    x_endog: Vec<String>,
    instruments: Vec<String>,
    options: &IVOptions,
) -> PyResult<IVResult> {
    let (input, cov_type, cov_type_lower, plan) =
        build_iv_input(df, y, x_exog, x_endog, instruments, options)?;

    // 識別の順序条件（`TwoSlsEstimator::fit`/`GmmEstimator::fit`のいずれも冒頭で検証する
    // のと同じチェック）を`compute_first_stage`より先に行う。過小識別な入力で無駄な
    // 第一段階回帰を走らせないため（`compute_first_stage`自体はこの条件を検証しない
    // ので呼び出し元の責務）。この時点で`build_iv_input`の
    // `validate_x_non_empty("x_endog"/"instruments", ...)`を既に通過して
    // いるため`k_endog`/`k_instruments`はともに1以上であり、ここでの`<`判定は「両方
    // 指定されているが数が足りない」過小識別ケースのみを扱う（「そもそも変数が
    // 指定されていない」退化ケースとは排他的）。
    if input.k_instruments() < input.k_endog() {
        return Err(iv_error_to_pyerr(IvError::InsufficientInstruments {
            n_instruments: input.k_instruments(),
            n_endog: input.k_endog(),
        }));
    }

    // 観測数`n`のチェック（構造方程式の`n <= k`を先に、次に第一段階の`n <= l`）を
    // `compute_first_stage`より先に行う。第一段階の回帰変数の数は識別条件により構造方程式
    // 以上のため、先に第一段階を走らせると構造方程式の`n <= k`が第一段階の
    // `FirstStageFailed`（第一段階の`k`）に隠れて報告されてしまう。
    validate_observation_counts(&input).map_err(iv_error_to_pyerr)?;

    // `cov_type=Cluster`の構造方程式の`G <= q`チェック（`TwoSlsEstimator::fit`/
    // `GmmEstimator::fit`冒頭と同じ実装）を`compute_first_stage`より先に行う。第一段階の
    // 傾き係数の数は識別条件により構造方程式以上のため、先に第一段階を走らせると
    // このチェックが常に`FirstStageFailed`（第一段階の`g`・`q`）に隠れ、エンジン側の
    // 構造方程式チェックがPythonから到達不能になる。
    validate_structural_cluster_count(&input, &cov_type).map_err(iv_error_to_pyerr)?;

    // 被説明変数が定数（切片あり）のモデルは説明するものが無い。入力だけから分かるため、これも
    // `compute_first_stage`より先に弾く（`TwoSlsEstimator::fit`/`GmmEstimator::fit`冒頭と同じ実装）。
    validate_outcome_not_constant(&input).map_err(|e| iv_error_to_pyerr(IvError::from(e)))?;

    // 第一段階回帰・弱操作変数診断は`estimator`によらず共通（モジュールdocコメント参照）。
    // `input`は下でestimatorごとの推定器に移動するため、参照のみで済むこの呼び出しを先に行う。
    let (first_stage, weak_instrument_f_statistics) =
        compute_first_stage(&input, &cov_type, options.confidence_level)
            .map_err(iv_error_to_pyerr)?;
    let weak_instrument_f_df = first_stage.first().map(|(_, first_stage_estimator)| {
        (input.k_instruments(), first_stage_estimator.df_resid())
    });
    // 第一段階の推定量（設計行列を持つ）は結果に変えて手放す。`IVResult`が持つのは結果だけで、
    // `first_stage()`の戻り値にも、pickleの状態にもそのまま使える。
    let first_stage: Vec<(String, OLSResult)> = first_stage
        .into_iter()
        .map(|(name, estimator)| {
            let result = ols_estimator_to_result(&estimator, cov_type_lower.clone());
            (name, result)
        })
        .collect();

    if let Some(gmm) = plan.gmm {
        let estimator = GmmEstimator::fit(
            input,
            gmm.gmm_type,
            options.raise_on_non_convergence.unwrap_or(true),
            cov_type,
            options.confidence_level,
        )
        .map_err(iv_error_to_pyerr)?;

        return Ok(IVResult {
            params: mat_to_vec(estimator.params()),
            std_errors: mat_to_vec(estimator.std_errors()),
            cov_params: mat_to_rows(estimator.cov_params()),
            test_stats: mat_to_vec(estimator.test_stats()),
            stat_dist: estimator.stat_dist().name().to_string(),
            stat_df: estimator.stat_dist().df().map(|df| df as i64),
            p_values: mat_to_vec(estimator.p_values()),
            conf_lower: mat_to_vec(estimator.conf_lower()),
            conf_upper: mat_to_vec(estimator.conf_upper()),
            param_names: estimator.param_names().to_vec(),
            residuals: mat_to_vec(estimator.residuals()),
            dep_var_name: estimator.dep_var_name().to_string(),
            n_obs: estimator.nobs(),
            df_resid: estimator.df_resid(),
            df_model: estimator.df_model(),
            converged: estimator.converged(),
            n_iter: estimator.n_iter(),
            cov_type: cov_type_lower,
            hac_lags_used: estimator.hac_lags_used().map(|lags| lags as i64),
            estimator: plan.estimator.name().to_string(),
            gmm_weight_type: gmm.weight_type_name,
            gmm_type: Some(gmm.gmm_type_name),
            wald_statistic: estimator.wald_statistic(),
            wald_p_value: estimator.wald_p_value(),
            wald_dist: "chi2".to_string(),
            wald_df_num: estimator.wald_df(),
            wald_df_denom: None,
            r_squared: estimator.r_squared(),
            adj_r_squared: estimator.adj_r_squared(),
            weak_instrument_f_statistics: weak_instrument_f_statistics.into_iter().collect(),
            overid_statistic: estimator.hansen_j_statistic(),
            overid_p_value: estimator.hansen_j_p_value(),
            overid_df: estimator.hansen_j_df(),
            wu_hausman_statistic: None,
            wu_hausman_p_value: None,
            wu_hausman_df_num: None,
            wu_hausman_df_denom: None,
            weak_instrument_f_df_num: weak_instrument_f_df.map(|(num, _)| num),
            weak_instrument_f_df_denom: weak_instrument_f_df.map(|(_, denom)| denom),
            first_stage,
        });
    }

    let estimator = TwoSlsEstimator::fit(input, cov_type, options.confidence_level)
        .map_err(iv_error_to_pyerr)?;

    Ok(IVResult {
        params: mat_to_vec(estimator.params()),
        std_errors: mat_to_vec(estimator.std_errors()),
        cov_params: mat_to_rows(estimator.cov_params()),
        test_stats: mat_to_vec(estimator.test_stats()),
        stat_dist: estimator.stat_dist().name().to_string(),
        stat_df: estimator.stat_dist().df().map(|df| df as i64),
        p_values: mat_to_vec(estimator.p_values()),
        conf_lower: mat_to_vec(estimator.conf_lower()),
        conf_upper: mat_to_vec(estimator.conf_upper()),
        param_names: estimator.param_names().to_vec(),
        residuals: mat_to_vec(estimator.residuals()),
        dep_var_name: estimator.dep_var_name().to_string(),
        n_obs: estimator.nobs(),
        df_resid: estimator.df_resid(),
        df_model: estimator.df_model(),
        // 2SLSは閉形式・非反復のため常に`converged=true`・`n_iter=1`
        // （`IVResult.converged`のdocコメント参照）。
        converged: true,
        n_iter: 1,
        cov_type: cov_type_lower,
        hac_lags_used: estimator.hac_lags_used().map(|lags| lags as i64),
        estimator: plan.estimator.name().to_string(),
        // `gmm_weight_type`はGMM専用の概念のため`estimator="2sls"`では常に`None`
        // （`IVResult.gmm_weight_type`のdocコメント参照）。
        gmm_weight_type: None,
        gmm_type: None,
        wald_statistic: estimator.wald_statistic(),
        wald_p_value: estimator.wald_p_value(),
        wald_dist: "f".to_string(),
        wald_df_num: estimator.wald_df().map(|(num, _)| num),
        wald_df_denom: estimator.wald_df().map(|(_, denom)| denom),
        r_squared: estimator.r_squared(),
        adj_r_squared: estimator.adj_r_squared(),
        weak_instrument_f_statistics: weak_instrument_f_statistics.into_iter().collect(),
        overid_statistic: estimator.sargan_statistic(),
        overid_p_value: estimator.sargan_p_value(),
        overid_df: estimator.sargan_df(),
        wu_hausman_statistic: estimator.wu_hausman_statistic(),
        wu_hausman_p_value: estimator.wu_hausman_p_value(),
        wu_hausman_df_num: estimator.wu_hausman_df().map(|(num, _)| num),
        wu_hausman_df_denom: estimator.wu_hausman_df().map(|(_, denom)| denom),
        weak_instrument_f_df_num: weak_instrument_f_df.map(|(num, _)| num),
        weak_instrument_f_df_denom: weak_instrument_f_df.map(|(_, denom)| denom),
        first_stage,
    })
}
