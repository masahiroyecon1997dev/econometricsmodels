//! Python境界の`fit`: `build_iv_input`の後にエンジンを呼び、`IVResult`を組み立てる。

use crate::linear::ols::{OLSResult, ols_estimator_to_result};
use crate::shared::column_extraction::SelectedData;
use crate::shared::mat_conversion::{mat_to_rows, mat_to_vec};
use engine::iv::common::{IvEstimate, compute_first_stage, validate_iv_fit_inputs};
use engine::iv::gmm::GmmEstimator;
use engine::iv::two_sls::TwoSlsEstimator;
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use std::collections::HashMap;

use super::error::iv_error_to_pyerr;
use super::input::{BuiltIvInput, build_iv_input};
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
///   `ValidationError`（`validate_iv_fit_inputs`、`compute_first_stage`より前）
/// - `cov_type="cluster"`でクラスター数が構造方程式の傾き係数の数以下（または2未満）:
///   `ValidationError`（同上）
/// - 切片があり`y`が全行で同じ値: `ValidationError`（同上）
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
    let BuiltIvInput {
        input,
        cov_type,
        cov_type_name,
        plan,
    } = build_iv_input(df, y, x_exog, x_endog, instruments, options)?;

    // 識別の順序条件（過小識別な入力で無駄な第一段階回帰を走らせない。この時点で
    // `build_iv_input`の`validate_role_non_empty("x_endog"/"instruments", ...)`を通過しており
    // `k_endog`/`k_instruments`はともに1以上なので、「両方指定されているが数が足りない」
    // ケースのみを扱う）・観測数`n`（構造方程式の`n <= k`を先に、次に第一段階の`n <= l`）・`cov_type=Cluster`の
    // 構造方程式の`G <= q`・被説明変数が定数、のチェック（`TwoSlsEstimator::fit`/
    // `GmmEstimator::fit`冒頭と同じ実装）を`compute_first_stage`より先に行う。第一段階の
    // 回帰変数・傾き係数の数は識別条件により構造方程式以上のため、先に第一段階を走らせると
    // 構造方程式側のチェックが`FirstStageFailed`（第一段階の`k`・`g`・`q`）に隠れ、
    // エンジン側のチェックがPythonから到達不能になる。
    validate_iv_fit_inputs(&input, &cov_type).map_err(iv_error_to_pyerr)?;

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
            let result = ols_estimator_to_result(&estimator, cov_type_name.clone());
            (name, result)
        })
        .collect();

    let shared = SharedFit {
        cov_type: cov_type_name,
        estimator: plan.estimator.name().to_string(),
        weak_instrument_f_statistics: weak_instrument_f_statistics.into_iter().collect(),
        weak_instrument_f_df,
        first_stage,
    };

    if let Some(gmm) = plan.gmm {
        let estimator = GmmEstimator::fit(
            input,
            gmm.gmm_type,
            options.raise_on_non_convergence.unwrap_or(true),
            cov_type,
            options.confidence_level,
        )
        .map_err(iv_error_to_pyerr)?;

        let specific = EstimatorSpecific {
            converged: estimator.converged(),
            n_iter: estimator.n_iter(),
            gmm_weight_type: gmm.weight_type_name,
            gmm_type: Some(gmm.gmm_type_name),
            wald_dist: "chi2",
            wald_df_num: estimator.wald_df(),
            wald_df_denom: None,
            overid_statistic: estimator.hansen_j_statistic(),
            overid_p_value: estimator.hansen_j_p_value(),
            overid_df: estimator.hansen_j_df(),
            wu_hausman_statistic: None,
            wu_hausman_p_value: None,
            wu_hausman_df_num: None,
            wu_hausman_df_denom: None,
        };
        return Ok(build_result(&estimator, shared, specific));
    }

    let estimator = TwoSlsEstimator::fit(input, cov_type, options.confidence_level)
        .map_err(iv_error_to_pyerr)?;

    let specific = EstimatorSpecific {
        // 2SLSは閉形式・非反復のため常に`converged=true`・`n_iter=1`
        // （`IVResult.converged`のdocコメント参照）。
        converged: true,
        n_iter: 1,
        // `gmm_weight_type`/`gmm_type`はGMM専用の概念のため`estimator="2sls"`では常に`None`
        // （`IVResult.gmm_weight_type`のdocコメント参照）。
        gmm_weight_type: None,
        gmm_type: None,
        wald_dist: "f",
        wald_df_num: estimator.wald_df().map(|(num, _)| num),
        wald_df_denom: estimator.wald_df().map(|(_, denom)| denom),
        overid_statistic: estimator.sargan_statistic(),
        overid_p_value: estimator.sargan_p_value(),
        overid_df: estimator.sargan_df(),
        wu_hausman_statistic: estimator.wu_hausman_statistic(),
        wu_hausman_p_value: estimator.wu_hausman_p_value(),
        wu_hausman_df_num: estimator.wu_hausman_df().map(|(num, _)| num),
        wu_hausman_df_denom: estimator.wu_hausman_df().map(|(_, denom)| denom),
    };
    Ok(build_result(&estimator, shared, specific))
}

/// `estimator`によらず同じ手順で決まる`IVResult`のフィールドのうち、推定量の読み出し口
/// （[`IvEstimate`]）から読まないもの。
struct SharedFit {
    cov_type: String,
    estimator: String,
    weak_instrument_f_statistics: HashMap<String, f64>,
    weak_instrument_f_df: Option<(usize, usize)>,
    first_stage: Vec<(String, OLSResult)>,
}

/// `IVResult`のフィールドのうち、`estimator`によって値の出どころや意味が変わるもの
/// （収束・反復、GMM専用の設定、Wald検定の分布、過剰識別検定、Wu-Hausman検定）。
///
/// `Default`は実装しない。`2sls`と`gmm`の両方の分岐で全フィールドを明示することで、
/// 新しいフィールドを足したときに片方の分岐だけ書き忘れるとビルドエラーになる。
struct EstimatorSpecific {
    converged: bool,
    n_iter: i64,
    gmm_weight_type: Option<String>,
    gmm_type: Option<String>,
    wald_dist: &'static str,
    wald_df_num: Option<usize>,
    wald_df_denom: Option<usize>,
    overid_statistic: Option<f64>,
    overid_p_value: Option<f64>,
    overid_df: Option<usize>,
    wu_hausman_statistic: Option<f64>,
    wu_hausman_p_value: Option<f64>,
    wu_hausman_df_num: Option<usize>,
    wu_hausman_df_denom: Option<usize>,
}

/// `IVResult`を組み立てる。推定量の読み出し口（[`IvEstimate`]）から読める共通のフィールドは
/// ここで一度だけ詰め、2SLSとGMMで食い違わないようにする。
fn build_result<E: IvEstimate>(
    estimator: &E,
    shared: SharedFit,
    specific: EstimatorSpecific,
) -> IVResult {
    IVResult {
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
        converged: specific.converged,
        n_iter: specific.n_iter,
        cov_type: shared.cov_type,
        hac_lags_used: estimator.hac_lags_used().map(|lags| lags as i64),
        estimator: shared.estimator,
        gmm_weight_type: specific.gmm_weight_type,
        gmm_type: specific.gmm_type,
        wald_statistic: estimator.wald_statistic(),
        wald_p_value: estimator.wald_p_value(),
        wald_dist: specific.wald_dist.to_string(),
        wald_df_num: specific.wald_df_num,
        wald_df_denom: specific.wald_df_denom,
        r_squared: estimator.r_squared(),
        adj_r_squared: estimator.adj_r_squared(),
        weak_instrument_f_statistics: shared.weak_instrument_f_statistics,
        overid_statistic: specific.overid_statistic,
        overid_p_value: specific.overid_p_value,
        overid_df: specific.overid_df,
        wu_hausman_statistic: specific.wu_hausman_statistic,
        wu_hausman_p_value: specific.wu_hausman_p_value,
        wu_hausman_df_num: specific.wu_hausman_df_num,
        wu_hausman_df_denom: specific.wu_hausman_df_denom,
        weak_instrument_f_df_num: shared.weak_instrument_f_df.map(|(num, _)| num),
        weak_instrument_f_df_denom: shared.weak_instrument_f_df.map(|(_, denom)| denom),
        first_stage: shared.first_stage,
    }
}
