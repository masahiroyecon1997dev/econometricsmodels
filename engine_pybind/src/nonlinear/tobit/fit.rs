//! Python境界の`fit`: `build_tobit_input`の後に`TobitEstimator::fit`を呼び、`TobitResult`を組み立てる。

use crate::nonlinear::common::{mle_error_to_pyerr, solver_name};
use crate::shared::column_extraction::SelectedData;
use crate::shared::mat_conversion::mat_to_rows;
use engine::nonlinear::common::MleFitOptions;
use engine::nonlinear::tobit::TobitEstimator;
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use super::input::build_tobit_input;
use super::options::TobitOptions;
use super::result::TobitResult;

/// Pythonから渡された `data` / `y` / `x` / `options` を検証し、
/// `build_tobit_input`で構築した`TobitInput`に対して`engine::nonlinear::tobit::
/// TobitEstimator::fit`を呼び出してTobitを推定し、`TobitResult`として返す。
///
/// # Errors
/// - `build_tobit_input`が返すエラー（列抽出・y/xの重複・`"const"`列衝突・
///   `cov_type`/`solver`文字列の検証・打ち切り境界の検証等）は`ValidationError`
/// - `TobitEstimator::fit`が返す`MleError`（`confidence_level`範囲外・`max_iter`が
///   0以下・観測数不足・非打ち切り観測ゼロ・未収束・特異Hessian・特異OPG行列・
///   特異設計行列・クラスターキー未指定・クラスター数不足等）は`mle_error_to_pyerr`で
///   変換（詳細は`engine::nonlinear::tobit::TobitEstimator::fit`のdocコメント参照）
pub(crate) fn fit(
    py: Python<'_>,
    data: SelectedData,
    y: String,
    x: Vec<String>,
    options: &TobitOptions,
) -> PyResult<TobitResult> {
    let SelectedData { source, frame } = data;
    // 取り込み済みの`frame`以降はPythonオブジェクトに触れないので、GILを手放して走らせる
    // （他のPythonスレッドを止めず、複数のfitをスレッドで並行できる）。結果の組み立てまで
    // 手放した側で行うため、`source`（`Py<PyAny>`は`Send`）もクロージャに渡す。
    py.detach(|| fit_frame(&frame, source, y, x, options))
}

/// [`fit`]の純Rust部分（GILを手放した状態で走る）。`PyErr`は遅延生成でGILを要求しない
/// ので、ここで`?`により返してよい。`source`は結果に載せるだけで、GILが要る操作はしない。
fn fit_frame(
    df: &DataFrame,
    source: Py<PyAny>,
    y: String,
    x: Vec<String>,
    options: &TobitOptions,
) -> PyResult<TobitResult> {
    let (input, cov_type, cov_type_lower, solver) = build_tobit_input(df, y, x, options)?;

    let estimator = TobitEstimator::fit(
        input,
        MleFitOptions {
            solver,
            max_iter: options.max_iter,
            tol: options.tol,
            raise_on_non_convergence: options.raise_on_non_convergence,
            cov_type,
            confidence_level: options.confidence_level,
        },
    )
    .map_err(mle_error_to_pyerr)?;

    // `params`/`param_names`を`(k+1)`長（`σ`を含む）に揃える
    // （`TobitResult`のdocコメント「非対称性の解消」参照）。
    let mut params = estimator.params().to_vec();
    params.push(estimator.sigma());
    let mut param_names = estimator.input().param_names().to_vec();
    param_names.push("sigma".to_string());

    Ok(TobitResult {
        params,
        std_errors: estimator.std_errors().to_vec(),
        cov_params: mat_to_rows(estimator.cov_params()),
        test_stats: estimator.test_stats().to_vec(),
        stat_dist: estimator.stat_dist().name().to_string(),
        stat_df: estimator.stat_dist().df().map(|df| df as i64),
        p_values: estimator.p_values().to_vec(),
        conf_lower: estimator.conf_lower().to_vec(),
        conf_upper: estimator.conf_upper().to_vec(),
        param_names,
        dep_var_name: estimator.input().dep_var_name().to_string(),
        sigma: estimator.sigma(),
        log_likelihood: estimator.log_likelihood(),
        aic: estimator.aic(),
        bic: estimator.bic(),
        n_obs: estimator.n_obs(),
        df_model: estimator.df_model(),
        df_resid: estimator.df_resid(),
        wald_statistic: estimator.wald_statistic(),
        wald_p_value: estimator.wald_p_value(),
        wald_dist: "chi2".to_string(),
        wald_df: estimator.wald_df(),
        converged: estimator.converged(),
        n_iter: estimator.n_iter(),
        cov_type: cov_type_lower,
        solver: solver_name(solver).to_string(),
        lower: options.lower,
        upper: options.upper,
        fitted: estimator.into_fitted(),
        source_data: source,
    })
}
