//! `OLSResult`（OLSの推定結果）と、その`#[pymethods]`（`predict`・`augment`・事後診断の入口）。

use crate::linear::common::predict_out_of_sample;
use crate::shared::augment::append_predicted_column;
use crate::shared::errors::{guard, user_error};
use engine::shared::error_catalog::{ErrorCode, ParamValue};
use pyo3::prelude::*;

use super::diagnostics::{
    BreuschGodfreyTestOutput, BreuschPaganTestOutput, WhiteTestOutput, run_breusch_godfrey_test,
    run_breusch_pagan_test, run_white_test,
};

/// Estimation results for OLS.
///
/// Structured data only (no `summary()`). All array-valued fields (`params`,
/// `std_errors`, etc.) share the same order as `param_names`.
// `OLSResult`はRust側で組み立ててPythonに返すだけの型で、Python側からの生成・引数として
// 受け取ることは想定していないため`skip_from_py_object`（`OLSOptions`の`from_py_object`とは
// 対照的。pyo3 0.28以降、Cloneを実装する#[pyclass]のFromPyObject自動導出はopt-inになった）。
//
// `get_all`（クラス単位で全フィールドに#[pyo3(get)]を付与する）ではなく、フィールドごとに
// 個別`#[pyo3(get)]`を付ける方式にしている。`fitted_values`（`predict(new_data=None)`が
// 返す値のキャッシュ、`docs/spec/ols-spec.md`「predict()」）はPython側に独立したプロパティとして
// 公開しない設計上の決定のため、この1フィールドだけ`#[pyo3(get)]`を付けずに残す必要がある。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug)]
pub struct OLSResult {
    #[pyo3(get)]
    pub params: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    /// Covariance matrix of the coefficients under `cov_type`, one row per entry of
    /// `param_names` (same order on both axes).
    #[pyo3(get)]
    pub cov_params: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub test_stats: Vec<f64>,
    /// Distribution of `test_stats`; always `"t"` for OLS.
    #[pyo3(get)]
    pub stat_dist: String,
    /// Degrees of freedom of the t distribution (always `Some` for OLS). May differ from
    /// `df_resid` (e.g. cluster-robust inference uses `G - 1`, where `G` is the number of
    /// clusters).
    #[pyo3(get)]
    pub stat_df: Option<i64>,
    #[pyo3(get)]
    pub p_values: Vec<f64>,
    #[pyo3(get)]
    pub conf_lower: Vec<f64>,
    #[pyo3(get)]
    pub conf_upper: Vec<f64>,
    #[pyo3(get)]
    pub param_names: Vec<String>,
    #[pyo3(get)]
    pub residuals: Vec<f64>,
    #[pyo3(get)]
    pub dep_var_name: String,
    #[pyo3(get)]
    pub n_obs: usize,
    /// Standard error type actually used (echoes `OLSOptions.cov_type`, normalized to
    /// lowercase; e.g. `"classical"`, `"hc1"`, `"hac"`, `"cluster"`).
    #[pyo3(get)]
    pub cov_type: String,
    /// Number of HAC (Newey-West) lags actually used: the explicit `hac_lags` if given,
    /// otherwise the value chosen automatically, `floor(4 * (n / 100) ^ (2 / 9))`.
    /// `None` unless `cov_type="hac"`.
    #[pyo3(get)]
    pub hac_lags_used: Option<i64>,
    #[pyo3(get)]
    pub r_squared: f64,
    #[pyo3(get)]
    pub adj_r_squared: f64,
    /// Overall F-test statistic of the slope coefficients. `None` when the test is not
    /// defined: there are no slope coefficients (intercept only), or, for a first-stage regression of
    /// `IVResults.first_stage()` with `cov_type="cluster"`, the number of clusters is not larger
    /// than the number of slope coefficients.
    #[pyo3(get)]
    pub f_statistic: Option<f64>,
    /// P-value of `f_statistic` (`None` exactly when `f_statistic` is `None`).
    #[pyo3(get)]
    pub f_p_value: Option<f64>,
    /// Numerator degrees of freedom of `f_statistic` (`None` when `f_statistic` is `None`).
    #[pyo3(get)]
    pub f_df_num: Option<usize>,
    /// Denominator degrees of freedom of `f_statistic` (`None` when `f_statistic` is `None`).
    /// Follows `cov_type` like the t-tests, so it equals `stat_df` whenever `f_statistic` is
    /// not `None`: the number of clusters minus 1 for `"cluster"` and `df_resid` otherwise.
    /// Use it to recompute the p-value from `f_statistic`.
    #[pyo3(get)]
    pub f_df_denom: Option<usize>,
    /// Residual degrees of freedom (`n - k`).
    #[pyo3(get)]
    pub df_resid: usize,
    /// Model degrees of freedom (number of slope coefficients, excluding the intercept).
    #[pyo3(get)]
    pub df_model: usize,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    /// Fitted values for the training data (`ŷ = Xβ̂`), cached at fit time.
    /// Not exposed to Python directly; only `predict(new_data=None)` reads it (there is
    /// no separate `fitted_values` property).
    pub(super) fitted_values: Vec<f64>,
    /// Whether `fit()` was called with `include_intercept=True`. Not exposed to
    /// Python; only `predict()` reads it to decide whether to auto-prepend a
    /// constant column for out-of-sample data.
    ///
    /// This must NOT be inferred from `param_names[0] == "const"`: when
    /// `include_intercept=False`, a user-supplied `x` column may legitimately be
    /// named `"const"` (the collision check in `fit()` only rejects that name when
    /// `include_intercept=True`), which would make such an inference silently
    /// wrong instead of erroring.
    pub(super) has_intercept: bool,
    /// The `polars.DataFrame` passed to `fit()`, as given. `augment(new_data=None)` appends
    /// the predicted column to it (so every column is kept), and the diagnostic tests read the
    /// columns they need from it on demand.
    ///
    /// `None` for `OLSResult`s built by `ols_estimator_to_result` without going
    /// through this file's `fit()` (currently only `IVResult.first_stage()`,
    /// `engine_pybind/src/iv/common.rs`): those per-equation regressions have no
    /// single source DataFrame, so `augment(new_data=None)` and the diagnostic tests
    /// on such a result raise `ValidationError` instead.
    pub(super) source_data: Option<Py<PyAny>>,
}

crate::shared::pickle::impl_pickle_state!(OLSResult {
    params,
    std_errors,
    cov_params,
    test_stats,
    stat_dist,
    stat_df,
    p_values,
    conf_lower,
    conf_upper,
    param_names,
    residuals,
    dep_var_name,
    n_obs,
    cov_type,
    hac_lags_used,
    r_squared,
    adj_r_squared,
    f_statistic,
    f_p_value,
    f_df_num,
    f_df_denom,
    df_resid,
    df_model,
    log_likelihood,
    aic,
    bic,
    fitted_values,
    has_intercept,
    source_data,
});

impl OLSResult {
    /// 全フィールドの複製（`source_data`は同じDataFrameへの参照を増やすだけで、データはコピーしない）。
    /// `Py<PyAny>`は`Clone`を実装しない（参照カウントの増加にGILが要る）ため`#[derive(Clone)]`ではなく、
    /// `IVResult`が保持する第一段階の結果から`first_stage()`の戻り値を作るのに使う。
    /// フィールドは分割代入で網羅するので、足し忘れはコンパイルエラーになる。
    pub(crate) fn clone_ref(&self, py: Python<'_>) -> Self {
        let Self {
            params,
            std_errors,
            cov_params,
            test_stats,
            stat_dist,
            stat_df,
            p_values,
            conf_lower,
            conf_upper,
            param_names,
            residuals,
            dep_var_name,
            n_obs,
            cov_type,
            hac_lags_used,
            r_squared,
            adj_r_squared,
            f_statistic,
            f_p_value,
            f_df_num,
            f_df_denom,
            df_resid,
            df_model,
            log_likelihood,
            aic,
            bic,
            fitted_values,
            has_intercept,
            source_data,
        } = self;
        Self {
            params: params.clone(),
            std_errors: std_errors.clone(),
            cov_params: cov_params.clone(),
            test_stats: test_stats.clone(),
            stat_dist: stat_dist.clone(),
            stat_df: *stat_df,
            p_values: p_values.clone(),
            conf_lower: conf_lower.clone(),
            conf_upper: conf_upper.clone(),
            param_names: param_names.clone(),
            residuals: residuals.clone(),
            dep_var_name: dep_var_name.clone(),
            n_obs: *n_obs,
            cov_type: cov_type.clone(),
            hac_lags_used: *hac_lags_used,
            r_squared: *r_squared,
            adj_r_squared: *adj_r_squared,
            f_statistic: *f_statistic,
            f_p_value: *f_p_value,
            f_df_num: *f_df_num,
            f_df_denom: *f_df_denom,
            df_resid: *df_resid,
            df_model: *df_model,
            log_likelihood: *log_likelihood,
            aic: *aic,
            bic: *bic,
            fitted_values: fitted_values.clone(),
            has_intercept: *has_intercept,
            source_data: source_data.as_ref().map(|data| data.clone_ref(py)),
        }
    }
}

// `#[pymethods]`ブロックの外に置く非公開実装（pyo3は`#[pymethods]`内の全メソッドを
// Python公開シグネチャとして扱おうとするため、Pythonへ公開しないヘルパーはこちらに置く）。
impl OLSResult {
    /// 元のデータ（`fit()`に渡された`polars.DataFrame`）。保持していない結果（`IVResult.first_stage()`）は
    /// `TRAINING_DATA_NOT_RETAINED`。
    pub(super) fn retained_source<'py>(
        &self,
        py: Python<'py>,
        method: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.source_data
            .as_ref()
            .map(|source| source.bind(py).clone())
            .ok_or_else(|| {
                user_error(
                    ErrorCode::TrainingDataNotRetained,
                    vec![("method", ParamValue::from(method))],
                )
            })
    }
}

#[pymethods]
impl OLSResult {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        crate::shared::pickle::reduce(slf)
    }

    /// Predicted values.
    ///
    /// With `new_data=None` (default), returns the fitted values for the training
    /// data used in `fit()`. With `new_data` given, computes out-of-sample
    /// predictions for a new polars DataFrame: it must contain columns with the same
    /// names as the `x` columns passed at fit time (matched by name; column order
    /// does not matter). If `include_intercept=True` was used at fit time, the
    /// constant column is added automatically and must not be included in `new_data`.
    ///
    /// # Errors
    /// - A required `x` column is missing from `new_data`, cannot be cast to a
    ///   numeric type, or contains missing/NaN/infinite values: `ValidationError`
    ///   (same validation as `fit()`'s column extraction, via `extract_f64_column`).
    #[pyo3(signature = (new_data=None))]
    fn predict(&self, new_data: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f64>> {
        guard(|| {
            let Some(new_data) = new_data else {
                return Ok(self.fitted_values.clone());
            };

            predict_out_of_sample(
                new_data,
                &self.param_names,
                &self.params,
                self.has_intercept,
            )
        })
    }

    /// The source data (training data, or `new_data` when given) with the
    /// predicted values appended as a new `"predicted"` column.
    ///
    /// Same `new_data`/`include_intercept` semantics as `predict()`, but returns
    /// a polars DataFrame (original columns plus `"predicted"`, row order
    /// preserved) instead of a bare list of floats.
    ///
    /// # Errors
    /// - Same as `predict()`: a required `x` column missing from `new_data`,
    ///   non-numeric, or containing missing/NaN/infinite values: `ValidationError`.
    /// - The source data already has a column named `"predicted"`:
    ///   `ValidationError` (would otherwise silently overwrite it).
    /// - `new_data=None` and this result has no retained source data (currently
    ///   only possible for `IVResult.first_stage()` results): `ValidationError`.
    #[pyo3(signature = (new_data=None))]
    fn augment(&self, py: Python<'_>, new_data: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        guard(|| {
            let (source, predicted) = match new_data {
                Some(new_data) => {
                    let predicted = predict_out_of_sample(
                        new_data,
                        &self.param_names,
                        &self.params,
                        self.has_intercept,
                    )?;
                    (new_data.clone(), predicted)
                }
                None => {
                    let source = self
                        .source_data
                        .as_ref()
                        .ok_or_else(|| user_error(ErrorCode::AugmentRequiresNewData, Vec::new()))?;
                    (source.bind(py).clone(), self.fitted_values.clone())
                }
            };

            append_predicted_column(&source, "predicted", predicted)
        })
    }

    /// White test for heteroskedasticity.
    ///
    /// Regresses the squared residuals on the independent variables, their squares and
    /// their pairwise products (always with a constant, even when the model was fitted
    /// with `include_intercept=False`) and returns either the LM version
    /// (`statistic="lm"`, `n * R^2`, chi-squared) or the F version (`statistic="f"`).
    /// Terms that are constant or numerically identical to an earlier term (for example
    /// the square of a 0/1 dummy) are dropped, and the degrees of freedom count the terms
    /// that remain. The test does not depend on `cov_type`.
    ///
    /// # Errors
    /// - `statistic` is not `"lm"` or `"f"`: `ValidationError`.
    /// - This result has no retained source data (currently only `IVResult.first_stage()`
    ///   results): `ValidationError`.
    /// - Every regressor is constant: `ValidationError`.
    /// - Too few observations for the auxiliary regression: `ValidationError`.
    /// - The auxiliary design matrix is still singular after dropping, or its R-squared
    ///   is undefined: `ComputationError`.
    #[pyo3(signature = (statistic="lm"))]
    fn white_test(&self, py: Python<'_>, statistic: &str) -> PyResult<WhiteTestOutput> {
        guard(|| run_white_test(self, py, statistic))
    }

    /// Breusch-Pagan test for heteroskedasticity (Koenker's studentized version).
    ///
    /// Regresses the squared residuals on a constant and the columns named in `variables`
    /// (always with a constant, even when the model was fitted with `include_intercept=False`)
    /// and returns either the LM version (`statistic="lm"`, `n * R^2`, chi-squared) or the
    /// F version (`statistic="f"`). It does not assume normal errors.
    ///
    /// `variables` is a list of column names of the data passed to `fit()`. It need not be
    /// the model's independent variables: any numeric column may be used, including one
    /// that is not in the model. When it is `None`, the independent variables of the model
    /// are used. Variables that are constant or numerically identical to an earlier one
    /// (for example a copy of a dummy) are dropped, and the degrees of freedom count the
    /// variables that remain. The test does not depend on `cov_type`.
    ///
    /// # Errors
    /// - `variables` is not a `list` of `str`: `TypeError`. `statistic` is not a `str`:
    ///   `TypeError`.
    /// - `statistic` is not `"lm"` or `"f"`, `variables` is empty or names a column twice, a
    ///   column does not exist, has an unsupported dtype or contains missing values, NaN or
    ///   infinity, every variable is constant, or too few observations for the auxiliary
    ///   regression: `ValidationError`.
    /// - This result has no retained source data (currently only `IVResult.first_stage()`
    ///   results): `ValidationError`.
    /// - The auxiliary design matrix is still singular after dropping, or its R-squared
    ///   is undefined: `ComputationError`.
    #[pyo3(signature = (variables=None, statistic="lm"))]
    fn breusch_pagan_test(
        &self,
        py: Python<'_>,
        variables: Option<&Bound<'_, PyAny>>,
        statistic: &str,
    ) -> PyResult<BreuschPaganTestOutput> {
        guard(|| run_breusch_pagan_test(self, py, variables, statistic))
    }

    /// Breusch-Godfrey test for serial correlation of the errors.
    ///
    /// Regresses the residuals on the model's independent variables and their own lags
    /// 1 to `nlags` (in the order given by the `time` column; lags before the first
    /// observation are 0), and tests that the lag coefficients are zero. The auxiliary
    /// regression uses the model's regressors as they are: no constant is added to a
    /// model fitted with `include_intercept=False`. Returns the LM version
    /// (`statistic="lm"`, `n * R^2`, chi-squared) or the F version (`statistic="f"`).
    /// The test does not depend on `cov_type`.
    ///
    /// `time` names a column of the data passed to `fit()`; the row order is never
    /// assumed to be the time order. The values only give the order (no gaps are
    /// checked): lags are taken in that order.
    ///
    /// # Errors
    /// - `time` or `statistic` is not a `str`, or `nlags` is not an `int` (`bool` and `float`
    ///   included): `TypeError`.
    /// - `statistic` is not `"lm"` or `"f"`, `nlags < 1`, or too few observations for the
    ///   auxiliary regression (`n <= k + nlags`): `ValidationError`.
    /// - `time` does not exist, has an unsupported dtype, or contains missing values, NaN,
    ///   infinity or duplicates: `ValidationError`.
    /// - This result has no retained source data (currently only `IVResult.first_stage()`
    ///   results): `ValidationError`.
    /// - The auxiliary design matrix is singular, or the residuals are all zero / fitted
    ///   exactly by the auxiliary regression: `ComputationError`.
    #[pyo3(signature = (time, nlags, statistic=None))]
    fn breusch_godfrey_test(
        &self,
        py: Python<'_>,
        time: &Bound<'_, PyAny>,
        nlags: &Bound<'_, PyAny>,
        statistic: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<BreuschGodfreyTestOutput> {
        guard(|| run_breusch_godfrey_test(self, py, time, nlags, statistic))
    }
}
