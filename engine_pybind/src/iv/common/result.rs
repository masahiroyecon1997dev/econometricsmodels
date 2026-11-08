//! `IVResult`（IVの推定結果）と、その`#[pymethods]`（`first_stage`など）。

use crate::linear::ols::OLSResult;
use crate::shared::errors::{guard, internal_error};
use crate::shared::option_values::extract_strict_float;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use std::collections::HashMap;

use super::diagnostics::{
    CraggDonaldOutput, EffectiveFOutput, KleibergenPaapOutput, run_cragg_donald, run_effective_f,
    run_kleibergen_paap,
};
use super::options::IVOptions;

/// Estimation results for IV (2SLS/GMM).
///
/// Structured data only (no `summary()`). All array-valued fields (`params`,
/// `std_errors`, etc.) share the same order as `param_names`.
///
/// `test_stats` holds the t-statistics (`estimator="2sls"`) or z-statistics
/// (`estimator="gmm"`); `stat_dist`/`stat_df` say which distribution (and, for
/// `"t"`, how many degrees of freedom) that is.
///
/// The first-stage regressions are not a field: `first_stage()` is a separate method.
/// `weak_instrument_f_statistics` and `first_stage()` are computed the same way for
/// both estimators. `wu_hausman_*` is available for `estimator="2sls"` only, and
/// `overid_*` is the Sargan test (`estimator="2sls"`) or the Hansen J test
/// (`estimator="gmm"`).
// `first_stage()`（内生変数ごとの第一段階回帰結果）はフィールドに含めず別メソッドとして公開する。
// `LogitResult`/`ProbitResult`が`predict()`用に推定量そのものを非公開フィールドとして保持する
// のと同じパターンだが、`IVResult`は`estimator`（2sls/gmm）非依存の非公開フィールド
// `first_stage: Vec<(String, OLSResult)>`から構築する。`engine::iv::common::compute_first_stage`が
// `estimator`によらず同じ第一段階回帰を計算するため、`fit`でこの表現に詰め替えるだけで済む。
// `weak_instrument_f_statistics`・`first_stage`は`compute_first_stage`、`wu_hausman_*`は
// `TwoSlsEstimator`、`overid_*`は`TwoSlsEstimator::sargan_*`/`GmmEstimator::hansen_j_*`から埋める。
// `IVResult`はRust側で組み立ててPythonに返すだけの型で、Python側からの生成・引数として
// 受け取ることは想定していないため`skip_from_py_object`（`IVOptions`の`from_py_object`とは
// 対照的、`OLSResult`/`LogitResult`と同じ理由）。
//
// `Clone`を派生しない: `first_stage`の要素`OLSResult`が、GILなしでは複製できない`Py<PyAny>`
// （`source_data`）を持つため`Clone`を実装していない。複製が要るときは`OLSResult::clone_ref`を使う。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug)]
pub struct IVResult {
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
    /// Distribution of `test_stats`: `"t"` or `"normal"`.
    #[pyo3(get)]
    pub stat_dist: String,
    /// Degrees of freedom of the t distribution (`None` for `"normal"`). May differ from
    /// `df_resid` (e.g. cluster-robust inference uses `G - 1`).
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
    #[pyo3(get)]
    pub df_resid: usize,
    #[pyo3(get)]
    pub df_model: usize,
    /// Whether GMM iteration converged (`estimator="gmm"` only). Always `true` for
    /// `estimator="2sls"` (2SLS is a closed-form, non-iterative estimator, so convergence is
    /// trivially satisfied). Also always `true` for `gmm_type="one_step"`/`"two_step"` —
    /// convergence is only actually checked for `gmm_type="iterated"`.
    #[pyo3(get)]
    pub converged: bool,
    /// Number of GMM estimations actually run, counting the initial estimate
    /// (`estimator="gmm"` only): 1 for "one_step", 2 for "two_step", at most `gmm_max_iter`
    /// for "iterated". Always `1` for `estimator="2sls"`.
    #[pyo3(get)]
    pub n_iter: i64,
    /// Standard error type actually used (echoes `IVOptions.cov_type`, normalized to
    /// lowercase; e.g. `"classical"`, `"hc1"`, `"hac"`, `"cluster"`).
    #[pyo3(get)]
    pub cov_type: String,
    /// Number of HAC (Newey-West) lags actually used: the explicit `hac_lags` if given,
    /// otherwise the value chosen automatically, `floor(4 * (n / 100) ^ (2 / 9))`. `Some` when
    /// `cov_type="hac"` or, with `estimator="gmm"` and `gmm_type` `"two_step"`/`"iterated"`,
    /// `gmm_weight_type="hac"` (both share `hac_lags`, so the value is the same); `None`
    /// otherwise.
    #[pyo3(get)]
    pub hac_lags_used: Option<i64>,
    /// Estimator actually used (echoes `IVOptions.estimator`, normalized to
    /// lowercase): `"2sls"` or `"gmm"`.
    #[pyo3(get)]
    pub estimator: String,
    /// Weight matrix actually used for GMM point estimation (echoes
    /// `IVOptions.gmm_weight_type`, normalized to lowercase; e.g. `"classical"`,
    /// `"robust"`, `"cluster"`, `"hac"`). `Some` only for `estimator="gmm"` (mirrors
    /// `overid_statistic`/`wu_hausman_statistic`'s use of `None` for "not applicable to
    /// this estimator"); always `None` for `estimator="2sls"`, which has no such concept, and
    /// for `gmm_type="one_step"`, which does not use a weight type.
    #[pyo3(get)]
    pub gmm_weight_type: Option<String>,
    /// GMM estimation type actually used (echoes `IVOptions.gmm_type`, normalized to
    /// lowercase): `"one_step"`, `"two_step"` or `"iterated"`. `None` for `estimator="2sls"`.
    #[pyo3(get)]
    pub gmm_type: Option<String>,
    #[pyo3(get)]
    pub wald_statistic: f64,
    #[pyo3(get)]
    pub wald_p_value: f64,
    /// Distribution of `wald_statistic`: `"f"` for `estimator="2sls"` (the Wald statistic
    /// divided by the number of slope coefficients), `"chi2"` for `estimator="gmm"` (the
    /// undivided Wald statistic).
    #[pyo3(get)]
    pub wald_dist: String,
    /// Numerator degrees of freedom of `wald_statistic` (number of slope coefficients;
    /// `None` when there are none and the statistic is NaN).
    #[pyo3(get)]
    pub wald_df_num: Option<usize>,
    /// Denominator degrees of freedom of `wald_statistic` for `"f"` (`df_resid`, or `G - 1`
    /// with cluster-robust inference). `None` for `"chi2"` and when the statistic is NaN.
    #[pyo3(get)]
    pub wald_df_denom: Option<usize>,
    #[pyo3(get)]
    pub r_squared: f64,
    #[pyo3(get)]
    pub adj_r_squared: f64,
    /// Weak-instrument diagnostic: the partial F-statistic for each endogenous
    /// variable (keyed by variable name), testing the excluded instruments' joint
    /// significance after partialling out `x_exog`.
    /// **Not** the same as the plain F-statistic of the corresponding regression in
    /// `first_stage()`, which includes `x_exog`'s contribution too.
    /// Computed the same way for both `estimator="2sls"` and `estimator="gmm"`.
    #[pyo3(get)]
    pub weak_instrument_f_statistics: HashMap<String, f64>,
    /// Numerator degrees of freedom of the weak-instrument F statistics (number of excluded
    /// instruments; the same for every endogenous variable). Never `None`, because `x_endog`
    /// always has at least one variable.
    #[pyo3(get)]
    pub weak_instrument_f_df_num: Option<usize>,
    /// Denominator degrees of freedom of the weak-instrument F statistics (residual df of
    /// the first-stage regressions; the same for every endogenous variable). Never `None`,
    /// because `x_endog` always has at least one variable.
    #[pyo3(get)]
    pub weak_instrument_f_df_denom: Option<usize>,
    /// Overidentification test statistic: Sargan (`estimator="2sls"`) or Hansen J
    /// (`estimator="gmm"`). `None` when just-identified (`len(instruments) ==
    /// len(x_endog)`, degrees of freedom 0).
    #[pyo3(get)]
    pub overid_statistic: Option<f64>,
    #[pyo3(get)]
    pub overid_p_value: Option<f64>,
    /// Degrees of freedom of the chi-squared overidentification test
    /// (`len(instruments) - len(x_endog)`); `None` when just identified.
    #[pyo3(get)]
    pub overid_df: Option<usize>,
    /// Wu-Hausman endogeneity test statistic (joint test over all endogenous
    /// variables, regression-based / `wooldridge_regression` formulation). Always computed
    /// under the `cov_type` in `options` (unlike `weak_instrument_f_statistics`, which is
    /// always classical; `linearmodels`' `wooldridge_regression` uses the same covariance as
    /// the underlying model, and this mirrors that). `None` when the augmented regression
    /// cannot be estimated (e.g. the augmented design is singular; `y` is exactly explained by
    /// the second-stage regressors and the first-stage residuals, so the augmented regression
    /// fits perfectly; or there are too few observations or clusters for the extra residual
    /// columns) — this does not fail
    /// `fit()` itself, since the other results remain valid. **Always `None` for
    /// `estimator="gmm"`** (not implemented for GMM).
    #[pyo3(get)]
    pub wu_hausman_statistic: Option<f64>,
    /// P-value of the Wu-Hausman test. `None` exactly when `wu_hausman_statistic` is `None`.
    #[pyo3(get)]
    pub wu_hausman_p_value: Option<f64>,
    /// Numerator degrees of freedom of the Wu-Hausman F test (the number of endogenous
    /// variables). `None` exactly when `wu_hausman_statistic` is `None`.
    #[pyo3(get)]
    pub wu_hausman_df_num: Option<usize>,
    /// Denominator degrees of freedom of the Wu-Hausman F test. `None` exactly when
    /// `wu_hausman_statistic` is `None`.
    #[pyo3(get)]
    pub wu_hausman_df_denom: Option<usize>,
    // `first_stage()`が読む。Python側には公開しない（`OLSResult`の`fitted_values`/
    // `has_intercept`、`LogitResult`/`ProbitResult`の`estimator`と同じ位置づけ）。
    // `estimator`非依存（`engine::iv::common::compute_first_stage`から構築）。
    pub(super) first_stage: Vec<(String, OLSResult)>,
    // 弱操作変数の診断（`cragg_donald()`・`kleibergen_paap()`・`effective_f()`）が、`fit()`と同じ
    // `IvInput`を作り直すために保持する。`source_data`は`fit()`に渡された`polars.DataFrame`
    // （参照を増やすだけでコピーしない）、`options`は`fit()`に渡した`IVOptions`の複製、
    // `x_exog`・`x_endog`・`instruments`は列名（`y`の列名は`dep_var_name`）。Python側には公開しない。
    // `source_data`と`options`は`fit`が結果を組み立てた後でGILを持って付ける（`OLSResult`と同じ）。
    pub(super) source_data: Option<Py<PyAny>>,
    pub(super) options: Option<IVOptions>,
    pub(super) x_exog: Vec<String>,
    pub(super) x_endog: Vec<String>,
    pub(super) instruments: Vec<String>,
}

impl IVResult {
    /// 元のデータ（`fit()`に渡された`polars.DataFrame`）。`fit`が必ず付けるので`None`にはならない
    /// （pickleにも含まれる）。`None`なら到達不能な状態として`INTERNAL_ERROR`。
    pub(super) fn retained_source<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.source_data
            .as_ref()
            .map(|source| source.bind(py).clone())
            .ok_or_else(|| internal_error("an IVResult without its training data"))
    }

    /// `fit()`に渡した`IVOptions`の複製。`retained_source`と同じく`None`にはならない。
    pub(super) fn retained_options(&self) -> PyResult<IVOptions> {
        self.options
            .clone()
            .ok_or_else(|| internal_error("an IVResult without its options"))
    }
}

// 保持するオプション（`IVOptions`）は、それ自身の`__reduce__`でpickleされるオブジェクトとして状態に入れる。
impl crate::shared::pickle::PickleField for IVOptions {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(Py::new(py, self.clone())?.into_bound(py).into_any())
    }

    fn from_py(obj: &Bound<'_, PyAny>) -> Option<Self> {
        let options = obj.cast::<IVOptions>().ok()?.try_borrow().ok()?;
        Some(options.clone())
    }
}

// 第一段階の結果（`OLSResult`）は、それぞれの`__reduce__`でpickleされるオブジェクトとして状態に入れる。
impl crate::shared::pickle::PickleField for Vec<(String, OLSResult)> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let items = self
            .iter()
            .map(|(name, result)| Ok((name.clone(), Py::new(py, result.clone_ref(py))?)))
            .collect::<PyResult<Vec<_>>>()?;
        items.into_bound_py_any(py)
    }

    fn from_py(obj: &Bound<'_, PyAny>) -> Option<Self> {
        let items: Vec<(String, Bound<'_, PyAny>)> = obj.extract().ok()?;
        items
            .into_iter()
            .map(|(name, item)| {
                let result = item.cast::<OLSResult>().ok()?.try_borrow().ok()?;
                Some((name, result.clone_ref(obj.py())))
            })
            .collect()
    }
}

crate::shared::pickle::impl_pickle_state!(IVResult {
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
    df_resid,
    df_model,
    converged,
    n_iter,
    cov_type,
    hac_lags_used,
    estimator,
    gmm_weight_type,
    gmm_type,
    wald_statistic,
    wald_p_value,
    wald_dist,
    wald_df_num,
    wald_df_denom,
    r_squared,
    adj_r_squared,
    weak_instrument_f_statistics,
    weak_instrument_f_df_num,
    weak_instrument_f_df_denom,
    overid_statistic,
    overid_p_value,
    overid_df,
    wu_hausman_statistic,
    wu_hausman_p_value,
    wu_hausman_df_num,
    wu_hausman_df_denom,
    first_stage,
    source_data,
    options,
    x_exog,
    x_endog,
    instruments,
});

#[pymethods]
impl IVResult {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        crate::shared::pickle::reduce(slf)
    }

    /// Per-endogenous-variable first-stage regression results
    /// (`x_endog[i] ~ x_exog + instruments`), keyed by the endogenous variable name.
    ///
    /// Each value is a full `OLSResults` (the same type OLS's `fit_ols` returns) — the
    /// first stage is a genuine, valid OLS regression in its own right, so no IV-specific
    /// result type is needed. Its `f_statistic`/`f_p_value`
    /// include `x_exog`'s contribution and are **not** the weak-instrument partial
    /// F-statistic (`weak_instrument_f_statistics`, computed separately).
    /// Computed the same way for both `estimator="2sls"` and `estimator="gmm"`.
    fn first_stage(&self, py: Python<'_>) -> PyResult<HashMap<String, OLSResult>> {
        guard(|| {
            Ok(self
                .first_stage
                .iter()
                .map(|(name, result)| (name.clone(), result.clone_ref(py)))
                .collect())
        })
    }

    /// Cragg-Donald statistic for the joint strength of the instruments, with the Stock-Yogo
    /// critical values where they are tabulated.
    ///
    /// A post-estimation diagnostic: never computed by `fit()`. It re-reads the variables from
    /// the data passed to `fit()`. Homoskedastic, so it does not depend on `cov_type`. With one
    /// endogenous variable it equals the partial F-statistic in `weak_instrument_f_statistics`.
    /// The same for `estimator="2sls"` and `estimator="gmm"`.
    ///
    /// Raises
    /// ------
    /// ComputationError
    ///     The statistic is undefined for these data (collinear instruments, a perfectly
    ///     fitted first stage, ...).
    fn cragg_donald(&self, py: Python<'_>) -> PyResult<CraggDonaldOutput> {
        guard(|| run_cragg_donald(self, py))
    }

    /// Kleibergen-Paap rk LM and rk Wald F statistics, robust to the `cov_type` of the fit.
    ///
    /// A post-estimation diagnostic: never computed by `fit()`. The covariance of the
    /// first-stage scores follows `cov_type`, `cluster`, `hac_lags` and `hac_time` of the
    /// options passed to `fit()`. Either statistic may be `None` when only the other is
    /// defined.
    ///
    /// Raises
    /// ------
    /// ComputationError
    ///     Neither statistic is defined for these data.
    fn kleibergen_paap(&self, py: Python<'_>) -> PyResult<KleibergenPaapOutput> {
        guard(|| run_kleibergen_paap(self, py))
    }

    /// Montiel Olea-Pflueger effective F statistic and its critical values.
    ///
    /// A post-estimation diagnostic: never computed by `fit()`. Defined for one endogenous
    /// variable; the covariance follows the options passed to `fit()`.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     `alpha` is not a real number.
    /// ValidationError
    ///     `alpha` is not strictly between 0 and 1, or there is not exactly one endogenous
    ///     variable.
    /// ComputationError
    ///     The statistic is undefined for these data.
    fn effective_f(&self, py: Python<'_>, alpha: &Bound<'_, PyAny>) -> PyResult<EffectiveFOutput> {
        guard(|| {
            let alpha = extract_strict_float(alpha, "alpha")?;
            run_effective_f(self, py, alpha)
        })
    }
}
