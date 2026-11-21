//! `FEIVResult`（FEIVの推定結果）のpyclass定義と、`first_stage()`・`fixed_effects()`・弱操作変数診断・
//! `c_stat()`のメソッド。`iv/common/result.rs`に対応する。
//!
//! ## 診断・`fixed_effects()`・`c_stat()`は元のデータから再計算する
//!
//! 結果は`fit()`に渡した`DataFrame`（`source_data`）と`FEIVOptions`・列名を保持し、`cragg_donald()`・
//! `kleibergen_paap()`・`effective_f()`・`fixed_effects()`は呼ばれたときに`fit()`と同じ手順で
//! `FeTwoSlsEstimator`を作り直して呼び、`c_stat()`は`fe_gmm_c_statistic`を呼ぶ（IVの診断と同じ。推定量
//! （設計行列を持つ）を結果に持たせない）。第一段階の結果は`first_stage()`が`FEResult`で返す。固定効果の辞書は
//! 第一段階ごとに`fit()`で計算して持つ（個体数に比例する小さなもの）ので、`FeEstimator`を保持しない。
//!
//! ## 推定量ごとの違い
//!
//! `estimator="2sls"`は第一段階・弱操作変数診断・Wu-Hausman・`fixed_effects()`を持ち、推論はt・F、過剰識別はSargan。
//! `estimator="gmm"`は推論が正規分布・χ²、過剰識別がHansen J、事後診断が`c_stat()`で、第一段階・弱操作変数診断・
//! Wu-Hausman・`fixed_effects()`は今は持たない（engineの`FeGmmEstimator`が持たない）。持たないメソッドは
//! `METHOD_NOT_AVAILABLE_FOR_ESTIMATOR`、持たないプロパティは`None`。

use std::collections::HashMap;

use crate::iv::common::{
    CStatisticOutput, CraggDonaldOutput, EffectiveFOutput, KleibergenPaapOutput,
    ResolvedCStatTarget, resolve_c_stat_target,
};
use crate::shared::errors::{guard, internal_error, user_error};
use engine::panel::fe_iv::gmm::FeGmmEstimator;
use engine::panel::fe_iv::two_sls::FeTwoSlsEstimator;
use engine::shared::error_catalog::{ErrorCode, ParamValue};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::super::common::{groups_dict, panel_error_to_pyerr};
use super::super::fe::{FEResult, fixed_effects_dict};
use super::fit::{compute_c_statistic, estimate_two_sls};
use super::input::FeIvColumns;
use super::option_values::EstimatorKind;
use super::options::FEIVOptions;
use crate::shared::column_extraction::select_columns;
use crate::shared::mat_conversion::{mat_to_rows, mat_to_vec};
use crate::shared::option_values::extract_strict_float;

// 第一段階の結果（`FEResult`）は、それぞれの`__reduce__`でpickleされるオブジェクトとして状態に入れる。
impl crate::shared::pickle::PickleField for Vec<(String, FEResult)> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::IntoPyObjectExt;
        let items = self
            .iter()
            .map(|(name, result)| Ok((name.clone(), Py::new(py, result.clone())?)))
            .collect::<PyResult<Vec<_>>>()?;
        items.into_bound_py_any(py)
    }

    fn from_py(obj: &Bound<'_, PyAny>) -> Option<Self> {
        let items: Vec<(String, Bound<'_, PyAny>)> = obj.extract().ok()?;
        items
            .into_iter()
            .map(|(name, item)| {
                let result = item.cast::<FEResult>().ok()?.try_borrow().ok()?;
                Some((name, result.clone()))
            })
            .collect()
    }
}

/// Estimation results for FEIV (fixed effects instrumental variables, with two-stage least
/// squares or GMM).
///
/// Structured data only (no `summary()`). All array-valued fields (`params`,
/// `std_errors`, etc.) share the same order as `param_names` (`x_exog`, then `x_endog`).
///
/// With `estimator="2sls"`, `test_stats` holds the t-statistics (`stat_df` is their degrees of
/// freedom) and `wald_*` is an F test. With `estimator="gmm"`, `test_stats` holds the
/// z-statistics (normal distribution, `stat_df` is `None`) and `wald_*` is a chi-squared test.
/// `r_squared_*` are computed from the structural residuals (`y - X b` with the actual endogenous
/// regressors `X`) and can be negative. `overid_*` is the Sargan test (`estimator="2sls"`) or the
/// Hansen J test (`estimator="gmm"`), computed from the within-transformed data (`None` when just
/// identified); for the Sargan test, and for the Hansen J test with the classical weight (and
/// `gmm_type="one_step"`), its variance is `e'e / (n - n_absorbed)`, where `n_absorbed` is the
/// number of absorbed fixed-effect parameters. The "robust" and "cluster" weights are not
/// adjusted. The "cluster" weight needs no adjustment when every column of `effects` is nested
/// in the clusters (for example one-way effects clustered by the entity, or by a coarser column).
/// With an effect that crosses the clusters, such as a time effect in a two-way model, that holds
/// only approximately, and not at all for clusters that cross the entities (such as time
/// periods). The "robust" weight makes the statistic over-reject on a short panel.
///
/// The weak-instrument partial F statistics and the Wu-Hausman test are computed by
/// `estimator="2sls"` only (`None` with `estimator="gmm"`). The first-stage regressions, the fixed
/// effects themselves and the Cragg-Donald, Kleibergen-Paap and effective F diagnostics are not
/// fields: they are the methods `first_stage()`, `fixed_effects()`, `cragg_donald()`,
/// `kleibergen_paap()` and `effective_f()`, available with `estimator="2sls"` only. The C
/// statistic is the method `c_stat()`, available with `estimator="gmm"` only.
// `Clone`を派生しない: 第一段階の`FEResult`は複製できるが、`source_data`（`Py<PyAny>`）が
// GILなしでは複製できない（`IVResult`と同じ理由）。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug)]
pub struct FEIVResult {
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
    /// Distribution of `test_stats`: `"t"` for `estimator="2sls"`, `"normal"` for
    /// `estimator="gmm"`.
    #[pyo3(get)]
    pub stat_dist: String,
    /// Degrees of freedom of the t distribution (`estimator="2sls"`); `None` for the normal
    /// distribution (`estimator="gmm"`). May differ from `df_resid`: cluster-robust inference
    /// uses `G - 1`, where `G` is the number of clusters, and Driscoll-Kraay uses the number of
    /// `dk_time` periods minus 1.
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
    /// Structural residuals `y - X b` of the within-transformed data, with the actual
    /// endogenous regressors `X`.
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
    /// The columns of `effects`, in the order they were given.
    effects: Vec<String>,
    /// Number of groups (unique values) of each column of `effects`, in the same order.
    group_counts: Vec<usize>,
    /// Standard error type actually used (echoes `FEIVOptions.cov_type`, normalized to lowercase).
    #[pyo3(get)]
    pub cov_type: String,
    /// Driscoll-Kraay bandwidth actually used. `None` unless `cov_type="dk"`.
    #[pyo3(get)]
    pub dk_bandwidth_used: Option<i64>,
    /// Estimator actually used: `"2sls"` or `"gmm"` (echoes `FEIVOptions.estimator`, normalized
    /// to lowercase).
    #[pyo3(get)]
    pub estimator: String,
    /// GMM variant actually used: `"one_step"`, `"two_step"` or `"iterated"`. `None` for
    /// `estimator="2sls"`.
    #[pyo3(get)]
    pub gmm_type: Option<String>,
    /// Weight matrix actually used: `"classical"`, `"robust"` or `"cluster"`. `None` for
    /// `estimator="2sls"` and for `gmm_type="one_step"`, which has no weight type.
    #[pyo3(get)]
    pub gmm_weight_type: Option<String>,
    /// Number of GMM estimations performed (1 for `"one_step"` and for `estimator="2sls"`, 2 for
    /// `"two_step"`).
    #[pyo3(get)]
    pub n_iter: i64,
    /// Whether the iteration converged (always `True` except for an `"iterated"` fit that did not
    /// converge and was allowed to return).
    #[pyo3(get)]
    pub converged: bool,
    /// Statistic of the joint test that all slope coefficients are zero.
    #[pyo3(get)]
    pub wald_statistic: f64,
    #[pyo3(get)]
    pub wald_p_value: f64,
    /// Distribution of `wald_statistic`: `"f"` for `estimator="2sls"`, `"chi2"` for
    /// `estimator="gmm"`.
    #[pyo3(get)]
    pub wald_dist: String,
    #[pyo3(get)]
    pub wald_df_num: usize,
    /// Follows `cov_type` like the t-tests, so it equals `stat_df`. `None` for `estimator="gmm"`
    /// (chi-squared).
    #[pyo3(get)]
    pub wald_df_denom: Option<usize>,
    #[pyo3(get)]
    pub r_squared_within: f64,
    #[pyo3(get)]
    pub r_squared_between: f64,
    #[pyo3(get)]
    pub r_squared_overall: f64,
    /// Weak-instrument partial F statistic of each endogenous variable, keyed by its name.
    /// Homoskedastic whatever `cov_type` is; its denominator degrees of freedom are
    /// `n - l - n_absorbed`, where `l` is the number of exogenous variables plus instruments and
    /// `n_absorbed` the number of absorbed fixed-effect parameters. `None` for `estimator="gmm"`.
    #[pyo3(get)]
    pub weak_instrument_f_statistics: Option<HashMap<String, f64>>,
    /// Numerator degrees of freedom of the weak-instrument F statistics (the number of excluded
    /// instruments).
    #[pyo3(get)]
    pub weak_instrument_f_df_num: Option<usize>,
    /// Denominator degrees of freedom of the weak-instrument F statistics.
    #[pyo3(get)]
    pub weak_instrument_f_df_denom: Option<usize>,
    /// Overidentification statistic: Sargan for `estimator="2sls"`, Hansen J for
    /// `estimator="gmm"`. `None` when the model is exactly identified.
    #[pyo3(get)]
    pub overid_statistic: Option<f64>,
    #[pyo3(get)]
    pub overid_p_value: Option<f64>,
    #[pyo3(get)]
    pub overid_df: Option<usize>,
    /// Wu-Hausman endogeneity test (the regression form, an auxiliary fixed-effects regression
    /// with the first-stage residuals added; follows `cov_type`). `None` when the auxiliary
    /// regression cannot be estimated, and for `estimator="gmm"`.
    #[pyo3(get)]
    pub wu_hausman_statistic: Option<f64>,
    #[pyo3(get)]
    pub wu_hausman_p_value: Option<f64>,
    #[pyo3(get)]
    pub wu_hausman_df_num: Option<usize>,
    #[pyo3(get)]
    pub wu_hausman_df_denom: Option<usize>,
    // `first_stage()`が読む。内生変数ごとの第一段階（固定効果は計算済み）。Python側には公開しない。
    // `estimator="gmm"`では空。
    first_stage: Vec<(String, FEResult)>,
    // 診断・`fixed_effects()`・`c_stat()`が入力を作り直すために保持する（`IVResult`と同じ）。`source_data`は
    // `fit()`に渡された`polars.DataFrame`（参照を増やすだけ）、`options`は渡した`FEIVOptions`の複製。
    // `source_data`と`options`は`fit`が結果を組み立てた後でGILを持って付ける。
    source_data: Option<Py<PyAny>>,
    options: Option<FEIVOptions>,
    x_exog: Vec<String>,
    x_endog: Vec<String>,
    instruments: Vec<String>,
}

crate::shared::pickle::impl_pickle_state!(FEIVResult {
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
    effects,
    group_counts,
    cov_type,
    dk_bandwidth_used,
    estimator,
    gmm_type,
    gmm_weight_type,
    n_iter,
    converged,
    wald_statistic,
    wald_p_value,
    wald_dist,
    wald_df_num,
    wald_df_denom,
    r_squared_within,
    r_squared_between,
    r_squared_overall,
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

impl FEIVResult {
    /// `fit`が結果を組み立てた後、診断が入力を作り直すための元のデータとオプションを付ける（GILを持って呼ぶ）。
    pub(super) fn retain_source(&mut self, source: Py<PyAny>, options: FEIVOptions) {
        self.source_data = Some(source);
        self.options = Some(options);
    }

    /// `FeTwoSlsEstimator`の結果。
    pub(super) fn from_two_sls(
        estimator: &FeTwoSlsEstimator,
        first_stage: Vec<(String, FEResult)>,
        columns: FeIvColumns,
        cov_type_name: String,
    ) -> PyResult<Self> {
        let FeIvColumns {
            x_exog,
            x_endog,
            instruments,
            effects,
            ..
        } = columns;
        let input = estimator.input();
        let (f_statistic, f_p_value) = estimator
            .f_statistic()
            .zip(estimator.f_p_value())
            .ok_or_else(|| internal_error("the joint test of a FEIV fit is missing"))?;
        let (wald_df_num, wald_df_denom) = estimator.f_df().ok_or_else(|| {
            internal_error("the degrees of freedom of a FEIV overall F test are missing")
        })?;
        let weak_df = estimator.weak_instrument_f_df();
        let wu_hausman_df = estimator.wu_hausman_df();

        Ok(Self {
            params: mat_to_vec(estimator.params()),
            std_errors: mat_to_vec(estimator.std_errors()),
            cov_params: mat_to_rows(estimator.cov_params()),
            test_stats: mat_to_vec(estimator.test_stats()),
            stat_dist: "t".to_string(),
            stat_df: Some(estimator.df_inference() as i64),
            p_values: mat_to_vec(estimator.p_values()),
            conf_lower: mat_to_vec(estimator.conf_lower()),
            conf_upper: mat_to_vec(estimator.conf_upper()),
            param_names: estimator.param_names().to_vec(),
            residuals: mat_to_vec(estimator.residuals()),
            dep_var_name: input.dep_var_name().to_string(),
            n_obs: input.nobs(),
            df_resid: estimator.df_resid(),
            df_model: estimator.df_model(),
            effects,
            group_counts: estimator.n_groups().to_vec(),
            cov_type: cov_type_name,
            dk_bandwidth_used: estimator.dk_bandwidth_used().map(|bw| bw as i64),
            estimator: EstimatorKind::TwoSls.name().to_string(),
            gmm_type: None,
            gmm_weight_type: None,
            n_iter: 1,
            converged: true,
            wald_statistic: f_statistic,
            wald_p_value: f_p_value,
            wald_dist: "f".to_string(),
            wald_df_num,
            wald_df_denom: Some(wald_df_denom),
            r_squared_within: estimator.r_squared_within(),
            r_squared_between: estimator.r_squared_between(),
            r_squared_overall: estimator.r_squared_overall(),
            weak_instrument_f_statistics: Some(
                estimator
                    .weak_instrument_f_statistics()
                    .iter()
                    .cloned()
                    .collect(),
            ),
            weak_instrument_f_df_num: weak_df.map(|(num, _)| num),
            weak_instrument_f_df_denom: weak_df.map(|(_, denom)| denom),
            overid_statistic: estimator.sargan_statistic(),
            overid_p_value: estimator.sargan_p_value(),
            overid_df: estimator.sargan_df(),
            wu_hausman_statistic: estimator.wu_hausman_statistic(),
            wu_hausman_p_value: estimator.wu_hausman_p_value(),
            wu_hausman_df_num: wu_hausman_df.map(|(num, _)| num),
            wu_hausman_df_denom: wu_hausman_df.map(|(_, denom)| denom),
            first_stage,
            source_data: None,
            options: None,
            x_exog,
            x_endog,
            instruments,
        })
    }

    /// `FeGmmEstimator`の結果。第一段階・弱操作変数診断・Wu-Hausmanは持たない（`None`・空）。
    pub(super) fn from_gmm(
        estimator: &FeGmmEstimator,
        columns: FeIvColumns,
        cov_type_name: String,
        gmm_type_name: String,
        weight_type_name: Option<String>,
    ) -> PyResult<Self> {
        let FeIvColumns {
            x_exog,
            x_endog,
            instruments,
            effects,
            ..
        } = columns;
        let input = estimator.input();
        let params = mat_to_vec(estimator.params());
        let (wald_statistic, wald_p_value) = estimator
            .wald_statistic()
            .zip(estimator.wald_p_value())
            .ok_or_else(|| internal_error("the joint test of a FEIV fit is missing"))?;

        Ok(Self {
            wald_df_num: params.len(),
            params,
            std_errors: mat_to_vec(estimator.std_errors()),
            cov_params: mat_to_rows(estimator.cov_params()),
            test_stats: mat_to_vec(estimator.test_stats()),
            stat_dist: "normal".to_string(),
            stat_df: None,
            p_values: mat_to_vec(estimator.p_values()),
            conf_lower: mat_to_vec(estimator.conf_lower()),
            conf_upper: mat_to_vec(estimator.conf_upper()),
            param_names: estimator.param_names().to_vec(),
            residuals: mat_to_vec(estimator.residuals()),
            dep_var_name: input.dep_var_name().to_string(),
            n_obs: input.nobs(),
            df_resid: estimator.df_resid(),
            df_model: estimator.df_model(),
            effects,
            group_counts: estimator.n_groups().to_vec(),
            cov_type: cov_type_name,
            dk_bandwidth_used: estimator.dk_bandwidth_used().map(|bw| bw as i64),
            estimator: EstimatorKind::Gmm.name().to_string(),
            gmm_type: Some(gmm_type_name),
            gmm_weight_type: weight_type_name,
            n_iter: estimator.n_iter(),
            converged: estimator.converged(),
            wald_statistic,
            wald_p_value,
            wald_dist: "chi2".to_string(),
            wald_df_denom: None,
            r_squared_within: estimator.r_squared_within(),
            r_squared_between: estimator.r_squared_between(),
            r_squared_overall: estimator.r_squared_overall(),
            weak_instrument_f_statistics: None,
            weak_instrument_f_df_num: None,
            weak_instrument_f_df_denom: None,
            overid_statistic: estimator.hansen_j_statistic(),
            overid_p_value: estimator.hansen_j_p_value(),
            overid_df: estimator.hansen_j_df(),
            wu_hausman_statistic: None,
            wu_hausman_p_value: None,
            wu_hausman_df_num: None,
            wu_hausman_df_denom: None,
            first_stage: Vec::new(),
            source_data: None,
            options: None,
            x_exog,
            x_endog,
            instruments,
        })
    }

    /// 元のデータ（`fit()`に渡された`polars.DataFrame`）。`fit`が必ず付けるので`None`にはならない
    /// （pickleにも含まれる）。`None`なら到達不能な状態として`INTERNAL_ERROR`。
    fn retained_source<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.source_data
            .as_ref()
            .map(|source| source.bind(py).clone())
            .ok_or_else(|| internal_error("an FEIVResult without its training data"))
    }

    /// `fit()`に渡した`FEIVOptions`の複製。`retained_source`と同じく`None`にはならない。
    fn retained_options(&self) -> PyResult<FEIVOptions> {
        self.options
            .clone()
            .ok_or_else(|| internal_error("an FEIVResult without its options"))
    }

    /// 結果の`estimator`が`supported`でなければ`METHOD_NOT_AVAILABLE_FOR_ESTIMATOR`にする。
    fn require_estimator(&self, method: &str, supported: EstimatorKind) -> PyResult<()> {
        let supported = supported.name();
        if self.estimator == supported {
            return Ok(());
        }
        Err(user_error(
            ErrorCode::MethodNotAvailableForEstimator,
            vec![
                ("method", ParamValue::from(method)),
                ("estimator", ParamValue::from(self.estimator.as_str())),
                ("supported", ParamValue::from(supported)),
            ],
        ))
    }

    fn columns(&self) -> FeIvColumns {
        FeIvColumns {
            y: self.dep_var_name.clone(),
            x_exog: self.x_exog.clone(),
            x_endog: self.x_endog.clone(),
            instruments: self.instruments.clone(),
            effects: self.effects.clone(),
        }
    }

    /// `fit()`と同じ手順で`FeTwoSlsEstimator`を作り直し、`compute`に渡す（`estimator="2sls"`のとき）。元のデータの
    /// 取り込み（GILが要る）を済ませた後は、GILを手放して走る。
    fn with_estimator<T: Send>(
        &self,
        py: Python<'_>,
        method: &str,
        compute: impl FnOnce(&FeTwoSlsEstimator) -> PyResult<T> + Send,
    ) -> PyResult<T> {
        self.require_estimator(method, EstimatorKind::TwoSls)?;
        let source = self.retained_source(py)?;
        let options = self.retained_options()?;
        let columns = self.columns();
        let frame = select_columns(&source, "data", &columns.data_columns(&options))?.frame;
        py.detach(|| {
            let estimator = estimate_two_sls(&frame, columns, &options)?;
            compute(&estimator)
        })
    }
}

#[pymethods]
impl FEIVResult {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        crate::shared::pickle::reduce(slf)
    }

    /// Number of groups (unique values) of each column of `effects`, as a `dict[str, int]`
    /// keyed by column name, in the order the columns were given.
    #[getter]
    fn n_groups<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        groups_dict(py, &self.effects, &self.group_counts)
    }

    /// Per-endogenous-variable first-stage regression results
    /// (`x_endog[i] ~ x_exog + instruments`, with the same fixed effects and `cov_type`),
    /// keyed by the endogenous variable name. Available with `estimator="2sls"` only.
    ///
    /// Each value is a full `FEResults` (the same type FE's `fit_fe` returns). Its
    /// `f_statistic` is `None` (the overall test is not computed for a first stage) and its
    /// `fixed_effects()` is available. The weak-instrument partial F statistic is
    /// `weak_instrument_f_statistics`, computed separately.
    ///
    /// Raises
    /// ------
    /// ValidationError
    ///     The model was fitted with `estimator="gmm"`.
    fn first_stage(&self) -> PyResult<HashMap<String, FEResult>> {
        guard(|| {
            self.require_estimator("first_stage()", EstimatorKind::TwoSls)?;
            Ok(self
                .first_stage
                .iter()
                .map(|(name, result)| (name.clone(), result.clone()))
                .collect())
        })
    }

    /// The fixed effects themselves, recovered post-hoc from the fitted coefficients
    /// (`α_i = ȳ_i - x̄_i'β̂` with the actual endogenous regressors). Available with
    /// `estimator="2sls"` only.
    ///
    /// Always a `dict[str, dict[str, float]]` keyed by the column names in `effects`, then by
    /// group. The normalization is the same as `FEResults.fixed_effects()`. It re-reads the
    /// data passed to `fit()` and estimates the model again on every call, so it costs about as
    /// much as `fit()`.
    ///
    /// Raises
    /// ------
    /// ValidationError
    ///     The model was fitted with `estimator="gmm"`.
    fn fixed_effects(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        guard(|| {
            let fixed = self.with_estimator(py, "fixed_effects()", |estimator| {
                Ok(estimator.fixed_effects())
            })?;
            fixed_effects_dict(py, &self.effects, &fixed)
        })
    }

    /// Cragg-Donald statistic for the joint strength of the instruments, with the Stock-Yogo
    /// critical values where they are tabulated. Available with `estimator="2sls"` only.
    ///
    /// A post-estimation diagnostic: never computed by `fit()`. It re-reads the variables from
    /// the data passed to `fit()` and estimates the model again on every call. Homoskedastic, so
    /// it does not depend on `cov_type`; the
    /// absorbed fixed-effect parameters are subtracted from its degrees of freedom. With one
    /// endogenous variable it equals the partial F-statistic in `weak_instrument_f_statistics`.
    ///
    /// Raises
    /// ------
    /// ValidationError
    ///     The model was fitted with `estimator="gmm"`.
    /// ComputationError
    ///     The statistic is undefined for these data (collinear instruments, a perfectly
    ///     fitted first stage, ...).
    fn cragg_donald(&self, py: Python<'_>) -> PyResult<CraggDonaldOutput> {
        guard(|| {
            self.with_estimator(py, "cragg_donald()", |estimator| {
                let cd = estimator.cragg_donald().map_err(panel_error_to_pyerr)?;
                Ok(CraggDonaldOutput::from_engine(cd))
            })
        })
    }

    /// Kleibergen-Paap rk LM and rk Wald F statistics, robust to the `cov_type` of the fit.
    /// Available with `estimator="2sls"` only.
    ///
    /// A post-estimation diagnostic: never computed by `fit()`; it estimates the model again
    /// on every call. The covariance of the first-stage scores follows `cov_type`, `cluster`,
    /// `dk_time` and `dk_bandwidth` of the options passed to `fit()`. Either statistic may be
    /// `None` when only the other is defined.
    ///
    /// Raises
    /// ------
    /// ValidationError
    ///     The model was fitted with `estimator="gmm"`.
    /// ComputationError
    ///     Neither statistic is defined for these data.
    fn kleibergen_paap(&self, py: Python<'_>) -> PyResult<KleibergenPaapOutput> {
        guard(|| {
            self.with_estimator(py, "kleibergen_paap()", |estimator| {
                let kp = estimator.kleibergen_paap().map_err(panel_error_to_pyerr)?;
                Ok(KleibergenPaapOutput::from_engine(kp))
            })
        })
    }

    /// Montiel Olea-Pflueger effective F statistic and its critical values. Available with
    /// `estimator="2sls"` only.
    ///
    /// A post-estimation diagnostic: never computed by `fit()`; it estimates the model again
    /// on every call. Defined for one endogenous variable; the covariance follows the options
    /// passed to `fit()`.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     `alpha` is not a real number.
    /// ValidationError
    ///     The model was fitted with `estimator="gmm"`, `alpha` is not strictly between 0 and 1,
    ///     or there is not exactly one endogenous variable.
    /// ComputationError
    ///     The statistic is undefined for these data.
    fn effective_f(&self, py: Python<'_>, alpha: &Bound<'_, PyAny>) -> PyResult<EffectiveFOutput> {
        guard(|| {
            // 推定量の検査を先にする（GMMの結果では、`alpha`の型によらず`METHOD_NOT_AVAILABLE_FOR_ESTIMATOR`）。
            self.require_estimator("effective_f()", EstimatorKind::TwoSls)?;
            let alpha = extract_strict_float(alpha, "alpha")?;
            self.with_estimator(py, "effective_f()", |estimator| {
                let eff = estimator.effective_f(alpha).map_err(panel_error_to_pyerr)?;
                Ok(EffectiveFOutput::from_engine(eff))
            })
        })
    }

    /// C statistic (difference-in-Hansen test) of the exogeneity of endogenous variables
    /// and of the orthogonality of excluded instruments, computed on the within-transformed
    /// data. Available with `estimator="gmm"` only.
    ///
    /// A post-estimation diagnostic: never computed by `fit()`. It re-reads the variables from
    /// the data passed to `fit()` (keep that data unchanged) and estimates the model again on
    /// every call. The model is
    /// estimated twice by two-step GMM, with and without the tested moment conditions, from
    /// the same covariance of the moment conditions; the statistic is the difference of the two
    /// Hansen J statistics. The weight follows `gmm_weight_type` (`cluster` included) for
    /// `gmm_type="two_step"` and `"iterated"`, and is `"classical"` for `gmm_type="one_step"`,
    /// which has no weight type. For the classical weight the variance of the moment
    /// conditions uses `n - n_absorbed` like the Hansen J test; the other weights are not
    /// adjusted. As for the Hansen J test, the cluster weight needs no adjustment when every
    /// column of `effects` is nested in the clusters, and the statistic over-rejects with a
    /// "robust" weight on a short panel.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     `variables` or `instruments` is not a list of str.
    /// ValidationError
    ///     The model was fitted with `estimator="2sls"`, nothing is tested, a list is empty or
    ///     repeats a name, a name is not an endogenous variable (instrument) of the model, or too
    ///     many instruments are tested.
    /// ComputationError
    ///     The statistic is undefined for these data (collinear instruments, a perfect fit,
    ///     or fewer clusters than the moment conditions of the full model).
    #[pyo3(signature = (variables=None, instruments=None))]
    fn c_stat(
        &self,
        py: Python<'_>,
        variables: Option<&Bound<'_, PyAny>>,
        instruments: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<CStatisticOutput> {
        guard(|| {
            self.require_estimator("c_stat()", EstimatorKind::Gmm)?;
            // 型の検査 → 値（空・重複・名前）の検査の順は`resolve_c_stat_target`が持つ。
            let ResolvedCStatTarget {
                target,
                variables: tested_endog,
                instruments: tested_instruments,
            } = resolve_c_stat_target(variables, instruments, &self.x_endog, &self.instruments)?;
            let source = self.retained_source(py)?;
            let options = self.retained_options()?;
            let columns = self.columns();
            let frame = select_columns(&source, "data", &columns.data_columns(&options))?.frame;
            py.detach(|| {
                let c = compute_c_statistic(&frame, columns, &options, &target)?;
                Ok(CStatisticOutput::from_engine(
                    c,
                    tested_endog,
                    tested_instruments,
                ))
            })
        })
    }
}
