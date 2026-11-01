//! `TobitResult`（Tobitの推定結果）と、その`#[pymethods]`（`predict`・`augment`・`marginal_effects`・`censoring_fit_check`）。

use crate::nonlinear::common::{
    MarginalEffectsResult, mle_error_to_pyerr, parse_marginal_effects_at,
};
use crate::shared::augment::append_predicted_column;
use crate::shared::column_error::ColumnSource;
use crate::shared::column_extraction::{
    SelectedData, extract_model_x_columns, select_model_x_columns, x_column_names,
};
use crate::shared::errors::{guard, unknown_option_value};
use engine::nonlinear::tobit::{MarginalEffectsTarget, TobitFitted, TobitInput};
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use super::censoring_fit::{CensoringFitCheckResult, censoring_fit_check_to_result};
use super::input::validate_no_sigma_collision;

/// Estimation results for Tobit.
///
/// Structured data only (no `summary()`). `predict()` / `marginal_effects()` /
/// `censoring_fit_check()` are separate methods, not fields.
///
/// `params`/`param_names`/`std_errors`/`test_stats`/`p_values`/`conf_lower`/`conf_upper` are all
/// `(k+1)`-length: `sigma` (the error term's standard deviation) is appended as the last
/// element, with `param_names[-1] == "sigma"`, so every array-valued field can be zipped
/// together. `sigma` is a convenience shortcut equal to `params[-1]`.
///
/// `log_likelihood_null`/`lr_statistic`/`lr_p_value`/`pseudo_r_squared` are not provided for
/// Tobit (no closed form exists under censoring); `wald_statistic`/`wald_p_value` give the
/// overall model significance test instead.
// `LogitResult`と同じ理由で`skip_from_py_object`・フィールドごとの個別`#[pyo3(get)]`・
// `Clone`非導出（`fitted: TobitFitted`が`Clone`を実装していないため）。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug)]
pub struct TobitResult {
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
    pub dep_var_name: String,
    /// Point estimate of `sigma` (the error term's standard deviation). Equal to
    /// `params[-1]`.
    #[pyo3(get)]
    pub sigma: f64,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    #[pyo3(get)]
    pub n_obs: usize,
    #[pyo3(get)]
    pub df_model: usize,
    #[pyo3(get)]
    pub df_resid: usize,
    /// Wald test statistic for overall model significance (all slope coefficients
    /// jointly zero). `NaN` when `df_model == 0` (no slope coefficients to test).
    #[pyo3(get)]
    pub wald_statistic: f64,
    /// p-value for `wald_statistic` (chi-squared distribution with `df_model`
    /// degrees of freedom). `NaN` when `df_model == 0`.
    #[pyo3(get)]
    pub wald_p_value: f64,
    /// Distribution of `wald_statistic`: always `"chi2"`.
    #[pyo3(get)]
    pub wald_dist: String,
    /// Degrees of freedom of the chi-squared Wald test (`None` when there are no slope
    /// coefficients).
    #[pyo3(get)]
    pub wald_df: Option<usize>,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub n_iter: usize,
    /// Standard error type actually used (echoes `TobitOptions.cov_type`, normalized
    /// to lowercase).
    #[pyo3(get)]
    pub cov_type: String,
    /// Optimization solver actually used (echoes `TobitOptions.solver`, normalized
    /// to lowercase; one of `"newton"`, `"bfgs"`, `"lbfgs"`).
    #[pyo3(get)]
    pub solver: String,
    /// Lower censoring bound actually used (echoes `TobitOptions.lower`).
    #[pyo3(get)]
    pub lower: Option<f64>,
    /// Upper censoring bound actually used (echoes `TobitOptions.upper`).
    #[pyo3(get)]
    pub upper: Option<f64>,
    /// Not exposed to Python; only `predict`/`marginal_effects`/`censoring_fit_check`
    /// read it (`LogitResult`の`fitted`と同じ位置づけ)。
    pub(super) fitted: TobitFitted,
    /// The `polars.DataFrame` passed to `fit()`, as given. `augment(new_data=None)` appends
    /// the predicted column to it, so every column is kept, including the ones the model
    /// does not read.
    pub(super) source_data: Py<PyAny>,
}

// 入力データのうち`y`・`x`は状態に入れない（`source_data`の列から作り直せる）。入れるのは、`x`の列名・`y`の列名・
// 切片の有無・打ち切り境界・係数・`σ`・分散共分散行列。復元は`from_columns`/`from_parts`を通して次元の整合を検査する
// （列が無い・不整合なら`PICKLE_FORMAT_MISMATCH`）。
impl crate::shared::pickle::PickleFieldFromSource for TobitFitted {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use crate::shared::pickle::{PickleField, to_tuple};
        let input = self.input();
        let x_names = x_column_names(input.param_names(), input.has_intercept()).to_vec();
        to_tuple(
            py,
            vec![
                x_names.to_py(py)?,
                input.dep_var_name().to_string().to_py(py)?,
                input.has_intercept().to_py(py)?,
                input.lower().to_py(py)?,
                input.upper().to_py(py)?,
                self.params().to_vec().to_py(py)?,
                self.sigma().to_py(py)?,
                self.cov_params().to_py(py)?,
            ],
        )
    }

    fn from_py(obj: &Bound<'_, PyAny>, source: &Bound<'_, PyAny>) -> Option<Self> {
        use crate::nonlinear::common::rebuild_columns;
        use crate::shared::pickle::{PickleField, from_tuple};
        let items = from_tuple(obj, 8)?;
        let x_names: Vec<String> = PickleField::from_py(&items[0])?;
        let dep_var_name: String = PickleField::from_py(&items[1])?;
        let has_intercept: bool = PickleField::from_py(&items[2])?;
        let params: Vec<f64> = PickleField::from_py(&items[5])?;
        validate_no_sigma_collision(&x_names).ok()?;
        let (y, x) = rebuild_columns(source, &dep_var_name, &x_names, has_intercept, params.len())?;
        let input = TobitInput::from_columns(
            &y,
            &x,
            x_names,
            has_intercept,
            dep_var_name,
            PickleField::from_py(&items[3])?,
            PickleField::from_py(&items[4])?,
        )
        .ok()?;
        TobitFitted::from_parts(
            input,
            params,
            PickleField::from_py(&items[6])?,
            PickleField::from_py(&items[7])?,
        )
        .ok()
    }
}

crate::shared::pickle::impl_pickle_state!(TobitResult {
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
    dep_var_name,
    sigma,
    log_likelihood,
    aic,
    bic,
    n_obs,
    df_model,
    df_resid,
    wald_statistic,
    wald_p_value,
    wald_dist,
    wald_df,
    converged,
    n_iter,
    cov_type,
    solver,
    lower,
    upper,
    source_data,
} rebuilt_from(source_data) {
    fitted
});

// `#[pymethods]`ブロックの外に置く非公開実装（`LogitResult`と同じ理由）。
impl TobitResult {
    /// `new_data`から`predict()`/`augment()`が使う`x`列だけを取り込む。
    fn select_new_data(&self, new_data: &Bound<'_, PyAny>) -> PyResult<SelectedData> {
        select_model_x_columns(
            new_data,
            "new_data",
            self.fitted.input().param_names(),
            self.fitted.input().has_intercept(),
        )
    }

    /// `predict()`/`augment()`のSome分岐で共有する、`df`に対するout-of-sample予測
    /// （`x`列の抽出→`predict_new_data`呼び出し）。`x`の列名はLogit/Probitと異なり
    /// `fitted`の係数名（`σ`を含まない。結果の`param_names`は末尾に`"sigma"`を足した別の値）の
    /// 先頭`"const"`（`has_intercept`時）を除いた部分（モジュールdocコメント参照）。結果側の
    /// `param_names`ではなく`fitted`の名前を使うので、pickleから復元した状態でも両者の食い違いで
    /// 列数が合わなくならない。
    fn predict_for(&self, target: MarginalEffectsTarget, df: &DataFrame) -> PyResult<Vec<f64>> {
        let has_intercept = self.fitted.input().has_intercept();
        let x_columns = extract_model_x_columns(
            df,
            self.fitted.input().param_names(),
            has_intercept,
            ColumnSource::new("x", "new_data"),
        )?;

        self.fitted
            .predict_new_data(target, &x_columns)
            .map_err(mle_error_to_pyerr)
    }
}

#[pymethods]
impl TobitResult {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        crate::shared::pickle::reduce(slf)
    }

    /// Predicted values for the training data used in `fit()`, or for `new_data`
    /// when given.
    ///
    /// `target` selects which quantity to predict: `"expected_latent"` (`E[y*|x]=x'β`),
    /// `"expected_observed"` (`E[y|x]`, the censoring-adjusted conditional expectation;
    /// default, directly comparable to the observed `y`), `"prob_uncensored"`
    /// (`P(uncensored|x)`), or `"expected_uncensored"` (`E[y|lower<y<upper,x]`, the
    /// expectation restricted to the uncensored region; `E[y|y>0,x]` for a single
    /// lower bound of 0). Independent of `new_data`: the same four targets are
    /// available whether predicting on the training data or new data.
    ///
    /// # Errors
    /// - `target` is not one of the four known values (case-insensitive):
    ///   `ValidationError`
    /// - A required `x` column is missing from `new_data`, cannot be cast to a
    ///   numeric type, or contains missing/NaN/infinite values: `ValidationError`
    ///   (same validation as `fit()`'s column extraction, via `extract_f64_column`).
    #[pyo3(signature = (new_data=None, target="expected_observed".to_string()))]
    fn predict(&self, new_data: Option<&Bound<'_, PyAny>>, target: String) -> PyResult<Vec<f64>> {
        guard(|| {
            let target = parse_marginal_effects_target(&target)?;
            let Some(new_data) = new_data else {
                return self.fitted.predict(target).map_err(mle_error_to_pyerr);
            };

            let selected = self.select_new_data(new_data)?;
            self.predict_for(target, &selected.frame)
        })
    }

    /// The source data (training data, or `new_data` when given) with the predicted
    /// values appended as a new column named `"predicted_{target}"`, using the
    /// lowercased `target` (e.g. `"predicted_expected_observed"` for the default
    /// `target`).
    ///
    /// Same `target`/`new_data`/`include_intercept` semantics as `predict()`, but
    /// returns a polars DataFrame (original columns plus the predicted column, row
    /// order preserved) instead of a bare list of floats (same design as
    /// `LogitResult::augment()`). The column name is `target`-dependent
    /// (unlike Logit/Probit's fixed `"predicted_probability"`) so that `augment()` can be
    /// called once per `target` on the same DataFrame without a name collision.
    ///
    /// # Errors
    /// - Same as `predict()`: `target` unknown, or a required `x` column missing
    ///   from `new_data`, non-numeric, or containing missing/NaN/infinite values:
    ///   `ValidationError`.
    /// - The source data already has a column named `"predicted_{target}"`:
    ///   `ValidationError` (would otherwise silently overwrite it).
    #[pyo3(signature = (new_data=None, target="expected_observed".to_string()))]
    fn augment(
        &self,
        py: Python<'_>,
        new_data: Option<&Bound<'_, PyAny>>,
        target: String,
    ) -> PyResult<Py<PyAny>> {
        guard(|| {
            let target_enum = parse_marginal_effects_target(&target)?;

            let (source, predicted) = match new_data {
                Some(new_data) => {
                    let selected = self.select_new_data(new_data)?;
                    let predicted = self.predict_for(target_enum, &selected.frame)?;
                    (new_data.clone(), predicted)
                }
                None => (
                    self.source_data.bind(py).clone(),
                    self.fitted
                        .predict(target_enum)
                        .map_err(mle_error_to_pyerr)?,
                ),
            };

            let column_name = format!("predicted_{}", target.to_lowercase());
            append_predicted_column(&source, &column_name, predicted)
        })
    }

    /// Marginal effects (`dy/dx`) with delta-method standard errors.
    ///
    /// `target` selects the same four quantities as `predict()` (see its doc); the
    /// formula differs per `target`.
    ///
    /// Independent of `fit()`'s `confidence_level` (re-evaluated here so callers can
    /// use a different confidence level without re-fitting).
    ///
    /// # Errors
    /// - `at` is not one of `"overall"`, `"mean"`, `"median"` (case-insensitive):
    ///   `ValidationError`
    /// - `target` is not one of the four known values (case-insensitive):
    ///   `ValidationError`
    /// - `confidence_level` is outside `(0, 1)`: `ValidationError`
    #[pyo3(signature = (
        at="overall".to_string(),
        target="expected_observed".to_string(),
        confidence_level=0.95,
    ))]
    fn marginal_effects(
        &self,
        py: Python<'_>,
        at: String,
        target: String,
        confidence_level: f64,
    ) -> PyResult<MarginalEffectsResult> {
        guard(|| {
            let at = parse_marginal_effects_at(&at)?;
            let target = parse_marginal_effects_target(&target)?;
            // 推定量だけを使う純Rustの計算なので、GILを手放して走らせる。
            let effects = py
                .detach(|| self.fitted.marginal_effects(at, target, confidence_level))
                .map_err(mle_error_to_pyerr)?;

            Ok(MarginalEffectsResult::from_engine(&effects))
        })
    }

    /// Censoring goodness-of-fit check: for each direction (`lower`/`uncensored`/`upper`)
    /// that applies to this model, compares the observed rate (fraction of training
    /// observations exactly at that boundary) against the model-implied average
    /// probability. Replaces Logit/Probit's `pred_table()` (which is not meaningful for
    /// Tobit's continuous `y`).
    ///
    /// `lower`/`upper` in the returned `CensoringFitCheckResult` are `None` when the
    /// corresponding `TobitOptions.lower`/`upper` was `None` (that direction has no
    /// censoring).
    fn censoring_fit_check(&self) -> PyResult<CensoringFitCheckResult> {
        guard(|| {
            Ok(censoring_fit_check_to_result(
                self.fitted.censoring_fit_check(),
            ))
        })
    }
}

/// `target`文字列（大文字小文字を区別しない）を`engine::nonlinear::tobit::
/// MarginalEffectsTarget`にパースする。`predict`/`marginal_effects`共通
/// （Tobit専用、`parse_marginal_effects_at`とは異なりLogit/Probitには無い概念のため
/// `nonlinear/common.rs`ではなくここに定義する）。
///
/// # Errors
/// `target`が既知の値のいずれでもない: `ValidationError`
fn parse_marginal_effects_target(target: &str) -> PyResult<MarginalEffectsTarget> {
    match target.to_lowercase().as_str() {
        "expected_latent" => Ok(MarginalEffectsTarget::ExpectedLatent),
        "expected_observed" => Ok(MarginalEffectsTarget::ExpectedObserved),
        "prob_uncensored" => Ok(MarginalEffectsTarget::ProbUncensored),
        "expected_uncensored" => Ok(MarginalEffectsTarget::ExpectedUncensored),
        _ => Err(unknown_option_value(
            "target",
            target,
            &[
                "expected_latent",
                "expected_observed",
                "prob_uncensored",
                "expected_uncensored",
            ],
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_marginal_effects_target_accepts_known_lowercase_values() {
        assert!(matches!(
            parse_marginal_effects_target("expected_latent"),
            Ok(MarginalEffectsTarget::ExpectedLatent)
        ));
        assert!(matches!(
            parse_marginal_effects_target("expected_observed"),
            Ok(MarginalEffectsTarget::ExpectedObserved)
        ));
        assert!(matches!(
            parse_marginal_effects_target("prob_uncensored"),
            Ok(MarginalEffectsTarget::ProbUncensored)
        ));
        assert!(matches!(
            parse_marginal_effects_target("expected_uncensored"),
            Ok(MarginalEffectsTarget::ExpectedUncensored)
        ));
    }

    #[test]
    fn parse_marginal_effects_target_returns_validation_error_for_unknown_value() {
        assert!(parse_marginal_effects_target("bogus").is_err());
    }
}
