mod column_extraction;
mod errors;
mod iv;
mod linear;
mod nonlinear;
mod option_values;
mod panel;
mod validation;

use pyo3::prelude::*;

use column_extraction::{extract_column_list, extract_column_name, extract_dataframe};
use errors::{ComputationError, ValidationError};
use iv::common::{IVOptions, IVResult};
use linear::ols::{BreuschGodfreyTestOutput, OLSOptions, OLSResult, WhiteTestOutput};
use linear::wls::{WLSOptions, WLSResult};
use nonlinear::common::MarginalEffectsResult;
use nonlinear::logit::{LogitOptions, LogitResult};
use nonlinear::probit::{ProbitOptions, ProbitResult};
use nonlinear::tobit::{
    CensoringFitCategoryResult, CensoringFitCheckResult, TobitOptions, TobitResult,
};
use panel::fe::{FEOptions, FEResult};
use panel::re::{REOptions, REResult};

/// Entry point for OLS estimation.
///
/// Parameters
/// ----------
/// data : polars.DataFrame
///     The input data. Must contain the `y`, `x`, and (if specified) cluster/time columns.
/// y : str
///     Column name of the dependent variable.
/// x : list[str]
///     Column names of the independent variables.
/// options : OLSOptions
///     Estimation options.
#[pyfunction]
fn fit_ols(
    data: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    options: OLSOptions,
) -> PyResult<OLSResult> {
    let data = extract_dataframe(data, "data")?;
    let y = extract_column_name(y, "y")?;
    let x = extract_column_list(x, "x")?;
    linear::ols::fit(data, y, x, &options)
}

/// Entry point for WLS estimation.
///
/// Parameters
/// ----------
/// data : polars.DataFrame
///     The input data. Must contain the `y`, `x`, `weight`, and (if specified)
///     cluster/time columns.
/// y : str
///     Column name of the dependent variable.
/// x : list[str]
///     Column names of the independent variables.
/// weight : str
///     Column name of the analytic weight (must be positive; not a frequency weight).
/// options : WLSOptions
///     Estimation options.
#[pyfunction]
fn fit_wls(
    data: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    weight: &Bound<'_, PyAny>,
    options: WLSOptions,
) -> PyResult<WLSResult> {
    let data = extract_dataframe(data, "data")?;
    let y = extract_column_name(y, "y")?;
    let x = extract_column_list(x, "x")?;
    let weight = extract_column_name(weight, "weight")?;
    linear::wls::fit(data, y, x, weight, &options)
}

/// Entry point for Logit estimation.
///
/// Parameters
/// ----------
/// data : polars.DataFrame
///     The input data. Must contain the `y`, `x`, and (if specified) cluster columns.
/// y : str
///     Column name of the dependent variable. Must be coded as 0.0 or 1.0
///     (binary outcome); any other value raises `ValidationError`.
/// x : list[str]
///     Column names of the independent variables.
/// options : LogitOptions
///     Estimation options.
#[pyfunction]
fn fit_logit(
    data: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    options: LogitOptions,
) -> PyResult<LogitResult> {
    let data = extract_dataframe(data, "data")?;
    let y = extract_column_name(y, "y")?;
    let x = extract_column_list(x, "x")?;
    nonlinear::logit::fit(data, y, x, &options)
}

/// Entry point for Probit estimation.
///
/// Parameters
/// ----------
/// data : polars.DataFrame
///     The input data. Must contain the `y`, `x`, and (if specified) cluster columns.
/// y : str
///     Column name of the dependent variable. Must be coded as 0.0 or 1.0
///     (binary outcome); any other value raises `ValidationError`.
/// x : list[str]
///     Column names of the independent variables.
/// options : ProbitOptions
///     Estimation options.
#[pyfunction]
fn fit_probit(
    data: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    options: ProbitOptions,
) -> PyResult<ProbitResult> {
    let data = extract_dataframe(data, "data")?;
    let y = extract_column_name(y, "y")?;
    let x = extract_column_list(x, "x")?;
    nonlinear::probit::fit(data, y, x, &options)
}

/// Entry point for Tobit estimation.
///
/// Parameters
/// ----------
/// data : polars.DataFrame
///     The input data. Must contain the `y`, `x`, and (if specified) cluster columns.
/// y : str
///     Column name of the dependent variable. May be censored at `options.lower`/
///     `options.upper` (values outside those bounds raise `ValidationError`).
/// x : list[str]
///     Column names of the independent variables.
/// options : TobitOptions
///     Estimation options.
#[pyfunction]
fn fit_tobit(
    data: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    options: TobitOptions,
) -> PyResult<TobitResult> {
    let data = extract_dataframe(data, "data")?;
    let y = extract_column_name(y, "y")?;
    let x = extract_column_list(x, "x")?;
    nonlinear::tobit::fit(data, y, x, &options)
}

/// Entry point for IV estimation (2SLS/GMM).
///
/// Parameters
/// ----------
/// data : polars.DataFrame
///     The input data. Must contain the `y`, `x_exog`, `x_endog`, `instruments`, and
///     (if specified) cluster/time columns.
/// y : str
///     Column name of the dependent variable.
/// x_exog : list[str]
///     Column names of the exogenous independent variables.
/// x_endog : list[str]
///     Column names of the endogenous independent variables.
/// instruments : list[str]
///     Column names of the excluded instruments (must not overlap with `x_exog`).
/// options : IVOptions
///     Estimation options. `options.estimator` selects "2sls" (default) or "gmm".
#[pyfunction]
fn fit_iv(
    data: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    x_exog: &Bound<'_, PyAny>,
    x_endog: &Bound<'_, PyAny>,
    instruments: &Bound<'_, PyAny>,
    options: IVOptions,
) -> PyResult<IVResult> {
    let data = extract_dataframe(data, "data")?;
    let y = extract_column_name(y, "y")?;
    let x_exog = extract_column_list(x_exog, "x_exog")?;
    let x_endog = extract_column_list(x_endog, "x_endog")?;
    let instruments = extract_column_list(instruments, "instruments")?;
    iv::common::fit(data, y, x_exog, x_endog, instruments, &options)
}

/// Entry point for FE (fixed effects panel regression) estimation.
///
/// Parameters
/// ----------
/// data : polars.DataFrame
///     The input data. Must contain the `y`, `x`, `entity`, and (if specified)
///     time/cluster/HAC time columns.
/// y : str
///     Column name of the dependent variable.
/// x : list[str]
///     Column names of the independent variables. Must contain at least one
///     column name.
/// entity : str
///     Column name of the entity (individual/panel unit) identifier.
/// options : FEOptions
///     Estimation options.
#[pyfunction]
fn fit_fe(
    data: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    entity: &Bound<'_, PyAny>,
    options: FEOptions,
) -> PyResult<FEResult> {
    let data = extract_dataframe(data, "data")?;
    let y = extract_column_name(y, "y")?;
    let x = extract_column_list(x, "x")?;
    let entity = extract_column_name(entity, "entity")?;
    panel::fe::fit(data, y, x, entity, &options)
}

/// Entry point for RE (random effects panel regression) estimation.
///
/// Parameters
/// ----------
/// data : polars.DataFrame
///     The input data. Must contain the `y`, `x`, `entity`, and (if specified)
///     time/cluster columns.
/// y : str
///     Column name of the dependent variable.
/// x : list[str]
///     Column names of the independent variables. Must contain at least one
///     column name.
/// entity : str
///     Column name of the entity (individual/panel unit) identifier.
/// options : REOptions
///     Estimation options.
#[pyfunction]
fn fit_re(
    data: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    entity: &Bound<'_, PyAny>,
    options: REOptions,
) -> PyResult<REResult> {
    let data = extract_dataframe(data, "data")?;
    let y = extract_column_name(y, "y")?;
    let x = extract_column_list(x, "x")?;
    let entity = extract_column_name(entity, "entity")?;
    panel::re::fit(data, y, x, entity, &options)
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // `import econometricsmodels` の時点で faer のグローバル並列度を Par::Seq に
    // 固定する。各 `Estimator::fit()` 冒頭でも呼ぶが（`cargo test`
    // との経路統一のため）、ここで一度呼んでおくことで将来 `fit()` 以外の Python
    // 入口が増えても確実に適用される。
    engine::parallelism::ensure_serial();

    m.add_function(wrap_pyfunction!(fit_ols, m)?)?;
    m.add_class::<OLSOptions>()?;
    m.add_class::<OLSResult>()?;
    m.add_class::<WhiteTestOutput>()?;
    m.add_class::<BreuschGodfreyTestOutput>()?;
    m.add_function(wrap_pyfunction!(fit_wls, m)?)?;
    m.add_class::<WLSOptions>()?;
    m.add_class::<WLSResult>()?;
    m.add_function(wrap_pyfunction!(fit_logit, m)?)?;
    m.add_class::<LogitOptions>()?;
    m.add_class::<LogitResult>()?;
    m.add_class::<MarginalEffectsResult>()?;
    m.add_function(wrap_pyfunction!(fit_probit, m)?)?;
    m.add_class::<ProbitOptions>()?;
    m.add_class::<ProbitResult>()?;
    m.add_function(wrap_pyfunction!(fit_tobit, m)?)?;
    m.add_class::<TobitOptions>()?;
    m.add_class::<TobitResult>()?;
    m.add_class::<CensoringFitCategoryResult>()?;
    m.add_class::<CensoringFitCheckResult>()?;
    m.add_function(wrap_pyfunction!(fit_iv, m)?)?;
    m.add_class::<IVOptions>()?;
    m.add_class::<IVResult>()?;
    m.add_function(wrap_pyfunction!(fit_fe, m)?)?;
    m.add_class::<FEOptions>()?;
    m.add_class::<FEResult>()?;
    m.add_function(wrap_pyfunction!(fit_re, m)?)?;
    m.add_class::<REOptions>()?;
    m.add_class::<REResult>()?;
    m.add("ValidationError", m.py().get_type::<ValidationError>())?;
    m.add("ComputationError", m.py().get_type::<ComputationError>())?;
    Ok(())
}
