mod column_extraction;
mod errors;
mod linear;
mod nonlinear;
mod validation;

use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use errors::{ComputationError, ValidationError};
use linear::ols::{OLSOptions, OLSResult};
use linear::wls::WLSResult;
use nonlinear::logit::{LogitOptions, LogitResult, MarginalEffectsResult};

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
    data: PyDataFrame,
    y: String,
    x: Vec<String>,
    options: OLSOptions,
) -> PyResult<OLSResult> {
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
/// options : OLSOptions
///     Estimation options. `WLS` reuses `OLSOptions` rather than defining a separate
///     options type (`docs/planning/specs/wls-api-design.md` section 3).
#[pyfunction]
fn fit_wls(
    data: PyDataFrame,
    y: String,
    x: Vec<String>,
    weight: String,
    options: OLSOptions,
) -> PyResult<WLSResult> {
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
    data: PyDataFrame,
    y: String,
    x: Vec<String>,
    options: LogitOptions,
) -> PyResult<LogitResult> {
    nonlinear::logit::fit(data, y, x, &options)
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_ols, m)?)?;
    m.add_class::<OLSOptions>()?;
    m.add_class::<OLSResult>()?;
    m.add_function(wrap_pyfunction!(fit_wls, m)?)?;
    m.add_class::<WLSResult>()?;
    m.add_function(wrap_pyfunction!(fit_logit, m)?)?;
    m.add_class::<LogitOptions>()?;
    m.add_class::<LogitResult>()?;
    m.add_class::<MarginalEffectsResult>()?;
    m.add("ValidationError", m.py().get_type::<ValidationError>())?;
    m.add("ComputationError", m.py().get_type::<ComputationError>())?;
    Ok(())
}
