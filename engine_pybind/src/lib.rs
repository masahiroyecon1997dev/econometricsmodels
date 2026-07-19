mod column_extraction;
mod errors;
mod linear;

use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use errors::{ComputationError, ValidationError};
use linear::ols::{OLSOptions, OLSResult};

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

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_ols, m)?)?;
    m.add_class::<OLSOptions>()?;
    m.add_class::<OLSResult>()?;
    m.add("ValidationError", m.py().get_type::<ValidationError>())?;
    m.add("ComputationError", m.py().get_type::<ComputationError>())?;
    Ok(())
}
