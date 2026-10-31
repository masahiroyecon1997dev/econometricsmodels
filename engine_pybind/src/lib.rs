mod iv;
mod linear;
mod nonlinear;
mod panel;
mod shared;

use pyo3::prelude::*;

use iv::common::{IVOptions, IVResult};
use linear::ols::{
    BreuschGodfreyTestOutput, BreuschPaganTestOutput, OLSOptions, OLSResult, WhiteTestOutput,
};
use linear::wls::{WLSOptions, WLSResult};
use nonlinear::common::MarginalEffectsResult;
use nonlinear::logit::{LogitOptions, LogitResult};
use nonlinear::probit::{ProbitOptions, ProbitResult};
use nonlinear::tobit::{
    CensoringFitCategoryResult, CensoringFitCheckResult, TobitOptions, TobitResult,
};
use panel::fe::{FEOptions, FEResult};
use panel::re::{REOptions, REResult};
use shared::catalog_export::error_catalog;
#[cfg(debug_assertions)]
use shared::catalog_export::{panic_for_test, raise_error_for_test};
use shared::column_extraction::{
    check_dataframe, extract_column_list, extract_column_name, select_columns,
};
use shared::errors::{ComputationError, ValidationError, guard};
use shared::option_values::extract_options;

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
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    options: &Bound<'_, PyAny>,
) -> PyResult<OLSResult> {
    guard(|| {
        check_dataframe(data, "data")?;
        let y = extract_column_name(y, "y")?;
        let x = extract_column_list(x, "x")?;
        let options: OLSOptions = extract_options(options, "options")?;
        let mut columns: Vec<String> = vec![y.clone()];
        columns.extend(x.iter().cloned());
        columns.extend(options.data_columns());
        let data = select_columns(data, "data", &columns)?;
        linear::ols::fit(py, data, y, x, &options)
    })
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
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    weight: &Bound<'_, PyAny>,
    options: &Bound<'_, PyAny>,
) -> PyResult<WLSResult> {
    guard(|| {
        check_dataframe(data, "data")?;
        let y = extract_column_name(y, "y")?;
        let x = extract_column_list(x, "x")?;
        let weight = extract_column_name(weight, "weight")?;
        let options: WLSOptions = extract_options(options, "options")?;
        let mut columns: Vec<String> = vec![y.clone(), weight.clone()];
        columns.extend(x.iter().cloned());
        columns.extend(options.data_columns());
        let data = select_columns(data, "data", &columns)?;
        linear::wls::fit(py, data, y, x, weight, &options)
    })
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
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    options: &Bound<'_, PyAny>,
) -> PyResult<LogitResult> {
    guard(|| {
        check_dataframe(data, "data")?;
        let y = extract_column_name(y, "y")?;
        let x = extract_column_list(x, "x")?;
        let options: LogitOptions = extract_options(options, "options")?;
        let mut columns: Vec<String> = vec![y.clone()];
        columns.extend(x.iter().cloned());
        columns.extend(options.data_columns());
        let data = select_columns(data, "data", &columns)?;
        nonlinear::logit::fit(py, data, y, x, &options)
    })
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
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    options: &Bound<'_, PyAny>,
) -> PyResult<ProbitResult> {
    guard(|| {
        check_dataframe(data, "data")?;
        let y = extract_column_name(y, "y")?;
        let x = extract_column_list(x, "x")?;
        let options: ProbitOptions = extract_options(options, "options")?;
        let mut columns: Vec<String> = vec![y.clone()];
        columns.extend(x.iter().cloned());
        columns.extend(options.data_columns());
        let data = select_columns(data, "data", &columns)?;
        nonlinear::probit::fit(py, data, y, x, &options)
    })
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
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    options: &Bound<'_, PyAny>,
) -> PyResult<TobitResult> {
    guard(|| {
        check_dataframe(data, "data")?;
        let y = extract_column_name(y, "y")?;
        let x = extract_column_list(x, "x")?;
        let options: TobitOptions = extract_options(options, "options")?;
        let mut columns: Vec<String> = vec![y.clone()];
        columns.extend(x.iter().cloned());
        columns.extend(options.data_columns());
        let data = select_columns(data, "data", &columns)?;
        nonlinear::tobit::fit(py, data, y, x, &options)
    })
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
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    x_exog: &Bound<'_, PyAny>,
    x_endog: &Bound<'_, PyAny>,
    instruments: &Bound<'_, PyAny>,
    options: &Bound<'_, PyAny>,
) -> PyResult<IVResult> {
    guard(|| {
        check_dataframe(data, "data")?;
        let y = extract_column_name(y, "y")?;
        let x_exog = extract_column_list(x_exog, "x_exog")?;
        let x_endog = extract_column_list(x_endog, "x_endog")?;
        let instruments = extract_column_list(instruments, "instruments")?;
        let options: IVOptions = extract_options(options, "options")?;
        let mut columns: Vec<String> = vec![y.clone()];
        columns.extend(x_exog.iter().cloned());
        columns.extend(x_endog.iter().cloned());
        columns.extend(instruments.iter().cloned());
        columns.extend(options.data_columns());
        let data = select_columns(data, "data", &columns)?;
        iv::common::fit(py, data, y, x_exog, x_endog, instruments, &options)
    })
}

/// Entry point for FE (fixed effects panel regression) estimation.
///
/// Parameters
/// ----------
/// data : polars.DataFrame
///     The input data. Must contain the `y`, `x`, `entity`, and (if specified)
///     time/cluster/dk_time columns.
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
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    entity: &Bound<'_, PyAny>,
    options: &Bound<'_, PyAny>,
) -> PyResult<FEResult> {
    guard(|| {
        check_dataframe(data, "data")?;
        let y = extract_column_name(y, "y")?;
        let x = extract_column_list(x, "x")?;
        let entity = extract_column_name(entity, "entity")?;
        let options: FEOptions = extract_options(options, "options")?;
        let mut columns: Vec<String> = vec![y.clone(), entity.clone()];
        columns.extend(x.iter().cloned());
        columns.extend(options.data_columns());
        let data = select_columns(data, "data", &columns)?;
        panel::fe::fit(py, data, y, x, entity, &options)
    })
}

/// Entry point for RE (random effects panel regression) estimation.
///
/// Parameters
/// ----------
/// data : polars.DataFrame
///     The input data. Must contain the `y`, `x`, `entity`, and (if specified)
///     time/cluster/dk_time columns.
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
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    entity: &Bound<'_, PyAny>,
    options: &Bound<'_, PyAny>,
) -> PyResult<REResult> {
    guard(|| {
        check_dataframe(data, "data")?;
        let y = extract_column_name(y, "y")?;
        let x = extract_column_list(x, "x")?;
        let entity = extract_column_name(entity, "entity")?;
        let options: REOptions = extract_options(options, "options")?;
        let mut columns: Vec<String> = vec![y.clone(), entity.clone()];
        columns.extend(x.iter().cloned());
        columns.extend(options.data_columns());
        let data = select_columns(data, "data", &columns)?;
        panel::re::fit(py, data, y, x, entity, &options)
    })
}

/// `_restore_pickled`の種別ごとの分岐。種別名は各型の`PickleState::KIND`と照合するので、
/// 綴りを別に書く必要がない（型を足してここに足し忘れても、`tests/cross_method/test_pickle.py`が検出する）。
macro_rules! restore_by_kind {
    ($kind:expr, $py:expr, $state:expr, $unknown:expr; $($ty:ty),+ $(,)?) => {
        match $kind {
            $( kind if kind == <$ty as shared::pickle::PickleState>::KIND => {
                shared::pickle::restore::<$ty>($py, $state)
            } )+
            _ => Err($unknown),
        }
    };
}

/// Restore function returned by `__reduce__` of the options and result classes.
///
/// Internal entry point called by `pickle` and `copy`; not meant to be called directly.
/// Raises `ValidationError` (`PICKLE_FORMAT_MISMATCH`) when `kind` is unknown, `version` differs
/// from the current format, or `state` does not have the fields and types of that class.
#[pyfunction]
fn _restore_pickled(
    py: Python<'_>,
    kind: &Bound<'_, PyAny>,
    version: &Bound<'_, PyAny>,
    state: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    guard(|| {
        // 引数の型が違うときも`PICKLE_FORMAT_MISMATCH`にする（壊れたpickleの報告を1つの`code`にそろえる）。
        let kind_name = kind
            .extract::<String>()
            .unwrap_or_else(|_| "(unknown)".to_string());
        let mismatch = || shared::pickle::format_mismatch(&kind_name);
        if version.extract::<u32>().ok() != Some(shared::pickle::FORMAT_VERSION) {
            return Err(mismatch());
        }
        let state = state
            .cast::<pyo3::types::PyDict>()
            .map_err(|_| mismatch())?;
        restore_by_kind!(
            kind_name.as_str(), py, state, mismatch();
            OLSOptions, OLSResult, WLSOptions, WLSResult, LogitOptions, LogitResult, MarginalEffectsResult,
            ProbitOptions, ProbitResult, TobitOptions, TobitResult, CensoringFitCategoryResult, CensoringFitCheckResult,
            IVOptions, IVResult, FEOptions, FEResult, REOptions, REResult,
        )
    })
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // `import econometricsmodels` の時点で faer のグローバル並列度を Par::Seq に
    // 固定する。各 `Estimator::fit()` 冒頭でも呼ぶが（`cargo test`
    // との経路統一のため）、ここで一度呼んでおくことで将来 `fit()` 以外の Python
    // 入口が増えても確実に適用される。
    engine::shared::parallelism::ensure_serial();

    m.add_function(wrap_pyfunction!(fit_ols, m)?)?;
    m.add_class::<OLSOptions>()?;
    m.add_class::<OLSResult>()?;
    m.add_class::<WhiteTestOutput>()?;
    m.add_class::<BreuschPaganTestOutput>()?;
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
    m.add_function(wrap_pyfunction!(error_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(_restore_pickled, m)?)?;
    // テスト専用の入口はデバッグビルドにだけ含める（配布するreleaseビルドには入れない）。
    #[cfg(debug_assertions)]
    {
        m.add_function(wrap_pyfunction!(raise_error_for_test, m)?)?;
        m.add_function(wrap_pyfunction!(panic_for_test, m)?)?;
    }
    m.add("ValidationError", m.py().get_type::<ValidationError>())?;
    m.add("ComputationError", m.py().get_type::<ComputationError>())?;
    Ok(())
}
