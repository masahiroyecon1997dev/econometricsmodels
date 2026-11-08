//! IVの弱操作変数診断（Cragg-Donald・Kleibergen-Paap・有効F）の出力型と、`IVResult`の診断メソッドの本体。
//!
//! `IVResult`の`#[pymethods]`は`result.rs`に1つだけ置く（`multiple-pymethods`を使わない）ので、
//! メソッドの中身はここの自由関数（`run_*`）に置き、`result.rs`の各メソッドが`guard`で包んで呼ぶ。
//! 診断は`fit()`では計算しない。結果が保持する元のデータと`y`・`x_exog`・`x_endog`・`instruments`・
//! オプション（`cov_type`・`cluster`・`hac_lags`・`hac_time`）から`IvInput`を作り直して、エンジンの
//! `compute_*`を呼ぶ（`fit()`と同じ`build_iv_input`なので、列の検査・並びは`fit()`と一致する）。

use engine::iv::common::{
    IvInput, PatnaikCriticalValue, StockYogoCriticalValues, compute_cragg_donald,
    compute_effective_f, compute_kleibergen_paap,
};
use engine::linear::ols::CovType;
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use super::error::iv_error_to_pyerr;
use super::input::build_iv_input;
use super::options::IVOptions;
use super::result::IVResult;
use crate::shared::column_extraction::select_columns;

/// Raw result of `IVResults.cragg_donald()`; the Python package wraps it in its
/// `CraggDonaldResult` dataclass.
///
/// `tsls_bias` and `tsls_size` are lists of `(level, critical value)` pairs from the Stock-Yogo
/// tables, or `None` when the number of endogenous variables and excluded instruments is
/// outside the tabulated range.
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct CraggDonaldOutput {
    #[pyo3(get)]
    pub statistic: f64,
    #[pyo3(get)]
    pub n_endog: usize,
    #[pyo3(get)]
    pub n_excluded: usize,
    #[pyo3(get)]
    pub tsls_bias: Option<Vec<(f64, f64)>>,
    #[pyo3(get)]
    pub tsls_size: Option<Vec<(f64, f64)>>,
}

/// Raw result of `IVResults.kleibergen_paap()`; the Python package wraps it in its
/// `KleibergenPaapResult` dataclass.
///
/// The rk LM statistic (with its degrees of freedom and p-value) and the rk Wald F statistic
/// are defined independently, so either may be `None`.
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct KleibergenPaapOutput {
    #[pyo3(get)]
    pub rk_lm_statistic: Option<f64>,
    #[pyo3(get)]
    pub rk_lm_df: Option<usize>,
    #[pyo3(get)]
    pub rk_lm_p_value: Option<f64>,
    #[pyo3(get)]
    pub rk_wald_f: Option<f64>,
    #[pyo3(get)]
    pub n_endog: usize,
    #[pyo3(get)]
    pub n_excluded: usize,
}

/// Raw result of `IVResults.effective_f()`; the Python package wraps it in its
/// `EffectiveFResult` dataclass.
///
/// `simplified_critical_values` and `critical_values` are lists of `(tau, critical value)`
/// pairs; `critical_values` and `bias_bound` are `None` when the generalized bound is not
/// defined.
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct EffectiveFOutput {
    #[pyo3(get)]
    pub statistic: f64,
    #[pyo3(get)]
    pub n_excluded: usize,
    #[pyo3(get)]
    pub alpha: f64,
    #[pyo3(get)]
    pub simplified_critical_values: Vec<(f64, f64)>,
    #[pyo3(get)]
    pub critical_values: Option<Vec<(f64, f64)>>,
    #[pyo3(get)]
    pub bias_bound: Option<f64>,
}

/// `fit()`に渡したデータから、`fit()`と同じ手順で`IvInput`と`CovType`を作り直す材料。
///
/// 元のデータの取り込み（GILが要る）までをここで済ませ、以降はPythonオブジェクトに触れない。
struct Rebuilt {
    frame: DataFrame,
    y: String,
    x_exog: Vec<String>,
    x_endog: Vec<String>,
    instruments: Vec<String>,
    options: IVOptions,
}

impl Rebuilt {
    fn new(model: &IVResult, py: Python<'_>) -> PyResult<Self> {
        let source = model.retained_source(py)?;
        let options: IVOptions = model.retained_options()?;
        let mut columns = vec![model.dep_var_name.clone()];
        columns.extend(model.x_exog.iter().cloned());
        columns.extend(model.x_endog.iter().cloned());
        columns.extend(model.instruments.iter().cloned());
        columns.extend(options.data_columns());
        let frame = select_columns(&source, "data", &columns)?.frame;
        Ok(Self {
            frame,
            y: model.dep_var_name.clone(),
            x_exog: model.x_exog.clone(),
            x_endog: model.x_endog.clone(),
            instruments: model.instruments.clone(),
            options,
        })
    }

    /// 診断の計算（`compute`）に`IvInput`と`cov_type`を渡す。
    fn with_input<T>(self, compute: impl FnOnce(&IvInput, &CovType) -> PyResult<T>) -> PyResult<T> {
        let built = build_iv_input(
            &self.frame,
            self.y,
            self.x_exog,
            self.x_endog,
            self.instruments,
            &self.options,
        )?;
        compute(&built.input, &built.cov_type)
    }
}

fn pairs(levels: &[f64], values: &[f64]) -> Vec<(f64, f64)> {
    levels.iter().copied().zip(values.iter().copied()).collect()
}

fn stock_yogo(row: Option<StockYogoCriticalValues>) -> Option<Vec<(f64, f64)>> {
    row.map(|row| pairs(&row.levels, &row.values))
}

fn patnaik(values: &[PatnaikCriticalValue]) -> Vec<(f64, f64)> {
    values.iter().map(|v| (v.tau, v.critical_value)).collect()
}

pub(super) fn run_cragg_donald(model: &IVResult, py: Python<'_>) -> PyResult<CraggDonaldOutput> {
    let rebuilt = Rebuilt::new(model, py)?;
    py.detach(|| {
        rebuilt.with_input(|input, _| {
            let cd = compute_cragg_donald(input).map_err(iv_error_to_pyerr)?;
            Ok(CraggDonaldOutput {
                statistic: cd.statistic,
                n_endog: cd.n_endog,
                n_excluded: cd.n_excluded,
                tsls_bias: stock_yogo(cd.tsls_bias),
                tsls_size: stock_yogo(cd.tsls_size),
            })
        })
    })
}

pub(super) fn run_kleibergen_paap(
    model: &IVResult,
    py: Python<'_>,
) -> PyResult<KleibergenPaapOutput> {
    let rebuilt = Rebuilt::new(model, py)?;
    py.detach(|| {
        rebuilt.with_input(|input, cov_type| {
            let kp = compute_kleibergen_paap(input, cov_type).map_err(iv_error_to_pyerr)?;
            Ok(KleibergenPaapOutput {
                rk_lm_statistic: kp.rk_lm.map(|lm| lm.statistic),
                rk_lm_df: kp.rk_lm.map(|lm| lm.df),
                rk_lm_p_value: kp.rk_lm.map(|lm| lm.p_value()),
                rk_wald_f: kp.rk_wald_f,
                n_endog: kp.n_endog,
                n_excluded: kp.n_excluded,
            })
        })
    })
}

pub(super) fn run_effective_f(
    model: &IVResult,
    py: Python<'_>,
    alpha: f64,
) -> PyResult<EffectiveFOutput> {
    let rebuilt = Rebuilt::new(model, py)?;
    py.detach(|| {
        rebuilt.with_input(|input, cov_type| {
            let eff = compute_effective_f(input, cov_type, alpha).map_err(iv_error_to_pyerr)?;
            Ok(EffectiveFOutput {
                statistic: eff.statistic,
                n_excluded: eff.n_instruments,
                alpha: eff.alpha,
                simplified_critical_values: patnaik(&eff.simplified_tsls),
                critical_values: eff.tsls.as_ref().map(|values| patnaik(values)),
                bias_bound: eff.bias_bound,
            })
        })
    })
}
