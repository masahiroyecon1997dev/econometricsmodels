//! IVの弱操作変数診断（Cragg-Donald・Kleibergen-Paap・有効F）の出力型と、`IVResult`の診断メソッドの本体。
//!
//! `IVResult`の`#[pymethods]`は`result.rs`に1つだけ置く（`multiple-pymethods`を使わない）ので、
//! メソッドの中身はここの自由関数（`run_*`）に置き、`result.rs`の各メソッドが`guard`で包んで呼ぶ。
//! 診断は`fit()`では計算しない。結果が保持する元のデータと`y`・`x_exog`・`x_endog`・`instruments`・
//! オプション（`cov_type`・`cluster`・`hac_lags`・`hac_time`）から`IvInput`を作り直して、エンジンの
//! `compute_*`を呼ぶ（`fit()`と同じ`build_iv_input`なので、列の検査・並びは`fit()`と一致する）。

use engine::iv::common::{
    CraggDonald, EffectiveF, IvError, IvInput, KleibergenPaap, PatnaikCriticalValue,
    StockYogoCriticalValues, compute_cragg_donald, compute_effective_f, compute_kleibergen_paap,
};
use engine::iv::gmm::{
    CStatisticTarget, GmmType, WeightType, compute_c_statistic, weight_type_for_cov_type,
};
use engine::linear::ols::CovType;
use engine::shared::error::CommonError;
use engine::shared::error_catalog::{ErrorCode, ParamValue};
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use super::error::iv_error_to_pyerr;
use super::input::{BuiltIvInput, build_iv_input};
use super::options::IVOptions;
use super::result::IVResult;
use crate::shared::column_extraction::{extract_column_list, select_columns};
use crate::shared::errors::user_error;
use crate::shared::validation::{validate_no_duplicate_within_role, validate_role_non_empty};

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

/// Raw result of `IVResults.c_stat()`; the Python package wraps it in its `CStatisticResult`
/// dataclass.
///
/// `variables` and `instruments` are the tested names, in the order of the model's
/// `x_endog` and `instruments`.
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct CStatisticOutput {
    #[pyo3(get)]
    pub statistic: f64,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub df: usize,
    #[pyo3(get)]
    pub j_with_tested: f64,
    #[pyo3(get)]
    pub j_without_tested: f64,
    #[pyo3(get)]
    pub variables: Vec<String>,
    #[pyo3(get)]
    pub instruments: Vec<String>,
}

/// `fit()`に渡したデータから、`fit()`と同じ手順で入力一式（`IvInput`・`CovType`・GMMの重みの設定）を
/// 作り直す材料。
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
        self.with_built(|built| compute(&built.input, &built.cov_type))
    }

    /// 診断の計算（`compute`）に、`fit()`と同じ手順で作った入力一式（GMMの重みの設定を含む）を渡す。
    fn with_built<T>(self, compute: impl FnOnce(&BuiltIvInput) -> PyResult<T>) -> PyResult<T> {
        let built = build_iv_input(
            &self.frame,
            self.y,
            self.x_exog,
            self.x_endog,
            self.instruments,
            &self.options,
        )?;
        compute(&built)
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

impl CraggDonaldOutput {
    /// エンジンの結果から作る（FEIVの診断も使う）。
    pub(crate) fn from_engine(cd: CraggDonald) -> Self {
        Self {
            statistic: cd.statistic,
            n_endog: cd.n_endog,
            n_excluded: cd.n_excluded,
            tsls_bias: stock_yogo(cd.tsls_bias),
            tsls_size: stock_yogo(cd.tsls_size),
        }
    }
}

impl KleibergenPaapOutput {
    /// エンジンの結果から作る（FEIVの診断も使う）。
    pub(crate) fn from_engine(kp: KleibergenPaap) -> Self {
        Self {
            rk_lm_statistic: kp.rk_lm.map(|lm| lm.statistic),
            rk_lm_df: kp.rk_lm.map(|lm| lm.df),
            rk_lm_p_value: kp.rk_lm.map(|lm| lm.p_value()),
            rk_wald_f: kp.rk_wald_f,
            n_endog: kp.n_endog,
            n_excluded: kp.n_excluded,
        }
    }
}

impl EffectiveFOutput {
    /// エンジンの結果から作る（FEIVの診断も使う）。
    pub(crate) fn from_engine(eff: EffectiveF) -> Self {
        Self {
            statistic: eff.statistic,
            n_excluded: eff.n_instruments,
            alpha: eff.alpha,
            simplified_critical_values: patnaik(&eff.simplified_tsls),
            critical_values: eff.tsls.as_ref().map(|values| patnaik(values)),
            bias_bound: eff.bias_bound,
        }
    }
}

pub(super) fn run_cragg_donald(model: &IVResult, py: Python<'_>) -> PyResult<CraggDonaldOutput> {
    let rebuilt = Rebuilt::new(model, py)?;
    py.detach(|| {
        rebuilt.with_input(|input, _| {
            let cd = compute_cragg_donald(input).map_err(iv_error_to_pyerr)?;
            Ok(CraggDonaldOutput::from_engine(cd))
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
            Ok(KleibergenPaapOutput::from_engine(kp))
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
            Ok(EffectiveFOutput::from_engine(eff))
        })
    })
}

/// 検定する名前を`names`（モデルの列名）の位置にする（昇順）。`names`に無い名前は`ValidationError`。
fn positions_of(
    tested: &[String],
    names: &[String],
    argument: &'static str,
    role: &'static str,
) -> PyResult<Vec<usize>> {
    let mut positions = tested
        .iter()
        .map(|name| {
            names
                .iter()
                .position(|candidate| candidate == name)
                .ok_or_else(|| {
                    user_error(
                        ErrorCode::CStatisticUnknownVariable,
                        vec![
                            ("name", ParamValue::from(name.as_str())),
                            ("argument", ParamValue::from(argument)),
                            ("role", ParamValue::from(role)),
                        ],
                    )
                })
        })
        .collect::<PyResult<Vec<usize>>>()?;
    positions.sort_unstable();
    Ok(positions)
}

/// 引数の型の検査（`list[str]`であること）。指定されたときだけ。値の検査は[`checked_names`]。
fn extract_names(
    value: Option<&Bound<'_, PyAny>>,
    argument: &'static str,
) -> PyResult<Option<Vec<String>>> {
    value
        .map(|value| extract_column_list(value, argument))
        .transpose()
}

/// 名前の一覧の値の検査（空の一覧と重複は`ValidationError`）。指定が無ければ空。
fn checked_names(names: Option<Vec<String>>, argument: &'static str) -> PyResult<Vec<String>> {
    let Some(names) = names else {
        return Ok(Vec::new());
    };
    validate_role_non_empty(argument, &names)?;
    validate_no_duplicate_within_role(argument, &names)?;
    Ok(names)
}

/// C統計量の重み: GMM（`two_step`・`iterated`）は`gmm_weight_type`の重み、2SLSと1段階GMM
/// （重み行列を持たない）は`cov_type`の系統の重み。
fn c_statistic_weight(built: &BuiltIvInput) -> WeightType {
    match built.plan.gmm.as_ref().map(|setup| &setup.gmm_type) {
        Some(GmmType::TwoStep { weight } | GmmType::Iterated { weight, .. }) => weight.clone(),
        // 重み行列を持たない（2SLSと1段階GMM）。
        Some(GmmType::OneStep) | None => weight_type_for_cov_type(&built.cov_type),
    }
}

pub(super) fn run_c_stat(
    model: &IVResult,
    py: Python<'_>,
    variables: Option<&Bound<'_, PyAny>>,
    instruments: Option<&Bound<'_, PyAny>>,
) -> PyResult<CStatisticOutput> {
    // 型の検査を両方の引数で先に済ませ、そのあとで値（空・重複・名前）を検査する
    // （`docs/guide/validation.md`の報告順）。
    let variables = extract_names(variables, "variables")?;
    let instruments = extract_names(instruments, "instruments")?;
    if variables.is_none() && instruments.is_none() {
        return Err(iv_error_to_pyerr(IvError::Common(
            CommonError::CStatisticNothingToTest,
        )));
    }
    let variables = checked_names(variables, "variables")?;
    let instruments = checked_names(instruments, "instruments")?;
    let endog_positions = positions_of(&variables, &model.x_endog, "variables", "x_endog")?;
    let instrument_positions = positions_of(
        &instruments,
        &model.instruments,
        "instruments",
        "instruments",
    )?;
    let target = CStatisticTarget {
        endog: endog_positions.clone(),
        instruments: instrument_positions.clone(),
    };
    let tested_endog: Vec<String> = endog_positions
        .iter()
        .map(|&i| model.x_endog[i].clone())
        .collect();
    let tested_instruments: Vec<String> = instrument_positions
        .iter()
        .map(|&i| model.instruments[i].clone())
        .collect();

    let rebuilt = Rebuilt::new(model, py)?;
    py.detach(|| {
        rebuilt.with_built(|built| {
            let weight = c_statistic_weight(built);
            let c =
                compute_c_statistic(&built.input, &target, &weight).map_err(iv_error_to_pyerr)?;
            Ok(CStatisticOutput {
                statistic: c.statistic,
                p_value: c.p_value,
                df: c.df,
                j_with_tested: c.j_with_tested,
                j_without_tested: c.j_without_tested,
                variables: tested_endog,
                instruments: tested_instruments,
            })
        })
    })
}
