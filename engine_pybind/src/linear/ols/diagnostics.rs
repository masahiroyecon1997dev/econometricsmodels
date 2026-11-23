//! OLSの事後診断（White・Breusch-Pagan・Breusch-Godfrey）の出力型と、`OLSResult`の診断メソッドの本体。
//!
//! `OLSResult`の`#[pymethods]`は`result.rs`に1つだけ置く（`multiple-pymethods`を使わない）ので、
//! メソッドの中身はここの自由関数（`run_*`）に置き、`result.rs`の各メソッドが`guard`で包んで呼ぶ。

use crate::linear::common::least_squares_error_to_pyerr;
use crate::shared::column_error::ColumnSource;
use crate::shared::column_extraction::{
    extract_column_list, extract_f64_columns, extract_model_x_columns, extract_time_order_ranks,
    select_columns, x_column_names,
};
use crate::shared::option_values::{extract_strict_int, extract_strict_text, parse_choice};
use crate::shared::validation::{validate_no_duplicate_within_role, validate_role_non_empty};
use engine::linear::diagnostics::{
    AuxRegressionTest, breusch_godfrey_test, breusch_pagan_test, white_test,
};
use pyo3::prelude::*;

use super::result::OLSResult;

/// Raw result of `OLSResult.white_test()`; the Python package wraps it in its
/// `WhiteTestResult` dataclass.
///
/// `statistic`/`p_value`/`df`/`df_denom`/`distribution` describe the selected version
/// (`"lm"` or `"f"`). `aux_terms` lists the auxiliary-regression terms actually used (the
/// first entry is always the auxiliary regression's constant, `"const"`), `dropped_terms`
/// the terms removed because they were constant or numerically identical to an earlier term.
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct WhiteTestOutput {
    #[pyo3(get)]
    pub statistic: f64,
    #[pyo3(get)]
    pub p_value: f64,
    /// Degrees of freedom of the LM test (chi-squared), or the numerator degrees of
    /// freedom of the F test.
    #[pyo3(get)]
    pub df: usize,
    /// Denominator degrees of freedom of the F test (`None` for the LM version).
    #[pyo3(get)]
    pub df_denom: Option<usize>,
    /// `"chi2"` for the LM version, `"f"` for the F version.
    #[pyo3(get)]
    pub distribution: String,
    #[pyo3(get)]
    pub aux_terms: Vec<String>,
    #[pyo3(get)]
    pub dropped_terms: Vec<String>,
}

/// Raw result of `OLSResult.breusch_pagan_test()`; the Python package wraps it in its
/// `BreuschPaganTestResult` dataclass.
///
/// `statistic`/`p_value`/`df`/`df_denom`/`distribution` describe the selected version
/// (`"lm"` or `"f"`). `aux_terms` lists the auxiliary-regression terms actually used (the
/// first entry is always the auxiliary regression's constant, `"const"`), `dropped_terms`
/// the variables removed because they were constant or numerically identical to an earlier
/// one.
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct BreuschPaganTestOutput {
    #[pyo3(get)]
    pub statistic: f64,
    #[pyo3(get)]
    pub p_value: f64,
    /// Degrees of freedom of the LM test (chi-squared), or the numerator degrees of
    /// freedom of the F test.
    #[pyo3(get)]
    pub df: usize,
    /// Denominator degrees of freedom of the F test (`None` for the LM version).
    #[pyo3(get)]
    pub df_denom: Option<usize>,
    /// `"chi2"` for the LM version, `"f"` for the F version.
    #[pyo3(get)]
    pub distribution: String,
    #[pyo3(get)]
    pub aux_terms: Vec<String>,
    #[pyo3(get)]
    pub dropped_terms: Vec<String>,
}

/// 診断検定（`white_test`・`breusch_pagan_test`・`breusch_godfrey_test`）の`statistic`引数で選ぶ検定統計量の版。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatisticVersion {
    Lm,
    F,
}

/// 診断検定の`statistic`引数（`"lm"`/`"f"`、大文字小文字を区別しない）をパースする。
///
/// # Errors
/// `statistic`が`"lm"`でも`"f"`でもない: `ValidationError`
fn parse_statistic_version(statistic: &str) -> PyResult<StatisticVersion> {
    parse_choice(
        "statistic",
        statistic,
        &[("lm", StatisticVersion::Lm), ("f", StatisticVersion::F)],
    )
}

/// 選んだ版（LM/F）の統計量・p値・分母自由度・分布名を取り出す（White検定・Breusch-Pagan検定
/// の結果の詰め替えで共通）。
fn select_statistic(
    result: &AuxRegressionTest,
    version: StatisticVersion,
) -> (f64, f64, Option<usize>, String) {
    match version {
        StatisticVersion::F => (
            result.f_statistic,
            result.f_p_value,
            Some(result.f_df_denom),
            "f".to_string(),
        ),
        StatisticVersion::Lm => (
            result.lm_statistic,
            result.lm_p_value,
            None,
            "chi2".to_string(),
        ),
    }
}

/// Raw result of `OLSResult.breusch_godfrey_test()`; the Python package wraps it in its
/// `BreuschGodfreyTestResult` dataclass.
///
/// `statistic`/`p_value`/`df`/`df_denom`/`distribution` describe the selected version
/// (`"lm"` or `"f"`); `nlags` is the number of residual lags used.
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct BreuschGodfreyTestOutput {
    #[pyo3(get)]
    pub statistic: f64,
    #[pyo3(get)]
    pub p_value: f64,
    /// Degrees of freedom of the LM test (chi-squared), or the numerator degrees of
    /// freedom of the F test. Equal to `nlags`.
    #[pyo3(get)]
    pub df: usize,
    /// Denominator degrees of freedom of the F test (`None` for the LM version).
    #[pyo3(get)]
    pub df_denom: Option<usize>,
    /// `"chi2"` for the LM version, `"f"` for the F version.
    #[pyo3(get)]
    pub distribution: String,
    #[pyo3(get)]
    pub nlags: usize,
}

pub(super) fn run_white_test(
    model: &OLSResult,
    py: Python<'_>,
    statistic: &str,
) -> PyResult<WhiteTestOutput> {
    let version = parse_statistic_version(statistic)?;
    let source = model.retained_source(py, "white_test")?;

    let x_names = x_column_names(&model.param_names, model.has_intercept);
    let df = select_columns(&source, "data", x_names)?.frame;
    // 列の取り込み（GILが要る）の後は、Pythonオブジェクトに触れない。
    py.detach(|| {
        let x_columns = extract_f64_columns(&df, x_names, ColumnSource::new("x", "data"))?;
        let result = white_test(&x_columns, x_names, &model.residuals)
            .map_err(least_squares_error_to_pyerr)?;

        let (statistic, p_value, df_denom, distribution) = select_statistic(&result, version);
        Ok(WhiteTestOutput {
            statistic,
            p_value,
            df: result.df,
            df_denom,
            distribution,
            aux_terms: result.aux_terms,
            dropped_terms: result.dropped_terms,
        })
    })
}

pub(super) fn run_breusch_pagan_test(
    model: &OLSResult,
    py: Python<'_>,
    variables: Option<&Bound<'_, PyAny>>,
    statistic: &str,
) -> PyResult<BreuschPaganTestOutput> {
    let version = parse_statistic_version(statistic)?;
    let variables = variables
        .map(|ob| extract_column_list(ob, "variables"))
        .transpose()?;
    let source = model.retained_source(py, "breusch_pagan_test")?;

    let z_names = match variables.as_deref() {
        Some(names) => {
            validate_role_non_empty("variables", names)?;
            validate_no_duplicate_within_role("variables", names)?;
            names
        }
        None => x_column_names(&model.param_names, model.has_intercept),
    };
    let z_argument = if variables.is_some() {
        "variables"
    } else {
        "x"
    };
    let df = select_columns(&source, "data", z_names)?.frame;
    // 列の取り込み（GILが要る）の後は、Pythonオブジェクトに触れない。
    py.detach(|| {
        let z_columns = extract_f64_columns(&df, z_names, ColumnSource::new(z_argument, "data"))?;
        let result = breusch_pagan_test(&z_columns, z_names, &model.residuals)
            .map_err(least_squares_error_to_pyerr)?;

        let (statistic, p_value, df_denom, distribution) = select_statistic(&result, version);
        Ok(BreuschPaganTestOutput {
            statistic,
            p_value,
            df: result.df,
            df_denom,
            distribution,
            aux_terms: result.aux_terms,
            dropped_terms: result.dropped_terms,
        })
    })
}

pub(super) fn run_breusch_godfrey_test(
    model: &OLSResult,
    py: Python<'_>,
    time: &Bound<'_, PyAny>,
    nlags: &Bound<'_, PyAny>,
    statistic: Option<&Bound<'_, PyAny>>,
) -> PyResult<BreuschGodfreyTestOutput> {
    let time = extract_strict_text(time, "time")?;
    let nlags = extract_strict_int(nlags, "nlags")?;
    let statistic = match statistic {
        Some(value) => extract_strict_text(value, "statistic")?,
        None => "lm".to_string(),
    };
    let version = parse_statistic_version(&statistic)?;
    let source = model.retained_source(py, "breusch_godfrey_test")?;

    let mut needed: Vec<String> = x_column_names(&model.param_names, model.has_intercept).to_vec();
    needed.push(time.clone());
    let df = select_columns(&source, "data", &needed)?.frame;
    // 列の取り込み（GILが要る）の後は、Pythonオブジェクトに触れない。
    py.detach(|| {
        let x_columns = extract_model_x_columns(
            &df,
            &model.param_names,
            model.has_intercept,
            ColumnSource::new("x", "data"),
        )?;
        let time_order = extract_time_order_ranks(&df, &time, ColumnSource::new("time", "data"))?;
        let result = breusch_godfrey_test(
            &x_columns,
            model.has_intercept,
            &model.residuals,
            &time_order,
            nlags,
        )
        .map_err(least_squares_error_to_pyerr)?;

        Ok(match version {
            StatisticVersion::F => BreuschGodfreyTestOutput {
                statistic: result.f_statistic,
                p_value: result.f_p_value,
                df: result.df,
                df_denom: Some(result.f_df_denom),
                distribution: "f".to_string(),
                nlags: result.df,
            },
            StatisticVersion::Lm => BreuschGodfreyTestOutput {
                statistic: result.lm_statistic,
                p_value: result.lm_p_value,
                df: result.df,
                df_denom: None,
                distribution: "chi2".to_string(),
                nlags: result.df,
            },
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_statistic_version_accepts_lm_and_f_case_insensitively() {
        assert_eq!(parse_statistic_version("lm").unwrap(), StatisticVersion::Lm);
        assert_eq!(parse_statistic_version("LM").unwrap(), StatisticVersion::Lm);
        assert_eq!(parse_statistic_version("f").unwrap(), StatisticVersion::F);
        assert_eq!(parse_statistic_version("F").unwrap(), StatisticVersion::F);
    }

    #[test]
    fn parse_statistic_version_rejects_unknown_value() {
        assert!(parse_statistic_version("chi2").is_err());
        assert!(parse_statistic_version("").is_err());
    }

    #[test]
    fn select_statistic_picks_the_requested_version() {
        let result = AuxRegressionTest {
            lm_statistic: 1.0,
            lm_p_value: 0.1,
            f_statistic: 2.0,
            f_p_value: 0.2,
            df: 3,
            f_df_denom: 11,
            aux_terms: vec!["const".to_string()],
            dropped_terms: Vec::new(),
        };
        assert_eq!(
            select_statistic(&result, StatisticVersion::Lm),
            (1.0, 0.1, None, "chi2".to_string())
        );
        assert_eq!(
            select_statistic(&result, StatisticVersion::F),
            (2.0, 0.2, Some(11), "f".to_string())
        );
    }
}
