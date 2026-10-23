//! エラーカタログのPythonへの書き出し（非公開API）。
//!
//! `ErrorCode`のStrEnum・公開ドキュメントの表を生成するスクリプトと、カタログの整合を検査するテストが読む。
//! 利用者向けのAPIではないので、`python_package`からは再エクスポートしない。

use engine::shared::error_catalog::{
    CATALOG, ErrorClass, ErrorCode, ErrorReport, ParamType, ParamValue, Stage, TEST_LABELS,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::errors::{guard, report_to_pyerr};

fn param_type_name(ty: ParamType) -> &'static str {
    match ty {
        ParamType::Int => "int",
        ParamType::Float => "float",
        ParamType::FloatOrNone => "float_or_none",
        ParamType::Str => "str",
        ParamType::StrList => "str_list",
        ParamType::ColumnList => "column_list",
        ParamType::Bool => "bool",
    }
}

/// 全`code`・`Stage`の文脈句・診断検定の英語名を`dict`で返す。
///
/// `codes`は`ErrorCode`の宣言順。各要素は`code`・`class`（`"ValidationError"`/`"ComputationError"`）・
/// `params`（`name`・`type`・`in_text`のリスト）・`message`・`hint`（無ければ`None`）・`stage_allowed`。
/// `stages`は`name`・`prefix`（`{name}`を含む文脈句のテンプレート）・`params`（文脈句が使う補助値の名前）、
/// `test_labels`は診断検定の識別子から英語名への対応。
#[pyfunction]
pub(crate) fn error_catalog(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    guard(|| {
        let codes = PyList::empty(py);
        for spec in CATALOG {
            let entry = PyDict::new(py);
            entry.set_item("code", spec.code.as_str())?;
            entry.set_item(
                "class",
                match spec.class {
                    ErrorClass::Validation => "ValidationError",
                    ErrorClass::Computation => "ComputationError",
                },
            )?;
            let params = PyList::empty(py);
            for p in spec.params {
                let item = PyDict::new(py);
                item.set_item("name", p.name)?;
                item.set_item("type", param_type_name(p.ty))?;
                item.set_item("in_text", p.in_text)?;
                params.append(item)?;
            }
            entry.set_item("params", params)?;
            entry.set_item("message", spec.message)?;
            entry.set_item("hint", spec.hint)?;
            entry.set_item("stage_allowed", spec.stage_allowed)?;
            codes.append(entry)?;
        }

        let stages = PyList::empty(py);
        for stage in Stage::samples() {
            let entry = PyDict::new(py);
            entry.set_item("name", stage.as_str())?;
            entry.set_item("prefix", stage.prefix())?;
            let aux: Vec<&str> = stage.aux_params().iter().map(|(n, _)| *n).collect();
            entry.set_item("params", aux)?;
            stages.append(entry)?;
        }

        let test_labels = PyDict::new(py);
        for (id, label) in TEST_LABELS {
            test_labels.set_item(id, label)?;
        }

        let out = PyDict::new(py);
        out.set_item("codes", codes)?;
        out.set_item("stages", stages)?;
        out.set_item("test_labels", test_labels)?;
        Ok(out)
    })
}

fn to_param_value(ty: ParamType, value: &Bound<'_, PyAny>) -> PyResult<ParamValue> {
    Ok(match ty {
        ParamType::Int => ParamValue::Int(value.extract()?),
        ParamType::Float => ParamValue::Float(value.extract()?),
        ParamType::FloatOrNone => ParamValue::from(value.extract::<Option<f64>>()?),
        ParamType::Str => ParamValue::Str(value.extract()?),
        ParamType::StrList | ParamType::ColumnList => ParamValue::StrList(value.extract()?),
        ParamType::Bool => ParamValue::Bool(value.extract()?),
    })
}

fn stage_from_params(name: &str, params: &Bound<'_, PyDict>) -> PyResult<Stage> {
    let mut stage = Stage::samples()
        .into_iter()
        .find(|s| s.as_str() == name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown stage {name:?}")))?;
    match &mut stage {
        Stage::FirstStage { endog_name } => {
            *endog_name = params
                .get_item("endog_name")?
                .ok_or_else(|| PyValueError::new_err("stage FIRST_STAGE needs endog_name"))?
                .extract()?;
        }
        Stage::DiagnosticAuxiliaryRegression { test } => {
            *test = params
                .get_item("test")?
                .ok_or_else(|| PyValueError::new_err("stage needs test"))?
                .extract()?;
        }
        _ => {}
    }
    Ok(stage)
}

/// テスト用: `code`と`params`（例外の`params`と同じ形。`stage`とその補助値を含む）から例外を送出する。
///
/// カタログの文面がRust側でどう描画されるかを、Pythonのテストが生成物と突き合わせるために使う。
#[pyfunction]
pub(crate) fn raise_error_for_test(code: &str, params: &Bound<'_, PyDict>) -> PyResult<()> {
    guard(|| {
        let code = ErrorCode::ALL
            .iter()
            .copied()
            .find(|c| c.as_str() == code)
            .ok_or_else(|| PyValueError::new_err(format!("unknown code {code:?}")))?;
        let spec = code.spec();
        let mut report = ErrorReport::new(code);
        for p in spec.params {
            let value = params
                .get_item(p.name)?
                .ok_or_else(|| PyValueError::new_err(format!("missing param {}", p.name)))?;
            report = report.param(p.name, to_param_value(p.ty, &value)?);
        }
        if let Some(stage) = params.get_item("stage")? {
            report = report.with_stage(stage_from_params(&stage.extract::<String>()?, params)?);
        }
        report
            .validate()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Err(report_to_pyerr(&report))
    })
}

/// テスト用: 境界の`guard`の中でpanicを起こし、`INTERNAL_ERROR`に変わることを確かめる。
///
/// `kind`は`"str"`（`&str`のpayload）・`"string"`（`String`のpayload）・`"other"`（文字列でないpayload）・
/// `"long"`（300文字のメッセージ。`detail`の切り詰めを見る）。
/// 実際の境界の関数と同じ`guard`を通る。
#[pyfunction]
pub(crate) fn panic_for_test(kind: &str) -> PyResult<()> {
    let kind = kind.to_owned();
    guard(move || match kind.as_str() {
        "str" => panic!("deliberate panic for a test"),
        "string" => panic!("deliberate panic for a test: {}", kind.len()),
        "other" => std::panic::panic_any(42_i32),
        "long" => panic!("{}", "x".repeat(300)),
        _ => Err(PyValueError::new_err(format!("unknown kind {kind:?}"))),
    })
}
