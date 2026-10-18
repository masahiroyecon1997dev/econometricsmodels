//! Python側に公開する例外クラス。
//!
//! `.claude/rules/rust-style.md`「エラーハンドリング」の方針に基づき、
//! 全エラーを`PyValueError`にまとめず、カテゴリ別に分ける。
//!
//! - `ValidationError`: 入力・パラメータが不正（列が存在しない、型が数値でない、
//!   欠損値がある、観測数不足、confidence_levelの範囲外等）。
//! - `ComputationError`: 計算過程で発覚した問題（特異行列等）。
//!
//! どちらも`code`・`message`・`hint`・`params`の属性を持つ構造化例外で、アプリ側が`code`と`params`から自前の文面を作れる。
//! `code`ごとの`message`/`hint`のテンプレートと`params`のキーは`engine::shared::error_catalog`が定義する。
//!
//! Python側では以下のように使う想定:
//! ```python
//! from econometricsmodels import ValidationError, ComputationError
//! try:
//!     result = ols(data=df, y="wage", x=["educ"], options=options)
//! except ValidationError as e:
//!     ...  # 入力を直すべきケース
//! except ComputationError as e:
//!     ...  # データの統計的な性質に起因するケース（特異行列等）
//! ```

use engine::shared::error::{CommonError, ErrorInfo};
use engine::shared::error_catalog::{ErrorClass, ErrorCode, ErrorReport, ParamValue};
use pyo3::PyErrArguments;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};

/// 移行中の内部限定の`code`。カタログに載せる前の文面（自由文）を包むのに使い、載せ終えたら撤去する。
const LEGACY_CODE: &str = "LEGACY_MESSAGE";

/// 構造化例外のクラスを1つ生成する。`ValidationError`と`ComputationError`は継承元以外が同じ。
///
/// 属性は`code`・`message`・`hint`（無ければ`None`）・`params`（`dict`）。`str(e)`は`message`と`hint`を空白1つでつないだ全文。
/// `args`は`__init__`で`(全文,)`にそろえる（`BaseException`の既定は引数の4つ組になるため）。
/// コンストラクタは公開で、アプリ側が自前の入力検証から同じ形の例外を作れる（`params`は浅いコピーを持つ）。
/// pickleには`__reduce__`で対応する（`multiprocessing`で子プロセスから戻る例外が壊れないように。`__dict__`も引き継ぐ）。
macro_rules! structured_exception {
    ($name:ident, $base:ty, $doc:literal) => {
        #[doc = $doc]
        #[pyclass(extends=$base, module="econometricsmodels._lib", subclass, frozen)]
        pub struct $name {
            /// Stable identifier of the error (UPPER_SNAKE_CASE).
            #[pyo3(get)]
            code: String,
            /// What is wrong, without the suggested fix.
            #[pyo3(get)]
            message: String,
            /// What to do next, or None.
            #[pyo3(get)]
            hint: Option<String>,
            /// Values that appear in the text, keyed by name.
            #[pyo3(get)]
            params: Py<PyDict>,
        }

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (code, message, hint=None, params=None))]
            fn new(
                py: Python<'_>,
                code: String,
                message: String,
                hint: Option<String>,
                params: Option<Bound<'_, PyDict>>,
            ) -> PyResult<Self> {
                let params = match params {
                    Some(p) => p.copy()?,
                    None => PyDict::new(py),
                };
                Ok(Self {
                    code,
                    message,
                    hint,
                    params: params.unbind(),
                })
            }

            /// `BaseException.__init__`が`args`を引数の4つ組で上書きするので、全文にそろえ直す。
            #[pyo3(signature = (*_args, **_kwargs))]
            fn __init__(
                slf: &Bound<'_, Self>,
                _args: &Bound<'_, PyTuple>,
                _kwargs: Option<&Bound<'_, PyDict>>,
            ) -> PyResult<()> {
                let this = slf.get();
                let text = full_text(&this.message, this.hint.as_deref());
                slf.as_any().setattr("args", (text,))
            }

            fn __str__(&self) -> String {
                full_text(&self.message, self.hint.as_deref())
            }

            fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
                let py = slf.py();
                let this = slf.get();
                let args = PyTuple::new(
                    py,
                    [
                        this.code.clone().into_pyobject(py)?.into_any(),
                        this.message.clone().into_pyobject(py)?.into_any(),
                        this.hint.clone().into_pyobject(py)?.into_any(),
                        this.params.bind(py).clone().into_any(),
                    ],
                )?;
                // `add_note()`の`__notes__`や利用者が足した属性も往復で残す。
                let state = slf.getattr("__dict__")?;
                if state.is_truthy()? {
                    PyTuple::new(py, [slf.get_type().into_any(), args.into_any(), state])
                } else {
                    PyTuple::new(py, [slf.get_type().into_any(), args.into_any()])
                }
            }
        }
    };
}

structured_exception!(
    ValidationError,
    PyValueError,
    "The input or an option is invalid. Subclass of `ValueError`."
);
structured_exception!(
    ComputationError,
    PyRuntimeError,
    "A problem was detected during the computation. Subclass of `RuntimeError`."
);

fn full_text(message: &str, hint: Option<&str>) -> String {
    match hint {
        Some(h) => format!("{message} {h}"),
        None => message.to_string(),
    }
}

fn param_to_py<'py>(py: Python<'py>, value: &ParamValue) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        ParamValue::Int(i) => i.into_pyobject(py)?.into_any(),
        ParamValue::Float(f) => f.into_pyobject(py)?.into_any(),
        ParamValue::Str(s) => s.into_pyobject(py)?.into_any(),
        ParamValue::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
        ParamValue::StrList(items) => items.into_pyobject(py)?.into_any(),
        ParamValue::None => py.None().into_bound(py),
    })
}

/// 例外クラスのコンストラクタに渡す引数。`PyErr`の正規化時（Pythonのコードに触れるとき）に初めてPythonの値にする。
///
/// 生成を遅らせるのは、`PyErr`を作る時点でGILを要求しないため（`cargo test`にはPythonインタプリタが無く、
/// バリデーション関数の単体テストが`Err`を作れなくなる）。`args`は`__init__`が`(全文,)`にそろえる。
struct ExceptionArgs {
    code: String,
    message: String,
    hint: Option<String>,
    params: Vec<(String, ParamValue)>,
}

impl PyErrArguments for ExceptionArgs {
    fn arguments(self, py: Python<'_>) -> Py<PyAny> {
        let dict = PyDict::new(py);
        for (name, value) in &self.params {
            // 値の変換が失敗するのはメモリ不足等だけ。その場合は項目を落としても例外自体は送出する。
            if let Ok(v) = param_to_py(py, value) {
                let _ = dict.set_item(name, v);
            }
        }
        let hint = match &self.hint {
            Some(h) => PyString::new(py, h).into_any(),
            None => py.None().into_bound(py),
        };
        let items = [
            PyString::new(py, &self.code).into_any(),
            PyString::new(py, &self.message).into_any(),
            hint,
            dict.into_any(),
        ];
        // タプルの確保が失敗するのはメモリ不足だけ。その場合は引数なしで構築しようとして`TypeError`になる（例外の送出自体は保たれる）。
        PyTuple::new(py, items).map_or_else(
            |_| PyTuple::empty(py).into_any().unbind(),
            |t| t.into_any().unbind(),
        )
    }
}

fn raise_args(
    class: ErrorClass,
    code: &str,
    message: String,
    hint: Option<String>,
    params: Vec<(String, ParamValue)>,
) -> PyErr {
    let args = ExceptionArgs {
        code: code.to_string(),
        message,
        hint,
        params,
    };
    match class {
        ErrorClass::Validation => PyErr::new::<ValidationError, _>(args),
        ErrorClass::Computation => PyErr::new::<ComputationError, _>(args),
    }
}

/// カタログの`code`で表せるエラーを、その`code`・`params`つきの例外にする。
pub(crate) fn report_to_pyerr(report: &ErrorReport) -> PyErr {
    raise_args(
        report.class(),
        report.code().as_str(),
        report.message(),
        report.hint(),
        report.all_params(),
    )
}

/// `engine`のエラー型（[`ErrorInfo`]を実装するもの）を例外にする。
pub(crate) fn to_pyerr(err: &impl ErrorInfo) -> PyErr {
    report_to_pyerr(&err.report())
}

/// `engine_pybind`自身が検出した入力不正を、カタログの`code`で例外にする。
#[allow(
    dead_code,
    reason = "引数・オプションの検証をカタログへ移す段階から使う"
)]
pub(crate) fn user_error(code: ErrorCode, params: Vec<(&'static str, ParamValue)>) -> PyErr {
    let mut report = ErrorReport::new(code);
    for (name, value) in params {
        report = report.param(name, value);
    }
    report_to_pyerr(&report)
}

/// 移行中: カタログに載せる前の自由文を`ValidationError`にする（`code`は`LEGACY_MESSAGE`）。
pub(crate) fn legacy_validation_error(message: impl Into<String>) -> PyErr {
    raise_args(
        ErrorClass::Validation,
        LEGACY_CODE,
        message.into(),
        None,
        Vec::new(),
    )
}

/// 移行中: カタログに載せる前の自由文を`ComputationError`にする（`code`は`LEGACY_MESSAGE`）。
pub(crate) fn legacy_computation_error(message: impl Into<String>) -> PyErr {
    raise_args(
        ErrorClass::Computation,
        LEGACY_CODE,
        message.into(),
        None,
        Vec::new(),
    )
}

/// `engine::shared::error::CommonError`をPython例外に変換する。
///
/// `engine::linear::common::LeastSquaresError`・`engine::nonlinear::common::MleError`
/// 等、`CommonError`を`#[error(transparent)] Common(#[from] CommonError)`で包む
/// 各系統のエラー型は、`Common`アームでこの関数に委譲する（系統ごとに
/// 同じ判定ロジックを重複させない）。orphan ruleにより`impl From`ではなく関数として
/// 実装する（`least_squares_error_to_pyerr`と同じ理由、`engine_pybind/src/linear/
/// CLAUDE.md`参照）。
pub(crate) fn common_error_to_pyerr(err: CommonError) -> PyErr {
    match err {
        CommonError::DimensionMismatch { .. }
        | CommonError::InsufficientObservations { .. }
        | CommonError::NoRegressors { .. }
        | CommonError::InvalidConfidenceLevel { .. }
        | CommonError::MissingClusterColumn
        | CommonError::ClusterDimensionMismatch { .. }
        | CommonError::InsufficientClusters { .. }
        | CommonError::InsufficientClustersForInference { .. }
        | CommonError::InvalidHacLags { .. } => legacy_validation_error(err.to_string()),
        CommonError::ComputationFailed(_) => legacy_computation_error(err.to_string()),
    }
}
