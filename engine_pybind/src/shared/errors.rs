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

/// `code`と`params`から[`ErrorReport`]を組み立てる（`PyErr`を作らないので、Pythonなしで検査できる）。
fn build_report(code: ErrorCode, params: Vec<(&'static str, ParamValue)>) -> ErrorReport {
    let mut report = ErrorReport::new(code);
    for (name, value) in params {
        report = report.param(name, value);
    }
    report
}

/// `engine_pybind`自身が検出した入力不正を、カタログの`code`で例外にする。
pub(crate) fn user_error(code: ErrorCode, params: Vec<(&'static str, ParamValue)>) -> PyErr {
    report_to_pyerr(&build_report(code, params))
}

/// 文字列オプション`option`に、受け付けない値`value`が渡された（`allowed`は受け付ける値の一覧）。
/// `value`は利用者が渡した文字列そのまま（大文字小文字を変えない）。
fn unknown_option_report(option: &'static str, value: &str, allowed: &[&str]) -> ErrorReport {
    build_report(
        ErrorCode::OptionUnknownValue,
        vec![
            ("option", ParamValue::from(option)),
            ("value", ParamValue::from(value)),
            ("allowed", ParamValue::from(allowed.to_vec())),
        ],
    )
}

/// 文字列オプションに受け付けない値が渡された場合の例外（[`unknown_option_report`]）。
pub(crate) fn unknown_option_value(option: &'static str, value: &str, allowed: &[&str]) -> PyErr {
    report_to_pyerr(&unknown_option_report(option, value, allowed))
}

/// `engine::shared::error::CommonError`をPython例外に変換する。
///
/// `CommonError`をそのまま返す箇所（グループ化の失敗など）で使う。各系統のエラー型
/// （`LeastSquaresError`等）は`CommonError`を包んでいても`Common`アームの`report()`が委譲する
/// ため、`to_pyerr(&err)`を呼べばよい。orphan ruleにより`impl From`ではなく関数として
/// 実装する。
pub(crate) fn common_error_to_pyerr(err: CommonError) -> PyErr {
    to_pyerr(&err)
}

/// `engine_pybind`自身が検出した、実装の誤りでだけ起きる状態を`INTERNAL_ERROR`にする。
pub(crate) fn internal_error(detail: impl Into<String>) -> PyErr {
    let detail: String = detail.into();
    user_error(
        ErrorCode::InternalError,
        vec![("detail", ParamValue::from(detail.trim_end_matches('.')))],
    )
}

// `panic = "abort"`だと`catch_unwind`が何も捕まえず、`guard`が黙って無効になる。ビルドを止める。
#[cfg(panic = "abort")]
compile_error!("engine_pybind needs panic = \"unwind\": the boundary guard relies on catch_unwind");

/// panicのメッセージ（`&str`・`String`のpayload）の最初の空でない行を、長さを抑えて`panic: `に続けたもの。
/// それ以外のpayloadは固定の文言、メッセージが空なら`(no message)`。
fn panic_detail(payload: &(dyn std::any::Any + Send)) -> String {
    const MAX_CHARS: usize = 200;
    let message = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("a panic with a payload that is not text");
    let first_line = message
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("(no message)");
    let mut detail: String = first_line.chars().take(MAX_CHARS).collect();
    if first_line.chars().count() > MAX_CHARS {
        // `internal_error`が末尾の`.`を落とし、テンプレートが`.`を足すので、記号ではなく語で示す。
        detail.push_str(" (truncated)");
    }
    format!("panic: {detail}")
}

/// 境界（`#[pyfunction]`・`#[pymethods]`）の本体を包み、Rustのpanicを`INTERNAL_ERROR`にする。
///
/// panicはそのままだとPyO3が`PanicException`（`BaseException`の子）にして、`except Exception`で
/// 捕まえられない。ここで`ComputationError`にして、アプリが通常のエラーとして扱えるようにする。
/// panicの場所（ファイルと行）は既定のpanicフックがstderrに出すので、`detail`には文言だけを入れる。
///
/// `AssertUnwindSafe`が妥当なのは、panicした呼び出しの結果を使わず、境界の関数は呼び出しをまたぐ
/// 可変な状態（`&mut self`・グローバル）を持たないため。包むのは本体だけで、戻り値のPythonオブジェクトへの
/// 変換はPyO3が`guard`の外で行う。PyO3の`#[pyfunction]`の外側には、包み忘れた関数や変換のために
/// PyO3自身のpanic捕捉（`PanicException`）が残る。abort・スタックオーバーフロー・メモリ不足は
/// `catch_unwind`では捕まえられない。**新しい境界の関数を足したら、本体をこの関数で包む。**
pub(crate) fn guard<T>(f: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    match catch_panic(f) {
        Ok(result) => result,
        Err(detail) => Err(internal_error(detail)),
    }
}

/// `f`を実行し、panicしたら`INTERNAL_ERROR`の`detail`にする文字列を返す（[`guard`]の中身。Python不要）。
fn catch_panic<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .map_err(|payload| panic_detail(payload.as_ref()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_option_report_keeps_the_value_as_given_and_matches_the_catalog() {
        let report = unknown_option_report("at", "BoGuS", &["overall", "mean", "median"]);
        assert_eq!(report.code(), ErrorCode::OptionUnknownValue);
        assert!(report.validate().is_ok());
        assert_eq!(
            report.full_text(),
            r#"Unknown value "BoGuS" for at. Use one of: "overall", "mean", "median"."#
        );
    }

    #[test]
    fn build_report_renders_a_role_error() {
        let report = build_report(
            ErrorCode::RoleEmpty,
            vec![("argument", ParamValue::from("x"))],
        );
        assert!(report.validate().is_ok());
        assert_eq!(
            report.full_text(),
            r#"x must contain at least one column name. Pass a non-empty list, for example x=["x1"]."#
        );
    }

    #[test]
    fn panic_detail_uses_the_message_of_str_and_string_payloads() {
        assert_eq!(panic_detail(&"boom"), "panic: boom");
        assert_eq!(
            panic_detail(&String::from("index 3 out of range")),
            "panic: index 3 out of range"
        );
    }

    #[test]
    fn panic_detail_keeps_the_first_non_empty_line_trimmed() {
        assert_eq!(panic_detail(&"first\nsecond"), "panic: first");
        assert_eq!(panic_detail(&"a\r\nb"), "panic: a");
        assert_eq!(panic_detail(&"\nreal message\nmore"), "panic: real message");
        assert_eq!(panic_detail(&"\n\n  spaced  \n"), "panic: spaced");
        assert_eq!(panic_detail(&"  x  "), "panic: x");
    }

    #[test]
    fn panic_detail_says_so_when_there_is_no_message() {
        assert_eq!(panic_detail(&""), "panic: (no message)");
        assert_eq!(panic_detail(&"\n"), "panic: (no message)");
        assert_eq!(panic_detail(&"  \n \t"), "panic: (no message)");
    }

    #[test]
    fn panic_detail_cuts_at_200_characters_and_marks_the_cut() {
        for (len, cut) in [(199, false), (200, false), (201, true)] {
            let message = "x".repeat(len);
            let expected = if cut {
                format!("panic: {} (truncated)", "x".repeat(200))
            } else {
                format!("panic: {message}")
            };
            assert_eq!(panic_detail(&message), expected, "len={len}");
        }
    }

    #[test]
    fn panic_detail_cuts_multibyte_text_at_a_character_boundary() {
        for (len, cut) in [(200, false), (201, true), (300, true)] {
            let message = "あ".repeat(len);
            let expected = if cut {
                format!("panic: {} (truncated)", "あ".repeat(200))
            } else {
                format!("panic: {message}")
            };
            assert_eq!(panic_detail(&message), expected, "len={len}");
        }
        // ASCIIとの混在でも、数えるのは文字（バイトではない）。
        let mixed = format!("{}{}", "a".repeat(199), "あ".repeat(5));
        assert_eq!(
            panic_detail(&mixed),
            format!("panic: {}あ (truncated)", "a".repeat(199))
        );
        // 4バイトの文字。
        assert_eq!(panic_detail(&"😀".repeat(3)), "panic: 😀😀😀");
    }

    #[test]
    fn panic_detail_describes_a_payload_that_is_not_text() {
        assert_eq!(
            panic_detail(&42_i32),
            "panic: a panic with a payload that is not text"
        );
    }

    #[test]
    fn catch_panic_passes_through_the_value_of_a_closure_that_does_not_panic() {
        assert_eq!(catch_panic(|| 7), Ok(7));
    }

    #[test]
    fn catch_panic_returns_the_detail_of_a_panic() {
        assert_eq!(
            catch_panic(|| -> i32 { panic!("boom") }),
            Err("panic: boom".to_string())
        );
        assert_eq!(
            catch_panic(|| -> i32 { std::panic::panic_any(42_i32) }),
            Err("panic: a panic with a payload that is not text".to_string())
        );
    }
}
