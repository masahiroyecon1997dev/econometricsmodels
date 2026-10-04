//! 推定オプション（`#[pyclass]`の`*Options`）の数値フィールドを厳密に取り出すユーティリティ。
//!
//! PyO3の標準の`i64`/`f64`抽出には2つの問題がある。
//! - `bool`が整数・実数として通る（`hac_lags=True`が1、`tol=True`が1.0になる）。`bool`を
//!   数値として渡す意図はないため、型の誤りとして`TypeError`にする。
//! - `i64`に収まらない巨大な整数が`OverflowError`になる。値の誤りは`ValidationError`で
//!   報告したいため、符号に応じて`i64::MIN`/`i64::MAX`に飽和させ、各手法の範囲検査
//!   （`hac_lags`は`[0, n)`等）に到達させる。固定の上限値は設けない。実数も同様に
//!   `±inf`に飽和させる。
//!
//! 型の誤り（`bool`・文字列・`None`等）は`TypeError`、値の誤り（範囲外・NaN・巨大な値）は
//! `ValidationError`（`fit()`時の検査）という分担にしている。
//!
//! コンストラクタの引数は`#[pyo3(from_py_with = ...)]`、属性の代入は手書きの
//! `#[setter]`でこのモジュールの関数を通す（`#[pyo3(get, set)]`の自動生成の
//! setterは標準の抽出を使うため）。メッセージに引数名を含めるため、フィールドごとの
//! 名前付きラッパー（`*_arg`）を`arg_extractors!`で生成している。

use pyo3::exceptions::{PyOverflowError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyBool;

use crate::column_extraction::type_name_of;

/// `ob`が`bool`なら`TypeError`にする（`bool`は`int`のサブクラスで標準の抽出を通ってしまう）。
fn reject_bool(ob: &Bound<'_, PyAny>, name: &str, expected: &str) -> PyResult<()> {
    if ob.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "'{name}' must be {expected}, got bool"
        )));
    }
    Ok(())
}

/// 整数のオプション値を`i64`として取り出す。
///
/// # Errors
/// - `bool`・整数でない値（`float`・`str`・`None`等）は`TypeError`
/// - `i64`に収まらない巨大な整数は、エラーにせず`i64::MIN`/`i64::MAX`に飽和させる
pub fn extract_strict_int(ob: &Bound<'_, PyAny>, name: &str) -> PyResult<i64> {
    reject_bool(ob, name, "an int")?;
    match ob.extract::<i64>() {
        Ok(value) => Ok(value),
        Err(err) if err.is_instance_of::<PyOverflowError>(ob.py()) => {
            // `PyOverflowError`は整数が`i64`に収まらないときだけ送出される。符号で飽和させる。
            if ob.lt(0)? {
                Ok(i64::MIN)
            } else {
                Ok(i64::MAX)
            }
        }
        Err(_) => Err(PyTypeError::new_err(format!(
            "'{name}' must be an int, got {}",
            type_name_of(ob)
        ))),
    }
}

/// 実数のオプション値を`f64`として取り出す（整数も受け付ける）。
///
/// # Errors
/// - `bool`・数値でない値（`str`・`None`等）は`TypeError`
/// - `f64`で表せない巨大な整数は`±inf`に飽和させる（範囲検査が拒否する）
pub fn extract_strict_float(ob: &Bound<'_, PyAny>, name: &str) -> PyResult<f64> {
    reject_bool(ob, name, "a real number")?;
    match ob.extract::<f64>() {
        Ok(value) => Ok(value),
        Err(err) if err.is_instance_of::<PyOverflowError>(ob.py()) => {
            if ob.lt(0)? {
                Ok(f64::NEG_INFINITY)
            } else {
                Ok(f64::INFINITY)
            }
        }
        Err(_) => Err(PyTypeError::new_err(format!(
            "'{name}' must be a real number, got {}",
            type_name_of(ob)
        ))),
    }
}

/// `None`を許す整数のオプション値。`None`は`None`のまま返す。
pub fn extract_strict_opt_int(ob: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<i64>> {
    if ob.is_none() {
        return Ok(None);
    }
    extract_strict_int(ob, name).map(Some)
}

/// `None`を許す実数のオプション値。`None`は`None`のまま返す。
pub fn extract_strict_opt_float(ob: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<f64>> {
    if ob.is_none() {
        return Ok(None);
    }
    extract_strict_float(ob, name).map(Some)
}

/// `#[pyo3(from_py_with = ...)]`に渡す、フィールド名入りの抽出関数を生成する。
macro_rules! arg_extractors {
    ($($fn_name:ident => $extract:ident($label:literal) -> $ty:ty;)*) => {
        $(
            pub fn $fn_name(ob: &Bound<'_, PyAny>) -> PyResult<$ty> {
                $extract(ob, $label)
            }
        )*
    };
}

arg_extractors! {
    confidence_level_arg => extract_strict_float("confidence_level") -> f64;
    hac_lags_arg => extract_strict_opt_int("hac_lags") -> Option<i64>;
    max_iter_arg => extract_strict_int("max_iter") -> i64;
    tol_arg => extract_strict_opt_float("tol") -> Option<f64>;
    lower_arg => extract_strict_opt_float("lower") -> Option<f64>;
    upper_arg => extract_strict_opt_float("upper") -> Option<f64>;
    gmm_max_iter_arg => extract_strict_opt_int("gmm_max_iter") -> Option<i64>;
    gmm_tol_arg => extract_strict_opt_float("gmm_tol") -> Option<f64>;
    dk_bandwidth_arg => extract_strict_opt_int("dk_bandwidth") -> Option<i64>;
}
