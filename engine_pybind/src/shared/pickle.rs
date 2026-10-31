//! 結果・オプションの`#[pyclass]`をpickle・`copy`に対応させる共通部品。
//!
//! 各クラスは`__reduce__`で`(_lib._restore_pickled, (種別名, 形式の版, 状態のdict))`を返す。状態のdictは
//! フィールド名→値で、値はPythonの組み込み型（`str`・`float`・`bytes`・`list`・`None`等）と、`fit()`に
//! 渡された`polars.DataFrame`（`source_data`）だけで作る。DataFrameはpolars自身のpickleに任せる。
//! `Vec<f64>`は、要素ごとにPythonの`float`を作らずに済むよう、リトルエンディアンの`bytes`にする
//! （残差などは観測数に比例して大きく、`list`にするとpickleの時間とサイズの大半を占めるため）。
//!
//! フィールドの型ごとの変換は[`PickleField`]で、状態の組み立てと復元は[`impl_pickle_state!`]が1つの
//! フィールド一覧から生成する。一覧は構造体の全フィールドを網羅する分割代入と構造体リテラルに展開されるので、
//! フィールドを足して一覧に足し忘れるとコンパイルエラーになる。
//!
//! 他のフィールドから作り直せる部分（Logit・Probit・Tobitの`y`・`x`は`source_data`の列から作れる）は、状態に入れず
//! 復元時に作り直す。`impl_pickle_state!`の`rebuilt_from`で、元になるフィールドを先に復元してから復元する。
//!
//! 復元は`ValidationError`の`PICKLE_FORMAT_MISMATCH`で失敗しうる。形式の版が違う、種別名が未知、キーの集合が
//! 違う、値の型が違う、のいずれか。別の版のパッケージが書いたpickleを、中途半端に有効なオブジェクトとして
//! 復元しないための扱い。フィールドの意味や並びを変えて既存のpickleを読めなくするときは[`FORMAT_VERSION`]を上げる。
//! フィールドの追加・削除はキー集合の検査で検出されるので、版を上げなくてもよい。
//!
//! 復元は値の意味（`cov_type`の綴り、配列の長さの整合など）を検査しない。型が合えば構造体を直接組み立てる
//! （`*Options`のコンストラクタを通さない）。pickleは信頼できる入力にだけ使う前提で、不正な値のオプションは
//! `fit()`が通常どおり検査し、`ValidationError`にする。

use std::collections::{BTreeMap, HashMap};

use engine::shared::error_catalog::{ErrorCode, ParamValue};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};

use super::column_extraction::check_dataframe;
use super::errors::user_error;

/// 状態のdictの形式の版。復元関数が一致を確かめる。
pub(crate) const FORMAT_VERSION: u32 = 1;

/// pickleできる`#[pyclass]`。[`impl_pickle_state!`]で実装する。
pub(crate) trait PickleState: Sized {
    /// 復元関数が種別を選ぶための名前（クラス名）。
    const KIND: &'static str;

    /// 全フィールドをフィールド名→Python値のdictにする。
    fn to_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>>;

    /// [`PickleState::to_state`]の逆。キーの集合・値の型が合わなければ`PICKLE_FORMAT_MISMATCH`。
    fn from_state(state: &Bound<'_, PyDict>) -> PyResult<Self>;
}

/// 状態のdictに入れるフィールドの型。`to_py`で作った値を`from_py`が同じ型に戻せる。
pub(crate) trait PickleField: Sized {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>;

    /// 型が合わなければ`None`（呼び出し側が`PICKLE_FORMAT_MISMATCH`にする）。
    fn from_py(obj: &Bound<'_, PyAny>) -> Option<Self>;
}

/// 状態のdictには一部だけを入れ、残りを復元済みの`source_data`から再構築するフィールドの型
/// （Logit・Probit・Tobitの`fitted`。`y`・`x`は元のDataFrameの列から作り直せるので、状態に入れない）。
/// [`impl_pickle_state!`]の`rebuilt_from`で使う。
pub(crate) trait PickleFieldFromSource: Sized {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>;

    /// `source`は復元済みの`source_data`（`polars.DataFrame`）。型が合わない・再構築できない
    /// （列が無い、数値でない、欠損値を含む等）ときは`None`。
    fn from_py(obj: &Bound<'_, PyAny>, source: &Bound<'_, PyAny>) -> Option<Self>;
}

/// 変換が`IntoPyObject`・`FromPyObject`そのままでよい型。
macro_rules! plain_pickle_field {
    ($($ty:ty),+ $(,)?) => {$(
        impl PickleField for $ty {
            fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                self.into_bound_py_any(py)
            }

            fn from_py(obj: &Bound<'_, PyAny>) -> Option<Self> {
                obj.extract::<Self>().ok()
            }
        }
    )+};
}

plain_pickle_field!(
    f64,
    bool,
    usize,
    i64,
    String,
    Vec<String>,
    (f64, f64),
    BTreeMap<String, f64>,
    HashMap<String, f64>,
);

impl PickleField for Vec<f64> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // 中間の`Vec<u8>`を作らず、`bytes`のバッファへ直接書き込む。
        let bytes = PyBytes::new_with(py, self.len() * size_of::<f64>(), |buf| {
            let (chunks, _) = buf.as_chunks_mut::<{ size_of::<f64>() }>();
            for (chunk, value) in chunks.iter_mut().zip(self) {
                *chunk = value.to_le_bytes();
            }
            Ok(())
        })?;
        Ok(bytes.into_any())
    }

    fn from_py(obj: &Bound<'_, PyAny>) -> Option<Self> {
        let bytes = obj.cast::<PyBytes>().ok()?.as_bytes();
        // 長さが8の倍数でなければ`as_chunks`の余り（`remainder`）が空にならない。
        let (chunks, remainder) = bytes.as_chunks::<{ size_of::<f64>() }>();
        remainder.is_empty().then(|| {
            chunks
                .iter()
                .map(|chunk| f64::from_le_bytes(*chunk))
                .collect()
        })
    }
}

/// 行ごとの`Vec<f64>`（係数の分散共分散行列`cov_params`）は、行ごとの`bytes`のタプル。
/// 行の長さの整合（正方行列か）は検査しない（値の意味は検査しない方針、モジュール冒頭参照）。
impl PickleField for Vec<Vec<f64>> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rows = self
            .iter()
            .map(|row| row.to_py(py))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, rows)?.into_any())
    }

    fn from_py(obj: &Bound<'_, PyAny>) -> Option<Self> {
        obj.cast::<PyTuple>()
            .ok()?
            .iter()
            .map(|row| <Vec<f64> as PickleField>::from_py(&row))
            .collect()
    }
}

/// 行列は`(行数, 列数, 列優先のf64のバイト列)`。形が`行数×列数`と合わないものは拒否する。
impl PickleField for faer::Mat<f64> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (nrows, ncols) = (self.nrows(), self.ncols());
        // 一時の`Vec`を作らず、`bytes`のバッファへ列ごとに直接書き込む。
        let bytes = PyBytes::new_with(py, nrows * ncols * size_of::<f64>(), |buf| {
            let (cells, _) = buf.as_chunks_mut::<{ size_of::<f64>() }>();
            if nrows > 0 {
                for (j, column) in cells.chunks_mut(nrows).enumerate() {
                    for (i, cell) in column.iter_mut().enumerate() {
                        *cell = self.get(i, j).to_le_bytes();
                    }
                }
            }
            Ok(())
        })?;
        (nrows, ncols, bytes).into_bound_py_any(py)
    }

    fn from_py(obj: &Bound<'_, PyAny>) -> Option<Self> {
        let (nrows, ncols, bytes) = obj.extract::<(usize, usize, Bound<'_, PyAny>)>().ok()?;
        let bytes = bytes.cast::<PyBytes>().ok()?.as_bytes();
        // 形が`行数×列数`のf64と一致しないものは拒否する（掛け算のオーバーフローも拒否）。
        let expected = nrows.checked_mul(ncols)?.checked_mul(size_of::<f64>())?;
        if bytes.len() != expected {
            return None;
        }
        let (cells, _) = bytes.as_chunks::<{ size_of::<f64>() }>();
        Some(faer::Mat::from_fn(nrows, ncols, |i, j| {
            f64::from_le_bytes(cells[j * nrows + i])
        }))
    }
}

/// 複数の値を1つのタプルにまとめる（入れ子の状態を`PickleField`にするとき用）。
pub(crate) fn to_tuple<'py>(
    py: Python<'py>,
    items: Vec<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyTuple::new(py, items)?.into_any())
}

/// [`to_tuple`]の逆。長さが`len`のタプルでなければ`None`。
pub(crate) fn from_tuple<'py>(
    obj: &Bound<'py, PyAny>,
    len: usize,
) -> Option<Vec<Bound<'py, PyAny>>> {
    let tuple = obj.cast::<PyTuple>().ok()?;
    (tuple.len() == len).then(|| tuple.iter().collect())
}

impl<T: PickleField> PickleField for Option<T> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Some(value) => value.to_py(py),
            None => Ok(py.None().into_bound(py)),
        }
    }

    fn from_py(obj: &Bound<'_, PyAny>) -> Option<Self> {
        if obj.is_none() {
            Some(None)
        } else {
            T::from_py(obj).map(Some)
        }
    }
}

/// `fit()`に渡された`polars.DataFrame`（`source_data`）。`Py<PyAny>`のフィールドは`source_data`だけなので、
/// 復元時にDataFrameであることを確かめる（別の型の値が入った状態から、後の`augment()`で
/// `AttributeError`になる結果が復元されないように）。
impl PickleField for Py<PyAny> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(self.bind(py).clone())
    }

    fn from_py(obj: &Bound<'_, PyAny>) -> Option<Self> {
        check_dataframe(obj, "source_data")
            .is_ok()
            .then(|| obj.clone().unbind())
    }
}

/// 復元できないpickleの`PICKLE_FORMAT_MISMATCH`。
pub(crate) fn format_mismatch(kind: &str) -> PyErr {
    user_error(
        ErrorCode::PickleFormatMismatch,
        vec![("object", ParamValue::from(kind))],
    )
}

/// 状態のdictから1フィールドを取り出す。キーが無い・型が合わないときは`PICKLE_FORMAT_MISMATCH`。
pub(crate) fn take_field<T: PickleField>(
    state: &Bound<'_, PyDict>,
    name: &str,
    kind: &str,
) -> PyResult<T> {
    state
        .get_item(name)?
        .and_then(|value| T::from_py(&value))
        .ok_or_else(|| format_mismatch(kind))
}

/// [`take_field`]の、`source_data`から再構築するフィールド用。
pub(crate) fn take_field_from_source<T: PickleFieldFromSource>(
    state: &Bound<'_, PyDict>,
    name: &str,
    kind: &str,
    source: &Bound<'_, PyAny>,
) -> PyResult<T> {
    state
        .get_item(name)?
        .and_then(|value| T::from_py(&value, source))
        .ok_or_else(|| format_mismatch(kind))
}

/// `__reduce__`の本体。クラスの`#[pymethods]`から1行で呼ぶ。
pub(crate) fn reduce<'py, T>(slf: &Bound<'py, T>) -> PyResult<Bound<'py, PyTuple>>
where
    T: PickleState + pyo3::PyClass,
{
    let py = slf.py();
    let state = T::to_state(&*slf.try_borrow()?, py)?;
    let restore = py
        .import("econometricsmodels._lib")?
        .getattr("_restore_pickled")?;
    let args = PyTuple::new(
        py,
        [
            T::KIND.into_pyobject(py)?.into_any(),
            FORMAT_VERSION.into_pyobject(py)?.into_any(),
            state.into_any(),
        ],
    )?;
    PyTuple::new(py, [restore, args.into_any()])
}

/// `_restore_pickled`の種別ごとの本体。版と種別の検査は呼び出し側（`lib.rs`）が先に行う。
pub(crate) fn restore<T>(py: Python<'_>, state: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>>
where
    T: PickleState + pyo3::PyClass + Into<pyo3::PyClassInitializer<T>>,
{
    Ok(Py::new(py, T::from_state(state)?)?.into_any())
}

/// 構造体の全フィールドを列挙して[`PickleState`]を実装する。
///
/// `impl_pickle_state!(OLSResult { params, std_errors, ... });`のように、構造体と同じモジュールで呼ぶ
/// （非公開フィールドも列挙するため）。フィールドの型は[`PickleField`]を実装している必要がある。
///
/// `impl_pickle_state!(LogitResult { ..., source_data } rebuilt_from(source_data) { fitted });`のように
/// `rebuilt_from(<DataFrameのフィールド>) { ... }`を付けると、波括弧の中のフィールドは
/// [`PickleFieldFromSource`]で、先に復元した`<DataFrameのフィールド>`を使って復元する
/// （`<DataFrameのフィールド>`は最初の波括弧にも含める）。
macro_rules! impl_pickle_state {
    ($ty:ident { $($field:ident),+ $(,)? } rebuilt_from($source:ident) { $($rebuilt:ident),+ $(,)? }) => {
        impl $crate::shared::pickle::PickleState for $ty {
            const KIND: &'static str = stringify!($ty);

            fn to_state<'py>(
                &self,
                py: ::pyo3::Python<'py>,
            ) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::pyo3::types::PyDict>> {
                let Self { $($field,)+ $($rebuilt),+ } = self;
                let state = ::pyo3::types::PyDict::new(py);
                $( state.set_item(
                    stringify!($field),
                    $crate::shared::pickle::PickleField::to_py($field, py)?,
                )?; )+
                $( state.set_item(
                    stringify!($rebuilt),
                    $crate::shared::pickle::PickleFieldFromSource::to_py($rebuilt, py)?,
                )?; )+
                Ok(state)
            }

            fn from_state(
                state: &::pyo3::Bound<'_, ::pyo3::types::PyDict>,
            ) -> ::pyo3::PyResult<Self> {
                const N_FIELDS: usize =
                    [$(stringify!($field),)+ $(stringify!($rebuilt)),+].len();
                if state.len() != N_FIELDS {
                    return Err($crate::shared::pickle::format_mismatch(Self::KIND));
                }
                // 再構築するフィールドは、元になるDataFrameを復元したあとで復元する。
                $( let $field = $crate::shared::pickle::take_field(
                    state,
                    stringify!($field),
                    Self::KIND,
                )?; )+
                let source: &::pyo3::Py<::pyo3::PyAny> = &$source;
                $( let $rebuilt = $crate::shared::pickle::take_field_from_source(
                    state,
                    stringify!($rebuilt),
                    Self::KIND,
                    source.bind(state.py()),
                )?; )+
                Ok(Self { $($field,)+ $($rebuilt),+ })
            }
        }
    };
    ($ty:ident { $($field:ident),+ $(,)? }) => {
        impl $crate::shared::pickle::PickleState for $ty {
            const KIND: &'static str = stringify!($ty);

            fn to_state<'py>(
                &self,
                py: ::pyo3::Python<'py>,
            ) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::pyo3::types::PyDict>> {
                let Self { $($field),+ } = self;
                let state = ::pyo3::types::PyDict::new(py);
                $( state.set_item(
                    stringify!($field),
                    $crate::shared::pickle::PickleField::to_py($field, py)?,
                )?; )+
                Ok(state)
            }

            fn from_state(
                state: &::pyo3::Bound<'_, ::pyo3::types::PyDict>,
            ) -> ::pyo3::PyResult<Self> {
                const N_FIELDS: usize = [$(stringify!($field)),+].len();
                if state.len() != N_FIELDS {
                    return Err($crate::shared::pickle::format_mismatch(Self::KIND));
                }
                Ok(Self {
                    $( $field: $crate::shared::pickle::take_field(
                        state,
                        stringify!($field),
                        Self::KIND,
                    )?, )+
                })
            }
        }
    };
}

pub(crate) use impl_pickle_state;
