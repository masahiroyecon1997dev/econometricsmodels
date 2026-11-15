//! DataFrame・列名の受け取りと、列の存在・欠損値の検査。

use polars::prelude::*;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyType};

use crate::shared::column_error::{ColumnExtractionError, ColumnRole, ColumnSource};
use crate::shared::errors::internal_error;

/// `ob`が`polars.DataFrame`であることを検査する。
///
/// `PyDataFrame`の`FromPyObject`実装（`pyo3-polars`）は型検証を行わず、polars以外のDataFrame
/// （pandas等）を渡すと内部実装が漏れた`AttributeError`が送出されてしまう。そのため、`#[pyfunction]`/
/// `#[pymethods]`の引数型は`PyDataFrame`ではなく`Bound<'_, PyAny>`にし、本体の先頭でこれを呼んで
/// `TypeError`にする（pyo3は引数の抽出をパラメータの型から自動生成するので、`PyDataFrame`を引数型の
/// まま使うと、本体に入る前に`FromPyObject`が走ってしまう）。
///
/// # Errors
/// `polars.DataFrame`でなければ`TypeError`（引数名・実際の型を含む。pandasと`LazyFrame`には
/// 変換・実行の方法を添える）。
pub fn check_dataframe(ob: &Bound<'_, PyAny>, param_name: &str) -> PyResult<()> {
    if is_instance_of_polars(ob, "DataFrame") {
        return Ok(());
    }
    let type_name = type_name_of(ob);
    let (shown_type, advice) = if is_instance_of_polars(ob, "LazyFrame") {
        ("LazyFrame".to_string(), " Call .collect() first.")
    } else if is_pandas_dataframe(ob) {
        (
            "pandas.DataFrame".to_string(),
            " Convert it with polars.from_pandas(df).",
        )
    } else {
        (type_name, "")
    };
    Err(PyTypeError::new_err(format!(
        "{param_name} must be a polars.DataFrame, got {shown_type}.{advice}"
    )))
}

/// polarsの`DataFrame`の全列を取り込もうとすると、polars-coreがpanicするdtype（Pythonの
/// クラス名）。取り込めないので、推定に使う列として指名されたときは`ValidationError`にし、
/// 指名されなければ読まない（`docs/guide/accepted-data.md`「指名した列だけ読む」）。
/// `pyo3-polars`が列の取り込みで使うpolarsの機能（`engine_pybind/Cargo.toml`の`dtype-*`）に
/// 合わせて保つ。
const UNREADABLE_DTYPES: &[&str] = &["UInt128"];

/// polarsのdtype（Pythonのオブジェクト）が、取り込めないdtype（[`UNREADABLE_DTYPES`]）そのもの、
/// または`List`・`Array`の内側の型・`Struct`のフィールドの型として含んでいれば、エラーに出す
/// dtypeの呼び名を返す。`UInt128`そのものは`UInt128`、`List(UInt128)`のように中に含むネスト型は
/// `List`（最上位の型名。ネスト型は数値・キーのどの役割にも使えない）。
///
/// dtypeの文字列表現（`str(dtype)`）の部分一致では判定しない。`Enum`のカテゴリ名や`Struct`の
/// フィールド名のように、利用者が決めた文字列が表現に入るdtypeがあり、`UInt128`という名前の
/// カテゴリを持つ`Enum`を取り込めないdtypeと取り違えるため。クラス名と型の構造だけを見る。
/// 内側の型を持つのは組み込みの`List`・`Array`・`Struct`だけで、`Extension`は辿らない。
///
/// Pythonのオブジェクトを受けるので、Rustの単体テストは書けない。検証は
/// `tests/cross_method/test_key_columns.py`と`tests/cross_method/test_input_dtypes.py`にある。
fn unreadable_label(dtype: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let name: String = dtype
        .call_method0("base_type")?
        .getattr("__name__")?
        .extract()?;
    if UNREADABLE_DTYPES.contains(&name.as_str()) {
        return Ok(Some(name));
    }
    // `List`・`Array`は`inner`、`Struct`は`fields`（各`Field`の`dtype`）に内側の型を持つ。
    // 型名で絞るのは、他のdtypeにたまたま同名の属性があっても辿らないため。
    match name.as_str() {
        "List" | "Array" => {
            if let Some(inner) = dtype.getattr_opt("inner")?
                && !inner.is_none()
                && unreadable_label(&inner)?.is_some()
            {
                return Ok(Some(name));
            }
        }
        "Struct" => {
            if let Some(fields) = dtype.getattr_opt("fields")?
                && !fields.is_none()
            {
                for field in fields.try_iter()? {
                    if unreadable_label(&field?.getattr("dtype")?)?.is_some() {
                        return Ok(Some(name));
                    }
                }
            }
        }
        _ => {}
    }
    Ok(None)
}

/// 目印の列の名前（[`UNREADABLE_MARKER`]に続けてdtypeの呼び名と元の列名を`\u{1}`で区切る）。
fn marker_name(label: &str, name: &str) -> String {
    format!("{UNREADABLE_MARKER}{label}\u{1}{name}")
}

/// 目印の列の名前`column_name`から（dtypeの呼び名、元の列名）を取り出す。目印でなければ`None`。
/// 元の列名に`\u{1}`が含まれていても、最初の区切りで切るので正しく復元される。
fn parse_marker(column_name: &str) -> Option<(&str, &str)> {
    column_name
        .strip_prefix(UNREADABLE_MARKER)?
        .split_once('\u{1}')
}

/// 取り込めない列の代わりに、取り込んだ`DataFrame`へ入れる目印の列の名前の接頭辞。
/// 目印は行数だけを持つ`Null`型の列で、名前に（dtype、元の列名）を符号化する。列を引く側
/// （[`missing_column_error`]）が「存在しない」ではなく「取り込めないdtype」と報告するために使う。
const UNREADABLE_MARKER: &str = "\u{1}unreadable\u{1}";

/// `fit()`に渡されたデータ。推定に使う列だけを取り込んだ`frame`と、元のPythonオブジェクト
/// `source`（`augment()`が全列を残して予測値の列を足すために保持する）。
pub struct SelectedData {
    /// 元の`polars.DataFrame`の浅いコピー（列は共有し、データはコピーしない）。
    pub source: Py<PyAny>,
    /// `columns`で指名された列（存在し、取り込めるもの）だけを取り込んだ`DataFrame`。
    pub frame: DataFrame,
}

/// `ob`（`polars.DataFrame`）から、推定に使う列`columns`だけをRustの`DataFrame`に取り込む。
///
/// 指名されない列は、dtypeにかかわらず読まない（`pyo3-polars`が全列を取り込むと、ネスト列や
/// 取り込めないdtypeの列が未使用でも失敗するため）。`columns`にあって`ob`に無い列は取り込まない
/// （後で列を引く側が`COLUMN_NOT_FOUND`にする）。取り込めないdtype（[`UNREADABLE_DTYPES`]）の列は
/// 目印の列に置き換え、引く側が`COLUMN_UNSUPPORTED_DTYPE`にする。どちらも、検査の順序は列を引く
/// 位置で決まる。
///
/// # Errors
/// - `polars.DataFrame`でなければ`TypeError`（[`check_dataframe`]）
/// - 本物の`polars.DataFrame`なのに列の取り込みが失敗した場合は`INTERNAL_ERROR`
pub fn select_columns(
    ob: &Bound<'_, PyAny>,
    param_name: &str,
    columns: &[String],
) -> PyResult<SelectedData> {
    check_dataframe(ob, param_name)?;
    let read = |err: PyErr| {
        internal_error(format!(
            "failed to read {param_name} as a polars.DataFrame: {err}"
        ))
    };
    let schema = ob.getattr("schema").map_err(read)?;
    let height: usize = ob
        .getattr("height")
        .and_then(|h| h.extract())
        .map_err(read)?;

    let mut seen = std::collections::HashSet::new();
    let mut frame_columns: Vec<Column> = Vec::with_capacity(columns.len());
    let mut markers: Vec<Column> = Vec::new();
    for name in columns {
        if !seen.insert(name.as_str()) || !schema.contains(name).map_err(read)? {
            continue;
        }
        let dtype = schema.get_item(name).map_err(read)?;
        if let Some(label) = unreadable_label(&dtype).map_err(read)? {
            let marker = marker_name(&label, name);
            markers.push(Series::new_null(marker.as_str().into(), height).into());
            continue;
        }
        let series: pyo3_polars::PySeries = ob
            .call_method1("get_column", (name,))
            .and_then(|s| s.extract())
            .map_err(read)?;
        frame_columns.push(series.0.into());
    }
    frame_columns.extend(markers);
    let frame = DataFrame::new(height, frame_columns).map_err(|err| {
        internal_error(format!(
            "failed to read {param_name} as a polars.DataFrame: {err}"
        ))
    })?;
    // 元のオブジェクトそのものではなく、列を共有する浅いコピー（`DataFrame.clone()`）を持つ。
    // 結果が保持する間に、利用者が元のデータをその場で変更（`replace_column`・`drop_in_place`
    // ・`extend`等）しても、結果の`augment()`や診断検定に混入しないようにするため。
    let source = ob.call_method0("clone").map_err(read)?.unbind();
    Ok(SelectedData { source, frame })
}

/// 列`name`が`df`に無いときのエラー。取り込めないdtypeのために目印の列が入っていれば
/// `UnsupportedDtype`、無ければ`NotFound`にする。
pub(super) fn missing_column_error(
    df: &DataFrame,
    name: &str,
    role: ColumnRole,
    source: ColumnSource<'_>,
) -> ColumnExtractionError {
    for column_name in df.get_column_names() {
        if let Some((label, original)) = parse_marker(column_name.as_str())
            && original == name
        {
            return ColumnExtractionError::unsupported_dtype(name, label.to_string(), role, source);
        }
    }
    ColumnExtractionError::not_found(name, source)
}

/// 欠損値（null）が1つでもあれば`ColumnExtractionError::ContainsNull`にする。
pub(super) fn reject_missing_values(
    name: &str,
    null_count: usize,
    source: ColumnSource<'_>,
) -> Result<(), ColumnExtractionError> {
    if null_count > 0 {
        return Err(ColumnExtractionError::contains_null(
            name, null_count, source,
        ));
    }
    Ok(())
}

/// 欠損値（null）が無いと検査した列で`None`が出たときの`detail`。
pub(super) const UNEXPECTED_NULL: &str = "unexpected null in a column checked to have none";

/// 欠損値（null）が無いと検査した列の値を取り出す。`None`は検査の後に起きない想定の防御的な経路で、
/// panicさせずに`ConversionFailed`（内部エラー）にする。
pub(super) fn present_value<T>(name: &str, value: Option<T>) -> Result<T, ColumnExtractionError> {
    value.ok_or_else(|| ColumnExtractionError::conversion_failed(name, UNEXPECTED_NULL))
}

/// `ob`がpandasの`DataFrame`（サブクラスを含む）か。pandasをimportせずに、型の継承順
/// （MRO）にある型の完全修飾名だけで判定する（pandasは依存に含まれない）。
fn is_pandas_dataframe(ob: &Bound<'_, PyAny>) -> bool {
    let Ok(mro) = ob.get_type().getattr("__mro__") else {
        return false;
    };
    let Ok(classes) = mro.try_iter() else {
        return false;
    };
    classes
        .flatten()
        .filter_map(|class| class.cast_into::<PyType>().ok())
        .filter_map(|class| class.fully_qualified_name().ok())
        .any(|name| is_pandas_dataframe_name(&name.to_string_lossy()))
}

/// 完全修飾名`type_name`がpandasの`DataFrame`クラスそのものか。
fn is_pandas_dataframe_name(type_name: &str) -> bool {
    type_name.starts_with("pandas.") && type_name.ends_with(".DataFrame")
}

/// `ob`の型の完全修飾名（`str`・`numpy.ndarray`等）。エラーメッセージ用。
///
/// polarsの型は内部のモジュール経路（`polars.series.series.Series`）を隠し、クラス名
/// （`Series`）で示す。本パッケージの型（`WLSOptions`等）も同様。
pub(crate) fn type_name_of(ob: &Bound<'_, PyAny>) -> String {
    let Ok(class) = ob.get_type().fully_qualified_name() else {
        return "unknown type".to_string();
    };
    let name = class.to_string_lossy();
    // 本パッケージの型（`*Options`等）は内部のモジュール（`econometricsmodels._lib`）を隠す。
    if let Some(rest) = name.strip_prefix("econometricsmodels._lib.") {
        return rest.to_string();
    }
    match name.strip_prefix("polars.") {
        Some(rest) => rest.rsplit('.').next().unwrap_or(rest).to_string(),
        None => name.into_owned(),
    }
}

/// 単一の列名を受け取る引数（`y`・`entity`・`weight`等）を`String`として取り出す。
///
/// 型の誤りは`ValueError`系の`ValidationError`ではなく、Pythonの慣習どおり`TypeError`に
/// する。PyO3の標準の引数抽出は引数名を含まないメッセージになるため、`#[pyfunction]`の
/// 引数型を`&Bound<PyAny>`にしてこの関数で取り出す（`check_dataframe`と同じ構成）。
///
/// # Errors
/// `str`でなければ`TypeError`（引数名・実際の型を含む）。
pub fn extract_column_name(ob: &Bound<'_, PyAny>, param_name: &str) -> PyResult<String> {
    ob.extract::<String>().map_err(|_| {
        PyTypeError::new_err(format!(
            "{param_name} must be a str column name, got {}.",
            type_name_of(ob)
        ))
    })
}

/// 列名のリストを受け取る引数（`x`・`x_exog`・`x_endog`・`instruments`）を`Vec<String>`として
/// 取り出す。`list`のみを受け付ける（`str`・tuple・`polars.Series`等は`TypeError`）。
///
/// `x="x1"`のように`str`を渡す誤りが最も多いため、期待する形（`x=["x1"]`）を
/// メッセージで示す。
///
/// # Errors
/// - `list`でなければ`TypeError`
/// - 要素に`str`以外があれば`TypeError`（要素の位置を含む）
pub fn extract_column_list(ob: &Bound<'_, PyAny>, param_name: &str) -> PyResult<Vec<String>> {
    let list = ob.cast::<PyList>().map_err(|_| {
        PyTypeError::new_err(format!(
            "{param_name} must be a list of column names, got {}. \
             Pass a list, for example {param_name}=[\"x1\"].",
            type_name_of(ob)
        ))
    })?;
    list.iter()
        .enumerate()
        .map(|(i, item)| {
            item.extract::<String>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "{param_name}[{i}] must be a str column name, got {}.",
                    type_name_of(&item)
                ))
            })
        })
        .collect()
}

/// `ob`が`polars.<class_name>`のインスタンスか（サブクラスを含む）。`polars`を読み込めない、
/// またはクラスが見つからない場合は`false`。
fn is_instance_of_polars(ob: &Bound<'_, PyAny>, class_name: &str) -> bool {
    ob.py()
        .import("polars")
        .and_then(|module| module.getattr(class_name))
        .and_then(|class| ob.is_instance(&class))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn present_value_returns_the_value_or_a_conversion_failure() {
        assert!(matches!(present_value("x", Some(1.0)), Ok(v) if v == 1.0));
        assert!(matches!(
            present_value::<f64>("x", None),
            Err(ColumnExtractionError::ConversionFailed { .. })
        ));
    }

    #[test]
    fn marker_name_round_trips_the_label_and_the_column_name() {
        for name in ["u", "年齢", "a b", "^a$", "", "with\u{1}separator"] {
            let marker = marker_name("UInt128", name);
            assert_eq!(parse_marker(&marker), Some(("UInt128", name)), "{name:?}");
        }
    }

    #[test]
    fn parse_marker_ignores_ordinary_column_names() {
        assert_eq!(parse_marker("x1"), None);
        assert_eq!(parse_marker("unreadable"), None);
    }

    #[test]
    fn is_pandas_dataframe_name_matches_only_the_pandas_dataframe_class() {
        assert!(is_pandas_dataframe_name("pandas.core.frame.DataFrame"));
        assert!(is_pandas_dataframe_name("pandas.DataFrame"));
        assert!(!is_pandas_dataframe_name("pandas.core.series.Series"));
        assert!(!is_pandas_dataframe_name(
            "polars.dataframe.frame.DataFrame"
        ));
        assert!(!is_pandas_dataframe_name("dict"));
    }
}
