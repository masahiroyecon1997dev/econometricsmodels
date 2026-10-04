//! polars DataFrameから検証済みの列を取り出す、全手法共通のユーティリティ。
//!
//! 【方針】欠損値（null、およびf64列ではNaN/無限大）は常にエラーとする（自動除外はしない）。
//! 理由: `docs/guide/validation.md`を参照（暗黙のサンプル除外という恣意的な判断を避け、サンプル
//! セレクションバイアスに気づかないまま推定されるのを防ぐため、除外・補完の判断はユーザーに明示的に
//! させる）。この方針はOLSに限らず全手法で共通。
//! polarsのnull（値が存在しない）とIEEE754のNaN（値は存在するが数値として無効）は別概念であり、
//! 両方を検出する必要がある。
//!
//! 【polarsのバージョン依存に関する注意（検証・修正済み）】
//! polars 0.55.2での実ビルドを確認済み（0.54.4でも同様）。当初の草案から2点修正した:
//! `ChunkedArray`の`.rechunk()`が`Cow<'_, ChunkedArray<T>>`を返すようになった影響で
//! （`IntoIterator`が実装されなくなったため）、値の取り出しは`.into_iter()`ではなく
//! `.iter()`（`ChunkedArray::iter`メソッド）を使う。

use polars::prelude::*;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3_polars::PyDataFrame;

use engine::panel::common::TimeKeys;

use crate::errors::ValidationError;
use crate::panel::common::panel_error_to_pyerr;

/// Pythonオブジェクト`ob`をpolars DataFrameとして取り出す。
///
/// `PyDataFrame`の`FromPyObject`実装（`pyo3-polars`）は`get_columns`/`width`を
/// 無条件に呼ぶだけで型検証を行わないため、polars以外のDataFrame（pandas等）を
/// 渡すと内部実装が漏れた`AttributeError`がそのまま送出されてしまう
/// （`ob.call_method0("get_columns")`がpandas.DataFrameには存在しないメソッドで
/// 失敗するため）。この関数を経由することで、その失敗を`ValidationError`に
/// 変換して隠蔽する。`data`/`new_data`いずれのパラメータもこの関数を通す
/// （`#[pyfunction]`/`#[pymethods]`の引数型を`PyDataFrame`ではなく
/// `Bound<'_, PyAny>`にし、関数本体の先頭でこれを呼ぶ設計にする必要がある。
/// pyo3が引数抽出をパラメータの型から自動生成する関係上、`PyDataFrame`を
/// 引数型のまま使うと関数本体に入る前に`FromPyObject`が走ってしまい、
/// 本体内でこの変換を挟めないため）。
pub fn extract_dataframe(ob: &Bound<'_, PyAny>, param_name: &str) -> PyResult<PyDataFrame> {
    ob.extract::<PyDataFrame>().map_err(|err| {
        // 渡されたオブジェクト自体が本物の`polars.DataFrame`なのに抽出が失敗している場合
        // （`pyo3`/`polars`/`pyo3-polars`のバージョンの組み合わせによるABI不整合等、
        // `.claude/rules/rust-style.md`「既知のリスク」参照）は、「polars.DataFrameではない」
        // という文言が事実に反し診断の妨げになるため、元のエラーを含めた別文言にする。
        // `LazyFrame`・`Series`等、他のpolarsオブジェクトは通常の「DataFrameではない」扱い
        // （`polars.`で始まる型名で一括りにすると、内部実装が漏れたメッセージになる）。
        if is_instance_of_polars(ob, "DataFrame") {
            return ValidationError::new_err(format!(
                "failed to read '{param_name}' as a polars.DataFrame: {err}"
            ));
        }
        let type_name = type_name_of(ob);
        let hint = if is_instance_of_polars(ob, "LazyFrame") {
            "; call .collect() first"
        } else {
            ""
        };
        ValidationError::new_err(format!(
            "'{param_name}' must be a polars.DataFrame, got {type_name}{hint}"
        ))
    })
}

/// `ob`の型の完全修飾名（`str`・`pandas.core.frame.DataFrame`等）。エラーメッセージ用。
pub(crate) fn type_name_of(ob: &Bound<'_, PyAny>) -> String {
    ob.get_type()
        .fully_qualified_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "unknown type".to_string())
}

/// 単一の列名を受け取る引数（`y`・`entity`・`weight`等）を`String`として取り出す。
///
/// 型の誤りは`ValueError`系の`ValidationError`ではなく、Pythonの慣習どおり`TypeError`に
/// する。PyO3の標準の引数抽出は引数名を含まないメッセージになるため、`#[pyfunction]`の
/// 引数型を`&Bound<PyAny>`にしてこの関数で取り出す（`extract_dataframe`と同じ構成）。
///
/// # Errors
/// `str`でなければ`TypeError`（引数名・実際の型を含む）。
pub fn extract_column_name(ob: &Bound<'_, PyAny>, param_name: &str) -> PyResult<String> {
    ob.extract::<String>().map_err(|_| {
        PyTypeError::new_err(format!(
            "'{param_name}' must be a str column name, got {}",
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
            "'{param_name}' must be a list of column names (e.g. {param_name}=[\"x1\"]), got {}",
            type_name_of(ob)
        ))
    })?;
    list.iter()
        .enumerate()
        .map(|(i, item)| {
            item.extract::<String>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "'{param_name}[{i}]' must be a str column name, got {}",
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

/// f64として取り出す列の使われ方。使われ方によって許可するdtypeが異なる。
#[derive(Clone, Copy, PartialEq, Eq)]
enum NumericRole {
    /// 値そのものを計算に使う列（`y`・`x`・重み・操作変数等）。
    Value,
    /// 行の並び順を決めるだけに使う列（HACの`hac_time`）。値の大小関係だけが意味を持つため、
    /// 順序が保たれる`Date`/`Datetime`も許可する。
    Ordering,
}

/// エラーメッセージ用のdtype名。Pythonの`pl.String`等と同じ呼び名にする
/// （polarsのDisplayは`str`・`cat`等の略称になるため）。内部パラメータ（時間単位・
/// 内側の型等）は含めない。
fn dtype_label(dtype: &DataType) -> String {
    match dtype {
        DataType::String => "String".to_string(),
        DataType::Boolean => "Boolean".to_string(),
        DataType::Decimal(..) => "Decimal".to_string(),
        DataType::Binary => "Binary".to_string(),
        DataType::Date => "Date".to_string(),
        DataType::Time => "Time".to_string(),
        DataType::Datetime(..) => "Datetime".to_string(),
        DataType::Duration(..) => "Duration".to_string(),
        DataType::Categorical(..) => "Categorical".to_string(),
        DataType::Enum(..) => "Enum".to_string(),
        DataType::List(_) => "List".to_string(),
        DataType::Array(..) => "Array".to_string(),
        DataType::Struct(_) => "Struct".to_string(),
        other => other.to_string(),
    }
}

/// `dtype`がf64に変換して使える数値の列かを検査し、そうでなければ`ValidationError`にする。
///
/// 整数・浮動小数・Boolean（`True`=1）・Decimalを許可する。文字列・日付・時刻・カテゴリ等は
/// キャストで黙って数値になったり欠損値扱いになったりして意図が判定できないため拒否する。
/// `Null`型（全値が欠損の列）は後続の欠損値チェックで「欠損値を含む」と報告させるため通す。
fn check_numeric_dtype(name: &str, dtype: &DataType, role: NumericRole) -> PyResult<()> {
    let is_ordering = role == NumericRole::Ordering;
    // `Boolean`は値が2種類しかなく、3行以上では必ず同値ができて順序が定まらないため、
    // 順序づけの列としては拒否する。
    let is_numeric = dtype.is_integer()
        || dtype.is_float()
        || dtype.is_decimal()
        || (dtype.is_bool() && !is_ordering)
        || dtype.is_null();
    let is_ordering_extra = matches!(dtype, DataType::Date | DataType::Datetime(..));
    if is_numeric || (is_ordering && is_ordering_extra) {
        return Ok(());
    }
    let label = dtype_label(dtype);
    let message = match role {
        NumericRole::Value => format!(
            "column '{name}' has dtype {label}, which cannot be used as a numeric column; \
             use an integer, float, boolean or decimal column (cast it first if it holds numbers)"
        ),
        NumericRole::Ordering => format!(
            "column '{name}' has dtype {label}, which cannot be used as a time-order column; \
             use an integer, float, Date or Datetime column"
        ),
    };
    Err(ValidationError::new_err(message))
}

/// `df`から`name`列をf64のVecとして取り出す。
///
/// 数値として使う列（`y`・`x`・重み・操作変数等）用。整数・浮動小数・Boolean・Decimalのみを
/// 受け付ける（`check_numeric_dtype`）。行の並び順だけに使う列は[`extract_time_order_ranks`]。
///
/// # Errors（すべて`ValidationError`）
/// - 列が存在しない
/// - 数値として使えないdtype（文字列・日付・時刻・カテゴリ等）
/// - 欠損値（null）を含む
/// - NaN・無限大（infinity）を含む
pub fn extract_f64_column(df: &DataFrame, name: &str) -> PyResult<Vec<f64>> {
    let series = df.column(name).map_err(|_| {
        ValidationError::new_err(format!("column '{name}' does not exist in the data"))
    })?;

    check_numeric_dtype(name, series.dtype(), NumericRole::Value)?;
    cast_to_finite_f64_values(name, series)
}

/// `df`から`name`列を、行の並び順を決めるための**順位**のVecで取り出す（HACの`hac_time`）。
///
/// 戻り値の`i`番目は、列の値を昇順に並べたときの`i`行目の位置（0始まり、`f64`）。エンジンは
/// 値の大小関係だけを使うため、値そのものではなく順位を渡す。f64への変換で値が潰れる
/// 整数（2^53超）・ナノ秒の`Datetime`でも、元のdtypeのまま比較するので順序が保たれる。
///
/// 値が1組でも重複する（全値が同一の列を含む）と順序が定まらない。エンジンは
/// 同値を行順で黙って並べてしまうため、ここで`ValidationError`にする。
///
/// 整数・`Date`・`Datetime`・`Decimal`は物理表現（`Int128`）で、浮動小数は
/// 値で比較する（`-0.0`と`0.0`は同値）。許可するdtypeは`check_numeric_dtype`の
/// `NumericRole::Ordering`（`extract_f64_column`の許可dtypeに`Date`/`Datetime`を加えたもの）。
///
/// # Errors（すべて`ValidationError`）
/// - 列が存在しない
/// - 順序づけに使えないdtype（文字列・時刻・カテゴリ等）
/// - 欠損値（null）を含む
/// - 浮動小数の列にNaN・無限大を含む
/// - 値が重複する
pub fn extract_time_order_ranks(df: &DataFrame, name: &str) -> PyResult<Vec<f64>> {
    let series = df.column(name).map_err(|_| {
        ValidationError::new_err(format!("column '{name}' does not exist in the data"))
    })?;
    let dtype = series.dtype();

    check_numeric_dtype(name, dtype, NumericRole::Ordering)?;

    if dtype.is_float() {
        let values = cast_to_finite_f64_values(name, series)?;
        return strict_ranks(name, &values);
    }

    reject_missing_values(name, series.null_count())?;
    // 整数は値そのもの、`Date`は日数、`Datetime`は時間単位ごとの経過時間、`Decimal`は
    // 同一列で共通のスケールを掛けた整数が物理表現になっており、その大小がそのまま順序になる（`extract_time_keys`と同じ。タイムゾーンの有無は順序に影響しない）。
    let as_i128 = series
        .to_physical_repr()
        .cast(&DataType::Int128)
        .map_err(|e| time_order_error(name, e))?;
    let values: Vec<i128> = as_i128
        .i128()
        .map_err(|e| time_order_error(name, e))?
        .iter()
        .map(|v| v.expect("null_countチェック済み"))
        .collect();
    strict_ranks(name, &values)
}

/// 欠損値（null）が1つでもあれば`ValidationError`にする。
fn reject_missing_values(name: &str, null_count: usize) -> PyResult<()> {
    if null_count > 0 {
        return Err(ValidationError::new_err(format!(
            "column '{name}' contains {null_count} missing value(s). Missing values are not \
             handled automatically; please impute or remove them before calling this function"
        )));
    }
    Ok(())
}

/// dtype検査済みの`series`を、欠損値・NaN・無限大が無いことを確かめたうえでf64のVecにする。
fn cast_to_finite_f64_values(name: &str, series: &Column) -> PyResult<Vec<f64>> {
    // 許可したdtypeはすべてf64にキャストできるため、ここでの失敗は想定していない
    // （防御的に`ValidationError`へ変換する）。
    let series = series.cast(&DataType::Float64).map_err(|e| {
        ValidationError::new_err(format!(
            "column '{name}' could not be cast to a numeric type (f64): {e}"
        ))
    })?;

    let ca = series
        .f64()
        .map_err(|e| ValidationError::new_err(format!("failed to convert column '{name}': {e}")))?;

    reject_missing_values(name, ca.null_count())?;

    // rechunk: 複数チャンクに分かれている場合に単一チャンクへ統合する。
    // 既に単一チャンクの場合は実質コピーが発生しない（安価な操作）。
    let ca = ca.rechunk();

    let values: Vec<f64> = match ca.cont_slice() {
        Ok(slice) => slice.to_vec(),
        Err(_) => {
            // 通常はここに来ないはずだが、フォールバックとしてイテレータ経由で構築
            ca.iter()
                .map(|v| v.expect("null_countチェック済み"))
                .collect()
        }
    };

    // polarsのnull_count()はNaN/無限大を検出しない（値としては存在するため）。
    // IEEE754のNaN・infinityは別途スキャンする必要がある。
    if let Some((row, bad_value)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(ValidationError::new_err(format!(
            "column '{name}' contains a non-finite value ({bad_value}) at row {row}. NaN and \
             infinite values are not handled automatically; please impute or remove them \
             before calling this function"
        )));
    }

    Ok(values)
}

/// `values`の昇順での位置（0始まり）を行ごとに返す。値が重複すれば`ValidationError`。
fn strict_ranks<T: PartialOrd>(name: &str, values: &[T]) -> PyResult<Vec<f64>> {
    rank_distinct_values(values).map_err(|(first, second)| {
        ValidationError::new_err(format!(
            "column '{name}' has the same value at rows {first} and {second}. A time-order \
             column must give every observation a distinct value, because tied observations \
             cannot be put in time order; make the values distinct, or omit the time-order \
             option to use the row order of the data"
        ))
    })
}

/// `values`の昇順での位置（0始まり）を行ごとに返す。同値の組があれば、同値の行のうち
/// 行番号が最も小さい行と、その同値グループの次の行の組を`Err`で返す（「最初に重複した
/// 2行」。値の大小ではなく行の並びで決まる）。
///
/// `values`はNaNを含まない前提（呼び出し元が検査済み）。同値の行は安定ソートで行番号の
/// 昇順に隣り合うため、報告する行の組は決定的になる。
fn rank_distinct_values<T: PartialOrd>(values: &[T]) -> Result<Vec<f64>, (usize, usize)> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // ソート後の隣接同値のうち、先頭の行番号が最小のもの（同値グループごとの最初の2行の
    // 中で、行の並びでもっとも早く現れる組）を選ぶ。
    if let Some(pair) = order
        .windows(2)
        .filter(|w| values[w[0]] == values[w[1]])
        .min_by_key(|w| w[0])
    {
        return Err((pair[0], pair[1]));
    }

    let mut ranks = vec![0.0; values.len()];
    for (rank, &row) in order.iter().enumerate() {
        ranks[row] = rank as f64;
    }
    Ok(ranks)
}

/// `param_names`から`x`列名部分を取り出す。`has_intercept`時は先頭の`"const"`、
/// `exclude_trailing`個は末尾の要素を除く（Tobitの`"sigma"`等、末尾に合成
/// パラメータ名を持つ手法向け。持たない手法は`exclude_trailing=0`を渡す）。
///
/// OLS/WLS/Logit/Probit/Tobitの`predict()`/`augment()`が`param_names`から`x`列名
/// だけを取り出す際に共通して使う規約（切片項の自動追加）を集約したもの。
pub fn x_column_names(
    param_names: &[String],
    has_intercept: bool,
    exclude_trailing: usize,
) -> &[String] {
    let start = usize::from(has_intercept);
    let end = param_names.len() - exclude_trailing;
    &param_names[start..end]
}

/// `df`から`names`（複数の列名）を、`names`と同じ順序の`Vec<Vec<f64>>`としてまとめて
/// 取り出す（`extract_f64_column`を列ごとに呼ぶだけの薄いラッパー）。`predict()`/
/// `augment()`の`x`列抽出ループがOLS/WLS/Logit/Probit/Tobitで重複していたため、
/// 共通ユーティリティとしてここに集約した。
///
/// # Errors
/// 各列につき`extract_f64_column`と同じ（列が存在しない・数値として使えないdtype・
/// 欠損値/NaN/無限大を含む場合に`ValidationError`）。最初にエラーになった列で打ち切る。
pub fn extract_f64_columns(df: &DataFrame, names: &[String]) -> PyResult<Vec<Vec<f64>>> {
    names
        .iter()
        .map(|name| extract_f64_column(df, name))
        .collect()
}

/// キー列（グループの同一性や時点を表す列）の使われ方。許可するdtypeが異なる。
#[derive(Clone, Copy, PartialEq, Eq)]
enum KeyRole {
    /// グループの同一性だけが意味を持つ列（`entity`・`cluster`）。
    Identity,
    /// 時点を表す列（FE/REの`time`・`dk_time`）。
    Time,
}

/// `dtype`がキー列として使えるかを検査し、そうでなければ`ValidationError`にする。
///
/// 整数・浮動小数・文字列・Categorical/Enum・`Date`・タイムゾーンなしの`Datetime`は両方の
/// 役割で許可する。`Boolean`は同一性のキーだけで許可する。タイムゾーンなしの`Datetime`は
/// 文字列表現の桁数が固定のため辞書順が時系列順に一致する。タイムゾーン付きの`Datetime`は
/// 拒否する（組み込んだpolarsが文字列化できず、夏時間の重複する1時間の順序の問題もあるため）。
/// `Null`型は後続の欠損値チェックに回す。
fn check_key_dtype(name: &str, dtype: &DataType, role: KeyRole) -> PyResult<()> {
    // タイムゾーン付きの`Datetime`は、この拡張に組み込んだpolarsがタイムゾーンを扱えず
    // 文字列化に失敗する。原因の分からないエラーにならないよう、対処法を示して拒否する。
    if let DataType::Datetime(_, Some(time_zone)) = dtype {
        let role_label = match role {
            KeyRole::Identity => "group identifier",
            KeyRole::Time => "time",
        };
        return Err(ValidationError::new_err(format!(
            "column '{name}' is a Datetime with time zone '{time_zone}', which cannot be used as a \
             {role_label} column; remove the time zone first, for example with \
             .dt.replace_time_zone(None) after converting to the zone you want to keep"
        )));
    }
    let common = dtype.is_integer()
        || dtype.is_float()
        || dtype.is_null()
        || matches!(
            dtype,
            DataType::String | DataType::Categorical(..) | DataType::Enum(..)
        );
    let by_role = match role {
        KeyRole::Identity => matches!(
            dtype,
            DataType::Boolean | DataType::Date | DataType::Datetime(..)
        ),
        KeyRole::Time => matches!(dtype, DataType::Date | DataType::Datetime(..)),
    };
    if common || by_role {
        return Ok(());
    }
    let label = dtype_label(dtype);
    let message = match role {
        KeyRole::Identity => format!(
            "column '{name}' has dtype {label}, which cannot be used as a group identifier \
             column; use an integer, float, string, categorical, boolean, Date or Datetime \
             column"
        ),
        KeyRole::Time => format!(
            "column '{name}' has dtype {label}, which cannot be used as a time column; \
             use an integer, float, string, categorical, Date or Datetime column"
        ),
    };
    Err(ValidationError::new_err(message))
}

/// `df`から`name`列を、同一性だけが意味を持つグループキー（`entity`・`cluster`）として
/// 文字列のVecで取り出す。
///
/// クラスター変数は整数IDとは限らない（州名・産業コード・企業ID等の文字列/
/// カテゴリカル変数であることが多い）ため、値そのものではなく「グループの
/// 同一性が判定できればよい」という前提でUtf8として扱う。
///
/// # Errors（すべて`ValidationError`）
/// - 列が存在しない
/// - キーとして使えないdtype（`List`・`Struct`・`Time`・`Duration`・`Decimal`等）
/// - 欠損値を含む
/// - 浮動小数の列にNaN・無限大を含む（数値列と同じく自動では扱わない）
pub fn extract_group_key_column(df: &DataFrame, name: &str) -> PyResult<Vec<String>> {
    extract_key_column(df, name, KeyRole::Identity)
}

/// `df`から`name`列を、時点を表すキー（FE/REの`time`・`dk_time`）の文字列のVecで取り出す。
///
/// [`extract_group_key_column`]と同じく文字列表現で扱い、許可するdtypeが異なる
/// （`Boolean`を除き`Datetime`を許可する）。時点の**順序**は含まない。順序が要る
/// 呼び出しは[`extract_time_keys`]を使う。
///
/// # Errors
/// [`extract_group_key_column`]と同じ（許可するdtypeだけが異なる）。
fn extract_time_key_column(df: &DataFrame, name: &str) -> PyResult<Vec<String>> {
    extract_key_column(df, name, KeyRole::Time)
}

/// `df`から`name`列を、時点を表すキー（FE/REの`time`・`dk_time`）として、ラベルと
/// **値の順序**を持つ`TimeKeys`で取り出す。
///
/// ラベルは文字列表現（同一性の判定と`fixed_effects()`のキーに使う）。時点の順序は列の
/// dtypeの値の順序で決める。文字列表現の辞書順は`1, 10, 11, 2, ...`のように時間順と
/// ずれるため使わない。
///
/// | dtype | 時点の順序 |
/// |---|---|
/// | 整数・`Date`・`Datetime`（タイムゾーンなし） | 値の昇順（`Date`・`Datetime`は時系列順） |
/// | 浮動小数 | 数値の昇順 |
/// | `Enum` | カテゴリの定義順 |
/// | 文字列・`Categorical` | ラベルの辞書順（バイト順） |
///
/// # Errors
/// [`extract_group_key_column`]と同じ（許可するdtypeだけが異なる）。
pub fn extract_time_keys(df: &DataFrame, name: &str) -> PyResult<TimeKeys> {
    let ids = extract_time_key_column(df, name)?;
    // `extract_time_key_column`が列の存在を確認済み。
    let series = df.column(name).map_err(|_| {
        ValidationError::new_err(format!("column '{name}' does not exist in the data"))
    })?;
    let dtype = series.dtype();

    if dtype.is_float() {
        let as_f64 = series
            .cast(&DataType::Float64)
            .map_err(|e| time_order_error(name, e))?;
        let values: Vec<f64> = as_f64
            .f64()
            .map_err(|e| time_order_error(name, e))?
            .iter()
            .map(|v| v.expect("null_countチェック済み"))
            .collect();
        TimeKeys::by_float(ids, &values).map_err(panel_error_to_pyerr)
    } else if dtype.is_integer()
        || matches!(
            dtype,
            DataType::Date | DataType::Datetime(..) | DataType::Enum(..)
        )
    {
        // 整数は値そのもの、`Date`は日数、`Datetime`は時間単位ごとの経過時間、`Enum`は
        // カテゴリの定義順の添字が内部表現の整数になっており、その大小がそのまま時間順になる。
        let as_i128 = series
            .to_physical_repr()
            .cast(&DataType::Int128)
            .map_err(|e| time_order_error(name, e))?;
        let values: Vec<i128> = as_i128
            .i128()
            .map_err(|e| time_order_error(name, e))?
            .iter()
            .map(|v| v.expect("null_countチェック済み"))
            .collect();
        TimeKeys::by_integer(ids, &values).map_err(panel_error_to_pyerr)
    } else {
        // 文字列・`Categorical`（順序を持たない型）はラベルの辞書順。
        Ok(TimeKeys::lexicographic(ids))
    }
}

fn time_order_error(name: &str, err: PolarsError) -> PyErr {
    ValidationError::new_err(format!(
        "column '{name}' could not be ordered as a time column: {err}"
    ))
}

fn extract_key_column(df: &DataFrame, name: &str, role: KeyRole) -> PyResult<Vec<String>> {
    let series = df.column(name).map_err(|_| {
        ValidationError::new_err(format!("column '{name}' does not exist in the data"))
    })?;

    check_key_dtype(name, series.dtype(), role)?;

    if series.null_count() > 0 {
        return Err(ValidationError::new_err(format!(
            "column '{name}' contains missing values"
        )));
    }

    // 浮動小数のキーのNaN・無限大は、nullと違い`null_count`に現れず、文字列化すると
    // `"NaN"`という1つのグループになってしまう。数値列と同じく拒否する。
    if series.dtype().is_float() {
        reject_non_finite(name, series)?;
    }

    // Utf8にキャストして文字列表現で比較する（元の型が数値・カテゴリカルでもよい）。
    let series = series.cast(&DataType::String).map_err(|e| {
        ValidationError::new_err(format!(
            "column '{name}' could not be interpreted as a group key: {e}"
        ))
    })?;
    let ca = series
        .str()
        .map_err(|e| ValidationError::new_err(format!("failed to convert column '{name}': {e}")))?;

    Ok(ca
        .iter()
        .map(|v| v.expect("null_countチェック済み").to_string())
        .collect())
}

/// 浮動小数の列に非有限値（NaN・無限大）があれば、最初の1件を`ValidationError`にする。
fn reject_non_finite(name: &str, series: &Column) -> PyResult<()> {
    let as_f64 = series
        .cast(&DataType::Float64)
        .map_err(|e| ValidationError::new_err(format!("failed to convert column '{name}': {e}")))?;
    let ca = as_f64
        .f64()
        .map_err(|e| ValidationError::new_err(format!("failed to convert column '{name}': {e}")))?;
    if let Some((row, bad_value)) = ca
        .iter()
        .enumerate()
        .find_map(|(row, v)| v.filter(|v| !v.is_finite()).map(|v| (row, v)))
    {
        return Err(ValidationError::new_err(format!(
            "column '{name}' contains a non-finite value ({bad_value}) at row {row}. NaN and \
             infinite values are not handled automatically; please impute or remove them \
             before calling this function"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    #[test]
    fn x_column_names_keeps_all_names_when_no_intercept_and_no_trailing_exclusion() {
        let param_names = names(&["x1", "x2"]);
        assert_eq!(x_column_names(&param_names, false, 0), &param_names[..]);
    }

    #[test]
    fn x_column_names_drops_leading_const_when_has_intercept() {
        let param_names = names(&["const", "x1", "x2"]);
        assert_eq!(x_column_names(&param_names, true, 0), &param_names[1..]);
    }

    #[test]
    fn x_column_names_drops_trailing_elements_for_exclude_trailing() {
        // Tobitの`param_names`が末尾に`"sigma"`を持つケースを想定。
        let param_names = names(&["const", "x1", "x2", "sigma"]);
        assert_eq!(x_column_names(&param_names, true, 1), &param_names[1..3]);
    }

    #[test]
    fn check_key_dtype_accepts_identity_and_time_dtypes_per_role() {
        let both = [
            DataType::Int32,
            DataType::UInt8,
            DataType::Float64,
            DataType::String,
            DataType::Null,
        ];
        for dtype in &both {
            for role in [KeyRole::Identity, KeyRole::Time] {
                assert!(check_key_dtype("k", dtype, role).is_ok(), "{dtype}");
            }
        }
        // Booleanは同一性のキーだけ。DateとDatetimeは両方の役割で許可する。
        assert!(check_key_dtype("k", &DataType::Boolean, KeyRole::Identity).is_ok());
        assert!(check_key_dtype("k", &DataType::Boolean, KeyRole::Time).is_err());
        let datetime = DataType::Datetime(TimeUnit::Microseconds, None);
        for role in [KeyRole::Identity, KeyRole::Time] {
            assert!(check_key_dtype("k", &DataType::Date, role).is_ok());
            assert!(check_key_dtype("k", &datetime, role).is_ok());
        }
    }

    #[test]
    fn check_key_dtype_rejects_datetime_with_time_zone() {
        let aware = DataType::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC));
        for role in [KeyRole::Identity, KeyRole::Time] {
            assert!(check_key_dtype("k", &aware, role).is_err());
        }
    }

    #[test]
    fn check_key_dtype_rejects_unusable_dtypes() {
        let rejected = [
            DataType::Time,
            DataType::Duration(TimeUnit::Microseconds),
            DataType::Decimal(18, 2),
            DataType::Binary,
            DataType::List(Box::new(DataType::Int64)),
        ];
        for dtype in &rejected {
            for role in [KeyRole::Identity, KeyRole::Time] {
                assert!(check_key_dtype("k", dtype, role).is_err(), "{dtype}");
            }
        }
    }

    #[test]
    fn extract_group_key_column_rejects_nan_and_infinity_in_float_keys() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let df = df!("k" => [1.0, bad, 2.0]).unwrap();

            assert!(extract_group_key_column(&df, "k").is_err(), "{bad}");
            assert!(extract_time_key_column(&df, "k").is_err(), "{bad}");
        }
    }

    #[test]
    fn extract_group_key_column_accepts_finite_float_keys() {
        let df = df!("k" => [1.0, 2.0, 1.0]).unwrap();

        let keys = extract_group_key_column(&df, "k").unwrap();

        assert_eq!(keys, vec!["1.0", "2.0", "1.0"]);
    }

    fn frame_of(series: Series) -> DataFrame {
        DataFrame::new(series.len(), vec![series.into()]).unwrap()
    }

    fn periods_of(df: &DataFrame) -> Vec<String> {
        extract_time_keys(df, "t").unwrap().periods().to_vec()
    }

    #[test]
    fn extract_time_keys_orders_integers_numerically() {
        let df = df!("t" => [10i64, 2, 9, -3, 2]).unwrap();

        assert_eq!(periods_of(&df), ["-3", "2", "9", "10"]);
    }

    #[test]
    fn extract_time_keys_orders_every_integer_dtype_numerically() {
        for dtype in [
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::Int128,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
        ] {
            let df = frame_of(
                Series::new("t".into(), [10i64, 2, 9, 2])
                    .cast(&dtype)
                    .unwrap(),
            );

            assert_eq!(periods_of(&df), ["2", "9", "10"], "{dtype}");
        }
    }

    #[test]
    fn extract_time_keys_orders_floats_numerically() {
        let df = df!("t" => [10.5, 2.0, -0.5, 9.25, 2.0]).unwrap();

        assert_eq!(periods_of(&df), ["-0.5", "2.0", "9.25", "10.5"]);
    }

    #[test]
    fn extract_time_keys_orders_dates_and_datetimes_chronologically() {
        let date_df = frame_of(
            Series::new("t".into(), [19_000i32, 18_000, 20_000])
                .cast(&DataType::Date)
                .unwrap(),
        );
        let datetime_df = frame_of(
            Series::new("t".into(), [3_000i64, 1_000, 2_000])
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .unwrap(),
        );

        let dates = periods_of(&date_df);
        let datetimes = periods_of(&datetime_df);

        // 日付・日時の文字列表現は桁数が固定で、辞書順も時系列順に一致する。1つずつ小さい値が
        // 先頭に来ていれば、内部表現の整数の順に並んでいる。
        assert_eq!(dates.len(), 3);
        assert!(dates.windows(2).all(|w| w[0] < w[1]), "{dates:?}");
        assert_eq!(datetimes.len(), 3);
        assert!(datetimes.windows(2).all(|w| w[0] < w[1]), "{datetimes:?}");
    }

    #[test]
    fn extract_time_keys_orders_strings_lexicographically() {
        let df = df!("t" => ["b", "10", "a", "9"]).unwrap();

        assert_eq!(periods_of(&df), ["10", "9", "a", "b"]);
    }

    #[test]
    fn extract_time_keys_rejects_unusable_columns_like_the_key_column() {
        let df = df!("t" => [1.0, f64::NAN]).unwrap();
        assert!(extract_time_keys(&df, "t").is_err());
        let df = df!("t" => [true, false]).unwrap();
        assert!(extract_time_keys(&df, "t").is_err());
        assert!(extract_time_keys(&df, "missing").is_err());
    }

    fn numeric_dtypes() -> Vec<DataType> {
        vec![
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::Int128,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
            DataType::Float32,
            DataType::Float64,
            DataType::Decimal(18, 2),
            DataType::Null,
        ]
    }

    #[test]
    fn check_numeric_dtype_accepts_numeric_dtypes_for_both_roles() {
        for dtype in numeric_dtypes() {
            for role in [NumericRole::Value, NumericRole::Ordering] {
                assert!(
                    check_numeric_dtype("x", &dtype, role).is_ok(),
                    "{dtype} should be accepted"
                );
            }
        }
    }

    #[test]
    fn check_numeric_dtype_accepts_boolean_only_for_value_role() {
        // 順序づけの列としては、値が2種類しかなく順序が定まらないため拒否する。
        assert!(check_numeric_dtype("x", &DataType::Boolean, NumericRole::Value).is_ok());
        assert!(check_numeric_dtype("x", &DataType::Boolean, NumericRole::Ordering).is_err());
    }

    #[test]
    fn check_numeric_dtype_rejects_non_numeric_dtypes_for_value_role() {
        let rejected = [
            DataType::String,
            DataType::Binary,
            DataType::Date,
            DataType::Time,
            DataType::Datetime(TimeUnit::Microseconds, None),
            DataType::Duration(TimeUnit::Microseconds),
            DataType::List(Box::new(DataType::Int64)),
        ];
        for dtype in rejected {
            assert!(
                check_numeric_dtype("x", &dtype, NumericRole::Value).is_err(),
                "{dtype} should be rejected"
            );
        }
    }

    #[test]
    fn check_numeric_dtype_accepts_date_and_datetime_only_for_ordering_role() {
        let date_like = [
            DataType::Date,
            DataType::Datetime(TimeUnit::Microseconds, None),
        ];
        for dtype in &date_like {
            assert!(check_numeric_dtype("t", dtype, NumericRole::Ordering).is_ok());
            assert!(check_numeric_dtype("t", dtype, NumericRole::Value).is_err());
        }
        // 時刻・期間・文字列は順序用でも拒否する。
        for dtype in [
            DataType::Time,
            DataType::Duration(TimeUnit::Microseconds),
            DataType::String,
        ] {
            assert!(check_numeric_dtype("t", &dtype, NumericRole::Ordering).is_err());
        }
    }

    #[test]
    fn extract_f64_column_rejects_string_column_even_if_values_look_numeric() {
        let df = df!("a" => ["1.0", "2.0"]).unwrap();

        assert!(extract_f64_column(&df, "a").is_err());
    }

    /// `values`を`dtype`にキャストした`t`列だけを持つ`DataFrame`。
    fn frame_with_t_as(values: Vec<i64>, dtype: DataType) -> DataFrame {
        let mut df = DataFrame::new(values.len(), vec![Column::new("t".into(), values)]).unwrap();
        let cast = df.column("t").unwrap().cast(&dtype).unwrap();
        df.with_column(cast).unwrap();
        df
    }

    #[test]
    fn extract_time_order_ranks_returns_positions_in_ascending_order() {
        let df = frame_with_t_as(vec![30_i64, 10, 20], DataType::Int64);

        let ranks = extract_time_order_ranks(&df, "t").unwrap();

        assert_eq!(ranks, vec![2.0, 0.0, 1.0]);
    }

    #[test]
    fn extract_time_order_ranks_keeps_date_order() {
        let df = frame_with_t_as(vec![3, 1, 2], DataType::Date);

        let ranks = extract_time_order_ranks(&df, "t").unwrap();

        assert_eq!(ranks, vec![2.0, 0.0, 1.0]);
    }

    #[test]
    fn extract_time_order_ranks_orders_floats_and_decimals_and_small_ints() {
        for dtype in [
            DataType::Float32,
            DataType::Float64,
            DataType::Decimal(18, 0),
            DataType::UInt8,
            DataType::Int128,
        ] {
            let df = frame_with_t_as(vec![5_i64, 1, 3], dtype.clone());

            let ranks = extract_time_order_ranks(&df, "t").unwrap();

            assert_eq!(ranks, vec![2.0, 0.0, 1.0], "{dtype}");
        }
    }

    #[test]
    fn extract_time_order_ranks_distinguishes_values_that_collapse_in_f64() {
        // 2^53を超える整数・ナノ秒の`Datetime`はf64に変換すると隣り合う値が同値になるが、
        // 元のdtypeで比較するので順序が保たれ、重複とも見なされない。
        let base = 1_700_000_000_000_000_000_i64;
        let values = vec![base + 2, base, base + 1];
        assert_eq!(values[1] as f64, values[2] as f64, "前提: f64では潰れる");
        for dtype in [
            DataType::Int64,
            DataType::Datetime(TimeUnit::Nanoseconds, None),
        ] {
            let df = frame_with_t_as(values.clone(), dtype.clone());

            let ranks = extract_time_order_ranks(&df, "t").unwrap();

            assert_eq!(ranks, vec![2.0, 0.0, 1.0], "{dtype}");
        }
    }

    #[test]
    fn extract_time_order_ranks_rejects_duplicates_in_every_orderable_dtype() {
        for dtype in [
            DataType::Int64,
            DataType::UInt64,
            DataType::Float32,
            DataType::Float64,
            DataType::Decimal(18, 0),
            DataType::Decimal(18, 2),
            DataType::Date,
            DataType::Datetime(TimeUnit::Microseconds, None),
        ] {
            // 行0と行2が同値（間に別の値を挟む）。
            let df = frame_with_t_as(vec![4, 9, 4, 7], dtype.clone());

            assert!(extract_time_order_ranks(&df, "t").is_err(), "{dtype}");
        }
    }

    #[test]
    fn rank_distinct_values_reports_the_first_tied_pair_by_row_order() {
        assert_eq!(rank_distinct_values(&[4, 9, 4, 7]), Err((0, 2)));
        // 値が最小の同値グループ（行2・3）ではなく、行番号が最も早い組（行0・1）を報告する。
        assert_eq!(rank_distinct_values(&[9, 9, 1, 1]), Err((0, 1)));
        assert_eq!(rank_distinct_values(&[1, 5, 9, 9, 5]), Err((1, 4)));
        // 3つ組の同値は先頭の2行を報告する。
        assert_eq!(rank_distinct_values(&[5.0, 1.0, 5.0, 5.0]), Err((0, 2)));
        assert_eq!(rank_distinct_values(&[4, 9, 2]), Ok(vec![1.0, 2.0, 0.0]));
        assert_eq!(rank_distinct_values::<i128>(&[]), Ok(vec![]));
    }

    #[test]
    fn extract_time_order_ranks_rejects_fully_tied_columns() {
        let tied = frame_with_t_as(vec![1_i64; 5], DataType::Int64);
        assert!(extract_time_order_ranks(&tied, "t").is_err());
    }

    #[test]
    fn extract_time_order_ranks_rejects_boolean_columns_by_dtype() {
        let boolean = frame_with_t_as(vec![0_i64, 1], DataType::Boolean);
        assert!(extract_time_order_ranks(&boolean, "t").is_err());
    }

    #[test]
    fn extract_time_order_ranks_treats_negative_zero_as_tied_with_zero() {
        let df = DataFrame::new(2, vec![Column::new("t".into(), vec![0.0_f64, -0.0])]).unwrap();

        assert!(extract_time_order_ranks(&df, "t").is_err());
    }

    #[test]
    fn extract_time_order_ranks_rejects_missing_and_non_finite_values() {
        let with_null = DataFrame::new(
            3,
            vec![Column::new("t".into(), vec![Some(1_i64), None, Some(3)])],
        )
        .unwrap();
        assert!(extract_time_order_ranks(&with_null, "t").is_err());

        let with_nan = DataFrame::new(
            3,
            vec![Column::new("t".into(), vec![1.0_f64, f64::NAN, 3.0])],
        )
        .unwrap();
        assert!(extract_time_order_ranks(&with_nan, "t").is_err());
    }

    #[test]
    fn extract_time_order_ranks_rejects_null_in_every_orderable_dtype() {
        // 浮動小数とそれ以外は別の抽出経路（キャスト後に検査するか、元のdtypeで検査するか）。
        for dtype in [
            DataType::Int8,
            DataType::UInt64,
            DataType::Float32,
            DataType::Float64,
            DataType::Decimal(18, 2),
            DataType::Date,
            DataType::Datetime(TimeUnit::Nanoseconds, None),
        ] {
            let mut df = DataFrame::new(
                3,
                vec![Column::new("t".into(), vec![Some(1_i64), None, Some(3)])],
            )
            .unwrap();
            let cast = df.column("t").unwrap().cast(&dtype).unwrap();
            df.with_column(cast).unwrap();

            assert!(extract_time_order_ranks(&df, "t").is_err(), "{dtype}");
        }
    }

    #[test]
    fn extract_time_order_ranks_rejects_all_null_column_and_infinity() {
        let all_null = DataFrame::new(
            2,
            vec![
                Column::new_empty("t".into(), &DataType::Null)
                    .extend_constant(AnyValue::Null, 2)
                    .unwrap(),
            ],
        )
        .unwrap();
        assert!(extract_time_order_ranks(&all_null, "t").is_err());

        for bad in [f64::INFINITY, f64::NEG_INFINITY] {
            for dtype in [DataType::Float32, DataType::Float64] {
                let mut df =
                    DataFrame::new(3, vec![Column::new("t".into(), vec![1.0_f64, bad, 3.0])])
                        .unwrap();
                let cast = df.column("t").unwrap().cast(&dtype).unwrap();
                df.with_column(cast).unwrap();

                assert!(extract_time_order_ranks(&df, "t").is_err(), "{dtype} {bad}");
            }
        }
    }

    #[test]
    fn extract_time_order_ranks_rejects_non_orderable_dtype_and_missing_column() {
        let strings = df!("t" => ["a", "b"]).unwrap();
        assert!(extract_time_order_ranks(&strings, "t").is_err());
        assert!(extract_time_order_ranks(&strings, "absent").is_err());
    }

    #[test]
    fn extract_f64_columns_preserves_name_order() {
        let df = df!(
            "a" => [1.0, 2.0],
            "b" => [10.0, 20.0],
        )
        .unwrap();

        let columns = extract_f64_columns(&df, &["b".to_string(), "a".to_string()]).unwrap();
        assert_eq!(columns, vec![vec![10.0, 20.0], vec![1.0, 2.0]]);
    }

    #[test]
    fn extract_f64_columns_returns_empty_vec_for_empty_names() {
        let df = df!("a" => [1.0, 2.0]).unwrap();

        let columns = extract_f64_columns(&df, &[]).unwrap();
        assert!(columns.is_empty());
    }

    #[test]
    fn extract_f64_columns_returns_error_when_a_column_is_missing() {
        let df = df!("a" => [1.0, 2.0]).unwrap();

        let result = extract_f64_columns(&df, &["a".to_string(), "does_not_exist".to_string()]);
        assert!(result.is_err());
    }
}
