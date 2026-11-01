//! グループの同一性・時点を表すキー列の検査と、整数コードへの分解。

use std::collections::HashMap;

use engine::shared::group_codes::GroupCodes;
use polars::prelude::*;
use pyo3::prelude::*;

use super::dtype::{KeyRole, check_key_dtype};
use super::frame::{missing_column_error, present_value, reject_missing_values};
use crate::shared::column_error::{ColumnExtractionError, ColumnRole, ColumnSource};
use crate::shared::errors::common_error_to_pyerr;

/// `df`から`name`列を、同一性だけが意味を持つグループキー（`effects`の先頭の列・`cluster`）として
/// 文字列のVecで取り出す。
///
/// 行ごとに`String`を作る素朴な実装で、本番のコードは[`extract_cluster_groups`]・
/// [`extract_entity_groups`]（行ごとの`String`を作らない）を使う。ここは、それらが
/// 本関数と同じコードになることを確かめるテストの比較対象（独立な参照実装）として置いている。
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
#[cfg(test)]
pub fn extract_group_key_column(
    df: &DataFrame,
    name: &str,
    source: ColumnSource<'_>,
) -> PyResult<Vec<String>> {
    extract_key_column(df, name, KeyRole::Identity, source)
}

/// 検査済みのキー列を、値そのもので分解した結果。
pub(crate) struct FactorizedKey {
    /// 各行の仮コード（出現順、値は`0..n_groups`）。
    pub(crate) codes: Vec<usize>,
    /// 仮グループごとの代表値（最初に現れた行の値、元のdtype）。
    pub(crate) representatives: Series,
}

/// 検査済みのキー列`column`を、値そのもの（文字列・整数の内部表現・浮動小数のビット列）で
/// ハッシュして仮コードに分解する。行ごとの`String`は作らない。
pub(crate) fn factorize_checked_key(
    name: &str,
    column: &Column,
) -> Result<FactorizedKey, ColumnExtractionError> {
    let (codes, first_rows) = factorize_key_column(name, column)?;
    let first_rows: Vec<IdxSize> = first_rows.iter().map(|&row| row as IdxSize).collect();
    let representatives = column
        .as_materialized_series()
        .take(&IdxCa::from_vec("first_rows".into(), first_rows))
        .map_err(|e| ColumnExtractionError::conversion_failed(name, e))?;
    Ok(FactorizedKey {
        codes,
        representatives,
    })
}

/// 仮グループごとの代表値を文字列にして`build`に渡す。辞書順・同じ文字列のまとめ方は
/// `engine`の`GroupCodes`が持つ規則で、ここでは文字列表現を作るだけ。
pub(crate) fn with_group_labels<T>(
    name: &str,
    representatives: &Series,
    build: impl FnOnce(&[&str]) -> PyResult<T>,
) -> PyResult<T> {
    let strings = cast_series_to_string(name, representatives)?;
    let ca = strings
        .str()
        .map_err(|e| ColumnExtractionError::conversion_failed(name, e))?;
    // キャストは非strictで、文字列にできない値はnullになりうる。panicせず`Err`にする。
    let labels: Vec<&str> = ca
        .iter()
        .map(|v| {
            v.ok_or_else(|| {
                ColumnExtractionError::conversion_failed(
                    name,
                    "a value could not be converted to a string",
                )
            })
        })
        .collect::<Result<_, _>>()?;
    build(&labels)
}

/// `df`から`name`列を、クラスター列（`cluster`）として整数コード化した`GroupCodes`で取り出す。
///
/// 行ごとの`String`は作らない。列の値そのもの（文字列・整数・浮動小数のビット列）でハッシュして
/// 仮コードを振り、仮グループごと（ユニークな値ごと）にだけ文字列表現を作って
/// `GroupCodes::from_factorized_without_keys`に渡す。コードの順序（ラベルの辞書順）と、
/// 同じ文字列になる値のまとめ方は`engine`側がこの型の規則として持つため、
/// [`extract_group_key_column`]の結果を`GroupCodes::from_labels_without_keys`に渡した場合と
/// 同じコードになる。クラスターのラベル自体は結果に出ないため、キーの`String`も作らない。
///
/// # Errors
/// [`extract_group_key_column`]と同じ。
fn extract_cluster_groups(
    df: &DataFrame,
    name: &str,
    source: ColumnSource<'_>,
) -> PyResult<GroupCodes> {
    let column = checked_key_column(df, name, KeyRole::Identity, source)?;
    let FactorizedKey {
        codes,
        representatives,
    } = factorize_checked_key(name, column)?;
    with_group_labels(name, &representatives, |labels| {
        GroupCodes::from_factorized_without_keys(codes, labels).map_err(common_error_to_pyerr)
    })
}

/// オプションのクラスター列名（`None`＝未指定）を[`extract_cluster_groups`]でコード化する。
/// `cov_type="cluster"`・`gmm_weight_type="cluster"`の各パース関数が共有する「指定があれば
/// 抽出する」ための入口で、`None`をどう扱うか（既定の列を使うか、エラーにするか）は
/// 呼び出し先の`engine`側が決める。
///
/// # Errors
/// 列名が指定されていて、[`extract_cluster_groups`]が失敗した場合。
pub fn extract_optional_cluster_groups(
    df: &DataFrame,
    name: Option<&str>,
    source: ColumnSource<'_>,
) -> PyResult<Option<GroupCodes>> {
    name.map(|name| extract_cluster_groups(df, name, source))
        .transpose()
}

/// `df`から`name`列を、パネルのエンティティ列（`effects`の先頭の列）として整数コード化した`GroupCodes`で
/// 取り出す。[`extract_cluster_groups`]と同じ分解で、グループのキー（ラベル）も作る
/// （`fixed_effects()`のキーに使う）。
///
/// # Errors
/// [`extract_group_key_column`]と同じ。
pub fn extract_entity_groups(
    df: &DataFrame,
    name: &str,
    source: ColumnSource<'_>,
) -> PyResult<GroupCodes> {
    let column = checked_key_column(df, name, KeyRole::Identity, source)?;
    let FactorizedKey {
        codes,
        representatives,
    } = factorize_checked_key(name, column)?;
    with_group_labels(name, &representatives, |labels| {
        GroupCodes::from_factorized(codes, labels).map_err(common_error_to_pyerr)
    })
}

/// キー列`name`の存在・dtype・欠損値・（浮動小数の）非有限値を検査し、問題が無ければその列を返す。
pub(crate) fn checked_key_column<'a>(
    df: &'a DataFrame,
    name: &str,
    role: KeyRole,
    source: ColumnSource<'_>,
) -> PyResult<&'a Column> {
    let column_role = match role {
        KeyRole::Identity => ColumnRole::GroupIdentifier,
        KeyRole::Time => ColumnRole::Time,
    };
    let series = df
        .column(name)
        .map_err(|_| missing_column_error(df, name, column_role, source))?;

    check_key_dtype(name, series.dtype(), role, source)?;

    reject_missing_values(name, series.null_count(), source)?;

    // 浮動小数のキーのNaN・無限大は、nullと違い`null_count`に現れず、文字列化すると
    // `"NaN"`という1つのグループになってしまう。数値列と同じく拒否する。
    if series.dtype().is_float() {
        reject_non_finite(name, series, source)?;
    }
    Ok(series)
}

/// キー列をUtf8にキャストする（元の型が数値・カテゴリカルでもよい。文字列表現で比較する）。
#[cfg(test)]
fn cast_key_column_to_string(name: &str, series: &Column) -> Result<Series, ColumnExtractionError> {
    cast_series_to_string(name, series.as_materialized_series())
}

fn cast_series_to_string(name: &str, series: &Series) -> Result<Series, ColumnExtractionError> {
    series
        .cast(&DataType::String)
        .map_err(|e| ColumnExtractionError::conversion_failed(name, e))
}

#[cfg(test)]
pub(crate) fn extract_key_column(
    df: &DataFrame,
    name: &str,
    role: KeyRole,
    source: ColumnSource<'_>,
) -> PyResult<Vec<String>> {
    let series = checked_key_column(df, name, role, source)?;
    let series = cast_key_column_to_string(name, series)?;
    let ca = series
        .str()
        .map_err(|e| ColumnExtractionError::conversion_failed(name, e))?;

    ca.iter()
        .map(|v| present_value(name, v).map(str::to_string))
        .collect::<Result<Vec<String>, _>>()
        .map_err(Into::into)
}

/// 各行のキーに出現順の仮コードを振る。戻り値は（行ごとの仮コード、仮コードごとの最初の行）。
///
/// `HashMap`は値の検索にだけ使い、反復しない（反復順序がハッシュシードに依存する非決定性は
/// 起きない。コードの順序は`engine`の`GroupCodes`がラベルの辞書順に振り直す）。
fn factorize_keys<K: std::hash::Hash + Eq>(
    keys: impl Iterator<Item = Result<K, ColumnExtractionError>>,
) -> Result<(Vec<usize>, Vec<usize>), ColumnExtractionError> {
    let mut seen: HashMap<K, usize> = HashMap::new();
    let mut first_rows: Vec<usize> = Vec::new();
    let mut codes: Vec<usize> = Vec::with_capacity(keys.size_hint().0);
    for (row, key) in keys.enumerate() {
        let code = *seen.entry(key?).or_insert_with(|| {
            first_rows.push(row);
            first_rows.len() - 1
        });
        codes.push(code);
    }
    Ok((codes, first_rows))
}

/// キー列の値そのもの（文字列・整数・浮動小数のビット列）でハッシュして、行ごとの仮コードと
/// 仮コードごとの最初の行を求める。行ごとの`String`は作らない。
///
/// 値の同一性は元のdtypeの値の等しさで判定する。ラベルの文字列表現との対応は、呼び出し側が
/// 仮グループごとに文字列化して`GroupCodes::from_factorized_without_keys`に渡すとき、同じ
/// 文字列になる仮グループがまとめられるため、文字列で比べた場合と同じグルーピングになる。
fn factorize_key_column(
    name: &str,
    series: &Column,
) -> Result<(Vec<usize>, Vec<usize>), ColumnExtractionError> {
    let convert = |e: PolarsError| ColumnExtractionError::conversion_failed(name, e);
    let dtype = series.dtype();
    if dtype.is_float() {
        let as_f64 = series.cast(&DataType::Float64).map_err(convert)?;
        let ca = as_f64.f64().map_err(convert)?;
        return factorize_keys(ca.iter().map(|v| present_value(name, v).map(f64::to_bits)));
    }
    if matches!(dtype, DataType::String) {
        let ca = series.str().map_err(convert)?;
        return factorize_keys(ca.iter().map(|v| present_value(name, v)));
    }
    // 整数・真偽値・`Date`・`Datetime`・`Categorical`・`Enum`は、内部表現の整数の等しさで
    // 判定する（`Categorical`/`Enum`は列内でカテゴリごとに異なる整数）。
    let as_i128 = series
        .to_physical_repr()
        .cast(&DataType::Int128)
        .map_err(convert)?;
    let ca = as_i128.i128().map_err(convert)?;
    factorize_keys(ca.iter().map(|v| present_value(name, v)))
}

/// 浮動小数の列に非有限値（NaN・無限大）があれば、最初の1件を`ValidationError`にする。
fn reject_non_finite(
    name: &str,
    series: &Column,
    source: ColumnSource<'_>,
) -> Result<(), ColumnExtractionError> {
    let as_f64 = series
        .cast(&DataType::Float64)
        .map_err(|e| ColumnExtractionError::conversion_failed(name, e))?;
    let ca = as_f64
        .f64()
        .map_err(|e| ColumnExtractionError::conversion_failed(name, e))?;
    if let Some((row, bad_value)) = ca
        .iter()
        .enumerate()
        .find_map(|(row, v)| v.filter(|v| !v.is_finite()).map(|v| (row, v)))
    {
        return Err(ColumnExtractionError::non_finite(
            name, row, bad_value, source,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::dtype::dtype_label;
    use super::*;
    use polars::df;

    const SRC: ColumnSource<'static> = ColumnSource::new("x", "data");

    #[test]
    fn extract_group_key_column_rejects_nan_and_infinity_in_float_keys() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let df = df!("k" => [1.0, bad, 2.0]).unwrap();

            assert!(extract_group_key_column(&df, "k", SRC).is_err(), "{bad}");
        }
    }

    #[test]
    fn extract_group_key_column_accepts_finite_float_keys() {
        let df = df!("k" => [1.0, 2.0, 1.0]).unwrap();

        let keys = extract_group_key_column(&df, "k", SRC).unwrap();

        assert_eq!(keys, vec!["1.0", "2.0", "1.0"]);
    }

    #[test]
    fn extract_cluster_groups_codes_labels_in_string_order_and_propagates_errors() {
        // 整数IDも文字列表現の辞書順（"10" < "9"）でコード化する（`BTreeMap<&str, _>`に入れたときの反復順と同じ）。
        let df = df!("k" => [9_i64, 10, 9, 2]).unwrap();

        let groups = extract_cluster_groups(&df, "k", SRC).unwrap();

        // 文字列の辞書順は "10" < "2" < "9"。順序が同じ別ラベル（a < b < c）で作ったコードと等しい。
        let by_string_order = ["c", "a", "c", "b"].map(String::from);
        assert_eq!(
            groups,
            GroupCodes::from_labels_without_keys(&by_string_order)
        );

        // 欠損値・存在しない列は`extract_group_key_column`と同じく`Err`。
        let with_null = df!("k" => [Some(1_i64), None]).unwrap();
        assert!(extract_cluster_groups(&with_null, "k", SRC).is_err());
        assert!(extract_cluster_groups(&df, "missing", SRC).is_err());
    }

    #[test]
    fn extract_optional_cluster_groups_passes_none_through_and_extracts_when_named() {
        let df = df!("k" => [9_i64, 10, 9, 2]).unwrap();

        assert!(matches!(
            extract_optional_cluster_groups(&df, None, SRC),
            Ok(None)
        ));

        let Ok(Some(groups)) = extract_optional_cluster_groups(&df, Some("k"), SRC) else {
            panic!("a named column must be extracted");
        };
        let Ok(expected) = extract_cluster_groups(&df, "k", SRC) else {
            panic!("extract_cluster_groups failed");
        };
        assert_eq!(groups, expected);

        // 指定した列が無い・不正な場合は`None`にせず`Err`。
        assert!(extract_optional_cluster_groups(&df, Some("missing"), SRC).is_err());
    }

    /// 行ごとの`String`を経由する`extract_group_key_column`＋`from_labels_without_keys`
    /// （参照実装）と、値そのものをハッシュして仮コードを振る`extract_cluster_groups`が、
    /// キーに使える全てのdtypeで同じコードになる。
    #[test]
    fn extract_cluster_groups_agrees_with_the_string_label_path_for_every_key_dtype() {
        let strings = ["b", "10", "a", "9", "b", "", "10", "é", "a"];
        let ints: [i64; 9] = [3, -1, 3, 10, 9, 0, -1, 100, 9];
        // 0.0と-0.0は文字列表現が異なる別グループ。0.1+0.2と0.3も別の値。
        let floats: [f64; 9] = [0.0, -0.0, 1.5, 0.1 + 0.2, 0.3, 1.5, 0.0, 1e30, -2.0];
        let bools = [true, false, true, true, false, false, true, false, true];

        let mut cases: Vec<(String, Series)> = vec![
            ("String".into(), Series::new("k".into(), strings)),
            ("Boolean".into(), Series::new("k".into(), bools)),
            ("Float64".into(), Series::new("k".into(), floats)),
            (
                "Float32".into(),
                Series::new("k".into(), floats.map(|v| v as f32)),
            ),
        ];
        let int_series = Series::new("k".into(), ints);
        for dtype in [
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::Int128,
        ] {
            cases.push((dtype_label(&dtype), int_series.cast(&dtype).unwrap()));
        }
        let unsigned = Series::new("k".into(), ints.map(|v| v.unsigned_abs()));
        for dtype in [
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
        ] {
            cases.push((dtype_label(&dtype), unsigned.cast(&dtype).unwrap()));
        }
        cases.push((
            "Date".into(),
            Series::new("k".into(), ints.map(|v| v as i32))
                .cast(&DataType::Date)
                .unwrap(),
        ));
        for unit in [
            TimeUnit::Milliseconds,
            TimeUnit::Microseconds,
            TimeUnit::Nanoseconds,
        ] {
            cases.push((
                format!("Datetime({unit:?})"),
                int_series.cast(&DataType::Datetime(unit, None)).unwrap(),
            ));
        }
        cases.push((
            "Categorical".into(),
            Series::new("k".into(), strings)
                .cast(&DataType::from_categories(Categories::global()))
                .unwrap(),
        ));

        for (label, series) in cases {
            assert_cluster_groups_agree_with_string_labels(&label, series);
        }
    }

    /// 行ごとの`String`を経由する経路（`extract_group_key_column`→`from_labels_without_keys`）と`extract_cluster_groups`の
    /// コードが等しいことを確認する。`PyErr`の`Debug`はGIL未初期化のテスト環境で二重panicするため、
    /// `unwrap()`は使わない。
    fn assert_cluster_groups_agree_with_string_labels(label: &str, series: Series) {
        let df = frame_of(series);
        let Ok(labels) = extract_group_key_column(&df, "k", SRC) else {
            panic!("{label}: extract_group_key_column failed");
        };
        let expected = GroupCodes::from_labels_without_keys(&labels);
        let Ok(actual) = extract_cluster_groups(&df, "k", SRC) else {
            panic!("{label}: extract_cluster_groups failed");
        };
        assert_eq!(actual, expected, "{label}");

        // エンティティ列用（キー付き）も、`from_labels`経由の結果と同じコード・同じキーになる。
        let expected_keyed = GroupCodes::from_labels(&labels);
        let Ok(actual_keyed) = extract_entity_groups(&df, "k", SRC) else {
            panic!("{label}: extract_entity_groups failed");
        };
        assert_eq!(actual_keyed, expected_keyed, "{label}");
        // キー（ユニークなラベル）は`FeInput::entity_labels`経由で確認する（`GroupCodes::keys`は
        // `engine`内部専用）。
        let y = vec![0.0; labels.len()];
        let labels_of = |codes: GroupCodes| {
            let Ok(input) = engine::panel::fe::FeInput::from_group_codes(
                &y,
                &[],
                vec![],
                codes,
                None,
                "y".into(),
            ) else {
                panic!("{label}: from_group_codes failed");
            };
            input.entity_labels().to_vec()
        };
        assert_eq!(
            labels_of(actual_keyed),
            labels_of(expected_keyed),
            "{label}"
        );
    }

    /// 値の範囲・書式・チャンクに関する境界のdtype別ケース: `Enum`、`Float16`、`i64`の範囲外の
    /// `UInt64`/`Int128`、秒境界ちょうどと端数ありが混在する`Datetime`、複数チャンクの列。
    #[test]
    fn extract_cluster_groups_agrees_with_the_string_label_path_for_boundary_values() {
        let mut cases: Vec<(String, Series)> = Vec::new();

        let enum_dtype = {
            let categories = FrozenCategories::new(["z", "10", "9", "a"]).unwrap();
            DataType::Enum(categories.clone(), categories.mapping().clone())
        };
        cases.push((
            "Enum".into(),
            Series::new("k".into(), ["10", "z", "a", "10", "9", "z"])
                .cast(&enum_dtype)
                .unwrap(),
        ));

        cases.push((
            "Float16".into(),
            Series::new("k".into(), [0.1_f64, 0.2, 0.1, -0.0, 0.0, 65504.0, 0.1])
                .cast(&DataType::Float16)
                .unwrap(),
        ));

        // `i64`に収まらない値と、`i64`の端の値（`i64::MAX`と`i64::MAX as u64 + 1`は別グループ）。
        cases.push((
            "UInt64".into(),
            Series::new(
                "k".into(),
                [
                    u64::MAX,
                    0,
                    i64::MAX as u64,
                    i64::MAX as u64 + 1,
                    u64::MAX,
                    7,
                ],
            ),
        ));
        cases.push((
            "Int128".into(),
            Series::new(
                "k".into(),
                [
                    i64::MAX as i128 + 1,
                    -(i64::MAX as i128) - 2,
                    0,
                    5,
                    0,
                    i64::MAX as i128 + 1,
                ]
                .map(|v| v as f64),
            )
            .cast(&DataType::Int128)
            .unwrap(),
        ));

        // 秒境界ちょうどの値（小数部なし）と端数ありの値が同じ列に混在しても、書式は値ごとに
        // 決まるため、ユニークな値だけを文字列化しても列全体を文字列化した場合と同じになる。
        for unit in [
            TimeUnit::Milliseconds,
            TimeUnit::Microseconds,
            TimeUnit::Nanoseconds,
        ] {
            let per_second = match unit {
                TimeUnit::Milliseconds => 1_000_i64,
                TimeUnit::Microseconds => 1_000_000,
                TimeUnit::Nanoseconds => 1_000_000_000,
            };
            cases.push((
                format!("Datetime({unit:?}) mixed seconds"),
                Series::new(
                    "k".into(),
                    [0, per_second, per_second + 1, 2 * per_second, 1, per_second],
                )
                .cast(&DataType::Datetime(unit, None))
                .unwrap(),
            ));
        }

        // 複数チャンク（`append`で連結。チャンクをまたいで同じ値が現れる）。
        let mut chunked = Series::new("k".into(), ["b", "a", "c"]);
        chunked
            .append(&Series::new("k".into(), ["a", "d", "b"]))
            .unwrap();
        assert!(chunked.n_chunks() > 1);
        cases.push(("String (chunked)".into(), chunked));
        let mut chunked_int = Series::new("k".into(), [3_i64, 1, 3]);
        chunked_int
            .append(&Series::new("k".into(), [10_i64, 1, 2]))
            .unwrap();
        assert!(chunked_int.n_chunks() > 1);
        cases.push(("Int64 (chunked)".into(), chunked_int));

        for (label, series) in cases {
            assert_cluster_groups_agree_with_string_labels(&label, series);
        }
    }

    #[test]
    fn extract_cluster_groups_handles_empty_columns() {
        for (label, series) in [
            ("Int64", Series::new("k".into(), Vec::<i64>::new())),
            ("String", Series::new("k".into(), Vec::<String>::new())),
            ("Float64", Series::new("k".into(), Vec::<f64>::new())),
        ] {
            assert_cluster_groups_agree_with_string_labels(label, series);
        }
    }

    fn frame_of(series: Series) -> DataFrame {
        DataFrame::new(series.len(), vec![series.into()]).unwrap()
    }
}
