//! 数値列（`f64`）・時点の順序づけに使う列・モデルの`x`列の抽出。

use polars::prelude::*;
use pyo3::prelude::*;

use super::dtype::{NumericRole, check_numeric_dtype};
use super::frame::{SelectedData, missing_column_error, reject_missing_values, select_columns};
use crate::shared::column_error::{ColumnExtractionError, ColumnRole, ColumnSource};

/// `new_data`から、`predict()`/`augment()`が使う`x`列（結果オブジェクトの`param_names`から決まる。
/// [`extract_model_x_columns`]と同じ規約）だけを取り込む。
pub fn select_model_x_columns(
    ob: &Bound<'_, PyAny>,
    param_name: &str,
    param_names: &[String],
    has_intercept: bool,
) -> PyResult<SelectedData> {
    select_columns(ob, param_name, x_column_names(param_names, has_intercept))
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
pub fn extract_f64_column(
    df: &DataFrame,
    name: &str,
    source: ColumnSource<'_>,
) -> PyResult<Vec<f64>> {
    let series = df
        .column(name)
        .map_err(|_| missing_column_error(df, name, ColumnRole::Numeric, source))?;

    check_numeric_dtype(name, series.dtype(), NumericRole::Value, source)?;
    Ok(cast_to_finite_f64_values(name, series, source)?)
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
pub fn extract_time_order_ranks(
    df: &DataFrame,
    name: &str,
    source: ColumnSource<'_>,
) -> PyResult<Vec<f64>> {
    let series = df
        .column(name)
        .map_err(|_| missing_column_error(df, name, ColumnRole::TimeOrder, source))?;
    let dtype = series.dtype();

    check_numeric_dtype(name, dtype, NumericRole::Ordering, source)?;

    if dtype.is_float() {
        let values = cast_to_finite_f64_values(name, series, source)?;
        return Ok(strict_ranks(name, &values, source)?);
    }

    reject_missing_values(name, series.null_count(), source)?;
    // 整数は値そのもの、`Date`は日数、`Datetime`は時間単位ごとの経過時間、`Decimal`は
    // 同一列で共通のスケールを掛けた整数が物理表現になっており、その大小がそのまま順序になる（`extract_time_keys`と同じ。タイムゾーンの有無は順序に影響しない）。
    let values = physical_i128(name, series.as_materialized_series())?;
    Ok(strict_ranks(name, &values, source)?)
}

/// dtype検査済みの`series`を、欠損値・NaN・無限大が無いことを確かめたうえでf64のVecにする。
fn cast_to_finite_f64_values(
    name: &str,
    series: &Column,
    source: ColumnSource<'_>,
) -> Result<Vec<f64>, ColumnExtractionError> {
    // 許可したdtypeはすべてf64にキャストできるため、ここでの失敗は想定していない
    // （防御的に内部エラー(`INTERNAL_ERROR`)へ変換する）。
    let series = series
        .cast(&DataType::Float64)
        .map_err(|e| ColumnExtractionError::conversion_failed(name, e))?;

    let ca = series
        .f64()
        .map_err(|e| ColumnExtractionError::conversion_failed(name, e))?;

    reject_missing_values(name, ca.null_count(), source)?;

    // rechunk: 複数チャンクに分かれている場合に単一チャンクへ統合する。
    // 既に単一チャンクの場合は実質コピーが発生しない（安価な操作）。
    let ca = ca.rechunk();

    let values: Vec<f64> = match ca.cont_slice() {
        Ok(slice) => slice.to_vec(),
        Err(_) => {
            // 通常はここに来ないはずだが、フォールバックとしてイテレータ経由で構築
            ca.iter()
                .map(|v| v.expect("null_count was checked to be zero"))
                .collect()
        }
    };

    // polarsのnull_count()はNaN/無限大を検出しない（値としては存在するため）。
    // IEEE754のNaN・infinityは別途スキャンする必要がある。
    if let Some((row, &bad_value)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(ColumnExtractionError::non_finite(
            name, row, bad_value, source,
        ));
    }

    Ok(values)
}

/// `values`の昇順での位置（0始まり）を行ごとに返す。値が重複すれば`TiedValues`。
fn strict_ranks<T: PartialOrd>(
    name: &str,
    values: &[T],
    source: ColumnSource<'_>,
) -> Result<Vec<f64>, ColumnExtractionError> {
    rank_distinct_values(values)
        .map_err(|(first, second)| ColumnExtractionError::tied_values(name, first, second, source))
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

/// `param_names`から`x`列名部分を取り出す。`has_intercept`時は先頭の`"const"`を除く。
/// Tobitも、`"sigma"`を含まない`k`長の係数名（`fitted.input().param_names()`）を渡す。
///
/// OLS/WLS/Logit/Probit/Tobitの`predict()`/`augment()`が`param_names`から`x`列名
/// だけを取り出す際に共通して使う規約（切片項の自動追加）。
pub fn x_column_names(param_names: &[String], has_intercept: bool) -> &[String] {
    &param_names[usize::from(has_intercept)..]
}

/// `df`から`names`（複数の列名）を、`names`と同じ順序の`Vec<Vec<f64>>`としてまとめて
/// 取り出す（`extract_f64_column`を列ごとに呼ぶだけの薄いラッパー）。OLS/WLS/Logit/
/// Probit/Tobitの`predict()`/`augment()`が共通で使う`x`列の抽出ループ。
///
/// # Errors
/// 各列につき`extract_f64_column`と同じ（列が存在しない・数値として使えないdtype・
/// 欠損値/NaN/無限大を含む場合に`ValidationError`）。最初にエラーになった列で打ち切る。
pub fn extract_f64_columns(
    df: &DataFrame,
    names: &[String],
    source: ColumnSource<'_>,
) -> PyResult<Vec<Vec<f64>>> {
    names
        .iter()
        .map(|name| extract_f64_column(df, name, source))
        .collect()
}

/// `param_names`から`x`列名を取り出し（`x_column_names`）、`df`からその列をまとめて抽出する
/// （`extract_f64_columns`）。結果オブジェクトが保持する`param_names`から学習データ・新規データの
/// `x`列を再抽出する`predict()`/`augment()`/事後診断の共通の手順を、1つにまとめたもの。
///
/// # Errors
/// `extract_f64_columns`と同じ。
pub fn extract_model_x_columns(
    df: &DataFrame,
    param_names: &[String],
    has_intercept: bool,
    source: ColumnSource<'_>,
) -> PyResult<Vec<Vec<f64>>> {
    extract_f64_columns(df, x_column_names(param_names, has_intercept), source)
}

pub(crate) fn time_order_error(name: &str, err: PolarsError) -> ColumnExtractionError {
    ColumnExtractionError::ordering_failed(name, err)
}

/// 時点の順序づけに使う列の物理表現を`i128`の`Vec`にする（整数は値そのもの、`Date`は日数、
/// `Datetime`は時間単位ごとの経過時間、`Decimal`は共通のスケールを掛けた整数、`Enum`は
/// カテゴリの定義順の添字。その大小がそのまま時間順になる）。欠損値（null）が無いことは
/// 呼び出し側が検査済みで、変換の失敗は`OrderingFailed`にする。
pub(crate) fn physical_i128(
    name: &str,
    series: &Series,
) -> Result<Vec<i128>, ColumnExtractionError> {
    let as_i128 = series
        .to_physical_repr()
        .cast(&DataType::Int128)
        .map_err(|e| time_order_error(name, e))?;
    as_i128
        .i128()
        .map_err(|e| time_order_error(name, e))?
        .iter()
        .map(|v| {
            v.ok_or_else(|| {
                ColumnExtractionError::ordering_failed(
                    name,
                    "unexpected null in a column checked to have none",
                )
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    const SRC: ColumnSource<'static> = ColumnSource::new("x", "data");

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    #[test]
    fn x_column_names_keeps_all_names_when_no_intercept() {
        let param_names = names(&["x1", "x2"]);
        assert_eq!(x_column_names(&param_names, false), &param_names[..]);
    }

    #[test]
    fn x_column_names_drops_leading_const_when_has_intercept() {
        let param_names = names(&["const", "x1", "x2"]);
        assert_eq!(x_column_names(&param_names, true), &param_names[1..]);
    }

    #[test]
    fn strict_ranks_reports_the_tied_rows() {
        assert!(matches!(
            strict_ranks("t", &[4, 9, 4], SRC),
            Err(ColumnExtractionError::TiedValues {
                first_row: 0,
                second_row: 2,
                ..
            })
        ));
        assert_eq!(
            strict_ranks("t", &[4, 9, 2], SRC).ok(),
            Some(vec![1.0, 2.0, 0.0])
        );
    }

    #[test]
    fn extract_f64_column_rejects_string_column_even_if_values_look_numeric() {
        let df = df!("a" => ["1.0", "2.0"]).unwrap();

        assert!(extract_f64_column(&df, "a", SRC).is_err());
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

        let ranks = extract_time_order_ranks(&df, "t", SRC).unwrap();

        assert_eq!(ranks, vec![2.0, 0.0, 1.0]);
    }

    #[test]
    fn extract_time_order_ranks_keeps_date_order() {
        let df = frame_with_t_as(vec![3, 1, 2], DataType::Date);

        let ranks = extract_time_order_ranks(&df, "t", SRC).unwrap();

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

            let ranks = extract_time_order_ranks(&df, "t", SRC).unwrap();

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

            let ranks = extract_time_order_ranks(&df, "t", SRC).unwrap();

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

            assert!(extract_time_order_ranks(&df, "t", SRC).is_err(), "{dtype}");
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
        assert!(extract_time_order_ranks(&tied, "t", SRC).is_err());
    }

    #[test]
    fn extract_time_order_ranks_rejects_boolean_columns_by_dtype() {
        let boolean = frame_with_t_as(vec![0_i64, 1], DataType::Boolean);
        assert!(extract_time_order_ranks(&boolean, "t", SRC).is_err());
    }

    #[test]
    fn extract_time_order_ranks_treats_negative_zero_as_tied_with_zero() {
        let df = DataFrame::new(2, vec![Column::new("t".into(), vec![0.0_f64, -0.0])]).unwrap();

        assert!(extract_time_order_ranks(&df, "t", SRC).is_err());
    }

    #[test]
    fn extract_time_order_ranks_rejects_missing_and_non_finite_values() {
        let with_null = DataFrame::new(
            3,
            vec![Column::new("t".into(), vec![Some(1_i64), None, Some(3)])],
        )
        .unwrap();
        assert!(extract_time_order_ranks(&with_null, "t", SRC).is_err());

        let with_nan = DataFrame::new(
            3,
            vec![Column::new("t".into(), vec![1.0_f64, f64::NAN, 3.0])],
        )
        .unwrap();
        assert!(extract_time_order_ranks(&with_nan, "t", SRC).is_err());
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

            assert!(extract_time_order_ranks(&df, "t", SRC).is_err(), "{dtype}");
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
        assert!(extract_time_order_ranks(&all_null, "t", SRC).is_err());

        for bad in [f64::INFINITY, f64::NEG_INFINITY] {
            for dtype in [DataType::Float32, DataType::Float64] {
                let mut df =
                    DataFrame::new(3, vec![Column::new("t".into(), vec![1.0_f64, bad, 3.0])])
                        .unwrap();
                let cast = df.column("t").unwrap().cast(&dtype).unwrap();
                df.with_column(cast).unwrap();

                assert!(
                    extract_time_order_ranks(&df, "t", SRC).is_err(),
                    "{dtype} {bad}"
                );
            }
        }
    }

    #[test]
    fn extract_time_order_ranks_rejects_non_orderable_dtype_and_missing_column() {
        let strings = df!("t" => ["a", "b"]).unwrap();
        assert!(extract_time_order_ranks(&strings, "t", SRC).is_err());
        assert!(extract_time_order_ranks(&strings, "absent", SRC).is_err());
    }

    #[test]
    fn extract_f64_columns_preserves_name_order() {
        let df = df!(
            "a" => [1.0, 2.0],
            "b" => [10.0, 20.0],
        )
        .unwrap();

        let columns = extract_f64_columns(&df, &["b".to_string(), "a".to_string()], SRC).unwrap();
        assert_eq!(columns, vec![vec![10.0, 20.0], vec![1.0, 2.0]]);
    }

    #[test]
    fn extract_model_x_columns_skips_const_name() {
        let df = df!("a" => [1.0, 2.0], "b" => [3.0, 4.0]).unwrap();
        let param_names = ["const", "b", "a"].map(String::from);
        let columns = extract_model_x_columns(&df, &param_names, true, SRC).unwrap();
        assert_eq!(columns, vec![vec![3.0, 4.0], vec![1.0, 2.0]]);
    }

    #[test]
    fn extract_model_x_columns_keeps_all_names_without_intercept() {
        let df = df!("a" => [1.0, 2.0]).unwrap();
        let param_names = ["a".to_string()];
        let columns = extract_model_x_columns(&df, &param_names, false, SRC).unwrap();
        assert_eq!(columns, vec![vec![1.0, 2.0]]);
    }

    #[test]
    fn extract_f64_columns_returns_empty_vec_for_empty_names() {
        let df = df!("a" => [1.0, 2.0]).unwrap();

        let columns = extract_f64_columns(&df, &[], SRC).unwrap();
        assert!(columns.is_empty());
    }

    #[test]
    fn extract_f64_columns_returns_error_when_a_column_is_missing() {
        let df = df!("a" => [1.0, 2.0]).unwrap();

        let result =
            extract_f64_columns(&df, &["a".to_string(), "does_not_exist".to_string()], SRC);
        assert!(result.is_err());
    }
}
