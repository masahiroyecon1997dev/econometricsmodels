//! polars DataFrameから検証済みの列を取り出す、全手法共通のユーティリティ。
//!
//! 【方針】欠損値（null、およびf64列ではNaN/無限大）は常にエラーとする（自動除外はしない）。
//! 理由: `docs/spec/ols-spec.md`「API引数」を参照（GUIアプリの初心者ユーザーに、暗黙のサンプル除外という
//! 恣意的な判断をさせないため）。この方針はOLSに限らず全手法で共通。
//! polarsのnull（値が存在しない）とIEEE754のNaN（値は存在するが数値として無効）は別概念であり、
//! 両方を検出する必要がある。
//!
//! 【polarsのバージョン依存に関する注意（検証・修正済み）】
//! polars 0.54.4での実ビルドを確認済み。当初の草案から2点修正した:
//! `ChunkedArray`の`.rechunk()`が`Cow<'_, ChunkedArray<T>>`を返すようになった影響で
//! （`IntoIterator`が実装されなくなったため）、値の取り出しは`.into_iter()`ではなく
//! `.iter()`（`ChunkedArray::iter`メソッド）を使う。

use polars::prelude::*;
use pyo3::prelude::*;

use crate::errors::ValidationError;

/// `df`から`name`列をf64のVecとして取り出す。
///
/// # Errors（すべて`ValidationError`）
/// - 列が存在しない
/// - 数値型にキャストできない
/// - 欠損値（null）を含む
/// - NaN・無限大（infinity）を含む
pub fn extract_f64_column(df: &DataFrame, name: &str) -> PyResult<Vec<f64>> {
    let series = df.column(name).map_err(|_| {
        ValidationError::new_err(format!("column '{name}' does not exist in the data"))
    })?;

    let series = series.cast(&DataType::Float64).map_err(|e| {
        ValidationError::new_err(format!(
            "column '{name}' could not be cast to a numeric type (f64): {e}"
        ))
    })?;

    let ca = series
        .f64()
        .map_err(|e| ValidationError::new_err(format!("failed to convert column '{name}': {e}")))?;

    if ca.null_count() > 0 {
        return Err(ValidationError::new_err(format!(
            "column '{name}' contains {} missing value(s). Missing values are not handled \
             automatically; please impute or remove them before calling this function",
            ca.null_count()
        )));
    }

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
/// 各列につき`extract_f64_column`と同じ（列が存在しない・数値型にキャストできない・
/// 欠損値/NaN/無限大を含む場合に`ValidationError`）。最初にエラーになった列で打ち切る。
pub fn extract_f64_columns(df: &DataFrame, names: &[String]) -> PyResult<Vec<Vec<f64>>> {
    names
        .iter()
        .map(|name| extract_f64_column(df, name))
        .collect()
}

/// `df`から`name`列を、クラスターのグループキーとして文字列のVecで取り出す。
///
/// クラスター変数は整数IDとは限らない（州名・産業コード・企業ID等の文字列/
/// カテゴリカル変数であることが多い）ため、値そのものではなく「グループの
/// 同一性が判定できればよい」という前提でUtf8として扱う。
///
/// # Errors（すべて`ValidationError`）
/// - 列が存在しない
/// - 欠損値を含む
pub fn extract_group_key_column(df: &DataFrame, name: &str) -> PyResult<Vec<String>> {
    let series = df.column(name).map_err(|_| {
        ValidationError::new_err(format!("column '{name}' does not exist in the data"))
    })?;

    if series.null_count() > 0 {
        return Err(ValidationError::new_err(format!(
            "column '{name}' contains missing values"
        )));
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
