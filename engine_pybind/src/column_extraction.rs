//! polars DataFrameから検証済みの列を取り出す、全手法共通のユーティリティ。
//!
//! 【方針】欠損値（null、およびf64列ではNaN/無限大）は常にエラーとする（自動除外はしない）。
//! 理由: `docs/planning/specs/ols-implementation-notes.md`
//! 「欠損値の扱い」を参照（GUIアプリの初心者ユーザーに、暗黙のサンプル除外という
//! 恣意的な判断をさせないため）。この方針はOLSに限らず全手法で共通。
//! polarsのnull（値が存在しない）とIEEE754のNaN（値は存在するが数値として無効）は別概念であり、
//! 両方を検出する必要がある。
//!
//! 【polarsのバージョン依存に関する注意】
//! この環境ではpolarsを実際にビルドして検証できていない。`DataFrame::column`の
//! 戻り値型（`&Series`か`&Column`か）等、pinするpolarsのバージョンに応じて
//! 微調整が必要な可能性がある。

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
    let series = df
        .column(name)
        .map_err(|_| ValidationError::new_err(format!("column '{name}' does not exist in the data")))?;

    let series = series.cast(&DataType::Float64).map_err(|e| {
        ValidationError::new_err(format!("column '{name}' could not be cast to a numeric type (f64): {e}"))
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
            ca.into_iter()
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
    let series = df
        .column(name)
        .map_err(|_| ValidationError::new_err(format!("column '{name}' does not exist in the data")))?;

    if series.null_count() > 0 {
        return Err(ValidationError::new_err(format!(
            "column '{name}' contains missing values"
        )));
    }

    // Utf8にキャストして文字列表現で比較する（元の型が数値・カテゴリカルでもよい）。
    let series = series.cast(&DataType::String).map_err(|e| {
        ValidationError::new_err(format!("column '{name}' could not be interpreted as a group key: {e}"))
    })?;
    let ca = series.str().map_err(|e| {
        ValidationError::new_err(format!("failed to convert column '{name}': {e}"))
    })?;

    Ok(ca
        .into_iter()
        .map(|v| v.expect("null_countチェック済み").to_string())
        .collect())
}
