//! 列抽出（`column_extraction.rs`）で起きる入力不正のエラー型。
//!
//! 列が存在しない・欠損値や非有限値を含む・polarsの変換に失敗する、といった同じ原因のエラーが
//! 抽出関数ごとに別々の`format!`で組み立てられ、メッセージが重複・不揃いになっていたため、
//! 1バリアント1メッセージテンプレートに集約する。`engine`の`CommonError`と同じ考え方だが、
//! polarsに依存するため`engine`ではなくここに置く。
//!
//! `PyErr`への変換は`impl From<ColumnExtractionError> for PyErr`で行う。`CommonError`等は
//! `engine`の型でorphan ruleにより`impl From`を書けず関数（`common_error_to_pyerr`）に
//! しているが、この型は`engine_pybind`自身の型なので`?`で変換できる。
//! `engine`由来のエラー（`CommonError`・`PanelError`）はここに包まず、従来どおり
//! `*_error_to_pyerr`で`PyErr`にする。

use pyo3::PyErr;
use thiserror::Error;

use super::errors::ValidationError;

/// 列の抽出・検査で見つかった入力不正。すべて`ValidationError`になる。
#[derive(Debug, Error)]
pub(crate) enum ColumnExtractionError {
    /// 指定した列がデータに存在しない。
    #[error("column '{name}' does not exist in the data")]
    NotFound { name: String },

    /// polarsの型変換・取り出しに失敗した。許可したdtypeでは起こらない想定の防御的な経路。
    #[error("failed to convert column '{name}': {detail}")]
    ConversionFailed { name: String, detail: String },

    /// 欠損値（null）を含む。
    #[error(
        "column '{name}' contains {count} missing value(s). Missing values are not handled \
         automatically; please impute or remove them before calling this function"
    )]
    ContainsNull { name: String, count: usize },

    /// NaN・無限大を含む。`row`は最初に見つかった行（0始まり）。
    #[error(
        "column '{name}' contains a non-finite value ({value}) at row {row}. NaN and infinite \
         values are not handled automatically; please impute or remove them before calling \
         this function"
    )]
    NonFinite {
        name: String,
        row: usize,
        value: f64,
    },
}

impl ColumnExtractionError {
    pub(crate) fn not_found(name: &str) -> Self {
        Self::NotFound {
            name: name.to_string(),
        }
    }

    /// `detail`は変換に失敗した原因（polarsのエラー等）。
    pub(crate) fn conversion_failed(name: &str, detail: impl std::fmt::Display) -> Self {
        Self::ConversionFailed {
            name: name.to_string(),
            detail: detail.to_string(),
        }
    }

    pub(crate) fn contains_null(name: &str, count: usize) -> Self {
        Self::ContainsNull {
            name: name.to_string(),
            count,
        }
    }

    pub(crate) fn non_finite(name: &str, row: usize, value: f64) -> Self {
        Self::NonFinite {
            name: name.to_string(),
            row,
            value,
        }
    }
}

impl From<ColumnExtractionError> for PyErr {
    fn from(err: ColumnExtractionError) -> Self {
        // バリアントを足したときに、`ValidationError`と`ComputationError`のどちらかを
        // 決め忘れないよう、`_`では受けずに列挙する。
        match err {
            ColumnExtractionError::NotFound { .. }
            | ColumnExtractionError::ConversionFailed { .. }
            | ColumnExtractionError::ContainsNull { .. }
            | ColumnExtractionError::NonFinite { .. } => ValidationError::new_err(err.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `PyErr::to_string()`はPythonインタプリタを要求しpanicするため、`PyErr`に変換する前の
    // この型の`to_string()`で文言を検証する。`tests/_error_messages.py`の複製元。

    #[test]
    fn not_found_message_names_the_column() {
        assert_eq!(
            ColumnExtractionError::not_found("wage").to_string(),
            "column 'wage' does not exist in the data"
        );
    }

    #[test]
    fn conversion_failed_message_includes_the_cause() {
        assert_eq!(
            ColumnExtractionError::conversion_failed("wage", "boom").to_string(),
            "failed to convert column 'wage': boom"
        );
    }

    #[test]
    fn contains_null_message_has_count_and_remedy() {
        assert_eq!(
            ColumnExtractionError::contains_null("wage", 3).to_string(),
            "column 'wage' contains 3 missing value(s). Missing values are not handled \
             automatically; please impute or remove them before calling this function"
        );
    }

    #[test]
    fn non_finite_message_has_value_and_row() {
        assert_eq!(
            ColumnExtractionError::non_finite("wage", 4, f64::NAN).to_string(),
            "column 'wage' contains a non-finite value (NaN) at row 4. NaN and infinite \
             values are not handled automatically; please impute or remove them before \
             calling this function"
        );
        assert!(
            ColumnExtractionError::non_finite("wage", 0, f64::NEG_INFINITY)
                .to_string()
                .contains("(-inf) at row 0")
        );
    }
}
