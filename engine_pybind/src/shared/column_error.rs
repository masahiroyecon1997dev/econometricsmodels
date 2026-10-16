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

/// 列の使われ方。dtype不許可・タイムゾーン拒否のメッセージで、使い道と代わりに使える
/// dtypeを示すために使う。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColumnRole {
    /// 値そのものを計算に使う数値の列（`y`・`x`・重み・操作変数等）。
    Numeric,
    /// 行の並び順を決めるだけに使う列（HACの`hac_time`）。
    TimeOrder,
    /// グループの同一性だけが意味を持つ列（`entity`・`cluster`）。
    GroupIdentifier,
    /// 時点を表す列（FE/REの`time`・`dk_time`）。
    Time,
}

impl ColumnRole {
    /// 「cannot be used as a ... column」に入る、使い道の呼び名。
    fn label(self) -> &'static str {
        match self {
            Self::Numeric => "numeric",
            Self::TimeOrder => "time-order",
            Self::GroupIdentifier => "group identifier",
            Self::Time => "time",
        }
    }

    /// dtypeを拒否したときに、代わりに使えるdtypeを示す文。
    fn accepted_dtypes_hint(self) -> &'static str {
        match self {
            Self::Numeric => {
                "use an integer, float, boolean or decimal column (cast it first if it holds numbers)"
            }
            Self::TimeOrder => "use an integer, float, Date or Datetime column",
            Self::GroupIdentifier => {
                "use an integer, float, string, categorical, boolean, Date or Datetime column"
            }
            Self::Time => "use an integer, float, string, categorical, Date or Datetime column",
        }
    }
}

/// 列の抽出・検査で見つかった入力不正。すべて`ValidationError`になる。
#[derive(Debug, Error)]
pub(crate) enum ColumnExtractionError {
    /// 指定した列がデータに存在しない。
    #[error("column '{name}' does not exist in the data")]
    NotFound { name: String },

    /// polarsの型変換・取り出しに失敗した。許可したdtypeでは起こらない想定の防御的な経路。
    #[error("failed to convert column '{name}': {detail}")]
    ConversionFailed { name: String, detail: String },

    /// 時点の列として順序づけできなかった（整数への変換・取り出しの失敗）。
    /// 許可したdtypeでは起こらない想定の防御的な経路。
    #[error("column '{name}' could not be ordered as a time column: {detail}")]
    OrderingFailed { name: String, detail: String },

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

    /// その使われ方では許可していないdtype。`dtype`はPythonの`pl.String`等と同じ呼び名。
    #[error(
        "column '{name}' has dtype {dtype}, which cannot be used as a {} column; {}",
        .role.label(),
        .role.accepted_dtypes_hint()
    )]
    UnsupportedDtype {
        name: String,
        dtype: String,
        role: ColumnRole,
    },

    /// タイムゾーン付きの`Datetime`（キー列としては拒否する）。
    #[error(
        "column '{name}' is a Datetime with time zone '{time_zone}', which cannot be used as a \
         {} column; remove the time zone first, for example with \
         .dt.replace_time_zone(None) after converting to the zone you want to keep",
        .role.label()
    )]
    TimeZoneNotSupported {
        name: String,
        time_zone: String,
        role: ColumnRole,
    },

    /// 時点の順序を決める列に同じ値の行がある。`first_row`・`second_row`は同値の2行（0始まり）。
    #[error(
        "column '{name}' has the same value at rows {first_row} and {second_row}. A time-order \
         column must give every observation a distinct value, because tied observations \
         cannot be put in time order; make the values distinct, or omit the time-order \
         option to use the row order of the data"
    )]
    TiedValues {
        name: String,
        first_row: usize,
        second_row: usize,
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

    /// `detail`は順序づけに失敗した原因（polarsのエラー等）。
    pub(crate) fn ordering_failed(name: &str, detail: impl std::fmt::Display) -> Self {
        Self::OrderingFailed {
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

    pub(crate) fn unsupported_dtype(name: &str, dtype: String, role: ColumnRole) -> Self {
        Self::UnsupportedDtype {
            name: name.to_string(),
            dtype,
            role,
        }
    }

    pub(crate) fn time_zone_not_supported(
        name: &str,
        time_zone: impl std::fmt::Display,
        role: ColumnRole,
    ) -> Self {
        Self::TimeZoneNotSupported {
            name: name.to_string(),
            time_zone: time_zone.to_string(),
            role,
        }
    }

    pub(crate) fn tied_values(name: &str, first_row: usize, second_row: usize) -> Self {
        Self::TiedValues {
            name: name.to_string(),
            first_row,
            second_row,
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
            | ColumnExtractionError::OrderingFailed { .. }
            | ColumnExtractionError::ContainsNull { .. }
            | ColumnExtractionError::NonFinite { .. }
            | ColumnExtractionError::UnsupportedDtype { .. }
            | ColumnExtractionError::TimeZoneNotSupported { .. }
            | ColumnExtractionError::TiedValues { .. } => ValidationError::new_err(err.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `PyErr::to_string()`はPythonインタプリタを要求しpanicするため、`PyErr`に変換する前の
    // この型の`to_string()`で文言を検証する。`tests/support/_error_messages.py`の複製元。

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
    fn ordering_failed_message_includes_the_cause() {
        assert_eq!(
            ColumnExtractionError::ordering_failed("t", "boom").to_string(),
            "column 't' could not be ordered as a time column: boom"
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

    #[test]
    fn unsupported_dtype_message_depends_on_the_role() {
        let message = |role| {
            ColumnExtractionError::unsupported_dtype("c", "String".to_string(), role).to_string()
        };
        assert_eq!(
            message(ColumnRole::Numeric),
            "column 'c' has dtype String, which cannot be used as a numeric column; \
             use an integer, float, boolean or decimal column (cast it first if it holds numbers)"
        );
        assert_eq!(
            message(ColumnRole::TimeOrder),
            "column 'c' has dtype String, which cannot be used as a time-order column; \
             use an integer, float, Date or Datetime column"
        );
        assert_eq!(
            message(ColumnRole::GroupIdentifier),
            "column 'c' has dtype String, which cannot be used as a group identifier column; \
             use an integer, float, string, categorical, boolean, Date or Datetime column"
        );
        assert_eq!(
            message(ColumnRole::Time),
            "column 'c' has dtype String, which cannot be used as a time column; \
             use an integer, float, string, categorical, Date or Datetime column"
        );
    }

    #[test]
    fn time_zone_message_names_the_zone_and_the_remedy() {
        assert_eq!(
            ColumnExtractionError::time_zone_not_supported("t", "UTC", ColumnRole::Time)
                .to_string(),
            "column 't' is a Datetime with time zone 'UTC', which cannot be used as a time \
             column; remove the time zone first, for example with .dt.replace_time_zone(None) \
             after converting to the zone you want to keep"
        );
    }

    #[test]
    fn tied_values_message_names_both_rows() {
        assert_eq!(
            ColumnExtractionError::tied_values("t", 1, 3).to_string(),
            "column 't' has the same value at rows 1 and 3. A time-order column must give \
             every observation a distinct value, because tied observations cannot be put in \
             time order; make the values distinct, or omit the time-order option to use the \
             row order of the data"
        );
    }
}
