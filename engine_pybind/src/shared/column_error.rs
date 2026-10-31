//! 列の抽出（`column_extraction.rs`）で起きる入力不正のエラー型。
//!
//! 列が存在しない・欠損値や非有限値を含む・dtypeが許可されない、といった同じ原因のエラーが
//! 抽出関数ごとに別々に組み立てられ、メッセージが重複・不揃いになっていたため、1バリアントに集約する。
//! `engine`の`CommonError`と同じ考え方だが、polarsに依存するため`engine`ではなくここに置く。
//!
//! 文面と`params`は`engine::shared::error_catalog`のカタログが決め、このバリアントは
//! [`ErrorInfo::report`]で`code`と`params`に写すだけ（`Display`もカタログの`message`）。
//! どの引数のどのデータか（`argument`・`data_argument`）は、アプリが利用者に「どこでエラーが
//! 出たか」を示せるよう、すべての抽出関数が呼び出し元から受け取る（[`ColumnSource`]）。
//!
//! `PyErr`への変換は`impl From<ColumnExtractionError> for PyErr`で行う。`engine`の型
//! （`CommonError`・`PanelError`等）はorphan ruleにより`impl From`を書けず、`to_pyerr(&err)`
//! （各系統の`*_error_to_pyerr`はこれを呼ぶだけ）で変換するが、この型は`engine_pybind`
//! 自身の型なので`?`で変換できる。`engine`由来のエラーはここに包まない。

use std::fmt;

use engine::shared::error::ErrorInfo;
use engine::shared::error_catalog::{ErrorCode, ErrorReport};
use pyo3::PyErr;

use super::errors::to_pyerr;

/// 列を読んだ引数の名前と、その列を含むデータの引数の名前。エラーに「どの引数の、どのデータの列か」を
/// 載せるために、抽出関数が呼び出し元から受け取る。`argument`は`y`・`x`・`cluster`・`hac_time`等、
/// `data_argument`は`data`・`new_data`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ColumnSource<'a> {
    pub(crate) argument: &'a str,
    pub(crate) data_argument: &'a str,
}

impl<'a> ColumnSource<'a> {
    pub(crate) const fn new(argument: &'a str, data_argument: &'a str) -> Self {
        Self {
            argument,
            data_argument,
        }
    }
}

/// エラーに残す、[`ColumnSource`]の所有版。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SourceNames {
    argument: String,
    data_argument: String,
}

impl From<ColumnSource<'_>> for SourceNames {
    fn from(source: ColumnSource<'_>) -> Self {
        Self {
            argument: source.argument.to_string(),
            data_argument: source.data_argument.to_string(),
        }
    }
}

/// 列の使われ方。dtypeの不許可で、代わりに使えるdtypeを示すために使う。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColumnRole {
    /// 値そのものを計算に使う数値の列（`y`・`x`・重み・操作変数等）。
    Numeric,
    /// 行の並び順を決めるだけに使う列（HACの`hac_time`）。
    TimeOrder,
    /// グループの同一性だけが意味を持つ列（`effects`の先頭の列・`cluster`）。
    GroupIdentifier,
    /// 時点を表す列（FE/REの`effects`の2つ目・`dk_time`）。
    Time,
}

impl ColumnRole {
    /// この使われ方で許可するdtypeの呼び名（`allowed`に入る）。許可するdtypeを変えたら、
    /// `check_numeric_dtype`・`check_key_dtype`と`docs/guide/accepted-data.md`も同時に更新する。
    pub(crate) fn allowed_dtypes(self) -> Vec<&'static str> {
        match self {
            Self::Numeric => vec!["integer", "float", "Boolean", "Decimal"],
            Self::TimeOrder => vec!["integer", "float", "Decimal", "Date", "Datetime"],
            Self::GroupIdentifier => vec![
                "integer",
                "float",
                "String",
                "Categorical",
                "Enum",
                "Boolean",
                "Date",
                "Datetime",
            ],
            Self::Time => vec![
                "integer",
                "float",
                "String",
                "Categorical",
                "Enum",
                "Date",
                "Datetime",
            ],
        }
    }
}

/// 列の抽出・検査で見つかった入力不正。`ConversionFailed`・`OrderingFailed`以外は`ValidationError`、
/// その2つは内部エラー（`ComputationError`）になる。
#[derive(Debug)]
pub(crate) enum ColumnExtractionError {
    /// 指定した列がデータに存在しない。
    NotFound { name: String, source: SourceNames },

    /// polarsの型変換・取り出しに失敗した。許可したdtypeでは起こらない想定の防御的な経路。
    ConversionFailed { name: String, detail: String },

    /// 時点の列として順序づけできなかった（整数への変換・取り出しの失敗）。
    /// 許可したdtypeでは起こらない想定の防御的な経路。
    OrderingFailed { name: String, detail: String },

    /// 欠損値（null）を含む。
    ContainsNull {
        name: String,
        count: usize,
        source: SourceNames,
    },

    /// NaN・無限大を含む。`row`は最初に見つかった行（0始まり）。
    NonFinite {
        name: String,
        row: usize,
        value: f64,
        source: SourceNames,
    },

    /// その使われ方では許可していないdtype。`dtype`はPythonの`pl.String`等と同じ呼び名。
    UnsupportedDtype {
        name: String,
        dtype: String,
        role: ColumnRole,
        source: SourceNames,
    },

    /// タイムゾーン付きの`Datetime`（キー列としては拒否する）。
    TimeZoneNotSupported {
        name: String,
        time_zone: String,
        source: SourceNames,
    },

    /// 時点の順序を決める列に同じ値の行がある。`first_row`・`second_row`は同値の2行（0始まり）。
    TiedValues {
        name: String,
        first_row: usize,
        second_row: usize,
        source: SourceNames,
    },
}

impl ColumnExtractionError {
    pub(crate) fn not_found(name: &str, source: ColumnSource<'_>) -> Self {
        Self::NotFound {
            name: name.to_string(),
            source: source.into(),
        }
    }

    /// `detail`は変換に失敗した原因（polarsのエラー等）。
    pub(crate) fn conversion_failed(name: &str, detail: impl fmt::Display) -> Self {
        Self::ConversionFailed {
            name: name.to_string(),
            detail: detail.to_string(),
        }
    }

    /// `detail`は順序づけに失敗した原因（polarsのエラー等）。
    pub(crate) fn ordering_failed(name: &str, detail: impl fmt::Display) -> Self {
        Self::OrderingFailed {
            name: name.to_string(),
            detail: detail.to_string(),
        }
    }

    pub(crate) fn contains_null(name: &str, count: usize, source: ColumnSource<'_>) -> Self {
        Self::ContainsNull {
            name: name.to_string(),
            count,
            source: source.into(),
        }
    }

    pub(crate) fn non_finite(name: &str, row: usize, value: f64, source: ColumnSource<'_>) -> Self {
        Self::NonFinite {
            name: name.to_string(),
            row,
            value,
            source: source.into(),
        }
    }

    pub(crate) fn unsupported_dtype(
        name: &str,
        dtype: String,
        role: ColumnRole,
        source: ColumnSource<'_>,
    ) -> Self {
        Self::UnsupportedDtype {
            name: name.to_string(),
            dtype,
            role,
            source: source.into(),
        }
    }

    pub(crate) fn time_zone_not_supported(
        name: &str,
        time_zone: impl fmt::Display,
        source: ColumnSource<'_>,
    ) -> Self {
        Self::TimeZoneNotSupported {
            name: name.to_string(),
            time_zone: time_zone.to_string(),
            source: source.into(),
        }
    }

    pub(crate) fn tied_values(
        name: &str,
        first_row: usize,
        second_row: usize,
        source: ColumnSource<'_>,
    ) -> Self {
        Self::TiedValues {
            name: name.to_string(),
            first_row,
            second_row,
            source: source.into(),
        }
    }
}

/// `column`・`argument`・`data_argument`を持つ`ErrorReport`の出発点。
fn located(code: ErrorCode, name: &str, source: &SourceNames) -> ErrorReport {
    ErrorReport::new(code)
        .param("column", name)
        .param("argument", source.argument.as_str())
        .param("data_argument", source.data_argument.as_str())
}

/// 内部エラー。`detail`は開発者向けの英語で、末尾のピリオドはテンプレート側が付ける。
fn internal(detail: String) -> ErrorReport {
    ErrorReport::new(ErrorCode::InternalError).param("detail", detail)
}

impl ErrorInfo for ColumnExtractionError {
    fn report(&self) -> ErrorReport {
        match self {
            Self::NotFound { name, source } => located(ErrorCode::ColumnNotFound, name, source),
            Self::ContainsNull {
                name,
                count,
                source,
            } => located(ErrorCode::ColumnContainsNull, name, source).param("count", *count),
            Self::NonFinite {
                name,
                row,
                value,
                source,
            } => located(ErrorCode::ColumnNonFinite, name, source)
                .param("row", *row)
                .param("value", *value),
            Self::UnsupportedDtype {
                name,
                dtype,
                role,
                source,
            } => located(ErrorCode::ColumnUnsupportedDtype, name, source)
                .param("dtype", dtype.as_str())
                .param("allowed", role.allowed_dtypes()),
            Self::TimeZoneNotSupported {
                name,
                time_zone,
                source,
            } => located(ErrorCode::ColumnTimezoneNotSupported, name, source)
                .param("time_zone", time_zone.as_str()),
            Self::TiedValues {
                name,
                first_row,
                second_row,
                source,
            } => located(ErrorCode::ColumnValuesTied, name, source)
                .param("first_row", *first_row)
                .param("second_row", *second_row),
            // 許可したdtypeでは起こらない想定の防御的な経路。利用者に出す文言を持たず内部エラーとする。
            Self::ConversionFailed { name, detail } => ErrorReport::new(ErrorCode::InternalError)
                .param(
                    "detail",
                    format!("failed to convert column '{name}': {detail}"),
                ),
            Self::OrderingFailed { name, detail } => internal(format!(
                "column '{name}' could not be ordered as a time column: {}",
                detail.trim_end_matches('.')
            )),
        }
    }
}

impl fmt::Display for ColumnExtractionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.report().message())
    }
}

impl std::error::Error for ColumnExtractionError {}

impl From<ColumnExtractionError> for PyErr {
    fn from(err: ColumnExtractionError) -> Self {
        to_pyerr(&err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::shared::error_catalog::ErrorClass;

    // `PyErr::to_string()`はPythonインタプリタを要求しpanicするため、`PyErr`に変換する前の
    // この型の`report()`で`code`・`params`・文面を検証する。

    const SRC: ColumnSource<'static> = ColumnSource::new("x", "data");

    fn full(err: &ColumnExtractionError) -> String {
        err.report().full_text()
    }

    #[test]
    fn not_found_names_the_column_the_argument_and_the_data() {
        let err = ColumnExtractionError::not_found("wage", SRC);
        let report = err.report();
        assert_eq!(report.code(), ErrorCode::ColumnNotFound);
        assert_eq!(
            full(&err),
            "Column 'wage' (given as x) does not exist in data. \
             Check the spelling against data.columns."
        );
        let new_data = ColumnExtractionError::not_found("wage", ColumnSource::new("x", "new_data"));
        assert_eq!(
            full(&new_data),
            "Column 'wage' (given as x) does not exist in new_data. \
             Check the spelling against new_data.columns."
        );
    }

    #[test]
    fn display_is_the_message_without_the_hint() {
        let err = ColumnExtractionError::not_found("wage", SRC);
        assert_eq!(
            err.to_string(),
            "Column 'wage' (given as x) does not exist in data."
        );
    }

    #[test]
    fn contains_null_has_count() {
        let err = ColumnExtractionError::contains_null("wage", 3, SRC);
        assert_eq!(err.report().code(), ErrorCode::ColumnContainsNull);
        assert_eq!(
            full(&err),
            "Column 'wage' (given as x) in data contains 3 missing value(s) (null), and missing \
             values are never dropped automatically. Drop or impute the missing values first, for \
             example with drop_nulls()."
        );
    }

    #[test]
    fn non_finite_has_value_and_zero_based_row() {
        let err = ColumnExtractionError::non_finite("wage", 4, f64::NAN, SRC);
        assert_eq!(err.report().code(), ErrorCode::ColumnNonFinite);
        assert_eq!(
            full(&err),
            "Column 'wage' (given as x) in data contains a non-finite value (NaN) at row 4 \
             (0-based), and non-finite values are never dropped automatically. Remove or replace \
             the NaN and infinite values first."
        );
        let negative = ColumnExtractionError::non_finite("wage", 0, f64::NEG_INFINITY, SRC);
        assert!(full(&negative).contains("(-inf) at row 0"));
    }

    #[test]
    fn unsupported_dtype_lists_the_dtypes_allowed_for_the_role() {
        let message = |role| {
            full(&ColumnExtractionError::unsupported_dtype(
                "c",
                "String".to_string(),
                role,
                SRC,
            ))
        };
        assert_eq!(
            message(ColumnRole::Numeric),
            "Column 'c' (given as x) in data has dtype String, which cannot be used for x. \
             Use a column with one of these dtypes: \"integer\", \"float\", \"Boolean\", \
             \"Decimal\", or cast the column first."
        );
        assert!(message(ColumnRole::TimeOrder).contains("\"Date\", \"Datetime\""));
        assert!(message(ColumnRole::GroupIdentifier).contains("\"Boolean\", \"Date\""));
        assert!(!message(ColumnRole::Time).contains("Boolean"));
    }

    #[test]
    fn time_zone_names_the_zone() {
        let err = ColumnExtractionError::time_zone_not_supported("t", "UTC", SRC);
        assert_eq!(err.report().code(), ErrorCode::ColumnTimezoneNotSupported);
        assert_eq!(
            full(&err),
            "Column 't' (given as x) in data is a Datetime with time zone 'UTC', which cannot be \
             used for x. Remove the time zone first, for example with \
             .dt.replace_time_zone(None) after converting to the zone you want to keep."
        );
    }

    #[test]
    fn tied_values_names_both_rows() {
        let err =
            ColumnExtractionError::tied_values("t", 1, 3, ColumnSource::new("hac_time", "data"));
        assert_eq!(err.report().code(), ErrorCode::ColumnValuesTied);
        assert_eq!(
            full(&err),
            "Column 't' (given as hac_time) in data has the same value at rows 1 and 3 (0-based), \
             so the observations cannot be put in time order. Make the values distinct, or add an \
             index column such as df.with_row_index(\"t\") if the rows are already in time order."
        );
    }

    #[test]
    fn defensive_failures_are_internal_errors() {
        let conversion = ColumnExtractionError::conversion_failed("wage", "boom");
        assert_eq!(conversion.report().code(), ErrorCode::InternalError);
        assert!(
            conversion
                .to_string()
                .contains("failed to convert column 'wage': boom")
        );
        assert_eq!(conversion.report().class(), ErrorClass::Computation);
        let ordering = ColumnExtractionError::ordering_failed("t", "boom.");
        assert_eq!(ordering.report().code(), ErrorCode::InternalError);
        // 原因の文の末尾のピリオドは、テンプレートの文末のピリオドと重ならない。
        assert!(!ordering.report().message().contains(".."));
        assert!(
            ordering
                .to_string()
                .contains("column 't' could not be ordered as a time column: boom")
        );
    }

    #[test]
    fn params_follow_the_catalog_for_every_variant() {
        let errors = [
            ColumnExtractionError::not_found("c", SRC),
            ColumnExtractionError::contains_null("c", 1, SRC),
            ColumnExtractionError::non_finite("c", 0, 1.0, SRC),
            ColumnExtractionError::unsupported_dtype("c", "String".into(), ColumnRole::Time, SRC),
            ColumnExtractionError::time_zone_not_supported("c", "UTC", SRC),
            ColumnExtractionError::tied_values("c", 0, 1, SRC),
            ColumnExtractionError::conversion_failed("c", "d"),
            ColumnExtractionError::ordering_failed("c", "d"),
        ];
        for err in errors {
            assert!(err.report().validate().is_ok(), "{err:?}");
        }
    }
}
