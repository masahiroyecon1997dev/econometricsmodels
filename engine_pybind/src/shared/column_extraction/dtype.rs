//! 列のdtypeの検査（数値列・キー列）。

use polars::prelude::*;

use crate::shared::column_error::{ColumnExtractionError, ColumnRole, ColumnSource};

/// f64として取り出す列の使われ方。使われ方によって許可するdtypeが異なる。
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum NumericRole {
    /// 値そのものを計算に使う列（`y`・`x`・重み・操作変数等）。
    Value,
    /// 行の並び順を決めるだけに使う列（HACの`hac_time`）。値の大小関係だけが意味を持つため、
    /// 順序が保たれる`Date`/`Datetime`も許可する。
    Ordering,
}

/// エラーメッセージ用のdtype名。Pythonの`pl.String`等と同じ呼び名にする
/// （polarsのDisplayは`str`・`cat`等の略称になるため）。内部パラメータ（時間単位・
/// 内側の型等）は含めない。
pub(super) fn dtype_label(dtype: &DataType) -> String {
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

/// `dtype`がf64に変換して使える数値の列かを検査し、そうでなければ`UnsupportedDtype`にする。
///
/// 整数・浮動小数・Boolean（`True`=1）・Decimalを許可する。文字列・日付・時刻・カテゴリ等は
/// キャストで黙って数値になったり欠損値扱いになったりして意図が判定できないため拒否する。
/// `Null`型（全値が欠損の列）は後続の欠損値チェックで「欠損値を含む」と報告させるため通す。
pub(super) fn check_numeric_dtype(
    name: &str,
    dtype: &DataType,
    role: NumericRole,
    source: ColumnSource<'_>,
) -> Result<(), ColumnExtractionError> {
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
    let column_role = match role {
        NumericRole::Value => ColumnRole::Numeric,
        NumericRole::Ordering => ColumnRole::TimeOrder,
    };
    Err(ColumnExtractionError::unsupported_dtype(
        name,
        dtype_label(dtype),
        column_role,
        source,
    ))
}

/// キー列（グループの同一性や時点を表す列）の使われ方。許可するdtypeが異なる。
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeyRole {
    /// グループの同一性だけが意味を持つ列（`effects`の先頭の列・`cluster`）。
    Identity,
    /// 時点を表す列（FE/REの`effects`の2つ目・`dk_time`）。
    Time,
}

/// `dtype`がキー列として使えるかを検査し、そうでなければ`ColumnExtractionError`にする。
///
/// 整数・浮動小数・文字列・Categorical/Enum・`Date`・タイムゾーンなしの`Datetime`は両方の
/// 役割で許可する。`Boolean`は同一性のキーだけで許可する。タイムゾーンなしの`Datetime`は
/// 文字列表現の桁数が固定のため辞書順が時系列順に一致する。タイムゾーン付きの`Datetime`は
/// 拒否する（組み込んだpolarsが文字列化できず、夏時間の重複する1時間の順序の問題もあるため）。
/// `Null`型は後続の欠損値チェックに回す。
///
/// 許可するdtypeを変えたら、`allowed`に出す一覧（`ColumnRole::allowed_dtypes`）と
/// `docs/guide/accepted-data.md`も同時に更新する（`check_numeric_dtype`も同じ）。
pub(super) fn check_key_dtype(
    name: &str,
    dtype: &DataType,
    role: KeyRole,
    source: ColumnSource<'_>,
) -> Result<(), ColumnExtractionError> {
    let column_role = match role {
        KeyRole::Identity => ColumnRole::GroupIdentifier,
        KeyRole::Time => ColumnRole::Time,
    };
    // タイムゾーン付きの`Datetime`は、この拡張に組み込んだpolarsがタイムゾーンを扱えず
    // 文字列化に失敗する。原因の分からないエラーにならないよう、対処法を示して拒否する。
    if let DataType::Datetime(_, Some(time_zone)) = dtype {
        return Err(ColumnExtractionError::time_zone_not_supported(
            name, time_zone, source,
        ));
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
    Err(ColumnExtractionError::unsupported_dtype(
        name,
        dtype_label(dtype),
        column_role,
        source,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: ColumnSource<'static> = ColumnSource::new("x", "data");

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
                assert!(check_key_dtype("k", dtype, role, SRC).is_ok(), "{dtype}");
            }
        }
        // Booleanは同一性のキーだけ。DateとDatetimeは両方の役割で許可する。
        assert!(check_key_dtype("k", &DataType::Boolean, KeyRole::Identity, SRC).is_ok());
        assert!(check_key_dtype("k", &DataType::Boolean, KeyRole::Time, SRC).is_err());
        let datetime = DataType::Datetime(TimeUnit::Microseconds, None);
        for role in [KeyRole::Identity, KeyRole::Time] {
            assert!(check_key_dtype("k", &DataType::Date, role, SRC).is_ok());
            assert!(check_key_dtype("k", &datetime, role, SRC).is_ok());
        }
    }

    #[test]
    fn check_key_dtype_rejects_datetime_with_time_zone() {
        let aware = DataType::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC));
        for role in [KeyRole::Identity, KeyRole::Time] {
            assert!(check_key_dtype("k", &aware, role, SRC).is_err());
        }
    }

    #[test]
    fn check_key_dtype_reports_the_role_of_the_rejected_dtype() {
        let aware = DataType::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC));
        for (key_role, expected) in [
            (KeyRole::Identity, ColumnRole::GroupIdentifier),
            (KeyRole::Time, ColumnRole::Time),
        ] {
            assert!(matches!(
                check_key_dtype("k", &aware, key_role, SRC),
                Err(ColumnExtractionError::TimeZoneNotSupported { .. })
            ));
            assert!(matches!(
                check_key_dtype("k", &DataType::Binary, key_role, SRC),
                Err(ColumnExtractionError::UnsupportedDtype { role, .. }) if role == expected
            ));
        }
    }

    #[test]
    fn check_numeric_dtype_reports_the_role_of_the_rejected_column() {
        for (numeric_role, expected) in [
            (NumericRole::Value, ColumnRole::Numeric),
            (NumericRole::Ordering, ColumnRole::TimeOrder),
        ] {
            assert!(matches!(
                check_numeric_dtype("x", &DataType::String, numeric_role, SRC),
                Err(ColumnExtractionError::UnsupportedDtype { role, .. }) if role == expected
            ));
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
                assert!(check_key_dtype("k", dtype, role, SRC).is_err(), "{dtype}");
            }
        }
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
                    check_numeric_dtype("x", &dtype, role, SRC).is_ok(),
                    "{dtype} should be accepted"
                );
            }
        }
    }

    #[test]
    fn check_numeric_dtype_accepts_boolean_only_for_value_role() {
        // 順序づけの列としては、値が2種類しかなく順序が定まらないため拒否する。
        assert!(check_numeric_dtype("x", &DataType::Boolean, NumericRole::Value, SRC).is_ok());
        assert!(check_numeric_dtype("x", &DataType::Boolean, NumericRole::Ordering, SRC).is_err());
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
                check_numeric_dtype("x", &dtype, NumericRole::Value, SRC).is_err(),
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
            assert!(check_numeric_dtype("t", dtype, NumericRole::Ordering, SRC).is_ok());
            assert!(check_numeric_dtype("t", dtype, NumericRole::Value, SRC).is_err());
        }
        // 時刻・期間・文字列は順序用でも拒否する。
        for dtype in [
            DataType::Time,
            DataType::Duration(TimeUnit::Microseconds),
            DataType::String,
        ] {
            assert!(check_numeric_dtype("t", &dtype, NumericRole::Ordering, SRC).is_err());
        }
    }

    /// 一覧に出す呼び名（`allowed`）の側に寄せた、dtypeの総称（整数・浮動小数は1つにまとめる）。
    fn family(dtype: &DataType) -> String {
        if dtype.is_integer() {
            "integer".to_string()
        } else if dtype.is_float() {
            "float".to_string()
        } else {
            dtype_label(dtype)
        }
    }

    fn sample_dtypes() -> Vec<DataType> {
        vec![
            DataType::Int8,
            DataType::UInt64,
            DataType::Float32,
            DataType::Float64,
            DataType::Boolean,
            DataType::Decimal(10, 2),
            DataType::String,
            DataType::Categorical(Categories::global(), Categories::global().mapping()),
            DataType::Date,
            DataType::Datetime(TimeUnit::Microseconds, None),
            DataType::Time,
            DataType::Duration(TimeUnit::Microseconds),
            DataType::Binary,
        ]
    }

    #[test]
    fn allowed_dtype_names_match_what_the_numeric_check_accepts() {
        for (numeric_role, column_role) in [
            (NumericRole::Value, ColumnRole::Numeric),
            (NumericRole::Ordering, ColumnRole::TimeOrder),
        ] {
            let accepted: std::collections::BTreeSet<String> = sample_dtypes()
                .iter()
                .filter(|d| check_numeric_dtype("c", d, numeric_role, SRC).is_ok())
                .map(family)
                .collect();
            let listed: std::collections::BTreeSet<String> = column_role
                .allowed_dtypes()
                .into_iter()
                .filter(|name| sample_dtypes().iter().any(|d| family(d) == *name))
                .map(str::to_string)
                .collect();
            assert_eq!(accepted, listed, "{column_role:?}");
        }
    }

    #[test]
    fn allowed_dtype_names_match_what_the_key_check_accepts() {
        let mut dtypes = sample_dtypes();
        let categories = FrozenCategories::new(["a"]).unwrap();
        dtypes.push(DataType::Enum(
            categories.clone(),
            categories.mapping().clone(),
        ));
        for (key_role, column_role) in [
            (KeyRole::Identity, ColumnRole::GroupIdentifier),
            (KeyRole::Time, ColumnRole::Time),
        ] {
            let accepted: std::collections::BTreeSet<String> = dtypes
                .iter()
                .filter(|d| check_key_dtype("c", d, key_role, SRC).is_ok())
                .map(family)
                .collect();
            let listed: std::collections::BTreeSet<String> = column_role
                .allowed_dtypes()
                .into_iter()
                .filter(|name| dtypes.iter().any(|d| family(d) == *name))
                .map(str::to_string)
                .collect();
            assert_eq!(accepted, listed, "{column_role:?}");
        }
    }
}
