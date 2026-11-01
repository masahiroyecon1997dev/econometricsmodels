//! polars DataFrameから検証済みの列を取り出す、全手法共通のユーティリティ。
//!
//! 【方針】欠損値（null、およびf64列ではNaN/無限大）は常にエラーとする（自動除外はしない）。
//! 理由: `docs/guide/validation.md`を参照（暗黙のサンプル除外という恣意的な判断を避け、サンプル
//! セレクションバイアスに気づかないまま推定されるのを防ぐため、除外・補完の判断はユーザーに明示的に
//! させる）。この方針はOLSに限らず全手法で共通。
//! polarsのnull（値が存在しない）とIEEE754のNaN（値は存在するが数値として無効）は別概念であり、
//! 両方を検出する必要がある。
//!
//! 【polarsのバージョン依存に関する注意】
//! `ChunkedArray`の`.rechunk()`は`Cow<'_, ChunkedArray<T>>`を返し、`IntoIterator`が
//! 実装されないため、値の取り出しは`.into_iter()`ではなく`.iter()`
//! （`ChunkedArray::iter`メソッド）を使う（polars 0.55.2で確認）。

mod dtype;
mod frame;
mod keys;
mod numeric;

pub(crate) use dtype::KeyRole;
pub(crate) use frame::type_name_of;
pub use frame::{
    SelectedData, check_dataframe, extract_column_list, extract_column_name, select_columns,
};
#[cfg(test)]
pub(crate) use keys::extract_key_column;
pub(crate) use keys::{
    FactorizedKey, checked_key_column, factorize_checked_key, with_group_labels,
};
pub use keys::{extract_entity_groups, extract_optional_cluster_groups};
pub use numeric::{
    extract_f64_column, extract_f64_columns, extract_model_x_columns, extract_time_order_ranks,
    select_model_x_columns, x_column_names,
};
pub(crate) use numeric::{physical_i128, time_order_error};
