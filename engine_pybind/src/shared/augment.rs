//! `augment()`（ソースデータに予測値の列を1列付加したDataFrameを返す）の共通処理。
//!
//! OLS/WLS/Logit/Probit/Tobitの`augment()`が、列名衝突の検査→列の付加→`DataFrame`化を
//! 同じ手順で重複して書いていたため、ここに集約した。

use engine::shared::error_catalog::{ErrorCode, ParamValue};
use polars::prelude::{NamedFrom, Series};
use pyo3::prelude::*;
use pyo3_polars::PySeries;

use super::errors::{internal_error, user_error};

/// `source`（Pythonの`polars.DataFrame`）に`values`を`column_name`という名前の新しい列として
/// 付加した、新しい`polars.DataFrame`を返す。元の列と行の順序は保たれる。
///
/// `source`はユーザーが渡した元のデータで、推定に使わない列（取り込めないdtypeの列を含む）も
/// そのまま残る。呼び出し側は、`values`を`source`自身から計算した値（`new_data`指定時）か、
/// `source`と同じ`fit()`呼び出しの同じ観測数に由来するペア（学習データと適合値・予測値、
/// `new_data=None`時）として渡す。どちらも長さが一致することが構築上保証されるため、長さの
/// 不一致は呼び出し側の不具合を意味し、入力の誤りではない。
///
/// # Errors
/// - `source`が既に`column_name`という列を持つ場合は`ValidationError`（黙って上書きしない）。
/// - `values.len() != source.height`の場合は`ComputationError`（上記の通り通常は到達しない。
///   Pythonの`PanicException`にせず通常の例外として返すため`Result`にしている）。
pub fn append_predicted_column(
    source: &Bound<'_, PyAny>,
    column_name: &str,
    values: Vec<f64>,
) -> PyResult<Py<PyAny>> {
    let columns: Vec<String> = source.getattr("columns")?.extract()?;
    if columns.iter().any(|name| name == column_name) {
        return Err(user_error(
            ErrorCode::AugmentColumnExists,
            vec![("column", ParamValue::from(column_name))],
        ));
    }
    let height: usize = source.getattr("height")?.extract()?;
    if values.len() != height {
        return Err(internal_error(format!(
            "cannot append column '{column_name}' with {} values to data with {height} rows",
            values.len(),
        )));
    }
    let series = PySeries(Series::new(column_name.into(), values));
    Ok(source.call_method1("with_columns", (series,))?.unbind())
}
