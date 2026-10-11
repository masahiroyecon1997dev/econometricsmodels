//! `augment()`（ソースデータに予測値の列を1列付加したDataFrameを返す）の共通処理。
//!
//! OLS/WLS/Logit/Probit/Tobitの`augment()`が、列名衝突の検査→列の付加→`PyDataFrame`化を
//! 同じ手順で重複して書いていたため、ここに集約した。

use polars::prelude::{Column, DataFrame};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use super::errors::ComputationError;
use super::validation::validate_no_existing_column;

/// `source`に`values`を`column_name`という名前の新しい列として付加して返す。元の列と行の順序は
/// 保たれる。
///
/// 呼び出し側は、`values`を`source`自身から計算した値（`new_data`指定時）か、`source`と同じ
/// `fit()`呼び出しの同じ観測数に由来するペア（学習データと適合値・予測値、`new_data=None`時）
/// として渡す。どちらも長さが一致することが構築上保証されるため、長さの不一致は
/// 呼び出し側の不具合を意味し、入力の誤りではない。
///
/// # Errors
/// - `source`が既に`column_name`という列を持つ場合は`ValidationError`（黙って上書きしない）。
/// - `values.len() != source.height()`の場合は`ComputationError`（上記の通り通常は到達しない。
///   Pythonの`PanicException`にせず通常の例外として返すため`Result`にしている）。
pub fn append_predicted_column(
    mut source: DataFrame,
    column_name: &str,
    values: Vec<f64>,
) -> PyResult<PyDataFrame> {
    validate_no_existing_column(&source, column_name)?;
    if values.len() != source.height() {
        return Err(ComputationError::new_err(format!(
            "internal error: cannot append column '{column_name}' with {} values to data with \
             {} rows",
            values.len(),
            source.height()
        )));
    }
    source
        .with_column(Column::new(column_name.into(), values))
        .map_err(|err| ComputationError::new_err(err.to_string()))?;
    Ok(PyDataFrame(source))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> DataFrame {
        DataFrame::new(2, vec![Column::new("x1".into(), &[1.0, 2.0])]).unwrap()
    }

    #[test]
    fn appends_the_column_after_the_original_columns() {
        let PyDataFrame(df) =
            append_predicted_column(source(), "predicted", vec![0.5, 0.25]).unwrap();
        let names: Vec<&str> = df.get_column_names().iter().map(|n| n.as_str()).collect();
        assert_eq!(names, ["x1", "predicted"]);
        let predicted: Vec<f64> = df
            .column("predicted")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(predicted, [0.5, 0.25]);
    }

    #[test]
    fn returns_error_when_the_column_already_exists() {
        let df = DataFrame::new(
            2,
            vec![
                Column::new("x1".into(), &[1.0, 2.0]),
                Column::new("predicted".into(), &[9.0, 9.0]),
            ],
        )
        .unwrap();
        assert!(append_predicted_column(df, "predicted", vec![0.5, 0.25]).is_err());
    }

    #[test]
    fn returns_error_when_the_length_does_not_match_the_rows() {
        assert!(append_predicted_column(source(), "predicted", vec![0.5]).is_err());
        assert!(append_predicted_column(source(), "predicted", vec![0.5, 0.25, 0.1]).is_err());
    }
}
