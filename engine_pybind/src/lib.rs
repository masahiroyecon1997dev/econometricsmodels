mod data;
mod errors;
mod options;

use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use data::extract_ols_input;
use errors::{ComputationError, ValidationError};
use options::OLSOptions;

/// OLS推定のエントリポイント。
///
/// 【現状のスコープ】
/// 「パラメータの受け口」（data/y/x/optionsの検証・faer行列への変換）のみ実装済み。
/// 実際の推定計算（正規方程式ソルバー・標準誤差計算等）は別issueで実装するため、
/// ここでは受け取った内容をそのまま`todo!`にしている。
///
/// Parameters
/// ----------
/// data : polars.DataFrame
///     実データ。y・x・（指定時）cluster列を含む必要がある。
/// y : str
///     被説明変数の列名。
/// x : list[str]
///     説明変数の列名のリスト。
/// options : OLSOptions
///     推定オプション。
#[pyfunction]
fn fit_ols(
    data: PyDataFrame,
    y: String,
    x: Vec<String>,
    options: OLSOptions,
) -> PyResult<PyObject> {
    let input = extract_ols_input(data, y, x, &options)?;

    // ── ここまでが「パラメータの受け口」 ──────────────────────────────
    // input.y, input.x (faer::Mat<f64>) が engine に渡す実データ。
    // input.cluster_ids, input.cov_type, input.confidence_level, input.param_names,
    // input.dep_var_name, input.include_intercept も揃っている。
    //
    // TODO(正規方程式ソルバー実装 / 標準誤差の実装 issue):
    //   engine::OlsEstimator::new(input.y, input.x, ...)? のような形で
    //   engine側の計算に渡す。engine側の型が確定していないため、
    //   現時点ではここで打ち切る。
    let _ = input;
    Err(ComputationError::new_err(
        "engine側の計算ロジックは未実装です（パラメータの受け口のみ実装済み）",
    ))
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_ols, m)?)?;
    m.add_class::<OLSOptions>()?;
    m.add(
        "ValidationError",
        m.py().get_type::<ValidationError>(),
    )?;
    m.add(
        "ComputationError",
        m.py().get_type::<ComputationError>(),
    )?;
    Ok(())
}
