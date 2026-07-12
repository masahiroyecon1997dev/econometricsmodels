use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use ols::{CovType, OlsConfig, OlsEstimator, OlsError, OlsResults};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::collections::HashMap;

// ---- エラー変換 --------------------------------------------------------

fn ols_err_to_py(e: OlsError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ---- fit_ols エントリポイント ------------------------------------------

/// Python 側から呼び出す OLS 推定のエントリポイント。
///
/// 入力のバリデーションと `OlsEstimator` の構築を行い、
/// `OlsEstimator::fit()` に計算を委譲する。
///
/// Parameters
/// ----------
/// y : np.ndarray, shape (n,), dtype float64
/// x : np.ndarray, shape (n, k), dtype float64 — 定数項は呼び出し元で追加済み
/// param_names : list[str]
/// dep_var_name : str
/// cov_type : str, default "nonrobust"
/// cluster_ids : np.ndarray | None, shape (n,), dtype int64
#[pyfunction]
#[pyo3(signature = (y, x, param_names, dep_var_name, cov_type="nonrobust", cluster_ids=None))]
fn fit_ols(
    _py: Python<'_>,
    y: PyReadonlyArray1<f64>,
    x: PyReadonlyArray2<f64>,
    param_names: Vec<String>,
    dep_var_name: String,
    cov_type: &str,
    cluster_ids: Option<PyReadonlyArray1<i64>>,
) -> PyResult<PyOlsResults> {
    let cov_type_enum = CovType::try_from(cov_type).map_err(PyValueError::new_err)?;

    let config = OlsConfig {
        cov_type: cov_type_enum,
        cluster_col: None,
        leverage_approx: false,
    };

    // NOTE: numpy → ndarray は 1 コピー発生。行列演算のため不可避。
    let y_arr: Array1<f64> = y.as_array().to_owned();
    let x_arr: Array2<f64> = x.as_array().to_owned();
    let ids: Option<Array1<i64>> = cluster_ids.map(|c| c.as_array().to_owned());

    let estimator =
        OlsEstimator::new(y_arr, x_arr, ids, param_names, dep_var_name, config)
            .map_err(ols_err_to_py)?;

    let results = estimator.fit().map_err(ols_err_to_py)?;

    Ok(PyOlsResults { inner: results })
}

// ---- PyOlsResults -------------------------------------------------------

/// OLS 推定結果を Python に公開するクラス。
///
/// `OlsResults`（Rust）をラップし、Python からアクセスできる属性・メソッドを提供する。
/// 数値配列は numpy array として、係数テーブルは dict[str, float] として返す。
/// Polars への変換は Python ラッパー（`econometricsmodels/__init__.py`）が担う。
#[pyclass(name = "OlsResults")]
pub struct PyOlsResults {
    inner: OlsResults,
}

#[pymethods]
impl PyOlsResults {
    // ---- 係数 -------------------------------------------------------------

    /// 回帰係数 dict[str, float]
    #[getter]
    fn params(&self) -> HashMap<String, f64> {
        zip_to_map(&self.inner.param_names, self.inner.params.as_slice().unwrap())
    }

    /// 標準誤差 dict[str, float]
    #[getter]
    fn std_errors(&self) -> HashMap<String, f64> {
        zip_to_map(
            &self.inner.param_names,
            self.inner.std_errors.as_slice().unwrap(),
        )
    }

    /// t 統計量 dict[str, float]
    #[getter]
    fn t_stats(&self) -> HashMap<String, f64> {
        zip_to_map(
            &self.inner.param_names,
            self.inner.t_stats.as_slice().unwrap(),
        )
    }

    /// p 値 dict[str, float]
    #[getter]
    fn p_values(&self) -> HashMap<String, f64> {
        zip_to_map(
            &self.inner.param_names,
            self.inner.p_values.as_slice().unwrap(),
        )
    }

    /// 信頼区間 dict[str, tuple[float, float]]
    ///
    /// alpha パラメータは将来の実装のためのシグネチャ。
    /// 現在は推定時に計算済みの conf_int（α=0.05）を返す。
    #[pyo3(signature = (alpha = 0.05))]
    fn conf_int(&self, alpha: f64) -> HashMap<String, (f64, f64)> {
        let _ = alpha; // TODO: alpha から t 分位点を計算して CI を再計算する
        self.inner
            .param_names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let lower = self.inner.conf_int[[i, 0]];
                let upper = self.inner.conf_int[[i, 1]];
                (name.clone(), (lower, upper))
            })
            .collect()
    }

    // ---- 残差・当てはめ値 ------------------------------------------------

    /// 残差 ε̂ を numpy array で返す。
    #[getter]
    fn residuals(&self, py: Python<'_>) -> Py<PyAny> {
        self.inner
            .residuals
            .clone()
            .into_pyarray(py)
            .into_any()
            .unbind()
    }

    /// 当てはめ値 Xβ̂ を numpy array で返す。
    #[getter]
    fn fitted_values(&self, py: Python<'_>) -> Py<PyAny> {
        self.inner
            .fitted_values
            .clone()
            .into_pyarray(py)
            .into_any()
            .unbind()
    }

    // ---- 適合度統計量 ----------------------------------------------------

    #[getter]
    fn nobs(&self) -> usize { self.inner.nobs }
    #[getter]
    fn df_resid(&self) -> usize { self.inner.df_resid }
    #[getter]
    fn df_model(&self) -> usize { self.inner.df_model }
    #[getter]
    fn r_squared(&self) -> f64 { self.inner.r_squared }
    #[getter]
    fn r_squared_adj(&self) -> f64 { self.inner.r_squared_adj }
    #[getter]
    fn f_statistic(&self) -> f64 { self.inner.f_statistic }
    #[getter]
    fn f_p_value(&self) -> f64 { self.inner.f_p_value }
    #[getter]
    fn aic(&self) -> f64 { self.inner.aic }
    #[getter]
    fn bic(&self) -> f64 { self.inner.bic }
    #[getter]
    fn log_likelihood(&self) -> f64 { self.inner.log_likelihood }
    #[getter]
    fn sigma2(&self) -> f64 { self.inner.sigma2 }

    // ---- メタ情報 --------------------------------------------------------

    /// 係数名リスト（例: ["const", "educ", "exper"]）
    #[getter]
    fn param_names(&self) -> Vec<String> {
        self.inner.param_names.clone()
    }

    /// 被説明変数名
    #[getter]
    fn dep_var_name(&self) -> &str {
        &self.inner.dep_var_name
    }

    /// 標準誤差種別文字列（例: "HC1"）
    #[getter]
    fn cov_type_str(&self) -> &str {
        &self.inner.cov_type
    }

    // ---- 予測 ------------------------------------------------------------

    /// 新しい設計行列（numpy, shape (m, k)）で予測値を返す。
    ///
    /// Polars → numpy の変換は Python 側ラッパーが行う。
    fn predict_array(&self, py: Python<'_>, x_new: PyReadonlyArray2<f64>) -> Py<PyAny> {
        let x = x_new.as_array();
        x.dot(&self.inner.params)
            .into_pyarray(py)
            .into_any()
            .unbind()
    }

    // ---- テキスト出力 ----------------------------------------------------

    /// statsmodels 風のサマリー文字列を返す。
    fn summary(&self) -> String {
        let sep = "=".repeat(62);
        let inner_sep = "-".repeat(62);
        let r = &self.inner;

        let mut out = format!(
            "OLS Regression Results\n\
             {sep}\n\
             Dep. Variable: {dv:<20} R-squared:  {r2:.3}\n\
             Model:                   OLS   Adj. R-sq:  {r2a:.3}\n\
             No. Observations: {nobs:<14} F-statistic: {fstat:.2}\n\
             Df Residuals:  {dfr:<20} Prob(F):    {fp:.2e}\n\
             Cov Type:      {cov:<20} AIC:        {aic:.1}\n\
             {inner_sep}\n\
             {name:>16}  {coef:>10}  {se:>8}  {t:>8}  {p:>7}  {ci_lo:>8}  {ci_hi:>8}\n\
             {inner_sep}",
            sep = sep,
            dv = r.dep_var_name,
            r2 = r.r_squared,
            r2a = r.r_squared_adj,
            nobs = r.nobs,
            fstat = r.f_statistic,
            dfr = r.df_resid,
            fp = r.f_p_value,
            cov = r.cov_type,
            aic = r.aic,
            inner_sep = inner_sep,
            name = "",
            coef = "coef",
            se = "std err",
            t = "t",
            p = "P>|t|",
            ci_lo = "[0.025",
            ci_hi = "0.975]",
        );

        for (i, name) in r.param_names.iter().enumerate() {
            out.push_str(&format!(
                "\n{name:>16}  {coef:>10.4}  {se:>8.4}  {t:>8.3}  {p:>7.4}  {lo:>8.4}  {hi:>8.4}",
                name = name,
                coef = r.params[i],
                se = r.std_errors[i],
                t = r.t_stats[i],
                p = r.p_values[i],
                lo = r.conf_int[[i, 0]],
                hi = r.conf_int[[i, 1]],
            ));
        }

        out.push_str(&format!("\n{sep}"));
        out
    }
}

// ---- ヘルパー -----------------------------------------------------------

fn zip_to_map(names: &[String], values: &[f64]) -> HashMap<String, f64> {
    names
        .iter()
        .zip(values.iter())
        .map(|(k, &v)| (k.clone(), v))
        .collect()
}

// ---- モジュール登録 ------------------------------------------------------

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_ols, m)?)?;
    m.add_class::<PyOlsResults>()?;
    Ok(())
}
