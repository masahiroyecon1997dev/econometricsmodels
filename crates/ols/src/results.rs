use ndarray::{Array1, Array2};

/// OLS 推定結果
#[derive(Debug)]
pub struct OlsResults {
    /// 回帰係数 β̂ (k,)
    pub params: Array1<f64>,
    /// 標準誤差 (k,)
    pub std_errors: Array1<f64>,
    /// t 統計量 (k,)
    pub t_stats: Array1<f64>,
    /// p 値 (k,)
    pub p_values: Array1<f64>,
    /// 信頼区間 (k, 2): 各行 [lower, upper]（α=0.05）
    pub conf_int: Array2<f64>,
    /// 残差 ε̂ (n,)
    pub residuals: Array1<f64>,
    /// 当てはめ値 Xβ̂ (n,)
    pub fitted_values: Array1<f64>,
    /// R²
    pub r_squared: f64,
    /// 自由度修正済み R²
    pub r_squared_adj: f64,
    /// F 統計量
    pub f_statistic: f64,
    /// F 統計量の p 値
    pub f_p_value: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// 対数尤度
    pub log_likelihood: f64,
    /// 残差分散 σ̂²
    pub sigma2: f64,
    /// 観測数 n
    pub nobs: usize,
    /// 残差自由度 n - k
    pub df_resid: usize,
    /// モデル自由度 k - 1（定数項を除く説明変数の数）
    pub df_model: usize,
    /// 係数の共分散行列 (k, k)
    pub cov_params: Array2<f64>,
    /// 係数名（例: ["const", "x1", "x2"]）
    pub param_names: Vec<String>,
    /// 被説明変数名
    pub dep_var_name: String,
    /// 標準誤差種別（人間が読める形式）
    pub cov_type: String,
}
