pub mod config;
pub mod error;
pub mod results;
pub(crate) mod input;
mod compute;

use ndarray::{Array1, Array2};

pub use config::{CovType, OlsConfig};
pub use error::OlsError;
pub use results::OlsResults;

/// OLS 推定量
///
/// `new` で入力を検証してから `fit` で推定を実行する。
pub struct OlsEstimator {
    /// 被説明変数 (n,)
    pub y: Array1<f64>,
    /// 設計行列（定数項追加済み）(n, k)
    pub x: Array2<f64>,
    /// クラスター帰属 ID（cov_type = Cluster のとき使用）(n,)
    pub cluster_ids: Option<Array1<i64>>,
    /// 係数名（例: ["const", "educ", "exper"]）
    pub param_names: Vec<String>,
    /// 被説明変数名
    pub dep_var_name: String,
    /// 推定設定
    pub config: OlsConfig,
}

impl OlsEstimator {
    /// 入力を検証して `OlsEstimator` を構築する。
    ///
    /// # Errors
    /// - `OlsError::InvalidInput`: NaN / 無限大の検出、次元不一致
    /// - `OlsError::InsufficientObservations`: n <= k
    /// - `OlsError::MissingClusterColumn`: cov_type=Cluster かつ cluster_ids=None
    pub fn new(
        y: Array1<f64>,
        x: Array2<f64>,
        cluster_ids: Option<Array1<i64>>,
        param_names: Vec<String>,
        dep_var_name: String,
        config: OlsConfig,
    ) -> Result<Self, OlsError> {
        input::validate_inputs(y.view(), x.view())?;

        if config.cov_type == CovType::Cluster && cluster_ids.is_none() {
            return Err(OlsError::MissingClusterColumn);
        }

        Ok(Self {
            y,
            x,
            cluster_ids,
            param_names,
            dep_var_name,
            config,
        })
    }

    // fit() は compute モジュールの impl OlsEstimator で定義する
}
