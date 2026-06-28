/// 標準誤差の種別
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CovType {
    NonRobust,
    HC0,
    HC1,
    HC2,
    HC3,
    Cluster,
}

impl CovType {
    pub fn as_str(&self) -> &'static str {
        match self {
            CovType::NonRobust => "nonrobust",
            CovType::HC0 => "HC0",
            CovType::HC1 => "HC1",
            CovType::HC2 => "HC2",
            CovType::HC3 => "HC3",
            CovType::Cluster => "cluster",
        }
    }
}

impl TryFrom<&str> for CovType {
    type Error = String;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "nonrobust" | "ols" => Ok(CovType::NonRobust),
            "hc0" => Ok(CovType::HC0),
            "hc1" => Ok(CovType::HC1),
            "hc2" => Ok(CovType::HC2),
            "hc3" => Ok(CovType::HC3),
            "cluster" => Ok(CovType::Cluster),
            other => Err(format!("未知の cov_type: '{other}'")),
        }
    }
}

/// OLS 推定の設定
#[derive(Debug, Clone)]
pub struct OlsConfig {
    /// 標準誤差の種別
    pub cov_type: CovType,
    /// クラスター列名ラベル（ログ・エラーメッセージ用）
    pub cluster_col: Option<String>,
    /// n > 10^5 でレバレッジ近似を使用するか
    pub leverage_approx: bool,
}

impl Default for OlsConfig {
    fn default() -> Self {
        Self {
            cov_type: CovType::NonRobust,
            cluster_col: None,
            leverage_approx: false,
        }
    }
}
