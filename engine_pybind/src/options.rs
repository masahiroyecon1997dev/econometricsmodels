//! OLSの推定オプション。Python側から`#[pyclass]`として構築される。
//!
//! ```python
//! from econometricsmodels import OLSOptions
//! options = OLSOptions(cov_type="HC1", confidence_level=0.90)
//! ```

use pyo3::prelude::*;

/// OLSの推定オプション。
///
/// フィールドの意味・デフォルト値の根拠は`docs/planning/specs/ols-implementation-notes.md`、
/// および対応するGitHub Issue（OLS: API・オプション設計 / OLS: 標準誤差の技術仕様確定）を参照。
#[pyclass]
#[derive(Debug, Clone)]
pub struct OLSOptions {
    /// 標準誤差の種別。"classical" | "hc0" | "hc1" | "hc2" | "hc3" | "cluster"。
    /// 大文字小文字は区別しない。HACは別途対応予定（未実装）。
    #[pyo3(get, set)]
    pub cov_type: String,

    /// 定数項（切片）をengine側で自動追加するか。
    /// trueの場合、設計行列の先頭に全要素1の列を追加する。
    /// ユーザーが`x`に自分で定数列を含めた状態でtrueにすると、
    /// 多重共線性となり`ComputationError`（特異行列）になる。
    #[pyo3(get, set)]
    pub include_intercept: bool,

    /// 信頼区間の信頼水準（0, 1)の範囲。デフォルト0.95（95%信頼区間）。
    /// 「alpha」ではなくこちらの名前を使う（0.05側との混同を避けるため）。
    #[pyo3(get, set)]
    pub confidence_level: f64,

    /// cov_type="cluster"のときに使うクラスター列名。`data`内の列名を指定する
    /// （別配列としては渡さない）。cov_type≠"cluster"のときは無視される。
    #[pyo3(get, set)]
    pub cluster_col: Option<String>,
}

#[pymethods]
impl OLSOptions {
    #[new]
    #[pyo3(signature = (
        cov_type = "classical".to_string(),
        include_intercept = true,
        confidence_level = 0.95,
        cluster_col = None,
    ))]
    fn new(
        cov_type: String,
        include_intercept: bool,
        confidence_level: f64,
        cluster_col: Option<String>,
    ) -> Self {
        Self {
            cov_type,
            include_intercept,
            confidence_level,
            cluster_col,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OLSOptions(cov_type={:?}, include_intercept={}, confidence_level={}, cluster_col={:?})",
            self.cov_type, self.include_intercept, self.confidence_level, self.cluster_col
        )
    }
}

/// 標準誤差の種別。文字列パースは`engine`側ではなくここ（境界）で行う
/// （草案コードでは`engine`側の`CovType::TryFrom<&str>`が担っていたが、
/// 「文字列を解釈する」のはPython境界の仕事とし、engineはenumだけを受け取る形にする）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovType {
    Classical,
    Hc0,
    Hc1,
    Hc2,
    Hc3,
    Cluster,
}

impl TryFrom<&str> for CovType {
    type Error = String;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "classical" | "nonrobust" => Ok(CovType::Classical),
            "hc0" => Ok(CovType::Hc0),
            "hc1" => Ok(CovType::Hc1),
            "hc2" => Ok(CovType::Hc2),
            "hc3" => Ok(CovType::Hc3),
            "cluster" => Ok(CovType::Cluster),
            other => Err(format!(
                "未知のcov_type: '{other}'。'classical', 'hc0'〜'hc3', 'cluster'のいずれかを指定してください"
            )),
        }
    }
}
