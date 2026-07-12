use thiserror::Error;

#[derive(Debug, Error)]
pub enum OlsError {
    #[error("'{column}' に NaN または無限大が含まれています")]
    InvalidInput { column: String },

    #[error("設計行列が特異または近特異です（条件数が過大）")]
    SingularMatrix,

    #[error("観測数不足: n={n} は k={k} より大きい必要があります")]
    InsufficientObservations { n: usize, k: usize },

    #[error("cov_type='cluster' のとき cluster_col が必要です")]
    MissingClusterColumn,

    #[error("統計的計算エラー: {msg}")]
    ComputationError { msg: String },
}
