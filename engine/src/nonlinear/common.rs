//! nonlinear系統（Logit/Probit/Tobit）で共有するエラー型。
//!
//! OLSは1手法1エラー型（`OlsError`）だったが、nonlinear系統は`raise_on_non_convergence`
//! 未収束・観測数不足・`confidence_level`範囲外等、3手法でほぼ共通のバリアントが多いため、
//! `<系統>/common.rs`に共有型として定義する（`.claude/rules/rust-style.md`「ファイル・
//! ディレクトリ構成」参照）。Tobit専用のバリアント（`InvalidCensoringBounds`）も、別のenumに
//! 分離せず同じ`MleError`に含める（Logit/Probitの`fit()`はそのバリアントを構築しないだけで、
//! 型を分ける必要はない。OLSの`CovType::Hac`/`CovType::Cluster`がフィールド付きバリアントとして
//! 共存しているのと同じ考え方）。
//!
//! バリアント一覧・Python例外との対応表は`docs/planning/specs/nonlinear-implementation-notes.md`
//! 「エラー型: nonlinear系統で共有（MleError）」を参照。

use thiserror::Error;

/// Logit/Probit/Tobitの計算過程で発生しうるエラー。
///
/// `engine`はPyO3を知らないため、Python例外への変換は`engine_pybind`側で行う
/// （`.claude/rules/rust-style.md`「エラーハンドリング」参照）。
#[derive(Debug, Error, PartialEq)]
pub enum MleError {
    /// `raise_on_non_convergence=true`（既定）かつ`max_iter`回で収束しなかった。
    #[error(
        "failed to converge after {n_iter} iterations. Set raise_on_non_convergence=False \
         to receive the result anyway, or increase max_iter"
    )]
    NonConvergence { n_iter: usize },

    /// 観測数nが説明変数の数k（定数項を含む）以下。
    #[error(
        "insufficient observations: n={n} must be greater than k={k} \
         (number of independent variables, including the intercept)"
    )]
    InsufficientObservations { n: usize, k: usize },

    /// `confidence_level`が`(0, 1)`の範囲外。
    #[error("confidence_level must be in the range (0, 1): {confidence_level}")]
    InvalidConfidenceLevel { confidence_level: f64 },

    /// `max_iter`が0以下。
    #[error("max_iter must be a positive integer, got {max_iter}")]
    InvalidMaxIter { max_iter: i64 },

    /// `cov_type="cluster"`なのにクラスターのグループキーが渡されていない。
    #[error("cov_type='cluster' requires cluster identifiers to be provided")]
    MissingClusterColumn,

    /// `cov_type="cluster"`のときのクラスター数が2未満。
    #[error("cov_type='cluster' requires at least 2 clusters, got {g}")]
    InsufficientClusters { g: usize },

    /// 収束点のHessianが特異で、観測情報行列（`cov_type="classical"`/`"nonrobust"`既定）の
    /// 逆行列が計算できない。
    #[error(
        "the Hessian at convergence is singular; cannot compute the observed information matrix"
    )]
    SingularHessian,

    /// 上記以外の計算過程での失敗（分布のCDF計算等）。
    #[error("computation failed: {0}")]
    ComputationFailed(String),

    /// Tobit専用: 打ち切り境界（下限/上限）の指定が不正（下限≧上限等）。
    #[error(
        "invalid censoring bounds: lower={lower:?}, upper={upper:?} \
         (at least one bound must be set, and lower must be less than upper when both are set)"
    )]
    InvalidCensoringBounds {
        lower: Option<f64>,
        upper: Option<f64>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mle_error_messages_are_human_readable() {
        assert_eq!(
            MleError::NonConvergence { n_iter: 35 }.to_string(),
            "failed to converge after 35 iterations. Set raise_on_non_convergence=False \
             to receive the result anyway, or increase max_iter"
        );
        assert_eq!(
            MleError::InsufficientObservations { n: 2, k: 3 }.to_string(),
            "insufficient observations: n=2 must be greater than k=3 \
             (number of independent variables, including the intercept)"
        );
        assert_eq!(
            MleError::InvalidConfidenceLevel {
                confidence_level: 1.5
            }
            .to_string(),
            "confidence_level must be in the range (0, 1): 1.5"
        );
        assert_eq!(
            MleError::InvalidMaxIter { max_iter: 0 }.to_string(),
            "max_iter must be a positive integer, got 0"
        );
        assert_eq!(
            MleError::MissingClusterColumn.to_string(),
            "cov_type='cluster' requires cluster identifiers to be provided"
        );
        assert_eq!(
            MleError::InsufficientClusters { g: 1 }.to_string(),
            "cov_type='cluster' requires at least 2 clusters, got 1"
        );
        assert_eq!(
            MleError::SingularHessian.to_string(),
            "the Hessian at convergence is singular; cannot compute the observed information matrix"
        );
        assert_eq!(
            MleError::ComputationFailed("normal CDF did not converge".to_string()).to_string(),
            "computation failed: normal CDF did not converge"
        );
        assert_eq!(
            MleError::InvalidCensoringBounds {
                lower: Some(10.0),
                upper: Some(5.0),
            }
            .to_string(),
            "invalid censoring bounds: lower=Some(10.0), upper=Some(5.0) \
             (at least one bound must be set, and lower must be less than upper when both are set)"
        );
    }

    #[test]
    fn mle_error_implements_partial_eq() {
        assert_eq!(MleError::SingularHessian, MleError::SingularHessian);
        assert_ne!(
            MleError::InsufficientClusters { g: 1 },
            MleError::InsufficientClusters { g: 0 }
        );
        assert_eq!(
            MleError::NonConvergence { n_iter: 35 },
            MleError::NonConvergence { n_iter: 35 }
        );
        assert_ne!(
            MleError::NonConvergence { n_iter: 35 },
            MleError::NonConvergence { n_iter: 10 }
        );
    }
}
