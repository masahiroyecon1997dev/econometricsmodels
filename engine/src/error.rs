//! 系統をまたいで共有するバリデーションエラー型。
//!
//! `engine::linear::common::LeastSquaresError`と`engine::nonlinear::common::MleError`で、
//! `DimensionMismatch`/`InsufficientObservations`/`InvalidConfidenceLevel`/
//! `MissingClusterColumn`/`InsufficientClusters`/`ComputationFailed`の6バリアントが
//! エラーメッセージの文言まで完全に重複していた。IV/panel/causal/io/
//! time_series等、今後7系統・20〜30手法に増える前提のため、系統をまたぐ共通バリデーション
//! エラーをここに集約し、各系統のエラー型はthiserrorの
//! `#[error(transparent)] Common(#[from] CommonError)`で包む設計にする。
//!
//! 各系統固有の追加バリアント（例: OLSの`WeightDimensionMismatch`、nonlinearの
//! `SingularHessian`）はこの型に含めず、従来通り各系統のエラー型に直接定義する。
//! 将来ある系統で「意味は同じだが追加フィールドが要る」ケースが出てきた場合も、
//! `CommonError`を拡張せずその系統独自のバリアントとして追加すればよい
//! （`CommonError`を使うかどうかは系統ごとに選べる。無理に全バリアントを
//! ここへ集約する設計にはしない）。
use thiserror::Error;

/// 系統をまたいで共通のバリデーション・計算エラー。
///
/// `engine`はPyO3を知らないため、Python例外への変換は`engine_pybind`側で行う
/// （`.claude/rules/rust-style.md`「エラーハンドリング」参照）。`engine_pybind`側の
/// 変換は`engine_pybind::errors::common_error_to_pyerr`に集約し、各系統の
/// `*_error_to_pyerr`関数はこれに委譲する。
#[derive(Debug, Error, PartialEq)]
pub enum CommonError {
    /// yとxの行数が一致しない。
    #[error("dimension mismatch: y has {y_rows} rows but x has {x_rows} rows")]
    DimensionMismatch { y_rows: usize, x_rows: usize },

    /// 観測数nが説明変数の数k（定数項を含む）以下。`k>=1`であることが前提
    /// （`k=0`は別バリアント`NoRegressors`で扱う。`.claude/rules/rust-style.md`
    /// 「エラーハンドリング」参照）。
    #[error(
        "insufficient observations: n={n} must be greater than k={k} \
         (number of independent variables, including the intercept)"
    )]
    InsufficientObservations { n: usize, k: usize },

    /// 定数項も説明変数も無い（`include_intercept=false`かつ説明変数0個、`k=0`）。
    /// `InsufficientObservations`（`n<=k`）とは原因が異なる別の不正のため区別する
    /// （`k=0`だと`n`の値に関わらず不等式`n>k`は常に成立してしまい、
    /// `InsufficientObservations`のメッセージを流用すると「条件を満たしているのに
    /// エラーになる」という誤解を招くメッセージになっていた）。
    #[error(
        "no regressors: k=0 (include_intercept=false and no independent variables). \
         At least one of the two is required"
    )]
    NoRegressors { n: usize },

    /// `confidence_level`が`(0, 1)`の範囲外。
    #[error("confidence_level must be in the range (0, 1): {confidence_level}")]
    InvalidConfidenceLevel { confidence_level: f64 },

    /// `cov_type="cluster"`なのにクラスターのグループキーが渡されていない。
    #[error("cov_type='cluster' requires cluster identifiers to be provided")]
    MissingClusterColumn,

    /// `cov_type="cluster"`のときのクラスター数が2未満。
    #[error("cov_type='cluster' requires at least 2 clusters, got {g}")]
    InsufficientClusters { g: usize },

    /// 上記以外の計算過程での失敗（分布のCDF計算等）。
    #[error("computation failed: {0}")]
    ComputationFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_error_messages_are_human_readable() {
        assert_eq!(
            CommonError::DimensionMismatch {
                y_rows: 10,
                x_rows: 8
            }
            .to_string(),
            "dimension mismatch: y has 10 rows but x has 8 rows"
        );
        assert_eq!(
            CommonError::InsufficientObservations { n: 2, k: 3 }.to_string(),
            "insufficient observations: n=2 must be greater than k=3 \
             (number of independent variables, including the intercept)"
        );
        assert_eq!(
            CommonError::NoRegressors { n: 5 }.to_string(),
            "no regressors: k=0 (include_intercept=false and no independent variables). \
             At least one of the two is required"
        );
        assert_eq!(
            CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5
            }
            .to_string(),
            "confidence_level must be in the range (0, 1): 1.5"
        );
        assert_eq!(
            CommonError::MissingClusterColumn.to_string(),
            "cov_type='cluster' requires cluster identifiers to be provided"
        );
        assert_eq!(
            CommonError::InsufficientClusters { g: 1 }.to_string(),
            "cov_type='cluster' requires at least 2 clusters, got 1"
        );
        assert_eq!(
            CommonError::ComputationFailed("t-distribution CDF did not converge".to_string())
                .to_string(),
            "computation failed: t-distribution CDF did not converge"
        );
    }

    #[test]
    fn common_error_implements_partial_eq() {
        assert_eq!(
            CommonError::DimensionMismatch {
                y_rows: 10,
                x_rows: 10
            },
            CommonError::DimensionMismatch {
                y_rows: 10,
                x_rows: 10
            }
        );
        assert_ne!(
            CommonError::DimensionMismatch {
                y_rows: 10,
                x_rows: 10
            },
            CommonError::DimensionMismatch {
                y_rows: 10,
                x_rows: 8
            }
        );
        assert_eq!(
            CommonError::InsufficientClusters { g: 1 },
            CommonError::InsufficientClusters { g: 1 }
        );
        assert_ne!(
            CommonError::InsufficientClusters { g: 1 },
            CommonError::InsufficientClusters { g: 0 }
        );
    }
}
