//! `linear`系統（OLS/WLS、将来のGLS・区分回帰）で共有するエラー型。
//!
//! OLS/WLS/GLS/区分回帰はいずれも最小二乗法ベースの推定のため、系統名`linear`ではなく
//! 推定方式名`LeastSquares`で命名する（nonlinear系統の`MleError`が「nonlinear」ではなく
//! 推定方式名「MLE」で命名されているのと同じ考え方。`.claude/rules/rust-style.md`
//! 「ファイル・ディレクトリ構成」参照）。
//!
//! 元々`OlsError`という名前でOLS単体のエラー型として`linear/ols`（当時は1ファイル）に定義されていたが、
//! WLSが同じ型をそのまま再利用する設計（`OlsInput::from_columns_weighted`・
//! `OlsEstimator::fit`を無変更で流用する、`docs/spec/wls-spec.md`「sqrt(w)変換」）に
//! なり、WLS固有のバリアント（`WeightDimensionMismatch`/`NonPositiveWeight`）も混在する
//! ことになった。実態（OLS・WLS共有）に合わせて`common.rs`に切り出し、`LeastSquaresError`に
//! 改名した。
//!
//! `DimensionMismatch`/`InsufficientObservations`/`InvalidConfidenceLevel`/
//! `MissingClusterColumn`/`InsufficientClusters`/`ComputationFailed`は、nonlinear系統の
//! `MleError`と文言まで完全に重複していたため`engine::error::CommonError`に切り出し、
//! `Common`バリアント経由で保持する。

use thiserror::Error;

use crate::error::CommonError;

/// OLS/WLSの計算過程で発生しうるエラー。
///
/// `engine`はPyO3を知らないため、Python例外への変換は`engine_pybind`側で行う
/// （`.claude/rules/rust-style.md`「エラーハンドリング」参照）。バリアントと
/// Python例外の対応は`docs/spec/ols-spec.md`の表を参照。
///
/// 【スコープの注意】欠損値（null）・`hac_time`の順序づけ失敗（同値）等、polarsの
/// 列データそのものに起因する検証は`engine_pybind::column_extraction`の責務であり、
/// ここには含めない（`engine`は`&[f64]`等、既にクリーンな値しか受け取らない前提）。
/// 正規方程式ソルバー実装等の後続issueで必要になった場合はバリアントを随時追加する。
#[derive(Debug, Error, PartialEq)]
pub enum LeastSquaresError {
    /// 系統をまたいで共通のバリデーション・計算エラー（`CommonError`参照）。
    #[error(transparent)]
    Common(#[from] CommonError),

    /// WLSの重み配列とyの行数が一致しない。
    #[error("dimension mismatch: y has {y_rows} rows but weight has {weight_rows} rows")]
    WeightDimensionMismatch { y_rows: usize, weight_rows: usize },

    /// WLSの重みが0以下（NaNを含む）。analytic weightとして扱うため正の値のみ許容する
    /// （`docs/spec/wls-spec.md`「API引数」参照）。
    #[error("weight at row {row} must be positive, got {weight}")]
    NonPositiveWeight { row: usize, weight: f64 },

    /// `hac_lags`が負、または観測数`n`以上。
    #[error("hac_lags must be in the range [0, n): got {hac_lags}, n={n}")]
    InvalidHacLags { hac_lags: i64, n: usize },

    /// 設計行列が特異（完全な多重共線性等）。
    #[error("design matrix is singular (perfect multicollinearity detected)")]
    SingularMatrix,

    /// Breusch-Godfrey検定のラグ次数`nlags`が1未満。
    #[error("nlags must be a positive integer: got {nlags}")]
    InvalidNlags { nlags: i64 },

    /// 事後診断検定（White検定等）の補助回帰に対して観測数が足りない。補助回帰の列数`k`
    /// （定数を含む）は元のモデルの説明変数の数から決まり、元のモデルの`n > k`より大きく
    /// なりうるため、`CommonError::InsufficientObservations`とは別のメッセージにする。
    #[error(
        "insufficient observations for the auxiliary regression of the diagnostic test: \
         n={n} must be greater than the number of auxiliary regressors including the \
         intercept (k={k})"
    )]
    InsufficientObservationsForAuxRegression { n: usize, k: usize },
}

/// テスト用の`time_order`: 行順をそのまま時系列順とする`[0.0, 1.0, ..., n-1]`。
#[cfg(test)]
pub(crate) fn row_time_order(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CommonError;

    #[test]
    fn least_squares_error_messages_are_human_readable() {
        // 6種の共通バリアント（DimensionMismatch等）のメッセージ検証は
        // `engine::error`側のテストに集約済み。ここではOLS/WLS固有の
        // バリアントに加え、`Common`が`CommonError`のDisplayをtransparentに転送する
        // ことだけを確認する。
        assert_eq!(
            LeastSquaresError::WeightDimensionMismatch {
                y_rows: 10,
                weight_rows: 8
            }
            .to_string(),
            "dimension mismatch: y has 10 rows but weight has 8 rows"
        );
        assert_eq!(
            LeastSquaresError::NonPositiveWeight {
                row: 3,
                weight: 0.0
            }
            .to_string(),
            "weight at row 3 must be positive, got 0"
        );
        assert_eq!(
            LeastSquaresError::InvalidHacLags {
                hac_lags: -1,
                n: 100
            }
            .to_string(),
            "hac_lags must be in the range [0, n): got -1, n=100"
        );
        assert_eq!(
            LeastSquaresError::SingularMatrix.to_string(),
            "design matrix is singular (perfect multicollinearity detected)"
        );
        assert_eq!(
            LeastSquaresError::Common(CommonError::MissingClusterColumn).to_string(),
            "cov_type='cluster' requires cluster identifiers to be provided"
        );
    }

    #[test]
    fn least_squares_error_implements_partial_eq() {
        assert_eq!(
            LeastSquaresError::SingularMatrix,
            LeastSquaresError::SingularMatrix
        );
        assert_ne!(
            LeastSquaresError::Common(CommonError::InsufficientClusters { g: 1 }),
            LeastSquaresError::Common(CommonError::InsufficientClusters { g: 0 })
        );
    }
}
