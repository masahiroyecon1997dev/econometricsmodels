//! `linear`系統（OLS/WLS、将来のGLS・区分回帰）で共有するエラー型。
//!
//! OLS/WLS/GLS/区分回帰はいずれも最小二乗法ベースの推定のため、系統名`linear`ではなく
//! 推定方式名`LeastSquares`で命名する（nonlinear系統の`MleError`が「nonlinear」ではなく
//! 推定方式名「MLE」で命名されているのと同じ考え方。`.claude/rules/rust-style.md`
//! 「ファイル・ディレクトリ構成」参照）。
//!
//! 元々`OlsError`という名前でOLS単体のエラー型として`linear/ols.rs`に定義されていたが、
//! WLSが同じ型をそのまま再利用する設計（`OlsInput::from_columns_weighted`・
//! `OlsEstimator::fit`を無変更で流用する、`docs/planning/specs/wls-api-design.md`4.1節）に
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
/// Python例外の対応は`docs/planning/specs/ols-implementation-notes.md`の表を参照。
///
/// 【スコープの注意】欠損値（null）・`time_col`の数値キャスト失敗等、polarsの
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
    /// （`docs/planning/specs/wls-api-design.md`3.1節参照）。
    #[error("weight at row {row} must be positive, got {weight}")]
    NonPositiveWeight { row: usize, weight: f64 },

    /// `hac_lags`が負、または観測数`n`以上。
    #[error("hac_lags must be in the range [0, n): got {hac_lags}, n={n}")]
    InvalidHacLags { hac_lags: i64, n: usize },

    /// 設計行列が特異（完全な多重共線性等）。
    #[error("design matrix is singular (perfect multicollinearity detected)")]
    SingularMatrix,
}
