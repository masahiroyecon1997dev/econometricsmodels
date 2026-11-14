//! nonlinear系統（Logit/Probit/Tobit）で共有する部品（エラー型・検証・オプション・適合度・限界効果・
//! 予測・ソルバー・初期値・標準化・共分散）。以下は、エラー型`MleError`の方針。
//!
//! OLSは1手法1エラー型（`LeastSquaresError`、旧`OlsError`）だったが、nonlinear系統は`raise_on_non_convergence`
//! 未収束・観測数不足・`confidence_level`範囲外等、3手法でほぼ共通のバリアントが多いため、
//! `<系統>/common.rs`に共有型として定義する（`.claude/rules/rust-style.md`「ファイル・
//! ディレクトリ構成」参照）。Tobit専用のバリアント（`CensoringBoundsMissing`等）も、別のenumに
//! 分離せず同じ`MleError`に含める（Logit/Probitの`fit()`はそのバリアントを構築しないだけで、
//! 型を分ける必要はない。OLSの`CovType::Hac`/`CovType::Cluster`がフィールド付きバリアントとして
//! 共存しているのと同じ考え方）。
//!
//! バリアント一覧・Python例外との対応表は`docs/spec/nonlinear-common.md`2章
//! 「エラー型（`MleError`、共有）」を参照。
//!
//! `DimensionMismatch`/`InsufficientObservations`/`InvalidConfidenceLevel`/
//! `MissingClusterColumn`/`InsufficientClusters`等は、linear系統の`LeastSquaresError`と
//! 文言まで完全に重複していたため`engine::shared::error::CommonError`に切り出し、
//! `Common`バリアント経由で保持する（数値的なほぼ特異性は`CommonError::NearSingularMatrix`）。
//!
//! # ファイル構成
//!
//! このファイルは公開する面（再エクスポート）だけを持つ。ロジックは子モジュールに置き、
//! テストは各ロジックと同じファイルの`mod tests`に置く。
//!
//! - `error.rs`: `MleError`
//! - `validation.rs`: `fit()`冒頭の入力検証
//! - `options.rs`: `SolverType`・`CovType`・`MleFitOptions`
//! - `goodness_of_fit.rs`: 適合度統計量
//! - `marginal_effects.rs`: 限界効果とデルタ法
//! - `predict.rs`: 線形予測子・予測確率・的中表
//! - `normal.rs`: 標準正規分布の`φ`・`Φ`の評価（クランプ）
//! - `numerics.rs`: 補償和・ノルム・小さな密行列演算
//! - `solver.rs`: `run_solver`と`BudgetedProblem`
//! - `line_search.rs`: line searchの定数と停滞の判定
//! - `newton.rs`: `FaerNewton`
//! - `bfgs.rs`: `FaerBfgs`
//! - `lbfgs.rs`: `FaerLbfgs`
//! - `initial.rs`: 設計行列のランクチェックと初期値
//! - `standardize.rs`: 列の標準化と逆変換
//! - `cov.rs`: `cov_type`別の共分散行列と、`CovType`からの振り分け
//! - `test_support.rs`: 複数ファイルのテストが共有するフィクスチャ（`#[cfg(test)]`）

mod bfgs;
mod cov;
mod error;
mod goodness_of_fit;
mod initial;
mod lbfgs;
mod line_search;
mod marginal_effects;
mod newton;
mod normal;
mod numerics;
mod options;
mod predict;
mod solver;
mod standardize;
#[cfg(test)]
pub(crate) mod test_support;
mod validation;

#[cfg(test)]
pub(crate) use cov::cluster_cov_params;
pub use cov::{
    SandwichVariant, observed_information_cov_params, opg_cov_params, sandwich_cov_params,
};
pub(crate) use cov::{cov_params_for_type, scores_needed};
pub use error::MleError;
pub use goodness_of_fit::{GoodnessOfFit, goodness_of_fit, log_likelihood_null};
pub use initial::{checked_design_matrix_qr, ols_based_initial_params};
pub use marginal_effects::{
    FittedModelForMarginalEffects, MarginalEffects, MarginalEffectsAt, column_means,
    column_medians, dydx_and_jacobian, marginal_effects_from_w_s,
};
pub use normal::{U_CLAMP, clamped_pdf_cdf};
pub(crate) use numerics::compensated_sum;
pub use options::{CovType, MleFitOptions, SolverType};
pub(crate) use predict::linear_predictor;
pub use predict::{pred_table, predict_from_link, predict_new_data};
pub use solver::{SolverOutput, run_solver};
pub use standardize::{
    ColumnScale, destandardize_cov_params, destandardize_params, standardize_columns,
};
pub use validation::{
    validate_binary_y, validate_confidence_level, validate_has_regressors, validate_mle_options,
    validate_sufficient_observations,
};
pub(crate) use validation::{validate_cluster_cov_type, validate_fit_preconditions};
