//! OLSの入力データ（被説明変数・設計行列）の型定義と、推定本体（`OlsEstimator`）。
//! 推定本体は最小二乗・共分散行列（classical/HC0-3/HAC/cluster）・Wald検定・適合度の部品
//! （`crate::shared`）を呼んで推論結果を組み立てる。予測は`predict_new_data`。
//!
//! `engine`はpolars/PyO3を一切知らない（`.claude/rules/rust-style.md`「責務分離」参照）。
//! `engine_pybind`はpolars DataFrameから列ごとに`Vec<f64>`を抽出するところまでを担い
//! （`column_extraction::extract_f64_column`）、それらの列を本モジュールの
//! `OlsInput::from_columns`に渡す。`faer::Mat`への組み立て（切片列の自動追加を含む）は
//! ここ（engine側）の責務とする。詳細は`docs/spec/ols-spec.md`
//! 「API引数」の`include_intercept`の項を参照。

mod cov_params;
mod cov_type;
mod estimator;
mod input;
mod predict;

pub use cov_type::CovType;
pub use estimator::OlsEstimator;
pub use input::OlsInput;
pub use predict::predict_new_data;
