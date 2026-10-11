//! 複数の系統（`linear`/`panel`/`iv`/`nonlinear`）が共有するコード。
//!
//! 「何をするコードか」（共分散行列・検定・最小二乗・入力検証・推論統計量等）で分け、どの系統が
//! 使うかでは分けない（利用側は手法の追加で変わるため）。各系統の`common.rs`（系統内で共有する
//! ロジック）とは別。
//!
//! `engine_pybind`から使うモジュール（`error`・`group_codes`・`parallelism`）のみ`pub`、engine内部
//! だけで使うものは`pub(crate)`。

pub mod error;
pub mod group_codes;
pub mod parallelism;

pub(crate) mod covariance;
pub(crate) mod design_matrix;
pub(crate) mod goodness_of_fit;
pub(crate) mod inference;
pub(crate) mod least_squares;
pub(crate) mod linear_algebra;
pub(crate) mod validation;
pub(crate) mod wald;
