//! 複数の系統（`linear`/`panel`/`iv`/`nonlinear`）が共有するコード。
//!
//! 「何をするコードか」（共分散行列・検定・最小二乗・入力検証・推論統計量等）で分け、どの系統が
//! 使うかでは分けない（利用側は手法の追加で変わるため）。各系統の`common.rs`（系統内で共有する
//! ロジック）とは別。
//!
//! `engine_pybind`から使うモジュールのみ`pub`、engine内部だけで使う計算部品は`pub(crate)`。

pub mod design_matrix;
pub mod error;
pub mod inference;
pub mod linear_algebra;
pub mod parallelism;
pub mod validation;

pub(crate) mod cluster;
pub(crate) mod covariance;
pub(crate) mod goodness_of_fit;
pub(crate) mod least_squares;
pub(crate) mod wald;
