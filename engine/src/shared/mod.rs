//! 複数の系統（`linear`/`panel`/`iv`/`nonlinear`）が共有する計算部品。
//!
//! 「何をするコードか」（共分散行列・検定・最小二乗等）で分け、どの系統が使うかでは分けない
//! （利用側は手法の追加で変わるため）。各系統の`common.rs`（系統内で共有するロジック）とは別。

pub(crate) mod cluster;
pub(crate) mod covariance;
pub(crate) mod goodness_of_fit;
pub(crate) mod least_squares;
pub(crate) mod wald;
