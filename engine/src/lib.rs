//! `engine`: econometricsmodels の計算コア（純粋Rust、PyO3非依存）。
//!
//! 各推定手法は本クレート配下にモジュールとして追加していく想定
//! （例: `ols`, `fe`, ...）。

pub mod linear;
pub mod nonlinear;
