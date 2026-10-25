//! `engine`: econometricsmodels の計算コア（純粋Rust、PyO3非依存）。
//!
//! 系統別のモジュール（`linear`/`nonlinear`/`iv`/`panel`）と、系統をまたいで共有する
//! `shared`で構成する。

pub mod iv;
pub mod linear;
pub mod nonlinear;
pub mod panel;
pub mod shared;
