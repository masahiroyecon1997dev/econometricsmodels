//! `newton.rs`（ニュートン法のソルバー）のテスト。観点別に子モジュールへ分ける。
//!
//! - `step.rs`: LMラダーの1ステップ（`regularized_newton_step`・`predicted_decrease`）
//! - `stall.rs`: コストの下限・停滞での収束判定
//! - `solver.rs`: `run_solver`の収束・非収束・特異Hessian

mod solver;
mod stall;
mod step;
