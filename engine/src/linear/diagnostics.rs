//! OLSの事後診断検定（推定後に利用者が選んで呼ぶ検定）。現在はWhite検定・Breusch-Pagan検定・
//! Breusch-Godfrey検定。
//!
//! 事後診断は`fit()`では計算せず、検定ごとの独立した関数として提供する
//! （`docs/spec/inference-conventions.md`6章）。入力は推定済みの残差と説明変数の列で、
//! `OlsEstimator`自体は要らない（`engine_pybind`の`OLSResult`も`OlsEstimator`を保持せず、
//! `training_data`から説明変数を再抽出して渡す）。
//!
//! - White検定・Breusch-Pagan検定: 残差の二乗を補助回帰する検定で、補助回帰に
//!   `OlsEstimator::fit`（`CovType::Classical`）を再利用する。違いは補助回帰の説明変数だけで、
//!   以降の処理（項の除外・標準化・統計量・p値）は共有する。
//! - Breusch-Godfrey検定: `R²`と残差二乗和だけが要るため`fit`は使わず、列ノルムでスケールした
//!   列ピボットQRで補助回帰の残差二乗和を直接求める。
//!
//! # ファイル構成
//!
//! このファイルは公開する面（再エクスポート）だけを持つ。ロジックは子モジュールに置き、テストは
//! 各ファイルの`mod tests`に置く。
//!
//! - `aux_regression.rs`: 残差の二乗を補助回帰する検定の共通部分（結果型`AuxRegressionTest`・項の選択・標準化・LM/F統計量）
//! - `white.rs`: White検定（補助回帰の項を作る）
//! - `breusch_pagan.rs`: Breusch-Pagan検定
//! - `breusch_godfrey.rs`: Breusch-Godfrey検定

mod aux_regression;
mod breusch_godfrey;
mod breusch_pagan;
#[cfg(test)]
mod test_support;
mod white;

pub use aux_regression::AuxRegressionTest;
pub use breusch_godfrey::{BreuschGodfreyTest, breusch_godfrey_test};
pub use breusch_pagan::{BreuschPaganTest, breusch_pagan_test};
pub use white::{WhiteTest, white_test};
