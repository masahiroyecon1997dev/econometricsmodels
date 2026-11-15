//! `fe_iv`系統: 固定効果つきの操作変数法（FE-2SLS・固定効果つきGMM）。
//!
//! 固定効果を除いた（within変換した）データに2SLSまたはGMMを当てる。`iv/`と同じ構成:
//!
//! - `common.rs`と`common/`: 入力型`FeIvInput`・`RoleColumns`と、2つの推定量が共有する前段（入力検査・
//!   within変換・自由度の検証）。`iv::common`に対応する。
//! - `two_sls.rs`と`two_sls/`: `FeTwoSlsEstimator`（FE-2SLS。第一段階・Sargan・Wu-Hausman・弱操作変数診断・
//!   固定効果の復元）。契約はそのモジュールdocのパイプライン。`iv::two_sls`に対応する。
//! - `gmm.rs`と`gmm/`: `FeGmmEstimator`（固定効果つきGMM。点推定・標準誤差・Hansen J）と、事後診断のC統計量
//!   `fe_gmm_c_statistic`。契約はそのモジュールdoc。`iv::gmm`に対応する。
//!
//! 定数項は持たない（固定効果が吸収する）。固定効果の方向は`FeEffects`、`cov_type`は`FeCovType`で、FEと同じ型。

pub mod common;
pub mod gmm;
pub mod two_sls;
