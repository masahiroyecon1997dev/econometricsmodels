//! `fe_iv`（FE-2SLS・固定効果つきGMM）で共有する入力型と、推定の前段。
//!
//! このファイルは再エクスポートだけを持ち、中身は`common/`の子モジュールに置く:
//! `input.rs`（`FeIvInput`・`RoleColumns`）・`prepare.rs`（入力検査・within変換・自由度の検証。
//! FE-2SLSと固定効果つきGMMが共有する前段）・`test_support.rs`（`#[cfg(test)]`の共有フィクスチャ）。

mod input;
mod prepare;
#[cfg(test)]
pub(in crate::panel::fe_iv) mod test_support;

pub use input::{FeIvInput, RoleColumns};
pub(in crate::panel::fe_iv) use prepare::{
    WithinPrepared, abs_max, in_first_stage, mat_from_columns, prepare_within,
    structural_fit_is_perfect,
};
