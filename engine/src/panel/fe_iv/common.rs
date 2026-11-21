//! `fe_iv`（FE-2SLS・固定効果つきGMM）で共有する入力型と、推定の前段・第一段階・診断。
//!
//! このファイルは再エクスポートだけを持ち、中身は`common/`の子モジュールに置く:
//! `input.rs`（`FeIvInput`・`RoleColumns`）・`prepare.rs`（入力検査・within変換・自由度の検証。
//! FE-2SLSと固定効果つきGMMが共有する前段）・`first_stage.rs`（内生変数ごとの第一段階）・
//! `weak_instruments.rs`（第一段階だけに依存する弱操作変数診断）・`wu_hausman.rs`（Wu-Hausman内生性検定）・
//! `test_support.rs`（`#[cfg(test)]`の共有フィクスチャ）。

mod first_stage;
mod input;
mod prepare;
#[cfg(test)]
pub(in crate::panel::fe_iv) mod test_support;
mod weak_instruments;
mod wu_hausman;

pub(in crate::panel::fe_iv) use first_stage::{
    FirstStageArgs, FirstStageWithin, FirstStages, effect_names, pick_absorbed, run_first_stages,
};
pub use input::{FeIvInput, RoleColumns};
pub(in crate::panel::fe_iv) use prepare::{
    WithinPrepared, mat_from_columns, prepare_within, structural_fit_is_perfect,
};
pub(in crate::panel::fe_iv) use weak_instruments::WeakInstrumentView;
pub(in crate::panel::fe_iv) use wu_hausman::{
    FeWuHausmanTest, WuHausmanContext, fe_wu_hausman_test,
};
