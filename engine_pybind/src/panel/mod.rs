pub(crate) mod common;
pub mod fe;
pub mod fe_gmm;
pub mod fe_iv;
pub mod re;

// fe_iv.rs: FEIV（固定効果つきの2SLS）のオプション・結果・`fit`。FEの`LazyFixedEffects`・`fe_result_fields`・
// `resolve_fe_cov_type`を使う。
// fe_gmm.rs: FEGMM（固定効果つきのGMM）のオプション・結果・`fit`・`c_stat`。FEIVの`FeIvColumns`・
// `resolve_fe_cov_type`を使う。
// common.rs: FE/RE間で共有するロジックを置く。エラー変換（panel_error_to_pyerr）、
// cov_type文字列のパースと使われないオプションの検査（parse_panel_cov_kind・
// check_panel_cov_options・require_dk_time）、y/x/effects/dk_timeの列の役割の検査
// （validate_panel_roles。内部でeffectsの列数の検査validate_effectsとdk_timeの役割の検査
// validate_dk_time_roleを呼ぶ）、effectsの列の抽出（extract_effects）、
// 結果のn_groupsの組み立て（group_counts・groups_dict）。
// crate外には公開しない（`pub(crate)`）。
// rust-style.md「ファイル・ディレクトリ構成」の「系統内で共有するロジックはcommon.rsに置く」参照。
