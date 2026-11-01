pub(crate) mod common;
pub mod fe;
pub mod re;

// common.rs: FE/RE間で共有するロジックを置く。エラー変換（panel_error_to_pyerr）、
// cov_type文字列のパースと使われないオプションの検査（parse_panel_cov_kind・
// check_panel_cov_options・require_dk_time）、dk_timeの列の役割の検査（validate_dk_time_role）、
// effectsの列数の検査と列の抽出（validate_effects・extract_effects）、
// 結果のn_groupsの組み立て（group_counts・groups_dict）。
// crate外には公開しない（`pub(crate)`）。
// rust-style.md「ファイル・ディレクトリ構成」の「系統内で共有するロジックはcommon.rsに置く」参照。
