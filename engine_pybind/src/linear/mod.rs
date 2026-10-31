pub(crate) mod common;
pub mod ols;
pub mod wls;

// GLS実装時に追加:
// pub mod gls;
//
// common.rs: OLS/WLS間で共有するエラー変換（least_squares_error_to_pyerr）・
// cov_type文字列のパース・使われないオプションの検査・cluster/hac_time列の抽出
// （parse_cov_kind・check_cov_options・resolve_cov_type）を置く。
// crate外には公開しないため`pub(crate) mod common;`（pubなし）。iv系統
// （engine_pybind/src/iv/common.rs）が`CovKind`・`parse_cov_kind`・`require_hac_time`・
// `resolve_cov_type`を再利用するため、同一クレート内の他系統から見える`pub(crate)`にしている。
// rust-style.md「ファイル・ディレクトリ構成」の「系統内で共有するロジックはcommon.rsに置く」参照。
