pub(crate) mod common;
pub mod fe;
pub mod re;

// common.rs: FE/RE間で共有するエラー変換（panel_error_to_pyerr）を置く。
// crate外には公開しない（`pub(crate)`）。列抽出（`column_extraction`）も使うためcrate内には見せる。
// rust-style.md「ファイル・ディレクトリ構成」の「系統内で共有するロジックはcommon.rsに置く」参照。
