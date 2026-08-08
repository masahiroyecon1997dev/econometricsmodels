mod common;

// 2SLS実装時に追加:
// pub mod two_sls;
// GMM実装時に追加:
// pub mod gmm;
//
// common.rs: 2SLS/GMM間で共有するエラー変換（iv_error_to_pyerr）を置く。
// crate外には公開しないため`mod common;`（pubなし）。
// rust-style.md「ファイル・ディレクトリ構成」の「系統内で共有するロジックはcommon.rsに置く」参照。
