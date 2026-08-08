pub(crate) mod common;

// 2SLS実装時に追加:
// pub mod two_sls;
// GMM実装時に追加:
// pub mod gmm;
//
// common.rs: 2SLS/GMM間で共有するエラー変換（iv_error_to_pyerr）・IvOptions/IvResult
// pyclass・データ抽出（build_iv_input）を置く。crate外（Pythonから見える公開API）には
// 公開しないため`pub`ではなく`pub(crate)`（`linear/mod.rs`の`pub(crate) mod common;`と
// 同じ理由）。crate内では`lib.rs`（クレートルート）がIssue #169で`#[pymodule]`登録の
// ために`crate::iv::common::{IvOptions, IvResult}`等を参照する必要があるため、`mod common;`
// （モジュールプライベート、`iv`の子孫からしか見えない）では不十分（rust-reviewerの指摘、
// 実際に`lib.rs`から参照するコードでE0603を確認済み）。
// rust-style.md「ファイル・ディレクトリ構成」の「系統内で共有するロジックはcommon.rsに置く」参照。
