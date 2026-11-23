pub(crate) mod common;

// `pub(crate)`なのは、crate外（Pythonから見える公開API）には公開しないため。`lib.rs`が
// `#[pymodule]`登録のために`crate::iv::common::{IVOptions, IVResult}`等を参照するので、
// モジュールプライベートの`mod common;`ではE0603になる（`linear/mod.rs`と同じ）。
// rust-style.md「ファイル・ディレクトリ構成」の「系統内で共有するロジックはcommon.rsに置く」参照。
