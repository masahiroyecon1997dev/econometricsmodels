//! OLSの推定オプション・結果、およびPython（polars DataFrame + 列名 + オプション）から
//! `engine::linear::ols`（正規方程式ソルバー・標準誤差・適合度統計量）を呼び出し、
//! 結果をPython側に返すところまでの一連の処理。
//!
//! 【責務分離】`.claude/rules/rust-style.md`「Python境界でのデータ受け渡し」参照。
//! polars DataFrameから列ごとの`Vec<f64>`/`Vec<String>`への抽出はここ（`column_extraction`
//! 経由）の責務。`faer::Mat`の組み立て（切片列の自動追加を含む）は`engine::linear::ols::OlsInput`
//! に委ねる（この系統のコードは`faer`を直接扱わない）。
//!
//! 【言語方針】`.claude/rules/rust-style.md`「言語方針」参照。
//! 公開API（`OLSOptions`/`OLSResult`）のdocコメントと、`ValidationError`のメッセージ文字列は英語。
//! それ以外（このファイルの説明・非公開関数のdocコメント等）は日本語のまま。

mod diagnostics;
mod estimator;
mod options;
mod result;

pub use diagnostics::{BreuschGodfreyTestOutput, BreuschPaganTestOutput, WhiteTestOutput};
pub use estimator::fit;
pub(crate) use estimator::ols_estimator_to_result;
pub use options::OLSOptions;
pub use result::OLSResult;
