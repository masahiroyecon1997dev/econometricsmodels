mod common;
pub mod ols;
pub mod wls;

// GLS実装時に追加:
// pub mod gls;
//
// common.rs: OLS/WLS間で共有するエラー変換（least_squares_error_to_pyerr）・Mat<f64>→Vec<f64>変換
// （mat_to_vec）を置く。crate外には公開しないため`mod common;`（pubなし）。
// rust-style.md「ファイル・ディレクトリ構成」の「系統内で共有するロジックはcommon.rsに置く」参照。
