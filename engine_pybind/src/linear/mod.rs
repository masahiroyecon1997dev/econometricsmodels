pub(crate) mod common;
pub mod ols;
pub mod wls;

// GLS実装時に追加:
// pub mod gls;
//
// common.rs: OLS/WLS間で共有するエラー変換（least_squares_error_to_pyerr）・Mat<f64>→Vec<f64>変換
// （mat_to_vec）・cov_type文字列パース＋cluster_col/time_col抽出（parse_cov_type）を置く。
// crate外には公開しないため`pub(crate) mod common;`（pubなし）。`least_squares_error_is_
// computation_error`をiv系統（engine_pybind/src/iv/common.rs）が再利用するため、
// 同一クレート内の他系統からは見える必要があり`pub(crate)`にしている（Issue #157）。
// rust-style.md「ファイル・ディレクトリ構成」の「系統内で共有するロジックはcommon.rsに置く」参照。
