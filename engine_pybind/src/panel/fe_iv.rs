//! FEIV（固定効果つきの操作変数法。`estimator`で2SLSとGMMを切り替える）の推定オプション・結果、および
//! Python（polars DataFrame + 列名 + オプション）から`engine::panel::fe_iv`の推定量を呼び出すところまでの
//! 一連の処理。`iv/common/`と同じ構成:
//!
//! - `options.rs`: `FEIVOptions`のpyclass定義
//! - `option_values.rs`: 文字列のオプションの種類への変換と、使われないオプションの検査
//! - `gmm_setup.rs`: `estimator="gmm"`のときの`FeGmmType`の組み立てと`c_stat`の重み
//! - `input.rs`: 列の役割（`FeIvColumns`）と、`FeIvInput`・推定設定の組み立て（データ抽出・バリデーション）
//! - `fit.rs`: 推定量の推定と`FEIVResult`の組み立て
//! - `result.rs`: `FEIVResult`のpyclass定義と、`first_stage()`・`fixed_effects()`・診断・`c_stat()`
//!
//! 【責務分離・言語方針】`panel/fe.rs`と同じ（`.claude/rules/rust-style.md`）。公開API（`FEIVOptions`/
//! `FEIVResult`）のdocコメントと`ValidationError`のメッセージは英語、それ以外は日本語。
//!
//! 2SLSとGMMは別クラスにしない（`IV`と同じ。インターフェースを揃えて利用者が迷わないようにするため）。
//! 推定量ごとに持たない結果の項目は`None`、持たないメソッドは`METHOD_NOT_AVAILABLE_FOR_ESTIMATOR`になる
//! （`result.rs`のモジュールdoc）。

mod fit;
mod gmm_setup;
mod input;
mod option_values;
mod options;
mod result;

pub(crate) use fit::fit;
pub(crate) use input::FeIvColumns;
pub use options::FEIVOptions;
pub use result::FEIVResult;
