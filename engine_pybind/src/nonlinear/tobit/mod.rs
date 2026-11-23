//! Tobitの推定オプション・結果、およびPython（polars DataFrame + 列名 + オプション）から
//! `engine::nonlinear::tobit`（Newton/BFGS/L-BFGSソルバー・標準誤差・適合度統計量・
//! 限界効果・予測値・打ち切り適合度チェック）を呼び出し、結果をPython側に返すところまでの
//! 一連の処理。基本構成は`logit.rs`/`probit.rs`と同型（`engine_pybind/src/nonlinear/
//! CLAUDE.md`参照）だが、以下の点がTobit固有:
//!
//! - `TobitOptions`は`lower`/`upper`（打ち切り境界）フィールドを追加で持つ
//!   （`docs/spec/nonlinear-common.md`7章で確定済みの既定値`lower=Some(0.0)`・`upper=None`）
//! - `TobitResult`の`params`/`param_names`/`std_errors`等は`(k+1)`長に統一する
//!   （`engine::nonlinear::tobit::TobitEstimator::params()`は`β`のみ`k`長だが、
//!   `std_errors()`等は`σ`を含む`k+1`長という非対称な設計になっているため、Python側では
//!   `param_names`の末尾に`"sigma"`を追加して`params`に`sigma()`の値を追加し、全フィールドを
//!   `(k+1)`長に揃えてzipできるようにする）
//! - `predict()`/`marginal_effects()`は`target`引数（`"expected_latent"`/
//!   `"expected_observed"`/`"prob_uncensored"`/`"expected_uncensored"`、Rust側
//!   `MarginalEffectsTarget`のsnake_case版）を追加で受け取る
//! - `pred_table()`の代わりに`censoring_fit_check()`を提供する（`y`が連続変数のため
//!   分類の的中表は意味を持たない、`docs/spec/nonlinear-common.md`6章）
//! - `log_likelihood_null`/`lr_statistic`/`lr_p_value`/`pseudo_r_squared`は無い
//!   （Tobitはこの共通コアから意図的に外れる、`docs/spec/nonlinear-common.md`5章）
//!
//! 【責務分離】【言語方針】は`logit.rs`のモジュールdocコメントと同じ

mod censoring_fit;
mod fit;
mod input;
mod options;
mod result;

pub use censoring_fit::{CensoringFitCategoryResult, CensoringFitCheckResult};
pub(crate) use fit::fit;
pub use options::TobitOptions;
pub use result::TobitResult;
