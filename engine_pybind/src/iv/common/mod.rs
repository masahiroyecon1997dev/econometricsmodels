//! `iv`系統（2SLS/GMM）で共有するユーティリティ。
//!
//! `.claude/rules/rust-style.md`「ファイル・ディレクトリ構成」: 系統内で共有するロジックは
//! `<系統>/common.rs`に置く（`engine_pybind/src/linear/common.rs`と同じ位置づけ）。
//! `IVOptions`/`IVResult`/`build_iv_input`/`fit_frame`は2SLS/GMMどちらの`estimator`でも共有する
//! （`fit_iv`という単一エントリポイントの背後で`estimator`により推定方式を切り替える設計、
//! `docs/spec/iv-spec.md`1.2節）ため、`two_sls.rs`/`gmm.rs`のような手法ごとのファイル分割は
//! しない。
//!
//! `IvError`の`Common`バリアント（`engine::shared::error::CommonError`）も、`IvError`の
//! `ErrorInfo::report()`が同じ`code`・`params`に変換する（系統ごとに同じ判定ロジックを
//! 重複させない）。
//!
//! ## `first_stage()`/`weak_instrument_f_statistics`は`estimator`に依存しない共通ロジック
//!
//! 第一段階回帰（`x_endog[j] ~ x_exog + instruments`）・弱操作変数診断（部分F統計量）は、
//! `engine::iv::common::compute_first_stage`（2SLS/GMM間で共有、`engine/src/iv/CLAUDE.md`
//! 参照）を`fit`が`estimator`によらず常に呼ぶことで、GMMでも2SLSと同じ診断情報を提供する。
//! `TwoSlsEstimator::fit`は内部でも同じ関数を呼ぶため、`estimator="2sls"`では第一段階回帰が
//! 二重計算になるが、OLS自体が軽量なため許容する（`GmmEstimator`のように第一段階回帰を
//! 必要としない推定器に合わせて`IVResult`側を単純にする方を優先した）。
//! `IVResult`は第一段階の`OlsEstimator`（設計行列を持つ）を`OLSResult`へ変換
//! （`linear::ols::ols_estimator_to_result`）して`first_stage: Vec<(String, OLSResult)>`として
//! 保持する。`first_stage()`は`clone_ref`で複製して返すだけで、pickleでは各`OLSResult`が
//! そのまま状態に入る。
//!
//! ## GMMの`gmm_weight_type`（`IVOptions.gmm_weight_type`/`cluster`/`hac_lags`/`hac_time`）
//!
//! `gmm_weight_type`は`cov_type`とは独立の軸（点推定に使う重み行列の選択、`engine::iv::gmm`の
//! モジュールdocコメント参照）だが、`cluster`/`hac_lags`/`hac_time`は`cov_type`と
//! 共用する（`IVOptions`に別フィールドを増やさない設計、`resolve_weight_type`参照）。
//! `gmm_weight_type`と`cov_type`が異なる場合でも同じ列を共用する。
//!
//! `wu_hausman_statistic`/`wu_hausman_p_value`は`estimator="gmm"`では常に`None`
//! （`GmmEstimator`はWu-Hausman検定を持たない、`docs/spec/iv-spec.md`3.6節はTwoSlsEstimator
//! のみのスコープ）。`overid_statistic`/`overid_p_value`は`estimator="gmm"`では
//! `GmmEstimator::hansen_j_statistic()`/`hansen_j_p_value()`から配線する
//! （`estimator="2sls"`のSargan検定と同じ`Option<f64>`同士の代入）。

mod diagnostics;
mod error;
mod fit;
mod gmm_setup;
mod input;
mod option_values;
mod options;
mod result;

pub(crate) use diagnostics::{CraggDonaldOutput, EffectiveFOutput, KleibergenPaapOutput};
pub(crate) use fit::fit;
pub(crate) use options::IVOptions;
pub(crate) use result::IVResult;
