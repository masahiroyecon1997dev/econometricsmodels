//! `panel`系統（FE/RE）で共有する部品（エラー型・パネル索引・時点・変換・検証・共分散行列）の入口。
//! 以下は、エラー型`PanelError`の方針とバリアントの一覧。
//!
//! `LeastSquaresError`（`engine::linear::common`）・`MleError`（`engine::nonlinear::common`）・
//! `IvError`（`engine::iv::common`）の前例に倣い、FE/REで個別に`FeError`/`ReError`を作らず
//! `PanelError`を共有する（`docs/spec/panel-common.md`4.4節）。
//!
//! `DimensionMismatch`/`InsufficientObservations`/`InvalidConfidenceLevel`/
//! `MissingClusterColumn`/`InsufficientClusters`/`NearSingularMatrix`等は`engine::shared::error::
//! CommonError`に切り出し済みのため、ここでは`Common`バリアント経由で保持する。
//!
//! FE/RE固有バリアントは、`docs/spec/fe-spec.md`（FE固有論点）・`docs/spec/re-spec.md`
//! （RE固有論点）で仕様が確定しているバリデーション条件をカバーする:
//!
//! - `IdentifierDimensionMismatch`: `y`と`entity`/`time`の長さ不一致（`panel-common.md`
//!   1章）
//! - `InsufficientDegreesOfFreedom`: パネル自由度調整（`fe-spec.md`3.2節）
//! - `SingletonGroup`: 観測数1のグループ（`fe-spec.md`1章）
//! - `UnbalancedPanelForTwoWay`: 2-way REのバランスパネル必須（2-way FEは不均衡パネルを扱う）
//! - `DuplicateEntityTimePair`: 同じ`(entity, time)`の組合せが複数の行にある（2-way）
//! - `ZeroVarianceAfterDemeaning`: within変換後に分散ゼロの説明変数（`fe-spec.md`1章）
//! - `TwoWayRequiresTime`: 2-way FE・RE指定時の`time`必須（`panel-common.md`1.1節）
//! - `InvalidDkBandwidth`: `FeCovType::Dk`の明示的な`bandwidth`が`[0, t)`の範囲外
//!   （`t`はユニークな時点数。`CommonError::InvalidHacLags`と同型だが
//!   上限が観測数`n`ではなく時点数`t`）
//! - `InsufficientDkPeriodsForInference`: DK指定時に時点数`t`が同時Wald検定の対象係数の
//!   数`q`以下（`rank(S) ≤ t-1`のため検定の部分行列が構造的に特異。
//!   `CommonError::InsufficientClustersForInference`のDK版）
//! - `InStage`: 別のバリアントを、起きた計算の文脈（`stage`）つきで包む（REの内部FE推定、
//!   FE・REのDK時点数不足の全体F検定）
//! - `DegenerateDkTwoPeriods`/`DegenerateClusterTwoGroups`: FEで時点数（DK）・クラスター数が
//!   2かつ吸収したFE次元（entity、2-wayではtimeも）のいずれかで全水準が2グループに1観測ずつの
//!   とき、within変換によりスコアが恒等的にゼロになり共分散がゼロに退化する
//! - `WithinRegressionFailed`: within変換済みデータの最小二乗推定委譲の失敗
//!   （`panel-common.md`4.3節）
//! - `FTestFailed`: F統計量（`fe.rs`モジュールdoc「自由度とF統計量」、REは
//!   `re.rs`モジュールdoc「自由度・標準誤差・F統計量」）のWald検定（`crate::shared::wald::wald_f_test`）が
//!   失敗した場合。`WithinRegressionFailed`と
//!   意味が異なる（最小二乗自体は既に成功した後の、F検定固有の共分散部分行列の
//!   ほぼ特異性というbackstopのみ、`shared/wald.rs`の`wald_f_test`docコメント参照）ため別バリアントに
//!   分離した（`IvError::FirstStageFailed`が`WithinRegressionFailed`と同じ`LeastSquaresError`
//!   ラップでも変換箇所ごとに専用バリアントにする判断と同じ）。
//! - `BetweenRegressionFailed`: RE（Swamy-Arora分散成分推定、`re-spec.md`3.1節）の
//!   between回帰（`between_ssr_and_df`。エンティティ平均への`least_squares`）が
//!   失敗した場合（エンティティ数が説明変数の数以下等）。`WithinRegressionFailed`と同じ
//!   `LeastSquaresError`ラップだが、対象がFEのwithin回帰ではなくREのbetween回帰のため
//!   別バリアントにする（`FTestFailed`と同じ判断）。
//! - `PooledRegressionFailed`・`AmemiyaRequiresWithinVariation`・
//!   `NonPositiveIdiosyncraticVariance`: REの`variance_method`（Wallace-Hussain・Amemiya、
//!   `re-spec.md`3.1.2節）の1段目の回帰の失敗、Amemiyaがwithin変動の無い列を受け付けない
//!   こと、σ_ε²が正にならないこと。
//! - `QuasiDemeanedRegressionFailed`: RE（`ReEstimator::fit`、`re-spec.md`3.2節）の
//!   準偏差変換済みデータ（`quasi_demean_transform`の出力に、同じθで変換した定数列を
//!   加えたもの）への`OlsEstimator::fit_auxiliary(include_intercept=false)`委譲が失敗した場合。
//!   `WithinRegressionFailed`（FEのwithin変換済みデータ）・`BetweenRegressionFailed`
//!   （REのbetween回帰）とは対象が異なるため別バリアントにする（同じ判断の3件目）。
//! - `HausmanTestFailed`: REのハウスマン検定（`re/hausman.rs`の`re_hausman_test`、`re-spec.md`3.7節）の
//!   補助回帰または`cov_type`連動のWald検定が失敗した場合（補助回帰のランク落ち、
//!   クラスター数`G`が補助回帰の傾き係数の数以下、共分散部分行列のほぼ特異性等）。
//!   計算自体が成立しないケースのため`fit()`全体を失敗させる（設計行列の多重共線性と同じ扱い）。
//!
//! バリアントは、計算コードを書く過程で必要になった時点で足す（`LeastSquaresError`・
//! `IvError`のdocコメントと同じ方針）。
//!
//! # ファイル構成
//!
//! このファイルは公開する面（再エクスポート）だけを持つ。ロジックは子モジュールに置き、テストは
//! 各ファイルの`mod tests`に置く。
//!
//! - `error.rs`: `PanelError`・`PanelDimension`・`BetweenMeans`
//! - `index.rs`: `PanelIndex`・`EffectNames`
//! - `time_keys.rs`: `TimeKeys`
//! - `validation.rs`: 入力の次元検証と、singleton・分散ゼロ・バランス・時間変化の検査
//! - `transform.rs`: 準偏差変換・within変換
//! - `cov_type.rs`: `PanelEffects`・`PanelCovType`
//! - `cov_params.rs`: cov_type別の共分散行列（Classical・HC・Cluster・Driscoll-Kraay）
//! - `fe_adjustment.rs`: FE固有の補正（フルレバレッジ・fixest流のK・2グループ退化）
//! - `panel_cov.rs`: `panel_cov_params`（cov_typeの振り分け）と`PanelStructure`
//! - `test_support.rs`: 複数ファイルのテストが共有するフィクスチャ（`#[cfg(test)]`）

mod cov_params;
mod cov_type;
mod dummy_leverage;
mod error;
mod fe_adjustment;
mod index;
mod panel_cov;
#[cfg(test)]
mod test_support;
mod time_keys;
mod transform;
mod validation;

pub(crate) use cov_params::{
    panel_cluster_cov_params, panel_driscoll_kraay_cov_params, resolve_dk_bandwidth, xtx_inverse,
};
pub use cov_type::{PanelCovType, PanelEffects};
pub(crate) use dummy_leverage::check_leverage_dimension;
pub use error::{BetweenMeans, PanelDimension, PanelError};
pub(crate) use fe_adjustment::absorbed_parameters;
pub(crate) use index::EffectNames;
pub use index::PanelIndex;
pub(crate) use panel_cov::{PanelCov, PanelCovInput, PanelStructure, panel_cov_params};
pub use time_keys::TimeKeys;
pub(crate) use transform::{
    TwoWayTheta, WithinTransform, quasi_demean_column, quasi_demean_two_way,
};
pub(crate) use validation::{
    covers_every_combination, ensure_no_duplicate_entity_time_pairs, ensure_no_singleton_groups,
    ensure_no_zero_variance_columns, time_varying_regressor_indices,
    two_way_time_varying_regressor_indices, validate_balanced_panel, validate_column_lengths,
    validate_panel_input,
};
