//! `panel`系統（FE/RE）で共有するエラー型。
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
//! - `UnbalancedPanelForTwoWay`: 2-way FE・REのバランスパネル必須（`fe-spec.md`1章）
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
//! - `FTestFailed`: F統計量（`fe.rs`モジュールdoc「自由度調整」のF統計量節、REは
//!   `re.rs`モジュールdoc「F統計量」）のWald検定（`crate::shared::wald::wald_f_test`）が
//!   失敗した場合。`WithinRegressionFailed`と
//!   意味が異なる（`OlsEstimator::fit`自体は既に成功した後の、F検定固有の共分散部分行列の
//!   ほぼ特異性というbackstopのみ、`shared/wald.rs`の`wald_f_test`docコメント参照）ため別バリアントに
//!   分離した（`IvError::FirstStageFailed`が`WithinRegressionFailed`と同じ`LeastSquaresError`
//!   ラップでも変換箇所ごとに専用バリアントにする判断と同じ）。
//! - `BetweenRegressionFailed`: RE（Swamy-Arora分散成分推定、`re-spec.md`3.1節）の
//!   between回帰（エンティティ平均への`OlsEstimator::fit(include_intercept=true)`）が
//!   失敗した場合（エンティティ数が説明変数の数以下等）。`WithinRegressionFailed`と同じ
//!   `LeastSquaresError`ラップだが、対象がFEのwithin回帰ではなくREのbetween回帰のため
//!   別バリアントにする（`FTestFailed`と同じ判断）。
//! - `PooledRegressionFailed`・`AmemiyaRequiresWithinVariation`・
//!   `NonPositiveIdiosyncraticVariance`: REの`variance_method`（Wallace-Hussain・Amemiya、
//!   `re-spec.md`3.1.2節）の1段目の回帰の失敗、Amemiyaがwithin変動の無い列を受け付けない
//!   こと、σ_ε²が正にならないこと。
//! - `QuasiDemeanedRegressionFailed`: RE（`ReEstimator::fit`、`re-spec.md`3.2節）の
//!   準偏差変換済みデータ（`quasi_demean_transform`の出力に、同じθで変換した定数列を
//!   加えたもの）への`OlsEstimator::fit(include_intercept=false)`委譲が失敗した場合。
//!   `WithinRegressionFailed`（FEのwithin変換済みデータ）・`BetweenRegressionFailed`
//!   （REのbetween回帰）とは対象が異なるため別バリアントにする（同じ判断の3件目）。
//! - `HausmanTestFailed`: REのハウスマン検定（`re.rs`の`re_hausman_test`、`re-spec.md`3.7節）の
//!   補助回帰または`cov_type`連動のWald検定が失敗した場合（補助回帰のランク落ち、
//!   クラスター数`G`が補助回帰の傾き係数の数以下、共分散部分行列のほぼ特異性等）。
//!   計算自体が成立しないケースのため`fit()`全体を失敗させる（設計行列の多重共線性と同じ扱い）。
//!
//! RE固有（`re-spec.md`）で追加のバリアントが必要になった場合は、FE/RE実装issueで実際に計算
//! コードを書く過程で随時追加する（`LeastSquaresError`・`IvError`のdocコメントと同じ
//! 「土台を用意し、必要になった時点で足す」方針）。

use std::fmt;

use faer::Mat;
use thiserror::Error;

use crate::linear::common::LeastSquaresError;
use crate::shared::covariance::{cluster_correction, cluster_meat, default_hac_lags, leverages};
use crate::shared::error::{ClusterSetting, CommonError, ErrorInfo, staged_report};
use crate::shared::error_catalog::{ErrorCode, ErrorReport, Stage};
use crate::shared::group_codes::GroupCodes;
use crate::shared::validation::{
    column_is_zero_variance, validate_cluster_count_covers_slopes, validate_cluster_groups,
};

/// パネルデータの2つの次元。エラーメッセージ・バリデーションで「どちらの次元の
/// 問題か」を区別するために使う。
///
/// 2-way FEではエンティティ・時点のsingletonを対称に検出する
/// （`docs/spec/fe-spec.md`1章）ため、`PanelError::SingletonGroup`が
/// この型をフィールドとして持つ。FE/RE実装が進んだ段階で、singleton検出以外
/// （`fixed_effects()`の次元指定等）でも再利用できる想定。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelDimension {
    /// エンティティ方向（個体・企業・州等、`effects`の1つ目の列）。
    Entity,
    /// 時点方向（`effects`の2つ目の列）。
    Time,
}

impl PanelDimension {
    /// エラーの`argument`に入る名前（公開される値のため、`Display`とは独立に固定する）。
    fn argument_name(self) -> &'static str {
        match self {
            PanelDimension::Entity => "entity",
            PanelDimension::Time => "time",
        }
    }
}

impl fmt::Display for PanelDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.argument_name())
    }
}

/// 効果の次元ごとの列名。エラーの`column`に載せるためだけに`FeInput`/`ReInput`が保持する
/// （`engine_pybind`が`effects`の各要素の列名を渡す）。計算には使わない。
///
/// 既定は`"entity"`・`"time"`で、`engine`単体（テスト等）の構築ではこのまま使う。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EffectNames {
    entity: String,
    time: String,
}

impl Default for EffectNames {
    fn default() -> Self {
        Self {
            entity: "entity".to_string(),
            time: "time".to_string(),
        }
    }
}

impl EffectNames {
    /// 1つ目の次元（個体）と、あれば2つ目の次元（時点）の列名を設定する。
    /// 2つ目が`None`なら既定の名前を残す（1-wayでは使われない）。
    pub(crate) fn new(entity: String, time: Option<String>) -> Self {
        let default = Self::default();
        Self {
            entity,
            time: time.unwrap_or(default.time),
        }
    }

    /// `dimension`に対応する列名。
    pub(crate) fn get(&self, dimension: PanelDimension) -> &str {
        match dimension {
            PanelDimension::Entity => &self.entity,
            PanelDimension::Time => &self.time,
        }
    }
}

/// REの分散成分推定で、between回帰がどの単位の平均に対する回帰か
/// （`PanelError::BetweenRegressionFailed`の`means`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BetweenMeans {
    /// 個体（エンティティ）平均への回帰（1-way・2-wayのσ_u²）。
    Entity,
    /// 時点平均への回帰（2-wayのσ_λ²）。
    Time,
}

impl std::fmt::Display for BetweenMeans {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Entity => "entity",
            Self::Time => "time",
        })
    }
}

/// FE/REの計算過程で発生しうるエラー。
///
/// `engine`はPyO3を知らないため、Python例外への変換は`engine_pybind`側で行う
/// （`.claude/rules/rust-style.md`「エラーハンドリング」参照）。
#[derive(Debug, Error, PartialEq)]
pub enum PanelError {
    /// 系統をまたいで共通のバリデーション・計算エラー（`CommonError`参照）。
    #[error(transparent)]
    Common(#[from] CommonError),

    /// `y`と`entity`または`time`の長さが一致しない（`FeInput::from_group_codes`、
    /// `docs/spec/panel-common.md`1章）。
    ///
    /// `y`と`x`列の不一致は`CommonError::DimensionMismatch`が既にカバーしている
    /// （対象列が異なるため専用バリアントにする）。`entity`/`time`のどちらの不一致かは
    /// `PanelDimension`で表す（`SingletonGroup`と同じ使い方）。`engine_pybind`が同じ
    /// polars DataFrameから列抽出する限り実際には起こり得ない（OLSの`y`/`x`長さ不一致
    /// チェックと同じ、`engine_pybind`〜`engine`間の契約に対する防御的な`Result`化）。
    #[error("{}", self.catalog_message())]
    IdentifierDimensionMismatch {
        dimension: PanelDimension,
        y_rows: usize,
        other_rows: usize,
    },

    /// パネル自由度調整後の残差自由度が正にならない。
    ///
    /// FEの`df_resid`は`n_obs`から個体ダミー相当の自由度を追加で消費するため、
    /// - 1-way: `df_resid = n_obs - n_entities - k`
    /// - 2-way: `df_resid = n_obs - n_entities - n_periods + 1 - k`
    ///   （entityダミーとtimeダミーの間の定数項ぶんのランク落ちを`+1`で補正、`fe-spec.md`3.2節）
    ///
    /// となる。`CommonError::InsufficientObservations`（単純な`n <= k`）とは
    /// 消費する自由度の内訳が異なるため別バリアントにする。`n_periods`は1-wayでは
    /// `None`（エラーの`n_effects`は個体数だけになる）。
    #[error("{}", self.catalog_message())]
    InsufficientDegreesOfFreedom {
        n_obs: usize,
        n_entities: usize,
        n_periods: Option<usize>,
        k: usize,
    },

    /// 観測数1のグループ（singleton）を検出した。エンティティ方向は常に、2-way FEでは
    /// 時点方向も対称に検出する（`docs/spec/fe-spec.md`1章）。
    ///
    /// listwise deletion等の自動除外はせず常にエラーとする（欠損値を常にエラーとする
    /// 全体方針の踏襲）。下流の特異行列エラーとして偶発的に検出される形にはせず、
    /// 「観測数1のグループ」を明示的にこのバリアントで弾く。
    #[error("{}", self.catalog_message())]
    SingletonGroup {
        dimension: PanelDimension,
        /// そのグループを定める列の名前（`effects`の要素）。
        column: String,
        group_id: String,
    },

    /// 2-way FE（entity + time FE）に不均衡（unbalanced）パネルが渡された。
    ///
    /// 2-way FEのwithin変換は閉形式の二重デミーニング
    /// （`ỹ_it = y_it - ȳ_i. - ȳ_.t + ȳ..`）で計算するが、この閉形式は
    /// バランスパネルでのみ正確なため、2-wayでは常にバランスパネルを必須とする
    /// （`docs/spec/fe-spec.md`1章）。1-way FEはエンティティ平均を引くだけで
    /// 不均衡でも数学的に正確に成立するため、このエラーは発生しない。
    ///
    /// なお「バランス」の判定は観測数カウント（`n_obs == n_entities * n_periods`）
    /// だけでは不十分で、あるペアの重複と別ペアの欠落が相殺してカウントだけ一致する
    /// 入力もありうる。このエラーを構築する検証は、(entity, time)ペアの一意性・
    /// 全組合せの充足まで確認する必要がある（`validate_balanced_panel`）。
    /// `entity_column`・`time_column`は`effects`の1つ目・2つ目の列名（メッセージ用）。
    #[error("{}", self.catalog_message())]
    UnbalancedPanelForTwoWay {
        n_obs: usize,
        entity_column: String,
        n_entities: usize,
        time_column: String,
        n_periods: usize,
    },

    /// within変換（デミーニング）後に分散がゼロになる説明変数がある。
    ///
    /// 時間不変変数（1-way FEで問題になる典型例）だけでなく、2-wayで time FE と
    /// 完全共線な「エンティティ間で変動しない列」も同じチェックで検出できる
    /// （`docs/spec/fe-spec.md`1章。1-way/2-wayで同一ロジックを共有する）。
    #[error("{}", self.catalog_message())]
    ZeroVarianceAfterDemeaning { column: String },

    /// 被説明変数がwithin変換後に分散ゼロ（個体内で定数、2-wayなら個体効果と時点効果だけで
    /// 説明される）。固定効果が説明するものが残らず、within回帰のR-squaredが定義できない。
    /// 説明変数の`ZeroVarianceAfterDemeaning`と同じ相対閾値で判定する
    /// （[`crate::shared::validation::column_is_zero_variance`]）。
    #[error("{}", self.catalog_message())]
    OutcomeConstantWithinEntity,

    /// 2-way FE（entity + time FE）を要求したのに`time`列が指定されていない。
    ///
    /// `time`は`FEOptions`内の`Option`フィールドで、2-way指定時のみ実質必須になる
    /// 「条件付き必須」パターン（`panel-common.md`1.1節。`OLSOptions.cluster`が
    /// `cov_type="cluster"`のときだけ必須になるのと同型）。未指定時のバリデーション
    /// エラーとしてここで担保する。
    ///
    /// `engine_pybind`は`time`の有無から1-way/2-wayを決めるため、Python APIからは起こらない
    /// （起きたら実装の誤りなので`INTERNAL_ERROR`として報告する）。
    #[error("{}", self.catalog_message())]
    TwoWayRequiresTime,

    /// `FeCovType::Dk`の明示的な`bandwidth`が`[0, t)`の範囲外（`t`はユニークな時点数）。
    ///
    /// `CommonError::InvalidHacLags`と同型のバリデーションだが、上限が観測数`n`
    /// ではなく時点数`t`になる点が異なる（DKのバンド幅は「時点のラグ」であり「観測の
    /// ラグ」ではないため、`engine/src/panel/CLAUDE.md`「Driscoll-Kraay型パネルHAC対応」
    /// 参照）。
    #[error("{}", self.catalog_message())]
    InvalidDkBandwidth { bandwidth: i64, t: usize },

    /// `FeCovType::Dk`/`ReCovType::Dk`でユニークな時点数`t_periods`が1しかない。
    ///
    /// fixestの`ssc()`小標本補正はDKの時点数`t_periods`を、clusterの
    /// クラスター数`G`と同じ役割（`G/(G-1)`補正・推論の自由度`G-1`）で使う
    /// （`panel_driscoll_kraay_cov_params`のdocコメント参照）。`t_periods=1`だと
    /// `t_periods/(t_periods-1)`が`1/0`に発散し、クラスターの`G=1`が
    /// `validate_cluster_groups`で拒否されるのと同じ理由で計算が成立しない
    /// （`ols::validate_cluster_groups`の`G>=2`要求と同型の前提）。
    #[error("{}", self.catalog_message())]
    InsufficientDkPeriods { t_periods: usize },

    /// `FeCovType::Dk`/`ReCovType::Dk`で、ユニークな時点数`t_periods`が同時Wald検定の
    /// 対象となる係数の数`q`以下（FEは傾き係数のF検定、REはハウスマン検定の`X̃`ブロック）。
    ///
    /// DKの`S`行列は時点ごとのスコア`h_t = Σ_i x_it·e_it`の外積（とそのラグ項）の和で
    /// `{h_1, …, h_T}`の張る空間に収まり、正規方程式`Σ_t h_t = X'e = 0`により
    /// `rank(S) ≤ t_periods - 1`。`t_periods <= q`だと検定の`q×q`部分行列が構造的に
    /// 特異になる（`CommonError::InsufficientClustersForInference`の`g <= q`と同じ構造）。
    /// 数値的な特異性判定（`ensure_well_conditioned_symmetric_matrix`）任せにすると、
    /// 理論上0の固有値に乗る丸め誤差が閾値を超えた場合に巨大な無意味な統計量を
    /// 黙って返しうるため、入力だけから判定できるこの条件を行列計算の前に弾く。
    #[error("{}", self.catalog_message())]
    InsufficientDkPeriodsForInference { t_periods: usize, q: usize },

    /// FEで`FeCovType::Dk`の時点数が2、かつ吸収したFE次元（常にentity、2-wayではtimeも）の
    /// いずれかで全水準がその2時点に1観測ずつ（傾き係数`k >= 1`のとき）。
    ///
    /// within変換は吸収した各次元の水準内で和をゼロにするため、そうした水準の2観測で
    /// `x̃`・`ẽ`が符号反転し（entity方向の例: `x̃_i1 = -x̃_i2`・`ẽ_i1 = -ẽ_i2`）、
    /// 時点スコアは`h_1 = h_2`。2-wayのtime方向は、エンティティ2つのパネルで`Dk.time`に
    /// entityと同じ分け方の列を渡した場合に起きる。正規方程式の
    /// `h_1 + h_2 = 0`と合わせて`h_1 = h_2 = 0`、DK共分散は恒等的にゼロになる
    /// （`InsufficientDkPeriodsForInference`の`rank(S) ≤ t-1 = 1`よりさらに強い退化で、
    /// `k = 1`でも成立しない）。数値的な特異性判定では`1×1`行列を検出できず、
    /// 約`1e-16`の標準誤差と巨大なF統計量を黙って返していたため入力から弾く。
    /// 1水準でもこのパターンを崩せば退化しない（その場合は通す）。
    /// REは残差がwithin変換されないため対象外。
    #[error("{}", self.catalog_message())]
    DegenerateDkTwoPeriods,

    /// FEで`FeCovType::Cluster`のクラスター数が2、かつ吸収したFE次元（常にentity、2-wayでは
    /// timeも）のいずれかで全水準が2つのクラスターに1観測ずつ（傾き係数`k >= 1`のとき）。
    /// 典型例は2時点のパネルを`time`でクラスタリング（entity方向）と、2-way FEでエンティティ
    /// 2つのパネルをentityでクラスタリング——Clusterの既定——（time方向）。
    ///
    /// `DegenerateDkTwoPeriods`と同じ理由でクラスタースコアが`s_1 = s_2 = 0`となり、
    /// クラスターロバスト共分散が恒等的にゼロになる。
    #[error("{}", self.catalog_message())]
    DegenerateClusterTwoGroups,

    /// within変換済みデータに対する最小二乗推定（`OlsEstimator::fit`への委譲、
    /// `panel-common.md`4.3節。WLSがOLSへ委譲するのと同型のパターン）が失敗した。
    ///
    /// 委譲先が返す`LeastSquaresError`をそのまま保持する（IVの`SecondStageFailed
    /// { source }`と同型のラップ）。`#[from]`で透過させず明示的に
    /// `.map_err(|source| PanelError::WithinRegressionFailed { source })`で包むのは、
    /// `CommonError`が`Common`と`WithinRegressionFailed(LeastSquaresError::Common)`の
    /// 2経路で`PanelError`になりうる曖昧さを避け、変換箇所を追跡可能にするため
    /// （`IvError::FirstStageFailed`が`#[from]`を使わない判断と同じ）。
    #[error("{}", self.catalog_message())]
    WithinRegressionFailed {
        #[source]
        source: LeastSquaresError,
    },

    /// F統計量のWald検定（`crate::shared::wald::wald_f_test`への委譲）が
    /// 失敗した。`WithinRegressionFailed`とは別バリアント（理由はモジュールdoc参照）。
    ///
    /// 実際に発生しうるのは`LeastSquaresError::Common(CommonError::NearSingularMatrix)`
    /// のみ（`wald_f_test`のdocコメント「backstop」参照。傾き係数間の極端なスケール差等で
    /// 共分散部分行列が数値的にほぼ特異な場合）。それでも型は`WithinRegressionFailed`と
    /// 同じ`LeastSquaresError`のまま保持する（`wald_f_test`のエラー型を独自に絞り込む
    /// メリットが無いため）。
    #[error("{}", self.catalog_message())]
    FTestFailed {
        #[source]
        source: LeastSquaresError,
    },

    /// RE（Swamy-Arora分散成分推定、`re-spec.md`3.1節）のbetween回帰
    /// （エンティティ平均への`OlsEstimator::fit(include_intercept=true)`）が失敗した。
    ///
    /// `WithinRegressionFailed`と同じ`LeastSquaresError`ラップだが、対象がFEのwithin回帰
    /// ではなくREのbetween回帰のため別バリアントにする（`FTestFailed`と同じ判断、
    /// モジュールdoc参照）。最も起こりやすいのはエンティティ数が説明変数の数以下
    /// （`CommonError::InsufficientObservations`）。
    #[error("{}", self.catalog_message())]
    BetweenRegressionFailed {
        /// どの平均への回帰が失敗したか（2-wayは個体平均・時点平均の2本を推定する）。
        means: BetweenMeans,
        #[source]
        source: LeastSquaresError,
    },

    /// RE（`variance_method`が`WallaceHussain`、`re-spec.md`3.1.2節）の1段目の回帰
    /// （全説明変数と切片へのプールドOLS）が失敗した。
    ///
    /// 残差の二乗和から分散成分を求めるため、設計行列が特異（完全な多重共線性）だと
    /// 推定できない。`WithinRegressionFailed`等と同じ`LeastSquaresError`ラップだが、
    /// 対象が別のため別バリアントにする。
    #[error("{}", self.catalog_message())]
    PooledRegressionFailed {
        #[source]
        source: LeastSquaresError,
    },

    /// RE（`variance_method`が`Amemiya`、`re-spec.md`3.1.2節）で、within変動の無い説明変数
    /// （時間不変の列。2-wayでは個体間でのみ・時点間でのみ変わる列も）がある。
    ///
    /// Amemiya法は1段目をwithin推定だけで行うため、within変動の無い列の効果が残差に残り、
    /// σ_u²が偏る。`plm`も不均衡パネルでは同じ理由で推定を拒否する。
    #[error("{}", self.catalog_message())]
    AmemiyaRequiresWithinVariation {
        /// within変動の無い列の名前。
        columns: Vec<String>,
    },

    /// RE（`variance_method`が`WallaceHussain`・`Amemiya`）で推定したσ_ε²が0以下になった
    /// （θが定まらない）。`plm`は0に切り上げてθが`NaN`になる。
    #[error("{}", self.catalog_message())]
    NonPositiveIdiosyncraticVariance {
        /// 推定法の名前（`variance_method`の値）。
        method: &'static str,
        sigma2_eps: f64,
    },

    /// RE（`variance_method`が`WallaceHussain`・`Amemiya`）の分散成分を決めるモーメント方程式
    /// （2×2）が特異で解けない。
    #[error("{}", self.catalog_message())]
    VarianceComponentsMomentsSingular {
        /// 推定法の名前（`variance_method`の値）。
        method: &'static str,
    },

    /// RE（`ReEstimator::fit`、`re-spec.md`3.2節）の準偏差変換済みデータへの
    /// `OlsEstimator::fit(include_intercept=false)`委譲が失敗した。
    ///
    /// `WithinRegressionFailed`（FEのwithin変換済みデータ）・`BetweenRegressionFailed`
    /// （REのbetween回帰）とは対象が異なるため別バリアントにする（同じ判断の3件目、
    /// モジュールdoc参照）。
    #[error("{}", self.catalog_message())]
    QuasiDemeanedRegressionFailed {
        #[source]
        source: LeastSquaresError,
    },

    /// REのハウスマン検定（`re-spec.md`3.7節）の補助回帰、または`cov_type`連動のWald検定が
    /// 失敗した。
    ///
    /// 補助回帰のランク落ち、クラスター数`G`が補助回帰の傾き係数の数`2k`以下
    /// （`CommonError::InsufficientClustersForInference`）、DK・ロバスト共分散部分行列の
    /// ほぼ特異性（`CommonError::NearSingularMatrix`）等。DKの時点数不足（`T <= q`、`q`はRE本体の
    /// 傾き係数の数）はこれに包まず、補助回帰の前に`InsufficientDkPeriodsForInference`で弾く。ハウスマン検定はRE本体の付随的な
    /// 診断情報だが、計算自体が成立しない場合は`None`で隠さずエラーにする
    /// （設計行列の多重共線性でエラーにするのと同じ方針）。
    #[error("{}", self.catalog_message())]
    HausmanTestFailed {
        #[source]
        source: LeastSquaresError,
    },

    /// `source`のエラーを、起きた計算の文脈（`stage`）つきで報告する。
    ///
    /// 同じ検証や計算が複数の文脈で呼ばれ、文面だけでは何の計算で失敗したか分からないときに使う
    /// （REの分散成分の推定に使う内部FEの自由度不足、FE・REの全体F検定のDK時点数不足）。
    /// `report()`は`source`の`code`に`stage`を付ける（`stage`を付けてよい`code`のみ。`source`が
    /// すでに別の`stage`を持つときは置き換える）。
    #[error("{}", self.catalog_message())]
    InStage {
        stage: Stage,
        #[source]
        source: Box<PanelError>,
    },
}

impl ErrorInfo for PanelError {
    /// `WithinRegressionFailed`等の段階ごとの包みは、`source`の`code`（`stage`を付けてよい
    /// `code`なら`stage`つき）を返す。
    fn report(&self) -> ErrorReport {
        match self {
            Self::Common(common) => common.report(),
            Self::WithinRegressionFailed { source } => {
                staged_report(source.report(), Stage::WithinRegression)
            }
            Self::FTestFailed { source } => staged_report(source.report(), Stage::OverallFTest),
            Self::BetweenRegressionFailed { means, source } => staged_report(
                source.report(),
                match means {
                    BetweenMeans::Entity => Stage::BetweenRegressionEntity,
                    BetweenMeans::Time => Stage::BetweenRegressionTime,
                },
            ),
            Self::PooledRegressionFailed { source } => {
                staged_report(source.report(), Stage::PooledRegression)
            }
            Self::QuasiDemeanedRegressionFailed { source } => {
                staged_report(source.report(), Stage::QuasiDemeanedRegression)
            }
            Self::HausmanTestFailed { source } => {
                staged_report(source.report(), Stage::HausmanRegression)
            }
            Self::InStage { stage, source } => staged_report(source.report(), stage.clone()),
            Self::IdentifierDimensionMismatch {
                dimension,
                y_rows,
                other_rows,
            } => ErrorReport::new(ErrorCode::InternalError).param(
                "detail",
                format!("y has {y_rows} rows but {dimension} has {other_rows} rows"),
            ),
            Self::TwoWayRequiresTime => ErrorReport::new(ErrorCode::InternalError).param(
                "detail",
                "two-way effects were requested without a time column",
            ),
            Self::InsufficientDegreesOfFreedom {
                n_obs,
                n_entities,
                n_periods,
                k,
            } => {
                // 2-wayは個体ダミーと時点ダミーの間の定数項ぶんを1つ引く。
                let n_effects = n_entities + n_periods.map_or(0, |t| t.saturating_sub(1));
                ErrorReport::new(ErrorCode::InsufficientDegreesOfFreedom)
                    .param("n_obs", *n_obs)
                    .param("n_params", *k)
                    .param("n_effects", n_effects)
                    .param("min_obs", n_effects + k + 1)
            }
            Self::SingletonGroup {
                dimension: _,
                column,
                group_id,
            } => ErrorReport::new(ErrorCode::SingletonGroup)
                .param("argument", "effects")
                .param("column", column.as_str())
                .param("group", group_id.as_str()),
            Self::UnbalancedPanelForTwoWay {
                n_obs,
                entity_column,
                n_entities,
                time_column,
                n_periods,
            } => ErrorReport::new(ErrorCode::PanelNotBalancedForTwoWay)
                .param("n_obs", *n_obs)
                .param("first_column", entity_column.as_str())
                .param("n_first_groups", *n_entities)
                .param("second_column", time_column.as_str())
                .param("n_second_groups", *n_periods),
            Self::ZeroVarianceAfterDemeaning { column } => {
                ErrorReport::new(ErrorCode::RegressorNoWithinVariation)
                    .param("column", column.as_str())
            }
            Self::OutcomeConstantWithinEntity => {
                ErrorReport::new(ErrorCode::OutcomeConstantWithinEntity)
            }
            Self::InvalidDkBandwidth { bandwidth, t } => {
                ErrorReport::new(ErrorCode::DkBandwidthOutOfRange)
                    .param("value", *bandwidth)
                    .param("n_periods", *t)
                    .param("max", t.saturating_sub(1))
            }
            Self::InsufficientDkPeriods { t_periods } => {
                ErrorReport::new(ErrorCode::DkPeriodsTooFew)
                    .param("n_periods", *t_periods)
                    .param("min_periods", 2_usize)
            }
            Self::InsufficientDkPeriodsForInference { t_periods, q } => {
                ErrorReport::new(ErrorCode::DkPeriodsNotAboveCoefficients)
                    .param("n_periods", *t_periods)
                    .param("n_coefficients", *q)
                    .param("min_periods", q + 1)
            }
            Self::DegenerateDkTwoPeriods => ErrorReport::new(ErrorCode::DkTwoPeriodsDegenerate),
            Self::DegenerateClusterTwoGroups => {
                ErrorReport::new(ErrorCode::ClusterTwoGroupsDegenerate)
            }
            Self::AmemiyaRequiresWithinVariation { columns } => {
                ErrorReport::new(ErrorCode::AmemiyaRequiresWithinVariation)
                    .param("columns", columns.clone())
            }
            Self::NonPositiveIdiosyncraticVariance { method, sigma2_eps } => {
                ErrorReport::new(ErrorCode::NonPositiveIdiosyncraticVariance)
                    .param("sigma2_eps", *sigma2_eps)
                    .param("variance_method", *method)
            }
            Self::VarianceComponentsMomentsSingular { method } => {
                ErrorReport::new(ErrorCode::VarianceComponentsMomentsSingular)
                    .param("variance_method", *method)
            }
        }
    }
}

impl PanelError {
    /// このエラーを、起きた計算の文脈（`stage`）つきで包む（[`PanelError::InStage`]）。
    pub(crate) fn in_stage(self, stage: Stage) -> Self {
        debug_assert!(
            self.report().code().spec().stage_allowed,
            "the code of {self:?} does not carry a stage"
        );
        Self::InStage {
            stage,
            source: Box::new(self),
        }
    }

    /// `Display`が使う`message`。
    fn catalog_message(&self) -> String {
        self.report().message()
    }
}

/// 時点列のラベルと、その時間順を表すコード。FE/REが時点の順序を使う
/// 計算（Driscoll-Kraay型HAC・2-way FEの`fixed_effects()`の基準時点とキー順）に渡す。
///
/// 時点の順序は**ラベルの文字列ではなく列の値の順序**で決める（整数・浮動小数点は数値順、
/// 日付・日時は時系列順、`Enum`はカテゴリ定義順）。文字列の辞書順では`1, 10, 11, 2, ...`の
/// ように並んで時点の対応がずれ、DKの標準誤差が黙って変わる。値の順序をどう取るかは
/// `engine_pybind`が列のdtypeから決め、コンストラクタで渡す。ラベル（`String`）は
/// 同一性の判定と`fixed_effects()`のキーに使う。
///
/// ユニークな時点は1つのコードに対応し、コードは0から時点の昇順に振る。行ごとのラベルの
/// `String`列は持たない（コードと、ユニークな時点のラベルだけ）。
///
/// 構築は、行ごとのラベルから作る`lexicographic`/`by_integer`/`by_float`と、呼び出し側が
/// 行ごとの`String`を作らずに振った仮コードから作る`*_factorized`の2系統。
#[derive(Debug, Clone)]
pub struct TimeKeys {
    /// 各行の時点コード（時間順、キー付き）。
    codes: GroupCodes,
}

impl PartialEq for TimeKeys {
    /// 時間順のコードと、ユニークな時点のラベル（時間順）の両方を比べる。`GroupCodes`の`==`は
    /// キーを見ないが、時点ではラベルが`fixed_effects()`のキーと`dk_time`の指定の同一性を決める
    /// ため、コードが同じでもラベルが違う時点列を等しいとみなさない。
    fn eq(&self, other: &Self) -> bool {
        self.codes == other.codes && self.periods() == other.periods()
    }
}

impl Eq for TimeKeys {}

impl TimeKeys {
    /// ラベルの辞書順を時間順とみなす（ISO 8601の日付・ゼロ埋めした年月等、辞書順が時間順と
    /// 一致するラベル向け）。テスト専用の簡便版（`FeInput::from_columns`/`ReInput::from_columns`）はこの順序を使う。
    pub fn lexicographic(ids: Vec<String>) -> Self {
        Self {
            codes: GroupCodes::from_labels(&ids),
        }
    }

    /// 各行の整数値`values`の昇順を時間順にする（整数列・日付・日時・`Enum`の物理順序）。
    ///
    /// # Errors
    /// `values`の長さが`ids`と一致しない場合は`PanelError::IdentifierDimensionMismatch`
    /// （`engine_pybind`が同じ列から作る限り起こり得ない契約違反に対する防御）。
    pub fn by_integer(ids: Vec<String>, values: &[i128]) -> Result<Self, PanelError> {
        check_key_lengths(ids.len(), values.len())?;
        let codes = GroupCodes::from_labels_ordered(&ids, |a, b| values[a].cmp(&values[b]));
        Ok(Self { codes })
    }

    /// 各行の浮動小数点値`values`の昇順（`f64::total_cmp`）を時間順にする。有限値であること
    /// （NaN・無限大は`engine_pybind`が列の抽出時に拒否する）。
    ///
    /// # Errors
    /// `values`の長さが`ids`と一致しない場合は`PanelError::IdentifierDimensionMismatch`。
    pub fn by_float(ids: Vec<String>, values: &[f64]) -> Result<Self, PanelError> {
        check_key_lengths(ids.len(), values.len())?;
        let codes = GroupCodes::from_labels_ordered(&ids, |a, b| values[a].total_cmp(&values[b]));
        Ok(Self { codes })
    }

    /// [`Self::lexicographic`]の仮コード版。各行の仮コード`codes`（値は`0..labels.len()`）と、
    /// 仮グループごとの時点ラベル`labels`から作る（`GroupCodes::from_factorized`と同じ
    /// 規則で、同じラベルの仮グループは1つにまとめる）。
    ///
    /// # Errors
    /// 仮コードが範囲外、または使われていない仮コードがある場合（`PanelError::Common`）。
    pub fn lexicographic_factorized(
        codes: Vec<usize>,
        labels: &[&str],
    ) -> Result<Self, PanelError> {
        Ok(Self {
            codes: GroupCodes::from_factorized(codes, labels)?,
        })
    }

    /// [`Self::by_integer`]の仮コード版。`group_values[g]`は仮グループ`g`の整数値。
    ///
    /// # Errors
    /// `group_values`の長さが`labels`と一致しない場合は`PanelError::IdentifierDimensionMismatch`。
    /// そのほか`lexicographic_factorized`と同じ。
    pub fn by_integer_factorized(
        codes: Vec<usize>,
        labels: &[&str],
        group_values: &[i128],
    ) -> Result<Self, PanelError> {
        check_key_lengths(labels.len(), group_values.len())?;
        let codes = GroupCodes::from_factorized_ordered(codes, labels, |a, b| {
            group_values[a].cmp(&group_values[b])
        })?;
        Ok(Self { codes })
    }

    /// [`Self::by_float`]の仮コード版。`group_values[g]`は仮グループ`g`の浮動小数点値。
    ///
    /// # Errors
    /// `by_integer_factorized`と同じ。
    pub fn by_float_factorized(
        codes: Vec<usize>,
        labels: &[&str],
        group_values: &[f64],
    ) -> Result<Self, PanelError> {
        check_key_lengths(labels.len(), group_values.len())?;
        let codes = GroupCodes::from_factorized_ordered(codes, labels, |a, b| {
            group_values[a].total_cmp(&group_values[b])
        })?;
        Ok(Self { codes })
    }

    /// 時点ラベルのコード（時間順）。
    pub(crate) fn codes(&self) -> &GroupCodes {
        &self.codes
    }

    /// 行数`n`。
    pub fn len(&self) -> usize {
        self.codes.nobs()
    }

    /// 行数が0か。
    pub fn is_empty(&self) -> bool {
        self.codes.nobs() == 0
    }

    /// ユニークな時点のラベルを時間順に並べたもの。
    pub fn periods(&self) -> &[String] {
        self.codes.keys()
    }
}

/// `TimeKeys`のコンストラクタが、ラベルと順序づけの値の長さを検証する。
fn check_key_lengths(labels: usize, values: usize) -> Result<(), PanelError> {
    if labels != values {
        return Err(PanelError::IdentifierDimensionMismatch {
            dimension: PanelDimension::Time,
            y_rows: labels,
            other_rows: values,
        });
    }
    Ok(())
}

/// `FeInput::from_group_codes`/`ReInput::from_group_codes`の入力検証（FE/REで共通）。
/// 検査の順序は、各`x_columns`の行数（`y`との不一致は`CommonError::DimensionMismatch`）、
/// `entity`の長さ、`time`の長さ（以上2つは`PanelError::IdentifierDimensionMismatch`）、
/// `entity`がキーを持つか（`require_entity_keys`）。
///
/// # パニックについて
/// `x_names.len() != x_columns.len()`の場合は`debug_assert!`でパニックする（呼び出し側
/// `engine_pybind`の実装バグでしか起こり得ない内部契約であり、実データに起因する
/// `ValidationError`とは性質が異なる。`OlsInput::from_columns`と同じ理由）。
pub(crate) fn validate_panel_input(
    y: &[f64],
    x_columns: &[Vec<f64>],
    x_names: &[String],
    entity: &GroupCodes,
    time: Option<&TimeKeys>,
) -> Result<(), PanelError> {
    debug_assert_eq!(
        x_columns.len(),
        x_names.len(),
        "x_columns and x_names must have the same length"
    );

    for col in x_columns {
        if col.len() != y.len() {
            return Err(CommonError::DimensionMismatch {
                y_rows: y.len(),
                x_rows: col.len(),
            }
            .into());
        }
    }

    if entity.nobs() != y.len() {
        return Err(PanelError::IdentifierDimensionMismatch {
            dimension: PanelDimension::Entity,
            y_rows: y.len(),
            other_rows: entity.nobs(),
        });
    }

    if let Some(time) = time
        && time.len() != y.len()
    {
        return Err(PanelError::IdentifierDimensionMismatch {
            dimension: PanelDimension::Time,
            y_rows: y.len(),
            other_rows: time.len(),
        });
    }
    require_entity_keys(entity)
}

/// パネル入力のentityのコードがキー（`keys()`）を持つことを確かめる。固定効果のキーに使うため、
/// キー無しのコード（`GroupCodes::from_*_without_keys`）は受け付けない。
fn require_entity_keys(entity: &GroupCodes) -> Result<(), PanelError> {
    if !entity.has_keys() {
        return Err(CommonError::internal(
            "entity codes must carry keys (build them with GroupCodes::from_labels or \
             GroupCodes::from_factorized)",
        )
        .into());
    }
    Ok(())
}

/// `(X̃'X̃)⁻¹`を求める（`X̃`はFE/REそれぞれの変換後の設計行列）。HC1〜HC3・Clusterいずれの
/// 計算でも共通して必要になる。`shared::covariance::xtx_inverse`と同じ発想だが、`OlsEstimator`が
/// 保持する`cov_params`はprivateで再利用できないため独立に計算し直す。
///
/// `X̃'X̃`が対称正定値であることは、`OlsEstimator::fit`が同じ`x`で既に成功している
/// （＝特異ではないと確認済み）ことから理論上保証されるが、`OlsEstimator`と同じく
/// 浮動小数点演算の境界的なケースに備えて`Result`化する。
pub(crate) fn xtx_inverse(x: &Mat<f64>) -> Result<Mat<f64>, PanelError> {
    crate::shared::covariance::xtx_inverse(x).map_err(|_| {
        CommonError::internal(
            "the Gram matrix of the transformed design matrix could not be inverted",
        )
        .into()
    })
}

/// classical: `σ̂² (X̃'X̃)⁻¹`（`σ̂² = SSR/df_resid`）。
pub(crate) fn panel_classical_cov_params(
    xtx_inv: &Mat<f64>,
    ssr: f64,
    df_resid: usize,
    k: usize,
) -> Mat<f64> {
    let sigma2 = ssr / (df_resid as f64);
    Mat::from_fn(k, k, |i, j| sigma2 * (*xtx_inv.get(i, j)))
}

/// `panel_hc_cov_params`が使うHCの種類（`shared::covariance::HcVariant`と同型だがHC0を含まない、
/// FE/REともにHC0はスコープ外、`fe.rs`モジュールdoc「`cov_type`対応」参照）。HC2/HC3は各行の
/// レバレッジ`h`（FEは`leverage_full`、REは`shared::covariance::leverages`の素のレバレッジ）を
/// バリアントに持つため、呼び出し側が`variant`と`h`の組み合わせを誤ることはない。
pub(crate) enum PanelHcVariant<'a> {
    Hc1,
    Hc2(&'a [f64]),
    Hc3(&'a [f64]),
}

/// HC1〜HC3の係数分散共分散行列（k×k）。`shared::covariance::hc_cov_params`と同型の構造だが、
/// 小標本補正がFE/RE用に異なる（`fe.rs`モジュールdoc「`cov_type`対応」参照）:
/// - HC1: `w_i = n/df_resid`
/// - HC2/HC3: `w_i`はバリアントが持つレバレッジ`h`ベース
pub(crate) fn panel_hc_cov_params(
    x: &Mat<f64>,
    residuals: &[f64],
    xtx_inv: &Mat<f64>,
    df_resid: usize,
    variant: PanelHcVariant<'_>,
) -> Mat<f64> {
    let n = x.nrows();
    let k = x.ncols();
    let hc1_correction = (n as f64 / df_resid as f64).sqrt();

    // 行ごとのスケールは列に依存しないため、行列の組み立て（n×kの二重ループ）の外で一度だけ求める。
    let row_scales: Vec<f64> = (0..n)
        .map(|i| {
            let resid = residuals[i];
            match &variant {
                PanelHcVariant::Hc1 => resid * hc1_correction,
                PanelHcVariant::Hc2(h) => resid / (1.0 - h[i]).sqrt(),
                PanelHcVariant::Hc3(h) => resid / (1.0 - h[i]),
            }
        })
        .collect();
    let x_scaled = Mat::from_fn(n, k, |i, j| row_scales[i] * (*x.get(i, j)));

    let psi_hat = x_scaled.transpose() * &x_scaled;
    xtx_inv * &psi_hat * xtx_inv
}

/// クラスターロバスト係数分散共分散行列（k×k）。`shared::covariance::cluster_cov_params`と同じ
/// Stata流`(G/(G-1))×((n-1)/(n-K))`小標本補正を使う（fixest（R）・Stataの`xtreg`/
/// `reghdfe`との数値一致のため、linearmodels方式`n/(n-extra_df-k)`から
/// 変更した。`fe.rs`モジュールdoc「`cov_type`対応」参照）。`G`はこの関数が
/// `groups`から数える（`shared::covariance::cluster_cov_params`と同じ）。`K`（`(n-1)/(n-K)`の分母）は
/// 呼び出し側が決める`k_correction`引数で渡す——FEは固定効果ダミーとクラスター変数の
/// ネスト関係で決まるfixest固有のK計算（`fe_cluster_k_correction`）、REは
/// 単純に`df_model`（FEのような固定効果ダミーのネスト補正が不要、モジュールdoc参照）。
pub(crate) fn panel_cluster_cov_params(
    x: &Mat<f64>,
    residuals: &[f64],
    xtx_inv: &Mat<f64>,
    n: usize,
    k: usize,
    groups: &GroupCodes,
    k_correction: usize,
) -> Mat<f64> {
    let (s_hat, n_groups) = cluster_meat(groups, k, |i, a| residuals[i] * (*x.get(i, a)));
    let correction = cluster_correction(n_groups, n, k_correction);
    let cov_uncorrected = xtx_inv * &s_hat * xtx_inv;
    Mat::from_fn(k, k, |i, j| correction * (*cov_uncorrected.get(i, j)))
}

/// `FeCovType::Dk`/`ReCovType::Dk`の`bandwidth`（`Option<i64>`）を実際に使う
/// バンド幅（`usize`）に解決する。
///
/// `Some(bw)`の場合は`0 <= bw < t`を検証してそのまま使う（`t`はユニークな時点数）。`None`の
/// 場合は`linearmodels`の`DriscollKraay`と同じ経験則`floor(4*(t/100)^(2/9))`で自動計算する
/// （`fe.rs`モジュールdoc「Driscoll-Kraay型パネルHAC対応」参照。式は
/// `shared::covariance::default_hac_lags`でOLS・IVのHACと共有するが、引数が観測数`n`では
/// なく時点数`t`になる点が異なる）。
///
/// `t < 2`は`bandwidth`の値によらず`PanelError::InsufficientDkPeriods`で拒否する
/// （fixest型の`G/(G-1)`相当補正を導入したことに伴う前提条件。同エラーの
/// docコメント参照）。この関数がDK計算の全呼び出し経路（`fe.rs`/`re.rs`/
/// `re_hausman_test`）の共通の入口になっているため、ここ一箇所で検証すれば足りる。
pub(crate) fn resolve_dk_bandwidth(bandwidth: Option<i64>, t: usize) -> Result<usize, PanelError> {
    if t < 2 {
        return Err(PanelError::InsufficientDkPeriods { t_periods: t });
    }
    match bandwidth {
        Some(bw) => {
            if bw < 0 || (bw as usize) >= t {
                return Err(PanelError::InvalidDkBandwidth { bandwidth: bw, t });
            }
            Ok(bw as usize)
        }
        None => Ok(default_hac_lags(t)),
    }
}

/// DKで同時Wald検定の対象となる係数の数`q`が、時点数`t_periods`で支えられることを
/// 検証する（`t_periods > q`）。理由は`PanelError::InsufficientDkPeriodsForInference`の
/// docコメント参照（`validate_cluster_count_covers_slopes`のDK版）。
///
/// `t_periods > q`でも係数間の悪条件で部分行列が数値的にほぼ特異になるケースは
/// 事前判定できないため、`wald_f_test`内の`ensure_well_conditioned_symmetric_matrix`が
/// backstopとして残る。`resolve_dk_bandwidth`（`t_periods >= 2`を保証）の後に呼ぶ前提で、
/// そのとき`q == 0`（検定対象なし）は常に`Ok`。
pub(crate) fn validate_dk_periods_cover_tested_coefficients(
    t_periods: usize,
    q: usize,
) -> Result<(), PanelError> {
    if t_periods <= q {
        return Err(PanelError::InsufficientDkPeriodsForInference { t_periods, q });
    }
    Ok(())
}

/// Driscoll-Kraay型パネルHAC共分散行列（k×k）。
///
/// `Ŝ = Σ_t ξ_t ξ_t' + Σ_{l=1}^{bandwidth} w_l (ξ_t ξ_{t-l}' + ξ_{t-l} ξ_t')`
/// （Bartlett重み`w_l = 1 - l/(bandwidth+1)`、`fe.rs`モジュールdoc参照）をまず求め、
/// 最後に`(t_periods/(t_periods-1)) × ((n-1)/(n-k_correction)) × (X̃'X̃)⁻¹ Ŝ (X̃'X̃)⁻¹`に
/// スケールする（fixestの`vcov="DK"`との数値一致のため、
/// linearmodels方式`n/df_resid`から変更した。fixestは時点数`t_periods`を
/// クラスター数`G`相当として扱い、`K.fixef="full"`が既定（クラスター変数が無いため
/// ネスト判定自体が発生しない）——`k_correction`は呼び出し側がFE/REそれぞれの
/// `df_model`をそのまま渡す。`panel_cluster_cov_params`の`G/(G-1)×(n-1)/(n-K)`と
/// 同型の式に、`G`を`t_periods`（`time`のユニークな時点数）に置き換えたものと理解できる）。
///
/// `time`のコード（`GroupCodes`、コード順＝時系列順、`TimeKeys`のdoc参照）で集計して
/// `ξ_t`（時点`t`でのクロスセクション和）を求める。
///
/// **`bandwidth < t_periods`が呼び出し元の`resolve_dk_bandwidth`から保証される**
/// （詳細は`fe.rs`モジュールdoc「Driscoll-Kraay型パネルHAC対応」参照）。上限の
/// `bandwidth == t_periods - 1`でも最終ラグ項`l = bandwidth`を含める標準のBartlettカーネル
/// で、この境界ではfixestと一致しない（`docs/spec/fe-spec.md`3.3節）。
///
/// ラグごとの`k×k`行列積（`xi_top.transpose() * xi_bot`等）は呼び出し元の`fit()`冒頭の
/// `ensure_serial()`が固定したグローバル`Par::Seq`に依存している（OLSの`hac_cov_params`が
/// `matmul(..., Par::Seq)`を明示するのと異なり、本関数は演算子オーバーロードを使うため
/// グローバル設定頼み。`panel_hc_cov_params`/`panel_cluster_cov_params`と同じ流儀）。
pub(crate) fn panel_driscoll_kraay_cov_params(
    x: &Mat<f64>,
    residuals: &[f64],
    xtx_inv: &Mat<f64>,
    time: &GroupCodes,
    k_correction: usize,
    bandwidth: usize,
) -> Mat<f64> {
    let n = x.nrows();
    let k = x.ncols();
    let t_periods = time.n_groups();
    let time_indices = time.group_indices();

    let mut xi = Mat::<f64>::zeros(t_periods, k);
    for (row, indices) in time_indices.iter().enumerate() {
        for &i in indices {
            let e = residuals[i];
            for col in 0..k {
                *xi.get_mut(row, col) += e * (*x.get(i, col));
            }
        }
    }

    // l=0項: Ŝ₀ = ξ'ξ（クラスターロバストのΨ̂と同形、時点をグループとみなした版）
    let mut s_hat = xi.transpose() * &xi;
    // l=1..=bandwidth項: w_l * (Ŝ_l + Ŝ_l')
    for l in 1..=bandwidth {
        let weight = 1.0 - (l as f64) / ((bandwidth + 1) as f64);
        let xi_top = xi.as_ref().subrows(l, t_periods - l);
        let xi_bot = xi.as_ref().subrows(0, t_periods - l);
        let s_l = xi_top.transpose() * xi_bot;
        for a in 0..k {
            for b in 0..k {
                *s_hat.get_mut(a, b) += weight * (*s_l.get(a, b) + *s_l.get(b, a));
            }
        }
    }

    let scale = (t_periods as f64 / (t_periods as f64 - 1.0))
        * ((n as f64 - 1.0) / ((n - k_correction) as f64));
    let cov_uncorrected = xtx_inv * &s_hat * xtx_inv;
    Mat::from_fn(k, k, |i, j| scale * (*cov_uncorrected.get(i, j)))
}

/// θでパラメータ化した準偏差変換を、設計行列の1つの列（`y`または`x`の1列）に適用し、
/// 変換後の新しい`Vec<f64>`を返す。
///
/// 各行`i`を次のように変換する:
///
/// ```text
/// col_transformed[i] = col[i] - θ_{e(i)} · mean_{e(i)}(col)
/// ```
///
/// ここで`e(i)`は行`i`が属するエンティティ、`mean_{e}(col)`はエンティティ`e`に属する
/// 行の`col`の単純平均（`ȳ_i.`）。
///
/// - **FE（within変換）**: 全エンティティに`θ_i = 1.0`を渡す → `col[i] - ȳ_i.`
///   （`docs/spec/re-spec.md`3.2節: FEはこの関数の`θ_i = 1`の特殊ケース）。
/// - **RE（準偏差変換）**: `θ_i = 1 - sqrt(σ_ε² / (T_i·σ_u² + σ_ε²))`（同3.2節）を渡す。
///
/// FE/REの`fit()`は`y`と`x`の各列にこの関数をループ適用し、変換後の列を
/// `OlsEstimator::fit`へ渡す（WLSがsqrt(w)変換したデータをOLSへ委譲するのと同型の
/// パターン、`panel-common.md`4.3節・`re-spec.md`3.2節）。列ごとに独立な変換のため、
/// 列単位の関数として実装し
/// 呼び出し側でループする（`y`/`x`をまとめて受けるより単体テストが単純）。
///
/// # 引数
/// - `col`: 変換対象の列（長さ`n`、行はパネルの観測順）。
/// - `entity`: 各行のエンティティ（`FeInput`/`ReInput`が構築時に一度だけ作った整数コード、
///   `GroupCodes`のdocコメント参照）。2-way FEの時点方向の変換では時点のコードを渡す。
/// - `theta`: `θ_i`をコード順（`entity.keys()`の並び）に並べたもの（長さ`n_groups`）。
///
/// # 前提（呼び出し側の契約、`engine`内部でのみ使用）
/// - `entity.nobs() == col.len()`。`FeInput`/`ReInput`が`y`・`x`と同じ長さの`entity`から
///   コードを作るため、違反は内部実装バグでしか起こり得ない。
/// - `theta.len() == entity.n_groups()`。RE/FEの`fit()`は同じ`entity`のコードから
///   `theta`を組み立てる。
/// - `col`は欠損値・非有限値を含まない（`engine`は常にクリーンな値を受け取る前提）。
///
/// 契約違反時は`assert!`/`expect`でpanicする（`Result`は返さない）。ユーザー入力起因の
/// エラーではなく`engine_pybind`〜`engine`間の内部契約違反のため、`validate_cluster_groups`
/// と同じ扱い。
///
/// グループ平均（`ȳ_i.`）は返さない。`fixed_effects()`（`fe-spec.md`3.5節）の`α_i`復元は
/// この関数を拡張せず`fe.rs`側で`FeInput`の元データから独立に再計算する形で実装済み
/// （`engine/src/panel/CLAUDE.md`参照）。σ_ε²再利用（`re-spec.md`3.2節、RE実装）で平均の保持が
/// 必要になった場合は、その時点で改めて検討する。
///
/// # Panics
/// - `entity.nobs() != col.len()`
/// - `theta.len() != entity.n_groups()`
pub(crate) fn quasi_demean_column(col: &[f64], entity: &GroupCodes, theta: &[f64]) -> Vec<f64> {
    assert_eq!(
        entity.nobs(),
        col.len(),
        "entity length must match column length (FeInput/ReInput contract)"
    );
    assert_eq!(
        theta.len(),
        entity.n_groups(),
        "theta must have one value per entity group"
    );

    // エンティティごとの和を観測順に積む（エンティティ「間」をまたぐ加算は無いため、
    // コードの振り方によらずビット単位で決定的。旧実装の`String`キーの`HashMap`集約と
    // 同じ加算順・同じ演算で、結果もビット単位で同じ）。
    let mut sums = vec![0.0_f64; entity.n_groups()];
    for (value, &c) in col.iter().zip(entity.codes()) {
        sums[c] += *value;
    }

    // エンティティ単位で `θ_i · ȳ_i.`（各行から引く量）を先に求めておく。
    let shift: Vec<f64> = sums
        .iter()
        .zip(entity.counts())
        .zip(theta)
        .map(|((sum, &count), &theta_i)| theta_i * (sum / count as f64))
        .collect();

    col.iter()
        .zip(entity.codes())
        .map(|(value, &c)| value - shift[c])
        .collect()
}

/// 2-way REのGLS変換の3つの重み（`re-spec.md`5章）。
///
/// バランスパネルでは全観測で共通の値で、変換は`y - θ₁ȳᵢ. - θ₂ȳ.ₜ + θ₃ȳ..`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TwoWayTheta {
    /// entity平均`ȳᵢ.`にかかる重み`θ₁`。
    pub(crate) entity: f64,
    /// time平均`ȳ.ₜ`にかかる重み`θ₂`。
    pub(crate) time: f64,
    /// 総平均`ȳ..`にかかる重み`θ₃`（`θ₁`と`θ₂`の積ではなく、`θ₁ + θ₂ - 1 + σ_ε/σ₄`）。
    pub(crate) total: f64,
}

/// 2-way RE（バランスパネル）のGLS変換`col - θ₁ȳᵢ. - θ₂ȳ.ₜ + θ₃ȳ..`を1列に適用する。
///
/// `quasi_demean_column`は1つのグルーピングしか扱えず、FEの2-wayのように「entityで
/// quasi-demean→その結果をtimeでquasi-demean」の2段階適用（θ₁・θ₂だけを使う）では
/// `θ₃`が合わずGLS変換にならない（`θ₃ ≠ θ₁θ₂`）ため、3つの平均を同時に引く専用の関数を
/// 用意した。θ=1・1・1なら2-way within変換（二重デミーニング）になる。
///
/// 平均はバランスパネルを前提に`GroupCodes::counts()`で割る（総平均は`n`で割る）。
/// バランスパネルであることの検証は呼び出し側の責務（`validate_balanced_panel`）。
///
/// # Panics
/// `entity`・`time`の観測数が`col.len()`と一致しない場合（`quasi_demean_column`と同じ内部契約）。
pub(crate) fn quasi_demean_two_way(
    col: &[f64],
    entity: &GroupCodes,
    time: &GroupCodes,
    theta: TwoWayTheta,
) -> Vec<f64> {
    assert_eq!(
        entity.nobs(),
        col.len(),
        "entity length must match column length (ReInput contract)"
    );
    assert_eq!(
        time.nobs(),
        col.len(),
        "time length must match column length (ReInput contract)"
    );

    let mut entity_sums = vec![0.0_f64; entity.n_groups()];
    let mut time_sums = vec![0.0_f64; time.n_groups()];
    let mut total = 0.0_f64;
    for ((value, &e), &t) in col.iter().zip(entity.codes()).zip(time.codes()) {
        entity_sums[e] += *value;
        time_sums[t] += *value;
        total += *value;
    }
    let grand_mean = total / col.len() as f64;
    let entity_shift: Vec<f64> = entity_sums
        .iter()
        .zip(entity.counts())
        .map(|(sum, &count)| theta.entity * (sum / count as f64))
        .collect();
    let time_shift: Vec<f64> = time_sums
        .iter()
        .zip(time.counts())
        .map(|(sum, &count)| theta.time * (sum / count as f64))
        .collect();
    let grand_shift = theta.total * grand_mean;

    col.iter()
        .zip(entity.codes())
        .zip(time.codes())
        .map(|((value, &e), &t)| value - entity_shift[e] - time_shift[t] + grand_shift)
        .collect()
}

/// LSDV相当のフルレバレッジ`h_ii_full`（HC2/HC3用、モジュールdoc「`cov_type`対応」の
/// 導出参照）。分割回帰（Frisch-Waugh-Lovell）のレバレッジ分解則により、固定効果ダミーを
/// 明示的に含めた設計行列でのレバレッジは、ダミーのみの回帰のレバレッジ（`1/T_i`、
/// 2-wayはさらに`1/N_t - 1/n`）とwithin変換後のレバレッジ（`h_within`）の和になる
/// （fixestの`vcov="HC2"`/`"HC3"`と数値一致を1-way・2-way双方で確認済み）。
///
/// グループサイズ`T_i`/`N_t`はコードの`counts()`から引く。
fn leverage_full(
    h_within: &[f64],
    entity: &GroupCodes,
    time: Option<&GroupCodes>,
    n: usize,
) -> Vec<f64> {
    let entity_codes = entity.codes();
    let entity_sizes = entity.counts();
    match time {
        None => (0..n)
            .map(|i| 1.0 / (entity_sizes[entity_codes[i]] as f64) + h_within[i])
            .collect(),
        Some(time) => {
            let time_codes = time.codes();
            let time_sizes = time.counts();
            (0..n)
                .map(|i| {
                    1.0 / (entity_sizes[entity_codes[i]] as f64)
                        + 1.0 / (time_sizes[time_codes[i]] as f64)
                        - 1.0 / (n as f64)
                        + h_within[i]
                })
                .collect()
        }
    }
}

/// FE次元（`entity`または`time`）の各値が`cluster`上でちょうど1つの値にしか対応しないか
/// （＝`cluster`がその次元と同じか、それを包含するより粗い分割か）を判定する
/// （fixestの`ssc_compute_K`が言う「FEがクラスター変数にネストしている」の定義。
/// `fe_cluster_k_correction`のdocコメント参照）。元は`entity_nested_within_cluster`
/// という1-way専用の名前だったが、2-way FEでtime次元にも同じ判定を適用する必要が
/// あるため汎用化した（ロジック自体は無変更）。
fn fixef_dimension_nested_within_cluster(dim: &GroupCodes, cluster: &GroupCodes) -> bool {
    // FE次元の各水準が最初に対応したクラスターコード（未出現は`usize::MAX`）。
    let mut mapping = vec![usize::MAX; dim.n_groups()];
    for (&d, &c) in dim.codes().iter().zip(cluster.codes()) {
        if mapping[d] == usize::MAX {
            mapping[d] = c;
        } else if mapping[d] != c {
            return false;
        }
    }
    true
}

/// fixestの`ssc_compute_K`（既定`K.fixef="nonnested"`・`K.exact=FALSE`）を移植した、
/// cluster小標本補正の`K`（`panel_cluster_cov_params`の`(n-1)/(n-K)`の`K`）計算。
///
/// fixest 0.14.2のRソース（`Rscript -e 'cat(deparse(fixest:::ssc_compute_K), sep="\n")'`）と
/// 実地数値実験（devcontainer内のfixest 0.14.2、実装時）で確定した式:
///
/// FE次元（1-wayは`entity`のみ、2-wayは`entity`+`time`）それぞれについて、その次元が
/// クラスター変数に「ネスト」している（＝各水準がクラスターの単一の値にしか対応しない、
/// `fixef_dimension_nested_within_cluster`で判定）かを調べ、ネストした次元の生の水準数
/// （`n_entities`/`n_periods`、`df_model`に含まれる冗長性補正前の値）の合計を
/// `nested_size_sum`、ネストした次元の数を`count_nested`、FE次元の総数を`m`（1か2）とする:
///
/// - `count_nested == 0`（どの次元もネストしていない、Stataの`xtreg,fe`型）:
///   `K = df_model`（固定効果ダミーをフルカウント）
/// - `count_nested == m`（全次元がネスト。1-way・cluster=entityがこの既定ケース）:
///   `K = df_model - nested_size_sum + m`
/// - それ以外（2-way FEで一部の次元だけネスト）:
///   `K = df_model - (nested_size_sum - count_nested)`
///
/// 最後にfixest自身の安全弁`K = max(K, k + 1)`（`k`は傾き係数の数。Rソースの
/// `K = max(K, length(object$coefficients) + 1)`）を適用する——fixestのRソースを
/// 忠実に移植する方針（1章）のため実装しているが、**このプロジェクトの`FeEstimator::fit`が
/// 保証する前提（`n_entities>=1`・2-wayなら`n_periods>=1`、および`df_model = k + neffects`の
/// 定義）の下では、3分岐のどのケースでもこのフロアは実質的にno-op（`raw_k`が既に
/// `k+1`以上）であることを代数的に確認済み**（rust-reviewerの指摘を受けて検算、
/// ）:
/// - `count_nested == m`（全次元ネスト）: `nested_size_sum`はネストした全次元の生サイズの
///   合計で、`df_model`の`neffects`部分もちょうど同じ次元から`Σsize - (m-1)`として
///   構成される（`neffects`の定義、モジュールdoc「自由度調整」参照）ため、
///   `K = df_model - nested_size_sum + m = k + (Σsize - (m-1)) - Σsize + m = k + 1`が
///   **恒等的に**成り立つ（1-way・2-way両次元ネストのどちらでも）。フロアと厳密に一致する
///   だけで、フロアが値を持ち上げる場面ではない。
/// - 部分ネスト・ネストなし（2-way限定）: ネストしていない側の次元の生サイズが
///   `df_model`にそのまま残るため、`K`はその生サイズの分だけ`k+1`を上回る
///   （`n_entities`/`n_periods`はいずれもパネルとして成立する以上`>=1`、実務上は
///   ほぼ常に`>=2`）。
///
/// 上記のため、`fit()`経由の統合テストではこのフロアの分岐（`raw_k < k+1`になる入力）を
/// 実際には構成できない。フロア自体の検証は`fe_cluster_k_correction`を直接呼ぶ
/// ユニットテスト（`fe_cluster_k_correction_floor_is_a_no_op_under_realistic_inputs`）で、
/// 上記の恒等式そのものを回帰ガードする（`.claude/rules/rust-style.md`「テスト」の
/// 「理論上到達不能な経路は`Result`化しdocで理由を明記すればカバレッジ対象外でよい」
/// 方針と同型——ここでは`Result`ではなく`usize`の恒等式だが、同じ考え方で
/// 「なぜ到達しないか」を明記する）。
///
/// 具体例（実地検証済み）: 1-way FE（`n_entities=20`）・`cluster=entity`（既定）・
/// `k=2`なら`df_model=22`・`nested_size_sum=20`・`count_nested=m=1`で`K=22-20+1=3`
/// （`k+1=3`と一致、フロアはno-op）。
///
/// `two_way_time`は2-wayの時点コード（1-wayは`None`）。FE次元は1-wayがentityのみ、
/// 2-wayがentityとtime（水準数は`time.n_groups()`）。
fn fe_cluster_k_correction(
    entity: &GroupCodes,
    two_way_time: Option<&GroupCodes>,
    n_entities: usize,
    df_model: usize,
    k: usize,
    cluster: &GroupCodes,
) -> usize {
    let mut dims = vec![(
        fixef_dimension_nested_within_cluster(entity, cluster),
        n_entities,
    )];
    if let Some(time) = two_way_time {
        dims.push((
            fixef_dimension_nested_within_cluster(time, cluster),
            time.n_groups(),
        ));
    }
    let m = dims.len();
    let count_nested = dims.iter().filter(|(nested, _)| *nested).count();
    let nested_size_sum: usize = dims
        .iter()
        .filter(|(nested, _)| *nested)
        .map(|(_, size)| size)
        .sum();

    let raw_k: i64 = if count_nested == 0 {
        df_model as i64
    } else if count_nested == m {
        df_model as i64 - nested_size_sum as i64 + m as i64
    } else {
        df_model as i64 - (nested_size_sum as i64 - count_nested as i64)
    };
    raw_k.max((k + 1) as i64) as usize
}

/// `levels`の全水準がちょうど2観測を持ち、その2観測の`groups`ラベルが異なるか
/// （`groups`のユニーク数が2の前提で呼ぶ。このとき各水準が2グループに1観測ずつ）。
/// 1水準でも3観測以上・同じグループに2観測・1観測があれば`false`。
///
/// `levels.nobs() == groups.nobs()`は呼び出し側の契約（`FeInput`のコードと、
/// `validate_cluster_groups`済みのクラスター列のコード、または`fit()`のDkアームで
/// 長さを検証済みのDK時点列のコード）。
fn every_level_splits_once_across_two_groups(levels: &GroupCodes, groups: &GroupCodes) -> bool {
    debug_assert_eq!(
        levels.nobs(),
        groups.nobs(),
        "levels and groups must have the same length (caller contract)"
    );
    // 水準ごとの（最初の観測のグループコード、観測数）。`levels`のコードは全水準が
    // 少なくとも1回現れるため、最後の`all`は未出現の水準を考慮しなくてよい。
    let mut first_group = vec![usize::MAX; levels.n_groups()];
    let mut count = vec![0_usize; levels.n_groups()];
    for (&level, &group) in levels.codes().iter().zip(groups.codes()) {
        match count[level] {
            0 => {
                first_group[level] = group;
                count[level] = 1;
            }
            seen if seen >= 2 || first_group[level] == group => return false,
            _ => count[level] += 1,
        }
    }
    count.iter().all(|&c| c == 2)
}

/// ユニーク数2の`groups`（Clusterのクラスター列・Dkの時点列）について、within変換後の
/// グループスコアが恒等的にゼロになるか（`PanelError::DegenerateDkTwoPeriods`/
/// `DegenerateClusterTwoGroups`の判定）。
///
/// within変換は吸収した各FE次元の水準内で和をゼロにする（1-wayはentity、2-wayは
/// entityとtimeの両方）。ある次元の全水準がちょうど2観測で2グループに1つずつ分かれると、
/// 各水準の2観測で`x̃`・`ẽ`が符号反転し、両グループに同じ寄与が入ってスコアが等しくなる。
/// 正規方程式でスコアの和はゼロのため、両方ゼロになる。entity方向（2時点のパネルを
/// timeでクラスタリング等）だけでなく、2-wayではtime方向（エンティティ2つのパネルを
/// entityでクラスタリング——Clusterの既定——等）も同じ構造になる。
///
/// `two_way_time`は2-wayの時点コード（1-wayは`None`、`two_way_time`関数参照）。
fn two_group_split_is_degenerate(
    entity: &GroupCodes,
    two_way_time: Option<&GroupCodes>,
    groups: &GroupCodes,
) -> bool {
    every_level_splits_once_across_two_groups(entity, groups)
        || two_way_time.is_some_and(|time| every_level_splits_once_across_two_groups(time, groups))
}

/// within変換（entity、2-wayはさらにtimeでθ=1のquasi-demean）。列スライスに適用できるため、
/// `FeInput`の`y`・`x`だけでなく、操作変数など他の列にも同じ変換をかけられる。
///
/// 2-wayは閉形式の二重デミーニングと数学的に等価な「entityでquasi-demean→その結果をtimeで
/// quasi-demean」の2段階適用で計算する（`fe.rs`モジュールdoc参照）。バランスパネルであることの
/// 検証はしない（呼び出し側が`validate_balanced_panel`で済ませる前提。不均衡だと二重デミーニングと
/// 一致しない）。θ=1のベクトルは変換の構築時に一度だけ作り、列ごとに使い回す。
pub(crate) struct WithinTransform<'a> {
    entity: &'a GroupCodes,
    entity_theta: Vec<f64>,
    /// 2-wayのtimeのコードとθ=1のベクトル（1-wayは`None`）。
    time: Option<(&'a GroupCodes, Vec<f64>)>,
}

impl<'a> WithinTransform<'a> {
    /// entityのみのwithin変換。
    pub(crate) fn one_way(entity: &'a GroupCodes) -> Self {
        Self {
            entity,
            entity_theta: vec![1.0; entity.n_groups()],
            time: None,
        }
    }

    /// entity + timeのwithin変換。
    pub(crate) fn two_way(entity: &'a GroupCodes, time: &'a GroupCodes) -> Self {
        Self {
            entity,
            entity_theta: vec![1.0; entity.n_groups()],
            time: Some((time, vec![1.0; time.n_groups()])),
        }
    }

    /// 1列にwithin変換をかける。
    pub(crate) fn apply(&self, col: &[f64]) -> Vec<f64> {
        let by_entity = quasi_demean_column(col, self.entity, &self.entity_theta);
        match &self.time {
            None => by_entity,
            Some((time, time_theta)) => quasi_demean_column(&by_entity, time, time_theta),
        }
    }
}

/// `entity`、2-wayなら`two_way_time`にも観測数1のグループがないことを検証する
/// （`fe-spec.md`1章）。entityを先に検査し、両方に該当するsingletonがあればentity側を報告する。
/// 列名は`names`から引く。
///
/// # Errors
/// 観測数1のグループが見つかった場合は`PanelError::SingletonGroup`（該当する`dimension`を含む）。
pub(crate) fn ensure_no_singleton_groups(
    names: &EffectNames,
    entity: &GroupCodes,
    two_way_time: Option<&GroupCodes>,
) -> Result<(), PanelError> {
    reject_singleton_group(
        PanelDimension::Entity,
        names.get(PanelDimension::Entity),
        entity,
    )?;
    if let Some(time) = two_way_time {
        reject_singleton_group(PanelDimension::Time, names.get(PanelDimension::Time), time)?;
    }
    Ok(())
}

/// within変換後の各列に分散ゼロの列がないことを検証する（`fe-spec.md`1章）。`names`・
/// `originals`・`transformed`は同じ列順・同じ列数で、列ごとに同じ長さを持つ（`engine`内の
/// 内部契約であり、ユーザー入力起因ではない。リリースビルドで`zip`が黙って短い方へ切り詰める
/// 事態を避けるため`debug_assert_eq!`ではなく`assert_eq!`で守る）。
///
/// # Errors
/// 分散ゼロの列が見つかった場合は`PanelError::ZeroVarianceAfterDemeaning`（該当列名を含む）。
/// 複数列が該当する場合は`names`の順で最初に見つかった列のみを報告する。
pub(crate) fn ensure_no_zero_variance_columns(
    names: &[String],
    originals: &[Vec<f64>],
    transformed: &[Vec<f64>],
) -> Result<(), PanelError> {
    assert_eq!(
        names.len(),
        originals.len(),
        "names must have one entry per original column"
    );
    assert_eq!(
        originals.len(),
        transformed.len(),
        "transformed must have the same number of columns as the original columns"
    );
    for ((name, original), transformed) in names.iter().zip(originals).zip(transformed) {
        assert_eq!(
            original.len(),
            transformed.len(),
            "transformed column '{name}' must have the same length as the original column"
        );
        if column_is_zero_variance(original, transformed) {
            return Err(PanelError::ZeroVarianceAfterDemeaning {
                column: name.clone(),
            });
        }
    }
    Ok(())
}

/// `ids`（`entity`または`time`の列）に観測数1のグループがあれば
/// `PanelError::SingletonGroup`を返す。
///
/// 複数のsingletonグループが存在する場合は、観測順で最初に現れるグループのみを報告する
/// （`validate_no_zero_variance_regressors`の「最初の1件を報告」方針と統一）。観測数は
/// `FeInput`の構築時に作ったコード（`GroupCodes::counts`）をそのまま使う。
pub(crate) fn reject_singleton_group(
    dimension: PanelDimension,
    column: &str,
    ids: &GroupCodes,
) -> Result<(), PanelError> {
    // 観測順で最初に現れたsingletonを報告する（旧実装の`String`版と同じ選び方）。
    for &c in ids.codes() {
        if ids.counts()[c] == 1 {
            return Err(PanelError::SingletonGroup {
                dimension,
                column: column.to_string(),
                group_id: ids.keys()[c].clone(),
            });
        }
    }
    Ok(())
}

/// 2-way FEがバランスパネル（`entity` × `time`の全組合せが過不足なく1回ずつ存在する）
/// であることを検証する（`fe-spec.md`3.1節）。
///
/// 観測数カウントの一致（`n_obs == n_entities * n_periods`）だけでは不十分
/// （`PanelError::UnbalancedPanelForTwoWay`のdocコメント参照: あるペアの重複と別ペアの
/// 欠落が相殺してカウントだけ一致する入力がありうる）。代わりに、`(entity, time)`
/// ペアが重複なく、かつ`n_obs == n_entities * n_periods`であることを検証する。
/// ペア集合は`entity × time`の全組合せグリッド（サイズ`n_entities * n_periods`）の
/// 部分集合であるため、重複が無く要素数がグリッドのサイズと一致すれば、部分集合が全体
/// （＝全組合せが埋まっている）と一致することが数学的に保証される。
///
/// `n_obs != n_entities * n_periods`ならその時点で不均衡（ペアの重複判定は不要）。一致する
/// ときだけ、グリッドのセル（`entity`コード×`n_periods`+`time`コード）を一度ずつ埋めて
/// 重複を検出する（`n_obs`個の`bool`）。
///
/// `entity.nobs() == time.nobs()`は`FeInput::from_group_codes`が既に保証している契約
/// （呼び出し側は常に同じ`FeInput`からこの2つを渡す）。
pub(crate) fn validate_balanced_panel(
    entity: &GroupCodes,
    time: &GroupCodes,
    names: &EffectNames,
) -> Result<(), PanelError> {
    debug_assert_eq!(
        entity.nobs(),
        time.nobs(),
        "entity and time must have the same length (FeInput contract)"
    );
    let n_obs = entity.nobs();
    let n_entities = entity.n_groups();
    let n_periods = time.n_groups();
    let expected = n_entities * n_periods;
    let unbalanced = || PanelError::UnbalancedPanelForTwoWay {
        n_obs,
        entity_column: names.get(PanelDimension::Entity).to_string(),
        n_entities,
        time_column: names.get(PanelDimension::Time).to_string(),
        n_periods,
    };

    if n_obs != expected {
        return Err(unbalanced());
    }
    let mut filled = vec![false; expected];
    for (&e, &t) in entity.codes().iter().zip(time.codes()) {
        let cell = &mut filled[e * n_periods + t];
        if *cell {
            return Err(unbalanced());
        }
        *cell = true;
    }
    Ok(())
}

/// `x`の各列のうち、1-way within変換（`entity`でθ=1のquasi-demean）後に分散がある列
/// （時間変化のある列）の添字を昇順で返す。分散ゼロの判定は
/// `validate_no_zero_variance_regressors`と同じ`column_is_zero_variance`（相対閾値）を使うため、
/// ここで時間変化ありと判定された列は、そのままFEに渡しても分散ゼロで拒否されない。
///
/// REが時間不変の列を許容するために使う（`re-spec.md`5章）。FEの`fit()`は分散ゼロの列を
/// 拒否する設計のままで、この関数は拒否の代わりに「FEに渡す列を選ぶ」ために同じ判定を使う。
pub(crate) fn time_varying_regressor_indices(x: &[Vec<f64>], entity: &GroupCodes) -> Vec<usize> {
    let within = WithinTransform::one_way(entity);
    x.iter()
        .enumerate()
        .filter(|(_, original)| !column_is_zero_variance(original, &within.apply(original)))
        .map(|(j, _)| j)
        .collect()
}

/// 2-way within変換後に分散がある列の添字を昇順で返す（`time_varying_regressor_indices`の
/// 2-way版）。2-wayのwithin変換は「entityでquasi-demean→timeでquasi-demean」の2段階
/// （`within_transform_two_way`）で、バランスパネルで閉形式の二重デミーニングと一致する。
/// 時点間でのみ（個体間でのみ）変わる列は2-way within変換後に全ゼロ列になり、時間不変の列と
/// 同じく「分散ゼロ」と判定される。
///
/// **バランスパネルであることの検証はしない**（呼び出し側が`validate_balanced_panel`で
/// 済ませている前提。不均衡だと2段階適用は二重デミーニングと一致しない）。
pub(crate) fn two_way_time_varying_regressor_indices(
    x: &[Vec<f64>],
    entity: &GroupCodes,
    time: &GroupCodes,
) -> Vec<usize> {
    let within = WithinTransform::two_way(entity, time);
    x.iter()
        .enumerate()
        .filter(|(_, original)| !column_is_zero_variance(original, &within.apply(original)))
        .map(|(j, _)| j)
        .collect()
}

/// 固定効果・変量効果の方向（1-way/2-way）を指定する。FEの`FeEffects`・REの`ReEffects`は
/// この型の別名で、`FeEstimator::fit`・`ReEstimator::fit`が受け取る
/// （`CovType`と同じ「小さな固定選択肢の公開enum」パターン）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelEffects {
    /// entityのみ（FEは`fe-spec.md`3.1節、REは`re-spec.md`3章）。REでは`ReInput`に`time`が
    /// あっても使わない。
    OneWay,
    /// entity + time（バランスパネル必須。FEは`fe-spec.md`1章、REは`re-spec.md`5章）。入力の
    /// `time`が時間次元になる（無ければ`PanelError::TwoWayRequiresTime`）。
    TwoWay,
}

/// パネル手法（FE・RE）が対応する`cov_type`（3.1節・3.2節）。FEの`FeCovType`・REの`ReCovType`は
/// この型の別名。`OlsEstimator`の`CovType`をそのまま再利用しない理由は`fe.rs`モジュールdoc
/// 「`cov_type`対応」参照——HC0を含まない、パネル専用の閉じた選択肢にすることで「無効な組み合わせを
/// 型で表現不可能にする」設計にしている（IVの`WeightType`と同じ判断）。`Dk`はOLSの`CovType::Dk`と
/// 異なるアルゴリズム（Driscoll-Kraay型パネルHAC、`fe.rs`モジュールdoc「Driscoll-Kraay型パネルHAC対応」
/// 参照）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PanelCovType {
    /// 等分散前提（`σ̂² (X̃'X̃)⁻¹`、`σ̂² = SSR/df_resid`）。
    Classical,
    /// White型の不均一分散ロバスト（小標本補正係数`n/df_resid`）。
    Hc1,
    /// レバレッジベースの不均一分散ロバスト。FEは`h_ii_full`（`leverage_full`）、REは素の
    /// レバレッジを使う。
    Hc2,
    /// Hc2よりさらに保守的なレバレッジ補正。
    Hc3,
    /// クラスターロバスト。`groups`が`None`なら`entity`引数の列を自動的に使う
    /// （3.2節、`cluster`省略時のデフォルト挙動）。`Some`のコードの行数が`n`と食い違うと
    /// `CommonError::ClusterDimensionMismatch`（`GroupCodes`のdoc参照）。
    Cluster { groups: Option<GroupCodes> },
    /// Driscoll-Kraay型パネルHAC（3.1節）。`bandwidth`が`None`なら
    /// `floor(4*(t/100)^(2/9))`（`t`はユニークな時点数）で自動計算する。
    ///
    /// `time`: DK計算の時点ラベルと時間順のコード（必須）。2-wayの固定効果・個体効果の時間次元
    /// （入力の`time`）とは独立で、DKカーネルを適用する時間粒度を呼び出し側が明示する
    /// （`engine_pybind`の`dk_time`が配線される）。入力の`time`へのフォールバックはしない——
    /// どの列が時点かを暗黙に借用すると、意図と違う列が選ばれても気づけないため。
    Dk {
        bandwidth: Option<i64>,
        time: TimeKeys,
    },
}

/// 設計行列の変換の違い（FE固有の補正を使うか）。`panel_cov_params`が`cov_type`別の補正を
/// 切り替える。
#[derive(Clone, Copy)]
pub(crate) enum PanelStructure<'a> {
    /// FE: within変換後の設計行列には吸収した固定効果ダミーが無い。HC2/HC3のレバレッジは
    /// LSDV相当の`leverage_full`、クラスターのKはfixest流のネスト判定（`fe_cluster_k_correction`）、
    /// 2グループ退化（`DegenerateClusterTwoGroups`・`DegenerateDkTwoPeriods`）の検査を行う。
    FixedEffects {
        /// 2-wayの時点コード（1-wayは`None`）。
        two_way_time: Option<&'a GroupCodes>,
    },
    /// RE: 変換済み設計行列に省略された固定効果ダミーが無い。素のレバレッジ、クラスターのK
    /// は`df_model`、退化検査は不要。
    RandomEffects,
}

/// [`panel_cov_params`]の入力。FE（within変換後）・RE（θ変換後）・FEIV（`X̂`と構造残差）が
/// 同じ形で渡せるよう、計算に使う材料だけを借用で持つ。
///
/// `x`・`xtx_inv`・`residuals`は同じ回帰のものでなければならない。FE・REは変換後の設計行列`X̃`・
/// `(X̃'X̃)⁻¹`・その回帰の残差。2SLSに基づくFEIVは設計行列に`X̂`（変換後の外生変数と第一段階の
/// 予測値）、`xtx_inv`に`(X̂'X̂)⁻¹`、`residuals`に構造残差`y - Xβ̂`（実際の内生変数`X`を使う。
/// 第二段階の残差`y - X̂β̂`ではない）を渡す。操作変数も`entity`（2-wayは`time`）の同じ
/// `WithinTransform`で変換しておくこと（`FixedEffects`の2グループ退化の検査は、変換後の列が各水準内で
/// 和ゼロになる前提に立つ）。
pub(crate) struct PanelCovInput<'a> {
    /// 変換後の設計行列（n, k）。
    pub(crate) x: &'a Mat<f64>,
    /// `x`に対応する残差。`cov_type=Classical`の`σ̂² = Σr²/df_resid`もここから求める。
    pub(crate) residuals: &'a [f64],
    /// `(X'X)⁻¹`（k, k）。
    pub(crate) xtx_inv: &'a Mat<f64>,
    pub(crate) df_resid: usize,
    /// モデルの自由度（FEは`k + neffects`、REは係数の数）。DKとクラスターの小標本補正のKの既定値。
    pub(crate) df_model: usize,
    /// ロバスト検定（F検定）の対象になる傾き係数の数`q`（FEは`k`、REは定数項を除く`k - 1`）。
    /// クラスター数・時点数が`q`を超えるかの事前検証に使う。
    pub(crate) n_slopes: usize,
    /// entityのコード。`Cluster { groups: None }`のクラスター変数の既定。
    pub(crate) entity: &'a GroupCodes,
    pub(crate) structure: PanelStructure<'a>,
}

/// [`panel_cov_params`]の結果。
pub(crate) struct PanelCov {
    pub(crate) cov_params: Mat<f64>,
    /// t検定・信頼区間・F検定の自由度。`Cluster`は`G-1`、`Dk`は`t_periods-1`
    /// （fixestの`ssc()`既定`t.df="min"`）、それ以外は`df_resid`。標準誤差のスケールに使う
    /// `K`とは別軸の値。
    pub(crate) df_inference: usize,
    /// `Dk`のとき実際に使われたバンド幅。`Dk`以外では`None`。
    pub(crate) dk_bandwidth_used: Option<usize>,
}

/// `cov_type`に応じたパネル手法の係数分散共分散行列と推論の自由度。FE・REの`fit()`が共有する
/// （`FeCovType`・`ReCovType`は[`PanelCovType`]の別名）。計算式そのものは
/// `panel_{classical,hc,cluster,driscoll_kraay}_cov_params`が持ち、ここは`cov_type`ごとの
/// 事前検証・補正の選択・自由度の切り替えだけを担う。`structure`で渡すFE固有の補正は
/// `fe.rs`モジュールdoc「`cov_type`対応」参照。
///
/// # Errors
/// - `Cluster`: 長さ不一致・`G < 2`・`G <= q`（`validate_cluster_groups`・
///   `validate_cluster_count_covers_slopes`）、FEの`G=2`退化は`DegenerateClusterTwoGroups`
/// - `Dk`: 時点列の長さ不一致は`IdentifierDimensionMismatch`、バンド幅・時点数の不正
///   （`resolve_dk_bandwidth`・`validate_dk_periods_cover_tested_coefficients`）、FEの時点2つの
///   退化は`DegenerateDkTwoPeriods`
pub(crate) fn panel_cov_params(
    cov_type: &PanelCovType,
    input: &PanelCovInput<'_>,
) -> Result<PanelCov, PanelError> {
    let PanelCovInput {
        x,
        residuals,
        xtx_inv,
        df_resid,
        df_model,
        n_slopes,
        entity,
        structure,
    } = *input;
    let n = x.nrows();
    let k = x.ncols();
    let plain = |cov_params: Mat<f64>| PanelCov {
        cov_params,
        df_inference: df_resid,
        dk_bandwidth_used: None,
    };

    match cov_type {
        PanelCovType::Classical => {
            let ssr: f64 = residuals.iter().map(|r| r * r).sum();
            Ok(plain(panel_classical_cov_params(xtx_inv, ssr, df_resid, k)))
        }
        PanelCovType::Hc1 => Ok(plain(panel_hc_cov_params(
            x,
            residuals,
            xtx_inv,
            df_resid,
            PanelHcVariant::Hc1,
        ))),
        PanelCovType::Hc2 | PanelCovType::Hc3 => {
            let h_plain = leverages(x, xtx_inv);
            let h = match structure {
                PanelStructure::FixedEffects { two_way_time } => {
                    leverage_full(&h_plain, entity, two_way_time, n)
                }
                PanelStructure::RandomEffects => h_plain,
            };
            let variant = if matches!(cov_type, PanelCovType::Hc2) {
                PanelHcVariant::Hc2(&h)
            } else {
                PanelHcVariant::Hc3(&h)
            };
            Ok(plain(panel_hc_cov_params(
                x, residuals, xtx_inv, df_resid, variant,
            )))
        }
        PanelCovType::Cluster { groups } => {
            // 既定（entityクラスター）は入力のコードを再利用し、明示指定の列は渡された
            // コードをそのまま使う。
            let group_codes = groups.as_ref().unwrap_or(entity);
            let n_groups = validate_cluster_groups(group_codes, n, ClusterSetting::CovType)?;
            validate_cluster_count_covers_slopes(n_groups, n_slopes)?;
            let k_correction = match structure {
                PanelStructure::FixedEffects { two_way_time } => {
                    // 直前の`G > q`により、ここで`G=2`なら`q`は高々1。
                    if n_slopes >= 1
                        && n_groups == 2
                        && two_group_split_is_degenerate(entity, two_way_time, group_codes)
                    {
                        return Err(PanelError::DegenerateClusterTwoGroups);
                    }
                    fe_cluster_k_correction(
                        entity,
                        two_way_time,
                        entity.n_groups(),
                        df_model,
                        n_slopes,
                        group_codes,
                    )
                }
                // REは固定効果ダミーのネスト補正が不要なため`K=df_model`をそのまま使う
                // （OLS自身の`cluster_cov_params`と数式的に同一になる）。
                PanelStructure::RandomEffects => df_model,
            };
            let cov_params =
                panel_cluster_cov_params(x, residuals, xtx_inv, n, k, group_codes, k_correction);
            Ok(PanelCov {
                cov_params,
                df_inference: n_groups - 1,
                dk_bandwidth_used: None,
            })
        }
        PanelCovType::Dk {
            bandwidth,
            time: dk_time,
        } => {
            // 入力の検証を通らない`dk_time`（公開APIの`Dk.time`）の長さをここで検証する。
            // 合わないと退化判定の`zip`が黙って切り詰められ、DK計算は範囲外アクセスになる。
            if dk_time.len() != n {
                return Err(PanelError::IdentifierDimensionMismatch {
                    dimension: PanelDimension::Time,
                    y_rows: n,
                    other_rows: dk_time.len(),
                });
            }
            let time_codes = dk_time.codes();
            let t_periods = time_codes.n_groups();
            let bw = resolve_dk_bandwidth(*bandwidth, t_periods)?;
            validate_dk_periods_cover_tested_coefficients(t_periods, n_slopes)
                .map_err(|e| e.in_stage(Stage::OverallFTest))?;
            if let PanelStructure::FixedEffects { two_way_time } = structure {
                // 直前の`t_periods > q`により、ここで`t_periods=2`なら`q`は高々1。
                if n_slopes >= 1
                    && t_periods == 2
                    && two_group_split_is_degenerate(entity, two_way_time, time_codes)
                {
                    return Err(PanelError::DegenerateDkTwoPeriods);
                }
            }
            // fixestのDKは`K.fixef="full"`が既定（クラスター変数が無くネスト判定自体が発生しない、
            // `fe_cluster_k_correction`のdocコメント参照）。`K=df_model`をそのまま使う。
            let cov_params =
                panel_driscoll_kraay_cov_params(x, residuals, xtx_inv, time_codes, df_model, bw);
            Ok(PanelCov {
                cov_params,
                df_inference: t_periods - 1,
                dk_bandwidth_used: Some(bw),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panel_cataloged_variants_display_the_catalog_message_and_common_delegates() {
        let err = PanelError::InvalidDkBandwidth { bandwidth: 6, t: 6 };
        assert_eq!(
            err.to_string(),
            "dk_bandwidth must be from 0 to 5, one less than the number of unique time periods \
             in dk_time (6), got 6."
        );
        let wrapped = PanelError::Common(CommonError::InvalidHacLags { hac_lags: 3, n: 3 });
        assert_eq!(wrapped.report().code(), ErrorCode::HacLagsOutOfRange);
    }

    #[test]
    fn dk_bandwidth_error_maps_to_the_catalog() {
        let report = PanelError::InvalidDkBandwidth { bandwidth: 7, t: 6 }.report();
        assert_eq!(report.code(), ErrorCode::DkBandwidthOutOfRange);
        assert!(report.validate().is_ok());
    }

    #[test]
    fn panel_input_errors_map_to_catalog_codes_with_valid_params() {
        let cases = [
            (
                PanelError::InsufficientDegreesOfFreedom {
                    n_obs: 10,
                    n_entities: 5,
                    n_periods: None,
                    k: 5,
                },
                ErrorCode::InsufficientDegreesOfFreedom,
                "Not enough observations for the panel model: 10 observation(s), but 5 coefficient(s) and 5 fixed-effect parameter(s) (entity effects, plus time effects in a two-way model) must be estimated; at least 11 are needed.",
            ),
            (
                PanelError::InsufficientDegreesOfFreedom {
                    n_obs: 10,
                    n_entities: 5,
                    n_periods: Some(2),
                    k: 1,
                },
                ErrorCode::InsufficientDegreesOfFreedom,
                "Not enough observations for the panel model: 10 observation(s), but 1 coefficient(s) and 6 fixed-effect parameter(s) (entity effects, plus time effects in a two-way model) must be estimated; at least 8 are needed.",
            ),
            (
                PanelError::SingletonGroup {
                    dimension: PanelDimension::Entity,
                    column: "firm".to_string(),
                    group_id: "a".to_string(),
                },
                ErrorCode::SingletonGroup,
                "The group 'a' of column 'firm' in effects has only one observation, and singleton groups are not dropped automatically.",
            ),
            (
                PanelError::SingletonGroup {
                    dimension: PanelDimension::Time,
                    column: "year".to_string(),
                    group_id: "2020".to_string(),
                },
                ErrorCode::SingletonGroup,
                "The group '2020' of column 'year' in effects has only one observation, and singleton groups are not dropped automatically.",
            ),
            (
                PanelError::UnbalancedPanelForTwoWay {
                    n_obs: 5,
                    entity_column: "firm".to_string(),
                    n_entities: 2,
                    time_column: "year".to_string(),
                    n_periods: 3,
                },
                ErrorCode::PanelNotBalancedForTwoWay,
                "Two-way estimation requires a balanced panel: 5 observation(s) for 2 groups of 'firm' and 3 of 'year', but every combination needs exactly one.",
            ),
            (
                PanelError::ZeroVarianceAfterDemeaning {
                    column: "x".to_string(),
                },
                ErrorCode::RegressorNoWithinVariation,
                "Regressor 'x' has no variation left after the within transformation: it is time-invariant, or collinear with the fixed effects.",
            ),
            (
                PanelError::OutcomeConstantWithinEntity,
                ErrorCode::OutcomeConstantWithinEntity,
                "The outcome has no variation left after the fixed effects are removed: it is constant within every entity, or fully explained by the entity and time effects.",
            ),
            (
                PanelError::InsufficientDkPeriods { t_periods: 1 },
                ErrorCode::DkPeriodsTooFew,
                "cov_type=\"dk\" needs at least 2 unique time periods in dk_time, but there are 1.",
            ),
            (
                PanelError::InsufficientDkPeriodsForInference { t_periods: 3, q: 3 },
                ErrorCode::DkPeriodsNotAboveCoefficients,
                "cov_type=\"dk\" needs more time periods than jointly tested coefficients: dk_time has 3 periods for 3 coefficient(s); at least 4 are needed.",
            ),
            (
                PanelError::AmemiyaRequiresWithinVariation {
                    columns: vec!["a".to_string(), "b".to_string()],
                },
                ErrorCode::AmemiyaRequiresWithinVariation,
                "variance_method=\"amemiya\" needs every regressor to vary within the effects, but these do not: 'a', 'b'.",
            ),
            (
                PanelError::NonPositiveIdiosyncraticVariance {
                    method: "amemiya",
                    sigma2_eps: -0.5,
                },
                ErrorCode::NonPositiveIdiosyncraticVariance,
                "The estimated idiosyncratic variance is not positive (-0.5) with variance_method=\"amemiya\", so the random-effects weights are undefined.",
            ),
        ];
        for (err, code, message) in cases {
            let report = err.report();
            assert_eq!(report.code(), code);
            assert!(report.validate().is_ok());
            assert_eq!(report.message(), message);
            assert_eq!(err.to_string(), message);
        }
        for err in [
            PanelError::DegenerateDkTwoPeriods,
            PanelError::DegenerateClusterTwoGroups,
            PanelError::VarianceComponentsMomentsSingular { method: "amemiya" },
        ] {
            let report = err.report();
            assert!(report.validate().is_ok());
            assert_eq!(err.to_string(), report.message());
        }
        // 実装の誤りでだけ起きるものは`INTERNAL_ERROR`になる。
        for err in [
            PanelError::TwoWayRequiresTime,
            PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Entity,
                y_rows: 3,
                other_rows: 2,
            },
        ] {
            let report = err.report();
            assert_eq!(report.code(), ErrorCode::InternalError);
            assert!(report.validate().is_ok());
        }
    }

    #[test]
    fn validate_panel_input_reports_the_first_failing_check_in_order() {
        let y = [1.0, 2.0, 3.0];
        let names = labels(&["x"]);
        let entity = GroupCodes::from_labels(&labels(&["a", "a", "b"]));
        let time = TimeKeys::lexicographic(labels(&["1", "2", "1"]));
        assert_eq!(
            validate_panel_input(&y, &[vec![1.0, 2.0, 3.0]], &names, &entity, Some(&time)),
            Ok(())
        );

        // x列 → entityの長さ → timeの長さ → entityのキー、の順。すべて同時に崩れていれば
        // x列の不一致が先に返る。
        let short_entity = GroupCodes::from_labels(&labels(&["a", "b"]));
        let short_time = TimeKeys::lexicographic(labels(&["1"]));
        let keyless = GroupCodes::from_labels_without_keys(&labels(&["a", "a"]));
        assert_eq!(
            validate_panel_input(&y, &[vec![1.0]], &names, &keyless, Some(&short_time)),
            Err(CommonError::DimensionMismatch {
                y_rows: 3,
                x_rows: 1
            }
            .into())
        );
        assert_eq!(
            validate_panel_input(
                &y,
                &[vec![1.0; 3]],
                &names,
                &short_entity,
                Some(&short_time)
            ),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Entity,
                y_rows: 3,
                other_rows: 2
            })
        );
        assert_eq!(
            validate_panel_input(&y, &[vec![1.0; 3]], &names, &entity, Some(&short_time)),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                y_rows: 3,
                other_rows: 1
            })
        );
        // キー無しのentityでtimeの長さも合っていなければ、キー検査は最後なのでtimeの不一致が先。
        let keyless_full = GroupCodes::from_labels_without_keys(&labels(&["a", "a", "b"]));
        assert_eq!(
            validate_panel_input(
                &y,
                &[vec![1.0; 3]],
                &names,
                &keyless_full,
                Some(&short_time)
            ),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                y_rows: 3,
                other_rows: 1
            })
        );
        assert!(matches!(
            validate_panel_input(&y, &[vec![1.0; 3]], &names, &keyless_full, None),
            Err(PanelError::Common(CommonError::Internal { .. }))
        ));
    }

    #[test]
    fn panel_dimension_displays_lowercase_name() {
        assert_eq!(PanelDimension::Entity.to_string(), "entity");
        assert_eq!(PanelDimension::Time.to_string(), "time");
    }

    #[test]
    fn in_stage_adds_a_stage_to_the_wrapped_code_and_replaces_an_inner_stage() {
        let dof = PanelError::InsufficientDegreesOfFreedom {
            n_obs: 10,
            n_entities: 5,
            n_periods: None,
            k: 5,
        };
        let wrapped = dof.in_stage(Stage::VarianceComponentWithinRegression);
        let report = wrapped.report();
        assert_eq!(report.code(), ErrorCode::InsufficientDegreesOfFreedom);
        assert_eq!(
            report.stage(),
            Some(&Stage::VarianceComponentWithinRegression)
        );
        assert!(report.validate().is_ok());
        assert_eq!(wrapped.to_string(), report.message());
        assert!(
            wrapped
                .to_string()
                .starts_with("In the within regression used to estimate a variance component: ")
        );
        // 内側の`stage`は外側に置き換わる。
        let inner = PanelError::WithinRegressionFailed {
            source: LeastSquaresError::SingularMatrix {
                columns: vec!["x".to_string()],
            },
        };
        assert_eq!(inner.report().stage(), Some(&Stage::WithinRegression));
        let outer = inner.in_stage(Stage::VarianceComponentWithinRegression);
        assert_eq!(
            outer.report().stage(),
            Some(&Stage::VarianceComponentWithinRegression)
        );
    }

    #[test]
    fn panel_error_messages_are_human_readable() {
        assert_eq!(
            PanelError::InvalidDkBandwidth { bandwidth: 7, t: 6 }.to_string(),
            "dk_bandwidth must be from 0 to 5, one less than the number of unique time periods \
             in dk_time (6), got 7."
        );
        assert!(
            PanelError::DegenerateDkTwoPeriods
                .to_string()
                .starts_with("With 2 time periods")
        );
        assert!(
            PanelError::DegenerateClusterTwoGroups
                .to_string()
                .starts_with("With 2 clusters")
        );
        // `#[error(transparent)]`: `Common`は内側`CommonError`のメッセージを
        // そのまま透過する（`IvError`/`MleError`のテストと同じ確認）。
        assert_eq!(
            PanelError::Common(CommonError::InsufficientClusters {
                setting: ClusterSetting::CovType,
                g: 1
            })
            .to_string(),
            r#"cov_type="cluster" needs at least 2 clusters, but the cluster column has 1."#
        );
        // 回帰の包みは原因のメッセージに、どの回帰かを示す文脈句が付く。
        let wrapped = PanelError::WithinRegressionFailed {
            source: LeastSquaresError::SingularMatrix {
                columns: vec!["x".to_string()],
            },
        };
        assert_eq!(wrapped.to_string(), wrapped.report().message());
        assert!(wrapped.to_string().contains("design matrix is singular"));
    }

    #[test]
    fn validate_dk_periods_cover_tested_coefficients_rejects_t_at_or_below_q() {
        // `rank(S) ≤ t-1`のため`t > q`が必要。`t == q`が境界（拒否）、`t == q + 1`は通す。
        assert_eq!(
            validate_dk_periods_cover_tested_coefficients(6, 6),
            Err(PanelError::InsufficientDkPeriodsForInference { t_periods: 6, q: 6 })
        );
        assert_eq!(
            validate_dk_periods_cover_tested_coefficients(6, 20),
            Err(PanelError::InsufficientDkPeriodsForInference {
                t_periods: 6,
                q: 20
            })
        );
        assert_eq!(validate_dk_periods_cover_tested_coefficients(6, 5), Ok(()));
        // 検定対象なし（FEの固定効果のみモデル）は常に通す。
        assert_eq!(validate_dk_periods_cover_tested_coefficients(2, 0), Ok(()));
    }

    #[test]
    fn panel_error_implements_partial_eq() {
        assert_eq!(
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                column: "entity".to_string(),
                group_id: "a".to_string(),
            },
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                column: "entity".to_string(),
                group_id: "a".to_string(),
            }
        );
        assert_ne!(
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                column: "entity".to_string(),
                group_id: "a".to_string(),
            },
            PanelError::SingletonGroup {
                dimension: PanelDimension::Time,
                column: "time".to_string(),
                group_id: "a".to_string(),
            }
        );
        assert_ne!(
            PanelError::InsufficientDegreesOfFreedom {
                n_obs: 10,
                n_entities: 8,
                n_periods: None,
                k: 2,
            },
            PanelError::InsufficientDegreesOfFreedom {
                n_obs: 10,
                n_entities: 8,
                n_periods: Some(2),
                k: 2,
            }
        );
    }

    #[test]
    fn panel_error_wraps_common_error_via_from() {
        let common = CommonError::InvalidConfidenceLevel {
            confidence_level: 1.5,
        };
        let panel_error: PanelError = common.into();
        assert_eq!(
            panel_error,
            PanelError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5,
            })
        );
    }

    #[test]
    fn panel_error_from_operator_propagates_common_error() {
        fn inner() -> Result<(), CommonError> {
            Err(CommonError::InsufficientClusters {
                setting: ClusterSetting::CovType,
                g: 1,
            })
        }
        fn outer() -> Result<(), PanelError> {
            inner()?;
            Ok(())
        }
        assert_eq!(
            outer().unwrap_err(),
            PanelError::Common(CommonError::InsufficientClusters {
                setting: ClusterSetting::CovType,
                g: 1
            })
        );
    }

    #[test]
    fn within_regression_failed_preserves_source() {
        let err = PanelError::WithinRegressionFailed {
            source: LeastSquaresError::Common(CommonError::InsufficientObservations { n: 2, k: 3 }),
        };
        assert_eq!(
            err,
            PanelError::WithinRegressionFailed {
                source: LeastSquaresError::Common(CommonError::InsufficientObservations {
                    n: 2,
                    k: 3,
                }),
            }
        );
    }

    #[test]
    fn nested_observation_count_error_carries_the_regression_stage() {
        let report = PanelError::BetweenRegressionFailed {
            means: BetweenMeans::Time,
            source: LeastSquaresError::Common(CommonError::InsufficientObservations { n: 2, k: 3 }),
        }
        .report();
        assert_eq!(report.code(), ErrorCode::InsufficientObservations);
        assert_eq!(report.stage(), Some(&Stage::BetweenRegressionTime));
        assert!(report.validate().is_ok());

        let report = PanelError::WithinRegressionFailed {
            source: LeastSquaresError::Common(CommonError::MissingClusterColumn {
                setting: ClusterSetting::CovType,
            }),
        }
        .report();
        assert_eq!(report.code(), ErrorCode::ClusterRequired);
        assert!(report.stage().is_none());
    }

    #[test]
    fn between_regression_failed_message_and_equality() {
        let err = PanelError::BetweenRegressionFailed {
            means: BetweenMeans::Time,
            source: LeastSquaresError::Common(CommonError::InsufficientObservations { n: 2, k: 3 }),
        };
        assert_eq!(err.to_string(), err.report().message());
        assert_eq!(err.report().stage(), Some(&Stage::BetweenRegressionTime));
        assert_eq!(
            err,
            PanelError::BetweenRegressionFailed {
                means: BetweenMeans::Time,
                source: LeastSquaresError::Common(CommonError::InsufficientObservations {
                    n: 2,
                    k: 3,
                }),
            }
        );
    }

    #[test]
    fn quasi_demeaned_regression_failed_message_and_equality() {
        let err = PanelError::QuasiDemeanedRegressionFailed {
            source: LeastSquaresError::SingularMatrix {
                columns: vec!["x".to_string()],
            },
        };
        assert_eq!(err.to_string(), err.report().message());
        assert_eq!(err.report().stage(), Some(&Stage::QuasiDemeanedRegression));
        assert_eq!(
            err,
            PanelError::QuasiDemeanedRegressionFailed {
                source: LeastSquaresError::SingularMatrix {
                    columns: vec!["x".to_string()]
                },
            }
        );
    }

    #[test]
    fn hausman_test_failed_message_and_equality() {
        let err = PanelError::HausmanTestFailed {
            source: LeastSquaresError::SingularMatrix {
                columns: vec!["x".to_string()],
            },
        };
        assert_eq!(err.to_string(), err.report().message());
        assert_eq!(err.report().stage(), Some(&Stage::HausmanRegression));
        assert_eq!(
            err,
            PanelError::HausmanTestFailed {
                source: LeastSquaresError::SingularMatrix {
                    columns: vec!["x".to_string()]
                },
            }
        );
    }

    // ── TimeKeys ───────────────────────────────────────────────────────────

    fn labels(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn time_keys_factorized_constructors_match_the_label_constructors() {
        // 仮コードは出現順の`[0,1,2,0]`（`"9"=0, "10"=1, "2"=2`）。
        let codes = vec![0, 1, 2, 0];
        let group_labels = ["9", "10", "2"];

        let by_label =
            TimeKeys::by_integer(labels(&["9", "10", "2", "9"]), &[9, 10, 2, 9]).unwrap();
        let factorized =
            TimeKeys::by_integer_factorized(codes.clone(), &group_labels, &[9, 10, 2]).unwrap();
        assert_eq!(factorized, by_label);
        assert_eq!(factorized.periods(), by_label.periods());
        assert_eq!(factorized.len(), 4);

        let lexicographic = TimeKeys::lexicographic(labels(&["9", "10", "2", "9"]));
        assert_eq!(
            TimeKeys::lexicographic_factorized(codes.clone(), &group_labels).unwrap(),
            lexicographic
        );

        // `-0.0`と`0.0`は別の時点（ラベルが異なる）で、`total_cmp`の順に並ぶ。
        let by_float = TimeKeys::by_float(
            labels(&["0.0", "-0.0", "1.0", "0.0"]),
            &[0.0, -0.0, 1.0, 0.0],
        )
        .unwrap();
        let float_factorized = TimeKeys::by_float_factorized(
            vec![0, 1, 2, 0],
            &["0.0", "-0.0", "1.0"],
            &[0.0, -0.0, 1.0],
        )
        .unwrap();
        assert_eq!(float_factorized, by_float);
        assert_eq!(float_factorized.periods(), ["-0.0", "0.0", "1.0"]);
    }

    #[test]
    fn time_keys_factorized_constructors_reject_inconsistent_inputs() {
        // 順序づけの値の数が仮グループ数と合わない。
        assert!(matches!(
            TimeKeys::by_integer_factorized(vec![0, 1], &["a", "b"], &[1]),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                ..
            })
        ));
        // 範囲外・未使用の仮コード。
        assert!(matches!(
            TimeKeys::lexicographic_factorized(vec![0, 2], &["a", "b"]),
            Err(PanelError::Common(CommonError::Internal { .. }))
        ));
    }

    #[test]
    fn time_keys_factorized_merges_provisional_groups_with_the_same_label_and_value() {
        // 仮グループ0と1は同じラベル"a"・同じ値1（併合される）。2は"b"・値2。
        let merged =
            TimeKeys::by_integer_factorized(vec![0, 1, 2, 1], &["a", "a", "b"], &[1, 1, 2])
                .unwrap();
        let expected = TimeKeys::by_integer(labels(&["a", "a", "b", "a"]), &[1, 1, 2, 1]).unwrap();
        assert_eq!(merged, expected);
        assert_eq!(merged.periods(), ["a", "b"]);
    }

    #[test]
    fn time_keys_factorized_rejects_same_label_with_different_values() {
        // 同じラベル"a"なのに値が違い、値の順序で"b"を挟むと隣り合わない。併合されずキーが
        // 重複するため、黙って受け入れずに`Err`にする（呼び出し側の契約違反）。
        assert!(matches!(
            TimeKeys::by_integer_factorized(vec![0, 1, 2], &["a", "b", "a"], &[1, 2, 3]),
            Err(PanelError::Common(CommonError::Internal { .. }))
        ));
        assert!(matches!(
            TimeKeys::by_float_factorized(vec![0, 1, 2], &["a", "b", "a"], &[1.0, 2.0, 3.0]),
            Err(PanelError::Common(CommonError::Internal { .. }))
        ));
    }

    #[test]
    fn time_keys_equality_compares_period_labels_not_only_codes() {
        // コード（時間順）は同じ`[0, 1, 0, 1]`でも、時点のラベルが違えば等しくない。
        let coarse = TimeKeys::lexicographic(labels(&["1", "2", "1", "2"]));
        let fine = TimeKeys::lexicographic(labels(&["q1", "q2", "q1", "q2"]));
        assert_ne!(coarse, fine);
        assert_eq!(
            coarse,
            TimeKeys::lexicographic(labels(&["1", "2", "1", "2"]))
        );
    }

    mod time_keys_proptests {
        use proptest::collection;
        use proptest::prelude::*;

        use super::*;

        /// 行ごとの値`values`について、仮コードを出現順と逆の順序で振り、`split`なら同じ値を
        /// 行の偶奇で2つの仮グループに割る。仮グループごとのラベルと代表値も返す。
        fn factorize_reversed<V: Copy + PartialEq>(
            values: &[V],
            label: impl Fn(V) -> String,
            split: bool,
        ) -> (Vec<usize>, Vec<String>, Vec<V>) {
            let key = |row: usize| (values[row], if split { row % 2 } else { 0 });
            let mut unique: Vec<(V, usize)> = Vec::new();
            for row in (0..values.len()).rev() {
                if !unique.contains(&key(row)) {
                    unique.push(key(row));
                }
            }
            let codes = (0..values.len())
                .map(|row| unique.iter().position(|k| *k == key(row)).unwrap())
                .collect();
            let labels = unique.iter().map(|(v, _)| label(*v)).collect();
            let group_values = unique.iter().map(|(v, _)| *v).collect();
            (codes, labels, group_values)
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            /// 整数の時点（`"10"`と`"9"`のように辞書順と値の順序が食い違う値を含む）で、
            /// `by_integer_factorized`は`by_integer`と同じ`TimeKeys`（コードとラベル）になる。
            #[test]
            fn by_integer_factorized_agrees_with_by_integer(
                values in collection::vec(-5i128..120, 1..60),
                split in any::<bool>(),
            ) {
                let ids: Vec<String> = values.iter().map(|v| v.to_string()).collect();
                let (codes, group_labels, group_values) =
                    factorize_reversed(&values, |v| v.to_string(), split);
                let refs: Vec<&str> = group_labels.iter().map(String::as_str).collect();

                let actual = TimeKeys::by_integer_factorized(codes, &refs, &group_values).unwrap();
                let expected = TimeKeys::by_integer(ids, &values).unwrap();
                prop_assert_eq!(actual, expected);
            }

            /// 浮動小数の時点（`0.0`と`-0.0`は別の時点）で、`by_float_factorized`は`by_float`と
            /// 同じ`TimeKeys`になる。
            #[test]
            fn by_float_factorized_agrees_with_by_float(
                picks in collection::vec(0usize..7, 1..60),
                split in any::<bool>(),
            ) {
                let pool = [0.0_f64, -0.0, 1.5, -2.25, 10.0, 9.0, 1e-3];
                let values: Vec<f64> = picks.iter().map(|&i| pool[i]).collect();
                // `f64`は`PartialEq`で`0.0 == -0.0`なので、ビット列で別の値として分解する。
                let bits: Vec<u64> = values.iter().map(|v| v.to_bits()).collect();
                let ids: Vec<String> = values.iter().map(|v| format!("{v:?}")).collect();
                let (codes, group_labels, group_bits) =
                    factorize_reversed(&bits, |b| format!("{:?}", f64::from_bits(b)), split);
                let group_values: Vec<f64> = group_bits.iter().map(|&b| f64::from_bits(b)).collect();
                let refs: Vec<&str> = group_labels.iter().map(String::as_str).collect();

                let actual = TimeKeys::by_float_factorized(codes, &refs, &group_values).unwrap();
                let expected = TimeKeys::by_float(ids, &values).unwrap();
                prop_assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn time_keys_by_integer_orders_periods_numerically_not_lexicographically() {
        // 辞書順なら "10" < "2" < "9"。値の順序では 2 < 9 < 10。
        let keys = TimeKeys::by_integer(labels(&["9", "10", "2", "9"]), &[9, 10, 2, 9]).unwrap();
        assert_eq!(keys.periods(), ["2", "9", "10"]);
        assert_eq!(keys.codes().codes(), [1, 2, 0, 1]);
        assert_eq!(keys.len(), 4);
    }

    #[test]
    fn time_keys_by_integer_orders_negative_values() {
        let keys = TimeKeys::by_integer(labels(&["-1", "-10", "3"]), &[-1, -10, 3]).unwrap();
        assert_eq!(keys.periods(), ["-10", "-1", "3"]);
    }

    #[test]
    fn time_keys_by_float_orders_periods_numerically() {
        let keys = TimeKeys::by_float(
            labels(&["1.5", "-0.5", "10.0", "2.5"]),
            &[1.5, -0.5, 10.0, 2.5],
        )
        .unwrap();
        assert_eq!(keys.periods(), ["-0.5", "1.5", "2.5", "10.0"]);
    }

    #[test]
    fn time_keys_by_float_breaks_ties_between_equal_values_by_label() {
        // `0.0`と`-0.0`は値としては等しいが別のラベル。順序は行の並びに依らずラベルの辞書順
        // （"-0.0" < "0.0"）で決まる。
        let forward =
            TimeKeys::by_float(labels(&["0.0", "-0.0", "1.0"]), &[0.0, -0.0, 1.0]).unwrap();
        let backward =
            TimeKeys::by_float(labels(&["1.0", "-0.0", "0.0"]), &[1.0, -0.0, 0.0]).unwrap();
        assert_eq!(forward.periods(), ["-0.0", "0.0", "1.0"]);
        assert_eq!(backward.periods(), ["-0.0", "0.0", "1.0"]);
    }

    #[test]
    fn time_keys_lexicographic_matches_group_codes_from_labels() {
        let ids = labels(&["b", "a", "c", "a"]);
        let keys = TimeKeys::lexicographic(ids.clone());
        assert_eq!(keys.codes(), &GroupCodes::from_labels(&ids));
        assert_eq!(keys.periods(), ["a", "b", "c"]);
    }

    #[test]
    fn time_keys_rejects_values_of_a_different_length() {
        let err = TimeKeys::by_integer(labels(&["1", "2"]), &[1]).unwrap_err();
        assert_eq!(
            err,
            PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                y_rows: 2,
                other_rows: 1,
            }
        );
        assert!(TimeKeys::by_float(labels(&["1"]), &[1.0, 2.0]).is_err());
    }

    #[test]
    fn time_keys_accepts_empty_input() {
        let keys = TimeKeys::by_integer(vec![], &[]).unwrap();
        assert!(keys.is_empty());
        assert!(keys.periods().is_empty());
    }

    // ── quasi_demean_column ────────────────────────────────────────────────

    /// `["a", "a", "b", "b", "b"]`のエンティティ列から整数コードを作るヘルパ。
    fn entities(ids: &[&str]) -> GroupCodes {
        let ids: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
        GroupCodes::from_labels(&ids)
    }

    /// エンティティID→θの組を、`entity`のコード順（キーの辞書順）の`Vec`に並べるヘルパ。
    fn theta_for(entity: &GroupCodes, pairs: &[(&str, f64)]) -> Vec<f64> {
        entity
            .keys()
            .iter()
            .map(|key| {
                pairs
                    .iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, v)| *v)
                    .expect("test helper: theta pair for every key")
            })
            .collect()
    }

    #[test]
    fn quasi_demean_column_with_theta_one_is_the_within_transformation() {
        // FE相当: θ_i = 1.0。col[i] - ȳ_i. になる。
        // a: mean = (10 + 20) / 2 = 15、b: mean = (3 + 6 + 9) / 3 = 6。
        let entity = entities(&["a", "a", "b", "b", "b"]);
        let col = [10.0, 20.0, 3.0, 6.0, 9.0];
        let theta = theta_for(&entity, &[("a", 1.0), ("b", 1.0)]);

        let out = quasi_demean_column(&col, &entity, &theta);

        assert_eq!(out, vec![-5.0, 5.0, -3.0, 0.0, 3.0]);
        // within変換後は各エンティティ内で和がゼロ。
        assert!((out[0] + out[1]).abs() < 1e-12);
        assert!((out[2] + out[3] + out[4]).abs() < 1e-12);
    }

    #[test]
    fn quasi_demean_column_with_theta_zero_is_identity() {
        // θ_i = 0.0 なら何も引かない（プーリングOLS相当）。
        let entity = entities(&["a", "a", "b", "b"]);
        let col = [1.5, -2.0, 7.0, 0.25];
        let theta = theta_for(&entity, &[("a", 0.0), ("b", 0.0)]);

        let out = quasi_demean_column(&col, &entity, &theta);

        assert_eq!(out, col.to_vec());
    }

    #[test]
    fn quasi_demean_column_with_arbitrary_per_entity_theta() {
        // RE相当: エンティティごとに異なる θ_i。
        // a: mean = 15、θ_a = 0.5 → 引く量 7.5
        // b: mean = 6、 θ_b = 1.0 → 引く量 6.0
        let entity = entities(&["a", "a", "b", "b", "b"]);
        let col = [10.0, 20.0, 3.0, 6.0, 9.0];
        let theta = theta_for(&entity, &[("a", 0.5), ("b", 1.0)]);

        let out = quasi_demean_column(&col, &entity, &theta);

        assert_eq!(out, vec![2.5, 12.5, -3.0, 0.0, 3.0]);
    }

    #[test]
    fn quasi_demean_column_handles_unbalanced_panel() {
        // 不均衡パネル（T_a = 1, T_b = 3）でも1-wayは各エンティティ平均を引くだけで
        // 正確に成立する（`docs/spec/fe-spec.md`1章）。
        // a: mean = 4.0（単一観測）→ θ_a = 1.0 で 0.0 になる。
        // b: mean = (2 + 4 + 6) / 3 = 4.0。
        let entity = entities(&["a", "b", "b", "b"]);
        let col = [4.0, 2.0, 4.0, 6.0];
        let theta = theta_for(&entity, &[("a", 1.0), ("b", 1.0)]);

        let out = quasi_demean_column(&col, &entity, &theta);

        assert_eq!(out, vec![0.0, -2.0, 0.0, 2.0]);
    }

    #[test]
    fn quasi_demean_column_preserves_length_and_row_order() {
        // エンティティがブロックにまとまっていない（インターリーブした）配置でも、
        // 行ごとに所属エンティティの平均を引く。出力長・行順は入力どおり。
        let entity = entities(&["x", "y", "x", "y", "x"]);
        let col = [1.0, 100.0, 2.0, 200.0, 3.0];
        let theta = theta_for(&entity, &[("x", 1.0), ("y", 1.0)]);

        let out = quasi_demean_column(&col, &entity, &theta);

        // x: mean = 2.0、y: mean = 150.0
        assert_eq!(out, vec![-1.0, -50.0, 0.0, 50.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "entity length must match column length")]
    fn quasi_demean_column_panics_on_length_mismatch() {
        let entity = entities(&["a", "b"]);
        let col = [1.0, 2.0, 3.0];
        let theta = theta_for(&entity, &[("a", 1.0), ("b", 1.0)]);
        let _ = quasi_demean_column(&col, &entity, &theta);
    }

    #[test]
    #[should_panic(expected = "names must have one entry per original column")]
    fn ensure_no_zero_variance_columns_panics_when_names_are_fewer_than_columns() {
        let originals = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let transformed = originals.clone();
        let _ = ensure_no_zero_variance_columns(&["x1".to_string()], &originals, &transformed);
    }

    #[test]
    #[should_panic(expected = "theta must have one value per entity group")]
    fn quasi_demean_column_panics_when_theta_length_differs_from_groups() {
        let entity = entities(&["a", "a", "b"]);
        let col = [1.0, 2.0, 3.0];
        let theta = vec![1.0]; // "b" の分が欠けている
        let _ = quasi_demean_column(&col, &entity, &theta);
    }

    #[test]
    fn quasi_demean_column_single_group_with_theta_one_is_all_zero() {
        // 全行が同一エンティティ（グループが1つだけ）＋ θ=1 → 全行がグループ平均に
        // 一致するため出力は全ゼロ（この列だけでは within 変換後に情報が残らない。
        // `fe-spec.md`1章の分散ゼロ検証・`fe-spec.md`1章のsingleton検証は消費側 fe.rs の責務）。
        let entity = entities(&["a", "a", "a", "a"]);
        let col = [3.0, 5.0, 7.0, 9.0]; // mean = 6.0
        let theta = theta_for(&entity, &[("a", 1.0)]);

        let out = quasi_demean_column(&col, &entity, &theta);

        assert_eq!(out, vec![-3.0, -1.0, 1.0, 3.0]);
        assert!(out.iter().map(|v| v.abs()).sum::<f64>() > 0.0); // 各行は非ゼロ
        assert!(out.iter().sum::<f64>().abs() < 1e-12); // 和はゼロ
    }

    #[test]
    fn quasi_demean_column_applies_theta_outside_unit_interval_verbatim() {
        // この関数は θ の値域（RE では [0, 1)）を検証しない。負・>1 でも式どおり
        // `col[i] - θ_i · ȳ_i.` を適用する（値域の担保は θ を計算する RE 側の責務）。
        // a: mean = 10、θ_a = -0.5 → 引く量 -5   → col[i] + 5
        // b: mean = 20、θ_b =  2.0 → 引く量 40   → col[i] - 40
        let entity = entities(&["a", "a", "b", "b"]);
        let col = [8.0, 12.0, 15.0, 25.0];
        let theta = theta_for(&entity, &[("a", -0.5), ("b", 2.0)]);

        let out = quasi_demean_column(&col, &entity, &theta);

        assert_eq!(out, vec![13.0, 17.0, -25.0, -15.0]);
    }

    // ── panel_driscoll_kraay_cov_params ────────────────────────────────────

    /// 1変数・各時点1観測の入力で、`bandwidth`ごとの未スケール`Ŝ`を標準Bartlettカーネルの
    /// 定義式どおりに素朴な二重ループで求める（`panel_driscoll_kraay_cov_params`の
    /// 行列積による実装とは独立な手計算）。`skip_last_lag`が真なら`l = bandwidth`の項を
    /// 落とす（fixestの`bandwidth == T-1`での挙動の再現）。
    fn naive_bartlett_meat(xi: &[f64], bandwidth: usize, skip_last_lag: bool) -> f64 {
        let t_periods = xi.len();
        let mut meat: f64 = xi.iter().map(|v| v * v).sum();
        for l in 1..=bandwidth {
            if skip_last_lag && l == bandwidth {
                continue;
            }
            let weight = 1.0 - (l as f64) / ((bandwidth + 1) as f64);
            let gamma: f64 = (l..t_periods).map(|t| xi[t] * xi[t - l]).sum();
            meat += 2.0 * weight * gamma;
        }
        meat
    }

    #[test]
    fn panel_cluster_cov_params_uses_k_for_the_meat_and_k_correction_for_the_scale() {
        // meatの次元は`k`（傾き係数の数）、小標本補正`(n-1)/(n-K)`の`K`は別引数`k_correction`。
        // 2つが取り違えられないよう、`k=1`・`k_correction=3`として手計算と比べる。
        // クラスター a={行0,行2}、b={行1,行3}、スコア s_i = e_i x_i。
        let x_values = [1.0, 2.0, -1.0, 0.5];
        let residuals = [0.5, -1.0, 2.0, 1.5];
        let x = Mat::from_fn(4, 1, |i, _| x_values[i]);
        let xtx_inv = Mat::from_fn(1, 1, |_, _| 0.25);
        let groups = GroupCodes::from_labels(&labels(&["a", "b", "a", "b"]));

        let cov = panel_cluster_cov_params(&x, &residuals, &xtx_inv, 4, 1, &groups, 3);

        let s_a = residuals[0] * x_values[0] + residuals[2] * x_values[2];
        let s_b = residuals[1] * x_values[1] + residuals[3] * x_values[3];
        let meat = s_a * s_a + s_b * s_b;
        let correction = (2.0 / 1.0) * ((4.0 - 1.0) / (4.0 - 3.0));
        let expected = correction * 0.25 * meat * 0.25;
        assert!((*cov.get(0, 0) - expected).abs() < 1e-12 * expected.abs());
    }

    #[test]
    fn panel_driscoll_kraay_cov_params_matches_naive_bartlett_for_every_bandwidth_up_to_t_minus_1()
    {
        // T=5の各時点に1観測を置き、`ξ_t = e_t·x_t`を既知の値にする。`(X̃'X̃)⁻¹ = 1`・
        // `K = 1`（スケールが`T/(T-1)`のみ）にして、`Ŝ`の定義式との直接比較にする。
        //
        // `bandwidth = T-1 = 4`（許容範囲の上限ちょうど）の最終ラグ項`l = 4`
        // （`ξ_4 ξ_0`、重み`1/5`）が含まれることを固定する。fixestの`cpp_driscoll_kraay`は
        // この境界で最大ラグを`T-2`に切り詰める（ラグの個数と最大ラグの取り違え）ため
        // 最終ラグ項を落とし、本実装とは一致しない。本実装は標準の定義どおり含める
        // （`docs/spec/fe-spec.md`3.3節）。`bandwidth <= T-2`ではfixestと一致する。
        let ids: Vec<String> = (0..5).map(|t| t.to_string()).collect();
        let time = GroupCodes::from_labels(&ids);
        let x_values = [1.0, -2.0, 0.5, 3.0, -1.5];
        let residuals = [0.7, 1.1, -0.4, 0.9, 2.0];
        let x = Mat::from_fn(5, 1, |i, _| x_values[i]);
        let xtx_inv = Mat::from_fn(1, 1, |_, _| 1.0);
        let xi: Vec<f64> = x_values
            .iter()
            .zip(&residuals)
            .map(|(x, e)| x * e)
            .collect();
        let scale = 5.0 / 4.0 * (4.0 / (5.0 - 1.0));

        for bandwidth in 0..=4 {
            let cov =
                panel_driscoll_kraay_cov_params(&x, &residuals, &xtx_inv, &time, 1, bandwidth);
            let expected = scale * naive_bartlett_meat(&xi, bandwidth, false);
            assert!(
                (*cov.get(0, 0) - expected).abs() < 1e-12 * expected.abs().max(1.0),
                "bandwidth={bandwidth}: got {}, expected {expected}",
                *cov.get(0, 0)
            );
        }

        // 境界で最終ラグ項が寄与していること（落とした値とは異なること）を確かめる。
        // これが成り立たないと上のループが境界での項の有無を区別できない。
        let full = naive_bartlett_meat(&xi, 4, false);
        let dropped = naive_bartlett_meat(&xi, 4, true);
        assert!((full - dropped).abs() > 1e-6);
    }

    fn codes(values: &[&str]) -> GroupCodes {
        let labels: Vec<String> = values.iter().map(|s| s.to_string()).collect();
        GroupCodes::from_labels(&labels)
    }

    #[test]
    fn leverage_full_one_way_uses_each_rows_entity_size_for_unbalanced_unordered_rows() {
        // 行がエンティティ順に並んでおらず、T_i（a=3・b=2・c=1）が不均衡。各行は自分の
        // エンティティの観測数`T_i`で引かれなければならない（行番号やコード順で引くと外れる）。
        let entity = codes(&["b", "a", "c", "a", "b", "a"]);
        let h_within = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let sizes = [2.0, 3.0, 1.0, 3.0, 2.0, 3.0];

        let h_full = leverage_full(&h_within, &entity, None, 6);

        for i in 0..6 {
            assert!(
                (h_full[i] - (1.0 / sizes[i] + h_within[i])).abs() < 1e-15,
                "row {i}"
            );
        }
    }

    #[test]
    fn leverage_full_two_way_uses_entity_and_time_sizes_for_unbalanced_unordered_rows() {
        // entity（a=3・b=2・c=1）とtime（t1=4・t2=1・t3=1）で行ごとの観測数が異なる。
        // `1/T_i + 1/N_t - 1/n + h`のentityとtimeの取り違えも検出できる。
        let entity = codes(&["b", "a", "c", "a", "b", "a"]);
        let time = codes(&["t1", "t1", "t2", "t3", "t1", "t1"]);
        let h_within = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let entity_sizes = [2.0, 3.0, 1.0, 3.0, 2.0, 3.0];
        let time_sizes = [4.0, 4.0, 1.0, 1.0, 4.0, 4.0];

        let h_full = leverage_full(&h_within, &entity, Some(&time), 6);

        for i in 0..6 {
            let expected = 1.0 / entity_sizes[i] + 1.0 / time_sizes[i] - 1.0 / 6.0 + h_within[i];
            assert!((h_full[i] - expected).abs() < 1e-15, "row {i}");
        }
    }

    #[test]
    fn fixef_dimension_nested_within_cluster_true_for_default_entity_grouping() {
        let entity = codes(&["a", "a", "b", "b"]);
        assert!(fixef_dimension_nested_within_cluster(&entity, &entity));
    }

    #[test]
    fn fixef_dimension_nested_within_cluster_true_for_coarser_grouping() {
        // stateはentityより粗い分割（a,b→east、c,d→west）で、各entityは単一のstateに
        // 属するため「nested」と判定されるべき（fixestの実測でも同じ挙動を確認済み、
        // モジュールdoc参照）。
        let entity = codes(&["a", "a", "b", "b", "c", "c", "d", "d"]);
        let state = codes(&[
            "east", "east", "east", "east", "west", "west", "west", "west",
        ]);
        assert!(fixef_dimension_nested_within_cluster(&entity, &state));
    }

    #[test]
    fn fixef_dimension_nested_within_cluster_false_when_an_entity_spans_multiple_clusters() {
        // entity "a" が異なる2つのクラスター（"1"と"2"）にまたがるため、nestedではない。
        let entity = codes(&["a", "a", "b", "b"]);
        let cluster = codes(&["1", "2", "1", "2"]);
        assert!(!fixef_dimension_nested_within_cluster(&entity, &cluster));
    }

    #[test]
    fn fe_cluster_k_correction_floor_is_a_no_op_under_realistic_inputs() {
        // `fit()`が保証する前提（`df_model = k + neffects`）の下では、フロア
        // `K = max(K, k+1)`は常にno-op（`fe_cluster_k_correction`関数docの代数的
        // 導出参照、rust-reviewer指摘を受けて追加）。1-way全ネストの現実的な入力で
        // `K`がちょうど`k+1`に一致することを確認する。
        let entity = codes(&["a", "a", "b", "b"]);
        let k = 1;
        let n_entities = 2;
        let df_model = k + n_entities; // 実際のFeEstimator::fitと同じneffects=n_entitiesの関係
        let k_correction = fe_cluster_k_correction(&entity, None, n_entities, df_model, k, &entity);
        assert_eq!(k_correction, k + 1);
    }

    #[test]
    fn fe_cluster_k_correction_floor_engages_for_inconsistent_inputs() {
        // フロア自体（`K = max(K, k+1)`）を直接検証する。`fit()`経由では`df_model`と
        // `n_entities`が常に整合しているため到達不能な組み合わせ（`fe_cluster_k_
        // correction`関数docの代数的導出参照）を、この関数を直接呼ぶことで意図的に
        // 構成する: `df_model=2`・`n_entities=2`（1-way全ネスト）だと、整合していれば
        // `df_model`は`k+n_entities=k+2`になるはずのところを`df_model=2`（`k=1`なら
        // `k+n_entities=3`のはず）に矛盾させ、`raw_k = df_model - nested_size_sum + 1
        // = 2 - 2 + 1 = 1`が`k+1=2`を下回る状況を作る。
        let entity = codes(&["a", "a", "b", "b"]);
        let k = 1;
        let n_entities = 2;
        let df_model = 2; // 本来のneffects=n_entities=2との整合を意図的に崩す
        let k_correction = fe_cluster_k_correction(&entity, None, n_entities, df_model, k, &entity);
        assert_eq!(
            k_correction,
            k + 1,
            "floor should clamp raw_k=1 up to k+1=2"
        );
    }

    #[test]
    fn every_level_splits_once_across_two_groups_detects_pattern() {
        let entity = codes(&["a", "a", "b", "b"]);
        assert!(every_level_splits_once_across_two_groups(
            &entity,
            &codes(&["1", "2", "2", "1"])
        ));
        // 同じグループに2観測。
        assert!(!every_level_splits_once_across_two_groups(
            &entity,
            &codes(&["1", "1", "1", "2"])
        ));
        // 3観測のエンティティ。
        assert!(!every_level_splits_once_across_two_groups(
            &codes(&["a", "a", "a", "b", "b"]),
            &codes(&["1", "2", "1", "1", "2"])
        ));
        // 1観測のエンティティ（singletonは通常`fit()`が先に弾く）。
        assert!(!every_level_splits_once_across_two_groups(
            &codes(&["a", "a", "b"]),
            &codes(&["1", "2", "1"])
        ));
    }
}
