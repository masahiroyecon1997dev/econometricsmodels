//! `panel`系統のエラー型`PanelError`と、エラーメッセージに載せる次元・平均の種別。
//! バリアントの方針と一覧は`common.rs`のモジュールdoc。

use std::fmt;

use thiserror::Error;

use super::fe_adjustment::absorbed_parameters;
use crate::linear::common::LeastSquaresError;
use crate::shared::error::{CommonError, ErrorInfo, staged_report};
use crate::shared::error_catalog::{ErrorCode, ErrorReport, Stage};
use crate::shared::validation::{MAX_ITER_LIMIT, MAX_ITER_MIN};

/// 固定効果の次元（`effects`の何番目の列か）。エラーメッセージ・バリデーションで「どの次元の
/// 問題か」を区別するために使う。
///
/// 次元の数は`effects`の長さで決まる。個体は0番目（[`ENTITY`](Self::ENTITY)）、2-wayの時点は
/// 1番目（[`TIME`](Self::TIME)）で、3つ目以降は添字で表す。`PanelError::SingletonGroup`等が
/// この型をフィールドとして持つ。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PanelDimension(usize);

impl PanelDimension {
    /// `effects`の1つ目（個体・企業・州等）。
    pub const ENTITY: Self = Self(0);
    /// `effects`の2つ目（2-wayの時点）。
    pub const TIME: Self = Self(1);

    /// `effects`の`index`番目（0始まり）の次元。
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// `effects`の何番目か（0始まり）。
    pub const fn index(self) -> usize {
        self.0
    }
}

impl fmt::Display for PanelDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "effects[{}]", self.0)
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
    /// FEの`df_resid`は`n_obs`から固定効果ダミー相当の自由度を追加で消費するため、
    /// `df_resid = n_obs - (水準数の合計 - (効果の数 - 1)) - k`となる
    /// （2つ目以降の効果は、ダミー同士の間の定数項ぶんのランク落ちを1つずつ引く。
    /// [`absorbed_parameters`]、`fe-spec.md`3.2節）。
    /// `CommonError::InsufficientObservations`（単純な`n <= k`）とは
    /// 消費する自由度の内訳が異なるため別バリアントにする。`n_groups`は`effects`の列ごとの
    /// 水準数で、1-wayなら要素は1つ。
    #[error("{}", self.catalog_message())]
    InsufficientDegreesOfFreedom {
        n_obs: usize,
        /// `effects`の列ごとの水準数（`effects`の順）。
        n_groups: Vec<usize>,
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
        /// 観測数1のグループを取り除くことを、新しいsingletonが現れなくなるまで繰り返したときに
        /// 落ちる行の総数（2-wayでは、個体のsingletonを落とすと時点のsingletonができる連鎖を含む）。
        n_dropped_rows: usize,
    },

    /// 2-way RE（entity + time）に不均衡（unbalanced）パネルが渡された。
    ///
    /// 2-way REのGLS変換（`quasi_demean_two_way`）はバランスパネルを前提にした閉形式で、不均衡では
    /// 成り立たない。2-way FEは不均衡パネル（同じ組合せの重複する行を含む）を交互射影で扱うため、
    /// このエラーは返さない。1-wayは不均衡でも成立する。
    ///
    /// 「バランス」の判定は観測数カウント（`n_obs == n_entities * n_periods`）だけでは不十分で、
    /// あるペアの重複と別ペアの欠落が相殺してカウントだけ一致する入力もありうる。このエラーを
    /// 構築する検証は、(entity, time)ペアの一意性・全組合せの充足まで確認する
    /// （`validate_balanced_panel`）。`entity_column`・`time_column`は`effects`の1つ目・2つ目の
    /// 列名（メッセージ用）。
    #[error("{}", self.catalog_message())]
    UnbalancedPanelForTwoWay {
        n_obs: usize,
        entity_column: String,
        n_entities: usize,
        time_column: String,
        n_periods: usize,
    },

    /// 不均衡な2-wayのHC2/HC3は、水準の少ない方の効果の水準数の2乗の密な行列を解くため、水準数に上限がある。
    #[error("{}", self.catalog_message())]
    LeverageDimensionTooLarge { n_levels: usize, max_levels: usize },

    /// 固定効果を除く交互射影が、反復の上限までに収束しなかった。`relative_change`は最後の1回の掃引で
    /// 引いた水準平均の最大を、列の絶対値の最大で割った値。
    #[error("{}", self.catalog_message())]
    AbsorptionNotConverged {
        n_sweeps: usize,
        relative_change: f64,
    },

    /// 交互射影の収束許容度`tol`が正の有限値でない。
    #[error("{}", self.catalog_message())]
    InvalidTol { tol: f64 },

    /// 交互射影の反復上限`max_iter`が範囲外。
    #[error("{}", self.catalog_message())]
    InvalidMaxIter { max_iter: i64 },

    /// 説明変数の値が全ての行で（丸め誤差を除いて）同じ。固定効果（個体ごとの切片）と共線になる。
    /// 固定効果を除いた後で分散がなくなる[`Self::ZeroVarianceAfterDemeaning`]とは、原因が生データの
    /// 定数かどうかで区別する。
    #[error("{}", self.catalog_message())]
    RegressorConstant { column: String },

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
    /// `time`は`effects`の2要素目で、2-wayのときだけ必要になる
    /// （`panel-common.md`1.1節）。未指定時のバリデーションエラーとしてここで担保する。
    ///
    /// `engine_pybind`は`time`の有無から1-way/2-wayを決めるため、Python APIからは起こらない
    /// （起きたら実装の誤りなので`INTERNAL_ERROR`として報告する）。
    #[error("{}", self.catalog_message())]
    TwoWayRequiresTime,

    /// `FeCovType::Dk`の明示的な`bandwidth`が`[0, t)`の範囲外（`t`はユニークな時点数）。
    ///
    /// `CommonError::InvalidHacLags`と同型のバリデーションだが、上限が観測数`n`
    /// ではなく時点数`t`になる点が異なる（DKのバンド幅は「時点のラグ」であり「観測の
    /// ラグ」ではないため、`fe.rs`モジュールdoc「Driscoll-Kraay型パネルHAC」
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

    /// within変換済みデータに対する最小二乗推定（`shared::least_squares::least_squares`の
    /// 直接呼び出し、`panel-common.md`4.3節）が失敗した。
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
    /// 失敗した。`WithinRegressionFailed`とは別バリアント（理由は`common.rs`のモジュールdoc参照）。
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
    /// （`between_ssr_and_df`。エンティティ平均への`least_squares`）が失敗した。
    ///
    /// `WithinRegressionFailed`と同じ`LeastSquaresError`ラップだが、対象がFEのwithin回帰
    /// ではなくREのbetween回帰のため別バリアントにする（`FTestFailed`と同じ判断、
    /// `common.rs`のモジュールdoc参照）。最も起こりやすいのはエンティティ数が説明変数の数以下
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
    /// `OlsEstimator::fit_auxiliary(include_intercept=false)`委譲が失敗した。
    ///
    /// `WithinRegressionFailed`（FEのwithin変換済みデータ）・`BetweenRegressionFailed`
    /// （REのbetween回帰）とは対象が異なるため別バリアントにする（同じ判断の3件目、
    /// `common.rs`のモジュールdoc参照）。
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
            Self::InsufficientDegreesOfFreedom { n_obs, n_groups, k } => {
                let n_effects = absorbed_parameters(n_groups);
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
                n_dropped_rows,
            } => ErrorReport::new(ErrorCode::SingletonGroup)
                .param("argument", "effects")
                .param("column", column.as_str())
                .param("group", group_id.as_str())
                .param("n_dropped_rows", *n_dropped_rows),
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
            Self::LeverageDimensionTooLarge {
                n_levels,
                max_levels,
            } => ErrorReport::new(ErrorCode::LeverageDimensionTooLarge)
                .param("n_levels", *n_levels)
                .param("max_levels", *max_levels),
            Self::AbsorptionNotConverged {
                n_sweeps,
                relative_change,
            } => ErrorReport::new(ErrorCode::AbsorptionNotConverged)
                .param("n_sweeps", *n_sweeps)
                .param("relative_change", *relative_change),
            Self::InvalidTol { tol } => ErrorReport::new(ErrorCode::ToleranceNotPositiveFinite)
                .param("option", "tol")
                .param("value", *tol),
            Self::InvalidMaxIter { max_iter } => ErrorReport::new(ErrorCode::MaxIterOutOfRange)
                .param("value", *max_iter)
                .param("min", MAX_ITER_MIN)
                .param("max", MAX_ITER_LIMIT),
            Self::RegressorConstant { column } => {
                ErrorReport::new(ErrorCode::RegressorConstant).param("column", column.as_str())
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::error::ClusterSetting;

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
                    n_groups: vec![5],
                    k: 5,
                },
                ErrorCode::InsufficientDegreesOfFreedom,
                "Not enough observations for the panel model: 10 observation(s), but 5 coefficient(s) and 5 fixed-effect parameter(s) (entity effects, plus time effects in a two-way model) must be estimated; at least 11 are needed.",
            ),
            (
                PanelError::InsufficientDegreesOfFreedom {
                    n_obs: 10,
                    n_groups: vec![5, 2],
                    k: 1,
                },
                ErrorCode::InsufficientDegreesOfFreedom,
                "Not enough observations for the panel model: 10 observation(s), but 1 coefficient(s) and 6 fixed-effect parameter(s) (entity effects, plus time effects in a two-way model) must be estimated; at least 8 are needed.",
            ),
            (
                PanelError::SingletonGroup {
                    dimension: PanelDimension::ENTITY,
                    column: "firm".to_string(),
                    group_id: "a".to_string(),
                    n_dropped_rows: 0,
                },
                ErrorCode::SingletonGroup,
                "The group 'a' of column 'firm' in effects has only one observation, and singleton groups are not dropped automatically; removing singleton groups repeatedly would drop 0 row(s).",
            ),
            (
                PanelError::SingletonGroup {
                    dimension: PanelDimension::TIME,
                    column: "year".to_string(),
                    group_id: "2020".to_string(),
                    n_dropped_rows: 0,
                },
                ErrorCode::SingletonGroup,
                "The group '2020' of column 'year' in effects has only one observation, and singleton groups are not dropped automatically; removing singleton groups repeatedly would drop 0 row(s).",
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
                "Two-way random effects require a balanced panel: 5 observation(s) for 2 groups of 'firm' and 3 of 'year', but every combination needs exactly one.",
            ),
            (
                PanelError::LeverageDimensionTooLarge {
                    n_levels: 5000,
                    max_levels: 3000,
                },
                ErrorCode::LeverageDimensionTooLarge,
                "The HC2 and HC3 covariances of an unbalanced two-way model need a dense matrix over the levels of the smaller effect, which has 5000 levels, but at most 3000 are supported.",
            ),
            (
                PanelError::AbsorptionNotConverged {
                    n_sweeps: 100,
                    relative_change: 0.5,
                },
                ErrorCode::AbsorptionNotConverged,
                "Removing the fixed effects did not converge after 100 pass(es); the last pass still changed the data by 0.5 relative to its scale.",
            ),
            (
                PanelError::InvalidTol { tol: 0.0 },
                ErrorCode::ToleranceNotPositiveFinite,
                "tol must be a positive finite number, got 0.0.",
            ),
            (
                PanelError::InvalidMaxIter { max_iter: 0 },
                ErrorCode::MaxIterOutOfRange,
                "max_iter must be an integer from 1 to 10000, got 0.",
            ),
            (
                PanelError::RegressorConstant {
                    column: "x".to_string(),
                },
                ErrorCode::RegressorConstant,
                "Regressor 'x' has the same value in every row up to rounding, so it is collinear with the fixed effects.",
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
                dimension: PanelDimension::ENTITY,
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
    fn insufficient_degrees_of_freedom_counts_effects_of_every_dimension() {
        let report = PanelError::InsufficientDegreesOfFreedom {
            n_obs: 10,
            n_groups: vec![5, 2, 3],
            k: 1,
        }
        .report();
        assert_eq!(report.code(), ErrorCode::InsufficientDegreesOfFreedom);
        assert!(report.validate().is_ok());
        // 5 + 2 + 3 - 2 = 8個の固定効果のパラメータと係数1つで、少なくとも10が要る。
        assert!(
            report
                .message()
                .contains("1 coefficient(s) and 8 fixed-effect parameter(s)")
        );
        assert!(report.message().contains("at least 10 are needed"));
    }

    #[test]
    fn panel_dimension_displays_its_position_in_effects() {
        assert_eq!(PanelDimension::ENTITY.to_string(), "effects[0]");
        assert_eq!(PanelDimension::TIME.to_string(), "effects[1]");
        assert_eq!(PanelDimension::new(2).to_string(), "effects[2]");
        assert_eq!(PanelDimension::new(2).index(), 2);
    }

    #[test]
    fn in_stage_adds_a_stage_to_the_wrapped_code_and_replaces_an_inner_stage() {
        let dof = PanelError::InsufficientDegreesOfFreedom {
            n_obs: 10,
            n_groups: vec![5],
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
    fn panel_error_implements_partial_eq() {
        assert_eq!(
            PanelError::SingletonGroup {
                dimension: PanelDimension::ENTITY,
                column: "entity".to_string(),
                group_id: "a".to_string(),
                n_dropped_rows: 0,
            },
            PanelError::SingletonGroup {
                dimension: PanelDimension::ENTITY,
                column: "entity".to_string(),
                group_id: "a".to_string(),
                n_dropped_rows: 0,
            }
        );
        assert_ne!(
            PanelError::SingletonGroup {
                dimension: PanelDimension::ENTITY,
                column: "entity".to_string(),
                group_id: "a".to_string(),
                n_dropped_rows: 0,
            },
            PanelError::SingletonGroup {
                dimension: PanelDimension::TIME,
                column: "time".to_string(),
                group_id: "a".to_string(),
                n_dropped_rows: 0,
            }
        );
        assert_ne!(
            PanelError::InsufficientDegreesOfFreedom {
                n_obs: 10,
                n_groups: vec![8],
                k: 2,
            },
            PanelError::InsufficientDegreesOfFreedom {
                n_obs: 10,
                n_groups: vec![8, 2],
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
}
