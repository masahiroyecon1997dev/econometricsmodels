//! `panel`系統（FE/RE）で共有するエラー型。
//!
//! `LeastSquaresError`（`engine::linear::common`）・`MleError`（`engine::nonlinear::common`）・
//! `IvError`（`engine::iv::common`）の前例に倣い、FE/REで個別に`FeError`/`ReError`を作らず
//! `PanelError`を共有する（`docs/planning/specs/panel-api-design.md`4.4節、Issue #172）。
//!
//! `DimensionMismatch`/`InsufficientObservations`/`InvalidConfidenceLevel`/
//! `MissingClusterColumn`/`InsufficientClusters`/`ComputationFailed`は`engine::error::
//! CommonError`に切り出し済みのため、ここでは`Common`バリアント経由で保持する。
//!
//! FE/RE固有バリアントは、`panel-api-design.md`6章（FE固有論点）・7章（RE固有論点）で
//! 仕様が確定しているバリデーション条件をカバーする:
//!
//! - `IdentifierDimensionMismatch`: `y`と`entity`/`time`の長さ不一致（1章、Issue #175）
//! - `InsufficientDegreesOfFreedom`: パネル自由度調整（6.3節）
//! - `SingletonGroup`: 観測数1のグループ（6.5節）
//! - `UnbalancedPanelForTwoWay`: 2-way FEのバランスパネル必須（6.4節）
//! - `ZeroVarianceAfterDemeaning`: within変換後に分散ゼロの説明変数（6.7節）
//! - `TwoWayRequiresTime`: 2-way FE指定時の`time`必須（1.1節）
//! - `HacRequiresTime`: Driscoll-Kraay型パネルHAC（`FeCovType::Hac`）指定時の`time`必須
//!   （3.1節、Issue #182。2-way FEは`TwoWayRequiresTime`で既に必須化されているため、
//!   1-way FEでのみ発生しうる）
//! - `InvalidHacBandwidth`: `FeCovType::Hac`の明示的な`bandwidth`が`[0, t)`の範囲外
//!   （`t`はユニークな時点数、Issue #182。`LeastSquaresError::InvalidHacLags`と同型だが
//!   上限が観測数`n`ではなく時点数`t`）
//! - `WithinRegressionFailed`: within変換済みデータの最小二乗推定委譲の失敗（4.3節）
//! - `FTestFailed`: F統計量（Issue #186、`fe.rs`モジュールdoc「自由度調整」のF統計量節）の
//!   Wald検定（`crate::linear::ols::wald_f_test`）が失敗した場合。`WithinRegressionFailed`と
//!   意味が異なる（`OlsEstimator::fit`自体は既に成功した後の、F検定固有の共分散部分行列の
//!   ほぼ特異性というbackstopのみ、`ols.rs`の`wald_f_test`docコメント参照）ため別バリアントに
//!   分離した（`IvError::FirstStageFailed`が`WithinRegressionFailed`と同じ`LeastSquaresError`
//!   ラップでも変換箇所ごとに専用バリアントにする判断と同じ）。
//! - `BetweenRegressionFailed`: RE（Swamy-Arora分散成分推定、7.1節、Issue #193）の
//!   between回帰（エンティティ平均への`OlsEstimator::fit(include_intercept=true)`）が
//!   失敗した場合（エンティティ数が説明変数の数以下等）。`WithinRegressionFailed`と同じ
//!   `LeastSquaresError`ラップだが、対象がFEのwithin回帰ではなくREのbetween回帰のため
//!   別バリアントにする（`FTestFailed`と同じ判断）。
//! - `QuasiDemeanedRegressionFailed`: RE（`ReEstimator::fit`、7.4節、Issue #195）の
//!   準偏差変換済みデータ（`quasi_demean_transform`の出力に、同じθで変換した定数列を
//!   加えたもの）への`OlsEstimator::fit(include_intercept=false)`委譲が失敗した場合。
//!   `WithinRegressionFailed`（FEのwithin変換済みデータ）・`BetweenRegressionFailed`
//!   （REのbetween回帰）とは対象が異なるため別バリアントにする（同じ判断の3件目）。
//!
//! RE固有（7章）で追加のバリアントが必要になった場合は、FE/RE実装issueで実際に計算
//! コードを書く過程で随時追加する（`LeastSquaresError`・`IvError`のdocコメントと同じ
//! 「土台を用意し、必要になった時点で足す」方針）。
//! ハウスマン統計量（`hausman_statistic`、Issue #174）は`CommonError`を返す
//! （`ensure_well_conditioned_symmetric_matrix`等の共通ヘルパーに揃える）。
//! `cov_fe - cov_re`が有限標本で非正定値になり統計量が負になるケースは**エラーにせず
//! そのまま返す**（R `plm::phtest`と同じ挙動、7.3節。差行列が数値的に特異なときだけ
//! `CommonError::ComputationFailed`）。

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use faer::prelude::{Solve, SolveLstsq};
use faer::{Mat, Side};
use statrs::distribution::{ChiSquared, ContinuousCDF};
use thiserror::Error;

use crate::error::CommonError;
use crate::linear::common::LeastSquaresError;

/// パネルデータの2つの次元。エラーメッセージ・バリデーションで「どちらの次元の
/// 問題か」を区別するために使う。
///
/// 2-way FEではエンティティ・時点のsingletonを対称に検出する
/// （`docs/planning/specs/panel-api-design.md`6.5節）ため、`PanelError::SingletonGroup`が
/// この型をフィールドとして持つ。FE/RE実装が進んだ段階で、singleton検出以外
/// （`fixed_effects()`の次元指定等）でも再利用できる想定。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelDimension {
    /// エンティティ方向（個体・企業・州等、`entity`引数の列）。
    Entity,
    /// 時点方向（`time`オプションの列）。
    Time,
}

impl fmt::Display for PanelDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PanelDimension::Entity => write!(f, "entity"),
            PanelDimension::Time => write!(f, "time"),
        }
    }
}

/// `InsufficientDegreesOfFreedom`のメッセージで、2-way（`n_periods`が`Some`）のときだけ
/// `, n_periods=<T>`を差し込む。`Option`のDebug表記（`Some(4)`/`None`）がユーザー向け
/// メッセージに漏れるのを避けるため（`.claude/rules/rust-style.md`「言語方針」——
/// ユーザー可視文字列は英語かつ実装の内部表現を出さない）。
fn n_periods_clause(n_periods: &Option<usize>) -> String {
    match n_periods {
        Some(n_periods) => format!(", n_periods={n_periods}"),
        None => String::new(),
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

    /// `y`と`entity`または`time`の長さが一致しない（`FeInput::from_columns`、
    /// `docs/planning/specs/panel-api-design.md`1章、Issue #175）。
    ///
    /// `y`と`x`列の不一致は`CommonError::DimensionMismatch`が既にカバーしている
    /// （対象列が異なるため専用バリアントにする）。`entity`/`time`のどちらの不一致かは
    /// `PanelDimension`で表す（`SingletonGroup`と同じ使い方）。`engine_pybind`が同じ
    /// polars DataFrameから列抽出する限り実際には起こり得ない（OLSの`y`/`x`長さ不一致
    /// チェックと同じ、`engine_pybind`〜`engine`間の契約に対する防御的な`Result`化）。
    #[error("dimension mismatch: y has {y_rows} rows but {dimension} has {other_rows} rows")]
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
    ///   （entityダミーとtimeダミーの間の定数項ぶんのランク落ちを`+1`で補正、6.3節）
    ///
    /// となる。`CommonError::InsufficientObservations`（単純な`n <= k`）とは
    /// 消費する自由度の内訳が異なるため別バリアントにする。`n_periods`は1-wayでは
    /// `None`（メッセージにも出さない、`n_periods_clause`参照）。
    #[error(
        "insufficient degrees of freedom for panel estimation: n_obs={n_obs}, \
         n_entities={n_entities}{}, k={k} \
         (the panel-adjusted residual degrees of freedom must be positive)",
        n_periods_clause(n_periods)
    )]
    InsufficientDegreesOfFreedom {
        n_obs: usize,
        n_entities: usize,
        n_periods: Option<usize>,
        k: usize,
    },

    /// 観測数1のグループ（singleton）を検出した。エンティティ方向は常に、2-way FEでは
    /// 時点方向も対称に検出する（`panel-api-design.md`6.5節）。
    ///
    /// listwise deletion等の自動除外はせず常にエラーとする（欠損値を常にエラーとする
    /// 全体方針の踏襲）。下流の特異行列エラーとして偶発的に検出される形にはせず、
    /// 「観測数1のグループ」を明示的にこのバリアントで弾く。
    #[error(
        "singleton {dimension} group detected: {dimension} '{group_id}' has only 1 \
         observation. Singleton groups are not dropped automatically; remove them from \
         the input"
    )]
    SingletonGroup {
        dimension: PanelDimension,
        group_id: String,
    },

    /// 2-way FE（entity + time FE）に不均衡（unbalanced）パネルが渡された。
    ///
    /// 2-way FEのwithin変換は閉形式の二重デミーニング
    /// （`ỹ_it = y_it - ȳ_i. - ȳ_.t + ȳ..`）で計算するが、この閉形式は
    /// バランスパネルでのみ正確なため、2-wayでは常にバランスパネルを必須とする
    /// （`panel-api-design.md`6.4節）。1-way FEはエンティティ平均を引くだけで
    /// 不均衡でも数学的に正確に成立するため、このエラーは発生しない。
    ///
    /// なお「バランス」の判定は観測数カウント（`n_obs == n_entities * n_periods`）
    /// だけでは不十分で、あるペアの重複と別ペアの欠落が相殺してカウントだけ一致する
    /// 入力もありうる。このエラーを構築する検証（FE実装issue）は、(entity, time)
    /// ペアの一意性・全組合せの充足まで確認する必要がある。`expected`はカウント上の
    /// 期待値（`n_entities * n_periods`）でメッセージ用。
    #[error(
        "two-way fixed effects requires a balanced panel: got n_obs={n_obs} for \
         n_entities={n_entities} x n_periods={n_periods} (expected {expected} observations)"
    )]
    UnbalancedPanelForTwoWay {
        n_obs: usize,
        n_entities: usize,
        n_periods: usize,
        expected: usize,
    },

    /// within変換（デミーニング）後に分散がゼロになる説明変数がある。
    ///
    /// 時間不変変数（1-way FEで問題になる典型例）だけでなく、2-wayで time FE と
    /// 完全共線な「エンティティ間で変動しない列」も同じチェックで検出できる
    /// （`panel-api-design.md`6.7節。1-way/2-wayで同一ロジックを共有する）。
    #[error(
        "regressor '{column}' has zero variance after the within-transformation \
         (it is time-invariant, or collinear with the fixed effects)"
    )]
    ZeroVarianceAfterDemeaning { column: String },

    /// 2-way FE（entity + time FE）を要求したのに`time`列が指定されていない。
    ///
    /// `time`は`FEOptions`内の`Option`フィールドで、2-way指定時のみ実質必須になる
    /// 「条件付き必須」パターン（`panel-api-design.md`1.1節。`OLSOptions.cluster_col`が
    /// `cov_type="cluster"`のときだけ必須になるのと同型）。未指定時のバリデーション
    /// エラーとしてここで担保する。
    #[error("two-way fixed effects requires the `time` option to be set")]
    TwoWayRequiresTime,

    /// Driscoll-Kraay型パネルHAC（`FeCovType::Hac`、Issue #182、3.1節）を指定したのに
    /// `time`列が指定されていない。
    ///
    /// DKは時点ごとにクロスセクション和を取ってからHACカーネルを適用するため`time`が
    /// 必須（`TwoWayRequiresTime`と同型の「条件付き必須」パターン）。2-way FEは
    /// `within_transform_two_way`/`validate_no_singleton_groups_two_way`の時点で既に
    /// `TwoWayRequiresTime`により`time`必須が担保されているため、このエラーは1-way FEで
    /// `FeCovType::Hac`を指定した場合にのみ発生しうる。
    #[error("Driscoll-Kraay panel HAC requires the `time` option to be set")]
    HacRequiresTime,

    /// `FeCovType::Hac`の明示的な`bandwidth`が`[0, t)`の範囲外（`t`はユニークな時点数）。
    ///
    /// `LeastSquaresError::InvalidHacLags`と同型のバリデーションだが、上限が観測数`n`
    /// ではなく時点数`t`になる点が異なる（DKのバンド幅は「時点のラグ」であり「観測の
    /// ラグ」ではないため、`engine/src/panel/CLAUDE.md`「Driscoll-Kraay型パネルHAC対応」
    /// 参照）。
    #[error("bandwidth must be in the range [0, t): got {bandwidth}, t={t}")]
    InvalidHacBandwidth { bandwidth: i64, t: usize },

    /// within変換済みデータに対する最小二乗推定（`OlsEstimator::fit`への委譲、
    /// `panel-api-design.md`4.3節。WLSがOLSへ委譲するのと同型のパターン）が失敗した。
    ///
    /// 委譲先が返す`LeastSquaresError`をそのまま保持する（IVの`SecondStageFailed
    /// { source }`と同型のラップ）。`#[from]`で透過させず明示的に
    /// `.map_err(|source| PanelError::WithinRegressionFailed { source })`で包むのは、
    /// `CommonError`が`Common`と`WithinRegressionFailed(LeastSquaresError::Common)`の
    /// 2経路で`PanelError`になりうる曖昧さを避け、変換箇所を追跡可能にするため
    /// （`IvError::FirstStageFailed`が`#[from]`を使わない判断と同じ）。
    #[error("within-transformed least-squares estimation failed: {source}")]
    WithinRegressionFailed {
        #[source]
        source: LeastSquaresError,
    },

    /// F統計量（Issue #186）のWald検定（`crate::linear::ols::wald_f_test`への委譲）が
    /// 失敗した。`WithinRegressionFailed`とは別バリアント（理由はモジュールdoc参照）。
    ///
    /// 実際に発生しうるのは`LeastSquaresError::Common(CommonError::ComputationFailed)`
    /// のみ（`wald_f_test`のdocコメント「backstop」参照。傾き係数間の極端なスケール差等で
    /// 共分散部分行列が数値的にほぼ特異な場合）。それでも型は`WithinRegressionFailed`と
    /// 同じ`LeastSquaresError`のまま保持する（`wald_f_test`のエラー型を独自に絞り込む
    /// メリットが無いため）。
    #[error("F-test for joint significance of the slope coefficients failed: {source}")]
    FTestFailed {
        #[source]
        source: LeastSquaresError,
    },

    /// RE（Swamy-Arora分散成分推定、7.1節、Issue #193）のbetween回帰
    /// （エンティティ平均への`OlsEstimator::fit(include_intercept=true)`）が失敗した。
    ///
    /// `WithinRegressionFailed`と同じ`LeastSquaresError`ラップだが、対象がFEのwithin回帰
    /// ではなくREのbetween回帰のため別バリアントにする（`FTestFailed`と同じ判断、
    /// モジュールdoc参照）。最も起こりやすいのはエンティティ数が説明変数の数以下
    /// （`CommonError::InsufficientObservations`）。
    #[error("between-regression least-squares estimation for variance component failed: {source}")]
    BetweenRegressionFailed {
        #[source]
        source: LeastSquaresError,
    },

    /// RE（`ReEstimator::fit`、7.4節、Issue #195）の準偏差変換済みデータへの
    /// `OlsEstimator::fit(include_intercept=false)`委譲が失敗した。
    ///
    /// `WithinRegressionFailed`（FEのwithin変換済みデータ）・`BetweenRegressionFailed`
    /// （REのbetween回帰）とは対象が異なるため別バリアントにする（同じ判断の3件目、
    /// モジュールdoc参照）。
    #[error("quasi-demeaned least-squares estimation for random effects failed: {source}")]
    QuasiDemeanedRegressionFailed {
        #[source]
        source: LeastSquaresError,
    },
}

/// `ids`の値ごとに観測インデックスをまとめる（`BTreeMap`のキー＝`ids`の辞書順）。
///
/// 元々`fe.rs`にFE専用のprivate関数として実装していたが、Issue #193（RE:
/// Swamy-Arora分散成分推定）でREのbetween回帰（エンティティ平均の集計）でも同じ
/// グルーピングが必要になったため、FE/RE間で共有するロジックとしてこちらに移設した
/// （`.claude/rules/rust-style.md`「系統内で共有するロジックは`<系統>/common.rsに置く`」）。
/// `pub(crate)`にする理由: `engine`クレート内部（`fe.rs`・`re.rs`）専用のヘルパーで、
/// `engine_pybind`や`engine`クレート外には公開しない内部実装詳細のため。
///
/// `BTreeMap`を使う理由: `HashMap`だと反復順序がプロセスごとのハッシュシードに依存し、
/// グループ間加算（`Σ_g S_g S_g'`等）の順序・延いては浮動小数点丸め誤差が実行のたびに
/// 変わりうる。FE側ではこれに加え、DKの時点集計でキー順序（`String`の辞書順）がそのまま
/// 時系列順序とみなす規約（`fe.rs`モジュールdoc「Driscoll-Kraay型パネルHAC対応」参照）とも
/// 一致するという二重の意味を持つ。
pub(crate) fn group_indices_by_key(ids: &[String]) -> BTreeMap<&str, Vec<usize>> {
    let mut indices: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, id) in ids.iter().enumerate() {
        indices.entry(id.as_str()).or_default().push(i);
    }
    indices
}

/// `ids`のユニークID数を数える（`n_entities`/`n_periods`のカウント）。純粋な
/// カーディナリティ集計のため`HashSet`でよい（`group_indices_by_key`と異なりグループ間の
/// 加算順序に依存する計算が無いため反復順序非依存）。`group_indices_by_key`と同じ理由
/// （Issue #193）でFE/RE間の共有ロジックとしてここに移設した。
pub(crate) fn count_unique(ids: &[String]) -> usize {
    ids.iter().collect::<HashSet<_>>().len()
}

/// `ids`に現れる全ユニークIDに`θ=1.0`を割り当てた`BTreeMap`を作る。
///
/// `quasi_demean_column`の`theta`引数はエンティティID→θ_iの対応（`&BTreeMap<String,
/// f64>`）を要求するが、θ=1固定の通常のwithin変換（FEのwithin変換そのもの、REの
/// `r_squared_within`計算——`panel-api-design.md`7.4節「FEはθ=1の特殊ケース」・
/// Issue #338「`linearmodels`の`_rsquared`のWithinセクションはRE/FEどちらのモデルでも
/// 共通してθ=1のFE型within変換を使う」参照）で毎回同じ組み立てが必要になるため、
/// FE/RE共有ロジックとしてここに置く（`group_indices_by_key`/`count_unique`と同じ理由、
/// Issue #193で最初にFE→common.rsへ移設した前例に倣い、Issue #338でFE→common.rsへ再移設）。
///
/// 先に`HashSet`でユニークなIDへ絞り込んでから`String`を複製する（`ids.iter().map(|id|
/// (id.clone(), 1.0)).collect()`のように観測順のまま素朴に`collect`すると、`BTreeMap`の
/// 重複キーは値のみ上書きされキー自体は複製されたまま即破棄されるため、観測数`n`分の
/// ヒープ確保が発生してしまう。rust-reviewer指摘、ユニークID数分のみ複製するよう修正済み）。
pub(crate) fn all_ones_theta(ids: &[String]) -> BTreeMap<String, f64> {
    ids.iter()
        .collect::<HashSet<_>>()
        .into_iter()
        .map(|id| (id.clone(), 1.0))
        .collect()
}

/// FE/RE共有のcov_type計算ヘルパー（Issue #197でFE→common.rsへ移設、元は`fe.rs`の
/// `fe_*_cov_params`。`group_indices_by_key`/`count_unique`/`all_ones_theta`と同じ
/// 「FE専用で書いたが後にREでも同じ数式が必要と判明したため共有ロジックとして移設した」
/// 経緯）。**数式自体はFE実装時（Issue #181・#182）のまま変更していない**——移設したのは
/// 呼び出し側（`fe.rs`/`re.rs`）が渡す`df_resid`・`extra_df`・レバレッジの値がFE/REで
/// 異なるだけで、計算ロジック自体はモデル非依存（`.claude/rules/rust-style.md`
/// 「全手法で共有するロジック」）。
///
/// **RE（Issue #197）で判明した重要な事実**: `linearmodels.RandomEffects.fit()`の
/// ソース確認により、REは`cov_type`によらず常に`extra_df=0`を使う（FEのような
/// `neffects`・`entity_nested_within_cluster`の条件分岐が一切不要）。REの変換済み
/// 設計行列は「省略された固定効果ダミー」を持たない（切片も含め全パラメータが実際に
/// 列として含まれている）ため、HC2/HC3のレバレッジも`leverage_full`（FE専用、
/// `fe.rs`に残置）ではなく本モジュールの`leverage_within`（＝素の`h_ii = x_i(X'X)⁻¹x_i'`）
/// で足りる。手動データで`linearmodels`（Classical/HC1/Cluster/HAC）・`plm`
/// （HC2/HC3、`linearmodels`に実装が無いため）との数値一致を実地検証済み
/// （`engine/src/panel/re.rs`のテスト参照、ユーザー確認済み・2026-09-19）。
///
/// within変換後の列（`Vec<Vec<f64>>`、列ごとに長さ`n`）から`faer::Mat`を組み立てる。
/// `OlsInput::from_columns`と同じ列順・行順の規約（`columns[j][i]`がi行j列）。
pub(crate) fn design_matrix_from_columns(columns: &[Vec<f64>], n: usize) -> Mat<f64> {
    let k = columns.len();
    Mat::from_fn(n, k, |i, j| columns[j][i])
}

/// `(X̃'X̃)⁻¹`を求める（`X̃`はFE/REそれぞれの変換後の設計行列）。HC1〜HC3・Clusterいずれの
/// 計算でも共通して必要になる。`ols::xtx_inverse`と同じ発想だが、`OlsEstimator`が
/// 保持する`cov_params`はprivateで再利用できないため独立に計算し直す。
///
/// `X̃'X̃`が対称正定値であることは、`OlsEstimator::fit`が同じ`x`で既に成功している
/// （＝特異ではないと確認済み）ことから理論上保証されるが、`OlsEstimator`と同じく
/// 浮動小数点演算の境界的なケースに備えて`Result`化する。
pub(crate) fn xtx_inverse(x: &Mat<f64>, k: usize) -> Result<Mat<f64>, PanelError> {
    let xtx = x.transpose() * x;
    let llt = xtx.llt(Side::Lower).map_err(|_| {
        CommonError::ComputationFailed(
            "failed to invert the transformed design matrix's Gram matrix for panel cov_type \
             computation"
                .to_string(),
        )
    })?;
    Ok(llt.solve(Mat::<f64>::identity(k, k)))
}

/// 行ごとのレバレッジ `h_ii = x̃_i (X̃'X̃)⁻¹ x̃_i'`（`ols::hc_cov_params`のレバレッジ計算と
/// 同じ式）。FEでは`leverage_full`（`fe.rs`）の材料（`h_within`）として使う。REでは
/// 変換済み設計行列に省略された固定効果ダミーが無いため、この値自体がそのままHC2/HC3の
/// レバレッジになる（モジュールdoc参照）。
pub(crate) fn leverage_within(x: &Mat<f64>, xtx_inv: &Mat<f64>, n: usize, k: usize) -> Vec<f64> {
    let xh = x * xtx_inv;
    (0..n)
        .map(|i| (0..k).map(|j| (*xh.get(i, j)) * (*x.get(i, j))).sum())
        .collect()
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

/// `panel_hc_cov_params`内部でのみ使うHCの種類（`ols::HcVariant`と同型だがHC0を含まない、
/// FE/REともにHC0はスコープ外、`fe.rs`モジュールdoc「`cov_type`対応」参照）。
pub(crate) enum PanelHcVariant {
    Hc1,
    Hc2,
    Hc3,
}

/// HC1〜HC3の係数分散共分散行列（k×k）。`ols::hc_cov_params`と同型の構造だが、
/// 小標本補正がFE/RE用に異なる（`fe.rs`モジュールdoc「`cov_type`対応」参照）:
/// - HC1: `w_i = n/df_resid`
/// - HC2/HC3: `w_i`はレバレッジ`h`ベース（FEは`leverage_full`、REは`leverage_within`を渡す）
///
/// `h`の`expect`（`Hc2`/`Hc3`分岐）は、呼び出し元の`fit()`が`variant=Hc2|Hc3`のときは
/// 必ず`Some(&h)`を渡す構造になっており、`variant`と`h`の組み合わせに呼び出し側のバグ
/// 以外で不整合が生じることはない（`ols::hc_cov_params`の同型の
/// `.expect("Hc2はleverage計算済み")`と同じ「型で表現しきれない呼び出し規約」の防御）。
pub(crate) fn panel_hc_cov_params(
    x: &Mat<f64>,
    residuals: &[f64],
    xtx_inv: &Mat<f64>,
    df_resid: usize,
    h: Option<&[f64]>,
    variant: PanelHcVariant,
) -> Mat<f64> {
    let n = x.nrows();
    let k = x.ncols();
    let hc1_correction = (n as f64 / df_resid as f64).sqrt();

    let x_scaled = Mat::from_fn(n, k, |i, j| {
        let resid = residuals[i];
        let scale = match variant {
            PanelHcVariant::Hc1 => resid * hc1_correction,
            PanelHcVariant::Hc2 => {
                let h = h.expect("Hc2 requires leverage")[i];
                resid / (1.0 - h).sqrt()
            }
            PanelHcVariant::Hc3 => {
                let h = h.expect("Hc3 requires leverage")[i];
                resid / (1.0 - h)
            }
        };
        scale * (*x.get(i, j))
    });

    let psi_hat = x_scaled.transpose() * &x_scaled;
    xtx_inv * &psi_hat * xtx_inv
}

/// クラスターロバスト係数分散共分散行列（k×k）。`ols::cluster_cov_params`と同型の
/// 構造だが、**Stata流の`(G/(G-1))×((n-1)/(n-k))`小標本補正を適用しない**
/// （`linearmodels`との数値一致のため、`fe.rs`モジュールdoc「`cov_type`対応」参照）。
/// 代わりに`n/(n-extra_df-k)`のみを使う（`extra_df`は呼び出し側が決める。FEは
/// `entity_nested_within_cluster`の判定結果、REは常に`0`——モジュールdoc参照）。
pub(crate) fn panel_cluster_cov_params(
    x: &Mat<f64>,
    residuals: &[f64],
    xtx_inv: &Mat<f64>,
    n: usize,
    k: usize,
    groups: &[String],
    extra_df: usize,
) -> Mat<f64> {
    let group_indices = group_indices_by_key(groups);

    let mut s_hat = Mat::<f64>::zeros(k, k);
    for indices in group_indices.values() {
        let mut s_g = vec![0.0_f64; k];
        for &i in indices {
            let e = residuals[i];
            for (a, s_g_a) in s_g.iter_mut().enumerate() {
                *s_g_a += e * (*x.get(i, a));
            }
        }
        for a in 0..k {
            for b in 0..k {
                *s_hat.get_mut(a, b) += s_g[a] * s_g[b];
            }
        }
    }

    let df_resid_for_scale = n - extra_df - k;
    let correction = n as f64 / df_resid_for_scale as f64;
    let cov_uncorrected = xtx_inv * &s_hat * xtx_inv;
    Mat::from_fn(k, k, |i, j| correction * (*cov_uncorrected.get(i, j)))
}

/// `FeCovType::Hac`/`ReCovType::Hac`の`bandwidth`（`Option<i64>`）を実際に使う
/// バンド幅（`usize`）に解決する。
///
/// `Some(bw)`の場合は`0 <= bw < t`を検証してそのまま使う（`t`はユニークな時点数）。`None`の
/// 場合は`linearmodels`の`DriscollKraay`と同じ経験則`floor(4*(t/100)^(2/9))`で自動計算する
/// （`fe.rs`モジュールdoc「Driscoll-Kraay型パネルHAC対応」参照。OLSの`resolve_hac_lags`と
/// 式の形は同じだが、観測数`n`ではなく時点数`t`が引数になる点が異なる）。
pub(crate) fn resolve_dk_bandwidth(bandwidth: Option<i64>, t: usize) -> Result<usize, PanelError> {
    match bandwidth {
        Some(bw) => {
            if bw < 0 || (bw as usize) >= t {
                return Err(PanelError::InvalidHacBandwidth { bandwidth: bw, t });
            }
            Ok(bw as usize)
        }
        None => Ok((4.0 * (t as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize),
    }
}

/// Driscoll-Kraay型パネルHAC共分散行列（k×k、Issue #182・#197）。
///
/// `Ŝ = Σ_t ξ_t ξ_t' + Σ_{l=1}^{bandwidth} w_l (ξ_t ξ_{t-l}' + ξ_{t-l} ξ_t')`
/// （Bartlett重み`w_l = 1 - l/(bandwidth+1)`、`fe.rs`モジュールdoc参照）をまず求め、
/// 最後に`(n/df_resid) × (X̃'X̃)⁻¹ Ŝ (X̃'X̃)⁻¹`にスケールする。`t_periods`（ユニークな
/// 時点数）は`resolve_dk_bandwidth`の呼び出しで既に計算済みの値を呼び出し元からそのまま
/// 受け取る（`time_indices.len()`で二重計算しない）。
///
/// `time`を`group_indices_by_key`で集計して`ξ_t`（時点`t`でのクロスセクション和）を求める。
/// キー順序（`String`の辞書順）がそのまま時系列順序とみなす規約（`fe.rs`モジュールdoc参照）と
/// 一致することを利用している。
///
/// **`bandwidth <= t_periods`（狭義の`<`ではない）が呼び出し元の`resolve_dk_bandwidth`から
/// 保証される**（詳細は`fe.rs`モジュールdoc「Driscoll-Kraay型パネルHAC対応」参照）。
///
/// ラグごとの`k×k`行列積（`xi_top.transpose() * xi_bot`等）は呼び出し元の`fit()`冒頭の
/// `ensure_serial()`が固定したグローバル`Par::Seq`に依存している（OLSの`hac_cov_params`が
/// `matmul(..., Par::Seq)`を明示するのと異なり、本関数は演算子オーバーロードを使うため
/// グローバル設定頼み。`panel_hc_cov_params`/`panel_cluster_cov_params`と同じ流儀）。
pub(crate) fn panel_driscoll_kraay_cov_params(
    x: &Mat<f64>,
    residuals: &[f64],
    xtx_inv: &Mat<f64>,
    time: &[String],
    df_resid: usize,
    bandwidth: usize,
    t_periods: usize,
) -> Mat<f64> {
    let n = x.nrows();
    let k = x.ncols();
    let time_indices = group_indices_by_key(time);

    let mut xi = Mat::<f64>::zeros(t_periods, k);
    for (row, indices) in time_indices.values().enumerate() {
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

    let scale = n as f64 / df_resid as f64;
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
///   （`docs/planning/specs/panel-api-design.md`7.4節: FEはこの関数の`θ_i = 1`の特殊ケース）。
/// - **RE（準偏差変換）**: `θ_i = 1 - sqrt(σ_ε² / (T_i·σ_u² + σ_ε²))`（同7.2節）を渡す。
///
/// FE/REの`fit()`は`y`と`x`の各列にこの関数をループ適用し、変換後の列を
/// `OlsEstimator::fit`へ渡す（WLSがsqrt(w)変換したデータをOLSへ委譲するのと同型の
/// パターン、同4.3節・7.4節）。列ごとに独立な変換のため、列単位の関数として実装し
/// 呼び出し側でループする（`y`/`x`をまとめて受けるより単体テストが単純）。
///
/// # 引数
/// - `col`: 変換対象の列（長さ`n`、行はパネルの観測順）。
/// - `entity`: 各行のエンティティID（長さ`n`）。「グループの同一性だけが意味を持つ列」の
///   ため文字列で扱う（`.claude/rules/rust-style.md`「Python境界でのデータ受け渡し」）。
/// - `theta`: エンティティID → `θ_i`の対応。`entity`に現れる全IDをキーに持つこと。
///
/// # 前提（呼び出し側の契約、`engine`内部でのみ使用）
/// - `entity.len() == col.len()`。`engine_pybind`の列抽出が保証する
///   （`validate_cluster_groups`の`groups.len() == n`契約と同じ位置づけ）。
/// - `theta`は`entity`の全ユニークIDをキーに持つ。RE/FEの`fit()`は同じ`entity`列から
///   `theta`を組み立てるため、欠けは内部実装バグでしか起こり得ない。
/// - `col`は欠損値・非有限値を含まない（`engine`は常にクリーンな値を受け取る前提）。
///
/// 契約違反時は`assert!`/`expect`でpanicする（`Result`は返さない）。ユーザー入力起因の
/// エラーではなく`engine_pybind`〜`engine`間の内部契約違反のため、`validate_cluster_groups`
/// と同じ扱い。
///
/// グループ平均（`ȳ_i.`）は返さない。`fixed_effects()`（6.6節、Issue #184）の`α_i`復元は
/// この関数を拡張せず`fe.rs`側で`FeInput`の元データから独立に再計算する形で実装済み
/// （`engine/src/panel/CLAUDE.md`参照）。σ_ε²再利用（7.4節、RE実装）で平均の保持が
/// 必要になった場合は、その時点で改めて検討する。
///
/// # Panics
/// - `entity.len() != col.len()`
/// - `theta`に`entity`内のいずれかのIDが無い
pub fn quasi_demean_column(
    col: &[f64],
    entity: &[String],
    theta: &BTreeMap<String, f64>,
) -> Vec<f64> {
    assert_eq!(
        entity.len(),
        col.len(),
        "entity length must match column length (engine_pybind contract)"
    );

    // エンティティごとに (合計, 件数) を集約する。ここは`HashMap`でよい（`cluster_cov_params`
    // の`BTreeMap`必須とは事情が異なる）: あるエンティティの和は観測順（＝入力行の固定順）に
    // 積まれ、各行の変換結果もそのエンティティの和だけに依存する。エンティティ「間」を
    // またぐ加算（`Σ_g S_g S_g'`のようにグループ順序が浮動小数点丸めに効く演算）は無いため、
    // 反復順序に関わらずビット単位で決定的。`HashMap`にすることで集約・引き当てが
    // O(n log G) → O(n)（G = エンティティ数）になる（rust-reviewer指摘）。
    let mut sums: HashMap<&str, (f64, usize)> = HashMap::new();
    for (value, id) in col.iter().zip(entity.iter()) {
        let entry = sums.entry(id.as_str()).or_insert((0.0, 0));
        entry.0 += *value;
        entry.1 += 1;
    }

    // エンティティ単位で `θ_i · ȳ_i.`（各行から引く量）を先に求めておく。
    let shift_by_entity: HashMap<&str, f64> = sums
        .iter()
        .map(|(id, (sum, count))| {
            let mean = sum / *count as f64;
            let theta_i = *theta
                .get(*id)
                .expect("theta must contain every entity id present in `entity`");
            (*id, theta_i * mean)
        })
        .collect();

    col.iter()
        .zip(entity.iter())
        .map(|(value, id)| value - shift_by_entity[id.as_str()])
        .collect()
}

/// 古典的ハウスマン検定の統計量・自由度・p値を計算する。
///
/// ```text
/// H = (β_FE - β_RE)' [Var(β_FE) - Var(β_RE)]⁻¹ (β_FE - β_RE)
/// ```
///
/// 帰無仮説 H0: 個体効果と説明変数が無相関（＝REが一致推定量）。棄却されればFEを使う。
/// `H` は自由度 `k`（比較する係数の数）のカイ二乗分布に漸近的に従う。
///
/// `docs/planning/specs/panel-api-design.md` 7.3節:
/// - **v1は classical Hausman のみ**（`cov_type`に依存せず、常にclassical SE前提で計算）。
///   呼び出し側（RE実装）は`cov_type="cluster"`等でfitした場合でも、この関数には
///   classical前提の`cov_fe`/`cov_re`を渡す。
/// - **比較対象はFE/RE間で重なりのあるスロープ係数のみ**。FEには切片が無いため、
///   RE側の切片・時間不変変数の係数は呼び出し側で除外し、対応する順序に揃えた
///   `beta_fe`/`beta_re`（同じ長さ`k`）と、その`k×k`部分共分散行列`cov_fe`/`cov_re`を
///   渡す（このalignは7.3節の通り呼び出し側の責務）。
///
/// # 戻り値
/// `(stat, df, p_value)`。`df == beta_fe.len()`、`p_value` は自由度 `df` のカイ二乗分布の
/// 上側確率 `χ²_df.sf(stat)`（＝ `1 - cdf`。`stat` が大きく H0 を強く棄却する場合に
/// `1.0 - cdf(stat)` だと生じる桁落ちを避けるため、`statrs` の `sf`——正則化上側不完全
/// ガンマの直接計算——を使う。`iv/gmm.rs` 等の既存箇所は `1.0 - cdf` のままで、一括移行は
/// 別issue）。
///
/// **`Var(β_FE) - Var(β_RE)`は理論上は半正定値だが、有限標本では非正定値になり`stat`が
/// 負になりうる**。その場合も`stat`をそのまま返す（`sf` は `stat <= 0` で `1.0` を返すため
/// `p_value == 1.0`）。参照実装 R `plm::phtest`と同じ挙動で、「classical HausmanのPSD仮定が
/// 有限標本で崩れている」ことを示す情報として呼び出し側に委ねる（`panel-api-design.md`
/// 7.3節、Issue #174）。
///
/// `df` には常に `k`（渡された係数の数）を使う。`Var(β_FE) - Var(β_RE)` が閾値は通過するが
/// 実効ランクが `k` 未満のとき、`stat` と `df` に不整合が生じうる（R `plm::phtest` も同じ
/// 制約）。
///
/// # Errors
/// `Var(β_FE) - Var(β_RE)`が数値的に特異（`col_piv_qr`のR対角成分が相対閾値以下、
/// またはNaN）で逆行列が計算できない場合に`CommonError::ComputationFailed`。
/// `ChiSquared::new`の失敗（`df`が非正、`k >= 1`のため理論上到達不能）も同じ。
///
/// # Panics
/// 呼び出し側の契約違反時（`engine`内部でのみ使用、`validate_cluster_groups`と同じ扱い）:
/// - `beta_fe.len() != beta_re.len()`、または長さが0
/// - `cov_fe`/`cov_re`が`k×k`でない
pub fn hausman_statistic(
    beta_fe: &[f64],
    cov_fe: &[Vec<f64>],
    beta_re: &[f64],
    cov_re: &[Vec<f64>],
) -> Result<(f64, usize, f64), CommonError> {
    let k = beta_fe.len();
    assert_eq!(
        k,
        beta_re.len(),
        "beta_fe and beta_re must have the same length (caller aligns overlapping slopes)"
    );
    assert!(
        k >= 1,
        "hausman_statistic requires at least one compared coefficient"
    );
    assert!(
        cov_fe.len() == k && cov_fe.iter().all(|row| row.len() == k),
        "cov_fe must be a k x k matrix matching beta_fe"
    );
    assert!(
        cov_re.len() == k && cov_re.iter().all(|row| row.len() == k),
        "cov_re must be a k x k matrix matching beta_re"
    );

    let d = Mat::from_fn(k, 1, |i, _| beta_fe[i] - beta_re[i]);
    let cov_diff = Mat::from_fn(k, k, |i, j| cov_fe[i][j] - cov_re[i][j]);

    // `cov_diff`は対称だが（有限標本では）正定値とは限らないため、Choleskyではなく
    // 列ピボットQRで解く（`nonlinear::common::newton_step`と同じ方針・同じ相対閾値での
    // 特異性検出。NaNは`diag <= threshold`をすり抜けるため明示的にチェックする）。
    let qr = cov_diff.col_piv_qr();
    let r = qr.thin_R();
    let max_abs_diag = (0..k).map(|i| (*r.get(i, i)).abs()).fold(0.0_f64, f64::max);
    let threshold = (k as f64) * f64::EPSILON * max_abs_diag;
    for i in 0..k {
        let diag = (*r.get(i, i)).abs();
        if diag.is_nan() || diag <= threshold {
            return Err(CommonError::ComputationFailed(
                "the Hausman variance difference Var(beta_FE) - Var(beta_RE) is singular \
                 and cannot be inverted"
                    .to_string(),
            ));
        }
    }

    let z = qr.solve_lstsq(&d);
    let stat: f64 = (0..k).map(|i| (*d.get(i, 0)) * (*z.get(i, 0))).sum();

    let chi2 =
        ChiSquared::new(k as f64).map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
    // `1.0 - chi2.cdf(stat)` ではなく `sf`（正則化上側不完全ガンマの直接計算）を使う。
    // 大きい `stat` で `cdf ≈ 1` になり小さいp値の相対精度が失われるのを避けるため。
    // `stat <= 0` では `sf` も `1.0` を返すので、統計量が負のときの挙動は変わらない。
    let p_value = chi2.sf(stat);

    Ok((stat, k, p_value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panel_dimension_displays_lowercase_name() {
        assert_eq!(PanelDimension::Entity.to_string(), "entity");
        assert_eq!(PanelDimension::Time.to_string(), "time");
    }

    #[test]
    fn panel_error_messages_are_human_readable() {
        assert_eq!(
            PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Entity,
                y_rows: 10,
                other_rows: 8,
            }
            .to_string(),
            "dimension mismatch: y has 10 rows but entity has 8 rows"
        );
        assert_eq!(
            PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                y_rows: 10,
                other_rows: 8,
            }
            .to_string(),
            "dimension mismatch: y has 10 rows but time has 8 rows"
        );
        assert_eq!(
            PanelError::InsufficientDegreesOfFreedom {
                n_obs: 10,
                n_entities: 8,
                n_periods: None,
                k: 2,
            }
            .to_string(),
            "insufficient degrees of freedom for panel estimation: n_obs=10, \
             n_entities=8, k=2 \
             (the panel-adjusted residual degrees of freedom must be positive)"
        );
        assert_eq!(
            PanelError::InsufficientDegreesOfFreedom {
                n_obs: 20,
                n_entities: 5,
                n_periods: Some(4),
                k: 3,
            }
            .to_string(),
            "insufficient degrees of freedom for panel estimation: n_obs=20, \
             n_entities=5, n_periods=4, k=3 \
             (the panel-adjusted residual degrees of freedom must be positive)"
        );
        assert_eq!(
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "firm_42".to_string(),
            }
            .to_string(),
            "singleton entity group detected: entity 'firm_42' has only 1 observation. \
             Singleton groups are not dropped automatically; remove them from the input"
        );
        assert_eq!(
            PanelError::SingletonGroup {
                dimension: PanelDimension::Time,
                group_id: "2020".to_string(),
            }
            .to_string(),
            "singleton time group detected: time '2020' has only 1 observation. \
             Singleton groups are not dropped automatically; remove them from the input"
        );
        assert_eq!(
            PanelError::UnbalancedPanelForTwoWay {
                n_obs: 39,
                n_entities: 10,
                n_periods: 4,
                expected: 40,
            }
            .to_string(),
            "two-way fixed effects requires a balanced panel: got n_obs=39 for \
             n_entities=10 x n_periods=4 (expected 40 observations)"
        );
        assert_eq!(
            PanelError::ZeroVarianceAfterDemeaning {
                column: "female".to_string(),
            }
            .to_string(),
            "regressor 'female' has zero variance after the within-transformation \
             (it is time-invariant, or collinear with the fixed effects)"
        );
        assert_eq!(
            PanelError::TwoWayRequiresTime.to_string(),
            "two-way fixed effects requires the `time` option to be set"
        );
        assert_eq!(
            PanelError::WithinRegressionFailed {
                source: LeastSquaresError::SingularMatrix,
            }
            .to_string(),
            "within-transformed least-squares estimation failed: design matrix is \
             singular (perfect multicollinearity detected)"
        );
        // `#[error(transparent)]`: `Common`は内側`CommonError`のメッセージを
        // そのまま透過する（`IvError`/`MleError`のテストと同じ確認）。
        assert_eq!(
            PanelError::Common(CommonError::InsufficientClusters { g: 1 }).to_string(),
            "cov_type='cluster' requires at least 2 clusters, got 1"
        );
    }

    #[test]
    fn panel_error_implements_partial_eq() {
        assert_eq!(
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "a".to_string(),
            },
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "a".to_string(),
            }
        );
        assert_ne!(
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "a".to_string(),
            },
            PanelError::SingletonGroup {
                dimension: PanelDimension::Time,
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
            Err(CommonError::InsufficientClusters { g: 1 })
        }
        fn outer() -> Result<(), PanelError> {
            inner()?;
            Ok(())
        }
        assert_eq!(
            outer().unwrap_err(),
            PanelError::Common(CommonError::InsufficientClusters { g: 1 })
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
    fn between_regression_failed_message_and_equality() {
        let err = PanelError::BetweenRegressionFailed {
            source: LeastSquaresError::Common(CommonError::InsufficientObservations { n: 2, k: 3 }),
        };
        assert_eq!(
            err.to_string(),
            "between-regression least-squares estimation for variance component failed: \
             insufficient observations: n=2 must be greater than k=3 \
             (number of independent variables, including the intercept)"
        );
        assert_eq!(
            err,
            PanelError::BetweenRegressionFailed {
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
            source: LeastSquaresError::SingularMatrix,
        };
        assert_eq!(
            err.to_string(),
            "quasi-demeaned least-squares estimation for random effects failed: design matrix \
             is singular (perfect multicollinearity detected)"
        );
        assert_eq!(
            err,
            PanelError::QuasiDemeanedRegressionFailed {
                source: LeastSquaresError::SingularMatrix,
            }
        );
    }

    // ── quasi_demean_column ────────────────────────────────────────────────

    /// `["a", "a", "b", "b", "b"]`のエンティティ列を作るヘルパ。
    fn entities(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    fn theta_map(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn quasi_demean_column_with_theta_one_is_the_within_transformation() {
        // FE相当: θ_i = 1.0。col[i] - ȳ_i. になる。
        // a: mean = (10 + 20) / 2 = 15、b: mean = (3 + 6 + 9) / 3 = 6。
        let entity = entities(&["a", "a", "b", "b", "b"]);
        let col = [10.0, 20.0, 3.0, 6.0, 9.0];
        let theta = theta_map(&[("a", 1.0), ("b", 1.0)]);

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
        let theta = theta_map(&[("a", 0.0), ("b", 0.0)]);

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
        let theta = theta_map(&[("a", 0.5), ("b", 1.0)]);

        let out = quasi_demean_column(&col, &entity, &theta);

        assert_eq!(out, vec![2.5, 12.5, -3.0, 0.0, 3.0]);
    }

    #[test]
    fn quasi_demean_column_handles_unbalanced_panel() {
        // 不均衡パネル（T_a = 1, T_b = 3）でも1-wayは各エンティティ平均を引くだけで
        // 正確に成立する（`panel-api-design.md`6.4節）。
        // a: mean = 4.0（単一観測）→ θ_a = 1.0 で 0.0 になる。
        // b: mean = (2 + 4 + 6) / 3 = 4.0。
        let entity = entities(&["a", "b", "b", "b"]);
        let col = [4.0, 2.0, 4.0, 6.0];
        let theta = theta_map(&[("a", 1.0), ("b", 1.0)]);

        let out = quasi_demean_column(&col, &entity, &theta);

        assert_eq!(out, vec![0.0, -2.0, 0.0, 2.0]);
    }

    #[test]
    fn quasi_demean_column_preserves_length_and_row_order() {
        // エンティティがブロックにまとまっていない（インターリーブした）配置でも、
        // 行ごとに所属エンティティの平均を引く。出力長・行順は入力どおり。
        let entity = entities(&["x", "y", "x", "y", "x"]);
        let col = [1.0, 100.0, 2.0, 200.0, 3.0];
        let theta = theta_map(&[("x", 1.0), ("y", 1.0)]);

        let out = quasi_demean_column(&col, &entity, &theta);

        // x: mean = 2.0、y: mean = 150.0
        assert_eq!(out, vec![-1.0, -50.0, 0.0, 50.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "entity length must match column length")]
    fn quasi_demean_column_panics_on_length_mismatch() {
        let entity = entities(&["a", "b"]);
        let col = [1.0, 2.0, 3.0];
        let theta = theta_map(&[("a", 1.0), ("b", 1.0)]);
        let _ = quasi_demean_column(&col, &entity, &theta);
    }

    #[test]
    #[should_panic(expected = "theta must contain every entity id")]
    fn quasi_demean_column_panics_when_theta_missing_an_entity() {
        let entity = entities(&["a", "a", "b"]);
        let col = [1.0, 2.0, 3.0];
        let theta = theta_map(&[("a", 1.0)]); // "b" が欠けている
        let _ = quasi_demean_column(&col, &entity, &theta);
    }

    #[test]
    fn quasi_demean_column_single_group_with_theta_one_is_all_zero() {
        // 全行が同一エンティティ（グループが1つだけ）＋ θ=1 → 全行がグループ平均に
        // 一致するため出力は全ゼロ（この列だけでは within 変換後に情報が残らない。
        // 6.7節の分散ゼロ検証・6.5節のsingleton検証は消費側 fe.rs の責務）。
        let entity = entities(&["a", "a", "a", "a"]);
        let col = [3.0, 5.0, 7.0, 9.0]; // mean = 6.0
        let theta = theta_map(&[("a", 1.0)]);

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
        let theta = theta_map(&[("a", -0.5), ("b", 2.0)]);

        let out = quasi_demean_column(&col, &entity, &theta);

        assert_eq!(out, vec![13.0, 17.0, -25.0, -15.0]);
    }

    // ── hausman_statistic ─────────────────────────────────────────────────

    #[test]
    fn hausman_statistic_is_zero_when_estimates_coincide() {
        // β_FE == β_RE → d = 0 → H = 0、df = k、p_value = 1.0（H0を棄却しない）。
        let beta_fe = [1.0, 2.0];
        let beta_re = [1.0, 2.0];
        let cov_fe = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let cov_re = vec![vec![0.5, 0.0], vec![0.0, 0.5]];

        let (stat, df, p_value) = hausman_statistic(&beta_fe, &cov_fe, &beta_re, &cov_re).unwrap();

        assert_eq!(stat, 0.0);
        assert_eq!(df, 2);
        assert_eq!(p_value, 1.0);
    }

    #[test]
    fn hausman_statistic_rejects_h0_for_large_divergent_estimates() {
        // d = [1, -1]、cov_diff = diag(0.1, 0.1) → inv = diag(10, 10)。
        // H = 10·1² + 10·(-1)² = 20。df = 2。
        // χ²_2 の上側確率は閉形式 sf(x) = exp(-x/2) なので p = exp(-10) ≈ 4.5400e-5。
        // `sf` を使うことでこの小さいp値が相対精度を保って得られる（`1 - cdf` だと
        // cdf ≈ 1 で桁落ちする）。
        let beta_fe = [2.0, -1.0];
        let beta_re = [1.0, 0.0];
        let cov_fe = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let cov_re = vec![vec![0.9, 0.0], vec![0.0, 0.9]];

        let (stat, df, p_value) = hausman_statistic(&beta_fe, &cov_fe, &beta_re, &cov_re).unwrap();

        assert!((stat - 20.0).abs() < 1e-10, "stat = {stat}");
        assert_eq!(df, 2);
        // `sf` を直接使っていること（`1 - cdf` へ退行していないこと）: 返された `stat` に
        // 対する χ²_2 の `sf` とビット単位で一致する（`1.0 - cdf(stat)` なら一致しない）。
        assert_eq!(p_value, ChiSquared::new(2.0).unwrap().sf(stat));
        // 数値の正しさ: χ²_2 の上側確率は閉形式 exp(-x/2) なので p ≈ exp(-10) ≈ 4.54e-5。
        assert!(
            (p_value - (-10.0_f64).exp()).abs() < 1e-12,
            "p_value = {p_value}"
        );
    }

    #[test]
    fn hausman_statistic_full_quadratic_form_with_off_diagonal_covariance() {
        // cov_diff = [[1.0, 0.5], [0.5, 1.0]] → inv = (4/3)·[[1, -0.5], [-0.5, 1]]。
        // d = [1, 1] なので H = inv の全要素和 = 4/3。
        let beta_fe = [3.0, 4.0];
        let beta_re = [2.0, 3.0];
        let cov_fe = vec![vec![2.0, 0.5], vec![0.5, 2.0]];
        let cov_re = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let (stat, df, _p_value) = hausman_statistic(&beta_fe, &cov_fe, &beta_re, &cov_re).unwrap();

        assert!((stat - 4.0 / 3.0).abs() < 1e-10, "stat = {stat}");
        assert_eq!(df, 2);
    }

    #[test]
    fn hausman_statistic_single_coefficient() {
        // k = 1: d = 2、cov_diff = [[1.0]] → H = 2·1·2 = 4。df = 1。χ²_1 の p ≈ 0.0455。
        let (stat, df, p_value) =
            hausman_statistic(&[3.0], &[vec![2.0]], &[1.0], &[vec![1.0]]).unwrap();

        assert!((stat - 4.0).abs() < 1e-10, "stat = {stat}");
        assert_eq!(df, 1);
        assert!(
            (p_value - 0.045_500_263_9).abs() < 1e-6,
            "p_value = {p_value}"
        );
    }

    #[test]
    fn hausman_statistic_returns_negative_stat_when_variance_diff_is_indefinite() {
        // cov_diff = [[-1.0, 0.0], [0.0, 0.5]]（非正定値だが可逆）→ inv = [[-1, 0], [0, 2]]。
        // d = [1, 0] → H = -1·1² = -1。有限標本でのPSD仮定崩れ。plm::phtest と同じく
        // そのまま返し、p_value は 1.0 になる（エラーにしない）。
        let beta_fe = [2.0, 5.0];
        let beta_re = [1.0, 5.0];
        let cov_fe = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let cov_re = vec![vec![2.0, 0.0], vec![0.0, 0.5]];

        let (stat, df, p_value) = hausman_statistic(&beta_fe, &cov_fe, &beta_re, &cov_re).unwrap();

        assert!((stat - (-1.0)).abs() < 1e-10, "stat = {stat}");
        assert_eq!(df, 2);
        assert_eq!(p_value, 1.0);
    }

    #[test]
    fn hausman_statistic_errors_when_variance_diff_is_singular() {
        // cov_fe == cov_re → cov_diff = 0 → 特異で逆行列が計算できない。
        let beta_fe = [2.0, 1.0];
        let beta_re = [1.0, 0.0];
        let cov = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let result = hausman_statistic(&beta_fe, &cov, &beta_re, &cov);

        assert!(matches!(result, Err(CommonError::ComputationFailed(_))));
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn hausman_statistic_panics_on_beta_length_mismatch() {
        let _ = hausman_statistic(
            &[1.0, 2.0],
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            &[1.0],
            &[vec![1.0]],
        );
    }

    #[test]
    #[should_panic(expected = "at least one compared coefficient")]
    fn hausman_statistic_panics_on_empty_beta() {
        let _ = hausman_statistic(&[], &[], &[], &[]);
    }

    #[test]
    #[should_panic(expected = "cov_fe must be a k x k matrix")]
    fn hausman_statistic_panics_when_cov_fe_is_not_k_by_k() {
        // k = 2 だが cov_fe が 1x1。
        let _ = hausman_statistic(
            &[1.0, 2.0],
            &[vec![1.0]],
            &[0.5, 1.0],
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
        );
    }

    #[test]
    #[should_panic(expected = "cov_re must be a k x k matrix")]
    fn hausman_statistic_panics_when_cov_re_row_length_is_wrong() {
        // k = 2、cov_re の行数は 2 だが 1 行の長さが 1。
        let _ = hausman_statistic(
            &[1.0, 2.0],
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            &[0.5, 1.0],
            &[vec![1.0, 0.0], vec![0.0]],
        );
    }
}
