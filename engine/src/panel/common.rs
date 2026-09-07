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
//! - `InsufficientDegreesOfFreedom`: パネル自由度調整（6.3節）
//! - `SingletonGroup`: 観測数1のグループ（6.5節）
//! - `UnbalancedPanelForTwoWay`: 2-way FEのバランスパネル必須（6.4節）
//! - `ZeroVarianceAfterDemeaning`: within変換後に分散ゼロの説明変数（6.7節）
//! - `TwoWayRequiresTime`: 2-way FE指定時の`time`必須（1.1節）
//! - `WithinRegressionFailed`: within変換済みデータの最小二乗推定委譲の失敗（4.3節）
//!
//! RE固有（7章）で追加のバリアントが必要になった場合は、FE/RE実装issueで実際に計算
//! コードを書く過程で随時追加する（`LeastSquaresError`・`IvError`のdocコメントと同じ
//! 「土台を用意し、必要になった時点で足す」方針）。想定される追加候補:
//!
//! - between回帰（エンティティ平均に対するOLS、σ_u²推定）の自由度不足。分母は
//!   `n_entities - k`（7.1節）で、`InsufficientDegreesOfFreedom`（within側／パネル調整後
//!   残差自由度）とは別軸のため専用バリアントになる見込み。
//! - classical Hausman統計量（7.3節）で`cov_fe - cov_re`が非正定値になり統計量が負に
//!   なるケースの扱い（エラーにするか`None`にするか未確定。2.4節の「FE推定失敗時は
//!   `None`」とは別軸）。

use std::collections::BTreeMap;
use std::fmt;

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
    /// `time`は`FeOptions`内の`Option`フィールドで、2-way指定時のみ実質必須になる
    /// 「条件付き必須」パターン（`panel-api-design.md`1.1節。`OLSOptions.cluster_col`が
    /// `cov_type="cluster"`のときだけ必須になるのと同型）。未指定時のバリデーション
    /// エラーとしてここで担保する。
    #[error("two-way fixed effects requires the `time` option to be set")]
    TwoWayRequiresTime,

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
/// グループ平均（`ȳ_i.`）は返さない。`fixed_effects()`（6.6節）の`α_i`復元や
/// σ_ε²再利用（7.4節）で平均の保持が必要になった場合は、そのFE/RE実装issueで
/// この関数を拡張する（現時点ではスコープ外）。
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

    // エンティティごとに (合計, 件数) を集約する。集約は`BTreeMap`で行う
    // （`engine/src/linear/CLAUDE.md`「クラスターのグループ化は`BTreeMap`」——
    // グループ単位の反復順序が浮動小数点加算の非結合性で結果に効く箇所の統一方針。
    // 本関数は各エンティティの和を観測順に積むため厳密には反復順序非依存だが、
    // パネル系のグループ集約はこの方針で揃える）。
    let mut sums: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
    for (value, id) in col.iter().zip(entity.iter()) {
        let entry = sums.entry(id.as_str()).or_insert((0.0, 0));
        entry.0 += *value;
        entry.1 += 1;
    }

    // エンティティ単位で `θ_i · ȳ_i.`（各行から引く量）を先に求めておく。
    let shift_by_entity: BTreeMap<&str, f64> = sums
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
}
