//! FEの入力データ型（`FeInput`）とwithin変換（1-way/2-way、Issue #176）。
//!
//! `engine`はpolars/PyO3を知らない（`.claude/rules/rust-style.md`「責務分離」）。
//! `engine_pybind`がpolars DataFrameから`y`/`x`/`entity`/`time`を列ごとに抽出し
//! （`entity`/`time`はグループの同一性だけが意味を持つ列のため文字列で抽出する、
//! 同「Python境界でのデータ受け渡し」）、それらの列を本モジュールの
//! `FeInput::from_columns`に渡す。
//!
//! `FeInput`自体はwithin変換前の生データを保持するだけの入れ物であり、`OlsInput`/
//! `IvInput`と異なり`faer::Mat`は組み立てない（within変換（`panel::common::
//! quasi_demean_column`）が`&[f64]`の列単位で動く設計のため、`Mat`に詰め直す
//! 変換をこの段階で行う意味が無い。`docs/planning/specs/panel-api-design.md`7.4節）。
//!
//! ## within変換（`within_transform_one_way`/`within_transform_two_way`）
//!
//! - **1-way**（`docs/planning/specs/panel-api-design.md`6.1節）: `y`/各`x`列に
//!   entityでのquasi-demean（θ=1、`col[i] - ȳ_{e(i)}.`）を適用する。不均衡パネルも
//!   無条件でサポートする（エンティティごとの平均を引くだけで数学的に正確に成立する
//!   ため）。
//! - **2-way**（同6.2節・6.4節）: 閉形式の二重デミーニング
//!   `ỹ_it = y_it - ȳ_i. - ȳ_.t + ȳ..`で計算する。この閉形式は**バランスパネルでのみ
//!   正確**なため、事前にバランスパネルであることを検証し
//!   （`PanelError::UnbalancedPanelForTwoWay`）、`time`が指定されていなければ
//!   `PanelError::TwoWayRequiresTime`を返す。
//!   - **実装はentityでquasi-demeanした結果をさらにtimeでquasi-demeanする2段階適用**
//!     （`quasi_demean_column`をentity・time双方に順に適用するだけで、専用の二重
//!     デミーニング式を別途実装しない）。この2段階適用がバランスパネルで閉形式と
//!     数学的に一致することの導出: エンティティ数`N`・時点数`T`のバランスパネル
//!     （`n=NT`）で、entity-demean後の列を`e_it = y_it - ȳ_i.`とすると、
//!     `(1/N)Σ_i e_it = ȳ_.t - (1/N)Σ_i ȳ_i. = ȳ_.t - ȳ..`（バランスパネルでは
//!     `(1/N)Σ_i ȳ_i. = ȳ..`が成り立つ——各エンティティの観測数がすべて`T`で
//!     等しいため）。したがって`e_it`をtimeでquasi-demeanすると
//!     `e_it - (ȳ_.t - ȳ..) = y_it - ȳ_i. - ȳ_.t + ȳ..`となり閉形式と一致する。
//!     不均衡パネルではこの等式が成り立たない（`(1/N)Σ_i ȳ_i. ≠ ȳ..`となりうる）ため、
//!     2-wayを不均衡パネルに適用してはならない（6.4節がバランスパネルを必須にする
//!     所以）。
//!
//! ## 分散ゼロ説明変数の検出（`validate_no_zero_variance_regressors`、Issue #177）
//!
//! within変換後の設計行列の各列の分散を確認し、ゼロの列があれば
//! `PanelError::ZeroVarianceAfterDemeaning`を返す（6.7節）。1-way/2-way共通ロジック
//! （`within_transform_one_way`/`within_transform_two_way`のどちらの出力にも適用できる、
//! `column_is_zero_variance`関数doc参照）。時間不変変数（1-way）だけでなく、2-wayで
//! time FEと完全共線な「エンティティ間で変動しない列」も同じチェックで検出できる。
//!
//! ## singleton検出（`validate_no_singleton_groups_one_way`/`validate_no_singleton_groups_two_way`、
//! Issue #179）
//!
//! 観測数1のグループ（singleton）を明示的に検出し`PanelError::SingletonGroup`を返す
//! （6.5節）。自動除外はしない。**下流の特異行列エラーとして偶発的に検出される形には
//! しない**——singletonのエンティティ/時点はwithin変換後にその行が全列ゼロになり
//! `OlsEstimator::fit`側で特異行列として（間接的に、かつ原因の分かりにくいエラー
//! メッセージで）検出されうるが、6.5節はこれを避け、within変換の**前**に生の
//! `entity`/`time`列から直接カウントして専用のバリデーションエラーにすることを要求する。
//! - **1-way**: entityのみ検出（`validate_no_singleton_groups_one_way`）。
//! - **2-way**: entity・time双方を対称に検出する（`validate_no_singleton_groups_two_way`。
//!   `within_transform_two_way`と同様、`time`が`None`なら`PanelError::TwoWayRequiresTime`）。
//! - 複数のsingletonグループが存在する場合は、観測順で最初に現れるグループのみを
//!   報告する（`validate_no_zero_variance_regressors`の「最初の1件を報告」方針と統一）。
//!
//! ## `OlsEstimator`への委譲（`FeEstimator`、Issue #178、4.3節）
//!
//! FEは**まず`OlsEstimator`への委譲を試す**（within変換したデータを`OlsEstimator::fit`に
//! 渡す、`WlsEstimator`と同型のパターン。`docs/planning/specs/panel-api-design.md`4.3節）。
//! `FeEstimator::fit`は「singleton検出→within変換（2-wayはバランスパネル検証も内包）→
//! 分散ゼロ検出→`OlsEstimator::fit`」の順にパイプラインを実行する。
//!
//! **`FeEstimator::fit`はwithin推定量`β̂`の委譲に加えて、自由度調整（Issue #180、6.3節）
//! まで実装している**。within変換後のOLS推定量`β̂`はwithin推定量として数学的に正しい値に
//! なる（自由度・`cov_type`に依存しない）ため、委譲だけで正しく求まる。一方、以下は
//! FE固有の再計算・補正が必要で**別issueで対応する**（4.3節。WLSがR²等を素のOLS計算の
//! ままでは使わなかったのと同じ教訓）:
//! - `cov_type`デフォルトのentity単位cluster化・Driscoll-Kraay型HAC（3章・6.8節・Issue #181）
//! - パネル固有R²（within/between/overall、2章・Issue #183。`r_squared_adj`とは別物、
//!   下記参照）
//!
//! そのため`FeEstimator::fit`は`OlsEstimator::fit`を`CovType::Classical`固定で呼ぶ
//! （`cov_type`補正が入るまでの暫定値。`estimator().std_errors()`等の**t検定・F検定関連
//! フィールドは`cov_type=Classical`前提でのみ正しい**）。`OlsInput::from_columns`は
//! `include_intercept=false`で呼ぶ（within変換で全体平均も含めて差し引かれているため、
//! 変換後データに切片は不要——`OlsEstimator::fit`が変換後の残差平均をゼロと仮定する
//! 通常のOLSと同じ考え方）。
//!
//! `OlsInput::from_columns`/`OlsEstimator::fit`が返す`LeastSquaresError`は
//! `PanelError::WithinRegressionFailed { source }`に包む（`common.rs`のdocコメント参照）。
//!
//! `FeEffects`（`OneWay`/`TwoWay`）で1-way/2-wayを切り替える。将来`FeOptions`
//! （Issue #186）が導入されたら、その一部（またはそのままのフィールド型）として
//! 統合する想定の暫定的なパラメータ（1-way/2-wayの区別自体は`panel-api-design.md`で
//! 確定済みの設計だが、`FeOptions`自体は未着手のため）。
//!
//! ## 自由度調整（Issue #180、6.3節）
//!
//! `df_model = k + neffects`（`neffects`は1-wayなら`n_entities`、2-wayなら
//! `n_entities + n_periods - 1`。entityダミー・timeダミー間の定数項ぶんの重複を`+1`で
//! 補正する、6.3節）。`df_resid = n - df_model`。`n <= df_model`なら
//! `PanelError::InsufficientDegreesOfFreedom`。
//!
//! **`OlsEstimator`自身のt検定・調整済みR²・AIC/BICは`df_resid_ols = n - k`（`k`のみ、
//! `neffects`を知らない）を前提に計算されているため誤り**（WLSの教訓と同型）。
//! `FeEstimator::fit`は以下を委譲後に再計算し、上書きする:
//! - **標準誤差・t値・p値・信頼区間**: `σ̂²`（残差分散）の分母が`df_resid_ols`から
//!   `df_resid`に変わるだけなので、`OlsEstimator`が計算済みの標準誤差を
//!   `sqrt(df_resid_ols / df_resid)`倍にスケールし直し（`cov_params`の再構築が不要）、
//!   t分布（自由度`df_resid`）でt値・p値・信頼区間を計算し直す（`crate::inference`の
//!   共有ヘルパーを使う、OLS自身と同じロジック）。
//! - **調整済みR²**: `OlsEstimator::r_squared()`（within R²、変換後の`y`の
//!   uncentered TSSベース）ではなく、**変換前の元の`y`の中心化TSSに基づく overall R²**
//!   を使う——`r_squared_adj = 1 - (1 - overall_R²) * (n-1) / df_resid`。`estimator()`の
//!   `r_squared()`は依然として妥当な値（within R²）だが、`r_squared_adj`とは別の
//!   概念であることに注意（within R²を`(n-1)/df_resid`で素朴に調整しても正しい
//!   調整済みR²にはならない）。
//! - **AIC/BIC**: `log_likelihood`自体は`SSR/n`のみに依存し`df_resid`非依存の式
//!   （`OlsEstimator::log_likelihood()`のformulaと同一）のためそのまま再利用できるが、
//!   ペナルティ項の乗数は`k`ではなく`df_model`（固定効果の実効パラメータ数を含む）を使う:
//!   `aic = -2*log_likelihood + 2*df_model`、`bic = -2*log_likelihood + ln(n)*df_model`。
//! - **F統計量はこの時点では未対応**（issue本文が明示的に「検定統計量（t検定）」と
//!   限定しているため、v1のスコープ外。`estimator().f_statistic()`/`f_p_value()`は
//!   `df_resid_ols`ベースのまま、FE用に補正されていない）。
//!
//! **検証の例外**: Python主リファレンスの`linearmodels`（`PanelOLS`）は`rsquared_adj`・
//! `aic`・`bic`を一切提供しない（`rsquared_within`/`between`/`overall`/`inclusive`・
//! `loglik`のみ）。そのためこれら3つの検証はRクロスチェック（`fixest`）のみで行う
//! （通常の「Python主リファレンス＋Rクロスチェックの2系統検証」の例外、ハウスマン検定
//! （5.3節）と同型の判断）。上記の式は`fixest::feols`の`AIC()`/`BIC()`/`summary()`の
//! `Adj. R2`と数値的に一致することをRで実地検証済み（ユーザー承認済み、2026-09-12）。

use std::collections::{BTreeMap, HashMap, HashSet};

use faer::Mat;
use statrs::distribution::StudentsT;

use crate::error::CommonError;
use crate::inference;
use crate::linear::ols::{CovType, OlsEstimator, OlsInput};
use crate::panel::common::{PanelDimension, PanelError, quasi_demean_column};

/// FEの被説明変数・説明変数・パネル識別子を保持する入力データ。
///
/// within変換前の生データを保持するだけの入れ物（`Mat`を組み立てない理由はモジュール
/// doc参照）。フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」）。
/// `from_columns`で構築した後はgetter経由でのみアクセスする。
#[derive(Debug)]
pub struct FeInput {
    /// 被説明変数（長さ`n`、行はパネルの観測順）。
    y: Vec<f64>,
    /// 説明変数（各列は長さ`n`）。within変換前の生の値。
    x: Vec<Vec<f64>>,
    /// 説明変数名。`x`の列と対応する。
    x_names: Vec<String>,
    /// 各行のエンティティID（長さ`n`）。
    entity: Vec<String>,
    /// 各行の時点ID（長さ`n`）。2-way FE（entity + time FE）を指定しない場合は`None`
    /// （`panel-api-design.md`1.1節: `time`は`FeOptions`内の条件付き必須オプション）。
    time: Option<Vec<String>>,
    /// 被説明変数名。
    dep_var_name: String,
}

impl FeInput {
    /// 列ごとの`Vec<f64>`/`Vec<String>`（`engine_pybind`がpolars DataFrameから抽出済み）
    /// から`FeInput`を組み立てる。
    ///
    /// # Errors
    /// - いずれかの`x_columns`の長さが`y`と一致しない場合は
    ///   `PanelError::Common(CommonError::DimensionMismatch)`
    /// - `entity`の長さが`y`と一致しない場合は
    ///   `PanelError::IdentifierDimensionMismatch { dimension: PanelDimension::Entity, .. }`
    /// - `time`が`Some`で、その長さが`y`と一致しない場合は
    ///   `PanelError::IdentifierDimensionMismatch { dimension: PanelDimension::Time, .. }`
    ///
    /// within変換の実施・singleton検出・分散ゼロ検証・バランスパネルの検証は行わない
    /// （いずれも別issueで`fit()`側が担う、`panel-api-design.md`6章）。
    ///
    /// # パニックについて
    /// `x_names.len() != x_columns.len()`の場合は`debug_assert!`でパニックする
    /// （`OlsInput::from_columns`と同じ理由: 呼び出し側`engine_pybind`の実装バグでしか
    /// 起こり得ない内部契約であり、実データに起因する`ValidationError`とは性質が異なる）。
    pub fn from_columns(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        entity: &[String],
        time: Option<&[String]>,
        dep_var_name: String,
    ) -> Result<Self, PanelError> {
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

        if entity.len() != y.len() {
            return Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Entity,
                y_rows: y.len(),
                other_rows: entity.len(),
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

        Ok(Self {
            y: y.to_vec(),
            x: x_columns.to_vec(),
            x_names,
            entity: entity.to_vec(),
            time: time.map(|t| t.to_vec()),
            dep_var_name,
        })
    }

    /// 被説明変数（長さ`n`）。
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    /// 説明変数（各列は長さ`n`）。
    pub fn x(&self) -> &[Vec<f64>] {
        &self.x
    }

    /// 説明変数名。`x()`の列と対応する。
    pub fn x_names(&self) -> &[String] {
        &self.x_names
    }

    /// 各行のエンティティID（長さ`n`）。
    pub fn entity(&self) -> &[String] {
        &self.entity
    }

    /// 各行の時点ID（長さ`n`）。1-way FEでは`None`。
    pub fn time(&self) -> Option<&[String]> {
        self.time.as_deref()
    }

    /// 被説明変数名。
    pub fn dep_var_name(&self) -> &str {
        &self.dep_var_name
    }

    /// 観測数 n
    pub fn nobs(&self) -> usize {
        self.y.len()
    }
}

/// FEの固定効果の方向（1-way/2-way）を指定する。`FeEstimator::fit`が受け取る
/// （モジュールdoc「`OlsEstimator`への委譲」参照。将来`FeOptions`（Issue #186）に
/// 統合される想定の暫定的なパラメータ）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeEffects {
    /// entityのみ（`within_transform_one_way`、6.1節）。
    OneWay,
    /// entity + time（`within_transform_two_way`、6.2節。バランスパネル必須、6.4節）。
    TwoWay,
}

/// FEの推定結果。`within`変換したデータを`OlsEstimator::fit`に委譲する
/// （モジュールdoc「`OlsEstimator`への委譲」参照。**係数推定のみがこの時点でのスコープ**——
/// 標準誤差等の統計量はFE固有の自由度・`cov_type`補正が入るまで正しくない）。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」）。
#[derive(Debug)]
pub struct FeEstimator {
    input: FeInput,
    effects: FeEffects,
    estimator: OlsEstimator,
    /// パネル自由度調整後のモデル自由度（`k + neffects`、6.3節）。
    df_model: usize,
    /// パネル自由度調整後の残差自由度（`n - df_model`、6.3節）。
    df_resid: usize,
    std_errors: Mat<f64>,
    t_stats: Mat<f64>,
    p_values: Mat<f64>,
    conf_lower: Mat<f64>,
    conf_upper: Mat<f64>,
    r_squared_adj: f64,
    aic: f64,
    bic: f64,
}

impl FeEstimator {
    /// `input`を`effects`が指定する方向でwithin変換した上で`OlsEstimator::fit`に委譲し、
    /// FEを推定する。パネル自由度調整（Issue #180、6.3節）を反映した標準誤差・t値・p値・
    /// 信頼区間・調整済みR²・AIC/BICを計算し直す（モジュールdoc「自由度調整」参照）。
    ///
    /// パイプライン: singleton検出
    /// （`validate_no_singleton_groups_one_way`/`validate_no_singleton_groups_two_way`、
    /// Issue #179）→ within変換（`within_transform_one_way`/`within_transform_two_way`、
    /// 2-wayはバランスパネル検証を内包、Issue #176）→ 自由度検証 → 分散ゼロ検出
    /// （`validate_no_zero_variance_regressors`、Issue #177）→ `OlsEstimator::fit`への委譲
    /// （`include_intercept=false`・`cov_type=CovType::Classical`固定。理由はモジュールdoc
    /// 参照）→ 自由度調整後の統計量の再計算。
    ///
    /// # Errors
    /// - `effects=TwoWay`で`input.time()`が`None`の場合は`PanelError::TwoWayRequiresTime`
    /// - singletonグループが見つかった場合は`PanelError::SingletonGroup`
    /// - `effects=TwoWay`でバランスパネルでない場合は`PanelError::UnbalancedPanelForTwoWay`
    /// - パネル自由度調整後の残差自由度（`df_resid`）が正にならない場合は
    ///   `PanelError::InsufficientDegreesOfFreedom`
    /// - within変換後に分散ゼロの説明変数がある場合は`PanelError::ZeroVarianceAfterDemeaning`
    /// - 委譲先の`OlsEstimator::fit`が失敗した場合（観測数不足・特異行列等）は
    ///   `PanelError::WithinRegressionFailed`
    pub fn fit(
        input: FeInput,
        effects: FeEffects,
        confidence_level: f64,
    ) -> Result<Self, PanelError> {
        // faerのグローバル並列度をPar::Seqに固定する（Issue #283、`crate::parallelism`。
        // 委譲先のOlsEstimator::fit自身も呼ぶが、`cargo test -p engine`でFeEstimator::fitを
        // 直接叩く経路との統一のためここでも呼ぶ、`engine/src/panel/CLAUDE.md`「faerの
        // グローバル並列度」参照）。
        crate::parallelism::ensure_serial();

        let (y, x) = match effects {
            FeEffects::OneWay => {
                validate_no_singleton_groups_one_way(&input)?;
                within_transform_one_way(&input)
            }
            FeEffects::TwoWay => {
                validate_no_singleton_groups_two_way(&input)?;
                within_transform_two_way(&input)?
            }
        };

        let n = input.nobs();
        let n_entities = count_unique(input.entity());
        let n_periods = match effects {
            FeEffects::OneWay => None,
            FeEffects::TwoWay => Some(count_unique(input.time().expect(
                "2-way already validated `time` is present \
                 (validate_no_singleton_groups_two_way/within_transform_two_way)",
            ))),
        };
        let k = input.x_names().len();
        // `neffects`: entityダミー・timeダミーの実効パラメータ数（6.3節）。2-wayは両者の
        // 間に定数項ぶんの重複が1つ生じるため`+1`補正（`n_entities + n_periods - 1`）。
        let neffects = match n_periods {
            None => n_entities,
            Some(n_periods) => n_entities + n_periods - 1,
        };
        let df_model = k + neffects;
        if n <= df_model {
            return Err(PanelError::InsufficientDegreesOfFreedom {
                n_obs: n,
                n_entities,
                n_periods,
                k,
            });
        }
        let df_resid = n - df_model;

        validate_no_zero_variance_regressors(&input, &x)?;

        // `OlsInput::from_columns`が返しうる`LeastSquaresError::Common(DimensionMismatch)`は
        // ここでは理論上到達不能: `y`/`x`はどちらも`within_transform_*`が`input.y()`/
        // `input.x()`（`FeInput::from_columns`が既に同じ長さであることを検証済み）から
        // 1対1で生成した同じ長さの列であり、この関数内で長さがずれる操作をしていない。
        // それでも`Result`を返す契約（`from_columns`のシグネチャ）をそのまま守り、
        // `unwrap`はしない（`ols::xtx_inverse`等の「理論上到達不能でも`Result`化する」方針
        // に揃える、`.claude/rules/rust-style.md`「テスト」参照）。
        let ols_input = OlsInput::from_columns(
            &y,
            &x,
            input.x_names().to_vec(),
            false,
            input.dep_var_name().to_string(),
        )
        .map_err(|source| PanelError::WithinRegressionFailed { source })?;
        let estimator = OlsEstimator::fit(ols_input, CovType::Classical, confidence_level)
            .map_err(|source| PanelError::WithinRegressionFailed { source })?;

        // `OlsEstimator`自身は`df_resid_ols = n - k`（`neffects`を知らない）を前提に
        // 標準誤差を計算済み（`σ̂²_ols = SSR/df_resid_ols`）。`σ̂²_fe = SSR/df_resid`との
        // 比は`df_resid_ols/df_resid`なので、標準誤差を`sqrt(df_resid_ols/df_resid)`倍に
        // スケールし直すだけで済む（`cov_params`の再構築は不要、モジュールdoc参照）。
        let df_resid_ols = n - k;
        let scale = (df_resid_ols as f64 / df_resid as f64).sqrt();

        let t_dist = StudentsT::new(0.0, 1.0, df_resid as f64)
            .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
        let t_crit = inference::critical_value(&t_dist, confidence_level);

        let k_dim = estimator.params().nrows();
        let mut std_errors = Mat::zeros(k_dim, 1);
        let mut t_stats = Mat::zeros(k_dim, 1);
        let mut p_values = Mat::zeros(k_dim, 1);
        let mut conf_lower = Mat::zeros(k_dim, 1);
        let mut conf_upper = Mat::zeros(k_dim, 1);
        for j in 0..k_dim {
            let coef = *estimator.params().get(j, 0);
            let se = *estimator.std_errors().get(j, 0) * scale;
            let stat = inference::compute_inference_stat(&t_dist, coef, se, t_crit);

            *std_errors.get_mut(j, 0) = se;
            *t_stats.get_mut(j, 0) = stat.stat;
            *p_values.get_mut(j, 0) = stat.p_value;
            *conf_lower.get_mut(j, 0) = stat.conf_low;
            *conf_upper.get_mut(j, 0) = stat.conf_high;
        }

        // 調整済みR²は変換前の元の`y`の中心化TSSに基づく overall R²から計算する
        // （`estimator().r_squared()`はwithin R²であり別概念、モジュールdoc参照）。
        // 残差はFWL定理により変換後・変換前どちらの尺度でも同じ値になるため
        // （`estimator.residuals()`をそのまま使える）、SSRの再計算は不要。
        let ssr: f64 = (0..n)
            .map(|i| (*estimator.residuals().get(i, 0)).powi(2))
            .sum();
        let y_mean: f64 = input.y().iter().sum::<f64>() / n as f64;
        let tss_overall: f64 = input.y().iter().map(|v| (v - y_mean).powi(2)).sum();
        let r_squared_overall = 1.0 - ssr / tss_overall;
        let r_squared_adj = 1.0 - (1.0 - r_squared_overall) * ((n - 1) as f64 / df_resid as f64);

        // `log_likelihood`自体は`SSR/n`のみに依存しdf非依存の式のためそのまま再利用できる
        // （モジュールdoc参照）。ペナルティ項の乗数だけ`k`から`df_model`に差し替える。
        let log_likelihood = estimator.log_likelihood();
        let aic = -2.0 * log_likelihood + 2.0 * (df_model as f64);
        let bic = -2.0 * log_likelihood + (n as f64).ln() * (df_model as f64);

        Ok(Self {
            input,
            effects,
            estimator,
            df_model,
            df_resid,
            std_errors,
            t_stats,
            p_values,
            conf_lower,
            conf_upper,
            r_squared_adj,
            aic,
            bic,
        })
    }

    /// within変換前の入力データ。
    pub fn input(&self) -> &FeInput {
        &self.input
    }

    /// 推定に使った固定効果の方向。
    pub fn effects(&self) -> FeEffects {
        self.effects
    }

    /// within変換済みデータに対する`OlsEstimator`本体。
    ///
    /// **係数（`params()`）・残差（`residuals()`）・within R²（`r_squared()`）は正しい値**
    /// だが、**標準誤差・t値・p値・信頼区間・調整済みR²・AIC/BICはパネル自由度調整前の
    /// 値のままで誤り**（`FeEstimator`自身の同名メソッド（`std_errors()`等）を使うこと、
    /// モジュールdoc「自由度調整」参照）。**F統計量はこの時点では未対応**
    /// （`estimator().f_statistic()`/`f_p_value()`は`df_resid_ols`ベースのまま）。
    /// `cov_type`はentity単位cluster化等の補正が入るまで`Classical`固定（Issue #181）。
    pub fn estimator(&self) -> &OlsEstimator {
        &self.estimator
    }

    /// パネル自由度調整後のモデル自由度（`k + neffects`、6.3節）。
    pub fn df_model(&self) -> usize {
        self.df_model
    }

    /// パネル自由度調整後の残差自由度（`n - df_model`、6.3節）。
    pub fn df_resid(&self) -> usize {
        self.df_resid
    }

    /// パネル自由度調整後の標準誤差（`(k, 1)`、`estimator().params()`と対応）。
    pub fn std_errors(&self) -> &Mat<f64> {
        &self.std_errors
    }

    /// パネル自由度調整後のt統計量（`(k, 1)`）。
    pub fn t_stats(&self) -> &Mat<f64> {
        &self.t_stats
    }

    /// パネル自由度調整後の両側p値（`(k, 1)`）。
    pub fn p_values(&self) -> &Mat<f64> {
        &self.p_values
    }

    /// パネル自由度調整後の信頼区間の下限（`(k, 1)`）。
    pub fn conf_lower(&self) -> &Mat<f64> {
        &self.conf_lower
    }

    /// パネル自由度調整後の信頼区間の上限（`(k, 1)`）。
    pub fn conf_upper(&self) -> &Mat<f64> {
        &self.conf_upper
    }

    /// パネル自由度調整済み決定係数（overall R²ベース、モジュールdoc参照）。
    pub fn r_squared_adj(&self) -> f64 {
        self.r_squared_adj
    }

    /// パネル自由度調整済みAIC（`df_model`をペナルティ項に使う、モジュールdoc参照）。
    pub fn aic(&self) -> f64 {
        self.aic
    }

    /// パネル自由度調整済みBIC。
    pub fn bic(&self) -> f64 {
        self.bic
    }
}

/// `ids`のユニークID数を数える（`n_entities`/`n_periods`のカウント）。純粋な
/// カーディナリティ集計のため`HashSet`でよい（`validate_balanced_panel`等と同じ理由）。
fn count_unique(ids: &[String]) -> usize {
    ids.iter().collect::<HashSet<_>>().len()
}

/// `ids`に現れる全ユニークIDに`θ=1.0`を割り当てた`BTreeMap`を作る。
///
/// `quasi_demean_column`の`theta`引数はエンティティID→θ_iの対応（`&BTreeMap<String,
/// f64>`）を要求するが、FEのwithin変換は常にθ=1（`panel-api-design.md`7.4節: FEは
/// `quasi_demean_column`のθ=1の特殊ケース）のため、呼び出し側で毎回組み立てる代わりに
/// ここに切り出す。1-way（`entity`列）・2-way（entity列・time列の両方）のどちらでも使う。
///
/// 先に`HashSet`でユニークなIDへ絞り込んでから`String`を複製する（`ids.iter().map(|id|
/// (id.clone(), 1.0)).collect()`のように観測順のまま素朴に`collect`すると、`BTreeMap`の
/// 重複キーは値のみ上書きされキー自体は複製されたまま即破棄されるため、観測数`n`分の
/// ヒープ確保が発生してしまう。rust-reviewer指摘、ユニークID数分のみ複製するよう修正）。
fn all_ones_theta(ids: &[String]) -> BTreeMap<String, f64> {
    ids.iter()
        .collect::<HashSet<_>>()
        .into_iter()
        .map(|id| (id.clone(), 1.0))
        .collect()
}

/// 1-way FE（entityのみ）のwithin変換。`y`と各`x`列にentityでのquasi-demean（θ=1）を
/// 適用する。不均衡パネルも無条件でサポートする（モジュールdoc・6.1節参照）。
///
/// 戻り値は`(y_transformed, x_transformed)`（元の列順を保持）。
pub fn within_transform_one_way(input: &FeInput) -> (Vec<f64>, Vec<Vec<f64>>) {
    let theta = all_ones_theta(input.entity());
    let y = quasi_demean_column(input.y(), input.entity(), &theta);
    let x = input
        .x()
        .iter()
        .map(|col| quasi_demean_column(col, input.entity(), &theta))
        .collect();
    (y, x)
}

/// 2-way FE（entity + time FE）のwithin変換。閉形式の二重デミーニングと数学的に等価な
/// 「entityでquasi-demean → その結果をtimeでquasi-demean」の2段階適用で計算する
/// （モジュールdoc参照）。事前にバランスパネルであることを検証する（6.4節）。
///
/// 戻り値は`(y_transformed, x_transformed)`（元の列順を保持）。
///
/// # Errors
/// - `input.time()`が`None`の場合は`PanelError::TwoWayRequiresTime`
/// - バランスパネルでない場合は`PanelError::UnbalancedPanelForTwoWay`
pub fn within_transform_two_way(input: &FeInput) -> Result<(Vec<f64>, Vec<Vec<f64>>), PanelError> {
    let time = input.time().ok_or(PanelError::TwoWayRequiresTime)?;
    validate_balanced_panel(input.entity(), time)?;

    let entity_theta = all_ones_theta(input.entity());
    let y_entity_demeaned = quasi_demean_column(input.y(), input.entity(), &entity_theta);
    let x_entity_demeaned: Vec<Vec<f64>> = input
        .x()
        .iter()
        .map(|col| quasi_demean_column(col, input.entity(), &entity_theta))
        .collect();

    let time_theta = all_ones_theta(time);
    let y = quasi_demean_column(&y_entity_demeaned, time, &time_theta);
    let x = x_entity_demeaned
        .iter()
        .map(|col| quasi_demean_column(col, time, &time_theta))
        .collect();

    Ok((y, x))
}

/// 1-way FE向けのsingleton検出（6.5節）。`entity`に観測数1のグループがあれば
/// `PanelError::SingletonGroup`を返す。
///
/// # Errors
/// entityに観測数1のグループが見つかった場合は`PanelError::SingletonGroup`
/// （`dimension: PanelDimension::Entity`）。
pub fn validate_no_singleton_groups_one_way(input: &FeInput) -> Result<(), PanelError> {
    reject_singleton_group(PanelDimension::Entity, input.entity())
}

/// 2-way FE向けのsingleton検出（6.5節）。entity・time双方を対称に検出する
/// （`within_transform_two_way`と同じく`time`必須）。
///
/// **`time`の存在チェックを最初に行う**（`within_transform_two_way`と同じ順序に揃える。
/// `fit()`側が複数のバリデーションを組み合わせて呼ぶ際、同じ前提条件——2-way FEには
/// `time`が要る——のチェックタイミングが関数ごとにばらつかないようにするため）。
/// その後、entityを先にチェックしてからtimeをチェックする（両方に該当するsingletonが
/// あった場合はentity側を先に報告する。`FeInput`のフィールド順（entity→time）に合わせた
/// 恣意的な優先順位）。
///
/// # Errors
/// - `input.time()`が`None`の場合は`PanelError::TwoWayRequiresTime`
/// - entityまたはtimeに観測数1のグループが見つかった場合は`PanelError::SingletonGroup`
///   （該当する`dimension`を含む）
pub fn validate_no_singleton_groups_two_way(input: &FeInput) -> Result<(), PanelError> {
    let time = input.time().ok_or(PanelError::TwoWayRequiresTime)?;
    reject_singleton_group(PanelDimension::Entity, input.entity())?;
    reject_singleton_group(PanelDimension::Time, time)
}

/// `ids`（`entity`または`time`の列）に観測数1のグループがあれば
/// `PanelError::SingletonGroup`を返す。
///
/// 複数のsingletonグループが存在する場合は、観測順で最初に現れるグループのみを報告する
/// （`validate_no_zero_variance_regressors`の「最初の1件を報告」方針と統一）。グループの
/// 出現回数を数える集計自体はカーディナリティのみが目的で、グループ「間」の浮動小数点
/// 加算順序に依存しないため`HashMap`でよい（`engine/src/panel/CLAUDE.md`「`quasi_demean_
/// column`の内部集約は`HashMap`でよい」と同じ理由）。
fn reject_singleton_group(dimension: PanelDimension, ids: &[String]) -> Result<(), PanelError> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for id in ids {
        *counts.entry(id.as_str()).or_insert(0) += 1;
    }

    for id in ids {
        if counts[id.as_str()] == 1 {
            return Err(PanelError::SingletonGroup {
                dimension,
                group_id: id.clone(),
            });
        }
    }
    Ok(())
}

/// within変換後の説明変数の各列に分散ゼロの列がないことを検証する（6.7節）。1-way/2-way
/// 共通ロジック（`within_transform_one_way`/`within_transform_two_way`のどちらの出力も
/// 引数に渡せる）。
///
/// 時間不変変数（1-way）だけでなく、2-wayでtime FEと完全共線な「エンティティ間で変動しない
/// 列」も同じチェックで検出できる（6.7節）。`x_transformed`は呼び出し側が`within_transform_*`
/// の戻り値をそのまま渡す想定で、`input.x()`（変換前の生の列）と同じ列順・同じ列数・列ごとに
/// 同じ長さを持つことを前提とする（`engine`内の内部契約であり、ユーザー入力起因ではない。
/// `quasi_demean_column`の呼び出し元契約と同じ扱いで`assert_eq!`で守る。`debug_assert_eq!`
/// だとリリースビルドで無効化され、列数不一致時に`zip`が黙って短い方へ切り詰め検証漏れの
/// 列が発生しうるため不可）。
///
/// # Errors
/// 分散ゼロの列が見つかった場合は`PanelError::ZeroVarianceAfterDemeaning`（該当列名を含む）。
/// 複数列が該当する場合は`x_names`の順で最初に見つかった列のみを報告する（OLSの特異性
/// 検出等、他のバリデーションも「最初の1件を報告」で統一している）。
pub fn validate_no_zero_variance_regressors(
    input: &FeInput,
    x_transformed: &[Vec<f64>],
) -> Result<(), PanelError> {
    assert_eq!(
        input.x().len(),
        x_transformed.len(),
        "x_transformed must have the same number of columns as input.x()"
    );

    for ((name, original), transformed) in input.x_names().iter().zip(input.x()).zip(x_transformed)
    {
        assert_eq!(
            original.len(),
            transformed.len(),
            "x_transformed column '{name}' must have the same length as the original column"
        );
        if column_is_zero_variance(original, transformed) {
            return Err(PanelError::ZeroVarianceAfterDemeaning {
                column: name.clone(),
            });
        }
    }
    Ok(())
}

/// `transformed`（within変換後の1列）の分散が、`original`（変換前の同じ列）のスケールに
/// 対して無視できるほど小さいかを判定する。
///
/// **絶対閾値ではなく相対閾値を使う**（`.claude/rules/rust-style.md`「線形代数」の特異性
/// 判定の方針と同じ。データのスケールに依存しないようにするため）。`original`の最大絶対値を
/// スケールの目安にする理由: 真に分散ゼロの列（時間不変・完全共線）は、within変換後の値が
/// 数学的には厳密に0だが浮動小数点演算では丸め誤差の残差（`original`のスケールに比例する
/// 大きさ）が残る。`transformed`自身の最大絶対値をスケールに使うと、この残差自身を基準に
/// 残差を判定する自己参照になり閾値が機能しないため、変換前の値を基準にする。
///
/// 閾値の乗数`n`（観測数）は、`ols::xtx_inverse`の特異性判定
/// （`(k as f64) * f64::EPSILON * max_abs_diag`、分解に関わる次元数を乗数にする）と同じ
/// 発想: グループ平均の集約（`quasi_demean_column`、観測数`n`項の和）→差分の丸め誤差の
/// 蓄積が観測数に比例しうることを踏まえた選択。2-wayは「entityでdemean→timeでdemean」の
/// 2段階適用（モジュールdoc参照）で丸め誤差が2回蓄積しうるが、閾値はこの2段階分を明示的に
/// 倍にはしていない（実測上、時間不変・完全共線変数の残差は乗数`2`の差では閾値を跨がない
/// 桁数——原点`scale`比`~1e-16`——であることを想定した割り切り。将来1-wayと2-wayで
/// 誤検出/見逃しの傾向差が実際に問題になったら、2-way用の乗数を分けることを検討する）。
fn column_is_zero_variance(original: &[f64], transformed: &[f64]) -> bool {
    let n = transformed.len();
    if n == 0 {
        // n=0は`FeInput::from_columns`が許容する境界ケース（`from_columns_with_zero_
        // observations_succeeds`）。分散の定義自体が意味を持たないため、ゼロ分散とは
        // 判定しない（呼び出し側の`fit()`は別途`InsufficientDegreesOfFreedom`等で
        // n=0を弾く想定、6.7節はあくまで「デミーニング後の分散」の検証に限定する）。
        return false;
    }

    let mean: f64 = transformed.iter().sum::<f64>() / n as f64;
    let variance: f64 = transformed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    let scale = original.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let threshold = n as f64 * f64::EPSILON * scale;

    std_dev <= threshold
}

/// 2-way FEがバランスパネル（`entity` × `time`の全組合せが過不足なく1回ずつ存在する）
/// であることを検証する（6.4節）。
///
/// 観測数カウントの一致（`n_obs == n_entities * n_periods`）だけでは不十分
/// （`PanelError::UnbalancedPanelForTwoWay`のdocコメント参照: あるペアの重複と別ペアの
/// 欠落が相殺してカウントだけ一致する入力がありうる）。代わりに、`(entity, time)`
/// ペアが重複なく（`unique_pairs.len() == n_obs`）、かつ`n_obs == n_entities *
/// n_periods`であることを検証する。ペア集合は`entity × time`の全組合せグリッド
/// （サイズ`n_entities * n_periods`）の部分集合であるため、重複が無く要素数がグリッドの
/// サイズと一致すれば、部分集合が全体（＝全組合せが埋まっている）と一致することが
/// 数学的に保証される。
///
/// `entity.len() == time.len()`は`FeInput::from_columns`が既に保証している契約
/// （呼び出し側は常に同じ`FeInput`からこの2つを渡す）。
fn validate_balanced_panel(entity: &[String], time: &[String]) -> Result<(), PanelError> {
    let n_obs = entity.len();
    let n_entities = entity.iter().collect::<HashSet<_>>().len();
    let n_periods = time.iter().collect::<HashSet<_>>().len();
    let unique_pairs: HashSet<(&str, &str)> = entity
        .iter()
        .zip(time.iter())
        .map(|(e, t)| (e.as_str(), t.as_str()))
        .collect();
    let expected = n_entities * n_periods;

    if unique_pairs.len() != n_obs || n_obs != expected {
        return Err(PanelError::UnbalancedPanelForTwoWay {
            n_obs,
            n_entities,
            n_periods,
            expected,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::common::LeastSquaresError;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn from_columns_builds_one_way_input() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let x1 = vec![10.0, 20.0, 30.0, 40.0];
        let entity = strings(&["a", "a", "b", "b"]);

        let input = FeInput::from_columns(
            &y,
            std::slice::from_ref(&x1),
            vec!["x1".to_string()],
            &entity,
            None,
            "y".to_string(),
        )
        .unwrap();

        assert_eq!(input.y(), &y);
        assert_eq!(input.x(), &[x1]);
        assert_eq!(input.x_names(), &["x1".to_string()]);
        assert_eq!(input.entity(), entity.as_slice());
        assert_eq!(input.time(), None);
        assert_eq!(input.dep_var_name(), "y");
        assert_eq!(input.nobs(), 4);
    }

    #[test]
    fn from_columns_builds_two_way_input() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["2020", "2021", "2020", "2021"]);

        let input = FeInput::from_columns(
            &y,
            &[vec![1.0, 1.0, 1.0, 1.0]],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".to_string(),
        )
        .unwrap();

        assert_eq!(input.time(), Some(time.as_slice()));
    }

    #[test]
    fn from_columns_with_no_regressors_succeeds() {
        // OLSと異なりFEは説明変数0個でも`FeInput`自体は構築できる（推定可能性の検証は
        // `fit()`側の責務、`IvInput`が識別可能性を検証しないのと同じ層分け）。
        let y = [1.0, 2.0];
        let entity = strings(&["a", "b"]);

        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".to_string()).unwrap();

        assert!(input.x().is_empty());
        assert!(input.x_names().is_empty());
    }

    #[test]
    fn from_columns_returns_dimension_mismatch_on_mismatched_x_column_length() {
        let y = [1.0, 2.0, 3.0];
        let x1 = vec![10.0, 20.0]; // yより短い
        let entity = strings(&["a", "b", "c"]);

        let result = FeInput::from_columns(
            &y,
            &[x1],
            vec!["x1".to_string()],
            &entity,
            None,
            "y".to_string(),
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::Common(CommonError::DimensionMismatch {
                y_rows: 3,
                x_rows: 2,
            })
        );
    }

    #[test]
    fn from_columns_returns_identifier_dimension_mismatch_for_entity() {
        let y = [1.0, 2.0, 3.0];
        let entity = strings(&["a", "b"]); // yより短い

        let result = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".to_string());

        assert_eq!(
            result.unwrap_err(),
            PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Entity,
                y_rows: 3,
                other_rows: 2,
            }
        );
    }

    #[test]
    fn from_columns_returns_identifier_dimension_mismatch_for_time() {
        let y = [1.0, 2.0, 3.0];
        let entity = strings(&["a", "b", "c"]);
        let time = strings(&["2020", "2021"]); // yより短い

        let result = FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".to_string());

        assert_eq!(
            result.unwrap_err(),
            PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                y_rows: 3,
                other_rows: 2,
            }
        );
    }

    #[test]
    fn from_columns_with_zero_observations_succeeds() {
        // n=0（y/entity/timeすべて空）でも次元は一致しているため`FeInput`自体の構築は
        // 成功する。推定可能性の検証（n<=kの類）は`fit()`側の責務であり、`from_columns`は
        // 次元検証のみを行う設計であることを明示するための境界値テスト
        // （`.claude/rules/testing-policy.md`「境界値・悪条件」）。
        let input =
            FeInput::from_columns(&[], &[], vec![], &[], Some(&[]), "y".to_string()).unwrap();

        assert_eq!(input.nobs(), 0);
        assert_eq!(input.time(), Some([].as_slice()));
    }

    #[test]
    #[should_panic(expected = "x_columns and x_names must have the same length")]
    fn from_columns_panics_on_mismatched_names_arity() {
        let y = [1.0, 2.0];
        let entity = strings(&["a", "b"]);
        let _ = FeInput::from_columns(
            &y,
            &[vec![1.0, 2.0]],
            vec![], // x_columnsは1列だがx_namesは0個
            &entity,
            None,
            "y".to_string(),
        );
    }

    // ── within_transform_one_way ────────────────────────────────────────────

    #[test]
    fn within_transform_one_way_demeans_y_and_all_x_columns_by_entity() {
        // a: mean(y)=15, mean(x1)=150 / b: mean(y)=6, mean(x1)=60（`quasi_demean_column`
        // の`quasi_demean_column_with_theta_one_is_the_within_transformation`と同じ数値）。
        let entity = strings(&["a", "a", "b", "b", "b"]);
        let y = [10.0, 20.0, 3.0, 6.0, 9.0];
        let x1 = vec![100.0, 200.0, 30.0, 60.0, 90.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let (y_out, x_out) = within_transform_one_way(&input);

        assert_eq!(y_out, vec![-5.0, 5.0, -3.0, 0.0, 3.0]);
        assert_eq!(x_out, vec![vec![-50.0, 50.0, -30.0, 0.0, 30.0]]);
    }

    #[test]
    fn within_transform_one_way_supports_unbalanced_panel() {
        // T_a=1, T_b=3の不均衡パネル。エンティティ平均を引くだけで正確に成立する
        // （`quasi_demean_column_handles_unbalanced_panel`と同じ数値）。
        let entity = strings(&["a", "b", "b", "b"]);
        let y = [4.0, 2.0, 4.0, 6.0];
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let (y_out, x_out) = within_transform_one_way(&input);

        assert_eq!(y_out, vec![0.0, -2.0, 0.0, 2.0]);
        assert!(x_out.is_empty());
    }

    // ── within_transform_two_way ────────────────────────────────────────────

    /// N=2（a, b）× T=2（"1", "2"）のバランスパネル。モジュールdocの導出で使った例と
    /// 同じ数値（ȳ..=4.5, ȳ_a.=2, ȳ_b.=7, ȳ_.1=3, ȳ_.2=6 →
    /// 閉形式`ỹ_it = y_it - ȳ_i. - ȳ_.t + ȳ..`で[0.5, -0.5, -0.5, 0.5]）。
    fn balanced_two_way_input(y: [f64; 4], x_columns: &[Vec<f64>]) -> FeInput {
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["1", "2", "1", "2"]);
        FeInput::from_columns(
            &y,
            x_columns,
            x_columns
                .iter()
                .enumerate()
                .map(|(i, _)| format!("x{i}"))
                .collect(),
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap()
    }

    #[test]
    fn within_transform_two_way_matches_closed_form_double_demeaning() {
        let y = [1.0, 3.0, 5.0, 9.0];
        let x1 = vec![2.0, 6.0, 10.0, 18.0]; // yのちょうど2倍（線形性の確認を兼ねる）
        let input = balanced_two_way_input(y, &[x1]);

        let (y_out, x_out) = within_transform_two_way(&input).unwrap();

        let expected_y = [0.5, -0.5, -0.5, 0.5];
        for (actual, expected) in y_out.iter().zip(expected_y.iter()) {
            assert!((actual - expected).abs() < 1e-12, "y_out = {y_out:?}");
        }
        let expected_x1: Vec<f64> = expected_y.iter().map(|v| v * 2.0).collect();
        for (actual, expected) in x_out[0].iter().zip(expected_x1.iter()) {
            assert!((actual - expected).abs() < 1e-12, "x_out = {x_out:?}");
        }
    }

    #[test]
    fn within_transform_two_way_requires_time() {
        let y = [1.0, 2.0];
        let entity = strings(&["a", "b"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let result = within_transform_two_way(&input);

        assert_eq!(result.unwrap_err(), PanelError::TwoWayRequiresTime);
    }

    #[test]
    fn within_transform_two_way_rejects_unbalanced_panel_with_missing_combination() {
        // entity=b, time="2"の観測が欠けている（n_obs=3 != n_entities*n_periods=4）。
        let y = [1.0, 2.0, 3.0];
        let entity = strings(&["a", "a", "b"]);
        let time = strings(&["1", "2", "1"]);
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let result = within_transform_two_way(&input);

        assert_eq!(
            result.unwrap_err(),
            PanelError::UnbalancedPanelForTwoWay {
                n_obs: 3,
                n_entities: 2,
                n_periods: 2,
                expected: 4,
            }
        );
    }

    #[test]
    fn within_transform_two_way_rejects_duplicate_pair_that_offsets_missing_combination() {
        // entity=[a,a,b,b], time=[1,1,2,2]: (a,1)が重複、(a,2)と(b,1)が欠落。
        // n_obs=4はn_entities(2)*n_periods(2)=4と一致してしまうが、ユニークな
        // (entity,time)ペアは{(a,1),(b,2)}の2個のみで4に満たないため、単純な
        // カウント一致チェックでは見逃す入力を正しく検出できることを確認する
        // （`PanelError::UnbalancedPanelForTwoWay`のdocコメント・`validate_balanced_panel`
        // 参照）。
        let y = [1.0, 2.0, 3.0, 4.0];
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["1", "1", "2", "2"]);
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let result = within_transform_two_way(&input);

        assert_eq!(
            result.unwrap_err(),
            PanelError::UnbalancedPanelForTwoWay {
                n_obs: 4,
                n_entities: 2,
                n_periods: 2,
                expected: 4,
            }
        );
    }

    // ── validate_no_singleton_groups_one_way / _two_way ─────────────────────

    #[test]
    fn validate_no_singleton_groups_one_way_detects_entity_singleton() {
        // entity "c"は観測数1（singleton）。
        let y = [1.0, 2.0, 3.0];
        let entity = strings(&["a", "a", "c"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let result = validate_no_singleton_groups_one_way(&input);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_singleton_groups_one_way_accepts_no_singleton() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let entity = strings(&["a", "a", "b", "b"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        assert_eq!(validate_no_singleton_groups_one_way(&input), Ok(()));
    }

    #[test]
    fn validate_no_singleton_groups_two_way_detects_entity_singleton() {
        // entity "c"は時点"1"のみの1観測（singleton）。time側は各時点2観測ずつで
        // singletonではない。
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let entity = strings(&["a", "a", "b", "b", "c"]);
        let time = strings(&["1", "2", "1", "2", "1"]);
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let result = validate_no_singleton_groups_two_way(&input);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_singleton_groups_two_way_detects_time_singleton() {
        // time "3"はentity "a"のみの1観測（singleton）。entity側はどちらも2観測ずつで
        // singletonではない。
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let entity = strings(&["a", "a", "a", "b", "b"]);
        let time = strings(&["1", "2", "3", "1", "2"]);
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let result = validate_no_singleton_groups_two_way(&input);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Time,
                group_id: "3".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_singleton_groups_two_way_reports_entity_before_time_when_both_present() {
        // entity "c"（1観測）とtime "3"（1観測、entity "c"自身の行）が両方singleton。
        // entityを先にチェックする方針（関数doc参照）により、entity側が報告される。
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let entity = strings(&["a", "a", "b", "b", "c"]);
        let time = strings(&["1", "2", "1", "2", "3"]);
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let result = validate_no_singleton_groups_two_way(&input);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_singleton_groups_two_way_requires_time() {
        // entity側はsingletonではない（"a"が2観測）ため、`time`未指定の
        // `PanelError::TwoWayRequiresTime`が先に検出されることを確認する。
        let y = [1.0, 2.0];
        let entity = strings(&["a", "a"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let result = validate_no_singleton_groups_two_way(&input);

        assert_eq!(result.unwrap_err(), PanelError::TwoWayRequiresTime);
    }

    #[test]
    fn validate_no_singleton_groups_two_way_reports_missing_time_even_with_entity_singleton() {
        // entity "c"はsingletonだが`time`も未指定。`time`の存在チェックを先に行う方針
        // （関数doc、`within_transform_two_way`と同じ順序）により`TwoWayRequiresTime`が
        // 優先される。
        let y = [1.0, 2.0, 3.0];
        let entity = strings(&["a", "a", "c"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let result = validate_no_singleton_groups_two_way(&input);

        assert_eq!(result.unwrap_err(), PanelError::TwoWayRequiresTime);
    }

    #[test]
    fn reject_singleton_group_with_no_observations_succeeds() {
        // n=0境界（`validate_no_zero_variance_regressors_with_zero_observations_and_a_
        // regressor_succeeds`と同様の境界値テストの慣習に合わせる）。空配列にはsingleton
        // となりうる要素自体が存在しないため`Ok(())`になる。
        assert_eq!(reject_singleton_group(PanelDimension::Entity, &[]), Ok(()));
    }

    #[test]
    fn validate_no_singleton_groups_two_way_accepts_no_singleton() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["1", "2", "1", "2"]);
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        assert_eq!(validate_no_singleton_groups_two_way(&input), Ok(()));
    }

    // ── validate_no_zero_variance_regressors ────────────────────────────────

    #[test]
    fn validate_no_zero_variance_regressors_detects_time_invariant_variable_in_one_way_fe() {
        // "female"は各エンティティ内で一定（時間不変）のため、1-way within変換後は
        // 浮動小数点誤差の範囲でゼロになる（6.7節のユースケースそのもの）。
        let entity = strings(&["a", "a", "b", "b"]);
        let y = [1.0, 2.0, 3.0, 5.0];
        let x_varying = vec![10.0, 20.0, 5.0, 15.0];
        let female = vec![0.0, 0.0, 1.0, 1.0];
        let input = FeInput::from_columns(
            &y,
            &[x_varying, female],
            vec!["x_varying".to_string(), "female".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let (_, x_out) = within_transform_one_way(&input);
        let result = validate_no_zero_variance_regressors(&input, &x_out);

        assert_eq!(
            result.unwrap_err(),
            PanelError::ZeroVarianceAfterDemeaning {
                column: "female".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_zero_variance_regressors_accepts_all_varying_columns() {
        let entity = strings(&["a", "a", "b", "b", "b"]);
        let y = [10.0, 20.0, 3.0, 6.0, 9.0];
        let x1 = vec![100.0, 200.0, 30.0, 60.0, 90.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let (_, x_out) = within_transform_one_way(&input);

        assert_eq!(validate_no_zero_variance_regressors(&input, &x_out), Ok(()));
    }

    #[test]
    fn validate_no_zero_variance_regressors_detects_entity_invariant_variable_in_two_way_fe() {
        // "year_dummy"はエンティティ間で変動しない（time FEと完全共線）ため、2-way
        // within変換後はゼロ分散になる（6.7節「time FEと完全共線な列も同じチェックで
        // 検出できる」の具体例）。
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["1", "2", "1", "2"]);
        let y = [1.0, 3.0, 5.0, 9.0];
        let x_varying = vec![2.0, 6.0, 10.0, 18.0];
        let year_dummy = vec![0.0, 1.0, 0.0, 1.0];
        let input = FeInput::from_columns(
            &y,
            &[x_varying, year_dummy],
            vec!["x_varying".to_string(), "year_dummy".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let (_, x_out) = within_transform_two_way(&input).unwrap();
        let result = validate_no_zero_variance_regressors(&input, &x_out);

        assert_eq!(
            result.unwrap_err(),
            PanelError::ZeroVarianceAfterDemeaning {
                column: "year_dummy".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_zero_variance_regressors_reports_first_offending_column_in_name_order() {
        // 2列とも時間不変。`x_names`の順で最初（"const_a"）のみを報告する
        // （関数docコメントの方針）。
        let entity = strings(&["a", "a", "b", "b"]);
        let y = [1.0, 2.0, 3.0, 4.0];
        let const_a = vec![1.0, 1.0, 2.0, 2.0];
        let const_b = vec![9.0, 9.0, 8.0, 8.0];
        let input = FeInput::from_columns(
            &y,
            &[const_a, const_b],
            vec!["const_a".to_string(), "const_b".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let (_, x_out) = within_transform_one_way(&input);
        let result = validate_no_zero_variance_regressors(&input, &x_out);

        assert_eq!(
            result.unwrap_err(),
            PanelError::ZeroVarianceAfterDemeaning {
                column: "const_a".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_zero_variance_regressors_with_no_regressors_succeeds() {
        let y = [1.0, 2.0];
        let entity = strings(&["a", "b"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        assert_eq!(validate_no_zero_variance_regressors(&input, &[]), Ok(()));
    }

    #[test]
    fn validate_no_zero_variance_regressors_with_zero_observations_and_a_regressor_succeeds() {
        // n=0（`column_is_zero_variance`のn=0分岐、モジュールdoc参照）は「回帰変数0列」
        // （上のテスト）とは別に、「観測数0だが回帰変数自体は1列存在する」ケースでも
        // 明示的に確認する（列は空`Vec`になるが、列は存在する点が上と異なる）。
        let input = FeInput::from_columns(
            &[],
            &[vec![]],
            vec!["x1".to_string()],
            &[],
            None,
            "y".into(),
        )
        .unwrap();

        assert_eq!(
            validate_no_zero_variance_regressors(&input, &[vec![]]),
            Ok(())
        );
    }

    // ── FeEstimator::fit ─────────────────────────────────────────────────

    #[test]
    fn fe_estimator_fit_one_way_recovers_known_slope() {
        // entity a: x=[1,2,3], y=2x+5（fixed effect=5）→ y=[7,9,11]
        // entity b: x=[4,5,6], y=2x+10（fixed effect=10）→ y=[18,20,22]
        // ノイズなしのため、within変換後のOLS（切片なし）は真のスロープ2.0を厳密に
        // 復元するはず（fixed effectはwithin変換で消去される）。
        let entity = strings(&["a", "a", "a", "b", "b", "b"]);
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<f64> = x1
            .iter()
            .zip(&entity)
            .map(|(x, e)| 2.0 * x + if e == "a" { 5.0 } else { 10.0 })
            .collect();
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, 0.95).unwrap();

        assert!((*fe.estimator().params().get(0, 0) - 2.0).abs() < 1e-9);
        assert_eq!(fe.effects(), FeEffects::OneWay);
        assert!(!fe.estimator().input().has_intercept());
        for r in fe.estimator().residuals().col(0).iter() {
            assert!(r.abs() < 1e-9);
        }
    }

    #[test]
    fn fe_estimator_fit_two_way_recovers_known_slope() {
        // `within_transform_two_way_matches_closed_form_double_demeaning`と同じ関係
        // （x1はyのちょうど2倍。この関係は線形変換の下で任意のN・Tで恒等的に保たれるため
        // 具体的な値は問わない）だが、3エンティティ×3時点（n=9）のバランスパネルに
        // 拡張する：df_model=k(1)+neffects(n_entities+n_periods-1=3+3-1=5)=6、n=9>6で
        // Issue #180の自由度検証（`n<=df_model`）を通過できる規模にする必要があるため
        // （N=2,T=2のn=4だとdf_model=1+3=4となりn<=df_modelで弾かれてしまう）。
        // 2-way within変換後、x1_out ≈ 2 * y_out がほぼ成り立つため、切片なしOLSの
        // スロープは0.5にほぼ一致するはず。x1はy*2からごくわずかに擾乱を入れる
        // （厳密にx1=2*yだと残差が全行ゼロになり、`OlsEstimator::fit`のF検定
        // （`wald_f_test`）が分散ゼロによる特異行列で`ComputationFailed`を返してしまう
        // 退化ケースを踏むため。既存の`WlsEstimator`のテストコメント
        // `fit_without_intercept_uses_uncentered_r_squared_and_omits_const`と同じ理由）。
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c"]);
        let time = strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3"]);
        let y = [1.0, 3.0, 5.0, 2.0, 4.0, 9.0, 6.0, 1.0, 8.0];
        let noise = [0.01, -0.02, 0.015, -0.01, 0.02, -0.015, 0.01, -0.01, 0.02];
        let x1: Vec<f64> = y.iter().zip(&noise).map(|(v, n)| 2.0 * v + n).collect();
        let input = FeInput::from_columns(
            &y,
            &[x1],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, 0.95).unwrap();

        // 許容誤差0.01は`testing-policy.md`の相対誤差1e-8方針の対象外（リファレンス実装
        // との数値比較ではなく、本テスト自身が注入した擾乱ノイズに対する内部整合性の
        // 確認のため）。ノイズの大きさ（最大0.02、xスケール比で見ると相対誤差1〜2%程度）
        // に対してスロープ推定への影響がこの範囲に収まることを確認する目的の閾値。
        assert!((*fe.estimator().params().get(0, 0) - 0.5).abs() < 0.01);
        assert_eq!(fe.effects(), FeEffects::TwoWay);
        assert_eq!(fe.df_model(), 6);
        assert_eq!(fe.df_resid(), 3);
    }

    #[test]
    fn fe_estimator_fit_propagates_singleton_error() {
        let entity = strings(&["a", "a", "c"]);
        let y = [1.0, 2.0, 3.0];
        let x1 = vec![1.0, 2.0, 3.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn fe_estimator_fit_propagates_zero_variance_error() {
        // "female"は各エンティティ内で一定（時間不変）。n=6・n_entities=2・k=2で
        // df_model=4・n>df_modelとなるよう、各エンティティ3観測に拡張する
        // （n=4だとdf_model=2+2=4でn<=df_modelとなりInsufficientDegreesOfFreedomが
        // 先に発火してしまうため、Issue #180の自由度検証を踏まえたサイズにする）。
        let entity = strings(&["a", "a", "a", "b", "b", "b"]);
        let y = [1.0, 2.0, 3.0, 5.0, 6.0, 8.0];
        let x_varying = vec![10.0, 20.0, 15.0, 5.0, 10.0, 20.0];
        let female = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let input = FeInput::from_columns(
            &y,
            &[x_varying, female],
            vec!["x_varying".to_string(), "female".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::ZeroVarianceAfterDemeaning {
                column: "female".to_string(),
            }
        );
    }

    #[test]
    fn fe_estimator_fit_two_way_requires_time() {
        let entity = strings(&["a", "a", "b", "b"]);
        let y = [1.0, 2.0, 3.0, 4.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = FeEstimator::fit(input, FeEffects::TwoWay, 0.95);

        assert_eq!(result.unwrap_err(), PanelError::TwoWayRequiresTime);
    }

    #[test]
    fn fe_estimator_fit_two_way_propagates_singleton_error() {
        // entity "c"は時点"1"のみの1観測（singleton）。`fit()`が
        // `validate_no_singleton_groups_two_way`（entity/time双方を対称にチェックする方）を
        // 正しく呼んでいることの配線確認（1-way用の検証関数を誤って呼んでいないか）。
        let entity = strings(&["a", "a", "b", "b", "c"]);
        let time = strings(&["1", "2", "1", "2", "1"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = FeInput::from_columns(
            &y,
            &[x1],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::TwoWay, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn fe_estimator_fit_two_way_propagates_unbalanced_panel_error() {
        // entity=[a,a,b,b], time=[1,1,2,2]: (a,1)が重複、(a,2)と(b,1)が欠落。
        // singletonではない（各entity/time値とも観測数2）ため、`fit()`が
        // `within_transform_two_way`のバランスパネル検証まで到達していることの配線確認。
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["1", "1", "2", "2"]);
        let y = [1.0, 2.0, 3.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let result = FeEstimator::fit(input, FeEffects::TwoWay, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::UnbalancedPanelForTwoWay {
                n_obs: 4,
                n_entities: 2,
                n_periods: 2,
                expected: 4,
            }
        );
    }

    #[test]
    fn fe_estimator_fit_two_way_propagates_zero_variance_error() {
        // "year_dummy"はエンティティ間で変動しない（time FEと完全共線）。`fit()`が
        // within変換後に`validate_no_zero_variance_regressors`を呼んでいることの配線確認。
        // n=9（3エンティティ×3時点、n_entities=3・n_periods=3）に拡張し、
        // df_model=k(2)+neffects(3+3-1=5)=7・n>df_modelとなるサイズにする
        // （Issue #180の自由度検証を先に通過させるため）。"x_varying"は
        // entity×timeの交互作用項（entity_idx*time_idx）にして、加法分離可能な
        // 主効果のみの列（2-way demeanで機械的にゼロになる）にならないようにする
        // （Pythonで2-way demeanした結果、分散0.444...とゼロでないことを確認済み）。
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c"]);
        let time = strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let x_varying = vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0];
        let year_dummy = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let input = FeInput::from_columns(
            &y,
            &[x_varying, year_dummy],
            vec!["x_varying".to_string(), "year_dummy".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::TwoWay, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::ZeroVarianceAfterDemeaning {
                column: "year_dummy".to_string(),
            }
        );
    }

    #[test]
    fn fe_estimator_fit_propagates_invalid_confidence_level_error() {
        // 委譲先の`OlsEstimator::fit`が検証する`confidence_level`の範囲チェック
        // （`(0, 1)`の範囲外）が、`PanelError::WithinRegressionFailed`として正しく
        // 伝播することを確認する（`iv::two_sls`の同型テストに倣う）。
        let entity = strings(&["a", "a", "b", "b"]);
        let y = [1.0, 2.0, 3.0, 4.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, 1.5);

        assert_eq!(
            result.unwrap_err(),
            PanelError::WithinRegressionFailed {
                source: LeastSquaresError::Common(CommonError::InvalidConfidenceLevel {
                    confidence_level: 1.5,
                }),
            }
        );
    }

    #[test]
    fn fe_estimator_fit_wraps_ols_failure_as_within_regression_failed() {
        // Issue #180の自由度検証（`n <= df_model`）が`OlsEstimator::fit`自身の`n <= k`
        // チェックより常に厳しい（`df_model = k + neffects > k`）ため、`OlsEstimator::fit`
        // 側の観測数不足はこの経路では発生しえなくなった（`fe_estimator_fit_propagates_
        // insufficient_degrees_of_freedom_error`が先に弾く）。そのため、ここでは
        // `OlsEstimator::fit`固有の別の失敗——完全な多重共線性（`LeastSquaresError::
        // SingularMatrix`）——を踏ませる: x2はx1のちょうど2倍で、within変換
        // （線形変換）後も比例関係`x2_demeaned = 2 * x1_demeaned`が保たれ完全共線になる。
        // n=6（3エンティティ、singletonではない）・k=2でdf_model=2+3=5、n=6>5と
        // 自由度検証は通過するようにする。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x1 = vec![1.0, 3.0, 2.0, 6.0, 4.0, 10.0];
        let x2: Vec<f64> = x1.iter().map(|v| 2.0 * v).collect();
        let input = FeInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, 0.95);

        assert!(matches!(
            result.unwrap_err(),
            PanelError::WithinRegressionFailed { .. }
        ));
    }

    #[test]
    fn fe_estimator_fit_propagates_insufficient_degrees_of_freedom_error() {
        // n=4・n_entities=2・k=2でdf_model=k+n_entities=4となり、n<=df_modelのため
        // `OlsEstimator::fit`へ委譲する前に`PanelError::InsufficientDegreesOfFreedom`で
        // 弾かれることを確認する（Issue #180）。
        let entity = strings(&["a", "a", "b", "b"]);
        let y = [1.0, 2.0, 3.0, 4.0];
        let x1 = vec![1.0, 3.0, 2.0, 6.0];
        let x2 = vec![2.0, 5.0, 1.0, 9.0];
        let input = FeInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::InsufficientDegreesOfFreedom {
                n_obs: 4,
                n_entities: 2,
                n_periods: None,
                k: 2,
            }
        );
    }

    /// N=4（id: a,b,c,d）×T=3（t: 1,2,3）のバランスパネル。`fe_estimator_fit_one_way_
    /// matches_fixest_reference`/`fe_estimator_fit_two_way_matches_fixest_reference`の
    /// 両方で共有する（1-way/2-wayを同じデータで比較できるようにするため）。
    fn fixest_reference_input() -> (Vec<String>, Vec<String>, Vec<f64>, Vec<f64>) {
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]);
        let time = strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2", "3"]);
        let x = vec![1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 1.0, 3.0, 6.0, 4.0, 2.0, 1.0];
        let y = vec![
            5.0, 7.0, 10.0, 3.0, 8.0, 9.0, 6.0, 10.0, 15.0, 2.0, 5.0, 4.0,
        ];
        (entity, time, x, y)
    }

    #[test]
    fn fe_estimator_fit_one_way_matches_fixest_reference() {
        // Rの`fixest::feols(y ~ x | id)`（`id`: a/b/c/d各3観測）の実測値と数値比較する
        // （5.2節・fixestがFEのRクロスチェック参照実装）。期待値はRで独立に計算・検算済み
        // （`options(digits=15)`でフルの浮動小数点精度を取得、2026-09-12）。
        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, 0.95).unwrap();

        assert_eq!(fe.df_model(), 5); // k(1) + n_entities(4)
        assert_eq!(fe.df_resid(), 7); // n(12) - df_model(5)
        assert!((*fe.estimator().params().get(0, 0) - 1.402_777_777_777_78).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.432_598_838_244_034).abs() < 1e-6);
        assert!((*fe.t_stats().get(0, 0) - 3.242_675_785_889_31).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.014_200_386_789_949_8).abs() < 1e-6);
        assert!((*fe.conf_lower().get(0, 0) - 0.379_844_073_655_07).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 2.425_711_481_900_49).abs() < 1e-6);
        assert!((fe.aic() - 55.612_545_928_611_7).abs() < 1e-6);
        assert!((fe.bic() - 58.037_079_177_551_7).abs() < 1e-6);
        assert!((fe.r_squared_adj() - 0.661_606_689_860_115).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_matches_fixest_reference() {
        // 同じデータでの`fixest::feols(y ~ x | id + t)`（2-way）の実測値と数値比較する。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, 0.95).unwrap();

        assert_eq!(fe.df_model(), 7); // k(1) + neffects(n_entities(4)+n_periods(3)-1=6)
        assert_eq!(fe.df_resid(), 5); // n(12) - df_model(7)
        assert!((*fe.estimator().params().get(0, 0) - 0.822_429_906_542_056).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.227_239_295_931_651).abs() < 1e-6);
        assert!((*fe.t_stats().get(0, 0) - 3.619_223_969_033_19).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.015_231_948_369_008_1).abs() < 1e-6);
        assert!((*fe.conf_lower().get(0, 0) - 0.238_292_700_077_37).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.406_567_113_006_74).abs() < 1e-6);
        assert!((fe.aic() - 36.559_692_739_993_7).abs() < 1e-6);
        assert!((fe.bic() - 39.954_039_288_509_7).abs() < 1e-6);
        assert!((fe.r_squared_adj() - 0.930_619_212_222_08).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_with_no_regressors_estimates_fixed_effects_only_model() {
        // k=0（回帰変数なし、固定効果のみのモデル）でも`fit()`本体がエンドツーエンドに
        // 動くことを確認する境界ケース（`from_columns_with_no_regressors_succeeds`は
        // `FeInput`構築のみの検証で、`fit()`のdf調整（`df_model=neffects`のみになる）・
        // `r_squared_adj`/`aic`/`bic`計算までは通していなかった、rust-reviewer指摘）。
        // n=6・n_entities=3・k=0でdf_model=neffects=3・df_resid=3。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let y = [1.0, 3.0, 5.0, 7.0, 2.0, 4.0];
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, 0.95).unwrap();

        assert_eq!(fe.df_model(), 3);
        assert_eq!(fe.df_resid(), 3);
        assert_eq!(fe.estimator().params().nrows(), 0);
        assert_eq!(fe.std_errors().nrows(), 0);
        assert!(fe.aic().is_finite());
        assert!(fe.bic().is_finite());
        assert!(fe.r_squared_adj().is_finite());
    }

    #[test]
    fn fe_estimator_fit_pins_faer_global_parallelism_to_seq() {
        // Issue #283: `fit()`冒頭の`crate::parallelism::ensure_serial()`がfaerのグローバル
        // 並列度を`Par::Seq`へ引き戻すことの回帰ガード（panel系統代表、
        // `engine/src/panel/CLAUDE.md`「faerのグローバル並列度」参照）。
        faer::set_global_parallelism(faer::Par::rayon(0));

        let entity = strings(&["a", "a", "b", "b"]);
        let y = [1.0, 2.0, 3.0, 4.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let _ = FeEstimator::fit(input, FeEffects::OneWay, 0.95).unwrap();

        assert!(matches!(faer::get_global_parallelism(), faer::Par::Seq));
    }

    /// property-basedテスト。固定シナリオ
    /// （`within_transform_two_way_matches_closed_form_double_demeaning`、`N=T=2`）とは別に、
    /// `N != T`の非対称なバランスパネルでも「entity→time逐次quasi-demean」が閉形式の
    /// 二重デミーニングと一致することをランダムデータで検証する。閉形式側は
    /// `quasi_demean_column`/`within_transform_two_way`を一切経由せず素朴な二重ループで
    /// 独立に計算する（`engine/src/iv/CLAUDE.md`「自己参照的なオラクルは実装と同じ間違いを
    /// 複製する」と同じ教訓を踏まえ、実装の内部関数を再利用しない独立実装にしている）。
    mod proptests {
        use super::*;
        use proptest::collection;
        use proptest::prelude::*;

        /// entity外側・time内側の行順で、`n_entities × n_periods`の完全なバランスパネル
        /// （`y`はランダムな連続一様分布）を生成するストラテジ。
        fn balanced_panel_strategy() -> impl Strategy<Value = (usize, usize, Vec<f64>)> {
            (2..=5usize, 2..=5usize).prop_flat_map(|(n_entities, n_periods)| {
                (
                    Just(n_entities),
                    Just(n_periods),
                    collection::vec(-100.0f64..100.0, n_entities * n_periods),
                )
            })
        }

        /// `balanced_panel_strategy`の`y`と同じ行順（entity外側・time内側）の
        /// `entity`/`time`ラベル列を作る。
        fn entity_time_labels(n_entities: usize, n_periods: usize) -> (Vec<String>, Vec<String>) {
            let mut entity = Vec::with_capacity(n_entities * n_periods);
            let mut time = Vec::with_capacity(n_entities * n_periods);
            for i in 0..n_entities {
                for t in 0..n_periods {
                    entity.push(format!("e{i}"));
                    time.push(format!("t{t}"));
                }
            }
            (entity, time)
        }

        /// 閉形式`ỹ_it = y_it - ȳ_i. - ȳ_.t + ȳ..`を素朴な二重ループで独立に計算する
        /// オラクル（`y`は`entity_time_labels`と同じ行順、entity外側・time内側）。
        fn closed_form_two_way_demean(y: &[f64], n_entities: usize, n_periods: usize) -> Vec<f64> {
            let n = y.len();
            let grand_mean: f64 = y.iter().sum::<f64>() / n as f64;

            let entity_means: Vec<f64> = (0..n_entities)
                .map(|i| {
                    let sum: f64 = (0..n_periods).map(|t| y[i * n_periods + t]).sum();
                    sum / n_periods as f64
                })
                .collect();
            let time_means: Vec<f64> = (0..n_periods)
                .map(|t| {
                    let sum: f64 = (0..n_entities).map(|i| y[i * n_periods + t]).sum();
                    sum / n_entities as f64
                })
                .collect();

            let mut out = Vec::with_capacity(n);
            for i in 0..n_entities {
                for t in 0..n_periods {
                    out.push(y[i * n_periods + t] - entity_means[i] - time_means[t] + grand_mean);
                }
            }
            out
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            #[test]
            fn within_transform_two_way_matches_independent_closed_form_oracle(
                (n_entities, n_periods, y) in balanced_panel_strategy()
            ) {
                let (entity, time) = entity_time_labels(n_entities, n_periods);
                let input =
                    FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".to_string())
                        .unwrap();

                let (y_out, _) = within_transform_two_way(&input).unwrap();
                let expected = closed_form_two_way_demean(&y, n_entities, n_periods);

                for (actual, expected) in y_out.iter().zip(expected.iter()) {
                    let scale = y.iter().fold(1.0_f64, |acc, v| acc.max(v.abs()));
                    prop_assert!(
                        (actual - expected).abs() <= 1e-8 * scale,
                        "actual={actual}, expected={expected}"
                    );
                }
            }
        }
    }
}
