//! FEの入力データ型（`FeInput`）・within変換・推定量（`FeEstimator`）。
//!
//! `engine`はpolars/PyO3を知らない（`.claude/rules/rust-style.md`「責務分離」）。
//! `engine_pybind`がpolars DataFrameから`y`/`x`/`entity`/`time`を列ごとに抽出し、
//! それらの列を`FeInput::from_group_codes`に渡す。`FeInput`はwithin変換前の生データを
//! 保持するだけの入れ物で、`OlsInput`/`IvInput`と異なり`faer::Mat`は組み立てない
//! （within変換（`panel::common::quasi_demean_column`）が`&[f64]`の列単位で動くため）。
//!
//! 計算の定義式は`docs/methods/fe.md`、導出と実装上の判断は`docs/spec/fe-spec.md`が正本。
//! ここには各部品の契約と、コードを読むだけでは分からない理由だけを置く。
//!
//! ## `FeEstimator::fit`のパイプライン
//!
//! singleton検出 → within変換（2-wayはバランスパネル検証を内包）→ 自由度検証 → 分散ゼロの
//! 説明変数・定数の被説明変数の検出 → `confidence_level`検証 → `shared::least_squares::least_squares`
//! → 完全フィット検出 → `cov_type`別の共分散行列 → 自由度調整後の統計量の再計算。
//!
//! - FEはwithin変換後のデータに最小二乗をあてはめる（Frisch-Waugh-Lovell定理により点推定は
//!   FE推定量`β̂`と一致する）。`OlsEstimator`は経由しない: FEが使うのは`β̂`・残差・`(X̃'X̃)⁻¹`・
//!   適合度だけで、標準誤差・F検定は`cov_type`ごとにFE自身が計算し直すため。`x=[]`
//!   （固定効果のみのモデル）でも`least_squares`は0列を受理する。
//! - `OlsInput::from_columns`は`include_intercept=false`で作る（within変換で全体平均も差し引か
//!   れ、変換後データに切片は不要）。within R²・対数尤度は`gaussian_goodness_of_fit`
//!   （`has_intercept=false`、非中心化TSS）。
//! - `confidence_level`の範囲外と設計行列の特異性は`PanelError::WithinRegressionFailed`
//!   （`LeastSquaresError`を包む）。観測数不足（`n <= df_model`）は先に
//!   `InsufficientDegreesOfFreedom`で弾く。
//! - 1-way/2-wayは`FeEffects`で切り替える。
//!
//! ## within変換
//!
//! - **1-way**: `y`/各`x`列にentityでのquasi-demean（θ=1）を適用する。不均衡パネルも
//!   無条件でサポートする。
//! - **2-way**: バランスパネル必須。`time`が無ければ`PanelError::TwoWayRequiresTime`、
//!   不均衡なら`PanelError::UnbalancedPanelForTwoWay`。閉形式の二重デミーニング
//!   `ỹ_it = y_it - ȳ_i. - ȳ_.t + ȳ..`を別実装せず、「entityでquasi-demean → その結果を
//!   timeでquasi-demean」の2段階適用にしている。バランスパネルでのみ閉形式と一致する
//!   （導出は`fe-spec.md`3.1節）ため、不均衡パネルには適用しない。
//!
//! ## 事前検証（within変換の前後で行うもの）
//!
//! - **singleton**（`validate_no_singleton_groups_*`）: 観測数1のグループを、within変換の**前**に
//!   生の`entity`/`time`から数えて`PanelError::SingletonGroup`にする。変換後に全列ゼロの行として
//!   最小二乗側の特異行列エラーに現れる形にはしない。自動除外はしない。複数あれば観測順で
//!   最初の1件を報告する。2-wayはentity・timeを対称に検出し、`time`の存在確認を最初に行う。
//! - **分散ゼロの説明変数**（`validate_no_zero_variance_regressors`）: within変換後の列の分散が
//!   ゼロなら`PanelError::ZeroVarianceAfterDemeaning`。時間不変の変数（1-way）と、time FEと
//!   完全共線な列（2-way）を同じ検査で拾う（`column_is_zero_variance`）。
//! - **定数の被説明変数・完全フィット**: `fe-spec.md`1章（`OutcomeConstantWithinEntity`・`PerfectFit`）。
//!
//! ## 自由度とF統計量
//!
//! - `df_model = k + neffects`（`neffects`は1-wayなら`n_entities`、2-wayなら
//!   `n_entities + n_periods - 1`）、`df_resid = n - df_model`。素の最小二乗の
//!   t検定・調整済みR²・AIC/BICは`df_resid = n - k`を前提にするためFEには使えず、FE自身で
//!   計算する。t値・p値・信頼区間は、自由度`df_inference`のt分布（`Classical`/`Hc1`〜`Hc3`は
//!   `df_resid`、`Cluster`は`G-1`、`Dk`は`t_periods-1`）で
//!   `shared::inference`のヘルパーを使う。AIC/BICはペナルティ項の乗数を`k`ではなく`df_model`にする。
//! - **F統計量**（`f_statistic`/`f_p_value`）: 傾き係数`k`個が同時にゼロというWald F検定
//!   （`linearmodels.PanelOLS`と同じ定義。固定効果ダミーは検定対象に含めない。`fixest`の
//!   `fitstat(m, "f")`はFEダミーを含むモデル全体の検定なので定義が異なる）。`cov_type`別の
//!   `cov_params`と`β̂`を`shared::wald::wald_f_test`（`k_constant=0`）に渡す。`k=0`は`None`。
//!   失敗は`PanelError::FTestFailed`。入力だけで構造的な特異性が判定できる場合
//!   （Clusterの`G <= k`、Dkの`t_periods <= k`）は共分散計算の前に
//!   `InsufficientClustersForInference`/`InsufficientDkPeriodsForInference`で弾く。2グループ
//!   （Clusterの`G=2`・Dkの`t_periods=2`）で全グループが各側に1観測ずつだとwithin変換で
//!   スコアが恒等的にゼロになるため`DegenerateClusterTwoGroups`/`DegenerateDkTwoPeriods`で弾く
//!   （`two_group_split_is_degenerate`。2-wayではtime方向の同じ構造も対象）。数値的な悪条件は
//!   `FTestFailed`が受ける。
//! - `linearmodels.PanelOLS`は`aic`・`bic`を提供しないため、この2つの検証は`fixest`との
//!   比較のみで行う。
//!
//! ## パネル固有R²
//!
//! `r_squared_within`/`r_squared_between`/`r_squared_overall`の3つ（bareの`adj_r_squared`は持たない）。
//!
//! - **within**: 実際に使ったFE構造でdemeanした残差（1-wayはentity、2-wayはentity+time）で、
//!   `gaussian_goodness_of_fit(.., has_intercept=false)`の`r_squared`。`linearmodels`の
//!   `rsquared_within`は2-wayでもentityのみのdemeanで固定なので、2-wayでは意図的に
//!   食い違う（2-wayは`fixest`の`fitstat(model, "wr2")`と比較する）。
//! - **between/overall**: `linearmodels`の`_rsquared`と一致させる。FEの`x`は定数列を含められない
//!   ため、非中心化TSSの分岐が常に適用される。`overall`は元の`y`・`x`に`β̂`だけを当てはめた残差
//!   （固定効果を含まない。`FeEstimator::residuals()`とは別物）、`between`はエンティティ平均への
//!   当てはめ。**`between`に`T_i`ベースの重みを付けてはいけない**: `linearmodels`の
//!   `_prepare_between`は不均衡パネル用の重みを計算するが、サンプルウェイト未指定なら
//!   `_rsquared`が`1.0`に上書きし、FEは`weights`を持たない。
//! - どちらも`TSS <= 0`なら`0.0`。
//!
//! ## `cov_type`対応（`FeCovType`）
//!
//! `shared::covariance`の`classical_cov_params`/`hc_cov_params`/`cluster_cov_params`は
//! そのまま使えない。計算は`panel/common.rs`の`panel_cov_params`（`PanelStructure::FixedEffects`）が
//! 担い、以下の点でOLSと式が異なる（導出は`fe-spec.md`3.3節）。
//!
//! 1. **HC0は提供しない**（小標本補正なしのサンドイッチはパネル/FEでは提供しない仕様判断。
//!    他パッケージの有無は理由にしない）。
//! 2. **HC1〜HC3**: HC1の補正は`n/(n-k)`ではなく`n/df_resid`（`neffects`込み）。HC2/HC3の
//!    レバレッジはwithin変換後の設計行列のものではなく、固定効果ダミーを含むLSDV相当の
//!    設計行列のもの（`leverage_full`）。
//! 3. **Cluster**: 補正は`(G/(G-1))×((n-1)/(n-K))`で、`K`はfixestの`ssc()`既定
//!    （`K.fixef="nonnested"`）に合わせたFE固有のロジック（`fe_cluster_k_correction`）。
//!    蓄積`Σ_g S_g S_g'`は`cluster_meat`、補正式は`cluster_correction`を共有する。
//!
//! **t値・p値・信頼区間・F検定の自由度は、`cov_type=Cluster`のとき`G-1`、`Dk`のとき
//! `t_periods-1`**（fixestの`t.df="min"`）。`df_resid`自体（σ̂²・調整済みR²・AIC/BIC）は
//! `cov_type`によらずパネル自由度調整済みの値のまま。`cov_type`の既定
//! （`"cluster"`、entity単位）は`engine_pybind`の責務で、`FeEstimator::fit`は呼び出し側が
//! 渡す`FeCovType`を使う（`Cluster { groups: None }`のときだけentityを使う）。
//!
//! ## Driscoll-Kraay型パネルHAC（`FeCovType::Dk`）
//!
//! OLSの`CovType::Dk`は異なるエンティティの観測を単一の時系列カーネルに混ぜるため、パネルでは
//! 別アルゴリズムとして実装する。
//!
//! - **式**: `Cov(β̂) = (t_periods/(t_periods-1)) × ((n-1)/(n-K)) × (X̃'X̃)⁻¹ Ŝ (X̃'X̃)⁻¹`、
//!   `Ŝ = Σ_t ξ_t ξ_t' + Σ_{l=1}^{bw} w_l (ξ_t ξ_{t-l}' + ξ_{t-l} ξ_t')`、
//!   `ξ_t = Σ_{i: time_i=t} x̃_i ε̂_i`。`x̃`はwithin変換後の設計行列。fixestの`vcov="DK"`に
//!   合わせ、`K = df_model`（DKにはクラスター変数が無くネスト判定が起きない）、`t_periods`を
//!   クラスターの`G`と同じ役割（`G/(G-1)`補正・自由度`G-1`）で使う。`t_periods < 2`は
//!   `resolve_dk_bandwidth`が`InsufficientDkPeriods`で拒否する。
//! - **カーネル**: Bartlett限定（`w_l = 1 - l/(bw+1)`）。OLSの`CovType::Dk`と揃える。
//! - **バンド幅**: `Some(bw)`なら`0 <= bw < t`を検証（`InvalidDkBandwidth`）。`None`なら
//!   `floor(4*(t/100)^(2/9))`（`linearmodels`の既定。OLSの`hac_lags`が観測数`n`ベースなのに
//!   対しDKは時点数`t`ベース）。
//! - **時系列順序**: `time`は`TimeKeys`で、時点のラベルと**時間順のコード**を持つ。順序は列の値の
//!   順序（数値は数値順、日付・日時は時系列順、`Enum`は定義順）で、呼び出し側が
//!   `TimeKeys`のコンストラクタで指定する。辞書順は`1, 10, 11, 2, ...`と時間順がずれて標準誤差が
//!   黙って変わるため使わない（`TimeKeys::lexicographic`・`FeInput::from_columns`は辞書順が時間順と
//!   一致するISO 8601日付やゼロ埋め年月向けの簡便版）。
//! - **1-way/2-wayとも対応**。DKの時点列は`FeCovType::Dk.time`として常に明示的に受け取る
//!   （`FeInput.time_keys()`にはフォールバックしない。どの列がDKの時点か呼び出し側が
//!   明示しない設計は、意図と違う列が選ばれても気づけないため。`FeInput.time_keys()`は
//!   2-wayの固定効果の時間次元専用）。
//!
//! ## 固定効果自体（α_i）の復元（`fixed_effects()`）
//!
//! `fit()`の戻り値本体には含めない別メソッド。`FeEstimator`が保持する変換前の入力と
//! `params()`から呼び出し時に計算する。
//!
//! - **1-way**: `α_i = ȳ_i. - x̄_i.'β̂`（一意）。
//! - **2-way**: `α_i`に定数を足し`γ_t`から引いても予測値が変わらないため一意に決まらない。
//!   `fe-spec.md`3.5節の式をentity/timeにそのまま当てはめると大域平均を二重計上するため、
//!   **`time`の時間順で最初の値`t_ref`を基準に`γ_{t_ref}=0`とし、`α_i`に大域的な水準を
//!   吸収させる**（導出は`fe-spec.md`3.5節）。`fixest::fixef()`は基準時点を観測順で最初に
//!   現れた値に取るように見えるため、数値が一致するのは両者の基準時点が一致する入力に限る。
//!   本実装は観測順に依存しない時間順の規約を優先する。

use std::collections::BTreeMap;

use faer::Mat;
use statrs::distribution::StudentsT;

use crate::linear::common::LeastSquaresError;
use crate::linear::ols::OlsInput;
use crate::panel::common::{
    EffectNames, PanelCov, PanelCovInput, PanelCovType, PanelEffects, PanelError, PanelStructure,
    TimeKeys, WithinTransform, ensure_no_singleton_groups, ensure_no_zero_variance_columns,
    panel_cov_params, validate_balanced_panel, validate_panel_input,
};
use crate::panel::re::{ReEffects, ReInput};
use crate::shared::error::CommonError;
use crate::shared::goodness_of_fit::{gaussian_goodness_of_fit, is_perfect_fit};
use crate::shared::group_codes::GroupCodes;
use crate::shared::inference::{self, InferenceTable, inference_table};
use crate::shared::least_squares::{LeastSquaresFit, least_squares};
use crate::shared::validation::column_is_zero_variance;
use crate::shared::wald::wald_f_test;

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
    /// 各行の時点コード（時間順、長さ`n`）。2-way FE（entity + time FE）を
    /// 指定しない場合は`None`（`panel-common.md`1.1節）。
    time: Option<TimeKeys>,
    /// 被説明変数名。
    dep_var_name: String,
    /// `entity`の整数コード（キー付き。構築時に一度だけ作る、`GroupCodes`のdocコメント参照）。
    entity_codes: GroupCodes,
    /// 効果の次元ごとの列名（エラーの`column`用。計算には使わない）。
    effect_names: EffectNames,
}

impl FeInput {
    /// 列ごとの`Vec<f64>`/`Vec<String>`（`engine_pybind`がpolars DataFrameから抽出済み）
    /// から`FeInput`を組み立てる。
    ///
    /// # Errors
    /// 検査の順序は`validate_panel_input`（`panel/common.rs`、FE/REで共通）が正本。
    /// - いずれかの`x_columns`の長さが`y`と一致しない場合は
    ///   `PanelError::Common(CommonError::DimensionMismatch)`
    /// - `entity`の長さが`y`と一致しない場合は
    ///   `PanelError::IdentifierDimensionMismatch { dimension: PanelDimension::Entity, .. }`
    /// - `time`が`Some`で、その長さが`y`と一致しない場合は
    ///   `PanelError::IdentifierDimensionMismatch { dimension: PanelDimension::Time, .. }`
    ///
    /// within変換の実施・singleton検出・分散ゼロ検証・バランスパネルの検証は行わない
    /// （いずれも`fit()`側が担う、`docs/spec/fe-spec.md`）。
    ///
    /// # パニックについて
    /// `x_names.len() != x_columns.len()`の場合は`debug_assert!`でパニックする
    /// （`OlsInput::from_columns`と同じ理由: 呼び出し側`engine_pybind`の実装バグでしか
    /// 起こり得ない内部契約であり、実データに起因する`ValidationError`とは性質が異なる）。
    #[cfg(test)]
    pub(crate) fn from_columns(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        entity: &[String],
        time: Option<&[String]>,
        dep_var_name: String,
    ) -> Result<Self, PanelError> {
        Self::from_columns_ordered(
            y,
            x_columns,
            x_names,
            entity,
            time.map(|t| TimeKeys::lexicographic(t.to_vec())),
            dep_var_name,
        )
    }

    /// `from_columns`の`time`を、順序を持つ`TimeKeys`で受け取る版。時点の順序を
    /// ラベルの辞書順ではなく列の値の順序にしたい場合（整数・日付等）は、こちらを使う
    /// （`TimeKeys`のdoc参照）。`from_columns`はラベルの辞書順を時間順とみなす簡便版。
    ///
    /// # Errors
    /// `from_columns`と同じ。
    #[cfg(test)]
    pub(crate) fn from_columns_ordered(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        entity: &[String],
        time: Option<TimeKeys>,
        dep_var_name: String,
    ) -> Result<Self, PanelError> {
        Self::from_group_codes(
            y,
            x_columns,
            x_names,
            GroupCodes::from_labels(entity),
            time,
            dep_var_name,
        )
    }

    /// `from_columns_ordered`の`entity`を、コード化済みの`GroupCodes`で受け取る版。
    /// `engine_pybind`は、行ごとの`String`を作らずに`GroupCodes::from_factorized`で作ったコードを
    /// 渡す。`from_columns`/`from_columns_ordered`は、この関数にラベルから作ったコードを渡す
    /// 簡便版。
    ///
    /// `entity`はキー付き（`from_labels`/`from_factorized`）であること。固定効果のキーに使う。
    ///
    /// # Errors
    /// `from_columns`と同じ。加えて、`entity`がキー無しのコードの場合は`PanelError::Common`
    /// （`CommonError::Internal`、呼び出し側の実装誤り）。
    pub fn from_group_codes(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        entity: GroupCodes,
        time: Option<TimeKeys>,
        dep_var_name: String,
    ) -> Result<Self, PanelError> {
        validate_panel_input(y, x_columns, &x_names, &entity, time.as_ref())?;

        Ok(Self {
            y: y.to_vec(),
            x: x_columns.to_vec(),
            x_names,
            time,
            dep_var_name,
            entity_codes: entity,
            effect_names: EffectNames::default(),
        })
    }

    /// REの分散成分推定（`swamy_arora_variance_components`）用のFE入力を、
    /// `ReInput`の`y`・`x`・`entity`（2-wayなら`time`も）から作る。`ReInput`が既に持つエンティティコードを
    /// 再利用し作り直さない。`ReInput::from_group_codes`が同じ次元検証を済ませているため
    /// 検証は不要で、`entity`とコードの対応も`ReInput`が保証する（別々の引数で受けて
    /// 食い違う余地を作らない）。
    ///
    /// `columns`は`input.x()`のうちFEに渡す列の添字（昇順）。REは時間不変の列を許容するため、
    /// 内部のFE推定にはwithin変換後の分散がある列だけを渡す（`time_varying_regressor_indices`）。
    /// FEは分散ゼロの列を拒否するため、全列を渡すと時間不変の列でREごと失敗する。
    ///
    /// `effects`が`TwoWay`のときだけ`ReInput`の`time`（2-wayの時間次元）を引き継ぐ
    /// （1-wayでは`time`があっても使わない）。
    pub(crate) fn from_re_input(input: &ReInput, columns: &[usize], effects: ReEffects) -> Self {
        Self {
            y: input.y().to_vec(),
            x: columns.iter().map(|&j| input.x()[j].clone()).collect(),
            x_names: columns
                .iter()
                .map(|&j| input.x_names()[j].clone())
                .collect(),
            time: match effects {
                ReEffects::OneWay => None,
                ReEffects::TwoWay => input.time_keys().cloned(),
            },
            dep_var_name: input.dep_var_name().to_string(),
            entity_codes: input.entity_codes().clone(),
            effect_names: input.effect_names().clone(),
        }
    }

    /// 効果の次元の列名を設定する（`engine_pybind`が`effects`の各要素を渡す）。エラーの`column`に
    /// 載るだけで、計算には影響しない。`time`が`None`なら2つ目の名前は既定のまま。
    pub fn with_effect_names(mut self, entity: String, time: Option<String>) -> Self {
        self.effect_names = EffectNames::new(entity, time);
        self
    }

    /// 効果の次元の列名。
    pub(crate) fn effect_names(&self) -> &EffectNames {
        &self.effect_names
    }

    /// `entity`の整数コード。
    pub(crate) fn entity_codes(&self) -> &GroupCodes {
        &self.entity_codes
    }

    /// ユニークなエンティティ数。
    pub fn n_entities(&self) -> usize {
        self.entity_codes.n_groups()
    }

    /// `time`の整数コード（1-way FEで`time`が無ければ`None`）。
    pub(crate) fn time_codes(&self) -> Option<&GroupCodes> {
        self.time.as_ref().map(TimeKeys::codes)
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

    /// ユニークなエンティティのラベル（コード順＝辞書順）。
    pub fn entity_labels(&self) -> &[String] {
        self.entity_codes.keys()
    }

    /// 時点のラベルと時間順のコード。1-way FEでは`None`。
    pub fn time_keys(&self) -> Option<&TimeKeys> {
        self.time.as_ref()
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

/// FEの固定効果の方向。[`PanelEffects`]の別名（モジュールdoc「`FeEstimator::fit`のパイプライン」参照）。
pub type FeEffects = PanelEffects;

/// FEが対応する`cov_type`。[`PanelCovType`]の別名（モジュールdoc「`cov_type`対応」参照）。
pub type FeCovType = PanelCovType;

/// 固定効果自体（α_i、2-wayはγ_tも）の復元結果（`FeEstimator::fixed_effects`、
/// `fe-spec.md`3.5節）。モジュールdoc「固定効果自体（α_i）の復元」参照。
///
/// `BTreeMap<String, f64>`（ID→効果）を使う理由: `fe-spec.md`3.5節のPython API形状
/// （1-wayは`dict[str, float]`、2-wayは`dict[str, dict[str, float]]`）にそのまま対応でき、
/// かつキー順序が決定的になる（`HashMap`だとプロセスごとのハッシュシードで反復順序が
/// 変わりうる、他のグループ集約と同じ理由）。
#[derive(Debug, Clone, PartialEq)]
pub enum FixedEffects {
    /// エンティティID → α_i。
    OneWay(BTreeMap<String, f64>),
    /// entity効果・time効果それぞれのID→効果（`fe-spec.md`3.5節のPython API形状のトップレベルキー
    /// `"entity"`/`"time"`に対応）。
    TwoWay {
        entity: BTreeMap<String, f64>,
        /// 時点ID → γ_t。**時点の昇順**（値の順序、`TimeKeys`のdoc参照）に並べ、先頭の
        /// 時点が基準（`γ=0`）。文字列の辞書順ではないため`BTreeMap`にしない。
        time: Vec<(String, f64)>,
    },
}

/// FEの推定結果。`within`変換したデータに最小二乗をあてはめ、`cov_type`
/// ・自由度調整を反映した標準誤差等を計算し直す
/// （モジュールdoc「`FeEstimator::fit`のパイプライン」「自由度調整」「`cov_type`対応」参照）。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」）。
#[derive(Debug)]
pub struct FeEstimator {
    input: FeInput,
    effects: FeEffects,
    cov_type: FeCovType,
    /// within変換済みの`y`・設計行列・係数名（切片なし）。
    within_input: OlsInput,
    /// FE推定量`β̂`（within変換後の最小二乗解、Frisch-Waugh-Lovell定理）。
    params: Mat<f64>,
    /// within変換後の回帰の残差（固定効果込み）。
    residuals: Mat<f64>,
    /// 正規誤差を仮定した対数尤度。`SSR/n`のみに依存しdf非依存の式（AIC/BICの罰則項の
    /// 乗数だけ`k`から`df_model`に差し替える、モジュールdoc参照）。
    log_likelihood: f64,
    /// パネル自由度調整後のモデル自由度（`k + neffects`、`fe-spec.md`3.2節）。
    df_model: usize,
    /// パネル自由度調整後の残差自由度（`n - df_model`、`fe-spec.md`3.2節）。
    /// `σ̂²`・調整済みR²・AIC/BICはこの値を使う。
    df_resid: usize,
    /// t検定・信頼区間・F検定に使う自由度。`cov_type=Cluster`のとき`G-1`、`Dk`のとき
    /// `t_periods-1`（fixestの`ssc()`既定`t.df="min"`）。それ以外
    /// （Classical/HC1-3）は`df_resid`と同じ値。
    df_inference: usize,
    /// `cov_type=Dk`のとき、実際に使われたバンド幅（`bandwidth`の明示指定値、または未指定時に
    /// `floor(4*(t/100)^(2/9))`で自動計算した値）。`FeCovType::Dk`の`bandwidth`はユーザー指定値の
    /// まま変更しないため別フィールドで保持する。`Dk`以外では`None`。
    dk_bandwidth_used: Option<usize>,
    /// `time`のユニーク数。2-wayのみ`Some`、1-wayは`None`（2-wayはバランスパネル必須の
    /// ため、各entityの観測数とも一致する）。
    n_periods: Option<usize>,
    std_errors: Mat<f64>,
    /// `cov_type`別の係数の分散共分散行列 `(k, k)`。`std_errors`はこの対角成分の平方根。
    cov_params: Mat<f64>,
    test_stats: Mat<f64>,
    p_values: Mat<f64>,
    conf_lower: Mat<f64>,
    conf_upper: Mat<f64>,
    /// 実際に使ったFE構造でdemeanしたR²（1-wayはentityのみ、2-wayはentity+time）。
    /// モジュールdoc「パネル固有R²」参照。
    r_squared_within: f64,
    /// エンティティ平均ベースのR²（linearmodelsの`rsquared_between`と完全一致、
    /// モジュールdoc参照）。
    r_squared_between: f64,
    /// 固定効果の切片項を含めないR²（linearmodelsの`rsquared_overall`と完全一致、
    /// モジュールdoc参照）。
    r_squared_overall: f64,
    aic: f64,
    bic: f64,
    /// 傾き係数`k`個の同時Wald F検定の`(F統計量, p値)`（素の最小二乗の`f_statistic`とは
    /// 異なりFE用にpanel自由度調整済み・`cov_type`反映済み、モジュールdoc「自由度とF統計量」の
    /// F統計量節参照）。`k=0`なら`None`。
    f_test: Option<(f64, f64)>,
}

impl FeEstimator {
    /// `input`を`effects`が指定する方向でwithin変換した上で最小二乗をあてはめ
    /// （`shared::least_squares::least_squares`）、FEを推定する。パネル自由度調整（`fe-spec.md`3.2節）・`cov_type`対応
    /// （3.1節・3.2節）を反映した標準誤差・t値・p値・信頼区間・AIC/BIC、
    /// パネル固有R²（2.3節）を計算し直す（モジュールdoc「自由度とF統計量」
    /// 「`cov_type`対応」「パネル固有R²」参照）。
    ///
    /// パイプライン: 2-wayの時点コードの解決（`two_way_time`。2-wayで`time`が無ければ
    /// `PanelError::TwoWayRequiresTime`。以降の検証より先に評価する）→ singleton検出
    /// （`validate_no_singleton_groups_one_way`/`validate_no_singleton_groups_two_way`）
    /// → within変換（`within_transform_one_way`/`within_transform_two_way`、
    /// 2-wayはバランスパネル検証を内包）→ 自由度検証 → 分散ゼロ検出
    /// （`validate_no_zero_variance_regressors`）→ `confidence_level`検証 → 最小二乗
    /// （切片なし。理由はモジュールdoc参照）→ `cov_type`別の共分散行列の計算 → 自由度調整後の統計量の再計算。
    ///
    /// # Errors
    /// - `effects=TwoWay`で`input.time_keys()`が`None`の場合は`PanelError::TwoWayRequiresTime`
    /// - singletonグループが見つかった場合は`PanelError::SingletonGroup`
    /// - `effects=TwoWay`でバランスパネルでない場合は`PanelError::UnbalancedPanelForTwoWay`
    /// - パネル自由度調整後の残差自由度（`df_resid`）が正にならない場合は
    ///   `PanelError::InsufficientDegreesOfFreedom`
    /// - within変換後に分散ゼロの説明変数がある場合は`PanelError::ZeroVarianceAfterDemeaning`
    /// - `cov_type=Cluster`でクラスター数が2未満・傾き係数の数以下の場合は
    ///   `PanelError::Common`（`CommonError::InsufficientClusters`/
    ///   `InsufficientClustersForInference`）
    /// - `cov_type=Dk`の`time`（DKの時点列）の長さが観測数と異なる場合は
    ///   `PanelError::IdentifierDimensionMismatch`
    /// - `cov_type=Dk`で時点数が2未満なら
    ///   `PanelError::InsufficientDkPeriods`、`bandwidth`が不正なら
    ///   `PanelError::InvalidDkBandwidth`、時点数が傾き係数の数以下なら
    ///   `PanelError::InsufficientDkPeriodsForInference`、時点数が2で全エンティティ（2-wayでは
    ///   全時点でも）が2時点に1観測ずつなら`PanelError::DegenerateDkTwoPeriods`
    /// - `cov_type=Cluster`でクラスター数が2で全エンティティ（2-wayでは全時点でも）が
    ///   2クラスターに1観測ずつなら`PanelError::DegenerateClusterTwoGroups`
    /// - `confidence_level`が`(0, 1)`の範囲外、または設計行列が特異な場合は
    ///   `PanelError::WithinRegressionFailed`
    /// - F検定の共分散部分行列が数値的にほぼ特異な場合は`PanelError::FTestFailed`
    pub fn fit(
        input: FeInput,
        effects: FeEffects,
        cov_type: FeCovType,
        confidence_level: f64,
    ) -> Result<Self, PanelError> {
        // faerのグローバル並列度をPar::Seqに固定する（`crate::shared::parallelism`。
        // `cargo test -p engine`でFeEstimator::fitを直接叩く経路でも担保するためここで呼ぶ、`engine/src/panel/CLAUDE.md`「faerの
        // グローバル並列度」参照）。
        crate::shared::parallelism::ensure_serial();

        // 2-wayの時点コード（1-wayは`None`）。`time`が無いのに2-wayを指定した入力はここで
        // 弾き、以降は`Some`かどうかだけで1-way/2-wayを分岐する（`effects`と`input.time`の
        // 食い違いを後段で`expect`せずに済む）。
        let two_way_time = two_way_time(effects, &input)?;

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
        let n_entities = input.entity_codes().n_groups();
        let n_periods = two_way_time.map(GroupCodes::n_groups);
        let k = input.x_names().len();
        // `neffects`: entityダミー・timeダミーの実効パラメータ数（`fe-spec.md`3.2節）。2-wayは両者の
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

        // 被説明変数もwithin変換後に分散が残らなければ、固定効果が説明するものが無い
        // （個体内で定数、2-wayなら個体効果と時点効果だけで説明される）。説明変数と同じ
        // 相対閾値で判定する。
        if column_is_zero_variance(input.y(), &y) {
            return Err(PanelError::OutcomeConstantWithinEntity);
        }

        // `OlsInput::from_columns`が返しうる`LeastSquaresError::Common(DimensionMismatch)`は
        // ここでは理論上到達不能: `y`/`x`はどちらも`within_transform_*`が`input.y()`/
        // `input.x()`（`FeInput::from_group_codes`が既に同じ長さであることを検証済み）から
        // 1対1で生成した同じ長さの列であり、この関数内で長さがずれる操作をしていない。
        // それでも`Result`を返す契約（`from_columns`のシグネチャ）をそのまま守り、
        // `unwrap`はしない（`.claude/rules/rust-style.md`「テスト」の「理論上到達不能でも
        // `Result`化する」方針に揃える）。
        let within_input = OlsInput::from_columns(
            &y,
            &x,
            input.x_names().to_vec(),
            false,
            input.dep_var_name().to_string(),
        )
        .map_err(|source| PanelError::WithinRegressionFailed { source })?;

        if !(confidence_level > 0.0 && confidence_level < 1.0) {
            return Err(PanelError::WithinRegressionFailed {
                source: CommonError::InvalidConfidenceLevel { confidence_level }.into(),
            });
        }

        // `β̂`・残差・`(X̃'X̃)⁻¹`は最小二乗の部品を直接呼んで求める（`cov_type`ごとの標準誤差は
        // 下記でFE自身が計算し直すため、`OlsEstimator`の推論は使わない。モジュールdoc
        // 「`cov_type`対応」参照）。`x=[]`（固定効果のみのモデル）だとwithin変換後の設計行列`x`も
        // 0列になり`k=0`になるが、`least_squares`は0列も受理する。
        let LeastSquaresFit {
            params,
            residuals: residual_mat,
            xtx_inv,
        } = least_squares(within_input.x(), within_input.y()).map_err(|err| {
            PanelError::WithinRegressionFailed {
                source: LeastSquaresError::from_rank_deficient(
                    err,
                    within_input.x(),
                    within_input.param_names(),
                ),
            }
        })?;

        // `cov_type`別の共分散行列の計算に使う共通の材料（within変換後の設計行列、残差・SSR）。
        let x_mat = within_input.x();
        let residuals: Vec<f64> = (0..n).map(|i| *residual_mat.get(i, 0)).collect();
        let ssr: f64 = residuals.iter().map(|r| r * r).sum();

        // within回帰の残差が丸め誤差の範囲で0なら完全フィット。標準誤差とF検定が定義できない。
        // within変換後の`y`の平均は0なので、総平方和は非中心化（`Σỹ²`）。
        let sst_within: f64 = y.iter().map(|v| v * v).sum();
        if is_perfect_fit(ssr, sst_within, k) {
            return Err(PanelError::WithinRegressionFailed {
                source: CommonError::PerfectFit.into(),
            });
        }

        // `cov_type`別の共分散行列と、t検定・信頼区間・F検定の自由度`df_inference`
        // （`Cluster`は`G-1`、`Dk`は`t_periods-1`、それ以外は`df_resid`。標準誤差のスケールに使う
        // `K`（`fe_cluster_k_correction`）とは別軸の値）。FE固有の補正（フルレバレッジ・
        // fixest流のK・2グループ退化の検査）は`PanelStructure::FixedEffects`で渡す。
        let PanelCov {
            cov_params,
            df_inference,
            dk_bandwidth_used,
        } = panel_cov_params(
            &cov_type,
            &PanelCovInput {
                x: x_mat,
                residuals: &residuals,
                xtx_inv: &xtx_inv,
                df_resid,
                df_model,
                n_slopes: k,
                entity: input.entity_codes(),
                structure: PanelStructure::FixedEffects { two_way_time },
            },
        )?;

        // `StudentsT::new`は自由度が正でない場合に失敗するが、`df_inference`は
        // `df_resid >= 1`（関数冒頭の`n <= df_model`検証）・`n_groups - 1 >= 1`
        // （`validate_cluster_groups`が`n_groups >= 2`を保証）・`t_periods - 1 >= 1`
        // （`resolve_dk_bandwidth`が`PanelError::InsufficientDkPeriods`で`t_periods < 2`を
        // 拒否済み、`fe_estimator_fit_one_way_hac_with_single_time_period_is_rejected`参照）の
        // いずれかであり理論上到達不能（`ReEstimator::fit`と同じ「保証済みの不変条件に対する
        // 防御的`Result`化」、`.claude/rules/rust-style.md`「テスト」参照）。
        let t_dist = StudentsT::new(0.0, 1.0, df_inference as f64)
            .map_err(|e| CommonError::internal(e.to_string()))?;
        let InferenceTable {
            std_errors,
            test_stats,
            p_values,
            conf_lower,
            conf_upper,
        } = inference_table(&params, &cov_params, &t_dist, confidence_level);

        // パネル固有R²（モジュールdoc「パネル固有R²」参照）。within R²は実際に使ったFE構造で
        // demeanした残差ベースで、切片なしのwithin変換後データへの最小二乗の適合度
        // （`has_intercept=false`、非中心化TSS）がそのままこの定義と一致する。
        // between/overallはlinearmodelsの`_rsquared`と完全一致させるため、変換前の元の
        // `y`/`x`から独立に計算し直す。
        let fit_stats = gaussian_goodness_of_fit(within_input.y(), ssr, k, false);
        let r_squared_within = fit_stats.r_squared;
        let r_squared_between =
            fe_r_squared_between(input.y(), input.x(), &params, input.entity_codes());
        let r_squared_overall = fe_r_squared_overall(input.y(), input.x(), &params);

        // `log_likelihood`自体は`SSR/n`のみに依存しdf非依存の式のためそのまま再利用できる
        // （モジュールdoc参照）。ペナルティ項の乗数だけ`k`から`df_model`に差し替える。
        let log_likelihood = fit_stats.log_likelihood;
        let aic = -2.0 * log_likelihood + 2.0 * (df_model as f64);
        let bic = -2.0 * log_likelihood + (n as f64).ln() * (df_model as f64);

        // F統計量（モジュールdoc「自由度とF統計量」）: 傾き係数`k`個の
        // 同時Wald検定。FEの`cov_params`（`cov_type`別、上で計算済み）・`df_inference`
        // （`cov_type=Cluster`/`Dk`のときだけ`df_resid`から切り替わる、`shared::wald::wald_f_test`の
        // `df_inference`引数と同じ扱い）を使う。`k_constant=0`（FEに切片は無い、上記
        // `OlsInput::from_columns`呼び出しと同じ理由）。
        let f_test = if k == 0 {
            // 説明変数が無いモデル。検定対象が存在しないため`OlsEstimator::fit`同様
            // 定義なし（`None`）。
            None
        } else {
            Some(
                wald_f_test(&params, &cov_params, 0, k, df_inference).map_err(|source| {
                    PanelError::FTestFailed {
                        source: source.into(),
                    }
                })?,
            )
        };

        Ok(Self {
            input,
            effects,
            cov_type,
            within_input,
            params,
            residuals: residual_mat,
            log_likelihood,
            df_model,
            df_resid,
            df_inference,
            dk_bandwidth_used,
            n_periods,
            std_errors,
            cov_params,
            test_stats,
            p_values,
            conf_lower,
            conf_upper,
            r_squared_within,
            r_squared_between,
            r_squared_overall,
            aic,
            bic,
            f_test,
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

    /// 標準誤差の計算に使った`cov_type`。
    pub fn cov_type(&self) -> &FeCovType {
        &self.cov_type
    }

    /// within変換済みの`y`・設計行列・係数名（切片なし）。`param_names()`・`dep_var_name()`・
    /// `nobs()`・`k()`もここから読む。
    pub fn within_input(&self) -> &OlsInput {
        &self.within_input
    }

    /// FE推定量`β̂`（k, 1）。within変換後の回帰の最小二乗解で、`within_input().param_names()`と
    /// 対応する。
    pub fn params(&self) -> &Mat<f64> {
        &self.params
    }

    /// within変換後の回帰の残差（n, 1）。
    pub fn residuals(&self) -> &Mat<f64> {
        &self.residuals
    }

    /// 対数尤度（正規誤差を仮定した最尤推定量`σ̂² = SSR/n`ベース、`df`非依存）。
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// パネル自由度調整後のモデル自由度（`k + neffects`、`fe-spec.md`3.2節）。
    pub fn df_model(&self) -> usize {
        self.df_model
    }

    /// パネル自由度調整後の残差自由度（`n - df_model`、`fe-spec.md`3.2節）。
    pub fn df_resid(&self) -> usize {
        self.df_resid
    }

    /// t検定・信頼区間・F検定に使う自由度（`cov_type=Cluster`のとき`G-1`、`Dk`のとき
    /// `t_periods-1`、それ以外は`df_resid`と同じ）。
    pub fn df_inference(&self) -> usize {
        self.df_inference
    }

    /// `cov_type=Dk`のとき、実際に使われたバンド幅（`bandwidth`の明示指定値、または未指定時に
    /// 経験則で自動計算した値）。`Dk`以外は`None`。
    pub fn dk_bandwidth_used(&self) -> Option<usize> {
        self.dk_bandwidth_used
    }

    /// `time`のユニーク数（2-wayのみ`Some`、1-wayは`None`）。
    pub fn n_periods(&self) -> Option<usize> {
        self.n_periods
    }

    /// `cov_type`別に計算し直した標準誤差（`(k, 1)`、`params()`と対応）。
    pub fn std_errors(&self) -> &Mat<f64> {
        &self.std_errors
    }

    /// `cov_type`別に計算し直した係数の分散共分散行列（`(k, k)`、`params()`と対応）。
    /// 固定効果（entity・timeダミー）の係数は含まない。
    pub fn cov_params(&self) -> &Mat<f64> {
        &self.cov_params
    }

    /// `cov_type`別に計算し直したt統計量（`(k, 1)`）。
    pub fn test_stats(&self) -> &Mat<f64> {
        &self.test_stats
    }

    /// `test_stats`の従う分布（t分布、自由度は`df_inference`）。
    pub fn stat_dist(&self) -> inference::StatDist {
        inference::StatDist::T {
            df: self.df_inference,
        }
    }

    /// `cov_type`別に計算し直した両側p値（`(k, 1)`）。
    pub fn p_values(&self) -> &Mat<f64> {
        &self.p_values
    }

    /// `cov_type`別に計算し直した信頼区間の下限（`(k, 1)`）。
    pub fn conf_lower(&self) -> &Mat<f64> {
        &self.conf_lower
    }

    /// `cov_type`別に計算し直した信頼区間の上限（`(k, 1)`）。
    pub fn conf_upper(&self) -> &Mat<f64> {
        &self.conf_upper
    }

    /// 実際に使ったFE構造でdemeanしたR²（モジュールdoc「パネル固有R²」参照）。
    pub fn r_squared_within(&self) -> f64 {
        self.r_squared_within
    }

    /// エンティティ平均ベースのR²（linearmodelsの`rsquared_between`と完全一致、
    /// モジュールdoc参照）。
    pub fn r_squared_between(&self) -> f64 {
        self.r_squared_between
    }

    /// 固定効果の切片項を含めないR²（linearmodelsの`rsquared_overall`と完全一致、
    /// モジュールdoc参照）。
    pub fn r_squared_overall(&self) -> f64 {
        self.r_squared_overall
    }

    /// パネル自由度調整済みAIC（`df_model`をペナルティ項に使う、モジュールdoc参照）。
    pub fn aic(&self) -> f64 {
        self.aic
    }

    /// パネル自由度調整済みBIC。
    pub fn bic(&self) -> f64 {
        self.bic
    }

    /// 傾き係数`k`個が同時にゼロという帰無仮説のWald F検定統計量
    /// （モジュールdoc「自由度とF統計量」参照）。`k=0`なら`None`。
    pub fn f_statistic(&self) -> Option<f64> {
        self.f_test.map(|(statistic, _)| statistic)
    }

    /// `f_statistic()`のp値。`f_statistic()`が`None`のとき`None`。
    pub fn f_p_value(&self) -> Option<f64> {
        self.f_test.map(|(_, p_value)| p_value)
    }

    /// `f_statistic()`の自由度`(分子, 分母)` = `(k, df_inference)`。`k=0`で`None`のときは`None`。
    pub fn f_df(&self) -> Option<(usize, usize)> {
        self.f_test
            .map(|_| (self.within_input.k(), self.df_inference))
    }

    /// 固定効果自体（α_i、2-wayはγ_tも）を事後的に復元する（`fe-spec.md`3.5節）。
    ///
    /// `fit()`の戻り値本体には含めない別メソッド（IVの`first_stage()`と同じ方針、
    /// モジュールdoc「固定効果自体（α_i）の復元」参照）。2-wayは正規化に任意性があるため
    /// `time`の時間順で最初の時点を基準に`γ_{t_ref}=0`とする規約を採用している（同モジュール
    /// doc参照。`fixest::fixef()`とは基準時点の選び方の前提が異なるため、数値一致は
    /// 観測順の最初の時点と時間順で最初の時点が一致する入力に限られる）。2-wayの`time`は
    /// 時間順の`Vec`で返す。
    pub fn fixed_effects(&self) -> FixedEffects {
        let y = self.input.y();
        let x = self.input.x();
        let params = &self.params;

        match self.effects {
            FeEffects::OneWay => FixedEffects::OneWay(
                group_residual_means(y, x, params, self.input.entity_codes())
                    .into_iter()
                    .collect(),
            ),
            FeEffects::TwoWay => {
                let time = self.input.time_codes().expect(
                    "2-way already validated `time` is present (`fit` resolves `two_way_time`)",
                );
                let entity_means = group_residual_means(y, x, params, self.input.entity_codes());
                let time_means = group_residual_means(y, x, params, time);
                let overall_mean = overall_residual_mean(y, x, params);
                // `time_means`は時点の昇順（コード順）のため`first()`が最初の時点（DKの
                // 時系列順序と同じ、モジュールdoc参照）。2-way FEは`n>=1`が
                // `InsufficientDegreesOfFreedom`検証で既に保証されているため、`time_means`は
                // 必ず1件以上の要素を持つ。
                let &(_, reference_value) = time_means
                    .first()
                    .expect("2-way FE guarantees at least one time period (df_resid check)");

                let entity = entity_means
                    .into_iter()
                    .map(|(id, mean)| (id, mean - overall_mean + reference_value))
                    .collect();
                let time = time_means
                    .into_iter()
                    .map(|(id, mean)| (id, mean - reference_value))
                    .collect();
                FixedEffects::TwoWay { entity, time }
            }
        }
    }
}

/// 固定効果の切片項を一切含めない残差`y_i - x_i'β̂`の1行分。
/// `fe_r_squared_overall`・`group_residual_means`/`overall_residual_mean`
/// （`fixed_effects`）で共有する「元の`y`/`x`に`β̂`だけを当てはめた残差」の定義
/// （モジュールdoc「パネル固有R²」「固定効果自体（α_i）の復元」参照。`fe_r_squared_between`
/// はエンティティ平均`ȳ_i.`/`x̄_i.`に集約してから当てはめるため、この関数とは行の単位が
/// 異なり共有しない）。
fn slope_only_residual(y: &[f64], x: &[Vec<f64>], params: &Mat<f64>, i: usize) -> f64 {
    let k = x.len();
    let fitted: f64 = (0..k).map(|j| x[j][i] * (*params.get(j, 0))).sum();
    y[i] - fitted
}

/// エンティティ平均ベースのbetween R²（2.3節）。`linearmodels`の
/// `PanelOLS._rsquared`のbetween式と完全一致させる（モジュールdoc「パネル固有R²」参照）。
///
/// `y`/`x`は**within変換前の元の列**（`FeInput::y`/`x`）を渡すこと。`params`は
/// within推定の`β̂`（切片を含まない、FEは常に`include_intercept=false`）。
///
/// **エンティティ観測数`T_i`による重み付けは行わない**（`w_i=1`で全エンティティ均等）。
/// `linearmodels`の`_prepare_between`自体は`w_i = T_i / mean(T)`を計算するが、`_rsquared`
/// 側で`self.weights.values2d`（サンプルウェイト、`PanelOLS`の`weights`引数）が全て`1.0`
/// （＝ユーザーが明示的な重みを指定していない）なら`w`を無条件に全要素`1.0`へ上書きする
/// （`if np.all(self.weights.values2d == 1.0): w = np.ones_like(w)`）。**FEは`weights`引数を
/// サポートしない**（CLAUDE.md 1.3節「見送り」）ため、この分岐が常に成立し
/// `T_i`ベースの重みは実質的に到達不能（重み付き版は不均衡パネルで`linearmodels`と
/// 数値が食い違う）。エンティティコード（`GroupCodes`、辞書順）で集計する（`panel_cluster_cov_params`
/// と同じ理由でグループ間加算の順序を固定する、モジュールdoc参照）。
///
/// `TSS <= 0`（全エンティティ平均がゼロ等）なら`linearmodels`と同じく`0.0`を返す。
fn fe_r_squared_between(y: &[f64], x: &[Vec<f64>], params: &Mat<f64>, entity: &GroupCodes) -> f64 {
    let k = x.len();
    let entity_indices = entity.group_indices();

    let mut ssr = 0.0;
    let mut tss = 0.0;
    for indices in entity_indices.iter() {
        let t_i = indices.len();
        let y_bar: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / t_i as f64;
        let fitted: f64 = (0..k)
            .map(|j| {
                let x_bar_j: f64 = indices.iter().map(|&i| x[j][i]).sum::<f64>() / t_i as f64;
                x_bar_j * (*params.get(j, 0))
            })
            .sum();
        let resid = y_bar - fitted;
        ssr += resid * resid;
        tss += y_bar * y_bar;
    }

    if tss > 0.0 { 1.0 - ssr / tss } else { 0.0 }
}

/// 固定効果の切片項を含めないoverall R²（2.3節）。`linearmodels`の
/// `PanelOLS._rsquared`のoverall式と完全一致させる（モジュールdoc「パネル固有R²」参照）。
///
/// `y`/`x`は**within変換前の元の列**を渡すこと。within推定の残差
/// （`residuals()`、FWL定理により固定効果込みの残差と一致）とは異なり、
/// ここでは`β̂`だけを元の`y`/`x`に当てはめた残差（固定効果の切片項を含めない）を使う。
///
/// `TSS <= 0`なら`linearmodels`と同じく`0.0`を返す。
fn fe_r_squared_overall(y: &[f64], x: &[Vec<f64>], params: &Mat<f64>) -> f64 {
    let n = y.len();

    let mut ssr = 0.0;
    let mut tss = 0.0;
    for i in 0..n {
        let resid = slope_only_residual(y, x, params, i);
        ssr += resid * resid;
        tss += y[i] * y[i];
    }

    if tss > 0.0 { 1.0 - ssr / tss } else { 0.0 }
}

/// `ids`のコードでグループ化した`slope_only_residual`の平均（`E_i`/`E_t`）を、コード順の
/// `(キー, 平均)`で返す。entityはコードが辞書順、timeは時点の昇順で、どちらも順序が
/// 決定的（他のグループ集約と同じ理由）。グループ内の加算順は観測順になる。
/// `fixed_effects`が1-way・2-wayのentity/time双方で使う。
fn group_residual_means(
    y: &[f64],
    x: &[Vec<f64>],
    params: &Mat<f64>,
    ids: &GroupCodes,
) -> Vec<(String, f64)> {
    ids.keys()
        .iter()
        .zip(ids.group_indices().iter())
        .map(|(key, indices)| {
            let mean = indices
                .iter()
                .map(|&i| slope_only_residual(y, x, params, i))
                .sum::<f64>()
                / indices.len() as f64;
            (key.clone(), mean)
        })
        .collect()
}

/// 全観測にわたる`slope_only_residual`の平均（`E`、2-way正規化で使う大域平均）。
fn overall_residual_mean(y: &[f64], x: &[Vec<f64>], params: &Mat<f64>) -> f64 {
    let n = y.len();
    (0..n)
        .map(|i| slope_only_residual(y, x, params, i))
        .sum::<f64>()
        / n as f64
}

/// 1-way FE（entityのみ）のwithin変換。`y`と各`x`列にentityでのquasi-demean（θ=1）を
/// 適用する。不均衡パネルも無条件でサポートする（モジュールdoc・`fe-spec.md`3.1節参照）。
///
/// 戻り値は`(y_transformed, x_transformed)`（元の列順を保持）。
pub(crate) fn within_transform_one_way(input: &FeInput) -> (Vec<f64>, Vec<Vec<f64>>) {
    let within = WithinTransform::one_way(input.entity_codes());
    let y = within.apply(input.y());
    let x = input.x().iter().map(|col| within.apply(col)).collect();
    (y, x)
}

/// 2-way FE（entity + time FE）のwithin変換。閉形式の二重デミーニングと数学的に等価な
/// 「entityでquasi-demean → その結果をtimeでquasi-demean」の2段階適用で計算する
/// （モジュールdoc参照）。事前にバランスパネルであることを検証する（`fe-spec.md`3.1節）。
///
/// 戻り値は`(y_transformed, x_transformed)`（元の列順を保持）。
///
/// `FeEstimator::fit`経由では`two_way_time`が`time`の無い入力を先に弾くため、ここの
/// `time`の存在チェックは単体利用時の防御（到達しない二重チェックだが意図的に残している）。
///
/// # Errors
/// - `input.time_keys()`が`None`の場合は`PanelError::TwoWayRequiresTime`
/// - バランスパネルでない場合は`PanelError::UnbalancedPanelForTwoWay`
pub(crate) fn within_transform_two_way(
    input: &FeInput,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), PanelError> {
    let time_codes = input.time_codes().ok_or(PanelError::TwoWayRequiresTime)?;
    let entity = input.entity_codes();
    validate_balanced_panel(entity, time_codes, input.effect_names())?;

    let within = WithinTransform::two_way(entity, time_codes);
    let y = within.apply(input.y());
    let x = input.x().iter().map(|col| within.apply(col)).collect();
    Ok((y, x))
}

/// 1-way FE向けのsingleton検出（`fe-spec.md`1章）。`entity`に観測数1のグループがあれば
/// `PanelError::SingletonGroup`を返す。
///
/// # Errors
/// entityに観測数1のグループが見つかった場合は`PanelError::SingletonGroup`
/// （`dimension: PanelDimension::Entity`）。
pub(crate) fn validate_no_singleton_groups_one_way(input: &FeInput) -> Result<(), PanelError> {
    ensure_no_singleton_groups(input.effect_names(), input.entity_codes(), None)
}

/// 2-way FE向けのsingleton検出（`fe-spec.md`1章）。entity・time双方を対称に検出する
/// （`within_transform_two_way`と同じく`time`必須）。
///
/// **`time`の存在チェックを最初に行う**（`within_transform_two_way`と同じ順序に揃える。
/// `fit()`側が複数のバリデーションを組み合わせて呼ぶ際、同じ前提条件——2-way FEには
/// `time`が要る——のチェックタイミングが関数ごとにばらつかないようにするため）。
/// その後、entityを先にチェックしてからtimeをチェックする（両方に該当するsingletonが
/// あった場合はentity側を先に報告する。`FeInput`のフィールド順（entity→time）に合わせた
/// 恣意的な優先順位）。
///
/// `FeEstimator::fit`経由では`two_way_time`が`time`の無い入力を先に弾くため、ここの
/// `time`の存在チェックは単体利用時の防御（到達しない二重チェックだが意図的に残している）。
///
/// # Errors
/// - `input.time_keys()`が`None`の場合は`PanelError::TwoWayRequiresTime`
/// - entityまたはtimeに観測数1のグループが見つかった場合は`PanelError::SingletonGroup`
///   （該当する`dimension`を含む）
pub(crate) fn validate_no_singleton_groups_two_way(input: &FeInput) -> Result<(), PanelError> {
    let time = input.time_codes().ok_or(PanelError::TwoWayRequiresTime)?;
    ensure_no_singleton_groups(input.effect_names(), input.entity_codes(), Some(time))
}

/// within変換後の説明変数の各列に分散ゼロの列がないことを検証する（`fe-spec.md`1章）。1-way/2-way
/// 共通ロジック（`within_transform_one_way`/`within_transform_two_way`のどちらの出力も
/// 引数に渡せる）。
///
/// 時間不変変数（1-way）だけでなく、2-wayでtime FEと完全共線な「エンティティ間で変動しない
/// 列」も同じチェックで検出できる（`fe-spec.md`1章）。`x_transformed`は呼び出し側が`within_transform_*`
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
pub(crate) fn validate_no_zero_variance_regressors(
    input: &FeInput,
    x_transformed: &[Vec<f64>],
) -> Result<(), PanelError> {
    ensure_no_zero_variance_columns(input.x_names(), input.x(), x_transformed)
}

/// 2-wayの時点コードを返す（1-wayは`None`）。2-wayなのに`input`に`time`が無ければ
/// `PanelError::TwoWayRequiresTime`。`FeEstimator::fit`が最初に呼び、以降の
/// cov_type別の計算は`effects`ではなくこの`Option`で1-way/2-wayを区別する。
fn two_way_time(effects: FeEffects, input: &FeInput) -> Result<Option<&GroupCodes>, PanelError> {
    match effects {
        FeEffects::OneWay => Ok(None),
        FeEffects::TwoWay => input
            .time_codes()
            .map(Some)
            .ok_or(PanelError::TwoWayRequiresTime),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::common::LeastSquaresError;
    use crate::panel::common::{
        PanelDimension, reject_singleton_group, time_varying_regressor_indices,
    };
    use crate::shared::error::{ClusterSetting, MatrixKind};
    use crate::shared::error_catalog::Stage;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    /// 辞書順の`TimeKeys`（`FeCovType::Dk.time`に渡す）。
    fn lex_time(values: &[&str]) -> TimeKeys {
        TimeKeys::lexicographic(strings(values))
    }

    /// 時点効果（時点の昇順の`Vec`）を、ラベルで引ける`BTreeMap`にする。
    fn by_label(effects: Vec<(String, f64)>) -> BTreeMap<String, f64> {
        effects.into_iter().collect()
    }

    fn codes(values: &[&str]) -> GroupCodes {
        GroupCodes::from_labels(&strings(values))
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
        // ユニークなラベルを辞書順（コード順）に並べたもの。
        assert_eq!(input.entity_labels(), ["a", "b"]);
        assert_eq!(input.time_keys(), None);
        assert_eq!(input.dep_var_name(), "y");
        assert_eq!(input.nobs(), 4);
        assert_eq!(input.n_entities(), 2);
    }

    #[test]
    fn from_group_codes_matches_from_columns_and_requires_entity_keys() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let x = vec![vec![1.0, 2.0, 3.0, 5.0]];
        let entity = strings(&["a", "a", "b", "b"]);

        let from_strings = FeInput::from_columns(
            &y,
            &x,
            vec!["x1".to_string()],
            &entity,
            None,
            "y".to_string(),
        )
        .unwrap();
        let from_codes = FeInput::from_group_codes(
            &y,
            &x,
            vec!["x1".to_string()],
            GroupCodes::from_factorized(vec![1, 1, 0, 0], &["b", "a"]).unwrap(),
            None,
            "y".to_string(),
        )
        .unwrap();
        assert_eq!(from_codes.entity_labels(), from_strings.entity_labels());
        assert_eq!(from_codes.n_entities(), 2);

        // キー無しのコードは固定効果のキーが作れないため受け付けない。
        let keyless = GroupCodes::from_labels_without_keys(&entity);
        assert!(matches!(
            FeInput::from_group_codes(
                &y,
                &x,
                vec!["x1".to_string()],
                keyless,
                None,
                "y".to_string()
            ),
            Err(PanelError::Common(CommonError::Internal { .. }))
        ));
        // 行数の食い違いは従来どおり`IdentifierDimensionMismatch`。
        assert!(matches!(
            FeInput::from_group_codes(
                &y,
                &x,
                vec!["x1".to_string()],
                GroupCodes::from_labels(&strings(&["a", "b"])),
                None,
                "y".to_string()
            ),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Entity,
                ..
            })
        ));
    }

    /// エラーの優先順位: 次元の食い違い（`y`↔各`x`列、`entity`の行数、`time`の行数）が先で、
    /// `entity`がキー無しであることの拒否は最後。
    #[test]
    fn from_group_codes_reports_dimension_mismatches_before_missing_entity_keys() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let x = vec![vec![1.0, 2.0, 3.0, 5.0]];
        let keyless = |labels: &[&str]| GroupCodes::from_labels_without_keys(&strings(labels));

        // `time`の行数の食い違い（`entity`は正しい長さでキー付き）。
        assert!(matches!(
            FeInput::from_group_codes(
                &y,
                &x,
                vec!["x1".to_string()],
                GroupCodes::from_labels(&strings(&["a", "a", "b", "b"])),
                Some(TimeKeys::lexicographic(strings(&["1", "2"]))),
                "y".to_string()
            ),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                ..
            })
        ));
        // キー無しで、かつ`entity`の行数も食い違う: 行数のエラーが先。
        assert!(matches!(
            FeInput::from_group_codes(
                &y,
                &x,
                vec!["x1".to_string()],
                keyless(&["a", "b"]),
                None,
                "y".to_string()
            ),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Entity,
                ..
            })
        ));
        // `x`の列の行数の食い違いは、`entity`の検証より先。
        assert!(matches!(
            FeInput::from_group_codes(
                &y,
                &[vec![1.0, 2.0]],
                vec!["x1".to_string()],
                keyless(&["a", "b"]),
                None,
                "y".to_string()
            ),
            Err(PanelError::Common(CommonError::DimensionMismatch { .. }))
        ));
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

        // 時点は`lexicographic`（ラベルの辞書順）。行数と、ユニークなラベルの時間順を確認する。
        let Some(time_keys) = input.time_keys() else {
            panic!("time keys must be set");
        };
        assert_eq!(time_keys.len(), 4);
        assert_eq!(time_keys.periods(), ["2020", "2021"]);
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
        assert_eq!(input.time_keys().map(TimeKeys::len), Some(0));
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
                entity_column: "entity".to_string(),
                n_entities: 2,
                time_column: "time".to_string(),
                n_periods: 2,
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
                entity_column: "entity".to_string(),
                n_entities: 2,
                time_column: "time".to_string(),
                n_periods: 2,
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
                column: "entity".to_string(),
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_singleton_groups_reports_first_singleton_in_observation_order() {
        // singletonが"b"と"a"の2つ。観測順では"b"が先、コード順（辞書順）では"a"が先。
        // 観測順で最初のもの（"b"）を報告する（整数コード化の前と同じ選び方）。
        let y = [1.0, 2.0, 3.0, 4.0];
        let entity = strings(&["b", "a", "c", "c"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        assert_eq!(
            validate_no_singleton_groups_one_way(&input).unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                column: "entity".to_string(),
                group_id: "b".to_string(),
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
                column: "entity".to_string(),
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
                column: "time".to_string(),
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
                column: "entity".to_string(),
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
        assert_eq!(
            reject_singleton_group(
                PanelDimension::Entity,
                "entity",
                &GroupCodes::from_labels(&[])
            ),
            Ok(())
        );
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

    #[test]
    fn singleton_group_error_reports_the_column_names_set_by_with_effect_names() {
        // 既定の名前は"entity"・"time"。`engine_pybind`が渡す列名がエラーの`column`に載る。
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let entity = strings(&["a", "a", "b", "b", "c"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();
        assert_eq!(
            validate_no_singleton_groups_one_way(&input),
            Err(PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                column: "entity".to_string(),
                group_id: "c".to_string(),
            })
        );
        let input = input.with_effect_names("firm".to_string(), None);
        assert_eq!(
            validate_no_singleton_groups_one_way(&input),
            Err(PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                column: "firm".to_string(),
                group_id: "c".to_string(),
            })
        );

        // 時点側の単独グループ（時点"2"は1観測）は2つ目の名前で報告する。
        let y = [1.0, 2.0, 3.0, 4.0];
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["1", "2", "1", "1"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into())
            .unwrap()
            .with_effect_names("firm".to_string(), Some("year".to_string()));
        assert_eq!(
            validate_no_singleton_groups_two_way(&input),
            Err(PanelError::SingletonGroup {
                dimension: PanelDimension::Time,
                column: "year".to_string(),
                group_id: "2".to_string(),
            })
        );
    }

    // ── validate_no_zero_variance_regressors ────────────────────────────────

    #[test]
    fn validate_no_zero_variance_regressors_detects_time_invariant_variable_in_one_way_fe() {
        // "female"は各エンティティ内で一定（時間不変）のため、1-way within変換後は
        // 浮動小数点誤差の範囲でゼロになる（`fe-spec.md`1章のユースケースそのもの）。
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
    fn time_varying_regressor_indices_returns_only_columns_with_within_variation() {
        // 先頭・末尾が時間不変（エンティティ内で一定）、中央が時間変化。列順を保ち、
        // 時間変化のある列の添字だけを昇順で返す。
        let entity = strings(&["a", "a", "b", "b", "b"]);
        let input = FeInput::from_columns(
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[
                vec![7.0, 7.0, -2.0, -2.0, -2.0],
                vec![1.0, 3.0, 2.0, 5.0, 9.0],
                vec![0.5, 0.5, 0.5, 0.5, 0.5],
            ],
            strings(&["z1", "x", "z2"]),
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        assert_eq!(
            time_varying_regressor_indices(input.x(), input.entity_codes()),
            vec![1]
        );
    }

    #[test]
    fn time_varying_regressor_indices_agrees_with_zero_variance_validation() {
        // 返した添字の列だけを集めた入力は`validate_no_zero_variance_regressors`を通り、
        // 返さなかった列だけを集めた入力は同じ検証で拒否される（FEに渡せる列の選別が
        // FE自身の判定とずれない）。
        let entity = strings(&["a", "a", "b", "b", "b", "c", "c"]);
        let columns = vec![
            vec![2.0, 2.0, 5.0, 5.0, 5.0, -1.0, -1.0],
            vec![1.0, 4.0, 2.0, 2.5, 9.0, 3.0, 8.0],
            vec![1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6],
        ];
        let names = strings(&["z", "x", "big"]);
        let y = [1.0; 7];
        let full =
            FeInput::from_columns(&y, &columns, names.clone(), &entity, None, "y".into()).unwrap();
        let keep = time_varying_regressor_indices(full.x(), full.entity_codes());
        assert_eq!(keep, vec![1]);

        let kept_input = FeInput::from_columns(
            &y,
            &[columns[1].clone()],
            vec![names[1].clone()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();
        let (_, x_kept) = within_transform_one_way(&kept_input);
        assert_eq!(
            validate_no_zero_variance_regressors(&kept_input, &x_kept),
            Ok(())
        );

        let dropped_input = FeInput::from_columns(
            &y,
            &[columns[0].clone(), columns[2].clone()],
            vec![names[0].clone(), names[2].clone()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();
        let (_, x_dropped) = within_transform_one_way(&dropped_input);
        assert!(validate_no_zero_variance_regressors(&dropped_input, &x_dropped).is_err());
    }

    #[test]
    fn time_varying_regressor_indices_is_empty_for_no_regressors_and_all_for_varying_columns() {
        let entity = strings(&["a", "a", "b", "b"]);
        let none =
            FeInput::from_columns(&[1.0; 4], &[], vec![], &entity, None, "y".into()).unwrap();
        assert!(time_varying_regressor_indices(none.x(), none.entity_codes()).is_empty());

        let all = FeInput::from_columns(
            &[1.0; 4],
            &[vec![1.0, 2.0, 3.0, 5.0], vec![9.0, 1.0, 2.0, 8.0]],
            strings(&["x1", "x2"]),
            &entity,
            None,
            "y".into(),
        )
        .unwrap();
        assert_eq!(
            time_varying_regressor_indices(all.x(), all.entity_codes()),
            vec![0, 1]
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
        // within変換後はゼロ分散になる（`fe-spec.md`1章「time FEと完全共線な列も同じチェックで
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
        // entity a: x=[1,2,3], y=2x+5（fixed effect=5）+ノイズ
        // entity b: x=[4,5,6], y=2x+10（fixed effect=10）+ノイズ
        // 小さなノイズのもとで、within変換後のOLS（切片なし）は真のスロープ2.0の近くに返る
        // （fixed effectはwithin変換で消去される）。
        let entity = strings(&["a", "a", "a", "b", "b", "b"]);
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, -0.2, 0.1, 0.05, -0.1, 0.05];
        let y: Vec<f64> = x1
            .iter()
            .zip(&entity)
            .zip(noise)
            .map(|((x, e), u)| 2.0 * x + if e == "a" { 5.0 } else { 10.0 } + u)
            .collect();
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert!((*fe.params().get(0, 0) - 2.0).abs() < 0.2);
        assert_eq!(fe.effects(), FeEffects::OneWay);
        assert!(!fe.within_input().has_intercept());
        assert_eq!(fe.n_periods(), None);
        for r in fe.residuals().col(0).iter() {
            assert!(r.abs() < 0.3);
        }
    }

    #[test]
    fn fe_estimator_fit_returns_perfect_fit_for_exact_data() {
        // ノイズなしでは、within回帰の残差が全て0で標準誤差とF検定が定義できない。
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

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::WithinRegressionFailed {
                source: CommonError::PerfectFit.into(),
            }
        );
    }

    #[test]
    fn fe_estimator_fit_returns_outcome_constant_within_entity_for_an_entity_level_outcome() {
        // yが個体内で定数（個体間でだけ違う）なら、固定効果が全てを説明してwithin変換後のyは0になる。
        // 0.1のように3回足して3で割ると丸め誤差が残る値も、相対閾値で定数とみなす。
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c"]);
        let y = [0.1, 0.1, 0.1, 0.7, 0.7, 0.7, 1.3, 1.3, 1.3];
        let x1 = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 1.0, 4.0, 2.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

        assert_eq!(result.unwrap_err(), PanelError::OutcomeConstantWithinEntity);
    }

    #[test]
    fn fe_estimator_fit_returns_outcome_constant_within_entity_when_effects_explain_the_outcome() {
        // 2-wayでは、y = 個体効果 + 時点効果 もwithin変換後に0になる。
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c"]);
        let time = strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3"]);
        let alpha = [1.0, 1.0, 1.0, 4.0, 4.0, 4.0, -2.0, -2.0, -2.0];
        let gamma = [0.0, 0.5, 1.5, 0.0, 0.5, 1.5, 0.0, 0.5, 1.5];
        let y: Vec<f64> = alpha.iter().zip(gamma).map(|(a, g)| a + g).collect();
        let x1 = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 1.0, 4.0, 2.0];
        let input = FeInput::from_columns(
            &y,
            &[x1],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95);

        assert_eq!(result.unwrap_err(), PanelError::OutcomeConstantWithinEntity);
    }

    #[test]
    fn fe_estimator_fit_exposes_input_and_cov_type_via_getters() {
        // `input()`/`cov_type()`（カバレッジ監査で判明した未検証の単純
        // getter）。`ols::fit_exposes_input_cov_type_and_residuals_via_getters`と同型。
        let entity = strings(&["a", "a", "b", "b"]);
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let y = [7.0, 9.2, 11.0, 12.6];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster { groups: None },
            0.95,
        )
        .unwrap();

        assert_eq!(fe.input().nobs(), 4);
        assert_eq!(fe.input().dep_var_name(), "y");
        assert_eq!(fe.cov_type(), &FeCovType::Cluster { groups: None });
        // `df_inference()`/`stat_dist()`（カバレッジ監査で判明した未検証の単純
        // getter、新設）。`n_entities=2`が既定クラスターのため
        // `df_inference = G-1 = 1`（`df_resid`の`n-df_model=4-3=1`とはこの
        // データではたまたま同じ値になるが、由来は別——後続の
        // `..._one_way_cluster_on_entity_matches_fixest_nested_k`等で
        // `df_resid`と乖離するケースを別途確認済み）。
        assert_eq!(fe.df_inference(), 1);
        assert_eq!(fe.stat_dist(), inference::StatDist::T { df: 1 });
    }

    #[test]
    fn fe_estimator_fit_two_way_recovers_known_slope() {
        // `within_transform_two_way_matches_closed_form_double_demeaning`と同じ関係
        // （x1はyのちょうど2倍。この関係は線形変換の下で任意のN・Tで恒等的に保たれるため
        // 具体的な値は問わない）だが、3エンティティ×3時点（n=9）のバランスパネルに
        // 拡張する：df_model=k(1)+neffects(n_entities+n_periods-1=3+3-1=5)=6、n=9>6で
        // 自由度検証（`n<=df_model`）を通過できる規模にする必要があるため
        // （N=2,T=2のn=4だとdf_model=1+3=4となりn<=df_modelで弾かれてしまう）。
        // 2-way within変換後、x1_out ≈ 2 * y_out がほぼ成り立つため、切片なしOLSの
        // スロープは0.5にほぼ一致するはず。x1はy*2からごくわずかに擾乱を入れる
        // （厳密にx1=2*yだと残差が全行ゼロになり、F検定
        // （`wald_f_test`）が分散ゼロによる特異行列で`NearSingularMatrix`を返してしまう
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

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();

        // 許容誤差0.01は`testing-policy.md`の相対誤差1e-8方針の対象外（リファレンス実装
        // との数値比較ではなく、本テスト自身が注入した擾乱ノイズに対する内部整合性の
        // 確認のため）。ノイズの大きさ（最大0.02、xスケール比で見ると相対誤差1〜2%程度）
        // に対してスロープ推定への影響がこの範囲に収まることを確認する目的の閾値。
        assert!((*fe.params().get(0, 0) - 0.5).abs() < 0.01);
        assert_eq!(fe.effects(), FeEffects::TwoWay);
        assert_eq!(fe.df_model(), 6);
        assert_eq!(fe.n_periods(), Some(3));
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

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                column: "entity".to_string(),
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn fe_estimator_fit_propagates_zero_variance_error() {
        // "female"は各エンティティ内で一定（時間不変）。n=6・n_entities=2・k=2で
        // df_model=4・n>df_modelとなるよう、各エンティティ3観測に拡張する
        // （n=4だとdf_model=2+2=4でn<=df_modelとなりInsufficientDegreesOfFreedomが
        // 先に発火してしまうため、自由度検証を踏まえたサイズにする）。
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

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

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

        let result = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95);

        assert_eq!(result.unwrap_err(), PanelError::TwoWayRequiresTime);
    }

    #[test]
    fn fe_estimator_fit_two_way_without_time_reports_requires_time_before_any_other_error() {
        // 2-wayで`time`が無いことは、他のどの欠陥（entityのsingleton・不正な
        // `confidence_level`・cov_type固有の検証）よりも先に報告される。優先順位は
        // `fit()`冒頭の`two_way_time`が守る。
        let entity = strings(&["a", "a", "c"]);
        let y = [1.0, 2.0, 3.0];
        let covs = || {
            vec![
                FeCovType::Classical,
                FeCovType::Hc2,
                FeCovType::Cluster { groups: None },
                FeCovType::Dk {
                    bandwidth: Some(100),
                    time: lex_time(&["1", "2", "3"]),
                },
            ]
        };
        for cov_type in covs() {
            let input = FeInput::from_columns(
                &y,
                &[vec![1.0, 2.0, 4.0]],
                vec!["x1".to_string()],
                &entity,
                None,
                "y".into(),
            )
            .unwrap();

            let result = FeEstimator::fit(input, FeEffects::TwoWay, cov_type.clone(), 1.5);

            assert_eq!(
                result.unwrap_err(),
                PanelError::TwoWayRequiresTime,
                "cov_type = {cov_type:?}"
            );
        }
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

        let result = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                column: "entity".to_string(),
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

        let result = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::UnbalancedPanelForTwoWay {
                n_obs: 4,
                entity_column: "entity".to_string(),
                n_entities: 2,
                time_column: "time".to_string(),
                n_periods: 2,
            }
        );
    }

    #[test]
    fn fe_estimator_fit_two_way_propagates_zero_variance_error() {
        // "year_dummy"はエンティティ間で変動しない（time FEと完全共線）。`fit()`が
        // within変換後に`validate_no_zero_variance_regressors`を呼んでいることの配線確認。
        // n=9（3エンティティ×3時点、n_entities=3・n_periods=3）に拡張し、
        // df_model=k(2)+neffects(3+3-1=5)=7・n>df_modelとなるサイズにする
        // （自由度検証を先に通過させるため）。"x_varying"は
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

        let result = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::ZeroVarianceAfterDemeaning {
                column: "year_dummy".to_string(),
            }
        );
    }

    #[test]
    fn fe_estimator_fit_propagates_invalid_confidence_level_error() {
        // `confidence_level`の範囲チェック
        // （`(0, 1)`の範囲外）が、`PanelError::WithinRegressionFailed`として正しく
        // 伝播することを確認する（`iv::two_sls`の同型テストに倣う）。
        let entity = strings(&["a", "a", "b", "b"]);
        let y = [1.0, 2.0, 3.0, 4.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 1.5);

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
        // 自由度検証（`n <= df_model`）が最小二乗側の`n <= k`相当の条件より常に厳しい
        // （`df_model = k + neffects > k`）ため、最小二乗側の観測数不足はこの経路では
        // 発生しえない（`fe_estimator_fit_propagates_insufficient_degrees_of_freedom_error`が
        // 先に弾く）。そのため、ここでは最小二乗固有の別の失敗——完全な多重共線性（`LeastSquaresError::
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

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

        assert!(matches!(
            result.unwrap_err(),
            PanelError::WithinRegressionFailed { .. }
        ));
    }

    #[test]
    fn fe_estimator_fit_propagates_insufficient_degrees_of_freedom_error() {
        // n=4・n_entities=2・k=2でdf_model=k+n_entities=4となり、n<=df_modelのため
        // 最小二乗へ進む前に`PanelError::InsufficientDegreesOfFreedom`で
        // 弾かれることを確認する。
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

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

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

    /// エラーの`min_obs`が`fit`の実際の境界と一致する: 拒否される最大の`n`（`n = df_model`）の
    /// 次の`n = min_obs`で成功する。1-wayの`n_effects`は個体数（ここでは2）。
    #[test]
    fn insufficient_degrees_of_freedom_min_obs_matches_the_fit_boundary() {
        use crate::shared::error::ErrorInfo;
        use crate::shared::error_catalog::ParamValue;

        let build = |n: usize, k: usize| {
            let entity: Vec<String> = (0..n)
                .map(|i| if i < n.div_ceil(2) { "a" } else { "b" }.to_string())
                .collect();
            let y: Vec<f64> = (0..n)
                .map(|i| (i as f64 * 0.9).sin() + 0.1 * i as f64)
                .collect();
            let x: Vec<Vec<f64>> = (0..k)
                .map(|j| {
                    (0..n)
                        .map(|i| ((i * (j + 2) + j) as f64 * 1.3).cos())
                        .collect()
                })
                .collect();
            let names = (0..k).map(|j| format!("x{j}")).collect();
            FeInput::from_columns(&y, &x, names, &entity, None, "y".into()).unwrap()
        };
        for k in 2..=4_usize {
            let n_effects = 2;
            let rejected_n = n_effects + k;
            let err = FeEstimator::fit(
                build(rejected_n, k),
                FeEffects::OneWay,
                FeCovType::Classical,
                0.95,
            )
            .unwrap_err();
            let min_obs = err
                .report()
                .all_params()
                .into_iter()
                .find_map(|(name, value)| match (name.as_str(), value) {
                    ("min_obs", ParamValue::Int(v)) => Some(v as usize),
                    _ => None,
                })
                .expect("min_obs");
            assert_eq!(min_obs, rejected_n + 1, "k={k}");
            assert!(
                FeEstimator::fit(
                    build(min_obs, k),
                    FeEffects::OneWay,
                    FeCovType::Classical,
                    0.95
                )
                .is_ok(),
                "k={k}"
            );
        }
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
        // （`options(digits=15)`でフルの浮動小数点精度を取得）。
        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert_eq!(fe.df_model(), 5); // k(1) + n_entities(4)
        assert_eq!(fe.df_resid(), 7); // n(12) - df_model(5)
        assert!((*fe.params().get(0, 0) - 1.402_777_777_777_78).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.432_598_838_244_034).abs() < 1e-6);
        assert!((*fe.test_stats().get(0, 0) - 3.242_675_785_889_31).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.014_200_386_789_949_8).abs() < 1e-6);
        assert!((*fe.conf_lower().get(0, 0) - 0.379_844_073_655_07).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 2.425_711_481_900_49).abs() < 1e-6);
        assert!((fe.aic() - 55.612_545_928_611_7).abs() < 1e-6);
        assert!((fe.bic() - 58.037_079_177_551_7).abs() < 1e-6);
        // パネル固有R²: 1-wayではwithinは`linearmodels`の`rsquared_within`と
        // `fixest`の`fitstat(m, "wr2")`が一致する（モジュールdoc「パネル固有R²」参照）。
        // between/overallは`linearmodels`の値（Pythonで独立に計算・検算済み）。
        assert!((fe.r_squared_within() - 0.600_341_337_099_812).abs() < 1e-9);
        assert!((fe.r_squared_between() - 0.748_302_743_867_978).abs() < 1e-9);
        assert!((fe.r_squared_overall() - 0.732_444_936_421_435).abs() < 1e-9);
        // F統計量: k=1のため「1自由度のF検定は両側t検定と代数的に等価」
        // （モジュールdoc「自由度とF統計量」参照）。
        assert!((fe.f_statistic().unwrap() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value().unwrap() - *fe.p_values().get(0, 0)).abs() < 1e-9);
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

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();

        assert_eq!(fe.df_model(), 7); // k(1) + neffects(n_entities(4)+n_periods(3)-1=6)
        assert_eq!(fe.df_resid(), 5); // n(12) - df_model(7)
        assert!((*fe.params().get(0, 0) - 0.822_429_906_542_056).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.227_239_295_931_651).abs() < 1e-6);
        assert!((*fe.test_stats().get(0, 0) - 3.619_223_969_033_19).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.015_231_948_369_008_1).abs() < 1e-6);
        assert!((*fe.conf_lower().get(0, 0) - 0.238_292_700_077_37).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.406_567_113_006_74).abs() < 1e-6);
        assert!((fe.aic() - 36.559_692_739_993_7).abs() < 1e-6);
        assert!((fe.bic() - 39.954_039_288_509_7).abs() < 1e-6);
        // パネル固有R²: 2-wayのwithinは`linearmodels`の`rsquared_within`
        // （常にentityのみdemean）とは意図的に食い違うため、`fixest`の
        // `fitstat(m, "wr2")`（0.723738317757009、`options(digits=15)`でR実地検証済み）を
        // 参照値にする（モジュールdoc「パネル固有R²」参照）。between/overallは
        // `linearmodels`の値。
        assert!((fe.r_squared_within() - 0.723_738_317_757_009).abs() < 1e-9);
        assert!((fe.r_squared_between() - 0.513_009_039_069_012).abs() < 1e-9);
        assert!((fe.r_squared_overall() - 0.511_356_250_429_877).abs() < 1e-9);
        // F統計量: k=1のため「1自由度のF検定は両側t検定と代数的に等価」
        // （モジュールdoc「自由度とF統計量」参照）。
        assert!((fe.f_statistic().unwrap() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value().unwrap() - *fe.p_values().get(0, 0)).abs() < 1e-9);
    }

    // ── F統計量 ───────────────────────────────────────────────────────────

    /// N=4（id: a,b,c,d）×T=3（t: 1,2,3）のバランスパネル、k=2（`fixest_reference_input`の
    /// 単回帰では真の同時検定（複数の傾き係数）を検証できないため、2変数版として別に用意
    /// する）。`fitstat(m, "f")`はFEダミーも含めたモデル全体のF検定でありここでの定義
    /// （FEダミーを除く傾き係数のみの同時検定、`linearmodels.PanelOLS.f_statistic`と同じ、
    /// モジュールdoc「自由度とF統計量」参照）と異なるため、fixestではなく
    /// `linearmodels`（`cov_type="unadjusted"`）の値と直接比較する（期待値はPythonで
    /// 独立に計算・検算済み）。
    #[allow(clippy::type_complexity)]
    fn f_test_reference_input() -> (Vec<String>, Vec<String>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]);
        let time = strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2", "3"]);
        let x1 = vec![1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 1.0, 3.0, 6.0, 4.0, 2.0, 1.0];
        let x2 = vec![2.0, 1.0, 4.0, 3.0, 2.0, 6.0, 5.0, 3.0, 1.0, 4.0, 2.0, 5.0];
        let y = vec![
            5.0, 7.0, 10.0, 3.0, 8.0, 9.0, 6.0, 10.0, 15.0, 2.0, 5.0, 4.0,
        ];
        (entity, time, x1, x2, y)
    }

    #[test]
    fn fe_estimator_fit_one_way_f_statistic_matches_linearmodels() {
        let (entity, _time, x1, x2, y) = f_test_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert_eq!(fe.df_resid(), 6); // n(12) - df_model(k(2)+n_entities(4))
        // `PanelOLS(y, [x1, x2], entity_effects=True).fit(cov_type="unadjusted",
        // debiased=True).f_statistic`: F(2,6)。
        assert!((fe.f_statistic().unwrap() - 4.544_331_119_544_591).abs() < 1e-9);
        assert!((fe.f_p_value().unwrap() - 0.062_878_408_429_192_23).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_f_statistic_matches_linearmodels() {
        let (entity, time, x1, x2, y) = f_test_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();

        assert_eq!(fe.df_resid(), 4); // n(12) - df_model(k(2)+neffects(4+3-1=6))
        // `PanelOLS(y, [x1, x2], entity_effects=True, time_effects=True).fit(
        // cov_type="unadjusted", debiased=True).f_statistic`: F(2,4)。
        assert!((fe.f_statistic().unwrap() - 9.574_202_321_891_761).abs() < 1e-9);
        assert!((fe.f_p_value().unwrap() - 0.029_859_178_280_428_5).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_returns_f_test_failed_for_extreme_scale_difference() {
        // x1は1e6オーダー、x2は1e-3オーダーとスケールが極端に異なる（x3は通常スケール）。
        // within変換後も3列は線形従属ではないため設計行列自体はフルランク
        // （`WithinRegressionFailed`にはならない）だが、傾き係数の同時共分散部分行列の条件数が
        // スケール比の2乗（≈1e18）相当となり倍精度の限界を超えるため、FE自身のF検定
        // （`wald_f_test`の`ensure_well_conditioned_symmetric_matrix`）が`FTestFailed`を返す。
        let n = 20;
        let entity: Vec<String> = (0..n).map(|i| format!("e{}", i % 5)).collect();
        let x1: Vec<f64> = (0..n).map(|i| 1e6 * (((i * 7) % 11) as f64)).collect();
        let x2: Vec<f64> = (0..n)
            .map(|i| 1e-3 * (((i * 5) % 13) as f64).powi(2))
            .collect();
        let x3: Vec<f64> = (0..n).map(|i| ((i * 3) % 7) as f64).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let noise = if i % 2 == 0 { 1e3 } else { -1e3 };
                2.0 * x1[i] + 3.0 * x2[i] + 0.5 * x3[i] + noise
            })
            .collect();
        let input = FeInput::from_columns(
            &y,
            &[x1, x2, x3],
            vec!["x1".to_string(), "x2".to_string(), "x3".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

        assert!(matches!(
            result.unwrap_err(),
            PanelError::FTestFailed {
                source: LeastSquaresError::Common(CommonError::NearSingularMatrix {
                    matrix: MatrixKind::CoefficientCovariance
                })
            }
        ));
    }

    // ── パネル固有R² ─────────────────────────────────────────────────────

    #[test]
    fn fe_estimator_fit_one_way_r_squared_between_matches_linearmodels_on_unbalanced_panel() {
        // `fe_r_squared_between`はエンティティ観測数による重み付け（`w_i = T_i/mean(T)`）を
        // しない（モジュールdoc「パネル固有R²」参照）。上の2本の`fixest_reference_input`
        // テストはバランスパネル（全エンティティ`T_i=3`）のため、重みの有無で値が変わらず
        // 検証できない。T_a=2, T_b=4, T_c=3の不均衡パネルで、重みなしの値が`linearmodels`と
        // 一致することを数値比較する（期待値はPythonで独立に計算・検算済み）。
        let entity = strings(&["a", "a", "b", "b", "b", "b", "c", "c", "c"]);
        let x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 1.0, 4.0, 2.0];
        let y = [2.0, 6.0, 5.0, 9.0, 8.0, 11.0, 3.0, 7.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert!((*fe.params().get(0, 0) - 1.497_297_297_297_297_5).abs() < 1e-9);
        assert!((fe.r_squared_within() - 0.975_885_532_591_415).abs() < 1e-9);
        assert!((fe.r_squared_between() - 0.943_825_384_692_178_9).abs() < 1e-9);
        assert!((fe.r_squared_overall() - 0.947_558_874_189_504_9).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_one_way_r_squared_between_returns_zero_when_entity_means_are_zero() {
        // `fe_r_squared_between`の`TSS <= 0`ガード（`linearmodels`と同じく`0.0`を返す）が
        // 他のテストでは通らない。各エンティティの
        // `y`平均がちょうどゼロ（`TSS_between = Σȳ_i.² = 0`）になるデータで検証する。
        // within/overallは退化しない（`y`自体の分散はあるため）ことも合わせて確認し、
        // `linearmodels`の実測値と数値比較する（Pythonで独立に計算・検算済み）。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let x = vec![1.0, 3.0, 2.0, 6.0, 1.0, 4.0];
        let y = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert!((*fe.params().get(0, 0) - (-1.310_344_827_586_206_9)).abs() < 1e-9);
        assert!((fe.r_squared_within() - 0.889_162_561_576_354_6).abs() < 1e-9);
        assert_eq!(fe.r_squared_between(), 0.0);
        assert!((fe.r_squared_overall() - (-2.330_219_126_889_757_4)).abs() < 1e-9);
    }

    // ── 固定効果自体（α_i）の復元 ───────────────────────────────────────

    #[test]
    fn fe_estimator_fit_one_way_fixed_effects_matches_fixest_reference() {
        // `fixest::feols(y ~ x | entity)`の`fixef()`と数値比較する（1-wayは正規化の任意性が
        // 無いため無条件に一致する、モジュールdoc「固定効果自体（α_i）の復元」参照）。
        // 期待値はRで独立に計算・検算済み（`options(digits=15)`）。
        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        let FixedEffects::OneWay(effects) = fe.fixed_effects() else {
            panic!("1-way FE must return FixedEffects::OneWay");
        };
        assert!((effects["a"] - 4.527_777_777_777_777).abs() < 1e-9);
        assert!((effects["b"] - 1.523_148_148_148_147).abs() < 1e-9);
        assert!((effects["c"] - 5.657_407_407_407_407).abs() < 1e-9);
        assert!((effects["d"] - 0.393_518_518_518_517).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_fixed_effects_matches_fixest_reference() {
        // `fixest::feols(y ~ x | entity + time)`の`fixef()`と数値比較する。
        // `fixest_reference_input`は観測順で最初に現れる時点（"1"）と辞書順で最小の時点
        // （"1"）が一致する構成のため、この入力に限り`fixest`と数値完全一致する
        // （モジュールdoc「固定効果自体（α_i）の復元」参照。期待値はRで独立に計算・
        // 検算済み）。
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

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();

        let FixedEffects::TwoWay { entity, time } = fe.fixed_effects() else {
            panic!("2-way FE must return FixedEffects::TwoWay");
        };
        let time = by_label(time);
        assert!((entity["a"] - 3.373_831_775_700_935).abs() < 1e-9);
        assert!((entity["b"] - 1.336_448_598_130_841).abs() < 1e-9);
        assert!((entity["c"] - 5.277_258_566_978_194).abs() < 1e-9);
        assert!((entity["d"] - (-0.566_978_193_146_417)).abs() < 1e-9);

        assert_eq!(time["1"], 0.0); // 辞書順で最初の時点が基準（γ_{t_ref}=0）
        assert!((time["2"] - 2.883_177_570_093_46).abs() < 1e-9);
        assert!((time["3"] - 4.060_747_663_551_40).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_fixed_effects_reproduces_fitted_values() {
        // 正規化の選び方に関わらず`x_it'β̂ + α_i + γ_t`は元の`y_it`から within残差
        // `ε̂_it`を引いた値に一致する（モデルの恒等式そのもの）ことを回帰ガードする
        // （`fixest`の具体的な基準時点選択に依存しない、正規化非依存の不変条件）。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            std::slice::from_ref(&x),
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();
        let beta = *fe.params().get(0, 0);
        let residuals = fe.residuals();

        let FixedEffects::TwoWay {
            entity: entity_effects,
            time: time_effects,
        } = fe.fixed_effects()
        else {
            panic!("2-way FE must return FixedEffects::TwoWay");
        };
        let time_effects = by_label(time_effects);

        for i in 0..y.len() {
            let predicted = beta * x[i]
                + entity_effects[&entity[i]]
                + time_effects[&time[i]]
                + *residuals.get(i, 0);
            assert!(
                (predicted - y[i]).abs() < 1e-9,
                "row {i}: predicted={predicted}, y={}",
                y[i]
            );
        }
    }

    #[test]
    fn fe_estimator_fit_one_way_fixed_effects_with_no_regressors_equals_group_means() {
        // k=0（回帰変数なし）では`α_i`は単に`ȳ_i.`そのものになる境界ケース
        // （`slope_only_residual`の`fitted=0`分岐、`fe_estimator_fit_with_no_regressors_
        // estimates_fixed_effects_only_model`と同じ入力）。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let y = [1.0, 3.0, 5.0, 7.0, 2.0, 4.0];
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        let FixedEffects::OneWay(effects) = fe.fixed_effects() else {
            panic!("1-way FE must return FixedEffects::OneWay");
        };
        assert!((effects["a"] - 2.0).abs() < 1e-12);
        assert!((effects["b"] - 6.0).abs() < 1e-12);
        assert!((effects["c"] - 3.0).abs() < 1e-12);
        // F統計量: k=0（検定対象の傾き係数が無い）は`OlsEstimator`と同じく`None`。
        assert!(fe.f_statistic().is_none());
        assert!(fe.f_p_value().is_none());
    }

    #[test]
    fn fe_estimator_fit_one_way_fixed_effects_on_unbalanced_panel() {
        // 1-wayの`α_i = ȳ_i. - x̄_i.'β̂`は正規化の任意性が無く、不均衡パネル
        // （T_a=2, T_b=4, T_c=3）でも単純にそのまま成立することを確認する（バランスパネル
        // のみのテストでは重み付けの誤りを見逃すため、`fixed_effects()`も不均衡ケースを
        // 回帰ガードする）。
        // 入力は`fe_estimator_fit_one_way_r_squared_between_matches_linearmodels_on_
        // unbalanced_panel`と同じ（期待値はPythonで独立に計算・検算済み）。
        let entity = strings(&["a", "a", "b", "b", "b", "b", "c", "c", "c"]);
        let x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 1.0, 4.0, 2.0];
        let y = [2.0, 6.0, 5.0, 9.0, 8.0, 11.0, 3.0, 7.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        let FixedEffects::OneWay(effects) = fe.fixed_effects() else {
            panic!("1-way FE must return FixedEffects::OneWay");
        };
        assert!((effects["a"] - 1.005_405_405_405_405).abs() < 1e-9);
        assert!((effects["b"] - 1.886_486_486_486_485_4).abs() < 1e-9);
        assert!((effects["c"] - 1.172_972_972_972_972_5).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_fixed_effects_with_no_regressors() {
        // 2-way・k=0（回帰変数なし）の組み合わせ境界ケース。`slope_only_residual`の
        // `fitted=0`分岐と2-way正規化ロジック（`t_ref`選択・大域平均吸収）の組み合わせを
        // 検証する（1-wayのk=0境界は上のテストにある）。time="9"/"10"はDKの辞書順規約が数値順と食い違う
        // ケース（辞書順では"10" < "9"）でも規約通り"10"が基準になることを合わせて
        // 確認する（期待値はPythonで独立に計算・検算済み）。
        let entity = strings(&["e1", "e1", "e2", "e2"]);
        let time = strings(&["9", "10", "9", "10"]);
        let y = [1.0, 3.0, 5.0, 9.0];
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();

        let FixedEffects::TwoWay { entity, time } = fe.fixed_effects() else {
            panic!("2-way FE must return FixedEffects::TwoWay");
        };
        let time = by_label(time);
        assert_eq!(time["10"], 0.0); // 辞書順で"10" < "9"のため基準はこちら
        assert!((time["9"] - (-3.0)).abs() < 1e-12);
        assert!((entity["e1"] - 3.5).abs() < 1e-12);
        assert!((entity["e2"] - 8.5).abs() < 1e-12);
        // F統計量: k=0（検定対象の傾き係数が無い）は`OlsEstimator`と同じく`None`。
        assert!(fe.f_statistic().is_none());
        assert!(fe.f_p_value().is_none());

        // 不変条件: k=0でも `α_i + γ_t + ε̂_it = y_it`（正規化の選び方に依存しない）。
        let residuals = fe.residuals();
        let entity_ids = ["e1", "e1", "e2", "e2"];
        let time_ids = ["9", "10", "9", "10"];
        for i in 0..y.len() {
            let predicted = entity[entity_ids[i]] + time[time_ids[i]] + *residuals.get(i, 0);
            assert!((predicted - y[i]).abs() < 1e-9);
        }
    }

    // ── cov_type対応 ─────────────────────────────────────────────────────

    #[test]
    fn fe_estimator_fit_one_way_hc1_hc2_hc3_match_fixest_reference() {
        // 同じデータでの`fixest::feols(y ~ x | id)`の`vcov="HC1"/"HC2"/"HC3"`と
        // 数値比較する（linearmodelsはHC1相当の"robust"のみでHC2/HC3を提供しないため、
        // 3種とも`fixest`のみで検証する例外、モジュールdoc「`cov_type`対応」参照）。
        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let hc1 = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Hc1, 0.95).unwrap();
        assert!((*hc1.std_errors().get(0, 0) - 0.467_996_773_819_759).abs() < 1e-9);
        assert!((*hc1.test_stats().get(0, 0) - 2.997_409_076_837).abs() < 1e-6);
        assert!((*hc1.p_values().get(0, 0) - 0.020_015_356_643_180_1).abs() < 1e-6);
        // F統計量: k=1のため「1自由度のF検定は両側t検定と代数的に等価」
        // （モジュールdoc「自由度とF統計量」参照）。HC1のcov_paramsが正しく
        // wald_f_testに渡っていることの回帰ガード（classical以外のcov_typeでの唯一の
        // F統計量検証）。
        assert!((hc1.f_statistic().unwrap() - hc1.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((hc1.f_p_value().unwrap() - *hc1.p_values().get(0, 0)).abs() < 1e-9);

        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();
        let hc2 = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Hc2, 0.95).unwrap();
        assert!((*hc2.std_errors().get(0, 0) - 0.492_939_313_874_837).abs() < 1e-9);
        assert!((*hc2.test_stats().get(0, 0) - 2.845_741_328_178_91).abs() < 1e-6);
        assert!((*hc2.p_values().get(0, 0) - 0.024_839_464_368_821_2).abs() < 1e-6);
        assert!((hc2.f_statistic().unwrap() - hc2.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((hc2.f_p_value().unwrap() - *hc2.p_values().get(0, 0)).abs() < 1e-9);

        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();
        let hc3 = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Hc3, 0.95).unwrap();
        assert!((*hc3.std_errors().get(0, 0) - 0.687_184_240_890_824).abs() < 1e-9);
        assert!((*hc3.test_stats().get(0, 0) - 2.041_341_599_975_14).abs() < 1e-6);
        assert!((*hc3.p_values().get(0, 0) - 0.080_553_228_223_064_4).abs() < 1e-6);
        assert!((hc3.f_statistic().unwrap() - hc3.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((hc3.f_p_value().unwrap() - *hc3.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_hc1_hc2_hc3_match_fixest_reference() {
        // 同じデータでの`fixest::feols(y ~ x | id + t)`の`vcov="HC1"/"HC2"/"HC3"`と
        // 数値比較する。
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
        let hc1 = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Hc1, 0.95).unwrap();
        assert!((*hc1.std_errors().get(0, 0) - 0.205_876_715_757_555).abs() < 1e-9);
        assert!((hc1.f_statistic().unwrap() - hc1.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((hc1.f_p_value().unwrap() - *hc1.p_values().get(0, 0)).abs() < 1e-9);

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
        let hc2 = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Hc2, 0.95).unwrap();
        assert!((*hc2.std_errors().get(0, 0) - 0.304_984_723_480_691).abs() < 1e-9);
        assert!((hc2.f_statistic().unwrap() - hc2.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((hc2.f_p_value().unwrap() - *hc2.p_values().get(0, 0)).abs() < 1e-9);

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
        let hc3 = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Hc3, 0.95).unwrap();
        assert!((*hc3.std_errors().get(0, 0) - 0.737_275_671_443_649).abs() < 1e-9);
        assert!((hc3.f_statistic().unwrap() - hc3.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((hc3.f_p_value().unwrap() - *hc3.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_one_way_cluster_on_entity_matches_fixest_nested_k() {
        // 1-way FEでクラスター変数がentityと同じ（`groups: None`＝デフォルト）場合、
        // fixestの`feols(y~x|entity, cluster=~entity)`と数値一致する（`K.fixef=
        // "nonnested"`の既定分岐、entity FEが全次元ネスト、`fe_cluster_k_correction`の
        // docコメント参照。`K=df_model-n_entities+1=5-4+1=2`ではなく`K=k+1=2`——
        // ここでは`k=1`なので`K=2`）。期待値はRで独立に計算・検算済み
        // （`options(digits=16)`、実装時）。
        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster { groups: None },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.603_104_702_643_124_5).abs() < 1e-9);
        assert!((*fe.test_stats().get(0, 0) - 2.325_927_441_172_424).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.102_527_902_595_719_5).abs() < 1e-6);
        assert!((fe.f_statistic().unwrap() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value().unwrap() - *fe.p_values().get(0, 0)).abs() < 1e-9);
        // `f_df()`（カバレッジ監査で判明した未検証の単純getter。分母自由度が
        // `df_resid`から`df_inference`に変わったため、`df_resid`
        // （`n-df_model=12-5=7`）とは異なる`df_inference`（`G-1=n_entities-1=3`）を
        // 返すことを確認する）。
        assert_eq!(fe.df_resid(), 7);
        assert_eq!(fe.df_inference(), 3);
        assert_eq!(fe.f_df(), Some((1, 3)));
    }

    #[test]
    fn fe_estimator_fit_two_way_cluster_on_entity_matches_fixest_partially_nested_k() {
        // 2-way FEでentityクラスターは、entity次元だけがネストしtime次元はネストしない
        // 「部分ネスト」ケース（`fe_cluster_k_correction`のdocコメント参照）。
        // fixestの`feols(y~x|entity+time, cluster=~entity)`と数値一致する。
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

        let fe = FeEstimator::fit(
            input,
            FeEffects::TwoWay,
            FeCovType::Cluster { groups: None },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.158_754_475_018_330_4).abs() < 1e-9);
        assert!((*fe.test_stats().get(0, 0) - 5.180_514_794_604_026).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.013_962_411_468_796_73).abs() < 1e-6);
        assert!((fe.f_statistic().unwrap() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value().unwrap() - *fe.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_one_way_cluster_on_non_nested_variable_matches_fixest_full_k() {
        // 1-way FEでも、クラスター変数がentityと無関係（ここでは`time`）なら
        // どの次元もネストせず`K=df_model`（フルカウント、`fixef_dimension_nested_
        // within_cluster`がfalseになるケース）。fixestの
        // `feols(y~x|entity, cluster=~time)`と数値一致する。
        let (entity, time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&time)),
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.115_060_764_365_329_2).abs() < 1e-9);
        assert!((fe.f_statistic().unwrap() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value().unwrap() - *fe.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn validate_balanced_panel_accepts_shuffled_rows_when_entities_and_periods_differ() {
        // 3エンティティ×2時点（n_entities != n_periods）で、行を観測順に並べていない。
        let entity = codes(&["c", "a", "b", "a", "c", "b"]);
        let time = codes(&["2", "1", "2", "2", "1", "1"]);
        assert_eq!(
            validate_balanced_panel(&entity, &time, &EffectNames::default()),
            Ok(())
        );
    }

    #[test]
    fn validate_balanced_panel_rejects_duplicate_pair_offsetting_a_missing_one() {
        // 3×2。(a,1)が重複し(a,2)が欠落するが、n_obs=6=3*2で件数は一致してしまう。
        let entity = codes(&["a", "a", "b", "b", "c", "c"]);
        let time = codes(&["1", "1", "1", "2", "1", "2"]);
        assert_eq!(
            validate_balanced_panel(&entity, &time, &EffectNames::default()),
            Err(PanelError::UnbalancedPanelForTwoWay {
                n_obs: 6,
                entity_column: "entity".to_string(),
                n_entities: 3,
                time_column: "time".to_string(),
                n_periods: 2,
            })
        );
    }

    #[test]
    fn validate_balanced_panel_reports_the_column_names_set_by_effect_names() {
        let entity = codes(&["a", "a", "b"]);
        let time = codes(&["1", "2", "1"]);
        let names = EffectNames::new("firm".to_string(), Some("year".to_string()));
        let err = validate_balanced_panel(&entity, &time, &names).unwrap_err();
        assert_eq!(
            err,
            PanelError::UnbalancedPanelForTwoWay {
                n_obs: 3,
                entity_column: "firm".to_string(),
                n_entities: 2,
                time_column: "year".to_string(),
                n_periods: 2,
            }
        );
    }

    #[test]
    fn fe_estimator_fit_cluster_propagates_insufficient_clusters_error() {
        // クラスター数2未満は`CommonError::InsufficientClusters`（`validate_cluster_groups`）
        // として伝播する。すべて同じクラスターに属する（g=1）データを使う。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 3.0, 2.0, 6.0, 4.0, 10.0];
        let single_cluster = strings(&["g", "g", "g", "g", "g", "g"]);
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&single_cluster)),
            },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::Common(CommonError::InsufficientClusters {
                setting: ClusterSetting::CovType,
                g: 1
            })
        );
    }

    #[test]
    fn fe_estimator_fit_cluster_propagates_insufficient_clusters_for_inference_error() {
        // クラスター数g(=2)が傾き係数の数q(=k=2)以下（g<=q、境界は厳密不等号）は
        // `CommonError::InsufficientClustersForInference`として伝播する
        // （`validate_cluster_count_covers_slopes`、`.claude/rules/testing-policy.md`
        // 「G<=qで書く」の方針）。g=2自体は`InsufficientClusters`（g<2）は回避している。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x1 = vec![1.0, 3.0, 2.0, 6.0, 4.0, 10.0];
        let x2 = vec![2.0, 5.0, 1.0, 9.0, 3.0, 7.0];
        let two_clusters = strings(&["1", "1", "1", "2", "2", "2"]);
        let input = FeInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&two_clusters)),
            },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::Common(CommonError::InsufficientClustersForInference { g: 2, q: 2 })
        );
    }

    // ── Driscoll-Kraay型パネルHAC対応 ───────────────────────────────────────

    #[test]
    fn fe_estimator_fit_one_way_hac_matches_fixest_default_bandwidth() {
        // fixestの`feols(y~x|entity, vcov="DK", panel.id=~entity+time)`と数値比較する
        // （5.1節、DKの主リファレンス）。n_periods=3のため既定バンド幅は
        // `floor(4*(3/100)^(2/9))=1`（`resolve_dk_bandwidth`）。期待値はRで独立に
        // 計算・検算済み（`options(digits=16)`、実装時）。
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

        let dk_time = dk_time_from(&input);
        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: None,
                time: dk_time,
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.params().get(0, 0) - 1.402_777_777_777_78).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.112_778_272_530_122_7).abs() < 1e-9);
        assert!((*fe.test_stats().get(0, 0) - 12.438_369_078_610_43).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.006_401_580_635_206_468).abs() < 1e-9);
        assert!((*fe.conf_lower().get(0, 0) - 0.917_532_035_619_617).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.888_023_519_935_939).abs() < 1e-6);
        assert!((fe.f_statistic().unwrap() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value().unwrap() - *fe.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_uses_explicit_time_override_without_fe_input_time() {
        // `FeCovType::Dk.time`（明示指定）は`FeInput.time_keys()`を経由せずに
        // DK HACを成立させられる（`engine_pybind`の`FEOptions.dk_time`が1-way FE + DK HAC
        // の組み合わせをこの経路で配線する想定）。`FeInput::from_columns`には`time=None`を
        // 渡し、`fe_estimator_fit_one_way_hac_matches_fixest_default_bandwidth`と
        // 同じ結果になることを確認する（同じ`time`列を使っているため数値は完全一致する）。
        let (entity, time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: None,
                time: TimeKeys::lexicographic(time),
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.params().get(0, 0) - 1.402_777_777_777_78).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.112_778_272_530_122_7).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_ignores_fe_input_time() {
        // `FeInput.time_keys()`にも`time`があっても、DKの時点列は`FeCovType::Dk.time`だけから
        // 決まる（`FeInput.time_keys()`は2-wayの固定効果の時間次元専用、モジュールdoc
        // 「Driscoll-Kraay型パネルHAC」参照）ことを確認する。`FeInput.time_keys()`にわざと
        // 全観測が同一のダミー時点列を渡し、それが無視されて`FeCovType::Dk.time`の方の結果と
        // 一致することを確認する。
        let (entity, time, x, y) = fixest_reference_input();
        let dummy_time = strings(&["z", "z", "z", "z", "z", "z", "z", "z", "z", "z", "z", "z"]);
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&dummy_time),
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: None,
                time: TimeKeys::lexicographic(time),
            },
            0.95,
        )
        .unwrap();

        // `dummy_time`（全観測が同一時点）をそのまま使っていたら`t_periods=1`となり
        // `resolve_dk_bandwidth`が`InsufficientDkPeriods`で拒否する（優先順位が
        // 逆だった場合はこのテスト自体がエラーで失敗する）。使われている`time`
        // （`t_periods=3`）を使った場合の既知の値と一致することで、優先順位を確認する。
        assert!((*fe.std_errors().get(0, 0) - 0.112_778_272_530_122_7).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_dk_bandwidth_used_reflects_resolved_bandwidth() {
        // `t_periods=3`の既定バンド幅は`floor(4*(3/100)^(2/9))=1`。明示指定はその値、
        // `Dk`以外は`None`。
        let (entity, time, x, y) = fixest_reference_input();
        let bandwidth_used = |cov_type: FeCovType| {
            let input = FeInput::from_columns(
                &y,
                std::slice::from_ref(&x),
                vec!["x".to_string()],
                &entity,
                Some(&time),
                "y".into(),
            )
            .unwrap();
            FeEstimator::fit(input, FeEffects::OneWay, cov_type, 0.95)
                .unwrap()
                .dk_bandwidth_used()
        };
        let dk = |bandwidth: Option<i64>| FeCovType::Dk {
            bandwidth,
            time: TimeKeys::lexicographic(time.clone()),
        };

        assert_eq!(bandwidth_used(dk(None)), Some(1));
        assert_eq!(bandwidth_used(dk(Some(0))), Some(0));
        assert_eq!(bandwidth_used(dk(Some(2))), Some(2));
        assert_eq!(bandwidth_used(FeCovType::Classical), None);
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_with_explicit_bandwidth_matches_default() {
        // 既定バンド幅（n_periods=3 → 1）と明示的に`bandwidth=Some(1)`を指定した場合が
        // 一致することを確認する（`resolve_dk_bandwidth`のNone分岐とSome分岐が同じ値に
        // 解決されることの回帰ガード）。
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

        let dk_time = dk_time_from(&input);
        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(1),
                time: dk_time,
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.112_778_272_530_122_7).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_with_bandwidth_two_scales_unchanged_kernel_by_new_correction() {
        // `bandwidth=Some(2)`（n_periods=3のため許容範囲`[0,3)`の上限）でラグ項ループ
        // （`for l in 1..=bandwidth`）が複数回（l=1,2）実行されるケースを検証する
        // （既定・`Some(1)`のテストはl=1の1回しか通らない。
        // `testing-policy.md`が警告する「ループ本体がテストで一度も複数回実行されない」
        // 落とし穴と同型）。
        //
        // **fixestの`vcov=DK(2)`とは意図的に数値比較しない**（devcontainer内のfixest
        // 0.14.2で実地確認済み）。本テストの`t_periods=3`・`bandwidth=2`は許容範囲
        // `[0, t_periods)`の上限ちょうど（`bandwidth == t_periods - 1`）で、この境界では
        // fixestが最後のラグ項（`l=bandwidth`）を落とした値を返し、標準のBartlettカーネルを
        // 実装した本実装と一致しない。原因はfixestのC++実装`cpp_driscoll_kraay`のoff-by-one
        // （ラグの個数`L = bandwidth + 1`に`L > T - 1`の上限を課すため、`bandwidth == T - 1`
        // のときだけ最大ラグが`T - 2`に切り詰められる）で、`bandwidth <= t_periods - 2`では
        // 一致する。本実装は標準カーネルを維持する（`docs/spec/fe-spec.md`3.3節7.参照）。
        // 境界の標準カーネルは`panel_driscoll_kraay_cov_params`の単体テスト（定義式の手計算）と
        // plmとの照合（`fe_plm_crosscheck.json`の`dk_max_bandwidth`）で検証している。
        //
        // このテスト自体は、変更していないカーネル計算（`bandwidth=1`の既定テストで
        // fixestと一致確認済みの実装）に新しい小標本補正（`(t_periods/(t_periods-1))×
        // ((n-1)/(n-K))`）が正しく適用されることを確認する回帰ガードとして残す
        // （期待値は本実装自身の出力を固定しただけで、外部リファレンスとの照合ではない）。
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

        let dk_time = dk_time_from(&input);
        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(2),
                time: dk_time,
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.092_083_073_923_780_41).abs() < 1e-9);
        assert!((*fe.test_stats().get(0, 0) - 15.233_828_737_503_853).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.004_281_399_881_913_783).abs() < 1e-9);
        assert!((*fe.conf_lower().get(0, 0) - 1.006_576_288_395_902).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.798_979_267_159_653_4).abs() < 1e-6);
    }

    #[test]
    fn fe_estimator_fit_two_way_hac_matches_fixest_default_bandwidth() {
        // 同じデータでの2-way FE版（`entity+time`固定効果）。fixestの
        // `feols(y~x|entity+time, vcov="DK", panel.id=~entity+time)`と数値比較する。
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

        let dk_time = dk_time_from(&input);
        let fe = FeEstimator::fit(
            input,
            FeEffects::TwoWay,
            FeCovType::Dk {
                bandwidth: None,
                time: dk_time,
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.params().get(0, 0) - 0.822_429_906_542_056).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.258_392_668_199_213_7).abs() < 1e-9);
        assert!((*fe.test_stats().get(0, 0) - 3.182_868_586_302_089).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.086_146_397_773_918_86).abs() < 1e-9);
        assert!((*fe.conf_lower().get(0, 0) - (-0.289_344_012_632_537_7)).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.934_203_825_716_65).abs() < 1e-6);
        assert!((fe.f_statistic().unwrap() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value().unwrap() - *fe.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_hac_with_bandwidth_two_scales_unchanged_kernel_by_new_correction() {
        // 1-way版（`fe_estimator_fit_one_way_hac_with_bandwidth_two_scales_unchanged_
        // kernel_by_new_correction`）と同様、2-way FEでもラグ項ループが複数回（l=1,2）
        // 実行されるケースを検証する。同テストのコメントの通り、
        // `bandwidth == t_periods - 1`（ここでは`2 == 3 - 1`）という境界値はfixestの
        // C++実装のoff-by-oneにより本実装と一致しないため（同テストのコメント参照）、
        // fixestとの数値比較はせず回帰ガードとして期待値を固定する。
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

        let dk_time = dk_time_from(&input);
        let fe = FeEstimator::fit(
            input,
            FeEffects::TwoWay,
            FeCovType::Dk {
                bandwidth: Some(2),
                time: dk_time,
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.210_976_730_121_450_27).abs() < 1e-9);
        assert!((*fe.test_stats().get(0, 0) - 3.898_201_977_386_883).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.059_950_140_715_886_67).abs() < 1e-9);
        assert!((*fe.conf_lower().get(0, 0) - (-0.085_329_697_228_618_5)).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.730_189_510_312_731).abs() < 1e-6);
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_with_zero_bandwidth_matches_cluster_on_non_nested_time() {
        // `bandwidth=Some(0)`はラグ項なし（`Ŝ = Ŝ₀`）に退化し、これは`time`でクラスター
        // した場合（`fe_estimator_fit_one_way_cluster_on_non_nested_variable_matches_
        // fixest_full_k`と同じ`entity`/`time`）の`Ŝ`と数式的に同一になる（どちらも
        // `Σ_t (Σ_{i:time_i=t} x̃_i ε̂_i)(...)'`で、`K=df_model`（DKは常にフルカウント・
        // clusterもこのケースではどの次元もネストしないためフルカウント）・`G=t_periods`
        // （clusterの`G`も同じ`time`列のユニーク数）のスケールも一致する。モジュールdoc
        // 「Driscoll-Kraay型パネルHAC」参照）。2つの独立した実装
        // （`panel_cluster_cov_params`と`panel_driscoll_kraay_cov_params`）が同じ値に
        // 収束することを確認する回帰ガード（OLSの`fit_hac_with_zero_lags_matches_hc0`と
        // 同型）。
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

        let dk_time = dk_time_from(&input);
        let hac = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: dk_time,
            },
            0.95,
        )
        .unwrap();

        let (entity, time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();
        let cluster = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&time)),
            },
            0.95,
        )
        .unwrap();

        assert!(
            (*hac.std_errors().get(0, 0) - *cluster.std_errors().get(0, 0)).abs() < 1e-9,
            "hac(bandwidth=0)={}, cluster(time)={}",
            *hac.std_errors().get(0, 0),
            *cluster.std_errors().get(0, 0)
        );
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_with_single_time_period_is_rejected() {
        // `t_periods=1`（全観測が同一の`time`ラベル）という退化した境界ケース
        // （`resolve_dk_bandwidth`のNone分岐が`bandwidth=t_periods`を
        // 返しうる唯一のケース）。
        //
        // DKの小標本補正を`(t_periods/(t_periods-1))×((n-1)/(n-K))`
        // （fixestの`ssc()`、`t_periods`をclusterの`G`と同じ役割で使う）に変更した結果、
        // `t_periods=1`は`t_periods/(t_periods-1)=1/0`が発散し計算が成立しなくなった
        // （クラスターの`G=1`が`validate_cluster_groups`で拒否されるのと同じ理由）。
        // `resolve_dk_bandwidth`が`PanelError::InsufficientDkPeriods`で早期に拒否する
        // （`G/(G-1)`型の補正は`t_periods=1`で未定義になる）。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let time = strings(&["1", "1", "1", "1", "1", "1"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 3.0, 2.0, 6.0, 4.0, 10.0];
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let dk_time = dk_time_from(&input);
        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: None,
                time: dk_time,
            },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::InsufficientDkPeriods { t_periods: 1 }
        );
    }

    /// 4エンティティ×3期間の1-way FE入力（説明変数は`x1`〜`x3`の先頭`k`個）。DKの
    /// `t_periods=3`に対し`k=3`（拒否）と`k=2`（成功）の境界を作るため。`T=2`は使わない:
    /// 1-wayのwithin変換では`x̃_i1 = -x̃_i2`・`ẽ_i1 = -ẽ_i2`となり時点スコアが
    /// `h_1 = h_2 = 0`に退化する（`rank(S) = 0`）ため、`t <= q`の規則を検証する入力として不適。
    fn three_period_input(k: usize, time: Option<&[String]>) -> FeInput {
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]);
        let columns = [
            vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 0.0, 2.0, 1.0, 3.0, 7.0, 4.0],
            vec![2.0, 1.0, 4.0, 0.0, 3.0, 1.0, 5.0, 2.0, 6.0, 1.0, 1.0, 3.0],
            vec![4.0, 2.0, 1.0, 3.0, 3.0, 5.0, 2.0, 6.0, 3.0, 0.0, 4.0, 2.0],
        ];
        let y = [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0, 5.0, 9.5, 4.0, 7.3];
        let names = ["x1", "x2", "x3"]
            .iter()
            .take(k)
            .map(|n| n.to_string())
            .collect();
        FeInput::from_columns(&y, &columns[..k], names, &entity, time, "y".into()).unwrap()
    }

    fn three_period_time() -> Vec<String> {
        strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2", "3"])
    }

    #[test]
    fn fe_estimator_fit_hac_rejects_periods_not_covering_slopes() {
        // `t_periods=3`ではDK共分散のrankが`t-1=2`以下で、F検定の傾き`k=3`個の部分行列が
        // 構造的に特異になる。`wald_f_test`の数値的な特異性判定を待たず弾く。
        let time = three_period_time();
        let dk_time = TimeKeys::lexicographic(time.clone());
        let result = FeEstimator::fit(
            three_period_input(3, Some(&time)),
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: dk_time,
            },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            PanelError::InsufficientDkPeriodsForInference { t_periods: 3, q: 3 }
                .in_stage(Stage::OverallFTest)
        );
    }

    #[test]
    fn fe_estimator_fit_hac_accepts_periods_one_above_slopes() {
        // 境界の成功パス: `t_periods=3 > k=2`。
        let time = three_period_time();
        let dk_time = TimeKeys::lexicographic(time.clone());
        let fe = FeEstimator::fit(
            three_period_input(2, Some(&time)),
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: dk_time,
            },
            0.95,
        )
        .unwrap();
        assert!(fe.f_statistic().unwrap().is_finite());
        assert!((0..2).all(|j| *fe.std_errors().get(j, 0) > 0.0));
    }

    #[test]
    fn fe_estimator_fit_hac_counts_periods_from_time_override() {
        // `FeCovType::Dk.time`の上書きがあれば、時点数はその列で数える（入力の`time`が
        // `None`の1-way FEでも同じ判定になる）。
        let result = FeEstimator::fit(
            three_period_input(3, None),
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: TimeKeys::lexicographic(three_period_time()),
            },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            PanelError::InsufficientDkPeriodsForInference { t_periods: 3, q: 3 }
                .in_stage(Stage::OverallFTest)
        );
    }

    /// 4エンティティ×2期間・`k=1`（`k = 0`なら説明変数なし）。`break_pattern`なら
    /// エンティティ`a`に時点`1`の観測を1つ足し（不均衡な1-way）、「全エンティティが
    /// 2時点に1観測ずつ」のパターンを崩す。
    fn two_period_input(k: usize, break_pattern: bool) -> (FeInput, Vec<String>) {
        let mut entity = strings(&["a", "a", "b", "b", "c", "c", "d", "d"]);
        let mut time = strings(&["1", "2", "1", "2", "1", "2", "1", "2"]);
        let mut x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0, 2.0];
        let mut y = vec![3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0];
        if break_pattern {
            entity.push("a".to_string());
            time.push("1".to_string());
            x.push(2.5);
            y.push(5.0);
        }
        let (columns, names) = if k == 0 {
            (vec![], vec![])
        } else {
            (vec![x], vec!["x".to_string()])
        };
        let input =
            FeInput::from_columns(&y, &columns, names, &entity, Some(&time), "y".into()).unwrap();
        (input, time)
    }

    /// `input`が持つ時点（ラベルの辞書順）を、DKの時点列としてそのまま渡す。
    fn dk_time_from(input: &FeInput) -> TimeKeys {
        let Some(time) = input.time_keys() else {
            panic!("test input must have a time column");
        };
        time.clone()
    }

    fn dk_bandwidth_zero(input: &FeInput) -> FeCovType {
        FeCovType::Dk {
            bandwidth: Some(0),
            time: dk_time_from(input),
        }
    }

    #[test]
    fn fe_estimator_fit_hac_rejects_degenerate_two_period_panel() {
        // 全エンティティが2時点に1観測ずつだと、within変換で時点スコアが`h_1 = h_2 = 0`に
        // 退化しDK共分散が恒等的にゼロになる（`k=1`でも`t > q`の検証は通ってしまう）。
        let (input, _) = two_period_input(1, false);
        let dk_cov = dk_bandwidth_zero(&input);
        let result = FeEstimator::fit(input, FeEffects::OneWay, dk_cov, 0.95);
        assert_eq!(result.unwrap_err(), PanelError::DegenerateDkTwoPeriods);
    }

    #[test]
    fn fe_estimator_fit_hac_rejects_degenerate_two_period_panel_two_way() {
        // 2-way（バランスパネルの二重デミーニング）でもエンティティ内の和がゼロになり同じ退化。
        let (input, _) = two_period_input(1, false);
        let dk_cov = dk_bandwidth_zero(&input);
        let result = FeEstimator::fit(input, FeEffects::TwoWay, dk_cov, 0.95);
        assert_eq!(result.unwrap_err(), PanelError::DegenerateDkTwoPeriods);
    }

    #[test]
    fn fe_estimator_fit_cluster_rejects_degenerate_two_group_split() {
        // 2時点のパネルを`time`でクラスタリング（`G=2 > k=1`の検証は通る）すると、
        // DKと同じ理由でクラスタースコアが恒等的にゼロになる。
        let (input, time) = two_period_input(1, false);
        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&time)),
            },
            0.95,
        );
        assert_eq!(result.unwrap_err(), PanelError::DegenerateClusterTwoGroups);
    }

    #[test]
    fn fe_estimator_fit_two_group_split_is_accepted_when_pattern_is_broken() {
        // 1エンティティでも同じ時点に2観測あればスコアは退化せず、標準誤差は正で有限。
        let (input, time) = two_period_input(1, true);
        let dk_cov = dk_bandwidth_zero(&input);
        let dk = FeEstimator::fit(input, FeEffects::OneWay, dk_cov, 0.95).unwrap();
        assert!(*dk.std_errors().get(0, 0) > 1e-8);
        assert!(dk.f_statistic().unwrap().is_finite());

        let (input, _) = two_period_input(1, true);
        let cluster = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&time)),
            },
            0.95,
        )
        .unwrap();
        assert!(*cluster.std_errors().get(0, 0) > 1e-8);
    }

    #[test]
    fn fe_estimator_fit_hac_two_period_panel_without_regressors_is_accepted() {
        // `k=0`（固定効果のみ）は標準誤差・F検定を計算しないため退化を問題にしない。
        let (input, _) = two_period_input(0, false);
        let dk_cov = dk_bandwidth_zero(&input);
        assert!(FeEstimator::fit(input, FeEffects::OneWay, dk_cov, 0.95).is_ok());
    }

    /// 4エンティティ×2観測・`k=1`の1-way FE入力で、入力の`time`列を`time`で与える
    /// （`Dk.time`上書きの経路を検証するため、時点ラベルの分け方だけを変えられるようにする）。
    fn four_by_two_input(time: &[&str]) -> FeInput {
        let entity = strings(&["a", "a", "b", "b", "c", "c", "d", "d"]);
        let x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0, 2.0];
        let y = [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0];
        FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&strings(time)),
            "y".into(),
        )
        .unwrap()
    }

    /// 3時点に分かれるが各エンティティ2観測のラベル（`t_periods=3`、退化しない）。
    const THREE_PERIOD_LABELS: [&str; 8] = ["1", "2", "2", "3", "3", "1", "1", "2"];
    const TWO_PERIOD_LABELS: [&str; 8] = ["1", "2", "1", "2", "1", "2", "1", "2"];

    #[test]
    fn fe_estimator_fit_hac_uses_resolved_dk_time_for_degeneracy_check() {
        // 入力の`time`は3時点だが、`Dk.time`上書きが2時点の退化パターン→拒否。
        let result = FeEstimator::fit(
            four_by_two_input(&THREE_PERIOD_LABELS),
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: lex_time(&TWO_PERIOD_LABELS),
            },
            0.95,
        );
        assert_eq!(result.unwrap_err(), PanelError::DegenerateDkTwoPeriods);

        // 逆に入力の`time`が2時点の退化パターンでも、上書きが3時点なら通す。
        let fe = FeEstimator::fit(
            four_by_two_input(&TWO_PERIOD_LABELS),
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: lex_time(&THREE_PERIOD_LABELS),
            },
            0.95,
        )
        .unwrap();
        assert!(*fe.std_errors().get(0, 0) > 1e-8);
    }

    #[test]
    fn fe_estimator_fit_cluster_rejects_degenerate_split_by_non_time_column() {
        // `time`以外の任意のクラスター列でも、全エンティティが2クラスターに1観測ずつなら退化。
        let groups = strings(&["g2", "g1", "g1", "g2", "g2", "g1", "g1", "g2"]);
        let result = FeEstimator::fit(
            four_by_two_input(&THREE_PERIOD_LABELS),
            FeEffects::OneWay,
            FeCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
            0.95,
        );
        assert_eq!(result.unwrap_err(), PanelError::DegenerateClusterTwoGroups);
    }

    #[test]
    fn fe_estimator_fit_cluster_two_group_split_without_regressors_is_accepted() {
        // `k=0`は標準誤差・F検定を計算しないため、Clusterでも退化を問題にしない。
        let (input, time) = two_period_input(0, false);
        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&time)),
            },
            0.95,
        );
        assert!(result.is_ok());
    }

    /// エンティティ2つ×5時点・`k=1`。2-wayではtime方向の各水準（時点）が2観測になる。
    fn two_entity_input() -> FeInput {
        let entity = strings(&["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]);
        let time = strings(&["1", "2", "3", "4", "5", "1", "2", "3", "4", "5"]);
        let x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 2.0, 1.0, 4.0, 3.0, 6.0];
        let y = [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0, 5.0, 9.5];
        FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap()
    }

    #[test]
    fn fe_estimator_fit_two_way_cluster_by_entity_rejects_two_entity_panel() {
        // 2-wayのwithin変換は各時点内でも和をゼロにするため、エンティティ2つのパネルを
        // entityでクラスタリング（`groups: None`＝既定）すると、全時点が2クラスターに
        // 1観測ずつになりクラスタースコアが恒等的にゼロになる（time方向の退化）。
        let result = FeEstimator::fit(
            two_entity_input(),
            FeEffects::TwoWay,
            FeCovType::Cluster { groups: None },
            0.95,
        );
        assert_eq!(result.unwrap_err(), PanelError::DegenerateClusterTwoGroups);
    }

    #[test]
    fn fe_estimator_fit_two_way_hac_rejects_two_entity_split_override() {
        // 同じtime方向の退化は、`Dk.time`にentityと同じ分け方の2水準列を渡した場合にも起きる。
        let result = FeEstimator::fit(
            two_entity_input(),
            FeEffects::TwoWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: lex_time(&["p", "p", "p", "p", "p", "q", "q", "q", "q", "q"]),
            },
            0.95,
        );
        assert_eq!(result.unwrap_err(), PanelError::DegenerateDkTwoPeriods);
    }

    #[test]
    fn fe_estimator_fit_one_way_cluster_by_entity_accepts_two_entity_panel() {
        // 1-wayはtime方向に和をゼロにしないため、同じデータのentityクラスタリングは退化しない。
        let fe = FeEstimator::fit(
            two_entity_input(),
            FeEffects::OneWay,
            FeCovType::Cluster { groups: None },
            0.95,
        )
        .unwrap();
        assert!(*fe.std_errors().get(0, 0) > 1e-8);
    }

    #[test]
    fn fe_estimator_fit_hac_rejects_time_override_with_wrong_length() {
        // `FeCovType::Dk.time`の上書き列は`FeInput`の検証を通らないため、`fit()`が長さを
        // 検証する（Clusterの`groups`と同じ水準）。
        let (entity, time, x, y) = fixest_reference_input();
        let n = y.len();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();
        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: TimeKeys::lexicographic(time[..n - 1].to_vec()),
            },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                y_rows: n,
                other_rows: n - 1,
            }
        );
    }

    #[test]
    fn fe_estimator_fit_hac_rejects_bandwidth_out_of_range() {
        // n_periods=3のため`bandwidth`の許容範囲は`[0, 3)`。`bandwidth=3`（`t`自体）は
        // 範囲外（OLSの`hac_lags`と同型の`[0, n)`境界、`t`版）。
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

        let dk_time = dk_time_from(&input);
        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(3),
                time: dk_time,
            },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::InvalidDkBandwidth { bandwidth: 3, t: 3 }
        );
    }

    #[test]
    fn fe_estimator_fit_hac_rejects_negative_bandwidth() {
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

        let dk_time = dk_time_from(&input);
        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(-1),
                time: dk_time,
            },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::InvalidDkBandwidth {
                bandwidth: -1,
                t: 3
            }
        );
    }

    #[test]
    fn fe_estimator_fit_with_no_regressors_estimates_fixed_effects_only_model() {
        // k=0（回帰変数なし、固定効果のみのモデル）でも`fit()`本体がエンドツーエンドに
        // 動くことを確認する境界ケース（`from_columns_with_no_regressors_succeeds`は
        // `FeInput`構築のみの検証で、`fit()`のdf調整（`df_model=neffects`のみになる）・
        // パネル固有R²/`aic`/`bic`計算までは通さない）。
        // n=6・n_entities=3・k=0でdf_model=neffects=3・df_resid=3。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let y = [1.0, 3.0, 5.0, 7.0, 2.0, 4.0];
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert_eq!(fe.df_model(), 3);
        assert_eq!(fe.df_resid(), 3);
        assert_eq!(fe.params().nrows(), 0);
        assert_eq!(fe.std_errors().nrows(), 0);
        assert!(fe.aic().is_finite());
        assert!(fe.bic().is_finite());
        assert!(fe.r_squared_within().is_finite());
        assert!(fe.r_squared_between().is_finite());
        assert!(fe.r_squared_overall().is_finite());
    }

    #[test]
    fn fe_estimator_fit_pins_faer_global_parallelism_to_seq() {
        // `fit()`冒頭の`crate::shared::parallelism::ensure_serial()`がfaerのグローバル
        // 並列度を`Par::Seq`へ引き戻すことの回帰ガード（panel系統代表、
        // `engine/src/panel/CLAUDE.md`「faerのグローバル並列度」参照）。
        faer::set_global_parallelism(faer::Par::rayon(0));

        let entity = strings(&["a", "a", "b", "b"]);
        let y = [1.0, 2.2, 3.0, 4.6];
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let _ = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

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
        use crate::linear::ols::{CovType, OlsEstimator};
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

        /// `fit`レベルのproperty-basedテスト用のランダムパネル。1-way（アンバランスを含む）と
        /// 2-way（バランスのみ、`fe-spec.md`3.1節）を`effects`で切り替える。行は
        /// entity外側・time内側の順で並ぶ（並べ替えは`keys`で別途行う）。
        #[derive(Debug, Clone)]
        struct FeCase {
            effects: FeEffects,
            entity_idx: Vec<usize>,
            time_idx: Vec<usize>,
            y: Vec<f64>,
            x: Vec<Vec<f64>>,
            /// 行の並べ替え用の乱数キー（長さ`n`）。
            keys: Vec<u64>,
            /// entity/timeごとの加法シフト（`y`に足して結果が不変であることの検証用）。
            entity_shift: Vec<f64>,
            time_shift: Vec<f64>,
        }

        const MAX_PERIODS: usize = 6;

        fn fe_case_strategy() -> impl Strategy<Value = FeCase> {
            (any::<bool>(), 4..=6usize, 3..=MAX_PERIODS, 1..=2usize)
                .prop_flat_map(|(two_way, n_entities, n_periods, k)| {
                    let sizes = if two_way {
                        Just(vec![n_periods; n_entities]).boxed()
                    } else {
                        collection::vec(2..=MAX_PERIODS, n_entities).boxed()
                    };
                    (Just(two_way), Just(k), sizes)
                })
                .prop_flat_map(|(two_way, k, sizes)| {
                    let n: usize = sizes.iter().sum();
                    let n_entities = sizes.len();
                    (
                        Just(two_way),
                        Just(sizes),
                        collection::vec(collection::vec(-10.0f64..10.0, n), k),
                        collection::vec(-5.0f64..5.0, n),
                        collection::vec(any::<u64>(), n),
                        collection::vec(-50.0f64..50.0, n_entities),
                        collection::vec(-50.0f64..50.0, MAX_PERIODS),
                    )
                })
                .prop_map(
                    |(two_way, sizes, x, noise, keys, entity_shift, time_shift)| {
                        let mut entity_idx = Vec::new();
                        let mut time_idx = Vec::new();
                        for (i, &size) in sizes.iter().enumerate() {
                            for t in 0..size {
                                entity_idx.push(i);
                                time_idx.push(t);
                            }
                        }
                        let y: Vec<f64> = (0..noise.len())
                            .map(|r| x.iter().map(|c| c[r]).sum::<f64>() + noise[r])
                            .collect();
                        FeCase {
                            effects: if two_way {
                                FeEffects::TwoWay
                            } else {
                                FeEffects::OneWay
                            },
                            entity_idx,
                            time_idx,
                            y,
                            x,
                            keys,
                            entity_shift,
                            time_shift,
                        }
                    },
                )
        }

        fn fe_cov_strategy() -> impl Strategy<Value = FeCovType> {
            prop_oneof![
                Just(FeCovType::Classical),
                Just(FeCovType::Hc1),
                Just(FeCovType::Hc2),
                Just(FeCovType::Hc3),
                Just(FeCovType::Cluster { groups: None }),
            ]
        }

        fn labels(prefix: &str, idx: &[usize]) -> Vec<String> {
            idx.iter().map(|i| format!("{prefix}{i}")).collect()
        }

        /// 傾き係数と標準誤差（`x`の列順）。推定に失敗したら`None`。
        fn fit_slopes(
            case: &FeCase,
            y: &[f64],
            x: &[Vec<f64>],
            entity: &[String],
            time: &[String],
            cov: FeCovType,
        ) -> Option<(Vec<f64>, Vec<f64>)> {
            let names: Vec<String> = (0..x.len()).map(|j| format!("x{j}")).collect();
            let input =
                FeInput::from_columns(y, x, names, entity, Some(time), "y".to_string()).ok()?;
            let est = FeEstimator::fit(input, case.effects, cov, 0.95).ok()?;
            let k = x.len();
            let params = (0..k).map(|j| *est.params().get(j, 0)).collect();
            let se = (0..k).map(|j| *est.std_errors().get(j, 0)).collect();
            Some((params, se))
        }

        fn fit_case(case: &FeCase, cov: FeCovType) -> Option<(Vec<f64>, Vec<f64>)> {
            fit_slopes(
                case,
                &case.y,
                &case.x,
                &labels("e", &case.entity_idx),
                &labels("t", &case.time_idx),
                cov,
            )
        }

        /// 固定効果推定の数値誤差（within変換・QR）を考慮し、固定フィクスチャ比較より緩めた
        /// 相対誤差（`ols/estimator.rs`のproptestと同じ方針）。
        fn assert_approx_eq(actual: f64, expected: f64, msg: &str) {
            let tol = 1e-6 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tol,
                "{msg}: actual={actual}, expected={expected}, tol={tol}"
            );
        }

        fn assert_all_approx_eq(actual: &[f64], expected: &[f64], msg: &str) {
            assert_eq!(actual.len(), expected.len());
            for (j, (a, e)) in actual.iter().zip(expected).enumerate() {
                assert_approx_eq(*a, *e, &format!("{msg}[{j}]"));
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            /// yにentity定数（2-wayはtime定数も）を加えても、傾き・SEは変わらない
            /// （固定効果が吸収するため）。
            #[test]
            fn slopes_and_se_are_invariant_to_additive_fixed_effects_in_y(
                case in fe_case_strategy(),
                cov in fe_cov_strategy(),
            ) {
                let base = fit_case(&case, cov.clone());
                prop_assume!(base.is_some());
                let (params, se) = base.unwrap();

                let y_shifted: Vec<f64> = (0..case.y.len())
                    .map(|r| {
                        let time_part = if case.effects == FeEffects::TwoWay {
                            case.time_shift[case.time_idx[r]]
                        } else {
                            0.0
                        };
                        case.y[r] + case.entity_shift[case.entity_idx[r]] + time_part
                    })
                    .collect();
                let shifted = fit_slopes(
                    &case,
                    &y_shifted,
                    &case.x,
                    &labels("e", &case.entity_idx),
                    &labels("t", &case.time_idx),
                    cov,
                );
                prop_assume!(shifted.is_some());
                let (params2, se2) = shifted.unwrap();

                assert_all_approx_eq(&params2, &params, "params");
                assert_all_approx_eq(&se2, &se, "se");
            }

            /// 行の並べ替えとentityラベルの付け替え（順序を反転した別名）で結果は変わらない。
            #[test]
            fn results_are_invariant_to_row_order_and_entity_relabeling(
                case in fe_case_strategy(),
                cov in fe_cov_strategy(),
            ) {
                let base = fit_case(&case, cov.clone());
                prop_assume!(base.is_some());
                let (params, se) = base.unwrap();

                let n = case.y.len();
                let n_entities = case.entity_shift.len();
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by_key(|&r| case.keys[r]);
                let y: Vec<f64> = order.iter().map(|&r| case.y[r]).collect();
                let x: Vec<Vec<f64>> = case
                    .x
                    .iter()
                    .map(|c| order.iter().map(|&r| c[r]).collect())
                    .collect();
                let entity: Vec<String> = order
                    .iter()
                    .map(|&r| format!("z{}", n_entities - 1 - case.entity_idx[r]))
                    .collect();
                let time: Vec<String> = order
                    .iter()
                    .map(|&r| format!("t{}", case.time_idx[r]))
                    .collect();

                let permuted = fit_slopes(&case, &y, &x, &entity, &time, cov);
                prop_assume!(permuted.is_some());
                let (params2, se2) = permuted.unwrap();

                assert_all_approx_eq(&params2, &params, "params");
                assert_all_approx_eq(&se2, &se, "se");
            }

            /// yをc倍すると傾き・標準誤差はそれぞれc倍・|c|倍になる。
            #[test]
            fn slopes_and_se_scale_with_y(
                case in fe_case_strategy(),
                cov in fe_cov_strategy(),
                c in prop_oneof![-10.0f64..-0.1, 0.1f64..10.0],
            ) {
                let base = fit_case(&case, cov.clone());
                prop_assume!(base.is_some());
                let (params, se) = base.unwrap();

                let y_scaled: Vec<f64> = case.y.iter().map(|v| v * c).collect();
                let scaled = fit_slopes(
                    &case,
                    &y_scaled,
                    &case.x,
                    &labels("e", &case.entity_idx),
                    &labels("t", &case.time_idx),
                    cov,
                );
                prop_assume!(scaled.is_some());
                let (params2, se2) = scaled.unwrap();

                let expected_params: Vec<f64> = params.iter().map(|p| p * c).collect();
                let expected_se: Vec<f64> = se.iter().map(|s| s * c.abs()).collect();
                assert_all_approx_eq(&params2, &expected_params, "params");
                assert_all_approx_eq(&se2, &expected_se, "se");
            }

            /// LSDV（entityダミー、2-wayはtimeダミーも加えた定数項付きOLS）と傾き・Classical SEが
            /// 一致する。LSDV側のOLS自由度`n-(k+N+T-1)`はFEのパネル自由度調整（`neffects`）と
            /// 一致するため、SEも同じ値になる。
            #[test]
            fn slopes_and_classical_se_match_lsdv_oracle(case in fe_case_strategy()) {
                let fe = fit_case(&case, FeCovType::Classical);
                prop_assume!(fe.is_some());
                let (params, se) = fe.unwrap();

                let k = case.x.len();
                let n = case.y.len();
                let n_entities = case.entity_shift.len();
                let mut x = case.x.clone();
                for i in 1..n_entities {
                    x.push((0..n).map(|r| f64::from(case.entity_idx[r] == i)).collect());
                }
                if case.effects == FeEffects::TwoWay {
                    let n_periods = *case.time_idx.iter().max().unwrap() + 1;
                    for t in 1..n_periods {
                        x.push((0..n).map(|r| f64::from(case.time_idx[r] == t)).collect());
                    }
                }
                let names: Vec<String> = (0..x.len()).map(|j| format!("x{j}")).collect();
                let input = OlsInput::from_columns(&case.y, &x, names, true, "y".to_string()).unwrap();
                let ols = OlsEstimator::fit(input, CovType::Classical, 0.95);
                prop_assume!(ols.is_ok());
                let ols = ols.unwrap();

                // 定数項が先頭に来るため、傾きは添字1..=k。
                let ols_params: Vec<f64> = (1..=k).map(|j| *ols.params().get(j, 0)).collect();
                let ols_se: Vec<f64> = (1..=k).map(|j| *ols.std_errors().get(j, 0)).collect();
                assert_all_approx_eq(&params, &ols_params, "params");
                assert_all_approx_eq(&se, &ols_se, "se");
            }
        }
    }

    // ── 時点の順序（`TimeKeys`） ───────────────────────────────────────────

    /// 4エンティティ×12時点の疑似データ（時点共通の自己相関ショックを持つ）。戻り値は
    /// `(y, x, entity, period)`で、`period`は各行の時点番号`0..12`。
    fn dk_panel_12() -> (Vec<f64>, Vec<f64>, Vec<String>, Vec<usize>) {
        let (n_entities, n_periods) = (4usize, 12usize);
        let mut shock = vec![0.0; n_periods];
        for t in 1..n_periods {
            shock[t] = 0.8 * shock[t - 1] + (((t * 37) % 11) as f64 - 5.0) / 5.0;
        }
        let (mut y, mut x, mut entity, mut period) = (vec![], vec![], vec![], vec![]);
        for e in 0..n_entities {
            for (t, &shock_t) in shock.iter().enumerate() {
                let xi = ((e * 7 + t * 13) % 17) as f64 / 3.0;
                let noise = ((e * 5 + t * 3) % 7) as f64 / 7.0;
                y.push(1.0 + 0.5 * xi + shock_t + noise);
                x.push(xi);
                entity.push(format!("e{e}"));
                period.push(t);
            }
        }
        (y, x, entity, period)
    }

    /// `dk_panel_12`の時点番号を`labels`で文字列にした`TimeKeys`（整数の値の順序）。
    fn integer_time_keys(period: &[usize], labels: impl Fn(usize) -> String) -> TimeKeys {
        let ids = period.iter().map(|&t| labels(t)).collect();
        let values: Vec<i128> = period.iter().map(|&t| t as i128).collect();
        TimeKeys::by_integer(ids, &values).unwrap()
    }

    fn dk_std_error(time: TimeKeys, rows: Option<&[usize]>) -> f64 {
        let (y, x, entity, period) = dk_panel_12();
        let rows: Vec<usize> = rows.map_or_else(|| (0..y.len()).collect(), |r| r.to_vec());
        let pick = |v: &[f64]| rows.iter().map(|&i| v[i]).collect::<Vec<_>>();
        let entity: Vec<String> = rows.iter().map(|&i| entity[i].clone()).collect();
        let _ = period;
        let input = FeInput::from_columns(
            &pick(&y),
            &[pick(&x)],
            vec!["x".into()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();
        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(3),
                time,
            },
            0.95,
        )
        .unwrap();
        *fe.std_errors().get(0, 0)
    }

    #[test]
    fn dk_standard_errors_follow_the_value_order_of_integer_periods() {
        let (_, _, _, period) = dk_panel_12();
        // 辞書順が時間順と一致するゼロ埋めラベル（基準）。
        let padded = integer_time_keys(&period, |t| format!("{t:03}"));
        // ゼロ埋めなしの整数ラベルを、数値順で並べる。
        let numeric = integer_time_keys(&period, |t| t.to_string());
        // ゼロ埋めなしのラベルを辞書順で並べる（`1, 10, 11, 2, ...`、誤った順序）。
        let lexicographic = TimeKeys::lexicographic(period.iter().map(|t| t.to_string()).collect());

        let expected = dk_std_error(padded, None);
        let got = dk_std_error(numeric, None);
        let wrong = dk_std_error(lexicographic, None);

        assert!(
            (got - expected).abs() < 1e-12,
            "numeric {got} vs padded {expected}"
        );
        assert!(
            (wrong - expected).abs() > 1e-6,
            "the lexicographic order of unpadded labels must differ: {wrong} vs {expected}"
        );
    }

    #[test]
    fn dk_standard_errors_do_not_depend_on_row_order() {
        let (_, _, _, period) = dk_panel_12();
        let n = period.len();
        let reversed: Vec<usize> = (0..n).rev().collect();
        let shuffled: Vec<usize> = (0..n).map(|i| (i * 7 + 3) % n).collect();
        assert_eq!(
            {
                let mut s = shuffled.clone();
                s.sort();
                s
            },
            (0..n).collect::<Vec<_>>(),
            "the shuffle must be a permutation"
        );
        let keys = |rows: &[usize]| {
            let p: Vec<usize> = rows.iter().map(|&i| period[i]).collect();
            integer_time_keys(&p, |t| t.to_string())
        };

        let base = dk_std_error(keys(&(0..n).collect::<Vec<_>>()), None);
        for rows in [&reversed, &shuffled] {
            let got = dk_std_error(keys(rows), Some(rows));
            assert!((got - base).abs() < 1e-12, "{got} vs {base}");
        }
    }

    #[test]
    fn fixed_effects_two_way_orders_periods_by_value_and_uses_the_first_as_reference() {
        let (y, x, entity, period) = dk_panel_12();
        let time = integer_time_keys(&period, |t| t.to_string());
        let input = FeInput::from_columns_ordered(
            &y,
            &[x],
            vec!["x".into()],
            &entity,
            Some(time),
            "y".into(),
        )
        .unwrap();
        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();

        let FixedEffects::TwoWay { time, .. } = fe.fixed_effects() else {
            panic!("2-way FE must return FixedEffects::TwoWay");
        };
        let labels: Vec<&str> = time.iter().map(|(label, _)| label.as_str()).collect();
        assert_eq!(
            labels,
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
        );
        assert_eq!(time[0].1, 0.0, "the first period is the reference");
    }

    #[test]
    fn fixed_effects_two_way_values_do_not_depend_on_the_labels_of_the_same_order() {
        let (y, x, entity, period) = dk_panel_12();
        let run = |labels: &dyn Fn(usize) -> String| {
            let input = FeInput::from_columns_ordered(
                &y,
                std::slice::from_ref(&x),
                vec!["x".into()],
                &entity,
                Some(integer_time_keys(&period, labels)),
                "y".into(),
            )
            .unwrap();
            let fe =
                FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();
            let FixedEffects::TwoWay { entity, time } = fe.fixed_effects() else {
                panic!("2-way FE must return FixedEffects::TwoWay");
            };
            (entity, time.into_iter().map(|(_, v)| v).collect::<Vec<_>>())
        };

        let (entity_a, time_a) = run(&|t| t.to_string());
        let (entity_b, time_b) = run(&|t| format!("{t:03}"));

        assert_eq!(time_a.len(), time_b.len());
        for (a, b) in time_a.iter().zip(&time_b) {
            assert!((a - b).abs() < 1e-12);
        }
        for (id, a) in &entity_a {
            assert!((a - entity_b[id]).abs() < 1e-12);
        }
    }

    /// 1-way・2-wayの全`cov_type`で、`cov_params()`が対称で、対角の平方根が`std_errors()`と
    /// 一致する。次元は`(k, k)`（固定効果の係数は含まない）。
    #[test]
    fn cov_params_is_symmetric_and_diagonal_matches_std_errors_for_every_cov_type() {
        let entity: Vec<String> = (0..24).map(|i| format!("e{}", i / 4)).collect();
        let time: Vec<String> = (0..24).map(|i| format!("t{}", i % 4)).collect();
        let x1: Vec<f64> = (0..24)
            .map(|i| ((i as f64) * 0.7).sin() * 4.0 + i as f64)
            .collect();
        let x2: Vec<f64> = (0..24).map(|i| ((i as f64) * 1.3).cos() * 3.0).collect();
        let y: Vec<f64> = (0..24)
            .map(|i| {
                2.0 + 1.5 * x1[i] - 0.8 * x2[i] + ((i as f64) * 2.1).sin() * 1.7 + (i / 4) as f64
            })
            .collect();

        for effects in [FeEffects::OneWay, FeEffects::TwoWay] {
            let cov_types = [
                FeCovType::Classical,
                FeCovType::Hc1,
                FeCovType::Hc2,
                FeCovType::Hc3,
                FeCovType::Cluster { groups: None },
                FeCovType::Dk {
                    bandwidth: Some(1),
                    time: TimeKeys::lexicographic(time.clone()),
                },
            ];
            for cov_type in cov_types {
                let name = format!("{effects:?}/{cov_type:?}");
                let input = FeInput::from_columns(
                    &y,
                    &[x1.clone(), x2.clone()],
                    vec!["x1".to_string(), "x2".to_string()],
                    &entity,
                    Some(&time),
                    "y".to_string(),
                )
                .unwrap();
                let fe = FeEstimator::fit(input, effects, cov_type, 0.95).unwrap();

                let cov = fe.cov_params();
                assert_eq!((cov.nrows(), cov.ncols()), (2, 2), "{name}: shape");
                for i in 0..2 {
                    for j in 0..2 {
                        let scale = (*cov.get(i, i) * *cov.get(j, j)).sqrt();
                        assert!(
                            (*cov.get(i, j) - *cov.get(j, i)).abs() <= 1e-12 * scale,
                            "{name}: cov_params not symmetric at ({i},{j})"
                        );
                    }
                    let se = *fe.std_errors().get(i, 0);
                    assert!(
                        (cov.get(i, i).sqrt() - se).abs() <= 1e-12 * se,
                        "{name}: sqrt(cov_params[{i}][{i}]) != std_errors[{i}]"
                    );
                }
            }
        }
    }
}
