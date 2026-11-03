//! FEの推定量（`FeEstimator`）。入力データ型（`FeInput`）・within変換・`fit`の本体は子モジュール`fe/`に置く。
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
//!
//! # ファイル構成
//!
//! このファイルは外から見える面（`FeEstimator`の構造体・getter・`fit`と`fixed_effects`の契約と
//! 1行委譲・再エクスポート）だけを持つ。ロジックは子モジュールに置き、テストは各ロジックと
//! 同じファイルの`mod tests`に置く（`fit`を通す統合的なテストと`fit`レベルのproperty-basedテスト
//! は`fit.rs`に集まる）。
//!
//! - `input.rs`: `FeInput`・`FeEffects`・`FeCovType`
//! - `within.rs`: within変換と事前検証（singleton・分散ゼロの説明変数）
//! - `fit.rs`: `fit`の本体（検証順序・最小二乗・`cov_type`別の共分散・統計量のオーケストレーション）
//! - `r_squared.rs`: between・overall R²
//! - `fixed_effects.rs`: `FixedEffects`と、固定効果自体の復元
//! - `test_support.rs`: 複数ファイルのテストが共有するフィクスチャ（`#[cfg(test)]`）

use faer::Mat;

use crate::linear::ols::OlsInput;
use crate::panel::common::PanelError;
use crate::shared::inference;

mod fit;
mod fixed_effects;
mod input;
mod r_squared;
#[cfg(test)]
mod test_support;
mod within;

pub use fixed_effects::FixedEffects;
pub use input::{FeCovType, FeEffects, FeInput};
// `re/`のテストだけが使う。
#[cfg(test)]
pub(crate) use within::{within_transform_one_way, within_transform_two_way};

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
        fit::fit_fe(input, effects, cov_type, confidence_level)
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
        fixed_effects::fixed_effects(self)
    }
}
