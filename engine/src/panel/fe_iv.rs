//! FEIV（固定効果IV、FE-2SLS）の推定量`FeIvEstimator`。
//!
//! 固定効果を除いた（within変換した）データへの2SLSで、構造式
//! `y = X_exog β_exog + X_endog β_endog + 固定効果 + ε`を推定する。定数項は持たない（固定効果が吸収する）。
//! 入力は`FeIvInput`（役割ごとの列と効果の各次元の索引）、固定効果の方向は`FeEffects`
//! （1-way・2-way・3次元以上）、`cov_type`は`FeCovType`（Classical・HC1〜HC3・Cluster・DK）で、FEと同じ型。
//!
//! # `FeIvEstimator::fit`のパイプライン
//!
//! 1. 操作変数の数の検査（`len(instruments) >= len(x_endog)`）、singleton・生の値が一定の列の検出。
//! 2. `y`・`x_exog`・`x_endog`・`instruments`の全列を**1回だけ**within変換する（`FeEstimator`と同じ
//!    変換。バランスな1-way・2-wayは閉形式、不均衡な2-wayと3次元以上は交互射影）。変換が残す誤差
//!    （`noise`）は全列で共有し、分散ゼロ・完全適合の判定に足す。
//! 3. 自由度の検証（構造式`n > k + neffects`を先に、次に第一段階`n > l + neffects`。`l = k_exog + k_instruments`）、
//!    変換後の分散ゼロの列（役割ごとに報告）。
//! 4. 第一段階: 内生変数ごとに、変換済みの列を`FeEstimator`の後半（`fit_from_within`）で推定する
//!    （`x̃_endog[j] ~ x̃_exog + 操作変数`）。`cov_type`は構造式と同じで、傾きのF検定はしない
//!    （クラスター数・時点数が`l`以下でも標準誤差は計算する。IV本体の`first_stage()`と同じ）。
//!    当てはめ値から`X̂ = [x̃_exog, x̂̃_endog]`を作る。
//! 5. 点推定`β̂ = (X̂'X̂)⁻¹X̂'ỹ`、構造残差`e = ỹ − X̃β̂`（`X̃`は実際の内生変数。第二段階の残差ではない）。
//! 6. `panel_cov_params`に`X̂`・`(X̂'X̂)⁻¹`・構造残差を渡して`cov_type`別の共分散を求め、t分布の推論表と
//!    傾き係数のWald F検定を計算する。
//!
//! # 自由度
//!
//! `df_model = k + neffects`、`df_resid = n − df_model`（`k = k_exog + k_endog`）。`neffects`は
//! `FeEstimator`と同じ`absorbed_parameters`（水準数の合計 − (効果の数 − 1)。fixestと同じ数え方で、連結成分の数や
//! ネストによるランク落ちは引かない）。クラスターの小標本補正の`K`は`fe_cluster_k_correction`
//! （fixestの`K.fixef="nonnested"`）、推論の自由度は`Cluster`が`G − 1`、`Dk`が`T − 1`、それ以外は`df_resid`。
//!
//! # 適合度
//!
//! R²は構造残差ベース: within = `1 − e'e / ỹ'ỹ`、between・overallは`β̂`と実際の`X`で`FeEstimator`と同じ式。
//! 負や1を超える値もありうる（IVのR²は意味が限定的）。対数尤度・AIC・BICは出さない。
//!
//! # 固定効果自体
//!
//! `fixed_effects()`は`FeEstimator`と同じ復元（`fixed_effects_of`）に、構造式の`y`・実際の`X`・`β̂`を渡す
//! （`α_i = ȳ_i − x̄_i'β̂`、`X`は実際の内生変数）。
//!
//! # ファイル構成
//!
//! - `input.rs`: `FeIvInput`・`RoleColumns`
//! - `fit.rs`: `fit_fe_iv`（上のパイプライン）
//!
//! 弱操作変数診断・Sargan・Wu-Hausmanは別途追加する（第一段階の推定量・残差を保持している）。

use faer::Mat;

use super::common::PanelError;
use super::fe::{FeAbsorption, FeCovType, FeEffects, FeEstimator, FeInput, FixedEffects};

mod fit;
mod input;

pub use input::{FeIvInput, RoleColumns};

/// FEIVの推定量。フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」）。
#[derive(Debug)]
pub struct FeIvEstimator {
    input: FeIvInput,
    effects: FeEffects,
    cov_type: FeCovType,
    /// 係数名（`x_exog`、`x_endog`の順）。
    param_names: Vec<String>,
    /// 構造式の係数`β̂`（k, 1）。
    params: Mat<f64>,
    /// 構造残差`ỹ − X̃β̂`（n, 1。`X̃`は実際の内生変数）。
    residuals: Mat<f64>,
    df_model: usize,
    df_resid: usize,
    /// t検定・信頼区間・F検定の自由度（`Cluster`は`G − 1`、`Dk`は`T − 1`、それ以外は`df_resid`）。
    df_inference: usize,
    dk_bandwidth_used: Option<usize>,
    /// 吸収した効果の次元ごとの水準数（`effects`の順）。
    n_groups: Vec<usize>,
    std_errors: Mat<f64>,
    cov_params: Mat<f64>,
    test_stats: Mat<f64>,
    p_values: Mat<f64>,
    conf_lower: Mat<f64>,
    conf_upper: Mat<f64>,
    r_squared_within: f64,
    r_squared_between: f64,
    r_squared_overall: f64,
    /// 傾き係数の同時Wald F検定の`(F統計量, p値)`。
    f_test: Option<(f64, f64)>,
    /// 内生変数ごとの第一段階（`x_endog`の順）。変換済みの列・残差（`residuals()`）を持つ。
    first_stages: Vec<FeEstimator>,
    /// 構造式の`y`・実際の`X = [x_exog, x_endog]`（変換前）。`fixed_effects()`が使う。
    structural: FeInput,
    /// 交互射影が`y`・`x_exog`・`x_endog`の列から引いた各次元の効果（`structural`の列と同じ順）。
    absorbed: Option<crate::panel::fe::AbsorbedEffects>,
}

impl FeIvEstimator {
    /// FEIVを推定する。契約はモジュールdocのパイプラインのとおり。
    ///
    /// # Errors
    /// - 操作変数が内生変数より少なければ`PanelError::InsufficientInstruments`
    /// - 2-wayで`effects`が1つなら`PanelError::TwoWayRequiresTime`
    /// - singleton（`PanelError::SingletonGroup`）、生の値が全ての行で同じ列
    ///   （`PanelError::RegressorConstant`は役割つき、被説明変数は`CommonError::OutcomeConstant`）、
    ///   within変換後に分散がなくなる列（`PanelError::ZeroVarianceAfterDemeaning`は役割つき、被説明変数は
    ///   `PanelError::OutcomeConstantWithinEntity`）
    /// - 自由度が足りなければ`PanelError::InsufficientDegreesOfFreedom`（構造式を先に、次に第一段階。
    ///   第一段階は`stage=FIRST_STAGE`つき）
    /// - 第一段階の推定の失敗は、原因の`code`に`stage=FIRST_STAGE`を付けて返す（クラスター数・時点数が
    ///   `l`以下でも、標準誤差が計算できれば失敗にしない）
    /// - 構造方程式の最小二乗・完全適合は`PanelError::SecondStageFailed`
    /// - `cov_type`別の共分散の検証は`FeEstimator::fit`と同じ（クラスター数・時点数が傾きの数`k`以下、
    ///   2グループ退化、DKのバンド幅など）
    /// - `confidence_level`が`(0, 1)`の範囲外なら`PanelError::WithinRegressionFailed`
    /// - 不均衡な2-wayと3次元以上で、交互射影が収束しなければ`PanelError::AbsorptionNotConverged`
    pub fn fit(
        input: FeIvInput,
        effects: FeEffects,
        cov_type: FeCovType,
        confidence_level: f64,
    ) -> Result<Self, PanelError> {
        Self::fit_with_absorption(
            input,
            effects,
            cov_type,
            confidence_level,
            &FeAbsorption::default(),
        )
    }

    /// [`Self::fit`]に、交互射影の打ち切り設定を渡す。
    ///
    /// # Errors
    /// [`Self::fit`]のエラーに加えて、`absorption`の値が範囲外なら`PanelError::InvalidTol`・
    /// `PanelError::InvalidMaxIter`。
    pub fn fit_with_absorption(
        input: FeIvInput,
        effects: FeEffects,
        cov_type: FeCovType,
        confidence_level: f64,
        absorption: &FeAbsorption,
    ) -> Result<Self, PanelError> {
        fit::fit_fe_iv(input, effects, cov_type, confidence_level, absorption)
    }

    /// 入力データ（within変換前）。
    pub fn input(&self) -> &FeIvInput {
        &self.input
    }

    /// 推定に使った固定効果の方向。
    pub fn effects(&self) -> FeEffects {
        self.effects
    }

    /// 推定に使った`cov_type`。
    pub fn cov_type(&self) -> &FeCovType {
        &self.cov_type
    }

    /// 係数名（`x_exog`、`x_endog`の順）。
    pub fn param_names(&self) -> &[String] {
        &self.param_names
    }

    /// 構造式の係数`β̂` (k, 1)。
    pub fn params(&self) -> &Mat<f64> {
        &self.params
    }

    /// 構造残差`ỹ − X̃β̂` (n, 1)。`X̃`は実際の内生変数で、第二段階の残差ではない。
    pub fn residuals(&self) -> &Mat<f64> {
        &self.residuals
    }

    /// モデル自由度`k + neffects`。
    pub fn df_model(&self) -> usize {
        self.df_model
    }

    /// 残差自由度`n − df_model`。
    pub fn df_resid(&self) -> usize {
        self.df_resid
    }

    /// t検定・信頼区間・F検定の自由度。
    pub fn df_inference(&self) -> usize {
        self.df_inference
    }

    /// `cov_type=Dk`のとき実際に使われたバンド幅。
    pub fn dk_bandwidth_used(&self) -> Option<usize> {
        self.dk_bandwidth_used
    }

    /// 吸収した効果の次元ごとの水準数（`effects`の順）。
    pub fn n_groups(&self) -> &[usize] {
        &self.n_groups
    }

    /// 標準誤差 (k, 1)。
    pub fn std_errors(&self) -> &Mat<f64> {
        &self.std_errors
    }

    /// `cov_type`別の係数の分散共分散行列 (k, k)。
    pub fn cov_params(&self) -> &Mat<f64> {
        &self.cov_params
    }

    /// t値 (k, 1)。
    pub fn test_stats(&self) -> &Mat<f64> {
        &self.test_stats
    }

    /// 両側p値 (k, 1)（t分布、自由度は`df_inference`）。
    pub fn p_values(&self) -> &Mat<f64> {
        &self.p_values
    }

    /// 信頼区間の下限 (k, 1)。
    pub fn conf_lower(&self) -> &Mat<f64> {
        &self.conf_lower
    }

    /// 信頼区間の上限 (k, 1)。
    pub fn conf_upper(&self) -> &Mat<f64> {
        &self.conf_upper
    }

    /// within R²（構造残差ベース: `1 − e'e / ỹ'ỹ`）。負や1を超える値もありうる。
    pub fn r_squared_within(&self) -> f64 {
        self.r_squared_within
    }

    /// between R²（個体平均に`β̂`を当てはめた構造残差ベース）。
    pub fn r_squared_between(&self) -> f64 {
        self.r_squared_between
    }

    /// overall R²（元の`y`・`X`に`β̂`だけを当てはめた残差ベース）。
    pub fn r_squared_overall(&self) -> f64 {
        self.r_squared_overall
    }

    /// 傾き係数`k`個が同時にゼロという帰無仮説のWald F検定統計量（`cov_type`反映済み）。
    pub fn f_statistic(&self) -> Option<f64> {
        self.f_test.map(|(statistic, _)| statistic)
    }

    /// `f_statistic()`のp値。
    pub fn f_p_value(&self) -> Option<f64> {
        self.f_test.map(|(_, p_value)| p_value)
    }

    /// `f_statistic()`の自由度`(分子, 分母)` = `(k, df_inference)`。
    pub fn f_df(&self) -> Option<(usize, usize)> {
        self.f_test
            .map(|_| (self.params.nrows(), self.df_inference))
    }

    /// 内生変数ごとの第一段階（`x_endog`の順）。各第一段階は`x̃_endog[j] ~ x̃_exog + 操作変数`の
    /// within回帰で、`cov_type`は構造式と同じ。傾きのF検定はしないので`f_statistic()`は`None`。
    pub fn first_stages(&self) -> &[FeEstimator] {
        &self.first_stages
    }

    /// 固定効果自体（α_i、2-wayはγ_tも）を事後的に復元する。構造式の`y`・実際の`X`・`β̂`から
    /// `FeEstimator::fixed_effects`と同じ規約で計算する（モジュールdoc「固定効果自体」参照）。
    pub fn fixed_effects(&self) -> FixedEffects {
        super::fe::fixed_effects_of(
            &self.structural,
            self.effects,
            &self.params,
            self.absorbed.as_ref(),
        )
    }
}
