//! 固定効果つきGMM（FE-GMM）の推定量`FeGmmEstimator`の点推定。
//!
//! 固定効果を除いた（within変換した）データへのGMMで、構造式
//! `y = X_exog β_exog + X_endog β_endog + 固定効果 + ε`を推定する。モーメント条件は
//! `E[z̃_it ε̃_it] = 0`（`Z̃ = [x̃_exog, z̃]`、`z̃`は変換後の操作変数）。入力（`FeIvInput`）・固定効果の方向
//! （`FeEffects`）・within変換は`FeIvEstimator`と共有する（`prepare.rs`）。定数項は持たない。
//!
//! # 範囲
//!
//! 点推定（係数・構造残差・点推定に使った重み行列`S`・反復回数・収束）、`cov_type`別の標準誤差・
//! z統計量・p値・信頼区間・傾き係数の同時Wald χ²検定、R²（within・between・overall）。過剰識別検定・
//! 内生性検定は持たない。
//!
//! # `fit`のパイプライン
//!
//! 1. `gmm_type`を共有の`GmmType`に直し、`max_iter`・`tol`の範囲を検証する（`GmmType::validate`）。
//! 2. `FeIvEstimator`と同じ前段（`prepare_within`）: 操作変数の数・singleton・定数列の検査、全列の
//!    within変換（1回だけ）、自由度の検証（`n > k + neffects`、`n > l + neffects`）、変換後に分散が
//!    残らない列の検査。
//! 3. 重み行列の検証（`weight_type`が`Cluster`のとき、クラスター数`G`と操作変数の数`l`の関係。
//!    `iv::gmm`と同じ`shared::gmm_weight::validate_weight_type`）。重みは`FeWeightType`（`Classical`・`Robust`・`Cluster`）で、
//!    行単位のNewey-Westの重み（`Hac`）は型に無い。
//! 4. 初期推定`W₀ = (Z̃'Z̃)⁻¹`（2SLSと同じ点推定）。構造残差が丸め誤差の範囲で0なら完全適合。
//! 5. 方式（`OneStep`・`TwoStep`・`Iterated`）に応じて「直前の残差から`S`を作り直して再推定」を繰り返す
//!    （`shared::gmm_iteration::gmm_iterate`。`iv::gmm`と同じコア）。
//! 6. 標準誤差: `cov_type`に応じたモーメント条件の分散`Ω̂`（`panel_moment_covariance`）を作り、サンドイッチ
//!    `B⁻¹(X̃'Z̃WΩ̂WZ̃'X̃)B⁻¹`（`B = X̃'Z̃WZ̃'X̃`、`W = S_used⁻¹`は点推定に使った重み。
//!    `shared::gmm::gmm_sandwich_covariance`）を求める。`cov_type`と重みは独立の選択で、一般形を常に使う。
//!    正規分布の推論表と、傾き係数の同時Wald χ²検定（自由度`k`）を計算する。
//! 7. R²: 構造残差ベース。within = `1 − e'e/ỹ'ỹ`、between・overallは`β̂`と実際の`X`で`FeEstimator`と同じ式。
//!
//! # `cov_type`と自由度
//!
//! `cov_type`は`FeIvEstimator`と同じ`FeCovType`（Classical・HC1〜HC3・Cluster・Dk）。`Ω̂`の小標本補正も
//! FE-2SLSと同じ（`df_resid = n − k − neffects`、クラスターの`K`は`fe_cluster_k_correction`、
//! DKは`T/(T−1)`と`K = df_model`）だが、`Ω̂`は変換後の操作変数`Z̃`のスコアに掛ける。HC2・HC3のレバレッジは
//! `Z̃`の素のレバレッジに吸収した効果のレバレッジを足したもの（ダミーを操作変数に入れたときのハット行列の
//! 対角と一致する）。推論は正規分布・χ²（`iv::gmm`と同じ。FE-2SLSのt・Fとは違う）で、クラスター数・時点数が
//! `k`以下ならWald検定が定義できないので`FeIvEstimator`と同じ検証で弾く。
//! `OneStep`と`Classical`重みの2段階・反復は2SLSと同じ係数で、`Classical`・`Hc1`・`Cluster`・`Dk`の共分散も
//! `FeIvEstimator`と一致する（`Hc2`・`Hc3`は、`X̂`と`Z̃`でレバレッジが違うので一致しない）。
//!
//! # `Robust`重みの限界
//!
//! within変換した誤差`ε̃_it`は個体内で相関し、`E[ε̃_it²]`が`σ_it²`だけでなく同じ個体の他の時点の
//! `σ_is²`にも依存する。`Robust`の`S = Σ ẽ²z̃z̃'`はこれを無視するので、不均一分散があり`T`が小さいと
//! 偏る。係数の一致性は保たれるが、効率性とHansen Jは影響を受けうる（`Classical`の`S`は
//! `Σ_t z̃ε̃ = Σ_t z̃ε`なので、この問題を持たない）。固定効果つきの既定の重みが個体でのクラスター
//! なのはこのため。`Robust`は小`T`では勧めない。
//!
//! # `s_used`の尺度
//!
//! `S`は`n`で割らない生の和（`iv::gmm`と同じ）。`Classical`と`OneStep`の`S`は`σ̂²・Z̃'Z̃`、`σ̂² = e'e/n`で、
//! `n`は観測数（吸収した効果の数は引かない）。within変換後の残差は`σ²(n − neffects)`程度の大きさしか
//! 持たないので、Hansen J（Sarganに相当）に使うときは分母を`n − neffects`に直す必要がある。
//!
//! # ファイル構成
//!
//! - `gmm/fit.rs`: `fit_fe_gmm`（上のパイプライン）
//! - `gmm/gmm_type.rs`: `FeGmmType`・`FeWeightType`（推定方式と重みの種類。`Hac`を持たない）

use faer::Mat;

use super::input::FeIvInput;
use crate::panel::common::PanelError;
use crate::panel::fe::{FeAbsorption, FeCovType, FeEffects};

mod fit;
mod gmm_type;

pub use gmm_type::{FeGmmType, FeWeightType};

/// 固定効果つきGMMの推定量。フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」）。
#[derive(Debug)]
pub struct FeGmmEstimator {
    input: FeIvInput,
    effects: FeEffects,
    gmm_type: FeGmmType,
    cov_type: FeCovType,
    /// 係数名（`x_exog`、`x_endog`の順）。
    param_names: Vec<String>,
    /// 構造式の係数`β̂`（k, 1）。
    params: Mat<f64>,
    /// 構造残差`ỹ − X̃β̂`（n, 1。within変換した空間）。
    residuals: Mat<f64>,
    /// 最終の係数に実際に使った重み行列`S`（l×l、`n`で割らない和）。
    s_used: Mat<f64>,
    n_iter: i64,
    converged: bool,
    df_model: usize,
    df_resid: usize,
    /// 吸収した効果の次元ごとの水準数（`effects`の順）。
    n_groups: Vec<usize>,
    dk_bandwidth_used: Option<usize>,
    std_errors: Mat<f64>,
    cov_params: Mat<f64>,
    test_stats: Mat<f64>,
    p_values: Mat<f64>,
    conf_lower: Mat<f64>,
    conf_upper: Mat<f64>,
    /// 傾き係数の同時Wald χ²検定の`(統計量, p値)`（自由度`k`）。`k = 0`なら`None`。
    wald: Option<(f64, f64)>,
    r_squared_within: f64,
    r_squared_between: f64,
    r_squared_overall: f64,
}

impl FeGmmEstimator {
    /// 固定効果つきGMMの点推定。契約はモジュールdocのパイプラインのとおり。
    ///
    /// `raise_on_non_convergence`が`true`で`gmm_type`が`Iterated`のとき、`max_iter`回以内に係数が収束
    /// しなければエラーにする（`false`なら`converged() == false`のまま返す）。`cov_type`は標準誤差の
    /// 種類（重みとは独立）、`confidence_level`は信頼区間の水準。
    ///
    /// # Errors
    /// - `Iterated`の`max_iter`・`tol`が範囲外なら`CommonError::InvalidGmmMaxIter`・`InvalidGmmTol`
    ///   （`PanelError::Common`）
    /// - 操作変数が内生変数より少なければ`PanelError::InsufficientInstruments`
    /// - 2-wayで`effects`が1つなら`PanelError::TwoWayRequiresTime`
    /// - singleton（`PanelError::SingletonGroup`）、生の値が全ての行で同じ列
    ///   （`PanelError::RegressorConstant`は役割つき、被説明変数は`CommonError::OutcomeConstant`）、
    ///   within変換後に分散がなくなる列（`PanelError::ZeroVarianceAfterDemeaning`は役割つき、被説明変数は
    ///   `PanelError::OutcomeConstantWithinEntity`）
    /// - 自由度が足りなければ`PanelError::InsufficientDegreesOfFreedom`（構造式を先に、次に操作変数）
    /// - `Cluster`の重みで、クラスター列の長さが観測数と違う・クラスター数が足りないとき
    ///   `CommonError::ClusterDimensionMismatch`・`InsufficientClusters`・
    ///   `InsufficientClustersForWeightMatrix`（過剰識別は`G < l`、丁度識別は`G <= l`）
    /// - 重み行列・`X̃'Z̃W Z̃'X̃`が数値的に特異なら`CommonError::NearSingularMatrix`
    /// - `confidence_level`が`(0, 1)`の範囲外なら`PanelError::WithinRegressionFailed`（`FeIvEstimator`と同じ）
    /// - `cov_type`の検証は`FeIvEstimator::fit`と同じ（クラスター数・時点数が`k`以下、2グループ退化、
    ///   DKのバンド幅など）。Wald χ²検定が数値的に定義できなければ`PanelError::FTestFailed`
    /// - 構造残差が丸め誤差の範囲で0なら`CommonError::PerfectFit`（完全適合。GMMに第二段階は無いので`stage`は付けない。`iv::gmm`と同じ）
    /// - `Iterated`が収束せず`raise_on_non_convergence`なら`CommonError::GmmNonConvergence`
    /// - 不均衡な2-wayと3次元以上で、交互射影が収束しなければ`PanelError::AbsorptionNotConverged`
    pub fn fit(
        input: FeIvInput,
        effects: FeEffects,
        gmm_type: FeGmmType,
        raise_on_non_convergence: bool,
        cov_type: FeCovType,
        confidence_level: f64,
    ) -> Result<Self, PanelError> {
        Self::fit_with_absorption(
            input,
            effects,
            gmm_type,
            raise_on_non_convergence,
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
        gmm_type: FeGmmType,
        raise_on_non_convergence: bool,
        cov_type: FeCovType,
        confidence_level: f64,
        absorption: &FeAbsorption,
    ) -> Result<Self, PanelError> {
        fit::fit_fe_gmm(
            input,
            effects,
            gmm_type,
            raise_on_non_convergence,
            cov_type,
            confidence_level,
            absorption,
        )
    }

    /// 入力データ（within変換前）。
    pub fn input(&self) -> &FeIvInput {
        &self.input
    }

    /// 推定に使った固定効果の方向。
    pub fn effects(&self) -> FeEffects {
        self.effects
    }

    /// 推定方式。
    pub fn gmm_type(&self) -> &FeGmmType {
        &self.gmm_type
    }

    /// 係数名（`x_exog`、`x_endog`の順）。
    pub fn param_names(&self) -> &[String] {
        &self.param_names
    }

    /// 構造式の係数`β̂` (k, 1)。
    pub fn params(&self) -> &Mat<f64> {
        &self.params
    }

    /// 構造残差`ỹ − X̃β̂` (n, 1)。within変換した空間の値（`FeIvEstimator::residuals`と同じ）。
    pub fn residuals(&self) -> &Mat<f64> {
        &self.residuals
    }

    /// 最終の係数に使った重み行列`S` (l×l)。`n`で割らない和で、尺度はモジュールdoc「`s_used`の尺度」のとおり。
    pub fn s_used(&self) -> &Mat<f64> {
        &self.s_used
    }

    /// 推定回数（`OneStep`は1、`TwoStep`は2、`Iterated`は実際の回数）。
    pub fn n_iter(&self) -> i64 {
        self.n_iter
    }

    /// 収束したか（`OneStep`・`TwoStep`は常に`true`）。
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// モデル自由度`k + neffects`。
    pub fn df_model(&self) -> usize {
        self.df_model
    }

    /// 残差自由度`n − df_model`。
    pub fn df_resid(&self) -> usize {
        self.df_resid
    }

    /// 吸収した効果の次元ごとの水準数（`effects`の順）。
    pub fn n_groups(&self) -> &[usize] {
        &self.n_groups
    }

    /// 観測数。
    pub fn nobs(&self) -> usize {
        self.input.nobs()
    }

    /// 推定に使った`cov_type`。
    pub fn cov_type(&self) -> &FeCovType {
        &self.cov_type
    }

    /// `Dk`のとき実際に使われたバンド幅。`Dk`以外では`None`。
    pub fn dk_bandwidth_used(&self) -> Option<usize> {
        self.dk_bandwidth_used
    }

    /// 標準誤差 (k, 1)。
    pub fn std_errors(&self) -> &Mat<f64> {
        &self.std_errors
    }

    /// 係数の分散共分散行列 (k, k)。
    pub fn cov_params(&self) -> &Mat<f64> {
        &self.cov_params
    }

    /// z統計量 (k, 1)。正規分布で検定する。
    pub fn test_stats(&self) -> &Mat<f64> {
        &self.test_stats
    }

    /// 両側p値 (k, 1)（正規分布）。
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

    /// 傾き係数の同時Wald χ²検定の統計量（自由度`k`）。係数が無ければ`None`。
    pub fn wald_statistic(&self) -> Option<f64> {
        self.wald.map(|(statistic, _)| statistic)
    }

    /// 傾き係数の同時Wald χ²検定のp値。係数が無ければ`None`。
    pub fn wald_p_value(&self) -> Option<f64> {
        self.wald.map(|(_, p_value)| p_value)
    }

    /// within R²（構造残差ベース。`1 − e'e/ỹ'ỹ`。負や1を超える値もありうる）。
    pub fn r_squared_within(&self) -> f64 {
        self.r_squared_within
    }

    /// between R²（`β̂`と実際の`X`で`FeEstimator`と同じ式）。
    pub fn r_squared_between(&self) -> f64 {
        self.r_squared_between
    }

    /// overall R²（`β̂`と実際の`X`で`FeEstimator`と同じ式）。
    pub fn r_squared_overall(&self) -> f64 {
        self.r_squared_overall
    }
}
