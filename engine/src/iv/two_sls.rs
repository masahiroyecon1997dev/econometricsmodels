//! 2SLS（二段階最小二乗法）の点推定・標準誤差・適合度統計量。
//!
//! ## 数式
//!
//! 第一段階: 内生変数ごとに`x_endog[j] ~ x_exog + instruments`をOLS推定し、
//! 予測値`x̂_endog[j]`を得る（`instruments`は除外操作変数のみ、`x_exog ++ instruments`が
//! 「全操作変数」、`docs/spec/iv-spec.md`1.1節）。
//!
//! 第二段階: `y ~ x_exog + x̂_endog`をOLS推定する。この係数が2SLS推定量
//! （`β̂_2SLS = (X'PzX)⁻¹X'Pzy`、`Pz`は全操作変数`Z`への射影行列）と数値的に一致する
//! （教科書的な2SLSの2段階回帰による構成）。
//!
//! ## 標準誤差・適合度統計量
//!
//! 第二段階の設計行列に`x̂_endog`（推定値）を使う都合上、**第二段階の`OlsEstimator`が
//! ナイーブに計算する標準誤差・t値・p値・信頼区間は2SLSとして正しくない**（教科書的に
//! 有名な罠。`X = [x_exog, x_endog]`を実際の（推定値ではない）内生変数、
//! `X̂ = Pz X = [x_exog, x̂_endog]`を第二段階の設計行列とすると、2SLSの正しい分散は
//! `(X'PzX)⁻¹X'Pz Ω Pz X(X'PzX)⁻¹ = (X̂'X̂)⁻¹X̂'ΩX̂(X̂'X̂)⁻¹`というサンドイッチ型で、
//! `Ω`の推定方法は`cov_type`により変わる。`docs/spec/iv-spec.md`3.1節）。
//!
//! 上式の`(X̂'X̂)⁻¹X̂'ΩX̂(X̂'X̂)⁻¹`は、**設計行列にX̂を使い、残差に構造残差
//! `e = y - Xβ̂`（実際のXを使う。第二段階回帰自身の残差`y - X̂β̂`ではない）を使えば、
//! OLSのclassical/HC0-3/cluster/hacサンドイッチ公式とまったく同じ形**になる
//! （`Pz`が射影行列＝冪等・対称であることから代数的に導出できる、Wooldridge (2010) 5章）。
//! そのため本モジュールは、第二段階の`OlsEstimator`（`params()`のみ利用、後述）とは別に、
//! この`(X̂, e)`に対する独立実装のサンドイッチ計算（`classical_cov_params`/`hc_cov_params`/
//! `hac_cov_params`/`wald_f_test`、いずれも`two_sls/cov.rs`。クラスターのサンドイッチは
//! 設計行列と残差を引数に取る`shared::covariance::cluster_cov_params`を、`X̂`・構造残差`e`で
//! 呼ぶ）を行う。
//! OLS（`engine::linear::ols`）の同名の同型計算とは**意図的に独立実装**にしている
//! （`docs/spec/iv-spec.md`3.1節「IVのサンドイッチ型分散計算は独自実装でよい……OLS/nonlinear
//! どちらの既存計算にも寄せない」）。例外はクラスターの蓄積・補正と、HACのラグ数の解決で、
//! 共有モジュール`shared::covariance`（OLS本体ではない）の`cluster_meat`/`cluster_correction`・
//! `resolve_hac_lags`を使う。
//!
//! 第一段階の各`OlsEstimator`はそれ自体が正しい（ナイーブな）OLS回帰であり、
//! （弱操作変数診断等で必要になる）SE・F統計量も含めてそのまま公開してよい
//! （`first_stage_estimators()`、`first_stage()`実装で利用済み）。
//!
//! ## `second_stage`フィールドの位置づけ（変わらず内部実装専用）
//!
//! `second_stage: OlsEstimator`は`SECOND_STAGE_COV_TYPE`（`two_sls/fit.rs`、`Classical`固定）で内部的に
//! フィットするが、これは**係数`β̂`と設計行列`X̂`（`input().x()`）を得るためだけ**に使う
//! （`Cluster`のようにクラスター列や十分なクラスター数を追加で要求せず、`β̂`計算に
//! 無関係な失敗経路を作らないため）。`second_stage`自身の`std_errors()`/`test_stats()`等
//! （ナイーブな第二段階OLSのSEで2SLSとして誤り）は`TwoSlsEstimator`の外部に公開しない。
//! 呼び出し元が指定した`cov_type`を反映した正しいSE・F統計量等は、`TwoSlsEstimator`
//! 自身のトップレベルフィールド（`std_errors()`/`test_stats()`/`wald_statistic()`等）として
//! 独立に計算・保持する。
//!
//! ## 第一段階・第二段階での`cov_type`/`confidence_level`の扱い
//!
//! 点推定（`params()`）の値は`cov_type`/`confidence_level`のどちらにも依存しない
//! （`cov_type`は分散の推定方法の選択、`confidence_level`は信頼区間の幅にのみ影響する）。
//! **第一段階・第二段階のいずれにも**、`fit()`の呼び出し元から渡された`cov_type`/
//! `confidence_level`をそのまま使う（第一段階は文字通り`OlsEstimator::fit`に渡す。
//! 第二段階は上記の独立実装のサンドイッチ計算に渡す。`second_stage`フィールド自身の
//! 内部委譲フィットだけは`SECOND_STAGE_COV_TYPE`固定のまま、前節参照）。
//! `confidence_level`の`(0, 1)`範囲チェックは、第二段階の内部委譲フィット
//! （`OlsEstimator::fit(second_stage_input, ...)`）に一度だけ委ねる（独立実装の
//! サンドイッチ計算側では検証しない。`x_endog=[]`の退化ケースでは第一段階ループが
//! 一度も回らないため、この検証が範囲外エラーを検知する唯一の経路になる）。
//!
//! ## ファイル構成
//!
//! このファイルは外から見える面（`TwoSlsEstimator`の構造体・getter・`IvEstimate`の実装と、
//! `fit`のエラー条件を含む契約）だけを持つ。ロジックは子モジュールに置く:
//! `fit.rs`（第一段階・第二段階・共分散・検定をつなぐ`fit`の本体）・`cov.rs`（cov_type別の
//! 共分散行列とWald F検定）・`wu_hausman.rs`・`sargan.rs`（診断検定）・`test_support.rs`
//! （複数ファイルのテストが共有するフィクスチャ）。テストは各ロジックと同じファイルの
//! `mod tests`に置く（`fit`を通すテストだけは`fit/tests/`にテーマ別で置く）。

use crate::iv::common::{IvError, IvEstimate, IvInput};
use crate::linear::ols::{CovType, OlsEstimator};
use crate::shared::inference;
use faer::Mat;

mod cov;
mod fit;
mod sargan;
#[cfg(test)]
mod test_support;
mod wu_hausman;

/// 2SLSの点推定結果。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug)]
pub struct TwoSlsEstimator {
    /// 内生変数ごとの第一段階回帰（`x_endog[j] ~ x_exog + instruments`）。
    /// タプルの`String`は内生変数名（`IvInput::x_endog_names`と対応）。
    first_stage: Vec<(String, OlsEstimator)>,
    /// 内生変数ごとの弱操作変数診断（部分F統計量、`docs/spec/iv-spec.md`3.4節）。
    /// `first_stage`と同じ順序・同じ内生変数名（`Vec<(String, f64)>`にしている理由も
    /// `first_stage`と同じ、`HashMap`にすると走査順序が非決定的になるため）。
    weak_instrument_f_statistics: Vec<(String, f64)>,
    /// 第二段階回帰（`y ~ x_exog + x̂_endog`）。`params()`/`param_names()`/`input()`（design
    /// 行列`X̂`）を得るためだけに使う内部実装専用フィールド。モジュール冒頭のdocコメント
    /// 「`second_stage`フィールドの位置づけ」参照。
    second_stage: OlsEstimator,
    /// 呼び出し元が指定した`cov_type`（第一段階・SE計算の両方に反映済み）。
    cov_type: CovType,
    /// 構造残差 `e = y - Xβ̂`（n, 1）。`X`は実際の内生変数を使う（第二段階回帰自身の残差
    /// `y - X̂β̂`とは異なる。モジュール冒頭のdocコメント参照）。
    residuals: Mat<f64>,
    /// 標準誤差 (k, 1)。`cov_type`に応じたサンドイッチ型分散の対角成分の平方根。
    std_errors: Mat<f64>,
    /// 係数の分散共分散行列 (k, k)。`std_errors`はこの対角成分の平方根。
    cov_params: Mat<f64>,
    /// t統計量 (k, 1) = params / std_errors（`docs/spec/iv-spec.md`3.2節、2SLSはt分布）。
    test_stats: Mat<f64>,
    /// 両側p値 (k, 1)。t分布（自由度`df_inference`）に基づく
    p_values: Mat<f64>,
    conf_lower: Mat<f64>,
    conf_upper: Mat<f64>,
    df_resid: usize,
    /// t検定・信頼区間・F検定に使った自由度。`cov_type=Cluster`のときだけ`G-1`
    /// （`fit()`のコメント参照）で、それ以外は`df_resid`と一致する。
    df_inference: usize,
    /// `cov_type=Hac`のとき、実際に使われたラグ数（`hac_lags`の明示指定値、または未指定時に
    /// 経験則で自動計算した値）。`CovType::Hac`の`lags`はユーザー指定値のまま変更しない
    /// ため別フィールドで保持する。`Hac`以外では`None`。
    hac_lags_used: Option<usize>,
    df_model: usize,
    r_squared: f64,
    adj_r_squared: f64,
    /// F統計量。`cov_type=Classical`なら古典的F検定、それ以外（HC0-3/HAC/cluster）は
    /// ロバストWald検定（OLSと同じ切り替えロジック、`docs/spec/iv-spec.md`2章）
    wald_statistic: f64,
    wald_p_value: f64,
    /// Wu-Hausman内生性検定（回帰ベース、`docs/spec/iv-spec.md`3.6節）の統計量。
    /// `x_endog=[]`（検定対象の内生変数が無い）、または拡張回帰が想定内の理由で推定不能
    /// （設計行列が特異・観測数不足、`fit()`のdocコメント参照）なら`None`。
    wu_hausman_statistic: Option<f64>,
    wu_hausman_p_value: Option<f64>,
    /// Wu-Hausman検定のF分布の自由度`(分子, 分母)`。`wu_hausman_statistic`と同じ条件で`None`。
    wu_hausman_df: Option<(usize, usize)>,
    /// Sargan過剰識別検定（`docs/spec/iv-spec.md`3.5節）の統計量。丁度識別
    /// （自由度`len(instruments) - len(x_endog)`が0）なら`None`。
    sargan_statistic: Option<f64>,
    sargan_p_value: Option<f64>,
    /// Sargan検定のχ²分布の自由度`len(instruments) - len(x_endog)`。丁度識別なら`None`。
    sargan_df: Option<usize>,
}

impl TwoSlsEstimator {
    /// `IvInput`から2SLSの点推定・標準誤差・適合度統計量を求める。
    ///
    /// `cov_type`/`confidence_level`は第一段階（`first_stage_estimators()`で公開する
    /// `OlsEstimator`）・第二段階（このメソッドが独立に計算するサンドイッチ型SE）の
    /// どちらにも反映される（モジュール冒頭のdocコメント「第一段階・第二段階での
    /// `cov_type`/`confidence_level`の扱い」参照。`second_stage`フィールド自身の内部
    /// 委譲フィットだけは対象外）。
    ///
    /// # Errors
    /// - 識別の順序条件`len(instruments) >= len(x_endog)`を満たさない:
    ///   `IvError::InsufficientInstruments`
    /// - 観測数`n`が構造方程式の係数の数`k`以下: `IvError::Common(CommonError::InsufficientObservations)`
    /// - 観測数`n`が第一段階の回帰変数の数`l`（`x_exog ++ instruments`、定数項込み）以下
    ///   （`n > k`でも）: `IvError::InsufficientObservationsForFirstStage`（`validate_iv_fit_inputs`）
    /// - 第一段階回帰（内生変数ごと）が失敗: `IvError::FirstStageFailed`
    ///   （`cov_type=Cluster`でグループキー未指定・信頼水準が範囲外等、`cov_type`/
    ///   `confidence_level`起因のエラーもここに含まれる）
    /// - 第二段階回帰（`second_stage`の内部委譲フィット）が失敗: `IvError::SecondStageFailed`
    ///   （`x_endog=[]`の退化ケースでは第一段階ループが一度も回らないため、
    ///   `confidence_level`が範囲外の場合のエラーもここ経由になる）
    /// - `cov_type=Hac`の`hac_lags`が不正: `CommonError::InvalidHacLags`
    /// - `cov_type=Cluster`でグループキー未指定・クラスター数不足:
    ///   `IvError::Common(CommonError::MissingClusterColumn` /
    ///   `CommonError::InsufficientClusters)`
    /// - `cov_type=Cluster`でクラスター数`g`が傾き係数の数`q`（`k - k_constant`）以下:
    ///   `IvError::Common(CommonError::InsufficientClustersForInference)`（`rank(Ŝ) ≤ g-1`
    ///   のためロバストWald/F検定の`q×q`部分行列が構造的に特異。`g > q`でも
    ///   悪条件で数値的にほぼ特異なら従来どおり`CommonError::NearSingularMatrix`が
    ///   backstop）
    ///
    /// Wu-Hausman検定（`wu_hausman_statistic()`/`wu_hausman_p_value()`）の
    /// 拡張回帰が想定内の理由（設計行列の特異性・観測数不足・Wald検定側の数値的な
    /// ほぼ特異性）で失敗する場合は`fit()`自体を失敗させず、該当フィールドが`None`に
    /// なるのみ（`engine/src/iv/CLAUDE.md`参照）。それ以外の理論上到達不能な理由で
    /// 拡張回帰が失敗した場合のみ`IvError::HausmanRegressionFailed`。
    ///
    /// 識別可能性の検証をここで行う理由は`IvInput`の構造体docコメント参照
    /// （`OlsEstimator::fit`が`n<=k`を検証するのと同じ層分け）。
    pub fn fit(input: IvInput, cov_type: CovType, confidence_level: f64) -> Result<Self, IvError> {
        fit::fit_two_sls(input, cov_type, confidence_level)
    }

    /// 内生変数ごとの第一段階回帰結果（`x_endog_names`と対応する順序）。
    /// タプルの`String`は内生変数名。
    pub fn first_stage_estimators(&self) -> &[(String, OlsEstimator)] {
        &self.first_stage
    }

    /// 内生変数ごとの弱操作変数診断（部分F統計量、`first_stage_estimators()`と同じ順序）。
    /// タプルの`String`は内生変数名。Stock-Yogo臨界値との照合は行わない（v1スコープ外、
    /// `docs/spec/iv-spec.md`3.4節）。
    pub fn weak_instrument_f_statistics(&self) -> &[(String, f64)] {
        &self.weak_instrument_f_statistics
    }

    /// Wu-Hausman内生性検定（回帰ベース）の統計量。`x_endog=[]`、または拡張回帰が想定内の
    /// 理由で推定不能な場合は`None`（`docs/spec/iv-spec.md`3.6節、`fit()`のdocコメント参照）。
    pub fn wu_hausman_statistic(&self) -> Option<f64> {
        self.wu_hausman_statistic
    }

    /// Wu-Hausman内生性検定のp値。`wu_hausman_statistic()`と同じ条件で`None`。
    pub fn wu_hausman_p_value(&self) -> Option<f64> {
        self.wu_hausman_p_value
    }

    /// Sargan過剰識別検定の統計量。丁度識別（自由度0）の場合は`None`
    /// （`docs/spec/iv-spec.md`3.5節、`fit()`のdocコメント参照）。
    pub fn sargan_statistic(&self) -> Option<f64> {
        self.sargan_statistic
    }

    /// Sargan過剰識別検定のp値。`sargan_statistic()`と同じ条件で`None`。
    pub fn sargan_p_value(&self) -> Option<f64> {
        self.sargan_p_value
    }

    /// Sargan検定の自由度（χ²）。`sargan_statistic()`と同じ条件で`None`。
    pub fn sargan_df(&self) -> Option<usize> {
        self.sargan_df
    }

    /// Wu-Hausman検定の自由度`(分子, 分母)`（F）。`wu_hausman_statistic()`と同じ条件で`None`。
    pub fn wu_hausman_df(&self) -> Option<(usize, usize)> {
        self.wu_hausman_df
    }

    /// `wald_statistic()`の自由度`(分子, 分母)`（F）。傾き係数が無くNaNのときは`None`。
    pub fn wald_df(&self) -> Option<(usize, usize)> {
        (self.df_model > 0).then_some((self.df_model, self.df_inference))
    }
}

impl IvEstimate for TwoSlsEstimator {
    /// 2SLSの点推定値（`param_names()`と対応する順序、`const`を含む）。
    fn params(&self) -> &Mat<f64> {
        self.second_stage.params()
    }

    /// 係数名（`x_exog_names ++ x_endog_names`、`x_exog`に定数項を含む場合は先頭が`"const"`）。
    fn param_names(&self) -> &[String] {
        self.second_stage.input().param_names()
    }

    /// 被説明変数名。
    fn dep_var_name(&self) -> &str {
        self.second_stage.input().dep_var_name()
    }

    /// 観測数 n。
    fn nobs(&self) -> usize {
        self.second_stage.input().nobs()
    }

    /// 係数の数 k（定数項を含む、`x_exog`と`x_endog`の合計）。
    fn k(&self) -> usize {
        self.second_stage.input().k()
    }

    /// 使用した標準誤差の種別（呼び出し元が指定した`cov_type`）。
    fn cov_type(&self) -> &CovType {
        &self.cov_type
    }

    /// `cov_type=Hac`のとき、実際に使われたラグ数（`hac_lags`の明示指定値、または
    /// 未指定時に経験則で自動計算した値）。`Hac`以外は`None`。
    fn hac_lags_used(&self) -> Option<usize> {
        self.hac_lags_used
    }

    /// 構造残差 `e = y - Xβ̂`（n, 1）。モジュール冒頭のdocコメント参照
    /// （第二段階回帰自身の残差`y - X̂β̂`ではない）。
    fn residuals(&self) -> &Mat<f64> {
        &self.residuals
    }

    /// 標準誤差 (k, 1)。
    fn std_errors(&self) -> &Mat<f64> {
        &self.std_errors
    }

    /// `cov_type`に応じた係数の分散共分散行列 (k, k)。
    fn cov_params(&self) -> &Mat<f64> {
        &self.cov_params
    }

    /// t統計量 (k, 1)。
    fn test_stats(&self) -> &Mat<f64> {
        &self.test_stats
    }

    /// `test_stats`の従う分布（t分布、自由度は`df_inference`。`cov_type=Cluster`のときだけ
    /// `df_resid`ではなく`G-1`になる）。
    fn stat_dist(&self) -> inference::StatDist {
        inference::StatDist::T {
            df: self.df_inference,
        }
    }

    /// 両側p値 (k, 1)。
    fn p_values(&self) -> &Mat<f64> {
        &self.p_values
    }

    /// 信頼区間の下限 (k, 1)。
    fn conf_lower(&self) -> &Mat<f64> {
        &self.conf_lower
    }

    /// 信頼区間の上限 (k, 1)。
    fn conf_upper(&self) -> &Mat<f64> {
        &self.conf_upper
    }

    /// 残差の自由度 n - k。
    fn df_resid(&self) -> usize {
        self.df_resid
    }

    /// モデルの自由度（定数項を除く傾き係数の数）。
    fn df_model(&self) -> usize {
        self.df_model
    }

    /// 決定係数。
    fn r_squared(&self) -> f64 {
        self.r_squared
    }

    /// 自由度調整済み決定係数。
    fn adj_r_squared(&self) -> f64 {
        self.adj_r_squared
    }

    /// F統計量。
    fn wald_statistic(&self) -> f64 {
        self.wald_statistic
    }

    /// F統計量のp値。
    fn wald_p_value(&self) -> f64 {
        self.wald_p_value
    }
}
