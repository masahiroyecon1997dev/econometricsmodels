//! 2SLSとGMMが共通で持つ、推定結果の読み出し口（`IvEstimate`）。

use crate::linear::ols::CovType;
use crate::shared::inference::StatDist;
use faer::Mat;

/// 2SLS（[`TwoSlsEstimator`](crate::iv::two_sls::TwoSlsEstimator)）とGMM
/// （[`GmmEstimator`](crate::iv::gmm::GmmEstimator)）が共通で持つ、推定結果の読み出し口。
///
/// 係数・標準誤差・検定・適合度など、`engine_pybind`が`IVResult`の共通フィールドに詰める値を
/// 集めている（`k()`・`cov_type()`のように`IVResult`では読まないものも、両推定量で署名が
/// 同じなので含める）。推定量ごとに意味や型が違うもの（過剰識別検定・Wu-Hausman検定・`wald_df`・
/// GMMの`n_iter`/`converged`等）は含めず、各推定量のメソッドとして持つ。
///
/// 共通の読み出し口をtraitにしておくと、共通フィールドを足すときに2つの推定量のどちらかで
/// 実装を忘れるとビルドエラーになる。今後IVの推定量（`FeIv`等）を足すときも同じ口を満たす
/// ことを型が強制する。メソッドを呼ぶ側はこのtraitを`use`する必要がある。共通の読み出し口の範囲を広げるのは、3つ目の推定量が実際に出て
/// 共通でないものが見えてからにする。
pub trait IvEstimate {
    /// 点推定値（`param_names()`と対応する順序、`const`を含む）。
    fn params(&self) -> &Mat<f64>;

    /// 係数名（`x_exog_names ++ x_endog_names`、`x_exog`に定数項を含む場合は先頭が`"const"`）。
    fn param_names(&self) -> &[String];

    /// 被説明変数名。
    fn dep_var_name(&self) -> &str;

    /// 観測数 n。
    fn nobs(&self) -> usize;

    /// 係数の数 k（定数項を含む、`x_exog`と`x_endog`の合計）。
    fn k(&self) -> usize;

    /// 使用した標準誤差の種別（呼び出し元が指定した`cov_type`）。
    fn cov_type(&self) -> &CovType;

    /// 実際に使われたHACラグ数。HACを使わなければ`None`。
    fn hac_lags_used(&self) -> Option<usize>;

    /// 構造残差 `e = y - Xβ̂`（n, 1）。
    fn residuals(&self) -> &Mat<f64>;

    /// 標準誤差 (k, 1)。
    fn std_errors(&self) -> &Mat<f64>;

    /// `cov_type`に応じた係数の分散共分散行列 (k, k)。
    fn cov_params(&self) -> &Mat<f64>;

    /// 検定統計量 (k, 1)（2SLSはt統計量、GMMはz統計量）。
    fn test_stats(&self) -> &Mat<f64>;

    /// `test_stats`の従う分布（2SLSはt分布、GMMは標準正規分布）。
    fn stat_dist(&self) -> StatDist;

    /// 両側p値 (k, 1)。
    fn p_values(&self) -> &Mat<f64>;

    /// 信頼区間の下限 (k, 1)。
    fn conf_lower(&self) -> &Mat<f64>;

    /// 信頼区間の上限 (k, 1)。
    fn conf_upper(&self) -> &Mat<f64>;

    /// 残差の自由度 n - k。
    fn df_resid(&self) -> usize;

    /// モデルの自由度（定数項を除く傾き係数の数）。
    fn df_model(&self) -> usize;

    /// 決定係数。
    fn r_squared(&self) -> f64;

    /// 自由度調整済み決定係数。
    fn adj_r_squared(&self) -> f64;

    /// 係数の同時検定の統計量（2SLSはF、GMMはχ²の生の二次形式）。
    fn wald_statistic(&self) -> f64;

    /// 係数の同時検定のp値。
    fn wald_p_value(&self) -> f64;
}
