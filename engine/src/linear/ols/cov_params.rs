//! OLSの`cov_type`別の係数分散共分散行列と、それに伴う推論の自由度。
//!
//! `shared::covariance`の部品（classical / HC0-3 / HAC / cluster）を、OLS固有の規則で
//! つなぐ層: HACのラグ数の解決（`shared::covariance::resolve_hac_lags`、範囲外は
//! `CommonError::InvalidHacLags`）、クラスター列の検証、推論の自由度`df_inference`（`cov_type=Cluster`のときだけ`G-1`）。エラー型が
//! `LeastSquaresError`のため`shared/`には置けない。
//!
//! テストは`cov_params/tests.rs`と`cov_params/tests/`にテーマ別で置く。

use faer::Mat;

use super::cov_type::CovType;
use crate::linear::common::LeastSquaresError;
use crate::shared::covariance::{
    HcVariant, classical_cov_params, cluster_cov_params, hac_cov_params, hc_cov_params,
    resolve_hac_lags, time_ordering,
};
use crate::shared::error::{ClusterSetting, CommonError};
use crate::shared::validation::{validate_cluster_count_covers_slopes, validate_cluster_groups};

/// [`compute_cov_params`]の結果。
pub(super) struct CovParams {
    /// 係数の分散共分散行列 (k, k)
    pub(super) cov_params: Mat<f64>,
    /// t検定・信頼区間・F検定に使う自由度。通常は`df_resid`（n-k）と同じだが、
    /// `cov_type=Cluster`のときだけ`G-1`（クラスター数-1）に切り替える
    /// （statsmodelsの`df_correction=True`という既定と同じ挙動。標準的な計量経済学の
    /// 慣行でもある。`df_resid`自体は分散推定量`σ̂²`・調整済みR²・AIC/BIC等では
    /// 引き続き`n-k`のまま使う。`docs/spec/ols-spec.md`「標準誤差」のクラスター参照）。
    pub(super) df_inference: usize,
    /// `cov_type=Hac`のとき、実際に使われたラグ数。`Hac`以外では`None`。
    pub(super) hac_lags_used: Option<usize>,
}

/// `cov_type=Cluster`のクラスター数`g`が傾き係数の数`q`（`n_slopes`）以下だと、
/// クラスター寄与スコアの総和がゼロ（正規方程式`X'e = 0`）で`rank(Ŝ) ≤ g - 1`の
/// ため、ロバストWald/F検定の`q×q`部分行列が構造的に特異になる。
/// `g`・`q`は入力だけから判定できるため、QR分解・残差計算より前に弾く
/// （nonlinear/IVと同じく`fit()`冒頭で検証する方針に揃える）。`groups=None`は
/// [`compute_cov_params`]の`CovType::Cluster`アームで`MissingClusterColumn`として扱う。
///
/// 戻り値は全体F検定が定義できるか。`g ≤ q`のとき、`allow_undefined_f_test`が偽なら
/// `InsufficientClustersForInference`で失敗し、真なら`Ok(false)`（F検定だけ定義なし）を返す。
/// クラスターでない`cov_type`は常に`Ok(true)`。
pub(super) fn validate_cluster_count(
    cov_type: &CovType,
    n: usize,
    n_slopes: usize,
    allow_undefined_f_test: bool,
) -> Result<bool, LeastSquaresError> {
    let Some(groups) = cov_type.cluster_groups() else {
        return Ok(true);
    };
    let g = validate_cluster_groups(groups, n, ClusterSetting::CovType)?;
    match validate_cluster_count_covers_slopes(g, n_slopes) {
        Ok(()) => Ok(true),
        // F検定だけを定義なしにするのは`g <= q`の1種類に限る。将来`validate_cluster_count_
        // covers_slopes`に別のエラーが足されても、黙って`Ok(false)`に変えない。
        Err(CommonError::InsufficientClustersForInference { .. }) if allow_undefined_f_test => {
            Ok(false)
        }
        Err(e) => Err(e.into()),
    }
}

/// `cov_type`に応じた係数分散共分散行列と推論の自由度を求める。
///
/// `sigma2`は`SSR/df_resid`（classicalでのみ使う）、`df_resid`は`n - k`。
pub(super) fn compute_cov_params(
    cov_type: &CovType,
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    sigma2: f64,
    df_resid: usize,
) -> Result<CovParams, LeastSquaresError> {
    let n = x.nrows();
    let k = x.ncols();
    let mut hac_lags_used = None;
    let (cov_params, df_inference) = match cov_type {
        CovType::Classical => (classical_cov_params(sigma2, xtx_inv, k), df_resid),
        CovType::Hc0 => (
            hc_cov_params(x, residuals, xtx_inv, HcVariant::Hc0),
            df_resid,
        ),
        CovType::Hc1 => (
            hc_cov_params(x, residuals, xtx_inv, HcVariant::Hc1),
            df_resid,
        ),
        CovType::Hc2 => (
            hc_cov_params(x, residuals, xtx_inv, HcVariant::Hc2),
            df_resid,
        ),
        CovType::Hc3 => (
            hc_cov_params(x, residuals, xtx_inv, HcVariant::Hc3),
            df_resid,
        ),
        CovType::Hac { lags, time_order } => {
            let lags = resolve_hac_lags(*lags, n)?;
            hac_lags_used = Some(lags);
            let order = time_ordering(time_order, n);
            (
                hac_cov_params(x, residuals, xtx_inv, lags, &order),
                df_resid,
            )
        }
        CovType::Cluster { groups } => {
            let groups = groups.as_ref().ok_or(CommonError::MissingClusterColumn {
                setting: ClusterSetting::CovType,
            })?;
            // クラスター数`g >= 2`・`g > q`（傾き係数の数）は`validate_cluster_count`で
            // 検証済み。ここでは`n_groups - 1`（検定の自由度）に再利用するため
            // コードのユニーク数を引くだけ。
            let n_groups = validate_cluster_groups(groups, n, ClusterSetting::CovType)?;
            let cov = cluster_cov_params(x, residuals, xtx_inv, groups);
            (cov, n_groups - 1)
        }
    };
    Ok(CovParams {
        cov_params,
        df_inference,
        hac_lags_used,
    })
}

#[cfg(test)]
mod tests;
