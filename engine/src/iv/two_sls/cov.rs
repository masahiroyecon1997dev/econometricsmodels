//! 2SLSの係数分散共分散行列（`X̂`・構造残差`e`のサンドイッチ型）と、Wald F検定。
//!
//! 定義と、共有のサンドイッチ計算を`X̂`・構造残差`e`で呼ぶ理由は`two_sls.rs`のモジュールdoc参照。
//!
//! テストは`cov/tests.rs`と`cov/tests/`にテーマ別で置く。

use crate::iv::common::IvError;
use crate::linear::ols::CovType;
use crate::shared::covariance::{
    HcVariant, classical_cov_params, cluster_cov_params, hac_cov_params, hc_cov_params,
    resolve_hac_lags, time_ordering,
};
use crate::shared::error::{ClusterSetting, CommonError};
use crate::shared::validation::validate_cluster_groups;
use faer::Mat;

/// [`covariance_stage`]の戻り値。
pub(super) struct CovStage {
    pub(super) cov_params: Mat<f64>,
    /// t検定・信頼区間・F検定に使う自由度。`cov_type=Cluster`のときだけ`G-1`、それ以外は
    /// `covariance_stage`に渡された`df_resid`と一致する。
    pub(super) df_inference: usize,
    /// `cov_type=Hac`のとき実際に使われたラグ数。`Hac`以外では`None`。
    pub(super) hac_lags_used: Option<usize>,
}

/// `cov_type`に応じた2SLSの係数分散共分散行列（`X̂`・構造残差`e`のサンドイッチ型）と推論の自由度。
///
/// 自由度は`df_resid`を引数で受け取る（`fit()`は`n - k`を渡す）。Classicalの`σ̂² = ssr/df_resid`、
/// t検定等の自由度の既定値が`df_resid`に従う。HC1の`n/(n-k)`、クラスターの
/// `(G/(G-1))((n-1)/(n-k))`、HACのラグ数の解決は`n`・`k`（`x_hat`の形）に従う。
///
/// `df_inference`は`cov_type=Cluster`のときだけ`G-1`に切り替える（OLSと同じ慣行、
/// `linear/ols.rs`の`fit()`docコメント参照）。クラスター数`g`が構造方程式の傾き係数の数`q`
/// 以下の入力は`fit()`冒頭で`InsufficientClustersForInference`として弾き済み。
///
/// # Errors
/// - `cov_type=Hac`の`lags`が不正: `CommonError::InvalidHacLags`
/// - `cov_type=Cluster`でグループキー未指定・長さ不一致・クラスター数不足:
///   `CommonError::MissingClusterColumn` / `CommonError::InsufficientClusters`等
pub(super) fn covariance_stage(
    cov_type: &CovType,
    x_hat: &Mat<f64>,
    residuals: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    ssr: f64,
    df_resid: usize,
) -> Result<CovStage, IvError> {
    let n = x_hat.nrows();
    let k = x_hat.ncols();
    let mut hac_lags_used = None;
    let (cov_params, df_inference) = match cov_type {
        CovType::Classical => {
            let sigma2 = ssr / (df_resid as f64);
            (classical_cov_params(sigma2, xtx_inv, k), df_resid)
        }
        CovType::Hc0 => (
            hc_cov_params(x_hat, residuals, xtx_inv, HcVariant::Hc0),
            df_resid,
        ),
        CovType::Hc1 => (
            hc_cov_params(x_hat, residuals, xtx_inv, HcVariant::Hc1),
            df_resid,
        ),
        CovType::Hc2 => (
            hc_cov_params(x_hat, residuals, xtx_inv, HcVariant::Hc2),
            df_resid,
        ),
        CovType::Hc3 => (
            hc_cov_params(x_hat, residuals, xtx_inv, HcVariant::Hc3),
            df_resid,
        ),
        CovType::Hac { lags, time_order } => {
            let lags = resolve_hac_lags(*lags, n)?;
            hac_lags_used = Some(lags);
            let order = time_ordering(time_order, n);
            (
                hac_cov_params(x_hat, residuals, xtx_inv, lags, &order),
                df_resid,
            )
        }
        CovType::Cluster { groups } => {
            let groups = groups.as_ref().ok_or(CommonError::MissingClusterColumn {
                setting: ClusterSetting::CovType,
            })?;
            let n_groups = validate_cluster_groups(groups, n, ClusterSetting::CovType)?;
            // 設計行列に`X̂`、残差に構造残差`e`を使う点のみがOLSとの違い（共有関数は
            // 設計行列と残差を引数に取る）。
            let cov = cluster_cov_params(x_hat, residuals, xtx_inv, groups);
            (cov, n_groups - 1)
        }
    };
    Ok(CovStage {
        cov_params,
        df_inference,
        hac_lags_used,
    })
}

#[cfg(test)]
mod tests;
