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
use crate::shared::error::{ClusterSetting, CommonError, MatrixKind};
use crate::shared::linear_algebra::ensure_well_conditioned_symmetric_matrix;
use crate::shared::validation::validate_cluster_groups;
use faer::prelude::Solve;
use faer::{Mat, Side};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

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

/// 傾き係数（切片を除く`df_model`個の係数）が全てゼロという帰無仮説のロバストWald検定を行い、
/// F統計量とそのp値を返す。数式・実装方針は`shared/wald.rs`の`wald_f_test`と同型（`cov_type=
/// Classical`のとき代数的に古典的F検定と一致することも同じ、`ensure_well_conditioned_
/// symmetric_matrix`による事前の条件数チェックが必要な理由も同じ）。ただし`cov_params`の
/// 対称化は行わない自前版で、共有の`wald_quadratic_form`（対称化込み）への置き換えは
/// 数値が変わるため別の段階で行う。
pub(super) fn wald_f_test(
    params: &Mat<f64>,
    cov_params: &Mat<f64>,
    k_constant: usize,
    df_model: usize,
    df_inference: usize,
) -> Result<(f64, f64), IvError> {
    let beta_slopes = Mat::from_fn(df_model, 1, |i, _| *params.get(i + k_constant, 0));
    let v_slopes = Mat::from_fn(df_model, df_model, |i, j| {
        *cov_params.get(i + k_constant, j + k_constant)
    });

    ensure_well_conditioned_symmetric_matrix(
        &v_slopes,
        df_model,
        MatrixKind::CoefficientCovariance,
    )?;

    let llt = v_slopes
        .llt(Side::Lower)
        .map_err(|_| CommonError::NearSingularMatrix {
            matrix: MatrixKind::CoefficientCovariance,
        })?;
    let v_slopes_inv_beta = llt.solve(&beta_slopes);

    let wald: f64 = (0..df_model)
        .map(|i| (*beta_slopes.get(i, 0)) * (*v_slopes_inv_beta.get(i, 0)))
        .sum();
    let wald_statistic = wald / (df_model as f64);

    let f_dist = FisherSnedecor::new(df_model as f64, df_inference as f64)
        .map_err(|e| CommonError::internal(e.to_string()))?;
    let wald_p_value = f_dist.sf(wald_statistic);

    Ok((wald_statistic, wald_p_value))
}

#[cfg(test)]
mod tests;
