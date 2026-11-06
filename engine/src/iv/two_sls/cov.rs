//! 2SLSの係数分散共分散行列（`X̂`・構造残差`e`のサンドイッチ型）と、Wald F検定。
//!
//! 定義と、OLSの同型計算を使わず独立実装にしている理由は`two_sls.rs`のモジュールdoc参照。
//!
//! テストは`cov/tests.rs`と`cov/tests/`にテーマ別で置く。

use crate::iv::common::IvError;
use crate::linear::ols::CovType;
use crate::shared::covariance::{cluster_cov_params, resolve_hac_lags, time_ordering};
use crate::shared::error::{ClusterSetting, CommonError, MatrixKind};
use crate::shared::linear_algebra::ensure_well_conditioned_symmetric_matrix;
use crate::shared::validation::validate_cluster_groups;
use faer::linalg::matmul::matmul;
use faer::prelude::Solve;
use faer::{Accum, Mat, Par, Side};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

/// `(X̂'X̂)⁻¹`を求める。`two_sls.rs`のモジュールdoc「標準誤差・適合度統計量」参照。
///
/// `X̂`（第二段階の設計行列）は`second_stage`の内部委譲フィット（`OlsEstimator::fit`）が
/// 既に`col_piv_qr`で特異性を検証済みのため、理論上ここで`LltError`は発生しないはずだが、
/// 浮動小数点演算の丸めにより境界的なケースで失敗しうる（`shared/covariance.rs`の`xtx_inverse`と同じ
/// 防御的な扱い）。
pub(super) fn xtx_inverse(x: &Mat<f64>, k: usize) -> Result<Mat<f64>, IvError> {
    let xtx = x.transpose() * x;
    let llt = xtx
        .llt(Side::Lower)
        .map_err(|_| CommonError::internal("the second-stage X̂'X̂ could not be factorized"))?;
    Ok(llt.solve(Mat::<f64>::identity(k, k)))
}

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
            hc_cov_params(x_hat, residuals, xtx_inv, n, k, HcVariant::Hc0),
            df_resid,
        ),
        CovType::Hc1 => (
            hc_cov_params(x_hat, residuals, xtx_inv, n, k, HcVariant::Hc1),
            df_resid,
        ),
        CovType::Hc2 => (
            hc_cov_params(x_hat, residuals, xtx_inv, n, k, HcVariant::Hc2),
            df_resid,
        ),
        CovType::Hc3 => (
            hc_cov_params(x_hat, residuals, xtx_inv, n, k, HcVariant::Hc3),
            df_resid,
        ),
        CovType::Hac { lags, time_order } => {
            let lags = resolve_hac_lags(*lags, n)?;
            hac_lags_used = Some(lags);
            let order = time_ordering(time_order, n);
            (
                hac_cov_params(x_hat, residuals, xtx_inv, n, k, lags, &order),
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

/// classical（等分散前提）の係数分散共分散行列: `σ̂²(X̂'X̂)⁻¹`（k×k）。`σ̂²`は構造残差
/// （`e = y - Xβ̂`）のSSRを`df_resid`で割った値。
fn classical_cov_params(sigma2: f64, xtx_inv: &Mat<f64>, k: usize) -> Mat<f64> {
    Mat::from_fn(k, k, |i, j| sigma2 * (*xtx_inv.get(i, j)))
}

/// `hc_cov_params`の内部でのみ使う、HCの種類（`shared/covariance.rs`の`HcVariant`と同じ位置づけ）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HcVariant {
    Hc0,
    Hc1,
    Hc2,
    Hc3,
}

/// HC0〜HC3ロバストな係数分散共分散行列: `(X̂'X̂)⁻¹Ψ̂(X̂'X̂)⁻¹`（k×k）。`Ψ̂`は構造残差
/// （`residuals`引数、`e = y - Xβ̂`）を使って計算する。数式・実装方針は`shared/covariance.rs`の
/// `hc_cov_params`と同型（`two_sls.rs`のモジュールdoc「独立実装」参照。設計行列に`X̂`、
/// 残差に構造残差`e`を使う点のみがOLSとの違い）。
fn hc_cov_params(
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    n: usize,
    k: usize,
    variant: HcVariant,
) -> Mat<f64> {
    let leverage: Option<Vec<f64>> = match variant {
        HcVariant::Hc2 | HcVariant::Hc3 => {
            let xh = x * xtx_inv; // (n, k)
            Some(
                (0..n)
                    .map(|i| (0..k).map(|j| (*xh.get(i, j)) * (*x.get(i, j))).sum())
                    .collect(),
            )
        }
        HcVariant::Hc0 | HcVariant::Hc1 => None,
    };

    let hc1_correction = ((n as f64) / ((n - k) as f64)).sqrt();

    let x_scaled = Mat::from_fn(n, k, |i, j| {
        let resid = *residuals.get(i, 0);
        let scale = match variant {
            HcVariant::Hc0 => resid,
            HcVariant::Hc1 => resid * hc1_correction,
            HcVariant::Hc2 => {
                let h = leverage.as_ref().expect("leverage is computed for HC2")[i];
                resid / (1.0 - h).sqrt()
            }
            HcVariant::Hc3 => {
                let h = leverage.as_ref().expect("leverage is computed for HC3")[i];
                resid / (1.0 - h)
            }
        };
        scale * (*x.get(i, j))
    });

    let psi_hat = x_scaled.transpose() * &x_scaled;
    xtx_inv * &psi_hat * xtx_inv
}

/// Newey-West HACの係数分散共分散行列: `(X̂'X̂)⁻¹Ŝ(X̂'X̂)⁻¹`（k×k）。数式・実装方針は
/// `shared/covariance.rs`の`hac_cov_params`と同型（`Par::Seq`を明示指定する理由も同じ、
/// `.claude/rules/rust-style.md`「パフォーマンス」参照）。設計行列に`X̂`、残差に構造残差
/// `e`を使う点のみがOLSとの違い。
fn hac_cov_params(
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    n: usize,
    k: usize,
    lags: usize,
    order: &[usize],
) -> Mat<f64> {
    let xe = Mat::<f64>::from_fn(n, k, |t, a| {
        let i = order[t];
        (*residuals.get(i, 0)) * (*x.get(i, a))
    });

    let mut s_hat = Mat::<f64>::zeros(k, k);
    matmul(
        s_hat.as_mut(),
        Accum::Replace,
        xe.transpose(),
        xe.as_ref(),
        1.0,
        Par::Seq,
    );

    let mut s_l = Mat::<f64>::zeros(k, k);
    for l in 1..=lags {
        let weight = 1.0 - (l as f64) / ((lags + 1) as f64);
        let xe_top = xe.as_ref().subrows(l, n - l);
        let xe_bot = xe.as_ref().subrows(0, n - l);
        matmul(
            s_l.as_mut(),
            Accum::Replace,
            xe_top.transpose(),
            xe_bot,
            1.0,
            Par::Seq,
        );

        for a in 0..k {
            for b in 0..k {
                *s_hat.get_mut(a, b) += weight * (*s_l.get(a, b) + *s_l.get(b, a));
            }
        }
    }

    xtx_inv * &s_hat * xtx_inv
}

/// 傾き係数（切片を除く`df_model`個の係数）が全てゼロという帰無仮説のロバストWald検定を行い、
/// F統計量とそのp値を返す。数式・実装方針は`shared/wald.rs`の`wald_f_test`と同型（`cov_type=
/// Classical`のとき代数的に古典的F検定と一致することも同じ、`ensure_well_conditioned_
/// symmetric_matrix`による事前の条件数チェックが必要な理由も同じ）。
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
