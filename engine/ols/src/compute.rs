use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};

use crate::config::CovType;
use crate::error::OlsError;
use crate::results::OlsResults;
use crate::OlsEstimator;

impl OlsEstimator {
    /// QR 分解で OLS を推定する。
    ///
    /// # アルゴリズム
    /// 1. X = QR（薄型 QR）
    /// 2. β̂ = R⁻¹ Qᵀy
    /// 3. ε̂ = y − Xβ̂、σ̂² = ε̂ᵀε̂ / (n−k)
    /// 4. cov_type に応じた分散推定
    /// 5. t 値・p 値・信頼区間・適合度統計量の計算
    pub fn fit(&self) -> Result<OlsResults, OlsError> {
        let n = self.y.len();
        let k = self.x.ncols();
        let df_resid = n - k;

        // 定数列（全要素 = 1）があるか検出して df_model を決定
        let has_constant = self.x.columns().into_iter()
            .any(|col| col.iter().all(|&v| (v - 1.0).abs() < 1e-10));
        let df_model = if has_constant { k - 1 } else { k };

        // ── 1. nalgebra 行列に変換 ────────────────────────────────────────
        let x_na = to_dmatrix(&self.x);
        let y_na = to_dvector(&self.y);

        // ── 2. QR 分解 ────────────────────────────────────────────────────
        let qr = x_na.qr();
        let (q_full, r_full) = qr.unpack();
        // 薄型 Q: n×k（全体の Q の先頭 k 列）
        let q_thin = q_full.columns(0, k).into_owned();
        // k×k 上三角 R（R の先頭 k 行）
        let r_kk = r_full.rows(0, k).into_owned();

        // 特異性チェック: R の対角要素の最小絶対値
        let r_diag_min = (0..k)
            .map(|i| r_kk[(i, i)].abs())
            .fold(f64::INFINITY, f64::min);
        if r_diag_min < 1e-10 {
            return Err(OlsError::SingularMatrix);
        }

        // ── 3. β̂ = R⁻¹ Qᵀy ──────────────────────────────────────────────
        let qty = q_thin.transpose() * &y_na;                // Qᵀy (k,)
        let beta_na = r_kk.clone().lu().solve(&qty).ok_or(OlsError::SingularMatrix)?;
        let beta = from_dvector(&beta_na);

        // ── 4. 残差・σ̂² ──────────────────────────────────────────────────
        let fitted: Array1<f64> = self.x.dot(&beta);
        let residuals: Array1<f64> = &self.y - &fitted;
        let ssr = residuals.dot(&residuals);
        let sigma2 = ssr / df_resid as f64;

        // ── 5. レバレッジ h_ii = Σ_j Q[i,j]² ────────────────────────────
        // hᵢᵢ = ‖Q_thin[i,:]‖²（HC2/HC3 用）
        let leverages: Array1<f64> = Array1::from_shape_fn(n, |i| {
            (0..k).map(|j| q_thin[(i, j)].powi(2)).sum()
        });

        // ── 6. (XᵀX)⁻¹ = R⁻¹ R⁻ᵀ ────────────────────────────────────────
        let eye_k = DMatrix::<f64>::identity(k, k);
        let r_inv = r_kk.lu().solve(&eye_k).ok_or(OlsError::SingularMatrix)?;
        let xtx_inv = &r_inv * r_inv.transpose();

        // ── 7. 分散共分散行列 ──────────────────────────────────────────────
        let cov_na = match &self.config.cov_type {
            CovType::NonRobust => {
                &xtx_inv * sigma2
            }
            CovType::HC0 => {
                hc_sandwich(&xtx_inv, &residuals, &self.x, n, df_resid, HcKind::HC0)
            }
            CovType::HC1 => {
                hc_sandwich(&xtx_inv, &residuals, &self.x, n, df_resid, HcKind::HC1)
            }
            CovType::HC2 => {
                hc_sandwich_lev(&xtx_inv, &residuals, &self.x, &leverages, HcKind::HC2)
            }
            CovType::HC3 => {
                hc_sandwich_lev(&xtx_inv, &residuals, &self.x, &leverages, HcKind::HC3)
            }
            CovType::Cluster => {
                let ids = self.cluster_ids.as_ref().expect("cluster_ids validated in new()");
                cluster_sandwich(&xtx_inv, &residuals, &self.x, ids, n, df_resid)
            }
        };
        let cov_params = from_dmatrix(&cov_na, k, k);

        // ── 8. 標準誤差・t 値・p 値 ───────────────────────────────────────
        let std_errors: Array1<f64> = cov_params.diag().mapv(f64::sqrt);
        let t_stats: Array1<f64> = &beta / &std_errors;

        let t_dist = StudentsT::new(0.0, 1.0, df_resid as f64)
            .map_err(|e| OlsError::ComputationError { msg: e.to_string() })?;
        let p_values: Array1<f64> = t_stats.mapv(|t| 2.0 * (1.0 - t_dist.cdf(t.abs())));

        // ── 9. 信頼区間（α = 0.05）────────────────────────────────────────
        let t_crit = t_dist.inverse_cdf(0.975);
        let conf_int = Array2::from_shape_fn((k, 2), |(i, col)| {
            if col == 0 {
                beta[i] - t_crit * std_errors[i]
            } else {
                beta[i] + t_crit * std_errors[i]
            }
        });

        // ── 10. 適合度統計量 ───────────────────────────────────────────────
        let y_mean = self.y.mean().unwrap_or(0.0);
        let sst = self.y.mapv(|yi| (yi - y_mean).powi(2)).sum();

        let r_squared = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };
        let r_squared_adj = if sst > 0.0 {
            1.0 - (ssr / df_resid as f64) / (sst / (n - 1) as f64)
        } else {
            0.0
        };

        let (f_statistic, f_p_value) = if df_model > 0 && ssr > 0.0 {
            let f = ((sst - ssr) / df_model as f64) / (ssr / df_resid as f64);
            let f_dist = FisherSnedecor::new(df_model as f64, df_resid as f64)
                .map_err(|e| OlsError::ComputationError { msg: e.to_string() })?;
            let p = 1.0 - f_dist.cdf(f);
            (f, p)
        } else {
            (0.0, 1.0)
        };

        // log-likelihood = -n/2 * [ln(2π) + ln(σ̂²) + 1]
        let log_likelihood =
            -0.5 * n as f64 * (std::f64::consts::TAU.ln() + sigma2.ln() + 1.0);
        let aic = n as f64 * (ssr / n as f64).ln() + 2.0 * k as f64;
        let bic = n as f64 * (ssr / n as f64).ln() + k as f64 * (n as f64).ln();

        Ok(OlsResults {
            params: beta,
            std_errors,
            t_stats,
            p_values,
            conf_int,
            residuals,
            fitted_values: fitted,
            r_squared,
            r_squared_adj,
            f_statistic,
            f_p_value,
            aic,
            bic,
            log_likelihood,
            sigma2,
            nobs: n,
            df_resid,
            df_model,
            cov_params,
            param_names: self.param_names.clone(),
            dep_var_name: self.dep_var_name.clone(),
            cov_type: self.config.cov_type.as_str().to_string(),
        })
    }
}

// ── 標準誤差ヘルパー ──────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum HcKind {
    HC0,
    HC1,
    HC2,
    HC3,
}

/// HC0 / HC1 サンドイッチ推定量
///
/// HC0: `(XᵀX)⁻¹ (Σᵢ ε̂ᵢ² xᵢxᵢᵀ) (XᵀX)⁻¹`
/// HC1: HC0 × n/(n−k)
fn hc_sandwich(
    xtx_inv: &DMatrix<f64>,
    residuals: &Array1<f64>,
    x: &Array2<f64>,
    n: usize,
    df_resid: usize,
    kind: HcKind,
) -> DMatrix<f64> {
    let k = x.ncols();
    let mut meat = DMatrix::<f64>::zeros(k, k);

    for i in 0..n {
        let xi = DVector::from_fn(k, |j, _| x[[i, j]]);
        meat += residuals[i].powi(2) * &xi * xi.transpose();
    }

    if matches!(kind, HcKind::HC1) {
        meat *= n as f64 / df_resid as f64;
    }

    xtx_inv * &meat * xtx_inv
}

/// HC2 / HC3 サンドイッチ推定量（レバレッジ補正あり）
///
/// HC2: `(XᵀX)⁻¹ (Σᵢ ε̂ᵢ²/(1−hᵢᵢ) xᵢxᵢᵀ) (XᵀX)⁻¹`
/// HC3: `(XᵀX)⁻¹ (Σᵢ ε̂ᵢ²/(1−hᵢᵢ)² xᵢxᵢᵀ) (XᵀX)⁻¹`
fn hc_sandwich_lev(
    xtx_inv: &DMatrix<f64>,
    residuals: &Array1<f64>,
    x: &Array2<f64>,
    leverages: &Array1<f64>,
    kind: HcKind,
) -> DMatrix<f64> {
    let n = residuals.len();
    let k = x.ncols();
    let mut meat = DMatrix::<f64>::zeros(k, k);

    let power = match kind {
        HcKind::HC2 => 1,
        HcKind::HC3 => 2,
        _ => unreachable!(),
    };

    for i in 0..n {
        let h = leverages[i].min(1.0 - 1e-10); // 数値安定化
        let weight = residuals[i].powi(2) / (1.0 - h).powi(power);
        let xi = DVector::from_fn(k, |j, _| x[[i, j]]);
        meat += weight * &xi * xi.transpose();
    }

    xtx_inv * &meat * xtx_inv
}

/// クラスター標準誤差
///
/// `(XᵀX)⁻¹ (Σ_g Xgᵀ ε̂g ε̂gᵀ Xg) (XᵀX)⁻¹ × G(n−1)/((G−1)(n−k))`
fn cluster_sandwich(
    xtx_inv: &DMatrix<f64>,
    residuals: &Array1<f64>,
    x: &Array2<f64>,
    cluster_ids: &Array1<i64>,
    n: usize,
    df_resid: usize,
) -> DMatrix<f64> {
    let k = x.ncols();

    // クラスターごとの観測インデックスを収集
    let mut cluster_map: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, &cid) in cluster_ids.iter().enumerate() {
        cluster_map.entry(cid).or_default().push(i);
    }
    let g = cluster_map.len();

    // Σ_g (Xgᵀ ε̂g)(Xgᵀ ε̂g)ᵀ — meat 行列
    let mut meat = DMatrix::<f64>::zeros(k, k);
    for indices in cluster_map.values() {
        // Xgᵀ ε̂g = Σ_{i∈g} ε̂ᵢ xᵢ (k,)
        let mut xg_eps = DVector::<f64>::zeros(k);
        for &i in indices {
            let xi = DVector::from_fn(k, |j, _| x[[i, j]]);
            xg_eps += residuals[i] * xi;
        }
        meat += &xg_eps * xg_eps.transpose();
    }

    // 小標本補正: G(n−1) / ((G−1)(n−k))
    let correction = (g * (n - 1)) as f64 / ((g - 1) * df_resid) as f64;
    meat *= correction;

    xtx_inv * &meat * xtx_inv
}

// ── 型変換ヘルパー ────────────────────────────────────────────────────────────

/// ndarray Array2 → nalgebra DMatrix（コピー発生）
fn to_dmatrix(a: &Array2<f64>) -> DMatrix<f64> {
    let (n, k) = a.dim();
    DMatrix::from_fn(n, k, |i, j| a[[i, j]])
}

/// ndarray Array1 → nalgebra DVector（コピー発生）
fn to_dvector(a: &Array1<f64>) -> DVector<f64> {
    DVector::from_fn(a.len(), |i, _| a[i])
}

/// nalgebra DVector → ndarray Array1（コピー発生）
fn from_dvector(v: &DVector<f64>) -> Array1<f64> {
    Array1::from_vec(v.iter().copied().collect())
}

/// nalgebra DMatrix → ndarray Array2（コピー発生）
fn from_dmatrix(m: &DMatrix<f64>, nrows: usize, ncols: usize) -> Array2<f64> {
    Array2::from_shape_fn((nrows, ncols), |(i, j)| m[(i, j)])
}

// ── ユニットテスト ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use crate::{
        config::{CovType, OlsConfig},
        error::OlsError,
        OlsEstimator,
    };

    fn default_estimator(y: Array1<f64>, x: Array2<f64>) -> Result<OlsEstimator, OlsError> {
        let k = x.ncols();
        let names: Vec<String> = (0..k).map(|i| format!("x{i}")).collect();
        OlsEstimator::new(y, x, None, names, "y".into(), OlsConfig::default())
    }

    /// β̂ が正解に近いことと R² > 0.99 を確認する。
    #[test]
    fn test_coefficients_and_r_squared() {
        // y ≈ 0.2 + 2.0*x（小ノイズ）
        let y = Array1::from_vec(vec![2.1, 3.9, 6.2, 8.1]);
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0],
        )
        .unwrap();
        let res = default_estimator(y, x).unwrap().fit().unwrap();
        assert!((res.params[0] - 0.2_f64).abs() < 0.15, "const={}", res.params[0]);
        assert!((res.params[1] - 2.0_f64).abs() < 0.15, "slope={}", res.params[1]);
        assert!(res.r_squared > 0.99, "R²={}", res.r_squared);
    }

    /// 線形従属列があると SingularMatrix を返す。
    #[test]
    fn test_singular_matrix() {
        // x[:,2] = 2*x[:,1] → ランク欠損
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 1.0, 2.0, 1.0, 2.0, 4.0, 1.0, 3.0, 6.0, 1.0, 4.0, 8.0],
        )
        .unwrap();
        let est = OlsEstimator::new(
            y,
            x,
            None,
            vec!["c".into(), "x1".into(), "x2".into()],
            "y".into(),
            OlsConfig::default(),
        )
        .unwrap();
        assert!(matches!(est.fit(), Err(OlsError::SingularMatrix)));
    }

    /// y に NaN が含まれると InvalidInput を返す。
    #[test]
    fn test_nan_in_y() {
        let y = Array1::from_vec(vec![1.0, f64::NAN, 3.0, 4.0]);
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0],
        )
        .unwrap();
        assert!(matches!(
            default_estimator(y, x),
            Err(OlsError::InvalidInput { .. })
        ));
    }

    /// n <= k で InsufficientObservations を返す。
    #[test]
    fn test_insufficient_observations() {
        let y = Array1::from_vec(vec![1.0, 2.0]);
        let x = Array2::from_shape_vec(
            (2, 3),
            vec![1.0, 1.0, 2.0, 1.0, 2.0, 3.0],
        )
        .unwrap();
        assert!(matches!(
            default_estimator(y, x),
            Err(OlsError::InsufficientObservations { .. })
        ));
    }

    /// cov_type=Cluster かつ cluster_ids=None で MissingClusterColumn を返す。
    #[test]
    fn test_missing_cluster_column() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0],
        )
        .unwrap();
        let config = OlsConfig {
            cov_type: CovType::Cluster,
            cluster_col: None,
            leverage_approx: false,
        };
        assert!(matches!(
            OlsEstimator::new(y, x, None, vec!["c".into(), "x1".into()], "y".into(), config),
            Err(OlsError::MissingClusterColumn)
        ));
    }
}
