//! 2SLSの係数分散共分散行列（`X̂`・構造残差`e`のサンドイッチ型）と、Wald F検定。
//!
//! 定義と、OLSの同型計算を使わず独立実装にしている理由は`two_sls.rs`のモジュールdoc参照。

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
/// `ols/estimator.rs`の`fit()`docコメント参照）。クラスター数`g`が構造方程式の傾き係数の数`q`
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
/// （`residuals`引数、`e = y - Xβ̂`）を使って計算する。数式・実装方針は`ols/estimator.rs`の
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
mod tests {
    use super::*;
    use crate::iv::common::{IvEstimate, IvInput};
    use crate::iv::two_sls::TwoSlsEstimator;
    use crate::iv::two_sls::test_support::*;
    use crate::linear::common::row_time_order;
    use crate::linear::ols::{OlsEstimator, OlsInput};
    use crate::shared::group_codes::GroupCodes;

    /// レバレッジ `h_ii = x_i'(X'X)⁻¹x_i` を素朴な二重ループで独立計算する（HC2/HC3用）。
    fn manual_leverage(x: &Mat<f64>, xtx_inv: &Mat<f64>, n: usize, k: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let mut h = 0.0;
                for a in 0..k {
                    for b in 0..k {
                        h += (*x.get(i, a)) * (*xtx_inv.get(a, b)) * (*x.get(i, b));
                    }
                }
                h
            })
            .collect()
    }

    /// HC0〜HC3の標準誤差を、行列積のショートカット（`x_scaled.transpose() * x_scaled`）を
    /// 使わず素朴な三重ループ（外積`x_i x_i'`の直接積み上げ）で独立計算する
    /// （`TwoSlsEstimator::fit`内部の`hc_cov_params`とは別経路の実装）。`weight(i)`は
    /// 観測`i`の`e_i²`に掛ける係数（HC0: 1、HC1: `n/(n-k)`、HC2: `1/(1-h_ii)`、
    /// HC3: `1/(1-h_ii)²`）。
    fn manual_hc_std_errors_with_weight(
        x: &Mat<f64>,
        e: &Mat<f64>,
        n: usize,
        k: usize,
        weight: impl Fn(usize) -> f64,
    ) -> Vec<f64> {
        let xtx_inv = manual_xtx_inverse(x, k);
        let mut psi = Mat::<f64>::zeros(k, k);
        for i in 0..n {
            let scaled_e2 = weight(i) * (*e.get(i, 0)).powi(2);
            for a in 0..k {
                for b in 0..k {
                    *psi.get_mut(a, b) += scaled_e2 * (*x.get(i, a)) * (*x.get(i, b));
                }
            }
        }
        let cov = &xtx_inv * &psi * &xtx_inv;
        (0..k).map(|j| (*cov.get(j, j)).sqrt()).collect()
    }

    fn manual_hc0_std_errors(x: &Mat<f64>, e: &Mat<f64>, n: usize, k: usize) -> Vec<f64> {
        manual_hc_std_errors_with_weight(x, e, n, k, |_| 1.0)
    }

    fn manual_hc1_std_errors(x: &Mat<f64>, e: &Mat<f64>, n: usize, k: usize) -> Vec<f64> {
        let correction = (n as f64) / ((n - k) as f64);
        manual_hc_std_errors_with_weight(x, e, n, k, move |_| correction)
    }

    fn manual_hc2_std_errors(x: &Mat<f64>, e: &Mat<f64>, n: usize, k: usize) -> Vec<f64> {
        let xtx_inv = manual_xtx_inverse(x, k);
        let leverage = manual_leverage(x, &xtx_inv, n, k);
        manual_hc_std_errors_with_weight(x, e, n, k, move |i| 1.0 / (1.0 - leverage[i]))
    }

    fn manual_hc3_std_errors(x: &Mat<f64>, e: &Mat<f64>, n: usize, k: usize) -> Vec<f64> {
        let xtx_inv = manual_xtx_inverse(x, k);
        let leverage = manual_leverage(x, &xtx_inv, n, k);
        manual_hc_std_errors_with_weight(x, e, n, k, move |i| 1.0 / (1.0 - leverage[i]).powi(2))
    }

    /// クラスターロバスト標準誤差を素朴なループ（クラスターごとの手動集約）で独立計算する。
    fn manual_cluster_std_errors(
        x: &Mat<f64>,
        e: &Mat<f64>,
        n: usize,
        k: usize,
        groups: &[String],
    ) -> Vec<f64> {
        let xtx_inv = manual_xtx_inverse(x, k);
        let mut group_sums: std::collections::BTreeMap<&str, Vec<f64>> =
            std::collections::BTreeMap::new();
        for (i, group) in groups.iter().enumerate().take(n) {
            let s_g = group_sums.entry(group.as_str()).or_insert(vec![0.0; k]);
            for (a, s_g_a) in s_g.iter_mut().enumerate() {
                *s_g_a += (*e.get(i, 0)) * (*x.get(i, a));
            }
        }
        let n_groups = group_sums.len();
        let mut s_hat = Mat::<f64>::zeros(k, k);
        for s_g in group_sums.values() {
            for a in 0..k {
                for b in 0..k {
                    *s_hat.get_mut(a, b) += s_g[a] * s_g[b];
                }
            }
        }
        let correction =
            (n_groups as f64 / (n_groups as f64 - 1.0)) * ((n as f64 - 1.0) / ((n - k) as f64));
        let cov = &xtx_inv * &s_hat * &xtx_inv;
        (0..k)
            .map(|j| (correction * (*cov.get(j, j))).sqrt())
            .collect()
    }

    /// Newey-West HAC標準誤差を、`hac_cov_params`の`matmul`ベースの実装とは別経路
    /// （素朴な二重ループでの外積の積み上げ）で独立計算する（`manual_hc_std_errors_with_weight`
    /// と同じ方針）。`order`は`time_ordering`の出力（時系列順の行インデックス列）。
    fn manual_hac_std_errors(
        x: &Mat<f64>,
        e: &Mat<f64>,
        n: usize,
        k: usize,
        lags: usize,
        order: &[usize],
    ) -> Vec<f64> {
        let xtx_inv = manual_xtx_inverse(x, k);
        let mut s_hat = Mat::<f64>::zeros(k, k);
        for &i in order {
            let ei = *e.get(i, 0);
            for a in 0..k {
                for b in 0..k {
                    *s_hat.get_mut(a, b) += ei * ei * (*x.get(i, a)) * (*x.get(i, b));
                }
            }
        }
        for l in 1..=lags {
            let weight = 1.0 - (l as f64) / ((lags + 1) as f64);
            for t in l..n {
                let i_t = order[t];
                let i_tl = order[t - l];
                let e_t = *e.get(i_t, 0);
                let e_tl = *e.get(i_tl, 0);
                for a in 0..k {
                    for b in 0..k {
                        let term = weight
                            * e_t
                            * e_tl
                            * ((*x.get(i_t, a)) * (*x.get(i_tl, b))
                                + (*x.get(i_tl, a)) * (*x.get(i_t, b)));
                        *s_hat.get_mut(a, b) += term;
                    }
                }
            }
        }
        let cov = &xtx_inv * &s_hat * &xtx_inv;
        (0..k).map(|j| (*cov.get(j, j)).sqrt()).collect()
    }

    fn assert_slices_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-8, "index {i}: got {a}, expected {e}");
        }
    }

    /// classicalの標準誤差が構造残差（`y - Xβ̂`、第二段階回帰自身の残差`y - X̂β̂`ではない）
    /// を使って計算されていることを確認する（`two_sls.rs`のモジュールdoc「標準誤差・
    /// 適合度統計量」参照。誤って第二段階の`OlsEstimator`自身のSEを使っていないことの
    /// 回帰テスト）。
    #[test]
    fn fit_computes_classical_std_errors_using_structural_residuals() {
        let estimator =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Classical, 0.95).unwrap();
        let (x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
        let n = x.nrows();
        let k = x.ncols();

        let expected_se = manual_classical_std_errors(&x, &e, n, k);
        let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
        assert_slices_close(&actual_se, &expected_se);

        // `residuals()`が構造残差（`y - Xβ̂`）そのものであることを確認する
        // （第二段階回帰自身の残差`y - X̂β̂`ではない）。
        assert_eq!(estimator.residuals().nrows(), n);
        for i in 0..n {
            assert!((*estimator.residuals().get(i, 0) - *e.get(i, 0)).abs() < 1e-8);
        }

        // 構造残差は第二段階の`OlsEstimator`自身の残差（`y - X̂β̂`）とは異なる値になる
        // はず（`x_endog`が操作変数に完全予測されない、内生性が残る非退化ケースのため。
        // `second_stage`は非公開フィールドだが、同一モジュールの子モジュールから
        // 直接参照できる）。
        let second_stage_residuals = estimator.second_stage.residuals();
        let mut any_differs = false;
        for i in 0..n {
            if (*estimator.residuals().get(i, 0) - *second_stage_residuals.get(i, 0)).abs() > 1e-8 {
                any_differs = true;
            }
        }
        assert!(
            any_differs,
            "expected structural residuals to differ from the second-stage OLS's own residuals"
        );
    }

    #[test]
    fn fit_computes_hc0_std_errors_matching_manual_sandwich_formula() {
        let estimator =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Hc0, 0.95).unwrap();
        let (x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
        let n = x.nrows();
        let k = x.ncols();

        let expected_se = manual_hc0_std_errors(&x, &e, n, k);
        let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
        assert_slices_close(&actual_se, &expected_se);
        assert_eq!(estimator.cov_type(), &CovType::Hc0);
    }

    #[test]
    fn fit_computes_hc1_std_errors_matching_manual_sandwich_formula() {
        let estimator =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Hc1, 0.95).unwrap();
        let (x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
        let n = x.nrows();
        let k = x.ncols();

        let expected_se = manual_hc1_std_errors(&x, &e, n, k);
        let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
        assert_slices_close(&actual_se, &expected_se);
    }

    /// HC2はレバレッジ`h_ii`によるスケーリングを要する（`X̂`のみからレバレッジを計算する
    /// 拡張、`docs/spec/iv-spec.md`3.1節参照）。このテストは本実装が意図した式（下記
    /// `manual_hc2_std_errors`と同一の式）通りに計算されていることを固定する、
    /// 数式レベルの細粒度回帰確認。真に独立した参照実装（R `ivreg`+
    /// `sandwich::vcovHC(type="HC2"/"HC3")`）との数値一致は
    /// `tests/iv/test_iv_crosscheck.py`で別途検証済み
    /// （`linearmodels`は引き続きhc2/hc3非対応のため対象外）。
    #[test]
    fn fit_computes_hc2_std_errors_matching_manual_sandwich_formula() {
        let estimator =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Hc2, 0.95).unwrap();
        let (x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
        let n = x.nrows();
        let k = x.ncols();

        let expected_se = manual_hc2_std_errors(&x, &e, n, k);
        let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
        assert_slices_close(&actual_se, &expected_se);
    }

    /// HC3も同様（`fit_computes_hc2_std_errors_matching_manual_sandwich_formula`のdoc
    /// コメント参照）。
    #[test]
    fn fit_computes_hc3_std_errors_matching_manual_sandwich_formula() {
        let estimator =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Hc3, 0.95).unwrap();
        let (x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
        let n = x.nrows();
        let k = x.ncols();

        let expected_se = manual_hc3_std_errors(&x, &e, n, k);
        let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
        assert_slices_close(&actual_se, &expected_se);
    }

    #[test]
    fn fit_computes_cluster_std_errors_matching_manual_formula() {
        let (input, groups) = cluster_test_input_and_groups();
        let cov_type = CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups)),
        };
        let estimator = TwoSlsEstimator::fit(input, cov_type, 0.95).unwrap();

        let y = [5.0, 3.0, 8.0, 6.0, 11.0, 10.0, 15.0, 13.0];
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x_endog = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0];
        let n = y.len();
        let x_structural = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => x1[i],
            _ => x_endog[i],
        });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);
        let e = &y_mat - &x_structural * estimator.params();
        let x_hat = estimator.second_stage.input().x();
        let k = x_hat.ncols();

        let expected_se = manual_cluster_std_errors(x_hat, &e, n, k, &groups);
        let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
        assert_slices_close(&actual_se, &expected_se);
    }

    /// `cov_type=Hac{lags: Some(0), ..}`は自己相関項（`l=1..=lags`のループ）が空になり、
    /// `l=0`項（HC0のΨ̂と同形）のみが残るため、HC0と数値的に一致するはず
    /// （`ols/estimator.rs`の`fit_hac_with_zero_lags_matches_hc0`と同じ内部整合性テスト）。
    #[test]
    fn fit_hac_with_zero_lags_matches_hc0() {
        let hac_estimator = TwoSlsEstimator::fit(
            nontrivial_x_exog_input(),
            CovType::Hac {
                lags: Some(0),
                time_order: row_time_order(8),
            },
            0.95,
        )
        .unwrap();
        let hc0_estimator =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Hc0, 0.95).unwrap();

        for j in 0..hac_estimator.k() {
            assert!(
                (*hac_estimator.std_errors().get(j, 0) - *hc0_estimator.std_errors().get(j, 0))
                    .abs()
                    < 1e-8
            );
        }
    }

    /// `lags=Some(2)`（自己相関項が実際に加算されるケース、`lags=0`の
    /// `fit_hac_with_zero_lags_matches_hc0`とは異なりHC0とは一致しないはず）を
    /// `manual_hac_std_errors`（独立実装）と照合する。
    #[test]
    fn fit_computes_hac_std_errors_with_explicit_lags_matching_manual_formula() {
        let estimator = TwoSlsEstimator::fit(
            nontrivial_x_exog_input(),
            CovType::Hac {
                lags: Some(2),
                time_order: row_time_order(8),
            },
            0.95,
        )
        .unwrap();
        let (x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
        let n = x.nrows();
        let k = x.ncols();
        let order: Vec<usize> = (0..n).collect();

        let expected_se = manual_hac_std_errors(&x, &e, n, k, 2, &order);
        let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
        assert_slices_close(&actual_se, &expected_se);
    }

    /// `lags=None`（経験則自動計算 `L = floor(4*(n/100)^(2/9))`）が、`n=8`では`L=2`と
    /// 一致するため、`lags=Some(2)`を明示指定した場合と数値的に一致するはず。
    #[test]
    fn fit_computes_hac_std_errors_with_auto_lags_matching_explicit_lags() {
        let auto_estimator = TwoSlsEstimator::fit(
            nontrivial_x_exog_input(),
            CovType::Hac {
                lags: None,
                time_order: row_time_order(8),
            },
            0.95,
        )
        .unwrap();
        let explicit_estimator = TwoSlsEstimator::fit(
            nontrivial_x_exog_input(),
            CovType::Hac {
                lags: Some(2),
                time_order: row_time_order(8),
            },
            0.95,
        )
        .unwrap();

        for j in 0..auto_estimator.k() {
            assert!(
                (*auto_estimator.std_errors().get(j, 0)
                    - *explicit_estimator.std_errors().get(j, 0))
                .abs()
                    < 1e-8
            );
        }
    }

    /// `hac_lags_used()`: `lags=None`なら経験則（`n=8`→2）、明示指定ならその値、Hac以外は
    /// `None`。第一段階のOLSも同じ`cov_type`で推定されるため同じ値になる。
    #[test]
    fn hac_lags_used_reflects_resolved_lags() {
        let fit = |cov_type: CovType| {
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), cov_type, 0.95).unwrap()
        };

        let auto = fit(CovType::Hac {
            lags: None,
            time_order: row_time_order(8),
        });
        assert_eq!(auto.hac_lags_used(), Some(2));
        for (_, first_stage) in auto.first_stage_estimators() {
            assert_eq!(first_stage.hac_lags_used(), Some(2));
        }

        let explicit = fit(CovType::Hac {
            lags: Some(3),
            time_order: row_time_order(8),
        });
        assert_eq!(explicit.hac_lags_used(), Some(3));

        assert_eq!(fit(CovType::Classical).hac_lags_used(), None);
        assert_eq!(fit(CovType::Hc1).hac_lags_used(), None);
    }

    /// `time_order`を指定した場合、行順がシャッフルされていても時系列順に並べ替えてから
    /// ラグ付き自己共分散を計算することを確認する（`ols/estimator.rs`の
    /// `fit_computes_hac_std_errors_respecting_time_order`と同じ検証方針）。データは
    /// `nontrivial_x_exog_columns()`と同一の内容を、時系列順の逆転を含む順序でシャッフル
    /// して与える。
    #[test]
    fn fit_computes_hac_std_errors_respecting_time_order() {
        let (x1, x_endog, z1, z2, y) = nontrivial_x_exog_columns();
        // 元の時系列順=[0..n)を、時系列順の逆転を含む順序でシャッフルする。
        let shuffle: Vec<usize> = vec![3, 1, 6, 0, 5, 2, 7, 4];
        assert_eq!(shuffle.len(), y.len());
        let shuffled_time: Vec<f64> = shuffle.iter().map(|&i| i as f64).collect();
        let shuffled_y: Vec<f64> = shuffle.iter().map(|&i| y[i]).collect();
        let shuffled_x1: Vec<f64> = shuffle.iter().map(|&i| x1[i]).collect();
        let shuffled_x_endog: Vec<f64> = shuffle.iter().map(|&i| x_endog[i]).collect();
        let shuffled_z1: Vec<f64> = shuffle.iter().map(|&i| z1[i]).collect();
        let shuffled_z2: Vec<f64> = shuffle.iter().map(|&i| z2[i]).collect();

        let shuffled_input = IvInput::from_columns(
            &shuffled_y,
            &[shuffled_x1],
            vec!["x1".to_string()],
            &[shuffled_x_endog],
            vec!["endog1".to_string()],
            &[shuffled_z1, shuffled_z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let shuffled_estimator = TwoSlsEstimator::fit(
            shuffled_input,
            CovType::Hac {
                lags: Some(2),
                time_order: shuffled_time,
            },
            0.95,
        )
        .unwrap();

        let unshuffled_estimator = TwoSlsEstimator::fit(
            nontrivial_x_exog_input(),
            CovType::Hac {
                lags: Some(2),
                time_order: row_time_order(8),
            },
            0.95,
        )
        .unwrap();

        for j in 0..shuffled_estimator.k() {
            assert!(
                (*shuffled_estimator.std_errors().get(j, 0)
                    - *unshuffled_estimator.std_errors().get(j, 0))
                .abs()
                    < 1e-8
            );
        }
    }

    #[test]
    fn fit_returns_invalid_hac_lags_error_when_out_of_range() {
        let result = TwoSlsEstimator::fit(
            x_endog_empty_input(),
            CovType::Hac {
                lags: Some(-1),
                time_order: row_time_order(5),
            },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::InvalidHacLags { hac_lags: -1, n: 5 })
        );
    }

    #[test]
    fn fit_returns_missing_cluster_column_error_when_groups_not_provided() {
        let result = TwoSlsEstimator::fit(
            x_endog_empty_input(),
            CovType::Cluster { groups: None },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::MissingClusterColumn {
                setting: ClusterSetting::CovType,
            })
        );
    }

    #[test]
    fn fit_returns_insufficient_clusters_error_when_only_one_group() {
        let groups = vec!["a".to_string(); 5];
        let result = TwoSlsEstimator::fit(
            x_endog_empty_input(),
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::InsufficientClusters {
                setting: ClusterSetting::CovType,
                g: 1
            })
        );
    }

    /// `cov_type=Cluster`でクラスター数`g`が傾き係数の数`q`以下だと、クラスター寄与
    /// スコアの総和がゼロ（正規方程式`X'e = 0`）で`rank(Ŝ) ≤ g - 1`のため全体Wald/F
    /// 検定の`q×q`部分行列が構造的に特異になる。`g`・`q`は入力だけから判定できるため
    /// `fit()`冒頭のバリデーションが`CommonError::InsufficientClustersForInference`で
    /// 弾く（OLSと同型）。`cluster_test_input_and_groups`の設計行列は
    /// `[const, x1, endog1]`（`k=3`・`k_constant=1`・`q=2`）なので、`g=2`（`g == q`）と
    /// 組み合わせる。
    #[test]
    fn fit_returns_validation_error_when_cluster_count_at_most_slopes() {
        let (input, _) = cluster_test_input_and_groups();
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];
        let result = TwoSlsEstimator::fit(
            input,
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::InsufficientClustersForInference { g: 2, q: 2 })
        );
    }

    /// 構造方程式は`G > q_struct`だが第一段階では`G == q_first`（`nontrivial_x_exog_input`は
    /// `[const, x1, endog1]`で`q_struct=2`、第一段階は`x1`・`z1`・`z2`で`q_first=3`、`g=3`）の
    /// とき、第一段階の全体F検定が定義できなくても`fit()`全体は成功する。第一段階の
    /// `f_statistic()`は`None`、構造方程式の結果（Wald検定を含む）は計算され、弱操作変数診断は
    /// `cov_type`に依存しないため古典的な`cov_type`と同じ値になる。
    #[test]
    fn fit_succeeds_when_cluster_count_is_between_structural_and_first_stage_slopes() {
        let n = nontrivial_x_exog_input().nobs();
        let groups: Vec<String> = (0..n).map(|i| format!("g{}", i % 3)).collect();
        let estimator = TwoSlsEstimator::fit(
            nontrivial_x_exog_input(),
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
            0.95,
        )
        .unwrap();

        let (_, first_stage) = &estimator.first_stage_estimators()[0];
        assert_eq!(first_stage.f_statistic(), None);
        assert_eq!(first_stage.f_df(), None);
        assert!(estimator.wald_statistic().is_finite());

        let classical =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Classical, 0.95).unwrap();
        assert_eq!(
            estimator.weak_instrument_f_statistics(),
            classical.weak_instrument_f_statistics()
        );
    }

    /// `G=2`クラスター・`x_exog=[]`・丁度識別（`instruments`1本）という、
    /// ベンチマーク作成中に発見した`ComputationError`（`engine/src/iv/CLAUDE.md`
    /// 「修正済み」参照）の再現条件そのもの。第一段階回帰の`OlsInput::
    /// from_columns`に`has_intercept=false`を渡していたため、`IvInput::x_exog()`が
    /// 焼き込み済みの定数項列まで「傾き係数」としてWald F検定の対象に含めてしまい
    /// （真のq=1のところq=2）、`rank(Ŝ)≤G-1=1`の構造的特異性で必ず失敗していた
    /// （`without_baked_in_intercept`の導入で修正、`common.rs`参照）。
    #[test]
    fn fit_succeeds_with_cluster_g2_boundary_when_x_exog_is_empty() {
        let y = vec![5.0, 3.0, 8.0, 6.0, 11.0, 10.0, 15.0, 13.0];
        let x_endog = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0];
        let z1 = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            std::slice::from_ref(&z1),
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let cov_type = CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups)),
        };
        let estimator = TwoSlsEstimator::fit(input, cov_type, 0.95).unwrap();

        // 第一段階（`endog1 ~ const + z1`）が、素の`OlsEstimator::fit`
        // （同じG=2クラスター、`has_intercept=true`）と一致することを確認する
        // （`compute_first_stage`の`has_intercept`修正が正しく効いていることの直接確認、
        // `test_matches_closed_form_ols_when_instrument_perfectly_predicts_endog`と
        // 同じ発想）。
        let (_, first_stage_estimator) = &estimator.first_stage_estimators()[0];
        let ols_input = OlsInput::from_columns(
            &x_endog,
            &[z1],
            vec!["z1".to_string()],
            true,
            "endog1".to_string(),
        )
        .unwrap();
        let expected = OlsEstimator::fit(
            ols_input,
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
            0.95,
        )
        .unwrap();

        assert_slices_close(
            &(0..2)
                .map(|j| *first_stage_estimator.params().get(j, 0))
                .collect::<Vec<_>>(),
            &(0..2)
                .map(|j| *expected.params().get(j, 0))
                .collect::<Vec<_>>(),
        );
        assert_slices_close(
            &(0..2)
                .map(|j| *first_stage_estimator.std_errors().get(j, 0))
                .collect::<Vec<_>>(),
            &(0..2)
                .map(|j| *expected.std_errors().get(j, 0))
                .collect::<Vec<_>>(),
        );

        // `has_intercept=false`を渡していた旧版では、`first_stage()`が返す`r_squared`/
        // `wald_statistic`も定数項を傾き係数扱いした非中心化TSS・過大なqで静かに間違って
        // いた（`engine/src/iv/CLAUDE.md`「修正済み」参照）。修正後は素の
        // `OlsEstimator::fit`と一致する。
        assert!((first_stage_estimator.r_squared() - expected.r_squared()).abs() < 1e-10);
        assert!(
            (first_stage_estimator.f_statistic().unwrap() - expected.f_statistic().unwrap()).abs()
                < 1e-8
        );
    }

    /// 全`cov_type`で、`cov_params()`が対称で、対角の平方根が`std_errors()`と一致する
    /// （`std_errors`は`cov_params`の対角から作るため、公開している行列が標準誤差の元そのもの
    /// であることの確認）。次元は`(k, k)`。
    #[test]
    fn cov_params_is_symmetric_and_diagonal_matches_std_errors_for_every_cov_type() {
        let z1: Vec<f64> = (1..=16).map(|i| i as f64).collect();
        let z2: Vec<f64> = z1.iter().map(|&v| (v * 1.7).sin() * 5.0 + v).collect();
        let noise = [
            -3.0, 2.5, -1.0, 4.0, -5.0, 6.5, -2.0, 7.0, -8.0, 3.0, -9.5, 10.0, -4.0, 11.0, -12.5,
            6.0,
        ];
        let x_endog: Vec<f64> = (0..16)
            .map(|i| z1[i] + 0.5 * z2[i] + noise[i] * (1.0 + 0.3 * z1[i]))
            .collect();
        let x_exog: Vec<f64> = (0..16).map(|i| ((i as f64) * 0.9).cos() * 3.0).collect();
        let y: Vec<f64> = (0..16)
            .map(|i| 2.0 + 1.5 * x_endog[i] + 0.5 * x_exog[i] + noise[i] * 0.6)
            .collect();
        let labels: Vec<String> = (0..16).map(|i| format!("g{}", i / 4)).collect();

        let cov_types = [
            CovType::Classical,
            CovType::Hc0,
            CovType::Hc1,
            CovType::Hc2,
            CovType::Hc3,
            CovType::Hac {
                lags: Some(2),
                time_order: row_time_order(16),
            },
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&labels)),
            },
        ];
        for cov_type in cov_types {
            let name = format!("{cov_type:?}");
            let input = IvInput::from_columns(
                &y,
                std::slice::from_ref(&x_exog),
                vec!["x_exog".to_string()],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();
            let estimator = TwoSlsEstimator::fit(input, cov_type, 0.95).unwrap();

            let cov = estimator.cov_params();
            let k = estimator.params().nrows();
            assert_eq!((cov.nrows(), cov.ncols()), (k, k), "{name}: shape");
            for i in 0..k {
                for j in 0..k {
                    let scale = (*cov.get(i, i) * *cov.get(j, j)).sqrt();
                    assert!(
                        (*cov.get(i, j) - *cov.get(j, i)).abs() <= 1e-12 * scale,
                        "{name}: cov_params not symmetric at ({i},{j})"
                    );
                }
                let se = *estimator.std_errors().get(i, 0);
                assert!(
                    (cov.get(i, i).sqrt() - se).abs() <= 1e-12 * se,
                    "{name}: sqrt(cov_params[{i}][{i}]) != std_errors[{i}]"
                );
            }
        }
    }
}
