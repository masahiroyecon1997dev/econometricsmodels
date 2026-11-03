//! `cov_type`別の係数分散共分散行列（サンドイッチ型）と、Wald検定（F統計量）。
//!
//! 定義は`gmm.rs`のモジュールdoc「標準誤差・検定統計量（`cov_type`対応）」参照。

use faer::prelude::Solve;
use faer::{Mat, Side};
use statrs::distribution::{ChiSquared, ContinuousCDF};

use super::weight::hac_moment_covariance;
use crate::iv::common::IvError;
use crate::linear::ols::CovType;
use crate::shared::covariance::{
    cluster_correction, cluster_meat, resolve_hac_lags, time_ordering,
};
use crate::shared::error::{ClusterSetting, CommonError, MatrixKind};
use crate::shared::group_codes::GroupCodes;
use crate::shared::linear_algebra::ensure_well_conditioned_symmetric_matrix;

/// [`sandwich_covariance`]の戻り値。
pub(super) struct GmmCovariance {
    pub(super) cov_params: Mat<f64>,
    /// `cov_type=Hac`のとき実際に使われたラグ数。それ以外では`None`。
    pub(super) hac_lags_used: Option<usize>,
}

/// 独立実装のサンドイッチ型係数分散共分散行列
/// `Avar(β̂) = B⁻¹(X'ZWΩ̂WZ'X)B⁻¹`（`B = X'ZWZ'X`、`W = S_used⁻¹`は点推定に実際に使った
/// 重み、k×k）。`gmm.rs`のモジュールdoc「標準誤差・検定統計量（cov_type対応）」参照。
///
/// `df_resid`は`cov_type=Classical`の`σ̂² = Σ(êᵢ-ē)²/df_resid`にだけ使う（`fit()`は`n - k`を
/// 渡す）。HC1・クラスターの小標本補正は`n`・`k`（`x`の形）に従う。
pub(super) fn sandwich_covariance(
    cov_type: &CovType,
    z: &Mat<f64>,
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    ztz: &Mat<f64>,
    s_used: &Mat<f64>,
    df_resid: usize,
) -> Result<GmmCovariance, IvError> {
    let n = z.nrows();
    let l = z.ncols();
    let k = x.ncols();
    let mut hac_lags_used = None;

    // W = S_used⁻¹（点推定に実際に使った重み）。bread = X'ZWZ'X（`gmm_point_estimate`
    // 内の`bread`と同型だが、こちらは後続のΩ̂サンドイッチでも使い回すため明示的な
    // 逆行列として保持する（`two_sls/cov.rs`の`xtx_inverse`と同じ考え方）。
    let s_inv = invert_spd(
        s_used,
        l,
        CommonError::NearSingularMatrix {
            matrix: MatrixKind::GmmWeight,
        },
    )?;
    let zx = z.transpose() * x; // (l, k)
    let bread = zx.transpose() * &s_inv * &zx; // (k, k) = X'ZWZ'X
    let bread_inv = invert_spd(
        &bread,
        k,
        CommonError::NearSingularMatrix {
            matrix: MatrixKind::GmmBread,
        },
    )?;

    let omega_hat = match cov_type {
        CovType::Classical => {
            // 残差を中心化した標本分散を使う（`weight_type=Classical`以外では
            // GMMの一次条件が`ē=0`を保証しないため、`gmm.rs`のモジュールdoc
            // 「標準誤差・検定統計量」`classical`の項参照）。
            let mean_residual: f64 = (0..n).map(|i| *residuals.get(i, 0)).sum::<f64>() / (n as f64);
            let centered_ssr: f64 = (0..n)
                .map(|i| (*residuals.get(i, 0) - mean_residual).powi(2))
                .sum();
            let sigma2 = centered_ssr / (df_resid as f64);
            Mat::from_fn(l, l, |i, j| sigma2 * (*ztz.get(i, j)))
        }
        CovType::Hc0 => gmm_hc_omega(z, residuals, ztz, n, k, l, HcVariant::Hc0)?,
        CovType::Hc1 => gmm_hc_omega(z, residuals, ztz, n, k, l, HcVariant::Hc1)?,
        CovType::Hc2 => gmm_hc_omega(z, residuals, ztz, n, k, l, HcVariant::Hc2)?,
        CovType::Hc3 => gmm_hc_omega(z, residuals, ztz, n, k, l, HcVariant::Hc3)?,
        CovType::Hac { lags, time_order } => {
            let lags = resolve_hac_lags(*lags, n)?;
            hac_lags_used = Some(lags);
            let order = time_ordering(time_order, n);
            // Newey-West重み付け以外の小標本補正を持たないため、点推定用の
            // `hac_moment_covariance`と計算式が一致する（`gmm.rs`の
            // モジュールdoc参照。`robust_moment_covariance`/`cluster_moment_covariance`は
            // 補正が無く使い回せないのと対照的）。
            hac_moment_covariance(z, residuals, n, l, lags, &order)
        }
        CovType::Cluster { groups } => {
            // クラスター数 `g >= 2`・`g > df_model`（構造方程式の傾き係数の数）は`fit()`冒頭
            // （`validate_iv_fit_inputs`）で検証済み。
            let groups = groups.as_ref().ok_or(CommonError::MissingClusterColumn {
                setting: ClusterSetting::CovType,
            })?;
            gmm_cluster_omega(z, residuals, n, k, l, groups)
        }
    };

    let meat_z = &s_inv * &omega_hat * &s_inv; // (l, l) = WΩ̂W
    let meat = zx.transpose() * &meat_z * &zx; // (k, k) = X'ZWΩ̂WZ'X
    let cov_params = &bread_inv * &meat * &bread_inv; // (k, k)
    Ok(GmmCovariance {
        cov_params,
        hac_lags_used,
    })
}

/// 対称正定値行列`mat`（`dim`×`dim`）の逆行列を明示的に構築する。`gmm.rs`の
/// モジュールdoc「標準誤差・検定統計量（cov_type対応）」参照（`two_sls/cov.rs`の`xtx_inverse`と
/// 同じ考え方だが、`s_used`・`bread`の2箇所で使い回すため独立したヘルパーにした）。
fn invert_spd(mat: &Mat<f64>, dim: usize, failure: CommonError) -> Result<Mat<f64>, IvError> {
    let llt = mat.llt(Side::Lower).map_err(|_| failure)?;
    Ok(llt.solve(Mat::<f64>::identity(dim, dim)))
}

/// HC0〜HC3ロバストなモーメント条件の分散共分散行列（cov_type用）: `Σᵢ scaleᵢ² zᵢzᵢ'`
/// （l×l）。`two_sls/cov.rs`の`hc_cov_params`と同型の自己拡張（`X̂`→`Z`）で、レバレッジは
/// `Z`（点推定用の`ztz`をそのまま流用）から計算する——**GMMのHC2/HC3は外部参照実装での
/// 検証は不可能**（`linearmodels`の`IVGMM`にもRクロスチェックの`momentfit`にも対応する
/// 共分散推定が無い、`docs/spec/iv-spec.md`4章）。2SLSのHC2/HC3は逆にR `ivreg`+
/// `sandwich::vcovHC`で検証可能なことを実機確認済み（`docs/spec/iv-spec.md`4章）だが、
/// GMMには同等の参照が無いため対象外のまま。`gmm.rs`のモジュールdoc
/// 「標準誤差・検定統計量（cov_type対応）」参照。
///
/// **HC1の小標本補正`n/(n-k)`は`l`（全操作変数の数）ではなく`k`（構造方程式の係数の数）を
/// 使う**: 補正対象の残差`êᵢ = yᵢ - xᵢ'β̂`は常にk個の
/// パラメータで推定された構造残差であり、消費された自由度は常に`k`——`l`は単に外積を
/// 取る操作変数の本数（過剰識別なら`l > k`）に過ぎず、消費した自由度とは無関係
/// （`two_sls/cov.rs`のHC1が`X̂`の列数=k自身を使うのと同じ理由。「`X̂`→`Z`、`k`→`l`」という
/// 機械的な置換はレバレッジの計算には妥当だが、補正係数の分母にまで機械的に適用したのは
/// 誤りだった）。
fn gmm_hc_omega(
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    ztz: &Mat<f64>,
    n: usize,
    k: usize,
    l: usize,
    variant: HcVariant,
) -> Result<Mat<f64>, IvError> {
    let leverage: Option<Vec<f64>> = match variant {
        HcVariant::Hc2 | HcVariant::Hc3 => {
            // Z'Zはtheory上full column rank（点推定のW₀=(Z'Z)⁻¹が既に反転成功済み）
            // のため、ここでの特異性は理論上到達不能だが、浮動小数点演算の丸めによる
            // 境界的な失敗に備えて`Result`のまま扱う（`two_sls/cov.rs`の`xtx_inverse`と同じ
            // 防御的な扱い、`.claude/rules/rust-style.md`「unwrap/expectはプロトタイプ
            // 段階を除き避ける」）。
            let ztz_inv = invert_spd(
                ztz,
                l,
                CommonError::internal("Z'Z could not be factorized for the HC2/HC3 leverage"),
            )?;
            let zh = z * &ztz_inv; // (n, l)
            Some(
                (0..n)
                    .map(|i| (0..l).map(|j| (*zh.get(i, j)) * (*z.get(i, j))).sum())
                    .collect(),
            )
        }
        HcVariant::Hc0 | HcVariant::Hc1 => None,
    };

    let hc1_correction = ((n as f64) / ((n - k) as f64)).sqrt();

    let z_scaled = Mat::from_fn(n, l, |i, j| {
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
        scale * (*z.get(i, j))
    });

    Ok(z_scaled.transpose() * &z_scaled)
}

/// `gmm_hc_omega`専用、HCの種類（`two_sls/cov.rs`の`HcVariant`と同じ位置づけ）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HcVariant {
    Hc0,
    Hc1,
    Hc2,
    Hc3,
}

/// クラスターロバストなモーメント条件の分散共分散行列（cov_type用）:
/// `Σ_g (Σ_{i∈g} êᵢzᵢ)(Σ_{i∈g} êᵢzᵢ)' * correction`（l×l）。`two_sls.rs`の
/// `shared::covariance::cluster_cov_params`（2SLSが呼ぶ共有関数）と同型の自己拡張
/// （`X̂`→`Z`、集計は`cluster_meat`共有）。点推定用の`cluster_moment_covariance`は
/// 小標本補正が無いため使い回さない（`gmm.rs`のモジュールdoc参照）。`groups`が
/// `G>=2`であることは`validate_cluster_groups`（呼び出し元）で検証済みの前提。
///
/// 小標本補正`(G/(G-1))((n-1)/(n-k))`は`gmm_hc_omega`のHC1補正と同じ理由で`l`ではなく
/// `k`（構造方程式の係数の数）を使う（`gmm_hc_omega`のdocコメント参照）。
fn gmm_cluster_omega(
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    n: usize,
    k: usize,
    l: usize,
    groups: &GroupCodes,
) -> Mat<f64> {
    let (s_hat, n_groups) = cluster_meat(groups, l, |i, a| (*residuals.get(i, 0)) * (*z.get(i, a)));
    let correction = cluster_correction(n_groups, n, k);
    Mat::from_fn(l, l, |i, j| correction * (*s_hat.get(i, j)))
}

/// 傾き係数（切片を除く`df_model`個の係数）が全てゼロという帰無仮説のロバストWald検定
/// （χ²、`df_model`で割らない生の二次形式）を行い、統計量とp値を返す。`gmm.rs`の
/// モジュールdoc「標準誤差・検定統計量（cov_type対応）」参照——`two_sls/cov.rs`の`wald_f_test`と
/// 数式は同型だが、F分布ではなくχ²分布を使い`df_model`で割らない点が異なる
/// （`docs/spec/iv-spec.md`2章「GMMは常にロバストWald検定（χ²）とする」）。
pub(super) fn gmm_wald_chi2_test(
    params: &Mat<f64>,
    cov_params: &Mat<f64>,
    k_constant: usize,
    df_model: usize,
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

    let chi2 =
        ChiSquared::new(df_model as f64).map_err(|e| CommonError::internal(e.to_string()))?;
    let p_value = chi2.sf(wald);

    Ok((wald, p_value))
}

#[cfg(test)]
mod tests {
    use crate::iv::common::{IvError, IvEstimate, IvInput};
    use crate::iv::gmm::GmmEstimator;
    use crate::iv::gmm::GmmType;
    use crate::iv::gmm::WeightType;
    use crate::iv::gmm::test_support::*;
    use crate::linear::common::row_time_order;
    use crate::linear::ols::CovType;
    use crate::shared::error::{ClusterSetting, CommonError};
    use crate::shared::group_codes::GroupCodes;
    use faer::prelude::Solve;
    use faer::{Mat, Side};
    use statrs::distribution::{ChiSquared, ContinuousCDF};

    /// `cov_type=Classical`/`Hc0`〜`Hc3`のSEを、`GmmEstimator`とは独立に手計算した
    /// サンドイッチ公式（`gmm.rs`のモジュールdoc「標準誤差・検定統計量（cov_type
    /// 対応）」参照）と数値照合する。点推定（`weight_type=Robust`・`TwoStep`）は
    /// `fit_computes_robust_weighted_estimate_matching_manual_formula`と同じオラクルを
    /// 再利用し、そこから独立に`s_used`・最終残差を再現したうえで、cov_typeごとの
    /// `Ω̂`・サンドイッチを別コードで組み立てる。
    #[test]
    fn fit_computes_classical_and_hc_std_errors_matching_manual_sandwich_formula() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let l = 3;
        let k = 2;

        let z = Mat::from_fn(n, l, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, k, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let ztz = z.transpose() * &z;
        let ztz_inv = ztz
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(l, l));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread0 = ztx.transpose() * &ztz_inv * &ztx;
        let meat0 = ztx.transpose() * &ztz_inv * &zty;
        let beta0 = bread0
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k))
            * &meat0;
        let e0 = &y_mat - &x * &beta0;

        // weight_type=Robustの2-step: S_used = Σê₀² zᵢzᵢ'（点推定用、小標本補正無し）。
        let z_scaled0 = Mat::from_fn(n, l, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
        let s_used = z_scaled0.transpose() * &z_scaled0;
        let s_inv = s_used
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(l, l));
        let bread = ztx.transpose() * &s_inv * &ztx;
        let meat_beta = ztx.transpose() * &s_inv * &zty;
        let beta1 = bread
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k))
            * &meat_beta;
        let residuals = &y_mat - &x * &beta1;
        let bread_inv = bread
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k));

        let df_resid = n - k;

        let input_for = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let assert_se_matches = |cov_type: CovType, omega: &Mat<f64>| {
            let meat_z = &s_inv * omega * &s_inv;
            let meat = ztx.transpose() * &meat_z * &ztx;
            let cov_params = &bread_inv * &meat * &bread_inv;

            let estimator = GmmEstimator::fit(
                input_for(),
                GmmType::TwoStep {
                    weight: WeightType::Robust,
                },
                true,
                cov_type.clone(),
                0.95,
            )
            .unwrap();
            for j in 0..k {
                let expected_se = (*cov_params.get(j, j)).sqrt();
                let got_se = *estimator.std_errors().get(j, 0);
                assert!(
                    (got_se - expected_se).abs() < 1e-8,
                    "{cov_type:?} se {j}: got {got_se}, expected {expected_se}"
                );
            }
        };

        // classical: Ω̂ = σ̂²・Z'Z（σ̂² = Σ(ê-ē)²/df_resid、中心化。この専用テストは
        // weight_type=Robustを使っており、GMMの一次条件はē=0を保証しないため
        // 中心化の有無が実際に効く。`gmm.rs`のモジュールdoc「classical」の項参照）。
        let mean_residual: f64 = (0..n).map(|i| *residuals.get(i, 0)).sum::<f64>() / (n as f64);
        let centered_ssr: f64 = (0..n)
            .map(|i| (*residuals.get(i, 0) - mean_residual).powi(2))
            .sum();
        let sigma2 = centered_ssr / (df_resid as f64);
        let omega_classical = Mat::from_fn(l, l, |i, j| sigma2 * (*ztz.get(i, j)));
        assert_se_matches(CovType::Classical, &omega_classical);

        // hc0/hc1: Ω̂ = Σscaleᵢ² zᵢzᵢ'（hc1は sqrt(n/(n-k)) 補正、gmm_hc_omegaのdocコメント
        // 参照——kは構造方程式の係数の数、lではない）。
        let hc1_scale = ((n as f64) / ((n - k) as f64)).sqrt();
        for (cov_type, scale) in [(CovType::Hc0, 1.0), (CovType::Hc1, hc1_scale)] {
            let z_scaled =
                Mat::from_fn(n, l, |i, j| (*residuals.get(i, 0)) * scale * (*z.get(i, j)));
            let omega = z_scaled.transpose() * &z_scaled;
            assert_se_matches(cov_type, &omega);
        }

        // hc2/hc3: レバレッジ h_ii = zᵢ'(Z'Z)⁻¹zᵢ（Zベースの自己拡張、`gmm.rs`のモジュールdoc
        // のdocコメント参照）。
        let leverage: Vec<f64> = (0..n)
            .map(|i| {
                (0..l)
                    .map(|a| {
                        (0..l)
                            .map(|b| (*z.get(i, a)) * (*ztz_inv.get(a, b)) * (*z.get(i, b)))
                            .sum::<f64>()
                    })
                    .sum()
            })
            .collect();
        for (cov_type, is_hc3) in [(CovType::Hc2, false), (CovType::Hc3, true)] {
            let z_scaled = Mat::from_fn(n, l, |i, j| {
                let h = leverage[i];
                let scale = if is_hc3 {
                    1.0 / (1.0 - h)
                } else {
                    1.0 / (1.0 - h).sqrt()
                };
                (*residuals.get(i, 0)) * scale * (*z.get(i, j))
            });
            let omega = z_scaled.transpose() * &z_scaled;
            assert_se_matches(cov_type, &omega);
        }
    }

    /// `cov_type`と`weight_type`が同じ`fit()`で別々のグルーピングを使うとき、それぞれが
    /// 自分のコードで計算される（片方のコードをもう片方に流用していない）ことの確認。
    /// 点推定は`cov_type`に依存しないので`cov_type=Classical`のときとビット一致し、標準誤差は
    /// 分割を入れ替えると変わる。ラベルだけ付け替えた同じ分割では（ラベルの辞書順が変わり
    /// 加算順が変わるため丸め誤差の範囲で）一致する。
    #[test]
    fn fit_uses_separate_codes_for_cov_type_and_weight_type_clusters() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let weight_groups = labelled_groups(n, 4, "a");
        let cov_groups = labelled_groups(n, 5, "b");
        let fit = |weight: &[String], cov: Option<&[String]>| {
            let input = IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();
            GmmEstimator::fit(
                input,
                GmmType::TwoStep {
                    weight: WeightType::Cluster {
                        groups: Some(GroupCodes::from_labels_without_keys(weight)),
                    },
                },
                true,
                match cov {
                    Some(groups) => CovType::Cluster {
                        groups: Some(GroupCodes::from_labels_without_keys(groups)),
                    },
                    None => CovType::Classical,
                },
                0.95,
            )
            .unwrap()
        };

        let mixed = fit(&weight_groups, Some(&cov_groups));
        let classical_cov = fit(&weight_groups, None);
        let swapped = fit(&cov_groups, Some(&weight_groups));
        let same = fit(&weight_groups, Some(&weight_groups));
        let relabelled: Vec<String> = cov_groups.iter().map(|g| format!("z{g}")).collect();
        let mixed_relabelled = fit(&weight_groups, Some(&relabelled));

        for j in 0..mixed.params().nrows() {
            assert_eq!(
                mixed.params().get(j, 0).to_bits(),
                classical_cov.params().get(j, 0).to_bits()
            );
            let se = *mixed.std_errors().get(j, 0);
            assert!(
                (se - *mixed_relabelled.std_errors().get(j, 0)).abs() < 1e-10 * se,
                "relabelling the cov_type clusters must not change the standard error"
            );
            assert_ne!(
                se.to_bits(),
                same.std_errors().get(j, 0).to_bits(),
                "cov_type clusters must not be replaced by the weight_type clusters"
            );
            assert_ne!(
                se.to_bits(),
                swapped.std_errors().get(j, 0).to_bits(),
                "swapping the two groupings must change the standard error"
            );
        }
    }

    #[test]
    fn fit_computes_cluster_std_errors_matching_manual_sandwich_formula() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let l = 3;
        let k = 2;
        let groups: Vec<String> = (0..n).map(|i| format!("g{}", i / (n / 4))).collect();
        let n_groups = 4.0_f64;

        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Robust,
            },
            true,
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
            0.95,
        )
        .unwrap();

        let z = Mat::from_fn(n, l, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, k, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);
        let ztz_inv = (z.transpose() * &z)
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(l, l));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread0 = ztx.transpose() * &ztz_inv * &ztx;
        let meat0 = ztx.transpose() * &ztz_inv * &zty;
        let beta0 = bread0
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k))
            * &meat0;
        let e0 = &y_mat - &x * &beta0;

        let z_scaled0 = Mat::from_fn(n, l, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
        let s_used = z_scaled0.transpose() * &z_scaled0;
        let s_inv = s_used
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(l, l));
        let bread = ztx.transpose() * &s_inv * &ztx;
        let meat_beta = ztx.transpose() * &s_inv * &zty;
        let beta1 = bread
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k))
            * &meat_beta;
        let residuals = &y_mat - &x * &beta1;
        let bread_inv = bread
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k));

        let mut s_omega = Mat::<f64>::zeros(l, l);
        for g in ["g0", "g1", "g2", "g3"] {
            let mut s_g = [0.0_f64; 3];
            for (i, group) in groups.iter().enumerate().take(n) {
                if group == g {
                    let e = *residuals.get(i, 0);
                    for (a, s_g_a) in s_g.iter_mut().enumerate() {
                        *s_g_a += e * (*z.get(i, a));
                    }
                }
            }
            for a in 0..l {
                for b in 0..l {
                    *s_omega.get_mut(a, b) += s_g[a] * s_g[b];
                }
            }
        }
        let correction = (n_groups / (n_groups - 1.0)) * ((n as f64 - 1.0) / ((n - k) as f64));
        let omega = Mat::from_fn(l, l, |i, j| correction * (*s_omega.get(i, j)));

        let meat_z = &s_inv * &omega * &s_inv;
        let meat = ztx.transpose() * &meat_z * &ztx;
        let cov_params = &bread_inv * &meat * &bread_inv;

        for j in 0..k {
            let expected_se = (*cov_params.get(j, j)).sqrt();
            let got_se = *estimator.std_errors().get(j, 0);
            assert!(
                (got_se - expected_se).abs() < 1e-8,
                "cluster se {j}: got {got_se}, expected {expected_se}"
            );
        }
    }

    /// `cov_type=Hac`のSEを、Bartlettカーネル（lags=2）を要素ごとのループで素朴に
    /// 計算したオラクルと数値照合する。小標本補正が無いため、点推定用の
    /// `hac_moment_covariance`と同じ計算式になるはず（`gmm.rs`のモジュールdoc
    /// 参照）——本テストは`hac_moment_covariance`を呼ばず独立に再計算することで、
    /// その主張自体を検証する。
    #[test]
    fn fit_computes_hac_std_errors_matching_manual_sandwich_formula() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let l = 3;
        let k = 2;

        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Robust,
            },
            true,
            CovType::Hac {
                lags: Some(2),
                time_order: row_time_order(n),
            },
            0.95,
        )
        .unwrap();

        let z = Mat::from_fn(n, l, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, k, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);
        let ztz_inv = (z.transpose() * &z)
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(l, l));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread0 = ztx.transpose() * &ztz_inv * &ztx;
        let meat0 = ztx.transpose() * &ztz_inv * &zty;
        let beta0 = bread0
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k))
            * &meat0;
        let e0 = &y_mat - &x * &beta0;

        let z_scaled0 = Mat::from_fn(n, l, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
        let s_used = z_scaled0.transpose() * &z_scaled0;
        let s_inv = s_used
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(l, l));
        let bread = ztx.transpose() * &s_inv * &ztx;
        let meat_beta = ztx.transpose() * &s_inv * &zty;
        let beta1 = bread
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k))
            * &meat_beta;
        let residuals = &y_mat - &x * &beta1;
        let bread_inv = bread
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k));

        let ze = Mat::from_fn(n, l, |i, j| (*residuals.get(i, 0)) * (*z.get(i, j)));
        let mut omega = Mat::<f64>::zeros(l, l);
        for a in 0..l {
            for b in 0..l {
                let mut acc = 0.0;
                for t in 0..n {
                    acc += (*ze.get(t, a)) * (*ze.get(t, b));
                }
                *omega.get_mut(a, b) = acc;
            }
        }
        for lag in 1..=2usize {
            let weight = 1.0 - (lag as f64) / 3.0;
            let mut s_l = Mat::<f64>::zeros(l, l);
            for a in 0..l {
                for b in 0..l {
                    let mut acc = 0.0;
                    for t in lag..n {
                        acc += (*ze.get(t, a)) * (*ze.get(t - lag, b));
                    }
                    *s_l.get_mut(a, b) = acc;
                }
            }
            for a in 0..l {
                for b in 0..l {
                    *omega.get_mut(a, b) += weight * (*s_l.get(a, b) + *s_l.get(b, a));
                }
            }
        }

        let meat_z = &s_inv * &omega * &s_inv;
        let meat = ztx.transpose() * &meat_z * &ztx;
        let cov_params = &bread_inv * &meat * &bread_inv;

        for j in 0..k {
            let expected_se = (*cov_params.get(j, j)).sqrt();
            let got_se = *estimator.std_errors().get(j, 0);
            assert!(
                (got_se - expected_se).abs() < 1e-8,
                "hac se {j}: got {got_se}, expected {expected_se}"
            );
        }
    }

    /// `cov_type=Cluster`で`groups`未指定なら`MissingClusterColumn`
    /// （`weight_type`側の同種の検証、`fit_returns_missing_cluster_column_error_when_
    /// cluster_weight_type_has_no_groups`と対になる、こちらは`cov_type`側）。
    #[test]
    fn fit_returns_missing_cluster_column_error_when_cov_type_cluster_has_no_groups() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Robust,
            },
            true,
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

    /// `cov_type=Cluster`でクラスター数が1（`G>=2`未満）なら`InsufficientClusters`
    /// （`weight_type`側の同種の検証と対になる、こちらは`cov_type`側）。
    #[test]
    fn fit_returns_insufficient_clusters_error_when_cov_type_cluster_has_only_one_group() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let groups: Vec<String> = vec!["g0".to_string(); n];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Robust,
            },
            true,
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
            0.95,
        );
        assert!(matches!(
            result.unwrap_err(),
            IvError::Common(CommonError::InsufficientClusters { .. })
        ));
    }

    /// `cov_type=Cluster`でクラスター数`g`が傾き係数の数`q`（`k - k_constant`）以下だと、
    /// クラスター寄与スコアの総和がゼロ（一次条件）で`rank(Ŝ) ≤ g - 1`のため、ロバスト
    /// Wald（χ²）検定の`q×q`部分行列が構造的に特異になる。`fit()`冒頭
    /// （反復推定・`validate_weight_type`より前）で構造方程式の`q`を使って
    /// `CommonError::InsufficientClustersForInference`を返す（2SLSと同型。
    /// `gmm_tol`非収束等が先に起きないよう最適化前に弾く）。`x_exog=[z1]`・
    /// `x_endog=[endog1]`・切片ありで`q = k - k_constant = 3 - 1 = 2`、`g=2`（`g == q`）。
    /// `weight_type=Cluster`の重み行列`S`（l×l）が`G<l`で特異になる別軸の問題とは区別する。
    #[test]
    fn fit_returns_validation_error_when_cov_type_cluster_count_at_most_slopes() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let groups: Vec<String> = (0..y.len())
            .map(|i| if i < 8 { "a" } else { "b" }.to_string())
            .collect();
        let input = IvInput::from_columns(
            &y,
            std::slice::from_ref(&z1),
            vec!["z1".to_string()],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            std::slice::from_ref(&z2),
            vec!["z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Robust,
            },
            true,
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

    /// `cov_type=Hac`の`lags`が範囲外（負・n以上）なら`InvalidHacLags`
    /// （`weight_type`側の同種の検証と対になる、こちらは`cov_type`側）。
    #[test]
    fn fit_returns_invalid_hac_lags_error_when_cov_type_hac_lags_out_of_range() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Robust,
            },
            true,
            CovType::Hac {
                lags: Some(-1),
                time_order: row_time_order(n),
            },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::InvalidHacLags { hac_lags: -1, n })
        );
    }

    /// 説明変数が定数項のみ（`x_exog=[]`・`x_endog=[]`、傾き係数`df_model=0`）のとき、
    /// F統計量は0除算を避けてNaNになる（`two_sls/fit.rs`の
    /// `fit_sets_f_statistic_and_f_p_value_to_nan_for_const_only_model`と同じ判断）。
    #[test]
    fn fit_sets_f_statistic_and_f_p_value_to_nan_for_const_only_model() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let z = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &[],
            vec![],
            std::slice::from_ref(&z),
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Classical,
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert_eq!(estimator.df_model(), 0);
        assert!(estimator.wald_statistic().is_nan());
        assert!(estimator.wald_p_value().is_nan());
    }

    /// `cov_type`対応で追加した各getterが期待通りの次元・値を返すことを確認する
    /// （`fit_exposes_basic_metadata_getters`と対になる、こちらは`cov_type`/SE系）。
    #[test]
    fn fit_exposes_cov_type_inference_getters() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Robust,
            },
            true,
            CovType::Hc1,
            0.95,
        )
        .unwrap();

        assert_eq!(estimator.cov_type(), &CovType::Hc1);
        assert_eq!(estimator.std_errors().nrows(), 2);
        assert_eq!(estimator.test_stats().nrows(), 2);
        assert_eq!(estimator.p_values().nrows(), 2);
        assert_eq!(estimator.conf_lower().nrows(), 2);
        assert_eq!(estimator.conf_upper().nrows(), 2);
        assert_eq!(estimator.df_resid(), n - 2);
        assert_eq!(estimator.df_model(), 1);
        assert!(estimator.r_squared() <= 1.0);
        assert!(estimator.wald_statistic() >= 0.0);
        assert!((0.0..=1.0).contains(&estimator.wald_p_value()));

        for j in 0..2 {
            let coef = *estimator.params().get(j, 0);
            let se = *estimator.std_errors().get(j, 0);
            let expected_z = coef / se;
            assert!((estimator.test_stats().get(j, 0) - expected_z).abs() < 1e-10);
            assert!(*estimator.conf_lower().get(j, 0) < *estimator.conf_upper().get(j, 0));
        }
    }

    /// `gmm_wald_chi2_test`（F統計量、常にロバストWald検定・χ²）を、`GmmEstimator`とは
    /// 独立に手計算したオラクルと数値照合する。
    #[test]
    fn fit_computes_f_statistic_matching_manual_wald_chi2_formula() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Robust,
            },
            true,
            CovType::Hc1,
            0.95,
        )
        .unwrap();

        // df_model=1（const以外はendog1のみ）なので手計算は単純なスカラー計算で済む:
        // wald = β_slope² / Var(β_slope)、χ²(1)のCDFからp値を求める。
        let beta_slope = *estimator.params().get(1, 0);
        let se_slope = *estimator.std_errors().get(1, 0);
        let expected_wald = (beta_slope / se_slope).powi(2);
        let chi2 = ChiSquared::new(1.0).unwrap();
        let expected_p = 1.0 - chi2.cdf(expected_wald);

        assert!(
            (estimator.wald_statistic() - expected_wald).abs() < 1e-8,
            "got {}, expected {}",
            estimator.wald_statistic(),
            expected_wald
        );
        assert!(
            (estimator.wald_p_value() - expected_p).abs() < 1e-8,
            "got {}, expected {}",
            estimator.wald_p_value(),
            expected_p
        );
    }

    /// `OneStep`は`weight_type`を持たないため（`weight_cluster_codes`は
    /// `None`）、`cov_type=Cluster`のグループは`validate_iv_fit_inputs`の構造方程式のクラスター数の検証
    /// だけで検証され、`cov_type`から借りたコードで集計される。この経路が、点推定は`cov_type`に依らず、
    /// 標準誤差はクラスター割り当てだけに依存し（ラベルの付け替えで変わらない）、
    /// 欠落・長さ不一致のコードは`Err`になることを確認する。
    #[test]
    fn fit_with_one_step_gmm_uses_the_cov_type_cluster_groups_only() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let groups = labelled_groups(n, 5, "c");
        let fit = |cov_type: CovType| {
            let input = IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();
            GmmEstimator::fit(input, GmmType::OneStep, true, cov_type, 0.95)
        };
        let cluster = |labels: &[String]| CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(labels)),
        };

        let classical = fit(CovType::Classical).unwrap();
        let clustered = fit(cluster(&groups)).unwrap();
        let relabelled_groups: Vec<String> = groups.iter().map(|g| format!("z{g}")).collect();
        let relabelled = fit(cluster(&relabelled_groups)).unwrap();

        for j in 0..clustered.params().nrows() {
            // 点推定は`cov_type`に依存しない。
            assert_eq!(
                clustered.params().get(j, 0).to_bits(),
                classical.params().get(j, 0).to_bits()
            );
            let se = *clustered.std_errors().get(j, 0);
            assert_ne!(se.to_bits(), classical.std_errors().get(j, 0).to_bits());
            // ラベルの付け替え（コードの順序は同じ）では標準誤差は変わらない。
            assert!((se - *relabelled.std_errors().get(j, 0)).abs() < 1e-10 * se);
        }

        assert_eq!(
            fit(CovType::Cluster { groups: None }).unwrap_err(),
            IvError::Common(CommonError::MissingClusterColumn {
                setting: ClusterSetting::CovType,
            })
        );
        assert_eq!(
            fit(cluster(&groups[..n - 1])).unwrap_err(),
            IvError::Common(CommonError::ClusterDimensionMismatch {
                groups_rows: n - 1,
                n
            })
        );
    }

    /// `OneStep`（ループが一度も回らず`s_used=classical_s=σ̂²₀・Z'Z`のまま）でも、
    /// `cov_type`（`Hc1`、非classicalで補正が効くケース）のSEが正しく計算されることを、
    /// 独立手計算オラクルと数値照合する。`gmm.rs`のモジュールdoc「標準誤差・検定統計量
    /// （cov_type対応）」が主張する「`s_used`の正のスカラー倍不変性によりcov_paramsは
    /// 変わらない」という設計判断が、`TwoStep`以外の経路でも成立することを検証する。
    #[test]
    fn fit_computes_hc1_std_errors_matching_manual_formula_with_one_step_gmm() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let l = 3;
        let k = 2;

        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator =
            GmmEstimator::fit(input, GmmType::OneStep, true, CovType::Hc1, 0.95).unwrap();

        let z = Mat::from_fn(n, l, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, k, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        // OneStep: β̂はW₀=(Z'Z)⁻¹による初期推定のみ（`gmm.rs`のモジュールdoc
        // 「`GmmType`と点推定への影響」参照）。
        let ztz = z.transpose() * &z;
        let ztz_inv = ztz
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(l, l));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread = ztx.transpose() * &ztz_inv * &ztx;
        let beta = bread
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k))
            * (ztx.transpose() * &ztz_inv * &zty);
        let residuals = &y_mat - &x * &beta;

        // s_used = classical_s = σ̂²₀・Z'Z（step-0残差=最終残差、Hansen J用と同じ計算）。
        let sigma2_0: f64 =
            (0..n).map(|i| (*residuals.get(i, 0)).powi(2)).sum::<f64>() / (n as f64);
        let s_used = Mat::from_fn(l, l, |i, j| sigma2_0 * (*ztz.get(i, j)));
        let s_inv = s_used
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(l, l));
        let bread_w = ztx.transpose() * &s_inv * &ztx;
        let bread_inv = bread_w
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k));

        // Ω̂_hc1 = n/(n-k)補正込みのΣêᵢ²zᵢzᵢ'。
        let hc1_scale = ((n as f64) / ((n - k) as f64)).sqrt();
        let z_scaled = Mat::from_fn(n, l, |i, j| {
            (*residuals.get(i, 0)) * hc1_scale * (*z.get(i, j))
        });
        let omega = z_scaled.transpose() * &z_scaled;

        let meat_z = &s_inv * &omega * &s_inv;
        let meat = ztx.transpose() * &meat_z * &ztx;
        let cov_params = &bread_inv * &meat * &bread_inv;

        for j in 0..k {
            let expected_se = (*cov_params.get(j, j)).sqrt();
            let got_se = *estimator.std_errors().get(j, 0);
            assert!(
                (got_se - expected_se).abs() < 1e-8,
                "one_step hc1 se {j}: got {got_se}, expected {expected_se}"
            );
        }
    }

    /// 全`cov_type`で、`cov_params()`が対称で、対角の平方根が`std_errors()`と一致する
    /// （`std_errors`は`cov_params`の対角から作るため、公開している行列が標準誤差の元そのもの
    /// であることの確認）。`bread⁻¹ · meat · bread⁻¹`の形は丸め誤差で僅かに非対称になりうるため、
    /// 許容は両係数の分散から決まる尺度（`sqrt(V_ii·V_jj)`）の`1e-12`倍。
    #[test]
    fn cov_params_is_symmetric_and_diagonal_matches_std_errors_for_every_cov_type() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
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
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();
            let estimator = GmmEstimator::fit(
                input,
                GmmType::TwoStep {
                    weight: WeightType::Robust,
                },
                true,
                cov_type,
                0.95,
            )
            .unwrap();

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
