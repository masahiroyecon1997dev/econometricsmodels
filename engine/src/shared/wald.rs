//! 傾き係数（または任意の係数部分集合）に対するWald検定（F統計量）。

use faer::Mat;
use faer::Side;
use faer::prelude::Solve;
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

use crate::shared::error::CommonError;
use crate::shared::linear_algebra::ensure_well_conditioned_symmetric_matrix;

/// 傾き係数（切片を除く`df_model`個の係数）が全てゼロという帰無仮説のロバストWald検定を行い、
/// F統計量とそのp値を返す。
///
/// `F = (β_slopes' Σ⁻¹ β_slopes) / q`（`Σ`は`cov_params`のうち傾き係数に対応する
/// `df_model × df_model`の部分行列、`q = df_model`）。`params`・`cov_params`の行/列は
/// `k_constant`が1（切片あり）なら先頭が切片（`OlsInput::from_columns`の設計行列の
/// 先頭列が定数項という規約）、0（切片なし）なら全パラメータが検定対象になる。
/// p値はF分布（自由度`(df_model, df_inference)`）の上側確率
/// （`OlsEstimator::fit`のdocコメント「F統計量も同じ方針で」参照。`cov_type=Classical`のとき
/// 古典的F検定と代数的に一致することを確認済み）。`df_inference`は通常`n-k`だが、
/// `cov_type=Cluster`のときは`G-1`になる（呼び出し元の`fit()`を参照）。
///
/// `Σ`の逆行列はCholesky分解（`Llt`）で求める。classical/HC0-3/HACでは`Σ`は
/// （`cov_params`全体の）正定値行列の主小行列であり理論上必ず正定値のため、`xtx_inverse`と
/// 同様、浮動小数点演算の丸めによる境界的な失敗に備えて`ComputationFailed`に変換している。
///
/// **`CovType::Cluster`の構造的特異性（`g <= q`）は、この関数に到達する前に
/// `fit()`冒頭のバリデーション（`validate_cluster_count_covers_slopes`、`CommonError::
/// InsufficientClustersForInference`）で弾かれる**。クラスターロバスト
/// 共分散`Ŝ = Σ_g S_g S_g'`はクラスター寄与スコアの総和がゼロ（正規方程式`X'e = 0`）に
/// なるため`rank(Ŝ) ≤ g - 1`であり、傾き係数の数`q ≥ g`なら`Σ`が構造的に特異になる。
/// `g`・`q`は入力だけから判定できるため、行列計算を待たず事前検証する方針にした。
///
/// この関数の`ensure_well_conditioned_symmetric_matrix`（`crate::shared::linear_algebra`、
/// 固有値分解ベースの相対閾値判定。系統をまたいで共有する純粋な線形代数ユーティリティ、
/// `.claude/rules/rust-style.md`「全手法で共有するロジック」参照）は、事前検証をすり抜ける
/// ケース——`g > q`だが傾き係数間の悪条件（極端なスケール差・準多重共線性等）で
/// `q×q`部分行列の条件数が倍精度の限界を超える場合——の**backstop**として残る
/// （`fit_returns_computation_failed_for_extreme_scale_difference_in_f_test`で固定）。
/// Cholesky分解（非ピボット）は数値的にほぼ特異な行列でも失敗せず桁違いに巨大な
/// F統計量を黙って返しうるため、`Llt`分解の**前**にこのチェックを置き
/// `ComputationFailed`で止める。`Llt`分解自体の`map_err`は、両方のチェックを
/// すり抜けるごく僅かな境界ケースに備えた防御的なフォールバック。
///
/// `pub(crate)`: `panel::fe::FeEstimator::fit`がFE独自に計算し直した
/// `cov_params`・`df_resid`（パネル自由度調整済み）でF検定するために再利用する
/// （`wald_test_last_columns`と同じ「サンドイッチ計算を複製しない」方針。FEは
/// `OlsEstimator`インスタンス自身の`cov_params`/`df_inference`とは異なる値を使うため
/// `wald_test_last_columns`メソッドは使えず、この下位の自由関数を直接呼ぶ、
/// `engine/src/panel/CLAUDE.md`参照）。
pub(crate) fn wald_f_test(
    params: &Mat<f64>,
    cov_params: &Mat<f64>,
    k_constant: usize,
    df_model: usize,
    df_inference: usize,
) -> Result<(f64, f64), CommonError> {
    let beta_slopes = Mat::from_fn(df_model, 1, |i, _| *params.get(i + k_constant, 0));
    let v_slopes = Mat::from_fn(df_model, df_model, |i, j| {
        *cov_params.get(i + k_constant, j + k_constant)
    });

    ensure_well_conditioned_symmetric_matrix(
        &v_slopes,
        df_model,
        "coefficient covariance submatrix for the F-test",
    )?;

    let llt = v_slopes.llt(Side::Lower).map_err(|_| {
        CommonError::ComputationFailed(
            "failed to invert coefficient covariance submatrix for the F-test".to_string(),
        )
    })?;
    let v_slopes_inv_beta = llt.solve(&beta_slopes);

    let wald: f64 = (0..df_model)
        .map(|i| (*beta_slopes.get(i, 0)) * (*v_slopes_inv_beta.get(i, 0)))
        .sum();
    let f_statistic = wald / (df_model as f64);

    let f_dist = FisherSnedecor::new(df_model as f64, df_inference as f64)
        .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
    let f_p_value = f_dist.sf(f_statistic);

    Ok((f_statistic, f_p_value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wald_f_test_keeps_precision_in_the_far_tail() {
        // `1.0 - cdf`では裾でp値が0.0に潰れる。単一の傾き（`β=10`、`V=1`）ならF=100、
        // R: `pf(100, 1, 100, lower.tail = FALSE)` = 9.90168898459409e-17。
        let params = Mat::from_fn(1, 1, |_, _| 10.0);
        let cov = Mat::from_fn(1, 1, |_, _| 1.0);
        let (f, p) = wald_f_test(&params, &cov, 0, 1, 100).unwrap();
        assert!((f - 100.0).abs() < 1e-12);
        assert!((p / 9.901_688_984_594_09e-17 - 1.0).abs() < 1e-8);

        // 傾き3個（`β=(10, 20, 30)`、`V=I`）: wald=1400、F=1400/3、F(3, 60)。
        // R: `pf(1400/3, 3, 60, lower.tail = FALSE)` = 1.59046502043898e-41
        let params3 = Mat::from_fn(3, 1, |i, _| (i as f64 + 1.0) * 10.0);
        let cov3 = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let (f3, p3) = wald_f_test(&params3, &cov3, 0, 3, 60).unwrap();
        assert!((f3 - 1400.0 / 3.0).abs() < 1e-9);
        assert!((p3 / 1.590_465_020_438_98e-41 - 1.0).abs() < 1e-8);
    }

    #[test]
    fn wald_f_test_skips_the_constant_and_uses_only_the_slope_block() {
        // params=[切片100, 3, 4]、傾きの共分散は diag(1, 4)。切片の分散（1e6）は無関係。
        // wald = 3²/1 + 4²/4 = 13、F = 13/2 = 6.5。
        let params = Mat::from_fn(3, 1, |i, _| [100.0, 3.0, 4.0][i]);
        let cov = Mat::from_fn(3, 3, |i, j| match (i, j) {
            (0, 0) => 1e6,
            (1, 1) => 1.0,
            (2, 2) => 4.0,
            _ => 0.0,
        });

        let (f, p) = wald_f_test(&params, &cov, 1, 2, 10).unwrap();

        assert!((f - 6.5).abs() < 1e-12);
        let expected_p = FisherSnedecor::new(2.0, 10.0).unwrap().sf(6.5);
        assert!((p - expected_p).abs() < 1e-15);
    }

    #[test]
    fn wald_f_test_accounts_for_off_diagonal_covariance() {
        // β=(1, 1)、Σ=[[2, 1], [1, 2]] → Σ⁻¹ = (1/3)[[2, -1], [-1, 2]]、
        // β'Σ⁻¹β = (1/3)(2 - 1 - 1 + 2) = 2/3、F = (2/3)/2 = 1/3。
        let params = Mat::from_fn(2, 1, |_, _| 1.0);
        let cov = Mat::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 1.0 });

        let (f, _) = wald_f_test(&params, &cov, 0, 2, 30).unwrap();

        assert!((f - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn wald_f_test_rejects_a_near_singular_covariance_submatrix() {
        // 対角のスケール比が1e18（倍精度の限界超え）。非ピボットCholeskyでは検出できず、
        // 固有値ベースの判定（`ensure_well_conditioned_symmetric_matrix`）で弾く。
        let params = Mat::from_fn(2, 1, |_, _| 1.0);
        let ill = Mat::from_fn(2, 2, |i, j| if i == j { [1e12, 1e-6][i] } else { 0.0 });
        assert!(matches!(
            wald_f_test(&params, &ill, 0, 2, 30),
            Err(CommonError::ComputationFailed(_))
        ));

        let zero = Mat::<f64>::zeros(2, 2);
        assert!(matches!(
            wald_f_test(&params, &zero, 0, 2, 30),
            Err(CommonError::ComputationFailed(_))
        ));
    }
}
