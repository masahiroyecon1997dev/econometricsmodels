//! 傾き係数（または任意の係数部分集合）に対するWald検定（F統計量・χ²統計量）。

use faer::Mat;
use faer::Side;
use faer::prelude::Solve;
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor};

use crate::shared::error::{CommonError, MatrixKind};
use crate::shared::linear_algebra::ensure_well_conditioned_symmetric_matrix;

/// 傾き係数（切片を除く`df_model`個の係数）が全てゼロという帰無仮説のWald二次形式
/// `β_slopes' Σ⁻¹ β_slopes`（`Σ`は`cov_params`のうち傾き係数に対応する`df_model × df_model`の
/// 部分行列）を返す。F検定（[`wald_f_test`]）も`χ²`検定も、この値から作る。`params`・
/// `cov_params`の行/列は`k_constant`が1（切片あり）なら先頭が切片、0（切片なし）なら全パラメータが
/// 検定対象になる。
///
/// `params`・`cov_params`は検定範囲（`k_constant..k_constant + df_model`）より大きくてよく、範囲外の
/// 行・列は読まない（Tobitは`cov_params`の末尾に`σ`を持つが、検定対象に含めない）。`df_model`は
/// 0より大きいこと（0だと空の行列を分解することになるので、呼び出し元で先に分岐する）。
///
/// `cov_params`は理論上対称だが、`(X'X)⁻¹ Ŝ (X'X)⁻¹`のような行列積の丸め誤差で非対称に
/// なりうる。下の`llt(Side::Lower)`は下三角しか読まないため、そのままだと片側の丸め誤差が
/// そのまま結果になる。検定対象の係数が準共線（部分行列がほぼ特異）だと誤差が条件数倍に
/// 増幅され、相対誤差1e-7に達する（RE相関の強い2列×Driscoll-Kraayのハウスマン検定で実測、
/// 上三角を使うと逆符号で同じ大きさ）。両側の平均を取ると入力の丸め誤差程度（3e-9）に戻る。
///
/// 検定対象の係数の単位（列のスケール）に依存しないよう、`Σ`ではなく相関スケール化した
/// `C = D^{-1/2} Σ D^{-1/2}`（`D = diag(Σ)`、対角が1）で悪条件の判定も二次形式も計算する。
/// `β'Σ⁻¹β = z'C⁻¹z`（`z = D^{-1/2}β`）は厳密な演算では同じ値で、列を定数倍して単位を
/// 変えても`C`と`z`は変わらない。`Σ`の固有値で判定すると、単位を変えるだけで列間の分散の比が
/// `1/(k·ε)`（約2e15）を超え、係数も標準誤差も計算できるデータが失敗した。`ε`・`k`は
/// `ensure_well_conditioned_symmetric_matrix`の閾値（`k·ε·max|固有値|`）をそのまま使うが、
/// 対象が`C`なので閾値は単位に依らない（`C`の固有値の和は`df_model`で、最大固有値は
/// 1以上`df_model`以下）。
///
/// `Σ`の対角が正の有限値でなければ（ゼロ行列・負の分散・非有限値）`C`が作れず、
/// そのまま`NearSingularMatrix`にする。`C`の最小固有値が閾値以下なら、スケールを揃えても
/// ほぼ共線な係数の組があるため、同じく`NearSingularMatrix`で止める。
///
/// `C`の逆行列はCholesky分解（`Llt`）で求める。classical/HC0-3/HACでは`Σ`は
/// （`cov_params`全体の）正定値行列の主小行列であり理論上必ず正定値のため、`xtx_inverse`と
/// 同様、浮動小数点演算の丸めによる境界的な失敗に備えて`NearSingularMatrix`に変換している。
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
/// ケース——`g > q`だが傾き係数間の準多重共線性で`C`が倍精度の限界を超えてほぼ特異になる
/// 場合——の**backstop**として残る（`wald_f_test_rejects_a_near_singular_covariance_submatrix`で
/// 固定）。Cholesky分解（非ピボット）は数値的にほぼ特異な行列でも失敗せず桁違いに巨大な統計量を
/// 黙って返しうるため、`Llt`分解の**前**にこのチェックを置き`NearSingularMatrix`で止める。
/// 固有値の判定は絶対値で行うため、桁落ちで`C`が正定値でなくなった場合（`hc1`・`cluster`の
/// サンドイッチ共分散で傾きの相関が極端に高いとき。相関が1を超える）は、この判定を通って
/// `Llt`分解の`map_err`が止める。
pub(crate) fn wald_quadratic_form(
    params: &Mat<f64>,
    cov_params: &Mat<f64>,
    k_constant: usize,
    df_model: usize,
) -> Result<f64, CommonError> {
    let near_singular = || CommonError::NearSingularMatrix {
        matrix: MatrixKind::CoefficientCovariance,
    };
    let v_slopes = |i: usize, j: usize| {
        0.5 * (*cov_params.get(i + k_constant, j + k_constant)
            + *cov_params.get(j + k_constant, i + k_constant))
    };

    // 標準偏差`sd_i = √Σ_ii`。単位を変えても`β_i/sd_i`と`C`は変わらない。非対角・係数の
    // 非有限値は対角の検査ではわからないので、`C`と`z`を作った後にも確かめる。
    let mut sds = Vec::with_capacity(df_model);
    for i in 0..df_model {
        let variance = v_slopes(i, i);
        if !(variance.is_finite() && variance > 0.0) {
            return Err(near_singular());
        }
        sds.push(variance.sqrt());
    }
    let correlation = Mat::from_fn(df_model, df_model, |i, j| {
        if i == j {
            1.0
        } else {
            // 積`sd_i·sd_j`は分散が極端に小さいと0やサブノーマルになるので、順に割る。
            v_slopes(i, j) / sds[i] / sds[j]
        }
    });
    let z = Mat::from_fn(df_model, 1, |i, _| *params.get(i + k_constant, 0) / sds[i]);
    let is_finite = (0..df_model).all(|i| {
        (0..df_model).all(|j| correlation.get(i, j).is_finite()) && z.get(i, 0).is_finite()
    });
    if !is_finite {
        return Err(near_singular());
    }

    ensure_well_conditioned_symmetric_matrix(
        &correlation,
        df_model,
        MatrixKind::CoefficientCovariance,
    )?;

    let llt = correlation.llt(Side::Lower).map_err(|_| near_singular())?;
    let correlation_inv_z = llt.solve(&z);

    Ok((0..df_model)
        .map(|i| (*z.get(i, 0)) * (*correlation_inv_z.get(i, 0)))
        .sum())
}

/// 傾き係数（または任意の係数部分集合）が全てゼロという帰無仮説のロバストWald検定を行い、
/// F統計量とそのp値を返す。
///
/// `F = W / q`（`W`は[`wald_quadratic_form`]の値、`q = df_model`）。p値はF分布（自由度
/// `(df_model, df_inference)`）の上側確率（`OlsEstimator::fit`のdocコメント「F統計量も
/// 同じ方針で」参照。`cov_type=Classical`のとき古典的F検定と代数的に一致することを確認済み）。
/// `df_inference`は通常`n-k`だが、`cov_type=Cluster`のときは`G-1`になる（呼び出し元の
/// `fit()`を参照）。
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
    let wald = wald_quadratic_form(params, cov_params, k_constant, df_model)?;
    let f_statistic = wald / (df_model as f64);

    let f_dist = FisherSnedecor::new(df_model as f64, df_inference as f64)
        .map_err(|e| CommonError::internal(e.to_string()))?;
    let f_p_value = f_dist.sf(f_statistic);

    Ok((f_statistic, f_p_value))
}

/// 傾き係数（または任意の係数部分集合）が全てゼロという帰無仮説のWald検定（χ²）を行い、
/// 統計量（`df_model`で割らない生の二次形式）とp値を返す。
///
/// 検定の形は[`wald_f_test`]と同じで、検定分布だけが`χ²(df_model)`に変わる
/// （残差自由度を持たない手法——GMM・Tobitのような漸近的な検定——が使う）。
/// `df_model`は0より大きいこと（`ChiSquared::new`が失敗するので呼び出し元で
/// 傾き係数が無いケースを先に分岐する）。
pub(crate) fn wald_chi2_test(
    params: &Mat<f64>,
    cov_params: &Mat<f64>,
    k_constant: usize,
    df_model: usize,
) -> Result<(f64, f64), CommonError> {
    let wald = wald_quadratic_form(params, cov_params, k_constant, df_model)?;

    let chi2 =
        ChiSquared::new(df_model as f64).map_err(|e| CommonError::internal(e.to_string()))?;
    let p_value = chi2.sf(wald);

    Ok((wald, p_value))
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
    fn wald_f_test_symmetrizes_a_slightly_asymmetric_covariance() {
        // 丸め誤差で非対称になった共分散（下三角2.8・上三角3.2）は、両側の平均
        // Σ=[[4, 3], [3, 4]]として扱う。Σ⁻¹ = (1/7)[[4, -3], [-3, 4]]、β=(1, 2)で
        // β'Σ⁻¹β = (4 - 12 + 16)/7 = 8/7、F = 4/7。下三角だけを使うと
        // (4 - 11.2 + 16)/8.16 ≠ 8/7となり、上三角だけでも別の値になる。
        let params = Mat::from_fn(2, 1, |i, _| [1.0, 2.0][i]);
        let cov = Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) | (1, 1) => 4.0,
            (0, 1) => 3.2,
            _ => 2.8,
        });

        let (f, _) = wald_f_test(&params, &cov, 0, 2, 30).unwrap();

        assert!((f - 4.0 / 7.0).abs() < 1e-14);
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
    fn wald_f_test_is_invariant_to_the_units_of_the_regressors() {
        // 対角のスケール比が1e18（倍精度の閾値`1/(k·ε)`を超える）でも、相関は0なので
        // 良条件。二次形式は`1/1e12 + 1/1e-6`ではなく、単位を揃えた値と同じになる。
        // β=(1, 1)、Σ=diag(1e12, 1e-6) → wald = 1e-12 + 1e6、F = wald/2。
        let params = Mat::from_fn(2, 1, |_, _| 1.0);
        let ill = Mat::from_fn(2, 2, |i, j| if i == j { [1e12, 1e-6][i] } else { 0.0 });
        let (f, _) = wald_f_test(&params, &ill, 0, 2, 30).unwrap();
        assert!((f / ((1e-12 + 1e6) / 2.0) - 1.0).abs() < 1e-14);

        // 相関のある共分散でも、列の単位を`c`倍するとβ・Σは`c`倍・`c²`倍になり、統計量は
        // 変わらない。2列目だけ1e-9倍・1e9倍して比べる。
        let base_params = Mat::from_fn(3, 1, |i, _| [0.7, 1.3, -0.4][i]);
        let base_cov = Mat::from_fn(3, 3, |i, j| match (i, j) {
            (0, 0) => 2.0,
            (1, 1) => 3.0,
            (2, 2) => 1.5,
            (0, 1) | (1, 0) => 0.9,
            (0, 2) | (2, 0) => -0.3,
            _ => 0.6,
        });
        let (f_base, p_base) = wald_f_test(&base_params, &base_cov, 0, 3, 40).unwrap();
        for scale in [1e-9, 1e-6, 1e6, 1e9] {
            let unit = |i: usize| if i == 1 { scale } else { 1.0 };
            let params = Mat::from_fn(3, 1, |i, _| *base_params.get(i, 0) / unit(i));
            let cov = Mat::from_fn(3, 3, |i, j| *base_cov.get(i, j) / (unit(i) * unit(j)));
            let (f, p) = wald_f_test(&params, &cov, 0, 3, 40).unwrap();
            assert!((f / f_base - 1.0).abs() < 1e-12, "scale={scale}");
            assert!((p / p_base - 1.0).abs() < 1e-10, "scale={scale}");
        }
    }

    #[test]
    fn wald_f_test_rejects_a_near_singular_covariance_submatrix() {
        let params = Mat::from_fn(2, 1, |_, _| 1.0);

        // ゼロ行列は対角が正でないため相関行列が作れない。
        let zero = Mat::<f64>::zeros(2, 2);
        assert!(matches!(
            wald_f_test(&params, &zero, 0, 2, 30),
            Err(CommonError::NearSingularMatrix { .. })
        ));

        // 相関が1（第2列は第1列の定数倍）。スケールが極端に異なっていても、単位を揃えると
        // 完全に共線なので弾く（対角の比1e18・相関1）。
        let collinear = Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 1e12,
            (1, 1) => 1e-6,
            _ => 1e3,
        });
        assert!(matches!(
            wald_f_test(&params, &collinear, 0, 2, 30),
            Err(CommonError::NearSingularMatrix { .. })
        ));

        // 単位を揃えても相関が`1 - 1e-16`（倍精度の限界）なら弾く。
        let nearly = Mat::from_fn(2, 2, |i, j| if i == j { 4.0 } else { 4.0 * (1.0 - 1e-16) });
        assert!(matches!(
            wald_f_test(&params, &nearly, 0, 2, 30),
            Err(CommonError::NearSingularMatrix { .. })
        ));

        // 負・非有限の分散も相関に直せない。
        let negative = Mat::from_fn(2, 2, |i, j| if i == j { [1.0, -1.0][i] } else { 0.0 });
        assert!(matches!(
            wald_f_test(&params, &negative, 0, 2, 30),
            Err(CommonError::NearSingularMatrix { .. })
        ));
        let nan = Mat::from_fn(2, 2, |i, j| if i == j { [1.0, f64::NAN][i] } else { 0.0 });
        assert!(matches!(
            wald_f_test(&params, &nan, 0, 2, 30),
            Err(CommonError::NearSingularMatrix { .. })
        ));
    }

    #[test]
    fn wald_f_test_accepts_a_highly_correlated_but_resolvable_covariance() {
        // 相関`r = 1 - 1e-10`（条件数約1e10）は倍精度の限界（約9e-16）よりずっと内側なので通す。
        // 2×2の相関行列`[[1, r], [r, 1]]`の二次形式は`(z1² - 2r·z1·z2 + z2²)/(1 - r²)`。
        // 分散は単位を変えた値（1e8・1e-8）にして、相関スケール化も同時に確かめる。
        let r = 1.0 - 1e-10;
        let (sd1, sd2) = (1e4_f64, 1e-4_f64);
        let cov = Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => sd1 * sd1,
            (1, 1) => sd2 * sd2,
            _ => r * sd1 * sd2,
        });
        let params = Mat::from_fn(2, 1, |i, _| [3.0 * sd1, 1.0 * sd2][i]);
        let (z1, z2) = (3.0, 1.0);
        let expected = (z1 * z1 - 2.0 * r * z1 * z2 + z2 * z2) / (1.0 - r * r);
        let (f, _) = wald_f_test(&params, &cov, 0, 2, 30).unwrap();
        assert!((f * 2.0 / expected - 1.0).abs() < 1e-5, "{f} {expected}");

        // 3個の傾きのうち1組だけがほぼ共線（`r = 1 - 1e-13`、倍精度の限界の内側）でも通る。
        let r3 = 1.0 - 1e-13;
        let cov3 = Mat::from_fn(3, 3, |i, j| match (i, j) {
            (0, 1) | (1, 0) => r3,
            (i, j) if i == j => 1.0,
            _ => 0.0,
        });
        let params3 = Mat::from_fn(3, 1, |_, _| 1.0);
        assert!(wald_f_test(&params3, &cov3, 0, 3, 30).is_ok());
    }

    #[test]
    fn wald_f_test_rejects_non_finite_entries_outside_the_diagonal() {
        let params = Mat::from_fn(2, 1, |_, _| 1.0);
        let off_diagonal_nan = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { f64::NAN });
        assert!(matches!(
            wald_f_test(&params, &off_diagonal_nan, 0, 2, 30),
            Err(CommonError::NearSingularMatrix { .. })
        ));
        let cov = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let nan_params = Mat::from_fn(2, 1, |i, _| [1.0, f64::NAN][i]);
        assert!(matches!(
            wald_f_test(&nan_params, &cov, 0, 2, 30),
            Err(CommonError::NearSingularMatrix { .. })
        ));
    }

    #[test]
    fn wald_chi2_test_keeps_precision_in_the_far_tail() {
        // `1.0 - cdf`では裾でp値が0.0に潰れる。傾き3個（`β=(10, 10, 10)`、`V=I`）で
        // wald=300。R: `pchisq(300, 3, lower.tail = FALSE)` = 9.94875834632771e-65。
        let params = Mat::from_fn(4, 1, |i, _| [0.0, 10.0, 10.0, 10.0][i]);
        let cov = Mat::from_fn(4, 4, |i, j| if i == j { 1.0 } else { 0.0 });
        let (wald, p) = wald_chi2_test(&params, &cov, 1, 3).unwrap();
        assert!((wald - 300.0).abs() < 1e-9);
        assert!((p / 9.948_758_346_327_71e-65 - 1.0).abs() < 1e-8);
    }

    #[test]
    fn wald_chi2_test_is_the_f_test_statistic_times_the_slope_count() {
        // 同じ二次形式から作るので、χ²統計量は`q × F`に（丸めの範囲で）一致する。
        let params = Mat::from_fn(2, 1, |i, _| [1.0, 2.0][i]);
        let cov = Mat::from_fn(2, 2, |i, j| if i == j { 4.0 } else { 3.0 });
        let (f, _) = wald_f_test(&params, &cov, 0, 2, 30).unwrap();
        let (wald, _) = wald_chi2_test(&params, &cov, 0, 2).unwrap();
        assert!((wald - 2.0 * f).abs() < 1e-14);
    }

    #[test]
    fn wald_chi2_test_symmetrizes_a_slightly_asymmetric_covariance() {
        // `wald_f_test`と同じ入力（下三角2.8・上三角3.2）。β'Σ⁻¹β = 8/7（Σ=[[4,3],[3,4]]）。
        let params = Mat::from_fn(2, 1, |i, _| [1.0, 2.0][i]);
        let cov = Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) | (1, 1) => 4.0,
            (0, 1) => 3.2,
            _ => 2.8,
        });
        let (wald, _) = wald_chi2_test(&params, &cov, 0, 2).unwrap();
        assert!((wald - 8.0 / 7.0).abs() < 1e-14);
    }

    #[test]
    fn wald_chi2_test_is_invariant_to_the_units_and_rejects_a_near_singular_submatrix() {
        let params = Mat::from_fn(2, 1, |_, _| 1.0);
        let ill = Mat::from_fn(2, 2, |i, j| if i == j { [1e12, 1e-6][i] } else { 0.0 });
        let (wald, _) = wald_chi2_test(&params, &ill, 0, 2).unwrap();
        assert!((wald / (1e-12 + 1e6) - 1.0).abs() < 1e-14);

        let collinear = Mat::from_fn(2, 2, |_, _| 4.0);
        assert!(matches!(
            wald_chi2_test(&params, &collinear, 0, 2),
            Err(CommonError::NearSingularMatrix { .. })
        ));
    }
}
