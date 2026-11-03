//! 設計行列の列の標準化（`ColumnScale`）と、係数・共分散の逆変換。

use faer::Mat;

/// 設計行列の列ごとの標準化スケール（標準偏差と、切片ありのときのみ平均）。
///
/// 切片ありのモデルでは`x̃ⱼ = (xⱼ - x̄ⱼ)/stdⱼ`（平均0・分散1）、切片なしのモデルでは
/// `x̃ⱼ = xⱼ/stdⱼ`（分散1のみ）に標準化する。切片列は無変換。標準偏差0の列（定数列）は`std`を`1.0`の
/// ままにするので、切片なしでは無変換、切片ありでは平均を引いて全ゼロ列になる（切片と共線なので
/// `checked_design_matrix_qr`が`SingularDesignMatrix`で弾く）。
///
/// **切片ありのとき平均を引く理由**: 平均を引かないと、平均が標準偏差に比べて大きい列
/// （例: Wooldridge appleの`ecoprc`は平均÷標準偏差が3.7）が切片列とほぼ同じ方向を向き、
/// 標準化後のHessianが悪条件になる（appleで条件数約7,300、平均を引くと約14）。悪条件のまま
/// だとBFGS/L-BFGSの反復数が桁違いに増える（appleのlbfgsが95反復、平均を引くと19反復。
/// newton・bfgsはほぼ変わらない）。解きやすい問題に直してからソルバーに渡す方針で、
/// ソルバーごとの補正（`FaerBfgs`の初期逆Hessian等）に頼らない。
///
/// **切片なしのときは引かない理由**: 平均を引く操作は「切片が平均分のズレを吸収する」ことで
/// 成立する再パラメータ化のため、吸収先の切片が無いと逆変換が成立しない。切片なしでは
/// 切片列との共線性という上の原因自体が無いので、スケーリングのみで足りる
/// （`docs/spec/nonlinear-common.md`1.3節「スケール依存への対処（標準化）」参照）。
///
/// 標準化空間のパラメータ`θ̃`と元のスケールの`θ`の対応（`x̄₀=0`・`std₀=1`、切片列は列0）:
///
/// - 傾き: `θⱼ = θ̃ⱼ/stdⱼ`
/// - 切片: `θ₀ = θ̃₀ - Σ_{j≥1} x̄ⱼθⱼ`（切片ありのときのみ。`x̃'θ̃ = x'θ`から導く）
#[derive(Debug, Clone)]
pub struct ColumnScale {
    /// 列ごとの標準偏差。切片列（`has_intercept=true`の先頭列）は`1.0`のまま
    /// （スケーリング対象外）。標準偏差が0の列（定数列）も`1.0`のまま扱う（0除算回避。
    /// 切片ありでは`means`が引かれて全ゼロ列になる、`ColumnScale`のdocコメント参照）。
    stds: Vec<f64>,
    /// 列ごとの平均。切片ありのとき、切片列以外は元の列の平均。切片列と切片なしの
    /// モデルは全列`0.0`（平均を引かない）。
    means: Vec<f64>,
    /// 先頭列が切片列か。切片の逆変換（`destandardize_params`）で使う。
    has_intercept: bool,
}

impl ColumnScale {
    /// `x`の各列の平均・標準偏差から標準化スケールを求める（切片列は除外、標準偏差0の列は
    /// `std`を`1.0`のままにする）。`has_intercept`のときのみ平均を保持する。
    /// `has_intercept`のときは列0が切片列であることを前提にする（設計行列の規約）。
    /// 設計行列を実際に標準化するのは`standardize`（Tobitのように、`x`以外（`y`等）の
    /// スケールも同時に決めたい呼び出し側が、スケールの計算と行列への適用を分けて呼べるようにしている）。
    pub fn fit(x: &Mat<f64>, has_intercept: bool) -> Self {
        let n = x.nrows();
        let k = x.ncols();
        let start = usize::from(has_intercept);

        let mut stds = vec![1.0; k];
        let mut means = vec![0.0; k];
        for j in start..k {
            let mean: f64 = (0..n).map(|i| *x.get(i, j)).sum::<f64>() / n as f64;
            let var: f64 = (0..n).map(|i| (*x.get(i, j) - mean).powi(2)).sum::<f64>() / n as f64;
            let sd = var.sqrt();
            if sd > 0.0 {
                stds[j] = sd;
            }
            if has_intercept {
                means[j] = mean;
            }
        }
        Self {
            stds,
            means,
            has_intercept,
        }
    }

    /// 設計行列`x`を標準化する: `x̃ᵢⱼ = (xᵢⱼ - x̄ⱼ)/stdⱼ`（切片なしでは`x̄ⱼ=0`、切片列は無変換）。
    /// `fit`で求めたのと同じ`x`（または同じ列構成の設計行列）に適用する。
    pub fn standardize(&self, x: &Mat<f64>) -> Mat<f64> {
        debug_assert_eq!(
            x.ncols(),
            self.stds.len(),
            "pass a design matrix with the same columns as `fit`"
        );
        Mat::from_fn(x.nrows(), x.ncols(), |i, j| {
            (*x.get(i, j) - self.means[j]) / self.stds[j]
        })
    }

    /// 列ごとの標準偏差（`standardize_columns`のdocコメント参照）。分離検出
    /// （`nonlinear::separation`）が、標準化後の行列を複製せず元の`x`から行を都度計算するために使う。
    ///
    /// 分離の有無は切片込みの再パラメータ化（平均を引く操作）で変わらないため、分離検出は
    /// 平均を引かない標準偏差だけの行のまま使い続ける（`means`は渡さない）。
    pub fn stds(&self) -> &[f64] {
        &self.stds
    }

    /// 元のスケールのパラメータ`θ`を標準化空間の`θ̃`へ変換する（`destandardize_params`の
    /// 逆変換。テストで標準化空間の値を再現するために使う）。
    #[cfg(test)]
    pub fn standardize_params(&self, params: &[f64]) -> Vec<f64> {
        let mut out: Vec<f64> = params.iter().zip(&self.stds).map(|(p, s)| p * s).collect();
        if self.has_intercept {
            let shift: f64 = params
                .iter()
                .zip(&self.means)
                .skip(1)
                .map(|(p, m)| p * m)
                .sum();
            if let Some(intercept) = out.first_mut() {
                *intercept += shift;
            }
        }
        out
    }

    /// 標準化空間のパラメータから元のスケールのパラメータへの線形変換の行列`J`
    /// （`θ = Jθ̃`、`k×k`）。`J = diag(1/stds)`で、切片ありのときのみ第0行の
    /// 傾き成分に`J[0][m] = -x̄ₘ/stdₘ`が入る（`destandardize_params`の式の係数）。
    pub fn param_jacobian(&self) -> Mat<f64> {
        let k = self.stds.len();
        Mat::from_fn(k, k, |i, m| {
            if i == m {
                1.0 / self.stds[i]
            } else if self.has_intercept && i == 0 {
                -self.means[m] / self.stds[m]
            } else {
                0.0
            }
        })
    }
}

/// 設計行列`x`の各列を標準化する（切片列は除外）。切片ありのときは平均も引く
/// （`ColumnScale`のdocコメント参照）。`ColumnScale::fit`と`ColumnScale::standardize`の合成。
/// 収束後は`destandardize_params`で元のスケールのパラメータへ逆変換する。
pub fn standardize_columns(x: &Mat<f64>, has_intercept: bool) -> (Mat<f64>, ColumnScale) {
    let scale = ColumnScale::fit(x, has_intercept);
    let x_std = scale.standardize(x);
    (x_std, scale)
}

/// `standardize_columns`で標準化した空間で最適化した`params_std`を、元のスケールの
/// パラメータへ逆変換する（`ColumnScale`のdocコメントの式: 傾き`θⱼ=θ̃ⱼ/stdⱼ`、
/// 切片ありのとき切片`θ₀=θ̃₀-Σ_{j≥1}x̄ⱼθⱼ`）。
pub fn destandardize_params(params_std: &[f64], scale: &ColumnScale) -> Vec<f64> {
    let mut out: Vec<f64> = params_std
        .iter()
        .zip(scale.stds.iter())
        .map(|(p, s)| p / s)
        .collect();
    if scale.has_intercept {
        // 確定した傾き係数（`out[1..]`）を使って切片を補正する。
        let shift: f64 = out
            .iter()
            .zip(&scale.means)
            .skip(1)
            .map(|(b, m)| b * m)
            .sum();
        if let Some(intercept) = out.first_mut() {
            *intercept -= shift;
        }
    }
    out
}

/// `standardize_columns`で標準化した空間で評価した係数分散共分散行列`cov_std`
/// （`observed_information_cov_params`等の`cov_type`共通行列演算の戻り値）を、
/// 元のスケールの係数分散共分散行列へ逆変換する。
///
/// `θ_orig = Jθ_std`（`J`は`ColumnScale::param_jacobian`、`destandardize_params`と同じ
/// 線形変換）のとき、対数尤度を`θ_std`の関数とみなすと連鎖律により
/// `H_std = JᵀH_origJ`（`H`はHessian）が成り立つ。分散共分散行列はHessianの逆行列に
/// 比例する（`Σ = -H⁻¹`等）ため、`J`が可逆なら
/// `Σ_orig = -H_orig⁻¹ = -(J⁻ᵀH_stdJ⁻¹)⁻¹ = J(-H_std⁻¹)Jᵀ = J Σ_std Jᵀ`。
/// この関係はOPG（`Σᵢsᵢsᵢ'`の逆行列）・サンドイッチ・クラスターの各`cov_type`でも
/// 同様に成り立つ（いずれも`H⁻¹`を両側から掛ける、または`H⁻¹`の逆行列を取る形の式のため。
/// スコアも`s_std = Jᵀ s_orig`と同じ連鎖律で変換される）。
/// 平均を引かない（切片なし）場合は`J`が対角になり、要素ごとに
/// `Σ_orig[i,j] = Σ_std[i,j] / (stds[i]*stds[j])`に帰着する。
pub fn destandardize_cov_params(cov_std: &Mat<f64>, scale: &ColumnScale) -> Mat<f64> {
    let j = scale.param_jacobian();
    &j * cov_std * j.transpose()
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn standardize_columns_scales_to_unit_variance_and_leaves_intercept_untouched() {
        let x = Mat::from_fn(4, 3, |i, j| {
            if j == 0 {
                1.0
            } else if j == 1 {
                [10.0, 20.0, 30.0, 40.0][i]
            } else {
                [1.0, 2.0, 3.0, 4.0][i]
            }
        });

        let (x_std, scale) = standardize_columns(&x, true);

        for i in 0..4 {
            assert_eq!(*x_std.get(i, 0), 1.0);
        }
        // 切片列以外の標準化後の列は平均0（切片あり）・分散1。ここでは分散（平均まわりの2次モーメント）で検証する
        // （2乗の平均=E[x²]ではなくVar(x)=E[x²]-E[x]²が1になるはず）。
        let variance = |col: usize| -> f64 {
            let values: Vec<f64> = (0..4).map(|i| *x_std.get(i, col)).collect();
            let mean = values.iter().sum::<f64>() / 4.0;
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 4.0
        };
        assert!((variance(1) - 1.0).abs() < 1e-9);
        assert!((variance(2) - 1.0).abs() < 1e-9);
        assert_eq!(scale.stds[0], 1.0);
    }

    #[test]
    fn standardize_and_destandardize_round_trips_without_intercept() {
        let x = Mat::from_fn(4, 2, |i, j| {
            if j == 0 {
                [10.0, 20.0, 30.0, 40.0][i]
            } else {
                [100.0, 200.0, 150.0, 50.0][i]
            }
        });

        let (_, scale) = standardize_columns(&x, false);
        let params_std = vec![6.0, -3.0];
        let params_orig = destandardize_params(&params_std, &scale);

        // 線形予測子 x_std' * params_std == x_orig' * params_orig が成立するはず
        for i in 0..4 {
            let pred_std: f64 = (0..2)
                .map(|j| (*x.get(i, j) / scale.stds[j]) * params_std[j])
                .sum();
            let pred_orig: f64 = (0..2).map(|j| *x.get(i, j) * params_orig[j]).sum();
            assert!((pred_std - pred_orig).abs() < 1e-9);
        }
    }

    #[test]
    fn standardize_and_destandardize_round_trips_with_intercept() {
        let x = Mat::from_fn(4, 3, |i, j| {
            if j == 0 {
                1.0
            } else if j == 1 {
                [10.0, 20.0, 30.0, 40.0][i]
            } else {
                [100.0, 200.0, 150.0, 50.0][i]
            }
        });

        let (x_std, scale) = standardize_columns(&x, true);
        let params_std = vec![1.5, 6.0, -3.0];
        let params_orig = destandardize_params(&params_std, &scale);

        // 切片以外の標準化後の列は平均0・分散1（平均を引くのは切片ありのときのみ）。
        for j in 1..3 {
            let mean = (0..4).map(|i| *x_std.get(i, j)).sum::<f64>() / 4.0;
            let var = (0..4)
                .map(|i| (*x_std.get(i, j) - mean).powi(2))
                .sum::<f64>()
                / 4.0;
            assert!(mean.abs() < 1e-12, "col {j} mean={mean}");
            assert!((var - 1.0).abs() < 1e-12, "col {j} var={var}");
        }
        // 切片列は無変換。
        for i in 0..4 {
            assert_eq!(*x_std.get(i, 0), 1.0);
        }

        // 傾きは`θⱼ=θ̃ⱼ/stdⱼ`、切片は`θ₀=θ̃₀-Σx̄ⱼθⱼ`（手計算した式）。標準化空間と元の
        // スケールで線形予測子が一致することも確認する。
        for j in 1..3 {
            assert!((params_orig[j] - params_std[j] / scale.stds[j]).abs() < 1e-12);
        }
        let expected_intercept =
            params_std[0] - (1..3).map(|j| scale.means[j] * params_orig[j]).sum::<f64>();
        assert!((params_orig[0] - expected_intercept).abs() < 1e-12);
        for i in 0..4 {
            let pred_std: f64 = (0..3).map(|j| *x_std.get(i, j) * params_std[j]).sum();
            let pred_orig: f64 = (0..3).map(|j| *x.get(i, j) * params_orig[j]).sum();
            assert!((pred_std - pred_orig).abs() < 1e-9);
        }

        // `standardize_params`は`destandardize_params`の逆変換。
        let round_trip = scale.standardize_params(&params_orig);
        for (a, b) in round_trip.iter().zip(&params_std) {
            assert!((a - b).abs() < 1e-12, "{round_trip:?} vs {params_std:?}");
        }
    }

    #[test]
    fn standardize_columns_without_intercept_does_not_subtract_the_mean() {
        // 切片なしでは吸収先の切片が無いため平均を引かない（`x/std`のみ）。`means`は全列0.0で、
        // 逆変換（`destandardize_params`・`destandardize_cov_params`）も対角（`/std`）のまま。
        let x = Mat::from_fn(4, 2, |i, j| {
            if j == 0 {
                [10.0, 20.0, 30.0, 40.0][i]
            } else {
                [100.0, 200.0, 150.0, 50.0][i]
            }
        });
        let (x_std, scale) = standardize_columns(&x, false);

        assert!(scale.means.iter().all(|m| *m == 0.0));
        for i in 0..4 {
            for j in 0..2 {
                assert_eq!(*x_std.get(i, j), *x.get(i, j) / scale.stds[j]);
            }
        }
        let cov_std = Mat::from_fn(2, 2, |i, j| [[4.0, 6.0], [6.0, 9.0]][i][j]);
        let cov_orig = destandardize_cov_params(&cov_std, &scale);
        for i in 0..2 {
            for j in 0..2 {
                let expected = *cov_std.get(i, j) / (scale.stds[i] * scale.stds[j]);
                assert!((*cov_orig.get(i, j) - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn standardize_columns_turns_a_constant_column_into_an_all_zero_column_with_intercept() {
        // 標準偏差0の列は`std=1.0`のままで、切片ありでは平均が引かれて全ゼロ列になる
        // （切片と共線。`fit()`の列ピボットQRランクチェックが弾く）。切片なしでは無変換。
        let x = Mat::from_fn(4, 3, |i, j| match j {
            0 => 1.0,
            1 => [1.0, 2.0, 3.0, 4.0][i],
            _ => 0.1,
        });

        let (x_std, scale) = standardize_columns(&x, true);
        assert_eq!(scale.stds[2], 1.0);
        assert_eq!(scale.means[2], 0.1);
        for i in 0..4 {
            assert_eq!(*x_std.get(i, 2), 0.0);
        }

        let (x_std_no_intercept, scale_no_intercept) = standardize_columns(&x, false);
        assert_eq!(scale_no_intercept.stds[2], 1.0);
        for i in 0..4 {
            assert_eq!(*x_std_no_intercept.get(i, 2), 0.1);
        }
    }

    #[test]
    fn column_scale_fit_then_standardize_applies_the_fitted_mean_and_std_to_other_rows() {
        // `fit`で求めた平均・標準偏差を、`fit`に使っていない別の行へ適用できる。
        let x_fit = Mat::from_fn(4, 3, |i, j| match j {
            0 => 1.0,
            1 => [10.0, 20.0, 30.0, 40.0][i],
            _ => 5.0,
        });
        let x_new = Mat::from_fn(2, 3, |i, j| match j {
            0 => 1.0,
            1 => [25.0, 100.0][i],
            _ => 5.0,
        });

        let scale = ColumnScale::fit(&x_fit, true);
        assert_eq!(scale.means, vec![0.0, 25.0, 5.0]);
        // 切片列と標準偏差0の列（定数列）は`std=1.0`。
        assert_eq!(scale.stds[0], 1.0);
        assert_eq!(scale.stds[2], 1.0);
        let sd1 = 125.0_f64.sqrt();
        assert_eq!(scale.stds[1], sd1);

        let std_new = scale.standardize(&x_new);
        assert_eq!(*std_new.get(0, 0), 1.0);
        assert_eq!(*std_new.get(0, 1), 0.0);
        assert_eq!(*std_new.get(1, 1), 75.0 / sd1);
        assert_eq!(*std_new.get(0, 2), 0.0);

        // 切片なしでは`means`が全0（平均を引かない）。
        let scale_no_intercept = ColumnScale::fit(&x_fit, false);
        assert!(scale_no_intercept.means.iter().all(|m| *m == 0.0));
        assert_eq!(
            *scale_no_intercept.standardize(&x_new).get(1, 1),
            100.0 / sd1
        );
    }

    #[test]
    fn destandardize_cov_params_applies_the_intercept_adjusted_linear_map() {
        let x = Mat::from_fn(4, 2, |i, j| {
            if j == 0 {
                1.0
            } else {
                [10.0, 20.0, 30.0, 40.0][i]
            }
        });
        let (_, scale) = standardize_columns(&x, true);
        assert_eq!(scale.stds[0], 1.0);
        assert!(scale.stds[1] > 1.0);
        assert_eq!(scale.means[1], 25.0);

        let cov_std = Mat::from_fn(2, 2, |i, j| [[4.0, 6.0], [6.0, 9.0]][i][j]);
        let cov_orig = destandardize_cov_params(&cov_std, &scale);

        // `θ₀ = θ̃₀ - (x̄/s)θ̃₁`、`θ₁ = θ̃₁/s`（`s=stds[1]`、`x̄=means[1]`）から、
        // 分散・共分散の展開式（`Var(a-bc)`型）を手で書いた期待値。
        let s = scale.stds[1];
        let r = 25.0 / s;
        let var0 = 4.0 - 2.0 * r * 6.0 + r * r * 9.0;
        let cov01 = (6.0 - r * 9.0) / s;
        let var1 = 9.0 / (s * s);
        assert!((*cov_orig.get(0, 0) - var0).abs() < 1e-12);
        assert!((*cov_orig.get(0, 1) - cov01).abs() < 1e-12);
        assert!((*cov_orig.get(1, 0) - cov01).abs() < 1e-12);
        assert!((*cov_orig.get(1, 1) - var1).abs() < 1e-12);
    }
}
