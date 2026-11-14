//! `TobitScaling`（`x`と`y`の標準化と、`(β, logσ)`の逆変換）。

use crate::nonlinear::common::ColumnScale;
use faer::Mat;

/// 列（または`y`）の母集団標準偏差（`ddof=0`）。標準偏差が0（定数列）の場合は`1.0`を
/// 返す（0除算回避）。
fn population_std(values: impl Iterator<Item = f64> + Clone, n: usize) -> f64 {
    let mean: f64 = values.clone().sum::<f64>() / n as f64;
    let var: f64 = values.map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let sd = var.sqrt();
    if sd > 0.0 { sd } else { 1.0 }
}

/// `TobitEstimator::fit`が最適化に使う、設計行列`x`・被説明変数`y`の標準化スケール。
///
/// `x`側（列の平均・標準偏差、切片の逆変換、ヤコビアンの`x`部分）はLogit/Probitと共通の
/// `nonlinear/common/standardize.rs`の`ColumnScale`（切片ありのとき平均も引く）に委ね、この構造体は
/// Tobit固有の`y`のスケーリング（`y_scale`）と`σ`（`logσ`）の扱いだけを持つ。
///
/// **Tobit固有の処理は`y`のスケーリング**（Logit/Probitの`standardize_columns`は`y`を扱わない）:
/// 1. **`y`のスケーリング**: 当時あった分離ヒューリスティック（標準化パラメータ空間のL2ノルムが
///    閾値を超えたら(準)分離と判定する、現在は廃止）はLogit/Probit（`y∈{0,1}`）で較正されていた。
///    Tobitは`y`が連続でスケールが任意のため、`y`のスケールが大きいだけで健全なMLE解でも
///    ノルムが閾値を超え誤発火した（Wooldridge mroz `hours`の生スケールTobitで発覚）。`x`の列標準化と同じ発想
///    （R `survival::survreg`は設計行列を標準化する）を`y`にも適用してスケール不変にする。
///    **`y_scale`は`y`の母集団標準偏差を「2の冪に丸めた」値**（`2^round(log2(sd))`）。
///    2の冪で割る操作は倍精度浮動小数点で指数部のシフトのみ（仮数部は不変）＝厳密なため、
///    `y`が既に程よいスケール（`sd ∈ [1/√2, √2)`）なら`y_scale=1`で完全な恒等変換になり、
///    そうでなくてもNewtonの反復軌道が「スケーリング前と厳密に相似」に保たれる。生の
///    標準偏差（例: `0.949`）で割ると、その丸め誤差（~1e-16）だけでも打ち切り率が高く
///    Hessianが際どい悪条件のデータ（合成`heavy_censoring`、左打ち切り60%）で正則化
///    NewtonがSingularHessianに倒れることが実測で判明したため、丸めて厳密化する。
///
/// **`x`列の平均センタリング（切片ありのときのみ）は`ColumnScale`が行う共通の処理**。Tobitでも
/// 必要な理由: `exper`(0〜45)と`expersq`(0〜2025)のように強く共線な列を含むと、スケーリング
/// だけでは標準化後のHessianが悪条件のままで、正則化Newton（`regularized_newton_step`）が
/// 回復不能な不定符号領域に落ちる（mroz `hours`で実測、`ols_initial_params`の丸め誤差~1e-14で
/// 結果が変わるほど際どい）。切片ありモデルでは列を平均センタリングすると`survreg`同様Newtonが
/// 安定して収束する。切片なしモデルではセンタリングせず（吸収先の切片が無く逆変換が壊れる
/// ため）、スケーリングのみ行う。
///
/// 標準化空間の内部パラメータ`(β̃, s̃=logσ̃)`と元のスケール`(β, s=logσ)`の対応
/// （`c = y_scale`、`x̄ⱼ`・`stdⱼ`は`x`列`j`の平均・母集団標準偏差、切片列は`x̄₀=0`・`std₀=1`）:
///
/// - 切片あり: `βⱼ = c·β̃ⱼ/stdⱼ`（`j≥1`）、`β₀ = c·β̃₀ - Σ_{j≥1} x̄ⱼ·βⱼ`、`s = s̃ + ln c`
/// - 切片なし: `βⱼ = c·β̃ⱼ/stdⱼ`（全`j`）、`s = s̃ + ln c`
///
/// `β`の式は`ColumnScale`の逆変換に`c`を掛けたもの。`c`は2の冪なので、オーバーフロー・
/// サブノーマルを除き、`c·β̃ⱼ/stdⱼ`と`c·(β̃ⱼ/stdⱼ)`、切片の補正項も掛ける順序によらず
/// ビット単位で同じ値になる。この一致は正しさのためではなく、Tobitの出力を丸めの
/// レベルで安定させるために保っている（`y_scale`を2の冪以外にしない理由でもある）。
pub(super) struct TobitScaling {
    /// `x`側の標準化スケール（列ごとの平均・標準偏差、切片の逆変換、ヤコビアンの`x`部分）。
    /// Logit/Probitと共通の`ColumnScale`（切片ありのとき平均も引く）。
    pub(super) x: ColumnScale,
    /// `y`（と打ち切り境界）のスケール = `y`の母集団標準偏差を2の冪に丸めた値
    /// （`2^round(log2(sd))`、必ず`> 0`。構造体docコメント参照）
    pub(super) y_scale: f64,
}

impl TobitScaling {
    /// `x`・`y`から標準化スケールを求める。
    pub(super) fn fit(x: &Mat<f64>, y: &Mat<f64>, has_intercept: bool) -> Self {
        let n = x.nrows();
        // `y`の母集団標準偏差を2の冪に丸める（構造体docコメント「`y`のスケーリング」参照。
        // 2の冪での除算は仮数部を変えないため、`y`が程よいスケールなら恒等変換になる）。
        // 指数は倍精度の正規化数の範囲に収める（極端な`y`はengine_pybind側の非有限値
        // チェックで既に弾かれているため、この`clamp`は防御的）。
        let y_exp = (population_std((0..n).map(|i| *y.get(i, 0)), n)
            .log2()
            .round() as i32)
            .clamp(-1000, 1000);
        let y_scale = 2.0_f64.powi(y_exp);
        Self {
            x: ColumnScale::fit(x, has_intercept),
            y_scale,
        }
    }

    /// 設計行列を標準化する: 切片ありは`x̃ⱼ = (xⱼ - x̄ⱼ)/stdⱼ`、切片なしは`x̃ⱼ = xⱼ/stdⱼ`
    /// （切片列は無変換）。`ColumnScale::standardize`に委譲する。
    pub(super) fn standardize_x(&self, x: &Mat<f64>) -> Mat<f64> {
        self.x.standardize(x)
    }

    /// `y`を`y_scale`でスケーリングする（平均は引かない）。
    pub(super) fn standardize_y(&self, y: &Mat<f64>) -> Mat<f64> {
        Mat::from_fn(y.nrows(), 1, |i, _| *y.get(i, 0) / self.y_scale)
    }

    /// 打ち切り境界を`y_scale`でスケーリングする（`y`のスケーリングと整合）。
    pub(super) fn standardize_bound(&self, bound: Option<f64>) -> Option<f64> {
        bound.map(|v| v / self.y_scale)
    }

    /// 標準化空間の内部パラメータ`(β̃, s̃)`（`k+1`次元）を元のスケールの`(β, logσ)`へ
    /// 逆変換する（構造体docコメントの式）。`β`は`x`側の逆変換
    /// （`ColumnScale`の式）に`y_scale`を掛けたもの。`y_scale`は2の冪なので、この掛け算は
    /// 掛ける順序によらずビット単位で同じ値になる（構造体docコメント参照）。
    pub(super) fn destandardize_params(&self, params_std: &[f64]) -> Vec<f64> {
        let k = self.x.stds().len();
        debug_assert_eq!(
            params_std.len(),
            k + 1,
            "params_std holds the k coefficients and log sigma"
        );
        let c = self.y_scale;
        let mut out = Vec::with_capacity(k + 1);
        out.extend(
            crate::nonlinear::common::destandardize_params(&params_std[..k], &self.x)
                .into_iter()
                .map(|b| c * b),
        );
        out.push(params_std[k] + c.ln());
        out
    }

    /// `(β̃, s̃)`空間から`(β, σ)`空間（元のスケール）への変換のヤコビアン`(k+1)×(k+1)`。
    /// `Cov(β, σ) = J · Cov(β̃, s̃) · Jᵀ`で使う（`destandardize_params`の線形部分の
    /// 微分に、`s̃→σ=c·exp(s̃)`のデルタ法`∂σ/∂s̃=σ`を末尾行に折り込んだもの）。`sigma`は
    /// 逆変換済みの`σ`。`β`の部分は`x`側のヤコビアン（`ColumnScale::param_jacobian`）の`c`倍。
    fn param_jacobian(&self, sigma: f64) -> Mat<f64> {
        let k = self.x.stds().len();
        let c = self.y_scale;
        let jx = self.x.param_jacobian();
        Mat::from_fn(k + 1, k + 1, |i, m| {
            if i == k {
                if m == k { sigma } else { 0.0 }
            } else if m < k {
                c * *jx.get(i, m)
            } else {
                0.0
            }
        })
    }

    /// `Cov(β, σ) = J · cov_std · Jᵀ`（`param_jacobian`のJを両側から適用）。
    pub(super) fn destandardize_cov_params(&self, cov_std: &Mat<f64>, sigma: f64) -> Mat<f64> {
        let j = self.param_jacobian(sigma);
        &j * cov_std * j.transpose()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn population_std_returns_population_std_and_falls_back_to_one_for_constant() {
        // 母集団分散 = ((1.5)² + (0.5)² + (0.5)² + (1.5)²)/4 = 1.25 → std = √1.25
        let v = [1.0, 2.0, 3.0, 4.0];
        assert!((population_std(v.iter().copied(), 4) - 1.25_f64.sqrt()).abs() < 1e-12);

        // 定数（std=0）のときは0除算回避のため`1.0`にフォールバックする。
        assert_eq!(population_std([7.0, 7.0, 7.0].iter().copied(), 3), 1.0);
    }

    /// `TobitScaling`の逆変換（`destandardize_params`）が、標準化した設計行列・`y`で
    /// 評価した線形予測子`x̃'β̃`を元のスケールの`x'β/c`に戻すことを、切片あり・切片なし
    /// 両方で確認する（往復変換の一貫性、`ColumnScale`の往復テストのTobit版）。
    #[test]
    fn tobit_scaling_destandardize_params_inverts_the_linear_predictor() {
        for has_intercept in [true, false] {
            // 切片あり: 列0が定数1.0。切片なし: 定数列なし。
            let (cols, k): (Vec<Vec<f64>>, usize) = if has_intercept {
                (
                    vec![
                        vec![1.0; 5],
                        vec![10.0, 12.0, 9.0, 15.0, 11.0],
                        vec![100.0, 400.0, 900.0, 1600.0, 2500.0],
                    ],
                    3,
                )
            } else {
                (
                    vec![
                        vec![2.0, 4.0, 6.0, 8.0, 10.0],
                        vec![1.0, 3.0, 2.0, 5.0, 4.0],
                    ],
                    2,
                )
            };
            let x = Mat::from_fn(5, k, |i, j| cols[j][i]);
            let y = Mat::from_fn(5, 1, |i, _| [3.0, 7.0, 5.0, 11.0, 9.0][i] * 50.0);
            let scaling = TobitScaling::fit(&x, &y, has_intercept);
            let x_std = scaling.standardize_x(&x);

            // 標準化空間の任意のパラメータ`(β̃, s̃)`。
            let params_std: Vec<f64> = (0..=k).map(|j| 0.3 + 0.2 * j as f64).collect();
            let params_orig = scaling.destandardize_params(&params_std);

            // 各観測で `x̃ᵢ'β̃ == xᵢ'β / c` が成り立つはず。
            for i in 0..5 {
                let lhs: f64 = (0..k).map(|j| *x_std.get(i, j) * params_std[j]).sum();
                let rhs: f64 =
                    (0..k).map(|j| *x.get(i, j) * params_orig[j]).sum::<f64>() / scaling.y_scale;
                assert!(
                    (lhs - rhs).abs() < 1e-9,
                    "has_intercept={has_intercept}, i={i}"
                );
            }
            // `s̃ = logσ - ln c`
            assert!((params_orig[k] - (params_std[k] + scaling.y_scale.ln())).abs() < 1e-12);
        }
    }

    /// 標準化に使う`y`のスケール`y_scale`は、`y`のスケールによらず厳密に2の冪になる。
    /// 掛け算の順序で値が変わらない性質（`tobit_scaling_destandardize_matches_the_closed_form_bit_for_bit`）
    /// と、反復軌道が標準化前と厳密に相似に保たれる性質の前提。
    #[test]
    fn tobit_scaling_y_scale_is_an_exact_power_of_two() {
        let x = Mat::from_fn(5, 2, |i, j| {
            if j == 0 {
                1.0
            } else {
                [1.0, 3.0, 2.0, 5.0, 4.0][i]
            }
        });
        for factor in [1e-6, 0.01, 0.3, 1.0, 7.0, 50.0, 1e4, 1e9] {
            let y = Mat::from_fn(5, 1, |i, _| [3.0, 7.0, 5.0, 11.0, 9.0][i] * factor);
            let scale = TobitScaling::fit(&x, &y, true).y_scale;
            assert!(scale > 0.0 && scale.is_finite(), "factor={factor}");
            // 2の冪は仮数部が0（`log2`が整数）で、`powi`で戻すと元の値に厳密に一致する。
            let exp = scale.log2();
            assert_eq!(exp.fract(), 0.0, "factor={factor}, y_scale={scale}");
            assert_eq!(2.0_f64.powi(exp as i32), scale, "factor={factor}");
        }
    }

    /// `TobitScaling`の逆変換・ヤコビアンが、`ColumnScale`に委ねる前の式
    /// （`βⱼ = c·β̃ⱼ/stdⱼ`、`β₀ = c·β̃₀ - Σ x̄ⱼβⱼ`、`J`の対角`c/stdⱼ`・第0行`-c·x̄ⱼ/stdⱼ`）と
    /// ビット単位で一致する。`y_scale`が2の冪なので、掛ける順序が違っても値が変わらないことを
    /// 固定する。許容誤差を使わないのは、丸め誤差が入ってもテストが気づけなくなるため。
    /// `y`のスケールを変えて`c≠1`（大小両方）の場合も確かめる。
    #[test]
    fn tobit_scaling_destandardize_matches_the_closed_form_bit_for_bit() {
        let n = 6;
        for has_intercept in [true, false] {
            let cols: Vec<Vec<f64>> = if has_intercept {
                vec![
                    vec![1.0; n],
                    vec![10.0, 12.0, 9.0, 15.0, 11.0, 13.0],
                    vec![100.0, 400.0, 900.0, 1600.0, 2500.0, 3600.0],
                ]
            } else {
                vec![
                    vec![2.0, 4.0, 6.0, 8.0, 10.0, 7.0],
                    vec![1.0, 3.0, 2.0, 5.0, 4.0, 9.0],
                    vec![0.5, 0.1, 0.9, 0.3, 0.7, 0.2],
                ]
            };
            let k = cols.len();
            let x = Mat::from_fn(n, k, |i, j| cols[j][i]);
            let start = usize::from(has_intercept);

            for y_factor in [0.003, 1.0, 40.0, 2500.0] {
                let y = Mat::from_fn(n, 1, |i, _| [3.0, 7.0, 5.0, 11.0, 9.0, 4.0][i] * y_factor);
                let scaling = TobitScaling::fit(&x, &y, has_intercept);
                let c = scaling.y_scale;

                // `x`の平均・母集団標準偏差を、実装とは独立に（切片列・切片なしでは平均0）求める。
                let means: Vec<f64> = (0..k)
                    .map(|j| {
                        if has_intercept && j >= start {
                            cols[j].iter().sum::<f64>() / n as f64
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let stds: Vec<f64> = (0..k)
                    .map(|j| {
                        if j < start {
                            return 1.0;
                        }
                        let m = cols[j].iter().sum::<f64>() / n as f64;
                        let var = cols[j].iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64;
                        if var.sqrt() > 0.0 { var.sqrt() } else { 1.0 }
                    })
                    .collect();

                let params_std: Vec<f64> = (0..=k).map(|j| 0.3 + 0.2 * j as f64).collect();
                let mut expected = vec![0.0; k + 1];
                for j in start..k {
                    expected[j] = c * params_std[j] / stds[j];
                }
                if has_intercept {
                    expected[0] =
                        c * params_std[0] - (1..k).map(|j| means[j] * expected[j]).sum::<f64>();
                }
                expected[k] = params_std[k] + c.ln();

                let actual = scaling.destandardize_params(&params_std);
                for j in 0..=k {
                    assert_eq!(
                        actual[j].to_bits(),
                        expected[j].to_bits(),
                        "has_intercept={has_intercept}, y_factor={y_factor}, j={j}: \
                         {} vs {}",
                        actual[j],
                        expected[j]
                    );
                }

                let sigma = 1.7;
                let jac = scaling.param_jacobian(sigma);
                for i in 0..=k {
                    for m in 0..=k {
                        let expected_ij = if i == k {
                            if m == k { sigma } else { 0.0 }
                        } else if m == k {
                            0.0
                        } else if has_intercept && i == 0 {
                            if m == 0 { c } else { -c * means[m] / stds[m] }
                        } else if m == i {
                            c / stds[i]
                        } else {
                            0.0
                        };
                        assert_eq!(
                            jac.get(i, m).to_bits(),
                            expected_ij.to_bits(),
                            "has_intercept={has_intercept}, y_factor={y_factor}, J[{i}][{m}]"
                        );
                    }
                }
            }
        }
    }

    /// `TobitScaling::param_jacobian`（`destandardize_cov_params`が両側から掛ける`J`）が、
    /// `(β̃, s̃) → (β, σ)`の写像の解析ヤコビアンと一致することを、中心差分で独立に検証する
    /// （`expected_cov_params`ヘルパーは最終段で`destandardize_cov_params`を本体と共用して
    /// おり`param_jacobian`のバグを検出できないため）。特に
    /// 切片行の非対角項`∂β₀/∂β̃ⱼ = -c·x̄ⱼ/stdⱼ`と末尾行`∂σ/∂s̃ = σ`（`σ = c·exp(s̃)`）を
    /// カバーする。写像は`g(p)ⱼ = destandardize_params(p)ⱼ`（`j<k`）・
    /// `g(p)_k = exp(destandardize_params(p)_k)`。
    #[test]
    fn param_jacobian_matches_finite_difference_of_the_destandardization_map() {
        for has_intercept in [true, false] {
            let (cols, k): (Vec<Vec<f64>>, usize) = if has_intercept {
                (
                    vec![
                        vec![1.0; 6],
                        vec![10.0, 12.0, 9.0, 15.0, 11.0, 13.0],
                        // 切片と強く相関する（平均が大きい）列。センタリング項を効かせる。
                        vec![100.0, 121.0, 81.0, 225.0, 121.0, 169.0],
                    ],
                    3,
                )
            } else {
                (
                    vec![
                        vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                        vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
                    ],
                    2,
                )
            };
            let x = Mat::from_fn(6, k, |i, j| cols[j][i]);
            let y = Mat::from_fn(6, 1, |i, _| [3.0, 7.0, 5.0, 11.0, 9.0, 8.0][i] * 30.0);
            let scaling = TobitScaling::fit(&x, &y, has_intercept);

            let params_std: Vec<f64> = (0..=k).map(|j| 0.2 + 0.15 * j as f64).collect();
            let g = |p: &[f64]| -> Vec<f64> {
                let orig = scaling.destandardize_params(p);
                (0..=k)
                    .map(|i| if i < k { orig[i] } else { orig[k].exp() })
                    .collect()
            };
            let sigma = g(&params_std)[k];
            let analytic = scaling.param_jacobian(sigma);

            let h = 1e-6;
            for m in 0..=k {
                let mut plus = params_std.clone();
                let mut minus = params_std.clone();
                plus[m] += h;
                minus[m] -= h;
                let gp = g(&plus);
                let gm = g(&minus);
                for i in 0..=k {
                    let numeric = (gp[i] - gm[i]) / (2.0 * h);
                    let a = *analytic.get(i, m);
                    assert!(
                        (a - numeric).abs() <= 1e-5 * a.abs().max(1.0),
                        "has_intercept={has_intercept} ({i},{m}): analytic={a}, numeric={numeric}"
                    );
                }
            }
        }
    }
}
