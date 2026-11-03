//! 設計行列のランクチェック（列ピボットQR）と、OLSによる初期値。

use faer::Mat;
use faer::prelude::SolveLstsq;

use super::error::MleError;
use crate::shared::linear_algebra::{
    RankDeficient, checked_col_piv_qr, collinear_columns, column_names,
};

/// 標準化済みの設計行列`x_std`を列ピボットQR分解し、ランク落ちが無ければQR分解をそのまま
/// 返す（呼び出し側が`solve_lstsq`で最小二乗解＝反復最適化の初期値を、再分解せずに
/// 取り出せるようにするため）。
///
/// 特異性判定は`linear_algebra::checked_col_piv_qr`
/// （OLS・`newton_step`と共通。`R`の対角成分に対する相対閾値`k·ε·max|R_ii|`とNaNの明示チェック、
/// `.claude/rules/rust-style.md`「線形代数」）で、列ごとの一様スケーリングに対して
/// 不変なため、標準化の有無で判定結果は変わらない。
///
/// **共線な列は標準化前の`x`で特定する**: 切片ありの標準化は非切片列の平均を引くので、
/// 標準化後の行列では切片との共線性（定数列・ダミー変数の罠）が切片を含まない形に見えてしまう。
/// 利用者に報告するのは元の設計の列なので、判定は`x_std`、列の特定は`x`で行う（`x`と`x_std`は
/// 同じ列名`names`を共有する）。
///
/// Logit/Probit（`ols_based_initial_params`）とTobit（`tobit::ols_initial_params`）が
/// `fit()`冒頭で共有する。Newton法が一度も反復していない段階での検出のため、エラーは
/// `SingularHessian`（最適化中・収束後のHessian逆行列計算）ではなく`SingularDesignMatrix`
/// （後者のdocコメント参照。`solver`（newton/bfgs/lbfgs）に関わらず同じこの単一経路で
/// 多重共線性を検出するのが本関数を共有する目的）。
///
/// # Errors
/// `x_std`がランク落ち（完全な多重共線性等）: `MleError::SingularDesignMatrix`
/// （`names`は設計の列名で、共線な列を`x`から求めて名前に直す）。
pub fn checked_design_matrix_qr(
    x_std: &Mat<f64>,
    x: &Mat<f64>,
    names: &[String],
) -> Result<faer::linalg::solvers::ColPivQr<f64>, MleError> {
    checked_col_piv_qr(x_std).map_err(|err| {
        debug_assert_eq!(
            err,
            RankDeficient::Columns,
            "checked_col_piv_qr only reports rank deficiency of the columns"
        );
        MleError::SingularDesignMatrix {
            columns: column_names(&collinear_columns(x), names),
        }
    })
}

/// 標準化空間での線形確率モデル（LPM）最小二乗解を、リンク関数のIRLS 1ステップ相当の
/// スケール補正を施してNewton/準Newton法の初期値（warm start）に変換する。Logit/Probitが
/// `fit()`冒頭で共有する（Tobitは`β`がOLSと同一スケールでスケール補正が不要なため、
/// `tobit::ols_initial_params`が`checked_design_matrix_qr`だけを共有し補正は行わない）。
///
/// ## 数式（nullモデル `p≡p̄` を起点にしたIRLSの1反復目）
///
/// `p̄ = ȳ`（標本平均）、`w`＝リンクのIRLS重み（logit: `p̄(1-p̄)`、probit: `φ(Φ⁻¹(p̄))`）、
/// `η₀`＝`p̄`でのリンク値（logit: `ln(p̄/(1-p̄))`、probit: `Φ⁻¹(p̄)`）とすると、nullモデルを
/// 起点にしたIRLSの1反復目の解は
/// `β⁽¹⁾ = b_lpm / w + (η₀ - p̄/w)·e₀`（`b_lpm`はLPM最小二乗解、`e₀`は切片成分の
/// 単位ベクトル）に整理できる（設計行列`X`が切片列を含むとき`(X'X)⁻¹X'𝟙 = e₀`・
/// `(X'X)⁻¹X'(y - p̄𝟙) = b_lpm - p̄e₀`を使う）。実装は全成分を`1/w`倍し、切片成分にのみ
/// `η₀ - p̄/w`を加える。切片なしモデルでは切片補正項を落とし`b_lpm/w`のみとする
/// （吸収先の切片列が無く上の整理が成り立たないため、素の`1/w`スケーリングに留める）。
///
/// `p̄`が0または1の近傍（全観測で`y`が同一＝完全分離）では`w→0`で補正が発散するため、
/// 補正を行わず素のLPM解を返す（分離の検出は`run_solver`の事後チェック・`NonConvergence`に
/// 委ねる。`tobit::ols_initial_params`が完全当てはまりで`σ=1`にフォールバックするのと
/// 同じ発想）。
///
/// `link_terms`は`p̄`を受け取り`(w, η₀)`を返すクロージャ（Logit/Probitでリンクが
/// 異なるためクロージャで受ける、`predict_from_link`と同じ方針）。
///
/// # Errors
/// `x_std`がランク落ち: `MleError::SingularDesignMatrix`（`checked_design_matrix_qr`。
/// `x`は標準化前の設計行列で、共線な列の特定に使う）
pub fn ols_based_initial_params(
    x_std: &Mat<f64>,
    x: &Mat<f64>,
    y: &Mat<f64>,
    has_intercept: bool,
    names: &[String],
    link_terms: impl Fn(f64) -> (f64, f64),
) -> Result<Vec<f64>, MleError> {
    let qr = checked_design_matrix_qr(x_std, x, names)?;
    let k = x_std.ncols();
    let n = y.nrows();
    let beta_lpm = qr.solve_lstsq(y);
    let mut params: Vec<f64> = (0..k).map(|i| *beta_lpm.get(i, 0)).collect();

    let p_bar = (0..n).map(|i| *y.get(i, 0)).sum::<f64>() / n as f64;
    // 全観測でyが同一（p̄が0/1近傍）だとIRLS重みwが0へ潰れて補正が発散する。その場合は
    // 素のLPM解を初期値にし、分離の検出はrun_solverに委ねる（docコメント参照）。
    const P_BAR_DEGENERACY_FLOOR: f64 = 1e-6;
    if p_bar > P_BAR_DEGENERACY_FLOOR && p_bar < 1.0 - P_BAR_DEGENERACY_FLOOR {
        let (w, eta0) = link_terms(p_bar);
        for param in params.iter_mut() {
            *param /= w;
        }
        if has_intercept {
            params[0] += eta0 - p_bar / w;
        }
    }
    Ok(params)
}

#[cfg(test)]
mod tests {

    use super::super::standardize::standardize_columns;

    use super::*;

    #[test]
    fn checked_design_matrix_qr_accepts_full_rank_and_returns_usable_decomposition() {
        // フルランクな設計行列（切片 + 独立な2列）。QR分解が返り、その`solve_lstsq`が
        // 最小二乗解を与えることを確認する（呼び出し側が再分解せず初期値を取り出せる）。
        let x = Mat::from_fn(5, 3, |i, j| match j {
            0 => 1.0,
            1 => [1.0, 2.0, 3.0, 4.0, 5.0][i],
            _ => [2.0, 1.0, 4.0, 3.0, 6.0][i],
        });
        let y = Mat::from_fn(5, 1, |i, _| [1.0, 2.0, 2.0, 4.0, 5.0][i]);

        let qr = checked_design_matrix_qr(&x, &x, &[])
            .expect("full-rank design matrix must be accepted");
        let beta = qr.solve_lstsq(&y);
        // 正規方程式 X'Xβ = X'y を満たすはず（残差が設計行列と直交）。
        let resid = &y - &x * &beta;
        for j in 0..3 {
            let dot: f64 = (0..5).map(|i| *x.get(i, j) * *resid.get(i, 0)).sum();
            assert!(
                dot.abs() < 1e-9,
                "column {j} not orthogonal to residual: {dot}"
            );
        }
    }

    #[test]
    fn checked_design_matrix_qr_rejects_rank_deficient_design_matrix() {
        // 3列目 = 2 × 2列目（完全な多重共線性）。
        let x = Mat::from_fn(5, 3, |i, j| match j {
            0 => 1.0,
            1 => [1.0, 2.0, 3.0, 4.0, 5.0][i],
            _ => 2.0 * [1.0, 2.0, 3.0, 4.0, 5.0][i],
        });
        let names = ["const".to_string(), "t".to_string(), "t2".to_string()];
        // 従属になるのは3列目と、それを表す2列目（切片は関わらない）。
        assert_eq!(
            checked_design_matrix_qr(&x, &x, &names).unwrap_err(),
            MleError::SingularDesignMatrix {
                columns: vec!["t".to_string(), "t2".to_string()]
            }
        );
    }

    #[test]
    fn checked_design_matrix_qr_rejects_all_zero_design_matrix_via_nan_guard() {
        // 全ゼロ列の`col_piv_qr`は`R`の対角にNaNを生じうる（`newton_step`と同じ罠）。
        let x = Mat::from_fn(4, 2, |_, _| 0.0);
        // 全ゼロの先頭列はそれ単独で従属。
        assert_eq!(
            checked_design_matrix_qr(&x, &x, &["a".to_string(), "b".to_string()]).unwrap_err(),
            MleError::SingularDesignMatrix {
                columns: vec!["a".to_string()]
            }
        );
    }

    #[test]
    fn ols_based_initial_params_recovers_exact_logit_mle_for_intercept_only_model() {
        // 切片のみ（`x`が定数列だけ）: LPMの最小二乗解は`b_lpm = [p̄]`、logitの補正で
        // `β⁽¹⁾[0] = p̄/w + (η₀ - p̄/w) = η₀ = ln(p̄/(1-p̄))`（切片のみモデルの厳密なMLE）。
        let y = Mat::from_fn(7, 1, |i, _| if i < 4 { 1.0 } else { 0.0 });
        let x = Mat::from_fn(7, 1, |_, _| 1.0);
        let params = ols_based_initial_params(&x, &x, &y, true, &[], |p_bar| {
            (p_bar * (1.0 - p_bar), (p_bar / (1.0 - p_bar)).ln())
        })
        .unwrap();

        let p_bar: f64 = 4.0 / 7.0;
        let expected = (p_bar / (1.0 - p_bar)).ln();
        assert!((params[0] - expected).abs() < 1e-12, "{params:?}");
    }

    #[test]
    fn ols_based_initial_params_scales_slopes_by_link_derivative_reciprocal() {
        // 切片なしモデル: 補正は全成分を`1/w`倍するだけ（切片補正項なし）。`w`を既知の
        // 定数に固定して、返り値が`b_lpm / w`（LPM解のスケール変換）であることを確認する。
        let x = Mat::from_fn(4, 1, |i, _| [1.0, 2.0, 3.0, 4.0][i]);
        let y = Mat::from_fn(4, 1, |i, _| [0.0, 0.0, 1.0, 1.0][i]);

        let raw = ols_based_initial_params(&x, &x, &y, false, &[], |_| (1.0, 0.0)).unwrap();
        let scaled = ols_based_initial_params(&x, &x, &y, false, &[], |_| (0.25, 0.0)).unwrap();
        assert!(
            (scaled[0] - raw[0] / 0.25).abs() < 1e-12,
            "{scaled:?} {raw:?}"
        );
    }

    #[test]
    fn ols_based_initial_params_falls_back_to_raw_lpm_when_all_y_are_identical() {
        // 全観測でy=1（p̄=1、完全分離）: IRLS重み`w`が0へ潰れるため補正を行わず、
        // 素のLPM解（切片=1.0、傾き=0.0）を返す（`link_terms`は呼ばれないので発散しない）。
        let x = Mat::from_fn(
            4,
            2,
            |i, j| if j == 0 { 1.0 } else { [1.0, 2.0, 3.0, 4.0][i] },
        );
        let y = Mat::from_fn(4, 1, |_, _| 1.0);
        let params = ols_based_initial_params(&x, &x, &y, true, &[], |_| {
            panic!("link_terms must not be called for a degenerate p_bar")
        })
        .unwrap();
        assert!(
            (params[0] - 1.0).abs() < 1e-9 && params[1].abs() < 1e-9,
            "{params:?}"
        );
    }

    #[test]
    fn ols_based_initial_params_propagates_singular_design_matrix_error() {
        let x = Mat::from_fn(5, 3, |i, j| match j {
            0 => 1.0,
            1 => [1.0, 2.0, 3.0, 4.0, 5.0][i],
            _ => 2.0 * [1.0, 2.0, 3.0, 4.0, 5.0][i],
        });
        let y = Mat::from_fn(5, 1, |i, _| [0.0, 1.0, 0.0, 1.0, 1.0][i]);
        assert_eq!(
            ols_based_initial_params(&x, &x, &y, true, &[], |p| (p * (1.0 - p), 0.0)).unwrap_err(),
            MleError::SingularDesignMatrix {
                columns: vec!["column 1".to_string(), "column 2".to_string()]
            }
        );
    }

    #[test]
    fn checked_design_matrix_qr_reports_the_intercept_for_collinearity_hidden_by_centering() {
        // 標準化（切片ありでは非切片列の平均を引く）後の行列では、切片との共線性が切片を含まない形に
        // 見える。共線な列は標準化前の設計行列から特定するので、切片も報告に含まれる。
        let names = ["const".to_string(), "d1".to_string(), "d2".to_string()];
        // ダミー変数の罠: `d1 + d2 = const`。
        let trap = Mat::from_fn(6, 3, |i, j| match j {
            0 => 1.0,
            1 => [1.0, 0.0, 1.0, 0.0, 1.0, 0.0][i],
            _ => [0.0, 1.0, 0.0, 1.0, 0.0, 1.0][i],
        });
        let (trap_std, _) = standardize_columns(&trap, true);
        assert_eq!(
            checked_design_matrix_qr(&trap_std, &trap, &names).unwrap_err(),
            MleError::SingularDesignMatrix {
                columns: names.to_vec()
            }
        );

        // 定数列は切片と共線（標準化後は全ゼロ列になる）。
        let constant = Mat::from_fn(6, 3, |i, j| match j {
            0 => 1.0,
            1 => [1.0, 2.0, 4.0, 3.0, 6.0, 5.0][i],
            _ => 3.0,
        });
        let (constant_std, _) = standardize_columns(&constant, true);
        let names = ["const".to_string(), "x".to_string(), "c".to_string()];
        assert_eq!(
            checked_design_matrix_qr(&constant_std, &constant, &names).unwrap_err(),
            MleError::SingularDesignMatrix {
                columns: vec!["const".to_string(), "c".to_string()]
            }
        );
    }
}
