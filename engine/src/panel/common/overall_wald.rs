//! 直交座標の全体Wald/F検定の失敗を、パネルの系統のエラーに直す部品（FEとFE-IVが共有する）。
//!
//! `fit()`の本体に埋め込まず切り出してあるのは、共分散が特異で検定できないときの契約
//! （`FTestFailed`で`fit()`全体を失敗させる）を、手組みの特異な`M̃`で決定的にテストするため。
//! 実データから厳密に特異な共分散を作ると、特異性の判定が丸め誤差の大きさで決まり不安定になる。

use faer::Mat;

use super::PanelError;
use crate::shared::wald::wald_f_test;

/// 切片のない`k`個の係数が同時にゼロという仮説のWald F検定（直交座標の`b̃`と`M̃`）。`(F統計量, p値)`。
///
/// # Errors
/// `M̃`が数値的に特異などで検定できないとき、`PanelError::FTestFailed`。
pub(crate) fn overall_f_test(
    coordinates: &Mat<f64>,
    meat: &Mat<f64>,
    k: usize,
    df_inference: usize,
) -> Result<(f64, f64), PanelError> {
    wald_f_test(coordinates, meat, 0, k, df_inference).map_err(|source| PanelError::FTestFailed {
        source: source.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_singular_covariance_fails_the_overall_f_test() {
        let coordinates = Mat::from_fn(2, 1, |i, _| 1.0 + i as f64);
        let meat = Mat::<f64>::zeros(2, 2);
        assert!(matches!(
            overall_f_test(&coordinates, &meat, 2, 10),
            Err(PanelError::FTestFailed { .. })
        ));
    }

    #[test]
    fn an_identity_covariance_gives_the_f_statistic() {
        // `b̃ = (1, 2)`、`M̃ = I`なら`W = 1 + 4 = 5`、`F = W / 2`。
        let coordinates = Mat::from_fn(2, 1, |i, _| 1.0 + i as f64);
        let meat = Mat::<f64>::identity(2, 2);
        let (f_statistic, p_value) = overall_f_test(&coordinates, &meat, 2, 10).unwrap();
        assert!((f_statistic - 2.5).abs() < 1e-12);
        assert!(p_value > 0.0 && p_value < 1.0);
    }
}
