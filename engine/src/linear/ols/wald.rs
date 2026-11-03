//! 設計行列の末尾`q`列に対応する係数が全てゼロという帰無仮説のWald検定。
//! 契約（用途・パニック条件）は`OlsEstimator::wald_test_last_columns`のdocコメント参照。

use super::OlsEstimator;
use crate::linear::common::LeastSquaresError;
use crate::shared::wald::wald_f_test;

pub(super) fn wald_test_last_columns(
    estimator: &OlsEstimator,
    q: usize,
) -> Result<(f64, f64), LeastSquaresError> {
    let k = estimator.input.k();
    debug_assert!(
        q > 0 && q <= k,
        "wald_test_last_columns: q must be in (0, k], got q={q}, k={k}"
    );
    Ok(wald_f_test(
        &estimator.params,
        &estimator.cov_params,
        k - q,
        q,
        estimator.df_inference,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::ols::{CovType, OlsInput};

    /// `wald_test_last_columns`が「切片を除く全傾き係数」（`q = df_model`）を対象に呼ばれた
    /// 場合、`fit()`が計算する`f_statistic()`/`f_p_value()`（同じ`wald_f_test`を
    /// `k_constant=1`で呼ぶ）と数値的に一致するはず（対象列の一般化が既存の挙動を
    /// 壊していないことの確認。IVのWu-Hausman検定用に追加）。
    #[test]
    fn wald_test_last_columns_matches_f_statistic_when_q_equals_df_model() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 6.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let x2 = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0];
        let input = OlsInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = OlsEstimator::fit(input, CovType::Hc1, 0.95).unwrap();

        let (stat, p_value) = estimator.wald_test_last_columns(2).unwrap();
        assert!((stat - estimator.f_statistic().unwrap()).abs() < 1e-10);
        assert!((p_value - estimator.f_p_value().unwrap()).abs() < 1e-10);
    }

    /// `q=1`（末尾1列だけを対象）のとき、`F = t²`という標準的な恒等式
    /// （`wald_f_test`のdocコメント参照、1自由度のF検定は両側t検定と代数的に等価）により、
    /// 既に個別に検証済みの`test_stats()`/`p_values()`（`fit()`本体が計算）と一致するはず。
    #[test]
    fn wald_test_last_columns_matches_squared_t_statistic_for_single_column() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 6.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let x2 = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0];
        let input = OlsInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = OlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        let (stat, p_value) = estimator.wald_test_last_columns(1).unwrap();
        let k = estimator.input().k();
        let t_last = *estimator.test_stats().get(k - 1, 0);
        let p_last = *estimator.p_values().get(k - 1, 0);
        assert!((stat - t_last.powi(2)).abs() < 1e-10);
        assert!((p_value - p_last).abs() < 1e-10);
    }
}
