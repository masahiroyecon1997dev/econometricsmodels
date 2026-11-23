//! Breusch-Pagan検定（不均一分散の検定、Koenkerの標準化版）。利用者が選んだ変数をそのまま補助回帰の
//! 項にして共通部分に渡す。

use super::aux_regression::{
    AuxRegressionTest, check_column_lengths, select_aux_terms, squared_residual_aux_test,
};
use crate::linear::common::{DiagnosticTest, LeastSquaresError};

/// Breusch-Pagan検定の結果（[`AuxRegressionTest`]）。
pub type BreuschPaganTest = AuxRegressionTest;

/// Breusch-Pagan検定（不均一分散の検定、Koenkerの標準化版）。残差の二乗を、利用者が選んだ
/// 変数`z`（と定数）に回帰する補助回帰を行い、`LM = n·R²`（`χ²(q)`）とF版を計算する。
///
/// 元のBreusch-Pagan（1979）の`ESS/2`版は誤差の正規性を仮定するため扱わない（`n·R²`版は
/// 正規性を仮定せず、R `bptest(studentize=TRUE)`・statsmodels `het_breuschpagan(robust=True)`
/// と同じ）。`z_columns`は定数列を含まない必要はない（定数列・数値的に同一の列は
/// [`white_test`](super::white::white_test)と同じ規則で除き、残った列数が自由度`q`になる）。補助回帰には常に定数を含める
/// （元のモデルが`include_intercept=false`でも同じ）。`z`はモデルの説明変数に限らず、任意の
/// 列でよい。
///
/// # Errors
/// [`white_test`](super::white::white_test)と同じ（「説明変数」を`z`と読み替える。メッセージの接頭辞は
/// `Breusch-Pagan test`）。
///
/// # パニックについて
/// `z_names.len() != z_columns.len()`は`engine_pybind`の実装バグでしか起こらない内部契約
/// なので`debug_assert!`でパニックさせる。
pub fn breusch_pagan_test(
    z_columns: &[Vec<f64>],
    z_names: &[String],
    residuals: &[f64],
) -> Result<BreuschPaganTest, LeastSquaresError> {
    debug_assert_eq!(z_columns.len(), z_names.len());
    check_column_lengths(z_columns, residuals.len())?;

    let candidates = z_names
        .iter()
        .cloned()
        .zip(z_columns.iter().cloned())
        .collect();
    let (kept, dropped_terms) = select_aux_terms(candidates, |_| false);
    squared_residual_aux_test(DiagnosticTest::BreuschPagan, kept, dropped_terms, residuals)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;
    use crate::shared::error::CommonError;

    #[test]
    fn breusch_pagan_matches_statsmodels_for_the_model_regressors() {
        // statsmodelsの`het_breuschpagan(robust=True)`（`exog_het`は定数＋`z`）。
        let result =
            breusch_pagan_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_A).unwrap();

        assert_close(result.lm_statistic, 9.791524511701063, "lm");
        assert_close(result.lm_p_value, 0.007478206743958307, "lm_p");
        assert_close(result.f_statistic, 11.279528376813682, "f");
        assert_close(result.f_p_value, 0.0017527347358498452, "f_p");
        assert_eq!(result.df, 2);
        assert_eq!(result.f_df_denom, 12);
        assert_eq!(result.aux_terms, names(&["const", "x1", "x2"]));
        assert!(result.dropped_terms.is_empty());

        let with_dummy =
            breusch_pagan_test(&cols(&[&X1, &D]), &names(&["x1", "d"]), &RESID_B).unwrap();
        assert_close(with_dummy.lm_statistic, 9.394149048449083, "lm_d");
        assert_close(with_dummy.lm_p_value, 0.009121924073057876, "lm_p_d");
        assert_close(with_dummy.f_statistic, 10.054654463315794, "f_d");
        assert_close(with_dummy.f_p_value, 0.0027245935549812376, "f_p_d");
    }

    #[test]
    fn breusch_pagan_accepts_variables_that_are_not_model_regressors() {
        // `z`はモデルの説明変数に限らない。1変数のときは`df = 1`。
        let result = breusch_pagan_test(&cols(&[&X2]), &names(&["x2"]), &RESID_A).unwrap();

        assert_close(result.lm_statistic, 0.05617588636961934, "lm");
        assert_close(result.lm_p_value, 0.8126455198130225, "lm_p");
        assert_close(result.f_statistic, 0.04886878467332563, "f");
        assert_close(result.f_p_value, 0.8284777578739884, "f_p");
        assert_eq!(result.df, 1);
        assert_eq!(result.f_df_denom, 13);
    }

    #[test]
    fn breusch_pagan_uses_a_constant_for_no_intercept_residuals() {
        // 元のモデルが定数なし（残差の和が0とは限らない）でも補助回帰は定数を含む。
        let result =
            breusch_pagan_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_C).unwrap();

        assert_close(result.lm_statistic, 6.806345878682998, "lm");
        assert_close(result.lm_p_value, 0.033267546415010646, "lm_p");
        assert_close(result.f_statistic, 4.984110223282637, "f");
        assert_close(result.f_p_value, 0.026565513096981214, "f_p");
        assert_eq!(result.aux_terms[0], "const");
    }

    #[test]
    fn breusch_pagan_df_equals_aux_terms_minus_constant() {
        let result =
            breusch_pagan_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_A).unwrap();
        assert_eq!(result.df, result.aux_terms.len() - 1);
    }

    #[test]
    fn breusch_pagan_drops_constant_and_duplicate_variables_and_uses_rank_based_df() {
        // `one`は定数、`d_copy`は`d`と同一の列。どちらも除き、自由度は残った列数の2。
        let ones = [1.0_f64; 15];
        let result = breusch_pagan_test(
            &cols(&[&ones, &X1, &D, &D]),
            &names(&["one", "x1", "d", "d_copy"]),
            &RESID_B,
        )
        .unwrap();

        assert_eq!(result.dropped_terms, names(&["one", "d_copy"]));
        assert_eq!(result.aux_terms, names(&["const", "x1", "d"]));
        assert_eq!(result.df, 2);
        assert_close(result.lm_statistic, 9.394149048449083, "lm");
        assert_close(result.f_statistic, 10.054654463315794, "f");
    }

    #[test]
    fn breusch_pagan_fails_when_every_variable_is_constant_or_none_is_given() {
        let ones = [1.0_f64; 15];
        for (z, z_names) in [(cols(&[&ones]), names(&["one"])), (Vec::new(), Vec::new())] {
            assert_eq!(
                breusch_pagan_test(&z, &z_names, &RESID_A).unwrap_err(),
                LeastSquaresError::DiagnosticVariablesConstant {
                    test: DiagnosticTest::BreuschPagan
                }
            );
        }
    }

    #[test]
    fn breusch_pagan_rejects_mismatched_column_length() {
        let short = [0.1_f64, 0.2, 0.3];
        let err = breusch_pagan_test(&cols(&[&short]), &names(&["z"]), &RESID_A).unwrap_err();
        assert_eq!(
            err,
            LeastSquaresError::Common(CommonError::DimensionMismatch {
                y_rows: 15,
                x_rows: 3
            })
        );
    }

    #[test]
    fn breusch_pagan_checks_the_sample_size_against_the_variables_that_remain() {
        // 補助回帰は定数込み`q + 1`列。`n <= q + 1`は弾き、`n = q + 2`なら通る。
        let z = cols(&[&X1[..3], &X2[..3]]);
        let z_names = names(&["x1", "x2"]);
        assert_eq!(
            breusch_pagan_test(&z, &z_names, &RESID_A[..3]).unwrap_err(),
            LeastSquaresError::InsufficientObservationsForAuxRegression {
                test: DiagnosticTest::BreuschPagan,
                n: 3,
                k: 3
            }
        );

        let z = cols(&[&X1[..4], &X2[..4]]);
        let result = breusch_pagan_test(&z, &z_names, &RESID_A[..4]).unwrap();
        assert_eq!(result.f_df_denom, 1);
    }

    #[test]
    fn breusch_pagan_reports_a_singular_auxiliary_regression_with_its_context() {
        // `2 * x1`は`x1`と同一の列ではないため除かれず、補助回帰が特異になる。
        let doubled: Vec<f64> = X1.iter().map(|v| 2.0 * v).collect();
        let err = breusch_pagan_test(
            &cols(&[&X1, &doubled]),
            &names(&["x1", "x1_doubled"]),
            &RESID_A,
        )
        .unwrap_err();
        assert_eq!(
            err,
            LeastSquaresError::AuxiliaryRegressionFailed {
                test: DiagnosticTest::BreuschPagan,
                source: Box::new(LeastSquaresError::SingularMatrix {
                    columns: vec!["x1".to_string(), "x1_doubled".to_string()]
                }),
            }
        );
    }

    #[test]
    fn breusch_pagan_fails_for_constant_squared_residuals_and_non_finite_residuals() {
        // 残差の二乗が定数（`±1`）だと`R²`が定義できない。
        let signs: Vec<f64> = (0..15)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        assert_eq!(
            breusch_pagan_test(&cols(&[&X1]), &names(&["x1"]), &signs).unwrap_err(),
            LeastSquaresError::DiagnosticStatisticUndefined {
                test: DiagnosticTest::BreuschPagan
            }
        );

        let mut resid = RESID_A;
        resid[3] = f64::NAN;
        assert_eq!(
            breusch_pagan_test(&cols(&[&X1]), &names(&["x1"]), &resid).unwrap_err(),
            LeastSquaresError::DiagnosticStatisticUndefined {
                test: DiagnosticTest::BreuschPagan
            }
        );
    }

    #[test]
    fn breusch_pagan_does_not_depend_on_the_scale_location_or_order_of_the_variables() {
        let base = breusch_pagan_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_A).unwrap();

        // 平行移動・スケール変更（年のように平均が大きい変数、単位違い）。
        let shifted: Vec<f64> = X1.iter().map(|v| 2000.0 + 1e-3 * v).collect();
        let scaled: Vec<f64> = X2.iter().map(|v| 1e6 * v).collect();
        let rescaled =
            breusch_pagan_test(&cols(&[&shifted, &scaled]), &names(&["x1", "x2"]), &RESID_A)
                .unwrap();
        assert_close(rescaled.lm_statistic, base.lm_statistic, "lm_scale");
        assert_close(rescaled.f_statistic, base.f_statistic, "f_scale");

        // 変数の並べ替え。
        let swapped =
            breusch_pagan_test(&cols(&[&X2, &X1]), &names(&["x2", "x1"]), &RESID_A).unwrap();
        assert_close(swapped.lm_statistic, base.lm_statistic, "lm_order");
        assert_close(swapped.f_p_value, base.f_p_value, "f_p_order");
    }
}
