//! White検定（不均一分散の検定）。補助回帰の項（`x`・二乗・交差項）を作り、共通部分に渡す。

use super::aux_regression::{
    AuxRegressionTest, check_column_lengths, is_constant, max_abs, select_aux_terms,
    squared_residual_aux_test,
};
use crate::linear::common::{DiagnosticTest, LeastSquaresError};

/// White検定の結果（[`AuxRegressionTest`]）。
pub type WhiteTest = AuxRegressionTest;

/// White検定（不均一分散の検定）。残差の二乗を、`x`・`x`の二乗・`x`同士の交差項に回帰する
/// 補助回帰を行い、`LM = n·R²`（`χ²(q)`）とF版を計算する。
///
/// `x_columns`は元のモデルの説明変数（定数列は含まない）、`residuals`は元のモデルの残差。
/// 補助回帰には常に定数を含める（元のモデルが`include_intercept=false`でも同じ）。
///
/// 補助回帰の項は`x`の各列・各列の二乗・列の組の積の順に作り、定数列（全て0を含む）と、
/// それより前に採用した項と数値的に同一の列（ダミー変数の二乗`d*d == d`等）を除く
/// （共通部分の定数`AUX_COLUMN_REL_TOL`）。除いたあとの項の数が自由度`q`になる。元のモデルは完全な
/// 多重共線性を`fit()`で弾いているため、ここで残りうるランク落ちは主にダミー変数由来で、
/// 上の2種類で足りる。除いたあとも数値的に共線の場合はエラーにする（`fit()`と同じ方針）。
/// 補助回帰の各列は標準化してから回帰する（`R²`は変わらず、列のスケールや平均の大きさによる
/// 誤った失敗を避けるため）。
///
/// # Errors
/// - いずれかの列の長さが`residuals`と一致しない: `CommonError::DimensionMismatch`
/// - 除外後に定数以外の補助回帰の項が1つも残らない（全ての`x`が定数）:
///   `LeastSquaresError`の診断系バリアント
/// - 観測数`n`が補助回帰の列数（定数を含む`q + 1`）以下:
///   `LeastSquaresError::InsufficientObservationsForAuxRegression`
/// - 補助回帰の設計行列が除外後も特異（全カテゴリのダミーと定数の組み合わせ等）、
///   または条件数が悪く推定できない: `LeastSquaresError`の診断系バリアント
///   （元のモデルの`SingularMatrix`と区別するため、White検定の補助回帰であることを
///   メッセージに含める）
/// - 補助回帰の`R²`が非有限、または1（残差の二乗が定数、または補助回帰で完全に説明され
///   てF統計量が定義できない）: `LeastSquaresError`の診断系バリアント
///
/// # パニックについて
/// `x_names.len() != x_columns.len()`は`engine_pybind`の実装バグでしか起こらない内部契約
/// なので`debug_assert!`でパニックさせる（`OlsInput::from_columns`と同じ扱い）。
pub fn white_test(
    x_columns: &[Vec<f64>],
    x_names: &[String],
    residuals: &[f64],
) -> Result<WhiteTest, LeastSquaresError> {
    debug_assert_eq!(x_columns.len(), x_names.len());
    check_column_lengths(x_columns, residuals.len())?;

    let (kept, dropped_terms) = build_white_aux_terms(x_columns, x_names);
    squared_residual_aux_test(DiagnosticTest::White, kept, dropped_terms, residuals)
}

/// 補助回帰の候補項（`x`・二乗・交差項、この順）を作り、定数列と、先に採用した項と
/// 数値的に同一の列を除く。`(採用した項(名前, 列), 除いた項の名前)`を返す。
fn build_white_aux_terms(
    x_columns: &[Vec<f64>],
    x_names: &[String],
) -> (Vec<(String, Vec<f64>)>, Vec<String>) {
    let p = x_columns.len();
    let mut candidates: Vec<(String, Vec<f64>)> = Vec::with_capacity(p * (p + 3) / 2);
    for (name, column) in x_names.iter().zip(x_columns) {
        candidates.push((name.clone(), column.clone()));
    }
    for (name, column) in x_names.iter().zip(x_columns) {
        candidates.push((format!("{name}^2"), column.iter().map(|v| v * v).collect()));
    }
    for i in 0..p {
        for j in (i + 1)..p {
            candidates.push((
                format!("{}:{}", x_names[i], x_names[j]),
                x_columns[i]
                    .iter()
                    .zip(&x_columns[j])
                    .map(|(a, b)| a * b)
                    .collect(),
            ));
        }
    }

    // 定数の説明変数（`include_intercept=false`のモデルに定数列を入れた場合）を含む項
    // （二乗・交差項）は、その定数倍の別の項（または定数）になって冗長なので、列の値の
    // 比較を待たずに除く（`c * x_k`は`c != 1`だと`x_k`と数値的に同一にならず、重複判定
    // だけでは見逃してランク落ちになるため）。
    let constant_vars: Vec<bool> = x_columns
        .iter()
        .map(|column| is_constant(column, max_abs(column)))
        .collect();
    let involves_constant_var = |name: &str| {
        x_names
            .iter()
            .zip(&constant_vars)
            .any(|(var, &is_const)| is_const && term_uses_variable(name, var))
    };
    select_aux_terms(candidates, involves_constant_var)
}

/// 項のラベル（`x`・`x^2`・`x1:x2`）が変数`var`を含むか。ラベルは`build_white_aux_terms`が
/// 作った形に限るため、`^2`・`:`で分けた各部分と変数名の完全一致で判定する。
fn term_uses_variable(label: &str, var: &str) -> bool {
    if label == var {
        return true;
    }
    if let Some(base) = label.strip_suffix("^2") {
        return base == var;
    }
    // `a:b`は列名に`:`を含むと曖昧になるため、全ての分割位置を試す。
    label
        .match_indices(':')
        .any(|(i, _)| &label[..i] == var || &label[i + 1..] == var)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;
    use crate::shared::error::CommonError;

    #[test]
    fn white_matches_statsmodels_for_continuous_regressors() {
        let result = white_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_A).unwrap();

        assert_close(result.lm_statistic, 12.420378012065749, "lm");
        assert_close(result.lm_p_value, 0.029460248325868164, "lm_p");
        assert_close(result.f_statistic, 8.66664981392156, "f");
        assert_close(result.f_p_value, 0.0029908305430781878, "f_p");
        assert_eq!(result.df, 5);
        assert_eq!(result.f_df_denom, 9);
        assert_eq!(
            result.aux_terms,
            names(&["const", "x1", "x2", "x1^2", "x2^2", "x1:x2"])
        );
        assert!(result.dropped_terms.is_empty());
    }

    #[test]
    fn white_df_equals_aux_terms_minus_constant() {
        let result = white_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_A).unwrap();
        assert_eq!(result.aux_terms[0], "const");
        assert_eq!(result.df, result.aux_terms.len() - 1);
    }

    #[test]
    fn white_uses_a_constant_in_the_auxiliary_regression_for_no_intercept_residuals() {
        // 元のモデルが定数なし（残差の和が0とは限らない）でも補助回帰は定数を含む。
        let result = white_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_C).unwrap();

        assert_close(result.lm_statistic, 8.42552364235829, "lm");
        assert_close(result.lm_p_value, 0.1342911746633339, "lm_p");
        assert_close(result.f_statistic, 2.3067909490034277, "f");
        assert_close(result.f_p_value, 0.1303758805692828, "f_p");
        assert_eq!(result.aux_terms[0], "const");
    }

    #[test]
    fn white_drops_squared_dummy_and_uses_rank_based_degrees_of_freedom() {
        // `d*d == d`なので`d^2`は`d`と同一の列として除かれ、自由度は5ではなく4になる。
        let result = white_test(&cols(&[&X1, &D]), &names(&["x1", "d"]), &RESID_B).unwrap();

        assert_eq!(result.dropped_terms, names(&["d^2"]));
        assert_eq!(
            result.aux_terms,
            names(&["const", "x1", "d", "x1^2", "x1:d"])
        );
        assert_eq!(result.df, 4);
        assert_eq!(result.f_df_denom, 10);
        assert_close(result.lm_statistic, 13.755561618633275, "lm");
        assert_close(result.lm_p_value, 0.008117487707630353, "lm_p");
        assert_close(result.f_statistic, 27.634075388140147, "f");
        assert_close(result.f_p_value, 2.195070259871854e-05, "f_p");
    }

    #[test]
    fn white_drops_constant_independent_variable() {
        // `include_intercept=false`のモデルで定数列を`x`に入れたケース。補助回帰の定数と
        // 重複するので、定数列とその二乗・交差項は全て除かれる。
        let ones = [1.0_f64; 15];
        let result = white_test(&cols(&[&ones, &X1]), &names(&["one", "x1"]), &RESID_A).unwrap();

        assert_eq!(result.dropped_terms, names(&["one", "one^2", "one:x1"]));
        assert_eq!(result.aux_terms, names(&["const", "x1", "x1^2"]));
        assert_eq!(result.df, 2);
    }

    #[test]
    fn white_fails_when_all_independent_variables_are_constant() {
        let ones = [1.0_f64; 15];
        let err = white_test(&cols(&[&ones]), &names(&["one"]), &RESID_A).unwrap_err();
        assert_eq!(
            err,
            LeastSquaresError::DiagnosticVariablesConstant {
                test: DiagnosticTest::White
            }
        );
    }

    #[test]
    fn white_rejects_mismatched_column_length() {
        let short = [0.1_f64, 0.2, 0.3];
        let err = white_test(&cols(&[&short]), &names(&["x1"]), &RESID_A).unwrap_err();
        assert_eq!(
            err,
            LeastSquaresError::Common(CommonError::DimensionMismatch {
                y_rows: 15,
                x_rows: 3
            })
        );
    }

    #[test]
    fn white_rejects_too_few_observations_for_the_auxiliary_regression() {
        // 2変数で補助回帰は定数込み6列。n=6以下は弾く（n=7なら通る）。
        let x1 = &X1[..6];
        let x2 = &X2[..6];
        let err = white_test(&cols(&[x1, x2]), &names(&["x1", "x2"]), &RESID_A[..6]).unwrap_err();
        assert_eq!(
            err,
            LeastSquaresError::InsufficientObservationsForAuxRegression {
                test: DiagnosticTest::White,
                n: 6,
                k: 6
            }
        );
    }

    #[test]
    fn white_reports_singular_when_auxiliary_columns_are_still_collinear() {
        // `x2 = x1 + 1`だと`x2^2`・`x1:x2`が`1`・`x1`・`x1^2`の線形結合になり、重複でも
        // 定数でもないままランク落ちする。White検定の補助回帰であることが分かる
        // `AuxiliaryRegressionFailed`（`DESIGN_MATRIX_SINGULAR`に`stage`が付く）にする。
        let x2: Vec<f64> = X1.iter().map(|v| v + 1.0).collect();
        let err = white_test(&cols(&[&X1, &x2]), &names(&["x1", "x2"]), &RESID_A).unwrap_err();
        assert!(
            matches!(
                &err,
                LeastSquaresError::AuxiliaryRegressionFailed {
                    test: DiagnosticTest::White,
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn white_fails_when_squared_residuals_are_constant() {
        // 残差の二乗が全観測で同じだと補助回帰のTSSが0で`R²`が定義できない。
        let resid = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let x1 = &X1[..10];
        let err = white_test(&cols(&[x1]), &names(&["x1"]), &resid).unwrap_err();
        assert_eq!(
            err,
            LeastSquaresError::DiagnosticStatisticUndefined {
                test: DiagnosticTest::White
            }
        );
    }

    #[test]
    fn white_p_values_are_in_the_open_unit_interval() {
        let result = white_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_A).unwrap();
        for p in [result.lm_p_value, result.f_p_value] {
            assert!(p > 0.0 && p < 1.0, "p={p}");
        }
    }

    #[test]
    fn white_does_not_depend_on_the_scale_or_location_of_the_variables() {
        // 平均が標準偏差より桁違いに大きい列（賃金・人口等）でも、補助回帰の`R²`は
        // スケール・平行移動で変わらないため同じ結果になる。標準化しないと二乗項が
        // 元の列とほぼ共線になり`AuxiliaryRegressionFailed`で誤って失敗する。
        let base = white_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_A).unwrap();
        let big1: Vec<f64> = X1.iter().map(|v| 5.0e3 + 1.0e3 * v).collect();
        let big2: Vec<f64> = X2.iter().map(|v| 4.0e4 + 10.0 * v).collect();
        let scaled = white_test(&cols(&[&big1, &big2]), &names(&["x1", "x2"]), &RESID_A).unwrap();

        assert_eq!(scaled.aux_terms, base.aux_terms);
        assert!((scaled.lm_statistic - base.lm_statistic).abs() < 1e-6);
        assert!((scaled.f_statistic - base.f_statistic).abs() < 1e-6);
        assert!((scaled.lm_p_value - base.lm_p_value).abs() < 1e-8);
    }

    #[test]
    fn white_drops_terms_that_involve_a_constant_other_than_one() {
        // 定数2.0の列`c`を含む交差項`c:x1`は`2*x1`で`x1`と数値的に同一にならないため、
        // 値の比較だけでは重複を見逃してランク落ちになる。定数変数を含む項は全て除く。
        let twos = [2.0_f64; 15];
        let result = white_test(&cols(&[&twos, &X1]), &names(&["c", "x1"]), &RESID_A).unwrap();

        assert_eq!(result.dropped_terms, names(&["c", "c^2", "c:x1"]));
        assert_eq!(result.aux_terms, names(&["const", "x1", "x1^2"]));
    }

    #[test]
    fn white_drops_the_interaction_of_mutually_exclusive_dummies() {
        // 排他的なダミー`d1`・`d2`は積が全て0の定数列になるため除かれる（`d1^2`・`d2^2`も
        // それぞれ`d1`・`d2`と同一）。
        let d1 = [
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ];
        let d2 = [
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ];
        let result = white_test(&cols(&[&d1, &d2]), &names(&["d1", "d2"]), &RESID_A).unwrap();

        assert_eq!(result.dropped_terms, names(&["d1^2", "d2^2", "d1:d2"]));
        assert_eq!(result.aux_terms, names(&["const", "d1", "d2"]));
        assert_eq!(result.df, 2);
    }

    #[test]
    fn white_rejects_non_finite_residuals_instead_of_masking_nan() {
        // `f64::max`はNaNを無視して`0.0`を返すため、`R²`が非有限のまま丸めると
        // 統計量が0.0として静かに返ってしまう。非有限の残差はエラーにする。
        let mut resid = RESID_A;
        resid[3] = f64::NAN;
        let err = white_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &resid).unwrap_err();
        assert_eq!(
            err,
            LeastSquaresError::DiagnosticStatisticUndefined {
                test: DiagnosticTest::White
            }
        );
    }

    #[test]
    fn white_succeeds_at_the_smallest_sample_size_with_one_denominator_degree_of_freedom() {
        // 2変数の補助回帰は定数込み6列。観測数`n = 7`（`k + 1`）が成功する最小で、
        // `df_denom = n - q - 1 = 1`。`n = 6`（`n = k`）は拒否される
        // （`white_rejects_too_few_observations_for_the_auxiliary_regression`）。
        let x1 = &X1[..7];
        let x2 = &X2[..7];
        let result = white_test(&cols(&[x1, x2]), &names(&["x1", "x2"]), &RESID_A[..7]).unwrap();

        assert_eq!(result.df, 5);
        assert_eq!(result.f_df_denom, 1);
        assert!(result.lm_statistic.is_finite() && result.f_statistic.is_finite());
    }

    #[test]
    fn white_checks_the_sample_size_against_the_terms_that_remain_after_dropping() {
        // `x1`と0/1の`d`は名目で補助回帰が定数込み6列だが、`d^2`を除くと5列。観測数6は
        // 名目の列数では足りない（`n <= 6`）が、除外後の列数では足りる（`n > 5`）。
        let x1 = [0.5, 1.2, -0.3, 2.1, 0.9, -1.4];
        let d = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let resid = [0.3, -0.8, 0.5, 1.4, -0.2, 0.1];
        let result = white_test(&cols(&[&x1, &d]), &names(&["x1", "d"]), &resid).unwrap();
        assert_eq!(result.df, 4);
        assert_eq!(result.f_df_denom, 1);

        // 5観測では除外後の列数（k = 5）でも足りず、メッセージの`k`も除外後の列数になる。
        let err = white_test(
            &cols(&[&x1[..5], &d[..5]]),
            &names(&["x1", "d"]),
            &resid[..5],
        )
        .unwrap_err();
        assert_eq!(
            err,
            LeastSquaresError::InsufficientObservationsForAuxRegression {
                test: DiagnosticTest::White,
                n: 5,
                k: 5
            }
        );
    }

    #[test]
    fn white_fails_when_squaring_overflows() {
        // |x| > 1e154だと二乗が`inf`になり標準化できない。
        let huge: Vec<f64> = X1.iter().map(|v| v * 1e200).collect();
        let err = white_test(&cols(&[&huge]), &names(&["x1"]), &RESID_A).unwrap_err();
        assert_eq!(
            err,
            LeastSquaresError::DiagnosticVariableDegenerate {
                test: DiagnosticTest::White
            }
        );
    }

    #[test]
    fn white_is_invariant_to_the_order_of_the_variables() {
        let a = white_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_A).unwrap();
        let b = white_test(&cols(&[&X2, &X1]), &names(&["x2", "x1"]), &RESID_A).unwrap();
        assert!((a.lm_statistic - b.lm_statistic).abs() < 1e-9);
        assert!((a.f_statistic - b.f_statistic).abs() < 1e-9);
        assert_eq!(a.df, b.df);
    }

    #[test]
    fn term_uses_variable_matches_whole_variable_names_only() {
        assert!(term_uses_variable("x1", "x1"));
        assert!(term_uses_variable("x1^2", "x1"));
        assert!(term_uses_variable("x1:x2", "x2"));
        assert!(!term_uses_variable("x10^2", "x1"));
        assert!(!term_uses_variable("x10:x2", "x1"));
    }
}
