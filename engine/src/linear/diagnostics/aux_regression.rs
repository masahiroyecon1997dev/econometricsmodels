//! 残差の二乗を補助回帰する検定（White検定・Breusch-Pagan検定）の共通部分: 結果型、補助回帰の項の
//! 選択（定数・重複の除外）、標準化、LM・F統計量とp値。

use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor};

use crate::linear::common::{DiagnosticTest, LeastSquaresError};
use crate::linear::ols::{CovType, OlsEstimator, OlsInput};
use crate::shared::error::CommonError;

/// 補助回帰の2列を「数値的に同一」、1列を「定数」とみなす相対許容誤差。
///
/// `d*d == d`のようなダミー変数の重複や、利用者が事前に作った`x1^2`列と
/// こちらで作る二乗項の1 ULP程度の差を吸収するための値で、列のスケール
/// （最大絶対値）に対する相対値として使う。これより大きい差を持つ列は別の列として扱う。
const AUX_COLUMN_REL_TOL: f64 = 1e-12;

/// 補助回帰の`confidence_level`。検定の統計量・p値には影響しない（`OlsEstimator::fit`の
/// 引数を埋めるだけ）。
const AUX_CONFIDENCE_LEVEL: f64 = 0.95;

/// 残差の二乗を補助回帰する検定（White検定・Breusch-Pagan検定）の結果。
///
/// LM版（`LM = n·R²`、帰無分布は`χ²(lm_df)`）とF版（補助回帰の全傾きがゼロという
/// 古典的なF検定、`F(lm_df, f_df_denom)`）の両方を持つ。どちらを使うかの選択は呼び出し側
/// （`engine_pybind`の`statistic`引数）の責務。
///
/// `aux_terms`は補助回帰に実際に使った項（定数`"const"`が先頭に必ず入る。補助回帰は
/// 元のモデルが定数を持つかに関わらず常に定数を含むため）、`dropped_terms`は重複・定数のため
/// 除いた項。項の表記は結果を人が読むための出力専用ラベル（式としてパースしない）で、
/// White検定では`"x1"`・`"x1^2"`・`"x1:x2"`、Breusch-Pagan検定では列名そのまま。
#[derive(Debug, Clone, PartialEq)]
pub struct AuxRegressionTest {
    pub lm_statistic: f64,
    pub lm_p_value: f64,
    pub f_statistic: f64,
    pub f_p_value: f64,
    /// LM検定の自由度、およびF検定の分子自由度（補助回帰の定数以外の列数`q`）。
    pub df: usize,
    /// F検定の分母自由度`n - q - 1`。
    pub f_df_denom: usize,
    pub aux_terms: Vec<String>,
    pub dropped_terms: Vec<String>,
}

/// いずれかの列の長さが`n`と一致しない場合に`DimensionMismatch`を返す。
pub(super) fn check_column_lengths(
    columns: &[Vec<f64>],
    n: usize,
) -> Result<(), LeastSquaresError> {
    for column in columns {
        if column.len() != n {
            return Err(CommonError::DimensionMismatch {
                y_rows: n,
                x_rows: column.len(),
            }
            .into());
        }
    }
    Ok(())
}

/// 残差の二乗を、採用済みの補助回帰の項（`kept`、定数は含まない）と定数に回帰して
/// LM・F統計量とp値を計算する（White検定・Breusch-Pagan検定の共通部分）。`test`は
/// エラーメッセージの接頭辞に使う検定名。
///
/// 補助回帰の各列は標準化してから回帰する（`R²`は変わらず、列のスケールや平均の大きさによる
/// 誤った失敗を避けるため）。
pub(super) fn squared_residual_aux_test(
    test: DiagnosticTest,
    kept: Vec<(String, Vec<f64>)>,
    dropped_terms: Vec<String>,
    residuals: &[f64],
) -> Result<AuxRegressionTest, LeastSquaresError> {
    let n = residuals.len();
    let q = kept.len();
    if q == 0 {
        return Err(LeastSquaresError::DiagnosticVariablesConstant { test });
    }
    if n <= q + 1 {
        return Err(
            LeastSquaresError::InsufficientObservationsForAuxRegression { test, n, k: q + 1 },
        );
    }

    let (aux_names, aux_columns): (Vec<String>, Vec<Vec<f64>>) = kept.into_iter().unzip();
    // 補助回帰の`R²`は定数ありなら各列の平行移動・スケール変更で変わらない。生の値のままだと
    // 平均が標準偏差より桁違いに大きい変数（賃金・人口・年等）の二乗項が元の列とほぼ共線に
    // なり、`fit`の特異性判定・条件数チェックで誤って失敗するため、各列を標準化して渡す。
    let aux_columns = aux_columns
        .iter()
        .map(|column| standardize(column, test))
        .collect::<Result<Vec<_>, _>>()?;
    // 非有限の残差（または二乗がオーバーフローする大きな残差）のまま補助回帰に渡すと、
    // 共線性とは無関係に係数の共分散が数値的に特異と判定され、原因を取り違えるエラーになる。
    if residuals.iter().any(|u| !u.is_finite()) {
        return Err(LeastSquaresError::DiagnosticStatisticUndefined { test });
    }
    let squared_residuals: Vec<f64> = residuals.iter().map(|u| u * u).collect();
    if squared_residuals.iter().any(|v| !v.is_finite()) {
        return Err(LeastSquaresError::DiagnosticVariableDegenerate { test });
    }
    let input = OlsInput::from_columns(
        &squared_residuals,
        &aux_columns,
        aux_names.clone(),
        true,
        "resid^2".to_string(),
    )?;
    let estimator = OlsEstimator::fit_auxiliary(input, CovType::Classical, AUX_CONFIDENCE_LEVEL)
        .map_err(|e| aux_fit_error(test, e))?;

    let r_squared = validate_aux_r_squared(estimator.r_squared(), test)?;

    let n_f = n as f64;
    let q_f = q as f64;
    let f_df_denom = n - q - 1;

    let lm_statistic = n_f * r_squared;
    let f_statistic = (r_squared / q_f) / ((1.0 - r_squared) / f_df_denom as f64);

    // `ChiSquared::new`・`FisherSnedecor::new`が失敗するのは自由度が0以下（または非有限）の
    // ときだけ。`q >= 1`と`n > q + 1`（`f_df_denom >= 1`）は上で検証済みのため到達不能で、
    // `unwrap`を避けるための`Result`化に過ぎない。
    let lm_p_value = ChiSquared::new(q_f)
        .map_err(|e| CommonError::internal(e.to_string()))?
        .sf(lm_statistic);
    let f_p_value = FisherSnedecor::new(q_f, f_df_denom as f64)
        .map_err(|e| CommonError::internal(e.to_string()))?
        .sf(f_statistic);

    let mut aux_terms = Vec::with_capacity(q + 1);
    aux_terms.push("const".to_string());
    aux_terms.extend(aux_names);

    Ok(AuxRegressionTest {
        lm_statistic,
        lm_p_value,
        f_statistic,
        f_p_value,
        df: q,
        f_df_denom,
        aux_terms,
        dropped_terms,
    })
}

/// 補助回帰の候補項から、定数列（全て0を含む）、`involves_constant`が真の項、先に採用した
/// 項と数値的に同一の列を除く。`(採用した項, 除いた項の名前)`を返す。
pub(super) fn select_aux_terms(
    candidates: Vec<(String, Vec<f64>)>,
    involves_constant: impl Fn(&str) -> bool,
) -> (Vec<(String, Vec<f64>)>, Vec<String>) {
    // 各列の最大絶対値（スケール）。同一判定を相対許容誤差で行うために一度だけ計算する。
    let mut kept: Vec<(String, Vec<f64>)> = Vec::new();
    let mut kept_scales: Vec<f64> = Vec::new();
    let mut dropped: Vec<String> = Vec::new();
    for (name, column) in candidates {
        let scale = max_abs(&column);
        if is_constant(&column, scale) || involves_constant(&name) {
            dropped.push(name);
            continue;
        }
        let duplicate = kept
            .iter()
            .zip(&kept_scales)
            .any(|((_, other), &other_scale)| {
                columns_equal(&column, other, scale.max(other_scale))
            });
        if duplicate {
            dropped.push(name);
        } else {
            kept.push((name, column));
            kept_scales.push(scale);
        }
    }
    (kept, dropped)
}

/// 列を平均0・標準偏差1に標準化する。
///
/// # Errors
/// 標準偏差が0または非有限（二乗項のオーバーフロー等）で標準化できない:
/// `LeastSquaresError`の診断系バリアント
fn standardize(column: &[f64], test: DiagnosticTest) -> Result<Vec<f64>, LeastSquaresError> {
    let n = column.len() as f64;
    let mean = column.iter().sum::<f64>() / n;
    let variance = column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let sd = variance.sqrt();
    if !(sd.is_finite() && sd > 0.0) {
        return Err(LeastSquaresError::DiagnosticVariableDegenerate { test });
    }
    Ok(column.iter().map(|v| (v - mean) / sd).collect())
}

/// 補助回帰の`R²`が検定統計量に使えるか検証し、丸めた値を返す。
///
/// 非有限、または1以上（F統計量の分母が0になり定義できない）はエラー。`f64::max`はNaNを
/// 無視して`0.0`を返してしまうため、下側を丸める前に非有限を弾く。定数ありの補助回帰の
/// `R²`は理論上`[0, 1]`だが、丸め誤差で`-ε`になりうるため下側だけ`0.0`に丸める。
///
/// # Errors
/// `raw`が非有限または`>= 1.0`: `LeastSquaresError`の診断系バリアント
fn validate_aux_r_squared(raw: f64, test: DiagnosticTest) -> Result<f64, LeastSquaresError> {
    if !raw.is_finite() || raw >= 1.0 {
        return Err(LeastSquaresError::DiagnosticStatisticUndefined { test });
    }
    Ok(raw.max(0.0))
}

/// 補助回帰の`fit`・ランク判定が返したエラーを、診断検定（`test`）の補助回帰の失敗として包む。
/// 元のモデルの`fit()`は通過済みなので、そのまま返すと「利用者のモデルが原因」と誤解されるため、
/// `stage`（補助回帰）を付けて報告する。実装の誤りを表す`Internal`だけは、利用者の入力と
/// 無関係なので包まずそのまま返す。
pub(super) fn aux_fit_error(test: DiagnosticTest, err: LeastSquaresError) -> LeastSquaresError {
    match err {
        err @ LeastSquaresError::Common(CommonError::Internal { .. }) => err,
        err => LeastSquaresError::AuxiliaryRegressionFailed {
            test,
            source: Box::new(err),
        },
    }
}

pub(super) fn max_abs(column: &[f64]) -> f64 {
    column.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

/// 列が定数（全て0を含む）か。`max - min`がスケールに対する相対許容誤差以内。
pub(super) fn is_constant(column: &[f64], scale: f64) -> bool {
    let (min, max) = column
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    max - min <= AUX_COLUMN_REL_TOL * scale
}

/// 2列が数値的に同一か。全要素の差がスケールに対する相対許容誤差以内。異なる列は
/// 最初の数要素で不一致になり早期に抜けるため、候補数が多くても重複判定は軽い。
fn columns_equal(a: &[f64], b: &[f64], scale: f64) -> bool {
    a.iter()
        .zip(b)
        .all(|(x, y)| (x - y).abs() <= AUX_COLUMN_REL_TOL * scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::error::ErrorInfo;
    use crate::shared::error_catalog::{ErrorCode, Stage};

    #[test]
    fn validate_aux_r_squared_rejects_undefined_values_and_clamps_rounding_noise() {
        for raw in [1.0, 1.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                validate_aux_r_squared(raw, DiagnosticTest::White),
                Err(LeastSquaresError::DiagnosticStatisticUndefined {
                    test: DiagnosticTest::White
                }),
                "raw={raw}"
            );
        }
        assert_eq!(
            validate_aux_r_squared(0.25, DiagnosticTest::White).unwrap(),
            0.25
        );
        assert_eq!(
            validate_aux_r_squared(-1e-17, DiagnosticTest::White).unwrap(),
            0.0
        );
    }

    #[test]
    fn standardize_rejects_zero_and_non_finite_variance() {
        let degenerate = LeastSquaresError::DiagnosticVariableDegenerate {
            test: DiagnosticTest::White,
        };
        assert_eq!(
            standardize(&[2.0, 2.0, 2.0], DiagnosticTest::White).unwrap_err(),
            degenerate
        );
        assert_eq!(
            standardize(&[1.0, f64::INFINITY, 3.0], DiagnosticTest::White).unwrap_err(),
            degenerate
        );
        let z = standardize(&[1.0, 2.0, 3.0], DiagnosticTest::White).unwrap();
        assert!((z.iter().sum::<f64>()).abs() < 1e-12);
        assert!((z.iter().map(|v| v * v).sum::<f64>() / 3.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn aux_fit_error_wraps_numerical_singularity_of_the_auxiliary_regression_only() {
        let singular = aux_fit_error(
            DiagnosticTest::White,
            LeastSquaresError::SingularMatrix {
                columns: vec!["x1".to_string(), "x1_doubled".to_string()],
            },
        );
        assert_eq!(
            singular,
            LeastSquaresError::AuxiliaryRegressionFailed {
                test: DiagnosticTest::White,
                source: Box::new(LeastSquaresError::SingularMatrix {
                    columns: vec!["x1".to_string(), "x1_doubled".to_string()]
                }),
            }
        );
        let report = singular.report();
        assert_eq!(report.code(), ErrorCode::DesignMatrixSingular);
        assert_eq!(
            report.stage(),
            Some(&Stage::DiagnosticAuxiliaryRegression {
                test: "WHITE".to_string()
            })
        );
        assert!(report.validate().is_ok());

        let near_singular = aux_fit_error(
            DiagnosticTest::BreuschPagan,
            LeastSquaresError::Common(CommonError::NearSingularMatrix {
                matrix: crate::shared::error::MatrixKind::CoefficientCovariance,
            }),
        );
        let report = near_singular.report();
        assert_eq!(report.code(), ErrorCode::MatrixNearSingular);
        assert!(report.stage().is_some());
        assert!(report.validate().is_ok());

        // 実装の誤り（`Internal`）は利用者の入力と無関係なので、包まずそのまま通す。
        let internal = LeastSquaresError::Common(CommonError::internal("bug"));
        assert_eq!(
            aux_fit_error(DiagnosticTest::White, internal),
            LeastSquaresError::Common(CommonError::internal("bug"))
        );
    }

    #[test]
    fn columns_equal_and_is_constant_use_relative_tolerance() {
        // 1ULP程度の差は同一、スケールに対して十分大きい差は別の列。
        let a = [1.0e8, 2.0e8, 3.0e8];
        let near = [1.0e8 + 1e-6, 2.0e8, 3.0e8];
        let far = [1.0e8 + 10.0, 2.0e8, 3.0e8];
        assert!(columns_equal(&a, &near, 3.0e8));
        assert!(!columns_equal(&a, &far, 3.0e8));
        assert!(is_constant(&[0.0, 0.0, 0.0], 0.0));
        assert!(is_constant(&[5.0, 5.0 + 1e-13, 5.0], 5.0));
        assert!(!is_constant(&[5.0, 5.1, 5.0], 5.1));
    }
}
