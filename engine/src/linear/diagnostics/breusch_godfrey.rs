//! Breusch-Godfrey検定（系列相関の検定）。`OlsEstimator::fit`を使わず、列ノルムでスケールした
//! 列ピボットQRで補助回帰の残差二乗和だけを求める。

use faer::Mat;
use faer::prelude::SolveLstsq;
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor};

use super::aux_regression::aux_fit_error;
use crate::linear::common::{DiagnosticTest, LeastSquaresError};
use crate::shared::covariance::time_ordering;
use crate::shared::error::CommonError;
use crate::shared::linear_algebra::checked_col_piv_qr;

/// Breusch-Godfrey検定（系列相関の検定）の結果。
///
/// LM版（`LM = n·R²`、帰無分布`χ²(df)`）とF版（`F(df, f_df_denom)`）の両方を持つ。
/// 選択は呼び出し側（`engine_pybind`の`statistic`引数）の責務。`df`はラグ次数`nlags`。
#[derive(Debug, Clone, PartialEq)]
pub struct BreuschGodfreyTest {
    pub lm_statistic: f64,
    pub lm_p_value: f64,
    pub f_statistic: f64,
    pub f_p_value: f64,
    /// LM検定の自由度、およびF検定の分子自由度（ラグ次数`nlags`）。
    pub df: usize,
    /// F検定の分母自由度`n - k - nlags`（`k`は元のモデルの係数の数）。
    pub f_df_denom: usize,
}

/// Breusch-Godfrey検定。残差`û`を、元のモデルの説明変数`X`と`û`自身の1〜`nlags`次のラグ
/// （時間順に並べた残差の遅れ）に回帰する補助回帰を行う。
///
/// - 補助回帰の説明変数は**元のモデルの`X`をそのまま**使い、元のモデルが定数を持たなければ
///   定数を足さない（R `lmtest::bgtest`・Greeneの定義。statsmodelsは切片なしのモデルでだけ
///   補助回帰に定数を足すため定義が異なる）。`has_intercept`なら`X`の先頭に定数列を足す。
/// - サンプル前期間のラグは0で埋める（statsmodels・R `bgtest`の既定・Stataと同じ）。
///   このため補助回帰の観測数は常に`n`。
/// - 観測の時間順は`time_order`（値の昇順が時間順、同値が無いこと）で決める。行順を時間順と
///   みなす暗黙の既定は置かない。値の間隔（欠番）は見ず、並べた順にラグを取る。
///
/// 前提: `residuals`は`x_columns`（と`has_intercept`の定数）を説明変数とするOLSの残差
/// （`X`と直交する）。この前提のもとで`û`を`X`に回帰した制約モデルの残差二乗和は`Σû²`に
/// 等しく、補助回帰の残差二乗和`SSR_u`から`LM = n·(1 - SSR_u/Σû²)`、
/// `F = ((Σû² - SSR_u)/m)/(SSR_u/(n - k - m))`（`m = nlags`）と書ける（R `bgtest`と同じ式）。
/// WLSの残差のように`X`と直交しない残差を渡すとFの分子が過大になるため、そのような入力に
/// 流用する場合は変換後の残差と`X`を渡すこと。補助回帰の列は列ノルムでスケールして解く
/// （切片の有無によらず`SSR_u`は列の正のスケールで変わらず、切片なしのモデルは元の`X`のまま
/// 扱うため中心化はしない）。
///
/// # Errors
/// - `nlags < 1`: `LeastSquaresError::InvalidNlags`
/// - いずれかの列・`time_order`の長さが`residuals`と一致しない: `CommonError::DimensionMismatch`
/// - 観測数`n`が補助回帰の列数`k + nlags`以下:
///   `LeastSquaresError::InsufficientObservationsForAuxRegression`
/// - 補助回帰の設計行列が特異（ラグが`X`と共線等）: `LeastSquaresError`の診断系バリアント
///   （元のモデルの`SingularMatrix`と区別するため、補助回帰であることをメッセージに含める）
/// - 残差が全て0、補助回帰が残差を完全に説明する、または統計量が非有限:
///   `LeastSquaresError`の診断系バリアント
///
/// # パニックについて
/// `time_order`に`NaN`が無いことが前提（`engine_pybind`が順位に変換済み）。
pub fn breusch_godfrey_test(
    x_columns: &[Vec<f64>],
    has_intercept: bool,
    residuals: &[f64],
    time_order: &[f64],
    nlags: i64,
) -> Result<BreuschGodfreyTest, LeastSquaresError> {
    if nlags < 1 {
        return Err(LeastSquaresError::InvalidNlags { nlags });
    }
    let n = residuals.len();
    for column in x_columns.iter().map(Vec::len).chain([time_order.len()]) {
        if column != n {
            return Err(CommonError::DimensionMismatch {
                y_rows: n,
                x_rows: column,
            }
            .into());
        }
    }

    let k = x_columns.len() + usize::from(has_intercept);
    // `nlags`が`usize`に収まらない巨大な値でも、観測数不足として扱えるよう飽和させる。
    let m = usize::try_from(nlags).unwrap_or(usize::MAX);
    let k_aux = k.saturating_add(m);
    if n <= k_aux {
        return Err(
            LeastSquaresError::InsufficientObservationsForAuxRegression {
                test: DiagnosticTest::BreuschGodfrey,
                n,
                k: k_aux,
            },
        );
    }

    let order = time_ordering(time_order, n);
    let sorted_resid: Vec<f64> = order.iter().map(|&i| residuals[i]).collect();
    let ssr_restricted: f64 = sorted_resid.iter().map(|u| u * u).sum();
    if !ssr_restricted.is_finite() || ssr_restricted <= 0.0 {
        return Err(LeastSquaresError::DiagnosticStatisticUndefined {
            test: DiagnosticTest::BreuschGodfrey,
        });
    }

    let mut aux = allocate_aux_matrix(n, k_aux)?;
    aux.resize_with(n, k_aux, |t, j| {
        if has_intercept && j == 0 {
            1.0
        } else if j < k {
            x_columns[j - usize::from(has_intercept)][order[t]]
        } else {
            // 列`k + l - 1`は`l`次のラグ（サンプル前期間は0）。
            let lag = j - k + 1;
            if t >= lag { sorted_resid[t - lag] } else { 0.0 }
        }
    });
    scale_columns_by_norm(&mut aux)?;

    crate::shared::parallelism::ensure_serial();
    let qr = checked_col_piv_qr(&aux).map_err(|err| {
        // 補助回帰の列は元のモデルの列名を持たないため、位置で名前を付ける。
        let mut names: Vec<String> = Vec::with_capacity(k_aux);
        if has_intercept {
            names.push("const".to_string());
        }
        names.extend((0..x_columns.len()).map(|j| format!("x[{j}]")));
        names.extend((1..=m).map(|lag| format!("residual_lag_{lag}")));
        aux_fit_error(
            DiagnosticTest::BreuschGodfrey,
            LeastSquaresError::from_rank_deficient(err, &aux, &names),
        )
    })?;
    let params = qr.solve_lstsq(Mat::from_fn(n, 1, |t, _| sorted_resid[t]));
    let fitted = &aux * &params;
    let ssr_unrestricted: f64 = (0..n)
        .map(|t| (sorted_resid[t] - *fitted.get(t, 0)).powi(2))
        .sum();
    // 補助回帰が残差を完全に説明するとFの分母が0になる。丸めで厳密な0にはならないため、
    // 補助回帰の列数に応じた相対許容誤差で判定する。
    if !ssr_unrestricted.is_finite()
        || ssr_unrestricted <= (k_aux as f64) * f64::EPSILON * ssr_restricted
    {
        return Err(LeastSquaresError::DiagnosticStatisticUndefined {
            test: DiagnosticTest::BreuschGodfrey,
        });
    }

    let n_f = n as f64;
    let m_f = m as f64;
    let f_df_denom = n - k_aux;
    // 丸め誤差で`SSR_u`が`Σû²`をわずかに超えうるため、説明される分だけ0に丸める。
    let explained = (ssr_restricted - ssr_unrestricted).max(0.0);
    let lm_statistic = n_f * explained / ssr_restricted;
    let f_statistic = (explained / m_f) / (ssr_unrestricted / f_df_denom as f64);

    // `ChiSquared::new`・`FisherSnedecor::new`が失敗するのは自由度が0以下（または非有限）の
    // ときだけ。`m >= 1`と`n > k + m`（`f_df_denom >= 1`）は上で検証済みのため到達不能。
    let lm_p_value = ChiSquared::new(m_f)
        .map_err(|e| CommonError::internal(e.to_string()))?
        .sf(lm_statistic);
    let f_p_value = FisherSnedecor::new(m_f, f_df_denom as f64)
        .map_err(|e| CommonError::internal(e.to_string()))?
        .sf(f_statistic);

    Ok(BreuschGodfreyTest {
        lm_statistic,
        lm_p_value,
        f_statistic,
        f_p_value,
        df: m,
        f_df_denom,
    })
}

/// 補助回帰の行列`n × k_aux`を、確保に失敗しても異常終了せずエラーにして確保する（中身は
/// 未初期化ではなく、呼び出し側が`resize_with`で埋める空の行列を返す）。
///
/// `nlags`が`n`に近いと行列が`n × n`に近づき、`n`が大きいと確保できずプロセスが異常終了
/// しうる。`nlags`に割合や固定値の上限を置くと統計的に正当化できないため、サイズの計算
/// （オーバーフロー）と確保の失敗だけを`BreuschGodfreyMatrixTooLarge`にする。QR分解は同サイズの
/// 複製を内部で確保する（失敗を検知できない）ため、その分もここで確保できるか試す。
///
/// # Errors
/// 要素数がオーバーフローする、または確保できない: `LeastSquaresError`の診断系バリアント
fn allocate_aux_matrix(n: usize, k_aux: usize) -> Result<Mat<f64>, LeastSquaresError> {
    let too_large = || LeastSquaresError::BreuschGodfreyMatrixTooLarge {
        n_obs: n,
        n_columns: k_aux,
    };
    let cells = n.checked_mul(k_aux).ok_or_else(too_large)?;
    let mut matrix = Mat::<f64>::new();
    matrix.try_reserve(n, k_aux).map_err(|_| too_large())?;
    // QR分解が内部で確保する同サイズの複製を、確保できるか先に試す（すぐ解放する）。
    Vec::<f64>::new()
        .try_reserve_exact(cells)
        .map_err(|_| too_large())?;
    Ok(matrix)
}

/// 各列をユークリッドノルムで割る（列のスケール差による誤った特異判定を避ける）。ノルムが
/// 0または非有限の列（全て0の列等）はそのまま残し、後段のランク判定で特異として扱う。
/// 追加の確保を避けるためその場で書き換える。
///
/// # Errors
/// 行列に非有限の値が含まれる: `LeastSquaresError`の診断系バリアント
fn scale_columns_by_norm(matrix: &mut Mat<f64>) -> Result<(), LeastSquaresError> {
    let (n, k) = (matrix.nrows(), matrix.ncols());
    // 最大絶対値で先に割ってからノルムを取る（要素が1e154超・1e-162未満でも二乗和が
    // オーバーフロー・アンダーフローしないように）。
    let norms: Vec<f64> = (0..k)
        .map(|j| {
            let max_abs = (0..n).fold(0.0_f64, |acc, t| acc.max(matrix.get(t, j).abs()));
            if max_abs > 0.0 && max_abs.is_finite() {
                max_abs
                    * (0..n)
                        .map(|t| (matrix.get(t, j) / max_abs).powi(2))
                        .sum::<f64>()
                        .sqrt()
            } else {
                max_abs
            }
        })
        .collect();
    if norms.iter().any(|v| !v.is_finite()) {
        return Err(LeastSquaresError::DiagnosticVariableDegenerate {
            test: DiagnosticTest::BreuschGodfrey,
        });
    }
    for (j, &norm) in norms.iter().enumerate() {
        if norm > 0.0 {
            for t in 0..n {
                *matrix.get_mut(t, j) /= norm;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;
    use crate::shared::error::ErrorInfo;
    use crate::shared::error_catalog::{ErrorCode, Stage};

    // Breusch-Godfrey用。`RESID_TIME_ORDER`は`y = 1 + 0.8*x1 - 0.5*x2 + u`（`u`は係数0.6の
    // AR(1)誤差）を時間順に並べてOLSした残差、`RESID_NOCONST`は同じ`y`を定数なしで
    // OLSした残差。期待値は切片ありがstatsmodels `acorr_breusch_godfrey`とR `lmtest::bgtest`
    // （`order = 1, 3`、`type = "Chisq"/"F"`、`fill = 0`）、切片なしはR `bgtest`
    // （statsmodelsは切片なしのモデルで補助回帰に定数を足すため使わない）。
    const RESID_TIME_ORDER: [f64; 15] = [
        0.06552583436527404,
        -0.5997757950579707,
        -0.06843863041649528,
        1.114676352485343,
        0.8280225544894275,
        0.4061920390911742,
        -0.964645282709298,
        0.2177511096865523,
        -0.6232545215088097,
        0.6204082400091209,
        -1.179617741776337,
        -0.40540689951312414,
        -0.10615559607639535,
        0.8325534027371426,
        -0.13783506580561355,
    ];

    const RESID_NOCONST: [f64; 15] = [
        0.6043813449503327,
        0.1744310040646766,
        0.8386532321134453,
        0.8171980075706022,
        1.948362051445645,
        1.894504749252854,
        -0.7828427276437557,
        1.5021839245453197,
        0.23494616323682427,
        1.0817693918348765,
        -0.8911913694976559,
        0.40390260744820305,
        1.453415084720323,
        1.2827311581375322,
        0.6316878637553516,
    ];

    /// 行`i`の時間は`PERM[i]`（行を並べ替えて時間列で戻せることの確認用）。
    const PERM: [usize; 15] = [7, 2, 11, 0, 13, 5, 9, 14, 3, 12, 1, 8, 10, 4, 6];

    fn time_order_identity() -> Vec<f64> {
        (0..15).map(|t| t as f64).collect()
    }

    fn assert_bg(result: &BreuschGodfreyTest, lm: f64, lm_p: f64, f: f64, f_p: f64, label: &str) {
        assert_close(result.lm_statistic, lm, &format!("{label}/lm"));
        assert_close(result.lm_p_value, lm_p, &format!("{label}/lm_p"));
        assert_close(result.f_statistic, f, &format!("{label}/f"));
        assert_close(result.f_p_value, f_p, &format!("{label}/f_p"));
    }

    #[test]
    fn breusch_godfrey_matches_statsmodels_and_r_with_an_intercept() {
        let x = cols(&[&X1, &X2]);
        let time = time_order_identity();

        let one = breusch_godfrey_test(&x, true, &RESID_TIME_ORDER, &time, 1).unwrap();
        assert_bg(
            &one,
            0.046548638347058136,
            0.8291815842612812,
            0.034241929132876936,
            0.8565608495840493,
            "m=1",
        );
        assert_eq!((one.df, one.f_df_denom), (1, 11));

        let three = breusch_godfrey_test(&x, true, &RESID_TIME_ORDER, &time, 3).unwrap();
        assert_bg(
            &three,
            4.400180501732203,
            0.22136865106975476,
            1.2453553107535298,
            0.3496049058383928,
            "m=3",
        );
        assert_eq!((three.df, three.f_df_denom), (3, 9));
    }

    #[test]
    fn breusch_godfrey_does_not_add_a_constant_for_a_model_without_an_intercept() {
        // R `bgtest`・Greeneの定義（補助回帰は元の`X`＋残差のラグ、定数を足さない）。
        let x = cols(&[&X1, &X2]);
        let time = time_order_identity();

        let one = breusch_godfrey_test(&x, false, &RESID_NOCONST, &time, 1).unwrap();
        assert_bg(
            &one,
            2.16721887092398,
            0.140980995068219,
            2.02657757422224,
            0.180046149039160,
            "noconst m=1",
        );
        assert_eq!(one.f_df_denom, 15 - 2 - 1);

        let three = breusch_godfrey_test(&x, false, &RESID_NOCONST, &time, 3).unwrap();
        assert_bg(
            &three,
            4.38167627945115,
            0.223090447716491,
            1.37550784686526,
            0.306071107743275,
            "noconst m=3",
        );
    }

    #[test]
    fn breusch_godfrey_orders_the_observations_by_the_time_column_not_by_row() {
        // 行を並べ替え、時間列（`PERM`）を付ければ、時間順に並べた結果と一致する。
        let x1: Vec<f64> = PERM.iter().map(|&t| X1[t]).collect();
        let x2: Vec<f64> = PERM.iter().map(|&t| X2[t]).collect();
        let resid: Vec<f64> = PERM.iter().map(|&t| RESID_TIME_ORDER[t]).collect();
        let time: Vec<f64> = PERM.iter().map(|&t| t as f64).collect();

        let shuffled = breusch_godfrey_test(&cols(&[&x1, &x2]), true, &resid, &time, 3).unwrap();
        let ordered = breusch_godfrey_test(
            &cols(&[&X1, &X2]),
            true,
            &RESID_TIME_ORDER,
            &time_order_identity(),
            3,
        )
        .unwrap();

        assert_bg(
            &shuffled,
            ordered.lm_statistic,
            ordered.lm_p_value,
            ordered.f_statistic,
            ordered.f_p_value,
            "shuffled",
        );
        // 行順のまま（時間順を無視）だと別の値になる。
        let ignored =
            breusch_godfrey_test(&cols(&[&x1, &x2]), true, &resid, &time_order_identity(), 3)
                .unwrap();
        assert!((ignored.lm_statistic - ordered.lm_statistic).abs() > 1e-6);
    }

    #[test]
    fn breusch_godfrey_does_not_depend_on_the_scale_of_the_regressors() {
        let big1: Vec<f64> = X1.iter().map(|v| v * 1.0e6).collect();
        let base = breusch_godfrey_test(
            &cols(&[&X1, &X2]),
            true,
            &RESID_TIME_ORDER,
            &time_order_identity(),
            2,
        )
        .unwrap();
        let scaled = breusch_godfrey_test(
            &cols(&[&big1, &X2]),
            true,
            &RESID_TIME_ORDER,
            &time_order_identity(),
            2,
        )
        .unwrap();
        assert!((scaled.lm_statistic - base.lm_statistic).abs() < 1e-9);
        assert!((scaled.f_statistic - base.f_statistic).abs() < 1e-9);
    }

    #[test]
    fn breusch_godfrey_succeeds_with_one_denominator_degree_of_freedom() {
        // k = 3、nlags = 11で補助回帰は14列。n = 15なら`f_df_denom = 1`で成功し、
        // nlags = 12（15列）は観測数不足。
        let x = cols(&[&X1, &X2]);
        let time = time_order_identity();
        let ok = breusch_godfrey_test(&x, true, &RESID_TIME_ORDER, &time, 11).unwrap();
        assert_eq!((ok.df, ok.f_df_denom), (11, 1));

        let err = breusch_godfrey_test(&x, true, &RESID_TIME_ORDER, &time, 12).unwrap_err();
        assert_eq!(
            err,
            LeastSquaresError::InsufficientObservationsForAuxRegression {
                test: DiagnosticTest::BreuschGodfrey,
                n: 15,
                k: 15
            }
        );
    }

    #[test]
    fn breusch_godfrey_is_robust_to_regressors_with_a_large_mean_such_as_calendar_years() {
        // 平均が標準偏差より桁違いに大きい列（年等）でも、定数を含む列空間は同じなので
        // 結果は変わらない。
        let years: Vec<f64> = X1.iter().map(|v| 2000.0 + v).collect();
        let time = time_order_identity();
        let base =
            breusch_godfrey_test(&cols(&[&X1, &X2]), true, &RESID_TIME_ORDER, &time, 2).unwrap();
        let shifted =
            breusch_godfrey_test(&cols(&[&years, &X2]), true, &RESID_TIME_ORDER, &time, 2).unwrap();
        assert!((shifted.lm_statistic - base.lm_statistic).abs() < 1e-6);
        assert!((shifted.f_statistic - base.f_statistic).abs() < 1e-6);
    }

    #[test]
    fn breusch_godfrey_fails_when_the_auxiliary_regression_fits_the_residuals_exactly() {
        // 説明変数が残差そのものだと補助回帰が残差を完全に説明し、Fの分母が0になる。
        let err = breusch_godfrey_test(
            &cols(&[&RESID_TIME_ORDER]),
            false,
            &RESID_TIME_ORDER,
            &time_order_identity(),
            1,
        )
        .unwrap_err();
        assert_eq!(
            err,
            LeastSquaresError::DiagnosticStatisticUndefined {
                test: DiagnosticTest::BreuschGodfrey
            }
        );
    }

    #[test]
    fn scale_columns_by_norm_does_not_overflow_for_huge_or_underflow_for_tiny_entries() {
        let mut huge = Mat::from_fn(2, 1, |t, _| if t == 0 { 3.0e200 } else { 4.0e200 });
        scale_columns_by_norm(&mut huge).unwrap();
        assert!((huge.get(0, 0) - 0.6).abs() < 1e-12);
        assert!((huge.get(1, 0) - 0.8).abs() < 1e-12);

        let mut tiny = Mat::from_fn(2, 1, |t, _| if t == 0 { 3.0e-200 } else { 4.0e-200 });
        scale_columns_by_norm(&mut tiny).unwrap();
        assert!((tiny.get(0, 0) - 0.6).abs() < 1e-12);
    }

    #[test]
    fn breusch_godfrey_rejects_invalid_nlags_and_mismatched_lengths() {
        let x = cols(&[&X1, &X2]);
        let time = time_order_identity();
        for nlags in [0, -1, i64::MIN] {
            assert_eq!(
                breusch_godfrey_test(&x, true, &RESID_TIME_ORDER, &time, nlags).unwrap_err(),
                LeastSquaresError::InvalidNlags { nlags }
            );
        }
        // 巨大なnlagsは観測数不足として扱う（オーバーフローしない）。
        assert!(matches!(
            breusch_godfrey_test(&x, true, &RESID_TIME_ORDER, &time, i64::MAX).unwrap_err(),
            LeastSquaresError::InsufficientObservationsForAuxRegression { .. }
        ));

        let short_time = &time[..10];
        assert_eq!(
            breusch_godfrey_test(&x, true, &RESID_TIME_ORDER, short_time, 1).unwrap_err(),
            LeastSquaresError::Common(CommonError::DimensionMismatch {
                y_rows: 15,
                x_rows: 10
            })
        );
        let short_x = cols(&[&X1[..10], &X2[..10]]);
        assert!(matches!(
            breusch_godfrey_test(&short_x, true, &RESID_TIME_ORDER, &time, 1).unwrap_err(),
            LeastSquaresError::Common(CommonError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn breusch_godfrey_reports_a_singular_auxiliary_regression_with_its_context() {
        // 説明変数が残差の1次ラグそのものだと補助回帰の列が共線になる。
        let mut lag1 = vec![0.0];
        lag1.extend_from_slice(&RESID_TIME_ORDER[..14]);
        let err = breusch_godfrey_test(
            &cols(&[&lag1]),
            true,
            &RESID_TIME_ORDER,
            &time_order_identity(),
            1,
        )
        .unwrap_err();
        assert!(
            matches!(
                &err,
                LeastSquaresError::AuxiliaryRegressionFailed {
                    test: DiagnosticTest::BreuschGodfrey,
                    ..
                }
            ),
            "{err:?}"
        );
        let report = err.report();
        assert_eq!(report.code(), ErrorCode::DesignMatrixSingular);
        assert_eq!(
            report.stage(),
            Some(&Stage::DiagnosticAuxiliaryRegression {
                test: "BREUSCH_GODFREY".to_string()
            })
        );
    }

    #[test]
    fn breusch_godfrey_fails_for_all_zero_or_non_finite_residuals() {
        let x = cols(&[&X1, &X2]);
        let time = time_order_identity();
        for resid in [[0.0; 15], {
            let mut r = RESID_TIME_ORDER;
            r[4] = f64::NAN;
            r
        }] {
            assert_eq!(
                breusch_godfrey_test(&x, true, &resid, &time, 1).unwrap_err(),
                LeastSquaresError::DiagnosticStatisticUndefined {
                    test: DiagnosticTest::BreuschGodfrey
                }
            );
        }
    }

    #[test]
    fn scale_columns_by_norm_keeps_zero_columns_and_rejects_non_finite_values() {
        let mut scaled = Mat::from_fn(3, 2, |t, j| if j == 0 { (t + 1) as f64 * 2.0 } else { 0.0 });
        scale_columns_by_norm(&mut scaled).unwrap();
        let norm: f64 = (0..3).map(|t| scaled.get(t, 0).powi(2)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12);
        assert_eq!(*scaled.get(1, 1), 0.0);

        let mut bad = Mat::from_fn(2, 1, |t, _| if t == 0 { f64::INFINITY } else { 1.0 });
        assert!(scale_columns_by_norm(&mut bad).is_err());
    }

    #[test]
    fn allocate_aux_matrix_returns_computation_error_instead_of_aborting() {
        // 要素数のオーバーフロー。
        let err = allocate_aux_matrix(usize::MAX / 2, 4).unwrap_err();
        assert_eq!(
            err,
            LeastSquaresError::BreuschGodfreyMatrixTooLarge {
                n_obs: usize::MAX / 2,
                n_columns: 4
            }
        );
        // オーバーフローしないが、アドレス空間を超えて確保できないサイズ（2^55バイト）。
        let err = allocate_aux_matrix(1 << 40, 1 << 12).unwrap_err();
        assert!(matches!(
            err,
            LeastSquaresError::BreuschGodfreyMatrixTooLarge { .. }
        ));
    }

    #[test]
    fn allocate_aux_matrix_succeeds_for_a_small_matrix() {
        let mut m = allocate_aux_matrix(5, 3).unwrap();
        m.resize_with(5, 3, |t, j| (t * 3 + j) as f64);
        assert_eq!((m.nrows(), m.ncols()), (5, 3));
        assert_eq!(*m.get(4, 2), 14.0);
    }
}
