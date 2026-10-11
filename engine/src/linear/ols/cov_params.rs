//! OLSの`cov_type`別の係数分散共分散行列と、それに伴う推論の自由度。
//!
//! `shared::covariance`の部品（classical / HC0-3 / HAC / cluster）を、OLS固有の規則で
//! つなぐ層: HACのラグ数の解決（`LeastSquaresError::InvalidHacLags`）、クラスター列の検証、
//! 推論の自由度`df_inference`（`cov_type=Cluster`のときだけ`G-1`）。エラー型が
//! `LeastSquaresError`のため`shared/`には置けない。

use faer::Mat;

use super::cov_type::CovType;
use crate::linear::common::LeastSquaresError;
use crate::shared::covariance::{
    HcVariant, classical_cov_params, cluster_cov_params, hac_cov_params, hc_cov_params,
    time_ordering,
};
use crate::shared::error::CommonError;
use crate::shared::validation::{validate_cluster_count_covers_slopes, validate_cluster_groups};

/// [`compute_cov_params`]の結果。
pub(super) struct CovParams {
    /// 係数の分散共分散行列 (k, k)
    pub(super) cov_params: Mat<f64>,
    /// t検定・信頼区間・F検定に使う自由度。通常は`df_resid`（n-k）と同じだが、
    /// `cov_type=Cluster`のときだけ`G-1`（クラスター数-1）に切り替える
    /// （statsmodelsの`df_correction=True`という既定と同じ挙動。標準的な計量経済学の
    /// 慣行でもある。`df_resid`自体は分散推定量`σ̂²`・調整済みR²・AIC/BIC等では
    /// 引き続き`n-k`のまま使う。`docs/spec/ols-spec.md`「標準誤差」のクラスター参照）。
    pub(super) df_inference: usize,
    /// `cov_type=Hac`のとき、実際に使われたラグ数。`Hac`以外では`None`。
    pub(super) hac_lags_used: Option<usize>,
}

/// `cov_type=Cluster`のクラスター数`g`が傾き係数の数`q`（`n_slopes`）以下だと、
/// クラスター寄与スコアの総和がゼロ（正規方程式`X'e = 0`）で`rank(Ŝ) ≤ g - 1`の
/// ため、ロバストWald/F検定の`q×q`部分行列が構造的に特異になる。
/// `g`・`q`は入力だけから判定できるため、QR分解・残差計算より前に弾く
/// （nonlinear/IVと同じく`fit()`冒頭で検証する方針に揃える）。`groups=None`は
/// [`compute_cov_params`]の`CovType::Cluster`アームで`MissingClusterColumn`として扱う。
pub(super) fn validate_cluster_count(
    cov_type: &CovType,
    n: usize,
    n_slopes: usize,
) -> Result<(), LeastSquaresError> {
    if let CovType::Cluster {
        groups: Some(groups),
    } = cov_type
    {
        let g = validate_cluster_groups(groups, n)?;
        validate_cluster_count_covers_slopes(g, n_slopes)?;
    }
    Ok(())
}

/// `cov_type`に応じた係数分散共分散行列と推論の自由度を求める。
///
/// `sigma2`は`SSR/df_resid`（classicalでのみ使う）、`df_resid`は`n - k`。
pub(super) fn compute_cov_params(
    cov_type: &CovType,
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    sigma2: f64,
    df_resid: usize,
) -> Result<CovParams, LeastSquaresError> {
    let n = x.nrows();
    let k = x.ncols();
    let mut hac_lags_used = None;
    let (cov_params, df_inference) = match cov_type {
        CovType::Classical => (classical_cov_params(sigma2, xtx_inv, k), df_resid),
        CovType::Hc0 => (
            hc_cov_params(x, residuals, xtx_inv, HcVariant::Hc0),
            df_resid,
        ),
        CovType::Hc1 => (
            hc_cov_params(x, residuals, xtx_inv, HcVariant::Hc1),
            df_resid,
        ),
        CovType::Hc2 => (
            hc_cov_params(x, residuals, xtx_inv, HcVariant::Hc2),
            df_resid,
        ),
        CovType::Hc3 => (
            hc_cov_params(x, residuals, xtx_inv, HcVariant::Hc3),
            df_resid,
        ),
        CovType::Hac { lags, time_order } => {
            let lags = resolve_hac_lags(*lags, n)?;
            hac_lags_used = Some(lags);
            let order = time_ordering(time_order, n);
            (
                hac_cov_params(x, residuals, xtx_inv, lags, &order),
                df_resid,
            )
        }
        CovType::Cluster { groups } => {
            let groups = groups.as_ref().ok_or(CommonError::MissingClusterColumn)?;
            // クラスター数`g >= 2`・`g > q`（傾き係数の数）は`validate_cluster_count`で
            // 検証済み。ここでは`n_groups - 1`（検定の自由度）に再利用するため
            // 再度ユニーク数を数えるだけ。
            let n_groups = validate_cluster_groups(groups, n)?;
            let cov = cluster_cov_params(x, residuals, xtx_inv, groups);
            (cov, n_groups - 1)
        }
    };
    Ok(CovParams {
        cov_params,
        df_inference,
        hac_lags_used,
    })
}

/// `CovType::Hac`の`lags`（`Option<i64>`）を実際に使うラグ数（`usize`）に解決する。
///
/// `Some(l)`の場合は`0 <= l < n`を検証してそのまま使う。`None`の場合は経験則
/// `L = floor(4*(n/100)^(2/9))`で自動計算する（`docs/spec/ols-spec.md`
/// 「標準誤差」のHAC。EViews等でも使われる、データに依存しない決定的な式）。
fn resolve_hac_lags(lags: Option<i64>, n: usize) -> Result<usize, LeastSquaresError> {
    match lags {
        Some(l) => {
            if l < 0 || (l as usize) >= n {
                return Err(LeastSquaresError::InvalidHacLags { hac_lags: l, n });
            }
            Ok(l as usize)
        }
        None => Ok((4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize),
    }
}

#[cfg(test)]
mod tests {
    use super::super::estimator::OlsEstimator;
    use super::super::input::OlsInput;
    use super::*;
    use crate::linear::common::row_time_order;

    /// 同じデータセット（x=[1..5], y=[2,4,5,4,5]）でのHC0〜HC3。
    /// 期待値はstatsmodels 0.14.6で`use_t=True`を明示指定して独立に計算・検算済み
    /// （`sm.OLS(Y, X).fit(cov_type=..., use_t=True)`）。`use_t=True`が必要な理由は
    /// `docs/spec/ols-spec.md`「標準誤差」、
    /// および`OlsEstimator::fit`のdocコメント参照
    /// （statsmodelsはHC0-3でuse_t=Falseが既定＝正規分布のため、素の既定値とは一致しない）。
    #[test]
    fn fit_computes_hc_std_errors_test_stats_p_values_and_conf_int() {
        // (cov_type, [se_const, se_x1], [t_const, t_x1], [p_const, p_x1],
        //  [lower_const, lower_x1], [upper_const, upper_x1])
        #[allow(clippy::type_complexity)]
        let cases: [(CovType, [f64; 2], [f64; 2], [f64; 2], [f64; 2], [f64; 2]); 4] = [
            (
                CovType::Hc0,
                [0.741_350_119_714_024_1, 0.185_472_369_909_913_61],
                [2.967_558_703_367_644, 3.234_983_196_103_162_8],
                [0.059_183_855_836_541_795, 0.048_033_568_062_853_735],
                [-0.159_306_949_405_533_25, 0.009_744_141_647_982_651],
                [4.559_306_949_405_528, 1.190_255_858_352_018_2],
            ),
            (
                CovType::Hc1,
                [0.957_078_889_120_43, 0.239_443_799_947_572_34],
                [2.298_661_087_407_152_2, 2.505_807_208_753_678_2],
                [0.105_117_351_189_905_94, 0.087_259_022_565_828_92],
                [-0.845_852_174_546_351, -0.162_017_036_466_242_44],
                [5.245_852_174_546_345, 1.362_017_036_466_243_2],
            ),
            (
                CovType::Hc2,
                [1.106_216_202_066_430_8, 0.279_795_843_939_315_45],
                [1.988_761_325_218_668_2, 2.144_420_701_724_696_3],
                [0.140_853_196_409_229_89, 0.121_345_243_297_999_42],
                [-1.320_473_665_111_291_2, -0.290_435_249_778_410_95],
                [5.720_473_665_111_285, 1.490_435_249_778_412],
            ),
            (
                CovType::Hc3,
                [1.689_236_634_744_594_6, 0.429_760_255_899_750_37],
                [1.302_363_419_517_377_2, 1.396_127_240_160_527_1],
                [0.283_757_574_453_598_06, 0.257_051_084_412_133_5],
                [-3.175_904_886_992_822, -0.767_688_938_545_941],
                [7.575_904_886_992_816_5, 1.967_688_938_545_941_7],
            ),
        ];

        for (cov_type, se, t, p, lower, upper) in cases {
            let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
            let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
            let input = OlsInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();

            let cov_type_label = format!("{cov_type:?}");
            let estimator = OlsEstimator::fit(input, cov_type, 0.95).unwrap();

            for j in 0..2 {
                let msg = format!("{cov_type_label}, param {j}");
                assert!(
                    (*estimator.std_errors().get(j, 0) - se[j]).abs() < 1e-6,
                    "std_errors mismatch: {msg}"
                );
                assert!(
                    (*estimator.test_stats().get(j, 0) - t[j]).abs() < 1e-6,
                    "test_stats mismatch: {msg}"
                );
                assert!(
                    (*estimator.p_values().get(j, 0) - p[j]).abs() < 1e-6,
                    "p_values mismatch: {msg}"
                );
                assert!(
                    (*estimator.conf_lower().get(j, 0) - lower[j]).abs() < 1e-6,
                    "conf_lower mismatch: {msg}"
                );
                assert!(
                    (*estimator.conf_upper().get(j, 0) - upper[j]).abs() < 1e-6,
                    "conf_upper mismatch: {msg}"
                );
            }
        }
    }

    /// 同じデータセット（x=[1..5], y=[2,4,5,4,5]）でのHAC（Newey-West、`maxlags=1`）。
    /// 期待値はstatsmodels 0.14.6で独立に計算・検算済み
    /// （`sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1}, use_t=True)`。
    /// `use_correction`はstatsmodelsの既定である`False`のまま、明示指定はしていない）。
    #[test]
    fn fit_computes_hac_std_errors_with_explicit_lags() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let cov_type = CovType::Hac {
            lags: Some(1),
            time_order: row_time_order(5),
        };
        let estimator = OlsEstimator::fit(input, cov_type, 0.95).unwrap();

        let se = estimator.std_errors();
        assert!((*se.get(0, 0) - 0.659_090_282_131_361_7).abs() < 1e-6);
        assert!((*se.get(1, 0) - 0.164_924_225_024_705_7).abs() < 1e-6);

        let t = estimator.test_stats();
        assert!((*t.get(0, 0) - 3.337_934_209_689_228).abs() < 1e-6);
        assert!((*t.get(1, 0) - 3.638_034_375_545_013_5).abs() < 1e-6);

        let p = estimator.p_values();
        assert!((*p.get(0, 0) - 0.044_455_744_969_471_62).abs() < 1e-6);
        assert!((*p.get(1, 0) - 0.035_791_053_269_350_51).abs() < 1e-6);

        let lower = estimator.conf_lower();
        let upper = estimator.conf_upper();
        assert!((*lower.get(0, 0) - 0.102_480_566_782_648_72).abs() < 1e-6);
        assert!((*upper.get(0, 0) - 4.297_519_433_217_346).abs() < 1e-6);
        assert!((*lower.get(1, 0) - 0.075_137_509_418_346_96).abs() < 1e-6);
        assert!((*upper.get(1, 0) - 1.124_862_490_581_653_8).abs() < 1e-6);
    }

    /// `hac_lags=None`（経験則自動計算）が`L = floor(4*(n/100)^(2/9))`と一致することを確認する。
    /// n=5の場合L=2。期待値はstatsmodelsで`maxlags=2`を明示指定して独立に計算・検算済み
    /// （`docs/spec/ols-spec.md`「標準誤差」のHACの式通りベンチマーク側もL=2を使う前提）。
    #[test]
    fn fit_computes_hac_std_errors_with_auto_lags() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let cov_type = CovType::Hac {
            lags: None,
            time_order: row_time_order(5),
        };
        let estimator = OlsEstimator::fit(input, cov_type, 0.95).unwrap();

        let se = estimator.std_errors();
        assert!((*se.get(0, 0) - 0.577_350_269_189_624_1).abs() < 1e-6);
        assert!((*se.get(1, 0) - 0.164_924_225_024_705_75).abs() < 1e-6);
    }

    /// `n`行の単純回帰（`y = x + (i % 7)`）を`cov_type`で推定し`hac_lags_used`を返す補助関数。
    fn hac_lags_used_for(n: usize, cov_type: CovType) -> Option<usize> {
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| i as f64 + (i % 7) as f64).collect();
        let input = OlsInput::from_columns(&y, &[x], vec!["x1".to_string()], true, "y".to_string())
            .unwrap();
        OlsEstimator::fit(input, cov_type, 0.95)
            .unwrap()
            .hac_lags_used()
    }

    /// `hac_lags`未指定（`None`）なら経験則`floor(4*(n/100)^(2/9))`で解決した値が
    /// `hac_lags_used()`に入る。`n=5`→2、`n=100`→4（`(n/100)^(2/9)=1`ちょうど）、
    /// `n=1000`→6（`4*10^(2/9)≈6.67`）。
    #[test]
    fn hac_lags_used_returns_auto_selected_lags_when_lags_is_none() {
        for (n, expected) in [(5, 2), (100, 4), (1000, 6)] {
            let cov_type = CovType::Hac {
                lags: None,
                time_order: row_time_order(n),
            };
            assert_eq!(hac_lags_used_for(n, cov_type), Some(expected), "n={n}");
        }
    }

    /// `hac_lags`を明示指定した場合は、自動計算値（`n=100`なら4）ではなく指定値が入る。
    #[test]
    fn hac_lags_used_returns_explicit_lags_when_specified() {
        for lags in [0, 3, 10] {
            let cov_type = CovType::Hac {
                lags: Some(lags),
                time_order: row_time_order(100),
            };
            assert_eq!(
                hac_lags_used_for(100, cov_type),
                Some(lags as usize),
                "lags={lags}"
            );
        }
    }

    /// `cov_type`がHac以外なら`None`。
    #[test]
    fn hac_lags_used_is_none_for_non_hac_cov_types() {
        for cov_type in [CovType::Classical, CovType::Hc0, CovType::Hc3] {
            let label = format!("{cov_type:?}");
            assert_eq!(hac_lags_used_for(20, cov_type), None, "{label}");
        }
    }

    /// `time_order`を指定した場合、行順がシャッフルされていても時系列順に並べ替えてから
    /// ラグ付き自己共分散を計算することを確認する。データはHAC(maxlags=1)テストと同一の
    /// (x, y)を、時系列順の逆転を含む順序（time値=xの値そのもの）でシャッフルして与える。
    /// 期待値は`fit_computes_hac_std_errors_with_explicit_lags`と同じ
    /// （時系列順に並べ替えれば同一データになるため）。
    #[test]
    fn fit_computes_hac_std_errors_respecting_time_order() {
        // 元の時系列順: time=[1,2,3,4,5], y=[2,4,5,4,5]
        // これを time順=[3,1,5,2,4] の並びでシャッフルして入力する
        let shuffled_time = vec![3.0, 1.0, 5.0, 2.0, 4.0];
        let shuffled_x = shuffled_time.clone();
        let shuffled_y = vec![5.0, 2.0, 5.0, 4.0, 4.0];

        let input = OlsInput::from_columns(
            &shuffled_y,
            &[shuffled_x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let cov_type = CovType::Hac {
            lags: Some(1),
            time_order: shuffled_time,
        };
        let estimator = OlsEstimator::fit(input, cov_type, 0.95).unwrap();

        let se = estimator.std_errors();
        assert!((*se.get(0, 0) - 0.659_090_282_131_361_7).abs() < 1e-6);
        assert!((*se.get(1, 0) - 0.164_924_225_024_705_7).abs() < 1e-6);
    }

    #[test]
    fn fit_returns_invalid_hac_lags_when_out_of_range() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let cov_type = CovType::Hac {
            lags: Some(-1),
            time_order: row_time_order(5),
        };
        let result = OlsEstimator::fit(input, cov_type, 0.95);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::InvalidHacLags { hac_lags: -1, n: 5 }
        );
    }

    /// `hac_lags`の上限側の境界。`n=5`のとき`lags=5`（`n`自体）は範囲外（`[0, n)`）、
    /// `lags=4`（`n-1`）は許容される最大値であることを確認する。
    #[test]
    fn fit_returns_invalid_hac_lags_when_equal_to_n() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let cov_type = CovType::Hac {
            lags: Some(5),
            time_order: row_time_order(5),
        };
        let result = OlsEstimator::fit(input, cov_type, 0.95);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::InvalidHacLags { hac_lags: 5, n: 5 }
        );
    }

    #[test]
    fn fit_accepts_hac_lags_at_upper_boundary_of_n_minus_one() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let cov_type = CovType::Hac {
            lags: Some(4), // n - 1、許容される最大値
            time_order: row_time_order(5),
        };
        let result = OlsEstimator::fit(input, cov_type, 0.95);

        assert!(result.is_ok());
    }

    /// `CovType::Hac { lags: Some(0), .. }`は`Ŝ = Ŝ₀`（ラグ項なし）に退化し、これは
    /// `HC0`の`Ψ̂ = Σ_i ε̂_i² x_i x_i'`と数学的に同一の式になる（`hac_cov_params`の
    /// l=0項のドキュメント参照）。2つの独立した実装（`hc_cov_params`と`hac_cov_params`）が
    /// この境界で一致することを確認する内部整合性テスト。
    #[test]
    fn fit_hac_with_zero_lags_matches_hc0() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 8.0, 3.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];

        let input_hac = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let hac = OlsEstimator::fit(
            input_hac,
            CovType::Hac {
                lags: Some(0),
                time_order: row_time_order(6),
            },
            0.95,
        )
        .unwrap();

        let input_hc0 = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let hc0 = OlsEstimator::fit(input_hc0, CovType::Hc0, 0.95).unwrap();

        for j in 0..2 {
            assert!(
                (*hac.std_errors().get(j, 0) - *hc0.std_errors().get(j, 0)).abs() < 1e-12,
                "param {j}: hac(lags=0)={}, hc0={}",
                *hac.std_errors().get(j, 0),
                *hc0.std_errors().get(j, 0)
            );
        }
    }

    /// 同じデータセット（x=[1..5], y=[2,4,5,4,5]）を2クラスター（groups=["a","a","b","b","b"]）
    /// でのクラスターロバスト標準誤差。期待値はstatsmodels 0.14.6で独立に計算・検算済み
    /// （`sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)`。
    /// 小標本補正はstatsmodelsの既定`use_correction=True`のまま、明示指定はしていない）。
    #[test]
    fn fit_computes_cluster_std_errors_test_stats_p_values_conf_int_and_f_test() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];
        let cov_type = CovType::Cluster {
            groups: Some(groups),
        };
        let estimator = OlsEstimator::fit(input, cov_type, 0.95).unwrap();

        let se = estimator.std_errors();
        assert!((*se.get(0, 0) - 0.785_196_366_097_886_8).abs() < 1e-6);
        assert!((*se.get(1, 0) - 0.230_940_107_675_849_05).abs() < 1e-6);

        let t = estimator.test_stats();
        assert!((*t.get(0, 0) - 2.801_846_894_596_724_5).abs() < 1e-6);
        assert!((*t.get(1, 0) - 2.598_076_211_353_332).abs() < 1e-6);

        let p = estimator.p_values();
        assert!((*p.get(0, 0) - 0.218_242_895_017_685_43).abs() < 1e-6);
        assert!((*p.get(1, 0) - 0.233_908_049_281_92).abs() < 1e-6);

        let lower = estimator.conf_lower();
        let upper = estimator.conf_upper();
        assert!((*lower.get(0, 0) - (-7.776_865_785_740_13)).abs() < 1e-6);
        assert!((*upper.get(0, 0) - 12.176_865_785_740_125).abs() < 1e-6);
        assert!((*lower.get(1, 0) - (-2.334_372_289_923_566_6)).abs() < 1e-6);
        assert!((*upper.get(1, 0) - 3.534_372_289_923_567_7).abs() < 1e-6);

        assert!((estimator.f_statistic() - 6.750_000_000_000_083_5).abs() < 1e-6);
        assert!((estimator.f_p_value() - 0.233_908_049_281_92).abs() < 1e-6);
    }

    /// クラスターロバスト標準誤差は、同じ入力に対して`fit()`を複数回呼んでも
    /// ビット単位で同じ結果になること（`cluster_cov_params`内部の集約が
    /// `HashMap`の反復順序に依存し、実行のたびに浮動小数点の丸め誤差レベルで
    /// 結果がぶれていた回帰。`BTreeMap`化で修正した）。
    #[test]
    fn fit_cluster_std_errors_are_deterministic_across_repeated_fits() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0, 3.0, 6.0, 1.0, 7.0, 2.5];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let groups: Vec<String> = (0..10).map(|i| format!("g{}", i % 4)).collect();

        let build = || {
            let input = OlsInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();
            let cov_type = CovType::Cluster {
                groups: Some(groups.clone()),
            };
            OlsEstimator::fit(input, cov_type, 0.95).unwrap()
        };

        let first = build();
        for _ in 0..20 {
            let repeat = build();
            assert_eq!(
                *repeat.std_errors().get(0, 0),
                *first.std_errors().get(0, 0)
            );
            assert_eq!(
                *repeat.std_errors().get(1, 0),
                *first.std_errors().get(1, 0)
            );
        }
    }

    #[test]
    fn fit_returns_missing_cluster_column_when_groups_not_provided() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = OlsEstimator::fit(input, CovType::Cluster { groups: None }, 0.95);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::MissingClusterColumn)
        );
    }

    #[test]
    fn fit_returns_insufficient_clusters_when_only_one_group() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let groups = vec!["a".to_string(); 5];
        let cov_type = CovType::Cluster {
            groups: Some(groups),
        };
        let result = OlsEstimator::fit(input, cov_type, 0.95);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::InsufficientClusters { g: 1 })
        );
    }

    /// `cov_type=Cluster`でクラスター数`g`が傾き係数の数`q`以下だと、クラスター寄与
    /// スコアの総和がゼロ（正規方程式`X'e = 0`）で`rank(Ŝ) ≤ g - 1`のため、ロバスト
    /// Wald/F検定の`q×q`部分行列が構造的に特異になる。`g`・`q`は入力だけから判定
    /// できるため`fit()`のバリデーションが`CommonError::InsufficientClustersForInference`
    /// で弾く。切片＋説明変数3個（`q = 4 - 1 = 3`）に対し`g = 2`
    /// （`g < q`）と`g = 3`（`g == q`、`rank(Ŝ) ≤ 2 < 3`で依然特異）の両方を確認する。
    /// backstop（`g > q`だが悪条件で数値的にほぼ特異）は
    /// `fit_returns_computation_failed_for_cluster_when_slope_submatrix_is_ill_conditioned`。
    #[test]
    fn fit_returns_validation_error_when_cluster_count_at_most_slopes() {
        let n = 12;
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * i as f64).collect();
        let x_columns = vec![
            (0..n).map(|i| i as f64).collect::<Vec<f64>>(),
            (0..n).map(|i| (i * i) as f64).collect::<Vec<f64>>(),
            (0..n).map(|i| ((i % 4) + 1) as f64).collect::<Vec<f64>>(),
        ];
        for g in [2_usize, 3_usize] {
            let groups: Vec<String> = (0..n).map(|i| format!("g{}", i % g)).collect();
            let input = OlsInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string(), "x2".to_string(), "x3".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();
            let result = OlsEstimator::fit(
                input,
                CovType::Cluster {
                    groups: Some(groups),
                },
                0.95,
            );
            assert_eq!(
                result.unwrap_err(),
                LeastSquaresError::Common(CommonError::InsufficientClustersForInference {
                    g,
                    q: 3
                }),
                "g={g}"
            );
        }
    }

    /// `cov_type=Cluster`かつ`g > q`（`fit()`冒頭のバリデーションを通過）でも、傾き係数
    /// 間のスケールが極端に異なると、ロバストWald/F検定の`q×q`部分行列の条件数が倍精度の
    /// 限界を超え`ensure_well_conditioned_symmetric_matrix`が`ComputationFailed`で止める
    /// （`fit_returns_computation_failed_for_extreme_scale_difference_in_f_test`のcluster版。
    /// `cov_type`によらず傾き部分行列の条件数は同じだが、cluster経路でもこのbackstopが
    /// 生きていることを明示的に固定する）。`x1`は1e6・`x2`は1e-3オーダー、`g=4 > q=3`。
    #[test]
    fn fit_returns_computation_failed_for_cluster_when_slope_submatrix_is_ill_conditioned() {
        let n = 12;
        let x1: Vec<f64> = (1..=n).map(|i| 1e6 * (i as f64)).collect();
        let x2: Vec<f64> = (1..=n).map(|i| 1e-3 * (i as f64).powi(2)).collect();
        let x3: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let noise = if i % 2 == 0 { 0.1 } else { -0.1 };
                1.0 + 2.0 * x1[i] + 3.0 * x2[i] + 0.5 * x3[i] + noise
            })
            .collect();
        let groups: Vec<String> = (0..n).map(|i| format!("g{}", i % 4)).collect();
        let input = OlsInput::from_columns(
            &y,
            &[x1, x2, x3],
            vec!["x1".to_string(), "x2".to_string(), "x3".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = OlsEstimator::fit(
            input,
            CovType::Cluster {
                groups: Some(groups),
            },
            0.95,
        );

        assert!(matches!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::ComputationFailed(_))
        ));
    }

    #[test]
    #[should_panic]
    fn fit_panics_when_cluster_groups_length_does_not_match_nobs() {
        // groups.len() != nはengine_pybind側の実装バグでしか起こり得ない内部契約違反のため、
        // Errではなくdebug_assert!でパニックする（from_columns_panics_on_mismatched_names_arity
        // と同じ性質）。
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let groups = vec!["a".to_string(), "b".to_string()]; // n=5のはずが長さ2
        let cov_type = CovType::Cluster {
            groups: Some(groups),
        };
        let _ = OlsEstimator::fit(input, cov_type, 0.95);
    }
}
