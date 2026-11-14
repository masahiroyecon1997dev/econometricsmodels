//! HC・HAC共分散のテスト。

use crate::linear::common::row_time_order;
use crate::linear::ols::OlsEstimator;
use crate::linear::ols::cov_type::CovType;
use crate::linear::ols::input::OlsInput;

/// `n`行の単純回帰（`y = x + (i * i % 7)`）を`cov_type`で推定し`hac_lags_used`を返す補助関数。
fn hac_lags_used_for(n: usize, cov_type: CovType) -> Option<usize> {
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| i as f64 + (i * i % 7) as f64).collect();
    let input =
        OlsInput::from_columns(&y, &[x], vec!["x1".to_string()], true, "y".to_string()).unwrap();
    OlsEstimator::fit(input, cov_type, 0.95)
        .unwrap()
        .hac_lags_used()
}

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
