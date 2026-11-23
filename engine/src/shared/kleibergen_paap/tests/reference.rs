//! 参照実装(linearmodels)・Cragg-Donald・閉形式との照合。

use super::*;
use crate::shared::weak_instruments::cragg_donald;

// 期待値はlinearmodels 7.1の`kleibergen_paap`・`kleibergen_paap_f`(`unadjusted`・`robust`・
// `clustered`・`kernel`、バンド幅2)。HC1〜HC3は同じ関数の`_rk_chi2`に行スケールを掛けた
// スコアを渡して得た値(クロネッカー積の`kron(irqxx.T, irqzz.T)`をそのまま使う形で、ここの
// 射影の実装とは独立)。HC1のFはlinearmodelsの`robust`のFに、HC0のLMは`robust`のLMに一致する。
const TWO_ENDOGENOUS: [(&str, f64, f64); 7] = [
    ("classical", 8.497665557296273, 5.661334354194701),
    ("hc0", 5.20374630742091, 11.40464437283528),
    ("hc1", 3.035518679328865, 6.652709217487246),
    ("hc2", 2.349943110670311, 5.40201528315421),
    ("hc3", 1.0460104688089573, 2.5532340821578416),
    ("cluster", 2.9797800411423454, 19.612521171644726),
    ("hac", 2.7216722971551834, 9.501421955858229),
];
const ONE_ENDOGENOUS: [(&str, f64, f64); 7] = [
    ("classical", 11.440933552209195, 47.75015872855707),
    ("hc0", 6.930118602641243, 94.68656117293756),
    ("hc1", 4.0425691848740595, 55.2338273508802),
    ("hc2", 3.637101911302648, 51.32155941079068),
    ("hc3", 1.8657852030142683, 27.924991337553962),
    ("cluster", 3.8353062226858023, 80.18732120574977),
    ("hac", 3.355712030910789, 86.3981271989337),
];

// 除外操作変数`z1`のみ(内生変数`d1`)と、`z1,z2`(内生変数`d1,d2`)。出典は上と同じ。
const JUST_IDENTIFIED_ONE: [(&str, f64, f64); 7] = [
    ("classical", 10.726475441603906, 75.80401833476832),
    ("hc0", 6.59372327418782, 72.56532917716098),
    ("hc1", 4.945292455640866, 54.42399688287073),
    ("hc2", 3.9949434914574455, 48.9669621855392),
    ("hc3", 2.289483090286586, 32.628042624213734),
    ("cluster", 3.496579231756079, 42.962098341592345),
    ("hac", 3.3399844473803424, 43.03601872969669),
];
const JUST_IDENTIFIED_TWO: [(&str, f64, f64); 7] = [
    ("classical", 3.692809533060165, 1.7781268156816465),
    ("hc0", 2.095247416094977, 4.105073957572708),
    ("hc1", 1.396831610729985, 2.736715971715137),
    ("hc2", 1.1607543563295013, 2.4018579140895526),
    ("hc3", 0.6356116070899693, 1.379324374083247),
    ("cluster", 1.4097812859866132, 2.460165755306101),
    ("hac", 2.008917822918212, 5.013467006266563),
];

fn check_against_reference(y: &Mat<f64>, z: &Mat<f64>, expected: &[(&str, f64, f64)], df: usize) {
    let (_, exog, _) = data();
    let groups = four_clusters();
    let order: Vec<usize> = (0..12).collect();
    for &(name, lm_value, f_value) in expected {
        let cov = covariance(name, &groups, &order);
        let test = lm(y, &exog, z, 0, cov);
        assert_eq!(test.df, df, "{name}");
        assert_close(test.statistic, lm_value, 1e-9);
        assert_close(f(y, &exog, z, 0, cov), f_value, 1e-9);
    }
}

#[test]
fn it_matches_the_reference_with_two_endogenous_variables() {
    let (y, _, z) = data();
    check_against_reference(&y, &z, &TWO_ENDOGENOUS, 2);
}

#[test]
fn it_matches_the_reference_with_one_endogenous_variable() {
    let (y, _, z) = data();
    check_against_reference(&first_column(&y), &z, &ONE_ENDOGENOUS, 3);
}

/// 除外操作変数と内生変数が同数(`q = k`)のとき、`λ`は1次元になる。
#[test]
fn it_matches_the_reference_when_instruments_equal_endogenous_variables() {
    let (y, _, z) = data();
    let columns = |m: &Mat<f64>, n_cols: usize| Mat::from_fn(12, n_cols, |i, j| *m.get(i, j));
    check_against_reference(&first_column(&y), &columns(&z, 1), &JUST_IDENTIFIED_ONE, 1);
    check_against_reference(&y, &columns(&z, 2), &JUST_IDENTIFIED_TWO, 1);
}

#[test]
fn it_matches_the_reference_on_a_larger_sample() {
    let n = 240;
    let (y, exog, z) = large_sample();
    let labels: Vec<String> = (0..n).map(|i| format!("g{:02}", i / 12)).collect();
    let groups = GroupCodes::from_labels_without_keys(&labels);
    let order: Vec<usize> = (0..n).collect();

    let expected = [
        (
            ScoreCovariance::Classical,
            17.68545662910589,
            Some(4.653763074225158),
        ),
        (
            ScoreCovariance::Hc(HcVariant::Hc0),
            15.381758604943311,
            None,
        ),
        (
            ScoreCovariance::Cluster(&groups),
            11.255378524365176,
            Some(7.626942746278178),
        ),
        (
            ScoreCovariance::Hac {
                lags: 4,
                order: &order,
            },
            11.419456702919037,
            Some(4.056764032347864),
        ),
    ];
    for (cov, lm_value, f_value) in expected {
        let test = lm(&y, &exog, &z, 0, cov);
        assert_eq!(test.df, 3);
        assert_close(test.statistic, lm_value, 1e-9);
        if let Some(f_value) = f_value {
            assert_close(f(&y, &exog, &z, 0, cov), f_value, 1e-9);
        }
    }
    // linearmodelsの`robust`のFはHC1に対応する。
    assert_close(
        f(&y, &exog, &z, 0, ScoreCovariance::Hc(HcVariant::Hc1)),
        4.396909457743589,
        1e-9,
    );
}

#[test]
fn classical_f_equals_cragg_donald_for_every_absorbed_count() {
    let (y, exog, z) = data();
    for y in [y.clone(), first_column(&y)] {
        for n_absorbed in [0, 1, 3] {
            let cd = cragg_donald(&y, &exog, &z, n_absorbed).unwrap();
            let kp = f(&y, &exog, &z, n_absorbed, ScoreCovariance::Classical);
            assert_close(kp, cd, 1e-10);
        }
    }
}

/// 等分散のLMはAndersonの`n·r²_min`で、`r²_min = λ/(1+λ)`(`λ`はCragg-Donaldの最小固有値)。
#[test]
fn classical_lm_is_n_times_the_smallest_squared_canonical_correlation() {
    let (y, exog, z) = data();
    let df = 12.0 - 2.0 - 3.0;
    let cd = cragg_donald(&y, &exog, &z, 0).unwrap();
    let lambda = cd * 3.0 / df;
    let expected = 12.0 * lambda / (1.0 + lambda);

    let test = lm(&y, &exog, &z, 0, ScoreCovariance::Classical);

    assert_close(test.statistic, expected, 1e-10);
}

#[test]
fn it_works_without_exogenous_regressors() {
    let (y, _, z) = data();
    let none = Mat::<f64>::zeros(12, 0);
    let cd = cragg_donald(&y, &none, &z, 0).unwrap();
    assert_close(f(&y, &none, &z, 0, ScoreCovariance::Classical), cd, 1e-10);
    assert!(kleibergen_paap_f(&y, &none, &z, 0, ScoreCovariance::Hc(HcVariant::Hc3)).is_some());
}
