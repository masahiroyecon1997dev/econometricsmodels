//! 統計量が定義できない入力(`None`になる条件)。

use super::*;

#[test]
fn it_is_undefined_when_the_statistic_cannot_be_formed() {
    let (y, exog, z) = data();
    let classical = ScoreCovariance::Classical;
    let no_endog = Mat::<f64>::zeros(12, 0);
    assert!(kleibergen_paap_lm(&no_endog, &exog, &z, 0, classical).is_none());
    assert!(kleibergen_paap_f(&no_endog, &exog, &z, 0, classical).is_none());
    // 除外操作変数が内生変数より少ない。
    let one_instrument = Mat::from_fn(12, 1, |i, _| *z.get(i, 0));
    assert!(kleibergen_paap_lm(&y, &exog, &one_instrument, 0, classical).is_none());
    // 自由度が残らない(12 − 2 − 3 − 7 = 0)。LMは自由度を要しない。
    assert!(kleibergen_paap_f(&y, &exog, &z, 7, classical).is_none());
    assert!(kleibergen_paap_lm(&y, &exog, &z, 7, classical).is_some());
    assert!(kleibergen_paap_lm(&y, &exog, &z, 7, ScoreCovariance::Hc(HcVariant::Hc1)).is_none());
    // 操作変数が外生変数の線形結合。
    let in_span = Mat::from_fn(12, 3, |i, j| {
        if j == 0 {
            *exog.get(i, 1) * 2.0 + 1.0
        } else {
            *z.get(i, j)
        }
    });
    assert!(kleibergen_paap_lm(&y, &exog, &in_span, 0, classical).is_none());
    // 内生変数が共線。
    let collinear = Mat::from_fn(12, 2, |i, j| *y.get(i, 0) * [1.0, -3.0][j]);
    assert!(kleibergen_paap_lm(&collinear, &exog, &z, 0, classical).is_none());
    // 非有限値。
    let mut nan_y = y.clone();
    *nan_y.get_mut(3, 0) = f64::NAN;
    assert!(kleibergen_paap_lm(&nan_y, &exog, &z, 0, classical).is_none());
}

#[test]
fn a_perfect_first_stage_is_undefined() {
    let (_, exog, z) = data();
    let y = Mat::from_fn(12, 1, |i, _| {
        2.0 * *z.get(i, 0) - *z.get(i, 1) + *exog.get(i, 1)
    });
    for cov in [
        ScoreCovariance::Classical,
        ScoreCovariance::Hc(HcVariant::Hc1),
    ] {
        assert!(kleibergen_paap_lm(&y, &exog, &z, 0, cov).is_none());
        assert!(kleibergen_paap_f(&y, &exog, &z, 0, cov).is_none());
    }
}

/// 1観測だけのダミーはレバレッジが1で、HC2・HC3は定義できない(HC0・HC1・クラスターは定義できる)。
#[test]
fn hc2_and_hc3_are_undefined_when_a_leverage_is_one() {
    let (y, exog, z) = data();
    let exog_with_dummy = Mat::from_fn(12, 3, |i, j| {
        if j < 2 {
            *exog.get(i, j)
        } else if i == 4 {
            1.0
        } else {
            0.0
        }
    });
    for variant in [HcVariant::Hc2, HcVariant::Hc3] {
        let cov = ScoreCovariance::Hc(variant);
        assert!(kleibergen_paap_lm(&y, &exog_with_dummy, &z, 0, cov).is_none());
    }
    assert!(
        kleibergen_paap_lm(
            &y,
            &exog_with_dummy,
            &z,
            0,
            ScoreCovariance::Hc(HcVariant::Hc1)
        )
        .is_some()
    );
}

/// クラスターが少なくてスコアの共分散がランク落ちするときは、一般化逆行列を使わず未定義とする。
/// LMの共分散は`(q − k + 1)`次元でクラスター数が階数の上限、Wald Fのスコアは合計が0なので
/// もう1つクラスターを要する。
#[test]
fn it_is_undefined_when_the_score_covariance_is_singular() {
    let (y, exog, z) = data();
    let two_labels: Vec<String> = (0..12).map(|i| ["a", "b"][i / 6].to_string()).collect();
    let two = GroupCodes::from_labels_without_keys(&two_labels);
    let cluster = ScoreCovariance::Cluster(&two);
    // 内生変数2個(自由度2): 2クラスターならLMは定義できるが、Wald Fはできない。
    assert!(kleibergen_paap_lm(&y, &exog, &z, 0, cluster).is_some());
    assert!(kleibergen_paap_f(&y, &exog, &z, 0, cluster).is_none());
    // 内生変数1個(自由度3): 2クラスターではLMも定義できない。
    let one_endogenous = first_column(&y);
    assert!(kleibergen_paap_lm(&one_endogenous, &exog, &z, 0, cluster).is_none());
    // 1クラスター。
    let one = GroupCodes::from_labels_without_keys(&vec!["a".to_string(); 12]);
    assert!(kleibergen_paap_lm(&y, &exog, &z, 0, ScoreCovariance::Cluster(&one)).is_none());
    // 行数の合わないクラスター・並び。
    let short = GroupCodes::from_labels_without_keys(&vec!["a".to_string(); 5]);
    assert!(kleibergen_paap_lm(&y, &exog, &z, 0, ScoreCovariance::Cluster(&short)).is_none());
    let order = [0usize, 1, 2];
    assert!(
        kleibergen_paap_lm(
            &y,
            &exog,
            &z,
            0,
            ScoreCovariance::Hac {
                lags: 1,
                order: &order
            }
        )
        .is_none()
    );
}

/// 大きいデータでも、クラスター数がちょうど`d`(= q − k + 1 = 3)ならWald Fのスコアの共分散は
/// ランク落ちし、丸め誤差で巨大な値を返さず`None`になる。LMは定義できる。
#[test]
fn a_covariance_with_exactly_as_many_clusters_as_dimensions_is_undefined_for_large_n() {
    let (y, exog, z) = large_sample();
    let labels: Vec<String> = (0..240).map(|i| format!("g{}", i / 80)).collect();
    let three = GroupCodes::from_labels_without_keys(&labels);
    let cluster = ScoreCovariance::Cluster(&three);
    assert!(kleibergen_paap_lm(&y, &exog, &z, 0, cluster).is_some());
    assert!(kleibergen_paap_f(&y, &exog, &z, 0, cluster).is_none());
}

/// `lags >= n`は`resolve_hac_lags`が拒否する設定で、ここでも未定義とする。
#[test]
fn hac_with_lags_not_below_the_number_of_rows_is_undefined() {
    let (y, exog, z) = data();
    let order: Vec<usize> = (0..12).collect();
    let hac = |lags| ScoreCovariance::Hac {
        lags,
        order: &order,
    };
    assert!(kleibergen_paap_lm(&y, &exog, &z, 0, hac(11)).is_some());
    assert!(kleibergen_paap_lm(&y, &exog, &z, 0, hac(12)).is_none());
}
