//! 吸収したパラメータ数`n_absorbed`の扱い。

use super::*;

/// 自由度補正は`n_absorbed`でF(とHC1の行スケール)にだけ入り、LMは変わらない。
/// 期待値はlinearmodelsの`_rk_chi2`にHC1の行スケール`√(12/(12−3−2−3))`を掛けたスコアを渡して得た値。
#[test]
fn n_absorbed_enters_only_the_small_sample_factors() {
    let (y, exog, z) = data();
    let groups = four_clusters();
    for cov in [
        ScoreCovariance::Classical,
        ScoreCovariance::Hc(HcVariant::Hc0),
        ScoreCovariance::Cluster(&groups),
    ] {
        let base = lm(&y, &exog, &z, 0, cov).statistic;
        assert_eq!(lm(&y, &exog, &z, 3, cov).statistic, base);
    }

    let hc1 = ScoreCovariance::Hc(HcVariant::Hc1);
    assert_close(
        lm(&y, &exog, &z, 3, hc1).statistic,
        1.7345821024736376,
        1e-9,
    );
    assert_close(f(&y, &exog, &z, 3, hc1), 3.801548124278425, 1e-9);
    // Hc0のFは自由度補正を持たないので、`n_absorbed`で変わらない。
    let hc0 = ScoreCovariance::Hc(HcVariant::Hc0);
    assert_eq!(f(&y, &exog, &z, 3, hc0), f(&y, &exog, &z, 0, hc0));
    // 等分散・HAC・クラスターのFは、自由度`n−l−n_absorbed`の分だけ小さくなる。
    assert!(
        f(&y, &exog, &z, 3, ScoreCovariance::Classical)
            < f(&y, &exog, &z, 0, ScoreCovariance::Classical)
    );
}

/// HC2・HC3のレバレッジは吸収した効果を含まないので、`n_absorbed > 0`では未定義(個体ダミーを
/// 回帰子に入れた定義と値が食い違うため。HC0・HC1などは一致する。下のFWLのテスト)。
#[test]
fn hc2_and_hc3_are_undefined_with_absorbed_parameters() {
    let (y, exog, z) = data();
    for variant in [HcVariant::Hc2, HcVariant::Hc3] {
        let cov = ScoreCovariance::Hc(variant);
        assert!(kleibergen_paap_lm(&y, &exog, &z, 1, cov).is_none());
        assert!(kleibergen_paap_f(&y, &exog, &z, 1, cov).is_none());
        assert!(kleibergen_paap_lm(&y, &exog, &z, 0, cov).is_some());
    }
}

fn demean_by_entity(m: &Mat<f64>) -> Mat<f64> {
    // 3行ずつ4つの個体(`four_clusters`と同じ並び)。
    Mat::from_fn(12, m.ncols(), |i, j| {
        let start = i / 3 * 3;
        let mean = (start..start + 3).map(|r| *m.get(r, j)).sum::<f64>() / 3.0;
        *m.get(i, j) - mean
    })
}

/// 個体ダミーを外生変数に入れた計算(`n_absorbed = 0`)と、within変換したデータに吸収した個体の数を
/// `n_absorbed`で渡した計算は、等分散・HC0・HC1・クラスター・HACで一致する(Frisch-Waugh-Lovell)。
#[test]
fn dummies_in_the_regressors_equal_the_demeaned_data_with_absorbed_parameters() {
    let (y, exog, z) = data();
    let x = Mat::from_fn(12, 1, |i, _| *exog.get(i, 1));
    let dummies = Mat::from_fn(12, 5, |i, j| {
        if j < 4 {
            f64::from(u8::from(i / 3 == j))
        } else {
            *x.get(i, 0)
        }
    });
    let groups = four_clusters();
    let order: Vec<usize> = (0..12).collect();
    let (y_w, x_w, z_w) = (
        demean_by_entity(&y),
        demean_by_entity(&x),
        demean_by_entity(&z),
    );
    for name in ["classical", "hc0", "hc1", "cluster", "hac"] {
        let cov = covariance(name, &groups, &order);
        let with_dummies_lm = lm(&y, &dummies, &z, 0, cov);
        let within_lm = lm(&y_w, &x_w, &z_w, 4, cov);
        assert_close(within_lm.statistic, with_dummies_lm.statistic, 1e-9);
        assert_close(
            f(&y_w, &x_w, &z_w, 4, cov),
            f(&y, &dummies, &z, 0, cov),
            1e-9,
        );
    }
}
