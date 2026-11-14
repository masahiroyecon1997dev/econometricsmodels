//! 列のスケール・線形結合・行の並べ替えに対する不変性と、`Hac`の時間順。

use super::*;

/// `Hac`の`order`は時間順を表す。行を並べ替えたデータに、元の時間順を表す`order`を渡した結果は、
/// 元のデータに恒等順序を渡した結果と一致し、恒等順序のままでは一致しない(逆順では核が対称で
/// 区別できないので、非自明な並べ替えを使う)。
#[test]
fn hac_uses_the_given_order_of_the_rows() {
    let (y, exog, z) = data();
    // 並べ替え後の行`i`は元の行`rows[i]`。
    let rows: Vec<usize> = vec![7, 2, 11, 0, 5, 9, 1, 10, 3, 6, 8, 4];
    let permute = |m: &Mat<f64>| Mat::from_fn(12, m.ncols(), |i, j| *m.get(rows[i], j));
    let (y_p, exog_p, z_p) = (permute(&y), permute(&exog), permute(&z));
    let identity: Vec<usize> = (0..12).collect();
    // 並べ替え後のデータを元の時間順に読む添字。
    let mut time_sorted: Vec<usize> = (0..12).collect();
    time_sorted.sort_by_key(|&i| rows[i]);

    fn hac(order: &[usize]) -> ScoreCovariance<'_> {
        ScoreCovariance::Hac { lags: 2, order }
    }
    let expected = f(&y, &exog, &z, 0, hac(&identity));
    assert_close(
        f(&y_p, &exog_p, &z_p, 0, hac(&time_sorted)),
        expected,
        1e-10,
    );
    let ignored = f(&y_p, &exog_p, &z_p, 0, hac(&identity));
    assert!((ignored - expected).abs() > 1e-6 * expected.abs());
}

/// 列のスケール・並べ替えと、操作変数の正則な線形結合は、統計量を変えない。
#[test]
fn it_is_invariant_to_column_scaling_and_to_invertible_recombination() {
    let (y, exog, z) = data();
    let groups = four_clusters();
    let order: Vec<usize> = (0..12).collect();
    let y_scaled = Mat::from_fn(12, 2, |i, j| *y.get(i, 1 - j) * [1000.0, 0.001][j]);
    let z_mixed = Mat::from_fn(12, 3, |i, j| match j {
        0 => *z.get(i, 0) * 50.0 + *z.get(i, 1) - 2.0 * *z.get(i, 2),
        1 => *z.get(i, 1) * 0.01,
        _ => *z.get(i, 2) + 0.5 * *z.get(i, 0),
    });
    let exog_scaled = Mat::from_fn(12, 2, |i, j| *exog.get(i, j) * [3.0, 1e-3][j]);
    for name in ["classical", "hc1", "hc2", "hc3", "cluster", "hac"] {
        let cov = covariance(name, &groups, &order);
        let base_lm = lm(&y, &exog, &z, 0, cov).statistic;
        let base_f = f(&y, &exog, &z, 0, cov);
        assert_close(
            lm(&y_scaled, &exog_scaled, &z_mixed, 0, cov).statistic,
            base_lm,
            1e-8,
        );
        assert_close(f(&y_scaled, &exog_scaled, &z_mixed, 0, cov), base_f, 1e-8);
    }
}

#[test]
fn it_does_not_depend_on_the_row_order_for_hc_and_cluster() {
    let (y, exog, z) = data();
    let rows: Vec<usize> = vec![7, 2, 11, 0, 5, 9, 1, 10, 3, 6, 8, 4];
    let permute = |m: &Mat<f64>| Mat::from_fn(12, m.ncols(), |i, j| *m.get(rows[i], j));
    let labels: Vec<String> = rows
        .iter()
        .map(|&r| ["a", "b", "c", "d"][r / 3].to_string())
        .collect();
    let permuted_groups = GroupCodes::from_labels_without_keys(&labels);
    let groups = four_clusters();

    for (cov, cov_permuted) in [
        (
            ScoreCovariance::Hc(HcVariant::Hc3),
            ScoreCovariance::Hc(HcVariant::Hc3),
        ),
        (
            ScoreCovariance::Cluster(&groups),
            ScoreCovariance::Cluster(&permuted_groups),
        ),
    ] {
        let a = f(&y, &exog, &z, 0, cov);
        let b = f(&permute(&y), &permute(&exog), &permute(&z), 0, cov_permuted);
        assert_close(b, a, 1e-10);
    }
}
