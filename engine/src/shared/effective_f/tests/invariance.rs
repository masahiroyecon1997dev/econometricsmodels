//! スケール・線形結合・行の並べ替えと、固定効果の吸収に対する不変性。

use super::*;

/// 列のスケール・操作変数の正則な線形結合・行の並べ替えで、統計量も臨界値も変わらない。
#[test]
fn it_is_invariant_to_scaling_recombination_and_row_order() {
    let (endog, exog, z) = large(4);
    let groups = clusters(240, 12);
    let endog_scaled = Mat::from_fn(240, 1, |i, _| *endog.get(i, 0) * 1000.0);
    let exog_scaled = Mat::from_fn(240, 2, |i, j| *exog.get(i, j) * [3.0, 1e-3][j]);
    let z_mixed = Mat::from_fn(240, 4, |i, j| match j {
        0 => *z.get(i, 0) * 50.0 + *z.get(i, 1) - 2.0 * *z.get(i, 2),
        1 => *z.get(i, 1) * 0.01,
        2 => *z.get(i, 2) + 0.5 * *z.get(i, 0),
        _ => *z.get(i, 3) - *z.get(i, 1),
    });
    for cov in [
        ScoreCovariance::Classical,
        ScoreCovariance::Hc(HcVariant::Hc3),
        ScoreCovariance::Cluster(&groups),
    ] {
        let base = effective_f(&endog, &exog, &z, 0, cov, 0.05).unwrap();
        let other = effective_f(&endog_scaled, &exog_scaled, &z_mixed, 0, cov, 0.05).unwrap();
        close(other.statistic, base.statistic, 1e-8);
        for (a, b) in other.simplified_tsls.iter().zip(base.simplified_tsls) {
            close(a.effective_df, b.effective_df, 1e-8);
            close(a.critical_value, b.critical_value, 1e-8);
        }
    }
    // 行の並べ替え(HC・クラスターは行の並びに依らない)。
    let rows: Vec<usize> = (0..240).map(|i| (i * 7 + 3) % 240).collect();
    let permute = |m: &Mat<f64>| Mat::from_fn(240, m.ncols(), |i, j| *m.get(rows[i], j));
    let labels: Vec<String> = rows.iter().map(|&r| format!("g{:03}", r / 12)).collect();
    let permuted_groups = GroupCodes::from_labels_without_keys(&labels);
    let a = effective_f(
        &endog,
        &exog,
        &z,
        0,
        ScoreCovariance::Cluster(&groups),
        0.05,
    )
    .unwrap();
    let b = effective_f(
        &permute(&endog),
        &permute(&exog),
        &permute(&z),
        0,
        ScoreCovariance::Cluster(&permuted_groups),
        0.05,
    )
    .unwrap();
    close(b.statistic, a.statistic, 1e-10);
}

/// 個体ダミーを外生変数に入れた計算(`n_absorbed = 0`)と、個体ごとに平均除去して吸収した個体の数を
/// `n_absorbed`で渡した計算は一致する(Frisch-Waugh-Lovell)。
#[test]
fn dummies_in_the_regressors_equal_the_demeaned_data_with_absorbed_parameters() {
    let (endog, exog, z) = large(4);
    let groups = clusters(240, 12);
    let order: Vec<usize> = (0..240).collect();
    let demean = |m: &Mat<f64>| {
        Mat::from_fn(240, m.ncols(), |i, j| {
            let start = i / 12 * 12;
            *m.get(i, j) - (start..start + 12).map(|r| *m.get(r, j)).sum::<f64>() / 12.0
        })
    };
    let x = column(&exog, 1);
    let dummies = Mat::from_fn(240, 21, |i, j| {
        if j < 20 {
            f64::from(u8::from(i / 12 == j))
        } else {
            *x.get(i, 0)
        }
    });
    let (endog_w, x_w, z_w) = (demean(&endog), demean(&x), demean(&z));
    for cov in [
        ScoreCovariance::Classical,
        ScoreCovariance::Hc(HcVariant::Hc0),
        ScoreCovariance::Hc(HcVariant::Hc1),
        ScoreCovariance::Cluster(&groups),
        ScoreCovariance::Hac {
            lags: 4,
            order: &order,
        },
    ] {
        let with_dummies = effective_f(&endog, &dummies, &z, 0, cov, 0.05).unwrap();
        let within = effective_f(&endog_w, &x_w, &z_w, 20, cov, 0.05).unwrap();
        close(within.statistic, with_dummies.statistic, 1e-8);
        for (a, b) in within
            .simplified_tsls
            .iter()
            .zip(with_dummies.simplified_tsls)
        {
            close(a.effective_df, b.effective_df, 1e-8);
            close(a.critical_value, b.critical_value, 1e-8);
        }
    }
}

/// 一般化版(`B_TSLS`と臨界値)も、列のスケール・被説明変数の単位・操作変数の線形結合・行の並べ替えで変わらない。
#[test]
fn generalized_values_are_invariant_to_scaling_recombination_and_row_order() {
    let (endog, exog, z) = large(4);
    let y = large_y(&endog, &exog);
    let groups = clusters(240, 12);
    let y_scaled = Mat::from_fn(240, 1, |i, _| *y.get(i, 0) * 7.0);
    let endog_scaled = Mat::from_fn(240, 1, |i, _| *endog.get(i, 0) * 1000.0);
    let exog_scaled = Mat::from_fn(240, 2, |i, j| *exog.get(i, j) * [3.0, 1e-3][j]);
    let z_mixed = Mat::from_fn(240, 4, |i, j| match j {
        0 => *z.get(i, 0) * 50.0 + *z.get(i, 1) - 2.0 * *z.get(i, 2),
        1 => *z.get(i, 1) * 0.01,
        2 => *z.get(i, 2) + 0.5 * *z.get(i, 0),
        _ => *z.get(i, 3) - *z.get(i, 1),
    });
    let general = crate::shared::effective_f::effective_f;
    for cov in [
        ScoreCovariance::Hc(HcVariant::Hc1),
        ScoreCovariance::Hc(HcVariant::Hc3),
        ScoreCovariance::Cluster(&groups),
    ] {
        let base = general(&endog, &exog, &z, 0, cov, 0.05, Some(&y)).unwrap();
        let other = general(
            &endog_scaled,
            &exog_scaled,
            &z_mixed,
            0,
            cov,
            0.05,
            Some(&y_scaled),
        )
        .unwrap();
        close(other.bias_bound.unwrap(), base.bias_bound.unwrap(), 1e-8);
        for (a, b) in other.tsls.unwrap().iter().zip(base.tsls.unwrap()) {
            close(a.effective_df, b.effective_df, 1e-8);
            close(a.critical_value, b.critical_value, 1e-8);
        }
    }
    // 行の並べ替え(クラスターは行の並びに依らない)。
    let rows: Vec<usize> = (0..240).map(|i| (i * 7 + 3) % 240).collect();
    let permute = |m: &Mat<f64>| Mat::from_fn(240, m.ncols(), |i, j| *m.get(rows[i], j));
    let labels: Vec<String> = rows.iter().map(|&r| format!("g{:03}", r / 12)).collect();
    let permuted_groups = GroupCodes::from_labels_without_keys(&labels);
    let a = general(
        &endog,
        &exog,
        &z,
        0,
        ScoreCovariance::Cluster(&groups),
        0.05,
        Some(&y),
    )
    .unwrap();
    let b = general(
        &permute(&endog),
        &permute(&exog),
        &permute(&z),
        0,
        ScoreCovariance::Cluster(&permuted_groups),
        0.05,
        Some(&permute(&y)),
    )
    .unwrap();
    close(b.bias_bound.unwrap(), a.bias_bound.unwrap(), 1e-9);
}

/// 一般化版でも、個体ダミーを外生変数に入れた計算と、平均除去して`n_absorbed`を渡した計算は一致する。
#[test]
fn generalized_dummies_in_the_regressors_equal_the_demeaned_data() {
    let (endog, exog, z) = large(4);
    let y = large_y(&endog, &exog);
    let groups = clusters(240, 12);
    let demean = |m: &Mat<f64>| {
        Mat::from_fn(240, m.ncols(), |i, j| {
            let start = i / 12 * 12;
            *m.get(i, j) - (start..start + 12).map(|r| *m.get(r, j)).sum::<f64>() / 12.0
        })
    };
    let x = column(&exog, 1);
    let dummies = Mat::from_fn(240, 21, |i, j| {
        if j < 20 {
            f64::from(u8::from(i / 12 == j))
        } else {
            *x.get(i, 0)
        }
    });
    let general = crate::shared::effective_f::effective_f;
    for cov in [
        ScoreCovariance::Classical,
        ScoreCovariance::Hc(HcVariant::Hc1),
        ScoreCovariance::Cluster(&groups),
    ] {
        let with_dummies = general(&endog, &dummies, &z, 0, cov, 0.05, Some(&y)).unwrap();
        let within = general(
            &demean(&endog),
            &demean(&x),
            &demean(&z),
            20,
            cov,
            0.05,
            Some(&demean(&y)),
        )
        .unwrap();
        close(
            within.bias_bound.unwrap(),
            with_dummies.bias_bound.unwrap(),
            1e-8,
        );
        for (a, b) in within.tsls.unwrap().iter().zip(with_dummies.tsls.unwrap()) {
            close(a.critical_value, b.critical_value, 1e-8);
        }
    }
}
