//! クラスター重み（クラスター数と操作変数の数・モーメント共分散）のテスト。

use crate::iv::common::IvEstimate;
use crate::iv::common::IvInput;
use crate::iv::gmm::GmmEstimator;
use crate::iv::gmm::GmmType;
use crate::iv::gmm::test_support::*;
use crate::linear::ols::CovType;
use crate::shared::gmm_weight::WeightType;
use crate::shared::gmm_weight::cluster_moment_covariance;
use crate::shared::gmm_weight::validate_weight_type;
use crate::shared::group_codes::GroupCodes;
use faer::Mat;
use faer::Side;
use faer::prelude::Solve;

/// クラスター頑健版が、4グループの手計算オラクルと一致することを確認する。
/// `robust`版のオラクル（`fit_computes_robust_weighted_estimate_matching_manual_
/// formula`）と同様、`gmm_point_estimate`（SUT内部関数）を経由せず`llt`/`solve`を
/// ゼロから書き下ろした完全に独立な再導出にする（`S`の構築だけでなく、
/// bread/meat計算自体も独立検証する）。
///
/// `S`（l×l、ここではl=3）はG個のランク1行列の和のため`rank(S) <= G`となる
/// （`.claude/rules/testing-policy.md`「テスト用データセット」3.、`two_sls/cov/tests/validation.rs`の
/// クラスターロバストWald検定のG境界と同じ構造的制約）。G=2だと`S`が必然的に
/// 特異になる（`fit_returns_computation_error_when_cluster_count_is_less_than_
/// instrument_count`で確認済み）ため、l=3に対してG=4を使う。
#[test]
fn fit_computes_cluster_weighted_estimate_matching_manual_formula() {
    let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
    let n = y.len();
    let groups: Vec<String> = (0..n).map(|i| format!("g{}", i / (n / 4))).collect();
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        std::slice::from_ref(&x_endog),
        vec!["endog1".to_string()],
        &[z1.clone(), z2.clone()],
        vec!["z1".to_string(), "z2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();
    let estimator = GmmEstimator::fit(
        input,
        GmmType::TwoStep {
            weight: WeightType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
        },
        true,
        CovType::Classical,
        0.95,
    )
    .unwrap();

    let z = Mat::from_fn(n, 3, |i, j| match j {
        0 => 1.0,
        1 => z1[i],
        _ => z2[i],
    });
    let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
    let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

    let ztz = z.transpose() * &z;
    let ztz_inv = ztz
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(3, 3));
    let zty = z.transpose() * &y_mat;
    let ztx = z.transpose() * &x;
    let bread0 = ztx.transpose() * &ztz_inv * &ztx;
    let meat0 = ztx.transpose() * &ztz_inv * &zty;
    let beta0 = bread0
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(2, 2))
        * &meat0;
    let e0 = &y_mat - &x * &beta0;

    let mut s = Mat::<f64>::zeros(3, 3);
    for g in ["g0", "g1", "g2", "g3"] {
        let mut s_g = [0.0_f64; 3];
        for (i, group) in groups.iter().enumerate().take(n) {
            if group == g {
                let e = *e0.get(i, 0);
                for (a, s_g_a) in s_g.iter_mut().enumerate() {
                    *s_g_a += e * (*z.get(i, a));
                }
            }
        }
        for a in 0..3 {
            for b in 0..3 {
                *s.get_mut(a, b) += s_g[a] * s_g[b];
            }
        }
    }
    let s_inv = s
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(3, 3));
    let bread1 = ztx.transpose() * &s_inv * &ztx;
    let meat1 = ztx.transpose() * &s_inv * &zty;
    let expected_beta = bread1
        .llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(2, 2))
        * &meat1;

    for j in 0..2 {
        assert!(
            (*estimator.params().get(j, 0) - *expected_beta.get(j, 0)).abs() < 1e-8,
            "param {j}: got {}, expected {}",
            *estimator.params().get(j, 0),
            *expected_beta.get(j, 0)
        );
    }
}

#[test]
fn validate_weight_type_returns_the_groups_only_for_cluster() {
    let n = 16;
    let cluster = WeightType::Cluster {
        groups: Some(GroupCodes::from_labels_without_keys(&labelled_groups(
            n, 4, "a",
        ))),
    };
    let codes = validate_weight_type(&cluster, n, 2, 3)
        .unwrap()
        .expect("Cluster weight must expose its groups");
    assert_eq!(codes.nobs(), n);
    assert_eq!(codes.n_groups(), 4);

    for weight in [WeightType::Classical, WeightType::Robust] {
        assert!(validate_weight_type(&weight, n, 2, 3).unwrap().is_none());
    }
}

/// 旧実装（`String`ラベルの`BTreeMap`で集計）をそのまま残したoracleと、
/// `cluster_moment_covariance`（コードで行を1回走査して集計）がビット単位で一致する。
#[test]
fn cluster_moment_covariance_is_bit_identical_to_the_string_keyed_btreemap_implementation() {
    let n = 300;
    let l = 4;
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let labels: Vec<String> = (0..n)
        .map(|_| ((next() * 29.0) as usize).to_string())
        .collect();
    let z = Mat::from_fn(n, l, |_, _| next() * 20.0 - 10.0);
    let residuals = Mat::from_fn(n, 1, |_, _| next() * 4.0 - 2.0);

    let mut by_label: std::collections::BTreeMap<&str, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, label) in labels.iter().enumerate() {
        by_label.entry(label.as_str()).or_default().push(i);
    }
    let mut expected = Mat::<f64>::zeros(l, l);
    for rows in by_label.values() {
        let mut s_g = vec![0.0_f64; l];
        for &i in rows {
            let e = *residuals.get(i, 0);
            for (a, s_g_a) in s_g.iter_mut().enumerate() {
                *s_g_a += e * (*z.get(i, a));
            }
        }
        for a in 0..l {
            for b in 0..l {
                *expected.get_mut(a, b) += s_g[a] * s_g[b];
            }
        }
    }

    let codes = GroupCodes::from_labels_without_keys(&labels);
    let actual = cluster_moment_covariance(&z, &residuals, l, &codes);
    for a in 0..l {
        for b in 0..l {
            assert_eq!(actual.get(a, b).to_bits(), expected.get(a, b).to_bits());
        }
    }
}
