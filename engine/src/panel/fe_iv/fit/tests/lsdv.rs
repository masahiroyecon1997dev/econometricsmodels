//! ダミー変数を明示した2SLS（LSDV）との照合。
//!
//! 効果の各次元のダミー（2つ目以降は基準の水準を落とす）を、構造式の外生変数と操作変数の両方に加えた
//! 2SLSは、Frisch-Waugh-Lovellにより、within変換した2SLSと係数・共分散・固定効果が一致する。共分散は、
//! ダミー込みの設計行列`X̂_full`から標準の式で組み立てる（古典・HC1〜HC3・クラスター）。HC2・HC3の
//! レバレッジは`X̂_full`のハット行列の対角で、within変換した`X̂`の素のレバレッジにダミーの
//! レバレッジを足したもの。クラスターの小標本補正の`K`は、fixestの`K.fixef="nonnested"`を、
//! 入れ子になった次元の水準数から組み直す。

use std::collections::BTreeMap;

use faer::Mat;

use super::{all_rows, entity_labels, fit, input_spec, input_with, third_labels, time_labels};
use crate::panel::fe::{FeCovType, FeEffects, FixedEffects};
use crate::panel::fe_iv::{FeIvEstimator, FeIvInput};
use crate::shared::group_codes::GroupCodes;
use crate::shared::least_squares::least_squares;

const RTOL: f64 = 1e-9;

fn assert_close(actual: f64, expected: f64, label: &str) {
    let scale = expected.abs().max(1e-300);
    assert!(
        (actual - expected).abs() <= RTOL * scale,
        "{label}: {actual} vs {expected}"
    );
}

/// 次元ごとのダミー列（1つ目の次元は全水準、2つ目以降は最初の水準を落とす）。
fn dummies(dimensions: &[GroupCodes]) -> Vec<Vec<f64>> {
    let mut columns = Vec::new();
    for (d, dimension) in dimensions.iter().enumerate() {
        for level in usize::from(d > 0)..dimension.n_groups() {
            columns.push(
                dimension
                    .codes()
                    .iter()
                    .map(|&code| f64::from(u8::from(code == level)))
                    .collect(),
            );
        }
    }
    columns
}

fn mat(columns: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(columns[0].len(), columns.len(), |i, j| columns[j][i])
}

/// 次元`level`の各水準が、`cluster`の1つの値にしか現れないか（クラスターに入れ子か）。
fn nested_in(level: &GroupCodes, cluster: &GroupCodes) -> bool {
    let mut owner = vec![usize::MAX; level.n_groups()];
    level
        .codes()
        .iter()
        .zip(cluster.codes())
        .all(|(&code, &group)| {
            if owner[code] == usize::MAX {
                owner[code] = group;
            }
            owner[code] == group
        })
}

struct Lsdv {
    /// 構造式の係数（`x_exog`、`x_endog`の順）。
    params: Vec<f64>,
    /// 係数の共分散（k, k）。
    cov: Mat<f64>,
    /// 推論の自由度。
    df_inference: usize,
    /// ダミーの係数（`dummies`の順）。
    dummy_params: Vec<f64>,
}

/// ダミー込みの2SLSで推定する。`cluster`は`FeCovType::Cluster`のクラスター変数（`None`は個体）。
fn lsdv(input: &FeIvInput, cov_type: &FeCovType, cluster: Option<&GroupCodes>) -> Lsdv {
    let codes = input.index().effect_codes().to_vec();
    let n = input.nobs();
    let d = dummies(&codes);
    let (ke, kn) = (input.k_exog(), input.k_endog());
    let k = ke + kn;

    let mut x_full: Vec<Vec<f64>> = input.x_exog().to_vec();
    x_full.extend(input.x_endog().iter().cloned());
    x_full.extend(d.iter().cloned());
    let mut z_full: Vec<Vec<f64>> = input.x_exog().to_vec();
    z_full.extend(input.instruments().iter().cloned());
    z_full.extend(d.iter().cloned());

    // 第一段階の当てはめ値で内生変数を置き換える。
    let z_mat = mat(&z_full);
    let mut x_hat_columns = x_full.clone();
    for j in 0..kn {
        let first = least_squares(&z_mat, &mat(&[input.x_endog()[j].clone()])).unwrap();
        let fitted = &z_mat * &first.params;
        x_hat_columns[ke + j] = (0..n).map(|i| *fitted.get(i, 0)).collect();
    }
    let x_hat = mat(&x_hat_columns);

    let y = mat(&[input.y().to_vec()]);
    let second = least_squares(&x_hat, &y).unwrap();
    let residuals = &y - mat(&x_full) * &second.params;
    let e: Vec<f64> = (0..n).map(|i| *residuals.get(i, 0)).collect();

    let big_k = x_full.len();
    let df_resid = n - big_k;
    let a = &second.xtx_inv;
    let sandwich = |scores: &Mat<f64>| a * scores * a;
    let meat = |weights: &[f64]| {
        Mat::from_fn(big_k, big_k, |r, c| {
            (0..n)
                .map(|i| weights[i] * x_hat.get(i, r) * x_hat.get(i, c))
                .sum::<f64>()
        })
    };
    let leverage: Vec<f64> = (0..n)
        .map(|i| {
            let row = Mat::from_fn(1, big_k, |_, c| *x_hat.get(i, c));
            *(&row * a * row.transpose()).get(0, 0)
        })
        .collect();
    let scaled = |scale: f64, m: Mat<f64>| Mat::from_fn(big_k, big_k, |r, c| scale * m.get(r, c));

    let (full_cov, df_inference) = match cov_type {
        FeCovType::Classical => {
            let sigma2 = e.iter().map(|v| v * v).sum::<f64>() / df_resid as f64;
            (scaled(sigma2, a.to_owned()), df_resid)
        }
        FeCovType::Hc1 => {
            let w: Vec<f64> = e.iter().map(|v| v * v).collect();
            let scale = n as f64 / df_resid as f64;
            (scaled(scale, sandwich(&meat(&w))), df_resid)
        }
        FeCovType::Hc2 => {
            let w: Vec<f64> = e
                .iter()
                .zip(&leverage)
                .map(|(v, h)| v * v / (1.0 - h))
                .collect();
            (sandwich(&meat(&w)), df_resid)
        }
        FeCovType::Hc3 => {
            let w: Vec<f64> = e
                .iter()
                .zip(&leverage)
                .map(|(v, h)| v * v / (1.0 - h) / (1.0 - h))
                .collect();
            (sandwich(&meat(&w)), df_resid)
        }
        FeCovType::Cluster { .. } => {
            let group = cluster.unwrap_or(&codes[0]);
            let g = group.n_groups();
            let mut s = Mat::<f64>::zeros(big_k, big_k);
            for indices in group.group_indices().iter() {
                let total: Vec<f64> = (0..big_k)
                    .map(|c| indices.iter().map(|&i| x_hat.get(i, c) * e[i]).sum())
                    .collect();
                s = &s + Mat::from_fn(big_k, big_k, |r, c| total[r] * total[c]);
            }
            // fixestの`K.fixef="nonnested"`: クラスターに入れ子の次元を、水準数ぶん`K`から除く
            // （全ての次元が入れ子なら次元の数ぶん、一部なら入れ子の次元の数ぶんを残す）。
            let nested: Vec<bool> = codes.iter().map(|c| nested_in(c, group)).collect();
            let nested_size: usize = codes
                .iter()
                .zip(&nested)
                .filter(|(_, is_nested)| **is_nested)
                .map(|(c, _)| c.n_groups())
                .sum();
            let count_nested = nested.iter().filter(|&&b| b).count();
            let m = codes.len();
            let k_cluster = if count_nested == 0 {
                big_k
            } else if count_nested == m {
                big_k - nested_size + m
            } else {
                big_k - (nested_size - count_nested)
            };
            let scale = g as f64 / (g - 1) as f64 * (n - 1) as f64 / (n - k_cluster) as f64;
            (scaled(scale, sandwich(&s)), g - 1)
        }
        FeCovType::Dk { .. } => unreachable!("not used by the oracle"),
    };
    Lsdv {
        params: (0..k).map(|j| *second.params.get(j, 0)).collect(),
        cov: Mat::from_fn(k, k, |r, c| *full_cov.get(r, c)),
        df_inference,
        dummy_params: (k..big_k).map(|j| *second.params.get(j, 0)).collect(),
    }
}

#[allow(clippy::too_many_arguments)]
fn check_spec(
    rows: &[usize],
    dimensions: &[Vec<String>],
    effects: FeEffects,
    cov_type: FeCovType,
    cluster: Option<&GroupCodes>,
    spec: (usize, usize, usize),
    label: &str,
) -> FeIvEstimator {
    let make = || input_spec(rows, dimensions, spec.0, spec.1, spec.2);
    let oracle = lsdv(&make(), &cov_type, cluster);
    let estimator = fit(make(), effects, cov_type).unwrap();
    let k = spec.0 + spec.1;
    for j in 0..k {
        assert_close(
            *estimator.params().get(j, 0),
            oracle.params[j],
            &format!("{label} coef {j}"),
        );
        for c in 0..k {
            assert_close(
                *estimator.cov_params().get(j, c),
                *oracle.cov.get(j, c),
                &format!("{label} cov {j}{c}"),
            );
        }
    }
    assert_eq!(estimator.df_inference(), oracle.df_inference, "{label} df");
    estimator
}

fn check(
    rows: &[usize],
    dimensions: &[Vec<String>],
    effects: FeEffects,
    cov_type: FeCovType,
    label: &str,
) {
    check_spec(rows, dimensions, effects, cov_type, None, (1, 1, 2), label);
}

const COVS: [fn() -> FeCovType; 4] = [
    || FeCovType::Classical,
    || FeCovType::Hc1,
    || FeCovType::Hc2,
    || FeCovType::Hc3,
];

fn unbalanced_rows() -> Vec<usize> {
    all_rows()
        .into_iter()
        .filter(|r| ![3, 17, 22, 30].contains(r))
        .collect()
}

#[test]
fn one_way_matches_the_dummy_variable_2sls() {
    let dims = [entity_labels()];
    for cov in COVS {
        check(&all_rows(), &dims, FeEffects::OneWay, cov(), "1-way");
    }
    check(
        &all_rows(),
        &dims,
        FeEffects::OneWay,
        FeCovType::Cluster { groups: None },
        "1-way cluster",
    );
}

#[test]
fn two_way_balanced_matches_the_dummy_variable_2sls() {
    let dims = [entity_labels(), time_labels()];
    for cov in COVS {
        check(&all_rows(), &dims, FeEffects::TwoWay, cov(), "2-way");
    }
    check(
        &all_rows(),
        &dims,
        FeEffects::TwoWay,
        FeCovType::Cluster { groups: None },
        "2-way cluster",
    );
}

/// 行を落とした不均衡な2-way（交互射影）。
#[test]
fn two_way_unbalanced_matches_the_dummy_variable_2sls() {
    let dims = [entity_labels(), time_labels()];
    let rows = unbalanced_rows();
    for cov in COVS {
        check(&rows, &dims, FeEffects::TwoWay, cov(), "2-way unbalanced");
    }
    check(
        &rows,
        &dims,
        FeEffects::TwoWay,
        FeCovType::Cluster { groups: None },
        "2-way unbalanced cluster",
    );
}

#[test]
fn three_dimensions_match_the_dummy_variable_2sls() {
    let dims = [entity_labels(), time_labels(), third_labels()];
    for cov in COVS {
        check(&all_rows(), &dims, FeEffects::MultiWay, cov(), "3-way");
    }
    check(
        &all_rows(),
        &dims,
        FeEffects::MultiWay,
        FeCovType::Cluster { groups: None },
        "3-way cluster",
    );
}

/// 内生変数2個・操作変数3個・外生変数0〜2個。列の並び（`x_endog[1]`の当てはめ値、`x̃`・`ẑ`の切り出し、
/// 吸収した効果の列の対応）は、`j = 0`だけでは確かめられない。
#[test]
fn several_endogenous_variables_match_the_dummy_variable_2sls() {
    let balanced = vec![entity_labels(), time_labels()];
    let three = vec![entity_labels(), time_labels(), third_labels()];
    let rows = unbalanced_rows();
    for spec in [(1, 2, 3), (0, 2, 3), (2, 2, 3), (0, 1, 2)] {
        for (dims, effects, rows, name) in [
            (&balanced, FeEffects::TwoWay, &all_rows(), "2-way"),
            (&balanced, FeEffects::TwoWay, &rows, "2-way unbalanced"),
            (&three, FeEffects::MultiWay, &all_rows(), "3-way"),
        ] {
            for cov in COVS {
                let label = format!("{name} spec {spec:?}");
                check_spec(rows, dims, effects, cov(), None, spec, &label);
            }
        }
    }
}

/// クラスターが個体に入れ子でない場合（時点、3つ目の次元）の小標本補正`K`。
#[test]
fn clusters_that_the_effects_are_not_nested_in_match_the_dummy_variable_2sls() {
    let entity = entity_labels();
    let time = time_labels();
    let third = third_labels();
    let by_time = GroupCodes::from_labels(&time);
    let by_third = GroupCodes::from_labels(&third);
    let cases = [
        // 個体のみのFEで、時点・3つ目でクラスタリング（どの次元も入れ子でない）。
        (
            vec![entity.clone()],
            FeEffects::OneWay,
            by_time.clone(),
            "1-way, cluster by time",
        ),
        // 個体+時点のFEで、時点でクラスタリング（時点の次元だけ入れ子）。
        (
            vec![entity.clone(), time.clone()],
            FeEffects::TwoWay,
            by_time.clone(),
            "2-way, cluster by time",
        ),
        // 個体+時点のFEで、3つ目でクラスタリング（入れ子の次元なし）。
        (
            vec![entity.clone(), time.clone()],
            FeEffects::TwoWay,
            by_third.clone(),
            "2-way, cluster by third",
        ),
        (
            vec![entity, time, third],
            FeEffects::MultiWay,
            by_third,
            "3-way, cluster by third",
        ),
    ];
    for (dims, effects, cluster, label) in cases {
        let cov = FeCovType::Cluster {
            groups: Some(cluster.clone()),
        };
        check_spec(
            &all_rows(),
            &dims,
            effects,
            cov,
            Some(&cluster),
            (1, 1, 2),
            label,
        );
    }
}

/// `fixed_effects()`は、ダミー込みの2SLSのダミーの係数（同じ基準の水準）と一致する。
#[test]
fn fixed_effects_match_the_dummy_coefficients() {
    let rows = unbalanced_rows();
    let dims = [entity_labels(), time_labels(), third_labels()];
    for (effects, dimensions, rows) in [
        (FeEffects::TwoWay, &dims[..2], &all_rows()),
        (FeEffects::TwoWay, &dims[..2], &rows),
        (FeEffects::MultiWay, &dims[..], &all_rows()),
    ] {
        let input = input_with(rows, dimensions);
        let oracle = lsdv(&input, &FeCovType::Classical, None);
        let estimator = fit(input_with(rows, dimensions), effects, FeCovType::Classical).unwrap();
        let codes = estimator.input().index().effect_codes().to_vec();

        // 次元ごとの(水準, ダミーの係数)。1つ目は全水準、2つ目以降は最初の水準が0。
        let mut expected: Vec<BTreeMap<String, f64>> = Vec::new();
        let mut at = 0;
        for (d, dimension) in codes.iter().enumerate() {
            let mut map = BTreeMap::new();
            for (level, key) in dimension.keys().iter().enumerate() {
                if d > 0 && level == 0 {
                    map.insert(key.clone(), 0.0);
                } else {
                    map.insert(key.clone(), oracle.dummy_params[at]);
                    at += 1;
                }
            }
            expected.push(map);
        }
        match estimator.fixed_effects() {
            FixedEffects::TwoWay { entity, time } => {
                for (key, value) in &entity {
                    assert_close(*value, expected[0][key], &format!("entity {key}"));
                }
                for (key, value) in &time {
                    assert_close(*value, expected[1][key], &format!("time {key}"));
                }
            }
            FixedEffects::MultiWay(per_dimension) => {
                for (d, effects) in per_dimension.iter().enumerate() {
                    for (key, value) in effects {
                        assert_close(*value, expected[d][key], &format!("dim {d} {key}"));
                    }
                }
            }
            FixedEffects::OneWay(_) => panic!("not a one-way fit"),
        }
    }
}
