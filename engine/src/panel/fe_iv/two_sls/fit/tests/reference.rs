//! fixestの`feols(y ~ x1 | 固定効果 | x2 ~ z1 + z2)`との照合（`data.rs`のデータ）。fixestの係数の順は
//! `fit_x2`・`x1`なので、下の定数は本実装の順（`x1`・`x2`）に並べ替えてある。
//!
//! fixestの`vcov`は、`iid`がClassical、`hetero`がHC1、`cluster = ~i`が個体クラスター。

use super::{entity_cluster, fit, one_way_input, time_labels, two_way_input};
use crate::panel::fe::{FeCovType, FeEffects};
use crate::shared::group_codes::GroupCodes;

/// 時点でクラスタリングする`cov_type`（`feols(..., cluster = ~t)`）。
fn time_cluster() -> FeCovType {
    FeCovType::Cluster {
        groups: Some(GroupCodes::from_labels(&time_labels())),
    }
}

const RTOL: f64 = 1e-9;

fn assert_close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= RTOL * expected.abs(),
        "{label}: {actual} vs {expected}"
    );
}

struct Reference {
    coef: [f64; 2],
    se: [f64; 2],
}

fn check(effects: FeEffects, cov_type: FeCovType, reference: &Reference, label: &str) {
    let input = match effects {
        FeEffects::OneWay => one_way_input(),
        _ => two_way_input(),
    };
    let estimator = fit(input, effects, cov_type).unwrap();
    for j in 0..2 {
        assert_close(*estimator.params().get(j, 0), reference.coef[j], label);
        assert_close(*estimator.std_errors().get(j, 0), reference.se[j], label);
    }
}

const ONE_WAY_COEF: [f64; 2] = [0.451138858245, 0.618754769098];
const TWO_WAY_COEF: [f64; 2] = [0.546846763863, 0.836314975881];

#[test]
fn one_way_matches_fixest() {
    for (cov_type, se, label) in [
        (
            FeCovType::Classical,
            [0.312261680415, 0.464895781385],
            "classical",
        ),
        (FeCovType::Hc1, [0.320210280559, 0.434012490890], "hc1"),
        (
            entity_cluster(),
            [0.307946174543, 0.448913149769],
            "cluster",
        ),
        (
            time_cluster(),
            [0.330945269947, 0.316506370130],
            "cluster by time",
        ),
    ] {
        let reference = Reference {
            coef: ONE_WAY_COEF,
            se,
        };
        check(
            FeEffects::OneWay,
            cov_type,
            &reference,
            &format!("1-way {label}"),
        );
    }
}

#[test]
fn two_way_matches_fixest() {
    for (cov_type, se, label) in [
        (
            FeCovType::Classical,
            [0.164639148226, 0.232318774091],
            "classical",
        ),
        (FeCovType::Hc1, [0.134098892425, 0.168693102370], "hc1"),
        (
            entity_cluster(),
            [0.111427896277, 0.139069054410],
            "cluster",
        ),
        (
            time_cluster(),
            [0.097340635684, 0.224708671260],
            "cluster by time",
        ),
    ] {
        let reference = Reference {
            coef: TWO_WAY_COEF,
            se,
        };
        check(
            FeEffects::TwoWay,
            cov_type,
            &reference,
            &format!("2-way {label}"),
        );
    }
}
