//! `cov_type`別の補助回帰との整合のテスト。

use super::manual_hausman_aux;
use crate::linear::ols::CovType;
use crate::panel::common::TimeKeys;
use crate::panel::common::panel_cluster_cov_params;
use crate::panel::common::xtx_inverse;
use crate::panel::re::ReEffects;
use crate::panel::re::ReEstimator;
use crate::panel::re::input::ReCovType;
use crate::panel::re::test_support::*;
use crate::shared::group_codes::GroupCodes;
use crate::shared::wald::wald_f_test;
use statrs::distribution::ChiSquared;
use statrs::distribution::ContinuousCDF;

#[test]
fn re_estimator_fit_hausman_classical_is_invariant_to_time() {
    // 1-wayでは`ReInput`の`time`の有無はハウスマン検定の結果を変えない（`time`は2-wayの時間次元で、DKの時点列は`Dk.time`）。
    let time = strings(&["1", "2", "3", "1", "2", "1", "2", "3", "1", "2", "1", "2"]);
    let base = ReEstimator::fit(
        hausman_two_slope_input(None),
        ReEffects::OneWay,
        ReCovType::Classical,
        0.95,
    )
    .unwrap();
    let with_time = ReEstimator::fit(
        hausman_two_slope_input(Some(&time)),
        ReEffects::OneWay,
        ReCovType::Classical,
        0.95,
    )
    .unwrap();
    assert_eq!(with_time.hausman_statistic(), base.hausman_statistic());
    assert_eq!(with_time.hausman_p_value(), base.hausman_p_value());
    assert_eq!(with_time.hausman_df(), base.hausman_df());
}

#[test]
fn re_estimator_fit_hausman_follows_cov_type_for_ols_covariances() {
    let entity = strings(&["a", "a", "a", "b", "b", "c", "c", "c", "d", "d", "e", "e"]);
    let classical = ReEstimator::fit(
        hausman_two_slope_input(None),
        ReEffects::OneWay,
        ReCovType::Classical,
        0.95,
    )
    .unwrap();
    let cases = [
        (ReCovType::Hc1, CovType::Hc1),
        (ReCovType::Hc2, CovType::Hc2),
        (ReCovType::Hc3, CovType::Hc3),
        (
            ReCovType::Cluster { groups: None },
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&entity)),
            },
        ),
    ];
    for (re_cov, ols_cov) in cases {
        let label = format!("{re_cov:?}");
        let re = ReEstimator::fit(
            hausman_two_slope_input(None),
            ReEffects::OneWay,
            re_cov,
            0.95,
        )
        .unwrap();
        let aux = manual_hausman_aux(re.input(), ols_cov);
        let (f_stat, _) = aux.wald_test_last_columns(2).unwrap();
        let expected_stat = 2.0 * f_stat;
        let expected_p = ChiSquared::new(2.0).unwrap().sf(expected_stat);

        // `OlsEstimator`経由の補助回帰とビット単位で一致する（Clusterは`OlsEstimator`の
        // `String`列グループ化ではなく整数コードの`panel_cluster_cov_params`を使うが、
        // 補正式・加算順は同じ）。
        assert_eq!(re.hausman_df(), Some(2), "{label}");
        assert_eq!(re.hausman_statistic(), Some(expected_stat), "{label}");
        assert_eq!(re.hausman_p_value(), Some(expected_p), "{label}");
        assert_ne!(
            re.hausman_statistic(),
            classical.hausman_statistic(),
            "{label}"
        );
    }
}

#[test]
fn re_estimator_fit_hausman_cluster_uses_explicit_groups() {
    // `groups`を明示するとentityではなくその列でクラスタリングする。
    let groups = strings(&[
        "g1", "g1", "g1", "g2", "g2", "g3", "g3", "g3", "g4", "g4", "g5", "g5",
    ]);
    let groups_by_obs = strings(&["1", "2", "3", "4", "5", "1", "2", "3", "4", "5", "1", "2"]);
    let re = ReEstimator::fit(
        hausman_two_slope_input(None),
        ReEffects::OneWay,
        ReCovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups_by_obs)),
        },
        0.95,
    )
    .unwrap();
    let by_entity = ReEstimator::fit(
        hausman_two_slope_input(None),
        ReEffects::OneWay,
        ReCovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups)),
        },
        0.95,
    )
    .unwrap();
    let aux = manual_hausman_aux(
        re.input(),
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&groups_by_obs)),
        },
    );
    let expected = 2.0 * aux.wald_test_last_columns(2).unwrap().0;
    assert_eq!(re.hausman_statistic(), Some(expected));
    assert_ne!(re.hausman_statistic(), by_entity.hausman_statistic());
}

#[test]
fn re_estimator_fit_hausman_dk_with_zero_bandwidth_equals_time_clustered_aux() {
    // Dkのバンド幅0は、時点でクラスタリングした補助回帰（fixest型`K=k_aux`・
    // `G=t_periods`スケール）と代数的に同値。
    let time = strings(&["1", "2", "3", "1", "2", "1", "2", "3", "1", "2", "1", "2"]);
    let re = ReEstimator::fit(
        hausman_two_slope_input(Some(&time)),
        ReEffects::OneWay,
        ReCovType::Dk {
            bandwidth: Some(0),
            time: TimeKeys::lexicographic(time.clone()),
        },
        0.95,
    )
    .unwrap();
    // 2-wayではないが`K=k_aux`（fixestのDKは常にフルカウント）・`G=t_periods`の
    // 時点クラスターは`panel_cluster_cov_params`と一致するため、その経路で
    // 独立に検算する。
    let aux = manual_hausman_aux(re.input(), CovType::Classical);
    let x_mat = aux.input().x();
    let n = aux.input().nobs();
    let k_aux = aux.input().k();
    let xtx_inv = xtx_inverse(x_mat).unwrap();
    let residuals: Vec<f64> = (0..n).map(|i| *aux.residuals().get(i, 0)).collect();
    let time_codes = GroupCodes::from_labels(&time);
    let cov = panel_cluster_cov_params(x_mat, &residuals, &xtx_inv, n, k_aux, &time_codes, k_aux);
    let t_periods = time_codes.n_groups();
    let (f_stat, _) = wald_f_test(aux.params(), &cov, k_aux - 2, 2, t_periods - 1).unwrap();

    assert_eq!(re.hausman_df(), Some(2));
    assert!((re.hausman_statistic().unwrap() - 2.0 * f_stat).abs() < 1e-9);
}

#[test]
fn re_estimator_fit_hausman_dk_default_bandwidth_matches_auto_resolved_value() {
    // `bandwidth: None`は`floor(4*(T/100)^(2/9))`（T=3で1）に解決され、
    // 明示的に同じ値を渡した場合と一致する。
    let time = strings(&["1", "2", "3", "1", "2", "1", "2", "3", "1", "2", "1", "2"]);
    let auto = ReEstimator::fit(
        hausman_two_slope_input(Some(&time)),
        ReEffects::OneWay,
        ReCovType::Dk {
            bandwidth: None,
            time: TimeKeys::lexicographic(time.clone()),
        },
        0.95,
    )
    .unwrap();
    let explicit = ReEstimator::fit(
        hausman_two_slope_input(Some(&time)),
        ReEffects::OneWay,
        ReCovType::Dk {
            bandwidth: Some(1),
            time: TimeKeys::lexicographic(time.clone()),
        },
        0.95,
    )
    .unwrap();
    assert_eq!(auto.hausman_statistic(), explicit.hausman_statistic());
    assert_eq!(auto.hausman_p_value(), explicit.hausman_p_value());
}
