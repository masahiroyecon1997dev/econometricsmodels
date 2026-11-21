//! 第一段階・弱操作変数診断・Wu-Hausman。FE-2SLSと共有する計算で、第一段階だけに依存するので、同じデータ・
//! 同じ`cov_type`なら、GMMの方式・重みによらずFE-2SLSと同じ値になる（FE-2SLSの側は、ダミー変数を明示した
//! IV本体との照合・fixestとの照合が済んでいる）。

use super::{
    STRUCTURES, all_rows, entity_labels, fit_with, fit_with_stages, model, time_labels, two_step,
    unbalanced_rows,
};
use crate::panel::common::{PanelError, TimeKeys};
use crate::panel::fe::FeCovType;
use crate::panel::fe_iv::gmm::{FeGmmEstimator, FeGmmType, FeWeightType};
use crate::panel::fe_iv::two_sls::FeTwoSlsEstimator;
use crate::shared::error::{CommonError, ErrorInfo};
use crate::shared::group_codes::GroupCodes;

fn assert_close(actual: f64, expected: f64, rtol: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= rtol * expected.abs().max(1e-300),
        "{label}: {actual} vs {expected}"
    );
}

fn cov_types(structure: &str) -> Vec<FeCovType> {
    let mut covs = vec![
        FeCovType::Classical,
        FeCovType::Hc1,
        FeCovType::Hc2,
        FeCovType::Hc3,
        FeCovType::Cluster { groups: None },
    ];
    // DKの時点キーは全行の長さが要る（不均衡は行を落とした長さ）。
    if structure != "two_way_unbalanced" {
        covs.push(FeCovType::Dk {
            bandwidth: Some(1),
            time: TimeKeys::lexicographic(time_labels()),
        });
    }
    covs
}

/// 個体でクラスタリングする重み（構造が使う行に揃える）。
fn entity_weight(structure: &str) -> FeWeightType {
    let rows = if structure == "two_way_unbalanced" {
        unbalanced_rows()
    } else {
        all_rows()
    };
    let labels: Vec<String> = rows.iter().map(|&r| entity_labels()[r].clone()).collect();
    FeWeightType::Cluster {
        groups: GroupCodes::from_labels_without_keys(&labels),
    }
}

fn gmm_types(structure: &str) -> Vec<(&'static str, FeGmmType)> {
    vec![
        ("one_step", FeGmmType::OneStep),
        ("two_step_classical", two_step(FeWeightType::Classical)),
        ("two_step_cluster", two_step(entity_weight(structure))),
    ]
}

fn fe_2sls(structure: &str, cov_type: FeCovType) -> FeTwoSlsEstimator {
    let (input, effects) = model(structure);
    FeTwoSlsEstimator::fit(input, effects, cov_type, 0.95).unwrap()
}

/// 両方が成功したときだけ`compare`で比べ、両方が`Err`なら同じ`code`であることを確かめる（診断が定義できない
/// 入力では、同じ理由で定義できない）。
/// 比べたとき（両方が成功したとき）は`true`を返す。
fn same_outcome<T>(
    gmm: Result<T, PanelError>,
    two_sls: Result<T, PanelError>,
    label: &str,
    compare: impl FnOnce(&T, &T),
) -> bool {
    match (gmm, two_sls) {
        (Ok(a), Ok(b)) => {
            compare(&a, &b);
            true
        }
        (Err(a), Err(b)) => {
            assert_eq!(a.report().code(), b.report().code(), "{label}");
            false
        }
        (a, b) => panic!("{label}: GMM {:?}, 2SLS {:?}", a.is_ok(), b.is_ok()),
    }
}

/// 診断ごとに、比較できた件数。
#[derive(Default)]
struct Compared {
    cragg_donald: usize,
    kleibergen_paap: usize,
    effective_f: usize,
}

/// 弱操作変数の部分F・自由度・Cragg-Donald・Kleibergen-Paap・有効Fが、FE-2SLSと一致する。
#[test]
fn weak_instrument_diagnostics_equal_those_of_fe_2sls() {
    let mut compared = Compared::default();
    for structure in STRUCTURES {
        for cov_type in cov_types(structure) {
            let two_sls = fe_2sls(structure, cov_type.clone());
            for (gmm_label, gmm_type) in gmm_types(structure) {
                let label = format!("{structure}/{cov_type:?}/{gmm_label}");
                let (input, effects) = model(structure);
                let gmm = fit_with(input, effects, gmm_type, cov_type.clone()).unwrap();
                weak_instrument_matches(&gmm, &two_sls, &label, &mut compared);
            }
        }
    }
    // 両方が`Err`で比較が空振りしていない（有効Fは内生変数が1個のときだけ定義される）。
    assert!(
        compared.cragg_donald > 50,
        "Cragg-Donald compared {}",
        compared.cragg_donald
    );
    assert!(
        compared.kleibergen_paap > 50,
        "KP compared {}",
        compared.kleibergen_paap
    );
    assert!(
        compared.effective_f > 50,
        "effective F compared {}",
        compared.effective_f
    );
}

fn weak_instrument_matches(
    gmm: &FeGmmEstimator,
    two_sls: &FeTwoSlsEstimator,
    label: &str,
    compared: &mut Compared,
) {
    let (a, b) = (
        gmm.weak_instrument_f_statistics(),
        two_sls.weak_instrument_f_statistics(),
    );
    assert_eq!(a.len(), b.len(), "{label}");
    for ((name_a, f_a), (name_b, f_b)) in a.iter().zip(b) {
        assert_eq!(name_a, name_b, "{label}");
        assert_close(*f_a, *f_b, 1e-12, &format!("{label}: partial F"));
    }
    assert_eq!(
        gmm.weak_instrument_f_df(),
        two_sls.weak_instrument_f_df(),
        "{label}"
    );
    compared.cragg_donald += usize::from(same_outcome(
        gmm.cragg_donald(),
        two_sls.cragg_donald(),
        label,
        |a, b| {
            assert_close(a.statistic, b.statistic, 1e-12, &format!("{label}: CD"));
            assert_eq!((a.n_endog, a.n_excluded), (b.n_endog, b.n_excluded));
        },
    ));
    compared.kleibergen_paap += usize::from(same_outcome(
        gmm.kleibergen_paap(),
        two_sls.kleibergen_paap(),
        label,
        |a, b| {
            assert_eq!(a.rk_lm.is_some(), b.rk_lm.is_some(), "{label}");
            if let (Some(a), Some(b)) = (a.rk_lm, b.rk_lm) {
                assert_close(a.statistic, b.statistic, 1e-10, &format!("{label}: rk LM"));
            }
            assert_eq!(a.rk_wald_f.is_some(), b.rk_wald_f.is_some(), "{label}");
            if let (Some(a), Some(b)) = (a.rk_wald_f, b.rk_wald_f) {
                assert_close(a, b, 1e-10, &format!("{label}: rk Wald F"));
            }
        },
    ));
    compared.effective_f += usize::from(same_outcome(
        gmm.effective_f(0.05),
        two_sls.effective_f(0.05),
        label,
        |a, b| {
            assert_close(a.statistic, b.statistic, 1e-10, &format!("{label}: F_eff"));
            // 一般化版の臨界値は被説明変数`ỹ = e + X̃β̂`から作る。`ỹ`は`β̂`によらないが、足し引きの丸めが違う。
            assert_eq!(a.tsls.is_some(), b.tsls.is_some(), "{label}");
            if let (Some(a), Some(b)) = (a.bias_bound, b.bias_bound) {
                assert_close(a, b, 1e-8, &format!("{label}: bias bound"));
            }
        },
    ));
}

/// Wu-Hausmanが、FE-2SLSと一致する（補助回帰は最小二乗で、GMMの点推定・重みに依らない）。
#[test]
fn wu_hausman_equals_that_of_fe_2sls() {
    for structure in STRUCTURES {
        for cov_type in cov_types(structure) {
            let two_sls = fe_2sls(structure, cov_type.clone());
            for (gmm_label, gmm_type) in gmm_types(structure) {
                let label = format!("{structure}/{cov_type:?}/{gmm_label}");
                let (input, effects) = model(structure);
                let gmm = fit_with(input, effects, gmm_type, cov_type.clone()).unwrap();
                assert_eq!(gmm.wu_hausman_df(), two_sls.wu_hausman_df(), "{label}");
                assert_eq!(
                    gmm.wu_hausman_statistic().is_some(),
                    two_sls.wu_hausman_statistic().is_some(),
                    "{label}"
                );
                if let (Some(a), Some(b)) =
                    (gmm.wu_hausman_statistic(), two_sls.wu_hausman_statistic())
                {
                    assert_close(a, b, 1e-12, &format!("{label}: statistic"));
                }
                if let (Some(a), Some(b)) = (gmm.wu_hausman_p_value(), two_sls.wu_hausman_p_value())
                {
                    assert_close(a, b, 1e-10, &format!("{label}: p-value"));
                }
            }
        }
    }
}

/// 第一段階の推定量は、FE-2SLSに渡すものと係数・残差・標準誤差が一致する（GMMの`fit`は第一段階を
/// 点推定の後に回すが、第一段階は第二段階の推定量に依らない）。
#[test]
fn first_stages_equal_those_of_fe_2sls() {
    for structure in STRUCTURES {
        for cov_type in cov_types(structure) {
            let (input, effects) = model(structure);
            let (_, two_sls_stages) = FeTwoSlsEstimator::fit_with_first_stages(
                input,
                effects,
                cov_type.clone(),
                0.95,
                &Default::default(),
                |name, stage| (name.to_string(), stage),
            )
            .unwrap();
            let (input, effects) = model(structure);
            let (_, gmm_stages) = fit_with_stages(
                input,
                effects,
                two_step(entity_weight(structure)),
                cov_type.clone(),
            )
            .unwrap();
            assert_eq!(gmm_stages.len(), two_sls_stages.len(), "{structure}");
            for ((gmm_name, gmm), (name, two_sls)) in gmm_stages.iter().zip(&two_sls_stages) {
                let label = format!("{structure}/{cov_type:?}/{name}");
                assert_eq!(gmm_name, name, "{label}");
                assert_eq!(
                    gmm.within_input().param_names(),
                    two_sls.within_input().param_names(),
                    "{label}"
                );
                assert_eq!(gmm.f_statistic(), None, "{label}: no overall test");
                for i in 0..gmm.params().nrows() {
                    assert_close(
                        *gmm.params().get(i, 0),
                        *two_sls.params().get(i, 0),
                        1e-14,
                        &format!("{label}: coefficient {i}"),
                    );
                    assert_close(
                        *gmm.std_errors().get(i, 0),
                        *two_sls.std_errors().get(i, 0),
                        1e-14,
                        &format!("{label}: standard error {i}"),
                    );
                }
                for i in 0..gmm.residuals().nrows() {
                    assert_eq!(
                        gmm.residuals().get(i, 0),
                        two_sls.residuals().get(i, 0),
                        "{label}: residual {i}"
                    );
                }
            }
        }
    }
}

/// 内生変数の数だけ第一段階が`consume`に渡され、`x_endog`の順に名前が付く。
#[test]
fn one_first_stage_per_endogenous_variable_in_order() {
    let (input, effects) = model("two_endogenous");
    let (gmm, stages) = fit_with_stages(
        input,
        effects,
        two_step(FeWeightType::Classical),
        FeCovType::Classical,
    )
    .unwrap();
    assert_eq!(stages.len(), 2);
    let names: Vec<&str> = gmm
        .weak_instrument_f_statistics()
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(names, ["x2", "x3"]);
    // 第一段階の被説明変数は内生変数で、回帰子は`x_exog`、操作変数の順。
    assert_eq!(
        stages[0].1.within_input().param_names(),
        ["x1", "z1", "z2", "z3"]
    );
}

/// GMMのエラーは、第一段階より先に返る（第一段階・Wu-Hausmanは点推定・共分散・Waldの後に回す）:
/// クラスター重みの列の長さが観測数と違えば、第一段階を回さずに`fit`が失敗する。
#[test]
fn gmm_errors_are_reported_before_the_first_stage_is_run() {
    let (input, effects) = model("one_way");
    let labels: Vec<String> = (0..39).map(|r| format!("g{}", r % 5)).collect();
    let mut calls = 0;
    let result = FeGmmEstimator::fit_with_first_stages(
        input,
        effects,
        two_step(FeWeightType::Cluster {
            groups: GroupCodes::from_labels_without_keys(&labels),
        }),
        true,
        FeCovType::Classical,
        0.95,
        &Default::default(),
        |_, _| calls += 1,
    );
    assert_eq!(
        result.err(),
        Some(PanelError::Common(CommonError::ClusterDimensionMismatch {
            groups_rows: 39,
            n: 40
        }))
    );
    assert_eq!(calls, 0, "the first stage must not run after a GMM error");
}
