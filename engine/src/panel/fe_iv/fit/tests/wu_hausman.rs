//! Wu-Hausman内生性検定。fixestで`v̂`を足した補助FE回帰との照合、ダミー変数を明示した補助回帰での
//! 再計算、`None`になる条件を確かめる。

use faer::Mat;
use faer::prelude::Solve;
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

use super::lsdv::lsdv;
use super::{
    all_rows, entity_labels, fit, input_spec, one_way_input, third_labels, time_labels,
    two_way_input,
};
use crate::panel::common::TimeKeys;
use crate::panel::fe::{FeCovType, FeEffects};
use crate::panel::fe_iv::{FeIvInput, RoleColumns};
use crate::shared::group_codes::GroupCodes;
use crate::shared::least_squares::least_squares;

fn mat(columns: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(columns[0].len(), columns.len(), |i, j| columns[j][i])
}

fn assert_close(actual: f64, expected: f64, rtol: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= rtol * expected.abs(),
        "{label}: {actual} vs {expected}"
    );
}

// ── fixest ─────────────────────────────────────────────────────────

/// `fitstat(feols(y ~ x1 | FE | x2 ~ z1 + z2), "wh")`は、`v̂`（第一段階の残差）を足した補助FE回帰
/// `feols(y ~ x1 + x2 + v | FE)`の`v̂`のt²（`iid`）と一致する（`data.rs`のデータ）。他の`cov_type`は、同じ
/// 補助回帰を`vcov = "hetero"`（HC1）・`cluster = ~i`で回した`v̂`のt²。
struct Reference {
    statistic: f64,
    df2: usize,
}

fn check(effects: FeEffects, cov_type: FeCovType, reference: &Reference, label: &str) {
    let input = match effects {
        FeEffects::OneWay => one_way_input(),
        _ => two_way_input(),
    };
    let estimator = fit(input, effects, cov_type).unwrap();
    assert_close(
        estimator.wu_hausman_statistic().unwrap(),
        reference.statistic,
        1e-9,
        label,
    );
    assert_eq!(
        estimator.wu_hausman_df(),
        Some((1, reference.df2)),
        "{label}"
    );
    let p = FisherSnedecor::new(1.0, reference.df2 as f64)
        .unwrap()
        .sf(reference.statistic);
    assert_close(estimator.wu_hausman_p_value().unwrap(), p, 1e-9, label);
}

#[test]
fn one_way_matches_fixest() {
    let cases = [
        (FeCovType::Classical, 4.948_412_079_51, 27),
        (FeCovType::Hc1, 6.309_517_863_4, 27),
        (FeCovType::Cluster { groups: None }, 5.573_636_320_09, 9),
    ];
    for (cov_type, statistic, df2) in cases {
        check(
            FeEffects::OneWay,
            cov_type,
            &Reference { statistic, df2 },
            "1-way",
        );
    }
    // `fitstat(m, "wh")`のp値。
    let estimator = fit(one_way_input(), FeEffects::OneWay, FeCovType::Classical).unwrap();
    assert_close(
        estimator.wu_hausman_p_value().unwrap(),
        0.034_666_221_117_3,
        1e-9,
        "wh p",
    );
}

#[test]
fn two_way_matches_fixest() {
    let cases = [
        (FeCovType::Classical, 3.241_203_241_01, 24),
        (FeCovType::Hc1, 4.427_623_182_5, 24),
        (FeCovType::Cluster { groups: None }, 3.687_461_357_99, 9),
    ];
    for (cov_type, statistic, df2) in cases {
        check(
            FeEffects::TwoWay,
            cov_type,
            &Reference { statistic, df2 },
            "2-way",
        );
    }
}

// ── ダミー変数を明示した補助回帰 ───────────────────────────────────

/// ダミー込みの第一段階の残差`v̂`（`[x_exog, 操作変数, ダミー]`への回帰）を、`x_exog`・`x_endog`と
/// 並べた補助回帰`y ~ [x_exog, x_endog, v̂, ダミー]`（ダミーは`lsdv`が付ける）を最小二乗で回し、
/// `v̂`の係数の同時Wald F検定を、`cov_type`の共分散（`lsdv`の古典・HC1〜HC3・クラスター）で求める。
/// `fit`のwithin変換・補助回帰・`fe_wu_hausman_test`は通らない。
fn dummy_variable_wu_hausman(
    input: &FeIvInput,
    cov_type: &FeCovType,
    cluster: Option<&GroupCodes>,
) -> (f64, usize, usize) {
    let codes = input.index().effect_codes().to_vec();
    let n = input.nobs();
    let (ke, kn) = (input.k_exog(), input.k_endog());

    let mut dummy_columns: Vec<Vec<f64>> = Vec::new();
    for (d, dimension) in codes.iter().enumerate() {
        for level in usize::from(d > 0)..dimension.n_groups() {
            dummy_columns.push(
                dimension
                    .codes()
                    .iter()
                    .map(|&code| f64::from(u8::from(code == level)))
                    .collect(),
            );
        }
    }
    let mut z_full: Vec<Vec<f64>> = input.x_exog().to_vec();
    z_full.extend(input.instruments().iter().cloned());
    z_full.extend(dummy_columns);
    let z_mat = mat(&z_full);
    let residuals: Vec<Vec<f64>> = input
        .x_endog()
        .iter()
        .map(|endog| {
            let first = least_squares(&z_mat, &mat(std::slice::from_ref(endog))).unwrap();
            (0..n).map(|i| *first.residuals.get(i, 0)).collect()
        })
        .collect();

    // 補助回帰は「内生変数も操作変数も無い」入力で`lsdv`に回す（ダミーは`lsdv`が付ける）。
    let mut regressors: Vec<Vec<f64>> = input.x_exog().to_vec();
    regressors.extend(input.x_endog().iter().cloned());
    regressors.extend(residuals);
    let names: Vec<String> = (0..regressors.len()).map(|j| format!("r{j}")).collect();
    let aux = FeIvInput::from_group_codes(
        input.y(),
        "y".to_string(),
        RoleColumns {
            names,
            columns: &regressors,
        },
        RoleColumns {
            names: vec![],
            columns: &[],
        },
        RoleColumns {
            names: vec![],
            columns: &[],
        },
        codes,
    )
    .unwrap();
    let oracle = lsdv(&aux, cov_type, cluster);

    let start = ke + kn;
    let beta = mat(&[(0..kn).map(|j| oracle.params[start + j]).collect()]);
    let v = Mat::from_fn(kn, kn, |r, c| *oracle.cov.get(start + r, start + c));
    let solved = v.llt(faer::Side::Lower).unwrap().solve(&beta);
    let wald: f64 = (0..kn).map(|j| beta.get(j, 0) * solved.get(j, 0)).sum();
    (wald / kn as f64, kn, oracle.df_inference)
}

fn check_dummy(
    input: FeIvInput,
    effects: FeEffects,
    cov_type: FeCovType,
    cluster: Option<&GroupCodes>,
    label: &str,
) {
    let (statistic, df1, df2) = dummy_variable_wu_hausman(&input, &cov_type, cluster);
    let fitted = fit(input, effects, cov_type).unwrap();
    assert_close(
        fitted.wu_hausman_statistic().unwrap(),
        statistic,
        1e-8,
        label,
    );
    assert_eq!(fitted.wu_hausman_df(), Some((df1, df2)), "{label}");
    let p = FisherSnedecor::new(df1 as f64, df2 as f64)
        .unwrap()
        .sf(statistic);
    assert_close(fitted.wu_hausman_p_value().unwrap(), p, 1e-7, label);
}

#[test]
fn matches_the_dummy_variable_auxiliary_regression() {
    let entity = entity_labels();
    let time = time_labels();
    let third = third_labels();
    let unbalanced: Vec<usize> = all_rows()
        .into_iter()
        .filter(|r| ![3, 17, 22, 30].contains(r))
        .collect();
    let covs: [fn() -> FeCovType; 5] = [
        || FeCovType::Classical,
        || FeCovType::Hc1,
        || FeCovType::Hc2,
        || FeCovType::Hc3,
        || FeCovType::Cluster { groups: None },
    ];
    // (外生変数, 内生変数, 操作変数)。内生変数が1個と2個、過剰識別と丁度識別。
    for spec in [(1, 1, 2), (0, 1, 1), (1, 2, 3), (0, 2, 2)] {
        for (rows, dims, effects, name) in [
            (all_rows(), vec![entity.clone()], FeEffects::OneWay, "1-way"),
            (
                all_rows(),
                vec![entity.clone(), time.clone()],
                FeEffects::TwoWay,
                "2-way",
            ),
            (
                unbalanced.clone(),
                vec![entity.clone(), time.clone()],
                FeEffects::TwoWay,
                "2-way unbalanced",
            ),
            (
                all_rows(),
                vec![entity.clone(), time.clone(), third.clone()],
                FeEffects::MultiWay,
                "3-way",
            ),
        ] {
            for cov in covs {
                let input = input_spec(&rows, &dims, spec.0, spec.1, spec.2);
                check_dummy(
                    input,
                    effects,
                    cov(),
                    None,
                    &format!("{name} spec {spec:?}"),
                );
            }
        }
    }
}

/// 効果が入れ子でないクラスター（時点）の小標本補正`K`。
#[test]
fn matches_the_dummy_variable_auxiliary_regression_with_non_nested_clusters() {
    let by_time = GroupCodes::from_labels(&time_labels());
    let cov = || FeCovType::Cluster {
        groups: Some(GroupCodes::from_labels(&time_labels())),
    };
    for spec in [(1, 1, 2), (0, 1, 2)] {
        let make = || input_spec(&all_rows(), &[entity_labels()], spec.0, spec.1, spec.2);
        check_dummy(make(), FeEffects::OneWay, cov(), Some(&by_time), "1-way");
        let two = || {
            input_spec(
                &all_rows(),
                &[entity_labels(), time_labels()],
                spec.0,
                spec.1,
                spec.2,
            )
        };
        check_dummy(two(), FeEffects::TwoWay, cov(), Some(&by_time), "2-way");
    }
}

// ── 構造 ───────────────────────────────────────────────────────────

/// ラグ項の無いDKは時点でクラスタリングした共分散と同じなので、検定も同じ（自由度は`T − 1`）。1-wayの
/// 個体FEだけで成り立つ（2-wayでは時点FEがクラスターに入れ子になり、`K`の補正が変わる）。
#[test]
fn dk_without_lags_equals_clustering_by_time() {
    let by_time = FeCovType::Cluster {
        groups: Some(GroupCodes::from_labels(&time_labels())),
    };
    let dk = FeCovType::Dk {
        bandwidth: Some(0),
        time: TimeKeys::lexicographic(time_labels()),
    };
    let reference = fit(one_way_input(), FeEffects::OneWay, by_time).unwrap();
    let estimator = fit(one_way_input(), FeEffects::OneWay, dk).unwrap();
    assert_close(
        estimator.wu_hausman_statistic().unwrap(),
        reference.wu_hausman_statistic().unwrap(),
        1e-10,
        "dk",
    );
    assert_eq!(estimator.wu_hausman_df(), Some((1, 3)));
}

/// 内生変数が無ければ、操作変数があっても検定しない。
#[test]
fn is_none_without_endogenous_variables() {
    let input = input_spec(&all_rows(), &[entity_labels()], 1, 0, 2);
    let fitted = fit(input, FeEffects::OneWay, FeCovType::Classical).unwrap();
    assert_eq!(fitted.wu_hausman_statistic(), None);
    assert_eq!(fitted.wu_hausman_p_value(), None);
    assert_eq!(fitted.wu_hausman_df(), None);
}

/// `y`が内生変数と第一段階の残差で厳密に説明されるとき、補助回帰が完全適合で統計量が定義できない。
/// 構造式の推定は成功して`None`になる。
#[test]
fn degrades_to_none_when_the_auxiliary_regression_fits_exactly() {
    let base = fit(one_way_input(), FeEffects::OneWay, FeCovType::Classical).unwrap();
    let v: Vec<f64> = (0..40)
        .map(|i| *base.first_stages()[0].residuals().get(i, 0))
        .collect();
    let input = base.input();
    // `y = 2 x_endog + 3 v̂`（固定効果なし）は、補助回帰`y ~ x_exog + x_endog + v̂ | FE`に厳密に載る。
    let y: Vec<f64> = (0..40)
        .map(|i| 2.0 * input.x_endog()[0][i] + 3.0 * v[i])
        .collect();
    let exact = FeIvInput::from_group_codes(
        &y,
        "y".to_string(),
        RoleColumns {
            names: input.x_exog_names().to_vec(),
            columns: input.x_exog(),
        },
        RoleColumns {
            names: input.x_endog_names().to_vec(),
            columns: input.x_endog(),
        },
        RoleColumns {
            names: input.instrument_names().to_vec(),
            columns: input.instruments(),
        },
        input.index().effect_codes().to_vec(),
    )
    .unwrap();
    let fitted = fit(exact, FeEffects::OneWay, FeCovType::Classical).unwrap();
    assert_eq!(fitted.wu_hausman_statistic(), None);
    assert_eq!(fitted.wu_hausman_p_value(), None);
    assert_eq!(fitted.wu_hausman_df(), None);
}

/// 補助回帰は構造式・第一段階より列が`k_endog`だけ多いので、その境界では補助回帰だけが自由度不足になる。
/// 構造式の推定は成功して`None`になる。
#[test]
fn degrades_to_none_when_only_the_auxiliary_regression_lacks_degrees_of_freedom() {
    // 3個体×2時点の6行。外生1・内生1・操作変数1（丁度識別）: 構造式と第一段階は`n − neffects = 3 > 2`、
    // 補助回帰は`k = 3`個の傾きに`n − neffects = 3`で残差自由度が0。
    let rows = [0, 1, 4, 5, 8, 9];
    let input = input_spec(&rows, &[entity_labels()], 1, 1, 1);
    let fitted = fit(input, FeEffects::OneWay, FeCovType::Classical).unwrap();
    assert_eq!(fitted.df_resid(), 1);
    assert_eq!(fitted.wu_hausman_statistic(), None);
    assert_eq!(fitted.wu_hausman_df(), None);
}

/// クラスター数`G`が`k + 1`〜`k + k_endog`のとき（補助回帰の傾き`k + k_endog`以下）、IV本体は`None`にするが、
/// ここで使うのは`v̂`の`k_endog × k_endog`部分行列だけなので、値を持つ。ダミー込みの補助回帰と一致する。
#[test]
fn has_a_value_when_the_cluster_count_is_at_most_the_auxiliary_slope_count() {
    // 外生0・内生2・操作変数2: 構造式の傾き`k = 2`、補助回帰は4。クラスターは3つ（個体に入れ子）。
    let three: Vec<String> = (0..40).map(|r| (r / 4 % 3).to_string()).collect();
    let groups = GroupCodes::from_labels(&three);
    let cov = FeCovType::Cluster {
        groups: Some(GroupCodes::from_labels(&three)),
    };
    let input = input_spec(&all_rows(), &[entity_labels()], 0, 2, 2);
    check_dummy(input, FeEffects::OneWay, cov, Some(&groups), "G = 3");
}

/// 不均衡な2-way（`noise > 0`）でも、補助回帰に厳密に載る`y = 2 x_endog + 3 v̂`は`None`になる（変換の誤差で
/// 残差は正確には0にならない）。補助回帰の`fit_from_within`の完全適合の判定が先に効くので、
/// `fe_wu_hausman_test`内の誤差の見積もり（`noise`と係数倍した各列のスケール）は、このテストでは
/// 単独で固定されない（その式を無効にしても通る）。防御的な保険。
#[test]
fn degrades_to_none_when_the_auxiliary_regression_fits_up_to_the_transform_noise() {
    let rows: Vec<usize> = all_rows()
        .into_iter()
        .filter(|r| ![3, 17, 22, 30].contains(r))
        .collect();
    let dims = [entity_labels(), time_labels()];
    let base = fit(
        input_spec(&rows, &dims, 1, 1, 2),
        FeEffects::TwoWay,
        FeCovType::Classical,
    )
    .unwrap();
    let n = base.input().nobs();
    let v: Vec<f64> = (0..n)
        .map(|i| *base.first_stages()[0].residuals().get(i, 0))
        .collect();
    let input = base.input();
    let y: Vec<f64> = (0..n)
        .map(|i| 2.0 * input.x_endog()[0][i] + 3.0 * v[i])
        .collect();
    let exact = FeIvInput::from_group_codes(
        &y,
        "y".to_string(),
        RoleColumns {
            names: input.x_exog_names().to_vec(),
            columns: input.x_exog(),
        },
        RoleColumns {
            names: input.x_endog_names().to_vec(),
            columns: input.x_endog(),
        },
        RoleColumns {
            names: input.instrument_names().to_vec(),
            columns: input.instruments(),
        },
        input.index().effect_codes().to_vec(),
    )
    .unwrap();
    let fitted = fit(exact, FeEffects::TwoWay, FeCovType::Classical).unwrap();
    assert_eq!(fitted.wu_hausman_statistic(), None);
}
