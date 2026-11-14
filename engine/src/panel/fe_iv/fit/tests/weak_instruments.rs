//! 弱操作変数診断（部分F・Cragg-Donald・Kleibergen-Paap・有効F）。ダミー変数を回帰子に明示した（吸収する
//! 効果なしの）IV本体の診断との照合、`cov_type`ごとの吸収したパラメータ数、DK、エラーを確かめる。
//!
//! 内生変数・外生変数・操作変数をwithin変換して`n_absorbed`で割り引く計算は、ダミーを回帰子に入れて
//! `n_absorbed = 0`で回した計算と、Frisch-Waugh-Lovellにより一致する（HC2・HC3のレバレッジも、ダミーを
//! 回帰子に入れたときのレバレッジ）。

use super::lsdv::dummies;
use super::{all_rows, entity_labels, fit, input_spec, one_way_input, third_labels, time_labels};
use crate::iv::common::{
    IvInput, compute_cragg_donald, compute_effective_f, compute_first_stage,
    compute_kleibergen_paap,
};
use crate::linear::ols::CovType;
use crate::panel::common::{PanelError, TimeKeys, absorbed_parameters};
use crate::panel::fe::{FeCovType, FeEffects};
use crate::panel::fe_iv::FeIvInput;
use crate::shared::error::ErrorInfo;
use crate::shared::error_catalog::ErrorCode;
use crate::shared::group_codes::GroupCodes;

fn assert_close(actual: f64, expected: f64, rtol: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= rtol * expected.abs().max(1e-300),
        "{label}: {actual} vs {expected}"
    );
}

/// ダミーを回帰子（定数項なし）に入れたIV本体の入力。`extra`は回帰子に足す列（切片の代わりなど）。
fn iv_with_regressors(input: &FeIvInput, extra: Vec<Vec<f64>>, intercept: bool) -> IvInput {
    let mut exog: Vec<Vec<f64>> = input.x_exog().to_vec();
    exog.extend(extra);
    let names: Vec<String> = (0..exog.len()).map(|j| format!("x{j}")).collect();
    IvInput::from_columns(
        input.y(),
        &exog,
        names,
        input.x_endog(),
        input.x_endog_names().to_vec(),
        input.instruments(),
        input.instrument_names().to_vec(),
        intercept,
        "y".to_string(),
    )
    .unwrap()
}

fn dummy_iv(input: &FeIvInput) -> IvInput {
    let d = dummies(input.index().effect_codes());
    iv_with_regressors(input, d, false)
}

/// `FeCovType`に対応するIV本体の`CovType`（`Dk`は対象外）。
fn iv_cov(cov_type: &FeCovType) -> CovType {
    match cov_type {
        FeCovType::Classical => CovType::Classical,
        FeCovType::Hc1 => CovType::Hc1,
        FeCovType::Hc2 => CovType::Hc2,
        FeCovType::Hc3 => CovType::Hc3,
        _ => unreachable!(),
    }
}

fn unbalanced_rows() -> Vec<usize> {
    all_rows()
        .into_iter()
        .filter(|r| ![3, 17, 22, 30].contains(r))
        .collect()
}

/// 全診断を、ダミーを回帰子に入れたIV本体の診断と比べる。
fn check_against_dummies(input: FeIvInput, effects: FeEffects, cov_type: FeCovType, label: &str) {
    let iv = dummy_iv(&input);
    let fitted = fit(input, effects, cov_type.clone()).unwrap();
    let cov = iv_cov(&cov_type);
    let n_endog = fitted.input().k_endog();

    // 部分F（等分散）。
    let (_, weak_f) = compute_first_stage(&iv, &CovType::Classical, 0.95).unwrap();
    for (j, (name, f)) in weak_f.iter().enumerate() {
        let (fe_name, fe_f) = &fitted.weak_instrument_f_statistics()[j];
        assert_eq!(fe_name, name);
        assert_close(*fe_f, *f, 1e-9, &format!("{label} partial F {j}"));
    }
    let l = fitted.input().k_exog() + fitted.input().k_instruments();
    assert_eq!(
        fitted.weak_instrument_f_df(),
        Some((
            fitted.input().k_instruments(),
            fitted.input().nobs() - l - absorbed_parameters(fitted.n_groups())
        )),
        "{label}"
    );

    // Cragg-Donald（等分散）。
    let cd = compute_cragg_donald(&iv).unwrap();
    let fe_cd = fitted.cragg_donald().unwrap();
    assert_close(fe_cd.statistic, cd.statistic, 1e-9, &format!("{label} CD"));
    assert_eq!(
        (
            fe_cd.n_endog,
            fe_cd.n_excluded,
            fe_cd.tsls_size,
            fe_cd.tsls_bias
        ),
        (cd.n_endog, cd.n_excluded, cd.tsls_size, cd.tsls_bias)
    );

    // Kleibergen-Paap（`cov_type`別）。
    let kp = compute_kleibergen_paap(&iv, &cov).unwrap();
    let fe_kp = fitted.kleibergen_paap().unwrap();
    match (fe_kp.rk_lm, kp.rk_lm) {
        (Some(a), Some(b)) => {
            assert_eq!(a.df, b.df);
            assert_close(a.statistic, b.statistic, 1e-8, &format!("{label} rk LM"));
        }
        (a, b) => assert_eq!(a.is_some(), b.is_some(), "{label} rk LM defined"),
    }
    match (fe_kp.rk_wald_f, kp.rk_wald_f) {
        (Some(a), Some(b)) => assert_close(a, b, 1e-8, &format!("{label} rk Wald F")),
        (a, b) => assert_eq!(a.is_some(), b.is_some(), "{label} rk Wald F defined"),
    }

    // 有効F（内生変数1個のときだけ）。
    if n_endog == 1 {
        let eff = compute_effective_f(&iv, &cov, 0.05).unwrap();
        let fe_eff = fitted.effective_f(0.05).unwrap();
        assert_close(
            fe_eff.statistic,
            eff.statistic,
            1e-8,
            &format!("{label} effective F"),
        );
        assert_eq!(fe_eff.bias_bound.is_some(), eff.bias_bound.is_some());
        if let (Some(a), Some(b)) = (fe_eff.bias_bound, eff.bias_bound) {
            assert_close(a, b, 1e-8, &format!("{label} bias bound"));
        }
        assert_eq!(fe_eff.simplified_tsls.len(), eff.simplified_tsls.len());
        assert_eq!(fe_eff.tsls.is_some(), eff.tsls.is_some());
        for (a, b) in fe_eff.simplified_tsls.iter().zip(&eff.simplified_tsls) {
            assert_close(
                a.critical_value,
                b.critical_value,
                1e-8,
                &format!("{label} critical value"),
            );
        }
    }
}

#[test]
fn matches_the_dummy_regressors_across_structures_and_covariance_types() {
    let entity = entity_labels();
    let time = time_labels();
    let third = third_labels();
    let covs: [fn() -> FeCovType; 4] = [
        || FeCovType::Classical,
        || FeCovType::Hc1,
        || FeCovType::Hc2,
        || FeCovType::Hc3,
    ];
    // (外生変数, 内生変数, 操作変数)。内生変数が1個（有効Fも）と2個。
    for spec in [(1, 1, 2), (0, 1, 3), (1, 2, 3)] {
        for (rows, dims, effects, name) in [
            (all_rows(), vec![entity.clone()], FeEffects::OneWay, "1-way"),
            (
                all_rows(),
                vec![entity.clone(), time.clone()],
                FeEffects::TwoWay,
                "2-way",
            ),
            (
                unbalanced_rows(),
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
                check_against_dummies(input, effects, cov(), &format!("{name} spec {spec:?}"));
            }
        }
    }
}

/// クラスターは`K.fixef = "nonnested"`: 効果が入れ子のクラスター（個体）は、within変換に切片を足した
/// IV本体の診断（`n_absorbed = 1`）と、入れ子でないクラスター（時点）はダミーを回帰子に入れたものと一致する。
#[test]
fn cluster_uses_the_nonnested_count_of_absorbed_parameters() {
    let entity_groups = GroupCodes::from_labels(&entity_labels());
    let time_groups = GroupCodes::from_labels(&time_labels());
    for spec in [(1, 1, 2), (1, 2, 3)] {
        let make = || input_spec(&all_rows(), &[entity_labels()], spec.0, spec.1, spec.2);

        // 時点クラスター（個体FEは入れ子でない）: ダミーを回帰子に入れる。
        let by_time = || FeCovType::Cluster {
            groups: Some(GroupCodes::from_labels(&time_labels())),
        };
        let iv_cov = CovType::Cluster {
            groups: Some(time_groups.clone()),
        };
        let fitted = fit(make(), FeEffects::OneWay, by_time()).unwrap();
        let iv = dummy_iv(&make());
        let (kp, fe_kp) = (
            compute_kleibergen_paap(&iv, &iv_cov).unwrap(),
            fitted.kleibergen_paap().unwrap(),
        );
        assert_close(
            fe_kp.rk_wald_f.unwrap(),
            kp.rk_wald_f.unwrap(),
            1e-8,
            "time cluster rk Wald F",
        );
        assert_close(
            fe_kp.rk_lm.unwrap().statistic,
            kp.rk_lm.unwrap().statistic,
            1e-8,
            "time cluster rk LM",
        );
        if spec.1 == 1 {
            let eff = compute_effective_f(&iv, &iv_cov, 0.05).unwrap();
            assert_close(
                fitted.effective_f(0.05).unwrap().statistic,
                eff.statistic,
                1e-8,
                "time cluster effective F",
            );
        }

        // 個体クラスター（個体FEが入れ子）: 個体内で平均を引いた列に切片を足す（`n_absorbed = 1`）。
        let fitted = fit(
            make(),
            FeEffects::OneWay,
            FeCovType::Cluster { groups: None },
        )
        .unwrap();
        let within = within_iv(&fitted);
        let iv_cov = CovType::Cluster {
            groups: Some(entity_groups.clone()),
        };
        let (kp, fe_kp) = (
            compute_kleibergen_paap(&within, &iv_cov).unwrap(),
            fitted.kleibergen_paap().unwrap(),
        );
        assert_close(
            fe_kp.rk_wald_f.unwrap(),
            kp.rk_wald_f.unwrap(),
            1e-8,
            "entity cluster rk Wald F",
        );
        assert_close(
            fe_kp.rk_lm.unwrap().statistic,
            kp.rk_lm.unwrap().statistic,
            1e-8,
            "entity cluster rk LM",
        );
        if spec.1 == 1 {
            let eff = compute_effective_f(&within, &iv_cov, 0.05).unwrap();
            assert_close(
                fitted.effective_f(0.05).unwrap().statistic,
                eff.statistic,
                1e-8,
                "entity cluster effective F",
            );
        }
    }
}

/// within変換した列に切片を足したIV本体の入力（吸収したパラメータ数1に相当）。第一段階の設計行列と
/// 各第一段階の`x̃_endog`から組む。
fn within_iv(fitted: &crate::panel::fe_iv::FeIvEstimator) -> IvInput {
    let input = fitted.input();
    let n = input.nobs();
    let columns = |stage: usize, from: usize, count: usize| -> Vec<Vec<f64>> {
        let x = fitted.first_stages()[stage].within_input().x();
        (from..from + count)
            .map(|j| (0..n).map(|i| *x.get(i, j)).collect())
            .collect()
    };
    let (ke, kz) = (input.k_exog(), input.k_instruments());
    let endog: Vec<Vec<f64>> = (0..input.k_endog())
        .map(|j| {
            let y = fitted.first_stages()[j].within_input().y();
            (0..n).map(|i| *y.get(i, 0)).collect()
        })
        .collect();
    // 被説明変数は診断に使わないが、有効Fの一般化版のために構造のwithin変換`ỹ`を渡す。
    let y_tilde: Vec<f64> = (0..n)
        .map(|i| {
            let mut v = *fitted.residuals().get(i, 0);
            for j in 0..ke {
                v += columns(0, 0, ke)[j][i] * fitted.params().get(j, 0);
            }
            for (j, e) in endog.iter().enumerate() {
                v += e[i] * fitted.params().get(ke + j, 0);
            }
            v
        })
        .collect();
    IvInput::from_columns(
        &y_tilde,
        &columns(0, 0, ke),
        input.x_exog_names().to_vec(),
        &endog,
        input.x_endog_names().to_vec(),
        &columns(0, ke, kz),
        input.instrument_names().to_vec(),
        true,
        "y".to_string(),
    )
    .unwrap()
}

/// ラグ項の無いDKは時点でクラスタリングしたものと同じ（1-wayの個体FEは時点クラスターに入れ子でなく、
/// どちらも`n_absorbed = neffects`、小標本の係数は`T`をクラスター数とみなす）。
#[test]
fn dk_without_lags_equals_clustering_by_time() {
    let dk = || FeCovType::Dk {
        bandwidth: Some(0),
        time: TimeKeys::lexicographic(time_labels()),
    };
    let by_time = || FeCovType::Cluster {
        groups: Some(GroupCodes::from_labels(&time_labels())),
    };
    for spec in [(1, 1, 2), (1, 2, 3)] {
        let make = || input_spec(&all_rows(), &[entity_labels()], spec.0, spec.1, spec.2);
        let a = fit(make(), FeEffects::OneWay, dk()).unwrap();
        let b = fit(make(), FeEffects::OneWay, by_time()).unwrap();
        let (ka, kb) = (a.kleibergen_paap().unwrap(), b.kleibergen_paap().unwrap());
        assert_close(
            ka.rk_wald_f.unwrap(),
            kb.rk_wald_f.unwrap(),
            1e-10,
            "dk rk Wald F",
        );
        assert_close(
            ka.rk_lm.unwrap().statistic,
            kb.rk_lm.unwrap().statistic,
            1e-10,
            "dk rk LM",
        );
        if spec.1 == 1 {
            assert_close(
                a.effective_f(0.05).unwrap().statistic,
                b.effective_f(0.05).unwrap().statistic,
                1e-10,
                "dk effective F",
            );
        }
    }
}

/// 部分F・Cragg-Donaldは、`cov_type`に依らない（等分散前提）。内生変数が1個なら一致する。
#[test]
fn partial_f_and_cragg_donald_do_not_depend_on_the_covariance_type_and_agree_for_one_endogenous() {
    let reference = fit(one_way_input(), FeEffects::OneWay, FeCovType::Classical).unwrap();
    let f = reference.weak_instrument_f_statistics()[0].1;
    assert_close(
        reference.cragg_donald().unwrap().statistic,
        f,
        1e-9,
        "CD = F",
    );
    assert_eq!(
        reference.weak_instrument_f_statistics()[0].0,
        reference.input().x_endog_names()[0]
    );
    for cov in [
        FeCovType::Hc1,
        FeCovType::Hc3,
        FeCovType::Cluster { groups: None },
        FeCovType::Dk {
            bandwidth: None,
            time: TimeKeys::lexicographic(time_labels()),
        },
    ] {
        let fitted = fit(one_way_input(), FeEffects::OneWay, cov).unwrap();
        assert_close(
            fitted.weak_instrument_f_statistics()[0].1,
            f,
            1e-12,
            "partial F",
        );
        assert_close(
            fitted.cragg_donald().unwrap().statistic,
            f,
            1e-12,
            "Cragg-Donald",
        );
    }
}

/// Classicalのrk Wald Fと有効Fは、Cragg-Donaldに一致する（内生変数1個）。
#[test]
fn classical_robust_statistics_equal_cragg_donald_for_one_endogenous() {
    let fitted = fit(one_way_input(), FeEffects::OneWay, FeCovType::Classical).unwrap();
    let cd = fitted.cragg_donald().unwrap().statistic;
    assert_close(
        fitted.kleibergen_paap().unwrap().rk_wald_f.unwrap(),
        cd,
        1e-10,
        "rk Wald F",
    );
    assert_close(
        fitted.effective_f(0.05).unwrap().statistic,
        cd,
        1e-10,
        "eff F",
    );
}

#[test]
fn rejects_invalid_effective_f_requests() {
    let two_endog = input_spec(&all_rows(), &[entity_labels()], 1, 2, 3);
    let fitted = fit(two_endog, FeEffects::OneWay, FeCovType::Classical).unwrap();
    assert_eq!(
        fitted.effective_f(0.05).unwrap_err().report().code(),
        ErrorCode::EffectiveFEndogenousCount
    );
    let no_endog = input_spec(&all_rows(), &[entity_labels()], 1, 0, 2);
    let fitted = fit(no_endog, FeEffects::OneWay, FeCovType::Classical).unwrap();
    assert!(matches!(
        fitted.effective_f(0.05),
        Err(PanelError::EffectiveFEndogenousCount { n_endog: 0 })
    ));
    let fitted = fit(one_way_input(), FeEffects::OneWay, FeCovType::Classical).unwrap();
    for alpha in [0.0, 1.0, -0.1, f64::NAN] {
        assert!(matches!(
            fitted.effective_f(alpha),
            Err(PanelError::AlphaOutOfRange { .. })
        ));
    }
}

#[test]
fn diagnostics_are_undefined_without_endogenous_variables() {
    let input = input_spec(&all_rows(), &[entity_labels()], 1, 0, 2);
    let fitted = fit(input, FeEffects::OneWay, FeCovType::Classical).unwrap();
    assert!(fitted.weak_instrument_f_statistics().is_empty());
    assert_eq!(fitted.weak_instrument_f_df(), None);
    for err in [
        fitted.cragg_donald().map(|_| ()).unwrap_err(),
        fitted.kleibergen_paap().map(|_| ()).unwrap_err(),
    ] {
        assert_eq!(
            err.report().code(),
            ErrorCode::WeakInstrumentStatisticUndefined
        );
    }
}

/// 時点数がスコアの次元`d = 除外操作変数の数 − 内生変数の数 + 1`以下のDKは、統計量がデータに依らない値になる
/// （`T = d`でLMがちょうど`T`）ので、未定義にする。部分F・Cragg-Donald（等分散）は計算できる。
#[test]
fn dk_with_at_most_as_many_periods_as_score_dimensions_is_undefined() {
    // 時点3つ（`r % 4 != 3`の行）。内生変数1個・操作変数3個で`d = 3 = T`。
    let rows: Vec<usize> = all_rows().into_iter().filter(|r| r % 4 != 3).collect();
    let time: Vec<String> = rows.iter().map(|&r| time_labels()[r].clone()).collect();
    for bandwidth in [Some(0), Some(1), None] {
        let cov = FeCovType::Dk {
            bandwidth,
            time: TimeKeys::lexicographic(time.clone()),
        };
        let input = input_spec(&rows, &[entity_labels()], 1, 1, 3);
        let fitted = fit(input, FeEffects::OneWay, cov).unwrap();
        assert!(fitted.cragg_donald().is_ok());
        assert!(!fitted.weak_instrument_f_statistics().is_empty());
        // LMは`T ≤ d`の検査で、Wald Fはスコアの和が0でランク落ち（共分散が特異）で、未定義。両方が未定義なので
        // `kleibergen_paap`は全体が未定義。有効Fも`Ŵ2`のランク落ちで未定義。
        for err in [
            fitted.kleibergen_paap().map(|_| ()).unwrap_err(),
            fitted.effective_f(0.05).map(|_| ()).unwrap_err(),
        ] {
            assert_eq!(
                err.report().code(),
                ErrorCode::WeakInstrumentStatisticUndefined,
                "bandwidth {bandwidth:?}"
            );
        }
    }
    // 時点が4つなら（`T > d`）LMは計算できる。
    let cov = FeCovType::Dk {
        bandwidth: Some(0),
        time: TimeKeys::lexicographic(time_labels()),
    };
    let fitted = fit(
        input_spec(&all_rows(), &[entity_labels()], 1, 1, 3),
        FeEffects::OneWay,
        cov,
    )
    .unwrap();
    assert!(fitted.kleibergen_paap().unwrap().rk_lm.is_some());
}

/// 既定のバンド幅（`None`）は、`fit`が解決した値が診断に渡る。バンド幅を変えると統計量も変わる。
#[test]
fn dk_diagnostics_use_the_bandwidth_that_fit_resolved() {
    let dk = |bandwidth| FeCovType::Dk {
        bandwidth,
        time: TimeKeys::lexicographic(time_labels()),
    };
    let make = || input_spec(&all_rows(), &[entity_labels()], 1, 1, 2);
    let automatic = fit(make(), FeEffects::OneWay, dk(None)).unwrap();
    let used = automatic.dk_bandwidth_used().unwrap();
    let explicit = fit(make(), FeEffects::OneWay, dk(Some(used as i64))).unwrap();
    let (a, b) = (
        automatic.kleibergen_paap().unwrap(),
        explicit.kleibergen_paap().unwrap(),
    );
    assert_eq!(a, b);
    assert_eq!(
        automatic.effective_f(0.05).unwrap(),
        explicit.effective_f(0.05).unwrap()
    );
    let other = fit(make(), FeEffects::OneWay, dk(Some(used as i64 + 1))).unwrap();
    assert_ne!(
        other.kleibergen_paap().unwrap().rk_lm,
        a.rk_lm,
        "a different bandwidth must change the statistic"
    );
}

/// 列を`group`内で平均を引く。
fn demean(column: &[f64], group: &GroupCodes) -> Vec<f64> {
    let mut sums = vec![0.0; group.n_groups()];
    for (&g, v) in group.codes().iter().zip(column) {
        sums[g] += v;
    }
    let counts = group.counts();
    group
        .codes()
        .iter()
        .zip(column)
        .map(|(&g, v)| v - sums[g] / counts[g] as f64)
        .collect()
}

/// `absorbed`で平均を引き、`dummy_dims`の次元のダミー（各次元の最初の水準を落とす）を回帰子に足し、
/// 切片を入れたIV本体の入力。`dummy_dims`のダミーも`absorbed`内で平均を引く。クラスターの
/// 「入れ子の次元を除く」`K`を、入れ子の次元は平均を引いて切片1つ、入れ子でない次元はダミーとして数える形。
fn partly_absorbed_iv(
    input: &FeIvInput,
    absorbed: &GroupCodes,
    dummy_dims: &[GroupCodes],
) -> IvInput {
    let dm = |c: &[f64]| demean(c, absorbed);
    let mut exog: Vec<Vec<f64>> = input.x_exog().iter().map(|c| dm(c)).collect();
    for dimension in dummy_dims {
        let dummy_columns = dummies(std::slice::from_ref(dimension));
        // `dummies`は最初の次元に全水準を返す。切片があるので最初の水準を落とす。
        exog.extend(dummy_columns.iter().skip(1).map(|c| dm(c)));
    }
    let names: Vec<String> = (0..exog.len()).map(|j| format!("x{j}")).collect();
    let endog: Vec<Vec<f64>> = input.x_endog().iter().map(|c| dm(c)).collect();
    let instruments: Vec<Vec<f64>> = input.instruments().iter().map(|c| dm(c)).collect();
    IvInput::from_columns(
        &dm(input.y()),
        &exog,
        names,
        &endog,
        input.x_endog_names().to_vec(),
        &instruments,
        input.instrument_names().to_vec(),
        true,
        "y".to_string(),
    )
    .unwrap()
}

/// 2-wayで、クラスターに入れ子の次元（個体クラスターなら個体、時点クラスターなら時点）は平均を引いて切片1つ、
/// 入れ子でない次元はダミーとして数える。`fe_cluster_k_correction`の「部分的に入れ子」の分岐。
#[test]
fn cluster_in_two_way_counts_only_the_dimensions_that_are_not_nested() {
    for (rows, name) in [(all_rows(), "balanced"), (unbalanced_rows(), "unbalanced")] {
        let pick = |labels: Vec<String>| -> Vec<String> {
            rows.iter().map(|&r| labels[r].clone()).collect()
        };
        let (entity, time) = (
            GroupCodes::from_labels(&pick(entity_labels())),
            GroupCodes::from_labels(&pick(time_labels())),
        );
        for spec in [(1, 1, 2), (1, 2, 3)] {
            let make = || {
                input_spec(
                    &rows,
                    &[entity_labels(), time_labels()],
                    spec.0,
                    spec.1,
                    spec.2,
                )
            };
            // (クラスター, 平均を引く次元, ダミーにする次元)
            let cases = [
                (
                    FeCovType::Cluster { groups: None },
                    &entity,
                    std::slice::from_ref(&time),
                    "entity cluster",
                ),
                (
                    FeCovType::Cluster {
                        groups: Some(time.clone()),
                    },
                    &time,
                    std::slice::from_ref(&entity),
                    "time cluster",
                ),
            ];
            for (cov_type, absorbed, dummy_dims, label) in cases {
                let fitted = fit(make(), FeEffects::TwoWay, cov_type.clone()).unwrap();
                let iv = partly_absorbed_iv(&make(), absorbed, dummy_dims);
                let iv_cov = CovType::Cluster {
                    groups: Some(match &cov_type {
                        FeCovType::Cluster { groups: Some(g) } => g.clone(),
                        _ => entity.clone(),
                    }),
                };
                let (fe_kp, kp) = (
                    fitted.kleibergen_paap().unwrap(),
                    compute_kleibergen_paap(&iv, &iv_cov).unwrap(),
                );
                let tag = format!("{name} {label} spec {spec:?}");
                assert_close(fe_kp.rk_wald_f.unwrap(), kp.rk_wald_f.unwrap(), 1e-8, &tag);
                assert_close(
                    fe_kp.rk_lm.unwrap().statistic,
                    kp.rk_lm.unwrap().statistic,
                    1e-8,
                    &tag,
                );
                if spec.1 == 1 {
                    assert_close(
                        fitted.effective_f(0.05).unwrap().statistic,
                        compute_effective_f(&iv, &iv_cov, 0.05).unwrap().statistic,
                        1e-8,
                        &tag,
                    );
                }
            }
        }
    }
}
