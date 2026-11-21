//! `fit`レベルのproperty-basedテスト（固定シナリオとは別に、加法シフト・スケール・行の並べ替えの
//! 不変性などを、ランダムなパネルで検証する）。

use crate::linear::ols::CovType;
use crate::linear::ols::OlsEstimator;
use crate::linear::ols::OlsInput;
use crate::panel::fe::FeEstimator;
use crate::panel::fe::input::FeCovType;
use crate::panel::fe::input::FeEffects;
use crate::panel::fe::input::FeInput;
use proptest::collection;
use proptest::prelude::*;

/// `fit`レベルのproperty-basedテスト用のランダムパネル。1-way（アンバランスを含む）と
/// 2-way（バランスのみ、`fe-spec.md`3.1節）を`effects`で切り替える。行は
/// entity外側・time内側の順で並ぶ（並べ替えは`keys`で別途行う）。
#[derive(Debug, Clone)]
struct FeCase {
    effects: FeEffects,
    entity_idx: Vec<usize>,
    time_idx: Vec<usize>,
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
    /// 行の並べ替え用の乱数キー（長さ`n`）。
    keys: Vec<u64>,
    /// entity/timeごとの加法シフト（`y`に足して結果が不変であることの検証用）。
    entity_shift: Vec<f64>,
    time_shift: Vec<f64>,
}

const MAX_PERIODS: usize = 6;

fn fe_case_strategy() -> impl Strategy<Value = FeCase> {
    (any::<bool>(), 4..=6usize, 3..=MAX_PERIODS, 1..=2usize)
        .prop_flat_map(|(two_way, n_entities, n_periods, k)| {
            let sizes = if two_way {
                Just(vec![n_periods; n_entities]).boxed()
            } else {
                collection::vec(2..=MAX_PERIODS, n_entities).boxed()
            };
            (Just(two_way), Just(k), sizes)
        })
        .prop_flat_map(|(two_way, k, sizes)| {
            let n: usize = sizes.iter().sum();
            let n_entities = sizes.len();
            (
                Just(two_way),
                Just(sizes),
                collection::vec(collection::vec(-10.0f64..10.0, n), k),
                collection::vec(-5.0f64..5.0, n),
                collection::vec(any::<u64>(), n),
                collection::vec(-50.0f64..50.0, n_entities),
                collection::vec(-50.0f64..50.0, MAX_PERIODS),
            )
        })
        .prop_map(
            |(two_way, sizes, x, noise, keys, entity_shift, time_shift)| {
                let mut entity_idx = Vec::new();
                let mut time_idx = Vec::new();
                for (i, &size) in sizes.iter().enumerate() {
                    for t in 0..size {
                        entity_idx.push(i);
                        time_idx.push(t);
                    }
                }
                let y: Vec<f64> = (0..noise.len())
                    .map(|r| x.iter().map(|c| c[r]).sum::<f64>() + noise[r])
                    .collect();
                FeCase {
                    effects: if two_way {
                        FeEffects::TwoWay
                    } else {
                        FeEffects::OneWay
                    },
                    entity_idx,
                    time_idx,
                    y,
                    x,
                    keys,
                    entity_shift,
                    time_shift,
                }
            },
        )
}

fn fe_cov_strategy() -> impl Strategy<Value = FeCovType> {
    prop_oneof![
        Just(FeCovType::Classical),
        Just(FeCovType::Hc1),
        Just(FeCovType::Hc2),
        Just(FeCovType::Hc3),
        Just(FeCovType::Cluster { groups: None }),
    ]
}

fn labels(prefix: &str, idx: &[usize]) -> Vec<String> {
    idx.iter().map(|i| format!("{prefix}{i}")).collect()
}

/// 傾き係数と標準誤差（`x`の列順）。推定に失敗したら`None`。
fn fit_slopes(
    case: &FeCase,
    y: &[f64],
    x: &[Vec<f64>],
    entity: &[String],
    time: &[String],
    cov: FeCovType,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let names: Vec<String> = (0..x.len()).map(|j| format!("x{j}")).collect();
    let input = FeInput::from_columns(y, x, names, entity, Some(time), "y".to_string()).ok()?;
    let est = FeEstimator::fit(input, case.effects, cov, 0.95).ok()?;
    let k = x.len();
    let params = (0..k).map(|j| *est.params().get(j, 0)).collect();
    let se = (0..k).map(|j| *est.std_errors().get(j, 0)).collect();
    Some((params, se))
}

fn fit_case(case: &FeCase, cov: FeCovType) -> Option<(Vec<f64>, Vec<f64>)> {
    fit_slopes(
        case,
        &case.y,
        &case.x,
        &labels("e", &case.entity_idx),
        &labels("t", &case.time_idx),
        cov,
    )
}

/// 固定効果推定の数値誤差（within変換・QR）を考慮し、固定フィクスチャ比較より緩めた
/// 相対誤差（`ols/fit/tests/proptests.rs`のproptestと同じ方針）。
fn assert_approx_eq(actual: f64, expected: f64, msg: &str) {
    let tol = 1e-6 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tol,
        "{msg}: actual={actual}, expected={expected}, tol={tol}"
    );
}

fn assert_all_approx_eq(actual: &[f64], expected: &[f64], msg: &str) {
    assert_eq!(actual.len(), expected.len());
    for (j, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_approx_eq(*a, *e, &format!("{msg}[{j}]"));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// yにentity定数（2-wayはtime定数も）を加えても、傾き・SEは変わらない
    /// （固定効果が吸収するため）。
    #[test]
    fn slopes_and_se_are_invariant_to_additive_fixed_effects_in_y(
        case in fe_case_strategy(),
        cov in fe_cov_strategy(),
    ) {
        let base = fit_case(&case, cov.clone());
        prop_assume!(base.is_some());
        let (params, se) = base.unwrap();

        let y_shifted: Vec<f64> = (0..case.y.len())
            .map(|r| {
                let time_part = if case.effects == FeEffects::TwoWay {
                    case.time_shift[case.time_idx[r]]
                } else {
                    0.0
                };
                case.y[r] + case.entity_shift[case.entity_idx[r]] + time_part
            })
            .collect();
        let shifted = fit_slopes(
            &case,
            &y_shifted,
            &case.x,
            &labels("e", &case.entity_idx),
            &labels("t", &case.time_idx),
            cov,
        );
        prop_assume!(shifted.is_some());
        let (params2, se2) = shifted.unwrap();

        assert_all_approx_eq(&params2, &params, "params");
        assert_all_approx_eq(&se2, &se, "se");
    }

    /// 行の並べ替えとentityラベルの付け替え（順序を反転した別名）で結果は変わらない。
    #[test]
    fn results_are_invariant_to_row_order_and_entity_relabeling(
        case in fe_case_strategy(),
        cov in fe_cov_strategy(),
    ) {
        let base = fit_case(&case, cov.clone());
        prop_assume!(base.is_some());
        let (params, se) = base.unwrap();

        let n = case.y.len();
        let n_entities = case.entity_shift.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&r| case.keys[r]);
        let y: Vec<f64> = order.iter().map(|&r| case.y[r]).collect();
        let x: Vec<Vec<f64>> = case
            .x
            .iter()
            .map(|c| order.iter().map(|&r| c[r]).collect())
            .collect();
        let entity: Vec<String> = order
            .iter()
            .map(|&r| format!("z{}", n_entities - 1 - case.entity_idx[r]))
            .collect();
        let time: Vec<String> = order
            .iter()
            .map(|&r| format!("t{}", case.time_idx[r]))
            .collect();

        let permuted = fit_slopes(&case, &y, &x, &entity, &time, cov);
        prop_assume!(permuted.is_some());
        let (params2, se2) = permuted.unwrap();

        assert_all_approx_eq(&params2, &params, "params");
        assert_all_approx_eq(&se2, &se, "se");
    }

    /// yをc倍すると傾き・標準誤差はそれぞれc倍・|c|倍になる。
    #[test]
    fn slopes_and_se_scale_with_y(
        case in fe_case_strategy(),
        cov in fe_cov_strategy(),
        c in prop_oneof![-10.0f64..-0.1, 0.1f64..10.0],
    ) {
        let base = fit_case(&case, cov.clone());
        prop_assume!(base.is_some());
        let (params, se) = base.unwrap();

        let y_scaled: Vec<f64> = case.y.iter().map(|v| v * c).collect();
        let scaled = fit_slopes(
            &case,
            &y_scaled,
            &case.x,
            &labels("e", &case.entity_idx),
            &labels("t", &case.time_idx),
            cov,
        );
        prop_assume!(scaled.is_some());
        let (params2, se2) = scaled.unwrap();

        let expected_params: Vec<f64> = params.iter().map(|p| p * c).collect();
        let expected_se: Vec<f64> = se.iter().map(|s| s * c.abs()).collect();
        assert_all_approx_eq(&params2, &expected_params, "params");
        assert_all_approx_eq(&se2, &expected_se, "se");
    }

    /// LSDV（entityダミー、2-wayはtimeダミーも加えた定数項付きOLS）と傾き・Classical SEが
    /// 一致する。LSDV側のOLS自由度`n-(k+N+T-1)`はFEのパネル自由度調整（`neffects`）と
    /// 一致するため、SEも同じ値になる。
    #[test]
    fn slopes_and_classical_se_match_lsdv_oracle(case in fe_case_strategy()) {
        let fe = fit_case(&case, FeCovType::Classical);
        prop_assume!(fe.is_some());
        let (params, se) = fe.unwrap();

        let k = case.x.len();
        let n = case.y.len();
        let n_entities = case.entity_shift.len();
        let mut x = case.x.clone();
        for i in 1..n_entities {
            x.push((0..n).map(|r| f64::from(case.entity_idx[r] == i)).collect());
        }
        if case.effects == FeEffects::TwoWay {
            let n_periods = *case.time_idx.iter().max().unwrap() + 1;
            for t in 1..n_periods {
                x.push((0..n).map(|r| f64::from(case.time_idx[r] == t)).collect());
            }
        }
        let names: Vec<String> = (0..x.len()).map(|j| format!("x{j}")).collect();
        let input = OlsInput::from_columns(&case.y, &x, names, true, "y".to_string()).unwrap();
        let ols = OlsEstimator::fit(input, CovType::Classical, 0.95);
        prop_assume!(ols.is_ok());
        let ols = ols.unwrap();

        // 定数項が先頭に来るため、傾きは添字1..=k。
        let ols_params: Vec<f64> = (1..=k).map(|j| *ols.params().get(j, 0)).collect();
        let ols_se: Vec<f64> = (1..=k).map(|j| *ols.std_errors().get(j, 0)).collect();
        assert_all_approx_eq(&params, &ols_params, "params");
        assert_all_approx_eq(&se, &ols_se, "se");
    }
}
