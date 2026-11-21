//! property-basedテスト。固定シナリオでは`σ_u²`・`T_i`・`cov_type`の組み合わせが
//! 限られるため、ランダムな不均衡パネルでREの不変条件（θの値域・`σ_u²=0`でのpooled OLS
//! への退化・行順序/ラベルへの不変性）を検証する。

use crate::linear::ols::CovType;
use crate::linear::ols::OlsEstimator;
use crate::linear::ols::OlsInput;
use crate::panel::common::PanelError;
use crate::panel::common::TimeKeys;
use crate::panel::re::ReCovType;
use crate::panel::re::ReEffects;
use crate::panel::re::ReEstimator;
use crate::panel::re::ReInput;
use crate::panel::re::VarianceMethod;
use crate::panel::re::transform::quasi_demean_transform;
use crate::panel::re::variance::VarianceComponents;
use crate::panel::re::variance::entity_means;
use crate::panel::re::variance::swamy_arora_variance_components;
use crate::shared::group_codes::GroupCodes;
use proptest::collection;
use proptest::prelude::*;

const MAX_T: usize = 6;

#[derive(Debug, Clone)]
struct ReCase {
    entity_idx: Vec<usize>,
    n_entities: usize,
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
    /// 行の並べ替え用の乱数キー（長さ`n`）。
    keys: Vec<u64>,
}

/// 不均衡パネル（entityごとに`T_i`が2..=6）。`has_effect=false`の場合はentity効果を
/// 持たないDGPになり、`σ_u²`が0に切り詰められる（Swamy-Aroraの`max(0)`）ケースが
/// 高頻度で現れる。
fn re_case_strategy() -> impl Strategy<Value = ReCase> {
    (6..=9usize, 1..=2usize, any::<bool>())
        .prop_flat_map(|(n_entities, k, has_effect)| {
            (
                Just(n_entities),
                Just(k),
                Just(has_effect),
                collection::vec(2..=MAX_T, n_entities),
            )
        })
        .prop_flat_map(|(n_entities, k, has_effect, sizes)| {
            let n: usize = sizes.iter().sum();
            (
                Just(n_entities),
                Just(has_effect),
                Just(sizes),
                collection::vec(collection::vec(-10.0f64..10.0, n), k),
                collection::vec(-5.0f64..5.0, n),
                collection::vec(-20.0f64..20.0, n_entities),
                collection::vec(any::<u64>(), n),
            )
        })
        .prop_map(|(n_entities, has_effect, sizes, x, noise, effect, keys)| {
            let mut entity_idx = Vec::new();
            for (i, &size) in sizes.iter().enumerate() {
                entity_idx.extend(std::iter::repeat_n(i, size));
            }
            let y: Vec<f64> = (0..noise.len())
                .map(|r| {
                    let u = if has_effect {
                        effect[entity_idx[r]]
                    } else {
                        0.0
                    };
                    x.iter().map(|c| c[r]).sum::<f64>() + u + noise[r]
                })
                .collect();
            ReCase {
                entity_idx,
                n_entities,
                y,
                x,
                keys,
            }
        })
}

/// バランスパネル（`n_entities`×`n_periods`、行はentity優先の並び）。
/// `(n_entities, n_periods, y, x, 並べ替え用の乱数キー)`。
type TwoWayCase = (usize, usize, Vec<f64>, Vec<Vec<f64>>, Vec<u64>);

fn two_way_case_strategy() -> impl Strategy<Value = TwoWayCase> {
    (4..=7usize, 3..=5usize, 1..=2usize)
        .prop_flat_map(|(n_entities, n_periods, k)| {
            let n = n_entities * n_periods;
            (
                Just(n_entities),
                Just(n_periods),
                collection::vec(-5.0f64..5.0, n),
                collection::vec(collection::vec(-10.0f64..10.0, n), k),
                collection::vec(any::<u64>(), n),
            )
        })
        .prop_map(|(n_entities, n_periods, noise, x, keys)| {
            let y: Vec<f64> = (0..noise.len())
                .map(|r| {
                    let effect = (r / n_periods) as f64 - (r % n_periods) as f64;
                    x.iter().map(|c| c[r]).sum::<f64>() + effect + noise[r]
                })
                .collect();
            (n_entities, n_periods, y, x, keys)
        })
}

fn re_cov_strategy() -> impl Strategy<Value = ReCovType> {
    prop_oneof![
        Just(ReCovType::Classical),
        Just(ReCovType::Hc1),
        Just(ReCovType::Hc2),
        Just(ReCovType::Hc3),
        Just(ReCovType::Cluster { groups: None }),
    ]
}

fn entity_labels(idx: &[usize]) -> Vec<String> {
    idx.iter().map(|i| format!("e{i}")).collect()
}

fn re_input(y: &[f64], x: &[Vec<f64>], entity: &[String]) -> ReInput {
    let names: Vec<String> = (0..x.len()).map(|j| format!("x{j}")).collect();
    ReInput::from_columns(y, x, names, entity, None, "y".to_string()).unwrap()
}

/// 固定フィクスチャ比較より緩めた相対誤差（`ols/fit/tests/proptests.rs`/`fe/fit/tests/proptests.rs`のproptestと同じ方針）。
fn assert_approx_eq(actual: f64, expected: f64, msg: &str) {
    let tol = 1e-6 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tol,
        "{msg}: actual={actual}, expected={expected}, tol={tol}"
    );
}

/// 係数（先頭が切片）と標準誤差を並べて取り出す。
fn params_and_se(est: &ReEstimator) -> (Vec<f64>, Vec<f64>) {
    let n = est.estimator().params().nrows();
    let params = (0..n)
        .map(|j| *est.estimator().params().get(j, 0))
        .collect();
    let se = (0..n).map(|j| *est.std_errors().get(j, 0)).collect();
    (params, se)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// 説明変数の個体平均の列を加えても（相関ランダム効果、Mundlak型）、between回帰ではその列が
    /// 元の列と同じ列になって落ちるため、分散成分は変わらず、最終回帰・ハウスマン検定も
    /// 失敗しない（ハウスマン検定の自由度は別名になった`x̃`の分だけ減る）。
    #[test]
    fn adding_an_entity_mean_column_leaves_the_variance_components_unchanged(
        case in re_case_strategy(),
    ) {
        let entity = entity_labels(&case.entity_idx);
        let codes = GroupCodes::from_labels(&entity);
        let base = re_input(&case.y, &case.x, &entity);
        let mean = entity_means(base.y(), base.x(), &codes);
        let bar: Vec<f64> = (0..base.nobs())
            .map(|i| mean.x[0][codes.codes()[i]])
            .collect();
        let mut x = case.x.clone();
        x.push(bar);
        let extended = re_input(&case.y, &x, &entity);

        let plain = swamy_arora_variance_components(&base, ReEffects::OneWay, 0.95);
        prop_assume!(plain.is_ok());
        let plain = plain.unwrap();
        let with_bar = swamy_arora_variance_components(&extended, ReEffects::OneWay, 0.95)
            .unwrap();
        assert_approx_eq(with_bar.sigma2_eps, plain.sigma2_eps, "sigma2_eps");
        assert_approx_eq(with_bar.sigma2_u, plain.sigma2_u, "sigma2_u");

        let fit = ReEstimator::fit(extended, ReEffects::OneWay, ReCovType::Classical, 0.95);
        prop_assume!(plain.sigma2_eps > 0.0);
        let fit = fit.unwrap();
        let base_fit = ReEstimator::fit(base, ReEffects::OneWay, ReCovType::Classical, 0.95)
            .unwrap();
        prop_assert_eq!(
            fit.hausman_df(),
            base_fit.hausman_df().and_then(|df| df.checked_sub(1)).filter(|&df| df > 0)
        );
    }

    /// Swamy-Arora分散成分から実際に組み立てたθも`[0, 1)`に収まる。
    #[test]
    fn fitted_theta_is_within_unit_interval(case in re_case_strategy()) {
        let entity = entity_labels(&case.entity_idx);
        let input = re_input(&case.y, &case.x, &entity);
        let vc = swamy_arora_variance_components(&input,ReEffects::OneWay, 0.95);
        prop_assume!(vc.is_ok());
        let VarianceComponents {
            sigma2_eps,
            sigma2_u,
            ..
        } = vc.unwrap();
        prop_assume!(sigma2_eps > 0.0);

        let (theta, _, _) = quasi_demean_transform(&input, sigma2_eps, sigma2_u);

        for (id, t) in theta.iter().enumerate() {
            prop_assert!((0.0..1.0).contains(t), "theta[{id}]={t}");
        }
    }

    /// `σ_u²`が0に切り詰められたときREはpooled OLS（定数項あり）と一致する
    /// （`θ_i=0`で準偏差変換が恒等になるため、係数もClassical SEも同じ）。
    #[test]
    fn matches_pooled_ols_when_sigma2_u_is_zero(case in re_case_strategy()) {
        let entity = entity_labels(&case.entity_idx);
        let vc = swamy_arora_variance_components(&re_input(&case.y, &case.x, &entity),ReEffects::OneWay, 0.95);
        prop_assume!(vc.is_ok());
        prop_assume!(vc.unwrap().sigma2_u == 0.0);

        let re = ReEstimator::fit(
            re_input(&case.y, &case.x, &entity),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,);
        prop_assume!(re.is_ok());
        let (params, se) = params_and_se(&re.unwrap());

        let names: Vec<String> = (0..case.x.len()).map(|j| format!("x{j}")).collect();
        let ols_input =
            OlsInput::from_columns(&case.y, &case.x, names, true, "y".to_string()).unwrap();
        let ols = OlsEstimator::fit(ols_input, CovType::Classical, 0.95);
        prop_assume!(ols.is_ok());
        let ols = ols.unwrap();

        for j in 0..params.len() {
            assert_approx_eq(params[j], *ols.params().get(j, 0), &format!("param[{j}]"));
            assert_approx_eq(se[j], *ols.std_errors().get(j, 0), &format!("se[{j}]"));
        }
    }

    /// 行の並べ替えとentityラベルの付け替えで、係数・標準誤差・ハウスマン統計量は
    /// 変わらない。
    #[test]
    fn results_are_invariant_to_row_order_and_entity_relabeling(
        case in re_case_strategy(),
        cov in re_cov_strategy(),
    ) {
        let entity = entity_labels(&case.entity_idx);
        let base = ReEstimator::fit(re_input(&case.y, &case.x, &entity),ReEffects::OneWay, cov.clone(), 0.95);
        prop_assume!(base.is_ok());
        let base = base.unwrap();
        let (params, se) = params_and_se(&base);

        let n = case.y.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&r| case.keys[r]);
        let y: Vec<f64> = order.iter().map(|&r| case.y[r]).collect();
        let x: Vec<Vec<f64>> = case
            .x
            .iter()
            .map(|c| order.iter().map(|&r| c[r]).collect())
            .collect();
        let entity2: Vec<String> = order
            .iter()
            .map(|&r| format!("z{}", case.n_entities - 1 - case.entity_idx[r]))
            .collect();

        let permuted = ReEstimator::fit(re_input(&y, &x, &entity2),ReEffects::OneWay, cov, 0.95);
        prop_assume!(permuted.is_ok());
        let permuted = permuted.unwrap();
        let (params2, se2) = params_and_se(&permuted);

        for j in 0..params.len() {
            assert_approx_eq(params2[j], params[j], &format!("param[{j}]"));
            assert_approx_eq(se2[j], se[j], &format!("se[{j}]"));
        }
        match (base.hausman_statistic(), permuted.hausman_statistic()) {
            (Some(a), Some(b)) => assert_approx_eq(b, a, "hausman_statistic"),
            (None, None) => {}
            (a, b) => prop_assert!(false, "hausman presence differs: {a:?} vs {b:?}"),
        }
    }

    /// 2-wayのバランスパネル: 行の並べ替えとエンティティ・時点のラベルの付け替えで
    /// 係数・標準誤差が変わらず、`y`を`c`倍すると係数・標準誤差が`|c|`倍になる
    /// （`σ²`は`c²`倍、θは不変のため）。
    #[test]
    fn two_way_results_are_invariant_to_relabeling_and_equivariant_to_y_scale(
        case in two_way_case_strategy(),
        scale in 0.5f64..4.0,
    ) {
        let (n_entities, n_periods, y, x, keys) = case;
        let n = y.len();
        let entity: Vec<String> = (0..n).map(|r| format!("e{}", r / n_periods)).collect();
        let time: Vec<String> = (0..n).map(|r| format!("t{}", r % n_periods)).collect();
        let build = |y: &[f64], x: &[Vec<f64>], entity: &[String], time: &[String]| {
            let names: Vec<String> = (0..x.len()).map(|j| format!("x{j}")).collect();
            ReInput::from_columns(y, x, names, entity, Some(time), "y".to_string())
                .and_then(|input| {
                    ReEstimator::fit(input, ReEffects::TwoWay, ReCovType::Classical, 0.95)
                })
        };
        let base = build(&y, &x, &entity, &time);
        prop_assume!(base.is_ok());
        let (params, se) = params_and_se(&base.unwrap());

        // 行の並べ替え＋エンティティ・時点ラベルの付け替え。
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&r| keys[r]);
        let y2: Vec<f64> = order.iter().map(|&r| y[r]).collect();
        let x2: Vec<Vec<f64>> =
            x.iter().map(|c| order.iter().map(|&r| c[r]).collect()).collect();
        let entity2: Vec<String> = order
            .iter()
            .map(|&r| format!("z{}", n_entities - 1 - r / n_periods))
            .collect();
        let time2: Vec<String> = order
            .iter()
            .map(|&r| format!("u{}", n_periods - 1 - r % n_periods))
            .collect();
        let permuted = build(&y2, &x2, &entity2, &time2);
        prop_assume!(permuted.is_ok());
        let (params2, se2) = params_and_se(&permuted.unwrap());
        for j in 0..params.len() {
            assert_approx_eq(params2[j], params[j], &format!("param[{j}]"));
            assert_approx_eq(se2[j], se[j], &format!("se[{j}]"));
        }

        // `y`のスケール同変性。
        let y3: Vec<f64> = y.iter().map(|v| v * scale).collect();
        let scaled = build(&y3, &x, &entity, &time);
        prop_assume!(scaled.is_ok());
        let (params3, se3) = params_and_se(&scaled.unwrap());
        for j in 0..params.len() {
            assert_approx_eq(params3[j], params[j] * scale, &format!("scaled param[{j}]"));
            assert_approx_eq(se3[j], se[j] * scale, &format!("scaled se[{j}]"));
        }
    }

    /// エンティティ内で一定の列`z`（時間不変）を足しても、内部FEの`σ_ε²`は変わらない
    /// （within変動のある列だけがσ_ε²を決める）。`time_varying`は元の列を指したまま。
    #[test]
    fn time_invariant_column_leaves_sigma2_eps_unchanged(case in re_case_strategy()) {
        let entity = entity_labels(&case.entity_idx);
        let base = swamy_arora_variance_components(
            &re_input(&case.y, &case.x, &entity),
            ReEffects::OneWay,
            0.95,);
        prop_assume!(base.is_ok());
        let base = base.unwrap();

        let z: Vec<f64> = case
            .entity_idx
            .iter()
            .map(|&i| ((i * 37) % 11) as f64 - 5.0)
            .collect();
        let mut x = vec![z];
        x.extend(case.x.iter().cloned());
        let with_z = swamy_arora_variance_components(&re_input(&case.y, &x, &entity),ReEffects::OneWay, 0.95);
        prop_assume!(with_z.is_ok());
        let with_z = with_z.unwrap();

        prop_assert_eq!(with_z.sigma2_eps, base.sigma2_eps);
        let shifted: Vec<usize> = base.time_varying.iter().map(|j| j + 1).collect();
        prop_assert_eq!(with_z.time_varying, shifted);
    }

    /// 時間不変列`z`に定数`c`を足しても、`z`と他の列の係数は変わらず、
    /// 切片だけが`-β_z·c`動く（定数は切片に吸収される）。
    #[test]
    fn shifting_time_invariant_column_only_moves_the_intercept(
        case in re_case_strategy(),
        c in -5.0f64..5.0,
    ) {
        let entity = entity_labels(&case.entity_idx);
        let z: Vec<f64> = case
            .entity_idx
            .iter()
            .map(|&i| ((i * 37) % 11) as f64 - 5.0)
            .collect();
        let build = |z: Vec<f64>| {
            let mut x = vec![z];
            x.extend(case.x.iter().cloned());
            ReEstimator::fit(re_input(&case.y, &x, &entity),ReEffects::OneWay, ReCovType::Classical, 0.95)
        };
        let base = build(z.clone());
        prop_assume!(base.is_ok());
        let shifted = build(z.iter().map(|v| v + c).collect());
        prop_assume!(shifted.is_ok());
        let (base, shifted) = (base.unwrap(), shifted.unwrap());
        let (p0, se0) = params_and_se(&base);
        let (p1, se1) = params_and_se(&shifted);

        assert_approx_eq(p1[0], p0[0] - p0[1] * c, "const");
        for j in 1..p0.len() {
            assert_approx_eq(p1[j], p0[j], &format!("param[{j}]"));
            assert_approx_eq(se1[j], se0[j], &format!("se[{j}]"));
        }
    }
}

/// 一次元でしか変わらない列（`xi`: 個体間のみ、`xt`: 時点間のみ）を含む2-wayのバランス
/// パネル。`has_entity_effect`・`has_time_effect`が偽のDGPでは、対応する分散成分が0に
/// 切り詰められるケースが高頻度で現れる。エンティティ数・時点数は、ハウスマンの補助回帰
/// （傾き最大6個）がクラスター（`G = N`）・Driscoll-Kraay（`T`）のどちらでも成立する
/// 7以上にする。
#[derive(Debug, Clone)]
struct OneDimCase {
    n_entities: usize,
    n_periods: usize,
    y: Vec<f64>,
    /// 両次元で変動する列、続いて`xi`・`xt`（使う場合）。
    x: Vec<Vec<f64>>,
    keys: Vec<u64>,
}

fn one_dim_case_strategy() -> impl Strategy<Value = OneDimCase> {
    (
        7..=9usize,
        7..=9usize,
        1..=2usize,
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
    )
        .prop_flat_map(
            |(n_entities, n_periods, k_both, has_xi, has_xt, has_ent, has_time)| {
                let n = n_entities * n_periods;
                (
                    Just((n_entities, n_periods, k_both, has_xi, has_xt)),
                    Just((has_ent, has_time)),
                    collection::vec(-5.0f64..5.0, n),
                    collection::vec(collection::vec(-10.0f64..10.0, n), k_both),
                    collection::vec(-10.0f64..10.0, n_entities),
                    collection::vec(-10.0f64..10.0, n_periods),
                    collection::vec(-15.0f64..15.0, n_entities),
                    collection::vec(-15.0f64..15.0, n_periods),
                    collection::vec(any::<u64>(), n),
                )
            },
        )
        .prop_map(
            |(
                (n_entities, n_periods, _k_both, has_xi, has_xt),
                (has_ent, has_time),
                noise,
                both,
                xi_by_entity,
                xt_by_time,
                entity_effect,
                time_effect,
                keys,
            )| {
                let n = n_entities * n_periods;
                let mut x = both;
                if has_xi {
                    x.push((0..n).map(|r| xi_by_entity[r / n_periods]).collect());
                }
                if has_xt {
                    x.push((0..n).map(|r| xt_by_time[r % n_periods]).collect());
                }
                let y: Vec<f64> = (0..n)
                    .map(|r| {
                        let e = if has_ent {
                            entity_effect[r / n_periods]
                        } else {
                            0.0
                        };
                        let t = if has_time {
                            time_effect[r % n_periods]
                        } else {
                            0.0
                        };
                        x.iter().map(|c| c[r]).sum::<f64>() + e + t + noise[r]
                    })
                    .collect();
                OneDimCase {
                    n_entities,
                    n_periods,
                    y,
                    x,
                    keys,
                }
            },
        )
}

/// 2-wayの全`cov_type`（インデックス0..=5: Classical, Hc1, Hc2, Hc3, Cluster(entity),
/// Dk(時点ラベルは`time`)）。
fn two_way_cov(index: usize, time: &[String]) -> ReCovType {
    match index {
        0 => ReCovType::Classical,
        1 => ReCovType::Hc1,
        2 => ReCovType::Hc2,
        3 => ReCovType::Hc3,
        4 => ReCovType::Cluster { groups: None },
        _ => ReCovType::Dk {
            bandwidth: None,
            time: TimeKeys::lexicographic(time.to_vec()),
        },
    }
}

fn two_way_labels(n_entities: usize, n_periods: usize) -> (Vec<String>, Vec<String>) {
    let n = n_entities * n_periods;
    (
        (0..n).map(|r| format!("e{}", r / n_periods)).collect(),
        (0..n).map(|r| format!("t{}", r % n_periods)).collect(),
    )
}

fn fit_two_way(
    y: &[f64],
    x: &[Vec<f64>],
    entity: &[String],
    time: &[String],
    cov_index: usize,
) -> Result<ReEstimator, PanelError> {
    fit_two_way_with(y, x, entity, time, cov_index, VarianceMethod::SwamyArora)
}

fn fit_two_way_with(
    y: &[f64],
    x: &[Vec<f64>],
    entity: &[String],
    time: &[String],
    cov_index: usize,
    method: VarianceMethod,
) -> Result<ReEstimator, PanelError> {
    let names: Vec<String> = (0..x.len()).map(|j| format!("x{j}")).collect();
    ReInput::from_columns(y, x, names, entity, Some(time), "y".to_string()).and_then(|input| {
        ReEstimator::fit_with_variance_method(
            input,
            ReEffects::TwoWay,
            two_way_cov(cov_index, time),
            method,
            0.95,
        )
    })
}

/// 一次元でしか変わらない列を含むバランスパネルの1-way用（`xt`は時点間のみ変わる）。
#[derive(Debug, Clone)]
struct TimeOnlyCase {
    n_entities: usize,
    n_periods: usize,
    y: Vec<f64>,
    /// 両次元で変動する列、続いて時点間のみ変わる`xt`。
    x: Vec<Vec<f64>>,
    keys: Vec<u64>,
}

fn time_only_case_strategy() -> impl Strategy<Value = TimeOnlyCase> {
    (6..=9usize, 3..=5usize, 1..=2usize, any::<bool>())
        .prop_flat_map(|(n_entities, n_periods, k, has_effect)| {
            let n = n_entities * n_periods;
            (
                Just((n_entities, n_periods, has_effect)),
                collection::vec(-5.0f64..5.0, n),
                collection::vec(collection::vec(-10.0f64..10.0, n), k),
                collection::vec(-10.0f64..10.0, n_periods),
                collection::vec(-15.0f64..15.0, n_entities),
                collection::vec(any::<u64>(), n),
            )
        })
        .prop_map(
            |((n_entities, n_periods, has_effect), noise, mut x, xt, effect, keys)| {
                let n = n_entities * n_periods;
                x.push((0..n).map(|r| xt[r % n_periods]).collect());
                let y: Vec<f64> = (0..n)
                    .map(|r| {
                        let e = if has_effect {
                            effect[r / n_periods]
                        } else {
                            0.0
                        };
                        x.iter().map(|c| c[r]).sum::<f64>() + e + noise[r]
                    })
                    .collect();
                TimeOnlyCase {
                    n_entities,
                    n_periods,
                    y,
                    x,
                    keys,
                }
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// 2-wayで一次元でしか変わらない列（`xi`・`xt`）を含んでも、全`cov_type`
    /// （Classical・HC1〜HC3・cluster・Driscoll-Kraay）で、行の並べ替えとエンティティ・時点
    /// ラベルの付け替えによって係数・標準誤差・ハウスマン統計量が変わらない。
    #[test]
    fn two_way_one_dimensional_columns_are_invariant_to_relabeling_for_every_cov_type(
        case in one_dim_case_strategy(),
        cov_index in 0..=5usize,
    ) {
        let (entity, time) = two_way_labels(case.n_entities, case.n_periods);
        let base = fit_two_way(&case.y, &case.x, &entity, &time, cov_index);
        prop_assert!(base.is_ok(), "{:?}", base.as_ref().err());
        let base = base.unwrap();
        let (params, se) = params_and_se(&base);

        let n = case.y.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&r| case.keys[r]);
        let y2: Vec<f64> = order.iter().map(|&r| case.y[r]).collect();
        let x2: Vec<Vec<f64>> = case
            .x
            .iter()
            .map(|c| order.iter().map(|&r| c[r]).collect())
            .collect();
        let entity2: Vec<String> = order
            .iter()
            .map(|&r| format!("z{}", case.n_entities - 1 - r / case.n_periods))
            .collect();
        let time2: Vec<String> = order
            .iter()
            .map(|&r| format!("u{}", case.n_periods - 1 - r % case.n_periods))
            .collect();
        let permuted = fit_two_way(&y2, &x2, &entity2, &time2, cov_index);
        prop_assert!(permuted.is_ok(), "{:?}", permuted.as_ref().err());
        let permuted = permuted.unwrap();
        let (params2, se2) = params_and_se(&permuted);

        for j in 0..params.len() {
            assert_approx_eq(params2[j], params[j], &format!("param[{j}]"));
            assert_approx_eq(se2[j], se[j], &format!("se[{j}]"));
        }
        match (base.hausman_statistic(), permuted.hausman_statistic()) {
            (Some(a), Some(b)) => assert_approx_eq(b, a, "hausman_statistic"),
            (None, None) => {}
            (a, b) => prop_assert!(false, "hausman presence differs: {a:?} vs {b:?}"),
        }
        prop_assert_eq!(base.hausman_df(), permuted.hausman_df());
    }

    /// 2-wayで`y`を`c`倍すると、係数・標準誤差は`c`倍、ハウスマン統計量（Wald、
    /// スケール不変）は不変。一次元の列（あれば先頭のもの）に定数を足すと、切片だけが
    /// 動き、他の係数・標準誤差は変わらない。全`cov_type`。
    #[test]
    fn two_way_is_equivariant_to_y_scale_and_a_constant_shift_moves_only_the_intercept(
        case in one_dim_case_strategy(),
        cov_index in 0..=5usize,
        scale in 0.5f64..4.0,
        shift in -5.0f64..5.0,
    ) {
        let (entity, time) = two_way_labels(case.n_entities, case.n_periods);
        let base = fit_two_way(&case.y, &case.x, &entity, &time, cov_index);
        prop_assert!(base.is_ok(), "{:?}", base.as_ref().err());
        let base = base.unwrap();
        let (params, se) = params_and_se(&base);

        let y_scaled: Vec<f64> = case.y.iter().map(|v| v * scale).collect();
        let scaled = fit_two_way(&y_scaled, &case.x, &entity, &time, cov_index);
        prop_assert!(scaled.is_ok(), "{:?}", scaled.as_ref().err());
        let scaled = scaled.unwrap();
        let (params_s, se_s) = params_and_se(&scaled);
        for j in 0..params.len() {
            assert_approx_eq(params_s[j], params[j] * scale, &format!("scaled param[{j}]"));
            assert_approx_eq(se_s[j], se[j] * scale, &format!("scaled se[{j}]"));
        }
        if let (Some(a), Some(b)) = (base.hausman_statistic(), scaled.hausman_statistic()) {
            assert_approx_eq(b, a, "hausman_statistic under y scaling");
        }

        // 一次元の列（なければ両次元の最後の列）に定数を足す。
        let target = case.x.len() - 1;
        let mut x_shifted = case.x.clone();
        for v in x_shifted[target].iter_mut() {
            *v += shift;
        }
        let shifted = fit_two_way(&case.y, &x_shifted, &entity, &time, cov_index);
        prop_assert!(shifted.is_ok(), "{:?}", shifted.as_ref().err());
        let (params_h, se_h) = params_and_se(&shifted.unwrap());
        assert_approx_eq(params_h[0], params[0] - params[1 + target] * shift, "const");
        for j in 1..params.len() {
            assert_approx_eq(params_h[j], params[j], &format!("shifted param[{j}]"));
        }
        for j in 1..se.len() {
            assert_approx_eq(se_h[j], se[j], &format!("shifted se[{j}]"));
        }
    }

    /// 2-wayのGLSは、報告した3つの重み`θ₁・θ₂・θ₃`で`y - θ₁ȳᵢ - θ₂ȳₜ + θ₃ȳ`と
    /// 各列を手で変換し（切片列は`1 - θ₁ - θ₂ + θ₃`倍）、そのOLSを当てはめた係数・
    /// 標準誤差（Classical・HC1〜HC3）と一致する。一次元の列を含む。`variance_method`は3法とも
    /// （重みは推定法ごとに異なるが、報告した重みでの変換との一致は共通）。変換式の各項
    /// （個体平均・時点平均・全体平均）の取り違えや符号の誤りを検出する。
    ///
    /// 検出できる誤り: 全体平均項の符号反転と全体平均の除数のずれは本テストが、個体・
    /// 時点の重みの入れ替えは本テストと
    /// `two_way_with_one_zero_component_equals_the_one_component_gls`が検出する
    /// （`quasi_demean_two_way`へのバグ注入で確認）。変換式が一貫して誤っていても
    /// 不変性・同変性のテストは通るため、変換式の正しさはこのテストと固定値
    /// （plm）のテストが担う。
    #[test]
    fn two_way_fit_equals_ols_on_the_hand_transformed_data(
        case in one_dim_case_strategy(),
        cov_index in 0..=3usize,
        method in prop_oneof![
            Just(VarianceMethod::SwamyArora),
            Just(VarianceMethod::WallaceHussain),
            Just(VarianceMethod::Amemiya)
        ],
    ) {
        let (entity, time) = two_way_labels(case.n_entities, case.n_periods);
        let est = fit_two_way_with(&case.y, &case.x, &entity, &time, cov_index, method);
        // Amemiyaはwithin変動の無い列（一次元の列）があると拒否される。それ以外の失敗は誤り。
        if matches!(est, Err(PanelError::AmemiyaRequiresWithinVariation { .. })) {
            prop_assert_eq!(method, VarianceMethod::Amemiya);
            prop_assume!(false);
        }
        prop_assert!(est.is_ok(), "{:?}", est.as_ref().err());
        let est = est.unwrap();
        let (theta1, theta2, theta3) = est.two_way_theta().unwrap();

        let n = case.y.len();
        let (n_e, n_t) = (case.n_entities, case.n_periods);
        let quasi = |v: &[f64]| -> Vec<f64> {
            let mut by_entity = vec![0.0; n_e];
            let mut by_time = vec![0.0; n_t];
            for (r, &value) in v.iter().enumerate() {
                by_entity[r / n_t] += value / n_t as f64;
                by_time[r % n_t] += value / n_e as f64;
            }
            let overall = v.iter().sum::<f64>() / n as f64;
            (0..n)
                .map(|r| {
                    v[r] - theta1 * by_entity[r / n_t] - theta2 * by_time[r % n_t]
                        + theta3 * overall
                })
                .collect()
        };
        let y_star = quasi(&case.y);
        let mut columns = vec![vec![1.0 - theta1 - theta2 + theta3; n]];
        columns.extend(case.x.iter().map(|c| quasi(c)));
        let names: Vec<String> = (0..columns.len()).map(|j| format!("c{j}")).collect();
        let ols_input =
            OlsInput::from_columns(&y_star, &columns, names, false, "y".to_string())
                .unwrap();
        let cov = match cov_index {
            0 => CovType::Classical,
            1 => CovType::Hc1,
            2 => CovType::Hc2,
            _ => CovType::Hc3,
        };
        let ols = OlsEstimator::fit(ols_input, cov, 0.95);
        prop_assert!(ols.is_ok(), "{:?}", ols.as_ref().err());
        let ols = ols.unwrap();

        let (params, se) = params_and_se(&est);
        for j in 0..params.len() {
            assert_approx_eq(params[j], *ols.params().get(j, 0), &format!("param[{j}]"));
            assert_approx_eq(se[j], *ols.std_errors().get(j, 0), &format!("se[{j}]"));
        }
    }

    /// 2-wayで`σ_u²`だけ（または`σ_λ²`だけ）が0に切り詰められると、2-way GLSは
    /// 残った成分だけのGLS（`θ₁ = θ₃ = 0`のとき`y - θ₂·ȳ_t`、`θ₂ = θ₃ = 0`のとき
    /// `y - θ₁·ȳ_i`）に退化する。手で準偏差変換した`y`・`X`（切片列は`1 - θ`倍）への
    /// OLSの係数・Classical標準誤差と一致する。
    #[test]
    fn two_way_with_one_zero_component_equals_the_one_component_gls(
        case in one_dim_case_strategy(),
    ) {
        let (entity, time) = two_way_labels(case.n_entities, case.n_periods);
        let est = fit_two_way(&case.y, &case.x, &entity, &time, 0);
        prop_assert!(est.is_ok(), "{:?}", est.as_ref().err());
        let est = est.unwrap();
        let (sigma2_u, sigma2_lambda) =
            (est.sigma2_u(), est.sigma2_lambda().unwrap());
        let (theta1, theta2, theta3) = est.two_way_theta().unwrap();
        // 片方だけが0（両方0ならθはすべて0でpooled OLSになり別の性質）。
        prop_assume!((sigma2_u == 0.0) != (sigma2_lambda == 0.0));
        prop_assert_eq!(theta3, 0.0);

        // 残った成分のグループ（`σ_u² = 0`なら時点、`σ_λ² = 0`ならエンティティ）。
        let (group_of, n_groups, theta) = if sigma2_u == 0.0 {
            prop_assert_eq!(theta1, 0.0);
            (
                (0..case.y.len()).map(|r| r % case.n_periods).collect::<Vec<_>>(),
                case.n_periods,
                theta2,
            )
        } else {
            prop_assert_eq!(theta2, 0.0);
            (
                (0..case.y.len()).map(|r| r / case.n_periods).collect::<Vec<_>>(),
                case.n_entities,
                theta1,
            )
        };
        let quasi = |v: &[f64]| -> Vec<f64> {
            let mut sum = vec![0.0; n_groups];
            let mut count = vec![0.0; n_groups];
            for (r, &g) in group_of.iter().enumerate() {
                sum[g] += v[r];
                count[g] += 1.0;
            }
            (0..v.len())
                .map(|r| v[r] - theta * sum[group_of[r]] / count[group_of[r]])
                .collect()
        };
        let y_star = quasi(&case.y);
        let mut columns = vec![vec![1.0 - theta; case.y.len()]];
        columns.extend(case.x.iter().map(|c| quasi(c)));
        let names: Vec<String> = (0..columns.len()).map(|j| format!("c{j}")).collect();
        let ols_input =
            OlsInput::from_columns(&y_star, &columns, names, false, "y".to_string())
                .unwrap();
        let ols = OlsEstimator::fit(ols_input, CovType::Classical, 0.95);
        prop_assert!(ols.is_ok(), "{:?}", ols.as_ref().err());
        let ols = ols.unwrap();

        let (params, se) = params_and_se(&est);
        for j in 0..params.len() {
            assert_approx_eq(params[j], *ols.params().get(j, 0), &format!("param[{j}]"));
            assert_approx_eq(se[j], *ols.std_errors().get(j, 0), &format!("se[{j}]"));
        }
    }

    /// 1-wayのバランスパネルで時点間でのみ変わる列`xt`を含んでも、行の並べ替えと
    /// エンティティラベルの付け替えで結果が変わらず、ハウスマン検定の自由度は
    /// 両次元で変動する列の数（`xt`は`X̃`が識別できず外れる）、`xt`に定数を足すと
    /// 切片だけが動く。
    #[test]
    fn one_way_balanced_time_only_column_is_invariant_and_excluded_from_hausman(
        case in time_only_case_strategy(),
        cov in re_cov_strategy(),
        shift in -5.0f64..5.0,
    ) {
        let n = case.y.len();
        let entity: Vec<String> =
            (0..n).map(|r| format!("e{}", r / case.n_periods)).collect();
        let base = ReEstimator::fit(
            re_input(&case.y, &case.x, &entity),
            ReEffects::OneWay,
            cov.clone(),
            0.95,
        );
        prop_assert!(base.is_ok(), "{:?}", base.as_ref().err());
        let base = base.unwrap();
        let (params, se) = params_and_se(&base);
        prop_assert_eq!(base.hausman_df(), Some(case.x.len() - 1));

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&r| case.keys[r]);
        let y2: Vec<f64> = order.iter().map(|&r| case.y[r]).collect();
        let x2: Vec<Vec<f64>> = case
            .x
            .iter()
            .map(|c| order.iter().map(|&r| c[r]).collect())
            .collect();
        let entity2: Vec<String> = order
            .iter()
            .map(|&r| format!("z{}", case.n_entities - 1 - r / case.n_periods))
            .collect();
        let permuted = ReEstimator::fit(
            re_input(&y2, &x2, &entity2),
            ReEffects::OneWay,
            cov.clone(),
            0.95,
        );
        prop_assert!(permuted.is_ok(), "{:?}", permuted.as_ref().err());
        let permuted = permuted.unwrap();
        let (params2, se2) = params_and_se(&permuted);
        for j in 0..params.len() {
            assert_approx_eq(params2[j], params[j], &format!("param[{j}]"));
            assert_approx_eq(se2[j], se[j], &format!("se[{j}]"));
        }
        match (base.hausman_statistic(), permuted.hausman_statistic()) {
            (Some(a), Some(b)) => assert_approx_eq(b, a, "hausman_statistic"),
            (a, b) => prop_assert!(false, "hausman must exist: {a:?} vs {b:?}"),
        }

        // `xt`（最後の列）に定数を足す。
        let target = case.x.len() - 1;
        let mut x_shifted = case.x.clone();
        for v in x_shifted[target].iter_mut() {
            *v += shift;
        }
        let shifted = ReEstimator::fit(
            re_input(&case.y, &x_shifted, &entity),
            ReEffects::OneWay,
            cov,
            0.95,
        );
        prop_assert!(shifted.is_ok(), "{:?}", shifted.as_ref().err());
        let (params_h, se_h) = params_and_se(&shifted.unwrap());
        assert_approx_eq(params_h[0], params[0] - params[1 + target] * shift, "const");
        for j in 1..params.len() {
            assert_approx_eq(params_h[j], params[j], &format!("shifted param[{j}]"));
            assert_approx_eq(se_h[j], se[j], &format!("shifted se[{j}]"));
        }
    }
}
