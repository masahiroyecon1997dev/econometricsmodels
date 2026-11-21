//! property-basedテスト。固定シナリオでは扱えない不均衡パネル・サイズの組み合わせで、分散成分の
//! 不変条件（`y`のスケール・シフト、説明変数のスケール、行順序・ラベルへの不変性）と、
//! 報告したθで手で準偏差変換したOLSとの一致を検証する。トレース項の式の書き間違い
//! （転置・取り違え）は、これらの不変条件を壊す。

use crate::linear::ols::CovType;
use crate::linear::ols::OlsEstimator;
use crate::linear::ols::OlsInput;
use crate::panel::re::ReCovType;
use crate::panel::re::ReEstimator;
use crate::panel::re::input::ReEffects;
use crate::panel::re::input::ReInput;
use crate::panel::re::variance_method::VarianceMethod;
use crate::panel::re::variance_method::variance_components;
use faer::Mat;
use proptest::collection;
use proptest::prelude::*;

#[derive(Debug, Clone)]
struct Case {
    entity_idx: Vec<usize>,
    n_entities: usize,
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
    keys: Vec<u64>,
}

/// 1-wayのパネル。`balanced`なら全個体の観測数が等しい（閉形式）、そうでなければ
/// 2〜6にばらつく（トレース項の式）。`has_effect`が偽なら`σ_u²`が0に切り上げられる
/// ケースが増える。
fn one_way_case() -> impl Strategy<Value = Case> {
    (
        6..=9usize,
        1..=2usize,
        any::<bool>(),
        any::<bool>(),
        3..=5usize,
    )
        .prop_flat_map(|(n_entities, k, has_effect, balanced, t)| {
            let sizes: BoxedStrategy<Vec<usize>> = if balanced {
                Just(vec![t; n_entities]).boxed()
            } else {
                collection::vec(2..=6usize, n_entities).boxed()
            };
            (Just((n_entities, k, has_effect)), sizes)
        })
        .prop_flat_map(|((n_entities, k, has_effect), sizes)| {
            let n: usize = sizes.iter().sum();
            (
                Just((n_entities, has_effect)),
                Just(sizes),
                collection::vec(collection::vec(-10.0f64..10.0, n), k),
                collection::vec(-5.0f64..5.0, n),
                collection::vec(-20.0f64..20.0, n_entities),
                collection::vec(any::<u64>(), n),
            )
        })
        .prop_map(
            |((n_entities, has_effect), sizes, x, noise, effect, keys)| {
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
                Case {
                    entity_idx,
                    n_entities,
                    y,
                    x,
                    keys,
                }
            },
        )
}

fn method_strategy() -> impl Strategy<Value = VarianceMethod> {
    prop_oneof![
        Just(VarianceMethod::WallaceHussain),
        Just(VarianceMethod::Amemiya)
    ]
}

fn build(case: &Case, y: &[f64], x: &[Vec<f64>], entity: &[String]) -> ReInput {
    let names: Vec<String> = (0..x.len()).map(|j| format!("x{j}")).collect();
    let _ = case;
    ReInput::from_columns(y, x, names, entity, None, "y".into()).unwrap()
}

fn labels(case: &Case) -> Vec<String> {
    case.entity_idx.iter().map(|i| format!("e{i}")).collect()
}

fn sigma(input: &ReInput, method: VarianceMethod) -> Option<(f64, f64)> {
    variance_components(input, ReEffects::OneWay, method, 0.95)
        .ok()
        .map(|vc| (vc.sigma2_eps, vc.sigma2_u))
}

/// `plm::ercomp`（`dfcor = 3`）のソースどおりに、`Z`の`Between`・`Sum`・within変換を
/// 観測ごとの行列に展開して`M`を組んだ参照実装（`variance_components`が使う
/// エンティティ単位の外積による実装とは独立）。不均衡の1-way専用。
fn observation_level_trace_form(case: &Case, method: VarianceMethod) -> (f64, f64) {
    let n = case.y.len();
    let n_ent = case.n_entities;
    let idx = &case.entity_idx;
    let mut counts = vec![0.0; n_ent];
    for &i in idx {
        counts[i] += 1.0;
    }
    let k = case.x.len();
    let p = k + 1;
    let z = Mat::from_fn(n, p, |r, j| if j == 0 { 1.0 } else { case.x[j - 1][r] });
    let entity_mean = |col: &dyn Fn(usize) -> f64| -> Vec<f64> {
        let mut sums = vec![0.0; n_ent];
        for r in 0..n {
            sums[idx[r]] += col(r);
        }
        (0..n_ent).map(|i| sums[i] / counts[i]).collect()
    };
    let z_means: Vec<Vec<f64>> = (0..p).map(|j| entity_mean(&|r| *z.get(r, j))).collect();
    let z_between = Mat::from_fn(n, p, |r, j| z_means[j][idx[r]]);
    let z_sum = Mat::from_fn(n, p, |r, j| z_means[j][idx[r]] * counts[idx[r]]);
    let z_within = Mat::from_fn(n, p, |r, j| *z.get(r, j) - z_means[j][idx[r]]);
    let gram = |a: &Mat<f64>, b: &Mat<f64>| a.transpose() * b;

    // 1段目の残差。
    let y_means = entity_mean(&|r| case.y[r]);
    let e: Vec<f64> = match method {
        VarianceMethod::WallaceHussain => {
            let inv = crate::shared::covariance::xtx_inverse(&z).unwrap();
            let zy = Mat::from_fn(p, 1, |j, _| {
                (0..n).map(|r| *z.get(r, j) * case.y[r]).sum::<f64>()
            });
            let beta = &inv * &zy;
            (0..n)
                .map(|r| case.y[r] - (0..p).map(|j| *z.get(r, j) * *beta.get(j, 0)).sum::<f64>())
                .collect()
        }
        _ => {
            let wx = Mat::from_fn(n, k, |r, j| *z_within.get(r, j + 1));
            let wy: Vec<f64> = (0..n).map(|r| case.y[r] - y_means[idx[r]]).collect();
            let inv = crate::shared::covariance::xtx_inverse(&wx).unwrap();
            let wxy = Mat::from_fn(k, 1, |j, _| {
                (0..n).map(|r| *wx.get(r, j) * wy[r]).sum::<f64>()
            });
            let beta = &inv * &wxy;
            let mean_y = case.y.iter().sum::<f64>() / n as f64;
            let mean_fit: f64 = (0..k)
                .map(|j| *beta.get(j, 0) * case.x[j].iter().sum::<f64>() / n as f64)
                .sum();
            (0..n)
                .map(|r| {
                    let fitted: f64 = (0..k).map(|j| *beta.get(j, 0) * case.x[j][r]).sum();
                    case.y[r] - fitted - (mean_y - mean_fit)
                })
                .collect()
        }
    };
    let e_means = entity_mean(&|r| e[r]);
    let q_w: f64 = (0..n).map(|r| (e[r] - e_means[idx[r]]).powi(2)).sum();
    let q_id: f64 = (0..n).map(|r| e_means[idx[r]].powi(2)).sum();

    let tr = |m: &Mat<f64>| -> f64 { (0..m.nrows()).map(|i| *m.get(i, i)).sum() };
    let (big_o, big_n) = (n as f64, n_ent as f64);
    let (a, b, c, d) = match method {
        VarianceMethod::WallaceHussain => {
            let inv = crate::shared::covariance::xtx_inverse(&z).unwrap();
            let cp_w = gram(&z_within, &z_within);
            let cp_b = gram(&z_between, &z_between);
            let cp_s = gram(&z_sum, &z);
            (
                big_o - big_n - tr(&(&inv * &cp_w)),
                tr(&(&(&(&inv * &cp_w) * &inv) * &cp_s)),
                big_n - tr(&(&inv * &cp_b)),
                big_o - 2.0 * tr(&(&inv * &cp_s)) + tr(&(&(&(&inv * &cp_b) * &inv) * &cp_s)),
            )
        }
        _ => {
            let wx = Mat::from_fn(n, k, |r, j| *z_within.get(r, j + 1));
            let xb_mean: Vec<f64> = (0..k)
                .map(|j| (0..n).map(|r| z_means[j + 1][idx[r]]).sum::<f64>() / big_o)
                .collect();
            let xb = Mat::from_fn(n, k, |r, j| z_means[j + 1][idx[r]] - xb_mean[j]);
            let inv = crate::shared::covariance::xtx_inverse(&wx).unwrap();
            (
                big_o - big_n - k as f64,
                0.0,
                big_n - 1.0 + tr(&(&inv * &gram(&xb, &xb))),
                big_o - counts.iter().map(|t| t * t).sum::<f64>() / big_o,
            )
        }
    };
    let det = a * d - b * c;
    ((q_w * d - b * q_id) / det, (a * q_id - c * q_w) / det)
}

fn close(actual: f64, expected: f64, tol: f64) -> bool {
    (actual - expected).abs() <= tol * expected.abs().max(1.0)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// 不均衡の1-wayでは、トレース項の式（エンティティ単位の外積で実装）が、`plm`のソースどおりに
    /// 観測ごとの行列へ展開した参照実装と一致する。
    ///
    /// 有効性の確認（`pooled_trace_moments`・`within_trace_moments`へのバグ注入、確認後に
    /// 元へ戻した）: `W`と`S`の取り違え・`M_id,η`の係数のずれ・`M_id,ν`の`-1`の欠落・
    /// `Σ T_i²`の割る数のずれ・`M_w,ν`のトレース項の取り違えのすべてを本テストが検出する
    /// （不変性のテストは、式が一貫して誤っていても通るため検出できない）。
    #[test]
    fn trace_forms_match_the_observation_level_reference(
        case in one_way_case(),
        method in method_strategy(),
    ) {
        let mut counts = vec![0usize; case.n_entities];
        for &i in &case.entity_idx {
            counts[i] += 1;
        }
        prop_assume!(counts.iter().any(|&c| c != counts[0]));
        let entity = labels(&case);
        let ours = variance_components(
            &build(&case, &case.y, &case.x, &entity),
            ReEffects::OneWay,
            method,
            0.95,
        );
        prop_assume!(ours.is_ok());
        let ours = ours.unwrap();
        let (eps, u) = observation_level_trace_form(&case, method);
        prop_assert!(close(ours.sigma2_eps, eps, 1e-8), "{} vs {eps}", ours.sigma2_eps);
        prop_assert!(close(ours.sigma2_u, u.max(0.0), 1e-8), "{} vs {u}", ours.sigma2_u);
    }

    /// `y`を`c`倍すると`σ_ε²`・`σ_u²`は`c²`倍、シフトしても変わらない。
    #[test]
    fn variance_components_scale_with_y_and_ignore_a_shift(
        case in one_way_case(),
        method in method_strategy(),
        scale in 0.5f64..4.0,
        shift in -50.0f64..50.0,
    ) {
        let entity = labels(&case);
        let base = sigma(&build(&case, &case.y, &case.x, &entity), method);
        prop_assume!(base.is_some());
        let (eps, u) = base.unwrap();
        let scaled_y: Vec<f64> = case.y.iter().map(|v| v * scale).collect();
        let scaled = sigma(&build(&case, &scaled_y, &case.x, &entity), method);
        prop_assume!(scaled.is_some());
        let (eps_s, u_s) = scaled.unwrap();
        prop_assert!(close(eps_s, eps * scale * scale, 1e-7), "{eps_s} vs {eps}");
        prop_assert!(close(u_s, u * scale * scale, 1e-7), "{u_s} vs {u}");
        let shifted_y: Vec<f64> = case.y.iter().map(|v| v + shift).collect();
        let shifted = sigma(&build(&case, &shifted_y, &case.x, &entity), method);
        prop_assume!(shifted.is_some());
        let (eps_h, u_h) = shifted.unwrap();
        prop_assert!(close(eps_h, eps, 1e-7), "{eps_h} vs {eps}");
        prop_assert!(close(u_h, u, 1e-7), "{u_h} vs {u}");
    }

    /// 説明変数の列をスケールしても（係数が逆数倍になるだけで）分散成分は変わらない。
    #[test]
    fn variance_components_ignore_the_scale_of_a_regressor(
        case in one_way_case(),
        method in method_strategy(),
        scale in 0.2f64..5.0,
    ) {
        let entity = labels(&case);
        let base = sigma(&build(&case, &case.y, &case.x, &entity), method);
        prop_assume!(base.is_some());
        let (eps, u) = base.unwrap();
        let mut x = case.x.clone();
        for v in x[0].iter_mut() {
            *v *= scale;
        }
        let scaled = sigma(&build(&case, &case.y, &x, &entity), method);
        prop_assume!(scaled.is_some());
        let (eps_s, u_s) = scaled.unwrap();
        prop_assert!(close(eps_s, eps, 1e-7), "{eps_s} vs {eps}");
        prop_assert!(close(u_s, u, 1e-7), "{u_s} vs {u}");
    }

    /// 行の並べ替えとエンティティラベルの付け替えで分散成分は変わらない。
    #[test]
    fn variance_components_are_invariant_to_row_order_and_relabeling(
        case in one_way_case(),
        method in method_strategy(),
    ) {
        let entity = labels(&case);
        let base = sigma(&build(&case, &case.y, &case.x, &entity), method);
        prop_assume!(base.is_some());
        let (eps, u) = base.unwrap();
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
        let permuted = sigma(&build(&case, &y, &x, &entity2), method);
        prop_assume!(permuted.is_some());
        let (eps_p, u_p) = permuted.unwrap();
        prop_assert!(close(eps_p, eps, 1e-7), "{eps_p} vs {eps}");
        prop_assert!(close(u_p, u, 1e-7), "{u_p} vs {u}");
    }

    /// どの推定法でも、報告したθ_iで手で準偏差変換した`y`・`X`（切片列は`1 - θ_i`倍）への
    /// OLSが、REの係数・Classical標準誤差と一致する（不均衡パネルを含む）。
    #[test]
    fn fit_equals_ols_on_the_data_quasi_demeaned_with_the_reported_theta(
        case in one_way_case(),
        method in prop_oneof![
            Just(VarianceMethod::SwamyArora),
            Just(VarianceMethod::WallaceHussain),
            Just(VarianceMethod::Amemiya)
        ],
    ) {
        let entity = labels(&case);
        let input = build(&case, &case.y, &case.x, &entity);
        let est = ReEstimator::fit_with_variance_method(
            input,
            ReEffects::OneWay,
            ReCovType::Classical,
            method,
            0.95,
        );
        prop_assume!(est.is_ok());
        let est = est.unwrap();
        let theta = est.theta_by_entity().unwrap();
        let n = case.y.len();
        let quasi = |v: &[f64]| -> Vec<f64> {
            let mut sums = vec![0.0; case.n_entities];
            let mut counts = vec![0.0; case.n_entities];
            for (r, &value) in v.iter().enumerate() {
                sums[case.entity_idx[r]] += value;
                counts[case.entity_idx[r]] += 1.0;
            }
            (0..n)
                .map(|r| {
                    let i = case.entity_idx[r];
                    v[r] - theta[i] * sums[i] / counts[i]
                })
                .collect()
        };
        let y_star = quasi(&case.y);
        let mut columns =
            vec![(0..n).map(|r| 1.0 - theta[case.entity_idx[r]]).collect::<Vec<f64>>()];
        columns.extend(case.x.iter().map(|c| quasi(c)));
        let names: Vec<String> = (0..columns.len()).map(|j| format!("c{j}")).collect();
        let ols_input =
            OlsInput::from_columns(&y_star, &columns, names, false, "y".to_string())
                .unwrap();
        let ols = OlsEstimator::fit(ols_input, CovType::Classical, 0.95);
        prop_assume!(ols.is_ok());
        let ols = ols.unwrap();
        for j in 0..est.estimator().params().nrows() {
            let (a, b) = (*est.estimator().params().get(j, 0), *ols.params().get(j, 0));
            prop_assert!(close(a, b, 1e-6), "param[{j}] {a} vs {b}");
            let (a, b) = (*est.std_errors().get(j, 0), *ols.std_errors().get(j, 0));
            prop_assert!(close(a, b, 1e-6), "se[{j}] {a} vs {b}");
        }
    }
}
