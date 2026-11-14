//! 不均衡パネルの2-way FE（交互射影）。ダミー変数回帰（LSDV）を密に解いた結果と照合する。

use faer::Mat;

use super::*;
use crate::panel::common::ColumnRole;
use crate::panel::common::PanelError;
use crate::panel::fe::FeEstimator;
use crate::panel::fe::FixedEffects;
use crate::panel::fe::input::{FeAbsorption, FeEffects};
use crate::shared::alternating_projections::GroupSystem;
use crate::shared::covariance::xtx_inverse;
use crate::shared::least_squares::least_squares;

/// 決定的な疑似乱数（`[−0.5, 0.5)`）。
fn uniform(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
}

/// 不均衡パネルの生データ。`blocks`個の独立なブロック（個体と時点の組）からなり、ブロックが1つなら連結、
/// 複数なら連結成分が複数に分かれる。各ブロックは個体`n_entities`×時点`n_periods`から確率`keep`で行を
/// 残し、各個体の最初の2時点と各時点の最初の2個体は必ず残す（singletonを避ける）。
struct Panel {
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
    entity: Vec<String>,
    time: Vec<String>,
}

fn panel(blocks: usize, n_entities: usize, n_periods: usize, keep: f64, seed: u64) -> Panel {
    let mut state = seed;
    let beta = [1.5, -0.7];
    let (mut y, mut x1, mut x2) = (Vec::new(), Vec::new(), Vec::new());
    let (mut entity, mut time) = (Vec::new(), Vec::new());
    for b in 0..blocks {
        let alpha: Vec<f64> = (0..n_entities).map(|_| 3.0 * uniform(&mut state)).collect();
        let gamma: Vec<f64> = (0..n_periods).map(|_| 3.0 * uniform(&mut state)).collect();
        for (i, &alpha_i) in alpha.iter().enumerate() {
            for (t, &gamma_t) in gamma.iter().enumerate() {
                if t >= 2 && i >= 2 && uniform(&mut state) + 0.5 >= keep {
                    continue;
                }
                let a = alpha_i + uniform(&mut state);
                let c = gamma_t + uniform(&mut state);
                let (v1, v2) = (a + 2.0 * uniform(&mut state), c + 2.0 * uniform(&mut state));
                y.push(beta[0] * v1 + beta[1] * v2 + alpha_i + gamma_t + 0.5 * uniform(&mut state));
                x1.push(v1);
                x2.push(v2);
                entity.push(format!("e{b}_{i:03}"));
                time.push(format!("t{b}_{t:03}"));
            }
        }
    }
    Panel {
        y,
        x: vec![x1, x2],
        entity,
        time,
    }
}

impl Panel {
    fn input(&self) -> FeInput {
        FeInput::from_columns(
            &self.y,
            &self.x,
            vec!["x1".to_string(), "x2".to_string()],
            &self.entity,
            Some(&self.time),
            "y".into(),
        )
        .unwrap()
    }

    fn fit(&self, cov_type: FeCovType) -> Result<FeEstimator, PanelError> {
        FeEstimator::fit(self.input(), FeEffects::TwoWay, cov_type, 0.95)
    }
}

/// LSDVの設計行列`[x, 個体ダミー全部, 時点ダミー]`。時点ダミーは、連結成分ごとに最初の1つを落とす
/// （成分ごとに個体ダミーの和と時点ダミーの和が等しくなり、そのままでは共線になるため）。
fn lsdv_design(panel: &Panel) -> Mat<f64> {
    let input = panel.input();
    let entity = input.entity_codes();
    let time = input.time_codes().unwrap();
    let components = GroupSystem::new(&[entity, time]).components();
    let mut dropped = vec![false; components.len()];
    let kept_times: Vec<usize> = (0..time.n_groups())
        .filter(|&t| {
            let c = components.of_level(1, t);
            std::mem::replace(&mut dropped[c], true)
        })
        .collect();
    let n = panel.y.len();
    let k = panel.x.len();
    let n_cols = k + entity.n_groups() + kept_times.len();
    Mat::from_fn(n, n_cols, |r, c| {
        if c < k {
            panel.x[c][r]
        } else if c < k + entity.n_groups() {
            f64::from(u8::from(entity.codes()[r] == c - k))
        } else {
            f64::from(u8::from(
                time.codes()[r] == kept_times[c - k - entity.n_groups()],
            ))
        }
    })
}

struct Lsdv {
    params: Vec<f64>,
    /// 個体ダミー・時点ダミー（成分ごとに最初の時点を落とした）の係数。
    effects: Vec<f64>,
    residuals: Vec<f64>,
    /// 傾き係数の古典的な共分散（`σ̂² (Z'Z)⁻¹`の傾きのブロック）。
    classical_cov: Mat<f64>,
    /// 傾き係数のHC0共分散（ダミーを含む設計行列でのサンドイッチの傾きのブロック）。
    hc0_cov: Mat<f64>,
    /// 傾き係数のHC2・HC3共分散（フルレバレッジ`h_ii`で`e²`を`1/(1−h)`・`1/(1−h)²`にした重み）。
    hc2_cov: Mat<f64>,
    hc3_cov: Mat<f64>,
    rank: usize,
}

fn components_in(panel: &Panel) -> usize {
    let input = panel.input();
    GroupSystem::new(&[input.entity_codes(), input.time_codes().unwrap()])
        .components()
        .len()
}

fn lsdv(panel: &Panel) -> Lsdv {
    let z = lsdv_design(panel);
    let k = panel.x.len();
    let n = panel.y.len();
    let y = Mat::from_fn(n, 1, |i, _| panel.y[i]);
    let fit = least_squares(&z, &y).unwrap();
    let residuals: Vec<f64> = (0..n).map(|i| *fit.residuals.get(i, 0)).collect();
    let ssr: f64 = residuals.iter().map(|e| e * e).sum();
    let rank = z.ncols();
    let sigma2 = ssr / (n - rank) as f64;
    let bread = xtx_inverse(&z).unwrap();
    let mut meat = Mat::<f64>::zeros(rank, rank);
    for (r, &e) in residuals.iter().enumerate() {
        for a in 0..rank {
            for b in 0..rank {
                *meat.get_mut(a, b) += e * e * *z.get(r, a) * *z.get(r, b);
            }
        }
    }
    let sandwich = &bread * &meat * &bread;
    let leverage: Vec<f64> = (0..n)
        .map(|r| {
            let row = Mat::from_fn(rank, 1, |c, _| *z.get(r, c));
            (row.transpose() * &bread * &row)[(0, 0)]
        })
        .collect();
    let weighted = |power: i32| {
        let mut meat = Mat::<f64>::zeros(rank, rank);
        for r in 0..n {
            let weight = residuals[r] * residuals[r] / (1.0 - leverage[r]).powi(power);
            for a in 0..rank {
                for b in 0..rank {
                    *meat.get_mut(a, b) += weight * *z.get(r, a) * *z.get(r, b);
                }
            }
        }
        let cov = &bread * &meat * &bread;
        Mat::from_fn(k, k, |a, b| *cov.get(a, b))
    };
    let (hc2_cov, hc3_cov) = (weighted(1), weighted(2));
    Lsdv {
        params: (0..k).map(|j| *fit.params.get(j, 0)).collect(),
        effects: (k..rank).map(|j| *fit.params.get(j, 0)).collect(),
        residuals,
        classical_cov: Mat::from_fn(k, k, |a, b| sigma2 * *bread.get(a, b)),
        hc0_cov: Mat::from_fn(k, k, |a, b| *sandwich.get(a, b)),
        hc2_cov,
        hc3_cov,
        rank,
    }
}

fn close(actual: f64, expected: f64, rtol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= rtol * expected.abs().max(1e-12),
        "{what}: {actual} vs {expected}"
    );
}

#[test]
fn an_unbalanced_panel_matches_the_dummy_variable_regression() {
    for (seed, keep) in [(1, 0.7), (2, 0.4), (3, 0.25)] {
        let data = panel(1, 40, 12, keep, seed);
        let oracle = lsdv(&data);
        let fe = data.fit(FeCovType::Classical).unwrap();
        assert_eq!(fe.df_model() + fe.df_resid(), data.y.len());
        assert_eq!(fe.df_model(), oracle.rank);
        for j in 0..2 {
            close(*fe.params().get(j, 0), oracle.params[j], 1e-8, "params");
            for k in 0..2 {
                close(
                    *fe.cov_params().get(j, k),
                    *oracle.classical_cov.get(j, k),
                    1e-7,
                    "classical covariance",
                );
            }
        }
        for (i, expected) in oracle.residuals.iter().enumerate() {
            assert!(
                (fe.residuals().get(i, 0) - expected).abs() < 1e-8,
                "residual {i}"
            );
        }
    }
}

#[test]
fn the_heteroskedasticity_robust_covariance_matches_the_dummy_variable_regression() {
    let data = panel(1, 30, 10, 0.5, 5);
    let oracle = lsdv(&data);
    let n = data.y.len() as f64;
    let df_resid = n - oracle.rank as f64;
    let fe = data.fit(FeCovType::Hc1).unwrap();
    // HC1の小標本補正は`n / df_resid`（`FeCovType::Hc1`のdoc）。
    for j in 0..2 {
        for k in 0..2 {
            close(
                *fe.cov_params().get(j, k),
                n / df_resid * *oracle.hc0_cov.get(j, k),
                1e-6,
                "HC1 covariance",
            );
        }
    }
}

#[test]
fn the_leverage_based_covariances_use_the_leverage_of_the_full_dummy_regression() {
    // HC2・HC3はフルレバレッジ（ダミーのレバレッジ + within変換後のレバレッジ）が要る。不均衡では
    // ダミーのレバレッジに閉形式が無く、密な行列を解いて求める。
    for (seed, keep) in [(6, 0.6), (7, 0.35)] {
        let data = panel(1, 25, 9, keep, seed);
        let oracle = lsdv(&data);
        for (cov_type, expected) in [
            (FeCovType::Hc2, &oracle.hc2_cov),
            (FeCovType::Hc3, &oracle.hc3_cov),
        ] {
            let fe = data.fit(cov_type.clone()).unwrap();
            for j in 0..2 {
                for k in 0..2 {
                    close(
                        *fe.cov_params().get(j, k),
                        *expected.get(j, k),
                        1e-6,
                        &format!("{cov_type:?} covariance"),
                    );
                }
            }
        }
    }
}

#[test]
fn a_panel_split_into_components_has_leverage_based_covariances_too() {
    let data = panel(2, 10, 6, 0.6, 8);
    let oracle = lsdv(&data);
    let fe = data.fit(FeCovType::Hc3).unwrap();
    for j in 0..2 {
        close(
            *fe.cov_params().get(j, j),
            *oracle.hc3_cov.get(j, j),
            1e-6,
            "HC3 variance",
        );
    }
}

#[test]
fn a_panel_split_into_components_matches_the_dummy_variable_regression() {
    let data = panel(3, 12, 6, 0.6, 9);
    let oracle = lsdv(&data);
    let fe = data.fit(FeCovType::Classical).unwrap();
    // 自由度は連結成分の数によらず`k + N + T − 1`（fixestとlinearmodelsの既定と同じ）。LSDVの
    // ランクは成分の数だけ小さい。
    let input = data.input();
    let (n_entities, n_periods) = (
        input.entity_codes().n_groups(),
        input.time_codes().unwrap().n_groups(),
    );
    assert_eq!(fe.df_model(), 2 + n_entities + n_periods - 1);
    assert_eq!(oracle.rank + components_in(&data) - 1, fe.df_model());
    for (j, &expected) in oracle.params.iter().take(2).enumerate() {
        close(*fe.params().get(j, 0), expected, 1e-8, "params");
    }
    // 古典的な共分散は`SSR / df_resid`のスケールが自由度の数え方で変わるので、そのぶんを合わせて比べる。
    let n = data.y.len() as f64;
    let ratio = (n - oracle.rank as f64) / fe.df_resid() as f64;
    for j in 0..2 {
        close(
            *fe.cov_params().get(j, j),
            ratio * *oracle.classical_cov.get(j, j),
            1e-7,
            "classical variance",
        );
    }
}

#[test]
fn the_estimates_do_not_depend_on_the_row_order() {
    let data = panel(1, 25, 8, 0.5, 11);
    let reference = data.fit(FeCovType::Classical).unwrap();
    let n = data.y.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut state = 77;
    for i in (1..n).rev() {
        let j = ((uniform(&mut state) + 0.5) * (i + 1) as f64) as usize;
        order.swap(i, j);
    }
    let pick = |v: &[f64]| order.iter().map(|&i| v[i]).collect::<Vec<_>>();
    let shuffled = Panel {
        y: pick(&data.y),
        x: data.x.iter().map(|c| pick(c)).collect(),
        entity: order.iter().map(|&i| data.entity[i].clone()).collect(),
        time: order.iter().map(|&i| data.time[i].clone()).collect(),
    }
    .fit(FeCovType::Classical)
    .unwrap();
    for j in 0..2 {
        close(
            *shuffled.params().get(j, 0),
            *reference.params().get(j, 0),
            1e-9,
            "params",
        );
        close(
            *shuffled.std_errors().get(j, 0),
            *reference.std_errors().get(j, 0),
            1e-9,
            "std errors",
        );
    }
}

/// 係数・古典的な共分散・残差・HC2・HC3の共分散・固定効果が、ダミー変数回帰と一致する。
fn assert_matches_the_dummy_regression(data: &Panel) {
    let oracle = lsdv(data);
    let fe = data.fit(FeCovType::Classical).unwrap();
    assert_eq!(fe.df_model(), oracle.rank);
    for j in 0..2 {
        close(*fe.params().get(j, 0), oracle.params[j], 1e-8, "params");
        for k in 0..2 {
            close(
                *fe.cov_params().get(j, k),
                *oracle.classical_cov.get(j, k),
                1e-7,
                "classical covariance",
            );
        }
    }
    for (i, expected) in oracle.residuals.iter().enumerate() {
        assert!(
            (fe.residuals().get(i, 0) - expected).abs() < 1e-8,
            "residual {i}"
        );
    }
    // HC2・HC3のレバレッジは、閉形式のバランス用の式ではなくダミー回帰のハット行列で求める。
    for (cov_type, expected) in [
        (FeCovType::Hc2, &oracle.hc2_cov),
        (FeCovType::Hc3, &oracle.hc3_cov),
    ] {
        let robust = data.fit(cov_type.clone()).unwrap();
        for j in 0..2 {
            for k in 0..2 {
                close(
                    *robust.cov_params().get(j, k),
                    *expected.get(j, k),
                    1e-6,
                    &format!("{cov_type:?} covariance"),
                );
            }
        }
    }
    assert_fixed_effects_match_the_dummy_regression(data);
}

/// 同じ(個体, 時点)の行が複数ある入力も受け付け、ダミー変数回帰と一致する（`effects`の列は州と郡の
/// ような組合せでもよく、組合せごとに1行とは限らない）。
#[test]
fn repeated_entity_time_pairs_are_accepted_and_match_the_dummy_variable_regression() {
    let mut data = panel(1, 20, 8, 0.7, 13);
    for source in [0, 3, 17, 40] {
        let (y, entity, time) = (
            data.y[source] + 0.3,
            data.entity[source].clone(),
            data.time[source].clone(),
        );
        data.y.push(y);
        for column in &mut data.x {
            let value = column[source] - 0.2;
            column.push(value);
        }
        data.entity.push(entity);
        data.time.push(time);
    }
    assert_matches_the_dummy_regression(&data);
}

/// 行数が全組合せの数と同じでも、組合せが重複して別の組合せが欠けていれば、バランスパネルの閉形式
/// （2段階の変換・バランス用のレバレッジ）ではなく不均衡として扱い、ダミー変数回帰と一致する。
#[test]
fn a_repeated_pair_that_offsets_a_missing_one_is_not_treated_as_balanced() {
    let mut data = panel(1, 10, 6, 2.0, 21);
    let (entities, periods) = (
        data.entity
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        data.time
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
    );
    assert_eq!(data.y.len(), entities * periods);
    // 最後の行を、先頭の行と同じ組合せにする。
    let last = data.y.len() - 1;
    data.entity[last] = data.entity[0].clone();
    data.time[last] = data.time[0].clone();
    assert_matches_the_dummy_regression(&data);
}

#[test]
fn a_regressor_constant_within_each_entity_is_rejected_as_collinear_with_the_fixed_effects() {
    let mut data = panel(1, 30, 10, 0.5, 15);
    let entity = data.input().entity_codes().clone();
    let level: Vec<f64> = (0..entity.n_groups()).map(|g| 1.0 + g as f64).collect();
    data.x[1] = entity.codes().iter().map(|&c| level[c]).collect();
    assert_eq!(
        data.fit(FeCovType::Classical).unwrap_err(),
        PanelError::ZeroVarianceAfterDemeaning {
            role: ColumnRole::X,
            column: "x2".to_string()
        }
    );
}

#[test]
fn an_outcome_explained_by_the_fixed_effects_is_rejected() {
    let mut data = panel(1, 30, 10, 0.5, 17);
    let input = data.input();
    let entity = input.entity_codes();
    let time = input.time_codes().unwrap();
    data.y = (0..data.y.len())
        .map(|r| 2.0 * entity.codes()[r] as f64 - 0.5 * time.codes()[r] as f64)
        .collect();
    assert_eq!(
        data.fit(FeCovType::Classical).unwrap_err(),
        PanelError::OutcomeConstantWithinEntity
    );
}

#[test]
fn it_reports_non_convergence_when_the_sweep_limit_is_too_small() {
    let data = panel(1, 40, 12, 0.3, 19);
    let absorption = FeAbsorption {
        tol: 1e-12,
        max_iter: 2,
    };
    let err = FeEstimator::fit_with_absorption(
        data.input(),
        FeEffects::TwoWay,
        FeCovType::Classical,
        0.95,
        &absorption,
    )
    .unwrap_err();
    assert!(
        matches!(err, PanelError::AbsorptionNotConverged { n_sweeps: 2, .. }),
        "{err:?}"
    );
}

#[test]
fn the_absorption_options_are_checked_for_every_kind_of_fit() {
    let data = panel(1, 10, 5, 0.8, 21);
    for effects in [FeEffects::OneWay, FeEffects::TwoWay] {
        let fit = |absorption: FeAbsorption| {
            FeEstimator::fit_with_absorption(
                data.input(),
                effects,
                FeCovType::Classical,
                0.95,
                &absorption,
            )
            .map(|_| ())
        };
        assert_eq!(
            fit(FeAbsorption {
                tol: 0.0,
                max_iter: 100
            }),
            Err(PanelError::InvalidTol { tol: 0.0 })
        );
        assert_eq!(
            fit(FeAbsorption {
                tol: f64::NAN,
                max_iter: 100
            })
            .unwrap_err()
            .to_string(),
            PanelError::InvalidTol { tol: f64::NAN }.to_string()
        );
        assert_eq!(
            fit(FeAbsorption {
                tol: 1e-10,
                max_iter: 0
            }),
            Err(PanelError::InvalidMaxIter { max_iter: 0 })
        );
        assert_eq!(
            fit(FeAbsorption {
                tol: 1e-10,
                max_iter: 10_001
            }),
            Err(PanelError::InvalidMaxIter { max_iter: 10_001 })
        );
    }
}

#[test]
fn the_test_panels_have_the_requested_number_of_connected_components() {
    // `lsdv_design`が使う連結成分の数が、データの作り方（ブロック数）と一致する。
    let data = panel(3, 8, 5, 0.7, 23);
    assert_eq!(components_in(&data), 3);
}

/// ダミー回帰の効果を、(個体ごとの効果, 時点ごとの効果)の順で返す。時点は成分ごとの最初の時点が0。
fn assert_fixed_effects_match_the_dummy_regression(data: &Panel) {
    let oracle = lsdv(data);
    let fe = data.fit(FeCovType::Classical).unwrap();
    let input = data.input();
    let entity = input.entity_codes();
    let time = input.time_codes().unwrap();
    let components = GroupSystem::new(&[entity, time]).components();
    let FixedEffects::TwoWay {
        entity: entity_effects,
        time: time_effects,
    } = fe.fixed_effects()
    else {
        panic!("a two-way fit gives two-way effects");
    };
    for (g, key) in entity.keys().iter().enumerate() {
        close(
            entity_effects[key],
            oracle.effects[g],
            1e-7,
            &format!("entity {key}"),
        );
    }
    // LSDVは成分ごとに最初の時点を落とした列だけを持つ。落とした時点の効果は0。
    let mut dropped = vec![false; components.len()];
    let mut next = entity.n_groups();
    for (g, (key, value)) in time_effects.iter().enumerate() {
        let first_in_component = !std::mem::replace(&mut dropped[components.of_level(1, g)], true);
        if first_in_component {
            assert!(value.abs() < 1e-12, "reference period {key}");
        } else {
            close(*value, oracle.effects[next], 1e-7, &format!("time {key}"));
            next += 1;
        }
    }
}

#[test]
fn the_fixed_effects_of_an_unbalanced_panel_match_the_dummy_regression() {
    assert_fixed_effects_match_the_dummy_regression(&panel(1, 30, 9, 0.5, 31));
}

#[test]
fn the_fixed_effects_of_a_panel_split_into_components_are_normalized_within_each_component() {
    assert_fixed_effects_match_the_dummy_regression(&panel(3, 8, 6, 0.6, 33));
}

/// 個体`i`が時点`i`・`i+1`・`i+2`（`n`で折り返す）に現れる環状のパネル。つながりが細く、交互射影の収束が遅い。
fn ring_panel(n: usize, seed: u64) -> Panel {
    let mut state = seed;
    let (mut y, mut x1, mut x2) = (Vec::new(), Vec::new(), Vec::new());
    let (mut entity, mut time) = (Vec::new(), Vec::new());
    for i in 0..n {
        for t in [i, (i + 1) % n, (i + 2) % n] {
            let (v1, v2) = (uniform(&mut state), uniform(&mut state));
            y.push(1.5 * v1 - 0.7 * v2 + 0.1 * uniform(&mut state));
            x1.push(v1);
            x2.push(v2);
            entity.push(format!("e{i:03}"));
            time.push(format!("t{t:03}"));
        }
    }
    Panel {
        y,
        x: vec![x1, x2],
        entity,
        time,
    }
}

#[test]
fn a_regressor_constant_within_each_entity_is_rejected_even_when_the_projection_converges_slowly() {
    // 環状のパネルは収束が遅く、打ち切りの誤差が許容度の何百倍にもなる。それでも、個体ごとに一定の列は
    // 固定効果と共線として検出される。
    let mut data = ring_panel(40, 51);
    let entity = data.input().entity_codes().clone();
    let level: Vec<f64> = (0..entity.n_groups()).map(|g| 1.0 + g as f64).collect();
    data.x[1] = entity.codes().iter().map(|&c| level[c]).collect();
    assert_eq!(
        data.fit(FeCovType::Classical).unwrap_err(),
        PanelError::ZeroVarianceAfterDemeaning {
            role: ColumnRole::X,
            column: "x2".to_string()
        }
    );
}

#[test]
fn a_ring_panel_matches_the_dummy_variable_regression() {
    let data = ring_panel(30, 53);
    let oracle = lsdv(&data);
    let fe = data.fit(FeCovType::Classical).unwrap();
    for (j, &expected) in oracle.params.iter().take(2).enumerate() {
        close(*fe.params().get(j, 0), expected, 1e-7, "params");
    }
}

#[test]
fn a_large_level_in_the_outcome_does_not_change_the_estimates() {
    // 水準が1e6でも、グループ内の変動に対する相対精度が保たれる（定数は固定効果に吸収される）。
    let data = panel(1, 30, 9, 0.5, 55);
    let shifted = Panel {
        y: data.y.iter().map(|v| v + 1.0e6).collect(),
        x: data.x.clone(),
        entity: data.entity.clone(),
        time: data.time.clone(),
    };
    let reference = data.fit(FeCovType::Classical).unwrap();
    let fe = shifted.fit(FeCovType::Classical).unwrap();
    for j in 0..2 {
        close(
            *fe.params().get(j, 0),
            *reference.params().get(j, 0),
            1e-7,
            "params",
        );
        close(
            *fe.std_errors().get(j, 0),
            *reference.std_errors().get(j, 0),
            1e-6,
            "std errors",
        );
    }
}

/// データから`cov_type`を作る関数。
type CovFactory = Box<dyn Fn(&Panel) -> FeCovType>;

#[test]
fn cluster_and_driscoll_kraay_covariances_work_on_an_unbalanced_panel() {
    let data = panel(1, 30, 10, 0.5, 57);
    let dk = |data: &Panel| FeCovType::Dk {
        bandwidth: Some(1),
        time: dk_time_from(&data.input()),
    };
    let covs: [(&str, CovFactory); 3] = [
        (
            "cluster by entity",
            Box::new(|_| FeCovType::Cluster { groups: None }),
        ),
        (
            "cluster by time",
            Box::new(|d| FeCovType::Cluster {
                groups: Some(d.input().time_codes().unwrap().clone()),
            }),
        ),
        ("dk", Box::new(dk)),
    ];
    let n = data.y.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut state = 91;
    for i in (1..n).rev() {
        let j = ((uniform(&mut state) + 0.5) * (i + 1) as f64) as usize;
        order.swap(i, j);
    }
    let pick = |v: &[f64]| order.iter().map(|&i| v[i]).collect::<Vec<_>>();
    let shuffled = Panel {
        y: pick(&data.y),
        x: data.x.iter().map(|c| pick(c)).collect(),
        entity: order.iter().map(|&i| data.entity[i].clone()).collect(),
        time: order.iter().map(|&i| data.time[i].clone()).collect(),
    };
    for (name, make) in &covs {
        let fe = data.fit(make(&data)).unwrap();
        let again = shuffled.fit(make(&shuffled)).unwrap();
        for j in 0..2 {
            let se = *fe.std_errors().get(j, 0);
            assert!(se.is_finite() && se > 0.0, "{name}: {se}");
            close(*again.std_errors().get(j, 0), se, 1e-8, name);
        }
    }
}
