//! 3次元以上の固定効果（`FeEffects::MultiWay`）。ダミー変数回帰（LSDV）を密に解いた結果と照合する。

use faer::Mat;

use super::*;
use crate::panel::common::PanelError;
use crate::panel::fe::FeEstimator;
use crate::panel::fe::FixedEffects;
use crate::panel::fe::input::{FeAbsorption, FeEffects};
use crate::shared::alternating_projections::GroupSystem;
use crate::shared::covariance::xtx_inverse;
use crate::shared::group_codes::GroupCodes;
use crate::shared::least_squares::least_squares;

/// 決定的な疑似乱数（`[−0.5, 0.5)`）。
fn uniform(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
}

/// 3つの効果（個体・時点・第3の効果）を持つ不均衡な生データ。個体`n_entities`×時点`n_periods`から
/// 確率`keep`で行を残し（各個体の最初の2時点と各時点の最初の2個体は必ず残す）、第3の効果の水準は
/// `nested`なら個体に入れ子（`個体 % n_third`）、そうでなければ行ごとに無作為に決める。
struct Panel3 {
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
    groups: Vec<Vec<String>>,
}

fn panel3(
    n_entities: usize,
    n_periods: usize,
    n_third: usize,
    keep: f64,
    seed: u64,
    nested: bool,
) -> Panel3 {
    let mut state = seed;
    let beta = [1.5, -0.7];
    let alpha: Vec<f64> = (0..n_entities).map(|_| 3.0 * uniform(&mut state)).collect();
    let gamma: Vec<f64> = (0..n_periods).map(|_| 3.0 * uniform(&mut state)).collect();
    let delta: Vec<f64> = (0..n_third).map(|_| 3.0 * uniform(&mut state)).collect();
    let (mut y, mut x1, mut x2) = (Vec::new(), Vec::new(), Vec::new());
    let mut groups = vec![Vec::new(), Vec::new(), Vec::new()];
    for (i, &alpha_i) in alpha.iter().enumerate() {
        for (t, &gamma_t) in gamma.iter().enumerate() {
            if t >= 2 && i >= 2 && uniform(&mut state) + 0.5 >= keep {
                continue;
            }
            let g = if nested {
                i % n_third
            } else {
                ((uniform(&mut state) + 0.5) * n_third as f64) as usize % n_third
            };
            let a = alpha_i + uniform(&mut state);
            let c = gamma_t + delta[g] + uniform(&mut state);
            let (v1, v2) = (a + 2.0 * uniform(&mut state), c + 2.0 * uniform(&mut state));
            y.push(
                beta[0] * v1
                    + beta[1] * v2
                    + alpha_i
                    + gamma_t
                    + delta[g]
                    + 0.5 * uniform(&mut state),
            );
            x1.push(v1);
            x2.push(v2);
            groups[0].push(format!("e{i:03}"));
            groups[1].push(format!("t{t:03}"));
            groups[2].push(format!("g{g:03}"));
        }
    }
    Panel3 {
        y,
        x: vec![x1, x2],
        groups,
    }
}

impl Panel3 {
    fn input(&self) -> FeInput {
        FeInput::from_effect_groups(
            &self.y,
            &self.x,
            vec!["x1".to_string(), "x2".to_string()],
            self.groups
                .iter()
                .map(|labels| GroupCodes::from_labels(labels))
                .collect(),
            "y".into(),
        )
        .unwrap()
    }

    fn fit(&self, cov_type: FeCovType) -> Result<FeEstimator, PanelError> {
        FeEstimator::fit(self.input(), FeEffects::MultiWay, cov_type, 0.95)
    }

    /// 最初の2つの効果だけを吸収する2-wayの推定（入れ子の第3の効果と係数を比べる）。
    fn fit_two_way(&self) -> FeEstimator {
        let input = FeInput::from_effect_groups(
            &self.y,
            &self.x,
            vec!["x1".to_string(), "x2".to_string()],
            self.groups[..2]
                .iter()
                .map(|labels| GroupCodes::from_labels(labels))
                .collect(),
            "y".into(),
        )
        .unwrap();
        FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap()
    }
}

/// LSDVの設計行列`[x, 個体ダミー全部, 2つ目以降の効果のダミー]`。2つ目以降の各効果は、連結成分ごとに
/// 最初の水準（コード順）を落とす（成分ごとに各効果のダミーの和が等しくなり、そのままでは共線になるため）。
fn lsdv_design(panel: &Panel3, levels: &[&GroupCodes]) -> Mat<f64> {
    let components = GroupSystem::new(levels).components();
    let mut kept: Vec<(usize, usize)> = Vec::new();
    for (dim, level) in levels.iter().enumerate().skip(1) {
        let mut dropped = vec![false; components.len()];
        for code in 0..level.n_groups() {
            if std::mem::replace(&mut dropped[components.of_level(dim, code)], true) {
                kept.push((dim, code));
            }
        }
    }
    let n = panel.y.len();
    let k = panel.x.len();
    let n_entities = levels[0].n_groups();
    Mat::from_fn(n, k + n_entities + kept.len(), |r, c| {
        if c < k {
            panel.x[c][r]
        } else if c < k + n_entities {
            f64::from(u8::from(levels[0].codes()[r] == c - k))
        } else {
            let (dim, code) = kept[c - k - n_entities];
            f64::from(u8::from(levels[dim].codes()[r] == code))
        }
    })
}

struct Lsdv {
    params: Vec<f64>,
    /// 個体ダミー・2つ目以降の効果のダミー（成分ごとに最初の水準を落とした）の係数。
    effects: Vec<f64>,
    residuals: Vec<f64>,
    classical_cov: Mat<f64>,
    hc1_cov: Mat<f64>,
    /// ダミーを含む設計行列のレバレッジ`h_ii`で重みを付けた、HC2・HC3の共分散。
    hc2_cov: Mat<f64>,
    hc3_cov: Mat<f64>,
    rank: usize,
}

fn lsdv(panel: &Panel3, levels: &[&GroupCodes]) -> Lsdv {
    let z = lsdv_design(panel, levels);
    let k = panel.x.len();
    let n = panel.y.len();
    let y = Mat::from_fn(n, 1, |i, _| panel.y[i]);
    let fit = least_squares(&z, &y).unwrap();
    let residuals: Vec<f64> = (0..n).map(|i| *fit.residuals.get(i, 0)).collect();
    let ssr: f64 = residuals.iter().map(|e| e * e).sum();
    let rank = z.ncols();
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
    let sigma2 = ssr / (n - rank) as f64;
    let hc1_scale = n as f64 / (n - rank) as f64;
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
        hc1_cov: Mat::from_fn(k, k, |a, b| hc1_scale * *sandwich.get(a, b)),
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

fn levels_of(input: &FeInput) -> Vec<&GroupCodes> {
    input.effect_codes().iter().collect()
}

#[test]
fn three_effects_match_the_dummy_variable_regression() {
    for (seed, keep) in [(1, 0.8), (2, 0.5)] {
        let data = panel3(30, 10, 4, keep, seed, false);
        let input = data.input();
        let oracle = lsdv(&data, &levels_of(&input));
        let fe = data.fit(FeCovType::Classical).unwrap();
        assert_eq!(fe.n_groups(), &[30, 10, 4]);
        // 水準数の合計から、2つ目以降の効果が1つずつ重複させる定数項を引く。
        assert_eq!(fe.df_model(), 2 + 30 + 10 + 4 - 2);
        assert_eq!(fe.df_model() + fe.df_resid(), data.y.len());
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
        assert_eq!(oracle.rank, fe.df_model());
    }
}

#[test]
fn the_heteroskedasticity_robust_covariance_matches_the_dummy_variable_regression() {
    let data = panel3(25, 9, 5, 0.6, 3, false);
    let input = data.input();
    let oracle = lsdv(&data, &levels_of(&input));
    let fe = data.fit(FeCovType::Hc1).unwrap();
    for j in 0..2 {
        for k in 0..2 {
            close(
                *fe.cov_params().get(j, k),
                *oracle.hc1_cov.get(j, k),
                1e-7,
                "HC1 covariance",
            );
        }
    }
}

#[test]
fn the_fixed_effects_rebuild_the_outcome_and_match_the_dummy_regression() {
    let data = panel3(20, 8, 3, 0.7, 5, false);
    let input = data.input();
    let levels = levels_of(&input);
    let oracle = lsdv(&data, &levels);
    let fe = data.fit(FeCovType::Classical).unwrap();
    let FixedEffects::MultiWay(effects) = fe.fixed_effects() else {
        panic!("a three-effect fit gives multi-way effects");
    };
    assert_eq!(effects.len(), 3);
    for (level, dimension) in levels.iter().zip(&effects) {
        assert_eq!(dimension.len(), level.n_groups());
        for ((key, _), expected) in dimension.iter().zip(level.keys()) {
            assert_eq!(key, expected);
        }
    }
    // 行ごとに、傾きと3つの効果と残差で被説明変数を組み立て直せる。
    for r in 0..data.y.len() {
        let fitted: f64 = (0..2)
            .map(|j| *fe.params().get(j, 0) * data.x[j][r])
            .sum::<f64>()
            + levels
                .iter()
                .zip(&effects)
                .map(|(level, dimension)| dimension[level.codes()[r]].1)
                .sum::<f64>();
        assert!(
            (fitted + fe.residuals().get(r, 0) - data.y[r]).abs() < 1e-8,
            "row {r}"
        );
    }
    // 連結成分ごとの基準化は、LSDVで同じ水準のダミーを落としたときの係数と一致する。
    let components = GroupSystem::new(&levels).components();
    let mut next = levels[0].n_groups();
    for (g, (key, value)) in effects[0].iter().enumerate() {
        // 個体の効果は、同じ成分の基準の効果のぶんだけ持ち上がる。LSDVでは落とした水準が0なので、
        // 個体のダミーの係数にそのぶんが含まれる。
        close(*value, oracle.effects[g], 1e-7, &format!("entity {key}"));
    }
    for (dim, dimension) in effects.iter().enumerate().skip(1) {
        let mut dropped = vec![false; components.len()];
        for (code, (key, value)) in dimension.iter().enumerate() {
            let first = !std::mem::replace(&mut dropped[components.of_level(dim, code)], true);
            if first {
                assert!(value.abs() < 1e-12, "reference level {key}");
            } else {
                close(*value, oracle.effects[next], 1e-7, &format!("level {key}"));
                next += 1;
            }
        }
    }
}

#[test]
fn a_third_effect_nested_in_the_first_keeps_the_slopes_and_the_counting_convention() {
    let data = panel3(24, 8, 4, 0.8, 7, true);
    let two_way = data.fit_two_way();
    let three_way = data.fit(FeCovType::Classical).unwrap();
    // 入れ子の効果は個体の効果に吸収されるので、係数は2-wayと同じ。
    for j in 0..2 {
        close(
            *three_way.params().get(j, 0),
            *two_way.params().get(j, 0),
            1e-8,
            "params",
        );
    }
    // 自由度はfixestと同じ数え方で、入れ子によるランク落ちは引かない。
    assert_eq!(three_way.df_model(), two_way.df_model() + (4 - 1));
}

#[test]
fn three_effects_support_cluster_and_driscoll_kraay_covariances() {
    let data = panel3(30, 10, 4, 0.7, 11, false);
    let cluster = data.fit(FeCovType::Cluster { groups: None }).unwrap();
    assert_eq!(cluster.df_inference(), 30 - 1);
    assert!(cluster.std_errors().get(0, 0).is_finite());
    let time = TimeKeys::lexicographic(data.groups[1].clone());
    let dk = data
        .fit(FeCovType::Dk {
            bandwidth: Some(1),
            time,
        })
        .unwrap();
    assert_eq!(dk.df_inference(), 10 - 1);
    assert!(dk.std_errors().get(0, 0).is_finite());
}

#[test]
fn singletons_in_any_effect_are_reported_with_the_chain() {
    let mut data = panel3(12, 6, 3, 1.1, 13, false);
    // 第3の効果に、1行だけの水準を作る。
    let last = data.y.len() - 1;
    data.groups[2][last] = "singleton".to_string();
    let err = data.fit(FeCovType::Classical).unwrap_err();
    assert!(
        matches!(
            &err,
            PanelError::SingletonGroup { dimension, group_id, .. }
                if dimension.index() == 2 && group_id == "singleton"
        ),
        "{err:?}"
    );
}

#[test]
fn the_leverage_based_covariances_match_the_dummy_variable_regression() {
    // HC2・HC3は、ダミーを含む設計行列のレバレッジ（ダミーのレバレッジ + within変換後のレバレッジ）が要る。
    for (seed, keep) in [(17, 0.8), (18, 0.45)] {
        let data = panel3(18, 7, 3, keep, seed, false);
        let input = data.input();
        let oracle = lsdv(&data, &levels_of(&input));
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
fn the_leverage_based_covariances_of_a_nested_third_effect_equal_the_two_way_ones() {
    // 第3の効果が個体に入れ子なら、列空間は個体と時点のダミーのものと同じで、HC2・HC3も変わらない。
    let data = panel3(16, 7, 4, 0.7, 19, true);
    let two_way = data.fit_two_way();
    for cov_type in [FeCovType::Hc2, FeCovType::Hc3] {
        let nested = data.fit(cov_type.clone()).unwrap();
        let reference = {
            let input = FeInput::from_effect_groups(
                &data.y,
                &data.x,
                vec!["x1".to_string(), "x2".to_string()],
                data.groups[..2]
                    .iter()
                    .map(|labels| GroupCodes::from_labels(labels))
                    .collect(),
                "y".into(),
            )
            .unwrap();
            FeEstimator::fit(input, FeEffects::TwoWay, cov_type.clone(), 0.95).unwrap()
        };
        for j in 0..2 {
            close(
                *nested.params().get(j, 0),
                *two_way.params().get(j, 0),
                1e-8,
                "params",
            );
            for k in 0..2 {
                close(
                    *nested.cov_params().get(j, k),
                    *reference.cov_params().get(j, k),
                    1e-7,
                    &format!("{cov_type:?} covariance"),
                );
            }
        }
    }
}

#[test]
fn the_leverage_based_covariances_refuse_a_dense_matrix_above_the_limit() {
    // 最大の次元（2500）を除いた水準数の合計（2000 + 1500）が上限（3000）を超える。各水準は2行以上に現れる。
    let rows = 5000;
    let make = |label: &str, modulus: usize| -> Vec<String> {
        (0..rows)
            .map(|r| format!("{label}{:05}", r % modulus))
            .collect()
    };
    let y: Vec<f64> = (0..rows).map(|r| ((r * 7919) % 101) as f64).collect();
    let x: Vec<f64> = (0..rows).map(|r| ((r * 104729) % 97) as f64).collect();
    let input = FeInput::from_effect_groups(
        &y,
        &[x],
        vec!["x1".to_string()],
        vec![
            GroupCodes::from_labels(&make("a", 2500)),
            GroupCodes::from_labels(&make("b", 2000)),
            GroupCodes::from_labels(&make("c", 1500)),
        ],
        "y".into(),
    )
    .unwrap();
    let err = FeEstimator::fit(input, FeEffects::MultiWay, FeCovType::Hc3, 0.95).unwrap_err();
    assert!(
        matches!(
            err,
            PanelError::LeverageDimensionTooLarge { n_levels: 3500, .. }
        ),
        "{err:?}"
    );
}

#[test]
fn multi_way_needs_at_least_three_dimensions() {
    let data = panel3(10, 5, 3, 1.1, 19, false);
    let input = FeInput::from_effect_groups(
        &data.y,
        &data.x,
        vec!["x1".to_string(), "x2".to_string()],
        data.groups[..2]
            .iter()
            .map(|labels| GroupCodes::from_labels(labels))
            .collect(),
        "y".into(),
    )
    .unwrap();
    assert!(matches!(
        FeEstimator::fit(input, FeEffects::MultiWay, FeCovType::Classical, 0.95).unwrap_err(),
        PanelError::Common(crate::shared::error::CommonError::Internal { .. })
    ));
}

#[test]
fn the_estimates_do_not_depend_on_the_order_of_the_effects() {
    let data = panel3(20, 8, 4, 0.7, 23, false);
    let reference = data.fit(FeCovType::Classical).unwrap();
    let permuted = Panel3 {
        y: data.y.clone(),
        x: data.x.clone(),
        groups: vec![
            data.groups[2].clone(),
            data.groups[0].clone(),
            data.groups[1].clone(),
        ],
    };
    // 個体を先頭にしなくても、係数・自由度・古典的な標準誤差は同じ。
    let shuffled = permuted.fit(FeCovType::Classical).unwrap();
    for j in 0..2 {
        close(
            *shuffled.params().get(j, 0),
            *reference.params().get(j, 0),
            1e-8,
            "params",
        );
        close(
            *shuffled.std_errors().get(j, 0),
            *reference.std_errors().get(j, 0),
            1e-7,
            "std errors",
        );
    }
    assert_eq!(shuffled.df_model(), reference.df_model());
}

// ── 分断・重複・バランス・4次元・エラー経路・クラスター標準誤差 ───────────────────

/// 2つのパネルを、水準のラベルが重ならない別の連結成分として並べる。
fn disjoint_union(first: &Panel3, second: &Panel3) -> Panel3 {
    let relabel = |panel: &Panel3, tag: &str| -> Vec<Vec<String>> {
        panel
            .groups
            .iter()
            .map(|labels| {
                labels
                    .iter()
                    .map(|label| format!("{tag}_{label}"))
                    .collect()
            })
            .collect()
    };
    let (a, b) = (relabel(first, "a"), relabel(second, "b"));
    Panel3 {
        y: first.y.iter().chain(&second.y).copied().collect(),
        x: (0..2)
            .map(|j| first.x[j].iter().chain(&second.x[j]).copied().collect())
            .collect(),
        groups: a
            .into_iter()
            .zip(b)
            .map(|(left, right)| left.into_iter().chain(right).collect())
            .collect(),
    }
}

/// LSDVと同じ推定値になること（係数・古典的な共分散・残差・固定効果）。
fn assert_matches_the_dummy_regression(data: &Panel3) {
    let input = data.input();
    let levels = levels_of(&input);
    let oracle = lsdv(data, &levels);
    let fe = data.fit(FeCovType::Classical).unwrap();
    // 自由度は連結成分の数によらず`k + ΣG − (K−1)`（fixestの既定と同じ）。LSDVのランクは、2つ目以降の
    // 各次元で成分の数（−1）だけ小さい。
    let n_components = GroupSystem::new(&levels).components().len();
    assert_eq!(
        oracle.rank + (n_components - 1) * (levels.len() - 1),
        fe.df_model()
    );
    // 古典的な共分散は`SSR / df_resid`のスケールが自由度の数え方で変わるので、そのぶんを合わせて比べる。
    let ratio = (data.y.len() as f64 - oracle.rank as f64) / fe.df_resid() as f64;
    for j in 0..2 {
        close(*fe.params().get(j, 0), oracle.params[j], 1e-8, "params");
        for k in 0..2 {
            close(
                *fe.cov_params().get(j, k),
                ratio * *oracle.classical_cov.get(j, k),
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
    let FixedEffects::MultiWay(effects) = fe.fixed_effects() else {
        panic!("a multi-way fit gives multi-way effects");
    };
    // 行ごとに、傾きと全ての効果と残差で被説明変数を組み立て直せる（LSDVに依らない確認）。
    for r in 0..data.y.len() {
        let fitted: f64 = (0..2)
            .map(|j| *fe.params().get(j, 0) * data.x[j][r])
            .sum::<f64>()
            + levels
                .iter()
                .zip(&effects)
                .map(|(level, dimension)| dimension[level.codes()[r]].1)
                .sum::<f64>();
        assert!(
            (fitted + fe.residuals().get(r, 0) - data.y[r]).abs() < 1e-8,
            "row {r}"
        );
    }
    // 基準化: 2つ目以降の各次元の、成分ごとの最初の水準が0。LSDVも同じ水準のダミーを落とすので、
    // 個体の効果と残りの水準の効果は係数と一致する（落とす水準の選び方は定義で、上の再構成が独立な確認）。
    let components = GroupSystem::new(&levels).components();
    for (g, (key, value)) in effects[0].iter().enumerate() {
        close(*value, oracle.effects[g], 1e-7, &format!("entity {key}"));
    }
    let mut next = levels[0].n_groups();
    for (dim, dimension) in effects.iter().enumerate().skip(1) {
        let mut seen = vec![false; components.len()];
        for (code, (key, value)) in dimension.iter().enumerate() {
            if std::mem::replace(&mut seen[components.of_level(dim, code)], true) {
                close(*value, oracle.effects[next], 1e-7, &format!("level {key}"));
                next += 1;
            } else {
                assert!(value.abs() < 1e-12, "reference level {key}");
            }
        }
    }
}

#[test]
fn panels_split_into_connected_components_match_the_dummy_regression() {
    let data = disjoint_union(
        &panel3(14, 7, 3, 0.8, 31, false),
        &panel3(11, 6, 3, 0.8, 32, false),
    );
    let input = data.input();
    assert_eq!(
        GroupSystem::new(&levels_of(&input)).components().len(),
        2,
        "the two blocks share no level"
    );
    assert_matches_the_dummy_regression(&data);
}

#[test]
fn repeated_combinations_of_all_effects_are_accepted() {
    let mut data = panel3(16, 7, 3, 0.8, 41, false);
    for source in [0, 5, 9, 20, 5] {
        let (y, labels) = (
            data.y[source] + 0.4,
            data.groups
                .iter()
                .map(|g| g[source].clone())
                .collect::<Vec<_>>(),
        );
        data.y.push(y);
        for column in &mut data.x {
            let value = column[source] - 0.3;
            column.push(value);
        }
        for (g, label) in data.groups.iter_mut().zip(labels) {
            g.push(label);
        }
    }
    assert_matches_the_dummy_regression(&data);
}

#[test]
fn a_full_grid_with_a_crossed_third_effect_is_not_given_a_closed_form() {
    // 個体 × 時点がすべてそろうが、第3の効果が行ごとに無作為に決まる。2-wayの閉形式は使えない。
    let data = panel3(12, 6, 3, 1.1, 51, false);
    assert_eq!(data.y.len(), 12 * 6);
    assert_matches_the_dummy_regression(&data);
}

#[test]
fn four_effects_match_the_dummy_regression() {
    let mut data = panel3(20, 8, 4, 0.7, 61, false);
    let fourth: Vec<String> = (0..data.y.len())
        .map(|r| format!("h{}", (r * 7 + r / 3) % 3))
        .collect();
    data.groups.push(fourth);
    let fe = data.fit(FeCovType::Classical).unwrap();
    assert_eq!(fe.n_groups(), &[20, 8, 4, 3]);
    assert_eq!(fe.df_model(), 2 + 20 + 8 + 4 + 3 - 3);
    assert_matches_the_dummy_regression(&data);
}

/// 個体でクラスタリングした、LSDVの共分散（傾きのブロック）。小標本補正は`G/(G−1) · (n−1)/(n−K)`。
fn lsdv_cluster_cov(panel: &Panel3, levels: &[&GroupCodes], k_correction: usize) -> Mat<f64> {
    let z = lsdv_design(panel, levels);
    let n = panel.y.len();
    let y = Mat::from_fn(n, 1, |i, _| panel.y[i]);
    let fit = least_squares(&z, &y).unwrap();
    let rank = z.ncols();
    let bread = xtx_inverse(&z).unwrap();
    let clusters = levels[0];
    let mut scores = vec![vec![0.0; rank]; clusters.n_groups()];
    for r in 0..n {
        let e = *fit.residuals.get(r, 0);
        for (c, score) in scores[clusters.codes()[r]].iter_mut().enumerate() {
            *score += *z.get(r, c) * e;
        }
    }
    let mut meat = Mat::<f64>::zeros(rank, rank);
    for score in &scores {
        for a in 0..rank {
            for b in 0..rank {
                *meat.get_mut(a, b) += score[a] * score[b];
            }
        }
    }
    let g = clusters.n_groups() as f64;
    let scale = g / (g - 1.0) * (n as f64 - 1.0) / (n as f64 - k_correction as f64);
    let cov = &bread * &meat * &bread;
    Mat::from_fn(2, 2, |a, b| scale * *cov.get(a, b))
}

#[test]
fn the_cluster_covariance_matches_the_dummy_regression_with_the_nested_effect_not_counted() {
    let (n_entities, n_periods, n_third) = (28, 9, 4);
    let data = panel3(n_entities, n_periods, n_third, 0.7, 71, false);
    let input = data.input();
    let levels = levels_of(&input);
    // 個体でクラスタリングすると個体の効果だけがクラスターにネストするので、fixestと同じく
    // K = df_model − (個体数 − 1) = k + 時点数 + 第3の効果の水準数 − 1。
    let k_correction = 2 + n_periods + n_third - 1;
    let expected = lsdv_cluster_cov(&data, &levels, k_correction);
    let fe = data.fit(FeCovType::Cluster { groups: None }).unwrap();
    for a in 0..2 {
        for b in 0..2 {
            close(
                *fe.cov_params().get(a, b),
                *expected.get(a, b),
                1e-7,
                "cluster covariance",
            );
        }
    }
}

#[test]
fn a_regressor_absorbed_by_the_third_effect_is_rejected() {
    let mut data = panel3(14, 6, 3, 0.9, 81, false);
    // 第3の効果の水準だけで決まる列は、固定効果が吸収する。
    data.x[0] = data.groups[2]
        .iter()
        .map(|label| label.len() as f64 + label.bytes().last().map_or(0.0, f64::from))
        .collect();
    assert_eq!(
        data.fit(FeCovType::Classical).unwrap_err(),
        PanelError::ZeroVarianceAfterDemeaning {
            column: "x1".to_string()
        }
    );
}

#[test]
fn an_outcome_explained_by_the_three_effects_is_rejected() {
    let mut data = panel3(14, 6, 3, 0.9, 82, false);
    let input = data.input();
    let levels = levels_of(&input);
    data.y = (0..data.y.len())
        .map(|r| {
            0.37 * levels[0].codes()[r] as f64
                + 0.91 * levels[1].codes()[r] as f64
                + 1.3 * levels[2].codes()[r] as f64
        })
        .collect();
    assert_eq!(
        data.fit(FeCovType::Classical).unwrap_err(),
        PanelError::OutcomeConstantWithinEntity
    );
}

#[test]
fn a_perfect_fit_is_rejected() {
    let mut data = panel3(14, 6, 3, 0.9, 83, false);
    let input = data.input();
    let levels = levels_of(&input);
    data.y = (0..data.y.len())
        .map(|r| {
            1.5 * data.x[0][r]
                + 0.37 * levels[0].codes()[r] as f64
                + 0.91 * levels[1].codes()[r] as f64
                + 1.3 * levels[2].codes()[r] as f64
        })
        .collect();
    assert!(matches!(
        data.fit(FeCovType::Classical).unwrap_err(),
        PanelError::WithinRegressionFailed { .. }
    ));
}

#[test]
fn the_sweep_limit_applies_to_three_effects() {
    let data = panel3(20, 8, 4, 0.7, 84, false);
    let absorption = FeAbsorption {
        tol: 1e-12,
        max_iter: 1,
    };
    let err = FeEstimator::fit_with_absorption(
        data.input(),
        FeEffects::MultiWay,
        FeCovType::Classical,
        0.95,
        &absorption,
    )
    .unwrap_err();
    assert!(
        matches!(err, PanelError::AbsorptionNotConverged { n_sweeps: 1, .. }),
        "{err:?}"
    );
}
