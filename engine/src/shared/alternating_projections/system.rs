//! グループの索引と、交互射影による列の射影。

use crate::shared::group_codes::GroupCodes;

/// 収束判定の既定の相対許容度。
pub(crate) const DEFAULT_TOLERANCE: f64 = 1e-12;
/// 掃引回数の既定の上限。
pub(crate) const DEFAULT_MAX_SWEEPS: usize = 10_000;
/// 加速の分母（`‖Δ₂ − Δ₁‖²`）が`‖Δ₂‖²`のこの割合以下なら加速を飛ばす（進みがほぼ止まっている）。
const ACCELERATION_DENOMINATOR_FLOOR: f64 = 1e-30;
/// 収束判定の尺度（第1掃引後の列の絶対値の最大）の下限を、元の列の絶対値の最大に対する倍率で表す。
/// 列が（丸め誤差を除いて）グループの張る空間に入るとき、第1掃引後の値は丸め誤差だけになり、そのまま
/// 尺度にすると、丸め誤差を追いかけて収束しない。
const SCALE_FLOOR_IN_EPSILONS: f64 = 16.0;
/// 収束の速さの推定（隣り合う掃引の変化量の比）の上限。1に近いと誤差の見積もりが発散する。
const MAX_CONTRACTION: f64 = 1.0 - 1e-6;
/// 収束の速さを推定できないとき（最初の掃引の組の前半で収束したとき）に仮定する値。
const DEFAULT_CONTRACTION: f64 = 0.5;

/// 射影の打ち切りの設定。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ProjectionOptions {
    /// 収束の許容度。1回の掃引で引く水準平均の絶対値の最大が、`tol × 尺度`以下になったら収束とする。
    /// 尺度は、第1掃引でグループの水準を除いた後の列の絶対値の最大（水準の大きい列でも、
    /// グループ内の変動に対する相対精度になる）。
    pub(crate) tol: f64,
    /// 掃引（全グループの平均を1回ずつ引く操作）の上限回数。
    pub(crate) max_sweeps: usize,
}

impl Default for ProjectionOptions {
    fn default() -> Self {
        Self {
            tol: DEFAULT_TOLERANCE,
            max_sweeps: DEFAULT_MAX_SWEEPS,
        }
    }
}

/// 上限回数までに収束しなかった。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NotConverged {
    /// 行った掃引の回数。
    pub(crate) n_sweeps: usize,
    /// 最後の掃引で引いた水準平均の絶対値の最大を、収束判定の尺度で割った値。
    pub(crate) relative_change: f64,
}

/// 射影の結果。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Projected {
    /// 全グループのダミー変数への最小二乗の残差（`列 − Σ_d effects[d][コード]`）。
    pub(crate) values: Vec<f64>,
    /// 行った掃引の回数。
    pub(crate) n_sweeps: usize,
    /// `values`に残る誤差の大きさ（絶対値の最大）の見積もり。最後の掃引の変化量と、隣り合う掃引の変化量の
    /// 比`ρ`から`変化量 · ρ / (1 − ρ)`とする。1つのグループだけなら0。
    pub(crate) error_bound: f64,
    /// `effects[d][g]`: グループ`d`の水準`g`に引いた量の合計。効果を求める呼び出し
    /// （[`GroupSystem::project_with_effects`]）でだけ入る。連結成分ごとに定数の不定性がある。
    pub(crate) effects: Vec<Vec<f64>>,
}

pub(super) struct Dimension<'a> {
    pub(super) codes: &'a [usize],
    inverse_counts: Vec<f64>,
}

impl Dimension<'_> {
    pub(super) fn n_groups(&self) -> usize {
        self.inverse_counts.len()
    }
}

/// 同じ行を持つ複数のグループ（`GroupCodes`）の索引。列ごとの射影で使い回す。
pub(crate) struct GroupSystem<'a> {
    pub(super) dims: Vec<Dimension<'a>>,
    pub(super) nobs: usize,
}

impl<'a> GroupSystem<'a> {
    /// `groups`の各要素が1つのグループ（行ごとの水準コード）。
    ///
    /// # Panics
    /// `groups`が空、または観測数が揃っていない場合（呼び出し側の実装誤り）。
    pub(crate) fn new(groups: &[&'a GroupCodes]) -> Self {
        assert!(!groups.is_empty(), "at least one group is required");
        let nobs = groups[0].nobs();
        assert!(
            groups.iter().all(|g| g.nobs() == nobs),
            "all groups must have the same number of rows"
        );
        let dims = groups
            .iter()
            .map(|g| Dimension {
                codes: g.codes(),
                inverse_counts: g.counts().iter().map(|&c| 1.0 / c as f64).collect(),
            })
            .collect();
        Self { dims, nobs }
    }

    /// 行数。
    #[cfg(test)]
    pub(crate) fn nobs(&self) -> usize {
        self.nobs
    }

    /// `col`から全グループの水準平均を除いた残差を返す。
    ///
    /// `col`は有限値であること（欠損・無限大は入力の検証で弾く前提。NaNや無限大が混ざると、尺度が
    /// 無限大になり、収束の判定が意味を失う）。
    ///
    /// # Errors
    /// `options.max_sweeps`回までに収束しなかったとき。
    ///
    /// # Panics
    /// `col.len()`が行数と一致しない場合。
    #[cfg(test)]
    pub(crate) fn project(
        &self,
        col: &[f64],
        options: &ProjectionOptions,
    ) -> Result<Projected, NotConverged> {
        self.solve(col, options, false)
    }

    /// [`Self::project`]に加えて、各水準に引いた量の合計（効果）も返す。
    ///
    /// # Errors
    /// [`Self::project`]と同じ。
    ///
    /// # Panics
    /// [`Self::project`]と同じ。
    pub(crate) fn project_with_effects(
        &self,
        col: &[f64],
        options: &ProjectionOptions,
    ) -> Result<Projected, NotConverged> {
        self.solve(col, options, true)
    }

    fn solve(
        &self,
        col: &[f64],
        options: &ProjectionOptions,
        track: bool,
    ) -> Result<Projected, NotConverged> {
        assert_eq!(col.len(), self.nobs, "column length must match the rows");
        let initial_scale = max_abs(col);
        let mut state = State::new(self, col, track);
        if initial_scale == 0.0 {
            return Ok(state.finish(0, 0.0));
        }
        // 1つのグループなら1回の掃引で厳密に除ける。
        if self.dims.len() == 1 {
            state.sweep(self);
            return Ok(state.finish(1, 0.0));
        }
        // 第1掃引で水準を除き、残ったグループ内の変動の大きさを収束判定の尺度にする。
        let level_change = state.sweep(self);
        let mut n_sweeps = 1;
        let scale = max_abs(&state.x).max(SCALE_FLOOR_IN_EPSILONS * f64::EPSILON * initial_scale);
        let threshold = options.tol * scale;
        let mut contraction = DEFAULT_CONTRACTION;
        // 第1掃引の変化量（水準の除去）。上限が1回だけのとき、非収束の報告に使う。
        let mut change = level_change;
        loop {
            // Irons–Tuck加速には、続けた2回の掃引の前後の3点`x₀, x₁, x₂`が要る。
            state.keep_before();
            let mut first_change = 0.0;
            for step in 0..2 {
                if n_sweeps == options.max_sweeps {
                    return Err(NotConverged {
                        n_sweeps,
                        relative_change: change / scale,
                    });
                }
                change = state.sweep(self);
                n_sweeps += 1;
                if step == 0 {
                    first_change = change;
                } else if first_change > 0.0 {
                    contraction = (change / first_change).min(MAX_CONTRACTION);
                }
                if change <= threshold {
                    let bound = change * contraction / (1.0 - contraction);
                    return Ok(state.finish(n_sweeps, bound.max(change)));
                }
                if step == 0 {
                    state.keep_middle();
                }
            }
            state.accelerate();
        }
    }
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// 反復中の列（と、追跡するときは効果）。
struct State {
    x: Vec<f64>,
    /// 加速に使う、組の最初の点`x₀`と途中の点`x₁`。
    before: Vec<f64>,
    middle: Vec<f64>,
    effects: Vec<Vec<f64>>,
    middle_effects: Vec<Vec<f64>>,
    track: bool,
    sums: Vec<f64>,
}

impl State {
    fn new(system: &GroupSystem<'_>, col: &[f64], track: bool) -> Self {
        let sizes = system.dims.iter().map(|d| d.n_groups());
        let effects: Vec<Vec<f64>> = if track {
            sizes.clone().map(|n| vec![0.0; n]).collect()
        } else {
            Vec::new()
        };
        Self {
            x: col.to_vec(),
            before: vec![0.0; col.len()],
            middle: vec![0.0; col.len()],
            middle_effects: effects.clone(),
            effects,
            track,
            sums: vec![0.0; sizes.max().unwrap_or(0)],
        }
    }

    /// 全グループの水準平均を順に引く。引いた平均の絶対値の最大を返す。
    fn sweep(&mut self, system: &GroupSystem<'_>) -> f64 {
        let mut largest = 0.0_f64;
        for (d, dim) in system.dims.iter().enumerate() {
            let n_groups = dim.n_groups();
            let sums = &mut self.sums[..n_groups];
            sums.fill(0.0);
            for (value, &g) in self.x.iter().zip(dim.codes) {
                sums[g] += *value;
            }
            for (sum, inverse) in sums.iter_mut().zip(&dim.inverse_counts) {
                *sum *= inverse;
                largest = largest.max(sum.abs());
            }
            for (value, &g) in self.x.iter_mut().zip(dim.codes) {
                *value -= sums[g];
            }
            if self.track {
                for (effect, mean) in self.effects[d].iter_mut().zip(sums.iter()) {
                    *effect += *mean;
                }
            }
        }
        largest
    }

    /// 組の最初の点`x₀`を控える。
    fn keep_before(&mut self) {
        self.before.copy_from_slice(&self.x);
    }

    /// 1回目の掃引の後の`x₁`を控える。
    fn keep_middle(&mut self) {
        self.middle.copy_from_slice(&self.x);
        if self.track {
            for (kept, current) in self.middle_effects.iter_mut().zip(&self.effects) {
                kept.copy_from_slice(current);
            }
        }
    }

    /// Irons–Tuck加速。`x₀`（`before`）、`x₁`（`middle`）、`x₂`（現在）から、`x₂`の先を外挿する。
    /// 動かす量は`x₂ − x₁`の倍率で、効果にも同じ倍率をかける（列は効果の線形な関数なので整合する）。
    fn accelerate(&mut self) {
        let (mut cross, mut norm, mut delta_norm) = (0.0, 0.0, 0.0);
        for ((x2, x1), x0) in self.x.iter().zip(&self.middle).zip(&self.before) {
            let delta2 = x2 - x1;
            let second = delta2 - (x1 - x0);
            cross += delta2 * second;
            norm += second * second;
            delta_norm += delta2 * delta2;
        }
        // NaNも「加速しない」側に倒す。
        let floor = ACCELERATION_DENOMINATOR_FLOOR * delta_norm;
        if norm.partial_cmp(&floor) != Some(std::cmp::Ordering::Greater) {
            return;
        }
        let alpha = cross / norm;
        for (x2, x1) in self.x.iter_mut().zip(&self.middle) {
            *x2 -= alpha * (*x2 - x1);
        }
        if self.track {
            for (effects, kept) in self.effects.iter_mut().zip(&self.middle_effects) {
                for (e2, e1) in effects.iter_mut().zip(kept) {
                    *e2 -= alpha * (*e2 - e1);
                }
            }
        }
    }

    fn finish(self, n_sweeps: usize, error_bound: f64) -> Projected {
        Projected {
            values: self.x,
            n_sweeps,
            error_bound,
            effects: self.effects,
        }
    }
}

#[cfg(test)]
mod tests {
    use faer::Mat;
    use faer::prelude::SolveLstsq;

    use super::*;

    /// 決定的な疑似乱数（`[−0.5, 0.5)`）。
    fn uniform(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }

    fn codes(levels: &[usize]) -> GroupCodes {
        let labels: Vec<String> = levels.iter().map(|l| format!("g{l:05}")).collect();
        GroupCodes::from_labels(&labels)
    }

    /// 不均衡パネル: 個体`n_entities`×時点`n_periods`から、確率`keep`で行を残す。個体にも時点にも
    /// 2行以上残るよう、各個体の最初の2時点と各時点の最初の2個体は必ず残す。
    fn unbalanced_panel(
        n_entities: usize,
        n_periods: usize,
        keep: f64,
        seed: u64,
    ) -> (GroupCodes, GroupCodes) {
        let mut state = seed;
        let (mut entity, mut time) = (Vec::new(), Vec::new());
        for i in 0..n_entities {
            for t in 0..n_periods {
                let forced = (t < 2) || (i < 2);
                if forced || uniform(&mut state) + 0.5 < keep {
                    entity.push(i);
                    time.push(t);
                }
            }
        }
        (codes(&entity), codes(&time))
    }

    fn random_column(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n).map(|_| 100.0 * uniform(&mut state)).collect()
    }

    /// LSDVの残差。最初のグループは全水準、それ以外は最初の水準を落としたダミーに最小二乗をあてはめる
    /// （連結なパネルで、ダミーがフルランクになる）。
    fn lsdv_residual(groups: &[&GroupCodes], col: &[f64]) -> Vec<f64> {
        let n = col.len();
        let mut columns: Vec<Vec<f64>> = Vec::new();
        for (d, g) in groups.iter().enumerate() {
            for level in (if d == 0 { 0 } else { 1 })..g.n_groups() {
                columns.push(
                    g.codes()
                        .iter()
                        .map(|&c| if c == level { 1.0 } else { 0.0 })
                        .collect(),
                );
            }
        }
        let design = Mat::from_fn(n, columns.len(), |i, j| columns[j][i]);
        let y = Mat::from_fn(n, 1, |i, _| col[i]);
        let beta = design.col_piv_qr().solve_lstsq(&y);
        let fitted = &design * &beta;
        (0..n).map(|i| col[i] - fitted[(i, 0)]).collect()
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
    }

    #[test]
    fn one_group_gives_the_group_demeaned_column_in_a_single_sweep() {
        let g = codes(&[0, 0, 1, 1, 1]);
        let system = GroupSystem::new(&[&g]);
        let out = system
            .project(&[1.0, 3.0, 2.0, 5.0, 8.0], &ProjectionOptions::default())
            .unwrap();
        assert_eq!(out.n_sweeps, 1);
        assert_eq!(out.values, vec![-1.0, 1.0, -3.0, 0.0, 3.0]);
    }

    #[test]
    fn a_balanced_panel_matches_the_closed_form_double_demeaning() {
        let (n_entities, n_periods) = (5, 4);
        let (mut entity, mut time) = (Vec::new(), Vec::new());
        for i in 0..n_entities {
            for t in 0..n_periods {
                entity.push(i);
                time.push(t);
            }
        }
        let (entity, time) = (codes(&entity), codes(&time));
        let y = random_column(n_entities * n_periods, 3);
        let n = y.len() as f64;
        let grand = y.iter().sum::<f64>() / n;
        let expected: Vec<f64> = (0..y.len())
            .map(|r| {
                let (i, t) = (r / n_periods, r % n_periods);
                let row_mean: f64 =
                    (0..n_periods).map(|s| y[i * n_periods + s]).sum::<f64>() / n_periods as f64;
                let col_mean: f64 =
                    (0..n_entities).map(|j| y[j * n_periods + t]).sum::<f64>() / n_entities as f64;
                y[r] - row_mean - col_mean + grand
            })
            .collect();
        let out = GroupSystem::new(&[&entity, &time])
            .project(&y, &ProjectionOptions::default())
            .unwrap();
        assert!(max_abs_diff(&out.values, &expected) < 1e-10);
    }

    #[test]
    fn an_unbalanced_two_way_panel_matches_the_dummy_variable_regression() {
        for (seed, keep) in [(1, 0.7), (2, 0.4), (3, 0.25)] {
            let (entity, time) = unbalanced_panel(40, 12, keep, seed);
            let system = GroupSystem::new(&[&entity, &time]);
            assert_eq!(system.components().len(), 1);
            let y = random_column(system.nobs(), 10 + seed);
            let out = system.project(&y, &ProjectionOptions::default()).unwrap();
            let expected = lsdv_residual(&[&entity, &time], &y);
            assert!(
                max_abs_diff(&out.values, &expected) < 1e-9,
                "seed {seed}: {}",
                max_abs_diff(&out.values, &expected)
            );
        }
    }

    #[test]
    fn three_groups_match_the_dummy_variable_regression() {
        let (entity, time) = unbalanced_panel(30, 10, 0.5, 7);
        let mut state = 99;
        let region = codes(
            &(0..entity.nobs())
                .map(|_| ((uniform(&mut state) + 0.5) * 6.0) as usize)
                .collect::<Vec<_>>(),
        );
        let system = GroupSystem::new(&[&entity, &time, &region]);
        let y = random_column(system.nobs(), 5);
        let out = system.project(&y, &ProjectionOptions::default()).unwrap();
        let expected = lsdv_residual(&[&entity, &time, &region], &y);
        assert!(max_abs_diff(&out.values, &expected) < 1e-9);
    }

    #[test]
    fn a_chain_of_overlapping_pairs_converges_slowly_but_within_the_limit() {
        // 個体iは時点iとi+1にだけ現れる。つながりが細い最悪の形で、掃引の回数は鎖の長さの2乗に
        // 比例して増える（長さ20で約900回、50で約6000回、200で約86000回）。
        let n = 30;
        let (mut entity, mut time) = (Vec::new(), Vec::new());
        for i in 0..n {
            entity.extend([i, i]);
            time.extend([i, i + 1]);
        }
        let (entity, time) = (codes(&entity), codes(&time));
        let system = GroupSystem::new(&[&entity, &time]);
        let y = random_column(system.nobs(), 21);
        let out = system.project(&y, &ProjectionOptions::default()).unwrap();
        let expected = lsdv_residual(&[&entity, &time], &y);
        assert!(max_abs_diff(&out.values, &expected) < 1e-8);
        assert!(out.n_sweeps < 5_000, "{}", out.n_sweeps);
    }

    #[test]
    fn a_disconnected_panel_projects_each_component_separately() {
        // 2成分。各成分は個体3×時点3の完全なパネル。
        let (mut entity, mut time) = (Vec::new(), Vec::new());
        for block in 0..2 {
            for i in 0..3 {
                for t in 0..3 {
                    entity.push(block * 3 + i);
                    time.push(block * 3 + t);
                }
            }
        }
        let (entity, time) = (codes(&entity), codes(&time));
        let system = GroupSystem::new(&[&entity, &time]);
        assert_eq!(system.components().len(), 2);
        let y = random_column(18, 4);
        let out = system.project(&y, &ProjectionOptions::default()).unwrap();
        // 成分ごとの二重デミーニング。
        for block in 0..2 {
            let part = &y[block * 9..(block + 1) * 9];
            let grand = part.iter().sum::<f64>() / 9.0;
            for r in 0..9 {
                let (i, t) = (r / 3, r % 3);
                let row = (0..3).map(|s| part[i * 3 + s]).sum::<f64>() / 3.0;
                let col = (0..3).map(|j| part[j * 3 + t]).sum::<f64>() / 3.0;
                let expected = part[r] - row - col + grand;
                assert!((out.values[block * 9 + r] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn the_tracked_effects_reproduce_the_projection() {
        let (entity, time) = unbalanced_panel(25, 8, 0.5, 11);
        let system = GroupSystem::new(&[&entity, &time]);
        let y = random_column(system.nobs(), 12);
        let out = system
            .project_with_effects(&y, &ProjectionOptions::default())
            .unwrap();
        let plain = system.project(&y, &ProjectionOptions::default()).unwrap();
        for (r, &value) in out.values.iter().enumerate() {
            let fitted = out.effects[0][entity.codes()[r]] + out.effects[1][time.codes()[r]];
            assert!((y[r] - fitted - value).abs() < 1e-9);
        }
        // 効果を追跡しても、残差は変わらない。
        assert!(max_abs_diff(&out.values, &plain.values) < 1e-12);
        assert!(plain.effects.is_empty());
    }

    #[test]
    fn a_column_inside_the_span_of_the_groups_projects_to_zero() {
        let (entity, time) = unbalanced_panel(20, 6, 0.6, 13);
        let system = GroupSystem::new(&[&entity, &time]);
        let a = random_column(20, 14);
        let b = random_column(6, 15);
        let y: Vec<f64> = (0..system.nobs())
            .map(|r| a[entity.codes()[r]] + b[time.codes()[r]])
            .collect();
        let out = system.project(&y, &ProjectionOptions::default()).unwrap();
        assert!(out.values.iter().all(|v| v.abs() < 1e-9));
        let zero = system
            .project(&vec![0.0; y.len()], &ProjectionOptions::default())
            .unwrap();
        assert_eq!(zero.n_sweeps, 0);
        assert!(zero.values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn a_column_with_a_large_level_keeps_its_precision_relative_to_the_within_variation() {
        // 水準が1e6でグループ内の変動が50程度。尺度を元の列の最大値にすると、絶対誤差が1e-6まで許される。
        let (entity, time) = unbalanced_panel(30, 10, 0.4, 41);
        let system = GroupSystem::new(&[&entity, &time]);
        let y = random_column(system.nobs(), 42);
        let shifted: Vec<f64> = y.iter().map(|v| v + 1.0e6).collect();
        let out = system
            .project(&shifted, &ProjectionOptions::default())
            .unwrap();
        let expected = lsdv_residual(&[&entity, &time], &y);
        assert!(max_abs_diff(&out.values, &expected) < 1e-9);
    }

    #[test]
    fn the_error_bound_covers_the_actual_error_even_on_a_slowly_converging_chain() {
        let n = 30;
        let (mut e, mut t) = (Vec::new(), Vec::new());
        for i in 0..n {
            e.extend([i, i]);
            t.extend([i, i + 1]);
        }
        let (entity, time) = (codes(&e), codes(&t));
        let system = GroupSystem::new(&[&entity, &time]);
        // 個体の効果と時点の効果の和は、グループの張る空間に入るので、射影は0になるはず。
        let in_span: Vec<f64> = entity
            .codes()
            .iter()
            .zip(time.codes())
            .map(|(&e, &t)| (0.3 * e as f64).sin() + (0.7 * t as f64).cos())
            .collect();
        let loose = ProjectionOptions {
            tol: 1e-6,
            max_sweeps: 10_000,
        };
        let out = system.project(&in_span, &loose).unwrap();
        let actual = out.values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(actual <= out.error_bound, "{actual} > {}", out.error_bound);
        // 許容度を厳しくすると、誤差も見積もりも小さくなる。
        let tight = system
            .project(&in_span, &ProjectionOptions::default())
            .unwrap();
        assert!(tight.error_bound < out.error_bound);
    }

    #[test]
    fn it_reports_non_convergence_when_the_sweep_limit_is_too_small() {
        let (entity, time) = unbalanced_panel(40, 12, 0.3, 17);
        let system = GroupSystem::new(&[&entity, &time]);
        let y = random_column(system.nobs(), 18);
        let options = ProjectionOptions {
            tol: 1e-12,
            max_sweeps: 3,
        };
        let err = system.project(&y, &options).unwrap_err();
        assert_eq!(err.n_sweeps, 3);
        assert!(err.relative_change > 1e-12);
    }

    #[test]
    fn a_limit_of_one_sweep_reports_a_finite_change() {
        let (entity, time) = unbalanced_panel(20, 8, 0.5, 61);
        let system = GroupSystem::new(&[&entity, &time]);
        let y = random_column(system.nobs(), 62);
        let options = ProjectionOptions {
            tol: 1e-12,
            max_sweeps: 1,
        };
        let err = system.project(&y, &options).unwrap_err();
        assert_eq!(err.n_sweeps, 1);
        assert!(err.relative_change.is_finite() && err.relative_change > 0.0);
    }

    #[test]
    #[should_panic(expected = "same number of rows")]
    fn groups_of_different_lengths_are_a_contract_violation() {
        let a = codes(&[0, 1, 0]);
        let b = codes(&[0, 1]);
        GroupSystem::new(&[&a, &b]);
    }
}
