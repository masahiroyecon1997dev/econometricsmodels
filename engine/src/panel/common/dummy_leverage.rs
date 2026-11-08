//! 不均衡な2-wayの固定効果ダミー（個体と時点）だけの回帰のレバレッジ。

use faer::linalg::solvers::Solve;
use faer::{Mat, Side};

use super::error::PanelError;
use super::validation::covers_every_combination;
use crate::shared::alternating_projections::GroupSystem;
use crate::shared::error::{CommonError, MatrixKind};
use crate::shared::group_codes::GroupCodes;

/// 密に解く行列の次元（水準の少ない方の効果の水準数）の上限。`n × n`でなく`水準数²`のメモリと
/// `水準数³`の計算量を抑える（3000で行列1つ約72MB、`llt`と逆行列などが同時に生きるので実際の最大は
/// その数倍、計算は数秒）。
pub(super) const MAX_DENSE_LEVELS: usize = 3000;

/// 組合せが欠ける2-wayのHC2/HC3が密に解く行列の次元が、上限以下であることを確かめる。バランスパネルは
/// 閉形式なので常に通る。`dummy_leverage`と同じ検査を、重い射影の前に行うために切り出している。
///
/// # Errors
/// 水準の少ない方の効果の水準数が上限を超えるとき`PanelError::LeverageDimensionTooLarge`。
pub(crate) fn check_leverage_dimension(
    entity: &GroupCodes,
    time: &GroupCodes,
) -> Result<(), PanelError> {
    let levels = entity.n_groups().min(time.n_groups());
    if levels > MAX_DENSE_LEVELS && !covers_every_combination(entity, time) {
        return Err(PanelError::LeverageDimensionTooLarge {
            n_levels: levels,
            max_levels: MAX_DENSE_LEVELS,
        });
    }
    Ok(())
}

/// 個体ダミーと時点ダミーだけの回帰（切片は個体ダミーに含まれる）のレバレッジ`h_ii`を、行ごとに返す。
///
/// 個体を先に消すと、`P = P_E + P_{M_E T}`（`M_E T`は時点ダミーから個体ごとの平均を引いたもの）。
/// 行`i`の`M_E T`の行は`e_{t_i} − c_{e_i} / n_{e_i}`（`c_e`は個体`e`の時点の出現を数えたベクトル）で、
/// `(M_E T)'(M_E T) = diag(N_t) − Σ_e c_e c_e' / n_e =: M`。`M`の零空間は連結成分の指示ベクトルなので、
/// それを足して正定値にした`A = (M + P_null)⁻¹`が`M`の擬似逆行列を与える（`P_null`の成分は`h_ii`に
/// 寄与しない）。よって
/// `h_ii = 1/n_e + A[t,t] − 2 (A c_e)[t] / n_e + c_e'A c_e / n_e²`。
/// 個体と時点の役割は対称なので、水準の少ない方を密に解く。計算量は`O(n · 少ない方の水準数 + 水準数³)`。
///
/// 同じ`(entity, time)`が複数の行にあってもよい（`c_e`は出現回数のベクトルで、式は回数について
/// そのまま成り立つ）。個体ごとに時点の出現回数を先に集約するので、計算量は個体内の行数の2乗ではなく、
/// 個体内の異なる時点の数に依存する。
///
/// # Errors
/// - 少ない方の水準数が[`MAX_DENSE_LEVELS`]を超えるとき`PanelError::LeverageDimensionTooLarge`
/// - 行列が正定値でない（連結が極端に細く、丸めで`M + P_null`の分解が失敗したとき）は
///   `CommonError::NearSingularMatrix`
pub(super) fn dummy_leverage(
    entity: &GroupCodes,
    time: &GroupCodes,
) -> Result<Vec<f64>, PanelError> {
    let (eliminated, dense) = if time.n_groups() <= entity.n_groups() {
        (entity, time)
    } else {
        (time, entity)
    };
    let levels = dense.n_groups();
    check_leverage_dimension(entity, time)?;
    let n = eliminated.nobs();
    let rows = eliminated.group_indices();
    let sizes = eliminated.counts();
    let dense_codes = dense.codes();

    // M = diag(N_t) − Σ_e c_e c_e' / n_e
    let mut m = Mat::<f64>::zeros(levels, levels);
    for (t, &count) in dense.counts().iter().enumerate() {
        *m.get_mut(t, t) = count as f64;
    }
    let mut tally = Tally::new(levels);
    for (group, group_rows) in rows.iter().enumerate() {
        let weight = 1.0 / sizes[group] as f64;
        let occurrences = tally.count(group_rows, dense_codes);
        for &(a, count_a) in occurrences {
            for &(b, count_b) in occurrences {
                *m.get_mut(a, b) -= weight * (count_a * count_b) as f64;
            }
        }
        tally.clear();
    }
    // 連結成分ごとの零空間を足す: 成分の指示ベクトルを単位ベクトルにして`u u'`。
    let components = GroupSystem::new(&[eliminated, dense]).components();
    let mut members = vec![0usize; components.len()];
    for t in 0..levels {
        members[components.of_level(1, t)] += 1;
    }
    for a in 0..levels {
        for b in 0..levels {
            let c = components.of_level(1, a);
            if c == components.of_level(1, b) {
                *m.get_mut(a, b) += 1.0 / members[c] as f64;
            }
        }
    }
    let llt = m
        .llt(Side::Lower)
        .map_err(|_| CommonError::NearSingularMatrix {
            matrix: MatrixKind::DesignGram,
        })?;
    let a = llt.solve(Mat::<f64>::identity(levels, levels));

    // 個体ごとに v = A c_e と c_e' v / n_e² を作り、行の値を組み立てる。
    let mut leverage = vec![0.0; n];
    let mut v = vec![0.0; levels];
    for (group, group_rows) in rows.iter().enumerate() {
        let size = sizes[group] as f64;
        v.fill(0.0);
        for &(column, count) in tally.count(group_rows, dense_codes) {
            for (t, value) in v.iter_mut().enumerate() {
                *value += count as f64 * *a.get(t, column);
            }
        }
        tally.clear();
        let quadratic: f64 =
            group_rows.iter().map(|&r| v[dense_codes[r]]).sum::<f64>() / (size * size);
        for &r in group_rows {
            let t = dense_codes[r];
            leverage[r] = 1.0 / size + *a.get(t, t) - 2.0 * v[t] / size + quadratic;
        }
    }
    Ok(leverage)
}

/// 1つのグループの行が、もう一方の効果の水準（`dense_codes`）に何回ずつ現れるかの集計。
/// 水準数の長さの作業領域を使い回し、グループごとに触った水準だけを数えて戻す。
struct Tally {
    counts: Vec<usize>,
    touched: Vec<(usize, usize)>,
}

impl Tally {
    fn new(levels: usize) -> Self {
        Self {
            counts: vec![0; levels],
            touched: Vec::new(),
        }
    }

    /// `group_rows`の行ごとに`dense_codes`の水準を数え、現れた水準と回数の組を初めて現れた順に返す。
    /// 次のグループの前に[`clear`](Self::clear)を呼ぶ。
    fn count(&mut self, group_rows: &[usize], dense_codes: &[usize]) -> &[(usize, usize)] {
        for &r in group_rows {
            let level = dense_codes[r];
            if self.counts[level] == 0 {
                self.touched.push((level, 0));
            }
            self.counts[level] += 1;
        }
        for entry in &mut self.touched {
            entry.1 = self.counts[entry.0];
        }
        &self.touched
    }

    /// 触った水準の回数を0に戻す。
    fn clear(&mut self) {
        for &(level, _) in &self.touched {
            self.counts[level] = 0;
        }
        self.touched.clear();
    }
}

#[cfg(test)]
mod tests {
    use faer::prelude::SolveLstsq;

    use super::*;

    fn uniform(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// 個体`n_entities` × 時点`n_periods`のブロックを`blocks`個並べ、確率`keep`で行を残す
    /// （各個体の最初の2時点と各時点の最初の2個体は必ず残す）。戻り値は`(個体のラベル, 時点のラベル)`。
    fn labels(
        blocks: usize,
        n_entities: usize,
        n_periods: usize,
        keep: f64,
        seed: u64,
    ) -> (Vec<String>, Vec<String>) {
        let mut state = seed;
        let (mut entity, mut time) = (Vec::new(), Vec::new());
        for b in 0..blocks {
            for i in 0..n_entities {
                for t in 0..n_periods {
                    if t >= 2 && i >= 2 && uniform(&mut state) >= keep {
                        continue;
                    }
                    entity.push(format!("e{b}_{i:03}"));
                    time.push(format!("t{b}_{t:03}"));
                }
            }
        }
        (entity, time)
    }

    /// ダミー回帰のハット行列の対角。連結成分ごとに最初の時点ダミーを1つ落としてフルランクにする。
    fn brute_force(entity: &GroupCodes, time: &GroupCodes) -> Vec<f64> {
        let n = entity.nobs();
        let components = GroupSystem::new(&[entity, time]).components();
        let mut dropped = vec![false; components.len()];
        let kept: Vec<usize> = (0..time.n_groups())
            .filter(|&t| std::mem::replace(&mut dropped[components.of_level(1, t)], true))
            .collect();
        let columns = entity.n_groups() + kept.len();
        let z = Mat::from_fn(n, columns, |r, c| {
            let hit = if c < entity.n_groups() {
                entity.codes()[r] == c
            } else {
                time.codes()[r] == kept[c - entity.n_groups()]
            };
            f64::from(u8::from(hit))
        });
        // H = Z (Z'Z)⁻¹ Z'。各列`e_i`への射影の`i`成分を、最小二乗で1行ずつ求める。
        let qr = z.col_piv_qr();
        (0..n)
            .map(|i| {
                let unit = Mat::from_fn(n, 1, |r, _| f64::from(u8::from(r == i)));
                let fitted = &z * qr.solve_lstsq(&unit);
                *fitted.get(i, 0)
            })
            .collect()
    }

    fn check(blocks: usize, n_entities: usize, n_periods: usize, keep: f64, seed: u64) {
        let (entity, time) = labels(blocks, n_entities, n_periods, keep, seed);
        let (entity, time) = (
            GroupCodes::from_labels(&entity),
            GroupCodes::from_labels(&time),
        );
        let got = dummy_leverage(&entity, &time).unwrap();
        let expected = brute_force(&entity, &time);
        for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
            assert!((a - b).abs() < 1e-9, "row {i}: {a} vs {b}");
        }
    }

    #[test]
    fn it_matches_the_hat_matrix_of_the_dummy_regression_for_an_unbalanced_panel() {
        check(1, 15, 6, 0.5, 1);
        check(1, 12, 9, 0.35, 2);
    }

    /// `labels`で作ったパネルに、既存の行を複製して同じ組合せの行を足す（観測数は増える）。
    fn check_with_repeated_rows(seed: u64) {
        let (mut entity, mut time) = labels(1, 12, 7, 0.5, seed);
        let n = entity.len();
        for source in [0, 4, n / 2, n - 1, 4, 4] {
            entity.push(entity[source].clone());
            time.push(time[source].clone());
        }
        let (entity, time) = (
            GroupCodes::from_labels(&entity),
            GroupCodes::from_labels(&time),
        );
        assert!(!covers_every_combination(&entity, &time));
        let got = dummy_leverage(&entity, &time).unwrap();
        let expected = brute_force(&entity, &time);
        for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
            assert!((a - b).abs() < 1e-9, "row {i}: {a} vs {b}");
        }
    }

    #[test]
    fn it_matches_the_hat_matrix_when_a_combination_repeats() {
        check_with_repeated_rows(6);
        check_with_repeated_rows(7);
    }

    #[test]
    fn it_handles_many_repeats_of_the_same_combinations() {
        // 個体ごとの行数が時点の数より桁違いに多い入力（行数の2乗の計算に戻っていないこと）。
        let (mut entity, mut time) = (Vec::new(), Vec::new());
        for e in 0..4 {
            for t in 0..3 {
                for _ in 0..2000 {
                    if (e + t) % 4 != 3 {
                        entity.push(format!("e{e}"));
                        time.push(format!("t{t}"));
                    }
                }
            }
        }
        let (entity, time) = (
            GroupCodes::from_labels(&entity),
            GroupCodes::from_labels(&time),
        );
        let got = dummy_leverage(&entity, &time).unwrap();
        // 個体・時点ごとのダミーだけの回帰のハット行列の対角の和は、ランク（連結なら 個体数+時点数-1）に等しい。
        let trace: f64 = got.iter().sum();
        assert!((trace - (entity.n_groups() + time.n_groups() - 1) as f64).abs() < 1e-8);
    }

    #[test]
    fn it_swaps_the_roles_when_there_are_more_periods_than_entities() {
        check(1, 6, 14, 0.5, 3);
    }

    #[test]
    fn it_handles_a_panel_split_into_connected_components() {
        check(3, 6, 5, 0.6, 4);
    }

    #[test]
    fn it_reproduces_the_closed_form_of_a_balanced_panel() {
        let (entity, time) = labels(1, 5, 4, 1.1, 5);
        let (entity, time) = (
            GroupCodes::from_labels(&entity),
            GroupCodes::from_labels(&time),
        );
        let n = entity.nobs() as f64;
        let got = dummy_leverage(&entity, &time).unwrap();
        for (r, &h) in got.iter().enumerate() {
            let expected = 1.0 / entity.counts()[entity.codes()[r]] as f64
                + 1.0 / time.counts()[time.codes()[r]] as f64
                - 1.0 / n;
            assert!((h - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn it_refuses_a_dense_matrix_above_the_limit() {
        let n = MAX_DENSE_LEVELS + 1;
        let labels: Vec<String> = (0..2 * n).map(|r| format!("g{:05}", r / 2)).collect();
        let entity = GroupCodes::from_labels(&labels);
        let time_labels: Vec<String> = (0..2 * n)
            .map(|r| format!("g{:05}", r.div_ceil(2) % n))
            .collect();
        let time = GroupCodes::from_labels(&time_labels);
        assert_eq!(
            dummy_leverage(&entity, &time),
            Err(PanelError::LeverageDimensionTooLarge {
                n_levels: n,
                max_levels: MAX_DENSE_LEVELS
            })
        );
    }
}
