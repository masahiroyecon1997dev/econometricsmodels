//! 組合せが欠ける・次元が3つ以上のときの、固定効果ダミー（各次元の水準）だけの回帰のレバレッジ。

use faer::linalg::solvers::Solve;
use faer::{Mat, Side};

use super::error::PanelError;
use super::validation::covers_every_combination;
use crate::shared::error::{CommonError, MatrixKind};
use crate::shared::group_codes::GroupCodes;

/// 密に解く行列の次元（最大の次元を除いた次元の水準数の合計）の上限。`n × n`でなく`水準数²`のメモリと
/// `水準数³`の計算量を抑える（3000で行列1つ約72MB、`llt`と逆行列などが同時に生きるので実際の最大は
/// その数倍、計算は数秒）。
pub(super) const MAX_DENSE_LEVELS: usize = 3000;

/// 密に解く行列の次元: 最大の次元（消去する次元）を除いた次元の水準数の合計。
fn dense_dimension(levels: &[&GroupCodes]) -> usize {
    let total: usize = levels.iter().map(|level| level.n_groups()).sum();
    total
        - levels
            .iter()
            .map(|level| level.n_groups())
            .max()
            .unwrap_or(0)
}

/// 密な行列を解くHC2/HC3のレバレッジが、行列の次元の上限以下であることを確かめる。全組合せがそろう
/// 2次元（バランスパネル）は閉形式なので常に通る。`dummy_leverage`と同じ検査を、重い射影の前に行うために
/// 切り出している。
///
/// # Errors
/// 最大の次元を除いた水準数の合計が上限を超えるとき`PanelError::LeverageDimensionTooLarge`。
pub(crate) fn check_leverage_dimension(levels: &[&GroupCodes]) -> Result<(), PanelError> {
    let dense = dense_dimension(levels);
    let closed_form = matches!(levels, [entity, time] if covers_every_combination(entity, time));
    if dense > MAX_DENSE_LEVELS && !closed_form {
        return Err(PanelError::LeverageDimensionTooLarge {
            n_levels: dense,
            max_levels: MAX_DENSE_LEVELS,
        });
    }
    Ok(())
}

/// 各次元の水準ダミーだけの回帰（切片は水準ダミーに含まれる）のレバレッジ`h_ii`を、行ごとに返す。
///
/// 最大の次元`e`を先に消すと、`P = P_E + P_{M_E W}`（`W`は残りの次元の水準ダミーを並べた`n × L`の
/// 行列、`M_E W`は`W`から`e`のグループごとの平均を引いたもの）。行`i`の`M_E W`の行は
/// `w_i − c_{e_i} / n_{e_i}`（`w_i`は行`i`の水準を指す単位ベクトルの和、`c_g`はグループ`g`の行の`w`の和
/// ＝水準の出現回数のベクトル）で、`(M_E W)'(M_E W) = W'W − Σ_g c_g c_g' / n_g =: M`。
/// `M`は連結成分・次元間の別名（ネストを含む）でランク落ちするので、列ピボット付きQRでランクを
/// 見て独立な列`S`を選び、`A = (M[S,S])⁻¹`を使う（`R = M_E W`の独立な列`R_S`で`P = R_S (R_S'R_S)⁻¹ R_S'`）。
/// よって`h_ii = 1/n_e + r' A r`（`r = w_i − c_e / n_e`の`S`成分）
/// `= 1/n_e + Σ_{a,b} A[w_a,w_b] − 2 (A c_e)・w_i / n_e + c_e'A c_e / n_e²`。
/// 計算量は`O(n · (K−1)² + Σ_g d_g² + L³)`（`d_g`はグループ`g`の行に現れる異なる水準の数、`L`は密な行列の次元）。
///
/// 同じ組合せが複数の行にあってもよい（`c_g`は出現回数のベクトルで、式は回数についてそのまま成り立つ）。
/// グループごとに水準の出現回数を先に集約するので、計算量はグループ内の行数の2乗ではなく、
/// グループ内の異なる水準の数に依存する。
///
/// # Errors
/// - 密な行列の次元が[`MAX_DENSE_LEVELS`]を超えるとき`PanelError::LeverageDimensionTooLarge`
/// - 選んだ列の行列が正定値でない（連結が極端に細く、丸めで分解が失敗したとき）は
///   `CommonError::NearSingularMatrix`
pub(super) fn dummy_leverage(levels: &[&GroupCodes]) -> Result<Vec<f64>, PanelError> {
    check_leverage_dimension(levels)?;
    // 最大の次元を消す（同数なら先の次元）。
    let eliminated_at = levels.iter().enumerate().fold(0, |best, (i, level)| {
        if level.n_groups() > levels[best].n_groups() {
            i
        } else {
            best
        }
    });
    let eliminated = levels[eliminated_at];
    let others: Vec<&GroupCodes> = levels
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != eliminated_at)
        .map(|(_, &level)| level)
        .collect();
    // 残りの次元を1本の添字に並べる: 次元`d`の水準`c`は`offsets[d] + c`。
    let mut offsets = Vec::with_capacity(others.len());
    let mut dense = 0;
    for level in &others {
        offsets.push(dense);
        dense += level.n_groups();
    }
    let columns: Vec<(usize, &[usize])> = offsets
        .iter()
        .zip(&others)
        .map(|(&offset, level)| (offset, level.codes()))
        .collect();
    let n = eliminated.nobs();
    let rows = eliminated.group_indices();
    let sizes = eliminated.counts();

    // M = W'W − Σ_g c_g c_g' / n_g
    let mut m = Mat::<f64>::zeros(dense, dense);
    for r in 0..n {
        for &(offset_a, codes_a) in &columns {
            for &(offset_b, codes_b) in &columns {
                *m.get_mut(offset_a + codes_a[r], offset_b + codes_b[r]) += 1.0;
            }
        }
    }
    let scale = (0..dense).map(|i| *m.get(i, i)).fold(0.0_f64, f64::max);
    let mut tally = Tally::new(dense);
    for (group, group_rows) in rows.iter().enumerate() {
        let weight = 1.0 / sizes[group] as f64;
        let occurrences = tally.count(group_rows, &columns);
        for &(a, count_a) in occurrences {
            for &(b, count_b) in occurrences {
                *m.get_mut(a, b) -= weight * (count_a * count_b) as f64;
            }
        }
        tally.clear();
    }
    let independent = independent_columns(&m, scale);
    let rank = independent.len();
    let mut position = vec![usize::MAX; dense];
    for (k, &column) in independent.iter().enumerate() {
        position[column] = k;
    }
    let a = if rank == 0 {
        Mat::<f64>::zeros(0, 0)
    } else {
        let selected = Mat::from_fn(rank, rank, |i, j| *m.get(independent[i], independent[j]));
        let llt = selected
            .llt(Side::Lower)
            .map_err(|_| CommonError::NearSingularMatrix {
                matrix: MatrixKind::DesignGram,
            })?;
        llt.solve(Mat::<f64>::identity(rank, rank))
    };

    // グループごとに v = A c_g と c_g' A c_g / n_g² を作り、行の値を組み立てる。`v`は行の水準と
    // グループが触った水準でしか読まれないので、触った水準の位置だけ計算する（グループ内の異なる水準の数の2乗）。
    let mut leverage = vec![0.0; n];
    let mut v = vec![0.0; rank];
    for (group, group_rows) in rows.iter().enumerate() {
        let size = sizes[group] as f64;
        let occurrences = tally.count(group_rows, &columns);
        let mut quadratic = 0.0;
        for &(column, _) in occurrences {
            let ka = position[column];
            if ka == usize::MAX {
                continue;
            }
            let mut value = 0.0;
            for &(other, count) in occurrences {
                let kb = position[other];
                if kb != usize::MAX {
                    value += count as f64 * *a.get(ka, kb);
                }
            }
            v[ka] = value;
        }
        for &(column, count) in occurrences {
            let k = position[column];
            if k != usize::MAX {
                quadratic += count as f64 * v[k];
            }
        }
        quadratic /= size * size;
        for &r in group_rows {
            let mut value = 1.0 / size + quadratic;
            for &(offset_a, codes_a) in &columns {
                let ka = position[offset_a + codes_a[r]];
                if ka == usize::MAX {
                    continue;
                }
                value -= 2.0 * v[ka] / size;
                for &(offset_b, codes_b) in &columns {
                    let kb = position[offset_b + codes_b[r]];
                    if kb != usize::MAX {
                        value += *a.get(ka, kb);
                    }
                }
            }
            leverage[r] = value;
        }
        for &(column, _) in occurrences {
            let k = position[column];
            if k != usize::MAX {
                v[k] = 0.0;
            }
        }
        tally.clear();
    }
    // 射影行列の対角は[0, 1]に入る。外れるのは、ランクの判定を丸めが誤らせて`M[S,S]`が実質特異になったとき
    // （過大な`A`がそのまま成功する）なので、黙って返さずエラーにする。
    if leverage
        .iter()
        .any(|&h| !h.is_finite() || !(-LEVERAGE_SLACK..=1.0 + LEVERAGE_SLACK).contains(&h))
    {
        return Err(CommonError::NearSingularMatrix {
            matrix: MatrixKind::DesignGram,
        }
        .into());
    }
    Ok(leverage)
}

/// ダミーだけの回帰のレバレッジが`[0, 1]`から外れてよい丸めの余裕。
const LEVERAGE_SLACK: f64 = 1e-8;

/// 半正定値の対称行列`m`の独立な列の添字（列ピボット付きQRでランクを見て選ぶ）を昇順に返す。
/// 打ち切りは`次元 × 機械イプシロン × scale`（`scale`は消去前の行列の対角の最大＝水準の最大の行数で、
/// `M`の各成分はこれ以下）。`M`の丸め誤差は`scale`のオーダーの`ε`倍なので、真に従属な列の主元はこの
/// 打ち切りの下に収まり、真に独立な列の主元（連結が細くても`1/n_g`程度）はその上に残る。判定を誤っても、
/// 呼び出し側がレバレッジの値域（`[0, 1]`）を確かめてエラーにする。
fn independent_columns(m: &Mat<f64>, scale: f64) -> Vec<usize> {
    let size = m.ncols();
    if size == 0 {
        return Vec::new();
    }
    let qr = m.col_piv_qr();
    let r = qr.thin_R();
    let threshold = (size as f64) * f64::EPSILON * scale;
    let rank = (0..size)
        .take_while(|&i| (*r.get(i, i)).abs() > threshold)
        .count();
    let mut columns: Vec<usize> = qr.P().arrays().0[..rank].to_vec();
    columns.sort_unstable();
    columns
}

/// 1つのグループの行が、残りの次元の水準（1本に並べた添字）に何回ずつ現れるかの集計。
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

    /// `group_rows`の行ごとに、`columns`（次元ごとの`(添字の開始位置, 水準のコード)`）の水準を数え、
    /// 現れた水準と回数の組を初めて現れた順に返す。次のグループの前に[`clear`](Self::clear)を呼ぶ。
    fn count(&mut self, group_rows: &[usize], columns: &[(usize, &[usize])]) -> &[(usize, usize)] {
        for &r in group_rows {
            for &(offset, codes) in columns {
                let level = offset + codes[r];
                if self.counts[level] == 0 {
                    self.touched.push((level, 0));
                }
                self.counts[level] += 1;
            }
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

    /// 次元`sizes`の格子から確率`keep`で行を残す（各次元の水準は少なくとも1行に現れ、最初の行は必ず残す）。
    /// 戻り値は次元ごとのラベル。`nested`が`Some(parent)`なら最後の次元を`parent`番目の次元から決める
    /// （最後の水準数は`sizes`の最後で、親の水準を水準数で割った余りで対応づける）。
    fn grid_labels(
        sizes: &[usize],
        keep: f64,
        nested_in: Option<usize>,
        seed: u64,
    ) -> Vec<Vec<String>> {
        let mut state = seed;
        let mut columns = vec![Vec::new(); sizes.len()];
        let total: usize = sizes.iter().product();
        for cell in 0..total {
            if cell > 0 && uniform(&mut state) >= keep {
                continue;
            }
            let mut rest = cell;
            let mut codes = Vec::with_capacity(sizes.len());
            for &size in sizes {
                codes.push(rest % size);
                rest /= size;
            }
            if let Some(parent) = nested_in {
                let last = sizes.len() - 1;
                codes[last] = codes[parent] % sizes[last];
            }
            for (column, code) in columns.iter_mut().zip(codes) {
                column.push(format!("g{code:03}"));
            }
        }
        columns
    }

    fn codes_of(columns: &[Vec<String>]) -> Vec<GroupCodes> {
        columns
            .iter()
            .map(|labels| GroupCodes::from_labels(labels))
            .collect()
    }

    /// ダミー回帰のハット行列の対角。ダミーを並べた`Z`の特異値分解で、列空間の正規直交基底から求める
    /// （ランク落ち・ネストにも依らない）。
    fn brute_force(levels: &[&GroupCodes]) -> Vec<f64> {
        let n = levels[0].nobs();
        let columns: usize = levels.iter().map(|level| level.n_groups()).sum();
        let z = Mat::from_fn(n, columns, |r, c| {
            let mut start = 0;
            for level in levels {
                if c < start + level.n_groups() {
                    return f64::from(u8::from(level.codes()[r] == c - start));
                }
                start += level.n_groups();
            }
            0.0
        });
        let svd = z.thin_svd().unwrap();
        let s = svd.S().column_vector();
        let largest = *s.get(0);
        let rank = (0..s.nrows())
            .filter(|&j| *s.get(j) > largest * 1e-10)
            .count();
        (0..n)
            .map(|i| (0..rank).map(|j| svd.U().get(i, j).powi(2)).sum())
            .collect()
    }

    fn assert_matches_brute_force(levels: &[&GroupCodes]) {
        let got = dummy_leverage(levels).unwrap();
        let expected = brute_force(levels);
        for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
            assert!((a - b).abs() < 1e-9, "row {i}: {a} vs {b}");
        }
    }

    fn check(blocks: usize, n_entities: usize, n_periods: usize, keep: f64, seed: u64) {
        let (entity, time) = labels(blocks, n_entities, n_periods, keep, seed);
        let (entity, time) = (
            GroupCodes::from_labels(&entity),
            GroupCodes::from_labels(&time),
        );
        assert_matches_brute_force(&[&entity, &time]);
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
        assert_matches_brute_force(&[&entity, &time]);
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
        let got = dummy_leverage(&[&entity, &time]).unwrap();
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
        let got = dummy_leverage(&[&entity, &time]).unwrap();
        for (r, &h) in got.iter().enumerate() {
            let expected = 1.0 / entity.counts()[entity.codes()[r]] as f64
                + 1.0 / time.counts()[time.codes()[r]] as f64
                - 1.0 / n;
            assert!((h - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn it_matches_the_hat_matrix_with_three_dimensions() {
        for (sizes, keep, seed) in [
            (vec![9, 5, 3], 0.6, 11),
            (vec![4, 12, 3], 0.5, 12),
            (vec![5, 4, 6], 0.4, 13),
        ] {
            let columns = codes_of(&grid_labels(&sizes, keep, None, seed));
            let levels: Vec<&GroupCodes> = columns.iter().collect();
            assert_matches_brute_force(&levels);
        }
    }

    #[test]
    fn it_matches_the_closed_form_of_a_complete_grid_with_three_dimensions() {
        // 格子の全ての組合せが1回ずつ: 直交なので h = Σ 1/N_d − (K−1)/n。
        let columns = codes_of(&grid_labels(&[4, 3, 5], 1.1, None, 14));
        let levels: Vec<&GroupCodes> = columns.iter().collect();
        let n = levels[0].nobs() as f64;
        let got = dummy_leverage(&levels).unwrap();
        for (r, &h) in got.iter().enumerate() {
            let expected: f64 = levels
                .iter()
                .map(|level| 1.0 / level.counts()[level.codes()[r]] as f64)
                .sum::<f64>()
                - 2.0 / n;
            assert!((h - expected).abs() < 1e-12, "row {r}: {h} vs {expected}");
        }
    }

    #[test]
    fn it_matches_the_hat_matrix_when_a_dimension_is_nested_in_another() {
        // 3つ目の次元が1つ目にネストする（3つ目のダミーは1つ目のダミーの和で、ランクが落ちる）。
        let columns = codes_of(&grid_labels(&[8, 5, 4], 0.7, Some(0), 15));
        let levels: Vec<&GroupCodes> = columns.iter().collect();
        assert_matches_brute_force(&levels);
    }

    #[test]
    fn it_matches_the_hat_matrix_when_two_dimensions_other_than_the_largest_are_nested() {
        // 最大の次元（8水準）は消去され、残りの2つ（5水準と4水準）の間でネストする。ネストした次元の列は
        // `M`で零行ではなく、他の列の線形結合になる（ランクの判定が試される）。
        for seed in [20, 21, 22] {
            let columns = codes_of(&grid_labels(&[8, 5, 4], 0.7, Some(1), seed));
            let levels: Vec<&GroupCodes> = columns.iter().collect();
            assert_matches_brute_force(&levels);
        }
    }

    #[test]
    fn it_matches_the_hat_matrix_with_tied_largest_dimensions_and_a_single_level() {
        for (sizes, seed) in [
            (vec![5, 5, 5], 23),
            (vec![6, 1, 4], 24),
            (vec![1, 5, 5], 25),
        ] {
            let columns = codes_of(&grid_labels(&sizes, 0.6, None, seed));
            let levels: Vec<&GroupCodes> = columns.iter().collect();
            assert_matches_brute_force(&levels);
        }
    }

    #[test]
    fn it_matches_the_hat_matrix_when_one_level_has_far_more_rows_than_the_others() {
        // 1つの水準だけ行数が桁違いに多い（`M`の成分のスケールが大きい）うえで、行数の少ない水準が
        // 成分をつなぐ。ランクの打ち切りが、細いつながりを落とさない。
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();
        for r in 0..3000 {
            a.push(format!("a{}", r % 40));
            b.push("b0".to_string());
            c.push(format!("c{}", r % 3));
        }
        for (x, y, z) in [
            ("a40", "b1", "c0"),
            ("a40", "b1", "c1"),
            ("a41", "b1", "c0"),
            ("a41", "b0", "c1"),
            ("a42", "b2", "c2"),
            ("a42", "b2", "c0"),
        ]
        .into_iter()
        {
            a.push(x.to_string());
            b.push(y.to_string());
            c.push(z.to_string());
        }
        let columns = codes_of(&[a, b, c]);
        let levels: Vec<&GroupCodes> = columns.iter().collect();
        assert_matches_brute_force(&levels);
    }

    #[test]
    fn it_matches_the_hat_matrix_with_three_dimensions_and_repeated_rows() {
        let mut columns = grid_labels(&[7, 4, 3], 0.55, None, 16);
        let n = columns[0].len();
        for source in [0, 3, n / 2, n - 1, 3, 3] {
            for column in &mut columns {
                let value = column[source].clone();
                column.push(value);
            }
        }
        let columns = codes_of(&columns);
        let levels: Vec<&GroupCodes> = columns.iter().collect();
        assert_matches_brute_force(&levels);
    }

    #[test]
    fn it_matches_the_hat_matrix_with_three_dimensions_split_into_components() {
        // 重ならない2つのブロックを並べる（連結成分が2つ）。
        let first = grid_labels(&[6, 4, 3], 0.6, None, 17);
        let second = grid_labels(&[5, 3, 3], 0.6, None, 18);
        let columns: Vec<Vec<String>> = first
            .iter()
            .zip(&second)
            .enumerate()
            .map(|(d, (a, b))| {
                a.iter()
                    .map(|v| format!("a{d}{v}"))
                    .chain(b.iter().map(|v| format!("b{d}{v}")))
                    .collect()
            })
            .collect();
        let columns = codes_of(&columns);
        let levels: Vec<&GroupCodes> = columns.iter().collect();
        assert_matches_brute_force(&levels);
    }

    #[test]
    fn it_matches_the_hat_matrix_with_four_dimensions() {
        let columns = codes_of(&grid_labels(&[6, 4, 3, 3], 0.4, None, 19));
        let levels: Vec<&GroupCodes> = columns.iter().collect();
        assert_matches_brute_force(&levels);
    }

    #[test]
    fn it_gives_the_reciprocal_group_size_when_the_other_dimensions_add_nothing() {
        // 時点が個体にネストする2-way: 時点のダミーは個体のダミーに含まれる（M = 0、ランク0）。
        let to_labels =
            |values: &[&str]| -> Vec<String> { values.iter().map(ToString::to_string).collect() };
        let entity =
            GroupCodes::from_labels(&to_labels(&["a", "a", "a", "b", "b", "c", "c", "c", "c"]));
        let time = GroupCodes::from_labels(&to_labels(&[
            "a1", "a2", "a1", "b1", "b1", "c1", "c2", "c1", "c3",
        ]));
        assert_matches_brute_force(&[&entity, &time]);
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
            dummy_leverage(&[&entity, &time]),
            Err(PanelError::LeverageDimensionTooLarge {
                n_levels: n,
                max_levels: MAX_DENSE_LEVELS
            })
        );
    }

    #[test]
    fn the_limit_counts_every_dimension_except_the_largest() {
        // 3つの次元: 水準数は 5000 / 2000 / 1500。最大を除いた合計 3500 が上限を超える。
        let rows = 5000;
        let a: Vec<String> = (0..rows).map(|r| format!("a{r:05}")).collect();
        let b: Vec<String> = (0..rows).map(|r| format!("b{:05}", r % 2000)).collect();
        let c: Vec<String> = (0..rows).map(|r| format!("c{:05}", r % 1500)).collect();
        let (a, b, c) = (
            GroupCodes::from_labels(&a),
            GroupCodes::from_labels(&b),
            GroupCodes::from_labels(&c),
        );
        assert_eq!(
            check_leverage_dimension(&[&a, &b, &c]),
            Err(PanelError::LeverageDimensionTooLarge {
                n_levels: 3500,
                max_levels: MAX_DENSE_LEVELS
            })
        );
    }
}
