//! 分離の有無を決める線形計画（制約生成）と、その閾値。

use faer::Mat;
use std::collections::HashSet;

use super::SeparationVerdict;
use super::rows::SignedRows;
use crate::nonlinear::common::{ColumnScale, MleError};
use crate::shared::linear_program::{BoxBoundedLp, LinearProgramSolver};

/// LPの解が制約を破っているとみなす閾値（単位長にそろえた行との余弦）。
const VIOLATION_TOLERANCE: f64 = 1e-9;

/// 作業集合内の行が`VIOLATION_TOLERANCE`を超えて違反していても、この値以内なら単体法の
/// 数値誤差とみなして許容する。これを超えたら数値的な異常として扱う。
const WORKING_SET_NOISE_TOLERANCE: f64 = 1e-6;

/// LPの解の最大マージンがこれを超えたら分離ありとする。正常なデータでは解は原点（雑音を
/// 含めても`1e-12`程度）で、分離があれば分離した観測に`O(1)`のマージンが付く。
const SEPARATION_MARGIN_THRESHOLD: f64 = 1e-6;

/// 制約生成の1ラウンドで作業集合に足す、違反の大きい制約の最大本数。
const ROWS_ADDED_PER_ROUND: usize = 64;

/// 作業集合の行数の上限（単体法のタブローが`行数×(2·変数+行数)`に膨らむため）。実測の
/// 作業集合は数百本以下。到達したら数値的な異常として扱う。
const MAX_WORKING_ROWS: usize = 2000;

/// 制約生成のラウンド数の上限（実測は3〜4ラウンド。到達したら数値的な異常として扱う）。
const MAX_ROUNDS: usize = 200;

/// 分離の有無を厳密に判定する（制約生成によるLP、`separation.rs`のモジュールdocコメント参照）。
pub(super) fn detect_separation<S: LinearProgramSolver>(
    x: &Mat<f64>,
    scale: &ColumnScale,
    y: &Mat<f64>,
    solver: &S,
) -> Result<SeparationVerdict, MleError> {
    let rows = SignedRows::new(x, scale, y);
    if lp_finds_separation(&rows, solver)? {
        Ok(SeparationVerdict::Separated)
    } else {
        Ok(SeparationVerdict::NotSeparated)
    }
}

fn numerical_failure(detail: &str) -> MleError {
    MleError::SeparationCheckFailed {
        detail: detail.to_string(),
    }
}

/// 制約生成でLPを解き、分離がある（解の最大マージンが閾値超）かを返す。
fn lp_finds_separation<S: LinearProgramSolver>(
    rows: &SignedRows<'_>,
    solver: &S,
) -> Result<bool, MleError> {
    let c = rows.objective();
    let c_scale = c.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if c_scale == 0.0 {
        // `Σuᵢ = 0`。分離があれば`c'θ = Σ uᵢ'θ > 0`となるはずなので、分離は無い。
        return Ok(false);
    }
    let scaled_c: Vec<f64> = c.iter().map(|v| v / c_scale).collect();

    let mut working_rows: Vec<Vec<f64>> = Vec::new();
    let mut in_working_set: HashSet<usize> = HashSet::new();

    for _ in 0..MAX_ROUNDS {
        let theta = if working_rows.is_empty() {
            // 制約が無いときの最適解は、目的関数の符号に沿った箱の頂点。
            scaled_c
                .iter()
                .map(|v| v.signum() * f64::from(*v != 0.0))
                .collect()
        } else {
            let rhs = vec![0.0; working_rows.len()];
            solver
                .maximize(&BoxBoundedLp {
                    objective: &scaled_c,
                    constraint_rows: &working_rows,
                    rhs: &rhs,
                    bound: 1.0,
                })
                .map_err(|e| numerical_failure(&e.to_string()))?
        };

        let margins = rows.margins(&theta);
        let mut new_violations: Vec<(f64, usize)> = Vec::new();
        let mut worst_in_set = 0.0_f64;
        let mut max_margin = f64::NEG_INFINITY;
        for (i, &m) in margins.iter().enumerate() {
            if !rows.is_active(i) {
                continue;
            }
            max_margin = max_margin.max(m);
            if m < -VIOLATION_TOLERANCE {
                if in_working_set.contains(&i) {
                    worst_in_set = worst_in_set.min(m);
                } else {
                    new_violations.push((m, i));
                }
            }
        }

        if new_violations.is_empty() {
            // 作業集合内の行が許容を大きく超えて違反しているのは、単体法の数値的な異常。
            if worst_in_set < -WORKING_SET_NOISE_TOLERANCE {
                return Err(numerical_failure(
                    "the linear program solution violates its own constraints",
                ));
            }
            return Ok(max_margin > SEPARATION_MARGIN_THRESHOLD);
        }
        if working_rows.len() + new_violations.len().min(ROWS_ADDED_PER_ROUND) > MAX_WORKING_ROWS {
            return Err(numerical_failure("the working set grew too large"));
        }
        if new_violations.len() > ROWS_ADDED_PER_ROUND {
            new_violations
                .select_nth_unstable_by(ROWS_ADDED_PER_ROUND - 1, |a, b| a.0.total_cmp(&b.0));
            new_violations.truncate(ROWS_ADDED_PER_ROUND);
        }
        for &(_, i) in &new_violations {
            in_working_set.insert(i);
            // 制約 `uᵢ'θ ≥ 0` を `-uᵢ'θ ≤ 0` の形で渡す。
            working_rows.push(rows.unit_row(i).into_iter().map(|v| -v).collect());
        }
    }
    Err(numerical_failure(
        "the constraint generation did not converge",
    ))
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;

    use super::*;

    use crate::nonlinear::common::standardize_columns;

    use crate::shared::linear_program::DenseSimplex;
    use crate::shared::linear_program::LpError;

    fn verdict(x: &Mat<f64>, y: &Mat<f64>, has_intercept: bool) -> SeparationVerdict {
        let (_, scale) = standardize_columns(x, has_intercept);
        detect_separation(x, &scale, y, &DenseSimplex).unwrap()
    }

    fn is_separated(x: &Mat<f64>, y: &Mat<f64>) -> bool {
        verdict(x, y, true) == SeparationVerdict::Separated
    }

    /// 簡易な線形合同法（テストの再現性のため外部の乱数に依存しない）。
    struct Lcg(u64);

    impl Lcg {
        fn next_unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    /// `k`変数の連続変数（切片込みで`k+1`列）で、`y = 1[Σxⱼ > 0]`の完全分離データ。
    fn complete_separation_data(n: usize, k: usize, seed: u64) -> (Mat<f64>, Mat<f64>) {
        let mut rng = Lcg(seed);
        let cols: Vec<Vec<f64>> = (0..k)
            .map(|_| (0..n).map(|_| rng.next_unit()).collect())
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| f64::from(cols.iter().map(|c| c[i]).sum::<f64>() > 0.0))
            .collect();
        let refs: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
        (design(&refs), response(&y))
    }

    /// 上のデータに、各変数について「同じ`x`で`y=0`と`y=1`の両方」を足して分離を壊す。どの方向
    /// `θ`でも`θ`と`-θ`の両方の符号条件を課されるため、許容解は原点だけ（分離なし、確実）。
    fn broken_separation_data(n: usize, k: usize, seed: u64) -> (Mat<f64>, Mat<f64>) {
        let (x, y) = complete_separation_data(n, k, seed);
        let extra = 2 * k;
        let total = n + extra;
        // 追加行: 切片1、軸`e_j`上の点（`j`番目の傾き列だけ1）。
        let xx = Mat::from_fn(total, k + 1, |i, j| {
            if i < n {
                *x.get(i, j)
            } else {
                f64::from(j == 0 || (i - n) / 2 == j - 1)
            }
        });
        let yy = Mat::from_fn(total, 1, |i, _| {
            if i < n {
                *y.get(i, 0)
            } else {
                f64::from((i - n).is_multiple_of(2))
            }
        });
        (xx, yy)
    }

    #[test]
    fn detects_complete_separation() {
        let x = design(&[&[-2.0, -1.0, 1.0, 2.0]]);
        let y = response(&[0.0, 0.0, 1.0, 1.0]);
        assert!(is_separated(&x, &y));
    }

    #[test]
    fn does_not_flag_overlapping_data() {
        let x = design(&[&[-2.0, -1.0, 1.0, 2.0, 0.5, -0.5]]);
        let y = response(&[0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        assert!(!is_separated(&x, &y));
    }

    #[test]
    fn detects_quasi_complete_separation_from_a_dummy_regressor() {
        let (x, y) = quasi_complete_dummy_data(60);
        assert!(is_separated(&x, &y));
    }

    #[test]
    fn detects_separation_caused_by_a_single_observation_at_large_n() {
        // 1観測だけが d=1 で y=1。他は d=0 で y が混在する。nが大きくても検出できる。
        for n in [300, 5000] {
            let d: Vec<f64> = (0..n).map(|i| f64::from(i == 0)).collect();
            let x: Vec<f64> = (0..n)
                .map(|i| ((i * 37 % 101) as f64) / 50.0 - 1.0)
                .collect();
            let y: Vec<f64> = (0..n).map(|i| f64::from(i == 0 || i % 2 == 0)).collect();
            assert!(is_separated(&design(&[&d, &x]), &response(&y)), "n={n}");
        }
    }

    #[test]
    fn does_not_flag_a_dummy_that_has_both_outcomes() {
        let n = 60;
        let d: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 37 % 101) as f64) / 50.0 - 1.0)
            .collect();
        let y: Vec<f64> = (0..n).map(|i| f64::from(i % 2 == 0)).collect();
        assert!(!is_separated(&design(&[&d, &x]), &response(&y)));
    }

    /// 完全分離データ（`x`が`±0.01〜±1.00`の0.01刻み、`x>0`なら`y=1`）に、`y=1`の観測を1つ
    /// `x=-0.01-overlap`に足す。`y=0`で最も`x=0`に近い観測（`x=-0.01`）より左に`y=1`が
    /// 1つあるため、`overlap>0`ならちょうど1観測が分離を壊す（有限のMLEが存在する）。
    fn single_overlap_data(overlap: f64) -> (Mat<f64>, Mat<f64>) {
        let grid: Vec<f64> = (1..=100).map(|i| 0.01 * f64::from(i)).collect();
        let mut x: Vec<f64> = grid.iter().map(|g| -g).collect();
        x.extend(grid.iter().copied());
        x.push(-0.01 - overlap);
        let mut y = vec![0.0; 100];
        y.extend(vec![1.0; 101]);
        (design(&[&x]), response(&y))
    }

    /// 1観測だけが分離を壊す境界の両側。重なりが大きければ分離とみなさず（有限のMLEが
    /// 存在し推定が進む）、`1e-6`未満のマージンしか持たないほど小さければ分離とみなす
    /// （`SEPARATION_MARGIN_THRESHOLD`）。実測した境界はこのデータで重なり`1e-5`〜`3e-5`の間
    /// （標準化後の幾何に依存するためデータごとに変わる）。境界そのものではなく、
    /// 明らかに両側にある値を固定する。
    #[test]
    fn single_overlapping_observation_is_flagged_only_when_the_overlap_is_tiny() {
        for overlap in [1e-1, 1e-2, 1e-3, 1e-4] {
            let (x, y) = single_overlap_data(overlap);
            assert!(!is_separated(&x, &y), "overlap={overlap}");
        }
        for overlap in [1e-7, 1e-9, 1e-12] {
            let (x, y) = single_overlap_data(overlap);
            assert!(is_separated(&x, &y), "overlap={overlap}");
        }
    }

    /// 分離方向が複数ある（錐の次元が2以上）準完全分離。d0・d2が1なら必ず`y=1`、d1が1なら
    /// 必ず`y=0`。かつては情報行列の最小固有ベクトルを使う事前スクリーニングが、錐の外を向いて
    /// 取りこぼすことがあった（lbfgsで実測）。常にLPで確かめるのでこの構成でも検出できる。
    #[test]
    fn detects_quasi_complete_separation_with_several_dummies() {
        let n = 600;
        let mut rng = Lcg(7);
        let d0: Vec<f64> = (0..n).map(|_| f64::from(rng.next_unit() > 0.8)).collect();
        let d1: Vec<f64> = (0..n).map(|_| f64::from(rng.next_unit() > 0.8)).collect();
        let d2: Vec<f64> = (0..n).map(|_| f64::from(rng.next_unit() > 0.9)).collect();
        let c: Vec<Vec<f64>> = (0..5)
            .map(|_| (0..n).map(|_| rng.next_unit()).collect())
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                if d0[i] == 1.0 || d2[i] == 1.0 {
                    1.0
                } else if d1[i] == 1.0 {
                    0.0
                } else {
                    f64::from(rng.next_unit() + c[0][i] > 0.0)
                }
            })
            .collect();
        let mut cols: Vec<&[f64]> = vec![&d0, &d1, &d2];
        cols.extend(c.iter().map(|v| v.as_slice()));
        assert!(is_separated(&design(&cols), &response(&y)));
    }

    #[test]
    fn handles_many_regressors_without_numerical_failure() {
        // 変数が多いと単体法が退化して失敗した（Bland規則のみの旧実装でk>=10）。
        for k in [10, 20, 30] {
            for seed in 1..=3 {
                let (x, y) = complete_separation_data(2000, k, seed);
                assert!(
                    is_separated(&x, &y),
                    "complete separation: k={k}, seed={seed}"
                );
                let (x, y) = broken_separation_data(2000, k, seed);
                assert!(
                    !is_separated(&x, &y),
                    "broken separation must not be flagged: k={k}, seed={seed}"
                );
            }
        }
    }

    /// 平均が標準偏差より大きい列（年齢・金融資産のような正の量）を含む混合スケールの説明変数
    /// `k`列で、`y`は重なりのある二値（分離しない）。標準偏差で割るだけの行は平均の大きい列に
    /// 支配されて互いにほぼ平行になり、錐制約の許容領域が原点だけになる。
    fn mixed_scale_overlapping_data(n: usize, k: usize, seed: u64) -> (Mat<f64>, Mat<f64>) {
        let mut rng = Lcg(seed);
        // 近似的に標準正規（一様4個の和を基準化）。
        let mut normal =
            move || (0..4).map(|_| rng.next_unit()).sum::<f64>() * 3.0_f64.sqrt() / 2.0;
        let mut cols: Vec<Vec<f64>> = Vec::with_capacity(k);
        let mut z: Vec<Vec<f64>> = Vec::with_capacity(k);
        for j in 0..k {
            let (mean, sd) = (10.0 + 7.0 * j as f64, 1.0 + 2.0 * (j % 5) as f64);
            let zj: Vec<f64> = (0..n).map(|_| normal()).collect();
            let xj: Vec<f64> = zj
                .iter()
                .map(|&v| match j % 4 {
                    0 => mean + sd * v,
                    1 => (0.5 * v).exp() * mean,
                    2 => f64::from(v > 0.5),
                    _ => (mean + sd * v).abs(),
                })
                .collect();
            cols.push(xj);
            z.push(zj);
        }
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let index: f64 = z.iter().map(|zj| 0.3 * zj[i]).sum::<f64>() / (k as f64).sqrt();
                f64::from(index + 1.5 * normal() - 0.8 > 0.0)
            })
            .collect();
        let refs: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
        (design(&refs), response(&y))
    }

    /// 分離のないデータで、列数や列の組み合わせによって単体法が終わらず`SeparationCheckFailed`に
    /// なっていた。この種のLPは許容領域が原点だけで、右辺の摂動（`1e-11`級）が領域の大きさそのもの
    /// になる。比検定の同点許容幅が摂動と同じ桁だと、別々の比を同点と扱って退出行を誤り、
    /// ピボット上限に達していた（`shared/linear_program.rs`の`RATIO_TIE_TOLERANCE`）。
    #[test]
    fn does_not_fail_on_mixed_scale_data_without_separation() {
        for k in [10, 14, 20, 24] {
            for seed in 1..=4 {
                let (x, y) = mixed_scale_overlapping_data(3000, k, seed);
                let (_, scale) = standardize_columns(&x, true);
                let result = detect_separation(&x, &scale, &y, &DenseSimplex);
                assert_eq!(
                    result.as_ref().map_err(|e| e.to_string()),
                    Ok(&SeparationVerdict::NotSeparated),
                    "k={k}, seed={seed}"
                );
            }
        }
    }

    #[test]
    fn works_without_an_intercept() {
        let x = Mat::from_fn(6, 1, |i, _| [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0][i]);
        let y = response(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(verdict(&x, &y, false), SeparationVerdict::Separated);
        let y_mixed = response(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        assert_eq!(
            verdict(&x, &y_mixed, false),
            SeparationVerdict::NotSeparated
        );
    }

    #[test]
    fn works_with_a_single_regressor() {
        // k=1（切片のみ）: 全て同じ`y`なら切片だけで完全に予測できる（分離）、混在なら分離なし。
        let x = Mat::from_fn(5, 1, |_, _| 1.0);
        assert!(is_separated(&x, &response(&[1.0; 5])));
        assert!(!is_separated(&x, &response(&[1.0, 0.0, 1.0, 0.0, 1.0])));
    }

    #[test]
    fn ignores_all_zero_rows() {
        // 切片なしで x=0 の行は制約にならない。残りは重なっているので分離なし。
        let x = Mat::from_fn(5, 1, |i, _| [0.0, 1.0, -1.0, 2.0, -2.0][i]);
        let y = response(&[1.0, 1.0, 1.0, 0.0, 0.0]);
        let (_, scale) = standardize_columns(&x, false);
        let rows = SignedRows::new(&x, &scale, &y);
        assert!(!rows.is_active(0));
        assert!(!lp_finds_separation(&rows, &DenseSimplex).unwrap());
    }

    /// 制約生成（作業集合を段階的に増やす）の判定が、全行を最初から制約に入れて一度に解いた
    /// 判定と一致すること。違反行が`ROWS_ADDED_PER_ROUND`より多く、複数ラウンドが必要なデータで
    /// 比べる。
    #[test]
    fn constraint_generation_matches_solving_with_all_rows_at_once() {
        let mut rng = Lcg(99);
        for trial in 0..12 {
            let n = 400;
            let k = 3 + trial % 4;
            // 半分は完全分離に近づけ（ラベルを少数反転）、残りは混在させる。
            let (x, y) = complete_separation_data(n, k, 100 + trial as u64);
            let flips = if trial % 2 == 0 { 0 } else { 5 };
            let y = Mat::from_fn(n, 1, |i, _| {
                let v = *y.get(i, 0);
                if i < flips { 1.0 - v } else { v }
            });
            let _ = rng.next_unit();
            let (_, scale) = standardize_columns(&x, true);
            let rows = SignedRows::new(&x, &scale, &y);
            let by_generation = lp_finds_separation(&rows, &DenseSimplex).unwrap();

            // 全行を一度に制約に入れて解く。
            let all_rows: Vec<Vec<f64>> = (0..n)
                .filter(|&i| rows.is_active(i))
                .map(|i| rows.unit_row(i).into_iter().map(|v| -v).collect())
                .collect();
            let c = rows.objective();
            let c_scale = c.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let scaled: Vec<f64> = c.iter().map(|v| v / c_scale).collect();
            let rhs = vec![0.0; all_rows.len()];
            let theta = DenseSimplex
                .maximize(&BoxBoundedLp {
                    objective: &scaled,
                    constraint_rows: &all_rows,
                    rhs: &rhs,
                    bound: 1.0,
                })
                .unwrap();
            let max_margin = rows
                .margins(&theta)
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max);
            let all_at_once = max_margin > SEPARATION_MARGIN_THRESHOLD;
            assert_eq!(by_generation, all_at_once, "trial {trial} (k={k})");
        }
    }

    /// 作業集合の行だけが違反している（他の行は違反なし）数値状況の挙動。
    /// ソルバーが制約を大きく破った解を返したら、違反なしと誤判定せず数値的な異常として扱う。
    struct ViolatingSolver;

    impl LinearProgramSolver for ViolatingSolver {
        fn maximize(&self, lp: &BoxBoundedLp<'_>) -> Result<Vec<f64>, LpError> {
            // 目的関数と逆向きの解（作業集合の制約を必ず破る）を返す。
            Ok(lp.objective.iter().map(|v| -v.signum()).collect())
        }
    }

    #[test]
    fn a_solver_that_violates_its_own_constraints_is_a_numerical_failure() {
        let (x, y) = quasi_complete_dummy_data(60);
        let (_, scale) = standardize_columns(&x, true);
        let rows = SignedRows::new(&x, &scale, &y);
        let err = lp_finds_separation(&rows, &ViolatingSolver).unwrap_err();
        assert!(matches!(err, MleError::SeparationCheckFailed { .. }));
    }

    #[test]
    fn lp_failure_is_reported_as_a_computation_error_that_says_separation_was_not_ruled_out() {
        let (x, y) = quasi_complete_dummy_data(60);
        let (_, scale) = standardize_columns(&x, true);
        let err = detect_separation(&x, &scale, &y, &FailingSolver).unwrap_err();
        match err {
            MleError::SeparationCheckFailed { detail } => {
                assert!(!detail.is_empty());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
