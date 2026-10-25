//! 小規模な線形計画（LP）ソルバー。系統（`linear`/`nonlinear`等）に依存しない純粋な数値部品
//! （`.claude/rules/rust-style.md`「全手法で共有するロジック」参照）。
//!
//! 現状の唯一の利用者はnonlinear系統の完全/準完全分離の検出（`nonlinear::separation`）で、
//! 「変数が数個〜数十個・制約が数百本程度」の小さなLPを何度も解く用途に特化している。
//! 変数が数十個でも数値的に破綻しないことをテストで固定している（k=30まで）。LPの
//! 解き方を差し替えられるよう、利用側は[`LinearProgramSolver`]トレイトだけに依存する。
//! 自前の密な単体法[`DenseSimplex`]を、将来外部のLPクレートによる実装へ置き換えるときは、
//! トレイトの実装を1つ足して既定の実装を切り替えるだけでよい（利用側の変更は不要）。
//!
//! 返すエラー[`LpError`]は系統非依存の型で、呼び出し側が`map_err`で自系統のエラーへ変換する
//! （`linear_algebra::RankDeficient`と同じ方針）。

use thiserror::Error;

/// [`LinearProgramSolver`]が解く問題。
///
/// ```text
/// maximize    c'x
/// subject to  A x <= b          （b >= 0）
///             -bound <= x_j <= bound   （全ての変数）
/// ```
///
/// `b >= 0`を要求するのは、原点`x = 0`が必ず許容解になり、単体法がPhase 1（実行可能な出発点
/// 探し）なしで始められるため。分離検出の錐制約`Zθ >= 0`（`-Zθ <= 0`）はこの形をしている。
#[derive(Debug, Clone, Copy)]
pub struct BoxBoundedLp<'a> {
    /// 目的関数の係数`c`（長さ`k`）。
    pub objective: &'a [f64],
    /// 制約行列`A`の行（各行の長さは`k`）。
    pub constraint_rows: &'a [Vec<f64>],
    /// 制約の右辺`b`（`constraint_rows`と同じ長さ、全要素が非負）。
    pub rhs: &'a [f64],
    /// 各変数の絶対値の上限（正の有限値）。
    pub bound: f64,
}

/// [`LinearProgramSolver`]が返すエラー。
#[derive(Debug, Clone, PartialEq, Error)]
pub enum LpError {
    /// 問題の次元が一致しない（制約行の長さが目的関数の長さと違う、`rhs`の長さが行数と違う）。
    #[error("linear program dimensions are inconsistent")]
    DimensionMismatch,

    /// 入力に非有限値（NaN・無限大）を含む、または`bound`が正の有限値でない。
    #[error("linear program input contains a non-finite value or a non-positive bound")]
    InvalidInput,

    /// `rhs`に負の値がある（原点が許容解でない問題は扱えない、[`BoxBoundedLp`]参照）。
    #[error("linear program requires a non-negative right-hand side (origin must be feasible)")]
    OriginInfeasible,

    /// ピボット回数が上限に達した（数値的な退化。通常は到達しない）。
    #[error("linear program did not finish within {pivots} simplex pivots")]
    IterationLimit { pivots: usize },

    /// ピボット行が選べなかった。全変数が箱制約で有界なので理論上は起きず、数値的な退化の
    /// ときだけ起きる。[`IterationLimit`](Self::IterationLimit)とは原因が違うため区別する。
    #[error("linear program hit a numerically degenerate pivot (no leaving row)")]
    DegeneratePivot,
}

/// [`BoxBoundedLp`]を解くソルバー。
pub trait LinearProgramSolver {
    /// 最適解`x`（長さ`k`、各成分は`[-bound, bound]`内）を返す。最適解が複数あるときは
    /// どれを返すかは実装に依存する。
    fn maximize(&self, lp: &BoxBoundedLp<'_>) -> Result<Vec<f64>, LpError>;
}

/// 密なタブローを使う単体法（Bland規則）。
///
/// 変数を`x = p - q`（`p, q >= 0`）に分け、箱制約を`p_j <= bound`・`q_j <= bound`の行として
/// 加えて標準形にする。`rhs >= 0`なのでスラック変数を基底にした原点が許容基底解になり、
/// Phase 1が要らない。分離検出の錐制約は右辺が全て0で強く退化する。
///
/// **ピボット規則**: 通常は、入る変数に最も負の簡約費用（Dantzig規則）を、比検定の同点では
/// 絶対値最大のピボット要素を選ぶ。Bland規則（添字最小）だけだと、原点付近で雑音程度の簡約費用の
/// 変数を入れ、極小のピボットで消去を繰り返して誤差が溜まり、`k>=10`でピボット上限に達した
/// （実測）。一方Dantzig規則は退化頂点で巡回しうるため、退化ピボット（右辺が0のままの
/// ピボット）が`DEGENERATE_PIVOTS_BEFORE_BLAND`回続いたらBland規則に切り替えて巡回を断ち、
/// 非退化ピボットが出たらDantzig規則に戻す。比の同点は絶対許容幅`RATIO_TIE_TOLERANCE`で判定する
/// （右辺が`1e-16`級の非ゼロになると厳密比較の同点判定が崩れるため）。さらに、制約の右辺には
/// 行ごとに異なる`1e-11`級の決定的な摂動を足して退化そのものを解く（`rhs_perturbation`）。
#[derive(Debug, Clone, Copy, Default)]
pub struct DenseSimplex;

/// 目的関数行の簡約費用がこの値より負なら「改善余地あり」とみなす。
const OPTIMALITY_TOLERANCE: f64 = 1e-10;
/// ピボット要素として許す正の値の下限（ゼロ割・桁落ちを避ける）。
const PIVOT_TOLERANCE: f64 = 1e-11;
/// ピボット回数の上限を決める係数（`行数+列数`に掛ける）。
const PIVOT_LIMIT_FACTOR: usize = 50;
/// 退化を避けるため、制約の右辺に足す決定的な微小摂動の基準値（[`rhs_perturbation`]）。
const RHS_PERTURBATION: f64 = 1e-11;
/// 比検定で同点とみなす比の差の絶対許容幅。
const RATIO_TIE_TOLERANCE: f64 = 1e-12;
/// 右辺がこれ以下のピボットを退化ピボットとみなす。
const DEGENERATE_RHS_TOLERANCE: f64 = 1e-12;
/// 退化ピボットがこの回数続いたらBland規則（巡回しない規則）に切り替える。
const DEGENERATE_PIVOTS_BEFORE_BLAND: usize = 30;

impl LinearProgramSolver for DenseSimplex {
    fn maximize(&self, lp: &BoxBoundedLp<'_>) -> Result<Vec<f64>, LpError> {
        let k = lp.objective.len();
        let m = lp.constraint_rows.len();
        validate(lp, k, m)?;
        if k == 0 {
            return Ok(Vec::new());
        }

        // 変数: p_0..p_{k-1}, q_0..q_{k-1}（2k個）、続いて行ごとのスラック（rows個）。
        let rows = m + 2 * k;
        let cols = 2 * k + rows;
        let width = cols + 1; // 末尾が右辺
        let mut tableau = vec![0.0_f64; (rows + 1) * width];

        for (i, (row, &rhs)) in lp.constraint_rows.iter().zip(lp.rhs).enumerate() {
            for (j, &a) in row.iter().enumerate() {
                tableau[i * width + j] = a;
                tableau[i * width + k + j] = -a;
            }
            tableau[i * width + 2 * k + i] = 1.0;
            tableau[i * width + cols] = rhs + rhs_perturbation(i);
        }
        for j in 0..k {
            let p_row = m + j;
            tableau[p_row * width + j] = 1.0;
            tableau[p_row * width + 2 * k + p_row] = 1.0;
            tableau[p_row * width + cols] = lp.bound;
            let q_row = m + k + j;
            tableau[q_row * width + k + j] = 1.0;
            tableau[q_row * width + 2 * k + q_row] = 1.0;
            tableau[q_row * width + cols] = lp.bound;
        }
        // 目的関数行（最大化なので`-c`を置く）。
        for (j, &c) in lp.objective.iter().enumerate() {
            tableau[rows * width + j] = -c;
            tableau[rows * width + k + j] = c;
        }

        // 各行の基底変数の添字（初期はスラック）。
        let mut basis: Vec<usize> = (0..rows).map(|i| 2 * k + i).collect();
        let pivot_limit = PIVOT_LIMIT_FACTOR * (rows + cols);

        // 連続した退化ピボットの回数（Bland規則への切り替え判定）。
        let mut degenerate_run = 0usize;

        for _ in 0..pivot_limit {
            let use_bland = degenerate_run >= DEGENERATE_PIVOTS_BEFORE_BLAND;
            let objective_row = &tableau[rows * width..rows * width + cols];
            let entering = if use_bland {
                // Bland規則: 簡約費用が負の変数のうち添字最小。
                objective_row
                    .iter()
                    .position(|&v| v < -OPTIMALITY_TOLERANCE)
            } else {
                // Dantzig規則: 最も負の簡約費用。
                let mut best: Option<usize> = None;
                let mut best_value = -OPTIMALITY_TOLERANCE;
                for (j, &v) in objective_row.iter().enumerate() {
                    if v < best_value {
                        best_value = v;
                        best = Some(j);
                    }
                }
                best
            };
            let Some(entering) = entering else {
                return Ok(extract_solution(&tableau, &basis, k, width, cols));
            };

            // 比検定。同点（絶対許容幅内）は、Dantzig規則のときは絶対値最大のピボット要素、
            // Bland規則のときは基底変数の添字が小さい行を選ぶ。
            let mut leaving: Option<usize> = None;
            let mut best_ratio = f64::INFINITY;
            let mut best_pivot = 0.0_f64;
            for i in 0..rows {
                let a = tableau[i * width + entering];
                if a <= PIVOT_TOLERANCE {
                    continue;
                }
                let ratio = tableau[i * width + cols] / a;
                let better = match leaving {
                    None => true,
                    Some(l) => {
                        if ratio < best_ratio - RATIO_TIE_TOLERANCE {
                            true
                        } else if (ratio - best_ratio).abs() <= RATIO_TIE_TOLERANCE {
                            if use_bland {
                                basis[i] < basis[l]
                            } else {
                                a > best_pivot
                            }
                        } else {
                            false
                        }
                    }
                };
                if better {
                    best_ratio = best_ratio.min(ratio);
                    best_pivot = a;
                    leaving = Some(i);
                }
            }
            // 全変数が箱制約で有界なので、比検定の候補が無いのは数値的な退化のときだけ。
            let Some(leaving) = leaving else {
                return Err(LpError::DegeneratePivot);
            };

            if tableau[leaving * width + cols].abs() <= DEGENERATE_RHS_TOLERANCE {
                degenerate_run += 1;
            } else {
                degenerate_run = 0;
            }
            pivot(&mut tableau, rows + 1, width, leaving, entering);
            basis[leaving] = entering;
        }
        Err(LpError::IterationLimit {
            pivots: pivot_limit,
        })
    }
}

/// 制約`i`の右辺に足す微小摂動（`1e-11`〜`2e-11`）。右辺が全て0の錐制約は原点で強く退化し、
/// 同じ`x`で結果が逆の行対のように錐が潰れる入力では、Dantzig/Bland規則でも誤差が溜まって
/// 終了しないことがあった（`k=30`で実測）。行ごとに異なる微小値を足して退化を解く標準的な
/// 手法で、解は元の問題の制約を`2e-11`（`bound`に対する相対）以内でしか破らない。
/// 呼び出し側の違反判定の許容幅はこれより十分大きくとること。
fn rhs_perturbation(i: usize) -> f64 {
    // 黄金比の小数部で行ごとにばらつかせる（決定的）。
    let frac = (i as f64 * 0.618_033_988_749_894_9).fract();
    RHS_PERTURBATION * (1.0 + frac)
}

fn validate(lp: &BoxBoundedLp<'_>, k: usize, m: usize) -> Result<(), LpError> {
    if lp.rhs.len() != m || lp.constraint_rows.iter().any(|row| row.len() != k) {
        return Err(LpError::DimensionMismatch);
    }
    let finite = lp.objective.iter().all(|v| v.is_finite())
        && lp.constraint_rows.iter().flatten().all(|v| v.is_finite())
        && lp.rhs.iter().all(|v| v.is_finite());
    if !finite || !lp.bound.is_finite() || lp.bound <= 0.0 {
        return Err(LpError::InvalidInput);
    }
    if lp.rhs.iter().any(|&b| b < 0.0) {
        return Err(LpError::OriginInfeasible);
    }
    Ok(())
}

/// `row`行`col`列の要素を軸にガウス・ジョルダン消去する。
fn pivot(tableau: &mut [f64], n_rows: usize, width: usize, row: usize, col: usize) {
    let inv = 1.0 / tableau[row * width + col];
    for v in &mut tableau[row * width..(row + 1) * width] {
        *v *= inv;
    }
    let pivot_row: Vec<f64> = tableau[row * width..(row + 1) * width].to_vec();
    for i in 0..n_rows {
        if i == row {
            continue;
        }
        let factor = tableau[i * width + col];
        if factor == 0.0 {
            continue;
        }
        for (v, &p) in tableau[i * width..(i + 1) * width]
            .iter_mut()
            .zip(&pivot_row)
        {
            *v -= factor * p;
        }
        tableau[i * width + col] = 0.0;
    }
}

/// 最終タブローから`x = p - q`を取り出す。
fn extract_solution(
    tableau: &[f64],
    basis: &[usize],
    k: usize,
    width: usize,
    cols: usize,
) -> Vec<f64> {
    let mut values = vec![0.0_f64; 2 * k];
    for (i, &var) in basis.iter().enumerate() {
        if var < 2 * k {
            values[var] = tableau[i * width + cols];
        }
    }
    (0..k).map(|j| values[j] - values[k + j]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::linalg::solvers::Solve;

    fn solve(objective: &[f64], rows: &[Vec<f64>], rhs: &[f64], bound: f64) -> Vec<f64> {
        DenseSimplex
            .maximize(&BoxBoundedLp {
                objective,
                constraint_rows: rows,
                rhs,
                bound,
            })
            .unwrap()
    }

    // 単体法は退化を避けるため右辺に`2e-11`以下の摂動を足す（`rhs_perturbation`）ので、
    // 解は厳密な値から最大でこの程度ずれる。以下のアサーションの許容幅`1e-9`はそれに合わせた値。
    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn without_constraints_the_solution_is_the_box_corner_matching_the_objective_signs() {
        let x = solve(&[2.0, -3.0, 0.5], &[], &[], 1.0);
        assert_eq!(x, vec![1.0, -1.0, 1.0]);
    }

    #[test]
    fn respects_the_bound_value() {
        let x = solve(&[1.0, -1.0], &[], &[], 2.5);
        assert_eq!(x, vec![2.5, -2.5]);
    }

    #[test]
    fn solves_a_simple_constrained_problem() {
        // max x + y  s.t. x + y <= 1.5, |x|,|y| <= 1  → 最適値1.5
        let x = solve(&[1.0, 1.0], &[vec![1.0, 1.0]], &[1.5], 1.0);
        assert!((dot(&[1.0, 1.0], &x) - 1.5).abs() < 1e-9);
        assert!(x.iter().all(|v| v.abs() <= 1.0 + 1e-9));
    }

    #[test]
    fn solves_a_degenerate_cone_problem_without_cycling() {
        // 錐制約（右辺が全て0）: -x <= 0, -y <= 0, x - y <= 0 → 0 <= x <= y <= 1。
        // max x の最適解は x = y = 1。
        let rows = vec![vec![-1.0, 0.0], vec![0.0, -1.0], vec![1.0, -1.0]];
        let x = solve(&[1.0, 0.0], &rows, &[0.0, 0.0, 0.0], 1.0);
        assert!((x[0] - 1.0).abs() < 1e-9);
        assert!((x[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn returns_the_origin_when_the_cone_contains_only_the_origin() {
        // x <= 0 かつ -x <= 0 のとき x = 0 のみが許容。
        let x = solve(&[1.0], &[vec![1.0], vec![-1.0]], &[0.0, 0.0], 1.0);
        assert!(x[0].abs() < 1e-9);
    }

    #[test]
    fn handles_duplicate_constraint_rows() {
        let rows = vec![vec![1.0, 1.0]; 5];
        let x = solve(&[1.0, 1.0], &rows, &[1.0; 5], 1.0);
        assert!((dot(&[1.0, 1.0], &x) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn returns_an_empty_solution_for_zero_variables() {
        let x = solve(&[], &[], &[], 1.0);
        assert!(x.is_empty());
    }

    #[test]
    fn rejects_a_negative_right_hand_side() {
        let err = DenseSimplex
            .maximize(&BoxBoundedLp {
                objective: &[1.0],
                constraint_rows: &[vec![1.0]],
                rhs: &[-1.0],
                bound: 1.0,
            })
            .unwrap_err();
        assert_eq!(err, LpError::OriginInfeasible);
    }

    #[test]
    fn rejects_inconsistent_dimensions() {
        let err = DenseSimplex
            .maximize(&BoxBoundedLp {
                objective: &[1.0, 1.0],
                constraint_rows: &[vec![1.0]],
                rhs: &[0.0],
                bound: 1.0,
            })
            .unwrap_err();
        assert_eq!(err, LpError::DimensionMismatch);

        let err = DenseSimplex
            .maximize(&BoxBoundedLp {
                objective: &[1.0],
                constraint_rows: &[vec![1.0]],
                rhs: &[0.0, 0.0],
                bound: 1.0,
            })
            .unwrap_err();
        assert_eq!(err, LpError::DimensionMismatch);
    }

    #[test]
    fn rejects_non_finite_input_and_invalid_bound() {
        for (objective, bound) in [
            (f64::NAN, 1.0),
            (1.0, f64::INFINITY),
            (1.0, 0.0),
            (1.0, -1.0),
        ] {
            let err = DenseSimplex
                .maximize(&BoxBoundedLp {
                    objective: &[objective],
                    constraint_rows: &[],
                    rhs: &[],
                    bound,
                })
                .unwrap_err();
            assert_eq!(err, LpError::InvalidInput);
        }
    }

    /// 頂点列挙による全探索（`k`個の有効な制約を選んで連立方程式を解き、許容解のうち目的関数最大のもの）。
    fn brute_force_optimum(objective: &[f64], rows: &[Vec<f64>], rhs: &[f64], bound: f64) -> f64 {
        let k = objective.len();
        // 制約を `a'x <= b` の形に統一（箱制約も `±x_j <= bound` として加える）。
        let mut all_rows: Vec<(Vec<f64>, f64)> =
            rows.iter().cloned().zip(rhs.iter().copied()).collect();
        for j in 0..k {
            let mut up = vec![0.0; k];
            up[j] = 1.0;
            all_rows.push((up.clone(), bound));
            up[j] = -1.0;
            all_rows.push((up, bound));
        }
        let mut best = f64::NEG_INFINITY;
        let total = all_rows.len();
        // `total`本から`k`本を選ぶ全ての組合せ（添字の昇順）を順に調べる。
        let mut chosen: Vec<usize> = (0..k).collect();
        loop {
            let a = faer::Mat::from_fn(k, k, |i, j| all_rows[chosen[i]].0[j]);
            let b = faer::Mat::from_fn(k, 1, |i, _| all_rows[chosen[i]].1);
            let x = a.full_piv_lu().solve(&b);
            let x: Vec<f64> = (0..k).map(|i| *x.get(i, 0)).collect();
            if x.iter().all(|v| v.is_finite())
                && all_rows.iter().all(|(row, rhs)| dot(row, &x) <= rhs + 1e-9)
            {
                best = best.max(dot(objective, &x));
            }
            // 次の組合せへ（最後の位置から繰り上げる）。
            let Some(pos) = (0..k).rev().find(|&p| chosen[p] < total - (k - p)) else {
                break;
            };
            chosen[pos] += 1;
            for p in pos + 1..k {
                chosen[p] = chosen[p - 1] + 1;
            }
        }
        best
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

    #[test]
    fn matches_brute_force_vertex_enumeration_on_random_problems() {
        let mut rng = Lcg(12345);
        for trial in 0..200 {
            let k = 2 + trial % 2; // 2または3変数
            let m = 3 + trial % 5;
            let objective: Vec<f64> = (0..k).map(|_| rng.next_unit()).collect();
            let rows: Vec<Vec<f64>> = (0..m)
                .map(|_| (0..k).map(|_| rng.next_unit()).collect())
                .collect();
            // 半分は右辺が全て0の錐制約（退化）、残りは非負の乱数。
            let rhs: Vec<f64> = (0..m)
                .map(|_| {
                    if trial % 2 == 0 {
                        0.0
                    } else {
                        rng.next_unit().abs()
                    }
                })
                .collect();
            let x = solve(&objective, &rows, &rhs, 1.0);
            let got = dot(&objective, &x);
            let want = brute_force_optimum(&objective, &rows, &rhs, 1.0);
            assert!(
                (got - want).abs() < 1e-8,
                "trial {trial}: simplex {got} vs brute force {want}"
            );
            // 返った解自体も許容解であること。
            for (row, &b) in rows.iter().zip(&rhs) {
                assert!(
                    dot(row, &x) <= b + 1e-9,
                    "trial {trial}: constraint violated"
                );
            }
            assert!(x.iter().all(|v| v.abs() <= 1.0 + 1e-9));
        }
    }
}
