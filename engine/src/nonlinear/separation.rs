//! Logit/Probitの完全分離・準完全分離（(quasi-)complete separation）の検出。
//!
//! ある`θ`について全観測で`zᵢ'θ ≥ 0`（`zᵢ = (2yᵢ-1)xᵢ`）かつどれかが厳密に正になるなら、
//! `θ`の方向へ係数を大きくするほど対数尤度が増え続けるため、有限の最尤推定量は存在しない
//! （Albert & Anderson 1984）。この判定は閾値を持たない厳密な条件で、次の線形計画（LP）に
//! 帰着する。
//!
//! ```text
//! maximize  Σᵢ uᵢ'θ     s.t.  uᵢ'θ ≥ 0 (全i),  |θⱼ| ≤ 1        （uᵢ = zᵢ/‖zᵢ‖）
//! ```
//!
//! 行を単位長にそろえるのは数値の尺度を一定にするため（正の定数倍は錐を変えない）。分離が
//! 無ければ許容解は原点だけで、解`θ`の最大マージン`maxᵢ uᵢ'θ`は0。分離があれば最適解は
//! 分離した観測のどれかに`O(1)`のマージンを持つ。判定は和ではなく最大マージンで行う
//! （和は`n`に比例して雑音が積もる）。
//!
//! 全観測を一度に制約に入れずに済むよう、**制約生成**で解く。小さな作業集合の制約でLPを解き、
//! 解が違反した制約（最大[`ROWS_ADDED_PER_ROUND`]本/ラウンド）だけを足して繰り返す。実測では
//! 3〜4ラウンド・作業集合は数百本以下で収束する。LPの解き方は[`LinearProgramSolver`]で
//! 差し替えられる。
//!
//! **事前スクリーニング（収束点の係数方向や情報行列の固有ベクトル方向でほぼ全観測が正側か、
//! を先に調べてLPを省く案）は採らない**。分離方向が複数ある（錐の次元が2以上）と、情報行列の
//! 近零固有空間が多次元になり、最小固有ベクトルが錐の外を向くことがあり、構造的に取りこぼす
//! （3つのダミーが同時に準完全分離するデータで実測）。取りこぼしは「分離を無警告の成功として返す」
//! 検出漏れになるため、常にLPで確かめる。

use faer::Mat;
use std::borrow::Cow;
use std::collections::HashSet;

use super::common::{ColumnScale, MleError};
use crate::shared::linear_program::{BoxBoundedLp, DenseSimplex, LinearProgramSolver};

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

/// 分離の有無。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SeparationVerdict {
    NotSeparated,
    Separated,
}

/// 収束後の分離チェックを行い、チェック後の`converged`を返す。`fit()`はこれを1回呼ぶだけでよい。
///
/// - `converged=false`（ソルバーが収束しなかった）ならチェックせずそのまま`false`を返す。
/// - 分離を検出したとき、`raise_on_non_convergence=true`なら[`MleError::SeparationDetected`]、
///   `false`なら`converged=false`を返す（結果は呼び出し側がそのまま組み立てる）。
/// - 判定そのものが数値的に完了しなかったとき（LPの失敗など、実測のデータでは起きていない）は、
///   分離が無いと確認できなかったので、`raise_on_non_convergence=true`ならそのエラー
///   （`SeparationCheckFailed`）を返し、`false`なら`converged=false`を返す。推定結果そのものは
///   得られているので、フラグの契約（「結果を`converged=false`で返す」）を保つ。
///
/// `x`は標準化前の設計行列で、`scale`・`y`は`fit()`が最適化に使ったもの。ここでは列を標準偏差で
/// 割るだけの行を使い（`fit()`の標準化が切片ありで平均も引くのとは別。切片込みの再パラメータ化は
/// 分離の有無を変えず、LPの閾値の較正もこの行に対するもの）、標準化後の行列を複製せず`x`と
/// `scale`から行を都度計算する。
pub(crate) fn verify_no_separation(
    x: &Mat<f64>,
    scale: &ColumnScale,
    y: &Mat<f64>,
    converged: bool,
    raise_on_non_convergence: bool,
) -> Result<bool, MleError> {
    verify_with_solver(
        x,
        scale,
        y,
        converged,
        raise_on_non_convergence,
        &DenseSimplex,
    )
}

/// [`verify_no_separation`]のLPソルバーを注入できる版（LP失敗時の方針をテストするため）。
fn verify_with_solver<S: LinearProgramSolver>(
    x: &Mat<f64>,
    scale: &ColumnScale,
    y: &Mat<f64>,
    converged: bool,
    raise_on_non_convergence: bool,
    solver: &S,
) -> Result<bool, MleError> {
    if !converged {
        return Ok(false);
    }
    match detect_separation(x, scale, y, solver) {
        Ok(SeparationVerdict::NotSeparated) => Ok(true),
        Ok(SeparationVerdict::Separated) if raise_on_non_convergence => {
            Err(MleError::SeparationDetected)
        }
        Ok(SeparationVerdict::Separated) => Ok(false),
        Err(e) if raise_on_non_convergence => Err(e),
        Err(_) => Ok(false),
    }
}

/// 分離の有無を厳密に判定する（制約生成によるLP、モジュールdocコメント参照）。
pub(crate) fn detect_separation<S: LinearProgramSolver>(
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

/// 設計行列の第`j`列の値。`Mat`は列方向に連続しているので通常は借用でき、要素ごとの境界チェックを
/// 避けられる（行列を走査する各パスが`O(nk)`で、大標本ではこのコストが目立つ）。連続でない場合
/// （通常は起きない）だけコピーする。
fn column_values(x: &Mat<f64>, j: usize) -> Cow<'_, [f64]> {
    match x.col(j).try_as_col_major() {
        Some(col) => Cow::Borrowed(col.as_slice()),
        None => Cow::Owned((0..x.nrows()).map(|i| *x.get(i, j)).collect()),
    }
}

/// 標準化空間の符号付き設計行列の行`zᵢ = (2yᵢ-1)·x_stdᵢ`を、単位長にそろえて扱う。
/// 標準化後の行列は持たず、`x`・`1/std`から都度計算する。
struct SignedRows<'a> {
    x: &'a Mat<f64>,
    inv_std: Vec<f64>,
    /// `(2yᵢ-1)/‖x_stdᵢ‖`。零ベクトルの行（制約にならない）は`0.0`。
    row_scale: Vec<f64>,
}

impl<'a> SignedRows<'a> {
    fn new(x: &'a Mat<f64>, scale: &ColumnScale, y: &Mat<f64>) -> Self {
        let n = x.nrows();
        let k = x.ncols();
        let inv_std: Vec<f64> = scale.stds().iter().map(|s| 1.0 / s).collect();
        let mut sq_norm = vec![0.0_f64; n];
        for (j, &inv) in inv_std.iter().enumerate().take(k) {
            for (acc, &xv) in sq_norm.iter_mut().zip(column_values(x, j).iter()) {
                let v = xv * inv;
                *acc += v * v;
            }
        }
        let row_scale = sq_norm
            .iter()
            .enumerate()
            .map(|(i, &sq)| {
                let norm = sq.sqrt();
                if norm > 0.0 && norm.is_finite() {
                    let sign = if *y.get(i, 0) > 0.5 { 1.0 } else { -1.0 };
                    sign / norm
                } else {
                    0.0
                }
            })
            .collect();
        Self {
            x,
            inv_std,
            row_scale,
        }
    }

    fn n(&self) -> usize {
        self.x.nrows()
    }

    fn k(&self) -> usize {
        self.x.ncols()
    }

    fn is_active(&self, i: usize) -> bool {
        self.row_scale[i] != 0.0
    }

    /// 各行の`uᵢ'θ`（単位長の行と`θ`の内積）。零ベクトルの行は`0.0`。
    fn margins(&self, theta: &[f64]) -> Vec<f64> {
        let mut acc = vec![0.0_f64; self.n()];
        for (j, (&inv, &t)) in self.inv_std.iter().zip(theta).enumerate() {
            let w = inv * t;
            if w == 0.0 {
                continue;
            }
            for (a, &xv) in acc.iter_mut().zip(column_values(self.x, j).iter()) {
                *a += xv * w;
            }
        }
        for (a, &s) in acc.iter_mut().zip(&self.row_scale) {
            *a *= s;
        }
        acc
    }

    /// 行`i`の単位ベクトル`uᵢ`。
    fn unit_row(&self, i: usize) -> Vec<f64> {
        (0..self.k())
            .map(|j| *self.x.get(i, j) * self.inv_std[j] * self.row_scale[i])
            .collect()
    }

    /// LPの目的関数の係数`c = Σᵢ uᵢ`。
    fn objective(&self) -> Vec<f64> {
        (0..self.k())
            .map(|j| {
                let sum: f64 = column_values(self.x, j)
                    .iter()
                    .zip(&self.row_scale)
                    .map(|(&xv, &s)| xv * s)
                    .sum();
                sum * self.inv_std[j]
            })
            .collect()
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
    use super::*;
    use crate::nonlinear::common::standardize_columns;
    use crate::shared::linear_program::LpError;

    /// `has_intercept=true`の設計行列（先頭列が1）を作る。
    fn design(columns: &[&[f64]]) -> Mat<f64> {
        let n = columns[0].len();
        Mat::from_fn(n, columns.len() + 1, |i, j| {
            if j == 0 { 1.0 } else { columns[j - 1][i] }
        })
    }

    fn response(values: &[f64]) -> Mat<f64> {
        Mat::from_fn(values.len(), 1, |i, _| values[i])
    }

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

    fn quasi_complete_dummy_data(n: usize) -> (Mat<f64>, Mat<f64>) {
        // d=1の観測は全てy=1、d=0の観測はyが交互（分離しない）。xは決定的な擬似乱数。
        let d: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 37 % 101) as f64) / 50.0 - 1.0)
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| if d[i] == 1.0 || i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        (design(&[&d, &x]), response(&y))
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

    /// 常にエラーを返すソルバー（LPの失敗が`SeparationCheckFailed`になることの検証用）。
    struct FailingSolver;
    impl LinearProgramSolver for FailingSolver {
        fn maximize(&self, _: &BoxBoundedLp<'_>) -> Result<Vec<f64>, LpError> {
            Err(LpError::IterationLimit { pivots: 1 })
        }
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

    #[test]
    fn verify_no_separation_follows_raise_policy_when_the_check_cannot_be_completed() {
        let (x, y) = quasi_complete_dummy_data(60);
        let (_, scale) = standardize_columns(&x, true);
        // raise=true: 分離が無いと確認できなかったことを`SeparationCheckFailed`で知らせる。
        let err = verify_with_solver(&x, &scale, &y, true, true, &FailingSolver).unwrap_err();
        assert!(matches!(err, MleError::SeparationCheckFailed { .. }));
        // raise=false: 推定結果は返し、`converged=false`にする。
        assert!(!verify_with_solver(&x, &scale, &y, true, false, &FailingSolver).unwrap());
    }

    #[test]
    fn verify_no_separation_returns_false_without_checking_when_not_converged() {
        let x = design(&[&[-2.0, -1.0, 1.0, 2.0]]);
        let y = response(&[0.0, 0.0, 1.0, 1.0]);
        let (_, scale) = standardize_columns(&x, true);
        assert!(!verify_no_separation(&x, &scale, &y, false, true).unwrap());
    }

    #[test]
    fn verify_no_separation_raises_or_clears_converged_on_separated_data() {
        let x = design(&[&[-2.0, -1.0, 1.0, 2.0]]);
        let y = response(&[0.0, 0.0, 1.0, 1.0]);
        let (_, scale) = standardize_columns(&x, true);
        let err = verify_no_separation(&x, &scale, &y, true, true).unwrap_err();
        assert_eq!(err, MleError::SeparationDetected);
        assert!(!verify_no_separation(&x, &scale, &y, true, false).unwrap());
    }

    #[test]
    fn verify_no_separation_keeps_converged_for_normal_data() {
        let x = design(&[&[-2.0, -1.0, 1.0, 2.0, 0.5, -0.5]]);
        let y = response(&[0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        let (_, scale) = standardize_columns(&x, true);
        assert!(verify_no_separation(&x, &scale, &y, true, true).unwrap());
    }
}
