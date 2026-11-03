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
//! 解が違反した制約（最大`ROWS_ADDED_PER_ROUND`本/ラウンド（`lp.rs`））だけを足して繰り返す。実測では
//! 3〜4ラウンド・作業集合は数百本以下で収束する。LPの解き方は[`LinearProgramSolver`]で
//! 差し替えられる。
//!
//! **事前スクリーニング（収束点の係数方向や情報行列の固有ベクトル方向でほぼ全観測が正側か、
//! を先に調べてLPを省く案）は採らない**。分離方向が複数ある（錐の次元が2以上）と、情報行列の
//! 近零固有空間が多次元になり、最小固有ベクトルが錐の外を向くことがあり、構造的に取りこぼす
//! （3つのダミーが同時に準完全分離するデータで実測）。取りこぼしは「分離を無警告の成功として返す」
//! 検出漏れになるため、常にLPで確かめる。
//!
//! # ファイル構成
//!
//! このファイルは分離チェックの入口（`SeparationVerdict`・`verify_no_separation`とそのソルバー注入版）だけを持つ。
//! LPの判定と行の生成は子モジュールに置き、テストは各ロジックと同じファイルの`mod tests`に置く。
//!
//! - `lp.rs`: 制約生成によるLPの判定（`detect_separation`・閾値）
//! - `rows.rs`: LPに渡す符号付きの行（`SignedRows`）
//! - `test_support.rs`: 複数ファイルのテストが共有するフィクスチャ（`#[cfg(test)]`）

use faer::Mat;

use crate::nonlinear::common::{ColumnScale, MleError};
use crate::shared::linear_program::{DenseSimplex, LinearProgramSolver};

mod lp;
mod rows;
#[cfg(test)]
mod test_support;

use lp::detect_separation;

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

#[cfg(test)]
mod tests {
    use super::test_support::*;

    use super::*;

    use crate::nonlinear::common::standardize_columns;

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
