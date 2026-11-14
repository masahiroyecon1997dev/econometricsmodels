//! 数値の小道具: 補償和・ノルム・単位行列・小さな密行列演算・正定値なら逆行列を返す関数。

use faer::prelude::SolveLstsq;
use faer::{Mat, Side};

use crate::shared::linear_algebra::checked_col_piv_qr;

pub(super) fn identity_matrix(k: usize) -> Vec<Vec<f64>> {
    (0..k)
        .map(|i| (0..k).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

/// L2ノルム（勾配ベクトルの収束判定`FaerNewton::terminate`等で使う）。
pub(super) fn l2_norm(g: &[f64]) -> f64 {
    g.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Neumaierの補償和（改良版Kahan和）。各加算で失われた下位ビットを補正項に蓄積し、
/// 最後に足し戻す。誤差は項数`n`にほぼ依存せず、結果の数ULP程度に収まる。
///
/// **対数尤度の総和（`cost`）に使う理由**: `n`個の観測の対数尤度を素朴に逐次加算すると、
/// 累積値（大標本では`|ℓ|≈4e5`）への加算ごとに丸め誤差が入り、合計で結果のULPの
/// `O(√n)`倍（`n=1e6`で実測`≈1e-8`、ULPの約100倍）のノイズになる。収束点近傍では
/// Newtonステップによる真のコスト減少量がこのノイズを下回り、`regularized_newton_step`
/// の`cost(θ-Δθ) < cost(θ)`比較が符号の運任せになる（正しいステップを棄却し、LMラダーで
/// 無意味な極小ステップを受理して反復とcost評価を浪費する）。補償和でノイズを
/// ULP程度に抑え、コスト比較が意味を持つ範囲を広げる。ULPを下回る減少量は補償和でも
/// 判定できないため、その領域は`regularized_newton_step`側で扱う
/// （`NEWTON_COST_RESOLUTION_FACTOR`参照）。
///
/// 勾配・Hessianの総和は素朴な加算のままでよい: 勾配は収束点で0に近づくため累積値が
/// 小さく丸め誤差も小さい（`n=1e6`のProbitでraw Newtonステップ後の勾配L2ノルム
/// `≈2e-11`を実測）。Hessianはステップ方向を決めるだけで、多少の誤差は2次収束を損なわない。
///
/// **非有限値の扱いは素朴な和と同じ**: 項に`±∞`が含まれると補正項が`∞-∞=NaN`になるため、
/// 和が非有限のときは補正項を足さずに和をそのまま返す（`-∞`を含む和は`-∞`、`NaN`を含む
/// 和は`NaN`。Tobitの対数尤度の項は`σ`のアンダーフロー等で`-∞`になりうる）。
pub(crate) fn compensated_sum<I>(values: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    for value in values {
        let t = sum + value;
        // 絶対値の大きい方を基準に、`t`で失われた小さい方の下位ビットを回収する。
        compensation += if sum.abs() >= value.abs() {
            (sum - t) + value
        } else {
            (value - t) + sum
        };
        sum = t;
    }
    if sum.is_finite() {
        sum + compensation
    } else {
        sum
    }
}

/// コスト関数のHessian`H`が正定値かつ可逆なとき、その逆行列`H⁻¹`（対称化済み）を返す。
/// そうでない（正定値でない・特異・非有限値を含む）場合は`None`。
///
/// `FaerBfgs::init`・`FaerLbfgs::init`が初期逆Hessian近似に使う。`H`が正定値でない場合の`H⁻¹`は
/// 降下方向を作らない（line searchが`Search direction must be a descent direction`で
/// 失敗する）ため、呼び出し側は`None`のとき従来の初期化（`FaerBfgs`は単位行列＋self-scaling、
/// `FaerLbfgs`は`γI`）へ戻す。
///
/// 列ピボットQRは1回だけ行い（特異性判定は`newton_step`と同じ`checked_col_piv_qr`）、単位行列を
/// 右辺にまとめて解く（`k`本の右辺ごとにQRをやり直すと`O(k⁴)`になる）。
pub(super) fn inverse_if_positive_definite(hessian: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    if !cost_hessian_is_positive_definite(hessian) {
        return None;
    }
    let k = hessian.len();
    let h = Mat::from_fn(k, k, |i, j| hessian[i][j]);
    let qr = checked_col_piv_qr(&h).ok()?;
    let identity = Mat::from_fn(k, k, |i, j| if i == j { 1.0 } else { 0.0 });
    let solved = qr.solve_lstsq(&identity);
    let inv: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..k).map(|j| *solved.get(i, j)).collect())
        .collect();
    let symmetrized: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..k).map(|j| 0.5 * (inv[i][j] + inv[j][i])).collect())
        .collect();
    symmetrized
        .iter()
        .all(|row| row.iter().all(|v| v.is_finite()))
        .then_some(symmetrized)
}

/// 対角成分が`scale`の`k×k`対角行列。`identity_matrix`の一般化（`scale=1.0`で
/// 一致する）だが、`FaerBfgs`のself-scaling初期化専用に使うため、意味が伝わる
/// 別名の関数として分離する。
pub(super) fn scaled_identity(scale: f64, k: usize) -> Vec<Vec<f64>> {
    (0..k)
        .map(|i| (0..k).map(|j| if i == j { scale } else { 0.0 }).collect())
        .collect()
}

pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub(super) fn mat_vec(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    mat.iter().map(|row| dot(row, v)).collect()
}

pub(super) fn mat_mat(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b.len();
    let p = b[0].len();
    (0..n)
        .map(|i| {
            (0..p)
                .map(|j| (0..m).map(|t| a[i][t] * b[t][j]).sum())
                .collect()
        })
        .collect()
}

pub(super) fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    (0..m).map(|j| (0..n).map(|i| a[i][j]).collect()).collect()
}

/// コスト関数（負の対数尤度`-ℓ`）のHessianが正定値か（Cholesky分解`llt`が成功するか）。
/// 真のMLE最大点＝`-ℓ`の最小点ではこれが正定値になる（内点最大の2階十分条件）。
///
/// `regularized_newton_step`が`RegularizedStep::BelowCostResolution`でコスト比較を
/// 省く前提条件（局所2次モデルが凸で、raw Newtonステップが降下方向）にも使う。
///
/// `FaerNewton::next_iter`が`RegularizedStep::NoProgress`（または`BelowCostResolution`）を
/// `stalled_at_optimum`（収束扱い）に昇格させる前の最終確認にも使う。`newton_step`の
/// 列ピボットQRは可逆性（フルランク）しか見ないため、鞍点（可逆だが不定符号）でも
/// `NoProgress`が返りうる。Tobitの`(β, logσ)`尤度は大域凹性が保証されない
/// （`docs/spec/tobit-spec.md`3.1節）ため、勾配ノルムの小ささだけを根拠に収束と
/// 判定すると、鞍点で`cov_type="opg"`（`-H`を使わない）が誤った推定値を黙って返す
/// リスクがある。ここで正定値性を明示的に確認して内点最大でのみ収束扱いにする。
pub(super) fn cost_hessian_is_positive_definite(hessian: &[Vec<f64>]) -> bool {
    let k = hessian.len();
    let h = Mat::from_fn(k, k, |i, j| hessian[i][j]);
    h.llt(Side::Lower).is_ok()
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn scaled_identity_returns_diagonal_matrix_with_given_scale() {
        let m = scaled_identity(2.5, 3);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 2.5 } else { 0.0 };
                assert_eq!(m[i][j], expected, "i={i}, j={j}, m={m:?}");
            }
        }
    }

    #[test]
    fn inverse_if_positive_definite_returns_symmetric_inverse_for_positive_definite_hessian() {
        // [[4,1],[1,3]]の逆行列は(1/11)·[[3,-1],[-1,4]]。
        let inv = inverse_if_positive_definite(&[vec![4.0, 1.0], vec![1.0, 3.0]]).unwrap();
        let expected = [[3.0 / 11.0, -1.0 / 11.0], [-1.0 / 11.0, 4.0 / 11.0]];
        for i in 0..2 {
            for j in 0..2 {
                assert!((inv[i][j] - expected[i][j]).abs() < 1e-12, "{inv:?}");
            }
        }
    }

    #[test]
    fn inverse_if_positive_definite_returns_none_for_indefinite_or_singular_hessian() {
        // 不定符号（固有値 +3, -1）: 逆行列は降下方向を作らないため弾く。
        assert!(inverse_if_positive_definite(&[vec![1.0, 2.0], vec![2.0, 1.0]]).is_none());
        // 特異（ランク1）。
        assert!(inverse_if_positive_definite(&[vec![1.0, 1.0], vec![1.0, 1.0]]).is_none());
        // 非有限値を含む。
        assert!(inverse_if_positive_definite(&[vec![f64::NAN, 0.0], vec![0.0, 1.0]]).is_none());
    }

    #[test]
    fn compensated_sum_recovers_low_order_bits_lost_by_naive_summation() {
        // 素朴な逐次加算では`1e16 + 1.0`で`1.0`が丸めにより失われ、結果は0になる。
        let values = [1.0e16, 1.0, -1.0e16];
        assert_eq!(values.iter().sum::<f64>(), 0.0);
        assert_eq!(compensated_sum(values), 1.0);
        // 小さい値が先に来る順序（`|value| > |sum|`の分岐）でも同じく回収できる。
        assert_eq!(compensated_sum([1.0, 1.0e16, -1.0e16]), 1.0);
        assert_eq!(compensated_sum(std::iter::empty()), 0.0);
    }

    /// 非有限値の扱いは素朴な和と同じ（補正項の`∞-∞=NaN`を持ち込まない）。
    #[test]
    fn compensated_sum_propagates_non_finite_values_like_naive_summation() {
        assert_eq!(
            compensated_sum([1.0, f64::NEG_INFINITY, 2.0]),
            f64::NEG_INFINITY
        );
        assert_eq!(compensated_sum([f64::INFINITY, 1.0]), f64::INFINITY);
        assert!(compensated_sum([f64::INFINITY, f64::NEG_INFINITY]).is_nan());
        assert!(compensated_sum([1.0, f64::NAN]).is_nan());
    }

    #[test]
    fn compensated_sum_of_many_terms_is_accurate_to_a_few_ulps() {
        // `0.1`は2進で正確に表せないため、素朴な逐次加算では誤差が項数に応じて蓄積する。
        // 補償和は正確な和`n·fl(0.1)`（`fl(0.1)`は`0.1`の倍精度表現）に数ULPで一致する。
        let n = 1_000_000;
        let exact = n as f64 * 0.1;
        let naive: f64 = std::iter::repeat_n(0.1, n).sum();
        let compensated = compensated_sum(std::iter::repeat_n(0.1, n));
        let ulp = f64::EPSILON * exact;
        assert!((compensated - exact).abs() <= 2.0 * ulp, "{compensated}");
        assert!((naive - exact).abs() > 100.0 * ulp, "{naive}");
    }
}
