//! property-basedテスト。`ols/fit/tests/proptests.rs`と同型の設計だが、Logitは
//! MLEベースのため不変条件が異なる（`testing-policy.md`「property-basedテスト」参照）。
//! OLSの`coefficients_scale_linearly_with_y`（yに関する線形性）に相当する不変条件は
//! 無い（yは0/1のためスケール自体に意味が無い）代わりに、MLEの一次条件
//! （スコア方程式）が収束点で恒等的に成り立つことを検証する。
//!
//! プロパティの有効性検証（バグ注入→検出確認→元に戻す）は3件とも実施済み:
//! `score_is_near_zero_at_converged_params`は`LogitProblem::gradient`の
//! `diff`計算に定数オフセットを注入、`coefficients_and_se_are_invariant_to_
//! column_order`は`LogitInput::from_columns`の`param_names`を逆順にする
//! バグを注入、`hc0_std_errors_are_at_most_hc1_std_errors`は
//! `nonlinear::common::sandwich_cov_params`のHC1補正係数を反転（`n/(n-k)`→
//! `(n-k)/n`）するバグを注入し、いずれも検出できることを確認した。

use crate::nonlinear::common::CovType;
use crate::nonlinear::common::MleFitOptions;
use crate::nonlinear::common::SolverType;
use crate::nonlinear::logit::LogitEstimator;
use crate::nonlinear::logit::LogitInput;
use proptest::collection;
use proptest::prelude::*;

// 高k（列数依存バグ・数値的頑健性）はbenchmarkのmany_regressorsシナリオで
// 別途カバーしているため、ここでは
// MAX_Kを小さく保つ（分離を避けるための較正、下記`logit_case_strategy`参照）。
const MAX_K: usize = 4;

/// `logit_case_strategy`が生成するタプル: `(n, k, x_cols, beta, u, keys)`。
type LogitCase = (usize, usize, Vec<Vec<f64>>, Vec<f64>, Vec<f64>, Vec<u64>);

/// `(n, k, x_cols, beta, u, keys)`を生成する共通ストラテジ。
///
/// `x_cols`は`-2.0..2.0`、`beta`（切片含むk+1個）は`-1.0..1.0`の一様分布
/// （OLSの`-100.0..100.0`よりずっと狭い範囲。分離を避けるための較正、
/// `n=100・seed多数で実測しComputationError/NonConvergenceが十分低頻度に
/// 収まることを確認済み）。`u`はBernoulliサンプリング用の一様乱数
/// （`0.0..1.0`）で、`y_i = u_i < p_i ? 1.0 : 0.0`として`simulate_y`内で計算する
/// （proptestのstrategyには二値分布が無いため、閾値判定で代用）。
fn logit_case_strategy() -> impl Strategy<Value = LogitCase> {
    (1..=MAX_K).prop_flat_map(|k| {
        (k + 20..=100usize).prop_flat_map(move |n| {
            (
                Just(n),
                Just(k),
                collection::vec(collection::vec(-2.0f64..2.0, n), k),
                collection::vec(-1.0f64..1.0, k + 1),
                collection::vec(0.0f64..1.0, n),
                collection::vec(any::<u64>(), k),
            )
        })
    })
}

fn x_names(k: usize) -> Vec<String> {
    (1..=k).map(|i| format!("x{i}")).collect()
}

/// 真の`beta`から線形予測子`z`・ロジスティック確率`p`を計算し、`u`との比較で
/// 二値`y`をサンプリングする。
fn simulate_y(n: usize, x_cols: &[Vec<f64>], beta: &[f64], u: &[f64]) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let mut z = beta[0];
            for (j, x_col) in x_cols.iter().enumerate() {
                z += x_col[i] * beta[j + 1];
            }
            let p = 1.0 / (1.0 + (-z).exp());
            if u[i] < p { 1.0 } else { 0.0 }
        })
        .collect()
}

/// `solver`ごとに`tol`の意味論が異なる（`docs/spec/logit-spec.md`3.2節）:
/// `newton`は総和勾配に対する絶対閾値（既定`1e-6`）、
/// `bfgs`/`lbfgs`は観測数`n_obs`で正規化した基準（既定`1e-8`、`run_solver`が
/// 内部で`tol*n_obs`を実効的な絶対閾値に変換する）。`engine_pybind`側の
/// `LogitOptions`と同じ既定値の解決をここでも行う（呼び出し元がこの対応を
/// 誤ると実質的に閾値が数桁ずれるため、テストコード側でも明示する）。
fn default_options(cov_type: CovType, solver: SolverType) -> MleFitOptions {
    let tol = match solver {
        SolverType::Newton => 1e-6,
        SolverType::Bfgs | SolverType::Lbfgs => 1e-8,
    };
    MleFitOptions {
        solver,
        max_iter: 50,
        tol,
        raise_on_non_convergence: true,
        cov_type,
        confidence_level: 0.95,
    }
}

/// `SolverType::Newton`/`Bfgs`/`Lbfgs`を等確率で生成する。3手法とも収束経路・
/// `tol`の意味論が異なるため（上記`default_options`参照）、各プロパティを
/// solver非依存で1種類だけ検証するとBFGS/L-BFGS固有の経路がproptestの
/// 恩恵を受けない。
fn solver_strategy() -> impl Strategy<Value = SolverType> {
    prop_oneof![
        Just(SolverType::Newton),
        Just(SolverType::Bfgs),
        Just(SolverType::Lbfgs),
    ]
}

/// `ols/fit/tests/proptests.rs`側の同名ヘルパー（`RTOL=1e-6`）より緩い`1e-4`を使う理由: OLSの
/// 列順序不変性は閉形式解（col_piv_qr）の数値誤差のみが要因だが、Logitは
/// 反復最適化（Newton/BFGS/L-BFGS）を経るため、列順序の違いで収束経路
/// （標準化空間での列スケールが変わる）が変わり、`tol`ちょうどで停止する
/// 反復回数がわずかにずれうる。この経路依存の誤差蓄積を吸収するため。
fn assert_approx_eq(actual: f64, expected: f64, msg: &str) {
    let tol = 1e-4 * expected.abs().max(1.0);
    let diff = (actual - expected).abs();
    assert!(
        diff <= tol,
        "{msg}: actual={actual}, expected={expected}, diff={diff}, tol={tol}"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// MLEの一次条件（スコア方程式）: 収束点で`Σᵢ(yᵢ-pᵢ)xᵢ ≈ 0`が全パラメータ列で
    /// 成り立つ（`docs/spec/logit-spec.md`3.1節のスコア`∂ℓ/∂θ=X'(y-p)`が
    /// 収束点でゼロになるという、MLEの定義そのものに由来する不変条件）。
    /// 許容誤差`1e-4`は`n`に依存しない絶対閾値（`newton`の収束判定`tol=1e-6`・
    /// `bfgs`/`lbfgs`の実効閾値`tol*n_obs=1e-8*n`のいずれも、標準化空間の
    /// 勾配ノルムに対する基準であり原スケールのスコア自体は`n`でスケールしない。
    /// 実測（n<=100の256ケース）でも最大7.5e-7程度に収まることを確認済み、
    /// `1e-4`は2桁以上の安全マージンを持つ）。
    #[test]
    fn score_is_near_zero_at_converged_params(
        (n, k, x_cols, beta, u, _keys) in logit_case_strategy(),
        solver in solver_strategy(),
    ) {
        let y = simulate_y(n, &x_cols, &beta, &u);
        let names = x_names(k);
        let input = LogitInput::from_columns(&y, &x_cols, names, true, "y".to_string()).unwrap();
        let result = LogitEstimator::fit(input, default_options(CovType::Classical, solver));
        prop_assume!(result.is_ok());
        let est = result.unwrap();

        let p = est.predict();
        let x = est.input().x();
        for j in 0..=k {
            let score: f64 = (0..n).map(|i| (y[i] - p[i]) * x.get(i, j)).sum();
            prop_assert!(
                score.abs() <= 1e-4,
                "score[{j}] should be ~0, got {score} (solver={solver:?})"
            );
        }
    }

    /// xの列順序を入れ替えても、係数名で対応付ければ係数・標準誤差の値は変わらない。
    #[test]
    fn coefficients_and_se_are_invariant_to_column_order(
        (n, k, x_cols, beta, u, keys) in logit_case_strategy()
            .prop_filter("need >=2 columns to permute", |(_, k, _, _, _, _)| *k >= 2),
        solver in solver_strategy(),
    ) {
        let y = simulate_y(n, &x_cols, &beta, &u);
        let names = x_names(k);
        let input1 = LogitInput::from_columns(&y, &x_cols, names.clone(), true, "y".to_string()).unwrap();
        let result1 = LogitEstimator::fit(input1, default_options(CovType::Classical, solver));
        prop_assume!(result1.is_ok());
        let est1 = result1.unwrap();

        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by_key(|&i| keys[i]);
        let permuted_x: Vec<Vec<f64>> = order.iter().map(|&i| x_cols[i].clone()).collect();
        let permuted_names: Vec<String> = order.iter().map(|&i| names[i].clone()).collect();

        let input2 = LogitInput::from_columns(&y, &permuted_x, permuted_names, true, "y".to_string()).unwrap();
        let result2 = LogitEstimator::fit(input2, default_options(CovType::Classical, solver));
        prop_assume!(result2.is_ok());
        let est2 = result2.unwrap();

        let names1 = est1.input().param_names().to_vec();
        let names2 = est2.input().param_names().to_vec();
        let (params1, params2) = (est1.params(), est2.params());
        let (se1, se2) = (est1.std_errors(), est2.std_errors());
        for (idx1, name) in names1.iter().enumerate() {
            let idx2 = names2.iter().position(|n| n == name)
                .expect("name should exist in permuted result");
            assert_approx_eq(params2[idx2], params1[idx1], &format!("param[{name}] under column permutation"));
            assert_approx_eq(se2[idx2], se1[idx1], &format!("std_error[{name}] under column permutation"));
        }
    }

    /// HC0の標準誤差は常にHC1以下（`hc1 = hc0`にn/(n-k)の小標本補正を乗じたもの、
    /// `docs/spec/logit-spec.md`3.3節）。
    #[test]
    fn hc0_std_errors_are_at_most_hc1_std_errors(
        (n, k, x_cols, beta, u, _keys) in logit_case_strategy(),
        solver in solver_strategy(),
    ) {
        let y = simulate_y(n, &x_cols, &beta, &u);
        let names = x_names(k);
        let input1 = LogitInput::from_columns(&y, &x_cols, names.clone(), true, "y".to_string()).unwrap();
        let result1 = LogitEstimator::fit(input1, default_options(CovType::Hc0, solver));
        prop_assume!(result1.is_ok());
        let est_hc0 = result1.unwrap();

        let input2 = LogitInput::from_columns(&y, &x_cols, names, true, "y".to_string()).unwrap();
        let result2 = LogitEstimator::fit(input2, default_options(CovType::Hc1, solver));
        prop_assume!(result2.is_ok());
        let est_hc1 = result2.unwrap();

        let (se_hc0, se_hc1) = (est_hc0.std_errors(), est_hc1.std_errors());
        for i in 0..=k {
            prop_assert!(
                se_hc0[i] <= se_hc1[i] + 1e-9,
                "HC0 se[{i}]={} should be <= HC1 se[{i}]={}", se_hc0[i], se_hc1[i]
            );
        }
    }
}
