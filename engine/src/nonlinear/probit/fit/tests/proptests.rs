//! property-basedテスト。`logit/fit/tests/proptests.rs`と同型の設計（ケース生成・
//! `solver_strategy`・許容誤差較正）だが、`score_is_near_zero_at_converged_params`
//! のスコア計算式はLogitの`Σᵢ(yᵢ-pᵢ)xᵢ`をそのまま移植せず、Probit固有の
//! `Σᵢλᵢxᵢ`（`ProbitProblem::gradient`と同じ一般化残差、プロパティ本体のdoc
//! コメント参照）に置き換えている。それ以外の不変条件の理論的根拠・較正方針の
//! 詳細は`logit/fit/tests/proptests.rs`側のdocコメント参照（`testing-policy.md`「property-basedテスト」
//! 参照）。
//!
//! プロパティの有効性検証（バグ注入→検出確認→元に戻す）は3件とも実施済み:
//! `score_is_near_zero_at_converged_params`は`ProbitProblem::gradient`の
//! スコア計算に定数オフセットを注入、`coefficients_and_se_are_invariant_to_
//! column_order`は`ProbitInput::from_columns`の`param_names`を逆順にする
//! バグを注入、`hc0_std_errors_are_at_most_hc1_std_errors`は
//! `nonlinear::common::sandwich_cov_params`のHC1補正係数を反転（`n/(n-k)`→
//! `(n-k)/n`、Logitと共有するロジックのため同一のバグ注入で確認）するバグを
//! 注入し、いずれも検出できることを確認した。

use crate::nonlinear::common::CovType;
use crate::nonlinear::common::MleFitOptions;
use crate::nonlinear::common::SolverType;
use crate::nonlinear::common::clamped_pdf_cdf;
use crate::nonlinear::probit::ProbitEstimator;
use crate::nonlinear::probit::ProbitInput;
use proptest::collection;
use proptest::prelude::*;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

// logit/fit/tests/proptests.rsと同じ理由でMAX_Kは小さく保つ（高kはbenchmarkのmany_regressors
// シナリオでカバー）。
const MAX_K: usize = 4;

/// `probit_case_strategy`が生成するタプル: `(n, k, x_cols, beta, u, keys)`。
type ProbitCase = (usize, usize, Vec<Vec<f64>>, Vec<f64>, Vec<f64>, Vec<u64>);

/// `(n, k, x_cols, beta, u, keys)`を生成する共通ストラテジ。
/// `logit/fit/tests/proptests.rs`の`logit_case_strategy`と同じ較正値（`x_cols`は`-2.0..2.0`、
/// `beta`は`-1.0..1.0`、`n=k+20..=100`）を使う。Probitはロジスティック分布より
/// 裾が薄い標準正規分布を使うため、同じ`beta`の大きさでもLogitより飽和
/// （p≈0/1）しやすいが、この範囲では分離を起こさないことを実測確認済み。
fn probit_case_strategy() -> impl Strategy<Value = ProbitCase> {
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

/// 真の`beta`から線形予測子`z`・標準正規CDF確率`p=Φ(z)`を計算し、`u`との比較で
/// 二値`y`をサンプリングする。
fn simulate_y(n: usize, x_cols: &[Vec<f64>], beta: &[f64], u: &[f64]) -> Vec<f64> {
    let normal = Normal::standard();
    (0..n)
        .map(|i| {
            let mut z = beta[0];
            for (j, x_col) in x_cols.iter().enumerate() {
                z += x_col[i] * beta[j + 1];
            }
            let p = normal.cdf(z);
            if u[i] < p { 1.0 } else { 0.0 }
        })
        .collect()
}

/// `solver`ごとの`tol`既定値の解決は`logit/fit/tests/proptests.rs`の`default_options`と同じ
/// （`docs/spec/probit-spec.md`3.2節、Logitと共有する`run_solver`のため
/// 同じ意味論）。
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

fn solver_strategy() -> impl Strategy<Value = SolverType> {
    prop_oneof![
        Just(SolverType::Newton),
        Just(SolverType::Bfgs),
        Just(SolverType::Lbfgs),
    ]
}

/// `logit/fit/tests/proptests.rs`と同じ理由で`1e-6`（OLSの列順序不変性）より緩い`1e-4`を使う
/// （反復最適化の収束経路が列順序で変わりうるため）。
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

    /// MLEの一次条件（スコア方程式）: 収束点で`Σᵢλᵢxᵢ ≈ 0`が全パラメータ列で
    /// 成り立つ（`λᵢ = qᵢφ(qᵢzᵢ)/Φ(qᵢzᵢ)`、`qᵢ=2yᵢ-1`。`ProbitProblem::gradient`
    /// が計算するのと同じ一般化残差、`clamped_pdf_cdf`で数値安定化）。
    ///
    /// **Logitの同名プロパティ（`Σᵢ(yᵢ-pᵢ)xᵢ≈0`）をそのまま移植すると誤り**:
    /// Probitのリンク関数`Φ`はLogitの`Λ`と異なり`dp/dz=φ(z)`が尤度の分母
    /// `p(1-p)`と綺麗にキャンセルしないため、素朴な`(yᵢ-pᵢ)xᵢ`の和は収束点でも
    /// ゼロに近づかない（実際に近似縮退列でこの式を使うテストを書いたところ、
    /// 3手法（Newton/Bfgs/Lbfgs）が同一パラメータへ収束したにもかかわらずスコアが
    /// `0.85`前後になり、一見BFGS固有の収束判定バグに見えるが実際はテスト側の
    /// スコア式の誤りだった）。
    /// 許容誤差の根拠は`logit/fit/tests/proptests.rs`の同名プロパティのdocコメント参照
    /// （`n`非依存の絶対閾値、実測でも同オーダーに収まることを確認済み）。
    #[test]
    fn score_is_near_zero_at_converged_params(
        (n, k, x_cols, beta, u, _keys) in probit_case_strategy(),
        solver in solver_strategy(),
    ) {
        let y = simulate_y(n, &x_cols, &beta, &u);
        let names = x_names(k);
        let input = ProbitInput::from_columns(&y, &x_cols, names, true, "y".to_string()).unwrap();
        let result = ProbitEstimator::fit(input, default_options(CovType::Classical, solver));
        prop_assume!(result.is_ok());
        let est = result.unwrap();

        let params = est.params();
        let x = est.input().x();
        let normal = Normal::standard();
        for j in 0..=k {
            let score: f64 = (0..n)
                .map(|i| {
                    let z: f64 = (0..=k).map(|c| params[c] * x.get(i, c)).sum();
                    let q = 2.0 * y[i] - 1.0;
                    let (phi, big_phi) = clamped_pdf_cdf(&normal, q * z);
                    let lambda = q * phi / big_phi;
                    lambda * x.get(i, j)
                })
                .sum();
            prop_assert!(
                score.abs() <= 1e-4,
                "score[{j}] should be ~0, got {score} (solver={solver:?})"
            );
        }
    }

    /// xの列順序を入れ替えても、係数名で対応付ければ係数・標準誤差の値は変わらない。
    #[test]
    fn coefficients_and_se_are_invariant_to_column_order(
        (n, k, x_cols, beta, u, keys) in probit_case_strategy()
            .prop_filter("need >=2 columns to permute", |(_, k, _, _, _, _)| *k >= 2),
        solver in solver_strategy(),
    ) {
        let y = simulate_y(n, &x_cols, &beta, &u);
        let names = x_names(k);
        let input1 = ProbitInput::from_columns(&y, &x_cols, names.clone(), true, "y".to_string()).unwrap();
        let result1 = ProbitEstimator::fit(input1, default_options(CovType::Classical, solver));
        prop_assume!(result1.is_ok());
        let est1 = result1.unwrap();

        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by_key(|&i| keys[i]);
        let permuted_x: Vec<Vec<f64>> = order.iter().map(|&i| x_cols[i].clone()).collect();
        let permuted_names: Vec<String> = order.iter().map(|&i| names[i].clone()).collect();

        let input2 = ProbitInput::from_columns(&y, &permuted_x, permuted_names, true, "y".to_string()).unwrap();
        let result2 = ProbitEstimator::fit(input2, default_options(CovType::Classical, solver));
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

    /// HC0の標準誤差は常にHC1以下（`docs/spec/probit-spec.md`3.3節）。
    #[test]
    fn hc0_std_errors_are_at_most_hc1_std_errors(
        (n, k, x_cols, beta, u, _keys) in probit_case_strategy(),
        solver in solver_strategy(),
    ) {
        let y = simulate_y(n, &x_cols, &beta, &u);
        let names = x_names(k);
        let input1 = ProbitInput::from_columns(&y, &x_cols, names.clone(), true, "y".to_string()).unwrap();
        let result1 = ProbitEstimator::fit(input1, default_options(CovType::Hc0, solver));
        prop_assume!(result1.is_ok());
        let est_hc0 = result1.unwrap();

        let input2 = ProbitInput::from_columns(&y, &x_cols, names, true, "y".to_string()).unwrap();
        let result2 = ProbitEstimator::fit(input2, default_options(CovType::Hc1, solver));
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
