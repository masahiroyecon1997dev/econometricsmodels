//! property-basedテスト。`logit/fit/tests/proptests.rs`/`probit/fit/tests/proptests.rs`と同型の設計だが、
//! Tobitは打ち切り回帰でありLogit/Probitと異なり(a)説明変数の係数`β`に加え誤差項の
//! 標準偏差`σ`（`s=logσ`でパラメータ化）を持つ、(b)観測ごとの尤度が非打ち切り/左
//! 打ち切り/右打ち切りの3分岐になる、という構造上の違いがある。そのため
//! `score_is_near_zero_at_converged_params`は3分岐+σパラメータ化のFOCを独立に
//! 再導出せず、`TobitProblem::scores()`（本番のOPG/サンドイッチSE計算が使うのと
//! 同じ観測ごとのスコア行列）を再利用する（Probitの
//! `clamped_pdf_cdf`共有と同種のトレードオフで、`scores()`自体のバグはこの
//! プロパティでは検出できない）。
//!
//! **単なる重複ではない**: `fit()`は`TobitScaling`で標準化した空間で最適化し、
//! `est.params()`/`est.sigma()`は逆変換（destandardization）後の値。このプロパティは
//! `TobitProblem::new(est.input())`（生スケール）に`est.params()`/`est.sigma().ln()`を
//! 渡してFOCを評価するため、多数のランダム構成で標準化⇔生スケールの逆変換自体を
//! 実質的に検証する（既存の固定テスト`gradient_matches_numerical_differentiation_of_
//! cost`/`scores_sum_to_negative_gradient`はいずれも`TobitScaling`を経由しないため、
//! この往復変換を検証できていなかった）。標準化パラメータ空間でスコアが0なら生
//! スケールでもスコアが0になることは、標準化がθに依存しないアフィン変換であり
//! 対数尤度の差が定数項（θ非依存のヤコビアン項）のみであることから理論的に導ける。
//!
//! ケース生成は左打ち切り（`lower=0`のみ）・右打ち切り（`upper=0`のみ）・両側打ち切り
//! （`lower=-1`・`upper=1`）の3通りを`censoring_strategy`でランダムに選ぶ
//! （`censored_contribution`の`direction=-1.0`分岐や左右混在のデータセットも通る）。
//! `y* = β₀ + Σxⱼβⱼ + σε`（`ε`は標準正規、`u∈(1e-6, 1-1e-6)`の逆CDFでサンプリング）
//! を計算し、下限未満・上限超過の観測をそれぞれの境界値で打ち切る。
//!
//! 較正値（`n=k+30..=80`, `beta∈[-1,1]`, `sigma∈[0.5,2]`）は`PROPTEST_CASES=5000`
//! （デフォルト256の約20倍）まで増やしても`NoUncensoredObservations`等による
//! `prop_assume`棄却が原因の失敗（"too many global rejects"）を起こさないことを
//! 実測済み。
//!
//! プロパティの有効性検証（バグ注入→検出確認→元に戻す）は当初の3件について実施済み
//! （4件目の`negating_y_and_swapping_bounds_negates_coefficients_only`は右打ち切り・
//! 両側打ち切りへの拡張と同時に追加し、右打ち切り分岐へのバグ注入で検出を確認した）:
//! `score_is_near_zero_at_converged_params`は`TobitProblem::gradient`の`grad[j]`
//! 計算に定数オフセットを注入、`coefficients_and_se_are_invariant_to_column_order`は
//! `TobitInput::from_columns`の`param_names`を逆順にするバグを注入、
//! `hc0_std_errors_are_at_most_hc1_std_errors`は`nonlinear::common::
//! sandwich_cov_params`のHC1補正係数を反転（`n/(n-k)`→`(n-k)/n`、Logit/Probitと
//! 共有するロジックのため同一のバグ注入で確認）するバグを注入し、いずれも検出
//! できることを確認した。
//!
//! `ExpectedUncensored`の3件（`decomposition_identity_holds_for_random_inputs`・
//! `expected_uncensored_stays_within_bounds_and_w_in_unit_interval`・
//! `mirror_symmetry_negates_mean_and_derivative_and_keeps_weight`）は`fit()`を介さず
//! `predicted_value`・`target_w_and_s`を乱数の`(μ,σ,境界)`で直接検証する。バグ注入で
//! 有効性を確認済み: 下側の裾の鏡像で`(Pa,Pb)`・`(A,B)`の入れ替えを外す→3件とも失敗、
//! 平均を`μ+σ(Pa+Pb)`にする→3件とも失敗、`∂w/∂μ`の符号を誤る→鏡像対称性が失敗。

use crate::nonlinear::common::CovType;
use crate::nonlinear::common::MleFitOptions;
use crate::nonlinear::common::SolverType;
use crate::nonlinear::tobit::TobitEstimator;
use crate::nonlinear::tobit::TobitInput;
use crate::nonlinear::tobit::problem::TobitProblem;
use proptest::collection;
use proptest::prelude::*;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

// 高kはbenchmarkのmany_regressorsシナリオでカバー済みのため小さく保つ
// （logit・probitの`fit/tests/proptests.rs`と同じ理由）。
const MAX_K: usize = 3;

/// `tobit_case_strategy`が生成するタプル: `(n, k, x_cols, beta, sigma, u, keys)`。
/// `k`は切片を除いた説明変数の数（`beta`は切片込みで`k+1`要素）。
type TobitCase = (
    usize,
    usize,
    Vec<Vec<f64>>,
    Vec<f64>,
    f64,
    Vec<f64>,
    Vec<u64>,
);

fn tobit_case_strategy() -> impl Strategy<Value = TobitCase> {
    (1..=MAX_K).prop_flat_map(|k| {
        (k + 30..=80usize).prop_flat_map(move |n| {
            (
                Just(n),
                Just(k),
                collection::vec(collection::vec(-2.0f64..2.0, n), k),
                collection::vec(-1.0f64..1.0, k + 1),
                0.5f64..2.0,
                collection::vec(1e-6f64..1.0 - 1e-6, n),
                collection::vec(any::<u64>(), k),
            )
        })
    })
}

fn x_names(k: usize) -> Vec<String> {
    (1..=k).map(|i| format!("x{i}")).collect()
}

/// 左打ち切りの下限（打ち切りなしのTobitの意味が薄れない程度に、真の`y*`分布の
/// 中心付近を狙う）。右打ち切りの上限はこの値と対称（`beta`の分布が0周りに対称な
/// ため、左・右で打ち切り割合が同程度になる）。
const LOWER_BOUND: f64 = 0.0;

/// 打ち切り境界`(lower, upper)`。
type Bounds = (Option<f64>, Option<f64>);

/// 左打ち切り・右打ち切り・両側打ち切りをランダムに選ぶ。両側打ち切りの幅
/// （`[-1, 1]`）は、`NoUncensoredObservations`等の`prop_assume`棄却が過剰にならない
/// 程度に真の`y*`分布の中心付近を含める。
fn censoring_strategy() -> impl Strategy<Value = Bounds> {
    prop_oneof![
        Just((Some(LOWER_BOUND), None)),
        Just((None, Some(LOWER_BOUND))),
        Just((Some(-1.0), Some(1.0))),
    ]
}

/// 真の`beta`・`sigma`から`y* = β₀+Σxⱼβⱼ+σε`を計算し、`bounds`で打ち切る
/// （下限未満は下限、上限超過は上限に置き換える）。
fn simulate_y(
    n: usize,
    x_cols: &[Vec<f64>],
    beta: &[f64],
    sigma: f64,
    u: &[f64],
    bounds: Bounds,
) -> Vec<f64> {
    let normal = Normal::standard();
    (0..n)
        .map(|i| {
            let mut y_star = beta[0];
            for (j, x_col) in x_cols.iter().enumerate() {
                y_star += x_col[i] * beta[j + 1];
            }
            y_star += sigma * normal.inverse_cdf(u[i]);
            if let Some(lower) = bounds.0
                && y_star < lower
            {
                return lower;
            }
            if let Some(upper) = bounds.1
                && y_star > upper
            {
                return upper;
            }
            y_star
        })
        .collect()
}

/// `solver`ごとの`tol`既定値の解決は`logit/fit/tests/proptests.rs`/`probit/fit/tests/proptests.rs`の`default_options`と
/// 同じ（`docs/spec/tobit-spec.md`、Logit/Probitと共有する`run_solver`のため
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

/// `logit/fit/tests/proptests.rs`/`probit/fit/tests/proptests.rs`と同じ理由で`1e-6`より緩い`1e-4`を使う
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

    /// MLEの一次条件（スコア方程式）: 収束点で`Σᵢsᵢ ≈ 0`（`sᵢ`は
    /// `TobitProblem::scores()`が返す観測ごとのスコア行、`β`の`k+1`列＋
    /// `s=logσ`の1列、計`k+2`列）が全パラメータ列で成り立つ。許容誤差の根拠は
    /// `logit/fit/tests/proptests.rs`の同名プロパティのdocコメント参照（`n`非依存の絶対閾値）。
    #[test]
    fn score_is_near_zero_at_converged_params(
        (n, k, x_cols, beta, sigma, u, _keys) in tobit_case_strategy(),
        bounds in censoring_strategy(),
        solver in solver_strategy(),
    ) {
        let y = simulate_y(n, &x_cols, &beta, sigma, &u, bounds);
        let names = x_names(k);
        let input = TobitInput::from_columns(
            &y, &x_cols, names, true, "y".to_string(), bounds.0, bounds.1,
        ).unwrap();
        let result = TobitEstimator::fit(input, default_options(CovType::Classical, solver));
        prop_assume!(result.is_ok());
        let est = result.unwrap();

        let mut full_params: Vec<f64> = est.params().to_vec();
        full_params.push(est.sigma().ln());

        let problem = TobitProblem::new(est.input());
        let scores = problem.scores(&full_params);
        for j in 0..=(k + 1) {
            let score: f64 = (0..n).map(|i| *scores.get(i, j)).sum();
            prop_assert!(
                score.abs() <= 1e-4,
                "score[{j}] should be ~0, got {score} (solver={solver:?})"
            );
        }
    }

    /// xの列順序を入れ替えても、係数名で対応付ければ係数・標準誤差の値は変わらない。
    /// `σ`は列順序に依存しない別パラメータのため、`std_errors()`の末尾要素
    /// （インデックス`k+1`）で個別に比較する。
    #[test]
    fn coefficients_and_se_are_invariant_to_column_order(
        (n, k, x_cols, beta, sigma, u, keys) in tobit_case_strategy()
            .prop_filter("need >=2 columns to permute", |(_, k, _, _, _, _, _)| *k >= 2),
        bounds in censoring_strategy(),
        solver in solver_strategy(),
    ) {
        let y = simulate_y(n, &x_cols, &beta, sigma, &u, bounds);
        let names = x_names(k);
        let input1 = TobitInput::from_columns(
            &y, &x_cols, names.clone(), true, "y".to_string(), bounds.0, bounds.1,
        ).unwrap();
        let result1 = TobitEstimator::fit(input1, default_options(CovType::Classical, solver));
        prop_assume!(result1.is_ok());
        let est1 = result1.unwrap();

        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by_key(|&i| keys[i]);
        let permuted_x: Vec<Vec<f64>> = order.iter().map(|&i| x_cols[i].clone()).collect();
        let permuted_names: Vec<String> = order.iter().map(|&i| names[i].clone()).collect();

        let input2 = TobitInput::from_columns(
            &y, &permuted_x, permuted_names, true, "y".to_string(), bounds.0, bounds.1,
        ).unwrap();
        let result2 = TobitEstimator::fit(input2, default_options(CovType::Classical, solver));
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
        assert_approx_eq(est2.sigma(), est1.sigma(), "sigma under column permutation");
        assert_approx_eq(se2[k + 1], se1[k + 1], "std_error[sigma] under column permutation");
    }

    /// HC0の標準誤差は常にHC1以下（`docs/spec/tobit-spec.md`）。
    #[test]
    fn hc0_std_errors_are_at_most_hc1_std_errors(
        (n, k, x_cols, beta, sigma, u, _keys) in tobit_case_strategy(),
        bounds in censoring_strategy(),
        solver in solver_strategy(),
    ) {
        let y = simulate_y(n, &x_cols, &beta, sigma, &u, bounds);
        let names = x_names(k);
        let input1 = TobitInput::from_columns(
            &y, &x_cols, names.clone(), true, "y".to_string(), bounds.0, bounds.1,
        ).unwrap();
        let result1 = TobitEstimator::fit(input1, default_options(CovType::Hc0, solver));
        prop_assume!(result1.is_ok());
        let est_hc0 = result1.unwrap();

        let input2 = TobitInput::from_columns(
            &y, &x_cols, names, true, "y".to_string(), bounds.0, bounds.1,
        ).unwrap();
        let result2 = TobitEstimator::fit(input2, default_options(CovType::Hc1, solver));
        prop_assume!(result2.is_ok());
        let est_hc1 = result2.unwrap();

        for j in 0..=(k + 1) {
            prop_assert!(
                est_hc0.std_errors()[j] <= est_hc1.std_errors()[j] + 1e-9,
                "HC0 se[{j}]={} should be <= HC1 se[{j}]={}",
                est_hc0.std_errors()[j], est_hc1.std_errors()[j]
            );
        }
    }

    /// 鏡像対称性: `y`と境界を符号反転して`(lower, upper)`→`(-upper, -lower)`にすると
    /// （`-y* = x(-β) + σ(-ε)`、`-ε`も標準正規）、係数は符号反転・`σ`と標準誤差は不変。
    /// 左打ち切りと右打ち切りで別々のコード分岐（`censored_contribution`の
    /// `direction`）を通るため、片方の分岐の符号・スケールのバグがここで顕在化する。
    #[test]
    fn negating_y_and_swapping_bounds_negates_coefficients_only(
        (n, k, x_cols, beta, sigma, u, _keys) in tobit_case_strategy(),
        bounds in censoring_strategy(),
        solver in solver_strategy(),
    ) {
        let y = simulate_y(n, &x_cols, &beta, sigma, &u, bounds);
        let names = x_names(k);
        let input1 = TobitInput::from_columns(
            &y, &x_cols, names.clone(), true, "y".to_string(), bounds.0, bounds.1,
        ).unwrap();
        let result1 = TobitEstimator::fit(input1, default_options(CovType::Classical, solver));
        prop_assume!(result1.is_ok());
        let est1 = result1.unwrap();

        let y_neg: Vec<f64> = y.iter().map(|v| -v).collect();
        let input2 = TobitInput::from_columns(
            &y_neg, &x_cols, names, true, "y".to_string(),
            bounds.1.map(|v| -v), bounds.0.map(|v| -v),
        ).unwrap();
        let result2 = TobitEstimator::fit(input2, default_options(CovType::Classical, solver));
        prop_assume!(result2.is_ok());
        let est2 = result2.unwrap();

        for j in 0..=k {
            assert_approx_eq(est2.params()[j], -est1.params()[j], &format!("param[{j}] under y negation"));
        }
        assert_approx_eq(est2.sigma(), est1.sigma(), "sigma under y negation");
        for j in 0..=(k + 1) {
            assert_approx_eq(est2.std_errors()[j], est1.std_errors()[j], &format!("std_error[{j}] under y negation"));
        }
    }
}

/// 大規模ランダム探索で捕捉した具体的な入力の1つを固定値化した回帰テスト。
///
/// **調査方法**: 評価回数バジェット方式の導入後は、line
/// searchの暴走が数秒程度の有限時間で`Err`に変換されるようになったため、
/// `kill`前提の外部タイムアウト無しで大量試行を安全に回せることを利用し、
/// `tobit_case_strategy()`・`solver_strategy()`を`TestRunner`で直接
/// サンプリングして30万回試行する使い捨ての探索ハーネスを一時的に実装し
/// 実行した（コミット履歴には残さず、本テストのみを結果として残す）。
///
/// **判明した事実（当時）**: 30万試行中8件が`MleError::EvaluationBudgetExceeded`
/// 相当のエラー（`SolverType::Lbfgs`は当時のargmin組み込みLBFGSの`SolverExit`経由で
/// 同じメッセージを含む計算エラーになっていた）を引き起こした。**8件
/// 全てが`solver=Lbfgs`**で、`solver=Bfgs`・`solver=Newton`は1件もヒットしな
/// かった。打ち切り割合（`n_uncensored/n`）は0.38〜0.80、`n`は41〜75、`k`は
/// 1〜3、`sigma`は0.53〜1.85とヒット全体に幅広く分布しており、「極端な打ち切り・
/// 特定の狭いパラメータ域が必要」という仮説は支持されなかった（8件中7件は
/// 0.65〜0.80という中程度の打ち切り）。真因は当時未確定で、argmin組み込み
/// `LBFGS`固有の何か（limited-memory two-loop recursionの初期スケーリング等）に
/// 起因すると推測されていたが、真因の特定は後続の調査に持ち越された。
///
/// **判明した実際の原因と解消（本テストの現在の主眼）**: `SolverType::Lbfgs`
/// を`FaerLbfgs`（`FaerBfgs`と同型のself-scaling初期化を持つ自前実装）に置き換えた
/// 結果、この入力は**もはや退化せず正常に収束する**（実測: `n_iter=11`、
/// `converged=true`）。`FaerBfgs`で既に確立していた「1回目の
/// 反復専用のline search初期ステップ幅`min(1,1/‖g₀‖)`」を、当時のargmin組み込み
/// `LBFGS`の公開APIでは適用できていなかったこと（1回目の反復は履歴が空で
/// `γ=1.0`固定のため、単位行列の逆Hessianを使う`FaerBfgs`の1回目と同型の
/// 暴走リスクを抱えたままだった）が実際の原因だったと裏付けられた。本テストは
/// 「境界の有限時間`Err`」から「実際に収束する」への挙動変化そのものを固定する
/// 回帰ガードとして残す。
#[test]
fn fit_lbfgs_converges_for_a_previously_stalling_case() {
    let beta = vec![0.9873339499036579, -0.12291865368107788];
    let sigma = 1.4304964761886263;
    let x_cols: Vec<Vec<f64>> = vec![vec![
        0.09730587261408116,
        -0.44608585527013445,
        -1.1780876932682702,
        -1.7218664313287149,
        -1.5668563688839525,
        1.569383650134455,
        -0.8904079788143034,
        -0.38220417882649177,
        -0.6650396911368611,
        1.5455987143552725,
        1.0914058868351664,
        1.5608174706710314,
        -1.0457715928011817,
        0.8460357774450576,
        -1.6746201868364683,
        -0.10662833055410269,
        -1.3549266627206265,
        1.802095632400922,
        0.22454103534302314,
        0.9226514870462461,
        -1.5960509851353555,
        -0.06299762418526257,
        -0.7361961389207361,
        -0.6159276360973034,
        1.6276092545963188,
        1.8777781260428934,
        1.4040046871028768,
        1.6191183983983046,
        0.7928232369566655,
        -1.2215837625042971,
        0.43098012504668753,
        0.6437939214254764,
        1.8726228041330524,
        0.18531824642137915,
        -1.2889799350480626,
        1.3893942618908828,
        0.9546063410017254,
        1.6709682203826888,
        -0.6385965449169932,
        1.801002621573055,
        1.89685718642964,
    ]];
    let u = vec![
        0.3617740489249382,
        0.8638655369835531,
        0.5854529043833009,
        0.17096769280771643,
        0.27549992513298815,
        0.29109304112930307,
        0.23572824129100706,
        0.6413758528582494,
        0.015031380540545333,
        0.9591408659161148,
        0.03755478548383856,
        0.5531593343087735,
        0.00290634638539887,
        0.6500237097120347,
        0.41895952824744176,
        0.6246296073538906,
        0.672262276485462,
        0.072963646484062,
        0.8961814455371079,
        0.37114140842146587,
        0.07134975672937831,
        0.42084724899948256,
        0.612656348188821,
        0.03682767217809841,
        0.7090082364949691,
        0.30669875984016237,
        0.9155739980338565,
        0.022268595803044702,
        0.9400939907877396,
        0.922790135057218,
        0.05561252335283864,
        0.9567466460014393,
        0.1117763895368753,
        0.16362361406720183,
        0.41560137699383287,
        0.08196797976110878,
        0.25881610622077167,
        0.33012720695491854,
        0.6634215047000254,
        0.23962574010778723,
        0.9444306832294833,
    ];
    let n = 41;

    let y = simulate_y(n, &x_cols, &beta, sigma, &u, (Some(LOWER_BOUND), None));
    let names = x_names(1);
    let input = TobitInput::from_columns(
        &y,
        &x_cols,
        names,
        true,
        "y".to_string(),
        Some(LOWER_BOUND),
        None,
    )
    .unwrap();

    let start = std::time::Instant::now();
    let result = TobitEstimator::fit(
        input,
        default_options(CovType::Classical, SolverType::Lbfgs),
    );
    let elapsed = start.elapsed();

    assert!(
        result.is_ok(),
        "expected FaerLbfgs's self-scaling to converge on this input, got {result:?}"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(30),
        "fit() took {elapsed:?}, expected a bounded run within seconds"
    );
    // 実測`n_iter=11`（緩い上限）。`FaerLbfgs`の1回目line
    // search初期ステップ幅の設計変更（`engine/src/nonlinear/CLAUDE.md`
    // 「実装時に踏んだ罠その2」参照）時に、収束はするが反復回数が
    // 大きく増える形の劣化が紛れ込んでいないかを検知するためのガード。
    let n_iter = result.unwrap().n_iter();
    assert!(
        n_iter < 30,
        "n_iter={n_iter}, expected roughly the same order as the observed 11 \
         iterations (a large increase would indicate a regression in FaerLbfgs's \
         first-iteration self-scaling, even though it still converges)"
    );
}
