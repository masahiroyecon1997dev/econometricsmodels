//! 複数のファイルのテストが共有するフィクスチャと独立実装の参照計算。

use super::*;
use faer::Side;
use faer::prelude::Solve;

/// 丁度識別（`len(instruments) == len(x_endog)`）の閉形式解と数値照合するテストデータ。
///
/// 構造式: `y = 1 + 2*x_endog + e`。`x_endog`は`z`に小さなノイズを足した値で、第一段階も
/// 構造式も完全フィットにならない。
pub(super) fn just_identified_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let z = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_endog = vec![1.2, 1.9, 3.3, 3.8, 5.1];
    let noise = [0.1, -0.2, 0.15, -0.05, 0.1];
    let y: Vec<f64> = x_endog
        .iter()
        .zip(noise)
        .map(|(&x, u)| 1.0 + 2.0 * x + u)
        .collect();
    (y, x_endog, z)
}

pub(super) fn just_identified_input() -> IvInput {
    let (y, x_endog, z) = just_identified_data();
    IvInput::from_columns(
        &y,
        &[],
        vec![],
        &[x_endog],
        vec!["x_endog".to_string()],
        &[z],
        vec!["z".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap()
}

/// `x_exog`に実変数（`x1`）を含む、教科書的な2SLSの典型ケース（外生変数＋内生変数＋
/// 操作変数が同時に存在する、過剰識別）のテストデータとfit済み`TwoSlsEstimator`。
/// `(x1, x_endog, z1, z2, y, estimator)`を返す。射影公式・第一段階の閉形式解の
/// 独立検証（`first_stage_estimators_match_independently_recomputed_ols_closed_form`・
/// `fit_matches_independently_recomputed_projection_formula_with_nontrivial_x_exog`）が
/// 同じデータ・同じfit呼び出しを共有するため、ここに集約する（重複していた際、
/// 片方だけデータを更新すると気づかずに非退化性が崩れるリスクがあったため、
/// レビューを受けて統合）。
/// `nontrivial_x_exog_*`ヘルパー群が共有する生データ（`(x1, x_endog, z1, z2, y)`）。
/// cov_typeごとに別々の`IvInput`が必要なテスト（cov_type対応のSE検証群）が同じデータで
/// 独立に`fit()`しなおせるように、フィット済みestimatorとは切り離して公開する。
#[allow(clippy::type_complexity)]
pub(super) fn nontrivial_x_exog_columns() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x_endog = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0];
    let z1 = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let z2 = vec![1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];
    let y = vec![5.0, 3.0, 8.0, 6.0, 11.0, 10.0, 15.0, 13.0];
    (x1, x_endog, z1, z2, y)
}

/// `nontrivial_x_exog_columns()`から`IvInput`を組み立てる（`IvInput`は`Clone`を
/// 実装しないため、cov_typeごとに別々の`fit()`呼び出しが必要なテストは、この関数を
/// その都度呼んで新しい`IvInput`を得る）。
pub(super) fn nontrivial_x_exog_input() -> IvInput {
    let (x1, x_endog, z1, z2, y) = nontrivial_x_exog_columns();
    IvInput::from_columns(
        &y,
        std::slice::from_ref(&x1),
        vec!["x1".to_string()],
        std::slice::from_ref(&x_endog),
        vec!["endog1".to_string()],
        &[z1, z2],
        vec!["z1".to_string(), "z2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap()
}

/// SE検証テスト群が共有するオラクル: `(X̂, e)`を返す。`X̂`は第二段階の設計行列
/// （`second_stage.input().x()`、サンドイッチ公式の「パン」に使う。`second_stage`は
/// 非公開フィールドだが同一モジュールの子モジュールから直接参照できる）、`e`は構造残差
/// `y - Xβ̂`（`X`は実際の内生変数、`nontrivial_x_exog_columns()`と対応する列順
/// `[const, x1, x_endog]`。サンドイッチ公式の「具」に使う）。`TwoSlsEstimator::fit`
/// とは独立に（`Mat`演算のみで）計算する。
pub(super) fn nontrivial_x_exog_x_hat_and_structural_residuals(
    estimator: &TwoSlsEstimator,
) -> (Mat<f64>, Mat<f64>) {
    let (x1, x_endog, _z1, _z2, y) = nontrivial_x_exog_columns();
    let n = y.len();
    let x_structural = Mat::from_fn(n, 3, |i, j| match j {
        0 => 1.0,
        1 => x1[i],
        _ => x_endog[i],
    });
    let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);
    let e = &y_mat - &x_structural * estimator.params();
    let x_hat = estimator.second_stage.input().x().clone();
    (x_hat, e)
}

/// `x_endog=[]`（第一段階ループが一度も回らない退化ケース）の`IvInput`を組み立てる。
/// サンドイッチ計算（`fit()`本体、第二段階）固有のcov_typeエラー
/// （`InvalidHacLags`/`MissingClusterColumn`/`InsufficientClusters`）を検証するテストで
/// 使う。`x_endog`が1つでもあると、同じ`cov_type`が第一段階にも渡るため
/// （`two_sls.rs`のモジュールdoc参照）、エラーは先に`FirstStageFailed`として
/// 発生してしまい、第二段階固有の経路を検証できない。
pub(super) fn x_endog_empty_input() -> IvInput {
    let y = vec![2.1, 3.8, 6.3, 7.9, 10.2];
    let x_exog = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    IvInput::from_columns(
        &y,
        &x_exog,
        vec!["x1".to_string()],
        &[],
        vec![],
        &[],
        vec![],
        true,
        "y".to_string(),
    )
    .unwrap()
}

/// クラスターロバストSEのテスト専用データ・グループ。`nontrivial_x_exog_input()`
/// （操作変数2個）だと第一段階の傾き係数`q=3`（x1, z1, z2）に対しクラスター数`G=4`が
/// 際どく、`Ŝ`（rank≤G）の傾き部分行列が数値的にほぼ特異になり`fit()`自体が
/// `NearSingularMatrix`で失敗する（`engine/src/linear/CLAUDE.md`「クラスター数`G`と
/// 傾き係数の数`q`の関係」参照、IVの第一段階にも同じ制約が当てはまる）。操作変数を
/// 1個に減らし（`q=2`）安全な余裕を持たせた専用データを使う。
pub(super) fn cluster_test_input_and_groups() -> (IvInput, Vec<String>) {
    let y = vec![5.0, 3.0, 8.0, 6.0, 11.0, 10.0, 15.0, 13.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x_endog = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0];
    let z1 = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let groups = vec![
        "a".to_string(),
        "a".to_string(),
        "b".to_string(),
        "b".to_string(),
        "c".to_string(),
        "c".to_string(),
        "d".to_string(),
        "d".to_string(),
    ];
    let input = IvInput::from_columns(
        &y,
        &[x1],
        vec!["x1".to_string()],
        &[x_endog],
        vec!["endog1".to_string()],
        &[z1],
        vec!["z1".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();
    (input, groups)
}

/// 正規方程式`β=(X'X)⁻¹X'y`を`faer`演算で直接解く、SUT（`partial_wald_statistic`・
/// `OlsEstimator`）とは独立した最小限のOLSオラクル（SEや検定統計量は不要なため
/// 点推定のみ）。
pub(super) fn manual_ols_beta(x: &Mat<f64>, y: &Mat<f64>) -> Mat<f64> {
    let xtx = x.transpose() * x;
    let xty = x.transpose() * y;
    let k = x.ncols();
    xtx.llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(k, k))
        * &xty
}

/// `(X'X)⁻¹`を`TwoSlsEstimator::fit`内部の`xtx_inverse`とは別に（テストのオラクルとして）
/// 直接計算する。
pub(super) fn manual_xtx_inverse(x: &Mat<f64>, k: usize) -> Mat<f64> {
    let xtx = x.transpose() * x;
    xtx.llt(Side::Lower)
        .unwrap()
        .solve(Mat::<f64>::identity(k, k))
}

/// classicalの標準誤差を素朴な式（`σ̂²(X'X)⁻¹`の対角成分の平方根）で独立計算する。
pub(super) fn manual_classical_std_errors(
    x: &Mat<f64>,
    e: &Mat<f64>,
    n: usize,
    k: usize,
) -> Vec<f64> {
    let ssr: f64 = (0..n).map(|i| (*e.get(i, 0)).powi(2)).sum();
    let sigma2 = ssr / ((n - k) as f64);
    let xtx_inv = manual_xtx_inverse(x, k);
    (0..k)
        .map(|j| (sigma2 * (*xtx_inv.get(j, j))).sqrt())
        .collect()
}
