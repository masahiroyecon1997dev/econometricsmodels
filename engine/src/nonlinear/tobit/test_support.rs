//! 複数のファイルのテストが共有するフィクスチャ。

use crate::nonlinear::tobit::TobitInput;
use statrs::distribution::Continuous;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

/// 打ち切りが（実質的に）発生しないよう境界を極端に広く取ったデータ。
pub(super) fn intercept_only_uncensored_input(y: &[f64]) -> TobitInput {
    TobitInput::from_columns(
        y,
        &[],
        vec![],
        true,
        "y".to_string(),
        Some(-1000.0),
        Some(1000.0),
    )
    .unwrap()
}

/// 打ち切りが実際に発生するデータ（`x=1,2`が`lower=0`で左打ち切り、残りは非打ち切り、
/// 打ち切り率25%）。OLS（打ち切りを無視）は真のTobit MLEと一致しないため、
/// `fit()`の収束性を「打ち切りが皆無」の既存テストとは独立に検証できる
/// （実際に打ち切りが発生するケースでのNewton収束を確認する）。
pub(super) fn censored_regression_input() -> TobitInput {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // 潜在変数 y* = -5 + 2x + noise（真の係数から意図的にわずかな残差を持たせる。
    // ノイズが皆無だと非打ち切り観測の当てはまりが完全になりσ̂→0という退化した
    // 境界ケースになり、いかなるソルバーでも不安定になることが判明したため
    // （デバッグ時に発見。`ols_initial_params`のsse≈0フォールバックと同種の問題が
    // 真のTobit尤度側でも起こりうる）。
    let y = vec![0.0, 0.0, 1.15, 2.9, 5.2, 6.85, 9.1, 10.95];
    TobitInput::from_columns(
        &y,
        &[x],
        vec!["x1".to_string()],
        true,
        "y".to_string(),
        Some(0.0),
        None,
    )
    .unwrap()
}

/// OPG/HC0/HC1/クラスターSEの独立再計算検証・組み合わせテストで共通に使う多変量
/// データセット（切片+説明変数2つ+`s`で`k+1=4`次元、実際の左打ち切りを含む）。
/// `censored_regression_input`（説明変数1つ）を拡張したもの。情報行列の等式が
/// 有限標本で厳密に成り立ってしまう切片のみ・単変量モデルでは`cov_type`の配線ミス
/// （例えば`Opg`の分岐で誤って`observed_information_cov_params`を呼んでいた場合等）を
/// 検出できないため、Logitの対応テスト（`fit_cov_type_opg_hc0_hc1_match_
/// independently_recomputed_values`のdocコメント）と同じ理由で多変量データセットが
/// 必須。
pub(super) fn multivariate_censored_input() -> TobitInput {
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x2 = vec![1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0];
    let y = vec![0.0, 0.0, 1.15, 2.9, 5.2, 6.85, 9.1, 10.95];
    TobitInput::from_columns(
        &y,
        &[x1, x2],
        vec!["x1".to_string(), "x2".to_string()],
        true,
        "y".to_string(),
        Some(0.0),
        None,
    )
    .unwrap()
}

/// `E[y|lower<y<upper,x]`の閉形式。`(lower,upper)`ごとに独立した分岐で、`φ/W`を
/// 素直に直接計算する教科書通りの式（`1-Φ`や`Φ(zb)-Φ(za)`をそのまま使うため、裾では
/// 桁落ちする。片側は生存関数`Φ(-z)`を使うので裾でも正確だが、両側は中程度の`z`
/// での検算専用。いずれも`truncation_pdf_ratios`とは別のコードパス）。
pub(super) fn expected_uncensored_closed_form(
    mu: f64,
    sigma: f64,
    lower: Option<f64>,
    upper: Option<f64>,
) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    match (lower, upper) {
        (Some(c), None) => {
            let z = (c - mu) / sigma;
            mu + sigma * normal.pdf(z) / normal.cdf(-z)
        }
        (None, Some(c)) => {
            let z = (c - mu) / sigma;
            mu - sigma * normal.pdf(z) / normal.cdf(z)
        }
        (Some(c1), Some(c2)) => {
            let za = (c1 - mu) / sigma;
            let zb = (c2 - mu) / sigma;
            mu + sigma * (normal.pdf(za) - normal.pdf(zb)) / (normal.cdf(zb) - normal.cdf(za))
        }
        (None, None) => unreachable!("TobitInput::from_columns rejects both-None bounds"),
    }
}

pub(super) fn assert_relative_eq(actual: f64, expected: f64, rtol: f64, label: &str) {
    let scale = expected.abs().max(f64::MIN_POSITIVE);
    assert!(
        (actual - expected).abs() <= rtol * scale,
        "{label}: actual={actual:e}, expected={expected:e}, rel={:e}",
        (actual - expected).abs() / scale
    );
}
