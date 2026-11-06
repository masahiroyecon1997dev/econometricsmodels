//! `Fitted`（限界効果・予測・打ち切りの適合度確認・`from_parts`）のテスト。機能別に子モジュールへ分ける。
//!
//! - `marginal_effects.rs`: 限界効果（期待値・非打ち切り確率）
//! - `predict.rs`: `predict`・`predict_new_data`・`from_parts`による復元
//! - `censoring_check.rs`: 打ち切りの適合度確認

use crate::nonlinear::tobit::TobitInput;
use statrs::distribution::Continuous;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

mod censoring_check;
mod marginal_effects;
mod predict;

// ── 複数のテーマが使うヘルパー ──────────────────────────────────────

/// `censored_regression_input`（左打ち切りのみ、`lower=Some(0.0)`）の右打ち切り版。
/// 潜在変数`y* = -5 + 2x + noise`は同じだが、`upper=Some(6.0)`で上側を打ち切る
/// （x=6,7,8の潜在値6.85・9.1・10.95が6.0に打ち切られる、打ち切り率37.5%）。
/// `marginal_effects`が右打ち切りのみのケースでも正しく計算できることを検証する
/// データセット（`target_w_and_s`のdocコメント「数式」参照）。
pub(super) fn right_censored_regression_input() -> TobitInput {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![-2.85, -0.95, 1.15, 2.9, 5.2, 6.0, 6.0, 6.0];
    TobitInput::from_columns(
        &y,
        &[x],
        vec!["x1".to_string()],
        true,
        "y".to_string(),
        None,
        Some(6.0),
    )
    .unwrap()
}

/// `censored_regression_input`の両側打ち切り版。同じ潜在変数`y* = -5 + 2x + noise`に
/// `lower=Some(0.0)`・`upper=Some(6.0)`の両方を適用する（x=1,2は下側、x=6,7,8は上側で
/// 打ち切り、非打ち切りはx=3,4,5のみ）。
pub(super) fn two_sided_censored_regression_input() -> TobitInput {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![0.0, 0.0, 1.15, 2.9, 5.2, 6.0, 6.0, 6.0];
    TobitInput::from_columns(
        &y,
        &[x],
        vec!["x1".to_string()],
        true,
        "y".to_string(),
        Some(0.0),
        Some(6.0),
    )
    .unwrap()
}

/// `E[y|x]`の閉形式（McDonald-Moffitt 1980）。左打ち切りのみ・右打ち切りのみ・両側
/// 打ち切りをそれぞれ独立した分岐で書き下す（`target_w_and_s`の統一的な
/// `boundary_terms`実装とは別のコードパスで検算するため、`(lower,upper)`ごとの
/// 教科書通りの式をそのまま使う）。
pub(super) fn expected_observed_closed_form(
    mu: f64,
    sigma: f64,
    lower: Option<f64>,
    upper: Option<f64>,
) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    match (lower, upper) {
        (Some(c), None) => {
            let z = (mu - c) / sigma;
            normal.cdf(z) * mu + sigma * normal.pdf(z) + (1.0 - normal.cdf(z)) * c
        }
        (None, Some(c)) => {
            let z = (c - mu) / sigma;
            normal.cdf(z) * mu - sigma * normal.pdf(z) + (1.0 - normal.cdf(z)) * c
        }
        (Some(c1), Some(c2)) => {
            let za = (c1 - mu) / sigma;
            let zb = (c2 - mu) / sigma;
            normal.cdf(za) * c1
                + (1.0 - normal.cdf(zb)) * c2
                + (normal.cdf(zb) - normal.cdf(za)) * mu
                - sigma * (normal.pdf(zb) - normal.pdf(za))
        }
        (None, None) => unreachable!("TobitInput::from_columns rejects both-None bounds"),
    }
}

/// `P(uncensored|x)`の閉形式。`expected_observed_closed_form`と同じ理由で
/// `(lower,upper)`ごとに独立した分岐で書く。
pub(super) fn prob_uncensored_closed_form(
    mu: f64,
    sigma: f64,
    lower: Option<f64>,
    upper: Option<f64>,
) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    match (lower, upper) {
        (Some(c), None) => normal.cdf((mu - c) / sigma),
        (None, Some(c)) => normal.cdf((c - mu) / sigma),
        (Some(c1), Some(c2)) => normal.cdf((c2 - mu) / sigma) - normal.cdf((c1 - mu) / sigma),
        (None, None) => unreachable!("TobitInput::from_columns rejects both-None bounds"),
    }
}

/// ノイズの大きい決定論的な打ち切りデータ（`y*=-2+0.7x+ε`、`ε`は`±2.5`の一様、
/// `x=0.25·i`、`i=0..60`）に`lower`/`upper`を適用した入力。既存の少数観測データは
/// 当てはめ後の`σ`が小さく`z`が裾に入るため、素直な閉形式（`expected_uncensored_closed_form`）
/// で独立に検算できる範囲に`z`を収める用途。
pub(super) fn noisy_censored_input(lower: Option<f64>, upper: Option<f64>) -> TobitInput {
    let x: Vec<f64> = (0..60).map(|i| 0.25 * f64::from(i)).collect();
    let y: Vec<f64> = (0..60u32)
        .map(|i| {
            let u = f64::from((i * 7919) % 1009) / 1009.0;
            let latent = -2.0 + 0.7 * x[i as usize] + (u - 0.5) * 5.0;
            let latent = lower.map_or(latent, |l| latent.max(l));
            upper.map_or(latent, |u| latent.min(u))
        })
        .collect();
    TobitInput::from_columns(
        &y,
        &[x],
        vec!["x1".to_string()],
        true,
        "y".to_string(),
        lower,
        upper,
    )
    .unwrap()
}
