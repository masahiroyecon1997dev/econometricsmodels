//! `fit`の補助: 対数尤度と、非打ち切り観測の有無の検査。

use super::problem::TobitProblem;
use crate::nonlinear::common::{MleError, compensated_sum};
use faer::Mat;
use statrs::distribution::Normal;

/// 非打ち切り観測（`y`が`lower`/`upper`いずれの境界にも一致しない観測）が1件も無い場合、
/// Tobitモデルは典型的には識別不能になる（`MleError::NoUncensoredObservations`のdoc
/// コメント参照。**厳密には十分条件であり必要条件ではない**——`x`による打ち切り
/// カテゴリの完全分離が実際の退化条件——点に注意。詳細は
/// `MleError::NoUncensoredObservations`のdocコメント参照）。
///
/// `TobitInput::validate_censoring_bounds`（構造的妥当性: 境界指定自体の整合性・`y`と境界の整合性）
/// より後の、`fit()`の最後の検査（推定可能性の前提条件）として検証する。
/// `validate_sufficient_observations`と同じ位置づけ: 「データとして
/// 構造的に妥当か」と「このデータで推定を試みる価値があるか」を分離し、後者を推定可能性の
/// 検査とする設計方針（境界の検証は`TobitInput`単体で完結し、
/// 推定アルゴリズムの成否に関わる検証は持ち込まない）。
///
/// 打ち切り判定は`censoring_fit_check`と同じ、`yᵢ==lower`/`yᵢ==upper`という浮動小数点の
/// 完全一致比較（`TobitInput`が`y`を変換せず保持する設計と整合）。
///
/// # Errors
/// 非打ち切り観測が1件も無い場合は`MleError::NoUncensoredObservations`を返す。
pub(super) fn validate_has_uncensored_observations(
    y: &Mat<f64>,
    lower: Option<f64>,
    upper: Option<f64>,
) -> Result<(), MleError> {
    let has_uncensored = (0..y.nrows()).any(|i| {
        let value = *y.get(i, 0);
        let at_lower = lower.is_some_and(|l| value == l);
        let at_upper = upper.is_some_and(|u| value == u);
        !at_lower && !at_upper
    });
    if has_uncensored {
        Ok(())
    } else {
        Err(MleError::NoUncensoredObservations { lower, upper })
    }
}

/// 対数尤度 `ℓ(θ) = Σᵢ ℓᵢ(θ)`（`tobit.rs`のモジュールdocの数式表参照）を`x`・`y`・打ち切り境界・
/// `params`（`(β, s)`、`k+1`次元）から直接計算する。`TobitProblem::cost`（`-ℓ(θ)`、argminの
/// `CostFunction`）と同じ`Contribution::log_lik`の総和を使うが、argminのトレイトが要求する
/// `Result`型を経由する必要が無い内部専用の計算（適合度統計量向け、収束後に1回だけ評価する）
/// のため、`LogitEstimator::log_likelihood`と同じ理由で独立した関数として切り出している。
/// `TobitProblem::from_standardized`は名前に反して標準化済みかどうかを問わない
/// （`x`・`params`のスケールが対応してさえいればよい）ため、ここでは元のスケールの`x`・
/// `params`をそのまま渡す。
pub(super) fn log_likelihood(
    x: &Mat<f64>,
    y: &Mat<f64>,
    lower: Option<f64>,
    upper: Option<f64>,
    params: &[f64],
) -> f64 {
    let problem = TobitProblem::from_standardized(x.clone(), y.clone(), lower, upper);
    let normal = Normal::standard();
    // `TobitProblem::cost`と同じ補償和にし、最適化で評価した値と適合度統計量の値を揃える。
    compensated_sum((0..x.nrows()).map(|i| problem.contribution(i, params, &normal).log_lik))
}
