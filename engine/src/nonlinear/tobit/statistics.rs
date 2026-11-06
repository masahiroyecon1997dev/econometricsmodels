//! `fit`の補助: OLSによる初期値、対数尤度、Wald検定、非打ち切り観測の有無の検査。

use super::problem::TobitProblem;
use crate::nonlinear::common::{MleError, checked_design_matrix_qr, compensated_sum};
use faer::Mat;
use faer::prelude::SolveLstsq;
use statrs::distribution::Normal;

/// 非打ち切り観測（`y`が`lower`/`upper`いずれの境界にも一致しない観測）が1件も無い場合、
/// Tobitモデルは典型的には識別不能になる（`MleError::NoUncensoredObservations`のdoc
/// コメント参照。**厳密には十分条件であり必要条件ではない**——`x`による打ち切り
/// カテゴリの完全分離が実際の退化条件——点に注意。詳細は
/// `MleError::NoUncensoredObservations`のdocコメント参照）。
///
/// `TobitInput::from_columns`（構造的妥当性: 境界指定自体の整合性・`y`と境界の整合性）
/// ではなく`fit()`冒頭（推定可能性の前提条件）で検証する。
/// `validate_sufficient_observations`と同じ位置づけ: 「データとして
/// 構造的に妥当か」と「このデータで推定を試みる価値があるか」を分離し、後者を`fit()`側の
/// 責務とする設計方針（`from_columns`はTobitInput単体で完結する検証のみを行い、
/// 推定アルゴリズムの成否に関わる検証は持ち込まない）。
///
/// 打ち切り判定は`censoring_fit_check`と同じ、`yᵢ==lower`/`yᵢ==upper`という浮動小数点の
/// 完全一致比較（`TobitInput::from_columns`が`y`を変換せず保持する設計と整合）。
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

/// 打ち切りを無視した単純なOLS（`X'Xβ=X'y`の最小二乗解）から、Newton法の初期値
/// `(β, logσ)`を計算する（`tobit.rs`のモジュールdoc「Newton法の初期値」節参照）。`x`は
/// `TobitScaling::standardize_x`で標準化済みの設計行列を渡す想定（`fit()`が最適化に使う
/// 空間と一致させ、初期値をそのまま`run_solver`に渡せるようにするため）。`y`も
/// `TobitScaling::standardize_y`でスケーリング済みの値を渡す。`y`は打ち切りを無視し、
/// 観測された値をそのまま連続値として扱う（あくまで初期値のヒューリスティックで
/// あり、Tobitの推定値そのものではない）。
///
/// 特異性検出（列ピボットQRの`R`対角成分の相対閾値、`.claude/rules/rust-style.md`
/// 「線形代数」）は`nonlinear::common::checked_design_matrix_qr`に委譲する（
/// Logit/Probitの`ols_based_initial_params`とランクチェックを共通化した。Logit/Probitは
/// 同じ関数のQR解に加えてリンクのスケール補正を施すが、Tobitの`β`はOLSと同一スケールの
/// ため補正は不要で、QR解をそのまま`β`初期値に使い、`σ`初期値だけ残差から別途求める）。
///
/// `x`は標準化済みの設計行列、`x_original`は標準化前の設計行列（共線な列の特定に使う。
/// `checked_design_matrix_qr`参照）。
///
/// # Errors
/// `x`が特異（完全な多重共線性等）な場合は`MleError::SingularDesignMatrix`を返す
/// （`MleError::SingularDesignMatrix`のdocコメント参照。Newton法が一度も反復していない
/// 段階で発生しうるため`SingularHessian`とは区別する）。
pub(super) fn ols_initial_params(
    x: &Mat<f64>,
    x_original: &Mat<f64>,
    y: &Mat<f64>,
    names: &[String],
) -> Result<Vec<f64>, MleError> {
    let n = x.nrows();
    let k = x.ncols();

    let qr = checked_design_matrix_qr(x, x_original, names)?;
    let beta_mat = qr.solve_lstsq(y);
    let beta: Vec<f64> = (0..k).map(|i| *beta_mat.get(i, 0)).collect();

    let residuals = y - x * &beta_mat;
    let sse: f64 = (0..n).map(|i| (*residuals.get(i, 0)).powi(2)).sum();
    // 完全な当てはまり（sse≈0、退化ケース）ではlog(0)=-infになり最適化が破綻するため、
    // あくまで初期値のヒューリスティックとしてσ=1（s=0）にフォールバックする。
    // `sse > 0.0`という厳密比較ではなく`y`のスケールに対する相対閾値を使う
    // （`.claude/rules/rust-style.md`「線形代数」の相対閾値方針。QRによる最小二乗解の
    // 浮動小数点丸め誤差により、数学的には完全な当てはまりでも`sse`がわずかに正の
    // 微小値になりうるため、`> 0.0`だと退化ケースを見逃す）。
    let candidate_sigma = (sse / n as f64).sqrt();
    let y_scale = (0..n)
        .map(|i| (*y.get(i, 0)).abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let sigma = if candidate_sigma > 1e-8 * y_scale {
        candidate_sigma
    } else {
        1.0
    };

    let mut params = beta;
    params.push(sigma.ln());
    Ok(params)
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

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn ols_initial_params_falls_back_to_sigma_one_for_perfect_fit() {
        // 完全な当てはまり（残差ゼロ）ではsse=0となり、log(0)=-infを避けるため
        // σ=1（s=0）にフォールバックする分岐を検証する。
        let xs = [1.0, 2.0, 3.0, 4.0];
        let x = Mat::from_fn(4, 2, |i, j| if j == 0 { 1.0 } else { xs[i] });
        let y = Mat::from_fn(4, 1, |i, _| 2.0 * xs[i] + 1.0); // y=2x+1、完全に当てはまる

        let params = ols_initial_params(&x, &x, &y, &[]).unwrap();

        assert_eq!(params.len(), 3);
        assert!((params[0] - 1.0).abs() < 1e-9, "intercept={}", params[0]);
        assert!((params[1] - 2.0).abs() < 1e-9, "slope={}", params[1]);
        assert_eq!(params[2], 0.0, "s(=ln sigma)={}", params[2]);
    }
}
