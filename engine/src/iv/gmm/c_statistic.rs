//! C統計量（difference-in-Hansen統計量、GMM distance test）。
//!
//! 定義と計算（`Z_full`・`Z_c`の2通りの推定、`S_e`の部分行列`S_c`を重みにした`J_c`、`C = J_e − J_c`の
//! `χ²(|B|)`検定）は、固定効果つきGMMと共有する`shared::gmm_c_statistic`にある。ここは`IvInput`の列を
//! 渡し、`cov_type`から重みの種類を決めるだけの層。

use crate::iv::common::{IvError, IvInput, mat_column_to_vec};
use crate::linear::ols::CovType;
use crate::shared::gmm_c_statistic::{CStatisticMatrices, CStatisticSettings, gmm_c_statistic};
use crate::shared::gmm_weight::WeightType;
use crate::shared::goodness_of_fit::{is_perfect_fit, total_sum_of_squares};

pub use crate::shared::gmm_c_statistic::{CStatistic, CStatisticTarget};

/// 2SLSの`cov_type`に対応するGMMの重みの種別。2SLSには重みの選択が無いので、`cov_type`の系統から
/// 決める: `Classical`は`Classical`、`Hc0`〜`Hc3`は`Robust`（`S`にHC1・HC2・HC3の補正は無い）、
/// `Hac`と`Cluster`はそのまま。
#[must_use]
pub fn weight_type_for_cov_type(cov_type: &CovType) -> WeightType {
    match cov_type {
        CovType::Classical => WeightType::Classical,
        CovType::Hc0 | CovType::Hc1 | CovType::Hc2 | CovType::Hc3 => WeightType::Robust,
        CovType::Hac { lags, time_order } => WeightType::Hac {
            lags: *lags,
            time_order: time_order.clone(),
        },
        CovType::Cluster { groups } => WeightType::Cluster {
            groups: groups.clone(),
        },
    }
}

/// C統計量を計算する。
///
/// 定義は`shared::gmm_c_statistic`のモジュールdoc。`weight`は`S_e`の種類で、GMMでは`fit()`の
/// `gmm_weight_type`、2SLSでは[`weight_type_for_cov_type`]。`fit()`の`gmm_type`・`gmm_max_iter`・
/// `cov_type`には依存しない。
///
/// # Errors
///
/// - `CommonError::CStatisticNothingToTest`: `target`が空。
/// - `CommonError::CStatisticTooManyTestedInstruments`: 外した後の除外操作変数が内生変数の数より少ない。
/// - `CommonError::CStatisticUndefined`: 必要な行列が特異（操作変数の共線、完全フィット、モーメント
///   共分散のランク落ち。クラスター数が`Z_full`の列数に足りない場合を含む）。
/// - `weight`の設定の誤り（クラスター列が無い・行数が合わない・HACのラグ数が範囲外）。HACの
///   `time_order`の長さは`n`と一致している前提（`engine_pybind`が保証する。`gmm_iterate`と同じ）。
/// - `target`の添字が範囲外・重複（呼び出し側の誤りで、`CommonError::Internal`）。
///
/// いずれも`IvError::Common`に包んで返す。
pub fn compute_c_statistic(
    input: &IvInput,
    target: &CStatisticTarget,
    weight: &WeightType,
) -> Result<CStatistic, IvError> {
    let y = input.y();
    // 初期推定の残差が丸め誤差の範囲で0なら完全フィット（`GmmEstimator::fit`と同じ判定）。
    let sst = total_sum_of_squares(&mat_column_to_vec(y, 0), None, input.has_intercept());
    let n_params = input.k_exog() + input.k_endog();
    let is_perfect = |_: &faer::Mat<f64>, ssr0: f64| is_perfect_fit(ssr0, sst, n_params);
    Ok(gmm_c_statistic(
        &CStatisticMatrices {
            y,
            x_exog: input.x_exog(),
            x_endog: input.x_endog(),
            instruments: input.instruments(),
        },
        target,
        &CStatisticSettings {
            weight,
            s_scale: 1.0,
            is_perfect_fit: &is_perfect,
        },
    )?)
}

#[cfg(test)]
mod tests;
