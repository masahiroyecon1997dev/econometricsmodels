//! 固定効果つきGMMのC統計量（difference-in-Hansen統計量）。
//!
//! 定義と計算は`iv::gmm`と共有する`shared::gmm_c_statistic`（`Z_full`・`Z_c`の2通りの推定、`S_e`の
//! 部分行列`S_c`を重みにした`J_c`、`C = J_e − J_c`の`χ²(|B|)`検定）で、within変換した行列に適用する。
//! 疑わしい変数を利用者が選ぶ事後診断なので`fit()`では計算せず、呼び出し時に`input`をwithin変換し直す
//! （推定量は変換後の行列を持たない）。`fit()`の`gmm_type`・`cov_type`には依存せず、`S_e`の種類は
//! `weight`で明示する（`OneStep`相当は`Classical`）。
//!
//! `Classical`重みのときだけ`S_e`を`n/(n − neffects)`倍する（`σ̂²`の分母を`n − neffects`にする。
//! `fe_hansen_j`と同じ理由と規則）。`J_e`・`J_c`が同じ`S_e`から作られるので、`C`は`Classical`のとき
//! 補正なしの`(n − neffects)/n`倍になる。`Robust`・`Cluster`の`S`は補正しない（`hansen_j.rs`参照）。
//!
//! 検査の順は`fit`と同じで、`prepare_within`（全列のwithin変換を含む）が先、`target`・`weight`の検査
//! （空の`target`・順序条件・添字・クラスター列・クラスター数）が後。`target`・`weight`だけで判定できる検査を
//! 変換より前に移さないのは、`fit`が`validate_weight_type`を`prepare_within`の後に呼ぶのと同じ順序を保ち、
//! 同時に成り立つエラー（`InsufficientInstruments`と`CStatisticNothingToTest`など）の優先順位を`fit`と
//! 揃えるため。

use faer::Mat;

use super::FeWeightType;
use crate::panel::common::PanelError;
use crate::panel::fe::{FeAbsorption, FeEffects};
use crate::panel::fe_iv::input::FeIvInput;
use crate::panel::fe_iv::prepare::{
    WithinPrepared, mat_from_columns, prepare_within, structural_fit_is_perfect,
};
use crate::shared::gmm_c_statistic::{
    CStatistic, CStatisticMatrices, CStatisticSettings, CStatisticTarget, gmm_c_statistic,
};

/// 固定効果つきGMMのC統計量。`target`の添字は`x_endog`・`instruments`の列の位置（昇順・重複なし）。
///
/// # Errors
/// - `prepare_within`の検査（`FeGmmEstimator::fit`と同じ。操作変数の数・singleton・定数列・自由度・
///   `absorption`など）。`confidence_level`は使わない。
/// - `CommonError::CStatisticNothingToTest`・`CStatisticTooManyTestedInstruments`・`CStatisticUndefined`
///   （`shared::gmm_c_statistic::gmm_c_statistic`と同じ。クラスター数が`Z_full`の列数に足りない場合を含む）
/// - `weight`の`Cluster`の列の長さが観測数と違うとき`CommonError::ClusterDimensionMismatch`等
/// - `target`の添字が範囲外・重複なら`CommonError::Internal`（呼び出し側の誤り）
pub fn fe_gmm_c_statistic(
    input: &FeIvInput,
    effects: FeEffects,
    absorption: &FeAbsorption,
    target: &CStatisticTarget,
    weight: &FeWeightType,
) -> Result<CStatistic, PanelError> {
    crate::shared::parallelism::ensure_serial();

    let n = input.nobs();
    let WithinPrepared {
        tilde,
        noise,
        neffects,
        exog_range,
        endog_range,
        instrument_range,
        ..
    } = prepare_within(input, effects, absorption, None, false, false)?;

    let column = |range: &std::ops::Range<usize>| -> Mat<f64> {
        let columns: Vec<&[f64]> = tilde[range.clone()].iter().map(Vec::as_slice).collect();
        mat_from_columns(n, &columns)
    };
    let y = mat_from_columns(n, &[tilde[0].as_slice()]);
    let (x_exog, x_endog, instruments) = (
        column(&exog_range),
        column(&endog_range),
        column(&instrument_range),
    );

    let sst_within: f64 = tilde[0].iter().map(|v| v * v).sum();
    // 変換後の列は行列に写したので、核が`Z_full`などを作る前に手放す。
    drop(tilde);
    let is_perfect = |beta0: &Mat<f64>, ssr0: f64| {
        structural_fit_is_perfect(input, beta0, ssr0, sst_within, noise)
    };
    // `prepare_within`が`n > k + neffects`を保証するので、分母は正。
    let s_scale = match weight {
        FeWeightType::Classical => n as f64 / (n - neffects) as f64,
        FeWeightType::Robust | FeWeightType::Cluster { .. } => 1.0,
    };
    let c_statistic = gmm_c_statistic(
        &CStatisticMatrices {
            y: &y,
            x_exog: &x_exog,
            x_endog: &x_endog,
            instruments: &instruments,
        },
        target,
        &CStatisticSettings {
            weight: &weight.to_weight_type(),
            s_scale,
            is_perfect_fit: &is_perfect,
        },
    )?;
    Ok(c_statistic)
}
