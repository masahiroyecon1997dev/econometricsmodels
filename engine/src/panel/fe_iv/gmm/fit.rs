//! `FeGmmEstimator::fit`の本体。パイプラインの契約はモジュールdoc（`gmm.rs`）が正本。
//!
//! テストは`fit/tests.rs`と`fit/tests/`にテーマ別で置く。

use super::FeGmmEstimator;
use crate::panel::common::PanelError;
use crate::panel::fe::{FeAbsorption, FeEffects};
use crate::panel::fe_iv::input::FeIvInput;
use crate::panel::fe_iv::prepare::{
    WithinPrepared, mat_from_columns, prepare_within, structural_fit_is_perfect,
};
use crate::shared::error::CommonError;
use crate::shared::gmm_iteration::{
    GmmInitialStep, GmmIteration, GmmType, gmm_initial_step, gmm_iterate,
};
use crate::shared::gmm_weight::{WeightType, validate_weight_type};

/// [`FeGmmEstimator::fit_with_absorption`]の本体。契約（エラー条件を含む）は`fit`のdocコメント参照。
pub(super) fn fit_fe_gmm(
    input: FeIvInput,
    effects: FeEffects,
    gmm_type: GmmType,
    raise_on_non_convergence: bool,
    absorption: &FeAbsorption,
) -> Result<FeGmmEstimator, PanelError> {
    // faerのグローバル並列度を`Par::Seq`に固定する（`FeEstimator::fit`と同じ理由）。
    crate::shared::parallelism::ensure_serial();

    gmm_type.validate()?;
    let weight_type = gmm_type.weight_type();
    // 固定効果つきGMMは、行単位のNewey-Westの重みを持たない（パネルには時点の順序を行単位で渡せず、
    // Driscoll-Kraay型の重みはミクロパネルで実用にならない）。バインディングは作らない。
    if matches!(weight_type, Some(WeightType::Hac { .. })) {
        return Err(CommonError::internal(
            "fixed-effects GMM does not support the HAC weight type",
        )
        .into());
    }

    let n = input.nobs();
    let (k_exog, k_instruments) = (input.k_exog(), input.k_instruments());
    let k = input.k();
    let l = k_exog + k_instruments;

    // `HC2`・`HC3`のレバレッジは要らない（点推定は共分散を作らない）。
    let WithinPrepared {
        levels: _,
        tilde,
        absorbed: _,
        noise,
        n_groups,
        neffects: _,
        df_model,
        exog_range,
        endog_range,
        instrument_range,
    } = prepare_within(&input, effects, absorption, None, false, false)?;

    // `weight_type=Cluster`のグループ（検証して返す）。丁度識別の閾値の違いは`validate_weight_type`が持つ。
    let weight_cluster_codes = match weight_type {
        Some(weight_type) => validate_weight_type(weight_type, n, k, l)?,
        None => None,
    };

    // Z̃ = [x̃_exog, z̃]（全操作変数）、X̃ = [x̃_exog, x̃_endog]（実際の内生変数）、ỹ。
    let column = |range: &std::ops::Range<usize>| -> Vec<&[f64]> {
        tilde[range.clone()].iter().map(Vec::as_slice).collect()
    };
    let (exog, endog, instruments) = (
        column(&exog_range),
        column(&endog_range),
        column(&instrument_range),
    );
    let z = mat_from_columns(n, &[exog.as_slice(), instruments.as_slice()].concat());
    let x = mat_from_columns(n, &[exog.as_slice(), endog.as_slice()].concat());
    let y = mat_from_columns(n, &[tilde[0].as_slice()]);

    let mut param_names = input.x_exog_names().to_vec();
    param_names.extend(input.x_endog_names().iter().cloned());

    // 初期推定: W₀ = (Z̃'Z̃)⁻¹（2SLSと同じ重み）。
    let GmmInitialStep {
        ztz,
        beta: beta0,
        residuals: residuals0,
        classical_s,
    } = gmm_initial_step(&z, &x, &y)?;

    // 初期推定の残差が丸め誤差の範囲で0なら完全適合。重み行列`S`が0になって後段の逆行列が
    // 特異になる前に、原因が分かるエラーで止める。
    let sst_within: f64 = tilde[0].iter().map(|v| v * v).sum();
    let ssr0: f64 = (0..n).map(|i| (*residuals0.get(i, 0)).powi(2)).sum();
    if structural_fit_is_perfect(&input, &beta0, ssr0, sst_within, noise) {
        return Err(PanelError::SecondStageFailed {
            source: CommonError::PerfectFit.into(),
        });
    }

    // 方式（`gmm_type`）に応じた回数の再推定。
    let GmmIteration {
        beta,
        residuals,
        s_used,
        n_iter,
        converged,
        hac_lags_used: _,
    } = gmm_iterate(
        &gmm_type,
        weight_type,
        weight_cluster_codes,
        &z,
        &x,
        &y,
        &ztz,
        beta0,
        residuals0,
        classical_s,
    )?;

    if !converged && raise_on_non_convergence {
        return Err(CommonError::GmmNonConvergence {
            n_iter: n_iter as usize,
        }
        .into());
    }

    Ok(FeGmmEstimator {
        input,
        effects,
        gmm_type,
        param_names,
        params: beta,
        residuals,
        s_used,
        n_iter,
        converged,
        df_model,
        df_resid: n - df_model,
        n_groups,
    })
}

#[cfg(test)]
mod tests;
