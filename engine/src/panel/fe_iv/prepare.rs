//! FE-2SLS（`fit.rs`）と固定効果つきGMM（`gmm/fit.rs`）が共有する、推定の前段。
//!
//! 操作変数の数・singleton・定数列の検査、全列（`y`・`x_exog`・`x_endog`・`instruments`）の
//! within変換（1回だけ）、自由度の検証、変換後に分散が残らない列の検査を、この順で行う。
//! 点推定の方法（2SLSかGMMか）には依存しない。契約は`fe_iv.rs`のモジュールdocのパイプライン1〜3。

use std::ops::Range;

use faer::Mat;

use super::input::FeIvInput;
use super::wu_hausman::abs_max;
use crate::panel::common::{
    ColumnRole, PanelError, absorbed_parameters, ensure_no_singleton_groups,
    ensure_no_zero_variance_columns,
};
use crate::panel::fe::{FeAbsorption, FeEffects, WithinColumns, within_transform_columns};
use crate::shared::error::{CommonError, ErrorInfo};
use crate::shared::error_catalog::Stage;
use crate::shared::goodness_of_fit::is_perfect_fit;
use crate::shared::group_codes::GroupCodes;
use crate::shared::validation::{column_is_zero_variance_with_noise, has_no_variation};

/// 第一段階の失敗に`stage=FIRST_STAGE`を付ける。`stage`を付けられない`code`（`confidence_level`の
/// 範囲違反など）は、そのまま返す。
pub(super) fn in_first_stage(err: PanelError, endog_name: &str) -> PanelError {
    if err.report().code().spec().stage_allowed {
        err.in_stage(Stage::FirstStage {
            endog_name: endog_name.to_string(),
        })
    } else {
        err
    }
}

/// [`prepare_within`]の結果。
pub(super) struct WithinPrepared<'a> {
    /// 吸収する次元（`effects`の順）。
    pub(super) levels: Vec<&'a GroupCodes>,
    /// within変換した全列。順は`y`、`x_exog`、`x_endog`、`instruments`。
    pub(super) tilde: Vec<Vec<f64>>,
    /// 交互射影が各列から引いた効果（`absorbed[列][次元][水準]`）。閉形式は`None`。
    pub(super) absorbed: Option<Vec<Vec<Vec<f64>>>>,
    /// 交互射影が残す誤差の見積もり（全列で共有）。
    pub(super) noise: f64,
    /// 吸収した次元ごとの水準数。
    pub(super) n_groups: Vec<usize>,
    /// 吸収したパラメータの数（`absorbed_parameters`）。
    pub(super) neffects: usize,
    /// `k + neffects`（`k = k_exog + k_endog`）。
    pub(super) df_model: usize,
    pub(super) exog_range: Range<usize>,
    pub(super) endog_range: Range<usize>,
    pub(super) instrument_range: Range<usize>,
}

/// [`WithinPrepared`]を作る。
///
/// `confidence_level`は`Some`のとき、`absorption`の検証の直後に`(0, 1)`かを調べる（推論をしない
/// 推定量は`None`）。`first_stage`は、操作変数の自由度の不足（`n <= l + neffects`）を第一段階の
/// 失敗（`stage=FIRST_STAGE`つき）として返すか（FE-2SLS）、段階の無いエラーとして返すか（GMM）。
///
/// # Errors
/// `fe_iv.rs`の`FeIvEstimator::fit`のdocのうち、第一段階・構造式の推定より前の検査と同じ。
pub(super) fn prepare_within<'a>(
    input: &'a FeIvInput,
    effects: FeEffects,
    absorption: &FeAbsorption,
    confidence_level: Option<f64>,
    leverage_needed: bool,
    first_stage: bool,
) -> Result<WithinPrepared<'a>, PanelError> {
    let groups = input.index().effect_codes();
    // 吸収する次元。`time`の無い2-wayや次元が足りない`MultiWay`はここで弾く。
    let levels = effects.levels(groups)?;
    absorption.validate()?;
    if let Some(confidence_level) = confidence_level
        && !(confidence_level > 0.0 && confidence_level < 1.0)
    {
        return Err(PanelError::WithinRegressionFailed {
            source: CommonError::InvalidConfidenceLevel { confidence_level }.into(),
        });
    }

    let n = input.nobs();
    let (k_exog, k_endog, k_instruments) = (input.k_exog(), input.k_endog(), input.k_instruments());
    let k = input.k();
    let l = k_exog + k_instruments;

    // 識別の順序条件（`IvInput`の検査と同じ`code`）。
    if k_instruments < k_endog {
        return Err(PanelError::InsufficientInstruments {
            n_instruments: k_instruments,
            n_endog: k_endog,
        });
    }

    ensure_no_singleton_groups(input.index().effect_names(), &levels)?;

    // 生の値が全ての行で同じ列は、固定効果を除く前に分かる。固定効果を除いた後で分散がなくなる場合
    // （時間不変の列など）とは、エラーを分ける。重い変換より先に調べる。
    let roles = [
        (ColumnRole::XExog, input.x_exog_names(), input.x_exog()),
        (ColumnRole::XEndog, input.x_endog_names(), input.x_endog()),
        (
            ColumnRole::Instruments,
            input.instrument_names(),
            input.instruments(),
        ),
    ];
    for (role, names, columns) in roles {
        for (name, column) in names.iter().zip(columns) {
            if has_no_variation(column) {
                return Err(PanelError::RegressorConstant {
                    role,
                    column: name.clone(),
                });
            }
        }
    }
    if has_no_variation(input.y()) {
        return Err(CommonError::OutcomeConstant.into());
    }

    // 全ての列を1回だけ変換する。列の順は`y`、`x_exog`、`x_endog`、`instruments`。
    let mut columns: Vec<&[f64]> = Vec::with_capacity(1 + k_exog + k_endog + k_instruments);
    columns.push(input.y());
    columns.extend(input.x_exog().iter().map(Vec::as_slice));
    columns.extend(input.x_endog().iter().map(Vec::as_slice));
    columns.extend(input.instruments().iter().map(Vec::as_slice));
    let WithinColumns {
        columns: tilde,
        absorbed,
        noise,
    } = within_transform_columns(&levels, &columns, absorption, leverage_needed)?;
    let (exog_range, endog_range, instrument_range) = (
        1..1 + k_exog,
        1 + k_exog..1 + k,
        1 + k..1 + k + k_instruments,
    );

    // 自由度。構造式を先に、次に操作変数（第一段階）を検証する。
    let n_groups: Vec<usize> = levels.iter().map(|level| level.n_groups()).collect();
    let neffects = absorbed_parameters(&n_groups);
    let df_model = k + neffects;
    if n <= df_model {
        return Err(PanelError::InsufficientDegreesOfFreedom {
            n_obs: n,
            n_groups,
            k,
        });
    }
    // 内生変数が無いときは第一段階が無いので検証しない。
    if k_endog > 0 && n <= l + neffects {
        let err = PanelError::InsufficientDegreesOfFreedom {
            n_obs: n,
            n_groups,
            k: l,
        };
        return Err(if first_stage {
            let first_name = input.x_endog_names().first().map_or("", String::as_str);
            in_first_stage(err, first_name)
        } else {
            err
        });
    }

    // 固定効果を除いた後に分散が残らない列（役割ごとに報告）と被説明変数。
    let transformed_roles = [
        (
            ColumnRole::XExog,
            input.x_exog_names(),
            input.x_exog(),
            &tilde[exog_range.clone()],
        ),
        (
            ColumnRole::XEndog,
            input.x_endog_names(),
            input.x_endog(),
            &tilde[endog_range.clone()],
        ),
        (
            ColumnRole::Instruments,
            input.instrument_names(),
            input.instruments(),
            &tilde[instrument_range.clone()],
        ),
    ];
    for (role, names, raw, transformed) in transformed_roles {
        ensure_no_zero_variance_columns(role, names, raw, transformed, noise)?;
    }
    if column_is_zero_variance_with_noise(input.y(), &tilde[0], noise) {
        return Err(PanelError::OutcomeConstantWithinEntity);
    }

    Ok(WithinPrepared {
        levels,
        n_groups,
        tilde,
        absorbed,
        noise,
        neffects,
        df_model,
        exog_range,
        endog_range,
        instrument_range,
    })
}

/// 列`columns`（各列は長さ`n`）を並べた`(n, 列数)`の行列。
pub(super) fn mat_from_columns(n: usize, columns: &[&[f64]]) -> Mat<f64> {
    Mat::from_fn(n, columns.len(), |i, j| columns[j][i])
}

/// 構造残差の平方和`ssr`が丸め誤差の範囲で0（完全適合）か。`sst_within`は`ỹ'ỹ`、`noise`は
/// within変換の誤差の見積もり（[`WithinPrepared::noise`]）、`params`は構造式の係数（`x_exog`、`x_endog`の順）。
///
/// 交互射影が残す誤差は、構造残差`ỹ − X̃β̂`では`y`だけでなく`|β̂_j|`倍した各`x`の分にも乗る。
pub(super) fn structural_fit_is_perfect(
    input: &FeIvInput,
    params: &Mat<f64>,
    ssr: f64,
    sst_within: f64,
    noise: f64,
) -> bool {
    let residual_scale = abs_max(input.y())
        + input
            .x_exog()
            .iter()
            .chain(input.x_endog())
            .enumerate()
            .map(|(j, column)| params.get(j, 0).abs() * abs_max(column))
            .sum::<f64>();
    is_perfect_fit(ssr, sst_within, input.k())
        || ssr <= input.nobs() as f64 * (noise * residual_scale).powi(2)
}
