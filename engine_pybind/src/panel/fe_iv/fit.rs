//! 入力を組み立てて`engine::panel::fe_iv`の推定量（`FeTwoSlsEstimator`・`FeGmmEstimator`）を推定し、
//! `FEIVResult`として返す。`iv/common/fit.rs`に対応する。
//!
//! `fit`自体は`#[cfg(test)]`から呼べない（`SelectedData`がGILを要求する）。検証は`maturin develop`後の
//! Pythonからの数値照合で行う。純Rust部分は`fit_frame`（`&DataFrame`だけを取り、`py.detach`の中で走る）。

use engine::panel::fe::FeAbsorption;
use engine::panel::fe_iv::gmm::{CStatistic, CStatisticTarget, FeGmmEstimator, fe_gmm_c_statistic};
use engine::panel::fe_iv::two_sls::FeTwoSlsEstimator;
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use super::super::common::panel_error_to_pyerr;
use super::super::fe::{FEResult, LazyFixedEffects, fe_result_fields};
use super::gmm_setup::c_statistic_weight;
use super::input::{BuiltFeIvInput, BuiltGmm, FeIvColumns, build_fe_iv_input};
use super::options::FEIVOptions;
use super::result::FEIVResult;
use crate::shared::column_extraction::SelectedData;
use crate::shared::errors::internal_error;

/// 交互射影の設定（`tol`・`max_iter`）。
fn absorption(options: &FEIVOptions) -> FeAbsorption {
    FeAbsorption {
        tol: options.tol,
        max_iter: options.max_iter,
    }
}

/// 推定した推定量。GMMは、結果に載せる`gmm_type`と重みの正規形の名前を一緒に持つ（別に持ち回さない）。
enum Estimated {
    /// `first_stage`は内生変数ごとの第一段階の結果（`x_endog`の順。固定効果は計算済み）。第一段階の
    /// 推定量（設計行列を持つ）は結果に変えてすぐ手放す。結果が要らない再推定では空。
    TwoSls {
        estimator: FeTwoSlsEstimator,
        first_stage: Vec<(String, FEResult)>,
    },
    Gmm {
        estimator: FeGmmEstimator,
        gmm_type_name: String,
        weight_type_name: Option<String>,
    },
}

/// `first_stage_effects`が`Some`（効果の列名）のとき、2SLSの第一段階を`FEResult`にして返す。
fn estimate_built(
    built: BuiltFeIvInput,
    options: &FEIVOptions,
    first_stage_effects: Option<&[String]>,
) -> PyResult<Estimated> {
    let BuiltFeIvInput {
        input,
        effects,
        cov_type,
        cov_type_name,
        gmm,
        ..
    } = built;
    let absorption = absorption(options);
    Ok(match gmm {
        None => {
            let mut first_stage: Vec<(String, FEResult)> = Vec::new();
            let (estimator, _) = FeTwoSlsEstimator::fit_with_first_stages(
                input,
                effects,
                cov_type,
                options.confidence_level,
                &absorption,
                |name, stage| {
                    if let Some(effect_columns) = first_stage_effects {
                        // 第一段階は固定効果をその場で計算して持つ（`FeEstimator`を結果に持たせない）。
                        let fixed = LazyFixedEffects::Computed(stage.fixed_effects());
                        first_stage.push((
                            name.to_string(),
                            fe_result_fields(
                                &stage,
                                effect_columns.to_vec(),
                                cov_type_name.clone(),
                                fixed,
                            ),
                        ));
                    }
                },
            )
            .map_err(panel_error_to_pyerr)?;
            Estimated::TwoSls {
                estimator,
                first_stage,
            }
        }
        Some(BuiltGmm {
            gmm_type,
            gmm_type_name,
            weight_type_name,
            raise_on_non_convergence,
        }) => Estimated::Gmm {
            estimator: FeGmmEstimator::fit_with_absorption(
                input,
                effects,
                gmm_type,
                raise_on_non_convergence,
                cov_type,
                options.confidence_level,
                &absorption,
            )
            .map_err(panel_error_to_pyerr)?,
            gmm_type_name,
            weight_type_name,
        },
    })
}

/// 入力を組み立てて`FeTwoSlsEstimator`を推定する（第一段階・診断・`fixed_effects()`の再計算が使う）。
/// `options.estimator`が`"2sls"`でなければ到達不能（呼び出し側が結果の`estimator`を先に検査する）。
pub(super) fn estimate_two_sls(
    df: &DataFrame,
    columns: FeIvColumns,
    options: &FEIVOptions,
) -> PyResult<FeTwoSlsEstimator> {
    match estimate_built(build_fe_iv_input(df, columns, options)?, options, None)? {
        Estimated::TwoSls { estimator, .. } => Ok(estimator),
        Estimated::Gmm { .. } => Err(internal_error(
            "a two-stage least squares re-estimation returned a GMM estimator",
        )),
    }
}

/// 固定効果つきGMMのC統計量を、`fit()`と同じ入力・重みで計算する（`options.estimator`が`"gmm"`のときだけ）。
pub(super) fn compute_c_statistic(
    df: &DataFrame,
    columns: FeIvColumns,
    options: &FEIVOptions,
    target: &CStatisticTarget,
) -> PyResult<CStatistic> {
    let built = build_fe_iv_input(df, columns, options)?;
    let Some(gmm) = &built.gmm else {
        return Err(internal_error(
            "the C statistic was requested for a model fitted by two-stage least squares",
        ));
    };
    let weight = c_statistic_weight(&gmm.gmm_type);
    fe_gmm_c_statistic(
        &built.input,
        built.effects,
        &absorption(options),
        target,
        &weight,
    )
    .map_err(panel_error_to_pyerr)
}

/// Pythonから渡された`data`・列名・`options`を検証し、推定量を推定して`FEIVResult`として返す。
///
/// # Errors
/// - `build_fe_iv_input`が返すエラー（`ValidationError`）
/// - `FeTwoSlsEstimator::fit_with_absorption`・`FeGmmEstimator::fit_with_absorption`が返す`PanelError`
///   （識別の順序条件・singleton・自由度不足・分散ゼロ・第一段階の失敗・完全適合・重み行列・共分散の検証・
///   非収束等）は`panel_error_to_pyerr`で変換
pub(crate) fn fit(
    py: Python<'_>,
    data: SelectedData,
    columns: FeIvColumns,
    options: &FEIVOptions,
) -> PyResult<FEIVResult> {
    // 取り込み済みの`frame`以降はPythonオブジェクトに触れないので、GILを手放して走らせる。`source`は結果に持たせる。
    let SelectedData { source, frame } = data;
    let mut result = py.detach(|| fit_frame(&frame, columns, options))?;
    // 診断・`fixed_effects()`・`c_stat()`が入力を作り直すための元のデータとオプション（GILを持って付ける）。
    result.retain_source(source, options.clone());
    Ok(result)
}

/// [`fit`]の純Rust部分（GILを手放した状態で走る）。
fn fit_frame(df: &DataFrame, columns: FeIvColumns, options: &FEIVOptions) -> PyResult<FEIVResult> {
    let built = build_fe_iv_input(df, columns.clone(), options)?;
    let cov_type_name = built.cov_type_name.clone();
    let effect_columns = columns.effects.clone();
    match estimate_built(built, options, Some(&effect_columns))? {
        Estimated::TwoSls {
            estimator,
            first_stage,
        } => FEIVResult::from_two_sls(&estimator, first_stage, columns, cov_type_name),
        Estimated::Gmm {
            estimator,
            gmm_type_name,
            weight_type_name,
        } => FEIVResult::from_gmm(
            &estimator,
            columns,
            cov_type_name,
            gmm_type_name,
            weight_type_name,
        ),
    }
}
