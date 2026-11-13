//! FEIVのWu-Hausman内生性検定（回帰ベース）。

use crate::linear::common::LeastSquaresError;
use crate::panel::common::PanelError;
use crate::panel::fe::{FeCovType, FeEffects, FeEstimator, FeInput, WithinOutput};
use crate::shared::error::{CommonError, ErrorInfo};
use crate::shared::error_catalog::Stage;
use crate::shared::instrumental_variables::wu_hausman_columns;
use crate::shared::wald::wald_f_test;

use super::input::FeIvInput;

/// [`fe_wu_hausman_test`]の戻り値。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct FeWuHausmanTest {
    pub(super) statistic: f64,
    pub(super) p_value: f64,
    /// F分布の自由度`(分子, 分母)`。
    pub(super) df: (usize, usize),
}

/// [`fe_wu_hausman_test`]に渡す、`fit_fe_iv`が持つ材料。
pub(super) struct WuHausmanContext<'a> {
    pub(super) input: &'a FeIvInput,
    pub(super) effects: FeEffects,
    pub(super) cov_type: &'a FeCovType,
    pub(super) confidence_level: f64,
    /// 効果の各次元の索引と列名。
    pub(super) groups: &'a [crate::shared::group_codes::GroupCodes],
    pub(super) effect_names: &'a [String],
    /// within変換した列（順は`y`、`x_exog`、`x_endog`、`instruments`）。
    pub(super) tilde: &'a [Vec<f64>],
    /// 内生変数ごとの第一段階（`x_endog`の順）。
    pub(super) first_stages: &'a [FeEstimator],
    /// 変換が残す誤差（完全適合の判定に使う）。
    pub(super) noise: f64,
}

/// 補助回帰の失敗のうち、この検定だけの問題として`None`にするもの（構造式・第一段階の推定とは無関係）。
/// 補助回帰は構造式より内生変数の数だけ列が多く、`v̂`は第一段階の残差なので、自由度が足りない・
/// `v̂`の分散が残らない・完全適合・設計行列の特異性は起こりうる。
fn is_expected_failure(err: &PanelError) -> bool {
    match err {
        PanelError::InsufficientDegreesOfFreedom { .. }
        | PanelError::ZeroVarianceAfterDemeaning { .. } => true,
        PanelError::WithinRegressionFailed { source } => matches!(
            source,
            LeastSquaresError::SingularMatrix { .. }
                | LeastSquaresError::Common(
                    CommonError::PerfectFit
                        | CommonError::InsufficientObservations { .. }
                        | CommonError::NearSingularMatrix { .. }
                )
        ),
        _ => false,
    }
}

pub(super) fn abs_max(column: &[f64]) -> f64 {
    column.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// Wu-Hausman内生性検定。構造式の`ỹ`を`[x̃_exog, x̃_endog, v̂]`に回帰し（`v̂`は第一段階の残差。
/// within変換済みの列の線形結合なので、そのまま補助回帰の列にできる）、`v̂`の係数がすべてゼロという
/// 仮説を、`fit()`に渡された`cov_type`の共分散でのWald検定（`F(k_endog, 補助回帰の df_inference)`）で調べる。
/// 補助回帰の吸収する固定効果は構造式と同じで、`df_resid = n − k − k_endog − neffects`。
///
/// 内生変数が無ければ`None`。補助回帰が自由度不足（構造式・第一段階は通っても、補助回帰は列が`k_endog`
/// だけ多い）・完全適合・特異・Waldの`k_endog × k_endog`部分行列が数値的に特異で失敗したときも、構造式の
/// 推定結果とは無関係な診断だけの問題なので、`fit`全体は失敗させず`None`にする（IV本体と同じ）。
/// `ZeroVarianceAfterDemeaning`（`v̂`の分散ゼロ）は、第一段階の完全適合が先に弾くので実際には到達しない
/// 防御的な分類。
///
/// **IV本体との違い**: 補助回帰はF検定をしない（`tests_overall_f = false`）。IV本体は補助回帰の傾きの数
/// `k + k_endog`に対して`G ≤ k + k_endog`で`None`にするが、ここで使うのは`v̂`の`k_endog × k_endog`部分行列
/// だけで、そのランクは高々`G − 1`（DKは`T − 1`）。構造式の検査（`G > k ≥ k_endog`）から`G − 1 ≥ k_endog`が
/// 保証されているので、`G`が`k + 1`〜`k + k_endog`でも検定は値を持つ（数値的に特異なら`None`）。
/// それ以外の失敗は、`stage=WU_HAUSMAN_REGRESSION`を付けて返す。
///
/// # Errors
/// 補助回帰の、`None`にしない失敗（`is_expected_failure`以外）。
pub(super) fn fe_wu_hausman_test(
    context: &WuHausmanContext<'_>,
) -> Result<Option<FeWuHausmanTest>, PanelError> {
    let WuHausmanContext {
        input,
        effects,
        cov_type,
        confidence_level,
        groups,
        effect_names,
        tilde,
        first_stages,
        noise,
    } = *context;
    let (k_exog, k_endog) = (input.k_exog(), input.k_endog());
    if k_endog == 0 {
        return Ok(None);
    }
    let n = input.nobs();

    let residuals: Vec<Vec<f64>> = first_stages
        .iter()
        .map(|stage| (0..n).map(|i| *stage.residuals().get(i, 0)).collect())
        .collect();
    let (raw_columns, names) = wu_hausman_columns(
        input.x_exog(),
        input.x_exog_names(),
        input.x_endog().to_vec(),
        input.x_endog_names(),
        residuals.clone(),
    );
    let mut within_x: Vec<Vec<f64>> = tilde[1..1 + k_exog + k_endog].to_vec();
    within_x.extend(residuals);

    let outcome = (|| -> Result<Option<FeWuHausmanTest>, PanelError> {
        let aux_input = FeInput::from_effect_groups(
            input.y(),
            &raw_columns,
            names,
            groups.to_vec(),
            input.dep_var_name().to_string(),
        )?
        .with_effect_names(effect_names.to_vec());
        let aux = FeEstimator::fit_from_within(
            aux_input,
            effects,
            cov_type.clone(),
            confidence_level,
            WithinOutput {
                y: tilde[0].clone(),
                x: within_x,
                absorbed: None,
                noise,
            },
            false,
        )?;

        // 交互射影が残す誤差の範囲の残差は、`σ̂² = 0`で統計量が定義できない（完全適合）。誤差は`y`だけでなく
        // 係数倍した各列にも乗る（構造式の完全適合の判定と同じ）。補助回帰の列は`[x_exog, x_endog, v̂]`の順で、
        // `v̂_j = x̃_endog[j] − [x̃_exog, z̃]π_j`の誤差は`x_endog[j]`と`π_j`倍した`x_exog`・操作変数の和。
        let ssr: f64 = (0..n).map(|i| aux.residuals().get(i, 0).powi(2)).sum();
        let first_stage_columns: Vec<f64> = input
            .x_exog()
            .iter()
            .chain(input.instruments())
            .map(|column| abs_max(column))
            .collect();
        let mut column_scales: Vec<f64> = input
            .x_exog()
            .iter()
            .chain(input.x_endog())
            .map(|column| abs_max(column))
            .collect();
        for (j, stage) in first_stages.iter().enumerate() {
            column_scales.push(
                abs_max(&input.x_endog()[j])
                    + first_stage_columns
                        .iter()
                        .enumerate()
                        .map(|(l, s)| stage.params().get(l, 0).abs() * s)
                        .sum::<f64>(),
            );
        }
        let scale = abs_max(input.y())
            + column_scales
                .iter()
                .enumerate()
                .map(|(j, s)| aux.params().get(j, 0).abs() * s)
                .sum::<f64>();
        if ssr <= n as f64 * (noise * scale).powi(2) {
            return Ok(None);
        }

        let df_inference = aux.df_inference();
        match wald_f_test(
            aux.params(),
            aux.cov_params(),
            k_exog + k_endog,
            k_endog,
            df_inference,
        ) {
            Ok((statistic, p_value)) => Ok(Some(FeWuHausmanTest {
                statistic,
                p_value,
                df: (k_endog, df_inference),
            })),
            Err(CommonError::NearSingularMatrix { .. }) => Ok(None),
            Err(err) => Err(err.into()),
        }
    })();

    match outcome {
        Err(err) if is_expected_failure(&err) => Ok(None),
        // 構造式の推定が通っているので、ここに来るのは到達不能な内部エラー等だけ。`stage`を付けられる`code`には
        // `WU_HAUSMAN_REGRESSION`を付ける。
        Err(err) if err.report().code().spec().stage_allowed => {
            Err(err.in_stage(Stage::WuHausmanRegression))
        }
        other => other,
    }
}
