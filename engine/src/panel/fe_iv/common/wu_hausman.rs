//! FEIVのWu-Hausman内生性検定（回帰ベース。FE-2SLSと固定効果つきGMMが共有）。
//!
//! 補助回帰は第一段階の残差を足した最小二乗で、第二段階の推定量（2SLSかGMMか）にも、GMMの重みにも依らない。
//! `cov_type`は`fit`に渡したものを使う。

use crate::linear::common::LeastSquaresError;
use crate::panel::common::PanelError;
use crate::panel::fe::{FeCovType, FeEffects, FeEstimator, FeInput, WithinOutput};
use crate::shared::error::{CommonError, ErrorInfo};
use crate::shared::error_catalog::Stage;
use crate::shared::instrumental_variables::wu_hausman_columns;
use crate::shared::orthonormal::WaldCoordinates;
use crate::shared::wald::wald_f_test;

use super::first_stage::FirstStageSummary;
use super::input::FeIvInput;
use super::prepare::abs_max;

/// [`fe_wu_hausman_test`]の戻り値。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::panel::fe_iv) struct FeWuHausmanTest {
    pub(in crate::panel::fe_iv) statistic: f64,
    pub(in crate::panel::fe_iv) p_value: f64,
    /// F分布の自由度`(分子, 分母)`。
    pub(in crate::panel::fe_iv) df: (usize, usize),
}

/// [`fe_wu_hausman_test`]に渡す、`fit`が持つ材料。
pub(in crate::panel::fe_iv) struct WuHausmanContext<'a> {
    pub(in crate::panel::fe_iv) input: &'a FeIvInput,
    pub(in crate::panel::fe_iv) effects: FeEffects,
    pub(in crate::panel::fe_iv) cov_type: &'a FeCovType,
    pub(in crate::panel::fe_iv) confidence_level: f64,
    /// 効果の各次元の索引と列名。
    pub(in crate::panel::fe_iv) groups: &'a [crate::shared::group_codes::GroupCodes],
    pub(in crate::panel::fe_iv) effect_names: &'a [String],
    /// within変換した列（順は`y`、`x_exog`、`x_endog`、`instruments`）。
    pub(in crate::panel::fe_iv) tilde: &'a [Vec<f64>],
    /// 内生変数ごとの第一段階から取り出した係数・残差（`x_endog`の順）。残差は補助回帰の列へ
    /// 複製せず移すので、`fe_wu_hausman_test`の後は空になる（係数は残る）。
    pub(in crate::panel::fe_iv) first_stages: &'a mut [FirstStageSummary],
    /// 変換が残す誤差（完全適合の判定に使う）。
    pub(in crate::panel::fe_iv) noise: f64,
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

/// 補助回帰の直交座標（`b̃`・`M̃`）で、末尾`k_endog`列（`v̂`の係数）がゼロという仮説をWald検定する。
/// `v̂`の部分行列が数値的に特異なときは、診断だけの問題として`None`（`fit`全体は失敗させない）。
/// 切り出してあるのは、この契約を手組みの特異な`M̃`で決定的にテストするため。
fn wu_hausman_wald(
    wald: &WaldCoordinates,
    k_exog: usize,
    k_endog: usize,
    df_inference: usize,
) -> Result<Option<FeWuHausmanTest>, PanelError> {
    match wald_f_test(
        &wald.coordinates,
        &wald.meat,
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
pub(in crate::panel::fe_iv) fn fe_wu_hausman_test(
    context: WuHausmanContext<'_>,
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
    } = context;
    let (k_exog, k_endog) = (input.k_exog(), input.k_endog());
    if k_endog == 0 {
        return Ok(None);
    }
    let n = input.nobs();

    // 第一段階の残差は、補助回帰のwithin空間の列へ移す（要約が持ち続けると同じ列が3つ生きる）。
    // 生の列には、`wu_hausman_columns`が所有を要るので1つだけ複製する。
    let residuals: Vec<Vec<f64>> = first_stages
        .iter_mut()
        .map(|stage| std::mem::take(&mut stage.residuals))
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
        let aux_input = FeInput::from_effect_groups_owned(
            input.y().to_vec(),
            raw_columns,
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
                        .map(|(l, s)| stage.params[l].abs() * s)
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
        // 補助回帰が直交座標で持つ`b̃`と`M̃`で、末尾`k_endog`列（残差の列）を検定する
        // （`shared::orthonormal`）。補助回帰は`k_exog + 2 k_endog >= 1`列なので必ずある。
        let wald = aux.wald_coordinates().ok_or_else(|| {
            PanelError::from(CommonError::internal(
                "the auxiliary regression of the Wu-Hausman test has no coordinates".to_string(),
            ))
        })?;
        wu_hausman_wald(wald, k_exog, k_endog, df_inference)
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

#[cfg(test)]
mod tests {
    use faer::Mat;

    use super::*;

    /// 外生変数1個・内生変数1個の補助回帰（係数は`x̃_exog`・`x̃_endog`・`v̂`の3個）の直交座標。
    fn coordinates(meat: Mat<f64>) -> WaldCoordinates {
        WaldCoordinates {
            coordinates: Mat::from_fn(3, 1, |i, _| 1.0 + i as f64),
            meat,
        }
    }

    #[test]
    fn a_singular_covariance_gives_none_instead_of_failing() {
        let wald = coordinates(Mat::<f64>::zeros(3, 3));
        assert_eq!(wu_hausman_wald(&wald, 1, 1, 10).unwrap(), None);
    }

    #[test]
    fn an_identity_covariance_gives_the_statistic_of_the_last_coefficient() {
        // 末尾1個の係数（`b̃ = 3`、`M̃ = 1`）の`W = 9`、`F = 9`、自由度`(1, 10)`。
        let wald = coordinates(Mat::<f64>::identity(3, 3));
        let test = wu_hausman_wald(&wald, 1, 1, 10).unwrap().unwrap();
        assert!((test.statistic - 9.0).abs() < 1e-12);
        assert_eq!(test.df, (1, 10));
    }
}
