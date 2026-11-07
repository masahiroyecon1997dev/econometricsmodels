//! `FeEstimator::fit`の本体。singleton検出・within変換・自由度検証・最小二乗・`cov_type`別の共分散・
//! 統計量の再計算をつなぐオーケストレーション。
//!
//! テストは`fit/tests.rs`と`fit/tests/`にテーマ別で置く（`basic`・`reference`・`cov_type`・`validation`・`proptests`等）。

use statrs::distribution::StudentsT;

use super::FeEstimator;
use super::input::{FeAbsorption, FeCovType, FeEffects, FeInput};
use super::r_squared::{fe_r_squared_between, fe_r_squared_overall};
use super::within::{
    validate_no_singleton_groups_one_way, validate_no_singleton_groups_two_way,
    validate_no_zero_variance_regressors, within_transform_one_way, within_transform_two_way_with,
};
use crate::linear::common::LeastSquaresError;
use crate::linear::ols::OlsInput;
use crate::panel::common::{PanelCov, PanelCovInput, PanelError, PanelStructure, panel_cov_params};
use crate::shared::error::CommonError;
use crate::shared::goodness_of_fit::{gaussian_goodness_of_fit, is_perfect_fit};
use crate::shared::group_codes::GroupCodes;
use crate::shared::inference::{InferenceTable, inference_table};
use crate::shared::least_squares::{LeastSquaresFit, least_squares};
use crate::shared::validation::column_is_zero_variance_with_noise;
use crate::shared::wald::wald_f_test;

/// 2-wayの時点コードを返す（1-wayは`None`）。2-wayなのに`input`に`time`が無ければ
/// `PanelError::TwoWayRequiresTime`。`FeEstimator::fit`が最初に呼び、以降の
/// cov_type別の計算は`effects`ではなくこの`Option`で1-way/2-wayを区別する。
fn two_way_time(effects: FeEffects, input: &FeInput) -> Result<Option<&GroupCodes>, PanelError> {
    match effects {
        FeEffects::OneWay => Ok(None),
        FeEffects::TwoWay => input
            .time_codes()
            .map(Some)
            .ok_or(PanelError::TwoWayRequiresTime),
    }
}

/// [`FeEstimator::fit`]の本体。契約（エラー条件を含む）は`fit`のdocコメント参照。
pub(super) fn fit_fe(
    input: FeInput,
    effects: FeEffects,
    cov_type: FeCovType,
    confidence_level: f64,
    absorption: &FeAbsorption,
) -> Result<FeEstimator, PanelError> {
    // faerのグローバル並列度をPar::Seqに固定する（`crate::shared::parallelism`。
    // `cargo test -p engine`でFeEstimator::fitを直接叩く経路でも担保するためここで呼ぶ、`engine/src/panel/CLAUDE.md`「faerの
    // グローバル並列度」参照）。
    crate::shared::parallelism::ensure_serial();

    // 2-wayの時点コード（1-wayは`None`）。`time`が無いのに2-wayを指定した入力はここで
    // 弾き、以降は`Some`かどうかだけで1-way/2-wayを分岐する（`effects`と`input.time`の
    // 食い違いを後段で`expect`せずに済む）。
    let two_way_time = two_way_time(effects, &input)?;

    absorption.validate()?;

    let mut absorbed = None;
    let (y, x, noise) = match effects {
        FeEffects::OneWay => {
            validate_no_singleton_groups_one_way(&input)?;
            let (y, x) = within_transform_one_way(&input);
            (y, x, 0.0)
        }
        FeEffects::TwoWay => {
            validate_no_singleton_groups_two_way(&input)?;
            // 組合せが欠ける2-wayのHC2/HC3は、水準の少ない方の効果の密な行列を解く。
            let leverage_needed = matches!(cov_type, FeCovType::Hc2 | FeCovType::Hc3);
            let within = within_transform_two_way_with(&input, absorption, leverage_needed)?;
            absorbed = within.absorbed;
            (within.y, within.x, within.noise)
        }
    };

    let n = input.nobs();
    let n_entities = input.entity_codes().n_groups();
    let n_periods = two_way_time.map(GroupCodes::n_groups);
    let k = input.x_names().len();
    // `neffects`: entityダミー・timeダミーの実効パラメータ数（`fe-spec.md`3.2節）。2-wayは両者の
    // 間に定数項ぶんの重複が1つ生じるため`+1`補正（`n_entities + n_periods - 1`）。
    let neffects = match n_periods {
        None => n_entities,
        Some(n_periods) => n_entities + n_periods - 1,
    };
    let df_model = k + neffects;
    if n <= df_model {
        return Err(PanelError::InsufficientDegreesOfFreedom {
            n_obs: n,
            n_entities,
            n_periods,
            k,
        });
    }
    let df_resid = n - df_model;

    validate_no_zero_variance_regressors(&input, &x, noise)?;

    // 被説明変数もwithin変換後に分散が残らなければ、固定効果が説明するものが無い
    // （個体内で定数、2-wayなら個体効果と時点効果だけで説明される）。説明変数と同じ
    // 相対閾値で判定する。
    if column_is_zero_variance_with_noise(input.y(), &y, noise) {
        return Err(PanelError::OutcomeConstantWithinEntity);
    }

    // `OlsInput::from_columns`が返しうる`LeastSquaresError::Common(DimensionMismatch)`は
    // ここでは理論上到達不能: `y`/`x`はどちらも`within_transform_*`が`input.y()`/
    // `input.x()`（`FeInput::from_group_codes`が既に同じ長さであることを検証済み）から
    // 1対1で生成した同じ長さの列であり、この関数内で長さがずれる操作をしていない。
    // それでも`Result`を返す契約（`from_columns`のシグネチャ）をそのまま守り、
    // `unwrap`はしない（`.claude/rules/rust-style.md`「テスト」の「理論上到達不能でも
    // `Result`化する」方針に揃える）。
    let within_input = OlsInput::from_columns(
        &y,
        &x,
        input.x_names().to_vec(),
        false,
        input.dep_var_name().to_string(),
    )
    .map_err(|source| PanelError::WithinRegressionFailed { source })?;

    if !(confidence_level > 0.0 && confidence_level < 1.0) {
        return Err(PanelError::WithinRegressionFailed {
            source: CommonError::InvalidConfidenceLevel { confidence_level }.into(),
        });
    }

    // `β̂`・残差・`(X̃'X̃)⁻¹`は最小二乗の部品を直接呼んで求める（`cov_type`ごとの標準誤差は
    // 下記でFE自身が計算し直すため、`OlsEstimator`の推論は使わない。`fe.rs`のモジュールdoc
    // 「`cov_type`対応」参照）。`x=[]`（固定効果のみのモデル）だとwithin変換後の設計行列`x`も
    // 0列になり`k=0`になるが、`least_squares`は0列も受理する。
    let LeastSquaresFit {
        params,
        residuals: residual_mat,
        xtx_inv,
    } = least_squares(within_input.x(), within_input.y()).map_err(|err| {
        PanelError::WithinRegressionFailed {
            source: LeastSquaresError::from_rank_deficient(
                err,
                within_input.x(),
                within_input.param_names(),
            ),
        }
    })?;

    // `cov_type`別の共分散行列の計算に使う共通の材料（within変換後の設計行列、残差・SSR）。
    let x_mat = within_input.x();
    let residuals: Vec<f64> = (0..n).map(|i| *residual_mat.get(i, 0)).collect();
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();

    // within回帰の残差が丸め誤差の範囲で0なら完全フィット。標準誤差とF検定が定義できない。
    // within変換後の`y`の平均は0なので、総平方和は非中心化（`Σỹ²`）。
    let sst_within: f64 = y.iter().map(|v| v * v).sum();
    // 交互射影が`y`に残す誤差（元の`y`の絶対値の最大の`noise`倍）の2乗和以下の残差も、完全適合とみなす。
    let y_scale = input.y().iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if is_perfect_fit(ssr, sst_within, k) || ssr <= n as f64 * (noise * y_scale).powi(2) {
        return Err(PanelError::WithinRegressionFailed {
            source: CommonError::PerfectFit.into(),
        });
    }

    // `cov_type`別の共分散行列と、t検定・信頼区間・F検定の自由度`df_inference`
    // （`Cluster`は`G-1`、`Dk`は`t_periods-1`、それ以外は`df_resid`。標準誤差のスケールに使う
    // `K`（`fe_cluster_k_correction`）とは別軸の値）。FE固有の補正（フルレバレッジ・
    // fixest流のK・2グループ退化の検査）は`PanelStructure::FixedEffects`で渡す。
    let PanelCov {
        cov_params,
        df_inference,
        dk_bandwidth_used,
    } = panel_cov_params(
        &cov_type,
        &PanelCovInput {
            x: x_mat,
            residuals: &residuals,
            xtx_inv: &xtx_inv,
            df_resid,
            df_model,
            n_slopes: k,
            entity: input.entity_codes(),
            structure: PanelStructure::FixedEffects { two_way_time },
        },
    )?;

    // `StudentsT::new`は自由度が正でない場合に失敗するが、`df_inference`は
    // `df_resid >= 1`（関数冒頭の`n <= df_model`検証）・`n_groups - 1 >= 1`
    // （`validate_cluster_groups`が`n_groups >= 2`を保証）・`t_periods - 1 >= 1`
    // （`resolve_dk_bandwidth`が`PanelError::InsufficientDkPeriods`で`t_periods < 2`を
    // 拒否済み、`fe_estimator_fit_one_way_hac_with_single_time_period_is_rejected`参照）の
    // いずれかであり理論上到達不能（`ReEstimator::fit`と同じ「保証済みの不変条件に対する
    // 防御的`Result`化」、`.claude/rules/rust-style.md`「テスト」参照）。
    let t_dist = StudentsT::new(0.0, 1.0, df_inference as f64)
        .map_err(|e| CommonError::internal(e.to_string()))?;
    let InferenceTable {
        std_errors,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
    } = inference_table(&params, &cov_params, &t_dist, confidence_level);

    // パネル固有R²（`fe.rs`のモジュールdoc「パネル固有R²」参照）。within R²は実際に使ったFE構造で
    // demeanした残差ベースで、切片なしのwithin変換後データへの最小二乗の適合度
    // （`has_intercept=false`、非中心化TSS）がそのままこの定義と一致する。
    // between/overallはlinearmodelsの`_rsquared`と完全一致させるため、変換前の元の
    // `y`/`x`から独立に計算し直す。
    let fit_stats = gaussian_goodness_of_fit(within_input.y(), ssr, k, false);
    let r_squared_within = fit_stats.r_squared;
    let r_squared_between =
        fe_r_squared_between(input.y(), input.x(), &params, input.entity_codes());
    let r_squared_overall = fe_r_squared_overall(input.y(), input.x(), &params);

    // `log_likelihood`自体は`SSR/n`のみに依存しdf非依存の式のためそのまま再利用できる
    // （`fe.rs`のモジュールdoc参照）。ペナルティ項の乗数だけ`k`から`df_model`に差し替える。
    let log_likelihood = fit_stats.log_likelihood;
    let aic = -2.0 * log_likelihood + 2.0 * (df_model as f64);
    let bic = -2.0 * log_likelihood + (n as f64).ln() * (df_model as f64);

    // F統計量（`fe.rs`のモジュールdoc「自由度とF統計量」）: 傾き係数`k`個の
    // 同時Wald検定。FEの`cov_params`（`cov_type`別、上で計算済み）・`df_inference`
    // （`cov_type=Cluster`/`Dk`のときだけ`df_resid`から切り替わる、`shared::wald::wald_f_test`の
    // `df_inference`引数と同じ扱い）を使う。`k_constant=0`（FEに切片は無い、上記
    // `OlsInput::from_columns`呼び出しと同じ理由）。
    let f_test = if k == 0 {
        // 説明変数が無いモデル。検定対象が存在しないため`OlsEstimator::fit`同様
        // 定義なし（`None`）。
        None
    } else {
        Some(
            wald_f_test(&params, &cov_params, 0, k, df_inference).map_err(|source| {
                PanelError::FTestFailed {
                    source: source.into(),
                }
            })?,
        )
    };

    Ok(FeEstimator {
        input,
        effects,
        cov_type,
        within_input,
        params,
        residuals: residual_mat,
        log_likelihood,
        df_model,
        df_resid,
        df_inference,
        dk_bandwidth_used,
        n_periods,
        absorbed,
        std_errors,
        cov_params,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
        r_squared_within,
        r_squared_between,
        r_squared_overall,
        aic,
        bic,
        f_test,
    })
}

#[cfg(test)]
mod tests;
