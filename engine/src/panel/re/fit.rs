//! `ReEstimator::fit`の本体。分散成分の推定 → θ計算・準偏差変換 → `OlsEstimator::fit_auxiliary`への
//! 委譲 → `cov_type`別の共分散・F統計量・パネル固有R²・ハウスマン検定の組み立て。
//!
//! テストは`fit/tests.rs`と`fit/tests/`にテーマ別で置く（`basic`・`reference`・`cov_type`・`validation`・`proptests`等）。

use statrs::distribution::StudentsT;

use super::ReEstimator;
use super::hausman::re_hausman_test;
use super::input::{ReCovType, ReEffects, ReInput};
use super::r_squared::re_r_squared;
use super::transform::{
    compute_two_way_theta, quasi_demean_transform, quasi_demean_transform_two_way,
};
use super::variance::VarianceComponents;
use super::variance_method::{VarianceMethod, variance_components};
use crate::linear::ols::{CovType, OlsEstimator, OlsInput};
use crate::panel::common::{
    PanelCov, PanelCovInput, PanelError, PanelStructure, panel_cov_params, quasi_demean_column,
    quasi_demean_two_way, xtx_inverse,
};
use crate::panel::fe::FeEffects;
use crate::shared::error::CommonError;
use crate::shared::group_codes::GroupCodes;
use crate::shared::inference::{InferenceTable, inference_table};
use crate::shared::wald::wald_f_test;

/// [`ReEstimator::fit_with_variance_method`]の本体。契約（エラー条件を含む）は`fit_with_variance_method`と
/// `fit`のdocコメント参照。
pub(super) fn fit_re(
    input: ReInput,
    effects: ReEffects,
    cov_type: ReCovType,
    variance_method: VarianceMethod,
    confidence_level: f64,
) -> Result<ReEstimator, PanelError> {
    // faerのグローバル並列度をPar::Seqに固定する（`crate::shared::parallelism`。
    // 委譲先の`FeEstimator::fit`/`OlsEstimator::fit`自身も呼ぶが、`cargo test -p engine`
    // で`ReEstimator::fit`を直接叩く経路との統一のためここでも呼ぶ、
    // `engine/src/panel/CLAUDE.md`「faerのグローバル並列度」参照）。
    crate::shared::parallelism::ensure_serial();

    let VarianceComponents {
        sigma2_eps,
        sigma2_u,
        sigma2_lambda,
        fe: fe_for_sigma2_eps,
        means: entity_means,
        time_varying,
    } = variance_components(&input, effects, variance_method, confidence_level)?;
    debug_assert_eq!(fe_for_sigma2_eps.input().x().len(), time_varying.len());
    // 内部FEはRE本体と同じ個体効果構造（within R²・ハウスマン検定がその within 変換を再利用する前提）。
    debug_assert_eq!(
        fe_for_sigma2_eps.effects(),
        match effects {
            ReEffects::OneWay => FeEffects::OneWay,
            ReEffects::TwoWay => FeEffects::TwoWay,
        }
    );

    // 切片復元用の定数列（すべて1.0）を、y/xと同じθで変換する
    // （`re.rs`のモジュールdoc「`ReEstimator::fit`のパイプライン」参照。`OlsInput::from_columns`の
    // `include_intercept=true`は使えない——それだと変換されない生の`1.0`列に
    // なってしまう）。2-wayでは定数列が`1 - θ₁ - θ₂ + θ₃`倍になる。
    let const_column = vec![1.0; input.nobs()];
    let (y, x, const_transformed, theta_by_entity, two_way_theta) = match effects {
        ReEffects::OneWay => {
            let (theta, y, x) = quasi_demean_transform(&input, sigma2_eps, sigma2_u);
            let c = quasi_demean_column(&const_column, input.entity_codes(), &theta);
            (y, x, c, Some(theta), None)
        }
        ReEffects::TwoWay => {
            let theta = compute_two_way_theta(
                sigma2_eps,
                sigma2_u,
                sigma2_lambda,
                input.n_entities(),
                input.time_codes().map_or(0, GroupCodes::n_groups),
            );
            let (y, x) = quasi_demean_transform_two_way(&input, theta)?;
            // `quasi_demean_transform_two_way`が`time`の存在を確認済み。
            let time = input.time_codes().ok_or(PanelError::TwoWayRequiresTime)?;
            let c = quasi_demean_two_way(&const_column, input.entity_codes(), time, theta);
            (y, x, c, None, Some(theta))
        }
    };

    let mut x_all = Vec::with_capacity(x.len() + 1);
    x_all.push(const_transformed);
    x_all.extend(x);

    let mut param_names = Vec::with_capacity(input.x_names().len() + 1);
    param_names.push("const".to_string());
    param_names.extend(input.x_names().iter().cloned());

    // `OlsInput::from_columns`が返しうる`LeastSquaresError::Common(DimensionMismatch)`は
    // ここでは理論上到達不能: `y`/`x_all`はどちらも`quasi_demean_column`が
    // `input.y()`/`input.x()`（`ReInput::from_group_codes`が既に同じ長さであることを
    // 検証済み）と`const_column`（`input.nobs()`で長さを揃えている）から1対1で
    // 生成した同じ長さの列であり、この関数内で長さがずれる操作をしていない
    // （`FeEstimator::fit`の同種のコメントと同じ判断）。
    let ols_input = OlsInput::from_columns(
        &y,
        &x_all,
        param_names.clone(),
        false,
        input.dep_var_name().to_string(),
    )
    .map_err(|source| PanelError::QuasiDemeanedRegressionFailed { source })?;
    let estimator = OlsEstimator::fit_auxiliary(ols_input, CovType::Classical, confidence_level)
        .map_err(|source| PanelError::QuasiDemeanedRegressionFailed { source })?;

    let n = estimator.input().nobs();
    let df_model = estimator.input().k();
    let df_resid = n - df_model;

    // `cov_type`別の共分散行列の計算に使う共通の材料。`OlsEstimator`は
    // `cov_params`をprivateで保持しており再利用できないため、`estimator.input().x()`
    // （既に持っている変換済み設計行列、`OlsInput::x()`は公開）から独立に計算し直す
    // （`FeEstimator::fit`と同型だが、REは`design_matrix_from_columns`で組み立て
    // 直す必要が無い——`x_all`は既に`Mat`化済み）。
    let x_mat = estimator.input().x();
    let xtx_inv = xtx_inverse(x_mat)?;
    let residuals: Vec<f64> = (0..n).map(|i| *estimator.residuals().get(i, 0)).collect();

    // `df_inference`はt検定・信頼区間に使う自由度。`cov_type=Cluster`のとき`G-1`、
    // `Dk`のとき`t_periods-1`に切り替える（fixestの`ssc()`既定`t.df="min"`、
    // 。`FeEstimator::fit`と同じ切り替えパターン）。それ以外
    // （Classical/HC1-3）は`df_resid`のまま。
    // クラスター列のコード。既定（entityクラスター）は`ReInput`のコードを再利用し、明示指定の
    // 列は渡されたコードを、RE本体の共分散とハウスマン補助回帰で共有する。
    let cluster_codes = match &cov_type {
        ReCovType::Cluster {
            groups: Some(groups),
        } => groups,
        _ => input.entity_codes(),
    };

    // `cov_type`別の共分散行列と、t検定・信頼区間の自由度`df_inference`（`Cluster`は`G-1`、
    // `Dk`は`t_periods-1`、それ以外は`df_resid`）。REの変換済み設計行列には省略された固定効果
    // ダミーが無いため、FEのような補正（フルレバレッジ・fixest流のK）は不要
    // （`PanelStructure::RandomEffects`、`fit()`のdocコメント「`cov_type`対応」参照）。
    // 検定対象の傾きの数`q`は`df_model - 1`（`ols::fit`の`k - k_constant`と同じ規約）。
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
            n_slopes: df_model - 1,
            entity: input.entity_codes(),
            structure: PanelStructure::RandomEffects,
        },
    )?;

    // `StudentsT::new`は自由度が正でない場合に失敗するが、`df_inference`は
    // `df_resid >= 1`（`OlsEstimator::fit`成功時点で保証済み）・
    // `n_groups - 1 >= 1`（`validate_cluster_count_covers_slopes`が保証）・
    // `t_periods - 1 >= 1`（`resolve_dk_bandwidth`が`PanelError::
    // InsufficientDkPeriods`で`t_periods<2`を拒否済み、`fe/fit.rs`と同じ保証）の
    // いずれかであり理論上到達不能（`FeEstimator::fit`と同じ「保証済みの不変
    // 条件に対する防御的`Result`化」、`.claude/rules/rust-style.md`「テスト」参照）。
    let t_dist = StudentsT::new(0.0, 1.0, df_inference as f64)
        .map_err(|e| CommonError::internal(e.to_string()))?;
    let InferenceTable {
        std_errors,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
    } = inference_table(estimator.params(), &cov_params, &t_dist, confidence_level);

    // F統計量（2.1節）: 傾き係数`df_model - 1`個（定数項を除く）が同時にゼロという
    // 帰無仮説のWald F検定（`β'V⁻¹β/q`、`plm::pwaldtest(test="F", vcov=...)`と同じ二次形式）。
    // FEの`FeEstimator::fit`と同じく、`cov_type`別に計算し直した`cov_params`
    // （上で計算済み）・`df_inference`（`Cluster`のとき`G-1`、`Dk`のとき`t_periods-1`、
    // それ以外は`df_resid`）を`wald_f_test`に渡す（サンドイッチ計算を複製しない）。
    // `estimator().f_statistic()`/`wald_test_last_columns`は`CovType::Classical`の
    // 内部OLSの共分散・`df_resid`ベースのため、RE自身の`cov_type`を反映できず使わない。
    // 先頭列が変換済み定数項なので`k_constant=1`、検定対象は傾き`df_model - 1`個。
    // Wald二次形式（`V`が正定値）のため負値にならない
    // （`linearmodels`のSST/SSR方式は極端な不均衡パネルで負値になる、`re.rs`のモジュールdoc
    // 「F統計量」参照）。
    let f_test = if df_model == 1 {
        // 傾き係数が無い（定数項のみ）モデル。検定対象が存在しないため`OlsEstimator::fit`
        // 自身の`df_model==0`分岐と同様定義なし（`None`）。
        None
    } else {
        Some(
            wald_f_test(
                estimator.params(),
                &cov_params,
                1,
                df_model - 1,
                df_inference,
            )
            .map_err(|source| PanelError::FTestFailed {
                source: source.into(),
            })?,
        )
    };

    // パネル固有R²（2.3節）。`input`はこの後`Self`に格納するため、
    // ムーブ前にここで計算する。
    //
    // 2-wayでも、`fe_for_sigma2_eps`はRE本体と同じ個体効果構造（2-way）の内部FE推定量で、
    // within R²のwithin変換済みデータ・ハウスマン検定の`X̃`はその`within_input()`から取る
    // （`entity_means`は個体平均で、between/overall R²は1-wayと同じ定義、`re-spec.md`3.6節）。
    let (r_squared_within, r_squared_between, r_squared_overall) = re_r_squared(
        &input,
        &fe_for_sigma2_eps,
        &time_varying,
        &entity_means,
        estimator.params(),
        df_model,
    );

    // ハウスマン検定（`re-spec.md`3.7節、`re.rs`のモジュールdoc「ハウスマン検定」参照）。
    // 比較用の内部FEは、`swamy_arora_variance_components`が返すFE推定量（RE本体と同じ
    // 個体効果構造、1-wayはentity、2-wayはentity+time）を再利用する。
    let hausman_result = re_hausman_test(
        &fe_for_sigma2_eps,
        &time_varying,
        &y,
        &x_all[1..],
        &param_names[1..],
        input.dep_var_name(),
        &cov_type,
        cluster_codes,
        confidence_level,
    )?;
    let (hausman_statistic, hausman_df, hausman_p_value) = match hausman_result {
        Some((stat, df, p_value)) => (Some(stat), Some(df), Some(p_value)),
        None => (None, None, None),
    };
    let n_periods = match effects {
        ReEffects::OneWay => None,
        ReEffects::TwoWay => input.time_codes().map(GroupCodes::n_groups),
    };

    Ok(ReEstimator {
        input,
        estimator,
        cov_type,
        std_errors,
        cov_params,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
        df_inference,
        effects,
        variance_method,
        sigma2_eps,
        sigma2_u,
        sigma2_lambda: match effects {
            ReEffects::OneWay => None,
            ReEffects::TwoWay => Some(sigma2_lambda),
        },
        theta_by_entity,
        two_way_theta,
        n_periods,
        dk_bandwidth_used,
        df_resid,
        df_model,
        f_test,
        r_squared_within,
        r_squared_between,
        r_squared_overall,
        hausman_statistic,
        hausman_p_value,
        hausman_df,
    })
}

#[cfg(test)]
mod tests;
