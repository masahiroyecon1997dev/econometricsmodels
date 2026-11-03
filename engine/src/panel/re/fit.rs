//! `ReEstimator::fit`の本体。分散成分の推定 → θ計算・準偏差変換 → `OlsEstimator::fit_auxiliary`への
//! 委譲 → `cov_type`別の共分散・F統計量・パネル固有R²・ハウスマン検定の組み立て。

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
mod tests {

    use super::super::test_support::*;

    use super::super::transform::quasi_demean_transform;

    use super::super::variance::VarianceComponents;
    use super::super::variance::entity_means;
    use super::super::variance::swamy_arora_variance_components;

    use crate::linear::common::LeastSquaresError;
    use crate::linear::ols::CovType;
    use crate::linear::ols::OlsEstimator;
    use crate::linear::ols::OlsInput;
    use crate::panel::common::BetweenMeans;

    use crate::panel::common::PanelDimension;

    use crate::panel::common::PanelError;

    use crate::panel::common::TimeKeys;

    use crate::panel::re::ReCovType;
    use crate::panel::re::ReEffects;
    use crate::panel::re::ReEstimator;
    use crate::panel::re::ReInput;
    use crate::panel::re::VarianceMethod;

    use crate::shared::error::CommonError;
    use crate::shared::error::ErrorInfo;

    use crate::shared::error_catalog::Stage;
    use crate::shared::group_codes::GroupCodes;
    use crate::shared::inference;

    #[test]
    fn re_estimator_fit_with_time_invariant_regressor_matches_plm_reference() {
        let re = ReEstimator::fit(
            time_invariant_input(&["z", "x1", "x2"]),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();

        // plm: coef(m)・sqrt(diag(vcov(m)))・pwaldtest(test = "F")（`y ~ z + x1 + x2`）。
        let expected_params = [
            1.048_652_338_506_021,
            0.268_400_519_056_541,
            1.279_994_377_183_84,
            -0.633_029_086_597_314,
        ];
        let expected_se = [
            0.496_087_736_469_640,
            0.337_996_679_061_904,
            0.124_772_510_093_148,
            0.131_263_803_920_653,
        ];
        for j in 0..4 {
            let b = *re.estimator().params().get(j, 0);
            let se = *re.std_errors().get(j, 0);
            assert!(
                (b - expected_params[j]).abs() < 1e-9 * expected_params[j].abs().max(1.0),
                "param {j}: {b}"
            );
            assert!(
                (se - expected_se[j]).abs() < 1e-9 * expected_se[j],
                "std_error {j}: {se}"
            );
        }
        assert!((re.f_statistic().unwrap() - 46.788_682_481_996_1).abs() < 1e-8 * 46.79);
        assert_eq!(re.df_model(), 4);
        assert_eq!(re.df_resid(), 24 - 4);

        // ハウスマン検定: `phtest(method = "aux")`の補助回帰を再現して`X̃`の係数を名前で
        // 選んだ参照値（`phtest`は時間不変の列があると別の係数を検定する）。
        assert_eq!(re.hausman_df(), Some(2));
        assert!((re.hausman_statistic().unwrap() - 3.306_375_679_056_07).abs() < 1e-8 * 3.31);
        assert!((re.hausman_p_value().unwrap() - 0.191_438_659_130_983).abs() < 1e-8);
    }

    #[test]
    fn re_estimator_fit_with_time_invariant_regressor_is_invariant_to_column_order() {
        // 時間不変の列が先頭・中央・末尾のどこにあっても、同じ列には同じ結果が付く
        // （内部FEに渡す列と`params`の対応が列順に依存しない）。R²のwithinは時間変化の
        // ある列の係数だけを使う。
        let base = ReEstimator::fit(
            time_invariant_input(&["z", "x1", "x2"]),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        for order in [["x1", "z", "x2"], ["x1", "x2", "z"]] {
            let re = ReEstimator::fit(
                time_invariant_input(&order),
                ReEffects::OneWay,
                ReCovType::Classical,
                0.95,
            )
            .unwrap();
            let names = re.estimator().input().param_names().to_vec();
            let base_names = base.estimator().input().param_names().to_vec();
            for name in ["const", "z", "x1", "x2"] {
                let a = names.iter().position(|n| n == name).unwrap();
                let b = base_names.iter().position(|n| n == name).unwrap();
                assert!(
                    (*re.estimator().params().get(a, 0) - *base.estimator().params().get(b, 0))
                        .abs()
                        < 1e-10,
                    "{name} in order {order:?}"
                );
            }
            assert!((re.r_squared_within() - base.r_squared_within()).abs() < 1e-12);
            assert!(
                (re.hausman_statistic().unwrap() - base.hausman_statistic().unwrap()).abs() < 1e-9
            );
            assert_eq!(re.hausman_df(), Some(2));
        }
    }

    #[test]
    fn re_estimator_fit_with_time_invariant_regressor_supports_every_cov_type_and_unbalanced_panel()
    {
        // 全`cov_type`で時間不変の列を含めて推定でき、ハウスマン検定の自由度は
        // 時間変化のある列の数（2）。不均衡パネル（先頭エンティティの末尾2行を落とす）でも
        // 同じ。係数は`cov_type`に依らず同じ。
        let TimeInvariantColumns {
            y,
            z,
            x1,
            x2,
            entity,
        } = time_invariant_columns();
        let keep: Vec<usize> = (0..24).filter(|&i| i != 2 && i != 3).collect();
        let pick = |v: &[f64]| keep.iter().map(|&i| v[i]).collect::<Vec<f64>>();
        let entity_u: Vec<String> = keep.iter().map(|&i| entity[i].clone()).collect();
        let time: Vec<String> = keep.iter().map(|&i| format!("t{}", i % 4)).collect();
        let build = |balanced: bool| {
            let (yy, zz, a, b, ee, tt) = if balanced {
                (
                    y.clone(),
                    z.clone(),
                    x1.clone(),
                    x2.clone(),
                    entity.clone(),
                    (0..24).map(|i| format!("t{}", i % 4)).collect::<Vec<_>>(),
                )
            } else {
                (
                    pick(&y),
                    pick(&z),
                    pick(&x1),
                    pick(&x2),
                    entity_u.clone(),
                    time.clone(),
                )
            };
            ReInput::from_columns(
                &yy,
                &[zz, a, b],
                strings(&["z", "x1", "x2"]),
                &ee,
                Some(&tt),
                "y".into(),
            )
            .unwrap()
        };
        for balanced in [true, false] {
            let base = ReEstimator::fit(
                build(balanced),
                ReEffects::OneWay,
                ReCovType::Classical,
                0.95,
            )
            .unwrap();
            for cov in [
                ReCovType::Hc1,
                ReCovType::Hc2,
                ReCovType::Hc3,
                ReCovType::Cluster { groups: None },
                ReCovType::Dk {
                    bandwidth: Some(1),
                    time: TimeKeys::lexicographic(if balanced {
                        (0..24).map(|i| format!("t{}", i % 4)).collect()
                    } else {
                        time.clone()
                    }),
                },
            ] {
                let re = ReEstimator::fit(build(balanced), ReEffects::OneWay, cov.clone(), 0.95)
                    .unwrap();
                assert_eq!(re.hausman_df(), Some(2), "{cov:?} balanced={balanced}");
                assert!(re.hausman_statistic().unwrap() >= 0.0);
                for j in 0..4 {
                    assert_eq!(
                        *re.estimator().params().get(j, 0),
                        *base.estimator().params().get(j, 0)
                    );
                    assert!(re.std_errors().get(j, 0).is_finite());
                }
            }
        }
    }

    #[test]
    fn re_estimator_fit_rejects_time_invariant_regressor_collinear_with_intercept() {
        // 全エンティティで同じ値の列は切片と共線になる。between回帰ではこの列を落とすため
        // そこでは失敗せず、変換後の最終回帰の設計行列が特異になって計算エラーになる
        // （時間不変の列を許容しても、識別できない入力は通さない）。
        let TimeInvariantColumns { y, x1, entity, .. } = time_invariant_columns();
        let input = ReInput::from_columns(
            &y,
            &[x1, vec![2.0; 24]],
            strings(&["x1", "c"]),
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        assert!(matches!(
            ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95),
            Err(PanelError::QuasiDemeanedRegressionFailed { .. })
        ));
    }

    #[test]
    fn one_way_mundlak_specification_is_estimated_and_tests_only_the_identified_columns() {
        // `x1`の個体平均を入れると`x̃1 = x1* - x1bar*`が別名になり、ハウスマン検定は`x̃2`だけを
        // 検定する（`hausman_df = 1`）。列順によらず同じ統計量。分散成分は個体平均を入れない
        // 場合と同じ（時間不変の`x1bar`は内部FEにも入らない）。
        let fit = |names: &[&str]| {
            ReEstimator::fit(
                mundlak_input(names, false),
                ReEffects::OneWay,
                ReCovType::Classical,
                0.95,
            )
            .unwrap()
        };
        let with_bar = fit(&["x1", "x2", "x1bar"]);
        let without = fit(&["x1", "x2"]);
        assert_eq!(with_bar.hausman_df(), Some(1));
        assert_eq!(without.hausman_df(), Some(2));
        assert_close(with_bar.sigma2_u(), without.sigma2_u(), 1e-12, "sigma2_u");
        assert_close(
            with_bar.sigma2_eps(),
            without.sigma2_eps(),
            1e-12,
            "sigma2_eps",
        );
        let reordered = fit(&["x1bar", "x1", "x2"]);
        assert_eq!(reordered.hausman_df(), Some(1));
        assert_close(
            reordered.hausman_statistic().unwrap(),
            with_bar.hausman_statistic().unwrap(),
            1e-9,
            "hausman statistic",
        );
        assert_eq!(with_bar.estimator().params().nrows(), 4);
    }

    #[test]
    fn mundlak_cluster_prevalidation_counts_the_identified_auxiliary_slopes() {
        // 補助回帰の傾き係数は`X*`の3個と別名を除いた`x̃`の1個で4個。クラスター数が5なら成功し、
        // 4なら補助回帰の事前検証で`HausmanTestFailed`になる（別名を数えると5個になり、
        // クラスター数5でも失敗してしまう）。RE本体の事前検証（傾き3個）は両方で通る。
        let fit = |labels: &[&str]| {
            ReEstimator::fit(
                mundlak_input(&["x1", "x2", "x1bar"], false),
                ReEffects::OneWay,
                ReCovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&strings(labels))),
                },
                0.95,
            )
        };
        let five = [
            "a", "a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c", "d", "d", "d", "d",
            "e", "e", "e", "e", "e", "e", "e",
        ];
        assert_eq!(fit(&five).unwrap().hausman_df(), Some(1));
        let four: Vec<&str> = five
            .iter()
            .map(|g| if *g == "e" { "d" } else { g })
            .collect();
        assert!(matches!(
            fit(&four),
            Err(PanelError::HausmanTestFailed { .. })
        ));
    }

    #[test]
    fn one_way_mundlak_with_a_single_regressor_has_no_testable_column() {
        // `x1`と`x1bar`だけなら`x̃1`が別名で、検定できる列が残らない（`hausman_*`は`None`）。
        let re = ReEstimator::fit(
            mundlak_input(&["x1", "x1bar"], false),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        assert_eq!(re.hausman_df(), None);
        assert_eq!(re.hausman_statistic(), None);
    }

    #[test]
    fn two_way_mundlak_with_entity_and_time_means_drops_the_aliased_columns() {
        // 個体平均と時点平均の両方を入れる。個体平均の回帰では`x1bar`が`x1`と、時点平均の回帰では
        // `x1tbar`が`x1`と同じ列になり、`σ_u²`・`σ_λ²`は平均を入れない場合と同じ。ハウスマン検定は
        // `x̃1`が別名で`x̃2`だけを検定する。
        let fit = |names: &[&str]| {
            ReEstimator::fit(
                mundlak_input(names, true),
                ReEffects::TwoWay,
                ReCovType::Classical,
                0.95,
            )
            .unwrap()
        };
        let with_bars = fit(&["x1", "x2", "x1bar", "x1tbar"]);
        let without = fit(&["x1", "x2"]);
        assert_close(with_bars.sigma2_u(), without.sigma2_u(), 1e-12, "sigma2_u");
        assert_close(
            with_bars.sigma2_lambda().unwrap(),
            without.sigma2_lambda().unwrap(),
            1e-12,
            "sigma2_lambda",
        );
        assert_eq!(with_bars.hausman_df(), Some(1));
        // 個体平均だけなら時点方向の別名は無く、`x̃1`・`x̃2`の両方を検定する。
        assert_eq!(fit(&["x1", "x2", "x1bar"]).hausman_df(), Some(2));
    }

    #[test]
    fn re_estimator_fit_rejects_proportional_time_invariant_regressors() {
        // 定数ではない時間不変の列同士が比例していれば、between回帰では後ろの列が落ちるが、
        // 最終回帰（全列を残す）は特異になる。
        let TimeInvariantColumns { y, x1, entity, .. } = time_invariant_columns();
        let z: Vec<f64> = (0..24).map(|r| (r / 4) as f64).collect();
        let z2: Vec<f64> = z.iter().map(|v| 2.0 * v).collect();
        let input = ReInput::from_columns(
            &y,
            &[x1, z, z2],
            strings(&["x1", "z", "z2"]),
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        assert!(matches!(
            ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95),
            Err(PanelError::QuasiDemeanedRegressionFailed { .. })
        ));
    }

    #[test]
    fn two_way_fit_matches_plm_for_every_covariance() {
        // plm: coef(m)・sqrt(diag(vcov(m)))・vcovHC(white1, HC1)・vcovHC(arellano, sss)・
        // pwaldtest(test = "F")（`y ~ x1 + xi + xt`）。一次元でしか変わらない列があっても、
        // 変換後の回帰は全列を使う。
        let expected_params = [
            0.307_929_272_368_423,
            0.891_324_961_003_34,
            -0.195_188_876_604_849,
            -1.092_637_705_135_03,
        ];
        let expected_classical = [
            1.267_090_128_928_92,
            0.364_650_749_295_775,
            1.004_605_484_001_94,
            0.959_418_507_617_424,
        ];
        let expected_hc1 = [
            1.266_310_230_032_36,
            0.413_092_525_743_793,
            0.843_297_450_276_094,
            0.919_097_624_137_469,
        ];
        let expected_cluster = [
            1.386_067_165_393_28,
            0.352_916_535_514_244,
            0.690_856_338_748_332,
            1.076_220_459_898_9,
        ];
        for (cov_type, expected_se, df_denom) in [
            (ReCovType::Classical, expected_classical, 20),
            (ReCovType::Hc1, expected_hc1, 20),
            (ReCovType::Cluster { groups: None }, expected_cluster, 5),
        ] {
            let label = format!("{cov_type:?}");
            let re = ReEstimator::fit(
                two_way_input(&["x1", "xi", "xt"]),
                ReEffects::TwoWay,
                cov_type,
                0.95,
            )
            .unwrap();
            for j in 0..4 {
                assert_close(
                    *re.estimator().params().get(j, 0),
                    expected_params[j],
                    1e-6,
                    &format!("{label} param {j}"),
                );
                assert_close(
                    *re.std_errors().get(j, 0),
                    expected_se[j],
                    1e-6,
                    &format!("{label} se {j}"),
                );
            }
            assert_eq!(re.df_model(), 4, "{label}");
            assert_eq!(re.df_resid(), 20, "{label}");
            assert_eq!(re.f_df(), Some((3, df_denom)), "{label}");
        }
        let classical = ReEstimator::fit(
            two_way_input(&["x1", "xi", "xt"]),
            ReEffects::TwoWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        assert_close(
            classical.f_statistic().unwrap(),
            2.341_321_617_208_2,
            1e-6,
            "F",
        );
        assert_eq!(classical.effects(), ReEffects::TwoWay);
        assert_eq!(classical.n_periods(), Some(4));
    }

    #[test]
    fn two_way_results_do_not_depend_on_the_column_order() {
        // `time_varying`の添字対応（`x_star[time_varying[m]]`・`params[time_varying[m] + 1]`）は、
        // 時間変化のある列が先頭以外にあるときに初めて効く。
        let fit = |names: &[&str]| {
            ReEstimator::fit(
                two_way_input(names),
                ReEffects::TwoWay,
                ReCovType::Classical,
                0.95,
            )
            .unwrap()
        };
        let base = fit(&["x1", "xi", "z", "x2"]);
        for names in [
            ["xi", "x1", "x2", "z"],
            ["z", "x2", "xi", "x1"],
            ["x2", "z", "x1", "xi"],
        ] {
            let other = fit(&names);
            let tag = format!("{names:?}");
            assert_eq!(other.hausman_df(), base.hausman_df(), "{tag}");
            assert_rel(
                other.hausman_statistic().unwrap(),
                base.hausman_statistic().unwrap(),
                1e-9,
                &tag,
            );
            for (a, b) in [
                (other.r_squared_within(), base.r_squared_within()),
                (other.r_squared_between(), base.r_squared_between()),
                (other.r_squared_overall(), base.r_squared_overall()),
            ] {
                assert_rel(a, b, 1e-9, &tag);
            }
        }
    }

    #[test]
    fn two_way_cluster_prevalidation_follows_the_auxiliary_slopes() {
        // 時点でクラスタリング（G = 4）。RE本体の傾き3個（`G > 3`）は満たすが、補助回帰の傾き
        // （`X*`の3個と`X̃`の1個で4個）には足りず、`fit()`はハウスマン検定の失敗で終わる。
        let c = two_way_columns();
        let by_time = || ReCovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&c.time)),
        };
        let err = ReEstimator::fit(
            two_way_input(&["x1", "xi", "xt"]),
            ReEffects::TwoWay,
            by_time(),
            0.95,
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                PanelError::HausmanTestFailed {
                    source: LeastSquaresError::Common(
                        CommonError::InsufficientClustersForInference { g: 4, q: 4 }
                    )
                }
            ),
            "{err:?}"
        );
        // 傾きが3個（`x1`・`xi`と`X̃`の`x1`）なら通り、参照値（手計算）と一致する。
        let re = ReEstimator::fit(
            two_way_input(&["x1", "xi"]),
            ReEffects::TwoWay,
            by_time(),
            0.95,
        )
        .unwrap();
        assert_eq!(re.hausman_df(), Some(1));
        assert_rel(
            re.hausman_statistic().unwrap(),
            2.27570721529081,
            1e-9,
            "cluster time",
        );
    }

    #[test]
    fn two_way_cluster_and_dk_time_columns_other_than_entity_and_time_are_used() {
        // `cluster`をentityと別の列（3グループ）にする。`dk_time`をtimeと別の並び順の列にする
        // （ラグ付きの計算は時点の順序に依存するため、`time`を借用すると値が変わる）。
        let c = two_way_columns();
        let group: Vec<String> = (0..24).map(|r| format!("g{}", r / 8)).collect();
        let re = ReEstimator::fit(
            two_way_input(&["x1"]),
            ReEffects::TwoWay,
            ReCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&group)),
            },
            0.95,
        )
        .unwrap();
        assert_rel(
            re.hausman_statistic().unwrap(),
            7.77470735811636,
            1e-9,
            "cluster groups",
        );

        let order = [30_i128, 10, 40, 20];
        let values: Vec<i128> = (0..24).map(|r| order[r % 4]).collect();
        let labels: Vec<String> = values.iter().map(|v| v.to_string()).collect();
        for (names, want) in [
            (&["x1"][..], 1.3729036142545),
            (&["x1", "xi", "xt"][..], 0.0199827525899269),
        ] {
            let re = ReEstimator::fit(
                two_way_input(names),
                ReEffects::TwoWay,
                ReCovType::Dk {
                    bandwidth: Some(1),
                    time: TimeKeys::by_integer(labels.clone(), &values).unwrap(),
                },
                0.95,
            )
            .unwrap();
            assert_rel(
                re.hausman_statistic().unwrap(),
                want,
                1e-9,
                &format!("{names:?}"),
            );
        }
        let _ = c;
    }

    #[test]
    fn two_way_dk_prevalidation_requires_more_periods_than_slopes() {
        // `dk_time`を3時点にまとめる（`min(t, 3)`）。RE本体の傾きは`x1 + xi + xt`で3個のため
        // `T = 3 <= q`で事前検証に落ち、`x1 + xi`（q = 2）は通って手計算と一致する。
        let values: Vec<i128> = (0..24).map(|r| ((r % 4) as i128 + 1).min(3)).collect();
        let labels: Vec<String> = values.iter().map(|v| v.to_string()).collect();
        let dk = || ReCovType::Dk {
            bandwidth: Some(1),
            time: TimeKeys::by_integer(labels.clone(), &values).unwrap(),
        };
        let err = ReEstimator::fit(
            two_way_input(&["x1", "xi", "xt"]),
            ReEffects::TwoWay,
            dk(),
            0.95,
        )
        .unwrap_err();
        assert!(
            matches!(
                &err,
                PanelError::InStage {
                    stage: Stage::OverallFTest,
                    source,
                } if **source == PanelError::InsufficientDkPeriodsForInference { t_periods: 3, q: 3 }
            ),
            "{err:?}"
        );
        let re =
            ReEstimator::fit(two_way_input(&["x1", "xi"]), ReEffects::TwoWay, dk(), 0.95).unwrap();
        assert_rel(
            re.hausman_statistic().unwrap(),
            4.37954416942038,
            1e-9,
            "dk 3 periods",
        );
    }

    #[test]
    fn fit_exposes_variance_components_and_weights() {
        // 1-way（plm::ercomp、`y ~ x1 + xt`）: σ²と個体ごとのθ（バランスなので全個体で同じ）。
        let one_way = ReEstimator::fit(
            subset_input(&(0..24).collect::<Vec<_>>(), &["x1", "xt"], false),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        assert_rel(
            one_way.sigma2_eps(),
            5.558_492_530_031_272,
            1e-9,
            "sigma2_eps",
        );
        assert_rel(one_way.sigma2_u(), 2.781_216_088_364_69, 1e-9, "sigma2_u");
        assert_eq!(one_way.sigma2_lambda(), None);
        assert_eq!(one_way.two_way_theta(), None);
        let theta = one_way.theta_by_entity().unwrap();
        assert_eq!(theta.len(), 6);
        for t in theta {
            assert_rel(*t, 0.422_786_083_747_044_6, 1e-9, "theta");
        }

        // 2-way（plm::ercomp、`y ~ x1 + xi + xt`）: σ²（idios, id, time）とθ（id, time, total）。
        let two_way = ReEstimator::fit(
            two_way_input(&["x1", "xi", "xt"]),
            ReEffects::TwoWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        assert_rel(
            two_way.sigma2_eps(),
            0.710_563_608_206_315_4,
            1e-9,
            "sigma2_eps",
        );
        assert_rel(two_way.sigma2_u(), 3.019_989_504_528_157, 1e-9, "sigma2_u");
        assert_rel(
            two_way.sigma2_lambda().unwrap(),
            2.596_736_326_025_116_6,
            1e-9,
            "sigma2_lambda",
        );
        assert_eq!(two_way.theta_by_entity(), None);
        let (te, tt, tot) = two_way.two_way_theta().unwrap();
        assert_rel(te, 0.764_301_210_325_029_3, 1e-9, "theta_entity");
        assert_rel(tt, 0.791_153_185_965_248_8, 1e-9, "theta_time");
        assert_rel(tot, 0.713_712_012_990_144_2, 1e-9, "theta_total");
    }

    #[test]
    fn one_way_fit_ignores_time_and_has_no_n_periods() {
        // 1-wayでは`ReInput`に`time`があっても使わず、`n_periods`は`None`。
        let two_way_data = two_way_input(&["x1"]);
        let c = two_way_columns();
        let without_time =
            ReInput::from_columns(&c.y, &[c.x1], strings(&["x1"]), &c.entity, None, "y".into())
                .unwrap();

        let a =
            ReEstimator::fit(two_way_data, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
        let b =
            ReEstimator::fit(without_time, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

        assert_eq!(a.n_periods(), None);
        assert_eq!(
            a.estimator().params().get(1, 0),
            b.estimator().params().get(1, 0)
        );
        assert!(a.hausman_statistic().is_some());
    }

    #[test]
    fn two_way_fit_rejects_unbalanced_panel_singletons_and_missing_time() {
        let c = two_way_columns();
        let build = |keep: &dyn Fn(usize) -> bool, time: bool| {
            let rows: Vec<usize> = (0..24).filter(|&r| keep(r)).collect();
            let pick = |v: &[f64]| rows.iter().map(|&r| v[r]).collect::<Vec<f64>>();
            let entity: Vec<String> = rows.iter().map(|&r| c.entity[r].clone()).collect();
            let period: Vec<String> = rows.iter().map(|&r| c.time[r].clone()).collect();
            ReInput::from_columns(
                &pick(&c.y),
                &[pick(&c.x1)],
                strings(&["x1"]),
                &entity,
                time.then_some(period.as_slice()),
                "y".into(),
            )
            .unwrap()
        };
        let fit = |input| ReEstimator::fit(input, ReEffects::TwoWay, ReCovType::Classical, 0.95);

        // 1行欠けた不均衡パネル。
        assert!(matches!(
            fit(build(&|r| r != 5, true)).unwrap_err(),
            PanelError::UnbalancedPanelForTwoWay { .. }
        ));
        // 先頭エンティティが1観測だけ（singleton）。
        assert!(matches!(
            fit(build(&|r| r >= 4 || r == 0, true)).unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                ..
            }
        ));
        // `time`が無い2-way。
        assert_eq!(
            fit(build(&|_| true, false)).unwrap_err(),
            PanelError::TwoWayRequiresTime
        );
    }

    #[test]
    fn one_way_time_only_column_detection_does_not_depend_on_row_order_or_centering() {
        // 並べ替えと標準化（平均0・個体平均が丸め誤差になる列）をしても、時点間でのみ変わる列が
        // 一定と判定され、結果は元のデータと一致する。
        let c = two_way_columns();
        let mean = c.xt.iter().take(4).sum::<f64>() / 4.0;
        let sd = (c.xt.iter().take(4).map(|v| (v - mean).powi(2)).sum::<f64>() / 3.0).sqrt();
        let standardized: Vec<f64> = c.xt.iter().map(|v| (v - mean) / sd).collect();
        // 行を決定的に並べ替える（エンティティ・時点が混ざる順）。
        let order: Vec<usize> = (0..24).map(|i| (i * 7) % 24).collect();
        let pick = |v: &[f64]| order.iter().map(|&r| v[r]).collect::<Vec<f64>>();
        let entity: Vec<String> = order.iter().map(|&r| c.entity[r].clone()).collect();
        let build = |xt: &[f64], shuffled: bool| {
            let (y, x1, xt, entity) = if shuffled {
                (pick(&c.y), pick(&c.x1), pick(xt), entity.clone())
            } else {
                (c.y.clone(), c.x1.clone(), xt.to_vec(), c.entity.clone())
            };
            ReInput::from_columns(
                &y,
                &[x1, xt],
                strings(&["x1", "xt"]),
                &entity,
                None,
                "y".into(),
            )
            .unwrap()
        };

        let base =
            swamy_arora_variance_components(&build(&standardized, false), ReEffects::OneWay, 0.95)
                .unwrap();
        let shuffled =
            swamy_arora_variance_components(&build(&standardized, true), ReEffects::OneWay, 0.95)
                .unwrap();
        assert_close(shuffled.sigma2_u, base.sigma2_u, 1e-9, "sigma2_u");
        assert_close(shuffled.sigma2_eps, base.sigma2_eps, 1e-9, "sigma2_eps");
        // 標準化しても元の列と同じ分散成分（平均・スケールの変更は列空間を変えない）。
        let raw =
            swamy_arora_variance_components(&build(&c.xt, false), ReEffects::OneWay, 0.95).unwrap();
        assert_close(raw.sigma2_u, base.sigma2_u, 1e-9, "sigma2_u raw");
    }

    #[test]
    fn one_way_with_several_time_only_columns_tests_only_the_identified_ones() {
        // `xt`と`xt2`（どちらも時点間でのみ変わる）と、個体間でのみ変わる`xi`、両方で変わる`x1`。
        // 検定されるのは`x1`だけ。列の順序を入れ替えても係数・統計量は同じ。
        let all: Vec<usize> = (0..24).collect();
        let fit = |names: &[&str]| {
            ReEstimator::fit(
                subset_input(&all, names, false),
                ReEffects::OneWay,
                ReCovType::Classical,
                0.95,
            )
            .unwrap()
        };
        let a = fit(&["x1", "xi", "xt", "xt2"]);
        let b = fit(&["xt2", "xt", "xi", "x1"]);
        assert_eq!(a.hausman_df(), Some(1));
        assert_eq!(b.hausman_df(), Some(1));
        assert_close(
            a.hausman_statistic().unwrap(),
            b.hausman_statistic().unwrap(),
            1e-9,
            "hausman",
        );
        assert_close(
            a.r_squared_within(),
            b.r_squared_within(),
            1e-9,
            "r2 within",
        );
        // `x1`の係数は列の順序によらない（`a`は2番目、`b`は5番目＝最後）。
        assert_close(
            *a.estimator().params().get(1, 0),
            *b.estimator().params().get(4, 0),
            1e-9,
            "x1",
        );
    }

    #[test]
    fn one_way_time_only_column_works_with_every_cov_type() {
        // 新規則は`cov_type`ごとに別の補助回帰の分岐（OLS委譲・クラスター・DK）を通る。
        let c = two_way_columns();
        for (label, cov_type) in all_cov_types(&c.time) {
            let input = subset_input(&(0..24).collect::<Vec<_>>(), &["x1", "xt"], false);
            let re = ReEstimator::fit(input, ReEffects::OneWay, cov_type, 0.95)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(re.hausman_df(), Some(1), "{label}");
            let stat = re.hausman_statistic().unwrap();
            assert!(stat.is_finite() && stat >= 0.0, "{label}: {stat}");
            for j in 0..3 {
                let se = *re.std_errors().get(j, 0);
                assert!(se.is_finite() && se > 0.0, "{label}: se[{j}]={se}");
            }
        }
    }

    #[test]
    fn two_way_fit_accepts_regressors_that_vary_in_only_one_dimension() {
        // 時間変化のある列が0個（内部の2-way FEの説明変数が0個）でも、一次元でしか変わらない列だけで推定できる。
        let all: Vec<usize> = (0..24).collect();
        for names in [vec!["xi", "xt"], vec!["xt"], vec!["xi"]] {
            let re = ReEstimator::fit(
                subset_input(&all, &names, true),
                ReEffects::TwoWay,
                ReCovType::Classical,
                0.95,
            )
            .unwrap_or_else(|e| panic!("{names:?}: {e}"));
            assert_eq!(re.n_periods(), Some(4));
            assert_eq!(re.hausman_statistic(), None);
            for j in 0..=names.len() {
                assert!(re.estimator().params().get(j, 0).is_finite(), "{names:?}");
            }
        }
    }

    #[test]
    fn two_way_fit_reports_computation_errors_by_the_stage_that_fails() {
        let c = two_way_columns();
        // 切片と共線な定数列: between回帰では落ち、最終回帰が特異になる。
        let constant = ReInput::from_columns(
            &c.y,
            &[c.x1.clone(), vec![2.0; 24]],
            strings(&["x1", "c"]),
            &c.entity,
            Some(&c.time),
            "y".into(),
        )
        .unwrap();
        assert!(matches!(
            ReEstimator::fit(constant, ReEffects::TwoWay, ReCovType::Classical, 0.95),
            Err(PanelError::QuasiDemeanedRegressionFailed { .. })
        ));
        // 定数ではない列同士の完全共線（`xi`と`2·xi`）は、between回帰（個体平均の回帰）では後ろの列が
        // 落ち、最終回帰が特異になる。
        let doubled: Vec<f64> = c.xi.iter().map(|v| 2.0 * v).collect();
        let collinear = ReInput::from_columns(
            &c.y,
            &[c.x1.clone(), c.xi.clone(), doubled],
            strings(&["x1", "xi", "xi_twice"]),
            &c.entity,
            Some(&c.time),
            "y".into(),
        )
        .unwrap();
        assert!(matches!(
            ReEstimator::fit(collinear, ReEffects::TwoWay, ReCovType::Classical, 0.95),
            Err(PanelError::QuasiDemeanedRegressionFailed { .. })
        ));
    }

    #[test]
    fn two_way_between_and_fe_degrees_of_freedom_boundaries() {
        let fit = |rows: &[usize], names: &[&str]| {
            ReEstimator::fit(
                subset_input(rows, names, true),
                ReEffects::TwoWay,
                ReCovType::Classical,
                0.95,
            )
        };
        // 6エンティティ×2時点: 時点側のbetween回帰は`T = 2 <= K_b + 1 = 2`で失敗する。
        let two_periods: Vec<usize> = (0..24).filter(|r| r % 4 < 2).collect();
        assert!(matches!(
            fit(&two_periods, &["x1"]),
            Err(PanelError::BetweenRegressionFailed {
                means: BetweenMeans::Time,
                ..
            })
        ));
        // 2エンティティ×4時点: 個体平均側のbetween回帰が`N = 2 <= K_b + 1 = 2`で失敗する
        // （個体平均の回帰が先に走るため、メッセージはどちらの回帰かを区別する）。
        let two_entities: Vec<usize> = (0..24).filter(|r| r / 4 < 2).collect();
        let err = fit(&two_entities, &["x1"]).unwrap_err();
        assert!(matches!(
            err,
            PanelError::BetweenRegressionFailed {
                means: BetweenMeans::Entity,
                ..
            }
        ));
        assert_eq!(
            crate::shared::error::ErrorInfo::report(&err).stage(),
            Some(&crate::shared::error_catalog::Stage::BetweenRegressionEntity)
        );
        // 2エンティティ×2時点: 内部の2-way FEの自由度が足りない（`n = 4 <= k + N + T - 1 = 4`）。
        let tiny: Vec<usize> = (0..24).filter(|&r| r / 4 < 2 && r % 4 < 2).collect();
        let err = fit(&tiny, &["x1"]).unwrap_err();
        assert!(matches!(
            &err,
            PanelError::InStage {
                stage: Stage::VarianceComponentWithinRegression,
                source,
            } if matches!(**source, PanelError::InsufficientDegreesOfFreedom { .. })
        ));
        assert_eq!(
            crate::shared::error::ErrorInfo::report(&err).stage(),
            Some(&Stage::VarianceComponentWithinRegression)
        );
        // 3エンティティ×3時点（`df_resid = 3`）は最小規模で成功する。
        let small: Vec<usize> = (0..24).filter(|&r| r / 4 < 3 && r % 4 < 3).collect();
        let re = fit(&small, &["x1"]).unwrap();
        assert_eq!(re.df_resid(), 9 - 2);
    }

    #[test]
    fn two_way_driscoll_kraay_without_lags_equals_clustering_by_time() {
        // `dk_bandwidth = 0`はラグ項が無く、時点でのクラスターと同じ共分散になる
        // （`K = df_model`、自由度`T - 1`も同じ）。entityと別に指定した時点列で確認する。
        let c = two_way_columns();
        let all: Vec<usize> = (0..24).collect();
        let dk = ReEstimator::fit(
            subset_input(&all, &["x1"], true),
            ReEffects::TwoWay,
            ReCovType::Dk {
                bandwidth: Some(0),
                time: TimeKeys::lexicographic(c.time.clone()),
            },
            0.95,
        )
        .unwrap();
        let cluster = ReEstimator::fit(
            subset_input(&all, &["x1"], true),
            ReEffects::TwoWay,
            ReCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&c.time)),
            },
            0.95,
        )
        .unwrap();
        for j in 0..2 {
            assert_close(
                *dk.std_errors().get(j, 0),
                *cluster.std_errors().get(j, 0),
                1e-10,
                "se",
            );
        }
        // 他の`cov_type`も2-wayで実行でき、標準誤差は正の有限値になる。
        for (label, cov_type) in all_cov_types(&c.time) {
            let re = ReEstimator::fit(
                subset_input(&all, &["x1", "xi", "xt"], true),
                ReEffects::TwoWay,
                cov_type,
                0.95,
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
            for j in 0..4 {
                let se = *re.std_errors().get(j, 0);
                assert!(se.is_finite() && se > 0.0, "{label}: se[{j}]={se}");
            }
        }
    }

    #[test]
    fn two_way_fit_reduces_to_pooled_ols_when_both_group_variances_are_zero() {
        // グループ効果が無いDGP（`y = 1 + 2·x1 + 擬似乱数ノイズ`）では、`σ_u²`・`σ_λ²`が
        // 0に切り上げられ、θがすべて0（変換なし、定数列は1のまま）になって係数・標準誤差が
        // pooled OLS（切片あり、classical）と一致する。
        let c = two_way_columns();
        let mut state = 4_u64;
        let mut noise = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1_u64 << 31) as f64) - 0.5
        };
        let y: Vec<f64> = c.x1.iter().map(|x| 1.0 + 2.0 * x + noise()).collect();
        let input = ReInput::from_columns(
            &y,
            std::slice::from_ref(&c.x1),
            strings(&["x1"]),
            &c.entity,
            Some(&c.time),
            "y".into(),
        )
        .unwrap();
        let vc = swamy_arora_variance_components(&input, ReEffects::TwoWay, 0.95).unwrap();
        assert_eq!(
            (vc.sigma2_u, vc.sigma2_lambda),
            (0.0, 0.0),
            "data must have no group effects"
        );

        let re = ReEstimator::fit(input, ReEffects::TwoWay, ReCovType::Classical, 0.95).unwrap();
        let pooled = OlsEstimator::fit(
            OlsInput::from_columns(
                &y,
                std::slice::from_ref(&c.x1),
                strings(&["x1"]),
                true,
                "y".into(),
            )
            .unwrap(),
            CovType::Classical,
            0.95,
        )
        .unwrap();
        for j in 0..2 {
            assert_close(
                *re.estimator().params().get(j, 0),
                *pooled.params().get(j, 0),
                1e-10,
                &format!("param {j}"),
            );
            assert_close(
                *re.std_errors().get(j, 0),
                *pooled.std_errors().get(j, 0),
                1e-10,
                &format!("se {j}"),
            );
        }
    }

    // ── ReEstimator::fit ─────────────────────────────────────────────────

    #[test]
    fn re_estimator_fit_matches_linearmodels_reference() {
        // `swamy_arora_variance_components_matches_linearmodels_reference`と同じ
        // データ（entity a: T=3, b: T=2, c: T=2）。`linearmodels.RandomEffects`
        // （Python、`exog=[const, x1]`）で実地検証済みの`params`・`resids`と比較する。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

        assert_eq!(
            re.estimator().input().param_names(),
            &["const".to_string(), "x1".to_string()]
        );
        assert!((*re.estimator().params().get(0, 0) - 2.224_977_596_254_189_6).abs() < 1e-9);
        assert!((*re.estimator().params().get(1, 0) - 1.400_245_546_585_47).abs() < 1e-9);

        let expected_resids = [
            0.007_456_619_017_130_906,
            -0.392_788_927_568_339_16,
            -0.193_280_020_739_278_4,
            0.983_357_826_800_407_3,
            0.583_112_280_214_937_3,
            -1.843_693_205_551_097,
            0.756_061_247_863_432_8,
        ];
        for (i, expected) in expected_resids.iter().enumerate() {
            let actual = *re.estimator().residuals().get(i, 0);
            assert!((actual - expected).abs() < 1e-9, "{actual} vs {expected}");
        }

        // `linearmodels.RandomEffects.fit(cov_type="unadjusted").df_resid`/`df_model`と
        // 数値一致（`re-spec.md`3.3節）。n=7、k=2（const+x1）。
        assert_eq!(re.df_resid(), 5);
        assert_eq!(re.df_model(), 2);

        // `estimator()`のdocコメントは「`std_errors`/`test_stats`/`p_values`/`conf_lower`/
        // `conf_upper`/`aic`/`bic`は既に正しいRE推定量になっている」と述べるため、
        // `linearmodels.RandomEffects.
        // fit(cov_type="unadjusted")`の`std_errors`/`tstats`/`pvalues`/`conf_int()`・
        // `loglik`から手計算した`aic`/`bic`と実地数値照合する（`HomoskedasticCovariance`
        // が`debiased=True`時`nobs_eff = nobs - nvar`を使うことの検証、`re.rs`のモジュールdoc
        // 「`ReEstimator::fit`のパイプライン」参照）。
        let expected_std_errors = [2.048_733_66, 0.404_536_75];
        let expected_test_stats = [1.086_025_79, 3.461_355_59];
        let expected_p_values = [0.327_029_7, 0.018_016_02];
        let expected_conf_lower = [-3.041_459_93, 0.360_350_72];
        let expected_conf_upper = [7.491_415_12, 2.440_140_37];
        for j in 0..2 {
            assert!(
                (*re.estimator().std_errors().get(j, 0) - expected_std_errors[j]).abs() < 1e-6,
                "std_errors[{j}]"
            );
            assert!(
                (*re.estimator().test_stats().get(j, 0) - expected_test_stats[j]).abs() < 1e-6,
                "test_stats[{j}]"
            );
            assert!(
                (*re.estimator().p_values().get(j, 0) - expected_p_values[j]).abs() < 1e-6,
                "p_values[{j}]"
            );
            assert!(
                (*re.estimator().conf_lower().get(j, 0) - expected_conf_lower[j]).abs() < 1e-6,
                "conf_lower[{j}]"
            );
            assert!(
                (*re.estimator().conf_upper().get(j, 0) - expected_conf_upper[j]).abs() < 1e-6,
                "conf_upper[{j}]"
            );
        }
        // `ReCovType::Classical`は`estimator()`委譲
        // でも数値的に正しい（上記アサーション）が、`ReEstimator`自身は常に独自計算
        // した`std_errors()`/`test_stats()`/`p_values()`/`conf_lower()`/`conf_upper()`を
        // 保持する設計にしたため（`fit()`のdocコメント「`ReEstimator`は常に自前の
        // フィールドを保持する」参照）、`panel_classical_cov_params`経由の値が
        // `estimator()`委譲の値と一致する（＝独立した2つの経路が同じ答えを出す）ことも
        // 確認する。
        for j in 0..2 {
            assert!(
                (*re.std_errors().get(j, 0) - expected_std_errors[j]).abs() < 1e-6,
                "re.std_errors[{j}]"
            );
            assert!(
                (*re.test_stats().get(j, 0) - expected_test_stats[j]).abs() < 1e-6,
                "re.test_stats[{j}]"
            );
            assert!(
                (*re.p_values().get(j, 0) - expected_p_values[j]).abs() < 1e-6,
                "re.p_values[{j}]"
            );
            assert!(
                (*re.conf_lower().get(j, 0) - expected_conf_lower[j]).abs() < 1e-6,
                "re.conf_lower[{j}]"
            );
            assert!(
                (*re.conf_upper().get(j, 0) - expected_conf_upper[j]).abs() < 1e-6,
                "re.conf_upper[{j}]"
            );
        }
        assert_eq!(re.cov_type(), &ReCovType::Classical);

        // `loglik = -9.069066112783611`（linearmodels実測）から手計算した参照値。
        assert!((re.estimator().aic() - 22.138_132_225_567_222).abs() < 1e-9);
        assert!((re.estimator().bic() - 22.029_952_523_677_85).abs() < 1e-9);

        // F統計量（2.1節）はWald二次形式のため、傾き係数が1個（q=1）のこのデータでは
        // 「1自由度のF検定は両側t検定と代数的に等価」（`f_statistic = test_stat²`、
        // `f_p_value = p_value`）が成り立つ。`estimator().f_statistic().unwrap()`（定数項も検定に
        // 含めてしまい誤り）とは異なる正しい値であることを確認する。`plm`との数値照合は
        // Python側のテスト（バランスパネルで機械精度一致、不均衡パネルは分散成分の
        // 推定差で許容誤差付き）で行う。
        assert!((re.f_statistic().unwrap() - re.test_stats().get(1, 0).powi(2)).abs() < 1e-9);
        assert!((re.f_p_value().unwrap() - *re.p_values().get(1, 0)).abs() < 1e-9);

        // `linearmodels.RandomEffects.fit(cov_type="unadjusted")`の`rsquared_within`/
        // `rsquared_between`/`rsquared_overall`と数値一致（2.3節）。
        // `rsquared_between`が負値になる（教科書的な入れ子モデル比較の保証が無いR²の
        // 定義のため、`linearmodels`のSST/SSR方式のF統計量と同型の性質）ことも含めて実地検証済み。
        assert!((re.r_squared_within() - 0.793_812_134_496_668).abs() < 1e-9);
        assert!((re.r_squared_between() - (-0.391_981_417_047_268_2)).abs() < 1e-9);
        assert!((re.r_squared_overall() - 0.279_701_906_945_653_1).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_hc1_matches_linearmodels_reference() {
        // `linearmodels.RandomEffects.fit(cov_type="robust").std_errors`と数値一致
        // （`linearmodels`の"robust"は素のHC0ではなく小標本補正
        // `n/df_resid`込みのHC1相当——`extra_df=0`のためこの補正がOLS自身のHC1と
        // 同じ`n/(n-k)`になることを`HeteroskedasticCovariance`のソースで確認済み）。
        let re = ReEstimator::fit(
            cov_type_reference_input(),
            ReEffects::OneWay,
            ReCovType::Hc1,
            0.95,
        )
        .unwrap();

        assert!((*re.std_errors().get(0, 0) - 1.712_718_460_250_846_5).abs() < 1e-9);
        assert!((*re.std_errors().get(1, 0) - 0.209_842_170_109_712_15).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_hc2_hc3_match_plm_reference() {
        // `linearmodels`にはHC2/HC3の実装が無い（`RandomEffects`/`PanelOLS`どちらも
        // `_cov_estimators`が単一の"heteroskedastic"＝HC1相当のみ）ため、Rの
        // `plm::vcovHC(fit, method="white1", type="HC2"/"HC3")`をクロスチェックに使う
        // （`fit()`のdocコメント「`cov_type`対応」参照）。`plm`は
        // 変量効果の分散成分推定法が`linearmodels`と僅かに異なる（点推定自体が僅かに
        // 違う、5.2節のRクロスチェックの一般的な位置づけ）ため、`plm`実測値
        // （`Rscript`で実地確認済み: HC2 std_errors=[1.627395, 0.212174]、
        // HC3=[1.830351, 0.2548547]）とは1e-3〜1e-2の桁で近いことのみ確認する。
        // **1e-9アサーションは「正しさの独立検証」ではなく回帰ガード**:
        // `linearmodels`が返す`params`/`resids`（本ファイルの
        // `re_estimator_fit_matches_linearmodels_reference`と同じ値）から、本実装
        // （`panel_hc_cov_params`）と**同じレバレッジベースHC2/HC3公式**をPythonで
        // 手計算した値であり、独立実装での検証にはならない
        // （`.claude/rules/testing-policy.md`「本実装と同じ計算式の手計算は独立検証
        // としての効力が薄い」通り）。真の独立検証は直後の`plm`比較（1e-2、`plm`は
        // REにHC2/HC3のネイティブ実装を持つ数少ない参照実装）のみ。
        let hc2 = ReEstimator::fit(
            cov_type_reference_input(),
            ReEffects::OneWay,
            ReCovType::Hc2,
            0.95,
        )
        .unwrap();
        assert!((*hc2.std_errors().get(0, 0) - 1.625_640_631_050_364).abs() < 1e-9);
        assert!((*hc2.std_errors().get(1, 0) - 0.212_277_913_827_970_76).abs() < 1e-9);
        // `plm`実測値との近さ（クロスチェック、緩い許容誤差）。
        assert!((*hc2.std_errors().get(0, 0) - 1.627_395).abs() < 1e-2);
        assert!((*hc2.std_errors().get(1, 0) - 0.212_174).abs() < 1e-2);

        let hc3 = ReEstimator::fit(
            cov_type_reference_input(),
            ReEffects::OneWay,
            ReCovType::Hc3,
            0.95,
        )
        .unwrap();
        assert!((*hc3.std_errors().get(0, 0) - 1.828_506_411_875_559_6).abs() < 1e-9);
        assert!((*hc3.std_errors().get(1, 0) - 0.254_973_430_724_023_45).abs() < 1e-9);
        assert!((*hc3.std_errors().get(0, 0) - 1.830_351).abs() < 1e-2);
        assert!((*hc3.std_errors().get(1, 0) - 0.254_854_7).abs() < 1e-2);
    }

    #[test]
    fn re_estimator_fit_cluster_defaults_to_entity_and_matches_fixest_style_correction() {
        // **linearmodels方式からfixest（R）・Stata型に変更**: REは
        // `extra_df`が常に`0`（固定効果ダミーが設計行列に無くFEのようなネスト判定が
        // 不要）なため、`K=df_model`をそのまま使う`(G/(G-1))×((n-1)/(n-K))`補正
        // （`plm::vcovHC(type="sss")`・OLS自身の`cluster_cov_params`と同式であることを
        // 確認済み）になる。旧`linearmodels`方式の値（変更前の期待値）から
        // `sqrt((G/(G-1))×(n-1)/n) = sqrt((3/2)×(6/7)) = sqrt(9/7)`倍した値
        // （`G=n_entities=3`、`n=7`。`K`が新旧で同じ`df_model`のまま変わらないため、
        // 変化するのは`G/(G-1)`補正の追加分のみ、という関係を使って手計算・検算した）。
        let re = ReEstimator::fit(
            cov_type_reference_input(),
            ReEffects::OneWay,
            ReCovType::Cluster { groups: None },
            0.95,
        )
        .unwrap();

        assert!((*re.std_errors().get(0, 0) - 2.137_431_252_437_596_5).abs() < 1e-9);
        assert!((*re.std_errors().get(1, 0) - 0.181_975_972_361_746_9).abs() < 1e-9);
        // `df_inference()`/`stat_dist()`（カバレッジ監査で判明した未検証の単純
        // getter、新設）。`n=7`・`df_model=2`（`n_entities=3`個の
        // クラスターがある既定設定）で`df_resid=5`だが`df_inference=G-1=2`
        // （両者が乖離することを確認するため、意図的に`cov_type=Classical`ではなく
        // このテストで検証する）。
        assert_eq!(re.df_resid(), 5);
        assert_eq!(re.df_inference(), 2);
        assert_eq!(re.stat_dist(), inference::StatDist::T { df: 2 });
    }

    #[test]
    fn re_estimator_dk_bandwidth_used_reflects_resolved_bandwidth() {
        // `t_periods=3`の既定バンド幅は`floor(4*(3/100)^(2/9))=1`。明示指定はその値、
        // `Dk`以外は`None`。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let time = strings(&["1", "2", "3", "1", "2", "1", "2"]);
        let dk = |bandwidth: Option<i64>| ReCovType::Dk {
            bandwidth,
            time: TimeKeys::lexicographic(time.clone()),
        };
        let bandwidth_used = |cov_type: ReCovType| {
            let input = ReInput::from_columns(
                &[3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0],
                &[vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0]],
                vec!["x1".to_string()],
                &entity,
                Some(&time),
                "y".into(),
            )
            .unwrap();
            ReEstimator::fit(input, ReEffects::OneWay, cov_type, 0.95)
                .unwrap()
                .dk_bandwidth_used()
        };

        assert_eq!(bandwidth_used(dk(None)), Some(1));
        assert_eq!(bandwidth_used(dk(Some(0))), Some(0));
        assert_eq!(bandwidth_used(dk(Some(2))), Some(2));
        assert_eq!(bandwidth_used(ReCovType::Hc1), None);
    }

    #[test]
    fn re_estimator_fit_hac_matches_fixest_style_correction() {
        // **linearmodels方式からfixest（R）型に変更**: FEのDKと同じく
        // `K=df_model`・`G`相当は`t_periods`を使う`(t_periods/(t_periods-1))×
        // ((n-1)/(n-K))`補正になる。旧`linearmodels`方式の値（変更前の期待値）から
        // `sqrt((t_periods/(t_periods-1))×(n-1)/n) = sqrt((3/2)×(6/7)) = sqrt(9/7)`
        // 倍した値（`t_periods=3`、`n=7`。`K`が新旧で同じ`df_model`のまま変わらない
        // ため、`bandwidth=0`・`1`（このデータのラグ項ループは1回以下）は`panel_
        // driscoll_kraay_cov_params`の生のサンドイッチ行列自体は変更前と同じで、
        // 変化するのは最終スケールのみという関係を使って手計算・検算した——
        // `fe_estimator_fit_one_way_hac_with_bandwidth_two_scales_unchanged_kernel_
        // by_new_correction`で確認した`bandwidth>=2`のfixestとの不一致はこの
        // テストの範囲外、`bandwidth<=1`はfixestの生カーネルと一致確認済み）。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let time = strings(&["1", "2", "3", "1", "2", "1", "2"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input = ReInput::from_columns(
            &y,
            &[x1],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let bw0 = ReEstimator::fit(
            input,
            ReEffects::OneWay,
            ReCovType::Dk {
                bandwidth: Some(0),
                time: TimeKeys::lexicographic(time.clone()),
            },
            0.95,
        )
        .unwrap();
        assert!((*bw0.std_errors().get(0, 0) - 0.077_007_356_453_778_4).abs() < 1e-9);
        assert!((*bw0.std_errors().get(1, 0) - 0.306_138_209_227_064_75).abs() < 1e-9);

        let input2 = ReInput::from_columns(
            &y,
            &[vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0]],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();
        let bw1 = ReEstimator::fit(
            input2,
            ReEffects::OneWay,
            ReCovType::Dk {
                bandwidth: Some(1),
                time: TimeKeys::lexicographic(time.clone()),
            },
            0.95,
        )
        .unwrap();
        assert!((*bw1.std_errors().get(0, 0) - 0.073_386_231_305_208_44).abs() < 1e-9);
        assert!((*bw1.std_errors().get(1, 0) - 0.191_848_292_228_156_4).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_f_statistic_follows_cov_type_and_matches_squared_t_statistic() {
        // F統計量（2.1節）は`cov_type`別に計算し直した`cov_params`・`df_inference`の
        // Wald検定のため、傾き係数が1個（q=1）のこのデータでは「1自由度のF検定は
        // 両側t検定と代数的に等価」（`f_statistic = test_stat²`、`f_p_value = p_value`）が
        // 全`cov_type`で成り立つ。`cov_type`別の`cov_params`・分母自由度が正しく
        // `wald_f_test`に渡っているかの回帰ガード（FEの同名の恒等式チェックと同じ）。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let time = strings(&["1", "2", "3", "1", "2", "1", "2"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input = || {
            ReInput::from_columns(
                &y,
                std::slice::from_ref(&x1),
                vec!["x1".to_string()],
                &entity,
                Some(&time),
                "y".into(),
            )
            .unwrap()
        };

        // 各`cov_type`の期待分母自由度: Clusterは`G-1=2`、Dkは`t_periods-1=2`、
        // それ以外は`df_resid = n - df_model = 5`。
        let cases = [
            (ReCovType::Classical, 5),
            (ReCovType::Hc1, 5),
            (ReCovType::Hc2, 5),
            (ReCovType::Hc3, 5),
            (ReCovType::Cluster { groups: None }, 2),
            (
                ReCovType::Dk {
                    bandwidth: Some(1),
                    time: TimeKeys::lexicographic(time.clone()),
                },
                2,
            ),
        ];
        for (cov_type, expected_df_denom) in cases {
            let label = format!("{cov_type:?}");
            let re = ReEstimator::fit(input(), ReEffects::OneWay, cov_type, 0.95).unwrap();
            let t = *re.test_stats().get(1, 0);
            assert!((re.f_statistic().unwrap() - t * t).abs() < 1e-9, "{label}");
            assert!(
                (re.f_p_value().unwrap() - *re.p_values().get(1, 0)).abs() < 1e-9,
                "{label}"
            );
            assert_eq!(re.f_df(), Some((1, expected_df_denom)), "{label}");
            assert_eq!(re.df_resid(), 5, "{label}");
        }
    }

    #[test]
    fn re_estimator_fit_dk_rejects_time_of_wrong_length() {
        // `ReCovType::Dk.time`は`ReInput`の検証を通らない公開引数のため、`fit()`が長さを
        // 検証する（`FeEstimator::fit`のDkアームと同じ）。
        let re = ReEstimator::fit(
            cov_type_reference_input(),
            ReEffects::OneWay,
            ReCovType::Dk {
                bandwidth: None,
                time: TimeKeys::lexicographic(strings(&["1", "2", "3"])),
            },
            0.95,
        );

        assert_eq!(
            re.unwrap_err(),
            PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                y_rows: 7,
                other_rows: 3,
            }
        );
    }

    #[test]
    fn re_estimator_fit_dk_follows_the_value_order_of_integer_periods() {
        // 4エンティティ×12時点。時点を整数の値の順序で並べた結果は、ゼロ埋めして辞書順が
        // 時間順になるラベルでの結果と一致し、ゼロ埋めなしの辞書順（`1, 10, 11, 2, ...`）とは
        // 異なる。
        let (n_entities, n_periods) = (4usize, 12usize);
        let mut shock = vec![0.0; n_periods];
        for t in 1..n_periods {
            shock[t] = 0.8 * shock[t - 1] + (((t * 37) % 11) as f64 - 5.0) / 5.0;
        }
        let (mut y, mut x, mut entity, mut period) = (vec![], vec![], vec![], vec![]);
        for e in 0..n_entities {
            for (t, &shock_t) in shock.iter().enumerate() {
                let xi = ((e * 7 + t * 13) % 17) as f64 / 3.0;
                let noise = ((e * 5 + t * 3) % 7) as f64 / 7.0 + 0.3 * e as f64;
                y.push(1.0 + 0.5 * xi + shock_t + noise);
                x.push(xi);
                entity.push(format!("e{e}"));
                period.push(t);
            }
        }
        let fit = |time: TimeKeys| {
            let input = ReInput::from_columns_ordered(
                &y,
                &[x.clone()],
                vec!["x".into()],
                &entity,
                Some(time.clone()),
                "y".into(),
            )
            .unwrap();
            let re = ReEstimator::fit(
                input,
                ReEffects::OneWay,
                ReCovType::Dk {
                    bandwidth: Some(3),
                    time,
                },
                0.95,
            )
            .unwrap();
            *re.std_errors().get(1, 0)
        };
        let values: Vec<i128> = period.iter().map(|&t| t as i128).collect();
        let by_value = |label: &dyn Fn(usize) -> String| {
            TimeKeys::by_integer(period.iter().map(|&t| label(t)).collect(), &values).unwrap()
        };

        let expected = fit(by_value(&|t| format!("{t:03}")));
        let numeric = fit(by_value(&|t| t.to_string()));
        let lexicographic = fit(TimeKeys::lexicographic(
            period.iter().map(|t| t.to_string()).collect(),
        ));

        assert!(
            (numeric - expected).abs() < 1e-12,
            "numeric {numeric} vs padded {expected}"
        );
        assert!(
            (lexicographic - expected).abs() > 1e-6,
            "the lexicographic order of unpadded labels must differ: {lexicographic} vs {expected}"
        );
    }

    #[test]
    fn re_estimator_fit_hac_rejects_bandwidth_at_least_t_periods() {
        // `fit()`のdocコメントに明記した`PanelError::InvalidDkBandwidth`の伝播経路を固定する
        // （`resolve_dk_bandwidth`自体はFE側のテストでカバー済みだが、RE経由の配線は
        // 別途確認する。`fe_estimator_fit_hac_rejects_bandwidth_at_least_t_periods`と
        // 同型）。t_periods=3（time∈{1,2,3}）に対しbandwidth=3（`>=t`）を指定する。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let time = strings(&["1", "2", "3", "1", "2", "1", "2"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input = ReInput::from_columns(
            &y,
            &[x1],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let result = ReEstimator::fit(
            input,
            ReEffects::OneWay,
            ReCovType::Dk {
                bandwidth: Some(3),
                time: TimeKeys::lexicographic(time.clone()),
            },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::InvalidDkBandwidth { bandwidth: 3, t: 3 }
        );
    }

    #[test]
    fn re_estimator_fit_cluster_supports_explicit_groups_column() {
        // `groups`に`entity`以外の任意の列を明示指定できることを確認する
        // （3.2節「`cluster`を明示指定すれば任意の列でもクラスター可能」）。
        // ここでは`entity`をそのまま複製した列を明示的に渡し、`groups: None`
        // （`re_estimator_fit_cluster_defaults_to_entity_and_matches_fixest_style_
        // correction`）と同じ結果になることを確認する。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();
        let explicit_groups = strings(&["a", "a", "a", "b", "b", "c", "c"]);

        let re = ReEstimator::fit(
            input,
            ReEffects::OneWay,
            ReCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&explicit_groups)),
            },
            0.95,
        )
        .unwrap();

        assert!((*re.std_errors().get(0, 0) - 2.137_431_252_437_596_5).abs() < 1e-9);
        assert!((*re.std_errors().get(1, 0) - 0.181_975_972_361_746_9).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_cluster_returns_error_when_cluster_count_at_most_slopes() {
        // クラスター数`G`が傾き係数の数`q`（`df_model - 1`）以下だと構造的に特異になる
        // （OLS/FE共通の制約。REもここに合わせる）。ここではq=1（傾き1個）
        // に対しG=1（全観測が同一クラスター）にして発火させる。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let all_same_cluster = strings(&["g0", "g0", "g0", "g0", "g0", "g0", "g0"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = ReEstimator::fit(
            input,
            ReEffects::OneWay,
            ReCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&all_same_cluster)),
            },
            0.95,
        );

        assert!(matches!(
            result,
            Err(PanelError::Common(CommonError::InsufficientClusters { .. }))
        ));
    }

    #[test]
    fn re_estimator_fit_f_statistic_is_wald_form_for_extremely_unbalanced_panel() {
        // `linearmodels`のSST/SSR方式のF統計量はこのデータ（極端に不均衡なパネル）で
        // -0.6837と負値になる。本実装のWald二次形式（`re.rs`のモジュールdoc「自由度・標準誤差・F統計量」）は
        // 傾き係数が1個のとき`f_statistic = test_stat²`（1自由度のF検定と両側t検定の
        // 代数的等価）になる。旧SST/SSR実装に戻すと負値になりこの恒等式が崩れるため、
        // その回帰ガードとして恒等式を固定する（`plm`との数値照合は分散成分の推定差が
        // 小標本で大きく効くためPython側の許容誤差付きテストで行う）。
        // T_i={2, 2, 15}という極端に不均衡なパネル（`linearmodels`でのランダム探索で
        // 発見、乱数シード固定・実地検証済み）。
        let entity = strings(&[
            "e0", "e0", "e1", "e1", "e2", "e2", "e2", "e2", "e2", "e2", "e2", "e2", "e2", "e2",
            "e2", "e2", "e2", "e2", "e2",
        ]);
        let x1 = vec![
            1.206_726_463,
            0.859_309_915_6,
            0.412_096_626_7,
            0.969_400_630_3,
            5.333_483_877_8,
            -2.910_149_551_1,
            -5.141_873_834_4,
            -4.522_814_339,
            -0.050_290_856_2,
            4.133_576_993_4,
            4.899_532_633_5,
            0.889_588_287_9,
            -4.997_800_337_1,
            -2.746_322_280_7,
            -3.757_660_566_6,
            -0.740_528_097_1,
            -7.434_903_680_5,
            0.472_973_696_2,
            2.712_179_870_5,
        ];
        let y = [
            -1.628_726_680_2,
            -1.987_971_224_5,
            -9.720_496_074_8,
            -5.349_549_902_4,
            15.454_243_372_1,
            11.482_129_011_1,
            12.946_870_275_7,
            17.902_202_464_2,
            18.976_838_217_0,
            16.352_866_361_3,
            21.000_419_954_5,
            15.678_612_541_9,
            18.693_076_035_8,
            16.672_988_029_6,
            13.874_059_095_9,
            18.135_545_024_4,
            11.856_646_694_1,
            13.925_593_855_6,
            13.265_351_905_9,
        ];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

        assert_eq!(re.df_resid(), 17);
        assert_eq!(re.df_model(), 2);
        let t = *re.test_stats().get(1, 0);
        assert!((re.f_statistic().unwrap() - t * t).abs() < 1e-9 * (1.0 + t * t));
        let p_value = *re.p_values().get(1, 0);
        assert!((re.f_p_value().unwrap() - p_value).abs() < 1e-9 * p_value.max(1e-300));
        assert!(re.f_p_value().unwrap() > 0.0 && re.f_p_value().unwrap() < 1.0);
    }

    #[test]
    fn re_estimator_fit_df_resid_and_df_model_with_no_slope_regressors() {
        // 説明変数0個（定数項のみ、k=1）のモデルでも`df_resid`/`df_model`が正しいことを
        // 確認する（`re_estimator_fit_matches_linearmodels_reference`とは別のn/kの組み合わせ
        // で検証、`linearmodels.RandomEffects(y, const).fit(cov_type="unadjusted")`で
        // 実地検証済み: df_resid=5, df_model=1, params=[4.666...]）。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let y = [3.0, 5.0, 2.0, 4.0, 6.0, 8.0];
        let input = ReInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

        assert_eq!(re.df_resid(), 5);
        assert_eq!(re.df_model(), 1);
        assert!((*re.estimator().params().get(0, 0) - 4.666_666_666_666_667).abs() < 1e-9);

        // 傾き係数0個（定数項のみ）のモデルはOLS/FE同様`None`（2.1節）。
        assert!(re.f_statistic().is_none());
        assert!(re.f_p_value().is_none());

        // 傾き係数0個（`df_model==1`）ならパネル固有R²は3種とも0.0（2.3節）。
        // `linearmodels`の`_rsquared`早期リターンと数値一致（実地検証済み）。
        assert_eq!(re.r_squared_within(), 0.0);
        assert_eq!(re.r_squared_between(), 0.0);
        assert_eq!(re.r_squared_overall(), 0.0);
    }

    #[test]
    fn re_estimator_fit_propagates_fe_singleton_error_from_variance_components() {
        // entity "c"は1観測のみ（singleton）。`swamy_arora_variance_components`内部の
        // FE推定が`PanelError::SingletonGroup`を返し、そのまま伝播することを確認する
        // （`swamy_arora_variance_components_propagates_fe_singleton_error`と同じ配線）。
        let entity = strings(&["a", "a", "c"]);
        let x1 = vec![1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 3.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                column: "entity".to_string(),
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn re_estimator_fit_propagates_invalid_confidence_level_error() {
        // `confidence_level`の範囲チェック（`(0, 1)`の範囲外）は、
        // `swamy_arora_variance_components`内部の最初の委譲（`FeEstimator::fit`）が
        // 真っ先に検証するため、`QuasiDemeanedRegressionFailed`ではなく
        // `WithinRegressionFailed`として伝播する（`FeEstimator::fit`自身の
        // `fe_estimator_fit_propagates_invalid_confidence_level_error`と同型）。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 1.5);

        assert_eq!(
            result.unwrap_err(),
            PanelError::WithinRegressionFailed {
                source: crate::linear::common::LeastSquaresError::Common(
                    CommonError::InvalidConfidenceLevel {
                        confidence_level: 1.5,
                    }
                ),
            }
        );
    }

    #[test]
    fn re_estimator_fit_pins_faer_global_parallelism_to_seq() {
        // `fit()`冒頭の`crate::shared::parallelism::ensure_serial()`がfaerの
        // グローバル並列度を`Par::Seq`へ引き戻すことの回帰ガード
        // （`fe_estimator_fit_pins_faer_global_parallelism_to_seq`と同型、
        // `engine/src/panel/CLAUDE.md`「faerのグローバル並列度」参照）。
        faer::set_global_parallelism(faer::Par::rayon(0));

        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let _ = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

        assert!(matches!(faer::get_global_parallelism(), faer::Par::Seq));
    }

    #[test]
    fn re_estimator_fit_returns_perfect_fit_when_the_within_regression_is_exact() {
        // ノイズ無しの線形DGP（`y = 2*x1 + entity固有の切片`）では、分散成分を求める内部FEの
        // within回帰の残差が全て0になり、σ_ε²=0（θ=1で最終回帰の定数列が全て0になる）に
        // なる。最終回帰の特異性（`QuasiDemeanedRegressionFailed`）に落ちる前に、原因の分かる
        // 完全フィットとして、分散成分のwithin回帰の文脈つきで止まる。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0];
        let y = [7.0, 9.0, 11.0, 14.0, 16.0, 9.0, 11.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95);

        let err = result.unwrap_err();
        assert_eq!(
            err,
            PanelError::WithinRegressionFailed {
                source: crate::shared::error::CommonError::PerfectFit.into(),
            }
            .in_stage(Stage::VarianceComponentWithinRegression)
        );
        assert_eq!(
            err.report().code(),
            crate::shared::error_catalog::ErrorCode::PerfectFit
        );
    }

    #[test]
    fn re_estimator_fit_passes_outcome_constant_within_entity_through_without_a_stage() {
        // 個体ごとに定数のyは、分散成分のwithin回帰が説明するものを持たない。入力の事実なので
        // 文脈句は付けない。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0];
        let y = [0.1, 0.1, 0.1, 0.7, 0.7, 1.3, 1.3];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95);

        assert_eq!(result.unwrap_err(), PanelError::OutcomeConstantWithinEntity);
    }

    #[test]
    fn re_estimator_fit_f_statistic_with_two_slopes_matches_wald_test_last_columns_for_classical() {
        // 傾き係数が2個（q=2）のとき、`cov_params`の部分行列の非対角要素と
        // `k_constant + j`の列オフセットが効く（`f = t²`の恒等式が使えるq=1では
        // 検証できない）。`Classical`では`estimator()`（準偏差変換済みデータへの
        // `CovType::Classical`のOLS）の`cov_params`がRE自身の`cov_params`と一致する
        // ため、`OlsEstimator::wald_test_last_columns(q)`（末尾`q`列＝傾き係数の
        // 同時Wald検定、`fit()`が使う経路とは別の呼び出し）と一致するはず。
        let input = hausman_two_slope_input(None);
        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
        assert_eq!(re.df_model(), 3);

        let (expected_f, expected_p) = re.estimator().wald_test_last_columns(2).unwrap();
        assert!((re.f_statistic().unwrap() - expected_f).abs() < 1e-9 * expected_f.abs());
        assert!((re.f_p_value().unwrap() - expected_p).abs() < 1e-9);
        // 非対角要素を無視した（独立な2つのt²の平均）値とは異なる（q=2の部分行列が
        // 実際に使われていることの確認）。
        let t1 = *re.test_stats().get(1, 0);
        let t2 = *re.test_stats().get(2, 0);
        assert!((re.f_statistic().unwrap() - (t1 * t1 + t2 * t2) / 2.0).abs() > 1e-6);

        assert_eq!(re.f_df(), Some((2, re.df_inference())));
        assert_eq!(re.df_inference(), re.df_resid());
    }

    /// property-basedテスト。固定シナリオでは`σ_u²`・`T_i`・`cov_type`の組み合わせが
    /// 限られるため、ランダムな不均衡パネルでREの不変条件（θの値域・`σ_u²=0`でのpooled OLS
    /// への退化・行順序/ラベルへの不変性）を検証する。
    mod proptests {
        use super::*;
        use proptest::collection;
        use proptest::prelude::*;

        const MAX_T: usize = 6;

        #[derive(Debug, Clone)]
        struct ReCase {
            entity_idx: Vec<usize>,
            n_entities: usize,
            y: Vec<f64>,
            x: Vec<Vec<f64>>,
            /// 行の並べ替え用の乱数キー（長さ`n`）。
            keys: Vec<u64>,
        }

        /// 不均衡パネル（entityごとに`T_i`が2..=6）。`has_effect=false`の場合はentity効果を
        /// 持たないDGPになり、`σ_u²`が0に切り詰められる（Swamy-Aroraの`max(0)`）ケースが
        /// 高頻度で現れる。
        fn re_case_strategy() -> impl Strategy<Value = ReCase> {
            (6..=9usize, 1..=2usize, any::<bool>())
                .prop_flat_map(|(n_entities, k, has_effect)| {
                    (
                        Just(n_entities),
                        Just(k),
                        Just(has_effect),
                        collection::vec(2..=MAX_T, n_entities),
                    )
                })
                .prop_flat_map(|(n_entities, k, has_effect, sizes)| {
                    let n: usize = sizes.iter().sum();
                    (
                        Just(n_entities),
                        Just(has_effect),
                        Just(sizes),
                        collection::vec(collection::vec(-10.0f64..10.0, n), k),
                        collection::vec(-5.0f64..5.0, n),
                        collection::vec(-20.0f64..20.0, n_entities),
                        collection::vec(any::<u64>(), n),
                    )
                })
                .prop_map(|(n_entities, has_effect, sizes, x, noise, effect, keys)| {
                    let mut entity_idx = Vec::new();
                    for (i, &size) in sizes.iter().enumerate() {
                        entity_idx.extend(std::iter::repeat_n(i, size));
                    }
                    let y: Vec<f64> = (0..noise.len())
                        .map(|r| {
                            let u = if has_effect {
                                effect[entity_idx[r]]
                            } else {
                                0.0
                            };
                            x.iter().map(|c| c[r]).sum::<f64>() + u + noise[r]
                        })
                        .collect();
                    ReCase {
                        entity_idx,
                        n_entities,
                        y,
                        x,
                        keys,
                    }
                })
        }

        /// バランスパネル（`n_entities`×`n_periods`、行はentity優先の並び）。
        /// `(n_entities, n_periods, y, x, 並べ替え用の乱数キー)`。
        type TwoWayCase = (usize, usize, Vec<f64>, Vec<Vec<f64>>, Vec<u64>);

        fn two_way_case_strategy() -> impl Strategy<Value = TwoWayCase> {
            (4..=7usize, 3..=5usize, 1..=2usize)
                .prop_flat_map(|(n_entities, n_periods, k)| {
                    let n = n_entities * n_periods;
                    (
                        Just(n_entities),
                        Just(n_periods),
                        collection::vec(-5.0f64..5.0, n),
                        collection::vec(collection::vec(-10.0f64..10.0, n), k),
                        collection::vec(any::<u64>(), n),
                    )
                })
                .prop_map(|(n_entities, n_periods, noise, x, keys)| {
                    let y: Vec<f64> = (0..noise.len())
                        .map(|r| {
                            let effect = (r / n_periods) as f64 - (r % n_periods) as f64;
                            x.iter().map(|c| c[r]).sum::<f64>() + effect + noise[r]
                        })
                        .collect();
                    (n_entities, n_periods, y, x, keys)
                })
        }

        fn re_cov_strategy() -> impl Strategy<Value = ReCovType> {
            prop_oneof![
                Just(ReCovType::Classical),
                Just(ReCovType::Hc1),
                Just(ReCovType::Hc2),
                Just(ReCovType::Hc3),
                Just(ReCovType::Cluster { groups: None }),
            ]
        }

        fn entity_labels(idx: &[usize]) -> Vec<String> {
            idx.iter().map(|i| format!("e{i}")).collect()
        }

        fn re_input(y: &[f64], x: &[Vec<f64>], entity: &[String]) -> ReInput {
            let names: Vec<String> = (0..x.len()).map(|j| format!("x{j}")).collect();
            ReInput::from_columns(y, x, names, entity, None, "y".to_string()).unwrap()
        }

        /// 固定フィクスチャ比較より緩めた相対誤差（`ols/fit.rs`/`fe/fit.rs`のproptestと同じ方針）。
        fn assert_approx_eq(actual: f64, expected: f64, msg: &str) {
            let tol = 1e-6 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tol,
                "{msg}: actual={actual}, expected={expected}, tol={tol}"
            );
        }

        /// 係数（先頭が切片）と標準誤差を並べて取り出す。
        fn params_and_se(est: &ReEstimator) -> (Vec<f64>, Vec<f64>) {
            let n = est.estimator().params().nrows();
            let params = (0..n)
                .map(|j| *est.estimator().params().get(j, 0))
                .collect();
            let se = (0..n).map(|j| *est.std_errors().get(j, 0)).collect();
            (params, se)
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            /// 説明変数の個体平均の列を加えても（相関ランダム効果、Mundlak型）、between回帰ではその列が
            /// 元の列と同じ列になって落ちるため、分散成分は変わらず、最終回帰・ハウスマン検定も
            /// 失敗しない（ハウスマン検定の自由度は別名になった`x̃`の分だけ減る）。
            #[test]
            fn adding_an_entity_mean_column_leaves_the_variance_components_unchanged(
                case in re_case_strategy(),
            ) {
                let entity = entity_labels(&case.entity_idx);
                let codes = GroupCodes::from_labels(&entity);
                let base = re_input(&case.y, &case.x, &entity);
                let mean = entity_means(base.y(), base.x(), &codes);
                let bar: Vec<f64> = (0..base.nobs())
                    .map(|i| mean.x[0][codes.codes()[i]])
                    .collect();
                let mut x = case.x.clone();
                x.push(bar);
                let extended = re_input(&case.y, &x, &entity);

                let plain = swamy_arora_variance_components(&base, ReEffects::OneWay, 0.95);
                prop_assume!(plain.is_ok());
                let plain = plain.unwrap();
                let with_bar = swamy_arora_variance_components(&extended, ReEffects::OneWay, 0.95)
                    .unwrap();
                assert_approx_eq(with_bar.sigma2_eps, plain.sigma2_eps, "sigma2_eps");
                assert_approx_eq(with_bar.sigma2_u, plain.sigma2_u, "sigma2_u");

                let fit = ReEstimator::fit(extended, ReEffects::OneWay, ReCovType::Classical, 0.95);
                prop_assume!(plain.sigma2_eps > 0.0);
                let fit = fit.unwrap();
                let base_fit = ReEstimator::fit(base, ReEffects::OneWay, ReCovType::Classical, 0.95)
                    .unwrap();
                prop_assert_eq!(
                    fit.hausman_df(),
                    base_fit.hausman_df().and_then(|df| df.checked_sub(1)).filter(|&df| df > 0)
                );
            }

            /// Swamy-Arora分散成分から実際に組み立てたθも`[0, 1)`に収まる。
            #[test]
            fn fitted_theta_is_within_unit_interval(case in re_case_strategy()) {
                let entity = entity_labels(&case.entity_idx);
                let input = re_input(&case.y, &case.x, &entity);
                let vc = swamy_arora_variance_components(&input,ReEffects::OneWay, 0.95);
                prop_assume!(vc.is_ok());
                let VarianceComponents {
                    sigma2_eps,
                    sigma2_u,
                    ..
                } = vc.unwrap();
                prop_assume!(sigma2_eps > 0.0);

                let (theta, _, _) = quasi_demean_transform(&input, sigma2_eps, sigma2_u);

                for (id, t) in theta.iter().enumerate() {
                    prop_assert!((0.0..1.0).contains(t), "theta[{id}]={t}");
                }
            }

            /// `σ_u²`が0に切り詰められたときREはpooled OLS（定数項あり）と一致する
            /// （`θ_i=0`で準偏差変換が恒等になるため、係数もClassical SEも同じ）。
            #[test]
            fn matches_pooled_ols_when_sigma2_u_is_zero(case in re_case_strategy()) {
                let entity = entity_labels(&case.entity_idx);
                let vc = swamy_arora_variance_components(&re_input(&case.y, &case.x, &entity),ReEffects::OneWay, 0.95);
                prop_assume!(vc.is_ok());
                prop_assume!(vc.unwrap().sigma2_u == 0.0);

                let re = ReEstimator::fit(
                    re_input(&case.y, &case.x, &entity),
                    ReEffects::OneWay,
                    ReCovType::Classical,
                    0.95,);
                prop_assume!(re.is_ok());
                let (params, se) = params_and_se(&re.unwrap());

                let names: Vec<String> = (0..case.x.len()).map(|j| format!("x{j}")).collect();
                let ols_input =
                    OlsInput::from_columns(&case.y, &case.x, names, true, "y".to_string()).unwrap();
                let ols = OlsEstimator::fit(ols_input, CovType::Classical, 0.95);
                prop_assume!(ols.is_ok());
                let ols = ols.unwrap();

                for j in 0..params.len() {
                    assert_approx_eq(params[j], *ols.params().get(j, 0), &format!("param[{j}]"));
                    assert_approx_eq(se[j], *ols.std_errors().get(j, 0), &format!("se[{j}]"));
                }
            }

            /// 行の並べ替えとentityラベルの付け替えで、係数・標準誤差・ハウスマン統計量は
            /// 変わらない。
            #[test]
            fn results_are_invariant_to_row_order_and_entity_relabeling(
                case in re_case_strategy(),
                cov in re_cov_strategy(),
            ) {
                let entity = entity_labels(&case.entity_idx);
                let base = ReEstimator::fit(re_input(&case.y, &case.x, &entity),ReEffects::OneWay, cov.clone(), 0.95);
                prop_assume!(base.is_ok());
                let base = base.unwrap();
                let (params, se) = params_and_se(&base);

                let n = case.y.len();
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by_key(|&r| case.keys[r]);
                let y: Vec<f64> = order.iter().map(|&r| case.y[r]).collect();
                let x: Vec<Vec<f64>> = case
                    .x
                    .iter()
                    .map(|c| order.iter().map(|&r| c[r]).collect())
                    .collect();
                let entity2: Vec<String> = order
                    .iter()
                    .map(|&r| format!("z{}", case.n_entities - 1 - case.entity_idx[r]))
                    .collect();

                let permuted = ReEstimator::fit(re_input(&y, &x, &entity2),ReEffects::OneWay, cov, 0.95);
                prop_assume!(permuted.is_ok());
                let permuted = permuted.unwrap();
                let (params2, se2) = params_and_se(&permuted);

                for j in 0..params.len() {
                    assert_approx_eq(params2[j], params[j], &format!("param[{j}]"));
                    assert_approx_eq(se2[j], se[j], &format!("se[{j}]"));
                }
                match (base.hausman_statistic(), permuted.hausman_statistic()) {
                    (Some(a), Some(b)) => assert_approx_eq(b, a, "hausman_statistic"),
                    (None, None) => {}
                    (a, b) => prop_assert!(false, "hausman presence differs: {a:?} vs {b:?}"),
                }
            }

            /// 2-wayのバランスパネル: 行の並べ替えとエンティティ・時点のラベルの付け替えで
            /// 係数・標準誤差が変わらず、`y`を`c`倍すると係数・標準誤差が`|c|`倍になる
            /// （`σ²`は`c²`倍、θは不変のため）。
            #[test]
            fn two_way_results_are_invariant_to_relabeling_and_equivariant_to_y_scale(
                case in two_way_case_strategy(),
                scale in 0.5f64..4.0,
            ) {
                let (n_entities, n_periods, y, x, keys) = case;
                let n = y.len();
                let entity: Vec<String> = (0..n).map(|r| format!("e{}", r / n_periods)).collect();
                let time: Vec<String> = (0..n).map(|r| format!("t{}", r % n_periods)).collect();
                let build = |y: &[f64], x: &[Vec<f64>], entity: &[String], time: &[String]| {
                    let names: Vec<String> = (0..x.len()).map(|j| format!("x{j}")).collect();
                    ReInput::from_columns(y, x, names, entity, Some(time), "y".to_string())
                        .and_then(|input| {
                            ReEstimator::fit(input, ReEffects::TwoWay, ReCovType::Classical, 0.95)
                        })
                };
                let base = build(&y, &x, &entity, &time);
                prop_assume!(base.is_ok());
                let (params, se) = params_and_se(&base.unwrap());

                // 行の並べ替え＋エンティティ・時点ラベルの付け替え。
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by_key(|&r| keys[r]);
                let y2: Vec<f64> = order.iter().map(|&r| y[r]).collect();
                let x2: Vec<Vec<f64>> =
                    x.iter().map(|c| order.iter().map(|&r| c[r]).collect()).collect();
                let entity2: Vec<String> = order
                    .iter()
                    .map(|&r| format!("z{}", n_entities - 1 - r / n_periods))
                    .collect();
                let time2: Vec<String> = order
                    .iter()
                    .map(|&r| format!("u{}", n_periods - 1 - r % n_periods))
                    .collect();
                let permuted = build(&y2, &x2, &entity2, &time2);
                prop_assume!(permuted.is_ok());
                let (params2, se2) = params_and_se(&permuted.unwrap());
                for j in 0..params.len() {
                    assert_approx_eq(params2[j], params[j], &format!("param[{j}]"));
                    assert_approx_eq(se2[j], se[j], &format!("se[{j}]"));
                }

                // `y`のスケール同変性。
                let y3: Vec<f64> = y.iter().map(|v| v * scale).collect();
                let scaled = build(&y3, &x, &entity, &time);
                prop_assume!(scaled.is_ok());
                let (params3, se3) = params_and_se(&scaled.unwrap());
                for j in 0..params.len() {
                    assert_approx_eq(params3[j], params[j] * scale, &format!("scaled param[{j}]"));
                    assert_approx_eq(se3[j], se[j] * scale, &format!("scaled se[{j}]"));
                }
            }

            /// エンティティ内で一定の列`z`（時間不変）を足しても、内部FEの`σ_ε²`は変わらない
            /// （within変動のある列だけがσ_ε²を決める）。`time_varying`は元の列を指したまま。
            #[test]
            fn time_invariant_column_leaves_sigma2_eps_unchanged(case in re_case_strategy()) {
                let entity = entity_labels(&case.entity_idx);
                let base = swamy_arora_variance_components(
                    &re_input(&case.y, &case.x, &entity),
                    ReEffects::OneWay,
                    0.95,);
                prop_assume!(base.is_ok());
                let base = base.unwrap();

                let z: Vec<f64> = case
                    .entity_idx
                    .iter()
                    .map(|&i| ((i * 37) % 11) as f64 - 5.0)
                    .collect();
                let mut x = vec![z];
                x.extend(case.x.iter().cloned());
                let with_z = swamy_arora_variance_components(&re_input(&case.y, &x, &entity),ReEffects::OneWay, 0.95);
                prop_assume!(with_z.is_ok());
                let with_z = with_z.unwrap();

                prop_assert_eq!(with_z.sigma2_eps, base.sigma2_eps);
                let shifted: Vec<usize> = base.time_varying.iter().map(|j| j + 1).collect();
                prop_assert_eq!(with_z.time_varying, shifted);
            }

            /// 時間不変列`z`に定数`c`を足しても、`z`と他の列の係数は変わらず、
            /// 切片だけが`-β_z·c`動く（定数は切片に吸収される）。
            #[test]
            fn shifting_time_invariant_column_only_moves_the_intercept(
                case in re_case_strategy(),
                c in -5.0f64..5.0,
            ) {
                let entity = entity_labels(&case.entity_idx);
                let z: Vec<f64> = case
                    .entity_idx
                    .iter()
                    .map(|&i| ((i * 37) % 11) as f64 - 5.0)
                    .collect();
                let build = |z: Vec<f64>| {
                    let mut x = vec![z];
                    x.extend(case.x.iter().cloned());
                    ReEstimator::fit(re_input(&case.y, &x, &entity),ReEffects::OneWay, ReCovType::Classical, 0.95)
                };
                let base = build(z.clone());
                prop_assume!(base.is_ok());
                let shifted = build(z.iter().map(|v| v + c).collect());
                prop_assume!(shifted.is_ok());
                let (base, shifted) = (base.unwrap(), shifted.unwrap());
                let (p0, se0) = params_and_se(&base);
                let (p1, se1) = params_and_se(&shifted);

                assert_approx_eq(p1[0], p0[0] - p0[1] * c, "const");
                for j in 1..p0.len() {
                    assert_approx_eq(p1[j], p0[j], &format!("param[{j}]"));
                    assert_approx_eq(se1[j], se0[j], &format!("se[{j}]"));
                }
            }
        }

        /// 一次元でしか変わらない列（`xi`: 個体間のみ、`xt`: 時点間のみ）を含む2-wayのバランス
        /// パネル。`has_entity_effect`・`has_time_effect`が偽のDGPでは、対応する分散成分が0に
        /// 切り詰められるケースが高頻度で現れる。エンティティ数・時点数は、ハウスマンの補助回帰
        /// （傾き最大6個）がクラスター（`G = N`）・Driscoll-Kraay（`T`）のどちらでも成立する
        /// 7以上にする。
        #[derive(Debug, Clone)]
        struct OneDimCase {
            n_entities: usize,
            n_periods: usize,
            y: Vec<f64>,
            /// 両次元で変動する列、続いて`xi`・`xt`（使う場合）。
            x: Vec<Vec<f64>>,
            keys: Vec<u64>,
        }

        fn one_dim_case_strategy() -> impl Strategy<Value = OneDimCase> {
            (
                7..=9usize,
                7..=9usize,
                1..=2usize,
                any::<bool>(),
                any::<bool>(),
                any::<bool>(),
                any::<bool>(),
            )
                .prop_flat_map(
                    |(n_entities, n_periods, k_both, has_xi, has_xt, has_ent, has_time)| {
                        let n = n_entities * n_periods;
                        (
                            Just((n_entities, n_periods, k_both, has_xi, has_xt)),
                            Just((has_ent, has_time)),
                            collection::vec(-5.0f64..5.0, n),
                            collection::vec(collection::vec(-10.0f64..10.0, n), k_both),
                            collection::vec(-10.0f64..10.0, n_entities),
                            collection::vec(-10.0f64..10.0, n_periods),
                            collection::vec(-15.0f64..15.0, n_entities),
                            collection::vec(-15.0f64..15.0, n_periods),
                            collection::vec(any::<u64>(), n),
                        )
                    },
                )
                .prop_map(
                    |(
                        (n_entities, n_periods, _k_both, has_xi, has_xt),
                        (has_ent, has_time),
                        noise,
                        both,
                        xi_by_entity,
                        xt_by_time,
                        entity_effect,
                        time_effect,
                        keys,
                    )| {
                        let n = n_entities * n_periods;
                        let mut x = both;
                        if has_xi {
                            x.push((0..n).map(|r| xi_by_entity[r / n_periods]).collect());
                        }
                        if has_xt {
                            x.push((0..n).map(|r| xt_by_time[r % n_periods]).collect());
                        }
                        let y: Vec<f64> = (0..n)
                            .map(|r| {
                                let e = if has_ent {
                                    entity_effect[r / n_periods]
                                } else {
                                    0.0
                                };
                                let t = if has_time {
                                    time_effect[r % n_periods]
                                } else {
                                    0.0
                                };
                                x.iter().map(|c| c[r]).sum::<f64>() + e + t + noise[r]
                            })
                            .collect();
                        OneDimCase {
                            n_entities,
                            n_periods,
                            y,
                            x,
                            keys,
                        }
                    },
                )
        }

        /// 2-wayの全`cov_type`（インデックス0..=5: Classical, Hc1, Hc2, Hc3, Cluster(entity),
        /// Dk(時点ラベルは`time`)）。
        fn two_way_cov(index: usize, time: &[String]) -> ReCovType {
            match index {
                0 => ReCovType::Classical,
                1 => ReCovType::Hc1,
                2 => ReCovType::Hc2,
                3 => ReCovType::Hc3,
                4 => ReCovType::Cluster { groups: None },
                _ => ReCovType::Dk {
                    bandwidth: None,
                    time: TimeKeys::lexicographic(time.to_vec()),
                },
            }
        }

        fn two_way_labels(n_entities: usize, n_periods: usize) -> (Vec<String>, Vec<String>) {
            let n = n_entities * n_periods;
            (
                (0..n).map(|r| format!("e{}", r / n_periods)).collect(),
                (0..n).map(|r| format!("t{}", r % n_periods)).collect(),
            )
        }

        fn fit_two_way(
            y: &[f64],
            x: &[Vec<f64>],
            entity: &[String],
            time: &[String],
            cov_index: usize,
        ) -> Result<ReEstimator, PanelError> {
            fit_two_way_with(y, x, entity, time, cov_index, VarianceMethod::SwamyArora)
        }

        fn fit_two_way_with(
            y: &[f64],
            x: &[Vec<f64>],
            entity: &[String],
            time: &[String],
            cov_index: usize,
            method: VarianceMethod,
        ) -> Result<ReEstimator, PanelError> {
            let names: Vec<String> = (0..x.len()).map(|j| format!("x{j}")).collect();
            ReInput::from_columns(y, x, names, entity, Some(time), "y".to_string()).and_then(
                |input| {
                    ReEstimator::fit_with_variance_method(
                        input,
                        ReEffects::TwoWay,
                        two_way_cov(cov_index, time),
                        method,
                        0.95,
                    )
                },
            )
        }

        /// 一次元でしか変わらない列を含むバランスパネルの1-way用（`xt`は時点間のみ変わる）。
        #[derive(Debug, Clone)]
        struct TimeOnlyCase {
            n_entities: usize,
            n_periods: usize,
            y: Vec<f64>,
            /// 両次元で変動する列、続いて時点間のみ変わる`xt`。
            x: Vec<Vec<f64>>,
            keys: Vec<u64>,
        }

        fn time_only_case_strategy() -> impl Strategy<Value = TimeOnlyCase> {
            (6..=9usize, 3..=5usize, 1..=2usize, any::<bool>())
                .prop_flat_map(|(n_entities, n_periods, k, has_effect)| {
                    let n = n_entities * n_periods;
                    (
                        Just((n_entities, n_periods, has_effect)),
                        collection::vec(-5.0f64..5.0, n),
                        collection::vec(collection::vec(-10.0f64..10.0, n), k),
                        collection::vec(-10.0f64..10.0, n_periods),
                        collection::vec(-15.0f64..15.0, n_entities),
                        collection::vec(any::<u64>(), n),
                    )
                })
                .prop_map(
                    |((n_entities, n_periods, has_effect), noise, mut x, xt, effect, keys)| {
                        let n = n_entities * n_periods;
                        x.push((0..n).map(|r| xt[r % n_periods]).collect());
                        let y: Vec<f64> = (0..n)
                            .map(|r| {
                                let e = if has_effect {
                                    effect[r / n_periods]
                                } else {
                                    0.0
                                };
                                x.iter().map(|c| c[r]).sum::<f64>() + e + noise[r]
                            })
                            .collect();
                        TimeOnlyCase {
                            n_entities,
                            n_periods,
                            y,
                            x,
                            keys,
                        }
                    },
                )
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(48))]

            /// 2-wayで一次元でしか変わらない列（`xi`・`xt`）を含んでも、全`cov_type`
            /// （Classical・HC1〜HC3・cluster・Driscoll-Kraay）で、行の並べ替えとエンティティ・時点
            /// ラベルの付け替えによって係数・標準誤差・ハウスマン統計量が変わらない。
            #[test]
            fn two_way_one_dimensional_columns_are_invariant_to_relabeling_for_every_cov_type(
                case in one_dim_case_strategy(),
                cov_index in 0..=5usize,
            ) {
                let (entity, time) = two_way_labels(case.n_entities, case.n_periods);
                let base = fit_two_way(&case.y, &case.x, &entity, &time, cov_index);
                prop_assert!(base.is_ok(), "{:?}", base.as_ref().err());
                let base = base.unwrap();
                let (params, se) = params_and_se(&base);

                let n = case.y.len();
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by_key(|&r| case.keys[r]);
                let y2: Vec<f64> = order.iter().map(|&r| case.y[r]).collect();
                let x2: Vec<Vec<f64>> = case
                    .x
                    .iter()
                    .map(|c| order.iter().map(|&r| c[r]).collect())
                    .collect();
                let entity2: Vec<String> = order
                    .iter()
                    .map(|&r| format!("z{}", case.n_entities - 1 - r / case.n_periods))
                    .collect();
                let time2: Vec<String> = order
                    .iter()
                    .map(|&r| format!("u{}", case.n_periods - 1 - r % case.n_periods))
                    .collect();
                let permuted = fit_two_way(&y2, &x2, &entity2, &time2, cov_index);
                prop_assert!(permuted.is_ok(), "{:?}", permuted.as_ref().err());
                let permuted = permuted.unwrap();
                let (params2, se2) = params_and_se(&permuted);

                for j in 0..params.len() {
                    assert_approx_eq(params2[j], params[j], &format!("param[{j}]"));
                    assert_approx_eq(se2[j], se[j], &format!("se[{j}]"));
                }
                match (base.hausman_statistic(), permuted.hausman_statistic()) {
                    (Some(a), Some(b)) => assert_approx_eq(b, a, "hausman_statistic"),
                    (None, None) => {}
                    (a, b) => prop_assert!(false, "hausman presence differs: {a:?} vs {b:?}"),
                }
                prop_assert_eq!(base.hausman_df(), permuted.hausman_df());
            }

            /// 2-wayで`y`を`c`倍すると、係数・標準誤差は`c`倍、ハウスマン統計量（Wald、
            /// スケール不変）は不変。一次元の列（あれば先頭のもの）に定数を足すと、切片だけが
            /// 動き、他の係数・標準誤差は変わらない。全`cov_type`。
            #[test]
            fn two_way_is_equivariant_to_y_scale_and_a_constant_shift_moves_only_the_intercept(
                case in one_dim_case_strategy(),
                cov_index in 0..=5usize,
                scale in 0.5f64..4.0,
                shift in -5.0f64..5.0,
            ) {
                let (entity, time) = two_way_labels(case.n_entities, case.n_periods);
                let base = fit_two_way(&case.y, &case.x, &entity, &time, cov_index);
                prop_assert!(base.is_ok(), "{:?}", base.as_ref().err());
                let base = base.unwrap();
                let (params, se) = params_and_se(&base);

                let y_scaled: Vec<f64> = case.y.iter().map(|v| v * scale).collect();
                let scaled = fit_two_way(&y_scaled, &case.x, &entity, &time, cov_index);
                prop_assert!(scaled.is_ok(), "{:?}", scaled.as_ref().err());
                let scaled = scaled.unwrap();
                let (params_s, se_s) = params_and_se(&scaled);
                for j in 0..params.len() {
                    assert_approx_eq(params_s[j], params[j] * scale, &format!("scaled param[{j}]"));
                    assert_approx_eq(se_s[j], se[j] * scale, &format!("scaled se[{j}]"));
                }
                if let (Some(a), Some(b)) = (base.hausman_statistic(), scaled.hausman_statistic()) {
                    assert_approx_eq(b, a, "hausman_statistic under y scaling");
                }

                // 一次元の列（なければ両次元の最後の列）に定数を足す。
                let target = case.x.len() - 1;
                let mut x_shifted = case.x.clone();
                for v in x_shifted[target].iter_mut() {
                    *v += shift;
                }
                let shifted = fit_two_way(&case.y, &x_shifted, &entity, &time, cov_index);
                prop_assert!(shifted.is_ok(), "{:?}", shifted.as_ref().err());
                let (params_h, se_h) = params_and_se(&shifted.unwrap());
                assert_approx_eq(params_h[0], params[0] - params[1 + target] * shift, "const");
                for j in 1..params.len() {
                    assert_approx_eq(params_h[j], params[j], &format!("shifted param[{j}]"));
                }
                for j in 1..se.len() {
                    assert_approx_eq(se_h[j], se[j], &format!("shifted se[{j}]"));
                }
            }

            /// 2-wayのGLSは、報告した3つの重み`θ₁・θ₂・θ₃`で`y - θ₁ȳᵢ - θ₂ȳₜ + θ₃ȳ`と
            /// 各列を手で変換し（切片列は`1 - θ₁ - θ₂ + θ₃`倍）、そのOLSを当てはめた係数・
            /// 標準誤差（Classical・HC1〜HC3）と一致する。一次元の列を含む。`variance_method`は3法とも
            /// （重みは推定法ごとに異なるが、報告した重みでの変換との一致は共通）。変換式の各項
            /// （個体平均・時点平均・全体平均）の取り違えや符号の誤りを検出する。
            ///
            /// 検出できる誤り: 全体平均項の符号反転と全体平均の除数のずれは本テストが、個体・
            /// 時点の重みの入れ替えは本テストと
            /// `two_way_with_one_zero_component_equals_the_one_component_gls`が検出する
            /// （`quasi_demean_two_way`へのバグ注入で確認）。変換式が一貫して誤っていても
            /// 不変性・同変性のテストは通るため、変換式の正しさはこのテストと固定値
            /// （plm）のテストが担う。
            #[test]
            fn two_way_fit_equals_ols_on_the_hand_transformed_data(
                case in one_dim_case_strategy(),
                cov_index in 0..=3usize,
                method in prop_oneof![
                    Just(VarianceMethod::SwamyArora),
                    Just(VarianceMethod::WallaceHussain),
                    Just(VarianceMethod::Amemiya)
                ],
            ) {
                let (entity, time) = two_way_labels(case.n_entities, case.n_periods);
                let est = fit_two_way_with(&case.y, &case.x, &entity, &time, cov_index, method);
                // Amemiyaはwithin変動の無い列（一次元の列）があると拒否される。それ以外の失敗は誤り。
                if matches!(est, Err(PanelError::AmemiyaRequiresWithinVariation { .. })) {
                    prop_assert_eq!(method, VarianceMethod::Amemiya);
                    prop_assume!(false);
                }
                prop_assert!(est.is_ok(), "{:?}", est.as_ref().err());
                let est = est.unwrap();
                let (theta1, theta2, theta3) = est.two_way_theta().unwrap();

                let n = case.y.len();
                let (n_e, n_t) = (case.n_entities, case.n_periods);
                let quasi = |v: &[f64]| -> Vec<f64> {
                    let mut by_entity = vec![0.0; n_e];
                    let mut by_time = vec![0.0; n_t];
                    for (r, &value) in v.iter().enumerate() {
                        by_entity[r / n_t] += value / n_t as f64;
                        by_time[r % n_t] += value / n_e as f64;
                    }
                    let overall = v.iter().sum::<f64>() / n as f64;
                    (0..n)
                        .map(|r| {
                            v[r] - theta1 * by_entity[r / n_t] - theta2 * by_time[r % n_t]
                                + theta3 * overall
                        })
                        .collect()
                };
                let y_star = quasi(&case.y);
                let mut columns = vec![vec![1.0 - theta1 - theta2 + theta3; n]];
                columns.extend(case.x.iter().map(|c| quasi(c)));
                let names: Vec<String> = (0..columns.len()).map(|j| format!("c{j}")).collect();
                let ols_input =
                    OlsInput::from_columns(&y_star, &columns, names, false, "y".to_string())
                        .unwrap();
                let cov = match cov_index {
                    0 => CovType::Classical,
                    1 => CovType::Hc1,
                    2 => CovType::Hc2,
                    _ => CovType::Hc3,
                };
                let ols = OlsEstimator::fit(ols_input, cov, 0.95);
                prop_assert!(ols.is_ok(), "{:?}", ols.as_ref().err());
                let ols = ols.unwrap();

                let (params, se) = params_and_se(&est);
                for j in 0..params.len() {
                    assert_approx_eq(params[j], *ols.params().get(j, 0), &format!("param[{j}]"));
                    assert_approx_eq(se[j], *ols.std_errors().get(j, 0), &format!("se[{j}]"));
                }
            }

            /// 2-wayで`σ_u²`だけ（または`σ_λ²`だけ）が0に切り詰められると、2-way GLSは
            /// 残った成分だけのGLS（`θ₁ = θ₃ = 0`のとき`y - θ₂·ȳ_t`、`θ₂ = θ₃ = 0`のとき
            /// `y - θ₁·ȳ_i`）に退化する。手で準偏差変換した`y`・`X`（切片列は`1 - θ`倍）への
            /// OLSの係数・Classical標準誤差と一致する。
            #[test]
            fn two_way_with_one_zero_component_equals_the_one_component_gls(
                case in one_dim_case_strategy(),
            ) {
                let (entity, time) = two_way_labels(case.n_entities, case.n_periods);
                let est = fit_two_way(&case.y, &case.x, &entity, &time, 0);
                prop_assert!(est.is_ok(), "{:?}", est.as_ref().err());
                let est = est.unwrap();
                let (sigma2_u, sigma2_lambda) =
                    (est.sigma2_u(), est.sigma2_lambda().unwrap());
                let (theta1, theta2, theta3) = est.two_way_theta().unwrap();
                // 片方だけが0（両方0ならθはすべて0でpooled OLSになり別の性質）。
                prop_assume!((sigma2_u == 0.0) != (sigma2_lambda == 0.0));
                prop_assert_eq!(theta3, 0.0);

                // 残った成分のグループ（`σ_u² = 0`なら時点、`σ_λ² = 0`ならエンティティ）。
                let (group_of, n_groups, theta) = if sigma2_u == 0.0 {
                    prop_assert_eq!(theta1, 0.0);
                    (
                        (0..case.y.len()).map(|r| r % case.n_periods).collect::<Vec<_>>(),
                        case.n_periods,
                        theta2,
                    )
                } else {
                    prop_assert_eq!(theta2, 0.0);
                    (
                        (0..case.y.len()).map(|r| r / case.n_periods).collect::<Vec<_>>(),
                        case.n_entities,
                        theta1,
                    )
                };
                let quasi = |v: &[f64]| -> Vec<f64> {
                    let mut sum = vec![0.0; n_groups];
                    let mut count = vec![0.0; n_groups];
                    for (r, &g) in group_of.iter().enumerate() {
                        sum[g] += v[r];
                        count[g] += 1.0;
                    }
                    (0..v.len())
                        .map(|r| v[r] - theta * sum[group_of[r]] / count[group_of[r]])
                        .collect()
                };
                let y_star = quasi(&case.y);
                let mut columns = vec![vec![1.0 - theta; case.y.len()]];
                columns.extend(case.x.iter().map(|c| quasi(c)));
                let names: Vec<String> = (0..columns.len()).map(|j| format!("c{j}")).collect();
                let ols_input =
                    OlsInput::from_columns(&y_star, &columns, names, false, "y".to_string())
                        .unwrap();
                let ols = OlsEstimator::fit(ols_input, CovType::Classical, 0.95);
                prop_assert!(ols.is_ok(), "{:?}", ols.as_ref().err());
                let ols = ols.unwrap();

                let (params, se) = params_and_se(&est);
                for j in 0..params.len() {
                    assert_approx_eq(params[j], *ols.params().get(j, 0), &format!("param[{j}]"));
                    assert_approx_eq(se[j], *ols.std_errors().get(j, 0), &format!("se[{j}]"));
                }
            }

            /// 1-wayのバランスパネルで時点間でのみ変わる列`xt`を含んでも、行の並べ替えと
            /// エンティティラベルの付け替えで結果が変わらず、ハウスマン検定の自由度は
            /// 両次元で変動する列の数（`xt`は`X̃`が識別できず外れる）、`xt`に定数を足すと
            /// 切片だけが動く。
            #[test]
            fn one_way_balanced_time_only_column_is_invariant_and_excluded_from_hausman(
                case in time_only_case_strategy(),
                cov in re_cov_strategy(),
                shift in -5.0f64..5.0,
            ) {
                let n = case.y.len();
                let entity: Vec<String> =
                    (0..n).map(|r| format!("e{}", r / case.n_periods)).collect();
                let base = ReEstimator::fit(
                    re_input(&case.y, &case.x, &entity),
                    ReEffects::OneWay,
                    cov.clone(),
                    0.95,
                );
                prop_assert!(base.is_ok(), "{:?}", base.as_ref().err());
                let base = base.unwrap();
                let (params, se) = params_and_se(&base);
                prop_assert_eq!(base.hausman_df(), Some(case.x.len() - 1));

                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by_key(|&r| case.keys[r]);
                let y2: Vec<f64> = order.iter().map(|&r| case.y[r]).collect();
                let x2: Vec<Vec<f64>> = case
                    .x
                    .iter()
                    .map(|c| order.iter().map(|&r| c[r]).collect())
                    .collect();
                let entity2: Vec<String> = order
                    .iter()
                    .map(|&r| format!("z{}", case.n_entities - 1 - r / case.n_periods))
                    .collect();
                let permuted = ReEstimator::fit(
                    re_input(&y2, &x2, &entity2),
                    ReEffects::OneWay,
                    cov.clone(),
                    0.95,
                );
                prop_assert!(permuted.is_ok(), "{:?}", permuted.as_ref().err());
                let permuted = permuted.unwrap();
                let (params2, se2) = params_and_se(&permuted);
                for j in 0..params.len() {
                    assert_approx_eq(params2[j], params[j], &format!("param[{j}]"));
                    assert_approx_eq(se2[j], se[j], &format!("se[{j}]"));
                }
                match (base.hausman_statistic(), permuted.hausman_statistic()) {
                    (Some(a), Some(b)) => assert_approx_eq(b, a, "hausman_statistic"),
                    (a, b) => prop_assert!(false, "hausman must exist: {a:?} vs {b:?}"),
                }

                // `xt`（最後の列）に定数を足す。
                let target = case.x.len() - 1;
                let mut x_shifted = case.x.clone();
                for v in x_shifted[target].iter_mut() {
                    *v += shift;
                }
                let shifted = ReEstimator::fit(
                    re_input(&case.y, &x_shifted, &entity),
                    ReEffects::OneWay,
                    cov,
                    0.95,
                );
                prop_assert!(shifted.is_ok(), "{:?}", shifted.as_ref().err());
                let (params_h, se_h) = params_and_se(&shifted.unwrap());
                assert_approx_eq(params_h[0], params[0] - params[1 + target] * shift, "const");
                for j in 1..params.len() {
                    assert_approx_eq(params_h[j], params[j], &format!("shifted param[{j}]"));
                    assert_approx_eq(se_h[j], se[j], &format!("shifted se[{j}]"));
                }
            }
        }
    }

    /// 1-way・2-wayの全`cov_type`で、`cov_params()`が対称で、対角の平方根が`std_errors()`と
    /// 一致する。次元は切片を含む`(df_model, df_model)`。
    #[test]
    fn cov_params_is_symmetric_and_diagonal_matches_std_errors_for_every_cov_type() {
        let entity: Vec<String> = (0..24).map(|i| format!("e{}", i / 4)).collect();
        let time: Vec<String> = (0..24).map(|i| format!("t{}", i % 4)).collect();
        let x1: Vec<f64> = (0..24)
            .map(|i| ((i as f64) * 0.7).sin() * 4.0 + i as f64)
            .collect();
        let x2: Vec<f64> = (0..24).map(|i| ((i as f64) * 1.3).cos() * 3.0).collect();
        let y: Vec<f64> = (0..24)
            .map(|i| {
                2.0 + 1.5 * x1[i] - 0.8 * x2[i] + ((i as f64) * 2.1).sin() * 1.7 + (i / 4) as f64
            })
            .collect();

        for effects in [ReEffects::OneWay, ReEffects::TwoWay] {
            let cov_types = [
                ReCovType::Classical,
                ReCovType::Hc1,
                ReCovType::Hc2,
                ReCovType::Hc3,
                ReCovType::Cluster { groups: None },
                ReCovType::Dk {
                    bandwidth: Some(1),
                    time: TimeKeys::lexicographic(time.clone()),
                },
            ];
            for cov_type in cov_types {
                let name = format!("{effects:?}/{cov_type:?}");
                let input = ReInput::from_columns(
                    &y,
                    &[x1.clone(), x2.clone()],
                    vec!["x1".to_string(), "x2".to_string()],
                    &entity,
                    Some(&time),
                    "y".to_string(),
                )
                .unwrap();
                let re = ReEstimator::fit(input, effects, cov_type, 0.95).unwrap();

                let cov = re.cov_params();
                let k = re.estimator().params().nrows();
                assert_eq!(k, 3, "{name}: const + two slopes");
                assert_eq!((cov.nrows(), cov.ncols()), (k, k), "{name}: shape");
                for i in 0..k {
                    for j in 0..k {
                        let scale = (*cov.get(i, i) * *cov.get(j, j)).sqrt();
                        assert!(
                            (*cov.get(i, j) - *cov.get(j, i)).abs() <= 1e-12 * scale,
                            "{name}: cov_params not symmetric at ({i},{j})"
                        );
                    }
                    let se = *re.std_errors().get(i, 0);
                    assert!(
                        (cov.get(i, i).sqrt() - se).abs() <= 1e-12 * se,
                        "{name}: sqrt(cov_params[{i}][{i}]) != std_errors[{i}]"
                    );
                }
            }
        }
    }
}
