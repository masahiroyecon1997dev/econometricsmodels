//! `FeEstimator::fit`の本体。singleton検出・within変換・自由度検証・最小二乗・`cov_type`別の共分散・
//! 統計量の再計算をつなぐオーケストレーション。

use statrs::distribution::StudentsT;

use super::FeEstimator;
use super::input::{FeCovType, FeEffects, FeInput};
use super::r_squared::{fe_r_squared_between, fe_r_squared_overall};
use super::within::{
    validate_no_singleton_groups_one_way, validate_no_singleton_groups_two_way,
    validate_no_zero_variance_regressors, within_transform_one_way, within_transform_two_way,
};
use crate::linear::common::LeastSquaresError;
use crate::linear::ols::OlsInput;
use crate::panel::common::{PanelCov, PanelCovInput, PanelError, PanelStructure, panel_cov_params};
use crate::shared::error::CommonError;
use crate::shared::goodness_of_fit::{gaussian_goodness_of_fit, is_perfect_fit};
use crate::shared::group_codes::GroupCodes;
use crate::shared::inference::{InferenceTable, inference_table};
use crate::shared::least_squares::{LeastSquaresFit, least_squares};
use crate::shared::validation::column_is_zero_variance;
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
) -> Result<FeEstimator, PanelError> {
    // faerのグローバル並列度をPar::Seqに固定する（`crate::shared::parallelism`。
    // `cargo test -p engine`でFeEstimator::fitを直接叩く経路でも担保するためここで呼ぶ、`engine/src/panel/CLAUDE.md`「faerの
    // グローバル並列度」参照）。
    crate::shared::parallelism::ensure_serial();

    // 2-wayの時点コード（1-wayは`None`）。`time`が無いのに2-wayを指定した入力はここで
    // 弾き、以降は`Some`かどうかだけで1-way/2-wayを分岐する（`effects`と`input.time`の
    // 食い違いを後段で`expect`せずに済む）。
    let two_way_time = two_way_time(effects, &input)?;

    let (y, x) = match effects {
        FeEffects::OneWay => {
            validate_no_singleton_groups_one_way(&input)?;
            within_transform_one_way(&input)
        }
        FeEffects::TwoWay => {
            validate_no_singleton_groups_two_way(&input)?;
            within_transform_two_way(&input)?
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

    validate_no_zero_variance_regressors(&input, &x)?;

    // 被説明変数もwithin変換後に分散が残らなければ、固定効果が説明するものが無い
    // （個体内で定数、2-wayなら個体効果と時点効果だけで説明される）。説明変数と同じ
    // 相対閾値で判定する。
    if column_is_zero_variance(input.y(), &y) {
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
    if is_perfect_fit(ssr, sst_within, k) {
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
mod tests {
    use super::super::FeEstimator;

    use super::super::input::FeCovType;
    use super::super::input::FeEffects;
    use super::super::input::FeInput;

    use super::super::test_support::*;

    use crate::linear::common::LeastSquaresError;
    use crate::linear::ols::OlsInput;

    use crate::panel::common::PanelDimension;

    use crate::panel::common::PanelError;

    use crate::panel::common::TimeKeys;

    use crate::shared::error::ClusterSetting;
    use crate::shared::error::CommonError;
    use crate::shared::error::MatrixKind;
    use crate::shared::error_catalog::Stage;

    use crate::shared::group_codes::GroupCodes;
    use crate::shared::inference;

    /// 辞書順の`TimeKeys`（`FeCovType::Dk.time`に渡す）。
    fn lex_time(values: &[&str]) -> TimeKeys {
        TimeKeys::lexicographic(strings(values))
    }

    // ── FeEstimator::fit ─────────────────────────────────────────────────

    #[test]
    fn fe_estimator_fit_one_way_recovers_known_slope() {
        // entity a: x=[1,2,3], y=2x+5（fixed effect=5）+ノイズ
        // entity b: x=[4,5,6], y=2x+10（fixed effect=10）+ノイズ
        // 小さなノイズのもとで、within変換後のOLS（切片なし）は真のスロープ2.0の近くに返る
        // （fixed effectはwithin変換で消去される）。
        let entity = strings(&["a", "a", "a", "b", "b", "b"]);
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, -0.2, 0.1, 0.05, -0.1, 0.05];
        let y: Vec<f64> = x1
            .iter()
            .zip(&entity)
            .zip(noise)
            .map(|((x, e), u)| 2.0 * x + if e == "a" { 5.0 } else { 10.0 } + u)
            .collect();
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert!((*fe.params().get(0, 0) - 2.0).abs() < 0.2);
        assert_eq!(fe.effects(), FeEffects::OneWay);
        assert!(!fe.within_input().has_intercept());
        assert_eq!(fe.n_periods(), None);
        for r in fe.residuals().col(0).iter() {
            assert!(r.abs() < 0.3);
        }
    }

    #[test]
    fn fe_estimator_fit_returns_perfect_fit_for_exact_data() {
        // ノイズなしでは、within回帰の残差が全て0で標準誤差とF検定が定義できない。
        let entity = strings(&["a", "a", "a", "b", "b", "b"]);
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<f64> = x1
            .iter()
            .zip(&entity)
            .map(|(x, e)| 2.0 * x + if e == "a" { 5.0 } else { 10.0 })
            .collect();
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::WithinRegressionFailed {
                source: CommonError::PerfectFit.into(),
            }
        );
    }

    #[test]
    fn fe_estimator_fit_returns_outcome_constant_within_entity_for_an_entity_level_outcome() {
        // yが個体内で定数（個体間でだけ違う）なら、固定効果が全てを説明してwithin変換後のyは0になる。
        // 0.1のように3回足して3で割ると丸め誤差が残る値も、相対閾値で定数とみなす。
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c"]);
        let y = [0.1, 0.1, 0.1, 0.7, 0.7, 0.7, 1.3, 1.3, 1.3];
        let x1 = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 1.0, 4.0, 2.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

        assert_eq!(result.unwrap_err(), PanelError::OutcomeConstantWithinEntity);
    }

    #[test]
    fn fe_estimator_fit_returns_outcome_constant_within_entity_when_effects_explain_the_outcome() {
        // 2-wayでは、y = 個体効果 + 時点効果 もwithin変換後に0になる。
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c"]);
        let time = strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3"]);
        let alpha = [1.0, 1.0, 1.0, 4.0, 4.0, 4.0, -2.0, -2.0, -2.0];
        let gamma = [0.0, 0.5, 1.5, 0.0, 0.5, 1.5, 0.0, 0.5, 1.5];
        let y: Vec<f64> = alpha.iter().zip(gamma).map(|(a, g)| a + g).collect();
        let x1 = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 1.0, 4.0, 2.0];
        let input = FeInput::from_columns(
            &y,
            &[x1],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95);

        assert_eq!(result.unwrap_err(), PanelError::OutcomeConstantWithinEntity);
    }

    #[test]
    fn fe_estimator_fit_exposes_input_and_cov_type_via_getters() {
        // `input()`/`cov_type()`（カバレッジ監査で判明した未検証の単純
        // getter）。`ols::fit_exposes_input_cov_type_and_residuals_via_getters`と同型。
        let entity = strings(&["a", "a", "b", "b"]);
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let y = [7.0, 9.2, 11.0, 12.6];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster { groups: None },
            0.95,
        )
        .unwrap();

        assert_eq!(fe.input().nobs(), 4);
        assert_eq!(fe.input().dep_var_name(), "y");
        assert_eq!(fe.cov_type(), &FeCovType::Cluster { groups: None });
        // `df_inference()`/`stat_dist()`（カバレッジ監査で判明した未検証の単純
        // getter、新設）。`n_entities=2`が既定クラスターのため
        // `df_inference = G-1 = 1`（`df_resid`の`n-df_model=4-3=1`とはこの
        // データではたまたま同じ値になるが、由来は別——後続の
        // `..._one_way_cluster_on_entity_matches_fixest_nested_k`等で
        // `df_resid`と乖離するケースを別途確認済み）。
        assert_eq!(fe.df_inference(), 1);
        assert_eq!(fe.stat_dist(), inference::StatDist::T { df: 1 });
    }

    #[test]
    fn fe_estimator_fit_two_way_recovers_known_slope() {
        // `within_transform_two_way_matches_closed_form_double_demeaning`と同じ関係
        // （x1はyのちょうど2倍。この関係は線形変換の下で任意のN・Tで恒等的に保たれるため
        // 具体的な値は問わない）だが、3エンティティ×3時点（n=9）のバランスパネルに
        // 拡張する：df_model=k(1)+neffects(n_entities+n_periods-1=3+3-1=5)=6、n=9>6で
        // 自由度検証（`n<=df_model`）を通過できる規模にする必要があるため
        // （N=2,T=2のn=4だとdf_model=1+3=4となりn<=df_modelで弾かれてしまう）。
        // 2-way within変換後、x1_out ≈ 2 * y_out がほぼ成り立つため、切片なしOLSの
        // スロープは0.5にほぼ一致するはず。x1はy*2からごくわずかに擾乱を入れる
        // （厳密にx1=2*yだと残差が全行ゼロになり、F検定
        // （`wald_f_test`）が分散ゼロによる特異行列で`NearSingularMatrix`を返してしまう
        // 退化ケースを踏むため。既存の`WlsEstimator`のテストコメント
        // `fit_without_intercept_uses_uncentered_r_squared_and_omits_const`と同じ理由）。
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c"]);
        let time = strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3"]);
        let y = [1.0, 3.0, 5.0, 2.0, 4.0, 9.0, 6.0, 1.0, 8.0];
        let noise = [0.01, -0.02, 0.015, -0.01, 0.02, -0.015, 0.01, -0.01, 0.02];
        let x1: Vec<f64> = y.iter().zip(&noise).map(|(v, n)| 2.0 * v + n).collect();
        let input = FeInput::from_columns(
            &y,
            &[x1],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();

        // 許容誤差0.01は`testing-policy.md`の相対誤差1e-8方針の対象外（リファレンス実装
        // との数値比較ではなく、本テスト自身が注入した擾乱ノイズに対する内部整合性の
        // 確認のため）。ノイズの大きさ（最大0.02、xスケール比で見ると相対誤差1〜2%程度）
        // に対してスロープ推定への影響がこの範囲に収まることを確認する目的の閾値。
        assert!((*fe.params().get(0, 0) - 0.5).abs() < 0.01);
        assert_eq!(fe.effects(), FeEffects::TwoWay);
        assert_eq!(fe.df_model(), 6);
        assert_eq!(fe.n_periods(), Some(3));
        assert_eq!(fe.df_resid(), 3);
    }

    #[test]
    fn fe_estimator_fit_propagates_singleton_error() {
        let entity = strings(&["a", "a", "c"]);
        let y = [1.0, 2.0, 3.0];
        let x1 = vec![1.0, 2.0, 3.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

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
    fn fe_estimator_fit_propagates_zero_variance_error() {
        // "female"は各エンティティ内で一定（時間不変）。n=6・n_entities=2・k=2で
        // df_model=4・n>df_modelとなるよう、各エンティティ3観測に拡張する
        // （n=4だとdf_model=2+2=4でn<=df_modelとなりInsufficientDegreesOfFreedomが
        // 先に発火してしまうため、自由度検証を踏まえたサイズにする）。
        let entity = strings(&["a", "a", "a", "b", "b", "b"]);
        let y = [1.0, 2.0, 3.0, 5.0, 6.0, 8.0];
        let x_varying = vec![10.0, 20.0, 15.0, 5.0, 10.0, 20.0];
        let female = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let input = FeInput::from_columns(
            &y,
            &[x_varying, female],
            vec!["x_varying".to_string(), "female".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::ZeroVarianceAfterDemeaning {
                column: "female".to_string(),
            }
        );
    }

    #[test]
    fn fe_estimator_fit_two_way_requires_time() {
        let entity = strings(&["a", "a", "b", "b"]);
        let y = [1.0, 2.0, 3.0, 4.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95);

        assert_eq!(result.unwrap_err(), PanelError::TwoWayRequiresTime);
    }

    #[test]
    fn fe_estimator_fit_two_way_without_time_reports_requires_time_before_any_other_error() {
        // 2-wayで`time`が無いことは、他のどの欠陥（entityのsingleton・不正な
        // `confidence_level`・cov_type固有の検証）よりも先に報告される。優先順位は
        // `fit()`冒頭の`two_way_time`が守る。
        let entity = strings(&["a", "a", "c"]);
        let y = [1.0, 2.0, 3.0];
        let covs = || {
            vec![
                FeCovType::Classical,
                FeCovType::Hc2,
                FeCovType::Cluster { groups: None },
                FeCovType::Dk {
                    bandwidth: Some(100),
                    time: lex_time(&["1", "2", "3"]),
                },
            ]
        };
        for cov_type in covs() {
            let input = FeInput::from_columns(
                &y,
                &[vec![1.0, 2.0, 4.0]],
                vec!["x1".to_string()],
                &entity,
                None,
                "y".into(),
            )
            .unwrap();

            let result = FeEstimator::fit(input, FeEffects::TwoWay, cov_type.clone(), 1.5);

            assert_eq!(
                result.unwrap_err(),
                PanelError::TwoWayRequiresTime,
                "cov_type = {cov_type:?}"
            );
        }
    }

    #[test]
    fn fe_estimator_fit_two_way_propagates_singleton_error() {
        // entity "c"は時点"1"のみの1観測（singleton）。`fit()`が
        // `validate_no_singleton_groups_two_way`（entity/time双方を対称にチェックする方）を
        // 正しく呼んでいることの配線確認（1-way用の検証関数を誤って呼んでいないか）。
        let entity = strings(&["a", "a", "b", "b", "c"]);
        let time = strings(&["1", "2", "1", "2", "1"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = FeInput::from_columns(
            &y,
            &[x1],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95);

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
    fn fe_estimator_fit_two_way_propagates_unbalanced_panel_error() {
        // entity=[a,a,b,b], time=[1,1,2,2]: (a,1)が重複、(a,2)と(b,1)が欠落。
        // singletonではない（各entity/time値とも観測数2）ため、`fit()`が
        // `within_transform_two_way`のバランスパネル検証まで到達していることの配線確認。
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["1", "1", "2", "2"]);
        let y = [1.0, 2.0, 3.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let result = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::UnbalancedPanelForTwoWay {
                n_obs: 4,
                entity_column: "entity".to_string(),
                n_entities: 2,
                time_column: "time".to_string(),
                n_periods: 2,
            }
        );
    }

    #[test]
    fn fe_estimator_fit_two_way_propagates_zero_variance_error() {
        // "year_dummy"はエンティティ間で変動しない（time FEと完全共線）。`fit()`が
        // within変換後に`validate_no_zero_variance_regressors`を呼んでいることの配線確認。
        // n=9（3エンティティ×3時点、n_entities=3・n_periods=3）に拡張し、
        // df_model=k(2)+neffects(3+3-1=5)=7・n>df_modelとなるサイズにする
        // （自由度検証を先に通過させるため）。"x_varying"は
        // entity×timeの交互作用項（entity_idx*time_idx）にして、加法分離可能な
        // 主効果のみの列（2-way demeanで機械的にゼロになる）にならないようにする
        // （Pythonで2-way demeanした結果、分散0.444...とゼロでないことを確認済み）。
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c"]);
        let time = strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let x_varying = vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0];
        let year_dummy = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let input = FeInput::from_columns(
            &y,
            &[x_varying, year_dummy],
            vec!["x_varying".to_string(), "year_dummy".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::ZeroVarianceAfterDemeaning {
                column: "year_dummy".to_string(),
            }
        );
    }

    #[test]
    fn fe_estimator_fit_propagates_invalid_confidence_level_error() {
        // `confidence_level`の範囲チェック
        // （`(0, 1)`の範囲外）が、`PanelError::WithinRegressionFailed`として正しく
        // 伝播することを確認する（`iv::two_sls`の同型テストに倣う）。
        let entity = strings(&["a", "a", "b", "b"]);
        let y = [1.0, 2.0, 3.0, 4.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 1.5);

        assert_eq!(
            result.unwrap_err(),
            PanelError::WithinRegressionFailed {
                source: LeastSquaresError::Common(CommonError::InvalidConfidenceLevel {
                    confidence_level: 1.5,
                }),
            }
        );
    }

    #[test]
    fn fe_estimator_fit_wraps_ols_failure_as_within_regression_failed() {
        // 自由度検証（`n <= df_model`）が最小二乗側の`n <= k`相当の条件より常に厳しい
        // （`df_model = k + neffects > k`）ため、最小二乗側の観測数不足はこの経路では
        // 発生しえない（`fe_estimator_fit_propagates_insufficient_degrees_of_freedom_error`が
        // 先に弾く）。そのため、ここでは最小二乗固有の別の失敗——完全な多重共線性（`LeastSquaresError::
        // SingularMatrix`）——を踏ませる: x2はx1のちょうど2倍で、within変換
        // （線形変換）後も比例関係`x2_demeaned = 2 * x1_demeaned`が保たれ完全共線になる。
        // n=6（3エンティティ、singletonではない）・k=2でdf_model=2+3=5、n=6>5と
        // 自由度検証は通過するようにする。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x1 = vec![1.0, 3.0, 2.0, 6.0, 4.0, 10.0];
        let x2: Vec<f64> = x1.iter().map(|v| 2.0 * v).collect();
        let input = FeInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

        assert!(matches!(
            result.unwrap_err(),
            PanelError::WithinRegressionFailed { .. }
        ));
    }

    #[test]
    fn fe_estimator_fit_propagates_insufficient_degrees_of_freedom_error() {
        // n=4・n_entities=2・k=2でdf_model=k+n_entities=4となり、n<=df_modelのため
        // 最小二乗へ進む前に`PanelError::InsufficientDegreesOfFreedom`で
        // 弾かれることを確認する。
        let entity = strings(&["a", "a", "b", "b"]);
        let y = [1.0, 2.0, 3.0, 4.0];
        let x1 = vec![1.0, 3.0, 2.0, 6.0];
        let x2 = vec![2.0, 5.0, 1.0, 9.0];
        let input = FeInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::InsufficientDegreesOfFreedom {
                n_obs: 4,
                n_entities: 2,
                n_periods: None,
                k: 2,
            }
        );
    }

    /// エラーの`min_obs`が`fit`の実際の境界と一致する: 拒否される最大の`n`（`n = df_model`）の
    /// 次の`n = min_obs`で成功する。1-wayの`n_effects`は個体数（ここでは2）。
    #[test]
    fn insufficient_degrees_of_freedom_min_obs_matches_the_fit_boundary() {
        use crate::shared::error::ErrorInfo;
        use crate::shared::error_catalog::ParamValue;

        let build = |n: usize, k: usize| {
            let entity: Vec<String> = (0..n)
                .map(|i| if i < n.div_ceil(2) { "a" } else { "b" }.to_string())
                .collect();
            let y: Vec<f64> = (0..n)
                .map(|i| (i as f64 * 0.9).sin() + 0.1 * i as f64)
                .collect();
            let x: Vec<Vec<f64>> = (0..k)
                .map(|j| {
                    (0..n)
                        .map(|i| ((i * (j + 2) + j) as f64 * 1.3).cos())
                        .collect()
                })
                .collect();
            let names = (0..k).map(|j| format!("x{j}")).collect();
            FeInput::from_columns(&y, &x, names, &entity, None, "y".into()).unwrap()
        };
        for k in 2..=4_usize {
            let n_effects = 2;
            let rejected_n = n_effects + k;
            let err = FeEstimator::fit(
                build(rejected_n, k),
                FeEffects::OneWay,
                FeCovType::Classical,
                0.95,
            )
            .unwrap_err();
            let min_obs = err
                .report()
                .all_params()
                .into_iter()
                .find_map(|(name, value)| match (name.as_str(), value) {
                    ("min_obs", ParamValue::Int(v)) => Some(v as usize),
                    _ => None,
                })
                .expect("min_obs");
            assert_eq!(min_obs, rejected_n + 1, "k={k}");
            assert!(
                FeEstimator::fit(
                    build(min_obs, k),
                    FeEffects::OneWay,
                    FeCovType::Classical,
                    0.95
                )
                .is_ok(),
                "k={k}"
            );
        }
    }

    #[test]
    fn fe_estimator_fit_one_way_matches_fixest_reference() {
        // Rの`fixest::feols(y ~ x | id)`（`id`: a/b/c/d各3観測）の実測値と数値比較する
        // （5.2節・fixestがFEのRクロスチェック参照実装）。期待値はRで独立に計算・検算済み
        // （`options(digits=15)`でフルの浮動小数点精度を取得）。
        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert_eq!(fe.df_model(), 5); // k(1) + n_entities(4)
        assert_eq!(fe.df_resid(), 7); // n(12) - df_model(5)
        assert!((*fe.params().get(0, 0) - 1.402_777_777_777_78).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.432_598_838_244_034).abs() < 1e-6);
        assert!((*fe.test_stats().get(0, 0) - 3.242_675_785_889_31).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.014_200_386_789_949_8).abs() < 1e-6);
        assert!((*fe.conf_lower().get(0, 0) - 0.379_844_073_655_07).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 2.425_711_481_900_49).abs() < 1e-6);
        assert!((fe.aic() - 55.612_545_928_611_7).abs() < 1e-6);
        assert!((fe.bic() - 58.037_079_177_551_7).abs() < 1e-6);
        // パネル固有R²: 1-wayではwithinは`linearmodels`の`rsquared_within`と
        // `fixest`の`fitstat(m, "wr2")`が一致する（`fe.rs`のモジュールdoc「パネル固有R²」参照）。
        // between/overallは`linearmodels`の値（Pythonで独立に計算・検算済み）。
        assert!((fe.r_squared_within() - 0.600_341_337_099_812).abs() < 1e-9);
        assert!((fe.r_squared_between() - 0.748_302_743_867_978).abs() < 1e-9);
        assert!((fe.r_squared_overall() - 0.732_444_936_421_435).abs() < 1e-9);
        // F統計量: k=1のため「1自由度のF検定は両側t検定と代数的に等価」
        // （`fe.rs`のモジュールdoc「自由度とF統計量」参照）。
        assert!((fe.f_statistic().unwrap() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value().unwrap() - *fe.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_matches_fixest_reference() {
        // 同じデータでの`fixest::feols(y ~ x | id + t)`（2-way）の実測値と数値比較する。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();

        assert_eq!(fe.df_model(), 7); // k(1) + neffects(n_entities(4)+n_periods(3)-1=6)
        assert_eq!(fe.df_resid(), 5); // n(12) - df_model(7)
        assert!((*fe.params().get(0, 0) - 0.822_429_906_542_056).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.227_239_295_931_651).abs() < 1e-6);
        assert!((*fe.test_stats().get(0, 0) - 3.619_223_969_033_19).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.015_231_948_369_008_1).abs() < 1e-6);
        assert!((*fe.conf_lower().get(0, 0) - 0.238_292_700_077_37).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.406_567_113_006_74).abs() < 1e-6);
        assert!((fe.aic() - 36.559_692_739_993_7).abs() < 1e-6);
        assert!((fe.bic() - 39.954_039_288_509_7).abs() < 1e-6);
        // パネル固有R²: 2-wayのwithinは`linearmodels`の`rsquared_within`
        // （常にentityのみdemean）とは意図的に食い違うため、`fixest`の
        // `fitstat(m, "wr2")`（0.723738317757009、`options(digits=15)`でR実地検証済み）を
        // 参照値にする（`fe.rs`のモジュールdoc「パネル固有R²」参照）。between/overallは
        // `linearmodels`の値。
        assert!((fe.r_squared_within() - 0.723_738_317_757_009).abs() < 1e-9);
        assert!((fe.r_squared_between() - 0.513_009_039_069_012).abs() < 1e-9);
        assert!((fe.r_squared_overall() - 0.511_356_250_429_877).abs() < 1e-9);
        // F統計量: k=1のため「1自由度のF検定は両側t検定と代数的に等価」
        // （`fe.rs`のモジュールdoc「自由度とF統計量」参照）。
        assert!((fe.f_statistic().unwrap() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value().unwrap() - *fe.p_values().get(0, 0)).abs() < 1e-9);
    }

    // ── F統計量 ───────────────────────────────────────────────────────────

    /// N=4（id: a,b,c,d）×T=3（t: 1,2,3）のバランスパネル、k=2（`fixest_reference_input`の
    /// 単回帰では真の同時検定（複数の傾き係数）を検証できないため、2変数版として別に用意
    /// する）。`fitstat(m, "f")`はFEダミーも含めたモデル全体のF検定でありここでの定義
    /// （FEダミーを除く傾き係数のみの同時検定、`linearmodels.PanelOLS.f_statistic`と同じ、
    /// `fe.rs`のモジュールdoc「自由度とF統計量」参照）と異なるため、fixestではなく
    /// `linearmodels`（`cov_type="unadjusted"`）の値と直接比較する（期待値はPythonで
    /// 独立に計算・検算済み）。
    #[allow(clippy::type_complexity)]
    fn f_test_reference_input() -> (Vec<String>, Vec<String>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]);
        let time = strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2", "3"]);
        let x1 = vec![1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 1.0, 3.0, 6.0, 4.0, 2.0, 1.0];
        let x2 = vec![2.0, 1.0, 4.0, 3.0, 2.0, 6.0, 5.0, 3.0, 1.0, 4.0, 2.0, 5.0];
        let y = vec![
            5.0, 7.0, 10.0, 3.0, 8.0, 9.0, 6.0, 10.0, 15.0, 2.0, 5.0, 4.0,
        ];
        (entity, time, x1, x2, y)
    }

    #[test]
    fn fe_estimator_fit_one_way_f_statistic_matches_linearmodels() {
        let (entity, _time, x1, x2, y) = f_test_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert_eq!(fe.df_resid(), 6); // n(12) - df_model(k(2)+n_entities(4))
        // `PanelOLS(y, [x1, x2], entity_effects=True).fit(cov_type="unadjusted",
        // debiased=True).f_statistic`: F(2,6)。
        assert!((fe.f_statistic().unwrap() - 4.544_331_119_544_591).abs() < 1e-9);
        assert!((fe.f_p_value().unwrap() - 0.062_878_408_429_192_23).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_f_statistic_matches_linearmodels() {
        let (entity, time, x1, x2, y) = f_test_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();

        assert_eq!(fe.df_resid(), 4); // n(12) - df_model(k(2)+neffects(4+3-1=6))
        // `PanelOLS(y, [x1, x2], entity_effects=True, time_effects=True).fit(
        // cov_type="unadjusted", debiased=True).f_statistic`: F(2,4)。
        assert!((fe.f_statistic().unwrap() - 9.574_202_321_891_761).abs() < 1e-9);
        assert!((fe.f_p_value().unwrap() - 0.029_859_178_280_428_5).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_returns_f_test_failed_for_extreme_scale_difference() {
        // x1は1e6オーダー、x2は1e-3オーダーとスケールが極端に異なる（x3は通常スケール）。
        // within変換後も3列は線形従属ではないため設計行列自体はフルランク
        // （`WithinRegressionFailed`にはならない）だが、傾き係数の同時共分散部分行列の条件数が
        // スケール比の2乗（≈1e18）相当となり倍精度の限界を超えるため、FE自身のF検定
        // （`wald_f_test`の`ensure_well_conditioned_symmetric_matrix`）が`FTestFailed`を返す。
        let n = 20;
        let entity: Vec<String> = (0..n).map(|i| format!("e{}", i % 5)).collect();
        let x1: Vec<f64> = (0..n).map(|i| 1e6 * (((i * 7) % 11) as f64)).collect();
        let x2: Vec<f64> = (0..n)
            .map(|i| 1e-3 * (((i * 5) % 13) as f64).powi(2))
            .collect();
        let x3: Vec<f64> = (0..n).map(|i| ((i * 3) % 7) as f64).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let noise = if i % 2 == 0 { 1e3 } else { -1e3 };
                2.0 * x1[i] + 3.0 * x2[i] + 0.5 * x3[i] + noise
            })
            .collect();
        let input = FeInput::from_columns(
            &y,
            &[x1, x2, x3],
            vec!["x1".to_string(), "x2".to_string(), "x3".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

        assert!(matches!(
            result.unwrap_err(),
            PanelError::FTestFailed {
                source: LeastSquaresError::Common(CommonError::NearSingularMatrix {
                    matrix: MatrixKind::CoefficientCovariance
                })
            }
        ));
    }

    // ── cov_type対応 ─────────────────────────────────────────────────────

    #[test]
    fn fe_estimator_fit_one_way_hc1_hc2_hc3_match_fixest_reference() {
        // 同じデータでの`fixest::feols(y ~ x | id)`の`vcov="HC1"/"HC2"/"HC3"`と
        // 数値比較する（linearmodelsはHC1相当の"robust"のみでHC2/HC3を提供しないため、
        // 3種とも`fixest`のみで検証する例外、`fe.rs`のモジュールdoc「`cov_type`対応」参照）。
        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let hc1 = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Hc1, 0.95).unwrap();
        assert!((*hc1.std_errors().get(0, 0) - 0.467_996_773_819_759).abs() < 1e-9);
        assert!((*hc1.test_stats().get(0, 0) - 2.997_409_076_837).abs() < 1e-6);
        assert!((*hc1.p_values().get(0, 0) - 0.020_015_356_643_180_1).abs() < 1e-6);
        // F統計量: k=1のため「1自由度のF検定は両側t検定と代数的に等価」
        // （`fe.rs`のモジュールdoc「自由度とF統計量」参照）。HC1のcov_paramsが正しく
        // wald_f_testに渡っていることの回帰ガード（classical以外のcov_typeでの唯一の
        // F統計量検証）。
        assert!((hc1.f_statistic().unwrap() - hc1.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((hc1.f_p_value().unwrap() - *hc1.p_values().get(0, 0)).abs() < 1e-9);

        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();
        let hc2 = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Hc2, 0.95).unwrap();
        assert!((*hc2.std_errors().get(0, 0) - 0.492_939_313_874_837).abs() < 1e-9);
        assert!((*hc2.test_stats().get(0, 0) - 2.845_741_328_178_91).abs() < 1e-6);
        assert!((*hc2.p_values().get(0, 0) - 0.024_839_464_368_821_2).abs() < 1e-6);
        assert!((hc2.f_statistic().unwrap() - hc2.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((hc2.f_p_value().unwrap() - *hc2.p_values().get(0, 0)).abs() < 1e-9);

        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();
        let hc3 = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Hc3, 0.95).unwrap();
        assert!((*hc3.std_errors().get(0, 0) - 0.687_184_240_890_824).abs() < 1e-9);
        assert!((*hc3.test_stats().get(0, 0) - 2.041_341_599_975_14).abs() < 1e-6);
        assert!((*hc3.p_values().get(0, 0) - 0.080_553_228_223_064_4).abs() < 1e-6);
        assert!((hc3.f_statistic().unwrap() - hc3.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((hc3.f_p_value().unwrap() - *hc3.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_hc1_hc2_hc3_match_fixest_reference() {
        // 同じデータでの`fixest::feols(y ~ x | id + t)`の`vcov="HC1"/"HC2"/"HC3"`と
        // 数値比較する。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();
        let hc1 = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Hc1, 0.95).unwrap();
        assert!((*hc1.std_errors().get(0, 0) - 0.205_876_715_757_555).abs() < 1e-9);
        assert!((hc1.f_statistic().unwrap() - hc1.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((hc1.f_p_value().unwrap() - *hc1.p_values().get(0, 0)).abs() < 1e-9);

        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();
        let hc2 = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Hc2, 0.95).unwrap();
        assert!((*hc2.std_errors().get(0, 0) - 0.304_984_723_480_691).abs() < 1e-9);
        assert!((hc2.f_statistic().unwrap() - hc2.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((hc2.f_p_value().unwrap() - *hc2.p_values().get(0, 0)).abs() < 1e-9);

        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();
        let hc3 = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Hc3, 0.95).unwrap();
        assert!((*hc3.std_errors().get(0, 0) - 0.737_275_671_443_649).abs() < 1e-9);
        assert!((hc3.f_statistic().unwrap() - hc3.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((hc3.f_p_value().unwrap() - *hc3.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_one_way_cluster_on_entity_matches_fixest_nested_k() {
        // 1-way FEでクラスター変数がentityと同じ（`groups: None`＝デフォルト）場合、
        // fixestの`feols(y~x|entity, cluster=~entity)`と数値一致する（`K.fixef=
        // "nonnested"`の既定分岐、entity FEが全次元ネスト、`fe_cluster_k_correction`の
        // docコメント参照。`K=df_model-n_entities+1=5-4+1=2`ではなく`K=k+1=2`——
        // ここでは`k=1`なので`K=2`）。期待値はRで独立に計算・検算済み
        // （`options(digits=16)`、実装時）。
        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster { groups: None },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.603_104_702_643_124_5).abs() < 1e-9);
        assert!((*fe.test_stats().get(0, 0) - 2.325_927_441_172_424).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.102_527_902_595_719_5).abs() < 1e-6);
        assert!((fe.f_statistic().unwrap() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value().unwrap() - *fe.p_values().get(0, 0)).abs() < 1e-9);
        // `f_df()`（カバレッジ監査で判明した未検証の単純getter。分母自由度が
        // `df_resid`から`df_inference`に変わったため、`df_resid`
        // （`n-df_model=12-5=7`）とは異なる`df_inference`（`G-1=n_entities-1=3`）を
        // 返すことを確認する）。
        assert_eq!(fe.df_resid(), 7);
        assert_eq!(fe.df_inference(), 3);
        assert_eq!(fe.f_df(), Some((1, 3)));
    }

    #[test]
    fn fe_estimator_fit_two_way_cluster_on_entity_matches_fixest_partially_nested_k() {
        // 2-way FEでentityクラスターは、entity次元だけがネストしtime次元はネストしない
        // 「部分ネスト」ケース（`fe_cluster_k_correction`のdocコメント参照）。
        // fixestの`feols(y~x|entity+time, cluster=~entity)`と数値一致する。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(
            input,
            FeEffects::TwoWay,
            FeCovType::Cluster { groups: None },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.158_754_475_018_330_4).abs() < 1e-9);
        assert!((*fe.test_stats().get(0, 0) - 5.180_514_794_604_026).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.013_962_411_468_796_73).abs() < 1e-6);
        assert!((fe.f_statistic().unwrap() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value().unwrap() - *fe.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_one_way_cluster_on_non_nested_variable_matches_fixest_full_k() {
        // 1-way FEでも、クラスター変数がentityと無関係（ここでは`time`）なら
        // どの次元もネストせず`K=df_model`（フルカウント、`fixef_dimension_nested_
        // within_cluster`がfalseになるケース）。fixestの
        // `feols(y~x|entity, cluster=~time)`と数値一致する。
        let (entity, time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&time)),
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.115_060_764_365_329_2).abs() < 1e-9);
        assert!((fe.f_statistic().unwrap() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value().unwrap() - *fe.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_cluster_propagates_insufficient_clusters_error() {
        // クラスター数2未満は`CommonError::InsufficientClusters`（`validate_cluster_groups`）
        // として伝播する。すべて同じクラスターに属する（g=1）データを使う。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 3.0, 2.0, 6.0, 4.0, 10.0];
        let single_cluster = strings(&["g", "g", "g", "g", "g", "g"]);
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&single_cluster)),
            },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::Common(CommonError::InsufficientClusters {
                setting: ClusterSetting::CovType,
                g: 1
            })
        );
    }

    #[test]
    fn fe_estimator_fit_cluster_propagates_insufficient_clusters_for_inference_error() {
        // クラスター数g(=2)が傾き係数の数q(=k=2)以下（g<=q、境界は厳密不等号）は
        // `CommonError::InsufficientClustersForInference`として伝播する
        // （`validate_cluster_count_covers_slopes`、`.claude/rules/testing-policy.md`
        // 「G<=qで書く」の方針）。g=2自体は`InsufficientClusters`（g<2）は回避している。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x1 = vec![1.0, 3.0, 2.0, 6.0, 4.0, 10.0];
        let x2 = vec![2.0, 5.0, 1.0, 9.0, 3.0, 7.0];
        let two_clusters = strings(&["1", "1", "1", "2", "2", "2"]);
        let input = FeInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&two_clusters)),
            },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::Common(CommonError::InsufficientClustersForInference { g: 2, q: 2 })
        );
    }

    // ── Driscoll-Kraay型パネルHAC対応 ───────────────────────────────────────

    #[test]
    fn fe_estimator_fit_one_way_hac_matches_fixest_default_bandwidth() {
        // fixestの`feols(y~x|entity, vcov="DK", panel.id=~entity+time)`と数値比較する
        // （5.1節、DKの主リファレンス）。n_periods=3のため既定バンド幅は
        // `floor(4*(3/100)^(2/9))=1`（`resolve_dk_bandwidth`）。期待値はRで独立に
        // 計算・検算済み（`options(digits=16)`、実装時）。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let dk_time = dk_time_from(&input);
        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: None,
                time: dk_time,
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.params().get(0, 0) - 1.402_777_777_777_78).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.112_778_272_530_122_7).abs() < 1e-9);
        assert!((*fe.test_stats().get(0, 0) - 12.438_369_078_610_43).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.006_401_580_635_206_468).abs() < 1e-9);
        assert!((*fe.conf_lower().get(0, 0) - 0.917_532_035_619_617).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.888_023_519_935_939).abs() < 1e-6);
        assert!((fe.f_statistic().unwrap() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value().unwrap() - *fe.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_uses_explicit_time_override_without_fe_input_time() {
        // `FeCovType::Dk.time`（明示指定）は`FeInput.time_keys()`を経由せずに
        // DK HACを成立させられる（`engine_pybind`の`FEOptions.dk_time`が1-way FE + DK HAC
        // の組み合わせをこの経路で配線する想定）。`FeInput::from_columns`には`time=None`を
        // 渡し、`fe_estimator_fit_one_way_hac_matches_fixest_default_bandwidth`と
        // 同じ結果になることを確認する（同じ`time`列を使っているため数値は完全一致する）。
        let (entity, time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: None,
                time: TimeKeys::lexicographic(time),
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.params().get(0, 0) - 1.402_777_777_777_78).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.112_778_272_530_122_7).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_ignores_fe_input_time() {
        // `FeInput.time_keys()`にも`time`があっても、DKの時点列は`FeCovType::Dk.time`だけから
        // 決まる（`FeInput.time_keys()`は2-wayの固定効果の時間次元専用、`fe.rs`のモジュールdoc
        // 「Driscoll-Kraay型パネルHAC」参照）ことを確認する。`FeInput.time_keys()`にわざと
        // 全観測が同一のダミー時点列を渡し、それが無視されて`FeCovType::Dk.time`の方の結果と
        // 一致することを確認する。
        let (entity, time, x, y) = fixest_reference_input();
        let dummy_time = strings(&["z", "z", "z", "z", "z", "z", "z", "z", "z", "z", "z", "z"]);
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&dummy_time),
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: None,
                time: TimeKeys::lexicographic(time),
            },
            0.95,
        )
        .unwrap();

        // `dummy_time`（全観測が同一時点）をそのまま使っていたら`t_periods=1`となり
        // `resolve_dk_bandwidth`が`InsufficientDkPeriods`で拒否する（優先順位が
        // 逆だった場合はこのテスト自体がエラーで失敗する）。使われている`time`
        // （`t_periods=3`）を使った場合の既知の値と一致することで、優先順位を確認する。
        assert!((*fe.std_errors().get(0, 0) - 0.112_778_272_530_122_7).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_dk_bandwidth_used_reflects_resolved_bandwidth() {
        // `t_periods=3`の既定バンド幅は`floor(4*(3/100)^(2/9))=1`。明示指定はその値、
        // `Dk`以外は`None`。
        let (entity, time, x, y) = fixest_reference_input();
        let bandwidth_used = |cov_type: FeCovType| {
            let input = FeInput::from_columns(
                &y,
                std::slice::from_ref(&x),
                vec!["x".to_string()],
                &entity,
                Some(&time),
                "y".into(),
            )
            .unwrap();
            FeEstimator::fit(input, FeEffects::OneWay, cov_type, 0.95)
                .unwrap()
                .dk_bandwidth_used()
        };
        let dk = |bandwidth: Option<i64>| FeCovType::Dk {
            bandwidth,
            time: TimeKeys::lexicographic(time.clone()),
        };

        assert_eq!(bandwidth_used(dk(None)), Some(1));
        assert_eq!(bandwidth_used(dk(Some(0))), Some(0));
        assert_eq!(bandwidth_used(dk(Some(2))), Some(2));
        assert_eq!(bandwidth_used(FeCovType::Classical), None);
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_with_explicit_bandwidth_matches_default() {
        // 既定バンド幅（n_periods=3 → 1）と明示的に`bandwidth=Some(1)`を指定した場合が
        // 一致することを確認する（`resolve_dk_bandwidth`のNone分岐とSome分岐が同じ値に
        // 解決されることの回帰ガード）。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let dk_time = dk_time_from(&input);
        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(1),
                time: dk_time,
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.112_778_272_530_122_7).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_with_bandwidth_two_scales_unchanged_kernel_by_new_correction() {
        // `bandwidth=Some(2)`（n_periods=3のため許容範囲`[0,3)`の上限）でラグ項ループ
        // （`for l in 1..=bandwidth`）が複数回（l=1,2）実行されるケースを検証する
        // （既定・`Some(1)`のテストはl=1の1回しか通らない。
        // `testing-policy.md`が警告する「ループ本体がテストで一度も複数回実行されない」
        // 落とし穴と同型）。
        //
        // **fixestの`vcov=DK(2)`とは意図的に数値比較しない**（devcontainer内のfixest
        // 0.14.2で実地確認済み）。本テストの`t_periods=3`・`bandwidth=2`は許容範囲
        // `[0, t_periods)`の上限ちょうど（`bandwidth == t_periods - 1`）で、この境界では
        // fixestが最後のラグ項（`l=bandwidth`）を落とした値を返し、標準のBartlettカーネルを
        // 実装した本実装と一致しない。原因はfixestのC++実装`cpp_driscoll_kraay`のoff-by-one
        // （ラグの個数`L = bandwidth + 1`に`L > T - 1`の上限を課すため、`bandwidth == T - 1`
        // のときだけ最大ラグが`T - 2`に切り詰められる）で、`bandwidth <= t_periods - 2`では
        // 一致する。本実装は標準カーネルを維持する（`docs/spec/fe-spec.md`3.3節7.参照）。
        // 境界の標準カーネルは`panel_driscoll_kraay_cov_params`の単体テスト（定義式の手計算）と
        // plmとの照合（`fe_plm_crosscheck.json`の`dk_max_bandwidth`）で検証している。
        //
        // このテスト自体は、変更していないカーネル計算（`bandwidth=1`の既定テストで
        // fixestと一致確認済みの実装）に新しい小標本補正（`(t_periods/(t_periods-1))×
        // ((n-1)/(n-K))`）が正しく適用されることを確認する回帰ガードとして残す
        // （期待値は本実装自身の出力を固定しただけで、外部リファレンスとの照合ではない）。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let dk_time = dk_time_from(&input);
        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(2),
                time: dk_time,
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.092_083_073_923_780_41).abs() < 1e-9);
        assert!((*fe.test_stats().get(0, 0) - 15.233_828_737_503_853).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.004_281_399_881_913_783).abs() < 1e-9);
        assert!((*fe.conf_lower().get(0, 0) - 1.006_576_288_395_902).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.798_979_267_159_653_4).abs() < 1e-6);
    }

    #[test]
    fn fe_estimator_fit_two_way_hac_matches_fixest_default_bandwidth() {
        // 同じデータでの2-way FE版（`entity+time`固定効果）。fixestの
        // `feols(y~x|entity+time, vcov="DK", panel.id=~entity+time)`と数値比較する。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let dk_time = dk_time_from(&input);
        let fe = FeEstimator::fit(
            input,
            FeEffects::TwoWay,
            FeCovType::Dk {
                bandwidth: None,
                time: dk_time,
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.params().get(0, 0) - 0.822_429_906_542_056).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.258_392_668_199_213_7).abs() < 1e-9);
        assert!((*fe.test_stats().get(0, 0) - 3.182_868_586_302_089).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.086_146_397_773_918_86).abs() < 1e-9);
        assert!((*fe.conf_lower().get(0, 0) - (-0.289_344_012_632_537_7)).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.934_203_825_716_65).abs() < 1e-6);
        assert!((fe.f_statistic().unwrap() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value().unwrap() - *fe.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_hac_with_bandwidth_two_scales_unchanged_kernel_by_new_correction() {
        // 1-way版（`fe_estimator_fit_one_way_hac_with_bandwidth_two_scales_unchanged_
        // kernel_by_new_correction`）と同様、2-way FEでもラグ項ループが複数回（l=1,2）
        // 実行されるケースを検証する。同テストのコメントの通り、
        // `bandwidth == t_periods - 1`（ここでは`2 == 3 - 1`）という境界値はfixestの
        // C++実装のoff-by-oneにより本実装と一致しないため（同テストのコメント参照）、
        // fixestとの数値比較はせず回帰ガードとして期待値を固定する。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let dk_time = dk_time_from(&input);
        let fe = FeEstimator::fit(
            input,
            FeEffects::TwoWay,
            FeCovType::Dk {
                bandwidth: Some(2),
                time: dk_time,
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.210_976_730_121_450_27).abs() < 1e-9);
        assert!((*fe.test_stats().get(0, 0) - 3.898_201_977_386_883).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.059_950_140_715_886_67).abs() < 1e-9);
        assert!((*fe.conf_lower().get(0, 0) - (-0.085_329_697_228_618_5)).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.730_189_510_312_731).abs() < 1e-6);
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_with_zero_bandwidth_matches_cluster_on_non_nested_time() {
        // `bandwidth=Some(0)`はラグ項なし（`Ŝ = Ŝ₀`）に退化し、これは`time`でクラスター
        // した場合（`fe_estimator_fit_one_way_cluster_on_non_nested_variable_matches_
        // fixest_full_k`と同じ`entity`/`time`）の`Ŝ`と数式的に同一になる（どちらも
        // `Σ_t (Σ_{i:time_i=t} x̃_i ε̂_i)(...)'`で、`K=df_model`（DKは常にフルカウント・
        // clusterもこのケースではどの次元もネストしないためフルカウント）・`G=t_periods`
        // （clusterの`G`も同じ`time`列のユニーク数）のスケールも一致する。`fe.rs`のモジュールdoc
        // 「Driscoll-Kraay型パネルHAC」参照）。2つの独立した実装
        // （`panel_cluster_cov_params`と`panel_driscoll_kraay_cov_params`）が同じ値に
        // 収束することを確認する回帰ガード（OLSの`fit_hac_with_zero_lags_matches_hc0`と
        // 同型）。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let dk_time = dk_time_from(&input);
        let hac = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: dk_time,
            },
            0.95,
        )
        .unwrap();

        let (entity, time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();
        let cluster = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&time)),
            },
            0.95,
        )
        .unwrap();

        assert!(
            (*hac.std_errors().get(0, 0) - *cluster.std_errors().get(0, 0)).abs() < 1e-9,
            "hac(bandwidth=0)={}, cluster(time)={}",
            *hac.std_errors().get(0, 0),
            *cluster.std_errors().get(0, 0)
        );
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_with_single_time_period_is_rejected() {
        // `t_periods=1`（全観測が同一の`time`ラベル）という退化した境界ケース
        // （`resolve_dk_bandwidth`のNone分岐が`bandwidth=t_periods`を
        // 返しうる唯一のケース）。
        //
        // DKの小標本補正を`(t_periods/(t_periods-1))×((n-1)/(n-K))`
        // （fixestの`ssc()`、`t_periods`をclusterの`G`と同じ役割で使う）に変更した結果、
        // `t_periods=1`は`t_periods/(t_periods-1)=1/0`が発散し計算が成立しなくなった
        // （クラスターの`G=1`が`validate_cluster_groups`で拒否されるのと同じ理由）。
        // `resolve_dk_bandwidth`が`PanelError::InsufficientDkPeriods`で早期に拒否する
        // （`G/(G-1)`型の補正は`t_periods=1`で未定義になる）。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let time = strings(&["1", "1", "1", "1", "1", "1"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 3.0, 2.0, 6.0, 4.0, 10.0];
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let dk_time = dk_time_from(&input);
        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: None,
                time: dk_time,
            },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::InsufficientDkPeriods { t_periods: 1 }
        );
    }

    /// 4エンティティ×3期間の1-way FE入力（説明変数は`x1`〜`x3`の先頭`k`個）。DKの
    /// `t_periods=3`に対し`k=3`（拒否）と`k=2`（成功）の境界を作るため。`T=2`は使わない:
    /// 1-wayのwithin変換では`x̃_i1 = -x̃_i2`・`ẽ_i1 = -ẽ_i2`となり時点スコアが
    /// `h_1 = h_2 = 0`に退化する（`rank(S) = 0`）ため、`t <= q`の規則を検証する入力として不適。
    fn three_period_input(k: usize, time: Option<&[String]>) -> FeInput {
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]);
        let columns = [
            vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 0.0, 2.0, 1.0, 3.0, 7.0, 4.0],
            vec![2.0, 1.0, 4.0, 0.0, 3.0, 1.0, 5.0, 2.0, 6.0, 1.0, 1.0, 3.0],
            vec![4.0, 2.0, 1.0, 3.0, 3.0, 5.0, 2.0, 6.0, 3.0, 0.0, 4.0, 2.0],
        ];
        let y = [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0, 5.0, 9.5, 4.0, 7.3];
        let names = ["x1", "x2", "x3"]
            .iter()
            .take(k)
            .map(|n| n.to_string())
            .collect();
        FeInput::from_columns(&y, &columns[..k], names, &entity, time, "y".into()).unwrap()
    }

    fn three_period_time() -> Vec<String> {
        strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2", "3"])
    }

    #[test]
    fn fe_estimator_fit_hac_rejects_periods_not_covering_slopes() {
        // `t_periods=3`ではDK共分散のrankが`t-1=2`以下で、F検定の傾き`k=3`個の部分行列が
        // 構造的に特異になる。`wald_f_test`の数値的な特異性判定を待たず弾く。
        let time = three_period_time();
        let dk_time = TimeKeys::lexicographic(time.clone());
        let result = FeEstimator::fit(
            three_period_input(3, Some(&time)),
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: dk_time,
            },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            PanelError::InsufficientDkPeriodsForInference { t_periods: 3, q: 3 }
                .in_stage(Stage::OverallFTest)
        );
    }

    #[test]
    fn fe_estimator_fit_hac_accepts_periods_one_above_slopes() {
        // 境界の成功パス: `t_periods=3 > k=2`。
        let time = three_period_time();
        let dk_time = TimeKeys::lexicographic(time.clone());
        let fe = FeEstimator::fit(
            three_period_input(2, Some(&time)),
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: dk_time,
            },
            0.95,
        )
        .unwrap();
        assert!(fe.f_statistic().unwrap().is_finite());
        assert!((0..2).all(|j| *fe.std_errors().get(j, 0) > 0.0));
    }

    #[test]
    fn fe_estimator_fit_hac_counts_periods_from_time_override() {
        // `FeCovType::Dk.time`の上書きがあれば、時点数はその列で数える（入力の`time`が
        // `None`の1-way FEでも同じ判定になる）。
        let result = FeEstimator::fit(
            three_period_input(3, None),
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: TimeKeys::lexicographic(three_period_time()),
            },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            PanelError::InsufficientDkPeriodsForInference { t_periods: 3, q: 3 }
                .in_stage(Stage::OverallFTest)
        );
    }

    /// 4エンティティ×2期間・`k=1`（`k = 0`なら説明変数なし）。`break_pattern`なら
    /// エンティティ`a`に時点`1`の観測を1つ足し（不均衡な1-way）、「全エンティティが
    /// 2時点に1観測ずつ」のパターンを崩す。
    fn two_period_input(k: usize, break_pattern: bool) -> (FeInput, Vec<String>) {
        let mut entity = strings(&["a", "a", "b", "b", "c", "c", "d", "d"]);
        let mut time = strings(&["1", "2", "1", "2", "1", "2", "1", "2"]);
        let mut x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0, 2.0];
        let mut y = vec![3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0];
        if break_pattern {
            entity.push("a".to_string());
            time.push("1".to_string());
            x.push(2.5);
            y.push(5.0);
        }
        let (columns, names) = if k == 0 {
            (vec![], vec![])
        } else {
            (vec![x], vec!["x".to_string()])
        };
        let input =
            FeInput::from_columns(&y, &columns, names, &entity, Some(&time), "y".into()).unwrap();
        (input, time)
    }

    /// `input`が持つ時点（ラベルの辞書順）を、DKの時点列としてそのまま渡す。
    fn dk_time_from(input: &FeInput) -> TimeKeys {
        let Some(time) = input.time_keys() else {
            panic!("test input must have a time column");
        };
        time.clone()
    }

    fn dk_bandwidth_zero(input: &FeInput) -> FeCovType {
        FeCovType::Dk {
            bandwidth: Some(0),
            time: dk_time_from(input),
        }
    }

    #[test]
    fn fe_estimator_fit_hac_rejects_degenerate_two_period_panel() {
        // 全エンティティが2時点に1観測ずつだと、within変換で時点スコアが`h_1 = h_2 = 0`に
        // 退化しDK共分散が恒等的にゼロになる（`k=1`でも`t > q`の検証は通ってしまう）。
        let (input, _) = two_period_input(1, false);
        let dk_cov = dk_bandwidth_zero(&input);
        let result = FeEstimator::fit(input, FeEffects::OneWay, dk_cov, 0.95);
        assert_eq!(result.unwrap_err(), PanelError::DegenerateDkTwoPeriods);
    }

    #[test]
    fn fe_estimator_fit_hac_rejects_degenerate_two_period_panel_two_way() {
        // 2-way（バランスパネルの二重デミーニング）でもエンティティ内の和がゼロになり同じ退化。
        let (input, _) = two_period_input(1, false);
        let dk_cov = dk_bandwidth_zero(&input);
        let result = FeEstimator::fit(input, FeEffects::TwoWay, dk_cov, 0.95);
        assert_eq!(result.unwrap_err(), PanelError::DegenerateDkTwoPeriods);
    }

    #[test]
    fn fe_estimator_fit_cluster_rejects_degenerate_two_group_split() {
        // 2時点のパネルを`time`でクラスタリング（`G=2 > k=1`の検証は通る）すると、
        // DKと同じ理由でクラスタースコアが恒等的にゼロになる。
        let (input, time) = two_period_input(1, false);
        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&time)),
            },
            0.95,
        );
        assert_eq!(result.unwrap_err(), PanelError::DegenerateClusterTwoGroups);
    }

    #[test]
    fn fe_estimator_fit_two_group_split_is_accepted_when_pattern_is_broken() {
        // 1エンティティでも同じ時点に2観測あればスコアは退化せず、標準誤差は正で有限。
        let (input, time) = two_period_input(1, true);
        let dk_cov = dk_bandwidth_zero(&input);
        let dk = FeEstimator::fit(input, FeEffects::OneWay, dk_cov, 0.95).unwrap();
        assert!(*dk.std_errors().get(0, 0) > 1e-8);
        assert!(dk.f_statistic().unwrap().is_finite());

        let (input, _) = two_period_input(1, true);
        let cluster = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&time)),
            },
            0.95,
        )
        .unwrap();
        assert!(*cluster.std_errors().get(0, 0) > 1e-8);
    }

    #[test]
    fn fe_estimator_fit_hac_two_period_panel_without_regressors_is_accepted() {
        // `k=0`（固定効果のみ）は標準誤差・F検定を計算しないため退化を問題にしない。
        let (input, _) = two_period_input(0, false);
        let dk_cov = dk_bandwidth_zero(&input);
        assert!(FeEstimator::fit(input, FeEffects::OneWay, dk_cov, 0.95).is_ok());
    }

    /// 4エンティティ×2観測・`k=1`の1-way FE入力で、入力の`time`列を`time`で与える
    /// （`Dk.time`上書きの経路を検証するため、時点ラベルの分け方だけを変えられるようにする）。
    fn four_by_two_input(time: &[&str]) -> FeInput {
        let entity = strings(&["a", "a", "b", "b", "c", "c", "d", "d"]);
        let x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0, 2.0];
        let y = [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0];
        FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&strings(time)),
            "y".into(),
        )
        .unwrap()
    }

    /// 3時点に分かれるが各エンティティ2観測のラベル（`t_periods=3`、退化しない）。
    const THREE_PERIOD_LABELS: [&str; 8] = ["1", "2", "2", "3", "3", "1", "1", "2"];

    const TWO_PERIOD_LABELS: [&str; 8] = ["1", "2", "1", "2", "1", "2", "1", "2"];

    #[test]
    fn fe_estimator_fit_hac_uses_resolved_dk_time_for_degeneracy_check() {
        // 入力の`time`は3時点だが、`Dk.time`上書きが2時点の退化パターン→拒否。
        let result = FeEstimator::fit(
            four_by_two_input(&THREE_PERIOD_LABELS),
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: lex_time(&TWO_PERIOD_LABELS),
            },
            0.95,
        );
        assert_eq!(result.unwrap_err(), PanelError::DegenerateDkTwoPeriods);

        // 逆に入力の`time`が2時点の退化パターンでも、上書きが3時点なら通す。
        let fe = FeEstimator::fit(
            four_by_two_input(&TWO_PERIOD_LABELS),
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: lex_time(&THREE_PERIOD_LABELS),
            },
            0.95,
        )
        .unwrap();
        assert!(*fe.std_errors().get(0, 0) > 1e-8);
    }

    #[test]
    fn fe_estimator_fit_cluster_rejects_degenerate_split_by_non_time_column() {
        // `time`以外の任意のクラスター列でも、全エンティティが2クラスターに1観測ずつなら退化。
        let groups = strings(&["g2", "g1", "g1", "g2", "g2", "g1", "g1", "g2"]);
        let result = FeEstimator::fit(
            four_by_two_input(&THREE_PERIOD_LABELS),
            FeEffects::OneWay,
            FeCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
            0.95,
        );
        assert_eq!(result.unwrap_err(), PanelError::DegenerateClusterTwoGroups);
    }

    #[test]
    fn fe_estimator_fit_cluster_two_group_split_without_regressors_is_accepted() {
        // `k=0`は標準誤差・F検定を計算しないため、Clusterでも退化を問題にしない。
        let (input, time) = two_period_input(0, false);
        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&time)),
            },
            0.95,
        );
        assert!(result.is_ok());
    }

    /// エンティティ2つ×5時点・`k=1`。2-wayではtime方向の各水準（時点）が2観測になる。
    fn two_entity_input() -> FeInput {
        let entity = strings(&["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]);
        let time = strings(&["1", "2", "3", "4", "5", "1", "2", "3", "4", "5"]);
        let x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 2.0, 1.0, 4.0, 3.0, 6.0];
        let y = [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0, 5.0, 9.5];
        FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap()
    }

    #[test]
    fn fe_estimator_fit_two_way_cluster_by_entity_rejects_two_entity_panel() {
        // 2-wayのwithin変換は各時点内でも和をゼロにするため、エンティティ2つのパネルを
        // entityでクラスタリング（`groups: None`＝既定）すると、全時点が2クラスターに
        // 1観測ずつになりクラスタースコアが恒等的にゼロになる（time方向の退化）。
        let result = FeEstimator::fit(
            two_entity_input(),
            FeEffects::TwoWay,
            FeCovType::Cluster { groups: None },
            0.95,
        );
        assert_eq!(result.unwrap_err(), PanelError::DegenerateClusterTwoGroups);
    }

    #[test]
    fn fe_estimator_fit_two_way_hac_rejects_two_entity_split_override() {
        // 同じtime方向の退化は、`Dk.time`にentityと同じ分け方の2水準列を渡した場合にも起きる。
        let result = FeEstimator::fit(
            two_entity_input(),
            FeEffects::TwoWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: lex_time(&["p", "p", "p", "p", "p", "q", "q", "q", "q", "q"]),
            },
            0.95,
        );
        assert_eq!(result.unwrap_err(), PanelError::DegenerateDkTwoPeriods);
    }

    #[test]
    fn fe_estimator_fit_one_way_cluster_by_entity_accepts_two_entity_panel() {
        // 1-wayはtime方向に和をゼロにしないため、同じデータのentityクラスタリングは退化しない。
        let fe = FeEstimator::fit(
            two_entity_input(),
            FeEffects::OneWay,
            FeCovType::Cluster { groups: None },
            0.95,
        )
        .unwrap();
        assert!(*fe.std_errors().get(0, 0) > 1e-8);
    }

    #[test]
    fn fe_estimator_fit_hac_rejects_time_override_with_wrong_length() {
        // `FeCovType::Dk.time`の上書き列は`FeInput`の検証を通らないため、`fit()`が長さを
        // 検証する（Clusterの`groups`と同じ水準）。
        let (entity, time, x, y) = fixest_reference_input();
        let n = y.len();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();
        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: TimeKeys::lexicographic(time[..n - 1].to_vec()),
            },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                y_rows: n,
                other_rows: n - 1,
            }
        );
    }

    #[test]
    fn fe_estimator_fit_hac_rejects_bandwidth_out_of_range() {
        // n_periods=3のため`bandwidth`の許容範囲は`[0, 3)`。`bandwidth=3`（`t`自体）は
        // 範囲外（OLSの`hac_lags`と同型の`[0, n)`境界、`t`版）。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let dk_time = dk_time_from(&input);
        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(3),
                time: dk_time,
            },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::InvalidDkBandwidth { bandwidth: 3, t: 3 }
        );
    }

    #[test]
    fn fe_estimator_fit_hac_rejects_negative_bandwidth() {
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let dk_time = dk_time_from(&input);
        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(-1),
                time: dk_time,
            },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::InvalidDkBandwidth {
                bandwidth: -1,
                t: 3
            }
        );
    }

    #[test]
    fn fe_estimator_fit_pins_faer_global_parallelism_to_seq() {
        // `fit()`冒頭の`crate::shared::parallelism::ensure_serial()`がfaerのグローバル
        // 並列度を`Par::Seq`へ引き戻すことの回帰ガード（panel系統代表、
        // `engine/src/panel/CLAUDE.md`「faerのグローバル並列度」参照）。
        faer::set_global_parallelism(faer::Par::rayon(0));

        let entity = strings(&["a", "a", "b", "b"]);
        let y = [1.0, 2.2, 3.0, 4.6];
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let _ = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert!(matches!(faer::get_global_parallelism(), faer::Par::Seq));
    }

    #[test]
    fn fe_estimator_fit_with_no_regressors_estimates_fixed_effects_only_model() {
        // k=0（回帰変数なし、固定効果のみのモデル）でも`fit()`本体がエンドツーエンドに
        // 動くことを確認する境界ケース（`from_columns_with_no_regressors_succeeds`は
        // `FeInput`構築のみの検証で、`fit()`のdf調整（`df_model=neffects`のみになる）・
        // パネル固有R²/`aic`/`bic`計算までは通さない）。
        // n=6・n_entities=3・k=0でdf_model=neffects=3・df_resid=3。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let y = [1.0, 3.0, 5.0, 7.0, 2.0, 4.0];
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert_eq!(fe.df_model(), 3);
        assert_eq!(fe.df_resid(), 3);
        assert_eq!(fe.params().nrows(), 0);
        assert_eq!(fe.std_errors().nrows(), 0);
        assert!(fe.aic().is_finite());
        assert!(fe.bic().is_finite());
        assert!(fe.r_squared_within().is_finite());
        assert!(fe.r_squared_between().is_finite());
        assert!(fe.r_squared_overall().is_finite());
    }

    /// `fit`レベルのproperty-basedテスト（固定シナリオとは別に、加法シフト・スケール・行の並べ替えの
    /// 不変性などを、ランダムなパネルで検証する）。
    mod proptests {
        use super::*;
        use crate::linear::ols::{CovType, OlsEstimator};
        use proptest::collection;
        use proptest::prelude::*;

        /// `fit`レベルのproperty-basedテスト用のランダムパネル。1-way（アンバランスを含む）と
        /// 2-way（バランスのみ、`fe-spec.md`3.1節）を`effects`で切り替える。行は
        /// entity外側・time内側の順で並ぶ（並べ替えは`keys`で別途行う）。
        #[derive(Debug, Clone)]
        struct FeCase {
            effects: FeEffects,
            entity_idx: Vec<usize>,
            time_idx: Vec<usize>,
            y: Vec<f64>,
            x: Vec<Vec<f64>>,
            /// 行の並べ替え用の乱数キー（長さ`n`）。
            keys: Vec<u64>,
            /// entity/timeごとの加法シフト（`y`に足して結果が不変であることの検証用）。
            entity_shift: Vec<f64>,
            time_shift: Vec<f64>,
        }

        const MAX_PERIODS: usize = 6;

        fn fe_case_strategy() -> impl Strategy<Value = FeCase> {
            (any::<bool>(), 4..=6usize, 3..=MAX_PERIODS, 1..=2usize)
                .prop_flat_map(|(two_way, n_entities, n_periods, k)| {
                    let sizes = if two_way {
                        Just(vec![n_periods; n_entities]).boxed()
                    } else {
                        collection::vec(2..=MAX_PERIODS, n_entities).boxed()
                    };
                    (Just(two_way), Just(k), sizes)
                })
                .prop_flat_map(|(two_way, k, sizes)| {
                    let n: usize = sizes.iter().sum();
                    let n_entities = sizes.len();
                    (
                        Just(two_way),
                        Just(sizes),
                        collection::vec(collection::vec(-10.0f64..10.0, n), k),
                        collection::vec(-5.0f64..5.0, n),
                        collection::vec(any::<u64>(), n),
                        collection::vec(-50.0f64..50.0, n_entities),
                        collection::vec(-50.0f64..50.0, MAX_PERIODS),
                    )
                })
                .prop_map(
                    |(two_way, sizes, x, noise, keys, entity_shift, time_shift)| {
                        let mut entity_idx = Vec::new();
                        let mut time_idx = Vec::new();
                        for (i, &size) in sizes.iter().enumerate() {
                            for t in 0..size {
                                entity_idx.push(i);
                                time_idx.push(t);
                            }
                        }
                        let y: Vec<f64> = (0..noise.len())
                            .map(|r| x.iter().map(|c| c[r]).sum::<f64>() + noise[r])
                            .collect();
                        FeCase {
                            effects: if two_way {
                                FeEffects::TwoWay
                            } else {
                                FeEffects::OneWay
                            },
                            entity_idx,
                            time_idx,
                            y,
                            x,
                            keys,
                            entity_shift,
                            time_shift,
                        }
                    },
                )
        }

        fn fe_cov_strategy() -> impl Strategy<Value = FeCovType> {
            prop_oneof![
                Just(FeCovType::Classical),
                Just(FeCovType::Hc1),
                Just(FeCovType::Hc2),
                Just(FeCovType::Hc3),
                Just(FeCovType::Cluster { groups: None }),
            ]
        }

        fn labels(prefix: &str, idx: &[usize]) -> Vec<String> {
            idx.iter().map(|i| format!("{prefix}{i}")).collect()
        }

        /// 傾き係数と標準誤差（`x`の列順）。推定に失敗したら`None`。
        fn fit_slopes(
            case: &FeCase,
            y: &[f64],
            x: &[Vec<f64>],
            entity: &[String],
            time: &[String],
            cov: FeCovType,
        ) -> Option<(Vec<f64>, Vec<f64>)> {
            let names: Vec<String> = (0..x.len()).map(|j| format!("x{j}")).collect();
            let input =
                FeInput::from_columns(y, x, names, entity, Some(time), "y".to_string()).ok()?;
            let est = FeEstimator::fit(input, case.effects, cov, 0.95).ok()?;
            let k = x.len();
            let params = (0..k).map(|j| *est.params().get(j, 0)).collect();
            let se = (0..k).map(|j| *est.std_errors().get(j, 0)).collect();
            Some((params, se))
        }

        fn fit_case(case: &FeCase, cov: FeCovType) -> Option<(Vec<f64>, Vec<f64>)> {
            fit_slopes(
                case,
                &case.y,
                &case.x,
                &labels("e", &case.entity_idx),
                &labels("t", &case.time_idx),
                cov,
            )
        }

        /// 固定効果推定の数値誤差（within変換・QR）を考慮し、固定フィクスチャ比較より緩めた
        /// 相対誤差（`ols/fit.rs`のproptestと同じ方針）。
        fn assert_approx_eq(actual: f64, expected: f64, msg: &str) {
            let tol = 1e-6 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tol,
                "{msg}: actual={actual}, expected={expected}, tol={tol}"
            );
        }

        fn assert_all_approx_eq(actual: &[f64], expected: &[f64], msg: &str) {
            assert_eq!(actual.len(), expected.len());
            for (j, (a, e)) in actual.iter().zip(expected).enumerate() {
                assert_approx_eq(*a, *e, &format!("{msg}[{j}]"));
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            /// yにentity定数（2-wayはtime定数も）を加えても、傾き・SEは変わらない
            /// （固定効果が吸収するため）。
            #[test]
            fn slopes_and_se_are_invariant_to_additive_fixed_effects_in_y(
                case in fe_case_strategy(),
                cov in fe_cov_strategy(),
            ) {
                let base = fit_case(&case, cov.clone());
                prop_assume!(base.is_some());
                let (params, se) = base.unwrap();

                let y_shifted: Vec<f64> = (0..case.y.len())
                    .map(|r| {
                        let time_part = if case.effects == FeEffects::TwoWay {
                            case.time_shift[case.time_idx[r]]
                        } else {
                            0.0
                        };
                        case.y[r] + case.entity_shift[case.entity_idx[r]] + time_part
                    })
                    .collect();
                let shifted = fit_slopes(
                    &case,
                    &y_shifted,
                    &case.x,
                    &labels("e", &case.entity_idx),
                    &labels("t", &case.time_idx),
                    cov,
                );
                prop_assume!(shifted.is_some());
                let (params2, se2) = shifted.unwrap();

                assert_all_approx_eq(&params2, &params, "params");
                assert_all_approx_eq(&se2, &se, "se");
            }

            /// 行の並べ替えとentityラベルの付け替え（順序を反転した別名）で結果は変わらない。
            #[test]
            fn results_are_invariant_to_row_order_and_entity_relabeling(
                case in fe_case_strategy(),
                cov in fe_cov_strategy(),
            ) {
                let base = fit_case(&case, cov.clone());
                prop_assume!(base.is_some());
                let (params, se) = base.unwrap();

                let n = case.y.len();
                let n_entities = case.entity_shift.len();
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by_key(|&r| case.keys[r]);
                let y: Vec<f64> = order.iter().map(|&r| case.y[r]).collect();
                let x: Vec<Vec<f64>> = case
                    .x
                    .iter()
                    .map(|c| order.iter().map(|&r| c[r]).collect())
                    .collect();
                let entity: Vec<String> = order
                    .iter()
                    .map(|&r| format!("z{}", n_entities - 1 - case.entity_idx[r]))
                    .collect();
                let time: Vec<String> = order
                    .iter()
                    .map(|&r| format!("t{}", case.time_idx[r]))
                    .collect();

                let permuted = fit_slopes(&case, &y, &x, &entity, &time, cov);
                prop_assume!(permuted.is_some());
                let (params2, se2) = permuted.unwrap();

                assert_all_approx_eq(&params2, &params, "params");
                assert_all_approx_eq(&se2, &se, "se");
            }

            /// yをc倍すると傾き・標準誤差はそれぞれc倍・|c|倍になる。
            #[test]
            fn slopes_and_se_scale_with_y(
                case in fe_case_strategy(),
                cov in fe_cov_strategy(),
                c in prop_oneof![-10.0f64..-0.1, 0.1f64..10.0],
            ) {
                let base = fit_case(&case, cov.clone());
                prop_assume!(base.is_some());
                let (params, se) = base.unwrap();

                let y_scaled: Vec<f64> = case.y.iter().map(|v| v * c).collect();
                let scaled = fit_slopes(
                    &case,
                    &y_scaled,
                    &case.x,
                    &labels("e", &case.entity_idx),
                    &labels("t", &case.time_idx),
                    cov,
                );
                prop_assume!(scaled.is_some());
                let (params2, se2) = scaled.unwrap();

                let expected_params: Vec<f64> = params.iter().map(|p| p * c).collect();
                let expected_se: Vec<f64> = se.iter().map(|s| s * c.abs()).collect();
                assert_all_approx_eq(&params2, &expected_params, "params");
                assert_all_approx_eq(&se2, &expected_se, "se");
            }

            /// LSDV（entityダミー、2-wayはtimeダミーも加えた定数項付きOLS）と傾き・Classical SEが
            /// 一致する。LSDV側のOLS自由度`n-(k+N+T-1)`はFEのパネル自由度調整（`neffects`）と
            /// 一致するため、SEも同じ値になる。
            #[test]
            fn slopes_and_classical_se_match_lsdv_oracle(case in fe_case_strategy()) {
                let fe = fit_case(&case, FeCovType::Classical);
                prop_assume!(fe.is_some());
                let (params, se) = fe.unwrap();

                let k = case.x.len();
                let n = case.y.len();
                let n_entities = case.entity_shift.len();
                let mut x = case.x.clone();
                for i in 1..n_entities {
                    x.push((0..n).map(|r| f64::from(case.entity_idx[r] == i)).collect());
                }
                if case.effects == FeEffects::TwoWay {
                    let n_periods = *case.time_idx.iter().max().unwrap() + 1;
                    for t in 1..n_periods {
                        x.push((0..n).map(|r| f64::from(case.time_idx[r] == t)).collect());
                    }
                }
                let names: Vec<String> = (0..x.len()).map(|j| format!("x{j}")).collect();
                let input = OlsInput::from_columns(&case.y, &x, names, true, "y".to_string()).unwrap();
                let ols = OlsEstimator::fit(input, CovType::Classical, 0.95);
                prop_assume!(ols.is_ok());
                let ols = ols.unwrap();

                // 定数項が先頭に来るため、傾きは添字1..=k。
                let ols_params: Vec<f64> = (1..=k).map(|j| *ols.params().get(j, 0)).collect();
                let ols_se: Vec<f64> = (1..=k).map(|j| *ols.std_errors().get(j, 0)).collect();
                assert_all_approx_eq(&params, &ols_params, "params");
                assert_all_approx_eq(&se, &ols_se, "se");
            }
        }
    }

    fn dk_std_error(time: TimeKeys, rows: Option<&[usize]>) -> f64 {
        let (y, x, entity, period) = dk_panel_12();
        let rows: Vec<usize> = rows.map_or_else(|| (0..y.len()).collect(), |r| r.to_vec());
        let pick = |v: &[f64]| rows.iter().map(|&i| v[i]).collect::<Vec<_>>();
        let entity: Vec<String> = rows.iter().map(|&i| entity[i].clone()).collect();
        let _ = period;
        let input = FeInput::from_columns(
            &pick(&y),
            &[pick(&x)],
            vec!["x".into()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();
        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(3),
                time,
            },
            0.95,
        )
        .unwrap();
        *fe.std_errors().get(0, 0)
    }

    #[test]
    fn dk_standard_errors_follow_the_value_order_of_integer_periods() {
        let (_, _, _, period) = dk_panel_12();
        // 辞書順が時間順と一致するゼロ埋めラベル（基準）。
        let padded = integer_time_keys(&period, |t| format!("{t:03}"));
        // ゼロ埋めなしの整数ラベルを、数値順で並べる。
        let numeric = integer_time_keys(&period, |t| t.to_string());
        // ゼロ埋めなしのラベルを辞書順で並べる（`1, 10, 11, 2, ...`、誤った順序）。
        let lexicographic = TimeKeys::lexicographic(period.iter().map(|t| t.to_string()).collect());

        let expected = dk_std_error(padded, None);
        let got = dk_std_error(numeric, None);
        let wrong = dk_std_error(lexicographic, None);

        assert!(
            (got - expected).abs() < 1e-12,
            "numeric {got} vs padded {expected}"
        );
        assert!(
            (wrong - expected).abs() > 1e-6,
            "the lexicographic order of unpadded labels must differ: {wrong} vs {expected}"
        );
    }

    #[test]
    fn dk_standard_errors_do_not_depend_on_row_order() {
        let (_, _, _, period) = dk_panel_12();
        let n = period.len();
        let reversed: Vec<usize> = (0..n).rev().collect();
        let shuffled: Vec<usize> = (0..n).map(|i| (i * 7 + 3) % n).collect();
        assert_eq!(
            {
                let mut s = shuffled.clone();
                s.sort();
                s
            },
            (0..n).collect::<Vec<_>>(),
            "the shuffle must be a permutation"
        );
        let keys = |rows: &[usize]| {
            let p: Vec<usize> = rows.iter().map(|&i| period[i]).collect();
            integer_time_keys(&p, |t| t.to_string())
        };

        let base = dk_std_error(keys(&(0..n).collect::<Vec<_>>()), None);
        for rows in [&reversed, &shuffled] {
            let got = dk_std_error(keys(rows), Some(rows));
            assert!((got - base).abs() < 1e-12, "{got} vs {base}");
        }
    }

    /// 1-way・2-wayの全`cov_type`で、`cov_params()`が対称で、対角の平方根が`std_errors()`と
    /// 一致する。次元は`(k, k)`（固定効果の係数は含まない）。
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

        for effects in [FeEffects::OneWay, FeEffects::TwoWay] {
            let cov_types = [
                FeCovType::Classical,
                FeCovType::Hc1,
                FeCovType::Hc2,
                FeCovType::Hc3,
                FeCovType::Cluster { groups: None },
                FeCovType::Dk {
                    bandwidth: Some(1),
                    time: TimeKeys::lexicographic(time.clone()),
                },
            ];
            for cov_type in cov_types {
                let name = format!("{effects:?}/{cov_type:?}");
                let input = FeInput::from_columns(
                    &y,
                    &[x1.clone(), x2.clone()],
                    vec!["x1".to_string(), "x2".to_string()],
                    &entity,
                    Some(&time),
                    "y".to_string(),
                )
                .unwrap();
                let fe = FeEstimator::fit(input, effects, cov_type, 0.95).unwrap();

                let cov = fe.cov_params();
                assert_eq!((cov.nrows(), cov.ncols()), (2, 2), "{name}: shape");
                for i in 0..2 {
                    for j in 0..2 {
                        let scale = (*cov.get(i, i) * *cov.get(j, j)).sqrt();
                        assert!(
                            (*cov.get(i, j) - *cov.get(j, i)).abs() <= 1e-12 * scale,
                            "{name}: cov_params not symmetric at ({i},{j})"
                        );
                    }
                    let se = *fe.std_errors().get(i, 0);
                    assert!(
                        (cov.get(i, i).sqrt() - se).abs() <= 1e-12 * se,
                        "{name}: sqrt(cov_params[{i}][{i}]) != std_errors[{i}]"
                    );
                }
            }
        }
    }
}
