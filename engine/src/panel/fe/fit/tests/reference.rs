//! `fit`の結果を参照実装（statsmodels・linearmodels・plm・R）や閉形式・独立再計算と照合するテスト。

use super::dk_time_from;
use crate::panel::fe::FeEstimator;
use crate::panel::fe::input::FeCovType;
use crate::panel::fe::input::FeEffects;
use crate::panel::fe::input::FeInput;
use crate::panel::fe::test_support::*;
use crate::shared::group_codes::GroupCodes;

// ── 全体F統計量（linearmodelsとの照合） ──────────────────────────────

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
fn fe_estimator_fit_one_way_matches_fixest_reference() {
    // Rの`fixest::feols(y ~ x | id)`（`id`: a/b/c/d各3観測）の実測値と数値比較する
    // （5.2節・fixestがFEのRクロスチェック参照実装）。期待値はRで独立に計算・検算済み
    // （`options(digits=15)`でフルの浮動小数点精度を取得）。
    let (entity, _time, x, y) = fixest_reference_input();
    let input =
        FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into()).unwrap();

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

// ── HC1・HC2・HC3・クラスターのfixest照合 ─────────────────────────────

#[test]
fn fe_estimator_fit_one_way_hc1_hc2_hc3_match_fixest_reference() {
    // 同じデータでの`fixest::feols(y ~ x | id)`の`vcov="HC1"/"HC2"/"HC3"`と
    // 数値比較する（linearmodelsはHC1相当の"robust"のみでHC2/HC3を提供しないため、
    // 3種とも`fixest`のみで検証する例外、`fe.rs`のモジュールdoc「`cov_type`対応」参照）。
    let (entity, _time, x, y) = fixest_reference_input();
    let input =
        FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into()).unwrap();

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
        FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into()).unwrap();
    let hc2 = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Hc2, 0.95).unwrap();
    assert!((*hc2.std_errors().get(0, 0) - 0.492_939_313_874_837).abs() < 1e-9);
    assert!((*hc2.test_stats().get(0, 0) - 2.845_741_328_178_91).abs() < 1e-6);
    assert!((*hc2.p_values().get(0, 0) - 0.024_839_464_368_821_2).abs() < 1e-6);
    assert!((hc2.f_statistic().unwrap() - hc2.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
    assert!((hc2.f_p_value().unwrap() - *hc2.p_values().get(0, 0)).abs() < 1e-9);

    let (entity, _time, x, y) = fixest_reference_input();
    let input =
        FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into()).unwrap();
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
        FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into()).unwrap();

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
        FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into()).unwrap();

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

// ── Driscoll-Kraay型パネルHACのfixest照合 ─────────────────────────────

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
