//! `fit`の結果を参照実装（statsmodels・linearmodels・plm・R）や閉形式・独立再計算と照合するテスト。

use crate::panel::common::TimeKeys;
use crate::panel::re::ReCovType;
use crate::panel::re::ReEffects;
use crate::panel::re::ReEstimator;
use crate::panel::re::ReInput;
use crate::panel::re::test_support::*;
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
fn re_estimator_fit_matches_linearmodels_reference() {
    // `swamy_arora_variance_components_matches_linearmodels_reference`と同じ
    // データ（entity a: T=3, b: T=2, c: T=2）。`linearmodels.RandomEffects`
    // （Python、`exog=[const, x1]`）で実地検証済みの`params`・`resids`と比較する。
    let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
    let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
    let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
    let input = ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
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
