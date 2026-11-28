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
fn re_estimator_fit_matches_plm_reference_on_unbalanced_panel() {
    // `swamy_arora_variance_components_matches_plm_on_unbalanced_panel`と同じ
    // データ（entity a: T=3, b: T=2, c: T=2）。`plm::plm(y ~ x1, model = "random")`
    // （`random.method = "swar"`、不均衡なので`dfcor = 3`のトレース形）の係数・残差
    // （`resid(m)`、準偏差変換後の回帰の残差）と比較する。`linearmodels.RandomEffects`は
    // 不均衡パネルでσ_u²の式が違う（`re-spec.md`3.1節）ため、ここでは参照にしない。
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
    assert!((*re.estimator().params().get(0, 0) - 2.223_314_140_197_062).abs() < 1e-9);
    assert!((*re.estimator().params().get(1, 0) - 1.400_738_076_181_155).abs() < 1e-9);

    let expected_resids = [
        0.008_597_936_748_064_805,
        -0.392_140_139_433_091_7,
        -0.193_616_291_795_401,
        0.982_284_062_826_489_2,
        0.581_545_986_645_334_1,
        -1.842_740_467_568_014_1,
        0.756_521_456_250_830_8,
    ];
    for (i, expected) in expected_resids.iter().enumerate() {
        let actual = *re.estimator().residuals().get(i, 0);
        assert!((actual - expected).abs() < 1e-9, "{actual} vs {expected}");
    }

    // `df_resid = n - k`・`df_model = k`（`re-spec.md`3.3節。`plm`の`df.residual`と同じ）。
    // n=7、k=2（const+x1）。
    assert_eq!(re.df_resid(), 5);
    assert_eq!(re.df_model(), 2);

    // `estimator()`のdocコメントは「`std_errors`/`test_stats`/`p_values`/`conf_lower`/
    // `conf_upper`/`aic`/`bic`は既に正しいRE推定量になっている」と述べるため、
    // `plm`の`vcov(m)`の標準誤差と、そこから本実装の規約（t分布、自由度`n - k = 5`）で
    // 計算したt値・p値・信頼区間、準偏差変換後の残差から計算した`aic`/`bic`と照合する
    // （`re.rs`のモジュールdoc「`ReEstimator::fit`のパイプライン」参照）。
    let expected_std_errors = [2.049_735_627_797_357_3, 0.404_387_523_099_868_7];
    let expected_test_stats = [1.084_683_366_013_514_7, 3.463_850_876_119_203_7];
    let expected_p_values = [0.327_569_982_746_242_5, 0.017_967_728_737_767_356];
    let expected_conf_lower = [-3.045_699_032_475_420_5, 0.361_226_854_742_672_24];
    let expected_conf_upper = [7.492_327_312_869_545, 2.440_249_297_619_638];
    for j in 0..2 {
        assert!(
            (*re.estimator().std_errors().get(j, 0) - expected_std_errors[j]).abs() < 1e-9,
            "std_errors[{j}]"
        );
        assert!(
            (*re.estimator().test_stats().get(j, 0) - expected_test_stats[j]).abs() < 1e-9,
            "test_stats[{j}]"
        );
        assert!(
            (*re.estimator().p_values().get(j, 0) - expected_p_values[j]).abs() < 1e-9,
            "p_values[{j}]"
        );
        assert!(
            (*re.estimator().conf_lower().get(j, 0) - expected_conf_lower[j]).abs() < 1e-9,
            "conf_lower[{j}]"
        );
        assert!(
            (*re.estimator().conf_upper().get(j, 0) - expected_conf_upper[j]).abs() < 1e-9,
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
            (*re.std_errors().get(j, 0) - expected_std_errors[j]).abs() < 1e-9,
            "re.std_errors[{j}]"
        );
        assert!(
            (*re.test_stats().get(j, 0) - expected_test_stats[j]).abs() < 1e-9,
            "re.test_stats[{j}]"
        );
        assert!(
            (*re.p_values().get(j, 0) - expected_p_values[j]).abs() < 1e-9,
            "re.p_values[{j}]"
        );
        assert!(
            (*re.conf_lower().get(j, 0) - expected_conf_lower[j]).abs() < 1e-9,
            "re.conf_lower[{j}]"
        );
        assert!(
            (*re.conf_upper().get(j, 0) - expected_conf_upper[j]).abs() < 1e-9,
            "re.conf_upper[{j}]"
        );
    }
    assert_eq!(re.cov_type(), &ReCovType::Classical);

    // `plm`の残差の`loglik = -9.064512282972062`から手計算した参照値。
    assert!((re.estimator().aic() - 22.129_024_565_944_125).abs() < 1e-9);
    assert!((re.estimator().bic() - 22.020_844_864_054_75).abs() < 1e-9);

    // F統計量（2.1節）はWald二次形式のため、傾き係数が1個（q=1）のこのデータでは
    // 「1自由度のF検定は両側t検定と代数的に等価」（`f_statistic = test_stat²`、
    // `f_p_value = p_value`）が成り立つ。`estimator().f_statistic().unwrap()`（定数項も検定に
    // 含めてしまい誤り）とは異なる正しい値であることを確認する。
    assert!((re.f_statistic().unwrap() - re.test_stats().get(1, 0).powi(2)).abs() < 1e-9);
    assert!((re.f_p_value().unwrap() - *re.p_values().get(1, 0)).abs() < 1e-9);

    // パネル固有R²（`re-spec.md`3.6節）は外部の参照が無く、`plm`の係数からRで定義どおりに
    // 計算した値と比べる（betweenは`T_i`重み・観測で重み付けした全体平均）。
    // `r_squared_between`が負値になる（入れ子モデル比較の保証が無いR²の定義のため）ことも含む。
    assert!((re.r_squared_within() - 0.793_863_618_456_671_9).abs() < 1e-9);
    assert!((re.r_squared_between() - (-0.117_203_320_997_885_11)).abs() < 1e-9);
    assert!((re.r_squared_overall() - 0.279_462_900_829_853_8).abs() < 1e-9);
}

#[test]
fn re_estimator_fit_hc1_matches_plm_reference() {
    // `plm::vcovHC(m, method = "white1", type = "HC1")`と数値一致（補正は`n/(n-k)`。
    // バランスパネルでは`linearmodels`の`cov_type="robust"`とも一致する）。
    let re = ReEstimator::fit(
        cov_type_reference_input(),
        ReEffects::OneWay,
        ReCovType::Hc1,
        0.95,
    )
    .unwrap();

    assert!((*re.std_errors().get(0, 0) - 1.714_688_597_358_583_3).abs() < 1e-9);
    assert!((*re.std_errors().get(1, 0) - 0.209_734_285_381_997_1).abs() < 1e-9);
}

#[test]
fn re_estimator_fit_hc2_hc3_match_plm_reference() {
    // `linearmodels`にはHC2/HC3の実装が無い（`RandomEffects`/`PanelOLS`どちらも
    // `_cov_estimators`が単一の"heteroskedastic"＝HC1相当のみ）ため、Rの
    // `plm::vcovHC(m, method = "white1", type = "HC2"/"HC3")`と比べる。レバレッジは
    // 準偏差変換後の設計行列の素の`h_ii`（`fit()`のdocコメント「`cov_type`対応」参照）。
    let hc2 = ReEstimator::fit(
        cov_type_reference_input(),
        ReEffects::OneWay,
        ReCovType::Hc2,
        0.95,
    )
    .unwrap();
    assert!((*hc2.std_errors().get(0, 0) - 1.627_394_573_108_865_7).abs() < 1e-9);
    assert!((*hc2.std_errors().get(1, 0) - 0.212_174_022_116_461_4).abs() < 1e-9);

    let hc3 = ReEstimator::fit(
        cov_type_reference_input(),
        ReEffects::OneWay,
        ReCovType::Hc3,
        0.95,
    )
    .unwrap();
    assert!((*hc3.std_errors().get(0, 0) - 1.830_351_255_951_733_9).abs() < 1e-9);
    assert!((*hc3.std_errors().get(1, 0) - 0.254_854_683_896_986_1).abs() < 1e-9);
}

#[test]
fn re_estimator_fit_cluster_defaults_to_entity_and_matches_fixest_style_correction() {
    // REは`extra_df`が常に`0`（固定効果ダミーが設計行列に無くFEのようなネスト判定が
    // 不要）なため、`K=df_model`をそのまま使う`(G/(G-1))×((n-1)/(n-K))`補正
    // （fixest・Stata型）になる。`plm::vcovHC(m, method = "arellano", type = "sss")`と数値一致。
    let re = ReEstimator::fit(
        cov_type_reference_input(),
        ReEffects::OneWay,
        ReCovType::Cluster { groups: None },
        0.95,
    )
    .unwrap();

    assert!((*re.std_errors().get(0, 0) - 2.138_005_211_311_433_4).abs() < 1e-9);
    assert!((*re.std_errors().get(1, 0) - 0.182_210_748_005_084).abs() < 1e-9);
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
    // FEのDKと同じく`K=df_model`・`G`相当は`t_periods`を使う
    // `(t_periods/(t_periods-1))×((n-1)/(n-K))`補正（fixest型）。
    // `plm::vcovSCC(m, maxlag = 0/1, type = "sss")`と数値一致（`t_periods=3`、`n=7`）。
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
    assert!((*bw0.std_errors().get(0, 0) - 0.077_350_818_526_441_61).abs() < 1e-9);
    assert!((*bw0.std_errors().get(1, 0) - 0.305_989_536_115_697_4).abs() < 1e-9);

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
    assert!((*bw1.std_errors().get(0, 0) - 0.074_136_414_310_167_04).abs() < 1e-9);
    assert!((*bw1.std_errors().get(1, 0) - 0.191_690_091_600_826_5).abs() < 1e-9);
}
