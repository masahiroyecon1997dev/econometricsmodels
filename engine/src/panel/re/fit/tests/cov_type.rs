//! `fit`の`cov_type`別（Classical・HC・Cluster・HAC・Driscoll-Kraay など）の共分散・検定統計量のテスト。

use crate::panel::common::TimeKeys;
use crate::panel::re::ReCovType;
use crate::panel::re::ReEffects;
use crate::panel::re::ReEstimator;
use crate::panel::re::ReInput;
use crate::panel::re::test_support::*;
use crate::shared::group_codes::GroupCodes;

#[test]
fn re_estimator_fit_with_time_invariant_regressor_supports_every_cov_type_and_unbalanced_panel() {
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
            let re =
                ReEstimator::fit(build(balanced), ReEffects::OneWay, cov.clone(), 0.95).unwrap();
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
fn re_estimator_fit_cluster_supports_explicit_groups_column() {
    // `groups`に`entity`以外の任意の列を明示指定できることを確認する
    // （3.2節「`cluster`を明示指定すれば任意の列でもクラスター可能」）。
    // ここでは`entity`をそのまま複製した列を明示的に渡し、`groups: None`
    // （`re_estimator_fit_cluster_defaults_to_entity_and_matches_fixest_style_
    // correction`）と同じ結果になることを確認する。
    let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
    let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
    let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
    let input = ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
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
        "e0", "e0", "e1", "e1", "e2", "e2", "e2", "e2", "e2", "e2", "e2", "e2", "e2", "e2", "e2",
        "e2", "e2", "e2", "e2",
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
    let input = ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
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
        .map(|i| 2.0 + 1.5 * x1[i] - 0.8 * x2[i] + ((i as f64) * 2.1).sin() * 1.7 + (i / 4) as f64)
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
