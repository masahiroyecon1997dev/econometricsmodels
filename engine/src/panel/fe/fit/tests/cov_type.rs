//! `fit`の`cov_type`別（Classical・HC・Cluster・HAC・Driscoll-Kraay など）の共分散・検定統計量のテスト。

use super::dk_bandwidth_zero;
use super::dk_time_from;
use super::three_period_input;
use super::three_period_time;
use super::two_entity_input;
use super::two_period_input;
use crate::panel::common::TimeKeys;
use crate::panel::fe::FeEstimator;
use crate::panel::fe::input::FeCovType;
use crate::panel::fe::input::FeEffects;
use crate::panel::fe::input::FeInput;
use crate::panel::fe::test_support::*;
use crate::shared::group_codes::GroupCodes;

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
fn fe_estimator_fit_one_way_hac_uses_explicit_time_override_without_fe_input_time() {
    // `FeCovType::Dk.time`（明示指定）は`FeInput.time_keys()`を経由せずに
    // DK HACを成立させられる（`engine_pybind`の`FEOptions.dk_time`が1-way FE + DK HAC
    // の組み合わせをこの経路で配線する想定）。`FeInput::from_columns`には`time=None`を
    // 渡し、`fe_estimator_fit_one_way_hac_matches_fixest_default_bandwidth`と
    // 同じ結果になることを確認する（同じ`time`列を使っているため数値は完全一致する）。
    let (entity, time, x, y) = fixest_reference_input();
    let input =
        FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into()).unwrap();

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
        FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into()).unwrap();
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
fn fe_estimator_fit_hac_two_period_panel_without_regressors_is_accepted() {
    // `k=0`（固定効果のみ）は標準誤差・F検定を計算しないため退化を問題にしない。
    let (input, _) = two_period_input(0, false);
    let dk_cov = dk_bandwidth_zero(&input);
    assert!(FeEstimator::fit(input, FeEffects::OneWay, dk_cov, 0.95).is_ok());
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
        .map(|i| 2.0 + 1.5 * x1[i] - 0.8 * x2[i] + ((i as f64) * 2.1).sin() * 1.7 + (i / 4) as f64)
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
