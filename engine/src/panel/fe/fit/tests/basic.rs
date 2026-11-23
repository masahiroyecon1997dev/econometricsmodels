//! `fit`の基本挙動（既知の係数の復元・適合度・getter・成功する完全適合や定数のyなど）のテスト。

use super::dk_bandwidth_zero;
use super::two_period_input;
use crate::panel::fe::FeEstimator;
use crate::panel::fe::input::FeCovType;
use crate::panel::fe::input::FeEffects;
use crate::panel::fe::input::FeInput;
use crate::panel::fe::test_support::*;
use crate::shared::group_codes::GroupCodes;
use crate::shared::inference;

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
    let input = FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
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
fn fe_estimator_fit_pins_faer_global_parallelism_to_seq() {
    // `fit()`冒頭の`crate::shared::parallelism::ensure_serial()`がfaerのグローバル
    // 並列度を`Par::Seq`へ引き戻すことの回帰ガード（panel系統代表、
    // `engine/src/panel/CLAUDE.md`「faerのグローバル並列度」参照）。
    faer::set_global_parallelism(faer::Par::rayon(0));

    let entity = strings(&["a", "a", "b", "b"]);
    let y = [1.0, 2.2, 3.0, 4.6];
    let x1 = vec![1.0, 2.0, 3.0, 4.0];
    let input = FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
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

#[test]
fn fe_estimator_fit_exposes_input_and_cov_type_via_getters() {
    // `input()`/`cov_type()`（カバレッジ監査で判明した未検証の単純
    // getter）。`ols::fit_exposes_input_cov_type_and_residuals_via_getters`と同型。
    let entity = strings(&["a", "a", "b", "b"]);
    let x1 = vec![1.0, 2.0, 3.0, 4.0];
    let y = [7.0, 9.2, 11.0, 12.6];
    let input = FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
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
