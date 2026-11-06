//! `fit`の基本挙動（既知の係数の復元・適合度・getter・成功する完全適合や定数のyなど）のテスト。

use crate::linear::ols::CovType;
use crate::linear::ols::OlsEstimator;
use crate::linear::ols::OlsInput;
use crate::panel::re::ReCovType;
use crate::panel::re::ReEffects;
use crate::panel::re::ReEstimator;
use crate::panel::re::ReInput;
use crate::panel::re::test_support::*;
use crate::panel::re::variance::swamy_arora_variance_components;

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
                (*re.estimator().params().get(a, 0) - *base.estimator().params().get(b, 0)).abs()
                    < 1e-10,
                "{name} in order {order:?}"
            );
        }
        assert!((re.r_squared_within() - base.r_squared_within()).abs() < 1e-12);
        assert!((re.hausman_statistic().unwrap() - base.hausman_statistic().unwrap()).abs() < 1e-9);
        assert_eq!(re.hausman_df(), Some(2));
    }
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

    let a = ReEstimator::fit(two_way_data, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
    let b = ReEstimator::fit(without_time, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

    assert_eq!(a.n_periods(), None);
    assert_eq!(
        a.estimator().params().get(1, 0),
        b.estimator().params().get(1, 0)
    );
    assert!(a.hausman_statistic().is_some());
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
fn re_estimator_fit_pins_faer_global_parallelism_to_seq() {
    // `fit()`冒頭の`crate::shared::parallelism::ensure_serial()`がfaerの
    // グローバル並列度を`Par::Seq`へ引き戻すことの回帰ガード
    // （`fe_estimator_fit_pins_faer_global_parallelism_to_seq`と同型、
    // `engine/src/panel/CLAUDE.md`「faerのグローバル並列度」参照）。
    faer::set_global_parallelism(faer::Par::rayon(0));

    let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
    let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
    let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
    let input = ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();

    let _ = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

    assert!(matches!(faer::get_global_parallelism(), faer::Par::Seq));
}
