//! linearmodels・plmとの照合・between回帰の列の扱いのテスト。

use crate::panel::common::BetweenMeans;
use crate::panel::common::PanelError;
use crate::panel::re::ReCovType;
use crate::panel::re::ReEstimator;
use crate::panel::re::input::ReEffects;
use crate::panel::re::input::ReInput;
use crate::panel::re::test_support::*;
use crate::panel::re::transform::compute_two_way_theta;
use crate::panel::re::variance::VarianceComponents;
use crate::panel::re::variance::between_column_is_constant;
use crate::panel::re::variance::between_ssr_and_df;
use crate::panel::re::variance::column_max_abs;
use crate::panel::re::variance::entity_means;
use crate::panel::re::variance::swamy_arora_variance_components;

#[test]
fn swamy_arora_variance_components_matches_linearmodels_reference() {
    // 手動データ（不均衡パネル、entity a: T=3, b: T=2, c: T=2）。
    // `linearmodels.RandomEffects`（Python）で実地検証済みの値と比較する
    // （`RandomEffects(y, [const, x1]).fit().variance_decomposition`）。
    // between回帰は切片+傾き1個の2パラメータのため、`n_entities > k+1`（3章参照）を
    // 満たすには最低3エンティティが必要（2エンティティだと`neffects-nvar=0`で
    // 除算不能になることを`linearmodels`自身でも実地確認済み）。
    let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
    let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
    let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
    let input = ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();

    let VarianceComponents {
        sigma2_eps,
        sigma2_u,
        ..
    } = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();

    assert!(
        (sigma2_eps - 1.132_352_941_176_471_5).abs() < 1e-9,
        "sigma2_eps = {sigma2_eps}"
    );
    assert!(
        (sigma2_u - 6.537_912_784_161_284).abs() < 1e-9,
        "sigma2_u = {sigma2_u}"
    );
}

#[test]
fn swamy_arora_variance_components_passes_only_time_varying_columns_to_internal_fe() {
    // 時間不変の`z`は内部FEに渡さない（FEは分散ゼロの列を拒否する）。σ_ε²は
    // `z`を含めても含めなくても同じ（within変動のある列だけで決まる）で、
    // 時間変化のある列の添字は列順どおり。
    let with_z = swamy_arora_variance_components(
        &time_invariant_input(&["z", "x1", "x2"]),
        ReEffects::OneWay,
        0.95,
    )
    .unwrap();
    let without_z = swamy_arora_variance_components(
        &time_invariant_input(&["x1", "x2"]),
        ReEffects::OneWay,
        0.95,
    )
    .unwrap();

    assert_eq!(with_z.time_varying, vec![1, 2]);
    assert_eq!(without_z.time_varying, vec![0, 1]);
    assert_eq!(with_z.fe.input().x_names(), &strings(&["x1", "x2"]));
    assert_eq!(with_z.sigma2_eps, without_z.sigma2_eps);
    // plm: ercomp(m)$sigma2["idios"]（`z`を含むモデル）。自由度は`n - N - k_tv`。
    assert!((with_z.sigma2_eps - 0.201_246_585_026_864).abs() < 1e-12);
}

#[test]
fn between_regression_drops_a_column_that_repeats_an_earlier_one() {
    // 個体平均の回帰で`x1bar`の個体平均は`x1`のそれと同じ列になる（`plm`の`lm`は後ろの列を
    // 落とし、残差平方和・ランクは変わらない）。落とした列を数えず、`σ_u²`の自由度が変わらない。
    let with_bar = mundlak_input(&["x1", "x2", "x1bar"], false);
    let without = mundlak_input(&["x1", "x2"], false);
    let stats = |input: &ReInput| {
        let means = entity_means(input.y(), input.x(), input.entity_codes());
        between_ssr_and_df(
            &means,
            &column_max_abs(input.x()),
            input.x_names(),
            BetweenMeans::Entity,
        )
        .unwrap()
    };
    let (ssr_bar, df_bar) = stats(&with_bar);
    let (ssr, df) = stats(&without);
    assert_eq!(df_bar, df);
    assert_close(ssr_bar, ssr, 1e-12, "ssr");
    assert!(ssr > 0.0);
    // 列順を入れ替えても（落ちる列が`x1`に変わっても）同じ。
    let reordered = mundlak_input(&["x1bar", "x1", "x2"], false);
    let (ssr_re, df_re) = stats(&reordered);
    assert_eq!(df_re, df);
    assert_close(ssr_re, ssr, 1e-12, "ssr reordered");
}

#[test]
fn one_way_balanced_panel_drops_time_only_column_from_between_regression_like_plm() {
    // plm: `plm(y ~ x1 + xt, model = "random", effect = "individual")`。`xt`は時点間でのみ
    // 変わる列で、バランスパネルでは個体平均が全個体で一致し切片と共線になる。between回帰から
    // 落とし`K_b`にも数えない（plmと同じ）。最終回帰には`xt`も残り、係数が推定される。
    let c = two_way_columns();
    let input = ReInput::from_columns(
        &c.y,
        &[c.x1.clone(), c.xt.clone()],
        strings(&["x1", "xt"]),
        &c.entity,
        None,
        "y".into(),
    )
    .unwrap();

    let vc = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();
    assert_close(vc.sigma2_eps, 5.558_492_530_031_27, 1e-9, "sigma2_eps");
    assert_close(vc.sigma2_u, 2.781_216_088_364_69, 1e-9, "sigma2_u");

    let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
    let params = [
        0.117_907_102_600_895,
        0.400_160_538_463_792,
        -1.029_152_589_790_345,
    ];
    let std_errors = [
        0.905_185_788_075_077,
        0.851_443_915_446_071,
        0.510_706_679_841_924,
    ];
    for j in 0..3 {
        assert_close(*re.estimator().params().get(j, 0), params[j], 1e-9, "param");
        assert_close(*re.std_errors().get(j, 0), std_errors[j], 1e-9, "std_err");
    }
    assert_eq!(re.df_resid(), 21);
    // ハウスマン検定は`xt`の`x̃t`が定数項と共線で識別できないため外し、`x1`だけを検定する
    // （補助回帰`y* ~ 1 + x1* + xt* + x̃1`のWald統計量をnumpyで独立に計算した値）。
    assert_eq!(re.hausman_df(), Some(1));
    assert_close(
        re.hausman_statistic().unwrap(),
        0.293_444_752_111_572_3,
        1e-9,
        "hausman",
    );
}

#[test]
fn one_way_balanced_panel_keeps_entity_only_column_in_between_regression() {
    // plm: `y ~ x1 + xi + xt`。`xi`（個体間でのみ変わる列）は個体平均でも変動するため
    // between回帰に残り、`xt`だけが落ちる。
    let c = two_way_columns();
    let input = ReInput::from_columns(
        &c.y,
        &[c.x1.clone(), c.xi.clone(), c.xt.clone()],
        strings(&["x1", "xi", "xt"]),
        &c.entity,
        None,
        "y".into(),
    )
    .unwrap();

    let vc = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();
    assert_close(vc.sigma2_u, 1.808_007_274_071_92, 1e-9, "sigma2_u");

    let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
    let params = [
        0.139_211_658_956_512_4,
        0.455_007_744_295_883_3,
        -0.053_752_290_510_623_2,
        -1.036_241_827_179_690_7,
    ];
    for (j, expected) in params.iter().enumerate() {
        assert_close(*re.estimator().params().get(j, 0), *expected, 1e-9, "param");
    }
    assert_eq!(re.df_resid(), 20);
    assert_eq!(re.hausman_df(), Some(1));
    assert_close(
        re.hausman_statistic().unwrap(),
        2.587_055_364_739_007,
        1e-9,
        "hausman",
    );
}

#[test]
fn two_way_variance_components_and_theta_match_plm() {
    // plm: ercomp(m)$sigma2（idios, id, time）・ercomp(m)$theta（id, time, total）。
    // `y ~ x1`（両次元で変動する列だけ）と、個体間でのみ変わる`xi`・時点間でのみ変わる`xt`を
    // 加えた`y ~ x1 + xi + xt`。後者はbetween回帰が変動のない列を落とす規則
    // （個体平均の回帰から`xt`、時点平均の回帰から`xi`）の確認でもある。
    for (names, sigma2, theta) in [
        (
            vec!["x1"],
            [
                0.710_563_608_206_315,
                3.993_198_318_827_93,
                1.831_135_868_289_6,
            ],
            [
                0.793_623_729_209_202,
                0.753_533_894_195_726,
                0.707_406_705_465_165,
            ],
        ),
        (
            vec!["x1", "xi", "xt"],
            [
                0.710_563_608_206_315,
                3.019_989_504_528_157,
                2.596_736_326_025_116_6,
            ],
            [
                0.764_301_210_325_029,
                0.791_153_185_965_249,
                0.713_712_012_990_144,
            ],
        ),
    ] {
        let input = two_way_input(&names);
        let vc = swamy_arora_variance_components(&input, ReEffects::TwoWay, 0.95).unwrap();
        let label = format!("{names:?}");
        assert_close(
            vc.sigma2_eps,
            sigma2[0],
            1e-9,
            &format!("{label} sigma2_eps"),
        );
        assert_rel(vc.sigma2_u, sigma2[1], 1e-9, &format!("{label} sigma2_u"));
        assert_rel(
            vc.sigma2_lambda,
            sigma2[2],
            1e-9,
            &format!("{label} sigma2_lambda"),
        );
        let t = compute_two_way_theta(vc.sigma2_eps, vc.sigma2_u, vc.sigma2_lambda, 6, 4);
        assert_close(t.entity, theta[0], 1e-9, &format!("{label} theta_entity"));
        assert_close(t.time, theta[1], 1e-9, &format!("{label} theta_time"));
        assert_close(t.total, theta[2], 1e-9, &format!("{label} theta_total"));
    }
}

#[test]
fn between_column_is_constant_follows_the_raw_scale_and_the_number_of_terms() {
    // 平均が丸め誤差程度に揺れる（元の列は一定）場合、元の列のスケールと項数が基準なら一定と判定する。
    let noise = [1.2e-17, -3.4e-17, 0.0, 2.2e-17];
    assert!(between_column_is_constant(&noise, 1.0, 5.0));
    // 平均の列自身のスケール（~1e-17）を基準にすると見逃す。
    assert!(!between_column_is_constant(&noise, 3.4e-17, 1.0));
    // 本物の変動（元のスケールの1e-3）は一定とみなさない。
    let varying = [1.0, 1.001, 0.999, 1.0];
    assert!(!between_column_is_constant(&varying, 1.0, 5.0));
    // 完全に一定。
    assert!(between_column_is_constant(&[0.3; 7], 0.3, 1.0));
}

#[test]
fn one_way_unbalanced_panel_keeps_time_only_column_in_between_regression() {
    // 1行欠けて不均衡になると個体平均が一定でなくなり、`xt`はbetween回帰に残る
    // （`K_b = 2`、自由度`6 - 3 = 3`。バランスなら`xt`が落ちて自由度は`6 - 2 = 4`）。
    let rows: Vec<usize> = (0..24).filter(|&r| r != 5).collect();
    let unbalanced = subset_input(&rows, &["x1", "xt"], false);
    let balanced = subset_input(&(0..24).collect::<Vec<_>>(), &["x1", "xt"], false);
    let df = |input: &ReInput| {
        let means = entity_means(input.y(), input.x(), input.entity_codes());
        between_ssr_and_df(
            &means,
            &column_max_abs(input.x()),
            input.x_names(),
            BetweenMeans::Entity,
        )
        .unwrap()
        .1
    };
    assert_eq!(df(&balanced), 4);
    assert_eq!(df(&unbalanced), 3);

    // ハウスマン検定でも`x̃t`が識別できる（θが個体ごとに異なる）ため両方検定する。
    let re = ReEstimator::fit(unbalanced, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
    assert_eq!(re.hausman_df(), Some(2));
}

#[test]
fn one_way_between_regression_counts_only_columns_that_still_vary() {
    // 3エンティティ×3時点。`x1 + xt`は`K_b = 1`（`xt`は落ちる）で`n_entities = 3 > 2`のため成功し、
    // 2エンティティなら`n_entities <= K_b + 1`で失敗する。
    // 落とさない場合は`n_entities <= k + 1 = 3`で失敗する。
    let build = |n_entities: usize| {
        let rows: Vec<usize> = (0..n_entities * 4).collect();
        // 4時点のうち先頭3時点だけ使う（`r % 4 < 3`）。
        let rows: Vec<usize> = rows.into_iter().filter(|r| r % 4 < 3).collect();
        subset_input(&rows, &["x1", "xt"], false)
    };
    assert!(swamy_arora_variance_components(&build(3), ReEffects::OneWay, 0.95).is_ok());
    assert!(matches!(
        swamy_arora_variance_components(&build(2), ReEffects::OneWay, 0.95),
        Err(PanelError::BetweenRegressionFailed { .. })
    ));
}
