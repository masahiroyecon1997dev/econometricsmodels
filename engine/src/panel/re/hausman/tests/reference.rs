//! plm・手計算の補助回帰との照合のテスト。

use super::manual_hausman_aux;
use crate::linear::ols::CovType;
use crate::linear::ols::OlsEstimator;
use crate::linear::ols::OlsInput;
use crate::panel::common::quasi_demean_column;
use crate::panel::re::ReEffects;
use crate::panel::re::ReEstimator;
use crate::panel::re::ReInput;
use crate::panel::re::input::ReCovType;
use crate::panel::re::test_support::*;
use crate::panel::re::transform::quasi_demean_transform;
use crate::panel::re::variance::VarianceComponents;
use crate::panel::re::variance::swamy_arora_variance_components;
use statrs::distribution::ChiSquared;
use statrs::distribution::ContinuousCDF;

#[test]
fn two_way_hausman_matches_the_auxiliary_regression_reference_for_every_cov_type() {
    // 補助回帰: 3つのθで変換した`y*`を`[1, X*, X̃]`（`X̃`は2-way within変換後に分散のある列）に
    // 回帰し`X̃`の係数をWald検定する。参照は`plm::phtest(method = "aux", effect = "twoways",
    // vcov = ...)`（`x1`のみ、全列が両次元で変動する入力）と、一次元でしか変わらない列を含む
    // 入力での手計算の補助回帰（plmは係数を位置で選ぶため、この入力ではplmの値が別の係数の
    // 検定になり使えない）。cov_type順: classical, HC1, HC2, HC3, cluster（entity）,
    // DK（bandwidth=1）。
    let c = two_way_columns();
    let cases: [(&[&str], [f64; 6]); 2] = [
        (
            &["x1"],
            [
                1.219_038_535_996_78,
                2.288_441_340_520_52,
                2.143_067_391_820_05,
                1.747_073_734_662_46,
                5.186_875_856_510_08,
                1.110_285_110_999_32,
            ],
        ),
        (
            &["x1", "xi", "xt"],
            [
                0.020_172_191_267_105_1,
                0.030_257_894_203_708_7,
                0.027_319_433_563_145_9,
                0.018_691_740_018_767,
                0.439_816_368_093_923,
                0.024_735_842_716_204_3,
            ],
        ),
    ];
    for (names, expected) in cases {
        for ((label, cov_type), want) in all_cov_types(&c.time).into_iter().zip(expected) {
            let re = ReEstimator::fit(two_way_input(names), ReEffects::TwoWay, cov_type, 0.95)
                .unwrap_or_else(|e| panic!("{names:?} {label}: {e}"));
            assert_eq!(re.hausman_df(), Some(1), "{names:?} {label}");
            assert_rel(
                re.hausman_statistic().unwrap(),
                want,
                1e-9,
                &format!("{names:?} {label}"),
            );
            let p = re.hausman_p_value().unwrap();
            assert!((0.0..=1.0).contains(&p), "{names:?} {label}: p={p}");
        }
    }
}

#[test]
fn two_way_hausman_with_several_tested_columns_matches_references() {
    // cov_type順: classical, HC1, HC2, HC3, cluster（entity）, DK（bandwidth=1）。各行は
    // （統計量, χ²_df上側確率）。`x1 + x2`は両次元で変動する2列（plm::phtest(aux, twoways)の値と
    // 手計算が一致）、`x1 + z + x2`は個体平均も時点平均も0の交互作用`z`を含む（`z`は識別できず
    // 検定対象から外れるため`hausman_df = 2`）、`x1 + z`は`df = 1`。参照は手計算の補助回帰。
    let c = two_way_columns();
    type Row = (f64, f64);
    let cases: [(&[&str], usize, [Row; 6]); 3] = [
        (
            &["x1", "x2"],
            2,
            [
                (9.92830801953696, 0.00698385648499824),
                (8.10342574438997, 0.0173925578818405),
                (6.72610170604262, 0.0346294482635305),
                (4.38401947725594, 0.111692051071814),
                (18.094162886712, 0.00011773414976345),
                (11.0661183693506, 0.00395387493521181),
            ],
        ),
        (
            &["x1", "z", "x2"],
            2,
            [
                (10.0371631198638, 0.00661390148183497),
                (9.07856691110755, 0.010681057264421),
                (7.64418368167015, 0.021881979367307),
                (4.7824250943718, 0.0915186458951753),
                (20.078462535067, 4.36533178816285e-05),
                (14.0801205135569, 0.000876073771284616),
            ],
        ),
        (
            &["x1", "z"],
            1,
            [
                (1.14428813461403, 0.284748043158543),
                (1.83787825176763, 0.175199832831764),
                (1.51093262045524, 0.218996831908603),
                (1.014108147529, 0.313920661605621),
                (3.63171340231645, 0.0566884270413592),
                (1.06814571765199, 0.301364660153732),
            ],
        ),
    ];
    for (names, df, rows) in cases {
        for ((label, cov_type), (stat, p)) in all_cov_types(&c.time).into_iter().zip(rows) {
            let re = ReEstimator::fit(two_way_input(names), ReEffects::TwoWay, cov_type, 0.95)
                .unwrap_or_else(|e| panic!("{names:?} {label}: {e}"));
            let tag = format!("{names:?} {label}");
            assert_eq!(re.hausman_df(), Some(df), "{tag}");
            assert_rel(re.hausman_statistic().unwrap(), stat, 1e-9, &tag);
            assert_rel(re.hausman_p_value().unwrap(), p, 1e-8, &tag);
        }
    }
}

#[test]
fn re_estimator_fit_hausman_matches_plm_aux_on_balanced_panel() {
    // バランスパネル（4エンティティ×3期間）ではSwamy-Arora分散成分が`plm`と一致し、
    // 補助回帰の値も1e-9で一致する。R: `plm::phtest(y ~ x1 + x2, data, method = "aux")`
    //   chisq = 0.076499376991569779, p = 0.96247259255247208, df = 2
    let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]);
    let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0, 4.0, 3.0, 7.0, 2.0, 5.0];
    let x2 = vec![2.0, 1.0, 5.0, 1.0, 4.0, 3.0, 2.0, 6.0, 5.0, 3.0, 4.0, 1.0];
    let y = [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0, 5.0, 9.5, 4.0, 7.3];
    let input = ReInput::from_columns(
        &y,
        &[x1, x2],
        vec!["x1".to_string(), "x2".to_string()],
        &entity,
        None,
        "y".into(),
    )
    .unwrap();

    let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

    assert_eq!(re.hausman_df(), Some(2));
    assert!((re.hausman_statistic().unwrap() - 0.076_499_376_991_569_78).abs() < 1e-9);
    assert!((re.hausman_p_value().unwrap() - 0.962_472_592_552_472_1).abs() < 1e-9);
}

#[test]
fn re_estimator_fit_hausman_is_close_to_plm_aux_on_unbalanced_panel() {
    // 不均衡パネルでは`plm`と分散成分（σ_u²）の推定式が僅かに異なる（`linearmodels`準拠の

    // R: `plm::phtest(y ~ x1 + x2, data, method = "aux")`
    //   chisq = 0.60715047844214221, p = 0.73817434737945442, df = 2
    let re = ReEstimator::fit(
        hausman_two_slope_input(None),
        ReEffects::OneWay,
        ReCovType::Classical,
        0.95,
    )
    .unwrap();

    assert_eq!(re.hausman_df(), Some(2));
    assert!((re.hausman_statistic().unwrap() - 0.607_150_478_442_142_2).abs() < 5e-2);
    assert!((re.hausman_p_value().unwrap() - 0.738_174_347_379_454_4).abs() < 5e-2);
}

#[test]
fn re_estimator_fit_hausman_matches_manual_auxiliary_regression() {
    // 補助回帰の独立な手計算（`OlsEstimator`を直接使い、準偏差変換・within変換を
    // 自前で組み立てる）との一致。単一傾き・不均衡パネル（`cov_type_reference_input`）。
    let re = ReEstimator::fit(
        cov_type_reference_input(),
        ReEffects::OneWay,
        ReCovType::Classical,
        0.95,
    )
    .unwrap();
    let input = re.input();
    let VarianceComponents {
        sigma2_eps,
        sigma2_u,
        ..
    } = swamy_arora_variance_components(input, ReEffects::OneWay, 0.95).unwrap();
    let (_, y_star, x_star) = quasi_demean_transform(input, sigma2_eps, sigma2_u);
    let entity = input.entity_codes().clone();
    let x_within = quasi_demean_column(&input.x()[0], &entity, &vec![1.0; entity.n_groups()]);

    let aux_input = OlsInput::from_columns(
        &y_star,
        &[x_star[0].clone(), x_within],
        vec!["x1".to_string(), "x1_within".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();
    let aux = OlsEstimator::fit(aux_input, CovType::Classical, 0.95).unwrap();
    // 1自由度のWaldは`t²`（`wald_test_last_columns`のdocコメント参照）。
    let t = *aux.test_stats().get(2, 0);
    let expected_stat = t * t;
    let expected_p = ChiSquared::new(1.0).unwrap().sf(expected_stat);

    assert_eq!(re.hausman_df(), Some(1));
    assert!((re.hausman_statistic().unwrap() - expected_stat).abs() < 1e-9);
    assert!((re.hausman_p_value().unwrap() - expected_p).abs() < 1e-9);
}

#[test]
fn re_estimator_fit_hausman_matches_fresh_within_transform_exactly() {
    // `re_hausman_test`の`X̃`は内部FEのwithin変換済み設計行列を再利用する。RE本体の入力に
    // within変換をかけ直して組んだ補助回帰（`manual_hausman_aux`、列順も同じ）と
    // ビット単位で一致する（不均衡パネル・2傾き）。
    let re = ReEstimator::fit(
        hausman_two_slope_input(None),
        ReEffects::OneWay,
        ReCovType::Classical,
        0.95,
    )
    .unwrap();
    let aux = manual_hausman_aux(re.input(), CovType::Classical);
    let (f_stat, _) = aux.wald_test_last_columns(2).unwrap();
    assert_eq!(re.hausman_statistic(), Some(2.0 * f_stat));
}
