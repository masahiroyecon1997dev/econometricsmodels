//! plmの閉形式との照合・振り分けのテスト。

use super::all_rows;
use super::components;
use super::input;
use crate::panel::re::ReCovType;
use crate::panel::re::ReEstimator;
use crate::panel::re::input::ReEffects;
use crate::panel::re::variance::swamy_arora_variance_components;
use crate::panel::re::variance_method::VarianceMethod;

fn assert_rel(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0),
        "{label}: actual={actual}, expected={expected}"
    );
}

#[test]
fn balanced_one_way_matches_plm_closed_forms() {
    // plm: ercomp(y ~ x1, d, effect = "individual", method = "walhus" | "amemiya")
    // （バランス、`dfcor = 1`の閉形式）。
    let inp = input(&all_rows(), &["x1"], false);
    for (method, eps, u) in [
        (
            VarianceMethod::WallaceHussain,
            6.180_679_388_810_379,
            1.446_976_393_641_097,
        ),
        (
            VarianceMethod::Amemiya,
            6.103_755_807_867_632,
            1.597_323_540_521_647,
        ),
    ] {
        let vc = components(&inp, ReEffects::OneWay, method).unwrap();
        assert_rel(vc.sigma2_eps, eps, "sigma2_eps");
        assert_rel(vc.sigma2_u, u, "sigma2_u");
        assert_eq!(vc.sigma2_lambda, 0.0);
    }
}

#[test]
fn balanced_one_way_with_a_time_only_regressor_matches_plm() {
    // plm: y ~ x1 + xt（`xt`は時点間でのみ変わる）。WHはプールドOLSの残差、Amemiyaはwithin
    // の係数（`xt`もwithin変動を持つ）を使う。
    let inp = input(&all_rows(), &["x1", "xt"], false);
    for (method, eps, u) in [
        (
            VarianceMethod::WallaceHussain,
            4.992_539_566_332_841,
            1.667_316_188_181_227,
        ),
        (
            VarianceMethod::Amemiya,
            4.940_882_248_916_686,
            1.768_844_088_951_385,
        ),
    ] {
        let vc = components(&inp, ReEffects::OneWay, method).unwrap();
        assert_rel(vc.sigma2_eps, eps, "sigma2_eps");
        assert_rel(vc.sigma2_u, u, "sigma2_u");
    }
}

#[test]
fn unbalanced_one_way_matches_plm_trace_forms() {
    // 4行（0始まり5・10・11・19）を落とした不均衡パネル（観測数 4,3,2,4,3,4）。
    // plm: 同じデータの`ercomp`（不均衡は`dfcor = 3`のトレース項の式）。
    let rows: Vec<usize> = (0..24).filter(|r| ![5, 10, 11, 19].contains(r)).collect();
    let inp = input(&rows, &["x1"], false);
    for (method, eps, u) in [
        (
            VarianceMethod::WallaceHussain,
            6.190_506_235_160_872,
            2.457_088_808_540_195,
        ),
        (
            VarianceMethod::Amemiya,
            6.229_221_018_750_94,
            2.375_376_415_113_585,
        ),
    ] {
        let vc = components(&inp, ReEffects::OneWay, method).unwrap();
        assert_rel(vc.sigma2_eps, eps, "sigma2_eps");
        assert_rel(vc.sigma2_u, u, "sigma2_u");
    }
}

#[test]
fn balanced_two_way_matches_plm_closed_forms() {
    // plm: ercomp(y ~ x1, d, effect = "twoways", method = ...)。
    let inp = input(&all_rows(), &["x1"], true);
    for (method, eps, u, lambda) in [
        (
            VarianceMethod::WallaceHussain,
            0.816_015_873_070_809_3,
            2.788_142_272_575_989,
            3.989_496_975_426_726_4,
        ),
        (
            VarianceMethod::Amemiya,
            0.663_192_700_992_561,
            2.699_290_005_810_702,
            4.371_238_538_448_286,
        ),
    ] {
        let vc = components(&inp, ReEffects::TwoWay, method).unwrap();
        assert_rel(vc.sigma2_eps, eps, "sigma2_eps");
        assert_rel(vc.sigma2_u, u, "sigma2_u");
        assert_rel(vc.sigma2_lambda, lambda, "sigma2_lambda");
    }
}

#[test]
fn swamy_arora_dispatch_is_the_existing_estimator() {
    let inp = input(&all_rows(), &["x1"], false);
    let direct = swamy_arora_variance_components(&inp, ReEffects::OneWay, 0.95).unwrap();
    let via = components(&inp, ReEffects::OneWay, VarianceMethod::SwamyArora).unwrap();
    assert_eq!(direct.sigma2_eps, via.sigma2_eps);
    assert_eq!(direct.sigma2_u, via.sigma2_u);
}

#[test]
fn fit_records_the_variance_method() {
    let inp = || input(&all_rows(), &["x1"], false);
    let default = ReEstimator::fit(inp(), ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
    assert_eq!(default.variance_method(), VarianceMethod::SwamyArora);
    let walhus = ReEstimator::fit_with_variance_method(
        inp(),
        ReEffects::OneWay,
        ReCovType::Classical,
        VarianceMethod::WallaceHussain,
        0.95,
    )
    .unwrap();
    assert_eq!(walhus.variance_method(), VarianceMethod::WallaceHussain);
    assert_rel(walhus.sigma2_eps(), 6.180_679_388_810_379, "sigma2_eps");
    assert_eq!(VarianceMethod::default(), VarianceMethod::SwamyArora);
    assert_eq!(VarianceMethod::Amemiya.as_str(), "amemiya");
}
