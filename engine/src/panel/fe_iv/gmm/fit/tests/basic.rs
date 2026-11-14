//! 基本挙動（getter・推定回数と収束・係数名・自由度）。

use super::{
    N_ENTITIES, N_PERIODS, entity_cluster_weight, entity_labels, fit, input_spec, one_way_input,
    two_step, two_way_input,
};
use crate::panel::fe::FeEffects;
use crate::panel::fe_iv::FeGmmEstimator;
use crate::shared::gmm_iteration::GmmType;
use crate::shared::gmm_weight::WeightType;

#[test]
fn getters_report_the_model() {
    let estimator = fit(
        two_way_input(),
        FeEffects::TwoWay,
        two_step(WeightType::Robust),
    )
    .unwrap();
    let n = N_ENTITIES * N_PERIODS;
    assert_eq!(estimator.nobs(), n);
    assert_eq!(estimator.effects(), FeEffects::TwoWay);
    assert_eq!(estimator.param_names(), ["x1", "x2"]);
    assert_eq!(estimator.params().nrows(), 2);
    assert_eq!(estimator.residuals().nrows(), n);
    assert_eq!(estimator.input().nobs(), n);
    assert_eq!(
        estimator.gmm_type(),
        &two_step(WeightType::Robust),
        "gmm_type"
    );
    // 傾き2個 + (個体10 + 時点4 − 1)。
    assert_eq!(estimator.n_groups(), [N_ENTITIES, N_PERIODS]);
    assert_eq!(estimator.df_model(), 2 + N_ENTITIES + N_PERIODS - 1);
    assert_eq!(estimator.df_resid(), n - estimator.df_model());
    // 操作変数`z1`・`z2`と外生変数`x1`の3つのモーメント。
    assert_eq!(estimator.s_used().nrows(), 3);
    assert_eq!(estimator.s_used().ncols(), 3);
}

#[test]
fn one_way_degrees_of_freedom_subtract_the_entity_effects() {
    let estimator = fit(
        one_way_input(),
        FeEffects::OneWay,
        two_step(WeightType::Classical),
    )
    .unwrap();
    assert_eq!(estimator.df_model(), 2 + N_ENTITIES);
    assert_eq!(
        estimator.df_resid(),
        N_ENTITIES * N_PERIODS - 2 - N_ENTITIES
    );
}

/// 係数名は`x_exog`、`x_endog`の順。
#[test]
fn param_names_list_exogenous_then_endogenous() {
    let input = input_spec(&super::all_rows(), &[entity_labels()], 2, 2, 3);
    let estimator = fit(input, FeEffects::OneWay, two_step(WeightType::Robust)).unwrap();
    assert_eq!(estimator.param_names(), ["x1", "w", "x2", "x3"]);
}

/// 推定回数: 1段階は1、2段階は2、反復は実際の回数。1段階・2段階は常に収束。
#[test]
fn iteration_counts_follow_the_gmm_type() {
    let make = || input_spec(&super::all_rows(), &[entity_labels()], 1, 1, 2);
    let one = fit(make(), FeEffects::OneWay, GmmType::OneStep).unwrap();
    assert_eq!((one.n_iter(), one.converged()), (1, true));
    let two = fit(make(), FeEffects::OneWay, two_step(WeightType::Robust)).unwrap();
    assert_eq!((two.n_iter(), two.converged()), (2, true));
    let iterated = fit(
        make(),
        FeEffects::OneWay,
        GmmType::Iterated {
            weight: WeightType::Classical,
            max_iter: 20,
            tol: 1e-8,
        },
    )
    .unwrap();
    // `Classical`重みは2回目の推定で値が動かない（1回目と比べて収束と判定される）。
    assert_eq!(iterated.n_iter(), 2);
    assert!(iterated.converged());
}

/// `raise_on_non_convergence = false`なら、収束しなくても`converged() == false`のまま結果を返す。
#[test]
fn non_convergence_is_returned_when_not_raised() {
    let make = || input_spec(&super::all_rows(), &[entity_labels()], 1, 1, 2);
    let tight = GmmType::Iterated {
        weight: WeightType::Robust,
        max_iter: 3,
        tol: 1e-300,
    };
    let estimator = FeGmmEstimator::fit(make(), FeEffects::OneWay, tight, false).unwrap();
    assert!(!estimator.converged());
    assert_eq!(estimator.n_iter(), 3);
}

/// クラスター重みの`s_used`は対称（`Σ_g s_g s_g'`）。
#[test]
fn cluster_weight_matrix_is_symmetric_and_positive() {
    let estimator = fit(
        two_way_input(),
        FeEffects::TwoWay,
        two_step(entity_cluster_weight()),
    )
    .unwrap();
    let s = estimator.s_used();
    for a in 0..s.nrows() {
        assert!(*s.get(a, a) > 0.0);
        for b in 0..a {
            let (upper, lower) = (*s.get(b, a), *s.get(a, b));
            assert!((upper - lower).abs() <= 1e-12 * upper.abs().max(1.0));
        }
    }
}

/// `fit()`冒頭の`ensure_serial()`が、faerのグローバル並列度を`Par::Seq`へ引き戻す回帰ガード。
#[test]
fn fit_pins_faer_global_parallelism_to_seq() {
    faer::set_global_parallelism(faer::Par::rayon(0));
    let _ = fit(
        one_way_input(),
        FeEffects::OneWay,
        two_step(WeightType::Classical),
    );
    assert!(matches!(faer::get_global_parallelism(), faer::Par::Seq));
}
