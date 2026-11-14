//! FE-2SLSへの還元、ダミー変数を明示したGMM（FWL）、クラスター重みの手計算との照合。
//!
//! - **還元**: `Classical`重みの2段階GMMと1段階GMMは、重みが`(Z̃'Z̃)⁻¹`に比例するので2SLSと同じ点推定になる。
//!   丁度識別ではどの重みでも同じ。`FeIvEstimator`との一致で確かめる。
//! - **FWL**: 効果のダミーを外生変数（したがってモーメント条件にも入る）に加えた`iv::gmm::GmmEstimator`の
//!   係数は、within変換した固定効果つきGMMと一致する。ダミーのモーメントは`α`で厳密に0にできるので、
//!   残りのモーメントの重みは`S`のその部分行列になり、within変換後の`S`と同じになるため
//!   （`Z̃'D = 0`なので`Z̃`のモーメントは`α`に依存しない）。ダミーの数だけ`l`が増えるので、重みの
//!   種類は`Classical`・`Robust`に限る（クラスター重みは`G < l`になる）。反復GMMは3回目の推定から
//!   一致しない: ダミー込みでは2段階目の最適解で`D'ê ≠ 0`になり、次の重みを作る残差がwithin変換した
//!   残差と変わるため（1段階目は2SLSで`D'ê = 0`）。反復は手計算（下）で確かめる。
//! - **手計算**: クラスター重みは、個体内の平均を引いた列から、素朴な二重ループで`S`と係数を求める。

use faer::Mat;
use faer::prelude::Solve;

use super::{
    all_rows, dummies, entity_cluster_weight, entity_labels, fit, input_spec, input_with,
    third_labels, time_cluster_weight, time_labels, two_step,
};
use crate::iv::common::{IvEstimate, IvInput};
use crate::iv::gmm::GmmEstimator;
use crate::linear::ols::CovType;
use crate::panel::fe::{FeCovType, FeEffects};
use crate::panel::fe_iv::{FeGmmEstimator, FeIvEstimator, FeIvInput};
use crate::panel::fe_iv::{FeGmmType, FeWeightType};
use crate::shared::group_codes::GroupCodes;
use crate::shared::least_squares::least_squares;

const RTOL: f64 = 1e-8;

pub(super) fn assert_params_close(actual: &Mat<f64>, expected: &[f64], label: &str) {
    assert_eq!(actual.nrows(), expected.len(), "{label}");
    for (j, &want) in expected.iter().enumerate() {
        let got = *actual.get(j, 0);
        assert!(
            (got - want).abs() <= RTOL * want.abs().max(1e-3),
            "{label}[{j}]: {got} vs {want}"
        );
    }
}

fn params_of(estimator: &FeGmmEstimator) -> Vec<f64> {
    (0..estimator.params().nrows())
        .map(|j| *estimator.params().get(j, 0))
        .collect()
}

/// 不均衡にした行（観測を一部落とす。どの次元にもsingletonは出ない）。
pub(super) fn unbalanced_rows() -> Vec<usize> {
    all_rows().into_iter().filter(|r| r % 7 != 3).collect()
}

/// 固定効果の構造。`rows`は使う行、`dimensions`は全行分のラベル。
pub(super) struct Structure {
    pub(super) name: &'static str,
    pub(super) rows: Vec<usize>,
    pub(super) dimensions: Vec<Vec<String>>,
    pub(super) effects: FeEffects,
}

pub(super) fn structures() -> Vec<Structure> {
    let structure = |name, rows, dimensions, effects| Structure {
        name,
        rows,
        dimensions,
        effects,
    };
    vec![
        structure(
            "one-way",
            all_rows(),
            vec![entity_labels()],
            FeEffects::OneWay,
        ),
        structure(
            "two-way",
            all_rows(),
            vec![entity_labels(), time_labels()],
            FeEffects::TwoWay,
        ),
        structure(
            "two-way unbalanced",
            unbalanced_rows(),
            vec![entity_labels(), time_labels()],
            FeEffects::TwoWay,
        ),
        structure(
            "three-way",
            all_rows(),
            vec![entity_labels(), time_labels(), third_labels()],
            FeEffects::MultiWay,
        ),
    ]
}

/// 外生変数・内生変数・操作変数の数の組（丁度識別・過剰識別・外生変数なし・内生変数が複数）。
const SPECS: [(usize, usize, usize); 5] = [(1, 1, 1), (1, 1, 2), (1, 1, 3), (0, 1, 2), (2, 2, 3)];

fn fe_2sls_params(input: FeIvInput, effects: FeEffects) -> Vec<f64> {
    let estimator = FeIvEstimator::fit(input, effects, FeCovType::Classical, 0.95).unwrap();
    (0..estimator.params().nrows())
        .map(|j| *estimator.params().get(j, 0))
        .collect()
}

// ── FE-2SLSへの還元 ───────────────────────────────────────────────

/// `Classical`重みの2段階GMM・1段階GMM・`Classical`重みの反復GMMは、FE-2SLSと同じ係数になる。
#[test]
fn classical_weight_reduces_to_fe_2sls() {
    for Structure {
        name,
        rows,
        dimensions,
        effects,
    } in structures()
    {
        for (k_exog, k_endog, k_inst) in SPECS {
            let make = || input_spec(&rows, &dimensions, k_exog, k_endog, k_inst);
            let expected = fe_2sls_params(make(), effects);
            let label = format!("{name} {k_exog}/{k_endog}/{k_inst}");
            for gmm_type in [
                FeGmmType::OneStep,
                two_step(FeWeightType::Classical),
                FeGmmType::Iterated {
                    weight: FeWeightType::Classical,
                    max_iter: 10,
                    tol: 1e-8,
                },
            ] {
                let estimator = fit(make(), effects, gmm_type.clone()).unwrap();
                assert_params_close(
                    estimator.params(),
                    &expected,
                    &format!("{label} {gmm_type:?}"),
                );
            }
        }
    }
}

/// 丁度識別（`l = k`）では、どの重みでもFE-2SLSと同じ係数になる。
#[test]
fn just_identified_equals_fe_2sls_for_any_weight() {
    let input = || input_spec(&all_rows(), &[entity_labels(), time_labels()], 1, 1, 1);
    let expected = fe_2sls_params(input(), FeEffects::TwoWay);
    for weight in [
        FeWeightType::Robust,
        entity_cluster_weight(),
        time_cluster_weight(),
    ] {
        let estimator = fit(input(), FeEffects::TwoWay, two_step(weight.clone())).unwrap();
        assert_params_close(estimator.params(), &expected, &format!("{weight:?}"));
    }
}

/// 1段階GMMは`weight_type`によらず2SLSと同じで、`n_iter = 1`・`s_used`は`σ̂²₀・Z̃'Z̃`。
#[test]
fn one_step_is_the_two_stage_least_squares_estimate() {
    let input = || input_spec(&all_rows(), &[entity_labels()], 1, 1, 3);
    let expected = fe_2sls_params(input(), FeEffects::OneWay);
    let estimator = fit(input(), FeEffects::OneWay, FeGmmType::OneStep).unwrap();
    assert_params_close(estimator.params(), &expected, "one step");
    assert_eq!(estimator.n_iter(), 1);
    assert!(estimator.converged());
}

// ── ダミー変数を明示したGMM（FWL） ──────────────────────────────────────

/// ダミーを外生変数に加えた`IvInput`（ダミーは外生変数の後ろ。切片なし）。
pub(super) fn dummy_iv_input(input: &FeIvInput) -> IvInput {
    let d = dummies(input.index().effect_codes());
    let mut x_exog: Vec<Vec<f64>> = input.x_exog().to_vec();
    x_exog.extend(d.iter().cloned());
    let mut exog_names: Vec<String> = input.x_exog_names().to_vec();
    exog_names.extend((0..d.len()).map(|j| format!("d{j}")));
    IvInput::from_columns(
        input.y(),
        &x_exog,
        exog_names,
        input.x_endog(),
        input.x_endog_names().to_vec(),
        input.instruments(),
        input.instrument_names().to_vec(),
        false,
        "y".to_string(),
    )
    .unwrap()
}

/// ダミーを外生変数に加えた`iv::gmm::GmmEstimator`の係数（実際の外生変数・内生変数の分だけ）。
pub(super) fn dummy_variable_gmm_params(input: &FeIvInput, gmm_type: &FeGmmType) -> Vec<f64> {
    let d = dummies(input.index().effect_codes());
    let (ke, kn) = (input.k_exog(), input.k_endog());
    let mut x_exog: Vec<Vec<f64>> = input.x_exog().to_vec();
    x_exog.extend(d.iter().cloned());
    let mut exog_names: Vec<String> = input.x_exog_names().to_vec();
    exog_names.extend((0..d.len()).map(|j| format!("d{j}")));
    let iv_input = IvInput::from_columns(
        input.y(),
        &x_exog,
        exog_names,
        input.x_endog(),
        input.x_endog_names().to_vec(),
        input.instruments(),
        input.instrument_names().to_vec(),
        false,
        "y".to_string(),
    )
    .unwrap();
    let estimator = GmmEstimator::fit(
        iv_input,
        gmm_type.to_gmm_type(),
        true,
        CovType::Classical,
        0.95,
    )
    .unwrap();
    let params = estimator.params();
    let endog_start = ke + d.len();
    (0..ke)
        .chain(endog_start..endog_start + kn)
        .map(|j| *params.get(j, 0))
        .collect()
}

#[test]
fn matches_dummy_variable_gmm() {
    // 3次元は、ダミーが多く`Robust`の`S`が悪条件になるので対象外。
    for Structure {
        name,
        rows,
        dimensions,
        effects,
    } in structures().into_iter().take(3)
    {
        for (k_exog, k_endog, k_inst) in [(1, 1, 2), (1, 1, 3), (0, 2, 3)] {
            let make = || input_spec(&rows, &dimensions, k_exog, k_endog, k_inst);
            for gmm_type in [
                FeGmmType::OneStep,
                two_step(FeWeightType::Classical),
                two_step(FeWeightType::Robust),
            ] {
                let expected = dummy_variable_gmm_params(&make(), &gmm_type);
                let estimator = fit(make(), effects, gmm_type.clone()).unwrap();
                assert_params_close(
                    estimator.params(),
                    &expected,
                    &format!("{name} {k_exog}/{k_endog}/{k_inst} {gmm_type:?}"),
                );
            }
        }
    }
}

// ── クラスター重みの手計算 ─────────────────────────────────────────────

/// 個体（連続する`period`行ずつ）内の平均を引く。
pub(super) fn demean_by_blocks(column: &[f64], period: usize) -> Vec<f64> {
    column
        .chunks(period)
        .flat_map(|block| {
            let mean = block.iter().sum::<f64>() / block.len() as f64;
            block.iter().map(move |v| v - mean)
        })
        .collect()
}

pub(super) fn mat(columns: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(columns[0].len(), columns.len(), |i, j| columns[j][i])
}

/// `W = S⁻¹`のGMM `(X'ZWZ'X)⁻¹X'ZWZ'y`。LU分解で解く（本体のCholeskyとは別の経路）。
pub(super) fn gmm_estimate(z: &Mat<f64>, x: &Mat<f64>, y: &Mat<f64>, s: &Mat<f64>) -> Mat<f64> {
    let zx = z.transpose() * x;
    let zy = z.transpose() * y;
    let s_inv_zx = s.partial_piv_lu().solve(&zx);
    let s_inv_zy = s.partial_piv_lu().solve(&zy);
    let bread = zx.transpose() * &s_inv_zx;
    let meat = zx.transpose() * &s_inv_zy;
    bread.partial_piv_lu().solve(&meat)
}

/// `S = Σ_g (Σ_{i∈g} ê_i z_i)(Σ_{i∈g} ê_i z_i)'`（クラスター`group`の添字は行ごと）。
pub(super) fn cluster_s(z: &Mat<f64>, e: &Mat<f64>, group: &[usize]) -> Mat<f64> {
    let l = z.ncols();
    let n_groups = group.iter().max().unwrap() + 1;
    let mut s = Mat::<f64>::zeros(l, l);
    for g in 0..n_groups {
        let score: Vec<f64> = (0..l)
            .map(|a| {
                (0..z.nrows())
                    .filter(|&i| group[i] == g)
                    .map(|i| e.get(i, 0) * z.get(i, a))
                    .sum()
            })
            .collect();
        for a in 0..l {
            for b in 0..l {
                *s.get_mut(a, b) += score[a] * score[b];
            }
        }
    }
    s
}

/// 1-way（個体）のGMMを、素朴な平均除去とクラスター重みから手で求める。`estimates`回目（2以上）の
/// 推定まで進めて`(係数, 最後の推定に使ったS)`を返す（2が2段階GMM、3以上は反復）。
fn manual_one_way_gmm(
    input: &FeIvInput,
    group: &[usize],
    estimates: usize,
) -> (Vec<f64>, Mat<f64>) {
    let demean = |c: &Vec<f64>| demean_by_blocks(c, 4);
    manual_gmm(input, &demean, group, estimates)
}

/// 効果のダミーへの回帰の残差で固定効果を除き（`least_squares`。本体の変換とは別の経路）、
/// 同じ手計算のGMMを行う。どの固定効果の構造（2-way・不均衡・3次元）でも使える。
fn manual_dummy_projection_gmm(
    input: &FeIvInput,
    group: &[usize],
    estimates: usize,
) -> (Vec<f64>, Mat<f64>) {
    let manual = manual_dummy_projection_full(input, group, estimates);
    (manual.params, manual.s)
}

/// [`manual_dummy_projection_gmm`]の全部入り（`X̃`・`Z̃`・最終の構造残差を含む）。
pub(super) fn manual_dummy_projection_full(
    input: &FeIvInput,
    group: &[usize],
    estimates: usize,
) -> ManualGmm {
    let d = mat(&dummies(input.index().effect_codes()));
    let demean = |c: &Vec<f64>| -> Vec<f64> {
        let column = mat(std::slice::from_ref(c));
        let fit = least_squares(&d, &column).unwrap();
        let residual = &column - &d * &fit.params;
        (0..c.len()).map(|i| *residual.get(i, 0)).collect()
    };
    manual_gmm_full(input, &demean, group, estimates)
}

/// 手計算のGMMの途中の量（変換後の`X̃`・`Z̃`・最終の構造残差・最後の推定に使った`S`）。
pub(super) struct ManualGmm {
    pub(super) params: Vec<f64>,
    pub(super) s: Mat<f64>,
    pub(super) x: Mat<f64>,
    pub(super) z: Mat<f64>,
    pub(super) residuals: Mat<f64>,
}

fn manual_gmm(
    input: &FeIvInput,
    demean: &dyn Fn(&Vec<f64>) -> Vec<f64>,
    group: &[usize],
    estimates: usize,
) -> (Vec<f64>, Mat<f64>) {
    let manual = manual_gmm_full(input, demean, group, estimates);
    (manual.params, manual.s)
}

pub(super) fn manual_gmm_full(
    input: &FeIvInput,
    demean: &dyn Fn(&Vec<f64>) -> Vec<f64>,
    group: &[usize],
    estimates: usize,
) -> ManualGmm {
    let y = mat(&[demean(&input.y().to_vec())]);
    let exog: Vec<Vec<f64>> = input.x_exog().iter().map(demean).collect();
    let endog: Vec<Vec<f64>> = input.x_endog().iter().map(demean).collect();
    let instruments: Vec<Vec<f64>> = input.instruments().iter().map(demean).collect();
    let x = mat(&[exog.clone(), endog].concat());
    let z = mat(&[exog, instruments].concat());

    let ztz = z.transpose() * &z;
    let mut beta = gmm_estimate(&z, &x, &y, &ztz);
    let mut s = ztz;
    for _ in 1..estimates {
        let e = &y - &x * &beta;
        s = cluster_s(&z, &e, group);
        beta = gmm_estimate(&z, &x, &y, &s);
    }
    let params = (0..beta.nrows()).map(|j| *beta.get(j, 0)).collect();
    let residuals = &y - &x * &beta;
    ManualGmm {
        params,
        s,
        x,
        z,
        residuals,
    }
}

/// 1-way（個体）の手計算（`manual_one_way_gmm`の全部入り）。
pub(super) fn manual_one_way_full(
    input: &FeIvInput,
    group: &[usize],
    estimates: usize,
) -> ManualGmm {
    let demean = |c: &Vec<f64>| demean_by_blocks(c, 4);
    manual_gmm_full(input, &demean, group, estimates)
}

#[test]
fn cluster_weight_matches_manual_computation() {
    let entity_group: Vec<usize> = (0..40).map(|r| r / 4).collect();
    for (k_exog, k_endog, k_inst) in [(1, 1, 2), (1, 1, 3), (0, 2, 3)] {
        let make = || input_spec(&all_rows(), &[entity_labels()], k_exog, k_endog, k_inst);
        let (expected, s_expected) = manual_one_way_gmm(&make(), &entity_group, 2);
        let estimator = fit(make(), FeEffects::OneWay, two_step(entity_cluster_weight())).unwrap();
        let label = format!("{k_exog}/{k_endog}/{k_inst}");
        assert_params_close(estimator.params(), &expected, &label);
        let s = estimator.s_used();
        for a in 0..s.nrows() {
            for b in 0..s.ncols() {
                let (got, want) = (*s.get(a, b), *s_expected.get(a, b));
                assert!(
                    (got - want).abs() <= 1e-8 * want.abs().max(1e-6),
                    "{label} S[{a}][{b}]: {got} vs {want}"
                );
            }
        }
    }
}

/// 2-way（均衡・不均衡）と3次元の効果でも、個体でクラスタリングする重みの係数・`S`は、ダミーへの回帰の
/// 残差から組んだ手計算と一致する（2段階・3回の反復）。
#[test]
fn cluster_weight_matches_manual_computation_for_multi_way_effects() {
    for Structure {
        name,
        rows,
        dimensions,
        effects,
    } in structures().into_iter().skip(1)
    {
        let make = || input_spec(&rows, &dimensions, 1, 1, 2);
        let entity_labels_of_rows: Vec<String> =
            rows.iter().map(|&r| dimensions[0][r].clone()).collect();
        let group: Vec<usize> = GroupCodes::from_labels(&entity_labels_of_rows)
            .codes()
            .to_vec();
        let weight = FeWeightType::Cluster {
            groups: GroupCodes::from_labels_without_keys(&entity_labels_of_rows),
        };
        for estimates in [2_usize, 3] {
            let (expected, s_expected) = manual_dummy_projection_gmm(&make(), &group, estimates);
            let gmm_type = if estimates == 2 {
                two_step(weight.clone())
            } else {
                FeGmmType::Iterated {
                    weight: weight.clone(),
                    max_iter: estimates,
                    tol: 1e-300,
                }
            };
            let estimator =
                FeGmmEstimator::fit(make(), effects, gmm_type, false, FeCovType::Classical, 0.95)
                    .unwrap();
            let label = format!("{name} estimates={estimates}");
            assert_params_close(estimator.params(), &expected, &label);
            let s = estimator.s_used();
            for a in 0..s.nrows() {
                for b in 0..s.ncols() {
                    let (got, want) = (*s.get(a, b), *s_expected.get(a, b));
                    assert!(
                        (got - want).abs() <= 1e-7 * want.abs().max(1e-6),
                        "{label} S[{a}][{b}]: {got} vs {want}"
                    );
                }
            }
        }
    }
}

/// 時点でクラスタリングする重み（1-wayの個体効果。クラスターは固定効果の次元と別の列でもよい）。
/// 行は個体ごとなので、時点は`r % 4`。
#[test]
fn cluster_weight_by_a_column_other_than_the_effect_matches_manual_computation() {
    let time_group: Vec<usize> = (0..40).map(|r| r % 4).collect();
    // `l = 3`、時点`G = 4`（過剰識別なので`G >= l`）。
    let make = || input_spec(&all_rows(), &[entity_labels()], 1, 1, 2);
    let (expected, _) = manual_one_way_gmm(&make(), &time_group, 2);
    let estimator = fit(make(), FeEffects::OneWay, two_step(time_cluster_weight())).unwrap();
    assert_params_close(estimator.params(), &expected, "time cluster");
}

/// 2段階GMMの`s_used`は、1段階目（2SLS）の構造残差から作ったロバストな`Σ ê²z̃z̃'`。
#[test]
fn two_step_robust_s_used_is_built_from_the_first_step_residuals() {
    let make = || input_spec(&all_rows(), &[entity_labels()], 1, 1, 2);
    // 各行を別クラスターにしたクラスター重みはロバスト重みと同じ。
    let singleton_groups: Vec<usize> = (0..40).collect();
    let (expected, s_expected) = manual_one_way_gmm(&make(), &singleton_groups, 2);
    let estimator = fit(make(), FeEffects::OneWay, two_step(FeWeightType::Robust)).unwrap();
    assert_params_close(estimator.params(), &expected, "robust");
    let s = estimator.s_used();
    for a in 0..s.nrows() {
        for b in 0..s.ncols() {
            let (got, want) = (*s.get(a, b), *s_expected.get(a, b));
            assert!(
                (got - want).abs() <= 1e-8 * want.abs().max(1e-6),
                "S[{a}][{b}]: {got} vs {want}"
            );
        }
    }
}

/// 反復GMM（クラスター重み）は、手で3回・4回再推定した係数・`S`と一致する（`tol`を極小にして
/// `max_iter`まで回す）。
#[test]
fn iterated_cluster_weight_matches_manual_iteration() {
    let entity_group: Vec<usize> = (0..40).map(|r| r / 4).collect();
    let make = || input_spec(&all_rows(), &[entity_labels()], 1, 1, 2);
    for max_iter in [3_usize, 4] {
        let (expected, s_expected) = manual_one_way_gmm(&make(), &entity_group, max_iter);
        let estimator = FeGmmEstimator::fit(
            make(),
            FeEffects::OneWay,
            FeGmmType::Iterated {
                weight: entity_cluster_weight(),
                max_iter,
                tol: 1e-300,
            },
            false,
            FeCovType::Classical,
            0.95,
        )
        .unwrap();
        assert_eq!(estimator.n_iter(), max_iter as i64);
        assert_params_close(
            estimator.params(),
            &expected,
            &format!("max_iter={max_iter}"),
        );
        let s = estimator.s_used();
        for a in 0..s.nrows() {
            for b in 0..s.ncols() {
                let (got, want) = (*s.get(a, b), *s_expected.get(a, b));
                assert!(
                    (got - want).abs() <= 1e-8 * want.abs().max(1e-6),
                    "max_iter={max_iter} S[{a}][{b}]: {got} vs {want}"
                );
            }
        }
    }
}

/// 個体ごとの並びを保ったまま個体の順を逆にしても、係数は変わらない。
#[test]
fn coefficients_do_not_depend_on_the_order_of_entities() {
    let forward = input_with(&all_rows(), &[entity_labels()]);
    let reversed_rows: Vec<usize> = (0..10)
        .rev()
        .flat_map(|entity| (0..4).map(move |t| entity * 4 + t))
        .collect();
    let reversed = input_with(&reversed_rows, &[entity_labels()]);
    let a = fit(forward, FeEffects::OneWay, two_step(FeWeightType::Robust)).unwrap();
    let b = fit(reversed, FeEffects::OneWay, two_step(FeWeightType::Robust)).unwrap();
    assert_params_close(a.params(), &params_of(&b), "reversed entities");
}
