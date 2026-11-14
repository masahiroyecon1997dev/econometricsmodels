//! Sargan過剰識別検定。ダミー変数を明示した2SLSでの独立な再計算、`σ̂²`の分母、帰無仮説下のサイズを確かめる。

use faer::Mat;
use statrs::distribution::{ChiSquared, ContinuousCDF};

use super::{
    N_ENTITIES, all_rows, entity_labels, fit, input_spec, one_way_input, third_labels, time_labels,
};
use crate::panel::common::TimeKeys;
use crate::panel::fe::{FeCovType, FeEffects};
use crate::panel::fe_iv::{FeIvInput, RoleColumns};
use crate::shared::group_codes::GroupCodes;
use crate::shared::least_squares::least_squares;

fn mat(columns: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(columns[0].len(), columns.len(), |i, j| columns[j][i])
}

/// 次元ごとのダミー列（1つ目の次元は全水準、2つ目以降は最初の水準を落とす）。
fn dummies(dimensions: &[GroupCodes]) -> Vec<Vec<f64>> {
    let mut columns = Vec::new();
    for (d, dimension) in dimensions.iter().enumerate() {
        for level in usize::from(d > 0)..dimension.n_groups() {
            columns.push(
                dimension
                    .codes()
                    .iter()
                    .map(|&code| f64::from(u8::from(code == level)))
                    .collect(),
            );
        }
    }
    columns
}

/// ダミー込みの2SLSの構造残差`e`から、`e'Z(Z'Z)⁻¹Z'e / (e'e/(n − ダミーの数))`を求める
/// （`Z = [x_exog, 操作変数, ダミー]`）。`fit`のwithin変換・`fe_sargan_test`は通らない。分子の同値
/// （Frisch-Waugh-Lovell）を確かめるもので、分母の選択は分母の手計算のテストと棄却率のテストが担う。
fn dummy_variable_sargan(input: &FeIvInput) -> f64 {
    let n = input.nobs();
    let d = dummies(input.index().effect_codes());
    let (ke, kn) = (input.k_exog(), input.k_endog());

    let mut x_full: Vec<Vec<f64>> = input.x_exog().to_vec();
    x_full.extend(input.x_endog().iter().cloned());
    x_full.extend(d.iter().cloned());
    let mut z_full: Vec<Vec<f64>> = input.x_exog().to_vec();
    z_full.extend(input.instruments().iter().cloned());
    z_full.extend(d.iter().cloned());

    let z_mat = mat(&z_full);
    let mut x_hat_columns = x_full.clone();
    for j in 0..kn {
        let first = least_squares(&z_mat, &mat(&[input.x_endog()[j].clone()])).unwrap();
        let fitted = &z_mat * &first.params;
        x_hat_columns[ke + j] = (0..n).map(|i| *fitted.get(i, 0)).collect();
    }
    let y = mat(&[input.y().to_vec()]);
    let second = least_squares(&mat(&x_hat_columns), &y).unwrap();
    let e = &y - mat(&x_full) * &second.params;

    let projected = &z_mat * least_squares(&z_mat, &e).unwrap().params;
    let quad: f64 = (0..n).map(|i| e.get(i, 0) * projected.get(i, 0)).sum();
    let ssr: f64 = (0..n).map(|i| e.get(i, 0).powi(2)).sum();
    quad / (ssr / (n - d.len()) as f64)
}

fn check(input: FeIvInput, effects: FeEffects, label: &str) {
    let expected = dummy_variable_sargan(&input);
    let df = input.k_instruments() - input.k_endog();
    let fitted = fit(input, effects, FeCovType::Classical).unwrap();
    let statistic = fitted.sargan_statistic().unwrap();
    assert!(
        (statistic - expected).abs() <= 1e-9 * expected.abs(),
        "{label}: {statistic} vs {expected}"
    );
    assert_eq!(fitted.sargan_df(), Some(df), "{label}");
    let p = 1.0 - ChiSquared::new(df as f64).unwrap().cdf(statistic);
    assert!(
        (fitted.sargan_p_value().unwrap() - p).abs() < 1e-12,
        "{label}"
    );
}

#[test]
fn matches_the_dummy_variable_2sls_across_structures_and_instrument_counts() {
    let entity = entity_labels();
    let time = time_labels();
    let third = third_labels();
    let unbalanced: Vec<usize> = all_rows()
        .into_iter()
        .filter(|r| ![3, 17, 22, 30].contains(r))
        .collect();
    // (外生変数, 内生変数, 操作変数)。過剰識別の度合いと外生変数・内生変数の数を変える。
    for spec in [(1, 1, 2), (0, 1, 3), (2, 1, 3), (1, 2, 3), (0, 2, 3)] {
        for (rows, dims, effects, name) in [
            (all_rows(), vec![entity.clone()], FeEffects::OneWay, "1-way"),
            (
                all_rows(),
                vec![entity.clone(), time.clone()],
                FeEffects::TwoWay,
                "2-way",
            ),
            (
                unbalanced.clone(),
                vec![entity.clone(), time.clone()],
                FeEffects::TwoWay,
                "2-way unbalanced",
            ),
            (
                all_rows(),
                vec![entity.clone(), time.clone(), third.clone()],
                FeEffects::MultiWay,
                "3-way",
            ),
        ] {
            let input = input_spec(&rows, &dims, spec.0, spec.1, spec.2);
            check(input, effects, &format!("{name} spec {spec:?}"));
        }
    }
}

/// 等分散前提なので、`cov_type`を変えても値は同じ。
#[test]
fn does_not_depend_on_the_covariance_type() {
    let reference = fit(one_way_input(), FeEffects::OneWay, FeCovType::Classical)
        .unwrap()
        .sargan_statistic()
        .unwrap();
    for cov in [
        FeCovType::Hc1,
        FeCovType::Hc3,
        FeCovType::Cluster { groups: None },
        FeCovType::Dk {
            bandwidth: None,
            time: TimeKeys::lexicographic(time_labels()),
        },
    ] {
        let statistic = fit(one_way_input(), FeEffects::OneWay, cov)
            .unwrap()
            .sargan_statistic()
            .unwrap();
        assert!((statistic - reference).abs() <= 1e-12 * reference.abs());
    }
}

/// 分母が`e'e/n`でなく`e'e/(n − neffects)`であること: 構造残差から手で作る。`e'e/n`にすると
/// `n / (n − neffects)`倍だけ小さくなる。
#[test]
fn denominator_is_the_residual_degrees_of_freedom_of_the_absorbed_effects() {
    let fitted = fit(one_way_input(), FeEffects::OneWay, FeCovType::Classical).unwrap();
    let input = fitted.input();
    let (n, neffects) = (input.nobs() as f64, N_ENTITIES as f64);
    let e = fitted.residuals();
    let ssr: f64 = (0..input.nobs()).map(|i| e.get(i, 0).powi(2)).sum();
    // 構造残差`e`と`Z̃`の射影: 第一段階の設計行列（`within_input().x()`、`[x̃_exog, z̃]`）を使う。
    let z = fitted.first_stages()[0].within_input().x();
    let projected = z * least_squares(z, e).unwrap().params;
    let quad: f64 = (0..input.nobs())
        .map(|i| e.get(i, 0) * projected.get(i, 0))
        .sum();
    let statistic = fitted.sargan_statistic().unwrap();
    assert!((statistic - quad / (ssr / (n - neffects))).abs() <= 1e-9 * statistic.abs());
    assert!((statistic - quad / (ssr / n)).abs() > 1e-3 * statistic.abs());
}

/// 内生変数が無いときは、操作変数があっても検定しない（第一段階が無い）。
#[test]
fn is_none_without_endogenous_variables() {
    let input = input_spec(&all_rows(), &[entity_labels()], 1, 0, 2);
    let fitted = fit(input, FeEffects::OneWay, FeCovType::Classical).unwrap();
    assert_eq!(fitted.sargan_statistic(), None);
    assert_eq!(fitted.sargan_p_value(), None);
    assert_eq!(fitted.sargan_df(), None);
}

#[test]
fn is_none_when_just_identified() {
    let input = input_spec(&all_rows(), &[entity_labels()], 1, 1, 1);
    let fitted = fit(input, FeEffects::OneWay, FeCovType::Classical).unwrap();
    assert_eq!(fitted.sargan_statistic(), None);
    assert_eq!(fitted.sargan_p_value(), None);
    assert_eq!(fitted.sargan_df(), None);
}

/// 乱数（線形合同法）。標準正規はBox-Muller。再現できる決定的な擬似データを作るためだけに使う。
struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// 帰無仮説（操作変数が外生）の下で、`σ̂² = e'e/(n − neffects)`の統計量が名目水準に近い頻度で
/// 棄却すること。`e'e/n`だと`T = 3`で10〜12%まで膨らむ（実測11.9%）。固定効果は操作変数・内生変数と相関させ、
/// 誤差は内生変数の誤差と相関させる（内生性あり）。
#[test]
fn rejection_rate_under_the_null_is_close_to_the_nominal_level() {
    const N: usize = 100;
    const REPS: usize = 1500;
    let critical = {
        let chi2 = ChiSquared::new(1.0).unwrap();
        // 5%点: `cdf`の逆関数の代わりに二分法で求める。
        let (mut lo, mut hi) = (0.0_f64, 20.0_f64);
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if chi2.cdf(mid) < 0.95 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };
    for periods in [3_usize, 6] {
        let mut rng = Lcg(12345 + periods as u64);
        let entity: Vec<String> = (0..N)
            .flat_map(|i| std::iter::repeat_n(i.to_string(), periods))
            .collect();
        let groups = GroupCodes::from_labels(&entity);
        let mut rejections = 0;
        for _ in 0..REPS {
            let alpha: Vec<f64> = (0..N).map(|_| rng.normal()).collect();
            let mut exog = Vec::new();
            let mut endog = Vec::new();
            let (mut z1, mut z2) = (Vec::new(), Vec::new());
            let mut y = Vec::new();
            for &a in &alpha {
                for _ in 0..periods {
                    let (w, s1, s2) = (rng.normal(), rng.normal(), rng.normal());
                    let v = rng.normal();
                    let u = 0.5 * v + 0.866 * rng.normal();
                    let x = 0.6 * s1 + 0.6 * s2 + 0.5 * a + v;
                    exog.push(w);
                    endog.push(x);
                    z1.push(s1 + 0.3 * a);
                    z2.push(s2);
                    y.push(0.5 * w + x + a + u);
                }
            }
            let input = FeIvInput::from_group_codes(
                &y,
                "y".to_string(),
                RoleColumns {
                    names: vec!["w".to_string()],
                    columns: &[exog],
                },
                RoleColumns {
                    names: vec!["x".to_string()],
                    columns: &[endog],
                },
                RoleColumns {
                    names: vec!["z1".to_string(), "z2".to_string()],
                    columns: &[z1, z2],
                },
                vec![groups.clone()],
            )
            .unwrap();
            let fitted = fit(input, FeEffects::OneWay, FeCovType::Classical).unwrap();
            if fitted.sargan_statistic().unwrap() > critical {
                rejections += 1;
            }
        }
        let rate = rejections as f64 / REPS as f64;
        // 標準誤差は約0.6%。`e'e/n`の棄却率（`T = 3`で10〜12%、`T = 6`で約7%）は範囲の外。
        // 分母の取り違えを実質的に止めるのは`T = 3`（`T = 6`は範囲との差が小さい）。
        assert!(
            (0.035..=0.065).contains(&rate),
            "T = {periods}: rejection rate {rate}"
        );
    }
}
