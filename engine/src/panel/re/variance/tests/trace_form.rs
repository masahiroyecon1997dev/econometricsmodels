//! 1-wayのSwamy-Arora分散成分（`plm`の`swar`のトレース形）を、観測ごとの行列で書いた参照と
//! バランスパネルの閉形式に照らすテスト。

use crate::panel::re::input::ReEffects;
use crate::panel::re::input::ReInput;
use crate::panel::re::test_support::*;
use crate::panel::re::variance::swamy_arora_variance_components;

/// `a·v = b`をピボット付きのガウス消去で解く（`p`が小さい参照計算用）。
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let p = a.len();
    for col in 0..p {
        let pivot = (col..p)
            .max_by(|&r, &s| a[r][col].abs().total_cmp(&a[s][col].abs()))
            .unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in 0..p {
            if row != col {
                let factor = a[row][col] / a[col][col];
                for c in 0..p {
                    a[row][c] -= factor * a[col][c];
                }
                for c in 0..b[row].len() {
                    b[row][c] -= factor * b[col][c];
                }
            }
        }
    }
    (0..p)
        .map(|r| b[r].iter().map(|v| v / a[r][r]).collect())
        .collect()
}

fn cross(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (p, q) = (a[0].len(), b[0].len());
    (0..p)
        .map(|r| {
            (0..q)
                .map(|c| a.iter().zip(b).map(|(x, y)| x[r] * y[c]).sum())
                .collect()
        })
        .collect()
}

/// `plm::ercomp`の`swar`・`dfcor = 3`の式を、観測ごとの行列でそのまま書いた参照。
/// `Z_B`は個体平均を観測に展開した`[1, x̄_i]`、`Z_S`は個体和を展開した行、`Z`は生の`[1, x_it]`。
/// `Q_id`は`Z_B`へのプールドOLS（= `T_i`重みのbetween回帰）の残差平方和で、
/// `σ_u² = (Q_id − σ_ε²(N − p)) / (O − tr((Z_B'Z_B)⁻¹ Z_S'Z))`。
///
/// between回帰で列を落とさない入力（個体平均が個体間で一定の列・別名の列が無い入力）に限る
/// （`p`を説明変数の数+1で数え、`Z_B'Z_B`を逆行列にするため）。
fn observation_level_sigma2_u(input: &ReInput, sigma2_eps: f64) -> f64 {
    let n = input.nobs();
    let codes = input.entity_codes().codes();
    let n_entities = input.entity_codes().n_groups();
    let row = |i: usize| -> Vec<f64> {
        std::iter::once(1.0)
            .chain(input.x().iter().map(|c| c[i]))
            .collect()
    };
    let p = input.x().len() + 1;
    let mut sums = vec![vec![0.0; p]; n_entities];
    let mut y_sums = vec![0.0; n_entities];
    let mut counts = vec![0.0; n_entities];
    for (i, &g) in codes.iter().enumerate() {
        for (s, v) in sums[g].iter_mut().zip(row(i)) {
            *s += v;
        }
        y_sums[g] += input.y()[i];
        counts[g] += 1.0;
    }
    let z: Vec<Vec<f64>> = (0..n).map(row).collect();
    let z_s: Vec<Vec<f64>> = (0..n).map(|i| sums[codes[i]].clone()).collect();
    let z_b: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            sums[codes[i]]
                .iter()
                .map(|v| v / counts[codes[i]])
                .collect()
        })
        .collect();
    let y_b: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![y_sums[codes[i]] / counts[codes[i]]])
        .collect();

    let ztz_b = cross(&z_b, &z_b);
    let beta = solve(ztz_b.clone(), cross(&z_b, &y_b));
    let q_id: f64 = (0..n)
        .map(|i| {
            let fitted: f64 = (0..p).map(|j| z_b[i][j] * beta[j][0]).sum();
            (y_b[i][0] - fitted).powi(2)
        })
        .sum();
    let b_inv_s = solve(ztz_b, cross(&z_s, &z));
    let trace: f64 = (0..p).map(|j| b_inv_s[j][j]).sum();
    ((q_id - sigma2_eps * (n_entities - p) as f64) / (n as f64 - trace)).max(0.0)
}

#[test]
fn one_way_sigma2_u_matches_the_observation_level_trace_form() {
    // 不均衡パネル（傾き1個・傾き2個・時間不変の列と時点間でのみ変わる列を含む行の部分集合・極端な不均衡）で、`between_regression`の
    // `√T_i`重み付きQRとレバレッジから求めた値が、行列をそのまま組んだ参照と一致する。
    let unbalanced_time_invariant = subset_input(
        &(0..24).filter(|&r| r != 5 && r != 14).collect::<Vec<_>>(),
        &["x1", "xi", "xt"],
        false,
    );
    let cases = [
        ("one slope", cov_type_reference_input()),
        ("two slopes", hausman_two_slope_input(None)),
        ("time-invariant columns", unbalanced_time_invariant),
        // T_i={2, 2, 15}・N = p + 1（between回帰の残差自由度が1で、分母`O − Σ T_i h_i`が小さい境界）。
        ("extreme imbalance", extreme_unbalanced_input()),
    ];
    for (label, input) in cases {
        let vc = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();
        let expected = observation_level_sigma2_u(&input, vc.sigma2_eps);
        assert!(expected > 0.0, "{label}: the reference must not be clipped");
        assert_rel(vc.sigma2_u, expected, 1e-12, label);
    }
}

#[test]
fn one_way_sigma2_u_matches_plm_with_two_slopes() {
    // R: `plm::ercomp(y ~ x1 + x2, method = "swar")$sigma2`（不均衡、`dfcor = 3`）。
    let vc =
        swamy_arora_variance_components(&hausman_two_slope_input(None), ReEffects::OneWay, 0.95)
            .unwrap();
    assert_rel(vc.sigma2_eps, 1.181_836_859_923_826_3, 1e-12, "sigma2_eps");
    assert_rel(vc.sigma2_u, 3.661_491_355_272_908, 1e-12, "sigma2_u");
}

#[test]
fn one_way_sigma2_u_equals_the_closed_form_when_all_entities_have_the_same_count() {
    // 全個体の`T_i = T`なら、トレース形は`SSR_b/(N − p) − σ_ε²/T`（`linearmodels`・`plm`の
    // バランスの式。重みなしのbetween回帰）に代数的に一致する。式を切り替えずに済む根拠。
    // 4個体×3期（`re_estimator_fit_hausman_matches_plm_aux_on_balanced_panel`と同じデータ）。
    let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]);
    let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0, 4.0, 3.0, 7.0, 2.0, 5.0];
    let y = [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0, 5.0, 9.5, 4.0, 7.3];
    let input = ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();
    let vc = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();
    let n_entities = input.entity_codes().n_groups();
    let t = (input.nobs() / n_entities) as f64;
    let means = &vc.means;
    let p = input.x().len() + 1;
    let z: Vec<Vec<f64>> = (0..n_entities)
        .map(|i| {
            std::iter::once(1.0)
                .chain(means.x.iter().map(|c| c[i]))
                .collect()
        })
        .collect();
    let y: Vec<Vec<f64>> = means.y.iter().map(|v| vec![*v]).collect();
    let beta = solve(cross(&z, &z), cross(&z, &y));
    let ssr: f64 = (0..n_entities)
        .map(|i| (means.y[i] - (0..p).map(|j| z[i][j] * beta[j][0]).sum::<f64>()).powi(2))
        .sum();
    let closed_form = ssr / (n_entities - p) as f64 - vc.sigma2_eps / t;
    assert!(closed_form > 0.0);
    assert_rel(vc.sigma2_u, closed_form, 1e-12, "balanced");
}
