//! `B_TSLS`(Nagarバイアスの上限)。

use super::*;

/// 決定的な疑似乱数の`2K×2K`の半正定値行列`W = A'A`(列ごとにスケールが違う)。
fn random_psd(k: usize, seed: u64) -> Mat<f64> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    };
    let rows = 2 * k + 2;
    let a = Mat::from_fn(rows, 2 * k, |_, col| {
        next() * 10f64.powi((col % 3) as i32 - 1)
    });
    a.transpose() * &a
}

fn blocks(w: &Mat<f64>, k: usize) -> (Mat<f64>, Mat<f64>, Mat<f64>) {
    let block = |r: usize, c: usize| Mat::from_fn(k, k, |a, b| *w.get(r * k + a, c * k + b));
    (block(0, 0), block(0, 1), block(1, 1))
}

/// 実装と別のコードで書いた総当たり: `β`を`center`からの距離が対数間隔の細かい格子で動かし、定義どおりに
/// `max_λ |tr(S12) − 2λ| / √(tr(S1)·tr(S2))`を評価する。
fn brute_force_bias_bound(w: &Mat<f64>, k: usize, steps: usize, center: f64) -> f64 {
    let (w1, w12, w2) = blocks(w, k);
    let tr = |m: &Mat<f64>| (0..k).map(|a| *m.get(a, a)).sum::<f64>();
    let mut best = 0.0f64;
    let mut betas = vec![center];
    for step in 0..=steps {
        let magnitude = (-15.0 + 30.0 * step as f64 / steps as f64).exp();
        betas.push(center + magnitude);
        betas.push(center - magnitude);
    }
    for beta in betas {
        let s12 = Mat::from_fn(k, k, |a, b| {
            0.5 * (*w12.get(a, b) + *w12.get(b, a)) - beta * *w2.get(a, b)
        });
        let s1 = Mat::from_fn(k, k, |a, b| {
            *w1.get(a, b) - beta * (*w12.get(a, b) + *w12.get(b, a)) + beta * beta * *w2.get(a, b)
        });
        let eigenvalues = s12.self_adjoint_eigenvalues(Side::Lower).unwrap();
        let denominator = (tr(&s1) * tr(&w2)).sqrt();
        for lambda in [eigenvalues[0], eigenvalues[k - 1]] {
            best = best.max((tr(&s12) - 2.0 * lambda).abs() / denominator);
        }
    }
    best
}

/// 同次分散(`W = Ω⊗I_K`)では`B_TSLS = |1 − 2/K|`(上限は`β → ±∞`で達成される)。
#[test]
fn bias_bound_is_one_minus_two_over_k_for_a_kronecker_covariance() {
    let omega = [[1.5, 0.6], [0.6, 2.0]];
    for k in [1usize, 2, 3, 5, 30] {
        let block =
            |r: usize, c: usize| Mat::from_fn(k, k, |a, b| if a == b { omega[r][c] } else { 0.0 });
        let bound = tsls_bias_bound(&block(0, 0), &block(0, 1), &block(1, 1)).unwrap();
        let expected = (1.0 - 2.0 / k as f64).abs();
        // K=2では期待値が0なので、相対ではなく絶対の許容で比べる。
        assert!(
            (bound - expected).abs() <= 1e-12,
            "K={k}: {bound} vs {expected}"
        );
    }
}

#[test]
fn bias_bound_agrees_with_a_brute_force_grid_and_stays_within_zero_and_one() {
    let mut interior = 0;
    for k in [2usize, 3] {
        for seed in 0..6u64 {
            let w = random_psd(k, 1000 * k as u64 + seed);
            let (w1, w12, w2) = blocks(&w, k);
            let bound = tsls_bias_bound(&w1, &w12, &w2).unwrap();
            let brute = brute_force_bias_bound(&w, k, 4_000, 0.0);
            assert!((0.0..=1.0).contains(&bound), "K={k} seed={seed}: {bound}");
            // 最適化は格子より悪くならず、格子より極端に良くもならない。
            assert!(
                bound >= brute - 1e-12,
                "K={k} seed={seed}: {bound} < {brute}"
            );
            assert!(
                bound <= brute + 1e-6,
                "K={k} seed={seed}: {bound} > {brute}"
            );
            let tr2: f64 = (0..k).map(|a| *w2.get(a, a)).sum();
            let eig = w2.self_adjoint_eigenvalues(Side::Lower).unwrap();
            let limit = (1.0 - 2.0 * eig[0] / tr2)
                .abs()
                .max((1.0 - 2.0 * eig[k - 1] / tr2).abs());
            assert!(bound >= limit - 1e-12);
            if bound > limit + 1e-6 {
                interior += 1;
            }
        }
    }
    // 上限が`β`の有限の値で達成されるケースも含まれる(極限だけを返していない)。
    assert!(interior > 0, "no case had an interior maximum");
}

/// 山が`|β|`のとても大きいところにある場合(`tan`の格子だと端の1区間に埋もれて見落とした)。
/// 細かい総当たりより悪くならない。
#[test]
fn a_maximum_at_a_very_large_beta_is_found() {
    let (k, seed) = (2usize, 1000 * 2 + 7);
    let w = random_psd(k, seed);
    let (w1, w12, w2) = blocks(&w, k);
    let bound = tsls_bias_bound(&w1, &w12, &w2).unwrap();
    let brute = brute_force_bias_bound(&w, k, 20_000, 0.0);
    assert!(bound >= brute - 1e-12, "{bound} < {brute}");
    assert!(bound <= brute + 1e-6, "{bound} > {brute}");
}

/// `W`のスケール、被説明変数の単位、操作変数の回転(`I⊗H`)で`B_TSLS`は変わらない。
#[test]
fn bias_bound_is_invariant_to_scaling_and_to_rotating_the_instruments() {
    let k = 4;
    let w = random_psd(k, 77);
    let (w1, w12, w2) = blocks(&w, k);
    let base = tsls_bias_bound(&w1, &w12, &w2).unwrap();
    let scaled = |m: &Mat<f64>, c: f64| Mat::from_fn(k, k, |a, b| *m.get(a, b) * c);
    // `W`全体を定数倍。
    close(
        tsls_bias_bound(&scaled(&w1, 7.0), &scaled(&w12, 7.0), &scaled(&w2, 7.0)).unwrap(),
        base,
        1e-9,
    );
    // 被説明変数の単位を`c`倍(`β → cβ`): `W1 → c²W1`、`W12 → cW12`。
    let c = 300.0;
    close(
        tsls_bias_bound(&scaled(&w1, c * c), &scaled(&w12, c), &w2).unwrap(),
        base,
        1e-9,
    );
    // 操作変数の回転(Householder反射)。
    let v = [1.0, -2.0, 0.5, 3.0];
    let norm: f64 = v.iter().map(|x| x * x).sum();
    let h = Mat::from_fn(k, k, |a, b| {
        (if a == b { 1.0 } else { 0.0 }) - 2.0 * v[a] * v[b] / norm
    });
    let rotate = |m: &Mat<f64>| &h * m * h.transpose();
    close(
        tsls_bias_bound(&rotate(&w1), &rotate(&w12), &rotate(&w2)).unwrap(),
        base,
        1e-9,
    );
}

#[test]
fn bias_bound_is_undefined_for_degenerate_covariances() {
    let k = 3;
    let identity = Mat::<f64>::identity(k, k);
    let zero = Mat::<f64>::zeros(k, k);
    // 被説明変数の誘導形の誤差が内生変数のそれの定数倍(`W1 = b²W2`、`W12 = bW2`): ベンチマークが0になる`β = b`がある。
    assert!(tsls_bias_bound(&(&identity * 4.0), &(&identity * 2.0), &identity).is_none());
    assert!(tsls_bias_bound(&zero, &zero, &identity).is_none());
    assert!(tsls_bias_bound(&identity, &zero, &zero).is_none());
}

/// 操作変数が1個なら`B_TSLS`は常に1(`W`が何でも)。一般化版の臨界値は単純版に一致する。
#[test]
fn a_single_instrument_has_a_bound_of_one() {
    for seed in 0..5u64 {
        let w = random_psd(1, 500 + seed);
        let (w1, w12, w2) = blocks(&w, 1);
        let bound = tsls_bias_bound(&w1, &w12, &w2).unwrap();
        assert!((bound - 1.0).abs() <= 1e-12, "seed={seed}: {bound}");
    }
    let (endog, exog, z) = large(4);
    let y = large_y(&endog, &exog);
    let z1 = column(&z, 0);
    let eff = crate::shared::effective_f::effective_f(
        &endog,
        &exog,
        &z1,
        0,
        ScoreCovariance::Hc(HcVariant::Hc1),
        0.05,
        Some(&y),
    )
    .unwrap();
    for (general, simplified) in eff.tsls.unwrap().iter().zip(eff.simplified_tsls) {
        close(general.critical_value, simplified.critical_value, 1e-9);
    }
}

/// 同次分散で操作変数が2個なら`B_TSLS = 0`で`x = 0`: 臨界値は中心カイ二乗分布の上側5%点(自由度2)を2で割った値。
#[test]
fn two_instruments_with_a_kronecker_covariance_give_a_central_chi_squared() {
    let (df, cv) = patnaik_critical_value(&[1.0, 1.0], 0.0, 0.05).unwrap();
    close(df, 2.0, 1e-12);
    // scipy.stats.chi2.ppf(0.95, 2) / 2
    close(cv, 2.995732273553991, 1e-9);
    let (endog, exog, z) = small();
    let y = small_y_for_bias_bound();
    let two = Mat::from_fn(12, 2, |i, j| *z.get(i, j));
    let eff = crate::shared::effective_f::effective_f(
        &endog,
        &exog,
        &two,
        0,
        ScoreCovariance::Classical,
        0.05,
        Some(&y),
    )
    .unwrap();
    assert!(eff.bias_bound.unwrap().abs() <= 1e-12);
    for general in eff.tsls.unwrap() {
        assert!(general.x.abs() <= 1e-10);
        close(general.critical_value, 2.995732273553991, 1e-8);
    }
}

fn small_y_for_bias_bound() -> Mat<f64> {
    Mat::from_fn(12, 1, |i, _| (i as f64 * 0.7).sin() + 0.5 * SMALL_X[i])
}

#[test]
fn golden_section_finds_the_maximum_of_a_unimodal_function() {
    let f = |x: f64| 2.0 - (x - 0.3) * (x - 0.3);
    close(maximize_golden(&f, -1.0, 2.0), 2.0, 1e-12);
    // 評価に失敗した点(負の無限大)があっても、残りの点の最大値を返す。
    let with_gap = |x: f64| {
        if x < 0.0 {
            f64::NEG_INFINITY
        } else {
            1.0 - (x - 0.5).powi(2)
        }
    };
    close(maximize_golden(&with_gap, -1.0, 2.0), 1.0, 1e-10);
}

/// 被説明変数と内生変数の誘導形の誤差がほぼ比例する(`min tr(S1)/tr(W1) ≈ 1e-8`)と、`β* = tr(W12)/tr(W2)`の
/// 近くに鋭い構造ができる。`β*`からの距離が対数間隔の細かい総当たりより悪くならない。
#[test]
fn a_nearly_proportional_reduced_form_is_resolved_near_the_minimizer_of_the_benchmark() {
    for scale in [1e-3, 1e-4, 1e-5] {
        check_nearly_proportional(scale);
    }
}

fn check_nearly_proportional(scale: f64) {
    let k = 3;
    let b = 2.5_f64;
    let mut state = 12345u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    };
    let rows = 10;
    let a_x = Mat::from_fn(rows, k, |_, _| next());
    let a_e = Mat::from_fn(rows, k, |_, _| next());
    // 被説明変数の列 = b·(内生変数の列) + scale·(別の誤差)。
    let a = Mat::from_fn(rows, 2 * k, |i, col| {
        if col < k {
            b * *a_x.get(i, col) + scale * *a_e.get(i, col)
        } else {
            *a_x.get(i, col - k)
        }
    });
    let w = a.transpose() * &a;
    let (w1, w12, w2) = blocks(&w, k);
    let tr = |m: &Mat<f64>| (0..k).map(|a| *m.get(a, a)).sum::<f64>();
    let min_tr_s1 = tr(&w1) - tr(&w12) * tr(&w12) / tr(&w2);
    assert!(
        min_tr_s1 / tr(&w1) < 1e-6,
        "not nearly proportional: {}",
        min_tr_s1 / tr(&w1)
    );
    let bound = tsls_bias_bound(&w1, &w12, &w2).unwrap();
    let brute = brute_force_bias_bound(&w, k, 8_000, tr(&w12) / tr(&w2));
    assert!(bound >= brute - 1e-9, "scale={scale}: {bound} < {brute}");
    assert!(bound <= brute + 1e-6, "scale={scale}: {bound} > {brute}");
}
