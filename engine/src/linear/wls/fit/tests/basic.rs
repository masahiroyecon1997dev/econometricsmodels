//! `fit`の基本挙動（既知の係数の復元・適合度・getter・成功する完全適合や定数のyなど）のテスト。

use crate::linear::common::row_time_order;
use crate::linear::ols::CovType;
use crate::linear::ols::OlsEstimator;
use crate::linear::ols::OlsInput;
use crate::linear::wls::WlsEstimator;
use crate::shared::group_codes::GroupCodes;

#[test]
fn fitted_values_equals_y_minus_residuals() {
    // `ols/fit/tests/basic.rs`の同名テスト（`fitted_values_equals_y_minus_residuals`）と同じ不変条件
    // （`fitted_values()`と`residuals()`が同じ`original_scale_fitted_and_residuals`
    // 呼び出しから一貫して導かれていることの確認）。
    let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
    let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let weights = vec![1.0, 4.0, 0.25, 9.0, 2.0];

    let wls = WlsEstimator::fit(
        &y,
        &x_columns,
        vec!["x1".to_string()],
        true,
        "y".to_string(),
        &weights,
        CovType::Classical,
        0.95,
    )
    .unwrap();

    for (i, &y_i) in y.iter().enumerate() {
        let expected = y_i - wls.residuals()[i];
        assert!((wls.fitted_values()[i] - expected).abs() < 1e-9);
    }
}

#[test]
fn fit_recovers_known_coefficients_with_nonuniform_weights() {
    // y = 1 + 2*x に小さなノイズ。重みを変えても推定値は真の値の近くに返る。
    let y = vec![1.1, 2.9, 5.1, 6.9, 9.1];
    let x_columns = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0]];
    let weights = vec![1.0, 2.0, 0.5, 3.0, 1.5];

    let wls = WlsEstimator::fit(
        &y,
        &x_columns,
        vec!["x1".to_string()],
        true,
        "y".to_string(),
        &weights,
        CovType::Classical,
        0.95,
    )
    .unwrap();

    // 重み付き正規方程式の閉形式解（単回帰）: b1 = Sxy / Sxx（重み付き偏差）、b0 = ȳ_w - b1*x̄_w。
    let x = &x_columns[0];
    let sw: f64 = weights.iter().sum();
    let xbar = x.iter().zip(&weights).map(|(a, w)| a * w).sum::<f64>() / sw;
    let ybar = y.iter().zip(&weights).map(|(a, w)| a * w).sum::<f64>() / sw;
    let sxy: f64 = (0..5)
        .map(|i| weights[i] * (x[i] - xbar) * (y[i] - ybar))
        .sum();
    let sxx: f64 = (0..5).map(|i| weights[i] * (x[i] - xbar).powi(2)).sum();
    let (want_b1, want_b0) = (sxy / sxx, ybar - sxy / sxx * xbar);

    let b0 = *wls.estimator().params().get(0, 0);
    let b1 = *wls.estimator().params().get(1, 0);
    assert!((b0 - want_b0).abs() < 1e-9);
    assert!((b1 - want_b1).abs() < 1e-9);
    assert!((b0 - 1.0).abs() < 0.3);
    assert!((b1 - 2.0).abs() < 0.15);
    for (i, r) in wls.residuals().iter().enumerate() {
        assert!((r - (y[i] - b0 - b1 * x_columns[0][i])).abs() < 1e-12);
    }
}

#[test]
fn fit_without_intercept_uses_uncentered_r_squared_and_omits_const() {
    // include_intercept=falseのとき、WLSがOLS側のuncentered TSS分岐
    // （has_intercept()=falseのときのr_squared計算）と、original_scale_fitted_and_residualsの
    // 「切片項を足さない」分岐を正しく通ることを確認する。
    //
    // y = 2*x1に対する厳密解（ノイズなし）は使わない: k=1（切片なし・x1のみ）で
    // 残差がすべて0になると分散が数値的にゼロになり、F検定（wald_f_test）の
    // Cholesky分解が`NearSingularMatrix`を返してしまう（本来は理論上到達不能な
    // 防御的分岐だが、ゼロ分散という退化ケースでは実際に到達しうる）。
    // このテストの目的はF検定の退化ケースの検証ではないため、小さなノイズを入れる。
    let y = vec![2.1, 3.9, 6.2, 7.8, 10.1]; // y ≈ 2*x1
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weights = vec![1.0, 2.0, 0.5, 3.0, 1.5];

    let wls = WlsEstimator::fit(
        &y,
        std::slice::from_ref(&x1),
        vec!["x1".to_string()],
        false,
        "y".to_string(),
        &weights,
        CovType::Classical,
        0.95,
    )
    .unwrap();

    assert_eq!(wls.estimator().input().param_names(), ["x1".to_string()]);
    assert!(!wls.estimator().input().has_intercept());
    let beta_hat = *wls.estimator().params().get(0, 0);
    assert!((beta_hat - 2.0).abs() < 0.05);
    // 切片なし（uncentered TSS）の場合、Σ(sqrt(w)*y)² == Σw*y² が数学的に厳密に
    // 成り立つため、wls.r_squared()（元スケールから計算）とwls.estimator().r_squared()
    // （変換後データから計算）は一致する。ここでは公開APIとして使うべき方
    // （wls.r_squared()）を確認する。
    assert!(wls.r_squared() > 0.99);
    assert!((wls.r_squared() - wls.estimator().r_squared()).abs() < 1e-12);

    // 内部整合性: residuals()は「元スケールのy - 推定された係数による予測値」であるはず
    // （original_scale_fitted_and_residualsの定義そのものの確認。真の係数2.0とは比較しない）。
    for (i, &r) in wls.residuals().iter().enumerate() {
        let expected = y[i] - beta_hat * x1[i];
        assert!(
            (r - expected).abs() < 1e-9,
            "residual {i}: got {r}, expected {expected}"
        );
    }
}

#[test]
fn fit_with_multiple_x_columns_recovers_known_coefficients() {
    // original_scale_fitted_and_residualsのx_columnsループ（複数列）が正しく計算できることを確認する
    // （これまでのテストはx列1本のみだった）。
    // 1 + 2*x1 + 3*x2 に小さなノイズ。
    let y = vec![9.1, 7.8, 19.15, 18.05, 28.9, 28.12];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x2 = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0];
    let weights = vec![1.0, 2.0, 0.5, 3.0, 1.5, 2.5];

    let wls = WlsEstimator::fit(
        &y,
        &[x1.clone(), x2.clone()],
        vec!["x1".to_string(), "x2".to_string()],
        true,
        "y".to_string(),
        &weights,
        CovType::Classical,
        0.95,
    )
    .unwrap();

    let params = wls.estimator().params();
    let (b0, b1, b2) = (*params.get(0, 0), *params.get(1, 0), *params.get(2, 0));
    assert!((b0 - 1.0).abs() < 0.5); // const
    assert!((b1 - 2.0).abs() < 0.5); // x1
    assert!((b2 - 3.0).abs() < 0.5); // x2
    for (i, r) in wls.residuals().iter().enumerate() {
        assert!((r - (y[i] - b0 - b1 * x1[i] - b2 * x2[i])).abs() < 1e-12);
    }
}

#[test]
fn fit_with_all_weights_one_matches_ols() {
    // wls-spec.md「sqrt(w)変換」の構造的保証: 重みが全て1のときWLSはOLSと完全一致する。
    let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
    let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let weights = vec![1.0; 5];

    let wls = WlsEstimator::fit(
        &y,
        &x_columns,
        vec!["x1".to_string()],
        true,
        "y".to_string(),
        &weights,
        CovType::Classical,
        0.95,
    )
    .unwrap();

    let ols_input = OlsInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();
    let ols = OlsEstimator::fit(ols_input, CovType::Classical, 0.95).unwrap();

    for j in 0..2 {
        assert_eq!(*wls.estimator().params().get(j, 0), *ols.params().get(j, 0));
        assert_eq!(
            *wls.estimator().std_errors().get(j, 0),
            *ols.std_errors().get(j, 0)
        );
        assert_eq!(
            *wls.estimator().conf_lower().get(j, 0),
            *ols.conf_lower().get(j, 0)
        );
    }
    assert_eq!(
        wls.estimator().f_statistic().unwrap(),
        ols.f_statistic().unwrap()
    );
    assert_eq!(
        wls.estimator().f_p_value().unwrap(),
        ols.f_p_value().unwrap()
    );

    // r_squared/adj_r_squared/log_likelihood/aic/bicはWlsEstimator側の（元スケールの
    // y・weightsから計算し直した）値を使う。重みが全て1なら重み付き平均=単純平均、
    // ヤコビアン補正項0.5*Σlog(1)=0となり、OLSと完全一致するはず
    // （WlsEstimator側での再計算が必要な理由はモジュール冒頭のdocコメント参照）。
    // weights=1でも、重み付き平均（Σw*y/Σw）と単純平均（Σy/n）は数学的には同じ値だが
    // 浮動小数点の加算順序が異なるため、完全な==ではなく丸め誤差レベルで比較する。
    assert!((wls.r_squared() - ols.r_squared()).abs() < 1e-12);
    assert!((wls.adj_r_squared() - ols.adj_r_squared()).abs() < 1e-12);
    assert!((wls.log_likelihood() - ols.log_likelihood()).abs() < 1e-9);
    assert!((wls.aic() - ols.aic()).abs() < 1e-9);
    assert!((wls.bic() - ols.bic()).abs() < 1e-9);

    // 残差の計算経路自体はwls/statistics.rs（手動ループ）とols/fit.rs（faerの行列演算）で異なるため、
    // 丸め誤差レベルでの一致を確認する（構造的な完全一致の保証対象はestimator()側）。
    for i in 0..5 {
        assert!((wls.residuals()[i] - *ols.residuals().get(i, 0)).abs() < 1e-12);
    }
}

/// x=[1..5], y=[2,4,5,4,5], weights=[1,4,0.25,9,2]（切片あり、classical）の
/// 適合度統計量。期待値はstatsmodels 0.14.6で独立に計算・検算済み
/// （`sm.WLS(y, sm.add_constant(x1), weights=w).fit(use_t=True)`）。
/// r_squared/log_likelihood（→aic/bic）の計算式修正（モジュール冒頭のdocコメント
/// 参照）を固定するためのエンジン単体テスト。
#[test]
fn fit_computes_r_squared_and_information_criteria_matching_statsmodels() {
    let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
    let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let weights = vec![1.0, 4.0, 0.25, 9.0, 2.0];

    let wls = WlsEstimator::fit(
        &y,
        &x_columns,
        vec!["x1".to_string()],
        true,
        "y".to_string(),
        &weights,
        CovType::Classical,
        0.95,
    )
    .unwrap();

    assert!((*wls.estimator().params().get(0, 0) - 2.783_764_87).abs() < 1e-6);
    assert!((*wls.estimator().params().get(1, 0) - 0.358_992_3).abs() < 1e-6);
    assert!((*wls.estimator().std_errors().get(0, 0) - 0.824_058_49).abs() < 1e-6);
    assert!((*wls.estimator().std_errors().get(1, 0) - 0.227_478_42).abs() < 1e-6);
    assert!((wls.r_squared() - 0.453_603_574_100_183_8).abs() < 1e-9);
    assert!((wls.adj_r_squared() - 0.271_471_432_133_578_5).abs() < 1e-9);
    assert!((wls.log_likelihood() - (-4.694_800_450_440_493)).abs() < 1e-9);
    assert!((wls.aic() - 13.389_600_900_880_986).abs() < 1e-9);
    assert!((wls.bic() - 12.608_476_725_749_187).abs() < 1e-9);
    assert!((wls.estimator().f_statistic().unwrap() - 2.490_519_076_986_167_6).abs() < 1e-6);
    assert!((wls.estimator().f_p_value().unwrap() - 0.212_642_917_457_664_06).abs() < 1e-6);
}

#[test]
fn fit_matches_manually_transformed_ols_for_all_cov_types() {
    // classical/HC0-3/HAC/clusterのいずれも、WlsEstimator::fitはcov_typeをそのまま
    // OlsEstimator::fitに渡すだけで正しく動作するはず（wls-spec.md「標準誤差」
    // の通り、新しい計算式の実装は不要という前提の確認）。
    //
    // 検証方法: `from_columns_weighted`を経由せず、y・x1・切片列をこのテスト内で手動で
    // sqrt(weight)倍した上で`from_columns`（include_intercept=false）に渡し、素の
    // OlsEstimator::fitを直接呼ぶ。これは`WlsEstimator::fit`の内部実装と独立した経路で
    // 同じ変換を行っているため、両者が一致すればcov_typeの受け渡し・適合度統計量の
    // 計算が正しく配線されていることの確認になる。
    let y: Vec<f64> = vec![2.0, 4.0, 5.0, 4.0, 6.0, 3.0];
    let x1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weights: Vec<f64> = vec![1.0, 4.0, 0.25, 9.0, 2.0, 0.5];
    let clusters = ["a", "a", "b", "b", "c", "c"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let sqrt_w: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
    let y_tilde: Vec<f64> = y.iter().zip(&sqrt_w).map(|(v, s)| v * s).collect();
    let const_tilde: Vec<f64> = sqrt_w.clone();
    let x1_tilde: Vec<f64> = x1.iter().zip(&sqrt_w).map(|(v, s)| v * s).collect();
    let manual_x_columns = vec![const_tilde, x1_tilde];

    let cov_types = [
        CovType::Classical,
        CovType::Hc0,
        CovType::Hc1,
        CovType::Hc2,
        CovType::Hc3,
        CovType::Hac {
            lags: Some(1),
            time_order: row_time_order(6),
        },
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&clusters)),
        },
    ];

    for cov_type in cov_types {
        let label = format!("{cov_type:?}");

        let manual_input = OlsInput::from_columns(
            &y_tilde,
            &manual_x_columns,
            vec!["const".to_string(), "x1".to_string()],
            false,
            "y".to_string(),
        )
        .unwrap();
        let manual = OlsEstimator::fit(manual_input, cov_type.clone(), 0.95).unwrap();

        let wls = WlsEstimator::fit(
            &y,
            std::slice::from_ref(&x1),
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            &weights,
            cov_type,
            0.95,
        )
        .unwrap();

        for j in 0..2 {
            assert!(
                (*wls.estimator().params().get(j, 0) - *manual.params().get(j, 0)).abs() < 1e-9,
                "{label}: params mismatch at index {j}"
            );
            assert!(
                (*wls.estimator().std_errors().get(j, 0) - *manual.std_errors().get(j, 0)).abs()
                    < 1e-9,
                "{label}: std_errors mismatch at index {j}"
            );
        }
        // r_squared・f_statistic・f_p_valueはここでは比較しない: `manual`側は
        // 切片列を自前で用意しているため`include_intercept=false`で渡しており、
        // `has_intercept`フラグ（centered/uncentered TSSの切替、F検定でconstを
        // 除外するかの切替に使う）が`wls`側（`include_intercept=true`）と異なる。
        // これは「同じ回帰に対する2通りの正しい計算」ではなく、意味的に異なる統計量に
        // なるため単純比較できない。R²・F検定が正しいことは
        // `fit_with_all_weights_one_matches_ols`（両者ともinclude_intercept=trueで
        // 揃っている）で別途確認する。
        //
        // aicはlog_likelihood（ssr, nのみに依存）とk（パラメータ数、どちらも2）のみで
        // 決まるため、has_intercept差の影響を受けず比較できる。ここでは意図的に
        // `wls.estimator().aic()`（変換後データに対するOLSのaic、ヤコビアン補正なし）を
        // 使う。`manual`側も同じ「変換後データに対するOLS」のaicのため、両者は一致する
        // はず（配線の確認が目的）。`wls.aic()`（ヤコビアン補正込みの正しい値）は
        // `manual`側に対応する計算がないためここでは比較しない
        // （`fit_with_all_weights_one_matches_ols`で別途確認）。
        assert!(
            (wls.estimator().aic() - manual.aic()).abs() < 1e-9,
            "{label}: aic mismatch"
        );
    }
}
