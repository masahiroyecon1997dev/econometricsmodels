//! 複数のファイルのテストが共有するフィクスチャ。

use crate::iv::common::IvInput;

/// 丁度識別の小さなデータ（`two_sls/test_support.rs`の`just_identified_data`と同じ）。第一段階も構造式も
/// 完全フィットにならない。`weight_type`によらず丁度識別ではGMMの点推定は2SLSと一致するはず
/// （`docs/spec/iv-spec.md`1.2節）。
pub(super) fn just_identified_input() -> IvInput {
    let z = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_endog = vec![1.2, 1.9, 3.3, 3.8, 5.1];
    let noise = [0.1, -0.2, 0.15, -0.05, 0.1];
    let y: Vec<f64> = x_endog
        .iter()
        .zip(noise)
        .map(|(&x, u)| 1.0 + 2.0 * x + u)
        .collect();
    IvInput::from_columns(
        &y,
        &[],
        vec![],
        &[x_endog],
        vec!["x_endog".to_string()],
        &[z],
        vec!["z".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap()
}

/// 過剰識別・不均一分散な合成データ（16観測、`x_endog`が`z1`/`z2`と相関しつつ、
/// 誤差項の分散が`z1`の大きさに依存する）。`WeightType::Robust`の数値検証に使う。
pub(super) fn heteroskedastic_test_columns() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let z1: Vec<f64> = (1..=16).map(|i| i as f64).collect();
    let z2: Vec<f64> = z1.iter().map(|&v| (v * 1.7).sin() * 5.0 + v).collect();
    // x_endog = z1 + 0.5*z2 + 不均一分散な内生的ノイズ（z1が大きいほど分散が大きい）。
    let noise = [
        -3.0, 2.5, -1.0, 4.0, -5.0, 6.5, -2.0, 7.0, -8.0, 3.0, -9.5, 10.0, -4.0, 11.0, -12.5, 6.0,
    ];
    let x_endog: Vec<f64> = (0..16)
        .map(|i| z1[i] + 0.5 * z2[i] + noise[i] * (1.0 + 0.3 * z1[i]))
        .collect();
    let y: Vec<f64> = (0..16)
        .map(|i| 2.0 + 1.5 * x_endog[i] + noise[i] * (1.0 + 0.3 * z1[i]) * 0.6)
        .collect();
    (y, x_endog, z1, z2)
}

/// `cov_type=Cluster`のSEを、小標本補正`(G/(G-1))((n-1)/(n-k))`込みの手計算
/// サンドイッチ公式と数値照合する（`fit_computes_cluster_weighted_estimate_matching_
/// manual_formula`と同じグループ構成、4グループ）。
pub(super) fn labelled_groups(n: usize, modulus: usize, prefix: &str) -> Vec<String> {
    (0..n).map(|i| format!("{prefix}{}", i % modulus)).collect()
}
