//! property-basedテスト。固定シナリオ
//! （`within_transform_two_way_matches_closed_form_double_demeaning`、`N=T=2`）とは別に、
//! `N != T`の非対称なバランスパネルでも「entity→time逐次quasi-demean」が閉形式の
//! 二重デミーニングと一致することをランダムデータで検証する。閉形式側は
//! `quasi_demean_column`/`within_transform_two_way`を一切経由せず素朴な二重ループで
//! 独立に計算する（`engine/src/iv/CLAUDE.md`「自己参照的なオラクルは実装と同じ間違いを
//! 複製する」と同じ教訓を踏まえ、実装の内部関数を再利用しない独立実装にしている）。

use crate::panel::fe::input::FeInput;
use crate::panel::fe::within::within_transform_two_way;
use proptest::collection;
use proptest::prelude::*;

/// entity外側・time内側の行順で、`n_entities × n_periods`の完全なバランスパネル
/// （`y`はランダムな連続一様分布）を生成するストラテジ。
fn balanced_panel_strategy() -> impl Strategy<Value = (usize, usize, Vec<f64>)> {
    (2..=5usize, 2..=5usize).prop_flat_map(|(n_entities, n_periods)| {
        (
            Just(n_entities),
            Just(n_periods),
            collection::vec(-100.0f64..100.0, n_entities * n_periods),
        )
    })
}

/// `balanced_panel_strategy`の`y`と同じ行順（entity外側・time内側）の
/// `entity`/`time`ラベル列を作る。
fn entity_time_labels(n_entities: usize, n_periods: usize) -> (Vec<String>, Vec<String>) {
    let mut entity = Vec::with_capacity(n_entities * n_periods);
    let mut time = Vec::with_capacity(n_entities * n_periods);
    for i in 0..n_entities {
        for t in 0..n_periods {
            entity.push(format!("e{i}"));
            time.push(format!("t{t}"));
        }
    }
    (entity, time)
}

/// 閉形式`ỹ_it = y_it - ȳ_i. - ȳ_.t + ȳ..`を素朴な二重ループで独立に計算する
/// オラクル（`y`は`entity_time_labels`と同じ行順、entity外側・time内側）。
fn closed_form_two_way_demean(y: &[f64], n_entities: usize, n_periods: usize) -> Vec<f64> {
    let n = y.len();
    let grand_mean: f64 = y.iter().sum::<f64>() / n as f64;

    let entity_means: Vec<f64> = (0..n_entities)
        .map(|i| {
            let sum: f64 = (0..n_periods).map(|t| y[i * n_periods + t]).sum();
            sum / n_periods as f64
        })
        .collect();
    let time_means: Vec<f64> = (0..n_periods)
        .map(|t| {
            let sum: f64 = (0..n_entities).map(|i| y[i * n_periods + t]).sum();
            sum / n_entities as f64
        })
        .collect();

    let mut out = Vec::with_capacity(n);
    for i in 0..n_entities {
        for t in 0..n_periods {
            out.push(y[i * n_periods + t] - entity_means[i] - time_means[t] + grand_mean);
        }
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn within_transform_two_way_matches_independent_closed_form_oracle(
        (n_entities, n_periods, y) in balanced_panel_strategy()
    ) {
        let (entity, time) = entity_time_labels(n_entities, n_periods);
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".to_string())
                .unwrap();

        let (y_out, _) = within_transform_two_way(&input).unwrap();
        let expected = closed_form_two_way_demean(&y, n_entities, n_periods);

        for (actual, expected) in y_out.iter().zip(expected.iter()) {
            let scale = y.iter().fold(1.0_f64, |acc, v| acc.max(v.abs()));
            prop_assert!(
                (actual - expected).abs() <= 1e-8 * scale,
                "actual={actual}, expected={expected}"
            );
        }
    }
}
