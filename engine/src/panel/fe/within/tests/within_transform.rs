//! within変換のテスト。

use crate::panel::common::PanelError;
use crate::panel::fe::input::FeInput;
use crate::panel::fe::test_support::*;
use crate::panel::fe::within::within_transform_one_way;
use crate::panel::fe::within::within_transform_two_way;

/// N=2（a, b）× T=2（"1", "2"）のバランスパネル。`fe.rs`のモジュールdocの導出で使った例と
/// 同じ数値（ȳ..=4.5, ȳ_a.=2, ȳ_b.=7, ȳ_.1=3, ȳ_.2=6 →
/// 閉形式`ỹ_it = y_it - ȳ_i. - ȳ_.t + ȳ..`で[0.5, -0.5, -0.5, 0.5]）。
fn balanced_two_way_input(y: [f64; 4], x_columns: &[Vec<f64>]) -> FeInput {
    let entity = strings(&["a", "a", "b", "b"]);
    let time = strings(&["1", "2", "1", "2"]);
    FeInput::from_columns(
        &y,
        x_columns,
        x_columns
            .iter()
            .enumerate()
            .map(|(i, _)| format!("x{i}"))
            .collect(),
        &entity,
        Some(&time),
        "y".into(),
    )
    .unwrap()
}

#[test]
fn within_transform_one_way_demeans_y_and_all_x_columns_by_entity() {
    // a: mean(y)=15, mean(x1)=150 / b: mean(y)=6, mean(x1)=60（`quasi_demean_column`
    // の`quasi_demean_column_with_theta_one_is_the_within_transformation`と同じ数値）。
    let entity = strings(&["a", "a", "b", "b", "b"]);
    let y = [10.0, 20.0, 3.0, 6.0, 9.0];
    let x1 = vec![100.0, 200.0, 30.0, 60.0, 90.0];
    let input = FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();

    let (y_out, x_out) = within_transform_one_way(&input);

    assert_eq!(y_out, vec![-5.0, 5.0, -3.0, 0.0, 3.0]);
    assert_eq!(x_out, vec![vec![-50.0, 50.0, -30.0, 0.0, 30.0]]);
}

#[test]
fn within_transform_one_way_supports_unbalanced_panel() {
    // T_a=1, T_b=3の不均衡パネル。エンティティ平均を引くだけで正確に成立する
    // （`quasi_demean_column_handles_unbalanced_panel`と同じ数値）。
    let entity = strings(&["a", "b", "b", "b"]);
    let y = [4.0, 2.0, 4.0, 6.0];
    let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

    let (y_out, x_out) = within_transform_one_way(&input);

    assert_eq!(y_out, vec![0.0, -2.0, 0.0, 2.0]);
    assert!(x_out.is_empty());
}

#[test]
fn within_transform_two_way_matches_closed_form_double_demeaning() {
    let y = [1.0, 3.0, 5.0, 9.0];
    let x1 = vec![2.0, 6.0, 10.0, 18.0]; // yのちょうど2倍（線形性の確認を兼ねる）
    let input = balanced_two_way_input(y, &[x1]);

    let (y_out, x_out) = within_transform_two_way(&input).unwrap();

    let expected_y = [0.5, -0.5, -0.5, 0.5];
    for (actual, expected) in y_out.iter().zip(expected_y.iter()) {
        assert!((actual - expected).abs() < 1e-12, "y_out = {y_out:?}");
    }
    let expected_x1: Vec<f64> = expected_y.iter().map(|v| v * 2.0).collect();
    for (actual, expected) in x_out[0].iter().zip(expected_x1.iter()) {
        assert!((actual - expected).abs() < 1e-12, "x_out = {x_out:?}");
    }
}

#[test]
fn within_transform_two_way_requires_time() {
    let y = [1.0, 2.0];
    let entity = strings(&["a", "b"]);
    let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

    let result = within_transform_two_way(&input);

    assert_eq!(result.unwrap_err(), PanelError::TwoWayRequiresTime);
}

#[test]
fn within_transform_two_way_removes_every_entity_mean_and_time_mean_of_an_unbalanced_panel() {
    // (c,1)・(b,3)が欠けた不均衡パネル。変換後の列は個体ごとにも時点ごとにも平均が0になる。
    let y = [1.0, 2.5, 4.0, 3.0, 5.5, 2.0, 7.0];
    let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
    let time = strings(&["1", "2", "3", "1", "2", "2", "3"]);
    let input = FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

    let (y_out, _) = within_transform_two_way(&input).unwrap();

    for group in [&entity, &time] {
        for label in ["a", "b", "c", "1", "2", "3"] {
            let rows: Vec<f64> = group
                .iter()
                .zip(&y_out)
                .filter(|(g, _)| g.as_str() == label)
                .map(|(_, v)| *v)
                .collect();
            if !rows.is_empty() {
                assert!(rows.iter().sum::<f64>().abs() < 1e-9, "{label}");
            }
        }
    }
}

#[test]
fn within_transform_two_way_treats_a_repeated_pair_that_offsets_a_missing_combination_as_unbalanced()
 {
    // entity=[a,a,b,b], time=[1,1,2,2]: (a,1)が重複、(a,2)と(b,1)が欠落。
    // n_obs=4はn_entities(2)*n_periods(2)=4と一致してしまうが、全組合せはそろっていない。
    // 閉形式の2段階変換を使うと誤るので、交互射影で計算する。連結成分は{a,1}と{b,2}の2つで、
    // 各成分の2行は個体平均を引いた値（∓0.5）になる。
    let y = [1.0, 2.0, 3.0, 4.0];
    let entity = strings(&["a", "a", "b", "b"]);
    let time = strings(&["1", "1", "2", "2"]);
    let input = FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

    let (y_within, _) = within_transform_two_way(&input).unwrap();

    for (actual, expected) in y_within.iter().zip([-0.5, 0.5, -0.5, 0.5]) {
        assert!((actual - expected).abs() < 1e-9, "{y_within:?}");
    }
}
