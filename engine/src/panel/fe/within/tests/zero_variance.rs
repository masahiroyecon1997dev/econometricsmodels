//! 変動のない説明変数の検出のテスト。

use crate::panel::common::PanelError;
use crate::panel::common::time_varying_regressor_indices;
use crate::panel::fe::input::FeInput;
use crate::panel::fe::test_support::*;
use crate::panel::fe::within::validate_no_zero_variance_regressors;
use crate::panel::fe::within::within_transform_one_way;
use crate::panel::fe::within::within_transform_two_way;

#[test]
fn validate_no_zero_variance_regressors_detects_time_invariant_variable_in_one_way_fe() {
    // "female"は各エンティティ内で一定（時間不変）のため、1-way within変換後は
    // 浮動小数点誤差の範囲でゼロになる（`fe-spec.md`1章のユースケースそのもの）。
    let entity = strings(&["a", "a", "b", "b"]);
    let y = [1.0, 2.0, 3.0, 5.0];
    let x_varying = vec![10.0, 20.0, 5.0, 15.0];
    let female = vec![0.0, 0.0, 1.0, 1.0];
    let input = FeInput::from_columns(
        &y,
        &[x_varying, female],
        vec!["x_varying".to_string(), "female".to_string()],
        &entity,
        None,
        "y".into(),
    )
    .unwrap();

    let (_, x_out) = within_transform_one_way(&input);
    let result = validate_no_zero_variance_regressors(&input, &x_out);

    assert_eq!(
        result.unwrap_err(),
        PanelError::ZeroVarianceAfterDemeaning {
            column: "female".to_string(),
        }
    );
}

#[test]
fn time_varying_regressor_indices_returns_only_columns_with_within_variation() {
    // 先頭・末尾が時間不変（エンティティ内で一定）、中央が時間変化。列順を保ち、
    // 時間変化のある列の添字だけを昇順で返す。
    let entity = strings(&["a", "a", "b", "b", "b"]);
    let input = FeInput::from_columns(
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        &[
            vec![7.0, 7.0, -2.0, -2.0, -2.0],
            vec![1.0, 3.0, 2.0, 5.0, 9.0],
            vec![0.5, 0.5, 0.5, 0.5, 0.5],
        ],
        strings(&["z1", "x", "z2"]),
        &entity,
        None,
        "y".into(),
    )
    .unwrap();

    assert_eq!(
        time_varying_regressor_indices(input.x(), input.entity_codes()),
        vec![1]
    );
}

#[test]
fn time_varying_regressor_indices_agrees_with_zero_variance_validation() {
    // 返した添字の列だけを集めた入力は`validate_no_zero_variance_regressors`を通り、
    // 返さなかった列だけを集めた入力は同じ検証で拒否される（FEに渡せる列の選別が
    // FE自身の判定とずれない）。
    let entity = strings(&["a", "a", "b", "b", "b", "c", "c"]);
    let columns = vec![
        vec![2.0, 2.0, 5.0, 5.0, 5.0, -1.0, -1.0],
        vec![1.0, 4.0, 2.0, 2.5, 9.0, 3.0, 8.0],
        vec![1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6],
    ];
    let names = strings(&["z", "x", "big"]);
    let y = [1.0; 7];
    let full =
        FeInput::from_columns(&y, &columns, names.clone(), &entity, None, "y".into()).unwrap();
    let keep = time_varying_regressor_indices(full.x(), full.entity_codes());
    assert_eq!(keep, vec![1]);

    let kept_input = FeInput::from_columns(
        &y,
        &[columns[1].clone()],
        vec![names[1].clone()],
        &entity,
        None,
        "y".into(),
    )
    .unwrap();
    let (_, x_kept) = within_transform_one_way(&kept_input);
    assert_eq!(
        validate_no_zero_variance_regressors(&kept_input, &x_kept),
        Ok(())
    );

    let dropped_input = FeInput::from_columns(
        &y,
        &[columns[0].clone(), columns[2].clone()],
        vec![names[0].clone(), names[2].clone()],
        &entity,
        None,
        "y".into(),
    )
    .unwrap();
    let (_, x_dropped) = within_transform_one_way(&dropped_input);
    assert!(validate_no_zero_variance_regressors(&dropped_input, &x_dropped).is_err());
}

#[test]
fn time_varying_regressor_indices_is_empty_for_no_regressors_and_all_for_varying_columns() {
    let entity = strings(&["a", "a", "b", "b"]);
    let none = FeInput::from_columns(&[1.0; 4], &[], vec![], &entity, None, "y".into()).unwrap();
    assert!(time_varying_regressor_indices(none.x(), none.entity_codes()).is_empty());

    let all = FeInput::from_columns(
        &[1.0; 4],
        &[vec![1.0, 2.0, 3.0, 5.0], vec![9.0, 1.0, 2.0, 8.0]],
        strings(&["x1", "x2"]),
        &entity,
        None,
        "y".into(),
    )
    .unwrap();
    assert_eq!(
        time_varying_regressor_indices(all.x(), all.entity_codes()),
        vec![0, 1]
    );
}

#[test]
fn validate_no_zero_variance_regressors_accepts_all_varying_columns() {
    let entity = strings(&["a", "a", "b", "b", "b"]);
    let y = [10.0, 20.0, 3.0, 6.0, 9.0];
    let x1 = vec![100.0, 200.0, 30.0, 60.0, 90.0];
    let input = FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();

    let (_, x_out) = within_transform_one_way(&input);

    assert_eq!(validate_no_zero_variance_regressors(&input, &x_out), Ok(()));
}

#[test]
fn validate_no_zero_variance_regressors_detects_entity_invariant_variable_in_two_way_fe() {
    // "year_dummy"はエンティティ間で変動しない（time FEと完全共線）ため、2-way
    // within変換後はゼロ分散になる（`fe-spec.md`1章「time FEと完全共線な列も同じチェックで
    // 検出できる」の具体例）。
    let entity = strings(&["a", "a", "b", "b"]);
    let time = strings(&["1", "2", "1", "2"]);
    let y = [1.0, 3.0, 5.0, 9.0];
    let x_varying = vec![2.0, 6.0, 10.0, 18.0];
    let year_dummy = vec![0.0, 1.0, 0.0, 1.0];
    let input = FeInput::from_columns(
        &y,
        &[x_varying, year_dummy],
        vec!["x_varying".to_string(), "year_dummy".to_string()],
        &entity,
        Some(&time),
        "y".into(),
    )
    .unwrap();

    let (_, x_out) = within_transform_two_way(&input).unwrap();
    let result = validate_no_zero_variance_regressors(&input, &x_out);

    assert_eq!(
        result.unwrap_err(),
        PanelError::ZeroVarianceAfterDemeaning {
            column: "year_dummy".to_string(),
        }
    );
}

#[test]
fn validate_no_zero_variance_regressors_reports_first_offending_column_in_name_order() {
    // 2列とも時間不変。`x_names`の順で最初（"const_a"）のみを報告する
    // （関数docコメントの方針）。
    let entity = strings(&["a", "a", "b", "b"]);
    let y = [1.0, 2.0, 3.0, 4.0];
    let const_a = vec![1.0, 1.0, 2.0, 2.0];
    let const_b = vec![9.0, 9.0, 8.0, 8.0];
    let input = FeInput::from_columns(
        &y,
        &[const_a, const_b],
        vec!["const_a".to_string(), "const_b".to_string()],
        &entity,
        None,
        "y".into(),
    )
    .unwrap();

    let (_, x_out) = within_transform_one_way(&input);
    let result = validate_no_zero_variance_regressors(&input, &x_out);

    assert_eq!(
        result.unwrap_err(),
        PanelError::ZeroVarianceAfterDemeaning {
            column: "const_a".to_string(),
        }
    );
}

#[test]
fn validate_no_zero_variance_regressors_with_no_regressors_succeeds() {
    let y = [1.0, 2.0];
    let entity = strings(&["a", "b"]);
    let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

    assert_eq!(validate_no_zero_variance_regressors(&input, &[]), Ok(()));
}

#[test]
fn validate_no_zero_variance_regressors_with_zero_observations_and_a_regressor_succeeds() {
    // n=0（`column_is_zero_variance`のn=0分岐、`fe.rs`のモジュールdoc参照）は「回帰変数0列」
    // （上のテスト）とは別に、「観測数0だが回帰変数自体は1列存在する」ケースでも
    // 明示的に確認する（列は空`Vec`になるが、列は存在する点が上と異なる）。
    let input = FeInput::from_columns(
        &[],
        &[vec![]],
        vec!["x1".to_string()],
        &[],
        None,
        "y".into(),
    )
    .unwrap();

    assert_eq!(
        validate_no_zero_variance_regressors(&input, &[vec![]]),
        Ok(())
    );
}
