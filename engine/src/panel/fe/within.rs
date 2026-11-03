//! FEのwithin変換と、変換の前後に行う事前検証（singleton・分散ゼロの説明変数）。

use super::input::FeInput;
use crate::panel::common::{
    PanelError, WithinTransform, ensure_no_singleton_groups, ensure_no_zero_variance_columns,
    validate_balanced_panel,
};

/// 1-way FE（entityのみ）のwithin変換。`y`と各`x`列にentityでのquasi-demean（θ=1）を
/// 適用する。不均衡パネルも無条件でサポートする（`fe.rs`のモジュールdoc・`fe-spec.md`3.1節参照）。
///
/// 戻り値は`(y_transformed, x_transformed)`（元の列順を保持）。
pub(crate) fn within_transform_one_way(input: &FeInput) -> (Vec<f64>, Vec<Vec<f64>>) {
    let within = WithinTransform::one_way(input.entity_codes());
    let y = within.apply(input.y());
    let x = input.x().iter().map(|col| within.apply(col)).collect();
    (y, x)
}

/// 2-way FE（entity + time FE）のwithin変換。閉形式の二重デミーニングと数学的に等価な
/// 「entityでquasi-demean → その結果をtimeでquasi-demean」の2段階適用で計算する
/// （`fe.rs`のモジュールdoc参照）。事前にバランスパネルであることを検証する（`fe-spec.md`3.1節）。
///
/// 戻り値は`(y_transformed, x_transformed)`（元の列順を保持）。
///
/// `FeEstimator::fit`経由では`two_way_time`が`time`の無い入力を先に弾くため、ここの
/// `time`の存在チェックは単体利用時の防御（到達しない二重チェックだが意図的に残している）。
///
/// # Errors
/// - `input.time_keys()`が`None`の場合は`PanelError::TwoWayRequiresTime`
/// - バランスパネルでない場合は`PanelError::UnbalancedPanelForTwoWay`
pub(crate) fn within_transform_two_way(
    input: &FeInput,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), PanelError> {
    let time_codes = input.time_codes().ok_or(PanelError::TwoWayRequiresTime)?;
    let entity = input.entity_codes();
    validate_balanced_panel(entity, time_codes, input.effect_names())?;

    let within = WithinTransform::two_way(entity, time_codes);
    let y = within.apply(input.y());
    let x = input.x().iter().map(|col| within.apply(col)).collect();
    Ok((y, x))
}

/// 1-way FE向けのsingleton検出（`fe-spec.md`1章）。`entity`に観測数1のグループがあれば
/// `PanelError::SingletonGroup`を返す。
///
/// # Errors
/// entityに観測数1のグループが見つかった場合は`PanelError::SingletonGroup`
/// （`dimension: PanelDimension::Entity`）。
pub(super) fn validate_no_singleton_groups_one_way(input: &FeInput) -> Result<(), PanelError> {
    ensure_no_singleton_groups(input.effect_names(), input.entity_codes(), None)
}

/// 2-way FE向けのsingleton検出（`fe-spec.md`1章）。entity・time双方を対称に検出する
/// （`within_transform_two_way`と同じく`time`必須）。
///
/// **`time`の存在チェックを最初に行う**（`within_transform_two_way`と同じ順序に揃える。
/// `fit()`側が複数のバリデーションを組み合わせて呼ぶ際、同じ前提条件——2-way FEには
/// `time`が要る——のチェックタイミングが関数ごとにばらつかないようにするため）。
/// その後、entityを先にチェックしてからtimeをチェックする（両方に該当するsingletonが
/// あった場合はentity側を先に報告する。`FeInput`のフィールド順（entity→time）に合わせた
/// 恣意的な優先順位）。
///
/// `FeEstimator::fit`経由では`two_way_time`が`time`の無い入力を先に弾くため、ここの
/// `time`の存在チェックは単体利用時の防御（到達しない二重チェックだが意図的に残している）。
///
/// # Errors
/// - `input.time_keys()`が`None`の場合は`PanelError::TwoWayRequiresTime`
/// - entityまたはtimeに観測数1のグループが見つかった場合は`PanelError::SingletonGroup`
///   （該当する`dimension`を含む）
pub(super) fn validate_no_singleton_groups_two_way(input: &FeInput) -> Result<(), PanelError> {
    let time = input.time_codes().ok_or(PanelError::TwoWayRequiresTime)?;
    ensure_no_singleton_groups(input.effect_names(), input.entity_codes(), Some(time))
}

/// within変換後の説明変数の各列に分散ゼロの列がないことを検証する（`fe-spec.md`1章）。1-way/2-way
/// 共通ロジック（`within_transform_one_way`/`within_transform_two_way`のどちらの出力も
/// 引数に渡せる）。
///
/// 時間不変変数（1-way）だけでなく、2-wayでtime FEと完全共線な「エンティティ間で変動しない
/// 列」も同じチェックで検出できる（`fe-spec.md`1章）。`x_transformed`は呼び出し側が`within_transform_*`
/// の戻り値をそのまま渡す想定で、`input.x()`（変換前の生の列）と同じ列順・同じ列数・列ごとに
/// 同じ長さを持つことを前提とする（`engine`内の内部契約であり、ユーザー入力起因ではない。
/// `quasi_demean_column`の呼び出し元契約と同じ扱いで`assert_eq!`で守る。`debug_assert_eq!`
/// だとリリースビルドで無効化され、列数不一致時に`zip`が黙って短い方へ切り詰め検証漏れの
/// 列が発生しうるため不可）。
///
/// # Errors
/// 分散ゼロの列が見つかった場合は`PanelError::ZeroVarianceAfterDemeaning`（該当列名を含む）。
/// 複数列が該当する場合は`x_names`の順で最初に見つかった列のみを報告する（OLSの特異性
/// 検出等、他のバリデーションも「最初の1件を報告」で統一している）。
pub(super) fn validate_no_zero_variance_regressors(
    input: &FeInput,
    x_transformed: &[Vec<f64>],
) -> Result<(), PanelError> {
    ensure_no_zero_variance_columns(input.x_names(), input.x(), x_transformed)
}

#[cfg(test)]
mod tests {

    use super::super::test_support::*;
    use super::*;

    use crate::panel::common::EffectNames;

    use crate::panel::common::PanelDimension;

    use crate::panel::common::reject_singleton_group;
    use crate::panel::common::time_varying_regressor_indices;

    use crate::shared::group_codes::GroupCodes;

    fn codes(values: &[&str]) -> GroupCodes {
        GroupCodes::from_labels(&strings(values))
    }

    // ── within_transform_one_way ────────────────────────────────────────────

    #[test]
    fn within_transform_one_way_demeans_y_and_all_x_columns_by_entity() {
        // a: mean(y)=15, mean(x1)=150 / b: mean(y)=6, mean(x1)=60（`quasi_demean_column`
        // の`quasi_demean_column_with_theta_one_is_the_within_transformation`と同じ数値）。
        let entity = strings(&["a", "a", "b", "b", "b"]);
        let y = [10.0, 20.0, 3.0, 6.0, 9.0];
        let x1 = vec![100.0, 200.0, 30.0, 60.0, 90.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
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

    // ── within_transform_two_way ────────────────────────────────────────────

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
    fn within_transform_two_way_rejects_unbalanced_panel_with_missing_combination() {
        // entity=b, time="2"の観測が欠けている（n_obs=3 != n_entities*n_periods=4）。
        let y = [1.0, 2.0, 3.0];
        let entity = strings(&["a", "a", "b"]);
        let time = strings(&["1", "2", "1"]);
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let result = within_transform_two_way(&input);

        assert_eq!(
            result.unwrap_err(),
            PanelError::UnbalancedPanelForTwoWay {
                n_obs: 3,
                entity_column: "entity".to_string(),
                n_entities: 2,
                time_column: "time".to_string(),
                n_periods: 2,
            }
        );
    }

    #[test]
    fn within_transform_two_way_rejects_duplicate_pair_that_offsets_missing_combination() {
        // entity=[a,a,b,b], time=[1,1,2,2]: (a,1)が重複、(a,2)と(b,1)が欠落。
        // n_obs=4はn_entities(2)*n_periods(2)=4と一致してしまうが、ユニークな
        // (entity,time)ペアは{(a,1),(b,2)}の2個のみで4に満たないため、単純な
        // カウント一致チェックでは見逃す入力を正しく検出できることを確認する
        // （`PanelError::UnbalancedPanelForTwoWay`のdocコメント・`validate_balanced_panel`
        // 参照）。
        let y = [1.0, 2.0, 3.0, 4.0];
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["1", "1", "2", "2"]);
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let result = within_transform_two_way(&input);

        assert_eq!(
            result.unwrap_err(),
            PanelError::UnbalancedPanelForTwoWay {
                n_obs: 4,
                entity_column: "entity".to_string(),
                n_entities: 2,
                time_column: "time".to_string(),
                n_periods: 2,
            }
        );
    }

    // ── validate_no_singleton_groups_one_way / _two_way ─────────────────────

    #[test]
    fn validate_no_singleton_groups_one_way_detects_entity_singleton() {
        // entity "c"は観測数1（singleton）。
        let y = [1.0, 2.0, 3.0];
        let entity = strings(&["a", "a", "c"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let result = validate_no_singleton_groups_one_way(&input);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                column: "entity".to_string(),
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_singleton_groups_reports_first_singleton_in_observation_order() {
        // singletonが"b"と"a"の2つ。観測順では"b"が先、コード順（辞書順）では"a"が先。
        // 観測順で最初のもの（"b"）を報告する（整数コード化の前と同じ選び方）。
        let y = [1.0, 2.0, 3.0, 4.0];
        let entity = strings(&["b", "a", "c", "c"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        assert_eq!(
            validate_no_singleton_groups_one_way(&input).unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                column: "entity".to_string(),
                group_id: "b".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_singleton_groups_one_way_accepts_no_singleton() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let entity = strings(&["a", "a", "b", "b"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        assert_eq!(validate_no_singleton_groups_one_way(&input), Ok(()));
    }

    #[test]
    fn validate_no_singleton_groups_two_way_detects_entity_singleton() {
        // entity "c"は時点"1"のみの1観測（singleton）。time側は各時点2観測ずつで
        // singletonではない。
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let entity = strings(&["a", "a", "b", "b", "c"]);
        let time = strings(&["1", "2", "1", "2", "1"]);
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let result = validate_no_singleton_groups_two_way(&input);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                column: "entity".to_string(),
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_singleton_groups_two_way_detects_time_singleton() {
        // time "3"はentity "a"のみの1観測（singleton）。entity側はどちらも2観測ずつで
        // singletonではない。
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let entity = strings(&["a", "a", "a", "b", "b"]);
        let time = strings(&["1", "2", "3", "1", "2"]);
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let result = validate_no_singleton_groups_two_way(&input);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Time,
                column: "time".to_string(),
                group_id: "3".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_singleton_groups_two_way_reports_entity_before_time_when_both_present() {
        // entity "c"（1観測）とtime "3"（1観測、entity "c"自身の行）が両方singleton。
        // entityを先にチェックする方針（関数doc参照）により、entity側が報告される。
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let entity = strings(&["a", "a", "b", "b", "c"]);
        let time = strings(&["1", "2", "1", "2", "3"]);
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let result = validate_no_singleton_groups_two_way(&input);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                column: "entity".to_string(),
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_singleton_groups_two_way_requires_time() {
        // entity側はsingletonではない（"a"が2観測）ため、`time`未指定の
        // `PanelError::TwoWayRequiresTime`が先に検出されることを確認する。
        let y = [1.0, 2.0];
        let entity = strings(&["a", "a"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let result = validate_no_singleton_groups_two_way(&input);

        assert_eq!(result.unwrap_err(), PanelError::TwoWayRequiresTime);
    }

    #[test]
    fn validate_no_singleton_groups_two_way_reports_missing_time_even_with_entity_singleton() {
        // entity "c"はsingletonだが`time`も未指定。`time`の存在チェックを先に行う方針
        // （関数doc、`within_transform_two_way`と同じ順序）により`TwoWayRequiresTime`が
        // 優先される。
        let y = [1.0, 2.0, 3.0];
        let entity = strings(&["a", "a", "c"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let result = validate_no_singleton_groups_two_way(&input);

        assert_eq!(result.unwrap_err(), PanelError::TwoWayRequiresTime);
    }

    #[test]
    fn reject_singleton_group_with_no_observations_succeeds() {
        // n=0境界（`validate_no_zero_variance_regressors_with_zero_observations_and_a_
        // regressor_succeeds`と同様の境界値テストの慣習に合わせる）。空配列にはsingleton
        // となりうる要素自体が存在しないため`Ok(())`になる。
        assert_eq!(
            reject_singleton_group(
                PanelDimension::Entity,
                "entity",
                &GroupCodes::from_labels(&[])
            ),
            Ok(())
        );
    }

    #[test]
    fn validate_no_singleton_groups_two_way_accepts_no_singleton() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["1", "2", "1", "2"]);
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        assert_eq!(validate_no_singleton_groups_two_way(&input), Ok(()));
    }

    #[test]
    fn singleton_group_error_reports_the_column_names_set_by_with_effect_names() {
        // 既定の名前は"entity"・"time"。`engine_pybind`が渡す列名がエラーの`column`に載る。
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let entity = strings(&["a", "a", "b", "b", "c"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();
        assert_eq!(
            validate_no_singleton_groups_one_way(&input),
            Err(PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                column: "entity".to_string(),
                group_id: "c".to_string(),
            })
        );
        let input = input.with_effect_names("firm".to_string(), None);
        assert_eq!(
            validate_no_singleton_groups_one_way(&input),
            Err(PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                column: "firm".to_string(),
                group_id: "c".to_string(),
            })
        );

        // 時点側の単独グループ（時点"2"は1観測）は2つ目の名前で報告する。
        let y = [1.0, 2.0, 3.0, 4.0];
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["1", "2", "1", "1"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into())
            .unwrap()
            .with_effect_names("firm".to_string(), Some("year".to_string()));
        assert_eq!(
            validate_no_singleton_groups_two_way(&input),
            Err(PanelError::SingletonGroup {
                dimension: PanelDimension::Time,
                column: "year".to_string(),
                group_id: "2".to_string(),
            })
        );
    }

    // ── validate_no_zero_variance_regressors ────────────────────────────────

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
        let none =
            FeInput::from_columns(&[1.0; 4], &[], vec![], &entity, None, "y".into()).unwrap();
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
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
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

    #[test]
    fn validate_balanced_panel_accepts_shuffled_rows_when_entities_and_periods_differ() {
        // 3エンティティ×2時点（n_entities != n_periods）で、行を観測順に並べていない。
        let entity = codes(&["c", "a", "b", "a", "c", "b"]);
        let time = codes(&["2", "1", "2", "2", "1", "1"]);
        assert_eq!(
            validate_balanced_panel(&entity, &time, &EffectNames::default()),
            Ok(())
        );
    }

    #[test]
    fn validate_balanced_panel_rejects_duplicate_pair_offsetting_a_missing_one() {
        // 3×2。(a,1)が重複し(a,2)が欠落するが、n_obs=6=3*2で件数は一致してしまう。
        let entity = codes(&["a", "a", "b", "b", "c", "c"]);
        let time = codes(&["1", "1", "1", "2", "1", "2"]);
        assert_eq!(
            validate_balanced_panel(&entity, &time, &EffectNames::default()),
            Err(PanelError::UnbalancedPanelForTwoWay {
                n_obs: 6,
                entity_column: "entity".to_string(),
                n_entities: 3,
                time_column: "time".to_string(),
                n_periods: 2,
            })
        );
    }

    #[test]
    fn validate_balanced_panel_reports_the_column_names_set_by_effect_names() {
        let entity = codes(&["a", "a", "b"]);
        let time = codes(&["1", "2", "1"]);
        let names = EffectNames::new("firm".to_string(), Some("year".to_string()));
        let err = validate_balanced_panel(&entity, &time, &names).unwrap_err();
        assert_eq!(
            err,
            PanelError::UnbalancedPanelForTwoWay {
                n_obs: 3,
                entity_column: "firm".to_string(),
                n_entities: 2,
                time_column: "year".to_string(),
                n_periods: 2,
            }
        );
    }

    /// property-basedテスト。固定シナリオ
    /// （`within_transform_two_way_matches_closed_form_double_demeaning`、`N=T=2`）とは別に、
    /// `N != T`の非対称なバランスパネルでも「entity→time逐次quasi-demean」が閉形式の
    /// 二重デミーニングと一致することをランダムデータで検証する。閉形式側は
    /// `quasi_demean_column`/`within_transform_two_way`を一切経由せず素朴な二重ループで
    /// 独立に計算する（`engine/src/iv/CLAUDE.md`「自己参照的なオラクルは実装と同じ間違いを
    /// 複製する」と同じ教訓を踏まえ、実装の内部関数を再利用しない独立実装にしている）。
    mod proptests {
        use super::*;
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
    }
}
