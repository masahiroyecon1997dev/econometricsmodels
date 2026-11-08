//! パネル入力の次元検証と、within変換の前後に行う事前検証（singleton・分散ゼロ・バランス・時間変化）。

use super::error::{PanelDimension, PanelError};
use super::index::EffectNames;
use super::transform::WithinTransform;
use crate::shared::error::CommonError;
use crate::shared::group_codes::GroupCodes;
use crate::shared::validation::{column_is_zero_variance, column_is_zero_variance_with_noise};

/// `FeInput::from_group_codes`/`ReInput::from_group_codes`の入力検証（FE/REで共通）。
/// 検査の順序は、各`x_columns`の行数（`y`との不一致は`CommonError::DimensionMismatch`）、
/// 効果の各次元の長さ（`PanelError::IdentifierDimensionMismatch`。`effects`の先頭から）、
/// 各次元がキーを持つか（`require_effect_keys`）。後半2つは[`validate_panel_index`]。
///
/// # パニックについて
/// `x_names.len() != x_columns.len()`の場合は`debug_assert!`でパニックする（呼び出し側
/// `engine_pybind`の実装バグでしか起こり得ない内部契約であり、実データに起因する
/// `ValidationError`とは性質が異なる。`OlsInput::from_columns`と同じ理由）。
pub(crate) fn validate_panel_input(
    y: &[f64],
    x_columns: &[Vec<f64>],
    x_names: &[String],
    groups: &[&GroupCodes],
) -> Result<(), PanelError> {
    debug_assert_eq!(
        x_columns.len(),
        x_names.len(),
        "x_columns and x_names must have the same length"
    );

    validate_column_lengths(y.len(), x_columns)?;
    validate_panel_index(y.len(), groups)
}

/// `columns`の各列の長さが被説明変数の長さ`y_rows`と一致することを確かめる
/// （不一致は`CommonError::DimensionMismatch`）。
pub(crate) fn validate_column_lengths(
    y_rows: usize,
    columns: &[Vec<f64>],
) -> Result<(), PanelError> {
    for col in columns {
        if col.len() != y_rows {
            return Err(CommonError::DimensionMismatch {
                y_rows,
                x_rows: col.len(),
            }
            .into());
        }
    }
    Ok(())
}

/// パネルの索引（効果の各次元`groups`、`effects`の順）が観測数`nobs`と整合することを確かめる。
/// 検査の順序は、各次元の長さ（`PanelError::IdentifierDimensionMismatch`。先頭から見て最初の不一致）、
/// 各次元がキーを持つか（`require_effect_keys`）。`groups`が空の場合は呼び出し側の実装誤り
/// （`CommonError::Internal`）。
pub(crate) fn validate_panel_index(nobs: usize, groups: &[&GroupCodes]) -> Result<(), PanelError> {
    if groups.is_empty() {
        return Err(CommonError::internal("effects must have at least one dimension").into());
    }
    for (index, group) in groups.iter().enumerate() {
        if group.nobs() != nobs {
            return Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::new(index),
                y_rows: nobs,
                other_rows: group.nobs(),
            });
        }
    }
    groups
        .iter()
        .try_for_each(|group| require_effect_keys(group))
}

/// 効果の次元のコードがキー（`keys()`）を持つことを確かめる。固定効果のキーに使うため、
/// キー無しのコード（`GroupCodes::from_*_without_keys`）は受け付けない。
fn require_effect_keys(group: &GroupCodes) -> Result<(), PanelError> {
    if !group.has_keys() {
        return Err(CommonError::internal(
            "effect codes must carry keys (build them with GroupCodes::from_labels or \
             GroupCodes::from_factorized)",
        )
        .into());
    }
    Ok(())
}

/// 効果の各次元`groups`（`effects`の順）に観測数1のグループがないことを検証する（`fe-spec.md`1章）。
/// 先頭の次元から順に検査し、複数の次元に該当するsingletonがあれば先の次元を報告する。
/// 列名は`names`から引く。エラーには、観測数1のグループを取り除くことを繰り返したときに落ちる行の総数
/// （`n_dropped_rows`）を載せる。
///
/// # Errors
/// 観測数1のグループが見つかった場合は`PanelError::SingletonGroup`（該当する`dimension`を含む）。
pub(crate) fn ensure_no_singleton_groups(
    names: &EffectNames,
    groups: &[&GroupCodes],
) -> Result<(), PanelError> {
    let found = groups.iter().enumerate().find_map(|(index, group)| {
        let dimension = PanelDimension::new(index);
        first_singleton_group(dimension, names.get(dimension), group)
    });
    match found {
        None => Ok(()),
        Some((dimension, column, group_id)) => Err(PanelError::SingletonGroup {
            dimension,
            column,
            group_id,
            n_dropped_rows: rows_dropped_by_repeated_removal(groups),
        }),
    }
}

/// within変換後の各列に分散ゼロの列がないことを検証する（`fe-spec.md`1章）。`names`・
/// `originals`・`transformed`は同じ列順・同じ列数で、列ごとに同じ長さを持つ（`engine`内の
/// 内部契約であり、ユーザー入力起因ではない。リリースビルドで`zip`が黙って短い方へ切り詰める
/// 事態を避けるため`debug_assert_eq!`ではなく`assert_eq!`で守る）。
///
/// # Errors
/// 分散ゼロの列が見つかった場合は`PanelError::ZeroVarianceAfterDemeaning`（該当列名を含む）。
/// 複数列が該当する場合は`names`の順で最初に見つかった列のみを報告する。`noise`は変換が残す誤差の相対的な大きさ
/// （閉形式の変換は0。`column_is_zero_variance_with_noise`参照）。
pub(crate) fn ensure_no_zero_variance_columns(
    names: &[String],
    originals: &[Vec<f64>],
    transformed: &[Vec<f64>],
    noise: f64,
) -> Result<(), PanelError> {
    assert_eq!(
        names.len(),
        originals.len(),
        "names must have one entry per original column"
    );
    assert_eq!(
        originals.len(),
        transformed.len(),
        "transformed must have the same number of columns as the original columns"
    );
    for ((name, original), transformed) in names.iter().zip(originals).zip(transformed) {
        assert_eq!(
            original.len(),
            transformed.len(),
            "transformed column '{name}' must have the same length as the original column"
        );
        if column_is_zero_variance_with_noise(original, transformed, noise) {
            return Err(PanelError::ZeroVarianceAfterDemeaning {
                column: name.clone(),
            });
        }
    }
    Ok(())
}

/// `ids`（`entity`または`time`の列）で、観測順に最初に現れる観測数1のグループ
/// （次元・列名・グループのID）。複数あれば最初の1つだけを返す（`validate_no_zero_variance_regressors`の
/// 「最初の1件を報告」方針と統一）。観測数は`FeInput`の構築時に作ったコード
/// （`GroupCodes::counts`）をそのまま使う。
fn first_singleton_group(
    dimension: PanelDimension,
    column: &str,
    ids: &GroupCodes,
) -> Option<(PanelDimension, String, String)> {
    ids.codes()
        .iter()
        .find(|&&c| ids.counts()[c] == 1)
        .map(|&c| (dimension, column.to_string(), ids.keys()[c].clone()))
}

/// 観測数1のグループを取り除くことを、新しいsingletonが現れなくなるまで繰り返したときに落ちる行の総数。
///
/// 1つの次元だけなら観測数1のグループの数。複数の次元では、ある次元のsingletonを落とすと別の次元の
/// グループが観測数1になる連鎖を含む。各行を最大1回落とすので`O(n·K)`（`K`は次元の数）。
fn rows_dropped_by_repeated_removal(groups: &[&GroupCodes]) -> usize {
    match groups {
        [only] => only.counts().iter().filter(|&&c| c == 1).count(),
        _ => rows_dropped_across_dimensions(groups),
    }
}

/// [`rows_dropped_by_repeated_removal`]の、次元が複数あるときの連鎖の計算。
fn rows_dropped_across_dimensions(groups: &[&GroupCodes]) -> usize {
    let Some(first) = groups.first() else {
        return 0;
    };
    let rows_of: Vec<_> = groups.iter().map(|group| group.group_indices()).collect();
    let mut alive = vec![true; first.nobs()];
    // 次元ごとの、まだ落ちていない行の数。
    let mut left: Vec<Vec<usize>> = groups.iter().map(|group| group.counts().to_vec()).collect();
    // (次元, グループ)。
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for (dim, counts) in left.iter().enumerate() {
        for (g, &c) in counts.iter().enumerate() {
            if c == 1 {
                stack.push((dim, g));
            }
        }
    }
    let mut dropped = 0;
    while let Some((dim, g)) = stack.pop() {
        if left[dim][g] != 1 {
            continue;
        }
        let Some(&row) = rows_of[dim].rows(g).iter().find(|&&r| alive[r]) else {
            continue;
        };
        alive[row] = false;
        dropped += 1;
        for (other, group) in groups.iter().enumerate() {
            let code = group.codes()[row];
            left[other][code] -= 1;
            // 落とした行の次元自身は0になるので積まれない。
            if left[other][code] == 1 {
                stack.push((other, code));
            }
        }
    }
    dropped
}

/// 2-way REがバランスパネル（`entity` × `time`の全組合せが過不足なく1回ずつ存在する）
/// であることを検証する（`fe-spec.md`3.1節）。
///
/// 観測数カウントの一致（`n_obs == n_entities * n_periods`）だけでは不十分
/// （`PanelError::UnbalancedPanelForTwoWay`のdocコメント参照: あるペアの重複と別ペアの
/// 欠落が相殺してカウントだけ一致する入力がありうる）ため、判定は[`covers_every_combination`]
/// に任せる。
///
/// `entity.nobs() == time.nobs()`は`FeInput::from_group_codes`が既に保証している契約
/// （呼び出し側は常に同じ`FeInput`からこの2つを渡す）。
pub(crate) fn validate_balanced_panel(
    entity: &GroupCodes,
    time: &GroupCodes,
    names: &EffectNames,
) -> Result<(), PanelError> {
    if covers_every_combination(entity, time) {
        return Ok(());
    }
    Err(PanelError::UnbalancedPanelForTwoWay {
        n_obs: entity.nobs(),
        entity_column: names.get(PanelDimension::ENTITY).to_string(),
        n_entities: entity.n_groups(),
        time_column: names.get(PanelDimension::TIME).to_string(),
        n_periods: time.n_groups(),
    })
}

/// `entity` × `time`の全組合せが過不足なく1回ずつ現れるか（バランスパネルか）。閉形式の2段階変換と
/// バランス用のレバレッジ（`1/T_i + 1/N_t − 1/n`）を使ってよい条件。
///
/// 同じ組合せが複数の行に現れる入力も正当で（`effects`の列は州と郡のような組合せでもよい）、
/// その場合は組合せが欠けるので、バランスではなく不均衡として交互射影で扱う。行数が
/// `N × T`でなければその時点で不均衡。一致するときだけ、グリッドのセル（`entity`コード×
/// `n_periods`+`time`コード、`n`個の`bool`）を一度ずつ埋めて、重複があれば不均衡とする。
///
/// `entity.nobs() == time.nobs()`は`FeInput::from_group_codes`が保証している契約。
pub(crate) fn covers_every_combination(entity: &GroupCodes, time: &GroupCodes) -> bool {
    debug_assert_eq!(
        entity.nobs(),
        time.nobs(),
        "entity and time must have the same length (FeInput contract)"
    );
    let n_periods = time.n_groups();
    if entity.nobs() != entity.n_groups().saturating_mul(n_periods) {
        return false;
    }
    let mut filled = vec![false; entity.nobs()];
    entity
        .codes()
        .iter()
        .zip(time.codes())
        .all(|(&e, &t)| !std::mem::replace(&mut filled[e * n_periods + t], true))
}

/// `x`の各列のうち、1-way within変換（`entity`でθ=1のquasi-demean）後に分散がある列
/// （時間変化のある列）の添字を昇順で返す。分散ゼロの判定は
/// `validate_no_zero_variance_regressors`と同じ`column_is_zero_variance`（相対閾値）を使うため、
/// ここで時間変化ありと判定された列は、そのままFEに渡しても分散ゼロで拒否されない。
///
/// REが時間不変の列を許容するために使う（`re-spec.md`5章）。FEの`fit()`は分散ゼロの列を
/// 拒否する設計のままで、この関数は拒否の代わりに「FEに渡す列を選ぶ」ために同じ判定を使う。
pub(crate) fn time_varying_regressor_indices(x: &[Vec<f64>], entity: &GroupCodes) -> Vec<usize> {
    let within = WithinTransform::one_way(entity);
    x.iter()
        .enumerate()
        .filter(|(_, original)| !column_is_zero_variance(original, &within.apply(original)))
        .map(|(j, _)| j)
        .collect()
}

/// 2-way within変換後に分散がある列の添字を昇順で返す（`time_varying_regressor_indices`の
/// 2-way版）。2-wayのwithin変換は「entityでquasi-demean→timeでquasi-demean」の2段階
/// （`within_transform_two_way`）で、バランスパネルで閉形式の二重デミーニングと一致する。
/// 時点間でのみ（個体間でのみ）変わる列は2-way within変換後に全ゼロ列になり、時間不変の列と
/// 同じく「分散ゼロ」と判定される。
///
/// **バランスパネルであることの検証はしない**（呼び出し側が`validate_balanced_panel`で
/// 済ませている前提。不均衡だと2段階適用は二重デミーニングと一致しない）。
pub(crate) fn two_way_time_varying_regressor_indices(
    x: &[Vec<f64>],
    entity: &GroupCodes,
    time: &GroupCodes,
) -> Vec<usize> {
    let within = WithinTransform::two_way(entity, time);
    x.iter()
        .enumerate()
        .filter(|(_, original)| !column_is_zero_variance(original, &within.apply(original)))
        .map(|(j, _)| j)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::super::time_keys::TimeKeys;
    use super::*;

    #[test]
    fn validate_panel_input_reports_the_first_failing_check_in_order() {
        let y = [1.0, 2.0, 3.0];
        let names = labels(&["x"]);
        let entity = GroupCodes::from_labels(&labels(&["a", "a", "b"]));
        let time = TimeKeys::lexicographic(labels(&["1", "2", "1"]));
        assert_eq!(
            validate_panel_input(&y, &[vec![1.0, 2.0, 3.0]], &names, &[&entity, time.codes()]),
            Ok(())
        );

        // x列 → 効果の各次元の長さ（先頭から）→ 各次元のキー、の順。すべて同時に崩れていれば
        // x列の不一致が先に返る。
        let short_entity = GroupCodes::from_labels(&labels(&["a", "b"]));
        let short_time = TimeKeys::lexicographic(labels(&["1"]));
        let keyless = GroupCodes::from_labels_without_keys(&labels(&["a", "a"]));
        assert_eq!(
            validate_panel_input(&y, &[vec![1.0]], &names, &[&keyless, short_time.codes()]),
            Err(CommonError::DimensionMismatch {
                y_rows: 3,
                x_rows: 1
            }
            .into())
        );
        assert_eq!(
            validate_panel_input(
                &y,
                &[vec![1.0; 3]],
                &names,
                &[&short_entity, short_time.codes()]
            ),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::ENTITY,
                y_rows: 3,
                other_rows: 2
            })
        );
        assert_eq!(
            validate_panel_input(&y, &[vec![1.0; 3]], &names, &[&entity, short_time.codes()]),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::TIME,
                y_rows: 3,
                other_rows: 1
            })
        );
        // 3つ目以降の次元も長さを検査し、次元の番号で報告する。
        assert_eq!(
            validate_panel_input(
                &y,
                &[vec![1.0; 3]],
                &names,
                &[&entity, time.codes(), &short_entity]
            ),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::new(2),
                y_rows: 3,
                other_rows: 2
            })
        );
        // キー無しのentityでtimeの長さも合っていなければ、キー検査は最後なのでtimeの不一致が先。
        let keyless_full = GroupCodes::from_labels_without_keys(&labels(&["a", "a", "b"]));
        assert_eq!(
            validate_panel_input(
                &y,
                &[vec![1.0; 3]],
                &names,
                &[&keyless_full, short_time.codes()]
            ),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::TIME,
                y_rows: 3,
                other_rows: 1
            })
        );
        assert!(matches!(
            validate_panel_input(&y, &[vec![1.0; 3]], &names, &[&keyless_full]),
            Err(PanelError::Common(CommonError::Internal { .. }))
        ));
        // どの次元のキー無しも拒否する。
        assert!(matches!(
            validate_panel_input(&y, &[vec![1.0; 3]], &names, &[&entity, &keyless_full]),
            Err(PanelError::Common(CommonError::Internal { .. }))
        ));
        // 次元が1つも無い入力は呼び出し側の実装誤り。
        assert!(matches!(
            validate_panel_input(&y, &[vec![1.0; 3]], &names, &[]),
            Err(PanelError::Common(CommonError::Internal { .. }))
        ));
    }

    #[test]
    #[should_panic(expected = "names must have one entry per original column")]
    fn ensure_no_zero_variance_columns_panics_when_names_are_fewer_than_columns() {
        let originals = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let transformed = originals.clone();
        let _ = ensure_no_zero_variance_columns(&["x1".to_string()], &originals, &transformed, 0.0);
    }
}
