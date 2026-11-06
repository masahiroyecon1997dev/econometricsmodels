//! FEのwithin変換と、変換の前後に行う事前検証（singleton・分散ゼロの説明変数）。
//!
//! テストは`within/tests.rs`と`within/tests/`にテーマ別で置く。

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
mod tests;
