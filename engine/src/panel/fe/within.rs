//! FEのwithin変換と、変換の前後に行う事前検証（singleton・分散ゼロの説明変数）。
//!
//! テストは`within/tests.rs`と`within/tests/`にテーマ別で置く。

use super::input::{FeAbsorption, FeInput};
use crate::panel::common::{
    PanelError, WithinTransform, check_leverage_dimension, covers_every_combination,
    ensure_no_duplicate_entity_time_pairs, ensure_no_singleton_groups,
    ensure_no_zero_variance_columns,
};
use crate::shared::alternating_projections::{GroupSystem, ProjectionOptions};

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

/// 2-wayのwithin変換の結果。
pub(crate) struct TwoWayWithin {
    pub(crate) y: Vec<f64>,
    pub(crate) x: Vec<Vec<f64>>,
    /// 不均衡な2-wayで、各列から引いた個体の効果と時点の効果（`fixed_effects()`の復元に使う）。
    /// バランスパネルは閉形式から復元できるので`None`。
    pub(crate) absorbed: Option<AbsorbedEffects>,
    /// 変換後の値に残る誤差の、元の値の最大絶対値に対する相対的な大きさ。閉形式（バランスパネル）は0、
    /// 交互射影（不均衡）は打ち切りの許容度に比例する。分散ゼロ・完全適合の判定に足す。
    pub(crate) noise: f64,
}

/// 交互射影が各列から引いた個体の効果と時点の効果。`y[d]`・`x[j][d]`は、グループ`d`（0が個体、1が時点）の
/// 水準ごとの量。
#[derive(Debug, Clone)]
pub(crate) struct AbsorbedEffects {
    pub(crate) y: Vec<Vec<f64>>,
    pub(crate) x: Vec<Vec<Vec<f64>>>,
}

/// 交互射影が列に残す誤差の見積もり（`Projected::error_bound`）にかける安全率。見積もりは隣り合う掃引の
/// 変化量の比から作るため、実際の誤差より小さくなりうる。
const ABSORPTION_NOISE_SAFETY: f64 = 10.0;

/// 2-way FE（entity + time FE）のwithin変換。事前に`(entity, time)`の重複がないことを検証する。
///
/// - 全ての組合せがそろうバランスパネルは、閉形式の二重デミーニングと数学的に等価な
///   「entityでquasi-demean → その結果をtimeでquasi-demean」の2段階適用で計算する
///   （`fe.rs`のモジュールdoc参照）。
/// - 組合せが欠ける不均衡パネルは、2つのグループの水準平均を引く交互射影
///   （`shared::alternating_projections`）で計算する（`fe-spec.md`3.1節）。
///
/// `absorption`は呼び出し側が`FeAbsorption::validate`で検査済みであること（`fit_fe`が行う）。
/// 戻り値の列は元の列順を保持する。
///
/// `FeEstimator::fit`経由では`two_way_time`が`time`の無い入力を先に弾くため、ここの
/// `time`の存在チェックは単体利用時の防御（到達しない二重チェックだが意図的に残している）。
///
/// # Errors
/// - `input.time_keys()`が`None`の場合は`PanelError::TwoWayRequiresTime`
/// - `(entity, time)`の重複は`PanelError::DuplicateEntityTimePair`
/// - `leverage_needed`（HC2/HC3）で、組合せが欠け、水準の少ない方の効果の水準数が上限を超えるなら
///   `PanelError::LeverageDimensionTooLarge`
/// - 交互射影が上限回数までに収束しなければ`PanelError::AbsorptionNotConverged`
pub(crate) fn within_transform_two_way_with(
    input: &FeInput,
    absorption: &FeAbsorption,
    leverage_needed: bool,
) -> Result<TwoWayWithin, PanelError> {
    let time_codes = input.time_codes().ok_or(PanelError::TwoWayRequiresTime)?;
    let entity = input.entity_codes();
    ensure_no_duplicate_entity_time_pairs(entity, time_codes, input.effect_names())?;
    if leverage_needed {
        // 組合せが欠けるときの密な行列の大きさは入力だけで決まるので、重い射影の前に確かめる。
        check_leverage_dimension(entity, time_codes)?;
    }

    if covers_every_combination(entity, time_codes) {
        let within = WithinTransform::two_way(entity, time_codes);
        let y = within.apply(input.y());
        let x = input.x().iter().map(|col| within.apply(col)).collect();
        return Ok(TwoWayWithin {
            y,
            x,
            absorbed: None,
            noise: 0.0,
        });
    }

    let system = GroupSystem::new(&[entity, time_codes]);
    let options = ProjectionOptions {
        tol: absorption.tol,
        // 呼び出し側が`FeAbsorption::validate`で`1..=MAX_ITER_LIMIT`を確かめている。
        max_sweeps: usize::try_from(absorption.max_iter).unwrap_or(usize::MAX),
    };
    // 元の列の絶対値の最大に対する、変換後の値に残る誤差の見積もりの最大。
    let mut relative_error = 0.0_f64;
    let mut project = |col: &[f64]| -> Result<(Vec<f64>, Vec<Vec<f64>>), PanelError> {
        let projected = system
            .project_with_effects(col, &options)
            .map_err(|failure| PanelError::AbsorptionNotConverged {
                n_sweeps: failure.n_sweeps,
                relative_change: failure.relative_change,
            })?;
        let scale = col.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if scale > 0.0 {
            relative_error = relative_error.max(projected.error_bound / scale);
        }
        Ok((projected.values, projected.effects))
    };
    let (y, y_effects) = project(input.y())?;
    let mut x = Vec::with_capacity(input.x().len());
    let mut x_effects = Vec::with_capacity(input.x().len());
    for col in input.x() {
        let (values, effects) = project(col)?;
        x.push(values);
        x_effects.push(effects);
    }
    Ok(TwoWayWithin {
        y,
        x,
        absorbed: Some(AbsorbedEffects {
            y: y_effects,
            x: x_effects,
        }),
        noise: ABSORPTION_NOISE_SAFETY * relative_error,
    })
}

/// [`within_transform_two_way_with`]を既定の打ち切り設定で呼び、`(y, x)`を返す。
///
/// # Errors
/// [`within_transform_two_way_with`]と同じ。
#[cfg(test)]
pub(crate) fn within_transform_two_way(
    input: &FeInput,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), PanelError> {
    let TwoWayWithin { y, x, .. } =
        within_transform_two_way_with(input, &FeAbsorption::default(), false)?;
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
/// `noise`は変換が残す誤差の相対的な大きさ（[`TwoWayWithin::noise`]）。
/// 複数列が該当する場合は`x_names`の順で最初に見つかった列のみを報告する（OLSの特異性
/// 検出等、他のバリデーションも「最初の1件を報告」で統一している）。
pub(super) fn validate_no_zero_variance_regressors(
    input: &FeInput,
    x_transformed: &[Vec<f64>],
    noise: f64,
) -> Result<(), PanelError> {
    ensure_no_zero_variance_columns(input.x_names(), input.x(), x_transformed, noise)
}

#[cfg(test)]
mod tests;
