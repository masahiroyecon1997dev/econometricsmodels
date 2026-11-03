//! パネル索引`PanelIndex`と、効果の次元ごとの列名`EffectNames`。

use super::error::{PanelDimension, PanelError};
use super::time_keys::TimeKeys;
use super::validation::validate_panel_index;
use crate::shared::group_codes::GroupCodes;

/// 効果の次元ごとの列名。エラーの`column`に載せるためだけに`FeInput`/`ReInput`が保持する
/// （`engine_pybind`が`effects`の各要素の列名を渡す）。計算には使わない。
///
/// 既定は`"entity"`・`"time"`で、`engine`単体（テスト等）の構築ではこのまま使う。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EffectNames {
    entity: String,
    time: String,
}

impl Default for EffectNames {
    fn default() -> Self {
        Self {
            entity: "entity".to_string(),
            time: "time".to_string(),
        }
    }
}

impl EffectNames {
    /// 1つ目の次元（個体）と、あれば2つ目の次元（時点）の列名を設定する。
    /// 2つ目が`None`なら既定の名前を残す（1-wayでは使われない）。
    pub(crate) fn new(entity: String, time: Option<String>) -> Self {
        let default = Self::default();
        Self {
            entity,
            time: time.unwrap_or(default.time),
        }
    }

    /// `dimension`に対応する列名。
    pub(crate) fn get(&self, dimension: PanelDimension) -> &str {
        match dimension {
            PanelDimension::Entity => &self.entity,
            PanelDimension::Time => &self.time,
        }
    }
}

/// パネルの索引（個体・時点）と効果の列名。観測の行ごとにどの個体・どの時点に属するかを
/// 持つ入れ物で、列データ（`y`・`x`等）は持たない。`FeIvInput`が保持し、役割の異なる複数の
/// 列を同じ索引で扱えるようにする。
///
/// `entity`はキー付きの`GroupCodes`（固定効果のキーに使う）、`time`は2-wayの時間次元
/// （`TimeKeys`、時間順のコード）。構築時に観測数との整合を検証するので、構築後は
/// 長さの不一致が起きない。フィールドはprivate。
#[derive(Debug, Clone)]
pub struct PanelIndex {
    entity: GroupCodes,
    time: Option<TimeKeys>,
    effect_names: EffectNames,
}

impl PanelIndex {
    /// 観測数`nobs`の索引を組み立てる。効果の列名は既定（`"entity"`・`"time"`）で、
    /// [`with_effect_names`](Self::with_effect_names)で設定する。
    ///
    /// # Errors
    /// 検査の順序は`validate_panel_index`が正本。
    /// - `entity`の長さが`nobs`と一致しない場合は
    ///   `PanelError::IdentifierDimensionMismatch { dimension: PanelDimension::Entity, .. }`
    /// - `time`が`Some`で、その長さが`nobs`と一致しない場合は
    ///   `PanelError::IdentifierDimensionMismatch { dimension: PanelDimension::Time, .. }`
    /// - `entity`がキー無しのコードの場合は`PanelError::Common`（`CommonError::Internal`、
    ///   呼び出し側の実装誤り）
    pub fn new(
        nobs: usize,
        entity: GroupCodes,
        time: Option<TimeKeys>,
    ) -> Result<Self, PanelError> {
        validate_panel_index(nobs, &entity, time.as_ref())?;
        Ok(Self {
            entity,
            time,
            effect_names: EffectNames::default(),
        })
    }

    /// 効果の次元の列名を設定する。エラーの`column`に載るだけで、計算には影響しない。
    /// 設定済みの名前は全て置き換わり、`time`が`None`なら2つ目の名前は既定（`"time"`）になる。
    pub fn with_effect_names(mut self, entity: String, time: Option<String>) -> Self {
        self.effect_names = EffectNames::new(entity, time);
        self
    }

    /// `dimension`に対応する効果の列名。
    pub fn effect_name(&self, dimension: PanelDimension) -> &str {
        self.effect_names.get(dimension)
    }

    /// `entity`の整数コード（キー付き）。
    pub fn entity_codes(&self) -> &GroupCodes {
        &self.entity
    }

    /// ユニークなエンティティ数。
    pub fn n_entities(&self) -> usize {
        self.entity.n_groups()
    }

    /// ユニークなエンティティのラベル（コード順＝辞書順）。
    pub fn entity_labels(&self) -> &[String] {
        self.entity.keys()
    }

    /// 時点のラベルと時間順のコード。1-way FEで`time`が無ければ`None`。
    pub fn time_keys(&self) -> Option<&TimeKeys> {
        self.time.as_ref()
    }

    /// 観測数。
    pub fn nobs(&self) -> usize {
        self.entity.nobs()
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;
    use crate::shared::error::CommonError;

    #[test]
    fn panel_index_new_exposes_codes_labels_and_time_keys() {
        let entity = GroupCodes::from_labels(&labels(&["b", "b", "a"]));
        let time = TimeKeys::lexicographic(labels(&["1", "2", "1"]));
        let index = PanelIndex::new(3, entity, Some(time.clone())).unwrap();

        assert_eq!(index.nobs(), 3);
        assert_eq!(index.n_entities(), 2);
        assert_eq!(index.entity_labels(), labels(&["a", "b"]).as_slice());
        assert_eq!(index.entity_codes().n_groups(), 2);
        assert_eq!(index.time_keys(), Some(&time));
        assert!(
            PanelIndex::new(3, GroupCodes::from_labels(&labels(&["a", "a", "b"])), None)
                .unwrap()
                .time_keys()
                .is_none()
        );
    }

    #[test]
    fn panel_index_new_reports_length_mismatches_then_missing_keys() {
        let entity = GroupCodes::from_labels(&labels(&["a", "a", "b"]));
        let short_time = TimeKeys::lexicographic(labels(&["1"]));
        let keyless = GroupCodes::from_labels_without_keys(&labels(&["a", "a", "b"]));

        assert!(matches!(
            PanelIndex::new(4, entity.clone(), None),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Entity,
                y_rows: 4,
                other_rows: 3
            })
        ));
        assert!(matches!(
            PanelIndex::new(3, entity, Some(short_time.clone())),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                y_rows: 3,
                other_rows: 1
            })
        ));
        // キーの検査は最後なので、timeの不一致が先に返る。
        assert!(matches!(
            PanelIndex::new(3, keyless.clone(), Some(short_time)),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                ..
            })
        ));
        assert!(matches!(
            PanelIndex::new(3, keyless, None),
            Err(PanelError::Common(CommonError::Internal { .. }))
        ));
    }

    #[test]
    fn panel_index_with_effect_names_keeps_default_time_name_when_none() {
        let entity = GroupCodes::from_labels(&labels(&["a", "a", "b"]));
        let index = PanelIndex::new(3, entity, None).unwrap();
        assert_eq!(index.effect_name(PanelDimension::Entity), "entity");
        assert_eq!(index.effect_name(PanelDimension::Time), "time");

        let index = index.with_effect_names("firm".to_string(), None);
        assert_eq!(index.effect_name(PanelDimension::Entity), "firm");
        assert_eq!(index.effect_name(PanelDimension::Time), "time");

        // 1-way（`time`なし）でも2つ目の名前は保持される（エラーには使われない）。
        let index = index.with_effect_names("firm".to_string(), Some("year".to_string()));
        assert_eq!(index.effect_name(PanelDimension::Time), "year");
    }
}
