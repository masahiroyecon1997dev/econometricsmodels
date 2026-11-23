//! パネル索引`PanelIndex`と、効果の次元ごとの列名`EffectNames`。

use super::error::{PanelDimension, PanelError};
use super::validation::validate_panel_index;
use crate::shared::group_codes::GroupCodes;

/// 効果の次元ごとの列名。エラーの`column`に載せるためだけに`FeInput`/`ReInput`が保持する
/// （`engine_pybind`が`effects`の各要素の列名を渡す）。計算には使わない。
///
/// 既定は`"entity"`・`"time"`・`"effect3"`・…で、`engine`単体（テスト等）の構築ではこのまま使う。
/// 次元の数以上の名前を持つ（持つ名前より後ろの次元は既定の名前になる）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EffectNames {
    names: Vec<String>,
}

impl Default for EffectNames {
    fn default() -> Self {
        Self::defaults(2)
    }
}

impl EffectNames {
    /// `n_effects`個の次元に既定の名前を付ける。1-wayでも2つ目の名前（`"time"`）は持つ
    /// （エラーには使われない）。
    pub(crate) fn defaults(n_effects: usize) -> Self {
        Self {
            names: (0..n_effects.max(2)).map(default_name).collect(),
        }
    }

    /// `names`の数を次元の数として、先頭の次元から順に当てはめる。
    pub(crate) fn new(names: Vec<String>) -> Self {
        Self::for_effects(names.len(), names)
    }

    /// `n_effects`個の次元の名前を持つ。`names`を先頭の次元から順に当てはめ、足りない次元は既定の
    /// 名前のまま、次元の数より多い名前は捨てる（1-wayや、名前を渡さなかった次元）。
    pub(crate) fn for_effects(n_effects: usize, names: Vec<String>) -> Self {
        let mut all = Self::defaults(n_effects);
        for (slot, name) in all.names.iter_mut().zip(names.into_iter().take(n_effects)) {
            *slot = name;
        }
        all
    }

    /// `dimension`に対応する列名。
    pub(crate) fn get(&self, dimension: PanelDimension) -> &str {
        self.names
            .get(dimension.index())
            .map_or("effects", String::as_str)
    }
}

/// 名前を渡されなかった`index`番目（0始まり）の次元の既定の名前。
fn default_name(index: usize) -> String {
    match index {
        0 => "entity".to_string(),
        1 => "time".to_string(),
        _ => format!("effect{}", index + 1),
    }
}

/// パネルの索引（効果の各次元）と効果の列名。観測の行ごとにどの水準に属するかを
/// 持つ入れ物で、列データ（`y`・`x`等）は持たない。`FeIvInput`が保持し、役割の異なる複数の
/// 列を同じ索引で扱えるようにする。
///
/// 次元は`effects`の順に`GroupCodes`で持つ（1つ目が個体、2つ目が2-wayの時点、3つ目以降は
/// 高次元の固定効果）。時点の次元のコードは時間順に振ってあり、順序づけは`TimeKeys`が作る
/// （呼び出し側が決める）。Driscoll-Kraayの時点キーは効果の次元とは別なので、ここには持たない。
/// 構築時に観測数との整合を検証するので、構築後は長さの不一致が起きない。フィールドはprivate。
#[derive(Debug, Clone)]
pub struct PanelIndex {
    /// 効果の各次元のコード（`effects`の順、キー付き、空でない）。
    groups: Vec<GroupCodes>,
    effect_names: EffectNames,
}

impl PanelIndex {
    /// 観測数`nobs`の索引を組み立てる。`groups`は`effects`の各列のコードで、空であってはならない。
    /// 効果の列名は既定（`"entity"`・`"time"`・`"effect3"`…）で、
    /// [`with_effect_names`](Self::with_effect_names)で設定する。
    ///
    /// # Errors
    /// 検査の順序は`validate_panel_index`が正本。
    /// - 次元の長さが`nobs`と一致しない場合は、`effects`の先頭から見て最初のものが
    ///   `PanelError::IdentifierDimensionMismatch { dimension, .. }`
    /// - 次元がキー無しのコードの場合は`PanelError::Common`（`CommonError::Internal`、
    ///   呼び出し側の実装誤り）。`groups`が空の場合も同じ
    pub fn new(nobs: usize, groups: Vec<GroupCodes>) -> Result<Self, PanelError> {
        validate_panel_index(nobs, &groups.iter().collect::<Vec<_>>())?;
        let effect_names = EffectNames::defaults(groups.len());
        Ok(Self {
            groups,
            effect_names,
        })
    }

    /// 効果の次元の列名を、`effects`の先頭から順に設定する。エラーの`column`に載るだけで、
    /// 計算には影響しない。次元の数より多い名前は無視し、足りない次元は既定の名前のまま。
    pub fn with_effect_names(mut self, names: Vec<String>) -> Self {
        let n_effects = self.groups.len();
        self.effect_names = EffectNames::for_effects(n_effects, names);
        self
    }

    /// 効果の列名。エラーの`column`に載せるために運ぶ。
    pub(crate) fn effect_names(&self) -> &EffectNames {
        &self.effect_names
    }

    /// `dimension`に対応する効果の列名。
    pub fn effect_name(&self, dimension: PanelDimension) -> &str {
        self.effect_names.get(dimension)
    }

    /// 効果の次元の数（`effects`の長さ）。
    pub fn n_effects(&self) -> usize {
        self.groups.len()
    }

    /// 効果の各次元のコード（`effects`の順、キー付き）。
    pub fn effect_codes(&self) -> &[GroupCodes] {
        &self.groups
    }

    /// 1つ目の次元（個体）の整数コード（キー付き）。
    pub fn entity_codes(&self) -> &GroupCodes {
        &self.groups[0]
    }

    /// ユニークなエンティティ数。
    pub fn n_entities(&self) -> usize {
        self.entity_codes().n_groups()
    }

    /// ユニークなエンティティのラベル（コード順＝辞書順）。
    pub fn entity_labels(&self) -> &[String] {
        self.entity_codes().keys()
    }

    /// 観測数。
    pub fn nobs(&self) -> usize {
        self.entity_codes().nobs()
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::super::time_keys::TimeKeys;
    use super::*;
    use crate::shared::error::CommonError;

    #[test]
    fn panel_index_new_exposes_codes_and_labels_of_every_dimension() {
        let entity = GroupCodes::from_labels(&labels(&["b", "b", "a"]));
        let time = TimeKeys::lexicographic(labels(&["1", "2", "1"]));
        let industry = GroupCodes::from_labels(&labels(&["x", "y", "y"]));
        let index =
            PanelIndex::new(3, vec![entity, time.codes().clone(), industry.clone()]).unwrap();

        assert_eq!(index.nobs(), 3);
        assert_eq!(index.n_effects(), 3);
        assert_eq!(index.n_entities(), 2);
        assert_eq!(index.entity_labels(), labels(&["a", "b"]).as_slice());
        assert_eq!(index.entity_codes().n_groups(), 2);
        assert_eq!(index.effect_codes()[1].keys(), time.periods());
        assert_eq!(index.effect_codes()[2], industry);
        assert_eq!(
            PanelIndex::new(3, vec![GroupCodes::from_labels(&labels(&["a", "a", "b"]))])
                .unwrap()
                .n_effects(),
            1
        );
    }

    #[test]
    fn panel_index_new_reports_length_mismatches_then_missing_keys() {
        let entity = GroupCodes::from_labels(&labels(&["a", "a", "b"]));
        let short = GroupCodes::from_labels(&labels(&["1"]));
        let keyless = GroupCodes::from_labels_without_keys(&labels(&["a", "a", "b"]));

        assert!(matches!(
            PanelIndex::new(4, vec![entity.clone()]),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::ENTITY,
                y_rows: 4,
                other_rows: 3
            })
        ));
        assert!(matches!(
            PanelIndex::new(3, vec![entity.clone(), short.clone()]),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::TIME,
                y_rows: 3,
                other_rows: 1
            })
        ));
        assert!(matches!(
            PanelIndex::new(3, vec![entity.clone(), entity.clone(), short.clone()]),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension,
                ..
            }) if dimension == PanelDimension::new(2)
        ));
        // キーの検査は最後なので、長さの不一致が先に返る。
        assert!(matches!(
            PanelIndex::new(3, vec![keyless.clone(), short]),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::TIME,
                ..
            })
        ));
        assert!(matches!(
            PanelIndex::new(3, vec![keyless]),
            Err(PanelError::Common(CommonError::Internal { .. }))
        ));
        assert!(matches!(
            PanelIndex::new(3, vec![]),
            Err(PanelError::Common(CommonError::Internal { .. }))
        ));
    }

    #[test]
    fn panel_index_with_effect_names_keeps_default_names_for_the_rest() {
        let entity = GroupCodes::from_labels(&labels(&["a", "a", "b"]));
        let index = PanelIndex::new(3, vec![entity.clone()]).unwrap();
        assert_eq!(index.effect_name(PanelDimension::ENTITY), "entity");
        assert_eq!(index.effect_name(PanelDimension::TIME), "time");

        let index = index.with_effect_names(labels(&["firm"]));
        assert_eq!(index.effect_name(PanelDimension::ENTITY), "firm");
        assert_eq!(index.effect_name(PanelDimension::TIME), "time");

        // 1-wayでは2つ目以降の名前は無視される（エラーには使われない）。
        let index = index.with_effect_names(labels(&["firm", "year"]));
        assert_eq!(index.effect_name(PanelDimension::TIME), "time");

        // 3次元: 渡した名前は先頭から当てはまり、残りは既定の名前になる。
        let three = PanelIndex::new(3, vec![entity.clone(), entity.clone(), entity.clone()])
            .unwrap()
            .with_effect_names(labels(&["firm", "year"]));
        assert_eq!(three.effect_name(PanelDimension::ENTITY), "firm");
        assert_eq!(three.effect_name(PanelDimension::TIME), "year");
        assert_eq!(three.effect_name(PanelDimension::new(2)), "effect3");
    }

    #[test]
    fn effect_names_default_to_entity_and_time() {
        let names = EffectNames::default();
        assert_eq!(names.get(PanelDimension::ENTITY), "entity");
        assert_eq!(names.get(PanelDimension::TIME), "time");
        assert_eq!(names.get(PanelDimension::new(3)), "effects");
        let names = EffectNames::new(labels(&["firm"]));
        assert_eq!(names.get(PanelDimension::ENTITY), "firm");
        assert_eq!(names.get(PanelDimension::TIME), "time");
    }
}
