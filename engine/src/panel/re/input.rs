//! `ReInput`（REの入力データ）と、REの個体効果の方向・`cov_type`の型別名。

use crate::panel::common::{
    EffectNames, PanelCovType, PanelEffects, PanelError, TimeKeys, validate_panel_input,
};
use crate::shared::group_codes::GroupCodes;

/// REの被説明変数・説明変数・パネル識別子を保持する入力データ。
///
/// 準偏差変換前の生データを保持するだけの入れ物（`Mat`を組み立てない理由はモジュール
/// doc参照）。フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」）。
/// `from_columns`で構築した後はgetter経由でのみアクセスする。
#[derive(Debug)]
pub struct ReInput {
    /// 被説明変数（長さ`n`、行はパネルの観測順）。
    y: Vec<f64>,
    /// 説明変数（各列は長さ`n`）。準偏差変換前の生の値。
    x: Vec<Vec<f64>>,
    /// 説明変数名。`x`の列と対応する。
    x_names: Vec<String>,
    /// 各行の時点コード（時間順、長さ`n`）。2-way RE（`ReEffects::TwoWay`）の時間次元
    /// （`effects`の2要素目）。1-wayでは使わない。Driscoll-Kraay HACの時点列とは別物で、
    /// DKの時点は`ReCovType::Dk.time`で別に渡す（FEと同じ規約、`panel-common.md`1.1節）。
    time: Option<TimeKeys>,
    /// 被説明変数名。
    dep_var_name: String,
    /// `entity`の整数コード（構築時に一度だけ作る、`GroupCodes`のdocコメント参照）。
    entity_codes: GroupCodes,
    /// 効果の次元ごとの列名（エラーの`column`用。計算には使わない）。
    effect_names: EffectNames,
}

impl ReInput {
    /// 列ごとの`Vec<f64>`/`Vec<String>`（`engine_pybind`がpolars DataFrameから抽出済み）
    /// から`ReInput`を組み立てる。`FeInput::from_columns`と同一の次元検証を行う。
    ///
    /// # Errors
    /// 検査の順序は`validate_panel_input`（`panel/common/validation.rs`、FE/REで共通）が正本。
    /// - いずれかの`x_columns`の長さが`y`と一致しない場合は
    ///   `PanelError::Common(CommonError::DimensionMismatch)`
    /// - `entity`の長さが`y`と一致しない場合は
    ///   `PanelError::IdentifierDimensionMismatch { dimension: PanelDimension::ENTITY, .. }`
    /// - `time`が`Some`で、その長さが`y`と一致しない場合は
    ///   `PanelError::IdentifierDimensionMismatch { dimension: PanelDimension::TIME, .. }`
    ///
    /// 準偏差変換の実施・θ計算・分散成分推定・ハウスマン検定は行わない
    /// （いずれも`fit()`側が担う、`docs/spec/re-spec.md`）。
    ///
    /// # パニックについて
    /// `x_names.len() != x_columns.len()`の場合は`debug_assert!`でパニックする
    /// （`FeInput::from_columns`と同じ理由: 呼び出し側`engine_pybind`の実装バグでしか
    /// 起こり得ない内部契約であり、実データに起因する`ValidationError`とは性質が異なる）。
    #[cfg(test)]
    pub(crate) fn from_columns(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        entity: &[String],
        time: Option<&[String]>,
        dep_var_name: String,
    ) -> Result<Self, PanelError> {
        Self::from_columns_ordered(
            y,
            x_columns,
            x_names,
            entity,
            time.map(|t| TimeKeys::lexicographic(t.to_vec())),
            dep_var_name,
        )
    }

    /// `from_columns`の`time`を、順序を持つ`TimeKeys`で受け取る版。時点の順序をラベルの
    /// 辞書順ではなく列の値の順序にしたい場合（整数・日付等）に使う（`FeInput::
    /// from_columns_ordered`と同じ）。
    ///
    /// # Errors
    /// `from_columns`と同じ。
    ///
    /// # パニックについて
    /// `from_columns`と同じ。
    #[cfg(test)]
    pub(crate) fn from_columns_ordered(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        entity: &[String],
        time: Option<TimeKeys>,
        dep_var_name: String,
    ) -> Result<Self, PanelError> {
        Self::from_group_codes(
            y,
            x_columns,
            x_names,
            GroupCodes::from_labels(entity),
            time,
            dep_var_name,
        )
    }

    /// `from_columns_ordered`の`entity`を、コード化済みの`GroupCodes`で受け取る版
    /// （`FeInput::from_group_codes`と同じ。`engine_pybind`が使う）。
    ///
    /// # Errors
    /// `from_columns`と同じ。加えて、`entity`がキー無しのコードの場合は`PanelError::Common`
    /// （`CommonError::Internal`、呼び出し側の実装誤り）。
    ///
    /// # パニックについて
    /// `from_columns`と同じ。
    pub fn from_group_codes(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        entity: GroupCodes,
        time: Option<TimeKeys>,
        dep_var_name: String,
    ) -> Result<Self, PanelError> {
        let mut groups = vec![&entity];
        groups.extend(time.as_ref().map(TimeKeys::codes));
        validate_panel_input(y, x_columns, &x_names, &groups)?;

        Ok(Self {
            y: y.to_vec(),
            x: x_columns.to_vec(),
            x_names,
            time,
            dep_var_name,
            entity_codes: entity,
            effect_names: EffectNames::default(),
        })
    }

    /// 効果の次元の列名を設定する（`engine_pybind`が`effects`の各要素を渡す）。エラーの`column`に
    /// 載るだけで、計算には影響しない。`time`が`None`なら2つ目の名前は既定のまま。
    pub fn with_effect_names(mut self, entity: String, time: Option<String>) -> Self {
        self.effect_names = EffectNames::new(std::iter::once(entity).chain(time).collect());
        self
    }

    /// 効果の次元の列名。
    pub(crate) fn effect_names(&self) -> &EffectNames {
        &self.effect_names
    }

    /// `entity`の整数コード。
    pub(crate) fn entity_codes(&self) -> &GroupCodes {
        &self.entity_codes
    }

    /// ユニークなエンティティ数。
    pub fn n_entities(&self) -> usize {
        self.entity_codes.n_groups()
    }

    /// `time`の整数コード（`time`が無ければ`None`）。
    pub(crate) fn time_codes(&self) -> Option<&GroupCodes> {
        self.time.as_ref().map(TimeKeys::codes)
    }

    /// 被説明変数（長さ`n`）。
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    /// 説明変数（各列は長さ`n`）。
    pub fn x(&self) -> &[Vec<f64>] {
        &self.x
    }

    /// 説明変数名。`x()`の列と対応する。
    pub fn x_names(&self) -> &[String] {
        &self.x_names
    }

    /// ユニークなエンティティのラベル（コード順＝辞書順）。
    pub fn entity_labels(&self) -> &[String] {
        self.entity_codes.keys()
    }

    /// 時点のラベルと時間順のコード。未指定なら`None`。
    pub fn time_keys(&self) -> Option<&TimeKeys> {
        self.time.as_ref()
    }

    /// 被説明変数名。
    pub fn dep_var_name(&self) -> &str {
        &self.dep_var_name
    }

    /// 観測数 n
    pub fn nobs(&self) -> usize {
        self.y.len()
    }
}

/// REの個体効果の方向。[`PanelEffects`]の別名。
pub type ReEffects = PanelEffects;

/// REの標準誤差計算方式。[`PanelCovType`]の別名。
pub type ReCovType = PanelCovType;

#[cfg(test)]
mod tests {
    use super::super::test_support::*;

    use crate::panel::common::PanelDimension;

    use crate::panel::common::PanelError;

    use crate::panel::common::TimeKeys;

    use crate::panel::re::ReInput;

    use crate::shared::error::CommonError;

    use crate::shared::group_codes::GroupCodes;

    #[test]
    fn from_columns_builds_input_without_time() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let x1 = vec![10.0, 20.0, 30.0, 40.0];
        let entity = strings(&["a", "a", "b", "b"]);

        let input = ReInput::from_columns(
            &y,
            std::slice::from_ref(&x1),
            vec!["x1".to_string()],
            &entity,
            None,
            "y".to_string(),
        )
        .unwrap();

        assert_eq!(input.y(), &y);
        assert_eq!(input.x(), &[x1]);
        assert_eq!(input.x_names(), &["x1".to_string()]);
        // ユニークなラベルを辞書順（コード順）に並べたもの。
        assert_eq!(input.entity_labels(), ["a", "b"]);
        assert_eq!(input.time_keys(), None);
        assert_eq!(input.dep_var_name(), "y");
        assert_eq!(input.nobs(), 4);
        assert_eq!(input.n_entities(), 2);
    }

    #[test]
    fn from_group_codes_matches_from_columns_and_requires_entity_keys() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let x = vec![vec![1.0, 2.0, 3.0, 5.0]];
        let entity = strings(&["a", "a", "b", "b"]);

        let from_strings = ReInput::from_columns(
            &y,
            &x,
            vec!["x1".to_string()],
            &entity,
            None,
            "y".to_string(),
        )
        .unwrap();
        let from_codes = ReInput::from_group_codes(
            &y,
            &x,
            vec!["x1".to_string()],
            GroupCodes::from_factorized(vec![1, 1, 0, 0], &["b", "a"]).unwrap(),
            None,
            "y".to_string(),
        )
        .unwrap();
        assert_eq!(from_codes.entity_labels(), from_strings.entity_labels());
        assert_eq!(from_codes.n_entities(), 2);

        // キー無しのコードは固定効果のキーが作れないため受け付けない。
        let keyless = GroupCodes::from_labels_without_keys(&entity);
        assert!(matches!(
            ReInput::from_group_codes(
                &y,
                &x,
                vec!["x1".to_string()],
                keyless,
                None,
                "y".to_string()
            ),
            Err(PanelError::Common(CommonError::Internal { .. }))
        ));
        // 行数の食い違いは従来どおり`IdentifierDimensionMismatch`。
        assert!(matches!(
            ReInput::from_group_codes(
                &y,
                &x,
                vec!["x1".to_string()],
                GroupCodes::from_labels(&strings(&["a", "b"])),
                None,
                "y".to_string()
            ),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::ENTITY,
                ..
            })
        ));
    }

    /// エラーの優先順位: 次元の食い違い（`y`↔各`x`列、`entity`の行数、`time`の行数）が先で、
    /// `entity`がキー無しであることの拒否は最後。
    #[test]
    fn from_group_codes_reports_dimension_mismatches_before_missing_entity_keys() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let x = vec![vec![1.0, 2.0, 3.0, 5.0]];
        let keyless = |labels: &[&str]| GroupCodes::from_labels_without_keys(&strings(labels));

        // `time`の行数の食い違い（`entity`は正しい長さでキー付き）。
        assert!(matches!(
            ReInput::from_group_codes(
                &y,
                &x,
                vec!["x1".to_string()],
                GroupCodes::from_labels(&strings(&["a", "a", "b", "b"])),
                Some(TimeKeys::lexicographic(strings(&["1", "2"]))),
                "y".to_string()
            ),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::TIME,
                ..
            })
        ));
        // キー無しで、かつ`entity`の行数も食い違う: 行数のエラーが先。
        assert!(matches!(
            ReInput::from_group_codes(
                &y,
                &x,
                vec!["x1".to_string()],
                keyless(&["a", "b"]),
                None,
                "y".to_string()
            ),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::ENTITY,
                ..
            })
        ));
        // `x`の列の行数の食い違いは、`entity`の検証より先。
        assert!(matches!(
            ReInput::from_group_codes(
                &y,
                &[vec![1.0, 2.0]],
                vec!["x1".to_string()],
                keyless(&["a", "b"]),
                None,
                "y".to_string()
            ),
            Err(PanelError::Common(CommonError::DimensionMismatch { .. }))
        ));
    }

    #[test]
    fn from_columns_builds_input_with_time() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["2020", "2021", "2020", "2021"]);

        let input = ReInput::from_columns(
            &y,
            &[vec![1.0, 1.0, 1.0, 1.0]],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".to_string(),
        )
        .unwrap();

        // 時点は`lexicographic`（ラベルの辞書順）。行数と、ユニークなラベルの時間順を確認する。
        let Some(time_keys) = input.time_keys() else {
            panic!("time keys must be set");
        };
        assert_eq!(time_keys.len(), 4);
        assert_eq!(time_keys.periods(), ["2020", "2021"]);
    }

    #[test]
    fn from_columns_with_no_regressors_succeeds() {
        // OLSと異なりREは説明変数0個でも`ReInput`自体は構築できる（推定可能性の検証は
        // `fit()`側の責務、`FeInput`と同じ層分け）。
        let y = [1.0, 2.0];
        let entity = strings(&["a", "b"]);

        let input = ReInput::from_columns(&y, &[], vec![], &entity, None, "y".to_string()).unwrap();

        assert!(input.x().is_empty());
        assert!(input.x_names().is_empty());
    }

    #[test]
    fn from_columns_returns_dimension_mismatch_on_mismatched_x_column_length() {
        let y = [1.0, 2.0, 3.0];
        let x1 = vec![10.0, 20.0]; // yより短い
        let entity = strings(&["a", "b", "c"]);

        let result = ReInput::from_columns(
            &y,
            &[x1],
            vec!["x1".to_string()],
            &entity,
            None,
            "y".to_string(),
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::Common(CommonError::DimensionMismatch {
                y_rows: 3,
                x_rows: 2,
            })
        );
    }

    #[test]
    fn from_columns_returns_identifier_dimension_mismatch_for_entity() {
        let y = [1.0, 2.0, 3.0];
        let entity = strings(&["a", "b"]); // yより短い

        let result = ReInput::from_columns(&y, &[], vec![], &entity, None, "y".to_string());

        assert_eq!(
            result.unwrap_err(),
            PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::ENTITY,
                y_rows: 3,
                other_rows: 2,
            }
        );
    }

    #[test]
    fn from_columns_returns_identifier_dimension_mismatch_for_time() {
        let y = [1.0, 2.0, 3.0];
        let entity = strings(&["a", "b", "c"]);
        let time = strings(&["2020", "2021"]); // yより短い

        let result = ReInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".to_string());

        assert_eq!(
            result.unwrap_err(),
            PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::TIME,
                y_rows: 3,
                other_rows: 2,
            }
        );
    }

    #[test]
    fn from_columns_with_zero_observations_succeeds() {
        // n=0（y/entity/timeすべて空）でも次元は一致しているため`ReInput`自体の構築は
        // 成功する（`FeInput`の同名テストと同じ境界値の意図、
        // `.claude/rules/testing-policy.md`「境界値・悪条件」）。
        let input =
            ReInput::from_columns(&[], &[], vec![], &[], Some(&[]), "y".to_string()).unwrap();

        assert_eq!(input.nobs(), 0);
        assert_eq!(input.time_keys().map(TimeKeys::len), Some(0));
    }

    #[test]
    #[should_panic(expected = "x_columns and x_names must have the same length")]
    fn from_columns_panics_on_mismatched_names_arity() {
        let y = [1.0, 2.0];
        let entity = strings(&["a", "b"]);
        let _ = ReInput::from_columns(
            &y,
            &[vec![1.0, 2.0]],
            vec![], // x_columnsは1列だがx_namesは0個
            &entity,
            None,
            "y".to_string(),
        );
    }
}
