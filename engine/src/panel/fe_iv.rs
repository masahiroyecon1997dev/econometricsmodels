//! FEIV（固定効果IV）の入力データ型（`FeIvInput`）。
//!
//! `FeInput`と同じく`engine`はpolars/PyO3を知らず、`engine_pybind`が抽出した列
//! （`&[f64]`/`&[Vec<f64>]`）と、コード化済みの`entity`（`GroupCodes`）・`time`（`TimeKeys`）を
//! 受け取る。`faer::Mat`は組み立てない（within変換が列単位のため、`FeInput`と同じ列スライス）。
//!
//! 列は役割で分ける: 被説明変数`y`、外生の説明変数`x_exog`、内生の説明変数`x_endog`、
//! 操作変数`instruments`。固定効果が定数項を吸収するため、定数項は持たない。
//! `entity`・`time`・効果の列名は`PanelIndex`（`common.rs`）にまとめ、各列は同じ索引を共有する。
//!
//! 空の役割（`x_exog`が空は正当、`x_endog`・`instruments`が空は推定できない）の検査は
//! `IvInput`と同じ層分けで、入力型は受理する。`x_endog`・`instruments`が空であることの拒否は
//! `engine_pybind`の列抽出側、`len(instruments) < len(x_endog)`は推定器側の責務。

use crate::shared::group_codes::GroupCodes;

use super::common::{PanelError, PanelIndex, TimeKeys, validate_column_lengths};

/// ある役割（`x_exog`・`x_endog`・`instruments`）の列と列名。
///
/// `names`と`columns`は同じ長さで、同じ順序に対応する。`FeIvInput::from_group_codes`の
/// 引数をまとめるための入れ物。
#[derive(Debug, Clone)]
pub struct RoleColumns<'a> {
    /// 列名。`columns`の列と対応する。
    pub names: Vec<String>,
    /// 列（各列は長さ`n`）。
    pub columns: &'a [Vec<f64>],
}

/// FEIVの入力データ。フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」）。
/// `from_group_codes`で構築した後はgetter経由でのみアクセスする。
#[derive(Debug, Clone)]
pub struct FeIvInput {
    /// 被説明変数（長さ`n`、行はパネルの観測順）。
    y: Vec<f64>,
    /// 被説明変数名。
    dep_var_name: String,
    /// 外生の説明変数（各列は長さ`n`）。within変換前の生の値。
    x_exog: Vec<Vec<f64>>,
    x_exog_names: Vec<String>,
    /// 内生の説明変数（各列は長さ`n`）。within変換前の生の値。
    x_endog: Vec<Vec<f64>>,
    x_endog_names: Vec<String>,
    /// 操作変数（各列は長さ`n`）。within変換前の生の値。
    instruments: Vec<Vec<f64>>,
    instrument_names: Vec<String>,
    /// 個体・時点の索引と効果の列名。
    index: PanelIndex,
}

impl FeIvInput {
    /// 抽出済みの列とコード化済みの索引から`FeIvInput`を組み立てる。
    ///
    /// `entity`はキー付き（`GroupCodes::from_labels`/`from_factorized`）であること。
    /// `time`は2-wayの時間次元で、1-wayなら`None`。
    ///
    /// # Errors
    /// 検査の順序は、`x_exog`・`x_endog`・`instruments`の各列の長さ（`y`との不一致は
    /// `PanelError::Common(CommonError::DimensionMismatch)`。この順に最初の不一致を返す）、
    /// 続いて`PanelIndex::new`の検査（`entity`の長さ、`time`の長さ、`entity`のキーの有無）。
    ///
    /// within変換・singleton検出・分散ゼロ検証・バランスパネルの検証・識別の順序条件
    /// （操作変数の数）は行わない（`fit()`側が担う）。
    ///
    /// # パニックについて
    /// 役割ごとに`names.len() != columns.len()`の場合は`debug_assert!`でパニックする
    /// （呼び出し側`engine_pybind`の実装バグでしか起こり得ない内部契約。`FeInput`と同じ）。
    pub fn from_group_codes(
        y: &[f64],
        dep_var_name: String,
        x_exog: RoleColumns<'_>,
        x_endog: RoleColumns<'_>,
        instruments: RoleColumns<'_>,
        entity: GroupCodes,
        time: Option<TimeKeys>,
    ) -> Result<Self, PanelError> {
        for role in [&x_exog, &x_endog, &instruments] {
            debug_assert_eq!(
                role.columns.len(),
                role.names.len(),
                "columns and names must have the same length"
            );
            validate_column_lengths(y.len(), role.columns)?;
        }
        let index = PanelIndex::new(y.len(), entity, time)?;

        Ok(Self {
            y: y.to_vec(),
            dep_var_name,
            x_exog: x_exog.columns.to_vec(),
            x_exog_names: x_exog.names,
            x_endog: x_endog.columns.to_vec(),
            x_endog_names: x_endog.names,
            instruments: instruments.columns.to_vec(),
            instrument_names: instruments.names,
            index,
        })
    }

    /// 効果の次元の列名を設定する（`engine_pybind`が`effects`の各要素を渡す）。エラーの`column`に
    /// 載るだけで、計算には影響しない。設定済みの名前は全て置き換わり、`time`が`None`なら
    /// 2つ目の名前は既定（`"time"`）になる。
    pub fn with_effect_names(mut self, entity: String, time: Option<String>) -> Self {
        self.index = self.index.with_effect_names(entity, time);
        self
    }

    /// 個体・時点の索引。
    pub fn index(&self) -> &PanelIndex {
        &self.index
    }

    /// 被説明変数（長さ`n`）。
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    /// 被説明変数名。
    pub fn dep_var_name(&self) -> &str {
        &self.dep_var_name
    }

    /// 外生の説明変数（各列は長さ`n`）。
    pub fn x_exog(&self) -> &[Vec<f64>] {
        &self.x_exog
    }

    /// 外生の説明変数名。`x_exog()`の列と対応する。
    pub fn x_exog_names(&self) -> &[String] {
        &self.x_exog_names
    }

    /// 内生の説明変数（各列は長さ`n`）。
    pub fn x_endog(&self) -> &[Vec<f64>] {
        &self.x_endog
    }

    /// 内生の説明変数名。`x_endog()`の列と対応する。
    pub fn x_endog_names(&self) -> &[String] {
        &self.x_endog_names
    }

    /// 操作変数（各列は長さ`n`）。
    pub fn instruments(&self) -> &[Vec<f64>] {
        &self.instruments
    }

    /// 操作変数名。`instruments()`の列と対応する。
    pub fn instrument_names(&self) -> &[String] {
        &self.instrument_names
    }

    /// 観測数 n
    pub fn nobs(&self) -> usize {
        self.y.len()
    }

    /// 外生の説明変数の数。
    pub fn k_exog(&self) -> usize {
        self.x_exog.len()
    }

    /// 内生の説明変数の数。
    pub fn k_endog(&self) -> usize {
        self.x_endog.len()
    }

    /// 操作変数の数。
    pub fn k_instruments(&self) -> usize {
        self.instruments.len()
    }

    /// 構造方程式の係数の数 `k = k_exog + k_endog`（定数項は持たない）。
    pub fn k(&self) -> usize {
        self.k_exog() + self.k_endog()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panel::common::PanelDimension;
    use crate::shared::error::CommonError;

    fn labels(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    fn role<'a>(names: &[&str], columns: &'a [Vec<f64>]) -> RoleColumns<'a> {
        RoleColumns {
            names: labels(names),
            columns,
        }
    }

    /// 2個体 × 3時点（観測順は個体ごと）。
    fn entity() -> GroupCodes {
        GroupCodes::from_labels(&labels(&["a", "a", "a", "b", "b", "b"]))
    }

    fn time() -> TimeKeys {
        TimeKeys::lexicographic(labels(&["1", "2", "3", "1", "2", "3"]))
    }

    fn y() -> Vec<f64> {
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    }

    fn col(offset: f64) -> Vec<f64> {
        (0..6).map(|i| i as f64 + offset).collect()
    }

    #[test]
    fn from_group_codes_builds_one_way_input() {
        let (exog, endog, inst) = ([col(0.0)], [col(1.0)], [col(2.0), col(3.0)]);
        let input = FeIvInput::from_group_codes(
            &y(),
            "y".to_string(),
            role(&["w"], &exog),
            role(&["d"], &endog),
            role(&["z1", "z2"], &inst),
            entity(),
            None,
        )
        .unwrap();

        assert_eq!(input.nobs(), 6);
        assert_eq!(input.y(), y().as_slice());
        assert_eq!(input.dep_var_name(), "y");
        assert_eq!(input.x_exog(), exog.as_slice());
        assert_eq!(input.x_exog_names(), labels(&["w"]).as_slice());
        assert_eq!(input.x_endog(), endog.as_slice());
        assert_eq!(input.x_endog_names(), labels(&["d"]).as_slice());
        assert_eq!(input.instruments(), inst.as_slice());
        assert_eq!(input.instrument_names(), labels(&["z1", "z2"]).as_slice());
        assert_eq!(
            (
                input.k_exog(),
                input.k_endog(),
                input.k_instruments(),
                input.k()
            ),
            (1, 1, 2, 2)
        );
        assert_eq!(input.index().n_entities(), 2);
        assert_eq!(
            input.index().entity_labels(),
            labels(&["a", "b"]).as_slice()
        );
        assert!(input.index().time_keys().is_none());
    }

    #[test]
    fn from_group_codes_builds_two_way_input() {
        let (endog, inst) = ([col(1.0)], [col(2.0)]);
        let input = FeIvInput::from_group_codes(
            &y(),
            "y".to_string(),
            role(&[], &[]),
            role(&["d"], &endog),
            role(&["z"], &inst),
            entity(),
            Some(time()),
        )
        .unwrap();

        assert_eq!(input.index().time_keys(), Some(&time()));
        assert_eq!(input.index().nobs(), 6);
    }

    #[test]
    fn from_group_codes_accepts_empty_exogenous_endogenous_and_instrument_roles() {
        // 空の役割の拒否は`engine_pybind`と`fit()`の責務（モジュールdoc）。
        let input = FeIvInput::from_group_codes(
            &y(),
            "y".to_string(),
            role(&[], &[]),
            role(&[], &[]),
            role(&[], &[]),
            entity(),
            None,
        )
        .unwrap();
        assert_eq!(
            (
                input.k_exog(),
                input.k_endog(),
                input.k_instruments(),
                input.k()
            ),
            (0, 0, 0, 0)
        );
    }

    #[test]
    fn from_group_codes_with_zero_observations_succeeds() {
        let input = FeIvInput::from_group_codes(
            &[],
            "y".to_string(),
            role(&[], &[]),
            role(&[], &[]),
            role(&[], &[]),
            GroupCodes::from_labels(&[]),
            None,
        )
        .unwrap();
        assert_eq!(input.nobs(), 0);
    }

    #[test]
    fn from_group_codes_returns_dimension_mismatch_for_each_role() {
        let short = [vec![1.0, 2.0]];
        let (full_a, full_b) = ([col(1.0)], [col(2.0)]);
        let expected = Err(CommonError::DimensionMismatch {
            y_rows: 6,
            x_rows: 2,
        }
        .into());
        let build = |exog: &[Vec<f64>], endog: &[Vec<f64>], inst: &[Vec<f64>]| {
            FeIvInput::from_group_codes(
                &y(),
                "y".to_string(),
                role(&vec!["c"; exog.len()], exog),
                role(&vec!["c"; endog.len()], endog),
                role(&vec!["c"; inst.len()], inst),
                entity(),
                None,
            )
            .map(|_| ())
        };

        assert_eq!(build(&short, &full_a, &full_b), expected);
        assert_eq!(build(&[], &short, &full_b), expected);
        assert_eq!(build(&[], &full_a, &short), expected);
    }

    #[test]
    fn from_group_codes_checks_roles_in_order_x_exog_x_endog_instruments() {
        // 3役割とも長さが違うとき、最初に検査される役割の長さが返る。
        let (exog, endog, inst) = ([vec![0.0; 2]], [vec![0.0; 3]], [vec![0.0; 4]]);
        let build = |exog: &[Vec<f64>], endog: &[Vec<f64>], inst: &[Vec<f64>]| {
            FeIvInput::from_group_codes(
                &y(),
                "y".to_string(),
                role(&vec!["c"; exog.len()], exog),
                role(&vec!["c"; endog.len()], endog),
                role(&vec!["c"; inst.len()], inst),
                entity(),
                None,
            )
            .map(|_| ())
        };
        let mismatch = |x_rows| Err(CommonError::DimensionMismatch { y_rows: 6, x_rows }.into());

        assert_eq!(build(&exog, &endog, &inst), mismatch(2));
        assert_eq!(build(&[], &endog, &inst), mismatch(3));
        assert_eq!(build(&[], &[], &inst), mismatch(4));
    }

    #[test]
    fn from_group_codes_checks_column_lengths_before_the_index() {
        // 列の長さも索引の長さも合っていないとき、先に列の不一致が返る。
        let short = [vec![1.0, 2.0]];
        let short_entity = GroupCodes::from_labels(&labels(&["a", "b"]));
        let err = FeIvInput::from_group_codes(
            &y(),
            "y".to_string(),
            role(&["w"], &short),
            role(&[], &[]),
            role(&[], &[]),
            short_entity,
            None,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            PanelError::Common(CommonError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn from_group_codes_returns_identifier_dimension_mismatch_for_entity_and_time() {
        let short_entity = GroupCodes::from_labels(&labels(&["a", "b"]));
        assert_eq!(
            FeIvInput::from_group_codes(
                &y(),
                "y".to_string(),
                role(&[], &[]),
                role(&[], &[]),
                role(&[], &[]),
                short_entity,
                None,
            )
            .map(|_| ()),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Entity,
                y_rows: 6,
                other_rows: 2,
            })
        );

        let short_time = TimeKeys::lexicographic(labels(&["1", "2"]));
        assert_eq!(
            FeIvInput::from_group_codes(
                &y(),
                "y".to_string(),
                role(&[], &[]),
                role(&[], &[]),
                role(&[], &[]),
                entity(),
                Some(short_time),
            )
            .map(|_| ()),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                y_rows: 6,
                other_rows: 2,
            })
        );
    }

    #[test]
    fn from_group_codes_rejects_entity_codes_without_keys() {
        let keyless =
            GroupCodes::from_labels_without_keys(&labels(&["a", "a", "a", "b", "b", "b"]));
        let err = FeIvInput::from_group_codes(
            &y(),
            "y".to_string(),
            role(&[], &[]),
            role(&[], &[]),
            role(&[], &[]),
            keyless,
            None,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            PanelError::Common(CommonError::Internal { .. })
        ));
    }

    #[test]
    fn with_effect_names_sets_the_names_reported_in_errors() {
        let input = FeIvInput::from_group_codes(
            &y(),
            "y".to_string(),
            role(&[], &[]),
            role(&[], &[]),
            role(&[], &[]),
            entity(),
            Some(time()),
        )
        .unwrap();
        assert_eq!(input.index().effect_name(PanelDimension::Entity), "entity");
        assert_eq!(input.index().effect_name(PanelDimension::Time), "time");

        let input = input.with_effect_names("firm".to_string(), Some("year".to_string()));
        assert_eq!(input.index().effect_name(PanelDimension::Entity), "firm");
        assert_eq!(input.index().effect_name(PanelDimension::Time), "year");
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "columns and names must have the same length")]
    fn from_group_codes_panics_on_mismatched_names_arity() {
        let endog = [col(1.0)];
        let _ = FeIvInput::from_group_codes(
            &y(),
            "y".to_string(),
            role(&[], &[]),
            role(&["d", "extra"], &endog),
            role(&[], &[]),
            entity(),
            None,
        );
    }
}
