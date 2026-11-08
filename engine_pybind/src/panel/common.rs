//! `panel`系統（FE/RE）で共有するユーティリティ。
//!
//! `.claude/rules/rust-style.md`「ファイル・ディレクトリ構成」: 系統内で共有するロジックは
//! `<系統>/common.rs`に置く（`engine_pybind/src/iv/common/`と同じ位置づけ）。
//!
//! `PanelError`は`engine::shared::error::ErrorInfo`を実装しており、`code`・`params`
//! （`Common`バリアントの`CommonError`は`report()`が委譲する）は`engine`側が決める。
//! ここでは`crate::shared::errors::to_pyerr`に渡すだけで、バリアントごとの分岐は持たない。

use engine::panel::common::{PanelError, TimeKeys};
use engine::shared::error_catalog::{ErrorCode, ParamValue};
use engine::shared::group_codes::GroupCodes;
use polars::prelude::{DataFrame, DataType};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::shared::column_error::ColumnSource;
use crate::shared::column_extraction::{
    FactorizedKey, KeyRole, checked_key_column, extract_entity_groups, factorize_checked_key,
    physical_f64, physical_i128, with_group_labels,
};
use crate::shared::errors::{to_pyerr, user_error};
use crate::shared::option_values::{allowed_names, parse_choice};
use crate::shared::validation::{
    RoleValue, reject_unused_option, validate_no_duplicate_roles,
    validate_no_duplicate_within_role, validate_role_non_empty,
};

/// `cov_type`の検査でFEとREの文言を切り替えるためのモデル種別。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PanelModel {
    Fe,
    Re,
}

impl PanelModel {
    fn name(self) -> &'static str {
        match self {
            Self::Fe => "FE",
            Self::Re => "RE",
        }
    }
}

/// FE/REが受け付ける`cov_type`の値（`hc0`は含まない）。
const PANEL_COV_CHOICES: &[(&str, PanelCovKind)] = &[
    ("classical", PanelCovKind::Classical),
    ("hc1", PanelCovKind::Hc1),
    ("hc2", PanelCovKind::Hc2),
    ("hc3", PanelCovKind::Hc3),
    ("cluster", PanelCovKind::Cluster),
    ("dk", PanelCovKind::Dk),
];

/// FE/REの`cov_type`の種類。列名や`dk_bandwidth`を持たない、文字列だけから決まる値。
///
/// `cov_type`の検査は、DataFrameを見ない順に3つに分ける
/// （`docs/guide/validation.md`「Order in which errors are reported」。
/// `linear::common::CovKind`と同じ構成）。
///
/// - 値: [`parse_panel_cov_kind`]が文字列を`PanelCovKind`にする
/// - 組合せ: [`check_panel_cov_options`]が`cluster`/`dk_time`/`dk_bandwidth`と
///   `PanelCovKind`の矛盾を検査する
/// - 列: `fe.rs`/`re.rs`の`resolve_*_cov_type`が列を抽出して`engine`の型にする
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PanelCovKind {
    Classical,
    Hc1,
    Hc2,
    Hc3,
    Cluster,
    Dk,
}

impl PanelCovKind {
    /// `*Result.cov_type`に格納する正規形の名前。
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Classical => "classical",
            Self::Hc1 => "hc1",
            Self::Hc2 => "hc2",
            Self::Hc3 => "hc3",
            Self::Cluster => "cluster",
            Self::Dk => "dk",
        }
    }
}

/// `cov_type`文字列（大文字小文字を区別しない）を[`PanelCovKind`]にする。DataFrameは見ない。
///
/// `hc0`は`engine`の`FeCovType`/`ReCovType`に無い（参照実装が無い）ため、他の未知の値と
/// 区別した専用のメッセージで弾く。`hc0`はOLS/WLS/IVでは有効な値で、利用者が混同しやすい。
///
/// # Errors
/// 既知の値のいずれでもない場合は`ValidationError`。
pub(crate) fn parse_panel_cov_kind(cov_type: &str, model: PanelModel) -> PyResult<PanelCovKind> {
    if cov_type.to_lowercase() == "hc0" {
        return Err(user_error(
            ErrorCode::CovTypeNotAvailableForModel,
            vec![
                ("value", ParamValue::from(cov_type)),
                ("model", ParamValue::from(model.name())),
                (
                    "allowed",
                    ParamValue::from(allowed_names(PANEL_COV_CHOICES)),
                ),
            ],
        ));
    }
    parse_choice("cov_type", cov_type, PANEL_COV_CHOICES)
}

/// DKの時点列`dk_time`が指定されていることを確かめ、その列名を返す。
///
/// DKの時点列は`dk_time`で明示する（必須）。2-wayの時点（`effects`の2つ目。固定効果・ランダム効果の
/// 構造）を暗黙に借用すると、意図と違う列が選ばれても気づけないため。
///
/// # Errors
/// `dk_time`が未指定なら`ValidationError`。
pub(crate) fn require_dk_time(dk_time: Option<&str>) -> PyResult<&str> {
    dk_time.ok_or_else(|| user_error(ErrorCode::DkTimeRequired, Vec::new()))
}

/// `cluster`/`dk_time`/`dk_bandwidth`が`kind`と矛盾しないことを検査する。DataFrameは見ない。
///
/// `cluster`は`Cluster`、`dk_time`/`dk_bandwidth`は`Dk`でなければ使われない。その状態で
/// 指定されていれば`ValidationError`にする。`Dk`では`dk_time`が必須。使われない
/// オプションの検査が先で、必須の欠落が後。
///
/// # Errors
/// 使われないオプションが指定されている場合、または`Dk`で`dk_time`が未指定の場合は
/// `ValidationError`。
pub(crate) fn check_panel_cov_options(
    kind: PanelCovKind,
    cluster: Option<&str>,
    dk_time: Option<&str>,
    dk_bandwidth: Option<i64>,
) -> PyResult<()> {
    let is_dk = kind == PanelCovKind::Dk;
    reject_unused_option(
        "cluster",
        cluster.is_some(),
        kind == PanelCovKind::Cluster,
        "cov_type=\"cluster\"",
    )?;
    reject_unused_option("dk_time", dk_time.is_some(), is_dk, "cov_type=\"dk\"")?;
    reject_unused_option(
        "dk_bandwidth",
        dk_bandwidth.is_some(),
        is_dk,
        "cov_type=\"dk\"",
    )?;
    if is_dk {
        require_dk_time(dk_time)?;
    }
    Ok(())
}

/// `dk_time`（Driscoll-Kraay HACの時点列、FE/RE共通）が`y`・`effects`の先頭の列（個体）と同じ列でないことを
/// 検証する。
///
/// `dk_time`が`y`や`effects`の先頭の列（個体）と同じ列だと、時点が数値の`y`の値や個体IDそのものになり、
/// DKの時点構造が意味を成さない（`dk_time`が個体の列と同じなら全個体が単一時点ずつになる退化入力）。
/// 一方、`x`との重複は許可する（年トレンドを説明変数に入れつつDKの時点にも使う、等は
/// 正当な使い方）。`effects`の2つ目（2-wayの時間次元）との重複も許可する（固定効果と
/// DKで同じ時間粒度を使うのが典型）。そのため`validate_no_duplicate_roles`に渡す
/// ロールを`y`・`effects`の先頭・`dk_time`だけに絞り、`x`・`effects`の2つ目は含めない。
fn validate_dk_time_role(y: &str, first_effect: &str, dk_time: Option<&str>) -> PyResult<()> {
    let Some(dk_time) = dk_time else {
        return Ok(());
    };
    validate_no_duplicate_roles(&[
        ("y", RoleValue::Single(y)),
        ("effects", RoleValue::Single(first_effect)),
        ("dk_time", RoleValue::Single(dk_time)),
    ])
}

/// `effects`の列数の上限。FEは1要素が1-way、2要素が2-way、3要素以上が高次元の固定効果で、
/// 5列まで受け付ける。REは1-wayと2-wayだけ（交差ランダム効果は標準的な推定量でない）。
fn max_effects(model: PanelModel) -> usize {
    match model {
        PanelModel::Fe => MAX_FE_EFFECTS,
        PanelModel::Re => MAX_RE_EFFECTS,
    }
}

/// FEの`effects`の列数の上限。実務で想定する次元数に余裕を持たせ、極端な値を弾くために置く。
const MAX_FE_EFFECTS: usize = 5;

/// REの`effects`の列数の上限（1-wayと2-wayだけ）。
const MAX_RE_EFFECTS: usize = 2;

/// `effects`の列数を検査する。DataFrameは見ない。
///
/// 空は`ROLE_EMPTY`、[`max_effects`]を超える場合は`EFFECTS_TOO_MANY`。重複と`y`・`x`との
/// 重複は、列の役割の検査（`validate_no_duplicate_roles`等）が担う。
///
/// # Errors
/// 空、または[`max_effects`]を超える場合は`ValidationError`。
fn validate_effects(model: PanelModel, effects: &[String]) -> PyResult<()> {
    validate_role_non_empty("effects", effects)?;
    let max_columns = max_effects(model);
    if effects.len() > max_columns {
        return Err(user_error(
            ErrorCode::EffectsTooMany,
            vec![
                ("model", ParamValue::from(model.name())),
                ("n_columns", ParamValue::from(effects.len())),
                ("max_columns", ParamValue::from(max_columns)),
            ],
        ));
    }
    Ok(())
}

/// `y`・`x`・`effects`・`dk_time`の列の役割（DataFrameを見ない構造）を検査する。FE/RE共通。
///
/// 次の順に検査し、最初の違反で返す。
///
/// 1. `effects`の列数（空→`ROLE_EMPTY`、上限（FE=5、RE=2）超→`EFFECTS_TOO_MANY`。[`validate_effects`]）
/// 2. `x`が空でない
/// 3. `y`・`effects`・`x`の間に同じ列が無い
/// 4. `x`の内部に重複が無い
/// 5. `effects`の内部に重複が無い
/// 6. `dk_time`が`y`・`effects[0]`と同じ列でない（[`validate_dk_time_role`]）
///
/// `cov_type`・`variance_method`の値の検査と組合せの検査（[`parse_panel_cov_kind`]・
/// [`check_panel_cov_options`]）は呼び出し側で、これより前に行うこと（報告順は
/// `tests/cross_method/test_validation_order.py`が固定している）。手順6は手順1で空の`effects`を
/// 弾いた後に`effects[0]`を参照するため、手順の順序を入れ替えない。
///
/// # Errors
/// 上のいずれかに違反する場合は`ValidationError`。
pub(crate) fn validate_panel_roles(
    model: PanelModel,
    y: &str,
    x: &[String],
    effects: &[String],
    dk_time: Option<&str>,
) -> PyResult<()> {
    // `effects`の列数（空・上限超過）を先に見る。
    validate_effects(model, effects)?;
    // `x`が空リストであることを許容しない（`x=[]`は他手法と同じく`validate_role_non_empty`で拒否）。
    validate_role_non_empty("x", x)?;
    validate_no_duplicate_roles(&[
        ("y", RoleValue::Single(y)),
        ("effects", RoleValue::Multi(effects)),
        ("x", RoleValue::Multi(x)),
    ])?;
    validate_no_duplicate_within_role("x", x)?;
    validate_no_duplicate_within_role("effects", effects)?;
    // `dk_time`は`y`・`effects[0]`との重複だけ拒否する（`x`との重複は許可、`validate_dk_time_role`参照）。
    // `effects`が空でないことは手順1で確かめ済み。
    validate_dk_time_role(y, &effects[0], dk_time)
}

/// `effects`の列をDataFrameから取り出す。1つ目は個体（同一性だけが意味を持つキー）、
/// あれば2つ目は時点（値の順序を使うキー）として抽出する。
///
/// 列の型の規則は位置で決まる（1つ目は`KeyRole::Identity`、2つ目は`KeyRole::Time`）。
/// 呼び出し側が`validate_effects`で列数を検査済みであること。
///
/// # Errors
/// 列の抽出時に発覚する問題は`column_extraction`の責務で`ValidationError`。
pub(crate) fn extract_effects(
    df: &DataFrame,
    effects: &[String],
) -> PyResult<(GroupCodes, Option<TimeKeys>)> {
    let entity = extract_entity_groups(df, &effects[0], ColumnSource::new("effects", "data"))?;
    let time = effects
        .get(1)
        .map(|column| extract_time_keys(df, column, ColumnSource::new("effects", "data")))
        .transpose()?;
    Ok((entity, time))
}

/// FEの`effects`の列を、次元ごとのコードとしてDataFrameから取り出す（`effects`の順）。1つ目は個体
/// （同一性だけが意味を持つキー）、2つ目は時点（値の順序を使うキーを時間順のコードにしたもの）、
/// 3つ目以降は1つ目と同じ同一性だけのキー。
///
/// 列の型の規則は位置で決まる。呼び出し側が`validate_effects`で列数を検査済みであること。
///
/// # Errors
/// 列の抽出時に発覚する問題は`column_extraction`の責務で`ValidationError`。
pub(crate) fn extract_effect_groups(
    df: &DataFrame,
    effects: &[String],
) -> PyResult<Vec<GroupCodes>> {
    let source = || ColumnSource::new("effects", "data");
    effects
        .iter()
        .enumerate()
        .map(|(position, column)| match position {
            1 => Ok(extract_time_keys(df, column, source())?.into_codes()),
            _ => Ok(extract_entity_groups(df, column, source())?),
        })
        .collect()
}

/// 結果の`n_groups`ゲッターが返す`dict`（列名→グループ数、`effects`の順）を組み立てる。
pub(crate) fn groups_dict<'py>(
    py: Python<'py>,
    effects: &[String],
    counts: &[usize],
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (column, count) in effects.iter().zip(counts) {
        dict.set_item(column, count)?;
    }
    Ok(dict)
}

/// 結果の`n_groups`に載せる、`effects`の列ごとのグループ数（`effects`と同じ順）。
/// 1つ目は個体数、あれば2つ目は時点数。
pub(crate) fn group_counts(n_entities: usize, n_periods: Option<usize>) -> Vec<usize> {
    std::iter::once(n_entities).chain(n_periods).collect()
}

/// `engine::panel::common::PanelError`をPython例外に変換する。
///
/// `PanelError`（`engine`クレート）と`PyErr`（`pyo3`クレート）はどちらもこのクレートの外で
/// 定義された型のため、orphan ruleにより`impl From<PanelError> for PyErr`は書けない。関数
/// として実装し、呼び出し側で`.map_err(panel_error_to_pyerr)?`する（`iv_error_to_pyerr`と
/// 同じ理由、`engine_pybind/src/iv/common/`参照）。
///
/// `PanelError`の各バリアントは`ErrorInfo::report()`が`code`・`params`に変換する。
/// `WithinRegressionFailed`等の回帰の包みは原因の`code`に`stage`を付け、`ValidationError`/
/// `ComputationError`の別もカタログの`code`で決まる。
pub(crate) fn panel_error_to_pyerr(err: PanelError) -> PyErr {
    to_pyerr(&err)
}

/// `df`から`name`列を、時点を表すキー（FE/REの`effects`の2つ目・`dk_time`）として、ラベルと
/// **値の順序**を持つ`TimeKeys`で取り出す。行ごとの`String`は作らない
/// （`extract_cluster_groups`（`shared/column_extraction/`）と同じ分解）。
///
/// ラベルは文字列表現（同一性の判定と`fixed_effects()`のキーに使う）。時点の順序は列の
/// dtypeの値の順序で決める。文字列表現の辞書順は`1, 10, 11, 2, ...`のように時間順と
/// ずれるため使わない。
///
/// | dtype | 時点の順序 |
/// |---|---|
/// | 整数・`Date`・`Datetime`（タイムゾーンなし） | 値の昇順（`Date`・`Datetime`は時系列順） |
/// | 浮動小数 | 数値の昇順 |
/// | `Enum` | カテゴリの定義順 |
/// | 文字列・`Categorical` | ラベルの辞書順（バイト順） |
///
/// # Errors
/// `extract_group_key_column`と同じ（許可するdtypeだけが異なる）。
pub(crate) fn extract_time_keys(
    df: &DataFrame,
    name: &str,
    source: ColumnSource<'_>,
) -> PyResult<TimeKeys> {
    let column = checked_key_column(df, name, KeyRole::Time, source)?;
    let FactorizedKey {
        codes,
        representatives,
    } = factorize_checked_key(name, column)?;
    let dtype = representatives.dtype().clone();

    with_group_labels(name, &representatives, |labels| {
        if dtype.is_float() {
            let values = physical_f64(name, &representatives)?;
            TimeKeys::by_float_factorized(codes, labels, &values).map_err(panel_error_to_pyerr)
        } else if dtype.is_integer()
            || matches!(
                dtype,
                DataType::Date | DataType::Datetime(..) | DataType::Enum(..)
            )
        {
            // 整数は値そのもの、`Date`は日数、`Datetime`は時間単位ごとの経過時間、`Enum`は
            // カテゴリの定義順の添字が内部表現の整数になっており、その大小がそのまま時間順になる。
            let values = physical_i128(name, &representatives)?;
            TimeKeys::by_integer_factorized(codes, labels, &values).map_err(panel_error_to_pyerr)
        } else {
            // 文字列・`Categorical`（順序を持たない型）はラベルの辞書順。
            TimeKeys::lexicographic_factorized(codes, labels).map_err(panel_error_to_pyerr)
        }
    })
}

/// `build_fe_input`/`build_re_input`のテストで使う、6行・3エンティティ・2期間の整った`DataFrame`
/// （`y`・`x1`・`id`・`t`）。
#[cfg(test)]
pub(super) fn well_formed_df() -> DataFrame {
    polars::df!(
        "y" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x1" => [2.0, 4.0, 1.0, 5.0, 3.0, 6.0],
        "id" => ["a", "a", "b", "b", "c", "c"],
        "t" => ["1", "2", "1", "2", "1", "2"],
    )
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::column_extraction::extract_key_column;
    use polars::df;
    use polars::prelude::*;

    const SRC: ColumnSource<'static> = ColumnSource::new("x", "data");

    /// `unwrap()`/`expect()`は使わない：`PyErr`の`Debug`実装はGIL取得を要求し、GIL未初期化の
    /// この環境では二重パニックを起こす（`nonlinear/CLAUDE.md`「テストの制約」参照）。
    #[test]
    fn parse_panel_cov_kind_is_case_insensitive_and_has_canonical_names() {
        for (input, expected) in [
            ("classical", "classical"),
            ("CLASSICAL", "classical"),
            ("Hc1", "hc1"),
            ("HC2", "hc2"),
            ("hc3", "hc3"),
            ("CLUSTER", "cluster"),
            ("Dk", "dk"),
        ] {
            let Ok(kind) = parse_panel_cov_kind(input, PanelModel::Fe) else {
                panic!("expected Ok for input={input}");
            };
            assert_eq!(kind.name(), expected, "input={input}");
        }
    }

    #[test]
    fn parse_panel_cov_kind_rejects_hc0_and_unknown_values() {
        for model in [PanelModel::Fe, PanelModel::Re] {
            // `hc0`は参照実装が無く非対応。他の未知の値とは別のメッセージで弾く。
            assert!(parse_panel_cov_kind("hc0", model).is_err());
            assert!(parse_panel_cov_kind("bogus", model).is_err());
            // 別名は無い（FE/REの`hac`は`dk`へ改名済み）。
            assert!(parse_panel_cov_kind("hac", model).is_err());
            assert!(parse_panel_cov_kind("nonrobust", model).is_err());
        }
    }

    #[test]
    fn check_panel_cov_options_rejects_options_the_kind_does_not_use() {
        assert!(check_panel_cov_options(PanelCovKind::Classical, Some("g"), None, None).is_err());
        assert!(check_panel_cov_options(PanelCovKind::Cluster, None, Some("t"), None).is_err());
        assert!(check_panel_cov_options(PanelCovKind::Cluster, None, None, Some(2)).is_err());
        assert!(check_panel_cov_options(PanelCovKind::Dk, Some("g"), Some("t"), None).is_err());
        assert!(check_panel_cov_options(PanelCovKind::Cluster, Some("g"), None, None).is_ok());
        // `Cluster`で`cluster`が未指定なのはここでは弾かない（`effects`の先頭の列にフォールバックする）。
        assert!(check_panel_cov_options(PanelCovKind::Cluster, None, None, None).is_ok());
        assert!(check_panel_cov_options(PanelCovKind::Dk, None, Some("t"), Some(2)).is_ok());
    }

    #[test]
    fn check_panel_cov_options_rejects_every_option_for_kinds_that_use_none() {
        for kind in [
            PanelCovKind::Classical,
            PanelCovKind::Hc1,
            PanelCovKind::Hc2,
            PanelCovKind::Hc3,
        ] {
            assert!(check_panel_cov_options(kind, Some("g"), None, None).is_err());
            assert!(check_panel_cov_options(kind, None, Some("t"), None).is_err());
            assert!(check_panel_cov_options(kind, None, None, Some(2)).is_err());
            assert!(check_panel_cov_options(kind, None, None, None).is_ok());
        }
    }

    #[test]
    fn check_panel_cov_options_requires_dk_time_for_dk() {
        assert!(check_panel_cov_options(PanelCovKind::Dk, None, None, None).is_err());
        assert!(check_panel_cov_options(PanelCovKind::Dk, None, Some("t"), None).is_ok());
    }

    #[test]
    fn require_dk_time_returns_column_name_or_errors() {
        assert!(matches!(require_dk_time(Some("t")), Ok("t")));
        assert!(require_dk_time(None).is_err());
    }

    fn names(columns: &[&str]) -> Vec<String> {
        columns.iter().map(|c| c.to_string()).collect()
    }

    #[test]
    fn validate_effects_accepts_one_or_two_columns() {
        for columns in [vec!["a"], vec!["a", "b"]] {
            for model in [PanelModel::Fe, PanelModel::Re] {
                assert!(validate_effects(model, &names(&columns)).is_ok());
            }
        }
    }

    #[test]
    fn validate_effects_accepts_up_to_five_columns_for_fe_only() {
        let columns = names(&["a", "b", "c", "d", "e"]);
        for count in 3..=5 {
            assert!(validate_effects(PanelModel::Fe, &columns[..count]).is_ok());
            assert!(validate_effects(PanelModel::Re, &columns[..count]).is_err());
        }
    }

    #[test]
    fn validate_effects_rejects_empty_and_more_than_max_effects() {
        assert_eq!(
            (max_effects(PanelModel::Fe), max_effects(PanelModel::Re)),
            (5, 2)
        );
        for model in [PanelModel::Fe, PanelModel::Re] {
            assert!(validate_effects(model, &[]).is_err());
            assert!(validate_effects(model, &names(&["a", "b", "c", "d", "e", "f"])).is_err());
        }
    }

    /// `validate_panel_roles`がFE/REとも`ok`どおりの結果になることを確かめる。
    fn assert_roles(y: &str, x: &[&str], effects: &[&str], dk_time: Option<&str>, ok: bool) {
        for model in [PanelModel::Fe, PanelModel::Re] {
            let result = validate_panel_roles(model, y, &names(x), &names(effects), dk_time);
            assert_eq!(
                result.is_ok(),
                ok,
                "model={model:?} y={y} x={x:?} effects={effects:?} dk_time={dk_time:?}"
            );
        }
    }

    #[test]
    fn validate_panel_roles_rejects_empty_x() {
        // 固定効果・ランダム効果のみのモデル（`x=[]`）は拒否する。
        assert_roles("y", &[], &["id"], None, false);
    }

    #[test]
    fn validate_panel_roles_rejects_y_that_overlaps_effects() {
        assert_roles("y", &["x1"], &["y"], None, false);
    }

    #[test]
    fn validate_panel_roles_rejects_x_that_overlaps_effects() {
        // 個体の列、2つ目の列（2-wayの時点）のどちらとの重複も拒否する。
        assert_roles("y", &["x1", "id"], &["id"], None, false);
        assert_roles("y", &["x1", "t"], &["id", "t"], None, false);
    }

    #[test]
    fn validate_panel_roles_rejects_second_effect_that_duplicates_another_role() {
        for column in ["y", "id", "x1"] {
            assert_roles("y", &["x1"], &["id", column], None, false);
        }
    }

    #[test]
    fn validate_panel_roles_rejects_duplicates_within_x() {
        assert_roles("y", &["x1", "x1"], &["id"], None, false);
    }

    #[test]
    fn validate_panel_roles_dk_time_may_equal_second_effect_or_x_but_not_y_or_first_effect() {
        // 固定効果・ランダム効果と同じ時間粒度(`dk_time == effects[1]`)や、`x`との重複は
        // 正当な使い方。`y`・`effects[0]`と同じ列だけ拒否する(`validate_dk_time_role`参照)。
        let cases: [(&[&str], &str, bool); 5] = [
            (&["id", "t"], "t", true),
            (&["id"], "x1", true),
            (&["id"], "y", false),
            (&["id"], "id", false),
            (&["id", "t"], "id", false),
        ];
        for (effects, dk_time, ok) in cases {
            assert_roles("y", &["x1"], effects, Some(dk_time), ok);
        }
    }

    #[test]
    fn group_counts_has_one_entry_per_effect_in_order() {
        assert_eq!(group_counts(4, None), vec![4]);
        assert_eq!(group_counts(4, Some(3)), vec![4, 3]);
    }

    /// 行ごとの`String`を経由する素朴な`extract_time_keys`（比較対象の独立な参照実装）。
    fn extract_time_keys_via_row_strings(df: &DataFrame, name: &str) -> PyResult<TimeKeys> {
        let ids = extract_key_column(df, name, KeyRole::Time, SRC)?;
        let series = df.column(name).unwrap();
        let dtype = series.dtype();
        if dtype.is_float() {
            let as_f64 = series.cast(&DataType::Float64).unwrap();
            let values: Vec<f64> = as_f64.f64().unwrap().iter().map(|v| v.unwrap()).collect();
            Ok(TimeKeys::by_float(ids, &values).unwrap())
        } else if dtype.is_integer()
            || matches!(
                dtype,
                DataType::Date | DataType::Datetime(..) | DataType::Enum(..)
            )
        {
            let as_i128 = series.to_physical_repr().cast(&DataType::Int128).unwrap();
            let values: Vec<i128> = as_i128.i128().unwrap().iter().map(|v| v.unwrap()).collect();
            Ok(TimeKeys::by_integer(ids, &values).unwrap())
        } else {
            Ok(TimeKeys::lexicographic(ids))
        }
    }

    /// 時点列も、行ごとの`String`を経由する参照実装と同じ`TimeKeys`（コードと時点のラベル）になる。
    #[test]
    fn extract_time_keys_agrees_with_the_row_string_path_for_every_time_dtype() {
        let ints: [i64; 8] = [10, 9, 2, 10, -3, 100, 9, 2];
        let enum_dtype = {
            let categories = FrozenCategories::new(["z", "10", "9", "a"]).unwrap();
            DataType::Enum(categories.clone(), categories.mapping().clone())
        };
        let mut cases: Vec<(String, Series)> = vec![
            ("Int64".into(), Series::new("k".into(), ints)),
            (
                "Int8".into(),
                Series::new("k".into(), ints).cast(&DataType::Int8).unwrap(),
            ),
            (
                "UInt16".into(),
                Series::new("k".into(), ints.map(|v| v.unsigned_abs()))
                    .cast(&DataType::UInt16)
                    .unwrap(),
            ),
            (
                "Float64".into(),
                Series::new("k".into(), [1.5, -0.0, 0.0, 10.0, 2.5, 1.5, 9.0, 0.0]),
            ),
            (
                "Float32".into(),
                Series::new("k".into(), [1.5_f32, -0.0, 0.0, 10.0, 2.5, 1.5, 9.0, 0.1]),
            ),
            (
                "String".into(),
                Series::new("k".into(), ["b", "10", "a", "9", "b", "", "10", "é"]),
            ),
            (
                "Date".into(),
                Series::new("k".into(), ints.map(|v| v as i32))
                    .cast(&DataType::Date)
                    .unwrap(),
            ),
            (
                "Enum".into(),
                Series::new("k".into(), ["10", "z", "a", "10", "9", "z", "a", "9"])
                    .cast(&enum_dtype)
                    .unwrap(),
            ),
            (
                "Categorical".into(),
                Series::new("k".into(), ["b", "10", "a", "9", "b", "c", "10", "a"])
                    .cast(&DataType::from_categories(Categories::global()))
                    .unwrap(),
            ),
        ];
        for unit in [
            TimeUnit::Milliseconds,
            TimeUnit::Microseconds,
            TimeUnit::Nanoseconds,
        ] {
            cases.push((
                format!("Datetime({unit:?})"),
                Series::new("k".into(), ints)
                    .cast(&DataType::Datetime(unit, None))
                    .unwrap(),
            ));
        }
        for (label, series) in cases {
            let df = frame_of(series);
            let Ok(expected) = extract_time_keys_via_row_strings(&df, "k") else {
                panic!("{label}: row string path failed");
            };
            let Ok(actual) = extract_time_keys(&df, "k", SRC) else {
                panic!("{label}: extract_time_keys failed");
            };
            assert_eq!(actual, expected, "{label}");
            assert_eq!(actual.periods(), expected.periods(), "{label}");
            assert_eq!(actual.len(), expected.len(), "{label}");
        }
    }

    fn frame_of(series: Series) -> DataFrame {
        DataFrame::new(series.len(), vec![series.into()]).unwrap()
    }

    fn periods_of(df: &DataFrame) -> Vec<String> {
        extract_time_keys(df, "t", SRC).unwrap().periods().to_vec()
    }

    #[test]
    fn extract_time_keys_orders_integers_numerically() {
        let df = df!("t" => [10i64, 2, 9, -3, 2]).unwrap();

        assert_eq!(periods_of(&df), ["-3", "2", "9", "10"]);
    }

    #[test]
    fn extract_time_keys_orders_every_integer_dtype_numerically() {
        for dtype in [
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::Int128,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
        ] {
            let df = frame_of(
                Series::new("t".into(), [10i64, 2, 9, 2])
                    .cast(&dtype)
                    .unwrap(),
            );

            assert_eq!(periods_of(&df), ["2", "9", "10"], "{dtype}");
        }
    }

    #[test]
    fn extract_time_keys_orders_floats_numerically() {
        let df = df!("t" => [10.5, 2.0, -0.5, 9.25, 2.0]).unwrap();

        assert_eq!(periods_of(&df), ["-0.5", "2.0", "9.25", "10.5"]);
    }

    #[test]
    fn extract_time_keys_orders_dates_and_datetimes_chronologically() {
        let date_df = frame_of(
            Series::new("t".into(), [19_000i32, 18_000, 20_000])
                .cast(&DataType::Date)
                .unwrap(),
        );
        let datetime_df = frame_of(
            Series::new("t".into(), [3_000i64, 1_000, 2_000])
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .unwrap(),
        );

        let dates = periods_of(&date_df);
        let datetimes = periods_of(&datetime_df);

        // 日付・日時の文字列表現は桁数が固定で、辞書順も時系列順に一致する。1つずつ小さい値が
        // 先頭に来ていれば、内部表現の整数の順に並んでいる。
        assert_eq!(dates.len(), 3);
        assert!(dates.windows(2).all(|w| w[0] < w[1]), "{dates:?}");
        assert_eq!(datetimes.len(), 3);
        assert!(datetimes.windows(2).all(|w| w[0] < w[1]), "{datetimes:?}");
    }

    #[test]
    fn extract_time_keys_orders_strings_lexicographically() {
        let df = df!("t" => ["b", "10", "a", "9"]).unwrap();

        assert_eq!(periods_of(&df), ["10", "9", "a", "b"]);
    }

    #[test]
    fn extract_time_keys_rejects_unusable_columns_like_the_key_column() {
        let df = df!("t" => [1.0, f64::NAN]).unwrap();
        assert!(extract_time_keys(&df, "t", SRC).is_err());
        let df = df!("t" => [true, false]).unwrap();
        assert!(extract_time_keys(&df, "t", SRC).is_err());
        assert!(extract_time_keys(&df, "missing", SRC).is_err());
    }
}
