//! `panel`系統（FE/RE）で共有するユーティリティ。
//!
//! `.claude/rules/rust-style.md`「ファイル・ディレクトリ構成」: 系統内で共有するロジックは
//! `<系統>/common.rs`に置く（`engine_pybind/src/iv/common.rs`と同じ位置づけ）。
//!
//! `PanelError`の`Common`バリアント（`engine::shared::error::CommonError`）は`crate::shared::errors::
//! common_error_to_pyerr`に委譲する（系統ごとに同じ判定ロジックを重複させない）。

use engine::panel::common::PanelError;
use engine::shared::error_catalog::{ErrorCode, ParamValue};
use pyo3::{PyErr, PyResult};

use crate::shared::errors::{to_pyerr, unknown_option_value, user_error};
use crate::shared::validation::{RoleValue, reject_unused_option, validate_no_duplicate_roles};

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
const PANEL_COV_TYPES: &[&str] = &["classical", "hc1", "hc2", "hc3", "cluster", "dk"];

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
    match cov_type.to_lowercase().as_str() {
        "classical" => Ok(PanelCovKind::Classical),
        "hc1" => Ok(PanelCovKind::Hc1),
        "hc2" => Ok(PanelCovKind::Hc2),
        "hc3" => Ok(PanelCovKind::Hc3),
        "cluster" => Ok(PanelCovKind::Cluster),
        "dk" => Ok(PanelCovKind::Dk),
        "hc0" => Err(user_error(
            ErrorCode::CovTypeNotAvailableForModel,
            vec![
                ("value", ParamValue::from(cov_type)),
                ("model", ParamValue::from(model.name())),
                ("allowed", ParamValue::from(PANEL_COV_TYPES.to_vec())),
            ],
        )),
        _ => Err(unknown_option_value("cov_type", cov_type, PANEL_COV_TYPES)),
    }
}

/// DKの時点列`dk_time`が指定されていることを確かめ、その列名を返す。
///
/// DKの時点列は`dk_time`で明示する（必須）。2-wayの`time`（固定効果・ランダム効果の
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

/// `dk_time`（Driscoll-Kraay HACの時点列、FE/RE共通）が`y`・`entity`と同じ列でないことを
/// 検証する。
///
/// `dk_time`が`y`や`entity`と同じ列だと、時点が数値の`y`の値や個体IDそのものになり、
/// DKの時点構造が意味を成さない（`dk_time == entity`は全個体が単一時点ずつになる退化入力）。
/// 一方、`x`との重複は許可する（年トレンドを説明変数に入れつつDKの時点にも使う、等は
/// 正当な使い方）。FEの`time`（2-way固定効果の時間次元）との重複も許可する（固定効果と
/// DKで同じ時間粒度を使うのが典型）。そのため`validate_no_duplicate_roles`に渡す
/// ロールを`y`・`entity`・`dk_time`だけに絞り、`x`・`time`は含めない。
pub(crate) fn validate_dk_time_role(y: &str, entity: &str, dk_time: Option<&str>) -> PyResult<()> {
    let Some(dk_time) = dk_time else {
        return Ok(());
    };
    validate_no_duplicate_roles(&[
        ("y", RoleValue::Single(y)),
        ("entity", RoleValue::Single(entity)),
        ("dk_time", RoleValue::Single(dk_time)),
    ])
}

/// `engine::panel::common::PanelError`をPython例外に変換する。
///
/// `PanelError`（`engine`クレート）と`PyErr`（`pyo3`クレート）はどちらもこのクレートの外で
/// 定義された型のため、orphan ruleにより`impl From<PanelError> for PyErr`は書けない。関数
/// として実装し、呼び出し側で`.map_err(panel_error_to_pyerr)?`する（`iv_error_to_pyerr`と
/// 同じ理由、`engine_pybind/src/iv/common.rs`参照）。
///
/// `PanelError`の各バリアントは`ErrorInfo::report()`が`code`・`params`に変換する。
/// `WithinRegressionFailed`等の回帰の包みは原因の`code`に`stage`を付け、`ValidationError`/
/// `ComputationError`の別もカタログの`code`で決まる。
pub(crate) fn panel_error_to_pyerr(err: PanelError) -> PyErr {
    to_pyerr(&err)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // `Cluster`で`cluster`が未指定なのはここでは弾かない（entity列にフォールバックする）。
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
}
