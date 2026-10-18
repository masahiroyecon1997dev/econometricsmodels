//! `panel`系統（FE/RE）で共有するユーティリティ。
//!
//! `.claude/rules/rust-style.md`「ファイル・ディレクトリ構成」: 系統内で共有するロジックは
//! `<系統>/common.rs`に置く（`engine_pybind/src/iv/common.rs`と同じ位置づけ）。
//!
//! `PanelError`の`Common`バリアント（`engine::shared::error::CommonError`）は`crate::shared::errors::
//! common_error_to_pyerr`に委譲する（系統ごとに同じ判定ロジックを重複させない）。

use engine::panel::common::PanelError;
use pyo3::{PyErr, PyResult};

use crate::linear::common::least_squares_error_is_computation_error;
use crate::shared::errors::{
    common_error_to_pyerr, legacy_computation_error, legacy_validation_error,
};
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

    /// 2-wayの効果の呼び名（`dk_time`が必須と伝えるメッセージ用）。
    fn effects_word(self) -> &'static str {
        match self {
            Self::Fe => "fixed",
            Self::Re => "random",
        }
    }

    /// `hc0`が非対応である理由（参照実装の事情）。
    fn hc0_reason(self) -> &'static str {
        match self {
            Self::Fe => "neither linearmodels nor fixest offer HC0 for panel/FE regressions",
            Self::Re => "no reference implementation offers HC0 for panel/RE regressions",
        }
    }
}

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
        "hc0" => Err(legacy_validation_error(format!(
            "cov_type='hc0' is not supported for {} ({}); use 'hc1', 'hc2', or 'hc3' instead",
            model.name(),
            model.hc0_reason()
        ))),
        other => Err(legacy_validation_error(format!(
            "unknown cov_type: '{other}'. Expected one of 'classical', 'hc1' through \
             'hc3', 'cluster', or 'dk'"
        ))),
    }
}

/// DKの時点列`dk_time`が指定されていることを確かめ、その列名を返す。
///
/// DKの時点列は`dk_time`で明示する（必須）。2-wayの`time`（固定効果・ランダム効果の
/// 構造）を暗黙に借用すると、意図と違う列が選ばれても気づけないため。
///
/// # Errors
/// `dk_time`が未指定なら`ValidationError`。
pub(crate) fn require_dk_time(dk_time: Option<&str>, model: PanelModel) -> PyResult<&str> {
    dk_time.ok_or_else(|| {
        legacy_validation_error(format!(
            "cov_type='dk' requires the `dk_time` option: the column that defines the time \
             periods of the Driscoll-Kraay estimator (it is not taken from `time`, which \
             only sets the two-way {} effects)",
            model.effects_word()
        ))
    })
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
    model: PanelModel,
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
        require_dk_time(dk_time, model)?;
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
/// バリアントの分類方針:
/// - `Common`: `common_error_to_pyerr`へ委譲。
/// - FE/RE固有のバリデーションエラー（`IdentifierDimensionMismatch`・
///   `InsufficientDegreesOfFreedom`・`SingletonGroup`・`UnbalancedPanelForTwoWay`・
///   `ZeroVarianceAfterDemeaning`・`TwoWayRequiresTime`・
///   `InvalidDkBandwidth`・`InsufficientDkPeriods`・`InsufficientDkPeriodsForInference`）は
///   いずれも入力・オプションの不正なので`ValidationError`（`InsufficientDkPeriods`・
///   `InsufficientDkPeriodsForInference`は追加。`InvalidDkBandwidth`と同じ「DKのバンド幅・
///   時点数に関する入力不正」という分類のため同じ`match`アームに含める。後者は
///   クラスター版の`InsufficientClustersForInference`が`ValidationError`なのと揃える）。
///   `DegenerateDkTwoPeriods`・`DegenerateClusterTwoGroups`も入力（パネル構造と時点・
///   クラスター列）だけから判定できる退化のため同じアームに含める。
/// - `WithinRegressionFailed`・`FTestFailed`・`BetweenRegressionFailed`・
///   `QuasiDemeanedRegressionFailed`・`HausmanTestFailed`: 委譲先の`LeastSquaresError`の分類基準
///   （`least_squares_error_is_computation_error`）にそのまま従う。`IvError::
///   SecondStageFailed`と同じ扱い。Pythonに渡すメッセージは`source.to_string()`ではなく
///   `PanelError`自身の`to_string()`（「within変換後の推定で失敗した」等の文脈を含む）を
///   使うため、`least_squares_error_to_pyerr`自体は呼ばない。`BetweenRegressionFailed`
///   （RE）・`QuasiDemeanedRegressionFailed`（RE）は対象が
///   FEのwithin回帰と異なるだけで、分類ロジックは同じため同じ`match`アームでまとめる。
///
/// 初期実装の時点ではFE/REの`fit()`本体が未実装で、この関数は`#[cfg(test)] mod tests`
/// からしか呼び出されなかった。当初は`#[expect(dead_code, ...)]`（`cargo build`では未到達で
/// 発火するが`cargo test`/`clippy --all-targets`ではテストから到達可能になり発火しない、
/// という非対称性を前提にした属性）を使っていたが、後の実装で`panel::fe::build_fe_input`
/// （同じく`#[cfg(test)] mod tests`からのみ呼ばれる）がこの関数を呼ぶようになったことで
/// `unfulfilled_lint_expectations`（`-D warnings`下でエラー）が発火し、`#[allow(dead_code)]`
/// （無条件抑制）に変更した経緯がある（`engine_pybind/src/iv/CLAUDE.md`「踏んだ罠」に
/// 記録済みの罠そのもの）。その後`panel::fe::fit`（`fit_fe`が`#[pymodule]`経由で
/// 呼ぶ本番経路）がこの関数を呼ぶようになったため、`#[allow(dead_code)]`は不要になった。
pub(crate) fn panel_error_to_pyerr(err: PanelError) -> PyErr {
    let message = err.to_string();
    match err {
        PanelError::Common(common) => common_error_to_pyerr(common),
        PanelError::IdentifierDimensionMismatch { .. }
        | PanelError::InsufficientDegreesOfFreedom { .. }
        | PanelError::SingletonGroup { .. }
        | PanelError::UnbalancedPanelForTwoWay { .. }
        | PanelError::ZeroVarianceAfterDemeaning { .. }
        | PanelError::TwoWayRequiresTime
        | PanelError::InvalidDkBandwidth { .. }
        | PanelError::InsufficientDkPeriods { .. }
        | PanelError::InsufficientDkPeriodsForInference { .. }
        | PanelError::DegenerateDkTwoPeriods
        | PanelError::DegenerateClusterTwoGroups
        | PanelError::AmemiyaRequiresWithinVariation { .. } => legacy_validation_error(message),
        PanelError::NonPositiveIdiosyncraticVariance { .. } => legacy_computation_error(message),
        PanelError::WithinRegressionFailed { source }
        | PanelError::FTestFailed { source }
        | PanelError::BetweenRegressionFailed { source, .. }
        | PanelError::PooledRegressionFailed { source }
        | PanelError::QuasiDemeanedRegressionFailed { source }
        | PanelError::HausmanTestFailed { source } => {
            if least_squares_error_is_computation_error(&source) {
                legacy_computation_error(message)
            } else {
                legacy_validation_error(message)
            }
        }
    }
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
        let m = PanelModel::Fe;
        assert!(
            check_panel_cov_options(PanelCovKind::Classical, m, Some("g"), None, None).is_err()
        );
        assert!(check_panel_cov_options(PanelCovKind::Cluster, m, None, Some("t"), None).is_err());
        assert!(check_panel_cov_options(PanelCovKind::Cluster, m, None, None, Some(2)).is_err());
        assert!(check_panel_cov_options(PanelCovKind::Dk, m, Some("g"), Some("t"), None).is_err());
        assert!(check_panel_cov_options(PanelCovKind::Cluster, m, Some("g"), None, None).is_ok());
        // `Cluster`で`cluster`が未指定なのはここでは弾かない（entity列にフォールバックする）。
        assert!(check_panel_cov_options(PanelCovKind::Cluster, m, None, None, None).is_ok());
        assert!(check_panel_cov_options(PanelCovKind::Dk, m, None, Some("t"), Some(2)).is_ok());
    }

    #[test]
    fn check_panel_cov_options_rejects_every_option_for_kinds_that_use_none() {
        for kind in [
            PanelCovKind::Classical,
            PanelCovKind::Hc1,
            PanelCovKind::Hc2,
            PanelCovKind::Hc3,
        ] {
            for model in [PanelModel::Fe, PanelModel::Re] {
                assert!(check_panel_cov_options(kind, model, Some("g"), None, None).is_err());
                assert!(check_panel_cov_options(kind, model, None, Some("t"), None).is_err());
                assert!(check_panel_cov_options(kind, model, None, None, Some(2)).is_err());
                assert!(check_panel_cov_options(kind, model, None, None, None).is_ok());
            }
        }
    }

    #[test]
    fn check_panel_cov_options_requires_dk_time_for_dk() {
        for model in [PanelModel::Fe, PanelModel::Re] {
            assert!(check_panel_cov_options(PanelCovKind::Dk, model, None, None, None).is_err());
            assert!(
                check_panel_cov_options(PanelCovKind::Dk, model, None, Some("t"), None).is_ok()
            );
        }
    }

    #[test]
    fn require_dk_time_returns_column_name_or_errors() {
        assert!(matches!(
            require_dk_time(Some("t"), PanelModel::Re),
            Ok("t")
        ));
        assert!(require_dk_time(None, PanelModel::Re).is_err());
    }
}
