//! `linear`系統（OLS/WLS等）で共有するユーティリティ。
//!
//! `.claude/rules/rust-style.md`「ファイル・ディレクトリ構成」: 系統内で共有するロジックは
//! `<系統>/common.rs`に置く。以前はOLSしかなく未作成だったが、WLSが`LeastSquaresError`の
//! エラー変換等をOLSと共有する形で実装されたため作成した（`docs/spec/wls-spec.md`「エラー型」）。
//!
//! `LeastSquaresError`は元々`OlsError`という名前だったが、OLS単体のエラー型ではなくWLSも
//! 含む`linear`系統共通のエラー型であることを名前に反映するため、`engine`側で
//! `engine::linear::common::LeastSquaresError`に改名・移動した。
//!
//! `LeastSquaresError`の`Common`バリアント（`engine::shared::error::CommonError`、nonlinear系統の
//! `MleError`と共有する6種のバリデーションエラー）は`crate::shared::errors::common_error_to_pyerr`
//! に委譲する（系統ごとに同じ判定ロジックを重複させない）。

use engine::linear::common::LeastSquaresError;
use engine::linear::ols::CovType as EngineCovType;
use polars::prelude::DataFrame;
use pyo3::{PyErr, PyResult};

use crate::shared::column_extraction::{extract_optional_cluster_groups, extract_time_order_ranks};
use crate::shared::errors::{
    common_error_to_pyerr, legacy_computation_error, legacy_validation_error,
};
use crate::shared::validation::reject_unused_option;

/// `engine::linear::common::LeastSquaresError`をPython例外に変換する。
///
/// `LeastSquaresError`（`engine`クレート）と`PyErr`（`pyo3`クレート）はどちらもこのクレートの
/// 外で定義された型のため、orphan rule（`impl`の対象は自クレート内で定義された
/// トレイトか型のどちらかを含む必要がある）により`impl From<LeastSquaresError> for PyErr`は
/// 書けない。関数として実装し、呼び出し側で`.map_err(least_squares_error_to_pyerr)?`する。
///
/// 対応表は`docs/spec/ols-spec.md`「engine/engine_pybind間のデータ受け渡し・エラー変換」参照。
pub(crate) fn least_squares_error_to_pyerr(err: LeastSquaresError) -> PyErr {
    match err {
        LeastSquaresError::Common(common) => common_error_to_pyerr(common),
        LeastSquaresError::WeightDimensionMismatch { .. }
        | LeastSquaresError::NonPositiveWeight { .. }
        | LeastSquaresError::InvalidNlags { .. }
        | LeastSquaresError::InsufficientObservationsForAuxRegression { .. } => {
            legacy_validation_error(err.to_string())
        }
        LeastSquaresError::SingularMatrix => legacy_computation_error(err.to_string()),
    }
}

/// `LeastSquaresError`が`ComputationError`（計算過程で発覚した問題）に分類されるか。
/// `ValidationError`（入力・パラメータが不正）との判定基準は[`least_squares_error_to_pyerr`]と
/// 同一にする（`SingularMatrix`または`CommonError::ComputationFailed`のみ`true`）。
///
/// `least_squares_error_to_pyerr`自体を呼ばずこの判定だけを独立させているのは、
/// `IvError::FirstStageFailed`/`SecondStageFailed`（`engine_pybind/src/iv/common.rs`）が
/// 内側の`LeastSquaresError`から分類だけを借りつつ、Pythonに渡すメッセージは
/// `IvError`自身の`to_string()`（文脈付き）を使いたいため（`source.to_string()`を
/// 使う`least_squares_error_to_pyerr`をそのまま呼ぶとメッセージの文脈が失われる）。
pub(crate) fn least_squares_error_is_computation_error(err: &LeastSquaresError) -> bool {
    matches!(
        err,
        LeastSquaresError::SingularMatrix
            | LeastSquaresError::Common(engine::shared::error::CommonError::ComputationFailed(_))
    )
}

/// `cov_type`の種類。列名や`hac_lags`を持たない、文字列だけから決まる値。
///
/// `cov_type`の検査は、DataFrameを見ない順に3つに分ける
/// （`docs/guide/validation.md`「Order in which errors are reported」）。
///
/// - 値: [`parse_cov_kind`]が文字列を`CovKind`にする（未知の値は`ValidationError`）
/// - 組合せ: [`check_cov_options`]が`cluster`/`hac_lags`/`hac_time`と`CovKind`の矛盾
///   （使われないオプションの指定、必須の`hac_time`の欠落）を検査する
/// - 列: [`resolve_cov_type`]が列を抽出して`engine`の`CovType`にする
///
/// 呼び出し側は、他のオプションの値の検査をすべて終えてから「組合せ」を、ロール構造・
/// `y`/`x`の列の検査を終えてから「列」を呼ぶ。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CovKind {
    Classical,
    Hc0,
    Hc1,
    Hc2,
    Hc3,
    Hac,
    Cluster,
}

impl CovKind {
    /// `*Result.cov_type`に格納する正規形の名前。
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Classical => "classical",
            Self::Hc0 => "hc0",
            Self::Hc1 => "hc1",
            Self::Hc2 => "hc2",
            Self::Hc3 => "hc3",
            Self::Hac => "hac",
            Self::Cluster => "cluster",
        }
    }
}

/// `cov_type`文字列（大文字小文字を区別しない）を[`CovKind`]にする。DataFrameは見ない。
///
/// 受け付ける綴りは正規形の大文字小文字違いだけで、別名は無い。
///
/// # Errors
/// 既知の値のいずれでもない場合は`ValidationError`。
pub(crate) fn parse_cov_kind(cov_type: &str) -> PyResult<CovKind> {
    match cov_type.to_lowercase().as_str() {
        "classical" => Ok(CovKind::Classical),
        "hc0" => Ok(CovKind::Hc0),
        "hc1" => Ok(CovKind::Hc1),
        "hc2" => Ok(CovKind::Hc2),
        "hc3" => Ok(CovKind::Hc3),
        "hac" => Ok(CovKind::Hac),
        "cluster" => Ok(CovKind::Cluster),
        other => Err(legacy_validation_error(format!(
            "unknown cov_type: '{other}'. Expected one of 'classical', 'hc0' through \
             'hc3', 'hac', or 'cluster'"
        ))),
    }
}

/// `cluster`/`hac_lags`/`hac_time`が`kind`と矛盾しないことを検査する。DataFrameは見ない。
///
/// `cluster`は`Cluster`、`hac_lags`/`hac_time`は`Hac`でなければ使われない。その状態で
/// 指定されていれば`ValidationError`にする（黙って無視すると`cov_type="cluster"`の
/// 書き忘れに気づけないため）。`Hac`では`hac_time`が必須（行順を時間順とみなす暗黙の既定は
/// 置かない、[`require_hac_time`]）。使われないオプションの検査が先で、必須の欠落が後。
///
/// IVは`gmm_weight_type`も同じ値を使い`cov_type`単独では判定できないため、
/// この関数ではなく`iv::common::check_iv_options`を使う。
///
/// # Errors
/// 使われないオプションが指定されている場合、または`Hac`で`hac_time`が未指定の場合は
/// `ValidationError`。
pub(crate) fn check_cov_options(
    kind: CovKind,
    cluster: Option<&str>,
    hac_lags: Option<i64>,
    hac_time: Option<&str>,
) -> PyResult<()> {
    let is_hac = kind == CovKind::Hac;
    reject_unused_option(
        "cluster",
        cluster.is_some(),
        kind == CovKind::Cluster,
        "cov_type=\"cluster\"",
    )?;
    reject_unused_option("hac_lags", hac_lags.is_some(), is_hac, "cov_type=\"hac\"")?;
    reject_unused_option("hac_time", hac_time.is_some(), is_hac, "cov_type=\"hac\"")?;
    if is_hac {
        require_hac_time(hac_time, "cov_type")?;
    }
    Ok(())
}

/// HAC（`cov_type="hac"`・IV GMMの`gmm_weight_type="hac"`）の時間順序列`hac_time`が
/// 指定されていることを確かめ、その列名を返す。`setting`は`hac_time`を要求している設定名
/// （エラーメッセージ用）。
///
/// 行順を時間順とみなす暗黙の既定は置かない。データが時間順に並んでいなくてもエラーに
/// ならず、時系列順のHACに見える誤った結果が黙って返るため、時間順は常に列で明示させる。
///
/// # Errors
/// `hac_time`が未指定なら`ValidationError`。
pub(crate) fn require_hac_time<'a>(hac_time: Option<&'a str>, setting: &str) -> PyResult<&'a str> {
    hac_time.ok_or_else(|| {
        legacy_validation_error(format!(
            "{setting}='hac' requires the `hac_time` option: the column that gives the time \
             order of the observations (the row order of the data is not assumed to be the \
             time order; add an explicit index column such as `df.with_row_index(\"t\")` if \
             the rows are already in time order)"
        ))
    })
}

/// [`CovKind`]に対応する列（`cluster`・`hac_time`）をDataFrameから抽出して、
/// `engine::linear::ols::CovType`を組み立てる。該当する`CovKind`の分岐でのみ列抽出を行う。
///
/// 使われないオプションの検査は[`check_cov_options`]の責務で、ここでは行わない。
///
/// # Errors
/// 列の抽出時に発覚する問題（列が存在しない、dtypeが不正、欠損値、`hac_time`の同値等）は
/// `column_extraction`の責務で`ValidationError`。`Hac`で`hac_time`が未指定の場合も
/// `ValidationError`（`check_cov_options`を通していれば起きない）。
pub(crate) fn resolve_cov_type(
    df: &DataFrame,
    kind: CovKind,
    cluster: Option<&str>,
    hac_lags: Option<i64>,
    hac_time: Option<&str>,
) -> PyResult<EngineCovType> {
    Ok(match kind {
        CovKind::Classical => EngineCovType::Classical,
        CovKind::Hc0 => EngineCovType::Hc0,
        CovKind::Hc1 => EngineCovType::Hc1,
        CovKind::Hc2 => EngineCovType::Hc2,
        CovKind::Hc3 => EngineCovType::Hc3,
        CovKind::Hac => {
            let col_name = require_hac_time(hac_time, "cov_type")?;
            EngineCovType::Hac {
                lags: hac_lags,
                time_order: extract_time_order_ranks(df, col_name)?,
            }
        }
        CovKind::Cluster => EngineCovType::Cluster {
            groups: extract_optional_cluster_groups(df, cluster)?,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    /// `unwrap()`/`expect()`は使わない：`PyErr`の`Debug`実装（`unwrap()`失敗時の
    /// panicメッセージ生成に使われる）はGIL取得を要求し、GIL未初期化のこのテスト
    /// 環境では二重パニック（テストバイナリ全体がSIGABRTでクラッシュし、他のテスト
    /// 結果も失われる）を起こす（`validation.rs`と同じ制約、`nonlinear/CLAUDE.md`
    /// 「テストの制約」参照。`let-else`のpanicメッセージ自体はErr値のDebug/Displayに
    /// 触れないため安全）。
    #[test]
    fn parse_cov_kind_is_case_insensitive_and_has_canonical_names() {
        for (input, expected) in [
            ("classical", "classical"),
            ("CLASSICAL", "classical"),
            ("Classical", "classical"),
            ("HC0", "hc0"),
            ("Hc1", "hc1"),
            ("HC2", "hc2"),
            ("hc3", "hc3"),
            ("CLUSTER", "cluster"),
            ("Hac", "hac"),
        ] {
            let Ok(kind) = parse_cov_kind(input) else {
                panic!("expected Ok for input={input}");
            };
            assert_eq!(kind.name(), expected, "input={input}");
        }
    }

    #[test]
    fn parse_cov_kind_returns_validation_error_for_unknown_value() {
        assert!(parse_cov_kind("bogus").is_err());
        // 別名は無い（`"nonrobust"`は`"classical"`の別名として受け付けない）。
        assert!(parse_cov_kind("nonrobust").is_err());
    }

    #[test]
    fn check_cov_options_rejects_options_the_kind_does_not_use() {
        assert!(check_cov_options(CovKind::Classical, Some("g"), None, None).is_err());
        assert!(check_cov_options(CovKind::Cluster, None, Some(1), None).is_err());
        assert!(check_cov_options(CovKind::Cluster, None, None, Some("t")).is_err());
        assert!(check_cov_options(CovKind::Hac, Some("g"), None, Some("t")).is_err());
        assert!(check_cov_options(CovKind::Cluster, Some("g"), None, None).is_ok());
        // `Cluster`で`cluster`が未指定なのはここでは弾かない（engineの検証に委ねる）。
        assert!(check_cov_options(CovKind::Cluster, None, None, None).is_ok());
    }

    #[test]
    fn check_cov_options_requires_hac_time_for_hac() {
        // 行順を時間順とみなす暗黙の既定は置かない(`require_hac_time`参照)。
        assert!(check_cov_options(CovKind::Hac, None, None, None).is_err());
        assert!(check_cov_options(CovKind::Hac, None, Some(1), Some("t")).is_ok());
    }

    #[test]
    fn resolve_cov_type_passes_time_order_ranks_for_hac() {
        // engineの`time_order`は必須。pybindは`hac_time`の順位を渡す。
        let df = df!("t" => [30.0, 10.0, 20.0]).unwrap();
        let Ok(cov_type) = resolve_cov_type(&df, CovKind::Hac, None, Some(1), Some("t")) else {
            panic!("expected Ok");
        };
        assert!(matches!(
            cov_type,
            EngineCovType::Hac { lags: Some(1), time_order: ref ranks }
                if *ranks == vec![2.0, 0.0, 1.0]
        ));
    }

    #[test]
    fn resolve_cov_type_returns_error_for_missing_column() {
        let df = df!("t" => [1.0, 2.0, 3.0]).unwrap();
        assert!(resolve_cov_type(&df, CovKind::Cluster, Some("nope"), None, None).is_err());
        assert!(resolve_cov_type(&df, CovKind::Hac, None, None, Some("nope")).is_err());
    }

    #[test]
    fn resolve_cov_type_leaves_cluster_groups_unset_when_cluster_is_omitted() {
        // 未指定はここでは弾かず、engineの`MissingClusterColumn`検証に委ねる。
        let df = df!("g" => [1.0, 2.0, 3.0]).unwrap();
        let Ok(resolved) = resolve_cov_type(&df, CovKind::Cluster, None, None, None) else {
            panic!("expected Ok");
        };
        assert!(matches!(resolved, EngineCovType::Cluster { groups: None }));
        let Ok(resolved) = resolve_cov_type(&df, CovKind::Cluster, Some("g"), None, None) else {
            panic!("expected Ok");
        };
        assert!(matches!(
            resolved,
            EngineCovType::Cluster { groups: Some(_) }
        ));
    }

    #[test]
    fn require_hac_time_returns_column_name_or_errors() {
        assert!(matches!(require_hac_time(Some("t"), "cov_type"), Ok("t")));
        assert!(require_hac_time(None, "cov_type").is_err());
    }
}
