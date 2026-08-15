//! `linear`系統（OLS/WLS等）で共有するユーティリティ。
//!
//! `.claude/rules/rust-style.md`「ファイル・ディレクトリ構成」: 系統内で共有するロジックは
//! `<系統>/common.rs`に置く。以前はOLSしかなく未作成だったが、WLSが`LeastSquaresError`の
//! エラー変換・`Mat<f64>`→`Vec<f64>`変換の両方をOLSと共有する形で実装されたため作成した
//! （`docs/spec/wls-spec.md`「エラー型」）。
//!
//! `LeastSquaresError`は元々`OlsError`という名前だったが、OLS単体のエラー型ではなくWLSも
//! 含む`linear`系統共通のエラー型であることを名前に反映するため、`engine`側で
//! `engine::linear::common::LeastSquaresError`に改名・移動した。
//!
//! `LeastSquaresError`の`Common`バリアント（`engine::error::CommonError`、nonlinear系統の
//! `MleError`と共有する6種のバリデーションエラー）は`crate::errors::common_error_to_pyerr`
//! に委譲する（系統ごとに同じ判定ロジックを重複させない）。

use engine::linear::common::LeastSquaresError;
use engine::linear::ols::CovType as EngineCovType;
use polars::prelude::DataFrame;
use pyo3::{PyErr, PyResult};

use super::ols::OLSOptions;
use crate::column_extraction::{extract_f64_column, extract_group_key_column};
use crate::errors::{ComputationError, ValidationError, common_error_to_pyerr};

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
        | LeastSquaresError::InvalidHacLags { .. } => ValidationError::new_err(err.to_string()),
        LeastSquaresError::SingularMatrix => ComputationError::new_err(err.to_string()),
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
            | LeastSquaresError::Common(engine::error::CommonError::ComputationFailed(_))
    )
}

/// `faer::Mat<f64>`（n×1またはk×1の列ベクトル）を`Vec<f64>`に変換する。
pub(crate) fn mat_to_vec(mat: &faer::Mat<f64>) -> Vec<f64> {
    (0..mat.nrows()).map(|i| *mat.get(i, 0)).collect()
}

/// `OLSOptions.cov_type`をパースし、該当する`cov_type`のときのみ`cluster_col`/`time_col`を
/// 抽出したうえで`engine::linear::ols::CovType`を組み立てる（OLS/WLS共通、
/// `docs/spec/ols-spec.md`「標準誤差」参照）。
///
/// `cluster_col`/`time_col`が指定されていても、`cov_type`がcluster/hacでなければ無視する。
/// 戻り値の2つ目は`*Result.cov_type`にそのまま格納する小文字化済み文字列
/// （呼び出し側で二重に`to_lowercase()`しないよう、ここでまとめて返す）。
///
/// `common.rs`が`super::ols::OLSOptions`という特定モジュールの型に依存する点は、
/// `mat_to_vec`等の汎用型のみを扱うヘルパーとは毛色が異なる。これは`WLSOptions`という
/// 専用型を新設せずWLSが`OLSOptions`をそのまま再利用する既存方針（`docs/spec/wls-spec.md`
/// 「API引数」）を踏まえた判断で、`OLSOptions`は実質的に`linear`系統共通のオプション型
/// という位置づけのため許容する。将来GLS等で`cov_type`の型が分岐する場合は、この関数を
/// 無理に拡張せず素直に系統・手法ごとの実装に切り替えること。
///
/// # Errors
/// `cov_type`の文字列が既知の値のいずれでもない場合は`ValidationError`。それ以外
/// （列の抽出時に発覚する問題等）は`column_extraction`の責務で`ValidationError`。
pub(crate) fn parse_cov_type(
    df: &DataFrame,
    options: &OLSOptions,
) -> PyResult<(EngineCovType, String)> {
    let cov_type_lower = options.cov_type.to_lowercase();

    let cluster_groups = if cov_type_lower == "cluster" {
        options
            .cluster_col
            .as_ref()
            .map(|col_name| extract_group_key_column(df, col_name))
            .transpose()?
    } else {
        None
    };

    let time_order = if cov_type_lower == "hac" {
        options
            .time_col
            .as_ref()
            .map(|col_name| extract_f64_column(df, col_name))
            .transpose()?
    } else {
        None
    };

    let cov_type = match cov_type_lower.as_str() {
        "classical" | "nonrobust" => EngineCovType::Classical,
        "hc0" => EngineCovType::Hc0,
        "hc1" => EngineCovType::Hc1,
        "hc2" => EngineCovType::Hc2,
        "hc3" => EngineCovType::Hc3,
        "hac" => EngineCovType::Hac {
            lags: options.hac_lags,
            time_order,
        },
        "cluster" => EngineCovType::Cluster {
            groups: cluster_groups,
        },
        other => {
            return Err(ValidationError::new_err(format!(
                "unknown cov_type: '{other}'. Expected one of 'classical', 'hc0' through \
                 'hc3', 'hac', or 'cluster'"
            )));
        }
    };

    Ok((cov_type, cov_type_lower))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `cov_type`以外はデフォルト値の`OLSOptions`を返す。cluster/hac以外のケースでは
    /// `cluster_col`/`time_col`が抽出されないため、`df`は空でよい。
    fn options(cov_type: &str) -> OLSOptions {
        OLSOptions {
            cov_type: cov_type.to_string(),
            include_intercept: true,
            confidence_level: 0.95,
            cluster_col: None,
            hac_lags: None,
            time_col: None,
        }
    }

    /// `unwrap()`/`expect()`は使わない：`PyErr`の`Debug`実装（`unwrap()`失敗時の
    /// panicメッセージ生成に使われる）はGIL取得を要求し、GIL未初期化のこのテスト
    /// 環境では二重パニック（テストバイナリ全体がSIGABRTでクラッシュし、他のテスト
    /// 結果も失われる）を起こす（`validation.rs`と同じ制約、`nonlinear/CLAUDE.md`
    /// 「テストの制約」参照。`let-else`のpanicメッセージ自体はErr値のDebug/Displayに
    /// 触れないため安全）。
    #[test]
    fn parse_cov_type_is_case_insensitive() {
        let df = DataFrame::empty();
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
            let Ok((_, normalized)) = parse_cov_type(&df, &options(input)) else {
                panic!("expected Ok for input={input}");
            };
            assert_eq!(normalized, expected, "input={input}");
        }
    }

    #[test]
    fn parse_cov_type_accepts_nonrobust_as_classical_alias() {
        let df = DataFrame::empty();
        for input in ["nonrobust", "NONROBUST", "NonRobust"] {
            let Ok((cov_type, normalized)) = parse_cov_type(&df, &options(input)) else {
                panic!("expected Ok for input={input}");
            };
            assert!(
                matches!(cov_type, EngineCovType::Classical),
                "input={input}"
            );
            // `parse_cov_type`のdocコメント通り、`*Result.cov_type`にはエイリアスでは
            // なく小文字化した入力文字列（"nonrobust"）がそのまま格納される
            // （"classical"に正規化はしない）。
            assert_eq!(normalized, "nonrobust", "input={input}");
        }
    }

    #[test]
    fn parse_cov_type_returns_validation_error_for_unknown_value() {
        let df = DataFrame::empty();
        assert!(parse_cov_type(&df, &options("bogus")).is_err());
    }
}
