//! `nonlinear`系統（Logit/Probit/Tobit等）で共有するユーティリティ。
//!
//! `.claude/rules/rust-style.md`「ファイル・ディレクトリ構成」: 系統内で共有するロジックは
//! `<系統>/common.rs`に置く（`engine_pybind/src/linear/common.rs`と同じ位置づけ）。
//!
//! `MleError`の`Common`バリアント（`engine::error::CommonError`、`linear`系統の
//! `LeastSquaresError`と共有する6種のバリデーションエラー）は`crate::errors::
//! common_error_to_pyerr`に委譲する（系統ごとに同じ判定ロジックを重複させない）。

use engine::nonlinear::common::{CovType, MarginalEffectsAt, Method, MleError};
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use crate::column_extraction::extract_group_key_column;
use crate::errors::{ComputationError, ValidationError, common_error_to_pyerr};

/// `engine::nonlinear::common::MleError`をPython例外に変換する。
///
/// `MleError`（`engine`クレート）と`PyErr`（`pyo3`クレート）はどちらもこのクレートの
/// 外で定義された型のため、orphan rule（`impl`の対象は自クレート内で定義された
/// トレイトか型のどちらかを含む必要がある）により`impl From<MleError> for PyErr`は
/// 書けない。関数として実装し、呼び出し側で`.map_err(mle_error_to_pyerr)?`する
/// （`least_squares_error_to_pyerr`と同じ理由、`engine_pybind/src/linear/common.rs`参照）。
///
/// 対応表は`docs/planning/specs/nonlinear-implementation-notes.md`「エラー型: nonlinear系統で
/// 共有（MleError）」参照。
pub(crate) fn mle_error_to_pyerr(err: MleError) -> PyErr {
    match err {
        MleError::Common(common) => common_error_to_pyerr(common),
        MleError::InvalidMaxIter { .. }
        | MleError::InvalidTol { .. }
        | MleError::InvalidCensoringBounds { .. }
        | MleError::InvalidBinaryY { .. }
        | MleError::YOutOfCensoringBounds { .. }
        | MleError::NoUncensoredObservations { .. } => ValidationError::new_err(err.to_string()),
        MleError::NonConvergence { .. }
        | MleError::SingularHessian
        | MleError::SingularOpgMatrix
        | MleError::SingularDesignMatrix
        | MleError::SeparationSuspected { .. } => ComputationError::new_err(err.to_string()),
    }
}

/// `faer::Mat<f64>`を行指向の`Vec<Vec<f64>>`に変換する（`pred_table`のような小さい
/// 2次元行列をPython側にそのまま返すため）。`engine_pybind/src/linear/common.rs`の
/// `mat_to_vec`は列ベクトル（n×1）専用のため、任意の形状に対応するこちらを別途用意した。
pub(crate) fn mat_to_nested_vec(mat: &faer::Mat<f64>) -> Vec<Vec<f64>> {
    (0..mat.nrows())
        .map(|i| (0..mat.ncols()).map(|j| *mat.get(i, j)).collect())
        .collect()
}

/// `LogitResult::marginal_effects`/`ProbitResult::marginal_effects`共通の結果型。
/// 定数項は除外される（`engine`側の`MarginalEffects`のdocコメント参照）。フィールドの
/// 並びは`param_names`と対応する（`LogitResult`/`ProbitResult`本体と同じ規約）。
///
/// 元々`logit.rs`にLogit専用として定義していたが、`engine::nonlinear::common::
/// MarginalEffects`がLogit/Probitで既に共有されているのに合わせ、こちらも
/// `<系統>/common.rs`に移動した（`rust-style.md`「ファイル・ディレクトリ構成」参照）。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct MarginalEffectsResult {
    #[pyo3(get)]
    pub param_names: Vec<String>,
    #[pyo3(get)]
    pub dydx: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub z_stats: Vec<f64>,
    #[pyo3(get)]
    pub p_values: Vec<f64>,
    #[pyo3(get)]
    pub conf_lower: Vec<f64>,
    #[pyo3(get)]
    pub conf_upper: Vec<f64>,
}

/// `cov_type`文字列（大文字小文字を区別しない）を`engine::nonlinear::common::CovType`に
/// パースする。`cov_type="cluster"`のときのみ`cluster_col`で指定された列を
/// `extract_group_key_column`で抽出する（他のcov_typeでは無視する、OLSの
/// `cluster_col`/`time_col`の扱いと同じ方針）。
///
/// Logit/Probit/Tobit共通（Issue #308で対応: 元は`logit.rs`/`probit.rs`/`tobit.rs`に
/// バイト単位で完全一致するコードとして独立複製されていたが、`CovType`自体が
/// 元々3手法共有の型であるのに合わせてここに集約した）。
///
/// # Errors
/// - `cov_type`が既知の値のいずれでもない: `ValidationError`
///
/// `cluster_col`未指定自体はここでは`ValidationError`にせず、`groups=None`のまま
/// `engine`側の`CommonError::MissingClusterColumn`検証に委ねる（OLSの`fit()`と同じ役割分担）。
pub(crate) fn parse_cov_type(
    df: &DataFrame,
    cov_type_lower: &str,
    cluster_col: &Option<String>,
) -> PyResult<CovType> {
    match cov_type_lower {
        "classical" | "nonrobust" => Ok(CovType::Classical),
        "opg" => Ok(CovType::Opg),
        "hc0" => Ok(CovType::Hc0),
        "hc1" => Ok(CovType::Hc1),
        "cluster" => {
            let groups = cluster_col
                .as_ref()
                .map(|col_name| extract_group_key_column(df, col_name))
                .transpose()?;
            Ok(CovType::Cluster { groups })
        }
        other => Err(ValidationError::new_err(format!(
            "unknown cov_type: '{other}'. Expected one of 'classical' (or 'nonrobust'), \
             'opg', 'hc0', 'hc1', or 'cluster'"
        ))),
    }
}

/// `method`文字列（大文字小文字を区別しない）を`engine::nonlinear::common::Method`に
/// パースする。Logit/Probit/Tobit共通（`parse_cov_type`と同じ理由・同じIssue #308で
/// ここに集約）。
///
/// # Errors
/// `method`が既知の値のいずれでもない: `ValidationError`
pub(crate) fn parse_method(method_lower: &str) -> PyResult<Method> {
    match method_lower {
        "newton" => Ok(Method::Newton),
        "bfgs" => Ok(Method::Bfgs),
        "lbfgs" => Ok(Method::Lbfgs),
        other => Err(ValidationError::new_err(format!(
            "unknown method: '{other}'. Expected one of 'newton', 'bfgs', or 'lbfgs'"
        ))),
    }
}

/// `at`文字列（大文字小文字を区別しない）を`engine::nonlinear::common::MarginalEffectsAt`に
/// パースする。Logit/Probit共通（`marginal_effects`の受け口はいずれもこの関数を呼ぶ前に
/// `.to_lowercase()`する）。
///
/// # Errors
/// `at`が既知の値のいずれでもない: `ValidationError`
pub(crate) fn parse_marginal_effects_at(at_lower: &str) -> PyResult<MarginalEffectsAt> {
    match at_lower {
        "overall" => Ok(MarginalEffectsAt::Overall),
        "mean" => Ok(MarginalEffectsAt::Mean),
        "median" => Ok(MarginalEffectsAt::Median),
        other => Err(ValidationError::new_err(format!(
            "unknown at: '{other}'. Expected one of 'overall', 'mean', or 'median'"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat_to_nested_vec_preserves_row_major_order() {
        let mat = faer::Mat::from_fn(2, 3, |i, j| (i * 3 + j) as f64);
        assert_eq!(
            mat_to_nested_vec(&mat),
            vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]]
        );
    }

    #[test]
    fn mat_to_nested_vec_handles_single_row() {
        let mat = faer::Mat::from_fn(1, 2, |_, j| j as f64);
        assert_eq!(mat_to_nested_vec(&mat), vec![vec![0.0, 1.0]]);
    }

    // `parse_marginal_effects_at`自体は渡された文字列をそのまま照合する（`parse_cov_type`/
    // `parse_method`と同じ設計）。大文字小文字を区別しない処理は呼び出し側
    // （`LogitResult::marginal_effects`/`ProbitResult::marginal_effects`）が
    // `.to_lowercase()`してから渡すことで実現するため、ここでは小文字化済みの入力を渡す。
    #[test]
    fn parse_marginal_effects_at_accepts_known_lowercase_values() {
        assert!(matches!(
            parse_marginal_effects_at("overall"),
            Ok(MarginalEffectsAt::Overall)
        ));
        assert!(matches!(
            parse_marginal_effects_at("mean"),
            Ok(MarginalEffectsAt::Mean)
        ));
        assert!(matches!(
            parse_marginal_effects_at("median"),
            Ok(MarginalEffectsAt::Median)
        ));
    }

    #[test]
    fn parse_marginal_effects_at_returns_validation_error_for_unknown_value() {
        assert!(parse_marginal_effects_at("bogus").is_err());
    }
}
