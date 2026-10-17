//! `nonlinear`系統（Logit/Probit/Tobit等）で共有するユーティリティ。
//!
//! `.claude/rules/rust-style.md`「ファイル・ディレクトリ構成」: 系統内で共有するロジックは
//! `<系統>/common.rs`に置く（`engine_pybind/src/linear/common.rs`と同じ位置づけ）。
//!
//! `MleError`の`Common`バリアント（`engine::shared::error::CommonError`、`linear`系統の
//! `LeastSquaresError`と共有する6種のバリデーションエラー）は`crate::shared::errors::
//! common_error_to_pyerr`に委譲する（系統ごとに同じ判定ロジックを重複させない）。

use engine::nonlinear::common::{CovType, MarginalEffectsAt, MleError, SolverType};
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use crate::shared::column_extraction::extract_optional_cluster_groups;
use crate::shared::errors::{ComputationError, ValidationError, common_error_to_pyerr};
use crate::shared::validation::reject_unused_option;

/// `engine::nonlinear::common::MleError`をPython例外に変換する。
///
/// `MleError`（`engine`クレート）と`PyErr`（`pyo3`クレート）はどちらもこのクレートの
/// 外で定義された型のため、orphan rule（`impl`の対象は自クレート内で定義された
/// トレイトか型のどちらかを含む必要がある）により`impl From<MleError> for PyErr`は
/// 書けない。関数として実装し、呼び出し側で`.map_err(mle_error_to_pyerr)?`する
/// （`least_squares_error_to_pyerr`と同じ理由、`engine_pybind/src/linear/common.rs`参照）。
///
/// 対応表は`docs/spec/nonlinear-common.md`2章「エラー型（`MleError`、共有）」参照。
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
        | MleError::SeparationDetected
        | MleError::UncensoredIntervalTooNarrow { .. }
        | MleError::EvaluationBudgetExceeded { .. } => ComputationError::new_err(err.to_string()),
    }
}

/// `faer::Mat<f64>`を行指向の`Vec<Vec<f64>>`に変換する（`pred_table`のような小さい
/// 2次元行列をPython側にそのまま返すため）。`engine_pybind/src/shared/mat_conversion.rs`の
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
    pub effect: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub test_stats: Vec<f64>,
    #[pyo3(get)]
    pub p_values: Vec<f64>,
    #[pyo3(get)]
    pub conf_lower: Vec<f64>,
    #[pyo3(get)]
    pub conf_upper: Vec<f64>,
}

/// `cov_type`文字列（大文字小文字を区別しない）を`engine::nonlinear::common::CovType`に
/// パースする。`cov_type="cluster"`のときのみ`cluster`で指定された列を
/// `extract_cluster_groups`で抽出する（他のcov_typeで`cluster`が指定されていれば
/// `ValidationError`、OLSの`cluster`/`hac_*`の扱いと同じ方針）。
///
/// 戻り値の2つ目は`*Result.cov_type`にそのまま格納する正規形の文字列。受け付ける綴りは
/// 正規形の大文字小文字違いだけ（別名は無い）ため小文字化したものが正規形になるが、
/// 呼び出し側が自前で小文字化せずここで返す値を使うことで、別名を足したときに結果へ
/// 別名がそのまま出る事故を防ぐ（`linear::common::build_cov_type`と同じ形）。
///
/// Logit/Probit/Tobit共通（元は`logit.rs`/`probit.rs`/`tobit.rs`に
/// バイト単位で完全一致するコードとして独立複製されていたが、`CovType`自体が
/// 元々3手法共有の型であるのに合わせてここに集約した）。
///
/// # Errors
/// - `cov_type`が既知の値のいずれでもない: `ValidationError`
/// - `cov_type`が`"cluster"`でないのに`cluster`が指定された: `ValidationError`
///
/// `cluster`未指定自体はここでは`ValidationError`にせず、`groups=None`のまま
/// `engine`側の`CommonError::MissingClusterColumn`検証に委ねる（OLSの`fit()`と同じ役割分担）。
pub(crate) fn parse_cov_type(
    df: &DataFrame,
    cov_type: &str,
    cluster: &Option<String>,
) -> PyResult<(CovType, String)> {
    let cov_type_lower = cov_type.to_lowercase();
    let parsed = match cov_type_lower.as_str() {
        "classical" => Ok(CovType::Classical),
        "opg" => Ok(CovType::Opg),
        "hc0" => Ok(CovType::Hc0),
        "hc1" => Ok(CovType::Hc1),
        "cluster" => {
            let groups = extract_optional_cluster_groups(df, cluster.as_deref())?;
            Ok(CovType::Cluster { groups })
        }
        other => Err(ValidationError::new_err(format!(
            "unknown cov_type: '{other}'. Expected one of 'classical', \
             'opg', 'hc0', 'hc1', or 'cluster'"
        ))),
    }?;
    // 未知の`cov_type`は「unknown cov_type」を優先して報告する（上のmatchが先）。
    reject_unused_option(
        "cluster",
        cluster.is_some(),
        cov_type_lower == "cluster",
        "cov_type=\"cluster\"",
    )?;
    Ok((parsed, cov_type_lower))
}

/// `solver`文字列（大文字小文字を区別しない）を`engine::nonlinear::common::SolverType`に
/// パースする。Logit/Probit/Tobit共通（`parse_cov_type`と同じ理由でここに集約）。
///
/// # Errors
/// `solver`が既知の値のいずれでもない: `ValidationError`
pub(crate) fn parse_solver_type(solver_lower: &str) -> PyResult<SolverType> {
    match solver_lower {
        "newton" => Ok(SolverType::Newton),
        "bfgs" => Ok(SolverType::Bfgs),
        "lbfgs" => Ok(SolverType::Lbfgs),
        other => Err(ValidationError::new_err(format!(
            "unknown solver: '{other}'. Expected one of 'newton', 'bfgs', or 'lbfgs'"
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

/// `LogitOptions`/`ProbitOptions`/`TobitOptions`のコンストラクタで`tol=None`のときの既定値を
/// `solver`から解決する。`newton`は絶対閾値`1e-6`、`bfgs`/`lbfgs`は観測数正規化後の`1e-8`。
/// 共有の単一既定値では、Newtonの既定値を締めると大標本で無視できない速度低下（実測:
/// n=1,000,000で0.98s→3.15s）が起きる一方、bfgs/lbfgsの既定値を緩めると小標本の
/// 精度検証テスト（RTOL=1e-8）を壊すため、solver別に分岐する。
/// `tol`が明示されていればそのまま返す。`solver`は大文字小文字を区別せず、未知の値は
/// ここでは検証せず（`fit()`時の`parse_solver_type`が`ValidationError`にする）`1e-8`側になる。
pub(crate) fn resolve_mle_tol_default(solver: &str, tol: Option<f64>) -> f64 {
    tol.unwrap_or(if solver.eq_ignore_ascii_case("newton") {
        1e-6
    } else {
        1e-8
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    /// 大文字小文字の綴りによらず、正規形の文字列を返すこと（`*Result.cov_type`がそのまま
    /// 使う値）。`let-else`を使う理由は`linear/common.rs`の同名テストと同じ（`PyErr`の
    /// Debug実装はGIL取得を要求するため、`unwrap()`/`expect()`を避ける）。
    #[test]
    fn parse_cov_type_is_case_insensitive_and_returns_the_canonical_name() {
        let df = df!("g" => [1.0, 2.0, 3.0]).unwrap();
        for (input, expected) in [
            ("classical", "classical"),
            ("CLASSICAL", "classical"),
            ("Classical", "classical"),
            ("OPG", "opg"),
            ("Hc0", "hc0"),
            ("HC1", "hc1"),
            ("CLUSTER", "cluster"),
        ] {
            let cluster = (expected == "cluster").then(|| "g".to_string());
            let Ok((_, normalized)) = parse_cov_type(&df, input, &cluster) else {
                panic!("expected Ok for input={input}");
            };
            assert_eq!(normalized, expected, "input={input}");
        }
    }

    #[test]
    fn parse_cov_type_rejects_unknown_value() {
        let df = df!("g" => [1.0, 2.0, 3.0]).unwrap();
        assert!(parse_cov_type(&df, "bogus", &None).is_err());
        // 別名は無い（`"nonrobust"`は`"classical"`の別名として受け付けない）。
        assert!(parse_cov_type(&df, "nonrobust", &None).is_err());
    }

    #[test]
    fn resolve_mle_tol_default_uses_solver_dependent_default_when_tol_is_none() {
        assert_eq!(resolve_mle_tol_default("newton", None), 1e-6);
        assert_eq!(resolve_mle_tol_default("NEWTON", None), 1e-6);
        assert_eq!(resolve_mle_tol_default("bfgs", None), 1e-8);
        assert_eq!(resolve_mle_tol_default("lbfgs", None), 1e-8);
    }

    #[test]
    fn resolve_mle_tol_default_ignores_ascii_case_of_solver() {
        assert_eq!(resolve_mle_tol_default("Newton", None), 1e-6);
        assert_eq!(resolve_mle_tol_default("BFGS", None), 1e-8);
        assert_eq!(resolve_mle_tol_default("LBFGS", None), 1e-8);
    }

    /// 未知のsolverはここでは検証しない（`fit()`時の`parse_solver_type`が`ValidationError`に
    /// する）。コンストラクタは`1e-8`側に倒れる、というdocコメントの契約を固定する。
    #[test]
    fn resolve_mle_tol_default_treats_unknown_solver_as_non_newton() {
        assert_eq!(resolve_mle_tol_default("bogus", None), 1e-8);
        assert_eq!(resolve_mle_tol_default("", None), 1e-8);
    }

    #[test]
    fn resolve_mle_tol_default_returns_explicit_tol_unchanged() {
        assert_eq!(resolve_mle_tol_default("newton", Some(1e-3)), 1e-3);
        assert_eq!(resolve_mle_tol_default("bfgs", Some(1e-3)), 1e-3);
    }

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
    // `parse_solver_type`と同じ設計）。大文字小文字を区別しない処理は呼び出し側
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
