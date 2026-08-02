//! Probitの推定オプション・結果、およびPython（polars DataFrame + 列名 + オプション）から
//! `engine::nonlinear::probit`（Newton/BFGS/L-BFGSソルバー・標準誤差・適合度統計量）を
//! 呼び出すためのデータ抽出・バリデーションまでの処理。
//!
//! 【本Issue（#81）のスコープ】`ProbitOptions`/`ProbitResult`のpyclass定義と、データ抽出・
//! バリデーション・`engine::nonlinear::probit::ProbitInput`構築までを担う`build_probit_input`
//! を実装する。`ProbitEstimator::fit()`の呼び出し・`ProbitResult`の実際の構築・
//! `#[pymodule]`への登録はIssue #82（engine呼び出し・エラー変換）のスコープとし、
//! ここでは行わない（Logitの対応するIssue #65/#66の分割と同じ、ユーザー確認済み）。
//!
//! 【責務分離】`.claude/rules/rust-style.md`「Python境界でのデータ受け渡し」参照。
//! polars DataFrameから列ごとの`Vec<f64>`/`Vec<String>`への抽出はここ（`column_extraction`
//! 経由）の責務。`faer::Mat`の組み立て（切片列の自動追加を含む）は`engine::nonlinear::
//! probit::ProbitInput`に委ねる。
//!
//! 【言語方針】`.claude/rules/rust-style.md`「言語方針」参照。
//! 公開API（`ProbitOptions`/`ProbitResult`）のdocコメントと、`ValidationError`のメッセージ
//! 文字列は英語。それ以外（このファイルの説明・非公開関数のdocコメント等）は日本語のまま。
//!
//! `build_probit_input`が`PyDataFrame`ではなく`polars::DataFrame`（プレーンなpolars型）を
//! 受け取る設計にしているのは、`column_extraction::extract_f64_column`等が既に同じ
//! シグネチャ（`&DataFrame`）を使っているため、およびPythonインタプリタ（GIL）を
//! 起動せずに`cargo test`で直接ユニットテストできるようにするため（Logitの`build_logit_input`
//! と同じ理由、Issue #65参照）。`fit`（Issue #82で実装予定）が`PyDataFrame`を受け取り、
//! `.into()`で`DataFrame`に変換してから`build_probit_input`を呼ぶ想定（`logit.rs`の`fit`
//! 関数と同じ変換パターン）。

use engine::nonlinear::common::{CovType as EngineCovType, Method as EngineMethod};
use engine::nonlinear::probit::ProbitInput;
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use super::common::mle_error_to_pyerr;
use crate::column_extraction::{extract_f64_column, extract_group_key_column};
use crate::errors::ValidationError;
use crate::validation::{
    validate_no_const_collision, validate_no_duplicate_roles, validate_no_duplicate_x,
    validate_x_non_empty,
};

/// Estimation options for Probit.
///
/// See `docs/planning/specs/nonlinear-api-design.md` and
/// `docs/planning/specs/nonlinear-implementation-notes.md` for the rationale behind
/// each field's meaning and default value. Field-for-field identical to `LogitOptions`
/// (the two models share the same option surface, `nonlinear-api-design.md` section 7).
///
/// `start_params` (user-specified initial values) is intentionally omitted: the
/// underlying engine (`ProbitEstimator::fit`) does not accept it yet (deferred,
/// same as `LogitOptions`). It will be added once the engine side supports it.
// `fit`（Issue #82）がPython側から`ProbitOptions`インスタンスを引数として受け取るため、
// `FromPyObject`実装を明示的に維持する（`LogitOptions`と同じ理由、pyo3 0.28以降、Cloneを
// 実装する#[pyclass]のFromPyObject自動導出はopt-inに変更されたため）。
// module: PyO3の#[pyclass]はデフォルトで__module__="builtins"になり、
// mkdocstrings（griffe）がPythonでの再エクスポートのalias解決に失敗する原因になる。
// 実際のインポート元(`econometricsmodels._lib`)を明示する。
#[pyclass(from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct ProbitOptions {
    /// Standard error type: one of "classical" (alias "nonrobust"), "opg", "hc0",
    /// "hc1", "cluster". Case-insensitive.
    #[pyo3(get, set)]
    pub cov_type: String,

    /// Whether the engine should automatically add an intercept column.
    /// When true, a column of all 1.0 is prepended to the design matrix.
    /// If the user's `x` already contains a constant column while this is true,
    /// the resulting perfect collinearity raises `ComputationError` (singular
    /// Hessian).
    #[pyo3(get, set)]
    pub include_intercept: bool,

    /// Confidence level for confidence intervals, in the range (0, 1).
    /// Defaults to 0.95 (a 95% confidence interval).
    #[pyo3(get, set)]
    pub confidence_level: f64,

    /// Column name to use as the cluster group key when `cov_type="cluster"`.
    /// Refers to a column in `data` rather than being passed as a separate array.
    /// Ignored when `cov_type` is not "cluster".
    #[pyo3(get, set)]
    pub cluster_col: Option<String>,

    /// Optimization solver: one of "newton" (default), "bfgs", "lbfgs".
    /// Case-insensitive.
    #[pyo3(get, set)]
    pub method: String,

    /// Maximum number of solver iterations.
    #[pyo3(get, set)]
    pub max_iter: i64,

    /// Gradient-norm convergence tolerance.
    #[pyo3(get, set)]
    pub tol: f64,

    /// If true (default), raise `ComputationError` when the solver fails to
    /// converge within `max_iter` iterations. If false, return the result at the
    /// final iterate instead (with `converged=False`).
    #[pyo3(get, set)]
    pub raise_on_non_convergence: bool,
}

#[pymethods]
impl ProbitOptions {
    #[new]
    #[pyo3(signature = (
        cov_type = "classical".to_string(),
        include_intercept = true,
        confidence_level = 0.95,
        cluster_col = None,
        method = "newton".to_string(),
        max_iter = 35,
        tol = 1e-6,
        raise_on_non_convergence = true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        cov_type: String,
        include_intercept: bool,
        confidence_level: f64,
        cluster_col: Option<String>,
        method: String,
        max_iter: i64,
        tol: f64,
        raise_on_non_convergence: bool,
    ) -> Self {
        Self {
            cov_type,
            include_intercept,
            confidence_level,
            cluster_col,
            method,
            max_iter,
            tol,
            raise_on_non_convergence,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ProbitOptions(cov_type={:?}, include_intercept={}, confidence_level={}, \
             cluster_col={:?}, method={:?}, max_iter={}, tol={}, raise_on_non_convergence={})",
            self.cov_type,
            self.include_intercept,
            self.confidence_level,
            self.cluster_col,
            self.method,
            self.max_iter,
            self.tol,
            self.raise_on_non_convergence
        )
    }
}

/// Estimation results for Probit.
///
/// Structured data only (no `summary()`); see `docs/planning/specs/nonlinear-api-design.md`
/// section 5. Row-oriented table construction (e.g. a `coef_table`) is left to
/// `python_package`. All array-valued fields (`params`, `std_errors`, etc.) share the
/// same order as `param_names`.
///
/// Not yet populated by any code in this crate (Issue #81 defines the type only;
/// `fit`, which constructs and returns it, is implemented in Issue #82, mirroring
/// `LogitResult`/Issue #65-#66).
// `ProbitResult`はRust側で組み立ててPythonに返すだけの型で、Python側からの生成・引数として
// 受け取ることは想定していないため`skip_from_py_object`（`ProbitOptions`の`from_py_object`とは
// 対照的。`LogitResult`と同じ理由）。
//
// `get_all`ではなく、フィールドごとに個別`#[pyo3(get)]`を付ける方式にしている
// （`LogitResult`と同じ設計。将来`predict`/`pred_table`/`marginal_effects`の呼び出しに
// 必要な`estimator`フィールド（Issue #83相当）を`#[pyo3(get)]`を付けずに済ませられる
// ようにするため）。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct ProbitResult {
    #[pyo3(get)]
    pub params: Vec<f64>,
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
    #[pyo3(get)]
    pub param_names: Vec<String>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub log_likelihood_null: f64,
    #[pyo3(get)]
    pub lr_statistic: f64,
    #[pyo3(get)]
    pub lr_p_value: f64,
    #[pyo3(get)]
    pub pseudo_r_squared: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    #[pyo3(get)]
    pub n_obs: usize,
    #[pyo3(get)]
    pub df_model: usize,
    #[pyo3(get)]
    pub df_resid: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub n_iter: usize,
    /// Standard error type actually used (echoes `ProbitOptions.cov_type`, normalized
    /// to lowercase; e.g. `"classical"`, `"opg"`, `"hc1"`, `"cluster"`).
    #[pyo3(get)]
    pub cov_type: String,
}

/// `cov_type`文字列（大文字小文字を区別しない）を`engine::nonlinear::common::CovType`に
/// パースする。`cov_type="cluster"`のときのみ`cluster_col`で指定された列を
/// `extract_group_key_column`で抽出する（他のcov_typeでは無視する、`build_logit_input`の
/// `parse_cov_type`と同じ方針）。
///
/// # Errors
/// - `cov_type`が既知の値のいずれでもない: `ValidationError`
///
/// `cluster_col`未指定自体はここでは`ValidationError`にせず、`groups=None`のまま
/// `engine`側の`CommonError::MissingClusterColumn`検証に委ねる（`build_logit_input`と
/// 同じ役割分担）。
///
/// `#[allow(dead_code)]`について: `build_probit_input`のdocコメント参照。
#[allow(dead_code)]
fn parse_cov_type(
    df: &DataFrame,
    cov_type_lower: &str,
    cluster_col: &Option<String>,
) -> PyResult<EngineCovType> {
    match cov_type_lower {
        "classical" | "nonrobust" => Ok(EngineCovType::Classical),
        "opg" => Ok(EngineCovType::Opg),
        "hc0" => Ok(EngineCovType::Hc0),
        "hc1" => Ok(EngineCovType::Hc1),
        "cluster" => {
            let groups = cluster_col
                .as_ref()
                .map(|col_name| extract_group_key_column(df, col_name))
                .transpose()?;
            Ok(EngineCovType::Cluster { groups })
        }
        other => Err(ValidationError::new_err(format!(
            "unknown cov_type: '{other}'. Expected one of 'classical' (or 'nonrobust'), \
             'opg', 'hc0', 'hc1', or 'cluster'"
        ))),
    }
}

/// `method`文字列（大文字小文字を区別しない）を`engine::nonlinear::common::Method`に
/// パースする。
///
/// # Errors
/// `method`が既知の値のいずれでもない: `ValidationError`
///
/// `#[allow(dead_code)]`について: `build_probit_input`のdocコメント参照。
#[allow(dead_code)]
fn parse_method(method_lower: &str) -> PyResult<EngineMethod> {
    match method_lower {
        "newton" => Ok(EngineMethod::Newton),
        "bfgs" => Ok(EngineMethod::Bfgs),
        "lbfgs" => Ok(EngineMethod::Lbfgs),
        other => Err(ValidationError::new_err(format!(
            "unknown method: '{other}'. Expected one of 'newton', 'bfgs', or 'lbfgs'"
        ))),
    }
}

/// Pythonから渡された `data` / `y` / `x` / `options` を検証し、
/// `engine::nonlinear::probit::ProbitInput::from_columns`を呼び出すところまでを行う。
/// `ProbitEstimator::fit`の呼び出し・`ProbitResult`の構築はIssue #82で実装する
/// （このファイル冒頭のdocコメント「本Issueのスコープ」参照）。
///
/// # Errors
/// - 列の抽出時に発覚する問題（列が存在しない、数値/文字列型にキャストできない、
///   欠損値・NaN・無限大を含む等）は`column_extraction`の責務で`ValidationError`
/// - `y`・`x`の重複、`include_intercept=true`のときの`"const"`列との衝突は
///   `validation.rs`の責務で`ValidationError`（`build_logit_input`と同じ役割分担）
/// - `cov_type`/`method`の文字列が不正な場合は`ValidationError`
/// - それ以外（次元不一致等）は`engine::nonlinear::common::MleError`から
///   `mle_error_to_pyerr`で変換
///
/// `#[allow(dead_code)]`について: Issue #81時点では呼び出し元が本ファイルの
/// `#[cfg(test)] mod tests`のみで、`ProbitEstimator::fit`を実際に呼ぶ`fit`（Issue #82で
/// 実装予定）がまだ無いため、`cargo build`（`#[cfg(test)]`を含まないlibターゲットの
/// ビルド）からは到達不能に見え`dead_code`警告になる。`build_logit_input`（Issue #65）と
/// 同じ理由・同じ対応方針（`pub`化での回避は`engine_pybind`がPython拡張モジュール専用の
/// 薄いバインディング層でありクレート外にRust APIを公開する設計ではないため見送り）。
/// Issue #82で`fit`がこの関数を実際に呼ぶようになった時点でこの属性は不要になる
/// （削除すること）。
#[allow(dead_code)]
pub(crate) fn build_probit_input(
    df: &DataFrame,
    y: String,
    x: Vec<String>,
    options: &ProbitOptions,
) -> PyResult<(ProbitInput, EngineCovType, EngineMethod)> {
    let cov_type_lower = options.cov_type.to_lowercase();
    let method_lower = options.method.to_lowercase();

    // 完全な多重共線性を早期に、分かりやすいエラーで防ぐ（`validation.rs`に集約、
    // OLS/WLS/Logitと共通、`.claude/rules/rust-style.md`参照）。
    validate_x_non_empty(&x)?;
    validate_no_duplicate_roles(&[("y", &y)], &x)?;
    validate_no_duplicate_x(&x)?;
    validate_no_const_collision(&x, options.include_intercept)?;

    // ── y列の抽出 ──────────────────────────────────────────────────────
    let y_slice = extract_f64_column(df, &y)?;

    // ── x列の抽出 ──────────────────────────────────────────────────────
    let mut x_slices: Vec<Vec<f64>> = Vec::with_capacity(x.len());
    for col_name in &x {
        x_slices.push(extract_f64_column(df, col_name)?);
    }

    let cov_type = parse_cov_type(df, &cov_type_lower, &options.cluster_col)?;
    let method = parse_method(&method_lower)?;

    let input = ProbitInput::from_columns(&y_slice, &x_slices, x, options.include_intercept, y)
        .map_err(mle_error_to_pyerr)?;

    Ok((input, cov_type, method))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    /// `build_probit_input`のテスト全体で使う既定の`ProbitOptions`（`cov_type="classical"`・
    /// `include_intercept=true`・`method="newton"`）。フィールドごとに上書きして使う。
    fn default_options() -> ProbitOptions {
        ProbitOptions::new(
            "classical".to_string(),
            true,
            0.95,
            None,
            "newton".to_string(),
            35,
            1e-6,
            true,
        )
    }

    fn well_formed_df() -> DataFrame {
        df!(
            "y" => [0.0, 1.0, 0.0, 1.0],
            "x1" => [10.0, 20.0, 30.0, 40.0],
            "x2" => [-5.0, 2.0, 8.0, -1.0],
        )
        .unwrap()
    }

    #[test]
    fn build_probit_input_succeeds_for_well_formed_data() {
        let df = well_formed_df();
        let options = default_options();

        let (input, cov_type, method) = build_probit_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string(), "x2".to_string()],
            &options,
        )
        .unwrap();

        assert_eq!(input.nobs(), 4);
        assert_eq!(input.k(), 3); // const + x1 + x2
        assert_eq!(
            input.param_names(),
            ["const".to_string(), "x1".to_string(), "x2".to_string()]
        );
        assert_eq!(input.dep_var_name(), "y");
        assert!(matches!(cov_type, EngineCovType::Classical));
        assert!(matches!(method, EngineMethod::Newton));
    }

    #[test]
    fn build_probit_input_without_intercept_omits_const_column() {
        let df = well_formed_df();
        let mut options = default_options();
        options.include_intercept = false;

        let (input, _, _) =
            build_probit_input(&df, "y".to_string(), vec!["x1".to_string()], &options).unwrap();

        assert_eq!(input.k(), 1);
        assert_eq!(input.param_names(), ["x1".to_string()]);
    }

    #[test]
    fn build_probit_input_returns_validation_error_for_empty_x() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_probit_input(&df, "y".to_string(), vec![], &options);
        assert!(result.is_err());
    }

    #[test]
    fn build_probit_input_returns_validation_error_when_y_is_also_in_x() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_probit_input(
            &df,
            "y".to_string(),
            vec!["y".to_string(), "x1".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_probit_input_returns_validation_error_for_duplicate_x_column() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_probit_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string(), "x1".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_probit_input_returns_validation_error_for_const_collision_when_include_intercept_is_true()
     {
        let df = df!(
            "y" => [0.0, 1.0, 0.0, 1.0],
            "const" => [1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let options = default_options();

        let result = build_probit_input(&df, "y".to_string(), vec!["const".to_string()], &options);
        assert!(result.is_err());
    }

    #[test]
    fn build_probit_input_returns_validation_error_for_missing_column() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_probit_input(
            &df,
            "y".to_string(),
            vec!["does_not_exist".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_probit_input_returns_validation_error_for_missing_values() {
        let df = df!(
            "y" => [0.0, 1.0, 0.0, 1.0],
            "x1" => [Some(1.0), None, Some(3.0), Some(4.0)],
        )
        .unwrap();
        let options = default_options();

        let result = build_probit_input(&df, "y".to_string(), vec!["x1".to_string()], &options);
        assert!(result.is_err());
    }

    #[test]
    fn build_probit_input_returns_validation_error_for_unknown_cov_type() {
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "bogus".to_string();

        let result = build_probit_input(&df, "y".to_string(), vec!["x1".to_string()], &options);
        assert!(result.is_err());
    }

    #[test]
    fn build_probit_input_returns_validation_error_for_unknown_method() {
        let df = well_formed_df();
        let mut options = default_options();
        options.method = "bogus".to_string();

        let result = build_probit_input(&df, "y".to_string(), vec!["x1".to_string()], &options);
        assert!(result.is_err());
    }

    #[test]
    fn build_probit_input_parses_cluster_cov_type_with_group_column() {
        let df = df!(
            "y" => [0.0, 1.0, 0.0, 1.0],
            "x1" => [10.0, 20.0, 30.0, 40.0],
            "cluster" => ["g1", "g1", "g2", "g2"],
        )
        .unwrap();
        let mut options = default_options();
        options.cov_type = "cluster".to_string();
        options.cluster_col = Some("cluster".to_string());

        let (_, cov_type, _) =
            build_probit_input(&df, "y".to_string(), vec!["x1".to_string()], &options).unwrap();

        match cov_type {
            EngineCovType::Cluster { groups } => {
                assert_eq!(
                    groups,
                    Some(vec![
                        "g1".to_string(),
                        "g1".to_string(),
                        "g2".to_string(),
                        "g2".to_string()
                    ])
                );
            }
            other => panic!("expected Cluster, got {other:?}"),
        }
    }

    #[test]
    fn build_probit_input_leaves_cluster_groups_none_when_cluster_col_not_specified() {
        // クラスターキー未指定自体はここではエラーにせず、`groups=None`のまま返す
        // （`engine`側の`CommonError::MissingClusterColumn`検証に委ねる設計、
        // `build_probit_input`のdocコメント参照）。
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "cluster".to_string();

        let (_, cov_type, _) =
            build_probit_input(&df, "y".to_string(), vec!["x1".to_string()], &options).unwrap();

        match cov_type {
            EngineCovType::Cluster { groups } => assert!(groups.is_none()),
            other => panic!("expected Cluster, got {other:?}"),
        }
    }
}
