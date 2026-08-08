//! Probitの推定オプション・結果、およびPython（polars DataFrame + 列名 + オプション）から
//! `engine::nonlinear::probit`（Newton/BFGS/L-BFGSソルバー・標準誤差・適合度統計量）を
//! 呼び出し、結果をPython側に返すところまでの一連の処理。
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
//! 起動せずに`cargo test`で直接ユニットテストできるようにするため（Issue #81）。
//! `fit`（本ファイルの`pub(crate)`関数、`#[pyfunction] fit_probit`本体は`lib.rs`側にあり
//! これに委譲する、Issue #82）が`PyDataFrame`を受け取り、`.into()`で`DataFrame`に変換して
//! から`build_probit_input`を呼ぶ（`logit.rs`の`fit`関数と同じ変換パターン）。

use engine::nonlinear::common::{CovType as EngineCovType, Method as EngineMethod};
use engine::nonlinear::probit::{ProbitEstimator, ProbitInput};
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use super::common::{
    MarginalEffectsResult, mat_to_nested_vec, mle_error_to_pyerr, parse_marginal_effects_at,
};
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
/// `predict()` / `pred_table()` / `marginal_effects()` are provided as separate methods
/// (not part of this struct's fields), matching `nonlinear-api-design.md` section 6.
// `ProbitResult`はRust側で組み立ててPythonに返すだけの型で、Python側からの生成・引数として
// 受け取ることは想定していないため`skip_from_py_object`（`ProbitOptions`の`from_py_object`とは
// 対照的。`LogitResult`と同じ理由）。
//
// `get_all`ではなく、フィールドごとに個別`#[pyo3(get)]`を付ける方式にしている
// （`LogitResult`と同じ設計。`predict`/`pred_table`/`marginal_effects`の呼び出しに必要な
// `estimator`フィールドを`#[pyo3(get)]`を付けずに済ませられるようにするため）。
//
// `Clone`を派生しない: `estimator`（`engine::nonlinear::probit::ProbitEstimator`）が
// `Clone`を実装していないため（`LogitResult`と同じ理由）。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug)]
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
    /// Not exposed to Python; only `predict`/`pred_table`/`marginal_effects` read it
    /// (`LogitResult`の`estimator`と同じ位置づけ、コメント参照)。
    estimator: ProbitEstimator,
}

#[pymethods]
impl ProbitResult {
    /// Predicted probabilities for the training data used in `fit()`.
    ///
    /// Out-of-sample prediction (a `new_data` argument) is not yet supported
    /// (tracked separately; see `docs/planning/specs/probit-implementation-notes.md`).
    fn predict(&self) -> Vec<f64> {
        self.estimator.predict()
    }

    /// 2x2 classification table as `[[row0], [row1]]`, where row/column index 0 is the
    /// negative class and 1 is the positive class: `table[actual][predicted]`.
    ///
    /// `actual` always uses a fixed 0.5 split; only `predicted` depends on `threshold`
    /// (matches statsmodels' `BinaryResults.pred_table(threshold)`; see
    /// `engine::nonlinear::probit::ProbitEstimator::pred_table` for the exact semantics).
    /// Out-of-sample data is not yet supported (same limitation as `predict()`).
    #[pyo3(signature = (threshold=0.5))]
    fn pred_table(&self, threshold: f64) -> Vec<Vec<f64>> {
        mat_to_nested_vec(&self.estimator.pred_table(threshold))
    }

    /// Marginal effects (`dy/dx`) with delta-method standard errors.
    ///
    /// Independent of `fit()`'s `confidence_level` (re-evaluated here so callers can
    /// use a different confidence level without re-fitting). See
    /// `docs/planning/specs/nonlinear-api-design.md` section 6.
    ///
    /// # Errors
    /// - `at` is not one of `"overall"`, `"mean"`, `"median"` (case-insensitive):
    ///   `ValidationError`
    /// - `confidence_level` is outside `(0, 1)`: `ValidationError`
    #[pyo3(signature = (at="overall".to_string(), confidence_level=0.95))]
    fn marginal_effects(
        &self,
        at: String,
        confidence_level: f64,
    ) -> PyResult<MarginalEffectsResult> {
        let at = parse_marginal_effects_at(&at.to_lowercase())?;
        let effects = self
            .estimator
            .marginal_effects(at, confidence_level)
            .map_err(mle_error_to_pyerr)?;

        Ok(MarginalEffectsResult {
            param_names: effects.param_names().to_vec(),
            dydx: effects.dydx().to_vec(),
            std_errors: effects.std_errors().to_vec(),
            z_stats: effects.z_stats().to_vec(),
            p_values: effects.p_values().to_vec(),
            conf_lower: effects.conf_lower().to_vec(),
            conf_upper: effects.conf_upper().to_vec(),
        })
    }
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
/// `ProbitEstimator::fit`の呼び出し・`ProbitResult`の構築は`fit`（本ファイル、Issue #82）が行う。
///
/// # Errors
/// - 列の抽出時に発覚する問題（列が存在しない、数値/文字列型にキャストできない、
///   欠損値・NaN・無限大を含む等）は`column_extraction`の責務で`ValidationError`
/// - `y`・`x`の重複、`include_intercept=true`のときの`"const"`列との衝突は
///   `validation.rs`の責務で`ValidationError`（`build_logit_input`と同じ役割分担）
/// - `cov_type`/`method`の文字列が不正な場合は`ValidationError`
/// - それ以外（次元不一致等）は`engine::nonlinear::common::MleError`から
///   `mle_error_to_pyerr`で変換
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

/// Pythonから渡された `data` / `y` / `x` / `options` を検証し、
/// `build_probit_input`で構築した`ProbitInput`に対して`engine::nonlinear::probit::
/// ProbitEstimator::fit`を呼び出してProbitを推定し、`ProbitResult`として返す。
///
/// # Errors
/// - `build_probit_input`が返すエラー（列抽出・y/xの重複・`"const"`列衝突・
///   `cov_type`/`method`文字列の検証等）は`ValidationError`
/// - `ProbitEstimator::fit`が返す`MleError`（`confidence_level`範囲外・`max_iter`が
///   0以下・観測数不足・未収束・特異Hessian・特異OPG行列・クラスターキー未指定・
///   クラスター数不足等）は`mle_error_to_pyerr`で変換（詳細は
///   `engine::nonlinear::probit::ProbitEstimator::fit`のdocコメント参照）
pub(crate) fn fit(
    data: PyDataFrame,
    y: String,
    x: Vec<String>,
    options: &ProbitOptions,
) -> PyResult<ProbitResult> {
    let df: DataFrame = data.into();
    let (input, cov_type, method) = build_probit_input(&df, y, x, options)?;

    let estimator = ProbitEstimator::fit(
        input,
        method,
        options.max_iter,
        options.tol,
        options.raise_on_non_convergence,
        cov_type,
        options.confidence_level,
    )
    .map_err(mle_error_to_pyerr)?;

    Ok(ProbitResult {
        params: estimator.params().to_vec(),
        std_errors: estimator.std_errors().to_vec(),
        z_stats: estimator.z_stats().to_vec(),
        p_values: estimator.p_values().to_vec(),
        conf_lower: estimator.conf_lower().to_vec(),
        conf_upper: estimator.conf_upper().to_vec(),
        param_names: estimator.input().param_names().to_vec(),
        log_likelihood: estimator.log_likelihood(),
        log_likelihood_null: estimator.log_likelihood_null(),
        lr_statistic: estimator.lr_statistic(),
        lr_p_value: estimator.lr_p_value(),
        pseudo_r_squared: estimator.pseudo_r_squared(),
        aic: estimator.aic(),
        bic: estimator.bic(),
        n_obs: estimator.n_obs(),
        df_model: estimator.df_model(),
        df_resid: estimator.df_resid(),
        converged: estimator.converged(),
        n_iter: estimator.n_iter(),
        cov_type: options.cov_type.to_lowercase(),
        estimator,
    })
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
