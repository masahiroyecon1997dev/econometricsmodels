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
//! 起動せずに`cargo test`で直接ユニットテストできるようにするため。
//! `fit`（本ファイルの`pub(crate)`関数、`#[pyfunction] fit_probit`本体は`lib.rs`側にあり
//! これに委譲する）が`PyDataFrame`を受け取り、`.into()`で`DataFrame`に変換して
//! から`build_probit_input`を呼ぶ（`logit.rs`の`fit`関数と同じ変換パターン）。

use crate::shared::errors::guard;
use engine::nonlinear::common::{
    CovType as EngineCovType, MleFitOptions, SolverType as EngineSolverType,
};
use engine::nonlinear::probit::{ProbitEstimator, ProbitFitted, ProbitInput};
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use super::common::{
    MarginalEffectsResult, check_cov_options, mat_to_nested_vec, mle_error_to_pyerr,
    parse_cov_kind, parse_marginal_effects_at, parse_solver_type, resolve_cov_type,
    resolve_mle_tol_default, solver_name,
};
use crate::shared::augment::append_predicted_column;
use crate::shared::column_error::ColumnSource;
use crate::shared::column_extraction::{
    SelectedData, extract_f64_column, extract_f64_columns, extract_model_x_columns,
    select_model_x_columns, x_column_names,
};
use crate::shared::option_repr::PyRepr;
use crate::shared::option_values::{
    extract_strict_float, extract_strict_int, extract_strict_opt_column, extract_strict_text,
};
use crate::shared::validation::validate_common_roles;

/// Estimation options for Probit.
///
/// All arguments are keyword-only.
///
/// Options by purpose:
///
/// - Standard errors: `cov_type`; with `"cluster"` also `cluster`.
/// - Optimizer: `solver`, `max_iter`, `tol`, `raise_on_non_convergence`.
/// - Confidence intervals: `confidence_level`.
/// - Model: `include_intercept`.
// 各フィールドの意味・既定値の根拠は`docs/spec/nonlinear-common.md`（7章: `LogitOptions`と
// 同じオプション構成）。`start_params`は`LogitOptions`と同じ理由で意図的に持たない。
// `fit`がPython側から`ProbitOptions`インスタンスを引数として受け取るため、
// `FromPyObject`実装を明示的に維持する（`LogitOptions`と同じ理由、pyo3 0.28以降、Cloneを
// 実装する#[pyclass]のFromPyObject自動導出はopt-inに変更されたため）。
// module: PyO3の#[pyclass]はデフォルトで__module__="builtins"になり、
// mkdocstrings（griffe）がPythonでの再エクスポートのalias解決に失敗する原因になる。
// 実際のインポート元(`econometricsmodels._lib`)を明示する。
#[pyclass(from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct ProbitOptions {
    /// Standard error type: one of "classical", "opg", "hc0",
    /// "hc1", "cluster". Case-insensitive.
    #[pyo3(get)]
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
    #[pyo3(get)]
    pub confidence_level: f64,

    /// Column name to use as the cluster group key when `cov_type="cluster"`.
    /// Refers to a column in `data` rather than being passed as a separate array.
    /// Specifying it with any other `cov_type` raises `ValidationError`.
    #[pyo3(get)]
    pub cluster: Option<String>,

    /// Optimization solver: one of "newton" (default), "bfgs", "lbfgs".
    /// Case-insensitive.
    #[pyo3(get)]
    pub solver: String,

    /// Maximum number of solver iterations, an integer from 1 to 10,000 (anything else
    /// raises `ValidationError`).
    #[pyo3(get)]
    pub max_iter: i64,

    /// Convergence tolerance. For `solver="newton"`, an absolute threshold on
    /// the total gradient norm (default `1e-6`). For `solver="bfgs"`/`"lbfgs"`,
    /// a per-observation-average gradient threshold (the total gradient norm
    /// divided by the number of observations must fall below `tol`, default
    /// `1e-8`), matching how statsmodels/scipy normalize convergence checks
    /// by `nobs`. The two defaults are not interchangeable: Newton's absolute
    /// semantics with a `1e-8` threshold (or bfgs/lbfgs's normalized
    /// semantics with a `1e-6` threshold) measurably degrades either speed or
    /// precision. Passing `tol` explicitly always uses the
    /// semantics of the chosen `solver`. For `bfgs`/`lbfgs`, the solver also
    /// reports convergence when the gradient norm is within `100 * tol` of the
    /// target and the line search can no longer make progress because the
    /// log-likelihood has reached floating-point precision (common with very
    /// large samples); this window scales with `tol`, so it widens if you
    /// loosen `tol`. Note: the solver-dependent default is
    /// resolved once, at construction time. Changing `solver` afterwards via
    /// the setter does not re-resolve `tol` — set both together (or set `tol`
    /// explicitly) if you change `solver` after construction.
    #[pyo3(get)]
    pub tol: f64,

    /// If true (default), raise `ComputationError` when the solver fails to
    /// converge within `max_iter` iterations. If false, return the result at the
    /// final iterate instead (with `converged=False`).
    #[pyo3(get, set)]
    pub raise_on_non_convergence: bool,
}

crate::shared::pickle::impl_pickle_state!(ProbitOptions {
    cov_type,
    include_intercept,
    confidence_level,
    cluster,
    solver,
    max_iter,
    tol,
    raise_on_non_convergence,
});

impl ProbitOptions {
    /// このオプションが指名するデータの列（`cluster`）。使われるかどうか（`cov_type`等）には
    /// よらず、指名されたものを全て返す。`fit()`はこれらと`y`・`x`等だけをデータから取り込む。
    pub(crate) fn data_columns(&self) -> Vec<String> {
        self.cluster.iter().cloned().collect()
    }
}

#[pymethods]
impl ProbitOptions {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        crate::shared::pickle::reduce(slf)
    }

    #[new]
    #[pyo3(signature = (
        *,
        cov_type = "classical".to_string(),
        include_intercept = true,
        confidence_level = 0.95,
        cluster = None,
        solver = "newton".to_string(),
        max_iter = 35,
        tol = None,
        raise_on_non_convergence = true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = crate::shared::option_values::cov_type_arg)] cov_type: String,
        include_intercept: bool,
        #[pyo3(from_py_with = crate::shared::option_values::confidence_level_arg)] confidence_level: f64,
        #[pyo3(from_py_with = crate::shared::option_values::cluster_arg)] cluster: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::solver_arg)] solver: String,
        #[pyo3(from_py_with = crate::shared::option_values::max_iter_arg)] max_iter: i64,
        #[pyo3(from_py_with = crate::shared::option_values::tol_arg)] tol: Option<f64>,
        raise_on_non_convergence: bool,
    ) -> Self {
        // `tol`の既定値はsolver依存（理由は`resolve_mle_tol_default`参照）。
        let tol = resolve_mle_tol_default(&solver, tol);
        Self {
            cov_type,
            include_intercept,
            confidence_level,
            cluster,
            solver,
            max_iter,
            tol,
            raise_on_non_convergence,
        }
    }

    #[setter]
    fn set_confidence_level(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.confidence_level = extract_strict_float(value, "confidence_level")?;
        Ok(())
    }

    #[setter]
    fn set_max_iter(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.max_iter = extract_strict_int(value, "max_iter")?;
        Ok(())
    }

    #[setter]
    fn set_tol(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.tol = extract_strict_float(value, "tol")?;
        Ok(())
    }

    #[setter]
    fn set_cov_type(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.cov_type = extract_strict_text(value, "cov_type")?;
        Ok(())
    }

    #[setter]
    fn set_cluster(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.cluster = extract_strict_opt_column(value, "cluster")?;
        Ok(())
    }

    #[setter]
    fn set_solver(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.solver = extract_strict_text(value, "solver")?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "ProbitOptions(cov_type={}, include_intercept={}, confidence_level={}, \
             cluster={}, solver={}, max_iter={}, tol={}, raise_on_non_convergence={})",
            self.cov_type.py_repr(),
            self.include_intercept.py_repr(),
            self.confidence_level.py_repr(),
            self.cluster.py_repr(),
            self.solver.py_repr(),
            self.max_iter.py_repr(),
            self.tol.py_repr(),
            self.raise_on_non_convergence.py_repr()
        )
    }
}

/// Estimation results for Probit.
///
/// Structured data only (no `summary()`); see `docs/spec/nonlinear-common.md`
/// section 5. Row-oriented table construction (e.g. a `params_table`) is left to
/// `python_package`. All array-valued fields (`params`, `std_errors`, etc.) share the
/// same order as `param_names`.
///
/// `predict()` / `pred_table()` / `marginal_effects()` are provided as separate methods
/// (not part of this struct's fields), matching `docs/spec/nonlinear-common.md` section 6.
// `ProbitResult`はRust側で組み立ててPythonに返すだけの型で、Python側からの生成・引数として
// 受け取ることは想定していないため`skip_from_py_object`（`ProbitOptions`の`from_py_object`とは
// 対照的。`LogitResult`と同じ理由）。
//
// `get_all`ではなく、フィールドごとに個別`#[pyo3(get)]`を付ける方式にしている
// （`LogitResult`と同じ設計。`predict`/`pred_table`/`marginal_effects`の呼び出しに必要な
// `fitted`フィールドを`#[pyo3(get)]`を付けずに済ませられるようにするため）。
//
// `Clone`を派生しない: `fitted`（`engine::nonlinear::probit::ProbitFitted`）が
// `Clone`を実装していないため（`LogitResult`と同じ理由）。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug)]
pub struct ProbitResult {
    #[pyo3(get)]
    pub params: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub test_stats: Vec<f64>,
    /// Distribution of `test_stats`: `"t"` or `"normal"`.
    #[pyo3(get)]
    pub stat_dist: String,
    /// Degrees of freedom of the t distribution (`None` for `"normal"`). May differ from
    /// `df_resid` (e.g. cluster-robust inference uses `G - 1`).
    #[pyo3(get)]
    pub stat_df: Option<i64>,
    #[pyo3(get)]
    pub p_values: Vec<f64>,
    #[pyo3(get)]
    pub conf_lower: Vec<f64>,
    #[pyo3(get)]
    pub conf_upper: Vec<f64>,
    #[pyo3(get)]
    pub param_names: Vec<String>,
    #[pyo3(get)]
    pub dep_var_name: String,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub log_likelihood_null: f64,
    #[pyo3(get)]
    pub lr_statistic: f64,
    #[pyo3(get)]
    pub lr_p_value: f64,
    /// Degrees of freedom of the chi-squared likelihood-ratio test (`None` when there are no
    /// slope coefficients).
    #[pyo3(get)]
    pub lr_df: Option<usize>,
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
    /// Optimization solver actually used (echoes `ProbitOptions.solver`, normalized
    /// to lowercase; one of `"newton"`, `"bfgs"`, `"lbfgs"`).
    #[pyo3(get)]
    pub solver: String,
    /// Not exposed to Python; only `predict`/`pred_table`/`marginal_effects` read it
    /// (`LogitResult`の`fitted`と同じ位置づけ、コメント参照)。
    fitted: ProbitFitted,
    /// The `polars.DataFrame` passed to `fit()`, as given. `augment(new_data=None)` appends
    /// the predicted column to it, so every column is kept, including the ones the model
    /// does not read.
    source_data: Py<PyAny>,
}

// 入力データのうち`y`・`x`は状態に入れない（`source_data`の列から作り直せる）。入れるのは、`x`の列名・`y`の列名・
// 切片の有無・係数・分散共分散行列。復元は`from_columns`/`from_parts`を通して次元の整合を検査する
// （列が無い・不整合なら`PICKLE_FORMAT_MISMATCH`）。
impl crate::shared::pickle::PickleFieldFromSource for ProbitFitted {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use crate::shared::pickle::{PickleField, to_tuple};
        let input = self.input();
        let x_names = x_column_names(input.param_names(), input.has_intercept(), 0).to_vec();
        to_tuple(
            py,
            vec![
                x_names.to_py(py)?,
                input.dep_var_name().to_string().to_py(py)?,
                input.has_intercept().to_py(py)?,
                self.params().to_vec().to_py(py)?,
                self.cov_params().to_py(py)?,
            ],
        )
    }

    fn from_py(obj: &Bound<'_, PyAny>, source: &Bound<'_, PyAny>) -> Option<Self> {
        use crate::nonlinear::common::rebuild_columns;
        use crate::shared::pickle::{PickleField, from_tuple};
        let items = from_tuple(obj, 5)?;
        let x_names: Vec<String> = PickleField::from_py(&items[0])?;
        let dep_var_name: String = PickleField::from_py(&items[1])?;
        let has_intercept: bool = PickleField::from_py(&items[2])?;
        let params: Vec<f64> = PickleField::from_py(&items[3])?;
        let (y, x) = rebuild_columns(source, &dep_var_name, &x_names, has_intercept, params.len())?;
        let input = ProbitInput::from_columns(&y, &x, x_names, has_intercept, dep_var_name).ok()?;
        ProbitFitted::from_parts(input, params, PickleField::from_py(&items[4])?).ok()
    }
}

crate::shared::pickle::impl_pickle_state!(ProbitResult {
    params,
    std_errors,
    test_stats,
    stat_dist,
    stat_df,
    p_values,
    conf_lower,
    conf_upper,
    param_names,
    dep_var_name,
    log_likelihood,
    log_likelihood_null,
    lr_statistic,
    lr_p_value,
    lr_df,
    pseudo_r_squared,
    aic,
    bic,
    n_obs,
    df_model,
    df_resid,
    converged,
    n_iter,
    cov_type,
    solver,
    source_data,
} rebuilt_from(source_data) {
    fitted
});

// `#[pymethods]`ブロックの外に置く非公開実装（`LogitResult`と同じ理由）。
impl ProbitResult {
    /// `new_data`から`predict()`/`augment()`が使う`x`列だけを取り込む。
    fn select_new_data(&self, new_data: &Bound<'_, PyAny>) -> PyResult<SelectedData> {
        select_model_x_columns(
            new_data,
            "new_data",
            self.fitted.input().param_names(),
            self.fitted.input().has_intercept(),
            0,
        )
    }

    /// `predict()`/`augment()`のSome分岐で共有する、`df`に対するout-of-sample予測
    /// （`x`列の抽出→`predict_new_data`呼び出し）。
    fn predict_for(&self, df: &DataFrame) -> PyResult<Vec<f64>> {
        let has_intercept = self.fitted.input().has_intercept();
        let x_columns = extract_model_x_columns(
            df,
            self.fitted.input().param_names(),
            has_intercept,
            0,
            ColumnSource::new("x", "new_data"),
        )?;

        Ok(self.fitted.predict_new_data(&x_columns))
    }
}

#[pymethods]
impl ProbitResult {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        crate::shared::pickle::reduce(slf)
    }

    /// Predicted probabilities for the training data used in `fit()`, or for
    /// `new_data` when given.
    ///
    /// # Errors
    /// - A required `x` column is missing from `new_data`, cannot be cast to a
    ///   numeric type, or contains missing/NaN/infinite values: `ValidationError`
    ///   (same validation as `fit()`'s column extraction, via `extract_f64_column`).
    #[pyo3(signature = (new_data=None))]
    fn predict(&self, new_data: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f64>> {
        guard(|| {
            let Some(new_data) = new_data else {
                return Ok(self.fitted.predict());
            };

            let selected = self.select_new_data(new_data)?;
            self.predict_for(&selected.frame)
        })
    }

    /// The source data (training data, or `new_data` when given) with the predicted
    /// probabilities appended as a new `"predicted_probability"` column.
    ///
    /// Same `new_data`/`include_intercept` semantics as `predict()`, but returns a
    /// polars DataFrame (original columns plus `"predicted_probability"`, row order preserved)
    /// instead of a bare list of floats (same design as `LogitResult::augment()`).
    ///
    /// # Errors
    /// - Same as `predict()`: a required `x` column missing from `new_data`,
    ///   non-numeric, or containing missing/NaN/infinite values: `ValidationError`.
    /// - The source data already has a column named `"predicted_probability"`:
    ///   `ValidationError` (would otherwise silently overwrite it).
    #[pyo3(signature = (new_data=None))]
    fn augment(&self, py: Python<'_>, new_data: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        guard(|| {
            let (source, probability) = match new_data {
                Some(new_data) => {
                    let selected = self.select_new_data(new_data)?;
                    let probability = self.predict_for(&selected.frame)?;
                    (new_data.clone(), probability)
                }
                None => (self.source_data.bind(py).clone(), self.fitted.predict()),
            };

            append_predicted_column(&source, "predicted_probability", probability)
        })
    }

    /// 2x2 classification table as `[[row0], [row1]]`, where row/column index 0 is the
    /// negative class and 1 is the positive class: `table[actual][predicted]`.
    ///
    /// `actual` always uses a fixed 0.5 split; only `predicted` depends on `threshold`
    /// (matches statsmodels' `BinaryResults.pred_table(threshold)`; see
    /// `engine::nonlinear::probit::ProbitEstimator::pred_table` for the exact semantics).
    /// Out-of-sample data is not yet supported (same limitation as `predict()`).
    #[pyo3(signature = (threshold=0.5))]
    fn pred_table(&self, threshold: f64) -> PyResult<Vec<Vec<f64>>> {
        guard(|| Ok(mat_to_nested_vec(&self.fitted.pred_table(threshold))))
    }

    /// Marginal effects (`dy/dx`) with delta-method standard errors.
    ///
    /// Independent of `fit()`'s `confidence_level` (re-evaluated here so callers can
    /// use a different confidence level without re-fitting). See
    /// `docs/spec/nonlinear-common.md` section 6.
    ///
    /// # Errors
    /// - `at` is not one of `"overall"`, `"mean"`, `"median"` (case-insensitive):
    ///   `ValidationError`
    /// - `confidence_level` is outside `(0, 1)`: `ValidationError`
    #[pyo3(signature = (at="overall".to_string(), confidence_level=0.95))]
    fn marginal_effects(
        &self,
        py: Python<'_>,
        at: String,
        confidence_level: f64,
    ) -> PyResult<MarginalEffectsResult> {
        guard(|| {
            let at = parse_marginal_effects_at(&at)?;
            // 推定量だけを使う純Rustの計算なので、GILを手放して走らせる。
            let effects = py
                .detach(|| self.fitted.marginal_effects(at, confidence_level))
                .map_err(mle_error_to_pyerr)?;

            Ok(MarginalEffectsResult {
                param_names: effects.param_names().to_vec(),
                effect: effects.effect().to_vec(),
                std_errors: effects.std_errors().to_vec(),
                test_stats: effects.test_stats().to_vec(),
                p_values: effects.p_values().to_vec(),
                conf_lower: effects.conf_lower().to_vec(),
                conf_upper: effects.conf_upper().to_vec(),
            })
        })
    }
}

/// Pythonから渡された `data` / `y` / `x` / `options` を検証し、
/// `engine::nonlinear::probit::ProbitInput::from_columns`を呼び出すところまでを行う。
/// `ProbitEstimator::fit`の呼び出し・`ProbitResult`の構築は`fit`（本ファイル）が行う。
///
/// # Errors
/// - 列の抽出時に発覚する問題（列が存在しない、数値/文字列型にキャストできない、
///   欠損値・NaN・無限大を含む等）は`column_extraction`の責務で`ValidationError`
/// - `y`・`x`の重複、`include_intercept=true`のときの`"const"`列との衝突は
///   `validation.rs`の責務で`ValidationError`（`build_logit_input`と同じ役割分担）
/// - `cov_type`/`solver`の文字列が不正な場合は`ValidationError`
/// - それ以外（次元不一致等）は`engine::nonlinear::common::MleError`から
///   `mle_error_to_pyerr`で変換
pub(crate) fn build_probit_input(
    df: &DataFrame,
    y: String,
    x: Vec<String>,
    options: &ProbitOptions,
) -> PyResult<(ProbitInput, EngineCovType, String, EngineSolverType)> {
    // DataFrameを見ない検査を先に行う。報告順は`docs/guide/validation.md`
    // 「Order in which errors are reported」: オプションの値（`cov_type` → `solver`）→
    // オプションの組合せ → ロール構造 → `y`/`x`の列 → オプション列 → エンジン。
    let cov_kind = parse_cov_kind(&options.cov_type)?;
    let solver = parse_solver_type(&options.solver)?;
    check_cov_options(cov_kind, options.cluster.as_deref())?;

    // 完全な多重共線性を早期に、分かりやすいエラーで防ぐ（`validation.rs`に集約、
    // OLS/WLS/Logitと共通、`.claude/rules/rust-style.md`参照）。
    validate_common_roles(&y, &x, options.include_intercept)?;

    // ── y列の抽出 ──────────────────────────────────────────────────────
    let y_slice = extract_f64_column(df, &y, ColumnSource::new("y", "data"))?;

    // ── x列の抽出 ──────────────────────────────────────────────────────
    let x_slices = extract_f64_columns(df, &x, ColumnSource::new("x", "data"))?;

    let cov_type = resolve_cov_type(df, cov_kind, options.cluster.as_deref())?;

    let input = ProbitInput::from_columns(&y_slice, &x_slices, x, options.include_intercept, y)
        .map_err(mle_error_to_pyerr)?;

    Ok((input, cov_type, cov_kind.name().to_string(), solver))
}

/// Pythonから渡された `data` / `y` / `x` / `options` を検証し、
/// `build_probit_input`で構築した`ProbitInput`に対して`engine::nonlinear::probit::
/// ProbitEstimator::fit`を呼び出してProbitを推定し、`ProbitResult`として返す。
///
/// # Errors
/// - `build_probit_input`が返すエラー（列抽出・y/xの重複・`"const"`列衝突・
///   `cov_type`/`solver`文字列の検証等）は`ValidationError`
/// - `ProbitEstimator::fit`が返す`MleError`（`confidence_level`範囲外・`max_iter`が
///   0以下・観測数不足・未収束・特異Hessian・特異OPG行列・クラスターキー未指定・
///   クラスター数不足等）は`mle_error_to_pyerr`で変換（詳細は
///   `engine::nonlinear::probit::ProbitEstimator::fit`のdocコメント参照）
pub(crate) fn fit(
    py: Python<'_>,
    data: SelectedData,
    y: String,
    x: Vec<String>,
    options: &ProbitOptions,
) -> PyResult<ProbitResult> {
    let SelectedData { source, frame } = data;
    // 取り込み済みの`frame`以降はPythonオブジェクトに触れないので、GILを手放して走らせる
    // （他のPythonスレッドを止めず、複数のfitをスレッドで並行できる）。結果の組み立てまで
    // 手放した側で行うため、`source`（`Py<PyAny>`は`Send`）もクロージャに渡す。
    py.detach(|| fit_frame(&frame, source, y, x, options))
}

/// [`fit`]の純Rust部分（GILを手放した状態で走る）。`PyErr`は遅延生成でGILを要求しない
/// ので、ここで`?`により返してよい。`source`は結果に載せるだけで、GILが要る操作はしない。
fn fit_frame(
    df: &DataFrame,
    source: Py<PyAny>,
    y: String,
    x: Vec<String>,
    options: &ProbitOptions,
) -> PyResult<ProbitResult> {
    let (input, cov_type, cov_type_lower, solver) = build_probit_input(df, y, x, options)?;

    let estimator = ProbitEstimator::fit(
        input,
        MleFitOptions {
            solver,
            max_iter: options.max_iter,
            tol: options.tol,
            raise_on_non_convergence: options.raise_on_non_convergence,
            cov_type,
            confidence_level: options.confidence_level,
        },
    )
    .map_err(mle_error_to_pyerr)?;

    Ok(ProbitResult {
        params: estimator.params().to_vec(),
        std_errors: estimator.std_errors().to_vec(),
        test_stats: estimator.test_stats().to_vec(),
        stat_dist: estimator.stat_dist().name().to_string(),
        stat_df: estimator.stat_dist().df().map(|df| df as i64),
        p_values: estimator.p_values().to_vec(),
        conf_lower: estimator.conf_lower().to_vec(),
        conf_upper: estimator.conf_upper().to_vec(),
        param_names: estimator.input().param_names().to_vec(),
        dep_var_name: estimator.input().dep_var_name().to_string(),
        log_likelihood: estimator.log_likelihood(),
        log_likelihood_null: estimator.log_likelihood_null(),
        lr_statistic: estimator.lr_statistic(),
        lr_p_value: estimator.lr_p_value(),
        lr_df: estimator.lr_df(),
        pseudo_r_squared: estimator.pseudo_r_squared(),
        aic: estimator.aic(),
        bic: estimator.bic(),
        n_obs: estimator.n_obs(),
        df_model: estimator.df_model(),
        df_resid: estimator.df_resid(),
        converged: estimator.converged(),
        n_iter: estimator.n_iter(),
        cov_type: cov_type_lower,
        solver: solver_name(solver).to_string(),
        fitted: estimator.into_fitted(),
        source_data: source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::shared::group_codes::GroupCodes;
    use polars::df;

    /// `build_probit_input`のテスト全体で使う既定の`ProbitOptions`（`cov_type="classical"`・
    /// `include_intercept=true`・`solver="newton"`）。フィールドごとに上書きして使う。
    fn default_options() -> ProbitOptions {
        ProbitOptions::new(
            "classical".to_string(),
            true,
            0.95,
            None,
            "newton".to_string(),
            35,
            Some(1e-6),
            true,
        )
    }

    /// `tol=None`のとき、`solver`に応じた既定値（`newton`は絶対閾値`1e-6`、
    /// `bfgs`/`lbfgs`は観測数正規化基準`1e-8`）が解決されるはず
    /// （`LogitOptions`と同じロジック）。
    #[test]
    fn new_resolves_tol_default_based_on_method_when_tol_is_none() {
        let newton = ProbitOptions::new(
            "classical".to_string(),
            true,
            0.95,
            None,
            "newton".to_string(),
            35,
            None,
            true,
        );
        assert_eq!(newton.tol, 1e-6);

        let bfgs = ProbitOptions::new(
            "classical".to_string(),
            true,
            0.95,
            None,
            "bfgs".to_string(),
            35,
            None,
            true,
        );
        assert_eq!(bfgs.tol, 1e-8);

        let lbfgs = ProbitOptions::new(
            "classical".to_string(),
            true,
            0.95,
            None,
            "lbfgs".to_string(),
            35,
            None,
            true,
        );
        assert_eq!(lbfgs.tol, 1e-8);
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

        let Ok((input, cov_type, _, solver)) = build_probit_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string(), "x2".to_string()],
            &options,
        ) else {
            panic!("expected Ok");
        };

        assert_eq!(input.nobs(), 4);
        assert_eq!(input.k(), 3); // const + x1 + x2
        assert_eq!(
            input.param_names(),
            ["const".to_string(), "x1".to_string(), "x2".to_string()]
        );
        assert_eq!(input.dep_var_name(), "y");
        assert!(matches!(cov_type, EngineCovType::Classical));
        assert!(matches!(solver, EngineSolverType::Newton));
    }

    #[test]
    fn build_probit_input_without_intercept_omits_const_column() {
        let df = well_formed_df();
        let mut options = default_options();
        options.include_intercept = false;

        let Ok((input, _, _, _)) =
            build_probit_input(&df, "y".to_string(), vec!["x1".to_string()], &options)
        else {
            panic!("expected Ok");
        };

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
        options.solver = "bogus".to_string();

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
        options.cluster = Some("cluster".to_string());

        let Ok((_, cov_type, _, _)) =
            build_probit_input(&df, "y".to_string(), vec!["x1".to_string()], &options)
        else {
            panic!("expected Ok");
        };

        match cov_type {
            EngineCovType::Cluster { groups } => {
                assert_eq!(
                    groups,
                    Some(GroupCodes::from_labels_without_keys(&[
                        "g1".to_string(),
                        "g1".to_string(),
                        "g2".to_string(),
                        "g2".to_string()
                    ]))
                );
            }
            other => panic!("expected Cluster, got {other:?}"),
        }
    }

    #[test]
    fn build_probit_input_leaves_cluster_groups_none_when_cluster_not_specified() {
        // クラスターキー未指定自体はここではエラーにせず、`groups=None`のまま返す
        // （`engine`側の`CommonError::MissingClusterColumn`検証に委ねる設計、
        // `build_probit_input`のdocコメント参照）。
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "cluster".to_string();

        let Ok((_, cov_type, _, _)) =
            build_probit_input(&df, "y".to_string(), vec!["x1".to_string()], &options)
        else {
            panic!("expected Ok");
        };

        match cov_type {
            EngineCovType::Cluster { groups } => assert!(groups.is_none()),
            other => panic!("expected Cluster, got {other:?}"),
        }
    }

    /// `build_probit_input`（Python境界から渡された文字列を受ける入口）を通した
    /// 大文字小文字非依存性と、戻り値の正規形の文字列（`ProbitResult.cov_type`に格納する値）の
    /// 検証（`build_logit_input`の同名テストと同じ理由、`testing-completeness-reviewer`指摘）。
    #[test]
    fn build_probit_input_cov_type_is_case_insensitive() {
        let df = well_formed_df();
        for (input, is_expected) in [
            (
                "classical",
                (|c: &EngineCovType| matches!(c, EngineCovType::Classical))
                    as fn(&EngineCovType) -> bool,
            ),
            ("CLASSICAL", |c| matches!(c, EngineCovType::Classical)),
            ("Classical", |c| matches!(c, EngineCovType::Classical)),
            ("OPG", |c| matches!(c, EngineCovType::Opg)),
            ("Hc0", |c| matches!(c, EngineCovType::Hc0)),
            ("HC1", |c| matches!(c, EngineCovType::Hc1)),
            ("CLUSTER", |c| matches!(c, EngineCovType::Cluster { .. })),
        ] {
            let mut options = default_options();
            options.cov_type = input.to_string();
            let Ok((_, cov_type, cov_type_name, _)) = build_probit_input(
                &df,
                "y".to_string(),
                vec!["x1".to_string(), "x2".to_string()],
                &options,
            ) else {
                panic!("expected Ok for cov_type={input}");
            };
            assert!(is_expected(&cov_type), "input={input}, got={cov_type:?}");
            assert_eq!(cov_type_name, input.to_lowercase(), "input={input}");
        }
    }
}
