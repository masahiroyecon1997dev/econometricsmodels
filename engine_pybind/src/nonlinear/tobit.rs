//! Tobitの推定オプション・結果、およびPython（polars DataFrame + 列名 + オプション）から
//! `engine::nonlinear::tobit`（Newton/BFGS/L-BFGSソルバー・標準誤差・適合度統計量・
//! 限界効果・予測値・打ち切り適合度チェック）を呼び出し、結果をPython側に返すところまでの
//! 一連の処理。基本構成は`logit.rs`/`probit.rs`と同型（`engine_pybind/src/nonlinear/
//! CLAUDE.md`参照）だが、以下の点がTobit固有:
//!
//! - `TobitOptions`は`lower`/`upper`（打ち切り境界）フィールドを追加で持つ
//!   （`nonlinear-api-design.md`7章で確定済みの既定値`lower=Some(0.0)`・`upper=None`）
//! - `TobitResult`の`params`/`param_names`/`std_errors`等は`(k+1)`長に統一する
//!   （`engine::nonlinear::tobit::TobitEstimator::params()`は`β`のみ`k`長だが、
//!   `std_errors()`等は`σ`を含む`k+1`長という非対称な設計になっているため、Python側では
//!   `param_names`の末尾に`"sigma"`を追加して`params`に`sigma()`の値を追加し、全フィールドを
//!   `(k+1)`長に揃えてzipできるようにする。ユーザー確認済み）
//! - `predict()`/`marginal_effects()`は`target`引数（`"expected_latent"`/
//!   `"expected_observed"`/`"prob_uncensored"`、Rust側`MarginalEffectsTarget`の
//!   snake_case版、ユーザー確認済み）を追加で受け取る
//! - `pred_table()`の代わりに`censoring_fit_check()`を提供する（`y`が連続変数のため
//!   分類の的中表は意味を持たない、`nonlinear-api-design.md`6章）
//! - `log_likelihood_null`/`lr_statistic`/`lr_p_value`/`pseudo_r_squared`は無い
//!   （Tobitはこの共通コアから意図的に外れる、`nonlinear-api-design.md`5章）
//!
//! 【責務分離】【言語方針】は`logit.rs`のモジュールdocコメントと同じ
//! （`.claude/rules/rust-style.md`参照）。
//!
//! `build_tobit_input`が`PyDataFrame`ではなく`polars::DataFrame`を受け取る設計にしている
//! 理由も`logit.rs`と同じ（GILなしで`cargo test`から直接ユニットテストできるようにするため）。

use engine::nonlinear::common::{CovType as EngineCovType, Method as EngineMethod};
use engine::nonlinear::tobit::{
    CensoringFitCategory, CensoringFitCheck, MarginalEffectsTarget, TobitEstimator, TobitInput,
};
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use super::common::{MarginalEffectsResult, mle_error_to_pyerr, parse_marginal_effects_at};
use crate::column_extraction::{extract_f64_column, extract_group_key_column};
use crate::errors::ValidationError;
use crate::validation::{
    RoleValue, validate_no_const_collision, validate_no_duplicate_roles,
    validate_no_duplicate_within_role, validate_x_non_empty,
};

/// Estimation options for Tobit.
///
/// See `docs/planning/specs/nonlinear-api-design.md` and
/// `docs/planning/specs/nonlinear-implementation-notes.md` for the rationale behind
/// each field's meaning and default value.
// `LogitOptions`と同じ理由（pyo3 0.28以降のFromPyObject自動導出の仕様変更、
// mkdocstringsのalias解決対応）で`from_py_object` + `module`を明示する。
#[pyclass(from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct TobitOptions {
    /// Standard error type: one of "classical" (alias "nonrobust"), "opg", "hc0",
    /// "hc1", "cluster". Case-insensitive.
    #[pyo3(get, set)]
    pub cov_type: String,

    /// Whether the engine should automatically add an intercept column.
    /// When true, a column of all 1.0 is prepended to the design matrix.
    /// If the user's `x` already contains a constant column while this is true,
    /// the resulting perfect collinearity raises `ComputationError` (singular
    /// design matrix).
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

    /// Lower censoring bound. `None` means "no censoring from below". Defaults to
    /// `0.0` (the standard left-censored-at-zero Tobit model). At least one of
    /// `lower`/`upper` must be set (both `None` raises `ValidationError`).
    #[pyo3(get, set)]
    pub lower: Option<f64>,

    /// Upper censoring bound. `None` (default) means "no censoring from above".
    #[pyo3(get, set)]
    pub upper: Option<f64>,
}

#[pymethods]
impl TobitOptions {
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
        lower = Some(0.0),
        upper = None,
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
        lower: Option<f64>,
        upper: Option<f64>,
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
            lower,
            upper,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TobitOptions(cov_type={:?}, include_intercept={}, confidence_level={}, \
             cluster_col={:?}, method={:?}, max_iter={}, tol={}, raise_on_non_convergence={}, \
             lower={:?}, upper={:?})",
            self.cov_type,
            self.include_intercept,
            self.confidence_level,
            self.cluster_col,
            self.method,
            self.max_iter,
            self.tol,
            self.raise_on_non_convergence,
            self.lower,
            self.upper
        )
    }
}

/// Estimation results for Tobit.
///
/// Structured data only (no `summary()`); see `docs/planning/specs/nonlinear-api-design.md`
/// section 5. `predict()` / `marginal_effects()` / `censoring_fit_check()` are provided as
/// separate methods (not part of this struct's fields), matching section 6.
///
/// `params`/`param_names`/`std_errors`/`z_stats`/`p_values`/`conf_lower`/`conf_upper` are all
/// `(k+1)`-length: `sigma` (the error term's standard deviation) is appended as the last
/// element, with `param_names[-1] == "sigma"`. This differs from the underlying
/// `engine::nonlinear::tobit::TobitEstimator`, where `params()` is `k`-length (`beta` only)
/// while the standard-error-related fields are `(k+1)`-length (an asymmetry stemming from
/// `cov_params` being a `(k+1)x(k+1)` matrix over the joint `(beta, sigma)` space). This
/// struct resolves that asymmetry so every array-valued field can be zipped together
/// (confirmed with the user). `sigma` is a convenience shortcut equal to `params[-1]`.
///
/// `log_likelihood_null`/`lr_statistic`/`lr_p_value`/`pseudo_r_squared` are not provided for
/// Tobit (no closed form exists under censoring; `wald_statistic`/`wald_p_value` provide the
/// overall model significance test instead, see `nonlinear-api-design.md` section 5).
// `LogitResult`と同じ理由で`skip_from_py_object`・フィールドごとの個別`#[pyo3(get)]`・
// `Clone`非導出（`estimator: TobitEstimator`が`Clone`を実装していないため）。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug)]
pub struct TobitResult {
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
    /// Point estimate of `sigma` (the error term's standard deviation). Equal to
    /// `params[-1]`.
    #[pyo3(get)]
    pub sigma: f64,
    #[pyo3(get)]
    pub log_likelihood: f64,
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
    /// Wald test statistic for overall model significance (all slope coefficients
    /// jointly zero). `NaN` when `df_model == 0` (no slope coefficients to test).
    #[pyo3(get)]
    pub wald_statistic: f64,
    /// p-value for `wald_statistic` (chi-squared distribution with `df_model`
    /// degrees of freedom). `NaN` when `df_model == 0`.
    #[pyo3(get)]
    pub wald_p_value: f64,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub n_iter: usize,
    /// Standard error type actually used (echoes `TobitOptions.cov_type`, normalized
    /// to lowercase).
    #[pyo3(get)]
    pub cov_type: String,
    /// Lower censoring bound actually used (echoes `TobitOptions.lower`).
    #[pyo3(get)]
    pub lower: Option<f64>,
    /// Upper censoring bound actually used (echoes `TobitOptions.upper`).
    #[pyo3(get)]
    pub upper: Option<f64>,
    /// Not exposed to Python; only `predict`/`marginal_effects`/`censoring_fit_check`
    /// read it (`LogitResult`の`estimator`と同じ位置づけ)。
    estimator: TobitEstimator,
}

#[pymethods]
impl TobitResult {
    /// Predicted values for the training data used in `fit()`.
    ///
    /// `target` selects which quantity to predict: `"expected_latent"` (`E[y*|x]=x'β`),
    /// `"expected_observed"` (`E[y|x]`, the censoring-adjusted conditional expectation;
    /// default, directly comparable to the observed `y`), or `"prob_uncensored"`
    /// (`P(uncensored|x)`).
    ///
    /// Out-of-sample prediction (a `new_data` argument) is not yet supported (same
    /// limitation as Logit/Probit's `predict()`).
    ///
    /// # Errors
    /// `target` is not one of the three known values (case-insensitive): `ValidationError`
    #[pyo3(signature = (target="expected_observed".to_string()))]
    fn predict(&self, target: String) -> PyResult<Vec<f64>> {
        let target = parse_marginal_effects_target(&target.to_lowercase())?;
        Ok(self.estimator.predict(target))
    }

    /// Marginal effects (`dy/dx`) with delta-method standard errors.
    ///
    /// `target` selects the same three quantities as `predict()` (see its doc). Unlike
    /// Logit/Probit, this is an independent implementation (not the shared
    /// `dydx_and_jacobian` pattern) because the formula differs per `target`
    /// (`nonlinear-api-design.md` section 6, Issue #211's conclusion).
    ///
    /// Independent of `fit()`'s `confidence_level` (re-evaluated here so callers can
    /// use a different confidence level without re-fitting).
    ///
    /// # Errors
    /// - `at` is not one of `"overall"`, `"mean"`, `"median"` (case-insensitive):
    ///   `ValidationError`
    /// - `target` is not one of the three known values (case-insensitive):
    ///   `ValidationError`
    /// - `confidence_level` is outside `(0, 1)`: `ValidationError`
    #[pyo3(signature = (
        at="overall".to_string(),
        target="expected_observed".to_string(),
        confidence_level=0.95,
    ))]
    fn marginal_effects(
        &self,
        at: String,
        target: String,
        confidence_level: f64,
    ) -> PyResult<MarginalEffectsResult> {
        let at = parse_marginal_effects_at(&at.to_lowercase())?;
        let target = parse_marginal_effects_target(&target.to_lowercase())?;
        let effects = self
            .estimator
            .marginal_effects(at, target, confidence_level)
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

    /// Censoring goodness-of-fit check: for each direction (`lower`/`uncensored`/`upper`)
    /// that applies to this model, compares the observed rate (fraction of training
    /// observations exactly at that boundary) against the model-implied average
    /// probability. Replaces Logit/Probit's `pred_table()` (which is not meaningful for
    /// Tobit's continuous `y`, `nonlinear-api-design.md` section 6).
    ///
    /// `lower`/`upper` in the returned `CensoringFitCheckResult` are `None` when the
    /// corresponding `TobitOptions.lower`/`upper` was `None` (that direction has no
    /// censoring).
    fn censoring_fit_check(&self) -> CensoringFitCheckResult {
        censoring_fit_check_to_result(self.estimator.censoring_fit_check())
    }
}

/// Censoring goodness-of-fit result for a single category (below the lower bound,
/// uncensored, or above the upper bound). See `TobitResult.censoring_fit_check`.
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone, Copy)]
pub struct CensoringFitCategoryResult {
    /// Fraction of training observations exactly at this category's boundary
    /// (for `uncensored`, the fraction strictly between the bounds).
    #[pyo3(get)]
    pub observed_rate: f64,
    /// Model-implied average probability of falling into this category.
    #[pyo3(get)]
    pub model_implied_rate: f64,
}

/// Censoring goodness-of-fit check. See `TobitResult.censoring_fit_check`.
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct CensoringFitCheckResult {
    /// `None` when `TobitOptions.lower` was `None` (no censoring from below).
    #[pyo3(get)]
    pub lower: Option<CensoringFitCategoryResult>,
    #[pyo3(get)]
    pub uncensored: CensoringFitCategoryResult,
    /// `None` when `TobitOptions.upper` was `None` (no censoring from above).
    #[pyo3(get)]
    pub upper: Option<CensoringFitCategoryResult>,
}

fn censoring_fit_category_to_result(category: CensoringFitCategory) -> CensoringFitCategoryResult {
    CensoringFitCategoryResult {
        observed_rate: category.observed_rate(),
        model_implied_rate: category.model_implied_rate(),
    }
}

fn censoring_fit_check_to_result(check: CensoringFitCheck) -> CensoringFitCheckResult {
    CensoringFitCheckResult {
        lower: check.lower().map(censoring_fit_category_to_result),
        uncensored: censoring_fit_category_to_result(check.uncensored()),
        upper: check.upper().map(censoring_fit_category_to_result),
    }
}

/// `cov_type`文字列（大文字小文字を区別しない）を`engine::nonlinear::common::CovType`に
/// パースする。`logit.rs`の`parse_cov_type`と同じロジック（`CovType`はLogit/Probit/Tobit
/// 共有の型のため）だが、モデルファイルごとに独立して定義する既存方針を踏襲する
/// （`probit.rs`も同様に複製している）。
///
/// # Errors
/// `cov_type`が既知の値のいずれでもない: `ValidationError`
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
/// パースする（`logit.rs`と同じロジック、`parse_cov_type`と同じ理由で複製）。
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

/// `target`文字列（大文字小文字を区別しない）を`engine::nonlinear::tobit::
/// MarginalEffectsTarget`にパースする。`predict`/`marginal_effects`共通
/// （Tobit専用、`parse_marginal_effects_at`とは異なりLogit/Probitには無い概念のため
/// `nonlinear/common.rs`ではなくここに定義する）。
///
/// # Errors
/// `target`が既知の値のいずれでもない: `ValidationError`
fn parse_marginal_effects_target(target_lower: &str) -> PyResult<MarginalEffectsTarget> {
    match target_lower {
        "expected_latent" => Ok(MarginalEffectsTarget::ExpectedLatent),
        "expected_observed" => Ok(MarginalEffectsTarget::ExpectedObserved),
        "prob_uncensored" => Ok(MarginalEffectsTarget::ProbUncensored),
        other => Err(ValidationError::new_err(format!(
            "unknown target: '{other}'. Expected one of 'expected_latent', \
             'expected_observed', or 'prob_uncensored'"
        ))),
    }
}

/// `x`に`"sigma"`という列名が含まれていないことを検証する。`fit()`は`TobitResult`の
/// `param_names`/`params`の末尾に合成パラメータ名`"sigma"`（誤差項の標準偏差）を
/// 無条件で追加する設計（`TobitResult`のdocコメント参照）のため、`x`にユーザー指定の
/// `"sigma"`列があると`param_names`に重複した`"sigma"`が現れ、`zip(param_names, params)`
/// のような素朴な利用をした際に`x`の`"sigma"`列に対応する係数がエラーにならず静かに
/// 上書きされる（`validate_no_const_collision`の`"const"`列衝突と同型の問題、
/// rust-reviewer指摘）。
///
/// # Errors
/// `x`に`"sigma"`という列名が含まれる場合は`ValidationError`
fn validate_no_sigma_collision(x: &[String]) -> PyResult<()> {
    if x.iter().any(|name| name == "sigma") {
        return Err(ValidationError::new_err(
            "x cannot contain a column named 'sigma' (it collides with the error term's \
             standard deviation, which TobitResult appends to param_names/params)",
        ));
    }
    Ok(())
}

/// Pythonから渡された `data` / `y` / `x` / `options` を検証し、
/// `engine::nonlinear::tobit::TobitInput::from_columns`を呼び出すところまでを行う。
/// `TobitEstimator::fit`の呼び出し・`TobitResult`の構築は`fit`（本ファイル）が行う。
///
/// # Errors
/// - 列の抽出時に発覚する問題（列が存在しない、数値/文字列型にキャストできない、
///   欠損値・NaN・無限大を含む等）は`column_extraction`の責務で`ValidationError`
/// - `y`・`x`の重複、`include_intercept=true`のときの`"const"`列との衝突、`x`に
///   `"sigma"`という列名がある場合は、ここ（受け口）の責務で`ValidationError`
///   （Logitの`build_logit_input`と同じ役割分担。`"sigma"`衝突はTobit固有、
///   `validate_no_sigma_collision`参照）
/// - `cov_type`/`method`の文字列が不正な場合は`ValidationError`
/// - 打ち切り境界（`options.lower`/`options.upper`）の不正・`y`との不整合は
///   `TobitInput::from_columns`が検出し`mle_error_to_pyerr`で`ValidationError`に変換
/// - それ以外（次元不一致等）は`engine::nonlinear::common::MleError`から
///   `mle_error_to_pyerr`で変換
///
/// Issue #212の結論通り、`validate_binary_y`相当の検証はここでは行わない
/// （`y`は連続変数のため）。打ち切り境界の検証は`TobitInput::from_columns`に委ねる
/// （engine層の責務、`.claude/rules/rust-style.md`「Python境界でのデータ受け渡し」参照）。
pub(crate) fn build_tobit_input(
    df: &DataFrame,
    y: String,
    x: Vec<String>,
    options: &TobitOptions,
) -> PyResult<(TobitInput, EngineCovType, EngineMethod)> {
    let cov_type_lower = options.cov_type.to_lowercase();
    let method_lower = options.method.to_lowercase();

    validate_x_non_empty(&x)?;
    validate_no_duplicate_roles(&[("y", RoleValue::Single(&y)), ("x", RoleValue::Multi(&x))])?;
    validate_no_duplicate_within_role("x", &x)?;
    validate_no_const_collision(&x, options.include_intercept)?;
    validate_no_sigma_collision(&x)?;

    // ── y列の抽出 ──────────────────────────────────────────────────────
    let y_slice = extract_f64_column(df, &y)?;

    // ── x列の抽出 ──────────────────────────────────────────────────────
    let mut x_slices: Vec<Vec<f64>> = Vec::with_capacity(x.len());
    for col_name in &x {
        x_slices.push(extract_f64_column(df, col_name)?);
    }

    let cov_type = parse_cov_type(df, &cov_type_lower, &options.cluster_col)?;
    let method = parse_method(&method_lower)?;

    let input = TobitInput::from_columns(
        &y_slice,
        &x_slices,
        x,
        options.include_intercept,
        y,
        options.lower,
        options.upper,
    )
    .map_err(mle_error_to_pyerr)?;

    Ok((input, cov_type, method))
}

/// Pythonから渡された `data` / `y` / `x` / `options` を検証し、
/// `build_tobit_input`で構築した`TobitInput`に対して`engine::nonlinear::tobit::
/// TobitEstimator::fit`を呼び出してTobitを推定し、`TobitResult`として返す。
///
/// # Errors
/// - `build_tobit_input`が返すエラー（列抽出・y/xの重複・`"const"`列衝突・
///   `cov_type`/`method`文字列の検証・打ち切り境界の検証等）は`ValidationError`
/// - `TobitEstimator::fit`が返す`MleError`（`confidence_level`範囲外・`max_iter`が
///   0以下・観測数不足・非打ち切り観測ゼロ・未収束・特異Hessian・特異OPG行列・
///   特異設計行列・クラスターキー未指定・クラスター数不足等）は`mle_error_to_pyerr`で
///   変換（詳細は`engine::nonlinear::tobit::TobitEstimator::fit`のdocコメント参照）
pub(crate) fn fit(
    data: PyDataFrame,
    y: String,
    x: Vec<String>,
    options: &TobitOptions,
) -> PyResult<TobitResult> {
    let df: DataFrame = data.into();
    let (input, cov_type, method) = build_tobit_input(&df, y, x, options)?;

    let estimator = TobitEstimator::fit(
        input,
        method,
        options.max_iter,
        options.tol,
        options.raise_on_non_convergence,
        cov_type,
        options.confidence_level,
    )
    .map_err(mle_error_to_pyerr)?;

    // `params`/`param_names`を`(k+1)`長（`σ`を含む）に揃える
    // （`TobitResult`のdocコメント「非対称性の解消」参照）。
    let mut params = estimator.params().to_vec();
    params.push(estimator.sigma());
    let mut param_names = estimator.input().param_names().to_vec();
    param_names.push("sigma".to_string());

    Ok(TobitResult {
        params,
        std_errors: estimator.std_errors().to_vec(),
        z_stats: estimator.z_stats().to_vec(),
        p_values: estimator.p_values().to_vec(),
        conf_lower: estimator.conf_lower().to_vec(),
        conf_upper: estimator.conf_upper().to_vec(),
        param_names,
        sigma: estimator.sigma(),
        log_likelihood: estimator.log_likelihood(),
        aic: estimator.aic(),
        bic: estimator.bic(),
        n_obs: estimator.n_obs(),
        df_model: estimator.df_model(),
        df_resid: estimator.df_resid(),
        wald_statistic: estimator.wald_statistic(),
        wald_p_value: estimator.wald_p_value(),
        converged: estimator.converged(),
        n_iter: estimator.n_iter(),
        cov_type: options.cov_type.to_lowercase(),
        lower: options.lower,
        upper: options.upper,
        estimator,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    /// `build_tobit_input`のテスト全体で使う既定の`TobitOptions`（`cov_type="classical"`・
    /// `include_intercept=true`・`method="newton"`・`lower=Some(0.0)`・`upper=None`）。
    /// フィールドごとに上書きして使う。
    fn default_options() -> TobitOptions {
        TobitOptions::new(
            "classical".to_string(),
            true,
            0.95,
            None,
            "newton".to_string(),
            35,
            1e-6,
            true,
            Some(0.0),
            None,
        )
    }

    /// `y`は`lower=0.0`（既定の境界）以上の値のみで構成する（`TobitInput::from_columns`の
    /// 境界整合性検証を通過させるため）。
    fn well_formed_df() -> DataFrame {
        df!(
            "y" => [0.0, 1.0, 2.0, 3.0],
            "x1" => [10.0, 20.0, 30.0, 40.0],
            "x2" => [-5.0, 2.0, 8.0, -1.0],
        )
        .unwrap()
    }

    #[test]
    fn build_tobit_input_succeeds_for_well_formed_data() {
        let df = well_formed_df();
        let options = default_options();

        let Ok((input, cov_type, method)) = build_tobit_input(
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
        assert_eq!(input.lower(), Some(0.0));
        assert_eq!(input.upper(), None);
        assert!(matches!(cov_type, EngineCovType::Classical));
        assert!(matches!(method, EngineMethod::Newton));
    }

    #[test]
    fn build_tobit_input_without_intercept_omits_const_column() {
        let df = well_formed_df();
        let mut options = default_options();
        options.include_intercept = false;

        let Ok((input, _, _)) =
            build_tobit_input(&df, "y".to_string(), vec!["x1".to_string()], &options)
        else {
            panic!("expected Ok");
        };

        assert_eq!(input.k(), 1);
        assert_eq!(input.param_names(), ["x1".to_string()]);
    }

    #[test]
    fn build_tobit_input_returns_validation_error_for_empty_x() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_tobit_input(&df, "y".to_string(), vec![], &options);
        assert!(result.is_err());
    }

    #[test]
    fn build_tobit_input_returns_validation_error_when_y_is_also_in_x() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_tobit_input(
            &df,
            "y".to_string(),
            vec!["y".to_string(), "x1".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_tobit_input_returns_validation_error_for_duplicate_x_column() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_tobit_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string(), "x1".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_tobit_input_returns_validation_error_for_const_collision_when_include_intercept_is_true()
     {
        let df = df!(
            "y" => [0.0, 1.0, 2.0, 3.0],
            "const" => [1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let options = default_options();

        let result = build_tobit_input(&df, "y".to_string(), vec!["const".to_string()], &options);
        assert!(result.is_err());
    }

    /// `x`に`"sigma"`という列名があると、`fit()`が`param_names`の末尾に追加する合成
    /// パラメータ名`"sigma"`と衝突する（`validate_no_sigma_collision`のdocコメント参照、
    /// rust-reviewer指摘）。`"const"`列衝突と同型のテストパターン。
    #[test]
    fn build_tobit_input_returns_validation_error_for_sigma_collision() {
        let df = df!(
            "y" => [0.0, 1.0, 2.0, 3.0],
            "sigma" => [1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let options = default_options();

        let result = build_tobit_input(&df, "y".to_string(), vec!["sigma".to_string()], &options);
        assert!(result.is_err());
    }

    #[test]
    fn build_tobit_input_returns_validation_error_for_missing_column() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_tobit_input(
            &df,
            "y".to_string(),
            vec!["does_not_exist".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_tobit_input_returns_validation_error_for_missing_values() {
        let df = df!(
            "y" => [0.0, 1.0, 2.0, 3.0],
            "x1" => [Some(1.0), None, Some(3.0), Some(4.0)],
        )
        .unwrap();
        let options = default_options();

        let result = build_tobit_input(&df, "y".to_string(), vec!["x1".to_string()], &options);
        assert!(result.is_err());
    }

    #[test]
    fn build_tobit_input_returns_validation_error_for_unknown_cov_type() {
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "bogus".to_string();

        let result = build_tobit_input(&df, "y".to_string(), vec!["x1".to_string()], &options);
        assert!(result.is_err());
    }

    #[test]
    fn build_tobit_input_returns_validation_error_for_unknown_method() {
        let df = well_formed_df();
        let mut options = default_options();
        options.method = "bogus".to_string();

        let result = build_tobit_input(&df, "y".to_string(), vec!["x1".to_string()], &options);
        assert!(result.is_err());
    }

    #[test]
    fn build_tobit_input_parses_cluster_cov_type_with_group_column() {
        let df = df!(
            "y" => [0.0, 1.0, 2.0, 3.0],
            "x1" => [10.0, 20.0, 30.0, 40.0],
            "cluster" => ["g1", "g1", "g2", "g2"],
        )
        .unwrap();
        let mut options = default_options();
        options.cov_type = "cluster".to_string();
        options.cluster_col = Some("cluster".to_string());

        let Ok((_, cov_type, _)) =
            build_tobit_input(&df, "y".to_string(), vec!["x1".to_string()], &options)
        else {
            panic!("expected Ok");
        };

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
    fn build_tobit_input_leaves_cluster_groups_none_when_cluster_col_not_specified() {
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "cluster".to_string();

        let Ok((_, cov_type, _)) =
            build_tobit_input(&df, "y".to_string(), vec!["x1".to_string()], &options)
        else {
            panic!("expected Ok");
        };

        match cov_type {
            EngineCovType::Cluster { groups } => assert!(groups.is_none()),
            other => panic!("expected Cluster, got {other:?}"),
        }
    }

    #[test]
    // `parse_cov_type`自体は前処理済みの小文字文字列を受け取る想定
    // （`parse_cov_type`のdocコメント参照）で、大文字小文字を区別しない処理は
    // 呼び出し元の`build_tobit_input`が`options.cov_type.to_lowercase()`してから渡す
    // ことで実現している。そのため、この不変条件は`parse_cov_type`単体ではなく
    // `build_tobit_input`（実際にPythonから渡される文字列を受ける入口）を通して検証する
    // （Logitの`build_logit_input_cov_type_is_case_insensitive`と同じ理由、Issue #231）。
    fn build_tobit_input_cov_type_is_case_insensitive() {
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
            let Ok((_, cov_type, _)) = build_tobit_input(
                &df,
                "y".to_string(),
                vec!["x1".to_string(), "x2".to_string()],
                &options,
            ) else {
                panic!("expected Ok for cov_type={input}");
            };
            assert!(is_expected(&cov_type), "input={input}, got={cov_type:?}");
        }
    }

    #[test]
    fn build_tobit_input_accepts_nonrobust_as_classical_alias() {
        let df = well_formed_df();
        for input in ["nonrobust", "NONROBUST", "NonRobust"] {
            let mut options = default_options();
            options.cov_type = input.to_string();
            let Ok((_, cov_type, _)) = build_tobit_input(
                &df,
                "y".to_string(),
                vec!["x1".to_string(), "x2".to_string()],
                &options,
            ) else {
                panic!("expected Ok for cov_type={input}");
            };
            assert!(
                matches!(cov_type, EngineCovType::Classical),
                "input={input}"
            );
        }
    }

    #[test]
    fn build_tobit_input_supports_custom_lower_and_upper() {
        let df = df!(
            "y" => [0.0, 1.0, 2.0, 6.0],
            "x1" => [10.0, 20.0, 30.0, 40.0],
        )
        .unwrap();
        let mut options = default_options();
        options.lower = Some(0.0);
        options.upper = Some(6.0);

        let Ok((input, _, _)) =
            build_tobit_input(&df, "y".to_string(), vec!["x1".to_string()], &options)
        else {
            panic!("expected Ok");
        };
        assert_eq!(input.lower(), Some(0.0));
        assert_eq!(input.upper(), Some(6.0));
    }

    #[test]
    fn build_tobit_input_supports_right_censoring_only_when_lower_is_none() {
        let df = well_formed_df();
        let mut options = default_options();
        options.lower = None;
        options.upper = Some(100.0);

        let Ok((input, _, _)) =
            build_tobit_input(&df, "y".to_string(), vec!["x1".to_string()], &options)
        else {
            panic!("expected Ok");
        };
        assert_eq!(input.lower(), None);
        assert_eq!(input.upper(), Some(100.0));
    }

    /// 打ち切り境界の不正（両方`None`）は`TobitInput::from_columns`（engine層）が検出し、
    /// `mle_error_to_pyerr`経由で`ValidationError`になる（`build_tobit_input`のdoc
    /// コメント参照。Issue #212の結論通り、ここでの独自バリデーションは行わない）。
    #[test]
    fn build_tobit_input_returns_validation_error_when_both_bounds_are_none() {
        let df = well_formed_df();
        let mut options = default_options();
        options.lower = None;
        options.upper = None;

        let result = build_tobit_input(&df, "y".to_string(), vec!["x1".to_string()], &options);
        assert!(result.is_err());
    }

    #[test]
    fn parse_marginal_effects_target_accepts_known_lowercase_values() {
        assert!(matches!(
            parse_marginal_effects_target("expected_latent"),
            Ok(MarginalEffectsTarget::ExpectedLatent)
        ));
        assert!(matches!(
            parse_marginal_effects_target("expected_observed"),
            Ok(MarginalEffectsTarget::ExpectedObserved)
        ));
        assert!(matches!(
            parse_marginal_effects_target("prob_uncensored"),
            Ok(MarginalEffectsTarget::ProbUncensored)
        ));
    }

    #[test]
    fn parse_marginal_effects_target_returns_validation_error_for_unknown_value() {
        assert!(parse_marginal_effects_target("bogus").is_err());
    }

    /// `censoring_fit_category_to_result`/`censoring_fit_check_to_result`（`CensoringFitCategory`/
    /// `CensoringFitCheck`のprivateフィールドをPython向けpyclassに詰め替えるだけの変換）は
    /// これまで検証されていなかった（フィールドの取り違え・`Option`ラップ漏れ等を検出でき
    /// ない状態だった、rust-reviewer指摘）。`TobitEstimator::fit`は`PyDataFrame`を経由しない
    /// ため（`engine_pybind/src/nonlinear/CLAUDE.md`の既知の制約通り、`fit`本体はGILが
    /// 必要な`PyDataFrame`引数のため`cargo test`から直接呼べないが、`TobitEstimator::fit`
    /// 自体はengine層の関数でGIL不要）、`build_tobit_input`が返す`TobitInput`にそのまま
    /// 適用してこの変換ロジックを独立に検証する。
    #[test]
    fn censoring_fit_check_to_result_correctly_maps_categories_and_fields() {
        let df = df!(
            "y" => [0.0, 0.0, 1.15, 2.9, 5.2, 6.85, 9.1, 10.95],
            "x1" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let options = default_options();
        let (input, cov_type, method) =
            build_tobit_input(&df, "y".to_string(), vec!["x1".to_string()], &options)
                .expect("expected Ok");
        let estimator = TobitEstimator::fit(
            input,
            method,
            options.max_iter,
            options.tol,
            options.raise_on_non_convergence,
            cov_type,
            options.confidence_level,
        )
        .expect("expected converged fit");

        let engine_check = estimator.censoring_fit_check();
        let result = censoring_fit_check_to_result(engine_check);

        assert!(result.upper.is_none());
        let lower_result = result.lower.expect("lower category should be Some");
        let engine_lower = engine_check.lower().expect("engine lower should be Some");
        assert_eq!(lower_result.observed_rate, engine_lower.observed_rate());
        assert_eq!(
            lower_result.model_implied_rate,
            engine_lower.model_implied_rate()
        );
        assert_eq!(
            result.uncensored.observed_rate,
            engine_check.uncensored().observed_rate()
        );
        assert_eq!(
            result.uncensored.model_implied_rate,
            engine_check.uncensored().model_implied_rate()
        );
    }
}
