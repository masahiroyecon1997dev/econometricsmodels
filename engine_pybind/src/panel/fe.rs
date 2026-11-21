//! FEの推定オプション・結果、およびPython（polars DataFrame + 列名 + オプション）から
//! `engine::panel::fe`（within変換・パネル自由度調整・`cov_type`対応・パネル固有R²）を
//! 呼び出すところまでの一連の処理（データ抽出・pyclass定義、`FeEstimator::fit`への
//! 配線、`fixed_effects()`メソッド）。
//!
//! 【責務分離】`.claude/rules/rust-style.md`「Python境界でのデータ受け渡し」参照。
//! polars DataFrameから列ごとの`Vec<f64>`/`Vec<String>`への抽出はここ（`column_extraction`
//! 経由）の責務。`faer::Mat`の組み立ては`engine`側（`FeInput`は`Mat`を組み立てない設計、
//! `engine/src/panel/fe.rs`モジュールdoc参照）に委ねる。
//!
//! 【言語方針】`.claude/rules/rust-style.md`「言語方針」参照。
//! 公開API（`FEOptions`/`FEResult`）のdocコメントと、`ValidationError`のメッセージ文字列は
//! 英語。それ以外（このファイルの説明・非公開関数のdocコメント等）は日本語のまま。
//!
//! ## `effects`と`FEOptions.dk_time`は別物（`panel-common.md`1.1節）
//!
//! 固定効果の次元は必須引数`effects`（`list[str]`、1〜5要素）で指定する。1要素が1-way、
//! 2要素が2-way（`effects[0]`が個体、`effects[1]`が時点）、3要素以上が高次元（3つ目以降は同一性だけの
//! キー）。`dk_time`（`cov_type="dk"`の値を
//! 接頭辞にした命名）はDriscoll-Kraay型HAC（`cov_type="dk"`）専用の時点列で、`effects`とは
//! 独立に指定できる。**`cov_type="dk"`では`dk_time`が必須**で、`effects`の列を暗黙に借用しない
//! （どの列がDKの時点かを明示させ、意図と違う列が選ばれても気づけない状況を避けるため。
//! `dk_time`未指定は`ValidationError`、`parse_panel_cov_kind`参照）。2-wayでも`dk_time`が
//! 常にDK計算に使われ、`effects[1]`と別の時間粒度（四半期の固定効果に年次のDK等）も指定できる
//! （詳細な経緯・engine側の対応する変更は`engine/src/panel/fe.rs`モジュールdoc
//! 「Driscoll-Kraay型パネルHAC対応」参照）。
//!
//! ## `cov_type`の非対応値
//!
//! FEは`hc0`を**サポートしない**（仕様判断で、他パッケージの有無は理由にしない。
//! `FeCovType` enumから除外済み）。OLS/WLS/IVとは
//! 異なりFEの`cov_type`文字列パースはこの1点で分岐が異なるため、`linear::common::
//! parse_cov_kind`を流用せず、FE/RE共通の`panel::common::parse_panel_cov_kind`を使う
//! （OLS/WLS/IVは`linear::common`を共有する。`hc0`非対応・`Dk`の意味論がパネル固有のため）。
//!
//! ## `x`の空リストを許容しない
//!
//! 固定効果のみのモデル（`x=[]`）は、`validate_role_non_empty`で拒否する（他手法のOLS/WLS/Logit/
//! Probit/IVと同じ）。`x`が空だと、説明変数が`y`に与える効果を推定するという営みが成立せず、
//! 実質「個体・時間固定効果による`y`の分解」という別の操作になるため。**`engine`側
//! （`FeInput::from_group_codes`・`FeEstimator::fit`）はk=0を受理したまま**で、
//! `engine/src/panel/fe/fit/tests/basic.rs`の`fe_estimator_fit_with_no_regressors_estimates_fixed_effects_only_model`が
//! engineレベルでは固定効果のみのモデルが動作することを検証する。拒否はPython向けAPI
//! （`engine_pybind`）の業務的なバリデーションとしてのみ行う（OLSも
//! `engine::linear::ols::OlsEstimator::fit`自体はk=0を受理するが、Python APIの
//! `validate_role_non_empty`が弾くのと同じ構図）。

use crate::shared::errors::{guard, internal_error};
use engine::panel::fe::{FeAbsorption, FeCovType, FeEffects, FeEstimator, FeInput, FixedEffects};
use polars::prelude::DataFrame;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use super::common::{
    PanelCovKind, PanelModel, check_panel_cov_options, extract_effect_groups, extract_time_keys,
    groups_dict, panel_error_to_pyerr, parse_panel_cov_kind, require_dk_time, validate_panel_roles,
};
use crate::shared::column_error::ColumnSource;
use crate::shared::column_extraction::{
    SelectedData, extract_f64_column, extract_f64_columns, extract_optional_cluster_groups,
};
use crate::shared::mat_conversion::{mat_to_rows, mat_to_vec};
use crate::shared::option_repr::PyRepr;
use crate::shared::option_values::{
    extract_strict_float, extract_strict_int, extract_strict_opt_column, extract_strict_opt_int,
    extract_strict_text,
};

/// Estimation options for FE (fixed effects panel regression).
///
/// All arguments are keyword-only.
///
/// Options by purpose:
///
/// - Standard errors: `cov_type`; with `"cluster"` also `cluster`; with
///   `"dk"` also `dk_time` (required) and `dk_bandwidth`.
/// - Confidence intervals: `confidence_level`.
/// - Removing the fixed effects of an unbalanced two-way model or of a model with
///   three or more effects: `tol`, `max_iter`.
// 各フィールドの意味・既定値の根拠は`docs/spec/panel-common.md`。
// module/from_py_objectの理由は`OLSOptions`と同じ（`engine_pybind/src/linear/ols/options.rs`参照）。
#[pyclass(from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct FEOptions {
    /// Standard error type: one of "classical", "hc1", "hc2", "hc3", "cluster", "dk".
    /// Case-insensitive. Unlike OLS/WLS/IV, "hc0" is **not** supported.
    #[pyo3(get)]
    pub cov_type: String,

    /// Confidence level for confidence intervals, in the range (0, 1).
    /// Defaults to 0.95 (a 95% confidence interval).
    #[pyo3(get)]
    pub confidence_level: f64,

    /// Column name to use as the cluster group key when `cov_type="cluster"`. When
    /// `None`, the first column of `effects` is used automatically. Specifying it with
    /// any other `cov_type` raises `ValidationError`.
    #[pyo3(get)]
    pub cluster: Option<String>,

    /// Column name that defines the time periods of the Driscoll-Kraay HAC, independent
    /// of `effects` (`effects` defines the fixed-effects structure, `dk_time` the HAC
    /// periods). Required when `cov_type="dk"`, whatever the number of columns in `effects`:
    /// leaving it unset raises `ValidationError`, and the columns of `effects` are not
    /// used instead. With two-way effects the HAC may use a different time granularity
    /// from the fixed effects. It may not be the same column as `y` or the first column
    /// of `effects` (`ValidationError`), but may be one of the regressors or the second
    /// column of `effects`. Specifying it with any other `cov_type` raises
    /// `ValidationError`. The periods are ordered by the values of the column:
    /// numerically for integers and floats, chronologically for `Date` and `Datetime`, in
    /// the order of the categories for `Enum`, and alphabetically for strings and
    /// `Categorical`.
    #[pyo3(get)]
    pub dk_time: Option<String>,

    /// Bandwidth for Driscoll-Kraay HAC when `cov_type="dk"`. When `None`, computed
    /// automatically via `floor(4*(t/100)^(2/9))` (`t` = number of unique time periods).
    /// Specifying it with any other `cov_type` raises `ValidationError`.
    #[pyo3(get)]
    pub dk_bandwidth: Option<i64>,

    /// Convergence tolerance of the alternating projections that remove the fixed
    /// effects of a two-way model on an **unbalanced** panel, or of a model with three or
    /// more effects (a balanced two-way panel and a one-way model are computed in closed
    /// form, so they ignore it). A pass stops
    /// when the largest group mean it subtracts is at most `tol` times the size of the
    /// column, so the tolerance is relative to the within-group variation. Must be a
    /// positive finite number. Defaults to 1e-12. A loose `tol` also coarsens the
    /// detection of regressors that are collinear with the fixed effects.
    #[pyo3(get)]
    pub tol: f64,

    /// Maximum number of passes of the alternating projections (see `tol`), from 1 to
    /// 10000. Defaults to 10000. A panel whose groups are only thinly connected needs
    /// many passes; if the limit is reached, `ComputationError` is
    /// raised.
    #[pyo3(get)]
    pub max_iter: i64,
}

crate::shared::pickle::impl_pickle_state!(FEOptions {
    cov_type,
    confidence_level,
    cluster,
    dk_time,
    dk_bandwidth,
    tol,
    max_iter,
});

impl FEOptions {
    /// このオプションが指名するデータの列（`cluster`・`dk_time`）。使われるかどうか（`cov_type`等）には
    /// よらず、指名されたものを全て返す。`fit()`はこれらと`y`・`x`等だけをデータから取り込む。
    pub(crate) fn data_columns(&self) -> Vec<String> {
        self.cluster.iter().chain(&self.dk_time).cloned().collect()
    }
}

#[pymethods]
impl FEOptions {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        crate::shared::pickle::reduce(slf)
    }

    #[new]
    #[pyo3(signature = (
        *,
        cov_type = "cluster".to_string(),
        confidence_level = 0.95,
        cluster = None,
        dk_time = None,
        dk_bandwidth = None,
        tol = None,
        max_iter = 10000,
    ))]
    fn new(
        #[pyo3(from_py_with = crate::shared::option_values::cov_type_arg)] cov_type: String,
        #[pyo3(from_py_with = crate::shared::option_values::confidence_level_arg)] confidence_level: f64,
        #[pyo3(from_py_with = crate::shared::option_values::cluster_arg)] cluster: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::dk_time_arg)] dk_time: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::dk_bandwidth_arg)] dk_bandwidth: Option<
            i64,
        >,
        #[pyo3(from_py_with = crate::shared::option_values::tol_arg)] tol: Option<f64>,
        #[pyo3(from_py_with = crate::shared::option_values::max_iter_arg)] max_iter: i64,
    ) -> Self {
        let defaults = FeAbsorption::default();
        Self {
            cov_type,
            confidence_level,
            cluster,
            dk_time,
            dk_bandwidth,
            tol: tol.unwrap_or(defaults.tol),
            max_iter,
        }
    }

    #[setter]
    fn set_confidence_level(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.confidence_level = extract_strict_float(value, "confidence_level")?;
        Ok(())
    }

    #[setter]
    fn set_dk_bandwidth(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.dk_bandwidth = extract_strict_opt_int(value, "dk_bandwidth")?;
        Ok(())
    }

    #[setter]
    fn set_tol(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.tol = extract_strict_float(value, "tol")?;
        Ok(())
    }

    #[setter]
    fn set_max_iter(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.max_iter = extract_strict_int(value, "max_iter")?;
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
    fn set_dk_time(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.dk_time = extract_strict_opt_column(value, "dk_time")?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "FEOptions(cov_type={}, confidence_level={}, cluster={}, \
             dk_time={}, dk_bandwidth={}, tol={}, max_iter={})",
            self.cov_type.py_repr(),
            self.confidence_level.py_repr(),
            self.cluster.py_repr(),
            self.dk_time.py_repr(),
            self.dk_bandwidth.py_repr(),
            self.tol.py_repr(),
            self.max_iter.py_repr()
        )
    }
}

/// Estimation results for FE.
///
/// Structured data only (no `summary()`). All array-valued fields (`params`,
/// `std_errors`, etc.) share the same order as `param_names`.
///
/// The fixed effects themselves (`α_i`/`γ_t`) are not a field: they are computed on
/// demand by the method `fixed_effects()`.
// `FEResult`はRust側で組み立ててPythonに返すだけの型で、Python側からの生成・引数として
// 受け取ることは想定していないため`skip_from_py_object`（`OLSResult`と同じ理由）。
//
// `Clone`は、`fixed_effects`フィールドが`FeEstimator`（`Clone`未実装）を`Arc`で共有するので派生できる
// （複製は推定量と計算済みの固定効果を共有する。FEIVの`first_stage()`が結果を複製して返すために使う）。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct FEResult {
    #[pyo3(get)]
    pub params: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    /// Covariance matrix of the coefficients under `cov_type`, one row per entry of
    /// `param_names` (same order on both axes).
    #[pyo3(get)]
    pub cov_params: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub test_stats: Vec<f64>,
    /// Distribution of `test_stats`; always `"t"` for FE.
    #[pyo3(get)]
    pub stat_dist: String,
    /// Degrees of freedom of the t distribution (always `Some`). May differ from
    /// `df_resid`: cluster-robust inference uses `G - 1`, where `G` is the number of
    /// clusters, and Driscoll-Kraay uses the number of `dk_time` periods minus 1.
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
    pub residuals: Vec<f64>,
    #[pyo3(get)]
    pub dep_var_name: String,
    #[pyo3(get)]
    pub n_obs: usize,
    #[pyo3(get)]
    pub df_resid: usize,
    #[pyo3(get)]
    pub df_model: usize,
    /// The columns of `effects`, in the order they were given.
    effects: Vec<String>,
    /// Number of groups (unique values) of each column of `effects`, in the same order.
    group_counts: Vec<usize>,
    /// Standard error type actually used (echoes `FEOptions.cov_type`, normalized to
    /// lowercase; e.g. "classical", "hc1", "cluster", "dk").
    #[pyo3(get)]
    pub cov_type: String,
    /// Driscoll-Kraay bandwidth actually used: the explicit `dk_bandwidth` if given, otherwise
    /// the value chosen automatically, `floor(4 * (t / 100) ^ (2 / 9))` where `t` is the number
    /// of unique time periods in `dk_time`. `None` unless `cov_type="dk"`.
    #[pyo3(get)]
    pub dk_bandwidth_used: Option<i64>,
    /// Overall F-test statistic of the slope coefficients. `None` when the test is not
    /// defined: there are no slope coefficients.
    #[pyo3(get)]
    pub f_statistic: Option<f64>,
    /// P-value of `f_statistic` (`None` exactly when `f_statistic` is `None`).
    #[pyo3(get)]
    pub f_p_value: Option<f64>,
    /// Numerator degrees of freedom of `f_statistic` (`None` when `f_statistic` is `None`).
    #[pyo3(get)]
    pub f_df_num: Option<usize>,
    /// Denominator degrees of freedom of `f_statistic` (`None` when `f_statistic` is `None`).
    /// Follows `cov_type` like the t-tests, so it equals `stat_df` whenever `f_statistic` is
    /// not `None`: the number of clusters minus 1 for `"cluster"`, the number of `dk_time`
    /// periods minus 1 for `"dk"`, and `df_resid` otherwise. Use it to recompute the p-value
    /// from `f_statistic`.
    #[pyo3(get)]
    pub f_df_denom: Option<usize>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    #[pyo3(get)]
    pub r_squared_within: f64,
    #[pyo3(get)]
    pub r_squared_between: f64,
    #[pyo3(get)]
    pub r_squared_overall: f64,
    /// `fixed_effects()`が読む。Python側には公開しない（`OLSResult`の`fitted_values`/
    /// `has_intercept`と同じ位置づけ）。固定効果は初回の`fixed_effects()`かpickle時まで計算を遅らせる。
    fixed_effects: LazyFixedEffects,
}

/// 固定効果（`α_i`・`γ_t`）。`fit()`では計算せず（エンティティ数に比例する辞書を作るため、使わない人に
/// 負担をかけない）、`fixed_effects()`かpickleで初めて必要になったときに`FeEstimator`から計算して保持する。
///
/// pickleで復元した結果は`FeEstimator`を持たず、計算済みの固定効果だけを持つ（`Computed`）。`FeEstimator`は
/// 設計行列とグループコードを持つが、固定効果の計算以外には使わない。「推定量が無く、計算済みでもない」
/// 状態は型として作れない。
///
/// 複製は推定量と計算済みの固定効果を共有する（`Arc`）。
#[derive(Debug, Clone)]
pub(crate) enum LazyFixedEffects {
    Lazy {
        estimator: Arc<FeEstimator>,
        cell: Arc<OnceLock<FixedEffects>>,
    },
    Computed(FixedEffects),
}

impl LazyFixedEffects {
    fn new(estimator: FeEstimator) -> Self {
        Self::Lazy {
            estimator: Arc::new(estimator),
            cell: Arc::new(OnceLock::new()),
        }
    }

    /// 固定効果。未計算なら`FeEstimator`から計算して保持する。
    fn get(&self) -> &FixedEffects {
        match self {
            Self::Lazy { estimator, cell } => cell.get_or_init(|| estimator.fixed_effects()),
            Self::Computed(effects) => effects,
        }
    }
}

// 状態は`("one_way", {id: α})`・`("two_way", {id: α}, [(period, γ), ...])`・
// `("multi_way", [[(id, 効果), ...], ...])`（次元ごと、`effects`の順）。時点は時間順の並びを保つ。
impl crate::shared::pickle::PickleField for LazyFixedEffects {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.get() {
            FixedEffects::OneWay(effects) => ("one_way", effects).into_bound_py_any(py),
            FixedEffects::TwoWay { entity, time } => {
                ("two_way", entity, time).into_bound_py_any(py)
            }
            FixedEffects::MultiWay(dimensions) => ("multi_way", dimensions).into_bound_py_any(py),
        }
    }

    fn from_py(obj: &Bound<'_, PyAny>) -> Option<Self> {
        let effects = if let Ok((tag, effects)) = obj.extract::<(String, BTreeMap<String, f64>)>() {
            (tag == "one_way").then_some(FixedEffects::OneWay(effects))?
        } else if let Ok((tag, dimensions)) = obj.extract::<(String, Vec<Vec<(String, f64)>>)>() {
            (tag == "multi_way").then_some(FixedEffects::MultiWay(dimensions))?
        } else {
            let (tag, entity, time) = obj
                .extract::<(String, BTreeMap<String, f64>, Vec<(String, f64)>)>()
                .ok()?;
            (tag == "two_way").then_some(FixedEffects::TwoWay { entity, time })?
        };
        Some(Self::Computed(effects))
    }
}

crate::shared::pickle::impl_pickle_state!(FEResult {
    params,
    std_errors,
    cov_params,
    test_stats,
    stat_dist,
    stat_df,
    p_values,
    conf_lower,
    conf_upper,
    param_names,
    residuals,
    dep_var_name,
    n_obs,
    df_resid,
    df_model,
    effects,
    group_counts,
    cov_type,
    dk_bandwidth_used,
    f_statistic,
    f_p_value,
    f_df_num,
    f_df_denom,
    log_likelihood,
    aic,
    bic,
    r_squared_within,
    r_squared_between,
    r_squared_overall,
    fixed_effects,
});

#[pymethods]
impl FEResult {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        crate::shared::pickle::reduce(slf)
    }

    /// Number of groups (unique values) of each column of `effects`, as a `dict[str, int]`
    /// keyed by column name, in the order the columns were given. One entry for a one-way
    /// model, two for a two-way model, and so on.
    #[getter]
    fn n_groups<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        groups_dict(py, &self.effects, &self.group_counts)
    }

    /// The fixed effects themselves (`α_i` for the first column of `effects`, `γ_t` for the
    /// second, and one effect per group of each further column), recovered post-hoc from
    /// the fitted coefficients (`α_i = ȳ_i - x̄_i'β̂` for one-way effects).
    ///
    /// Always a `dict[str, dict[str, float]]` keyed by the column names in `effects`, then
    /// by group (the value of the column as a string): one entry per column of `effects`.
    ///
    /// Two-way normalization: `α_i`/`γ_t` are not individually identified (adding a
    /// constant to one and subtracting it from the other leaves `α_i + γ_t`, and
    /// therefore the fitted values, unchanged). This implementation fixes the
    /// reference time period to `γ_{t_ref} = 0`, where `t_ref` is the first period in
    /// time order (the order of the values of the time column: numeric for integer and
    /// float columns, chronological for `Date`/`Datetime`, the category order for
    /// `Enum`, alphabetical for strings and `Categorical`) — a deterministic convention
    /// independent of row order. The dictionary of the second column lists the periods in
    /// the same order. With three or more columns, or an unbalanced panel whose groups
    /// form several connected sets, one reference group per connected set is fixed to 0
    /// in every column after the first (the first period in time order for the second
    /// column, the first label in sorted order for the others), and only the sum of the
    /// effects of a row is identified when the groups of one column lie inside those of
    /// another. `fixest::fixef()` chooses its own reference levels, so its values for the
    /// time effects can differ from these by a constant.
    fn fixed_effects(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        guard(|| fixed_effects_dict(py, &self.effects, self.fixed_effects.get()))
    }
}

/// 固定効果を、列名をキーにした入れ子の`dict`（`{列名: {水準: 効果}}`、`effects`の順）にする。
///
/// 1-wayは1項目、2-wayは2項目で、時点は時間順に並べる。`effects`の長さと固定効果の1-way/2-way/高次元は
/// `fit`が揃えて作る。食い違う状態（壊れたpickle等）は`INTERNAL_ERROR`にする。
pub(crate) fn fixed_effects_dict(
    py: Python<'_>,
    effects: &[String],
    fixed_effects: &FixedEffects,
) -> PyResult<Py<PyDict>> {
    let first = effects
        .first()
        .ok_or_else(|| internal_error("the effects columns are missing"))?;
    let outer = PyDict::new(py);
    match fixed_effects {
        FixedEffects::OneWay(entity) => {
            outer.set_item(first, entity)?;
        }
        FixedEffects::TwoWay { entity, time } => {
            let second = effects.get(1).ok_or_else(|| {
                internal_error("two-way fixed effects without a second effects column")
            })?;
            // 時点は時間順に並べた`dict`にする。
            let time_effects = PyDict::new(py);
            for (period, effect) in time {
                time_effects.set_item(period, effect)?;
            }
            outer.set_item(first, entity)?;
            outer.set_item(second, time_effects)?;
        }
        FixedEffects::MultiWay(dimensions) => {
            if dimensions.len() != effects.len() {
                return Err(internal_error(
                    "the fixed effects do not match the effects columns",
                ));
            }
            for (column, levels) in effects.iter().zip(dimensions) {
                // 水準は次元ごとの順（個体・3つ目以降は辞書順、時点は時間順）の`dict`にする。
                let level_effects = PyDict::new(py);
                for (level, effect) in levels {
                    level_effects.set_item(level, effect)?;
                }
                outer.set_item(column, level_effects)?;
            }
        }
    }
    Ok(outer.unbind())
}

/// `cov_type`が使う列とバンド幅（FEとFEIVのオプションから取り出す）。
#[derive(Clone, Copy)]
pub(crate) struct CovColumns<'a> {
    pub(crate) cluster: Option<&'a str>,
    pub(crate) dk_time: Option<&'a str>,
    pub(crate) dk_bandwidth: Option<i64>,
}

/// [`PanelCovKind`]に対応する列（`cluster`・`dk_time`）をDataFrameから抽出して
/// `engine::panel::fe::FeCovType`を組み立てる。
///
/// `linear::common::resolve_cov_type`（OLS/WLS用）を流用しない理由はモジュールdoc
/// 「`cov_type`の非対応値」参照（`hc0`が無い・`Dk`の時点列の上書きがFE固有のため）。
/// 使われないオプションの検査は`check_panel_cov_options`の責務。
///
/// # Errors
/// 列の抽出時に発覚する問題は`column_extraction`の責務で`ValidationError`。`Dk`で
/// `dk_time`が未指定の場合も`ValidationError`（`check_panel_cov_options`を通していれば
/// 起きない）。
pub(crate) fn resolve_fe_cov_type(
    df: &DataFrame,
    kind: PanelCovKind,
    options: &CovColumns<'_>,
) -> PyResult<FeCovType> {
    Ok(match kind {
        PanelCovKind::Classical => FeCovType::Classical,
        PanelCovKind::Hc1 => FeCovType::Hc1,
        PanelCovKind::Hc2 => FeCovType::Hc2,
        PanelCovKind::Hc3 => FeCovType::Hc3,
        PanelCovKind::Cluster => FeCovType::Cluster {
            groups: extract_optional_cluster_groups(
                df,
                options.cluster,
                ColumnSource::new("cluster", "data"),
            )?,
        },
        PanelCovKind::Dk => {
            let dk_time = require_dk_time(options.dk_time)?;
            FeCovType::Dk {
                bandwidth: options.dk_bandwidth,
                time: extract_time_keys(df, dk_time, ColumnSource::new("dk_time", "data"))?,
            }
        }
    })
}

/// [`build_fe_input`]の戻り値。
#[derive(Debug)]
pub(crate) struct BuiltFeInput {
    pub(crate) input: FeInput,
    pub(crate) effects: FeEffects,
    /// エンジンに渡す`cov_type`（`cluster`のグループや`dk`の時点を持つ）。
    pub(crate) cov_type: FeCovType,
    /// `FEResult.cov_type`に格納する正規形（小文字）の名前。
    pub(crate) cov_type_name: String,
}

/// Pythonから渡された `data` / `y` / `x` / `effects` / `options` を検証し、
/// `engine::panel::fe::FeInput::from_effect_groups`を呼び出すところまでを行う。
/// `FeEstimator::fit`の呼び出し・`FEResult`の構築は`fit`（本ファイル）が行う。
///
/// `effects`の要素数で切り替える（1要素が`FeEffects::OneWay`、2要素が`FeEffects::TwoWay`、
/// 3要素以上が`FeEffects::MultiWay`。モジュールdoc参照）。
///
/// # Errors
/// - 列の抽出時に発覚する問題（列が存在しない、数値/文字列型にキャストできない、
///   欠損値・NaN・無限大を含む等）は`column_extraction`の責務で`ValidationError`
/// - `x`が空リストの場合は`ValidationError`（OLS/WLS/Logit/Probit/IVと同じ
///   `validate_role_non_empty`。「固定効果のみのモデル」の許容を見直し、
///   他手法と揃えた——経緯はモジュールdoc「`x`の空リストを許容しない」参照）。
///   `y`/`effects`/`x`間の重複・`x`・`effects`内部の重複も同じく`validation.rs`の責務で
///   `ValidationError`
/// - `effects`が空、または6要素以上の場合は`ValidationError`（`validate_panel_roles`内の`validate_effects`）
/// - `cov_type`の文字列が不正な場合は`ValidationError`（`parse_panel_cov_kind`参照）
/// - それ以外（`y`/`effects`間の行数不一致等）は`engine::panel::common::PanelError`
///   から`panel_error_to_pyerr`で変換
pub(crate) fn build_fe_input(
    df: &DataFrame,
    y: String,
    x: Vec<String>,
    effects: Vec<String>,
    options: &FEOptions,
) -> PyResult<BuiltFeInput> {
    // DataFrameを見ない検査を先に行う。報告順は`docs/guide/validation.md`
    // 「Order in which errors are reported」: オプションの値 → オプションの組合せ →
    // ロール構造 → `y`/`x`の列 → オプション列 → エンジン。
    let cov_kind = parse_panel_cov_kind(&options.cov_type, PanelModel::Fe)?;
    check_panel_cov_options(
        cov_kind,
        options.cluster.as_deref(),
        options.dk_time.as_deref(),
        options.dk_bandwidth,
    )?;

    // `effects`の列数・`x`が空（モジュールdoc参照）・列の役割の重複・`dk_time`の役割を検査する。
    validate_panel_roles(PanelModel::Fe, &y, &x, &effects, options.dk_time.as_deref())?;

    // ── y/x/effects列の抽出 ─────────────────────────────────────────────
    let y_slice = extract_f64_column(df, &y, ColumnSource::new("y", "data"))?;

    let x_slices = extract_f64_columns(df, &x, ColumnSource::new("x", "data"))?;

    // ── `effects`の列の抽出（1つ目が個体、2つ目が時点、3つ目以降は同一性だけのキー）──
    let groups = extract_effect_groups(df, &effects)?;
    let fe_effects = match groups.len() {
        1 => FeEffects::OneWay,
        2 => FeEffects::TwoWay,
        _ => FeEffects::MultiWay,
    };

    // ── cov_type固有の追加列の抽出（該当するcov_typeのときのみ）─────────────
    let cov_type = resolve_fe_cov_type(
        df,
        cov_kind,
        &CovColumns {
            cluster: options.cluster.as_deref(),
            dk_time: options.dk_time.as_deref(),
            dk_bandwidth: options.dk_bandwidth,
        },
    )?;

    let input = FeInput::from_effect_groups(&y_slice, &x_slices, x, groups, y)
        .map_err(panel_error_to_pyerr)?
        .with_effect_names(effects.clone());

    Ok(BuiltFeInput {
        input,
        effects: fe_effects,
        cov_type,
        cov_type_name: cov_kind.name().to_string(),
    })
}

/// Pythonから渡された `data` / `y` / `x` / `effects` / `options` を検証し、
/// `build_fe_input`で構築した`FeInput`に対して`engine::panel::fe::FeEstimator::fit`を
/// 呼び出し、`FEResult`として返す。
///
/// `n_groups`の元になる`group_counts`は、`FeEstimator::n_groups()`（`fit`が吸収した次元の水準数を
/// `effects`の順に数えた値）をそのまま使う。
///
/// `params`/`param_names`/`residuals`/`dep_var_name`/`n_obs`/`log_likelihood`は
/// `FeEstimator`のgetter（`params()`・`within_input()`・`residuals()`・`log_likelihood()`）から
/// 取得する
/// （`std_errors`/`test_stats`/`p_values`/`conf_lower`/`conf_upper`はFE自身が`cov_type`・
/// 自由度調整を反映して計算し直した値のため、`FeEstimator`自身のgetterを使う——
/// `engine/src/panel/fe.rs`モジュールdoc「`OlsEstimator`への委譲」参照）。
///
/// # Errors
/// - `build_fe_input`が返すエラー（列抽出・y/x/effects/dk_timeの重複・`cov_type`文字列の
///   検証等）は`ValidationError`
/// - `FeEstimator::fit`が返す`engine::panel::common::PanelError`（singleton検出・
///   自由度不足・分散ゼロ・委譲先`OlsEstimator::fit`の失敗・FE自身のF検定の失敗等）は
///   `panel_error_to_pyerr`で変換
pub(crate) fn fit(
    py: Python<'_>,
    data: SelectedData,
    y: String,
    x: Vec<String>,
    effects: Vec<String>,
    options: &FEOptions,
) -> PyResult<FEResult> {
    // 取り込み済みの`frame`以降はPythonオブジェクトに触れないので、GILを手放して走らせる
    // （他のPythonスレッドを止めず、複数のfitをスレッドで並行できる）。`source`は使わず、
    // 関数の末尾（`detach`から戻ってGILを持った状態）で`data`ごとdropされる。
    let SelectedData { frame, .. } = data;
    py.detach(|| fit_frame(&frame, y, x, effects, options))
}

/// [`fit`]の純Rust部分（GILを手放した状態で走る）。`PyErr`は遅延生成でGILを要求しない
/// ので、ここで`?`により返してよい。
fn fit_frame(
    df: &DataFrame,
    y: String,
    x: Vec<String>,
    effects: Vec<String>,
    options: &FEOptions,
) -> PyResult<FEResult> {
    let BuiltFeInput {
        input,
        effects: fe_effects,
        cov_type,
        cov_type_name,
    } = build_fe_input(df, y, x, effects.clone(), options)?;

    let absorption = FeAbsorption {
        tol: options.tol,
        max_iter: options.max_iter,
    };
    let estimator = FeEstimator::fit_with_absorption(
        input,
        fe_effects,
        cov_type,
        options.confidence_level,
        &absorption,
    )
    .map_err(panel_error_to_pyerr)?;
    // 固定効果は`fixed_effects()`かpickleまで計算しない（`FeEstimator`を結果に持たせる）。
    let mut result = fe_result_fields(&estimator, effects, cov_type_name, placeholder_effects());
    result.fixed_effects = LazyFixedEffects::new(estimator);
    Ok(result)
}

/// `fe_result_fields`の`fixed_effects`に仮に入れる値（呼び出し側が差し替える）。
fn placeholder_effects() -> LazyFixedEffects {
    LazyFixedEffects::Computed(FixedEffects::OneWay(BTreeMap::new()))
}

/// `FeEstimator`から`FEResult`を作る。`fixed_effects`は呼び出し側が決める（FEは`FeEstimator`を持たせて計算を
/// 遅らせ、FEIVの第一段階は計算済みの辞書を持たせて`FeEstimator`を保持しない）。
pub(crate) fn fe_result_fields(
    estimator: &FeEstimator,
    effects: Vec<String>,
    cov_type_name: String,
    fixed_effects: LazyFixedEffects,
) -> FEResult {
    let group_counts = estimator.n_groups().to_vec();
    FEResult {
        params: mat_to_vec(estimator.params()),
        std_errors: mat_to_vec(estimator.std_errors()),
        cov_params: mat_to_rows(estimator.cov_params()),
        test_stats: mat_to_vec(estimator.test_stats()),
        stat_dist: estimator.stat_dist().name().to_string(),
        stat_df: estimator.stat_dist().df().map(|df| df as i64),
        p_values: mat_to_vec(estimator.p_values()),
        conf_lower: mat_to_vec(estimator.conf_lower()),
        conf_upper: mat_to_vec(estimator.conf_upper()),
        param_names: estimator.within_input().param_names().to_vec(),
        residuals: mat_to_vec(estimator.residuals()),
        dep_var_name: estimator.within_input().dep_var_name().to_string(),
        n_obs: estimator.within_input().nobs(),
        df_resid: estimator.df_resid(),
        df_model: estimator.df_model(),
        effects,
        group_counts,
        cov_type: cov_type_name,
        dk_bandwidth_used: estimator.dk_bandwidth_used().map(|bw| bw as i64),
        f_statistic: estimator.f_statistic(),
        f_p_value: estimator.f_p_value(),
        f_df_num: estimator.f_df().map(|(num, _)| num),
        f_df_denom: estimator.f_df().map(|(_, denom)| denom),
        log_likelihood: estimator.log_likelihood(),
        aic: estimator.aic(),
        bic: estimator.bic(),
        r_squared_within: estimator.r_squared_within(),
        r_squared_between: estimator.r_squared_between(),
        r_squared_overall: estimator.r_squared_overall(),
        fixed_effects,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panel::common::well_formed_df;
    use engine::panel::common::TimeKeys;
    use engine::shared::group_codes::GroupCodes;
    use polars::df;

    /// `build_fe_input`のテスト全体で使う既定の`FEOptions`（`cov_type="cluster"`・
    /// `dk_time=None`）。フィールドごとに上書きして使う。
    fn default_options() -> FEOptions {
        FEOptions::new("cluster".to_string(), 0.95, None, None, None, None, 10000)
    }

    #[test]
    fn build_fe_input_succeeds_for_well_formed_one_way_data() {
        let df = well_formed_df();
        let options = default_options();

        let BuiltFeInput {
            input,
            effects,
            cov_type,
            cov_type_name,
        } = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["id".to_string()],
            &options,
        )
        .unwrap();

        assert_eq!(input.nobs(), 6);
        assert_eq!(input.x_names(), &["x1".to_string()]);
        assert_eq!(input.time_periods(), None);
        assert_eq!(effects, FeEffects::OneWay);
        assert_eq!(
            cov_type,
            FeCovType::Cluster { groups: None },
            "cluster未指定時はNone（engine側でentity列にフォールバック）"
        );
        assert_eq!(cov_type_name, "cluster");
    }

    #[test]
    fn build_fe_input_selects_two_way_effects_for_two_columns() {
        let df = well_formed_df();
        let options = default_options();

        let BuiltFeInput { input, effects, .. } = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["id".to_string(), "t".to_string()],
            &options,
        )
        .unwrap();

        assert_eq!(effects, FeEffects::TwoWay);
        assert_eq!(input.n_effects(), 2);
        assert_eq!(input.nobs(), 6);
        assert_eq!(
            input.time_periods(),
            Some(["1".to_string(), "2".to_string()].as_slice())
        );
    }

    #[test]
    fn build_fe_input_rejects_empty_x() {
        // 列の役割の検査が`validate_panel_roles`経由で配線されていることの確認
        // （検査の中身は`panel/common.rs`のテストが持つ）。
        let df = well_formed_df();
        let options = default_options();

        let result = build_fe_input(
            &df,
            "y".to_string(),
            vec![],
            vec!["id".to_string()],
            &options,
        );

        assert!(result.is_err());
    }

    #[test]
    fn build_fe_input_returns_error_for_unknown_cov_type() {
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "unknown".to_string();

        let result = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["id".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_fe_input_returns_error_for_hc0_cov_type() {
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "hc0".to_string();

        let result = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["id".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_fe_input_extracts_cluster_groups_when_cov_type_is_cluster_and_cluster_set() {
        let df = df!(
            "y" => [1.0, 2.0, 3.0, 4.0],
            "x1" => [2.0, 4.0, 1.0, 5.0],
            "id" => ["a", "a", "b", "b"],
            "state" => ["x", "y", "x", "y"],
        )
        .unwrap();
        let mut options = default_options();
        options.cluster = Some("state".to_string());

        let BuiltFeInput { cov_type, .. } = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["id".to_string()],
            &options,
        )
        .unwrap();

        assert_eq!(
            cov_type,
            FeCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&[
                    "x".to_string(),
                    "y".to_string(),
                    "x".to_string(),
                    "y".to_string(),
                ]))
            }
        );
    }

    #[test]
    fn build_fe_input_dk_uses_dk_time_for_one_way_effects() {
        // 1-way FE + DK HAC（`effects`は1要素・`dk_time`のみ指定）の組み合わせ
        // （モジュールdoc「`effects`と`FEOptions.dk_time`は別物」参照）。
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "dk".to_string();
        options.dk_time = Some("t".to_string());

        let BuiltFeInput {
            input,
            effects,
            cov_type,
            ..
        } = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["id".to_string()],
            &options,
        )
        .unwrap();

        assert_eq!(effects, FeEffects::OneWay);
        assert_eq!(input.time_periods(), None); // dk_timeはFeInputの効果の次元には渡らない
        assert_eq!(
            cov_type,
            FeCovType::Dk {
                bandwidth: None,
                time: TimeKeys::by_integer(
                    ["1", "2", "1", "2", "1", "2"].map(str::to_string).to_vec(),
                    &[1, 2, 1, 2, 1, 2]
                )
                .unwrap(),
            }
        );
    }

    #[test]
    fn build_fe_input_dk_uses_dk_time_independently_of_effects_when_two_way() {
        // 2-way（`effects`が2要素）でもDKの時点列は`dk_time`だけから決まり、`FeInput`の2つ目の次元
        // （固定効果の時間次元）とは独立であることを確認する（モジュールdoc参照）。
        let df = df!(
            "y" => [1.0, 2.0, 3.0, 4.0],
            "x1" => [2.0, 4.0, 1.0, 5.0],
            "id" => ["a", "a", "b", "b"],
            "t" => ["1", "2", "1", "2"],
            "t_fine" => ["q1", "q2", "q1", "q2"],
        )
        .unwrap();
        let mut options = default_options();
        options.cov_type = "dk".to_string();
        options.dk_time = Some("t_fine".to_string());

        let BuiltFeInput {
            input,
            effects,
            cov_type,
            ..
        } = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["id".to_string(), "t".to_string()],
            &options,
        )
        .unwrap();

        assert_eq!(effects, FeEffects::TwoWay);
        assert_eq!(input.n_effects(), 2);
        assert_eq!(input.nobs(), 4);
        assert_eq!(
            input.time_periods(),
            Some(["1".to_string(), "2".to_string()].as_slice())
        );
        assert_eq!(
            cov_type,
            FeCovType::Dk {
                bandwidth: None,
                time: TimeKeys::lexicographic(
                    ["q1", "q2", "q1", "q2"].map(str::to_string).to_vec()
                ),
            }
        );
    }

    #[test]
    fn build_fe_input_dk_requires_dk_time_even_when_two_way() {
        // `effects[1]`（2-wayの固定効果の時間次元）はDKの時点列に借用されない。`dk_time`なしは
        // 1-way・2-wayとも`ValidationError`（モジュールdoc参照）。
        let df = well_formed_df();
        for effects in [vec!["id"], vec!["id", "t"]] {
            let mut options = default_options();
            options.cov_type = "dk".to_string();

            let result = build_fe_input(
                &df,
                "y".to_string(),
                vec!["x1".to_string()],
                effects.iter().map(|c| c.to_string()).collect(),
                &options,
            );

            assert!(result.is_err());
        }
    }
}
