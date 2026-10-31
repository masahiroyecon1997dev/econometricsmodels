//! FEの推定オプション・結果、およびPython（polars DataFrame + 列名 + オプション）から
//! `engine::panel::fe`（within変換・パネル自由度調整・`cov_type`対応・パネル固有R²）を
//! 呼び出すところまでの一連の処理（データ抽出・pyclass定義、`FeEstimator::fit`への
//! 実際の配線・`#[pymodule]`登録、`fixed_effects()`メソッドを段階的に実装した）。
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
//! ## 実装フェーズの分割方針（IV・Logitと同じ3段階、`engine_pybind/src/iv/CLAUDE.md`参照）
//!
//! 1. **データ抽出・pyclass定義段階（完了）**: `FEOptions`/`FEResult`のpyclass定義、
//!    列抽出・バリデーション・`engine::panel::fe::FeInput`構築までを行う`build_fe_input`を
//!    実装した。
//! 2. **engine呼び出し段階**: `build_fe_input`を実際に呼び出す`fit`関数を追加し、`lib.rs`に
//!    `#[pyfunction] fit_fe`を新設して`#[pymodule]`に登録する。`build_fe_input`/
//!    `parse_panel_cov_kind`/`panel_error_to_pyerr`はこの時点で本番経路（`fit_fe`）から
//!    実際に呼ばれるようになるため、`#[allow(dead_code)]`はすべて削除する
//!    （`engine_pybind/src/iv/CLAUDE.md`「実装フェーズの分割方針」の初期実装段階と同じ）。
//! 3. **`fixed_effects()`メソッド追加段階**: IVの`first_stage()`と同じ
//!    「追加結果は別メソッド」方針、`docs/spec/fe-spec.md`3.5節）を追加する。`FEResult`に
//!    非公開フィールド（`LazyFixedEffects`。`FeEstimator`を保持し、固定効果は初回の
//!    `fixed_effects()`かpickle時に計算して保持する）を追加し、`fixed_effects()`は
//!    そこから返すだけ（IVの`IVResult.first_stage`フィールドが初期実装ではなく後続の拡張で
//!    追加されたのと同じ段階分割）。pickleで復元した結果は`FeEstimator`を持たず、計算済みの
//!    固定効果だけを持つ。
//!
//! ## `effects`と`FEOptions.dk_time`は別物（`panel-common.md`1.1節）
//!
//! 固定効果の次元は必須引数`effects`（`list[str]`、1〜2要素）で指定する。1要素が1-way、
//! 2要素が2-way（`effects[0]`が個体、`effects[1]`が時点）。`dk_time`（`cov_type="dk"`の値を
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
//! v1では「固定効果のみのモデル」（`x=[]`）を意図的に許容していた（`validate_x_non_empty`を
//! 呼ばない設計）。しかしユーザーからの指摘で、この判断が独立して吟味された
//! 記録が`panel-common.md`に見当たらないこと・`x`が空だと何らかの説明変数が`y`に与える
//! 効果を推定するという因果推論の営みが成立しない（実質「個体・時間固定効果によるyの分解」
//! という別の操作になる）ことが指摘され、他手法（OLS/WLS/Logit/Probit/IV）と同じ
//! `validate_x_non_empty`を適用し空を拒否する方針に変更した。**`engine`側
//! （`FeInput::from_columns`・`FeEstimator::fit`）は変更していない**——k=0を受理する
//! 既存の振る舞いはそのまま残す（`engine/src/panel/fe.rs`の
//! `fe_estimator_fit_with_no_regressors_estimates_fixed_effects_only_model`が
//! 引き続き固定効果のみのモデルがengineレベルでは動作することを検証する）。あくまで
//! Python向けAPI（`engine_pybind`）の業務的なバリデーションとしてのみ拒否する
//! （OLS自身も`engine::linear::ols::OlsEstimator::fit`自体はk=0をpanicなく受理するが
//! Python APIの`validate_x_non_empty`が弾く、という既存の非対称性と同じ構図）。

use crate::shared::errors::{guard, internal_error};
use engine::panel::fe::{FeCovType, FeEffects, FeEstimator, FeInput, FixedEffects};
use polars::prelude::DataFrame;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BTreeMap;
use std::sync::OnceLock;

use super::common::{
    PanelCovKind, PanelModel, check_panel_cov_options, extract_effects, group_counts, groups_dict,
    panel_error_to_pyerr, parse_panel_cov_kind, require_dk_time, validate_dk_time_role,
    validate_effects,
};
use crate::shared::column_error::ColumnSource;
use crate::shared::column_extraction::{
    SelectedData, extract_f64_column, extract_f64_columns, extract_optional_cluster_groups,
    extract_time_keys,
};
use crate::shared::mat_conversion::{mat_to_rows, mat_to_vec};
use crate::shared::option_repr::PyRepr;
use crate::shared::option_values::{
    extract_strict_float, extract_strict_opt_column, extract_strict_opt_int, extract_strict_text,
};
use crate::shared::validation::{
    RoleValue, validate_no_duplicate_roles, validate_no_duplicate_within_role, validate_x_non_empty,
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
// 各フィールドの意味・既定値の根拠は`docs/spec/panel-common.md`。
// module/from_py_objectの理由は`OLSOptions`と同じ（`engine_pybind/src/linear/ols.rs`参照）。
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
    /// periods). Required when `cov_type="dk"`, in one-way and two-way models alike:
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
}

crate::shared::pickle::impl_pickle_state!(FEOptions {
    cov_type,
    confidence_level,
    cluster,
    dk_time,
    dk_bandwidth,
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
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = crate::shared::option_values::cov_type_arg)] cov_type: String,
        #[pyo3(from_py_with = crate::shared::option_values::confidence_level_arg)] confidence_level: f64,
        #[pyo3(from_py_with = crate::shared::option_values::cluster_arg)] cluster: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::dk_time_arg)] dk_time: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::dk_bandwidth_arg)] dk_bandwidth: Option<
            i64,
        >,
    ) -> Self {
        Self {
            cov_type,
            confidence_level,
            cluster,
            dk_time,
            dk_bandwidth,
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
             dk_time={}, dk_bandwidth={})",
            self.cov_type.py_repr(),
            self.confidence_level.py_repr(),
            self.cluster.py_repr(),
            self.dk_time.py_repr(),
            self.dk_bandwidth.py_repr()
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
// `Clone`を派生しない: `fixed_effects`フィールドが保持する`FeEstimator`（内部の`OlsEstimator`も）が
// `Clone`を実装していないため（`LogitResult`/`ProbitResult`と同じ理由、
// `.claude/rules/rust-style.md`「推定量構造体の設計」の通りprivateフィールドのみで、
// `Clone`を要求する既存の呼び出し元も無い）。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug)]
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
    /// Denominator degrees of freedom of `f_statistic` (`None` when `f_statistic` is `None`). May differ
    /// from `df_resid` (e.g. OLS/WLS with cluster-robust inference uses `G - 1`).
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
#[derive(Debug)]
enum LazyFixedEffects {
    Lazy {
        estimator: Box<FeEstimator>,
        cell: OnceLock<FixedEffects>,
    },
    Computed(FixedEffects),
}

impl LazyFixedEffects {
    fn new(estimator: FeEstimator) -> Self {
        Self::Lazy {
            estimator: Box::new(estimator),
            cell: OnceLock::new(),
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

// 状態は`("one_way", {id: α})`か`("two_way", {id: α}, [(period, γ), ...])`。時点は時間順の並びを保つ。
impl crate::shared::pickle::PickleField for LazyFixedEffects {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.get() {
            FixedEffects::OneWay(effects) => ("one_way", effects).into_bound_py_any(py),
            FixedEffects::TwoWay { entity, time } => {
                ("two_way", entity, time).into_bound_py_any(py)
            }
        }
    }

    fn from_py(obj: &Bound<'_, PyAny>) -> Option<Self> {
        let effects = if let Ok((tag, effects)) = obj.extract::<(String, BTreeMap<String, f64>)>() {
            (tag == "one_way").then_some(FixedEffects::OneWay(effects))?
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
    /// model, two for a two-way model. A two-way model requires a balanced panel, so the
    /// count of the second column equals the number of observations per group of the first.
    #[getter]
    fn n_groups<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        groups_dict(py, &self.effects, &self.group_counts)
    }

    /// The fixed effects themselves (`α_i` for the first column of `effects`, `γ_t` for the
    /// second), recovered post-hoc from the fitted coefficients (`α_i = ȳ_i - x̄_i'β̂` for
    /// one-way effects).
    ///
    /// Always a `dict[str, dict[str, float]]` keyed by the column names in `effects`, then
    /// by group (the value of the column as a string): one entry for a one-way model, two
    /// for a two-way model.
    ///
    /// Two-way normalization: `α_i`/`γ_t` are not individually identified (adding a
    /// constant to one and subtracting it from the other leaves `α_i + γ_t`, and
    /// therefore the fitted values, unchanged). This implementation fixes the
    /// reference time period to `γ_{t_ref} = 0`, where `t_ref` is the first period in
    /// time order (the order of the values of the time column: numeric for integer and
    /// float columns, chronological for `Date`/`Datetime`, the category order for
    /// `Enum`, alphabetical for strings and `Categorical`) — a deterministic convention
    /// independent of row order. The dictionary of the second column lists the periods in
    /// the same order. `fixest::fixef()` instead uses the time value that appears first in
    /// observation order, so numerical agreement with `fixest` for two-way effects
    /// only holds when those two choices of `t_ref` coincide for the given data.
    fn fixed_effects(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        guard(|| {
            // 列名をキーにした入れ子の`dict`にする（1-wayは1項目、2-wayは2項目。
            // Pythonの`dict`は挿入順を保つ）。
            // `effects`の長さと固定効果の1-way/2-wayは`fit`が揃えて作る。食い違う状態
            // （壊れたpickle等）は`INTERNAL_ERROR`にする。
            let first = self
                .effects
                .first()
                .ok_or_else(|| internal_error("the effects columns are missing"))?;
            let outer = PyDict::new(py);
            match self.fixed_effects.get() {
                FixedEffects::OneWay(entity) => {
                    outer.set_item(first, entity)?;
                }
                FixedEffects::TwoWay { entity, time } => {
                    let second = self.effects.get(1).ok_or_else(|| {
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
            }
            Ok(outer.unbind())
        })
    }
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
fn resolve_fe_cov_type(
    df: &DataFrame,
    kind: PanelCovKind,
    options: &FEOptions,
) -> PyResult<FeCovType> {
    Ok(match kind {
        PanelCovKind::Classical => FeCovType::Classical,
        PanelCovKind::Hc1 => FeCovType::Hc1,
        PanelCovKind::Hc2 => FeCovType::Hc2,
        PanelCovKind::Hc3 => FeCovType::Hc3,
        PanelCovKind::Cluster => FeCovType::Cluster {
            groups: extract_optional_cluster_groups(
                df,
                options.cluster.as_deref(),
                ColumnSource::new("cluster", "data"),
            )?,
        },
        PanelCovKind::Dk => {
            let dk_time = require_dk_time(options.dk_time.as_deref())?;
            FeCovType::Dk {
                bandwidth: options.dk_bandwidth,
                time: extract_time_keys(df, dk_time, ColumnSource::new("dk_time", "data"))?,
            }
        }
    })
}

/// Pythonから渡された `data` / `y` / `x` / `effects` / `options` を検証し、
/// `engine::panel::fe::FeInput::from_group_codes`を呼び出すところまでを行う。
/// `FeEstimator::fit`の呼び出し・`FEResult`の構築は`fit`（本ファイル）が行う。
///
/// `effects`の要素数で1-way/2-wayを切り替える（1要素が`FeEffects::OneWay`、2要素が
/// `FeEffects::TwoWay`。モジュールdoc参照）。
///
/// # Errors
/// - 列の抽出時に発覚する問題（列が存在しない、数値/文字列型にキャストできない、
///   欠損値・NaN・無限大を含む等）は`column_extraction`の責務で`ValidationError`
/// - `x`が空リストの場合は`ValidationError`（OLS/WLS/Logit/Probit/IVと同じ
///   `validate_x_non_empty`。「固定効果のみのモデル」の許容を見直し、
///   他手法と揃えた——経緯はモジュールdoc「`x`の空リストを許容しない」参照）。
///   `y`/`effects`/`x`間の重複・`x`・`effects`内部の重複も同じく`validation.rs`の責務で
///   `ValidationError`
/// - `effects`が空、または3要素以上の場合は`ValidationError`（`validate_effects`）
/// - `cov_type`の文字列が不正な場合は`ValidationError`（`parse_panel_cov_kind`参照）
/// - それ以外（`y`/`effects`間の行数不一致等）は`engine::panel::common::PanelError`
///   から`panel_error_to_pyerr`で変換
pub(crate) fn build_fe_input(
    df: &DataFrame,
    y: String,
    x: Vec<String>,
    effects: Vec<String>,
    options: &FEOptions,
) -> PyResult<(FeInput, FeEffects, FeCovType, String)> {
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

    // `effects`の列数（空・3つ以上）を先に見る。
    validate_effects(PanelModel::Fe, &effects)?;
    // `x`が空リストであることを許容しない（モジュールdoc参照）。
    // OLS/WLS/Logit/Probit/IVと同じ`validate_x_non_empty`を呼ぶ（`.claude/rules/
    // rust-style.md`「バリデーションの責務分担」）。
    validate_x_non_empty("x", &x)?;
    validate_no_duplicate_roles(&[
        ("y", RoleValue::Single(&y)),
        ("effects", RoleValue::Multi(&effects)),
        ("x", RoleValue::Multi(&x)),
    ])?;
    validate_no_duplicate_within_role("x", &x)?;
    validate_no_duplicate_within_role("effects", &effects)?;
    validate_dk_time_role(&y, &effects[0], options.dk_time.as_deref())?;

    // ── y/x/entity列の抽出 ─────────────────────────────────────────────
    let y_slice = extract_f64_column(df, &y, ColumnSource::new("y", "data"))?;

    let x_slices = extract_f64_columns(df, &x, ColumnSource::new("x", "data"))?;

    // ── `effects`の列の抽出（1つ目が個体、2つ目があれば時点）──────────────
    let (entity_groups, time_keys) = extract_effects(df, &effects)?;
    let fe_effects = if time_keys.is_some() {
        FeEffects::TwoWay
    } else {
        FeEffects::OneWay
    };

    // ── cov_type固有の追加列の抽出（該当するcov_typeのときのみ）─────────────
    let cov_type = resolve_fe_cov_type(df, cov_kind, options)?;

    let input = FeInput::from_group_codes(&y_slice, &x_slices, x, entity_groups, time_keys, y)
        .map_err(panel_error_to_pyerr)?
        .with_effect_names(effects[0].clone(), effects.get(1).cloned());

    Ok((input, fe_effects, cov_type, cov_kind.name().to_string()))
}

/// Pythonから渡された `data` / `y` / `x` / `effects` / `options` を検証し、
/// `build_fe_input`で構築した`FeInput`に対して`engine::panel::fe::FeEstimator::fit`を
/// 呼び出し、`FEResult`として返す。
///
/// `n_entities`は`FeEstimator::input().n_entities()`（`FeInput`が構築時に一度だけ作った
/// エンティティコードのユニーク数）から取得する。
///
/// `params`/`param_names`/`residuals`/`dep_var_name`/`n_obs`/`log_likelihood`は
/// `FeEstimator`のgetter（`params()`・`within_input()`・`residuals()`・`log_likelihood()`）から
/// 取得する
/// （`std_errors`/`test_stats`/`p_values`/`conf_lower`/`conf_upper`はFE自身が`cov_type`・
/// 自由度調整を反映して計算し直した値のため、`FeEstimator`自身のgetterを使う——
/// `engine/src/panel/fe.rs`モジュールdoc「`OlsEstimator`への委譲」参照）。
///
/// # Errors
/// - `build_fe_input`が返すエラー（列抽出・y/x/entity/timeの重複・`cov_type`文字列の
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
    let (input, fe_effects, cov_type, cov_type_lower) =
        build_fe_input(df, y, x, effects.clone(), options)?;

    let estimator = FeEstimator::fit(input, fe_effects, cov_type, options.confidence_level)
        .map_err(panel_error_to_pyerr)?;
    let group_counts = group_counts(estimator.input().n_entities(), estimator.n_periods());

    Ok(FEResult {
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
        cov_type: cov_type_lower,
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
        fixed_effects: LazyFixedEffects::new(estimator),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::panel::common::TimeKeys;
    use engine::shared::group_codes::GroupCodes;
    use polars::df;

    /// `build_fe_input`のテスト全体で使う既定の`FEOptions`（`cov_type="cluster"`・
    /// `dk_time=None`）。フィールドごとに上書きして使う。
    fn default_options() -> FEOptions {
        FEOptions::new("cluster".to_string(), 0.95, None, None, None)
    }

    fn well_formed_df() -> DataFrame {
        df!(
            "y" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x1" => [2.0, 4.0, 1.0, 5.0, 3.0, 6.0],
            "id" => ["a", "a", "b", "b", "c", "c"],
            "t" => ["1", "2", "1", "2", "1", "2"],
        )
        .unwrap()
    }

    #[test]
    fn build_fe_input_succeeds_for_well_formed_one_way_data() {
        let df = well_formed_df();
        let options = default_options();

        let (input, effects, cov_type, cov_type_lower) = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["id".to_string()],
            &options,
        )
        .unwrap();

        assert_eq!(input.nobs(), 6);
        assert_eq!(input.x_names(), &["x1".to_string()]);
        assert_eq!(input.time_keys(), None);
        assert_eq!(effects, FeEffects::OneWay);
        assert_eq!(
            cov_type,
            FeCovType::Cluster { groups: None },
            "cluster未指定時はNone（engine側でentity列にフォールバック）"
        );
        assert_eq!(cov_type_lower, "cluster");
    }

    #[test]
    fn build_fe_input_selects_two_way_effects_for_two_columns() {
        let df = well_formed_df();
        let options = default_options();

        let (input, effects, ..) = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["id".to_string(), "t".to_string()],
            &options,
        )
        .unwrap();

        assert_eq!(effects, FeEffects::TwoWay);
        let Some(time_keys) = input.time_keys() else {
            panic!("time keys must be set");
        };
        assert_eq!(time_keys.len(), 6);
        assert_eq!(time_keys.periods(), ["1", "2"]);
    }

    #[test]
    fn build_fe_input_returns_error_for_empty_x() {
        // 固定効果のみのモデル（`x=[]`）を拒否する（モジュールdoc「`x`の
        // 空リストを許容しない」参照。OLS/WLS/Logit/Probit/IVと同じ`validate_x_non_empty`）。
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
    fn build_fe_input_returns_error_when_y_overlaps_entity() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["y".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_fe_input_returns_error_when_x_overlaps_entity() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string(), "id".to_string()],
            vec!["id".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_fe_input_returns_error_when_x_overlaps_second_effect() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string(), "t".to_string()],
            vec!["id".to_string(), "t".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_fe_input_returns_error_when_x_contains_duplicate() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string(), "x1".to_string()],
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

        let (_, _, cov_type, _) = build_fe_input(
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

        let (input, effects, cov_type, _) = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["id".to_string()],
            &options,
        )
        .unwrap();

        assert_eq!(effects, FeEffects::OneWay);
        assert_eq!(input.time_keys(), None); // dk_timeはFeInput.timeには渡らない
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
        // 2-way（`effects`が2要素）でもDKの時点列は`dk_time`だけから決まり、`FeInput.time_keys()`
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

        let (input, effects, cov_type, _) = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["id".to_string(), "t".to_string()],
            &options,
        )
        .unwrap();

        assert_eq!(effects, FeEffects::TwoWay);
        let Some(time_keys) = input.time_keys() else {
            panic!("time keys must be set");
        };
        assert_eq!(time_keys.len(), 4);
        assert_eq!(time_keys.periods(), ["1", "2"]);
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
    fn build_fe_input_dk_time_may_equal_second_effect_or_x_but_not_y_or_first_effect() {
        // 固定効果と同じ時間粒度(`dk_time == effects[1]`)や、`x`との重複は正当な使い方。
        // `y`・`effects[0]`と同じ列だけ拒否する(`validate_dk_time_role`参照)。
        let df = well_formed_df();
        let cases = [
            (vec!["id", "t"], "t", vec!["x1"], true),
            (vec!["id"], "x1", vec!["x1"], true),
            (vec!["id"], "y", vec!["x1"], false),
            (vec!["id"], "id", vec!["x1"], false),
            (vec!["id", "t"], "id", vec!["x1"], false),
        ];
        for (effects, dk_time, x, ok) in cases {
            let mut options = default_options();
            options.cov_type = "dk".to_string();
            options.dk_time = Some(dk_time.to_string());

            let result = build_fe_input(
                &df,
                "y".to_string(),
                x.iter().map(|c| c.to_string()).collect(),
                effects.iter().map(|c| c.to_string()).collect(),
                &options,
            );

            assert_eq!(result.is_ok(), ok, "effects={effects:?} dk_time={dk_time}");
        }
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
