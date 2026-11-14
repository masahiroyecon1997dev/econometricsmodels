//! FEGMM（固定効果つきのGMM）の推定オプション・結果、およびPython（polars DataFrame + 列名 + オプション）から
//! `engine::panel::fe_iv::FeGmmEstimator`を呼び出すところまでの一連の処理（データ抽出・pyclass定義、
//! `FeGmmEstimator::fit_with_absorption`への配線、事後診断`c_stat`）。
//!
//! 【責務分離・言語方針】`panel/fe_iv.rs`と同じ（`.claude/rules/rust-style.md`）。公開API（`FEGMMOptions`/
//! `FEGMMResult`）のdocコメントと`ValidationError`のメッセージは英語、それ以外は日本語。
//!
//! ## FEIVとの違い
//!
//! 列の役割（`y`・`x_exog`・`x_endog`・`instruments`・`effects`）と、`effects`・`cluster`・`dk_time`の規約は
//! `FEIV`と同じ（`panel/fe_iv.rs`のモジュールdoc）。違うのは推定量とその周辺だけ:
//!
//! - 推定方式のオプション`gmm_type`・`gmm_weight_type`・`gmm_max_iter`・`gmm_tol`・`raise_on_non_convergence`
//!   （`IVOptions`と同名・同規則。指定が無ければ使われるモードの実効既定値に解決し、使われないモードでの
//!   明示指定は`ValidationError`）。**重みの語彙は`classical`・`robust`・`cluster`の3つ**（`hac`は型に無い。
//!   行単位のNewey-Westは時点の順序を行ごとに渡せずパネルに使えない）で、**既定は`cluster`**（`effects[0]`）。
//! - `cluster`は`cov_type`と重みで共用する。指定が無ければどちらも`effects[0]`。
//! - 推論は正規分布・χ²（`stat_dist="normal"`・`wald_dist="chi2"`）。
//! - 結果は推定量自身の出力（係数・推論・Wald・R²・Hansen J・`n_iter`・`converged`）と事後診断`c_stat`だけ。
//!   第一段階・弱操作変数診断・Wu-Hausman・`fixed_effects()`は持たない（engineの`FeGmmEstimator`が持たない）。
//!
//! ## `c_stat`は元のデータから再計算する
//!
//! 結果は`fit()`に渡した`DataFrame`（`source_data`）と`FEGMMOptions`・列名を保持し、`c_stat()`は呼ばれたときに
//! `fit()`と同じ手順で入力を作り直して`fe_gmm_c_statistic`を呼ぶ（推定量・設計行列を結果に持たせない）。
//! 重みは`fit()`のもの（`two_step`・`iterated`は`gmm_weight_type`、重み行列を持たない`one_step`は`Classical`）。

use crate::iv::common::{CStatisticOutput, ResolvedCStatTarget, resolve_c_stat_target};
use crate::shared::errors::{guard, internal_error};
use engine::panel::common::PanelError;
use engine::panel::fe::{FeAbsorption, FeCovType, FeEffects};
use engine::panel::fe_iv::{
    FeGmmEstimator, FeGmmType, FeIvInput, FeWeightType, RoleColumns, fe_gmm_c_statistic,
};
use engine::shared::error::CommonError;
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::common::{
    PanelCovKind, PanelModel, extract_effect_groups, groups_dict, panel_error_to_pyerr,
    parse_panel_cov_kind, require_dk_time, validate_fe_iv_roles,
};
use super::fe::{CovColumns, resolve_fe_cov_type};
use super::fe_iv::FeIvColumns;
use crate::shared::column_error::ColumnSource;
use crate::shared::column_extraction::{
    SelectedData, extract_f64_column, extract_f64_columns, extract_optional_cluster_groups,
    select_columns,
};
use crate::shared::mat_conversion::{mat_to_rows, mat_to_vec};
use crate::shared::option_repr::PyRepr;
use crate::shared::option_values::{
    extract_strict_float, extract_strict_int, extract_strict_opt_column, extract_strict_opt_float,
    extract_strict_opt_int, extract_strict_opt_text, extract_strict_text, parse_choice,
};
use crate::shared::validation::reject_unused_option;

/// `gmm_max_iter`・`gmm_tol`の実効既定値（`IVOptions`と同じ。`FEGMMOptions`の既定値は`None`のため、使われる
/// モード（`gmm_type="iterated"`）でここに解決する）。
const DEFAULT_GMM_MAX_ITER: usize = 100;
const DEFAULT_GMM_TOL: f64 = 1e-6;

/// Estimation options for FEGMM (fixed effects generalized method of moments).
///
/// All arguments are keyword-only.
///
/// Options by purpose:
///
/// - Estimation method: `gmm_type`; with `"two_step"` or `"iterated"` also
///   `gmm_weight_type`; with `"iterated"` also `gmm_max_iter`, `gmm_tol` and
///   `raise_on_non_convergence`.
/// - Standard errors: `cov_type`; with `"cluster"` also `cluster`; with
///   `"dk"` also `dk_time` (required) and `dk_bandwidth`. `cluster` is also used by
///   `gmm_weight_type="cluster"`.
/// - Confidence intervals: `confidence_level`.
/// - Removing the fixed effects of an unbalanced two-way model or of a model with
///   three or more effects: `tol`, `max_iter`.
// 各フィールドの意味・既定値の根拠は`FEIVOptions`・`IVOptions`と同じ（`panel/fe_iv.rs`・`iv/common/options.rs`）。
#[pyclass(from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct FEGMMOptions {
    /// Standard error type: one of "classical", "hc1", "hc2", "hc3", "cluster", "dk".
    /// Case-insensitive. Unlike OLS/WLS/IV, "hc0" is **not** supported. It is independent
    /// of `gmm_weight_type` (the weight matrix of the point estimate): the sandwich
    /// covariance is built from the weight matrix actually used and the moment variance of
    /// this type. There is no reference implementation for "hc2" and "hc3" with GMM.
    #[pyo3(get)]
    pub cov_type: String,

    /// Confidence level for confidence intervals, in the range (0, 1).
    /// Defaults to 0.95 (a 95% confidence interval).
    #[pyo3(get)]
    pub confidence_level: f64,

    /// Column name to use as the cluster group key, for `cov_type="cluster"` and for
    /// `gmm_weight_type="cluster"` (`gmm_type="two_step"` or `"iterated"`). When `None`, the
    /// first column of `effects` is used automatically. Specifying it when neither uses it
    /// raises `ValidationError`.
    #[pyo3(get)]
    pub cluster: Option<String>,

    /// Column name that defines the time periods of the Driscoll-Kraay HAC, independent
    /// of `effects`. Required when `cov_type="dk"`: leaving it unset raises
    /// `ValidationError`, and the columns of `effects` are not used instead. It may not be
    /// the same column as `y` or the first column of `effects` (`ValidationError`).
    /// Specifying it with any other `cov_type` raises `ValidationError`. The periods are
    /// ordered by the values of the column: numerically for integers and floats,
    /// chronologically for `Date` and `Datetime`, in the order of the categories for
    /// `Enum`, and alphabetically for strings and `Categorical`.
    #[pyo3(get)]
    pub dk_time: Option<String>,

    /// Bandwidth for Driscoll-Kraay HAC when `cov_type="dk"`. When `None`, computed
    /// automatically via `floor(4*(t/100)^(2/9))` (`t` = number of unique time periods).
    /// Specifying it with any other `cov_type` raises `ValidationError`.
    #[pyo3(get)]
    pub dk_bandwidth: Option<i64>,

    /// GMM variant: one of "one_step", "two_step", "iterated" (case-insensitive).
    /// `None` (default) means "two_step". "one_step" uses the initial weight
    /// `(Z'Z)^-1` only and gives the two-stage least squares point estimate; "two_step"
    /// re-estimates once with the weight built from the first-step residuals; "iterated"
    /// repeats that until the coefficients converge.
    #[pyo3(get)]
    pub gmm_type: Option<String>,

    /// Weight matrix of the moment conditions: one of "classical", "robust", "cluster"
    /// (case-insensitive). `None` (default) means "cluster", with the groups of `cluster`
    /// (the first column of `effects` if `cluster` is `None`). "classical" gives the same
    /// point estimate as two-stage least squares; "robust" assumes the within-transformed
    /// errors are uncorrelated over time, which does not hold for short panels, so it is
    /// not recommended there. There is no "hac" weight. Specifying it with
    /// `gmm_type="one_step"` (which does not use a weight matrix) raises `ValidationError`.
    #[pyo3(get)]
    pub gmm_weight_type: Option<String>,

    /// Maximum number of GMM estimations for `gmm_type="iterated"`, counting the initial
    /// estimate; an integer from 3 to 10,000 (use `gmm_type="two_step"` for two steps).
    /// `None` (default) means 100. Specifying it with any other `gmm_type` raises
    /// `ValidationError`.
    #[pyo3(get)]
    pub gmm_max_iter: Option<i64>,

    /// Convergence tolerance for `gmm_type="iterated"`: iteration stops once every
    /// coefficient changes by at most `gmm_tol` relative to its previous value (with an
    /// absolute floor of 1e-8, so that coefficients near zero converge). `None` (default) means
    /// 1e-6. Specifying it with any other `gmm_type` raises `ValidationError`.
    #[pyo3(get)]
    pub gmm_tol: Option<f64>,

    /// Whether to raise an error if `gmm_type="iterated"` does not converge within
    /// `gmm_max_iter`. If `False`, returns the result with `converged=False` instead of
    /// raising. `None` (default) means `True`. Specifying it with any other `gmm_type`
    /// (which never checks convergence) raises `ValidationError`.
    #[pyo3(get, set)]
    pub raise_on_non_convergence: Option<bool>,

    /// Convergence tolerance of the alternating projections that remove the fixed
    /// effects of a two-way model on an **unbalanced** panel, or of a model with three or
    /// more effects (a balanced two-way panel and a one-way model are computed in closed
    /// form, so they ignore it). A pass stops when the largest group mean it subtracts is at
    /// most `tol` times the size of the column. Must be a positive finite number. Defaults
    /// to 1e-12. Not related to `gmm_tol`.
    #[pyo3(get)]
    pub tol: f64,

    /// Maximum number of passes of the alternating projections (see `tol`), from 1 to
    /// 10000. Defaults to 10000. If the limit is reached, `ComputationError` is raised.
    /// Not related to `gmm_max_iter`.
    #[pyo3(get)]
    pub max_iter: i64,
}

crate::shared::pickle::impl_pickle_state!(FEGMMOptions {
    cov_type,
    confidence_level,
    cluster,
    dk_time,
    dk_bandwidth,
    gmm_type,
    gmm_weight_type,
    gmm_max_iter,
    gmm_tol,
    raise_on_non_convergence,
    tol,
    max_iter,
});

impl FEGMMOptions {
    /// このオプションが指名するデータの列（`cluster`・`dk_time`）。
    pub(crate) fn data_columns(&self) -> Vec<String> {
        self.cluster.iter().chain(&self.dk_time).cloned().collect()
    }
}

#[pymethods]
impl FEGMMOptions {
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
        gmm_type = None,
        gmm_weight_type = None,
        gmm_max_iter = None,
        gmm_tol = None,
        raise_on_non_convergence = None,
        tol = None,
        max_iter = 10000,
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
        #[pyo3(from_py_with = crate::shared::option_values::gmm_type_arg)] gmm_type: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::gmm_weight_type_arg)]
        gmm_weight_type: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::gmm_max_iter_arg)] gmm_max_iter: Option<
            i64,
        >,
        #[pyo3(from_py_with = crate::shared::option_values::gmm_tol_arg)] gmm_tol: Option<f64>,
        raise_on_non_convergence: Option<bool>,
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
            gmm_type,
            gmm_weight_type,
            gmm_max_iter,
            gmm_tol,
            raise_on_non_convergence,
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

    #[setter]
    fn set_gmm_type(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.gmm_type = extract_strict_opt_text(value, "gmm_type")?;
        Ok(())
    }

    #[setter]
    fn set_gmm_weight_type(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.gmm_weight_type = extract_strict_opt_text(value, "gmm_weight_type")?;
        Ok(())
    }

    #[setter]
    fn set_gmm_max_iter(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.gmm_max_iter = extract_strict_opt_int(value, "gmm_max_iter")?;
        Ok(())
    }

    #[setter]
    fn set_gmm_tol(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.gmm_tol = extract_strict_opt_float(value, "gmm_tol")?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "FEGMMOptions(cov_type={}, confidence_level={}, cluster={}, \
             dk_time={}, dk_bandwidth={}, gmm_type={}, gmm_weight_type={}, \
             gmm_max_iter={}, gmm_tol={}, raise_on_non_convergence={}, tol={}, max_iter={})",
            self.cov_type.py_repr(),
            self.confidence_level.py_repr(),
            self.cluster.py_repr(),
            self.dk_time.py_repr(),
            self.dk_bandwidth.py_repr(),
            self.gmm_type.py_repr(),
            self.gmm_weight_type.py_repr(),
            self.gmm_max_iter.py_repr(),
            self.gmm_tol.py_repr(),
            self.raise_on_non_convergence.py_repr(),
            self.tol.py_repr(),
            self.max_iter.py_repr()
        )
    }
}

// 保持するオプションは、それ自身の`__reduce__`でpickleされるオブジェクトとして状態に入れる（`FEIVOptions`と同じ）。
impl crate::shared::pickle::PickleField for FEGMMOptions {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(Py::new(py, self.clone())?.into_bound(py).into_any())
    }

    fn from_py(obj: &Bound<'_, PyAny>) -> Option<Self> {
        let options = obj.cast::<FEGMMOptions>().ok()?.try_borrow().ok()?;
        Some(options.clone())
    }
}

/// `gmm_type`の種類。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GmmKind {
    OneStep,
    TwoStep,
    Iterated,
}

impl GmmKind {
    /// `FEGMMResult.gmm_type`に格納する正規形の名前。
    fn name(self) -> &'static str {
        match self {
            Self::OneStep => "one_step",
            Self::TwoStep => "two_step",
            Self::Iterated => "iterated",
        }
    }
}

/// `gmm_weight_type`の種類。`IVOptions`と違い`hac`は無い（`FeWeightType`が`Hac`を型で持たない）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeightKind {
    Classical,
    Robust,
    Cluster,
}

impl WeightKind {
    fn name(self) -> &'static str {
        match self {
            Self::Classical => "classical",
            Self::Robust => "robust",
            Self::Cluster => "cluster",
        }
    }
}

/// 文字列だけで決まるオプションの値。`gmm_type`・`gmm_weight_type`は、指定が無ければ実効既定値
/// （`two_step`・`cluster`）に解決する（`FEGMMOptions`は既定値と明示指定を区別できないので、使われない
/// モードでの明示指定は[`check_fe_gmm_options`]が弾く）。
#[derive(Clone, Copy, Debug)]
struct OptionValues {
    cov: PanelCovKind,
    gmm_type: GmmKind,
    weight: WeightKind,
}

impl OptionValues {
    /// 重み行列（`gmm_weight_type`）を使うか。`gmm_type`が`two_step`・`iterated`のとき。
    fn uses_weight(&self) -> bool {
        self.gmm_type != GmmKind::OneStep
    }

    fn is_iterated(&self) -> bool {
        self.gmm_type == GmmKind::Iterated
    }
}

/// 文字列のオプション（`cov_type`・`gmm_type`・`gmm_weight_type`）を種類にする。DataFrameは見ない。
/// 報告順は`cov_type` → `gmm_type` → `gmm_weight_type`。`gmm_weight_type`は指定されていれば`gmm_type`に
/// よらず値を検査する（`"hac"`は未知の値として`UNKNOWN_OPTION_VALUE`）。
fn parse_option_values(options: &FEGMMOptions) -> PyResult<OptionValues> {
    let cov = parse_panel_cov_kind(&options.cov_type, PanelModel::FeGmm)?;
    let gmm_type = match options.gmm_type.as_deref() {
        Some(value) => parse_choice(
            "gmm_type",
            value,
            &[
                ("one_step", GmmKind::OneStep),
                ("two_step", GmmKind::TwoStep),
                ("iterated", GmmKind::Iterated),
            ],
        )?,
        None => GmmKind::TwoStep,
    };
    let weight = match options.gmm_weight_type.as_deref() {
        Some(value) => parse_choice(
            "gmm_weight_type",
            value,
            &[
                ("classical", WeightKind::Classical),
                ("robust", WeightKind::Robust),
                ("cluster", WeightKind::Cluster),
            ],
        )?,
        None => WeightKind::Cluster,
    };
    Ok(OptionValues {
        cov,
        gmm_type,
        weight,
    })
}

/// 選んだ`gmm_type`・`cov_type`・`gmm_weight_type`では使われないオプションが明示指定されていれば
/// `ValidationError`にする（黙って無視すると書き忘れに気づけないため）。そのあと、必須の`dk_time`の欠落を
/// 検査する。DataFrameは見ない。
///
/// - `gmm_weight_type`: `gmm_type`が`two_step`・`iterated`のときだけ。`gmm_max_iter`・`gmm_tol`・
///   `raise_on_non_convergence`: `iterated`のときだけ。
/// - `cluster`: `cov_type`と`gmm_weight_type`の両方から参照されるので、どちらか一方でも使えば有効
///   （重み側は重み行列を使うときのみ）。
/// - `dk_time`・`dk_bandwidth`: `cov_type="dk"`のときだけ。`dk_time`は`dk`で必須。
///
/// # Errors
/// 使われないオプションが指定されている場合、または`dk`で`dk_time`が未指定の場合は`ValidationError`。
fn check_fe_gmm_options(options: &FEGMMOptions, values: &OptionValues) -> PyResult<()> {
    const WEIGHTED: &str = "gmm_type in (\"two_step\", \"iterated\")";
    const ITERATED: &str = "gmm_type=\"iterated\"";
    const CLUSTER_CONDITION: &str = "cov_type=\"cluster\" or (gmm_type in (\"two_step\", \"iterated\") and gmm_weight_type=\"cluster\")";

    let uses_weight = values.uses_weight();
    reject_unused_option(
        "gmm_weight_type",
        options.gmm_weight_type.is_some(),
        uses_weight,
        WEIGHTED,
    )?;
    for (name, is_set) in [
        ("gmm_max_iter", options.gmm_max_iter.is_some()),
        ("gmm_tol", options.gmm_tol.is_some()),
        (
            "raise_on_non_convergence",
            options.raise_on_non_convergence.is_some(),
        ),
    ] {
        reject_unused_option(name, is_set, values.is_iterated(), ITERATED)?;
    }
    reject_unused_option(
        "cluster",
        options.cluster.is_some(),
        values.cov == PanelCovKind::Cluster
            || (uses_weight && values.weight == WeightKind::Cluster),
        CLUSTER_CONDITION,
    )?;
    let is_dk = values.cov == PanelCovKind::Dk;
    reject_unused_option(
        "dk_time",
        options.dk_time.is_some(),
        is_dk,
        "cov_type=\"dk\"",
    )?;
    reject_unused_option(
        "dk_bandwidth",
        options.dk_bandwidth.is_some(),
        is_dk,
        "cov_type=\"dk\"",
    )?;
    if is_dk {
        require_dk_time(options.dk_time.as_deref())?;
    }
    Ok(())
}

/// [`build_fe_gmm_input`]の戻り値。
#[derive(Debug)]
pub(crate) struct BuiltFeGmmInput {
    pub(crate) input: FeIvInput,
    pub(crate) effects: FeEffects,
    /// エンジンに渡す`cov_type`（`cluster`のグループや`dk`の時点を持つ）。
    pub(crate) cov_type: FeCovType,
    /// `FEGMMResult.cov_type`に格納する正規形（小文字）の名前。
    pub(crate) cov_type_name: String,
    pub(crate) gmm_type: FeGmmType,
    /// `FEGMMResult.gmm_type`に格納する正規形の名前。
    pub(crate) gmm_type_name: String,
    /// `FEGMMResult.gmm_weight_type`に格納する正規形の名前。`one_step`は重み行列を使わないので`None`。
    pub(crate) weight_type_name: Option<String>,
    /// `gmm_type="iterated"`が収束しないときにエラーにするか（他の`gmm_type`では使われない）。
    pub(crate) raise_on_non_convergence: bool,
}

/// `gmm_type`・重み・反復の設定を`FeGmmType`にする。`groups`は重みが`cluster`で`cluster`列が無いときの既定
/// （`effects[0]`）。
fn resolve_gmm_type(
    df: &DataFrame,
    options: &FEGMMOptions,
    values: &OptionValues,
    entity_groups: &engine::shared::group_codes::GroupCodes,
) -> PyResult<FeGmmType> {
    let weight = || -> PyResult<FeWeightType> {
        Ok(match values.weight {
            WeightKind::Classical => FeWeightType::Classical,
            WeightKind::Robust => FeWeightType::Robust,
            WeightKind::Cluster => FeWeightType::Cluster {
                groups: extract_optional_cluster_groups(
                    df,
                    options.cluster.as_deref(),
                    ColumnSource::new("cluster", "data"),
                )?
                .unwrap_or_else(|| entity_groups.clone()),
            },
        })
    };
    Ok(match values.gmm_type {
        GmmKind::OneStep => FeGmmType::OneStep,
        GmmKind::TwoStep => FeGmmType::TwoStep { weight: weight()? },
        GmmKind::Iterated => {
            let weight = weight()?;
            let max_iter = match options.gmm_max_iter {
                Some(value) => usize::try_from(value).map_err(|_| {
                    panel_error_to_pyerr(PanelError::Common(CommonError::InvalidGmmMaxIter {
                        max_iter: value,
                    }))
                })?,
                None => DEFAULT_GMM_MAX_ITER,
            };
            FeGmmType::Iterated {
                weight,
                max_iter,
                tol: options.gmm_tol.unwrap_or(DEFAULT_GMM_TOL),
            }
        }
    })
}

/// Pythonから渡された`data`・列名・`options`を検証し、`engine::panel::fe_iv::FeIvInput`と推定方式を組み立てる。
///
/// 報告順は`build_fe_iv_input`と同じ: オプションの値（`cov_type` → `gmm_type` → `gmm_weight_type`）→
/// オプションの組合せ → ロール構造 → 列（`y`・`x_exog`・`x_endog`・`instruments`・`effects`）→
/// `cov_type`・重みの列 → エンジン。
///
/// # Errors
/// - オプションの値・組合せ、ロール構造（`x_endog`・`instruments`が空、役割間・役割内の重複、`effects`の
///   列数、`dk_time`の役割）、列の抽出時に発覚する問題は`ValidationError`
/// - `gmm_max_iter`が負なら`CommonError::InvalidGmmMaxIter`（3未満・上限超過と`gmm_tol`の範囲はengine）
/// - それ以外（行数の不一致等）は`PanelError`から`panel_error_to_pyerr`で変換
pub(crate) fn build_fe_gmm_input(
    df: &DataFrame,
    columns: FeIvColumns,
    options: &FEGMMOptions,
) -> PyResult<BuiltFeGmmInput> {
    let FeIvColumns {
        y,
        x_exog,
        x_endog,
        instruments,
        effects,
    } = columns;
    let values = parse_option_values(options)?;
    check_fe_gmm_options(options, &values)?;
    validate_fe_iv_roles(
        PanelModel::FeGmm,
        &y,
        &x_exog,
        &x_endog,
        &instruments,
        &effects,
        options.dk_time.as_deref(),
    )?;

    let y_slice = extract_f64_column(df, &y, ColumnSource::new("y", "data"))?;
    let exog_slices = extract_f64_columns(df, &x_exog, ColumnSource::new("x_exog", "data"))?;
    let endog_slices = extract_f64_columns(df, &x_endog, ColumnSource::new("x_endog", "data"))?;
    let instrument_slices =
        extract_f64_columns(df, &instruments, ColumnSource::new("instruments", "data"))?;

    let groups = extract_effect_groups(df, &effects)?;
    let fe_effects = match groups.len() {
        1 => FeEffects::OneWay,
        2 => FeEffects::TwoWay,
        _ => FeEffects::MultiWay,
    };
    let cov_type = resolve_fe_cov_type(
        df,
        values.cov,
        &CovColumns {
            cluster: options.cluster.as_deref(),
            dk_time: options.dk_time.as_deref(),
            dk_bandwidth: options.dk_bandwidth,
        },
    )?;
    let gmm_type = resolve_gmm_type(df, options, &values, &groups[0])?;

    let input = FeIvInput::from_group_codes(
        &y_slice,
        y,
        RoleColumns {
            names: x_exog,
            columns: &exog_slices,
        },
        RoleColumns {
            names: x_endog,
            columns: &endog_slices,
        },
        RoleColumns {
            names: instruments,
            columns: &instrument_slices,
        },
        groups,
    )
    .map_err(panel_error_to_pyerr)?
    .with_effect_names(effects);

    Ok(BuiltFeGmmInput {
        input,
        effects: fe_effects,
        cov_type,
        cov_type_name: values.cov.name().to_string(),
        gmm_type,
        gmm_type_name: values.gmm_type.name().to_string(),
        weight_type_name: values
            .uses_weight()
            .then(|| values.weight.name().to_string()),
        raise_on_non_convergence: options.raise_on_non_convergence.unwrap_or(true),
    })
}

/// データから取り込む列（役割の列と、`options`が指名する`cluster`・`dk_time`）。`fit`と`c_stat`が共有する。
pub(crate) fn data_columns(columns: &FeIvColumns, options: &FEGMMOptions) -> Vec<String> {
    let mut names = columns.role_columns();
    names.extend(options.data_columns());
    names
}

/// 交互射影の設定（`tol`・`max_iter`）。
fn absorption(options: &FEGMMOptions) -> FeAbsorption {
    FeAbsorption {
        tol: options.tol,
        max_iter: options.max_iter,
    }
}

/// `c_stat`の重み: `two_step`・`iterated`は`fit()`の重み、重み行列を持たない`one_step`は`Classical`。
fn c_statistic_weight(gmm_type: &FeGmmType) -> FeWeightType {
    match gmm_type {
        FeGmmType::TwoStep { weight } | FeGmmType::Iterated { weight, .. } => weight.clone(),
        FeGmmType::OneStep => FeWeightType::Classical,
    }
}

/// Estimation results for FEGMM (fixed effects generalized method of moments).
///
/// Structured data only (no `summary()`). All array-valued fields (`params`,
/// `std_errors`, etc.) share the same order as `param_names` (`x_exog`, then `x_endog`).
///
/// `test_stats` holds the z-statistics (normal distribution). `r_squared_*` are computed from the
/// structural residuals (`y - X b` with the actual endogenous regressors `X`) and can be
/// negative. `overid_*` is the Hansen J overidentification test, computed from the
/// within-transformed data; for the classical weight (and `gmm_type="one_step"`) its variance is
/// `e'e / (n - n_absorbed)`, where `n_absorbed` is the number of absorbed fixed-effect parameters.
/// The "robust" and "cluster" weights are not adjusted: the "cluster" weight needs no adjustment when
/// the clusters contain the entities (the entities themselves, or coarser groups), but does not hold
/// for clusters that cross the entities (such as time periods), and the "robust" weight makes the
/// statistic over-reject on a short panel.
///
/// The C statistic is not a field: it is the method `c_stat()`.
// `Clone`を派生しない: `source_data`（`Py<PyAny>`）がGILなしでは複製できない（`FEIVResult`と同じ理由）。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug)]
pub struct FEGMMResult {
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
    /// Distribution of `test_stats`; always `"normal"` for FEGMM.
    #[pyo3(get)]
    pub stat_dist: String,
    /// Degrees of freedom of the distribution of `test_stats`; always `None` (normal).
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
    /// Structural residuals `y - X b` of the within-transformed data, with the actual
    /// endogenous regressors `X`.
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
    /// Standard error type actually used (echoes `FEGMMOptions.cov_type`, normalized to lowercase).
    #[pyo3(get)]
    pub cov_type: String,
    /// Driscoll-Kraay bandwidth actually used. `None` unless `cov_type="dk"`.
    #[pyo3(get)]
    pub dk_bandwidth_used: Option<i64>,
    /// GMM variant actually used: `"one_step"`, `"two_step"` or `"iterated"`.
    #[pyo3(get)]
    pub gmm_type: String,
    /// Weight matrix actually used: `"classical"`, `"robust"` or `"cluster"`. `None` for
    /// `gmm_type="one_step"`, which does not use a weight matrix.
    #[pyo3(get)]
    pub gmm_weight_type: Option<String>,
    /// Number of GMM estimations performed (1 for `"one_step"`, 2 for `"two_step"`).
    #[pyo3(get)]
    pub n_iter: i64,
    /// Whether the iteration converged (always `True` for `"one_step"` and `"two_step"`).
    #[pyo3(get)]
    pub converged: bool,
    /// Statistic of the joint test that all slope coefficients are zero.
    #[pyo3(get)]
    pub wald_statistic: f64,
    #[pyo3(get)]
    pub wald_p_value: f64,
    /// Distribution of `wald_statistic`; always `"chi2"` for FEGMM.
    #[pyo3(get)]
    pub wald_dist: String,
    #[pyo3(get)]
    pub wald_df_num: Option<usize>,
    /// Always `None` (chi-squared).
    #[pyo3(get)]
    pub wald_df_denom: Option<usize>,
    #[pyo3(get)]
    pub r_squared_within: f64,
    #[pyo3(get)]
    pub r_squared_between: f64,
    #[pyo3(get)]
    pub r_squared_overall: f64,
    /// Hansen J overidentification statistic. `None` when the model is exactly identified.
    #[pyo3(get)]
    pub overid_statistic: Option<f64>,
    #[pyo3(get)]
    pub overid_p_value: Option<f64>,
    #[pyo3(get)]
    pub overid_df: Option<usize>,
    // `c_stat()`が入力を作り直すために保持する（`FEIVResult`と同じ）。`source_data`は`fit()`に渡された
    // `polars.DataFrame`（参照を増やすだけ）、`options`は渡した`FEGMMOptions`の複製。`source_data`と`options`は
    // `fit`が結果を組み立てた後でGILを持って付ける。
    source_data: Option<Py<PyAny>>,
    options: Option<FEGMMOptions>,
    x_exog: Vec<String>,
    x_endog: Vec<String>,
    instruments: Vec<String>,
}

crate::shared::pickle::impl_pickle_state!(FEGMMResult {
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
    gmm_type,
    gmm_weight_type,
    n_iter,
    converged,
    wald_statistic,
    wald_p_value,
    wald_dist,
    wald_df_num,
    wald_df_denom,
    r_squared_within,
    r_squared_between,
    r_squared_overall,
    overid_statistic,
    overid_p_value,
    overid_df,
    source_data,
    options,
    x_exog,
    x_endog,
    instruments,
});

impl FEGMMResult {
    /// 元のデータ（`fit()`に渡された`polars.DataFrame`）。`fit`が必ず付けるので`None`にはならない
    /// （pickleにも含まれる）。`None`なら到達不能な状態として`INTERNAL_ERROR`。
    fn retained_source<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.source_data
            .as_ref()
            .map(|source| source.bind(py).clone())
            .ok_or_else(|| internal_error("an FEGMMResult without its training data"))
    }

    /// `fit()`に渡した`FEGMMOptions`の複製。`retained_source`と同じく`None`にはならない。
    fn retained_options(&self) -> PyResult<FEGMMOptions> {
        self.options
            .clone()
            .ok_or_else(|| internal_error("an FEGMMResult without its options"))
    }
}

#[pymethods]
impl FEGMMResult {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        crate::shared::pickle::reduce(slf)
    }

    /// Number of groups (unique values) of each column of `effects`, as a `dict[str, int]`
    /// keyed by column name, in the order the columns were given.
    #[getter]
    fn n_groups<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        groups_dict(py, &self.effects, &self.group_counts)
    }

    /// C statistic (difference-in-Hansen test) of the exogeneity of endogenous variables
    /// and of the orthogonality of excluded instruments, computed on the within-transformed
    /// data.
    ///
    /// A post-estimation diagnostic: never computed by `fit()`. It re-reads the variables from
    /// the data passed to `fit()` (keep that data unchanged) and estimates the model again on
    /// every call. The model is
    /// estimated twice by two-step GMM, with and without the tested moment conditions, from
    /// the same covariance of the moment conditions; the statistic is the difference of the two
    /// Hansen J statistics. The weight follows `gmm_weight_type` (`cluster` included) for
    /// `gmm_type="two_step"` and `"iterated"`, and is `"classical"` for `gmm_type="one_step"`,
    /// which has no weight matrix. For the classical weight the variance of the moment
    /// conditions uses `n - n_absorbed` like the Hansen J test; the other weights are not
    /// adjusted, so the statistic over-rejects with a "robust" weight on a short panel.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     `variables` or `instruments` is not a list of str.
    /// ValidationError
    ///     Nothing is tested, a list is empty or repeats a name, a name is not an endogenous
    ///     variable (instrument) of the model, or too many instruments are tested.
    /// ComputationError
    ///     The statistic is undefined for these data (collinear instruments, a perfect fit,
    ///     or fewer clusters than the moment conditions of the full model).
    #[pyo3(signature = (variables=None, instruments=None))]
    fn c_stat(
        &self,
        py: Python<'_>,
        variables: Option<&Bound<'_, PyAny>>,
        instruments: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<CStatisticOutput> {
        guard(|| {
            // 型の検査 → 値（空・重複・名前）の検査の順は`resolve_c_stat_target`が持つ。
            let ResolvedCStatTarget {
                target,
                variables: tested_endog,
                instruments: tested_instruments,
            } = resolve_c_stat_target(variables, instruments, &self.x_endog, &self.instruments)?;
            let source = self.retained_source(py)?;
            let options = self.retained_options()?;
            let columns = FeIvColumns {
                y: self.dep_var_name.clone(),
                x_exog: self.x_exog.clone(),
                x_endog: self.x_endog.clone(),
                instruments: self.instruments.clone(),
                effects: self.effects.clone(),
            };
            let frame = select_columns(&source, "data", &data_columns(&columns, &options))?.frame;
            py.detach(|| {
                let built = build_fe_gmm_input(&frame, columns, &options)?;
                let weight = c_statistic_weight(&built.gmm_type);
                let c = fe_gmm_c_statistic(
                    &built.input,
                    built.effects,
                    &absorption(&options),
                    &target,
                    &weight,
                )
                .map_err(panel_error_to_pyerr)?;
                Ok(CStatisticOutput::from_engine(
                    c,
                    tested_endog,
                    tested_instruments,
                ))
            })
        })
    }
}

/// 入力を組み立てて`FeGmmEstimator`を推定する。
fn estimate_built(built: BuiltFeGmmInput, options: &FEGMMOptions) -> PyResult<FeGmmEstimatorParts> {
    let BuiltFeGmmInput {
        input,
        effects,
        cov_type,
        cov_type_name,
        gmm_type,
        gmm_type_name,
        weight_type_name,
        raise_on_non_convergence,
    } = built;
    let estimator = FeGmmEstimator::fit_with_absorption(
        input,
        effects,
        gmm_type,
        raise_on_non_convergence,
        cov_type,
        options.confidence_level,
        &absorption(options),
    )
    .map_err(panel_error_to_pyerr)?;
    Ok(FeGmmEstimatorParts {
        estimator,
        cov_type_name,
        gmm_type_name,
        weight_type_name,
    })
}

/// 推定した`FeGmmEstimator`と、結果に載せる正規形の名前。
struct FeGmmEstimatorParts {
    estimator: FeGmmEstimator,
    cov_type_name: String,
    gmm_type_name: String,
    weight_type_name: Option<String>,
}

/// Pythonから渡された`data`・列名・`options`を検証し、`FeGmmEstimator`を推定して`FEGMMResult`として返す。
///
/// # Errors
/// - `build_fe_gmm_input`が返すエラー（`ValidationError`）
/// - `FeGmmEstimator::fit_with_absorption`が返す`PanelError`（識別の順序条件・singleton・自由度不足・
///   分散ゼロ・完全適合・重み行列・共分散の検証・非収束等）は`panel_error_to_pyerr`で変換
pub(crate) fn fit(
    py: Python<'_>,
    data: SelectedData,
    columns: FeIvColumns,
    options: &FEGMMOptions,
) -> PyResult<FEGMMResult> {
    // 取り込み済みの`frame`以降はPythonオブジェクトに触れないので、GILを手放して走らせる。`source`は結果に持たせる。
    let SelectedData { source, frame } = data;
    let mut result = py.detach(|| fit_frame(&frame, columns, options))?;
    // `c_stat()`が入力を作り直すための元のデータとオプション（GILを持って付ける）。
    result.source_data = Some(source);
    result.options = Some(options.clone());
    Ok(result)
}

/// [`fit`]の純Rust部分（GILを手放した状態で走る）。
fn fit_frame(
    df: &DataFrame,
    columns: FeIvColumns,
    options: &FEGMMOptions,
) -> PyResult<FEGMMResult> {
    let built = build_fe_gmm_input(df, columns.clone(), options)?;
    let parts = estimate_built(built, options)?;
    result_from(parts, columns)
}

fn result_from(parts: FeGmmEstimatorParts, columns: FeIvColumns) -> PyResult<FEGMMResult> {
    let FeGmmEstimatorParts {
        estimator,
        cov_type_name,
        gmm_type_name,
        weight_type_name,
    } = parts;
    let FeIvColumns {
        x_exog,
        x_endog,
        instruments,
        effects,
        ..
    } = columns;
    let input = estimator.input();
    let params = mat_to_vec(estimator.params());
    let (wald_statistic, wald_p_value) =
        estimator
            .wald_statistic()
            .zip(estimator.wald_p_value())
            .ok_or_else(|| internal_error("the joint test of a FEGMM fit is missing"))?;

    Ok(FEGMMResult {
        wald_df_num: Some(params.len()),
        params,
        std_errors: mat_to_vec(estimator.std_errors()),
        cov_params: mat_to_rows(estimator.cov_params()),
        test_stats: mat_to_vec(estimator.test_stats()),
        stat_dist: "normal".to_string(),
        stat_df: None,
        p_values: mat_to_vec(estimator.p_values()),
        conf_lower: mat_to_vec(estimator.conf_lower()),
        conf_upper: mat_to_vec(estimator.conf_upper()),
        param_names: estimator.param_names().to_vec(),
        residuals: mat_to_vec(estimator.residuals()),
        dep_var_name: input.dep_var_name().to_string(),
        n_obs: input.nobs(),
        df_resid: estimator.df_resid(),
        df_model: estimator.df_model(),
        effects,
        group_counts: estimator.n_groups().to_vec(),
        cov_type: cov_type_name,
        dk_bandwidth_used: estimator.dk_bandwidth_used().map(|bw| bw as i64),
        gmm_type: gmm_type_name,
        gmm_weight_type: weight_type_name,
        n_iter: estimator.n_iter(),
        converged: estimator.converged(),
        wald_statistic,
        wald_p_value,
        wald_dist: "chi2".to_string(),
        wald_df_denom: None,
        r_squared_within: estimator.r_squared_within(),
        r_squared_between: estimator.r_squared_between(),
        r_squared_overall: estimator.r_squared_overall(),
        overid_statistic: estimator.hansen_j_statistic(),
        overid_p_value: estimator.hansen_j_p_value(),
        overid_df: estimator.hansen_j_df(),
        source_data: None,
        options: None,
        x_exog,
        x_endog,
        instruments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_options() -> FEGMMOptions {
        FEGMMOptions {
            cov_type: "cluster".to_string(),
            confidence_level: 0.95,
            cluster: None,
            dk_time: None,
            dk_bandwidth: None,
            gmm_type: None,
            gmm_weight_type: None,
            gmm_max_iter: None,
            gmm_tol: None,
            raise_on_non_convergence: None,
            tol: FeAbsorption::default().tol,
            max_iter: 10000,
        }
    }

    /// オプションの検査（値 → 組合せ）の判定表（`Err`になる/ならない）。エラー文言そのものは`PyErr`の
    /// DisplayがGILを要求するためpytest側（`test_fe_gmm_validation.py`）で確認する。
    fn options_are_ok(configure: impl FnOnce(&mut FEGMMOptions)) -> bool {
        let mut options = default_options();
        configure(&mut options);
        let Ok(values) = parse_option_values(&options) else {
            return false;
        };
        check_fe_gmm_options(&options, &values).is_ok()
    }

    #[test]
    fn defaults_resolve_to_a_two_step_fit_weighted_by_the_cluster() {
        let values = parse_option_values(&default_options()).unwrap();
        assert_eq!(values.gmm_type, GmmKind::TwoStep);
        assert_eq!(values.weight, WeightKind::Cluster);
        assert!(values.uses_weight());
        assert!(!values.is_iterated());
        assert!(options_are_ok(|_| {}));
    }

    #[test]
    fn hac_is_not_a_weight_type() {
        assert!(!options_are_ok(|o| o.gmm_weight_type = Some("hac".into())));
        assert!(options_are_ok(|o| o.gmm_weight_type = Some("Robust".into())));
    }

    #[test]
    fn the_weight_is_tied_to_the_gmm_type() {
        let one_step = |o: &mut FEGMMOptions| o.gmm_type = Some("one_step".into());
        assert!(options_are_ok(one_step));
        assert!(!options_are_ok(|o| {
            one_step(o);
            o.gmm_weight_type = Some("robust".into());
        }));
        // 反復の設定は`iterated`のときだけ。
        assert!(!options_are_ok(|o| o.gmm_max_iter = Some(10)));
        assert!(!options_are_ok(|o| o.gmm_tol = Some(1e-4)));
        assert!(!options_are_ok(|o| o.raise_on_non_convergence = Some(false)));
        assert!(options_are_ok(|o| {
            o.gmm_type = Some("iterated".into());
            o.gmm_max_iter = Some(10);
            o.gmm_tol = Some(1e-4);
            o.raise_on_non_convergence = Some(false);
        }));
    }

    #[test]
    fn cluster_is_used_by_the_cov_type_or_the_weight() {
        let cluster = |o: &mut FEGMMOptions| o.cluster = Some("g".into());
        // 既定（`cov_type="cluster"`・重み`cluster`）はどちらも使う。
        assert!(options_are_ok(cluster));
        // `cov_type`が使わなくても、重みが使えば有効。
        assert!(options_are_ok(|o| {
            cluster(o);
            o.cov_type = "hc1".into();
        }));
        // どちらも使わなければ拒否する。
        assert!(!options_are_ok(|o| {
            cluster(o);
            o.cov_type = "hc1".into();
            o.gmm_weight_type = Some("robust".into());
        }));
        assert!(!options_are_ok(|o| {
            cluster(o);
            o.cov_type = "classical".into();
            o.gmm_type = Some("one_step".into());
        }));
    }

    #[test]
    fn dk_options_need_cov_type_dk_and_dk_requires_dk_time() {
        assert!(!options_are_ok(|o| o.dk_time = Some("t".into())));
        assert!(!options_are_ok(|o| o.dk_bandwidth = Some(1)));
        assert!(!options_are_ok(|o| o.cov_type = "dk".into()));
        assert!(options_are_ok(|o| {
            o.cov_type = "dk".into();
            o.dk_time = Some("t".into());
            o.dk_bandwidth = Some(1);
        }));
    }

    /// 値の検査は`cov_type` → `gmm_type` → `gmm_weight_type`の順に、最初の誤りで返る（どれが
    /// 報告されたかは、直す順に1つずつ消していって確かめる）。
    #[test]
    fn values_are_checked_in_the_order_cov_type_gmm_type_weight() {
        let mut options = default_options();
        options.cov_type = "bad".into();
        options.gmm_type = Some("bad".into());
        options.gmm_weight_type = Some("bad".into());
        assert!(parse_option_values(&options).is_err());
        options.cov_type = "hc1".into();
        assert!(parse_option_values(&options).is_err());
        options.gmm_type = Some("one_step".into());
        assert!(parse_option_values(&options).is_err());
        options.gmm_weight_type = Some("robust".into());
        assert!(parse_option_values(&options).is_ok());
    }

    #[test]
    fn the_chosen_names_are_the_canonical_lowercase_spellings() {
        let mut options = default_options();
        options.gmm_type = Some("ITERATED".into());
        options.gmm_weight_type = Some("Classical".into());
        let values = parse_option_values(&options).unwrap();
        assert_eq!(values.gmm_type.name(), "iterated");
        assert_eq!(values.weight.name(), "classical");
        assert_eq!(values.cov.name(), "cluster");
    }

    #[test]
    fn c_statistic_weight_follows_the_fit_and_falls_back_to_classical() {
        assert_eq!(
            c_statistic_weight(&FeGmmType::OneStep),
            FeWeightType::Classical
        );
        assert_eq!(
            c_statistic_weight(&FeGmmType::TwoStep {
                weight: FeWeightType::Robust
            }),
            FeWeightType::Robust
        );
        assert_eq!(
            c_statistic_weight(&FeGmmType::Iterated {
                weight: FeWeightType::Classical,
                max_iter: 5,
                tol: 1e-6
            }),
            FeWeightType::Classical
        );
    }
}
