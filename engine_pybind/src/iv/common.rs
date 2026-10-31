//! `iv`系統（2SLS/GMM）で共有するユーティリティ。
//!
//! `.claude/rules/rust-style.md`「ファイル・ディレクトリ構成」: 系統内で共有するロジックは
//! `<系統>/common.rs`に置く（`engine_pybind/src/linear/common.rs`と同じ位置づけ）。
//! `IVOptions`/`IVResult`/`build_iv_input`は2SLS/GMMどちらの`estimator`でも共有する
//! （`fit_iv`という単一エントリポイントの背後で`estimator`により推定方式を切り替える設計、
//! `docs/spec/iv-spec.md`1.2節）ため、系統内共有ロジックの
//! 置き場所という位置づけに素直に合致する（`two_sls.rs`/`gmm.rs`のような手法ごとの
//! ファイル分割はしない）。
//!
//! `IvError`の`Common`バリアント（`engine::shared::error::CommonError`）も、`IvError`の
//! `ErrorInfo::report()`が同じ`code`・`params`に変換する（系統ごとに同じ判定ロジックを
//! 重複させない）。
//!
//! ## 実装の経緯（要点のみ、詳細は各コミット・`engine/src/iv/CLAUDE.md`参照）
//!
//! `IVOptions`/`IVResult`のpyclass定義・`build_iv_input`→`TwoSlsEstimator::fit`
//! への配線→弱操作変数診断・Wu-Hausman・Sargan→`first_stage()`の順に段階実装した。
//! **`estimator="gmm"`は当初`GmmEstimator`（engine側）が点推定のみのスコープだったため
//! 長らく`ValidationError`で弾いていたが、GMM側のcov_type対応（完了条件だったが
//! 実装漏れだったことが発覚、`gmm.rs`参照）を実装したうえで、本ファイルでも
//! 実際に配線した**（`fit_iv`から両`estimator`を呼び分ける）。
//!
//! ## `first_stage()`/`weak_instrument_f_statistics`は`estimator`に依存しない共通ロジック
//!
//! 第一段階回帰（`x_endog[j] ~ x_exog + instruments`）・弱操作変数診断（部分F統計量）は、
//! `engine::iv::common::compute_first_stage`（2SLS/GMM間で共有、`engine/src/iv/CLAUDE.md`
//! 参照）を`fit`が`estimator`によらず常に呼ぶことで、GMMでも2SLSと同じ診断情報を提供する
//! （ユーザー確認済み）。`TwoSlsEstimator::fit`は内部でも同じ関数を呼ぶため、
//! `estimator="2sls"`では第一段階回帰が二重計算になるが、OLS自体が軽量なため許容する
//! （`GmmEstimator`のように第一段階回帰を必要としない推定器に合わせて`IVResult`側を
//! 単純にする方を優先した設計判断）。
//!
//! `IVResult`は元々`estimator: TwoSlsEstimator`という2SLS専用の非公開フィールドで
//! `first_stage()`を実装していたが、GMM配線にあたり`estimator`非依存の表現に置き換え、
//! pickle対応にあたり、`fit`が第一段階の`OlsEstimator`（設計行列を持つ）を`OLSResult`へ
//! 変換して`first_stage: Vec<(String, OLSResult)>`として保持する形にした（変換は
//! `linear::ols::ols_estimator_to_result`を再利用、抽出済み）。`first_stage()`は`clone_ref`で
//! 複製して返すだけで、pickleでは各`OLSResult`がそのまま状態に入る。
//!
//! ## GMMの`gmm_weight_type`（`IVOptions.gmm_weight_type`/`cluster`/`hac_lags`/`hac_time`）
//!
//! `gmm_weight_type`は`cov_type`とは独立の軸（点推定に使う重み行列の選択、`engine::iv::gmm`の
//! モジュールdocコメント参照）だが、`cluster`/`hac_lags`/`hac_time`は`cov_type`と
//! 共用する（`IVOptions`に別フィールドを増やさない設計、`resolve_weight_type`参照）。
//! `gmm_weight_type="cluster"`かつ`cov_type="cluster"`のように両軸が同じクラスター列を
//! 参照する使い方を主に想定するが、`gmm_weight_type`と`cov_type`が異なる場合でも同じ列を
//! 共用する（別々のクラスター変数を使い分けたいニーズが出てきたら別フィールド化を検討）。
//!
//! `wu_hausman_statistic`/`wu_hausman_p_value`は`estimator="gmm"`では常に`None`
//! （`GmmEstimator`はWu-Hausman検定を持たない、`docs/spec/iv-spec.md`3.6節はTwoSlsEstimator
//! のみのスコープ）。`overid_statistic`/`overid_p_value`は`estimator="gmm"`では
//! `GmmEstimator::hansen_j_statistic()`/`hansen_j_p_value()`から配線する
//! （`estimator="2sls"`のSargan検定と同じ`Option<f64>`同士の代入）。

use std::collections::HashMap;

use engine::iv::common::{
    IvError, IvInput, compute_first_stage, validate_observation_counts,
    validate_outcome_not_constant, validate_structural_cluster_count,
};
use engine::iv::gmm::{GmmEstimator, GmmType, WeightType};
use engine::iv::two_sls::TwoSlsEstimator;
use engine::linear::ols::CovType as EngineCovType;
use polars::prelude::DataFrame;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

use crate::linear::common::{CovKind, parse_cov_kind, require_hac_time, resolve_cov_type};
use crate::linear::ols::{OLSResult, ols_estimator_to_result};
use crate::shared::column_error::ColumnSource;
use crate::shared::column_extraction::{
    SelectedData, extract_f64_column, extract_optional_cluster_groups, extract_time_order_ranks,
};
use crate::shared::errors::{guard, to_pyerr, unknown_option_value};
use crate::shared::mat_conversion::mat_to_vec;
use crate::shared::option_repr::PyRepr;
use crate::shared::option_values::{
    extract_strict_float, extract_strict_opt_column, extract_strict_opt_float,
    extract_strict_opt_int, extract_strict_opt_text, extract_strict_text,
};
use crate::shared::validation::{
    RoleValue, reject_unused_option, validate_no_const_collision, validate_no_duplicate_roles,
    validate_no_duplicate_within_role, validate_x_non_empty,
};

/// `engine::iv::common::IvError`をPython例外に変換する。
///
/// `IvError`（`engine`クレート）と`PyErr`（`pyo3`クレート）はどちらもこのクレートの外で
/// 定義された型のため、orphan rule（`impl`の対象は自クレート内で定義されたトレイトか型の
/// どちらかを含む必要がある）により`impl From<IvError> for PyErr`は書けない。関数として
/// 実装し、呼び出し側で`.map_err(iv_error_to_pyerr)?`する（`least_squares_error_to_pyerr`と
/// 同じ理由、`engine_pybind/src/linear/common.rs`参照）。
///
/// `FirstStageFailed`/`SecondStageFailed`/`HausmanRegressionFailed`は、内部で委譲する
/// `OlsEstimator::fit`の失敗を包んだもの。`report()`が原因の`code`に`stage`を付けて返すので、
/// ここでは分岐しない（`ValidationError`/`ComputationError`もカタログの`code`で決まる）。
pub(crate) fn iv_error_to_pyerr(err: IvError) -> PyErr {
    to_pyerr(&err)
}

/// Estimation options for IV (2SLS/GMM).
///
/// All arguments are keyword-only.
///
/// A single options class serves both estimation methods; fields that apply to
/// only one estimator are documented as such.
///
/// Options by purpose:
///
/// - Estimator: `estimator`.
/// - Standard errors: `cov_type`; with `"cluster"` also `cluster`; with
///   `"hac"` also `hac_time` (required) and `hac_lags`.
/// - GMM only: `gmm_type`, `gmm_weight_type` (the weight matrix of the point
///   estimate, with the same `cluster`, `hac_time` and `hac_lags` fields).
/// - GMM, `gmm_type="iterated"` only: `gmm_max_iter`, `gmm_tol`,
///   `raise_on_non_convergence`.
/// - Confidence intervals: `confidence_level`.
/// - Model: `include_intercept`.
// 各フィールドの意味・既定値の根拠は`docs/spec/iv-spec.md`。
// module/from_py_objectの理由は`OLSOptions`/`LogitOptions`と同じ
// （`engine_pybind/src/linear/ols.rs`のコメント参照）。
#[pyclass(from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct IVOptions {
    /// Estimation estimator: "2sls" (default) or "gmm". Case-insensitive.
    #[pyo3(get)]
    pub estimator: String,

    /// Standard error type: one of "classical", "hc0" through "hc3", "hac", "cluster".
    /// Case-insensitive. For `estimator="gmm"`, this is independent of `gmm_weight_type`
    /// (the weight matrix used for point estimation, see `gmm_weight_type` below).
    #[pyo3(get)]
    pub cov_type: String,

    /// Whether the engine should automatically add an intercept column to `x_exog`.
    /// `x_endog`/`instruments` never get an automatic intercept column.
    #[pyo3(get, set)]
    pub include_intercept: bool,

    /// Confidence level for confidence intervals, in the range (0, 1).
    /// Defaults to 0.95 (a 95% confidence interval).
    #[pyo3(get)]
    pub confidence_level: f64,

    /// Column name to use as the cluster group key. Used by `cov_type="cluster"` and,
    /// with `estimator="gmm"`, by `gmm_weight_type="cluster"` (also when
    /// `cov_type` is not "cluster"). Specifying it when neither uses it raises
    /// `ValidationError`.
    #[pyo3(get)]
    pub cluster: Option<String>,

    /// Number of lags (bandwidth) for HAC (Newey-West), used by `cov_type="hac"` and, with
    /// `estimator="gmm"`, by `gmm_weight_type="hac"`. When `None`, computed automatically.
    /// Specifying it when neither uses it raises `ValidationError`.
    #[pyo3(get)]
    pub hac_lags: Option<i64>,

    /// Column name giving the time order for HAC, used by `cov_type="hac"` and, with
    /// `estimator="gmm"`, by `gmm_weight_type="hac"`. Required when either is "hac"
    /// (omitting it raises `ValidationError`): the row order of the data is never assumed
    /// to be the time order. Specifying it when neither uses it raises `ValidationError`.
    #[pyo3(get)]
    pub hac_time: Option<String>,

    /// GMM estimation type: "one_step" (weight matrix `(Z'Z)^-1` only), "two_step"
    /// (efficient two-step GMM), or "iterated" (repeat until convergence).
    /// Case-insensitive. `None` (default) means "two_step" for `estimator="gmm"`.
    /// Specifying it with `estimator="2sls"` raises `ValidationError`.
    #[pyo3(get)]
    pub gmm_type: Option<String>,

    /// Weight matrix used for GMM point estimation: one of "classical" (homoskedastic),
    /// "robust" (heteroskedasticity-robust), "cluster", "hac" (Newey-West). Same vocabulary
    /// as `cov_type`. Case-insensitive. `None` (default) means "classical" for
    /// `gmm_type="two_step"`/`"iterated"`. Specifying it with `estimator="2sls"` or
    /// `gmm_type="one_step"` (which does not use a weight matrix) raises `ValidationError`.
    /// "cluster"/"hac" draw from the same `cluster`/`hac_lags`/`hac_time` fields as
    /// `cov_type` (there are no separate fields for GMM).
    #[pyo3(get)]
    pub gmm_weight_type: Option<String>,

    /// Maximum number of GMM estimations for `gmm_type="iterated"`, counting the initial
    /// estimate; an integer from 3 to 10,000 (use `gmm_type="two_step"` for two steps).
    /// `None` (default) means 100. Specifying it with any other `gmm_type` or with
    /// `estimator="2sls"` raises `ValidationError`.
    #[pyo3(get)]
    pub gmm_max_iter: Option<i64>,

    /// Convergence tolerance for `gmm_type="iterated"`: iteration stops once every
    /// coefficient changes by less than this (relative/absolute mix). `None` (default)
    /// means 1e-6. Specifying it with any other `gmm_type` or with `estimator="2sls"`
    /// raises `ValidationError`.
    #[pyo3(get)]
    pub gmm_tol: Option<f64>,

    /// Whether to raise an error if `gmm_type="iterated"` does not converge within
    /// `gmm_max_iter`. If `False`, returns the result with `converged=False` instead of
    /// raising. `None` (default) means `True`. Specifying it with any other `gmm_type`
    /// (which never checks convergence) or with `estimator="2sls"` raises
    /// `ValidationError`.
    #[pyo3(get, set)]
    pub raise_on_non_convergence: Option<bool>,
}

crate::shared::pickle::impl_pickle_state!(IVOptions {
    estimator,
    cov_type,
    include_intercept,
    confidence_level,
    cluster,
    hac_lags,
    hac_time,
    gmm_type,
    gmm_weight_type,
    gmm_max_iter,
    gmm_tol,
    raise_on_non_convergence,
});

impl IVOptions {
    /// このオプションが指名するデータの列（`cluster`・`hac_time`）。使われるかどうか（`cov_type`等）には
    /// よらず、指名されたものを全て返す。`fit()`はこれらと`y`・`x`等だけをデータから取り込む。
    pub(crate) fn data_columns(&self) -> Vec<String> {
        self.cluster.iter().chain(&self.hac_time).cloned().collect()
    }
}

#[pymethods]
impl IVOptions {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        crate::shared::pickle::reduce(slf)
    }

    #[new]
    #[pyo3(signature = (
        *,
        estimator = "2sls".to_string(),
        cov_type = "classical".to_string(),
        include_intercept = true,
        confidence_level = 0.95,
        cluster = None,
        hac_lags = None,
        hac_time = None,
        gmm_type = None,
        gmm_weight_type = None,
        gmm_max_iter = None,
        gmm_tol = None,
        raise_on_non_convergence = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = crate::shared::option_values::estimator_arg)] estimator: String,
        #[pyo3(from_py_with = crate::shared::option_values::cov_type_arg)] cov_type: String,
        include_intercept: bool,
        #[pyo3(from_py_with = crate::shared::option_values::confidence_level_arg)] confidence_level: f64,
        #[pyo3(from_py_with = crate::shared::option_values::cluster_arg)] cluster: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::hac_lags_arg)] hac_lags: Option<i64>,
        #[pyo3(from_py_with = crate::shared::option_values::hac_time_arg)] hac_time: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::gmm_type_arg)] gmm_type: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::gmm_weight_type_arg)]
        gmm_weight_type: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::gmm_max_iter_arg)] gmm_max_iter: Option<
            i64,
        >,
        #[pyo3(from_py_with = crate::shared::option_values::gmm_tol_arg)] gmm_tol: Option<f64>,
        raise_on_non_convergence: Option<bool>,
    ) -> Self {
        Self {
            estimator,
            cov_type,
            include_intercept,
            confidence_level,
            cluster,
            hac_lags,
            hac_time,
            gmm_type,
            gmm_weight_type,
            gmm_max_iter,
            gmm_tol,
            raise_on_non_convergence,
        }
    }

    #[setter]
    fn set_confidence_level(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.confidence_level = extract_strict_float(value, "confidence_level")?;
        Ok(())
    }

    #[setter]
    fn set_hac_lags(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.hac_lags = extract_strict_opt_int(value, "hac_lags")?;
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

    #[setter]
    fn set_estimator(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.estimator = extract_strict_text(value, "estimator")?;
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
    fn set_hac_time(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.hac_time = extract_strict_opt_column(value, "hac_time")?;
        Ok(())
    }

    #[setter]
    fn set_gmm_weight_type(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.gmm_weight_type = extract_strict_opt_text(value, "gmm_weight_type")?;
        Ok(())
    }

    #[setter]
    fn set_gmm_type(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.gmm_type = extract_strict_opt_text(value, "gmm_type")?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "IVOptions(estimator={}, cov_type={}, include_intercept={}, \
             confidence_level={}, cluster={}, hac_lags={}, hac_time={}, \
             gmm_type={}, gmm_weight_type={}, gmm_max_iter={}, gmm_tol={}, \
             raise_on_non_convergence={})",
            self.estimator.py_repr(),
            self.cov_type.py_repr(),
            self.include_intercept.py_repr(),
            self.confidence_level.py_repr(),
            self.cluster.py_repr(),
            self.hac_lags.py_repr(),
            self.hac_time.py_repr(),
            self.gmm_type.py_repr(),
            self.gmm_weight_type.py_repr(),
            self.gmm_max_iter.py_repr(),
            self.gmm_tol.py_repr(),
            self.raise_on_non_convergence.py_repr(),
        )
    }
}

/// Estimation results for IV (2SLS/GMM).
///
/// Structured data only (no `summary()`); see `docs/spec/iv-spec.md`
/// section 2. All array-valued fields (`params`, `std_errors`, etc.) share the same
/// order as `param_names`.
///
/// `test_stats` holds the t-statistics (`estimator="2sls"`) or z-statistics
/// (`estimator="gmm"`), depending on which distribution the fitted model uses for inference
/// (`docs/spec/iv-spec.md` 3.2節); `stat_dist`/`stat_df` say which distribution (and, for
/// `"t"`, how many degrees of freedom) that is.
///
/// `first_stage()`（内生変数ごとの第一段階回帰結果）はここにフィールドとして含めない。
/// `fit()`の戻り値本体には含めず別メソッドとして公開する（`docs/spec/iv-spec.md`2章、
/// 実装済み）。`predict()`/`marginal_effects()`用に`LogitResult`/
/// `ProbitResult`が推定量そのものを非公開フィールド`estimator`として保持するのと同じ
/// パターンだが、`IVResult`は`estimator`（2sls/gmm）非依存の非公開フィールド`first_stage:
/// Vec<(String, OLSResult)>`から`first_stage()`を構築する（下記`first_stage`フィールド参照。
/// 当初は`estimator: TwoSlsEstimator`という2sls専用のフィールドだったが、GMM配線時に
/// estimator非依存の表現へ置き換えた——`engine::iv::common::compute_first_stage`が
/// `estimator`によらず同じ第一段階回帰を計算するため、`fit`でこの表現に詰め替えるだけで済む）。
///
/// `fit_iv` (`fit` in this file) constructs and returns it. The core fields above are
/// populated by `TwoSlsEstimator` (`estimator="2sls"`) or `GmmEstimator` (`estimator="gmm"`).
/// `weak_instrument_f_statistics`/`first_stage` are populated from `engine::iv::common::
/// compute_first_stage`, independent of `estimator` (module docstring参照).
/// `wu_hausman_statistic`/`wu_hausman_p_value` are populated from `TwoSlsEstimator::
/// wu_hausman_statistic()`/`wu_hausman_p_value()` for `estimator="2sls"`;
/// always `None` for `estimator="gmm"` (`GmmEstimator` has no Wu-Hausman test).
/// `overid_statistic`/`overid_p_value` are populated from `TwoSlsEstimator::
/// sargan_statistic()`/`sargan_p_value()` (Sargan test, `estimator="2sls"`) or
/// `GmmEstimator::hansen_j_statistic()`/`hansen_j_p_value()` (Hansen J test,
/// `estimator="gmm"`).
// `IVResult`はRust側で組み立ててPythonに返すだけの型で、Python側からの生成・引数として
// 受け取ることは想定していないため`skip_from_py_object`（`IVOptions`の`from_py_object`とは
// 対照的、`OLSResult`/`LogitResult`と同じ理由）。
//
// `Clone`を派生しない: `first_stage`の要素`OLSResult`が、GILなしでは複製できない`Py<PyAny>`
// （`source_data`）を持つため`Clone`を実装していない。複製が要るときは`OLSResult::clone_ref`を使う。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug)]
pub struct IVResult {
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
    pub residuals: Vec<f64>,
    #[pyo3(get)]
    pub dep_var_name: String,
    #[pyo3(get)]
    pub n_obs: usize,
    #[pyo3(get)]
    pub df_resid: usize,
    #[pyo3(get)]
    pub df_model: usize,
    /// Whether GMM iteration converged (`estimator="gmm"` only). Always `true` for
    /// `estimator="2sls"` (2SLS is a closed-form, non-iterative estimator, so convergence is
    /// trivially satisfied). Also always `true` for `gmm_type="one_step"`/`"two_step"` —
    /// convergence is only actually checked for `gmm_type="iterated"`
    /// (`docs/spec/iv-spec.md` 3.3節).
    #[pyo3(get)]
    pub converged: bool,
    /// Number of GMM estimations actually run, counting the initial estimate
    /// (`estimator="gmm"` only): 1 for "one_step", 2 for "two_step", at most `gmm_max_iter`
    /// for "iterated". Always `1` for `estimator="2sls"`.
    #[pyo3(get)]
    pub n_iter: i64,
    /// Standard error type actually used (echoes `IVOptions.cov_type`, normalized to
    /// lowercase; e.g. `"classical"`, `"hc1"`, `"hac"`, `"cluster"`).
    #[pyo3(get)]
    pub cov_type: String,
    /// Number of HAC (Newey-West) lags actually used: the explicit `hac_lags` if given,
    /// otherwise the value chosen automatically, `floor(4 * (n / 100) ^ (2 / 9))`. `Some` when
    /// `cov_type="hac"` or, with `estimator="gmm"` and `gmm_type` `"two_step"`/`"iterated"`,
    /// `gmm_weight_type="hac"` (both share `hac_lags`, so the value is the same); `None`
    /// otherwise.
    #[pyo3(get)]
    pub hac_lags_used: Option<i64>,
    /// Estimator actually used (echoes `IVOptions.estimator`, normalized to
    /// lowercase): `"2sls"` or `"gmm"`.
    #[pyo3(get)]
    pub estimator: String,
    /// Weight matrix actually used for GMM point estimation (echoes
    /// `IVOptions.gmm_weight_type`, normalized to lowercase; e.g. `"classical"`,
    /// `"robust"`, `"cluster"`, `"hac"`). `Some` only for `estimator="gmm"` (mirrors
    /// `overid_statistic`/`wu_hausman_statistic`'s use of `None` for "not applicable to
    /// this estimator"); always `None` for `estimator="2sls"`, which has no such concept, and
    /// for `gmm_type="one_step"`, which does not use a weight type.
    #[pyo3(get)]
    pub gmm_weight_type: Option<String>,
    /// GMM estimation type actually used (echoes `IVOptions.gmm_type`, normalized to
    /// lowercase): `"one_step"`, `"two_step"` or `"iterated"`. `None` for `estimator="2sls"`.
    #[pyo3(get)]
    pub gmm_type: Option<String>,
    #[pyo3(get)]
    pub wald_statistic: f64,
    #[pyo3(get)]
    pub wald_p_value: f64,
    /// Distribution of `wald_statistic`: `"f"` for `estimator="2sls"` (the Wald statistic
    /// divided by the number of slope coefficients), `"chi2"` for `estimator="gmm"` (the
    /// undivided Wald statistic).
    #[pyo3(get)]
    pub wald_dist: String,
    /// Numerator degrees of freedom of `wald_statistic` (number of slope coefficients;
    /// `None` when there are none and the statistic is NaN).
    #[pyo3(get)]
    pub wald_df_num: Option<usize>,
    /// Denominator degrees of freedom of `wald_statistic` for `"f"` (`df_resid`, or `G - 1`
    /// with cluster-robust inference). `None` for `"chi2"` and when the statistic is NaN.
    #[pyo3(get)]
    pub wald_df_denom: Option<usize>,
    #[pyo3(get)]
    pub r_squared: f64,
    #[pyo3(get)]
    pub adj_r_squared: f64,
    /// Weak-instrument diagnostic: the partial F-statistic for each endogenous
    /// variable (keyed by variable name), testing the excluded instruments' joint
    /// significance after partialling out `x_exog` (`docs/spec/iv-spec.md` 3.4節).
    /// **Not** the same as the plain F-statistic of the corresponding regression in
    /// `first_stage()`, which includes `x_exog`'s contribution too. Empty when
    /// `x_endog=[]`. Computed the same way for both `estimator="2sls"` and `estimator="gmm"`
    /// (`engine::iv::common::compute_first_stage`, module docstring参照).
    #[pyo3(get)]
    pub weak_instrument_f_statistics: HashMap<String, f64>,
    /// Numerator degrees of freedom of the weak-instrument F statistics (number of excluded
    /// instruments; the same for every endogenous variable). `None` when `x_endog=[]`.
    #[pyo3(get)]
    pub weak_instrument_f_df_num: Option<usize>,
    /// Denominator degrees of freedom of the weak-instrument F statistics (residual df of
    /// the first-stage regressions; the same for every endogenous variable).
    #[pyo3(get)]
    pub weak_instrument_f_df_denom: Option<usize>,
    /// Overidentification test statistic: Sargan (`estimator="2sls"`) or Hansen J
    /// (`estimator="gmm"`). `None` when just-identified (`len(instruments) ==
    /// len(x_endog)`, degrees of freedom 0), per `docs/spec/iv-spec.md` 3.5節.
    #[pyo3(get)]
    pub overid_statistic: Option<f64>,
    #[pyo3(get)]
    pub overid_p_value: Option<f64>,
    /// Degrees of freedom of the chi-squared overidentification test
    /// (`len(instruments) - len(x_endog)`); `None` when just identified.
    #[pyo3(get)]
    pub overid_df: Option<usize>,
    /// Wu-Hausman endogeneity test statistic (joint test over all endogenous
    /// variables, regression-based / `wooldridge_regression` formulation,
    /// `docs/spec/iv-spec.md` 3.6節). Always computed under the `cov_type` passed to
    /// `fit()` (unlike `weak_instrument_f_statistics`, which is always classical;
    /// `linearmodels`' `wooldridge_regression` uses the same covariance as the
    /// underlying model, and this mirrors that). `None` when there are no endogenous
    /// variables to test (`x_endog=[]`), or when the augmented regression cannot be
    /// estimated (e.g. the first-stage residual has zero variance, or there are too
    /// few observations for the extra residual columns) — neither case fails `fit()`
    /// itself, since the other results remain valid. **Always `None` for
    /// `estimator="gmm"`** (`GmmEstimator` does not implement this test; `docs/spec/iv-spec.md`
    /// 3.6節's implementation is `TwoSlsEstimator`-only).
    #[pyo3(get)]
    pub wu_hausman_statistic: Option<f64>,
    #[pyo3(get)]
    pub wu_hausman_p_value: Option<f64>,
    /// Numerator degrees of freedom of the Wu-Hausman F test; `None` when it is `None`.
    #[pyo3(get)]
    pub wu_hausman_df_num: Option<usize>,
    /// Denominator degrees of freedom of the Wu-Hausman F test; `None` when it is `None`.
    #[pyo3(get)]
    pub wu_hausman_df_denom: Option<usize>,
    /// `first_stage()`が読む。Python側には公開しない（`OLSResult`の`fitted_values`/
    /// `has_intercept`、`LogitResult`/`ProbitResult`の`estimator`と同じ位置づけ）。
    /// `estimator`非依存（`engine::iv::common::compute_first_stage`から構築、モジュール
    /// docコメント参照）。
    first_stage: Vec<(String, OLSResult)>,
}

// 第一段階の結果（`OLSResult`）は、それぞれの`__reduce__`でpickleされるオブジェクトとして状態に入れる。
impl crate::shared::pickle::PickleField for Vec<(String, OLSResult)> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let items = self
            .iter()
            .map(|(name, result)| Ok((name.clone(), Py::new(py, result.clone_ref(py))?)))
            .collect::<PyResult<Vec<_>>>()?;
        items.into_bound_py_any(py)
    }

    fn from_py(obj: &Bound<'_, PyAny>) -> Option<Self> {
        let items: Vec<(String, Bound<'_, PyAny>)> = obj.extract().ok()?;
        items
            .into_iter()
            .map(|(name, item)| {
                let result = item.cast::<OLSResult>().ok()?.try_borrow().ok()?;
                Some((name, result.clone_ref(obj.py())))
            })
            .collect()
    }
}

crate::shared::pickle::impl_pickle_state!(IVResult {
    params,
    std_errors,
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
    converged,
    n_iter,
    cov_type,
    hac_lags_used,
    estimator,
    gmm_weight_type,
    gmm_type,
    wald_statistic,
    wald_p_value,
    wald_dist,
    wald_df_num,
    wald_df_denom,
    r_squared,
    adj_r_squared,
    weak_instrument_f_statistics,
    weak_instrument_f_df_num,
    weak_instrument_f_df_denom,
    overid_statistic,
    overid_p_value,
    overid_df,
    wu_hausman_statistic,
    wu_hausman_p_value,
    wu_hausman_df_num,
    wu_hausman_df_denom,
    first_stage,
});

#[pymethods]
impl IVResult {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        crate::shared::pickle::reduce(slf)
    }

    /// Per-endogenous-variable first-stage regression results
    /// (`x_endog[i] ~ x_exog + instruments`), keyed by the endogenous variable name.
    ///
    /// Each value is a full `OLSResults` (the same type OLS's `fit_ols` returns) — the
    /// first stage is a genuine, valid OLS regression in its own right, so no IV-specific
    /// result type is needed (`docs/spec/iv-spec.md` 2章). Its `f_statistic`/`f_p_value`
    /// include `x_exog`'s contribution and are **not** the weak-instrument partial
    /// F-statistic (`weak_instrument_f_statistics`, computed separately).
    /// Computed the same way for both `estimator="2sls"` and `estimator="gmm"` (module
    /// docstring参照).
    fn first_stage(&self, py: Python<'_>) -> PyResult<HashMap<String, OLSResult>> {
        guard(|| {
            Ok(self
                .first_stage
                .iter()
                .map(|(name, result)| (name.clone(), result.clone_ref(py)))
                .collect())
        })
    }
}

/// `gmm_max_iter`/`gmm_tol`の実効既定値（`IVOptions`の既定値は`None`のため、使われる
/// モードでここに解決する。`gmm_type`/`gmm_weight_type`の既定値は[`parse_iv_option_values`]）。
const DEFAULT_GMM_MAX_ITER: usize = 100;
const DEFAULT_GMM_TOL: f64 = 1e-6;

/// `IVOptions.estimator`の種類。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IvEstimator {
    TwoSls,
    Gmm,
}

impl IvEstimator {
    /// `IVResult.estimator`に格納する正規形の名前。
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::TwoSls => "2sls",
            Self::Gmm => "gmm",
        }
    }
}

/// `IVOptions.gmm_type`の種類。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GmmKind {
    OneStep,
    TwoStep,
    Iterated,
}

impl GmmKind {
    fn name(self) -> &'static str {
        match self {
            Self::OneStep => "one_step",
            Self::TwoStep => "two_step",
            Self::Iterated => "iterated",
        }
    }
}

/// `IVOptions.gmm_weight_type`の種類。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WeightKind {
    Classical,
    Robust,
    Cluster,
    Hac,
}

impl WeightKind {
    fn name(self) -> &'static str {
        match self {
            Self::Classical => "classical",
            Self::Robust => "robust",
            Self::Cluster => "cluster",
            Self::Hac => "hac",
        }
    }
}

/// 文字列だけで決まるオプションの値。`gmm_type`/`gmm_weight_type`は、指定が無ければ実効
/// 既定値（`two_step`/`classical`）で、指定の有無は`IVOptions`側で見る。
///
/// IVのオプションの検査は、DataFrameを見ない順に3つに分ける
/// （`docs/guide/validation.md`「Order in which errors are reported」）。
///
/// - 値: [`parse_iv_option_values`]が`estimator` → `cov_type` → `gmm_type` →
///   `gmm_weight_type`の順に文字列を種類にする。使われないオプションの未知の値も、
///   ここで報告する（値の誤りは、使われるかどうかより根本的なため）
/// - 組合せ: [`check_iv_options`]が使われないオプションの指定と、必須の`hac_time`の
///   欠落を検査する
/// - 列: [`resolve_gmm_setup`]と`linear::common::resolve_cov_type`が列を抽出する
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IvOptionValues {
    pub(crate) estimator: IvEstimator,
    pub(crate) cov: CovKind,
    pub(crate) gmm_type: GmmKind,
    pub(crate) weight: WeightKind,
}

impl IvOptionValues {
    fn is_gmm(&self) -> bool {
        self.estimator == IvEstimator::Gmm
    }

    /// 重み行列（`gmm_weight_type`）を使うか。`estimator="gmm"`かつ`gmm_type`が
    /// `two_step`/`iterated`のとき。
    fn uses_weight(&self) -> bool {
        self.is_gmm() && self.gmm_type != GmmKind::OneStep
    }

    fn is_iterated(&self) -> bool {
        self.is_gmm() && self.gmm_type == GmmKind::Iterated
    }
}

fn parse_estimator(estimator: &str) -> PyResult<IvEstimator> {
    match estimator.to_lowercase().as_str() {
        "2sls" => Ok(IvEstimator::TwoSls),
        "gmm" => Ok(IvEstimator::Gmm),
        _ => Err(unknown_option_value(
            "estimator",
            estimator,
            &["2sls", "gmm"],
        )),
    }
}

fn parse_gmm_kind(gmm_type: &str) -> PyResult<GmmKind> {
    match gmm_type.to_lowercase().as_str() {
        "one_step" => Ok(GmmKind::OneStep),
        "two_step" => Ok(GmmKind::TwoStep),
        "iterated" => Ok(GmmKind::Iterated),
        _ => Err(unknown_option_value(
            "gmm_type",
            gmm_type,
            &["one_step", "two_step", "iterated"],
        )),
    }
}

fn parse_weight_kind(weight_type: &str) -> PyResult<WeightKind> {
    match weight_type.to_lowercase().as_str() {
        "classical" => Ok(WeightKind::Classical),
        "robust" => Ok(WeightKind::Robust),
        "cluster" => Ok(WeightKind::Cluster),
        "hac" => Ok(WeightKind::Hac),
        _ => Err(unknown_option_value(
            "gmm_weight_type",
            weight_type,
            &["classical", "robust", "cluster", "hac"],
        )),
    }
}

/// 文字列のオプション（`estimator`・`cov_type`・`gmm_type`・`gmm_weight_type`）を種類に
/// する。DataFrameは見ない。報告順は`estimator` → `cov_type` → `gmm_type` →
/// `gmm_weight_type`。`gmm_type`/`gmm_weight_type`は指定されていれば`estimator`によらず
/// 検査し、指定が無ければ実効既定値（`two_step`/`classical`）にする。
///
/// # Errors
/// いずれかが既知の値でなければ`ValidationError`。
pub(crate) fn parse_iv_option_values(options: &IVOptions) -> PyResult<IvOptionValues> {
    let estimator = parse_estimator(&options.estimator)?;
    let cov = parse_cov_kind(&options.cov_type)?;
    let gmm_type = match options.gmm_type.as_deref() {
        Some(value) => parse_gmm_kind(value)?,
        None => GmmKind::TwoStep,
    };
    let weight = match options.gmm_weight_type.as_deref() {
        Some(value) => parse_weight_kind(value)?,
        None => WeightKind::Classical,
    };
    Ok(IvOptionValues {
        estimator,
        cov,
        gmm_type,
        weight,
    })
}

/// 選んだ`estimator`/`gmm_type`/`cov_type`/`gmm_weight_type`では使われないオプションが
/// 明示指定されていれば`ValidationError`にする（黙って無視すると`cov_type="cluster"`の
/// 書き忘れ等に気づけないため）。そのあと、必須の`hac_time`の欠落を検査する。DataFrameは
/// 見ない。
///
/// - `gmm_type`/`gmm_weight_type`/`gmm_max_iter`/`gmm_tol`/`raise_on_non_convergence`:
///   `estimator="gmm"`のときのみ。さらに`gmm_weight_type`は`gmm_type`が
///   `"two_step"`/`"iterated"`のとき、`gmm_max_iter`/`gmm_tol`/`raise_on_non_convergence`
///   は`"iterated"`のときのみ使われる。
/// - `cluster`/`hac_lags`/`hac_time`: `cov_type`と`gmm_weight_type`の両方から参照される
///   ため、どちらか一方でも使えば有効（`gmm_weight_type`側は重み行列を使うときのみ）。
/// - `hac_time`の必須: `cov_type="hac"`が先、`gmm_weight_type="hac"`が後。
///
/// # Errors
/// 使われないオプションが指定されている場合、または`hac`で`hac_time`が未指定の場合は
/// `ValidationError`。
pub(crate) fn check_iv_options(options: &IVOptions, values: &IvOptionValues) -> PyResult<()> {
    const GMM: &str = "estimator=\"gmm\"";
    const GMM_WEIGHTED: &str = "estimator=\"gmm\" and gmm_type in (\"two_step\", \"iterated\")";
    const GMM_ITERATED: &str = "estimator=\"gmm\" and gmm_type=\"iterated\"";
    // 重み行列は`gmm_type`が`two_step`/`iterated`のときだけ使われる（`GMM_WEIGHTED`）。
    const CLUSTER_CONDITION: &str = "cov_type=\"cluster\" or (estimator=\"gmm\" and gmm_type in (\"two_step\", \"iterated\") and gmm_weight_type=\"cluster\")";
    const HAC_CONDITION: &str = "cov_type=\"hac\" or (estimator=\"gmm\" and gmm_type in (\"two_step\", \"iterated\") and gmm_weight_type=\"hac\")";

    let uses_weight = values.uses_weight();
    let is_iterated = values.is_iterated();
    reject_unused_option("gmm_type", options.gmm_type.is_some(), values.is_gmm(), GMM)?;
    reject_unused_option(
        "gmm_weight_type",
        options.gmm_weight_type.is_some(),
        uses_weight,
        GMM_WEIGHTED,
    )?;
    reject_unused_option(
        "gmm_max_iter",
        options.gmm_max_iter.is_some(),
        is_iterated,
        GMM_ITERATED,
    )?;
    reject_unused_option(
        "gmm_tol",
        options.gmm_tol.is_some(),
        is_iterated,
        GMM_ITERATED,
    )?;
    reject_unused_option(
        "raise_on_non_convergence",
        options.raise_on_non_convergence.is_some(),
        is_iterated,
        GMM_ITERATED,
    )?;

    reject_unused_option(
        "cluster",
        options.cluster.is_some(),
        values.cov == CovKind::Cluster || (uses_weight && values.weight == WeightKind::Cluster),
        CLUSTER_CONDITION,
    )?;
    let hac_used = values.cov == CovKind::Hac || (uses_weight && values.weight == WeightKind::Hac);
    reject_unused_option(
        "hac_lags",
        options.hac_lags.is_some(),
        hac_used,
        HAC_CONDITION,
    )?;
    reject_unused_option(
        "hac_time",
        options.hac_time.is_some(),
        hac_used,
        HAC_CONDITION,
    )?;

    if values.cov == CovKind::Hac {
        require_hac_time(options.hac_time.as_deref(), "cov_type")?;
    }
    if uses_weight && values.weight == WeightKind::Hac {
        require_hac_time(options.hac_time.as_deref(), "gmm_weight_type")?;
    }
    Ok(())
}

/// `gmm_weight_type`に対応する列（`cluster`/`hac_time`）を抽出して
/// `engine::iv::gmm::WeightType`を組み立てる（`estimator="gmm"`のみで使用、`cov_type`側の
/// `linear::common::resolve_cov_type`と対になる。`WeightType`が`CovType`と異なる型のため
/// 独立実装）。
///
/// `cluster`/`hac_lags`/`hac_time`は`cov_type`と共用する（モジュールdocコメント
/// 「GMMのgmm_weight_type」参照、`IVOptions`に別フィールドを増やさない設計）。
///
/// # Errors
/// 列の抽出時に発覚する問題は`column_extraction`の責務で`ValidationError`。`hac`で
/// `hac_time`が未指定の場合も`ValidationError`（`check_iv_options`を通していれば起きない）。
fn resolve_weight_type(
    df: &DataFrame,
    options: &IVOptions,
    kind: WeightKind,
) -> PyResult<WeightType> {
    Ok(match kind {
        WeightKind::Classical => WeightType::Classical,
        WeightKind::Robust => WeightType::Robust,
        WeightKind::Cluster => WeightType::Cluster {
            groups: extract_optional_cluster_groups(
                df,
                options.cluster.as_deref(),
                ColumnSource::new("cluster", "data"),
            )?,
        },
        WeightKind::Hac => {
            let col_name = require_hac_time(options.hac_time.as_deref(), "gmm_weight_type")?;
            WeightType::Hac {
                lags: options.hac_lags,
                time_order: extract_time_order_ranks(
                    df,
                    col_name,
                    ColumnSource::new("hac_time", "data"),
                )?,
            }
        }
    })
}

/// `estimator="gmm"`のときの、`engine::iv::gmm::GmmType`と、`IVResult`に格納する正規形の
/// 名前。
#[derive(Debug)]
pub(crate) struct GmmSetup {
    pub(crate) gmm_type: GmmType,
    pub(crate) gmm_type_name: String,
    /// `"one_step"`は重み行列を使わないため`None`。
    pub(crate) weight_type_name: Option<String>,
}

/// `IVOptions`の`gmm_type`/`gmm_max_iter`/`gmm_tol`と重み行列の列から`GmmType`を組み立てる
/// （`estimator="gmm"`のみで使用）。
///
/// `gmm_type`/`gmm_weight_type`/`gmm_max_iter`/`gmm_tol`の既定値は`None`で、使われる
/// モードのときだけ実効既定値（`"two_step"`/`"classical"`/`100`/`1e-6`）に解決する
/// （`IVOptions`はpyclassで既定値と明示指定を区別できないため）。使われないモードでの
/// 明示指定は[`check_iv_options`]が事前に弾いている。
///
/// # Errors
/// - 重み行列の列の抽出時に発覚する問題: `ValidationError`
/// - `"iterated"`で`gmm_max_iter`が負値: `IvError::InvalidGmmMaxIter`（3未満・上限超過と
///   `gmm_tol`が0以下の検証はengine）
fn resolve_gmm_setup(
    df: &DataFrame,
    options: &IVOptions,
    values: &IvOptionValues,
) -> PyResult<GmmSetup> {
    let gmm_type_name = values.gmm_type.name().to_string();
    let weight_type_name = values.weight.name().to_string();
    match values.gmm_type {
        GmmKind::OneStep => Ok(GmmSetup {
            gmm_type: GmmType::OneStep,
            gmm_type_name,
            weight_type_name: None,
        }),
        GmmKind::TwoStep => Ok(GmmSetup {
            gmm_type: GmmType::TwoStep {
                weight: resolve_weight_type(df, options, values.weight)?,
            },
            gmm_type_name,
            weight_type_name: Some(weight_type_name),
        }),
        GmmKind::Iterated => {
            let weight = resolve_weight_type(df, options, values.weight)?;
            let max_iter = match options.gmm_max_iter {
                Some(v) => usize::try_from(v)
                    .map_err(|_| iv_error_to_pyerr(IvError::InvalidGmmMaxIter { max_iter: v }))?,
                None => DEFAULT_GMM_MAX_ITER,
            };
            let tol = options.gmm_tol.unwrap_or(DEFAULT_GMM_TOL);
            Ok(GmmSetup {
                gmm_type: GmmType::Iterated {
                    weight,
                    max_iter,
                    tol,
                },
                gmm_type_name,
                weight_type_name: Some(weight_type_name),
            })
        }
    }
}

/// [`build_iv_input`]が返す、推定の分岐に必要な情報。
#[derive(Debug)]
pub(crate) struct IvPlan {
    pub(crate) estimator: IvEstimator,
    /// `estimator="gmm"`のときのみ`Some`。
    pub(crate) gmm: Option<GmmSetup>,
}

/// Pythonから渡された `data` / `y` / `x_exog` / `x_endog` / `instruments` / `options` を
/// 検証し、`engine::iv::common::IvInput::from_columns`を呼び出すところまでを行う。
/// `TwoSlsEstimator::fit`/`GmmEstimator::fit`の呼び出し・`IVResult`の構築は`fit`
/// （本ファイル）が行う。
///
/// 戻り値の3つ目は`IVResult.cov_type`に格納する正規形の名前、4つ目の[`IvPlan`]は`fit`が
/// 2SLS/GMMのどちらを呼ぶかの分岐と、GMMの`GmmType`（重み行列の列の抽出済み）・
/// `IVResult.gmm_type`/`gmm_weight_type`に格納する正規形の名前を運ぶ。ここでの妥当性
/// チェックと同じ正規化ロジックを`fit`で再実装しないためである。
///
/// 検査の順序は`docs/guide/validation.md`「Order in which errors are reported」: DataFrameを
/// 見ない検査（オプションの値 → 組合せ）→ ロール構造 → `y`/`x`の列 → オプションの列。
///
/// # Errors
/// - 列の抽出時に発覚する問題（列が存在しない、数値/文字列型にキャストできない、
///   欠損値・NaN・無限大を含む等）は`column_extraction`の責務で`ValidationError`
/// - `y`/`x_exog`/`x_endog`/`instruments`間の重複、各ロール内部の重複、
///   `include_intercept=true`のときの`x_exog`/`x_endog`/`instruments`いずれかと
///   `"const"`列との衝突（`x_exog`だけでなく全ロールが対象）、
///   `x_endog`/`instruments`が空リストの場合（`x_exog`は対象外）は
///   ここ（受け口）の責務で`ValidationError`
/// - `estimator`/`cov_type`/`gmm_type`/`gmm_weight_type`の文字列が不正な場合は
///   `ValidationError`（使われないオプションでも。`parse_iv_option_values`参照）
/// - 使われないオプションの指定、`hac`で`hac_time`が未指定の場合は`ValidationError`
///   （`check_iv_options`参照）
/// - 重み行列の列（`estimator="gmm"`のみ）の抽出時に発覚する問題、`gmm_max_iter`が負値の
///   場合は`ValidationError`（`resolve_gmm_setup`参照。範囲と`gmm_tol`の検証はengine）
/// - それ以外（行数不一致等）は`engine::iv::common::IvError`から`iv_error_to_pyerr`で変換
///   （`IvInput::from_columns`はこの時点では識別可能性を検証しないため、
///   `InsufficientInstruments`はここでは発生しない。`IvInput`の構造体docコメント参照）
pub(crate) fn build_iv_input(
    df: &DataFrame,
    y: String,
    x_exog: Vec<String>,
    x_endog: Vec<String>,
    instruments: Vec<String>,
    options: &IVOptions,
) -> PyResult<(IvInput, EngineCovType, String, IvPlan)> {
    // DataFrameを見ない検査を先に行う。報告順は`docs/guide/validation.md`
    // 「Order in which errors are reported」: オプションの値（`estimator` → `cov_type` →
    // `gmm_type` → `gmm_weight_type`）→ オプションの組合せ → ロール構造 → `y`/`x`の列 →
    // オプション列 → エンジン。
    let values = parse_iv_option_values(options)?;
    check_iv_options(options, &values)?;

    // 完全な多重共線性・意図しない列の重複を早期に、分かりやすいエラーで防ぐ
    // （`validation.rs`に集約、OLS/WLS/Logit/Probitと共通の方針）。`instruments`を
    // リストの末尾に置くのは、`x_exog`/`x_endog`と重複した場合にメッセージの主語を
    // `instruments`側にするため（`validate_no_duplicate_roles`のdocコメント
    // 「呼び出し側の契約」、`docs/spec/iv-spec.md`1.1節参照）。
    validate_no_duplicate_roles(&[
        ("y", RoleValue::Single(&y)),
        ("x_exog", RoleValue::Multi(&x_exog)),
        ("x_endog", RoleValue::Multi(&x_endog)),
        ("instruments", RoleValue::Multi(&instruments)),
    ])?;
    validate_no_duplicate_within_role("x_exog", &x_exog)?;
    validate_no_duplicate_within_role("x_endog", &x_endog)?;
    validate_no_duplicate_within_role("instruments", &instruments)?;
    // `"const"`衝突は`x_exog`だけでなく`x_endog`/`instruments`でも起こりうる。
    // `include_intercept=true`が自動追加する切片列は`x_exog`側の
    // 設計行列にのみ足されるが、`first_stage()`の`param_names`（`x_exog`+
    // `instruments`）・構造方程式本体の`param_names`（`x_exog`+`x_endog`、
    // `IvInput::from_columns`参照）はいずれも`x_exog`の`"const"`と同名の列を
    // 連結してしまうため、衝突源が`x_endog`/`instruments`側でも同じ実害
    // （`OlsResults.params`/`IVResult.params`の`dict(zip(param_names, params))`
    // 構築時の後勝ちによる真の切片係数のサイレントな上書き）が起きる。
    validate_no_const_collision("x_exog", &x_exog, options.include_intercept)?;
    validate_no_const_collision("x_endog", &x_endog, options.include_intercept)?;
    validate_no_const_collision("instruments", &instruments, options.include_intercept)?;

    // `x_exog`は空リストを許容する（内生変数のみのモデルも成立するため、
    // `docs/spec/iv-spec.md`1.1節）が、`x_endog`/`instruments`はいずれも最低1要素を要求する
    // （2026-08-30ユーザー決定）。`x_endog=[]`は実質OLSと等価な退化ケースで
    // あり「そもそもIVを使用すること自体が誤り」と判断し、`OLS`への切り替えなしにそのまま
    // `IV`に渡せる利便性よりも誤用防止を優先した。`x_endog`/`instruments`を独立に検証する
    // ため、`instruments=[]`だが`x_endog`が非空という順序条件違反（`InsufficientInstruments`、
    // `fit`関数参照）とは別に、`x_endog=[]`だが`instruments`が非空という「操作変数はあるが
    // 対応する内生変数が無い」誤用も検出できる。`engine::iv::common::IvInput`自体は
    // このビジネスルールを持たず、引き続き空リストを許容する薄い構造体のまま
    // （識別可能性を含む業務ルールの検証はPython API境界である`engine_pybind`側の責務、
    // `IvInput`の構造体docコメント参照）。
    validate_x_non_empty("x_endog", &x_endog)?;
    validate_x_non_empty("instruments", &instruments)?;

    // ── y列の抽出 ──────────────────────────────────────────────────────
    let y_slice = extract_f64_column(df, &y, ColumnSource::new("y", "data"))?;

    // ── x_exog/x_endog/instruments列の抽出 ─────────────────────────────
    let mut x_exog_columns: Vec<Vec<f64>> = Vec::with_capacity(x_exog.len());
    for col_name in &x_exog {
        x_exog_columns.push(extract_f64_column(
            df,
            col_name,
            ColumnSource::new("x_exog", "data"),
        )?);
    }
    let mut x_endog_columns: Vec<Vec<f64>> = Vec::with_capacity(x_endog.len());
    for col_name in &x_endog {
        x_endog_columns.push(extract_f64_column(
            df,
            col_name,
            ColumnSource::new("x_endog", "data"),
        )?);
    }
    let mut instrument_columns: Vec<Vec<f64>> = Vec::with_capacity(instruments.len());
    for col_name in &instruments {
        instrument_columns.push(extract_f64_column(
            df,
            col_name,
            ColumnSource::new("instruments", "data"),
        )?);
    }

    // ── cov_type固有の追加列の抽出（該当するcov_typeのときのみ）─────────────
    let cov_type = resolve_cov_type(
        df,
        values.cov,
        options.cluster.as_deref(),
        options.hac_lags,
        options.hac_time.as_deref(),
    )?;
    // 重み行列の列（`estimator="gmm"`のみ）も、エンジンの検査より前に抽出する。
    let gmm = if values.is_gmm() {
        Some(resolve_gmm_setup(df, options, &values)?)
    } else {
        None
    };

    let input = IvInput::from_columns(
        &y_slice,
        &x_exog_columns,
        x_exog,
        &x_endog_columns,
        x_endog,
        &instrument_columns,
        instruments,
        options.include_intercept,
        y,
    )
    .map_err(iv_error_to_pyerr)?;

    let plan = IvPlan {
        estimator: values.estimator,
        gmm,
    };
    Ok((input, cov_type, values.cov.name().to_string(), plan))
}

/// Pythonから渡された `data` / `y` / `x_exog` / `x_endog` / `instruments` / `options` を
/// 検証し、`build_iv_input`で構築した`IvInput`に対して`estimator`に応じた推定
/// （`TwoSlsEstimator::fit`または`GmmEstimator::fit`）を呼び出し、`IVResult`として返す。
///
/// `first_stage`/`weak_instrument_f_statistics`は`estimator`によらず`engine::iv::common::
/// compute_first_stage`（`IVResult`のdocコメント・モジュールdocコメント「`first_stage()`/
/// `weak_instrument_f_statistics`は`estimator`に依存しない共通ロジック」参照）から構築する。
/// `overid_statistic`/`overid_p_value`は`estimator="2sls"`では`TwoSlsEstimator::
/// sargan_statistic()`/`sargan_p_value()`（Sargan検定）、`estimator="gmm"`では
/// `GmmEstimator::hansen_j_statistic()`/`hansen_j_p_value()`（Hansen J検定）から構築する。
/// `wu_hausman_statistic`/`wu_hausman_p_value`は`estimator="2sls"`では`TwoSlsEstimator::
/// wu_hausman_statistic()`/`wu_hausman_p_value()`、`estimator="gmm"`では
/// 常に`None`（`GmmEstimator`は実装しない、モジュールdocコメント参照）。
///
/// # Errors
/// - 観測数が構造方程式の係数の数以下、または第一段階の回帰変数の数以下:
///   `ValidationError`（`validate_observation_counts`、`compute_first_stage`より前）
/// - `cov_type="cluster"`でクラスター数が構造方程式の傾き係数の数以下（または2未満）:
///   `ValidationError`（`validate_structural_cluster_count`、`compute_first_stage`より前）
/// - 切片があり`y`が全行で同じ値: `ValidationError`（`validate_outcome_not_constant`、
///   `compute_first_stage`より前）
/// - `build_iv_input`が返すエラー（列抽出・y/x_exog/x_endog/instrumentsの重複・
///   `"const"`列衝突・`estimator`/`cov_type`文字列の検証等）は`ValidationError`
/// - `gmm_type`/`gmm_weight_type`の文字列が不正（使われないオプションでも）、または使われない
///   オプションの指定: `ValidationError`（`build_iv_input`が`parse_iv_option_values`/
///   `check_iv_options`で報告する）
/// - `TwoSlsEstimator::fit`/`GmmEstimator::fit`/`compute_first_stage`が返す
///   `engine::iv::common::IvError`（識別の順序条件・第一段階回帰の失敗・`cov_type`起因の
///   エラー・GMM固有のエラー等）は`iv_error_to_pyerr`で変換
pub(crate) fn fit(
    py: Python<'_>,
    data: SelectedData,
    y: String,
    x_exog: Vec<String>,
    x_endog: Vec<String>,
    instruments: Vec<String>,
    options: &IVOptions,
) -> PyResult<IVResult> {
    // 取り込み済みの`frame`以降はPythonオブジェクトに触れないので、GILを手放して走らせる
    // （他のPythonスレッドを止めず、複数のfitをスレッドで並行できる）。`source`は使わず、
    // 関数の末尾（`detach`から戻ってGILを持った状態）で`data`ごとdropされる。
    let SelectedData { frame, .. } = data;
    py.detach(|| fit_frame(&frame, y, x_exog, x_endog, instruments, options))
}

/// [`fit`]の純Rust部分（GILを手放した状態で走る）。`PyErr`は遅延生成でGILを要求しない
/// ので、ここで`?`により返してよい。
fn fit_frame(
    df: &DataFrame,
    y: String,
    x_exog: Vec<String>,
    x_endog: Vec<String>,
    instruments: Vec<String>,
    options: &IVOptions,
) -> PyResult<IVResult> {
    let (input, cov_type, cov_type_lower, plan) =
        build_iv_input(df, y, x_exog, x_endog, instruments, options)?;

    // 識別の順序条件（`TwoSlsEstimator::fit`/`GmmEstimator::fit`のいずれも冒頭で検証する
    // のと同じチェック）を`compute_first_stage`より先に行う。過小識別な入力で無駄な
    // 第一段階回帰を走らせないため（rust-reviewerの指摘、`compute_first_stage`自体は
    // この条件を検証しないため呼び出し元の責務）。この時点で`build_iv_input`の
    // `validate_x_non_empty("x_endog"/"instruments", ...)`を既に通過して
    // いるため`k_endog`/`k_instruments`はともに1以上であり、ここでの`<`判定は「両方
    // 指定されているが数が足りない」過小識別ケースのみを扱う（「そもそも変数が
    // 指定されていない」退化ケースとは排他的、rust-reviewerの指摘で明記）。
    if input.k_instruments() < input.k_endog() {
        return Err(iv_error_to_pyerr(IvError::InsufficientInstruments {
            n_instruments: input.k_instruments(),
            n_endog: input.k_endog(),
        }));
    }

    // 観測数`n`のチェック（構造方程式の`n <= k`を先に、次に第一段階の`n <= l`）を
    // `compute_first_stage`より先に行う。第一段階の回帰変数の数は識別条件により構造方程式
    // 以上のため、先に第一段階を走らせると構造方程式の`n <= k`が第一段階の
    // `FirstStageFailed`（第一段階の`k`）に隠れて報告されてしまう。
    validate_observation_counts(&input).map_err(iv_error_to_pyerr)?;

    // `cov_type=Cluster`の構造方程式の`G <= q`チェック（`TwoSlsEstimator::fit`/
    // `GmmEstimator::fit`冒頭と同じ実装）を`compute_first_stage`より先に行う。第一段階の
    // 傾き係数の数は識別条件により構造方程式以上のため、先に第一段階を走らせると
    // このチェックが常に`FirstStageFailed`（第一段階の`g`・`q`）に隠れ、エンジン側の
    // 構造方程式チェックがPythonから到達不能になる。
    validate_structural_cluster_count(&input, &cov_type).map_err(iv_error_to_pyerr)?;

    // 被説明変数が定数（切片あり）のモデルは説明するものが無い。入力だけから分かるため、これも
    // `compute_first_stage`より先に弾く（`TwoSlsEstimator::fit`/`GmmEstimator::fit`冒頭と同じ実装）。
    validate_outcome_not_constant(&input).map_err(|e| iv_error_to_pyerr(IvError::from(e)))?;

    // 第一段階回帰・弱操作変数診断は`estimator`によらず共通（モジュールdocコメント参照）。
    // `input`は下でestimatorごとの推定器に移動するため、参照のみで済むこの呼び出しを先に行う。
    let (first_stage, weak_instrument_f_statistics) =
        compute_first_stage(&input, &cov_type, options.confidence_level)
            .map_err(iv_error_to_pyerr)?;
    let weak_instrument_f_df = first_stage.first().map(|(_, first_stage_estimator)| {
        (input.k_instruments(), first_stage_estimator.df_resid())
    });
    // 第一段階の推定量（設計行列を持つ）は結果に変えて手放す。`IVResult`が持つのは結果だけで、
    // `first_stage()`の戻り値にも、pickleの状態にもそのまま使える。
    let first_stage: Vec<(String, OLSResult)> = first_stage
        .into_iter()
        .map(|(name, estimator)| {
            let result = ols_estimator_to_result(&estimator, cov_type_lower.clone());
            (name, result)
        })
        .collect();

    if let Some(gmm) = plan.gmm {
        let estimator = GmmEstimator::fit(
            input,
            gmm.gmm_type,
            options.raise_on_non_convergence.unwrap_or(true),
            cov_type,
            options.confidence_level,
        )
        .map_err(iv_error_to_pyerr)?;

        return Ok(IVResult {
            params: mat_to_vec(estimator.params()),
            std_errors: mat_to_vec(estimator.std_errors()),
            test_stats: mat_to_vec(estimator.test_stats()),
            stat_dist: estimator.stat_dist().name().to_string(),
            stat_df: estimator.stat_dist().df().map(|df| df as i64),
            p_values: mat_to_vec(estimator.p_values()),
            conf_lower: mat_to_vec(estimator.conf_lower()),
            conf_upper: mat_to_vec(estimator.conf_upper()),
            param_names: estimator.param_names().to_vec(),
            residuals: mat_to_vec(estimator.residuals()),
            dep_var_name: estimator.dep_var_name().to_string(),
            n_obs: estimator.nobs(),
            df_resid: estimator.df_resid(),
            df_model: estimator.df_model(),
            converged: estimator.converged(),
            n_iter: estimator.n_iter(),
            cov_type: cov_type_lower,
            hac_lags_used: estimator.hac_lags_used().map(|lags| lags as i64),
            estimator: plan.estimator.name().to_string(),
            gmm_weight_type: gmm.weight_type_name,
            gmm_type: Some(gmm.gmm_type_name),
            wald_statistic: estimator.wald_statistic(),
            wald_p_value: estimator.wald_p_value(),
            wald_dist: "chi2".to_string(),
            wald_df_num: estimator.wald_df(),
            wald_df_denom: None,
            r_squared: estimator.r_squared(),
            adj_r_squared: estimator.adj_r_squared(),
            weak_instrument_f_statistics: weak_instrument_f_statistics.into_iter().collect(),
            overid_statistic: estimator.hansen_j_statistic(),
            overid_p_value: estimator.hansen_j_p_value(),
            overid_df: estimator.hansen_j_df(),
            wu_hausman_statistic: None,
            wu_hausman_p_value: None,
            wu_hausman_df_num: None,
            wu_hausman_df_denom: None,
            weak_instrument_f_df_num: weak_instrument_f_df.map(|(num, _)| num),
            weak_instrument_f_df_denom: weak_instrument_f_df.map(|(_, denom)| denom),
            first_stage,
        });
    }

    let estimator = TwoSlsEstimator::fit(input, cov_type, options.confidence_level)
        .map_err(iv_error_to_pyerr)?;

    Ok(IVResult {
        params: mat_to_vec(estimator.params()),
        std_errors: mat_to_vec(estimator.std_errors()),
        test_stats: mat_to_vec(estimator.test_stats()),
        stat_dist: estimator.stat_dist().name().to_string(),
        stat_df: estimator.stat_dist().df().map(|df| df as i64),
        p_values: mat_to_vec(estimator.p_values()),
        conf_lower: mat_to_vec(estimator.conf_lower()),
        conf_upper: mat_to_vec(estimator.conf_upper()),
        param_names: estimator.param_names().to_vec(),
        residuals: mat_to_vec(estimator.residuals()),
        dep_var_name: estimator.dep_var_name().to_string(),
        n_obs: estimator.nobs(),
        df_resid: estimator.df_resid(),
        df_model: estimator.df_model(),
        // 2SLSは閉形式・非反復のため常に`converged=true`・`n_iter=1`
        // （`IVResult.converged`のdocコメント参照）。
        converged: true,
        n_iter: 1,
        cov_type: cov_type_lower,
        hac_lags_used: estimator.hac_lags_used().map(|lags| lags as i64),
        estimator: plan.estimator.name().to_string(),
        // `gmm_weight_type`はGMM専用の概念のため`estimator="2sls"`では常に`None`
        // （`IVResult.gmm_weight_type`のdocコメント参照）。
        gmm_weight_type: None,
        gmm_type: None,
        wald_statistic: estimator.wald_statistic(),
        wald_p_value: estimator.wald_p_value(),
        wald_dist: "f".to_string(),
        wald_df_num: estimator.wald_df().map(|(num, _)| num),
        wald_df_denom: estimator.wald_df().map(|(_, denom)| denom),
        r_squared: estimator.r_squared(),
        adj_r_squared: estimator.adj_r_squared(),
        weak_instrument_f_statistics: weak_instrument_f_statistics.into_iter().collect(),
        overid_statistic: estimator.sargan_statistic(),
        overid_p_value: estimator.sargan_p_value(),
        overid_df: estimator.sargan_df(),
        wu_hausman_statistic: estimator.wu_hausman_statistic(),
        wu_hausman_p_value: estimator.wu_hausman_p_value(),
        wu_hausman_df_num: estimator.wu_hausman_df().map(|(num, _)| num),
        wu_hausman_df_denom: estimator.wu_hausman_df().map(|(_, denom)| denom),
        weak_instrument_f_df_num: weak_instrument_f_df.map(|(num, _)| num),
        weak_instrument_f_df_denom: weak_instrument_f_df.map(|(_, denom)| denom),
        first_stage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::shared::group_codes::GroupCodes;
    use polars::df;

    /// `build_iv_input`のテスト全体で使う既定の`IVOptions`（`estimator="2sls"`・
    /// `cov_type="classical"`・`include_intercept=true`）。フィールドごとに上書きして使う。
    fn default_options() -> IVOptions {
        IVOptions::new(
            "2sls".to_string(),
            "classical".to_string(),
            true,
            0.95,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    fn well_formed_df() -> DataFrame {
        df!(
            "y" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x1" => [2.0, 4.0, 1.0, 5.0, 3.0, 6.0],
            "endog1" => [5.0, 4.0, 3.0, 6.0, 2.0, 1.0],
            "z1" => [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            "z2" => [1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
        )
        .unwrap()
    }

    /// オプションの検査（値 → 組合せ）の判定表（`Err`になる/ならない）。エラー文言そのものは
    /// `PyErr`のDisplayがGILを要求するためpytest側（`test_iv_validation.py`）で確認する。
    fn options_are_ok(configure: impl FnOnce(&mut IVOptions), estimator: &str) -> bool {
        let mut options = default_options();
        options.estimator = estimator.to_string();
        configure(&mut options);
        let Ok(values) = parse_iv_option_values(&options) else {
            return false;
        };
        check_iv_options(&options, &values).is_ok()
    }

    #[test]
    fn check_iv_options_rejects_gmm_options_for_2sls() {
        assert!(!options_are_ok(
            |o| o.gmm_type = Some("two_step".into()),
            "2sls"
        ));
        assert!(!options_are_ok(|o| o.gmm_max_iter = Some(10), "2sls"));
        assert!(options_are_ok(|_| {}, "2sls"));
    }

    #[test]
    fn check_iv_options_ties_options_to_gmm_type() {
        let one_step = |o: &mut IVOptions| o.gmm_type = Some("one_step".into());
        assert!(!options_are_ok(
            |o| {
                one_step(o);
                o.gmm_weight_type = Some("robust".into());
            },
            "gmm"
        ));
        assert!(!options_are_ok(
            |o| o.gmm_max_iter = Some(10),
            "gmm" // 既定のgmm_type（two_step）は反復しない
        ));
        assert!(options_are_ok(
            |o| {
                o.gmm_type = Some("iterated".into());
                o.gmm_weight_type = Some("robust".into());
                o.gmm_max_iter = Some(10);
                o.gmm_tol = Some(1e-6);
                o.raise_on_non_convergence = Some(false);
            },
            "gmm"
        ));
    }

    #[test]
    fn check_iv_options_shares_cluster_and_hac_options_between_cov_type_and_weight() {
        // gmm_weight_typeだけがcluster/hac_*を使う（どちらか一方でも使えば有効）
        assert!(options_are_ok(
            |o| {
                o.gmm_weight_type = Some("cluster".into());
                o.cluster = Some("g".into());
            },
            "gmm"
        ));
        assert!(options_are_ok(
            |o| {
                o.gmm_weight_type = Some("hac".into());
                o.hac_lags = Some(2);
                o.hac_time = Some("t".into());
            },
            "gmm"
        ));
        // cov_typeだけが使う
        assert!(options_are_ok(
            |o| {
                o.cov_type = "HAC".into();
                o.hac_lags = Some(2);
                o.hac_time = Some("t".into());
            },
            "2sls"
        ));
        // どちらも使わない（weightがhacなのにclusterを指定、2slsでweightは使えない）
        assert!(!options_are_ok(
            |o| {
                o.gmm_weight_type = Some("hac".into());
                o.cluster = Some("g".into());
            },
            "gmm"
        ));
        assert!(!options_are_ok(|o| o.cluster = Some("g".into()), "2sls"));
    }

    #[test]
    fn check_iv_options_requires_hac_time_for_hac_cov_type_and_hac_weight() {
        assert!(!options_are_ok(|o| o.cov_type = "hac".into(), "2sls"));
        assert!(!options_are_ok(
            |o| o.gmm_weight_type = Some("hac".into()),
            "gmm"
        ));
    }

    #[test]
    fn parse_iv_option_values_reports_unknown_values_even_when_the_option_is_unused() {
        // 使われないオプションの未知の値も、使われない指摘より先に値の誤りとして弾く。
        assert!(!options_are_ok(
            |o| o.gmm_type = Some("bogus".into()),
            "2sls"
        ));
        assert!(!options_are_ok(
            |o| o.gmm_weight_type = Some("bogus".into()),
            "2sls"
        ));
        assert!(!options_are_ok(
            |o| {
                o.gmm_type = Some("bogus".into());
                o.gmm_max_iter = Some(10);
            },
            "gmm"
        ));
        assert!(!options_are_ok(
            |o| {
                o.cov_type = "bogus".into();
                o.cluster = Some("g".into());
            },
            "2sls"
        ));
    }

    #[test]
    fn parse_iv_option_values_has_canonical_names_and_effective_defaults() {
        let mut options = default_options();
        options.estimator = "GMM".to_string();
        options.cov_type = "HC1".to_string();
        let Ok(values) = parse_iv_option_values(&options) else {
            panic!("expected Ok");
        };
        assert_eq!(values.estimator.name(), "gmm");
        assert_eq!(values.cov.name(), "hc1");
        assert_eq!(values.gmm_type.name(), "two_step");
        assert_eq!(values.weight.name(), "classical");
    }

    #[test]
    fn parse_iv_option_values_is_case_insensitive_for_every_kind() {
        for (estimator, expected) in [("2sls", "2sls"), ("2SLS", "2sls"), ("Gmm", "gmm")] {
            let Ok(kind) = parse_estimator(estimator) else {
                panic!("expected Ok for estimator={estimator}");
            };
            assert_eq!(kind.name(), expected);
        }
        for (input, expected) in [
            ("one_step", "one_step"),
            ("ONE_STEP", "one_step"),
            ("Two_Step", "two_step"),
            ("Iterated", "iterated"),
        ] {
            let Ok(kind) = parse_gmm_kind(input) else {
                panic!("expected Ok for gmm_type={input}");
            };
            assert_eq!(kind.name(), expected);
        }
        for (input, expected) in [
            ("classical", "classical"),
            ("ROBUST", "robust"),
            ("Cluster", "cluster"),
            ("HAC", "hac"),
        ] {
            let Ok(kind) = parse_weight_kind(input) else {
                panic!("expected Ok for gmm_weight_type={input}");
            };
            assert_eq!(kind.name(), expected);
        }
    }

    #[test]
    fn parse_iv_option_values_rejects_aliases() {
        for alias in ["3sls", "tsls", "iv"] {
            assert!(parse_estimator(alias).is_err(), "{alias}");
        }
        for alias in ["onestep", "twostep", "two-step", "continuous"] {
            assert!(parse_gmm_kind(alias).is_err(), "{alias}");
        }
        // 旧名（`unadjusted`/`kernel`）と、`cov_type`の語彙に無い別名は受け付けない。
        for alias in [
            "unadjusted",
            "kernel",
            "white",
            "homoskedastic",
            "heteroskedastic",
        ] {
            assert!(parse_weight_kind(alias).is_err(), "{alias}");
        }
    }

    #[test]
    fn check_iv_options_covers_the_remaining_gmm_and_cov_combinations() {
        // `gmm_tol`/`raise_on_non_convergence`は`iterated`のときだけ使われる。
        assert!(!options_are_ok(|o| o.gmm_tol = Some(1e-3), "gmm"));
        assert!(!options_are_ok(
            |o| o.raise_on_non_convergence = Some(false),
            "gmm"
        ));
        assert!(options_are_ok(
            |o| {
                o.gmm_type = Some("iterated".into());
                // `gmm_weight_type`未指定は実効既定値（`classical`）になる。
                o.gmm_tol = Some(1e-3);
            },
            "gmm"
        ));
        // `cov_type`が`classical`/`cluster`のとき`hac_*`は使われない。
        assert!(!options_are_ok(|o| o.hac_lags = Some(2), "2sls"));
        assert!(!options_are_ok(|o| o.hac_time = Some("t".into()), "2sls"));
        assert!(!options_are_ok(
            |o| {
                o.cov_type = "cluster".into();
                o.cluster = Some("g".into());
                o.hac_lags = Some(2);
            },
            "2sls"
        ));
        // `two_step`で`gmm_weight_type="cluster"`と`cluster`の組は使われる。
        assert!(options_are_ok(
            |o| {
                o.gmm_weight_type = Some("cluster".into());
                o.cluster = Some("g".into());
            },
            "gmm"
        ));
        // `cov_type="cluster"`と`gmm_weight_type="hac"`が同時に使われるとき、`cluster`は
        // `cov_type`側で、`hac_*`は`gmm_weight_type`側で使われる。
        assert!(options_are_ok(
            |o| {
                o.cov_type = "cluster".into();
                o.cluster = Some("g".into());
                o.gmm_weight_type = Some("hac".into());
                o.hac_time = Some("t".into());
                o.hac_lags = Some(1);
            },
            "gmm"
        ));
    }

    #[test]
    fn build_iv_input_succeeds_for_well_formed_data() {
        let df = well_formed_df();
        let options = default_options();

        let (input, cov_type, cov_type_lower, plan) = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        )
        .unwrap();

        assert_eq!(input.nobs(), 6);
        assert_eq!(input.k_exog(), 2); // const + x1
        assert_eq!(input.k_endog(), 1);
        assert_eq!(input.k_instruments(), 2);
        assert_eq!(cov_type, EngineCovType::Classical);
        assert_eq!(cov_type_lower, "classical");
        assert_eq!(plan.estimator.name(), "2sls");
        assert!(plan.gmm.is_none());
    }

    #[test]
    fn build_iv_input_allows_empty_x_exog() {
        let df = well_formed_df();
        let options = default_options();

        let (input, ..) = build_iv_input(
            &df,
            "y".to_string(),
            vec![],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        )
        .unwrap();

        assert_eq!(input.k_exog(), 1); // const only
    }

    #[test]
    fn build_iv_input_returns_error_when_x_endog_and_instruments_are_both_empty() {
        // `x_endog=[]`かつ`instruments=[]`（実質OLSと等価な退化ケース）を
        // 誤用として`ValidationError`で弾く（旧仕様では成功していた、
        // `docs/spec/iv-spec.md`1.1節）。
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec![],
            vec![],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_x_endog_is_empty_but_instruments_is_not() {
        // `x_endog`/`instruments`は独立に最低1要素を要求するため、
        // 対応する内生変数の無い操作変数だけを指定する誤用も検出する。
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec![],
            vec!["z1".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_instruments_is_empty_but_x_endog_is_not() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec![],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_for_unknown_method() {
        let df = well_formed_df();
        let mut options = default_options();
        options.estimator = "3sls".to_string();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_succeeds_for_gmm_method() {
        let df = well_formed_df();
        let mut options = default_options();
        options.estimator = "GMM".to_string(); // 大文字小文字を区別しないことも確認

        let (_, _, _, plan) = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        )
        .unwrap();
        assert_eq!(plan.estimator.name(), "gmm");
        let Some(gmm) = plan.gmm else {
            panic!("expected a GMM setup");
        };
        assert_eq!(gmm.gmm_type_name, "two_step");
        assert_eq!(gmm.weight_type_name.as_deref(), Some("classical"));
    }

    #[test]
    fn build_iv_input_returns_error_when_y_overlaps_x_exog() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["y".to_string(), "x1".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_y_overlaps_x_endog() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["y".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_y_overlaps_instruments() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["y".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_x_exog_overlaps_x_endog() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string(), "endog1".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_instruments_overlaps_x_exog() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["x1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_x_endog_overlaps_instruments() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["z1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_instruments_contains_duplicate() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z1".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_include_intercept_and_x_exog_contains_const() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string(), "const".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_include_intercept_and_x_endog_contains_const() {
        // `x_exog`だけでなく`x_endog`に`"const"`を含めた場合も
        // 同じ`ValidationError`で弾く（構造方程式本体の`param_names`から
        // 真の切片係数がサイレントに失われるケース、列抽出前にここで検出する
        // ため`df`に実際の`"const"`列は不要）。
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["const".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_include_intercept_and_instruments_contains_const() {
        // `instruments`に`"const"`を含めた場合も同じ`ValidationError`
        // で弾く（`first_stage()`の`param_names`が`x_exog`の`"const"`（真の切片）と
        // 衝突し後勝ちでサイレントに上書きされるケース）。
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["const".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_for_unknown_cov_type() {
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "unknown".to_string();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_extracts_cluster_groups_when_cov_type_is_cluster() {
        let df = df!(
            "y" => [1.0, 2.0, 3.0, 4.0],
            "endog1" => [4.0, 3.0, 2.0, 1.0],
            "z1" => [2.0, 1.0, 4.0, 3.0],
            "group" => ["a", "a", "b", "b"],
        )
        .unwrap();
        let mut options = default_options();
        options.cov_type = "cluster".to_string();
        options.cluster = Some("group".to_string());

        let (_, cov_type, ..) = build_iv_input(
            &df,
            "y".to_string(),
            vec![],
            vec!["endog1".to_string()],
            vec!["z1".to_string()],
            &options,
        )
        .unwrap();

        assert_eq!(
            cov_type,
            EngineCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&[
                    "a".to_string(),
                    "a".to_string(),
                    "b".to_string(),
                    "b".to_string()
                ]))
            }
        );
    }

    #[test]
    fn resolve_weight_type_passes_time_order_ranks_for_hac_weight() {
        let df = df!("t" => [30.0, 10.0, 40.0, 20.0]).unwrap();
        let mut options = default_options();
        options.estimator = "gmm".to_string();
        options.gmm_weight_type = Some("hac".to_string());
        options.hac_time = Some("t".to_string());
        options.hac_lags = Some(1);

        let Ok(weight_type) = resolve_weight_type(&df, &options, WeightKind::Hac) else {
            panic!("expected Ok");
        };

        assert_eq!(
            weight_type,
            WeightType::Hac {
                lags: Some(1),
                time_order: vec![2.0, 0.0, 3.0, 1.0],
            }
        );
    }

    #[test]
    fn resolve_weight_type_requires_hac_time_for_hac_weight() {
        let df = df!("t" => [1.0, 2.0, 3.0]).unwrap();
        let options = default_options();
        assert!(resolve_weight_type(&df, &options, WeightKind::Hac).is_err());
    }

    #[test]
    fn resolve_weight_type_rejects_tied_hac_time() {
        let df = df!("t" => [1.0, 2.0, 1.0]).unwrap();
        let mut options = default_options();
        options.hac_time = Some("t".to_string());

        assert!(resolve_weight_type(&df, &options, WeightKind::Hac).is_err());
    }

    #[test]
    fn build_iv_input_resolves_gmm_weight_columns_before_the_engine() {
        // 重み行列の列の存在は、`build_iv_input`（エンジンより前）で検査する。
        let df = well_formed_df();
        let mut options = default_options();
        options.estimator = "gmm".to_string();
        options.gmm_weight_type = Some("cluster".to_string());
        options.cluster = Some("nope".to_string());

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_passes_time_order_ranks_when_cov_type_is_hac() {
        let df = df!(
            "y" => [1.0, 2.0, 3.0, 4.0],
            "endog1" => [4.0, 3.0, 2.0, 1.0],
            "z1" => [2.0, 1.0, 4.0, 3.0],
            "t" => [30.0, 10.0, 40.0, 20.0],
        )
        .unwrap();
        let mut options = default_options();
        options.cov_type = "hac".to_string();
        options.hac_time = Some("t".to_string());
        options.hac_lags = Some(1);

        let (_, cov_type, ..) = build_iv_input(
            &df,
            "y".to_string(),
            vec![],
            vec!["endog1".to_string()],
            vec!["z1".to_string()],
            &options,
        )
        .unwrap();

        assert_eq!(
            cov_type,
            EngineCovType::Hac {
                lags: Some(1),
                // 値ではなく昇順の位置（順位）をエンジンに渡す。
                time_order: vec![2.0, 0.0, 3.0, 1.0],
            }
        );
    }
}
