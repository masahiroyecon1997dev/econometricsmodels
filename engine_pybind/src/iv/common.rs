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
//! `IvError`の`Common`バリアント（`engine::error::CommonError`）は`crate::errors::
//! common_error_to_pyerr`に委譲する（系統ごとに同じ判定ロジックを重複させない）。
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
//! `first_stage()`を実装していたが、GMM配線にあたり`first_stage: Vec<(String,
//! OlsEstimator)>`という`estimator`非依存の表現に置き換えた（`OlsEstimator → OLSResult`
//! 変換は`linear::ols::ols_estimator_to_result`を再利用、抽出済み）。
//!
//! ## GMMの`gmm_weight_type`（`IVOptions.gmm_weight_type`/`cluster`/`hac_lags`/`hac_time`）
//!
//! `gmm_weight_type`は`cov_type`とは独立の軸（点推定に使う重み行列の選択、`engine::iv::gmm`の
//! モジュールdocコメント参照）だが、`cluster`/`hac_lags`/`hac_time`は`cov_type`と
//! 共用する（`IVOptions`に別フィールドを増やさない設計、`parse_weight_type`参照）。
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

use engine::iv::common::{IvError, IvInput, compute_first_stage};
use engine::iv::gmm::{GmmEstimator, GmmType, WeightType};
use engine::iv::two_sls::TwoSlsEstimator;
use engine::linear::ols::CovType as EngineCovType;
use engine::linear::ols::OlsEstimator;
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::column_extraction::{
    extract_f64_column, extract_group_key_column, extract_ordering_f64_column,
};
use crate::errors::{ComputationError, ValidationError, common_error_to_pyerr};
use crate::linear::common::{build_cov_type, least_squares_error_is_computation_error, mat_to_vec};
use crate::linear::ols::{OLSResult, ols_estimator_to_result};
use crate::option_values::{
    extract_strict_float, extract_strict_opt_column, extract_strict_opt_float,
    extract_strict_opt_int, extract_strict_opt_text, extract_strict_text,
};
use crate::validation::{
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
/// `FirstStageFailed`/`SecondStageFailed`は2SLS（`engine::iv::two_sls`）が内部で委譲する
/// `OlsEstimator::fit`の失敗を包んだもの。`HausmanRegressionFailed`
/// も同型だが、Wu-Hausman検定の拡張回帰が理論上到達不能な理由で失敗した
/// 場合のみ構築される防御的なバリアント（想定内の失敗——設計行列の特異性・観測数不足等
/// ——は`wu_hausman_statistic`が`None`になるだけで`IvError`自体は発生しない、
/// `engine/src/iv/CLAUDE.md`参照）。`ValidationError`/`ComputationError`の
/// 判定は`least_squares_error_is_computation_error`（`engine_pybind/src/linear/common.rs`）に
/// 委譲し、`least_squares_error_to_pyerr`と同じ基準を保つ（分類ロジックを重複させない）。
/// Pythonに渡すメッセージは`source.to_string()`ではなく`IvError`自身の`to_string()`
/// （「第一段階/第二段階のどの内生変数で失敗したか」という文脈を含む）を使うため、
/// `least_squares_error_to_pyerr`自体はそのまま呼ばない。
///
/// 現在は`fit`（本ファイル）が実際に`#[pymodule]`経路（`fit_iv`）から呼び出すように
/// なっている。当初は`#[cfg(test)] mod tests`からしか呼ばれておらず
/// `#[allow(dead_code)]`が必要だった（`--all-targets`ビルドでの`#[expect]`の罠、
/// `engine_pybind/src/iv/CLAUDE.md`参照）が、本番経路から呼ばれるようになった今は不要。
pub(crate) fn iv_error_to_pyerr(err: IvError) -> PyErr {
    let message = err.to_string();
    match err {
        IvError::Common(common) => common_error_to_pyerr(common),
        IvError::InsufficientInstruments { .. }
        | IvError::InvalidHacLags { .. }
        | IvError::InvalidGmmMaxIter { .. }
        | IvError::InvalidGmmTol { .. }
        | IvError::InsufficientClustersForWeightMatrix { .. } => ValidationError::new_err(message),
        // `MleError::NonConvergence`（`nonlinear/common.rs`の`mle_error_to_pyerr`）と同じ
        // 分類: パラメータの不正ではなく、計算過程（反復推定）で発覚した問題のため
        // `ComputationError`（`engine/src/iv/CLAUDE.md`参照）。
        IvError::GmmNonConvergence { .. } => ComputationError::new_err(message),
        IvError::FirstStageFailed { source, .. }
        | IvError::SecondStageFailed { source }
        | IvError::HausmanRegressionFailed { source } => {
            if least_squares_error_is_computation_error(&source) {
                ComputationError::new_err(message)
            } else {
                ValidationError::new_err(message)
            }
        }
    }
}

/// Estimation options for IV (2SLS/GMM).
///
/// See `docs/spec/iv-spec.md` for the rationale behind each field's
/// meaning and default value. A single `IVOptions`/`fit_iv` pair serves both
/// estimation methods; fields that apply to only one estimator are documented as such.
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
    /// `estimator="gmm"`, by `gmm_weight_type="hac"`. Specifying it when neither uses it
    /// raises `ValidationError`.
    #[pyo3(get)]
    pub hac_time: Option<String>,

    /// Weight matrix used for GMM point estimation: one of "classical" (homoskedastic),
    /// "robust" (heteroskedasticity-robust), "cluster", "hac" (Newey-West). Same vocabulary
    /// as `cov_type`. Case-insensitive. `None` (default) means "classical" for
    /// `gmm_type="two_step"`/`"iterated"`. Specifying it with `estimator="2sls"` or
    /// `gmm_type="one_step"` (which does not use a weight matrix) raises `ValidationError`.
    /// "cluster"/"hac" draw from the same `cluster`/`hac_lags`/`hac_time` fields as
    /// `cov_type` (no separate fields; see module docstring "GMMのgmm_weight_type").
    #[pyo3(get)]
    pub gmm_weight_type: Option<String>,

    /// GMM estimation type: "one_step" (weight matrix `(Z'Z)^-1` only), "two_step"
    /// (efficient two-step GMM), or "iterated" (repeat until convergence).
    /// Case-insensitive. `None` (default) means "two_step" for `estimator="gmm"`.
    /// Specifying it with `estimator="2sls"` raises `ValidationError`.
    #[pyo3(get)]
    pub gmm_type: Option<String>,

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

#[pymethods]
impl IVOptions {
    #[new]
    #[pyo3(signature = (
        estimator = "2sls".to_string(),
        cov_type = "classical".to_string(),
        include_intercept = true,
        confidence_level = 0.95,
        cluster = None,
        hac_lags = None,
        hac_time = None,
        gmm_weight_type = None,
        gmm_type = None,
        gmm_max_iter = None,
        gmm_tol = None,
        raise_on_non_convergence = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = crate::option_values::estimator_arg)] estimator: String,
        #[pyo3(from_py_with = crate::option_values::cov_type_arg)] cov_type: String,
        include_intercept: bool,
        #[pyo3(from_py_with = crate::option_values::confidence_level_arg)] confidence_level: f64,
        #[pyo3(from_py_with = crate::option_values::cluster_arg)] cluster: Option<String>,
        #[pyo3(from_py_with = crate::option_values::hac_lags_arg)] hac_lags: Option<i64>,
        #[pyo3(from_py_with = crate::option_values::hac_time_arg)] hac_time: Option<String>,
        #[pyo3(from_py_with = crate::option_values::gmm_weight_type_arg)] gmm_weight_type: Option<
            String,
        >,
        #[pyo3(from_py_with = crate::option_values::gmm_type_arg)] gmm_type: Option<String>,
        #[pyo3(from_py_with = crate::option_values::gmm_max_iter_arg)] gmm_max_iter: Option<i64>,
        #[pyo3(from_py_with = crate::option_values::gmm_tol_arg)] gmm_tol: Option<f64>,
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
            gmm_weight_type,
            gmm_type,
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
            "IVOptions(estimator={:?}, cov_type={:?}, include_intercept={}, \
             confidence_level={}, cluster={:?}, hac_lags={:?}, hac_time={:?}, \
             gmm_weight_type={:?}, gmm_type={:?}, gmm_max_iter={:?}, gmm_tol={:?}, \
             raise_on_non_convergence={:?})",
            self.estimator,
            self.cov_type,
            self.include_intercept,
            self.confidence_level,
            self.cluster,
            self.hac_lags,
            self.hac_time,
            self.gmm_weight_type,
            self.gmm_type,
            self.gmm_max_iter,
            self.gmm_tol,
            self.raise_on_non_convergence,
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
/// Vec<(String, OlsEstimator)>`から`first_stage()`をオンデマンドに構築する（下記
/// `first_stage`フィールド参照。当初は`estimator: TwoSlsEstimator`という2sls専用の
/// フィールドだったが、GMM配線時にestimator非依存の表現へ置き換えた——`engine::iv::common::
/// compute_first_stage`が`estimator`によらず同じ第一段階回帰を計算するため、`fit`の
/// 呼び出し元でこの表現に詰め替えるだけで済む）。
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
// `Clone`を派生しない: `first_stage`の要素`OlsEstimator`が`Clone`を実装していないため
// （`LogitResult`/`ProbitResult`と同じ理由、`.claude/rules/rust-style.md`「推定量構造体の
// 設計」の通りprivateフィールドのみで、Cloneを要求する既存の呼び出し元も無い）。
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
    first_stage: Vec<(String, OlsEstimator)>,
}

#[pymethods]
impl IVResult {
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
    fn first_stage(&self) -> HashMap<String, OLSResult> {
        self.first_stage
            .iter()
            .map(|(name, estimator)| {
                (
                    name.clone(),
                    ols_estimator_to_result(estimator, self.cov_type.clone()),
                )
            })
            .collect()
    }
}

/// `gmm_type`/`gmm_weight_type`の実効既定値（`IVOptions`の既定値は`None`のため、使われる
/// モードでここに解決する）。
const DEFAULT_GMM_TYPE: &str = "two_step";
const DEFAULT_GMM_WEIGHT_TYPE: &str = "classical";

/// 選んだ`estimator`/`gmm_type`/`cov_type`/`gmm_weight_type`では使われないオプションが
/// 明示指定されていれば`ValidationError`にする（黙って無視すると`cov_type="cluster"`の
/// 書き忘れ等に気づけないため）。
///
/// - `gmm_type`/`gmm_weight_type`/`gmm_max_iter`/`gmm_tol`/`raise_on_non_convergence`:
///   `estimator="gmm"`のときのみ。さらに`gmm_weight_type`は`gmm_type`が
///   `"two_step"`/`"iterated"`のとき、`gmm_max_iter`/`gmm_tol`/`raise_on_non_convergence`
///   は`"iterated"`のときのみ使われる。
/// - `cluster`/`hac_lags`/`hac_time`: `cov_type`と`gmm_weight_type`の両方から参照される
///   ため、どちらか一方でも使えば有効（`gmm_weight_type`側は`estimator="gmm"`かつ
///   `gmm_type`が`"two_step"`/`"iterated"`のときのみ）。
///
/// `gmm_type`/`gmm_weight_type`の文字列が未知の値の場合はここでは何も言わず、後段の
/// `parse_gmm_type`/`parse_weight_type`の「unknown ...」エラーに委ねる（誤った値と
/// 使われないオプションの二重指摘で本筋のエラーが埋もれないようにするため）。
fn validate_iv_option_usage(options: &IVOptions, estimator_lower: &str) -> PyResult<()> {
    // 未知の`cov_type`は後段の「unknown cov_type」を優先して報告する。
    let cov_type_lower = options.cov_type.to_lowercase();
    if !matches!(
        cov_type_lower.as_str(),
        "classical" | "hc0" | "hc1" | "hc2" | "hc3" | "hac" | "cluster"
    ) {
        return Ok(());
    }
    let is_gmm = estimator_lower == "gmm";
    let gmm_type_lower = options
        .gmm_type
        .as_deref()
        .unwrap_or(DEFAULT_GMM_TYPE)
        .to_lowercase();
    let weight_lower = options
        .gmm_weight_type
        .as_deref()
        .unwrap_or(DEFAULT_GMM_WEIGHT_TYPE)
        .to_lowercase();
    if is_gmm
        && (!matches!(
            gmm_type_lower.as_str(),
            "one_step" | "two_step" | "iterated"
        ) || (gmm_type_lower != "one_step"
            && !matches!(
                weight_lower.as_str(),
                "classical" | "robust" | "cluster" | "hac"
            )))
    {
        return Ok(());
    }

    let uses_weight = is_gmm && gmm_type_lower != "one_step";
    let is_iterated = is_gmm && gmm_type_lower == "iterated";
    const GMM: &str = "estimator=\"gmm\"";
    const GMM_WEIGHTED: &str = "estimator=\"gmm\" with gmm_type=\"two_step\" or \"iterated\"";
    const GMM_ITERATED: &str = "estimator=\"gmm\" with gmm_type=\"iterated\"";
    reject_unused_option("gmm_type", options.gmm_type.is_some(), is_gmm, GMM)?;
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
        cov_type_lower == "cluster" || (uses_weight && weight_lower == "cluster"),
        "cov_type=\"cluster\" (or gmm_weight_type=\"cluster\" with estimator=\"gmm\")",
    )?;
    let hac_used = cov_type_lower == "hac" || (uses_weight && weight_lower == "hac");
    const HAC_CONDITION: &str =
        "cov_type=\"hac\" (or gmm_weight_type=\"hac\" with estimator=\"gmm\")";
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
    Ok(())
}

/// `IVOptions.gmm_weight_type`をパースし、該当するgmm_weight_typeのときのみ`cluster`/
/// `hac_lags`/`hac_time`を抽出したうえで`engine::iv::gmm::WeightType`を組み立てる
/// （`estimator="gmm"`のみで使用、`cov_type`側の同種の関数は`linear::common::parse_cov_type`
/// を共有しているのに対し、こちらは`WeightType`が`CovType`と異なる型のため独立実装）。
///
/// `cluster`/`hac_lags`/`hac_time`は`cov_type`と共用する（モジュールdocコメント
/// 「GMMのgmm_weight_type」参照、`IVOptions`に別フィールドを増やさない設計）。
///
/// 戻り値に正規化済み小文字文字列を含めるのは`linear::common::parse_cov_type`と同じ理由
/// （`IVResult.gmm_weight_type`の構築時に`options.gmm_weight_type.to_lowercase()`を
/// 再計算せずに済ませるため）。
///
/// # Errors
/// `gmm_weight_type`の文字列が既知の値のいずれでもない場合は`ValidationError`。それ以外
/// （列の抽出時に発覚する問題等）は`column_extraction`の責務で`ValidationError`。
fn parse_weight_type(df: &DataFrame, options: &IVOptions) -> PyResult<(WeightType, String)> {
    let weight_type_lower = options
        .gmm_weight_type
        .as_deref()
        .unwrap_or(DEFAULT_GMM_WEIGHT_TYPE)
        .to_lowercase();

    let gmm_weight_type = match weight_type_lower.as_str() {
        "classical" => WeightType::Classical,
        "robust" => WeightType::Robust,
        "cluster" => {
            let groups = options
                .cluster
                .as_ref()
                .map(|col_name| extract_group_key_column(df, col_name))
                .transpose()?;
            WeightType::Cluster { groups }
        }
        "hac" => {
            let time_order = options
                .hac_time
                .as_ref()
                .map(|col_name| extract_ordering_f64_column(df, col_name))
                .transpose()?;
            WeightType::Hac {
                lags: options.hac_lags,
                time_order,
            }
        }
        other => {
            return Err(ValidationError::new_err(format!(
                "unknown gmm_weight_type: '{other}'. Expected one of 'classical', \
                 'robust', 'cluster', or 'hac'"
            )));
        }
    };

    Ok((gmm_weight_type, weight_type_lower))
}

/// `IVOptions.gmm_type`/`gmm_max_iter`/`gmm_tol`をパースして`engine::iv::gmm::GmmType`を
/// 組み立てる（`estimator="gmm"`のみで使用）。
///
/// 戻り値は`(GmmType, 正規化済み小文字のgmm_type, 正規化済み小文字のgmm_weight_type)`。
/// `"one_step"`は`gmm_weight_type`を使わない（検証もしない）ため3つ目は`None`。
/// `gmm_type`/`gmm_weight_type`/`gmm_max_iter`/`gmm_tol`の既定値は`None`で、使われる
/// モードのときだけ実効既定値（`"two_step"`/`"classical"`/`100`/`1e-6`）に解決する
/// （`IVOptions`はpyclassで既定値と明示指定を区別できないため）。使われないモードでの
/// 明示指定は`validate_iv_option_usage`が事前に弾いている。
///
/// # Errors
/// - `gmm_type`が未知の値: `ValidationError`
/// - `"iterated"`で`gmm_max_iter`が3未満（負値を含む）、`gmm_tol`が0以下:
///   `IvError::InvalidGmmMaxIter`/`InvalidGmmTol`（engineの検証）
/// - `"two_step"`/`"iterated"`で`gmm_weight_type`が未知の値: `ValidationError`
fn parse_gmm_type(
    df: &DataFrame,
    options: &IVOptions,
) -> PyResult<(GmmType, String, Option<String>)> {
    const DEFAULT_MAX_ITER: usize = 100;
    const DEFAULT_TOL: f64 = 1e-6;

    let gmm_type_lower = options
        .gmm_type
        .as_deref()
        .unwrap_or(DEFAULT_GMM_TYPE)
        .to_lowercase();
    match gmm_type_lower.as_str() {
        "one_step" | "two_step" => {
            if gmm_type_lower == "one_step" {
                return Ok((GmmType::OneStep, gmm_type_lower, None));
            }
            let (weight, weight_lower) = parse_weight_type(df, options)?;
            Ok((
                GmmType::TwoStep { weight },
                gmm_type_lower,
                Some(weight_lower),
            ))
        }
        "iterated" => {
            let (weight, weight_lower) = parse_weight_type(df, options)?;
            let max_iter = match options.gmm_max_iter {
                Some(v) => usize::try_from(v)
                    .map_err(|_| iv_error_to_pyerr(IvError::InvalidGmmMaxIter { max_iter: v }))?,
                None => DEFAULT_MAX_ITER,
            };
            let tol = options.gmm_tol.unwrap_or(DEFAULT_TOL);
            Ok((
                GmmType::Iterated {
                    weight,
                    max_iter,
                    tol,
                },
                gmm_type_lower,
                Some(weight_lower),
            ))
        }
        other => Err(ValidationError::new_err(format!(
            "unknown gmm_type: '{other}'. Expected one of 'one_step', 'two_step', or 'iterated'"
        ))),
    }
}

/// Pythonから渡された `data` / `y` / `x_exog` / `x_endog` / `instruments` / `options` を
/// 検証し、`engine::iv::common::IvInput::from_columns`を呼び出すところまでを行う。
/// `TwoSlsEstimator::fit`/`GmmEstimator::fit`の呼び出し・`IVResult`の構築は`fit`
/// （本ファイル）が行う。
///
/// 戻り値に`estimator`のパース済み小文字文字列（`"2sls"`/`"gmm"`のいずれか）を含めるのは、
/// `cov_type_lower`と同じ理由（`fit`が2SLS/GMMのどちらを呼ぶか分岐する際、ここでの妥当性
/// チェックと同じ正規化ロジックを再実装せずに済ませるため。`Logit`の`build_logit_input`が
/// `solver`を`SolverType`にパースして返す設計と同じ考え方だが、IVには`TwoSlsEstimator`/
/// `GmmEstimator`を横断する共通enumが`engine`側に無いため、ここでは正規化済み文字列の
/// まま返す）。
///
/// # Errors
/// - 列の抽出時に発覚する問題（列が存在しない、数値/文字列型にキャストできない、
///   欠損値・NaN・無限大を含む等）は`column_extraction`の責務で`ValidationError`
/// - `y`/`x_exog`/`x_endog`/`instruments`間の重複、各ロール内部の重複、
///   `include_intercept=true`のときの`x_exog`/`x_endog`/`instruments`いずれかと
///   `"const"`列との衝突（`x_exog`だけでなく全ロールが対象）、
///   `x_endog`/`instruments`が空リストの場合（`x_exog`は対象外）は
///   ここ（受け口）の責務で`ValidationError`
/// - `estimator`の文字列が`"2sls"`/`"gmm"`のいずれでもない場合は`ValidationError`
/// - `cov_type`の文字列が不正な場合は`ValidationError`（`linear::common::parse_cov_type`参照）
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
) -> PyResult<(IvInput, EngineCovType, String, String)> {
    let estimator_lower = options.estimator.to_lowercase();
    if estimator_lower != "2sls" && estimator_lower != "gmm" {
        return Err(ValidationError::new_err(format!(
            "unknown estimator: '{}'. Expected one of '2sls' or 'gmm'",
            options.estimator
        )));
    }

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
    let y_slice = extract_f64_column(df, &y)?;

    // ── x_exog/x_endog/instruments列の抽出 ─────────────────────────────
    let mut x_exog_columns: Vec<Vec<f64>> = Vec::with_capacity(x_exog.len());
    for col_name in &x_exog {
        x_exog_columns.push(extract_f64_column(df, col_name)?);
    }
    let mut x_endog_columns: Vec<Vec<f64>> = Vec::with_capacity(x_endog.len());
    for col_name in &x_endog {
        x_endog_columns.push(extract_f64_column(df, col_name)?);
    }
    let mut instrument_columns: Vec<Vec<f64>> = Vec::with_capacity(instruments.len());
    for col_name in &instruments {
        instrument_columns.push(extract_f64_column(df, col_name)?);
    }

    // ── cov_type固有の追加列の抽出（該当するcov_typeのときのみ）─────────────
    validate_iv_option_usage(options, &estimator_lower)?;
    let (cov_type, cov_type_lower) = build_cov_type(
        df,
        &options.cov_type,
        options.cluster.as_deref(),
        options.hac_lags,
        options.hac_time.as_deref(),
    )?;

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

    Ok((input, cov_type, cov_type_lower, estimator_lower))
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
/// - `build_iv_input`が返すエラー（列抽出・y/x_exog/x_endog/instrumentsの重複・
///   `"const"`列衝突・`estimator`/`cov_type`文字列の検証等）は`ValidationError`
/// - `estimator="gmm"`で`gmm_type`/`gmm_weight_type`の文字列が不正、または`gmm_type`と
///   `gmm_max_iter`/`gmm_tol`の組み合わせが矛盾: `ValidationError`（`parse_gmm_type`参照）
/// - `TwoSlsEstimator::fit`/`GmmEstimator::fit`/`compute_first_stage`が返す
///   `engine::iv::common::IvError`（識別の順序条件・第一段階回帰の失敗・`cov_type`起因の
///   エラー・GMM固有のエラー等）は`iv_error_to_pyerr`で変換
pub(crate) fn fit(
    data: PyDataFrame,
    y: String,
    x_exog: Vec<String>,
    x_endog: Vec<String>,
    instruments: Vec<String>,
    options: &IVOptions,
) -> PyResult<IVResult> {
    let df: DataFrame = data.into();
    let (input, cov_type, cov_type_lower, estimator_lower) =
        build_iv_input(&df, y, x_exog, x_endog, instruments, options)?;

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

    // 第一段階回帰・弱操作変数診断は`estimator`によらず共通（モジュールdocコメント参照）。
    // `input`は下でestimatorごとの推定器に移動するため、参照のみで済むこの呼び出しを先に行う。
    let (first_stage, weak_instrument_f_statistics) =
        compute_first_stage(&input, &cov_type, options.confidence_level)
            .map_err(iv_error_to_pyerr)?;
    let weak_instrument_f_df = first_stage.first().map(|(_, first_stage_estimator)| {
        (input.k_instruments(), first_stage_estimator.df_resid())
    });

    if estimator_lower == "gmm" {
        let (gmm_type, gmm_type_lower, weight_type_lower) = parse_gmm_type(&df, options)?;
        let estimator = GmmEstimator::fit(
            input,
            gmm_type,
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
            estimator: estimator_lower,
            gmm_weight_type: weight_type_lower,
            gmm_type: Some(gmm_type_lower),
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
        estimator: estimator_lower,
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

    /// `validate_iv_option_usage`の判定表（`Err`になる/ならない）。エラー文言そのものは
    /// `PyErr`のDisplayがGILを要求するためpytest側（`test_iv_validation.py`）で確認する。
    fn usage_is_ok(configure: impl FnOnce(&mut IVOptions), estimator: &str) -> bool {
        let mut options = default_options();
        options.estimator = estimator.to_string();
        configure(&mut options);
        validate_iv_option_usage(&options, estimator).is_ok()
    }

    #[test]
    fn validate_iv_option_usage_rejects_gmm_options_for_2sls() {
        assert!(!usage_is_ok(
            |o| o.gmm_type = Some("two_step".into()),
            "2sls"
        ));
        assert!(!usage_is_ok(|o| o.gmm_max_iter = Some(10), "2sls"));
        assert!(usage_is_ok(|_| {}, "2sls"));
    }

    #[test]
    fn validate_iv_option_usage_ties_options_to_gmm_type() {
        let one_step = |o: &mut IVOptions| o.gmm_type = Some("one_step".into());
        assert!(!usage_is_ok(
            |o| {
                one_step(o);
                o.gmm_weight_type = Some("robust".into());
            },
            "gmm"
        ));
        assert!(!usage_is_ok(
            |o| o.gmm_max_iter = Some(10),
            "gmm" // 既定のgmm_type（two_step）は反復しない
        ));
        assert!(usage_is_ok(
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
    fn validate_iv_option_usage_shares_cluster_and_hac_options_between_cov_type_and_weight() {
        // gmm_weight_typeだけがcluster/hac_*を使う（どちらか一方でも使えば有効）
        assert!(usage_is_ok(
            |o| {
                o.gmm_weight_type = Some("cluster".into());
                o.cluster = Some("g".into());
            },
            "gmm"
        ));
        assert!(usage_is_ok(
            |o| {
                o.gmm_weight_type = Some("hac".into());
                o.hac_lags = Some(2);
                o.hac_time = Some("t".into());
            },
            "gmm"
        ));
        // cov_typeだけが使う
        assert!(usage_is_ok(
            |o| {
                o.cov_type = "HAC".into();
                o.hac_lags = Some(2);
            },
            "2sls"
        ));
        // どちらも使わない（weightがhacなのにclusterを指定、2slsでweightは使えない）
        assert!(!usage_is_ok(
            |o| {
                o.gmm_weight_type = Some("hac".into());
                o.cluster = Some("g".into());
            },
            "gmm"
        ));
        assert!(!usage_is_ok(|o| o.cluster = Some("g".into()), "2sls"));
    }

    #[test]
    fn validate_iv_option_usage_defers_unknown_values_to_later_parsing() {
        assert!(usage_is_ok(
            |o| {
                o.gmm_type = Some("bogus".into());
                o.gmm_max_iter = Some(10);
            },
            "gmm"
        ));
        assert!(usage_is_ok(
            |o| {
                o.cov_type = "bogus".into();
                o.cluster = Some("g".into());
            },
            "2sls"
        ));
    }

    #[test]
    fn build_iv_input_succeeds_for_well_formed_data() {
        let df = well_formed_df();
        let options = default_options();

        let (input, cov_type, cov_type_lower, estimator_lower) = build_iv_input(
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
        assert_eq!(estimator_lower, "2sls");
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

        let (_, _, _, estimator_lower) = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        )
        .unwrap();
        assert_eq!(estimator_lower, "gmm");
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
                groups: Some(vec![
                    "a".to_string(),
                    "a".to_string(),
                    "b".to_string(),
                    "b".to_string()
                ])
            }
        );
    }

    #[test]
    fn build_iv_input_extracts_time_order_when_cov_type_is_hac() {
        let df = df!(
            "y" => [1.0, 2.0, 3.0, 4.0],
            "endog1" => [4.0, 3.0, 2.0, 1.0],
            "z1" => [2.0, 1.0, 4.0, 3.0],
            "t" => [1.0, 2.0, 3.0, 4.0],
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
                time_order: Some(vec![1.0, 2.0, 3.0, 4.0]),
            }
        );
    }
}
