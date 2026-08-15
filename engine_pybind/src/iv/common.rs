//! `iv`系統（2SLS/GMM）で共有するユーティリティ。
//!
//! `.claude/rules/rust-style.md`「ファイル・ディレクトリ構成」: 系統内で共有するロジックは
//! `<系統>/common.rs`に置く（`engine_pybind/src/linear/common.rs`と同じ位置づけ）。
//! `IvOptions`/`IvResult`/`build_iv_input`は2SLS/GMMどちらの`method`でも共有する
//! （`fit_iv`という単一エントリポイントの背後で`method`により推定方式を切り替える設計、
//! `docs/planning/specs/iv-api-design.md`1.2節・6.2節）ため、系統内共有ロジックの
//! 置き場所という位置づけに素直に合致する（`two_sls.rs`/`gmm.rs`のような手法ごとの
//! ファイル分割はしない）。
//!
//! `IvError`の`Common`バリアント（`engine::error::CommonError`）は`crate::errors::
//! common_error_to_pyerr`に委譲する（系統ごとに同じ判定ロジックを重複させない）。
//!
//! ## 実装の経緯（要点のみ、詳細は各コミット・`engine/src/iv/CLAUDE.md`参照）
//!
//! `IvOptions`/`IvResult`のpyclass定義・`build_iv_input`（Issue #159）→`TwoSlsEstimator::fit`
//! への配線（Issue #169）→弱操作変数診断・Wu-Hausman・Sargan（Issue #163/#164/#167）→
//! `first_stage()`（Issue #170）の順に段階実装した。**`method="gmm"`は当初
//! `GmmEstimator`（engine側）が点推定のみのスコープ（Issue #160）だったため長らく
//! `ValidationError`で弾いていたが、GMM側のcov_type対応（本来Issue #166の完了条件
//! だったが実装漏れだったことが発覚、`gmm.rs`参照）を実装したうえで、本ファイルでも
//! 実際に配線した**（`fit_iv`から両`method`を呼び分ける）。
//!
//! ## `first_stage()`/`weak_instrument_f_statistics`は`method`に依存しない共通ロジック
//!
//! 第一段階回帰（`x_endog[j] ~ x_exog + instruments`）・弱操作変数診断（部分F統計量）は、
//! `engine::iv::common::compute_first_stage`（2SLS/GMM間で共有、`engine/src/iv/CLAUDE.md`
//! 参照）を`fit`が`method`によらず常に呼ぶことで、GMMでも2SLSと同じ診断情報を提供する
//! （ユーザー確認済み）。`TwoSlsEstimator::fit`は内部でも同じ関数を呼ぶため、
//! `method="2sls"`では第一段階回帰が二重計算になるが、OLS自体が軽量なため許容する
//! （`GmmEstimator`のように第一段階回帰を必要としない推定器に合わせて`IvResult`側を
//! 単純にする方を優先した設計判断）。
//!
//! `IvResult`は元々`estimator: TwoSlsEstimator`という2SLS専用の非公開フィールドで
//! `first_stage()`を実装していたが、GMM配線にあたり`first_stage: Vec<(String,
//! OlsEstimator)>`という`method`非依存の表現に置き換えた（`OlsEstimator → OLSResult`
//! 変換は`linear::ols::ols_estimator_to_result`を再利用、Issue #170で抽出済み）。
//!
//! ## GMMの`weight_type`（`IvOptions.weight_type`/`cluster_col`/`hac_lags`/`time_col`）
//!
//! `weight_type`は`cov_type`とは独立の軸（点推定に使う重み行列の選択、`engine::iv::gmm`の
//! モジュールdocコメント参照）だが、`cluster_col`/`hac_lags`/`time_col`は`cov_type`と
//! 共用する（`IvOptions`に別フィールドを増やさない設計、`parse_weight_type`参照）。
//! `weight_type="cluster"`かつ`cov_type="cluster"`のように両軸が同じクラスター列を
//! 参照する使い方を主に想定するが、`weight_type`と`cov_type`が異なる場合でも同じ列を
//! 共用する（別々のクラスター変数を使い分けたいニーズが出てきたら別フィールド化を検討）。
//!
//! `wu_hausman_statistic`/`wu_hausman_p_value`は`method="gmm"`では常に`None`
//! （`GmmEstimator`はWu-Hausman検定を持たない、`iv-api-design.md`6.6節はTwoSlsEstimator
//! のみのスコープ）。`overid_statistic`/`overid_p_value`は`method="gmm"`では
//! `GmmEstimator::hansen_j_statistic()`/`hansen_j_p_value()`から配線する
//! （`method="2sls"`のSargan検定と同じ`Option<f64>`同士の代入）。

use std::collections::HashMap;

use engine::iv::common::{IvError, IvInput, compute_first_stage};
use engine::iv::gmm::{GmmEstimator, WeightType};
use engine::iv::two_sls::TwoSlsEstimator;
use engine::linear::ols::CovType as EngineCovType;
use engine::linear::ols::OlsEstimator;
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::column_extraction::{extract_f64_column, extract_group_key_column};
use crate::errors::{ComputationError, ValidationError, common_error_to_pyerr};
use crate::linear::common::{least_squares_error_is_computation_error, mat_to_vec};
use crate::linear::ols::{OLSResult, ols_estimator_to_result};
use crate::validation::{
    RoleValue, validate_no_const_collision, validate_no_duplicate_roles,
    validate_no_duplicate_within_role,
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
/// `OlsEstimator::fit`の失敗を包んだもの（Issue #157）。`HausmanRegressionFailed`
/// （Issue #164）も同型だが、Wu-Hausman検定の拡張回帰が理論上到達不能な理由で失敗した
/// 場合のみ構築される防御的なバリアント（想定内の失敗——設計行列の特異性・観測数不足等
/// ——は`wu_hausman_statistic`が`None`になるだけで`IvError`自体は発生しない、
/// `engine/src/iv/CLAUDE.md`参照）。`ValidationError`/`ComputationError`の
/// 判定は`least_squares_error_is_computation_error`（`engine_pybind/src/linear/common.rs`）に
/// 委譲し、`least_squares_error_to_pyerr`と同じ基準を保つ（分類ロジックを重複させない）。
/// Pythonに渡すメッセージは`source.to_string()`ではなく`IvError`自身の`to_string()`
/// （「第一段階/第二段階のどの内生変数で失敗したか」という文脈を含む）を使うため、
/// `least_squares_error_to_pyerr`自体はそのまま呼ばない。
///
/// Issue #169で`fit`（本ファイル）が実際に`#[pymodule]`経路（`fit_iv`）から呼び出すように
/// なった。Issue #159時点では`#[cfg(test)] mod tests`からしか呼ばれておらず
/// `#[allow(dead_code)]`が必要だった（`--all-targets`ビルドでの`#[expect]`の罠、
/// `engine_pybind/src/iv/CLAUDE.md`参照）が、本番経路から呼ばれるようになった今は不要。
pub(crate) fn iv_error_to_pyerr(err: IvError) -> PyErr {
    let message = err.to_string();
    match err {
        IvError::Common(common) => common_error_to_pyerr(common),
        IvError::InsufficientInstruments { .. }
        | IvError::InvalidHacLags { .. }
        | IvError::InvalidGmmIterations { .. }
        | IvError::InvalidGmmConvergence { .. } => ValidationError::new_err(message),
        // `MleError::NonConvergence`（`nonlinear/common.rs`の`mle_error_to_pyerr`）と同じ
        // 分類: パラメータの不正ではなく、計算過程（反復推定）で発覚した問題のため
        // `ComputationError`（Issue #229、`engine/src/iv/CLAUDE.md`参照）。
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
/// See `docs/planning/specs/iv-api-design.md` for the rationale behind each field's
/// meaning and default value. A single `IvOptions`/`fit_iv` pair serves both
/// estimation methods; fields that apply to only one method are documented as such.
// module/from_py_objectの理由は`OLSOptions`/`LogitOptions`と同じ
// （`engine_pybind/src/linear/ols.rs`のコメント参照）。
#[pyclass(from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct IvOptions {
    /// Estimation method: "2sls" (default) or "gmm". Case-insensitive.
    #[pyo3(get, set)]
    pub method: String,

    /// Standard error type: one of "classical", "hc0" through "hc3", "hac", "cluster".
    /// Case-insensitive. For `method="gmm"`, this is independent of `weight_type`
    /// (the weight matrix used for point estimation, see `weight_type` below).
    #[pyo3(get, set)]
    pub cov_type: String,

    /// Whether the engine should automatically add an intercept column to `x_exog`.
    /// `x_endog`/`instruments` never get an automatic intercept column.
    #[pyo3(get, set)]
    pub include_intercept: bool,

    /// Confidence level for confidence intervals, in the range (0, 1).
    /// Defaults to 0.95 (a 95% confidence interval).
    #[pyo3(get, set)]
    pub confidence_level: f64,

    /// Column name to use as the cluster group key when `cov_type="cluster"`.
    /// Ignored when `cov_type` is not "cluster".
    #[pyo3(get, set)]
    pub cluster_col: Option<String>,

    /// Number of lags (bandwidth) for HAC (Newey-West) when `cov_type="hac"`.
    /// When `None`, computed automatically. Ignored when `cov_type` is not "hac".
    #[pyo3(get, set)]
    pub hac_lags: Option<i64>,

    /// Column name giving the time order for HAC when `cov_type="hac"`.
    /// Ignored when `cov_type` is not "hac".
    #[pyo3(get, set)]
    pub time_col: Option<String>,

    /// Weight matrix used for GMM point estimation (`method="gmm"` only): one of
    /// "unadjusted" (alias "homoskedastic"), "robust" (alias "heteroskedastic"),
    /// "cluster", "kernel". Case-insensitive. Ignored when `method="2sls"`. "cluster"/"kernel"
    /// draw from the same `cluster_col`/`hac_lags`/`time_col` fields as `cov_type` (no separate
    /// fields; see module docstring "GMMのweight_type").
    #[pyo3(get, set)]
    pub weight_type: String,

    /// Number of GMM iterations (`method="gmm"` only): 2 (default) for efficient
    /// two-step GMM, 1 for one-step GMM, 3+ for iterated GMM. Ignored when `method="2sls"`.
    #[pyo3(get, set)]
    pub gmm_iterations: i64,

    /// Convergence tolerance for GMM iteration (`method="gmm"` only). When `None` (default),
    /// `gmm_iterations` is treated as a fixed iteration count. When set, `gmm_iterations`
    /// becomes the maximum number of iterations (safety valve) and iteration stops early once
    /// coefficients converge within this tolerance. Ignored when `method="2sls"`.
    #[pyo3(get, set)]
    pub gmm_convergence: Option<f64>,

    /// Whether to raise an error if GMM does not converge within `gmm_iterations` when
    /// `gmm_convergence` is set (`method="gmm"` only). If `False`, returns the result with
    /// `converged=False` instead of raising. Ignored when `method="2sls"` or when
    /// `gmm_convergence` is `None`.
    #[pyo3(get, set)]
    pub raise_on_non_convergence: bool,
}

#[pymethods]
impl IvOptions {
    #[new]
    #[pyo3(signature = (
        method = "2sls".to_string(),
        cov_type = "classical".to_string(),
        include_intercept = true,
        confidence_level = 0.95,
        cluster_col = None,
        hac_lags = None,
        time_col = None,
        weight_type = "unadjusted".to_string(),
        gmm_iterations = 2,
        gmm_convergence = None,
        raise_on_non_convergence = true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        method: String,
        cov_type: String,
        include_intercept: bool,
        confidence_level: f64,
        cluster_col: Option<String>,
        hac_lags: Option<i64>,
        time_col: Option<String>,
        weight_type: String,
        gmm_iterations: i64,
        gmm_convergence: Option<f64>,
        raise_on_non_convergence: bool,
    ) -> Self {
        Self {
            method,
            cov_type,
            include_intercept,
            confidence_level,
            cluster_col,
            hac_lags,
            time_col,
            weight_type,
            gmm_iterations,
            gmm_convergence,
            raise_on_non_convergence,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "IvOptions(method={:?}, cov_type={:?}, include_intercept={}, \
             confidence_level={}, cluster_col={:?}, hac_lags={:?}, time_col={:?}, \
             weight_type={:?}, gmm_iterations={}, gmm_convergence={:?}, \
             raise_on_non_convergence={})",
            self.method,
            self.cov_type,
            self.include_intercept,
            self.confidence_level,
            self.cluster_col,
            self.hac_lags,
            self.time_col,
            self.weight_type,
            self.gmm_iterations,
            self.gmm_convergence,
            self.raise_on_non_convergence,
        )
    }
}

/// Estimation results for IV (2SLS/GMM).
///
/// Structured data only (no `summary()`); see `docs/planning/specs/iv-api-design.md`
/// section 2. All array-valued fields (`params`, `std_errors`, etc.) share the same
/// order as `param_names`.
///
/// `stats` holds the t-statistics (`method="2sls"`) or z-statistics (`method="gmm"`),
/// depending on which distribution the fitted model uses for inference
/// (`iv-api-design.md` 3.2節) — named generically rather than `t_stats`/`z_stats`
/// because this single type is shared by both methods (mirrors the distribution-agnostic
/// naming already used internally by `engine::inference::InferenceStat`).
///
/// `first_stage()`（内生変数ごとの第一段階回帰結果）はここにフィールドとして含めない。
/// `fit()`の戻り値本体には含めず別メソッドとして公開する（`iv-api-design.md`2.2節、
/// Issue #170で実装済み）。`predict()`/`marginal_effects()`用に`LogitResult`/
/// `ProbitResult`が推定量そのものを非公開フィールド`estimator`として保持するのと同じ
/// パターンだが、`IvResult`は`method`（2sls/gmm）非依存の非公開フィールド`first_stage:
/// Vec<(String, OlsEstimator)>`から`first_stage()`をオンデマンドに構築する（下記
/// `first_stage`フィールド参照。当初は`estimator: TwoSlsEstimator`という2sls専用の
/// フィールドだったが、GMM配線時にmethod非依存の表現へ置き換えた——`engine::iv::common::
/// compute_first_stage`が`method`によらず同じ第一段階回帰を計算するため、`fit`の
/// 呼び出し元でこの表現に詰め替えるだけで済む）。
///
/// `fit_iv` (`fit` in this file) constructs and returns it. The core fields above are
/// populated by `TwoSlsEstimator` (`method="2sls"`) or `GmmEstimator` (`method="gmm"`).
/// `weak_instrument_f_statistics`/`first_stage` are populated from `engine::iv::common::
/// compute_first_stage`, independent of `method` (module docstring参照).
/// `wu_hausman_statistic`/`wu_hausman_p_value` are populated from `TwoSlsEstimator::
/// wu_hausman_statistic()`/`wu_hausman_p_value()` for `method="2sls"` (Issue #164);
/// always `None` for `method="gmm"` (`GmmEstimator` has no Wu-Hausman test).
/// `overid_statistic`/`overid_p_value` are populated from `TwoSlsEstimator::
/// sargan_statistic()`/`sargan_p_value()` (Sargan test, `method="2sls"`) or
/// `GmmEstimator::hansen_j_statistic()`/`hansen_j_p_value()` (Hansen J test,
/// `method="gmm"`) (Issue #167).
// `IvResult`はRust側で組み立ててPythonに返すだけの型で、Python側からの生成・引数として
// 受け取ることは想定していないため`skip_from_py_object`（`IvOptions`の`from_py_object`とは
// 対照的、`OLSResult`/`LogitResult`と同じ理由）。
//
// `Clone`を派生しない: `first_stage`の要素`OlsEstimator`が`Clone`を実装していないため
// （`LogitResult`/`ProbitResult`と同じ理由、`.claude/rules/rust-style.md`「推定量構造体の
// 設計」の通りprivateフィールドのみで、Cloneを要求する既存の呼び出し元も無い）。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug)]
pub struct IvResult {
    #[pyo3(get)]
    pub params: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub stats: Vec<f64>,
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
    /// Whether GMM iteration converged (`method="gmm"` only). Always `true` for
    /// `method="2sls"` (2SLS is a closed-form, non-iterative estimator, so convergence is
    /// trivially satisfied — mirrors `GmmEstimator`'s own `gmm_iterations=1` convention,
    /// `engine/src/iv/gmm.rs`参照). When `IvOptions.gmm_convergence` is `None` (fixed
    /// iteration count, the default), always `true` — convergence is only actually checked
    /// when `gmm_convergence` is set (`iv-api-design.md` 6.2節).
    #[pyo3(get)]
    pub converged: bool,
    /// Number of GMM iterations actually run (`method="gmm"` only). Always `1` for
    /// `method="2sls"`.
    #[pyo3(get)]
    pub n_iterations: i64,
    /// Standard error type actually used (echoes `IvOptions.cov_type`, normalized to
    /// lowercase; e.g. `"classical"`, `"hc1"`, `"hac"`, `"cluster"`).
    #[pyo3(get)]
    pub cov_type: String,
    #[pyo3(get)]
    pub f_statistic: f64,
    #[pyo3(get)]
    pub f_p_value: f64,
    #[pyo3(get)]
    pub r_squared: f64,
    #[pyo3(get)]
    pub r_squared_adj: f64,
    /// Weak-instrument diagnostic: the partial F-statistic for each endogenous
    /// variable (keyed by variable name), testing the excluded instruments' joint
    /// significance after partialling out `x_exog` (`iv-api-design.md` 6.4節).
    /// **Not** the same as the plain F-statistic of the corresponding regression in
    /// `first_stage()`, which includes `x_exog`'s contribution too. Empty when
    /// `x_endog=[]`. Computed the same way for both `method="2sls"` and `method="gmm"`
    /// (`engine::iv::common::compute_first_stage`, module docstring参照).
    #[pyo3(get)]
    pub weak_instrument_f_statistics: HashMap<String, f64>,
    /// Overidentification test statistic: Sargan (`method="2sls"`) or Hansen J
    /// (`method="gmm"`). `None` when just-identified (`len(instruments) ==
    /// len(x_endog)`, degrees of freedom 0), per `iv-api-design.md` 6.5節.
    #[pyo3(get)]
    pub overid_statistic: Option<f64>,
    #[pyo3(get)]
    pub overid_p_value: Option<f64>,
    /// Wu-Hausman endogeneity test statistic (joint test over all endogenous
    /// variables, regression-based / `wooldridge_regression` formulation,
    /// `iv-api-design.md` 6.6節). Always computed under the `cov_type` passed to
    /// `fit()` (unlike `weak_instrument_f_statistics`, which is always classical;
    /// `linearmodels`' `wooldridge_regression` uses the same covariance as the
    /// underlying model, and this mirrors that). `None` when there are no endogenous
    /// variables to test (`x_endog=[]`), or when the augmented regression cannot be
    /// estimated (e.g. the first-stage residual has zero variance, or there are too
    /// few observations for the extra residual columns) — neither case fails `fit()`
    /// itself, since the other results remain valid. **Always `None` for
    /// `method="gmm"`** (`GmmEstimator` does not implement this test; `iv-api-design.md`
    /// 6.6節's implementation is `TwoSlsEstimator`-only).
    #[pyo3(get)]
    pub wu_hausman_statistic: Option<f64>,
    #[pyo3(get)]
    pub wu_hausman_p_value: Option<f64>,
    /// `first_stage()`が読む。Python側には公開しない（`OLSResult`の`fitted_values`/
    /// `has_intercept`、`LogitResult`/`ProbitResult`の`estimator`と同じ位置づけ）。
    /// `method`非依存（`engine::iv::common::compute_first_stage`から構築、モジュール
    /// docコメント参照）。
    first_stage: Vec<(String, OlsEstimator)>,
}

#[pymethods]
impl IvResult {
    /// Per-endogenous-variable first-stage regression results
    /// (`x_endog[i] ~ x_exog + instruments`), keyed by the endogenous variable name.
    ///
    /// Each value is a full `OlsResults` (the same type OLS's `fit_ols` returns) — the
    /// first stage is a genuine, valid OLS regression in its own right, so no IV-specific
    /// result type is needed (`iv-api-design.md` 2.2節). Its `f_statistic`/`f_p_value`
    /// include `x_exog`'s contribution and are **not** the weak-instrument partial
    /// F-statistic (`weak_instrument_f_statistics`, computed separately by Issue #163).
    /// Computed the same way for both `method="2sls"` and `method="gmm"` (module
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

/// `IvOptions.cov_type`をパースし、該当する`cov_type`のときのみ`cluster_col`/`time_col`を
/// 抽出したうえで`engine::linear::ols::CovType`を組み立てる。
///
/// `engine::linear::ols::CovType`（型そのもの）を流用しているのは、`TwoSlsEstimator::fit`が
/// 第一段階（`OlsEstimator`への委譲）・第二段階（独立実装のサンドイッチ計算、Issue #166）
/// のどちらもこの型で`cov_type`を受け取るため。対応するcov_typeの範囲は`engine::linear::
/// common::parse_cov_type`（OLS/WLS用）と同じだが、`OLSOptions`ではなく`IvOptions`という
/// 別の型に依存するため独立実装している（`linear/common.rs`の`parse_cov_type`のdocコメントが
/// 明記する通り、無理に共通化せず系統ごとに素直に実装する方針）。
///
/// # Errors
/// `cov_type`の文字列が既知の値のいずれでもない場合は`ValidationError`。それ以外
/// （列の抽出時に発覚する問題等）は`column_extraction`の責務で`ValidationError`。
fn parse_iv_cov_type(df: &DataFrame, options: &IvOptions) -> PyResult<(EngineCovType, String)> {
    let cov_type_lower = options.cov_type.to_lowercase();

    let cluster_groups = if cov_type_lower == "cluster" {
        options
            .cluster_col
            .as_ref()
            .map(|col_name| extract_group_key_column(df, col_name))
            .transpose()?
    } else {
        None
    };

    let time_order = if cov_type_lower == "hac" {
        options
            .time_col
            .as_ref()
            .map(|col_name| extract_f64_column(df, col_name))
            .transpose()?
    } else {
        None
    };

    let cov_type = match cov_type_lower.as_str() {
        "classical" | "nonrobust" => EngineCovType::Classical,
        "hc0" => EngineCovType::Hc0,
        "hc1" => EngineCovType::Hc1,
        "hc2" => EngineCovType::Hc2,
        "hc3" => EngineCovType::Hc3,
        "hac" => EngineCovType::Hac {
            lags: options.hac_lags,
            time_order,
        },
        "cluster" => EngineCovType::Cluster {
            groups: cluster_groups,
        },
        other => {
            return Err(ValidationError::new_err(format!(
                "unknown cov_type: '{other}'. Expected one of 'classical', 'hc0' through \
                 'hc3', 'hac', or 'cluster'"
            )));
        }
    };

    Ok((cov_type, cov_type_lower))
}

/// `IvOptions.weight_type`をパースし、該当するweight_typeのときのみ`cluster_col`/
/// `hac_lags`/`time_col`を抽出したうえで`engine::iv::gmm::WeightType`を組み立てる
/// （`method="gmm"`のみで使用、`parse_iv_cov_type`と対になる関数）。
///
/// `cluster_col`/`hac_lags`/`time_col`は`cov_type`と共用する（モジュールdocコメント
/// 「GMMのweight_type」参照、`IvOptions`に別フィールドを増やさない設計）。
///
/// # Errors
/// `weight_type`の文字列が既知の値のいずれでもない場合は`ValidationError`。それ以外
/// （列の抽出時に発覚する問題等）は`column_extraction`の責務で`ValidationError`。
fn parse_weight_type(df: &DataFrame, options: &IvOptions) -> PyResult<WeightType> {
    let weight_type_lower = options.weight_type.to_lowercase();

    match weight_type_lower.as_str() {
        "unadjusted" | "homoskedastic" => Ok(WeightType::Unadjusted),
        "robust" | "heteroskedastic" => Ok(WeightType::Robust),
        "cluster" => {
            let groups = options
                .cluster_col
                .as_ref()
                .map(|col_name| extract_group_key_column(df, col_name))
                .transpose()?;
            Ok(WeightType::Cluster { groups })
        }
        "kernel" => {
            let time_order = options
                .time_col
                .as_ref()
                .map(|col_name| extract_f64_column(df, col_name))
                .transpose()?;
            Ok(WeightType::Kernel {
                lags: options.hac_lags,
                time_order,
            })
        }
        other => Err(ValidationError::new_err(format!(
            "unknown weight_type: '{other}'. Expected one of 'unadjusted' ('homoskedastic'), \
             'robust' ('heteroskedastic'), 'cluster', or 'kernel'"
        ))),
    }
}

/// Pythonから渡された `data` / `y` / `x_exog` / `x_endog` / `instruments` / `options` を
/// 検証し、`engine::iv::common::IvInput::from_columns`を呼び出すところまでを行う。
/// `TwoSlsEstimator::fit`/`GmmEstimator::fit`の呼び出し・`IvResult`の構築は`fit`
/// （本ファイル）が行う。
///
/// 戻り値に`method`のパース済み小文字文字列（`"2sls"`/`"gmm"`のいずれか）を含めるのは、
/// `cov_type_lower`と同じ理由（`fit`が2SLS/GMMのどちらを呼ぶか分岐する際、ここでの妥当性
/// チェックと同じ正規化ロジックを再実装せずに済ませるため。`Logit`の`build_logit_input`が
/// `method`を`EngineMethod`にパースして返す設計と同じ考え方だが、IVには`TwoSlsEstimator`/
/// `GmmEstimator`を横断する共通enumが`engine`側に無いため、ここでは正規化済み文字列の
/// まま返す）。
///
/// # Errors
/// - 列の抽出時に発覚する問題（列が存在しない、数値/文字列型にキャストできない、
///   欠損値・NaN・無限大を含む等）は`column_extraction`の責務で`ValidationError`
/// - `y`/`x_exog`/`x_endog`/`instruments`間の重複、各ロール内部の重複、
///   `include_intercept=true`のときの`x_exog`と`"const"`列との衝突は
///   ここ（受け口）の責務で`ValidationError`
/// - `method`の文字列が`"2sls"`/`"gmm"`のいずれでもない場合は`ValidationError`
/// - `cov_type`の文字列が不正な場合は`ValidationError`（`parse_iv_cov_type`参照）
/// - それ以外（行数不一致等）は`engine::iv::common::IvError`から`iv_error_to_pyerr`で変換
///   （`IvInput::from_columns`はこの時点では識別可能性を検証しないため、
///   `InsufficientInstruments`はここでは発生しない。`IvInput`の構造体docコメント参照）
pub(crate) fn build_iv_input(
    df: &DataFrame,
    y: String,
    x_exog: Vec<String>,
    x_endog: Vec<String>,
    instruments: Vec<String>,
    options: &IvOptions,
) -> PyResult<(IvInput, EngineCovType, String, String)> {
    let method_lower = options.method.to_lowercase();
    if method_lower != "2sls" && method_lower != "gmm" {
        return Err(ValidationError::new_err(format!(
            "unknown method: '{}'. Expected one of '2sls' or 'gmm'",
            options.method
        )));
    }

    // 完全な多重共線性・意図しない列の重複を早期に、分かりやすいエラーで防ぐ
    // （`validation.rs`に集約、OLS/WLS/Logit/Probitと共通の方針）。`instruments`を
    // リストの末尾に置くのは、`x_exog`/`x_endog`と重複した場合にメッセージの主語を
    // `instruments`側にするため（`validate_no_duplicate_roles`のdocコメント
    // 「呼び出し側の契約」、`iv-api-design.md`1.1.1節参照）。
    validate_no_duplicate_roles(&[
        ("y", RoleValue::Single(&y)),
        ("x_exog", RoleValue::Multi(&x_exog)),
        ("x_endog", RoleValue::Multi(&x_endog)),
        ("instruments", RoleValue::Multi(&instruments)),
    ])?;
    validate_no_duplicate_within_role("x_exog", &x_exog)?;
    validate_no_duplicate_within_role("x_endog", &x_endog)?;
    validate_no_duplicate_within_role("instruments", &instruments)?;
    validate_no_const_collision(&x_exog, options.include_intercept)?;

    // `x_exog`/`x_endog`/`instruments`はいずれも空リストを許容する
    // （`iv-api-design.md`1.1節、`IvInput`の構造体docコメント参照。識別可能性の検証は
    // 2SLS/GMM推定器側の責務）ため、`validate_x_non_empty`は呼ばない。

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
    let (cov_type, cov_type_lower) = parse_iv_cov_type(df, options)?;

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

    Ok((input, cov_type, cov_type_lower, method_lower))
}

/// Pythonから渡された `data` / `y` / `x_exog` / `x_endog` / `instruments` / `options` を
/// 検証し、`build_iv_input`で構築した`IvInput`に対して`method`に応じた推定
/// （`TwoSlsEstimator::fit`または`GmmEstimator::fit`）を呼び出し、`IvResult`として返す。
///
/// `first_stage`/`weak_instrument_f_statistics`は`method`によらず`engine::iv::common::
/// compute_first_stage`（`IvResult`のdocコメント・モジュールdocコメント「`first_stage()`/
/// `weak_instrument_f_statistics`は`method`に依存しない共通ロジック」参照）から構築する。
/// `overid_statistic`/`overid_p_value`は`method="2sls"`では`TwoSlsEstimator::
/// sargan_statistic()`/`sargan_p_value()`（Sargan検定、Issue #167）、`method="gmm"`では
/// `GmmEstimator::hansen_j_statistic()`/`hansen_j_p_value()`（Hansen J検定）から構築する。
/// `wu_hausman_statistic`/`wu_hausman_p_value`は`method="2sls"`では`TwoSlsEstimator::
/// wu_hausman_statistic()`/`wu_hausman_p_value()`（Issue #164）、`method="gmm"`では
/// 常に`None`（`GmmEstimator`は実装しない、モジュールdocコメント参照）。
///
/// # Errors
/// - `build_iv_input`が返すエラー（列抽出・y/x_exog/x_endog/instrumentsの重複・
///   `"const"`列衝突・`method`/`cov_type`文字列の検証等）は`ValidationError`
/// - `method="gmm"`で`weight_type`の文字列が不正: `ValidationError`（`parse_weight_type`参照）
/// - `TwoSlsEstimator::fit`/`GmmEstimator::fit`/`compute_first_stage`が返す
///   `engine::iv::common::IvError`（識別の順序条件・第一段階回帰の失敗・`cov_type`起因の
///   エラー・GMM固有のエラー等）は`iv_error_to_pyerr`で変換
pub(crate) fn fit(
    data: PyDataFrame,
    y: String,
    x_exog: Vec<String>,
    x_endog: Vec<String>,
    instruments: Vec<String>,
    options: &IvOptions,
) -> PyResult<IvResult> {
    let df: DataFrame = data.into();
    let (input, cov_type, cov_type_lower, method_lower) =
        build_iv_input(&df, y, x_exog, x_endog, instruments, options)?;

    // 識別の順序条件（`TwoSlsEstimator::fit`/`GmmEstimator::fit`のいずれも冒頭で検証する
    // のと同じチェック）を`compute_first_stage`より先に行う。過小識別な入力で無駄な
    // 第一段階回帰を走らせないため（rust-reviewerの指摘、`compute_first_stage`自体は
    // この条件を検証しないため呼び出し元の責務）。
    if input.k_instruments() < input.k_endog() {
        return Err(iv_error_to_pyerr(IvError::InsufficientInstruments {
            n_instruments: input.k_instruments(),
            n_endog: input.k_endog(),
        }));
    }

    // 第一段階回帰・弱操作変数診断は`method`によらず共通（モジュールdocコメント参照）。
    // `input`は下でmethodごとの推定器に移動するため、参照のみで済むこの呼び出しを先に行う。
    let (first_stage, weak_instrument_f_statistics) =
        compute_first_stage(&input, &cov_type, options.confidence_level)
            .map_err(iv_error_to_pyerr)?;

    if method_lower == "gmm" {
        let weight_type = parse_weight_type(&df, options)?;
        let estimator = GmmEstimator::fit(
            input,
            weight_type,
            options.gmm_iterations,
            options.gmm_convergence,
            options.raise_on_non_convergence,
            cov_type,
            options.confidence_level,
        )
        .map_err(iv_error_to_pyerr)?;

        return Ok(IvResult {
            params: mat_to_vec(estimator.params()),
            std_errors: mat_to_vec(estimator.std_errors()),
            stats: mat_to_vec(estimator.z_stats()),
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
            n_iterations: estimator.n_iterations(),
            cov_type: cov_type_lower,
            f_statistic: estimator.f_statistic(),
            f_p_value: estimator.f_p_value(),
            r_squared: estimator.r_squared(),
            r_squared_adj: estimator.r_squared_adj(),
            weak_instrument_f_statistics: weak_instrument_f_statistics.into_iter().collect(),
            overid_statistic: estimator.hansen_j_statistic(),
            overid_p_value: estimator.hansen_j_p_value(),
            wu_hausman_statistic: None,
            wu_hausman_p_value: None,
            first_stage,
        });
    }

    let estimator = TwoSlsEstimator::fit(input, cov_type, options.confidence_level)
        .map_err(iv_error_to_pyerr)?;

    Ok(IvResult {
        params: mat_to_vec(estimator.params()),
        std_errors: mat_to_vec(estimator.std_errors()),
        stats: mat_to_vec(estimator.t_stats()),
        p_values: mat_to_vec(estimator.p_values()),
        conf_lower: mat_to_vec(estimator.conf_lower()),
        conf_upper: mat_to_vec(estimator.conf_upper()),
        param_names: estimator.param_names().to_vec(),
        residuals: mat_to_vec(estimator.residuals()),
        dep_var_name: estimator.dep_var_name().to_string(),
        n_obs: estimator.nobs(),
        df_resid: estimator.df_resid(),
        df_model: estimator.df_model(),
        // 2SLSは閉形式・非反復のため常に`converged=true`・`n_iterations=1`
        // （`IvResult.converged`のdocコメント参照）。
        converged: true,
        n_iterations: 1,
        cov_type: cov_type_lower,
        f_statistic: estimator.f_statistic(),
        f_p_value: estimator.f_p_value(),
        r_squared: estimator.r_squared(),
        r_squared_adj: estimator.r_squared_adj(),
        weak_instrument_f_statistics: weak_instrument_f_statistics.into_iter().collect(),
        overid_statistic: estimator.sargan_statistic(),
        overid_p_value: estimator.sargan_p_value(),
        wu_hausman_statistic: estimator.wu_hausman_statistic(),
        wu_hausman_p_value: estimator.wu_hausman_p_value(),
        first_stage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    /// `build_iv_input`のテスト全体で使う既定の`IvOptions`（`method="2sls"`・
    /// `cov_type="classical"`・`include_intercept=true`）。フィールドごとに上書きして使う。
    fn default_options() -> IvOptions {
        IvOptions::new(
            "2sls".to_string(),
            "classical".to_string(),
            true,
            0.95,
            None,
            None,
            None,
            "unadjusted".to_string(),
            2,
            None,
            true,
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

    #[test]
    fn build_iv_input_succeeds_for_well_formed_data() {
        let df = well_formed_df();
        let options = default_options();

        let (input, cov_type, cov_type_lower, method_lower) = build_iv_input(
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
        assert_eq!(method_lower, "2sls");
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
    fn build_iv_input_allows_empty_x_endog_and_instruments() {
        let df = well_formed_df();
        let options = default_options();

        let (input, ..) = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec![],
            vec![],
            &options,
        )
        .unwrap();

        assert_eq!(input.k_endog(), 0);
        assert_eq!(input.k_instruments(), 0);
    }

    #[test]
    fn build_iv_input_returns_error_for_unknown_method() {
        let df = well_formed_df();
        let mut options = default_options();
        options.method = "3sls".to_string();

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
        options.method = "GMM".to_string(); // 大文字小文字を区別しないことも確認

        let (_, _, _, method_lower) = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        )
        .unwrap();
        assert_eq!(method_lower, "gmm");
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
        options.cluster_col = Some("group".to_string());

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
        options.time_col = Some("t".to_string());
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
