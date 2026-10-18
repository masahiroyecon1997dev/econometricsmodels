//! REの推定オプション・結果、およびPython（polars DataFrame + 列名 + オプション）から
//! `engine::panel::re`（Swamy-Arora分散成分推定・準偏差変換・`cov_type`対応・ハウスマン検定）
//! を呼び出すところまでの一連の処理（データ抽出・pyclass定義、
//! `ReEstimator::fit`への実際の配線・`#[pymodule]`登録を段階的に実装した）。
//!
//! 【責務分離】`.claude/rules/rust-style.md`「Python境界でのデータ受け渡し」参照。
//! polars DataFrameから列ごとの`Vec<f64>`/`Vec<String>`への抽出はここ（`column_extraction`
//! 経由）の責務。`faer::Mat`の組み立ては`engine`側に委ねる。
//!
//! 【言語方針】`.claude/rules/rust-style.md`「言語方針」参照。
//! 公開API（`REOptions`/`REResult`）のdocコメントと、`ValidationError`のメッセージ文字列は
//! 英語。それ以外（このファイルの説明・非公開関数のdocコメント等）は日本語のまま。
//!
//! ## 実装フェーズの分割方針（FE・IV・Logitと同じ3段階、`engine_pybind/src/panel/CLAUDE.md`
//! 「実装フェーズの分割方針」参照）
//!
//! 1. **データ抽出・pyclass定義（完了）**: `REOptions`/`REResult`のpyclass定義、
//!    列抽出・バリデーション・`engine::panel::re::ReInput`構築までを行う`build_re_input`を
//!    実装した。この時点では`#[pymodule]`への登録・実際の`ReEstimator::fit`呼び出しは
//!    行わなかった（FEの1段階目と同じ分割）。
//! 2. **本対応**: `build_re_input`を実際に呼び出す`fit`関数を追加し、`lib.rs`に
//!    `#[pyfunction] fit_re`を新設して`#[pymodule]`に登録する（FEの2段階目相当）。
//!    `build_re_input`/`resolve_re_cov_type`の`#[allow(dead_code)]`属性はこの時点で削除する
//!    （本番経路（`fit_re`）から実際に呼ばれるようになったため）。
//!
//! RE自身に`fixed_effects()`のような追加メソッドは無い（ハウスマン検定は`fit()`内で自動
//! 計算し`REResult`のフィールドに直接含める、`panel-common.md`2.4節）ため、FEの
//! `fixed_effects()`段階（`fixed_effects()`メソッド）に相当する3段目は存在しない。
//! 本対応でこの系統の実装は完結する。
//!
//! ## `REOptions.time`と`REOptions.dk_time`は別物（`FEOptions`と同じ規約）
//!
//! `time`（bareネーミング）は2-way RE（entity + time、バランスパネル）の指定に使う: `Some`なら
//! 2-way・`None`なら1-way（`FEOptions.time`と同じ意味）。`dk_time`はDriscoll-Kraay型パネルHAC
//! （`cov_type="dk"`）専用の時点列で、`time`とは独立に指定できる（`FEOptions.dk_time`と同じ
//! 名前・同じ意味）。**`cov_type="dk"`では`dk_time`が必須**で、`time`にフォールバックしない
//! （意図と違う列が選ばれても気づけない状況を避けるため）。`dk_time`は
//! `ReCovType::Dk { time }`に渡り、RE自身の準偏差変換（`time`）には影響しない。`cov_type`が
//! `"dk"`以外で`dk_time`を指定すると黙って無視されるため`ValidationError`にする
//! （`parse_panel_cov_kind`）。
//!
//! ハウスマン検定とパネル固有R²は1-way・2-wayとも計算される（`re-spec.md`3.6・3.7節）。
//!
//! ## `cov_type`の非対応値
//!
//! REも`hc0`を**サポートしない**（`ReCovType` enumから除外済み、FEと同じ理由・同じ
//! `hc0`専用エラーメッセージ）。
//!
//! ## `x`の空リストを許容しない（ユーザー確認済み・2026-09-20）
//!
//! FEと同じ`validate_x_non_empty`を適用し、`x=[]`を拒否する。REで`x=[]`は
//! 「分散成分（ICC）のみを推定するnullモデル」として単独で意味を持つ標準的なユースケース
//! （パネル・混合モデル分析の"null model"）だが、`panel-common.md`にこの点の明示的な
//! 決定が無く、他手法（OLS/WLS/Logit/Probit/IV/FE）と一貫させる方針をユーザーが
//! 選択した。nullモデル・ICC推定のサポートは別途検討する。

use std::collections::BTreeMap;

use engine::panel::common::TimeKeys;
use engine::panel::re::{ReCovType, ReEffects, ReEstimator, ReInput, VarianceMethod};
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use super::common::{
    PanelCovKind, PanelModel, check_panel_cov_options, panel_error_to_pyerr, parse_panel_cov_kind,
    require_dk_time, validate_dk_time_role,
};
use crate::shared::column_error::ColumnSource;
use crate::shared::column_extraction::{
    extract_entity_groups, extract_f64_column, extract_f64_columns,
    extract_optional_cluster_groups, extract_time_keys,
};
use crate::shared::errors::legacy_validation_error;
use crate::shared::mat_conversion::mat_to_vec;
use crate::shared::option_repr::PyRepr;
use crate::shared::option_values::{
    extract_strict_float, extract_strict_opt_column, extract_strict_opt_int, extract_strict_text,
};
use crate::shared::validation::{
    RoleValue, validate_no_duplicate_roles, validate_no_duplicate_within_role, validate_x_non_empty,
};

/// Estimation options for RE (random effects panel regression).
///
/// See `docs/spec/panel-common.md` for the rationale behind each field's
/// meaning and default value.
// module/from_py_objectの理由は`OLSOptions`と同じ（`engine_pybind/src/linear/ols.rs`参照）。
#[pyclass(from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct REOptions {
    /// Standard error type: one of "classical", "hc1", "hc2", "hc3", "cluster", "dk".
    /// Case-insensitive. Unlike OLS/WLS/IV, "hc0" is **not** supported (no reference
    /// implementation offers it for panel/RE regressions; see the module docstring).
    #[pyo3(get)]
    pub cov_type: String,

    /// Confidence level for confidence intervals, in the range (0, 1).
    /// Defaults to 0.95 (a 95% confidence interval).
    #[pyo3(get)]
    pub confidence_level: f64,

    /// Column name of the time identifier. When set, requests two-way random effects
    /// (entity + time; balanced panels only, an unbalanced panel raises `ValidationError`);
    /// when `None` (default), one-way (entity only). It only defines the random-effects
    /// structure: it is never used as the Driscoll-Kraay time periods (set `dk_time` for
    /// that). The results do not depend on the order of the periods.
    #[pyo3(get)]
    pub time: Option<String>,

    /// Column name to use as the cluster group key when `cov_type="cluster"`. When
    /// `None`, the `entity` argument's column is used automatically. Specifying it with
    /// any other `cov_type` raises `ValidationError`.
    #[pyo3(get)]
    pub cluster: Option<String>,

    /// Column name that defines the time periods of the Driscoll-Kraay HAC when
    /// `cov_type="dk"` (required there, as in `FEOptions.dk_time`). Specifying it with any other
    /// `cov_type` raises `ValidationError`. It may not be the same column as `y` or
    /// `entity` (`ValidationError`), but may be one of the regressors or the same column as
    /// `time`. The periods are ordered by the values of the
    /// column: numerically for integers and floats, chronologically for `Date` and
    /// `Datetime`, in the order of the categories for `Enum`, and alphabetically for
    /// strings and `Categorical`.
    #[pyo3(get)]
    pub dk_time: Option<String>,

    /// Bandwidth for Driscoll-Kraay HAC when `cov_type="dk"`. When `None`, computed
    /// automatically via `floor(4*(t/100)^(2/9))` (`t` = number of unique time periods).
    /// Specifying it with any other `cov_type` raises `ValidationError`.
    #[pyo3(get)]
    pub dk_bandwidth: Option<i64>,

    /// Estimator of the variance components `sigma_eps^2`, `sigma_u^2` (and `sigma_lambda^2`
    /// for two-way effects), one of "swamy_arora" (default), "wallace_hussain", "amemiya"
    /// (case-insensitive; `plm`'s `random.method` "swar", "walhus", "amemiya"). Applies to
    /// one-way and two-way effects alike. "swamy_arora" on an unbalanced one-way panel is the
    /// `linearmodels` estimator, which differs from `plm`'s; "wallace_hussain" and "amemiya" follow
    /// `plm` (closed forms on a balanced panel, the trace-term forms on an unbalanced one-way
    /// panel). "amemiya" raises `ValidationError` when a regressor has no within variation
    /// (time-invariant; for two-way effects, varying in at most one dimension).
    #[pyo3(get)]
    pub variance_method: String,
}

#[pymethods]
impl REOptions {
    #[new]
    #[pyo3(signature = (
        cov_type = "cluster".to_string(),
        confidence_level = 0.95,
        time = None,
        cluster = None,
        dk_time = None,
        dk_bandwidth = None,
        variance_method = "swamy_arora".to_string(),
    ))]
    fn new(
        #[pyo3(from_py_with = crate::shared::option_values::cov_type_arg)] cov_type: String,
        #[pyo3(from_py_with = crate::shared::option_values::confidence_level_arg)] confidence_level: f64,
        #[pyo3(from_py_with = crate::shared::option_values::time_arg)] time: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::cluster_arg)] cluster: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::dk_time_arg)] dk_time: Option<String>,
        #[pyo3(from_py_with = crate::shared::option_values::dk_bandwidth_arg)] dk_bandwidth: Option<
            i64,
        >,
        #[pyo3(from_py_with = crate::shared::option_values::variance_method_arg)]
        variance_method: String,
    ) -> Self {
        Self {
            cov_type,
            confidence_level,
            time,
            cluster,
            dk_time,
            dk_bandwidth,
            variance_method,
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
    fn set_variance_method(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.variance_method = extract_strict_text(value, "variance_method")?;
        Ok(())
    }

    #[setter]
    fn set_time(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.time = extract_strict_opt_column(value, "time")?;
        Ok(())
    }

    #[setter]
    fn set_dk_time(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.dk_time = extract_strict_opt_column(value, "dk_time")?;
        Ok(())
    }

    #[setter]
    fn set_cluster(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.cluster = extract_strict_opt_column(value, "cluster")?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "REOptions(cov_type={}, confidence_level={}, time={}, cluster={}, dk_time={}, \
             dk_bandwidth={}, variance_method={})",
            self.cov_type.py_repr(),
            self.confidence_level.py_repr(),
            self.time.py_repr(),
            self.cluster.py_repr(),
            self.dk_time.py_repr(),
            self.dk_bandwidth.py_repr(),
            self.variance_method.py_repr()
        )
    }
}

/// Estimation results for RE.
///
/// Structured data only (no `summary()`); see `docs/spec/panel-common.md`
/// section 2. All array-valued fields (`params`, `std_errors`, etc.) share the same order
/// as `param_names` (`param_names[0]` is always `"const"`, since RE — unlike FE — has an
/// intercept).
///
/// The Hausman test (`hausman_statistic` / `hausman_p_value` / `hausman_df`) is computed
/// automatically inside `fit()` and included directly here (unlike FE's
/// `fixed_effects()`, RE has no separate diagnostic method; see `panel-common.md`
/// section 2.4). All three are `None` when there is no regressor left to test: every
/// regressor is time-invariant, or varies only over time on a balanced panel, or the
/// within-transformed copy of every time-varying regressor is aliased (its entity or time mean
/// is also a regressor, as in a correlated-random-effects specification); for two-way effects,
/// also when every regressor varies in at most one dimension. If the comparison cannot be
/// computed, `fit()` fails instead.
// `REResult`はRust側で組み立ててPythonに返すだけの型で、Python側からの生成・引数として
// 受け取ることは想定していないため`skip_from_py_object`（`OLSResult`/`FEResult`と同じ理由）。
//
// 全フィールドが即値（`Vec`/`String`/`f64`/`usize`/`Option<...>`）で、FEの
// `fixed_effects()`のようなオンデマンド計算メソッドを持たないため、`estimator`のような
// 非公開フィールドは不要（`Clone`も問題なく派生できる）。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct REResult {
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
    /// Number of panel entities (`panel-common.md` section 2.1, following the
    /// pyfixest/plm precedent).
    #[pyo3(get)]
    pub n_entities: usize,
    /// Number of unique time periods for two-way random effects (`REOptions.time` set);
    /// `None` for one-way, also when only `dk_time` is given. Two-way requires a balanced
    /// panel, so this equals the number of observations per entity.
    #[pyo3(get)]
    pub n_periods: Option<usize>,
    /// Estimator of the variance components actually used (echoes `REOptions.variance_method`,
    /// normalized to lowercase): "swamy_arora", "wallace_hussain" or "amemiya". `sigma2_eps`,
    /// `sigma2_u`, `sigma2_lambda` and the weights `theta*` are the values of this estimator.
    #[pyo3(get)]
    pub variance_method: String,
    /// Estimated variance of the idiosyncratic error, `sigma_eps^2`. Swamy-Arora: the residual
    /// variance of the internal fixed-effects regression with the same effect structure as the
    /// model (entity for one-way, entity and time for two-way), whose degrees of freedom exclude
    /// the regressors without within variation, as in `plm` (`docs/spec/re-spec.md` section
    /// 3.1). Wallace-Hussain and Amemiya: from the within-transformed first-stage residuals, see
    /// section 3.1.2.
    #[pyo3(get)]
    pub sigma2_eps: f64,
    /// Estimated variance of the entity effect, `sigma_u^2` (a negative estimate is raised to
    /// 0). Swamy-Arora on an unbalanced panel is the `linearmodels` estimator, which differs
    /// from `plm`'s.
    #[pyo3(get)]
    pub sigma2_u: f64,
    /// Estimated variance of the time effect, `sigma_lambda^2`, for two-way effects (a negative
    /// estimate is raised to 0); `None` for one-way.
    #[pyo3(get)]
    pub sigma2_lambda: Option<f64>,
    /// One-way quasi-demeaning weight of each entity,
    /// `theta_i = 1 - sqrt(sigma_eps^2 / (T_i * sigma_u^2 + sigma_eps^2))` with `T_i` the number
    /// of observations of the entity (0 when `sigma_u^2` is 0; different across entities only on
    /// an unbalanced panel). The keys are the entity ids as strings, in lexicographic order (so
    /// `"10"` precedes `"2"` for integer ids); `None` for two-way.
    #[pyo3(get)]
    pub theta: Option<BTreeMap<String, f64>>,
    /// Two-way weight of the entity means in the GLS transformation
    /// `x - theta_1 * mean_i(x) - theta_2 * mean_t(x) + theta_3 * mean(x)`:
    /// `theta_1 = 1 - sigma_eps / sigma_1` with `sigma_1^2 = sigma_eps^2 + T * sigma_u^2`;
    /// `None` for one-way.
    #[pyo3(get)]
    pub theta_entity: Option<f64>,
    /// Two-way weight of the time means: `theta_2 = 1 - sigma_eps / sigma_2` with
    /// `sigma_2^2 = sigma_eps^2 + N * sigma_lambda^2`; `None` for one-way.
    #[pyo3(get)]
    pub theta_time: Option<f64>,
    /// Two-way weight of the overall mean:
    /// `theta_3 = theta_1 + theta_2 - 1 + sigma_eps / sigma_4` with
    /// `sigma_4^2 = sigma_eps^2 + T * sigma_u^2 + N * sigma_lambda^2`. It is not the product
    /// `theta_1 * theta_2`, and it is exactly 0 when `sigma_u^2` or `sigma_lambda^2` is 0;
    /// `None` for one-way.
    #[pyo3(get)]
    pub theta_total: Option<f64>,
    /// Standard error type actually used (echoes `REOptions.cov_type`, normalized to
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
    /// Denominator degrees of freedom of `f_statistic` (`None` when `f_statistic` is `None`). Follows
    /// `cov_type` like the t-tests (`G - 1` for cluster, `T - 1` for Driscoll-Kraay,
    /// `df_resid` otherwise).
    #[pyo3(get)]
    pub f_df_denom: Option<usize>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    /// Within R² (entity-demeaned, or demeaned by entity and time for two-way effects);
    /// `0.0` when no regressor varies after demeaning.
    #[pyo3(get)]
    pub r_squared_within: f64,
    /// Between R² (entity means, with the intercept).
    #[pyo3(get)]
    pub r_squared_between: f64,
    /// Overall R² (untransformed variables, with the intercept).
    #[pyo3(get)]
    pub r_squared_overall: f64,
    /// Regression-based Hausman test statistic (chi-squared version, Wald test that the
    /// within-transformed regressors are jointly zero in the auxiliary regression;
    /// `docs/spec/re-spec.md` section 3.7). The comparison is against FE with the same
    /// structure as the RE model (entity for one-way, entity and time for two-way). The Wald
    /// test's covariance follows `cov_type`. Always non-negative. `None` when no slope
    /// coefficient can be tested: a regressor that does not vary after the within
    /// transformation (time-invariant; for two-way effects, varying in only one dimension) or
    /// whose within-transformed copy is not identified (for example a regressor that varies
    /// only over time on a balanced panel, or an interaction with zero entity and time means)
    /// is left out of the comparison. If the auxiliary regression cannot be computed (e.g.
    /// too few clusters or periods for the robust covariance) `fit()` fails.
    #[pyo3(get)]
    pub hausman_statistic: Option<f64>,
    /// p-value of `hausman_statistic` (upper-tail chi-squared probability).
    #[pyo3(get)]
    pub hausman_p_value: Option<f64>,
    /// Degrees of freedom of the Hausman test (number of tested slope coefficients: the
    /// regressors that still vary after the within transformation and can be identified,
    /// which is `df_model - 1` when every regressor does). A within-transformed copy is not
    /// counted when it is aliased, for example when the regressor's entity (or time) mean is
    /// also a regressor (correlated random effects).
    #[pyo3(get)]
    pub hausman_df: Option<usize>,
}

/// [`PanelCovKind`]に対応する列（`cluster`・`dk_time`）をDataFrameから抽出して
/// `engine::panel::re::ReCovType`を組み立てる。
///
/// `cov_type="dk"`では`dk_time`（必須）の列から`TimeKeys`を抽出して`ReCovType::Dk`に渡す
/// （`FeCovType::Dk`と同じ。2-wayの`time`とは独立、モジュールdoc参照）。使われない
/// オプションの検査は`check_panel_cov_options`の責務。
///
/// # Errors
/// 列の抽出時に発覚する問題は`column_extraction`の責務で`ValidationError`。`Dk`で
/// `dk_time`が未指定の場合も`ValidationError`（`check_panel_cov_options`を通していれば
/// 起きない）。
fn resolve_re_cov_type(
    df: &DataFrame,
    kind: PanelCovKind,
    options: &REOptions,
) -> PyResult<ReCovType> {
    Ok(match kind {
        PanelCovKind::Classical => ReCovType::Classical,
        PanelCovKind::Hc1 => ReCovType::Hc1,
        PanelCovKind::Hc2 => ReCovType::Hc2,
        PanelCovKind::Hc3 => ReCovType::Hc3,
        PanelCovKind::Cluster => ReCovType::Cluster {
            groups: extract_optional_cluster_groups(
                df,
                options.cluster.as_deref(),
                ColumnSource::new("cluster", "data"),
            )?,
        },
        PanelCovKind::Dk => {
            let dk_time = require_dk_time(options.dk_time.as_deref(), PanelModel::Re)?;
            ReCovType::Dk {
                bandwidth: options.dk_bandwidth,
                time: extract_time_keys(df, dk_time, ColumnSource::new("dk_time", "data"))?,
            }
        }
    })
}

/// `REOptions.variance_method`（大文字小文字を区別しない）を`VarianceMethod`にする。
///
/// # Errors
/// 未知の値は`ValidationError`。
fn parse_variance_method(value: &str) -> PyResult<VarianceMethod> {
    match value.to_lowercase().as_str() {
        "swamy_arora" => Ok(VarianceMethod::SwamyArora),
        "wallace_hussain" => Ok(VarianceMethod::WallaceHussain),
        "amemiya" => Ok(VarianceMethod::Amemiya),
        _ => Err(legacy_validation_error(format!(
            "unknown variance_method: '{value}'. Expected one of 'swamy_arora', \
             'wallace_hussain', or 'amemiya'"
        ))),
    }
}

/// Pythonから渡された `data` / `y` / `x` / `entity` / `options` を検証し、
/// `engine::panel::re::ReInput::from_columns`を呼び出すところまでを行う。
/// `ReEstimator::fit`の呼び出し・`REResult`の構築は後続issueの`fit`関数が行う
/// （モジュールdoc「実装フェーズの分割方針」参照）。
///
/// `options.time`は2-wayの時間次元として`ReInput`に渡し、`ReEffects::TwoWay`を選ぶ
/// （`None`なら`OneWay`）。`options.dk_time`は`resolve_re_cov_type`が`ReCovType::Dk`に渡す
/// （`"dk"`以外で指定された場合は`ValidationError`）。
///
/// # Errors
/// - `x`が空リストの場合は`ValidationError`（モジュールdoc「`x`の空リストを許容しない」
///   参照）。`y`/`entity`/`dk_time`/`x`間の重複・`x`内部の重複も同じく`validation.rs`の
///   責務で`ValidationError`
/// - 列の抽出時に発覚する問題（列が存在しない、数値/文字列型にキャストできない、
///   欠損値・NaN・無限大を含む等）は`column_extraction`の責務で`ValidationError`
/// - `cov_type`の文字列が不正な場合は`ValidationError`（`parse_panel_cov_kind`参照）
/// - それ以外（`y`/`entity`/`dk_time`間の行数不一致等）は`engine::panel::common::PanelError`
///   から`panel_error_to_pyerr`で変換
pub(crate) fn build_re_input(
    df: &DataFrame,
    y: String,
    x: Vec<String>,
    entity: String,
    options: &REOptions,
) -> PyResult<(ReInput, ReEffects, ReCovType, String, VarianceMethod)> {
    // DataFrameを見ない検査を先に行う。報告順は`docs/guide/validation.md`
    // 「Order in which errors are reported」: オプションの値（`cov_type` →
    // `variance_method`）→ オプションの組合せ → ロール構造 → `y`/`x`の列 →
    // オプション列 → エンジン。
    let cov_kind = parse_panel_cov_kind(&options.cov_type, PanelModel::Re)?;
    let variance_method = parse_variance_method(&options.variance_method)?;
    check_panel_cov_options(
        cov_kind,
        PanelModel::Re,
        options.cluster.as_deref(),
        options.dk_time.as_deref(),
        options.dk_bandwidth,
    )?;

    validate_x_non_empty("x", &x)?;
    let mut roles = vec![
        ("y", RoleValue::Single(&y)),
        ("entity", RoleValue::Single(&entity)),
    ];
    if let Some(time) = &options.time {
        roles.push(("time", RoleValue::Single(time)));
    }
    roles.push(("x", RoleValue::Multi(&x)));
    validate_no_duplicate_roles(&roles)?;
    validate_no_duplicate_within_role("x", &x)?;
    // `dk_time`は`y`・`entity`との重複だけ拒否する（`x`との重複は許可、`validate_dk_time_role`参照）。
    validate_dk_time_role(&y, &entity, options.dk_time.as_deref())?;

    // ── y/x/entity列の抽出 ─────────────────────────────────────────────
    let y_slice = extract_f64_column(df, &y, ColumnSource::new("y", "data"))?;

    let x_slices = extract_f64_columns(df, &x, ColumnSource::new("x", "data"))?;

    let entity_groups = extract_entity_groups(df, &entity, ColumnSource::new("entity", "data"))?;

    // ── `time`列の抽出（2-way REを指定した場合のみ）───────────────────────
    let time_keys: Option<TimeKeys> = options
        .time
        .as_ref()
        .map(|col_name| extract_time_keys(df, col_name, ColumnSource::new("time", "data")))
        .transpose()?;
    let effects = if time_keys.is_some() {
        ReEffects::TwoWay
    } else {
        ReEffects::OneWay
    };

    // ── cov_type固有の追加列の抽出（該当するcov_typeのときのみ）─────────────
    let cov_type = resolve_re_cov_type(df, cov_kind, options)?;

    let input = ReInput::from_group_codes(&y_slice, &x_slices, x, entity_groups, time_keys, y)
        .map_err(panel_error_to_pyerr)?;

    Ok((
        input,
        effects,
        cov_type,
        cov_kind.name().to_string(),
        variance_method,
    ))
}

/// Pythonから渡された `data` / `y` / `x` / `entity` / `options` を検証し、
/// `build_re_input`で構築した`ReInput`に対して`engine::panel::re::ReEstimator::fit`を
/// 呼び出し、`REResult`として返す。
///
/// `n_entities`は`ReEstimator::input().n_entities()`（`ReInput`が構築時に一度だけ作った
/// エンティティコードのユニーク数）から取得する。
///
/// `params`/`param_names`/`residuals`/`dep_var_name`/`n_obs`/`log_likelihood`/`aic`/`bic`は
/// `ReEstimator::estimator()`（内部で委譲した`OlsEstimator`）から取得する。FEと異なり
/// `aic`/`bic`もそのまま`estimator()`委譲でよい——REの`df_model`が`OlsInput::k()`と自動的に
/// 一致する設計のため、`OlsEstimator`委譲時点で既に正しい値になっている
/// （`engine/src/panel/CLAUDE.md`「`df_resid`/`df_model`（`re-spec.md`3.3節）」参照。FEの
/// `aic`/`bic`のようなRE独自の再計算は不要）。`std_errors`/`test_stats`/`p_values`/
/// `conf_lower`/`conf_upper`/`df_resid`/`df_model`/`f_statistic`/`f_p_value`/
/// `r_squared_within`/`r_squared_between`/`r_squared_overall`/`hausman_statistic`/
/// `hausman_p_value`/`hausman_df`は`ReEstimator`自身のgetterから取得する（`cov_type`・
/// REの切片除外等を反映して独自に計算し直した値のため、`estimator()`委譲では取り違えに
/// なる、`FeEstimator`と同じ理由）。
///
/// # Errors
/// - `build_re_input`が返すエラー（列抽出・y/x/entity/dk_timeの重複・`cov_type`文字列の
///   検証等）は`ValidationError`
/// - `ReEstimator::fit`が返す`engine::panel::common::PanelError`（Swamy-Arora分散成分
///   推定の失敗——内部FE推定のsingleton検出・自由度不足、between回帰の失敗——、
///   準偏差変換済みデータへの委譲失敗、クラスター数不足、HAC関連のバリデーション等）は
///   `panel_error_to_pyerr`で変換
pub(crate) fn fit(
    data: PyDataFrame,
    y: String,
    x: Vec<String>,
    entity: String,
    options: &REOptions,
) -> PyResult<REResult> {
    let df: DataFrame = data.into();
    let (input, effects, cov_type, cov_type_lower, variance_method) =
        build_re_input(&df, y, x, entity, options)?;

    let estimator = ReEstimator::fit_with_variance_method(
        input,
        effects,
        cov_type,
        variance_method,
        options.confidence_level,
    )
    .map_err(panel_error_to_pyerr)?;
    let ols = estimator.estimator();
    let two_way_theta = estimator.two_way_theta();

    Ok(REResult {
        params: mat_to_vec(ols.params()),
        std_errors: mat_to_vec(estimator.std_errors()),
        test_stats: mat_to_vec(estimator.test_stats()),
        stat_dist: estimator.stat_dist().name().to_string(),
        stat_df: estimator.stat_dist().df().map(|df| df as i64),
        p_values: mat_to_vec(estimator.p_values()),
        conf_lower: mat_to_vec(estimator.conf_lower()),
        conf_upper: mat_to_vec(estimator.conf_upper()),
        param_names: ols.input().param_names().to_vec(),
        residuals: mat_to_vec(ols.residuals()),
        dep_var_name: ols.input().dep_var_name().to_string(),
        n_obs: ols.input().nobs(),
        df_resid: estimator.df_resid(),
        df_model: estimator.df_model(),
        n_entities: estimator.input().n_entities(),
        n_periods: estimator.n_periods(),
        variance_method: variance_method.as_str().to_string(),
        sigma2_eps: estimator.sigma2_eps(),
        sigma2_u: estimator.sigma2_u(),
        sigma2_lambda: estimator.sigma2_lambda(),
        theta: estimator.theta_by_entity().map(|theta| {
            estimator
                .input()
                .entity_labels()
                .iter()
                .cloned()
                .zip(theta.iter().copied())
                .collect()
        }),
        theta_entity: two_way_theta.map(|t| t.0),
        theta_time: two_way_theta.map(|t| t.1),
        theta_total: two_way_theta.map(|t| t.2),
        cov_type: cov_type_lower,
        dk_bandwidth_used: estimator.dk_bandwidth_used().map(|bw| bw as i64),
        f_statistic: estimator.f_statistic(),
        f_p_value: estimator.f_p_value(),
        f_df_num: estimator.f_df().map(|(num, _)| num),
        f_df_denom: estimator.f_df().map(|(_, denom)| denom),
        log_likelihood: ols.log_likelihood(),
        aic: ols.aic(),
        bic: ols.bic(),
        r_squared_within: estimator.r_squared_within(),
        r_squared_between: estimator.r_squared_between(),
        r_squared_overall: estimator.r_squared_overall(),
        hausman_statistic: estimator.hausman_statistic(),
        hausman_p_value: estimator.hausman_p_value(),
        hausman_df: estimator.hausman_df(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::shared::group_codes::GroupCodes;
    use polars::df;

    /// `build_re_input`のテスト全体で使う既定の`REOptions`（`cov_type="cluster"`・
    /// `dk_time=None`）。フィールドごとに上書きして使う。
    fn default_options() -> REOptions {
        REOptions::new(
            "cluster".to_string(),
            0.95,
            None,
            None,
            None,
            None,
            "swamy_arora".to_string(),
        )
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
    fn build_re_input_succeeds_for_well_formed_data() {
        let df = well_formed_df();
        let options = default_options();

        let (input, effects, cov_type, cov_type_lower, _) = build_re_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            "id".to_string(),
            &options,
        )
        .unwrap();

        assert_eq!(input.nobs(), 6);
        assert_eq!(input.x_names(), &["x1".to_string()]);
        assert_eq!(input.time_keys(), None);
        assert_eq!(effects, ReEffects::OneWay);
        assert_eq!(
            cov_type,
            ReCovType::Cluster { groups: None },
            "cluster未指定時はNone（engine側でentity列にフォールバック）"
        );
        assert_eq!(cov_type_lower, "cluster");
    }

    #[test]
    fn build_re_input_passes_dk_time_to_the_cov_type_not_to_the_input() {
        // `dk_time`は`ReCovType::Dk.time`に渡り（HACの時系列順序）、`ReInput`の`time`
        // （2-wayの時間次元）にはならない。
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "dk".to_string();
        options.dk_time = Some("t".to_string());

        let (input, effects, cov_type, _, _) = build_re_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            "id".to_string(),
            &options,
        )
        .unwrap();

        assert_eq!(input.time_keys(), None);
        assert_eq!(effects, ReEffects::OneWay);
        let ReCovType::Dk { time, .. } = cov_type else {
            panic!("expected ReCovType::Dk");
        };
        assert_eq!(time.len(), 6);
        assert_eq!(time.periods(), ["1", "2"]);
    }

    #[test]
    fn build_re_input_selects_two_way_effects_when_time_is_set() {
        let df = well_formed_df();
        let mut options = default_options();
        options.time = Some("t".to_string());

        let (input, effects, ..) = build_re_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            "id".to_string(),
            &options,
        )
        .unwrap();

        assert_eq!(effects, ReEffects::TwoWay);
        assert_eq!(
            input.time_keys().map(|t| t.periods().to_vec()),
            Some(vec!["1".to_string(), "2".to_string()])
        );
    }

    #[test]
    fn build_re_input_rejects_time_that_duplicates_another_role() {
        let df = well_formed_df();
        for column in ["y", "id", "x1"] {
            let mut options = default_options();
            options.time = Some(column.to_string());
            let result = build_re_input(
                &df,
                "y".to_string(),
                vec!["x1".to_string()],
                "id".to_string(),
                &options,
            );
            assert!(result.is_err(), "time={column}");
        }
    }

    #[test]
    fn build_re_input_allows_dk_time_to_equal_time() {
        // 2-wayの時間次元とDKの時点列に同じ列を使うのは自然（`FEOptions`と同じ）。
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "dk".to_string();
        options.time = Some("t".to_string());
        options.dk_time = Some("t".to_string());

        let result = build_re_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            "id".to_string(),
            &options,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn build_re_input_rejects_dk_time_when_cov_type_is_not_dk() {
        // `dk_time`はDK HACの時系列順序専用（ハウスマン検定の比較相手には影響しない）。`cov_type`が"dk"以外だと
        // 黙って無視されるため`ValidationError`にする。
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "classical".to_string();
        options.dk_time = Some("t".to_string());

        let result = build_re_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            "id".to_string(),
            &options,
        );

        assert!(result.is_err());
    }

    #[test]
    fn build_re_input_returns_error_for_empty_x() {
        // `x=[]`を拒否する（モジュールdoc「`x`の空リストを許容しない」参照。
        // FE・OLS/WLS/Logit/Probit/IVと同じ`validate_x_non_empty`）。
        let df = well_formed_df();
        let options = default_options();

        let result = build_re_input(&df, "y".to_string(), vec![], "id".to_string(), &options);

        assert!(result.is_err());
    }

    #[test]
    fn build_re_input_returns_error_when_y_overlaps_entity() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_re_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            "y".to_string(),
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_re_input_returns_error_when_x_overlaps_entity() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_re_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string(), "id".to_string()],
            "id".to_string(),
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_re_input_allows_x_to_overlap_dk_time() {
        // 年トレンドを説明変数に入れつつDKの時点にも使う、等は正当な使い方
        // （`validate_dk_time_role`参照）。
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "dk".to_string();
        options.dk_time = Some("x1".to_string());

        let result = build_re_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            "id".to_string(),
            &options,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn build_re_input_rejects_dk_time_that_is_y_or_entity() {
        let df = well_formed_df();
        for column in ["y", "id"] {
            let mut options = default_options();
            options.cov_type = "dk".to_string();
            options.dk_time = Some(column.to_string());

            let result = build_re_input(
                &df,
                "y".to_string(),
                vec!["x1".to_string()],
                "id".to_string(),
                &options,
            );
            assert!(result.is_err(), "dk_time={column}");
        }
    }

    #[test]
    fn build_re_input_returns_error_when_x_contains_duplicate() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_re_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string(), "x1".to_string()],
            "id".to_string(),
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_re_input_returns_error_for_unknown_cov_type() {
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "unknown".to_string();

        let result = build_re_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            "id".to_string(),
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_re_input_returns_error_for_hc0_cov_type() {
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "hc0".to_string();

        let result = build_re_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            "id".to_string(),
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_re_input_extracts_cluster_groups_when_cov_type_is_cluster_and_cluster_set() {
        let df = df!(
            "y" => [1.0, 2.0, 3.0, 4.0],
            "x1" => [2.0, 4.0, 1.0, 5.0],
            "id" => ["a", "a", "b", "b"],
            "state" => ["x", "y", "x", "y"],
        )
        .unwrap();
        let mut options = default_options();
        options.cluster = Some("state".to_string());

        let (_, _, cov_type, _, _) = build_re_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            "id".to_string(),
            &options,
        )
        .unwrap();

        assert_eq!(
            cov_type,
            ReCovType::Cluster {
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
    fn build_re_input_dk_requires_dk_time_and_uses_bandwidth() {
        // `cov_type="dk"`は`dk_time`が必須で、`time`にはフォールバックしない。
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "dk".to_string();
        options.dk_bandwidth = Some(2);
        options.time = Some("t".to_string());
        let build = |options: &REOptions| {
            build_re_input(
                &df,
                "y".to_string(),
                vec!["x1".to_string()],
                "id".to_string(),
                options,
            )
        };

        assert!(build(&options).is_err(), "dk_time unset must be rejected");

        options.dk_time = Some("t".to_string());
        let (_, _, cov_type, _, _) = build(&options).unwrap();
        let ReCovType::Dk { bandwidth, .. } = cov_type else {
            panic!("expected ReCovType::Dk");
        };
        assert_eq!(bandwidth, Some(2));
    }

    #[test]
    fn parse_variance_method_is_case_insensitive_and_returns_the_canonical_name() {
        for (input, expected) in [
            ("swamy_arora", "swamy_arora"),
            ("Swamy_Arora", "swamy_arora"),
            ("WALLACE_HUSSAIN", "wallace_hussain"),
            ("Amemiya", "amemiya"),
        ] {
            let Ok(method) = parse_variance_method(input) else {
                panic!("expected Ok for input={input}");
            };
            assert_eq!(method.as_str(), expected, "input={input}");
        }
    }

    #[test]
    fn parse_variance_method_rejects_aliases_and_unknown_values() {
        for value in ["swamy-arora", "swamyarora", "walhus", "bogus", ""] {
            assert!(parse_variance_method(value).is_err(), "{value}");
        }
    }
}
