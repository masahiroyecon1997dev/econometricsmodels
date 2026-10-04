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
//!    `parse_fe_cov_type`/`panel_error_to_pyerr`はこの時点で本番経路（`fit_fe`）から
//!    実際に呼ばれるようになるため、`#[allow(dead_code)]`はすべて削除する
//!    （`engine_pybind/src/iv/CLAUDE.md`「実装フェーズの分割方針」の初期実装段階と同じ）。
//! 3. **`fixed_effects()`メソッド追加段階**: IVの`first_stage()`と同じ
//!    「追加結果は別メソッド」方針、`docs/spec/fe-spec.md`3.5節）を追加する。`FEResult`に
//!    非公開フィールド`estimator: FeEstimator`（内部で`OlsEstimator`まで保持する）を
//!    追加し、`fixed_effects()`はそこから`FeEstimator::fixed_effects()`をオンデマンドに
//!    呼ぶだけ（IVの`IVResult.first_stage`フィールドが初期実装ではなく後続の拡張で
//!    追加されたのと同じ段階分割）。
//!
//! ## `FEOptions.time`と`FEOptions.dk_time`は別物（`panel-common.md`1.1節）
//!
//! `time`（bareネーミング）は2-way FE（entity + time FE）の指定に使う: `Some`なら2-way・
//! `None`なら1-way。`dk_time`（`cov_type="dk"`の値を接頭辞にした命名）は
//! Driscoll-Kraay型HAC（`cov_type="dk"`）専用の時点列で、`time`とは独立に指定できる
//! （ユーザー確認済み、2026-09-12）。**`cov_type="dk"`では`dk_time`が必須**で、`time`
//! （2-way FEの固定効果の時間次元）を暗黙に借用しない（どの列がDKの時点かを明示させ、
//! 意図と違う列が選ばれても気づけない状況を避けるため。`dk_time`未指定は
//! `ValidationError`、`parse_fe_cov_type`参照）。2-wayでも`dk_time`が常にDK計算に使われ、
//! `time`と別の時間粒度（四半期の固定効果に年次のDK等）も指定できる（詳細な経緯・engine側の
//! 対応する変更は`engine/src/panel/fe.rs`モジュールdoc「Driscoll-Kraay型パネルHAC対応」参照）。
//!
//! ## `cov_type`の非対応値
//!
//! FEは`hc0`を**サポートしない**（linearmodels・fixestともにパネル/FE向けの`hc0`
//! オプションが存在しないため、`FeCovType` enumから除外済み）。OLS/WLS/IVとは
//! 異なりFEの`cov_type`文字列パースはこの1点で分岐が異なるため、`linear::common::
//! parse_cov_type`を流用せず独立実装する。IVの`cov_type`パースは元々（`iv::common::
//! parse_iv_cov_type`という）別実装を持っていたが、`OLSOptions`/`IVOptions`が同名
//! フィールドを持つ偶然の一致により後から`linear::common::parse_cov_type`へ統合された
//! ——FEはこの一致が無く
//! （`hc0`非対応・`Dk`の意味論がFE固有）、意図的に独立実装を維持している点でIVとは事情が
//! 異なる。
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

use engine::panel::common::TimeKeys;
use engine::panel::fe::{FeCovType, FeEffects, FeEstimator, FeInput, FixedEffects};
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_polars::PyDataFrame;

use super::common::{panel_error_to_pyerr, validate_dk_time_role};
use crate::column_extraction::{
    extract_f64_column, extract_f64_columns, extract_group_key_column, extract_time_keys,
};
use crate::errors::ValidationError;
use crate::linear::common::mat_to_vec;
use crate::option_values::{
    extract_strict_float, extract_strict_opt_column, extract_strict_opt_int, extract_strict_text,
};
use crate::validation::{
    RoleValue, reject_unused_option, validate_no_duplicate_roles,
    validate_no_duplicate_within_role, validate_x_non_empty,
};

/// Estimation options for FE (fixed effects panel regression).
///
/// See `docs/spec/panel-common.md` for the rationale behind each field's
/// meaning and default value.
// module/from_py_objectの理由は`OLSOptions`と同じ（`engine_pybind/src/linear/ols.rs`参照）。
#[pyclass(from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct FEOptions {
    /// Standard error type: one of "classical", "hc1", "hc2", "hc3", "cluster", "dk".
    /// Case-insensitive. Unlike OLS/WLS/IV, "hc0" is **not** supported (neither
    /// linearmodels nor fixest offer it for panel/FE regressions).
    #[pyo3(get)]
    pub cov_type: String,

    /// Confidence level for confidence intervals, in the range (0, 1).
    /// Defaults to 0.95 (a 95% confidence interval).
    #[pyo3(get)]
    pub confidence_level: f64,

    /// Column name of the time identifier. When set, requests two-way fixed effects
    /// (entity + time); when `None` (default), one-way (entity only). It only defines the
    /// fixed-effects structure: it is never used as the Driscoll-Kraay time periods (set
    /// `dk_time` for that). The periods are ordered by the values of the column:
    /// numerically for integers and floats, chronologically for `Date` and `Datetime`, in
    /// the order of the categories for `Enum`, and alphabetically for strings and
    /// `Categorical`.
    #[pyo3(get)]
    pub time: Option<String>,

    /// Column name to use as the cluster group key when `cov_type="cluster"`. When
    /// `None`, the `entity` argument's column is used automatically. Specifying it with
    /// any other `cov_type` raises `ValidationError`.
    #[pyo3(get)]
    pub cluster: Option<String>,

    /// Column name that defines the time periods of the Driscoll-Kraay HAC, independent
    /// of `time` (`time` and `dk_time` serve different purposes; see the module
    /// docstring). Required when `cov_type="dk"`, in one-way and two-way models alike:
    /// leaving it unset raises `ValidationError`, and `time` is not used instead. With
    /// two-way effects the HAC may use a different time granularity from the fixed
    /// effects. It may not be the same column as `y` or `entity` (`ValidationError`), but
    /// may be one of the regressors or the same column as `time`. Specifying it with any
    /// other `cov_type` raises `ValidationError`. The periods are ordered by the values of the column (see `time`).
    #[pyo3(get)]
    pub dk_time: Option<String>,

    /// Bandwidth for Driscoll-Kraay HAC when `cov_type="dk"`. When `None`, computed
    /// automatically via `floor(4*(t/100)^(2/9))` (`t` = number of unique time periods).
    /// Specifying it with any other `cov_type` raises `ValidationError`.
    #[pyo3(get)]
    pub dk_bandwidth: Option<i64>,
}

#[pymethods]
impl FEOptions {
    #[new]
    #[pyo3(signature = (
        cov_type = "cluster".to_string(),
        confidence_level = 0.95,
        time = None,
        cluster = None,
        dk_time = None,
        dk_bandwidth = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = crate::option_values::cov_type_arg)] cov_type: String,
        #[pyo3(from_py_with = crate::option_values::confidence_level_arg)] confidence_level: f64,
        #[pyo3(from_py_with = crate::option_values::time_arg)] time: Option<String>,
        #[pyo3(from_py_with = crate::option_values::cluster_arg)] cluster: Option<String>,
        #[pyo3(from_py_with = crate::option_values::dk_time_arg)] dk_time: Option<String>,
        #[pyo3(from_py_with = crate::option_values::dk_bandwidth_arg)] dk_bandwidth: Option<i64>,
    ) -> Self {
        Self {
            cov_type,
            confidence_level,
            time,
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
    fn set_time(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.time = extract_strict_opt_column(value, "time")?;
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
            "FEOptions(cov_type={:?}, confidence_level={}, time={:?}, cluster={:?}, \
             dk_time={:?}, dk_bandwidth={:?})",
            self.cov_type,
            self.confidence_level,
            self.time,
            self.cluster,
            self.dk_time,
            self.dk_bandwidth
        )
    }
}

/// Estimation results for FE.
///
/// Structured data only (no `summary()`); see `docs/spec/panel-common.md`
/// section 2. All array-valued fields (`params`, `std_errors`, etc.) share the same order
/// as `param_names`.
///
/// `fixed_effects()` (recovering the fixed effects themselves, `α_i`/`γ_t`) is
/// intentionally not included as a field here. It is exposed as a separate method
/// instead (see `docs/spec/fe-spec.md` section 3.5 — the same pattern as IV's
/// `first_stage()`).
// `FEResult`はRust側で組み立ててPythonに返すだけの型で、Python側からの生成・引数として
// 受け取ることは想定していないため`skip_from_py_object`（`OLSResult`と同じ理由）。
//
// `Clone`を派生しない: `estimator`フィールドの`FeEstimator`（内部の`OlsEstimator`も）が
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
    /// Number of unique time periods for two-way effects (`FEOptions.time` set); `None`
    /// for one-way. Two-way requires a balanced panel, so this equals the number of
    /// observations per entity.
    #[pyo3(get)]
    pub n_periods: Option<usize>,
    /// Standard error type actually used (echoes `FEOptions.cov_type`, normalized to
    /// lowercase; e.g. "classical", "hc1", "cluster", "dk").
    #[pyo3(get)]
    pub cov_type: String,
    /// Driscoll-Kraay bandwidth actually used: the explicit `dk_bandwidth` if given, otherwise
    /// the value chosen automatically, `floor(4 * (t / 100) ^ (2 / 9))` where `t` is the number
    /// of unique time periods in `dk_time`. `None` unless `cov_type="dk"`.
    #[pyo3(get)]
    pub dk_bandwidth_used: Option<i64>,
    #[pyo3(get)]
    pub f_statistic: f64,
    #[pyo3(get)]
    pub f_p_value: f64,
    /// Numerator degrees of freedom of `f_statistic` (`None` when it is NaN).
    #[pyo3(get)]
    pub f_df_num: Option<usize>,
    /// Denominator degrees of freedom of `f_statistic` (`None` when it is NaN). May differ
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
    /// `has_intercept`、`LogitResult`/`ProbitResult`の`estimator`と同じ位置づけ）。
    estimator: FeEstimator,
}

#[pymethods]
impl FEResult {
    /// The fixed effects themselves (`α_i` for entity, `γ_t` for time), recovered
    /// post-hoc from the fitted coefficients (`α_i = ȳ_i - x̄_i'β̂`; see
    /// `docs/spec/fe-spec.md` section 3.5 and
    /// `engine::panel::fe::FeEstimator::fixed_effects`'s doc comment for the exact
    /// formula, including the two-way normalization convention).
    ///
    /// One-way: `dict[str, float]` keyed by entity id. Two-way: `dict[str, dict[str,
    /// float]]` with top-level keys `"entity"`/`"time"`.
    ///
    /// Two-way normalization: `α_i`/`γ_t` are not individually identified (adding a
    /// constant to one and subtracting it from the other leaves `α_i + γ_t`, and
    /// therefore the fitted values, unchanged). This implementation fixes the
    /// reference time period to `γ_{t_ref} = 0`, where `t_ref` is the first period in
    /// time order (the order of the values of the time column: numeric for integer and
    /// float columns, chronological for `Date`/`Datetime`, the category order for
    /// `Enum`, alphabetical for strings and `Categorical`) — a deterministic convention
    /// independent of row order. The `"time"` dictionary lists the periods in the same
    /// order. `fixest::fixef()` instead uses the time value that appears first in
    /// observation order, so numerical agreement with `fixest` for two-way effects
    /// only holds when those two choices of `t_ref` coincide for the given data.
    fn fixed_effects(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        match self.estimator.fixed_effects() {
            FixedEffects::OneWay(effects) => Ok(effects.into_pyobject(py)?.unbind()),
            FixedEffects::TwoWay { entity, time } => {
                // 時点は時間順に並べた`dict`にする（Pythonの`dict`は挿入順を保つ）。
                let time_effects = PyDict::new(py);
                for (period, effect) in time {
                    time_effects.set_item(period, effect)?;
                }
                let outer = PyDict::new(py);
                outer.set_item("entity", entity)?;
                outer.set_item("time", time_effects)?;
                Ok(outer.unbind())
            }
        }
    }
}

/// `FEOptions.cov_type`をパースし、該当する`cov_type`のときのみ`cluster`/`dk_time`を
/// 抽出したうえで`engine::panel::fe::FeCovType`を組み立てる。
///
/// `linear::common::parse_cov_type`（OLS/WLS用）を流用しない理由はモジュールdoc
/// 「`cov_type`の非対応値」参照（`hc0`が無い・`Dk`の`time`上書きがFE固有のため）。
///
/// # Errors
/// `cov_type`の文字列が既知の値のいずれでもない場合は`ValidationError`（`hc0`は非対応の
/// 専用メッセージ、それ以外の未知の値は一般的な「unknown cov_type」メッセージ）。それ以外
/// （列の抽出時に発覚する問題等）は`column_extraction`の責務で`ValidationError`。
fn parse_fe_cov_type(df: &DataFrame, options: &FEOptions) -> PyResult<(FeCovType, String)> {
    let cov_type_lower = options.cov_type.to_lowercase();
    let cov_type = match cov_type_lower.as_str() {
        "classical" => FeCovType::Classical,
        "hc1" => FeCovType::Hc1,
        "hc2" => FeCovType::Hc2,
        "hc3" => FeCovType::Hc3,
        "cluster" => {
            let groups = options
                .cluster
                .as_ref()
                .map(|col_name| extract_group_key_column(df, col_name))
                .transpose()?;
            FeCovType::Cluster { groups }
        }
        "dk" => {
            // DKの時点列は`dk_time`で明示する（必須）。2-way FEの`time`（固定効果構造）を
            // 暗黙に借用すると、意図と違う列が選ばれても気づけないため
            // （モジュールdoc「`FEOptions.time`と`FEOptions.dk_time`は別物」参照）。
            let Some(dk_time) = options.dk_time.as_ref() else {
                return Err(ValidationError::new_err(
                    "cov_type='dk' requires the `dk_time` option: the column that defines \
                     the time periods of the Driscoll-Kraay estimator (it is not taken from \
                     `time`, which only sets the two-way fixed effects)",
                ));
            };
            let time = extract_time_keys(df, dk_time)?;
            FeCovType::Dk {
                bandwidth: options.dk_bandwidth,
                time,
            }
        }
        "hc0" => {
            return Err(ValidationError::new_err(
                "cov_type='hc0' is not supported for FE (neither linearmodels nor fixest \
                 offer HC0 for panel/FE regressions); use 'hc1', 'hc2', or 'hc3' instead",
            ));
        }
        other => {
            return Err(ValidationError::new_err(format!(
                "unknown cov_type: '{other}'. Expected one of 'classical', 'hc1' through \
                 'hc3', 'cluster', or 'dk'"
            )));
        }
    };

    // 未知の`cov_type`は「unknown cov_type」を優先して報告する（上のmatchが先）。
    reject_unused_option(
        "cluster",
        options.cluster.is_some(),
        cov_type_lower == "cluster",
        "cov_type=\"cluster\"",
    )?;
    let is_dk = cov_type_lower == "dk";
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

    Ok((cov_type, cov_type_lower))
}

/// Pythonから渡された `data` / `y` / `x` / `entity` / `options` を検証し、
/// `engine::panel::fe::FeInput::from_columns`を呼び出すところまでを行う。
/// `FeEstimator::fit`の呼び出し・`FEResult`の構築は`fit`（本ファイル）が行う。
///
/// `FEOptions.time`の有無で1-way/2-wayを切り替える（`options.time`が`Some`なら
/// `FeEffects::TwoWay`、`None`なら`FeEffects::OneWay`。モジュールdoc参照）。
///
/// # Errors
/// - 列の抽出時に発覚する問題（列が存在しない、数値/文字列型にキャストできない、
///   欠損値・NaN・無限大を含む等）は`column_extraction`の責務で`ValidationError`
/// - `x`が空リストの場合は`ValidationError`（OLS/WLS/Logit/Probit/IVと同じ
///   `validate_x_non_empty`。「固定効果のみのモデル」の許容を見直し、
///   他手法と揃えた——経緯はモジュールdoc「`x`の空リストを許容しない」参照）。
///   `y`/`entity`/`time`/`x`間の重複・`x`内部の重複も同じく`validation.rs`の責務で
///   `ValidationError`
/// - `cov_type`の文字列が不正な場合は`ValidationError`（`parse_fe_cov_type`参照）
/// - それ以外（`y`/`entity`/`time`間の行数不一致等）は`engine::panel::common::PanelError`
///   から`panel_error_to_pyerr`で変換
pub(crate) fn build_fe_input(
    df: &DataFrame,
    y: String,
    x: Vec<String>,
    entity: String,
    options: &FEOptions,
) -> PyResult<(FeInput, FeEffects, FeCovType, String)> {
    // `x`が空リストであることを許容しない（モジュールdoc参照）。
    // OLS/WLS/Logit/Probit/IVと同じ`validate_x_non_empty`を呼ぶ（`.claude/rules/
    // rust-style.md`「バリデーションの責務分担」）。
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
    validate_dk_time_role(&y, &entity, options.dk_time.as_deref())?;

    // ── y/x/entity列の抽出 ─────────────────────────────────────────────
    let y_slice = extract_f64_column(df, &y)?;

    let x_slices = extract_f64_columns(df, &x)?;

    let entity_slice = extract_group_key_column(df, &entity)?;

    // ── `time`列の抽出（2-way FEを指定した場合のみ）───────────────────────
    let time_keys: Option<TimeKeys> = options
        .time
        .as_ref()
        .map(|col_name| extract_time_keys(df, col_name))
        .transpose()?;
    let effects = if time_keys.is_some() {
        FeEffects::TwoWay
    } else {
        FeEffects::OneWay
    };

    // ── cov_type固有の追加列の抽出（該当するcov_typeのときのみ）─────────────
    let (cov_type, cov_type_lower) = parse_fe_cov_type(df, options)?;

    let input = FeInput::from_columns_ordered(&y_slice, &x_slices, x, &entity_slice, time_keys, y)
        .map_err(panel_error_to_pyerr)?;

    Ok((input, effects, cov_type, cov_type_lower))
}

/// Pythonから渡された `data` / `y` / `x` / `entity` / `options` を検証し、
/// `build_fe_input`で構築した`FeInput`に対して`engine::panel::fe::FeEstimator::fit`を
/// 呼び出し、`FEResult`として返す。
///
/// `n_entities`は`FeEstimator::input().n_entities()`（`FeInput`が構築時に一度だけ作った
/// エンティティコードのユニーク数）から取得する。
///
/// `params`/`param_names`/`residuals`/`dep_var_name`/`n_obs`/`log_likelihood`は
/// `FeEstimator::estimator()`（内部で委譲した`OlsEstimator`）から取得する
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
    data: PyDataFrame,
    y: String,
    x: Vec<String>,
    entity: String,
    options: &FEOptions,
) -> PyResult<FEResult> {
    let df: DataFrame = data.into();
    let (input, effects, cov_type, cov_type_lower) = build_fe_input(&df, y, x, entity, options)?;

    let estimator = FeEstimator::fit(input, effects, cov_type, options.confidence_level)
        .map_err(panel_error_to_pyerr)?;
    let ols = estimator.estimator();

    Ok(FEResult {
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
        cov_type: cov_type_lower,
        dk_bandwidth_used: estimator.dk_bandwidth_used().map(|bw| bw as i64),
        f_statistic: estimator.f_statistic(),
        f_p_value: estimator.f_p_value(),
        f_df_num: estimator.f_df().map(|(num, _)| num),
        f_df_denom: estimator.f_df().map(|(_, denom)| denom),
        log_likelihood: ols.log_likelihood(),
        aic: estimator.aic(),
        bic: estimator.bic(),
        r_squared_within: estimator.r_squared_within(),
        r_squared_between: estimator.r_squared_between(),
        r_squared_overall: estimator.r_squared_overall(),
        estimator,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    /// `build_fe_input`のテスト全体で使う既定の`FEOptions`（`cov_type="cluster"`・
    /// `time=None`）。フィールドごとに上書きして使う。
    fn default_options() -> FEOptions {
        FEOptions::new("cluster".to_string(), 0.95, None, None, None, None)
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
            "id".to_string(),
            &options,
        )
        .unwrap();

        assert_eq!(input.nobs(), 6);
        assert_eq!(input.x_names(), &["x1".to_string()]);
        assert_eq!(input.time(), None);
        assert_eq!(effects, FeEffects::OneWay);
        assert_eq!(
            cov_type,
            FeCovType::Cluster { groups: None },
            "cluster未指定時はNone（engine側でentity列にフォールバック）"
        );
        assert_eq!(cov_type_lower, "cluster");
    }

    #[test]
    fn build_fe_input_selects_two_way_effects_when_time_is_set() {
        let df = well_formed_df();
        let mut options = default_options();
        options.time = Some("t".to_string());

        let (input, effects, ..) = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            "id".to_string(),
            &options,
        )
        .unwrap();

        assert_eq!(effects, FeEffects::TwoWay);
        let expected_time = ["1", "2", "1", "2", "1", "2"].map(str::to_string);
        assert_eq!(input.time(), Some(expected_time.as_slice()));
    }

    #[test]
    fn build_fe_input_returns_error_for_empty_x() {
        // 固定効果のみのモデル（`x=[]`）を拒否する（モジュールdoc「`x`の
        // 空リストを許容しない」参照。OLS/WLS/Logit/Probit/IVと同じ`validate_x_non_empty`）。
        let df = well_formed_df();
        let options = default_options();

        let result = build_fe_input(&df, "y".to_string(), vec![], "id".to_string(), &options);

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
            "y".to_string(),
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
            "id".to_string(),
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_fe_input_returns_error_when_x_overlaps_time() {
        let df = well_formed_df();
        let mut options = default_options();
        options.time = Some("t".to_string());

        let result = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string(), "t".to_string()],
            "id".to_string(),
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
            "id".to_string(),
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
            "id".to_string(),
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
            "id".to_string(),
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
            "id".to_string(),
            &options,
        )
        .unwrap();

        assert_eq!(
            cov_type,
            FeCovType::Cluster {
                groups: Some(vec![
                    "x".to_string(),
                    "y".to_string(),
                    "x".to_string(),
                    "y".to_string(),
                ])
            }
        );
    }

    #[test]
    fn build_fe_input_hac_uses_dk_time_when_time_is_not_set() {
        // 1-way FE + DK HAC（`time`未指定・`dk_time`のみ指定）の組み合わせ
        // （モジュールdoc「`FEOptions.time`と`FEOptions.dk_time`は別物」参照）。
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "dk".to_string();
        options.dk_time = Some("t".to_string());

        let (input, effects, cov_type, _) = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            "id".to_string(),
            &options,
        )
        .unwrap();

        assert_eq!(effects, FeEffects::OneWay);
        assert_eq!(input.time(), None); // dk_timeはFeInput.timeには渡らない
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
    fn build_fe_input_hac_uses_dk_time_independently_of_time_when_both_set() {
        // 2-way（`time`指定あり）でもDKの時点列は`dk_time`だけから決まり、`FeInput.time()`
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
        options.time = Some("t".to_string());
        options.dk_time = Some("t_fine".to_string());

        let (input, effects, cov_type, _) = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            "id".to_string(),
            &options,
        )
        .unwrap();

        assert_eq!(effects, FeEffects::TwoWay);
        let expected_time = ["1", "2", "1", "2"].map(str::to_string);
        assert_eq!(input.time(), Some(expected_time.as_slice()));
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
    fn build_fe_input_dk_time_may_equal_time_or_x_but_not_y_or_entity() {
        // 固定効果と同じ時間粒度(`dk_time == time`)や、`x`との重複は正当な使い方。
        // `y`・`entity`と同じ列だけ拒否する(`validate_dk_time_role`参照)。
        let df = well_formed_df();
        let cases = [
            (Some("t"), "t", vec!["x1"], true),
            (None, "x1", vec!["x1"], true),
            (None, "y", vec!["x1"], false),
            (None, "id", vec!["x1"], false),
        ];
        for (time, dk_time, x, ok) in cases {
            let mut options = default_options();
            options.cov_type = "dk".to_string();
            options.time = time.map(str::to_string);
            options.dk_time = Some(dk_time.to_string());

            let result = build_fe_input(
                &df,
                "y".to_string(),
                x.iter().map(|c| c.to_string()).collect(),
                "id".to_string(),
                &options,
            );

            assert_eq!(result.is_ok(), ok, "time={time:?} dk_time={dk_time}");
        }
    }

    #[test]
    fn build_fe_input_dk_requires_dk_time_even_when_time_is_set() {
        // `time`（2-wayの固定効果の時間次元）はDKの時点列に借用されない。`dk_time`なしは
        // 1-way・2-wayとも`ValidationError`（モジュールdoc参照）。
        let df = well_formed_df();
        for time in [None, Some("t".to_string())] {
            let mut options = default_options();
            options.cov_type = "dk".to_string();
            options.time = time;

            let result = build_fe_input(
                &df,
                "y".to_string(),
                vec!["x1".to_string()],
                "id".to_string(),
                &options,
            );

            assert!(result.is_err());
        }
    }
}
