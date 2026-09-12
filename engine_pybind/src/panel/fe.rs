//! FEの推定オプション・結果、およびPython（polars DataFrame + 列名 + オプション）から
//! `engine::panel::fe`（within変換・パネル自由度調整・`cov_type`対応・パネル固有R²）を
//! 呼び出すところまでの一連の処理（Issue #186でデータ抽出・pyclass定義、Issue #187で
//! `FeEstimator::fit`への実際の配線・`#[pymodule]`登録、Issue #188で`fixed_effects()`
//! メソッド）。
//!
//! 【責務分離】`.claude/rules/rust-style.md`「Python境界でのデータ受け渡し」参照。
//! polars DataFrameから列ごとの`Vec<f64>`/`Vec<String>`への抽出はここ（`column_extraction`
//! 経由）の責務。`faer::Mat`の組み立ては`engine`側（`FeInput`は`Mat`を組み立てない設計、
//! `engine/src/panel/fe.rs`モジュールdoc参照）に委ねる。
//!
//! 【言語方針】`.claude/rules/rust-style.md`「言語方針」参照。
//! 公開API（`FeOptions`/`FeResult`）のdocコメントと、`ValidationError`のメッセージ文字列は
//! 英語。それ以外（このファイルの説明・非公開関数のdocコメント等）は日本語のまま。
//!
//! ## 実装フェーズの分割方針（IV・Logitと同じ3段階、`engine_pybind/src/iv/CLAUDE.md`参照）
//!
//! 1. **データ抽出・pyclass定義issue（#186、完了）**: `FeOptions`/`FeResult`のpyclass定義、
//!    列抽出・バリデーション・`engine::panel::fe::FeInput`構築までを行う`build_fe_input`を
//!    実装した。
//! 2. **本Issue（#187）**: `build_fe_input`を実際に呼び出す`fit`関数を追加し、`lib.rs`に
//!    `#[pyfunction] fit_fe`を新設して`#[pymodule]`に登録する。`build_fe_input`/
//!    `parse_fe_cov_type`/`panel_error_to_pyerr`はこの時点で本番経路（`fit_fe`）から
//!    実際に呼ばれるようになるため、`#[allow(dead_code)]`はすべて削除する
//!    （`engine_pybind/src/iv/CLAUDE.md`「実装フェーズの分割方針」の#169と同じ）。
//! 3. **本Issue（#188）**: `fixed_effects()`メソッド（IVの`first_stage()`と同じ
//!    「追加結果は別メソッド」方針、`panel-api-design.md`6.6節）を追加する。`FeResult`に
//!    非公開フィールド`estimator: FeEstimator`（内部で`OlsEstimator`まで保持する）を
//!    追加し、`fixed_effects()`はそこから`FeEstimator::fixed_effects()`をオンデマンドに
//!    呼ぶだけ（IVの`IvResult.first_stage`フィールドが#159ではなく#170で追加されたのと
//!    同じ段階分割）。
//!
//! ## `FeOptions.time`と`FeOptions.time_col`は別物（`panel-api-design.md`1.1節）
//!
//! `time`（bareネーミング）は2-way FE（entity + time FE）の指定に使う: `Some`なら2-way・
//! `None`なら1-way。`time_col`（OLSの`cluster_col`/`time_col`と同じ「補助列」命名規則）は
//! Driscoll-Kraay型HAC（`cov_type="hac"`）専用の時系列順序で、`time`とは独立に指定できる
//! （ユーザー確認済み、2026-09-12）。**`time_col`が指定されていれば、2-way（`time`指定あり）
//! でも常にこちらがDK計算に優先される**（`time`未指定なら`FeCovType::Hac.time`の上書きが
//! `None`になり`FeInput.time()`——`time`から構築——にフォールバックする、`parse_fe_cov_type`
//! 参照）。1-way FE + DK HAC（`time`未指定・`time_col`のみ指定）を表現するために導入した
//! 設計（詳細な経緯・engine側の対応する変更は`engine/src/panel/fe.rs`モジュールdoc
//! 「Driscoll-Kraay型パネルHAC対応」参照）。
//!
//! ## `cov_type`の非対応値
//!
//! FEは`hc0`を**サポートしない**（linearmodels・fixestともにパネル/FE向けの`hc0`
//! オプションが存在しないため、Issue #181で`FeCovType` enumから除外済み）。OLS/WLS/IVとは
//! 異なりFEの`cov_type`文字列パースはこの1点で分岐が異なるため、`linear::common::
//! parse_cov_type`を流用せず独立実装する（`iv::common::parse_iv_cov_type`と同じ
//! 「無理に共通化しない」方針）。

use std::collections::{HashMap, HashSet};

use engine::panel::fe::{FeCovType, FeEffects, FeEstimator, FeInput, FixedEffects};
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_polars::PyDataFrame;

use super::common::panel_error_to_pyerr;
use crate::column_extraction::{extract_f64_column, extract_group_key_column};
use crate::errors::ValidationError;
use crate::linear::common::mat_to_vec;
use crate::validation::{
    RoleValue, validate_no_duplicate_roles, validate_no_duplicate_within_role,
};

/// Estimation options for FE (fixed effects panel regression).
///
/// See `docs/planning/specs/panel-api-design.md` for the rationale behind each field's
/// meaning and default value.
// module/from_py_objectの理由は`OLSOptions`と同じ（`engine_pybind/src/linear/ols.rs`参照）。
#[pyclass(from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct FeOptions {
    /// Standard error type: one of "classical", "hc1", "hc2", "hc3", "cluster", "hac".
    /// Case-insensitive. Unlike OLS/WLS/IV, "hc0" is **not** supported (neither
    /// linearmodels nor fixest offer it for panel/FE regressions).
    #[pyo3(get, set)]
    pub cov_type: String,

    /// Confidence level for confidence intervals, in the range (0, 1).
    /// Defaults to 0.95 (a 95% confidence interval).
    #[pyo3(get, set)]
    pub confidence_level: f64,

    /// Column name of the time identifier. When set, requests two-way fixed effects
    /// (entity + time); when `None` (default), one-way (entity only). Also used as the
    /// Driscoll-Kraay HAC time ordering when `cov_type="hac"`, unless `time_col` is set
    /// (see `time_col`).
    #[pyo3(get, set)]
    pub time: Option<String>,

    /// Column name to use as the cluster group key when `cov_type="cluster"`. When
    /// `None`, the `entity` argument's column is used automatically. Ignored when
    /// `cov_type` is not "cluster".
    #[pyo3(get, set)]
    pub cluster_col: Option<String>,

    /// Column name giving the time order for Driscoll-Kraay HAC, independent of `time`
    /// (`time` and `time_col` serve different purposes; see the module docstring). When
    /// set, always takes priority over `time` for the HAC computation (even with
    /// two-way effects). When `None`, falls back to `time`. Ignored when `cov_type` is
    /// not "hac".
    #[pyo3(get, set)]
    pub time_col: Option<String>,

    /// Bandwidth for Driscoll-Kraay HAC when `cov_type="hac"`. When `None`, computed
    /// automatically via `floor(4*(t/100)^(2/9))` (`t` = number of unique time periods).
    /// Ignored when `cov_type` is not "hac".
    #[pyo3(get, set)]
    pub dk_bandwidth: Option<i64>,
}

#[pymethods]
impl FeOptions {
    #[new]
    #[pyo3(signature = (
        cov_type = "cluster".to_string(),
        confidence_level = 0.95,
        time = None,
        cluster_col = None,
        time_col = None,
        dk_bandwidth = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        cov_type: String,
        confidence_level: f64,
        time: Option<String>,
        cluster_col: Option<String>,
        time_col: Option<String>,
        dk_bandwidth: Option<i64>,
    ) -> Self {
        Self {
            cov_type,
            confidence_level,
            time,
            cluster_col,
            time_col,
            dk_bandwidth,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FeOptions(cov_type={:?}, confidence_level={}, time={:?}, cluster_col={:?}, \
             time_col={:?}, dk_bandwidth={:?})",
            self.cov_type,
            self.confidence_level,
            self.time,
            self.cluster_col,
            self.time_col,
            self.dk_bandwidth
        )
    }
}

/// Estimation results for FE.
///
/// Structured data only (no `summary()`); see `docs/planning/specs/panel-api-design.md`
/// section 2. All array-valued fields (`params`, `std_errors`, etc.) share the same order
/// as `param_names`.
///
/// `fixed_effects()` (recovering the fixed effects themselves, `α_i`/`γ_t`) is
/// intentionally not included as a field here. It is exposed as a separate method
/// instead (see `panel-api-design.md` section 6.6 — the same pattern as IV's
/// `first_stage()`).
// `FeResult`はRust側で組み立ててPythonに返すだけの型で、Python側からの生成・引数として
// 受け取ることは想定していないため`skip_from_py_object`（`OLSResult`と同じ理由）。
//
// `Clone`を派生しない: `estimator`フィールドの`FeEstimator`（内部の`OlsEstimator`も）が
// `Clone`を実装していないため（`LogitResult`/`ProbitResult`と同じ理由、
// `.claude/rules/rust-style.md`「推定量構造体の設計」の通りprivateフィールドのみで、
// `Clone`を要求する既存の呼び出し元も無い）。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug)]
pub struct FeResult {
    #[pyo3(get)]
    pub params: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub t_stats: Vec<f64>,
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
    /// Number of panel entities (`panel-api-design.md` section 2.1, following the
    /// pyfixest/plm precedent).
    #[pyo3(get)]
    pub n_entities: usize,
    /// Standard error type actually used (echoes `FeOptions.cov_type`, normalized to
    /// lowercase; e.g. "classical", "hc1", "cluster", "hac").
    #[pyo3(get)]
    pub cov_type: String,
    #[pyo3(get)]
    pub f_statistic: f64,
    #[pyo3(get)]
    pub f_p_value: f64,
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
impl FeResult {
    /// The fixed effects themselves (`α_i` for entity, `γ_t` for time), recovered
    /// post-hoc from the fitted coefficients (`α_i = ȳ_i - x̄_i'β̂`; see
    /// `docs/planning/specs/panel-api-design.md` section 6.6 and
    /// `engine::panel::fe::FeEstimator::fixed_effects`'s doc comment for the exact
    /// formula, including the two-way normalization convention).
    ///
    /// One-way: `dict[str, float]` keyed by entity id. Two-way: `dict[str, dict[str,
    /// float]]` with top-level keys `"entity"`/`"time"`.
    ///
    /// Two-way normalization: `α_i`/`γ_t` are not individually identified (adding a
    /// constant to one and subtracting it from the other leaves `α_i + γ_t`, and
    /// therefore the fitted values, unchanged). This implementation fixes the
    /// reference time period to `γ_{t_ref} = 0`, where `t_ref` is the lexicographically
    /// smallest value of the time identifier — a deterministic convention independent
    /// of row order. `fixest::fixef()` instead uses the time value that appears first
    /// in observation order, so numerical agreement with `fixest` for two-way effects
    /// only holds when those two choices of `t_ref` coincide for the given data.
    fn fixed_effects(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        match self.estimator.fixed_effects() {
            FixedEffects::OneWay(effects) => Ok(effects.into_pyobject(py)?.unbind()),
            FixedEffects::TwoWay { entity, time } => {
                let mut outer = HashMap::with_capacity(2);
                outer.insert("entity", entity);
                outer.insert("time", time);
                Ok(outer.into_pyobject(py)?.unbind())
            }
        }
    }
}

/// `FeOptions.cov_type`をパースし、該当する`cov_type`のときのみ`cluster_col`/`time_col`を
/// 抽出したうえで`engine::panel::fe::FeCovType`を組み立てる。
///
/// `linear::common::parse_cov_type`（OLS/WLS用）を流用しない理由はモジュールdoc
/// 「`cov_type`の非対応値」参照（`hc0`が無い・`Hac`の`time`上書きがFE固有のため）。
///
/// # Errors
/// `cov_type`の文字列が既知の値のいずれでもない場合は`ValidationError`（`hc0`は非対応の
/// 専用メッセージ、それ以外の未知の値は一般的な「unknown cov_type」メッセージ）。それ以外
/// （列の抽出時に発覚する問題等）は`column_extraction`の責務で`ValidationError`。
fn parse_fe_cov_type(df: &DataFrame, options: &FeOptions) -> PyResult<(FeCovType, String)> {
    let cov_type_lower = options.cov_type.to_lowercase();

    let cov_type = match cov_type_lower.as_str() {
        "classical" => FeCovType::Classical,
        "hc1" => FeCovType::Hc1,
        "hc2" => FeCovType::Hc2,
        "hc3" => FeCovType::Hc3,
        "cluster" => {
            let groups = options
                .cluster_col
                .as_ref()
                .map(|col_name| extract_group_key_column(df, col_name))
                .transpose()?;
            FeCovType::Cluster { groups }
        }
        "hac" => {
            // `time_col`が優先（モジュールdoc「`FeOptions.time`と`FeOptions.time_col`は
            // 別物」参照）。`None`なら`FeCovType::Hac.time`も`None`にし、engine側で
            // `FeInput.time()`（`time`から構築）へのフォールバックに委ねる
            // （`engine::panel::fe`モジュールdoc「Driscoll-Kraay型パネルHAC対応」参照）。
            let time = options
                .time_col
                .as_ref()
                .map(|col_name| extract_group_key_column(df, col_name))
                .transpose()?;
            FeCovType::Hac {
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
                 'hc3', 'cluster', or 'hac'"
            )));
        }
    };

    Ok((cov_type, cov_type_lower))
}

/// Pythonから渡された `data` / `y` / `x` / `entity` / `options` を検証し、
/// `engine::panel::fe::FeInput::from_columns`を呼び出すところまでを行う。
/// `FeEstimator::fit`の呼び出し・`FeResult`の構築は`fit`（本ファイル）が行う。
///
/// `FeOptions.time`の有無で1-way/2-wayを切り替える（`options.time`が`Some`なら
/// `FeEffects::TwoWay`、`None`なら`FeEffects::OneWay`。モジュールdoc参照）。
///
/// # Errors
/// - 列の抽出時に発覚する問題（列が存在しない、数値/文字列型にキャストできない、
///   欠損値・NaN・無限大を含む等）は`column_extraction`の責務で`ValidationError`
/// - `y`/`entity`/`time`/`x`間の重複、`x`内部の重複は`validation.rs`の責務で
///   `ValidationError`（`x`が空リストであることは許容する——固定効果のみのモデルも
///   成立するため、`panel-api-design.md`6章の実装で確認済み、`validate_x_non_empty`は
///   呼ばない）
/// - `cov_type`の文字列が不正な場合は`ValidationError`（`parse_fe_cov_type`参照）
/// - それ以外（`y`/`entity`/`time`間の行数不一致等）は`engine::panel::common::PanelError`
///   から`panel_error_to_pyerr`で変換
pub(crate) fn build_fe_input(
    df: &DataFrame,
    y: String,
    x: Vec<String>,
    entity: String,
    options: &FeOptions,
) -> PyResult<(FeInput, FeEffects, FeCovType, String)> {
    // `x`が空リストであることを許容する（固定効果のみのモデルが成立するため、OLS等と
    // 異なり`validate_x_non_empty`は呼ばない、モジュールdoc参照）。それ以外の重複検証は
    // 共通ロジックに従う（`.claude/rules/rust-style.md`「バリデーションの責務分担」）。
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

    // ── y/x/entity列の抽出 ─────────────────────────────────────────────
    let y_slice = extract_f64_column(df, &y)?;

    let mut x_slices: Vec<Vec<f64>> = Vec::with_capacity(x.len());
    for col_name in &x {
        x_slices.push(extract_f64_column(df, col_name)?);
    }

    let entity_slice = extract_group_key_column(df, &entity)?;

    // ── `time`列の抽出（2-way FEを指定した場合のみ）───────────────────────
    let time_slice: Option<Vec<String>> = options
        .time
        .as_ref()
        .map(|col_name| extract_group_key_column(df, col_name))
        .transpose()?;
    let effects = if time_slice.is_some() {
        FeEffects::TwoWay
    } else {
        FeEffects::OneWay
    };

    // ── cov_type固有の追加列の抽出（該当するcov_typeのときのみ）─────────────
    let (cov_type, cov_type_lower) = parse_fe_cov_type(df, options)?;

    let input = FeInput::from_columns(
        &y_slice,
        &x_slices,
        x,
        &entity_slice,
        time_slice.as_deref(),
        y,
    )
    .map_err(panel_error_to_pyerr)?;

    Ok((input, effects, cov_type, cov_type_lower))
}

/// Pythonから渡された `data` / `y` / `x` / `entity` / `options` を検証し、
/// `build_fe_input`で構築した`FeInput`に対して`engine::panel::fe::FeEstimator::fit`を
/// 呼び出し、`FeResult`として返す。
///
/// `n_entities`はengine側に対応するpublicなgetterが無いため（`FeEstimator`内部の
/// privateな`count_unique`を使うのみ）、`FeInput::entity()`（`build_fe_input`が返す
/// `input`から取得可能）から独立に計算する（`engine_pybind/src/panel/CLAUDE.md`
/// 「`FeResult`のスコープ」参照）。
///
/// `params`/`param_names`/`residuals`/`dep_var_name`/`n_obs`/`log_likelihood`は
/// `FeEstimator::estimator()`（内部で委譲した`OlsEstimator`）から取得する
/// （`std_errors`/`t_stats`/`p_values`/`conf_lower`/`conf_upper`はFE自身が`cov_type`・
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
    options: &FeOptions,
) -> PyResult<FeResult> {
    let df: DataFrame = data.into();
    let (input, effects, cov_type, cov_type_lower) = build_fe_input(&df, y, x, entity, options)?;

    let n_entities = input.entity().iter().collect::<HashSet<_>>().len();

    let estimator = FeEstimator::fit(input, effects, cov_type, options.confidence_level)
        .map_err(panel_error_to_pyerr)?;
    let ols = estimator.estimator();

    Ok(FeResult {
        params: mat_to_vec(ols.params()),
        std_errors: mat_to_vec(estimator.std_errors()),
        t_stats: mat_to_vec(estimator.t_stats()),
        p_values: mat_to_vec(estimator.p_values()),
        conf_lower: mat_to_vec(estimator.conf_lower()),
        conf_upper: mat_to_vec(estimator.conf_upper()),
        param_names: ols.input().param_names().to_vec(),
        residuals: mat_to_vec(ols.residuals()),
        dep_var_name: ols.input().dep_var_name().to_string(),
        n_obs: ols.input().nobs(),
        df_resid: estimator.df_resid(),
        df_model: estimator.df_model(),
        n_entities,
        cov_type: cov_type_lower,
        f_statistic: estimator.f_statistic(),
        f_p_value: estimator.f_p_value(),
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

    /// `build_fe_input`のテスト全体で使う既定の`FeOptions`（`cov_type="cluster"`・
    /// `time=None`）。フィールドごとに上書きして使う。
    fn default_options() -> FeOptions {
        FeOptions::new("cluster".to_string(), 0.95, None, None, None, None)
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
            "cluster_col未指定時はNone（engine側でentity列にフォールバック）"
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
    fn build_fe_input_allows_empty_x() {
        // 固定効果のみのモデル（`x=[]`）を許容する（モジュールdoc参照、OLSと異なり
        // `validate_x_non_empty`を呼ばない）。
        let df = well_formed_df();
        let options = default_options();

        let (input, ..) =
            build_fe_input(&df, "y".to_string(), vec![], "id".to_string(), &options).unwrap();

        assert_eq!(input.x_names(), &[] as &[String]);
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
    fn build_fe_input_extracts_cluster_groups_when_cov_type_is_cluster_and_cluster_col_set() {
        let df = df!(
            "y" => [1.0, 2.0, 3.0, 4.0],
            "x1" => [2.0, 4.0, 1.0, 5.0],
            "id" => ["a", "a", "b", "b"],
            "state" => ["x", "y", "x", "y"],
        )
        .unwrap();
        let mut options = default_options();
        options.cluster_col = Some("state".to_string());

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
    fn build_fe_input_hac_uses_time_col_when_time_is_not_set() {
        // 1-way FE + DK HAC（`time`未指定・`time_col`のみ指定）の組み合わせ
        // （モジュールdoc「`FeOptions.time`と`FeOptions.time_col`は別物」参照）。
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "hac".to_string();
        options.time_col = Some("t".to_string());

        let (input, effects, cov_type, _) = build_fe_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            "id".to_string(),
            &options,
        )
        .unwrap();

        assert_eq!(effects, FeEffects::OneWay);
        assert_eq!(input.time(), None); // time_colはFeInput.timeには渡らない
        assert_eq!(
            cov_type,
            FeCovType::Hac {
                bandwidth: None,
                time: Some(vec![
                    "1".to_string(),
                    "2".to_string(),
                    "1".to_string(),
                    "2".to_string(),
                    "1".to_string(),
                    "2".to_string(),
                ]),
            }
        );
    }

    #[test]
    fn build_fe_input_hac_prefers_time_col_over_time_when_both_set() {
        // 2-way（`time`指定あり）でも`time_col`が優先されることを確認する
        // （モジュールdoc参照）。
        let df = df!(
            "y" => [1.0, 2.0, 3.0, 4.0],
            "x1" => [2.0, 4.0, 1.0, 5.0],
            "id" => ["a", "a", "b", "b"],
            "t" => ["1", "2", "1", "2"],
            "t_fine" => ["q1", "q2", "q1", "q2"],
        )
        .unwrap();
        let mut options = default_options();
        options.cov_type = "hac".to_string();
        options.time = Some("t".to_string());
        options.time_col = Some("t_fine".to_string());

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
            FeCovType::Hac {
                bandwidth: None,
                time: Some(vec![
                    "q1".to_string(),
                    "q2".to_string(),
                    "q1".to_string(),
                    "q2".to_string(),
                ]),
            }
        );
    }

    #[test]
    fn build_fe_input_hac_falls_back_to_time_when_time_col_is_not_set() {
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "hac".to_string();
        options.time = Some("t".to_string());

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
            FeCovType::Hac {
                bandwidth: None,
                time: None,
            },
            "time_col未指定時はNone（engine側でFeInput.time()にフォールバック）"
        );
    }
}
