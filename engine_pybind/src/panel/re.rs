//! REの推定オプション・結果、およびPython（polars DataFrame + 列名 + オプション）から
//! `engine::panel::re`（Swamy-Arora分散成分推定・準偏差変換・`cov_type`対応・ハウスマン検定）
//! を呼び出すところまでの一連の処理（Issue #200でデータ抽出・pyclass定義）。
//!
//! 【責務分離】`.claude/rules/rust-style.md`「Python境界でのデータ受け渡し」参照。
//! polars DataFrameから列ごとの`Vec<f64>`/`Vec<String>`への抽出はここ（`column_extraction`
//! 経由）の責務。`faer::Mat`の組み立ては`engine`側に委ねる。
//!
//! 【言語方針】`.claude/rules/rust-style.md`「言語方針」参照。
//! 公開API（`ReOptions`/`ReResult`）のdocコメントと、`ValidationError`のメッセージ文字列は
//! 英語。それ以外（このファイルの説明・非公開関数のdocコメント等）は日本語のまま。
//!
//! ## 実装フェーズの分割方針（FE・IV・Logitと同じ3段階、`engine_pybind/src/panel/CLAUDE.md`
//! 「実装フェーズの分割方針」参照）
//!
//! 1. **本Issue（#200）**: `ReOptions`/`ReResult`のpyclass定義、列抽出・バリデーション・
//!    `engine::panel::re::ReInput`構築までを行う`build_re_input`を実装する。この時点では
//!    `#[pymodule]`への登録・実際の`ReEstimator::fit`呼び出しは行わない
//!    （FEの#186と同じ分割）。
//! 2. **後続Issue**: `build_re_input`を実際に呼び出す`fit`関数を追加し、`lib.rs`に
//!    `#[pyfunction] fit_re`を新設して`#[pymodule]`に登録する（FEの#187相当）。
//!
//! ## `ReOptions`に`time_col`が無い理由（`FeOptions`との相違点）
//!
//! `FeOptions`は`time`（2-way FE構造の指定）と`time_col`（Driscoll-Kraay HAC専用の
//! 時系列順序、`time`とは独立に指定できる）を分離しているが、`ReOptions`にはこの分離が
//! 無く`time`のみを持つ。理由: `engine::panel::re::ReCovType::Hac`は`FeCovType::Hac`と
//! 異なり`time`オーバーライドフィールドを持たない（`engine::panel::re`モジュールdoc・
//! `engine/src/panel/CLAUDE.md`「`cov_type`対応（Issue #197）」参照）。RE自身が2-way
//! 構造を持たない（v1はentity方向のみ、7.6節）ため、FEのような「2-way FEの固定効果構造に
//! 使う時点粒度」と「HACカーネルに使う時系列粒度」を分離する必要が無い——DK HAC計算は
//! `ReInput::time()`をそのまま使う設計。`ReOptions.time`は以下2つの用途を1つのフィールドで
//! 兼ねる（`panel-api-design.md`7.3節）:
//! - `cov_type="hac"`時のDriscoll-Kraay型パネルHACの時系列順序（`None`なら
//!   `PanelError::HacRequiresTime`）
//! - ハウスマン検定用の内部FE呼び出しの1-way/2-way選択（`Some`なら2-way FE、`None`なら
//!   1-way FE。RE自身が2-wayをサポートしないこととは独立の判断、7.3節）
//!
//! ## `cov_type`の非対応値
//!
//! REも`hc0`を**サポートしない**（`ReCovType` enumから除外済み、FEと同じ理由・同じ
//! `hc0`専用エラーメッセージ）。
//!
//! ## `x`の空リストを許容しない（Issue #200、ユーザー確認済み・2026-09-20）
//!
//! FE（Issue #320）と同じ`validate_x_non_empty`を適用し、`x=[]`を拒否する。REで`x=[]`は
//! 「分散成分（ICC）のみを推定するnullモデル」として単独で意味を持つ標準的なユースケース
//! （パネル・混合モデル分析の"null model"）だが、`panel-api-design.md`にこの点の明示的な
//! 決定が無く、他手法（OLS/WLS/Logit/Probit/IV/FE post-#320）と一貫させる方針をユーザーが
//! 選択した。nullモデル・ICC推定のサポートは別Issue（#346）で検討する。

use engine::panel::re::{ReCovType, ReInput};
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use super::common::panel_error_to_pyerr;
use crate::column_extraction::{extract_f64_column, extract_group_key_column};
use crate::errors::ValidationError;
use crate::validation::{
    RoleValue, validate_no_duplicate_roles, validate_no_duplicate_within_role, validate_x_non_empty,
};

/// Estimation options for RE (random effects panel regression).
///
/// See `docs/planning/specs/panel-api-design.md` for the rationale behind each field's
/// meaning and default value.
// module/from_py_objectの理由は`OLSOptions`と同じ（`engine_pybind/src/linear/ols.rs`参照）。
#[pyclass(from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct ReOptions {
    /// Standard error type: one of "classical", "hc1", "hc2", "hc3", "cluster", "hac".
    /// Case-insensitive. Unlike OLS/WLS/IV, "hc0" is **not** supported (no reference
    /// implementation offers it for panel/RE regressions; see the module docstring).
    #[pyo3(get, set)]
    pub cov_type: String,

    /// Confidence level for confidence intervals, in the range (0, 1).
    /// Defaults to 0.95 (a 95% confidence interval).
    #[pyo3(get, set)]
    pub confidence_level: f64,

    /// Column name of the time identifier. Serves two purposes (see the module
    /// docstring): the Driscoll-Kraay HAC time ordering when `cov_type="hac"`, and the
    /// one-way/two-way choice for the internal FE regression used by the Hausman test
    /// (`Some` requests two-way FE, `None` one-way). Unlike `FeOptions`, RE has no
    /// separate `time_col` field, since it never needs to decouple these two uses.
    #[pyo3(get, set)]
    pub time: Option<String>,

    /// Column name to use as the cluster group key when `cov_type="cluster"`. When
    /// `None`, the `entity` argument's column is used automatically. Ignored when
    /// `cov_type` is not "cluster".
    #[pyo3(get, set)]
    pub cluster_col: Option<String>,

    /// Bandwidth for Driscoll-Kraay HAC when `cov_type="hac"`. When `None`, computed
    /// automatically via `floor(4*(t/100)^(2/9))` (`t` = number of unique time periods).
    /// Ignored when `cov_type` is not "hac".
    #[pyo3(get, set)]
    pub dk_bandwidth: Option<i64>,
}

#[pymethods]
impl ReOptions {
    #[new]
    #[pyo3(signature = (
        cov_type = "cluster".to_string(),
        confidence_level = 0.95,
        time = None,
        cluster_col = None,
        dk_bandwidth = None,
    ))]
    fn new(
        cov_type: String,
        confidence_level: f64,
        time: Option<String>,
        cluster_col: Option<String>,
        dk_bandwidth: Option<i64>,
    ) -> Self {
        Self {
            cov_type,
            confidence_level,
            time,
            cluster_col,
            dk_bandwidth,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ReOptions(cov_type={:?}, confidence_level={}, time={:?}, cluster_col={:?}, \
             dk_bandwidth={:?})",
            self.cov_type, self.confidence_level, self.time, self.cluster_col, self.dk_bandwidth
        )
    }
}

/// Estimation results for RE.
///
/// Structured data only (no `summary()`); see `docs/planning/specs/panel-api-design.md`
/// section 2. All array-valued fields (`params`, `std_errors`, etc.) share the same order
/// as `param_names` (`param_names[0]` is always `"const"`, since RE — unlike FE — has an
/// intercept).
///
/// The Hausman test (`hausman_statistic` / `hausman_p_value` / `hausman_df`) is computed
/// automatically inside `fit()` and included directly here (unlike FE's
/// `fixed_effects()`, RE has no separate diagnostic method; see `panel-api-design.md`
/// section 2.4). All three are `None` when the internal FE comparison fails (singleton
/// entities, zero-variance regressors after demeaning, etc.) or when the comparison is
/// otherwise not well-defined — RE's own result is still returned normally in that case.
// `ReResult`はRust側で組み立ててPythonに返すだけの型で、Python側からの生成・引数として
// 受け取ることは想定していないため`skip_from_py_object`（`OLSResult`/`FeResult`と同じ理由）。
//
// 全フィールドが即値（`Vec`/`String`/`f64`/`usize`/`Option<...>`）で、FEの
// `fixed_effects()`のようなオンデマンド計算メソッドを持たないため、`estimator`のような
// 非公開フィールドは不要（`Clone`も問題なく派生できる）。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct ReResult {
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
    /// Standard error type actually used (echoes `ReOptions.cov_type`, normalized to
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
    /// Classical Hausman test statistic comparing RE against the equivalent FE
    /// specification (`panel-api-design.md` section 7.3). Computed with classical
    /// standard errors regardless of `cov_type`. `None` if the internal FE comparison
    /// is unavailable (see the struct-level docstring).
    #[pyo3(get)]
    pub hausman_statistic: Option<f64>,
    /// p-value of `hausman_statistic` (upper-tail chi-squared probability).
    #[pyo3(get)]
    pub hausman_p_value: Option<f64>,
    /// Degrees of freedom of the Hausman test (number of compared slope coefficients,
    /// i.e. `df_model - 1`).
    #[pyo3(get)]
    pub hausman_df: Option<usize>,
}

/// `ReOptions.cov_type`をパースし、該当する`cov_type`のときのみ`cluster_col`を抽出した
/// うえで`engine::panel::re::ReCovType`を組み立てる。
///
/// `ReCovType::Hac`は`FeCovType::Hac`と異なり`time`オーバーライドフィールドを持たない
/// （モジュールdoc「`ReOptions`に`time_col`が無い理由」参照）ため、`cov_type="hac"`の
/// 分岐でも追加の列抽出は不要——HAC計算は`ReEstimator::fit`内部で`ReInput::time()`を
/// 直接使う。
///
/// # Errors
/// `cov_type`の文字列が既知の値のいずれでもない場合は`ValidationError`（`hc0`は非対応の
/// 専用メッセージ、それ以外の未知の値は一般的な「unknown cov_type」メッセージ）。それ以外
/// （`cluster_col`列の抽出時に発覚する問題等）は`column_extraction`の責務で`ValidationError`。
fn parse_re_cov_type(df: &DataFrame, options: &ReOptions) -> PyResult<(ReCovType, String)> {
    let cov_type_lower = options.cov_type.to_lowercase();

    let cov_type = match cov_type_lower.as_str() {
        "classical" => ReCovType::Classical,
        "hc1" => ReCovType::Hc1,
        "hc2" => ReCovType::Hc2,
        "hc3" => ReCovType::Hc3,
        "cluster" => {
            let groups = options
                .cluster_col
                .as_ref()
                .map(|col_name| extract_group_key_column(df, col_name))
                .transpose()?;
            ReCovType::Cluster { groups }
        }
        "hac" => ReCovType::Hac {
            bandwidth: options.dk_bandwidth,
        },
        "hc0" => {
            return Err(ValidationError::new_err(
                "cov_type='hc0' is not supported for RE (no reference implementation \
                 offers HC0 for panel/RE regressions); use 'hc1', 'hc2', or 'hc3' instead",
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
/// `engine::panel::re::ReInput::from_columns`を呼び出すところまでを行う。
/// `ReEstimator::fit`の呼び出し・`ReResult`の構築は後続issueの`fit`関数が行う
/// （モジュールdoc「実装フェーズの分割方針」参照）。
///
/// `options.time`は無条件で抽出し`ReInput`に渡す（RE自身の準偏差変換ではtimeを
/// 使わないが、内部FE呼び出し——ハウスマン検定用——の1-way/2-way選択とHAC計算に
/// 使われるため、`FeOptions.time`と同じ「渡すだけ」の扱い）。
///
/// # Errors
/// - `x`が空リストの場合は`ValidationError`（モジュールdoc「`x`の空リストを許容しない」
///   参照）。`y`/`entity`/`time`/`x`間の重複・`x`内部の重複も同じく`validation.rs`の
///   責務で`ValidationError`
/// - 列の抽出時に発覚する問題（列が存在しない、数値/文字列型にキャストできない、
///   欠損値・NaN・無限大を含む等）は`column_extraction`の責務で`ValidationError`
/// - `cov_type`の文字列が不正な場合は`ValidationError`（`parse_re_cov_type`参照）
/// - それ以外（`y`/`entity`/`time`間の行数不一致等）は`engine::panel::common::PanelError`
///   から`panel_error_to_pyerr`で変換
#[allow(
    dead_code,
    reason = "本Issue（#200）では#[pymodule]への登録・fit()の呼び出しを行わないため \
              #[cfg(test)] mod testsからのみ呼ばれる。後続issueでfit_reが#[pymodule]に \
              登録されたら削除する（engine_pybind/src/panel/CLAUDE.md「踏んだ罠」参照）"
)]
pub(crate) fn build_re_input(
    df: &DataFrame,
    y: String,
    x: Vec<String>,
    entity: String,
    options: &ReOptions,
) -> PyResult<(ReInput, ReCovType, String)> {
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

    // ── y/x/entity列の抽出 ─────────────────────────────────────────────
    let y_slice = extract_f64_column(df, &y)?;

    let mut x_slices: Vec<Vec<f64>> = Vec::with_capacity(x.len());
    for col_name in &x {
        x_slices.push(extract_f64_column(df, col_name)?);
    }

    let entity_slice = extract_group_key_column(df, &entity)?;

    // ── `time`列の抽出（内部FE呼び出し・HAC用、モジュールdoc参照）───────────
    let time_slice: Option<Vec<String>> = options
        .time
        .as_ref()
        .map(|col_name| extract_group_key_column(df, col_name))
        .transpose()?;

    // ── cov_type固有の追加列の抽出（該当するcov_typeのときのみ）─────────────
    let (cov_type, cov_type_lower) = parse_re_cov_type(df, options)?;

    let input = ReInput::from_columns(
        &y_slice,
        &x_slices,
        x,
        &entity_slice,
        time_slice.as_deref(),
        y,
    )
    .map_err(panel_error_to_pyerr)?;

    Ok((input, cov_type, cov_type_lower))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    /// `build_re_input`のテスト全体で使う既定の`ReOptions`（`cov_type="cluster"`・
    /// `time=None`）。フィールドごとに上書きして使う。
    fn default_options() -> ReOptions {
        ReOptions::new("cluster".to_string(), 0.95, None, None, None)
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

        let (input, cov_type, cov_type_lower) = build_re_input(
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
        assert_eq!(
            cov_type,
            ReCovType::Cluster { groups: None },
            "cluster_col未指定時はNone（engine側でentity列にフォールバック）"
        );
        assert_eq!(cov_type_lower, "cluster");
    }

    #[test]
    fn build_re_input_extracts_time_when_set() {
        // `time`は無条件で`ReInput`に渡る（RE自身の準偏差変換では使わないが、内部FE呼び出し
        // ・HAC用に保持される、モジュールdoc参照）。
        let df = well_formed_df();
        let mut options = default_options();
        options.time = Some("t".to_string());

        let (input, ..) = build_re_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            "id".to_string(),
            &options,
        )
        .unwrap();

        let expected_time = ["1", "2", "1", "2", "1", "2"].map(str::to_string);
        assert_eq!(input.time(), Some(expected_time.as_slice()));
    }

    #[test]
    fn build_re_input_returns_error_for_empty_x() {
        // `x=[]`を拒否する（Issue #200、モジュールdoc「`x`の空リストを許容しない」参照。
        // FE post-#320・OLS/WLS/Logit/Probit/IVと同じ`validate_x_non_empty`）。
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
    fn build_re_input_returns_error_when_x_overlaps_time() {
        let df = well_formed_df();
        let mut options = default_options();
        options.time = Some("t".to_string());

        let result = build_re_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string(), "t".to_string()],
            "id".to_string(),
            &options,
        );
        assert!(result.is_err());
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
    fn build_re_input_extracts_cluster_groups_when_cov_type_is_cluster_and_cluster_col_set() {
        let df = df!(
            "y" => [1.0, 2.0, 3.0, 4.0],
            "x1" => [2.0, 4.0, 1.0, 5.0],
            "id" => ["a", "a", "b", "b"],
            "state" => ["x", "y", "x", "y"],
        )
        .unwrap();
        let mut options = default_options();
        options.cluster_col = Some("state".to_string());

        let (_, cov_type, _) = build_re_input(
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
    fn build_re_input_hac_uses_dk_bandwidth_and_no_time_override() {
        // `ReCovType::Hac`は`FeCovType::Hac`と異なり`time`オーバーライドを持たない
        // （モジュールdoc「`ReOptions`に`time_col`が無い理由」参照）。
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "hac".to_string();
        options.dk_bandwidth = Some(2);

        let (_, cov_type, _) = build_re_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            "id".to_string(),
            &options,
        )
        .unwrap();

        assert_eq!(cov_type, ReCovType::Hac { bandwidth: Some(2) });
    }
}
