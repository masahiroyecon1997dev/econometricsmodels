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
//! 【本Issue（#159）のスコープ】`IvOptions`/`IvResult`のpyclass定義と、データ抽出・
//! バリデーション・`engine::iv::common::IvInput`構築までを担う`build_iv_input`を実装する。
//! `TwoSlsEstimator::fit`/将来の`GmmEstimator::fit`の呼び出し・`IvResult`の実際の構築・
//! `#[pymodule]`への登録はIssue #169（engine呼び出し・エラー変換）のスコープとし、
//! ここでは行わない（Logit/Probitの`build_logit_input`/`build_probit_input`と同じ前例、
//! ユーザー確認済み）。弱操作変数診断・過剰識別検定・内生性検定の各フィールドの実際の
//! 計算はさらに別issue（#163/#167/#164）で行う。

use std::collections::HashMap;

use engine::iv::common::{IvError, IvInput};
use engine::linear::ols::CovType as EngineCovType;
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use crate::column_extraction::{extract_f64_column, extract_group_key_column};
use crate::errors::{ComputationError, ValidationError, common_error_to_pyerr};
use crate::linear::common::least_squares_error_is_computation_error;
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
/// `OlsEstimator::fit`の失敗を包んだもの（Issue #157）。`ValidationError`/`ComputationError`の
/// 判定は`least_squares_error_is_computation_error`（`engine_pybind/src/linear/common.rs`）に
/// 委譲し、`least_squares_error_to_pyerr`と同じ基準を保つ（分類ロジックを重複させない）。
/// Pythonに渡すメッセージは`source.to_string()`ではなく`IvError`自身の`to_string()`
/// （「第一段階/第二段階のどの内生変数で失敗したか」という文脈を含む）を使うため、
/// `least_squares_error_to_pyerr`自体はそのまま呼ばない。
///
/// この関数は`engine_pybind`側の2SLS/GMM `fit()`接続（Issue #169）が実装されるまで
/// `#[pymodule]`等の「本番」経路からは呼び出されない。Issue #159で追加した`build_iv_input`
/// のテストがこの関数を間接的に呼ぶようになったため（`build_iv_input`が`iv_error_to_pyerr`
/// を呼ぶ）、`#[allow(dead_code)]`を使う（`#[expect]`は使わない。`#[expect]`は`cargo build`
/// （テストコードなし）では未到達で正しく発火するが、`--all-targets`/`cargo test`では
/// テスト経由で到達可能になり「期待した警告が発火しない」`unfulfilled_lint_expectations`
/// エラーになってしまうため、テストからのみ呼ばれる関数には`#[expect]`は使えない。
/// `#[allow]`はどちらの状況でも無条件に警告を抑制するため、この非対称性に対応できる。
/// `build_iv_input`/`parse_iv_cov_type`も同じ理由で`#[allow(dead_code)]`にしている）。
/// 接続issueで実際に呼び出されるようになったら、この属性ごと削除すること。
#[allow(dead_code)]
pub(crate) fn iv_error_to_pyerr(err: IvError) -> PyErr {
    let message = err.to_string();
    match err {
        IvError::Common(common) => common_error_to_pyerr(common),
        IvError::InsufficientInstruments { .. } | IvError::InvalidHacLags { .. } => {
            ValidationError::new_err(message)
        }
        IvError::FirstStageFailed { source, .. } | IvError::SecondStageFailed { source } => {
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
    /// "cluster", "kernel". Case-insensitive. Ignored when `method="2sls"`.
    #[pyo3(get, set)]
    pub weight_type: String,

    /// Number of GMM iterations (`method="gmm"` only): 2 (default) for efficient
    /// two-step GMM, 1 for one-step GMM. Ignored when `method="2sls"`.
    #[pyo3(get, set)]
    pub gmm_iterations: i64,
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
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "IvOptions(method={:?}, cov_type={:?}, include_intercept={}, \
             confidence_level={}, cluster_col={:?}, hac_lags={:?}, time_col={:?}, \
             weight_type={:?}, gmm_iterations={})",
            self.method,
            self.cov_type,
            self.include_intercept,
            self.confidence_level,
            self.cluster_col,
            self.hac_lags,
            self.time_col,
            self.weight_type,
            self.gmm_iterations,
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
/// `fit()`の戻り値本体には含めず別メソッドとして公開する設計（`iv-api-design.md`2.2節）
/// のため、`engine_pybind`側の配線はIssue #170で行う。
///
/// Not yet populated by any code in this crate (Issue #159 defines the type only;
/// `fit_iv`, which constructs and returns it, is implemented in Issue #169 for the
/// core fields above, and in Issue #163/#167/#164 for the diagnostic fields below).
// `IvResult`はRust側で組み立ててPythonに返すだけの型で、Python側からの生成・引数として
// 受け取ることは想定していないため`skip_from_py_object`（`IvOptions`の`from_py_object`とは
// 対照的、`OLSResult`/`LogitResult`と同じ理由）。
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
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
    /// `first_stage()`, which includes `x_exog`'s contribution too.
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
    /// `iv-api-design.md` 6.6節). `None` when there are no endogenous variables to
    /// test (`x_endog=[]`).
    #[pyo3(get)]
    pub wu_hausman_statistic: Option<f64>,
    #[pyo3(get)]
    pub wu_hausman_p_value: Option<f64>,
}

/// `IvOptions.cov_type`をパースし、該当する`cov_type`のときのみ`cluster_col`/`time_col`を
/// 抽出したうえで`engine::linear::ols::CovType`を組み立てる。
///
/// 2SLSの分散計算が`OlsEstimator`への委譲で構成される（`engine::iv::two_sls::
/// TwoSlsEstimator`）ため、対応するcov_typeの範囲は`engine::linear::common::
/// parse_cov_type`（OLS/WLS用）と同じだが、`OLSOptions`ではなく`IvOptions`という別の
/// 型に依存するため独立実装している（`linear/common.rs`の`parse_cov_type`のdocコメントが
/// 明記する通り、無理に共通化せず系統ごとに素直に実装する方針）。
///
/// # Errors
/// `cov_type`の文字列が既知の値のいずれでもない場合は`ValidationError`。それ以外
/// （列の抽出時に発覚する問題等）は`column_extraction`の責務で`ValidationError`。
///
/// `#[allow(dead_code)]`について: `build_iv_input`のdocコメント参照
/// （`#[expect]`ではなく`#[allow]`を使う理由）。
#[allow(dead_code)]
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

/// Pythonから渡された `data` / `y` / `x_exog` / `x_endog` / `instruments` / `options` を
/// 検証し、`engine::iv::common::IvInput::from_columns`を呼び出すところまでを行う。
/// `TwoSlsEstimator::fit`/将来の`GmmEstimator::fit`の呼び出し・`IvResult`の構築は
/// Issue #169（engine呼び出し・エラー変換）で実装する（このファイル冒頭のdocコメント
/// 「本Issueのスコープ」参照）。
///
/// 戻り値に`method`のパース済み小文字文字列（`"2sls"`/`"gmm"`のいずれか）を含めるのは、
/// `cov_type_lower`と同じ理由（Issue #169の`fit_iv`が2SLS/GMMのどちらを呼ぶか分岐する際、
/// ここでの妥当性チェックと同じ正規化ロジックを再実装せずに済ませるため。`Logit`の
/// `build_logit_input`が`method`を`EngineMethod`にパースして返す設計と同じ考え方だが、
/// IVには対応する`engine`側のenumがまだ無い——GMMの`engine`実装（Issue #160）が
/// 無いため——ので、ここでは正規化済み文字列のまま返す）。
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
///
/// `#[allow(dead_code)]`について: Issue #159時点では呼び出し元が本ファイルの
/// `#[cfg(test)] mod tests`のみで、`TwoSlsEstimator::fit`を実際に呼ぶ`fit_iv`
/// （Issue #169で実装予定の`#[pyfunction]`）がまだ無いため、`cargo build`
/// （`#[cfg(test)]`を含まないlibターゲットのビルド）からは到達不能に見え`dead_code`警告に
/// なる（Logit「`build_logit_input`、Issue #65」と同じ状況）。ここで`#[expect]`ではなく
/// `#[allow]`を使うのは、テストからは実際に呼ばれているため（`--all-targets`/`cargo test`
/// では到達可能になり、`#[expect]`だと「期待した警告が発火しない」
/// `unfulfilled_lint_expectations`エラーになってしまう）。`iv_error_to_pyerr`のdocコメント
/// 「`#[expect]`と`#[allow]`の使い分け」も参照。Issue #169でこの関数が実際に
/// `#[pymodule]`経路から呼ばれるようになった時点でこの属性・`parse_iv_cov_type`の属性
/// ともに不要になる（削除すること）。
#[allow(dead_code)]
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
