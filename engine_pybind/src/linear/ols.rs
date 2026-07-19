//! OLSの推定オプション、およびPython（polars DataFrame + 列名 + オプション）から
//! Rust（faerの行列）への受け渡し・検証。
//!
//! 【スコープの注意】
//! ここまでが「パラメータの受け口」の実装で、この先（正規方程式ソルバー・標準誤差計算等）は
//! `engine::linear::ols`側の別issue（正規方程式ソルバー実装、標準誤差の実装、等）に委ねる。
//! 本ファイル末尾の`OlsFitInput`は、そこに渡すための最小限の橋渡し用の型で、
//! `engine`のデータ構造定義issue（「デザイン行列・目的変数のデータ構造定義」）で
//! 正式に確定するまでの暫定案。
//!
//! 【言語方針】`.claude/rules/rust-style.md`「言語方針」参照。
//! 公開API（`OLSOptions`）のdocコメントと、`ValidationError`のメッセージ文字列は英語。
//! それ以外（このファイルの説明・非公開関数のdocコメント等）は日本語のまま。

use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::column_extraction::{extract_f64_column, extract_group_key_column};
use crate::errors::ValidationError;

/// Estimation options for OLS.
///
/// See `docs/planning/specs/ols-implementation-notes.md` and the corresponding GitHub
/// issues ("OLS: API and options design" / "OLS: standard error specification") for the
/// rationale behind each field's meaning and default value.
// `fit_ols`がPython側から`OLSOptions`インスタンスを引数として受け取るため、
// `FromPyObject`実装を明示的に維持する（pyo3 0.28以降、Cloneを実装する#[pyclass]の
// FromPyObject自動導出はopt-inに変更されたため）。
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct OLSOptions {
    /// Standard error type: one of "classical", "hc0", "hc1", "hc2", "hc3", "cluster".
    /// Case-insensitive. HAC support is planned separately and not yet implemented.
    #[pyo3(get, set)]
    pub cov_type: String,

    /// Whether the engine should automatically add an intercept column.
    /// When true, a column of all 1.0 is prepended to the design matrix.
    /// If the user's `x` already contains a constant column while this is true,
    /// the resulting perfect collinearity raises `ComputationError` (singular matrix).
    #[pyo3(get, set)]
    pub include_intercept: bool,

    /// Confidence level for confidence intervals, in the range (0, 1).
    /// Defaults to 0.95 (a 95% confidence interval). Named `confidence_level` rather
    /// than `alpha` to avoid confusion with the significance level (the 0.05 side).
    #[pyo3(get, set)]
    pub confidence_level: f64,

    /// Column name to use as the cluster group key when `cov_type="cluster"`.
    /// Refers to a column in `data` rather than being passed as a separate array.
    /// Ignored when `cov_type` is not "cluster".
    #[pyo3(get, set)]
    pub cluster_col: Option<String>,
}

#[pymethods]
impl OLSOptions {
    #[new]
    #[pyo3(signature = (
        cov_type = "classical".to_string(),
        include_intercept = true,
        confidence_level = 0.95,
        cluster_col = None,
    ))]
    fn new(
        cov_type: String,
        include_intercept: bool,
        confidence_level: f64,
        cluster_col: Option<String>,
    ) -> Self {
        Self {
            cov_type,
            include_intercept,
            confidence_level,
            cluster_col,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OLSOptions(cov_type={:?}, include_intercept={}, confidence_level={}, cluster_col={:?})",
            self.cov_type, self.include_intercept, self.confidence_level, self.cluster_col
        )
    }
}

/// 標準誤差の種別。文字列パースはengine側ではなくここ（境界）で行う。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovType {
    Classical,
    Hc0,
    Hc1,
    Hc2,
    Hc3,
    Cluster,
}

impl TryFrom<&str> for CovType {
    type Error = String;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "classical" | "nonrobust" => Ok(CovType::Classical),
            "hc0" => Ok(CovType::Hc0),
            "hc1" => Ok(CovType::Hc1),
            "hc2" => Ok(CovType::Hc2),
            "hc3" => Ok(CovType::Hc3),
            "cluster" => Ok(CovType::Cluster),
            other => Err(format!(
                "unknown cov_type: '{other}'. Expected one of 'classical', 'hc0' through 'hc3', or 'cluster'"
            )),
        }
    }
}

/// 受け口の検証・変換を終えた後、engine側に渡すための暫定データ構造。
///
/// TODO(Issue #14: engine_pybind engine呼び出し・エラー変換実装): `engine::linear::ols::OlsInput`/
/// `OlsEstimator`を直接呼び出す形に差し替え、この型自体を削除する。
/// `#[allow(dead_code)]`: `fit_ols`は現状ここで受け取った値を`engine`に渡さず
/// エラーで打ち切っている（Issue #14のスコープ）ため、フィールドが未使用のまま。
#[allow(dead_code)]
pub struct OlsFitInput {
    /// 被説明変数 (n,)
    pub y: faer::Mat<f64>,
    /// 設計行列 (n, k)。`include_intercept=true`の場合、先頭列が定数項。
    pub x: faer::Mat<f64>,
    /// 係数名（`include_intercept=true`なら先頭が"const"）
    pub param_names: Vec<String>,
    /// 被説明変数名
    pub dep_var_name: String,
    /// クラスターのグループキー（cov_type=Clusterのときのみ Some）。
    /// 州名・企業ID等の文字列/カテゴリカル変数を想定し、整数に限定しない。
    pub cluster_ids: Option<Vec<String>>,
    /// パース済みの標準誤差種別
    pub cov_type: CovType,
    /// 信頼水準（0, 1)
    pub confidence_level: f64,
    /// 定数項を含めたかどうか（df_model計算に使う。実行時ヒューリスティックは使わない）
    pub include_intercept: bool,
}

/// Pythonから渡された `data` / `y` / `x` / `options` を検証し、
/// engineに渡せる形（faerの行列）に変換する。
///
/// # Errors
/// 以下はすべて`ValidationError`として返す（`ComputationError`ではない。
/// ここでの失敗は全て「入力が悪い」ケースであり、統計計算の結果として
/// 発覚する問題ではないため）。
/// - `y`・`x`・`cluster_col`に指定された列が`data`に存在しない
/// - `y`・`x`・`cluster_col`の列が数値型にキャストできない
/// - `y`・`x`・`cluster_col`に欠損値（null）が含まれる
/// - `cov_type`の文字列が不正
/// - `confidence_level`が(0, 1)の範囲外
/// - `cov_type="cluster"`なのに`cluster_col`が指定されていない
/// - 観測数 n が 説明変数の数 k 以下
pub fn extract_ols_input(
    data: PyDataFrame,
    y: String,
    x: Vec<String>,
    options: &OLSOptions,
) -> PyResult<OlsFitInput> {
    let df: DataFrame = data.into();

    let cov_type =
        CovType::try_from(options.cov_type.as_str()).map_err(ValidationError::new_err)?;

    if !(options.confidence_level > 0.0 && options.confidence_level < 1.0) {
        return Err(ValidationError::new_err(format!(
            "confidence_level must be in the range (0, 1): {}",
            options.confidence_level
        )));
    }

    if cov_type == CovType::Cluster && options.cluster_col.is_none() {
        return Err(ValidationError::new_err(
            "options.cluster_col must be set when cov_type='cluster'",
        ));
    }

    if x.is_empty() {
        return Err(ValidationError::new_err(
            "x must contain at least one column name",
        ));
    }

    // ── y/xの重複チェック（完全な多重共線性を早期に、分かりやすいエラーで防ぐ）──
    if x.contains(&y) {
        return Err(ValidationError::new_err(format!(
            "the column '{y}' specified as y is also included in x"
        )));
    }
    {
        let mut seen = std::collections::HashSet::new();
        for name in &x {
            if !seen.insert(name) {
                return Err(ValidationError::new_err(format!(
                    "column '{name}' is specified more than once in x"
                )));
            }
        }
    }
    if options.include_intercept && x.iter().any(|name| name == "const") {
        return Err(ValidationError::new_err(
            "when include_intercept=true, x cannot contain a column named 'const' \
             (it collides with the automatically added intercept)",
        ));
    }

    // ── y列の抽出 ──────────────────────────────────────────────────────
    let y_slice = extract_f64_column(&df, &y)?;
    let n = y_slice.len();

    // ── x列の抽出（列ごとに検証しつつスライスを集める）────────────────────
    let mut x_slices: Vec<Vec<f64>> = Vec::with_capacity(x.len());
    for col_name in &x {
        let s = extract_f64_column(&df, col_name)?;
        if s.len() != n {
            return Err(ValidationError::new_err(format!(
                "row count of column '{col_name}' does not match y (y: {n} rows, {col_name}: {} rows)",
                s.len()
            )));
        }
        x_slices.push(s);
    }

    // ── クラスター列の抽出（指定時のみ）───────────────────────────────────
    let cluster_ids = match &options.cluster_col {
        Some(col_name) => {
            let ids = extract_group_key_column(&df, col_name)?;
            if ids.len() != n {
                return Err(ValidationError::new_err(format!(
                    "row count of cluster column '{col_name}' does not match y"
                )));
            }
            Some(ids)
        }
        None => None,
    };

    // ── 設計行列の組み立て（定数項の自動追加を含む）──────────────────────
    // NOTE: polarsは列ごとに別々のバッファを持つ（columnar layout）。faerのMatは
    // 1つの連続領域を期待するため、ここで各列を1回コピーして詰め直す必要がある。
    // 詳細は .claude/rules/rust-style.md「Python境界でのデータ受け渡し」参照。
    let k = if options.include_intercept {
        x_slices.len() + 1
    } else {
        x_slices.len()
    };

    if n <= k {
        return Err(ValidationError::new_err(format!(
            "insufficient observations: n={n} must be greater than k={k} \
             (number of independent variables, including the intercept)"
        )));
    }

    let x_mat = faer::Mat::from_fn(n, k, |i, j| {
        if options.include_intercept {
            if j == 0 { 1.0 } else { x_slices[j - 1][i] }
        } else {
            x_slices[j][i]
        }
    });

    let y_mat = faer::Mat::from_fn(n, 1, |i, _| y_slice[i]);

    let mut param_names = Vec::with_capacity(k);
    if options.include_intercept {
        param_names.push("const".to_string());
    }
    param_names.extend(x.iter().cloned());

    Ok(OlsFitInput {
        y: y_mat,
        x: x_mat,
        param_names,
        dep_var_name: y,
        cluster_ids,
        cov_type,
        confidence_level: options.confidence_level,
        include_intercept: options.include_intercept,
    })
}
