//! OLSの推定オプション、およびPython（polars DataFrame + 列名 + オプション）から
//! Rust（faerの行列）への受け渡し・検証。
//!
//! 【スコープの注意】
//! ここまでが「パラメータの受け口」の実装で、この先（正規方程式ソルバー・標準誤差計算等）は
//! `engine::linear::ols`側の別issue（正規方程式ソルバー実装、標準誤差の実装、等）に委ねる。
//! 本ファイル末尾の`OlsFitInput`は、そこに渡すための最小限の橋渡し用の型で、
//! `engine`のデータ構造定義issue（「デザイン行列・目的変数のデータ構造定義」）で
//! 正式に確定するまでの暫定案。

use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::column_extraction::{extract_f64_column, extract_group_key_column};
use crate::errors::ValidationError;

/// OLSの推定オプション。
///
/// フィールドの意味・デフォルト値の根拠は`docs/planning/specs/ols-implementation-notes.md`、
/// および対応するGitHub Issue（OLS: API・オプション設計 / OLS: 標準誤差の技術仕様確定）を参照。
#[pyclass]
#[derive(Debug, Clone)]
pub struct OLSOptions {
    /// 標準誤差の種別。"classical" | "hc0" | "hc1" | "hc2" | "hc3" | "cluster"。
    /// 大文字小文字は区別しない。HACは別途対応予定（未実装）。
    #[pyo3(get, set)]
    pub cov_type: String,

    /// 定数項（切片）をengine側で自動追加するか。
    /// trueの場合、設計行列の先頭に全要素1の列を追加する。
    /// ユーザーが`x`に自分で定数列を含めた状態でtrueにすると、
    /// 多重共線性となり`ComputationError`（特異行列）になる。
    #[pyo3(get, set)]
    pub include_intercept: bool,

    /// 信頼区間の信頼水準（0, 1)の範囲。デフォルト0.95（95%信頼区間）。
    /// 「alpha」ではなくこちらの名前を使う（0.05側との混同を避けるため）。
    #[pyo3(get, set)]
    pub confidence_level: f64,

    /// cov_type="cluster"のときに使うクラスター列名。`data`内の列名を指定する
    /// （別配列としては渡さない）。cov_type≠"cluster"のときは無視される。
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
                "未知のcov_type: '{other}'。'classical', 'hc0'〜'hc3', 'cluster'のいずれかを指定してください"
            )),
        }
    }
}

/// 受け口の検証・変換を終えた後、engine側に渡すための暫定データ構造。
///
/// TODO(デザイン行列・目的変数のデータ構造定義issue): engine側のfaerベースの
/// 型が確定したら、ここは`engine::linear::ols::OlsInput`のようなengine側の型を
/// 直接組み立てる形に差し替える。
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

    let cov_type = CovType::try_from(options.cov_type.as_str()).map_err(ValidationError::new_err)?;

    if !(options.confidence_level > 0.0 && options.confidence_level < 1.0) {
        return Err(ValidationError::new_err(format!(
            "confidence_levelは(0, 1)の範囲で指定してください: {}",
            options.confidence_level
        )));
    }

    if cov_type == CovType::Cluster && options.cluster_col.is_none() {
        return Err(ValidationError::new_err(
            "cov_type='cluster'を指定する場合、options.cluster_colの指定が必要です",
        ));
    }

    if x.is_empty() {
        return Err(ValidationError::new_err("xに最低1つは列名を指定してください"));
    }

    // ── y/xの重複チェック（完全な多重共線性を早期に、分かりやすいエラーで防ぐ）──
    if x.contains(&y) {
        return Err(ValidationError::new_err(format!(
            "yに指定した列'{y}'がxにも含まれています"
        )));
    }
    {
        let mut seen = std::collections::HashSet::new();
        for name in &x {
            if !seen.insert(name) {
                return Err(ValidationError::new_err(format!(
                    "xに列'{name}'が重複して指定されています"
                )));
            }
        }
    }
    if options.include_intercept && x.iter().any(|name| name == "const") {
        return Err(ValidationError::new_err(
            "include_intercept=trueのとき、xに'const'という列名は使用できません\
             （自動追加される定数項の名前と衝突するため）",
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
                "列'{col_name}'の行数がyと一致しません（y: {n}行, {col_name}: {}行）",
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
                    "クラスター列'{col_name}'の行数がyと一致しません"
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
            "観測数不足: n={n} は k={k}（説明変数の数、定数項含む）より大きい必要があります"
        )));
    }

    let x_mat = faer::Mat::from_fn(n, k, |i, j| {
        if options.include_intercept {
            if j == 0 {
                1.0
            } else {
                x_slices[j - 1][i]
            }
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
