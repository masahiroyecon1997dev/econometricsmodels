//! Python（polars DataFrame + 列名 + オプション）からRust（faerの行列）への
//! 受け渡し・検証を行う層。
//!
//! 【スコープの注意】
//! ここまでが「パラメータの受け口」の実装で、この先（正規方程式ソルバー・標準誤差計算等）は
//! `engine`側の別issue（正規方程式ソルバー実装、標準誤差の実装、等）に委ねる。
//! 本ファイル末尾の`OlsFitInput`は、そこに渡すための最小限の橋渡し用の型で、
//! `engine`のデータ構造定義issue（「デザイン行列・目的変数のデータ構造定義」）で
//! 正式に確定するまでの暫定案。
//!
//! 【polarsのバージョン依存に関する注意】
//! この環境ではpolarsを実際にビルドして検証できていない。`DataFrame::column`の戻り値型
//! （`&Series`か`&Column`か）等、polarsのバージョンによって差異がありうる箇所がある。
//! 実際にビルドする際、pyproject.toml/Cargo.tomlで固定するpolarsのバージョンに合わせて
//! 微調整が必要な可能性がある。

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::errors::ValidationError;
use crate::options::{CovType, OLSOptions};

/// 受け口の検証・変換を終えた後、engine側に渡すための暫定データ構造。
///
/// TODO(デザイン行列・目的変数のデータ構造定義issue): engine側のfaerベースの
/// 型が確定したら、ここは`engine::OlsInput`のようなengine側の型を直接
/// 組み立てる形に差し替える。現状はengine未実装のため、engine_pybind内に
/// 暫定的に置いている。
pub struct OlsFitInput {
    /// 被説明変数 (n,)
    pub y: faer::Mat<f64>,
    /// 設計行列 (n, k)。`include_intercept=true`の場合、先頭列が定数項。
    pub x: faer::Mat<f64>,
    /// 係数名（`include_intercept=true`なら先頭が"const"）
    pub param_names: Vec<String>,
    /// 被説明変数名
    pub dep_var_name: String,
    /// クラスターID（cov_type=Clusterのときのみ Some）
    pub cluster_ids: Option<Vec<i64>>,
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

    let cov_type = CovType::try_from(options.cov_type.as_str())
        .map_err(ValidationError::new_err)?;

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

    // ── y列の抽出 ──────────────────────────────────────────────────────
    let y_slice = extract_f64_column(&df, &y)?;
    let n = y_slice.len();

    // ── x列の抽出（列ごとに検証しつつスライスを集める）────────────────────
    let mut x_slices: Vec<Vec<f64>> = Vec::with_capacity(x.len());
    for col_name in &x {
        let s = extract_f64_column(&df, col_name)?;
        if s.len() != n {
            // 通常はdf由来なので起こらないはずだが、念のため
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
            let ids = extract_i64_column(&df, col_name)?;
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
    // NOTE: polarsは列ごとに別々のバッファを持つ（columnar layout）。
    // faerのMatは1つの連続領域を期待するため、ここで各列を1回コピーして
    // 詰め直す必要がある（避けられないコピー。理由は
    // docs/planning/specs/ols-implementation-notes.md参照 相当の内容として
    // このコメントに残す）。
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

/// `data`から指定した列をf64のVecとして取り出す。
/// - 列が存在しない、数値型にキャストできない、欠損値を含む場合は`ValidationError`。
fn extract_f64_column(df: &DataFrame, name: &str) -> PyResult<Vec<f64>> {
    let series = df.column(name).map_err(|_| {
        ValidationError::new_err(format!("列'{name}'がデータに存在しません"))
    })?;

    let series = series
        .cast(&DataType::Float64)
        .map_err(|e| {
            ValidationError::new_err(format!(
                "列'{name}'を数値型(f64)に変換できません: {e}"
            ))
        })?;

    let ca = series.f64().map_err(|e| {
        ValidationError::new_err(format!("列'{name}'の変換に失敗しました: {e}"))
    })?;

    if ca.null_count() > 0 {
        return Err(ValidationError::new_err(format!(
            "列'{name}'に欠損値が{}件含まれています。欠損値は自動では扱えないため、\
             事前に補完・除外等の処理をユーザー側で行ってください",
            ca.null_count()
        )));
    }

    // rechunk: 複数チャンクに分かれている場合に単一チャンクへ統合する。
    // 既に単一チャンクの場合は実質コピーが発生しない（安価な操作）。
    let ca = ca.rechunk();

    // cont_slice: 単一チャンクかつ欠損なしの場合に、コピーなしで&[f64]を取得できる。
    // 上でrechunk・null_countチェック済みなので、ここは基本的に成功するはず。
    match ca.cont_slice() {
        Ok(slice) => Ok(slice.to_vec()), // Vec<f64>として所有権を持たせるため1回コピー
        Err(_) => {
            // 通常はここに来ないはずだが、フォールバックとしてイテレータ経由で構築
            Ok(ca.into_iter().map(|v| v.expect("null_countチェック済み")).collect())
        }
    }
}

/// クラスターID列をi64のVecとして取り出す。欠損値があればエラー。
fn extract_i64_column(df: &DataFrame, name: &str) -> PyResult<Vec<i64>> {
    let series = df.column(name).map_err(|_| {
        ValidationError::new_err(format!("クラスター列'{name}'がデータに存在しません"))
    })?;

    let series = series.cast(&DataType::Int64).map_err(|e| {
        ValidationError::new_err(format!(
            "クラスター列'{name}'を整数型(i64)に変換できません: {e}"
        ))
    })?;

    let ca = series.i64().map_err(|e| {
        ValidationError::new_err(format!("クラスター列'{name}'の変換に失敗しました: {e}"))
    })?;

    if ca.null_count() > 0 {
        return Err(ValidationError::new_err(format!(
            "クラスター列'{name}'に欠損値が含まれています"
        )));
    }

    Ok(ca
        .into_iter()
        .map(|v| v.expect("null_countチェック済み"))
        .collect())
}
