//! OLSの入力データ（被説明変数・設計行列）の型定義。
//!
//! `engine`はpolars/PyO3を一切知らない（`.claude/rules/rust-style.md`「責務分離」参照）。
//! `engine_pybind`はpolars DataFrameから列ごとに`Vec<f64>`を抽出するところまでを担い
//! （`column_extraction::extract_f64_column`）、それらの列を本モジュールの
//! `OlsInput::from_columns`に渡す。`faer::Mat`への組み立て（切片列の自動追加を含む）は
//! ここ（engine側）の責務とする。詳細は`docs/planning/specs/ols-api-design.md`
//! 「OLSOptions」の`include_intercept`の項を参照。

use faer::Mat;
use thiserror::Error;

/// OLSの計算過程で発生しうるエラー。
///
/// `engine`はPyO3を知らないため、Python例外への変換は`engine_pybind`側で行う
/// （`.claude/rules/rust-style.md`「エラーハンドリング」参照）。バリアントと
/// Python例外の対応は`docs/planning/specs/ols-implementation-notes.md`の表を参照。
///
/// 【スコープの注意】欠損値（null）・`time_col`の数値キャスト失敗等、polarsの
/// 列データそのものに起因する検証は`engine_pybind::column_extraction`の責務であり、
/// ここには含めない（`engine`は`&[f64]`等、既にクリーンな値しか受け取らない前提）。
/// 正規方程式ソルバー実装等の後続issueで必要になった場合はバリアントを随時追加する。
#[derive(Debug, Error, PartialEq)]
pub enum OlsError {
    /// yとxの行数が一致しない。
    #[error("dimension mismatch: y has {y_rows} rows but x has {x_rows} rows")]
    DimensionMismatch { y_rows: usize, x_rows: usize },

    /// 観測数nが説明変数の数k（定数項を含む）以下。
    #[error(
        "insufficient observations: n={n} must be greater than k={k} \
         (number of independent variables, including the intercept)"
    )]
    InsufficientObservations { n: usize, k: usize },

    /// `cov_type=Cluster`のときのクラスター数が2未満。
    #[error("cov_type='cluster' requires at least 2 clusters, got {g}")]
    InsufficientClusters { g: usize },

    /// `confidence_level`が`(0, 1)`の範囲外。
    #[error("confidence_level must be in the range (0, 1): {confidence_level}")]
    InvalidConfidenceLevel { confidence_level: f64 },

    /// `hac_lags`が負、または観測数`n`以上。
    #[error("hac_lags must be in the range [0, n): got {hac_lags}, n={n}")]
    InvalidHacLags { hac_lags: i64, n: usize },

    /// `cov_type=Cluster`なのにクラスターのグループキーが渡されていない。
    #[error("cov_type='cluster' requires cluster identifiers to be provided")]
    MissingClusterColumn,

    /// 設計行列が特異（完全な多重共線性等）。
    #[error("design matrix is singular (perfect multicollinearity detected)")]
    SingularMatrix,

    /// 上記以外の計算過程での失敗（t分布のCDF計算等）。
    #[error("computation failed: {0}")]
    ComputationFailed(String),
}

/// OLSの被説明変数・設計行列を保持する入力データ。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
/// `from_columns`で組み立てた後は、getter経由でのみアクセスする。
pub struct OlsInput {
    /// 被説明変数 (n, 1)
    y: Mat<f64>,
    /// 設計行列 (n, k)。`include_intercept=true`の場合、先頭列が定数項（すべて1.0）
    x: Mat<f64>,
    /// 係数名（`include_intercept=true`なら先頭が"const"）。`x`の列と対応する
    param_names: Vec<String>,
    /// 被説明変数名
    dep_var_name: String,
}

impl OlsInput {
    /// 列ごとの`Vec<f64>`（`engine_pybind`がpolars DataFrameから抽出済み）から
    /// `OlsInput`を組み立てる。`include_intercept=true`の場合、設計行列の先頭列に
    /// 定数項（すべて1.0）を自動追加する。
    ///
    /// # パニックについて
    /// `y`と各`x_columns`の長さが一致しない、または`x_names.len() != x_columns.len()`の
    /// 場合は`debug_assert!`でパニックする。これらは呼び出し側（`engine_pybind`）が事前に
    /// 検証済みであることを前提とした内部契約であり、ユーザー起因の検証（`ValidationError`）は
    /// engine_pybind側の責務。次元不一致・観測数不足等、統計的に意味のある検証
    /// （`InvalidInput`等）は`OlsEstimator`のコンストラクタ（別issue）で行う。
    pub fn from_columns(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        include_intercept: bool,
        dep_var_name: String,
    ) -> Self {
        debug_assert_eq!(
            x_columns.len(),
            x_names.len(),
            "x_columns and x_names must have the same length"
        );
        for (i, col) in x_columns.iter().enumerate() {
            debug_assert_eq!(
                col.len(),
                y.len(),
                "x_columns[{i}] must have the same length as y"
            );
        }

        let n = y.len();
        let k = if include_intercept {
            x_columns.len() + 1
        } else {
            x_columns.len()
        };

        let x = Mat::from_fn(n, k, |i, j| {
            if include_intercept {
                if j == 0 { 1.0 } else { x_columns[j - 1][i] }
            } else {
                x_columns[j][i]
            }
        });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let mut param_names = Vec::with_capacity(k);
        if include_intercept {
            param_names.push("const".to_string());
        }
        param_names.extend(x_names);

        Self {
            y: y_mat,
            x,
            param_names,
            dep_var_name,
        }
    }

    pub fn y(&self) -> &Mat<f64> {
        &self.y
    }

    pub fn x(&self) -> &Mat<f64> {
        &self.x
    }

    pub fn param_names(&self) -> &[String] {
        &self.param_names
    }

    pub fn dep_var_name(&self) -> &str {
        &self.dep_var_name
    }

    /// 観測数 n
    pub fn nobs(&self) -> usize {
        self.y.nrows()
    }

    /// 説明変数の数 k（定数項を含む）
    pub fn k(&self) -> usize {
        self.x.ncols()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_columns_with_intercept_prepends_const_column() {
        let y = vec![1.0, 2.0, 3.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        );

        assert_eq!(input.nobs(), 3);
        assert_eq!(input.k(), 2);
        assert_eq!(input.param_names(), ["const".to_string(), "x1".to_string()]);
        assert_eq!(input.dep_var_name(), "y");
        assert_eq!(*input.x().get(0, 0), 1.0);
        assert_eq!(*input.x().get(1, 0), 1.0);
        assert_eq!(*input.x().get(0, 1), 10.0);
        assert_eq!(*input.x().get(2, 1), 30.0);
        assert_eq!(*input.y().get(2, 0), 3.0);
    }

    #[test]
    fn from_columns_without_intercept_omits_const_column() {
        let y = vec![1.0, 2.0];
        let x_columns = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            false,
            "y".to_string(),
        );

        assert_eq!(input.k(), 2);
        assert_eq!(input.param_names(), ["x1".to_string(), "x2".to_string()]);
        assert_eq!(*input.x().get(0, 0), 5.0);
        assert_eq!(*input.x().get(1, 1), 8.0);
    }

    #[test]
    #[should_panic]
    fn from_columns_panics_on_mismatched_column_length() {
        let y = vec![1.0, 2.0, 3.0];
        let x_columns = vec![vec![10.0, 20.0]]; // yより短い
        OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        );
    }

    #[test]
    fn ols_error_messages_are_human_readable() {
        assert_eq!(
            OlsError::DimensionMismatch {
                y_rows: 10,
                x_rows: 8
            }
            .to_string(),
            "dimension mismatch: y has 10 rows but x has 8 rows"
        );
        assert_eq!(
            OlsError::InsufficientObservations { n: 2, k: 3 }.to_string(),
            "insufficient observations: n=2 must be greater than k=3 \
             (number of independent variables, including the intercept)"
        );
        assert_eq!(
            OlsError::InsufficientClusters { g: 1 }.to_string(),
            "cov_type='cluster' requires at least 2 clusters, got 1"
        );
        assert_eq!(
            OlsError::InvalidConfidenceLevel {
                confidence_level: 1.5
            }
            .to_string(),
            "confidence_level must be in the range (0, 1): 1.5"
        );
        assert_eq!(
            OlsError::InvalidHacLags {
                hac_lags: -1,
                n: 100
            }
            .to_string(),
            "hac_lags must be in the range [0, n): got -1, n=100"
        );
        assert_eq!(
            OlsError::MissingClusterColumn.to_string(),
            "cov_type='cluster' requires cluster identifiers to be provided"
        );
        assert_eq!(
            OlsError::SingularMatrix.to_string(),
            "design matrix is singular (perfect multicollinearity detected)"
        );
        assert_eq!(
            OlsError::ComputationFailed("t-distribution CDF did not converge".to_string())
                .to_string(),
            "computation failed: t-distribution CDF did not converge"
        );
    }

    #[test]
    fn ols_error_implements_partial_eq() {
        assert_eq!(OlsError::SingularMatrix, OlsError::SingularMatrix);
        assert_ne!(
            OlsError::InsufficientClusters { g: 1 },
            OlsError::InsufficientClusters { g: 0 }
        );
    }
}
