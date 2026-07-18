//! OLSの入力データ（被説明変数・設計行列）の型定義。
//!
//! `engine`はpolars/PyO3を一切知らない（`.claude/rules/rust-style.md`「責務分離」参照）。
//! `engine_pybind`はpolars DataFrameから列ごとに`Vec<f64>`を抽出するところまでを担い
//! （`column_extraction::extract_f64_column`）、それらの列を本モジュールの
//! `OlsInput::from_columns`に渡す。`faer::Mat`への組み立て（切片列の自動追加を含む）は
//! ここ（engine側）の責務とする。詳細は`docs/planning/specs/ols-api-design.md`
//! 「OLSOptions」の`include_intercept`の項を参照。

use faer::prelude::{Solve, SolveLstsq};
use faer::{Mat, Side};
use statrs::distribution::{ContinuousCDF, StudentsT};
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

/// 標準誤差の種別。文字列パース（Python文字列 → この型への変換）は`engine_pybind`側の
/// 責務（PyO3境界の関心事のため）。ここでは`OlsEstimator::fit`が計算方法を分岐するための
/// 純粋な列挙型のみを定義する。
///
/// 【スコープの注意（Issue #10時点）】`Cluster`・`Hac`はまだ未実装（対応する実装issueは
/// Issue #11がHACのみで、clusterには対応するissueが現時点で存在しない。
/// `docs/planning/specs/ols-implementation-notes.md`参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovType {
    /// 等分散前提（`σ̂²(X'X)⁻¹`）
    Classical,
    Hc0,
    Hc1,
    Hc2,
    Hc3,
}

/// OLSの被説明変数・設計行列を保持する入力データ。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
/// `from_columns`で組み立てた後は、getter経由でのみアクセスする。
#[derive(Debug)]
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
    /// # Errors
    /// `y`といずれかの`x_columns`の長さが一致しない場合は`OlsError::DimensionMismatch`を返す。
    ///
    /// # パニックについて
    /// `x_names.len() != x_columns.len()`の場合は`debug_assert!`でパニックする。これは
    /// 呼び出し側（`engine_pybind`）の実装バグでしか起こり得ない内部契約であり、
    /// 実データに起因する`ValidationError`とは性質が異なるため区別している。
    pub fn from_columns(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        include_intercept: bool,
        dep_var_name: String,
    ) -> Result<Self, OlsError> {
        debug_assert_eq!(
            x_columns.len(),
            x_names.len(),
            "x_columns and x_names must have the same length"
        );
        for col in x_columns {
            if col.len() != y.len() {
                return Err(OlsError::DimensionMismatch {
                    y_rows: y.len(),
                    x_rows: col.len(),
                });
            }
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

        Ok(Self {
            y: y_mat,
            x,
            param_names,
            dep_var_name,
        })
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

/// OLSの推定結果。適合度統計量（R²・F統計量・AIC/BIC等）は対応するissueがまだ無いため
/// 未実装（`docs/planning/specs/ols-implementation-notes.md`参照）。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
/// `fit`でのバリデーション（観測数・特異性・信頼水準）を通過した状態のみを表す。
///
/// 【スコープの注意（Issue #10時点）】`cov_type`はclassical/HC0-3まで対応。
/// cluster・HACへの分岐は対応するissue（HACはIssue #11。clusterは対応するissueが
/// 現時点で存在しない）で追加する。
#[derive(Debug)]
pub struct OlsEstimator {
    input: OlsInput,
    /// 使用した標準誤差の種別
    cov_type: CovType,
    /// 係数 (k, 1)。`input.param_names()`と対応する
    params: Mat<f64>,
    /// 残差 (n, 1) = y - Xβ̂
    residuals: Mat<f64>,
    /// 標準誤差 (k, 1)
    std_errors: Mat<f64>,
    /// t統計量 (k, 1) = params / std_errors
    t_stats: Mat<f64>,
    /// 両側p値 (k, 1)。t分布（自由度 n-k）に基づく
    p_values: Mat<f64>,
    /// 信頼区間の下限 (k, 1)
    conf_lower: Mat<f64>,
    /// 信頼区間の上限 (k, 1)
    conf_upper: Mat<f64>,
}

impl OlsEstimator {
    /// 正規方程式を列ピボットQR分解（`col_piv_qr`）で解き、OLS係数・標準誤差・
    /// t統計量・p値・信頼区間を求める。
    ///
    /// Cholesky（`X'Xβ=X'y`をXᵀXのCholesky分解で解く）ではなく列ピボットQRを採用する理由:
    /// `X'X`を明示的に作ると条件数が2乗になり数値的に不利な上、特異性検出
    /// （`.claude/rules/rust-style.md`「線形代数」が要求する`col_piv_qr`）と係数計算を
    /// 同じ分解で一度に行える。標準誤差の計算では`X'X`の逆行列が別途必要になるため
    /// （classical: `σ̂²(X'X)⁻¹`、HC0-3: `(X'X)⁻¹Ψ̂(X'X)⁻¹`）、そちらは`X'X`自体の
    /// Cholesky分解（対称正定値であることは上記の特異性検出で既に確認済み）で個別に求める。
    ///
    /// `confidence_level`は`fit`実行時に一度だけ使用し、信頼区間に固定して含める
    /// （`docs/planning/specs/ols-implementation-notes.md`「信頼区間」参照。実行時可変引数にはしない）。
    ///
    /// `cov_type`によらず、p値・信頼区間の算出にはt分布（自由度n-k）を使う。
    /// 主リファレンスのstatsmodelsはHC0-3で正規分布を既定とするが（`use_t=False`）、
    /// 本プロジェクトはt分布で統一する方針（`docs/planning/specs/ols-api-design.md`
    /// 「検定分布」、Issue #10で確認済み）。ベンチマーク生成側
    /// （`benchmark/run_statsmodels_benchmark.py`）は`use_t=True`を明示指定して合わせている。
    ///
    /// # Errors
    /// - `confidence_level`が`(0, 1)`の範囲外: `OlsError::InvalidConfidenceLevel`
    /// - 観測数`n`が`k`（定数項を含む説明変数の数）以下: `OlsError::InsufficientObservations`
    /// - 設計行列が特異（完全な多重共線性等）: `OlsError::SingularMatrix`
    pub fn fit(
        input: OlsInput,
        cov_type: CovType,
        confidence_level: f64,
    ) -> Result<Self, OlsError> {
        if !(confidence_level > 0.0 && confidence_level < 1.0) {
            return Err(OlsError::InvalidConfidenceLevel { confidence_level });
        }

        let n = input.nobs();
        let k = input.k();

        if n <= k {
            return Err(OlsError::InsufficientObservations { n, k });
        }

        let qr = input.x().col_piv_qr();
        ensure_full_rank(&qr, k)?;

        let params = qr.solve_lstsq(input.y());
        let residuals = input.y() - input.x() * &params;

        let df = n - k;
        let ssr: f64 = (0..n).map(|i| (*residuals.get(i, 0)).powi(2)).sum();
        let sigma2 = ssr / (df as f64);

        let xtx_inv = xtx_inverse(input.x(), k)?;

        let std_errors = match cov_type {
            CovType::Classical => classical_std_errors(sigma2, &xtx_inv, k),
            CovType::Hc0 => hc_std_errors(input.x(), &residuals, &xtx_inv, n, k, HcVariant::Hc0),
            CovType::Hc1 => hc_std_errors(input.x(), &residuals, &xtx_inv, n, k, HcVariant::Hc1),
            CovType::Hc2 => hc_std_errors(input.x(), &residuals, &xtx_inv, n, k, HcVariant::Hc2),
            CovType::Hc3 => hc_std_errors(input.x(), &residuals, &xtx_inv, n, k, HcVariant::Hc3),
        };

        let t_dist = StudentsT::new(0.0, 1.0, df as f64)
            .map_err(|e| OlsError::ComputationFailed(e.to_string()))?;
        let alpha = 1.0 - confidence_level;
        let t_crit = t_dist.inverse_cdf(1.0 - alpha / 2.0);

        let mut t_stats = Mat::zeros(k, 1);
        let mut p_values = Mat::zeros(k, 1);
        let mut conf_lower = Mat::zeros(k, 1);
        let mut conf_upper = Mat::zeros(k, 1);

        for j in 0..k {
            let coef = *params.get(j, 0);
            let se = *std_errors.get(j, 0);
            let t_stat = coef / se;

            *t_stats.get_mut(j, 0) = t_stat;
            *p_values.get_mut(j, 0) = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));
            *conf_lower.get_mut(j, 0) = coef - t_crit * se;
            *conf_upper.get_mut(j, 0) = coef + t_crit * se;
        }

        Ok(Self {
            input,
            cov_type,
            params,
            residuals,
            std_errors,
            t_stats,
            p_values,
            conf_lower,
            conf_upper,
        })
    }

    /// 推定に使った入力データ
    pub fn input(&self) -> &OlsInput {
        &self.input
    }

    /// 使用した標準誤差の種別
    pub fn cov_type(&self) -> CovType {
        self.cov_type
    }

    /// 係数 (k, 1)
    pub fn params(&self) -> &Mat<f64> {
        &self.params
    }

    /// 残差 (n, 1)
    pub fn residuals(&self) -> &Mat<f64> {
        &self.residuals
    }

    /// 標準誤差 (k, 1)
    pub fn std_errors(&self) -> &Mat<f64> {
        &self.std_errors
    }

    /// t統計量 (k, 1)
    pub fn t_stats(&self) -> &Mat<f64> {
        &self.t_stats
    }

    /// 両側p値 (k, 1)
    pub fn p_values(&self) -> &Mat<f64> {
        &self.p_values
    }

    /// 信頼区間の下限 (k, 1)
    pub fn conf_lower(&self) -> &Mat<f64> {
        &self.conf_lower
    }

    /// 信頼区間の上限 (k, 1)
    pub fn conf_upper(&self) -> &Mat<f64> {
        &self.conf_upper
    }
}

/// `(X'X)⁻¹`を求める。classical・HC0-3いずれの標準誤差計算でも共通して必要になる。
///
/// `X'X`は対称正定値であることが`ensure_full_rank`（Xの特異性検出）で既に保証されている
/// ため、Cholesky分解（`Llt`）で逆行列を求める。理論上ここで`LltError`は発生しないはずだが、
/// 浮動小数点演算の丸めにより境界的なケースで失敗しうるため、`SingularMatrix`として扱う。
fn xtx_inverse(x: &Mat<f64>, k: usize) -> Result<Mat<f64>, OlsError> {
    let xtx = x.transpose() * x;
    let llt = xtx.llt(Side::Lower).map_err(|_| OlsError::SingularMatrix)?;
    Ok(llt.solve(Mat::<f64>::identity(k, k)))
}

/// classical（等分散前提）標準誤差: `σ̂²(X'X)⁻¹`の対角成分の平方根。
fn classical_std_errors(sigma2: f64, xtx_inv: &Mat<f64>, k: usize) -> Mat<f64> {
    let mut std_errors = Mat::zeros(k, 1);
    for j in 0..k {
        let variance = sigma2 * (*xtx_inv.get(j, j));
        *std_errors.get_mut(j, 0) = variance.sqrt();
    }
    std_errors
}

/// HC0〜HC3ロバスト標準誤差: `(X'X)⁻¹Ψ̂(X'X)⁻¹`の対角成分の平方根。
///
/// `Ψ̂ = Σ_i w_i ε̂_i² x_i x_i'`（`w_i`はHCの種類ごとの重み）を、各行を
/// `scale_i = sqrt(w_i) * ε̂_i`でスケーリングした行列`Xw`を使って`Ψ̂ = Xw'Xw`として計算する
/// （符号は二乗で相殺されるため、`scale_i`の符号自体は`ε̂_i`のままでよい）。
/// `x_i x_i'`の外積を行ごとに手動で積み上げるより、既存の行列積を再利用できて簡潔なため。
///
/// - HC0: `w_i = 1`
/// - HC1: `w_i = n/(n-k)`（定数）
/// - HC2: `w_i = 1/(1-h_ii)`（`h_ii`はレバレッジ）
/// - HC3: `w_i = 1/(1-h_ii)²`
///
/// レバレッジ`h_ii = x_i'(X'X)⁻¹x_i`はHC2/HC3でのみ必要なため、それ以外では計算しない。
fn hc_std_errors(
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    n: usize,
    k: usize,
    variant: HcVariant,
) -> Mat<f64> {
    let leverage: Option<Vec<f64>> = match variant {
        HcVariant::Hc2 | HcVariant::Hc3 => {
            // h_ii = (X (X'X)⁻¹ X')_ii を、n×n の行列を作らずに行ごとの内積で求める。
            let xh = x * xtx_inv; // (n, k)
            Some(
                (0..n)
                    .map(|i| (0..k).map(|j| (*xh.get(i, j)) * (*x.get(i, j))).sum())
                    .collect(),
            )
        }
        HcVariant::Hc0 | HcVariant::Hc1 => None,
    };

    let hc1_correction = ((n as f64) / ((n - k) as f64)).sqrt();

    let x_scaled = Mat::from_fn(n, k, |i, j| {
        let resid = *residuals.get(i, 0);
        let scale = match variant {
            HcVariant::Hc0 => resid,
            HcVariant::Hc1 => resid * hc1_correction,
            HcVariant::Hc2 => {
                let h = leverage.as_ref().expect("Hc2はleverage計算済み")[i];
                resid / (1.0 - h).sqrt()
            }
            HcVariant::Hc3 => {
                let h = leverage.as_ref().expect("Hc3はleverage計算済み")[i];
                resid / (1.0 - h)
            }
        };
        scale * (*x.get(i, j))
    });

    let psi_hat = x_scaled.transpose() * &x_scaled;
    let var_hc = xtx_inv * &psi_hat * xtx_inv;

    let mut std_errors = Mat::zeros(k, 1);
    for j in 0..k {
        *std_errors.get_mut(j, 0) = (*var_hc.get(j, j)).sqrt();
    }
    std_errors
}

/// `hc_std_errors`の内部でのみ使う、HCの種類。`CovType`はclassicalも含む上位概念のため、
/// HC計算専用の分岐であることを型で明確にする（`CovType::Classical`が紛れ込まない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HcVariant {
    Hc0,
    Hc1,
    Hc2,
    Hc3,
}

/// 列ピボットQRの`R`の対角成分から設計行列のランク落ちを検出する。
///
/// 絶対閾値ではなく相対閾値を使う（`.claude/rules/rust-style.md`「線形代数」参照）。
/// `R`は列ピボットにより対角成分が絶対値の降順になるため、最大値
/// （`|R[0,0]|`、通常は最初の対角成分）を基準に相対的な小ささを判定する。
fn ensure_full_rank(qr: &faer::linalg::solvers::ColPivQr<f64>, k: usize) -> Result<(), OlsError> {
    let r = qr.thin_R();
    let max_abs_diag = (0..k).map(|i| (*r.get(i, i)).abs()).fold(0.0_f64, f64::max);
    let threshold = (k as f64) * f64::EPSILON * max_abs_diag;

    for i in 0..k {
        if (*r.get(i, i)).abs() <= threshold {
            return Err(OlsError::SingularMatrix);
        }
    }
    Ok(())
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
        )
        .unwrap();

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
        )
        .unwrap();

        assert_eq!(input.k(), 2);
        assert_eq!(input.param_names(), ["x1".to_string(), "x2".to_string()]);
        assert_eq!(*input.x().get(0, 0), 5.0);
        assert_eq!(*input.x().get(1, 1), 8.0);
    }

    #[test]
    fn from_columns_returns_dimension_mismatch_on_mismatched_column_length() {
        let y = vec![1.0, 2.0, 3.0];
        let x_columns = vec![vec![10.0, 20.0]]; // yより短い
        let result = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        );

        assert_eq!(
            result.unwrap_err(),
            OlsError::DimensionMismatch {
                y_rows: 3,
                x_rows: 2
            }
        );
    }

    #[test]
    #[should_panic]
    fn from_columns_panics_on_mismatched_names_arity() {
        // x_names.len() != x_columns.len()はengine_pybind側の実装バグでしか
        // 起こり得ない内部契約違反のため、Errではなくdebug_assert!でパニックする。
        let y = vec![1.0, 2.0, 3.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0]];
        let _ = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
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

    #[test]
    fn fit_recovers_known_coefficients_for_exact_fit_data() {
        // y = 1 + 2*x、ノイズなしの厳密解を持つデータ
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let x_columns = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = OlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();
        let params = estimator.params();

        assert!(
            (*params.get(0, 0) - 1.0).abs() < 1e-9,
            "const: {}",
            *params.get(0, 0)
        );
        assert!(
            (*params.get(1, 0) - 2.0).abs() < 1e-9,
            "x1: {}",
            *params.get(1, 0)
        );
    }

    #[test]
    fn fit_returns_insufficient_observations_when_n_le_k() {
        let y = vec![1.0, 2.0];
        let x_columns = vec![vec![1.0, 2.0]];
        // include_intercept=trueでk=2、n=2 (n<=k)
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = OlsEstimator::fit(input, CovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            OlsError::InsufficientObservations { n: 2, k: 2 }
        );
    }

    #[test]
    fn fit_returns_singular_matrix_for_perfectly_collinear_columns() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let x2 = vec![2.0, 4.0, 6.0, 8.0]; // x2 = 2 * x1 (完全な多重共線性)
        let input = OlsInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = OlsEstimator::fit(input, CovType::Classical, 0.95);

        assert_eq!(result.unwrap_err(), OlsError::SingularMatrix);
    }

    #[test]
    fn fit_returns_invalid_confidence_level_when_out_of_range() {
        let y = vec![1.0, 2.0, 3.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = OlsEstimator::fit(input, CovType::Classical, 1.5);

        assert_eq!(
            result.unwrap_err(),
            OlsError::InvalidConfidenceLevel {
                confidence_level: 1.5
            }
        );
    }

    /// x = [1,2,3,4,5], y = [2,4,5,4,5] の教科書的データセット。
    /// 期待値はscipy.stats（`scipy.stats.t`、`ppf`/`cdf`）で独立に計算・検算済み
    /// （手計算: b0=2.2, b1=0.6, SSR=2.4, df=3, sigma2=0.8）。
    #[test]
    fn fit_computes_classical_std_errors_t_stats_p_values_and_conf_int() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = OlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        let params = estimator.params();
        assert!((*params.get(0, 0) - 2.2).abs() < 1e-9);
        assert!((*params.get(1, 0) - 0.6).abs() < 1e-9);

        let se = estimator.std_errors();
        assert!((*se.get(0, 0) - 0.938_083_151_964_686).abs() < 1e-9);
        assert!((*se.get(1, 0) - 0.282_842_712_474_619).abs() < 1e-9);

        let t = estimator.t_stats();
        assert!((*t.get(0, 0) - 2.345_207_879_911_715).abs() < 1e-9);
        assert!((*t.get(1, 0) - 2.121_320_343_559_642_4).abs() < 1e-9);

        let p = estimator.p_values();
        assert!((*p.get(0, 0) - 0.100_743_456_085_420_12).abs() < 1e-6);
        assert!((*p.get(1, 0) - 0.124_027_062_657_554_59).abs() < 1e-6);

        let lower = estimator.conf_lower();
        let upper = estimator.conf_upper();
        assert!((*lower.get(0, 0) - (-0.785_399_261_018_909_6)).abs() < 1e-6);
        assert!((*upper.get(0, 0) - 5.185_399_261_018_91).abs() < 1e-6);
        assert!((*lower.get(1, 0) - (-0.300_131_745_291_273_4)).abs() < 1e-6);
        assert!((*upper.get(1, 0) - 1.500_131_745_291_273_2).abs() < 1e-6);
    }

    /// 同じデータセット（x=[1..5], y=[2,4,5,4,5]）でのHC0〜HC3。
    /// 期待値はstatsmodels 0.14.6で`use_t=True`を明示指定して独立に計算・検算済み
    /// （`sm.OLS(Y, X).fit(cov_type=..., use_t=True)`）。`use_t=True`が必要な理由は
    /// `docs/planning/specs/ols-api-design.md`「検定分布」、
    /// および`OlsEstimator::fit`のdocコメント参照
    /// （statsmodelsはHC0-3でuse_t=Falseが既定＝正規分布のため、素の既定値とは一致しない）。
    #[test]
    fn fit_computes_hc_std_errors_t_stats_p_values_and_conf_int() {
        // (cov_type, [se_const, se_x1], [t_const, t_x1], [p_const, p_x1],
        //  [lower_const, lower_x1], [upper_const, upper_x1])
        #[allow(clippy::type_complexity)]
        let cases: [(CovType, [f64; 2], [f64; 2], [f64; 2], [f64; 2], [f64; 2]); 4] = [
            (
                CovType::Hc0,
                [0.741_350_119_714_024_1, 0.185_472_369_909_913_61],
                [2.967_558_703_367_644, 3.234_983_196_103_162_8],
                [0.059_183_855_836_541_795, 0.048_033_568_062_853_735],
                [-0.159_306_949_405_533_25, 0.009_744_141_647_982_651],
                [4.559_306_949_405_528, 1.190_255_858_352_018_2],
            ),
            (
                CovType::Hc1,
                [0.957_078_889_120_43, 0.239_443_799_947_572_34],
                [2.298_661_087_407_152_2, 2.505_807_208_753_678_2],
                [0.105_117_351_189_905_94, 0.087_259_022_565_828_92],
                [-0.845_852_174_546_351, -0.162_017_036_466_242_44],
                [5.245_852_174_546_345, 1.362_017_036_466_243_2],
            ),
            (
                CovType::Hc2,
                [1.106_216_202_066_430_8, 0.279_795_843_939_315_45],
                [1.988_761_325_218_668_2, 2.144_420_701_724_696_3],
                [0.140_853_196_409_229_89, 0.121_345_243_297_999_42],
                [-1.320_473_665_111_291_2, -0.290_435_249_778_410_95],
                [5.720_473_665_111_285, 1.490_435_249_778_412],
            ),
            (
                CovType::Hc3,
                [1.689_236_634_744_594_6, 0.429_760_255_899_750_37],
                [1.302_363_419_517_377_2, 1.396_127_240_160_527_1],
                [0.283_757_574_453_598_06, 0.257_051_084_412_133_5],
                [-3.175_904_886_992_822, -0.767_688_938_545_941],
                [7.575_904_886_992_816_5, 1.967_688_938_545_941_7],
            ),
        ];

        for (cov_type, se, t, p, lower, upper) in cases {
            let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
            let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
            let input = OlsInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();

            let estimator = OlsEstimator::fit(input, cov_type, 0.95).unwrap();

            for j in 0..2 {
                let msg = format!("{cov_type:?}, param {j}");
                assert!(
                    (*estimator.std_errors().get(j, 0) - se[j]).abs() < 1e-6,
                    "std_errors mismatch: {msg}"
                );
                assert!(
                    (*estimator.t_stats().get(j, 0) - t[j]).abs() < 1e-6,
                    "t_stats mismatch: {msg}"
                );
                assert!(
                    (*estimator.p_values().get(j, 0) - p[j]).abs() < 1e-6,
                    "p_values mismatch: {msg}"
                );
                assert!(
                    (*estimator.conf_lower().get(j, 0) - lower[j]).abs() < 1e-6,
                    "conf_lower mismatch: {msg}"
                );
                assert!(
                    (*estimator.conf_upper().get(j, 0) - upper[j]).abs() < 1e-6,
                    "conf_upper mismatch: {msg}"
                );
            }
        }
    }
}
