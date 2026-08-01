//! OLSの入力データ（被説明変数・設計行列）の型定義。
//!
//! `engine`はpolars/PyO3を一切知らない（`.claude/rules/rust-style.md`「責務分離」参照）。
//! `engine_pybind`はpolars DataFrameから列ごとに`Vec<f64>`を抽出するところまでを担い
//! （`column_extraction::extract_f64_column`）、それらの列を本モジュールの
//! `OlsInput::from_columns`に渡す。`faer::Mat`への組み立て（切片列の自動追加を含む）は
//! ここ（engine側）の責務とする。詳細は`docs/spec/ols-spec.md`
//! 「API引数」の`include_intercept`の項を参照。

use faer::linalg::matmul::matmul;
use faer::prelude::{Solve, SolveLstsq};
use faer::{Accum, Mat, Par, Side};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};

use super::common::LeastSquaresError;
use crate::error::CommonError;
use crate::linear_algebra::ensure_well_conditioned_symmetric_matrix;
use crate::validation::validate_cluster_groups;

/// 標準誤差の種別。文字列パース（Python文字列 → この型への変換）は`engine_pybind`側の
/// 責務（PyO3境界の関心事のため）。ここでは`OlsEstimator::fit`が計算方法を分岐するための
/// 純粋な列挙型のみを定義する。
///
/// `Hac`・`Cluster`のみ、他のバリアントと異なり追加パラメータを持つため
/// フィールド付きバリアントにしている（`fit`のシグネチャに`hac_lags`等を常に生える
/// 引数として追加するより、cov_type固有のデータをcov_type自身に持たせる方が
/// 「その cov_type 以外では無意味な引数」を作らずに済むため）。
#[derive(Debug, Clone, PartialEq)]
pub enum CovType {
    /// 等分散前提（`σ̂²(X'X)⁻¹`）
    Classical,
    Hc0,
    Hc1,
    Hc2,
    Hc3,
    /// Newey-West HAC（Bartlettカーネル）。
    Hac {
        /// ラグ数（バンド幅）。`None`なら経験則 `L = floor(4*(n/100)^(2/9))` で自動計算する
        /// （`docs/spec/ols-spec.md`「標準誤差」のHAC参照）。
        lags: Option<i64>,
        /// 時系列順序。`None`なら`OlsInput`の行順をそのまま時系列順とみなす。`Some`の場合、
        /// `OlsInput`の行と対応する長さnの配列で、この値の昇順でラグ付き自己共分散を計算する
        /// （同3.3節）。値そのものの単位・意味（期間番号・UNIX時刻等）は問わない。
        time_order: Option<Vec<f64>>,
    },
    /// クラスターロバスト標準誤差（Stata方式の小標本補正込み。常に補正を適用し、
    /// 無効化するオプションは設けない。`docs/spec/ols-spec.md`
    /// 「標準誤差」のクラスター参照）。
    Cluster {
        /// クラスターのグループキー。`OlsInput`の行と対応する長さnの配列。
        /// `None`の場合、`OlsEstimator::fit`は`CommonError::MissingClusterColumn`を返す
        /// （`hac_lags: Option<i64>`と同じ設計パターンで、値の妥当性検証を`engine`内で
        /// 行うため`Option`にしている。`engine_pybind`側で`cluster_col`未指定を
        /// 事前に弾かない）。
        groups: Option<Vec<String>>,
    },
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
    /// 定数項を含むか。R²・調整済みR²（center済み/uncenteredのSSTの選択）で必要
    has_intercept: bool,
}

impl OlsInput {
    /// 列ごとの`Vec<f64>`（`engine_pybind`がpolars DataFrameから抽出済み）から
    /// `OlsInput`を組み立てる。`include_intercept=true`の場合、設計行列の先頭列に
    /// 定数項（すべて1.0）を自動追加する。
    ///
    /// # Errors
    /// `y`といずれかの`x_columns`の長さが一致しない場合は`CommonError::DimensionMismatch`を返す。
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
    ) -> Result<Self, LeastSquaresError> {
        Self::from_columns_impl(y, x_columns, x_names, include_intercept, dep_var_name, None)
    }

    /// `from_columns`のWLS版。各観測の行（自動追加される切片列を含む）を
    /// `sqrt(weights[i])`倍してから組み立てる。この変換により、`OlsEstimator::fit`
    /// （無変更）をそのまま適用するとWLSの推定になる
    /// （`docs/planning/specs/wls-api-design.md`4.1節参照）。`weights`の全要素が1.0のときは
    /// `from_columns`と数値的に完全に同じ結果になる。
    ///
    /// `weights`はanalytic weightとして扱う（`docs/planning/specs/wls-api-design.md`0章）。
    ///
    /// # Errors
    /// - `y`といずれかの`x_columns`の長さが一致しない場合は`CommonError::DimensionMismatch`
    /// - `weights`の長さが`y`と一致しない場合は`LeastSquaresError::WeightDimensionMismatch`
    /// - `weights`に0以下（NaN含む）の値が含まれる場合は`LeastSquaresError::NonPositiveWeight`
    #[allow(clippy::too_many_arguments)]
    pub fn from_columns_weighted(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        include_intercept: bool,
        dep_var_name: String,
        weights: &[f64],
    ) -> Result<Self, LeastSquaresError> {
        if weights.len() != y.len() {
            return Err(LeastSquaresError::WeightDimensionMismatch {
                y_rows: y.len(),
                weight_rows: weights.len(),
            });
        }
        for (row, &w) in weights.iter().enumerate() {
            // `w <= 0.0`だけだとNaNを捕捉できない（NaNとの比較は常にfalse）ため、
            // `is_nan()`を別途チェックする（clippy::neg_cmp_op_on_partial_ordを避けるため
            // `!(w > 0.0)`は使わない）。NaN/無限大は`engine_pybind::column_extraction`が
            // 既に検出している前提だが、`engine`側の防御的チェックとして残す。
            if w.is_nan() || w <= 0.0 {
                return Err(LeastSquaresError::NonPositiveWeight { row, weight: w });
            }
        }

        Self::from_columns_impl(
            y,
            x_columns,
            x_names,
            include_intercept,
            dep_var_name,
            Some(weights),
        )
    }

    /// `from_columns`/`from_columns_weighted`共通の組み立てロジック。`weights`が`Some`の場合、
    /// 設計行列（自動追加される切片列を含む）・yの各行を`sqrt(weights[i])`倍する。`None`の場合は
    /// 全行`1.0`倍（`sqrt(1.0) = 1.0`）と等価で、`from_columns`はこれまでと完全に同じ結果になる。
    ///
    /// **切片列の重み付けについて**: 単純に「`x_columns`を先に重み変換してから、この関数の
    /// `weights=None`版を呼ぶ」という実装は誤り。それだと自動追加される切片列（すべて1.0）が
    /// 重み付けされないままになる。重み変換は行列組み立てそのものの中（この関数）で行う必要がある
    /// （`docs/planning/specs/wls-api-design.md`4.1節参照）。
    fn from_columns_impl(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        include_intercept: bool,
        dep_var_name: String,
        weights: Option<&[f64]>,
    ) -> Result<Self, LeastSquaresError> {
        debug_assert_eq!(
            x_columns.len(),
            x_names.len(),
            "x_columns and x_names must have the same length"
        );
        for col in x_columns {
            if col.len() != y.len() {
                return Err(CommonError::DimensionMismatch {
                    y_rows: y.len(),
                    x_rows: col.len(),
                }
                .into());
            }
        }

        let n = y.len();
        let k = if include_intercept {
            x_columns.len() + 1
        } else {
            x_columns.len()
        };

        // 行ごとの`sqrt(weight)`を事前計算する（`Mat::from_fn`のセルごとに`sqrt`を呼び直すより、
        // 行数分の計算で済む）。`weights=None`（OLS）のときは常に1.0倍で、掛けても値は変わらない
        // （`raw * 1.0`はIEEE754で丸め誤差なく`raw`と一致する）。
        let sqrt_weights: Option<Vec<f64>> =
            weights.map(|w| w.iter().map(|wi| wi.sqrt()).collect());
        let scale = |i: usize| sqrt_weights.as_ref().map_or(1.0, |sw| sw[i]);

        let x = Mat::from_fn(n, k, |i, j| {
            design_matrix_element(include_intercept, x_columns, i, j) * scale(i)
        });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i] * scale(i));

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
            has_intercept: include_intercept,
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

    /// 定数項を含むか
    pub fn has_intercept(&self) -> bool {
        self.has_intercept
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

/// 設計行列の`(i, j)`要素を返す（`has_intercept`なら先頭列が定数項1.0、それ以外は
/// `columns[j または j-1][i]`）。`from_columns_impl`（学習データの設計行列組み立て）と
/// `predict_new_data`（新規データの設計行列組み立て）が独立に同じ規約を重複実装すると、
/// 将来どちらか一方だけ規約を変更（例: 切片列の位置）した場合に静かに不整合になる
/// リスクがあるため、共有ヘルパーとして切り出している。
fn design_matrix_element(has_intercept: bool, columns: &[Vec<f64>], i: usize, j: usize) -> f64 {
    if has_intercept {
        if j == 0 { 1.0 } else { columns[j - 1][i] }
    } else {
        columns[j][i]
    }
}

/// OLSの推定結果。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
/// `fit`でのバリデーション（観測数・特異性・信頼水準）を通過した状態のみを表す。
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
    /// 決定係数（`include_intercept`に応じてcentered/uncentered TSSを切り替える）
    r_squared: f64,
    /// 自由度調整済み決定係数
    r_squared_adj: f64,
    /// F統計量。`cov_type=Classical`なら古典的F検定、それ以外（HC0-3/HAC）は
    /// `cov_params`を使ったロバストWald検定（`docs/spec/ols-spec.md`
    /// 「適合度統計量」参照）
    f_statistic: f64,
    /// F統計量のp値（F分布、自由度は`(k - k_constant, n - k)`）
    f_p_value: f64,
    /// 対数尤度（正規分布を仮定した最尤推定量ベース。`σ̂²`は`SSR/n`であり、
    /// classical標準誤差の不偏推定量`SSR/(n-k)`とは異なる点に注意）
    log_likelihood: f64,
    aic: f64,
    bic: f64,
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
    /// （`docs/spec/ols-spec.md`「API引数」参照。実行時可変引数にはしない）。
    ///
    /// `cov_type`によらず、p値・信頼区間の算出にはt分布（自由度n-k）を使う。
    /// 主リファレンスのstatsmodelsはHC0-3で正規分布を既定とするが（`use_t=False`）、
    /// 本プロジェクトはt分布で統一する方針（`docs/spec/ols-spec.md`
    /// 「標準誤差」）。ベンチマーク生成側
    /// （`benchmark/run_statsmodels_benchmark.py`）は`use_t=True`を明示指定して合わせている。
    ///
    /// F統計量も同じ方針で、`cov_type`によらず単一のWald検定の式
    /// `F = (β_slopes' Σ⁻¹ β_slopes) / q`（`Σ`は切片以外の係数に対応する`cov_params`の
    /// 部分行列、`q`はその次元）で計算する。`cov_type=Classical`のとき、この式は代数的に
    /// 古典的F検定`((SST-SSR)/q) / (SSR/df_resid)`と完全に一致する（標準的な計量経済学の
    /// 恒等式）ため、分岐を分ける必要がない。HC0-3・HACでは`cov_params`がロバストな
    /// 分散共分散行列になるため、この式がそのままロバストWald検定になる
    /// （`docs/spec/ols-spec.md`「適合度統計量」参照）。
    ///
    /// # Errors
    /// - `confidence_level`が`(0, 1)`の範囲外: `CommonError::InvalidConfidenceLevel`
    /// - 観測数`n`が`k`（定数項を含む説明変数の数）以下: `CommonError::InsufficientObservations`
    /// - 設計行列が特異（完全な多重共線性等）: `LeastSquaresError::SingularMatrix`
    pub fn fit(
        input: OlsInput,
        cov_type: CovType,
        confidence_level: f64,
    ) -> Result<Self, LeastSquaresError> {
        if !(confidence_level > 0.0 && confidence_level < 1.0) {
            return Err(CommonError::InvalidConfidenceLevel { confidence_level }.into());
        }

        let n = input.nobs();
        let k = input.k();

        if n <= k {
            return Err(CommonError::InsufficientObservations { n, k }.into());
        }

        let qr = input.x().col_piv_qr();
        ensure_full_rank(&qr, k)?;

        let params = qr.solve_lstsq(input.y());
        let residuals = input.y() - input.x() * &params;

        let df_resid = n - k;
        let ssr: f64 = (0..n).map(|i| (*residuals.get(i, 0)).powi(2)).sum();
        let sigma2 = ssr / (df_resid as f64);

        let xtx_inv = xtx_inverse(input.x(), k)?;

        // `df_inference`はt検定・信頼区間・F検定に使う自由度。通常は`df_resid`（n-k）と
        // 同じだが、`cov_type=Cluster`のときだけ`G-1`（クラスター数-1）に切り替える
        // （statsmodelsの`df_correction=True`という既定と同じ挙動。標準的な計量経済学の
        // 慣行でもある。`df_resid`自体は分散推定量`σ̂²`・調整済みR²・AIC/BIC等では
        // 引き続き`n-k`のまま使う。`docs/spec/ols-spec.md`
        // 「標準誤差」のクラスター参照）。
        let (cov_params, df_inference) = match &cov_type {
            CovType::Classical => (classical_cov_params(sigma2, &xtx_inv, k), df_resid),
            CovType::Hc0 => (
                hc_cov_params(input.x(), &residuals, &xtx_inv, n, k, HcVariant::Hc0),
                df_resid,
            ),
            CovType::Hc1 => (
                hc_cov_params(input.x(), &residuals, &xtx_inv, n, k, HcVariant::Hc1),
                df_resid,
            ),
            CovType::Hc2 => (
                hc_cov_params(input.x(), &residuals, &xtx_inv, n, k, HcVariant::Hc2),
                df_resid,
            ),
            CovType::Hc3 => (
                hc_cov_params(input.x(), &residuals, &xtx_inv, n, k, HcVariant::Hc3),
                df_resid,
            ),
            CovType::Hac { lags, time_order } => {
                let lags = resolve_hac_lags(*lags, n)?;
                let order = time_ordering(time_order.as_deref(), n);
                (
                    hac_cov_params(input.x(), &residuals, &xtx_inv, n, k, lags, &order),
                    df_resid,
                )
            }
            CovType::Cluster { groups } => {
                let groups = groups.as_ref().ok_or(CommonError::MissingClusterColumn)?;
                let n_groups = validate_cluster_groups(groups, n)?;
                let cov = cluster_cov_params(input.x(), &residuals, &xtx_inv, n, k, groups);
                (cov, n_groups - 1)
            }
        };

        let mut std_errors = Mat::zeros(k, 1);
        for j in 0..k {
            *std_errors.get_mut(j, 0) = (*cov_params.get(j, j)).sqrt();
        }

        let t_dist = StudentsT::new(0.0, 1.0, df_inference as f64)
            .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
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

        let k_constant = usize::from(input.has_intercept());
        let sst: f64 = if input.has_intercept() {
            let y_mean: f64 = (0..n).map(|i| *input.y().get(i, 0)).sum::<f64>() / (n as f64);
            (0..n)
                .map(|i| (*input.y().get(i, 0) - y_mean).powi(2))
                .sum()
        } else {
            (0..n).map(|i| (*input.y().get(i, 0)).powi(2)).sum()
        };
        let r_squared = 1.0 - ssr / sst;
        let r_squared_adj = 1.0 - ((n - k_constant) as f64 / df_resid as f64) * (1.0 - r_squared);

        let log_likelihood =
            -(n as f64 / 2.0) * ((2.0 * std::f64::consts::PI).ln() + (ssr / n as f64).ln() + 1.0);
        let aic = -2.0 * log_likelihood + 2.0 * (k as f64);
        let bic = -2.0 * log_likelihood + (n as f64).ln() * (k as f64);

        let df_model = k - k_constant;
        let (f_statistic, f_p_value) = if df_model == 0 {
            // 説明変数が定数項のみ（傾き係数が無い）モデル。検定対象が存在しないため
            // statsmodels同様NaNを返す（0除算を避ける）。
            (f64::NAN, f64::NAN)
        } else {
            wald_f_test(&params, &cov_params, k_constant, df_model, df_inference)?
        };

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
            r_squared,
            r_squared_adj,
            f_statistic,
            f_p_value,
            log_likelihood,
            aic,
            bic,
        })
    }

    /// 推定に使った入力データ
    pub fn input(&self) -> &OlsInput {
        &self.input
    }

    /// 使用した標準誤差の種別
    pub fn cov_type(&self) -> &CovType {
        &self.cov_type
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

    /// 決定係数
    pub fn r_squared(&self) -> f64 {
        self.r_squared
    }

    /// 自由度調整済み決定係数
    pub fn r_squared_adj(&self) -> f64 {
        self.r_squared_adj
    }

    /// F統計量
    pub fn f_statistic(&self) -> f64 {
        self.f_statistic
    }

    /// F統計量のp値
    pub fn f_p_value(&self) -> f64 {
        self.f_p_value
    }

    /// 対数尤度
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// 赤池情報量規準
    pub fn aic(&self) -> f64 {
        self.aic
    }

    /// ベイズ情報量規準
    pub fn bic(&self) -> f64 {
        self.bic
    }

    /// 学習データに対する予測値 `ŷ = Xβ̂`（`predict(new_data=None)`のPython APIが返す値、
    /// `docs/spec/ols-spec.md`「predict()」参照）。`fit()`のReturn本体には含めず、
    /// 必要なときに計算する別メソッドとする（Logitの`predict()`と同じ設計方針）。
    pub fn fitted_values(&self) -> Mat<f64> {
        self.input.x() * &self.params
    }
}

/// 学習済み係数`params`を使って、新規データ（`new_x_columns`）に対する予測値を計算する
/// （`predict(new_data)`のPython APIが`new_data`指定時に呼ぶ経路、`docs/spec/ols-spec.md`「predict()」）。
///
/// `OlsEstimator`のメソッドにしていない理由: `engine_pybind`側の`OLSResult`は`params`・
/// `param_names`等のフラットな値のみを保持し、`OlsEstimator`本体（`faer::Mat`を含む）は
/// `fit()`呼び出し後に破棄されるため、`OLSResult`から呼べる独立関数として提供する。
///
/// `new_x_columns`は、fit時に`x`で渡した列（`param_names`から`has_intercept`なら`"const"`を
/// 除いたもの）と同じ本数・同じ順序である必要がある。`has_intercept`が`true`の場合、定数項の
/// 列はここで自動的に先頭に付加するため`new_x_columns`に含めない
/// （`OlsInput::from_columns`の切片列自動追加と同じ挙動）。
///
/// # パニックについて
/// `new_x_columns.len()`が`params.len() - usize::from(has_intercept)`と一致しない場合は
/// `debug_assert_eq!`でパニックする。呼び出し側（`engine_pybind`）が`param_names`に基づいて
/// 必要な列だけを渡す実装契約であり、実データに起因する`ValidationError`とは性質が異なる
/// ため区別している（`OlsInput::from_columns`の`x_names.len() != x_columns.len()`と同じ
/// パターン）。
pub fn predict_new_data(
    params: &[f64],
    has_intercept: bool,
    new_x_columns: &[Vec<f64>],
) -> Vec<f64> {
    let k = params.len();
    let expected = k - usize::from(has_intercept);
    debug_assert_eq!(
        new_x_columns.len(),
        expected,
        "new_x_columns length must match the number of x columns used at fit time"
    );

    // `x`は空リストにできない（engine_pybind側でValidationErrorとして弾く、
    // `docs/spec/ols-spec.md`参照）ため、fit時にx列が1本もないケース
    // （has_intercept=falseかつexpected=0）は到達しない。したがって`new_x_columns`は
    // 常に少なくとも1列持ち、`.first()`でnを安全に取得できる。
    let n = new_x_columns.first().map_or(0, |col| col.len());

    (0..n)
        .map(|i| {
            (0..k)
                .map(|j| design_matrix_element(has_intercept, new_x_columns, i, j) * params[j])
                .sum()
        })
        .collect()
}

/// `(X'X)⁻¹`を求める。classical・HC0-3いずれの標準誤差計算でも共通して必要になる。
///
/// `X'X`は対称正定値であることが`ensure_full_rank`（Xの特異性検出）で既に保証されている
/// ため、Cholesky分解（`Llt`）で逆行列を求める。理論上ここで`LltError`は発生しないはずだが、
/// 浮動小数点演算の丸めにより境界的なケースで失敗しうるため、`SingularMatrix`として扱う。
fn xtx_inverse(x: &Mat<f64>, k: usize) -> Result<Mat<f64>, LeastSquaresError> {
    let xtx = x.transpose() * x;
    let llt = xtx
        .llt(Side::Lower)
        .map_err(|_| LeastSquaresError::SingularMatrix)?;
    Ok(llt.solve(Mat::<f64>::identity(k, k)))
}

/// classical（等分散前提）の係数分散共分散行列: `σ̂²(X'X)⁻¹`（k×k）。
fn classical_cov_params(sigma2: f64, xtx_inv: &Mat<f64>, k: usize) -> Mat<f64> {
    Mat::from_fn(k, k, |i, j| sigma2 * (*xtx_inv.get(i, j)))
}

/// HC0〜HC3ロバストな係数分散共分散行列: `(X'X)⁻¹Ψ̂(X'X)⁻¹`（k×k）。
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
fn hc_cov_params(
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
    xtx_inv * &psi_hat * xtx_inv
}

/// `hc_cov_params`の内部でのみ使う、HCの種類。`CovType`はclassicalも含む上位概念のため、
/// HC計算専用の分岐であることを型で明確にする（`CovType::Classical`が紛れ込まない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HcVariant {
    Hc0,
    Hc1,
    Hc2,
    Hc3,
}

/// `CovType::Hac`の`lags`（`Option<i64>`）を実際に使うラグ数（`usize`）に解決する。
///
/// `Some(l)`の場合は`0 <= l < n`を検証してそのまま使う。`None`の場合は経験則
/// `L = floor(4*(n/100)^(2/9))`で自動計算する（`docs/spec/ols-spec.md`
/// 「標準誤差」のHAC。EViews等でも使われる、データに依存しない決定的な式）。
fn resolve_hac_lags(lags: Option<i64>, n: usize) -> Result<usize, LeastSquaresError> {
    match lags {
        Some(l) => {
            if l < 0 || (l as usize) >= n {
                return Err(LeastSquaresError::InvalidHacLags { hac_lags: l, n });
            }
            Ok(l as usize)
        }
        None => Ok((4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize),
    }
}

/// `CovType::Hac`の`time_order`から、時系列の昇順に並べたときの行インデックス列を求める。
///
/// `None`（`time_col`未指定）の場合は`OlsInput`の行順をそのまま時系列順とみなし、恒等順序
/// `[0, 1, ..., n-1]`を返す。
///
/// `partial_cmp().unwrap()`について: `time_order`の値はNaN/無限大を含まないことが
/// `engine_pybind::column_extraction`側で既に保証されている前提（本関数は`engine`の
/// 責務境界の内側であり、クリーンな値しか受け取らない。モジュール冒頭のdocコメント参照）。
fn time_ordering(time_order: Option<&[f64]>, n: usize) -> Vec<usize> {
    match time_order {
        Some(values) => {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());
            order
        }
        None => (0..n).collect(),
    }
}

/// Newey-West HACの係数分散共分散行列: `(X'X)⁻¹Ŝ(X'X)⁻¹`（k×k）。
///
/// `Ŝ = Ŝ₀ + Σ_{l=1}^{L} w_l (Ŝ_l + Ŝ_l')`（Bartlett重み `w_l = 1 - l/(L+1)`）、
/// `Ŝ_l = Σ_{t=l+1}^{n} ε̂_t ε̂_{t-l} x_t x_{t-l}'`（`docs/spec/ols-spec.md`
/// 「標準誤差」のHAC）。`order`で指定された時系列順に並べ替えた残差・行を使ってラグ付き自己共分散を計算する。
///
/// 残差でスケールした行列`Xe`（`Xe[t,a] = ε̂_t・x_t[a]`、`order`の時系列順）を使うと、
/// `Ŝ₀ = Xe'Xe`、`Ŝ_l = Xe[l:,:]'Xe[:n-l,:]`という行列積に落とし込める（`Ŝ_l'`は転置を
/// 取るだけで再計算不要）。手書きの三重ループ（ラグ×観測×`k²`）よりfaerの行列積を使う方が
/// 大幅に高速（実測値は`docs/spec/ols-performance-notes.md`参照）。
///
/// **`Par::Seq`を明示指定する理由**: `Ŝ_l`の行列積はラグの数だけ繰り返し呼ぶことになるが、
/// 1回あたりの行列積は`k×k`という小さい出力サイズのため、faer既定の並列実行（グローバル
/// スレッドプールへのディスパッチ）のオーバーヘッドが計算本体を上回り、**三重ループより
/// 遅くなる**ことを実測済み（n=10,000, k=2で0.13倍＝約6倍の悪化）。この関数
/// 内だけ`Par::Seq`にスコープを切ることで、他のcov_type計算・将来手法のグローバル並列化
/// 設定に影響を与えずにこの罠を回避している。
fn hac_cov_params(
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    n: usize,
    k: usize,
    lags: usize,
    order: &[usize],
) -> Mat<f64> {
    let xe = Mat::<f64>::from_fn(n, k, |t, a| {
        let i = order[t];
        (*residuals.get(i, 0)) * (*x.get(i, a))
    });

    // l=0項: Ŝ₀ = Xe'Xe（HC0のΨ̂と同形）
    let mut s_hat = Mat::<f64>::zeros(k, k);
    matmul(
        s_hat.as_mut(),
        Accum::Replace,
        xe.transpose(),
        xe.as_ref(),
        1.0,
        Par::Seq,
    );

    // l=1..=lags項: w_l * (Ŝ_l + Ŝ_l')
    let mut s_l = Mat::<f64>::zeros(k, k);
    for l in 1..=lags {
        let weight = 1.0 - (l as f64) / ((lags + 1) as f64);
        let xe_top = xe.as_ref().subrows(l, n - l);
        let xe_bot = xe.as_ref().subrows(0, n - l);
        matmul(
            s_l.as_mut(),
            Accum::Replace,
            xe_top.transpose(),
            xe_bot,
            1.0,
            Par::Seq,
        );

        for a in 0..k {
            for b in 0..k {
                // (Ŝ_l + Ŝ_l')[a,b] = Ŝ_l[a,b] + Ŝ_l[b,a]
                *s_hat.get_mut(a, b) += weight * (*s_l.get(a, b) + *s_l.get(b, a));
            }
        }
    }

    xtx_inv * &s_hat * xtx_inv
}

/// クラスターロバストな係数分散共分散行列: `(X'X)⁻¹Ŝ(X'X)⁻¹ * correction`（k×k）。
///
/// `Ŝ = Σ_{g=1}^{G} S_g S_g'`、`S_g = Σ_{i∈g} ε̂_i x_i`（クラスター内の`x_i ε̂_i`の合計。
/// クラスター内の観測を先に合計してから外積を取ることで、クラスター内の相関を許容する）。
/// `correction = G/(G-1) * (n-1)/(n-k)`（Stata方式の小標本補正。常に適用する。
/// `docs/spec/ols-spec.md`「標準誤差」のクラスター参照）。
///
/// `groups`が2種類以上の値を持つこと（`G >= 2`）は呼び出し側（`validate_cluster_groups`）で
/// 検証済みの前提とする。
fn cluster_cov_params(
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    n: usize,
    k: usize,
    groups: &[String],
) -> Mat<f64> {
    // `HashMap`は反復順序がプロセスごとのハッシュシードに依存し非決定的なため、`Σ_g S_g S_g'`
    // の加算順序（延いては浮動小数点丸め誤差）が実行のたびに変わりうる。`BTreeMap`（クラスター
    // 名の辞書順）を使い、同じ入力に対して常に同じ合計順序・同じ結果になるようにする。
    let mut group_indices: std::collections::BTreeMap<&str, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        group_indices.entry(g.as_str()).or_default().push(i);
    }
    let n_groups = group_indices.len();

    let mut s_hat = Mat::<f64>::zeros(k, k);
    for indices in group_indices.values() {
        let mut s_g = vec![0.0_f64; k];
        for &i in indices {
            let e = *residuals.get(i, 0);
            for (a, s_g_a) in s_g.iter_mut().enumerate() {
                *s_g_a += e * (*x.get(i, a));
            }
        }
        for a in 0..k {
            for b in 0..k {
                *s_hat.get_mut(a, b) += s_g[a] * s_g[b];
            }
        }
    }

    let correction =
        (n_groups as f64 / (n_groups as f64 - 1.0)) * ((n as f64 - 1.0) / ((n - k) as f64));
    let cov_uncorrected = xtx_inv * &s_hat * xtx_inv;
    Mat::from_fn(k, k, |i, j| correction * (*cov_uncorrected.get(i, j)))
}

/// 列ピボットQRの`R`の対角成分から設計行列のランク落ちを検出する。
///
/// 絶対閾値ではなく相対閾値を使う（`.claude/rules/rust-style.md`「線形代数」参照）。
/// `R`は列ピボットにより対角成分が絶対値の降順になるため、最大値
/// （`|R[0,0]|`、通常は最初の対角成分）を基準に相対的な小ささを判定する。
fn ensure_full_rank(
    qr: &faer::linalg::solvers::ColPivQr<f64>,
    k: usize,
) -> Result<(), LeastSquaresError> {
    let r = qr.thin_R();
    let max_abs_diag = (0..k).map(|i| (*r.get(i, i)).abs()).fold(0.0_f64, f64::max);
    let threshold = (k as f64) * f64::EPSILON * max_abs_diag;

    for i in 0..k {
        if (*r.get(i, i)).abs() <= threshold {
            return Err(LeastSquaresError::SingularMatrix);
        }
    }
    Ok(())
}

/// 傾き係数（切片を除く`df_model`個の係数）が全てゼロという帰無仮説のロバストWald検定を行い、
/// F統計量とそのp値を返す。
///
/// `F = (β_slopes' Σ⁻¹ β_slopes) / q`（`Σ`は`cov_params`のうち傾き係数に対応する
/// `df_model × df_model`の部分行列、`q = df_model`）。`params`・`cov_params`の行/列は
/// `k_constant`が1（切片あり）なら先頭が切片（`OlsInput::from_columns`の設計行列の
/// 先頭列が定数項という規約）、0（切片なし）なら全パラメータが検定対象になる。
/// p値はF分布（自由度`(df_model, df_inference)`）の上側確率
/// （`OlsEstimator::fit`のdocコメント「F統計量も同じ方針で」参照。`cov_type=Classical`のとき
/// 古典的F検定と代数的に一致することを確認済み）。`df_inference`は通常`n-k`だが、
/// `cov_type=Cluster`のときは`G-1`になる（呼び出し元の`fit()`を参照）。
///
/// `Σ`の逆行列はCholesky分解（`Llt`）で求める。classical/HC0-3/HACでは`Σ`は
/// （`cov_params`全体の）正定値行列の主小行列であり理論上必ず正定値のため、`xtx_inverse`と
/// 同様、浮動小数点演算の丸めによる境界的な失敗に備えて`ComputationFailed`に変換している。
/// **`CovType::Cluster`は例外**: クラスターロバスト共分散`Ŝ = Σ_g S_g S_g'`はG個の
/// ランク1行列の和のため`rank(Ŝ) ≤ G`（クラスター数）であり、傾き係数の数`q`がGを超える
/// と`Σ`は構造的に（丸め誤差ではなく）特異になる。
///
/// `Σ`が数値的にほぼ特異（上記のクラスターの構造的特異性に加え、変数間のスケールが
/// 極端に異なる設計行列等で、傾き係数の同時共分散部分行列の条件数が倍精度の限界を
/// 超える場合を含む）だと、Cholesky分解自体は（非ピボットのため）失敗せずに数値的に
/// 無意味なF統計量（桁違いに巨大な値等）を黙って返してしまうことがある。
/// そのため`Llt`分解の**前**に`ensure_well_conditioned_symmetric_matrix`（`crate::
/// linear_algebra`、固有値分解ベースの相対閾値判定。系統をまたいで共有する純粋な
/// 線形代数ユーティリティ、`.claude/rules/rust-style.md`「全手法で共有するロジック」
/// 参照。nonlinear系統の`observed_information_cov_params`等でも同じ理由で使われている）
/// を呼び、`ComputationFailed`で止める。
///
/// この事前チェックにより、**`CovType::Cluster`のG<qによる構造的特異性も、実際には
/// 下の`Llt`分解に到達する前に`ensure_well_conditioned_symmetric_matrix`側で先に検出
/// される**（`cargo llvm-cov`で確認: `Llt`失敗の`map_err`分岐は0ヒット）。`Llt`分解自体の
/// `map_err`は、両方のチェックをすり抜けるごく僅かな境界ケースに備えた防御的な
/// フォールバックとして残している。
fn wald_f_test(
    params: &Mat<f64>,
    cov_params: &Mat<f64>,
    k_constant: usize,
    df_model: usize,
    df_inference: usize,
) -> Result<(f64, f64), LeastSquaresError> {
    let beta_slopes = Mat::from_fn(df_model, 1, |i, _| *params.get(i + k_constant, 0));
    let v_slopes = Mat::from_fn(df_model, df_model, |i, j| {
        *cov_params.get(i + k_constant, j + k_constant)
    });

    ensure_well_conditioned_symmetric_matrix(
        &v_slopes,
        df_model,
        "coefficient covariance submatrix for the F-test",
    )?;

    let llt = v_slopes.llt(Side::Lower).map_err(|_| {
        CommonError::ComputationFailed(
            "failed to invert coefficient covariance submatrix for the F-test".to_string(),
        )
    })?;
    let v_slopes_inv_beta = llt.solve(&beta_slopes);

    let wald: f64 = (0..df_model)
        .map(|i| (*beta_slopes.get(i, 0)) * (*v_slopes_inv_beta.get(i, 0)))
        .sum();
    let f_statistic = wald / (df_model as f64);

    let f_dist = FisherSnedecor::new(df_model as f64, df_inference as f64)
        .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
    let f_p_value = 1.0 - f_dist.cdf(f_statistic);

    Ok((f_statistic, f_p_value))
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
            LeastSquaresError::Common(CommonError::DimensionMismatch {
                y_rows: 3,
                x_rows: 2
            })
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
    fn from_columns_weighted_with_all_ones_matches_from_columns() {
        // 重みが全て1のとき、from_columns_weightedはfrom_columnsと数値的に完全一致するはず
        // （from_columns_impl内の`scale(i) = 1.0`分岐、docs/planning/specs/wls-api-design.md
        // 4.1節の構造的保証）。
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let weights = vec![1.0; 5];

        let weighted = OlsInput::from_columns_weighted(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            &weights,
        )
        .unwrap();
        let plain = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        assert_eq!(*weighted.y().get(0, 0), *plain.y().get(0, 0));
        for i in 0..5 {
            for j in 0..2 {
                assert_eq!(*weighted.x().get(i, j), *plain.x().get(i, j));
            }
        }
    }

    #[test]
    fn from_columns_weighted_scales_intercept_column_too() {
        // 切片列（すべて1.0）も重み変換の対象であることを確認する回帰テスト
        // （wls-api-design.md 4.1節「切片列の重み付け」で明記した誤りやすいポイント）。
        let y = vec![1.0, 2.0];
        let x_columns = vec![vec![10.0, 20.0]];
        let weights = vec![4.0, 9.0]; // sqrt(4)=2, sqrt(9)=3

        let input = OlsInput::from_columns_weighted(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            &weights,
        )
        .unwrap();

        assert_eq!(*input.x().get(0, 0), 2.0); // 1.0 * sqrt(4)
        assert_eq!(*input.x().get(1, 0), 3.0); // 1.0 * sqrt(9)
        assert_eq!(*input.x().get(0, 1), 20.0); // 10.0 * sqrt(4)
        assert_eq!(*input.x().get(1, 1), 60.0); // 20.0 * sqrt(9)
        assert_eq!(*input.y().get(0, 0), 2.0); // 1.0 * sqrt(4)
        assert_eq!(*input.y().get(1, 0), 6.0); // 2.0 * sqrt(9)
    }

    #[test]
    fn from_columns_weighted_returns_weight_dimension_mismatch() {
        let y = vec![1.0, 2.0, 3.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0]];
        let weights = vec![1.0, 2.0]; // yより短い

        let result = OlsInput::from_columns_weighted(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            &weights,
        );

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::WeightDimensionMismatch {
                y_rows: 3,
                weight_rows: 2
            }
        );
    }

    #[test]
    fn from_columns_weighted_rejects_zero_and_negative_and_nan_weights() {
        let y = vec![1.0, 2.0, 3.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0]];

        for (bad_weight, bad_row) in [(0.0, 0), (-1.0, 1), (f64::NAN, 2)] {
            let mut weights = vec![1.0, 1.0, 1.0];
            weights[bad_row] = bad_weight;

            let result = OlsInput::from_columns_weighted(
                &y,
                &x_columns,
                vec!["x1".to_string()],
                true,
                "y".to_string(),
                &weights,
            );

            match result.unwrap_err() {
                LeastSquaresError::NonPositiveWeight { row, weight } => {
                    assert_eq!(row, bad_row);
                    if bad_weight.is_nan() {
                        assert!(weight.is_nan());
                    } else {
                        assert_eq!(weight, bad_weight);
                    }
                }
                other => panic!("expected NonPositiveWeight, got {other:?}"),
            }
        }
    }

    #[test]
    fn least_squares_error_messages_are_human_readable() {
        // 6種の共通バリアント（DimensionMismatch等）のメッセージ検証は
        // `engine::error`側のテストに集約済み。ここではOLS/WLS固有の
        // バリアントに加え、`Common`が`CommonError`のDisplayをtransparentに転送する
        // ことだけを確認する。
        assert_eq!(
            LeastSquaresError::WeightDimensionMismatch {
                y_rows: 10,
                weight_rows: 8
            }
            .to_string(),
            "dimension mismatch: y has 10 rows but weight has 8 rows"
        );
        assert_eq!(
            LeastSquaresError::NonPositiveWeight {
                row: 3,
                weight: 0.0
            }
            .to_string(),
            "weight at row 3 must be positive, got 0"
        );
        assert_eq!(
            LeastSquaresError::InvalidHacLags {
                hac_lags: -1,
                n: 100
            }
            .to_string(),
            "hac_lags must be in the range [0, n): got -1, n=100"
        );
        assert_eq!(
            LeastSquaresError::SingularMatrix.to_string(),
            "design matrix is singular (perfect multicollinearity detected)"
        );
        assert_eq!(
            LeastSquaresError::Common(CommonError::MissingClusterColumn).to_string(),
            "cov_type='cluster' requires cluster identifiers to be provided"
        );
    }

    #[test]
    fn least_squares_error_implements_partial_eq() {
        assert_eq!(
            LeastSquaresError::SingularMatrix,
            LeastSquaresError::SingularMatrix
        );
        assert_ne!(
            LeastSquaresError::Common(CommonError::InsufficientClusters { g: 1 }),
            LeastSquaresError::Common(CommonError::InsufficientClusters { g: 0 })
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
            LeastSquaresError::Common(CommonError::InsufficientObservations { n: 2, k: 2 })
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

        assert_eq!(result.unwrap_err(), LeastSquaresError::SingularMatrix);
    }

    #[test]
    fn fit_returns_computation_failed_for_extreme_scale_difference_in_f_test() {
        // x1は1e6オーダー、x2は1e-3オーダーとスケールが極端に異なる（x3は通常
        // スケール）。x1・x2・x3は互いに線形従属ではないため設計行列自体は
        // フルランク（SingularMatrixにはならない）だが、傾き係数の同時共分散
        // 部分行列（wald_f_testが使う3x3部分行列）の条件数がスケール比の2乗
        // （≈1e18）相当となり倍精度の限界を超える
        // （ensure_well_conditioned_symmetric_matrixで検出）。
        let n = 10;
        let x1: Vec<f64> = (1..=n).map(|i| 1e6 * (i as f64)).collect();
        let x2: Vec<f64> = (1..=n).map(|i| 1e-3 * (i as f64).powi(2)).collect();
        let x3: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let noise = if i % 2 == 0 { 0.1 } else { -0.1 };
                1.0 + 2.0 * x1[i] + 3.0 * x2[i] + 0.5 * x3[i] + noise
            })
            .collect();

        let input = OlsInput::from_columns(
            &y,
            &[x1, x2, x3],
            vec!["x1".to_string(), "x2".to_string(), "x3".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = OlsEstimator::fit(input, CovType::Classical, 0.95);

        assert!(matches!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::ComputationFailed(_))
        ));
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
            LeastSquaresError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5
            })
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
    /// `docs/spec/ols-spec.md`「標準誤差」、
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

            let cov_type_label = format!("{cov_type:?}");
            let estimator = OlsEstimator::fit(input, cov_type, 0.95).unwrap();

            for j in 0..2 {
                let msg = format!("{cov_type_label}, param {j}");
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

    /// 同じデータセット（x=[1..5], y=[2,4,5,4,5]）でのHAC（Newey-West、`maxlags=1`）。
    /// 期待値はstatsmodels 0.14.6で独立に計算・検算済み
    /// （`sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1}, use_t=True)`。
    /// `use_correction`はstatsmodelsの既定である`False`のまま、明示指定はしていない）。
    #[test]
    fn fit_computes_hac_std_errors_with_explicit_lags() {
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

        let cov_type = CovType::Hac {
            lags: Some(1),
            time_order: None,
        };
        let estimator = OlsEstimator::fit(input, cov_type, 0.95).unwrap();

        let se = estimator.std_errors();
        assert!((*se.get(0, 0) - 0.659_090_282_131_361_7).abs() < 1e-6);
        assert!((*se.get(1, 0) - 0.164_924_225_024_705_7).abs() < 1e-6);

        let t = estimator.t_stats();
        assert!((*t.get(0, 0) - 3.337_934_209_689_228).abs() < 1e-6);
        assert!((*t.get(1, 0) - 3.638_034_375_545_013_5).abs() < 1e-6);

        let p = estimator.p_values();
        assert!((*p.get(0, 0) - 0.044_455_744_969_471_62).abs() < 1e-6);
        assert!((*p.get(1, 0) - 0.035_791_053_269_350_51).abs() < 1e-6);

        let lower = estimator.conf_lower();
        let upper = estimator.conf_upper();
        assert!((*lower.get(0, 0) - 0.102_480_566_782_648_72).abs() < 1e-6);
        assert!((*upper.get(0, 0) - 4.297_519_433_217_346).abs() < 1e-6);
        assert!((*lower.get(1, 0) - 0.075_137_509_418_346_96).abs() < 1e-6);
        assert!((*upper.get(1, 0) - 1.124_862_490_581_653_8).abs() < 1e-6);
    }

    /// `hac_lags=None`（経験則自動計算）が`L = floor(4*(n/100)^(2/9))`と一致することを確認する。
    /// n=5の場合L=2。期待値はstatsmodelsで`maxlags=2`を明示指定して独立に計算・検算済み
    /// （`docs/spec/ols-spec.md`「標準誤差」のHACの式通りベンチマーク側もL=2を使う前提）。
    #[test]
    fn fit_computes_hac_std_errors_with_auto_lags() {
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

        let cov_type = CovType::Hac {
            lags: None,
            time_order: None,
        };
        let estimator = OlsEstimator::fit(input, cov_type, 0.95).unwrap();

        let se = estimator.std_errors();
        assert!((*se.get(0, 0) - 0.577_350_269_189_624_1).abs() < 1e-6);
        assert!((*se.get(1, 0) - 0.164_924_225_024_705_75).abs() < 1e-6);
    }

    /// `time_order`を指定した場合、行順がシャッフルされていても時系列順に並べ替えてから
    /// ラグ付き自己共分散を計算することを確認する。データはHAC(maxlags=1)テストと同一の
    /// (x, y)を、時系列順の逆転を含む順序（time値=xの値そのもの）でシャッフルして与える。
    /// 期待値は`fit_computes_hac_std_errors_with_explicit_lags`と同じ
    /// （時系列順に並べ替えれば同一データになるため）。
    #[test]
    fn fit_computes_hac_std_errors_respecting_time_order() {
        // 元の時系列順: time=[1,2,3,4,5], y=[2,4,5,4,5]
        // これを time順=[3,1,5,2,4] の並びでシャッフルして入力する
        let shuffled_time = vec![3.0, 1.0, 5.0, 2.0, 4.0];
        let shuffled_x = shuffled_time.clone();
        let shuffled_y = vec![5.0, 2.0, 5.0, 4.0, 4.0];

        let input = OlsInput::from_columns(
            &shuffled_y,
            &[shuffled_x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let cov_type = CovType::Hac {
            lags: Some(1),
            time_order: Some(shuffled_time),
        };
        let estimator = OlsEstimator::fit(input, cov_type, 0.95).unwrap();

        let se = estimator.std_errors();
        assert!((*se.get(0, 0) - 0.659_090_282_131_361_7).abs() < 1e-6);
        assert!((*se.get(1, 0) - 0.164_924_225_024_705_7).abs() < 1e-6);
    }

    #[test]
    fn fit_returns_invalid_hac_lags_when_out_of_range() {
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

        let cov_type = CovType::Hac {
            lags: Some(-1),
            time_order: None,
        };
        let result = OlsEstimator::fit(input, cov_type, 0.95);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::InvalidHacLags { hac_lags: -1, n: 5 }
        );
    }

    /// `hac_lags`の上限側の境界。`n=5`のとき`lags=5`（`n`自体）は範囲外（`[0, n)`）、
    /// `lags=4`（`n-1`）は許容される最大値であることを確認する。
    #[test]
    fn fit_returns_invalid_hac_lags_when_equal_to_n() {
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

        let cov_type = CovType::Hac {
            lags: Some(5),
            time_order: None,
        };
        let result = OlsEstimator::fit(input, cov_type, 0.95);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::InvalidHacLags { hac_lags: 5, n: 5 }
        );
    }

    #[test]
    fn fit_accepts_hac_lags_at_upper_boundary_of_n_minus_one() {
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

        let cov_type = CovType::Hac {
            lags: Some(4), // n - 1、許容される最大値
            time_order: None,
        };
        let result = OlsEstimator::fit(input, cov_type, 0.95);

        assert!(result.is_ok());
    }

    /// `confidence_level`の境界値（0.0・1.0ちょうど）が範囲外として拒否されることを確認する
    /// （`!(level > 0.0 && level < 1.0)`という判定式の境界そのものの検証）。
    #[test]
    fn fit_returns_invalid_confidence_level_at_exact_boundaries() {
        for level in [0.0, 1.0, -0.1] {
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

            let result = OlsEstimator::fit(input, CovType::Classical, level);

            assert_eq!(
                result.unwrap_err(),
                LeastSquaresError::Common(CommonError::InvalidConfidenceLevel {
                    confidence_level: level
                }),
                "level={level}"
            );
        }
    }

    /// `CovType::Hac { lags: Some(0), .. }`は`Ŝ = Ŝ₀`（ラグ項なし）に退化し、これは
    /// `HC0`の`Ψ̂ = Σ_i ε̂_i² x_i x_i'`と数学的に同一の式になる（`hac_cov_params`の
    /// l=0項のドキュメント参照）。2つの独立した実装（`hc_cov_params`と`hac_cov_params`）が
    /// この境界で一致することを確認する内部整合性テスト。
    #[test]
    fn fit_hac_with_zero_lags_matches_hc0() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 8.0, 3.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];

        let input_hac = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let hac = OlsEstimator::fit(
            input_hac,
            CovType::Hac {
                lags: Some(0),
                time_order: None,
            },
            0.95,
        )
        .unwrap();

        let input_hc0 = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let hc0 = OlsEstimator::fit(input_hc0, CovType::Hc0, 0.95).unwrap();

        for j in 0..2 {
            assert!(
                (*hac.std_errors().get(j, 0) - *hc0.std_errors().get(j, 0)).abs() < 1e-12,
                "param {j}: hac(lags=0)={}, hc0={}",
                *hac.std_errors().get(j, 0),
                *hc0.std_errors().get(j, 0)
            );
        }
    }

    /// x=[1..5], y=[2,4,5,4,5]（切片あり、classical）の適合度統計量。
    /// 期待値はstatsmodels 0.14.6で独立に計算・検算済み
    /// （`sm.OLS(Y, X).fit(use_t=True)`。`fvalue`/`f_pvalue`は古典的F検定と
    /// ロバストWald検定の式が代数的に一致することも別途手計算で確認済み）。
    #[test]
    fn fit_computes_r_squared_and_information_criteria_with_intercept() {
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

        assert!((estimator.r_squared() - 0.599_999_999_999_999_9).abs() < 1e-9);
        assert!((estimator.r_squared_adj() - 0.466_666_666_666_666_56).abs() < 1e-9);
        assert!((estimator.log_likelihood() - (-5.259_769_728_322_863)).abs() < 1e-9);
        assert!((estimator.aic() - 14.519_539_456_645_726).abs() < 1e-9);
        assert!((estimator.bic() - 13.738_415_281_513_927).abs() < 1e-9);
        assert!((estimator.f_statistic() - 4.499_999_999_999_999).abs() < 1e-6);
        assert!((estimator.f_p_value() - 0.124_027_062_657_554_59).abs() < 1e-6);
    }

    /// 同じ(x, y)を切片なしで推定した場合。R²・調整済みR²がuncentered TSS
    /// （`Σy_i²`）を基準に計算されることを確認する（statsmodelsの`k_constant=0`の
    /// 挙動と一致。`docs/spec/ols-spec.md`「適合度統計量」参照）。
    #[test]
    fn fit_computes_r_squared_without_intercept_uses_uncentered_tss() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            false,
            "y".to_string(),
        )
        .unwrap();

        let estimator = OlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        assert!((estimator.r_squared() - 0.920_930_232_558_139_5).abs() < 1e-9);
        assert!((estimator.r_squared_adj() - 0.901_162_790_697_674_5).abs() < 1e-9);
        assert!((estimator.log_likelihood() - (-7.863_404_415_393_264)).abs() < 1e-9);
        assert!((estimator.aic() - 17.726_808_830_786_528).abs() < 1e-9);
        assert!((estimator.bic() - 17.336_246_743_220_627).abs() < 1e-9);
        assert!((estimator.f_statistic() - 46.588_235_294_117_66).abs() < 1e-6);
        assert!((estimator.f_p_value() - 0.002_409_205_984_197_115_5).abs() < 1e-6);
    }

    /// HC1・HAC(maxlags=1)でのF統計量がロバストWald検定になることを確認する
    /// （R²・AIC/BIC・対数尤度は`cov_type`に依存しないため、ここではF統計量のみ検証）。
    /// 期待値はstatsmodelsで独立に計算・検算済み
    /// （`sm.OLS(Y, X).fit(cov_type=..., use_t=True)`）。
    #[test]
    fn fit_computes_robust_wald_f_test_for_hc_and_hac() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];

        let input_hc1 = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator_hc1 = OlsEstimator::fit(input_hc1, CovType::Hc1, 0.95).unwrap();
        assert!((estimator_hc1.f_statistic() - 6.279_069_767_441_904).abs() < 1e-6);
        assert!((estimator_hc1.f_p_value() - 0.087_259_022_565_828_96).abs() < 1e-6);

        let input_hac = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let cov_type_hac = CovType::Hac {
            lags: Some(1),
            time_order: None,
        };
        let estimator_hac = OlsEstimator::fit(input_hac, cov_type_hac, 0.95).unwrap();
        assert!((estimator_hac.f_statistic() - 13.235_294_117_647_193).abs() < 1e-6);
        assert!((estimator_hac.f_p_value() - 0.035_791_053_269_350_51).abs() < 1e-6);
    }

    /// 同じデータセット（x=[1..5], y=[2,4,5,4,5]）を2クラスター（groups=["a","a","b","b","b"]）
    /// でのクラスターロバスト標準誤差。期待値はstatsmodels 0.14.6で独立に計算・検算済み
    /// （`sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)`。
    /// 小標本補正はstatsmodelsの既定`use_correction=True`のまま、明示指定はしていない）。
    #[test]
    fn fit_computes_cluster_std_errors_t_stats_p_values_conf_int_and_f_test() {
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

        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];
        let cov_type = CovType::Cluster {
            groups: Some(groups),
        };
        let estimator = OlsEstimator::fit(input, cov_type, 0.95).unwrap();

        let se = estimator.std_errors();
        assert!((*se.get(0, 0) - 0.785_196_366_097_886_8).abs() < 1e-6);
        assert!((*se.get(1, 0) - 0.230_940_107_675_849_05).abs() < 1e-6);

        let t = estimator.t_stats();
        assert!((*t.get(0, 0) - 2.801_846_894_596_724_5).abs() < 1e-6);
        assert!((*t.get(1, 0) - 2.598_076_211_353_332).abs() < 1e-6);

        let p = estimator.p_values();
        assert!((*p.get(0, 0) - 0.218_242_895_017_685_43).abs() < 1e-6);
        assert!((*p.get(1, 0) - 0.233_908_049_281_92).abs() < 1e-6);

        let lower = estimator.conf_lower();
        let upper = estimator.conf_upper();
        assert!((*lower.get(0, 0) - (-7.776_865_785_740_13)).abs() < 1e-6);
        assert!((*upper.get(0, 0) - 12.176_865_785_740_125).abs() < 1e-6);
        assert!((*lower.get(1, 0) - (-2.334_372_289_923_566_6)).abs() < 1e-6);
        assert!((*upper.get(1, 0) - 3.534_372_289_923_567_7).abs() < 1e-6);

        assert!((estimator.f_statistic() - 6.750_000_000_000_083_5).abs() < 1e-6);
        assert!((estimator.f_p_value() - 0.233_908_049_281_92).abs() < 1e-6);
    }

    /// クラスターロバスト標準誤差は、同じ入力に対して`fit()`を複数回呼んでも
    /// ビット単位で同じ結果になること（`cluster_cov_params`内部の集約が
    /// `HashMap`の反復順序に依存し、実行のたびに浮動小数点の丸め誤差レベルで
    /// 結果がぶれていた回帰。`BTreeMap`化で修正した）。
    #[test]
    fn fit_cluster_std_errors_are_deterministic_across_repeated_fits() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0, 3.0, 6.0, 1.0, 7.0, 2.5];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let groups: Vec<String> = (0..10).map(|i| format!("g{}", i % 4)).collect();

        let build = || {
            let input = OlsInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();
            let cov_type = CovType::Cluster {
                groups: Some(groups.clone()),
            };
            OlsEstimator::fit(input, cov_type, 0.95).unwrap()
        };

        let first = build();
        for _ in 0..20 {
            let repeat = build();
            assert_eq!(
                *repeat.std_errors().get(0, 0),
                *first.std_errors().get(0, 0)
            );
            assert_eq!(
                *repeat.std_errors().get(1, 0),
                *first.std_errors().get(1, 0)
            );
        }
    }

    #[test]
    fn fit_returns_missing_cluster_column_when_groups_not_provided() {
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

        let result = OlsEstimator::fit(input, CovType::Cluster { groups: None }, 0.95);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::MissingClusterColumn)
        );
    }

    #[test]
    fn fit_returns_insufficient_clusters_when_only_one_group() {
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

        let groups = vec!["a".to_string(); 5];
        let cov_type = CovType::Cluster {
            groups: Some(groups),
        };
        let result = OlsEstimator::fit(input, cov_type, 0.95);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::InsufficientClusters { g: 1 })
        );
    }

    /// 説明変数が定数項のみ（傾き係数が無い）モデル。F検定は検定対象が存在しないため、
    /// statsmodels同様NaNを返す（`OlsEstimator::fit`の`df_model == 0`分岐）。
    #[test]
    fn fit_returns_nan_f_statistic_when_model_has_no_slope_regressors() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = OlsInput::from_columns(&y, &[], vec![], true, "y".to_string()).unwrap();

        let estimator = OlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        assert!(estimator.f_statistic().is_nan());
        assert!(estimator.f_p_value().is_nan());
    }

    #[test]
    fn fit_exposes_input_cov_type_and_residuals_via_getters() {
        // y = 1 + 2*x、ノイズなしの厳密解を持つデータ（残差が全て0に近いことを確認しやすい）
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

        assert_eq!(estimator.input().nobs(), 5);
        assert_eq!(estimator.cov_type(), &CovType::Classical);
        let residuals = estimator.residuals();
        for i in 0..5 {
            assert!((*residuals.get(i, 0)).abs() < 1e-9);
        }
    }

    #[test]
    fn fitted_values_equals_y_minus_residuals() {
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

        let fitted = estimator.fitted_values();
        for (i, &y_i) in y.iter().enumerate() {
            let expected = y_i - *estimator.residuals().get(i, 0);
            assert!((*fitted.get(i, 0) - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn predict_new_data_matches_manually_computed_linear_combination_with_intercept() {
        // params = [const=1.0, x1=2.0]、新規データx1=[10, 20]に対する予測値は1+2*10=21, 1+2*20=41
        let params = vec![1.0, 2.0];
        let new_x_columns = vec![vec![10.0, 20.0]];

        let predicted = predict_new_data(&params, true, &new_x_columns);

        assert_eq!(predicted, vec![21.0, 41.0]);
    }

    #[test]
    fn predict_new_data_matches_manually_computed_linear_combination_without_intercept() {
        // params = [x1=2.0, x2=3.0]（切片なし）、新規データ(x1,x2)=(10,1)と(20,2)に対する
        // 予測値は2*10+3*1=23, 2*20+3*2=46
        let params = vec![2.0, 3.0];
        let new_x_columns = vec![vec![10.0, 20.0], vec![1.0, 2.0]];

        let predicted = predict_new_data(&params, false, &new_x_columns);

        assert_eq!(predicted, vec![23.0, 46.0]);
    }

    #[test]
    #[should_panic]
    fn predict_new_data_panics_when_column_count_does_not_match_params() {
        // new_x_columns.len()が期待する列数（params.len() - has_intercept）と
        // 一致しない場合はengine_pybind側の実装バグでしか起こり得ない内部契約違反のため、
        // from_columns_panics_on_mismatched_names_arityと同じ性質でdebug_assert_eq!が
        // パニックする。
        let params = vec![1.0, 2.0, 3.0]; // has_intercept=trueなら期待列数は2
        let new_x_columns = vec![vec![10.0, 20.0]]; // 1列しかない

        let _ = predict_new_data(&params, true, &new_x_columns);
    }

    #[test]
    fn predict_new_data_ignores_column_name_and_uses_has_intercept_flag_directly() {
        // has_intercept=falseで学習した場合、xの列名がたまたま"const"であっても
        // （include_intercept=falseならこの列名は許可される、engine_pybind::fitの
        // 衝突チェックはinclude_intercept=trueのときのみ適用）、predict_new_dataは
        // param_names等の文字列を一切見ずhas_intercept引数のみで組み立てるため
        // 誤動作しない（この引数自体をどう決定するかはengine_pybind側の責務）。
        // params = [const=2.0, x2=3.0]（切片なし、"const"という名前のただの説明変数）
        let params = vec![2.0, 3.0];
        let new_x_columns = vec![vec![100.0, 200.0], vec![10.0, 20.0]];

        let predicted = predict_new_data(&params, false, &new_x_columns);

        // 2*100+3*10=230, 2*200+3*20=460（"const"列も普通の説明変数として2倍される）
        assert_eq!(predicted, vec![230.0, 460.0]);
    }

    #[test]
    #[should_panic]
    fn fit_panics_when_cluster_groups_length_does_not_match_nobs() {
        // groups.len() != nはengine_pybind側の実装バグでしか起こり得ない内部契約違反のため、
        // Errではなくdebug_assert!でパニックする（from_columns_panics_on_mismatched_names_arity
        // と同じ性質）。
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

        let groups = vec!["a".to_string(), "b".to_string()]; // n=5のはずが長さ2
        let cov_type = CovType::Cluster {
            groups: Some(groups),
        };
        let _ = OlsEstimator::fit(input, cov_type, 0.95);
    }
}
