//! Tobitの入力データ（被説明変数・設計行列・打ち切り境界）の型定義。
//!
//! `engine`はpolars/PyO3を一切知らない（`.claude/rules/rust-style.md`「責務分離」参照）。
//! `engine_pybind`はpolars DataFrameから列ごとに`Vec<f64>`を抽出するところまでを担い
//! （`column_extraction::extract_f64_column`）、それらの列を本モジュールの
//! `TobitInput::from_columns`に渡す。`faer::Mat`への組み立て（切片列の自動追加を含む）は
//! ここ（engine側）の責務とする。`LogitInput`/`ProbitInput`と同型の設計だが、Tobitは
//! 打ち切り境界（`lower`/`upper`）という他の2手法にはない引数・検証を追加で持つ
//! （`docs/planning/specs/nonlinear-api-design.md`7章「Tobitの打ち切り境界オプション」）。
//!
//! ## Logit/Probitとの設計上の違い: 検証の実施箇所
//!
//! Logit/Probitの`validate_binary_y`（`nonlinear/common.rs`）は`fit()`冒頭で呼ばれ、
//! `from_columns`自体は次元検証のみを行う。Tobitは打ち切り境界（`lower`/`upper`）を
//! `from_columns`の追加引数として受け取る都合上、境界自体の妥当性検証・`y`との整合性検証も
//! `from_columns`内で完結させる（Issue #213で確定）。
//!
//! ## 数式（打ち切り正規回帰）
//!
//! 内部最適化パラメータは`(β, s)`という`k+1`次元ベクトル（`s = log σ`、`σ>0`の制約を
//! 回避するための変数変換。`docs/planning/specs/nonlinear-implementation-notes.md`
//! 「パラメータ化（内部最適化変数）」参照。Olsen(1978)の`(β/σ, 1/σ)`変換は不採用）。
//!
//! `TobitInput::from_columns`が保証する通り、各観測は`lower`/`upper`との比較で
//! 3種類に分類される（`lower`/`upper`いずれかが`None`ならその側の分類は発生しない）:
//!
//! - **非打ち切り**（`lower < yᵢ < upper`）: `vᵢ = (yᵢ - xᵢ'β)/σ`とおくと
//!   `ℓᵢ(θ) = -log σ - (1/2)log(2π) - vᵢ²/2`（通常の正規回帰の対数尤度）
//! - **左打ち切り**（`yᵢ = lower`）: `zᵢ = (lower - xᵢ'β)/σ`とおくと`ℓᵢ(θ) = log Φ(zᵢ)`
//!   （`y*ᵢ ≦ lower`となる確率）
//! - **右打ち切り**（`yᵢ = upper`）: `wᵢ = (xᵢ'β - upper)/σ`とおくと`ℓᵢ(θ) = log Φ(wᵢ)`
//!   （`log(1-Φ((upper-xᵢ'β)/σ)) = log Φ(wᵢ)`と同値、`Φ(-a)=1-Φ(a)`を使用）
//!
//! `λ(u) = φ(u)/Φ(u)`（逆ミルズ比）とおくと、`(β, s)`それぞれについてのスコア
//! （対数尤度の1階微分）は次の通り（`xᵢⱼ`は観測`i`・変数`j`の説明変数値）:
//!
//! | | `∂ℓᵢ/∂βⱼ` | `∂ℓᵢ/∂s` |
//! |---|---|---|
//! | 非打ち切り | `vᵢxᵢⱼ/σ` | `vᵢ²-1` |
//! | 左打ち切り | `-λ(zᵢ)xᵢⱼ/σ` | `-zᵢλ(zᵢ)` |
//! | 右打ち切り | `λ(wᵢ)xᵢⱼ/σ` | `-wᵢλ(wᵢ)` |
//!
//! （左右で`β`成分の符号が反転するのは、`zᵢ`・`wᵢ`の`β`に対する偏微分の符号が
//! `∂zᵢ/∂βⱼ=-xᵢⱼ/σ`・`∂wᵢ/∂βⱼ=+xᵢⱼ/σ`と逆になるため。`s`成分はいずれも`∂zᵢ/∂s=-zᵢ`・
//! `∂wᵢ/∂s=-wᵢ`という同じ形の式に帰着するため符号は反転しない）
//!
//! Hessian（対数尤度の2階微分）は、`A(u) = λ(u)(u+λ(u))`・`C(u) = uA(u)-λ(u)`とおくと
//! （`λ'(u)=-λ(u)(u+λ(u))`という逆ミルズ比の既知の微分公式を使用。Probitの
//! Hessian`W=diag(λᵢ(λᵢ+zᵢ))`の`A(u)`と同じ構造）:
//!
//! | | `∂²ℓᵢ/∂βⱼ∂βₘ` | `∂²ℓᵢ/∂βⱼ∂s` | `∂²ℓᵢ/∂s²` |
//! |---|---|---|---|
//! | 非打ち切り | `-xᵢⱼxᵢₘ/σ²` | `-2xᵢⱼvᵢ/σ` | `-2vᵢ²` |
//! | 左打ち切り | `-xᵢⱼxᵢₘA(zᵢ)/σ²` | `-xᵢⱼC(zᵢ)/σ` | `-zᵢC(zᵢ)` |
//! | 右打ち切り | `-xᵢⱼxᵢₘA(wᵢ)/σ²` | `xᵢⱼC(wᵢ)/σ` | `-wᵢC(wᵢ)` |
//!
//! `CostFunction`は`-ℓ(θ) = -Σᵢℓᵢ(θ)`（argminは最小化フレームワークのため）。
//! `Gradient`/`Hessian`トレイトは`CostFunction`と同じ符号（`-ℓ`の1階・2階微分）で実装する
//! （`run_solver`のdocコメント「`Hessian`トレイトの符号規約」参照）。`scores()`
//! （`cov_type`共通行列演算向け）は符号反転しない生のスコア`sᵢ=∂ℓᵢ/∂θ`を返す
//! （`LogitProblem::scores`・`ProbitProblem::scores`と同じ規約）。
//!
//! ## 数値安定化について
//!
//! 打ち切り観測の`λ(u)=φ(u)/Φ(u)`は、`Probit`の一般化残差と全く同じ`0.0/0.0`のNaN化
//! リスクを持つ（`u`が極端に負のとき）。`nonlinear/common.rs`の`clamped_pdf_cdf`
//! （Tobit実装時にProbit専用から共有ユーティリティへ移設）をそのまま再利用する。

use crate::error::CommonError;
use crate::nonlinear::common::{MleError, clamped_pdf_cdf};
use argmin::core::{CostFunction, Error as OptimizerError, Gradient, Hessian};
use faer::Mat;
use statrs::distribution::Normal;

/// Tobitの被説明変数・設計行列・打ち切り境界を保持する入力データ。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
/// `from_columns`で組み立てた後は、getter経由でのみアクセスする。
#[derive(Debug)]
pub struct TobitInput {
    /// 被説明変数 (n, 1)
    y: Mat<f64>,
    /// 設計行列 (n, k)。`include_intercept=true`の場合、先頭列が定数項（すべて1.0）
    x: Mat<f64>,
    /// 係数名（`include_intercept=true`なら先頭が"const"）。`x`の列と対応する
    param_names: Vec<String>,
    /// 被説明変数名
    dep_var_name: String,
    /// 定数項を含むか。`nonlinear/common.rs`の`standardize_columns`等で必要
    has_intercept: bool,
    /// 打ち切りの下限。`None`は「左側は打ち切りなし」を意味する
    lower: Option<f64>,
    /// 打ち切りの上限。`None`は「右側は打ち切りなし」を意味する
    upper: Option<f64>,
}

impl TobitInput {
    /// 列ごとの`Vec<f64>`（`engine_pybind`がpolars DataFrameから抽出済み）から
    /// `TobitInput`を組み立てる。`include_intercept=true`の場合、設計行列の先頭列に
    /// 定数項（すべて1.0）を自動追加する。
    ///
    /// 検証は次の順序で行う（モジュール冒頭のdocコメント参照）:
    /// 1. 次元検証（`y`と各`x_columns`の長さ一致）
    /// 2. 打ち切り境界自体の検証（`lower`/`upper`の少なくとも一方が有限の`Some`、かつ
    ///    両方`Some`の場合は`lower < upper`。`NaN`・無限大は不正な境界として弾く）
    /// 3. `y`と境界の整合性検証（`lower`指定時に`y < lower`の行が無い、`upper`指定時に
    ///    `y > upper`の行が無い）
    ///
    /// # Errors
    /// - `y`といずれかの`x_columns`の長さが一致しない: `CommonError::DimensionMismatch`
    /// - `lower`/`upper`が両方`None`、いずれかが`NaN`または無限大、または両方`Some`で
    ///   `lower >= upper`: `MleError::InvalidCensoringBounds`
    /// - `y`が指定された境界の範囲外の値を含む: `MleError::YOutOfCensoringBounds`
    ///
    /// # パニックについて
    /// `x_names.len() != x_columns.len()`の場合は`debug_assert!`でパニックする。これは
    /// 呼び出し側（`engine_pybind`）の実装バグでしか起こり得ない内部契約であり、
    /// 実データに起因する`ValidationError`とは性質が異なるため区別している
    /// （`LogitInput::from_columns`と同じ方針）。
    #[allow(clippy::too_many_arguments)]
    pub fn from_columns(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        include_intercept: bool,
        dep_var_name: String,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> Result<Self, MleError> {
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

        // 肯定形の妥当性条件を`!`で囲み、NaNを自動的に弾く書き方
        // （`validate_fit_preconditions`の`confidence_level`検証と同じパターン。
        // `l >= u`のような否定形の直接比較だとNaNに対して常に`false`になり
        // すり抜けてしまう。`is_finite()`で無限大も合わせて弾く）。
        let bounds_valid = match (lower, upper) {
            (None, None) => false,
            (Some(l), None) => l.is_finite(),
            (None, Some(u)) => u.is_finite(),
            (Some(l), Some(u)) => l.is_finite() && u.is_finite() && l < u,
        };
        if !bounds_valid {
            return Err(MleError::InvalidCensoringBounds { lower, upper });
        }

        for (i, &value) in y.iter().enumerate() {
            if let Some(l) = lower
                && value < l
            {
                return Err(MleError::YOutOfCensoringBounds {
                    row: i,
                    value,
                    lower,
                    upper,
                });
            }
            if let Some(u) = upper
                && value > u
            {
                return Err(MleError::YOutOfCensoringBounds {
                    row: i,
                    value,
                    lower,
                    upper,
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
            has_intercept: include_intercept,
            lower,
            upper,
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

    /// 打ち切りの下限。`None`は「左側は打ち切りなし」を意味する
    pub fn lower(&self) -> Option<f64> {
        self.lower
    }

    /// 打ち切りの上限。`None`は「右側は打ち切りなし」を意味する
    pub fn upper(&self) -> Option<f64> {
        self.upper
    }
}

/// `0.5 * ln(2π)`。正規分布の対数密度`log φ(v) = -0.5*ln(2π) - v²/2`の定数項
/// （コンパイル時定数としてハードコード。`ln`は非`const fn`のため`nonlinear/common.rs`の
/// `U_CLAMP`と同じ方式）。
const HALF_LN_2PI: f64 = 0.918_938_533_204_672_7;

/// 観測1件の対数尤度・スコア・Hessianへの寄与（モジュール冒頭の数式表参照）。`cost`/
/// `gradient`/`hessian`/`scores`いずれも観測ごとに同じ分類（非打ち切り/左打ち切り/
/// 右打ち切り）を経由するため、重複を避けてここに集約する。
struct Contribution {
    /// 対数尤度への寄与 `ℓᵢ(θ)`
    log_lik: f64,
    /// 生のスコア`∂ℓᵢ/∂θ`のうち`β`成分の係数（実際の値は`coef * xᵢⱼ/σ`、`Gradient`
    /// トレイトとは符号が逆。`scores()`が返す値そのもの）
    score_beta_coef: f64,
    /// 生のスコアのうち`s`（`log σ`）成分 `∂ℓᵢ/∂s`
    score_s: f64,
    /// `-ℓᵢ`のHessianのうち`β`ブロックの係数（実際の値は`coef * xᵢⱼxᵢₘ/σ²`）
    h_beta_coef: f64,
    /// `-ℓᵢ`のHessianのうち`β`-`s`ブロックの係数（実際の値は`coef * xᵢⱼ/σ`）
    h_beta_s_coef: f64,
    /// `-ℓᵢ`のHessianのうち`s`-`s`要素
    h_ss: f64,
}

/// 非打ち切り観測（モジュール冒頭の数式表「非打ち切り」列）。`v = (yᵢ-xᵢ'β)/σ`、
/// `s = log σ`。
fn uncensored_contribution(v: f64, s: f64) -> Contribution {
    Contribution {
        log_lik: -s - HALF_LN_2PI - 0.5 * v * v,
        score_beta_coef: v,
        score_s: v * v - 1.0,
        h_beta_coef: 1.0,
        h_beta_s_coef: 2.0 * v,
        h_ss: 2.0 * v * v,
    }
}

/// 打ち切り観測（左右いずれか、モジュール冒頭の数式表参照）。`zeta`は打ち切り境界を
/// 標準化した値（左打ち切り: `(lower-xᵢ'β)/σ`、右打ち切り: `(xᵢ'β-upper)/σ`）、
/// `direction`は`β`成分の符号（左: `1.0`、右: `-1.0`。`s`成分・Hessianはいずれも
/// `direction`に依らず同じ式に帰着する、モジュール冒頭のdocコメント参照）。
fn censored_contribution(normal: &Normal, zeta: f64, direction: f64) -> Contribution {
    let (phi, big_phi) = clamped_pdf_cdf(normal, zeta);
    let lambda = phi / big_phi;
    let a = lambda * (zeta + lambda);
    let c = zeta * a - lambda;
    Contribution {
        log_lik: big_phi.ln(),
        score_beta_coef: -direction * lambda,
        score_s: -zeta * lambda,
        h_beta_coef: a,
        h_beta_s_coef: direction * c,
        h_ss: zeta * c,
    }
}

/// Tobitの負の対数尤度・スコア・Hessian（argminの`CostFunction`/`Gradient`/`Hessian`
/// トレイト実装）。`TobitInput`の`X`・`y`・打ち切り境界を保持する（`run_solver`が
/// `problem`の所有権を必要とするため、`TobitInput`とは独立した所有データとして持つ。
/// `Clone`は`argmin::core::Executor`が要求する。`LogitProblem`/`ProbitProblem`と
/// 同じ設計）。
#[derive(Debug, Clone)]
pub struct TobitProblem {
    x: Mat<f64>,
    y: Mat<f64>,
    lower: Option<f64>,
    upper: Option<f64>,
}

impl TobitProblem {
    /// `input`の`x`・`y`・打ち切り境界をそのまま（未標準化のスケールで）複製して構築する。
    /// 閉じた形の解・OLSとの整合性と突き合わせる単体テスト専用（`LogitProblem::new`と
    /// 同じ位置づけ）。標準化済み設計行列から構築するコンストラクタ（`LogitProblem::
    /// from_standardized`相当）は`TobitEstimator::fit`実装時（Issue #215以降）に追加する
    /// （本Issueのスコープ外、`fit()`が存在しない状態で追加すると未使用の`dead_code`に
    /// なるため）。
    #[cfg(test)]
    fn new(input: &TobitInput) -> Self {
        Self {
            x: input.x().clone(),
            y: input.y().clone(),
            lower: input.lower(),
            upper: input.upper(),
        }
    }

    /// 説明変数の数`k`（定数項を含む、`params`の先頭`k`要素が`β`に対応）。
    fn k(&self) -> usize {
        self.x.ncols()
    }

    /// 観測`i`の線形予測子 `xᵢ'β`（`params`の先頭`k`要素のみ使う。`params[k]`は`s=logσ`）。
    fn linear_predictor(&self, i: usize, params: &[f64]) -> f64 {
        let k = self.k();
        (0..k).map(|j| *self.x.get(i, j) * params[j]).sum()
    }

    /// 観測`i`の分類（非打ち切り/左打ち切り/右打ち切り）に応じた[`Contribution`]を計算する。
    /// `TobitInput::from_columns`が`lower <= yᵢ <= upper`を保証しているため、境界値との
    /// 一致判定（`==`）で打ち切り観測を識別できる（`y*ᵢ`が境界値ちょうどになる確率は
    /// 連続分布の下で0のため、境界値ちょうどの観測は定義上すべて打ち切り観測）。
    fn contribution(&self, i: usize, params: &[f64], normal: &Normal) -> Contribution {
        let k = self.k();
        let s = params[k];
        let sigma = s.exp();
        let xb = self.linear_predictor(i, params);
        let y_i = *self.y.get(i, 0);

        if let Some(l) = self.lower
            && y_i == l
        {
            let zeta = (l - xb) / sigma;
            censored_contribution(normal, zeta, 1.0)
        } else if let Some(u) = self.upper
            && y_i == u
        {
            let zeta = (xb - u) / sigma;
            censored_contribution(normal, zeta, -1.0)
        } else {
            let v = (y_i - xb) / sigma;
            uncensored_contribution(v, s)
        }
    }

    /// 観測ごとのスコア行列（n×(k+1)、末尾の列が`s=logσ`成分）。各行が`sᵢ=∂ℓᵢ/∂θ`
    /// （対数尤度の1階微分そのもの、`Gradient`トレイトとは符号が逆）。OPG/サンドイッチ/
    /// クラスターSEの計算に使う（`LogitProblem::scores`/`ProbitProblem::scores`と同じ役割）。
    pub fn scores(&self, params: &[f64]) -> Mat<f64> {
        let n = self.x.nrows();
        let k = self.k();
        let sigma = params[k].exp();
        let normal = Normal::standard();
        Mat::from_fn(n, k + 1, |i, col| {
            let contribution = self.contribution(i, params, &normal);
            if col < k {
                contribution.score_beta_coef * (*self.x.get(i, col)) / sigma
            } else {
                contribution.score_s
            }
        })
    }
}

impl CostFunction for TobitProblem {
    type Param = Vec<f64>;
    type Output = f64;

    /// 負の対数尤度 `-ℓ(θ) = -Σᵢ ℓᵢ(θ)`（モジュール冒頭の数式表参照）。
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, OptimizerError> {
        let n = self.x.nrows();
        let normal = Normal::standard();
        let ll: f64 = (0..n)
            .map(|i| self.contribution(i, param, &normal).log_lik)
            .sum();
        Ok(-ll)
    }
}

impl Gradient for TobitProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    /// `-ℓ(θ)`の勾配（対数尤度のスコアの符号反転、モジュール冒頭の数式表参照）。
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, OptimizerError> {
        let n = self.x.nrows();
        let k = self.k();
        let sigma = param[k].exp();
        let normal = Normal::standard();
        let mut grad = vec![0.0; k + 1];
        for i in 0..n {
            let contribution = self.contribution(i, param, &normal);
            for (j, grad_j) in grad.iter_mut().enumerate().take(k) {
                *grad_j += -contribution.score_beta_coef * (*self.x.get(i, j)) / sigma;
            }
            grad[k] += -contribution.score_s;
        }
        Ok(grad)
    }
}

impl Hessian for TobitProblem {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    /// `-ℓ(θ)`のHessian（`(k+1)×(k+1)`、モジュール冒頭の数式表参照。`Contribution`の
    /// `h_*`フィールドは既に`-ℓᵢ`のHessian成分として定義済みのため、追加の符号反転は
    /// 不要）。`run_solver`のdocコメント「`Hessian`トレイトの符号規約」参照。
    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
        let n = self.x.nrows();
        let k = self.k();
        let sigma = param[k].exp();
        let sigma2 = sigma * sigma;
        let normal = Normal::standard();
        let mut h = vec![vec![0.0; k + 1]; k + 1];
        for i in 0..n {
            let contribution = self.contribution(i, param, &normal);
            for (a, row) in h.iter_mut().enumerate().take(k) {
                let xa = *self.x.get(i, a);
                for (b, cell) in row.iter_mut().enumerate().take(k) {
                    *cell += contribution.h_beta_coef * xa * (*self.x.get(i, b)) / sigma2;
                }
                // 対称性から`h[k][a]`も同じ値になるが、行`k`は`h.iter_mut()`の範囲外
                // （`take(k)`）のため、`row[k]`（`h[a][k]`）のみここで更新し、`h[k][a]`は
                // ループ後に一括で複製する（`h`全体を可変・不変で同時に借用できないため）。
                row[k] += contribution.h_beta_s_coef * xa / sigma;
            }
            h[k][k] += contribution.h_ss;
        }
        let cross_terms: Vec<f64> = h.iter().take(k).map(|row| row[k]).collect();
        h[k][..k].copy_from_slice(&cross_terms);
        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use statrs::distribution::{Continuous, ContinuousCDF};

    #[test]
    fn from_columns_with_intercept_prepends_const_column() {
        let y = vec![0.0, 1.0, 2.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0]];
        let input = TobitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap();

        assert_eq!(input.nobs(), 3);
        assert_eq!(input.k(), 2);
        assert!(input.has_intercept());
        assert_eq!(input.param_names(), ["const".to_string(), "x1".to_string()]);
        assert_eq!(input.dep_var_name(), "y");
        assert_eq!(input.lower(), Some(0.0));
        assert_eq!(input.upper(), None);
        for i in 0..3 {
            assert_eq!(*input.x().get(i, 0), 1.0);
            assert_eq!(*input.x().get(i, 1), x_columns[0][i]);
            assert_eq!(*input.y().get(i, 0), y[i]);
        }
    }

    #[test]
    fn from_columns_without_intercept_omits_const_column() {
        let y = vec![1.0, 0.0, 3.0, 0.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0], vec![4.0, 3.0, 2.0, 1.0]];
        let input = TobitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            false,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap();

        assert_eq!(input.k(), 2);
        assert!(!input.has_intercept());
        assert_eq!(input.param_names(), ["x1".to_string(), "x2".to_string()]);
    }

    #[test]
    fn from_columns_supports_right_censoring_only() {
        // lower=None（左側は打ち切りなし）でupperのみ指定する構成
        // （`nonlinear-api-design.md`7章の「右打ち切りのみ」）。
        let y = vec![-100.0, 5.0, 10.0];
        let input =
            TobitInput::from_columns(&y, &[], vec![], true, "y".to_string(), None, Some(10.0))
                .unwrap();

        assert_eq!(input.lower(), None);
        assert_eq!(input.upper(), Some(10.0));
    }

    #[test]
    fn from_columns_returns_y_out_of_censoring_bounds_above_upper_when_lower_is_none() {
        // lower=Noneのとき、y整合性検証のlower方向の分岐（`if let Some(l) = lower`）を
        // 経由せずupper方向の分岐だけが正しく機能することを確認する
        // （両方Someのテストだけではlower=Noneの経路を独立に検証できない）。
        let y = vec![-100.0, 5.0, 11.0];
        let result =
            TobitInput::from_columns(&y, &[], vec![], true, "y".to_string(), None, Some(10.0));

        assert_eq!(
            result.unwrap_err(),
            MleError::YOutOfCensoringBounds {
                row: 2,
                value: 11.0,
                lower: None,
                upper: Some(10.0),
            }
        );
    }

    #[test]
    fn from_columns_returns_invalid_censoring_bounds_when_lower_is_nan() {
        let y = vec![0.0, 1.0, 2.0];
        let result = TobitInput::from_columns(
            &y,
            &[],
            vec![],
            true,
            "y".to_string(),
            Some(f64::NAN),
            Some(10.0),
        );

        assert!(matches!(
            result.unwrap_err(),
            MleError::InvalidCensoringBounds {
                lower: Some(l),
                upper: Some(u)
            } if l.is_nan() && u == 10.0
        ));
    }

    #[test]
    fn from_columns_returns_invalid_censoring_bounds_when_upper_is_nan() {
        let y = vec![0.0, 1.0, 2.0];
        let result = TobitInput::from_columns(
            &y,
            &[],
            vec![],
            true,
            "y".to_string(),
            Some(0.0),
            Some(f64::NAN),
        );

        assert!(matches!(
            result.unwrap_err(),
            MleError::InvalidCensoringBounds {
                lower: Some(l),
                upper: Some(u)
            } if l == 0.0 && u.is_nan()
        ));
    }

    #[test]
    fn from_columns_returns_invalid_censoring_bounds_when_only_bound_is_nan() {
        // 片側のみ指定（upper=None）でも、`lower`単体がNaNなら弾かれることを確認する
        // （`(Some(l), None) => l.is_finite()`分岐、両方Someのケースとは別分岐）。
        let y = vec![0.0, 1.0, 2.0];
        let result =
            TobitInput::from_columns(&y, &[], vec![], true, "y".to_string(), Some(f64::NAN), None);

        assert!(matches!(
            result.unwrap_err(),
            MleError::InvalidCensoringBounds {
                lower: Some(l),
                upper: None
            } if l.is_nan()
        ));
    }

    #[test]
    fn from_columns_supports_two_sided_censoring() {
        let y = vec![0.0, 5.0, 10.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0]];
        let input = TobitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            Some(10.0),
        )
        .unwrap();

        assert_eq!(input.lower(), Some(0.0));
        assert_eq!(input.upper(), Some(10.0));
    }

    #[test]
    fn from_columns_returns_dimension_mismatch_on_mismatched_column_length() {
        let y = vec![0.0, 1.0, 2.0];
        let x_columns = vec![vec![1.0, 2.0]];
        let result = TobitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            None,
        );

        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::DimensionMismatch {
                y_rows: 3,
                x_rows: 2
            })
        );
    }

    #[test]
    #[should_panic(expected = "x_columns and x_names must have the same length")]
    fn from_columns_panics_on_mismatched_names_arity() {
        // x_names.len() != x_columns.len()はengine_pybind側の実装バグでしか
        // 起こり得ない内部契約違反のため、`debug_assert!`でパニックする
        // （`LogitInput::from_columns`と同じ方針）。
        let y = vec![0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0]];
        let _ = TobitInput::from_columns(
            &y,
            &x_columns,
            vec![],
            true,
            "y".to_string(),
            Some(0.0),
            None,
        );
    }

    #[test]
    fn from_columns_returns_invalid_censoring_bounds_when_both_none() {
        let y = vec![0.0, 1.0, 2.0];
        let result = TobitInput::from_columns(&y, &[], vec![], true, "y".to_string(), None, None);

        assert_eq!(
            result.unwrap_err(),
            MleError::InvalidCensoringBounds {
                lower: None,
                upper: None
            }
        );
    }

    #[test]
    fn from_columns_returns_invalid_censoring_bounds_when_lower_ge_upper() {
        let y = vec![0.0, 1.0, 2.0];
        let result =
            TobitInput::from_columns(&y, &[], vec![], true, "y".to_string(), Some(5.0), Some(5.0));

        assert_eq!(
            result.unwrap_err(),
            MleError::InvalidCensoringBounds {
                lower: Some(5.0),
                upper: Some(5.0)
            }
        );
    }

    #[test]
    fn from_columns_returns_y_out_of_censoring_bounds_when_y_below_lower() {
        let y = vec![0.0, -1.0, 2.0];
        let result =
            TobitInput::from_columns(&y, &[], vec![], true, "y".to_string(), Some(0.0), None);

        assert_eq!(
            result.unwrap_err(),
            MleError::YOutOfCensoringBounds {
                row: 1,
                value: -1.0,
                lower: Some(0.0),
                upper: None,
            }
        );
    }

    #[test]
    fn from_columns_returns_y_out_of_censoring_bounds_when_y_above_upper() {
        let y = vec![0.0, 5.0, 11.0];
        let result = TobitInput::from_columns(
            &y,
            &[],
            vec![],
            true,
            "y".to_string(),
            Some(0.0),
            Some(10.0),
        );

        assert_eq!(
            result.unwrap_err(),
            MleError::YOutOfCensoringBounds {
                row: 2,
                value: 11.0,
                lower: Some(0.0),
                upper: Some(10.0),
            }
        );
    }

    #[test]
    fn from_columns_accepts_y_exactly_at_bounds() {
        // 境界値ちょうど（lower/upperそのもの）は打ち切り観測として正常な値のため、
        // 厳密な `<`/`>` 比較で弾かれないことを確認する（`<=`/`>=`との取り違え回帰防止）。
        let y = vec![0.0, 5.0, 10.0];
        let result = TobitInput::from_columns(
            &y,
            &[],
            vec![],
            true,
            "y".to_string(),
            Some(0.0),
            Some(10.0),
        );

        assert!(result.is_ok());
    }

    /// n=4, k=2（切片+x1）。`y=[0.0, 2.0, 5.0, 3.0]`・`lower=Some(0.0)`・`upper=Some(5.0)`で、
    /// 観測0が左打ち切り・観測2が右打ち切り・観測1,3が非打ち切りという3分類すべてを含む
    /// （`Contribution`の3分岐・左右の符号反転を1つのテストデータで網羅的に検証するため）。
    fn mixed_censoring_input() -> TobitInput {
        let y = vec![0.0, 2.0, 5.0, 3.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0]];
        TobitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            Some(5.0),
        )
        .unwrap()
    }

    #[test]
    fn gradient_matches_numerical_differentiation_of_cost() {
        let input = mixed_censoring_input();
        let problem = TobitProblem::new(&input);
        let params = vec![0.5, 0.3, -0.2];
        let h = 1e-6;

        let analytic = problem.gradient(&params).unwrap();
        for j in 0..3 {
            let mut plus = params.clone();
            plus[j] += h;
            let mut minus = params.clone();
            minus[j] -= h;
            let numeric =
                (problem.cost(&plus).unwrap() - problem.cost(&minus).unwrap()) / (2.0 * h);
            assert!(
                (analytic[j] - numeric).abs() < 1e-6,
                "j={j}, analytic={}, numeric={}",
                analytic[j],
                numeric
            );
        }
    }

    #[test]
    fn hessian_matches_numerical_differentiation_of_gradient() {
        let input = mixed_censoring_input();
        let problem = TobitProblem::new(&input);
        let params = vec![0.5, 0.3, -0.2];
        let h = 1e-5;

        let analytic = problem.hessian(&params).unwrap();
        for j in 0..3 {
            let mut plus = params.clone();
            plus[j] += h;
            let mut minus = params.clone();
            minus[j] -= h;
            let grad_plus = problem.gradient(&plus).unwrap();
            let grad_minus = problem.gradient(&minus).unwrap();
            for i in 0..3 {
                let numeric = (grad_plus[i] - grad_minus[i]) / (2.0 * h);
                assert!(
                    (analytic[i][j] - numeric).abs() < 1e-4,
                    "i={i}, j={j}, analytic={}, numeric={}",
                    analytic[i][j],
                    numeric
                );
            }
        }
    }

    #[test]
    fn scores_sum_to_negative_gradient() {
        // scoresは対数尤度の生のスコア（符号反転なし）、gradientはCostFunction
        // （負の対数尤度）の勾配のため、観測方向に合計すると符号が逆になるはず
        // （LogitProblem/ProbitProblemの同名テストと同じ検証）。
        let input = mixed_censoring_input();
        let problem = TobitProblem::new(&input);
        let params = vec![0.5, 0.3, -0.2];

        let scores = problem.scores(&params);
        let grad = problem.gradient(&params).unwrap();
        let n = scores.nrows();

        for j in 0..3 {
            let sum: f64 = (0..n).map(|i| *scores.get(i, j)).sum();
            assert!(
                (sum - (-grad[j])).abs() < 1e-9,
                "j={j}, sum={sum}, grad={:?}",
                grad
            );
        }
    }

    #[test]
    fn cost_matches_closed_form_normal_log_likelihood_when_no_observation_is_censored() {
        // 境界を極端に広く取り、全観測が非打ち切りになるデータ。この場合Tobitの対数尤度は
        // 通常の正規回帰の対数尤度に一致するはず（Issue #214完了条件「打ち切りなし観測のみの
        // データで、cost/gradient/hessianがOLSの対数尤度と整合すること」の境界ケース検算）。
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let input = TobitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(-100.0),
            Some(100.0),
        )
        .unwrap();
        let problem = TobitProblem::new(&input);
        let params = vec![0.5, 0.3, -0.2];

        let cost = problem.cost(&params).unwrap();

        let sigma = params[2].exp();
        let n = y.len() as f64;
        let sse: f64 = (0..y.len())
            .map(|i| {
                let xb = params[0] + params[1] * x_columns[0][i];
                (y[i] - xb).powi(2)
            })
            .sum();
        let expected_ll = -(n / 2.0) * (2.0 * std::f64::consts::PI).ln()
            - n * sigma.ln()
            - sse / (2.0 * sigma * sigma);

        assert!(
            (cost - (-expected_ll)).abs() < 1e-9,
            "cost={cost}, expected={}",
            -expected_ll
        );
    }

    #[test]
    fn cost_gradient_hessian_stay_finite_for_extreme_linear_predictor() {
        // x1=1000のとき、打ち切り観測のzeta=(lower-x'β)/σが極端な値になり、クランプが
        // 無ければclamped_pdf_cdf内のφ(u)/Φ(u)が0.0/0.0のNaNになりうる
        // （Probitのcost_gradient_hessian_stay_finite_for_extreme_linear_predictorと
        // 同じ数値安定化の検証、モジュール冒頭「数値安定化について」参照）。
        let y = vec![0.0, 5.0];
        let x_columns = vec![vec![1000.0, 1000.0]];
        let input = TobitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            Some(5.0),
        )
        .unwrap();
        let problem = TobitProblem::new(&input);
        let params = vec![0.0, 1.0, 0.0];

        let cost = problem.cost(&params).unwrap();
        assert!(cost.is_finite(), "cost={cost}");

        let grad = problem.gradient(&params).unwrap();
        for (j, g) in grad.iter().enumerate() {
            assert!(g.is_finite(), "grad[{j}]={g}");
        }

        let hessian = problem.hessian(&params).unwrap();
        for (i, row) in hessian.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                assert!(v.is_finite(), "hessian[{i}][{j}]={v}");
            }
        }
    }

    /// 打ち切り分岐（`censored_contribution`）を、数値微分ではなく`φ`/`Φ`の
    /// プリミティブから独立に再計算した値と直接突き合わせる（rust-reviewer指摘:
    /// 数値微分だけでは`log_lik`/`score_*`/`h_*`に共通するスケール・符号誤りを
    /// 検出できないため）。`z=(lower-x'β)/σ=0`となるよう`β0=0`・`σ=1`（`s=0`）・
    /// `lower=0`を選ぶと、`Φ(0)=0.5`・`φ(0)=1/√(2π)`という暗算しやすい値になる。
    #[test]
    fn left_censored_cost_gradient_hessian_scores_match_closed_form_at_zeta_zero() {
        let y = vec![0.0];
        let input =
            TobitInput::from_columns(&y, &[], vec![], true, "y".to_string(), Some(0.0), None)
                .unwrap();
        let problem = TobitProblem::new(&input);
        let params = vec![0.0, 0.0]; // β0=0, s=0(σ=1) → z=(0-0)/1=0

        let normal = Normal::standard();
        let phi0 = normal.pdf(0.0);
        let big_phi0 = normal.cdf(0.0);
        let lambda0 = phi0 / big_phi0;
        let a0 = lambda0 * (0.0 + lambda0);
        let c0 = 0.0 * a0 - lambda0;

        let cost = problem.cost(&params).unwrap();
        assert!((cost - (-big_phi0.ln())).abs() < 1e-12, "cost={cost}");

        // 左打ち切り: direction=1.0 → cost_grad_beta_coef=direction*lambda0=lambda0
        let grad = problem.gradient(&params).unwrap();
        assert!((grad[0] - lambda0).abs() < 1e-12, "grad[0]={}", grad[0]);
        assert!(grad[1].abs() < 1e-12, "grad[1]={}", grad[1]);

        let hessian = problem.hessian(&params).unwrap();
        assert!((hessian[0][0] - a0).abs() < 1e-12, "h00={}", hessian[0][0]);
        assert!((hessian[0][1] - c0).abs() < 1e-12, "h01={}", hessian[0][1]);
        assert!((hessian[1][0] - c0).abs() < 1e-12, "h10={}", hessian[1][0]);
        assert!(hessian[1][1].abs() < 1e-12, "h11={}", hessian[1][1]);

        // scoresは生スコア: score_beta_coef=-direction*lambda0=-lambda0
        let scores = problem.scores(&params);
        assert!((*scores.get(0, 0) - (-lambda0)).abs() < 1e-12);
        assert!((*scores.get(0, 1)).abs() < 1e-12);
    }

    /// `left_censored_cost_gradient_hessian_scores_match_closed_form_at_zeta_zero`の
    /// 右打ち切り版。`w=(x'β-upper)/σ=0`となるよう`upper=0`を選ぶ（`zeta`の値自体は
    /// 左打ち切り版と同じ0だが、`direction=-1.0`によりβ成分の符号が反転することを検証する）。
    #[test]
    fn right_censored_cost_gradient_hessian_scores_match_closed_form_at_zeta_zero() {
        let y = vec![0.0];
        let input =
            TobitInput::from_columns(&y, &[], vec![], true, "y".to_string(), None, Some(0.0))
                .unwrap();
        let problem = TobitProblem::new(&input);
        let params = vec![0.0, 0.0]; // β0=0, s=0(σ=1) → w=(0-0)/1=0

        let normal = Normal::standard();
        let phi0 = normal.pdf(0.0);
        let big_phi0 = normal.cdf(0.0);
        let lambda0 = phi0 / big_phi0;
        let a0 = lambda0 * (0.0 + lambda0);
        let c0 = 0.0 * a0 - lambda0;

        let cost = problem.cost(&params).unwrap();
        assert!((cost - (-big_phi0.ln())).abs() < 1e-12, "cost={cost}");

        // 右打ち切り: direction=-1.0 → cost_grad_beta_coef=direction*lambda0=-lambda0
        // （左打ち切り版とちょうど符号が反転する）
        let grad = problem.gradient(&params).unwrap();
        assert!((grad[0] - (-lambda0)).abs() < 1e-12, "grad[0]={}", grad[0]);
        assert!(grad[1].abs() < 1e-12, "grad[1]={}", grad[1]);

        let hessian = problem.hessian(&params).unwrap();
        assert!((hessian[0][0] - a0).abs() < 1e-12, "h00={}", hessian[0][0]);
        assert!(
            (hessian[0][1] - (-c0)).abs() < 1e-12,
            "h01={}",
            hessian[0][1]
        );
        assert!(
            (hessian[1][0] - (-c0)).abs() < 1e-12,
            "h10={}",
            hessian[1][0]
        );
        assert!(hessian[1][1].abs() < 1e-12, "h11={}", hessian[1][1]);

        let scores = problem.scores(&params);
        assert!((*scores.get(0, 0) - lambda0).abs() < 1e-12);
        assert!((*scores.get(0, 1)).abs() < 1e-12);
    }
}
