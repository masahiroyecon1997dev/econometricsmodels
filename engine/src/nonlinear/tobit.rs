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
//!
//! ## Newton法の初期値: OLS推定値を使う（ゼロベクトルではない、Logit/Probitとの違い）
//!
//! `docs/planning/specs/nonlinear-implementation-notes.md`の当初計画では、`(β, logσ)`
//! パラメータ化でもLogit/Probitと同じくゼロベクトル初期値からのNewton収束をまず試す
//! 方針だったが、Issue #215の実装時に**打ち切りが皆無のデータ（通常の正規回帰に退化する
//! 単純なケース）でもゼロベクトル初期値からNewtonが発散する**ことが実測で判明した
//! （切片`β→-∞`・`s=logσ→+∞`という非有界な方向へ発散し、最終的にHessianが0行列に
//! なり`NaN`に到達。ユーザー確認済み）。
//!
//! 原因はLogit/Probitと異なり`(β, logσ)`パラメータ化のTobit尤度が大域凹であることを
//! 保証されていない点（Olsen(1978)の`(β/σ, 1/σ)`変換を採用していないため、
//! モジュール冒頭「数式」節参照）。初期値`σ=1`が真の値から大きく離れていると、
//! Newtonの2次近似が悪条件な領域（Hessianが不定符号）を通過し、そこでの
//! 局所線形近似が誤った方向への大きなステップを生む。
//!
//! 対策として、`ols_initial_params`で打ち切りを無視した単純なOLS（`β`とその残差の
//! 標本標準偏差）を計算し、Newtonの初期値として使う（R `survreg`/`censReg`等の
//! 標準的なTobit実装と同じ方針）。真の最尤推定値に近い出発点から始めることで、
//! 上記の不安定な領域を通過するリスクを大きく下げる。
//!
//! ## Newtonステップの正則化（Levenberg-Marquardt型、OLS初期値だけでは不十分だった）
//!
//! OLS初期値の導入後も、**実際に打ち切りが発生するデータ**（左右どちらかの打ち切り率が
//! 有意にあるデータ。打ち切りが皆無または無視できる場合は上記のOLS初期値のみで
//! 十分だった）では、なおNewton法が`SingularHessian`で失敗することが判明した
//! （n=8の小規模データ、n=30・打ち切り率40%の穏やかなデータの両方で再現。実測で
//! 手動追跡した結果、Hessianが不定符号になる領域で**生のNewtonステップが降下方向で
//! さえない**（ステップをどれだけ小さくスケールしても`cost`が改善しない）ことを確認した）。
//!
//! `nonlinear/common.rs`の`FaerNewton`（Logit/Probitと共有）に、Levenberg-Marquardt型の
//! 減衰Newton法（`regularized_newton_step`）を追加して対応した。`H+λI`（`λ≥0`）で
//! ステップを求め、`cost`が減少する候補が見つかるまで`λ`を段階的に増やす。
//! Logit/Probitのように尤度が大域凹な問題では、収束点に向かう正常な軌道上は`λ=0`の
//! 生のステップが最初の試行で受理されるため収束の挙動は変わらないが、設計行列が
//! 構造的に特異な入力（完全な多重共線性等）では内部の反復過程が変化する
//! （最終的に`fit()`が返すエラーバリアントは変わらない。詳細は
//! `regularized_newton_step`のdocコメント参照、rust-reviewer指摘・独立シミュレーションで
//! 確認済み）。Logit/Probitの既存テストが数値的に変化なく全通過することは確認済み。
//! 詳細な導入経緯は
//! `regularized_newton_step`のdocコメント参照（ユーザー確認済み）。

use crate::error::CommonError;
use crate::inference;
use crate::nonlinear::common::{
    Method, MleError, clamped_pdf_cdf, destandardize_cov_params, destandardize_params,
    observed_information_cov_params, run_solver, standardize_columns, validate_confidence_level,
    validate_max_iter, validate_sufficient_observations, validate_tol,
};
use argmin::core::{CostFunction, Error as OptimizerError, Gradient, Hessian};
use faer::Mat;
use faer::prelude::SolveLstsq;
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
    /// 同じ位置づけ。`TobitEstimator::fit`は標準化後の設計行列を使うため、こちらではなく
    /// `from_standardized`を使う）。
    #[cfg(test)]
    fn new(input: &TobitInput) -> Self {
        Self {
            x: input.x().clone(),
            y: input.y().clone(),
            lower: input.lower(),
            upper: input.upper(),
        }
    }

    /// `standardize_columns`で標準化済みの設計行列`x_std`と`y`・打ち切り境界から構築する。
    /// `TobitEstimator::fit`が最適化に使う経路（`LogitProblem::from_standardized`と
    /// 同じ位置づけ）。打ち切り境界`lower`/`upper`は`y`のスケールで表現された値であり
    /// `x`の列スケーリングとは無関係なため、標準化の影響を受けずそのまま渡す。
    fn from_standardized(
        x_std: Mat<f64>,
        y: Mat<f64>,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> Self {
        Self {
            x: x_std,
            y,
            lower,
            upper,
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

/// 打ち切りを無視した単純なOLS（`X'Xβ=X'y`の最小二乗解）から、Newton法の初期値
/// `(β, logσ)`を計算する（モジュール冒頭「Newton法の初期値」節参照）。`x`は
/// `standardize_columns`で標準化済みの設計行列を渡す想定（`fit()`が最適化に使う空間と
/// 一致させ、初期値をそのまま`run_solver`に渡せるようにするため）。`y`は打ち切りを
/// 無視し、観測された値をそのまま連続値として扱う（あくまで初期値のヒューリスティックで
/// あり、Tobitの推定値そのものではない）。
///
/// 特異性検出は`engine::linear::ols::OlsEstimator`の`ensure_full_rank`と同じ相対閾値
/// パターン（列ピボットQRの`R`の対角成分、`.claude/rules/rust-style.md`「線形代数」
/// 参照）を踏襲する。`OlsEstimator`側のprivate関数は再利用できないため同型のロジックを
/// ここに複製している。
///
/// # Errors
/// `x`が特異（完全な多重共線性等）な場合は`MleError::SingularDesignMatrix`を返す
/// （`MleError::SingularDesignMatrix`のdocコメント参照。Newton法が一度も反復していない
/// 段階で発生しうるため`SingularHessian`とは区別する）。
fn ols_initial_params(x: &Mat<f64>, y: &Mat<f64>) -> Result<Vec<f64>, MleError> {
    let n = x.nrows();
    let k = x.ncols();

    let qr = x.col_piv_qr();
    let r = qr.thin_R();
    let max_abs_diag = (0..k).map(|i| (*r.get(i, i)).abs()).fold(0.0_f64, f64::max);
    let threshold = (k as f64) * f64::EPSILON * max_abs_diag;
    for i in 0..k {
        let diag = (*r.get(i, i)).abs();
        if diag.is_nan() || diag <= threshold {
            return Err(MleError::SingularDesignMatrix);
        }
    }

    let beta_mat = qr.solve_lstsq(y);
    let beta: Vec<f64> = (0..k).map(|i| *beta_mat.get(i, 0)).collect();

    let residuals = y - x * &beta_mat;
    let sse: f64 = (0..n).map(|i| (*residuals.get(i, 0)).powi(2)).sum();
    // 完全な当てはまり（sse≈0、退化ケース）ではlog(0)=-infになり最適化が破綻するため、
    // あくまで初期値のヒューリスティックとしてσ=1（s=0）にフォールバックする。
    // `sse > 0.0`という厳密比較ではなく`y`のスケールに対する相対閾値を使う
    // （`.claude/rules/rust-style.md`「線形代数」の相対閾値方針。QRによる最小二乗解の
    // 浮動小数点丸め誤差により、数学的には完全な当てはまりでも`sse`がわずかに正の
    // 微小値になりうるため、`> 0.0`だと退化ケースを見逃す）。
    let candidate_sigma = (sse / n as f64).sqrt();
    let y_scale = (0..n)
        .map(|i| (*y.get(i, 0)).abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let sigma = if candidate_sigma > 1e-8 * y_scale {
        candidate_sigma
    } else {
        1.0
    };

    let mut params = beta;
    params.push(sigma.ln());
    Ok(params)
}

/// Tobitの推定結果。`fit`でのバリデーション・最適化・観測情報行列によるSE計算を
/// 通過した状態を表す。`cov_type`（OPG/サンドイッチ/クラスター）・適合度統計量・
/// 限界効果は後続issue（#218以降）で追加する（`LogitEstimator`/`ProbitEstimator`との違い。
/// Logitも`cov_type`パラメータの追加は#59、観測情報行列によるSEは#58と段階的に実装した
/// 経緯があり、同じ分割方針を踏襲している）。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug)]
pub struct TobitEstimator {
    input: TobitInput,
    /// 係数（元のスケール、`β`部分のみ）。`input.param_names()`と対応する
    params: Vec<f64>,
    /// 誤差項の標準偏差（元のスケール）。内部最適化パラメータ`logσ`を`σ=exp(logσ)`で
    /// 逆変換したもの（`.claude/rules/rust-style.md`の`ColumnScale::extend_unscaled`
    /// docコメント参照。`logσ`は`x`の列スケーリングとは無関係な量なので、逆変換は
    /// この指数変換のみで完結する）
    sigma: f64,
    /// `(β, σ)`の分散共分散行列（元のスケール、`(k+1)×(k+1)`）。現時点では常に観測情報
    /// 行列（`Σ = -H⁻¹`、`cov_type="classical"`/`"nonrobust"`相当）。最適化は内部
    /// パラメータ化`(β, s=logσ)`空間のHessianで行われるため、`observed_information_
    /// cov_params`・`destandardize_cov_params`で`(β, s)`空間の分散共分散行列を得た後、
    /// デルタ法のヤコビアン`diag(1,...,1,σ)`（`s`から`σ=exp(s)`への変換、`dσ/ds=σ`）を
    /// 両側から適用して`(β, σ)`空間へ変換する（`k+1`行目・列目が`σ`に対応、
    /// `β`部分の`k×k`ブロックはヤコビアンが恒等写像のため無変換）。`β`-`σ`間の
    /// 共分散も含めて変換するため、対角成分だけでなく行列全体が一貫した値になる
    /// （`fit`のdocコメント「`σ`のSE」節参照。ユーザー確認済み、単なる対角成分の
    /// `Var(σ)≈σ²Var(logσ)`だけでなく将来の限界効果等での再利用を見据えてフル行列を
    /// 変換する設計を採用）。
    cov_params: Mat<f64>,
    /// 標準誤差（`k+1`、元のスケール、`β∪{σ}`。`k`番目の要素が`σ`に対応）。
    /// `cov_params`の対角成分の平方根
    std_errors: Vec<f64>,
    /// z統計量（`k+1`）= `coef / std_errors`（`β∪{σ}`それぞれについて）
    z_stats: Vec<f64>,
    /// 両側p値（`k+1`）。標準正規分布に基づく
    p_values: Vec<f64>,
    /// 信頼区間の下限（`k+1`）
    conf_lower: Vec<f64>,
    /// 信頼区間の上限（`k+1`）
    conf_upper: Vec<f64>,
    /// 収束したかどうか
    converged: bool,
    /// 実際の反復回数
    n_iter: usize,
}

impl TobitEstimator {
    /// `method`（Newton-Raphson/BFGS/L-BFGS）で負の対数尤度を最小化し、Tobitの係数`β`・
    /// 誤差項の標準偏差`σ`を推定する。
    ///
    /// 内部最適化パラメータは`(β, s=logσ)`という`k+1`次元ベクトル（モジュール冒頭の
    /// 数式参照）。`x`は`standardize_columns`で内部的に標準化してから最適化し
    /// （`LogitEstimator::fit`と同じ理由。勾配ノルムに基づく収束判定`tol`が設計行列の
    /// スケールに依存しないようにするため）、収束後に`destandardize_params`で元の
    /// スケールへ逆変換する。`logσ`（`k+1`番目の要素）は`x`の列スケーリングとは無関係な
    /// 量（線形再パラメータ化`x_std=x/std`の下で不変）なので、`ColumnScale::
    /// extend_unscaled(1)`でスケール`1.0`（無変換）の要素を追加し、既存の`zip`ベースの
    /// `destandardize_params`をそのまま`(k+1)`次元ベクトルに適用する
    /// （`docs/planning/specs/nonlinear-implementation-notes.md`「standardize_columnsと
    /// σの扱い」節）。打ち切り境界`lower`/`upper`はこの標準化の影響を受けない
    /// （`y`のスケールで表現された値のため）。
    ///
    /// 初期値はゼロベクトルではなく`ols_initial_params`が計算するOLS推定値
    /// （`LogitEstimator::fit`とは異なる。モジュール冒頭「Newton法の初期値」節参照）。
    ///
    /// `cov_type`は本Issue（#217）では扱わない。常に観測情報行列（`Σ=-H⁻¹`、
    /// `cov_type="classical"`/`"nonrobust"`相当）を使う（OPG/サンドイッチ/クラスターは
    /// #218・#219で追加）。観測数の十分性検証（`validate_sufficient_observations`）には
    /// `x`の列数`k`ではなく総最適化パラメータ数`k+1`を使う（Issue #212の結論、
    /// `validate_sufficient_observations`のdocコメント参照）。Logit/Probitの
    /// `validate_has_regressors`（`k==0`検証）はTobitでは呼ばない（`logσ`が常に存在する
    /// ため対応するケースが生じない、同関数のdocコメント参照）。適合度統計量・限界効果は
    /// 後続issue（#220・#221）で追加する。
    ///
    /// ## `σ`のSE（デルタ法）
    ///
    /// `run_solver`が返すHessianは内部パラメータ化`(β, s=logσ)`空間で評価されたもの
    /// （`standardize_columns`で標準化済みの`x`に対応する標準化空間でもある）。
    /// `observed_information_cov_params`・`destandardize_cov_params`で`x`の標準化のみを
    /// 解いて`(β, s)`空間（元のスケール）の分散共分散行列を得た後、`s→σ=exp(s)`の
    /// デルタ法変換（ヤコビアン`diag(1,...,1,σ)`、`dσ/ds=σ`）を分散共分散行列全体に
    /// 適用して`(β, σ)`空間へ変換する（`cov_params`のdocコメント参照。対角成分のみ見ると
    /// `Var(σ)≈σ²Var(logσ)`という`docs/planning/specs/nonlinear-implementation-notes.md`
    /// 「パラメータ化」節に記載の式に一致する）。
    ///
    /// # Errors
    /// - `confidence_level`が`(0, 1)`の範囲外: `CommonError::InvalidConfidenceLevel`
    /// - `max_iter`が0以下: `MleError::InvalidMaxIter`
    /// - `tol`が0以下: `MleError::InvalidTol`
    /// - 観測数`n`が総パラメータ数`k+1`以下: `CommonError::InsufficientObservations`
    /// - OLS初期値計算時に`x`が特異（完全な多重共線性等）: `MleError::SingularDesignMatrix`
    ///   （`ols_initial_params`参照）
    /// - `raise_on_non_convergence=true`かつ`max_iter`回で未収束: `MleError::NonConvergence`
    /// - 収束点（または`raise_on_non_convergence=false`時の打ち切り点）のHessianが特異:
    ///   `MleError::SingularHessian`
    pub fn fit(
        input: TobitInput,
        method: Method,
        max_iter: i64,
        tol: f64,
        raise_on_non_convergence: bool,
        confidence_level: f64,
    ) -> Result<Self, MleError> {
        validate_confidence_level(confidence_level)?;
        validate_max_iter(max_iter)?;
        validate_tol(tol)?;

        let n = input.nobs();
        let k = input.k();
        validate_sufficient_observations(n, k + 1)?;

        let (x_std, scale) = standardize_columns(input.x(), input.has_intercept());
        let scale = scale.extend_unscaled(1);
        let initial_params = ols_initial_params(&x_std, input.y())?;
        let problem =
            TobitProblem::from_standardized(x_std, input.y().clone(), input.lower(), input.upper());

        let output = run_solver(
            problem,
            method,
            initial_params,
            max_iter as u64,
            tol,
            raise_on_non_convergence,
        )?;

        let params_full = destandardize_params(&output.params, &scale);
        let sigma = params_full[k].exp();
        let params = params_full[..k].to_vec();

        let hessian_std = Mat::from_fn(k + 1, k + 1, |i, j| output.hessian[i][j]);
        let cov_params_beta_s = destandardize_cov_params(
            &observed_information_cov_params(&hessian_std, k + 1)?,
            &scale,
        );
        // `s=logσ→σ=exp(s)`のデルタ法ヤコビアン。`β`部分は無変換（恒等写像）なので`1.0`、
        // `k`番目（`s`/`σ`に対応する行・列）だけ`dσ/ds=σ`を掛ける。
        let jacobian: Vec<f64> = (0..=k).map(|i| if i < k { 1.0 } else { sigma }).collect();
        let cov_params = Mat::from_fn(k + 1, k + 1, |i, j| {
            *cov_params_beta_s.get(i, j) * jacobian[i] * jacobian[j]
        });

        let normal = Normal::standard();
        let z_crit = inference::critical_value(&normal, confidence_level);

        let mut std_errors = vec![0.0; k + 1];
        let mut z_stats = vec![0.0; k + 1];
        let mut p_values = vec![0.0; k + 1];
        let mut conf_lower = vec![0.0; k + 1];
        let mut conf_upper = vec![0.0; k + 1];

        for j in 0..=k {
            let coef = if j < k { params[j] } else { sigma };
            let se = (*cov_params.get(j, j)).sqrt();
            let stat = inference::compute_inference_stat(&normal, coef, se, z_crit);

            std_errors[j] = se;
            z_stats[j] = stat.stat;
            p_values[j] = stat.p_value;
            conf_lower[j] = stat.conf_low;
            conf_upper[j] = stat.conf_high;
        }

        Ok(Self {
            input,
            params,
            sigma,
            cov_params,
            std_errors,
            z_stats,
            p_values,
            conf_lower,
            conf_upper,
            converged: output.converged,
            n_iter: output.n_iter,
        })
    }

    /// 推定に使った入力データ
    pub fn input(&self) -> &TobitInput {
        &self.input
    }

    /// 係数（元のスケール、`β`部分のみ）
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    /// 誤差項の標準偏差（元のスケール）
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// `(β, σ)`の分散共分散行列（元のスケール、`(k+1)×(k+1)`。`k`番目の行・列が`σ`に対応）
    pub fn cov_params(&self) -> &Mat<f64> {
        &self.cov_params
    }

    /// 標準誤差（`k+1`、元のスケール、`β∪{σ}`。`k`番目の要素が`σ`に対応）
    pub fn std_errors(&self) -> &[f64] {
        &self.std_errors
    }

    /// z統計量（`k+1`）
    pub fn z_stats(&self) -> &[f64] {
        &self.z_stats
    }

    /// 両側p値（`k+1`）
    pub fn p_values(&self) -> &[f64] {
        &self.p_values
    }

    /// 信頼区間の下限（`k+1`）
    pub fn conf_lower(&self) -> &[f64] {
        &self.conf_lower
    }

    /// 信頼区間の上限（`k+1`）
    pub fn conf_upper(&self) -> &[f64] {
        &self.conf_upper
    }

    /// 収束したかどうか
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// 実際の反復回数
    pub fn n_iter(&self) -> usize {
        self.n_iter
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

    /// 打ち切りが（実質的に）発生しないよう境界を極端に広く取ったデータ。
    fn intercept_only_uncensored_input(y: &[f64]) -> TobitInput {
        TobitInput::from_columns(
            y,
            &[],
            vec![],
            true,
            "y".to_string(),
            Some(-1000.0),
            Some(1000.0),
        )
        .unwrap()
    }

    /// 切片のみ（説明変数なし）・打ち切りなしのTobitは、通常の正規分布の最尤推定
    /// （`β̂=ȳ`・`σ̂²=Σ(y-ȳ)²/n`という閉じた形の解析解、OLSの不偏推定量`n-1`除算とは
    /// 異なる`n`除算のML推定量）に一致するはず（Issue #215完了条件「打ち切りが極端に
    /// 少ないデータで、Newton法がOLSの閉形式解に近い値に収束すること」の境界ケース）。
    #[test]
    fn fit_newton_converges_to_closed_form_solution_for_intercept_only_uncensored_data() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 10.0];
        let input = intercept_only_uncensored_input(&y);

        let n = y.len() as f64;
        let y_bar: f64 = y.iter().sum::<f64>() / n;
        let sse: f64 = y.iter().map(|v| (v - y_bar).powi(2)).sum();
        let expected_sigma = (sse / n).sqrt();

        let estimator = TobitEstimator::fit(input, Method::Newton, 100, 1e-8, true, 0.95).unwrap();

        assert!(estimator.converged());
        assert_eq!(estimator.params().len(), 1);
        assert!(
            (estimator.params()[0] - y_bar).abs() < 1e-5,
            "params={:?}, expected={}",
            estimator.params(),
            y_bar
        );
        assert!(
            (estimator.sigma() - expected_sigma).abs() < 1e-5,
            "sigma={}, expected={}",
            estimator.sigma(),
            expected_sigma
        );
    }

    /// 切片のみ・打ち切りなしのTobit（`(β,s=logσ)`空間の対数尤度が通常の正規分布の
    /// 対数尤度に一致する）は、観測情報行列も閉じた形で書ける: `∂²ℓ/∂β²=-n/σ²`・
    /// `∂²ℓ/∂s²=-2n`・`∂²ℓ/∂β∂s=0`（MLE点で`Σ(yᵢ-β̂)=0`となるため）という対角行列になり、
    /// `Var(β̂)=σ̂²/n`・`Var(ŝ)=1/(2n)`・`Cov(β̂,ŝ)=0`が導ける（`docs/planning/specs/
    /// nonlinear-implementation-notes.md`「パラメータ化」節の一般形の特殊ケース）。
    /// `σ`のデルタ法変換（ヤコビアン`diag(1,σ)`）を適用すると、この対角性はそのまま
    /// 保たれ`Var(σ̂)≈σ̂²Var(ŝ)=σ̂²/(2n)`・`Cov(β̂,σ̂)≈σ̂Cov(β̂,ŝ)=0`になる。
    /// Logitの`fit_computes_std_errors_z_stats_p_values_and_ci_matching_closed_form_for_
    /// intercept_only_model`と同じ設計（本体実装と同じ式を繰り返すのではなく、独立に
    /// 導出した閉じた形の解でSE・z値・p値・信頼区間を検算する）。
    #[test]
    fn fit_computes_std_errors_z_stats_p_values_and_ci_matching_closed_form_for_intercept_only_uncensored_model()
     {
        let y = vec![1.0, 2.0, 3.0, 4.0, 10.0];
        let input = intercept_only_uncensored_input(&y);

        let n = y.len() as f64;
        let y_bar: f64 = y.iter().sum::<f64>() / n;
        let sse: f64 = y.iter().map(|v| (v - y_bar).powi(2)).sum();
        let sigma_hat = (sse / n).sqrt();

        let expected_var_beta = sigma_hat.powi(2) / n;
        let expected_var_sigma = sigma_hat.powi(2) / (2.0 * n);

        let estimator = TobitEstimator::fit(input, Method::Newton, 100, 1e-8, true, 0.95).unwrap();

        assert!((*estimator.cov_params().get(0, 0) - expected_var_beta).abs() < 1e-6);
        assert!((*estimator.cov_params().get(1, 1) - expected_var_sigma).abs() < 1e-6);
        assert!(
            estimator.cov_params().get(0, 1).abs() < 1e-6,
            "cov(beta,sigma)={}",
            estimator.cov_params().get(0, 1)
        );

        let expected_se_beta = expected_var_beta.sqrt();
        let expected_se_sigma = expected_var_sigma.sqrt();
        assert!((estimator.std_errors()[0] - expected_se_beta).abs() < 1e-6);
        assert!((estimator.std_errors()[1] - expected_se_sigma).abs() < 1e-6);

        // z値・p値・信頼区間はstatrsのNormalで独立に検算する（本体実装と同じ計算式を
        // 繰り返すのではなく、標準正規分布の性質から直接導出する）。
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z_crit = normal.inverse_cdf(0.975);

        let expected_z_beta = estimator.params()[0] / expected_se_beta;
        let expected_p_beta = 2.0 * (1.0 - normal.cdf(expected_z_beta.abs()));
        assert!((estimator.z_stats()[0] - expected_z_beta).abs() < 1e-6);
        assert!((estimator.p_values()[0] - expected_p_beta).abs() < 1e-6);
        assert!(
            (estimator.conf_lower()[0] - (estimator.params()[0] - z_crit * expected_se_beta)).abs()
                < 1e-6
        );
        assert!(
            (estimator.conf_upper()[0] - (estimator.params()[0] + z_crit * expected_se_beta)).abs()
                < 1e-6
        );

        let expected_z_sigma = estimator.sigma() / expected_se_sigma;
        let expected_p_sigma = 2.0 * (1.0 - normal.cdf(expected_z_sigma.abs()));
        assert!((estimator.z_stats()[1] - expected_z_sigma).abs() < 1e-6);
        assert!((estimator.p_values()[1] - expected_p_sigma).abs() < 1e-6);
        assert!(
            (estimator.conf_lower()[1] - (estimator.sigma() - z_crit * expected_se_sigma)).abs()
                < 1e-6
        );
        assert!(
            (estimator.conf_upper()[1] - (estimator.sigma() + z_crit * expected_se_sigma)).abs()
                < 1e-6
        );
    }

    /// 多変量（説明変数が1つ、切片込みで`k=2`・`s`を含め合計3パラメータ）の場合、
    /// 閉じた形の解析解は無いため、`cov_params`の対称性・各種統計量の内部整合性
    /// （z値・信頼区間の定義式通りの関係）を検証する回帰テスト。Logitの
    /// `fit_cov_params_is_symmetric_and_stats_are_internally_consistent`と同じ設計。
    /// 特に`destandardize_cov_params`・デルタ法ヤコビアン`diag(1,...,1,σ)`の適用が
    /// 非対角成分も含めて正しく機能しているかを確認する（対角成分だけでは
    /// 転置ミス等の一部のバグを検出できない）。
    #[test]
    fn fit_cov_params_is_symmetric_and_stats_are_internally_consistent() {
        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            Method::Newton,
            100,
            1e-8,
            true,
            0.95,
        )
        .unwrap();

        let k_plus_1 = estimator.params().len() + 1;
        for i in 0..k_plus_1 {
            for j in 0..k_plus_1 {
                assert!(
                    (*estimator.cov_params().get(i, j) - *estimator.cov_params().get(j, i)).abs()
                        < 1e-9,
                    "cov_params not symmetric at ({i},{j})"
                );
            }
            assert!(
                *estimator.cov_params().get(i, i) > 0.0,
                "diagonal[{i}] <= 0"
            );
        }

        let normal = Normal::new(0.0, 1.0).unwrap();
        let z_crit = normal.inverse_cdf(0.975);
        let mut coefs = estimator.params().to_vec();
        coefs.push(estimator.sigma());

        for (j, &coef) in coefs.iter().enumerate().take(k_plus_1) {
            let se = estimator.std_errors()[j];
            assert!((se - estimator.cov_params().get(j, j).sqrt()).abs() < 1e-9);
            assert!((estimator.z_stats()[j] - coef / se).abs() < 1e-9);
            assert!(
                (estimator.p_values()[j] - 2.0 * (1.0 - normal.cdf(estimator.z_stats()[j].abs())))
                    .abs()
                    < 1e-9
            );
            assert!((estimator.conf_lower()[j] - (coef - z_crit * se)).abs() < 1e-9);
            assert!((estimator.conf_upper()[j] - (coef + z_crit * se)).abs() < 1e-9);
        }
    }

    /// `fit`は非収束（`raise_on_non_convergence=false`）でも常に収束点（打ち切り点）の
    /// Hessianから`cov_params`を計算しようとするため、その打ち切り点でHessianが
    /// 不定符号・特異な場合は`MleError::SingularHessian`が返る（Newtonの最適化過程
    /// 自体は成功したが、SE計算のための逆行列計算が失敗するケース。`ols_initial_params`
    /// 由来の`SingularDesignMatrix`（設計行列`X`自体の特異性、最適化前に発生）とは
    /// 異なるエラー経路であることに注意）。`censored_regression_input`で`max_iter=1`
    /// にすると、Newton初回ステップ後の打ち切り点で実際にこれが発生することを実測で
    /// 確認済み（`max_iter=3`以降は`cov_params`計算が安定する。真の収束は`max_iter=11`、
    /// `fit_returns_unconverged_result_without_raising_when_raise_on_non_convergence_is_false`
    /// のコメント参照）。
    #[test]
    fn fit_returns_singular_hessian_error_when_cov_params_computation_fails_at_truncated_point() {
        let input = censored_regression_input();
        let result = TobitEstimator::fit(input, Method::Newton, 1, 1e-12, false, 0.95);
        assert!(
            matches!(result, Err(MleError::SingularHessian)),
            "{result:?}"
        );
    }

    /// 説明変数ありのモデルでも、打ち切りが実質発生しないデータではNewton法がOLSの
    /// 閉じた形の解（正規方程式）に近い値に収束するはず（Issue #215完了条件の本体、
    /// `expected_*`はOLSの公式から本テスト内で独立に計算する）。
    #[test]
    fn fit_newton_converges_near_ols_closed_form_when_censoring_is_negligible() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
        let input = TobitInput::from_columns(
            &y,
            std::slice::from_ref(&x),
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(-1000.0),
            Some(1000.0),
        )
        .unwrap();

        let n = y.len() as f64;
        let x_bar: f64 = x.iter().sum::<f64>() / n;
        let y_bar: f64 = y.iter().sum::<f64>() / n;
        let sxy: f64 = x
            .iter()
            .zip(&y)
            .map(|(xi, yi)| (xi - x_bar) * (yi - y_bar))
            .sum();
        let sxx: f64 = x.iter().map(|xi| (xi - x_bar).powi(2)).sum();
        let expected_slope = sxy / sxx;
        let expected_intercept = y_bar - expected_slope * x_bar;
        let expected_sse: f64 = x
            .iter()
            .zip(&y)
            .map(|(xi, yi)| (yi - (expected_intercept + expected_slope * xi)).powi(2))
            .sum();
        let expected_sigma = (expected_sse / n).sqrt();

        let estimator = TobitEstimator::fit(input, Method::Newton, 100, 1e-8, true, 0.95).unwrap();

        assert!(estimator.converged());
        assert!(
            (estimator.params()[0] - expected_intercept).abs() < 1e-4,
            "intercept={}, expected={}",
            estimator.params()[0],
            expected_intercept
        );
        assert!(
            (estimator.params()[1] - expected_slope).abs() < 1e-4,
            "slope={}, expected={}",
            estimator.params()[1],
            expected_slope
        );
        assert!(
            (estimator.sigma() - expected_sigma).abs() < 1e-4,
            "sigma={}, expected={}",
            estimator.sigma(),
            expected_sigma
        );
    }

    #[test]
    fn fit_returns_invalid_confidence_level_error_out_of_range() {
        let input = intercept_only_uncensored_input(&[1.0, 2.0, 3.0]);
        let result = TobitEstimator::fit(input, Method::Newton, 100, 1e-8, true, 1.5);
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5
            })
        );
    }

    #[test]
    fn fit_returns_invalid_max_iter_error() {
        let input = intercept_only_uncensored_input(&[1.0, 2.0, 3.0]);
        let result = TobitEstimator::fit(input, Method::Newton, 0, 1e-8, true, 0.95);
        assert_eq!(
            result.unwrap_err(),
            MleError::InvalidMaxIter { max_iter: 0 }
        );
    }

    #[test]
    fn fit_returns_invalid_tol_error() {
        let input = intercept_only_uncensored_input(&[1.0, 2.0, 3.0]);
        let result = TobitEstimator::fit(input, Method::Newton, 100, 0.0, true, 0.95);
        assert_eq!(result.unwrap_err(), MleError::InvalidTol { tol: 0.0 });
    }

    #[test]
    fn fit_returns_insufficient_observations_error() {
        // n=2, 総パラメータ数k+1=2(切片1+logσ1) → n<=k+1でエラー
        // （`validate_sufficient_observations`にx列数ではなくk+1を渡す設計、
        // Issue #212の結論）。
        let input = intercept_only_uncensored_input(&[1.0, 2.0]);
        let result = TobitEstimator::fit(input, Method::Newton, 100, 1e-8, true, 0.95);
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InsufficientObservations { n: 2, k: 2 })
        );
    }

    #[test]
    fn ols_initial_params_falls_back_to_sigma_one_for_perfect_fit() {
        // 完全な当てはまり（残差ゼロ）ではsse=0となり、log(0)=-infを避けるため
        // σ=1（s=0）にフォールバックする分岐を検証する（rust-reviewer指摘）。
        let xs = [1.0, 2.0, 3.0, 4.0];
        let x = Mat::from_fn(4, 2, |i, j| if j == 0 { 1.0 } else { xs[i] });
        let y = Mat::from_fn(4, 1, |i, _| 2.0 * xs[i] + 1.0); // y=2x+1、完全に当てはまる

        let params = ols_initial_params(&x, &y).unwrap();

        assert_eq!(params.len(), 3);
        assert!((params[0] - 1.0).abs() < 1e-9, "intercept={}", params[0]);
        assert!((params[1] - 2.0).abs() < 1e-9, "slope={}", params[1]);
        assert_eq!(params[2], 0.0, "s(=ln sigma)={}", params[2]);
    }

    #[test]
    fn fit_returns_singular_design_matrix_error_for_perfectly_collinear_data() {
        // x2=2*x1（完全な多重共線性）。ols_initial_paramsのQR分解が特異性を検出し、
        // Newton法が一度も反復する前にエラーを返す（rust-reviewer指摘: SingularHessian
        // とは区別される新設バリアント`SingularDesignMatrix`の検証）。
        // n=6 > 総パラメータ数k+1=4（切片1+x1+x2+logσ1）で`InsufficientObservations`を
        // 回避しつつ、x2=2*x1の構造的な多重共線性を検証する。
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_columns = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        ];
        let input = TobitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
            Some(-1000.0),
            Some(1000.0),
        )
        .unwrap();

        let result = TobitEstimator::fit(input, Method::Newton, 100, 1e-8, true, 0.95);
        assert!(
            matches!(result, Err(MleError::SingularDesignMatrix)),
            "{:?}",
            result
        );
    }

    /// 打ち切りが実際に発生するデータ（`x=1,2`が`lower=0`で左打ち切り、残りは非打ち切り、
    /// 打ち切り率25%）。OLS（打ち切りを無視）は真のTobit MLEと一致しないため、
    /// `fit()`の収束性を「打ち切りが皆無」の既存テストとは独立に検証できる
    /// （rust-reviewer指摘: 実際に打ち切りが発生するケースでのNewton収束確認が
    /// 無かったため追加）。
    fn censored_regression_input() -> TobitInput {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // 潜在変数 y* = -5 + 2x + noise（真の係数から意図的にわずかな残差を持たせる。
        // ノイズが皆無だと非打ち切り観測の当てはまりが完全になりσ̂→0という退化した
        // 境界ケースになり、いかなるソルバーでも不安定になることが判明したため
        // （デバッグ時に発見。`ols_initial_params`のsse≈0フォールバックと同種の問題が
        // 真のTobit尤度側でも起こりうる）。
        let y = vec![0.0, 0.0, 1.15, 2.9, 5.2, 6.85, 9.1, 10.95];
        TobitInput::from_columns(
            &y,
            &[x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap()
    }

    #[test]
    fn fit_newton_converges_for_data_with_actual_censoring() {
        let input = censored_regression_input();
        let estimator = TobitEstimator::fit(input, Method::Newton, 100, 1e-8, true, 0.95).unwrap();

        assert!(estimator.converged());
        assert_eq!(estimator.params().len(), 2);
        assert!(estimator.sigma() > 0.0 && estimator.sigma().is_finite());
        for &p in estimator.params() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn fit_bfgs_and_lbfgs_converge_to_similar_solution_as_newton_for_censored_data() {
        let newton = TobitEstimator::fit(
            censored_regression_input(),
            Method::Newton,
            100,
            1e-8,
            true,
            0.95,
        )
        .unwrap();

        for method in [Method::Bfgs, Method::Lbfgs] {
            let estimator =
                TobitEstimator::fit(censored_regression_input(), method, 200, 1e-8, true, 0.95)
                    .unwrap();
            assert!(estimator.converged(), "method={:?}", method);
            for (a, b) in estimator.params().iter().zip(newton.params()) {
                assert!((a - b).abs() < 1e-3, "method={:?}, a={a}, b={b}", method);
            }
            assert!(
                (estimator.sigma() - newton.sigma()).abs() < 1e-3,
                "method={:?}, sigma={}, newton_sigma={}",
                method,
                estimator.sigma(),
                newton.sigma()
            );
        }
    }

    /// `censored_regression_input`の`x`（`1..8`）は`standardize_columns`が実質no-opになる
    /// ほど自明なスケールではないが、`fit_bfgs_and_lbfgs_agree_with_newton_when_design_
    /// matrix_has_nontrivial_scale`（`logit.rs`/`probit.rs`）と同じ観点で、桁が大きく離れた
    /// スケール（`std`が1から大きく離れた値）でも標準化・逆標準化の往復が壊れないことを
    /// 明示的に検証する。`x`を100倍しつつ係数を1/100にスケールして同じ潜在変数
    /// `y*=-5+2x+noise`を再現しているため、期待される`β`・`σ`は`censored_regression_input`と
    /// 同一だが、ここではLogit/Probitの当該テストと同じくNewtonの結果を参照値とする
    /// クロスメソッド一致検証として書く（標準化空間での差異のみを見るのが目的のため）。
    #[test]
    fn fit_bfgs_and_lbfgs_agree_with_newton_when_design_matrix_has_nontrivial_scale() {
        let x = vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0];
        let y = vec![0.0, 0.0, 1.15, 2.9, 5.2, 6.85, 9.1, 10.95];
        let make_input = || {
            TobitInput::from_columns(
                &y,
                std::slice::from_ref(&x),
                vec!["x1".to_string()],
                true,
                "y".to_string(),
                Some(0.0),
                None,
            )
            .unwrap()
        };

        let newton =
            TobitEstimator::fit(make_input(), Method::Newton, 100, 1e-8, true, 0.95).unwrap();
        assert!(newton.converged());

        for method in [Method::Bfgs, Method::Lbfgs] {
            let estimator =
                TobitEstimator::fit(make_input(), method, 200, 1e-8, true, 0.95).unwrap();
            assert!(estimator.converged(), "method={:?}", method);
            for (a, b) in estimator.params().iter().zip(newton.params()) {
                assert!((a - b).abs() < 1e-3, "method={:?}, a={a}, b={b}", method);
            }
            assert!(
                (estimator.sigma() - newton.sigma()).abs() < 1e-3,
                "method={:?}, sigma={}, newton_sigma={}",
                method,
                estimator.sigma(),
                newton.sigma()
            );
        }
    }

    #[test]
    fn fit_returns_non_convergence_error_when_max_iter_is_too_small_and_raise_is_true() {
        let input = censored_regression_input();
        let result = TobitEstimator::fit(input, Method::Newton, 1, 1e-12, true, 0.95);
        assert!(
            matches!(result, Err(MleError::NonConvergence { .. })),
            "{:?}",
            result
        );
    }

    #[test]
    fn fit_returns_unconverged_result_without_raising_when_raise_on_non_convergence_is_false() {
        // `max_iter=1`（このテストの元々の値）だと、`censored_regression_input`の
        // 打ち切り点（Newtonの初回ステップ、まだ真の最尤推定点から遠い）でHessianが
        // 不定符号になり、`fit`が非収束時でも`cov_params`を計算するようになった
        // （Issue #217）ことで`SingularHessian`が先に発生してしまう（実測で確認、
        // `max_iter=1,2`はいずれも`SingularHessian`、`max_iter=3`以降で`cov_params`
        // 計算が安定し`converged=false`が返るようになる。真の収束は`max_iter=11`）。
        // `max_iter=3`に変更し、「非収束だがcov_params計算は成功する」ケースを踏む。
        let input = censored_regression_input();
        let estimator = TobitEstimator::fit(input, Method::Newton, 3, 1e-12, false, 0.95).unwrap();
        assert!(!estimator.converged());
    }
}
