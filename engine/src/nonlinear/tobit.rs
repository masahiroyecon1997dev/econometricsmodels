//! Tobitの入力データ（被説明変数・設計行列・打ち切り境界）の型定義。
//!
//! `engine`はpolars/PyO3を一切知らない（`.claude/rules/rust-style.md`「責務分離」参照）。
//! `engine_pybind`はpolars DataFrameから列ごとに`Vec<f64>`を抽出するところまでを担い
//! （`column_extraction::extract_f64_column`）、それらの列を本モジュールの
//! `TobitInput::from_columns`に渡す。`faer::Mat`への組み立て（切片列の自動追加を含む）は
//! ここ（engine側）の責務とする。`LogitInput`/`ProbitInput`と同型の設計だが、Tobitは
//! 打ち切り境界（`lower`/`upper`）という他の2手法にはない引数・検証を追加で持つ
//! （`docs/spec/nonlinear-common.md`7章「Tobitの打ち切り境界オプション」）。
//!
//! ## Logit/Probitとの設計上の違い: 検証の実施箇所
//!
//! Logit/Probitの`validate_binary_y`（`nonlinear/common.rs`）は`fit()`冒頭で呼ばれ、
//! `from_columns`自体は次元検証のみを行う。Tobitは打ち切り境界（`lower`/`upper`）を
//! `from_columns`の追加引数として受け取る都合上、境界自体の妥当性検証・`y`との整合性検証も
//! `from_columns`内で完結させる。
//!
//! ## 数式（打ち切り正規回帰）
//!
//! 内部最適化パラメータは`(β, s)`という`k+1`次元ベクトル（`s = log σ`、`σ>0`の制約を
//! 回避するための変数変換。`docs/spec/tobit-spec.md`3.1節参照。Olsen(1978)の
//! `(β/σ, 1/σ)`変換は不採用）。
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
//! 当初計画（`(β, logσ)`パラメータ化でもLogit/Probitと同じくゼロベクトル初期値から
//! Newton収束をまず試す）では、**打ち切りが皆無のデータ（通常の正規回帰に退化する
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

use crate::nonlinear::common::{
    ColumnScale, CovType, MarginalEffects, MarginalEffectsAt, MleError, MleFitOptions,
    SandwichVariant, U_CLAMP, checked_design_matrix_qr, clamped_pdf_cdf, cluster_cov_params,
    column_means, column_medians, compensated_sum, observed_information_cov_params, opg_cov_params,
    predict_new_data, run_solver, sandwich_cov_params, validate_cluster_cov_type,
    validate_confidence_level, validate_mle_options, validate_sufficient_observations,
};
use crate::shared::error::CommonError;
use crate::shared::inference;
use crate::shared::linear_algebra::ensure_well_conditioned_symmetric_matrix;
use argmin::core::{CostFunction, Error as OptimizerError, Gradient, Hessian};
use faer::prelude::{Solve, SolveLstsq};
use faer::{Mat, Side};
use statrs::distribution::{ChiSquared, Continuous, ContinuousCDF, Normal};

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
    /// 定数項を含むか。`fit()`の`TobitScaling`（列のセンタリング可否）等で必要
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
///
/// **Hessian項（`a=A(u)`・`c=C(u)`・`h_ss`）が使う`u`は、`λ`と同じクランプ済み
/// 引数から再構成する**（Probitの`ProbitProblem`と同型のバグ）。
/// `λ`は`clamped_pdf_cdf`が`zeta`を`[-U_CLAMP, U_CLAMP]`にクランプした後の値を
/// 使うため、`|zeta|>U_CLAMP`の領域では`λ`は`zeta`に対して事実上定数になる。
/// ここで生の（非クランプの）`zeta`を`A(u)=λ(u+λ)`の計算に混ぜると、`A(u)>0`と
/// いう恒等式（モジュール冒頭の数式表、Probitの大域凹性と同型の根拠）が数値的に
/// 破れ、`h_beta_coef`が負になりうる。`score_s`（勾配）は今回のスコープ外
/// （`U_CLAMP`領域でのcost/gradientの数学的非整合は別の既知の課題、
/// `docs/spec/probit-spec.md`4章参照）のため、`zeta`のまま変更しない。
fn censored_contribution(normal: &Normal, zeta: f64, direction: f64) -> Contribution {
    let (phi, big_phi) = clamped_pdf_cdf(normal, zeta);
    let lambda = phi / big_phi;
    let zeta_for_hessian = zeta.clamp(-U_CLAMP, U_CLAMP);
    let a = lambda * (zeta_for_hessian + lambda);
    let c = zeta_for_hessian * a - lambda;
    Contribution {
        log_lik: big_phi.ln(),
        score_beta_coef: -direction * lambda,
        score_s: -zeta * lambda,
        h_beta_coef: a,
        h_beta_s_coef: direction * c,
        h_ss: zeta_for_hessian * c,
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

    /// `TobitScaling`で標準化済みの設計行列`x_std`・`y`（`fit()`では`y`もスケーリング
    /// 済み）・打ち切り境界（`y`と同じ`y_scale`でスケーリング済み）から構築する。
    /// `TobitEstimator::fit`が最適化に使う経路（`LogitProblem::from_standardized`と
    /// 同じ位置づけ）。名前に反して標準化済みかどうかは問わない（`x`・`y`・境界・
    /// `params`のスケールが対応してさえいればよい。単体テストは生スケールでも呼ぶ）。
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
        // `contribution`（`linear_predictor`・`φ`/`Φ`・打ち切り分岐）は列によらないため、
        // 観測あたり1回だけ計算する。`Contribution`ごとは保持せず、使う2成分だけ集める。
        let mut beta_coefs = Vec::with_capacity(n);
        let mut score_s = Vec::with_capacity(n);
        for i in 0..n {
            let contribution = self.contribution(i, params, &normal);
            beta_coefs.push(contribution.score_beta_coef);
            score_s.push(contribution.score_s);
        }
        Mat::from_fn(n, k + 1, |i, col| {
            if col < k {
                beta_coefs[i] * (*self.x.get(i, col)) / sigma
            } else {
                score_s[i]
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
        // 補償和（`compensated_sum`のdocコメント参照。大標本のNewtonのコスト比較のため）。
        let ll = compensated_sum((0..n).map(|i| self.contribution(i, param, &normal).log_lik));
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

/// 非打ち切り観測（`y`が`lower`/`upper`いずれの境界にも一致しない観測）が1件も無い場合、
/// Tobitモデルは典型的には識別不能になる（`MleError::NoUncensoredObservations`のdoc
/// コメント参照。**厳密には十分条件であり必要条件ではない**——`x`による打ち切り
/// カテゴリの完全分離が実際の退化条件——点に注意。rust-reviewer指摘、詳細は
/// `MleError::NoUncensoredObservations`のdocコメント参照）。
///
/// `TobitInput::from_columns`（構造的妥当性: 境界指定自体の整合性・`y`と境界の整合性）
/// ではなく`fit()`冒頭（推定可能性の前提条件、rust-reviewer指摘で明記）で検証する。
/// `validate_sufficient_observations`と同じ位置づけ: 「データとして
/// 構造的に妥当か」と「このデータで推定を試みる価値があるか」を分離し、後者を`fit()`側の
/// 責務とする設計方針（`from_columns`はTobitInput単体で完結する検証のみを行い、
/// 推定アルゴリズムの成否に関わる検証は持ち込まない）。
///
/// 打ち切り判定は`censoring_fit_check`と同じ、`yᵢ==lower`/`yᵢ==upper`という浮動小数点の
/// 完全一致比較（`TobitInput::from_columns`が`y`を変換せず保持する設計と整合）。
///
/// # Errors
/// 非打ち切り観測が1件も無い場合は`MleError::NoUncensoredObservations`を返す。
fn validate_has_uncensored_observations(
    y: &Mat<f64>,
    lower: Option<f64>,
    upper: Option<f64>,
) -> Result<(), MleError> {
    let has_uncensored = (0..y.nrows()).any(|i| {
        let value = *y.get(i, 0);
        let at_lower = lower.is_some_and(|l| value == l);
        let at_upper = upper.is_some_and(|u| value == u);
        !at_lower && !at_upper
    });
    if has_uncensored {
        Ok(())
    } else {
        Err(MleError::NoUncensoredObservations { lower, upper })
    }
}

/// 列（または`y`）の母集団標準偏差（`ddof=0`）。標準偏差が0（定数列）の場合は`1.0`を
/// 返す（0除算回避）。
fn population_std(values: impl Iterator<Item = f64> + Clone, n: usize) -> f64 {
    let mean: f64 = values.clone().sum::<f64>() / n as f64;
    let var: f64 = values.map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let sd = var.sqrt();
    if sd > 0.0 { sd } else { 1.0 }
}

/// `TobitEstimator::fit`が最適化に使う、設計行列`x`・被説明変数`y`の標準化スケール。
///
/// `x`側（列の平均・標準偏差、切片の逆変換、ヤコビアンの`x`部分）はLogit/Probitと共通の
/// `nonlinear/common.rs`の`ColumnScale`（切片ありのとき平均も引く）に委ね、この構造体は
/// Tobit固有の`y`のスケーリング（`y_scale`）と`σ`（`logσ`）の扱いだけを持つ。
///
/// **Tobit固有の処理は`y`のスケーリング**（Logit/Probitの`standardize_columns`は`y`を扱わない）:
/// 1. **`y`のスケーリング**: 当時あった分離ヒューリスティック（標準化パラメータ空間のL2ノルムが
///    閾値を超えたら(準)分離と判定する、現在は廃止）はLogit/Probit（`y∈{0,1}`）で較正されていた。
///    Tobitは`y`が連続でスケールが任意のため、`y`のスケールが大きいだけで健全なMLE解でも
///    ノルムが閾値を超え誤発火した（Wooldridge mroz `hours`の生スケールTobitで発覚）。`x`の列標準化と同じ発想
///    （R `survival::survreg`は設計行列を標準化する）を`y`にも適用してスケール不変にする。
///    **`y_scale`は`y`の母集団標準偏差を「2の冪に丸めた」値**（`2^round(log2(sd))`）。
///    2の冪で割る操作は倍精度浮動小数点で指数部のシフトのみ（仮数部は不変）＝厳密なため、
///    `y`が既に程よいスケール（`sd ∈ [1/√2, √2)`）なら`y_scale=1`で完全な恒等変換になり、
///    そうでなくてもNewtonの反復軌道が「スケーリング前と厳密に相似」に保たれる。生の
///    標準偏差（例: `0.949`）で割ると、その丸め誤差（~1e-16）だけでも打ち切り率が高く
///    Hessianが際どい悪条件のデータ（合成`heavy_censoring`、左打ち切り60%）で正則化
///    NewtonがSingularHessianに倒れることが実測で判明したため、丸めて厳密化する。
///
/// **`x`列の平均センタリング（切片ありのときのみ）は`ColumnScale`が行う共通の処理**。Tobitでも
/// 必要な理由: `exper`(0〜45)と`expersq`(0〜2025)のように強く共線な列を含むと、スケーリング
/// だけでは標準化後のHessianが悪条件のままで、正則化Newton（`regularized_newton_step`）が
/// 回復不能な不定符号領域に落ちる（mroz `hours`で実測、`ols_initial_params`の丸め誤差~1e-14で
/// 結果が変わるほど際どい）。切片ありモデルでは列を平均センタリングすると`survreg`同様Newtonが
/// 安定して収束する。切片なしモデルではセンタリングせず（吸収先の切片が無く逆変換が壊れる
/// ため）、スケーリングのみ行う。
///
/// 標準化空間の内部パラメータ`(β̃, s̃=logσ̃)`と元のスケール`(β, s=logσ)`の対応
/// （`c = y_scale`、`x̄ⱼ`・`stdⱼ`は`x`列`j`の平均・母集団標準偏差、切片列は`x̄₀=0`・`std₀=1`）:
///
/// - 切片あり: `βⱼ = c·β̃ⱼ/stdⱼ`（`j≥1`）、`β₀ = c·β̃₀ - Σ_{j≥1} x̄ⱼ·βⱼ`、`s = s̃ + ln c`
/// - 切片なし: `βⱼ = c·β̃ⱼ/stdⱼ`（全`j`）、`s = s̃ + ln c`
///
/// `β`の式は`ColumnScale`の逆変換に`c`を掛けたもの。`c`は2の冪なので、オーバーフロー・
/// サブノーマルを除き、`c·β̃ⱼ/stdⱼ`と`c·(β̃ⱼ/stdⱼ)`、切片の補正項も掛ける順序によらず
/// ビット単位で同じ値になる。この一致は正しさのためではなく、Tobitの出力を丸めの
/// レベルで安定させるために保っている（`y_scale`を2の冪以外にしない理由でもある）。
struct TobitScaling {
    /// `x`側の標準化スケール（列ごとの平均・標準偏差、切片の逆変換、ヤコビアンの`x`部分）。
    /// Logit/Probitと共通の`ColumnScale`（切片ありのとき平均も引く）。
    x: ColumnScale,
    /// `y`（と打ち切り境界）のスケール = `y`の母集団標準偏差を2の冪に丸めた値
    /// （`2^round(log2(sd))`、必ず`> 0`。構造体docコメント参照）
    y_scale: f64,
}

impl TobitScaling {
    /// `x`・`y`から標準化スケールを求める。
    fn fit(x: &Mat<f64>, y: &Mat<f64>, has_intercept: bool) -> Self {
        let n = x.nrows();
        // `y`の母集団標準偏差を2の冪に丸める（構造体docコメント「`y`のスケーリング」参照。
        // 2の冪での除算は仮数部を変えないため、`y`が程よいスケールなら恒等変換になる）。
        // 指数は倍精度の正規化数の範囲に収める（極端な`y`はengine_pybind側の非有限値
        // チェックで既に弾かれているため、この`clamp`は防御的）。
        let y_exp = (population_std((0..n).map(|i| *y.get(i, 0)), n)
            .log2()
            .round() as i32)
            .clamp(-1000, 1000);
        let y_scale = 2.0_f64.powi(y_exp);
        Self {
            x: ColumnScale::fit(x, has_intercept),
            y_scale,
        }
    }

    /// 設計行列を標準化する: 切片ありは`x̃ⱼ = (xⱼ - x̄ⱼ)/stdⱼ`、切片なしは`x̃ⱼ = xⱼ/stdⱼ`
    /// （切片列は無変換）。`ColumnScale::standardize`に委譲する。
    fn standardize_x(&self, x: &Mat<f64>) -> Mat<f64> {
        self.x.standardize(x)
    }

    /// `y`を`y_scale`でスケーリングする（平均は引かない）。
    fn standardize_y(&self, y: &Mat<f64>) -> Mat<f64> {
        Mat::from_fn(y.nrows(), 1, |i, _| *y.get(i, 0) / self.y_scale)
    }

    /// 打ち切り境界を`y_scale`でスケーリングする（`y`のスケーリングと整合）。
    fn standardize_bound(&self, bound: Option<f64>) -> Option<f64> {
        bound.map(|v| v / self.y_scale)
    }

    /// 標準化空間の内部パラメータ`(β̃, s̃)`（`k+1`次元）を元のスケールの`(β, logσ)`へ
    /// 逆変換する（構造体docコメントの式）。`β`は`x`側の逆変換
    /// （`ColumnScale`の式）に`y_scale`を掛けたもの。`y_scale`は2の冪なので、この掛け算は
    /// 掛ける順序によらずビット単位で同じ値になる（構造体docコメント参照）。
    fn destandardize_params(&self, params_std: &[f64]) -> Vec<f64> {
        let k = self.x.stds().len();
        debug_assert_eq!(params_std.len(), k + 1, "(β̃, s̃)は列数+1個");
        let c = self.y_scale;
        let mut out = Vec::with_capacity(k + 1);
        out.extend(
            crate::nonlinear::common::destandardize_params(&params_std[..k], &self.x)
                .into_iter()
                .map(|b| c * b),
        );
        out.push(params_std[k] + c.ln());
        out
    }

    /// `(β̃, s̃)`空間から`(β, σ)`空間（元のスケール）への変換のヤコビアン`(k+1)×(k+1)`。
    /// `Cov(β, σ) = J · Cov(β̃, s̃) · Jᵀ`で使う（`destandardize_params`の線形部分の
    /// 微分に、`s̃→σ=c·exp(s̃)`のデルタ法`∂σ/∂s̃=σ`を末尾行に折り込んだもの）。`sigma`は
    /// 逆変換済みの`σ`。`β`の部分は`x`側のヤコビアン（`ColumnScale::param_jacobian`）の`c`倍。
    fn param_jacobian(&self, sigma: f64) -> Mat<f64> {
        let k = self.x.stds().len();
        let c = self.y_scale;
        let jx = self.x.param_jacobian();
        Mat::from_fn(k + 1, k + 1, |i, m| {
            if i == k {
                if m == k { sigma } else { 0.0 }
            } else if m < k {
                c * *jx.get(i, m)
            } else {
                0.0
            }
        })
    }

    /// `Cov(β, σ) = J · cov_std · Jᵀ`（`param_jacobian`のJを両側から適用）。
    fn destandardize_cov_params(&self, cov_std: &Mat<f64>, sigma: f64) -> Mat<f64> {
        let j = self.param_jacobian(sigma);
        &j * cov_std * j.transpose()
    }
}

/// 打ち切りを無視した単純なOLS（`X'Xβ=X'y`の最小二乗解）から、Newton法の初期値
/// `(β, logσ)`を計算する（モジュール冒頭「Newton法の初期値」節参照）。`x`は
/// `TobitScaling::standardize_x`で標準化済みの設計行列を渡す想定（`fit()`が最適化に使う
/// 空間と一致させ、初期値をそのまま`run_solver`に渡せるようにするため）。`y`も
/// `TobitScaling::standardize_y`でスケーリング済みの値を渡す。`y`は打ち切りを無視し、
/// 観測された値をそのまま連続値として扱う（あくまで初期値のヒューリスティックで
/// あり、Tobitの推定値そのものではない）。
///
/// 特異性検出（列ピボットQRの`R`対角成分の相対閾値、`.claude/rules/rust-style.md`
/// 「線形代数」）は`nonlinear::common::checked_design_matrix_qr`に委譲する（
/// Logit/Probitの`ols_based_initial_params`とランクチェックを共通化した。Logit/Probitは
/// 同じ関数のQR解に加えてリンクのスケール補正を施すが、Tobitの`β`はOLSと同一スケールの
/// ため補正は不要で、QR解をそのまま`β`初期値に使い、`σ`初期値だけ残差から別途求める）。
///
/// # Errors
/// `x`が特異（完全な多重共線性等）な場合は`MleError::SingularDesignMatrix`を返す
/// （`MleError::SingularDesignMatrix`のdocコメント参照。Newton法が一度も反復していない
/// 段階で発生しうるため`SingularHessian`とは区別する）。
fn ols_initial_params(x: &Mat<f64>, y: &Mat<f64>) -> Result<Vec<f64>, MleError> {
    let n = x.nrows();
    let k = x.ncols();

    let qr = checked_design_matrix_qr(x)?;
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

/// 対数尤度 `ℓ(θ) = Σᵢ ℓᵢ(θ)`（モジュール冒頭の数式表参照）を`x`・`y`・打ち切り境界・
/// `params`（`(β, s)`、`k+1`次元）から直接計算する。`TobitProblem::cost`（`-ℓ(θ)`、argminの
/// `CostFunction`）と同じ`Contribution::log_lik`の総和を使うが、argminのトレイトが要求する
/// `Result`型を経由する必要が無い内部専用の計算（適合度統計量向け、収束後に1回だけ評価する）
/// のため、`LogitEstimator::log_likelihood`と同じ理由で独立した関数として切り出している。
/// `TobitProblem::from_standardized`は名前に反して標準化済みかどうかを問わない
/// （`x`・`params`のスケールが対応してさえいればよい）ため、ここでは元のスケールの`x`・
/// `params`をそのまま渡す。
fn log_likelihood(
    x: &Mat<f64>,
    y: &Mat<f64>,
    lower: Option<f64>,
    upper: Option<f64>,
    params: &[f64],
) -> f64 {
    let problem = TobitProblem::from_standardized(x.clone(), y.clone(), lower, upper);
    let normal = Normal::standard();
    // `TobitProblem::cost`と同じ補償和にし、最適化で評価した値と適合度統計量の値を揃える。
    compensated_sum((0..x.nrows()).map(|i| problem.contribution(i, params, &normal).log_lik))
}

/// モデル全体の有意性検定（切片以外の係数`β`が同時にゼロという帰無仮説のWald検定）。
/// `docs/spec/nonlinear-common.md`5章「Tobitはこの共通コアから2点を意図的に
/// 外す」の通り、Tobitは`llnull`（打ち切りがあると閉形式解を持たない）に基づく尤度比検定
/// ではなく、`cov_params`から直接計算できるWald検定を使う（`AER::tobit`の`summary.tobit`と
/// 同じ方式）。
///
/// `W = β_slopes' Σ_slopes⁻¹ β_slopes`（`β_slopes`は`params`の`k_constant`番目以降
/// `df_model`個、`Σ_slopes`は`cov_params`（`(k+1)×(k+1)`、`β∪{σ}`空間）のうち対応する
/// `df_model×df_model`の主小行列）。`σ`は`cov_params`の`k`番目の行・列にあるため、
/// `df_model`個の範囲（`k_constant..k`）には含まれず、検定対象から自動的に除外される。
/// OLSの`wald_f_test`と同型の構成（`ensure_well_conditioned_symmetric_matrix`による
/// 悪条件検出→Cholesky分解）だが、検定分布はF分布ではなく標準正規分布に基づく
/// カイ二乗分布（`docs/spec/nonlinear-common.md`4章「検定分布」、自由度で正規化する
/// `F=W/df_model`の変換を行わない）。
///
/// `Logit`/`Probit`と異なりTobit専用（`llnull`を使わない検定方式のため`common.rs`には
/// 置かない、`.claude/rules/rust-style.md`「系統内で共有するロジック」の対象は現時点で
/// Tobitのみ）。
///
/// # Errors
/// `Σ_slopes`が悪条件・特異な場合は`CommonError::ComputationFailed`
/// （OLSの`wald_f_test`と同じ理由、理論上は正定値のはずだが浮動小数点演算の丸めによる
/// 境界的な失敗に備えた防御的な扱い）。`ChiSquared::new`の失敗（`df_model<=0`）は
/// 呼び出し元が`df_model>0`を事前に保証するため理論上到達不能。
fn wald_chi2_test(
    params: &[f64],
    cov_params: &Mat<f64>,
    k_constant: usize,
    df_model: usize,
) -> Result<(f64, f64), MleError> {
    let beta_slopes = Mat::from_fn(df_model, 1, |i, _| params[i + k_constant]);
    let v_slopes = Mat::from_fn(df_model, df_model, |i, j| {
        *cov_params.get(i + k_constant, j + k_constant)
    });

    ensure_well_conditioned_symmetric_matrix(
        &v_slopes,
        df_model,
        "coefficient covariance submatrix for the Wald test",
    )?;

    let llt = v_slopes.llt(Side::Lower).map_err(|_| {
        CommonError::ComputationFailed(
            "failed to invert coefficient covariance submatrix for the Wald test".to_string(),
        )
    })?;
    let v_slopes_inv_beta = llt.solve(&beta_slopes);

    let wald_statistic: f64 = (0..df_model)
        .map(|i| (*beta_slopes.get(i, 0)) * (*v_slopes_inv_beta.get(i, 0)))
        .sum();

    let chi2 = ChiSquared::new(df_model as f64)
        .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
    let wald_p_value = chi2.sf(wald_statistic);

    Ok((wald_statistic, wald_p_value))
}

/// `marginal_effects`が評価する対象（McDonald-Moffitt 1980）。Logit/Probitの
/// `dydx_and_jacobian`型の共通化はしない（対象ごとに式が異なり、
/// 同型の`(w,s)`分解に無理に収める価値がないと判断した。
/// `docs/spec/nonlinear-common.md`6章参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarginalEffectsTarget {
    /// 潜在変数の期待値 `E[y*|x] = x'β`
    ExpectedLatent,
    /// 打ち切りを考慮した観測値の期待値 `E[y|x]`（既定、実測`y`と直接比較できる量）
    ExpectedObserved,
    /// 非打ち切り確率 `P(uncensored|x)`
    ProbUncensored,
    /// 非打ち切り領域での条件付き期待値 `E[y|lower<y<upper,x]`（切断正規分布の平均）。
    /// 下限0のみの左打ち切りなら `E[y|y>0,x]`。`E[y|x]`との関係は
    /// `E[y|x]=lower·P(下限)+upper·P(上限)+P(uncensored|x)·E[y|lower<y<upper,x]`
    ExpectedUncensored,
}

/// 逆ミルズ比・生存関数を連分数展開で評価する`a`の下限。連分数は`a>=3`で60段あれば相対誤差
/// `3e-17`以下に収束する。`a`がこれ未満では`normal_survival`（テイラー級数）で`Φ(-a)`を
/// 作って`φ(a)/Φ(-a)`を直接計算する。`a≈38`では`φ`・`Φ(-a)`がアンダーフローするため、
/// 裾は連分数でなければ評価できない。
const MILLS_CONTINUED_FRACTION_THRESHOLD: f64 = 3.0;

/// 連分数展開の段数（`a>=MILLS_CONTINUED_FRACTION_THRESHOLD`で倍精度に十分収束する）。
const MILLS_CONTINUED_FRACTION_DEPTH: u32 = 60;

/// `Φ(x)-1/2`（`x>=0`）を絶対誤差`≈5e-16`以下で計算する。`statrs`の`Normal::cdf`は絶対誤差が
/// 約`1.4e-11`（`Φ(-1)`で相対`8.6e-11`）あり、`W=Φ(zb)-Φ(za)`で割る量では限界効果の
/// 標準誤差に`100`倍程度に増幅されて現れる（`k=20`で相対`1e-8`）ため使わない。
/// `x<MILLS_CONTINUED_FRACTION_THRESHOLD`では`normal_cdf_minus_half_series`、それ以上では
/// `1/2-φ(x)/λ(x)`（`λ`は連分数、`Φ(x)`が1に近いので絶対誤差で足りる）。
fn normal_cdf_minus_half(normal: &Normal, x: f64) -> f64 {
    debug_assert!(x >= 0.0 || x.is_nan(), "requires x >= 0");
    if x >= MILLS_CONTINUED_FRACTION_THRESHOLD {
        let (lambda, _) = inverse_mills_ratio_continued_fraction(x, MILLS_CONTINUED_FRACTION_DEPTH);
        return 0.5 - normal.pdf(x) / lambda;
    }
    normal_cdf_minus_half_series(normal, x)
}

/// `Φ(x)-1/2=φ(x)·Σₙ x^{2n+1}/(2n+1)!!`（`x>=0`、全項が正で桁落ちなし）。`x<3`なら最大33項で
/// 収束する（打ち切り条件は項が和の`1e-17`以下）。上限500は収束しない入力（NaN等）での
/// 無限ループ防止。`x`が大きいほど項数が増えるため`x<MILLS_CONTINUED_FRACTION_THRESHOLD`で使う。
fn normal_cdf_minus_half_series(normal: &Normal, x: f64) -> f64 {
    let x2 = x * x;
    let mut term = x;
    let mut sum = x;
    for n in 1..=500u32 {
        term *= x2 / (2.0 * f64::from(n) + 1.0);
        sum += term;
        if term <= sum * 1e-17 {
            break;
        }
    }
    normal.pdf(x) * sum
}

/// 標準正規分布の生存関数`Φ(-a)`。`a<0`は`1/2+(Φ(-a)-1/2)`、`0<=a<3`は`1/2-(Φ(a)-1/2)`
/// （`Φ(-3)≈1.3e-3`で桁落ちは約3桁、`a→3`で相対誤差が最大`4e-13`）、`a>=3`は
/// `φ(a)/λ(a)`（連分数、相対`5e-16`以下）。`a>=3`は`a≈38`で`φ(a)/λ`が非正規化数になり
/// `a≳38.6`で0になるので、それより遠い裾は`inverse_mills_ratio_and_gap`（`φ`を作らない）を
/// 使う。本番経路（`inverse_mills_ratio_and_gap`）が呼ぶのは`a<3`のみ。
fn normal_survival(normal: &Normal, a: f64) -> f64 {
    if a < 0.0 {
        0.5 + normal_cdf_minus_half(normal, -a)
    } else if a < MILLS_CONTINUED_FRACTION_THRESHOLD {
        0.5 - normal_cdf_minus_half(normal, a)
    } else {
        let (lambda, _) = inverse_mills_ratio_continued_fraction(a, MILLS_CONTINUED_FRACTION_DEPTH);
        normal.pdf(a) / lambda
    }
}

/// 逆ミルズ比`λ(a)=φ(a)/Φ(-a)`（標準正規分布の右裾のハザード率）と、その`a`からの
/// 超過分`λ(a)-a`を返す。`a`が大きいと`λ≈a`で、引き算`λ-a`は桁落ちするため
/// （真値は`≈1/a`）、超過分は引き算せず直接計算して一緒に返す。
///
/// `a`が小さい（負を含む）ときは`normal_survival`の`Φ(-a)`を使い（`1-Φ(a)`は裾で0に潰れる
/// ため使わない）、`a>=MILLS_CONTINUED_FRACTION_THRESHOLD`では連分数展開
/// `λ(a)=a+1/(a+2/(a+3/(a+…)))`に切り替える（`φ`と`Φ(-a)`が`a≈38`でアンダーフローする
/// 領域でも有効）。
fn inverse_mills_ratio_and_gap(normal: &Normal, a: f64) -> (f64, f64) {
    if a < MILLS_CONTINUED_FRACTION_THRESHOLD {
        let lambda = normal.pdf(a) / normal_survival(normal, a);
        (lambda, lambda - a)
    } else {
        inverse_mills_ratio_continued_fraction(a, MILLS_CONTINUED_FRACTION_DEPTH)
    }
}

/// `inverse_mills_ratio_and_gap`の連分数展開（`depth>=2`段で打ち切り、内側から評価する）。
/// `a>0`が前提。`λ(a)-a=1/(a+2/(a+3/(a+…)))`は第2項以降の連分数の逆数として
/// 引き算なしに得られる。
fn inverse_mills_ratio_continued_fraction(a: f64, depth: u32) -> (f64, f64) {
    let mut t = a;
    for k in (2..=depth).rev() {
        t = a + f64::from(k) / t;
    }
    let gap = 1.0 / t;
    (a + gap, gap)
}

/// `0<=za<zb`の上側裾での`(Pa, Pb, A, B)`。`Pa=φ(za)/W`・`Pb=φ(zb)/W`
/// （`W=Φ(zb)-Φ(za)`）、`A=Pa-za`・`B=zb+Pb`（`ExpectedUncensored`の式で`za`・`zb`との
/// 大きな数同士の差を避けるための量。`truncation_pdf_ratios`参照）。
///
/// `W`を直接作ると両方が裾にあるとき桁落ち・アンダーフローするため、`W=Φ(-za)-Φ(-zb)`を
/// `Φ(-za)`で割った比で表す。`p=λ(za)`・`q=λ(zb)`・`r=φ(zb)/φ(za)=exp(-(zb²-za²)/2)`と
/// すると`φ(zb)/Φ(-za)=r·p`・`Φ(-zb)/Φ(-za)=r·p/q`なので、`W/Φ(-za)=1-r·p/q`、
/// `Pa=p·q/(q-r·p)`、`Pb=r·p·q/(q-r·p)`となる（`q>p>=r·p`で分母は正）。
/// `A=(p-za)+(Pa-p)=gap_p+r·p²/(q-r·p)`は正の項の和で、`B=zb+Pb`も正の項の和。
fn upper_tail_pdf_ratios(normal: &Normal, za: f64, zb: f64) -> (f64, f64, f64, f64) {
    debug_assert!(0.0 <= za && za < zb, "upper tail requires 0 <= za < zb");
    let (p, gap_p) = inverse_mills_ratio_and_gap(normal, za);
    let (q, _) = inverse_mills_ratio_and_gap(normal, zb);
    let r = (-0.5 * (zb - za) * (zb + za)).exp();
    let denom = q - r * p;
    let pa = p * q / denom;
    let pb = r * p * q / denom;
    (pa, pb, gap_p + r * p * p / denom, zb + pb)
}

/// `truncation_pdf_ratios`の返り値。`pa=φ(za)/W`・`pb=φ(zb)/W`（`W=Φ(zb)-Φ(za)`、
/// 境界が無い方向は0）、`a_gap=pa-za`・`b_gap=zb+pb`。
struct TruncationRatios {
    pa: f64,
    pb: f64,
    a_gap: f64,
    b_gap: f64,
}

/// 非打ち切り領域`(za,zb)`（標準化した下限・上限。`None`はその方向の打ち切りなし）に
/// 切断した標準正規分布について、`W=Φ(zb)-Φ(za)`で割る量を`W→0`となる裾でも
/// 破綻しない形で返す（`TruncationRatios`）。`E[y|uncensored]`の`μ`からのずれ
/// `σ(Pa-Pb)`、切断分散比`w`など、`W`で割る量は全てこれで書ける。`A=Pa-za`・
/// `B=zb+Pb`は`|z|`が大きいと`Pa≈za`のように大きな数同士の差になるため、引き算せず
/// 直接計算して返す（`uncensored_w_and_s`が`z²`同士の桁落ちを避けるのに使う）。
///
/// - 片側のみ: 逆ミルズ比（左のみ`Pa=λ(za)`、`A=λ(za)-za`。右のみ`Pb=λ(-zb)`、
///   `B=λ(-zb)-(-zb)`）
/// - 両側で`za>=0`（上側の裾）: `upper_tail_pdf_ratios`
/// - 両側で`zb<=0`（下側の裾）: 符号反転`(za,zb)→(-zb,-za)`で上側の裾へ写し、
///   `(Pa,Pb)`と`(A,B)`をそれぞれ入れ替える
/// - 両側で`za<0<zb`: `W`が裾に潰れないので直接計算する（`A`・`B`は正の項の和）
fn truncation_pdf_ratios(normal: &Normal, za: Option<f64>, zb: Option<f64>) -> TruncationRatios {
    let (pa, pb, a_gap, b_gap) = match (za, zb) {
        (None, None) => (0.0, 0.0, 0.0, 0.0),
        (Some(a), None) => {
            let (lambda, gap) = inverse_mills_ratio_and_gap(normal, a);
            (lambda, 0.0, gap, 0.0)
        }
        (None, Some(b)) => {
            let (lambda, gap) = inverse_mills_ratio_and_gap(normal, -b);
            (0.0, lambda, 0.0, gap)
        }
        (Some(a), Some(b)) => {
            if a >= 0.0 {
                upper_tail_pdf_ratios(normal, a, b)
            } else if b <= 0.0 {
                let (pa_m, pb_m, a_gap_m, b_gap_m) = upper_tail_pdf_ratios(normal, -b, -a);
                (pb_m, pa_m, b_gap_m, a_gap_m)
            } else {
                // `Φ(b)-Φ(a)=(Φ(b)-1/2)+(1/2-Φ(a))`は正の項の和（`a<0<b`）
                let w = normal_cdf_minus_half(normal, b) + normal_cdf_minus_half(normal, -a);
                let pa = normal.pdf(a) / w;
                let pb = normal.pdf(b) / w;
                (pa, pb, pa - a, b + pb)
            }
        }
    };
    TruncationRatios {
        pa,
        pb,
        a_gap,
        b_gap,
    }
}

/// `ExpectedUncensored`の評価に使う、線形予測子`mu`での標準化境界と`TruncationRatios`。
/// `za`/`zb`は境界が無い方向を`0.0`のダミー値にする（対応する`pa`/`pb`が0なので
/// 以降の式に影響しない）。
struct UncensoredTerms {
    za: f64,
    zb: f64,
    ratios: TruncationRatios,
}

impl UncensoredTerms {
    fn new(mu: f64, sigma: f64, lower: Option<f64>, upper: Option<f64>, normal: &Normal) -> Self {
        let za = lower.map(|l| (l - mu) / sigma);
        let zb = upper.map(|u| (u - mu) / sigma);
        Self {
            za: za.unwrap_or(0.0),
            zb: zb.unwrap_or(0.0),
            ratios: truncation_pdf_ratios(normal, za, zb),
        }
    }

    /// `E[y|uncensored]=μ+σ·(Pa-Pb)`
    fn mean(&self, mu: f64, sigma: f64) -> f64 {
        mu + sigma * (self.ratios.pa - self.ratios.pb)
    }
}

/// 打ち切り境界`bound`を標準化した`z=(bound-μ)/σ`・`φ(z)`・`Φ(z)`を計算する。`None`は
/// 「その方向は打ち切りなし」を意味し、`Φ(∓∞)=0/1`・`φ(∓∞)=0`に相当する値を返す
/// （`z`自体はダミー値`0.0`を返すが、後続の計算では常に`φ=0.0`とセットでしか使わないため
/// 結果に影響しない。`is_lower`で下限側（`Φ(-∞)=0`）か上限側（`Φ(+∞)=1`）かを切り替える）。
fn boundary_terms(
    bound: Option<f64>,
    mu: f64,
    sigma: f64,
    normal: &Normal,
    is_lower: bool,
) -> (f64, f64, f64) {
    match bound {
        Some(c) => {
            let z = (c - mu) / sigma;
            let (phi, big_phi) = clamped_pdf_cdf(normal, z);
            (z, phi, big_phi)
        }
        None => (0.0, 0.0, if is_lower { 0.0 } else { 1.0 }),
    }
}

/// `target`ごとの限界効果`dydx_j=w*βⱼ`（`marginal_effects_from_tobit_w_s`のdocコメント
/// 「数式」参照）における`w`とその勾配`(s_beta, s_sigma)`を、評価点`x_point`
/// （`k`要素、`at="overall"`なら観測`i`の行、`at="mean"`/`"median"`なら代表点）で計算する。
///
/// ## 数式
///
/// `μ=x_point'β`とし、`za=(lower-μ)/σ`・`zb=(upper-μ)/σ`（`boundary_terms`、`None`の
/// 方向は`φ=0`かつ`Φ`は0/1の定数として扱う）とすると、非打ち切り確率は
/// `W(θ)=Φ(zb)-Φ(za)`（左のみ打ち切りなら`Φ(zb)=1`で`W=Φ(za)`相当、右のみなら
/// `Φ(za)=0`で`W=Φ(zb)`相当、両側打ち切りなら両方が効く——単一の式で3ケースとも
/// 表現できる）。
///
/// - `ExpectedLatent`（`E[y*|x]=x'β`）: `dydx_j=βⱼ`（`x_point`に依存しない自明な形。
///   `w=1`・`s_beta=0`・`s_sigma=0`）
/// - `ExpectedObserved`（`E[y|x]`）: `w=W(θ)`。`∂W/∂μ=(φ(za)-φ(zb))/σ`、
///   `∂W/∂σ=(za·φ(za)-zb·φ(zb))/σ`（`s_beta_m=∂W/∂μ・x_point[m]`、連鎖律`∂μ/∂βₘ=x_point[m]`）
/// - `ProbUncensored`（`P(uncensored|x)=W(θ)`自体）: `w=∂W/∂μ=(φ(za)-φ(zb))/σ`。
///   その勾配は`∂²W/∂μ²=(za·φ(za)-zb·φ(zb))/σ²`・
///   `∂(∂W/∂μ)/∂σ=(φ(za)(za²-1)-φ(zb)(zb²-1))/σ²`
/// - `ExpectedUncensored`（`E[y|uncensored,x]`）: `uncensored_w_and_s`
///
/// いずれも境界項がある方向のみ`φ≠0`となるため（`boundary_terms`のNone時`φ=0`）、
/// 左/右/両側打ち切りいずれでも同じ式で正しく計算できる（両側打ち切りは
/// McDonald-Moffitt(1980)の一般形、片側打ち切りはその特殊ケースとして導出済み、
/// `docs/spec/tobit-spec.md`3.5節参照）。
fn target_w_and_s(
    target: MarginalEffectsTarget,
    x_point: &[f64],
    beta: &[f64],
    sigma: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    normal: &Normal,
) -> (f64, Vec<f64>, f64) {
    let k = beta.len();
    if target == MarginalEffectsTarget::ExpectedLatent {
        return (1.0, vec![0.0; k], 0.0);
    }

    let mu: f64 = (0..k).map(|m| x_point[m] * beta[m]).sum();
    if target == MarginalEffectsTarget::ExpectedUncensored {
        return uncensored_w_and_s(x_point, mu, sigma, lower, upper, normal);
    }
    let (za, phi_za, cdf_za) = boundary_terms(lower, mu, sigma, normal, true);
    let (zb, phi_zb, cdf_zb) = boundary_terms(upper, mu, sigma, normal, false);

    match target {
        MarginalEffectsTarget::ExpectedLatent | MarginalEffectsTarget::ExpectedUncensored => {
            unreachable!("handled by early return above")
        }
        MarginalEffectsTarget::ExpectedObserved => {
            let w = cdf_zb - cdf_za;
            let dw_dmu = (phi_za - phi_zb) / sigma;
            let s_beta: Vec<f64> = x_point.iter().map(|&x_m| dw_dmu * x_m).collect();
            let s_sigma = (za * phi_za - zb * phi_zb) / sigma;
            (w, s_beta, s_sigma)
        }
        MarginalEffectsTarget::ProbUncensored => {
            let w = (phi_za - phi_zb) / sigma;
            let dw_dmu = (za * phi_za - zb * phi_zb) / (sigma * sigma);
            let s_beta: Vec<f64> = x_point.iter().map(|&x_m| dw_dmu * x_m).collect();
            let s_sigma = (phi_za * (za * za - 1.0) - phi_zb * (zb * zb - 1.0)) / (sigma * sigma);
            (w, s_beta, s_sigma)
        }
    }
}

/// `ExpectedUncensored`を信頼できる精度で評価できる、非打ち切り区間の幅`(upper-lower)/σ`の下限。
/// `w`の絶対誤差は約`ε/δ²`、真値は`≈δ²/12`なので相対誤差は約`12ε/δ⁴`で、デルタ法に使う
/// `∂w/∂μ`・`∂w/∂σ`の誤差はさらに大きい（`ε/δ³`級。`|z|=10`で`δ=1e-3`の`s`は
/// 絶対`2e-8`で`w≈1e-7`の`1/5`）。`δ=1e-2`で`w`の相対誤差が全領域（`za∈[-10,10]`）で
/// `6e-7`以下（80桁参照の実測）、`δ=1e-3`では`w`の相対誤差が`3e-3`〜、`s`が`w`の1%超になる。
/// `σ̂`が区間幅の100倍以上（非打ち切りの確率が約0.4%未満）でほぼ識別できない当てはめのときだけ
/// 超える。
const MIN_UNCENSORED_WIDTH_RATIO: f64 = 1e-2;

/// 両側打ち切りで非打ち切り区間が`σ`に比べて狭すぎる場合に
/// [`MleError::UncensoredIntervalTooNarrow`]を返す（`MIN_UNCENSORED_WIDTH_RATIO`）。
/// 片側打ち切りは区間が無限大なので常に`Ok`。幅`(upper-lower)/σ`はモデル（`σ̂`）で1つに
/// 決まり観測に依存しないので、呼び出しごとに1回検査すれば足りる。
fn ensure_uncensored_interval_resolvable(
    lower: Option<f64>,
    upper: Option<f64>,
    sigma: f64,
) -> Result<(), MleError> {
    if let (Some(lower), Some(upper)) = (lower, upper)
        && (upper - lower) / sigma < MIN_UNCENSORED_WIDTH_RATIO
    {
        return Err(MleError::UncensoredIntervalTooNarrow {
            lower,
            upper,
            sigma,
        });
    }
    Ok(())
}

/// `ExpectedUncensored`（`m(μ,σ)=E[y|lower<y<upper,x]=μ+σ(Pa-Pb)`）の`(w, s_beta, s_sigma)`。
///
/// ## 数式
///
/// `m`は`x`に`μ=x'β`を通してしか依存しないので`dydx_j=w·βⱼ`、`w=∂m/∂μ`。
/// `Pa=φ(za)/W`・`Pb=φ(zb)/W`・`ρ1=Pa-Pb`・`ρ2=za·Pa-zb·Pb`とおくと
/// `w=1+ρ2-ρ1²`（切断正規分布の分散を`σ²`で割った値で、`(0,1)`に入る）。
/// `za`/`zb`についての偏微分は`W_a=-φa`・`W_b=φb`などから
///
/// - `∂ρ1/∂za=Pa(ρ1-za)`・`∂ρ1/∂zb=Pb(zb-ρ1)`
/// - `∂ρ2/∂za=Pa(1-za²+ρ2)`・`∂ρ2/∂zb=Pb(zb²-1-ρ2)`
/// - `w_a=∂ρ2/∂za-2ρ1∂ρ1/∂za`・`w_b=∂ρ2/∂zb-2ρ1∂ρ1/∂zb`
///
/// で、`∂za/∂μ=∂zb/∂μ=-1/σ`・`∂za/∂σ=-za/σ`・`∂zb/∂σ=-zb/σ`より
/// `∂w/∂μ=-(w_a+w_b)/σ`・`∂w/∂σ=-(za·w_a+zb·w_b)/σ`。左のみ打ち切り（`zb`なし）で
/// `Pa=λ(za)`とすると`w=1+za·λ-λ²`（McDonald-Moffittの`1-zλ-λ²`型の式）に一致する。
///
/// ## 数値安定性
///
/// 上の式をそのまま書くと、`|z|`が大きい裾（`Pa≈za`）では`za²`と`za·Pa`の大きな数同士の
/// 引き算になり、`w_a`の絶対誤差が`ε·z³`で増える（`z=72`の観測を含む実データで`s_sigma`の
/// 相対誤差が`5e-9`、限界効果の標準誤差が`1e-8`ずれた）。
/// `A=Pa-za`・`B=zb+Pb`（`truncation_pdf_ratios`が引き算なしに返す）で整理すると
///
/// - `w=1-Pa·A-Pb·B+2·Pa·Pb`
/// - `w_a=Pa[1+za·A-zb·Pb-2ρ1(A-Pb)]`
/// - `w_b=Pb[-1+zb·B-za·Pa-2ρ1(B-Pa)]`
///
/// となり、括弧内は`O(1)`同士の和なので絶対誤差は`ε·Pa`程度に収まる。`W→0`となる裾でも
/// `Pa`・`Pb`が直接計算されるため割り算は破綻しない。`w`は理論上`(0,1)`なので、丸め誤差で
/// 範囲外になった値が平均に混ざらないよう`[0,1]`にクランプする。
///
/// ## 既知の限界
///
/// - **遠い裾の相対精度**: 片側の裾で`|z|`が大きいと、括弧内の`O(1)`同士が`O(z⁻⁴)`まで
///   打ち消し合うため、絶対誤差は小さいままでも`w`・`w_a`・`w_b`の**相対**精度は
///   `|z|≳1e3`から落ち、`|z|≳1e4`で`w_a`は符号も保証されない（`w≈1/z²`・`w_a≈-2/z³`自体が
///   無視できる大きさなので、全観測の平均には影響しない）。
/// - **非打ち切り区間が極端に狭い場合**: 区間幅`δ=zb-za=(upper-lower)/σ`が小さいと、上側/下側の
///   裾の`q-r·p`や中央の`W`・`1-Pa·A-Pb·B+…`が桁落ちする（`MIN_UNCENSORED_WIDTH_RATIO`の
///   docコメント参照）。この関数は`δ`を検査しない。`TobitEstimator`の`predict`・
///   `predict_new_data`・`marginal_effects`が入口で`ensure_uncensored_interval_resolvable`
///   により`δ<1e-2`を`MleError::UncensoredIntervalTooNarrow`として拒否する。
fn uncensored_w_and_s(
    x_point: &[f64],
    mu: f64,
    sigma: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    normal: &Normal,
) -> (f64, Vec<f64>, f64) {
    let UncensoredTerms {
        za,
        zb,
        ratios:
            TruncationRatios {
                pa,
                pb,
                a_gap,
                b_gap,
            },
    } = UncensoredTerms::new(mu, sigma, lower, upper, normal);
    let rho1 = pa - pb;
    let w = (1.0 - pa * a_gap - pb * b_gap + 2.0 * pa * pb).clamp(0.0, 1.0);

    let w_a = pa * (1.0 + za * a_gap - zb * pb - 2.0 * rho1 * (a_gap - pb));
    let w_b = pb * (-1.0 + zb * b_gap - za * pa - 2.0 * rho1 * (b_gap - pa));
    let dw_dmu = -(w_a + w_b) / sigma;
    let s_beta: Vec<f64> = x_point.iter().map(|&x_m| dw_dmu * x_m).collect();
    let s_sigma = -(za * w_a + zb * w_b) / sigma;
    (w, s_beta, s_sigma)
}

/// `target_w_and_s`の`at="overall"`（AME）版。全観測を1回走査し、各観測の`(w,s_beta,
/// s_sigma)`を平均する（`w`/`s`は`β`について線形に`dydx_j=w*βⱼ`へ寄与するため、平均を
/// 先に取ってから`dydx_j`を計算しても、観測ごとに`dydx_j`を計算してから平均しても
/// 同じ結果になる。Logitの`overall_w_and_s`と同じ考え方）。
fn overall_target_w_and_s(
    x: &Mat<f64>,
    beta: &[f64],
    sigma: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    target: MarginalEffectsTarget,
    normal: &Normal,
) -> (f64, Vec<f64>, f64) {
    let n = x.nrows();
    let k = x.ncols();
    let mut w_sum = 0.0;
    let mut s_beta_sum = vec![0.0; k];
    let mut s_sigma_sum = 0.0;
    // `x_row`はループの外で1回だけ確保し、観測ごとに上書きして使い回す
    // （`target_w_and_s`が`&[f64]`を要求する都合上コピー自体は避けられないが、
    // 観測数`n`回分のヒープ確保は不要、rust-reviewer指摘）。
    let mut x_row = vec![0.0; k];
    for i in 0..n {
        for (j, v) in x_row.iter_mut().enumerate() {
            *v = *x.get(i, j);
        }
        let (w, s_beta, s_sigma) =
            target_w_and_s(target, &x_row, beta, sigma, lower, upper, normal);
        w_sum += w;
        for (m, s_m) in s_beta_sum.iter_mut().enumerate() {
            *s_m += s_beta[m];
        }
        s_sigma_sum += s_sigma;
    }
    let n_f = n as f64;
    for s_m in s_beta_sum.iter_mut() {
        *s_m /= n_f;
    }
    (w_sum / n_f, s_beta_sum, s_sigma_sum / n_f)
}

/// `target_w_and_s`の`at="mean"`/`"median"`版。代表点`x_bar`（各説明変数の標本平均または
/// 中央値）で1回だけ評価する（Logitの`at_point_w_and_s`と同じ考え方、`target_w_and_s`
/// 自体がこの形なので薄いラッパー）。
fn at_point_target_w_and_s(
    x_bar: &[f64],
    beta: &[f64],
    sigma: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    target: MarginalEffectsTarget,
    normal: &Normal,
) -> (f64, Vec<f64>, f64) {
    target_w_and_s(target, x_bar, beta, sigma, lower, upper, normal)
}

/// `marginal_effects_from_tobit_w_s`の呼び出しに必要な、フィット済みモデルの情報を束ねる
/// （`clippy::too_many_arguments`を避けるため。`nonlinear/common.rs`の
/// `FittedModelForMarginalEffects`と同じ理由・同じパターンだが、Tobitは`cov_params`が
/// `(k+1)×(k+1)`（`σ`を含む）である点が異なるため独立の型として定義する）。
struct TobitMarginalEffectsInputs<'a> {
    /// 説明変数名（定数項を含む）
    param_names: &'a [String],
    /// 定数項を含むか
    has_intercept: bool,
    /// 説明変数の数（定数項を含む、`σ`は含まない）
    k: usize,
    /// 係数（元のスケール、`β`部分のみ）
    beta: &'a [f64],
    /// `(β,σ)`の分散共分散行列（元のスケール、`(k+1)×(k+1)`）
    cov_params: &'a Mat<f64>,
}

/// `target_w_and_s`系が計算する`(w, s_beta, s_sigma)`から、限界効果`dydx_j=w*βⱼ`と
/// そのヤコビアン（`β`の`k`列+`σ`の1列、計`k+1`列）を組み立て、デルタ法による標準誤差・
/// z値・p値・信頼区間を計算し、定数項を除いた`MarginalEffects`を返す。
///
/// `nonlinear/common.rs`の`marginal_effects_from_w_s`と数式の骨格
/// （`∂dydx_j/∂θₘ=βⱼ*s_m+[j==m]*w`）は同じだが、パラメータ次元が`k`ではなく`k+1`
/// （`σ`を含む）である点が異なるため独立実装とする。
///
/// # Errors
/// `confidence_level`が`(0, 1)`の範囲外: `CommonError::InvalidConfidenceLevel`
fn marginal_effects_from_tobit_w_s(
    model: TobitMarginalEffectsInputs,
    w: f64,
    s_beta: &[f64],
    s_sigma: f64,
    confidence_level: f64,
) -> Result<MarginalEffects, MleError> {
    validate_confidence_level(confidence_level)?;
    let TobitMarginalEffectsInputs {
        param_names,
        has_intercept,
        k,
        beta,
        cov_params,
    } = model;

    let dydx: Vec<f64> = (0..k).map(|j| w * beta[j]).collect();
    // ヤコビアンの各行は長さ`k+1`（`β`のk列 + `σ`の1列）
    let jacobian: Vec<Vec<f64>> = (0..k)
        .map(|j| {
            let mut row: Vec<f64> = (0..k)
                .map(|m| beta[j] * s_beta[m] + if j == m { w } else { 0.0 })
                .collect();
            row.push(beta[j] * s_sigma);
            row
        })
        .collect();

    // `Normal::new(0.0, 1.0)`は標準正規分布であり、標準偏差が正であることを要求する
    // statrsの検証を常に満たすため、この`map_err`分岐は理論上到達不能
    // （`.claude/rules/rust-style.md`「テスト」のカバレッジ方針参照、`marginal_effects_from_w_s`
    // と同じ扱い）。
    let normal =
        Normal::new(0.0, 1.0).map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
    let z_crit = inference::critical_value(&normal, confidence_level);

    let k_constant = usize::from(has_intercept);
    let k_plus_1 = k + 1;
    let mut out_param_names = Vec::with_capacity(k - k_constant);
    let mut out_dydx = Vec::with_capacity(k - k_constant);
    let mut std_errors = Vec::with_capacity(k - k_constant);
    let mut test_stats = Vec::with_capacity(k - k_constant);
    let mut p_values = Vec::with_capacity(k - k_constant);
    let mut conf_lower = Vec::with_capacity(k - k_constant);
    let mut conf_upper = Vec::with_capacity(k - k_constant);

    for j in k_constant..k {
        let jac_row = &jacobian[j];
        let mut var_j = 0.0;
        for a in 0..k_plus_1 {
            for b in 0..k_plus_1 {
                var_j += jac_row[a] * (*cov_params.get(a, b)) * jac_row[b];
            }
        }
        let se = var_j.sqrt();
        let stat = inference::compute_inference_stat(&normal, dydx[j], se, z_crit);

        out_param_names.push(param_names[j].clone());
        out_dydx.push(dydx[j]);
        std_errors.push(se);
        test_stats.push(stat.stat);
        p_values.push(stat.p_value);
        conf_lower.push(stat.conf_low);
        conf_upper.push(stat.conf_high);
    }

    Ok(MarginalEffects::from_parts(
        out_param_names,
        out_dydx,
        std_errors,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
    ))
}

/// `predict`が返す予測量`target`の値を、線形予測子`mu=x'β`・`σ`・打ち切り境界から計算する
/// （McDonald-Moffitt 1980）。`target_w_and_s`と同じ`boundary_terms`（`lower`/`upper`の
/// `None`側を`Φ(∓∞)=0/1`・`φ(∓∞)=0`相当の定数として扱う統一的な扱い）を再利用するため、
/// 左のみ・右のみ・両側打ち切りいずれでも同じ式で正しく計算できる（`target_w_and_s`の
/// docコメント「数式」と同じ考え方。こちらは値そのもの、`target_w_and_s`はその微分）。
///
/// `ExpectedUncensored`は`E[y|lower<y<upper,x]=μ+σ(Pa-Pb)`（`UncensoredTerms::mean`、
/// 裾でも安定な`truncation_pdf_ratios`経由）。
///
/// `E[y|x]`の閉形式は`Φ(za)*lower + (1-Φ(zb))*upper + (Φ(zb)-Φ(za))*mu - σ(φ(zb)-φ(za))`
/// （両側打ち切りの一般形。`lower`/`upper`が`None`の側は`boundary_terms`の規約により
/// 係数が0になるため、ダミー値`lower.unwrap_or(0.0)`/`upper.unwrap_or(0.0)`を掛けても
/// 結果に影響しない）。`P(uncensored|x)`は`Φ(zb)-Φ(za)`そのもの。
fn predicted_value(
    target: MarginalEffectsTarget,
    mu: f64,
    sigma: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    normal: &Normal,
) -> f64 {
    if target == MarginalEffectsTarget::ExpectedLatent {
        return mu;
    }
    if target == MarginalEffectsTarget::ExpectedUncensored {
        return UncensoredTerms::new(mu, sigma, lower, upper, normal).mean(mu, sigma);
    }
    let (_, phi_za, cdf_za) = boundary_terms(lower, mu, sigma, normal, true);
    let (_, phi_zb, cdf_zb) = boundary_terms(upper, mu, sigma, normal, false);
    match target {
        MarginalEffectsTarget::ExpectedLatent | MarginalEffectsTarget::ExpectedUncensored => {
            unreachable!("handled by early return above")
        }
        MarginalEffectsTarget::ExpectedObserved => {
            let lower_c = lower.unwrap_or(0.0);
            let upper_c = upper.unwrap_or(0.0);
            cdf_za * lower_c + (1.0 - cdf_zb) * upper_c + (cdf_zb - cdf_za) * mu
                - sigma * (phi_zb - phi_za)
        }
        MarginalEffectsTarget::ProbUncensored => cdf_zb - cdf_za,
    }
}

/// 1カテゴリ（下側打ち切り／非打ち切り／上側打ち切り）分の打ち切り適合度チェック結果
/// （`CensoringFitCheck`のdocコメント参照）。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CensoringFitCategory {
    /// 実測の該当カテゴリの比率（`y`がちょうど境界値に一致する観測の割合。非打ち切り
    /// カテゴリは`1 - 下側比率 - 上側比率`）
    observed_rate: f64,
    /// モデル含意の平均確率（各観測の`P(該当カテゴリ|xᵢ)`の平均。`target_w_and_s`の
    /// `ProbUncensored`と同じ`Φ`の組み合わせ、`predicted_value`のdocコメント参照）
    model_implied_rate: f64,
}

impl CensoringFitCategory {
    /// 実測の該当カテゴリの比率
    pub fn observed_rate(&self) -> f64 {
        self.observed_rate
    }

    /// モデル含意の平均確率
    pub fn model_implied_rate(&self) -> f64 {
        self.model_implied_rate
    }
}

/// 打ち切り予測の適合度チェック（`TobitEstimator::censoring_fit_check`のdocコメント参照。
/// Logit/Probitの`pred_table`の代替、`docs/spec/nonlinear-common.md`6章「Tobitは
/// predict()/marginal_effects()/pred_table()のいずれも独自の形になる」で確定済み）。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CensoringFitCheck {
    /// 下側打ち切り（`lower`が`None`のときは`None`）
    lower: Option<CensoringFitCategory>,
    /// 非打ち切り
    uncensored: CensoringFitCategory,
    /// 上側打ち切り（`upper`が`None`のときは`None`）
    upper: Option<CensoringFitCategory>,
}

impl CensoringFitCheck {
    /// 下側打ち切り（`lower`が`None`のときは`None`）
    pub fn lower(&self) -> Option<CensoringFitCategory> {
        self.lower
    }

    /// 非打ち切り
    pub fn uncensored(&self) -> CensoringFitCategory {
        self.uncensored
    }

    /// 上側打ち切り（`upper`が`None`のときは`None`）
    pub fn upper(&self) -> Option<CensoringFitCategory> {
        self.upper
    }
}

/// Tobitの推定結果。`fit`でのバリデーション・最適化・`cov_type`に応じたSE計算・
/// 適合度統計量・Wald検定・限界効果を通過した状態を表す。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug)]
pub struct TobitEstimator {
    input: TobitInput,
    /// 係数（元のスケール、`β`部分のみ）。`input.param_names()`と対応する
    params: Vec<f64>,
    /// 誤差項の標準偏差（元のスケール）。内部最適化パラメータ`s̃=logσ̃`（`x`のセンタリング・
    /// スケーリング・`y`のスケーリング`c=y_scale`後の空間、`fit`のdocコメント「標準化空間」
    /// 節・`TobitScaling`参照）を`logσ = s̃ + ln c`で元のスケールへ戻し、`σ = exp(logσ)`で
    /// 逆変換したもの。`logσ`は`x`の変換とは無関係だが`y`のスケーリングには
    /// 比例する（`σ̃ = σ/c`）。
    sigma: f64,
    /// `(β, σ)`の分散共分散行列（元のスケール、`(k+1)×(k+1)`）。`fit`に渡した`cov_type`に
    /// 応じて観測情報行列（`Classical`）・OPG（`Opg`）・サンドイッチ型（`Hc0`/`Hc1`）・
    /// クラスターロバスト（`Cluster`）のいずれかで計算される。最適化は`x`のセンタリング・
    /// スケーリング・`y`のスケーリング後の内部パラメータ化`(β̃, s̃=logσ̃)`空間のHessianで
    /// 行われるため、`cov_type`に応じた行列演算で標準化空間の分散共分散行列を得た後、
    /// `Cov(β,σ) = J·Cov(β̃,s̃)·Jᵀ`（`J`は`TobitScaling::param_jacobian`。`x`の
    /// センタリング・スケーリングの解除と`s̃→σ=c·exp(s̃)`のデルタ法`∂σ/∂s̃=σ`を1本に
    /// 合成したヤコビアン）で`(β, σ)`空間へ変換する（`k+1`行目・列目が`σ`に対応）。
    /// 切片ありモデルでは`J`の切片行が全傾き係数に依存する非対角要素を持つため、
    /// `β`-`σ`間だけでなく`β₀`-`βⱼ`間の共分散も正しく変換され、行列全体が一貫した値になる
    /// （`fit`のdocコメント「`σ`のSE」節参照。ユーザー確認済み、単なる対角成分の
    /// `Var(σ)≈σ²Var(logσ)`だけでなく将来の限界効果等での再利用を見据えてフル行列を
    /// 変換する設計を採用）。
    cov_params: Mat<f64>,
    /// 標準誤差（`k+1`、元のスケール、`β∪{σ}`。`k`番目の要素が`σ`に対応）。
    /// `cov_params`の対角成分の平方根
    std_errors: Vec<f64>,
    /// z統計量（`k+1`）= `coef / std_errors`（`β∪{σ}`それぞれについて）
    test_stats: Vec<f64>,
    /// 両側p値（`k+1`）。標準正規分布に基づく
    p_values: Vec<f64>,
    /// 信頼区間の下限（`k+1`）
    conf_lower: Vec<f64>,
    /// 信頼区間の上限（`k+1`）
    conf_upper: Vec<f64>,
    /// 収束点での対数尤度 `ℓ(θ̂)`
    log_likelihood: f64,
    /// 赤池情報量規準 `-2ℓ(θ̂) + 2(k+1)`（総パラメータ数`k+1`は`σ`を含む）
    aic: f64,
    /// ベイズ情報量規準 `-2ℓ(θ̂) + ln(n)(k+1)`
    bic: f64,
    /// 観測数
    n_obs: usize,
    /// モデルの自由度（切片以外の`β`の数、Wald検定のカイ二乗分布の自由度でもある）
    df_model: usize,
    /// 残差自由度 `n-(k+1)`（`σ`を含む総パラメータ数を差し引く。AER::tobit/survregの
    /// `df.residual`と同じ規約、`docs/spec/tobit-spec.md`2章参照）
    df_resid: usize,
    /// Wald検定統計量（`wald_chi2_test`のdocコメント参照）。`df_model==0`のときNaN
    wald_statistic: f64,
    /// Wald検定のp値（自由度`df_model`のカイ二乗分布、上側確率）。`df_model==0`のときNaN
    wald_p_value: f64,
    /// 収束したかどうか
    converged: bool,
    /// 実際の反復回数
    n_iter: usize,
}

impl TobitEstimator {
    /// `solver`（Newton-Raphson/BFGS/L-BFGS）で負の対数尤度を最小化し、Tobitの係数`β`・
    /// 誤差項の標準偏差`σ`を推定する。
    ///
    /// 内部最適化パラメータは`(β, s=logσ)`という`k+1`次元ベクトル（モジュール冒頭の
    /// 数式参照）。最適化前に`TobitScaling`で設計行列と`y`を標準化する（
    /// 詳細は`TobitScaling`のdocコメント）: `x`は切片ありなら列を平均センタリング＋
    /// スケーリング・切片なしならスケーリングのみ、`y`（と打ち切り境界`lower`/`upper`）は
    /// `y`の母集団標準偏差を2の冪に丸めた`c`で一律スケーリングする。標準化空間`(β̃, s̃=logσ̃)`で
    /// 最適化し、`TobitScaling::destandardize_params`で元のスケール`(β, s=logσ)`へ
    /// 逆変換する（`x'β/c = x̃'β̃`・`σ̃ = σ/c`を満たす線形＋定数シフトの写像。
    /// 切片ありのとき`β₀`は全傾き係数に依存する）。
    ///
    /// 初期値はゼロベクトルではなく`ols_initial_params`が計算するOLS推定値
    /// （`LogitEstimator::fit`とは異なる。モジュール冒頭「Newton法の初期値」節参照）。
    ///
    /// `cov_type`は観測情報行列（`Classical`）・OPG（`Opg`）・サンドイッチ型
    /// （`Hc0`/`Hc1`）・クラスターロバスト（`Cluster`）に対応する（Probitの前例＝
    /// コミット`c868912`と同じ理由でまとめて実装。`CovType`はLogit/Probit/
    /// Tobit共有の1つのenumで既に`Cluster`バリアントを含んでおり、`match cov_type`を
    /// 網羅的にする都合上、OPG/サンドイッチのみを先に実装してクラスターを未実装のまま
    /// 残す設計は取れない。ユーザー確認済み）。`Opg`/`Hc0`/`Hc1`/`Cluster`は収束点での
    /// 観測ごとのスコア（`TobitProblem::scores`）が必要なため、標準化空間の設計行列を
    /// 保持したまま`TobitProblem`をクローンしておき（`Classical`のときは不要な複製を
    /// 避けるため条件付きで行う、`LogitEstimator::fit`と同じ理由）、`run_solver`が返す
    /// 収束点のパラメータで評価する。観測数の十分性検証（`validate_sufficient_
    /// observations`）には`x`の列数`k`ではなく総最適化パラメータ数`k+1`を使う
    /// （`validate_sufficient_observations`のdocコメント参照）。
    /// Logit/Probitの`validate_has_regressors`（`k==0`検証）はTobitでは呼ばない
    /// （`logσ`が常に存在するため対応するケースが生じない、同関数のdocコメント参照）。
    /// 推定オプションの検証は3手法共通の`validate_mle_options`を使い、Logit/Probit用の
    /// `validate_fit_preconditions`は呼ばない（手法ごとの違いは同関数のdocコメント参照）。
    /// 打ち切り境界の検証（`MleError::InvalidCensoringBounds`/`YOutOfCensoringBounds`）は
    /// `fit()`ではなく`TobitInput::from_columns`で完了済み（モジュール冒頭のdocコメント参照）。
    ///
    /// ## `σ`のSE（デルタ法）
    ///
    /// `run_solver`が返すHessianは内部パラメータ化`(β̃, s̃=logσ̃)`空間で評価されたもの
    /// （`TobitScaling`で標準化した`x`・`y`に対応する空間、上記「標準化空間」節参照）。
    /// `cov_type`に応じた行列演算（`observed_information_cov_params`/`opg_cov_params`/
    /// `sandwich_cov_params`/`cluster_cov_params`）で標準化空間の分散共分散行列を得た後、
    /// `TobitScaling::destandardize_cov_params`が`Cov(β,σ) = J·Cov(β̃,s̃)·Jᵀ`
    /// （`J = TobitScaling::param_jacobian`。`x`のセンタリング・スケーリングの解除と
    /// `s̃→σ=c·exp(s̃)`のデルタ法`∂σ/∂s̃=σ`を1本に合成したヤコビアン）で`(β, σ)`空間へ
    /// 変換する（`cov_params`のdocコメント参照。`σ`の対角成分のみ見ると
    /// `Var(σ)≈σ²Var(logσ)`という`docs/spec/tobit-spec.md`3.3節に記載の式に一致する。
    /// この変換は`cov_type`の種類に依存せず、
    /// いずれの行列演算の結果にも同じ`J`を適用すればよい）。
    ///
    /// # Errors
    /// - `confidence_level`が`(0, 1)`の範囲外: `CommonError::InvalidConfidenceLevel`
    /// - `max_iter`が0以下: `MleError::InvalidMaxIter`
    /// - `tol`が0以下: `MleError::InvalidTol`
    /// - 観測数`n`が総パラメータ数`k+1`以下: `CommonError::InsufficientObservations`
    /// - 非打ち切り観測（`y`が`lower`/`upper`いずれの境界にも一致しない観測）が1件も無い:
    ///   `MleError::NoUncensoredObservations`（`validate_has_uncensored_observations`参照）
    /// - `cov_type=Cluster`でグループキー未指定: `CommonError::MissingClusterColumn`
    /// - `cov_type=Cluster`でクラスター数が2未満: `CommonError::InsufficientClusters`
    /// - `cov_type=Cluster`でクラスター数`g`が傾き係数の数`q`（`k - k_constant`）以下:
    ///   `CommonError::InsufficientClustersForInference`（`rank(Ŝ) ≤ g - 1`のため全体
    ///   Wald検定の`q×q`部分行列が構造的に特異。従来は`wald_chi2_test`内の
    ///   `ComputationFailed`だったものを`fit()`冒頭のバリデーションへ前倒し）
    /// - OLS初期値計算時に`x`が特異（完全な多重共線性等）: `MleError::SingularDesignMatrix`
    ///   （`ols_initial_params`参照）
    /// - `raise_on_non_convergence=true`かつ`max_iter`回で未収束: `MleError::NonConvergence`
    /// - 収束点（または`raise_on_non_convergence=false`時の打ち切り点）のHessianが特異:
    ///   `MleError::SingularHessian`
    /// - `cov_type=Opg`でOPG行列（`Σᵢ sᵢsᵢ'`）が特異: `MleError::SingularOpgMatrix`
    /// - 傾き係数間の悪条件（極端なスケール差等）でWald検定用の係数分散共分散部分行列が
    ///   数値的にほぼ特異: `CommonError::ComputationFailed`（`wald_chi2_test`参照。`g > q`
    ///   でも起こりうるbackstop。`df_model==0`（切片以外の`β`が無い）のときはこの検定自体を
    ///   スキップし`wald_statistic`/`wald_p_value`はNaNになる）
    pub fn fit(input: TobitInput, options: MleFitOptions) -> Result<Self, MleError> {
        let MleFitOptions {
            solver,
            max_iter,
            tol,
            raise_on_non_convergence,
            cov_type,
            confidence_level,
        } = options;

        // faer のグローバル並列度を Par::Seq に固定する（`crate::shared::parallelism`）。
        crate::shared::parallelism::ensure_serial();

        validate_mle_options(confidence_level, max_iter, tol)?;

        let n = input.nobs();
        let k = input.k();
        validate_sufficient_observations(n, k + 1)?;
        validate_cluster_cov_type(&cov_type, n, k - usize::from(input.has_intercept()))?;
        validate_has_uncensored_observations(input.y(), input.lower(), input.upper())?;

        // `x`（切片ありなら列を平均センタリング＋スケーリング、切片なしはスケーリング
        // のみ）と`y`（と打ち切り境界、`y`の母集団標準偏差を2の冪に丸めた値で一律
        // スケーリング）を標準化してから最適化する（`TobitScaling`のdocコメント・
        // `fit`のdocコメント「標準化空間」節参照）。
        let scaling = TobitScaling::fit(input.x(), input.y(), input.has_intercept());
        let x_std = scaling.standardize_x(input.x());
        let y_std = scaling.standardize_y(input.y());
        let lower_std = scaling.standardize_bound(input.lower());
        let upper_std = scaling.standardize_bound(input.upper());
        let initial_params = ols_initial_params(&x_std, &y_std)?;
        let problem = TobitProblem::from_standardized(x_std, y_std, lower_std, upper_std);
        // `cov_type`がOPG/サンドイッチ型/クラスターの場合、収束点でのスコア評価に
        // 標準化空間の`TobitProblem`が必要になる。`run_solver`は`problem`の所有権を
        // 取り込むため、事前にクローンしておく（`LogitEstimator::fit`と同じ理由）。
        let problem_for_scores = match &cov_type {
            CovType::Classical => None,
            CovType::Opg | CovType::Hc0 | CovType::Hc1 | CovType::Cluster { .. } => {
                Some(problem.clone())
            }
        };

        let output = run_solver(
            problem,
            solver,
            initial_params,
            max_iter as u64,
            tol,
            input.y().nrows(),
            raise_on_non_convergence,
        )?;
        // Tobitの真の分離は係数発散ではなく`σ→0`退化として現れるため、Logit/Probitの分離検出
        // （`nonlinear::separation`）は行わない。全件打ち切りは
        // `validate_has_uncensored_observations`（`NoUncensoredObservations`）、
        // 部分的な準完全分離は`NonConvergence`で捕捉される。

        // 標準化空間の内部パラメータ`(β̃, s̃)`を元のスケールの`(β, logσ)`へ逆変換する
        // （`TobitScaling`のdocコメントの式）。
        let params_full = scaling.destandardize_params(&output.params);
        let sigma = params_full[k].exp();
        let params = params_full[..k].to_vec();

        let hessian_std = Mat::from_fn(k + 1, k + 1, |i, j| output.hessian[i][j]);
        // `problem_for_scores.as_ref().expect(...)`は各非`Classical`分岐でのみ呼ばれ、
        // 直前の`match cov_type`により常に`Some`であることが保証されている内部契約
        // （`LogitEstimator::fit`と同じ防御的な扱い）。
        let cov_params_std = match &cov_type {
            CovType::Classical => observed_information_cov_params(&hessian_std, k + 1)?,
            CovType::Opg => {
                let problem = problem_for_scores
                    .as_ref()
                    .expect("problem_for_scores must be Some for CovType::Opg");
                opg_cov_params(&problem.scores(&output.params), k + 1)?
            }
            CovType::Hc0 => {
                let problem = problem_for_scores
                    .as_ref()
                    .expect("problem_for_scores must be Some for CovType::Hc0");
                sandwich_cov_params(
                    &hessian_std,
                    &problem.scores(&output.params),
                    n,
                    k + 1,
                    SandwichVariant::Hc0,
                )?
            }
            CovType::Hc1 => {
                let problem = problem_for_scores
                    .as_ref()
                    .expect("problem_for_scores must be Some for CovType::Hc1");
                sandwich_cov_params(
                    &hessian_std,
                    &problem.scores(&output.params),
                    n,
                    k + 1,
                    SandwichVariant::Hc1,
                )?
            }
            CovType::Cluster { groups } => {
                let problem = problem_for_scores
                    .as_ref()
                    .expect("problem_for_scores must be Some for CovType::Cluster");
                // `groups`のNone・クラスター数不足の検証はfit()冒頭の
                // `validate_cluster_cov_type`で完了済み。
                // ここでの`expect`はその契約を明記する防御的な扱い（`LogitEstimator::fit`と同じ）。
                let groups = groups
                    .as_ref()
                    .expect("groups are validated by validate_cluster_cov_type for Cluster");
                cluster_cov_params(
                    &hessian_std,
                    &problem.scores(&output.params),
                    n,
                    k + 1,
                    groups,
                )?
            }
        };
        // 標準化空間`(β̃, s̃)`の分散共分散行列を、元のスケール`(β, σ)`空間へ
        // `Cov(β,σ) = J·Cov(β̃,s̃)·Jᵀ`で変換する（`J`は`TobitScaling::param_jacobian`。
        // `x`の列標準化・センタリングの解除と、`s̃→σ=c·exp(s̃)`のデルタ法`∂σ/∂s̃=σ`を
        // 1本のヤコビアンに合成済み。`cov_type`の種類に依存しない共通の後処理、
        // 上記docコメント「標準化空間」「σのSE」節参照）。
        let cov_params = scaling.destandardize_cov_params(&cov_params_std, sigma);

        let normal = Normal::standard();
        let z_crit = inference::critical_value(&normal, confidence_level);

        let mut std_errors = vec![0.0; k + 1];
        let mut test_stats = vec![0.0; k + 1];
        let mut p_values = vec![0.0; k + 1];
        let mut conf_lower = vec![0.0; k + 1];
        let mut conf_upper = vec![0.0; k + 1];

        for j in 0..=k {
            let coef = if j < k { params[j] } else { sigma };
            let se = (*cov_params.get(j, j)).sqrt();
            let stat = inference::compute_inference_stat(&normal, coef, se, z_crit);

            std_errors[j] = se;
            test_stats[j] = stat.stat;
            p_values[j] = stat.p_value;
            conf_lower[j] = stat.conf_low;
            conf_upper[j] = stat.conf_high;
        }

        let llf = log_likelihood(
            input.x(),
            input.y(),
            input.lower(),
            input.upper(),
            &params_full,
        );
        // `σ`は推定されるパラメータなので、`aic`/`bic`の罰則と`df_resid`はそれを含む総パラメータ数
        // `k+1`を使う。一方`df_model`（下）はWald検定の対象となる傾き係数の数で、`σ`は検定対象
        // ではないため`k - k_constant`になる（両者は意図的に別の数え方。
        // `docs/spec/tobit-spec.md`「適合度統計量」参照）。
        let aic = -2.0 * llf + 2.0 * ((k + 1) as f64);
        let bic = -2.0 * llf + (n as f64).ln() * ((k + 1) as f64);
        let df_resid = n - (k + 1);

        let k_constant = usize::from(input.has_intercept());
        let df_model = k - k_constant;
        let (wald_statistic, wald_p_value) = if df_model == 0 {
            // 検定対象の傾き係数が存在しない（切片のみ、または切片も無いモデル）。
            // OLSの`f_p_value`・Logit/Probitの`lr_p_value`と同じ扱いでNaNを返す
            // （0除算・自由度0のカイ二乗分布の構築を避ける）。
            (f64::NAN, f64::NAN)
        } else {
            wald_chi2_test(&params, &cov_params, k_constant, df_model)?
        };

        Ok(Self {
            input,
            params,
            sigma,
            cov_params,
            std_errors,
            test_stats,
            p_values,
            conf_lower,
            conf_upper,
            log_likelihood: llf,
            aic,
            bic,
            n_obs: n,
            df_model,
            df_resid,
            wald_statistic,
            wald_p_value,
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
    pub fn test_stats(&self) -> &[f64] {
        &self.test_stats
    }

    /// `test_stats`の従う分布（標準正規分布、自由度なし）。
    pub fn stat_dist(&self) -> inference::StatDist {
        inference::StatDist::Normal
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

    /// 収束点での対数尤度
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

    /// 観測数
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// モデルの自由度（切片以外の`β`の数）
    pub fn df_model(&self) -> usize {
        self.df_model
    }

    /// 残差自由度
    pub fn df_resid(&self) -> usize {
        self.df_resid
    }

    /// Wald検定統計量（`df_model==0`のときNaN）
    pub fn wald_statistic(&self) -> f64 {
        self.wald_statistic
    }

    /// Wald検定のp値（`df_model==0`のときNaN）
    pub fn wald_p_value(&self) -> f64 {
        self.wald_p_value
    }

    /// Wald検定の自由度（χ²、`df_model`）。`df_model==0`でNaNのときは`None`。
    pub fn wald_df(&self) -> Option<usize> {
        (self.df_model() > 0).then_some(self.df_model())
    }

    /// 収束したかどうか
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// 実際の反復回数
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// 限界効果（`marginal_effects`）。`fit()`とは独立した別メソッド（`fit()`のReturn
    /// 本体には含めない、`docs/spec/nonlinear-common.md`6章で確定済み）。`fit()`時の
    /// `cov_params`（`(k+1)×(k+1)`、`β∪{σ}`空間）を再利用するため再最適化は不要
    /// （`confidence_level`は`fit()`とは独立したパラメータとして受け取る、
    /// `LogitEstimator::marginal_effects`と同じ設計）。
    ///
    /// `target`で評価対象を選ぶ（`MarginalEffectsTarget`のdocコメント参照）。
    /// Logit/Probitの`overall_w_and_s`/`at_point_w_and_s`型の共通化はしない
    /// （計算式自体は`target_w_and_s`/`marginal_effects_from_tobit_w_s`
    /// のdocコメント参照）。左打ち切りのみ・右打ち切りのみ・両側打ち切りいずれの
    /// `TobitInput`でも同じ式で正しく計算できる（`target_w_and_s`のdocコメント「数式」参照、
    /// ユーザー確認済み）。
    ///
    /// 定数項（切片）は出力から除外する（Logit/Probitと同じ方針）。離散変数（0/1ダミー）の
    /// 自動判定は行わない（常に連続変数として扱う、Logit/Probitと同じ方針）。
    ///
    /// # Errors
    /// - `confidence_level`が`(0, 1)`の範囲外: `CommonError::InvalidConfidenceLevel`
    /// - `target`が`ExpectedUncensored`で、両側打ち切りの非打ち切り区間が`σ̂`に比べて狭すぎる:
    ///   `MleError::UncensoredIntervalTooNarrow`
    pub fn marginal_effects(
        &self,
        at: MarginalEffectsAt,
        target: MarginalEffectsTarget,
        confidence_level: f64,
    ) -> Result<MarginalEffects, MleError> {
        let x = self.input.x();
        let k = self.input.k();
        let lower = self.input.lower();
        let upper = self.input.upper();
        let normal = Normal::standard();
        if target == MarginalEffectsTarget::ExpectedUncensored {
            ensure_uncensored_interval_resolvable(lower, upper, self.sigma)?;
        }

        let (w, s_beta, s_sigma) = match at {
            MarginalEffectsAt::Overall => {
                overall_target_w_and_s(x, &self.params, self.sigma, lower, upper, target, &normal)
            }
            MarginalEffectsAt::Mean => at_point_target_w_and_s(
                &column_means(x),
                &self.params,
                self.sigma,
                lower,
                upper,
                target,
                &normal,
            ),
            MarginalEffectsAt::Median => at_point_target_w_and_s(
                &column_medians(x),
                &self.params,
                self.sigma,
                lower,
                upper,
                target,
                &normal,
            ),
        };

        marginal_effects_from_tobit_w_s(
            TobitMarginalEffectsInputs {
                param_names: self.input.param_names(),
                has_intercept: self.input.has_intercept(),
                k,
                beta: &self.params,
                cov_params: &self.cov_params,
            },
            w,
            &s_beta,
            s_sigma,
            confidence_level,
        )
    }

    /// 予測値（`predict`）。`target`で予測量を選ぶ（`MarginalEffectsTarget`を
    /// `marginal_effects`と共有する。「限界効果の対象」と「予測値の対象」は同じ3種
    /// （`E[y*|x]`・`E[y|x]`・`P(uncensored|x)`）を指すため、専用のenumを別途新設せず
    /// 再利用する）。`fit()`に使った学習データ（`self.input.x()`）の各行について返す
    /// （`predicted_value`のdocコメント「数式」参照。左のみ・右のみ・両側打ち切り
    /// いずれでも同じ式で正しく計算できる）。
    ///
    /// 新規データでの予測（out-of-sample）は`predict_new_data`。デフォルト
    /// （`target`省略時の`E[y|x]`、`docs/spec/nonlinear-common.md`
    /// 6章）はPython層（engine_pybind）の責務（`SolverType`/`CovType`等と同じ設計、
    /// `.claude/rules/rust-style.md`参照）。
    ///
    /// # Errors
    /// `target`が`ExpectedUncensored`で、両側打ち切りの非打ち切り区間が`σ̂`に比べて狭すぎる:
    /// `MleError::UncensoredIntervalTooNarrow`
    pub fn predict(&self, target: MarginalEffectsTarget) -> Result<Vec<f64>, MleError> {
        let x = self.input.x();
        let n = x.nrows();
        let k = self.input.k();
        let lower = self.input.lower();
        let upper = self.input.upper();
        let normal = Normal::standard();
        if target == MarginalEffectsTarget::ExpectedUncensored {
            ensure_uncensored_interval_resolvable(lower, upper, self.sigma)?;
        }
        Ok((0..n)
            .map(|i| {
                let mu: f64 = (0..k).map(|j| *x.get(i, j) * self.params[j]).sum();
                predicted_value(target, mu, self.sigma, lower, upper, &normal)
            })
            .collect())
    }

    /// 新規データ（out-of-sample、`new_x_columns`）に対する予測値。
    /// `target`は`predict`と同じ3種。
    ///
    /// `nonlinear::common::predict_new_data`に`predicted_value`（`mu`から`target`の
    /// 値を計算する関数）を部分適用したクロージャを`link`として渡すだけの薄い
    /// ラッパー（`LogitEstimator::predict_new_data`/`ProbitEstimator::predict_new_data`
    /// と同じ設計。`predict_new_data`は「`x_i'θ`を計算して`link`に渡す」処理しか
    /// 行わないため、`link`が`logistic`/正規分布CDFのような単純な関数でなくても
    /// （`target`・`sigma`・`lower`・`upper`を閉じ込めたクロージャでも）そのまま使える）。
    ///
    /// `new_x_columns`の本数・順序の契約、パニック条件は
    /// `nonlinear::common::predict_new_data`のdocコメント参照。
    ///
    /// # Errors
    /// `predict`と同じ（`MleError::UncensoredIntervalTooNarrow`）。
    pub fn predict_new_data(
        &self,
        target: MarginalEffectsTarget,
        new_x_columns: &[Vec<f64>],
    ) -> Result<Vec<f64>, MleError> {
        let has_intercept = self.input.has_intercept();
        let lower = self.input.lower();
        let upper = self.input.upper();
        let sigma = self.sigma;
        let normal = Normal::standard();
        if target == MarginalEffectsTarget::ExpectedUncensored {
            ensure_uncensored_interval_resolvable(lower, upper, sigma)?;
        }
        Ok(predict_new_data(
            &self.params,
            has_intercept,
            new_x_columns,
            |mu| predicted_value(target, mu, sigma, lower, upper, &normal),
        ))
    }

    /// 打ち切り予測の適合度チェック（Logit/Probitの`pred_table`の代替、`predict`とは独立
    /// した別メソッド）。`lower`/`upper`それぞれが定義する方向（下側打ち切り・非打ち切り・
    /// 上側打ち切り）ごとに、実測の比率（`y`がちょうど境界値に一致する観測の割合）と
    /// モデル含意の平均確率（各観測の`P(該当カテゴリ|xᵢ)`の平均、`predicted_value`の
    /// `ProbUncensored`と同じ`Φ`の組み合わせ）を突き合わせる（ユーザー確認済み、方向別
    /// 内訳を採用。単一集約値ではなく、両側打ち切りでどちらの境界に不整合があるかを
    /// 区別できる形にする）。`lower`/`upper`が`None`の方向は出力からも`None`になる。
    ///
    /// 3カテゴリの`model_implied_rate`は理論上必ず合計1になる
    /// （`Φ(za) + (Φ(zb)-Φ(za)) + (1-Φ(zb)) = 1`）。`observed_rate`も同様
    /// （定義上、3カテゴリで観測を排他的に分割するため）。
    ///
    /// **前提（rust-reviewer指摘）**: 打ち切り観測の判定は`yᵢ == lower`/`yᵢ == upper`という
    /// 浮動小数点の完全一致比較で行う。Tobitの定義上「打ち切り観測の`y`はちょうど境界値」
    /// という理論的性質自体は正しく、`TobitInput::from_columns`が`y`を変換せずそのまま
    /// 保持する設計（クリッピング等を行わない）とも整合するが、これは呼び出し側
    /// （`engine_pybind`、将来的にはPython層）が渡す`y`列の打ち切り観測の値と、`lower`/
    /// `upper`として渡す値がビット単位で完全に一致することを前提とする。CSV/Parquet
    /// 経由での読み込みや`Float32`→`Float64`変換等、この前提が崩れうる経路がある場合は
    /// 呼び出し側で丸め誤差が生じないよう注意する必要がある（許容誤差付き比較は
    /// 未導入。相対閾値が要る場合は`ensure_well_conditioned_symmetric_matrix`等と同じ
    /// 発想で別途検討する）。
    pub fn censoring_fit_check(&self) -> CensoringFitCheck {
        let x = self.input.x();
        let y = self.input.y();
        let n = x.nrows();
        let k = self.input.k();
        let lower = self.input.lower();
        let upper = self.input.upper();
        let normal = Normal::standard();

        let mut observed_lower = 0usize;
        let mut observed_upper = 0usize;
        let mut model_lower_sum = 0.0;
        let mut model_upper_sum = 0.0;
        let mut model_uncensored_sum = 0.0;

        for i in 0..n {
            let mu: f64 = (0..k).map(|j| *x.get(i, j) * self.params[j]).sum();
            let (_, _, cdf_za) = boundary_terms(lower, mu, self.sigma, &normal, true);
            let (_, _, cdf_zb) = boundary_terms(upper, mu, self.sigma, &normal, false);
            model_lower_sum += cdf_za;
            model_upper_sum += 1.0 - cdf_zb;
            model_uncensored_sum += cdf_zb - cdf_za;

            let y_i = *y.get(i, 0);
            if let Some(l) = lower
                && y_i == l
            {
                observed_lower += 1;
            }
            if let Some(u) = upper
                && y_i == u
            {
                observed_upper += 1;
            }
        }

        let n_f = n as f64;
        let lower_check = lower.map(|_| CensoringFitCategory {
            observed_rate: observed_lower as f64 / n_f,
            model_implied_rate: model_lower_sum / n_f,
        });
        let upper_check = upper.map(|_| CensoringFitCategory {
            observed_rate: observed_upper as f64 / n_f,
            model_implied_rate: model_upper_sum / n_f,
        });
        let uncensored_observed = (n - observed_lower - observed_upper) as f64 / n_f;

        CensoringFitCheck {
            lower: lower_check,
            uncensored: CensoringFitCategory {
                observed_rate: uncensored_observed,
                model_implied_rate: model_uncensored_sum / n_f,
            },
            upper: upper_check,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonlinear::common::SolverType;
    use crate::shared::error::ClusterSetting;
    use crate::shared::group_codes::GroupCodes;

    #[test]
    fn wald_chi2_test_keeps_precision_in_the_far_tail() {
        // `1.0 - cdf`では裾でp値が0.0に潰れる。傾き3個（`β=(10, 10, 10)`、`V=I`）で
        // wald=300。R: `pchisq(300, 3, lower.tail = FALSE)` = 9.94875834632771e-65。
        let params = [0.0, 10.0, 10.0, 10.0];
        let cov = Mat::from_fn(4, 4, |i, j| if i == j { 1.0 } else { 0.0 });
        let (wald, p) = wald_chi2_test(&params, &cov, 1, 3).unwrap();
        assert!((wald - 300.0).abs() < 1e-9);
        assert!((p / 9.948_758_346_327_71e-65 - 1.0).abs() < 1e-8);
    }

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
        // （`docs/spec/nonlinear-common.md`7章の「右打ち切りのみ」）。
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

    /// `scores`は`contribution`を観測あたり1回だけ計算して行を埋める。セルごとに再計算する
    /// 素朴な式と、ビット単位で一致することを確認する（左・右・非打ち切りの観測と、
    /// `|zeta|`が`U_CLAMP`を超える極端なパラメータ点を含む）。
    /// 参照式は本体と同じヘルパーを呼ぶため、検出するのは行列に詰める配置・順序の退行で、
    /// ヘルパー側の式の変更は別のテスト（閉形式・数値微分）が担う。
    #[test]
    fn scores_match_per_cell_reference_bit_for_bit() {
        let input = mixed_censoring_input();
        let problem = TobitProblem::new(&input);
        let normal = Normal::standard();
        // 通常点に加え、`zeta`が両側でクランプされる点（`β₀`の符号違い）と、`σ=e^s`が極小・
        // 極大の点を含める。
        for params in [
            vec![0.5, 0.3, -0.2],
            vec![-1.0, 2.0, 0.4],
            vec![30.0, -9.0, -1.5],
            vec![-30.0, -9.0, -1.5],
            vec![0.5, 0.3, -6.0],
            vec![0.5, 0.3, 4.0],
        ] {
            let scores = problem.scores(&params);
            let (k, sigma) = (problem.k(), params[problem.k()].exp());
            for i in 0..scores.nrows() {
                for col in 0..=k {
                    let c = problem.contribution(i, &params, &normal);
                    let expected = if col < k {
                        c.score_beta_coef * (*problem.x.get(i, col)) / sigma
                    } else {
                        c.score_s
                    };
                    assert_eq!(
                        scores.get(i, col).to_bits(),
                        expected.to_bits(),
                        "({i}, {col})"
                    );
                }
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
        // 通常の正規回帰の対数尤度に一致するはず（「打ち切りなし観測のみの
        // データで、cost/gradient/hessianがOLSの対数尤度と整合すること」という完了条件の境界ケース検算）。
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

    /// probit.rsと同型のバグ回帰ガード: `censored_contribution`の
    /// `A(u)=λ(u+λ)`計算で、クランプ済み`λ`と生の（非クランプの）`zeta`を混在させると、
    /// `|zeta|>U_CLAMP`の領域で`h_beta_coef`（`A(u)`）が負になりうる（`A(u)>0`という
    /// 恒等式が数値的に破れる、モジュール冒頭の数式表参照）。左打ち切り（`lower=0.0`）・
    /// 切片のみで`β0=1000, σ=1`とすると`zeta=(0-1000)/1=-1000`となり、この経路を踏む。
    #[test]
    fn hessian_weight_is_non_negative_even_when_censored_observation_exceeds_u_clamp() {
        let y = vec![0.0];
        let input =
            TobitInput::from_columns(&y, &[], vec![], true, "y".to_string(), Some(0.0), None)
                .unwrap();
        let problem = TobitProblem::new(&input);
        let params = vec![1000.0, 0.0]; // β0=1000, s=0（σ=1）→ zeta=(0-1000)/1=-1000

        let hessian = problem.hessian(&params).unwrap();
        assert!(
            hessian[0][0] >= 0.0,
            "Hessian weight A(u) should stay non-negative, got {}",
            hessian[0][0]
        );
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
    /// 異なる`n`除算のML推定量）に一致するはず（「打ち切りが極端に
    /// 少ないデータで、Newton法がOLSの閉形式解に近い値に収束すること」という完了条件の境界ケース）。
    #[test]
    fn fit_newton_converges_to_closed_form_solution_for_intercept_only_uncensored_data() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 10.0];
        let input = intercept_only_uncensored_input(&y);

        let n = y.len() as f64;
        let y_bar: f64 = y.iter().sum::<f64>() / n;
        let sse: f64 = y.iter().map(|v| (v - y_bar).powi(2)).sum();
        let expected_sigma = (sse / n).sqrt();

        let estimator = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

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
    /// `Var(β̂)=σ̂²/n`・`Var(ŝ)=1/(2n)`・`Cov(β̂,ŝ)=0`が導ける（`docs/spec/tobit-spec.md`
    /// 3.1〜3.3節の一般形の特殊ケース）。
    /// `σ`のデルタ法変換（ヤコビアン`diag(1,σ)`）を適用すると、この対角性はそのまま
    /// 保たれ`Var(σ̂)≈σ̂²Var(ŝ)=σ̂²/(2n)`・`Cov(β̂,σ̂)≈σ̂Cov(β̂,ŝ)=0`になる。
    /// Logitの`fit_computes_std_errors_test_stats_p_values_and_ci_matching_closed_form_for_
    /// intercept_only_model`と同じ設計（本体実装と同じ式を繰り返すのではなく、独立に
    /// 導出した閉じた形の解でSE・z値・p値・信頼区間を検算する）。
    #[test]
    fn fit_computes_std_errors_test_stats_p_values_and_ci_matching_closed_form_for_intercept_only_uncensored_model()
     {
        let y = vec![1.0, 2.0, 3.0, 4.0, 10.0];
        let input = intercept_only_uncensored_input(&y);

        let n = y.len() as f64;
        let y_bar: f64 = y.iter().sum::<f64>() / n;
        let sse: f64 = y.iter().map(|v| (v - y_bar).powi(2)).sum();
        let sigma_hat = (sse / n).sqrt();

        let expected_var_beta = sigma_hat.powi(2) / n;
        let expected_var_sigma = sigma_hat.powi(2) / (2.0 * n);

        let estimator = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

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
        assert!((estimator.test_stats()[0] - expected_z_beta).abs() < 1e-6);
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
        assert!((estimator.test_stats()[1] - expected_z_sigma).abs() < 1e-6);
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

    /// 切片のみ・打ち切りなしのTobit（通常の正規分布のMLEに一致する）の対数尤度は
    /// 閉じた形の解析解`ℓ=-(n/2)(ln(2π)+ln(σ̂²)+1)`を持つ（OLSの`log_likelihood`と同じ式。
    /// `σ̂²=Σ(y-ȳ)²/n`はMLE推定量で、OLSの不偏推定量（`n-1`除算）とは異なる）。
    /// `aic`/`bic`はこの`llf`と総パラメータ数`k+1=2`（切片1+`logσ`1）から直接計算できる。
    /// 切片のみモデルは傾き係数が無い（`df_model=0`）ため、Wald検定がスキップされ
    /// `wald_statistic`/`wald_p_value`がNaNになることも合わせて検証する。
    #[test]
    fn fit_computes_log_likelihood_aic_bic_matching_closed_form_for_intercept_only_uncensored_model()
     {
        let y = vec![1.0, 2.0, 3.0, 4.0, 10.0];
        let input = intercept_only_uncensored_input(&y);

        let n = y.len() as f64;
        let y_bar: f64 = y.iter().sum::<f64>() / n;
        let sse: f64 = y.iter().map(|v| (v - y_bar).powi(2)).sum();
        let sigma_sq_hat = sse / n;
        let expected_llf =
            -(n / 2.0) * ((2.0 * std::f64::consts::PI).ln() + sigma_sq_hat.ln() + 1.0);
        let expected_aic = -2.0 * expected_llf + 2.0 * 2.0;
        let expected_bic = -2.0 * expected_llf + n.ln() * 2.0;

        let estimator = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        assert!((estimator.log_likelihood() - expected_llf).abs() < 1e-6);
        assert!((estimator.aic() - expected_aic).abs() < 1e-6);
        assert!((estimator.bic() - expected_bic).abs() < 1e-6);
        assert_eq!(estimator.n_obs(), 5);
        assert_eq!(estimator.df_model(), 0);
        assert_eq!(estimator.df_resid(), 3);
        assert!(estimator.wald_statistic().is_nan());
        assert!(estimator.wald_p_value().is_nan());
    }

    /// Wald検定統計量・p値が独立再計算（2×2部分行列の手動逆行列計算、`Llt`分解を経由
    /// しない）と一致することを検証する。`multivariate_censored_input`（切片+`x1`+`x2`、
    /// `k=3`、傾き係数`q=2`）を使う。
    #[test]
    fn fit_wald_statistic_and_p_value_match_independently_recomputed_values() {
        let estimator = TobitEstimator::fit(
            multivariate_censored_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        // 傾き係数（切片を除く`params[1]`・`params[2]`）と、対応する`cov_params`の
        // 2×2部分行列（`σ`次元は含まれない）を使い、2×2逆行列の公式で手動計算する。
        let beta1 = estimator.params()[1];
        let beta2 = estimator.params()[2];
        let v = estimator.cov_params();
        let (v11, v12, v21, v22) = (*v.get(1, 1), *v.get(1, 2), *v.get(2, 1), *v.get(2, 2));
        let det = v11 * v22 - v12 * v21;
        let (inv11, inv12, inv21, inv22) = (v22 / det, -v12 / det, -v21 / det, v11 / det);
        let expected_wald =
            beta1 * (inv11 * beta1 + inv12 * beta2) + beta2 * (inv21 * beta1 + inv22 * beta2);

        let chi2 = ChiSquared::new(2.0).unwrap();
        let expected_p = 1.0 - chi2.cdf(expected_wald);

        assert_eq!(estimator.df_model(), 2);
        assert!((estimator.wald_statistic() - expected_wald).abs() < 1e-6);
        assert!((estimator.wald_p_value() - expected_p).abs() < 1e-9);
    }

    /// `df_model=1`（傾き係数が1個）のとき、`W=z²`（カイ二乗(1)統計量は標準正規のz値の
    /// 2乗に一致するという代数的恒等式）が成り立つことを検証する。`fit_wald_statistic_
    /// and_p_value_match_independently_recomputed_values`（`q=2`、手動2×2逆行列）とは
    /// 別角度の検算（OLSの`wald_test_last_columns_matches_squared_t_statistic_for_
    /// single_column`と同じ発想）。`censored_regression_input`（`q=1`）を使う。
    #[test]
    fn fit_wald_statistic_matches_squared_z_statistic_for_single_slope() {
        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        assert_eq!(estimator.df_model(), 1);
        let z_slope = estimator.test_stats()[1];
        assert!((estimator.wald_statistic() - z_slope.powi(2)).abs() < 1e-9);

        let chi2 = ChiSquared::new(1.0).unwrap();
        let expected_p = 1.0 - chi2.cdf(estimator.wald_statistic());
        assert!((estimator.wald_p_value() - expected_p).abs() < 1e-9);
    }

    /// `cov_type=Cluster`のクラスターロバスト共分散`Ŝ=Σ_g S_gS_g'`は、クラスター寄与
    /// スコアの総和がゼロ（MLEの一次条件`Σ_i s_i = 0`）になるため`rank(Ŝ) ≤ G - 1`。
    /// `multivariate_censored_input`（傾き係数`q=2`）に対し`G=2`（`G=q`ちょうど）を
    /// 組み合わせると`q×q`部分行列は`rank ≤ 1 < 2`で構造的に特異になる。`G`・`q`は
    /// 入力だけから判定できるため、`fit()`冒頭のバリデーション
    /// （`validate_cluster_cov_type` → `validate_cluster_count_covers_slopes`）が
    /// `CommonError::InsufficientClustersForInference`で弾く（
    /// 従来は`wald_chi2_test`内の`ComputationFailed`だった）。`wald_chi2_test`の
    /// `ensure_well_conditioned_symmetric_matrix`側のbackstop（`g > q`だが悪条件で
    /// 数値的にほぼ特異なケース）は、`wald_f_test`と共有する純粋な線形代数
    /// ユーティリティであり`shared/linear_algebra.rs`の単体テスト・OLSの
    /// `fit_returns_computation_failed_for_extreme_scale_difference_in_f_test`で担保する
    /// （Tobitは`TobitScaling`が`x`を内部で列標準化するため、極端なスケール差による
    /// 悪条件をTobitの`fit()`経由で再現するのが困難で、Tobit固有テストは設けない）。
    #[test]
    fn fit_returns_validation_error_when_cluster_count_equals_slopes() {
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];
        let result = TobitEstimator::fit(
            multivariate_censored_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&groups)),
                },
                confidence_level: 0.95,
            },
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InsufficientClustersForInference { g: 2, q: 2 })
        );
    }

    /// 多変量（説明変数が1つ、切片込みで`k=2`・`s`を含め合計3パラメータ）の場合、
    /// 閉じた形の解析解は無いため、`cov_params`の対称性・各種統計量の内部整合性
    /// （z値・信頼区間の定義式通りの関係）を検証する回帰テスト。Logitの
    /// `fit_cov_params_is_symmetric_and_stats_are_internally_consistent`と同じ設計。
    /// 特に`TobitScaling::destandardize_cov_params`（`J·Cov·Jᵀ`）の適用が非対角成分も
    /// 含めて正しく機能しているかを確認する（対角成分だけでは転置ミス等の一部のバグを
    /// 検出できない）。
    #[test]
    fn fit_cov_params_is_symmetric_and_stats_are_internally_consistent() {
        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
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
            assert!((estimator.test_stats()[j] - coef / se).abs() < 1e-9);
            assert!(
                (estimator.p_values()[j]
                    - 2.0 * (1.0 - normal.cdf(estimator.test_stats()[j].abs())))
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
    /// 確認済み（`max_iter=2`以降は`cov_params`計算が安定する。真の収束は`max_iter=11`、
    /// `fit_returns_unconverged_result_without_raising_when_raise_on_non_convergence_is_false`
    /// のコメント参照）。
    #[test]
    fn fit_returns_singular_hessian_error_when_cov_params_computation_fails_at_truncated_point() {
        let input = censored_regression_input();
        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 1,
                tol: 1e-12,
                raise_on_non_convergence: false,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert!(
            matches!(result, Err(MleError::SingularHessian)),
            "{result:?}"
        );
    }

    /// 上のテストは`cov_type=Classical`（`observed_information_cov_params`）のみで
    /// `SingularHessian`エラー伝播を検証しているが、`sandwich_cov_params`（`Hc0`/`Hc1`）も
    /// 内部で同じHessianの逆行列計算を行うため、同じ打ち切り点で同じエラーが伝播する
    /// はず。Logit/Probitの`fit_returns_singular_design_matrix_error_for_perfectly_
    /// collinear_design_matrix`（`solver`×`cov_type`を1テストに集約）と同じ
    /// 「cov_type分岐ごとのエラー伝播`?`」のギャップパターンを
    /// Tobitでも確認する（`cargo llvm-cov`で発覚）。
    #[test]
    fn fit_returns_singular_hessian_error_when_cov_params_computation_fails_at_truncated_point_with_hc0_and_hc1()
     {
        for cov_type in [CovType::Hc0, CovType::Hc1] {
            let result = TobitEstimator::fit(
                censored_regression_input(),
                MleFitOptions {
                    solver: SolverType::Newton,
                    max_iter: 1,
                    tol: 1e-12,
                    raise_on_non_convergence: false,
                    cov_type: cov_type.clone(),
                    confidence_level: 0.95,
                },
            );
            assert!(
                matches!(result, Err(MleError::SingularHessian)),
                "cov_type={cov_type:?}, result={result:?}"
            );
        }
    }

    /// `cluster_cov_params`（`Cluster`）も同じ理由で`SingularHessian`が伝播するはず
    /// （上記テストのdocコメント参照）。
    #[test]
    fn fit_returns_singular_hessian_error_when_cov_params_computation_fails_at_truncated_point_with_cluster()
     {
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];
        let result = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 1,
                tol: 1e-12,
                raise_on_non_convergence: false,
                cov_type: CovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&groups)),
                },
                confidence_level: 0.95,
            },
        );
        assert!(
            matches!(result, Err(MleError::SingularHessian)),
            "{result:?}"
        );
    }

    /// 説明変数ありのモデルでも、打ち切りが実質発生しないデータではNewton法がOLSの
    /// 閉じた形の解（正規方程式）に近い値に収束するはず（完了条件の本体、
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

        let estimator = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

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
        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 1.5,
            },
        );
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
        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 0,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::InvalidMaxIter { max_iter: 0 }
        );
    }

    #[test]
    fn fit_returns_invalid_tol_error() {
        let input = intercept_only_uncensored_input(&[1.0, 2.0, 3.0]);
        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 0.0,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert_eq!(result.unwrap_err(), MleError::InvalidTol { tol: 0.0 });
    }

    #[test]
    fn fit_returns_insufficient_observations_error() {
        // n=2, 総パラメータ数k+1=2(切片1+logσ1) → n<=k+1でエラー
        // （`validate_sufficient_observations`にx列数ではなくk+1を渡す設計）。
        let input = intercept_only_uncensored_input(&[1.0, 2.0]);
        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        let err = result.unwrap_err();
        assert_eq!(
            err,
            MleError::Common(CommonError::InsufficientObservations { n: 2, k: 2 })
        );
        // `n_params`は`β`の数に`log σ`を足した値になる。
        let report = err.try_report().expect("cataloged");
        assert_eq!(
            report.message(),
            "Not enough observations: 2 observations for 2 parameters; at least 3 are needed."
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

        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
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
        let estimator = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        assert!(estimator.converged());
        assert_eq!(estimator.params().len(), 2);
        assert!(estimator.sigma() > 0.0 && estimator.sigma().is_finite());
        for &p in estimator.params() {
            assert!(p.is_finite());
        }
        // `n_iter()`が他のどのテストでも未使用だった（`cargo llvm-cov`で発覚）。
        // 収束時は`0 < n_iter <= max_iter`のはず。
        assert!(estimator.n_iter() > 0 && estimator.n_iter() <= 100);
    }

    #[test]
    fn fit_bfgs_and_lbfgs_converge_to_similar_solution_as_newton_for_censored_data() {
        let newton = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        for solver in [SolverType::Bfgs, SolverType::Lbfgs] {
            let estimator = TobitEstimator::fit(
                censored_regression_input(),
                MleFitOptions {
                    solver,
                    max_iter: 200,
                    tol: 1e-8,
                    raise_on_non_convergence: true,
                    cov_type: CovType::Classical,
                    confidence_level: 0.95,
                },
            )
            .unwrap();
            assert!(estimator.converged(), "solver={:?}", solver);
            for (a, b) in estimator.params().iter().zip(newton.params()) {
                assert!((a - b).abs() < 1e-3, "solver={:?}, a={a}, b={b}", solver);
            }
            assert!(
                (estimator.sigma() - newton.sigma()).abs() < 1e-3,
                "solver={:?}, sigma={}, newton_sigma={}",
                solver,
                estimator.sigma(),
                newton.sigma()
            );
        }
    }

    /// `censored_regression_input`の`x`（`1..8`）は`TobitScaling`の列標準化が実質no-opに
    /// なるほど自明なスケールではないが、`fit_bfgs_and_lbfgs_agree_with_newton_when_design_
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

        let newton = TobitEstimator::fit(
            make_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        assert!(newton.converged());

        for solver in [SolverType::Bfgs, SolverType::Lbfgs] {
            let estimator = TobitEstimator::fit(
                make_input(),
                MleFitOptions {
                    solver,
                    max_iter: 200,
                    tol: 1e-8,
                    raise_on_non_convergence: true,
                    cov_type: CovType::Classical,
                    confidence_level: 0.95,
                },
            )
            .unwrap();
            assert!(estimator.converged(), "solver={:?}", solver);
            for (a, b) in estimator.params().iter().zip(newton.params()) {
                assert!((a - b).abs() < 1e-3, "solver={:?}, a={a}, b={b}", solver);
            }
            assert!(
                (estimator.sigma() - newton.sigma()).abs() < 1e-3,
                "solver={:?}, sigma={}, newton_sigma={}",
                solver,
                estimator.sigma(),
                newton.sigma()
            );
        }
    }

    #[test]
    fn population_std_returns_population_std_and_falls_back_to_one_for_constant() {
        // 母集団分散 = ((1.5)² + (0.5)² + (0.5)² + (1.5)²)/4 = 1.25 → std = √1.25
        let v = [1.0, 2.0, 3.0, 4.0];
        assert!((population_std(v.iter().copied(), 4) - 1.25_f64.sqrt()).abs() < 1e-12);

        // 定数（std=0）のときは0除算回避のため`1.0`にフォールバックする。
        assert_eq!(population_std([7.0, 7.0, 7.0].iter().copied(), 3), 1.0);
    }

    /// `TobitScaling`の逆変換（`destandardize_params`）が、標準化した設計行列・`y`で
    /// 評価した線形予測子`x̃'β̃`を元のスケールの`x'β/c`に戻すことを、切片あり・切片なし
    /// 両方で確認する（往復変換の一貫性、`ColumnScale`の往復テストのTobit版）。
    #[test]
    fn tobit_scaling_destandardize_params_inverts_the_linear_predictor() {
        for has_intercept in [true, false] {
            // 切片あり: 列0が定数1.0。切片なし: 定数列なし。
            let (cols, k): (Vec<Vec<f64>>, usize) = if has_intercept {
                (
                    vec![
                        vec![1.0; 5],
                        vec![10.0, 12.0, 9.0, 15.0, 11.0],
                        vec![100.0, 400.0, 900.0, 1600.0, 2500.0],
                    ],
                    3,
                )
            } else {
                (
                    vec![
                        vec![2.0, 4.0, 6.0, 8.0, 10.0],
                        vec![1.0, 3.0, 2.0, 5.0, 4.0],
                    ],
                    2,
                )
            };
            let x = Mat::from_fn(5, k, |i, j| cols[j][i]);
            let y = Mat::from_fn(5, 1, |i, _| [3.0, 7.0, 5.0, 11.0, 9.0][i] * 50.0);
            let scaling = TobitScaling::fit(&x, &y, has_intercept);
            let x_std = scaling.standardize_x(&x);

            // 標準化空間の任意のパラメータ`(β̃, s̃)`。
            let params_std: Vec<f64> = (0..=k).map(|j| 0.3 + 0.2 * j as f64).collect();
            let params_orig = scaling.destandardize_params(&params_std);

            // 各観測で `x̃ᵢ'β̃ == xᵢ'β / c` が成り立つはず。
            for i in 0..5 {
                let lhs: f64 = (0..k).map(|j| *x_std.get(i, j) * params_std[j]).sum();
                let rhs: f64 =
                    (0..k).map(|j| *x.get(i, j) * params_orig[j]).sum::<f64>() / scaling.y_scale;
                assert!(
                    (lhs - rhs).abs() < 1e-9,
                    "has_intercept={has_intercept}, i={i}"
                );
            }
            // `s̃ = logσ - ln c`
            assert!((params_orig[k] - (params_std[k] + scaling.y_scale.ln())).abs() < 1e-12);
        }
    }

    /// 標準化に使う`y`のスケール`y_scale`は、`y`のスケールによらず厳密に2の冪になる。
    /// 掛け算の順序で値が変わらない性質（`tobit_scaling_destandardize_matches_the_closed_form_bit_for_bit`）
    /// と、反復軌道が標準化前と厳密に相似に保たれる性質の前提。
    #[test]
    fn tobit_scaling_y_scale_is_an_exact_power_of_two() {
        let x = Mat::from_fn(5, 2, |i, j| {
            if j == 0 {
                1.0
            } else {
                [1.0, 3.0, 2.0, 5.0, 4.0][i]
            }
        });
        for factor in [1e-6, 0.01, 0.3, 1.0, 7.0, 50.0, 1e4, 1e9] {
            let y = Mat::from_fn(5, 1, |i, _| [3.0, 7.0, 5.0, 11.0, 9.0][i] * factor);
            let scale = TobitScaling::fit(&x, &y, true).y_scale;
            assert!(scale > 0.0 && scale.is_finite(), "factor={factor}");
            // 2の冪は仮数部が0（`log2`が整数）で、`powi`で戻すと元の値に厳密に一致する。
            let exp = scale.log2();
            assert_eq!(exp.fract(), 0.0, "factor={factor}, y_scale={scale}");
            assert_eq!(2.0_f64.powi(exp as i32), scale, "factor={factor}");
        }
    }

    /// `TobitScaling`の逆変換・ヤコビアンが、`ColumnScale`に委ねる前の式
    /// （`βⱼ = c·β̃ⱼ/stdⱼ`、`β₀ = c·β̃₀ - Σ x̄ⱼβⱼ`、`J`の対角`c/stdⱼ`・第0行`-c·x̄ⱼ/stdⱼ`）と
    /// ビット単位で一致する。`y_scale`が2の冪なので、掛ける順序が違っても値が変わらないことを
    /// 固定する。許容誤差を使わないのは、丸め誤差が入ってもテストが気づけなくなるため。
    /// `y`のスケールを変えて`c≠1`（大小両方）の場合も確かめる。
    #[test]
    fn tobit_scaling_destandardize_matches_the_closed_form_bit_for_bit() {
        let n = 6;
        for has_intercept in [true, false] {
            let cols: Vec<Vec<f64>> = if has_intercept {
                vec![
                    vec![1.0; n],
                    vec![10.0, 12.0, 9.0, 15.0, 11.0, 13.0],
                    vec![100.0, 400.0, 900.0, 1600.0, 2500.0, 3600.0],
                ]
            } else {
                vec![
                    vec![2.0, 4.0, 6.0, 8.0, 10.0, 7.0],
                    vec![1.0, 3.0, 2.0, 5.0, 4.0, 9.0],
                    vec![0.5, 0.1, 0.9, 0.3, 0.7, 0.2],
                ]
            };
            let k = cols.len();
            let x = Mat::from_fn(n, k, |i, j| cols[j][i]);
            let start = usize::from(has_intercept);

            for y_factor in [0.003, 1.0, 40.0, 2500.0] {
                let y = Mat::from_fn(n, 1, |i, _| [3.0, 7.0, 5.0, 11.0, 9.0, 4.0][i] * y_factor);
                let scaling = TobitScaling::fit(&x, &y, has_intercept);
                let c = scaling.y_scale;

                // `x`の平均・母集団標準偏差を、実装とは独立に（切片列・切片なしでは平均0）求める。
                let means: Vec<f64> = (0..k)
                    .map(|j| {
                        if has_intercept && j >= start {
                            cols[j].iter().sum::<f64>() / n as f64
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let stds: Vec<f64> = (0..k)
                    .map(|j| {
                        if j < start {
                            return 1.0;
                        }
                        let m = cols[j].iter().sum::<f64>() / n as f64;
                        let var = cols[j].iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64;
                        if var.sqrt() > 0.0 { var.sqrt() } else { 1.0 }
                    })
                    .collect();

                let params_std: Vec<f64> = (0..=k).map(|j| 0.3 + 0.2 * j as f64).collect();
                let mut expected = vec![0.0; k + 1];
                for j in start..k {
                    expected[j] = c * params_std[j] / stds[j];
                }
                if has_intercept {
                    expected[0] =
                        c * params_std[0] - (1..k).map(|j| means[j] * expected[j]).sum::<f64>();
                }
                expected[k] = params_std[k] + c.ln();

                let actual = scaling.destandardize_params(&params_std);
                for j in 0..=k {
                    assert_eq!(
                        actual[j].to_bits(),
                        expected[j].to_bits(),
                        "has_intercept={has_intercept}, y_factor={y_factor}, j={j}: \
                         {} vs {}",
                        actual[j],
                        expected[j]
                    );
                }

                let sigma = 1.7;
                let jac = scaling.param_jacobian(sigma);
                for i in 0..=k {
                    for m in 0..=k {
                        let expected_ij = if i == k {
                            if m == k { sigma } else { 0.0 }
                        } else if m == k {
                            0.0
                        } else if has_intercept && i == 0 {
                            if m == 0 { c } else { -c * means[m] / stds[m] }
                        } else if m == i {
                            c / stds[i]
                        } else {
                            0.0
                        };
                        assert_eq!(
                            jac.get(i, m).to_bits(),
                            expected_ij.to_bits(),
                            "has_intercept={has_intercept}, y_factor={y_factor}, J[{i}][{m}]"
                        );
                    }
                }
            }
        }
    }

    /// `TobitScaling::param_jacobian`（`destandardize_cov_params`が両側から掛ける`J`）が、
    /// `(β̃, s̃) → (β, σ)`の写像の解析ヤコビアンと一致することを、中心差分で独立に検証する
    /// （`expected_cov_params`ヘルパーは最終段で`destandardize_cov_params`を本体と共用して
    /// おり`param_jacobian`のバグを検出できないため、rust-reviewer指摘で追加）。特に
    /// 切片行の非対角項`∂β₀/∂β̃ⱼ = -c·x̄ⱼ/stdⱼ`と末尾行`∂σ/∂s̃ = σ`（`σ = c·exp(s̃)`）を
    /// カバーする。写像は`g(p)ⱼ = destandardize_params(p)ⱼ`（`j<k`）・
    /// `g(p)_k = exp(destandardize_params(p)_k)`。
    #[test]
    fn param_jacobian_matches_finite_difference_of_the_destandardization_map() {
        for has_intercept in [true, false] {
            let (cols, k): (Vec<Vec<f64>>, usize) = if has_intercept {
                (
                    vec![
                        vec![1.0; 6],
                        vec![10.0, 12.0, 9.0, 15.0, 11.0, 13.0],
                        // 切片と強く相関する（平均が大きい）列。センタリング項を効かせる。
                        vec![100.0, 121.0, 81.0, 225.0, 121.0, 169.0],
                    ],
                    3,
                )
            } else {
                (
                    vec![
                        vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                        vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
                    ],
                    2,
                )
            };
            let x = Mat::from_fn(6, k, |i, j| cols[j][i]);
            let y = Mat::from_fn(6, 1, |i, _| [3.0, 7.0, 5.0, 11.0, 9.0, 8.0][i] * 30.0);
            let scaling = TobitScaling::fit(&x, &y, has_intercept);

            let params_std: Vec<f64> = (0..=k).map(|j| 0.2 + 0.15 * j as f64).collect();
            let g = |p: &[f64]| -> Vec<f64> {
                let orig = scaling.destandardize_params(p);
                (0..=k)
                    .map(|i| if i < k { orig[i] } else { orig[k].exp() })
                    .collect()
            };
            let sigma = g(&params_std)[k];
            let analytic = scaling.param_jacobian(sigma);

            let h = 1e-6;
            for m in 0..=k {
                let mut plus = params_std.clone();
                let mut minus = params_std.clone();
                plus[m] += h;
                minus[m] -= h;
                let gp = g(&plus);
                let gm = g(&minus);
                for i in 0..=k {
                    let numeric = (gp[i] - gm[i]) / (2.0 * h);
                    let a = *analytic.get(i, m);
                    assert!(
                        (a - numeric).abs() <= 1e-5 * a.abs().max(1.0),
                        "has_intercept={has_intercept} ({i},{m}): analytic={a}, numeric={numeric}"
                    );
                }
            }
        }
    }

    /// `y`（および打ち切り境界）を大きなスケールに引き伸ばしても、`fit()`が生スケールの
    /// データと数学的に同値な推定値に収束することを確認する回帰テスト。
    ///
    /// `y`を標準化せず最適化していた頃は、`y`のスケールが大きいと健全なMLE解でも標準化
    /// パラメータ空間のL2ノルムが大きくなり、当時あった標準化パラメータのノルムによる
    /// (準)分離ヒューリスティックが誤発火して失敗していた（Wooldridge mroz `hours`の
    /// 生スケールTobitで発覚）。`TobitScaling`が`y`を2の冪に丸めたスケールでスケーリング
    /// してから最適化し収束後に逆変換することで解消した（本テストは`TobitScaling`による
    /// スケール同値性の回帰テストとして維持する）。
    ///
    /// `y' = c·y`・`lower' = c·lower`（`c`は2の冪）とすると`β' = c·β`・`σ' = c·σ`・
    /// `SE' = c·SE`が厳密に成り立ち（`TobitScaling`が`y_scale`も`c`倍の2の冪に丸めるため、
    /// 両者は完全に同一の標準化問題を解く）、z統計量・p値・Wald統計量はスケール不変。
    #[test]
    fn fit_converges_for_large_response_scale_without_false_separation() {
        let c = 1024.0;
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y_base = vec![0.0, 0.0, 1.15, 2.9, 5.2, 6.85, 9.1, 10.95];
        let y_scaled: Vec<f64> = y_base.iter().map(|v| v * c).collect();

        let make = |y: &[f64], lower: f64| {
            TobitInput::from_columns(
                y,
                std::slice::from_ref(&x),
                vec!["x1".to_string()],
                true,
                "y".to_string(),
                Some(lower),
                None,
            )
            .unwrap()
        };
        let fit = |y: &[f64], lower: f64| {
            TobitEstimator::fit(
                make(y, lower),
                MleFitOptions {
                    solver: SolverType::Newton,
                    max_iter: 100,
                    tol: 1e-8,
                    raise_on_non_convergence: true,
                    cov_type: CovType::Classical,
                    confidence_level: 0.95,
                },
            )
            .unwrap()
        };

        let base = fit(&y_base, 0.0);
        let scaled = fit(&y_scaled, 0.0);

        assert!(scaled.converged());

        let rtol = |a: f64, b: f64| (a - b).abs() <= 1e-8 * b.abs().max(1.0);
        for (b, s) in base.params().iter().zip(scaled.params()) {
            assert!(rtol(s / c, *b), "beta: base={b}, scaled/c={}", s / c);
        }
        assert!(
            rtol(scaled.sigma() / c, base.sigma()),
            "sigma: base={}, scaled/c={}",
            base.sigma(),
            scaled.sigma() / c
        );
        for (b, s) in base.std_errors().iter().zip(scaled.std_errors()) {
            assert!(rtol(s / c, *b), "se: base={b}, scaled/c={}", s / c);
        }
        for (b, s) in base.test_stats().iter().zip(scaled.test_stats()) {
            assert!(rtol(*s, *b), "z: base={b}, scaled={s}");
        }
        assert!(
            rtol(scaled.wald_statistic(), base.wald_statistic()),
            "wald: base={}, scaled={}",
            base.wald_statistic(),
            scaled.wald_statistic()
        );
        // 対数尤度は変数変換のヤコビアン分ずれる（非打ち切り観測ごとに `-ln c`）。
        let n_uncensored = y_base.iter().filter(|&&v| v != 0.0).count() as f64;
        assert!(
            rtol(
                scaled.log_likelihood() + n_uncensored * c.ln(),
                base.log_likelihood()
            ),
            "llf: base={}, scaled+adj={}",
            base.log_likelihood(),
            scaled.log_likelihood() + n_uncensored * c.ln()
        );
    }

    /// 上のテストは切片あり・`c`が2の冪（＝標準化問題が両者で完全一致）だが、こちらは
    /// (a) **切片なし**モデル、(b) **`y`の標準偏差が2の冪ちょうどではない**大スケール、で
    /// `fit()`が既定Newtonで収束し真の傾き（生成係数）を回復することを確認する
    /// （`TobitScaling`の切片なし経路の end-to-end カバレッジと、`y_scale`が恒等変換で
    /// ない一般ケースの分離ヒューリスティック誤発火回避のリグレッションガード、
    /// rust-reviewer指摘で追加）。
    #[test]
    fn fit_converges_for_large_non_power_of_two_response_scale_without_intercept() {
        // 潜在変数 y* = 40·x（切片なし）、生成 σ ≈ 300 → y の std は数百オーダーで
        // 2の冪（256 / 512）ちょうどにはならない。左打ち切り 0。
        let x: Vec<f64> = (1..=40).map(|i| i as f64).collect();
        let mut rng_state = 0x1234_5678_u64;
        let mut next = || {
            // 決定的な xorshift でノイズを作る（テストの再現性、外部 rng 非依存）。
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            ((rng_state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 600.0
        };
        let y: Vec<f64> = x.iter().map(|&xi| (40.0 * xi + next()).max(0.0)).collect();
        let input = TobitInput::from_columns(
            &y,
            std::slice::from_ref(&x),
            vec!["x1".to_string()],
            false,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap();

        let est = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        assert!(est.converged());
        // 傾きは真値 40 の近傍（打ち切り＋ノイズがあるため緩め）。以前は`y`の
        // 大スケールで当時のノルムによる分離ヒューリスティックが誤発火し`.unwrap()`が
        // panicしていた（現在はこのヒューリスティック自体が無い）。
        assert!(
            (est.params()[0] - 40.0).abs() < 5.0,
            "slope={}, expected≈40",
            est.params()[0]
        );
        assert!(est.sigma() > 0.0 && est.sigma().is_finite());
    }

    #[test]
    fn fit_returns_non_convergence_error_when_max_iter_is_too_small_and_raise_is_true() {
        let input = censored_regression_input();
        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 1,
                tol: 1e-12,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert!(
            matches!(result, Err(MleError::NonConvergence { .. })),
            "{:?}",
            result
        );
    }

    /// Tobitの(準)完全分離は`MleError::SeparationDetected`ではなく
    /// `MleError::NonConvergence`として現れることを固定する。
    ///
    /// `fit()`はLogit/Probit用の分離検出（`nonlinear::separation`）を呼ばない。Tobitの分離は
    /// 係数の発散ではなく`σ→0`への退化として現れるため、この失敗モードが`NonConvergence`で
    /// あること自体を、Tobitの分離の現れ方の正本として固定する（`SeparationDetected`と
    /// `NonConvergence`はどちらも`ComputationError`にマップされるため、Python API境界の
    /// テストでは区別できない）。
    ///
    /// DGP: `y* = 100·x1 + 0.5·x2`（ノイズ皆無）を`lower=0`で左打ち切り。非打ち切り
    /// 観測は存在する（`NoUncensoredObservations`は通過する）が、分離方向に沿って
    /// `σ→0`へ退化し、Newtonは`max_iter`まで収束しない。
    #[test]
    fn fit_returns_non_convergence_not_separation_detected_for_noise_free_separation() {
        let n = 120;
        let x1: Vec<f64> = (0..n)
            .map(|i| -2.0 + 4.0 * (i as f64) / ((n - 1) as f64))
            .collect();
        let x2: Vec<f64> = (0..n).map(|i| (0.7 * i as f64).sin()).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (100.0 * x1[i] + 0.5 * x2[i]).max(0.0))
            .collect();
        // 非打ち切り観測が1件以上あることを前提とする（無ければ
        // `NoUncensoredObservations`が先に返り、このテストの意図がずれる）。
        assert!(y.iter().any(|&v| v > 0.0));

        let input = TobitInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap();

        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert!(
            matches!(result, Err(MleError::NonConvergence { .. })),
            "expected NonConvergence, got {result:?}"
        );
    }

    #[test]
    fn fit_returns_unconverged_result_without_raising_when_raise_on_non_convergence_is_false() {
        // `max_iter=1`（このテストの元々の値）だと、`censored_regression_input`の
        // 打ち切り点（Newtonの初回ステップ、まだ真の最尤推定点から遠い）でHessianが
        // 不定符号になり、`fit`が非収束時でも`cov_params`を計算するようになったことで
        // `SingularHessian`が先に発生してしまう（実測で確認、
        // `max_iter=1`は`SingularHessian`、`max_iter=2`以降で`cov_params`計算が
        // 安定し`converged=false`が返るようになる。真の収束は`max_iter=11`）。
        // `max_iter=3`のまま（`max_iter=2`でも通るが余裕を持たせる）、「非収束だが
        // cov_params計算は成功する」ケースを踏む。
        let input = censored_regression_input();
        let estimator = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 3,
                tol: 1e-12,
                raise_on_non_convergence: false,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        assert!(!estimator.converged());
    }

    /// 打ち切りが重い（60観測中51観測＝85%が0で左打ち切り）合成データ。外部の乱数に依存しない
    /// 線形合同法で作る（`tests/support/_nonlinear_helpers.py`の`heavily_censored_tobit_dataset`と同じ手順・
    /// 同じ値。Python側の同名テストと同じ入力で挙動を固定する）。潜在変数は
    /// `y* = -1.8 + x1 + 0.5·x2 + e`（`x1`・`x2`・`e`は12個の一様乱数の和−6で近似した正規乱数）。
    fn heavily_censored_input() -> TobitInput {
        let mut state: u64 = 167;
        let mut unit = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut normal = move || (0..12).map(|_| unit()).sum::<f64>() - 6.0;
        let n = 60;
        // 引数の評価順（`x1`→`x2`→`e`）がPython側と同じになるよう、1つのクロージャを
        // 3回続けて回す。
        let x1: Vec<f64> = (0..n).map(|_| normal()).collect();
        let x2: Vec<f64> = (0..n).map(|_| normal()).collect();
        let e: Vec<f64> = (0..n).map(|_| normal()).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (-1.8 + x1[i] + 0.5 * x2[i] + e[i]).max(0.0))
            .collect();
        assert_eq!(y.iter().filter(|&&v| v == 0.0).count(), 51);
        TobitInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap()
    }

    fn heavily_censored_groups() -> Vec<String> {
        (0..60).map(|i| (i % 4).to_string()).collect()
    }

    fn truncated_fit_options(solver: SolverType, cov_type: CovType) -> MleFitOptions {
        MleFitOptions {
            solver,
            max_iter: 1,
            tol: 1e-12,
            raise_on_non_convergence: false,
            cov_type,
            confidence_level: 0.95,
        }
    }

    fn all_cov_types_for_heavily_censored_input() -> Vec<CovType> {
        let groups = heavily_censored_groups();
        vec![
            CovType::Classical,
            CovType::Opg,
            CovType::Hc0,
            CovType::Hc1,
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
        ]
    }

    /// `max_iter=1`・`raise_on_non_convergence=false`で打ち切った点は、重く打ち切られたデータでは
    /// `(β, log σ)`尤度が凹でない領域の近くにある。`Newton`（LM減衰で凹な領域に留まる）・`Bfgs`・`Lbfgs`
    /// （初期点のHessianが正定値ならその厳密な逆行列から出発するため、1手目がNewtonステップと一致）は、
    /// 全`cov_type`で未収束の結果を返す（Hessianの逆行列を使う`Classical`/`Hc0`/`Hc1`/`Cluster`でも
    /// `SingularHessian`にならない）。かつて`Lbfgs`は初期点のHessianを使わず1手目が勾配方向だったため、
    /// このデータでは凹でない領域（費用関数Hessianの最小固有値が約-0.6）に着地し、Hessianの逆行列を使う
    /// `cov_type`が`SingularHessian`になっていた。
    #[test]
    fn fit_returns_unconverged_result_for_every_solver_and_cov_type_at_truncated_point() {
        for solver in [SolverType::Newton, SolverType::Bfgs, SolverType::Lbfgs] {
            for cov_type in all_cov_types_for_heavily_censored_input() {
                let result = TobitEstimator::fit(
                    heavily_censored_input(),
                    truncated_fit_options(solver, cov_type.clone()),
                );
                let estimator = result
                    .unwrap_or_else(|e| panic!("solver={solver:?}, cov_type={cov_type:?}: {e:?}"));
                assert!(
                    !estimator.converged(),
                    "solver={solver:?}, cov_type={cov_type:?}"
                );
            }
        }
    }

    /// OPG/HC0/HC1/クラスターSEの独立再計算検証・組み合わせテストで共通に使う多変量
    /// データセット（切片+説明変数2つ+`s`で`k+1=4`次元、実際の左打ち切りを含む）。
    /// `censored_regression_input`（説明変数1つ）を拡張したもの。情報行列の等式が
    /// 有限標本で厳密に成り立ってしまう切片のみ・単変量モデルでは`cov_type`の配線ミス
    /// （例えば`Opg`の分岐で誤って`observed_information_cov_params`を呼んでいた場合等）を
    /// 検出できないため、Logitの対応テスト（`fit_cov_type_opg_hc0_hc1_match_
    /// independently_recomputed_values`のdocコメント）と同じ理由で多変量データセットが
    /// 必須。
    fn multivariate_censored_input() -> TobitInput {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x2 = vec![1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0];
        let y = vec![0.0, 0.0, 1.15, 2.9, 5.2, 6.85, 9.1, 10.95];
        TobitInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap()
    }

    /// `fit()`と同じ手順（`TobitScaling`による`x`のセンタリング＋スケーリングと`y`の
    /// スケーリング→収束点でのscores/Hessian評価→`common.rs`の`cov_type`別の共通行列
    /// 演算→`TobitScaling::destandardize_cov_params`）をテスト側で独立に再現し、
    /// `cov_params_std_fn`が計算する標準化空間の分散共分散行列から最終的な`(β,σ)`空間の
    /// `cov_params`を得る。`Logit`の対応テストと同じ技法（`fit()`本体のロジックを
    /// 再利用せず、独立に再現することで配線ミスを検出する。`TobitScaling`の各メソッドは
    /// 元々の`standardize_columns`/`destandardize_cov_params`と同じく共有ヘルパー扱い）。
    fn expected_cov_params(
        input: &TobitInput,
        classical: &TobitEstimator,
        cov_params_std_fn: impl FnOnce(&Mat<f64>, &Mat<f64>) -> Mat<f64>,
    ) -> Mat<f64> {
        let k = input.k();
        let scaling = TobitScaling::fit(input.x(), input.y(), input.has_intercept());
        let x_std = scaling.standardize_x(input.x());
        let y_std = scaling.standardize_y(input.y());
        let lower_std = scaling.standardize_bound(input.lower());
        let upper_std = scaling.standardize_bound(input.upper());

        // 元のスケールの`(β, σ)`から標準化空間の内部パラメータ`(β̃, s̃=logσ̃)`を再構成する
        // （`destandardize_params`の逆写像。`TobitScaling`のdocコメントの式を反転）。
        let c = scaling.y_scale;
        // `β/c`を`ColumnScale::standardize_params`（`x`側の逆写像）で標準化空間へ戻す。
        let beta_over_c: Vec<f64> = classical.params()[..k].iter().map(|b| b / c).collect();
        let mut params_std = scaling.x.standardize_params(&beta_over_c);
        params_std.push(classical.sigma().ln() - c.ln());

        let problem_std = TobitProblem::from_standardized(x_std, y_std, lower_std, upper_std);
        let scores_std = problem_std.scores(&params_std);
        let cost_hessian_std = problem_std.hessian(&params_std).unwrap();
        let hessian_std = Mat::from_fn(k + 1, k + 1, |i, j| -cost_hessian_std[i][j]);

        scaling.destandardize_cov_params(
            &cov_params_std_fn(&hessian_std, &scores_std),
            classical.sigma(),
        )
    }

    fn assert_cov_params_close(actual: &Mat<f64>, expected: &Mat<f64>, k_plus_1: usize, tol: f64) {
        for i in 0..k_plus_1 {
            for j in 0..k_plus_1 {
                assert!(
                    (*actual.get(i, j) - *expected.get(i, j)).abs() < tol,
                    "({i},{j}): actual={}, expected={}",
                    *actual.get(i, j),
                    *expected.get(i, j)
                );
            }
        }
    }

    #[test]
    fn fit_cov_type_opg_hc0_hc1_match_independently_recomputed_values() {
        let k_plus_1 = 4;
        let n = 8;

        let classical = TobitEstimator::fit(
            multivariate_censored_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        let expected_opg =
            expected_cov_params(&multivariate_censored_input(), &classical, |_h, s| {
                opg_cov_params(s, k_plus_1).unwrap()
            });
        let expected_hc0 =
            expected_cov_params(&multivariate_censored_input(), &classical, |h, s| {
                sandwich_cov_params(h, s, n, k_plus_1, SandwichVariant::Hc0).unwrap()
            });
        let expected_hc1 =
            expected_cov_params(&multivariate_censored_input(), &classical, |h, s| {
                sandwich_cov_params(h, s, n, k_plus_1, SandwichVariant::Hc1).unwrap()
            });

        let cases = [
            (CovType::Opg, &expected_opg),
            (CovType::Hc0, &expected_hc0),
            (CovType::Hc1, &expected_hc1),
        ];
        for (cov_type, expected) in cases {
            let estimator = TobitEstimator::fit(
                multivariate_censored_input(),
                MleFitOptions {
                    solver: SolverType::Newton,
                    max_iter: 100,
                    tol: 1e-8,
                    raise_on_non_convergence: true,
                    cov_type,
                    confidence_level: 0.95,
                },
            )
            .unwrap();
            assert_cov_params_close(estimator.cov_params(), expected, k_plus_1, 1e-6);
        }
    }

    /// `solver`（`bfgs`/`lbfgs`）と`cov_type`（`Opg`/`Hc0`/`Hc1`）の組み合わせが正しく
    /// 機能することを確認する（Logitの`fit_non_classical_cov_types_work_with_bfgs_and_
    /// lbfgs`と同じ理由。`scores_std`の評価は収束点のパラメータにのみ依存し最適化
    /// アルゴリズムの種類に依存しない設計のため、`newton`で計算した`cov_params`
    /// （上のテストで既に正しさを検証済み）と`bfgs`/`lbfgs`の結果が一致するはず）。
    ///
    /// クラスターのグループ数は`G=4`（2件ずつ）にする。`multivariate_censored_input`は
    /// 傾き係数`q=2`（`x1`・`x2`、切片を除く）を持ち、Wald検定が`fit()`に
    /// 常時組み込まれたことで、クラスターロバスト共分散`Ŝ=Σ_g S_gS_g'`の構造的な制約
    /// （`rank(Ŝ)≤G`、`engine/src/linear/CLAUDE.md`「クラスター数`G`と傾き係数の数`q`の
    /// 関係」参照）がWald検定の`q×q`部分行列にも及ぶことが判明した。`G=2`（`q`と同数）
    /// では実測でこの部分行列が特異になり`fit()`全体が`ComputationFailed`になったため、
    /// `G>q`を満たす`G=4`に変更した（OLSの既存ガイドライン「境界の成功パスのテストでは
    /// `q`を`G`以下に保つ」をTobitでも踏襲）。
    #[test]
    fn fit_non_classical_cov_types_work_with_bfgs_and_lbfgs() {
        let k_plus_1 = 4;
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "c".to_string(),
            "c".to_string(),
            "d".to_string(),
            "d".to_string(),
        ];

        for cov_type in [
            CovType::Opg,
            CovType::Hc0,
            CovType::Hc1,
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
        ] {
            let newton = TobitEstimator::fit(
                multivariate_censored_input(),
                MleFitOptions {
                    solver: SolverType::Newton,
                    max_iter: 100,
                    tol: 1e-8,
                    raise_on_non_convergence: true,
                    cov_type: cov_type.clone(),
                    confidence_level: 0.95,
                },
            )
            .unwrap();

            for solver in [SolverType::Bfgs, SolverType::Lbfgs] {
                let estimator = TobitEstimator::fit(
                    multivariate_censored_input(),
                    MleFitOptions {
                        solver,
                        max_iter: 300,
                        tol: 1e-8,
                        raise_on_non_convergence: true,
                        cov_type: cov_type.clone(),
                        confidence_level: 0.95,
                    },
                )
                .unwrap();
                assert!(estimator.converged(), "cov_type={cov_type:?}, {solver:?}");
                assert_cov_params_close(
                    estimator.cov_params(),
                    newton.cov_params(),
                    k_plus_1,
                    1e-4,
                );
            }
        }
    }

    /// `multivariate_censored_input`ではなく`censored_regression_input`（傾き係数
    /// `q=1`、`x1`のみ）を使う。Wald検定が`fit()`に常時組み込まれたことで、
    /// クラスターロバスト共分散`Ŝ=Σ_g S_gS_g'`の構造的な制約（`rank(Ŝ)≤G`、
    /// `engine/src/linear/CLAUDE.md`「クラスター数`G`と傾き係数の数`q`の関係」参照）が
    /// Wald検定の`q×q`部分行列にも及ぶことが判明した。`multivariate_censored_input`
    /// （`q=2`）に対し`G=2`（G=2の境界値）を組み合わせると`q`と
    /// 同数になり、実測でこの部分行列が特異になり`fit()`全体が`ComputationFailed`に
    /// なった。`q=1`のデータセットなら`G=2>q=1`を満たしたまま「G=2の境界値」を検証できる
    /// ため、こちらに切り替えた（OLSの既存ガイドライン「境界の成功パスのテストでは`q`を
    /// `G`以下に保つ」をTobitでも踏襲）。
    #[test]
    fn fit_cov_type_cluster_matches_independently_recomputed_values() {
        let k_plus_1 = 3;
        let n = 8;
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];

        let classical = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        let groups_for_expected = groups.clone();
        let expected_cluster =
            expected_cov_params(&censored_regression_input(), &classical, |h, s| {
                cluster_cov_params(
                    h,
                    s,
                    n,
                    k_plus_1,
                    &crate::shared::group_codes::GroupCodes::from_labels(&groups_for_expected),
                )
                .unwrap()
            });

        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&groups)),
                },
                confidence_level: 0.95,
            },
        )
        .unwrap();
        assert_cov_params_close(estimator.cov_params(), &expected_cluster, k_plus_1, 1e-6);
    }

    /// 上のテストは4:4の均等サイズのグループのみを検証しているが、
    /// `testing-policy.md`が指摘する通り均等サイズのみのテストは実務で起こりやすい
    /// 偏った分布のグループサイズを見逃しうる。5:3の不均衡なグループ（G=2の境界値を
    /// 含む）でも同じ独立再計算の
    /// 技法で検証する（`fit_cov_type_cluster_matches_independently_recomputed_values`と
    /// 同じデータセット・同じ理由でq=1のデータセットを使う、グループ分割のみ変更）。
    #[test]
    fn fit_cov_type_cluster_matches_independently_recomputed_values_with_unbalanced_groups() {
        let k_plus_1 = 3;
        let n = 8;
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];

        let classical = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        let groups_for_expected = groups.clone();
        let expected_cluster =
            expected_cov_params(&censored_regression_input(), &classical, |h, s| {
                cluster_cov_params(
                    h,
                    s,
                    n,
                    k_plus_1,
                    &crate::shared::group_codes::GroupCodes::from_labels(&groups_for_expected),
                )
                .unwrap()
            });

        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&groups)),
                },
                confidence_level: 0.95,
            },
        )
        .unwrap();
        assert_cov_params_close(estimator.cov_params(), &expected_cluster, k_plus_1, 1e-6);
    }

    #[test]
    fn fit_returns_missing_cluster_column_error_when_groups_not_provided() {
        let result = TobitEstimator::fit(
            multivariate_censored_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Cluster { groups: None },
                confidence_level: 0.95,
            },
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::MissingClusterColumn {
                setting: ClusterSetting::CovType,
            })
        );
    }

    #[test]
    fn fit_returns_insufficient_clusters_error_when_only_one_group() {
        let groups = vec!["a".to_string(); 8];
        let result = TobitEstimator::fit(
            multivariate_censored_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&groups)),
                },
                confidence_level: 0.95,
            },
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InsufficientClusters {
                setting: ClusterSetting::CovType,
                g: 1
            })
        );
    }

    /// Logit（`fit_returns_singular_opg_matrix_error_for_perfectly_collinear_
    /// design_matrix`）はxの完全な多重共線性（x2=2*x1）でOPG行列`Σᵢsᵢsᵢ'`を特異にして
    /// いるが、Tobitでは`ols_initial_params`が最適化前にxのQRベースの階数検定を行うため
    /// （`fit_returns_singular_design_matrix_error_for_perfectly_collinear_data`参照）、
    /// xが完全に多重共線だと最適化に入る前に`SingularDesignMatrix`になり同じ手法は使えない。
    ///
    /// このテストのデータ（x1・x2は互いに無関係、多重共線性なし）は実測で見つけた
    /// 別の退化パターンで、`SingularOpgMatrix`の再現に多重共線性は必須ではないことが
    /// 分かっている（レビュー時の独立検証で、x2をx1と無関係な値に変えても収束点での
    /// OPG行列の最小固有値が機械精度オーダーまで落ちる同じ現象を確認済み）。原因は
    /// 観測数`n=6`が総パラメータ数`k+1=4`に対して少なく（`validate_sufficient_
    /// observations`の下限ぎりぎり）、かつ左打ち切り観測が2件という小標本特有の
    /// 収束点の数値配置に依存すると推測されるが、厳密な理論的必然性までは未解明
    /// （`n=7`など観測数を増やすと再現しなくなることを確認済み）。「なぜこの具体的な
    /// データで起こるか」を将来の保守者が誤って多重共線性のせいだと読み違えないよう、
    /// ここに正直に記録する。
    #[test]
    fn fit_returns_singular_opg_matrix_error_for_small_sample_degenerate_case() {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![0.0, 0.0, 1.15, 2.9, 5.2, 6.85];
        let x2: Vec<f64> = vec![5.0, -3.0, 7.0, 1.0, -2.0, 9.0];
        let input = TobitInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap();

        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Opg,
                confidence_level: 0.95,
            },
        );
        assert!(
            matches!(result, Err(MleError::SingularOpgMatrix)),
            "{result:?}"
        );
    }

    /// `censored_regression_input`（左打ち切りのみ、`lower=Some(0.0)`）の右打ち切り版。
    /// 潜在変数`y* = -5 + 2x + noise`は同じだが、`upper=Some(6.0)`で上側を打ち切る
    /// （x=6,7,8の潜在値6.85・9.1・10.95が6.0に打ち切られる、打ち切り率37.5%）。
    /// `marginal_effects`が右打ち切りのみのケースでも正しく計算できることを検証する
    /// データセット（`target_w_and_s`のdocコメント「数式」参照、ユーザー確認済み）。
    fn right_censored_regression_input() -> TobitInput {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![-2.85, -0.95, 1.15, 2.9, 5.2, 6.0, 6.0, 6.0];
        TobitInput::from_columns(
            &y,
            &[x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            None,
            Some(6.0),
        )
        .unwrap()
    }

    /// `censored_regression_input`の両側打ち切り版。同じ潜在変数`y* = -5 + 2x + noise`に
    /// `lower=Some(0.0)`・`upper=Some(6.0)`の両方を適用する（x=1,2は下側、x=6,7,8は上側で
    /// 打ち切り、非打ち切りはx=3,4,5のみ）。
    fn two_sided_censored_regression_input() -> TobitInput {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![0.0, 0.0, 1.15, 2.9, 5.2, 6.0, 6.0, 6.0];
        TobitInput::from_columns(
            &y,
            &[x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            Some(6.0),
        )
        .unwrap()
    }

    /// `E[y|x]`の閉形式（McDonald-Moffitt 1980）。左打ち切りのみ・右打ち切りのみ・両側
    /// 打ち切りをそれぞれ独立した分岐で書き下す（`target_w_and_s`の統一的な
    /// `boundary_terms`実装とは別のコードパスで検算するため、`(lower,upper)`ごとの
    /// 教科書通りの式をそのまま使う）。
    fn expected_observed_closed_form(
        mu: f64,
        sigma: f64,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        match (lower, upper) {
            (Some(c), None) => {
                let z = (mu - c) / sigma;
                normal.cdf(z) * mu + sigma * normal.pdf(z) + (1.0 - normal.cdf(z)) * c
            }
            (None, Some(c)) => {
                let z = (c - mu) / sigma;
                normal.cdf(z) * mu - sigma * normal.pdf(z) + (1.0 - normal.cdf(z)) * c
            }
            (Some(c1), Some(c2)) => {
                let za = (c1 - mu) / sigma;
                let zb = (c2 - mu) / sigma;
                normal.cdf(za) * c1
                    + (1.0 - normal.cdf(zb)) * c2
                    + (normal.cdf(zb) - normal.cdf(za)) * mu
                    - sigma * (normal.pdf(zb) - normal.pdf(za))
            }
            (None, None) => unreachable!("TobitInput::from_columns rejects both-None bounds"),
        }
    }

    /// `P(uncensored|x)`の閉形式。`expected_observed_closed_form`と同じ理由で
    /// `(lower,upper)`ごとに独立した分岐で書く。
    fn prob_uncensored_closed_form(
        mu: f64,
        sigma: f64,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        match (lower, upper) {
            (Some(c), None) => normal.cdf((mu - c) / sigma),
            (None, Some(c)) => normal.cdf((c - mu) / sigma),
            (Some(c1), Some(c2)) => normal.cdf((c2 - mu) / sigma) - normal.cdf((c1 - mu) / sigma),
            (None, None) => unreachable!("TobitInput::from_columns rejects both-None bounds"),
        }
    }

    /// `∂P(uncensored|x)/∂μ`の閉形式（`prob_uncensored_closed_form`を`mu`について解析的に
    /// 微分した式。`target_w_and_s`の`ProbUncensored`分岐とは独立に書き下したコード）。
    fn dprob_uncensored_dmu_closed_form(
        mu: f64,
        sigma: f64,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        match (lower, upper) {
            (Some(c), None) => normal.pdf((c - mu) / sigma) / sigma,
            (None, Some(c)) => -normal.pdf((c - mu) / sigma) / sigma,
            (Some(c1), Some(c2)) => {
                (normal.pdf((c1 - mu) / sigma) - normal.pdf((c2 - mu) / sigma)) / sigma
            }
            (None, None) => unreachable!("TobitInput::from_columns rejects both-None bounds"),
        }
    }

    /// `E[y|lower<y<upper,x]`の閉形式。`(lower,upper)`ごとに独立した分岐で、`φ/W`を
    /// 素直に直接計算する教科書通りの式（`1-Φ`や`Φ(zb)-Φ(za)`をそのまま使うため、裾では
    /// 桁落ちする。片側は生存関数`Φ(-z)`を使うので裾でも正確だが、両側は中程度の`z`
    /// での検算専用。いずれも`truncation_pdf_ratios`とは別のコードパス）。
    fn expected_uncensored_closed_form(
        mu: f64,
        sigma: f64,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        match (lower, upper) {
            (Some(c), None) => {
                let z = (c - mu) / sigma;
                mu + sigma * normal.pdf(z) / normal.cdf(-z)
            }
            (None, Some(c)) => {
                let z = (c - mu) / sigma;
                mu - sigma * normal.pdf(z) / normal.cdf(z)
            }
            (Some(c1), Some(c2)) => {
                let za = (c1 - mu) / sigma;
                let zb = (c2 - mu) / sigma;
                mu + sigma * (normal.pdf(za) - normal.pdf(zb)) / (normal.cdf(zb) - normal.cdf(za))
            }
            (None, None) => unreachable!("TobitInput::from_columns rejects both-None bounds"),
        }
    }

    /// ノイズの大きい決定論的な打ち切りデータ（`y*=-2+0.7x+ε`、`ε`は`±2.5`の一様、
    /// `x=0.25·i`、`i=0..60`）に`lower`/`upper`を適用した入力。既存の少数観測データは
    /// 当てはめ後の`σ`が小さく`z`が裾に入るため、素直な閉形式（`expected_uncensored_closed_form`）
    /// で独立に検算できる範囲に`z`を収める用途。
    fn noisy_censored_input(lower: Option<f64>, upper: Option<f64>) -> TobitInput {
        let x: Vec<f64> = (0..60).map(|i| 0.25 * f64::from(i)).collect();
        let y: Vec<f64> = (0..60u32)
            .map(|i| {
                let u = f64::from((i * 7919) % 1009) / 1009.0;
                let latent = -2.0 + 0.7 * x[i as usize] + (u - 0.5) * 5.0;
                let latent = lower.map_or(latent, |l| latent.max(l));
                upper.map_or(latent, |u| latent.min(u))
            })
            .collect();
        TobitInput::from_columns(
            &y,
            &[x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            lower,
            upper,
        )
        .unwrap()
    }

    /// `(lower,upper)`の代表的な3構成（左のみ・右のみ・両側）。
    fn bounds_configs() -> [(Option<f64>, Option<f64>); 3] {
        [
            (Some(0.0), None),
            (None, Some(1.0)),
            (Some(-1.0), Some(1.5)),
        ]
    }

    fn assert_relative_eq(actual: f64, expected: f64, rtol: f64, label: &str) {
        let scale = expected.abs().max(f64::MIN_POSITIVE);
        assert!(
            (actual - expected).abs() <= rtol * scale,
            "{label}: actual={actual:e}, expected={expected:e}, rel={:e}",
            (actual - expected).abs() / scale
        );
    }

    /// 逆ミルズ比`λ(a)=φ(a)/Φ(-a)`とその超過分`λ(a)-a`の参照値。`a<=1`は
    /// `scipy.special.erfcx`（`λ=√(2/π)/erfcx(a/√2)`）、`a>=3`は連分数を60桁の10進演算で
    /// 20000段評価した値（超過分`λ-a`も同じ演算で直接作った値）。`a∈[2.5,3)`の参照は
    /// 70桁の10進演算のテイラー級数（`scipy`の`erfcx`と一致を確認済み）。直接計算
    /// （`a<MILLS_CONTINUED_FRACTION_THRESHOLD=3`、テイラー級数の`Φ`）と連分数の両方の領域、
    /// および切り替え点の前後（`2.9999`・`3.0`）を含む。
    #[test]
    fn inverse_mills_ratio_and_gap_match_high_precision_reference_on_both_branches() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        // (a, lambda, gap)。`a<3`のgapは`λ-a`の引き算で、`λ/gap≈12`倍に誤差が増幅される
        let cases = [
            (-5.0, 1.486_719_940_904_908_6e-6, None),
            (0.0, 0.797_884_560_802_865_4, None),
            (1.0, 1.525_135_276_160_981_3, None),
            (2.5, 2.822_744_797_663_907, Some(0.322_744_797_663_907_2)),
            (2.9, 3.190_315_139_542_982, Some(0.290_315_139_542_982)),
            (2.99, 3.273_805_823_492_689, Some(0.283_805_823_492_688_8)),
            (2.9999, 3.283_005_711_006_472, Some(0.283_105_711_006_471_4)),
            (3.0, 3.283_098_654_930_436, Some(0.283_098_654_930_436_5)),
            (5.0, 5.186_503_967_125_842, Some(0.186_503_967_125_842_1)),
            (10.0, 10.098_093_233_962_512, Some(0.098_093_233_962_511_96)),
            (29.9, 29.933_370_411_296_167, None),
            (38.0, 38.026_279_466_575_87, Some(0.026_279_466_575_868_99)),
            (
                100.0,
                100.009_998_000_999_26,
                Some(0.009_998_000_999_260_705),
            ),
            (1000.0, 1_000.000_999_998, Some(0.000_999_998_000_01)),
        ];
        for (a, lambda, gap) in cases {
            let (actual, actual_gap) = inverse_mills_ratio_and_gap(&normal, a);
            // 級数側（`a<3`）は`a→3`で`Φ(-a)=1/2-(Φ(a)-1/2)`の桁落ちが増え、`λ`は最大`4e-13`、
            // `gap`は最大`4e-12`（実測）。連分数側（`a>=3`）は`5e-16`以下
            let (lambda_rtol, gap_rtol) = if a < MILLS_CONTINUED_FRACTION_THRESHOLD {
                (1e-12, 1e-11)
            } else {
                (1e-14, 1e-13)
            };
            assert_relative_eq(actual, lambda, lambda_rtol, &format!("lambda({a})"));
            if let Some(gap) = gap {
                assert_relative_eq(actual_gap, gap, gap_rtol, &format!("gap({a})"));
            }
        }
    }

    /// 切り替え点`MILLS_CONTINUED_FRACTION_THRESHOLD`の前後で`λ`・`Φ(x)-1/2`・`Φ(-a)`が
    /// 連続（飛びがない）。直接計算側（テイラー級数）と連分数側が同じ値を返すことの確認。
    #[test]
    fn mills_ratio_and_normal_cdf_are_continuous_across_the_series_to_continued_fraction_switch() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let t = MILLS_CONTINUED_FRACTION_THRESHOLD;
        // 切り替え点の直前の浮動小数点数（傾きによる変化を除き、飛びだけを見る）
        let below = f64::from_bits(t.to_bits() - 1);
        let (lambda_below, _) = inverse_mills_ratio_and_gap(&normal, below);
        let (lambda_at, _) = inverse_mills_ratio_and_gap(&normal, t);
        // 飛びの実測は`λ`で`1.3e-13`、`Φ(-a)`で`3.6e-13`（級数側の桁落ち）。許容は`1e-12`
        assert!(
            (lambda_at - lambda_below).abs() < 1e-12,
            "lambda: below={lambda_below}, at={lambda_at}"
        );
        assert_relative_eq(
            normal_survival(&normal, below),
            normal_survival(&normal, t),
            1e-12,
            "survival around switch",
        );
        assert_relative_eq(
            normal_cdf_minus_half(&normal, below),
            normal_cdf_minus_half(&normal, t),
            1e-12,
            "cdf-1/2 around switch",
        );
    }

    /// `statrs`の`Normal::cdf`（絶対誤差約`1.4e-11`）に代わる`normal_cdf_minus_half`・
    /// `normal_survival`の精度。参照は`scipy.special`（`0.5·erf(x/√2)`・`ndtr(-a)`）。
    /// `Φ(-8)≈6.2e-16`のような裾でも相対`1e-13`、`1/2`近傍も絶対`1e-16`級で一致する。
    #[test]
    fn normal_cdf_minus_half_and_survival_match_scipy_to_machine_precision() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        for (x, expected) in [
            (0.5, 0.191_462_461_274_013_06),
            (1.0, 0.341_344_746_068_542_9),
            (2.0, 0.477_249_868_051_820_8),
            (2.99, 0.498_605_112_764_507_75),
            (3.0, 0.498_650_101_968_369_9),
            (5.0, 0.499_999_713_348_428_13),
        ] {
            assert_relative_eq(
                normal_cdf_minus_half(&normal, x),
                expected,
                1e-14,
                &format!("Phi({x})-1/2"),
            );
        }
        for (a, expected) in [
            (-2.0, 0.977_249_868_051_820_8),
            (0.0, 0.5),
            (1.0, 0.158_655_253_931_457_07),
            (2.0, 0.022_750_131_948_179_195),
            (2.99, 0.001_394_887_235_492_246_8),
            (3.0, 0.001_349_898_031_630_093_3),
            (4.0, 3.167_124_183_311_986e-5),
            (8.0, 6.220_960_574_271_74e-16),
        ] {
            assert_relative_eq(
                normal_survival(&normal, a),
                expected,
                1e-13,
                &format!("Phi(-{a})"),
            );
        }
    }

    /// 連分数展開は、切り替え点より上（`a=3`・`3.5`・`4`）でも、テイラー級数
    /// （`normal_cdf_minus_half_series`、切り替え点より上へ外挿して`Φ(-a)=1/2-(Φ(a)-1/2)`を作る）
    /// 由来の`φ(a)/Φ(-a)`と一致する。段数`MILLS_CONTINUED_FRACTION_DEPTH`が切り替え点でも
    /// 収束していることの確認で、段数を落とす（20等）と失敗する。級数側は`a`が大きいほど
    /// `Φ(-a)=1/2-(Φ(a)-1/2)`の桁落ち（相対`ε/Φ(-a)`）で精度が落ちるため、許容は`a`に応じて
    /// `Φ(-3)=1.3e-3`で`5e-13`、`Φ(-3.5)=2.3e-4`で`2e-12`、`Φ(-4)=3.2e-5`で`1e-11`（実測`4e-12`）。
    #[test]
    fn inverse_mills_ratio_continued_fraction_agrees_with_series_based_survival_near_the_switch() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        for (a, rtol) in [(3.0, 5e-13), (3.5, 2e-12), (4.0, 1e-11)] {
            let series_survival = 0.5 - normal_cdf_minus_half_series(&normal, a);
            let series_lambda = normal.pdf(a) / series_survival;
            let (cf, _) = inverse_mills_ratio_continued_fraction(a, MILLS_CONTINUED_FRACTION_DEPTH);
            assert_relative_eq(cf, series_lambda, rtol, &format!("cf vs series at a={a}"));
        }
    }

    /// 補助関数は`NaN`・`±inf`・`±40`でもpanicや無限ループなしに有限時間で戻り、値は
    /// `Φ(-a)`の定義（`a=-inf`で1、`a=+inf`で0、`NaN`は`NaN`）に従う。
    #[test]
    fn normal_cdf_helpers_handle_non_finite_and_far_tail_inputs_without_panicking() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        assert!(normal_survival(&normal, f64::NAN).is_nan());
        assert!(normal_cdf_minus_half(&normal, f64::NAN).is_nan());
        assert_eq!(normal_survival(&normal, f64::INFINITY), 0.0);
        assert_eq!(normal_survival(&normal, f64::NEG_INFINITY), 1.0);
        assert_eq!(normal_survival(&normal, -40.0), 1.0);
        assert!(normal_survival(&normal, 40.0) >= 0.0);
        assert_eq!(normal_cdf_minus_half(&normal, 40.0), 0.5);
        // 遠い裾の逆ミルズ比は有限（`φ`を作らない連分数）
        for a in [-1e8, -40.0, 40.0, 1e8] {
            let (lambda, gap) = inverse_mills_ratio_and_gap(&normal, a);
            assert!(lambda.is_finite() && gap.is_finite(), "a={a}");
        }
    }

    /// `(Pa,Pb)`と`ExpectedUncensored`の`(w, ∂w/∂μ, ∂w/∂σ)`を、70〜80桁の10進演算で
    /// 直接評価した参照値（定義通りに`φ/W`・`w=1+ρ2-ρ1²`・偏微分を計算。裾は`W=Φ(-za)-Φ(-zb)`を
    /// 連分数の`Φ(-z)`から、中央・級数側は`Φ(x)-1/2`のテイラー級数から作る）と突き合わせる。
    /// 左のみ・右のみ・両側の上側の裾・下側の裾（符号反転）・`zb`が遥か先で片側と一致する場合に
    /// 加え、切り替え点の前後の両側`(2.9,3.1)`・`(-3.2,-2.9)`・`(2.99,3.5)`、中央`(-1,1.5)`、
    /// 級数側の上側`(0.5,1)`を含む。`z=40`では`Φ(-40)≈3.7e-350`が倍精度でアンダーフローし、
    /// 直接計算では`W=0`になる領域。許容は実測（`w`の絶対誤差`2e-16`、`Pa`/`Pb`の相対`1e-16`、
    /// `s_beta`は`z=40`で`3e-15`、`s_sigma`は`z=40`で`1e-13`・`z=1000`で`9e-12`）の約10倍。
    /// 整理前の式（`ε·z³`の桁落ち）は`z=40`以上で許容を超える。
    #[test]
    fn uncensored_terms_and_w_match_high_precision_reference_in_extreme_tails() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        // (label, za, zb, Pa, Pb, w, w_a, w_b)
        #[allow(clippy::type_complexity)]
        let cases: [(&str, Option<f64>, Option<f64>, f64, f64, f64, f64, f64); 15] = [
            (
                "left-only z=40",
                Some(40.0),
                None,
                40.024_968_847_207_26,
                0.0,
                6.226_683_785_913_888e-4,
                -3.101_744_039_648_625e-5,
                0.0,
            ),
            (
                "left-only z=100",
                Some(100.0),
                None,
                100.009_998_000_999_26,
                0.0,
                9.994_004_994_826_346e-5,
                -1.997_602_995_862_343e-6,
                0.0,
            ),
            (
                "left-only z=1000",
                Some(1000.0),
                None,
                1_000.000_999_998,
                0.0,
                9.999_940_000_499_995e-7,
                -1.999_976_000_299_996e-9,
                0.0,
            ),
            (
                "right-only z=-40",
                None,
                Some(-40.0),
                0.0,
                40.024_968_847_207_26,
                6.226_683_785_913_888e-4,
                0.0,
                3.101_744_039_648_625e-5,
            ),
            (
                "two-sided upper (40,45)",
                Some(40.0),
                Some(45.0),
                40.024_968_847_207_26,
                2.064_208_671_428_423_9e-91,
                6.226_683_785_913_888e-4,
                -3.101_744_039_648_625e-5,
                5.108_980_927_616_573e-90,
            ),
            (
                "two-sided upper (30,31)",
                Some(30.0),
                Some(31.0),
                30.033_259_667_435_328,
                1.704_593_283_822_866_3e-12,
                1.103_771_511_835_282_2e-3,
                -7.309_993_169_431_127e-5,
                1.591_209_021_285_825_6e-12,
            ),
            (
                "two-sided upper (40,1000)",
                Some(40.0),
                Some(1000.0),
                40.024_968_847_207_26,
                0.0,
                6.226_683_785_913_888e-4,
                -3.101_744_039_648_625e-5,
                0.0,
            ),
            (
                "two-sided lower (-45,-40)",
                Some(-45.0),
                Some(-40.0),
                2.064_208_671_428_423_9e-91,
                40.024_968_847_207_26,
                6.226_683_785_913_888e-4,
                -5.108_980_927_616_573e-90,
                3.101_744_039_648_625e-5,
            ),
            (
                "two-sided lower (-31,-30)",
                Some(-31.0),
                Some(-30.0),
                1.704_593_283_822_866_3e-12,
                30.033_259_667_435_328,
                1.103_771_511_835_282_2e-3,
                -1.591_209_021_285_825_6e-12,
                7.309_993_169_431_127e-5,
            ),
            (
                "two-sided upper (12,12.5)",
                Some(12.0),
                Some(12.5),
                12.107_653_550_680_133,
                0.026_485_384_604_148_263,
                6.151_043_601_970_409e-3,
                -5.293_800_087_694_461_5e-3,
                4.483_156_195_574_498e-3,
            ),
            (
                "two-sided middle (-1,1.5)",
                Some(-1.0),
                Some(1.5),
                0.312_406_707_904_292_9,
                0.167_219_260_751_766_74,
                0.415_685_006_157_389_35,
                -0.279_844_332_730_801_1,
                0.237_423_265_323_732_5,
            ),
            (
                "two-sided series side (0.5,1)",
                Some(0.5),
                Some(1.0),
                2.348_945_555_820_274,
                1.614_405_096_978_975_4,
                0.020_517_995_256_376_27,
                -0.081_018_025_127_821_73,
                0.080_640_782_102_733_44,
            ),
            (
                "two-sided across switch (2.9,3.1)",
                Some(2.9),
                Some(3.1),
                6.627_104_844_212_096,
                3.637_032_252_118_688_6,
                0.003_269_960_658_955_57,
                -0.032_095_805_611_538_79,
                0.032_057_072_683_910_62,
            ),
            (
                "two-sided lower across switch (-3.2,-2.9)",
                Some(-3.2),
                Some(-2.9),
                2.022_684_343_271_988,
                5.050_188_210_696_992,
                0.007_176_245_277_561_196,
                -0.045_669_523_263_236_76,
                0.045_860_713_340_026_13,
            ),
            (
                "two-sided just above switch (2.99,3.5)",
                Some(2.99),
                Some(3.5),
                3.929_066_816_439_395_4,
                0.750_850_996_568_809_9,
                0.018_875_495_487_299_693,
                -0.065_024_874_586_292_15,
                0.063_574_225_828_370_7,
            ),
        ];
        // `mu=1, sigma=2`で`za=(lower-mu)/sigma`が上の値になるよう境界を逆算する
        let (mu, sigma) = (1.0, 2.0);
        for (label, za, zb, pa, pb, w, w_a, w_b) in cases {
            let lower = za.map(|z| mu + sigma * z);
            let upper = zb.map(|z| mu + sigma * z);
            let terms = UncensoredTerms::new(mu, sigma, lower, upper, &normal);
            // 切り替え点の手前（`a<3`）を通る級数側は`λ`の相対誤差が最大`4e-13`
            let near_switch = label.contains("switch");
            let ratio_rtol = if near_switch { 1e-12 } else { 1e-14 };
            assert_relative_eq(terms.ratios.pa, pa, ratio_rtol, &format!("{label}: Pa"));
            assert_relative_eq(terms.ratios.pb, pb, ratio_rtol, &format!("{label}: Pb"));

            let x_point = [1.0, 2.0];
            let (w_actual, s_beta, s_sigma) =
                uncensored_w_and_s(&x_point, mu, sigma, lower, upper, &normal);
            let z_scale = za.unwrap_or(0.0).abs().max(zb.unwrap_or(0.0).abs());
            // `w`は`1-Pa·A-Pb·B+2·Pa·Pb`（`O(1)`同士の和）なので絶対誤差は数`ε`。切り替え点付近の
            // 級数側は`Pa`・`Pb`・`A`・`B`の誤差（`λ`で最大`4e-13`、`gap`で最大`4e-12`）が`O(10)`倍で効く
            assert!(
                (w_actual - w).abs() <= if near_switch { 2e-12 } else { 2e-15 },
                "{label}: w actual={w_actual:e}, expected={w:e}"
            );
            let dw_dmu = -(w_a + w_b) / sigma;
            let dw_dsigma = -(za.unwrap_or(0.0) * w_a + zb.unwrap_or(0.0) * w_b) / sigma;
            assert!(
                (s_beta[1] - dw_dmu * x_point[1]).abs()
                    <= if near_switch {
                        5e-12
                    } else {
                        1e-15 * z_scale + 1e-15
                    },
                "{label}: s_beta actual={:e}, expected={:e}",
                s_beta[1],
                dw_dmu * x_point[1]
            );
            assert!(
                (s_sigma - dw_dsigma).abs()
                    <= if near_switch {
                        2e-11
                    } else {
                        1e-15 * z_scale * z_scale + 1e-15
                    },
                "{label}: s_sigma actual={s_sigma:e}, expected={dw_dsigma:e}"
            );
        }
    }

    /// 極端な入力（`z`が`1e8`、`w`が丸め誤差に支配される領域を含む）でも、`m`・`w`・`s`が
    /// 有限で、`w`は`[0,1]`に収まり`m`は非打ち切り領域`[lower,upper]`に収まる。
    #[test]
    fn expected_uncensored_stays_finite_and_bounded_for_extreme_inputs() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let sigma = 1.0;
        for (lower, upper) in bounds_configs() {
            for mu in [
                -1e8, -1e4, -100.0, -39.0, -8.0, 0.0, 0.5, 8.0, 39.0, 100.0, 1e4, 1e8,
            ] {
                let m = predicted_value(
                    MarginalEffectsTarget::ExpectedUncensored,
                    mu,
                    sigma,
                    lower,
                    upper,
                    &normal,
                );
                let (w, s_beta, s_sigma) = target_w_and_s(
                    MarginalEffectsTarget::ExpectedUncensored,
                    &[1.0],
                    &[mu],
                    sigma,
                    lower,
                    upper,
                    &normal,
                );
                let label = format!("lower={lower:?}, upper={upper:?}, mu={mu}");
                assert!(m.is_finite(), "{label}: m={m}");
                assert!(w.is_finite() && (0.0..=1.0).contains(&w), "{label}: w={w}");
                assert!(s_beta[0].is_finite(), "{label}: s_beta={}", s_beta[0]);
                assert!(s_sigma.is_finite(), "{label}: s_sigma={s_sigma}");
                // 非打ち切り領域への条件付き期待値は領域内（境界は丸め誤差分だけ許す）
                let eps = 1e-9 * (1.0 + m.abs());
                if let Some(l) = lower {
                    assert!(m >= l - eps, "{label}: m={m} < lower");
                }
                if let Some(u) = upper {
                    assert!(m <= u + eps, "{label}: m={m} > upper");
                }
            }
        }
    }

    /// 遠い裾では`E[y|uncensored]`は最寄りの境界に漸近する（`μ`が下限よりずっと下なら
    /// `m→lower`、上限よりずっと上なら`m→upper`）。素直な`φ/W`だと`W=0`で`NaN`になる領域。
    #[test]
    fn expected_uncensored_converges_to_nearest_bound_in_far_tail() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let m_left = predicted_value(
            MarginalEffectsTarget::ExpectedUncensored,
            -100.0,
            1.0,
            Some(0.0),
            None,
            &normal,
        );
        // 左のみ: m = μ+σλ(za), za=100 → λ=100.00999800099926 → m=0.00999800099926
        assert_relative_eq(m_left, 0.009_998_000_999_26, 1e-9, "left-only far tail");
        let m_right = predicted_value(
            MarginalEffectsTarget::ExpectedUncensored,
            100.0,
            1.0,
            None,
            Some(0.0),
            &normal,
        );
        assert_relative_eq(m_right, -0.009_998_000_999_26, 1e-9, "right-only far tail");
        let m_two_sided_above = predicted_value(
            MarginalEffectsTarget::ExpectedUncensored,
            1000.0,
            1.0,
            Some(-1.0),
            Some(2.0),
            &normal,
        );
        // 上限より遥か上（`zb=-998`）: m = μ - σ·λ(998) = 1000 - (998 + gap), gap = λ(998)-998 ≈ 1/(998+2/998)
        assert!(
            (m_two_sided_above - (2.0 - 1.0 / (998.0 + 2.0 / 998.0))).abs() < 1e-9,
            "m={m_two_sided_above}"
        );
    }

    /// 全領域（左のみ・右のみ・両側、`za=0`・`zb=0`の領域境界を含む）で、
    /// `predicted_value`が独立の閉形式`expected_uncensored_closed_form`と一致する。
    #[test]
    fn predicted_value_expected_uncensored_matches_independent_closed_form() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        for (lower, upper) in bounds_configs() {
            for sigma in [0.8, 1.0, 3.0] {
                for mu in [-3.5, -1.5, -1.0, 0.0, 0.3, 1.0, 1.5, 2.5, 4.5] {
                    let actual = predicted_value(
                        MarginalEffectsTarget::ExpectedUncensored,
                        mu,
                        sigma,
                        lower,
                        upper,
                        &normal,
                    );
                    let expected = expected_uncensored_closed_form(mu, sigma, lower, upper);
                    assert!(
                        (actual - expected).abs() < 1e-9 * (1.0 + expected.abs()),
                        "lower={lower:?}, upper={upper:?}, sigma={sigma}, mu={mu}: \
                         actual={actual}, expected={expected}"
                    );
                }
            }
        }
    }

    /// 分解恒等式 `E[y|x]=lower·P(下限)+upper·P(上限)+P(uncensored|x)·E[y|uncensored,x]`
    /// （下限0のみの左打ち切りなら`E[y|x]=P(y>0|x)·E[y|y>0,x]`）。既存の
    /// `ExpectedObserved`・`ProbUncensored`と新`ExpectedUncensored`が互いに整合することの確認。
    #[test]
    fn expected_observed_equals_censoring_mass_plus_prob_uncensored_times_expected_uncensored() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        for (lower, upper) in bounds_configs() {
            for sigma in [0.8, 1.0, 3.0] {
                for mu in [-3.5, -1.0, 0.0, 0.7, 1.5, 2.5, 4.5] {
                    let predict =
                        |target| predicted_value(target, mu, sigma, lower, upper, &normal);
                    let p_lower = lower.map_or(0.0, |l| normal.cdf((l - mu) / sigma));
                    let p_upper = upper.map_or(0.0, |u| 1.0 - normal.cdf((u - mu) / sigma));
                    let decomposed = lower.unwrap_or(0.0) * p_lower
                        + upper.unwrap_or(0.0) * p_upper
                        + predict(MarginalEffectsTarget::ProbUncensored)
                            * predict(MarginalEffectsTarget::ExpectedUncensored);
                    let observed = predict(MarginalEffectsTarget::ExpectedObserved);
                    assert!(
                        (observed - decomposed).abs() < 1e-9 * (1.0 + observed.abs()),
                        "lower={lower:?}, upper={upper:?}, sigma={sigma}, mu={mu}: \
                         observed={observed}, decomposed={decomposed}"
                    );
                }
            }
        }
    }

    /// `w=∂m/∂μ`・`s_beta=∂w/∂μ·x`・`s_sigma=∂w/∂σ`を、独立の閉形式`m`の中心差分
    /// （`w`は`m`の`μ`微分、`s`は`w`の微分なので`m`の2階・3階の差分）と突き合わせる。
    /// 3構成 × `μ`・`σ`の格子で、領域の切り替え（`za>=0`・`zb<=0`・中央）をまたぐ。
    /// `w`の比較に使うオラクル`expected_uncensored_closed_form`は`statrs`の`cdf`（絶対誤差
    /// 約`1.4e-11`）を使うため、中心差分（`h=1e-4`）経由で約`1e-7`・`h`で割る分`1e-9`級の誤差を
    /// 含む（実測の最大差`1.1e-9`）。許容`1e-8`はその約10倍。`s`の比較は本番の`w`の差分なので
    /// 中心差分の打ち切り誤差`h²·|w'''|/6`だけ（実測の最大`1.4e-8`）で、許容は`1e-7`。
    #[test]
    fn target_w_and_s_expected_uncensored_matches_finite_differences_of_closed_form_mean() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let w_of = |mu: f64, sigma: f64, lower, upper| {
            target_w_and_s(
                MarginalEffectsTarget::ExpectedUncensored,
                &[1.0],
                &[mu],
                sigma,
                lower,
                upper,
                &normal,
            )
            .0
        };
        for (lower, upper) in bounds_configs() {
            for sigma in [0.7, 1.0, 2.5] {
                for mu in [-3.0, -1.0, -0.4, 0.0, 0.4, 1.0, 1.5, 2.0, 4.0] {
                    let h = 1e-4;
                    let m = |mu: f64, sigma: f64| {
                        expected_uncensored_closed_form(mu, sigma, lower, upper)
                    };
                    let label = format!("lower={lower:?}, upper={upper:?}, sigma={sigma}, mu={mu}");

                    let (w, s_beta, s_sigma) = target_w_and_s(
                        MarginalEffectsTarget::ExpectedUncensored,
                        &[1.0],
                        &[mu],
                        sigma,
                        lower,
                        upper,
                        &normal,
                    );
                    let w_fd = (m(mu + h, sigma) - m(mu - h, sigma)) / (2.0 * h);
                    assert!((w - w_fd).abs() < 1e-8, "{label}: w={w}, fd={w_fd}");
                    assert!((0.0..=1.0).contains(&w), "{label}: w={w}");

                    let dw_dmu_fd = (w_of(mu + h, sigma, lower, upper)
                        - w_of(mu - h, sigma, lower, upper))
                        / (2.0 * h);
                    assert!(
                        (s_beta[0] - dw_dmu_fd).abs() < 1e-7,
                        "{label}: s_beta={}, fd={dw_dmu_fd}",
                        s_beta[0]
                    );
                    let dw_dsigma_fd = (w_of(mu, sigma + h, lower, upper)
                        - w_of(mu, sigma - h, lower, upper))
                        / (2.0 * h);
                    assert!(
                        (s_sigma - dw_dsigma_fd).abs() < 1e-7,
                        "{label}: s_sigma={s_sigma}, fd={dw_dsigma_fd}"
                    );
                }
            }
        }
    }

    /// `at="overall"`の限界効果（effect・デルタ法のSE）を、独立の閉形式`m`から作った
    /// AME関数の中心差分ヤコビアンで再計算した値と突き合わせる（左のみ・右のみ・両側）。
    #[test]
    fn fit_marginal_effects_expected_uncensored_overall_matches_independent_recomputation() {
        let inputs = [
            noisy_censored_input(Some(0.0), None),
            noisy_censored_input(None, Some(6.0)),
            noisy_censored_input(Some(0.0), Some(6.0)),
        ];
        for input in inputs {
            let estimator = TobitEstimator::fit(
                input,
                MleFitOptions {
                    solver: SolverType::Newton,
                    max_iter: 100,
                    tol: 1e-8,
                    raise_on_non_convergence: true,
                    cov_type: CovType::Classical,
                    confidence_level: 0.95,
                },
            )
            .unwrap();
            let lower = estimator.input().lower();
            let upper = estimator.input().upper();
            let x = estimator.input().x();
            let n = x.nrows();
            let k = 2;
            let label = format!("lower={lower:?}, upper={upper:?}");

            let effects = estimator
                .marginal_effects(
                    MarginalEffectsAt::Overall,
                    MarginalEffectsTarget::ExpectedUncensored,
                    0.95,
                )
                .unwrap();

            // AME_j = (1/n)Σ w_i·β_j, w_i=∂m/∂μ を m の中心差分で作る（k=2なのでj=1のみ）
            let ame = |params: &[f64], sigma: f64| -> f64 {
                let h = 1e-5;
                (0..n)
                    .map(|i| {
                        let mu: f64 = params[0] + params[1] * (*x.get(i, 1));
                        let m = |mu: f64| expected_uncensored_closed_form(mu, sigma, lower, upper);
                        (m(mu + h) - m(mu - h)) / (2.0 * h) * params[1]
                    })
                    .sum::<f64>()
                    / (n as f64)
            };
            let params = estimator.params();
            let sigma = estimator.sigma();
            assert!(
                (effects.effect()[0] - ame(params, sigma)).abs() < 1e-6,
                "{label}: actual={}, expected={}",
                effects.effect()[0],
                ame(params, sigma)
            );

            let h_param = 1e-3;
            let mut jac_row = vec![0.0; k + 1];
            for (m_idx, jac) in jac_row.iter_mut().take(k).enumerate() {
                let mut plus_p = params.to_vec();
                plus_p[m_idx] += h_param;
                let mut minus_p = params.to_vec();
                minus_p[m_idx] -= h_param;
                *jac = (ame(&plus_p, sigma) - ame(&minus_p, sigma)) / (2.0 * h_param);
            }
            jac_row[k] =
                (ame(params, sigma + h_param) - ame(params, sigma - h_param)) / (2.0 * h_param);
            let mut var = 0.0;
            for a in 0..=k {
                for b in 0..=k {
                    var += jac_row[a] * (*estimator.cov_params().get(a, b)) * jac_row[b];
                }
            }
            let expected_se = var.sqrt();
            assert!(
                (effects.std_errors()[0] - expected_se).abs() < 1e-5 * (1.0 + expected_se),
                "{label}: actual SE={}, expected SE={expected_se}",
                effects.std_errors()[0]
            );
        }
    }

    /// `predict`/`predict_new_data`が`ExpectedUncensored`で独立の閉形式と一致し、
    /// 学習データへの`predict`と同じ`x`での`predict_new_data`が一致する。
    #[test]
    fn predict_and_predict_new_data_expected_uncensored_match_closed_form() {
        let inputs = [
            noisy_censored_input(Some(0.0), None),
            noisy_censored_input(None, Some(6.0)),
            noisy_censored_input(Some(0.0), Some(6.0)),
        ];
        for input in inputs {
            let estimator = TobitEstimator::fit(
                input,
                MleFitOptions {
                    solver: SolverType::Newton,
                    max_iter: 100,
                    tol: 1e-8,
                    raise_on_non_convergence: true,
                    cov_type: CovType::Classical,
                    confidence_level: 0.95,
                },
            )
            .unwrap();
            let lower = estimator.input().lower();
            let upper = estimator.input().upper();
            let x = estimator.input().x();
            let params = estimator.params();
            let sigma = estimator.sigma();

            let predicted = estimator
                .predict(MarginalEffectsTarget::ExpectedUncensored)
                .unwrap();
            for (i, &p) in predicted.iter().enumerate() {
                let mu: f64 = params[0] + params[1] * (*x.get(i, 1));
                let expected = expected_uncensored_closed_form(mu, sigma, lower, upper);
                assert!(
                    (p - expected).abs() < 1e-9 * (1.0 + expected.abs()),
                    "lower={lower:?}, upper={upper:?}, i={i}: actual={p}, expected={expected}"
                );
            }

            let x1: Vec<f64> = (0..x.nrows()).map(|i| *x.get(i, 1)).collect();
            let predicted_new = estimator
                .predict_new_data(MarginalEffectsTarget::ExpectedUncensored, &[x1])
                .unwrap();
            assert_eq!(predicted_new.len(), predicted.len());
            for (a, b) in predicted_new.iter().zip(&predicted) {
                assert!((a - b).abs() < 1e-12, "predict_new_data={a}, predict={b}");
            }
        }
    }

    /// 非打ち切り区間の幅`(upper-lower)/σ`が`MIN_UNCENSORED_WIDTH_RATIO`未満のときだけ
    /// `UncensoredIntervalTooNarrow`を返す。境界ちょうど（`=1e-2`）は通し、片側打ち切り
    /// （`lower`か`upper`が`None`）は`σ`がどれほど大きくても通す。
    #[test]
    fn ensure_uncensored_interval_resolvable_rejects_only_narrow_two_sided_intervals() {
        // 幅/σ = 1/100 = 0.01（境界ちょうど）は通す
        assert!(ensure_uncensored_interval_resolvable(Some(0.0), Some(1.0), 100.0).is_ok());
        // 幅/σ = 1/100.5 < 0.01 は拒否し、境界とσを持つ
        match ensure_uncensored_interval_resolvable(Some(0.0), Some(1.0), 100.5) {
            Err(MleError::UncensoredIntervalTooNarrow {
                lower,
                upper,
                sigma,
            }) => {
                assert_eq!((lower, upper, sigma), (0.0, 1.0, 100.5));
            }
            other => panic!("expected UncensoredIntervalTooNarrow, got {other:?}"),
        }
        // 片側打ち切りは区間が無限大なので`σ`が巨大でも通す
        assert!(ensure_uncensored_interval_resolvable(Some(0.0), None, 1e12).is_ok());
        assert!(ensure_uncensored_interval_resolvable(None, Some(0.0), 1e12).is_ok());
        // 広い区間は通す
        assert!(ensure_uncensored_interval_resolvable(Some(-5.0), Some(5.0), 1.0).is_ok());
    }

    /// ほぼ全観測が両端に打ち切られた当てはめ（`σ̂`が区間幅の100倍超）。
    fn narrow_interval_estimator(n_interior: usize) -> TobitEstimator {
        let n = 2000usize;
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 7919) % 1009) as f64 / 1009.0)
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                if i < n_interior {
                    0.5
                } else if i % 2 == 0 {
                    0.0
                } else {
                    1.0
                }
            })
            .collect();
        let input = TobitInput::from_columns(
            &y,
            &[x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            Some(1.0),
        )
        .unwrap();
        TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 200,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap()
    }

    /// 狭い区間の当てはめ（`σ̂≈160`、`δ≈6e-3`）では`ExpectedUncensored`の`predict`・
    /// `predict_new_data`・`marginal_effects`が`UncensoredIntervalTooNarrow`になり、他の3targetは
    /// 通常どおり計算できる（エラーは`expected_uncensored`固有）。区間がちょうど十分な当てはめ
    /// （`σ̂≈40`、`δ≈0.025`）では`ExpectedUncensored`も計算できる。
    #[test]
    fn expected_uncensored_errors_only_when_the_fitted_interval_is_too_narrow() {
        let narrow = narrow_interval_estimator(5);
        assert!(
            1.0 / narrow.sigma() < MIN_UNCENSORED_WIDTH_RATIO,
            "fixture is not narrow"
        );
        let new_x = vec![vec![0.2, 0.8]];
        assert!(matches!(
            narrow.predict(MarginalEffectsTarget::ExpectedUncensored),
            Err(MleError::UncensoredIntervalTooNarrow { .. })
        ));
        assert!(matches!(
            narrow.predict_new_data(MarginalEffectsTarget::ExpectedUncensored, &new_x),
            Err(MleError::UncensoredIntervalTooNarrow { .. })
        ));
        assert!(matches!(
            narrow.marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ExpectedUncensored,
                0.95
            ),
            Err(MleError::UncensoredIntervalTooNarrow { .. })
        ));
        for target in [
            MarginalEffectsTarget::ExpectedLatent,
            MarginalEffectsTarget::ExpectedObserved,
            MarginalEffectsTarget::ProbUncensored,
        ] {
            let predicted = narrow.predict(target).unwrap();
            assert!(predicted.iter().all(|v| v.is_finite()), "{target:?}");
            assert!(
                narrow.predict_new_data(target, &new_x).is_ok(),
                "{target:?}"
            );
            assert!(
                narrow
                    .marginal_effects(MarginalEffectsAt::Overall, target, 0.95)
                    .is_ok(),
                "{target:?}"
            );
        }

        let wide_enough = narrow_interval_estimator(20);
        assert!(1.0 / wide_enough.sigma() >= MIN_UNCENSORED_WIDTH_RATIO);
        let predicted = wide_enough
            .predict(MarginalEffectsTarget::ExpectedUncensored)
            .unwrap();
        assert!(
            predicted
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
        );
    }

    /// 切片のみモデル（k=1）は限界効果の出力対象となる説明変数が存在しない
    /// （定数項は出力から除外するため）。`marginal_effects`が空の結果を返す
    /// （パニックしない）ことを確認する境界ケース（Logitの同名テストと同じ理由）。
    #[test]
    fn fit_marginal_effects_returns_empty_result_for_intercept_only_model() {
        let estimator = TobitEstimator::fit(
            intercept_only_uncensored_input(&[1.0, 2.0, 3.0, 4.0]),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        let effects = estimator
            .marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ExpectedObserved,
                0.95,
            )
            .unwrap();
        assert!(effects.param_names().is_empty());
        assert!(effects.effect().is_empty());
    }

    #[test]
    fn fit_marginal_effects_returns_invalid_confidence_level_error_out_of_range() {
        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        let result = estimator.marginal_effects(
            MarginalEffectsAt::Overall,
            MarginalEffectsTarget::ExpectedObserved,
            1.5,
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5
            })
        );
    }

    /// `target="E[y*|x]"`（潜在変数の期待値）は`dydx_j=βⱼ`という自明な形（評価点`x`に
    /// 依存しない）ため、`at`（overall/mean/median）に関わらず常に同じ値になり、
    /// その標準誤差も`β`自体の標準誤差（`std_errors()`の対応する要素）と厳密に一致する
    /// （ヤコビアンが`β`ブロックで恒等写像・`σ`列がゼロに退化するため、数値微分に頼らず
    /// 厳密等式として検証できる）。
    #[test]
    fn fit_marginal_effects_expected_latent_equals_beta_and_matches_beta_std_error() {
        let estimator = TobitEstimator::fit(
            multivariate_censored_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        for at in [
            MarginalEffectsAt::Overall,
            MarginalEffectsAt::Mean,
            MarginalEffectsAt::Median,
        ] {
            let effects = estimator
                .marginal_effects(at, MarginalEffectsTarget::ExpectedLatent, 0.95)
                .unwrap();
            assert_eq!(effects.param_names(), ["x1".to_string(), "x2".to_string()]);
            for (idx, j) in (1..3).enumerate() {
                assert!((effects.effect()[idx] - estimator.params()[j]).abs() < 1e-12);
                assert!(
                    (effects.std_errors()[idx] - estimator.std_errors()[j]).abs() < 1e-9,
                    "at={at:?}, idx={idx}, actual={}, expected={}",
                    effects.std_errors()[idx],
                    estimator.std_errors()[j]
                );
            }
        }
    }

    /// `target="E[y|x]"`（既定）の`at="overall"`（AME）を、実装内部（`target_w_and_s`）とは
    /// 別に書いた閉形式`expected_observed_closed_form`を各観測で平均する方法で独立に
    /// 再計算し、突き合わせる。標準誤差は、`dydx_j`を`prob_uncensored_closed_form`
    /// （非打ち切り確率の閉形式）×`βⱼ`という独立な式で表し、fit済みパラメータの周りで
    /// 数値微分したヤコビアンと`cov_params`の二次形式で計算した分散の平方根と突き合わせる
    /// （Logitの`marginal_effects_overall_matches_independently_recomputed_dydx_and_delta_
    /// method_se`と同じ検算パターン）。
    #[test]
    fn fit_marginal_effects_expected_observed_overall_matches_independently_recomputed_dydx_and_delta_method_se()
     {
        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 2;

        let effects = estimator
            .marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ExpectedObserved,
                0.95,
            )
            .unwrap();
        assert_eq!(effects.param_names(), ["x1".to_string()]);

        // effectの独立再計算: 各観測でE[y|x]を数値微分してx1の限界効果を求め、平均する
        // （`target_w_and_s`の`P(uncensored)*βⱼ`という閉形式は経由しない）。
        let h = 1e-6;
        let dydx_at_row = |params: &[f64], sigma: f64, i: usize| -> f64 {
            let mu = |x1_val: f64| -> f64 { params[0] + params[1] * x1_val };
            let x1 = *x.get(i, 1);
            let plus = expected_observed_closed_form(mu(x1 + h), sigma, lower, upper);
            let minus = expected_observed_closed_form(mu(x1 - h), sigma, lower, upper);
            (plus - minus) / (2.0 * h)
        };
        let dydx_ame = |params: &[f64], sigma: f64| -> f64 {
            (0..n).map(|i| dydx_at_row(params, sigma, i)).sum::<f64>() / (n as f64)
        };
        let params = estimator.params();
        let sigma = estimator.sigma();
        assert!(
            (effects.effect()[0] - dydx_ame(params, sigma)).abs() < 1e-6,
            "actual={}, expected={}",
            effects.effect()[0],
            dydx_ame(params, sigma)
        );

        // 数値微分とは別の、McDonald-Moffitt(1980)の厳密な恒等式
        // `∂E[y|x]/∂xⱼ = P(uncensored|x)*βⱼ`（`target_w_and_s`のdocコメント「数式」参照）
        // による検算。`prob_uncensored_closed_form`は数値微分を一切経由しない厳密な閉形式
        // のため、数値誤差なしで一致するはず。
        let expected_dydx_exact: f64 = (0..n)
            .map(|i| {
                let mu: f64 = params[0] + params[1] * (*x.get(i, 1));
                prob_uncensored_closed_form(mu, sigma, lower, upper) * params[1]
            })
            .sum::<f64>()
            / (n as f64);
        assert!(
            (effects.effect()[0] - expected_dydx_exact).abs() < 1e-9,
            "actual={}, expected={expected_dydx_exact}",
            effects.effect()[0]
        );

        // 標準誤差の独立検証: `dydx_ame`をfit済み`(β,σ)`の周りで数値微分してヤコビアン行を
        // 求め、`cov_params`との二次形式で分散を計算する。
        let h_param = 1e-5;
        let mut jac_row = vec![0.0; k + 1];
        for m in 0..k {
            let mut plus_p = params.to_vec();
            plus_p[m] += h_param;
            let mut minus_p = params.to_vec();
            minus_p[m] -= h_param;
            jac_row[m] = (dydx_ame(&plus_p, sigma) - dydx_ame(&minus_p, sigma)) / (2.0 * h_param);
        }
        jac_row[k] = (dydx_ame(params, sigma + h_param) - dydx_ame(params, sigma - h_param))
            / (2.0 * h_param);

        let mut var = 0.0;
        for a in 0..=k {
            for b in 0..=k {
                var += jac_row[a] * (*estimator.cov_params().get(a, b)) * jac_row[b];
            }
        }
        let expected_se = var.sqrt();
        assert!(
            (effects.std_errors()[0] - expected_se).abs() < 1e-4,
            "actual={}, expected={expected_se}",
            effects.std_errors()[0]
        );
    }

    /// `at="mean"`/`"median"`が`at="overall"`と異なる代表点で評価されるため、一般には
    /// 異なる値になることを確認する（Logitの同名テストと同じ理由）。
    #[test]
    fn fit_marginal_effects_expected_observed_at_mean_and_median_differ_from_overall() {
        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        let overall = estimator
            .marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ExpectedObserved,
                0.95,
            )
            .unwrap();
        let at_mean = estimator
            .marginal_effects(
                MarginalEffectsAt::Mean,
                MarginalEffectsTarget::ExpectedObserved,
                0.95,
            )
            .unwrap();
        let at_median = estimator
            .marginal_effects(
                MarginalEffectsAt::Median,
                MarginalEffectsTarget::ExpectedObserved,
                0.95,
            )
            .unwrap();

        assert!((overall.effect()[0] - at_mean.effect()[0]).abs() > 1e-9);
        assert!((at_mean.effect()[0] - at_median.effect()[0]).abs() < 1e-12);

        // x1の平均・中央値がともに4.5（[1..8]の対称なデータのため）であることを利用し、
        // `expected_observed_closed_form`から独立に再計算したdydxと突き合わせる。
        let params = estimator.params();
        let sigma = estimator.sigma();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        let mu = |x1_val: f64| -> f64 { params[0] + params[1] * x1_val };
        let h = 1e-6;
        let expected_dydx = (expected_observed_closed_form(mu(4.5 + h), sigma, lower, upper)
            - expected_observed_closed_form(mu(4.5 - h), sigma, lower, upper))
            / (2.0 * h);
        assert!((at_mean.effect()[0] - expected_dydx).abs() < 1e-6);
    }

    /// `target="E[y|x]"`が右打ち切りのみのデータでも正しく計算できることを、
    /// `expected_observed_closed_form`（右打ち切り分岐）で独立に再計算した`effect`と
    /// 突き合わせて検証する。
    #[test]
    fn fit_marginal_effects_expected_observed_matches_independent_recomputation_for_right_only_censoring()
     {
        let estimator = TobitEstimator::fit(
            right_censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        assert_eq!(lower, None);
        let x = estimator.input().x();
        let n = x.nrows();
        let params = estimator.params();
        let sigma = estimator.sigma();

        let effects = estimator
            .marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ExpectedObserved,
                0.95,
            )
            .unwrap();

        let h = 1e-6;
        let mu = |x1_val: f64| -> f64 { params[0] + params[1] * x1_val };
        let expected_dydx: f64 = (0..n)
            .map(|i| {
                let x1 = *x.get(i, 1);
                let plus = expected_observed_closed_form(mu(x1 + h), sigma, lower, upper);
                let minus = expected_observed_closed_form(mu(x1 - h), sigma, lower, upper);
                (plus - minus) / (2.0 * h)
            })
            .sum::<f64>()
            / (n as f64);

        assert!(
            (effects.effect()[0] - expected_dydx).abs() < 1e-6,
            "actual={}, expected={expected_dydx}",
            effects.effect()[0]
        );
    }

    /// `target="E[y|x]"`が両側打ち切りのデータでも正しく計算できることを、
    /// `expected_observed_closed_form`（両側打ち切り分岐）で独立に再計算した`effect`と
    /// 突き合わせて検証する。
    #[test]
    fn fit_marginal_effects_expected_observed_matches_independent_recomputation_for_two_sided_censoring()
     {
        let estimator = TobitEstimator::fit(
            two_sided_censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        assert_eq!(lower, Some(0.0));
        assert_eq!(upper, Some(6.0));
        let x = estimator.input().x();
        let n = x.nrows();
        let params = estimator.params();
        let sigma = estimator.sigma();

        let effects = estimator
            .marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ExpectedObserved,
                0.95,
            )
            .unwrap();

        let h = 1e-6;
        let mu = |x1_val: f64| -> f64 { params[0] + params[1] * x1_val };
        let expected_dydx: f64 = (0..n)
            .map(|i| {
                let x1 = *x.get(i, 1);
                let plus = expected_observed_closed_form(mu(x1 + h), sigma, lower, upper);
                let minus = expected_observed_closed_form(mu(x1 - h), sigma, lower, upper);
                (plus - minus) / (2.0 * h)
            })
            .sum::<f64>()
            / (n as f64);

        assert!(
            (effects.effect()[0] - expected_dydx).abs() < 1e-6,
            "actual={}, expected={expected_dydx}",
            effects.effect()[0]
        );
    }

    /// `target="P(uncensored)"`（左打ち切りのみ）を、`prob_uncensored_closed_form`を
    /// `mu`について数値微分した`dprob_uncensored_dmu_closed_form`×`βⱼ`という
    /// （`target_w_and_s`の`ProbUncensored`分岐とは独立の）式で再計算し、effect・標準誤差
    /// 双方を突き合わせる。
    #[test]
    fn fit_marginal_effects_prob_uncensored_overall_matches_independently_recomputed_dydx_and_delta_method_se()
     {
        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 2;

        let effects = estimator
            .marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ProbUncensored,
                0.95,
            )
            .unwrap();

        let dydx_ame = |params: &[f64], sigma: f64| -> f64 {
            (0..n)
                .map(|i| {
                    let mu: f64 = params[0] + params[1] * (*x.get(i, 1));
                    dprob_uncensored_dmu_closed_form(mu, sigma, lower, upper) * params[1]
                })
                .sum::<f64>()
                / (n as f64)
        };
        let params = estimator.params();
        let sigma = estimator.sigma();
        assert!(
            (effects.effect()[0] - dydx_ame(params, sigma)).abs() < 1e-9,
            "actual={}, expected={}",
            effects.effect()[0],
            dydx_ame(params, sigma)
        );

        let h_param = 1e-5;
        let mut jac_row = vec![0.0; k + 1];
        for m in 0..k {
            let mut plus_p = params.to_vec();
            plus_p[m] += h_param;
            let mut minus_p = params.to_vec();
            minus_p[m] -= h_param;
            jac_row[m] = (dydx_ame(&plus_p, sigma) - dydx_ame(&minus_p, sigma)) / (2.0 * h_param);
        }
        jac_row[k] = (dydx_ame(params, sigma + h_param) - dydx_ame(params, sigma - h_param))
            / (2.0 * h_param);

        let mut var = 0.0;
        for a in 0..=k {
            for b in 0..=k {
                var += jac_row[a] * (*estimator.cov_params().get(a, b)) * jac_row[b];
            }
        }
        let expected_se = var.sqrt();
        assert!(
            (effects.std_errors()[0] - expected_se).abs() < 1e-4,
            "actual={}, expected={expected_se}",
            effects.std_errors()[0]
        );
    }

    /// `target="P(uncensored)"`は右打ち切りのみだと符号が反転する（`x`の増加が上側の
    /// 打ち切り確率を高めるため非打ち切り確率は減少する。`target_w_and_s`のdocコメント
    /// 「数式」参照）。この符号反転が正しく実装されていることを、右打ち切りのみの
    /// データで独立再計算したdydxと突き合わせて確認する。
    #[test]
    fn fit_marginal_effects_prob_uncensored_matches_independent_recomputation_for_right_only_censoring()
     {
        let estimator = TobitEstimator::fit(
            right_censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        let x = estimator.input().x();
        let n = x.nrows();
        let params = estimator.params();
        let sigma = estimator.sigma();

        let effects = estimator
            .marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ProbUncensored,
                0.95,
            )
            .unwrap();

        let expected_dydx: f64 = (0..n)
            .map(|i| {
                let mu: f64 = params[0] + params[1] * (*x.get(i, 1));
                dprob_uncensored_dmu_closed_form(mu, sigma, lower, upper) * params[1]
            })
            .sum::<f64>()
            / (n as f64);

        assert!(
            (effects.effect()[0] - expected_dydx).abs() < 1e-9,
            "actual={}, expected={expected_dydx}",
            effects.effect()[0]
        );
        // x1の真の係数は正（+2相当）なので、右打ち切りのみでは非打ち切り確率の限界効果は
        // 負になるはず（xが増えるほど上側で打ち切られやすくなるため）。
        assert!(effects.effect()[0] < 0.0, "effect={}", effects.effect()[0]);
    }

    /// `target="P(uncensored)"`が両側打ち切りのデータでも正しく計算できることを検証する。
    #[test]
    fn fit_marginal_effects_prob_uncensored_matches_independent_recomputation_for_two_sided_censoring()
     {
        let estimator = TobitEstimator::fit(
            two_sided_censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        let x = estimator.input().x();
        let n = x.nrows();
        let params = estimator.params();
        let sigma = estimator.sigma();

        let effects = estimator
            .marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ProbUncensored,
                0.95,
            )
            .unwrap();

        let expected_dydx: f64 = (0..n)
            .map(|i| {
                let mu: f64 = params[0] + params[1] * (*x.get(i, 1));
                dprob_uncensored_dmu_closed_form(mu, sigma, lower, upper) * params[1]
            })
            .sum::<f64>()
            / (n as f64);

        assert!(
            (effects.effect()[0] - expected_dydx).abs() < 1e-9,
            "actual={}, expected={expected_dydx}",
            effects.effect()[0]
        );
    }

    /// `predict(ExpectedLatent)`は`E[y*|x]=x'β`という自明な式（打ち切りに依存しない）
    /// なので、`x.get(i,·)`から直接計算した線形予測子と厳密に一致するはず。
    #[test]
    fn fit_predict_expected_latent_equals_linear_predictor() {
        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 2;
        let params = estimator.params();

        let predicted = estimator
            .predict(MarginalEffectsTarget::ExpectedLatent)
            .unwrap();
        assert_eq!(predicted.len(), n);
        for (i, &predicted_i) in predicted.iter().enumerate() {
            let expected: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            assert!((predicted_i - expected).abs() < 1e-12);
        }
    }

    /// `predict(ExpectedObserved)`・`predict(ProbUncensored)`を、`marginal_effects`の
    /// テストで使った閉形式（`expected_observed_closed_form`/`prob_uncensored_closed_form`、
    /// `target_w_and_s`/`predicted_value`の実装コードとは独立の`(lower,upper)`ごとの
    /// `match`分岐）と観測ごとに突き合わせる（左打ち切りのみ）。
    #[test]
    fn fit_predict_matches_independent_recomputation_for_left_only_censoring() {
        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 2;
        let params = estimator.params();
        let sigma = estimator.sigma();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        assert_eq!(lower, Some(0.0));
        assert_eq!(upper, None);

        let predicted_observed = estimator
            .predict(MarginalEffectsTarget::ExpectedObserved)
            .unwrap();
        let predicted_prob = estimator
            .predict(MarginalEffectsTarget::ProbUncensored)
            .unwrap();
        for i in 0..n {
            let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            let expected_observed = expected_observed_closed_form(mu, sigma, lower, upper);
            let expected_prob = prob_uncensored_closed_form(mu, sigma, lower, upper);
            assert!(
                (predicted_observed[i] - expected_observed).abs() < 1e-9,
                "row={i}, actual={}, expected={expected_observed}",
                predicted_observed[i]
            );
            assert!(
                (predicted_prob[i] - expected_prob).abs() < 1e-9,
                "row={i}, actual={}, expected={expected_prob}",
                predicted_prob[i]
            );
        }
    }

    /// 右打ち切りのみのデータでも`predict(ExpectedObserved)`・`predict(ProbUncensored)`が
    /// 独立再計算と一致することを確認する。
    #[test]
    fn fit_predict_matches_independent_recomputation_for_right_only_censoring() {
        let estimator = TobitEstimator::fit(
            right_censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 2;
        let params = estimator.params();
        let sigma = estimator.sigma();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        assert_eq!(lower, None);
        assert_eq!(upper, Some(6.0));

        let predicted_observed = estimator
            .predict(MarginalEffectsTarget::ExpectedObserved)
            .unwrap();
        let predicted_prob = estimator
            .predict(MarginalEffectsTarget::ProbUncensored)
            .unwrap();
        for i in 0..n {
            let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            let expected_observed = expected_observed_closed_form(mu, sigma, lower, upper);
            let expected_prob = prob_uncensored_closed_form(mu, sigma, lower, upper);
            assert!((predicted_observed[i] - expected_observed).abs() < 1e-9);
            assert!((predicted_prob[i] - expected_prob).abs() < 1e-9);
        }
    }

    /// 両側打ち切りのデータでも`predict(ExpectedObserved)`・`predict(ProbUncensored)`が
    /// 独立再計算と一致することを確認する。
    #[test]
    fn fit_predict_matches_independent_recomputation_for_two_sided_censoring() {
        let estimator = TobitEstimator::fit(
            two_sided_censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 2;
        let params = estimator.params();
        let sigma = estimator.sigma();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        assert_eq!(lower, Some(0.0));
        assert_eq!(upper, Some(6.0));

        let predicted_observed = estimator
            .predict(MarginalEffectsTarget::ExpectedObserved)
            .unwrap();
        let predicted_prob = estimator
            .predict(MarginalEffectsTarget::ProbUncensored)
            .unwrap();
        for i in 0..n {
            let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            let expected_observed = expected_observed_closed_form(mu, sigma, lower, upper);
            let expected_prob = prob_uncensored_closed_form(mu, sigma, lower, upper);
            assert!((predicted_observed[i] - expected_observed).abs() < 1e-9);
            assert!((predicted_prob[i] - expected_prob).abs() < 1e-9);
        }
    }

    /// `censoring_fit_check`（左打ち切りのみ）: `lower`カテゴリが`Some`・`upper`カテゴリが
    /// `None`であること、実測比率（`y==0.0`の観測数/n）・モデル含意の平均確率
    /// （`Φ((lower-μᵢ)/σ)`の平均、`predicted_value`とは独立に`Normal::cdf`を直接呼んで
    /// 計算）の両方を独立再計算と突き合わせる。3カテゴリの`model_implied_rate`が合計1に
    /// なる恒等式も確認する。
    #[test]
    fn fit_censoring_fit_check_matches_independent_recomputation_for_left_only_censoring() {
        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 2;
        let params = estimator.params();
        let sigma = estimator.sigma();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let check = estimator.censoring_fit_check();
        assert!(check.upper().is_none());
        let lower_check = check.lower().unwrap();

        // censored_regression_input: y=[0,0,1.15,2.9,5.2,6.85,9.1,10.95]、lower=0.0で
        // 打ち切られているのはx=1,2の2件（実測打ち切り率2/8=0.25）
        assert!((lower_check.observed_rate() - 0.25).abs() < 1e-12);
        assert!((check.uncensored().observed_rate() - 0.75).abs() < 1e-12);

        let expected_lower_rate: f64 = (0..n)
            .map(|i| {
                let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
                normal.cdf((0.0 - mu) / sigma)
            })
            .sum::<f64>()
            / (n as f64);
        assert!((lower_check.model_implied_rate() - expected_lower_rate).abs() < 1e-9);

        let expected_uncensored_rate: f64 = (0..n)
            .map(|i| {
                let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
                prob_uncensored_closed_form(mu, sigma, Some(0.0), None)
            })
            .sum::<f64>()
            / (n as f64);
        assert!((check.uncensored().model_implied_rate() - expected_uncensored_rate).abs() < 1e-9);

        // model_implied_rateは合計1になるはず（lower + uncensored、upperはこのケースでは無い）
        assert!(
            (lower_check.model_implied_rate() + check.uncensored().model_implied_rate() - 1.0)
                .abs()
                < 1e-9
        );
    }

    /// `censoring_fit_check`（右打ち切りのみ）: `lower`が`None`・`upper`が`Some`になること、
    /// 実測比率・モデル含意の平均確率を独立再計算と突き合わせる。
    #[test]
    fn fit_censoring_fit_check_matches_independent_recomputation_for_right_only_censoring() {
        let estimator = TobitEstimator::fit(
            right_censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 2;
        let params = estimator.params();
        let sigma = estimator.sigma();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let check = estimator.censoring_fit_check();
        assert!(check.lower().is_none());
        let upper_check = check.upper().unwrap();

        // right_censored_regression_input: y=[-2.85,-0.95,1.15,2.9,5.2,6.0,6.0,6.0]、
        // upper=6.0で打ち切られているのはx=6,7,8の3件（実測打ち切り率3/8=0.375）
        assert!((upper_check.observed_rate() - 0.375).abs() < 1e-12);
        assert!((check.uncensored().observed_rate() - 0.625).abs() < 1e-12);

        let expected_upper_rate: f64 = (0..n)
            .map(|i| {
                let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
                1.0 - normal.cdf((6.0 - mu) / sigma)
            })
            .sum::<f64>()
            / (n as f64);
        assert!((upper_check.model_implied_rate() - expected_upper_rate).abs() < 1e-9);
    }

    /// `censoring_fit_check`（両側打ち切り）: `lower`・`upper`双方が`Some`になり、
    /// 3カテゴリの実測比率・モデル含意の平均確率いずれも独立再計算と一致し、かつ
    /// それぞれ合計1になることを確認する。
    #[test]
    fn fit_censoring_fit_check_matches_independent_recomputation_for_two_sided_censoring() {
        let estimator = TobitEstimator::fit(
            two_sided_censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 2;
        let params = estimator.params();
        let sigma = estimator.sigma();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let check = estimator.censoring_fit_check();
        let lower_check = check.lower().unwrap();
        let upper_check = check.upper().unwrap();

        // two_sided_censored_regression_input: y=[0,0,1.15,2.9,5.2,6.0,6.0,6.0]、
        // lower=0.0でx=1,2（2件）・upper=6.0でx=6,7,8（3件）が打ち切り
        assert!((lower_check.observed_rate() - 0.25).abs() < 1e-12);
        assert!((upper_check.observed_rate() - 0.375).abs() < 1e-12);
        assert!((check.uncensored().observed_rate() - 0.375).abs() < 1e-12);
        assert!(
            (lower_check.observed_rate()
                + check.uncensored().observed_rate()
                + upper_check.observed_rate()
                - 1.0)
                .abs()
                < 1e-12
        );

        let expected_lower_rate: f64 = (0..n)
            .map(|i| {
                let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
                normal.cdf((0.0 - mu) / sigma)
            })
            .sum::<f64>()
            / (n as f64);
        let expected_upper_rate: f64 = (0..n)
            .map(|i| {
                let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
                1.0 - normal.cdf((6.0 - mu) / sigma)
            })
            .sum::<f64>()
            / (n as f64);
        let expected_uncensored_rate: f64 = (0..n)
            .map(|i| {
                let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
                prob_uncensored_closed_form(mu, sigma, Some(0.0), Some(6.0))
            })
            .sum::<f64>()
            / (n as f64);
        assert!((lower_check.model_implied_rate() - expected_lower_rate).abs() < 1e-9);
        assert!((upper_check.model_implied_rate() - expected_upper_rate).abs() < 1e-9);
        assert!((check.uncensored().model_implied_rate() - expected_uncensored_rate).abs() < 1e-9);
        assert!(
            (lower_check.model_implied_rate()
                + check.uncensored().model_implied_rate()
                + upper_check.model_implied_rate()
                - 1.0)
                .abs()
                < 1e-9
        );
    }

    /// 上記の`predict`テストは全て単一の傾き係数（`k=2`、切片+`x1`）のフィクスチャのみを
    /// 使っており、`(0..k).map(|j| *x.get(i,j)*self.params[j])`という列インデックス計算の
    /// 配線ミス（列の取り違え等）を構造的に検出できない（`multivariate_censored_input`の
    /// docコメント「配線ミスは単一変数データでは検出できない」と同じ理由、
    /// rust-reviewer指摘）。傾き係数2個（`x1`・`x2`）の多変量データセットで
    /// `expected_observed_closed_form`/`prob_uncensored_closed_form`との独立再計算を検証する。
    #[test]
    fn fit_predict_matches_independent_recomputation_for_multivariate_design() {
        let estimator = TobitEstimator::fit(
            multivariate_censored_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 3;
        let params = estimator.params();
        let sigma = estimator.sigma();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();

        let predicted_latent = estimator
            .predict(MarginalEffectsTarget::ExpectedLatent)
            .unwrap();
        let predicted_observed = estimator
            .predict(MarginalEffectsTarget::ExpectedObserved)
            .unwrap();
        let predicted_prob = estimator
            .predict(MarginalEffectsTarget::ProbUncensored)
            .unwrap();
        for i in 0..n {
            let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            assert!((predicted_latent[i] - mu).abs() < 1e-12);
            let expected_observed = expected_observed_closed_form(mu, sigma, lower, upper);
            let expected_prob = prob_uncensored_closed_form(mu, sigma, lower, upper);
            assert!((predicted_observed[i] - expected_observed).abs() < 1e-9);
            assert!((predicted_prob[i] - expected_prob).abs() < 1e-9);
        }
    }

    /// `predict_new_data`（out-of-sample）が独立に再計算した
    /// 値と一致すること。`fit_predict_matches_independent_recomputation_for_
    /// multivariate_design`と同じモデルを使い、学習データとは異なる新規のx値で
    /// 3つの`target`すべてを検証する。
    #[test]
    fn predict_new_data_matches_independent_recomputation_for_multivariate_design() {
        let estimator = TobitEstimator::fit(
            multivariate_censored_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let params = estimator.params().to_vec();
        let sigma = estimator.sigma();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();

        // 学習データには無い新規のx値
        let new_x_columns = vec![vec![15.0, 25.0], vec![3.0, -1.0]];
        let predicted_latent = estimator
            .predict_new_data(MarginalEffectsTarget::ExpectedLatent, &new_x_columns)
            .unwrap();
        let predicted_observed = estimator
            .predict_new_data(MarginalEffectsTarget::ExpectedObserved, &new_x_columns)
            .unwrap();
        let predicted_prob = estimator
            .predict_new_data(MarginalEffectsTarget::ProbUncensored, &new_x_columns)
            .unwrap();

        assert_eq!(predicted_latent.len(), 2);
        for i in 0..2 {
            let mu = params[0] + new_x_columns[0][i] * params[1] + new_x_columns[1][i] * params[2];
            assert!((predicted_latent[i] - mu).abs() < 1e-12);
            let expected_observed = expected_observed_closed_form(mu, sigma, lower, upper);
            let expected_prob = prob_uncensored_closed_form(mu, sigma, lower, upper);
            assert!((predicted_observed[i] - expected_observed).abs() < 1e-9);
            assert!((predicted_prob[i] - expected_prob).abs() < 1e-9);
        }
    }

    /// `predict_new_data`が`has_intercept=false`でも3つの`target`すべてで
    /// 正しく動作すること（Logit/Probitの対応するテストと同じ理由の対称性確保、
    /// rust-reviewer指摘）。
    #[test]
    fn predict_new_data_without_intercept_matches_independent_recomputation() {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x2 = vec![1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0];
        let y = vec![0.0, 0.0, 1.15, 2.9, 5.2, 6.85, 9.1, 10.95];
        let input = TobitInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            false,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap();

        let estimator = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let params = estimator.params().to_vec();
        let sigma = estimator.sigma();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();

        let new_x_columns = vec![vec![15.0, 25.0], vec![3.0, -1.0]];
        let predicted_latent = estimator
            .predict_new_data(MarginalEffectsTarget::ExpectedLatent, &new_x_columns)
            .unwrap();
        let predicted_observed = estimator
            .predict_new_data(MarginalEffectsTarget::ExpectedObserved, &new_x_columns)
            .unwrap();
        let predicted_prob = estimator
            .predict_new_data(MarginalEffectsTarget::ProbUncensored, &new_x_columns)
            .unwrap();

        assert_eq!(predicted_observed.len(), 2);
        for i in 0..2 {
            let mu = new_x_columns[0][i] * params[0] + new_x_columns[1][i] * params[1];
            assert!((predicted_latent[i] - mu).abs() < 1e-12);
            let expected_observed = expected_observed_closed_form(mu, sigma, lower, upper);
            let expected_prob = prob_uncensored_closed_form(mu, sigma, lower, upper);
            assert!((predicted_observed[i] - expected_observed).abs() < 1e-9);
            assert!((predicted_prob[i] - expected_prob).abs() < 1e-9);
        }
    }

    /// `censoring_fit_check`版の多変量配線ミス検出テスト（上記`fit_predict_matches_
    /// independent_recomputation_for_multivariate_design`と同じ理由、rust-reviewer指摘）。
    #[test]
    fn fit_censoring_fit_check_matches_independent_recomputation_for_multivariate_design() {
        let estimator = TobitEstimator::fit(
            multivariate_censored_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 3;
        let params = estimator.params();
        let sigma = estimator.sigma();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let check = estimator.censoring_fit_check();
        assert!(check.upper().is_none());
        let lower_check = check.lower().unwrap();

        // multivariate_censored_input: y=[0,0,1.15,2.9,5.2,6.85,9.1,10.95]、lower=0.0で
        // 打ち切られているのはx1=1,2の2件（実測打ち切り率2/8=0.25、censored_regression_
        // inputと同じyだがx2列が追加された多変量設計行列）
        assert!((lower_check.observed_rate() - 0.25).abs() < 1e-12);
        assert!((check.uncensored().observed_rate() - 0.75).abs() < 1e-12);

        let expected_lower_rate: f64 = (0..n)
            .map(|i| {
                let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
                normal.cdf((0.0 - mu) / sigma)
            })
            .sum::<f64>()
            / (n as f64);
        assert!((lower_check.model_implied_rate() - expected_lower_rate).abs() < 1e-9);
    }

    /// 全件が下限（`lower`）で打ち切られている（非打ち切り観測が1件も無い）場合は
    /// `MleError::NoUncensoredObservations`を返す。実測で確認済み: このバリデーションが
    /// 無いと`fit()`は`converged=true`のまま統計的に無意味な巨大SE（100万倍オーダー）を
    /// 返す退化収束を起こしていた（rust-reviewer指摘ではなく`cargo llvm-cov`
    /// で病理ケースを調査中に発覚。参照実装`survival::survreg`は同種のデータで
    /// エラーを返すことをdevcontainer内で実際に確認した上で対応方針をユーザーに確認済み）。
    #[test]
    fn fit_returns_no_uncensored_observations_error_when_all_observations_are_censored_at_lower() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![0.0; 8];
        let input = TobitInput::from_columns(
            &y,
            &[x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap();

        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::NoUncensoredObservations {
                lower: Some(0.0),
                upper: None
            }
        );
    }

    /// 上のテストは`lower`側のみの片側打ち切りを検証しているが、`at_lower`/`at_upper`の
    /// 判定ロジックが対称的に実装されていることを、`upper`側のみの片側打ち切りでも
    /// 独立に確認する（rust-reviewer指摘）。
    #[test]
    fn fit_returns_no_uncensored_observations_error_when_all_observations_are_censored_at_upper() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![6.0; 8];
        let input = TobitInput::from_columns(
            &y,
            &[x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            None,
            Some(6.0),
        )
        .unwrap();

        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::NoUncensoredObservations {
                lower: None,
                upper: Some(6.0)
            }
        );
    }

    /// 両側打ち切りで、非打ち切り観測が1件も無い（全観測が`lower`または`upper`ちょうど）
    /// 場合も同じエラーになることを確認する（`lower`側だけの判定に留まらないことの検証）。
    #[test]
    fn fit_returns_no_uncensored_observations_error_when_all_observations_are_censored_two_sided() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 6.0];
        let input = TobitInput::from_columns(
            &y,
            &[x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            Some(6.0),
        )
        .unwrap();

        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::NoUncensoredObservations {
                lower: Some(0.0),
                upper: Some(6.0)
            }
        );
    }

    /// 非打ち切り観測がちょうど1件ある場合は`NoUncensoredObservations`を返さない
    /// （「1件以上」という条件の境界値、`validate_sufficient_observations`と同じ
    /// `>=`境界の考え方）。このデータは病理的に近い（実際にNewtonは収束しない）ため
    /// `raise_on_non_convergence=false`で受け、`NonConvergence`ではなく
    /// バリデーション自体は通過することだけを確認する。
    #[test]
    fn fit_does_not_return_no_uncensored_observations_error_when_exactly_one_observation_is_uncensored()
     {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.95];
        let input = TobitInput::from_columns(
            &y,
            &[x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap();

        let result = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: false,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        );
        assert!(
            !matches!(result, Err(MleError::NoUncensoredObservations { .. })),
            "{result:?}"
        );
    }

    /// property-basedテスト。`logit.rs`/`probit.rs`の`mod proptests`と同型の設計だが、
    /// Tobitは打ち切り回帰でありLogit/Probitと異なり(a)説明変数の係数`β`に加え誤差項の
    /// 標準偏差`σ`（`s=logσ`でパラメータ化）を持つ、(b)観測ごとの尤度が非打ち切り/左
    /// 打ち切り/右打ち切りの3分岐になる、という構造上の違いがある。そのため
    /// `score_is_near_zero_at_converged_params`は3分岐+σパラメータ化のFOCを独立に
    /// 再導出せず、`TobitProblem::scores()`（本番のOPG/サンドイッチSE計算が使うのと
    /// 同じ観測ごとのスコア行列）を再利用する（ユーザー確認済み。Probitの
    /// `clamped_pdf_cdf`共有と同種のトレードオフで、`scores()`自体のバグはこの
    /// プロパティでは検出できない）。
    ///
    /// **単なる重複ではない**: `fit()`は`TobitScaling`で標準化した空間で最適化し、
    /// `est.params()`/`est.sigma()`は逆変換（destandardization）後の値。このプロパティは
    /// `TobitProblem::new(est.input())`（生スケール）に`est.params()`/`est.sigma().ln()`を
    /// 渡してFOCを評価するため、多数のランダム構成で標準化⇔生スケールの逆変換自体を
    /// 実質的に検証する（既存の固定テスト`gradient_matches_numerical_differentiation_of_
    /// cost`/`scores_sum_to_negative_gradient`はいずれも`TobitScaling`を経由しないため、
    /// この往復変換を検証できていなかった）。標準化パラメータ空間でスコアが0なら生
    /// スケールでもスコアが0になることは、標準化がθに依存しないアフィン変換であり
    /// 対数尤度の差が定数項（θ非依存のヤコビアン項）のみであることから理論的に導ける
    /// （rust-reviewer確認済み）。
    ///
    /// ケース生成は左打ち切り（`lower=0`のみ）・右打ち切り（`upper=0`のみ）・両側打ち切り
    /// （`lower=-1`・`upper=1`）の3通りを`censoring_strategy`でランダムに選ぶ
    /// （`censored_contribution`の`direction=-1.0`分岐や左右混在のデータセットも通る）。
    /// `y* = β₀ + Σxⱼβⱼ + σε`（`ε`は標準正規、`u∈(1e-6, 1-1e-6)`の逆CDFでサンプリング）
    /// を計算し、下限未満・上限超過の観測をそれぞれの境界値で打ち切る。
    ///
    /// 較正値（`n=k+30..=80`, `beta∈[-1,1]`, `sigma∈[0.5,2]`）は`PROPTEST_CASES=5000`
    /// （デフォルト256の約20倍）まで増やしても`NoUncensoredObservations`等による
    /// `prop_assume`棄却が原因の失敗（"too many global rejects"）を起こさないことを
    /// 実測済み（rust-reviewer確認）。
    ///
    /// プロパティの有効性検証（バグ注入→検出確認→元に戻す）は当初の3件について実施済み
    /// （4件目の`negating_y_and_swapping_bounds_negates_coefficients_only`は右打ち切り・
    /// 両側打ち切りへの拡張と同時に追加し、右打ち切り分岐へのバグ注入で検出を確認した）:
    /// `score_is_near_zero_at_converged_params`は`TobitProblem::gradient`の`grad[j]`
    /// 計算に定数オフセットを注入、`coefficients_and_se_are_invariant_to_column_order`は
    /// `TobitInput::from_columns`の`param_names`を逆順にするバグを注入、
    /// `hc0_std_errors_are_at_most_hc1_std_errors`は`nonlinear::common::
    /// sandwich_cov_params`のHC1補正係数を反転（`n/(n-k)`→`(n-k)/n`、Logit/Probitと
    /// 共有するロジックのため同一のバグ注入で確認）するバグを注入し、いずれも検出
    /// できることを確認した。
    ///
    /// `ExpectedUncensored`の3件（`decomposition_identity_holds_for_random_inputs`・
    /// `expected_uncensored_stays_within_bounds_and_w_in_unit_interval`・
    /// `mirror_symmetry_negates_mean_and_derivative_and_keeps_weight`）は`fit()`を介さず
    /// `predicted_value`・`target_w_and_s`を乱数の`(μ,σ,境界)`で直接検証する。バグ注入で
    /// 有効性を確認済み: 下側の裾の鏡像で`(Pa,Pb)`・`(A,B)`の入れ替えを外す→3件とも失敗、
    /// 平均を`μ+σ(Pa+Pb)`にする→3件とも失敗、`∂w/∂μ`の符号を誤る→鏡像対称性が失敗。
    mod proptests {
        use super::*;
        use proptest::collection;
        use proptest::prelude::*;

        // 高kはbenchmarkのmany_regressorsシナリオでカバー済みのため小さく保つ
        // （logit.rs/probit.rsと同じ理由）。
        const MAX_K: usize = 3;

        /// `tobit_case_strategy`が生成するタプル: `(n, k, x_cols, beta, sigma, u, keys)`。
        /// `k`は切片を除いた説明変数の数（`beta`は切片込みで`k+1`要素）。
        type TobitCase = (
            usize,
            usize,
            Vec<Vec<f64>>,
            Vec<f64>,
            f64,
            Vec<f64>,
            Vec<u64>,
        );

        fn tobit_case_strategy() -> impl Strategy<Value = TobitCase> {
            (1..=MAX_K).prop_flat_map(|k| {
                (k + 30..=80usize).prop_flat_map(move |n| {
                    (
                        Just(n),
                        Just(k),
                        collection::vec(collection::vec(-2.0f64..2.0, n), k),
                        collection::vec(-1.0f64..1.0, k + 1),
                        0.5f64..2.0,
                        collection::vec(1e-6f64..1.0 - 1e-6, n),
                        collection::vec(any::<u64>(), k),
                    )
                })
            })
        }

        fn x_names(k: usize) -> Vec<String> {
            (1..=k).map(|i| format!("x{i}")).collect()
        }

        /// 左打ち切りの下限（打ち切りなしのTobitの意味が薄れない程度に、真の`y*`分布の
        /// 中心付近を狙う）。右打ち切りの上限はこの値と対称（`beta`の分布が0周りに対称な
        /// ため、左・右で打ち切り割合が同程度になる）。
        const LOWER_BOUND: f64 = 0.0;

        /// 打ち切り境界`(lower, upper)`。
        type Bounds = (Option<f64>, Option<f64>);

        /// 左打ち切り・右打ち切り・両側打ち切りをランダムに選ぶ。両側打ち切りの幅
        /// （`[-1, 1]`）は、`NoUncensoredObservations`等の`prop_assume`棄却が過剰にならない
        /// 程度に真の`y*`分布の中心付近を含める。
        fn censoring_strategy() -> impl Strategy<Value = Bounds> {
            prop_oneof![
                Just((Some(LOWER_BOUND), None)),
                Just((None, Some(LOWER_BOUND))),
                Just((Some(-1.0), Some(1.0))),
            ]
        }

        /// 真の`beta`・`sigma`から`y* = β₀+Σxⱼβⱼ+σε`を計算し、`bounds`で打ち切る
        /// （下限未満は下限、上限超過は上限に置き換える）。
        fn simulate_y(
            n: usize,
            x_cols: &[Vec<f64>],
            beta: &[f64],
            sigma: f64,
            u: &[f64],
            bounds: Bounds,
        ) -> Vec<f64> {
            let normal = Normal::standard();
            (0..n)
                .map(|i| {
                    let mut y_star = beta[0];
                    for (j, x_col) in x_cols.iter().enumerate() {
                        y_star += x_col[i] * beta[j + 1];
                    }
                    y_star += sigma * normal.inverse_cdf(u[i]);
                    if let Some(lower) = bounds.0
                        && y_star < lower
                    {
                        return lower;
                    }
                    if let Some(upper) = bounds.1
                        && y_star > upper
                    {
                        return upper;
                    }
                    y_star
                })
                .collect()
        }

        /// `solver`ごとの`tol`既定値の解決は`logit.rs`/`probit.rs`の`default_options`と
        /// 同じ（`docs/spec/tobit-spec.md`、Logit/Probitと共有する`run_solver`のため
        /// 同じ意味論）。
        fn default_options(cov_type: CovType, solver: SolverType) -> MleFitOptions {
            let tol = match solver {
                SolverType::Newton => 1e-6,
                SolverType::Bfgs | SolverType::Lbfgs => 1e-8,
            };
            MleFitOptions {
                solver,
                max_iter: 50,
                tol,
                raise_on_non_convergence: true,
                cov_type,
                confidence_level: 0.95,
            }
        }

        fn solver_strategy() -> impl Strategy<Value = SolverType> {
            prop_oneof![
                Just(SolverType::Newton),
                Just(SolverType::Bfgs),
                Just(SolverType::Lbfgs),
            ]
        }

        /// `logit.rs`/`probit.rs`と同じ理由で`1e-6`より緩い`1e-4`を使う
        /// （反復最適化の収束経路が列順序で変わりうるため）。
        fn assert_approx_eq(actual: f64, expected: f64, msg: &str) {
            let tol = 1e-4 * expected.abs().max(1.0);
            let diff = (actual - expected).abs();
            assert!(
                diff <= tol,
                "{msg}: actual={actual}, expected={expected}, diff={diff}, tol={tol}"
            );
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            /// MLEの一次条件（スコア方程式）: 収束点で`Σᵢsᵢ ≈ 0`（`sᵢ`は
            /// `TobitProblem::scores()`が返す観測ごとのスコア行、`β`の`k+1`列＋
            /// `s=logσ`の1列、計`k+2`列）が全パラメータ列で成り立つ。許容誤差の根拠は
            /// `logit.rs`の同名プロパティのdocコメント参照（`n`非依存の絶対閾値）。
            #[test]
            fn score_is_near_zero_at_converged_params(
                (n, k, x_cols, beta, sigma, u, _keys) in tobit_case_strategy(),
                bounds in censoring_strategy(),
                solver in solver_strategy(),
            ) {
                let y = simulate_y(n, &x_cols, &beta, sigma, &u, bounds);
                let names = x_names(k);
                let input = TobitInput::from_columns(
                    &y, &x_cols, names, true, "y".to_string(), bounds.0, bounds.1,
                ).unwrap();
                let result = TobitEstimator::fit(input, default_options(CovType::Classical, solver));
                prop_assume!(result.is_ok());
                let est = result.unwrap();

                let mut full_params: Vec<f64> = est.params().to_vec();
                full_params.push(est.sigma().ln());

                let problem = TobitProblem::new(est.input());
                let scores = problem.scores(&full_params);
                for j in 0..=(k + 1) {
                    let score: f64 = (0..n).map(|i| *scores.get(i, j)).sum();
                    prop_assert!(
                        score.abs() <= 1e-4,
                        "score[{j}] should be ~0, got {score} (solver={solver:?})"
                    );
                }
            }

            /// xの列順序を入れ替えても、係数名で対応付ければ係数・標準誤差の値は変わらない。
            /// `σ`は列順序に依存しない別パラメータのため、`std_errors()`の末尾要素
            /// （インデックス`k+1`）で個別に比較する。
            #[test]
            fn coefficients_and_se_are_invariant_to_column_order(
                (n, k, x_cols, beta, sigma, u, keys) in tobit_case_strategy()
                    .prop_filter("need >=2 columns to permute", |(_, k, _, _, _, _, _)| *k >= 2),
                bounds in censoring_strategy(),
                solver in solver_strategy(),
            ) {
                let y = simulate_y(n, &x_cols, &beta, sigma, &u, bounds);
                let names = x_names(k);
                let input1 = TobitInput::from_columns(
                    &y, &x_cols, names.clone(), true, "y".to_string(), bounds.0, bounds.1,
                ).unwrap();
                let result1 = TobitEstimator::fit(input1, default_options(CovType::Classical, solver));
                prop_assume!(result1.is_ok());
                let est1 = result1.unwrap();

                let mut order: Vec<usize> = (0..k).collect();
                order.sort_by_key(|&i| keys[i]);
                let permuted_x: Vec<Vec<f64>> = order.iter().map(|&i| x_cols[i].clone()).collect();
                let permuted_names: Vec<String> = order.iter().map(|&i| names[i].clone()).collect();

                let input2 = TobitInput::from_columns(
                    &y, &permuted_x, permuted_names, true, "y".to_string(), bounds.0, bounds.1,
                ).unwrap();
                let result2 = TobitEstimator::fit(input2, default_options(CovType::Classical, solver));
                prop_assume!(result2.is_ok());
                let est2 = result2.unwrap();

                let names1 = est1.input().param_names().to_vec();
                let names2 = est2.input().param_names().to_vec();
                let (params1, params2) = (est1.params(), est2.params());
                let (se1, se2) = (est1.std_errors(), est2.std_errors());
                for (idx1, name) in names1.iter().enumerate() {
                    let idx2 = names2.iter().position(|n| n == name)
                        .expect("name should exist in permuted result");
                    assert_approx_eq(params2[idx2], params1[idx1], &format!("param[{name}] under column permutation"));
                    assert_approx_eq(se2[idx2], se1[idx1], &format!("std_error[{name}] under column permutation"));
                }
                assert_approx_eq(est2.sigma(), est1.sigma(), "sigma under column permutation");
                assert_approx_eq(se2[k + 1], se1[k + 1], "std_error[sigma] under column permutation");
            }

            /// HC0の標準誤差は常にHC1以下（`docs/spec/tobit-spec.md`）。
            #[test]
            fn hc0_std_errors_are_at_most_hc1_std_errors(
                (n, k, x_cols, beta, sigma, u, _keys) in tobit_case_strategy(),
                bounds in censoring_strategy(),
                solver in solver_strategy(),
            ) {
                let y = simulate_y(n, &x_cols, &beta, sigma, &u, bounds);
                let names = x_names(k);
                let input1 = TobitInput::from_columns(
                    &y, &x_cols, names.clone(), true, "y".to_string(), bounds.0, bounds.1,
                ).unwrap();
                let result1 = TobitEstimator::fit(input1, default_options(CovType::Hc0, solver));
                prop_assume!(result1.is_ok());
                let est_hc0 = result1.unwrap();

                let input2 = TobitInput::from_columns(
                    &y, &x_cols, names, true, "y".to_string(), bounds.0, bounds.1,
                ).unwrap();
                let result2 = TobitEstimator::fit(input2, default_options(CovType::Hc1, solver));
                prop_assume!(result2.is_ok());
                let est_hc1 = result2.unwrap();

                for j in 0..=(k + 1) {
                    prop_assert!(
                        est_hc0.std_errors()[j] <= est_hc1.std_errors()[j] + 1e-9,
                        "HC0 se[{j}]={} should be <= HC1 se[{j}]={}",
                        est_hc0.std_errors()[j], est_hc1.std_errors()[j]
                    );
                }
            }

            /// 鏡像対称性: `y`と境界を符号反転して`(lower, upper)`→`(-upper, -lower)`にすると
            /// （`-y* = x(-β) + σ(-ε)`、`-ε`も標準正規）、係数は符号反転・`σ`と標準誤差は不変。
            /// 左打ち切りと右打ち切りで別々のコード分岐（`censored_contribution`の
            /// `direction`）を通るため、片方の分岐の符号・スケールのバグがここで顕在化する。
            #[test]
            fn negating_y_and_swapping_bounds_negates_coefficients_only(
                (n, k, x_cols, beta, sigma, u, _keys) in tobit_case_strategy(),
                bounds in censoring_strategy(),
                solver in solver_strategy(),
            ) {
                let y = simulate_y(n, &x_cols, &beta, sigma, &u, bounds);
                let names = x_names(k);
                let input1 = TobitInput::from_columns(
                    &y, &x_cols, names.clone(), true, "y".to_string(), bounds.0, bounds.1,
                ).unwrap();
                let result1 = TobitEstimator::fit(input1, default_options(CovType::Classical, solver));
                prop_assume!(result1.is_ok());
                let est1 = result1.unwrap();

                let y_neg: Vec<f64> = y.iter().map(|v| -v).collect();
                let input2 = TobitInput::from_columns(
                    &y_neg, &x_cols, names, true, "y".to_string(),
                    bounds.1.map(|v| -v), bounds.0.map(|v| -v),
                ).unwrap();
                let result2 = TobitEstimator::fit(input2, default_options(CovType::Classical, solver));
                prop_assume!(result2.is_ok());
                let est2 = result2.unwrap();

                for j in 0..=k {
                    assert_approx_eq(est2.params()[j], -est1.params()[j], &format!("param[{j}] under y negation"));
                }
                assert_approx_eq(est2.sigma(), est1.sigma(), "sigma under y negation");
                for j in 0..=(k + 1) {
                    assert_approx_eq(est2.std_errors()[j], est1.std_errors()[j], &format!("std_error[{j}] under y negation"));
                }
            }
        }

        /// （Phase 2、実際の退化ケースの特定）で捕捉した具体的な
        /// 入力の1つを固定値化した回帰テスト。
        ///
        /// **調査方法**: Phase 1（評価回数バジェット方式）導入後は、line
        /// searchの暴走が数秒程度の有限時間で`Err`に変換されるようになったため、
        /// `kill`前提の外部タイムアウト無しで大量試行を安全に回せることを利用し、
        /// `tobit_case_strategy()`・`solver_strategy()`を`TestRunner`で直接
        /// サンプリングして30万回試行する使い捨ての探索ハーネスを一時的に実装し
        /// 実行した（コミット履歴には残さず、本テストのみを結果として残す）。
        ///
        /// **判明した事実（当時）**: 30万試行中8件が`MleError::EvaluationBudgetExceeded`
        /// 相当のエラー（`SolverType::Lbfgs`は当時のargmin組み込みLBFGSの`SolverExit`経由で
        /// 同じメッセージを含む`ComputationFailed`になっていた）を引き起こした。**8件
        /// 全てが`solver=Lbfgs`**で、`solver=Bfgs`・`solver=Newton`は1件もヒットしな
        /// かった。打ち切り割合（`n_uncensored/n`）は0.38〜0.80、`n`は41〜75、`k`は
        /// 1〜3、`sigma`は0.53〜1.85とヒット全体に幅広く分布しており、「極端な打ち切り・
        /// 特定の狭いパラメータ域が必要」という仮説は支持されなかった（8件中7件は
        /// 0.65〜0.80という中程度の打ち切り）。真因は当時未確定で、argmin組み込み
        /// `LBFGS`固有の何か（limited-memory two-loop recursionの初期スケーリング等）に
        /// 起因すると推測されていたが、真因の特定は後続の調査に持ち越された。
        ///
        /// **判明した実際の原因と解消（本テストの現在の主眼）**: `SolverType::Lbfgs`
        /// を`FaerLbfgs`（`FaerBfgs`と同型のself-scaling初期化を持つ自前実装）に置き換えた
        /// 結果、この入力は**もはや退化せず正常に収束する**（実測: `n_iter=11`、
        /// `converged=true`）。`FaerBfgs`で既に確立していた「1回目の
        /// 反復専用のline search初期ステップ幅`min(1,1/‖g₀‖)`」を、当時のargmin組み込み
        /// `LBFGS`の公開APIでは適用できていなかったこと（1回目の反復は履歴が空で
        /// `γ=1.0`固定のため、単位行列の逆Hessianを使う`FaerBfgs`の1回目と同型の
        /// 暴走リスクを抱えたままだった）が実際の原因だったと裏付けられた。本テストは
        /// 「境界の有限時間`Err`」から「実際に収束する」への挙動変化そのものを固定する
        /// 回帰ガードとして残す。
        #[test]
        fn fit_lbfgs_converges_for_a_previously_stalling_case_from_issue_344() {
            let beta = vec![0.9873339499036579, -0.12291865368107788];
            let sigma = 1.4304964761886263;
            let x_cols: Vec<Vec<f64>> = vec![vec![
                0.09730587261408116,
                -0.44608585527013445,
                -1.1780876932682702,
                -1.7218664313287149,
                -1.5668563688839525,
                1.569383650134455,
                -0.8904079788143034,
                -0.38220417882649177,
                -0.6650396911368611,
                1.5455987143552725,
                1.0914058868351664,
                1.5608174706710314,
                -1.0457715928011817,
                0.8460357774450576,
                -1.6746201868364683,
                -0.10662833055410269,
                -1.3549266627206265,
                1.802095632400922,
                0.22454103534302314,
                0.9226514870462461,
                -1.5960509851353555,
                -0.06299762418526257,
                -0.7361961389207361,
                -0.6159276360973034,
                1.6276092545963188,
                1.8777781260428934,
                1.4040046871028768,
                1.6191183983983046,
                0.7928232369566655,
                -1.2215837625042971,
                0.43098012504668753,
                0.6437939214254764,
                1.8726228041330524,
                0.18531824642137915,
                -1.2889799350480626,
                1.3893942618908828,
                0.9546063410017254,
                1.6709682203826888,
                -0.6385965449169932,
                1.801002621573055,
                1.89685718642964,
            ]];
            let u = vec![
                0.3617740489249382,
                0.8638655369835531,
                0.5854529043833009,
                0.17096769280771643,
                0.27549992513298815,
                0.29109304112930307,
                0.23572824129100706,
                0.6413758528582494,
                0.015031380540545333,
                0.9591408659161148,
                0.03755478548383856,
                0.5531593343087735,
                0.00290634638539887,
                0.6500237097120347,
                0.41895952824744176,
                0.6246296073538906,
                0.672262276485462,
                0.072963646484062,
                0.8961814455371079,
                0.37114140842146587,
                0.07134975672937831,
                0.42084724899948256,
                0.612656348188821,
                0.03682767217809841,
                0.7090082364949691,
                0.30669875984016237,
                0.9155739980338565,
                0.022268595803044702,
                0.9400939907877396,
                0.922790135057218,
                0.05561252335283864,
                0.9567466460014393,
                0.1117763895368753,
                0.16362361406720183,
                0.41560137699383287,
                0.08196797976110878,
                0.25881610622077167,
                0.33012720695491854,
                0.6634215047000254,
                0.23962574010778723,
                0.9444306832294833,
            ];
            let n = 41;

            let y = simulate_y(n, &x_cols, &beta, sigma, &u, (Some(LOWER_BOUND), None));
            let names = x_names(1);
            let input = TobitInput::from_columns(
                &y,
                &x_cols,
                names,
                true,
                "y".to_string(),
                Some(LOWER_BOUND),
                None,
            )
            .unwrap();

            let start = std::time::Instant::now();
            let result = TobitEstimator::fit(
                input,
                default_options(CovType::Classical, SolverType::Lbfgs),
            );
            let elapsed = start.elapsed();

            assert!(
                result.is_ok(),
                "expected FaerLbfgs's self-scaling to converge on this input, got {result:?}"
            );
            assert!(
                elapsed < std::time::Duration::from_secs(30),
                "fit() took {elapsed:?}, expected a bounded run within seconds"
            );
            // 実測`n_iter=11`（緩い上限、rust-reviewer指摘）。`FaerLbfgs`の1回目line
            // search初期ステップ幅の設計変更（`engine/src/nonlinear/CLAUDE.md`
            // 「実装時に踏んだ罠その2」参照）時に、収束はするが反復回数が
            // 大きく増える形の劣化が紛れ込んでいないかを検知するためのガード。
            let n_iter = result.unwrap().n_iter();
            assert!(
                n_iter < 30,
                "n_iter={n_iter}, expected roughly the same order as the observed 11 \
                 iterations (a large increase would indicate a regression in FaerLbfgs's \
                 first-iteration self-scaling, even though it still converges)"
            );
        }

        // ── ExpectedUncensored（`E[y|lower<y<upper,x]`）の不変条件 ──────────
        // 以降は`fit()`を介さず`predicted_value`・`target_w_and_s`を直接検証する。
        // 乱数の`(μ, σ, lower, upper)`は、非打ち切り区間の幅/σが
        // `MIN_UNCENSORED_WIDTH_RATIO`以上になる範囲に限る（それ未満は`TobitEstimator`が
        // `UncensoredIntervalTooNarrow`で拒否する）。`μ`は`±60`まで振って`z`が数十に
        // なる裾（連分数・鏡像変換の分岐）も踏む。

        /// `(mu, sigma, lower, upper)`。`lower`/`upper`は片側・両側のいずれかで、
        /// 両側のときは幅が`σ`の`1/100`以上。
        type UncensoredCase = (f64, f64, Option<f64>, Option<f64>);

        fn uncensored_case_strategy() -> impl Strategy<Value = UncensoredCase> {
            (
                -60.0f64..60.0,
                0.2f64..5.0,
                -3.0f64..3.0,
                0.0f64..8.0,
                0u8..3,
            )
                .prop_map(|(mu, sigma, lower, width, kind)| {
                    // 幅の下限: `δ=width/σ>=0.01`を満たすよう`σ/100`以上にする
                    let width = width + sigma / 100.0;
                    let (lo, up) = match kind {
                        0 => (Some(lower), None),
                        1 => (None, Some(lower)),
                        _ => (Some(lower), Some(lower + width)),
                    };
                    (mu, sigma, lo, up)
                })
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(512))]

            /// 分解恒等式: `E[y|x] = lower·P(下限) + upper·P(上限) + P(uncensored|x)·E[y|uncensored,x]`
            /// （下限0のみの左打ち切りなら`E[y|x]=P(y>0|x)·E[y|y>0,x]`）。既存の`ExpectedObserved`・
            /// `ProbUncensored`（`statrs`の`cdf`、絶対誤差`1e-11`級）と新`ExpectedUncensored`
            /// （別の安定実装）が、乱数の`(μ,σ,境界)`全域で整合する。裾（`z`が数十）では
            /// `P(uncensored)≈0`なので右辺は`lower`/`upper`の項が支配する。
            #[test]
            fn decomposition_identity_holds_for_random_inputs(
                (mu, sigma, lower, upper) in uncensored_case_strategy(),
            ) {
                let normal = Normal::standard();
                let value = |t| predicted_value(t, mu, sigma, lower, upper, &normal);
                let p_lower = lower.map_or(0.0, |l| normal.cdf((l - mu) / sigma));
                let p_upper = upper.map_or(0.0, |u| 1.0 - normal.cdf((u - mu) / sigma));
                let decomposed = lower.unwrap_or(0.0) * p_lower
                    + upper.unwrap_or(0.0) * p_upper
                    + value(MarginalEffectsTarget::ProbUncensored)
                        * value(MarginalEffectsTarget::ExpectedUncensored);
                let observed = value(MarginalEffectsTarget::ExpectedObserved);
                prop_assert!(
                    (observed - decomposed).abs() <= 1e-8 * (1.0 + observed.abs()),
                    "observed={observed}, decomposed={decomposed}"
                );
            }

            /// 値域: `E[y|lower<y<upper,x]`は非打ち切り領域`[lower, upper]`に収まり
            /// （境界は丸め誤差分だけ許す）、重み`w`（切断分散/σ²）は`[0,1]`、`s_beta`・
            /// `s_sigma`は有限。裾（`z`が数十）でも破綻しない。
            #[test]
            fn expected_uncensored_stays_within_bounds_and_w_in_unit_interval(
                (mu, sigma, lower, upper) in uncensored_case_strategy(),
            ) {
                let normal = Normal::standard();
                let m = predicted_value(
                    MarginalEffectsTarget::ExpectedUncensored, mu, sigma, lower, upper, &normal,
                );
                let (w, s_beta, s_sigma) = target_w_and_s(
                    MarginalEffectsTarget::ExpectedUncensored,
                    &[1.0], &[mu], sigma, lower, upper, &normal,
                );
                prop_assert!(m.is_finite() && w.is_finite() && s_beta[0].is_finite() && s_sigma.is_finite());
                prop_assert!((0.0..=1.0).contains(&w), "w={w}");
                let eps = 1e-9 * (1.0 + m.abs());
                if let Some(l) = lower {
                    prop_assert!(m >= l - eps, "m={m} < lower={l}");
                }
                if let Some(u) = upper {
                    prop_assert!(m <= u + eps, "m={m} > upper={u}");
                }
            }

            /// 鏡像対称性: `(μ, lower, upper)→(-μ, -upper, -lower)`で、平均`m`は符号反転、
            /// 重み`w`は不変、`∂w/∂μ`は符号反転、`∂w/∂σ`は不変。左のみ↔右のみに入れ替わり、
            /// 両側の上側の裾（`za>=0`）↔下側の裾（`zb<=0`）が別の分岐（符号反転した
            /// `upper_tail_pdf_ratios`）を通るため、片方の分岐の符号・入れ替え（`Pa↔Pb`・`A↔B`）の
            /// バグを検出できる。
            #[test]
            fn mirror_symmetry_negates_mean_and_derivative_and_keeps_weight(
                (mu, sigma, lower, upper) in uncensored_case_strategy(),
            ) {
                let normal = Normal::standard();
                let (mirror_lower, mirror_upper) = (upper.map(|u| -u), lower.map(|l| -l));
                let m = predicted_value(
                    MarginalEffectsTarget::ExpectedUncensored, mu, sigma, lower, upper, &normal,
                );
                let m_mirror = predicted_value(
                    MarginalEffectsTarget::ExpectedUncensored, -mu, sigma, mirror_lower, mirror_upper, &normal,
                );
                let (w, s_beta, s_sigma) = target_w_and_s(
                    MarginalEffectsTarget::ExpectedUncensored, &[1.0], &[mu], sigma, lower, upper, &normal,
                );
                let (w_mirror, s_beta_mirror, s_sigma_mirror) = target_w_and_s(
                    MarginalEffectsTarget::ExpectedUncensored, &[1.0], &[-mu], sigma, mirror_lower, mirror_upper, &normal,
                );
                // `z`が数十の裾では`ε·z`〜`ε·z²`の丸め誤差があるので`z`に応じて許容を広げる
                let z_max = [lower, upper]
                    .iter()
                    .flatten()
                    .map(|b| ((b - mu) / sigma).abs())
                    .fold(0.0f64, f64::max);
                let tol = 1e-9 * (1.0 + z_max * z_max);
                prop_assert!((m + m_mirror).abs() <= 1e-9 * (1.0 + m.abs() + mu.abs()), "m={m}, mirror={m_mirror}");
                prop_assert!((w - w_mirror).abs() <= tol, "w={w}, mirror={w_mirror}");
                prop_assert!((s_beta[0] + s_beta_mirror[0]).abs() <= tol, "s_beta={}, mirror={}", s_beta[0], s_beta_mirror[0]);
                prop_assert!((s_sigma - s_sigma_mirror).abs() <= tol, "s_sigma={s_sigma}, mirror={s_sigma_mirror}");
            }
        }

        /// `censored_hessian_weight_is_non_negative_at_extreme_params`が生成するケース:
        /// `(n, k, x_cols, y_raw, censored_flags, bounds, intercept, slopes, s, v)`。
        /// `intercept`・`slopes`は`(符号, 10の指数)`、`s`は`logσ`、`v`は二次形式の方向。
        type ExtremeCase = (
            usize,
            usize,
            Vec<Vec<f64>>,
            Vec<f64>,
            Vec<bool>,
            Bounds,
            f64,
            Vec<(bool, f64)>,
            f64,
            Vec<f64>,
        );

        /// `fit()`を通さずHessianを直接評価する点のケース。`tobit_case_strategy`は
        /// `|zeta|`が`U_CLAMP≈8.13`に届くことがほとんどないため、`σ=e^s`（`s∈[-1,1]`）に
        /// 対して切片を`|β₀|∈[10^1.5, 1000]`で振り、0行目の`|zeta|=|β₀|/σ`が`U_CLAMP`を
        /// 超えるようにする。
        fn extreme_param_case_strategy() -> impl Strategy<Value = ExtremeCase> {
            (1..=MAX_K).prop_flat_map(|k| {
                (k + 5..=60usize).prop_flat_map(move |n| {
                    (
                        Just(n),
                        Just(k),
                        collection::vec(collection::vec(-2.0f64..2.0, n), k),
                        collection::vec(0.05f64..0.9, n),
                        collection::vec(any::<bool>(), n),
                        censoring_strategy(),
                        1.5f64..3.0,
                        collection::vec((any::<bool>(), 0.0f64..3.0), k),
                        -1.0f64..1.0,
                        collection::vec(-1.0f64..1.0, k + 1),
                    )
                })
            })
        }

        fn signed_power_of_ten((positive, exponent): (bool, f64)) -> f64 {
            let magnitude = 10f64.powf(exponent);
            if positive { magnitude } else { -magnitude }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            /// 任意のパラメータ点で、打ち切り観測の`A(u)=λ(λ+u)`（`h_beta_coef`）が非負で、
            /// Hessianの`β`ブロック（`A(u)`と非打ち切りの`1`を係数とする`Σ coef·xx'/σ²`）が
            /// 半正定値になる。`λ`だけクランプして生の`zeta`を混ぜると`|zeta|>U_CLAMP`の
            /// 領域で`A(u)`が負になる回帰を捕まえる（Probitの`hessian_weight_is_non_negative_
            /// at_extreme_params`と対）。`(β, logσ)`全体のHessianは大域凹でないため、符号の性質
            /// （非負・半正定値）は`β`ブロックだけに表明する。`β`-`s`・`s`-`s`ブロックは符号の
            /// 性質が無いので、代わりに(1)0行目の寄与がクランプ境界`zeta=-U_CLAMP`での閉形式
            /// （`A=m(m-U)`、`c=-U·A-m`、`h_beta_s=direction·c`、`h_ss=-U·c`、`m=φ/Φ`）に
            /// 一致すること、(2)`hessian()`が`Contribution`の和として組み立てられていること
            /// （全ブロック）を確認する。保証するのは「クランプ後の関数としての性質」で、クランプ
            /// しない真のHessianとの一致ではない（`score_s`は生の`zeta`を使うため、クランプ領域で
            /// `gradient`とHessianが非整合な既知の課題は別）。
            ///
            /// 0行目は説明変数を全て0にして`xb=β₀`とし、`β₀`の符号を打ち切り方向と逆に置く
            /// （左・両側は下限で打ち切り、`β₀>0`で`zeta=(l-β₀)/σ<-U_CLAMP`。右は
            /// `β₀<0`で`zeta=(β₀-u)/σ<-U_CLAMP`）ので、どのケースでもクランプ領域の打ち切り
            /// 観測を少なくとも1件含む。1行目は非打ち切りにして、打ち切りと非打ち切りの両方を含める。
            ///
            /// 有効性確認済み: `censored_contribution`の`zeta_for_hessian`を生の`zeta`に戻すと、
            /// 左・右どちらの分岐だけを戻しても失敗する（通常の`tobit_case_strategy`は`|zeta|`が
            /// `U_CLAMP`に届かず検出できない）。
            #[test]
            fn censored_hessian_weight_is_non_negative_at_extreme_params(
                (n, k, mut x_cols, y_raw, censored, bounds, intercept_exp, slopes, s, v)
                    in extreme_param_case_strategy(),
            ) {
                for col in &mut x_cols {
                    col[0] = 0.0;
                }
                let (lower, upper) = bounds;
                let beta0 = signed_power_of_ten((lower.is_some(), intercept_exp));
                let mut params = vec![beta0];
                params.extend(slopes.iter().map(|&slope| signed_power_of_ten(slope)));
                params.push(s);

                // 観測の値: 打ち切りなら境界値、そうでなければ区間内部の値。0行目は必ず
                // 打ち切り、1行目は必ず非打ち切り。0行目の境界は`β₀`の符号と逆側にそろえる
                // （下限があれば下限、なければ上限）。
                let y: Vec<f64> = (0..n)
                    .map(|i| {
                        let r = y_raw[i];
                        let is_censored = i == 0 || (i != 1 && censored[i]);
                        match (lower, upper) {
                            (Some(l), None) => if is_censored { l } else { l + r },
                            (None, Some(u)) => if is_censored { u } else { u - r },
                            (Some(l), Some(_)) if i == 0 => l,
                            (Some(l), Some(u)) => {
                                if !is_censored { l + (u - l) * r } else if r > 0.5 { u } else { l }
                            }
                            (None, None) => unreachable!("censoring_strategy always sets a bound"),
                        }
                    })
                    .collect();

                let input = TobitInput::from_columns(&y, &x_cols, x_names(k), true, "y".to_string(), lower, upper).unwrap();
                let problem = TobitProblem::new(&input);
                let normal = Normal::standard();
                let sigma = s.exp();

                let zeta0 = match lower {
                    Some(l) => (l - params[0]) / sigma,
                    None => (params[0] - upper.unwrap_or(0.0)) / sigma,
                };
                prop_assert!(zeta0 < -U_CLAMP, "row 0 must be censored beyond U_CLAMP, got zeta={zeta0}");

                let contributions: Vec<Contribution> =
                    (0..n).map(|i| problem.contribution(i, &params, &normal)).collect();
                for (i, c) in contributions.iter().enumerate() {
                    prop_assert!(c.h_beta_coef >= 0.0, "A(u)[{i}]={} should be non-negative", c.h_beta_coef);
                }
                // 0行目はクランプ境界`zeta=-U_CLAMP`と同じ寄与（0行目は`lower`があれば左、なければ右）
                let (phi, big_phi) = clamped_pdf_cdf(&normal, -U_CLAMP);
                let m = phi / big_phi;
                let a0 = m * (m - U_CLAMP);
                let c0 = -U_CLAMP * a0 - m;
                let direction = if lower.is_some() { 1.0 } else { -1.0 };
                let close = |actual: f64, expected: f64| (actual - expected).abs() <= 1e-9 * (1.0 + expected.abs());
                prop_assert!(close(contributions[0].h_beta_coef, a0), "A[0]={} vs {a0}", contributions[0].h_beta_coef);
                prop_assert!(close(contributions[0].h_beta_s_coef, direction * c0), "h_beta_s[0]={} vs {}", contributions[0].h_beta_s_coef, direction * c0);
                prop_assert!(close(contributions[0].h_ss, -U_CLAMP * c0), "h_ss[0]={} vs {}", contributions[0].h_ss, -U_CLAMP * c0);

                // `hessian()`は`Contribution`の和: 全ブロックを別ループで組み立て直して比べる
                let hessian = problem.hessian(&params).unwrap();
                let kb = k + 1; // 切片を含む`β`の本数
                let sigma2 = sigma * sigma;
                let scale: f64 = 1.0 + (0..=kb).map(|a| hessian[a][a].abs()).sum::<f64>();
                for (a, row) in hessian.iter().enumerate().take(kb) {
                    for (b, cell) in row.iter().enumerate().take(kb) {
                        let expected: f64 = (0..n)
                            .map(|i| contributions[i].h_beta_coef * problem.x.get(i, a) * problem.x.get(i, b) / sigma2)
                            .sum();
                        prop_assert!((cell - expected).abs() <= 1e-9 * scale, "H[{a}][{b}]={cell} vs {expected}");
                    }
                    let cross: f64 = (0..n)
                        .map(|i| contributions[i].h_beta_s_coef * problem.x.get(i, a) / sigma)
                        .sum();
                    prop_assert!((hessian[a][kb] - cross).abs() <= 1e-9 * scale, "H[{a}][s]={} vs {cross}", hessian[a][kb]);
                    prop_assert!((hessian[kb][a] - cross).abs() <= 1e-9 * scale, "H[s][{a}]={} vs {cross}", hessian[kb][a]);
                }
                let h_ss: f64 = contributions.iter().map(|c| c.h_ss).sum();
                prop_assert!((hessian[kb][kb] - h_ss).abs() <= 1e-9 * scale, "H[s][s]={} vs {h_ss}", hessian[kb][kb]);

                // `β`ブロックの半正定値性: 対角が非負、任意方向`v'H_ββv ≥ 0`
                for (a, row) in hessian.iter().enumerate().take(kb) {
                    prop_assert!(row[a] >= 0.0, "H[{a}][{a}]={} should be non-negative", row[a]);
                }
                let quad: f64 = (0..kb)
                    .flat_map(|a| (0..kb).map(move |b| (a, b)))
                    .map(|(a, b)| v[a] * hessian[a][b] * v[b])
                    .sum();
                prop_assert!(quad >= -1e-9 * scale, "v'H_bb v={quad} should be non-negative (scale={scale})");
            }
        }
    }
}
