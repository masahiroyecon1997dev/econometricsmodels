//! Probitの入力データ（被説明変数・設計行列）の型定義、および負の対数尤度・スコア・
//! Hessian（argminの`CostFunction`/`Gradient`/`Hessian`トレイト実装）。
//!
//! `engine`はpolars/PyO3を一切知らない（`.claude/rules/rust-style.md`「責務分離」参照）。
//! `engine_pybind`はpolars DataFrameから列ごとに`Vec<f64>`を抽出するところまでを担い
//! （`column_extraction::extract_f64_column`）、それらの列を本モジュールの
//! `ProbitInput::from_columns`に渡す。`faer::Mat`への組み立て（切片列の自動追加を含む）は
//! ここ（engine側）の責務とする。`engine::nonlinear::logit::LogitInput`とほぼ同型の設計
//! （`docs/planning/specs/nonlinear-api-design.md`参照）。
//!
//! OLS/Logitと同様、Phase2（Logit/Probit/Tobit）では`weights`/`offset`を見送っているため
//! （`nonlinear-api-design.md`7章）、`from_columns_weighted`に相当するものはない。
//!
//! ## 数式（プロビット回帰）
//!
//! `z_i = x_i'θ`、`Φ`・`φ`を標準正規分布のCDF・PDFとする。観測`i`の対数尤度への寄与は
//! `ℓ_i(θ) = y_i log Φ(z_i) + (1-y_i) log Φ(-z_i)`（`Φ(-z)=1-Φ(z)`を使用）。
//! `q_i = 2y_i-1 ∈ {-1,+1}`とおくと`ℓ_i(θ) = log Φ(q_i z_i)`という同値な形に書き換えられる
//! （`y_i=1`なら`q_i=1`で`logΦ(z_i)`、`y_i=0`なら`q_i=-1`で`logΦ(-z_i)`に一致）。
//!
//! `λ_i = q_i φ(q_i z_i)/Φ(q_i z_i)`（一般化残差、逆ミルズ比に相当）とおくと:
//!
//! - スコア: `∂ℓ/∂θ = Σᵢ λᵢxᵢ = X'λ`
//! - Hessian: `∂²ℓ/∂θ∂θ' = -Σᵢ λᵢ(λᵢ+zᵢ)xᵢxᵢ' = -X'WX`（`W = diag(λᵢ(λᵢ+zᵢ))`）
//!
//! （導出: `u=q_i z_i`とおき`g(u)=φ(u)/Φ(u)`とすると`λ_i=q_i g(u)`。
//! `g'(u) = -u g(u) - g(u)²`（`φ'(u)=-uφ(u)`より）を使うと
//! `dλ_i/dz_i = q_i² g'(u) = g'(u) = -λ_i(λ_i+z_i)`（`q_i²=1`、`u=q_i z_i`と`g(u)=q_i λ_i`を代入）。
//! Logitの`-X'WX`（`W=diag(pᵢ(1-pᵢ))`）より複雑だが、`λᵢ(λᵢ+zᵢ) > 0`は常に成り立つため
//! （プロビットの対数尤度が大域的に凹であることの根拠）、`X'WX`は常に正定値。）
//!
//! `CostFunction`は`-ℓ(θ)`（argminは最小化フレームワークのため）。`Gradient`/`Hessian`
//! トレイトは`CostFunction`と同じ符号（`-ℓ`の1階・2階微分）で実装する
//! （`run_solver`のdocコメント「`Hessian`トレイトの符号規約」参照）。`scores()`
//! （`cov_type`共通行列演算向け）は符号反転しない生のスコア`sᵢ=λᵢxᵢ`を返す。
//!
//! ## 数値安定化について
//!
//! `Φ(q_i z_i)`・`φ(q_i z_i)`は`statrs::distribution::Normal`（`cdf`/`pdf`）をそのまま使う。
//! `1-Φ(z)`を手動計算せず常に`Φ(q_i z_i)`の形（`q_i`で符号を吸収）で評価するため、
//! `statrs`の`cdf`実装（`erfc`ベース）が両裾で提供する精度をそのまま活かせる
//! （手動で`1.0 - cdf(z)`を計算する場合に生じる桁落ちを避けられる）。
//!
//! `λ_i = φ(u)/Φ(u)`（`u=q_i z_i`）は、`u`が極端に負（実測`|u|≳39`）だと`φ(u)`・`Φ(u)`が
//! ともに0にアンダーフローし`0.0/0.0`のNaNになる（Issue #71実装時にrust-reviewerの
//! レビューで判明。`Logit`の`logistic`/`softplus`が有限の`z`ではどれだけ極端でも
//! 絶対にNaNを産まない設計だったのとは異なる、Probit固有のリスク）。既定手法の
//! `Method::Newton`（`FaerNewton`）はline searchなしで`gradient`/`hessian`を直接
//! 使うため、Logitで実際に問題になった「(準)完全分離データでの収束判定誤検知」
//! （勾配ノルムのアンダーフロー、`nonlinear-implementation-notes.md`参照）よりも
//! 緩い条件でこのNaN汚染に到達しうる（Issue #72のコメント参照）。
//!
//! 対策として、`u`を`φ`/`Φ`評価前に`[-U_CLAMP, U_CLAMP]`にクランプする
//! （`U_CLAMP`のdocコメント参照）。R言語`stats::binomial(link="probit")`の
//! `linkinv`が線形予測子を`pnorm`評価前に同じ閾値でクランプする実装
//! （`thresh <- -qnorm(.Machine$double.eps); eta <- pmin(pmax(eta, -thresh), thresh)`）
//! を参考にした（ユーザー確認済み）。statsmodelsの`Probit`は`Φ`の**出力**を
//! `np.clip(cdf, FLOAT_EPS, 1-FLOAT_EPS)`でクリップする方式だが、`score`/`loglike`
//! にのみ適用され`hessian`には適用されていない（非対称）。今回は`u`（入力側）を
//! クランプする方式を採用し、`cost`/`gradient`/`hessian`/`scores`すべてに同じ
//! 関所（`clamped_pdf_cdf`）を経由させることでこの非対称性を避けている。

use crate::error::CommonError;
use crate::inference;
use crate::nonlinear::common::{
    CovType, FittedModelForMarginalEffects, GoodnessOfFit, MarginalEffects, MarginalEffectsAt,
    Method, MleError, SandwichVariant, cluster_cov_params, column_means, column_medians,
    destandardize_cov_params, destandardize_params, goodness_of_fit, log_likelihood_null,
    marginal_effects_from_w_s, observed_information_cov_params, opg_cov_params, pred_table,
    run_solver, sandwich_cov_params, standardize_columns, validate_binary_y,
};
use crate::validation::validate_cluster_groups;
use argmin::core::{CostFunction, Error as OptimizerError, Gradient, Hessian};
use faer::Mat;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

/// Probitの被説明変数・設計行列を保持する入力データ。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
/// `from_columns`で組み立てた後は、getter経由でのみアクセスする。
#[derive(Debug)]
pub struct ProbitInput {
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
}

impl ProbitInput {
    /// 列ごとの`Vec<f64>`（`engine_pybind`がpolars DataFrameから抽出済み）から
    /// `ProbitInput`を組み立てる。`include_intercept=true`の場合、設計行列の先頭列に
    /// 定数項（すべて1.0）を自動追加する。
    ///
    /// `y`が{0.0, 1.0}の二値であることの検証は、この関数自体では行わない
    /// （`LogitInput::from_columns`と同じ方針）。`ProbitEstimator::fit`冒頭で
    /// `validate_binary_y`（`nonlinear/common.rs`）により検証される。
    ///
    /// # Errors
    /// `y`といずれかの`x_columns`の長さが一致しない場合は`CommonError::DimensionMismatch`を返す。
    ///
    /// # パニックについて
    /// `x_names.len() != x_columns.len()`の場合は`debug_assert!`でパニックする。これは
    /// 呼び出し側（`engine_pybind`）の実装バグでしか起こり得ない内部契約であり、
    /// 実データに起因する`ValidationError`とは性質が異なるため区別している
    /// （`LogitInput::from_columns`と同じ方針）。
    pub fn from_columns(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        include_intercept: bool,
        dep_var_name: String,
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

/// `φ(u)`・`Φ(u)`を評価する前に`u`をこの絶対値以下にクランプする閾値。`u`がこれより
/// 極端になると`λ_i=φ(u)/Φ(u)`（一般化残差）が`0.0/0.0`のNaNになりうる
/// （実測では`|u|≳39`から発生。本閾値`≈8.126`はそれよりずっと手前で安全に倒す、
/// モジュール冒頭「数値安定化について」参照）。
///
/// R言語`stats::binomial(link="probit")$linkinv`の`thresh <- -qnorm(.Machine$double.eps)`
/// と同じ値（`-Φ⁻¹(f64::EPSILON)`）。`Normal::inverse_cdf`は反復計算のためホットパスで
/// 毎回呼ぶのを避け、コンパイル時定数としてハードコードしている（Rとscipyの両方で
/// `8.125890664701908`と算出されることを確認済み）。
const U_CLAMP: f64 = 8.125_890_664_701_908;

/// `u`を`[-U_CLAMP, U_CLAMP]`にクランプしてから`(φ(u), Φ(u))`を評価する
/// （`U_CLAMP`のdocコメント参照）。`log_likelihood`（`cost`はこれを符号反転して呼ぶ）・
/// `linear_predictor_and_residual`（`gradient`/`hessian`/`scores`が経由する）の両方が
/// 経由する共通の関所にすることで、`statsmodels`の`Probit`実装に見られる非対称性
/// （`score`/`loglike`はクリップするが`hessian`はしない）を避ける。
fn clamped_pdf_cdf(normal: &Normal, u: f64) -> (f64, f64) {
    let u = u.clamp(-U_CLAMP, U_CLAMP);
    (normal.pdf(u), normal.cdf(u))
}

/// 対数尤度 `ℓ(θ) = Σᵢ log Φ(qᵢzᵢ)`（`zᵢ=xᵢ'θ`、`qᵢ=2yᵢ-1`、モジュール冒頭の数式参照）を
/// `x`・`y`・`params`から直接計算する。`ProbitProblem::cost`（`-ℓ(θ)`、argminの
/// `CostFunction`）と同じ数式のΣ部分を共有する（`cost`はこの関数を符号反転して呼ぶ）。
/// argminのトレイトが要求する`Result`型を経由する必要が無い内部専用の計算
/// （適合度統計量向け、収束後のパラメータで1回だけ評価する）のため、独立した
/// 関数として切り出している（`LogitProblem`の`log_likelihood`と同じ位置づけ）。
fn log_likelihood(x: &Mat<f64>, y: &Mat<f64>, params: &[f64]) -> f64 {
    let normal = Normal::standard();
    let n = x.nrows();
    (0..n)
        .map(|i| {
            let z: f64 = (0..x.ncols()).map(|j| *x.get(i, j) * params[j]).sum();
            let q = 2.0 * (*y.get(i, 0)) - 1.0;
            let (_, big_phi) = clamped_pdf_cdf(&normal, q * z);
            big_phi.ln()
        })
        .sum()
}

/// 限界効果（`ProbitEstimator::marginal_effects`のdocコメント「数式（デルタ法）」参照）の
/// `at="overall"`（AME）における`w=(1/n)Σᵢφ(zᵢ)`・`s_m=(1/n)Σᵢ(-zᵢ)φ(zᵢ)xᵢₘ`を
/// 全観測を1回走査して計算する（`φ'(z)=-zφ(z)`、標準正規PDFの導関数。Logitの
/// `overall_w_and_s`の`w=p(1-p)`・`s_m=(1-2p)p(1-p)xᵢₘ`に相当するProbit版）。
///
/// `U_CLAMP`によるクランプは行わない: `φ(z)`単体は`z`が極端でも滑らかに0へ収束するのみで、
/// `λ=φ/Φ`のような`0.0/0.0`のNaN化リスクが無いため（モジュール冒頭「数値安定化について」
/// 参照、クランプが必要なのは`Φ`で割る箇所のみ）。
///
/// `z`が有限であることを暗黙の前提にしている（`coef=-z*φ(z)`は`z=±∞`だと`0*∞`のNaNに
/// なりうる）。`fit()`を通過した収束済み・有限の`θ`と、`engine_pybind`側で既に有限性が
/// 保証された`x`から計算する`z`は実質常に有限のため到達不能だが、`logistic`/`softplus`
/// （Logit）が「有限の`z`では絶対にNaNを生まない」設計だったのとは異なる前提であることに
/// 注意（rust-reviewer指摘、Issue #78）。
fn overall_w_and_s(x: &Mat<f64>, params: &[f64]) -> (f64, Vec<f64>) {
    let n = x.nrows();
    let k = x.ncols();
    let normal = Normal::standard();
    let mut w = 0.0;
    let mut s = vec![0.0; k];
    for i in 0..n {
        let z: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
        let phi = normal.pdf(z);
        w += phi;
        let coef = -z * phi;
        for (m, s_m) in s.iter_mut().enumerate() {
            *s_m += coef * (*x.get(i, m));
        }
    }
    let n_f = n as f64;
    w /= n_f;
    for s_m in s.iter_mut() {
        *s_m /= n_f;
    }
    (w, s)
}

/// 限界効果の`at="mean"`/`"median"`における、代表点`x_bar`（各説明変数の標本平均または
/// 中央値）で評価した`w=φ(z̄)`・`s_m=-z̄φ(z̄)x̄ₘ`（`z̄=x̄'θ`）を計算する
/// （Logitの`at_point_w_and_s`に相当するProbit版）。
fn at_point_w_and_s(x_bar: &[f64], params: &[f64]) -> (f64, Vec<f64>) {
    let k = x_bar.len();
    let z: f64 = (0..k).map(|m| x_bar[m] * params[m]).sum();
    let phi = Normal::standard().pdf(z);
    let coef = -z * phi;
    let s: Vec<f64> = (0..k).map(|m| coef * x_bar[m]).collect();
    (phi, s)
}

/// Probitの負の対数尤度・スコア・Hessian（argminの`CostFunction`/`Gradient`/`Hessian`
/// トレイト実装）。`ProbitInput`の`X`・`y`を保持する（`run_solver`が`problem`の所有権を
/// 必要とするため、`ProbitInput`とは独立した所有データとして持つ。`Clone`は
/// `argmin::core::Executor`が要求する。`LogitProblem`と同じ設計）。
#[derive(Debug, Clone)]
pub struct ProbitProblem {
    x: Mat<f64>,
    y: Mat<f64>,
}

impl ProbitProblem {
    /// `input`の`x`・`y`をそのまま（未標準化のスケールで）複製して構築する。
    /// 閉じた形の解と突き合わせる単体テスト専用（`LogitProblem::new`と同じ位置づけ）。
    #[cfg(test)]
    fn new(input: &ProbitInput) -> Self {
        Self {
            x: input.x().clone(),
            y: input.y().clone(),
        }
    }

    /// `standardize_columns`で標準化済みの設計行列`x_std`と`y`から構築する。
    /// `ProbitEstimator::fit`が最適化・収束点でのスコア評価に使う経路
    /// （`LogitProblem::from_standardized`と同じ位置づけ）。
    fn from_standardized(x_std: Mat<f64>, y: Mat<f64>) -> Self {
        Self { x: x_std, y }
    }

    /// 観測`i`の線形予測子 `z_i = x_i'θ`。
    fn linear_predictor(&self, i: usize, params: &[f64]) -> f64 {
        (0..self.x.ncols())
            .map(|j| *self.x.get(i, j) * params[j])
            .sum()
    }

    /// 観測`i`の線形予測子`z_i`と一般化残差`λ_i = q_i φ(q_i z_i)/Φ(q_i z_i)`
    /// （`q_i=2y_i-1`、モジュール冒頭の数式参照）をまとめて計算する。`hessian`が
    /// `z_i`・`λ_i`の両方を必要とするため、個別に呼び出すより重複計算を避けられる。
    /// `normal`は呼び出し側（観測`n`件のループ全体）で1回だけ構築して渡す
    /// （`cost`/`gradient`/`hessian`いずれもn回ではなく1回の構築で済ませる）。
    fn linear_predictor_and_residual(
        &self,
        i: usize,
        params: &[f64],
        normal: &Normal,
    ) -> (f64, f64) {
        let z = self.linear_predictor(i, params);
        let q = 2.0 * (*self.y.get(i, 0)) - 1.0;
        let (phi, big_phi) = clamped_pdf_cdf(normal, q * z);
        let lambda = q * phi / big_phi;
        (z, lambda)
    }

    /// 観測ごとのスコア行列（n×k）。各行が`sᵢ = λᵢxᵢ`（対数尤度の1階微分そのもの、
    /// `Gradient`トレイトとは符号が逆）。OPG/サンドイッチ/クラスターSEの計算に使う
    /// （`LogitProblem::scores`と同じ役割）。
    pub fn scores(&self, params: &[f64]) -> Mat<f64> {
        let n = self.x.nrows();
        let k = self.x.ncols();
        let normal = Normal::standard();
        Mat::from_fn(n, k, |i, j| {
            let (_, lambda) = self.linear_predictor_and_residual(i, params, &normal);
            lambda * (*self.x.get(i, j))
        })
    }
}

impl CostFunction for ProbitProblem {
    type Param = Vec<f64>;
    type Output = f64;

    /// 負の対数尤度 `-ℓ(θ) = -Σᵢ log Φ(q_i z_i)`（モジュール冒頭の数式参照）。
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, OptimizerError> {
        Ok(-log_likelihood(&self.x, &self.y, param))
    }
}

impl Gradient for ProbitProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    /// `-ℓ(θ)`の勾配 `-Σᵢ λᵢxᵢ = -X'λ`（対数尤度のスコアの符号反転）。
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, OptimizerError> {
        let n = self.x.nrows();
        let k = self.x.ncols();
        let normal = Normal::standard();
        let mut grad = vec![0.0; k];
        for i in 0..n {
            let (_, lambda) = self.linear_predictor_and_residual(i, param, &normal);
            for (j, grad_j) in grad.iter_mut().enumerate() {
                *grad_j += -lambda * (*self.x.get(i, j));
            }
        }
        Ok(grad)
    }
}

impl Hessian for ProbitProblem {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    /// `-ℓ(θ)`のHessian `X'WX`（`W = diag(λᵢ(λᵢ+zᵢ))`、対数尤度のHessian`-X'WX`の
    /// 符号反転）。`run_solver`のdocコメント「`Hessian`トレイトの符号規約」参照。
    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
        let n = self.x.nrows();
        let k = self.x.ncols();
        let normal = Normal::standard();
        let mut h = vec![vec![0.0; k]; k];
        for i in 0..n {
            let (z, lambda) = self.linear_predictor_and_residual(i, param, &normal);
            let w = lambda * (lambda + z);
            for (a, row) in h.iter_mut().enumerate().take(k) {
                let xa = *self.x.get(i, a);
                for (b, cell) in row.iter_mut().enumerate().take(k) {
                    *cell += w * xa * (*self.x.get(i, b));
                }
            }
        }
        Ok(h)
    }
}

/// Probitの推定結果。`fit`でのバリデーション・最適化・`cov_type`に応じたSE計算・
/// 適合度統計量の計算を通過した状態を表す。
///
/// 限界効果等は未実装。`docs/planning/specs/logit-probit-issue-breakdown.md`の
/// 対応する後続Issueで`fit`とは別のメソッドとして追加していく想定。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug)]
pub struct ProbitEstimator {
    input: ProbitInput,
    /// 係数（元のスケール。`standardize_columns`で標準化した空間で最適化した後、
    /// `destandardize_params`で逆変換済み）。`input.param_names()`と対応する
    params: Vec<f64>,
    /// 係数の分散共分散行列（元のスケール、k×k）。`fit`に渡した`cov_type`に応じて
    /// 観測情報行列（`Classical`）・OPG（`Opg`）・サンドイッチ型（`Hc0`/`Hc1`）の
    /// いずれかで計算される。限界効果（デルタ法、`logit-probit-issue-breakdown.md`の
    /// 対応する後続Issue）で再利用するため、対角成分（`std_errors`）だけでなく
    /// 行列そのものを保持する。
    cov_params: Mat<f64>,
    /// 標準誤差（k, 元のスケール）。`cov_params`の対角成分の平方根
    std_errors: Vec<f64>,
    /// z統計量（k）= `params / std_errors`
    z_stats: Vec<f64>,
    /// 両側p値（k）。標準正規分布に基づく
    p_values: Vec<f64>,
    /// 信頼区間の下限（k）
    conf_lower: Vec<f64>,
    /// 信頼区間の上限（k）
    conf_upper: Vec<f64>,
    /// 収束したかどうか
    converged: bool,
    /// 実際の反復回数
    n_iter: usize,
    /// 対数尤度 `ℓ(θ̂)`（収束点で評価）
    log_likelihood: f64,
    /// 切片のみモデルの対数尤度 `ℓ(θ̂_null)`。`ȳ=n1/n`の閉じた形の解析解
    /// （`nᵢ log(ȳ) + (1-nᵢ) log(1-ȳ)`の総和）から直接計算する（ソルバーの
    /// 再フィットは経由しない。`LogitEstimator`と同じ方針）。
    /// `include_intercept`の値に関わらず常にこの「切片のみ」モデルを参照する
    /// （`nonlinear-api-design.md`5章の定義通り）。
    ///
    /// この閉じた形は「切片のみモデルのMLEは`Φ(θ̂)=ȳ`を満たす」という性質
    /// （リンク関数に依らず成り立つ、`fit_newton_converges_to_closed_form_solution_
    /// for_intercept_only_model`参照）から導かれるため、`LogitEstimator`の
    /// `log_likelihood_null`と全く同じ式になる（リンク関数がロジスティックか
    /// 標準正規分布かに依存しない）。
    ///
    /// **`include_intercept=false`のとき、この値が参照する「切片のみ」モデルは
    /// フィット対象のモデルの部分集合（入れ子）にならない**（`LogitEstimator`の
    /// 対応するdocコメントと同じ注意。`lr_statistic`が負になったり`lr_p_value`が
    /// 統計的に意味の薄い値になったりしうるが、statsmodels準拠の仕様上の挙動で
    /// ありバグではない）。
    log_likelihood_null: f64,
    /// 尤度比検定統計量 `2*(ℓ(θ̂)-ℓ(θ̂_null))`。`include_intercept=false`のときの
    /// 非入れ子性については`log_likelihood_null`のdocコメント参照
    lr_statistic: f64,
    /// 尤度比検定のp値（自由度`df_model`のカイ二乗分布、上側確率）。
    /// `df_model==0`（説明変数なし）のときはNaN（OLSの`f_p_value`と同じ扱い）
    lr_p_value: f64,
    /// McFadden疑似決定係数 `1 - ℓ(θ̂)/ℓ(θ̂_null)`
    pseudo_r_squared: f64,
    /// 赤池情報量規準 `-2ℓ(θ̂) + 2k`
    aic: f64,
    /// ベイズ情報量規準 `-2ℓ(θ̂) + ln(n)k`
    bic: f64,
    /// モデルの自由度 `k-1`（切片のみのnullモデルとのパラメータ数差。statsmodels準拠、
    /// `include_intercept`の値に関わらず常にこの式。`LogitEstimator`と同じ）
    df_model: usize,
    /// 残差自由度 `n-k`
    df_resid: usize,
}

impl ProbitEstimator {
    /// `method`（Newton-Raphson/BFGS/L-BFGS）で負の対数尤度を最小化し、Probitの係数・
    /// 観測情報行列によるSE・z値・p値・信頼区間を推定する。`LogitEstimator::fit`
    /// （Issue #56の骨格実装＋Issue #57のmethod分岐＋Issue #58のSE計算）と同じ設計・スコープ。
    ///
    /// `method`の選択に関わらず、収束点でのHessian評価（SE計算用）は常に解析的に行う
    /// （`run_solver`の実装方針、`docs/planning/specs/nonlinear-implementation-notes.md`
    /// 「engine内のtrait設計」参照）。BFGS/L-BFGSが最適化中に内部で保持する近似Hessianは
    /// 使い回さない。
    ///
    /// 初期値は常にゼロベクトル（`start_params`によるユーザー指定は未対応、
    /// `LogitEstimator::fit`と同じ理由でユーザー確認の上見送り）。
    ///
    /// 設計行列は`standardize_columns`で内部的に標準化してから最適化し、収束後の
    /// パラメータを`destandardize_params`で元のスケールへ逆変換する
    /// （`LogitEstimator::fit`のdocコメント参照）。`run_solver`が返すHessianは
    /// 標準化空間（θ_std）で評価されたものであり、分散共分散行列もいったん標準化空間で
    /// 計算してから`destandardize_cov_params`で元のスケールへ逆変換する
    /// （`destandardize_params`を先に適用してから逆算するのではなく、標準化空間の
    /// `cov_params`を直接destandardizeする。`LogitEstimator::fit`と同じ理由）。
    ///
    /// `cov_type`は観測情報行列（`Classical`）・OPG（`Opg`）・サンドイッチ型
    /// （`Hc0`/`Hc1`）・クラスターロバスト（`Cluster`）に対応する。`Cluster`の
    /// グループキー未指定・クラスター数不足は、最適化を実行する前（`fit()`冒頭）に
    /// 検証して早期に返す（`LogitEstimator::fit`と同じ、反復最適化のため無駄な計算を
    /// 避ける）。`Opg`/`Hc0`/`Hc1`/`Cluster`は収束点での観測ごとのスコア
    /// （`ProbitProblem::scores`）が必要なため、標準化空間の設計行列を保持したまま
    /// `ProbitProblem`をクローンしておき（`argmin::core::Executor`向けに元々`Clone`を
    /// 要求しているため追加コストは`Clone`実装自体のみ）、`run_solver`が返す収束点の
    /// パラメータで評価する。検定分布は標準正規分布（`nonlinear-api-design.md`5章、
    /// OLSのt分布とは異なる）。
    ///
    /// `n <= k`で`CommonError::InsufficientObservations`、`k == 0`で
    /// `CommonError::NoRegressors`を返す閾値・使い分けは`LogitEstimator::fit`と同じ
    /// （後者はIssue #118でLogitのfaer内部panic、Issue #130を修正した経緯があり、
    /// Probitでは当初から同じ検証を入れている）。
    ///
    /// `y`が`{0.0, 1.0}`の二値であることの検証（`validate_binary_y`）も、Logitでは
    /// Issue #54→#55を経てIssue #135で事後的に追加された経緯があるが、Probitでは
    /// 既に共有実装（`nonlinear/common.rs`）があるため当初から呼び出している。
    ///
    /// # Errors
    /// - `confidence_level`が`(0, 1)`の範囲外: `CommonError::InvalidConfidenceLevel`
    /// - `max_iter`が0以下: `MleError::InvalidMaxIter`
    /// - `tol`が0以下: `MleError::InvalidTol`
    /// - `y`が`{0.0, 1.0}`以外の値を含む: `MleError::InvalidBinaryY`
    /// - `k`（定数項を含む説明変数の数）が0（定数項も説明変数も無い）: `CommonError::NoRegressors`
    /// - 観測数`n`が`k`以下: `CommonError::InsufficientObservations`
    /// - `raise_on_non_convergence=true`かつ`max_iter`回で未収束: `MleError::NonConvergence`
    /// - 収束点（または`raise_on_non_convergence=false`時の打ち切り点）のHessianが特異
    ///   （設計行列の完全な多重共線性等）: `MleError::SingularHessian`
    /// - `cov_type=Opg`でOPG行列（`Σᵢ sᵢsᵢ'`）が特異: `MleError::SingularOpgMatrix`
    /// - `cov_type=Cluster`でグループキー未指定: `CommonError::MissingClusterColumn`
    /// - `cov_type=Cluster`でクラスター数が2未満: `CommonError::InsufficientClusters`
    pub fn fit(
        input: ProbitInput,
        method: Method,
        max_iter: i64,
        tol: f64,
        raise_on_non_convergence: bool,
        cov_type: CovType,
        confidence_level: f64,
    ) -> Result<Self, MleError> {
        if !(confidence_level > 0.0 && confidence_level < 1.0) {
            return Err(CommonError::InvalidConfidenceLevel { confidence_level }.into());
        }
        if max_iter <= 0 {
            return Err(MleError::InvalidMaxIter { max_iter });
        }
        if tol <= 0.0 {
            return Err(MleError::InvalidTol { tol });
        }
        validate_binary_y(input.y())?;

        let n = input.nobs();
        let k = input.k();
        if k == 0 {
            return Err(CommonError::NoRegressors { n }.into());
        }
        if n <= k {
            return Err(CommonError::InsufficientObservations { n, k }.into());
        }
        if let CovType::Cluster { groups } = &cov_type {
            let groups = groups.as_ref().ok_or(CommonError::MissingClusterColumn)?;
            validate_cluster_groups(groups, n)?;
        }

        let (x_std, scale) = standardize_columns(input.x(), input.has_intercept());
        let problem = ProbitProblem::from_standardized(x_std, input.y().clone());
        // `cov_type`がOPG/サンドイッチ型/クラスターロバストの場合、収束点でのスコア評価に
        // 元の`ProbitProblem`（標準化空間のx_std）が必要になる。`run_solver`は`problem`の
        // 所有権を取り込む（内部で保持していたモデルを呼び出し元へ返さない設計）ため、
        // 事前にクローンしておく必要がある（`ProbitProblem`は`argmin::core::Executor`
        // 向けに元々`Clone`を要求しているため、この用途のための追加のtraitではない）。
        // `Classical`はスコアを使わないため、無駄な複製（設計行列を含む）を避けるために
        // `cov_type`に応じて条件付きで行う（`LogitEstimator::fit`と同じ、rust-reviewer指摘）。
        let problem_for_scores = match &cov_type {
            CovType::Classical => None,
            CovType::Opg | CovType::Hc0 | CovType::Hc1 | CovType::Cluster { .. } => {
                Some(problem.clone())
            }
        };

        let output = run_solver(
            problem,
            method,
            vec![0.0; k],
            max_iter as u64,
            tol,
            raise_on_non_convergence,
        )?;

        let params = destandardize_params(&output.params, &scale);

        let hessian_std = Mat::from_fn(k, k, |i, j| output.hessian[i][j]);
        // `problem_for_scores.as_ref().expect(...)`は各非`Classical`分岐でのみ呼ばれ、
        // 直前の`match cov_type { CovType::Classical => None, _ => Some(...) }`により
        // 常に`Some`であることが保証されている内部契約（`LogitEstimator::fit`と同じ、
        // `expect`のメッセージで契約を明記して防御的に扱う）。
        let cov_params_std = match &cov_type {
            CovType::Classical => observed_information_cov_params(&hessian_std, k)?,
            CovType::Opg => {
                let problem = problem_for_scores
                    .as_ref()
                    .expect("problem_for_scores must be Some for CovType::Opg");
                opg_cov_params(&problem.scores(&output.params), k)?
            }
            CovType::Hc0 => {
                let problem = problem_for_scores
                    .as_ref()
                    .expect("problem_for_scores must be Some for CovType::Hc0");
                sandwich_cov_params(
                    &hessian_std,
                    &problem.scores(&output.params),
                    n,
                    k,
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
                    k,
                    SandwichVariant::Hc1,
                )?
            }
            CovType::Cluster { groups } => {
                let problem = problem_for_scores
                    .as_ref()
                    .expect("problem_for_scores must be Some for CovType::Cluster");
                // `groups`のNone・クラスター数不足の検証はfit()冒頭で完了済み
                // （MissingClusterColumn/InsufficientClustersを最適化前に早期に返す
                // ため）。ここでの`expect`はその契約を明記する防御的な扱い。
                let groups = groups
                    .as_ref()
                    .expect("groups is validated as Some at the top of fit()");
                cluster_cov_params(&hessian_std, &problem.scores(&output.params), n, k, groups)?
            }
        };
        let cov_params = destandardize_cov_params(&cov_params_std, &scale);

        let normal = Normal::standard();
        let z_crit = inference::critical_value(&normal, confidence_level);

        let mut std_errors = vec![0.0; k];
        let mut z_stats = vec![0.0; k];
        let mut p_values = vec![0.0; k];
        let mut conf_lower = vec![0.0; k];
        let mut conf_upper = vec![0.0; k];

        for j in 0..k {
            let se = (*cov_params.get(j, j)).sqrt();
            let stat = inference::compute_inference_stat(&normal, params[j], se, z_crit);

            std_errors[j] = se;
            z_stats[j] = stat.stat;
            p_values[j] = stat.p_value;
            conf_lower[j] = stat.conf_low;
            conf_upper[j] = stat.conf_high;
        }

        let llf = log_likelihood(input.x(), input.y(), &params);
        // 切片のみモデルの対数尤度: `ȳ=n1/n`の閉じた形の解析解（`fit`の再帰呼び出しは
        // 経由しない）。この式はリンク関数（標準正規CDF）に依存しないため、
        // `nonlinear/common.rs`の`log_likelihood_null`（Logitと共通）をそのまま使う。
        let llnull = log_likelihood_null(input.y());
        let gof = goodness_of_fit(llf, llnull, n, k)?;
        let GoodnessOfFit {
            lr_statistic,
            lr_p_value,
            pseudo_r_squared,
            aic,
            bic,
            df_model,
            df_resid,
        } = gof;

        Ok(Self {
            input,
            params,
            cov_params,
            std_errors,
            z_stats,
            p_values,
            conf_lower,
            conf_upper,
            converged: output.converged,
            n_iter: output.n_iter,
            log_likelihood: llf,
            log_likelihood_null: llnull,
            lr_statistic,
            lr_p_value,
            pseudo_r_squared,
            aic,
            bic,
            df_model,
            df_resid,
        })
    }

    /// 推定に使った入力データ
    pub fn input(&self) -> &ProbitInput {
        &self.input
    }

    /// 係数の分散共分散行列（元のスケール、k×k）
    pub fn cov_params(&self) -> &Mat<f64> {
        &self.cov_params
    }

    /// 標準誤差（k、元のスケール）
    pub fn std_errors(&self) -> &[f64] {
        &self.std_errors
    }

    /// z統計量（k）
    pub fn z_stats(&self) -> &[f64] {
        &self.z_stats
    }

    /// 両側p値（k）
    pub fn p_values(&self) -> &[f64] {
        &self.p_values
    }

    /// 信頼区間の下限（k）
    pub fn conf_lower(&self) -> &[f64] {
        &self.conf_lower
    }

    /// 信頼区間の上限（k）
    pub fn conf_upper(&self) -> &[f64] {
        &self.conf_upper
    }

    /// 係数（元のスケール）
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    /// 収束したかどうか
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// 実際の反復回数
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// 対数尤度 `ℓ(θ̂)`
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// 切片のみモデルの対数尤度 `ℓ(θ̂_null)`
    pub fn log_likelihood_null(&self) -> f64 {
        self.log_likelihood_null
    }

    /// 尤度比検定統計量
    pub fn lr_statistic(&self) -> f64 {
        self.lr_statistic
    }

    /// 尤度比検定のp値（`df_model==0`のときNaN）
    pub fn lr_p_value(&self) -> f64 {
        self.lr_p_value
    }

    /// McFadden疑似決定係数
    pub fn pseudo_r_squared(&self) -> f64 {
        self.pseudo_r_squared
    }

    /// 赤池情報量規準
    pub fn aic(&self) -> f64 {
        self.aic
    }

    /// ベイズ情報量規準
    pub fn bic(&self) -> f64 {
        self.bic
    }

    /// 観測数。`self.input.nobs()`への委譲（`OlsEstimator`/`LogitEstimator`と同じ
    /// パターン、`n`という同じ値の出どころを2つに分けない）
    pub fn n_obs(&self) -> usize {
        self.input.nobs()
    }

    /// モデルの自由度（`k-1`）
    pub fn df_model(&self) -> usize {
        self.df_model
    }

    /// 残差自由度（`n-k`）
    pub fn df_resid(&self) -> usize {
        self.df_resid
    }

    /// 限界効果（`marginal_effects`）。`fit()`とは独立した別メソッド（`fit()`のReturn
    /// 本体には含めない、`nonlinear-api-design.md`6章で確定済み）。`fit()`時の
    /// `cov_params`を再利用するため再最適化は不要（`confidence_level`は`fit()`とは
    /// 独立したパラメータとして受け取り、`fit()`時の値に縛られず事後的に異なる
    /// CI幅を見られるようにする、`LogitEstimator::marginal_effects`と同じ設計）。
    ///
    /// ## 数式（デルタ法）
    ///
    /// `p_i = Φ(x_i'θ)`のとき、変数`j`（連続変数として扱う。`dummy=False`が既定の
    /// statsmodelsの`get_margeff()`に倣い、離散変数の自動判定は行わない設計、
    /// `nonlinear-implementation-notes.md`「限界効果」参照）の限界効果は
    /// `dy/dx_j = φ(x_i'θ)θ_j`（Logitの`p(1-p)θ_j`とは異なり標準正規PDF`φ`を使う。
    /// Issue #78参照）。
    ///
    /// - `at="overall"`（AME）: `g_j(θ) = w(θ)*θ_j`、`w(θ) = (1/n)Σᵢ φ(zᵢ)`
    /// - `at="mean"`/`"median"`: `g_j(θ) = w(θ)*θ_j`、`w(θ) = φ(z̄)`
    ///   （`z̄=x̄'θ`、`x̄`は各説明変数の標本平均または中央値からなる代表点）
    ///
    /// いずれも同じ`g_j(θ)=w(θ)*θ_j`という形に帰着するため、`w`とその勾配
    /// `s_m=∂w/∂θ_m`さえ計算できれば、ヤコビアンは
    /// `∂g_j/∂θ_m = θ_j*s_m + [j==m]*w`という共通の式で書ける
    /// （`overall_w_and_s`/`at_point_w_and_s`が`(w,s)`を計算し、`w`・`s`の計算方法に
    /// 依らない残りの計算——ヤコビアン・デルタ法標準誤差・定数項の除外——は
    /// `nonlinear/common.rs`の`marginal_effects_from_w_s`に共通化されている
    /// （Logitと数式まで同型であることを確認済み、Issue #78）。
    ///
    /// 変数`j`の分散は`Var(g_j) = jac_j · Σ · jac_jᵀ`（`jac_j`はヤコビアンの`j`行目、
    /// `Σ=cov_params`）。標準誤差はこの平方根、検定分布は標準正規分布
    /// （`fit()`本体と同じ、`nonlinear-api-design.md`5章）。
    ///
    /// 定数項（切片）は出力から除外する（切片の限界効果は意味を持たない、
    /// statsmodelsも同様）。
    ///
    /// # Errors
    /// `confidence_level`が`(0, 1)`の範囲外: `CommonError::InvalidConfidenceLevel`
    pub fn marginal_effects(
        &self,
        at: MarginalEffectsAt,
        confidence_level: f64,
    ) -> Result<MarginalEffects, MleError> {
        let x = self.input.x();
        let k = self.input.k();
        let (w, s) = match at {
            MarginalEffectsAt::Overall => overall_w_and_s(x, &self.params),
            MarginalEffectsAt::Mean => at_point_w_and_s(&column_means(x), &self.params),
            MarginalEffectsAt::Median => at_point_w_and_s(&column_medians(x), &self.params),
        };
        marginal_effects_from_w_s(
            FittedModelForMarginalEffects {
                param_names: self.input.param_names(),
                has_intercept: self.input.has_intercept(),
                k,
                params: &self.params,
                cov_params: &self.cov_params,
            },
            w,
            &s,
            confidence_level,
        )
    }

    /// 予測確率 `p_i = Φ(x_i'θ)` を、`fit()`に使った学習データ（`self.input.x()`）の
    /// 各行について返す（`fit()`のReturn本体には含めない別メソッド、
    /// `nonlinear-api-design.md`6章。`LogitEstimator::predict`の`Λ`を`Φ`に置き換えた
    /// Probit版）。
    ///
    /// **新規データでの予測（out-of-sample）は未対応**（本Issueのスコープ外、
    /// 別issueでトラッキング。`LogitEstimator::predict`と同じ、ユーザー確認済み）。
    pub fn predict(&self) -> Vec<f64> {
        let x = self.input.x();
        let n = x.nrows();
        let k = x.ncols();
        let normal = Normal::standard();
        (0..n)
            .map(|i| {
                let z: f64 = (0..k).map(|j| *x.get(i, j) * self.params[j]).sum();
                normal.cdf(z)
            })
            .collect()
    }

    /// 分類の的中表（2×2、`table[actual][predicted]`のカウント。行=実測クラス、
    /// 列=予測クラス）。`predict()`が返す予測確率のみを`threshold`で二値化し、実測`y`は
    /// `threshold`に関わらず常に`0.5`で二値化する。数式・statsmodelsとの整合性の詳細は
    /// `nonlinear/common.rs`の`pred_table`のdocコメント参照（リンク関数に依存しない計算
    /// のため`common.rs`に共通化されている、Logitと共通、Issue #79）。
    ///
    /// **新規データでの的中表（out-of-sample）は未対応**（本Issueのスコープ外、
    /// 別issueでトラッキング。`LogitEstimator::pred_table`と同じ、ユーザー確認済み）。
    pub fn pred_table(&self, threshold: f64) -> Mat<f64> {
        pred_table(&self.predict(), self.input.y(), threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonlinear::common::dydx_and_jacobian;
    use statrs::distribution::ChiSquared;

    #[test]
    fn from_columns_with_intercept_prepends_const_column() {
        let y = vec![0.0, 1.0, 0.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0]];
        let input = ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        assert_eq!(input.nobs(), 3);
        assert_eq!(input.k(), 2);
        assert!(input.has_intercept());
        assert_eq!(input.param_names(), ["const".to_string(), "x1".to_string()]);
        assert_eq!(input.dep_var_name(), "y");
        for i in 0..3 {
            assert_eq!(*input.x().get(i, 0), 1.0);
            assert_eq!(*input.x().get(i, 1), x_columns[0][i]);
            assert_eq!(*input.y().get(i, 0), y[i]);
        }
    }

    #[test]
    fn from_columns_without_intercept_omits_const_column() {
        let y = vec![1.0, 0.0, 1.0, 0.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0], vec![4.0, 3.0, 2.0, 1.0]];
        let input = ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            false,
            "y".to_string(),
        )
        .unwrap();

        assert_eq!(input.k(), 2);
        assert!(!input.has_intercept());
        assert_eq!(input.param_names(), ["x1".to_string(), "x2".to_string()]);
    }

    #[test]
    fn from_columns_returns_dimension_mismatch_on_mismatched_column_length() {
        let y = vec![0.0, 1.0, 0.0];
        let x_columns = vec![vec![1.0, 2.0]];
        let result = ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
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
        let _ = ProbitInput::from_columns(&y, &x_columns, vec![], true, "y".to_string());
    }

    /// n=4, k=2（切片+x1）の小規模データ。`θ=[0,0]`のとき`z_i=0`（全観測で共通）となり、
    /// `Φ(0)=0.5`・`φ(0)=1/√(2π)`という既知の値から`cost`/`gradient`/`hessian`が
    /// 閉じた形で手計算できる（`Φ(0)=0.5`はLogitの`logistic(0)=0.5`と同じ値になるため、
    /// `cost`は`LogitInput`の対応するテストと同じ`4*ln(2)`になる）。
    fn small_input() -> ProbitInput {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0]];
        ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap()
    }

    #[test]
    fn cost_gradient_hessian_match_closed_form_at_zero_params() {
        let input = small_input();
        let problem = ProbitProblem::new(&input);
        let params = vec![0.0, 0.0];

        // z_i=0のときΦ(0)=0.5、cost = -Σlog(0.5) = 4*ln(2)
        let cost = problem.cost(&params).unwrap();
        assert!(
            (cost - 4.0 * std::f64::consts::LN_2).abs() < 1e-12,
            "{cost}"
        );

        // λ_i = q_i*φ(0)/Φ(0) = q_i*c（c=φ(0)/0.5=√(2/π)）、y=[0,1,0,1]→q=[-1,1,-1,1]
        // grad(-ℓ) = -Σλ_i*x_i、切片成分: -c*(-1+1-1+1)=0、x1成分: -c*(-1+2-3+4)=-2c
        let c = (2.0 / std::f64::consts::PI).sqrt();
        let grad = problem.gradient(&params).unwrap();
        assert!(grad[0].abs() < 1e-9, "{:?}", grad);
        assert!((grad[1] - (-2.0 * c)).abs() < 1e-9, "{:?}", grad);

        // z_i=0なのでw_i=λ_i(λ_i+0)=λ_i²=c²=2/π（符号によらず全観測共通）
        // Hessian(-ℓ) = (2/π)*X'X、X'X=[[4,10],[10,30]]（n=4,Σx1=10,Σx1²=30、Logitと同じ設計行列）
        let w = c * c;
        let hessian = problem.hessian(&params).unwrap();
        assert!((hessian[0][0] - w * 4.0).abs() < 1e-9, "{:?}", hessian);
        assert!((hessian[0][1] - w * 10.0).abs() < 1e-9, "{:?}", hessian);
        assert!((hessian[1][0] - w * 10.0).abs() < 1e-9, "{:?}", hessian);
        assert!((hessian[1][1] - w * 30.0).abs() < 1e-9, "{:?}", hessian);
    }

    #[test]
    fn scores_match_closed_form_at_zero_params() {
        let input = small_input();
        let problem = ProbitProblem::new(&input);
        let scores = problem.scores(&[0.0, 0.0]);

        // score_i = λ_i*x_i = q_i*c*x_i（c=√(2/π)）、y=[0,1,0,1]→q=[-1,1,-1,1]
        let c = (2.0 / std::f64::consts::PI).sqrt();
        let q = [-1.0, 1.0, -1.0, 1.0];
        let x1 = [1.0, 2.0, 3.0, 4.0];
        for i in 0..4 {
            let lambda = q[i] * c;
            assert!((*scores.get(i, 0) - lambda).abs() < 1e-9, "row {i}");
            assert!((*scores.get(i, 1) - lambda * x1[i]).abs() < 1e-9, "row {i}");
        }
    }

    #[test]
    fn scores_sum_to_negative_gradient() {
        // scoresは対数尤度の生のスコア（符号反転なし）、gradientはCostFunction
        // （負の対数尤度）の勾配のため、観測方向に合計すると符号が逆になるはず。
        let input = small_input();
        let problem = ProbitProblem::new(&input);
        let params = vec![0.3, -0.2];

        let scores = problem.scores(&params);
        let grad = problem.gradient(&params).unwrap();
        let n = scores.nrows();

        for j in 0..2 {
            let sum: f64 = (0..n).map(|i| *scores.get(i, j)).sum();
            assert!(
                (sum - (-grad[j])).abs() < 1e-9,
                "j={j}, sum={sum}, grad={:?}",
                grad
            );
        }
    }

    #[test]
    fn gradient_matches_numerical_differentiation_of_cost() {
        let input = small_input();
        let problem = ProbitProblem::new(&input);
        let params = vec![0.3, -0.2];
        let h = 1e-6;

        let analytic = problem.gradient(&params).unwrap();
        for j in 0..2 {
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
        let input = small_input();
        let problem = ProbitProblem::new(&input);
        let params = vec![0.3, -0.2];
        let h = 1e-5;

        let analytic = problem.hessian(&params).unwrap();
        for j in 0..2 {
            let mut plus = params.clone();
            plus[j] += h;
            let mut minus = params.clone();
            minus[j] -= h;
            let grad_plus = problem.gradient(&plus).unwrap();
            let grad_minus = problem.gradient(&minus).unwrap();
            for i in 0..2 {
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
    fn cost_gradient_hessian_stay_finite_for_extreme_linear_predictor() {
        // z_i=1000（クランプが無ければu=q*z=±1000でφ(u)/Φ(u)が0にアンダーフローし
        // NaNになる、モジュール冒頭「数値安定化について」参照）。U_CLAMPでのクランプ
        // によりcost/gradient/hessianが常に有限であることを確認する。
        let y = vec![1.0, 0.0];
        let input = ProbitInput::from_columns(&y, &[], vec![], true, "y".to_string()).unwrap();
        let problem = ProbitProblem::new(&input);
        let params = vec![1000.0];

        let cost = problem.cost(&params).unwrap();
        assert!(cost.is_finite(), "{cost}");

        let grad = problem.gradient(&params).unwrap();
        assert!(grad[0].is_finite(), "{:?}", grad);

        let hessian = problem.hessian(&params).unwrap();
        assert!(hessian[0][0].is_finite(), "{:?}", hessian);

        let scores = problem.scores(&params);
        for i in 0..2 {
            assert!(scores.get(i, 0).is_finite(), "row {i}");
        }
    }

    /// 切片のみ（説明変数なし）のProbitは、MLEの一階条件`Σ(y_i-Φ(θ))=0`（`z_i=θ`が
    /// 全観測共通）から`Φ(θ̂) = ȳ`、すなわち`θ̂ = Φ⁻¹(ȳ)`という閉じた形の解析解を持つ
    /// （`LogitInput`の`θ̂ = ln(ȳ/(1-ȳ))`に相当するProbit版。`fit`が最適化ロジックを
    /// 経ずに正しい値へ収束することを検証できる）。
    fn intercept_only_input() -> ProbitInput {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        ProbitInput::from_columns(&y, &[], vec![], true, "y".to_string()).unwrap()
    }

    #[test]
    fn fit_newton_converges_to_closed_form_solution_for_intercept_only_model() {
        let input = intercept_only_input();
        let estimator = ProbitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let y_bar: f64 = 4.0 / 7.0;
        let expected = Normal::standard().inverse_cdf(y_bar);

        assert!(estimator.converged());
        assert_eq!(estimator.params().len(), 1);
        assert!(
            (estimator.params()[0] - expected).abs() < 1e-6,
            "params={:?}, expected={}",
            estimator.params(),
            expected
        );
        // 切片のみの1次元凹関数のNewton法は数回で収束するはず
        assert!(estimator.n_iter() <= 10, "n_iter={}", estimator.n_iter());
    }

    /// 切片のみモデルは観測情報行列も閉じた形で書ける。MLEの一階条件`Σλᵢ=0`が
    /// 収束点で厳密に成り立つことを使うと、Hessian（`-ℓ`の2階微分）は
    /// `H(θ̂) = Σᵢλᵢ² = n*φ(θ̂)²/(ȳ(1-ȳ))`という形に単純化できる
    /// （`λᵢ(λᵢ+θ)`の`θ`の項が`θ*Σλᵢ=0`で消える。導出はモジュールdocコメント
    /// 「数式」節の`λᵢ`の定義と合わせて`probit-implementation-notes.md`参照）。
    /// `Var(θ̂) = H(θ̂)⁻¹ = ȳ(1-ȳ)/(n*φ(θ̂)²)`。z値・p値・信頼区間はこの分散から
    /// 標準正規分布（独立に`statrs::Normal`で検算）で導出できる。
    #[test]
    fn fit_computes_std_errors_z_stats_p_values_and_ci_matching_closed_form_for_intercept_only_model()
     {
        let estimator = ProbitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let n: f64 = 7.0;
        let y_bar: f64 = 4.0 / 7.0;
        let theta_hat = Normal::standard().inverse_cdf(y_bar);
        let phi_theta_hat = Normal::standard().pdf(theta_hat);
        let expected_var = y_bar * (1.0 - y_bar) / (n * phi_theta_hat * phi_theta_hat);
        let expected_se = expected_var.sqrt();

        // Newtonの収束判定（勾配ノルム`tol=1e-6`）による数値誤差があるため、
        // 他の閉じた形テスト（`fit_newton_converges_to_closed_form_solution_...`）と
        // 同じ桁の許容誤差（1e-6）を使う。
        assert!((*estimator.cov_params().get(0, 0) - expected_var).abs() < 1e-6);
        assert!((estimator.std_errors()[0] - expected_se).abs() < 1e-6);

        let expected_z = estimator.params()[0] / expected_se;
        assert!((estimator.z_stats()[0] - expected_z).abs() < 1e-6);

        // p値・信頼区間はstatrsのNormalで独立に検算する（本体実装と同じ計算式を
        // 繰り返すのではなく、標準正規分布の性質から直接導出する）。
        let normal = Normal::standard();
        let expected_p = 2.0 * (1.0 - normal.cdf(expected_z.abs()));
        assert!((estimator.p_values()[0] - expected_p).abs() < 1e-6);

        let z_crit = normal.inverse_cdf(0.975);
        let expected_lower = estimator.params()[0] - z_crit * expected_se;
        let expected_upper = estimator.params()[0] + z_crit * expected_se;
        assert!((estimator.conf_lower()[0] - expected_lower).abs() < 1e-6);
        assert!((estimator.conf_upper()[0] - expected_upper).abs() < 1e-6);
    }

    /// 多変量（説明変数が2つ以上）の場合、標準誤差に閉じた形の解析解は無いため、
    /// `cov_params`の対称性・各種統計量の内部整合性（z値・信頼区間の定義式通りの関係）を
    /// 検証する回帰テスト。特に`destandardize_cov_params`が非対角成分も含めて
    /// 正しく元のスケールへ逆変換できているかを確認する（対角成分だけでは
    /// `destandardize_cov_params`の`stds[i]*stds[j]`の掛け違い（転置ミス等）を検出できない）。
    #[test]
    fn fit_cov_params_is_symmetric_and_stats_are_internally_consistent() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = ProbitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        let k = 3;

        for i in 0..k {
            for j in 0..k {
                assert!(
                    (*estimator.cov_params().get(i, j) - *estimator.cov_params().get(j, i)).abs()
                        < 1e-9,
                    "cov_params is not symmetric at ({i},{j})"
                );
            }
            assert!(
                *estimator.cov_params().get(i, i) > 0.0,
                "diagonal[{i}] <= 0"
            );
        }

        let normal = Normal::standard();
        let z_crit = normal.inverse_cdf(0.975);
        for j in 0..k {
            let se = estimator.std_errors()[j];
            assert!((se * se - *estimator.cov_params().get(j, j)).abs() < 1e-9);
            assert!((estimator.z_stats()[j] - estimator.params()[j] / se).abs() < 1e-9);
            assert!(
                (estimator.conf_upper()[j] - estimator.conf_lower()[j] - 2.0 * z_crit * se).abs()
                    < 1e-9
            );
        }
    }

    /// 切片のみモデルは`log_likelihood`と`log_likelihood_null`が定義上一致する
    /// （どちらも同じ「切片のみ」モデルを参照するため）。この性質を使い、
    /// `df_model=0`分岐（`lr_p_value=NaN`、OLSの`f_p_value`と同じ扱い）を含めた
    /// 適合度統計量一式を検証する（`LogitEstimator`の対応するテストと同じ構成）。
    #[test]
    fn fit_computes_goodness_of_fit_statistics_for_intercept_only_model() {
        let estimator = ProbitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let n: f64 = 7.0;
        let y_bar: f64 = 4.0 / 7.0;
        // 切片のみモデルの対数尤度の閉じた形: n1*ln(ȳ) + n0*ln(1-ȳ)（n1=4, n0=3）
        let expected_ll = 4.0 * y_bar.ln() + 3.0 * (1.0 - y_bar).ln();

        assert!((estimator.log_likelihood() - expected_ll).abs() < 1e-6);
        assert!((estimator.log_likelihood_null() - expected_ll).abs() < 1e-9);
        assert!(estimator.lr_statistic().abs() < 1e-6);
        assert!(estimator.pseudo_r_squared().abs() < 1e-6);
        assert_eq!(estimator.df_model(), 0);
        assert!(estimator.lr_p_value().is_nan());
        assert_eq!(estimator.n_obs(), 7);
        assert_eq!(estimator.df_resid(), 6);

        let expected_aic = -2.0 * expected_ll + 2.0;
        let expected_bic = -2.0 * expected_ll + n.ln();
        assert!((estimator.aic() - expected_aic).abs() < 1e-6);
        assert!((estimator.bic() - expected_bic).abs() < 1e-6);
    }

    /// 多変量（k=3）モデルでの適合度統計量を、実装（`clamped_pdf_cdf`ベース）とは異なる式
    /// （標準正規CDF`Φ`から直接`Σ[y ln Φ(z) + (1-y) ln(1-Φ(z))]`を計算するベルヌーイ
    /// 対数尤度の定義式そのもの）で独立に再計算し、突き合わせる。
    /// `fit_cov_params_is_symmetric_and_stats_are_internally_consistent`と同じ
    /// データセットを再利用する（`LogitEstimator`の対応するテストと同じ構成）。
    #[test]
    fn fit_computes_goodness_of_fit_statistics_matching_independently_recomputed_values() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = ProbitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let n = 4usize;
        let k = 3usize;
        let x = estimator.input().x();
        let params = estimator.params();
        let normal = Normal::standard();
        let expected_ll: f64 = (0..n)
            .map(|i| {
                let z: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
                let p = normal.cdf(z);
                y[i] * p.ln() + (1.0 - y[i]) * (1.0 - p).ln()
            })
            .sum();
        assert!((estimator.log_likelihood() - expected_ll).abs() < 1e-9);

        let y_bar: f64 = 2.0 / 4.0; // n1=2, n0=2
        let expected_llnull = 2.0 * y_bar.ln() + 2.0 * (1.0 - y_bar).ln();
        assert!((estimator.log_likelihood_null() - expected_llnull).abs() < 1e-9);

        let expected_lr = 2.0 * (expected_ll - expected_llnull);
        assert!((estimator.lr_statistic() - expected_lr).abs() < 1e-9);

        let expected_pseudo_r2 = 1.0 - expected_ll / expected_llnull;
        assert!((estimator.pseudo_r_squared() - expected_pseudo_r2).abs() < 1e-9);

        assert_eq!(estimator.df_model(), k - 1);
        assert_eq!(estimator.df_resid(), n - k);
        assert_eq!(estimator.n_obs(), n);

        let chi2 = ChiSquared::new((k - 1) as f64).unwrap();
        let expected_lr_p = 1.0 - chi2.cdf(expected_lr);
        assert!((estimator.lr_p_value() - expected_lr_p).abs() < 1e-9);

        let expected_aic = -2.0 * expected_ll + 2.0 * (k as f64);
        let expected_bic = -2.0 * expected_ll + (n as f64).ln() * (k as f64);
        assert!((estimator.aic() - expected_aic).abs() < 1e-9);
        assert!((estimator.bic() - expected_bic).abs() < 1e-9);
    }

    /// `include_intercept=false`のとき、`log_likelihood_null`が参照する「切片のみ」
    /// モデルはフィット対象のモデルの部分集合（入れ子）にならない
    /// （`ProbitEstimator`の`log_likelihood_null`フィールドdocコメント参照）。
    /// この場合`lr_statistic`が負になりうる（statsmodels準拠の仕様上の挙動、
    /// `LogitEstimator`の対応するテストと同じ回帰テスト）ことを固定する。
    /// `df_model`/`df_resid`/`aic`/`bic`は`include_intercept`の値に関わらず
    /// 同じ式（`k-1`/`n-k`/`-2ℓ+2k`/`-2ℓ+ln(n)k`）で計算されることも確認する。
    #[test]
    fn fit_lr_statistic_can_be_negative_when_include_intercept_is_false() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            false,
            "y".to_string(),
        )
        .unwrap();

        let estimator = ProbitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let n = 4usize;
        let k = 2usize; // 切片なし、x1・x2のみ
        assert_eq!(estimator.df_model(), k - 1);
        assert_eq!(estimator.df_resid(), n - k);

        let expected_aic = -2.0 * estimator.log_likelihood() + 2.0 * (k as f64);
        let expected_bic = -2.0 * estimator.log_likelihood() + (n as f64).ln() * (k as f64);
        assert!((estimator.aic() - expected_aic).abs() < 1e-9);
        assert!((estimator.bic() - expected_bic).abs() < 1e-9);

        // 非入れ子のため`lr_statistic`が負になりうる（NaN/Infにはならない）。
        assert!(estimator.lr_statistic().is_finite());
        assert!(estimator.lr_p_value().is_finite());
    }

    /// `cov_type=Opg`/`Hc0`/`Hc1`が返す`cov_params`を、`fit()`と同じ手順
    /// （標準化→収束点でのscores/Hessian評価→`common.rs`の共通行列演算→
    /// `destandardize_cov_params`）をテスト側で独立に再現した値と突き合わせる。
    /// 多変量（k=3）データセットを使う理由: 切片のみモデルでは情報行列の等式
    /// （`Σᵢsᵢsᵢ' = -H`）が有限標本で厳密に成り立ってしまい、`classical`/`opg`/`hc0`
    /// が偶然同じ値になるため、`fit()`の`match cov_type`の配線ミスを検出できない
    /// （`fit_cov_params_is_symmetric_and_stats_are_internally_consistent`と同じ
    /// データセットを再利用する。Logitの対応するテストと同じ構成）。
    #[test]
    fn fit_cov_type_opg_hc0_hc1_match_independently_recomputed_values() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let make_input = || {
            ProbitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string(), "x2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };
        let k = 3;
        let n = 4;

        let classical = ProbitEstimator::fit(
            make_input(),
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        // `fit()`と同じ手順を独立に再現する: 標準化→θ_stdへ変換→収束点でのscores/
        // Hessian評価→destandardize_cov_params。`ProbitProblem::hessian`（argminトレイト）は
        // コスト関数（負の対数尤度）のHessianを返す符号規約のため、`run_solver`が
        // `SolverOutput.hessian`に格納する対数尤度そのもののHessianに合わせて1回符号反転する
        // （`run_solver`のdocコメント「Hessianトレイトの符号規約」と同じ変換）。
        let input_for_reconstruction = make_input();
        let (x_std, scale) = standardize_columns(
            input_for_reconstruction.x(),
            input_for_reconstruction.has_intercept(),
        );
        let params_std: Vec<f64> = classical
            .params()
            .iter()
            .zip(scale.stds())
            .map(|(p, s)| p * s)
            .collect();
        let problem_std =
            ProbitProblem::from_standardized(x_std, input_for_reconstruction.y().clone());
        let scores_std = problem_std.scores(&params_std);
        let cost_hessian_std = problem_std.hessian(&params_std).unwrap();
        let hessian_std = Mat::from_fn(k, k, |i, j| -cost_hessian_std[i][j]);

        let expected_opg =
            destandardize_cov_params(&opg_cov_params(&scores_std, k).unwrap(), &scale);
        let expected_hc0 = destandardize_cov_params(
            &sandwich_cov_params(&hessian_std, &scores_std, n, k, SandwichVariant::Hc0).unwrap(),
            &scale,
        );
        let expected_hc1 = destandardize_cov_params(
            &sandwich_cov_params(&hessian_std, &scores_std, n, k, SandwichVariant::Hc1).unwrap(),
            &scale,
        );

        let cases = [
            (CovType::Opg, &expected_opg),
            (CovType::Hc0, &expected_hc0),
            (CovType::Hc1, &expected_hc1),
        ];
        for (cov_type, expected) in cases {
            let estimator = ProbitEstimator::fit(
                make_input(),
                Method::Newton,
                35,
                1e-8,
                true,
                cov_type.clone(),
                0.95,
            )
            .unwrap();
            for i in 0..k {
                for j in 0..k {
                    assert!(
                        (*estimator.cov_params().get(i, j) - *expected.get(i, j)).abs() < 1e-8,
                        "cov_type={:?}, ({i},{j}): actual={}, expected={}",
                        cov_type,
                        *estimator.cov_params().get(i, j),
                        *expected.get(i, j)
                    );
                }
            }
        }
    }

    /// `cov_type=Cluster`が返す`cov_params`を、`fit()`と同じ手順をテスト側で独立に
    /// 再現した値と突き合わせる（上の`fit_cov_type_opg_hc0_hc1_match_independently_
    /// recomputed_values`と同じ技法・同じ多変量データセット。情報行列の等式が
    /// 厳密に成り立つ切片のみモデルでは配線ミスを検出できないため。Logitの
    /// 対応するテストと同じ構成）。
    #[test]
    fn fit_cov_type_cluster_matches_independently_recomputed_values() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let make_input = || {
            ProbitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string(), "x2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };
        let k = 3;
        let n = 4;
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];

        let classical = ProbitEstimator::fit(
            make_input(),
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let input_for_reconstruction = make_input();
        let (x_std, scale) = standardize_columns(
            input_for_reconstruction.x(),
            input_for_reconstruction.has_intercept(),
        );
        let params_std: Vec<f64> = classical
            .params()
            .iter()
            .zip(scale.stds())
            .map(|(p, s)| p * s)
            .collect();
        let problem_std =
            ProbitProblem::from_standardized(x_std, input_for_reconstruction.y().clone());
        let scores_std = problem_std.scores(&params_std);
        let cost_hessian_std = problem_std.hessian(&params_std).unwrap();
        let hessian_std = Mat::from_fn(k, k, |i, j| -cost_hessian_std[i][j]);

        let expected_cluster = destandardize_cov_params(
            &cluster_cov_params(&hessian_std, &scores_std, n, k, &groups).unwrap(),
            &scale,
        );

        let estimator = ProbitEstimator::fit(
            make_input(),
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Cluster {
                groups: Some(groups),
            },
            0.95,
        )
        .unwrap();
        for i in 0..k {
            for j in 0..k {
                assert!(
                    (*estimator.cov_params().get(i, j) - *expected_cluster.get(i, j)).abs() < 1e-8,
                    "({i},{j}): actual={}, expected={}",
                    *estimator.cov_params().get(i, j),
                    *expected_cluster.get(i, j)
                );
            }
        }
    }

    /// 上のテストは2:2の均等サイズのグループのみを検証しているが、`testing-policy.md`が
    /// 指摘する通り均等サイズのみのテストは実務で起こりやすい偏った分布のグループサイズ
    /// を見逃しうる。OLS/Logit側の対応するテストに倣い、3:2の不均衡なグループでも
    /// 同じ独立再計算の技法で検証する。
    #[test]
    fn fit_cov_type_cluster_matches_independently_recomputed_values_with_unbalanced_groups() {
        let y = vec![0.0, 1.0, 0.0, 1.0, 1.0];
        let x_columns = vec![
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            vec![-5.0, 2.0, 8.0, -1.0, 3.0],
        ];
        let make_input = || {
            ProbitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string(), "x2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };
        let k = 3;
        let n = 5;
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];

        let classical = ProbitEstimator::fit(
            make_input(),
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let input_for_reconstruction = make_input();
        let (x_std, scale) = standardize_columns(
            input_for_reconstruction.x(),
            input_for_reconstruction.has_intercept(),
        );
        let params_std: Vec<f64> = classical
            .params()
            .iter()
            .zip(scale.stds())
            .map(|(p, s)| p * s)
            .collect();
        let problem_std =
            ProbitProblem::from_standardized(x_std, input_for_reconstruction.y().clone());
        let scores_std = problem_std.scores(&params_std);
        let cost_hessian_std = problem_std.hessian(&params_std).unwrap();
        let hessian_std = Mat::from_fn(k, k, |i, j| -cost_hessian_std[i][j]);

        let expected_cluster = destandardize_cov_params(
            &cluster_cov_params(&hessian_std, &scores_std, n, k, &groups).unwrap(),
            &scale,
        );

        let estimator = ProbitEstimator::fit(
            make_input(),
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Cluster {
                groups: Some(groups),
            },
            0.95,
        )
        .unwrap();
        for i in 0..k {
            for j in 0..k {
                assert!(
                    (*estimator.cov_params().get(i, j) - *expected_cluster.get(i, j)).abs() < 1e-8,
                    "({i},{j}): actual={}, expected={}",
                    *estimator.cov_params().get(i, j),
                    *expected_cluster.get(i, j)
                );
            }
        }
    }

    #[test]
    fn fit_returns_missing_cluster_column_error_when_groups_not_provided() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let input = ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = ProbitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Cluster { groups: None },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::MissingClusterColumn)
        );
    }

    #[test]
    fn fit_returns_insufficient_clusters_error_when_only_one_group() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let input = ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let groups = vec!["a".to_string(); 4];

        let result = ProbitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Cluster {
                groups: Some(groups),
            },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InsufficientClusters { g: 1 })
        );
    }

    /// `method`（`bfgs`/`lbfgs`）と`cov_type`（`Opg`/`Hc0`/`Hc1`/`Cluster`）の組み合わせが
    /// 正しく機能することを確認する（Logitの対応するテストと同じ理由: `method`横断が
    /// `CovType::Classical`のみ、`cov_type`横断が`Method::Newton`のみのテストでは、
    /// 両方を同時に変える組み合わせが未検証になる）。`scores_std`の評価は収束点の
    /// パラメータにのみ依存し最適化アルゴリズムの種類に依存しない設計のため、
    /// `newton`で計算した`cov_params`（既に上のテストで正しさを検証済み）と
    /// `bfgs`/`lbfgs`の結果が一致するはず。
    #[test]
    fn fit_non_classical_cov_types_work_with_bfgs_and_lbfgs() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let make_input = || {
            ProbitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string(), "x2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };
        let k = 3;
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];

        for cov_type in [
            CovType::Opg,
            CovType::Hc0,
            CovType::Hc1,
            CovType::Cluster {
                groups: Some(groups),
            },
        ] {
            let newton = ProbitEstimator::fit(
                make_input(),
                Method::Newton,
                35,
                1e-8,
                true,
                cov_type.clone(),
                0.95,
            )
            .unwrap();

            for method in [Method::Bfgs, Method::Lbfgs] {
                let estimator = ProbitEstimator::fit(
                    make_input(),
                    method,
                    200,
                    1e-8,
                    true,
                    cov_type.clone(),
                    0.95,
                )
                .unwrap();

                assert!(
                    estimator.converged(),
                    "cov_type={:?}, {:?}",
                    cov_type,
                    method
                );
                for i in 0..k {
                    for j in 0..k {
                        assert!(
                            (*estimator.cov_params().get(i, j) - *newton.cov_params().get(i, j))
                                .abs()
                                < 1e-4,
                            "cov_type={:?}, method={:?}, ({i},{j}): actual={}, newton={}",
                            cov_type,
                            method,
                            *estimator.cov_params().get(i, j),
                            *newton.cov_params().get(i, j)
                        );
                    }
                }
            }
        }
    }

    /// `newton`と同じデータセット（既知の解析解を持つ切片のみモデル）で`bfgs`/`lbfgs`を
    /// 実行し、いずれも同じ解析解へ収束することを検証する（Issue #73完了条件）。
    #[test]
    fn fit_bfgs_and_lbfgs_converge_to_same_solution_as_newton() {
        let y_bar: f64 = 4.0 / 7.0;
        let expected = Normal::standard().inverse_cdf(y_bar);

        for method in [Method::Bfgs, Method::Lbfgs] {
            let estimator = ProbitEstimator::fit(
                intercept_only_input(),
                method,
                100,
                1e-6,
                true,
                CovType::Classical,
                0.95,
            )
            .unwrap();

            assert!(estimator.converged(), "{:?}", method);
            assert!(
                (estimator.params()[0] - expected).abs() < 1e-4,
                "method={:?}, params={:?}, expected={}",
                method,
                estimator.params(),
                expected
            );
        }
    }

    /// 切片のみモデル（`intercept_only_input`）は`x`が定数列（切片）だけのため、
    /// `standardize_columns`のスケーリングが実質no-op（`stds`が全て`1.0`のまま）になり、
    /// 標準化・逆標準化の往復ロジックを通らない（Logitの対応するテストと同じ理由）。
    /// このテストは非自明なスケール（`std`が1から離れた値）を持つ説明変数を含む
    /// データセットで`newton`/`bfgs`/`lbfgs`を実行し、3手法が同じ解へ収束することを
    /// 検証する（閉じた形の解析解は存在しないため、`newton`の結果を参照値として使う
    /// クロスメソッド一致検証）。
    #[test]
    fn fit_bfgs_and_lbfgs_agree_with_newton_when_design_matrix_has_nontrivial_scale() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0]];
        let make_input = || {
            ProbitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let newton = ProbitEstimator::fit(
            make_input(),
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert!(newton.converged());

        for method in [Method::Bfgs, Method::Lbfgs] {
            let estimator = ProbitEstimator::fit(
                make_input(),
                method,
                200,
                1e-8,
                true,
                CovType::Classical,
                0.95,
            )
            .unwrap();

            assert!(estimator.converged(), "{:?}", method);
            for j in 0..2 {
                assert!(
                    (estimator.params()[j] - newton.params()[j]).abs() < 1e-4,
                    "method={:?}, j={j}, params={:?}, newton_params={:?}",
                    method,
                    estimator.params(),
                    newton.params()
                );
            }
        }
    }

    #[test]
    fn fit_returns_invalid_confidence_level_error_out_of_range() {
        let result = ProbitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            1.5,
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5
            })
        );
    }

    #[test]
    fn fit_returns_invalid_max_iter_error_for_non_positive_max_iter() {
        let result = ProbitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            0,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::InvalidMaxIter { max_iter: 0 }
        );
    }

    #[test]
    fn fit_returns_invalid_tol_error_for_non_positive_tol() {
        for tol in [0.0, -1.0] {
            let result = ProbitEstimator::fit(
                intercept_only_input(),
                Method::Newton,
                35,
                tol,
                true,
                CovType::Classical,
                0.95,
            );
            assert_eq!(result.unwrap_err(), MleError::InvalidTol { tol });
        }
    }

    #[test]
    fn fit_returns_invalid_binary_y_error_for_non_binary_y() {
        let y = vec![0.0, 1.0, 0.5, 1.0];
        let input = ProbitInput::from_columns(&y, &[], vec![], true, "y".to_string()).unwrap();
        let result = ProbitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::InvalidBinaryY { row: 2, value: 0.5 }
        );
    }

    #[test]
    fn fit_returns_no_regressors_error_when_k_is_zero() {
        // `include_intercept=false`かつ説明変数も無い（k=0）病的な入力。`n<=k`チェック
        // （`n>=1`なら常に通過してしまう）をすり抜けて後段の分散共分散行列計算で
        // panicすることをLogit実装（Issue #118/#130）で経験済みのため、Probitでは
        // 当初からk==0を明示的に検証している（モジュールdocコメント・`fit`のdocコメント参照）。
        let y = vec![0.0, 1.0, 0.0, 1.0, 1.0];
        let input = ProbitInput::from_columns(&y, &[], vec![], false, "y".to_string()).unwrap();
        assert_eq!(input.k(), 0);

        let result = ProbitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::NoRegressors { n: 5 })
        );
    }

    #[test]
    fn fit_returns_insufficient_observations_error_when_n_less_equal_k() {
        let y = vec![0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0]];
        let input = ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = ProbitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InsufficientObservations { n: 2, k: 2 })
        );
    }

    #[test]
    fn fit_returns_singular_hessian_error_for_perfectly_collinear_design_matrix() {
        // x2 = 2*x1（完全な多重共線性）。θ=0でのHessianはw*X'X（w=λ(λ+z)、z=0のとき
        // w=2/π）で、X'X自体が構造的に特異（yの値に関わらず常に特異）なので、
        // Newtonの初回ステップで確実に特異性検出に引っかかる（Logitの対応するテストと
        // 同じ理由、完全分離のような「収束の挙動に依存する」ケースと異なり決定的に再現できる）。
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 4.0, 6.0, 8.0]];
        let input = ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = ProbitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        );
        assert!(
            matches!(result, Err(MleError::SingularHessian)),
            "{:?}",
            result
        );
    }

    /// 同じ完全な多重共線性のデータセットを`bfgs`/`lbfgs`で最適化した場合の
    /// `SingularHessian`伝播経路（`LogitEstimator`の対応するテスト・Issue #129と同じ理由）:
    /// `newton`は`newton_step`内の特異性検出（最適化のステップ計算中）で検出するが、
    /// `bfgs`/`lbfgs`は`newton_step`を一切経由しない（準ニュートン法は内部の近似逆Hessianで
    /// 降下方向を決めるため、モデルの解析的Hessianの特異性に依存しない）。この場合、
    /// 収束後に`observed_information_cov_params`（`neg_hessian_inverse`）が呼ぶ
    /// `ensure_well_conditioned_symmetric_matrix`（固有値ベースの悪条件検出）が、
    /// `bfgs`/`lbfgs`にとって唯一の特異性検出経路になる。Issue #80（カバレッジ確認）で、
    /// Logit側で既に判明済みのギャップパターンとして追加（`probit-implementation-notes.md`
    /// 「既知のテストギャップ」参照）。
    #[test]
    fn fit_returns_singular_hessian_error_for_perfectly_collinear_design_matrix_with_bfgs_and_lbfgs()
     {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 4.0, 6.0, 8.0]];

        for method in [Method::Bfgs, Method::Lbfgs] {
            let input = ProbitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string(), "x2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();

            let result =
                ProbitEstimator::fit(input, method, 100, 1e-6, true, CovType::Classical, 0.95);
            assert!(
                matches!(result, Err(MleError::SingularHessian)),
                "method={:?}, result={:?}",
                method,
                result
            );
        }
    }

    /// `sandwich_cov_params`（`cov_type=Hc0`/`Hc1`）も内部で`neg_hessian_inverse`を
    /// 呼ぶため、`Classical`と同じ完全な多重共線性のデータセットで`SingularHessian`に
    /// なるはずだが、`fit()`の`CovType::Hc0`/`Hc1`分岐の`?`（エラー伝播）を通るテストが
    /// 無かった（`cargo-llvm-cov`で判明。`LogitEstimator`の対応するテストと同じ理由、
    /// Issue #80）。
    ///
    /// `method=Newton`は使わない: `newton_step`内の特異性検出（ピボット付きQR）が
    /// `cov_type`の分岐に到達する前（最適化中）に`SingularHessian`を返してしまうため
    /// （上の`_with_bfgs_and_lbfgs`テストと同じ理由）。
    #[test]
    fn fit_returns_singular_hessian_error_for_perfectly_collinear_design_matrix_with_hc0_and_hc1() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 4.0, 6.0, 8.0]];

        for cov_type in [CovType::Hc0, CovType::Hc1] {
            let input = ProbitInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string(), "x2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();

            let result =
                ProbitEstimator::fit(input, Method::Bfgs, 100, 1e-6, true, cov_type.clone(), 0.95);
            assert!(
                matches!(result, Err(MleError::SingularHessian)),
                "cov_type={:?}, result={:?}",
                cov_type,
                result
            );
        }
    }

    /// `cov_type=Opg`のエラー伝播（`opg_cov_params`が返す`SingularOpgMatrix`。
    /// `SingularHessian`とは別のエラー型）も、Hc0/Hc1と同じ完全な多重共線性データセットで
    /// 検証する。`scores_i=λᵢxᵢ`かつ`x2=2*x1`のため、スコア行列も`x1`と同じ構造的な
    /// 多重共線性を持ち（列2=2×列1）、OPG行列`Σsᵢsᵢ'`も特異になる
    /// （`LogitEstimator`の対応するテストと同じ理由、Issue #80）。
    #[test]
    fn fit_returns_singular_opg_matrix_error_for_perfectly_collinear_design_matrix() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 4.0, 6.0, 8.0]];
        let input = ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = ProbitEstimator::fit(input, Method::Bfgs, 100, 1e-6, true, CovType::Opg, 0.95);
        assert!(
            matches!(result, Err(MleError::SingularOpgMatrix)),
            "{:?}",
            result
        );
    }

    /// `cov_type=Cluster`のエラー伝播（`cluster_cov_params`も内部で`neg_hessian_inverse`を
    /// 呼ぶため`SingularHessian`）も、Hc0/Hc1と同じ完全な多重共線性データセットで検証する
    /// （`LogitEstimator`の対応するテストと同じ理由、Issue #80）。
    #[test]
    fn fit_returns_singular_hessian_error_for_perfectly_collinear_design_matrix_with_cluster() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 4.0, 6.0, 8.0]];
        let groups = vec![
            "g1".to_string(),
            "g1".to_string(),
            "g2".to_string(),
            "g2".to_string(),
        ];
        let input = ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = ProbitEstimator::fit(
            input,
            Method::Bfgs,
            100,
            1e-6,
            true,
            CovType::Cluster {
                groups: Some(groups),
            },
            0.95,
        );
        assert!(
            matches!(result, Err(MleError::SingularHessian)),
            "{:?}",
            result
        );
    }

    #[test]
    fn fit_returns_non_convergence_error_when_max_iter_is_too_small_and_raise_is_true() {
        let result = ProbitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            1,
            1e-12,
            true,
            CovType::Classical,
            0.95,
        );
        assert!(
            matches!(result, Err(MleError::NonConvergence { .. })),
            "{:?}",
            result
        );
    }

    #[test]
    fn fit_returns_unconverged_result_without_raising_when_raise_on_non_convergence_is_false() {
        let estimator = ProbitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            1,
            1e-12,
            false,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert!(!estimator.converged());
    }

    /// `at="overall"`（AME）のヤコビアン（`dydx_and_jacobian`が`overall_w_and_s`の
    /// 出力から計算する`∂dydx_j/∂θ_m`）を、`overall_w_and_s`が返す`w`を`θ`の関数として
    /// 中心差分で数値微分した値と比較する（`LogitEstimator`の対応するテストと同じ技法。
    /// Probitでは`w=φ(z)`ベースの式になる点が異なる）。
    #[test]
    fn dydx_and_jacobian_matches_numerical_differentiation_for_overall_w_and_s() {
        let x = Mat::from_fn(4, 3, |i, j| match j {
            0 => 1.0,
            1 => [10.0, 20.0, 30.0, 40.0][i],
            _ => [-5.0, 2.0, 8.0, -1.0][i],
        });
        let params = vec![0.3, -0.02, 0.05];
        let k = 3;
        let h = 1e-6;

        let (w, s) = overall_w_and_s(&x, &params);
        let (_, jacobian) = dydx_and_jacobian(k, &params, w, &s);

        for j in 0..k {
            for m in 0..k {
                let mut plus = params.clone();
                plus[m] += h;
                let mut minus = params.clone();
                minus[m] -= h;
                let (w_plus, _) = overall_w_and_s(&x, &plus);
                let (w_minus, _) = overall_w_and_s(&x, &minus);
                let dydx_plus_j = w_plus * plus[j];
                let dydx_minus_j = w_minus * minus[j];
                let numeric = (dydx_plus_j - dydx_minus_j) / (2.0 * h);
                assert!(
                    (*jacobian.get(j, m) - numeric).abs() < 1e-6,
                    "j={j}, m={m}, analytic={}, numeric={numeric}",
                    *jacobian.get(j, m)
                );
            }
        }
    }

    /// `at="mean"`/`"median"`（`at_point_w_and_s`、代表点`x_bar`固定）のヤコビアンについても、
    /// 上記と同じ数値微分による独立検証を行う。
    #[test]
    fn dydx_and_jacobian_matches_numerical_differentiation_for_at_point_w_and_s() {
        let x_bar = vec![1.0, 25.0, 1.0];
        let params = vec![0.3, -0.02, 0.05];
        let k = 3;
        let h = 1e-6;

        let (w, s) = at_point_w_and_s(&x_bar, &params);
        let (_, jacobian) = dydx_and_jacobian(k, &params, w, &s);

        for j in 0..k {
            for m in 0..k {
                let mut plus = params.clone();
                plus[m] += h;
                let mut minus = params.clone();
                minus[m] -= h;
                let (w_plus, _) = at_point_w_and_s(&x_bar, &plus);
                let (w_minus, _) = at_point_w_and_s(&x_bar, &minus);
                let dydx_plus_j = w_plus * plus[j];
                let dydx_minus_j = w_minus * minus[j];
                let numeric = (dydx_plus_j - dydx_minus_j) / (2.0 * h);
                assert!(
                    (*jacobian.get(j, m) - numeric).abs() < 1e-6,
                    "j={j}, m={m}, analytic={}, numeric={numeric}",
                    *jacobian.get(j, m)
                );
            }
        }
    }

    /// 切片のみモデル（k=1、`k_constant=1`）は、限界効果の出力対象となる説明変数が
    /// 存在しない（定数項は出力から除外するため）。`marginal_effects`が空の結果を
    /// 返す（パニックしない）ことを確認する境界ケース。
    #[test]
    fn marginal_effects_returns_empty_result_for_intercept_only_model() {
        let estimator = ProbitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let effects = estimator
            .marginal_effects(MarginalEffectsAt::Overall, 0.95)
            .unwrap();
        assert!(effects.param_names().is_empty());
        assert!(effects.dydx().is_empty());
    }

    /// `marginal_effects(at="overall")`の`dydx`を、実装の内部ヘルパー（`overall_w_and_s`/
    /// `dydx_and_jacobian`）とは別に、定義式`dy/dx_j = (1/n)Σᵢφ(zᵢ)θⱼ`を`Normal::standard()`
    /// から直接計算する式で独立に再計算し、突き合わせる（`LogitEstimator`の対応する
    /// テストと同じ技法。標準誤差は`dydx_j`自体をfit済みパラメータの周りで数値微分した
    /// ヤコビアンで独立に求め、解析的な標準誤差と突き合わせる）。
    #[test]
    fn marginal_effects_overall_matches_independently_recomputed_dydx_and_delta_method_se() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = ProbitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        let k = 3;
        let n = 4;
        let x = estimator.input().x();

        let effects = estimator
            .marginal_effects(MarginalEffectsAt::Overall, 0.95)
            .unwrap();
        assert_eq!(effects.param_names(), ["x1".to_string(), "x2".to_string()]);

        // dydxの独立再計算（`Normal::standard().pdf`から直接、`overall_w_and_s`とは別の式）
        let normal = Normal::standard();
        let dydx_j = |params: &[f64], j: usize| -> f64 {
            (0..n)
                .map(|i| {
                    let z: f64 = (0..k).map(|m| *x.get(i, m) * params[m]).sum();
                    normal.pdf(z)
                })
                .sum::<f64>()
                / (n as f64)
                * params[j]
        };
        let params = estimator.params();
        for (idx, j) in (1..k).enumerate() {
            assert!((effects.dydx()[idx] - dydx_j(params, j)).abs() < 1e-9);
        }

        // 標準誤差の独立検証: `dydx_j`をfit済みパラメータの周りで数値微分して
        // ヤコビアン行（j行目）を求め、`cov_params`との二次形式で分散を計算する。
        let h = 1e-6;
        for (idx, j) in (1..k).enumerate() {
            let mut jac_row = vec![0.0; k];
            for m in 0..k {
                let mut plus = params.to_vec();
                plus[m] += h;
                let mut minus = params.to_vec();
                minus[m] -= h;
                jac_row[m] = (dydx_j(&plus, j) - dydx_j(&minus, j)) / (2.0 * h);
            }
            let mut var_j = 0.0;
            for a in 0..k {
                for b in 0..k {
                    var_j += jac_row[a] * (*estimator.cov_params().get(a, b)) * jac_row[b];
                }
            }
            let expected_se = var_j.sqrt();
            assert!(
                (effects.std_errors()[idx] - expected_se).abs() < 1e-6,
                "idx={idx}, actual={}, expected={expected_se}",
                effects.std_errors()[idx]
            );
        }

        // z値・p値・信頼区間の内部整合性（既存のSE系テストと同じ検算パターン）
        let z_crit = normal.inverse_cdf(0.975);
        for idx in 0..2 {
            let se = effects.std_errors()[idx];
            assert!((effects.z_stats()[idx] - effects.dydx()[idx] / se).abs() < 1e-9);
            let expected_p = 2.0 * (1.0 - normal.cdf(effects.z_stats()[idx].abs()));
            assert!((effects.p_values()[idx] - expected_p).abs() < 1e-9);
            assert!(
                (effects.conf_upper()[idx] - effects.conf_lower()[idx] - 2.0 * z_crit * se).abs()
                    < 1e-9
            );
        }
    }

    /// `at="mean"`は`at="overall"`と異なる代表点（標本平均）で評価するため、一般には
    /// 異なる値になる。実装がこの違いを正しく反映していること（`at`の分岐が機能して
    /// いること）を確認する。`dydx`を`column_means`から独立に再計算した値とも突き合わせる。
    #[test]
    fn marginal_effects_at_mean_differs_from_overall_and_matches_independent_recomputation() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = ProbitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let overall = estimator
            .marginal_effects(MarginalEffectsAt::Overall, 0.95)
            .unwrap();
        let at_mean = estimator
            .marginal_effects(MarginalEffectsAt::Mean, 0.95)
            .unwrap();

        assert!((overall.dydx()[0] - at_mean.dydx()[0]).abs() > 1e-9);

        // 独立再計算: x̄=[1, 25, 1]（定数項1、x1の平均25、x2の平均1）でφ(z̄)を評価
        let params = estimator.params();
        let x_bar = [1.0, 25.0, 1.0];
        let z_bar: f64 = (0..3).map(|m| x_bar[m] * params[m]).sum();
        let w = Normal::standard().pdf(z_bar);
        for (idx, j) in (1..3).enumerate() {
            assert!((at_mean.dydx()[idx] - w * params[j]).abs() < 1e-9);
        }
    }

    /// `at="median"`が`at="mean"`/`at="overall"`と異なる代表点で評価されること
    /// （非対称なデータセットで平均・中央値が異なる値になるよう構成）、および
    /// `dydx`を`column_medians`から独立に再計算した値と突き合わせる。
    #[test]
    fn marginal_effects_at_median_differs_from_mean_and_overall_and_matches_independent_recomputation()
     {
        // x1: 平均40（=200/5）・中央値30（sorted: 10,20,30,40,100）と非対称にずらす
        let y = vec![0.0, 1.0, 0.0, 1.0, 1.0];
        let x_columns = vec![
            vec![10.0, 20.0, 30.0, 40.0, 100.0],
            vec![-5.0, 2.0, 8.0, -1.0, 50.0],
        ];
        let input = ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = ProbitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let overall = estimator
            .marginal_effects(MarginalEffectsAt::Overall, 0.95)
            .unwrap();
        let at_mean = estimator
            .marginal_effects(MarginalEffectsAt::Mean, 0.95)
            .unwrap();
        let at_median = estimator
            .marginal_effects(MarginalEffectsAt::Median, 0.95)
            .unwrap();

        assert!((at_median.dydx()[0] - at_mean.dydx()[0]).abs() > 1e-9);
        assert!((at_median.dydx()[0] - overall.dydx()[0]).abs() > 1e-9);

        // 独立再計算: x̄=[1, 30, 2]（定数項1、x1の中央値30、x2の中央値2）でφ(z̄)を評価
        let params = estimator.params();
        let x_bar = [1.0, 30.0, 2.0];
        let z_bar: f64 = (0..3).map(|m| x_bar[m] * params[m]).sum();
        let w = Normal::standard().pdf(z_bar);
        for (idx, j) in (1..3).enumerate() {
            assert!((at_median.dydx()[idx] - w * params[j]).abs() < 1e-9);
        }
    }

    #[test]
    fn marginal_effects_returns_invalid_confidence_level_error_out_of_range() {
        let estimator = ProbitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let result = estimator.marginal_effects(MarginalEffectsAt::Overall, 1.5);
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5
            })
        );
    }

    /// 切片のみモデルは全観測で`p_i=ȳ`（closed form、他のテストと同じ性質）なので、
    /// `predict()`が返す予測確率が`ȳ`と一致することを検証できる
    /// （`LogitEstimator`の対応するテストと同じ技法）。
    #[test]
    fn predict_matches_closed_form_for_intercept_only_model() {
        let estimator = ProbitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let y_bar: f64 = 4.0 / 7.0;
        let predicted = estimator.predict();
        assert_eq!(predicted.len(), 7);
        for p in predicted {
            assert!((p - y_bar).abs() < 1e-6);
        }
    }

    /// 多変量モデルでは`predict()`に閉じた形の解析解が無いため、`Normal::standard().cdf`
    /// から直接`p_i=Φ(x_i'θ)`を計算する式で独立に再計算し、突き合わせる
    /// （`LogitEstimator`の対応するテストの`logistic`をΦに置き換えたProbit版）。
    #[test]
    fn predict_matches_independently_recomputed_normal_cdf_of_linear_predictor() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = ProbitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let params = estimator.params();
        let x = estimator.input().x();
        let n = 4;
        let k = 3;
        let normal = Normal::standard();
        let predicted = estimator.predict();
        for (i, &p_i) in predicted.iter().enumerate().take(n) {
            let z: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            assert!((p_i - normal.cdf(z)).abs() < 1e-12);
        }
    }

    /// 切片のみモデルは全観測で`p_i=ȳ=4/7≈0.571`（closed form）のため、`threshold`に
    /// よって全観測が一方のクラスに分類される自明なケースになる。この性質を使い、
    /// `pred_table`の的中表を手計算で検証する（`y=[0,0,0,1,1,1,1]`、実測は`y_i>=0.5`で
    /// 二値化。`LogitEstimator`の対応するテストと同じ技法）。
    #[test]
    fn pred_table_matches_hand_computed_counts_for_intercept_only_model() {
        let estimator = ProbitEstimator::fit(
            intercept_only_input(),
            Method::Newton,
            35,
            1e-6,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        // threshold=0.5: p_i=4/7>0.5 なので全観測が予測クラス1。
        // 実測は y=[0,0,0,1,1,1,1] なので actual0が3件・actual1が4件。
        let table_low = estimator.pred_table(0.5);
        assert!((*table_low.get(0, 0) - 0.0).abs() < 1e-12);
        assert!((*table_low.get(0, 1) - 3.0).abs() < 1e-12);
        assert!((*table_low.get(1, 0) - 0.0).abs() < 1e-12);
        assert!((*table_low.get(1, 1) - 4.0).abs() < 1e-12);

        // threshold=0.99: p_i=4/7<0.99 なので全観測が予測クラス0。
        let table_high = estimator.pred_table(0.99);
        assert!((*table_high.get(0, 0) - 3.0).abs() < 1e-12);
        assert!((*table_high.get(0, 1) - 0.0).abs() < 1e-12);
        assert!((*table_high.get(1, 0) - 4.0).abs() < 1e-12);
        assert!((*table_high.get(1, 1) - 0.0).abs() < 1e-12);
    }

    /// `pred_table`が返すカウントの総和が観測数`n`と一致すること、および`predict()`の
    /// 出力から独立に再計算した分類結果（同じ`threshold`での二値化）と一致することを
    /// 多変量モデルで検証する（`LogitEstimator`の対応するテストと同じ技法）。
    #[test]
    fn pred_table_matches_independently_recomputed_classification() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = ProbitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        // `threshold≠0.5`にする（`actual`側は`threshold`に依存せず常に0.5固定である
        // ことを検出できるようにするため）。
        let threshold = 0.2;
        let predicted = estimator.predict();
        let table = estimator.pred_table(threshold);

        let mut expected = [[0.0; 2]; 2];
        for i in 0..4 {
            let actual = usize::from(y[i] >= 0.5);
            let pred = usize::from(predicted[i] > threshold);
            expected[actual][pred] += 1.0;
        }

        let mut total = 0.0;
        for (a, row) in expected.iter().enumerate() {
            for (p, &expected_count) in row.iter().enumerate() {
                assert!((*table.get(a, p) - expected_count).abs() < 1e-12);
                total += *table.get(a, p);
            }
        }
        assert!((total - 4.0).abs() < 1e-12);
    }

    /// `pred_table`の実測クラス（行方向の合計）が`threshold`の値に関わらず不変である
    /// 性質（`common.rs`の`pred_table`のdocコメント参照）を、`ProbitEstimator::predict`→
    /// `pred_table`という実際の呼び出し経路を通して検証する。この性質自体は`common.rs`
    /// 側の合成データによる一般テスト（`pred_table_actual_class_counts_are_invariant_
    /// to_threshold`）で既にカバーされているが、`ProbitEstimator`側の配線（`predict()`の
    /// 出力を正しく`pred_table`へ渡せているか）を壊す変更を検知できるようにするため、
    /// `LogitEstimator`の対応するテストと対称に追加する（rust-reviewer指摘、Issue #79）。
    #[test]
    fn pred_table_actual_class_counts_are_invariant_to_threshold() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = ProbitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = ProbitEstimator::fit(
            input,
            Method::Newton,
            35,
            1e-8,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let table = estimator.pred_table(threshold);
            let actual0 = *table.get(0, 0) + *table.get(0, 1);
            let actual1 = *table.get(1, 0) + *table.get(1, 1);
            // y=[0,1,0,1] → actual0=2件・actual1=2件（`threshold`に関わらず常に一定）
            assert!(
                (actual0 - 2.0).abs() < 1e-12,
                "threshold={threshold}, actual0={actual0}"
            );
            assert!(
                (actual1 - 2.0).abs() < 1e-12,
                "threshold={threshold}, actual1={actual1}"
            );
        }
    }
}
