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
use crate::nonlinear::common::{
    Method, MleError, destandardize_params, run_solver, standardize_columns, validate_binary_y,
};
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
/// （`U_CLAMP`のdocコメント参照）。`cost`/`linear_predictor_and_residual`の両方が
/// 経由する共通の関所にすることで、`statsmodels`の`Probit`実装に見られる非対称性
/// （`score`/`loglike`はクリップするが`hessian`はしない）を避ける。
fn clamped_pdf_cdf(normal: &Normal, u: f64) -> (f64, f64) {
    let u = u.clamp(-U_CLAMP, U_CLAMP);
    (normal.pdf(u), normal.cdf(u))
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
        let normal = Normal::standard();
        let n = self.x.nrows();
        let cost: f64 = (0..n)
            .map(|i| {
                let z = self.linear_predictor(i, param);
                let q = 2.0 * (*self.y.get(i, 0)) - 1.0;
                let (_, big_phi) = clamped_pdf_cdf(&normal, q * z);
                -big_phi.ln()
            })
            .sum();
        Ok(cost)
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

/// Probitの推定結果（現時点では骨格のみ）。`fit`でのバリデーション・Newton-Raphsonでの
/// 最適化を通過した状態を表す。
///
/// 標準誤差・z値・p値・信頼区間・適合度統計量等は未実装（本Issue #72の完了条件は
/// Newton-Raphsonでの最適化・収束判定のみ、`LogitEstimator`の骨格実装（Issue #56）と
/// 同じ切り分け）。`docs/planning/specs/logit-probit-issue-breakdown.md`の対応する
/// 後続Issueで`fit`に追加していく想定。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug)]
pub struct ProbitEstimator {
    input: ProbitInput,
    /// 係数（元のスケール。`standardize_columns`で標準化した空間で最適化した後、
    /// `destandardize_params`で逆変換済み）。`input.param_names()`と対応する
    params: Vec<f64>,
    /// 収束したかどうか
    converged: bool,
    /// 実際の反復回数
    n_iter: usize,
}

impl ProbitEstimator {
    /// `method`（Newton-Raphson/BFGS/L-BFGS）で負の対数尤度を最小化し、Probitの係数を
    /// 推定する。`LogitEstimator::fit`（Issue #56の骨格実装＋Issue #57のmethod分岐）と
    /// 同じ設計・スコープ。
    ///
    /// `method`の選択に関わらず、収束点でのHessian評価（SE計算用、後続Issueで使用）は
    /// 常に解析的に行う（`run_solver`の実装方針、`docs/planning/specs/
    /// nonlinear-implementation-notes.md`「engine内のtrait設計」参照）。BFGS/L-BFGSが
    /// 最適化中に内部で保持する近似Hessianは使い回さない。
    ///
    /// 初期値は常にゼロベクトル（`start_params`によるユーザー指定は未対応、
    /// `LogitEstimator::fit`と同じ理由でユーザー確認の上見送り）。
    ///
    /// 設計行列は`standardize_columns`で内部的に標準化してから最適化し、収束後の
    /// パラメータを`destandardize_params`で元のスケールへ逆変換する
    /// （`LogitEstimator::fit`のdocコメント参照）。
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
    pub fn fit(
        input: ProbitInput,
        method: Method,
        max_iter: i64,
        tol: f64,
        raise_on_non_convergence: bool,
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

        let (x_std, scale) = standardize_columns(input.x(), input.has_intercept());
        let problem = ProbitProblem::from_standardized(x_std, input.y().clone());

        let output = run_solver(
            problem,
            method,
            vec![0.0; k],
            max_iter as u64,
            tol,
            raise_on_non_convergence,
        )?;

        let params = destandardize_params(&output.params, &scale);

        Ok(Self {
            input,
            params,
            converged: output.converged,
            n_iter: output.n_iter,
        })
    }

    /// 推定に使った入力データ
    pub fn input(&self) -> &ProbitInput {
        &self.input
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let estimator = ProbitEstimator::fit(input, Method::Newton, 35, 1e-6, true, 0.95).unwrap();

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

    /// `newton`と同じデータセット（既知の解析解を持つ切片のみモデル）で`bfgs`/`lbfgs`を
    /// 実行し、いずれも同じ解析解へ収束することを検証する（Issue #73完了条件）。
    #[test]
    fn fit_bfgs_and_lbfgs_converge_to_same_solution_as_newton() {
        let y_bar: f64 = 4.0 / 7.0;
        let expected = Normal::standard().inverse_cdf(y_bar);

        for method in [Method::Bfgs, Method::Lbfgs] {
            let estimator =
                ProbitEstimator::fit(intercept_only_input(), method, 100, 1e-6, true, 0.95)
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

        let newton =
            ProbitEstimator::fit(make_input(), Method::Newton, 35, 1e-8, true, 0.95).unwrap();
        assert!(newton.converged());

        for method in [Method::Bfgs, Method::Lbfgs] {
            let estimator =
                ProbitEstimator::fit(make_input(), method, 200, 1e-8, true, 0.95).unwrap();

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
        let result =
            ProbitEstimator::fit(intercept_only_input(), Method::Newton, 35, 1e-6, true, 1.5);
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5
            })
        );
    }

    #[test]
    fn fit_returns_invalid_max_iter_error_for_non_positive_max_iter() {
        let result =
            ProbitEstimator::fit(intercept_only_input(), Method::Newton, 0, 1e-6, true, 0.95);
        assert_eq!(
            result.unwrap_err(),
            MleError::InvalidMaxIter { max_iter: 0 }
        );
    }

    #[test]
    fn fit_returns_invalid_tol_error_for_non_positive_tol() {
        for tol in [0.0, -1.0] {
            let result =
                ProbitEstimator::fit(intercept_only_input(), Method::Newton, 35, tol, true, 0.95);
            assert_eq!(result.unwrap_err(), MleError::InvalidTol { tol });
        }
    }

    #[test]
    fn fit_returns_invalid_binary_y_error_for_non_binary_y() {
        let y = vec![0.0, 1.0, 0.5, 1.0];
        let input = ProbitInput::from_columns(&y, &[], vec![], true, "y".to_string()).unwrap();
        let result = ProbitEstimator::fit(input, Method::Newton, 35, 1e-6, true, 0.95);
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

        let result = ProbitEstimator::fit(input, Method::Newton, 35, 1e-6, true, 0.95);
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

        let result = ProbitEstimator::fit(input, Method::Newton, 35, 1e-6, true, 0.95);
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

        let result = ProbitEstimator::fit(input, Method::Newton, 35, 1e-6, true, 0.95);
        assert!(
            matches!(result, Err(MleError::SingularHessian)),
            "{:?}",
            result
        );
    }

    #[test]
    fn fit_returns_non_convergence_error_when_max_iter_is_too_small_and_raise_is_true() {
        let result =
            ProbitEstimator::fit(intercept_only_input(), Method::Newton, 1, 1e-12, true, 0.95);
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
            0.95,
        )
        .unwrap();
        assert!(!estimator.converged());
    }
}
