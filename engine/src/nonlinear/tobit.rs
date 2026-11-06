//! Tobitの推定量`TobitEstimator`（`fit`の契約とgetter）。入力データ（被説明変数・設計行列・
//! 打ち切り境界）、負の対数尤度・スコア・Hessian、標準化、限界効果・予測、予測・限界効果に必要な
//! 最小の状態`TobitFitted`、`fit`の本体は子モジュールに置く。
//!
//! `engine`はpolars/PyO3を一切知らない（`.claude/rules/rust-style.md`「責務分離」参照）。
//! `engine_pybind`はpolars DataFrameから列ごとに`Vec<f64>`を抽出するところまでを担い
//! （`column_extraction::extract_f64_column`）、それらの列を
//! `TobitInput::from_columns`（`tobit/input.rs`）に渡す。`faer::Mat`への組み立て（切片列の自動追加を含む）は
//! ここ（engine側）の責務とする。`LogitInput`/`ProbitInput`と同型の設計だが、Tobitは
//! 打ち切り境界（`lower`/`upper`）という他の2手法にはない引数・検証を追加で持つ
//! （`docs/spec/nonlinear-common.md`7章「Tobitの打ち切り境界オプション」）。
//!
//! ## Logit/Probitとの設計上の違い: 検証の実施箇所
//!
//! Logit/Probitの`validate_binary_y`（`nonlinear/common/validation.rs`）は`fit()`冒頭で呼ばれ、
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
//! リスクを持つ（`u`が極端に負のとき）。`nonlinear/common/normal.rs`の`clamped_pdf_cdf`
//! （Tobit実装時にProbit専用から共有ユーティリティへ移設）をそのまま再利用する。
//!
//! ## Newton法の初期値: OLS推定値を使う（ゼロベクトルではない、Logit/Probitとの違い）
//!
//! 当初計画（`(β, logσ)`パラメータ化でもLogit/Probitと同じくゼロベクトル初期値から
//! Newton収束をまず試す）では、**打ち切りが皆無のデータ（通常の正規回帰に退化する
//! 単純なケース）でもゼロベクトル初期値からNewtonが発散する**ことが実測で判明した
//! （切片`β→-∞`・`s=logσ→+∞`という非有界な方向へ発散し、最終的にHessianが0行列に
//! なり`NaN`に到達）。
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
//! `nonlinear/common/newton.rs`の`FaerNewton`（Logit/Probitと共有）に、Levenberg-Marquardt型の
//! 減衰Newton法（`regularized_newton_step`）を追加して対応した。`H+λI`（`λ≥0`）で
//! ステップを求め、`cost`が減少する候補が見つかるまで`λ`を段階的に増やす。
//! Logit/Probitのように尤度が大域凹な問題では、収束点に向かう正常な軌道上は`λ=0`の
//! 生のステップが最初の試行で受理されるため収束の挙動は変わらないが、設計行列が
//! 構造的に特異な入力（完全な多重共線性等）では内部の反復過程が変化する
//! （最終的に`fit()`が返すエラーバリアントは変わらない。詳細は
//! `regularized_newton_step`のdocコメント参照、独立シミュレーションで確認済み）。
//! Logit/Probitの既存テストが数値的に変化なく全通過することは確認済み。
//! 詳細な導入経緯は`regularized_newton_step`のdocコメント参照。
//!
//! # ファイル構成
//!
//! このファイルは外から見える面（`TobitEstimator`の構造体・getter・`fit`の契約と委譲・
//! 再エクスポート）だけを持つ。ロジックは子モジュールに置き、テストは各ロジックと同じ
//! ファイルの`mod tests`に置く（`fit`を通すテストだけは
//! `fit/tests/`にテーマ別で置く）。
//!
//! - `input.rs`: `TobitInput`
//! - `problem.rs`: `TobitProblem`（`CostFunction`/`Gradient`/`Hessian`）と観測ごとの尤度の寄与
//! - `scaling.rs`: `TobitScaling`（標準化と逆変換）
//! - `statistics.rs`: OLS初期値・対数尤度・Wald検定・非打ち切り観測の検査
//! - `mills.rs`: 正規分布の裾の安定な計算（逆ミルズ比・切断正規の密度比）
//! - `target.rs`: `MarginalEffectsTarget`・`predicted_value`・`w`/`s`
//! - `marginal_effects.rs`: 限界効果（`overall`・`mean`・`median`）の計算
//! - `censoring_check.rs`: `CensoringFitCheck`
//! - `fitted.rs`: `TobitFitted`（`from_parts`・限界効果・予測・打ち切り適合度）
//! - `fit.rs`: `fit`の本体
//! - `test_support.rs`: 複数ファイルのテストが共有するフィクスチャ（`#[cfg(test)]`）

use faer::Mat;

use crate::nonlinear::common::{MarginalEffects, MarginalEffectsAt, MleError, MleFitOptions};
use crate::shared::inference;

mod censoring_check;
mod fit;
mod fitted;
mod input;
mod marginal_effects;
mod mills;
mod problem;
mod scaling;
mod statistics;
mod target;
#[cfg(test)]
mod test_support;

pub use censoring_check::{CensoringFitCategory, CensoringFitCheck};
pub use fitted::TobitFitted;
pub use input::TobitInput;
pub use problem::TobitProblem;
pub use target::MarginalEffectsTarget;

/// Tobitの推定結果。`fit`でのバリデーション・最適化・`cov_type`に応じたSE計算・
/// 適合度統計量・Wald検定・限界効果を通過した状態を表す。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug)]
pub struct TobitEstimator {
    /// 予測・限界効果・打ち切りの当てはまりの確認に必要な状態（入力データ・係数・`σ`・`(β, σ)`の分散共分散行列）。
    fitted: TobitFitted,
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
    /// （`Hc0`/`Hc1`）・クラスターロバスト（`Cluster`）に対応する（Probitと同様に
    /// まとめて実装している。`CovType`はLogit/Probit/Tobit共有の1つのenumで既に
    /// `Cluster`バリアントを含んでおり、`match cov_type`を網羅的にする都合上、
    /// OPG/サンドイッチのみを実装してクラスターを未実装のまま残す設計は取れない）。
    /// `Opg`/`Hc0`/`Hc1`/`Cluster`は収束点での観測ごとのスコア（`TobitProblem::scores`）が必要なため、標準化空間の設計行列を
    /// 保持したまま`TobitProblem`をクローンしておき（`Classical`のときは不要な複製を
    /// 避けるため条件付きで行う、`LogitEstimator::fit`と同じ理由）、`run_solver`が返す
    /// 収束点のパラメータで評価する。観測数の十分性検証（`validate_sufficient_
    /// observations`）には`x`の列数`k`ではなく総最適化パラメータ数`k+1`を使う
    /// （`validate_sufficient_observations`のdocコメント参照）。
    /// Logit/Probitの`validate_has_regressors`（`k==0`検証）はTobitでは呼ばない
    /// （`logσ`が常に存在するため対応するケースが生じない、同関数のdocコメント参照）。
    /// 推定オプションの検証は3手法共通の`validate_mle_options`を使い、Logit/Probit用の
    /// `validate_fit_preconditions`は呼ばない（手法ごとの違いは同関数のdocコメント参照）。
    /// 打ち切り境界の検証（`MleError::CensoringBoundsMissing`等・`YOutOfCensoringBounds`）は
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
    ///   Wald検定の`q×q`部分行列が構造的に特異。`wald_chi2_test`内の
    ///   `NearSingularMatrix`（`ComputationError`）に任せず`fit()`冒頭のバリデーションで弾く）
    /// - OLS初期値計算時に`x`が特異（完全な多重共線性等）: `MleError::SingularDesignMatrix`
    ///   （`ols_initial_params`参照）
    /// - `raise_on_non_convergence=true`かつ`max_iter`回で未収束: `MleError::NonConvergence`
    /// - 収束点（または`raise_on_non_convergence=false`時の打ち切り点）のHessianが特異:
    ///   `MleError::SingularHessian`
    /// - `cov_type=Opg`でOPG行列（`Σᵢ sᵢsᵢ'`）が特異: `MleError::SingularOpgMatrix`
    /// - 傾き係数間の悪条件（極端なスケール差等）でWald検定用の係数分散共分散部分行列が
    ///   数値的にほぼ特異: `CommonError::NearSingularMatrix`（`wald_chi2_test`参照。`g > q`
    ///   でも起こりうるbackstop。`df_model==0`（切片以外の`β`が無い）のときはこの検定自体を
    ///   スキップし`wald_statistic`/`wald_p_value`はNaNになる）
    pub fn fit(input: TobitInput, options: MleFitOptions) -> Result<Self, MleError> {
        fit::fit_tobit(input, options)
    }

    /// 推定に使った入力データ
    pub fn input(&self) -> &TobitInput {
        self.fitted.input()
    }

    /// 係数（元のスケール、`β`部分のみ）
    pub fn params(&self) -> &[f64] {
        self.fitted.params()
    }

    /// 誤差項の標準偏差（元のスケール）
    pub fn sigma(&self) -> f64 {
        self.fitted.sigma()
    }

    /// `(β, σ)`の分散共分散行列（元のスケール、`(k+1)×(k+1)`。`k`番目の行・列が`σ`に対応）
    pub fn cov_params(&self) -> &Mat<f64> {
        self.fitted.cov_params()
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

    /// 限界効果。実装は[`TobitFitted::marginal_effects`]。
    pub fn marginal_effects(
        &self,
        at: MarginalEffectsAt,
        target: MarginalEffectsTarget,
        confidence_level: f64,
    ) -> Result<MarginalEffects, MleError> {
        self.fitted.marginal_effects(at, target, confidence_level)
    }

    /// 学習データの予測。実装は[`TobitFitted::predict`]。
    pub fn predict(&self, target: MarginalEffectsTarget) -> Result<Vec<f64>, MleError> {
        self.fitted.predict(target)
    }

    /// 新規データの予測。実装は[`TobitFitted::predict_new_data`]。
    pub fn predict_new_data(
        &self,
        target: MarginalEffectsTarget,
        new_x_columns: &[Vec<f64>],
    ) -> Result<Vec<f64>, MleError> {
        self.fitted.predict_new_data(target, new_x_columns)
    }

    /// 打ち切りの当てはまりの確認。実装は[`TobitFitted::censoring_fit_check`]。
    pub fn censoring_fit_check(&self) -> CensoringFitCheck {
        self.fitted.censoring_fit_check()
    }

    /// 予測・限界効果・打ち切りの当てはまりの確認に必要な状態だけを取り出す。結果オブジェクトが
    /// 推定量全体（標準誤差や適合度統計量。結果側に値のコピーがある）を持たずに済む。
    pub fn into_fitted(self) -> TobitFitted {
        self.fitted
    }
}
