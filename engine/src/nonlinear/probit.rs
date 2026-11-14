//! Probitの推定量`ProbitEstimator`（`fit`の契約とgetter）。入力データ（被説明変数・設計行列）、
//! 負の対数尤度・スコア・Hessian（argminの`CostFunction`/`Gradient`/`Hessian`トレイト実装）、
//! 予測・限界効果に必要な最小の状態`ProbitFitted`、`fit`の本体は子モジュールに置く。
//!
//! `engine`はpolars/PyO3を一切知らない（`.claude/rules/rust-style.md`「責務分離」参照）。
//! `engine_pybind`はpolars DataFrameから列ごとに`Vec<f64>`を抽出するところまでを担い
//! （`column_extraction::extract_f64_column`）、それらの列を
//! `ProbitInput::from_columns`（`probit/input.rs`）に渡す。`faer::Mat`への組み立て（切片列の自動追加を含む）は
//! ここ（engine側）の責務とする。`engine::nonlinear::logit::LogitInput`とほぼ同型の設計。
//!
//! OLS/Logitと同様、Phase2（Logit/Probit/Tobit）では`weights`/`offset`を見送っているため
//! （`docs/spec/nonlinear-common.md`7章）、`from_columns_weighted`に相当するものはない。
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
//! ともに0にアンダーフローし`0.0/0.0`のNaNになる（`Logit`の`logistic`/`softplus`が
//! 有限の`z`ではどれだけ極端でも絶対にNaNを産まない設計だったのとは異なる、Probit固有の
//! リスク）。既定手法の
//! `SolverType::Newton`（`FaerNewton`）はline searchなしで`gradient`/`hessian`を直接
//! 使うため、Logitで実際に問題になった「(準)完全分離データでの収束判定誤検知」
//! （勾配ノルムのアンダーフロー、`docs/spec/logit-spec.md`3.2節参照）よりも
//! 緩い条件でこのNaN汚染に到達しうる。
//!
//! 対策として、`u`を`φ`/`Φ`評価前にクランプする（`nonlinear/common.rs`の`clamped_pdf_cdf`・
//! `U_CLAMP`のdocコメント参照。Tobitの打ち切り観測の尤度も同型の`λ=φ/Φ`（逆ミルズ比）を
//! 含み同じリスクを共有するため、Tobit実装時に共通化した）。`cost`/`gradient`/`hessian`/
//! `scores`すべてに同じ関所を経由させることで、statsmodelsの`Probit`実装に見られる
//! 非対称性（`score`/`loglike`はクリップするが`hessian`はしない）を避けている。
//!
//! **Hessianの重み`w=λᵢ(λᵢ+zᵢ)`の`zᵢ`も、`λᵢ`と同じクランプ済み引数から再構成する
//! 必要がある**: `λᵢ`はクランプ後の`u`から計算されるため、`|u|>U_CLAMP`
//! の領域では`zᵢ`に対して事実上定数になる。ここで生の（非クランプの）`zᵢ`を`w`の
//! 計算に混ぜると、上記導出の前提（`λᵢ`が`zᵢ`の滑らかな関数であること）が崩れ、
//! `λᵢ(λᵢ+zᵢ)>0`という大域凹性の恒等式が数値的に破れて`w`が負になりうる（実測で
//! 悪条件パラメータ点にて確認済み）。`hessian_weight`は`λᵢ`と同じクランプ済み`u`から
//! `zᵢ`を再構成して重みを作っており、この非対称性を避けている。
//!
//! # ファイル構成
//!
//! このファイルは外から見える面（`ProbitEstimator`の構造体・getter・`fit`の契約と委譲・
//! 再エクスポート）だけを持つ。ロジックは子モジュールに置き、テストは各ロジックと同じ
//! ファイルの`mod tests`に置く（`fit`を通すテストと、テストが大きいファイルは
//! `<ファイル名>/tests/`にテーマ別で置く）。
//!
//! - `input.rs`: `ProbitInput`
//! - `model.rs`: 対数尤度、限界効果の`w`と`s`
//! - `problem.rs`: `ProbitProblem`（`CostFunction`/`Gradient`/`Hessian`）
//! - `fitted.rs`: `ProbitFitted`（`from_parts`・限界効果・予測・的中表）
//! - `fit.rs`: `fit`の本体
//! - `test_support.rs`: 複数ファイルのテストが共有するフィクスチャ（`#[cfg(test)]`）

use faer::Mat;

use crate::nonlinear::common::{MarginalEffects, MarginalEffectsAt, MleError, MleFitOptions};
use crate::shared::inference;

mod fit;
mod fitted;
mod input;
mod model;
mod problem;
#[cfg(test)]
mod test_support;

pub use fitted::ProbitFitted;
pub use input::ProbitInput;
pub use problem::ProbitProblem;

/// Probitの推定結果。`fit`でのバリデーション・最適化・`cov_type`に応じたSE計算・
/// 適合度統計量の計算を通過した状態を表す。
///
/// 限界効果等は`fit`とは別のメソッドとして提供する（`docs/spec/probit-spec.md`参照）。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug)]
pub struct ProbitEstimator {
    /// 予測・的中表・限界効果に必要な状態（入力データ・係数・係数の分散共分散行列）。
    fitted: ProbitFitted,
    /// 標準誤差（k, 元のスケール）。`cov_params`の対角成分の平方根
    std_errors: Vec<f64>,
    /// z統計量（k）= `params / std_errors`
    test_stats: Vec<f64>,
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
    /// （`docs/spec/nonlinear-common.md`5章の定義通り）。
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
    /// `solver`（Newton-Raphson/BFGS/L-BFGS）で負の対数尤度を最小化し、Probitの係数・
    /// 観測情報行列によるSE・z値・p値・信頼区間を推定する。`LogitEstimator::fit`
    /// （骨格実装＋solver分岐＋SE計算）と同じ設計・スコープ。
    ///
    /// `solver`の選択に関わらず、収束点でのHessian評価（SE計算用）は常に解析的に行う
    /// （`run_solver`の実装方針、`docs/spec/nonlinear-common.md`1.2節参照）。
    /// BFGS/L-BFGSが最適化中に内部で保持する近似Hessianは
    /// 使い回さない。
    ///
    /// 初期値（warm start）は標準化空間でのLPM最小二乗解に、probitのIRLS 1ステップ相当の
    /// スケール補正（`p̄=ȳ`での `1/φ(Φ⁻¹(p̄))` 倍＋切片補正）を施したもの
    /// （`ols_based_initial_params`。`LogitEstimator::fit`と同じ設計で、
    /// 従来のゼロベクトルから変更）。前段で`standardize_columns`後の設計行列を列ピボットQR
    /// しランク落ちを検出する（`checked_design_matrix_qr`、`solver`によらず単一経路で
    /// `SingularDesignMatrix`）。`start_params`によるユーザー指定初期値は引き続き未対応。
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
    /// パラメータで評価する。検定分布は標準正規分布（`docs/spec/nonlinear-common.md`4章、
    /// OLSのt分布とは異なる）。
    ///
    /// `n <= k`で`CommonError::InsufficientObservations`、`k == 0`で
    /// `CommonError::NoRegressors`を返す閾値・使い分けは`LogitEstimator::fit`と同じ
    /// （後者はLogitのfaer内部panicを修正した経緯があり、
    /// Probitでは当初から同じ検証を入れている）。
    ///
    /// `y`が`{0.0, 1.0}`の二値であることの検証（`validate_binary_y`）も、Logitでは
    /// 事後的に追加された経緯があるが、Probitでは
    /// 既に共有実装（`nonlinear/common.rs`）があるため当初から呼び出している。
    ///
    /// # Errors
    /// - `confidence_level`が`(0, 1)`の範囲外: `CommonError::InvalidConfidenceLevel`
    /// - `max_iter`が0以下: `MleError::InvalidMaxIter`
    /// - `tol`が0以下: `MleError::InvalidTol`
    /// - `y`が`{0.0, 1.0}`以外の値を含む: `MleError::InvalidBinaryY`
    /// - `k`（定数項を含む説明変数の数）が0（定数項も説明変数も無い）: `CommonError::NoRegressors`
    /// - 観測数`n`が`k`以下: `CommonError::InsufficientObservations`
    /// - 設計行列がランク落ち（完全な多重共線性等）: `MleError::SingularDesignMatrix`
    ///   （最適化前の列ピボットQRランクチェックで`solver`によらず検出）
    /// - `raise_on_non_convergence=true`かつ`max_iter`回で未収束: `MleError::NonConvergence`
    /// - 収束点（または`raise_on_non_convergence=false`時の打ち切り点）のHessianが特異:
    ///   `MleError::SingularHessian`（ランク落ちは前段で`SingularDesignMatrix`として弾くため、
    ///   ここに至るのは丸め誤差境界での不定符号Hessian等）
    /// - `cov_type=Opg`でOPG行列（`Σᵢ sᵢsᵢ'`）が特異: `MleError::SingularOpgMatrix`
    /// - `cov_type=Cluster`でグループキー未指定: `CommonError::MissingClusterColumn`
    /// - `cov_type=Cluster`でクラスター数が2未満: `CommonError::InsufficientClusters`
    /// - `cov_type=Cluster`でクラスター数`g`が傾き係数の数`q`（`k - k_constant`）以下:
    ///   `CommonError::InsufficientClustersForInference`（`rank(Ŝ) ≤ g - 1`のため
    ///   クラスターロバスト共分散が退化する識別失敗。Logit/Probitでは
    ///   新規制約）
    pub fn fit(input: ProbitInput, options: MleFitOptions) -> Result<Self, MleError> {
        fit::fit_probit(input, options)
    }

    /// 推定に使った入力データ
    pub fn input(&self) -> &ProbitInput {
        self.fitted.input()
    }

    /// 係数の分散共分散行列（元のスケール、k×k）
    pub fn cov_params(&self) -> &Mat<f64> {
        self.fitted.cov_params()
    }

    /// 標準誤差（k、元のスケール）
    pub fn std_errors(&self) -> &[f64] {
        &self.std_errors
    }

    /// z統計量（k）
    pub fn test_stats(&self) -> &[f64] {
        &self.test_stats
    }

    /// `test_stats`の従う分布（標準正規分布、自由度なし）。
    pub fn stat_dist(&self) -> inference::StatDist {
        inference::StatDist::Normal
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
        self.fitted.params()
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

    /// 尤度比検定の自由度（χ²、`df_model`）。`df_model==0`でp値がNaNのときは`None`。
    pub fn lr_df(&self) -> Option<usize> {
        (self.df_model() > 0).then_some(self.df_model())
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
        self.fitted.input().nobs()
    }

    /// モデルの自由度（`k-1`）
    pub fn df_model(&self) -> usize {
        self.df_model
    }

    /// 残差自由度（`n-k`）
    pub fn df_resid(&self) -> usize {
        self.df_resid
    }

    /// 限界効果。実装は[`ProbitFitted::marginal_effects`]。
    pub fn marginal_effects(
        &self,
        at: MarginalEffectsAt,
        confidence_level: f64,
    ) -> Result<MarginalEffects, MleError> {
        self.fitted.marginal_effects(at, confidence_level)
    }

    /// 学習データの予測確率。実装は[`ProbitFitted::predict`]。
    pub fn predict(&self) -> Vec<f64> {
        self.fitted.predict()
    }

    /// 新規データの予測確率。実装は[`ProbitFitted::predict_new_data`]。
    pub fn predict_new_data(&self, new_x_columns: &[Vec<f64>]) -> Vec<f64> {
        self.fitted.predict_new_data(new_x_columns)
    }

    /// 分類の的中表。実装は[`ProbitFitted::pred_table`]。
    pub fn pred_table(&self, threshold: f64) -> Mat<f64> {
        self.fitted.pred_table(threshold)
    }

    /// 予測・的中表・限界効果に必要な状態だけを取り出す。結果オブジェクトが推定量全体
    /// （標準誤差や適合度統計量。結果側に値のコピーがある）を持たずに済む。
    pub fn into_fitted(self) -> ProbitFitted {
        self.fitted
    }
}
