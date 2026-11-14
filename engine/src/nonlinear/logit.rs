//! Logitの推定量`LogitEstimator`（`fit`の契約とgetter）。入力データ（被説明変数・設計行列）、
//! 負の対数尤度・スコア・Hessian（argminの`CostFunction`/`Gradient`/`Hessian`トレイト実装）、
//! 予測・限界効果に必要な最小の状態`LogitFitted`、`fit`の本体は子モジュールに置く。
//!
//! `engine`はpolars/PyO3を一切知らない（`.claude/rules/rust-style.md`「責務分離」参照）。
//! `engine_pybind`はpolars DataFrameから列ごとに`Vec<f64>`を抽出するところまでを担い
//! （`column_extraction::extract_f64_column`）、それらの列を
//! `LogitInput::from_columns`（`logit/input.rs`）に渡す。`faer::Mat`への組み立て（切片列の自動追加を含む）は
//! ここ（engine側）の責務とする。`engine::linear::ols::OlsInput`と同型の設計。
//!
//! OLSと異なり、Phase2（Logit/Probit/Tobit）では`weights`/`offset`を見送っているため
//! （`docs/spec/nonlinear-common.md`7章）、`from_columns_weighted`に相当するものはない。
//!
//! ## 数式（ロジスティック回帰）
//!
//! `p_i = Λ(x_i'θ) = 1/(1+exp(-x_i'θ))`（ロジスティック関数）。観測`i`の対数尤度への
//! 寄与は`ℓ_i(θ) = y_i log(p_i) + (1-y_i) log(1-p_i)`で、`z_i = x_i'θ`とおくと
//! `ℓ_i(θ) = y_i z_i - softplus(z_i)`（`softplus(z) = log(1+exp(z))`）という同値な形に
//! 書き換えられる（`log(p_i) = z_i - softplus(z_i)`、`log(1-p_i) = -softplus(z_i)`）。
//! `softplus`形の方が指数関数のオーバーフローを避けやすいため、`cost`の実装はこちらを使う。
//!
//! - スコア（対数尤度の1階微分）: `∂ℓ/∂θ = Σᵢ (yᵢ-pᵢ)xᵢ = X'(y-p)`
//! - Hessian（対数尤度の2階微分）: `∂²ℓ/∂θ∂θ' = -Σᵢ pᵢ(1-pᵢ)xᵢxᵢ' = -X'WX`
//!   （`W = diag(pᵢ(1-pᵢ))`。ロジットの対数尤度は大域的に凹なので`-X'WX`は
//!   （厳密な多重共線性がなければ）常に負定値、`X'WX`は常に正定値）
//!
//! `CostFunction`は`-ℓ(θ)`（argminは最小化フレームワークのため）。`run_solver`の
//! docコメント「`Hessian`トレイトの符号規約」の通り、`Gradient`/`Hessian`トレイトは
//! `CostFunction`と同じ符号（`-ℓ`の1階・2階微分）で実装する。`scores()`（`cov_type`
//! 共通行列演算向け）は符号反転しない生のスコア`sᵢ=(yᵢ-pᵢ)xᵢ`を返す
//! （`nonlinear/common/solver.rs`の`SolverOutput.hessian`と同じく対数尤度そのものの符号）。
//!
//! # ファイル構成
//!
//! このファイルは外から見える面（`LogitEstimator`の構造体・getter・`fit`の契約と委譲・
//! 再エクスポート）だけを持つ。ロジックは子モジュールに置き、テストは各ロジックと同じ
//! ファイルの`mod tests`に置く（`fit`を通すテストと、テストが大きいファイルは
//! `<ファイル名>/tests/`にテーマ別で置く）。
//!
//! - `input.rs`: `LogitInput`
//! - `model.rs`: `softplus`・ロジスティック関数・対数尤度、限界効果の`w`と`s`
//! - `problem.rs`: `LogitProblem`（`CostFunction`/`Gradient`/`Hessian`）
//! - `fitted.rs`: `LogitFitted`（`from_parts`・限界効果・予測・的中表）
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

pub use fitted::LogitFitted;
pub use input::LogitInput;
pub use problem::LogitProblem;

/// Logitの推定結果。`fit`でのバリデーション・最適化・`cov_type`に応じたSE計算・
/// 適合度統計量の計算を通過した状態を表す。
///
/// `marginal_effects`/`predict`/`pred_table`は`fit`とは別のメソッドとして提供する
/// （`docs/spec/nonlinear-common.md`6章）。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug)]
pub struct LogitEstimator {
    /// 予測・的中表・限界効果に必要な状態（入力データ・係数・係数の分散共分散行列）。
    fitted: LogitFitted,
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
    /// 再フィットは経由しない。`log_likelihood`のdocコメント
    /// 「切片のみモデルのllf」参照）。`include_intercept`の値に関わらず常にこの
    /// 「切片のみ」モデルを参照する（`docs/spec/nonlinear-common.md`5章の定義通り、
    /// statsmodelsも`k_constant`の有無に関わらず同じ挙動）。
    ///
    /// **`include_intercept=false`のとき、この値が参照する「切片のみ」モデルは
    /// フィット対象のモデルの部分集合（入れ子）にならない**（statsmodelsで
    /// `Logit(y, X_without_const).fit()`をした場合と同じ状況、実測でも
    /// `llr`が負値・`llr_pvalue`がほぼ1.0になる例を確認済み）。この場合
    /// `lr_statistic`が負になったり`lr_p_value`が統計的に意味の薄い値になったり
    /// しうるが、これはstatsmodels準拠の仕様上の挙動でありバグではない。
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
    /// `include_intercept`の値に関わらず常にこの式）
    df_model: usize,
    /// 残差自由度 `n-k`
    df_resid: usize,
}

impl LogitEstimator {
    /// `solver`（Newton-Raphson/BFGS/L-BFGS）で負の対数尤度を最小化し、Logitの係数・
    /// 観測情報行列によるSE・z値・p値・信頼区間を推定する。
    ///
    /// `solver`の選択に関わらず、収束点でのHessian評価（SE計算用）は常に解析的に行う
    /// （`run_solver`の実装方針、`docs/spec/nonlinear-common.md`1.2節参照）。
    /// BFGS/L-BFGSが最適化中に内部で保持する近似Hessianは
    /// 使い回さない。
    ///
    /// 初期値（warm start）は標準化空間でのLPM（線形確率モデル）最小二乗解に、logitの
    /// IRLS 1ステップ相当のスケール補正（`p̄=ȳ`での `1/(p̄(1-p̄))` 倍＋切片補正）を施した
    /// もの（`ols_based_initial_params`。従来のゼロベクトルから変更）。前段で
    /// `standardize_columns`後の設計行列を列ピボットQRしランク落ちを検出する
    /// （`checked_design_matrix_qr`、`solver`によらず単一経路で`SingularDesignMatrix`）。
    /// `start_params`によるユーザー指定初期値は引き続き未対応（`docs/spec/
    /// nonlinear-common.md`7章では確定オプションだが、現状は見送っている）。
    ///
    /// 設計行列は`standardize_columns`で内部的に標準化してから最適化し（勾配ノルムに
    /// 基づく収束判定`tol`が設計行列のスケールに依存しないようにするため、
    /// `docs/spec/nonlinear-common.md`1.3節参照）、
    /// 収束後のパラメータを`destandardize_params`で元のスケールへ逆変換する。
    /// `run_solver`が返すHessianは標準化空間（θ_std）で評価されたものであり、
    /// 分散共分散行列もいったん標準化空間で計算してから`destandardize_cov_params`で
    /// 元のスケールへ逆変換する（`destandardize_params`を先に適用してから逆算するのではなく、
    /// 標準化空間のcov_paramsを直接destandardizeする。数学的に等価だが後者の方が
    /// 中間にHessianの逆変換を挟まず単純、`destandardize_cov_params`のdocコメント参照）。
    ///
    /// `cov_type`は観測情報行列（`Classical`）・OPG（`Opg`）・サンドイッチ型
    /// （`Hc0`/`Hc1`）・クラスターロバスト（`Cluster`）に対応する。`Cluster`の
    /// グループキー未指定・クラスター数不足は、最適化を実行する前（`fit()`冒頭）に
    /// 検証して早期に返す（OLSの`cov_type=Cluster`は閉形式解のため事後検証でも
    /// コストが変わらないが、Logitは反復最適化のため無駄な計算を避ける）。
    /// `Opg`/`Hc0`/`Hc1`/`Cluster`は収束点での観測ごとのスコア（`LogitProblem::
    /// scores`）が必要なため、標準化空間の設計行列を保持したまま`LogitProblem`を
    /// クローンしておき（`argmin::core::Executor`向けに元々`Clone`を要求しているため
    /// 追加コストは`Clone`実装自体のみ）、`run_solver`が返す収束点のパラメータで
    /// 評価する。検定分布は標準正規分布（`docs/spec/nonlinear-common.md`4章、OLSのt分布とは
    /// 異なる）。
    ///
    /// `n <= k`（観測数が説明変数の数、定数項を含む、以下）のとき`CommonError::
    /// InsufficientObservations`で弾く閾値はOLSと同じ式だが、根拠は異なる。OLSでは
    /// 残差自由度`n-k`が0以下だと分散推定が原理的に不可能という数学的必要条件だが、
    /// LogitのようなMLEベースのモデルでは`n<=k`はほぼ確実に完全分離
    /// （perfect separation。ある説明変数の値でyの値が完全に分かれてしまい、
    /// 尤度が発散してMLEが存在しない状態）を引き起こす経験則としての安全側の判断。
    /// 数学的な必要条件ではないが、後続のProbit/Tobit実装でもこの閾値をそのまま
    /// 踏襲する（他パッケージがこの水準で明示的な検証をしていない場合でも、
    /// 発散した推定量を黙って返すよりは早期にエラーにする方針を優先する）。
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
    ///   ロバストWald検定（LRではなくクラスターロバスト共分散側）で退化する識別失敗。
    ///   Logit/Probitでは新規制約——従来は縮退した共分散から読んだSEを
    ///   無警告で返していた）
    pub fn fit(input: LogitInput, options: MleFitOptions) -> Result<Self, MleError> {
        fit::fit_logit(input, options)
    }

    /// 推定に使った入力データ
    pub fn input(&self) -> &LogitInput {
        self.fitted.input()
    }

    /// 係数（元のスケール）
    pub fn params(&self) -> &[f64] {
        self.fitted.params()
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

    /// 観測数。`self.input.nobs()`への委譲（`OlsEstimator`と同じパターン、
    /// `n`という同じ値の出どころを2つに分けない）
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

    /// 限界効果。実装は[`LogitFitted::marginal_effects`]。
    pub fn marginal_effects(
        &self,
        at: MarginalEffectsAt,
        confidence_level: f64,
    ) -> Result<MarginalEffects, MleError> {
        self.fitted.marginal_effects(at, confidence_level)
    }

    /// 学習データの予測確率。実装は[`LogitFitted::predict`]。
    pub fn predict(&self) -> Vec<f64> {
        self.fitted.predict()
    }

    /// 新規データの予測確率。実装は[`LogitFitted::predict_new_data`]。
    pub fn predict_new_data(&self, new_x_columns: &[Vec<f64>]) -> Vec<f64> {
        self.fitted.predict_new_data(new_x_columns)
    }

    /// 分類の的中表。実装は[`LogitFitted::pred_table`]。
    pub fn pred_table(&self, threshold: f64) -> Mat<f64> {
        self.fitted.pred_table(threshold)
    }

    /// 予測・的中表・限界効果に必要な状態だけを取り出す。結果オブジェクトが推定量全体
    /// （標準誤差や適合度統計量。結果側に値のコピーがある）を持たずに済む。
    pub fn into_fitted(self) -> LogitFitted {
        self.fitted
    }
}
