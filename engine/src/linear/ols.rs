//! OLSの入力データ（被説明変数・設計行列）の型定義と、推定本体（`OlsEstimator`）。
//! 推定本体は最小二乗・共分散行列（classical/HC0-3/HAC/cluster）・Wald検定・適合度の部品
//! （`crate::shared`）を呼んで推論結果を組み立てる。予測は`predict_new_data`。
//!
//! `engine`はpolars/PyO3を一切知らない（`.claude/rules/rust-style.md`「責務分離」参照）。
//! `engine_pybind`はpolars DataFrameから列ごとに`Vec<f64>`を抽出するところまでを担い
//! （`column_extraction::extract_f64_column`）、それらの列を本モジュールの
//! `OlsInput::from_columns`に渡す。`faer::Mat`への組み立て（切片列の自動追加を含む）は
//! ここ（engine側）の責務とする。詳細は`docs/spec/ols-spec.md`
//! 「API引数」の`include_intercept`の項を参照。
//!
//! # ファイル構成
//!
//! このファイルは外から見える面（`OlsEstimator`の構造体・getter・`fit`の入口・再エクスポート）
//! だけを持つ。ロジックは子モジュールに置き、テストは各ファイルの`mod tests`に置く（`fit`を通すテストだけは`fit/tests/`にテーマ別で置く）。
//!
//! - `input.rs`: `OlsInput`（設計行列の組み立て・検証）
//! - `cov_type.rs`: `CovType`
//! - `cov_params.rs`: cov_type別の共分散行列と推論の自由度、HACラグの解決、クラスター数の事前検証
//! - `fit.rs`: `fit`の本体（検証順序・最小二乗・共分散・t検定・適合度・F検定のオーケストレーション）
//! - `wald.rs`: 設計行列の末尾`q`列に対するWald検定（`wald_test_last_columns`）
//! - `predict.rs`: `predict_new_data`

use faer::Mat;

use crate::linear::common::LeastSquaresError;
use crate::shared::inference;

mod cov_params;
mod cov_type;
mod fit;
mod input;
mod predict;
mod wald;

pub use cov_type::CovType;
pub use input::OlsInput;
pub use predict::predict_new_data;

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
    test_stats: Mat<f64>,
    /// 両側p値 (k, 1)。t分布（自由度`df_inference`。通常`n-k`、`cov_type=Cluster`のときだけ`G-1`）に基づく
    p_values: Mat<f64>,
    /// 信頼区間の下限 (k, 1)
    conf_lower: Mat<f64>,
    /// 信頼区間の上限 (k, 1)
    conf_upper: Mat<f64>,
    /// 決定係数（`include_intercept`に応じてcentered/uncentered TSSを切り替える）
    r_squared: f64,
    /// 自由度調整済み決定係数
    adj_r_squared: f64,
    /// 全体F検定の`(F統計量, p値)`。`cov_type=Classical`なら古典的F検定、それ以外
    /// （HC0-3/HAC/Cluster）は`cov_params`を使ったロバストWald検定（`docs/spec/ols-spec.md`
    /// 「適合度統計量」参照）。p値はF分布（自由度`(k - k_constant, df_inference)`。
    /// `df_inference`は通常`n - k`、`cov_type=Cluster`のときだけ`G-1`）。
    /// 検定が定義できないとき`None`: 傾き係数が無い（定数項のみ）、または
    /// [`OlsEstimator::fit_with_optional_f_test`]で`cov_type=Cluster`のクラスター数`G`が
    /// 傾き係数の数`q`以下（`rank(Ŝ) ≤ G-1 < q`で`q×q`部分行列が構造的に特異）
    f_test: Option<(f64, f64)>,
    /// 対数尤度（正規分布を仮定した最尤推定量ベース。`σ̂²`は`SSR/n`であり、
    /// classical標準誤差の不偏推定量`SSR/(n-k)`とは異なる点に注意）
    log_likelihood: f64,
    aic: f64,
    bic: f64,
    /// 係数の分散共分散行列 (k, k)。Python側には公開しない（`docs/spec/ols-spec.md`
    /// 「結果構造体」）。クレート内の他系統から呼ばれる[`Self::wald_test_last_columns`]
    /// のためだけに保持している（`engine::iv::two_sls`のWu-Hausman検定が
    /// 唯一の呼び出し元。それまでは`fit()`内のローカル変数として使い切っていた）。
    cov_params: Mat<f64>,
    /// t検定・F検定・信頼区間に使う自由度。通常`df_resid`と同じだが`cov_type=Cluster`の
    /// ときだけ`G-1`になる（`fit()`のdocコメント「df_inference」参照）。`cov_params`と
    /// 同じ理由で保持している。
    df_inference: usize,
    /// `cov_type=Hac`のとき、実際に使われたラグ数（`hac_lags`明示指定、または`None`なら
    /// 経験則による自動計算の結果）。`CovType::Hac`の`lags`はユーザー指定値のまま
    /// 変更しないため別フィールドで保持する。`Hac`以外では`None`。
    hac_lags_used: Option<usize>,
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
    /// `cov_type`によらず、p値・信頼区間の算出にはt分布を使う（自由度は通常n-k、
    /// `cov_type=Cluster`のときだけ`G-1`。上記の`df_inference`）。
    /// 主リファレンスのstatsmodelsはHC0-3で正規分布を既定とするが（`use_t=False`）、
    /// 本プロジェクトはt分布で統一する方針（`docs/spec/ols-spec.md`
    /// 「標準誤差」）。ベンチマーク生成側
    /// （`benchmark/linear/references/statsmodels_ref.py`）は`use_t=True`を明示指定して合わせている。
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
    /// - `k`（定数項を含む説明変数の数）が0（`include_intercept=false`かつ説明変数も無い）:
    ///   `CommonError::NoRegressors`（Logit/Probitと同じ早期リジェクト）。**このチェックは
    ///   `confidence_level`より先に行う**。nonlinear系統の`validate_fit_preconditions`は
    ///   逆順（`confidence_level`→…→`k==0`）だが、Python APIの`validate_role_non_empty`により
    ///   どちらの入力もそもそも到達不能なため実害はなく、系統間で順序を揃えるという
    ///   明示的な方針も無い
    /// - `confidence_level`が`(0, 1)`の範囲外: `CommonError::InvalidConfidenceLevel`
    /// - 観測数`n`が`k`（定数項を含む説明変数の数）以下: `CommonError::InsufficientObservations`
    /// - 設計行列が特異（完全な多重共線性等）: `LeastSquaresError::SingularMatrix`
    /// - `cov_type=Cluster`でグループキー未指定: `CommonError::MissingClusterColumn`
    /// - `cov_type=Cluster`でクラスター数が2未満: `CommonError::InsufficientClusters`
    /// - `cov_type=Cluster`でクラスター数`g`が傾き係数の数`q`（`k - k_constant`）以下:
    ///   `CommonError::InsufficientClustersForInference`（`rank(Ŝ) ≤ g - 1`のため
    ///   ロバストWald/F検定の`q×q`部分行列が構造的に特異）
    /// - 傾き係数間の悪条件（極端なスケール差等）でロバストWald/F検定の`q×q`部分行列が
    ///   数値的にほぼ特異: `CommonError::NearSingularMatrix`（`g > q`でも起こりうる
    ///   backstop、`wald_f_test`参照）
    pub fn fit(
        input: OlsInput,
        cov_type: CovType,
        confidence_level: f64,
    ) -> Result<Self, LeastSquaresError> {
        fit::fit_ols(input, cov_type, confidence_level, false, true)
    }

    /// [`fit`](Self::fit)と同じだが、被説明変数が定数（`CommonError::OutcomeConstant`）か、
    /// 完全フィット（`CommonError::PerfectFit`）かを調べない。
    ///
    /// モデルの結果として利用者に見せる回帰ではなく、別の統計量の部品として使う補助的な回帰
    /// （診断検定の補助回帰、IVの第二段階、REの準偏差変換後の回帰とハウスマンの補助回帰）に使う。
    /// 診断検定の補助回帰・REの回帰の被説明変数は元のモデルの被説明変数ではない（残差の二乗、
    /// 変換後の値）ため、IVの第二段階は`X̂`への回帰で残差が構造残差ではないため、定数や完全フィットの
    /// 判定の基準（元のモデルの総平方和・残差）が当てはまらない。完全フィットは元のモデルの回帰
    /// （構造残差、内部のFE、OLS）が先に検出する。
    ///
    /// # Errors
    /// 被説明変数に関する2つを除き[`fit`](Self::fit)と同じ。
    pub(crate) fn fit_auxiliary(
        input: OlsInput,
        cov_type: CovType,
        confidence_level: f64,
    ) -> Result<Self, LeastSquaresError> {
        fit::fit_ols(input, cov_type, confidence_level, false, false)
    }

    /// [`fit`](Self::fit)と同じだが、`cov_type=Cluster`でクラスター数`g`が傾き係数の数`q`
    /// （`k - k_constant`）以下のときも`InsufficientClustersForInference`で失敗させず、
    /// 全体F検定だけを定義なし（`f_statistic()`/`f_p_value()`/`f_df()`が`None`）にして
    /// 推定を続ける。
    ///
    /// `rank(Ŝ) ≤ g - 1`のため`g ≤ q`では`q×q`部分行列が構造的に特異で全体F検定だけが
    /// 計算できなくなるが、係数・各係数の標準誤差・t検定・信頼区間・適合度は`g - 1`自由度で
    /// そのまま計算できる。呼び出し元が全体F検定を必須の出力にしない（診断として扱う）場合に
    /// 使う（`engine::iv::common::compute_first_stage`の第一段階回帰）。クラスター数が2未満
    /// （`InsufficientClusters`）・`groups`未指定（`MissingClusterColumn`）は従来どおり失敗し、
    /// `g > q`でのほぼ特異性（`NearSingularMatrix`）も`None`へ握りつぶさず失敗のまま伝播する。
    ///
    /// # Errors
    /// `g ≤ q`による`InsufficientClustersForInference`を除き[`fit`](Self::fit)と同じ。
    pub fn fit_with_optional_f_test(
        input: OlsInput,
        cov_type: CovType,
        confidence_level: f64,
    ) -> Result<Self, LeastSquaresError> {
        fit::fit_ols(input, cov_type, confidence_level, true, true)
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

    /// `cov_type`に応じた係数の分散共分散行列 (k, k)。`std_errors`はこの対角成分の平方根
    pub fn cov_params(&self) -> &Mat<f64> {
        &self.cov_params
    }

    /// t統計量 (k, 1)
    pub fn test_stats(&self) -> &Mat<f64> {
        &self.test_stats
    }

    /// `test_stats`の従う分布（t分布、自由度は`df_inference`。`cov_type=Cluster`のときだけ
    /// `df_resid`ではなく`G-1`になる）。
    pub fn stat_dist(&self) -> inference::StatDist {
        inference::StatDist::T {
            df: self.df_inference,
        }
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
    pub fn adj_r_squared(&self) -> f64 {
        self.adj_r_squared
    }

    /// 全体F検定の統計量。検定が定義できない（傾き係数が無い、または
    /// [`fit_with_optional_f_test`](Self::fit_with_optional_f_test)でクラスター数が傾き係数の数
    /// 以下）ときは`None`。
    pub fn f_statistic(&self) -> Option<f64> {
        self.f_test.map(|(statistic, _)| statistic)
    }

    /// 全体F検定のp値。`f_statistic()`が`None`のとき`None`。
    pub fn f_p_value(&self) -> Option<f64> {
        self.f_test.map(|(_, p_value)| p_value)
    }

    /// 残差自由度 `n - k`。
    pub fn df_resid(&self) -> usize {
        self.input.nobs() - self.input.k()
    }

    /// モデルの自由度（定数項を除く傾き係数の数 `k - k_constant`）。
    pub fn df_model(&self) -> usize {
        self.input.k() - usize::from(self.input.has_intercept())
    }

    /// t検定・信頼区間・F検定に使った自由度（[`stat_dist`](Self::stat_dist)と同じ値）。
    /// 通常は`df_resid()`だが`cov_type=Cluster`のときだけ`G-1`。
    pub fn df_inference(&self) -> usize {
        self.df_inference
    }

    /// `cov_type=Hac`のとき、実際に使われたラグ数（`hac_lags`の明示指定値、または
    /// 未指定時に経験則で自動計算した値）。`Hac`以外は`None`。
    pub fn hac_lags_used(&self) -> Option<usize> {
        self.hac_lags_used
    }

    /// F統計量の自由度`(分子, 分母)`。`f_statistic()`が`None`のときは`None`。
    pub fn f_df(&self) -> Option<(usize, usize)> {
        self.f_test.map(|_| (self.df_model(), self.df_inference))
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

    /// 設計行列の**末尾`q`列**に対応する係数が全てゼロという帰無仮説のロバストWald検定を行い、
    /// F統計量とそのp値を返す（`fit_ols`（`ols/fit.rs`）が呼ぶ`shared::wald::wald_f_test`の対象列を、「切片を除く
    /// 全傾き係数」から「任意の末尾`q`列」に一般化したもの。数式・分布・p値の向きは
    /// `wald_f_test`のdocコメントと同じ）。
    ///
    /// クレート内の他系統から、この`OlsEstimator`自身が構築した設計行列の一部（末尾に
    /// 追加した列）だけをまとめて検定したい場合に使う（`engine::iv::two_sls`の
    /// Wu-Hausman検定——構造式に第一段階残差を追加回帰し、追加した残差係数のジョイント
    /// 有意性を検定する——が現時点で唯一の呼び出し元）。
    ///
    /// # Panics
    /// `q == 0`または`q > self.input.k()`は呼び出し元の実装バグでしか起こり得ない
    /// （検定対象が空、または設計行列の列数を超える）内部契約違反のため、`debug_assert!`で
    /// 検出する（`IvInput::from_columns`の引数長チェックと同じ方針、
    /// `.claude/rules/rust-style.md`は触れていないがこのクレート内（`ols/wald.rs`等）で既に使われている
    /// パターン）。
    ///
    /// # Errors
    /// `wald_f_test`と同じ（末尾`q`列に対応する`cov_params`の部分行列が数値的にほぼ特異な
    /// 場合、`LeastSquaresError::Common(CommonError::NearSingularMatrix)`）。
    pub fn wald_test_last_columns(&self, q: usize) -> Result<(f64, f64), LeastSquaresError> {
        wald::wald_test_last_columns(self, q)
    }

    /// 学習データに対する予測値 `ŷ = Xβ̂`（`predict(new_data=None)`のPython APIが返す値、
    /// `docs/spec/ols-spec.md`「predict()」参照）。`fit()`のReturn本体には含めず、
    /// 必要なときに計算する別メソッドとする（Logitの`predict()`と同じ設計方針）。
    pub fn fitted_values(&self) -> Mat<f64> {
        self.input.x() * &self.params
    }
}
