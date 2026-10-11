use faer::Mat;
use statrs::distribution::StudentsT;

use super::cov_params::{CovParams, compute_cov_params, validate_cluster_count};
use super::cov_type::CovType;
use super::input::OlsInput;
use crate::linear::common::LeastSquaresError;
use crate::shared::error::CommonError;
use crate::shared::goodness_of_fit::{GaussianGoodnessOfFit, gaussian_goodness_of_fit};
use crate::shared::group_codes::GroupCodes;
use crate::shared::inference;
use crate::shared::least_squares::{LeastSquaresFit, least_squares, residual_sum_of_squares};
use crate::shared::validation::validate_has_regressors;
use crate::shared::wald::wald_f_test;

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
    /// F統計量。`cov_type=Classical`なら古典的F検定、それ以外（HC0-3/HAC）は
    /// `cov_params`を使ったロバストWald検定（`docs/spec/ols-spec.md`
    /// 「適合度統計量」参照）
    f_statistic: f64,
    /// F統計量のp値（F分布、自由度は`(k - k_constant, df_inference)`。`df_inference`は
    /// 通常`n - k`、`cov_type=Cluster`のときだけ`G-1`）
    f_p_value: f64,
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
    ///   逆順（`confidence_level`→…→`k==0`）だが、Python APIの`validate_x_non_empty`により
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
    ///   数値的にほぼ特異: `CommonError::ComputationFailed`（`g > q`でも起こりうる
    ///   backstop、`wald_f_test`参照）
    pub fn fit(
        input: OlsInput,
        cov_type: CovType,
        confidence_level: f64,
    ) -> Result<Self, LeastSquaresError> {
        // クラスターのグループキーは、検証（クラスター数）と集計（`Σ_g S_g S_g'`）の両方で
        // 使うため、ここで一度だけ整数コードにする（`CovType::cluster_codes`参照）。
        let cluster_codes = cov_type.cluster_codes();
        Self::fit_with_cluster_codes(input, cov_type, cluster_codes.as_ref(), confidence_level)
    }

    /// [`Self::fit`]と同じだが、`cov_type=Cluster`のグループキーを整数コード化済みの
    /// `cluster_codes`で受け取る。IVの第一段階・Wu-Hausman拡張回帰のように、同じクラスター列で
    /// `fit`を何度も呼ぶ呼び出し元が、列ごとに（内生変数の数だけ）コード化し直さずに済む
    /// ようにするための内部入口。`cluster_codes`は`cov_type.cluster_codes()`と同じ内容で
    /// あること（`cov_type=Cluster`以外では使われない）。
    pub(crate) fn fit_with_cluster_codes(
        input: OlsInput,
        cov_type: CovType,
        cluster_codes: Option<&GroupCodes>,
        confidence_level: f64,
    ) -> Result<Self, LeastSquaresError> {
        validate_has_regressors(input.nobs(), input.k())?;

        // faer のグローバル並列度を Par::Seq に固定する（`crate::shared::parallelism`）。
        crate::shared::parallelism::ensure_serial();

        if !(confidence_level > 0.0 && confidence_level < 1.0) {
            return Err(CommonError::InvalidConfidenceLevel { confidence_level }.into());
        }

        let n = input.nobs();
        let k = input.k();

        if n <= k {
            return Err(CommonError::InsufficientObservations { n, k }.into());
        }

        // 入力だけから判定できるクラスター数の検証は、QR分解・残差計算より前に行う
        // （`cov_params::validate_cluster_count`参照）。
        validate_cluster_count(cluster_codes, n, k - usize::from(input.has_intercept()))?;

        let LeastSquaresFit {
            params,
            residuals,
            xtx_inv,
        } = least_squares(input.x(), input.y()).map_err(|_| LeastSquaresError::SingularMatrix)?;

        let df_resid = n - k;
        let ssr = residual_sum_of_squares(&residuals);
        let sigma2 = ssr / (df_resid as f64);

        let CovParams {
            cov_params,
            df_inference,
            hac_lags_used,
        } = compute_cov_params(
            &cov_type,
            cluster_codes,
            input.x(),
            &residuals,
            &xtx_inv,
            sigma2,
            df_resid,
        )?;

        let mut std_errors = Mat::zeros(k, 1);
        for j in 0..k {
            *std_errors.get_mut(j, 0) = (*cov_params.get(j, j)).sqrt();
        }

        let t_dist = StudentsT::new(0.0, 1.0, df_inference as f64)
            .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
        let t_crit = inference::critical_value(&t_dist, confidence_level);

        let mut test_stats = Mat::zeros(k, 1);
        let mut p_values = Mat::zeros(k, 1);
        let mut conf_lower = Mat::zeros(k, 1);
        let mut conf_upper = Mat::zeros(k, 1);

        for j in 0..k {
            let coef = *params.get(j, 0);
            let se = *std_errors.get(j, 0);
            let stat = inference::compute_inference_stat(&t_dist, coef, se, t_crit);

            *test_stats.get_mut(j, 0) = stat.stat;
            *p_values.get_mut(j, 0) = stat.p_value;
            *conf_lower.get_mut(j, 0) = stat.conf_low;
            *conf_upper.get_mut(j, 0) = stat.conf_high;
        }

        let k_constant = usize::from(input.has_intercept());
        let GaussianGoodnessOfFit {
            r_squared,
            adj_r_squared,
            log_likelihood,
            aic,
            bic,
        } = gaussian_goodness_of_fit(input.y(), ssr, k, input.has_intercept());

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
            test_stats,
            p_values,
            conf_lower,
            conf_upper,
            r_squared,
            adj_r_squared,
            f_statistic,
            f_p_value,
            log_likelihood,
            aic,
            bic,
            cov_params,
            df_inference,
            hac_lags_used,
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

    /// F統計量
    pub fn f_statistic(&self) -> f64 {
        self.f_statistic
    }

    /// F統計量のp値
    pub fn f_p_value(&self) -> f64 {
        self.f_p_value
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

    /// F統計量の自由度`(分子, 分母)`。傾き係数が無く`f_statistic()`がNaNのときは`None`。
    pub fn f_df(&self) -> Option<(usize, usize)> {
        let df_model = self.df_model();
        (df_model > 0).then_some((df_model, self.df_inference))
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
    /// F統計量とそのp値を返す（`fit()`が呼ぶ内部関数`wald_f_test`の対象列を、「切片を除く
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
    /// `.claude/rules/rust-style.md`は触れていないがこのファイル内で既に使われている
    /// パターン）。
    ///
    /// # Errors
    /// `wald_f_test`と同じ（末尾`q`列に対応する`cov_params`の部分行列が数値的にほぼ特異な
    /// 場合、`LeastSquaresError::Common(CommonError::ComputationFailed)`）。
    pub fn wald_test_last_columns(&self, q: usize) -> Result<(f64, f64), LeastSquaresError> {
        let k = self.input.k();
        debug_assert!(
            q > 0 && q <= k,
            "wald_test_last_columns: q must be in (0, k], got q={q}, k={k}"
        );
        Ok(wald_f_test(
            &self.params,
            &self.cov_params,
            k - q,
            q,
            self.df_inference,
        )?)
    }

    /// 学習データに対する予測値 `ŷ = Xβ̂`（`predict(new_data=None)`のPython APIが返す値、
    /// `docs/spec/ols-spec.md`「predict()」参照）。`fit()`のReturn本体には含めず、
    /// 必要なときに計算する別メソッドとする（Logitの`predict()`と同じ設計方針）。
    pub fn fitted_values(&self) -> Mat<f64> {
        self.input.x() * &self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::common::row_time_order;

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
    fn fit_returns_no_regressors_error_when_k_is_zero() {
        // include_intercept=falseかつx_columns=[]の病的な入力。
        // Logit/Probitと同じ`CommonError::NoRegressors`で早期リジェクトされる。
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = OlsInput::from_columns(&y, &[], vec![], false, "y".to_string()).unwrap();

        let result = OlsEstimator::fit(input, CovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::NoRegressors { n: 5 })
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
    fn fit_pins_faer_global_parallelism_to_seq() {
        // `fit()` 冒頭の `crate::shared::parallelism::ensure_serial()` が faer の
        // グローバル並列度を `Par::Seq` へ引き戻すことの回帰ガード（linear 系統代表）。
        // 別テストが既に `Seq` にしている可能性があるため、まず `Rayon` に戻してから
        // `fit()` を通す。ここで扱う設計行列は極小なので、この一時的な `Rayon` 設定が
        // その病理（大標本 tall-skinny での不安定化）を招くことはない。
        faer::set_global_parallelism(faer::Par::rayon(0));

        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 6.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let input =
            OlsInput::from_columns(&y, &[x1], vec!["x1".to_string()], true, "y".to_string())
                .unwrap();

        let _ = OlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        assert!(matches!(faer::get_global_parallelism(), faer::Par::Seq));
    }

    /// `wald_test_last_columns`が「切片を除く全傾き係数」（`q = df_model`）を対象に呼ばれた
    /// 場合、`fit()`が計算する`f_statistic()`/`f_p_value()`（同じ`wald_f_test`を
    /// `k_constant=1`で呼ぶ）と数値的に一致するはず（対象列の一般化が既存の挙動を
    /// 壊していないことの確認。IVのWu-Hausman検定用に追加）。
    #[test]
    fn wald_test_last_columns_matches_f_statistic_when_q_equals_df_model() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 6.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let x2 = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0];
        let input = OlsInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = OlsEstimator::fit(input, CovType::Hc1, 0.95).unwrap();

        let (stat, p_value) = estimator.wald_test_last_columns(2).unwrap();
        assert!((stat - estimator.f_statistic()).abs() < 1e-10);
        assert!((p_value - estimator.f_p_value()).abs() < 1e-10);
    }

    /// `q=1`（末尾1列だけを対象）のとき、`F = t²`という標準的な恒等式
    /// （`wald_f_test`のdocコメント参照、1自由度のF検定は両側t検定と代数的に等価）により、
    /// 既に個別に検証済みの`test_stats()`/`p_values()`（`fit()`本体が計算）と一致するはず。
    #[test]
    fn wald_test_last_columns_matches_squared_t_statistic_for_single_column() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 6.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let x2 = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0];
        let input = OlsInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = OlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        let (stat, p_value) = estimator.wald_test_last_columns(1).unwrap();
        let k = estimator.input().k();
        let t_last = *estimator.test_stats().get(k - 1, 0);
        let p_last = *estimator.p_values().get(k - 1, 0);
        assert!((stat - t_last.powi(2)).abs() < 1e-10);
        assert!((p_value - p_last).abs() < 1e-10);
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
    fn fit_computes_classical_std_errors_test_stats_p_values_and_conf_int() {
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

        let t = estimator.test_stats();
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
        assert!((estimator.adj_r_squared() - 0.466_666_666_666_666_56).abs() < 1e-9);
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
        assert!((estimator.adj_r_squared() - 0.901_162_790_697_674_5).abs() < 1e-9);
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
            time_order: row_time_order(5),
        };
        let estimator_hac = OlsEstimator::fit(input_hac, cov_type_hac, 0.95).unwrap();
        assert!((estimator_hac.f_statistic() - 13.235_294_117_647_193).abs() < 1e-6);
        assert!((estimator_hac.f_p_value() - 0.035_791_053_269_350_51).abs() < 1e-6);
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

    /// property-basedテスト。固定シナリオでの値の一致確認（上記）とは別に、
    /// OLSが満たすべき不変条件をランダムなデータで検証する（`testing-policy.md`
    /// 「将来的に検討する技術」参照）。
    mod proptests {
        use super::*;
        use proptest::collection;
        use proptest::prelude::*;

        const MAX_K: usize = 20;

        /// `(n, k, y, x_cols, keys)`を生成する共通ストラテジ。
        ///
        /// `n = k+10..=60`という十分なマージンを取り、値は独立な連続一様分布
        /// （`-100.0..100.0`）からサンプリングする。この条件下では生成される設計行列は
        /// 実務上ほぼ確実にフルランクになるため、各プロパティ側では追加の制約は課さず、
        /// `prop_assume!(result.is_ok())`で非フルランクになるレアケース（丸め誤差起因の
        /// 境界事例等）のみを除外する（「ランダムに生成する設計行列は
        /// SingularMatrixにならない範囲に制約する」という方針に対応）。
        /// `MAX_K=20`（旧4）はbenchmarkの高次元シナリオ`many_regressors`と揃えた値
        /// （列数依存バグ・高kでの数値的挙動の検証）。
        /// `k`が最大でも`n-k>=10`のマージンは保たれる。
        ///
        /// `keys`は列順序入れ替えテスト専用の補助データ（他のプロパティでは未使用）。
        fn ols_case_strategy()
        -> impl Strategy<Value = (usize, usize, Vec<f64>, Vec<Vec<f64>>, Vec<u64>)> {
            (1..=MAX_K).prop_flat_map(|k| {
                (k + 10..=60usize).prop_flat_map(move |n| {
                    (
                        Just(n),
                        Just(k),
                        collection::vec(-100.0f64..100.0, n),
                        collection::vec(collection::vec(-100.0f64..100.0, n), k),
                        collection::vec(any::<u64>(), k),
                    )
                })
            })
        }

        fn x_names(k: usize) -> Vec<String> {
            (1..=k).map(|i| format!("x{i}")).collect()
        }

        /// 相対誤差ベースの近似比較（絶対誤差フロア込み）。乱数生成で値のスケールが
        /// 揃わないため、固定フィクスチャ比較（`testing-policy.md`のRTOL=1e-8）より
        /// 緩めた閾値にしている（col_piv_qrの数値誤差の蓄積を考慮）。
        fn assert_approx_eq(actual: f64, expected: f64, msg: &str) {
            let tol = 1e-6 * expected.abs().max(1.0);
            let diff = (actual - expected).abs();
            assert!(
                diff <= tol,
                "{msg}: actual={actual}, expected={expected}, diff={diff}, tol={tol}"
            );
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            /// 定数項ありOLSでの残差和は常に0（正規方程式`X'e=0`のうち切片列に対応する
            /// 行から従う）。
            #[test]
            fn residuals_sum_to_zero_when_intercept_included(
                (n, k, y, x_cols, _keys) in ols_case_strategy()
            ) {
                let input = OlsInput::from_columns(&y, &x_cols, x_names(k), true, "y".to_string()).unwrap();
                let result = OlsEstimator::fit(input, CovType::Classical, 0.95);
                prop_assume!(result.is_ok());
                let est = result.unwrap();

                let sum: f64 = (0..n).map(|i| *est.residuals().get(i, 0)).sum();
                let scale = y.iter().fold(1.0_f64, |acc, v| acc.max(v.abs()));
                prop_assert!(
                    sum.abs() <= 1e-6 * scale * (n as f64),
                    "residual sum should be ~0, got {sum} (scale={scale}, n={n})"
                );
            }

            /// yをc倍すると、切片を含む全ての係数がc倍にスケールする
            /// （OLSはyに関して線形なため、切片も例外ではない）。
            #[test]
            fn coefficients_scale_linearly_with_y(
                (_n, k, y, x_cols, _keys) in ols_case_strategy(),
                c in prop_oneof![-10.0f64..-0.1, 0.1f64..10.0],
            ) {
                let input1 = OlsInput::from_columns(&y, &x_cols, x_names(k), true, "y".to_string()).unwrap();
                let result1 = OlsEstimator::fit(input1, CovType::Classical, 0.95);
                prop_assume!(result1.is_ok());
                let est1 = result1.unwrap();

                let y_scaled: Vec<f64> = y.iter().map(|v| v * c).collect();
                let input2 =
                    OlsInput::from_columns(&y_scaled, &x_cols, x_names(k), true, "y".to_string()).unwrap();
                let result2 = OlsEstimator::fit(input2, CovType::Classical, 0.95);
                prop_assume!(result2.is_ok());
                let est2 = result2.unwrap();

                for i in 0..=k {
                    let expected = c * *est1.params().get(i, 0);
                    let actual = *est2.params().get(i, 0);
                    assert_approx_eq(actual, expected, &format!("param[{i}] scaled by c={c}"));
                }
            }

            /// xの列順序を入れ替えても、係数名で対応付ければ係数・標準誤差の値は変わらない。
            #[test]
            fn coefficients_and_se_are_invariant_to_column_order(
                (_n, k, y, x_cols, keys) in ols_case_strategy()
                    .prop_filter("need >=2 columns to permute", |(_, k, _, _, _)| *k >= 2)
            ) {
                let names = x_names(k);
                let input1 =
                    OlsInput::from_columns(&y, &x_cols, names.clone(), true, "y".to_string()).unwrap();
                let result1 = OlsEstimator::fit(input1, CovType::Classical, 0.95);
                prop_assume!(result1.is_ok());
                let est1 = result1.unwrap();

                let mut order: Vec<usize> = (0..k).collect();
                order.sort_by_key(|&i| keys[i]);
                let permuted_x: Vec<Vec<f64>> = order.iter().map(|&i| x_cols[i].clone()).collect();
                let permuted_names: Vec<String> = order.iter().map(|&i| names[i].clone()).collect();

                let input2 =
                    OlsInput::from_columns(&y, &permuted_x, permuted_names, true, "y".to_string()).unwrap();
                let result2 = OlsEstimator::fit(input2, CovType::Classical, 0.95);
                prop_assume!(result2.is_ok());
                let est2 = result2.unwrap();

                let names1 = est1.input().param_names().to_vec();
                let names2 = est2.input().param_names().to_vec();
                for (idx1, name) in names1.iter().enumerate() {
                    let idx2 = names2
                        .iter()
                        .position(|n| n == name)
                        .expect("name should exist in permuted result");
                    let p1 = *est1.params().get(idx1, 0);
                    let p2 = *est2.params().get(idx2, 0);
                    assert_approx_eq(p2, p1, &format!("param[{name}] under column permutation"));
                    let se1 = *est1.std_errors().get(idx1, 0);
                    let se2 = *est2.std_errors().get(idx2, 0);
                    assert_approx_eq(se2, se1, &format!("std_error[{name}] under column permutation"));
                }
            }

            /// HC0の標準誤差は常にHC1以下（`HC1 = HC0 * n/(n-k)`で`n/(n-k) >= 1`のため）。
            #[test]
            fn hc0_std_errors_are_at_most_hc1_std_errors(
                (_n, k, y, x_cols, _keys) in ols_case_strategy()
            ) {
                let input1 = OlsInput::from_columns(&y, &x_cols, x_names(k), true, "y".to_string()).unwrap();
                let result1 = OlsEstimator::fit(input1, CovType::Hc0, 0.95);
                prop_assume!(result1.is_ok());
                let est_hc0 = result1.unwrap();

                let input2 = OlsInput::from_columns(&y, &x_cols, x_names(k), true, "y".to_string()).unwrap();
                let result2 = OlsEstimator::fit(input2, CovType::Hc1, 0.95);
                prop_assume!(result2.is_ok());
                let est_hc1 = result2.unwrap();

                for i in 0..=k {
                    let se_hc0 = *est_hc0.std_errors().get(i, 0);
                    let se_hc1 = *est_hc1.std_errors().get(i, 0);
                    // 浮動小数点の丸め誤差の余地として小さな絶対許容誤差を加える。
                    prop_assert!(
                        se_hc0 <= se_hc1 + 1e-9,
                        "HC0 se[{i}]={se_hc0} should be <= HC1 se[{i}]={se_hc1}"
                    );
                }
            }
        }
    }
}
