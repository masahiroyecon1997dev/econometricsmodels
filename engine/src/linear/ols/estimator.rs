use faer::Mat;
use faer::prelude::SolveLstsq;
use statrs::distribution::StudentsT;

use super::cov_type::CovType;
use super::input::OlsInput;
use crate::error::CommonError;
use crate::inference;
use crate::linear::common::LeastSquaresError;
use crate::linear_algebra::checked_col_piv_qr;
use crate::shared::covariance::{
    HcVariant, classical_cov_params, cluster_cov_params, hac_cov_params, hc_cov_params,
    time_ordering, xtx_inverse,
};
use crate::shared::wald::wald_f_test;
use crate::validation::{
    validate_cluster_count_covers_slopes, validate_cluster_groups, validate_has_regressors,
};

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
    ///   `CommonError::NoRegressors`（Logit/Probitと同じ早期リジェクト。
    ///   `fit_allowing_no_regressors`はこのチェックを行わない）。**このチェックは
    ///   `confidence_level`より先に行う**（`fit`が`fit_allowing_no_regressors`へ委譲する
    ///   実装構造上、外側でしか検証できないため）。nonlinear系統の
    ///   `validate_fit_preconditions`は逆順（`confidence_level`→…→`k==0`）だが、
    ///   Python APIの`validate_x_non_empty`によりどちらの入力もそもそも到達不能なため
    ///   実害はなく、系統間で順序を揃えるという明示的な方針も無い
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
        validate_has_regressors(input.nobs(), input.k())?;
        Self::fit_allowing_no_regressors(input, cov_type, confidence_level)
    }

    /// `fit`と同じ計算を行うが、`k`（定数項を含む説明変数の数）が0の入力も受理する
    /// （`fit`が行う`CommonError::NoRegressors`の早期リジェクトをスキップする）。
    ///
    /// `pub(crate)`: `panel::fe::FeEstimator::fit`が「固定効果のみのモデル」（`x=[]`。
    /// FEは常に`include_intercept=false`で委譲するためこの場合`k=0`になる）を
    /// サポートするために、`fit`のガードを迂回してこの内部実装を直接呼ぶ。他の呼び出し元
    /// （`WlsEstimator::fit`・`panel::re::ReEstimator::fit`・IV系統）はいずれも構造的に
    /// `k=0`になりえない呼び出し方をしており（RE/IVは常に切片または操作変数由来の列を
    /// 最低1列持つ、`engine/src/linear/CLAUDE.md`「k=0の扱い」参照）、`fit`（ゲート付き）を
    /// そのまま使う。`k=0`でも`col_piv_qr`・`wald_f_test`が安全に動作することは
    /// `checked_col_piv_qr`のNaN明示チェック・`df_model==0`分岐により保証済み（同ドキュメント
    /// 参照）。
    ///
    /// # Errors
    /// `fit`から`NoRegressors`を除いたもの。
    pub(crate) fn fit_allowing_no_regressors(
        input: OlsInput,
        cov_type: CovType,
        confidence_level: f64,
    ) -> Result<Self, LeastSquaresError> {
        // faer のグローバル並列度を Par::Seq に固定する（`crate::parallelism`）。
        crate::parallelism::ensure_serial();

        if !(confidence_level > 0.0 && confidence_level < 1.0) {
            return Err(CommonError::InvalidConfidenceLevel { confidence_level }.into());
        }

        let n = input.nobs();
        let k = input.k();

        if n <= k {
            return Err(CommonError::InsufficientObservations { n, k }.into());
        }

        // `cov_type=Cluster`のクラスター数`g`が傾き係数の数`q`（`k - k_constant`）以下だと、
        // クラスター寄与スコアの総和がゼロ（正規方程式`X'e = 0`）で`rank(Ŝ) ≤ g - 1`の
        // ため、ロバストWald/F検定の`q×q`部分行列が構造的に特異になる。
        // `g`・`q`は入力だけから判定できるため、QR分解・残差計算より前に弾く
        // （nonlinear/IVと同じく`fit()`冒頭で検証する方針に揃える）。`groups=None`は
        // 下の`CovType::Cluster`アームで`MissingClusterColumn`として扱う。
        if let CovType::Cluster {
            groups: Some(groups),
        } = &cov_type
        {
            let g = validate_cluster_groups(groups, n)?;
            validate_cluster_count_covers_slopes(g, k - usize::from(input.has_intercept()))?;
        }

        let qr = checked_col_piv_qr(input.x()).map_err(|_| LeastSquaresError::SingularMatrix)?;

        let params = qr.solve_lstsq(input.y());
        let residuals = input.y() - input.x() * &params;

        let df_resid = n - k;
        let ssr: f64 = (0..n).map(|i| (*residuals.get(i, 0)).powi(2)).sum();
        let sigma2 = ssr / (df_resid as f64);

        let xtx_inv = xtx_inverse(input.x(), k)?;

        // `df_inference`はt検定・信頼区間・F検定に使う自由度。通常は`df_resid`（n-k）と
        // 同じだが、`cov_type=Cluster`のときだけ`G-1`（クラスター数-1）に切り替える
        // （statsmodelsの`df_correction=True`という既定と同じ挙動。標準的な計量経済学の
        // 慣行でもある。`df_resid`自体は分散推定量`σ̂²`・調整済みR²・AIC/BIC等では
        // 引き続き`n-k`のまま使う。`docs/spec/ols-spec.md`
        // 「標準誤差」のクラスター参照）。
        let mut hac_lags_used = None;
        let (cov_params, df_inference) = match &cov_type {
            CovType::Classical => (classical_cov_params(sigma2, &xtx_inv, k), df_resid),
            CovType::Hc0 => (
                hc_cov_params(input.x(), &residuals, &xtx_inv, n, k, HcVariant::Hc0),
                df_resid,
            ),
            CovType::Hc1 => (
                hc_cov_params(input.x(), &residuals, &xtx_inv, n, k, HcVariant::Hc1),
                df_resid,
            ),
            CovType::Hc2 => (
                hc_cov_params(input.x(), &residuals, &xtx_inv, n, k, HcVariant::Hc2),
                df_resid,
            ),
            CovType::Hc3 => (
                hc_cov_params(input.x(), &residuals, &xtx_inv, n, k, HcVariant::Hc3),
                df_resid,
            ),
            CovType::Hac { lags, time_order } => {
                let lags = resolve_hac_lags(*lags, n)?;
                hac_lags_used = Some(lags);
                let order = time_ordering(time_order, n);
                (
                    hac_cov_params(input.x(), &residuals, &xtx_inv, n, k, lags, &order),
                    df_resid,
                )
            }
            CovType::Cluster { groups } => {
                let groups = groups.as_ref().ok_or(CommonError::MissingClusterColumn)?;
                // クラスター数`g >= 2`・`g > q`（傾き係数の数）は`fit()`冒頭で検証済み。
                // ここでは`n_groups - 1`（検定の自由度）に再利用するため
                // 再度ユニーク数を数えるだけ。
                let n_groups = validate_cluster_groups(groups, n)?;
                let cov = cluster_cov_params(input.x(), &residuals, &xtx_inv, n, k, groups);
                (cov, n_groups - 1)
            }
        };

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
        let sst: f64 = if input.has_intercept() {
            let y_mean: f64 = (0..n).map(|i| *input.y().get(i, 0)).sum::<f64>() / (n as f64);
            (0..n)
                .map(|i| (*input.y().get(i, 0) - y_mean).powi(2))
                .sum()
        } else {
            (0..n).map(|i| (*input.y().get(i, 0)).powi(2)).sum()
        };
        let r_squared = 1.0 - ssr / sst;
        let adj_r_squared = 1.0 - ((n - k_constant) as f64 / df_resid as f64) * (1.0 - r_squared);

        let log_likelihood =
            -(n as f64 / 2.0) * ((2.0 * std::f64::consts::PI).ln() + (ssr / n as f64).ln() + 1.0);
        let aic = -2.0 * log_likelihood + 2.0 * (k as f64);
        let bic = -2.0 * log_likelihood + (n as f64).ln() * (k as f64);

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
        wald_f_test(&self.params, &self.cov_params, k - q, q, self.df_inference)
    }

    /// 学習データに対する予測値 `ŷ = Xβ̂`（`predict(new_data=None)`のPython APIが返す値、
    /// `docs/spec/ols-spec.md`「predict()」参照）。`fit()`のReturn本体には含めず、
    /// 必要なときに計算する別メソッドとする（Logitの`predict()`と同じ設計方針）。
    pub fn fitted_values(&self) -> Mat<f64> {
        self.input.x() * &self.params
    }
}

/// `CovType::Hac`の`lags`（`Option<i64>`）を実際に使うラグ数（`usize`）に解決する。
///
/// `Some(l)`の場合は`0 <= l < n`を検証してそのまま使う。`None`の場合は経験則
/// `L = floor(4*(n/100)^(2/9))`で自動計算する（`docs/spec/ols-spec.md`
/// 「標準誤差」のHAC。EViews等でも使われる、データに依存しない決定的な式）。
fn resolve_hac_lags(lags: Option<i64>, n: usize) -> Result<usize, LeastSquaresError> {
    match lags {
        Some(l) => {
            if l < 0 || (l as usize) >= n {
                return Err(LeastSquaresError::InvalidHacLags { hac_lags: l, n });
            }
            Ok(l as usize)
        }
        None => Ok((4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize),
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
    fn fit_allowing_no_regressors_succeeds_when_k_is_zero() {
        // `fit`とは異なり`NoRegressors`ガードを迂回する（`panel::fe::FeEstimator::fit`が
        // 固定効果のみモデルのために直接呼ぶ経路）。k=0でもOkを返し、
        // paramsは空（0行）になる。
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = OlsInput::from_columns(&y, &[], vec![], false, "y".to_string()).unwrap();

        let estimator =
            OlsEstimator::fit_allowing_no_regressors(input, CovType::Classical, 0.95).unwrap();

        assert_eq!(estimator.params().nrows(), 0);
        assert_eq!(estimator.residuals().nrows(), 5);
        for (i, &yi) in y.iter().enumerate() {
            assert_eq!(*estimator.residuals().get(i, 0), yi);
        }
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
        // `fit()` 冒頭の `crate::parallelism::ensure_serial()` が faer の
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

    /// 同じデータセット（x=[1..5], y=[2,4,5,4,5]）でのHC0〜HC3。
    /// 期待値はstatsmodels 0.14.6で`use_t=True`を明示指定して独立に計算・検算済み
    /// （`sm.OLS(Y, X).fit(cov_type=..., use_t=True)`）。`use_t=True`が必要な理由は
    /// `docs/spec/ols-spec.md`「標準誤差」、
    /// および`OlsEstimator::fit`のdocコメント参照
    /// （statsmodelsはHC0-3でuse_t=Falseが既定＝正規分布のため、素の既定値とは一致しない）。
    #[test]
    fn fit_computes_hc_std_errors_test_stats_p_values_and_conf_int() {
        // (cov_type, [se_const, se_x1], [t_const, t_x1], [p_const, p_x1],
        //  [lower_const, lower_x1], [upper_const, upper_x1])
        #[allow(clippy::type_complexity)]
        let cases: [(CovType, [f64; 2], [f64; 2], [f64; 2], [f64; 2], [f64; 2]); 4] = [
            (
                CovType::Hc0,
                [0.741_350_119_714_024_1, 0.185_472_369_909_913_61],
                [2.967_558_703_367_644, 3.234_983_196_103_162_8],
                [0.059_183_855_836_541_795, 0.048_033_568_062_853_735],
                [-0.159_306_949_405_533_25, 0.009_744_141_647_982_651],
                [4.559_306_949_405_528, 1.190_255_858_352_018_2],
            ),
            (
                CovType::Hc1,
                [0.957_078_889_120_43, 0.239_443_799_947_572_34],
                [2.298_661_087_407_152_2, 2.505_807_208_753_678_2],
                [0.105_117_351_189_905_94, 0.087_259_022_565_828_92],
                [-0.845_852_174_546_351, -0.162_017_036_466_242_44],
                [5.245_852_174_546_345, 1.362_017_036_466_243_2],
            ),
            (
                CovType::Hc2,
                [1.106_216_202_066_430_8, 0.279_795_843_939_315_45],
                [1.988_761_325_218_668_2, 2.144_420_701_724_696_3],
                [0.140_853_196_409_229_89, 0.121_345_243_297_999_42],
                [-1.320_473_665_111_291_2, -0.290_435_249_778_410_95],
                [5.720_473_665_111_285, 1.490_435_249_778_412],
            ),
            (
                CovType::Hc3,
                [1.689_236_634_744_594_6, 0.429_760_255_899_750_37],
                [1.302_363_419_517_377_2, 1.396_127_240_160_527_1],
                [0.283_757_574_453_598_06, 0.257_051_084_412_133_5],
                [-3.175_904_886_992_822, -0.767_688_938_545_941],
                [7.575_904_886_992_816_5, 1.967_688_938_545_941_7],
            ),
        ];

        for (cov_type, se, t, p, lower, upper) in cases {
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

            let cov_type_label = format!("{cov_type:?}");
            let estimator = OlsEstimator::fit(input, cov_type, 0.95).unwrap();

            for j in 0..2 {
                let msg = format!("{cov_type_label}, param {j}");
                assert!(
                    (*estimator.std_errors().get(j, 0) - se[j]).abs() < 1e-6,
                    "std_errors mismatch: {msg}"
                );
                assert!(
                    (*estimator.test_stats().get(j, 0) - t[j]).abs() < 1e-6,
                    "test_stats mismatch: {msg}"
                );
                assert!(
                    (*estimator.p_values().get(j, 0) - p[j]).abs() < 1e-6,
                    "p_values mismatch: {msg}"
                );
                assert!(
                    (*estimator.conf_lower().get(j, 0) - lower[j]).abs() < 1e-6,
                    "conf_lower mismatch: {msg}"
                );
                assert!(
                    (*estimator.conf_upper().get(j, 0) - upper[j]).abs() < 1e-6,
                    "conf_upper mismatch: {msg}"
                );
            }
        }
    }

    /// 同じデータセット（x=[1..5], y=[2,4,5,4,5]）でのHAC（Newey-West、`maxlags=1`）。
    /// 期待値はstatsmodels 0.14.6で独立に計算・検算済み
    /// （`sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1}, use_t=True)`。
    /// `use_correction`はstatsmodelsの既定である`False`のまま、明示指定はしていない）。
    #[test]
    fn fit_computes_hac_std_errors_with_explicit_lags() {
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

        let cov_type = CovType::Hac {
            lags: Some(1),
            time_order: row_time_order(5),
        };
        let estimator = OlsEstimator::fit(input, cov_type, 0.95).unwrap();

        let se = estimator.std_errors();
        assert!((*se.get(0, 0) - 0.659_090_282_131_361_7).abs() < 1e-6);
        assert!((*se.get(1, 0) - 0.164_924_225_024_705_7).abs() < 1e-6);

        let t = estimator.test_stats();
        assert!((*t.get(0, 0) - 3.337_934_209_689_228).abs() < 1e-6);
        assert!((*t.get(1, 0) - 3.638_034_375_545_013_5).abs() < 1e-6);

        let p = estimator.p_values();
        assert!((*p.get(0, 0) - 0.044_455_744_969_471_62).abs() < 1e-6);
        assert!((*p.get(1, 0) - 0.035_791_053_269_350_51).abs() < 1e-6);

        let lower = estimator.conf_lower();
        let upper = estimator.conf_upper();
        assert!((*lower.get(0, 0) - 0.102_480_566_782_648_72).abs() < 1e-6);
        assert!((*upper.get(0, 0) - 4.297_519_433_217_346).abs() < 1e-6);
        assert!((*lower.get(1, 0) - 0.075_137_509_418_346_96).abs() < 1e-6);
        assert!((*upper.get(1, 0) - 1.124_862_490_581_653_8).abs() < 1e-6);
    }

    /// `hac_lags=None`（経験則自動計算）が`L = floor(4*(n/100)^(2/9))`と一致することを確認する。
    /// n=5の場合L=2。期待値はstatsmodelsで`maxlags=2`を明示指定して独立に計算・検算済み
    /// （`docs/spec/ols-spec.md`「標準誤差」のHACの式通りベンチマーク側もL=2を使う前提）。
    #[test]
    fn fit_computes_hac_std_errors_with_auto_lags() {
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

        let cov_type = CovType::Hac {
            lags: None,
            time_order: row_time_order(5),
        };
        let estimator = OlsEstimator::fit(input, cov_type, 0.95).unwrap();

        let se = estimator.std_errors();
        assert!((*se.get(0, 0) - 0.577_350_269_189_624_1).abs() < 1e-6);
        assert!((*se.get(1, 0) - 0.164_924_225_024_705_75).abs() < 1e-6);
    }

    /// `n`行の単純回帰（`y = x + (i % 7)`）を`cov_type`で推定し`hac_lags_used`を返す補助関数。
    fn hac_lags_used_for(n: usize, cov_type: CovType) -> Option<usize> {
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| i as f64 + (i % 7) as f64).collect();
        let input = OlsInput::from_columns(&y, &[x], vec!["x1".to_string()], true, "y".to_string())
            .unwrap();
        OlsEstimator::fit(input, cov_type, 0.95)
            .unwrap()
            .hac_lags_used()
    }

    /// `hac_lags`未指定（`None`）なら経験則`floor(4*(n/100)^(2/9))`で解決した値が
    /// `hac_lags_used()`に入る。`n=5`→2、`n=100`→4（`(n/100)^(2/9)=1`ちょうど）、
    /// `n=1000`→6（`4*10^(2/9)≈6.67`）。
    #[test]
    fn hac_lags_used_returns_auto_selected_lags_when_lags_is_none() {
        for (n, expected) in [(5, 2), (100, 4), (1000, 6)] {
            let cov_type = CovType::Hac {
                lags: None,
                time_order: row_time_order(n),
            };
            assert_eq!(hac_lags_used_for(n, cov_type), Some(expected), "n={n}");
        }
    }

    /// `hac_lags`を明示指定した場合は、自動計算値（`n=100`なら4）ではなく指定値が入る。
    #[test]
    fn hac_lags_used_returns_explicit_lags_when_specified() {
        for lags in [0, 3, 10] {
            let cov_type = CovType::Hac {
                lags: Some(lags),
                time_order: row_time_order(100),
            };
            assert_eq!(
                hac_lags_used_for(100, cov_type),
                Some(lags as usize),
                "lags={lags}"
            );
        }
    }

    /// `cov_type`がHac以外なら`None`。
    #[test]
    fn hac_lags_used_is_none_for_non_hac_cov_types() {
        for cov_type in [CovType::Classical, CovType::Hc0, CovType::Hc3] {
            let label = format!("{cov_type:?}");
            assert_eq!(hac_lags_used_for(20, cov_type), None, "{label}");
        }
    }

    /// `time_order`を指定した場合、行順がシャッフルされていても時系列順に並べ替えてから
    /// ラグ付き自己共分散を計算することを確認する。データはHAC(maxlags=1)テストと同一の
    /// (x, y)を、時系列順の逆転を含む順序（time値=xの値そのもの）でシャッフルして与える。
    /// 期待値は`fit_computes_hac_std_errors_with_explicit_lags`と同じ
    /// （時系列順に並べ替えれば同一データになるため）。
    #[test]
    fn fit_computes_hac_std_errors_respecting_time_order() {
        // 元の時系列順: time=[1,2,3,4,5], y=[2,4,5,4,5]
        // これを time順=[3,1,5,2,4] の並びでシャッフルして入力する
        let shuffled_time = vec![3.0, 1.0, 5.0, 2.0, 4.0];
        let shuffled_x = shuffled_time.clone();
        let shuffled_y = vec![5.0, 2.0, 5.0, 4.0, 4.0];

        let input = OlsInput::from_columns(
            &shuffled_y,
            &[shuffled_x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let cov_type = CovType::Hac {
            lags: Some(1),
            time_order: shuffled_time,
        };
        let estimator = OlsEstimator::fit(input, cov_type, 0.95).unwrap();

        let se = estimator.std_errors();
        assert!((*se.get(0, 0) - 0.659_090_282_131_361_7).abs() < 1e-6);
        assert!((*se.get(1, 0) - 0.164_924_225_024_705_7).abs() < 1e-6);
    }

    #[test]
    fn fit_returns_invalid_hac_lags_when_out_of_range() {
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

        let cov_type = CovType::Hac {
            lags: Some(-1),
            time_order: row_time_order(5),
        };
        let result = OlsEstimator::fit(input, cov_type, 0.95);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::InvalidHacLags { hac_lags: -1, n: 5 }
        );
    }

    /// `hac_lags`の上限側の境界。`n=5`のとき`lags=5`（`n`自体）は範囲外（`[0, n)`）、
    /// `lags=4`（`n-1`）は許容される最大値であることを確認する。
    #[test]
    fn fit_returns_invalid_hac_lags_when_equal_to_n() {
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

        let cov_type = CovType::Hac {
            lags: Some(5),
            time_order: row_time_order(5),
        };
        let result = OlsEstimator::fit(input, cov_type, 0.95);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::InvalidHacLags { hac_lags: 5, n: 5 }
        );
    }

    #[test]
    fn fit_accepts_hac_lags_at_upper_boundary_of_n_minus_one() {
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

        let cov_type = CovType::Hac {
            lags: Some(4), // n - 1、許容される最大値
            time_order: row_time_order(5),
        };
        let result = OlsEstimator::fit(input, cov_type, 0.95);

        assert!(result.is_ok());
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

    /// `CovType::Hac { lags: Some(0), .. }`は`Ŝ = Ŝ₀`（ラグ項なし）に退化し、これは
    /// `HC0`の`Ψ̂ = Σ_i ε̂_i² x_i x_i'`と数学的に同一の式になる（`hac_cov_params`の
    /// l=0項のドキュメント参照）。2つの独立した実装（`hc_cov_params`と`hac_cov_params`）が
    /// この境界で一致することを確認する内部整合性テスト。
    #[test]
    fn fit_hac_with_zero_lags_matches_hc0() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 8.0, 3.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];

        let input_hac = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let hac = OlsEstimator::fit(
            input_hac,
            CovType::Hac {
                lags: Some(0),
                time_order: row_time_order(6),
            },
            0.95,
        )
        .unwrap();

        let input_hc0 = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let hc0 = OlsEstimator::fit(input_hc0, CovType::Hc0, 0.95).unwrap();

        for j in 0..2 {
            assert!(
                (*hac.std_errors().get(j, 0) - *hc0.std_errors().get(j, 0)).abs() < 1e-12,
                "param {j}: hac(lags=0)={}, hc0={}",
                *hac.std_errors().get(j, 0),
                *hc0.std_errors().get(j, 0)
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

    /// 同じデータセット（x=[1..5], y=[2,4,5,4,5]）を2クラスター（groups=["a","a","b","b","b"]）
    /// でのクラスターロバスト標準誤差。期待値はstatsmodels 0.14.6で独立に計算・検算済み
    /// （`sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)`。
    /// 小標本補正はstatsmodelsの既定`use_correction=True`のまま、明示指定はしていない）。
    #[test]
    fn fit_computes_cluster_std_errors_test_stats_p_values_conf_int_and_f_test() {
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

        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];
        let cov_type = CovType::Cluster {
            groups: Some(groups),
        };
        let estimator = OlsEstimator::fit(input, cov_type, 0.95).unwrap();

        let se = estimator.std_errors();
        assert!((*se.get(0, 0) - 0.785_196_366_097_886_8).abs() < 1e-6);
        assert!((*se.get(1, 0) - 0.230_940_107_675_849_05).abs() < 1e-6);

        let t = estimator.test_stats();
        assert!((*t.get(0, 0) - 2.801_846_894_596_724_5).abs() < 1e-6);
        assert!((*t.get(1, 0) - 2.598_076_211_353_332).abs() < 1e-6);

        let p = estimator.p_values();
        assert!((*p.get(0, 0) - 0.218_242_895_017_685_43).abs() < 1e-6);
        assert!((*p.get(1, 0) - 0.233_908_049_281_92).abs() < 1e-6);

        let lower = estimator.conf_lower();
        let upper = estimator.conf_upper();
        assert!((*lower.get(0, 0) - (-7.776_865_785_740_13)).abs() < 1e-6);
        assert!((*upper.get(0, 0) - 12.176_865_785_740_125).abs() < 1e-6);
        assert!((*lower.get(1, 0) - (-2.334_372_289_923_566_6)).abs() < 1e-6);
        assert!((*upper.get(1, 0) - 3.534_372_289_923_567_7).abs() < 1e-6);

        assert!((estimator.f_statistic() - 6.750_000_000_000_083_5).abs() < 1e-6);
        assert!((estimator.f_p_value() - 0.233_908_049_281_92).abs() < 1e-6);
    }

    /// クラスターロバスト標準誤差は、同じ入力に対して`fit()`を複数回呼んでも
    /// ビット単位で同じ結果になること（`cluster_cov_params`内部の集約が
    /// `HashMap`の反復順序に依存し、実行のたびに浮動小数点の丸め誤差レベルで
    /// 結果がぶれていた回帰。`BTreeMap`化で修正した）。
    #[test]
    fn fit_cluster_std_errors_are_deterministic_across_repeated_fits() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0, 3.0, 6.0, 1.0, 7.0, 2.5];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let groups: Vec<String> = (0..10).map(|i| format!("g{}", i % 4)).collect();

        let build = || {
            let input = OlsInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();
            let cov_type = CovType::Cluster {
                groups: Some(groups.clone()),
            };
            OlsEstimator::fit(input, cov_type, 0.95).unwrap()
        };

        let first = build();
        for _ in 0..20 {
            let repeat = build();
            assert_eq!(
                *repeat.std_errors().get(0, 0),
                *first.std_errors().get(0, 0)
            );
            assert_eq!(
                *repeat.std_errors().get(1, 0),
                *first.std_errors().get(1, 0)
            );
        }
    }

    #[test]
    fn fit_returns_missing_cluster_column_when_groups_not_provided() {
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

        let result = OlsEstimator::fit(input, CovType::Cluster { groups: None }, 0.95);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::MissingClusterColumn)
        );
    }

    #[test]
    fn fit_returns_insufficient_clusters_when_only_one_group() {
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

        let groups = vec!["a".to_string(); 5];
        let cov_type = CovType::Cluster {
            groups: Some(groups),
        };
        let result = OlsEstimator::fit(input, cov_type, 0.95);

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::InsufficientClusters { g: 1 })
        );
    }

    /// `cov_type=Cluster`でクラスター数`g`が傾き係数の数`q`以下だと、クラスター寄与
    /// スコアの総和がゼロ（正規方程式`X'e = 0`）で`rank(Ŝ) ≤ g - 1`のため、ロバスト
    /// Wald/F検定の`q×q`部分行列が構造的に特異になる。`g`・`q`は入力だけから判定
    /// できるため`fit()`のバリデーションが`CommonError::InsufficientClustersForInference`
    /// で弾く。切片＋説明変数3個（`q = 4 - 1 = 3`）に対し`g = 2`
    /// （`g < q`）と`g = 3`（`g == q`、`rank(Ŝ) ≤ 2 < 3`で依然特異）の両方を確認する。
    /// backstop（`g > q`だが悪条件で数値的にほぼ特異）は
    /// `fit_returns_computation_failed_for_cluster_when_slope_submatrix_is_ill_conditioned`。
    #[test]
    fn fit_returns_validation_error_when_cluster_count_at_most_slopes() {
        let n = 12;
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * i as f64).collect();
        let x_columns = vec![
            (0..n).map(|i| i as f64).collect::<Vec<f64>>(),
            (0..n).map(|i| (i * i) as f64).collect::<Vec<f64>>(),
            (0..n).map(|i| ((i % 4) + 1) as f64).collect::<Vec<f64>>(),
        ];
        for g in [2_usize, 3_usize] {
            let groups: Vec<String> = (0..n).map(|i| format!("g{}", i % g)).collect();
            let input = OlsInput::from_columns(
                &y,
                &x_columns,
                vec!["x1".to_string(), "x2".to_string(), "x3".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();
            let result = OlsEstimator::fit(
                input,
                CovType::Cluster {
                    groups: Some(groups),
                },
                0.95,
            );
            assert_eq!(
                result.unwrap_err(),
                LeastSquaresError::Common(CommonError::InsufficientClustersForInference {
                    g,
                    q: 3
                }),
                "g={g}"
            );
        }
    }

    /// `cov_type=Cluster`かつ`g > q`（`fit()`冒頭のバリデーションを通過）でも、傾き係数
    /// 間のスケールが極端に異なると、ロバストWald/F検定の`q×q`部分行列の条件数が倍精度の
    /// 限界を超え`ensure_well_conditioned_symmetric_matrix`が`ComputationFailed`で止める
    /// （`fit_returns_computation_failed_for_extreme_scale_difference_in_f_test`のcluster版。
    /// `cov_type`によらず傾き部分行列の条件数は同じだが、cluster経路でもこのbackstopが
    /// 生きていることを明示的に固定する）。`x1`は1e6・`x2`は1e-3オーダー、`g=4 > q=3`。
    #[test]
    fn fit_returns_computation_failed_for_cluster_when_slope_submatrix_is_ill_conditioned() {
        let n = 12;
        let x1: Vec<f64> = (1..=n).map(|i| 1e6 * (i as f64)).collect();
        let x2: Vec<f64> = (1..=n).map(|i| 1e-3 * (i as f64).powi(2)).collect();
        let x3: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let noise = if i % 2 == 0 { 0.1 } else { -0.1 };
                1.0 + 2.0 * x1[i] + 3.0 * x2[i] + 0.5 * x3[i] + noise
            })
            .collect();
        let groups: Vec<String> = (0..n).map(|i| format!("g{}", i % 4)).collect();
        let input = OlsInput::from_columns(
            &y,
            &[x1, x2, x3],
            vec!["x1".to_string(), "x2".to_string(), "x3".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = OlsEstimator::fit(
            input,
            CovType::Cluster {
                groups: Some(groups),
            },
            0.95,
        );

        assert!(matches!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::ComputationFailed(_))
        ));
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

    #[test]
    #[should_panic]
    fn fit_panics_when_cluster_groups_length_does_not_match_nobs() {
        // groups.len() != nはengine_pybind側の実装バグでしか起こり得ない内部契約違反のため、
        // Errではなくdebug_assert!でパニックする（from_columns_panics_on_mismatched_names_arity
        // と同じ性質）。
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

        let groups = vec!["a".to_string(), "b".to_string()]; // n=5のはずが長さ2
        let cov_type = CovType::Cluster {
            groups: Some(groups),
        };
        let _ = OlsEstimator::fit(input, cov_type, 0.95);
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
