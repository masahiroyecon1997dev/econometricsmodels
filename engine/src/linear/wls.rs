//! WLS（Weighted Least Squares）の推定。
//!
//! `docs/spec/wls-spec.md`で確定した設計に基づき、`sqrt(weight)`で
//! 変換した`OlsInput`（`OlsInput::from_columns_weighted`）に対して既存の`OlsEstimator::fit`
//! （無変更）をそのまま適用する。標準誤差・係数・t値・p値・信頼区間・F統計量の計算式が
//! 変換後データに対するOLSの計算式そのままで正しいことの確認は
//! `docs/spec/wls-spec.md`「標準誤差」参照。
//!
//! **ただしR²・対数尤度（→AIC/BIC）はこの変換だけでは正しくならない**（正しい導出・
//! statsmodelsとの数式対応は`docs/spec/wls-spec.md`「適合度統計量」参照）。
//! そのためこの2つ（とそこから導かれるR²調整済み・AIC・BIC）は`WlsEstimator::fit`側で
//! 元の（変換前の）`y`・`weights`を使って計算し直す。`OlsEstimator`/`OlsInput`自体は
//! 重みを一切知らない設計のまま変更しない（係数・SE・F統計量の構造的保証は保つ）。
//!
//! # ファイル構成
//!
//! このファイルは外から見える面（`WlsEstimator`の構造体・getter・`fit`の入口）だけを持つ。
//! ロジックは子モジュールに置き、テストは各ファイルの`mod tests`に置く。
//!
//! - `fit.rs`: `fit`の本体（重み変換した`OlsInput`の推定→元スケールの予測値・残差→適合度の組み立て）
//! - `statistics.rs`: 元スケールの予測値・残差と、重み付きのR²・対数尤度・AIC・BICの計算

use super::common::LeastSquaresError;
use super::ols::{CovType, OlsEstimator};

mod fit;
mod statistics;

/// WLSの推定結果。
///
/// `estimator()`が返す`OlsEstimator`本体は、`sqrt(weight)`で変換したデータに対する
/// 計算結果であり、`params`/`std_errors`/`test_stats`/`p_values`/`conf_lower`/`conf_upper`/
/// `f_statistic`/`f_p_value`はここから取得する（変換後データに対するOLSの計算式が
/// そのまま正しいため。重みが全て1のときOLSと数値的に完全一致するのもこの型を経由する
/// ためで、`docs/spec/wls-spec.md`「sqrt(w)変換」の構造的保証がそのまま成り立つ）。
///
/// 一方`r_squared`/`adj_r_squared`/`log_likelihood`/`aic`/`bic`は`estimator()`側の値を
/// 使わず、この型が元の（変換前の）`y`・`weights`から計算し直した値を使う（モジュール
/// 冒頭のdocコメント参照）。
///
/// `estimator().residuals()`は**重み付き残差** `sqrt(w_i)(y_i - x_i'β̂)`
/// （statsmodelsでいう`.wresid`相当）を返す。ユーザー向けに公開する残差は元スケール
/// （unweighted）の`residuals()`を使う（`docs/spec/wls-spec.md`「結果構造体」参照）。
#[derive(Debug)]
pub struct WlsEstimator {
    estimator: OlsEstimator,
    /// 元スケール（unweighted）の予測値 `ŷ_i = x_i'β̂`（`predict(new_data=None)`が返す値）
    fitted_values: Vec<f64>,
    /// 元スケール（unweighted）の残差 `y_i - x_i'β̂`
    residuals: Vec<f64>,
    r_squared: f64,
    adj_r_squared: f64,
    log_likelihood: f64,
    aic: f64,
    bic: f64,
}

impl WlsEstimator {
    /// 列ごとの`Vec<f64>`（`engine_pybind`がpolars DataFrameから抽出済み）と重み配列から
    /// WLSを推定する。
    ///
    /// # Errors
    /// `OlsInput::from_columns_weighted`・`OlsEstimator::fit`が返しうるエラーをそのまま返す
    /// （次元不一致・非正の重み・観測数不足・特異行列・信頼水準の範囲外・クラスター数不足・
    /// `hac_lags`の範囲外等）。
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        include_intercept: bool,
        dep_var_name: String,
        weights: &[f64],
        cov_type: CovType,
        confidence_level: f64,
    ) -> Result<Self, LeastSquaresError> {
        fit::fit_wls(
            y,
            x_columns,
            x_names,
            include_intercept,
            dep_var_name,
            weights,
            cov_type,
            confidence_level,
        )
    }

    /// 変換後データ（重み付き）に対する`OlsEstimator`本体。
    ///
    /// `r_squared`/`adj_r_squared`/`log_likelihood`/`aic`/`bic`はここではなく
    /// `WlsEstimator`自身のメソッドを使うこと（型ドキュメント参照）。
    pub fn estimator(&self) -> &OlsEstimator {
        &self.estimator
    }

    /// 元スケール（unweighted）の予測値 `ŷ_i = x_i'β̂`（学習データに対する`predict()`が返す値）。
    pub fn fitted_values(&self) -> &[f64] {
        &self.fitted_values
    }

    /// 元スケール（unweighted）の残差 `y_i - x_i'β̂`。
    pub fn residuals(&self) -> &[f64] {
        &self.residuals
    }

    /// 決定係数（切片ありなら元のyの重み付き平均を使ったcentered TSS、
    /// 切片なしならuncentered TSSに基づく。statsmodelsの`WLS`と同じ定義）。
    pub fn r_squared(&self) -> f64 {
        self.r_squared
    }

    /// 自由度調整済み決定係数。
    pub fn adj_r_squared(&self) -> f64 {
        self.adj_r_squared
    }

    /// 対数尤度（`sqrt(weight)`変換のヤコビアン補正込み）。
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// 赤池情報量規準。
    pub fn aic(&self) -> f64 {
        self.aic
    }

    /// ベイズ情報量規準。
    pub fn bic(&self) -> f64 {
        self.bic
    }
}
