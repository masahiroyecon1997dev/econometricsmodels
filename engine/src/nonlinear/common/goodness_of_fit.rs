//! 適合度統計量（切片のみモデルの対数尤度、尤度比検定、疑似R²、AIC・BIC）。

use faer::Mat;
use statrs::distribution::{ChiSquared, ContinuousCDF};

use super::error::MleError;
use crate::shared::error::CommonError;

/// 切片のみモデルの対数尤度 `ℓ_null = n1*ln(ȳ) + n0*ln(1-ȳ)`（`ȳ=n1/n`の閉じた形の
/// 解析解）を`y`から直接計算する。Logit/Probitいずれも「切片のみモデルのMLEは
/// `link(θ̂)=ȳ`を満たす」という性質（`fit_newton_converges_to_closed_form_solution_
/// for_intercept_only_model`で検証済み）を持つため、この式はリンク関数（ロジスティック/
/// 標準正規CDF）に依存しない。ベルヌーイ尤度が切片のみモデルでは`p=link(θ)`のみに
/// 依存する形に落ちることに由来する（`docs/spec/logit-spec.md`「適合度統計量」節参照）。
///
/// `n1`または`n0`が0（全観測が同じ値）のときの`0*ln(0)`（NaN）を避けるため、該当項を
/// 明示的に0として扱う（情報理論の`0 log 0 = 0`規約）。`Result`を経由しない内部専用の
/// 計算（適合度統計量向け、収束後に1回だけ評価する）で、退化ケースを`fit()`の反復
/// 最適化を経由せず直接テストできるよう独立した関数として切り出している
/// （Logit/Probitで同一のコードになるため、ここに集約している）。
pub fn log_likelihood_null(y: &Mat<f64>) -> f64 {
    let n = y.nrows();
    let n1: f64 = (0..n).map(|i| *y.get(i, 0)).sum();
    let n0 = n as f64 - n1;
    let y_bar = n1 / n as f64;
    (if n1 > 0.0 { n1 * y_bar.ln() } else { 0.0 })
        + (if n0 > 0.0 {
            n0 * (1.0 - y_bar).ln()
        } else {
            0.0
        })
}

/// 対数尤度から導かれる適合度統計量一式（`docs/spec/nonlinear-common.md`
/// 5章参照）。`goodness_of_fit`の戻り値。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GoodnessOfFit {
    /// 尤度比検定統計量 `2*(ℓ(θ̂)-ℓ(θ̂_null))`
    pub lr_statistic: f64,
    /// 尤度比検定のp値（自由度`df_model`のカイ二乗分布、上側確率）。`df_model==0`のときNaN
    pub lr_p_value: f64,
    /// McFadden疑似決定係数 `1 - ℓ(θ̂)/ℓ(θ̂_null)`
    pub pseudo_r_squared: f64,
    /// 赤池情報量規準 `-2ℓ(θ̂) + 2k`
    pub aic: f64,
    /// ベイズ情報量規準 `-2ℓ(θ̂) + ln(n)k`
    pub bic: f64,
    /// モデルの自由度 `k-1`
    pub df_model: usize,
    /// 残差自由度 `n-k`
    pub df_resid: usize,
}

/// 収束点の対数尤度`llf`・切片のみモデルの対数尤度`llnull`（[`log_likelihood_null`]）・
/// 観測数`n`・パラメータ数`k`から[`GoodnessOfFit`]一式を計算する。モデル固有の尤度計算
/// （リンク関数）には依存しないため、Logit/Probit間で完全に共通の計算として
/// `nonlinear/common.rs`に置く（Logit/Probitの`fit()`で同一のコードになるため集約している）。
///
/// `df_model = k.saturating_sub(1)`: `include_intercept=false`かつ説明変数も無い
/// （`k=0`）という病的な入力でもアンダーフローしないための防御。`df_model==0`
/// （説明変数が定数項のみ、または定数項も無い）のときの`lr_p_value`は、検定対象の
/// 傾き係数が存在しないためOLSの`f_p_value`と同じ扱いでNaNを返す。
///
/// # Errors
/// `df_model>0`が保証されている経路でのみ`ChiSquared::new`を呼ぶため、失敗は理論上
/// 到達不能（`.claude/rules/rust-style.md`「テスト」のカバレッジ方針参照）。呼び出し側の
/// `?`演算子での伝播のため`Result`を返す。
pub fn goodness_of_fit(
    llf: f64,
    llnull: f64,
    n: usize,
    k: usize,
) -> Result<GoodnessOfFit, MleError> {
    let lr_statistic = 2.0 * (llf - llnull);
    let df_model = k.saturating_sub(1);
    let lr_p_value = if df_model == 0 {
        f64::NAN
    } else {
        let chi2 =
            ChiSquared::new(df_model as f64).map_err(|e| CommonError::internal(e.to_string()))?;
        chi2.sf(lr_statistic)
    };

    Ok(GoodnessOfFit {
        lr_statistic,
        lr_p_value,
        pseudo_r_squared: 1.0 - llf / llnull,
        aic: -2.0 * llf + 2.0 * (k as f64),
        bic: -2.0 * llf + (n as f64).ln() * (k as f64),
        df_model,
        df_resid: n - k,
    })
}

#[cfg(test)]
mod tests {

    use super::*;

    /// `log_likelihood_null`は`n1`（y=1の観測数）または`n0`（y=0の観測数）が0の
    /// 退化ケース（全観測が同じ値）で、`0*ln(0)`によるNaNを避ける分岐（`if n1 > 0.0
    /// {...} else {0.0}`等）を通る。この分岐は`fit()`経由（反復最適化）だと、全観測が
    /// 同じyの場合に完全分離が起きて収束の挙動が不安定になりうるため、`fit()`を
    /// 経由せず`log_likelihood_null`を直接呼んで検証する（Logitのカバレッジ確認
    /// 判明したギャップと同じ理由。common.rsへの集約に伴い、
    /// このテストもLogit/Probitそれぞれの重複から`goodness_of_fit.rs`へ集約した）。
    #[test]
    fn log_likelihood_null_returns_zero_for_degenerate_all_same_y() {
        // 全観測y=1（n0=0）: n0側の`else{0.0}`分岐を通る。ℓ_null = n*ln(1) + 0 = 0
        let all_ones = Mat::from_fn(4, 1, |_, _| 1.0);
        assert!((log_likelihood_null(&all_ones) - 0.0).abs() < 1e-12);

        // 全観測y=0（n1=0）: n1側の`else{0.0}`分岐を通る。ℓ_null = 0 + n*ln(1) = 0
        let all_zeros = Mat::from_fn(4, 1, |_, _| 0.0);
        assert!((log_likelihood_null(&all_zeros) - 0.0).abs() < 1e-12);
    }

    /// `goodness_of_fit`をLogit/Probitの尤度計算とは独立に、直接与えた`llf`/`llnull`から
    /// 検証する。数式は`docs/spec/nonlinear-common.md`5章通り。
    #[test]
    fn goodness_of_fit_computes_expected_statistics() {
        let llf = -5.0;
        let llnull = -8.0;
        let n = 10;
        let k = 3;

        let gof = goodness_of_fit(llf, llnull, n, k).unwrap();

        assert!((gof.lr_statistic - 2.0 * (llf - llnull)).abs() < 1e-12);
        assert_eq!(gof.df_model, k - 1);
        assert_eq!(gof.df_resid, n - k);
        assert!((gof.pseudo_r_squared - (1.0 - llf / llnull)).abs() < 1e-12);
        assert!((gof.aic - (-2.0 * llf + 2.0 * k as f64)).abs() < 1e-12);
        assert!((gof.bic - (-2.0 * llf + (n as f64).ln() * k as f64)).abs() < 1e-12);

        let chi2 = ChiSquared::new((k - 1) as f64).unwrap();
        let expected_lr_p = 1.0 - chi2.cdf(gof.lr_statistic);
        assert!((gof.lr_p_value - expected_lr_p).abs() < 1e-12);
    }

    /// `df_model==0`（`k<=1`）のとき`lr_p_value`はNaN（検定対象の傾き係数が存在しない
    /// ため、OLSの`f_p_value`と同じ扱い）。
    #[test]
    fn goodness_of_fit_returns_nan_lr_p_value_when_df_model_is_zero() {
        let gof = goodness_of_fit(-5.0, -8.0, 10, 1).unwrap();
        assert_eq!(gof.df_model, 0);
        assert!(gof.lr_p_value.is_nan());
    }
}
