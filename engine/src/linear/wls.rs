//! WLS（Weighted Least Squares）の推定。
//!
//! `docs/planning/specs/wls-api-design.md`で確定した設計に基づき、`sqrt(weight)`で
//! 変換した`OlsInput`（`OlsInput::from_columns_weighted`）に対して既存の`OlsEstimator::fit`
//! （無変更）をそのまま適用する。標準誤差・係数・t値・p値・信頼区間・F統計量の計算式が
//! 変換後データに対するOLSの計算式そのままで正しいことの確認は
//! `docs/planning/specs/wls-standard-errors.md`参照。
//!
//! **ただしR²・対数尤度（→AIC/BIC）はこの変換だけでは正しくならない**（Issue #44で
//! statsmodelsとのクロスチェック時に判明）。理由:
//! - R²（切片ありの場合のcentered TSS）: 変換後yの単純平均を使うと、正しい
//!   「元のyの重み付き平均」（`Σw_i y_i/Σw_i`）とは異なる値になる。
//! - 対数尤度: 変換後データに対するOLSの対数尤度は、`sqrt(w)`変換のヤコビアンに
//!   由来する補正項`+0.5·Σlog(w_i)`が欠落する（statsmodelsの`WLS.loglike`と同じ導出）。
//!
//! そのためこの2つ（とそこから導かれるR²調整済み・AIC・BIC）は`WlsEstimator::fit`側で
//! 元の（変換前の）`y`・`weights`を使って計算し直す。`OlsEstimator`/`OlsInput`自体は
//! 重みを一切知らない設計のまま変更しない（係数・SE・F統計量の構造的保証は保つ）。

use super::common::LeastSquaresError;
use super::ols::{CovType, OlsEstimator, OlsInput};

/// WLSの推定結果。
///
/// `estimator()`が返す`OlsEstimator`本体は、`sqrt(weight)`で変換したデータに対する
/// 計算結果であり、`params`/`std_errors`/`t_stats`/`p_values`/`conf_lower`/`conf_upper`/
/// `f_statistic`/`f_p_value`はここから取得する（変換後データに対するOLSの計算式が
/// そのまま正しいため。重みが全て1のときOLSと数値的に完全一致するのもこの型を経由する
/// ためで、`docs/planning/specs/wls-api-design.md`4.1節の構造的保証がそのまま成り立つ）。
///
/// 一方`r_squared`/`r_squared_adj`/`log_likelihood`/`aic`/`bic`は`estimator()`側の値を
/// 使わず、この型が元の（変換前の）`y`・`weights`から計算し直した値を使う（モジュール
/// 冒頭のdocコメント参照）。
///
/// `estimator().residuals()`は**重み付き残差** `sqrt(w_i)(y_i - x_i'β̂)`
/// （statsmodelsでいう`.wresid`相当）を返す。ユーザー向けに公開する残差は元スケール
/// （unweighted）の`residuals()`を使う（`docs/planning/specs/wls-api-design.md`4.3節参照）。
#[derive(Debug)]
pub struct WlsEstimator {
    estimator: OlsEstimator,
    /// 元スケール（unweighted）の残差 `y_i - x_i'β̂`
    residuals: Vec<f64>,
    r_squared: f64,
    r_squared_adj: f64,
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
        let input = OlsInput::from_columns_weighted(
            y,
            x_columns,
            x_names,
            include_intercept,
            dep_var_name,
            weights,
        )?;
        let estimator = OlsEstimator::fit(input, cov_type, confidence_level)?;
        let residuals = original_scale_residuals(y, x_columns, include_intercept, &estimator);
        let (r_squared, r_squared_adj, log_likelihood, aic, bic) =
            weighted_fit_statistics(y, weights, &residuals, &estimator);

        Ok(Self {
            estimator,
            residuals,
            r_squared,
            r_squared_adj,
            log_likelihood,
            aic,
            bic,
        })
    }

    /// 変換後データ（重み付き）に対する`OlsEstimator`本体。
    ///
    /// `r_squared`/`r_squared_adj`/`log_likelihood`/`aic`/`bic`はここではなく
    /// `WlsEstimator`自身のメソッドを使うこと（型ドキュメント参照）。
    pub fn estimator(&self) -> &OlsEstimator {
        &self.estimator
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
    pub fn r_squared_adj(&self) -> f64 {
        self.r_squared_adj
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

/// R²・調整済みR²・対数尤度・AIC・BICを、元の（変換前の）`y`・`weights`から計算し直す。
///
/// 統計量ごとの理由はモジュール冒頭のdocコメント参照。SSR自体は重み付き残差の二乗和
/// （`Σ w_i (y_i - ŷ_i)²`）で、`estimator`が変換後データに対して計算したSSRと数学的に
/// 同一の値になる（`sqrt(w_i)(y_i-ŷ_i)`の二乗が`w_i(y_i-ŷ_i)²`のため）ため、
/// `original_scale_residuals`と`weights`から計算し直しても内部で二重計算にはならない。
fn weighted_fit_statistics(
    y: &[f64],
    weights: &[f64],
    residuals: &[f64],
    estimator: &OlsEstimator,
) -> (f64, f64, f64, f64, f64) {
    let n = estimator.input().nobs();
    let k = estimator.input().k();
    let k_constant = usize::from(estimator.input().has_intercept());
    let df_resid = n - k;

    let ssr: f64 = residuals.iter().zip(weights).map(|(r, w)| w * r * r).sum();

    let sst: f64 = if estimator.input().has_intercept() {
        let weight_sum: f64 = weights.iter().sum();
        let weighted_mean: f64 =
            y.iter().zip(weights).map(|(v, w)| v * w).sum::<f64>() / weight_sum;
        y.iter()
            .zip(weights)
            .map(|(v, w)| w * (v - weighted_mean).powi(2))
            .sum()
    } else {
        y.iter().zip(weights).map(|(v, w)| w * v * v).sum()
    };

    let r_squared = 1.0 - ssr / sst;
    let r_squared_adj = 1.0 - ((n - k_constant) as f64 / df_resid as f64) * (1.0 - r_squared);

    // statsmodels WLS.loglikeと同じ導出（変換後データに対するOLSの対数尤度に、
    // sqrt(weight)変換のヤコビアン補正項 +0.5*Σlog(w_i) を加える）。
    let sum_log_weights: f64 = weights.iter().map(|w| w.ln()).sum();
    let log_likelihood = -(n as f64 / 2.0)
        * ((2.0 * std::f64::consts::PI).ln() + (ssr / n as f64).ln() + 1.0)
        + 0.5 * sum_log_weights;
    let aic = -2.0 * log_likelihood + 2.0 * (k as f64);
    let bic = -2.0 * log_likelihood + (n as f64).ln() * (k as f64);

    (r_squared, r_squared_adj, log_likelihood, aic, bic)
}

/// 元の（重み変換前の）`y`・`x_columns`と推定済みの係数から、元スケールの残差
/// `y_i - x_i'β̂`を計算する。`estimator.residuals()`（重み付き残差）をそのまま使わない理由は
/// `WlsEstimator`のdocコメントを参照。
fn original_scale_residuals(
    y: &[f64],
    x_columns: &[Vec<f64>],
    include_intercept: bool,
    estimator: &OlsEstimator,
) -> Vec<f64> {
    let params = estimator.params();
    (0..y.len())
        .map(|i| {
            let mut fitted = 0.0;
            let mut col = 0;
            if include_intercept {
                fitted += *params.get(0, 0);
                col = 1;
            }
            for (j, x_col) in x_columns.iter().enumerate() {
                fitted += x_col[i] * *params.get(col + j, 0);
            }
            y[i] - fitted
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_with_all_weights_one_matches_ols() {
        // wls-api-design.md 4.1節の構造的保証: 重みが全て1のときWLSはOLSと完全一致する。
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let weights = vec![1.0; 5];

        let wls = WlsEstimator::fit(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            &weights,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let ols_input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let ols = OlsEstimator::fit(ols_input, CovType::Classical, 0.95).unwrap();

        for j in 0..2 {
            assert_eq!(*wls.estimator().params().get(j, 0), *ols.params().get(j, 0));
            assert_eq!(
                *wls.estimator().std_errors().get(j, 0),
                *ols.std_errors().get(j, 0)
            );
            assert_eq!(
                *wls.estimator().conf_lower().get(j, 0),
                *ols.conf_lower().get(j, 0)
            );
        }
        assert_eq!(wls.estimator().f_statistic(), ols.f_statistic());
        assert_eq!(wls.estimator().f_p_value(), ols.f_p_value());

        // r_squared/r_squared_adj/log_likelihood/aic/bicはWlsEstimator側の（元スケールの
        // y・weightsから計算し直した）値を使う。重みが全て1なら重み付き平均=単純平均、
        // ヤコビアン補正項0.5*Σlog(1)=0となり、OLSと完全一致するはず
        // （Issue #44でstatsmodelsとのクロスチェック時に判明した、WlsEstimator側での
        // 再計算が必要な理由はモジュール冒頭のdocコメント参照）。
        // weights=1でも、重み付き平均（Σw*y/Σw）と単純平均（Σy/n）は数学的には同じ値だが
        // 浮動小数点の加算順序が異なるため、完全な==ではなく丸め誤差レベルで比較する。
        assert!((wls.r_squared() - ols.r_squared()).abs() < 1e-12);
        assert!((wls.r_squared_adj() - ols.r_squared_adj()).abs() < 1e-12);
        assert!((wls.log_likelihood() - ols.log_likelihood()).abs() < 1e-9);
        assert!((wls.aic() - ols.aic()).abs() < 1e-9);
        assert!((wls.bic() - ols.bic()).abs() < 1e-9);

        // 残差の計算経路自体はwls.rs（手動ループ）とols.rs（faerの行列演算）で異なるため、
        // 丸め誤差レベルでの一致を確認する（構造的な完全一致の保証対象はestimator()側）。
        for i in 0..5 {
            assert!((wls.residuals()[i] - *ols.residuals().get(i, 0)).abs() < 1e-12);
        }
    }

    /// x=[1..5], y=[2,4,5,4,5], weights=[1,4,0.25,9,2]（切片あり、classical）の
    /// 適合度統計量。期待値はstatsmodels 0.14.6で独立に計算・検算済み
    /// （`sm.WLS(y, sm.add_constant(x1), weights=w).fit(use_t=True)`）。
    /// Issue #44でのstatsmodelsクロスチェック時に発覚した、r_squared/log_likelihood
    /// （→aic/bic）の計算式修正（モジュール冒頭のdocコメント参照）を固定するための
    /// エンジン単体テスト。
    #[test]
    fn fit_computes_r_squared_and_information_criteria_matching_statsmodels() {
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let weights = vec![1.0, 4.0, 0.25, 9.0, 2.0];

        let wls = WlsEstimator::fit(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            &weights,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        assert!((*wls.estimator().params().get(0, 0) - 2.783_764_87).abs() < 1e-6);
        assert!((*wls.estimator().params().get(1, 0) - 0.358_992_3).abs() < 1e-6);
        assert!((*wls.estimator().std_errors().get(0, 0) - 0.824_058_49).abs() < 1e-6);
        assert!((*wls.estimator().std_errors().get(1, 0) - 0.227_478_42).abs() < 1e-6);
        assert!((wls.r_squared() - 0.453_603_574_100_183_8).abs() < 1e-9);
        assert!((wls.r_squared_adj() - 0.271_471_432_133_578_5).abs() < 1e-9);
        assert!((wls.log_likelihood() - (-4.694_800_450_440_493)).abs() < 1e-9);
        assert!((wls.aic() - 13.389_600_900_880_986).abs() < 1e-9);
        assert!((wls.bic() - 12.608_476_725_749_187).abs() < 1e-9);
        assert!((wls.estimator().f_statistic() - 2.490_519_076_986_167_6).abs() < 1e-6);
        assert!((wls.estimator().f_p_value() - 0.212_642_917_457_664_06).abs() < 1e-6);
    }

    #[test]
    fn fit_recovers_known_coefficients_with_nonuniform_weights() {
        // y = 1 + 2*x（ノイズなしの厳密解）に対しては、重みを変えても推定値は変わらないはず。
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let x_columns = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0]];
        let weights = vec![1.0, 2.0, 0.5, 3.0, 1.5];

        let wls = WlsEstimator::fit(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            &weights,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        assert!((*wls.estimator().params().get(0, 0) - 1.0).abs() < 1e-9);
        assert!((*wls.estimator().params().get(1, 0) - 2.0).abs() < 1e-9);
        for r in wls.residuals() {
            assert!(r.abs() < 1e-9);
        }
    }

    #[test]
    fn fit_propagates_non_positive_weight_error() {
        let y = vec![1.0, 2.0, 3.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0]];
        let weights = vec![1.0, 0.0, 1.0];

        let result = WlsEstimator::fit(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            &weights,
            CovType::Classical,
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::NonPositiveWeight {
                row: 1,
                weight: 0.0
            }
        );
    }

    #[test]
    fn fit_matches_manually_transformed_ols_for_all_cov_types() {
        // Issue #36: classical/HC0-3/HAC/clusterのいずれも、WlsEstimator::fitはcov_typeを
        // そのままOlsEstimator::fitに渡すだけで正しく動作するはず（wls-standard-errors.md
        // の通り、新しい計算式の実装は不要という前提の確認）。
        //
        // 検証方法: `from_columns_weighted`を経由せず、y・x1・切片列をこのテスト内で手動で
        // sqrt(weight)倍した上で`from_columns`（include_intercept=false）に渡し、素の
        // OlsEstimator::fitを直接呼ぶ。これは`WlsEstimator::fit`の内部実装と独立した経路で
        // 同じ変換を行っているため、両者が一致すればcov_typeの受け渡し・適合度統計量の
        // 計算が正しく配線されていることの確認になる。
        let y: Vec<f64> = vec![2.0, 4.0, 5.0, 4.0, 6.0, 3.0];
        let x1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights: Vec<f64> = vec![1.0, 4.0, 0.25, 9.0, 2.0, 0.5];
        let clusters = ["a", "a", "b", "b", "c", "c"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let sqrt_w: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
        let y_tilde: Vec<f64> = y.iter().zip(&sqrt_w).map(|(v, s)| v * s).collect();
        let const_tilde: Vec<f64> = sqrt_w.clone();
        let x1_tilde: Vec<f64> = x1.iter().zip(&sqrt_w).map(|(v, s)| v * s).collect();
        let manual_x_columns = vec![const_tilde, x1_tilde];

        let cov_types = [
            CovType::Classical,
            CovType::Hc0,
            CovType::Hc1,
            CovType::Hc2,
            CovType::Hc3,
            CovType::Hac {
                lags: Some(1),
                time_order: None,
            },
            CovType::Cluster {
                groups: Some(clusters.clone()),
            },
        ];

        for cov_type in cov_types {
            let label = format!("{cov_type:?}");

            let manual_input = OlsInput::from_columns(
                &y_tilde,
                &manual_x_columns,
                vec!["const".to_string(), "x1".to_string()],
                false,
                "y".to_string(),
            )
            .unwrap();
            let manual = OlsEstimator::fit(manual_input, cov_type.clone(), 0.95).unwrap();

            let wls = WlsEstimator::fit(
                &y,
                std::slice::from_ref(&x1),
                vec!["x1".to_string()],
                true,
                "y".to_string(),
                &weights,
                cov_type,
                0.95,
            )
            .unwrap();

            for j in 0..2 {
                assert!(
                    (*wls.estimator().params().get(j, 0) - *manual.params().get(j, 0)).abs() < 1e-9,
                    "{label}: params mismatch at index {j}"
                );
                assert!(
                    (*wls.estimator().std_errors().get(j, 0) - *manual.std_errors().get(j, 0))
                        .abs()
                        < 1e-9,
                    "{label}: std_errors mismatch at index {j}"
                );
            }
            // r_squared・f_statistic・f_p_valueはここでは比較しない: `manual`側は
            // 切片列を自前で用意しているため`include_intercept=false`で渡しており、
            // `has_intercept`フラグ（centered/uncentered TSSの切替、F検定でconstを
            // 除外するかの切替に使う）が`wls`側（`include_intercept=true`）と異なる。
            // これは「同じ回帰に対する2通りの正しい計算」ではなく、意味的に異なる統計量に
            // なるため単純比較できない。R²・F検定が正しいことは
            // `fit_with_all_weights_one_matches_ols`（両者ともinclude_intercept=trueで
            // 揃っている）で別途確認する。
            //
            // aicはlog_likelihood（ssr, nのみに依存）とk（パラメータ数、どちらも2）のみで
            // 決まるため、has_intercept差の影響を受けず比較できる。ここでは意図的に
            // `wls.estimator().aic()`（変換後データに対するOLSのaic、ヤコビアン補正なし）を
            // 使う。`manual`側も同じ「変換後データに対するOLS」のaicのため、両者は一致する
            // はず（配線の確認が目的）。`wls.aic()`（ヤコビアン補正込みの正しい値）は
            // `manual`側に対応する計算がないためここでは比較しない
            // （`fit_with_all_weights_one_matches_ols`で別途確認）。
            assert!(
                (wls.estimator().aic() - manual.aic()).abs() < 1e-9,
                "{label}: aic mismatch"
            );
        }
    }

    #[test]
    fn fit_without_intercept_uses_uncentered_r_squared_and_omits_const() {
        // include_intercept=falseのとき、WLSがOLS側のuncentered TSS分岐
        // （has_intercept()=falseのときのr_squared計算）と、original_scale_residualsの
        // 「切片項を足さない」分岐を正しく通ることを確認する。
        //
        // y = 2*x1に対する厳密解（ノイズなし）は使わない: k=1（切片なし・x1のみ）で
        // 残差がすべて0になると分散が数値的にゼロになり、F検定（wald_f_test）の
        // Cholesky分解がComputationFailedを返してしまう（本来は理論上到達不能な
        // 防御的分岐だが、ゼロ分散という退化ケースでは実際に到達しうる）。
        // このテストの目的はF検定の退化ケースの検証ではないため、小さなノイズを入れる。
        let y = vec![2.1, 3.9, 6.2, 7.8, 10.1]; // y ≈ 2*x1
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = vec![1.0, 2.0, 0.5, 3.0, 1.5];

        let wls = WlsEstimator::fit(
            &y,
            std::slice::from_ref(&x1),
            vec!["x1".to_string()],
            false,
            "y".to_string(),
            &weights,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        assert_eq!(wls.estimator().input().param_names(), ["x1".to_string()]);
        assert!(!wls.estimator().input().has_intercept());
        let beta_hat = *wls.estimator().params().get(0, 0);
        assert!((beta_hat - 2.0).abs() < 0.05);
        // 切片なし（uncentered TSS）の場合、Σ(sqrt(w)*y)² == Σw*y² が数学的に厳密に
        // 成り立つため、wls.r_squared()（元スケールから計算）とwls.estimator().r_squared()
        // （変換後データから計算）は一致する。ここでは公開APIとして使うべき方
        // （wls.r_squared()）を確認する。
        assert!(wls.r_squared() > 0.99);
        assert!((wls.r_squared() - wls.estimator().r_squared()).abs() < 1e-12);

        // 内部整合性: residuals()は「元スケールのy - 推定された係数による予測値」であるはず
        // （original_scale_residualsの定義そのものの確認。真の係数2.0とは比較しない）。
        for (i, &r) in wls.residuals().iter().enumerate() {
            let expected = y[i] - beta_hat * x1[i];
            assert!(
                (r - expected).abs() < 1e-9,
                "residual {i}: got {r}, expected {expected}"
            );
        }
    }

    #[test]
    fn fit_with_multiple_x_columns_recovers_known_coefficients() {
        // original_scale_residualsのx_columnsループ（複数列）が正しく計算できることを確認する
        // （これまでのテストはx列1本のみだった）。
        let y = vec![9.0, 8.0, 19.0, 18.0, 29.0, 28.0]; // 1 + 2*x1 + 3*x2
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x2 = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0];
        let weights = vec![1.0, 2.0, 0.5, 3.0, 1.5, 2.5];

        let wls = WlsEstimator::fit(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
            &weights,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        assert!((*wls.estimator().params().get(0, 0) - 1.0).abs() < 1e-9); // const
        assert!((*wls.estimator().params().get(1, 0) - 2.0).abs() < 1e-9); // x1
        assert!((*wls.estimator().params().get(2, 0) - 3.0).abs() < 1e-9); // x2
        for r in wls.residuals() {
            assert!(r.abs() < 1e-9);
        }
    }
}
