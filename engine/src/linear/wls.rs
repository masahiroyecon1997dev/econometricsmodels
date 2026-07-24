//! WLS（Weighted Least Squares）の推定。
//!
//! `docs/planning/specs/wls-api-design.md`で確定した設計に基づき、`sqrt(weight)`で
//! 変換した`OlsInput`（`OlsInput::from_columns_weighted`）に対して既存の`OlsEstimator::fit`
//! （無変更）をそのまま適用する。標準誤差・適合度統計量の計算式が変換後データに対する
//! OLSの計算式そのままで正しいことの確認は`docs/planning/specs/wls-standard-errors.md`参照。

use super::ols::{CovType, OlsError, OlsEstimator, OlsInput};

/// WLSの推定結果。
///
/// `estimator()`が返す`OlsEstimator`本体は、`sqrt(weight)`で変換したデータに対する計算結果
/// （`params`/`std_errors`/`t_stats`/`p_values`/`conf_lower`/`conf_upper`/適合度統計量は
/// すべてここから取得する。重みが全て1のときOLSと数値的に完全一致するのはこの型を経由する
/// ためであり、`docs/planning/specs/wls-api-design.md`4.1節の構造的保証がそのまま成り立つ）。
///
/// ただし`estimator().residuals()`は**重み付き残差** `sqrt(w_i)(y_i - x_i'β̂)`
/// （statsmodelsでいう`.wresid`相当）を返す。ユーザー向けに公開する残差は元スケール
/// （unweighted）の`residuals()`を使う（`docs/planning/specs/wls-api-design.md`4.3節参照）。
#[derive(Debug)]
pub struct WlsEstimator {
    estimator: OlsEstimator,
    /// 元スケール（unweighted）の残差 `y_i - x_i'β̂`
    residuals: Vec<f64>,
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
    ) -> Result<Self, OlsError> {
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

        Ok(Self {
            estimator,
            residuals,
        })
    }

    /// 変換後データ（重み付き）に対する`OlsEstimator`本体。
    pub fn estimator(&self) -> &OlsEstimator {
        &self.estimator
    }

    /// 元スケール（unweighted）の残差 `y_i - x_i'β̂`。
    pub fn residuals(&self) -> &[f64] {
        &self.residuals
    }
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
        assert_eq!(wls.estimator().r_squared(), ols.r_squared());
        assert_eq!(wls.estimator().r_squared_adj(), ols.r_squared_adj());
        assert_eq!(wls.estimator().f_statistic(), ols.f_statistic());
        assert_eq!(wls.estimator().f_p_value(), ols.f_p_value());

        // 残差の計算経路自体はwls.rs（手動ループ）とols.rs（faerの行列演算）で異なるため、
        // 丸め誤差レベルでの一致を確認する（構造的な完全一致の保証対象はestimator()側）。
        for i in 0..5 {
            assert!((wls.residuals()[i] - *ols.residuals().get(i, 0)).abs() < 1e-12);
        }
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
            OlsError::NonPositiveWeight {
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
            // 決まるため、has_intercept差の影響を受けず比較できる。
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
        assert!(wls.estimator().r_squared() > 0.99);

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
