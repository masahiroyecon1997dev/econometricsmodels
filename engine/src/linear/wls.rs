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
}
