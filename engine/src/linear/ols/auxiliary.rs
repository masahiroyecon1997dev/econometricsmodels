//! 補助回帰（ハウスマン検定など、検定の部品として当てはめる回帰）向けの、設計行列を消費する当てはめ。
//!
//! `OlsEstimator::fit_auxiliary`は`OlsInput`を借りて分解する（`OrthonormalDesign::new_checked`が
//! 設計行列の複製を作る）ので、設計行列・QRの格納・`Q`の3つ（`3·n·k`）が同時に生きる。補助回帰は
//! 係数・標準誤差・適合度を公開せず、結果として使うのは検定統計量だけなので、設計行列を
//! 受け取って、その領域をそのままQRの格納に使う（[`OrthonormalDesign::new_checked_owned`]。
//! ピークは`2·n·k`）。`OlsEstimator`は作らないので、設計行列を持ち続けることもない。
//!
//! 検査・順序・数値は`fit_auxiliary`と同じ（`OlsEstimator::fit_auxiliary`との一致は`mod tests`）。

use faer::Mat;

use super::cov_params::{CovParams, compute_cov_params};
use super::cov_type::CovType;
use super::fit::preflight;
use super::input::OlsInput;
use crate::linear::common::LeastSquaresError;
use crate::shared::error::CommonError;
use crate::shared::goodness_of_fit::is_perfect_fit;
use crate::shared::least_squares::residual_sum_of_squares;
use crate::shared::orthonormal::OrthonormalDesign;
use crate::shared::wald::wald_f_test;

/// 設計行列を消費して当てはめた補助回帰（直交座標）。`Q`・`R`・`b̃ = Q'y`・残差を持つ。
#[derive(Debug)]
pub(crate) struct AuxiliaryFit {
    design: OrthonormalDesign,
    coordinates: Mat<f64>,
    residuals: Mat<f64>,
    has_intercept: bool,
}

impl AuxiliaryFit {
    /// `input`の設計行列を消費して、`OlsEstimator::fit`（`check_outcome`が真）・`fit_auxiliary`
    /// （偽）と同じ検査と順序（回帰変数の有無・信頼水準・観測数・クラスター数・被説明変数・ランク落ち・
    /// 完全フィット）のうえで直交座標に分解する。`cov_type`はクラスター数の検証にだけ使う
    /// （共分散そのものは[`Self::wald_last_columns`]で求める）。
    ///
    /// `check_outcome`が真のとき、被説明変数が定数（切片あり）なら`OutcomeConstant`、完全フィットなら
    /// `PerfectFit`で失敗させる（`OlsEstimator::fit`と同じ。元のモデルの被説明変数を持つ回帰に使う）。
    ///
    /// # Errors
    /// `OlsEstimator::fit`/`fit_auxiliary`が共分散より前に返すエラーと同じ。
    pub(crate) fn fit(
        input: OlsInput,
        cov_type: &CovType,
        confidence_level: f64,
        check_outcome: bool,
    ) -> Result<Self, LeastSquaresError> {
        preflight(&input, cov_type, confidence_level, false, check_outcome)?;
        let k = input.k();
        let outcome_sst = input.outcome_sst();

        let (x, y, names, has_intercept) = input.into_design();
        let design = OrthonormalDesign::new_checked_owned(x)
            .map_err(|err| LeastSquaresError::from_rank_deficient(err.kind, &err.x, &names))?;
        let coordinates = design.coordinates(&y);
        let residuals = design.residuals(&y, &coordinates);
        if check_outcome && is_perfect_fit(residual_sum_of_squares(&residuals), outcome_sst, k) {
            return Err(CommonError::PerfectFit.into());
        }
        Ok(Self {
            design,
            coordinates,
            residuals,
            has_intercept,
        })
    }

    /// 直交化した設計行列`Q`と`R`。
    pub(crate) fn design(&self) -> &OrthonormalDesign {
        &self.design
    }

    /// 係数の直交座標 `b̃ = Q'y`。
    pub(crate) fn coordinates(&self) -> &Mat<f64> {
        &self.coordinates
    }

    /// 残差（n×1）。
    pub(crate) fn residuals(&self) -> &Mat<f64> {
        &self.residuals
    }

    /// 観測数。
    pub(crate) fn nobs(&self) -> usize {
        self.residuals.nrows()
    }

    /// 説明変数の数（定数項を含む）。
    pub(crate) fn k(&self) -> usize {
        self.coordinates.nrows()
    }

    /// `cov_type`の共分散のミート`M̃`（直交座標）と推論の自由度。
    fn covariance(&self, cov_type: &CovType) -> Result<CovParams, LeastSquaresError> {
        let (n, k) = (self.nobs(), self.k());
        let df_resid = n - k;
        let sigma2 = residual_sum_of_squares(&self.residuals) / df_resid as f64;
        compute_cov_params(
            cov_type,
            self.design.q(),
            &self.residuals,
            &self.design.identity(),
            sigma2,
            df_resid,
        )
    }

    /// 全体Wald/F検定（切片を除く全ての傾きが同時にゼロ）が計算できることを確かめる。
    /// `OlsEstimator::fit_auxiliary`は`fit`の一部として全体F検定を計算し、共分散の部分行列が
    /// 数値的にほぼ特異なら失敗するので、その失敗の有無を同じ条件で再現する。
    ///
    /// # Errors
    /// `fit_auxiliary`が全体F検定で返すエラーと同じ。
    pub(crate) fn require_overall_f(&self, cov_type: &CovType) -> Result<(), LeastSquaresError> {
        let CovParams {
            cov_params: meat,
            df_inference,
            ..
        } = self.covariance(cov_type)?;
        self.overall_f(&meat, df_inference)
    }

    fn overall_f(&self, meat: &Mat<f64>, df_inference: usize) -> Result<(), LeastSquaresError> {
        let k_constant = usize::from(self.has_intercept);
        let df_model = self.k() - k_constant;
        if df_model > 0 {
            wald_f_test(&self.coordinates, meat, k_constant, df_model, df_inference)?;
        }
        Ok(())
    }

    /// 設計行列の末尾`q`列の係数が同時にゼロという帰無仮説のWald F検定の`(F統計量, p値, 推論の自由度)`
    /// （`OlsEstimator::fit(..)`の後の`wald_test_last_columns(q)`と`df_inference()`と同じ）。
    ///
    /// # Errors
    /// `fit`と`wald_test_last_columns`が返すエラーと同じ（共分散の検査、全体F検定、
    /// 末尾`q`列の部分行列が数値的にほぼ特異な場合）。
    pub(crate) fn wald_last_columns(
        &self,
        cov_type: &CovType,
        q: usize,
    ) -> Result<(f64, f64, usize), LeastSquaresError> {
        debug_assert!(
            q > 0 && q <= self.k(),
            "wald_last_columns: q must be in (0, k], got q={q}, k={}",
            self.k()
        );
        let CovParams {
            cov_params: meat,
            df_inference,
            ..
        } = self.covariance(cov_type)?;
        self.overall_f(&meat, df_inference)?;
        let (statistic, p_value) =
            wald_f_test(&self.coordinates, &meat, self.k() - q, q, df_inference)?;
        Ok((statistic, p_value, df_inference))
    }

    /// [`Self::wald_last_columns`]のF統計量だけ。
    pub(crate) fn wald_f_last_columns(
        &self,
        cov_type: &CovType,
        q: usize,
    ) -> Result<f64, LeastSquaresError> {
        Ok(self.wald_last_columns(cov_type, q)?.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::ols::OlsEstimator;
    use crate::shared::group_codes::GroupCodes;

    /// 決定的な非自明データ（n=60、切片 + 4列、不均一な誤差）。
    fn columns(n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        let x: Vec<Vec<f64>> = (0..4)
            .map(|j| {
                (0..n)
                    .map(|i| ((i * (3 + 2 * j) + 5 * j) % 17) as f64 - 8.0 + 0.1 * j as f64)
                    .collect()
            })
            .collect();
        let y = (0..n)
            .map(|i| {
                let noise = (((i * 11) % 7) as f64 - 3.0) * (1.0 + (i % 4) as f64 * 0.5);
                1.0 + x[0][i] - 0.5 * x[1][i] + 0.3 * x[2][i] + 0.2 * x[3][i] + noise
            })
            .collect();
        (y, x)
    }

    fn input(y: &[f64], x: &[Vec<f64>]) -> OlsInput {
        OlsInput::from_columns(
            y,
            x,
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            true,
            "y".into(),
        )
        .unwrap()
    }

    #[test]
    fn wald_f_last_columns_matches_the_estimator_for_every_cov_type() {
        let (y, x) = columns(60);
        for cov_type in [
            CovType::Classical,
            CovType::Hc0,
            CovType::Hc1,
            CovType::Hc2,
            CovType::Hc3,
        ] {
            for q in 1..=3 {
                let estimator =
                    OlsEstimator::fit_auxiliary(input(&y, &x), cov_type.clone(), 0.95).unwrap();
                let expected = estimator.wald_test_last_columns(q).unwrap().0;
                let aux =
                    AuxiliaryFit::fit(input(&y, &x), &CovType::Classical, 0.95, false).unwrap();
                let got = aux.wald_f_last_columns(&cov_type, q).unwrap();
                assert_eq!(got, expected, "{cov_type:?} q={q}");
            }
        }
    }

    fn all_cov_types(n: usize) -> Vec<CovType> {
        let groups: Vec<String> = (0..n).map(|i| format!("g{}", i % 6)).collect();
        vec![
            CovType::Classical,
            CovType::Hc0,
            CovType::Hc1,
            CovType::Hc2,
            CovType::Hc3,
            CovType::Hac {
                lags: Some(2),
                time_order: (0..n).map(|i| i as f64).collect(),
            },
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
        ]
    }

    #[test]
    fn wald_last_columns_matches_fit_with_the_outcome_checks_for_every_cov_type() {
        // Wu-Hausmanの拡張回帰と同じ使い方: `OlsEstimator::fit`（被説明変数の検査あり）の後の
        // `wald_test_last_columns(q)`と`df_inference()`。
        let (y, x) = columns(60);
        for cov_type in all_cov_types(60) {
            for q in 1..=2 {
                let estimator = OlsEstimator::fit(input(&y, &x), cov_type.clone(), 0.95).unwrap();
                let (expected_f, expected_p) = estimator.wald_test_last_columns(q).unwrap();
                let aux = AuxiliaryFit::fit(input(&y, &x), &cov_type, 0.95, true).unwrap();
                let (f, p, df) = aux.wald_last_columns(&cov_type, q).unwrap();
                assert_eq!(
                    (f, p, df),
                    (expected_f, expected_p, estimator.df_inference())
                );
            }
        }
    }

    #[test]
    fn fit_with_the_outcome_checks_returns_the_same_errors_as_the_estimator() {
        let (y, x) = columns(60);
        let exact: Vec<f64> = (0..60)
            .map(|i| 1.0 + x[0][i] - 0.5 * x[1][i] + 0.3 * x[2][i] + 0.2 * x[3][i])
            .collect();
        let constant = vec![3.0; 60];
        let few_groups: Vec<String> = (0..60).map(|i| format!("g{}", i % 4)).collect();
        let cluster = |groups: &[String]| CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(groups)),
        };
        let cases: Vec<(&str, Vec<f64>, CovType)> = vec![
            ("perfect fit", exact, CovType::Classical),
            ("constant y", constant, CovType::Classical),
            // クラスター数（4）が傾きの数（4）以下。
            ("too few clusters", y.clone(), cluster(&few_groups)),
            // クラスター列なし。
            (
                "missing groups",
                y.clone(),
                CovType::Cluster { groups: None },
            ),
        ];
        for (label, y, cov_type) in cases {
            let expected = OlsEstimator::fit(input(&y, &x), cov_type.clone(), 0.95)
                .map(|_| ())
                .unwrap_err();
            // 共分散の検査（クラスター列なしなど）は`wald_last_columns`の段階で返る。
            let got = AuxiliaryFit::fit(input(&y, &x), &cov_type, 0.95, true)
                .and_then(|aux| aux.wald_last_columns(&cov_type, 1))
                .map(|_| ())
                .unwrap_err();
            assert_eq!(got, expected, "{label}");
        }
        // `check_outcome=false`（`fit_auxiliary`と同じ）では、完全フィットも定数の`y`も判定しない。
        let constant = vec![3.0; 60];
        assert!(AuxiliaryFit::fit(input(&constant, &x), &CovType::Classical, 0.95, false).is_ok());
    }

    #[test]
    fn residuals_and_coordinates_match_the_estimator_and_the_borrowed_decomposition() {
        let (y, x) = columns(60);
        let estimator =
            OlsEstimator::fit_auxiliary(input(&y, &x), CovType::Classical, 0.95).unwrap();
        let aux = AuxiliaryFit::fit(input(&y, &x), &CovType::Classical, 0.95, false).unwrap();
        assert_eq!(aux.residuals(), estimator.residuals());
        assert_eq!((aux.nobs(), aux.k()), (60, 5));
        let borrowed = OrthonormalDesign::new_checked(estimator.input().x()).unwrap();
        assert_eq!(aux.design().q(), borrowed.q());
        assert_eq!(
            aux.coordinates(),
            &borrowed.coordinates(estimator.input().y())
        );
    }

    #[test]
    fn require_overall_f_succeeds_where_the_estimator_succeeds() {
        let (y, x) = columns(60);
        let aux = AuxiliaryFit::fit(input(&y, &x), &CovType::Classical, 0.95, false).unwrap();
        aux.require_overall_f(&CovType::Classical).unwrap();
        aux.require_overall_f(&CovType::Hc1).unwrap();
    }

    #[test]
    fn fit_returns_the_same_errors_as_fit_auxiliary_before_the_covariance() {
        let (y, x) = columns(60);

        // 信頼水準の範囲外。
        for level in [0.0, 1.0, 1.5, f64::NAN] {
            let expected =
                OlsEstimator::fit_auxiliary(input(&y, &x), CovType::Classical, level).unwrap_err();
            let got =
                AuxiliaryFit::fit(input(&y, &x), &CovType::Classical, level, false).unwrap_err();
            // `NaN != NaN`なので、`Debug`の表示で比べる。
            assert_eq!(format!("{got:?}"), format!("{expected:?}"), "level={level}");
        }

        // 観測数が足りない（n <= k）。
        let (short_y, short_x) = columns(5);
        let expected =
            OlsEstimator::fit_auxiliary(input(&short_y, &short_x), CovType::Classical, 0.95)
                .unwrap_err();
        let got = AuxiliaryFit::fit(input(&short_y, &short_x), &CovType::Classical, 0.95, false)
            .unwrap_err();
        assert_eq!(got, expected);

        // 共線な列（列名つきで報告する）。
        let mut collinear = x.clone();
        collinear[3] = collinear[1].iter().map(|v| 2.0 * v).collect();
        let expected = OlsEstimator::fit_auxiliary(input(&y, &collinear), CovType::Classical, 0.95)
            .unwrap_err();
        let got =
            AuxiliaryFit::fit(input(&y, &collinear), &CovType::Classical, 0.95, false).unwrap_err();
        assert!(matches!(got, LeastSquaresError::SingularMatrix { .. }));
        assert_eq!(got, expected);
    }
}
