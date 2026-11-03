//! `LogitFitted`（予測・的中表・限界効果に必要な最小の状態）と、その復元用の`from_parts`。

use super::input::LogitInput;
use super::model::{at_point_w_and_s, logistic, overall_w_and_s};
use crate::nonlinear::common::{
    FittedModelForMarginalEffects, MarginalEffects, MarginalEffectsAt, MleError, column_means,
    column_medians, marginal_effects_from_w_s, pred_table, predict_from_link, predict_new_data,
};
use crate::shared::error::CommonError;
use faer::Mat;

/// Logitの推定結果のうち、予測・的中表・限界効果に必要な最小の状態（入力データ・係数・係数の分散共分散行列）。
///
/// [`super::LogitEstimator::into_fitted`]で取り出す。`engine_pybind`の結果オブジェクトは推定量全体ではなくこれを
/// 保持し、pickleでは[`LogitFitted::from_parts`]で復元する。フィールドはprivate（`.claude/rules/rust-style.md`
/// 「推定量構造体の設計」参照）で、構築経路は`fit`と`from_parts`だけ。
#[derive(Debug)]
pub struct LogitFitted {
    pub(super) input: LogitInput,
    /// 係数（元のスケール）。`input.param_names()`と対応する
    pub(super) params: Vec<f64>,
    /// 係数の分散共分散行列（元のスケール、k×k）。限界効果（デルタ法）で使う
    pub(super) cov_params: Mat<f64>,
}

impl LogitFitted {
    /// 入力データ・係数・係数の分散共分散行列から組み立てる。
    ///
    /// 最適性は検証できない（`fit`が収束した結果であることは呼び出し側が保証する）。ここで検査するのは、
    /// 後続のメソッドが添字の範囲外にならないための次元の整合だけ。
    ///
    /// 壊れた・改ざんされたpickleだけが到達する入力エラーで、`engine_pybind`が`PICKLE_FORMAT_MISMATCH`へ変換する。
    /// `params`・`cov_params`の値が有限かは検査しない（非有限値は後続の計算にNaNとして伝わるだけ）。
    ///
    /// # Errors
    /// `params`の長さまたは`cov_params`の形が`k`と合わない: `CommonError::Internal`
    pub fn from_parts(
        input: LogitInput,
        params: Vec<f64>,
        cov_params: Mat<f64>,
    ) -> Result<Self, MleError> {
        let k = input.k();
        if params.len() != k || cov_params.nrows() != k || cov_params.ncols() != k {
            return Err(CommonError::internal("inconsistent fitted model parts").into());
        }
        Ok(Self {
            input,
            params,
            cov_params,
        })
    }

    pub fn input(&self) -> &LogitInput {
        &self.input
    }

    pub fn params(&self) -> &[f64] {
        &self.params
    }

    pub fn cov_params(&self) -> &Mat<f64> {
        &self.cov_params
    }

    /// 限界効果（`marginal_effects`）。`fit()`とは独立した別メソッド（`fit()`のReturn
    /// 本体には含めない、`docs/spec/nonlinear-common.md`6章で確定済み）。`fit()`時の
    /// `cov_params`を再利用するため再最適化は不要（`confidence_level`は`fit()`とは
    /// 独立したパラメータとして受け取り、`fit()`時の値に縛られず事後的に異なる
    /// CI幅を見られるようにする）。
    ///
    /// ## 数式（デルタ法）
    ///
    /// `p_i = Λ(x_i'θ)`のとき、変数`j`（連続変数として扱う。`dummy=False`が既定の
    /// statsmodelsの`get_margeff()`に倣い、離散変数の自動判定は行わない設計、
    /// `docs/spec/nonlinear-common.md`6章参照）の限界効果は
    /// `dy/dx_j = p(1-p)θ_j`。
    ///
    /// - `at="overall"`（AME）: `g_j(θ) = w(θ)*θ_j`、`w(θ) = (1/n)Σᵢ pᵢ(1-pᵢ)`
    /// - `at="mean"`/`"median"`: `g_j(θ) = w(θ)*θ_j`、`w(θ) = p̄(1-p̄)`
    ///   （`p̄=Λ(x̄'θ)`、`x̄`は各説明変数の標本平均または中央値からなる代表点）
    ///
    /// いずれも同じ`g_j(θ)=w(θ)*θ_j`という形に帰着するため、`w`とその勾配
    /// `s_m=∂w/∂θ_m`さえ計算できれば、ヤコビアンは
    /// `∂g_j/∂θ_m = θ_j*s_m + [j==m]*w`という共通の式で書ける
    /// （`overall_w_and_s`/`at_point_w_and_s`が`(w,s)`を計算し、`w`・`s`の計算方法に
    /// 依らない残りの計算——ヤコビアン・デルタ法標準誤差・定数項の除外——は
    /// `nonlinear/common/marginal_effects.rs`の`marginal_effects_from_w_s`に共通化されている
    /// （Probitでも同型の式になることを確認済み）。
    ///
    /// 変数`j`の分散は`Var(g_j) = jac_j · Σ · jac_jᵀ`（`jac_j`はヤコビアンの`j`行目、
    /// `Σ=cov_params`）。標準誤差はこの平方根、検定分布は標準正規分布
    /// （`fit()`本体と同じ、`docs/spec/nonlinear-common.md`4章）。
    ///
    /// 定数項（切片）は出力から除外する（切片の限界効果は意味を持たない、
    /// statsmodelsも同様）。
    ///
    /// # Errors
    /// `confidence_level`が`(0, 1)`の範囲外: `CommonError::InvalidConfidenceLevel`
    pub fn marginal_effects(
        &self,
        at: MarginalEffectsAt,
        confidence_level: f64,
    ) -> Result<MarginalEffects, MleError> {
        let x = self.input.x();
        let k = self.input.k();
        let (w, s) = match at {
            MarginalEffectsAt::Overall => overall_w_and_s(x, &self.params),
            MarginalEffectsAt::Mean => at_point_w_and_s(&column_means(x), &self.params),
            MarginalEffectsAt::Median => at_point_w_and_s(&column_medians(x), &self.params),
        };
        marginal_effects_from_w_s(
            FittedModelForMarginalEffects {
                param_names: self.input.param_names(),
                has_intercept: self.input.has_intercept(),
                k,
                params: &self.params,
                cov_params: &self.cov_params,
            },
            w,
            &s,
            confidence_level,
        )
    }

    /// 予測確率 `p_i = Λ(x_i'θ)` を、`fit()`に使った学習データ（`self.input.x()`）の
    /// 各行について返す（`fit()`のReturn本体には含めない別メソッド、
    /// `docs/spec/nonlinear-common.md`6章）。
    ///
    /// 新規データでの予測（out-of-sample）は`predict_new_data`。
    pub fn predict(&self) -> Vec<f64> {
        predict_from_link(self.input.x(), &self.params, logistic)
    }

    /// 新規データ（out-of-sample、`new_x_columns`）に対する予測確率
    /// `p_i = Λ(x_i'θ)`。
    ///
    /// `nonlinear::common::predict_new_data`に`logistic`をリンク関数として渡すだけの
    /// 薄いラッパー。リンク関数の選択（Logitはロジスティック関数）という手法固有の
    /// 知識を`engine`側に閉じ込め、`engine_pybind`には計算ロジックを持たせない方針
    /// （`.claude/rules/rust-style.md`「責務分離」）を守るため、`engine_pybind`から
    /// `common::predict_new_data`を直接呼ばずこのメソッド経由にしている
    /// （`OLSResult::predict`が`engine::linear::ols::predict_new_data`という
    /// フリー関数を直接呼べるのは、OLSにはリンク関数の選択という手法固有の分岐が
    /// 無いため）。
    ///
    /// `new_x_columns`の本数・順序の契約、パニック条件は
    /// `nonlinear::common::predict_new_data`のdocコメント参照。
    pub fn predict_new_data(&self, new_x_columns: &[Vec<f64>]) -> Vec<f64> {
        predict_new_data(
            &self.params,
            self.input.has_intercept(),
            new_x_columns,
            logistic,
        )
    }

    /// 分類の的中表（2×2、`table[actual][predicted]`のカウント。行=実測クラス、
    /// 列=予測クラス）。`predict()`が返す予測確率のみを`threshold`で二値化し、実測`y`は
    /// `threshold`に関わらず常に`0.5`で二値化する。数式・statsmodelsとの整合性の詳細は
    /// `nonlinear/common/predict.rs`の`pred_table`のdocコメント参照（リンク関数に依存しない計算
    /// のため`common/predict.rs`に共通化されている）。
    ///
    /// **新規データでの的中表（out-of-sample）は未対応**（`predict()`とは異なり
    /// スコープ外）。
    pub fn pred_table(&self, threshold: f64) -> Mat<f64> {
        pred_table(&self.predict(), self.input.y(), threshold)
    }
}

#[cfg(test)]
mod tests {

    use super::super::model::logistic;

    use super::super::test_support::*;

    use crate::nonlinear::common::CovType;

    use crate::nonlinear::common::MarginalEffectsAt;
    use crate::nonlinear::common::MleError;
    use crate::nonlinear::common::MleFitOptions;

    use crate::nonlinear::common::SolverType;

    use crate::nonlinear::logit::LogitEstimator;
    use crate::nonlinear::logit::LogitFitted;
    use crate::nonlinear::logit::LogitInput;

    use crate::shared::error::CommonError;

    use faer::Mat;

    use statrs::distribution::ContinuousCDF;
    use statrs::distribution::Normal;

    /// 切片のみモデル（k=1、`k_constant=1`）は、限界効果の出力対象となる説明変数が
    /// 存在しない（定数項は出力から除外するため）。`marginal_effects`が空の結果を
    /// 返す（パニックしない）ことを確認する境界ケース。
    #[test]
    fn marginal_effects_returns_empty_result_for_intercept_only_model() {
        let estimator = LogitEstimator::fit(
            intercept_only_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-6,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        let effects = estimator
            .marginal_effects(MarginalEffectsAt::Overall, 0.95)
            .unwrap();
        assert!(effects.param_names().is_empty());
        assert!(effects.effect().is_empty());
    }

    /// `marginal_effects(at="overall")`の`effect`を、実装の内部ヘルパー（`overall_w_and_s`/
    /// `dydx_and_jacobian`）とは別に、定義式`dy/dx_j = (1/n)Σᵢpᵢ(1-pᵢ)θⱼ`を`logistic`
    /// から直接計算する式で独立に再計算し、突き合わせる。標準誤差は、デルタ法の
    /// ヤコビアンを`overall_w_and_s`経由ではなく数値微分（`dydx_j`自体をfit済みパラメータ
    /// の周りで直接数値微分したもの）で独立に求め、`marginal_effects`が返す解析的な
    /// 標準誤差と突き合わせることで、`dydx_and_jacobian`内の式（`θⱼ*s_m + [j==m]*w`）が
    /// 正しく実装に配線されていることを検証する（配線ミスがあればこのテストで検出できる）。
    #[test]
    fn marginal_effects_overall_matches_independently_recomputed_dydx_and_delta_method_se() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = LogitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let k = 3;
        let n = 4;
        let x = estimator.input().x();

        let effects = estimator
            .marginal_effects(MarginalEffectsAt::Overall, 0.95)
            .unwrap();
        assert_eq!(effects.param_names(), ["x1".to_string(), "x2".to_string()]);

        // effectの独立再計算（`logistic`から直接、`overall_w_and_s`とは別の式）
        let dydx_j = |params: &[f64], j: usize| -> f64 {
            (0..n)
                .map(|i| {
                    let z: f64 = (0..k).map(|m| *x.get(i, m) * params[m]).sum();
                    let p = logistic(z);
                    p * (1.0 - p)
                })
                .sum::<f64>()
                / (n as f64)
                * params[j]
        };
        let params = estimator.params();
        for (idx, j) in (1..k).enumerate() {
            assert!((effects.effect()[idx] - dydx_j(params, j)).abs() < 1e-9);
        }

        // 標準誤差の独立検証: `dydx_j`をfit済みパラメータの周りで数値微分して
        // ヤコビアン行（j行目）を求め、`cov_params`との二次形式で分散を計算する。
        let h = 1e-6;
        for (idx, j) in (1..k).enumerate() {
            let mut jac_row = vec![0.0; k];
            for m in 0..k {
                let mut plus = params.to_vec();
                plus[m] += h;
                let mut minus = params.to_vec();
                minus[m] -= h;
                jac_row[m] = (dydx_j(&plus, j) - dydx_j(&minus, j)) / (2.0 * h);
            }
            let mut var_j = 0.0;
            for a in 0..k {
                for b in 0..k {
                    var_j += jac_row[a] * (*estimator.cov_params().get(a, b)) * jac_row[b];
                }
            }
            let expected_se = var_j.sqrt();
            assert!(
                (effects.std_errors()[idx] - expected_se).abs() < 1e-6,
                "idx={idx}, actual={}, expected={expected_se}",
                effects.std_errors()[idx]
            );
        }

        // z値・p値・信頼区間の内部整合性（既存のSE系テストと同じ検算パターン）
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z_crit = normal.inverse_cdf(0.975);
        for idx in 0..2 {
            let se = effects.std_errors()[idx];
            assert!((effects.test_stats()[idx] - effects.effect()[idx] / se).abs() < 1e-9);
            let expected_p = 2.0 * (1.0 - normal.cdf(effects.test_stats()[idx].abs()));
            assert!((effects.p_values()[idx] - expected_p).abs() < 1e-9);
            assert!(
                (effects.conf_upper()[idx] - effects.conf_lower()[idx] - 2.0 * z_crit * se).abs()
                    < 1e-9
            );
        }
    }

    /// `at="mean"`は`at="overall"`と異なる代表点（標本平均）で評価するため、一般には
    /// 異なる値になる。実装がこの違いを正しく反映していること（`at`の分岐が機能して
    /// いること）を確認する。`effect`を`column_means`から独立に再計算した値とも突き合わせる。
    #[test]
    fn marginal_effects_at_mean_differs_from_overall_and_matches_independent_recomputation() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = LogitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        let overall = estimator
            .marginal_effects(MarginalEffectsAt::Overall, 0.95)
            .unwrap();
        let at_mean = estimator
            .marginal_effects(MarginalEffectsAt::Mean, 0.95)
            .unwrap();

        assert!((overall.effect()[0] - at_mean.effect()[0]).abs() > 1e-9);

        // 独立再計算: x̄=[1, 25, 1]（定数項1、x1の平均25、x2の平均1）でp̄を評価
        let params = estimator.params();
        let x_bar = [1.0, 25.0, 1.0];
        let z_bar: f64 = (0..3).map(|m| x_bar[m] * params[m]).sum();
        let p_bar = logistic(z_bar);
        let w = p_bar * (1.0 - p_bar);
        for (idx, j) in (1..3).enumerate() {
            assert!((at_mean.effect()[idx] - w * params[j]).abs() < 1e-9);
        }
    }

    /// `at="median"`が`at="mean"`/`at="overall"`と異なる代表点で評価されること
    /// （非対称なデータセットで平均・中央値が異なる値になるよう構成）、および
    /// `effect`を`column_medians`から独立に再計算した値と突き合わせる。
    #[test]
    fn marginal_effects_at_median_differs_from_mean_and_overall_and_matches_independent_recomputation()
     {
        // x1: 平均40（=200/5）・中央値30（sorted: 10,20,30,40,100）と非対称にずらす
        let y = vec![0.0, 1.0, 0.0, 1.0, 1.0];
        let x_columns = vec![
            vec![10.0, 20.0, 30.0, 40.0, 100.0],
            vec![-5.0, 2.0, 8.0, -1.0, 50.0],
        ];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = LogitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        let overall = estimator
            .marginal_effects(MarginalEffectsAt::Overall, 0.95)
            .unwrap();
        let at_mean = estimator
            .marginal_effects(MarginalEffectsAt::Mean, 0.95)
            .unwrap();
        let at_median = estimator
            .marginal_effects(MarginalEffectsAt::Median, 0.95)
            .unwrap();

        assert!((at_median.effect()[0] - at_mean.effect()[0]).abs() > 1e-9);
        assert!((at_median.effect()[0] - overall.effect()[0]).abs() > 1e-9);

        // 独立再計算: x̄=[1, 30, 2]（定数項1、x1の中央値30、x2の中央値2）でp̄を評価
        let params = estimator.params();
        let x_bar = [1.0, 30.0, 2.0];
        let z_bar: f64 = (0..3).map(|m| x_bar[m] * params[m]).sum();
        let p_bar = logistic(z_bar);
        let w = p_bar * (1.0 - p_bar);
        for (idx, j) in (1..3).enumerate() {
            assert!((at_median.effect()[idx] - w * params[j]).abs() < 1e-9);
        }
    }

    #[test]
    fn marginal_effects_returns_invalid_confidence_level_error_out_of_range() {
        let estimator = LogitEstimator::fit(
            intercept_only_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-6,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        let result = estimator.marginal_effects(MarginalEffectsAt::Overall, 1.5);
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5
            })
        );
    }

    /// 切片のみモデルは全観測で`p_i=ȳ`（closed form、他のテストと同じ性質）なので、
    /// `predict()`が返す予測確率が`ȳ`と一致することを検証できる。
    #[test]
    fn predict_matches_closed_form_for_intercept_only_model() {
        let estimator = LogitEstimator::fit(
            intercept_only_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-6,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        let y_bar: f64 = 4.0 / 7.0;
        let predicted = estimator.predict();
        assert_eq!(predicted.len(), 7);
        for p in predicted {
            assert!((p - y_bar).abs() < 1e-6);
        }
    }

    /// 多変量モデルでは`predict()`に閉じた形の解析解が無いため、`logistic`から直接
    /// `p_i=Λ(x_i'θ)`を計算する式で独立に再計算し、突き合わせる。
    #[test]
    fn predict_matches_independently_recomputed_logistic_of_linear_predictor() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = LogitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        let params = estimator.params();
        let x = estimator.input().x();
        let n = 4;
        let k = 3;
        let predicted = estimator.predict();
        for (i, &p_i) in predicted.iter().enumerate().take(n) {
            let z: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            let expected = 1.0 / (1.0 + (-z).exp());
            assert!((p_i - expected).abs() < 1e-12);
        }
    }

    /// `predict_new_data`（out-of-sample）が独立に再計算した
    /// `p_i=Λ(x_i'θ)`と一致すること。`predict_matches_independently_recomputed_
    /// logistic_of_linear_predictor`と同じモデルを使い、学習データとは異なる新規の
    /// x値で検証する。
    #[test]
    fn predict_new_data_matches_independently_recomputed_logistic_of_linear_predictor() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = LogitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        let params = estimator.params().to_vec();
        // 学習データには無い新規のx値
        let new_x_columns = vec![vec![15.0, 25.0], vec![0.0, -3.0]];
        let predicted = estimator.predict_new_data(&new_x_columns);

        assert_eq!(predicted.len(), 2);
        for (i, &p_i) in predicted.iter().enumerate() {
            let z = params[0] + new_x_columns[0][i] * params[1] + new_x_columns[1][i] * params[2];
            let expected = 1.0 / (1.0 + (-z).exp());
            assert!((p_i - expected).abs() < 1e-12);
        }
    }

    /// `predict_new_data`が`has_intercept=false`（`LogitInput::from_columns`の
    /// `include_intercept=false`）でも正しく動作すること
    /// （`has_intercept=true`を検証する`predict_new_data_matches_
    /// independently_recomputed_logistic_of_linear_predictor`と対になる）。
    #[test]
    fn predict_new_data_without_intercept_matches_independently_recomputed_logistic_of_linear_predictor()
     {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            false,
            "y".to_string(),
        )
        .unwrap();

        let estimator = LogitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        let params = estimator.params().to_vec();
        // 学習データには無い新規のx値
        let new_x_columns = vec![vec![15.0, 25.0], vec![0.0, -3.0]];
        let predicted = estimator.predict_new_data(&new_x_columns);

        assert_eq!(predicted.len(), 2);
        for (i, &p_i) in predicted.iter().enumerate() {
            let z = new_x_columns[0][i] * params[0] + new_x_columns[1][i] * params[1];
            let expected = 1.0 / (1.0 + (-z).exp());
            assert!((p_i - expected).abs() < 1e-12);
        }
    }

    /// 切片のみモデルは全観測で`p_i=ȳ=4/7≈0.571`（closed form）のため、`threshold`に
    /// よって全観測が一方のクラスに分類される自明なケースになる。この性質を使い、
    /// `pred_table`の的中表を手計算で検証する（`y=[0,0,0,1,1,1,1]`、実測は
    /// `y_i>threshold`で二値化）。
    #[test]
    fn pred_table_matches_hand_computed_counts_for_intercept_only_model() {
        let estimator = LogitEstimator::fit(
            intercept_only_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-6,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        // threshold=0.5: p_i=4/7>0.5 なので全観測が予測クラス1。
        // 実測は y=[0,0,0,1,1,1,1] なので actual0が3件・actual1が4件。
        let table_low = estimator.pred_table(0.5);
        assert!((*table_low.get(0, 0) - 0.0).abs() < 1e-12);
        assert!((*table_low.get(0, 1) - 3.0).abs() < 1e-12);
        assert!((*table_low.get(1, 0) - 0.0).abs() < 1e-12);
        assert!((*table_low.get(1, 1) - 4.0).abs() < 1e-12);

        // threshold=0.99: p_i=4/7<0.99 なので全観測が予測クラス0。
        let table_high = estimator.pred_table(0.99);
        assert!((*table_high.get(0, 0) - 3.0).abs() < 1e-12);
        assert!((*table_high.get(0, 1) - 0.0).abs() < 1e-12);
        assert!((*table_high.get(1, 0) - 4.0).abs() < 1e-12);
        assert!((*table_high.get(1, 1) - 0.0).abs() < 1e-12);
    }

    /// `pred_table`が返すカウントの総和が観測数`n`と一致すること、および`predict()`の
    /// 出力から独立に再計算した分類結果（同じ`threshold`での二値化）と一致することを
    /// 多変量モデルで検証する（`pred_table`内部の配線ミスを検出できる設計、
    /// `fit_cov_type_opg_hc0_hc1_match_independently_recomputed_values`と同じ技法）。
    #[test]
    fn pred_table_matches_independently_recomputed_classification() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = LogitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        // `threshold≠0.5`にする（`actual`側は`threshold`に依存せず常に0.5固定である
        // ことを検出できるようにするため。`threshold=0.5`固定のテストでは、この
        // 区別がつかない）。
        let threshold = 0.2;
        let predicted = estimator.predict();
        let table = estimator.pred_table(threshold);

        let mut expected = [[0.0; 2]; 2];
        for i in 0..4 {
            let actual = usize::from(y[i] >= 0.5);
            let pred = usize::from(predicted[i] > threshold);
            expected[actual][pred] += 1.0;
        }

        let mut total = 0.0;
        for (a, row) in expected.iter().enumerate() {
            for (p, &expected_count) in row.iter().enumerate() {
                assert!((*table.get(a, p) - expected_count).abs() < 1e-12);
                total += *table.get(a, p);
            }
        }
        assert!((total - 4.0).abs() < 1e-12);
    }

    /// `pred_table`の実測クラス（行方向の合計、`actual0`の件数+`actual1`の件数）は
    /// `threshold`の値に関わらず不変であるべき（statsmodelsの`pred_table`が実測`y`を
    /// 常に固定の0.5分割でバケット化し、`threshold`は予測確率側にのみ適用する仕様、
    /// `pred_table`のdocコメント参照）。この不変性が保たれているかを回帰テストとして
    /// 固定する（`actual`まで`threshold`で二値化すると、`threshold≠0.5`で
    /// この不変性が壊れる）。
    #[test]
    fn pred_table_actual_class_counts_are_invariant_to_threshold() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = LogitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let table = estimator.pred_table(threshold);
            let actual0 = *table.get(0, 0) + *table.get(0, 1);
            let actual1 = *table.get(1, 0) + *table.get(1, 1);
            // y=[0,1,0,1] → actual0=2件・actual1=2件（`threshold`に関わらず常に一定）
            assert!(
                (actual0 - 2.0).abs() < 1e-12,
                "threshold={threshold}, actual0={actual0}"
            );
            assert!(
                (actual1 - 2.0).abs() < 1e-12,
                "threshold={threshold}, actual1={actual1}"
            );
        }
    }

    // ── Fitted（pickleからの復元に使う最小の状態） ─────────────────────

    fn fitted_test_estimator() -> LogitEstimator {
        let y = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let x_columns = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![0.5, -1.0, 2.0, 0.0, 1.5, -0.5, 1.0, 2.5],
        ];
        let input = LogitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        LogitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 35,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap()
    }

    /// `original`と同じ設計行列を、本番の復元と同じ`from_columns`で作り直す。
    fn rebuilt_input(original: &LogitFitted) -> LogitInput {
        let input = original.input();
        let start = usize::from(input.has_intercept());
        let n = input.nobs();
        let x_columns: Vec<Vec<f64>> = (start..input.k())
            .map(|j| (0..n).map(|i| *input.x().get(i, j)).collect())
            .collect();
        let y: Vec<f64> = (0..n).map(|i| *input.y().get(i, 0)).collect();
        LogitInput::from_columns(
            &y,
            &x_columns,
            input.param_names()[start..].to_vec(),
            input.has_intercept(),
            input.dep_var_name().to_string(),
        )
        .unwrap()
    }

    fn rebuilt_from_parts(original: &LogitFitted) -> LogitFitted {
        LogitFitted::from_parts(
            rebuilt_input(original),
            original.params().to_vec(),
            original.cov_params().clone(),
        )
        .unwrap()
    }

    #[test]
    fn fitted_rebuilt_from_parts_reproduces_predict_pred_table_and_marginal_effects() {
        let estimator = fitted_test_estimator();
        let expected_effects = estimator
            .marginal_effects(MarginalEffectsAt::Overall, 0.9)
            .unwrap();
        let expected_table = estimator.pred_table(0.4);
        let expected_predict = estimator.predict();
        let original = estimator.into_fitted();
        let rebuilt = rebuilt_from_parts(&original);

        assert_eq!(rebuilt.predict(), expected_predict);
        assert_eq!(rebuilt.params(), original.params());
        let table = rebuilt.pred_table(0.4);
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(table.get(i, j), expected_table.get(i, j));
            }
        }
        let effects = rebuilt
            .marginal_effects(MarginalEffectsAt::Overall, 0.9)
            .unwrap();
        assert_eq!(effects.effect(), expected_effects.effect());
        assert_eq!(effects.std_errors(), expected_effects.std_errors());
    }

    #[test]
    fn from_parts_rejects_params_or_covariance_of_the_wrong_size() {
        let original = fitted_test_estimator().into_fitted();
        let make_input = || rebuilt_input(&original);
        let k = original.params().len();
        assert!(
            LogitFitted::from_parts(
                make_input(),
                vec![0.0; k + 1],
                original.cov_params().clone()
            )
            .is_err()
        );
        assert!(
            LogitFitted::from_parts(
                make_input(),
                original.params().to_vec(),
                Mat::zeros(k, k + 1)
            )
            .is_err()
        );
        assert!(
            LogitFitted::from_parts(
                make_input(),
                original.params().to_vec(),
                Mat::zeros(k + 1, k)
            )
            .is_err()
        );
    }
}
