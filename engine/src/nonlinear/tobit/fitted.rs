//! `TobitFitted`（予測・限界効果・打ち切り適合度の確認に必要な最小の状態）と、その復元用の`from_parts`。

use super::censoring_check::{CensoringFitCategory, CensoringFitCheck};
use super::input::TobitInput;
use super::marginal_effects::{
    TobitMarginalEffectsInputs, at_point_target_w_and_s, marginal_effects_from_tobit_w_s,
    overall_target_w_and_s,
};
use super::mills::boundary_terms;
use super::target::{
    MarginalEffectsTarget, ensure_uncensored_interval_resolvable, predicted_value,
};
use crate::nonlinear::common::{
    MarginalEffects, MarginalEffectsAt, MleError, column_means, column_medians, predict_new_data,
};
use crate::shared::error::CommonError;
use faer::Mat;
use statrs::distribution::Normal;

/// Tobitの推定結果のうち、予測・限界効果・打ち切りの当てはまりの確認に必要な最小の状態
/// （入力データ・係数・`σ`・`(β, σ)`の分散共分散行列）。
///
/// [`super::TobitEstimator::into_fitted`]で取り出す。`engine_pybind`の結果オブジェクトは推定量全体ではなくこれを
/// 保持し、pickleでは[`TobitFitted::from_parts`]で復元する。フィールドはprivate（`.claude/rules/rust-style.md`
/// 「推定量構造体の設計」参照）で、構築経路は`fit`と`from_parts`だけ。
#[derive(Debug)]
pub struct TobitFitted {
    pub(super) input: TobitInput,
    /// 係数（元のスケール、`β`部分のみ）。`input.param_names()`と対応する
    pub(super) params: Vec<f64>,
    /// 誤差項の標準偏差（元のスケール）
    pub(super) sigma: f64,
    /// `(β, σ)`の分散共分散行列（元のスケール、`(k+1)×(k+1)`）
    pub(super) cov_params: Mat<f64>,
}

impl TobitFitted {
    /// 入力データ・係数・`σ`・分散共分散行列から組み立てる。
    ///
    /// 最適性は検証できない（`fit`が収束した結果であることは呼び出し側が保証する）。ここで検査するのは、
    /// 後続のメソッドが添字の範囲外にならないための次元の整合と、`σ`が正の有限値であることだけ。
    ///
    /// 壊れた・改ざんされたpickleだけが到達する入力エラーで、`engine_pybind`が`PICKLE_FORMAT_MISMATCH`へ変換する。
    /// `params`・`cov_params`の値が有限かは検査しない（非有限値は後続の計算にNaNとして伝わるだけ）。
    ///
    /// # Errors
    /// `params`の長さが`k`と合わない、`cov_params`が`(k+1)×(k+1)`でない、`σ`が正の有限値でない:
    /// `CommonError::Internal`
    pub fn from_parts(
        input: TobitInput,
        params: Vec<f64>,
        sigma: f64,
        cov_params: Mat<f64>,
    ) -> Result<Self, MleError> {
        let k = input.k();
        let consistent = params.len() == k
            && cov_params.nrows() == k + 1
            && cov_params.ncols() == k + 1
            && sigma.is_finite()
            && sigma > 0.0;
        if !consistent {
            return Err(CommonError::internal("inconsistent fitted model parts").into());
        }
        Ok(Self {
            input,
            params,
            sigma,
            cov_params,
        })
    }

    pub fn input(&self) -> &TobitInput {
        &self.input
    }

    pub fn params(&self) -> &[f64] {
        &self.params
    }

    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    pub fn cov_params(&self) -> &Mat<f64> {
        &self.cov_params
    }

    /// 限界効果（`marginal_effects`）。`fit()`とは独立した別メソッド（`fit()`のReturn
    /// 本体には含めない、`docs/spec/nonlinear-common.md`6章で確定済み）。`fit()`時の
    /// `cov_params`（`(k+1)×(k+1)`、`β∪{σ}`空間）を再利用するため再最適化は不要
    /// （`confidence_level`は`fit()`とは独立したパラメータとして受け取る、
    /// `LogitEstimator::marginal_effects`と同じ設計）。
    ///
    /// `target`で評価対象を選ぶ（`MarginalEffectsTarget`のdocコメント参照）。
    /// Logit/Probitの`overall_w_and_s`/`at_point_w_and_s`型の共通化はしない
    /// （計算式自体は`target_w_and_s`/`marginal_effects_from_tobit_w_s`
    /// のdocコメント参照）。左打ち切りのみ・右打ち切りのみ・両側打ち切りいずれの
    /// `TobitInput`でも同じ式で正しく計算できる（`target_w_and_s`のdocコメント「数式」参照）。
    ///
    /// 定数項（切片）は出力から除外する（Logit/Probitと同じ方針）。離散変数（0/1ダミー）の
    /// 自動判定は行わない（常に連続変数として扱う、Logit/Probitと同じ方針）。
    ///
    /// # Errors
    /// - `confidence_level`が`(0, 1)`の範囲外: `CommonError::InvalidConfidenceLevel`
    /// - `target`が`ExpectedUncensored`で、両側打ち切りの非打ち切り区間が`σ̂`に比べて狭すぎる:
    ///   `MleError::UncensoredIntervalTooNarrow`
    pub fn marginal_effects(
        &self,
        at: MarginalEffectsAt,
        target: MarginalEffectsTarget,
        confidence_level: f64,
    ) -> Result<MarginalEffects, MleError> {
        let x = self.input.x();
        let k = self.input.k();
        let lower = self.input.lower();
        let upper = self.input.upper();
        let normal = Normal::standard();
        if target == MarginalEffectsTarget::ExpectedUncensored {
            ensure_uncensored_interval_resolvable(lower, upper, self.sigma)?;
        }

        let (w, s_beta, s_sigma) = match at {
            MarginalEffectsAt::Overall => {
                overall_target_w_and_s(x, &self.params, self.sigma, lower, upper, target, &normal)
            }
            MarginalEffectsAt::Mean => at_point_target_w_and_s(
                &column_means(x),
                &self.params,
                self.sigma,
                lower,
                upper,
                target,
                &normal,
            ),
            MarginalEffectsAt::Median => at_point_target_w_and_s(
                &column_medians(x),
                &self.params,
                self.sigma,
                lower,
                upper,
                target,
                &normal,
            ),
        };

        marginal_effects_from_tobit_w_s(
            TobitMarginalEffectsInputs {
                param_names: self.input.param_names(),
                has_intercept: self.input.has_intercept(),
                k,
                beta: &self.params,
                cov_params: &self.cov_params,
            },
            w,
            &s_beta,
            s_sigma,
            confidence_level,
        )
    }

    /// 予測値（`predict`）。`target`で予測量を選ぶ（`MarginalEffectsTarget`を
    /// `marginal_effects`と共有する。「限界効果の対象」と「予測値の対象」は同じ3種
    /// （`E[y*|x]`・`E[y|x]`・`P(uncensored|x)`）を指すため、専用のenumを別途新設せず
    /// 再利用する）。`fit()`に使った学習データ（`self.input.x()`）の各行について返す
    /// （`predicted_value`のdocコメント「数式」参照。左のみ・右のみ・両側打ち切り
    /// いずれでも同じ式で正しく計算できる）。
    ///
    /// 新規データでの予測（out-of-sample）は`predict_new_data`。デフォルト
    /// （`target`省略時の`E[y|x]`、`docs/spec/nonlinear-common.md`
    /// 6章）はPython層（engine_pybind）の責務（`SolverType`/`CovType`等と同じ設計、
    /// `.claude/rules/rust-style.md`参照）。
    ///
    /// # Errors
    /// `target`が`ExpectedUncensored`で、両側打ち切りの非打ち切り区間が`σ̂`に比べて狭すぎる:
    /// `MleError::UncensoredIntervalTooNarrow`
    pub fn predict(&self, target: MarginalEffectsTarget) -> Result<Vec<f64>, MleError> {
        let x = self.input.x();
        let n = x.nrows();
        let k = self.input.k();
        let lower = self.input.lower();
        let upper = self.input.upper();
        let normal = Normal::standard();
        if target == MarginalEffectsTarget::ExpectedUncensored {
            ensure_uncensored_interval_resolvable(lower, upper, self.sigma)?;
        }
        Ok((0..n)
            .map(|i| {
                let mu: f64 = (0..k).map(|j| *x.get(i, j) * self.params[j]).sum();
                predicted_value(target, mu, self.sigma, lower, upper, &normal)
            })
            .collect())
    }

    /// 新規データ（out-of-sample、`new_x_columns`）に対する予測値。
    /// `target`は`predict`と同じ3種。
    ///
    /// `nonlinear::common::predict_new_data`に`predicted_value`（`mu`から`target`の
    /// 値を計算する関数）を部分適用したクロージャを`link`として渡すだけの薄い
    /// ラッパー（`LogitEstimator::predict_new_data`/`ProbitEstimator::predict_new_data`
    /// と同じ設計。`predict_new_data`は「`x_i'θ`を計算して`link`に渡す」処理しか
    /// 行わないため、`link`が`logistic`/正規分布CDFのような単純な関数でなくても
    /// （`target`・`sigma`・`lower`・`upper`を閉じ込めたクロージャでも）そのまま使える）。
    ///
    /// `new_x_columns`の本数・順序の契約、パニック条件は
    /// `nonlinear::common::predict_new_data`のdocコメント参照。
    ///
    /// # Errors
    /// `predict`と同じ（`MleError::UncensoredIntervalTooNarrow`）。
    pub fn predict_new_data(
        &self,
        target: MarginalEffectsTarget,
        new_x_columns: &[Vec<f64>],
    ) -> Result<Vec<f64>, MleError> {
        let has_intercept = self.input.has_intercept();
        let lower = self.input.lower();
        let upper = self.input.upper();
        let sigma = self.sigma;
        let normal = Normal::standard();
        if target == MarginalEffectsTarget::ExpectedUncensored {
            ensure_uncensored_interval_resolvable(lower, upper, sigma)?;
        }
        Ok(predict_new_data(
            &self.params,
            has_intercept,
            new_x_columns,
            |mu| predicted_value(target, mu, sigma, lower, upper, &normal),
        ))
    }

    /// 打ち切り予測の適合度チェック（Logit/Probitの`pred_table`の代替、`predict`とは独立
    /// した別メソッド）。`lower`/`upper`それぞれが定義する方向（下側打ち切り・非打ち切り・
    /// 上側打ち切り）ごとに、実測の比率（`y`がちょうど境界値に一致する観測の割合）と
    /// モデル含意の平均確率（各観測の`P(該当カテゴリ|xᵢ)`の平均、`predicted_value`の
    /// `ProbUncensored`と同じ`Φ`の組み合わせ）を突き合わせる（方向別
    /// 内訳を採用する。単一集約値ではなく、両側打ち切りでどちらの境界に不整合があるかを
    /// 区別できる形にする）。`lower`/`upper`が`None`の方向は出力からも`None`になる。
    ///
    /// 3カテゴリの`model_implied_rate`は理論上必ず合計1になる
    /// （`Φ(za) + (Φ(zb)-Φ(za)) + (1-Φ(zb)) = 1`）。`observed_rate`も同様
    /// （定義上、3カテゴリで観測を排他的に分割するため）。
    ///
    /// **前提**: 打ち切り観測の判定は`yᵢ == lower`/`yᵢ == upper`という
    /// 浮動小数点の完全一致比較で行う。Tobitの定義上「打ち切り観測の`y`はちょうど境界値」
    /// という理論的性質自体は正しく、`TobitInput::from_columns`が`y`を変換せずそのまま
    /// 保持する設計（クリッピング等を行わない）とも整合するが、これは呼び出し側
    /// （`engine_pybind`、将来的にはPython層）が渡す`y`列の打ち切り観測の値と、`lower`/
    /// `upper`として渡す値がビット単位で完全に一致することを前提とする。CSV/Parquet
    /// 経由での読み込みや`Float32`→`Float64`変換等、この前提が崩れうる経路がある場合は
    /// 呼び出し側で丸め誤差が生じないよう注意する必要がある（許容誤差付き比較は
    /// 未導入。相対閾値が要る場合は`ensure_well_conditioned_symmetric_matrix`等と同じ
    /// 発想で別途検討する）。
    pub fn censoring_fit_check(&self) -> CensoringFitCheck {
        let x = self.input.x();
        let y = self.input.y();
        let n = x.nrows();
        let k = self.input.k();
        let lower = self.input.lower();
        let upper = self.input.upper();
        let normal = Normal::standard();

        let mut observed_lower = 0usize;
        let mut observed_upper = 0usize;
        let mut model_lower_sum = 0.0;
        let mut model_upper_sum = 0.0;
        let mut model_uncensored_sum = 0.0;

        for i in 0..n {
            let mu: f64 = (0..k).map(|j| *x.get(i, j) * self.params[j]).sum();
            let (_, _, cdf_za) = boundary_terms(lower, mu, self.sigma, &normal, true);
            let (_, _, cdf_zb) = boundary_terms(upper, mu, self.sigma, &normal, false);
            model_lower_sum += cdf_za;
            model_upper_sum += 1.0 - cdf_zb;
            model_uncensored_sum += cdf_zb - cdf_za;

            let y_i = *y.get(i, 0);
            if let Some(l) = lower
                && y_i == l
            {
                observed_lower += 1;
            }
            if let Some(u) = upper
                && y_i == u
            {
                observed_upper += 1;
            }
        }

        let n_f = n as f64;
        let lower_check = lower.map(|_| CensoringFitCategory {
            observed_rate: observed_lower as f64 / n_f,
            model_implied_rate: model_lower_sum / n_f,
        });
        let upper_check = upper.map(|_| CensoringFitCategory {
            observed_rate: observed_upper as f64 / n_f,
            model_implied_rate: model_upper_sum / n_f,
        });
        let uncensored_observed = (n - observed_lower - observed_upper) as f64 / n_f;

        CensoringFitCheck {
            lower: lower_check,
            uncensored: CensoringFitCategory {
                observed_rate: uncensored_observed,
                model_implied_rate: model_uncensored_sum / n_f,
            },
            upper: upper_check,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::super::target::MIN_UNCENSORED_WIDTH_RATIO;
    use super::super::target::MarginalEffectsTarget;

    use super::super::test_support::*;

    use crate::nonlinear::common::CovType;

    use crate::nonlinear::common::MarginalEffectsAt;
    use crate::nonlinear::common::MleError;
    use crate::nonlinear::common::MleFitOptions;

    use crate::nonlinear::common::SolverType;

    use crate::nonlinear::tobit::TobitEstimator;
    use crate::nonlinear::tobit::TobitFitted;
    use crate::nonlinear::tobit::TobitInput;

    use crate::shared::error::CommonError;

    use faer::Mat;

    use statrs::distribution::Continuous;
    use statrs::distribution::ContinuousCDF;
    use statrs::distribution::Normal;

    /// `censored_regression_input`（左打ち切りのみ、`lower=Some(0.0)`）の右打ち切り版。
    /// 潜在変数`y* = -5 + 2x + noise`は同じだが、`upper=Some(6.0)`で上側を打ち切る
    /// （x=6,7,8の潜在値6.85・9.1・10.95が6.0に打ち切られる、打ち切り率37.5%）。
    /// `marginal_effects`が右打ち切りのみのケースでも正しく計算できることを検証する
    /// データセット（`target_w_and_s`のdocコメント「数式」参照）。
    fn right_censored_regression_input() -> TobitInput {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![-2.85, -0.95, 1.15, 2.9, 5.2, 6.0, 6.0, 6.0];
        TobitInput::from_columns(
            &y,
            &[x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            None,
            Some(6.0),
        )
        .unwrap()
    }

    /// `censored_regression_input`の両側打ち切り版。同じ潜在変数`y* = -5 + 2x + noise`に
    /// `lower=Some(0.0)`・`upper=Some(6.0)`の両方を適用する（x=1,2は下側、x=6,7,8は上側で
    /// 打ち切り、非打ち切りはx=3,4,5のみ）。
    fn two_sided_censored_regression_input() -> TobitInput {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![0.0, 0.0, 1.15, 2.9, 5.2, 6.0, 6.0, 6.0];
        TobitInput::from_columns(
            &y,
            &[x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            Some(6.0),
        )
        .unwrap()
    }

    /// `E[y|x]`の閉形式（McDonald-Moffitt 1980）。左打ち切りのみ・右打ち切りのみ・両側
    /// 打ち切りをそれぞれ独立した分岐で書き下す（`target_w_and_s`の統一的な
    /// `boundary_terms`実装とは別のコードパスで検算するため、`(lower,upper)`ごとの
    /// 教科書通りの式をそのまま使う）。
    fn expected_observed_closed_form(
        mu: f64,
        sigma: f64,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        match (lower, upper) {
            (Some(c), None) => {
                let z = (mu - c) / sigma;
                normal.cdf(z) * mu + sigma * normal.pdf(z) + (1.0 - normal.cdf(z)) * c
            }
            (None, Some(c)) => {
                let z = (c - mu) / sigma;
                normal.cdf(z) * mu - sigma * normal.pdf(z) + (1.0 - normal.cdf(z)) * c
            }
            (Some(c1), Some(c2)) => {
                let za = (c1 - mu) / sigma;
                let zb = (c2 - mu) / sigma;
                normal.cdf(za) * c1
                    + (1.0 - normal.cdf(zb)) * c2
                    + (normal.cdf(zb) - normal.cdf(za)) * mu
                    - sigma * (normal.pdf(zb) - normal.pdf(za))
            }
            (None, None) => unreachable!("TobitInput::from_columns rejects both-None bounds"),
        }
    }

    /// `P(uncensored|x)`の閉形式。`expected_observed_closed_form`と同じ理由で
    /// `(lower,upper)`ごとに独立した分岐で書く。
    fn prob_uncensored_closed_form(
        mu: f64,
        sigma: f64,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        match (lower, upper) {
            (Some(c), None) => normal.cdf((mu - c) / sigma),
            (None, Some(c)) => normal.cdf((c - mu) / sigma),
            (Some(c1), Some(c2)) => normal.cdf((c2 - mu) / sigma) - normal.cdf((c1 - mu) / sigma),
            (None, None) => unreachable!("TobitInput::from_columns rejects both-None bounds"),
        }
    }

    /// `∂P(uncensored|x)/∂μ`の閉形式（`prob_uncensored_closed_form`を`mu`について解析的に
    /// 微分した式。`target_w_and_s`の`ProbUncensored`分岐とは独立に書き下したコード）。
    fn dprob_uncensored_dmu_closed_form(
        mu: f64,
        sigma: f64,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        match (lower, upper) {
            (Some(c), None) => normal.pdf((c - mu) / sigma) / sigma,
            (None, Some(c)) => -normal.pdf((c - mu) / sigma) / sigma,
            (Some(c1), Some(c2)) => {
                (normal.pdf((c1 - mu) / sigma) - normal.pdf((c2 - mu) / sigma)) / sigma
            }
            (None, None) => unreachable!("TobitInput::from_columns rejects both-None bounds"),
        }
    }

    /// ノイズの大きい決定論的な打ち切りデータ（`y*=-2+0.7x+ε`、`ε`は`±2.5`の一様、
    /// `x=0.25·i`、`i=0..60`）に`lower`/`upper`を適用した入力。既存の少数観測データは
    /// 当てはめ後の`σ`が小さく`z`が裾に入るため、素直な閉形式（`expected_uncensored_closed_form`）
    /// で独立に検算できる範囲に`z`を収める用途。
    fn noisy_censored_input(lower: Option<f64>, upper: Option<f64>) -> TobitInput {
        let x: Vec<f64> = (0..60).map(|i| 0.25 * f64::from(i)).collect();
        let y: Vec<f64> = (0..60u32)
            .map(|i| {
                let u = f64::from((i * 7919) % 1009) / 1009.0;
                let latent = -2.0 + 0.7 * x[i as usize] + (u - 0.5) * 5.0;
                let latent = lower.map_or(latent, |l| latent.max(l));
                upper.map_or(latent, |u| latent.min(u))
            })
            .collect();
        TobitInput::from_columns(
            &y,
            &[x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            lower,
            upper,
        )
        .unwrap()
    }

    /// `at="overall"`の限界効果（effect・デルタ法のSE）を、独立の閉形式`m`から作った
    /// AME関数の中心差分ヤコビアンで再計算した値と突き合わせる（左のみ・右のみ・両側）。
    #[test]
    fn fit_marginal_effects_expected_uncensored_overall_matches_independent_recomputation() {
        let inputs = [
            noisy_censored_input(Some(0.0), None),
            noisy_censored_input(None, Some(6.0)),
            noisy_censored_input(Some(0.0), Some(6.0)),
        ];
        for input in inputs {
            let estimator = TobitEstimator::fit(
                input,
                MleFitOptions {
                    solver: SolverType::Newton,
                    max_iter: 100,
                    tol: 1e-8,
                    raise_on_non_convergence: true,
                    cov_type: CovType::Classical,
                    confidence_level: 0.95,
                },
            )
            .unwrap();
            let lower = estimator.input().lower();
            let upper = estimator.input().upper();
            let x = estimator.input().x();
            let n = x.nrows();
            let k = 2;
            let label = format!("lower={lower:?}, upper={upper:?}");

            let effects = estimator
                .marginal_effects(
                    MarginalEffectsAt::Overall,
                    MarginalEffectsTarget::ExpectedUncensored,
                    0.95,
                )
                .unwrap();

            // AME_j = (1/n)Σ w_i·β_j, w_i=∂m/∂μ を m の中心差分で作る（k=2なのでj=1のみ）
            let ame = |params: &[f64], sigma: f64| -> f64 {
                let h = 1e-5;
                (0..n)
                    .map(|i| {
                        let mu: f64 = params[0] + params[1] * (*x.get(i, 1));
                        let m = |mu: f64| expected_uncensored_closed_form(mu, sigma, lower, upper);
                        (m(mu + h) - m(mu - h)) / (2.0 * h) * params[1]
                    })
                    .sum::<f64>()
                    / (n as f64)
            };
            let params = estimator.params();
            let sigma = estimator.sigma();
            assert!(
                (effects.effect()[0] - ame(params, sigma)).abs() < 1e-6,
                "{label}: actual={}, expected={}",
                effects.effect()[0],
                ame(params, sigma)
            );

            let h_param = 1e-3;
            let mut jac_row = vec![0.0; k + 1];
            for (m_idx, jac) in jac_row.iter_mut().take(k).enumerate() {
                let mut plus_p = params.to_vec();
                plus_p[m_idx] += h_param;
                let mut minus_p = params.to_vec();
                minus_p[m_idx] -= h_param;
                *jac = (ame(&plus_p, sigma) - ame(&minus_p, sigma)) / (2.0 * h_param);
            }
            jac_row[k] =
                (ame(params, sigma + h_param) - ame(params, sigma - h_param)) / (2.0 * h_param);
            let mut var = 0.0;
            for a in 0..=k {
                for b in 0..=k {
                    var += jac_row[a] * (*estimator.cov_params().get(a, b)) * jac_row[b];
                }
            }
            let expected_se = var.sqrt();
            assert!(
                (effects.std_errors()[0] - expected_se).abs() < 1e-5 * (1.0 + expected_se),
                "{label}: actual SE={}, expected SE={expected_se}",
                effects.std_errors()[0]
            );
        }
    }

    /// `predict`/`predict_new_data`が`ExpectedUncensored`で独立の閉形式と一致し、
    /// 学習データへの`predict`と同じ`x`での`predict_new_data`が一致する。
    #[test]
    fn predict_and_predict_new_data_expected_uncensored_match_closed_form() {
        let inputs = [
            noisy_censored_input(Some(0.0), None),
            noisy_censored_input(None, Some(6.0)),
            noisy_censored_input(Some(0.0), Some(6.0)),
        ];
        for input in inputs {
            let estimator = TobitEstimator::fit(
                input,
                MleFitOptions {
                    solver: SolverType::Newton,
                    max_iter: 100,
                    tol: 1e-8,
                    raise_on_non_convergence: true,
                    cov_type: CovType::Classical,
                    confidence_level: 0.95,
                },
            )
            .unwrap();
            let lower = estimator.input().lower();
            let upper = estimator.input().upper();
            let x = estimator.input().x();
            let params = estimator.params();
            let sigma = estimator.sigma();

            let predicted = estimator
                .predict(MarginalEffectsTarget::ExpectedUncensored)
                .unwrap();
            for (i, &p) in predicted.iter().enumerate() {
                let mu: f64 = params[0] + params[1] * (*x.get(i, 1));
                let expected = expected_uncensored_closed_form(mu, sigma, lower, upper);
                assert!(
                    (p - expected).abs() < 1e-9 * (1.0 + expected.abs()),
                    "lower={lower:?}, upper={upper:?}, i={i}: actual={p}, expected={expected}"
                );
            }

            let x1: Vec<f64> = (0..x.nrows()).map(|i| *x.get(i, 1)).collect();
            let predicted_new = estimator
                .predict_new_data(MarginalEffectsTarget::ExpectedUncensored, &[x1])
                .unwrap();
            assert_eq!(predicted_new.len(), predicted.len());
            for (a, b) in predicted_new.iter().zip(&predicted) {
                assert!((a - b).abs() < 1e-12, "predict_new_data={a}, predict={b}");
            }
        }
    }

    /// ほぼ全観測が両端に打ち切られた当てはめ（`σ̂`が区間幅の100倍超）。
    fn narrow_interval_estimator(n_interior: usize) -> TobitEstimator {
        let n = 2000usize;
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 7919) % 1009) as f64 / 1009.0)
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                if i < n_interior {
                    0.5
                } else if i % 2 == 0 {
                    0.0
                } else {
                    1.0
                }
            })
            .collect();
        let input = TobitInput::from_columns(
            &y,
            &[x],
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            Some(1.0),
        )
        .unwrap();
        TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 200,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap()
    }

    /// 狭い区間の当てはめ（`σ̂≈160`、`δ≈6e-3`）では`ExpectedUncensored`の`predict`・
    /// `predict_new_data`・`marginal_effects`が`UncensoredIntervalTooNarrow`になり、他の3targetは
    /// 通常どおり計算できる（エラーは`expected_uncensored`固有）。区間がちょうど十分な当てはめ
    /// （`σ̂≈40`、`δ≈0.025`）では`ExpectedUncensored`も計算できる。
    #[test]
    fn expected_uncensored_errors_only_when_the_fitted_interval_is_too_narrow() {
        let narrow = narrow_interval_estimator(5);
        assert!(
            1.0 / narrow.sigma() < MIN_UNCENSORED_WIDTH_RATIO,
            "fixture is not narrow"
        );
        let new_x = vec![vec![0.2, 0.8]];
        assert!(matches!(
            narrow.predict(MarginalEffectsTarget::ExpectedUncensored),
            Err(MleError::UncensoredIntervalTooNarrow { .. })
        ));
        assert!(matches!(
            narrow.predict_new_data(MarginalEffectsTarget::ExpectedUncensored, &new_x),
            Err(MleError::UncensoredIntervalTooNarrow { .. })
        ));
        assert!(matches!(
            narrow.marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ExpectedUncensored,
                0.95
            ),
            Err(MleError::UncensoredIntervalTooNarrow { .. })
        ));
        for target in [
            MarginalEffectsTarget::ExpectedLatent,
            MarginalEffectsTarget::ExpectedObserved,
            MarginalEffectsTarget::ProbUncensored,
        ] {
            let predicted = narrow.predict(target).unwrap();
            assert!(predicted.iter().all(|v| v.is_finite()), "{target:?}");
            assert!(
                narrow.predict_new_data(target, &new_x).is_ok(),
                "{target:?}"
            );
            assert!(
                narrow
                    .marginal_effects(MarginalEffectsAt::Overall, target, 0.95)
                    .is_ok(),
                "{target:?}"
            );
        }

        let wide_enough = narrow_interval_estimator(20);
        assert!(1.0 / wide_enough.sigma() >= MIN_UNCENSORED_WIDTH_RATIO);
        let predicted = wide_enough
            .predict(MarginalEffectsTarget::ExpectedUncensored)
            .unwrap();
        assert!(
            predicted
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
        );
    }

    /// 切片のみモデル（k=1）は限界効果の出力対象となる説明変数が存在しない
    /// （定数項は出力から除外するため）。`marginal_effects`が空の結果を返す
    /// （パニックしない）ことを確認する境界ケース（Logitの同名テストと同じ理由）。
    #[test]
    fn fit_marginal_effects_returns_empty_result_for_intercept_only_model() {
        let estimator = TobitEstimator::fit(
            intercept_only_uncensored_input(&[1.0, 2.0, 3.0, 4.0]),
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

        let effects = estimator
            .marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ExpectedObserved,
                0.95,
            )
            .unwrap();
        assert!(effects.param_names().is_empty());
        assert!(effects.effect().is_empty());
    }

    #[test]
    fn fit_marginal_effects_returns_invalid_confidence_level_error_out_of_range() {
        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        let result = estimator.marginal_effects(
            MarginalEffectsAt::Overall,
            MarginalEffectsTarget::ExpectedObserved,
            1.5,
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5
            })
        );
    }

    /// `target="E[y*|x]"`（潜在変数の期待値）は`dydx_j=βⱼ`という自明な形（評価点`x`に
    /// 依存しない）ため、`at`（overall/mean/median）に関わらず常に同じ値になり、
    /// その標準誤差も`β`自体の標準誤差（`std_errors()`の対応する要素）と厳密に一致する
    /// （ヤコビアンが`β`ブロックで恒等写像・`σ`列がゼロに退化するため、数値微分に頼らず
    /// 厳密等式として検証できる）。
    #[test]
    fn fit_marginal_effects_expected_latent_equals_beta_and_matches_beta_std_error() {
        let estimator = TobitEstimator::fit(
            multivariate_censored_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        for at in [
            MarginalEffectsAt::Overall,
            MarginalEffectsAt::Mean,
            MarginalEffectsAt::Median,
        ] {
            let effects = estimator
                .marginal_effects(at, MarginalEffectsTarget::ExpectedLatent, 0.95)
                .unwrap();
            assert_eq!(effects.param_names(), ["x1".to_string(), "x2".to_string()]);
            for (idx, j) in (1..3).enumerate() {
                assert!((effects.effect()[idx] - estimator.params()[j]).abs() < 1e-12);
                assert!(
                    (effects.std_errors()[idx] - estimator.std_errors()[j]).abs() < 1e-9,
                    "at={at:?}, idx={idx}, actual={}, expected={}",
                    effects.std_errors()[idx],
                    estimator.std_errors()[j]
                );
            }
        }
    }

    /// `target="E[y|x]"`（既定）の`at="overall"`（AME）を、実装内部（`target_w_and_s`）とは
    /// 別に書いた閉形式`expected_observed_closed_form`を各観測で平均する方法で独立に
    /// 再計算し、突き合わせる。標準誤差は、`dydx_j`を`prob_uncensored_closed_form`
    /// （非打ち切り確率の閉形式）×`βⱼ`という独立な式で表し、fit済みパラメータの周りで
    /// 数値微分したヤコビアンと`cov_params`の二次形式で計算した分散の平方根と突き合わせる
    /// （Logitの`marginal_effects_overall_matches_independently_recomputed_dydx_and_delta_
    /// method_se`と同じ検算パターン）。
    #[test]
    fn fit_marginal_effects_expected_observed_overall_matches_independently_recomputed_dydx_and_delta_method_se()
     {
        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 2;

        let effects = estimator
            .marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ExpectedObserved,
                0.95,
            )
            .unwrap();
        assert_eq!(effects.param_names(), ["x1".to_string()]);

        // effectの独立再計算: 各観測でE[y|x]を数値微分してx1の限界効果を求め、平均する
        // （`target_w_and_s`の`P(uncensored)*βⱼ`という閉形式は経由しない）。
        let h = 1e-6;
        let dydx_at_row = |params: &[f64], sigma: f64, i: usize| -> f64 {
            let mu = |x1_val: f64| -> f64 { params[0] + params[1] * x1_val };
            let x1 = *x.get(i, 1);
            let plus = expected_observed_closed_form(mu(x1 + h), sigma, lower, upper);
            let minus = expected_observed_closed_form(mu(x1 - h), sigma, lower, upper);
            (plus - minus) / (2.0 * h)
        };
        let dydx_ame = |params: &[f64], sigma: f64| -> f64 {
            (0..n).map(|i| dydx_at_row(params, sigma, i)).sum::<f64>() / (n as f64)
        };
        let params = estimator.params();
        let sigma = estimator.sigma();
        assert!(
            (effects.effect()[0] - dydx_ame(params, sigma)).abs() < 1e-6,
            "actual={}, expected={}",
            effects.effect()[0],
            dydx_ame(params, sigma)
        );

        // 数値微分とは別の、McDonald-Moffitt(1980)の厳密な恒等式
        // `∂E[y|x]/∂xⱼ = P(uncensored|x)*βⱼ`（`target_w_and_s`のdocコメント「数式」参照）
        // による検算。`prob_uncensored_closed_form`は数値微分を一切経由しない厳密な閉形式
        // のため、数値誤差なしで一致するはず。
        let expected_dydx_exact: f64 = (0..n)
            .map(|i| {
                let mu: f64 = params[0] + params[1] * (*x.get(i, 1));
                prob_uncensored_closed_form(mu, sigma, lower, upper) * params[1]
            })
            .sum::<f64>()
            / (n as f64);
        assert!(
            (effects.effect()[0] - expected_dydx_exact).abs() < 1e-9,
            "actual={}, expected={expected_dydx_exact}",
            effects.effect()[0]
        );

        // 標準誤差の独立検証: `dydx_ame`をfit済み`(β,σ)`の周りで数値微分してヤコビアン行を
        // 求め、`cov_params`との二次形式で分散を計算する。
        let h_param = 1e-5;
        let mut jac_row = vec![0.0; k + 1];
        for m in 0..k {
            let mut plus_p = params.to_vec();
            plus_p[m] += h_param;
            let mut minus_p = params.to_vec();
            minus_p[m] -= h_param;
            jac_row[m] = (dydx_ame(&plus_p, sigma) - dydx_ame(&minus_p, sigma)) / (2.0 * h_param);
        }
        jac_row[k] = (dydx_ame(params, sigma + h_param) - dydx_ame(params, sigma - h_param))
            / (2.0 * h_param);

        let mut var = 0.0;
        for a in 0..=k {
            for b in 0..=k {
                var += jac_row[a] * (*estimator.cov_params().get(a, b)) * jac_row[b];
            }
        }
        let expected_se = var.sqrt();
        assert!(
            (effects.std_errors()[0] - expected_se).abs() < 1e-4,
            "actual={}, expected={expected_se}",
            effects.std_errors()[0]
        );
    }

    /// `at="mean"`/`"median"`が`at="overall"`と異なる代表点で評価されるため、一般には
    /// 異なる値になることを確認する（Logitの同名テストと同じ理由）。
    #[test]
    fn fit_marginal_effects_expected_observed_at_mean_and_median_differ_from_overall() {
        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();

        let overall = estimator
            .marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ExpectedObserved,
                0.95,
            )
            .unwrap();
        let at_mean = estimator
            .marginal_effects(
                MarginalEffectsAt::Mean,
                MarginalEffectsTarget::ExpectedObserved,
                0.95,
            )
            .unwrap();
        let at_median = estimator
            .marginal_effects(
                MarginalEffectsAt::Median,
                MarginalEffectsTarget::ExpectedObserved,
                0.95,
            )
            .unwrap();

        assert!((overall.effect()[0] - at_mean.effect()[0]).abs() > 1e-9);
        assert!((at_mean.effect()[0] - at_median.effect()[0]).abs() < 1e-12);

        // x1の平均・中央値がともに4.5（[1..8]の対称なデータのため）であることを利用し、
        // `expected_observed_closed_form`から独立に再計算したdydxと突き合わせる。
        let params = estimator.params();
        let sigma = estimator.sigma();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        let mu = |x1_val: f64| -> f64 { params[0] + params[1] * x1_val };
        let h = 1e-6;
        let expected_dydx = (expected_observed_closed_form(mu(4.5 + h), sigma, lower, upper)
            - expected_observed_closed_form(mu(4.5 - h), sigma, lower, upper))
            / (2.0 * h);
        assert!((at_mean.effect()[0] - expected_dydx).abs() < 1e-6);
    }

    /// `target="E[y|x]"`が右打ち切りのみのデータでも正しく計算できることを、
    /// `expected_observed_closed_form`（右打ち切り分岐）で独立に再計算した`effect`と
    /// 突き合わせて検証する。
    #[test]
    fn fit_marginal_effects_expected_observed_matches_independent_recomputation_for_right_only_censoring()
     {
        let estimator = TobitEstimator::fit(
            right_censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        assert_eq!(lower, None);
        let x = estimator.input().x();
        let n = x.nrows();
        let params = estimator.params();
        let sigma = estimator.sigma();

        let effects = estimator
            .marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ExpectedObserved,
                0.95,
            )
            .unwrap();

        let h = 1e-6;
        let mu = |x1_val: f64| -> f64 { params[0] + params[1] * x1_val };
        let expected_dydx: f64 = (0..n)
            .map(|i| {
                let x1 = *x.get(i, 1);
                let plus = expected_observed_closed_form(mu(x1 + h), sigma, lower, upper);
                let minus = expected_observed_closed_form(mu(x1 - h), sigma, lower, upper);
                (plus - minus) / (2.0 * h)
            })
            .sum::<f64>()
            / (n as f64);

        assert!(
            (effects.effect()[0] - expected_dydx).abs() < 1e-6,
            "actual={}, expected={expected_dydx}",
            effects.effect()[0]
        );
    }

    /// `target="E[y|x]"`が両側打ち切りのデータでも正しく計算できることを、
    /// `expected_observed_closed_form`（両側打ち切り分岐）で独立に再計算した`effect`と
    /// 突き合わせて検証する。
    #[test]
    fn fit_marginal_effects_expected_observed_matches_independent_recomputation_for_two_sided_censoring()
     {
        let estimator = TobitEstimator::fit(
            two_sided_censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        assert_eq!(lower, Some(0.0));
        assert_eq!(upper, Some(6.0));
        let x = estimator.input().x();
        let n = x.nrows();
        let params = estimator.params();
        let sigma = estimator.sigma();

        let effects = estimator
            .marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ExpectedObserved,
                0.95,
            )
            .unwrap();

        let h = 1e-6;
        let mu = |x1_val: f64| -> f64 { params[0] + params[1] * x1_val };
        let expected_dydx: f64 = (0..n)
            .map(|i| {
                let x1 = *x.get(i, 1);
                let plus = expected_observed_closed_form(mu(x1 + h), sigma, lower, upper);
                let minus = expected_observed_closed_form(mu(x1 - h), sigma, lower, upper);
                (plus - minus) / (2.0 * h)
            })
            .sum::<f64>()
            / (n as f64);

        assert!(
            (effects.effect()[0] - expected_dydx).abs() < 1e-6,
            "actual={}, expected={expected_dydx}",
            effects.effect()[0]
        );
    }

    /// `target="P(uncensored)"`（左打ち切りのみ）を、`prob_uncensored_closed_form`を
    /// `mu`について数値微分した`dprob_uncensored_dmu_closed_form`×`βⱼ`という
    /// （`target_w_and_s`の`ProbUncensored`分岐とは独立の）式で再計算し、effect・標準誤差
    /// 双方を突き合わせる。
    #[test]
    fn fit_marginal_effects_prob_uncensored_overall_matches_independently_recomputed_dydx_and_delta_method_se()
     {
        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 2;

        let effects = estimator
            .marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ProbUncensored,
                0.95,
            )
            .unwrap();

        let dydx_ame = |params: &[f64], sigma: f64| -> f64 {
            (0..n)
                .map(|i| {
                    let mu: f64 = params[0] + params[1] * (*x.get(i, 1));
                    dprob_uncensored_dmu_closed_form(mu, sigma, lower, upper) * params[1]
                })
                .sum::<f64>()
                / (n as f64)
        };
        let params = estimator.params();
        let sigma = estimator.sigma();
        assert!(
            (effects.effect()[0] - dydx_ame(params, sigma)).abs() < 1e-9,
            "actual={}, expected={}",
            effects.effect()[0],
            dydx_ame(params, sigma)
        );

        let h_param = 1e-5;
        let mut jac_row = vec![0.0; k + 1];
        for m in 0..k {
            let mut plus_p = params.to_vec();
            plus_p[m] += h_param;
            let mut minus_p = params.to_vec();
            minus_p[m] -= h_param;
            jac_row[m] = (dydx_ame(&plus_p, sigma) - dydx_ame(&minus_p, sigma)) / (2.0 * h_param);
        }
        jac_row[k] = (dydx_ame(params, sigma + h_param) - dydx_ame(params, sigma - h_param))
            / (2.0 * h_param);

        let mut var = 0.0;
        for a in 0..=k {
            for b in 0..=k {
                var += jac_row[a] * (*estimator.cov_params().get(a, b)) * jac_row[b];
            }
        }
        let expected_se = var.sqrt();
        assert!(
            (effects.std_errors()[0] - expected_se).abs() < 1e-4,
            "actual={}, expected={expected_se}",
            effects.std_errors()[0]
        );
    }

    /// `target="P(uncensored)"`は右打ち切りのみだと符号が反転する（`x`の増加が上側の
    /// 打ち切り確率を高めるため非打ち切り確率は減少する。`target_w_and_s`のdocコメント
    /// 「数式」参照）。この符号反転が正しく実装されていることを、右打ち切りのみの
    /// データで独立再計算したdydxと突き合わせて確認する。
    #[test]
    fn fit_marginal_effects_prob_uncensored_matches_independent_recomputation_for_right_only_censoring()
     {
        let estimator = TobitEstimator::fit(
            right_censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        let x = estimator.input().x();
        let n = x.nrows();
        let params = estimator.params();
        let sigma = estimator.sigma();

        let effects = estimator
            .marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ProbUncensored,
                0.95,
            )
            .unwrap();

        let expected_dydx: f64 = (0..n)
            .map(|i| {
                let mu: f64 = params[0] + params[1] * (*x.get(i, 1));
                dprob_uncensored_dmu_closed_form(mu, sigma, lower, upper) * params[1]
            })
            .sum::<f64>()
            / (n as f64);

        assert!(
            (effects.effect()[0] - expected_dydx).abs() < 1e-9,
            "actual={}, expected={expected_dydx}",
            effects.effect()[0]
        );
        // x1の真の係数は正（+2相当）なので、右打ち切りのみでは非打ち切り確率の限界効果は
        // 負になるはず（xが増えるほど上側で打ち切られやすくなるため）。
        assert!(effects.effect()[0] < 0.0, "effect={}", effects.effect()[0]);
    }

    /// `target="P(uncensored)"`が両側打ち切りのデータでも正しく計算できることを検証する。
    #[test]
    fn fit_marginal_effects_prob_uncensored_matches_independent_recomputation_for_two_sided_censoring()
     {
        let estimator = TobitEstimator::fit(
            two_sided_censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        let x = estimator.input().x();
        let n = x.nrows();
        let params = estimator.params();
        let sigma = estimator.sigma();

        let effects = estimator
            .marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ProbUncensored,
                0.95,
            )
            .unwrap();

        let expected_dydx: f64 = (0..n)
            .map(|i| {
                let mu: f64 = params[0] + params[1] * (*x.get(i, 1));
                dprob_uncensored_dmu_closed_form(mu, sigma, lower, upper) * params[1]
            })
            .sum::<f64>()
            / (n as f64);

        assert!(
            (effects.effect()[0] - expected_dydx).abs() < 1e-9,
            "actual={}, expected={expected_dydx}",
            effects.effect()[0]
        );
    }

    /// `predict(ExpectedLatent)`は`E[y*|x]=x'β`という自明な式（打ち切りに依存しない）
    /// なので、`x.get(i,·)`から直接計算した線形予測子と厳密に一致するはず。
    #[test]
    fn fit_predict_expected_latent_equals_linear_predictor() {
        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 2;
        let params = estimator.params();

        let predicted = estimator
            .predict(MarginalEffectsTarget::ExpectedLatent)
            .unwrap();
        assert_eq!(predicted.len(), n);
        for (i, &predicted_i) in predicted.iter().enumerate() {
            let expected: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            assert!((predicted_i - expected).abs() < 1e-12);
        }
    }

    /// `predict(ExpectedObserved)`・`predict(ProbUncensored)`を、`marginal_effects`の
    /// テストで使った閉形式（`expected_observed_closed_form`/`prob_uncensored_closed_form`、
    /// `target_w_and_s`/`predicted_value`の実装コードとは独立の`(lower,upper)`ごとの
    /// `match`分岐）と観測ごとに突き合わせる（左打ち切りのみ）。
    #[test]
    fn fit_predict_matches_independent_recomputation_for_left_only_censoring() {
        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 2;
        let params = estimator.params();
        let sigma = estimator.sigma();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        assert_eq!(lower, Some(0.0));
        assert_eq!(upper, None);

        let predicted_observed = estimator
            .predict(MarginalEffectsTarget::ExpectedObserved)
            .unwrap();
        let predicted_prob = estimator
            .predict(MarginalEffectsTarget::ProbUncensored)
            .unwrap();
        for i in 0..n {
            let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            let expected_observed = expected_observed_closed_form(mu, sigma, lower, upper);
            let expected_prob = prob_uncensored_closed_form(mu, sigma, lower, upper);
            assert!(
                (predicted_observed[i] - expected_observed).abs() < 1e-9,
                "row={i}, actual={}, expected={expected_observed}",
                predicted_observed[i]
            );
            assert!(
                (predicted_prob[i] - expected_prob).abs() < 1e-9,
                "row={i}, actual={}, expected={expected_prob}",
                predicted_prob[i]
            );
        }
    }

    /// 右打ち切りのみのデータでも`predict(ExpectedObserved)`・`predict(ProbUncensored)`が
    /// 独立再計算と一致することを確認する。
    #[test]
    fn fit_predict_matches_independent_recomputation_for_right_only_censoring() {
        let estimator = TobitEstimator::fit(
            right_censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 2;
        let params = estimator.params();
        let sigma = estimator.sigma();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        assert_eq!(lower, None);
        assert_eq!(upper, Some(6.0));

        let predicted_observed = estimator
            .predict(MarginalEffectsTarget::ExpectedObserved)
            .unwrap();
        let predicted_prob = estimator
            .predict(MarginalEffectsTarget::ProbUncensored)
            .unwrap();
        for i in 0..n {
            let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            let expected_observed = expected_observed_closed_form(mu, sigma, lower, upper);
            let expected_prob = prob_uncensored_closed_form(mu, sigma, lower, upper);
            assert!((predicted_observed[i] - expected_observed).abs() < 1e-9);
            assert!((predicted_prob[i] - expected_prob).abs() < 1e-9);
        }
    }

    /// 両側打ち切りのデータでも`predict(ExpectedObserved)`・`predict(ProbUncensored)`が
    /// 独立再計算と一致することを確認する。
    #[test]
    fn fit_predict_matches_independent_recomputation_for_two_sided_censoring() {
        let estimator = TobitEstimator::fit(
            two_sided_censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 2;
        let params = estimator.params();
        let sigma = estimator.sigma();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();
        assert_eq!(lower, Some(0.0));
        assert_eq!(upper, Some(6.0));

        let predicted_observed = estimator
            .predict(MarginalEffectsTarget::ExpectedObserved)
            .unwrap();
        let predicted_prob = estimator
            .predict(MarginalEffectsTarget::ProbUncensored)
            .unwrap();
        for i in 0..n {
            let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            let expected_observed = expected_observed_closed_form(mu, sigma, lower, upper);
            let expected_prob = prob_uncensored_closed_form(mu, sigma, lower, upper);
            assert!((predicted_observed[i] - expected_observed).abs() < 1e-9);
            assert!((predicted_prob[i] - expected_prob).abs() < 1e-9);
        }
    }

    /// `censoring_fit_check`（左打ち切りのみ）: `lower`カテゴリが`Some`・`upper`カテゴリが
    /// `None`であること、実測比率（`y==0.0`の観測数/n）・モデル含意の平均確率
    /// （`Φ((lower-μᵢ)/σ)`の平均、`predicted_value`とは独立に`Normal::cdf`を直接呼んで
    /// 計算）の両方を独立再計算と突き合わせる。3カテゴリの`model_implied_rate`が合計1に
    /// なる恒等式も確認する。
    #[test]
    fn fit_censoring_fit_check_matches_independent_recomputation_for_left_only_censoring() {
        let estimator = TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 2;
        let params = estimator.params();
        let sigma = estimator.sigma();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let check = estimator.censoring_fit_check();
        assert!(check.upper().is_none());
        let lower_check = check.lower().unwrap();

        // censored_regression_input: y=[0,0,1.15,2.9,5.2,6.85,9.1,10.95]、lower=0.0で
        // 打ち切られているのはx=1,2の2件（実測打ち切り率2/8=0.25）
        assert!((lower_check.observed_rate() - 0.25).abs() < 1e-12);
        assert!((check.uncensored().observed_rate() - 0.75).abs() < 1e-12);

        let expected_lower_rate: f64 = (0..n)
            .map(|i| {
                let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
                normal.cdf((0.0 - mu) / sigma)
            })
            .sum::<f64>()
            / (n as f64);
        assert!((lower_check.model_implied_rate() - expected_lower_rate).abs() < 1e-9);

        let expected_uncensored_rate: f64 = (0..n)
            .map(|i| {
                let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
                prob_uncensored_closed_form(mu, sigma, Some(0.0), None)
            })
            .sum::<f64>()
            / (n as f64);
        assert!((check.uncensored().model_implied_rate() - expected_uncensored_rate).abs() < 1e-9);

        // model_implied_rateは合計1になるはず（lower + uncensored、upperはこのケースでは無い）
        assert!(
            (lower_check.model_implied_rate() + check.uncensored().model_implied_rate() - 1.0)
                .abs()
                < 1e-9
        );
    }

    /// `censoring_fit_check`（右打ち切りのみ）: `lower`が`None`・`upper`が`Some`になること、
    /// 実測比率・モデル含意の平均確率を独立再計算と突き合わせる。
    #[test]
    fn fit_censoring_fit_check_matches_independent_recomputation_for_right_only_censoring() {
        let estimator = TobitEstimator::fit(
            right_censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 2;
        let params = estimator.params();
        let sigma = estimator.sigma();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let check = estimator.censoring_fit_check();
        assert!(check.lower().is_none());
        let upper_check = check.upper().unwrap();

        // right_censored_regression_input: y=[-2.85,-0.95,1.15,2.9,5.2,6.0,6.0,6.0]、
        // upper=6.0で打ち切られているのはx=6,7,8の3件（実測打ち切り率3/8=0.375）
        assert!((upper_check.observed_rate() - 0.375).abs() < 1e-12);
        assert!((check.uncensored().observed_rate() - 0.625).abs() < 1e-12);

        let expected_upper_rate: f64 = (0..n)
            .map(|i| {
                let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
                1.0 - normal.cdf((6.0 - mu) / sigma)
            })
            .sum::<f64>()
            / (n as f64);
        assert!((upper_check.model_implied_rate() - expected_upper_rate).abs() < 1e-9);
    }

    /// `censoring_fit_check`（両側打ち切り）: `lower`・`upper`双方が`Some`になり、
    /// 3カテゴリの実測比率・モデル含意の平均確率いずれも独立再計算と一致し、かつ
    /// それぞれ合計1になることを確認する。
    #[test]
    fn fit_censoring_fit_check_matches_independent_recomputation_for_two_sided_censoring() {
        let estimator = TobitEstimator::fit(
            two_sided_censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 2;
        let params = estimator.params();
        let sigma = estimator.sigma();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let check = estimator.censoring_fit_check();
        let lower_check = check.lower().unwrap();
        let upper_check = check.upper().unwrap();

        // two_sided_censored_regression_input: y=[0,0,1.15,2.9,5.2,6.0,6.0,6.0]、
        // lower=0.0でx=1,2（2件）・upper=6.0でx=6,7,8（3件）が打ち切り
        assert!((lower_check.observed_rate() - 0.25).abs() < 1e-12);
        assert!((upper_check.observed_rate() - 0.375).abs() < 1e-12);
        assert!((check.uncensored().observed_rate() - 0.375).abs() < 1e-12);
        assert!(
            (lower_check.observed_rate()
                + check.uncensored().observed_rate()
                + upper_check.observed_rate()
                - 1.0)
                .abs()
                < 1e-12
        );

        let expected_lower_rate: f64 = (0..n)
            .map(|i| {
                let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
                normal.cdf((0.0 - mu) / sigma)
            })
            .sum::<f64>()
            / (n as f64);
        let expected_upper_rate: f64 = (0..n)
            .map(|i| {
                let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
                1.0 - normal.cdf((6.0 - mu) / sigma)
            })
            .sum::<f64>()
            / (n as f64);
        let expected_uncensored_rate: f64 = (0..n)
            .map(|i| {
                let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
                prob_uncensored_closed_form(mu, sigma, Some(0.0), Some(6.0))
            })
            .sum::<f64>()
            / (n as f64);
        assert!((lower_check.model_implied_rate() - expected_lower_rate).abs() < 1e-9);
        assert!((upper_check.model_implied_rate() - expected_upper_rate).abs() < 1e-9);
        assert!((check.uncensored().model_implied_rate() - expected_uncensored_rate).abs() < 1e-9);
        assert!(
            (lower_check.model_implied_rate()
                + check.uncensored().model_implied_rate()
                + upper_check.model_implied_rate()
                - 1.0)
                .abs()
                < 1e-9
        );
    }

    /// 上記の`predict`テストは全て単一の傾き係数（`k=2`、切片+`x1`）のフィクスチャのみを
    /// 使っており、`(0..k).map(|j| *x.get(i,j)*self.params[j])`という列インデックス計算の
    /// 配線ミス（列の取り違え等）を構造的に検出できない（`multivariate_censored_input`の
    /// docコメント「配線ミスは単一変数データでは検出できない」と同じ理由）。
    /// 傾き係数2個（`x1`・`x2`）の多変量データセットで
    /// `expected_observed_closed_form`/`prob_uncensored_closed_form`との独立再計算を検証する。
    #[test]
    fn fit_predict_matches_independent_recomputation_for_multivariate_design() {
        let estimator = TobitEstimator::fit(
            multivariate_censored_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 3;
        let params = estimator.params();
        let sigma = estimator.sigma();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();

        let predicted_latent = estimator
            .predict(MarginalEffectsTarget::ExpectedLatent)
            .unwrap();
        let predicted_observed = estimator
            .predict(MarginalEffectsTarget::ExpectedObserved)
            .unwrap();
        let predicted_prob = estimator
            .predict(MarginalEffectsTarget::ProbUncensored)
            .unwrap();
        for i in 0..n {
            let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            assert!((predicted_latent[i] - mu).abs() < 1e-12);
            let expected_observed = expected_observed_closed_form(mu, sigma, lower, upper);
            let expected_prob = prob_uncensored_closed_form(mu, sigma, lower, upper);
            assert!((predicted_observed[i] - expected_observed).abs() < 1e-9);
            assert!((predicted_prob[i] - expected_prob).abs() < 1e-9);
        }
    }

    /// `predict_new_data`（out-of-sample）が独立に再計算した
    /// 値と一致すること。`fit_predict_matches_independent_recomputation_for_
    /// multivariate_design`と同じモデルを使い、学習データとは異なる新規のx値で
    /// 3つの`target`すべてを検証する。
    #[test]
    fn predict_new_data_matches_independent_recomputation_for_multivariate_design() {
        let estimator = TobitEstimator::fit(
            multivariate_censored_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let params = estimator.params().to_vec();
        let sigma = estimator.sigma();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();

        // 学習データには無い新規のx値
        let new_x_columns = vec![vec![15.0, 25.0], vec![3.0, -1.0]];
        let predicted_latent = estimator
            .predict_new_data(MarginalEffectsTarget::ExpectedLatent, &new_x_columns)
            .unwrap();
        let predicted_observed = estimator
            .predict_new_data(MarginalEffectsTarget::ExpectedObserved, &new_x_columns)
            .unwrap();
        let predicted_prob = estimator
            .predict_new_data(MarginalEffectsTarget::ProbUncensored, &new_x_columns)
            .unwrap();

        assert_eq!(predicted_latent.len(), 2);
        for i in 0..2 {
            let mu = params[0] + new_x_columns[0][i] * params[1] + new_x_columns[1][i] * params[2];
            assert!((predicted_latent[i] - mu).abs() < 1e-12);
            let expected_observed = expected_observed_closed_form(mu, sigma, lower, upper);
            let expected_prob = prob_uncensored_closed_form(mu, sigma, lower, upper);
            assert!((predicted_observed[i] - expected_observed).abs() < 1e-9);
            assert!((predicted_prob[i] - expected_prob).abs() < 1e-9);
        }
    }

    /// `predict_new_data`が`has_intercept=false`でも3つの`target`すべてで
    /// 正しく動作すること（Logit/Probitの対応するテストと同じ理由の対称性確保）。
    #[test]
    fn predict_new_data_without_intercept_matches_independent_recomputation() {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x2 = vec![1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0];
        let y = vec![0.0, 0.0, 1.15, 2.9, 5.2, 6.85, 9.1, 10.95];
        let input = TobitInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            false,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap();

        let estimator = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let params = estimator.params().to_vec();
        let sigma = estimator.sigma();
        let lower = estimator.input().lower();
        let upper = estimator.input().upper();

        let new_x_columns = vec![vec![15.0, 25.0], vec![3.0, -1.0]];
        let predicted_latent = estimator
            .predict_new_data(MarginalEffectsTarget::ExpectedLatent, &new_x_columns)
            .unwrap();
        let predicted_observed = estimator
            .predict_new_data(MarginalEffectsTarget::ExpectedObserved, &new_x_columns)
            .unwrap();
        let predicted_prob = estimator
            .predict_new_data(MarginalEffectsTarget::ProbUncensored, &new_x_columns)
            .unwrap();

        assert_eq!(predicted_observed.len(), 2);
        for i in 0..2 {
            let mu = new_x_columns[0][i] * params[0] + new_x_columns[1][i] * params[1];
            assert!((predicted_latent[i] - mu).abs() < 1e-12);
            let expected_observed = expected_observed_closed_form(mu, sigma, lower, upper);
            let expected_prob = prob_uncensored_closed_form(mu, sigma, lower, upper);
            assert!((predicted_observed[i] - expected_observed).abs() < 1e-9);
            assert!((predicted_prob[i] - expected_prob).abs() < 1e-9);
        }
    }

    /// `censoring_fit_check`版の多変量配線ミス検出テスト（上記`fit_predict_matches_
    /// independent_recomputation_for_multivariate_design`と同じ理由）。
    #[test]
    fn fit_censoring_fit_check_matches_independent_recomputation_for_multivariate_design() {
        let estimator = TobitEstimator::fit(
            multivariate_censored_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap();
        let x = estimator.input().x();
        let n = x.nrows();
        let k = 3;
        let params = estimator.params();
        let sigma = estimator.sigma();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let check = estimator.censoring_fit_check();
        assert!(check.upper().is_none());
        let lower_check = check.lower().unwrap();

        // multivariate_censored_input: y=[0,0,1.15,2.9,5.2,6.85,9.1,10.95]、lower=0.0で
        // 打ち切られているのはx1=1,2の2件（実測打ち切り率2/8=0.25、censored_regression_
        // inputと同じyだがx2列が追加された多変量設計行列）
        assert!((lower_check.observed_rate() - 0.25).abs() < 1e-12);
        assert!((check.uncensored().observed_rate() - 0.75).abs() < 1e-12);

        let expected_lower_rate: f64 = (0..n)
            .map(|i| {
                let mu: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
                normal.cdf((0.0 - mu) / sigma)
            })
            .sum::<f64>()
            / (n as f64);
        assert!((lower_check.model_implied_rate() - expected_lower_rate).abs() < 1e-9);
    }

    // ── Fitted（pickleからの復元に使う最小の状態） ─────────────────────

    fn fitted_test_estimator() -> TobitEstimator {
        TobitEstimator::fit(
            censored_regression_input(),
            MleFitOptions {
                solver: SolverType::Newton,
                max_iter: 100,
                tol: 1e-8,
                raise_on_non_convergence: true,
                cov_type: CovType::Classical,
                confidence_level: 0.95,
            },
        )
        .unwrap()
    }

    /// `original`と同じ設計行列を、本番の復元と同じ`from_columns`で作り直す。
    fn rebuilt_input(original: &TobitFitted) -> TobitInput {
        let input = original.input();
        let start = usize::from(input.has_intercept());
        let n = input.nobs();
        let x_columns: Vec<Vec<f64>> = (start..input.k())
            .map(|j| (0..n).map(|i| *input.x().get(i, j)).collect())
            .collect();
        let y: Vec<f64> = (0..n).map(|i| *input.y().get(i, 0)).collect();
        TobitInput::from_columns(
            &y,
            &x_columns,
            input.param_names()[start..].to_vec(),
            input.has_intercept(),
            input.dep_var_name().to_string(),
            input.lower(),
            input.upper(),
        )
        .unwrap()
    }

    #[test]
    fn fitted_rebuilt_from_parts_reproduces_predict_marginal_effects_and_censoring_check() {
        let estimator = fitted_test_estimator();
        let targets = [
            MarginalEffectsTarget::ExpectedLatent,
            MarginalEffectsTarget::ExpectedObserved,
        ];
        let expected_predict: Vec<_> = targets
            .iter()
            .map(|&t| estimator.predict(t).unwrap())
            .collect();
        let expected_effects = estimator
            .marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ExpectedObserved,
                0.9,
            )
            .unwrap();
        let expected_check = estimator.censoring_fit_check();
        let original = estimator.into_fitted();
        let rebuilt = TobitFitted::from_parts(
            rebuilt_input(&original),
            original.params().to_vec(),
            original.sigma(),
            original.cov_params().clone(),
        )
        .unwrap();

        for (target, expected) in targets.iter().zip(&expected_predict) {
            assert_eq!(&rebuilt.predict(*target).unwrap(), expected);
        }
        let effects = rebuilt
            .marginal_effects(
                MarginalEffectsAt::Overall,
                MarginalEffectsTarget::ExpectedObserved,
                0.9,
            )
            .unwrap();
        assert_eq!(effects.effect(), expected_effects.effect());
        assert_eq!(effects.std_errors(), expected_effects.std_errors());
        let check = rebuilt.censoring_fit_check();
        assert_eq!(
            check.uncensored().observed_rate(),
            expected_check.uncensored().observed_rate()
        );
        assert_eq!(
            check.uncensored().model_implied_rate(),
            expected_check.uncensored().model_implied_rate()
        );
    }

    #[test]
    fn from_parts_rejects_parts_of_the_wrong_size_or_a_bad_sigma() {
        let original = fitted_test_estimator().into_fitted();
        let k = original.params().len();
        let cov = || original.cov_params().clone();
        let make = |params: Vec<f64>, sigma: f64, cov: Mat<f64>| {
            TobitFitted::from_parts(rebuilt_input(&original), params, sigma, cov)
        };
        let params = || original.params().to_vec();
        let sigma = original.sigma();
        assert!(make(params(), sigma, cov()).is_ok());
        assert!(make(vec![0.0; k + 1], sigma, cov()).is_err());
        // `cov_params`は`(k+1)×(k+1)`（`σ`の行・列を含む）
        assert!(make(params(), sigma, Mat::zeros(k, k)).is_err());
        assert!(make(params(), sigma, Mat::zeros(k + 1, k + 2)).is_err());
        assert!(make(params(), 0.0, cov()).is_err());
        assert!(make(params(), -1.0, cov()).is_err());
        assert!(make(params(), f64::NAN, cov()).is_err());
    }
}
