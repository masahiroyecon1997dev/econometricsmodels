//! `TobitFitted`（予測・限界効果・打ち切り適合度の確認に必要な最小の状態）と、その復元用の`from_parts`。
//!
//! テストは`fitted/tests.rs`と`fitted/tests/`にテーマ別で置く。

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
    MarginalEffects, MarginalEffectsAt, MleError, column_means, column_medians, linear_predictor,
    predict_new_data,
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
        let lower = self.input.lower();
        let upper = self.input.upper();
        let normal = Normal::standard();
        if target == MarginalEffectsTarget::ExpectedUncensored {
            ensure_uncensored_interval_resolvable(lower, upper, self.sigma)?;
        }
        Ok((0..n)
            .map(|i| {
                let mu = linear_predictor(x, i, &self.params);
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
        let lower = self.input.lower();
        let upper = self.input.upper();
        let normal = Normal::standard();

        let mut observed_lower = 0usize;
        let mut observed_upper = 0usize;
        let mut model_lower_sum = 0.0;
        let mut model_upper_sum = 0.0;
        let mut model_uncensored_sum = 0.0;

        for i in 0..n {
            let mu = linear_predictor(x, i, &self.params);
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
mod tests;
