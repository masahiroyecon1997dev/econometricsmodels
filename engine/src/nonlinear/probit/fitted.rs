//! `ProbitFitted`（予測・的中表・限界効果に必要な最小の状態）と、その復元用の`from_parts`。
//!
//! テストは`fitted/tests.rs`と`fitted/tests/`にテーマ別で置く。

use super::input::ProbitInput;
use super::model::{at_point_w_and_s, overall_w_and_s};
use crate::nonlinear::common::{
    FittedModelForMarginalEffects, MarginalEffects, MarginalEffectsAt, MleError, column_means,
    column_medians, marginal_effects_from_w_s, pred_table, predict_from_link, predict_new_data,
};
use crate::shared::error::CommonError;
use faer::Mat;
use statrs::distribution::{ContinuousCDF, Normal};

/// Probitの推定結果のうち、予測・的中表・限界効果に必要な最小の状態（入力データ・係数・係数の分散共分散行列）。
///
/// [`super::ProbitEstimator::into_fitted`]で取り出す。`engine_pybind`の結果オブジェクトは推定量全体ではなくこれを
/// 保持し、pickleでは[`ProbitFitted::from_parts`]で復元する。フィールドはprivate（`.claude/rules/rust-style.md`
/// 「推定量構造体の設計」参照）で、構築経路は`fit`と`from_parts`だけ。
#[derive(Debug)]
pub struct ProbitFitted {
    pub(super) input: ProbitInput,
    /// 係数（元のスケール）。`input.param_names()`と対応する
    pub(super) params: Vec<f64>,
    /// 係数の分散共分散行列（元のスケール、k×k）。限界効果（デルタ法）で使う
    pub(super) cov_params: Mat<f64>,
}

impl ProbitFitted {
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
        input: ProbitInput,
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

    pub fn input(&self) -> &ProbitInput {
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
    /// CI幅を見られるようにする、`LogitEstimator::marginal_effects`と同じ設計）。
    ///
    /// ## 数式（デルタ法）
    ///
    /// `p_i = Φ(x_i'θ)`のとき、変数`j`（連続変数として扱う。`dummy=False`が既定の
    /// statsmodelsの`get_margeff()`に倣い、離散変数の自動判定は行わない設計、
    /// `docs/spec/nonlinear-common.md`6章参照）の限界効果は
    /// `dy/dx_j = φ(x_i'θ)θ_j`（Logitの`p(1-p)θ_j`とは異なり標準正規PDF`φ`を使う。
    /// 参照）。
    ///
    /// - `at="overall"`（AME）: `g_j(θ) = w(θ)*θ_j`、`w(θ) = (1/n)Σᵢ φ(zᵢ)`
    /// - `at="mean"`/`"median"`: `g_j(θ) = w(θ)*θ_j`、`w(θ) = φ(z̄)`
    ///   （`z̄=x̄'θ`、`x̄`は各説明変数の標本平均または中央値からなる代表点）
    ///
    /// いずれも同じ`g_j(θ)=w(θ)*θ_j`という形に帰着するため、`w`とその勾配
    /// `s_m=∂w/∂θ_m`さえ計算できれば、ヤコビアンは
    /// `∂g_j/∂θ_m = θ_j*s_m + [j==m]*w`という共通の式で書ける
    /// （`overall_w_and_s`/`at_point_w_and_s`が`(w,s)`を計算し、`w`・`s`の計算方法に
    /// 依らない残りの計算——ヤコビアン・デルタ法標準誤差・定数項の除外——は
    /// `nonlinear/common/marginal_effects.rs`の`marginal_effects_from_w_s`に共通化されている
    /// （Logitと数式まで同型であることを確認済み）。
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

    /// 予測確率 `p_i = Φ(x_i'θ)` を、`fit()`に使った学習データ（`self.input.x()`）の
    /// 各行について返す（`fit()`のReturn本体には含めない別メソッド、
    /// `docs/spec/nonlinear-common.md`6章。`LogitEstimator::predict`の`Λ`を`Φ`に置き換えた
    /// Probit版）。
    ///
    /// 新規データでの予測（out-of-sample）は`predict_new_data`。
    pub fn predict(&self) -> Vec<f64> {
        let normal = Normal::standard();
        predict_from_link(self.input.x(), &self.params, |z| normal.cdf(z))
    }

    /// 新規データ（out-of-sample、`new_x_columns`）に対する予測確率
    /// `p_i = Φ(x_i'θ)`。`LogitEstimator::predict_new_data`の`Λ`を`Φ`に
    /// 置き換えたProbit版（設計上の理由は同メソッドのdocコメント参照）。
    ///
    /// `new_x_columns`の本数・順序の契約、パニック条件は
    /// `nonlinear::common::predict_new_data`のdocコメント参照。
    pub fn predict_new_data(&self, new_x_columns: &[Vec<f64>]) -> Vec<f64> {
        let normal = Normal::standard();
        predict_new_data(
            &self.params,
            self.input.has_intercept(),
            new_x_columns,
            |z| normal.cdf(z),
        )
    }

    /// 分類の的中表（2×2、`table[actual][predicted]`のカウント。行=実測クラス、
    /// 列=予測クラス）。`predict()`が返す予測確率のみを`threshold`で二値化し、実測`y`は
    /// `threshold`に関わらず常に`0.5`で二値化する。数式・statsmodelsとの整合性の詳細は
    /// `nonlinear/common/predict.rs`の`pred_table`のdocコメント参照（リンク関数に依存しない計算
    /// のため`common/predict.rs`に共通化されている、Logitと共通）。
    ///
    /// **新規データでの的中表（out-of-sample）は未対応**（`predict()`とは異なり
    /// スコープ外。`LogitEstimator::pred_table`と同じ）。
    pub fn pred_table(&self, threshold: f64) -> Mat<f64> {
        pred_table(&self.predict(), self.input.y(), threshold)
    }
}

#[cfg(test)]
mod tests;
