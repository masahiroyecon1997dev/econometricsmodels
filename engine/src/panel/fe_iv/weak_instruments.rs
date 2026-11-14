//! FEIVの弱操作変数診断。部分F統計量（`fit()`時）と、Cragg-Donald・Kleibergen-Paap・有効F（推定後のメソッド）。
//!
//! 計算は行列を受け取る共有関数（`shared::instrumental_variables::partial_f_statistic`・
//! `shared::weak_instruments::cragg_donald`・`shared::kleibergen_paap`・`shared::effective_f`）で、ここは
//! within変換した列（第一段階の設計行列`[x̃_exog, z̃]`と各第一段階の`x̃_endog[j]`）を渡して呼ぶ層。
//! 吸収したパラメータ数`n_absorbed`は診断・`cov_type`ごとに違う（`fe_iv.rs`のモジュールdoc「自由度」）:
//!
//! - 部分F・Cragg-Donald（等分散前提で`cov_type`に依存しない）: `neffects`（全ての効果）。
//! - Kleibergen-Paap・有効Fの`cov_type`別の共分散:
//!   - `Classical`・`Hc1`・`Hc2`・`Hc3`・`Dk`: `neffects`。`Hc2`・`Hc3`は吸収した効果のレバレッジ
//!     （`leverage_full`の、within変換後の素のレバレッジを0にしたもの。ダミーを回帰子に入れたときと同じ）を足す。
//!   - `Cluster`: クラスターに入れ子の次元を除いたfixestの`K.fixef="nonnested"`の`K`から一次段階の傾きの数
//!     `l`を引いた数（構造式の`fe_cluster_k_correction`と同じ規則）。
//!
//! Stock-Yogoの臨界値は、吸収自由度を統計量の自由度に反映すればそのまま使える（within変換後のデータで
//! 帰無分布が変わらない）。

use faer::Mat;

use super::FeIvEstimator;
use crate::panel::common::{
    PanelError, absorbed_parameters, fe_cluster_k_correction, leverage_full,
};
use crate::panel::fe::{FeCovType, FeEstimator};
use crate::shared::covariance::HcVariant;
use crate::shared::effective_f::{EffectiveF, effective_f};
use crate::shared::error::CommonError;
use crate::shared::group_codes::GroupCodes;
use crate::shared::instrumental_variables::partial_f_statistic;
use crate::shared::kleibergen_paap::{KleibergenPaap, kleibergen_paap};
use crate::shared::score_covariance::ScoreCovariance;
use crate::shared::weak_instruments::{CraggDonald, cragg_donald};

/// エラーの文に出す、診断統計量の名前。
const CRAGG_DONALD_LABEL: &str = "Cragg-Donald";
const KLEIBERGEN_PAAP_LABEL: &str = "Kleibergen-Paap";
const EFFECTIVE_F_LABEL: &str = "effective F";

fn undefined(statistic: &'static str) -> PanelError {
    PanelError::WeakInstrumentStatisticUndefined { statistic }
}

/// 第一段階`stage`（`x̃_endog[j] ~ x̃_exog + z̃`）の弱操作変数の部分F統計量。除外操作変数`q`個の係数が
/// ゼロという制限を、等分散前提のネストF検定で調べる。分母の自由度は`n − l − neffects`。
///
/// # Errors
/// `n <= l + neffects`は`fit()`が先に弾くので到達しない。制限モデルの設計行列は制限なしモデルの列の
/// 部分集合なので、特異でもない（`CommonError::Internal`も到達しない）。
pub(super) fn first_stage_partial_f(
    stage: &FeEstimator,
    k_exog: usize,
    k_instruments: usize,
    n_absorbed: usize,
) -> Result<f64, CommonError> {
    let within = stage.within_input();
    let x_exog = within.x().subcols(0, k_exog).to_owned();
    let ssr: f64 = (0..within.nobs())
        .map(|i| stage.residuals().get(i, 0).powi(2))
        .sum();
    partial_f_statistic(
        within.y(),
        &x_exog,
        ssr,
        k_exog + k_instruments,
        k_instruments,
        n_absorbed,
    )
}

/// 診断に渡す行列。第一段階の設計行列から取り出す（`[x̃_exog, z̃]`は全ての第一段階で同じ）。
struct Matrices {
    y_endog: Mat<f64>,
    x_exog: Mat<f64>,
    z: Mat<f64>,
}

impl FeIvEstimator {
    fn matrices(&self, statistic: &'static str) -> Result<Matrices, PanelError> {
        // 第一段階が無い（内生変数が無い）と、診断は定義できない。
        let Some(first) = self.first_stages.first() else {
            return Err(undefined(statistic));
        };
        let (k_exog, k_instruments) = (self.input.k_exog(), self.input.k_instruments());
        let x = first.within_input().x();
        let n = x.nrows();
        Ok(Matrices {
            y_endog: Mat::from_fn(n, self.first_stages.len(), |i, j| {
                *self.first_stages[j].within_input().y().get(i, 0)
            }),
            x_exog: x.subcols(0, k_exog).to_owned(),
            z: x.subcols(k_exog, k_instruments).to_owned(),
        })
    }

    /// 吸収した効果の数`neffects`。
    fn neffects(&self) -> usize {
        absorbed_parameters(&self.n_groups)
    }

    /// 部分F統計量のF分布の自由度（[`FeIvEstimator::weak_instrument_f_df`]の本体）。`n > l + neffects`は
    /// `fit`の第一段階の自由度の検査が保証する。
    pub(super) fn compute_weak_instrument_f_df(&self) -> Option<(usize, usize)> {
        if self.weak_instrument_f.is_empty() {
            return None;
        }
        let l = self.input.k_exog() + self.input.k_instruments();
        Some((
            self.input.k_instruments(),
            self.input.nobs() - l - self.neffects(),
        ))
    }

    /// `cov_type`に対応する一次段階のスコアの共分散と、そのときの吸収したパラメータ数を作って`body`に渡す
    /// （`body`が`None`を返したら`Ok(None)`）。
    ///
    /// # Errors
    /// `Hc2`・`Hc3`のダミーのレバレッジの計算の失敗（`fit`が同じ計算を済ませているので、通常は到達しない）。
    fn with_score_covariance<T>(
        &self,
        body: impl FnOnce(ScoreCovariance<'_>, usize) -> Option<T>,
    ) -> Result<Option<T>, PanelError> {
        let groups = self.input.index().effect_codes();
        let levels = self.effects.levels(groups)?;
        let n = self.input.nobs();
        let l = self.input.k_exog() + self.input.k_instruments();
        let neffects = self.neffects();
        Ok(match &self.cov_type {
            FeCovType::Classical => body(ScoreCovariance::Classical, neffects),
            FeCovType::Hc1 => body(ScoreCovariance::Hc(HcVariant::Hc1), neffects),
            hc @ (FeCovType::Hc2 | FeCovType::Hc3) => {
                let variant = match hc {
                    FeCovType::Hc2 => HcVariant::Hc2,
                    _ => HcVariant::Hc3,
                };
                // 吸収した効果のレバレッジ（within変換後の素のレバレッジは0にして、ダミー分だけ）。
                // `fit`が同じ`levels`で計算済みなので、ここで`LeverageDimensionTooLarge`になることはない。
                let leverage = leverage_full(&vec![0.0; n], &levels, n)?;
                body(
                    ScoreCovariance::HcAbsorbed {
                        variant,
                        leverage: &leverage,
                    },
                    neffects,
                )
            }
            FeCovType::Cluster { groups: explicit } => {
                let cluster: &GroupCodes = explicit.as_ref().unwrap_or(&groups[0]);
                // `K ≥ l + 1`（`fe_cluster_k_correction`の下限）なので`K − l`は引き算で足りる。全ての次元が
                // クラスターに入れ子なら1、どれも入れ子でなければ`neffects`、部分的なら中間。
                let k = fe_cluster_k_correction(&levels, l + neffects, l, cluster);
                debug_assert!(k > l);
                body(ScoreCovariance::Cluster(cluster), k - l)
            }
            FeCovType::Dk { time, .. } => {
                // 構造式のDKと同じ時点・バンド幅（`fit`が解決したもの）。
                let Some(bandwidth) = self.dk_bandwidth_used else {
                    return Err(
                        CommonError::internal("the Driscoll-Kraay bandwidth is missing").into(),
                    );
                };
                body(
                    ScoreCovariance::Dk {
                        time: time.codes(),
                        bandwidth,
                    },
                    neffects,
                )
            }
        })
    }

    /// Cragg-Donald統計量とStock-Yogoの臨界値（[`FeIvEstimator::cragg_donald`]の本体）。
    pub(super) fn compute_cragg_donald(&self) -> Result<CraggDonald, PanelError> {
        crate::shared::parallelism::ensure_serial();
        let m = self.matrices(CRAGG_DONALD_LABEL)?;
        let statistic = cragg_donald(&m.y_endog, &m.x_exog, &m.z, self.neffects())
            .ok_or_else(|| undefined(CRAGG_DONALD_LABEL))?;
        Ok(CraggDonald::new(
            statistic,
            self.input.k_endog(),
            self.input.k_instruments(),
        ))
    }

    /// Kleibergen-Paap統計量（[`FeIvEstimator::kleibergen_paap`]の本体）。
    pub(super) fn compute_kleibergen_paap(&self) -> Result<KleibergenPaap, PanelError> {
        crate::shared::parallelism::ensure_serial();
        let m = self.matrices(KLEIBERGEN_PAAP_LABEL)?;
        let statistics = self
            .with_score_covariance(|covariance, n_absorbed| {
                kleibergen_paap(&m.y_endog, &m.x_exog, &m.z, n_absorbed, covariance)
            })?
            .ok_or_else(|| undefined(KLEIBERGEN_PAAP_LABEL))?;
        if statistics.rk_lm.is_none() && statistics.rk_wald_f.is_none() {
            return Err(undefined(KLEIBERGEN_PAAP_LABEL));
        }
        Ok(KleibergenPaap {
            rk_lm: statistics.rk_lm,
            rk_wald_f: statistics.rk_wald_f,
            n_endog: self.input.k_endog(),
            n_excluded: self.input.k_instruments(),
        })
    }

    /// 有効F統計量と臨界値（[`FeIvEstimator::effective_f`]の本体）。
    pub(super) fn compute_effective_f(&self, alpha: f64) -> Result<EffectiveF, PanelError> {
        crate::shared::parallelism::ensure_serial();
        if self.input.k_endog() != 1 {
            return Err(PanelError::EffectiveFEndogenousCount {
                n_endog: self.input.k_endog(),
            });
        }
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(PanelError::AlphaOutOfRange { alpha });
        }
        let m = self.matrices(EFFECTIVE_F_LABEL)?;
        // within変換した被説明変数: 構造残差`e = ỹ − X̃β̂`から`ỹ = e + X̃β̂`（`X̃ = [x̃_exog, x̃_endog]`）。
        let n = m.x_exog.nrows();
        let k_exog = m.x_exog.ncols();
        let y_tilde = Mat::from_fn(n, 1, |i, _| {
            let fitted: f64 = (0..k_exog)
                .map(|j| m.x_exog.get(i, j) * self.params.get(j, 0))
                .sum::<f64>()
                + m.y_endog.get(i, 0) * self.params.get(k_exog, 0);
            self.residuals.get(i, 0) + fitted
        });
        self.with_score_covariance(|covariance, n_absorbed| {
            effective_f(
                &m.y_endog,
                &m.x_exog,
                &m.z,
                n_absorbed,
                covariance,
                alpha,
                Some(&y_tilde),
            )
        })?
        .ok_or_else(|| undefined(EFFECTIVE_F_LABEL))
    }
}
