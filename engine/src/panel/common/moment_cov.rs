//! `cov_type`別の、固定効果つきGMMのモーメント条件の分散共分散行列`Ω̂`（l×l）。
//!
//! 係数の共分散行列（サンドイッチ）の中身にあたる部分で、サンドイッチの代数は
//! `shared::gmm::gmm_sandwich_covariance`が行う。`panel_cov_params`の`cov_type`ごとの補正
//! （HC1の`n/df_resid`、HC2/HC3のレバレッジ、クラスターの`K`、DKの`T/(T-1)`）と事前検証を、設計行列の
//! 代わりに変換後の操作変数`Z̃`のスコア`ẽ_it z̃_it`に適用する。検証と`K`の決め方は`panel_cov_params`と
//! 同じ関数（`cluster_setup`・`dk_setup`）を共有する。

use faer::Mat;

use super::cov_params::{driscoll_kraay_scale, panel_driscoll_kraay_meat};
use super::cov_type::PanelCovType;
use super::error::PanelError;
use super::fe_adjustment::leverage_full;
use super::panel_cov::{ClusterSetup, DkSetup, PanelStructure, cluster_setup, dk_setup};
use crate::shared::covariance::{
    HcScale, cluster_correction, cluster_meat, hc_meat, hc_scales, leverages,
};
use crate::shared::error::CommonError;
use crate::shared::gmm::invert_spd;
use crate::shared::group_codes::GroupCodes;

/// [`panel_moment_covariance`]の入力。
pub(crate) struct PanelMomentInput<'a> {
    /// 変換後の操作変数`Z̃`（n, l。外生変数と操作変数）。
    pub(crate) z: &'a Mat<f64>,
    /// `Z̃'Z̃`（l, l）。
    pub(crate) ztz: &'a Mat<f64>,
    /// 構造残差`ẽ = ỹ − X̃β̂`。
    pub(crate) residuals: &'a [f64],
    /// `n − k − neffects`。`Classical`の`σ̂²`とHC1の補正に使う。
    pub(crate) df_resid: usize,
    /// `k + neffects`。DKとクラスターの`K`の既定値。
    pub(crate) df_model: usize,
    /// 検定する傾き係数の数`q`（構造式の係数の数`k`）。クラスター数・時点数が`q`を超えるかの検証に使う。
    pub(crate) n_slopes: usize,
    /// entityのコード。`Cluster { groups: None }`のクラスター変数の既定。
    pub(crate) entity: &'a GroupCodes,
    /// 吸収した効果の次元（`effects`の順。1つ目が`entity`）。HC2・HC3のレバレッジ、クラスターの`K`、
    /// 2グループ退化の検査に使う。
    pub(crate) levels: &'a [&'a GroupCodes],
}

/// [`panel_moment_covariance`]の結果。
pub(crate) struct PanelMoment {
    /// `Ω̂`（l×l）。小標本補正込み。
    pub(crate) omega: Mat<f64>,
    /// `Dk`のとき実際に使われたバンド幅。`Dk`以外では`None`。
    pub(crate) dk_bandwidth_used: Option<usize>,
}

/// `cov_type`に応じたモーメント条件の分散共分散行列`Ω̂`。傾き係数のWald検定をする前提で検証する。
///
/// - `Classical`: `σ̂² Z̃'Z̃`、`σ̂² = ẽ'ẽ/df_resid`（within変換後の`ẽ`は平均0なので中心化は要らない）。
/// - `Hc1`: `Σ (n/df_resid) ẽ² z̃z̃'`。
/// - `Hc2`・`Hc3`: レバレッジ`h = z̃'(Z̃'Z̃)⁻¹z̃`に吸収した効果のレバレッジ（`leverage_full`）を足す
///   （ダミーを操作変数に入れたときのハット行列の対角と一致する）。
/// - `Cluster`: `(G/(G-1))((n-1)/(n-K)) Σ_g (Σ ẽ z̃)(Σ ẽ z̃)'`。
/// - `Dk`: `T/(T-1) × (n-1)/(n-K)` と、時点ごとのスコアの和へのBartlettカーネル和。`K = df_model`。
///
/// # Errors
/// - `Cluster`・`Dk`の事前検証は`panel_cov_params`と同じ
/// - `Z̃'Z̃`が正定値でなく`HC2`・`HC3`のレバレッジが求まらないとき`PanelError::Common`（到達不能のはずの内部エラー）
pub(crate) fn panel_moment_covariance(
    cov_type: &PanelCovType,
    input: &PanelMomentInput<'_>,
) -> Result<PanelMoment, PanelError> {
    let PanelMomentInput {
        z,
        ztz,
        residuals,
        df_resid,
        df_model,
        n_slopes,
        entity,
        levels,
    } = *input;
    let structure = PanelStructure::FixedEffects { levels };
    let (n, l) = (z.nrows(), z.ncols());
    let plain = |omega: Mat<f64>| PanelMoment {
        omega,
        dk_bandwidth_used: None,
    };

    match cov_type {
        PanelCovType::Classical => {
            let ssr: f64 = residuals.iter().map(|r| r * r).sum();
            let sigma2 = ssr / (df_resid as f64);
            Ok(plain(Mat::from_fn(l, l, |i, j| sigma2 * (*ztz.get(i, j)))))
        }
        PanelCovType::Hc1 => {
            let factor = (n as f64 / df_resid as f64).sqrt();
            let scales = hc_scales(n, |i| residuals[i], HcScale::Hc1 { factor });
            Ok(plain(hc_meat(z, &scales)))
        }
        PanelCovType::Hc2 | PanelCovType::Hc3 => {
            // `Z̃'Z̃`は点推定の初期重み`(Z̃'Z̃)⁻¹`で反転成功済み。
            let ztz_inv = invert_spd(
                ztz,
                l,
                CommonError::internal("Z'Z could not be factorized for the HC2/HC3 leverage"),
            )?;
            let h = leverage_full(&leverages(z, &ztz_inv), levels, n)?;
            let scale = if matches!(cov_type, PanelCovType::Hc2) {
                HcScale::Hc2 { leverage: &h }
            } else {
                HcScale::Hc3 { leverage: &h }
            };
            let scales = hc_scales(n, |i| residuals[i], scale);
            Ok(plain(hc_meat(z, &scales)))
        }
        PanelCovType::Cluster { groups } => {
            let ClusterSetup {
                group_codes,
                n_groups,
                k_correction,
            } = cluster_setup(
                groups.as_ref(),
                entity,
                n,
                df_model,
                n_slopes,
                true,
                structure,
            )?;
            let (s_hat, _) = cluster_meat(group_codes, l, |i, a| residuals[i] * (*z.get(i, a)));
            let correction = cluster_correction(n_groups, n, k_correction);
            Ok(plain(Mat::from_fn(l, l, |i, j| {
                correction * (*s_hat.get(i, j))
            })))
        }
        PanelCovType::Dk {
            bandwidth,
            time: dk_time,
        } => {
            let DkSetup {
                time_codes,
                t_periods,
                bandwidth: bw,
            } = dk_setup(dk_time, *bandwidth, n, n_slopes, true, structure)?;
            let s_hat = panel_driscoll_kraay_meat(z, residuals, time_codes, bw);
            let scale = driscoll_kraay_scale(n, t_periods, df_model);
            Ok(PanelMoment {
                omega: Mat::from_fn(l, l, |i, j| scale * (*s_hat.get(i, j))),
                dk_bandwidth_used: Some(bw),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panel::common::TimeKeys;

    /// 6行・2列の`Z̃`と、2個体（3行ずつ）。
    fn fixture() -> (Mat<f64>, Mat<f64>, Vec<f64>, GroupCodes) {
        let z = Mat::from_fn(6, 2, |i, j| ((i + 1) as f64 * (j as f64 + 0.7)).sin());
        let ztz = z.transpose() * &z;
        let residuals = vec![0.3, -0.2, 0.1, -0.4, 0.25, 0.05];
        let entity = GroupCodes::from_labels(&["a", "a", "a", "b", "b", "b"].map(String::from));
        (z, ztz, residuals, entity)
    }

    fn input<'a>(
        z: &'a Mat<f64>,
        ztz: &'a Mat<f64>,
        residuals: &'a [f64],
        entity: &'a GroupCodes,
        n_slopes: usize,
        levels: &'a [&'a GroupCodes],
    ) -> PanelMomentInput<'a> {
        PanelMomentInput {
            z,
            ztz,
            residuals,
            df_resid: 3,
            df_model: 3,
            n_slopes,
            entity,
            levels,
        }
    }

    /// クラスター数が`q`以下、時点数が`q`以下、バンド幅の範囲外はエラー。`Dk`は使ったバンド幅を返す。
    #[test]
    fn cluster_and_dk_are_validated_and_dk_reports_its_bandwidth() {
        let (z, ztz, e, entity) = fixture();
        let levels = [&entity];

        // 個体2つ・`q = 2`は`G <= q`。
        let rejected = panel_moment_covariance(
            &PanelCovType::Cluster { groups: None },
            &input(&z, &ztz, &e, &entity, 2, &levels),
        );
        assert!(matches!(
            rejected,
            Err(PanelError::Common(
                CommonError::InsufficientClustersForInference { g: 2, q: 2 }
            ))
        ));

        let time =
            TimeKeys::lexicographic(["0", "1", "2", "0", "1", "2"].map(String::from).to_vec());
        let dk = |bandwidth| PanelCovType::Dk {
            bandwidth,
            time: time.clone(),
        };
        let ok = panel_moment_covariance(&dk(Some(1)), &input(&z, &ztz, &e, &entity, 1, &levels))
            .unwrap();
        assert_eq!(ok.dk_bandwidth_used, Some(1));
        let bad_bandwidth =
            panel_moment_covariance(&dk(Some(3)), &input(&z, &ztz, &e, &entity, 1, &levels));
        assert_eq!(
            bad_bandwidth.err(),
            Some(PanelError::InvalidDkBandwidth { bandwidth: 3, t: 3 })
        );
    }
}
