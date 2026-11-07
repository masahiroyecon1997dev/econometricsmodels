//! 一次段階のスコアの共分散の選び方。弱操作変数の診断(`kleibergen_paap`・`effective_f`)が共有する。
//!
//! スコアは観測ごとのベクトル(行列の行)で、`ScoreCovariance`が等分散・HC0〜HC3・クラスター・HACを選ぶ。
//! meatは`shared::covariance`の関数を使い、HCの行スケールと小標本の係数をここで一か所に定める。
//! 呼び出し側は、`n_params = (除外操作変数の数) + (外生変数の数) + n_absorbed`を自由度の計算に渡す。

use faer::Mat;

use super::covariance::{
    HcScale, HcVariant, cluster_correction, cluster_meat, hac_meat, hc_meat, hc_scales, leverages,
    xtx_inverse,
};
use super::group_codes::GroupCodes;
use super::linear_algebra::unit_columns;

/// 一次段階のスコアの共分散の推定方法。
#[derive(Debug, Clone, Copy)]
pub(crate) enum ScoreCovariance<'a> {
    /// 等分散・無相関(スコアの共分散がクロネッカー積になる)。
    Classical,
    /// 観測ごとのスコアに行スケールを掛けたHC。`Hc1`は`√(n/(n−n_params))`、`Hc2`・`Hc3`は
    /// `[X_exog, Z]`のレバレッジによる([`hc_row_scales`])。
    Hc(HcVariant),
    /// クラスターロバスト。`groups`の長さは行数に等しい。
    Cluster(&'a GroupCodes),
    /// Bartlett核のNewey-West。`order`は`0..n`の置換(時間順)、`lags < n`。
    Hac { lags: usize, order: &'a [usize] },
}

/// `1 − h_ii`がこれ以下の行があるとHC2・HC3のスケールが定義できない。
const UNIT_LEVERAGE_TOLERANCE: f64 = 1e-10;

/// HCの行スケール(スコア全体に掛ける)。`Hc1`は`√(n/(n−n_params))`、`Hc2`・`Hc3`は一次段階の回帰子
/// `[X_exog, Z]`のレバレッジ(`z_tilde`は`X`と直交なので`h(X) + h(Z̃)`)による。
///
/// - `z_tilde`: 外生変数で直交化した除外操作変数(n×q。列のスケールは問わない)、`x_exog`: 外生変数(n×k_exog)。
/// - `Hc1`は自由度`n − n_params`が1未満のとき`None`。
/// - レバレッジは吸収した効果の分を含まないので、`n_absorbed > 0`のHC2・HC3は`None`(個体ダミーを回帰子に
///   入れた定義と食い違うため。吸収した効果のレバレッジを受け取る口ができるまでの扱い)。レバレッジが1の
///   行があるときも`None`。
pub(crate) fn hc_row_scales(
    variant: HcVariant,
    z_tilde: &Mat<f64>,
    x_exog: &Mat<f64>,
    n_params: usize,
    n_absorbed: usize,
) -> Option<Vec<f64>> {
    let n = z_tilde.nrows();
    let one = |_: usize| 1.0;
    match variant {
        HcVariant::Hc0 => Some(hc_scales(n, one, HcScale::Hc0)),
        HcVariant::Hc1 => {
            let df = n.checked_sub(n_params).filter(|&d| d >= 1)?;
            let factor = ((n as f64) / (df as f64)).sqrt();
            Some(hc_scales(n, one, HcScale::Hc1 { factor }))
        }
        HcVariant::Hc2 | HcVariant::Hc3 => {
            if n_absorbed > 0 {
                return None;
            }
            let mut leverage = leverages(z_tilde, &xtx_inverse(z_tilde).ok()?);
            if x_exog.ncols() > 0 {
                let x_unit = unit_columns(x_exog);
                let x_leverage = leverages(&x_unit, &xtx_inverse(&x_unit).ok()?);
                for (h, h_x) in leverage.iter_mut().zip(x_leverage) {
                    *h += h_x;
                }
            }
            if leverage
                .iter()
                .any(|h| h.is_nan() || 1.0 - h <= UNIT_LEVERAGE_TOLERANCE)
            {
                return None;
            }
            let scale = if variant == HcVariant::Hc2 {
                HcScale::Hc2 {
                    leverage: &leverage,
                }
            } else {
                HcScale::Hc3 {
                    leverage: &leverage,
                }
            };
            Some(hc_scales(n, one, scale))
        }
    }
}

/// スコア行列`psi`(n×d)のmeat(d×d。`n`で割る前の和)と、クラスターのときのクラスター数。
///
/// `Classical`は共分散がクロネッカー積になり、スコアの構造に依るので、`classical`が返す行列をそのまま
/// meatとして使う。`Hc`の行スケールは`row_scales`が作る(`hc_row_scales`を呼ぶ側の事情で遅延評価する)。
/// 次のとき`None`: `row_scales`が`None`、クラスターが2未満または行数が合わない、`Hac`の時間順の長さが
/// 行数と違う・`lags >= n`。
pub(crate) fn score_meat(
    psi: &Mat<f64>,
    covariance: ScoreCovariance<'_>,
    row_scales: impl FnOnce(HcVariant) -> Option<Vec<f64>>,
    classical: impl FnOnce() -> Mat<f64>,
) -> Option<(Mat<f64>, Option<usize>)> {
    let (n, d) = (psi.nrows(), psi.ncols());
    match covariance {
        ScoreCovariance::Classical => Some((classical(), None)),
        ScoreCovariance::Hc(variant) => {
            let scales = row_scales(variant)?;
            Some((hc_meat(psi, &scales), None))
        }
        ScoreCovariance::Cluster(groups) => {
            if groups.nobs() != n || groups.n_groups() < 2 {
                return None;
            }
            let (meat, g) = cluster_meat(groups, d, |i, a| *psi.get(i, a));
            Some((meat, Some(g)))
        }
        ScoreCovariance::Hac { lags, order } => {
            if order.len() != n || lags >= n {
                return None;
            }
            Some((hac_meat(n, d, lags, order, |i, a| *psi.get(i, a)), None))
        }
    }
}

/// 共分散行列に掛ける小標本の係数`c`(`Ŵ = c · meat / n`の`c`)。
///
/// - `Classical`・`Hac`: `n/(n − n_params)`
/// - `Hc`: 1(`Hc1`の補正は[`hc_row_scales`]の行スケールに含まれる)
/// - `Cluster`: `G/(G−1) · (n−1)/(n − n_params)`(`covariance::cluster_correction`)。`n_groups`はクラスター数で、
///   `Cluster`のとき`score_meat`が返した値を渡す。
///
/// 自由度`n − n_params`が1未満のとき、または`Cluster`で`n_groups`が無いとき`None`。
pub(crate) fn covariance_multiplier(
    covariance: ScoreCovariance<'_>,
    n: usize,
    n_params: usize,
    n_groups: Option<usize>,
) -> Option<f64> {
    let df = n.checked_sub(n_params).filter(|&d| d >= 1)?;
    match covariance {
        ScoreCovariance::Classical | ScoreCovariance::Hac { .. } => Some((n as f64) / (df as f64)),
        ScoreCovariance::Hc(_) => Some(1.0),
        ScoreCovariance::Cluster(_) => Some(cluster_correction(n_groups?, n, n_params)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(groups: &[usize]) -> GroupCodes {
        let labels: Vec<String> = groups.iter().map(|g| format!("g{g}")).collect();
        GroupCodes::from_labels_without_keys(&labels)
    }

    fn column(values: &[f64]) -> Mat<f64> {
        Mat::from_fn(values.len(), 1, |i, _| values[i])
    }

    #[test]
    fn hc1_scale_is_the_root_of_the_degrees_of_freedom_ratio() {
        let z = column(&[1.0, -2.0, 0.5, 3.0, -1.0, 0.2]);
        let x = Mat::from_fn(6, 1, |i, _| if i < 3 { 1.0 } else { -1.0 });
        let scales = hc_row_scales(HcVariant::Hc1, &z, &x, 4, 0).unwrap();
        // n = 6, n_params = 4
        assert!(
            scales
                .iter()
                .all(|s| (s - (6.0f64 / 2.0).sqrt()).abs() < 1e-12)
        );
        assert!(hc_row_scales(HcVariant::Hc1, &z, &x, 6, 0).is_none());
    }

    #[test]
    fn hc2_and_hc3_use_the_leverage_of_the_regressors() {
        // z, xが直交(平均ゼロの列と定数列)なら、h = 1/n + z_i²/Σz²。
        let z = column(&[1.0, -1.0, 2.0, -2.0, 0.5, -0.5]);
        let x = Mat::from_fn(6, 1, |_, _| 1.0);
        let szz: f64 = [1.0f64, 1.0, 4.0, 4.0, 0.25, 0.25].iter().sum();
        let zi = [1.0f64, -1.0, 2.0, -2.0, 0.5, -0.5];
        let hc2 = hc_row_scales(HcVariant::Hc2, &z, &x, 2, 0).unwrap();
        let hc3 = hc_row_scales(HcVariant::Hc3, &z, &x, 2, 0).unwrap();
        for i in 0..6 {
            let h = 1.0 / 6.0 + zi[i] * zi[i] / szz;
            assert!((hc2[i] - 1.0 / (1.0 - h).sqrt()).abs() < 1e-12);
            assert!((hc3[i] - 1.0 / (1.0 - h)).abs() < 1e-12);
        }
    }

    #[test]
    fn hc2_and_hc3_are_undefined_with_absorbed_parameters() {
        let z = column(&[1.0, -2.0, 0.5, 3.0, -1.0, 0.2]);
        let x = Mat::from_fn(6, 1, |_, _| 1.0);
        for variant in [HcVariant::Hc2, HcVariant::Hc3] {
            assert!(hc_row_scales(variant, &z, &x, 3, 1).is_none());
            assert!(hc_row_scales(variant, &z, &x, 2, 0).is_some());
        }
        // HC0・HC1は吸収したパラメータがあっても定義できる。
        assert!(hc_row_scales(HcVariant::Hc0, &z, &x, 3, 1).is_some());
        assert!(hc_row_scales(HcVariant::Hc1, &z, &x, 3, 1).is_some());
    }

    #[test]
    fn a_unit_leverage_makes_hc2_and_hc3_undefined() {
        let z = column(&[1.0, -1.0, 2.0, -2.0, 0.5, -0.5]);
        // 1観測だけのダミーはレバレッジが1。
        let x = Mat::from_fn(6, 1, |i, _| if i == 2 { 1.0 } else { 0.0 });
        assert!(hc_row_scales(HcVariant::Hc2, &z, &x, 2, 0).is_none());
        assert!(hc_row_scales(HcVariant::Hc3, &z, &x, 2, 0).is_none());
    }

    #[test]
    fn multiplier_follows_the_covariance_type() {
        let c = |cov, g| covariance_multiplier(cov, 100, 10, g).unwrap();
        assert_eq!(c(ScoreCovariance::Classical, None), 100.0 / 90.0);
        let order: Vec<usize> = (0..100).collect();
        assert_eq!(
            c(
                ScoreCovariance::Hac {
                    lags: 3,
                    order: &order
                },
                None
            ),
            100.0 / 90.0
        );
        assert_eq!(c(ScoreCovariance::Hc(HcVariant::Hc0), None), 1.0);
        assert_eq!(c(ScoreCovariance::Hc(HcVariant::Hc1), None), 1.0);
        let groups = labels(&[0; 100]);
        let expected = 20.0 / 19.0 * 99.0 / 90.0;
        assert!((c(ScoreCovariance::Cluster(&groups), Some(20)) - expected).abs() < 1e-12);
        assert!(covariance_multiplier(ScoreCovariance::Cluster(&groups), 100, 10, None).is_none());
        assert!(covariance_multiplier(ScoreCovariance::Classical, 10, 10, None).is_none());
    }

    #[test]
    fn score_meat_dispatches_on_the_covariance_type() {
        let psi = Mat::from_fn(6, 2, |i, a| {
            (i as f64 + 1.0) * if a == 0 { 1.0 } else { -0.5 }
        });
        let one = |_| Some(vec![1.0; 6]);
        let (hc, groups) = score_meat(
            &psi,
            ScoreCovariance::Hc(HcVariant::Hc0),
            one,
            || unreachable!(),
        )
        .unwrap();
        assert!(groups.is_none());
        let direct = psi.transpose() * &psi;
        for a in 0..2 {
            for b in 0..2 {
                assert!((hc.get(a, b) - direct.get(a, b)).abs() < 1e-12);
            }
        }
        let g = labels(&[0, 0, 1, 1, 2, 2]);
        let (_, n_groups) = score_meat(
            &psi,
            ScoreCovariance::Cluster(&g),
            |_| None,
            || unreachable!(),
        )
        .unwrap();
        assert_eq!(n_groups, Some(3));
        // 不正な設定は`None`。
        let one_group = labels(&[0; 6]);
        assert!(
            score_meat(
                &psi,
                ScoreCovariance::Cluster(&one_group),
                |_| None,
                || unreachable!()
            )
            .is_none()
        );
        let order = [0usize, 1, 2];
        assert!(
            score_meat(
                &psi,
                ScoreCovariance::Hac {
                    lags: 1,
                    order: &order
                },
                |_| None,
                || unreachable!()
            )
            .is_none()
        );
        let identity: Vec<usize> = (0..6).collect();
        assert!(
            score_meat(
                &psi,
                ScoreCovariance::Hac {
                    lags: 6,
                    order: &identity
                },
                |_| None,
                || unreachable!()
            )
            .is_none()
        );
        // `Classical`は呼び出し側の行列をそのまま返す。
        let (c, _) = score_meat(
            &psi,
            ScoreCovariance::Classical,
            |_| None,
            || Mat::identity(2, 2),
        )
        .unwrap();
        assert_eq!(*c.get(0, 0), 1.0);
    }
}
