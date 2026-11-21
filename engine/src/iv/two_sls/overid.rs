//! 2SLSの過剰識別検定（`cov_type`に追従する。docs/spec/iv-spec.md 3.5節）。
//!
//! `cov_type=Classical`は等分散前提のSargan検定（`sargan.rs`）。それ以外は、2SLS残差から
//! `cov_type`に対応する重み`S`を作り、その`S`でGMMを再推定した解でのHansen J統計量
//! `J = (Z'ê)'S⁻¹(Z'ê)`（`ê = y - Xβ̂(S)`）を使う。これは不均一分散・クラスター・自己相関に頑健な
//! score検定（Wooldridge, 1995）と代数的に同じ量で、2SLS推定値そのもので評価した
//! `(Z'e)'S⁻¹(Z'e)`とは別物（後者は`S⁻¹`が`Z'e`の漸近分散の一般化逆行列にならず、χ²に従わない）。
//! 定義と理由は`docs/spec/iv-spec.md`3.5節・`docs/methods/iv.md`。

use crate::iv::common::{IvError, IvInput, hcat};
use crate::linear::ols::CovType;
use crate::shared::covariance::time_ordering;
use crate::shared::error::{ClusterSetting, CommonError, MatrixKind};
use crate::shared::gmm::gmm_solve;
use crate::shared::gmm_iteration::gmm_whiten_for_weight;
use crate::shared::gmm_weight::WeightType;
use faer::Mat;

use super::sargan::sargan_test;

/// [`overid_test`]の戻り値。`df`はχ²分布の自由度。
pub(super) struct OverIdTest {
    pub(super) statistic: f64,
    pub(super) p_value: f64,
    pub(super) df: usize,
}

/// `cov_type`に追従する過剰識別検定。自由度は`len(instruments) - len(x_endog)`
/// （`fit()`冒頭で`k_instruments() >= k_endog()`を検証済み）。次の場合は`None`:
/// - 丁度識別（自由度0）。
/// - `cov_type=Cluster`でクラスター数`G`が全操作変数の数`l`より少ない。クラスターの`S`は
///   `rank(S) ≤ G`なので`G < l`で必ず特異になり、検定が定義できない。入力だけで決まる条件で、
///   診断の1つが定義できないだけなので`fit()`は失敗させない（Wu-Hausmanの拡張回帰が定義できない
///   場合と同じ扱い。構造方程式の`G <= q`のように必須の出力が定義できない場合は`fit()`冒頭で弾く）。
/// - 重み`S`の白色化、または再推定の`Ĝ = Z̃'X`の分解が数値的にほぼ特異で失敗する
///   （`NearSingularMatrix`の`GmmWeight`・`GmmBread`）。近共線な外生変数とクラスターの重みなどで、
///   `gmm_weight_type`が同じGMMも同じ入力で失敗する。この2種類以外の失敗は伝播する。
///
/// `Hc0`〜`Hc3`は同じ検定を返す。重みの`S`は補正を含まない素のロバストな`Σᵢ êᵢ²zᵢzᵢ'`で
/// （GMMの`gmm_weight_type="robust"`と同じ）、クラスターは`(G/(G-1))((n-1)/(n-k))`の補正なし、
/// HACはBartlettカーネルの重み付けのみ。補正を入れない理由は2つ: `Hc1`の`n/(n-k)`のように`S`を
/// 一様に定数倍する補正は、点推定を変えない一方で`J`の尺度を`1/c`倍に変える。`Hc2`・`Hc3`の
/// レバレッジ補正は行ごとに違う重みで、スカラーでさえない。
///
/// `hac_lags_used`は`cov_type=Hac`のとき共分散の計算が解決したラグ数。
///
/// # Errors
/// 上の`None`にする条件以外の失敗（到達不能な内部の不整合等）。数値上の問題を`None`で隠しすぎない
/// ため、`None`にするのは上の条件だけにする。
pub(super) fn overid_test(
    input: &IvInput,
    cov_type: &CovType,
    hac_lags_used: Option<usize>,
    residuals: &Mat<f64>,
    ssr: f64,
) -> Result<Option<OverIdTest>, IvError> {
    if input.k_instruments() == input.k_endog() {
        return Ok(None);
    }
    let df = input.k_instruments() - input.k_endog();
    if matches!(cov_type, CovType::Classical) {
        return sargan_test(input, residuals, ssr, df).map(Some);
    }
    robust_overid_test(input, cov_type, hac_lags_used, residuals, df)
}

/// `cov_type`がClassical以外のときの検定。2SLS残差から`S`を作って`Z`を白色化し
/// （`gmm_whiten_for_weight`）、`gmm_solve`で再推定した解のHansen J。2段階GMMの2回目の推定と同じ処理で、
/// `gmm_iterate`のループ本体（`gmm_reweighted_step`）と同じ部品を、`Z`を`X`の組み立て前に
/// 手放せるよう2つに分けて呼ぶ。
fn robust_overid_test(
    input: &IvInput,
    cov_type: &CovType,
    hac_lags_used: Option<usize>,
    residuals: &Mat<f64>,
    df: usize,
) -> Result<Option<OverIdTest>, IvError> {
    let n = input.nobs();
    let z = hcat(input.x_exog(), input.instruments());
    let l = z.ncols();
    // `gmm_whiten_for_weight`は`weight_type`の種類だけを見る（クラスターのグループとHACのラグ数・
    // 時系列順は別の引数で渡す）ので、中身は空でよい。`ztz`（`Z'Z`）は`Classical`でだけ使う。
    let (weight_type, cluster, hac) = match cov_type {
        CovType::Classical => {
            return Err(CommonError::internal(
                "the classical covariance has no robust overidentification test",
            )
            .into());
        }
        CovType::Hc0 | CovType::Hc1 | CovType::Hc2 | CovType::Hc3 => {
            (WeightType::Robust, None, None)
        }
        CovType::Hac { time_order, .. } => {
            let lags = hac_lags_used.ok_or_else(|| {
                CommonError::internal("the HAC lags were not resolved for the overidentification")
            })?;
            (
                WeightType::Hac {
                    lags: None,
                    time_order: Vec::new(),
                },
                None,
                Some((lags, time_ordering(time_order, n))),
            )
        }
        CovType::Cluster { groups } => {
            let groups = groups.as_ref().ok_or(CommonError::MissingClusterColumn {
                setting: ClusterSetting::CovType,
            })?;
            if groups.n_groups() < l {
                return Ok(None);
            }
            (WeightType::Cluster { groups: None }, Some(groups), None)
        }
    };

    // 古典的な重みではないので`Z'Z`は使わない。
    let ztz = Mat::<f64>::zeros(0, 0);
    // 重みの白色化と`Ĝ = Z̃'X`の分解が数値的に失敗するのは、検定が計算できないだけで、2SLSの
    // 係数・標準誤差には関係しない（近共線な外生変数とクラスターの重みなど。2SLSの推定は直交座標で
    // 近共線の設計行列まで扱えるが、`S`の階数と条件数はクラスターのスコアの数で決まる）。
    // 診断の1つのために`fit()`を失敗させず、この2種類の失敗だけを`None`にする。それ以外の失敗は伝播する。
    let near_singular = |err: CommonError| match err {
        CommonError::NearSingularMatrix {
            matrix: MatrixKind::GmmWeight | MatrixKind::GmmBread,
        } => Ok(None),
        other => Err(IvError::from(other)),
    };
    let (whitening, _s) =
        match gmm_whiten_for_weight(&weight_type, cluster, hac.as_ref(), &z, &ztz, residuals) {
            Ok(whitened) => whitened,
            Err(err) => return near_singular(err),
        };
    // `Z`は白色化で使い終わった。`X`（n×k）を組み立てる前に手放して、ピークのヒープを
    // `Z`と`Z̃`と`X`の3つではなく、`Z`と`Z̃`、`Z̃`と`X`のどちらかにする。
    drop(z);
    let x = hcat(input.x_exog(), input.x_endog());
    let solution = match gmm_solve(&x, input.y(), whitening) {
        Ok(solution) => solution,
        Err(err) => return near_singular(err),
    };
    let gmm_residuals = input.y() - &x * solution.beta();
    let test = solution.hansen_j_test(&gmm_residuals, 1.0, df)?;
    Ok(Some(OverIdTest {
        statistic: test.statistic,
        p_value: test.p_value,
        df,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iv::common::IvEstimate;
    use crate::iv::gmm::{GmmEstimator, GmmType};
    use crate::iv::two_sls::TwoSlsEstimator;
    use crate::iv::two_sls::test_support::just_identified_input;
    use crate::shared::group_codes::GroupCodes;
    use faer::Side;
    use faer::prelude::Solve;
    use statrs::distribution::{ChiSquared, ContinuousCDF};

    const N: usize = 24;

    /// 不均一分散な過剰識別データ（`x_endog`1個・除外操作変数3個、`l=4`・`k=2`で自由度2）。
    fn columns() -> (Vec<f64>, Vec<f64>, [Vec<f64>; 3]) {
        let f = |i: usize| i as f64;
        let z1: Vec<f64> = (0..N)
            .map(|i| (f(i) * 0.9).sin() * 3.0 + f(i) * 0.2)
            .collect();
        let z2: Vec<f64> = (0..N).map(|i| (f(i) * 1.3 + 0.5).cos() * 2.0).collect();
        let z3: Vec<f64> = (0..N).map(|i| ((i * 7) % 11) as f64 - 5.0).collect();
        let v: Vec<f64> = (0..N).map(|i| (f(i) * 2.1).sin() * 1.5).collect();
        let u: Vec<f64> = (0..N)
            .map(|i| (1.0 + 0.4 * z1[i].abs()) * ((f(i) * 3.7).cos() + 0.5 * v[i]))
            .collect();
        let x: Vec<f64> = (0..N)
            .map(|i| 0.8 * z1[i] + 0.5 * z2[i] - 0.4 * z3[i] + v[i])
            .collect();
        let y: Vec<f64> = (0..N).map(|i| 1.0 + 0.7 * x[i] + u[i]).collect();
        (y, x, [z1, z2, z3])
    }

    fn input() -> IvInput {
        let (y, x, z) = columns();
        IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x),
            vec!["x".to_string()],
            &z,
            vec!["z1".to_string(), "z2".to_string(), "z3".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap()
    }

    fn labels(n_groups: usize) -> Vec<String> {
        (0..N).map(|i| format!("g{}", i % n_groups)).collect()
    }

    fn cluster(n_groups: usize) -> CovType {
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&labels(n_groups))),
        }
    }

    fn hac(lags: i64) -> CovType {
        CovType::Hac {
            lags: Some(lags),
            time_order: (0..N).map(|i| i as f64).collect(),
        }
    }

    fn two_sls(cov_type: CovType) -> TwoSlsEstimator {
        TwoSlsEstimator::fit(input(), cov_type, 0.95).unwrap()
    }

    fn gmm_j(weight: WeightType) -> f64 {
        gmm_j_of(input(), weight)
    }

    fn gmm_j_of(input: IvInput, weight: WeightType) -> f64 {
        GmmEstimator::fit(
            input,
            GmmType::TwoStep { weight },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap()
        .hansen_j_statistic()
        .unwrap()
    }

    fn inverse(m: &Mat<f64>) -> Mat<f64> {
        m.llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(m.nrows(), m.ncols()))
    }

    /// `S`の構築を`s_of`（2SLS残差の列と`Z`から`S`を作る）として受け取り、2SLSの残差→`S`→
    /// `β̂ = (X'ZS⁻¹Z'X)⁻¹X'ZS⁻¹Z'y`→`J = (Z'ê)'S⁻¹(Z'ê)`を、`fit()`の部品を使わず
    /// 明示的な行列で組み立てる手計算オラクル。
    fn manual_j(s_of: impl Fn(&[f64], &Mat<f64>) -> Mat<f64>) -> f64 {
        let (y, x_endog, z_excl) = columns();
        let z = Mat::from_fn(N, 4, |i, j| if j == 0 { 1.0 } else { z_excl[j - 1][i] });
        let x = Mat::from_fn(N, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y = Mat::from_fn(N, 1, |i, _| y[i]);
        let gmm_beta = |w: &Mat<f64>| {
            let ztx = z.transpose() * &x;
            let bread = ztx.transpose() * w * &ztx;
            let rhs = ztx.transpose() * w * (z.transpose() * &y);
            inverse(&bread) * rhs
        };
        let beta_2sls = gmm_beta(&inverse(&(z.transpose() * &z)));
        let e0 = &y - &x * &beta_2sls;
        let e0_vec: Vec<f64> = (0..N).map(|i| *e0.get(i, 0)).collect();
        let w = inverse(&s_of(&e0_vec, &z));
        let beta = gmm_beta(&w);
        let e = &y - &x * &beta;
        let g = z.transpose() * &e;
        let quad = g.transpose() * &w * &g;
        *quad.get(0, 0)
    }

    fn manual_robust_s(e: &[f64], z: &Mat<f64>) -> Mat<f64> {
        Mat::from_fn(4, 4, |a, b| {
            (0..N)
                .map(|i| e[i].powi(2) * *z.get(i, a) * *z.get(i, b))
                .sum()
        })
    }

    /// `groups[i]`が行`i`のクラスター（0からの連番）。
    fn manual_cluster_s(groups: Vec<usize>) -> impl Fn(&[f64], &Mat<f64>) -> Mat<f64> {
        let n_groups = groups.iter().max().unwrap() + 1;
        move |e, z| {
            let score = |g: usize, a: usize| -> f64 {
                (0..N)
                    .filter(|&i| groups[i] == g)
                    .map(|i| e[i] * *z.get(i, a))
                    .sum()
            };
            Mat::from_fn(4, 4, |a, b| {
                (0..n_groups).map(|g| score(g, a) * score(g, b)).sum()
            })
        }
    }

    /// Bartlettカーネルの`S = Γ₀ + Σ_h (1 - h/(L+1))(Γ_h + Γ_h')`（行順が時系列順）。
    fn manual_hac_s(lags: usize) -> impl Fn(&[f64], &Mat<f64>) -> Mat<f64> {
        manual_hac_s_ordered(lags, (0..N).collect())
    }

    /// 時点`t`の行が`order[t]`のBartlettカーネルの`S`（行が時系列順でないとき）。
    fn manual_hac_s_ordered(
        lags: usize,
        order: Vec<usize>,
    ) -> impl Fn(&[f64], &Mat<f64>) -> Mat<f64> {
        move |e, z| {
            let gamma = |h: usize, a: usize, b: usize| -> f64 {
                (h..N)
                    .map(|t| {
                        let (r, q) = (order[t], order[t - h]);
                        e[r] * *z.get(r, a) * e[q] * *z.get(q, b)
                    })
                    .sum()
            };
            Mat::from_fn(4, 4, |a, b| {
                let mut s = gamma(0, a, b);
                for h in 1..=lags {
                    let w = 1.0 - (h as f64) / ((lags + 1) as f64);
                    s += w * (gamma(h, a, b) + gamma(h, b, a));
                }
                s
            })
        }
    }

    fn assert_close(actual: f64, expected: f64, label: &str) {
        assert!(
            (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0),
            "{label}: actual={actual}, expected={expected}"
        );
    }

    /// `Hc0`は素のロバストな`S = Σ êᵢ²zᵢzᵢ'`で、独立に組み立てた手計算の`J`と一致する。
    /// `Hc1`〜`Hc3`は小標本補正もレバレッジ補正も入れないので`Hc0`と同じ値になる。
    #[test]
    fn hc_variants_equal_the_manual_robust_hansen_j() {
        let expected = manual_j(manual_robust_s);
        for cov_type in [CovType::Hc0, CovType::Hc1, CovType::Hc2, CovType::Hc3] {
            let estimator = two_sls(cov_type.clone());
            assert_close(
                estimator.overid_statistic().unwrap(),
                expected,
                &format!("{cov_type:?}"),
            );
            assert_eq!(estimator.overid_df(), Some(2));
        }
    }

    #[test]
    fn cluster_equals_the_manual_cluster_hansen_j() {
        let expected = manual_j(manual_cluster_s((0..N).map(|i| i % 6).collect()));
        assert_close(
            two_sls(cluster(6)).overid_statistic().unwrap(),
            expected,
            "cluster",
        );
    }

    #[test]
    fn hac_equals_the_manual_newey_west_hansen_j() {
        let expected = manual_j(manual_hac_s(2));
        assert_close(two_sls(hac(2)).overid_statistic().unwrap(), expected, "hac");
    }

    /// 2SLS残差から作った`S`での再推定は、2段階GMM（同じ重みの種類）のHansen Jと一致する。
    #[test]
    fn equals_the_two_step_gmm_hansen_j_of_the_same_weight() {
        assert_close(
            two_sls(CovType::Hc0).overid_statistic().unwrap(),
            gmm_j(WeightType::Robust),
            "robust",
        );
        assert_close(
            two_sls(cluster(6)).overid_statistic().unwrap(),
            gmm_j(WeightType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&labels(6))),
            }),
            "cluster",
        );
        assert_close(
            two_sls(hac(2)).overid_statistic().unwrap(),
            gmm_j(WeightType::Hac {
                lags: Some(2),
                time_order: (0..N).map(|i| i as f64).collect(),
            }),
            "hac",
        );
    }

    /// `Classical`は従来どおりのSarganで、等分散の2段階GMMのHansen Jとも一致する。
    #[test]
    fn classical_keeps_the_sargan_statistic() {
        let estimator = two_sls(CovType::Classical);
        assert_close(
            estimator.overid_statistic().unwrap(),
            gmm_j(WeightType::Classical),
            "classical",
        );
        assert_eq!(estimator.overid_df(), Some(2));
    }

    #[test]
    fn p_value_is_the_chi_squared_tail_with_the_overidentification_degrees_of_freedom() {
        let estimator = two_sls(CovType::Hc0);
        let expected = ChiSquared::new(2.0)
            .unwrap()
            .sf(estimator.overid_statistic().unwrap());
        assert_close(estimator.overid_p_value().unwrap(), expected, "p-value");
    }

    #[test]
    fn is_none_when_just_identified_for_every_cov_type() {
        // `just_identified_input`は5行。
        let hac_5 = CovType::Hac {
            lags: Some(1),
            time_order: (0..5).map(|i| i as f64).collect(),
        };
        let cluster_2 = CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(
                &["g0", "g1", "g0", "g1", "g0"].map(String::from),
            )),
        };
        for cov_type in [
            CovType::Classical,
            CovType::Hc0,
            CovType::Hc3,
            hac_5,
            cluster_2,
        ] {
            let estimator = TwoSlsEstimator::fit(just_identified_input(), cov_type, 0.95).unwrap();
            assert_eq!(estimator.overid_statistic(), None);
            assert_eq!(estimator.overid_p_value(), None);
            assert_eq!(estimator.overid_df(), None);
        }
    }

    /// 行が時系列順に並んでいないとき（`time_order`が行の置換）、時点の順に並べ直した手計算の`S`と
    /// 一致する。恒等順の値とは異なる（並べ替えが効いている）。GMMの同じ重みとも一致する。
    #[test]
    fn hac_with_permuted_time_order_matches_the_manual_oracle_on_sorted_rows() {
        let time: Vec<f64> = (0..N).map(|i| ((i * 7 + 3) % N) as f64).collect();
        // 時点`t`の行（`7⁻¹ = 7 (mod 24)`）。
        let order: Vec<usize> = (0..N).map(|t| ((t + N - 3) % N * 7) % N).collect();
        let cov = || CovType::Hac {
            lags: Some(2),
            time_order: time.clone(),
        };
        let expected = manual_j(manual_hac_s_ordered(2, order));
        let actual = two_sls(cov()).overid_statistic().unwrap();
        assert_close(actual, expected, "permuted time order");
        assert!(
            (actual - manual_j(manual_hac_s(2))).abs() > 1e-6 * actual,
            "the permuted order must change the statistic"
        );
        assert_close(
            actual,
            gmm_j(WeightType::Hac {
                lags: Some(2),
                time_order: time.clone(),
            }),
            "permuted time order vs gmm",
        );
    }

    /// ラグ数の自動選択（`lags=None`）はGMMの同じ設定と一致し、ラグ0は`Hc0`と一致する。
    #[test]
    fn hac_with_automatic_lags_matches_gmm_and_zero_lags_match_hc0() {
        let time: Vec<f64> = (0..N).map(|i| i as f64).collect();
        let automatic = CovType::Hac {
            lags: None,
            time_order: time.clone(),
        };
        assert_close(
            two_sls(automatic).overid_statistic().unwrap(),
            gmm_j(WeightType::Hac {
                lags: None,
                time_order: time.clone(),
            }),
            "automatic lags",
        );
        let zero = CovType::Hac {
            lags: Some(0),
            time_order: time,
        };
        assert_close(
            two_sls(zero).overid_statistic().unwrap(),
            two_sls(CovType::Hc0).overid_statistic().unwrap(),
            "zero lags",
        );
    }

    /// 大きさが不均衡なクラスター（3, 3, 3, 3, 3, 9行）でも、手計算のクラスターの`S`と一致する。
    #[test]
    fn unbalanced_clusters_match_the_manual_oracle() {
        let groups: Vec<usize> = (0..N).map(|i| (i / 3).min(5)).collect();
        let labels: Vec<String> = groups.iter().map(|g| format!("g{g}")).collect();
        let cov = CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&labels)),
        };
        assert_close(
            two_sls(cov).overid_statistic().unwrap(),
            manual_j(manual_cluster_s(groups)),
            "unbalanced clusters",
        );
    }

    /// 外生変数が追加の列（`w`）を持つ入力（切片あり・なし）で、`Z = [x_exog, instruments]`・
    /// `X = [x_exog, x_endog]`の列の組み立てが、GMMの同じ重みと一致する。
    #[test]
    fn extra_exogenous_column_matches_gmm_with_and_without_intercept() {
        let (y, x, z) = columns();
        let w: Vec<f64> = (0..N)
            .map(|i| ((i as f64) * 0.37).sin() * 2.0 + (i % 5) as f64)
            .collect();
        let build = |include_intercept: bool| {
            IvInput::from_columns(
                &y,
                std::slice::from_ref(&w),
                vec!["w".to_string()],
                std::slice::from_ref(&x),
                vec!["x".to_string()],
                &z,
                vec!["z1".to_string(), "z2".to_string(), "z3".to_string()],
                include_intercept,
                "y".to_string(),
            )
            .unwrap()
        };
        for include_intercept in [true, false] {
            let label = format!("intercept={include_intercept}");
            let robust =
                TwoSlsEstimator::fit(build(include_intercept), CovType::Hc0, 0.95).unwrap();
            assert_eq!(robust.overid_df(), Some(2), "{label}");
            assert_close(
                robust.overid_statistic().unwrap(),
                gmm_j_of(build(include_intercept), WeightType::Robust),
                &format!("{label}/robust"),
            );
            let clustered =
                TwoSlsEstimator::fit(build(include_intercept), cluster(6), 0.95).unwrap();
            assert_close(
                clustered.overid_statistic().unwrap(),
                gmm_j_of(
                    build(include_intercept),
                    WeightType::Cluster {
                        groups: Some(GroupCodes::from_labels_without_keys(&labels(6))),
                    },
                ),
                &format!("{label}/cluster"),
            );
        }
    }

    /// 外生変数がほぼ共線（`x2 = x1 + 1e-7·ノイズ`）でクラスター重みの`S`がほぼ特異になる入力。
    /// 同じ重みのGMMも失敗する。2SLSの`fit()`は成功し（係数・標準誤差は有効）、過剰識別検定の
    /// 3つの値だけが`None`になる。ロバストな重み（`Hc0`）は同じ入力で計算できる。
    #[test]
    fn near_singular_cluster_weight_degrades_to_none_without_failing_fit() {
        let f = |i: usize| i as f64;
        let x1: Vec<f64> = (0..60).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();
        let x2: Vec<f64> = (0..60)
            .map(|i| x1[i] + 1e-7 * (((i * 5) % 11) as f64 - 5.0))
            .collect();
        let z: Vec<Vec<f64>> = vec![
            (0..60).map(|i| ((i * 3) % 7) as f64 - 3.0).collect(),
            (0..60).map(|i| ((i * 11) % 9) as f64 - 4.0).collect(),
            (0..60).map(|i| ((i * 2) % 5) as f64 - 2.0).collect(),
        ];
        let endog: Vec<f64> = (0..60)
            .map(|i| {
                0.8 * z[0][i] + 0.5 * z[1][i] + 0.3 * z[2][i] + 0.4 * x1[i] + (f(i) * 1.7).sin()
            })
            .collect();
        let y: Vec<f64> = (0..60)
            .map(|i| 1.0 + 2.0 * x1[i] + 3.0 * x2[i] + 1.5 * endog[i] + (f(i) * 2.9).cos())
            .collect();
        let build = || {
            IvInput::from_columns(
                &y,
                &[x1.clone(), x2.clone()],
                vec!["x1".to_string(), "x2".to_string()],
                std::slice::from_ref(&endog),
                vec!["endog".to_string()],
                &z,
                vec!["z1".to_string(), "z2".to_string(), "z3".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };
        let groups = GroupCodes::from_labels_without_keys(
            &(0..60).map(|i| format!("g{}", i % 8)).collect::<Vec<_>>(),
        );

        let clustered = TwoSlsEstimator::fit(
            build(),
            CovType::Cluster {
                groups: Some(groups),
            },
            0.95,
        )
        .unwrap();
        assert_eq!(clustered.overid_statistic(), None);
        assert_eq!(clustered.overid_p_value(), None);
        assert_eq!(clustered.overid_df(), None);
        assert!((0..4).all(|j| clustered.std_errors().get(j, 0).is_finite()));

        let robust = TwoSlsEstimator::fit(build(), CovType::Hc0, 0.95).unwrap();
        assert!(robust.overid_statistic().is_some());
        assert_eq!(robust.overid_df(), Some(2));
    }

    /// クラスター数が全操作変数の数`l`より少ないと`S`が必ず特異になる。`fit()`は成功し、
    /// 検定だけが定義できず`None`になる（`G = l`は定義できる）。
    #[test]
    fn cluster_is_none_below_the_instrument_count_and_defined_at_it() {
        let below = two_sls(cluster(3));
        assert_eq!(below.overid_statistic(), None);
        assert_eq!(below.overid_p_value(), None);
        assert_eq!(below.overid_df(), None);
        assert_eq!(below.params().nrows(), 2);

        let at = two_sls(cluster(4));
        assert!(at.overid_statistic().is_some());
        assert_eq!(at.overid_df(), Some(2));
    }
}
