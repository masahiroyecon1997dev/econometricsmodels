//! 参照(Cragg-Donald・Kleibergen-Paap・手計算・独立実装)との照合。

use super::*;
use crate::shared::kleibergen_paap::kleibergen_paap;
use crate::shared::weak_instruments::cragg_donald;

/// 等分散なら`F_eff`はCragg-Donaldの`F`、`K_eff`は操作変数の数。
#[test]
fn classical_effective_f_equals_cragg_donald() {
    for (endog, exog, z) in [small(), large(3), large(4)] {
        for n_absorbed in [0, 1, 3] {
            let cd = cragg_donald(&endog, &exog, &z, n_absorbed).unwrap();
            let result = effective_f(
                &endog,
                &exog,
                &z,
                n_absorbed,
                ScoreCovariance::Classical,
                0.05,
            )
            .unwrap();
            close(result.statistic, cd, 1e-10);
            for cv in result.simplified_tsls {
                close(cv.effective_df, z.ncols() as f64, 1e-10);
            }
        }
    }
}

/// 操作変数が1個なら`Ŵ2`はスカラーで、`F_eff`は頑健な`t²`(Kleibergen-PaapのWald F)に一致する
/// (独立に書いた統計量どうしの照合)。
#[test]
fn with_one_instrument_it_equals_the_kleibergen_paap_wald_f() {
    let order: Vec<usize> = (0..240).collect();
    let groups = clusters(240, 12);
    let (endog, exog, z) = large(4);
    let z1 = column(&z, 0);
    let covariances = [
        ScoreCovariance::Classical,
        ScoreCovariance::Hc(HcVariant::Hc0),
        ScoreCovariance::Hc(HcVariant::Hc1),
        ScoreCovariance::Hc(HcVariant::Hc2),
        ScoreCovariance::Hc(HcVariant::Hc3),
        ScoreCovariance::Cluster(&groups),
        ScoreCovariance::Hac {
            lags: 4,
            order: &order,
        },
    ];
    for cov in covariances {
        let kp = kleibergen_paap(&endog, &exog, &z1, 0, cov)
            .unwrap()
            .rk_wald_f
            .unwrap();
        let eff = effective_f(&endog, &exog, &z1, 0, cov, 0.05).unwrap();
        close(eff.statistic, kp, 1e-9);
        for cv in eff.simplified_tsls {
            close(cv.effective_df, 1.0, 1e-12);
        }
    }
}

/// 異分散のデータでは、ロバスト版の`K_eff`は`K`より小さくなる。
#[test]
fn robust_effective_degrees_of_freedom_falls_below_the_number_of_instruments() {
    let (endog, exog, z) = large(4);
    let eff = effective_f(
        &endog,
        &exog,
        &z,
        0,
        ScoreCovariance::Hc(HcVariant::Hc1),
        0.05,
    )
    .unwrap();
    for cv in eff.simplified_tsls {
        assert!(
            cv.effective_df < 4.0 - 1e-6 && cv.effective_df >= 1.0,
            "{}",
            cv.effective_df
        );
    }
}

/// 実装と別の道筋の手計算(K=4、異分散): 外生変数の偏回帰は正規方程式、直交正規化は対称平方根
/// `Q^{-1/2} = V Λ^{-1/2} V'`、共分散は定義どおりの和。生のスケールのまま与える。
#[test]
fn it_matches_a_direct_computation_with_a_symmetric_square_root() {
    use faer::prelude::Solve;
    let (endog, exog, z) = large(4);
    let (n, k) = (240usize, 4usize);
    let n_f = n as f64;
    let partial_out = |m: &Mat<f64>| {
        let beta = (exog.transpose() * &exog)
            .partial_piv_lu()
            .solve(exog.transpose() * m);
        m - &exog * beta
    };
    let (y, zt) = (partial_out(&endog), partial_out(&z));
    let q = Mat::from_fn(k, k, |a, b| {
        (0..n).map(|i| *zt.get(i, a) * *zt.get(i, b)).sum::<f64>() / n_f
    });
    let eigen = q.self_adjoint_eigen(Side::Lower).unwrap();
    let (values, vectors) = (eigen.S().column_vector(), eigen.U());
    let inv_root = Mat::<f64>::from_fn(k, k, |a, b| {
        (0..k)
            .map(|c| *vectors.get(a, c) * *vectors.get(b, c) / values.get(c).sqrt())
            .sum::<f64>()
    });
    let u = &zt * &inv_root;
    let pi: Vec<f64> = (0..k)
        .map(|a| (0..n).map(|i| *u.get(i, a) * *y.get(i, 0)).sum::<f64>() / n_f)
        .collect();
    let e: Vec<f64> = (0..n)
        .map(|i| *y.get(i, 0) - (0..k).map(|a| *u.get(i, a) * pi[a]).sum::<f64>())
        .collect();
    let pi_sq: f64 = pi.iter().map(|p| p * p).sum();

    // 共分散`V2 = c/n² · meat`を定義どおりに作り、`F_eff`と`K_eff`を比べる。
    let groups = clusters(n, 12);
    let l = (k + 2) as f64;
    let hc0_meat = Mat::<f64>::from_fn(k, k, |a, b| {
        (0..n)
            .map(|i| e[i] * e[i] * *u.get(i, a) * *u.get(i, b))
            .sum::<f64>()
    });
    let cluster_meat = Mat::from_fn(k, k, |a, b| {
        (0..20)
            .map(|g| {
                let sum = |c: usize| {
                    (12 * g..12 * g + 12)
                        .map(|i| *u.get(i, c) * e[i])
                        .sum::<f64>()
                };
                sum(a) * sum(b)
            })
            .sum()
    });
    let cluster_c = 20.0 / 19.0 * (n_f - 1.0) / (n_f - l);
    for (covariance, meat, c) in [
        (ScoreCovariance::Hc(HcVariant::Hc0), &hc0_meat, 1.0),
        (ScoreCovariance::Cluster(&groups), &cluster_meat, cluster_c),
    ] {
        let w2 = Mat::from_fn(k, k, |a, b| c * *meat.get(a, b) / n_f);
        let trace: f64 = (0..k).map(|a| *w2.get(a, a)).sum();
        let expected_f = pi_sq / (trace / n_f);
        let eigenvalues = w2.self_adjoint_eigenvalues(Side::Lower).unwrap();
        let (expected_df, expected_cv) = patnaik_critical_value(&eigenvalues, 10.0, 0.05).unwrap();

        let result = effective_f(&endog, &exog, &z, 0, covariance, 0.05).unwrap();
        close(result.statistic, expected_f, 1e-9);
        close(result.simplified_tsls[1].effective_df, expected_df, 1e-9);
        close(result.simplified_tsls[1].critical_value, expected_cv, 1e-9);
    }
}

#[test]
fn classical_generalized_critical_values_follow_the_kronecker_closed_form() {
    // 等分散の`Ŵ`は`Ω̂⊗I`なので、被説明変数が何であっても`B_TSLS = |1 − 2/K|`。
    let (endog, exog, z) = small();
    let ys = [
        Mat::from_fn(12, 1, |i, _| {
            (i as f64 * 0.7).sin() + 0.3 * *endog.get(i, 0)
        }),
        Mat::from_fn(12, 1, |i, _| (i as f64 * 1.9).cos() * 3.0 + *z.get(i, 2)),
    ];
    for y in &ys {
        let eff = crate::shared::effective_f::effective_f(
            &endog,
            &exog,
            &z,
            0,
            ScoreCovariance::Classical,
            0.05,
            Some(y),
        )
        .unwrap();
        close(eff.bias_bound.unwrap(), 1.0 / 3.0, 1e-9);
        let tsls = eff.tsls.unwrap();
        // 公表値(Table 1, K_eff = 3, x = 10/3): 8.53。
        assert!((tsls[1].critical_value - 8.53).abs() <= 0.0051);
        close(tsls[1].x, (1.0 / 3.0) / 0.10, 1e-9);
        // 一般化版は単純版を超えない。
        for (generalized, simplified) in tsls.iter().zip(eff.simplified_tsls) {
            assert!(generalized.critical_value <= simplified.critical_value + 1e-12);
            assert_eq!(generalized.tau, simplified.tau);
        }
    }
}

#[test]
fn robust_generalized_critical_values_lie_below_the_simplified_ones() {
    let (endog, exog, z) = large(4);
    let y = Mat::from_fn(240, 1, |i, _| {
        *endog.get(i, 0) * 0.8
            + (0.9 * i as f64).cos() * (1.0 + 0.8 * (0.05 * i as f64).sin().abs())
    });
    let groups = clusters(240, 12);
    for cov in [
        ScoreCovariance::Hc(HcVariant::Hc1),
        ScoreCovariance::Hc(HcVariant::Hc3),
        ScoreCovariance::Cluster(&groups),
    ] {
        let eff =
            crate::shared::effective_f::effective_f(&endog, &exog, &z, 0, cov, 0.05, Some(&y))
                .unwrap();
        let bound = eff.bias_bound.unwrap();
        assert!((0.0..=1.0).contains(&bound));
        let tsls = eff.tsls.unwrap();
        for (g, sim) in tsls.iter().zip(eff.simplified_tsls) {
            assert!(g.critical_value <= sim.critical_value + 1e-12);
            close(g.x, bound / g.tau, 1e-12);
            // 同じ`Ŵ2`なので有効自由度の式は同じ形で、`x`が小さい分だけ違う。
            assert!(g.effective_df >= 1.0 && g.effective_df <= 4.0 + 1e-9);
        }
        // 被説明変数を渡しても、統計量と単純版は渡さないときと(丸めの範囲で)同じ。
        let without = effective_f(&endog, &exog, &z, 0, cov, 0.05).unwrap();
        close(eff.statistic, without.statistic, 1e-10);
        for (a, b) in eff.simplified_tsls.iter().zip(without.simplified_tsls) {
            close(a.critical_value, b.critical_value, 1e-9);
        }
    }
}

/// 独立実装のオラクルの出力の1ケース(τ=0.10の単純版と一般化版)。
struct OracleCase {
    data: &'static str,
    covariance: &'static str,
    n_absorbed: usize,
    alpha: f64,
    statistic: f64,
    bound: f64,
    simplified_df: f64,
    simplified_cv: f64,
    general_df: f64,
    general_cv: f64,
}

// 期待値は、論文の式から独立に書いたPython(numpy/scipy)の第二の実装の出力。
const ORACLE_CASES: [OracleCase; 17] = [
    OracleCase {
        data: "D1",
        covariance: "classical",
        n_absorbed: 0,
        alpha: 0.05,
        statistic: 47.75015872855724,
        bound: 0.3333333333333334,
        simplified_df: 2.999999999999999,
        simplified_cv: 17.66865500987857,
        general_df: 3.0,
        general_cv: 8.525146907521838,
    },
    OracleCase {
        data: "D1",
        covariance: "hc0",
        n_absorbed: 0,
        alpha: 0.05,
        statistic: 77.82863811302731,
        bound: 0.6897176200477763,
        simplified_df: 1.9488088209914376,
        simplified_cv: 19.411520022491604,
        general_df: 1.9587382058737826,
        general_cv: 15.041539562225234,
    },
    OracleCase {
        data: "D1",
        covariance: "hc1",
        n_absorbed: 0,
        alpha: 0.05,
        statistic: 45.40003889926593,
        bound: 0.6897176200477728,
        simplified_df: 1.948808820991438,
        simplified_cv: 19.411520022491608,
        general_df: 1.9587382058737837,
        general_cv: 15.041539562225191,
    },
    OracleCase {
        data: "D1",
        covariance: "hc2",
        n_absorbed: 0,
        alpha: 0.05,
        statistic: 37.16924711424886,
        bound: 0.7216075170562607,
        simplified_df: 1.8673468108720372,
        simplified_cv: 19.60828450226825,
        general_df: 1.8757869738877289,
        general_cv: 15.67348214105031,
    },
    OracleCase {
        data: "D1",
        covariance: "hc3",
        n_absorbed: 0,
        alpha: 0.05,
        statistic: 16.613560879049405,
        bound: 0.7634194602561791,
        simplified_df: 1.6860522818899637,
        simplified_cv: 20.098737065784178,
        general_df: 1.6922956256182853,
        general_cv: 16.715105528920297,
    },
    OracleCase {
        data: "D1",
        covariance: "cluster",
        n_absorbed: 0,
        alpha: 0.05,
        statistic: 81.7468209723379,
        bound: 0.9790253173385269,
        simplified_df: 1.5664108545642823,
        simplified_cv: 20.470315103257054,
        general_df: 1.566672646487494,
        general_cv: 20.17277109519812,
    },
    OracleCase {
        data: "D1",
        covariance: "hac",
        n_absorbed: 0,
        alpha: 0.05,
        statistic: 60.89653663505676,
        bound: 0.7954550255789193,
        simplified_df: 1.6834302762428008,
        simplified_cv: 20.106433759432853,
        general_df: 1.688070787118372,
        general_cv: 17.190175417532082,
    },
    OracleCase {
        data: "D2_K4",
        covariance: "hc1",
        n_absorbed: 0,
        alpha: 0.05,
        statistic: 10.771408098611731,
        bound: 0.5252612086378783,
        simplified_df: 3.88707257035086,
        simplified_cv: 16.807738812481166,
        general_df: 3.891463199841886,
        general_cv: 10.637575916671347,
    },
    OracleCase {
        data: "D2_K4",
        covariance: "hc3",
        n_absorbed: 0,
        alpha: 0.05,
        statistic: 10.453904373965598,
        bound: 0.523726566513362,
        simplified_df: 3.8880036335923673,
        simplified_cv: 16.80699893917816,
        general_df: 3.892397470373615,
        general_cv: 10.616242412555888,
    },
    OracleCase {
        data: "D2_K4",
        covariance: "cluster",
        n_absorbed: 0,
        alpha: 0.05,
        statistic: 14.51108974792093,
        bound: 0.9952874462860861,
        simplified_df: 1.1514610400270275,
        simplified_cv: 22.205127281777123,
        general_df: 1.151488725687264,
        general_cv: 22.135048207402942,
    },
    OracleCase {
        data: "D2_K4",
        covariance: "hac",
        n_absorbed: 0,
        alpha: 0.05,
        statistic: 9.265451053280662,
        bound: 0.8302422433101112,
        simplified_df: 2.1501431670808095,
        simplified_cv: 18.97618282398947,
        general_df: 2.155434028112408,
        general_cv: 16.64665303847565,
    },
    OracleCase {
        data: "D2_K3",
        covariance: "hc2",
        n_absorbed: 0,
        alpha: 0.05,
        statistic: 13.706236798557088,
        bound: 0.33985663787882303,
        simplified_df: 2.9575962835559473,
        simplified_cv: 17.7196734039714,
        general_df: 2.961111218842441,
        general_cv: 8.653124717205056,
    },
    OracleCase {
        data: "D2_K4",
        covariance: "classical",
        n_absorbed: 3,
        alpha: 0.05,
        statistic: 10.457852537037246,
        bound: 0.5000000000000001,
        simplified_df: 4.0,
        simplified_cv: 16.719962607437473,
        general_df: 4.0,
        general_cv: 10.231461192414844,
    },
    OracleCase {
        data: "D2_K4",
        covariance: "hc1",
        n_absorbed: 3,
        alpha: 0.05,
        statistic: 10.633313122988504,
        bound: 0.5252612086378782,
        simplified_df: 3.88707257035086,
        simplified_cv: 16.807738812481166,
        general_df: 3.8914631998418847,
        general_cv: 10.63757591667135,
    },
    OracleCase {
        data: "D2_K4",
        covariance: "cluster",
        n_absorbed: 3,
        alpha: 0.05,
        statistic: 14.325050135768096,
        bound: 0.995287446286086,
        simplified_df: 1.1514610400270275,
        simplified_cv: 22.205127281777123,
        general_df: 1.151488725687264,
        general_cv: 22.135048207402953,
    },
    OracleCase {
        data: "D2_K4",
        covariance: "hac",
        n_absorbed: 3,
        alpha: 0.05,
        statistic: 9.146663219264244,
        bound: 0.8302422433101112,
        simplified_df: 2.15014316708081,
        simplified_cv: 18.97618282398947,
        general_df: 2.1554340281124085,
        general_cv: 16.646653038475662,
    },
    OracleCase {
        data: "D2_K4",
        covariance: "hc1",
        n_absorbed: 0,
        alpha: 0.1,
        statistic: 10.771408098611731,
        bound: 0.5252612086378783,
        simplified_df: 3.88707257035086,
        simplified_cv: 15.350213644794838,
        general_df: 3.891463199841886,
        general_cv: 9.49593292649413,
    },
];

/// 論文の式から独立に書いた第二の実装と、F・`B_TSLS`・有効自由度・臨界値(単純版と一般化版)が一致する
/// (相対差1e-9)。D1はn=12、D2はn=240の異分散データ。
#[test]
fn it_matches_the_independent_implementation_of_the_paper() {
    let small_data = small();
    let small_dep = small_y();
    let large4 = large(4);
    let large4_dep = large_y(&large4.0, &large4.1);
    let large3 = large(3);
    let large3_dep = large_y(&large3.0, &large3.1);
    let (small_groups, large_groups) = (clusters(12, 3), clusters(240, 12));
    let (small_order, large_order): (Vec<usize>, Vec<usize>) =
        ((0..12).collect(), (0..240).collect());
    for case in &ORACLE_CASES {
        let (name, cov_name, n_absorbed, alpha) =
            (case.data, case.covariance, case.n_absorbed, case.alpha);
        let (data, dep, is_small) = match name {
            "D1" => (&small_data, &small_dep, true),
            "D2_K4" => (&large4, &large4_dep, false),
            "D2_K3" => (&large3, &large3_dep, false),
            _ => unreachable!(),
        };
        let covariance = match cov_name {
            "classical" => ScoreCovariance::Classical,
            "hc0" => ScoreCovariance::Hc(HcVariant::Hc0),
            "hc1" => ScoreCovariance::Hc(HcVariant::Hc1),
            "hc2" => ScoreCovariance::Hc(HcVariant::Hc2),
            "hc3" => ScoreCovariance::Hc(HcVariant::Hc3),
            "cluster" => ScoreCovariance::Cluster(if is_small {
                &small_groups
            } else {
                &large_groups
            }),
            "hac" => ScoreCovariance::Hac {
                lags: if is_small { 2 } else { 4 },
                order: if is_small { &small_order } else { &large_order },
            },
            _ => unreachable!(),
        };
        let eff = crate::shared::effective_f::effective_f(
            &data.0,
            &data.1,
            &data.2,
            n_absorbed,
            covariance,
            alpha,
            Some(dep),
        )
        .unwrap();
        let label = format!("{name} {cov_name} n_absorbed={n_absorbed} alpha={alpha}");
        let tolerance = 1e-9;
        let close_to = |actual: f64, expected: f64, what: &str| {
            assert!(
                (actual - expected).abs() <= tolerance * expected.abs(),
                "{label} {what}: {actual} vs {expected}"
            );
        };
        close_to(eff.statistic, case.statistic, "F_eff");
        close_to(eff.bias_bound.unwrap(), case.bound, "B_TSLS");
        let simplified = eff.simplified_tsls[1];
        close_to(
            simplified.effective_df,
            case.simplified_df,
            "simplified K_eff",
        );
        close_to(
            simplified.critical_value,
            case.simplified_cv,
            "simplified cv",
        );
        let general = eff.tsls.unwrap()[1];
        close_to(general.effective_df, case.general_df, "generalized K_eff");
        close_to(general.critical_value, case.general_cv, "generalized cv");
    }
}

/// 被説明変数(n=12): `y = Y − 0.5·d2 + 0.1x + u1`、`d2 = 0.2 z1 + 0.7 z3 − 0.2 x + u2`。
fn small_y() -> Mat<f64> {
    Mat::from_fn(12, 1, |i, _| {
        let endog = 0.9 * SMALL_Z1[i] - 0.4 * SMALL_Z2[i] + 0.3 * SMALL_X[i] + SMALL_U1[i];
        let d2 = 0.2 * SMALL_Z1[i] + 0.7 * SMALL_Z3[i] - 0.2 * SMALL_X[i] + SMALL_U2[i];
        endog - 0.5 * d2 + 0.1 * SMALL_X[i] + SMALL_U1[i]
    })
}

/// `Ŵ2 = A_x'A_x`は`scale`によらない。独立実装のオラクルの出力(降順)。
const NEARLY_PROPORTIONAL_W2_EIGENVALUES: [f64; 3] =
    [6.143221039650903, 5.788179326174888, 4.801157909289455];

/// 独立実装のオラクルの出力(誤差がほぼ比例する`Ŵ`)。`ratio`は`min tr(S1)/tr(W1)`(概数。上限の確認だけに使う)、
/// `tolerance`は照合の相対許容誤差。`Ŵ`の要素に1e-16の相対ノイズを載せたときの`B_TSLS`の変化はおよそ
/// `1e-18 / ratio`(`scale = 1e-2`で4e-14、`1e-4`で5e-10)で、`sin`/`cos`の実装差(1 ulp)で`Ŵ`が動く分の余裕を取る。
/// `general`の`(有効自由度, 臨界値)`はτ = 0.05, 0.10, 0.20, 0.30の一般化版(`α = 0.05`)。
struct NearlyProportionalCase {
    scale: f64,
    ratio: f64,
    tolerance: f64,
    bound: f64,
    general: [(f64, f64); 4],
}

const NEARLY_PROPORTIONAL_CASES: [NearlyProportionalCase; 2] = [
    NearlyProportionalCase {
        scale: 1e-2,
        ratio: 1.498e-5,
        tolerance: 1e-9,
        bound: 0.42721529292073007,
        general: [
            (2.736249944464222, 16.0814675910672),
            (2.747543879854669, 10.13347054898915),
            (2.7671371742263857, 6.831455147291523),
            (2.783548290032974, 5.6146532630268995),
        ],
    },
    NearlyProportionalCase {
        scale: 1e-4,
        ratio: 1.498e-9,
        tolerance: 1e-7,
        bound: 0.4272152929790336,
        general: [
            (2.736249944462602, 16.081467592625643),
            (2.7475438798517358, 10.133470549846376),
            (2.7671371742215114, 6.831455147772618),
            (2.783548290026801, 5.614653263372654),
        ],
    },
];

/// 誤差がほぼ比例する`Ŵ`で、`B_TSLS`と一般化版の臨界値がオラクルと一致する(許容誤差は各ケースの`tolerance`)。
/// `β*`の近くの鋭い山を取りこぼすと`B_TSLS`が小さくなり、臨界値が下がる。固有値は本番と同じ`psd_eigenvalues`の経路。
#[test]
fn a_nearly_proportional_reduced_form_matches_the_independent_implementation() {
    for case in &NEARLY_PROPORTIONAL_CASES {
        let w = nearly_proportional_w(case.scale);
        let (w1, w12, w2) = blocks(&w, 3);
        let tr = |m: &Mat<f64>| (0..3).map(|a| *m.get(a, a)).sum::<f64>();
        let ratio = (tr(&w1) - tr(&w12) * tr(&w12) / tr(&w2)) / tr(&w1);
        assert!(
            (ratio - case.ratio).abs() <= 1e-2 * case.ratio,
            "scale={}: ratio {ratio} vs {}",
            case.scale,
            case.ratio
        );

        let eigenvalues = psd_eigenvalues(&w2).unwrap();
        let mut descending = eigenvalues.clone();
        descending.sort_by(|a, b| b.total_cmp(a));
        for (actual, expected) in descending.iter().zip(NEARLY_PROPORTIONAL_W2_EIGENVALUES) {
            close(*actual, expected, 1e-12);
        }

        let (bound, values) = generalized_critical_values(&w, 3, &eigenvalues, 0.05).unwrap();
        close(bound, case.bound, case.tolerance);
        for (value, (df, cv)) in values.iter().zip(case.general) {
            close(value.effective_df, df, case.tolerance);
            close(value.critical_value, cv, case.tolerance);
        }
    }
}
