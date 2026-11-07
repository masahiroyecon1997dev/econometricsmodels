//! Cragg-Donald統計量とStock-Yogoの臨界値。2SLS/GMMが共有する。
//!
//! 計算は行列を受け取る共有関数(`shared::weak_instruments::cragg_donald`、臨界値は
//! `shared::stock_yogo`、ロバスト版は`shared::kleibergen_paap`)で、ここは`IvInput`から行列を
//! 取り出して呼ぶだけの層。Cragg-Donaldは`cov_type`に依存しない(等分散前提の統計量。弱操作変数の
//! 部分F統計量と同じ判断、`docs/spec/iv-spec.md`3.4節)。Kleibergen-Paapは`cov_type`ごとの共分散を使う。

use super::input::IvInput;
use crate::linear::ols::CovType;
use crate::shared::covariance::{HcVariant, resolve_hac_lags, time_ordering};
use crate::shared::effective_f::{EffectiveF, effective_f};
use crate::shared::kleibergen_paap::{RankTest, kleibergen_paap};
use crate::shared::score_covariance::ScoreCovariance;
use crate::shared::stock_yogo::{
    TSLS_BIAS_LEVELS, TSLS_SIZE_LEVELS, tsls_bias_critical_values, tsls_size_critical_values,
};
use crate::shared::weak_instruments::cragg_donald;

/// Stock-Yogoの臨界値表の1行: 水準(`levels`)とその臨界値(`values`)。同じ位置が対応する。
/// Cragg-Donald統計量が臨界値を超えれば、その水準で「操作変数は弱い」を棄却する。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StockYogoCriticalValues {
    /// 水準(バイアス表は相対バイアスの上限、サイズ表は5%のWald検定のサイズの上限)。
    pub levels: [f64; 4],
    /// 水準ごとの臨界値(5%有意水準のテスト)。
    pub values: [f64; 4],
}

/// [`compute_cragg_donald`]の結果。
#[derive(Debug, Clone, PartialEq)]
pub struct CraggDonald {
    /// Cragg-Donald Wald F統計量。内生変数が1個なら弱操作変数の部分F統計量に一致する。
    pub statistic: f64,
    /// 内生変数の数。
    pub n_endog: usize,
    /// 除外操作変数の数`K2`。
    pub n_excluded: usize,
    /// 2SLSの相対バイアスの臨界値(内生変数1〜3個、`K2`が`n_endog + 2`〜30のとき)。
    pub tsls_bias: Option<StockYogoCriticalValues>,
    /// 2SLSのサイズ歪みの臨界値(内生変数1〜2個、`K2`が`n_endog`〜30のとき)。
    pub tsls_size: Option<StockYogoCriticalValues>,
}

/// Cragg-Donald統計量とStock-Yogoの臨界値を計算する。
///
/// 統計量が定義できないとき(内生変数が無い、識別不足、内生変数が共線、第一段階が完全フィット、
/// 外生変数と除外操作変数の共線など)は`None`。診断なので`fit()`は失敗させない。臨界値は表にある
/// 組み合わせだけ`Some`。外生変数の定数列は`input.x_exog()`に含まれており、平均を引かずに
/// 偏回帰で落とす。
#[must_use]
pub fn compute_cragg_donald(input: &IvInput) -> Option<CraggDonald> {
    // `fit()`の外から呼ばれてもfaerを逐次に固定する(`crate::shared::parallelism`)。
    crate::shared::parallelism::ensure_serial();
    let statistic = cragg_donald(input.x_endog(), input.x_exog(), input.instruments(), 0)?;
    let (n_endog, n_excluded) = (input.k_endog(), input.k_instruments());
    let row = |levels: [f64; 4], values: Option<[f64; 4]>| {
        values.map(|values| StockYogoCriticalValues { levels, values })
    };
    Some(CraggDonald {
        statistic,
        n_endog,
        n_excluded,
        tsls_bias: row(
            TSLS_BIAS_LEVELS,
            tsls_bias_critical_values(n_endog, n_excluded),
        ),
        tsls_size: row(
            TSLS_SIZE_LEVELS,
            tsls_size_critical_values(n_endog, n_excluded),
        ),
    })
}

/// [`compute_kleibergen_paap`]の結果。LMとWald Fは共分散の階数の条件が違い(Wald Fのスコアは合計が0に
/// なるので、クラスターがLMより1つ多く要る)、片方だけ定義できることがあるため、独立に`Option`にする。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KleibergenPaap {
    /// rk LM統計量(過小識別の検定。`χ²(n_excluded − n_endog + 1)`)。
    pub rk_lm: Option<RankTest>,
    /// rk Wald F統計量(Cragg-Donald Fの、`cov_type`に頑健な版。検定ではなくp値は無い)。
    pub rk_wald_f: Option<f64>,
    /// 内生変数の数。
    pub n_endog: usize,
    /// 除外操作変数の数。
    pub n_excluded: usize,
}

/// `CovType`を一次段階のスコアの共分散の選択に直す。`Hac`の時間順は`order`に書き込んで借用する。
/// 不正な設定(グループの無い`Cluster`、HACのラグ数や時間順の長さの不正)は`None`。
fn score_covariance<'a>(
    cov_type: &'a CovType,
    n: usize,
    order: &'a mut Vec<usize>,
) -> Option<ScoreCovariance<'a>> {
    Some(match cov_type {
        CovType::Classical => ScoreCovariance::Classical,
        CovType::Hc0 => ScoreCovariance::Hc(HcVariant::Hc0),
        CovType::Hc1 => ScoreCovariance::Hc(HcVariant::Hc1),
        CovType::Hc2 => ScoreCovariance::Hc(HcVariant::Hc2),
        CovType::Hc3 => ScoreCovariance::Hc(HcVariant::Hc3),
        CovType::Cluster { .. } => ScoreCovariance::Cluster(cov_type.cluster_groups()?),
        CovType::Hac { lags, time_order } => {
            if time_order.len() != n {
                return None;
            }
            *order = time_ordering(time_order, n);
            ScoreCovariance::Hac {
                lags: resolve_hac_lags(*lags, n).ok()?,
                order: order.as_slice(),
            }
        }
    })
}

/// `cov_type`に頑健なMontiel Olea-Pflueger有効F統計量と、単純版TSLSのPatnaik臨界値を計算する。
///
/// 内生変数が1個のときだけ定義される(2個以上は`None`)。`cov_type`の扱いと`None`の条件は
/// [`compute_kleibergen_paap`]と同じ規約(`Classical`の有効FはCragg-Donaldに一致する)。`alpha`は
/// 臨界値の水準(0〜1)。診断なので`fit()`は失敗させない。
#[must_use]
pub fn compute_effective_f(input: &IvInput, cov_type: &CovType, alpha: f64) -> Option<EffectiveF> {
    // `fit()`の外から呼ばれてもfaerを逐次に固定する(`crate::shared::parallelism`)。
    crate::shared::parallelism::ensure_serial();
    let mut order = Vec::new();
    let covariance = score_covariance(cov_type, input.nobs(), &mut order)?;
    effective_f(
        input.x_endog(),
        input.x_exog(),
        input.instruments(),
        0,
        covariance,
        alpha,
    )
}

/// `cov_type`に頑健なKleibergen-Paap統計量(rk LMとrk Wald F)を計算する。
///
/// `cov_type`のHC・クラスター・HACが、一次段階のスコアの共分散に使われる(`Classical`のWald Fは
/// Cragg-Donaldに一致)。`Hac`の`lags`は`None`なら経験則で解決し、時間順は`time_order`による。
/// 両方とも定義できないとき(内生変数が無い、識別不足、`Cluster`でグループが無い、HACのラグ数が
/// 不正など)は`None`。診断なので`fit()`は失敗させない。
#[must_use]
pub fn compute_kleibergen_paap(input: &IvInput, cov_type: &CovType) -> Option<KleibergenPaap> {
    // `fit()`の外から呼ばれてもfaerを逐次に固定する(`crate::shared::parallelism`)。
    crate::shared::parallelism::ensure_serial();
    let mut order = Vec::new();
    let covariance = score_covariance(cov_type, input.nobs(), &mut order)?;
    let (y, x, z) = (input.x_endog(), input.x_exog(), input.instruments());
    let statistics = kleibergen_paap(y, x, z, 0, covariance)?;
    if statistics.rk_lm.is_none() && statistics.rk_wald_f.is_none() {
        return None;
    }
    Some(KleibergenPaap {
        rk_lm: statistics.rk_lm,
        rk_wald_f: statistics.rk_wald_f,
        n_endog: input.k_endog(),
        n_excluded: input.k_instruments(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iv::common::compute_first_stage;
    use crate::linear::ols::CovType;

    type Columns = (Vec<f64>, Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>);

    fn columns() -> Columns {
        let x = vec![
            0.3, -1.2, 0.8, 2.0, -0.5, 1.1, 0.7, -0.4, 1.6, 0.2, -1.0, 0.9,
        ];
        let z1 = vec![
            1.0, 0.2, -0.7, 1.5, 0.9, -1.1, 0.3, 0.8, -0.6, 1.2, -0.3, 0.5,
        ];
        let z2 = vec![
            0.4, -0.9, 1.3, 0.1, -0.2, 0.7, -1.4, 0.6, 0.0, -0.8, 1.1, 0.35,
        ];
        let z3 = vec![
            -0.5, 0.7, 0.2, -1.1, 0.8, 0.4, 1.0, -0.3, 0.9, -0.6, 0.15, -0.2,
        ];
        let u1 = [
            0.2, -0.5, 0.1, 0.4, -0.3, 0.6, -0.2, 0.05, -0.45, 0.3, 0.15, -0.1,
        ];
        let u2 = [
            -0.3, 0.4, 0.5, -0.2, 0.1, -0.6, 0.35, 0.2, -0.1, -0.4, 0.25, 0.3,
        ];
        let d1: Vec<f64> = (0..12)
            .map(|i| 0.9 * z1[i] - 0.4 * z2[i] + 0.3 * x[i] + u1[i])
            .collect();
        let d2: Vec<f64> = (0..12)
            .map(|i| 0.2 * z1[i] + 0.7 * z3[i] - 0.2 * x[i] + u2[i])
            .collect();
        let y: Vec<f64> = (0..12)
            .map(|i| d1[i] - 0.5 * d2[i] + 0.1 * x[i] + u1[i])
            .collect();
        (y, x, vec![d1, d2], vec![z1, z2, z3])
    }

    fn input(n_endog: usize, include_intercept: bool) -> IvInput {
        let (y, x, d, z) = columns();
        let names = ["d1".to_string(), "d2".to_string()];
        IvInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &d[..n_endog],
            names[..n_endog].to_vec(),
            &z,
            vec!["z1".to_string(), "z2".to_string(), "z3".to_string()],
            include_intercept,
            "y".to_string(),
        )
        .unwrap()
    }

    /// 内生変数が1個なら、既存の弱操作変数の部分F統計量に一致する(定数項の有無によらず)。
    #[test]
    fn with_one_endogenous_variable_it_equals_the_weak_instrument_f_statistic() {
        for include_intercept in [true, false] {
            let input = input(1, include_intercept);
            let (_, weak_f) = compute_first_stage(&input, &CovType::Classical, 0.95).unwrap();

            let cd = compute_cragg_donald(&input).unwrap();

            let f = weak_f[0].1;
            assert!(
                (cd.statistic - f).abs() < 1e-9 * f.abs(),
                "intercept={include_intercept} cd={} f={f}",
                cd.statistic
            );
            assert_eq!((cd.n_endog, cd.n_excluded), (1, 3));
        }
    }

    #[test]
    fn it_attaches_the_stock_yogo_critical_values_that_the_tables_contain() {
        let one = compute_cragg_donald(&input(1, true)).unwrap();
        assert_eq!(
            one.tsls_size,
            Some(StockYogoCriticalValues {
                levels: [0.10, 0.15, 0.20, 0.25],
                values: [22.30, 12.83, 9.54, 7.80],
            })
        );
        assert_eq!(
            one.tsls_bias,
            Some(StockYogoCriticalValues {
                levels: [0.05, 0.10, 0.20, 0.30],
                values: [13.91, 9.08, 6.46, 5.39],
            })
        );

        // 内生変数2個・除外操作変数3個: サイズ表(K2=3)・バイアス表(K2>=4が必要なので無し)。
        let two = compute_cragg_donald(&input(2, true)).unwrap();
        assert_eq!(two.tsls_size.map(|c| c.values[0]), Some(13.43));
        assert_eq!(two.tsls_bias, None);
        assert_eq!((two.n_endog, two.n_excluded), (2, 3));
    }

    #[test]
    fn it_works_without_an_intercept_for_two_endogenous_variables() {
        let cd = compute_cragg_donald(&input(2, false)).unwrap();
        assert!(cd.statistic.is_finite() && cd.statistic > 0.0);
        assert_eq!((cd.n_endog, cd.n_excluded), (2, 3));
    }

    #[test]
    fn it_returns_none_when_the_model_is_underidentified() {
        let (y, x, d, z) = columns();
        let input = IvInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &d,
            vec!["d1".to_string(), "d2".to_string()],
            &z[..1],
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        assert_eq!(compute_cragg_donald(&input), None);
    }

    fn clusters() -> CovType {
        let labels: Vec<String> = (0..12).map(|i| format!("g{}", i / 3)).collect();
        CovType::Cluster {
            groups: Some(crate::shared::group_codes::GroupCodes::from_labels_without_keys(&labels)),
        }
    }

    /// `cov_type`ごとの共分散が共有関数に渡る。期待値はlinearmodels 7.1の`kleibergen_paap`・
    /// `kleibergen_paap_f`(HC1のFは`robust`のF)。
    #[test]
    fn kleibergen_paap_uses_the_covariance_of_the_cov_type() {
        let input = input(2, true);
        let hac = CovType::Hac {
            lags: Some(2),
            time_order: (0..12).map(f64::from).collect(),
        };
        let cases = [
            (CovType::Classical, 8.497665557296273, 5.661334354194701),
            (CovType::Hc0, 5.20374630742091, 11.40464437283528),
            (CovType::Hc1, 3.035518679328865, 6.652709217487246),
            (CovType::Hc3, 1.0460104688089573, 2.5532340821578416),
            (clusters(), 2.9797800411423454, 19.612521171644726),
            (hac, 2.7216722971551834, 9.501421955858229),
        ];
        for (cov_type, lm, f) in cases {
            let kp = compute_kleibergen_paap(&input, &cov_type).unwrap();
            let test = kp.rk_lm.unwrap();
            assert_eq!(test.df, 2);
            assert!((test.statistic - lm).abs() < 1e-9 * lm, "{cov_type:?}");
            let wald_f = kp.rk_wald_f.unwrap();
            assert!((wald_f - f).abs() < 1e-9 * f, "{cov_type:?}");
            assert_eq!((kp.n_endog, kp.n_excluded), (2, 3));
        }
    }

    #[test]
    fn kleibergen_paap_with_classical_covariance_equals_cragg_donald() {
        for n_endog in [1, 2] {
            let input = input(n_endog, true);
            let cd = compute_cragg_donald(&input).unwrap().statistic;
            let kp = compute_kleibergen_paap(&input, &CovType::Classical).unwrap();
            let wald_f = kp.rk_wald_f.unwrap();
            assert!((wald_f - cd).abs() < 1e-10 * cd, "{wald_f} {cd}");
        }
    }

    /// 時間順は`time_order`の昇順で、行の並びとは無関係: 行を時間順に並べ替えた入力に恒等の
    /// `time_order`を渡した結果は、元の行順にシャッフルした`time_order`を渡した結果と一致する。
    #[test]
    fn kleibergen_paap_hac_follows_the_time_order() {
        let time: [f64; 12] = [7.0, 2.0, 11.0, 0.0, 5.0, 9.0, 1.0, 10.0, 3.0, 6.0, 8.0, 4.0];
        let mut by_time: Vec<usize> = (0..12).collect();
        by_time.sort_by(|&a, &b| time[a].total_cmp(&time[b]));

        let (y, x, d, z) = columns();
        let pick = |v: &[f64]| by_time.iter().map(|&i| v[i]).collect::<Vec<f64>>();
        let sorted_input = IvInput::from_columns(
            &pick(&y),
            &[pick(&x)],
            vec!["x".to_string()],
            &d.iter().map(|c| pick(c)).collect::<Vec<_>>(),
            vec!["d1".to_string(), "d2".to_string()],
            &z.iter().map(|c| pick(c)).collect::<Vec<_>>(),
            vec!["z1".to_string(), "z2".to_string(), "z3".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let identity = CovType::Hac {
            lags: Some(2),
            time_order: (0..12).map(f64::from).collect(),
        };
        let shuffled = CovType::Hac {
            lags: Some(2),
            time_order: time.to_vec(),
        };

        let expected = compute_kleibergen_paap(&sorted_input, &identity).unwrap();
        let actual = compute_kleibergen_paap(&input(2, true), &shuffled).unwrap();
        let (a, e) = (actual.rk_wald_f.unwrap(), expected.rk_wald_f.unwrap());
        assert!((a - e).abs() < 1e-10 * e, "{a} {e}");
        // 時間順を無視して行順に使うと別の値になる。
        let ignored = compute_kleibergen_paap(&input(2, true), &identity).unwrap();
        assert!((ignored.rk_wald_f.unwrap() - e).abs() > 1e-6 * e);
    }

    #[test]
    fn kleibergen_paap_is_none_for_unusable_covariance_settings() {
        let input = input(2, true);
        assert_eq!(
            compute_kleibergen_paap(&input, &CovType::Cluster { groups: None }),
            None
        );
        let bad_lags = CovType::Hac {
            lags: Some(12),
            time_order: (0..12).map(f64::from).collect(),
        };
        assert_eq!(compute_kleibergen_paap(&input, &bad_lags), None);
        let short_order = CovType::Hac {
            lags: None,
            time_order: vec![0.0, 1.0],
        };
        assert_eq!(compute_kleibergen_paap(&input, &short_order), None);
    }

    /// LMとWald Fは独立に定義の可否が決まる: 2クラスターでLM(自由度2)は定義できるが、Wald Fはできない。
    #[test]
    fn kleibergen_paap_reports_the_two_statistics_independently() {
        let labels: Vec<String> = (0..12).map(|i| format!("g{}", i / 6)).collect();
        let two_clusters = CovType::Cluster {
            groups: Some(crate::shared::group_codes::GroupCodes::from_labels_without_keys(&labels)),
        };
        let kp = compute_kleibergen_paap(&input(2, true), &two_clusters).unwrap();
        assert!(kp.rk_lm.is_some());
        assert_eq!(kp.rk_wald_f, None);
    }

    #[test]
    fn effective_f_with_classical_covariance_equals_cragg_donald() {
        let input = input(1, true);
        let cd = compute_cragg_donald(&input).unwrap().statistic;
        let eff = compute_effective_f(&input, &CovType::Classical, 0.05).unwrap();
        assert!((eff.statistic - cd).abs() < 1e-10 * cd);
        assert_eq!(eff.n_instruments, 3);
        assert_eq!(eff.alpha, 0.05);
        // Table 1 (K_eff = 3)の`x = 10`の臨界値は17.67。
        let tau_10 = eff.simplified_tsls[1];
        assert_eq!(tau_10.tau, 0.10);
        assert!((tau_10.critical_value - 17.67).abs() < 0.005);
    }

    /// 操作変数が1個なら、`cov_type`ごとの有効Fはrk Wald Fに一致する(同じ`CovType`の変換を共有する
    /// 2つの診断の配線の照合)。HACは`lags = None`の既定のラグ数も通る。
    #[test]
    fn effective_f_with_one_instrument_equals_the_kleibergen_paap_wald_f() {
        let (y, x, d, z) = columns();
        let input = IvInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &d[..1],
            vec!["d1".to_string()],
            &z[..1],
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let hac = |lags| CovType::Hac {
            lags,
            time_order: (0..12).map(f64::from).collect(),
        };
        let cov_types = [
            CovType::Classical,
            CovType::Hc0,
            CovType::Hc1,
            CovType::Hc2,
            CovType::Hc3,
            clusters(),
            hac(None),
            hac(Some(3)),
        ];
        for cov_type in cov_types {
            let kp = compute_kleibergen_paap(&input, &cov_type).unwrap();
            let eff = compute_effective_f(&input, &cov_type, 0.05).unwrap();
            let wald_f = kp.rk_wald_f.unwrap();
            assert!(
                (eff.statistic - wald_f).abs() < 1e-9 * wald_f,
                "{cov_type:?}: {} vs {wald_f}",
                eff.statistic
            );
            assert_eq!(eff.n_instruments, 1);
        }
    }

    #[test]
    fn effective_f_is_none_for_several_endogenous_variables_and_bad_settings() {
        assert_eq!(
            compute_effective_f(&input(2, true), &CovType::Classical, 0.05),
            None
        );
        let input = input(1, true);
        assert_eq!(
            compute_effective_f(&input, &CovType::Cluster { groups: None }, 0.05),
            None
        );
        let short_order = CovType::Hac {
            lags: None,
            time_order: vec![0.0, 1.0],
        };
        assert_eq!(compute_effective_f(&input, &short_order, 0.05), None);
        assert_eq!(compute_effective_f(&input, &CovType::Classical, 1.5), None);
    }

    /// 内生変数が4個(表は3個まで)なら、統計量は出るが臨界値は無い。
    #[test]
    fn it_has_no_critical_values_outside_the_published_tables() {
        let mut state = 99u64;
        let mut column = || -> Vec<f64> {
            (0..12)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
                })
                .collect()
        };
        let y = column();
        let x = column();
        let endog: Vec<Vec<f64>> = (0..4).map(|_| column()).collect();
        let instruments: Vec<Vec<f64>> = (0..5).map(|_| column()).collect();
        let input = IvInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &endog,
            (1..=4).map(|j| format!("d{j}")).collect(),
            &instruments,
            (1..=5).map(|j| format!("z{j}")).collect(),
            true,
            "y".to_string(),
        )
        .unwrap();

        let cd = compute_cragg_donald(&input).unwrap();

        assert!(cd.statistic.is_finite() && cd.statistic >= 0.0);
        assert_eq!((cd.n_endog, cd.n_excluded), (4, 5));
        assert_eq!((cd.tsls_bias, cd.tsls_size), (None, None));
    }

    /// ちょうど識別(`K2 == n`)では、サイズ表にはあるがバイアス表には無い(`K2 >= n + 2`が必要)。
    #[test]
    fn it_has_only_the_size_row_when_just_identified() {
        let (y, x, d, z) = columns();
        let input = IvInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &d,
            vec!["d1".to_string(), "d2".to_string()],
            &z[..2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let cd = compute_cragg_donald(&input).unwrap();

        assert_eq!(cd.tsls_bias, None);
        assert_eq!(cd.tsls_size.map(|c| c.values[0]), Some(7.03));
    }

    #[test]
    fn it_returns_none_without_endogenous_variables() {
        let (y, x, _, z) = columns();
        let input = IvInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &[],
            vec![],
            &z,
            vec!["z1".to_string(), "z2".to_string(), "z3".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        assert_eq!(compute_cragg_donald(&input), None);
    }
}
