//! Cragg-Donald統計量とStock-Yogoの臨界値。2SLS/GMMが共有する。
//!
//! 計算は行列を受け取る共有関数(`shared::weak_instruments::cragg_donald`、臨界値は
//! `shared::stock_yogo`)で、ここは`IvInput`から行列を取り出して呼ぶだけの層。`cov_type`には
//! 依存しない(等分散前提の統計量。弱操作変数の部分F統計量と同じ判断、`docs/spec/iv-spec.md`3.4節)。

use super::input::IvInput;
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
