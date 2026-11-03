//! ハウスマン検定（回帰ベース、`plm::phtest(method = "aux")`相当）。

use statrs::distribution::{ChiSquared, ContinuousCDF};

use super::input::ReCovType;
use super::variance::debug_assert_within_reuse_contract;
use crate::linear::common::LeastSquaresError;
use crate::linear::ols::{CovType, OlsEstimator, OlsInput};
use crate::panel::common::{
    PanelError, panel_cluster_cov_params, panel_driscoll_kraay_cov_params, resolve_dk_bandwidth,
    xtx_inverse,
};
use crate::panel::fe::FeEstimator;
use crate::shared::error::{ClusterSetting, CommonError};
use crate::shared::group_codes::GroupCodes;
use crate::shared::linear_algebra::{COLLINEARITY_TOLERANCE, independent_columns_in_order};
use crate::shared::validation::{validate_cluster_count_covers_slopes, validate_cluster_groups};
use crate::shared::wald::wald_f_test;

/// ハウスマン補助回帰で`x̃_j`が識別できない列か（`x*_j - x̃_j`が全観測で一定か）。
///
/// `x*_j`は準偏差変換済み、`x̃_j`はwithin変換済みの同じ列。一定かどうかは差の標準偏差が
/// `n·ε·raw_scale`以下かで判定する（相対閾値。`raw_scale`は変換前の元の列の`max|x_j|`。
/// 差そのものや変換後の列を基準にすると、変換の打ち消しで残る丸め誤差の大きさと比べられない）。
fn hausman_within_column_is_aliased(x_star: &[f64], x_within: &[f64], raw_scale: f64) -> bool {
    let n = x_star.len() as f64;
    let diff: Vec<f64> = x_star.iter().zip(x_within).map(|(a, b)| a - b).collect();
    let mean = diff.iter().sum::<f64>() / n;
    let std_dev = (diff.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    std_dev <= n * f64::EPSILON * raw_scale
}

/// ハウスマン補助回帰を`OlsEstimator`のcov_type（Classical/Hc1〜Hc3）で当てはめ、`X̃`の
/// 末尾`k`列が同時にゼロというロバストWald検定のF統計量を返す。
fn hausman_aux_wald_f(
    aux_input: OlsInput,
    cov_type: CovType,
    k: usize,
    confidence_level: f64,
) -> Result<f64, LeastSquaresError> {
    let aux = OlsEstimator::fit_auxiliary(aux_input, cov_type, confidence_level)?;
    Ok(aux.wald_test_last_columns(k)?.0)
}

/// ハウスマン検定（`re-spec.md`3.7節、`re.rs`のモジュールdoc「ハウスマン検定」参照）。
///
/// Wooldridge (2010) 10.7.3節の補助回帰版（`plm::phtest(method = "aux")`相当）:
/// 準偏差変換済みの`y*`を、定数項（**未変換の`1`**）・準偏差変換済みの傾き`X*`・
/// within変換済みの`X̃`にpooled OLSで回帰し、`X̃`の係数`k`個が同時にゼロという
/// Wald検定を行う。定数項をθ変換しない（変換済み定数列`1-θ_i`を使わない）
/// のは`plm::phtest(method = "aux")`と同じ扱い。バランスパネルでは`θ_i`が全個体共通のため
/// どちらでも同値だが、不均衡パネルでは値が異なる。
///
/// 補助回帰の共分散はRE本体の`cov_type`に連動させる（Classical/Hc1〜Hc3は`OlsEstimator`の
/// 同名`CovType`、Cluster/Dkは`panel_cluster_cov_params`/`panel_driscoll_kraay_cov_params`を
/// 補助回帰の設計行列・残差に適用。`re.rs`のモジュールdoc「ハウスマン検定」参照）。`cluster_codes`は
/// RE本体の共分散で使うクラスター列のコード（`groups`が`None`なら`entity`のコード）。
/// 統計量はWald統計量そのもの（`wald_test_last_columns`/`wald_f_test`が返すF統計量の`k`倍）、
/// p値は`χ²_k.sf(stat)`。
///
/// `fe`は`swamy_arora_variance_components`が返した1-way FE推定量で、`X̃`は
/// その推定が`OlsEstimator`へ委譲したwithin変換済み設計行列（`fe.within_input().x()`）を
/// 再利用する。`fe`はRE本体と同じ個体効果構造（1-wayは`time`なし、2-wayは`time`あり）で`within_transform_one_way`/`within_transform_two_way`を適用済みの
/// ため、変換し直すのとビット単位で同じ値になる（同じ変換を二度計算しないため再利用する）。`y_star`/`x_star`はRE本体の
/// 回帰に渡した準偏差変換済みの`y`と傾き`X`（変換済み定数列は含まない）そのもの。
///
/// **識別できない`X̃`の列は検定対象から外す**: バランスパネルで時点間でのみ変わる列（年トレンド等）は
/// 個体平均が個体間で一定になり、`x*_j = x̃_j + 定数`で未変換の定数項と共線になる（`plm`の
/// `phtest(method = "aux")`はこの入力で添字エラーになる）。その列の係数δは識別できないため
/// （`hausman_within_column_is_aliased`）、残りの列だけで検定し`hausman_df`もその分小さくなる。
/// 時点間でのみ変わる列が無ければ結果は変わらない。
///
/// 比較対象の傾き係数が0個、または検定できる列が残らないなら`Ok(None)`（検定対象が無い）。補助回帰・Wald検定の失敗
/// （ランク落ち・クラスター数不足・共分散部分行列のほぼ特異性等）は、計算自体が成立しない
/// ため`PanelError::HausmanTestFailed`として伝播し`fit()`全体を失敗させる（設計行列の
/// 多重共線性と同じ方針。`re.rs`のモジュールdoc「`None`になるのは…」の項参照）。
#[allow(clippy::too_many_arguments)]
pub(super) fn re_hausman_test(
    fe: &FeEstimator,
    time_varying: &[usize],
    y_star: &[f64],
    x_star: &[Vec<f64>],
    x_star_names: &[String],
    dep_var_name: &str,
    cov_type: &ReCovType,
    cluster_codes: &GroupCodes,
    confidence_level: f64,
) -> Result<Option<(f64, usize, f64)>, PanelError> {
    let to_err = |source| PanelError::HausmanTestFailed { source };
    // 検定対象は時間変化のある列の`X̃`だけ（`fe`の説明変数）。時間不変の列は`x_star`には
    // 含まれるが、within変換後に全ゼロ列になるため`X̃`には入れない（`re-spec.md`5章）。
    let k_tv = fe.input().x().len();
    if k_tv == 0 {
        return Ok(None);
    }

    // `X̃`は内部FE推定が`OlsEstimator`へ委譲したwithin変換済み設計行列（切片なし、
    // `fe.within_input().x()`）から取り出す。`within_transform_one_way`/`within_transform_two_way`で
    // 変換し直すのとビット単位で同じ値（関数docコメント参照）。
    debug_assert_within_reuse_contract(fe, k_tv);
    debug_assert_eq!(time_varying.len(), k_tv);
    let x_within_mat = fe.within_input().x();
    debug_assert_eq!(x_within_mat.ncols(), k_tv);

    // `x̃_j`が補助回帰で識別できない列（`x*_j - x̃_j`が全観測で一定で、未変換の定数項と
    // 共線になる。バランスパネルで時点間でのみ変わる列等）は検定対象から外す。
    // `x*_j - x̃_j = (1-θ_i)·x̄_ij`なので、個体平均`x̄_ij`が個体間で等しいバランスパネルで起きる。
    let tested: Vec<usize> = (0..k_tv)
        .filter(|&m| {
            let raw_scale = fe.input().x()[m]
                .iter()
                .fold(0.0_f64, |acc, v| acc.max(v.abs()));
            !hausman_within_column_is_aliased(
                &x_star[time_varying[m]],
                x_within_mat.col_as_slice(m),
                raw_scale,
            )
        })
        .collect();
    // 残った`x̃_j`のうち、定数項・`X*`・前の`x̃`の線形結合になっている列も外す。個体平均を説明変数に
    // 加える相関ランダム効果（Mundlak）の指定では、`x*_j - x̃_j = (1-θ)·x̄_j`が加えた個体平均の列
    // （の`X*`）で表せるため、`x̃_j`が別名になる。落とすのは検定対象の`x̃`側だけ（列順で後ろの列を
    // 落とす規則で、`X̃`は列順で最後のため`X*`の列は落ちない。`X*`の列同士が従属なら補助回帰が
    // 特異になって`HausmanTestFailed`）。
    let ones = vec![1.0; y_star.len()];
    let mut candidates: Vec<&[f64]> = vec![ones.as_slice()];
    candidates.extend(x_star.iter().map(Vec::as_slice));
    candidates.extend(tested.iter().map(|&m| x_within_mat.col_as_slice(m)));
    let independent = independent_columns_in_order(&candidates, COLLINEARITY_TOLERANCE);
    let tested: Vec<usize> = tested
        .iter()
        .zip(&independent[1 + x_star.len()..])
        .filter_map(|(&m, &keep)| keep.then_some(m))
        .collect();
    let k = tested.len();
    if k == 0 {
        return Ok(None);
    }

    let mut columns = x_star.to_vec();
    columns.extend(
        tested
            .iter()
            .map(|&m| x_within_mat.col_as_slice(m).to_vec()),
    );
    let mut names = x_star_names.to_vec();
    names.extend(
        tested
            .iter()
            .map(|&m| format!("{}_within", fe.input().x_names()[m])),
    );

    let aux_input = OlsInput::from_columns(y_star, &columns, names, true, dep_var_name.to_string())
        .map_err(to_err)?;

    // Classical/Hc1〜Hc3は`OlsEstimator`の同名`CovType`でそのまま当てはめる。Cluster・Dkは
    // `OlsEstimator`が`String`列でグループ化する・Newey-West型HACと別物のため使わず、
    // `params`・`residuals`がcov_typeによらないことを使って、Classicalで当てはめた補助回帰から
    // 得て共分散だけ整数コード版（`panel_cluster_cov_params`/`panel_driscoll_kraay_cov_params`）で
    // 計算し直す。
    let f_stat = match cov_type {
        ReCovType::Classical => {
            hausman_aux_wald_f(aux_input, CovType::Classical, k, confidence_level)
                .map_err(to_err)?
        }
        ReCovType::Hc1 => {
            hausman_aux_wald_f(aux_input, CovType::Hc1, k, confidence_level).map_err(to_err)?
        }
        ReCovType::Hc2 => {
            hausman_aux_wald_f(aux_input, CovType::Hc2, k, confidence_level).map_err(to_err)?
        }
        ReCovType::Hc3 => {
            hausman_aux_wald_f(aux_input, CovType::Hc3, k, confidence_level).map_err(to_err)?
        }
        ReCovType::Cluster { .. } => {
            // `OlsEstimator::fit`の`cov_type=Cluster`と同じ事前検証（`G >= 2`・`G > q`、`q`は
            // 補助回帰の傾き係数の数）を、同じ順序（QR分解より前）で行う。
            let n = aux_input.nobs();
            let g = validate_cluster_groups(cluster_codes, n, ClusterSetting::CovType)
                .map_err(|e| to_err(LeastSquaresError::Common(e)))?;
            validate_cluster_count_covers_slopes(
                g,
                aux_input.k() - usize::from(aux_input.has_intercept()),
            )
            .map_err(|e| to_err(LeastSquaresError::Common(e)))?;

            let aux = OlsEstimator::fit_auxiliary(aux_input, CovType::Classical, confidence_level)
                .map_err(to_err)?;
            let k_aux = aux.input().k();
            let x_mat = aux.input().x();
            let xtx_inv = xtx_inverse(x_mat)?;
            let residuals: Vec<f64> = (0..n).map(|i| *aux.residuals().get(i, 0)).collect();
            // 小標本補正`G/(G-1)·(n-1)/(n-K)`は`K=k_aux`（`OlsEstimator`のクラスター共分散と
            // 同式）、検定の自由度は`G-1`。
            let cov_params = panel_cluster_cov_params(
                x_mat,
                &residuals,
                &xtx_inv,
                n,
                k_aux,
                cluster_codes,
                k_aux,
            );
            wald_f_test(aux.params(), &cov_params, k_aux - k, k, g - 1)
                .map_err(|e| to_err(e.into()))?
                .0
        }
        ReCovType::Dk { bandwidth, time } => {
            let aux = OlsEstimator::fit_auxiliary(aux_input, CovType::Classical, confidence_level)
                .map_err(to_err)?;
            let n = aux.input().nobs();
            let k_aux = aux.input().k();
            // 時点列の長さ・バンド幅・`T > k`（`validate_dk_periods_cover_tested_coefficients`）は
            // RE本体のDK計算（`fit()`）が先に検証済みのため、ここでは再検証しない。
            let time = time.codes();
            let t_periods = time.n_groups();
            let bw = resolve_dk_bandwidth(*bandwidth, t_periods)?;
            let x_mat = aux.input().x();
            let xtx_inv = xtx_inverse(x_mat)?;
            let residuals: Vec<f64> = (0..n).map(|i| *aux.residuals().get(i, 0)).collect();
            // fixestのDKは`K.fixef="full"`が既定（RE本体の`fit()`のDk分岐と同じ、
            // `k_correction=k_aux`）。Wald検定の分母自由度も`t_periods-1`に揃える
            // （RE本体・FEのDK分岐と同じ`t.df="min"`）。
            let cov_params =
                panel_driscoll_kraay_cov_params(x_mat, &residuals, &xtx_inv, time, k_aux, bw);
            wald_f_test(aux.params(), &cov_params, k_aux - k, k, t_periods - 1)
                .map_err(|e| to_err(e.into()))?
                .0
        }
    };

    let stat = k as f64 * f_stat;
    // `wald_f_test`が成功した時点でF統計量は有限（`ensure_well_conditioned_symmetric_matrix`
    // 通過後の二次形式）、`k >= 1`で`ChiSquared::new`も失敗しないため、以下は防御的。
    if !stat.is_finite() {
        return Err(to_err(LeastSquaresError::Common(CommonError::internal(
            "the Hausman Wald statistic is not finite",
        ))));
    }
    let chi2 = ChiSquared::new(k as f64).map_err(|e| {
        to_err(LeastSquaresError::Common(CommonError::internal(
            e.to_string(),
        )))
    })?;
    Ok(Some((stat, k, chi2.sf(stat))))
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::super::transform::quasi_demean_transform;
    use super::super::variance::VarianceComponents;

    use super::super::variance::swamy_arora_variance_components;
    use super::*;

    use crate::panel::common::TimeKeys;

    use crate::panel::common::quasi_demean_column;

    use crate::panel::fe::FeCovType;
    use crate::panel::fe::FeEffects;
    use crate::panel::fe::FeInput;

    use crate::panel::re::ReEffects;
    use crate::panel::re::ReEstimator;
    use crate::panel::re::ReInput;

    use crate::shared::error_catalog::Stage;

    #[test]
    fn re_estimator_fit_with_only_time_invariant_regressors_has_no_hausman_test() {
        // 説明変数がすべて時間不変でも推定できる。内部FEの説明変数は0個で、
        // σ_ε²は`y`のwithin変動の平均二乗（自由度`n - N`）。比較対象の傾きが無いので
        // ハウスマン検定は`None`、within R²は0。
        let input = time_invariant_input(&["z"]);
        let vc = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();
        assert!(vc.time_varying.is_empty());
        assert_eq!(vc.fe.input().x().len(), 0);

        let y = time_invariant_columns().y;
        let ssr_within: f64 = y
            .chunks(4)
            .map(|c| {
                let m = c.iter().sum::<f64>() / 4.0;
                c.iter().map(|v| (v - m).powi(2)).sum::<f64>()
            })
            .sum();
        assert!((vc.sigma2_eps - ssr_within / (24.0 - 6.0)).abs() < 1e-12);

        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
        assert_eq!(re.hausman_statistic(), None);
        assert_eq!(re.hausman_p_value(), None);
        assert_eq!(re.hausman_df(), None);
        assert_eq!(re.r_squared_within(), 0.0);
        assert_eq!(re.df_model(), 2);
    }

    #[test]
    fn hausman_is_none_when_every_time_varying_regressor_varies_only_over_time() {
        // 時間変化のある列が`xt`（時点間でのみ変わる列）だけなら、`x̃t`は識別できず、
        // 検定できる列が残らないため`hausman_*`は`None`（`fit()`自体は成功する）。
        let c = two_way_columns();
        let input = ReInput::from_columns(
            &c.y,
            std::slice::from_ref(&c.xt),
            strings(&["xt"]),
            &c.entity,
            None,
            "y".into(),
        )
        .unwrap();

        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

        assert_eq!(re.hausman_statistic(), None);
        assert_eq!(re.hausman_p_value(), None);
        assert_eq!(re.hausman_df(), None);
    }

    #[test]
    fn two_way_hausman_matches_the_auxiliary_regression_reference_for_every_cov_type() {
        // 補助回帰: 3つのθで変換した`y*`を`[1, X*, X̃]`（`X̃`は2-way within変換後に分散のある列）に
        // 回帰し`X̃`の係数をWald検定する。参照は`plm::phtest(method = "aux", effect = "twoways",
        // vcov = ...)`（`x1`のみ、全列が両次元で変動する入力）と、一次元でしか変わらない列を含む
        // 入力での手計算の補助回帰（plmは係数を位置で選ぶため、この入力ではplmの値が別の係数の
        // 検定になり使えない）。cov_type順: classical, HC1, HC2, HC3, cluster（entity）,
        // DK（bandwidth=1）。
        let c = two_way_columns();
        let cases: [(&[&str], [f64; 6]); 2] = [
            (
                &["x1"],
                [
                    1.219_038_535_996_78,
                    2.288_441_340_520_52,
                    2.143_067_391_820_05,
                    1.747_073_734_662_46,
                    5.186_875_856_510_08,
                    1.110_285_110_999_32,
                ],
            ),
            (
                &["x1", "xi", "xt"],
                [
                    0.020_172_191_267_105_1,
                    0.030_257_894_203_708_7,
                    0.027_319_433_563_145_9,
                    0.018_691_740_018_767,
                    0.439_816_368_093_923,
                    0.024_735_842_716_204_3,
                ],
            ),
        ];
        for (names, expected) in cases {
            for ((label, cov_type), want) in all_cov_types(&c.time).into_iter().zip(expected) {
                let re = ReEstimator::fit(two_way_input(names), ReEffects::TwoWay, cov_type, 0.95)
                    .unwrap_or_else(|e| panic!("{names:?} {label}: {e}"));
                assert_eq!(re.hausman_df(), Some(1), "{names:?} {label}");
                assert_rel(
                    re.hausman_statistic().unwrap(),
                    want,
                    1e-9,
                    &format!("{names:?} {label}"),
                );
                let p = re.hausman_p_value().unwrap();
                assert!((0.0..=1.0).contains(&p), "{names:?} {label}: p={p}");
            }
        }
    }

    #[test]
    fn two_way_hausman_with_several_tested_columns_matches_references() {
        // cov_type順: classical, HC1, HC2, HC3, cluster（entity）, DK（bandwidth=1）。各行は
        // （統計量, χ²_df上側確率）。`x1 + x2`は両次元で変動する2列（plm::phtest(aux, twoways)の値と
        // 手計算が一致）、`x1 + z + x2`は個体平均も時点平均も0の交互作用`z`を含む（`z`は識別できず
        // 検定対象から外れるため`hausman_df = 2`）、`x1 + z`は`df = 1`。参照は手計算の補助回帰。
        let c = two_way_columns();
        type Row = (f64, f64);
        let cases: [(&[&str], usize, [Row; 6]); 3] = [
            (
                &["x1", "x2"],
                2,
                [
                    (9.92830801953696, 0.00698385648499824),
                    (8.10342574438997, 0.0173925578818405),
                    (6.72610170604262, 0.0346294482635305),
                    (4.38401947725594, 0.111692051071814),
                    (18.094162886712, 0.00011773414976345),
                    (11.0661183693506, 0.00395387493521181),
                ],
            ),
            (
                &["x1", "z", "x2"],
                2,
                [
                    (10.0371631198638, 0.00661390148183497),
                    (9.07856691110755, 0.010681057264421),
                    (7.64418368167015, 0.021881979367307),
                    (4.7824250943718, 0.0915186458951753),
                    (20.078462535067, 4.36533178816285e-05),
                    (14.0801205135569, 0.000876073771284616),
                ],
            ),
            (
                &["x1", "z"],
                1,
                [
                    (1.14428813461403, 0.284748043158543),
                    (1.83787825176763, 0.175199832831764),
                    (1.51093262045524, 0.218996831908603),
                    (1.014108147529, 0.313920661605621),
                    (3.63171340231645, 0.0566884270413592),
                    (1.06814571765199, 0.301364660153732),
                ],
            ),
        ];
        for (names, df, rows) in cases {
            for ((label, cov_type), (stat, p)) in all_cov_types(&c.time).into_iter().zip(rows) {
                let re = ReEstimator::fit(two_way_input(names), ReEffects::TwoWay, cov_type, 0.95)
                    .unwrap_or_else(|e| panic!("{names:?} {label}: {e}"));
                let tag = format!("{names:?} {label}");
                assert_eq!(re.hausman_df(), Some(df), "{tag}");
                assert_rel(re.hausman_statistic().unwrap(), stat, 1e-9, &tag);
                assert_rel(re.hausman_p_value().unwrap(), p, 1e-8, &tag);
            }
        }
    }

    #[test]
    fn two_way_without_regressors_has_zero_r_squared_and_no_hausman_test() {
        // 傾き係数が0個（定数項のみ）。Pythonからは`x=[]`が拒否されるためエンジンだけの経路。
        let c = two_way_columns();
        let input =
            ReInput::from_columns(&c.y, &[], vec![], &c.entity, Some(&c.time), "y".into()).unwrap();
        let re = ReEstimator::fit(input, ReEffects::TwoWay, ReCovType::Classical, 0.95).unwrap();
        assert_eq!(
            (
                re.r_squared_within(),
                re.r_squared_between(),
                re.r_squared_overall()
            ),
            (0.0, 0.0, 0.0)
        );
        assert_eq!(re.hausman_statistic(), None);
    }

    #[test]
    fn two_way_hausman_is_none_without_two_way_within_variation() {
        // 一次元でしか変わらない列だけなら、2-way within変換後の`X̃`が0列で検定できない。
        let re = ReEstimator::fit(
            two_way_input(&["xi", "xt"]),
            ReEffects::TwoWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        assert_eq!(re.hausman_statistic(), None);
        assert_eq!(re.hausman_p_value(), None);
        assert_eq!(re.hausman_df(), None);
        // within R²は1-wayと同じく0（within変換後の当てはめ値が恒等的に0）。
        assert_eq!(re.r_squared_within(), 0.0);
    }

    #[test]
    fn two_way_hausman_leaves_out_a_column_that_is_its_own_within_transform() {
        // 個体平均も時点平均も0の交互作用列`(e - ē)(t - t̄)`は、準偏差変換しても変わらず
        // （`x* = x̃`）、補助回帰で識別できない。検定対象から外し、`x1`だけを検定する。
        let c = two_way_columns();
        let interaction: Vec<f64> = (0..24)
            .map(|r| ((r / 4) as f64 - 2.5) * ((r % 4) as f64 - 1.5))
            .collect();
        let input = ReInput::from_columns(
            &c.y,
            &[c.x1.clone(), interaction],
            strings(&["x1", "z"]),
            &c.entity,
            Some(&c.time),
            "y".into(),
        )
        .unwrap();
        let re = ReEstimator::fit(input, ReEffects::TwoWay, ReCovType::Classical, 0.95).unwrap();
        assert_eq!(re.hausman_df(), Some(1));
        assert!(re.hausman_statistic().unwrap().is_finite());
    }

    #[test]
    fn hausman_within_column_is_aliased_only_when_the_difference_is_constant() {
        let within = [-1.0, 1.0, -2.0, 2.0];
        // x* = x̃ + 定数: 識別できない。
        let aliased: Vec<f64> = within.iter().map(|v| v + 0.37).collect();
        assert!(hausman_within_column_is_aliased(&aliased, &within, 2.0));
        // 差が個体ごとに違う（θ・平均が個体間で異なる）: 識別できる。
        let identified = [-1.0 + 0.3, 1.0 + 0.3, -2.0 + 0.9, 2.0 + 0.9];
        assert!(!hausman_within_column_is_aliased(&identified, &within, 2.9));
    }

    #[test]
    fn re_estimator_fit_hausman_matches_plm_aux_on_balanced_panel() {
        // バランスパネル（4エンティティ×3期間）ではSwamy-Arora分散成分が`plm`と一致し、
        // 補助回帰の値も1e-9で一致する。R: `plm::phtest(y ~ x1 + x2, data, method = "aux")`
        //   chisq = 0.076499376991569779, p = 0.96247259255247208, df = 2
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0, 4.0, 3.0, 7.0, 2.0, 5.0];
        let x2 = vec![2.0, 1.0, 5.0, 1.0, 4.0, 3.0, 2.0, 6.0, 5.0, 3.0, 4.0, 1.0];
        let y = [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0, 5.0, 9.5, 4.0, 7.3];
        let input = ReInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

        assert_eq!(re.hausman_df(), Some(2));
        assert!((re.hausman_statistic().unwrap() - 0.076_499_376_991_569_78).abs() < 1e-9);
        assert!((re.hausman_p_value().unwrap() - 0.962_472_592_552_472_1).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_hausman_is_close_to_plm_aux_on_unbalanced_panel() {
        // 不均衡パネルでは`plm`と分散成分（σ_u²）の推定式が僅かに異なる（`linearmodels`準拠の

        // R: `plm::phtest(y ~ x1 + x2, data, method = "aux")`
        //   chisq = 0.60715047844214221, p = 0.73817434737945442, df = 2
        let re = ReEstimator::fit(
            hausman_two_slope_input(None),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();

        assert_eq!(re.hausman_df(), Some(2));
        assert!((re.hausman_statistic().unwrap() - 0.607_150_478_442_142_2).abs() < 5e-2);
        assert!((re.hausman_p_value().unwrap() - 0.738_174_347_379_454_4).abs() < 5e-2);
    }

    #[test]
    fn re_estimator_fit_hausman_matches_manual_auxiliary_regression() {
        // 補助回帰の独立な手計算（`OlsEstimator`を直接使い、準偏差変換・within変換を
        // 自前で組み立てる）との一致。単一傾き・不均衡パネル（`cov_type_reference_input`）。
        let re = ReEstimator::fit(
            cov_type_reference_input(),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        let input = re.input();
        let VarianceComponents {
            sigma2_eps,
            sigma2_u,
            ..
        } = swamy_arora_variance_components(input, ReEffects::OneWay, 0.95).unwrap();
        let (_, y_star, x_star) = quasi_demean_transform(input, sigma2_eps, sigma2_u);
        let entity = input.entity_codes().clone();
        let x_within = quasi_demean_column(&input.x()[0], &entity, &vec![1.0; entity.n_groups()]);

        let aux_input = OlsInput::from_columns(
            &y_star,
            &[x_star[0].clone(), x_within],
            vec!["x1".to_string(), "x1_within".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let aux = OlsEstimator::fit(aux_input, CovType::Classical, 0.95).unwrap();
        // 1自由度のWaldは`t²`（`wald_test_last_columns`のdocコメント参照）。
        let t = *aux.test_stats().get(2, 0);
        let expected_stat = t * t;
        let expected_p = ChiSquared::new(1.0).unwrap().sf(expected_stat);

        assert_eq!(re.hausman_df(), Some(1));
        assert!((re.hausman_statistic().unwrap() - expected_stat).abs() < 1e-9);
        assert!((re.hausman_p_value().unwrap() - expected_p).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_hausman_classical_is_invariant_to_time() {
        // 1-wayでは`ReInput`の`time`の有無はハウスマン検定の結果を変えない（`time`は2-wayの時間次元で、DKの時点列は`Dk.time`）。
        let time = strings(&["1", "2", "3", "1", "2", "1", "2", "3", "1", "2", "1", "2"]);
        let base = ReEstimator::fit(
            hausman_two_slope_input(None),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        let with_time = ReEstimator::fit(
            hausman_two_slope_input(Some(&time)),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        assert_eq!(with_time.hausman_statistic(), base.hausman_statistic());
        assert_eq!(with_time.hausman_p_value(), base.hausman_p_value());
        assert_eq!(with_time.hausman_df(), base.hausman_df());
    }

    /// 補助回帰（`y*`を定数項・`X*`・`X̃`に回帰）を手で組み立てる。`hausman_two_slope_input`用。
    fn manual_hausman_aux(input: &ReInput, cov_type: CovType) -> OlsEstimator {
        let VarianceComponents {
            sigma2_eps,
            sigma2_u,
            ..
        } = swamy_arora_variance_components(input, ReEffects::OneWay, 0.95).unwrap();
        let (_, y_star, x_star) = quasi_demean_transform(input, sigma2_eps, sigma2_u);
        let entity = input.entity_codes().clone();
        let theta = vec![1.0; entity.n_groups()];
        let mut columns = x_star;
        for col in input.x() {
            columns.push(quasi_demean_column(col, &entity, &theta));
        }
        let names = ["x1", "x2", "x1_w", "x2_w"].map(String::from).to_vec();
        let aux_input = OlsInput::from_columns(&y_star, &columns, names, true, "y".into()).unwrap();
        OlsEstimator::fit(aux_input, cov_type, 0.95).unwrap()
    }

    #[test]
    fn re_estimator_fit_hausman_matches_fresh_within_transform_exactly() {
        // `re_hausman_test`の`X̃`は内部FEのwithin変換済み設計行列を再利用する。RE本体の入力に
        // within変換をかけ直して組んだ補助回帰（`manual_hausman_aux`、列順も同じ）と
        // ビット単位で一致する（不均衡パネル・2傾き）。
        let re = ReEstimator::fit(
            hausman_two_slope_input(None),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        let aux = manual_hausman_aux(re.input(), CovType::Classical);
        let (f_stat, _) = aux.wald_test_last_columns(2).unwrap();
        assert_eq!(re.hausman_statistic(), Some(2.0 * f_stat));
    }

    #[test]
    fn re_estimator_fit_hausman_follows_cov_type_for_ols_covariances() {
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c", "c", "d", "d", "e", "e"]);
        let classical = ReEstimator::fit(
            hausman_two_slope_input(None),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        let cases = [
            (ReCovType::Hc1, CovType::Hc1),
            (ReCovType::Hc2, CovType::Hc2),
            (ReCovType::Hc3, CovType::Hc3),
            (
                ReCovType::Cluster { groups: None },
                CovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&entity)),
                },
            ),
        ];
        for (re_cov, ols_cov) in cases {
            let label = format!("{re_cov:?}");
            let re = ReEstimator::fit(
                hausman_two_slope_input(None),
                ReEffects::OneWay,
                re_cov,
                0.95,
            )
            .unwrap();
            let aux = manual_hausman_aux(re.input(), ols_cov);
            let (f_stat, _) = aux.wald_test_last_columns(2).unwrap();
            let expected_stat = 2.0 * f_stat;
            let expected_p = ChiSquared::new(2.0).unwrap().sf(expected_stat);

            // `OlsEstimator`経由の補助回帰とビット単位で一致する（Clusterは`OlsEstimator`の
            // `String`列グループ化ではなく整数コードの`panel_cluster_cov_params`を使うが、
            // 補正式・加算順は同じ）。
            assert_eq!(re.hausman_df(), Some(2), "{label}");
            assert_eq!(re.hausman_statistic(), Some(expected_stat), "{label}");
            assert_eq!(re.hausman_p_value(), Some(expected_p), "{label}");
            assert_ne!(
                re.hausman_statistic(),
                classical.hausman_statistic(),
                "{label}"
            );
        }
    }

    #[test]
    fn re_estimator_fit_hausman_cluster_uses_explicit_groups() {
        // `groups`を明示するとentityではなくその列でクラスタリングする。
        let groups = strings(&[
            "g1", "g1", "g1", "g2", "g2", "g3", "g3", "g3", "g4", "g4", "g5", "g5",
        ]);
        let groups_by_obs = strings(&["1", "2", "3", "4", "5", "1", "2", "3", "4", "5", "1", "2"]);
        let re = ReEstimator::fit(
            hausman_two_slope_input(None),
            ReEffects::OneWay,
            ReCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups_by_obs)),
            },
            0.95,
        )
        .unwrap();
        let by_entity = ReEstimator::fit(
            hausman_two_slope_input(None),
            ReEffects::OneWay,
            ReCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
            0.95,
        )
        .unwrap();
        let aux = manual_hausman_aux(
            re.input(),
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups_by_obs)),
            },
        );
        let expected = 2.0 * aux.wald_test_last_columns(2).unwrap().0;
        assert_eq!(re.hausman_statistic(), Some(expected));
        assert_ne!(re.hausman_statistic(), by_entity.hausman_statistic());
    }

    #[test]
    fn re_estimator_fit_hausman_dk_with_zero_bandwidth_equals_time_clustered_aux() {
        // Dkのバンド幅0は、時点でクラスタリングした補助回帰（fixest型`K=k_aux`・
        // `G=t_periods`スケール）と代数的に同値。
        let time = strings(&["1", "2", "3", "1", "2", "1", "2", "3", "1", "2", "1", "2"]);
        let re = ReEstimator::fit(
            hausman_two_slope_input(Some(&time)),
            ReEffects::OneWay,
            ReCovType::Dk {
                bandwidth: Some(0),
                time: TimeKeys::lexicographic(time.clone()),
            },
            0.95,
        )
        .unwrap();
        // 2-wayではないが`K=k_aux`（fixestのDKは常にフルカウント）・`G=t_periods`の
        // 時点クラスターは`panel_cluster_cov_params`と一致するため、その経路で
        // 独立に検算する。
        let aux = manual_hausman_aux(re.input(), CovType::Classical);
        let x_mat = aux.input().x();
        let n = aux.input().nobs();
        let k_aux = aux.input().k();
        let xtx_inv = xtx_inverse(x_mat).unwrap();
        let residuals: Vec<f64> = (0..n).map(|i| *aux.residuals().get(i, 0)).collect();
        let time_codes = GroupCodes::from_labels(&time);
        let cov =
            panel_cluster_cov_params(x_mat, &residuals, &xtx_inv, n, k_aux, &time_codes, k_aux);
        let t_periods = time_codes.n_groups();
        let (f_stat, _) = wald_f_test(aux.params(), &cov, k_aux - 2, 2, t_periods - 1).unwrap();

        assert_eq!(re.hausman_df(), Some(2));
        assert!((re.hausman_statistic().unwrap() - 2.0 * f_stat).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_hausman_is_none_when_no_slope_regressors() {
        // 比較対象の傾き係数が0個（`x=[]`、定数項のみのモデル）の場合はNone
        // （`re.rs`のモジュールdoc「ハウスマン検定」参照）。`time`の有無によらない。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let time = strings(&["1", "2", "1", "2", "1", "2"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        for time in [None, Some(time.as_slice())] {
            let input = ReInput::from_columns(&y, &[], vec![], &entity, time, "y".into()).unwrap();
            let re =
                ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

            assert_eq!(re.hausman_statistic(), None);
            assert_eq!(re.hausman_p_value(), None);
            assert_eq!(re.hausman_df(), None);
        }
    }

    #[test]
    fn re_hausman_test_returns_none_when_auxiliary_regression_is_rank_deficient() {
        // 補助回帰のランク落ち→None（`re.rs`のモジュールdoc「ハウスマン検定」参照）。
        // `ReEstimator::fit`経由では内部FE推定・RE本体が先に失敗するため自然な入力から
        // 再現しにくい。private関数`re_hausman_test`を直接呼び、準偏差変換済み側の
        // 列にwithin変換済み`x1`と同一の列を含めて完全共線にする。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let fe_input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();
        let fe = FeEstimator::fit(fe_input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();
        // `x*`に同じ列を2回入れて、補助回帰の設計行列を（`x̃`の識別可否とは無関係に）
        // 完全共線にする。
        let x_star = vec![fe.input().x()[0].clone(), fe.input().x()[0].clone()];

        let result = re_hausman_test(
            &fe,
            &[0],
            &y,
            &x_star,
            &["x1_star".to_string(), "x1_star2".to_string()],
            "y",
            &ReCovType::Classical,
            &GroupCodes::from_labels(&entity),
            0.95,
        );

        assert!(matches!(result, Err(PanelError::HausmanTestFailed { .. })));
    }

    #[test]
    fn re_estimator_fit_hausman_returns_error_when_clusters_do_not_cover_auxiliary_slopes() {
        // RE本体は`G=3 > q=2`で成功するが、補助回帰の傾き係数は`2k=4 >= G`のため
        // ロバスト共分散が構造的に特異になり、`fit()`は`HausmanTestFailed`で失敗する。
        let groups = strings(&[
            "g1", "g1", "g1", "g2", "g2", "g3", "g3", "g3", "g1", "g2", "g3", "g3",
        ]);
        let result = ReEstimator::fit(
            hausman_two_slope_input(None),
            ReEffects::OneWay,
            ReCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
            0.95,
        );
        assert!(matches!(
            result.unwrap_err(),
            PanelError::HausmanTestFailed {
                source: LeastSquaresError::Common(CommonError::InsufficientClustersForInference {
                    g: 3,
                    q: 4
                })
            }
        ));
    }

    #[test]
    fn re_estimator_fit_hausman_dk_default_bandwidth_matches_auto_resolved_value() {
        // `bandwidth: None`は`floor(4*(T/100)^(2/9))`（T=3で1）に解決され、
        // 明示的に同じ値を渡した場合と一致する。
        let time = strings(&["1", "2", "3", "1", "2", "1", "2", "3", "1", "2", "1", "2"]);
        let auto = ReEstimator::fit(
            hausman_two_slope_input(Some(&time)),
            ReEffects::OneWay,
            ReCovType::Dk {
                bandwidth: None,
                time: TimeKeys::lexicographic(time.clone()),
            },
            0.95,
        )
        .unwrap();
        let explicit = ReEstimator::fit(
            hausman_two_slope_input(Some(&time)),
            ReEffects::OneWay,
            ReCovType::Dk {
                bandwidth: Some(1),
                time: TimeKeys::lexicographic(time.clone()),
            },
            0.95,
        )
        .unwrap();
        assert_eq!(auto.hausman_statistic(), explicit.hausman_statistic());
        assert_eq!(auto.hausman_p_value(), explicit.hausman_p_value());
    }

    #[test]
    fn re_estimator_fit_returns_error_when_dk_periods_do_not_cover_hausman_slopes() {
        // 時点数`T=2`ではDK共分散のrankが`T-1=1`で、全体F検定の傾き係数`q=2`個の
        // 部分行列が構造的に特異になる。補助回帰の数値的な特異性判定を待たず、入力から
        // 判定できる`InsufficientDkPeriodsForInference`で弾く（`T=3 > k=2`で成功する
        // 境界は`re_estimator_fit_hausman_dk_default_bandwidth_matches_auto_resolved_value`）。
        let time = strings(&["1", "2", "1", "2", "1", "2", "1", "2", "1", "2", "1", "2"]);
        let result = ReEstimator::fit(
            hausman_two_slope_input(Some(&time)),
            ReEffects::OneWay,
            ReCovType::Dk {
                bandwidth: Some(0),
                time: TimeKeys::lexicographic(time.clone()),
            },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            PanelError::InsufficientDkPeriodsForInference { t_periods: 2, q: 2 }
                .in_stage(Stage::OverallFTest)
        );
    }
}
