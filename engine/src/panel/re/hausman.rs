//! ハウスマン検定（回帰ベース、`plm::phtest(method = "aux")`相当）。
//!
//! テストは`hausman/tests.rs`と`hausman/tests/`にテーマ別で置く。

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
mod tests;
