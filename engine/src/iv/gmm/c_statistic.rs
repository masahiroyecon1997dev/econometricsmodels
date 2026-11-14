//! C統計量（difference-in-Hansen統計量、GMM distance test）。
//!
//! 同じ構造方程式`y = Xβ + u`（`X = [x_exog, x_endog]`）を、検定するモーメント条件`B`を含む操作変数
//! の集合`Z_full`と、含まない集合`Z_c`の2通りで推定し、Hansen Jの差を`χ²(|B|)`で検定する。
//! `B`は次の2種類（同時に指定してよい）:
//!
//! - **内生変数の外生性**: `x_endog`の一部`B₁`を自分自身の操作変数として`Z_full`に加える
//!   （`Z_full`は`B₁`を外生変数として扱ったモデルの操作変数）。
//! - **除外操作変数の直交性**: `instruments`の一部`B₂`を`Z_c`から外す（`Z_full`は元の全操作変数）。
//!
//! `Z_full = [x_exog, B₁, instruments]`、`Z_c = [x_exog, instruments \ B₂]`で、`B = B₁ ∪ B₂`。
//!
//! 手順（Hayashi 2000, pp. 218-221 と 232-234、Baum-Schaffer-Stillman 2003 の4.4節。linearmodelsの
//! `IVGMMResults.c_stat`と同じ）:
//!
//! 1. `Z_full`で2段階GMMを推定する。1段階目の重みは`(Z_full'Z_full)⁻¹`、2段階目は1段階目の残差から
//!    作ったモーメント共分散`S_e`（`weight`が決める。`gmm_type`・`cov_type`には依存しない）の逆行列。
//!    このときのHansen Jが`J_e`。
//! 2. `Z_c`に対応する`S_e`の部分行列`S_c`を取り、`S_c⁻¹`を重みに`Z_c`でGMM推定する（`S_e⁻¹`の
//!    部分行列ではない）。このときのHansen Jが`J_c`。
//! 3. `C = J_e − J_c`を`χ²(|B|)`で検定する。
//!
//! 両方のJが同じ`S_e`から作られるので`C ≥ 0`が保証される（丸め誤差で負になる分は0に丸める）。
//! `Z_c`が丁度識別なら`J_c = 0`で、`C = J_e`。`S`は`n`で割らない生の和で、Jも`n`で割らない
//! （`iv/CLAUDE.md`参照）。

use faer::Mat;
use statrs::distribution::{ChiSquared, ContinuousCDF};

use crate::iv::common::{IvError, IvInput, hcat, mat_column_to_vec};
use crate::linear::ols::CovType;
use crate::shared::covariance::{resolve_hac_lags, time_ordering};
use crate::shared::error::{ClusterSetting, CommonError};
use crate::shared::gmm::hansen_j_statistic;
use crate::shared::gmm_iteration::gmm_point_estimate;
use crate::shared::gmm_weight::{WeightType, moment_covariance};
use crate::shared::goodness_of_fit::{is_perfect_fit, total_sum_of_squares};
use crate::shared::group_codes::GroupCodes;
use crate::shared::validation::validate_cluster_groups;

/// 検定するモーメント条件。添字は昇順・重複なしで、`IvInput`の列の位置を指す。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CStatisticTarget {
    /// 外生性を検定する内生変数（`x_endog`の列の位置）。
    pub endog: Vec<usize>,
    /// 直交性を検定する除外操作変数（`instruments`の列の位置）。
    pub instruments: Vec<usize>,
}

/// [`compute_c_statistic`]の結果。
#[derive(Debug, Clone, PartialEq)]
pub struct CStatistic {
    /// C統計量`J_e − J_c`。
    pub statistic: f64,
    /// `χ²(df)`の上側確率。
    pub p_value: f64,
    /// 自由度（検定するモーメント条件の数。内生変数と操作変数の数の和）。
    pub df: usize,
    /// 検定するモーメント条件を含むモデルのHansen J（`J_e`）。
    pub j_with_tested: f64,
    /// 検定するモーメント条件を含まないモデルのHansen J（`J_c`）。丁度識別なら0。
    pub j_without_tested: f64,
}

/// 2SLSの`cov_type`に対応するGMMの重みの種別。2SLSには重みの選択が無いので、`cov_type`の系統から
/// 決める: `Classical`は`Classical`、`Hc0`〜`Hc3`は`Robust`（`S`にHC1・HC2・HC3の補正は無い）、
/// `Hac`と`Cluster`はそのまま。
#[must_use]
pub fn weight_type_for_cov_type(cov_type: &CovType) -> WeightType {
    match cov_type {
        CovType::Classical => WeightType::Classical,
        CovType::Hc0 | CovType::Hc1 | CovType::Hc2 | CovType::Hc3 => WeightType::Robust,
        CovType::Hac { lags, time_order } => WeightType::Hac {
            lags: *lags,
            time_order: time_order.clone(),
        },
        CovType::Cluster { groups } => WeightType::Cluster {
            groups: groups.clone(),
        },
    }
}

/// C統計量を計算する。
///
/// 定義は上のモジュールdoc。`weight`は`S_e`の種類で、GMMでは`fit()`の`gmm_weight_type`、2SLSでは
/// [`weight_type_for_cov_type`]。`fit()`の`gmm_type`・`gmm_max_iter`・`cov_type`には依存しない。
///
/// # Errors
///
/// - `IvError::CStatisticNothingToTest`: `target`が空。
/// - `IvError::CStatisticTooManyTestedInstruments`: 外した後の除外操作変数が内生変数の数より少ない。
/// - `IvError::CStatisticUndefined`: 必要な行列が特異（操作変数の共線、完全フィット、モーメント
///   共分散のランク落ち。クラスター数が`Z_full`の列数に足りない場合を含む）。
/// - `weight`の設定の誤り（クラスター列が無い・行数が合わない・HACのラグ数が範囲外）。HACの
///   `time_order`の長さは`n`と一致している前提（`engine_pybind`が保証する。`gmm_iterate`と同じ）。
/// - `target`の添字が範囲外・重複（呼び出し側の誤りで、`INTERNAL_ERROR`）。
pub fn compute_c_statistic(
    input: &IvInput,
    target: &CStatisticTarget,
    weight: &WeightType,
) -> Result<CStatistic, IvError> {
    crate::shared::parallelism::ensure_serial();

    let (n, k_exog, k_endog, q) = (
        input.nobs(),
        input.k_exog(),
        input.k_endog(),
        input.k_instruments(),
    );
    let df = target.endog.len() + target.instruments.len();
    if df == 0 {
        return Err(IvError::CStatisticNothingToTest);
    }
    validate_positions(&target.endog, k_endog, "x_endog")?;
    validate_positions(&target.instruments, q, "instruments")?;
    if q - target.instruments.len() < k_endog {
        return Err(IvError::CStatisticTooManyTestedInstruments {
            n_tested: target.instruments.len(),
            n_instruments: q,
            n_endog: k_endog,
        });
    }

    // Z_full = [x_exog, B₁, instruments]。Z_cは、その列のうちB₂を除いたもの。
    let m_endog = target.endog.len();
    let z_full = Mat::from_fn(n, k_exog + m_endog + q, |i, j| {
        if j < k_exog {
            *input.x_exog().get(i, j)
        } else if j < k_exog + m_endog {
            *input.x_endog().get(i, target.endog[j - k_exog])
        } else {
            *input.instruments().get(i, j - k_exog - m_endog)
        }
    });
    let kept: Vec<usize> = (0..k_exog)
        .chain(
            (0..q)
                .filter(|j| !target.instruments.contains(j))
                .map(|j| k_exog + m_endog + j),
        )
        .collect();
    let z_restricted = Mat::from_fn(n, kept.len(), |i, a| *z_full.get(i, kept[a]));
    let x = hcat(input.x_exog(), input.x_endog());
    let y = input.y();

    let prepared = prepare_weight(weight, n, z_full.ncols())?;
    let ztz = z_full.transpose() * &z_full;

    // 1. Z_fullで2段階GMM: 1段階目の残差からS_eを作り、J_eを求める。
    let beta0 = gmm_point_estimate(&z_full, &x, y, &ztz).map_err(undefined_if_singular)?;
    let residuals0 = y - &x * &beta0;
    // 初期推定の残差が丸め誤差の範囲で0なら完全フィット。`S_e`が0に近い行列になり、Jが丸め誤差から
    // 作られた無意味な値になる前に未定義とする（`GmmEstimator::fit`と同じ判定）。
    let sst = total_sum_of_squares(&mat_column_to_vec(y, 0), None, input.has_intercept());
    let ssr0: f64 = (0..n).map(|i| (*residuals0.get(i, 0)).powi(2)).sum();
    if is_perfect_fit(ssr0, sst, x.ncols()) {
        return Err(IvError::CStatisticUndefined);
    }
    let s_e = moment_covariance(
        weight,
        prepared.cluster,
        prepared.hac.as_ref(),
        &z_full,
        &residuals0,
        &ztz,
    )?;
    let beta_e = gmm_point_estimate(&z_full, &x, y, &s_e).map_err(undefined_if_singular)?;
    let j_with_tested = hansen_j(&z_full, &(y - &x * &beta_e), &s_e)?;

    // 2. S_eの部分行列S_cを重みにZ_cで推定し、J_cを求める。Z_cが丁度識別ならモーメント条件を
    //    すべて0にできるのでJ_c = 0（推定せず厳密に0を返す）。
    let j_without_tested = if kept.len() == x.ncols() {
        0.0
    } else {
        let s_c = Mat::from_fn(kept.len(), kept.len(), |a, b| *s_e.get(kept[a], kept[b]));
        let beta_c =
            gmm_point_estimate(&z_restricted, &x, y, &s_c).map_err(undefined_if_singular)?;
        hansen_j(&z_restricted, &(y - &x * &beta_c), &s_c)?
    };

    // 3. 共通のSなので差は理論上非負。丸め誤差の分だけ負になりうる。
    let statistic = (j_with_tested - j_without_tested).max(0.0);
    // `df >= 1`（上で空の`target`を弾いている）なので`ChiSquared::new`は失敗しない。
    let p_value = ChiSquared::new(df as f64)
        .map_err(|_| {
            CommonError::internal("the degrees of freedom of the C statistic are invalid")
        })?
        .sf(statistic);
    Ok(CStatistic {
        statistic,
        p_value,
        df,
        j_with_tested,
        j_without_tested,
    })
}

/// 添字が範囲内・昇順・重複なしであること（呼び出し側が保証する前提の検査）。
fn validate_positions(positions: &[usize], len: usize, what: &str) -> Result<(), IvError> {
    let valid = positions.windows(2).all(|w| w[0] < w[1]) && positions.iter().all(|&p| p < len);
    if valid {
        Ok(())
    } else {
        Err(CommonError::internal(format!(
            "the tested positions in {what} must be increasing and below {len}"
        ))
        .into())
    }
}

/// `S`の組み立てに要る前処理（`weight`に応じたクラスターのコードか、HACのラグ数と時系列順）。
struct PreparedWeight<'a> {
    cluster: Option<&'a GroupCodes>,
    hac: Option<(usize, Vec<usize>)>,
}

/// `weight`から前処理を1度だけ作る（`time_order`の長さは`n`と一致している前提）。
///
/// クラスターは`S_e`（`l_full`×`l_full`）が`rank(S_e) ≤ G`なので、`G < l_full`は構造的に特異で、
/// 数値的な特異の検出に任せず未定義にする（`Z_full`は常に過剰識別なので丁度識別の補正は要らない）。
fn prepare_weight(
    weight: &WeightType,
    n: usize,
    l_full: usize,
) -> Result<PreparedWeight<'_>, IvError> {
    let mut prepared = PreparedWeight {
        cluster: None,
        hac: None,
    };
    match weight {
        WeightType::Classical | WeightType::Robust => {}
        WeightType::Cluster { groups } => {
            let groups = groups.as_ref().ok_or(CommonError::MissingClusterColumn {
                setting: ClusterSetting::GmmWeightType,
            })?;
            let n_groups = validate_cluster_groups(groups, n, ClusterSetting::GmmWeightType)?;
            if n_groups < l_full {
                return Err(IvError::CStatisticUndefined);
            }
            prepared.cluster = Some(groups);
        }
        WeightType::Hac { lags, time_order } => {
            let lags = resolve_hac_lags(*lags, n)?;
            prepared.hac = Some((lags, time_ordering(time_order, n)));
        }
    }
    Ok(prepared)
}

/// 行列の特異性による失敗は、C統計量が定義できないことを表す。それ以外はそのまま返す。
fn undefined_if_singular(err: CommonError) -> IvError {
    match err {
        CommonError::NearSingularMatrix { .. } => IvError::CStatisticUndefined,
        other => other.into(),
    }
}

/// `shared::gmm::hansen_j_statistic`の結果が有限でないときや`S`が正定値でないときは未定義。
fn hansen_j(z: &Mat<f64>, residuals: &Mat<f64>, s: &Mat<f64>) -> Result<f64, IvError> {
    hansen_j_statistic(z, residuals, s)
        .ok()
        .filter(|j| j.is_finite())
        .ok_or(IvError::CStatisticUndefined)
}

#[cfg(test)]
mod tests;
