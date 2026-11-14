//! C統計量（difference-in-Hansen統計量、GMM distance test）の計算コア。
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
//!
//! IVのGMM（`iv::gmm::compute_c_statistic`）と固定効果つきGMM（within変換後の行列を渡す）が共有する。
//! 行列だけを受け取り、`IvInput`・系統のエラー型には依存しない。倍率`s_scale`と完全適合の判定の受け渡しは
//! このファイルの`mod tests`で、推定量を通すテストは`iv/gmm/c_statistic/tests/`と
//! `panel/fe_iv/gmm/fit/tests/c_statistic.rs`にある。

use faer::Mat;
use statrs::distribution::{ChiSquared, ContinuousCDF};

use super::covariance::{resolve_hac_lags, time_ordering};
use super::error::{ClusterSetting, CommonError};
use super::gmm::hansen_j_statistic;
use super::gmm_iteration::gmm_point_estimate;
use super::gmm_weight::{WeightType, moment_covariance};
use super::group_codes::GroupCodes;
use super::validation::validate_cluster_groups;

/// 検定するモーメント条件。添字は昇順・重複なしで、`x_endog`・`instruments`の列の位置を指す。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CStatisticTarget {
    /// 外生性を検定する内生変数（`x_endog`の列の位置）。
    pub endog: Vec<usize>,
    /// 直交性を検定する除外操作変数（`instruments`の列の位置）。
    pub instruments: Vec<usize>,
}

/// C統計量の結果。
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

/// [`gmm_c_statistic`]に渡す行列。列の役割は`IvInput`と同じ（固定効果つきGMMではwithin変換後の列）。
pub(crate) struct CStatisticMatrices<'a> {
    /// 被説明変数（n, 1）。
    pub y: &'a Mat<f64>,
    /// 外生の説明変数（n, k_exog）。
    pub x_exog: &'a Mat<f64>,
    /// 内生変数（n, k_endog）。
    pub x_endog: &'a Mat<f64>,
    /// 除外操作変数（n, q）。
    pub instruments: &'a Mat<f64>,
}

/// [`gmm_c_statistic`]の系統ごとの設定。
pub(crate) struct CStatisticSettings<'a> {
    /// `S_e`の種類（`gmm_type`・`cov_type`には依存しない）。
    pub weight: &'a WeightType,
    /// `S_e`にかける正の倍率。`1.0`なら補正しない。点推定は重みのスカラー倍で変わらず、`J_e`・`J_c`だけが
    /// `1 / s_scale`倍になる（固定効果つきGMMが`Classical`の`S`の`σ̂²`の分母を`n − neffects`にするために使う）。
    pub s_scale: f64,
    /// 初期推定（`Z_full`の1段階GMM）の係数と残差の平方和から、構造残差が完全適合か。`S_e`が0に近い
    /// 行列になり、Jが丸め誤差から作られた無意味な値になる前に未定義とするため、系統ごとの判定を渡す。
    pub is_perfect_fit: &'a dyn Fn(&Mat<f64>, f64) -> bool,
}

/// C統計量を計算する。定義は上のモジュールdoc。
///
/// # Errors
///
/// - `CommonError::CStatisticNothingToTest`: `target`が空。
/// - `CommonError::CStatisticTooManyTestedInstruments`: 外した後の除外操作変数が内生変数の数より少ない。
/// - `CommonError::CStatisticUndefined`: 必要な行列が特異（操作変数の共線、完全フィット、モーメント
///   共分散のランク落ち。クラスター数が`Z_full`の列数に足りない場合を含む）。
/// - `weight`の設定の誤り（クラスター列が無い・行数が合わない・HACのラグ数が範囲外）。HACの
///   `time_order`の長さは`n`と一致している前提（`engine_pybind`が保証する。`gmm_iterate`と同じ）。
/// - `target`の添字が範囲外・重複（呼び出し側の誤りで、`CommonError::Internal`）。
pub(crate) fn gmm_c_statistic(
    matrices: &CStatisticMatrices<'_>,
    target: &CStatisticTarget,
    settings: &CStatisticSettings<'_>,
) -> Result<CStatistic, CommonError> {
    super::parallelism::ensure_serial();

    let CStatisticMatrices {
        y,
        x_exog,
        x_endog,
        instruments,
    } = matrices;
    let n = y.nrows();
    let (k_exog, k_endog, q) = (x_exog.ncols(), x_endog.ncols(), instruments.ncols());
    let df = target.endog.len() + target.instruments.len();
    if df == 0 {
        return Err(CommonError::CStatisticNothingToTest);
    }
    validate_positions(&target.endog, k_endog, "x_endog")?;
    validate_positions(&target.instruments, q, "instruments")?;
    if q - target.instruments.len() < k_endog {
        return Err(CommonError::CStatisticTooManyTestedInstruments {
            n_tested: target.instruments.len(),
            n_instruments: q,
            n_endog: k_endog,
        });
    }

    // Z_full = [x_exog, B₁, instruments]。Z_cは、その列のうちB₂を除いたもの。
    let m_endog = target.endog.len();
    let z_full = Mat::from_fn(n, k_exog + m_endog + q, |i, j| {
        if j < k_exog {
            *x_exog.get(i, j)
        } else if j < k_exog + m_endog {
            *x_endog.get(i, target.endog[j - k_exog])
        } else {
            *instruments.get(i, j - k_exog - m_endog)
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
    let x = Mat::from_fn(n, k_exog + k_endog, |i, j| {
        if j < k_exog {
            *x_exog.get(i, j)
        } else {
            *x_endog.get(i, j - k_exog)
        }
    });

    let prepared = prepare_weight(settings.weight, n, z_full.ncols())?;
    let ztz = z_full.transpose() * &z_full;

    // 1. Z_fullで2段階GMM: 1段階目の残差からS_eを作り、J_eを求める。
    let beta0 = gmm_point_estimate(&z_full, &x, y, &ztz).map_err(undefined_if_singular)?;
    let residuals0 = *y - &x * &beta0;
    let ssr0: f64 = (0..n).map(|i| (*residuals0.get(i, 0)).powi(2)).sum();
    if (settings.is_perfect_fit)(&beta0, ssr0) {
        return Err(CommonError::CStatisticUndefined);
    }
    let mut s_e = moment_covariance(
        settings.weight,
        prepared.cluster,
        prepared.hac.as_ref(),
        &z_full,
        &residuals0,
        &ztz,
    )?;
    if settings.s_scale != 1.0 {
        s_e = Mat::from_fn(s_e.nrows(), s_e.ncols(), |i, j| {
            settings.s_scale * *s_e.get(i, j)
        });
    }
    let beta_e = gmm_point_estimate(&z_full, &x, y, &s_e).map_err(undefined_if_singular)?;
    let j_with_tested = hansen_j(&z_full, &(*y - &x * &beta_e), &s_e)?;

    // 2. S_eの部分行列S_cを重みにZ_cで推定し、J_cを求める。Z_cが丁度識別ならモーメント条件を
    //    すべて0にできるのでJ_c = 0（推定せず厳密に0を返す）。
    let j_without_tested = if kept.len() == x.ncols() {
        0.0
    } else {
        let s_c = Mat::from_fn(kept.len(), kept.len(), |a, b| *s_e.get(kept[a], kept[b]));
        let beta_c =
            gmm_point_estimate(&z_restricted, &x, y, &s_c).map_err(undefined_if_singular)?;
        hansen_j(&z_restricted, &(*y - &x * &beta_c), &s_c)?
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
fn validate_positions(positions: &[usize], len: usize, what: &str) -> Result<(), CommonError> {
    let valid = positions.windows(2).all(|w| w[0] < w[1]) && positions.iter().all(|&p| p < len);
    if valid {
        Ok(())
    } else {
        Err(CommonError::internal(format!(
            "the tested positions in {what} must be increasing and below {len}"
        )))
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
) -> Result<PreparedWeight<'_>, CommonError> {
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
                return Err(CommonError::CStatisticUndefined);
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
fn undefined_if_singular(err: CommonError) -> CommonError {
    match err {
        CommonError::NearSingularMatrix { .. } => CommonError::CStatisticUndefined,
        other => other,
    }
}

/// `shared::gmm::hansen_j_statistic`の結果が有限でないときや`S`が正定値でないときは未定義。
fn hansen_j(z: &Mat<f64>, residuals: &Mat<f64>, s: &Mat<f64>) -> Result<f64, CommonError> {
    hansen_j_statistic(z, residuals, s)
        .ok()
        .filter(|j| j.is_finite())
        .ok_or(CommonError::CStatisticUndefined)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    const N: usize = 40;

    /// 決定的な疑似データ。外生変数1個・内生変数1個・除外操作変数3個（内生変数は操作変数と誤差の両方に依存）。
    fn data() -> [Mat<f64>; 4] {
        let noise = |r: usize, k: f64| (r as f64 * k).sin();
        let instruments = Mat::from_fn(N, 3, |r, j| noise(r, 1.1 + 0.7 * j as f64));
        let x_exog = Mat::from_fn(N, 1, |r, _| noise(r, 0.37));
        let x_endog = Mat::from_fn(N, 1, |r, _| {
            0.6 * instruments.get(r, 0) + 0.5 * instruments.get(r, 1) + noise(r, 2.9)
        });
        let y = Mat::from_fn(N, 1, |r, _| {
            0.8 * x_exog.get(r, 0) + 0.5 * x_endog.get(r, 0) + noise(r, 4.3)
        });
        [y, x_exog, x_endog, instruments]
    }

    fn run(
        s_scale: f64,
        is_perfect_fit: &dyn Fn(&Mat<f64>, f64) -> bool,
        target: &CStatisticTarget,
    ) -> Result<CStatistic, CommonError> {
        let [y, x_exog, x_endog, instruments] = data();
        gmm_c_statistic(
            &CStatisticMatrices {
                y: &y,
                x_exog: &x_exog,
                x_endog: &x_endog,
                instruments: &instruments,
            },
            target,
            &CStatisticSettings {
                weight: &WeightType::Robust,
                s_scale,
                is_perfect_fit,
            },
        )
    }

    fn target(endog: &[usize], instruments: &[usize]) -> CStatisticTarget {
        CStatisticTarget {
            endog: endog.to_vec(),
            instruments: instruments.to_vec(),
        }
    }

    /// `S_e`を`c`倍すると、点推定は変わらず`J_e`・`J_c`・`C`が`1/c`倍になる。自由度は変わらない。
    #[test]
    fn scaling_s_divides_the_statistics_and_keeps_the_degrees_of_freedom() {
        let never = |_: &Mat<f64>, _: f64| false;
        for target in [target(&[0], &[]), target(&[], &[2]), target(&[0], &[1])] {
            let base = run(1.0, &never, &target).unwrap();
            let scaled = run(2.5, &never, &target).unwrap();
            assert!(base.j_with_tested > 0.0 && base.j_without_tested > 0.0);
            for (got, want, what) in [
                (scaled.j_with_tested, base.j_with_tested, "J_e"),
                (scaled.j_without_tested, base.j_without_tested, "J_c"),
                (scaled.statistic, base.statistic, "C"),
            ] {
                assert!(
                    (got * 2.5 - want).abs() <= 1e-9 * want.abs(),
                    "{target:?} {what}: {got} vs {want} / 2.5"
                );
            }
            assert_eq!(scaled.df, base.df);
        }
    }

    /// 完全適合の判定には、初期推定の係数（`k × 1`）と残差の平方和が渡され、`true`なら未定義になる。
    #[test]
    fn the_perfect_fit_check_receives_the_initial_estimate_and_makes_the_statistic_undefined() {
        let seen = Cell::new((0_usize, -1.0_f64));
        let always = |beta0: &Mat<f64>, ssr0: f64| {
            assert_eq!(beta0.ncols(), 1);
            seen.set((beta0.nrows(), ssr0));
            true
        };
        let err = run(1.0, &always, &target(&[], &[2])).unwrap_err();
        assert_eq!(err, CommonError::CStatisticUndefined);
        let (k, ssr0) = seen.get();
        assert_eq!(k, 2);
        assert!(ssr0 > 0.0 && ssr0.is_finite());
    }
}
