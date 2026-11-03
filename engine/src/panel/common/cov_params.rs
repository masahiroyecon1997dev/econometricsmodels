//! cov_type別の共分散行列（Classical・HC・Cluster・Driscoll-Kraay）の計算。FE固有の補正は
//! `fe_adjustment.rs`、cov_typeの振り分けは`panel_cov.rs`。

use faer::Mat;

use super::error::PanelError;
use crate::shared::covariance::{cluster_correction, cluster_meat, default_hac_lags};
use crate::shared::error::CommonError;
use crate::shared::group_codes::GroupCodes;

/// `(X̃'X̃)⁻¹`を求める（`X̃`はFE/REそれぞれの変換後の設計行列）。HC1〜HC3・Clusterいずれの
/// 計算でも共通して必要になる。`shared::covariance::xtx_inverse`と同じ発想だが、`OlsEstimator`が
/// 保持する`cov_params`はprivateで再利用できないため独立に計算し直す。
///
/// `X̃'X̃`が対称正定値であることは、`OlsEstimator::fit`が同じ`x`で既に成功している
/// （＝特異ではないと確認済み）ことから理論上保証されるが、`OlsEstimator`と同じく
/// 浮動小数点演算の境界的なケースに備えて`Result`化する。
pub(crate) fn xtx_inverse(x: &Mat<f64>) -> Result<Mat<f64>, PanelError> {
    crate::shared::covariance::xtx_inverse(x).map_err(|_| {
        CommonError::internal(
            "the Gram matrix of the transformed design matrix could not be inverted",
        )
        .into()
    })
}

/// classical: `σ̂² (X̃'X̃)⁻¹`（`σ̂² = SSR/df_resid`）。
pub(crate) fn panel_classical_cov_params(
    xtx_inv: &Mat<f64>,
    ssr: f64,
    df_resid: usize,
    k: usize,
) -> Mat<f64> {
    let sigma2 = ssr / (df_resid as f64);
    Mat::from_fn(k, k, |i, j| sigma2 * (*xtx_inv.get(i, j)))
}

/// `panel_hc_cov_params`が使うHCの種類（`shared::covariance::HcVariant`と同型だがHC0を含まない、
/// FE/REともにHC0はスコープ外、`fe.rs`モジュールdoc「`cov_type`対応」参照）。HC2/HC3は各行の
/// レバレッジ`h`（FEは`leverage_full`、REは`shared::covariance::leverages`の素のレバレッジ）を
/// バリアントに持つため、呼び出し側が`variant`と`h`の組み合わせを誤ることはない。
pub(crate) enum PanelHcVariant<'a> {
    Hc1,
    Hc2(&'a [f64]),
    Hc3(&'a [f64]),
}

/// HC1〜HC3の係数分散共分散行列（k×k）。`shared::covariance::hc_cov_params`と同型の構造だが、
/// 小標本補正がFE/RE用に異なる（`fe.rs`モジュールdoc「`cov_type`対応」参照）:
/// - HC1: `w_i = n/df_resid`
/// - HC2/HC3: `w_i`はバリアントが持つレバレッジ`h`ベース
pub(crate) fn panel_hc_cov_params(
    x: &Mat<f64>,
    residuals: &[f64],
    xtx_inv: &Mat<f64>,
    df_resid: usize,
    variant: PanelHcVariant<'_>,
) -> Mat<f64> {
    let n = x.nrows();
    let k = x.ncols();
    let hc1_correction = (n as f64 / df_resid as f64).sqrt();

    // 行ごとのスケールは列に依存しないため、行列の組み立て（n×kの二重ループ）の外で一度だけ求める。
    let row_scales: Vec<f64> = (0..n)
        .map(|i| {
            let resid = residuals[i];
            match &variant {
                PanelHcVariant::Hc1 => resid * hc1_correction,
                PanelHcVariant::Hc2(h) => resid / (1.0 - h[i]).sqrt(),
                PanelHcVariant::Hc3(h) => resid / (1.0 - h[i]),
            }
        })
        .collect();
    let x_scaled = Mat::from_fn(n, k, |i, j| row_scales[i] * (*x.get(i, j)));

    let psi_hat = x_scaled.transpose() * &x_scaled;
    xtx_inv * &psi_hat * xtx_inv
}

/// クラスターロバスト係数分散共分散行列（k×k）。`shared::covariance::cluster_cov_params`と同じ
/// Stata流`(G/(G-1))×((n-1)/(n-K))`小標本補正を使う（fixest（R）・Stataの`xtreg`/
/// `reghdfe`との数値一致のため、linearmodels方式`n/(n-extra_df-k)`から
/// 変更した。`fe.rs`モジュールdoc「`cov_type`対応」参照）。`G`はこの関数が
/// `groups`から数える（`shared::covariance::cluster_cov_params`と同じ）。`K`（`(n-1)/(n-K)`の分母）は
/// 呼び出し側が決める`k_correction`引数で渡す——FEは固定効果ダミーとクラスター変数の
/// ネスト関係で決まるfixest固有のK計算（`fe_cluster_k_correction`）、REは
/// 単純に`df_model`（FEのような固定効果ダミーのネスト補正が不要、`re.rs`のモジュールdoc参照）。
pub(crate) fn panel_cluster_cov_params(
    x: &Mat<f64>,
    residuals: &[f64],
    xtx_inv: &Mat<f64>,
    n: usize,
    k: usize,
    groups: &GroupCodes,
    k_correction: usize,
) -> Mat<f64> {
    let (s_hat, n_groups) = cluster_meat(groups, k, |i, a| residuals[i] * (*x.get(i, a)));
    let correction = cluster_correction(n_groups, n, k_correction);
    let cov_uncorrected = xtx_inv * &s_hat * xtx_inv;
    Mat::from_fn(k, k, |i, j| correction * (*cov_uncorrected.get(i, j)))
}

/// `FeCovType::Dk`/`ReCovType::Dk`の`bandwidth`（`Option<i64>`）を実際に使う
/// バンド幅（`usize`）に解決する。
///
/// `Some(bw)`の場合は`0 <= bw < t`を検証してそのまま使う（`t`はユニークな時点数）。`None`の
/// 場合は`linearmodels`の`DriscollKraay`と同じ経験則`floor(4*(t/100)^(2/9))`で自動計算する
/// （`fe.rs`モジュールdoc「Driscoll-Kraay型パネルHAC」参照。式は
/// `shared::covariance::default_hac_lags`でOLS・IVのHACと共有するが、引数が観測数`n`では
/// なく時点数`t`になる点が異なる）。
///
/// `t < 2`は`bandwidth`の値によらず`PanelError::InsufficientDkPeriods`で拒否する
/// （fixest型の`G/(G-1)`相当補正を導入したことに伴う前提条件。同エラーの
/// docコメント参照）。この関数がDK計算の全呼び出し経路（`fe/fit.rs`/`re/fit.rs`/
/// `re/hausman.rs`の`re_hausman_test`）の共通の入口になっているため、ここ一箇所で検証すれば足りる。
pub(crate) fn resolve_dk_bandwidth(bandwidth: Option<i64>, t: usize) -> Result<usize, PanelError> {
    if t < 2 {
        return Err(PanelError::InsufficientDkPeriods { t_periods: t });
    }
    match bandwidth {
        Some(bw) => {
            if bw < 0 || (bw as usize) >= t {
                return Err(PanelError::InvalidDkBandwidth { bandwidth: bw, t });
            }
            Ok(bw as usize)
        }
        None => Ok(default_hac_lags(t)),
    }
}

/// DKで同時Wald検定の対象となる係数の数`q`が、時点数`t_periods`で支えられることを
/// 検証する（`t_periods > q`）。理由は`PanelError::InsufficientDkPeriodsForInference`の
/// docコメント参照（`validate_cluster_count_covers_slopes`のDK版）。
///
/// `t_periods > q`でも係数間の悪条件で部分行列が数値的にほぼ特異になるケースは
/// 事前判定できないため、`wald_f_test`内の`ensure_well_conditioned_symmetric_matrix`が
/// backstopとして残る。`resolve_dk_bandwidth`（`t_periods >= 2`を保証）の後に呼ぶ前提で、
/// そのとき`q == 0`（検定対象なし）は常に`Ok`。
pub(crate) fn validate_dk_periods_cover_tested_coefficients(
    t_periods: usize,
    q: usize,
) -> Result<(), PanelError> {
    if t_periods <= q {
        return Err(PanelError::InsufficientDkPeriodsForInference { t_periods, q });
    }
    Ok(())
}

/// Driscoll-Kraay型パネルHAC共分散行列（k×k）。
///
/// `Ŝ = Σ_t ξ_t ξ_t' + Σ_{l=1}^{bandwidth} w_l (ξ_t ξ_{t-l}' + ξ_{t-l} ξ_t')`
/// （Bartlett重み`w_l = 1 - l/(bandwidth+1)`、`fe.rs`モジュールdoc参照）をまず求め、
/// 最後に`(t_periods/(t_periods-1)) × ((n-1)/(n-k_correction)) × (X̃'X̃)⁻¹ Ŝ (X̃'X̃)⁻¹`に
/// スケールする（fixestの`vcov="DK"`との数値一致のため、
/// linearmodels方式`n/df_resid`から変更した。fixestは時点数`t_periods`を
/// クラスター数`G`相当として扱い、`K.fixef="full"`が既定（クラスター変数が無いため
/// ネスト判定自体が発生しない）——`k_correction`は呼び出し側がFE/REそれぞれの
/// `df_model`をそのまま渡す。`panel_cluster_cov_params`の`G/(G-1)×(n-1)/(n-K)`と
/// 同型の式に、`G`を`t_periods`（`time`のユニークな時点数）に置き換えたものと理解できる）。
///
/// `time`のコード（`GroupCodes`、コード順＝時系列順、`TimeKeys`のdoc参照）で集計して
/// `ξ_t`（時点`t`でのクロスセクション和）を求める。
///
/// **`bandwidth < t_periods`が呼び出し元の`resolve_dk_bandwidth`から保証される**
/// （詳細は`fe.rs`モジュールdoc「Driscoll-Kraay型パネルHAC」参照）。上限の
/// `bandwidth == t_periods - 1`でも最終ラグ項`l = bandwidth`を含める標準のBartlettカーネル
/// で、この境界ではfixestと一致しない（`docs/spec/fe-spec.md`3.3節）。
///
/// ラグごとの`k×k`行列積（`xi_top.transpose() * xi_bot`等）は呼び出し元の`fit()`冒頭の
/// `ensure_serial()`が固定したグローバル`Par::Seq`に依存している（OLSの`hac_cov_params`が
/// `matmul(..., Par::Seq)`を明示するのと異なり、本関数は演算子オーバーロードを使うため
/// グローバル設定頼み。`panel_hc_cov_params`/`panel_cluster_cov_params`と同じ流儀）。
pub(crate) fn panel_driscoll_kraay_cov_params(
    x: &Mat<f64>,
    residuals: &[f64],
    xtx_inv: &Mat<f64>,
    time: &GroupCodes,
    k_correction: usize,
    bandwidth: usize,
) -> Mat<f64> {
    let n = x.nrows();
    let k = x.ncols();
    let t_periods = time.n_groups();
    let time_indices = time.group_indices();

    let mut xi = Mat::<f64>::zeros(t_periods, k);
    for (row, indices) in time_indices.iter().enumerate() {
        for &i in indices {
            let e = residuals[i];
            for col in 0..k {
                *xi.get_mut(row, col) += e * (*x.get(i, col));
            }
        }
    }

    // l=0項: Ŝ₀ = ξ'ξ（クラスターロバストのΨ̂と同形、時点をグループとみなした版）
    let mut s_hat = xi.transpose() * &xi;
    // l=1..=bandwidth項: w_l * (Ŝ_l + Ŝ_l')
    for l in 1..=bandwidth {
        let weight = 1.0 - (l as f64) / ((bandwidth + 1) as f64);
        let xi_top = xi.as_ref().subrows(l, t_periods - l);
        let xi_bot = xi.as_ref().subrows(0, t_periods - l);
        let s_l = xi_top.transpose() * xi_bot;
        for a in 0..k {
            for b in 0..k {
                *s_hat.get_mut(a, b) += weight * (*s_l.get(a, b) + *s_l.get(b, a));
            }
        }
    }

    let scale = (t_periods as f64 / (t_periods as f64 - 1.0))
        * ((n as f64 - 1.0) / ((n - k_correction) as f64));
    let cov_uncorrected = xtx_inv * &s_hat * xtx_inv;
    Mat::from_fn(k, k, |i, j| scale * (*cov_uncorrected.get(i, j)))
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;

    #[test]
    fn validate_dk_periods_cover_tested_coefficients_rejects_t_at_or_below_q() {
        // `rank(S) ≤ t-1`のため`t > q`が必要。`t == q`が境界（拒否）、`t == q + 1`は通す。
        assert_eq!(
            validate_dk_periods_cover_tested_coefficients(6, 6),
            Err(PanelError::InsufficientDkPeriodsForInference { t_periods: 6, q: 6 })
        );
        assert_eq!(
            validate_dk_periods_cover_tested_coefficients(6, 20),
            Err(PanelError::InsufficientDkPeriodsForInference {
                t_periods: 6,
                q: 20
            })
        );
        assert_eq!(validate_dk_periods_cover_tested_coefficients(6, 5), Ok(()));
        // 検定対象なし（FEの固定効果のみモデル）は常に通す。
        assert_eq!(validate_dk_periods_cover_tested_coefficients(2, 0), Ok(()));
    }

    // ── panel_driscoll_kraay_cov_params ────────────────────────────────────

    /// 1変数・各時点1観測の入力で、`bandwidth`ごとの未スケール`Ŝ`を標準Bartlettカーネルの
    /// 定義式どおりに素朴な二重ループで求める（`panel_driscoll_kraay_cov_params`の
    /// 行列積による実装とは独立な手計算）。`skip_last_lag`が真なら`l = bandwidth`の項を
    /// 落とす（fixestの`bandwidth == T-1`での挙動の再現）。
    fn naive_bartlett_meat(xi: &[f64], bandwidth: usize, skip_last_lag: bool) -> f64 {
        let t_periods = xi.len();
        let mut meat: f64 = xi.iter().map(|v| v * v).sum();
        for l in 1..=bandwidth {
            if skip_last_lag && l == bandwidth {
                continue;
            }
            let weight = 1.0 - (l as f64) / ((bandwidth + 1) as f64);
            let gamma: f64 = (l..t_periods).map(|t| xi[t] * xi[t - l]).sum();
            meat += 2.0 * weight * gamma;
        }
        meat
    }

    #[test]
    fn panel_cluster_cov_params_uses_k_for_the_meat_and_k_correction_for_the_scale() {
        // meatの次元は`k`（傾き係数の数）、小標本補正`(n-1)/(n-K)`の`K`は別引数`k_correction`。
        // 2つが取り違えられないよう、`k=1`・`k_correction=3`として手計算と比べる。
        // クラスター a={行0,行2}、b={行1,行3}、スコア s_i = e_i x_i。
        let x_values = [1.0, 2.0, -1.0, 0.5];
        let residuals = [0.5, -1.0, 2.0, 1.5];
        let x = Mat::from_fn(4, 1, |i, _| x_values[i]);
        let xtx_inv = Mat::from_fn(1, 1, |_, _| 0.25);
        let groups = GroupCodes::from_labels(&labels(&["a", "b", "a", "b"]));

        let cov = panel_cluster_cov_params(&x, &residuals, &xtx_inv, 4, 1, &groups, 3);

        let s_a = residuals[0] * x_values[0] + residuals[2] * x_values[2];
        let s_b = residuals[1] * x_values[1] + residuals[3] * x_values[3];
        let meat = s_a * s_a + s_b * s_b;
        let correction = (2.0 / 1.0) * ((4.0 - 1.0) / (4.0 - 3.0));
        let expected = correction * 0.25 * meat * 0.25;
        assert!((*cov.get(0, 0) - expected).abs() < 1e-12 * expected.abs());
    }

    #[test]
    fn panel_driscoll_kraay_cov_params_matches_naive_bartlett_for_every_bandwidth_up_to_t_minus_1()
    {
        // T=5の各時点に1観測を置き、`ξ_t = e_t·x_t`を既知の値にする。`(X̃'X̃)⁻¹ = 1`・
        // `K = 1`（スケールが`T/(T-1)`のみ）にして、`Ŝ`の定義式との直接比較にする。
        //
        // `bandwidth = T-1 = 4`（許容範囲の上限ちょうど）の最終ラグ項`l = 4`
        // （`ξ_4 ξ_0`、重み`1/5`）が含まれることを固定する。fixestの`cpp_driscoll_kraay`は
        // この境界で最大ラグを`T-2`に切り詰める（ラグの個数と最大ラグの取り違え）ため
        // 最終ラグ項を落とし、本実装とは一致しない。本実装は標準の定義どおり含める
        // （`docs/spec/fe-spec.md`3.3節）。`bandwidth <= T-2`ではfixestと一致する。
        let ids: Vec<String> = (0..5).map(|t| t.to_string()).collect();
        let time = GroupCodes::from_labels(&ids);
        let x_values = [1.0, -2.0, 0.5, 3.0, -1.5];
        let residuals = [0.7, 1.1, -0.4, 0.9, 2.0];
        let x = Mat::from_fn(5, 1, |i, _| x_values[i]);
        let xtx_inv = Mat::from_fn(1, 1, |_, _| 1.0);
        let xi: Vec<f64> = x_values
            .iter()
            .zip(&residuals)
            .map(|(x, e)| x * e)
            .collect();
        let scale = 5.0 / 4.0 * (4.0 / (5.0 - 1.0));

        for bandwidth in 0..=4 {
            let cov =
                panel_driscoll_kraay_cov_params(&x, &residuals, &xtx_inv, &time, 1, bandwidth);
            let expected = scale * naive_bartlett_meat(&xi, bandwidth, false);
            assert!(
                (*cov.get(0, 0) - expected).abs() < 1e-12 * expected.abs().max(1.0),
                "bandwidth={bandwidth}: got {}, expected {expected}",
                *cov.get(0, 0)
            );
        }

        // 境界で最終ラグ項が寄与していること（落とした値とは異なること）を確かめる。
        // これが成り立たないと上のループが境界での項の有無を区別できない。
        let full = naive_bartlett_meat(&xi, 4, false);
        let dropped = naive_bartlett_meat(&xi, 4, true);
        assert!((full - dropped).abs() > 1e-6);
    }
}
