//! `cov_type`別の係数分散共分散行列（サンドイッチ型）と、Wald検定（F統計量）。
//!
//! 定義は`gmm.rs`のモジュールdoc「標準誤差・検定統計量（`cov_type`対応）」参照。
//!
//! テストは`sandwich/tests.rs`と`sandwich/tests/`にテーマ別で置く。

use faer::prelude::Solve;
use faer::{Mat, Side};

use super::weight::hac_moment_covariance;
use crate::iv::common::IvError;
use crate::linear::ols::CovType;
use crate::shared::covariance::{
    HcScale, HcVariant, cluster_correction, cluster_meat, hc_meat, hc_scales, leverages,
    resolve_hac_lags, time_ordering,
};
use crate::shared::error::{ClusterSetting, CommonError, MatrixKind};
use crate::shared::group_codes::GroupCodes;

/// [`sandwich_covariance`]の戻り値。
pub(super) struct GmmCovariance {
    pub(super) cov_params: Mat<f64>,
    /// `cov_type=Hac`のとき実際に使われたラグ数。それ以外では`None`。
    pub(super) hac_lags_used: Option<usize>,
}

/// GMMのサンドイッチ型係数分散共分散行列
/// `Avar(β̂) = B⁻¹(X'ZWΩ̂WZ'X)B⁻¹`（`B = X'ZWZ'X`、`W = S_used⁻¹`は点推定に実際に使った
/// 重み、k×k）。`gmm.rs`のモジュールdoc「標準誤差・検定統計量（cov_type対応）」参照。
///
/// `df_resid`は`cov_type=Classical`の`σ̂² = Σ(êᵢ-ē)²/df_resid`にだけ使う（`fit()`は`n - k`を
/// 渡す）。HC1・クラスターの小標本補正は`n`・`k`（`x`の形）に従う。
pub(super) fn sandwich_covariance(
    cov_type: &CovType,
    z: &Mat<f64>,
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    ztz: &Mat<f64>,
    s_used: &Mat<f64>,
    df_resid: usize,
) -> Result<GmmCovariance, IvError> {
    let n = z.nrows();
    let l = z.ncols();
    let k = x.ncols();
    let mut hac_lags_used = None;

    // W = S_used⁻¹（点推定に実際に使った重み）。bread = X'ZWZ'X（`gmm_point_estimate`
    // 内の`bread`と同型だが、こちらは後続のΩ̂サンドイッチでも使い回すため明示的な
    // 逆行列として保持する（`two_sls/cov.rs`の`xtx_inverse`と同じ考え方）。
    let s_inv = invert_spd(
        s_used,
        l,
        CommonError::NearSingularMatrix {
            matrix: MatrixKind::GmmWeight,
        },
    )?;
    let zx = z.transpose() * x; // (l, k)
    let bread = zx.transpose() * &s_inv * &zx; // (k, k) = X'ZWZ'X
    let bread_inv = invert_spd(
        &bread,
        k,
        CommonError::NearSingularMatrix {
            matrix: MatrixKind::GmmBread,
        },
    )?;

    let omega_hat = match cov_type {
        CovType::Classical => {
            // 残差を中心化した標本分散を使う（`weight_type=Classical`以外では
            // GMMの一次条件が`ē=0`を保証しないため、`gmm.rs`のモジュールdoc
            // 「標準誤差・検定統計量」`classical`の項参照）。
            let mean_residual: f64 = (0..n).map(|i| *residuals.get(i, 0)).sum::<f64>() / (n as f64);
            let centered_ssr: f64 = (0..n)
                .map(|i| (*residuals.get(i, 0) - mean_residual).powi(2))
                .sum();
            let sigma2 = centered_ssr / (df_resid as f64);
            Mat::from_fn(l, l, |i, j| sigma2 * (*ztz.get(i, j)))
        }
        CovType::Hc0 => gmm_hc_omega(z, residuals, ztz, n, k, l, HcVariant::Hc0)?,
        CovType::Hc1 => gmm_hc_omega(z, residuals, ztz, n, k, l, HcVariant::Hc1)?,
        CovType::Hc2 => gmm_hc_omega(z, residuals, ztz, n, k, l, HcVariant::Hc2)?,
        CovType::Hc3 => gmm_hc_omega(z, residuals, ztz, n, k, l, HcVariant::Hc3)?,
        CovType::Hac { lags, time_order } => {
            let lags = resolve_hac_lags(*lags, n)?;
            hac_lags_used = Some(lags);
            let order = time_ordering(time_order, n);
            // Newey-West重み付け以外の小標本補正を持たないため、点推定用の
            // `hac_moment_covariance`と計算式が一致する（`gmm.rs`の
            // モジュールdoc参照。`robust_moment_covariance`/`cluster_moment_covariance`は
            // 補正が無く使い回せないのと対照的）。
            hac_moment_covariance(z, residuals, n, l, lags, &order)
        }
        CovType::Cluster { groups } => {
            // クラスター数 `g >= 2`・`g > df_model`（構造方程式の傾き係数の数）は`fit()`冒頭
            // （`validate_iv_fit_inputs`）で検証済み。
            let groups = groups.as_ref().ok_or(CommonError::MissingClusterColumn {
                setting: ClusterSetting::CovType,
            })?;
            gmm_cluster_omega(z, residuals, n, k, l, groups)
        }
    };

    let meat_z = &s_inv * &omega_hat * &s_inv; // (l, l) = WΩ̂W
    let meat = zx.transpose() * &meat_z * &zx; // (k, k) = X'ZWΩ̂WZ'X
    let cov_params = &bread_inv * &meat * &bread_inv; // (k, k)
    Ok(GmmCovariance {
        cov_params,
        hac_lags_used,
    })
}

/// 対称正定値行列`mat`（`dim`×`dim`）の逆行列を明示的に構築する。`gmm.rs`の
/// モジュールdoc「標準誤差・検定統計量（cov_type対応）」参照（`two_sls/cov.rs`の`xtx_inverse`と
/// 同じ考え方だが、`s_used`・`bread`の2箇所で使い回すため独立したヘルパーにした）。
fn invert_spd(mat: &Mat<f64>, dim: usize, failure: CommonError) -> Result<Mat<f64>, IvError> {
    let llt = mat.llt(Side::Lower).map_err(|_| failure)?;
    Ok(llt.solve(Mat::<f64>::identity(dim, dim)))
}

/// HC0〜HC3ロバストなモーメント条件の分散共分散行列（cov_type用）: `Σᵢ scaleᵢ² zᵢzᵢ'`
/// （l×l）。係数のHC（`shared::covariance::hc_cov_params`）と同じ外積の和（`hc_meat`）で、
/// 設計行列の代わりに`Z`を使い、レバレッジは`Z`（点推定用の`ztz`をそのまま流用）から計算する
/// ——**GMMのHC2/HC3は外部参照実装での検証は不可能**（`linearmodels`の`IVGMM`にもRクロス
/// チェックの`momentfit`にも対応する共分散推定が無い、`docs/spec/iv-spec.md`4章）。2SLSの
/// HC2/HC3は逆にR `ivreg`+`sandwich::vcovHC`で検証可能なことを実機確認済み
/// （`docs/spec/iv-spec.md`4章）だが、GMMには同等の参照が無いため対象外のまま。`gmm.rs`の
/// モジュールdoc「標準誤差・検定統計量（cov_type対応）」参照。
///
/// **HC1の小標本補正`n/(n-k)`は`l`（全操作変数の数）ではなく`k`（構造方程式の係数の数）を
/// 使う**: 補正対象の残差`êᵢ = yᵢ - xᵢ'β̂`は常にk個の
/// パラメータで推定された構造残差であり、消費された自由度は常に`k`——`l`は単に外積を
/// 取る操作変数の本数（過剰識別なら`l > k`）に過ぎず、消費した自由度とは無関係
/// （共有の`hc_cov_params`のHC1が設計行列の列数=k自身を使うのと同じ理由。「`X̂`→`Z`、`k`→`l`」という
/// 機械的な置換はレバレッジの計算には妥当だが、補正係数の分母にまで機械的に適用したのは
/// 誤りだった）。このようにスコアに依存する部分（レバレッジと補正係数）をここで決めて、
/// 共有の`hc_scales`・`hc_meat`に渡す。
fn gmm_hc_omega(
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    ztz: &Mat<f64>,
    n: usize,
    k: usize,
    l: usize,
    variant: HcVariant,
) -> Result<Mat<f64>, IvError> {
    let resid = |i: usize| *residuals.get(i, 0);
    let scales = match variant {
        HcVariant::Hc0 => hc_scales(n, resid, HcScale::Hc0),
        HcVariant::Hc1 => {
            let factor = ((n as f64) / ((n - k) as f64)).sqrt();
            hc_scales(n, resid, HcScale::Hc1 { factor })
        }
        HcVariant::Hc2 => {
            let h = z_leverages(z, ztz, l)?;
            hc_scales(n, resid, HcScale::Hc2 { leverage: &h })
        }
        HcVariant::Hc3 => {
            let h = z_leverages(z, ztz, l)?;
            hc_scales(n, resid, HcScale::Hc3 { leverage: &h })
        }
    };
    Ok(hc_meat(z, &scales))
}

/// `Z`のレバレッジ`h_ii = z_i'(Z'Z)⁻¹z_i`（HC2/HC3用）。
///
/// `Z'Z`は理論上full column rank（点推定の`W₀=(Z'Z)⁻¹`が既に反転成功済み）のため、ここでの
/// 特異性は理論上到達不能だが、浮動小数点演算の丸めによる境界的な失敗に備えて`Result`のまま
/// 扱う（`two_sls/cov.rs`の`xtx_inverse`と同じ防御的な扱い、`.claude/rules/rust-style.md`
/// 「unwrap/expectはプロトタイプ段階を除き避ける」）。
fn z_leverages(z: &Mat<f64>, ztz: &Mat<f64>, l: usize) -> Result<Vec<f64>, IvError> {
    let ztz_inv = invert_spd(
        ztz,
        l,
        CommonError::internal("Z'Z could not be factorized for the HC2/HC3 leverage"),
    )?;
    Ok(leverages(z, &ztz_inv))
}

/// クラスターロバストなモーメント条件の分散共分散行列（cov_type用）:
/// `Σ_g (Σ_{i∈g} êᵢzᵢ)(Σ_{i∈g} êᵢzᵢ)' * correction`（l×l）。`two_sls.rs`の
/// `shared::covariance::cluster_cov_params`（2SLSが呼ぶ共有関数）と同型の自己拡張
/// （`X̂`→`Z`、集計は`cluster_meat`共有）。点推定用の`cluster_moment_covariance`は
/// 小標本補正が無いため使い回さない（`gmm.rs`のモジュールdoc参照）。`groups`が
/// `G>=2`であることは`validate_cluster_groups`（呼び出し元）で検証済みの前提。
///
/// 小標本補正`(G/(G-1))((n-1)/(n-k))`は`gmm_hc_omega`のHC1補正と同じ理由で`l`ではなく
/// `k`（構造方程式の係数の数）を使う（`gmm_hc_omega`のdocコメント参照）。
fn gmm_cluster_omega(
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    n: usize,
    k: usize,
    l: usize,
    groups: &GroupCodes,
) -> Mat<f64> {
    let (s_hat, n_groups) = cluster_meat(groups, l, |i, a| (*residuals.get(i, 0)) * (*z.get(i, a)));
    let correction = cluster_correction(n_groups, n, k);
    Mat::from_fn(l, l, |i, j| correction * (*s_hat.get(i, j)))
}

#[cfg(test)]
mod tests;
