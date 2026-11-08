//! GMMの重み行列`S`（モーメント条件の分散共分散行列）の種別（`WeightType`）と、その計算。
//!
//! 定義は`gmm.rs`のモジュールdoc「`weight_type`ごとの`S`」参照。
//!
//! テストは`weight/tests.rs`と`weight/tests/`にテーマ別で置く。

use faer::Mat;

use crate::iv::common::IvError;
use crate::shared::covariance::{
    HcScale, cluster_meat, hac_meat, hc_meat, hc_scales, resolve_hac_lags,
};
use crate::shared::error::{ClusterSetting, CommonError};
use crate::shared::group_codes::GroupCodes;
use crate::shared::validation::validate_cluster_groups;

/// GMMの点推定に使う重み行列の種別（`docs/spec/iv-spec.md`1.2節）。
///
/// `cov_type`（標準誤差の報告方法）とは独立の概念で、こちらは点推定自体に影響する
/// （`gmm.rs`のモジュールdoc参照）。
#[derive(Debug, Clone, PartialEq)]
pub enum WeightType {
    /// 等分散前提。`W₁ ∝ (Z'Z)⁻¹`となり2SLSと数値的に一致する。
    Classical,
    /// 不均一分散頑健。
    Robust,
    /// クラスター頑健。`groups`が`None`の場合は`CommonError::MissingClusterColumn`、コードの
    /// 行数が`n`と食い違う場合は`CommonError::ClusterDimensionMismatch`。
    Cluster { groups: Option<GroupCodes> },
    /// Newey-West（Bartlettカーネル）によるHAC型。`lags=None`なら
    /// `shared::covariance::resolve_hac_lags`の経験則で自動計算する。`time_order`は`IvInput`の行と対応する長さnの配列で、この値の昇順を
    /// 時系列順とする（必須。行順を暗黙に時系列順とみなす既定は置かない）。
    Hac {
        lags: Option<i64>,
        time_order: Vec<f64>,
    },
}

/// `weight_type`自体の引数が妥当かどうかだけを検証する（`S`は構築しない）。
///
/// `TwoStep`/`Iterated`の`weight_type`について、呼び出し元の設定ミス（`Cluster`指定なのに
/// `groups`未指定、`Hac`の`lags`が範囲外、`Cluster`でクラスター数が重み行列`S`の階数に
/// 足りない等）を検出する。`OneStep`は`weight_type`を持たないため呼ばれない。
/// `fit()`本体の`match`も`S`構築の過程で同じ検証を重ねて行う
/// （冗長だが検証コスト自体は軽微で、各分岐を自己完結させる方を優先した）。
///
/// `k`は構造方程式の係数の数、`l`は全操作変数（`x_exog ++ instruments`）の数。
/// `Cluster`の`S`（l×l）は`rank(S) ≤ G`で、丁度識別（`l == k`）では`Z'ê = 0`により
/// `rank(S) ≤ G-1`となる。よって過剰識別は`G < l`、丁度識別は`G <= l`で構造的に特異に
/// なるため`IvError::InsufficientClustersForWeightMatrix`で弾く。
///
/// `Cluster`のとき、検証したグループ（整数コード）を返す（それ以外は`None`）。
/// 呼び出し元は、このコードを反復の間そのまま借りて`cluster_moment_covariance`に渡す。
pub(super) fn validate_weight_type(
    weight_type: &WeightType,
    n: usize,
    k: usize,
    l: usize,
) -> Result<Option<&GroupCodes>, IvError> {
    match weight_type {
        WeightType::Classical | WeightType::Robust => {}
        WeightType::Cluster { groups } => {
            let groups = groups.as_ref().ok_or(CommonError::MissingClusterColumn {
                setting: ClusterSetting::GmmWeightType,
            })?;
            let g = validate_cluster_groups(groups, n, ClusterSetting::GmmWeightType)?;
            // 丁度識別は`G <= l`、過剰識別は`G < l`で`S`が特異になる。
            let min_clusters = if l == k { l + 1 } else { l };
            if g < min_clusters {
                return Err(IvError::InsufficientClustersForWeightMatrix { g, l, min_clusters });
            }
            return Ok(Some(groups));
        }
        WeightType::Hac { lags, .. } => {
            resolve_hac_lags(*lags, n)?;
        }
    }
    Ok(None)
}

/// 不均一分散頑健なモーメント分散共分散行列: `Σᵢ êᵢ² zᵢzᵢ'`（l×l）。`gmm.rs`の
/// モジュールdoc「`weight_type`ごとの`S`」参照（`shared::covariance`のHC0と同じ外積の和で、
/// レバレッジ調整は無し。`S`はスカラー倍で点推定が変わらないため小標本補正も不要）。
pub(super) fn robust_moment_covariance(z: &Mat<f64>, residuals: &Mat<f64>, n: usize) -> Mat<f64> {
    let scales = hc_scales(n, |i| *residuals.get(i, 0), HcScale::Hc0);
    hc_meat(z, &scales)
}

/// クラスター頑健なモーメント分散共分散行列: `Σ_g (Σ_{i∈g} êᵢzᵢ)(Σ_{i∈g} êᵢzᵢ)'`（l×l）。
/// `shared::covariance::cluster_meat`で`êᵢzᵢ`を集計するだけで小標本補正は無し
/// （`gmm.rs`のモジュールdoc参照）。グループの反復順は`GroupCodes`のコード順
/// （キーの辞書順）で、`cluster_meat`と同じ（`engine/src/linear/CLAUDE.md`
/// 「踏んだ罠」参照、`HashMap`はグループの反復順序が実行のたびに変わりうる）。`groups`が
/// `G>=2`であることは`validate_cluster_groups`（呼び出し元）で検証済みの前提。
pub(super) fn cluster_moment_covariance(
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    l: usize,
    groups: &GroupCodes,
) -> Mat<f64> {
    cluster_meat(groups, l, |i, a| (*residuals.get(i, 0)) * (*z.get(i, a))).0
}

/// Newey-West（Bartlettカーネル）によるモーメント分散共分散行列: `shared::covariance::hac_meat`
/// （`Par::Seq`を明示指定する理由もそちら）を、スコア`êᵢzᵢ`で呼ぶ。小標本補正が無い点が
/// `two_sls/cov.rs`のHAC共分散との違い（`gmm.rs`のモジュールdoc参照）。
pub(super) fn hac_moment_covariance(
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    n: usize,
    l: usize,
    lags: usize,
    order: &[usize],
) -> Mat<f64> {
    hac_meat(n, l, lags, order, |i, a| {
        (*residuals.get(i, 0)) * (*z.get(i, a))
    })
}

/// 残差から重み行列の元になる`S`（モーメント条件の分散共分散行列、`n`で割らない生の和）を作る。
///
/// `weight_type`が`Cluster`なら`cluster`（検証済みのコード）、`Hac`なら`hac`（解決済みのラグ数と
/// 時系列順の添字）が必要で、反復の外で1度だけ用意して渡す。`Classical`の`S`は`σ̂²・Z'Z`
/// （`σ̂²=ê'ê/n`、中心化しない。Hansen Jの絶対スケールがSarganと一致するために必要）。
pub(super) fn moment_covariance(
    weight_type: &WeightType,
    cluster: Option<&GroupCodes>,
    hac: Option<&(usize, Vec<usize>)>,
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    ztz: &Mat<f64>,
) -> Result<Mat<f64>, IvError> {
    let (n, l) = (z.nrows(), z.ncols());
    Ok(match weight_type {
        WeightType::Classical => {
            let sigma2: f64 =
                (0..n).map(|i| (*residuals.get(i, 0)).powi(2)).sum::<f64>() / (n as f64);
            Mat::from_fn(l, l, |i, j| sigma2 * (*ztz.get(i, j)))
        }
        WeightType::Robust => robust_moment_covariance(z, residuals, n),
        WeightType::Cluster { .. } => {
            let groups = cluster.ok_or(CommonError::MissingClusterColumn {
                setting: ClusterSetting::GmmWeightType,
            })?;
            cluster_moment_covariance(z, residuals, l, groups)
        }
        WeightType::Hac { .. } => {
            let (lags, order) = hac.ok_or_else(|| {
                CommonError::internal("the HAC lags and time order were not resolved")
            })?;
            hac_moment_covariance(z, residuals, n, l, *lags, order)
        }
    })
}

#[cfg(test)]
mod tests;
