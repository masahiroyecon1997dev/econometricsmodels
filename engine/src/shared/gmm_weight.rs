//! GMMの重み行列`S`（モーメント条件の分散共分散行列）の種別（`WeightType`）と、その計算。
//!
//! 定義は`iv/gmm.rs`のモジュールdoc「`weight_type`ごとの`S`」参照。IVのGMMと固定効果つきのGMM
//! （within変換後の行列を渡す）が共有する。行列だけを受け取り、`IvInput`・系統のエラー型には依存しない。
//!
//! 推定量を通すテストは`iv/gmm/fit/tests/`にある（`IvInput`・`GmmEstimator`が要るため）。

use faer::Mat;

use crate::shared::covariance::{cluster_meat, hac_meat, hc_meat, resolve_hac_lags};
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
/// なるため`CommonError::InsufficientClustersForWeightMatrix`で弾く。
///
/// `Cluster`のとき、検証したグループ（整数コード）を返す（それ以外は`None`）。
/// 呼び出し元は、このコードを反復の間そのまま借りて`cluster_moment_covariance`に渡す。
pub(crate) fn validate_weight_type(
    weight_type: &WeightType,
    n: usize,
    k: usize,
    l: usize,
) -> Result<Option<&GroupCodes>, CommonError> {
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
                return Err(CommonError::InsufficientClustersForWeightMatrix {
                    g,
                    l,
                    min_clusters,
                });
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
pub(crate) fn robust_moment_covariance(z: &Mat<f64>, residuals: &Mat<f64>, n: usize) -> Mat<f64> {
    // HC0の行スケールは残差そのもの（レバレッジを使わないので失敗しない）。
    let scales: Vec<f64> = (0..n).map(|i| *residuals.get(i, 0)).collect();
    hc_meat(z, &scales)
}

/// クラスター頑健なモーメント分散共分散行列: `Σ_g (Σ_{i∈g} êᵢzᵢ)(Σ_{i∈g} êᵢzᵢ)'`（l×l）。
/// `shared::covariance::cluster_meat`で`êᵢzᵢ`を集計するだけで小標本補正は無し
/// （`gmm.rs`のモジュールdoc参照）。グループの反復順は`GroupCodes`のコード順
/// （キーの辞書順）で、`cluster_meat`と同じ（`docs/spec/ols-spec.md`
/// 3.2節「クラスター」参照、`HashMap`はグループの反復順序が実行のたびに変わりうる）。`groups`が
/// `G>=2`であることは`validate_cluster_groups`（呼び出し元）で検証済みの前提。
pub(crate) fn cluster_moment_covariance(
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
pub(crate) fn hac_moment_covariance(
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

/// `S = A'A`となる行列`A`（`m×l`）を返す。ロバスト重みは行`êᵢzᵢ`（`m = n`）、クラスター重みは
/// クラスターごとのスコア`Σ_{i∈g} êᵢzᵢ`（`m = G`）、HAC重み（Bartlettカーネル）は長さ`L+1`の窓の
/// スコア和`w_t = Σ_{j=0}^{L} u_{t−j}`（`u_s = ê z`を時系列順に並べ、範囲の外は0、`t = 0..n+L`）を
/// `√(L+1)`で割った行（`m = n + L`）。HACは`Σ_t w_t w_t' = Σ_h (L+1−|h|) Γ_h`（窓が2つのスコアを
/// 両方含む数が`L+1−h`）なので、`(1/(L+1)) Σ_t w_t w_t' = Γ₀ + Σ_h (1 − h/(L+1))(Γ_h + Γ_h')`
/// で`hac_meat`の`S`に厳密に一致する。`S`を作って分解せず、`A`のQR分解から白色化するため
/// （`WhitenedInstruments::from_factor_rows`）。`Classical`は`A`の形に書けない（`σ̂²Z'Z`で`Z`のQRを使う）
/// ので`None`。`Hac`で`hac`が無い、`Cluster`で`cluster`が無いときも`None`。
pub(crate) fn moment_factor_rows(
    weight_type: &WeightType,
    cluster: Option<&GroupCodes>,
    hac: Option<&(usize, Vec<usize>)>,
    z: &Mat<f64>,
    residuals: &Mat<f64>,
) -> Option<Mat<f64>> {
    let (n, l) = (z.nrows(), z.ncols());
    match weight_type {
        WeightType::Robust => Some(Mat::from_fn(n, l, |i, a| {
            *residuals.get(i, 0) * *z.get(i, a)
        })),
        WeightType::Cluster { .. } => {
            let groups = cluster?;
            let mut scores = Mat::<f64>::zeros(groups.n_groups(), l);
            for (i, &code) in groups.codes().iter().enumerate() {
                for a in 0..l {
                    *scores.get_mut(code, a) += *residuals.get(i, 0) * *z.get(i, a);
                }
            }
            Some(scores)
        }
        WeightType::Hac { .. } => {
            let (lags, order) = hac?;
            let lags = *lags;
            let norm = 1.0 / ((lags + 1) as f64).sqrt();
            Some(Mat::from_fn(n + lags, l, |t, a| {
                // 窓`[t − L, t]`のうち`0..n`に入る位置のスコアの和（直接足して、累積和の丸めを避ける）。
                let first = t.saturating_sub(lags);
                let last = t.min(n - 1);
                let sum: f64 = (first..=last)
                    .map(|s| {
                        let row = order[s];
                        *residuals.get(row, 0) * *z.get(row, a)
                    })
                    .sum();
                norm * sum
            }))
        }
        WeightType::Classical => None,
    }
}

/// 残差から重み行列の元になる`S`（モーメント条件の分散共分散行列、`n`で割らない生の和）を作る。
///
/// `weight_type`が`Cluster`なら`cluster`（検証済みのコード）、`Hac`なら`hac`（解決済みのラグ数と
/// 時系列順の添字）が必要で、反復の外で1度だけ用意して渡す。`Classical`の`S`は`σ̂²・Z'Z`
/// （`σ̂²=ê'ê/n`、中心化しない。Hansen Jの絶対スケールがSarganと一致するために必要）。
pub(crate) fn moment_covariance(
    weight_type: &WeightType,
    cluster: Option<&GroupCodes>,
    hac: Option<&(usize, Vec<usize>)>,
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    ztz: &Mat<f64>,
) -> Result<Mat<f64>, CommonError> {
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
mod tests {
    use super::*;

    fn groups(n: usize, g: usize) -> WeightType {
        let labels: Vec<String> = (0..n).map(|i| format!("g{}", i % g)).collect();
        WeightType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&labels)),
        }
    }

    fn moment_data() -> (Mat<f64>, Mat<f64>) {
        let n = 24;
        let wave = |i: usize| ((i * 37 % 29) as f64 - 14.0) / 6.0;
        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => wave(i),
            _ => wave(i + 5) + 0.4 * wave(i),
        });
        let e = Mat::from_fn(n, 1, |i, _| 0.3 + wave(i + 9));
        (z, e)
    }

    /// `moment_factor_rows`の`A`は、`A'A`が`moment_covariance`の`S`と一致する（ロバスト・クラスター・HAC）。
    #[test]
    fn factor_rows_reproduce_the_moment_covariance() {
        let (z, e) = moment_data();
        let n = z.nrows();
        let ztz = z.transpose() * &z;
        let cluster = groups(n, 4);
        let codes = match &cluster {
            WeightType::Cluster { groups: Some(g) } => g,
            _ => unreachable!(),
        };
        for lags in [0_usize, 1, 3, 7] {
            let order: Vec<usize> = (0..n).rev().collect();
            let hac = (lags, order);
            let weight = WeightType::Hac {
                lags: Some(lags as i64),
                time_order: (0..n).map(|i| (n - i) as f64).collect(),
            };
            let cases = [
                (&WeightType::Robust, None, None),
                (&cluster, Some(codes), None),
                (&weight, None, Some(&hac)),
            ];
            for (index, (weight_type, cluster, hac)) in cases.into_iter().enumerate() {
                let a = moment_factor_rows(weight_type, cluster, hac, &z, &e).unwrap();
                let s = moment_covariance(weight_type, cluster, hac, &z, &e, &ztz).unwrap();
                let gram = a.transpose() * &a;
                for i in 0..3 {
                    for j in 0..3 {
                        let scale = s.get(i, j).abs().max(1e-12);
                        assert!(
                            (*gram.get(i, j) - *s.get(i, j)).abs() <= 1e-12 * scale.max(1.0),
                            "lags={lags} case={index} ({i},{j})"
                        );
                    }
                }
            }
        }
        // `Classical`は`A`の形に書けない。`Hac`で時系列順・ラグが無いときは`None`。
        assert!(moment_factor_rows(&WeightType::Classical, None, None, &z, &e).is_none());
        let hac_weight = WeightType::Hac {
            lags: None,
            time_order: vec![],
        };
        assert!(moment_factor_rows(&hac_weight, None, None, &z, &e).is_none());
    }

    /// 過剰識別（`l > k`）は`G < l`で拒否し`G == l`は受理、丁度識別（`l == k`）は`G <= l`で拒否し
    /// `G == l + 1`は受理する。`Classical`・`Robust`は常に受理する。
    #[test]
    fn validate_weight_type_cluster_boundaries() {
        let (n, k) = (30, 2);
        let reject = |g, l| validate_weight_type(&groups(n, g), n, k, l).unwrap_err();
        assert_eq!(
            reject(2, 3),
            CommonError::InsufficientClustersForWeightMatrix {
                g: 2,
                l: 3,
                min_clusters: 3
            }
        );
        assert!(
            validate_weight_type(&groups(n, 3), n, k, 3)
                .unwrap()
                .is_some()
        );
        assert_eq!(
            reject(2, 2),
            CommonError::InsufficientClustersForWeightMatrix {
                g: 2,
                l: 2,
                min_clusters: 3
            }
        );
        assert!(
            validate_weight_type(&groups(n, 3), n, k, 2)
                .unwrap()
                .is_some()
        );
        for weight in [WeightType::Classical, WeightType::Robust] {
            assert_eq!(validate_weight_type(&weight, n, k, 3), Ok(None));
        }
    }

    /// 反復の外で解決する前提のグループが無い`Cluster`は`MissingClusterColumn`。
    #[test]
    fn validate_weight_type_requires_cluster_groups() {
        let err =
            validate_weight_type(&WeightType::Cluster { groups: None }, 10, 2, 3).unwrap_err();
        assert!(matches!(err, CommonError::MissingClusterColumn { .. }));
    }

    /// `moment_covariance`の各アームは、定義式を素朴な二重ループで書いた値と一致する。
    #[test]
    fn moment_covariance_matches_the_definitions() {
        let n = 6;
        let z = Mat::from_fn(n, 2, |i, j| ((i + 1) as f64).powi(j as i32 + 1) * 0.3 - 1.0);
        let e = Mat::from_fn(n, 1, |i, _| (i as f64 * 1.7).sin());
        let ztz = z.transpose() * &z;

        // Classical: σ̂²・Z'Z（σ̂² = e'e/n）。
        let sigma2: f64 = (0..n).map(|i| e.get(i, 0).powi(2)).sum::<f64>() / n as f64;
        let classical =
            moment_covariance(&WeightType::Classical, None, None, &z, &e, &ztz).unwrap();
        // Robust: Σ e² z z'。
        let robust = moment_covariance(&WeightType::Robust, None, None, &z, &e, &ztz).unwrap();
        // Cluster: 3つのクラスター（i % 3）の`Σ_g s_g s_g'`。
        let labels: Vec<String> = (0..n).map(|i| format!("g{}", i % 3)).collect();
        let codes = GroupCodes::from_labels_without_keys(&labels);
        let weight = WeightType::Cluster {
            groups: Some(codes.clone()),
        };
        let cluster = moment_covariance(&weight, Some(&codes), None, &z, &e, &ztz).unwrap();

        for a in 0..2 {
            for b in 0..2 {
                let want_classical = sigma2 * ztz.get(a, b);
                let want_robust: f64 = (0..n)
                    .map(|i| e.get(i, 0).powi(2) * z.get(i, a) * z.get(i, b))
                    .sum();
                let want_cluster: f64 = (0..3)
                    .map(|g| {
                        let score = |c: usize| -> f64 {
                            (0..n)
                                .filter(|i| i % 3 == g)
                                .map(|i| e.get(i, 0) * z.get(i, c))
                                .sum()
                        };
                        score(a) * score(b)
                    })
                    .sum();
                assert!((classical.get(a, b) - want_classical).abs() < 1e-12);
                assert!((robust.get(a, b) - want_robust).abs() < 1e-12);
                assert!((cluster.get(a, b) - want_cluster).abs() < 1e-12);
            }
        }
    }
}
