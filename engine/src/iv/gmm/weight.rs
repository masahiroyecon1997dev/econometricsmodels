//! GMMの重み行列`S`（モーメント条件の分散共分散行列）の種別（`WeightType`）と、その計算。
//!
//! 定義は`gmm.rs`のモジュールdoc「`weight_type`ごとの`S`」参照。

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, Par};

use crate::iv::common::IvError;
use crate::shared::covariance::{cluster_meat, resolve_hac_lags};
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
    /// Newey-West（Bartlettカーネル）によるHAC型。`lags=None`なら`two_sls/cov.rs`と同じ
    /// 経験則で自動計算する。`time_order`は`IvInput`の行と対応する長さnの配列で、この値の昇順を
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
/// モジュールdoc「`weight_type`ごとの`S`」参照（`two_sls/cov.rs`の`hc_cov_params`のHC0相当だが
/// レバレッジ調整は無し。`S`はスカラー倍で点推定が変わらないため小標本補正も不要）。
pub(super) fn robust_moment_covariance(
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    n: usize,
    l: usize,
) -> Mat<f64> {
    let z_scaled = Mat::from_fn(n, l, |i, j| (*residuals.get(i, 0)) * (*z.get(i, j)));
    z_scaled.transpose() * &z_scaled
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

/// Newey-West（Bartlettカーネル）によるモーメント分散共分散行列: `two_sls/cov.rs`の
/// `hac_cov_params`と同型（`Par::Seq`を明示指定する理由も同じ、`.claude/rules/
/// rust-style.md`「パフォーマンス」参照）。設計行列が`Z`である点、小標本補正が無い点が
/// `two_sls.rs`との違い（`gmm.rs`のモジュールdoc参照）。
pub(super) fn hac_moment_covariance(
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    n: usize,
    l: usize,
    lags: usize,
    order: &[usize],
) -> Mat<f64> {
    let ze = Mat::<f64>::from_fn(n, l, |t, a| {
        let i = order[t];
        (*residuals.get(i, 0)) * (*z.get(i, a))
    });

    let mut s_hat = Mat::<f64>::zeros(l, l);
    matmul(
        s_hat.as_mut(),
        Accum::Replace,
        ze.transpose(),
        ze.as_ref(),
        1.0,
        Par::Seq,
    );

    let mut s_l = Mat::<f64>::zeros(l, l);
    for lag in 1..=lags {
        let weight = 1.0 - (lag as f64) / ((lags + 1) as f64);
        let ze_top = ze.as_ref().subrows(lag, n - lag);
        let ze_bot = ze.as_ref().subrows(0, n - lag);
        matmul(
            s_l.as_mut(),
            Accum::Replace,
            ze_top.transpose(),
            ze_bot,
            1.0,
            Par::Seq,
        );

        for a in 0..l {
            for b in 0..l {
                *s_hat.get_mut(a, b) += weight * (*s_l.get(a, b) + *s_l.get(b, a));
            }
        }
    }

    s_hat
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iv::common::{IvEstimate, IvInput};
    use crate::iv::gmm::GmmEstimator;
    use crate::iv::gmm::GmmType;
    use crate::iv::gmm::test_support::*;
    use crate::linear::common::row_time_order;
    use crate::linear::ols::CovType;
    use faer::Side;
    use faer::prelude::Solve;

    /// 2段階GMMで`weight_type`が不正（`Cluster`で`groups`未指定）なら
    /// `MissingClusterColumn`になる（1段階GMMは`weight_type`を持たないため検証もしない、
    /// `gmm.rs`のモジュールdoc参照）。
    #[test]
    fn fit_returns_missing_cluster_column_error_when_two_step_gmm_has_invalid_cluster_weight_type()
    {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Cluster { groups: None },
            },
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::MissingClusterColumn {
                setting: ClusterSetting::GmmWeightType,
            })
        );
    }

    /// 同様に2段階GMMで`weight_type=Hac`の`lags`が範囲外なら
    /// `InvalidHacLags`になる。
    #[test]
    fn fit_returns_invalid_hac_lags_error_when_two_step_gmm_has_invalid_hac_weight_type() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Hac {
                    lags: Some(-1),
                    time_order: row_time_order(n),
                },
            },
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::InvalidHacLags { hac_lags: -1, n })
        );
    }

    /// `weight_type=Cluster`の重み行列`S`（l×l）のクラスター数`g`と全操作変数数`l`の
    /// テスト用ヘルパー。`n_instruments`本の操作変数（`z1`のみ／`z1,z2`）で、
    /// `g`個のクラスターに`i % g`で割り当てて`fit`し結果を返す。
    fn fit_cluster_weighted(n_instruments: usize, g: usize) -> Result<GmmEstimator, IvError> {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let groups: Vec<String> = (0..n).map(|i| format!("g{}", i % g)).collect();
        let (instruments, names) = if n_instruments == 1 {
            (vec![z1], vec!["z1".to_string()])
        } else {
            (vec![z1, z2], vec!["z1".to_string(), "z2".to_string()])
        };
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &instruments,
            names,
            true,
            "y".to_string(),
        )
        .unwrap();
        GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&groups)),
                },
            },
            true,
            CovType::Classical,
            0.95,
        )
    }

    /// 過剰識別（l=3: const, z1, z2、k=2）では`S`は`rank(S) <= G`のため`G < l`で構造的に
    /// 特異になり、`fit()`冒頭で`InsufficientClustersForWeightMatrix`
    /// （`ValidationError`）として弾く（行列計算に入ると`NearSingularMatrix`の
    /// `ComputationError`になる。
    /// `fit_computes_cluster_weighted_estimate_matching_manual_formula`のdocコメント参照）。
    #[test]
    fn fit_returns_insufficient_clusters_for_weight_matrix_when_overidentified_and_g_less_than_l() {
        let result = fit_cluster_weighted(2, 2);
        assert_eq!(
            result.unwrap_err(),
            IvError::InsufficientClustersForWeightMatrix {
                g: 2,
                l: 3,
                min_clusters: 3
            }
        );
    }

    /// 過剰識別では`G == l`はエラーにならない（`Z'ê = 0`が成り立たず`rank(S) <= G`が
    /// 上限のため、`G == l`は非特異になりうる）。
    #[test]
    fn fit_succeeds_when_overidentified_and_g_equals_l() {
        assert!(fit_cluster_weighted(2, 3).is_ok());
    }

    /// 丁度識別（l=2: const, z1、k=2）では`Z'ê = 0`により`rank(S) <= G-1`のため、
    /// `G == l`でも特異になり`InsufficientClustersForWeightMatrix`で弾く。
    #[test]
    fn fit_returns_insufficient_clusters_for_weight_matrix_when_just_identified_and_g_equals_l() {
        let result = fit_cluster_weighted(1, 2);
        assert_eq!(
            result.unwrap_err(),
            IvError::InsufficientClustersForWeightMatrix {
                g: 2,
                l: 2,
                min_clusters: 3
            }
        );
    }

    /// 丁度識別でも`G > l`なら成功する（`G=l+1`が成功パスの境界）。
    #[test]
    fn fit_succeeds_when_just_identified_and_g_exceeds_l() {
        assert!(fit_cluster_weighted(1, 3).is_ok());
    }

    /// 過剰識別・不均一分散なデータで、`WeightType::Robust`の点推定が
    /// `Classical`（=2SLS）と異なることを確認する（`weight_type`が実際に点推定へ
    /// 影響することの動作確認。手計算オラクルとの数値一致は
    /// `fit_computes_robust_weighted_estimate_matching_manual_formula`で検証する）。
    #[test]
    fn fit_with_robust_weight_type_differs_from_classical_when_heteroskedastic() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let build_input = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let classical = GmmEstimator::fit(
            build_input(),
            GmmType::TwoStep {
                weight: WeightType::Classical,
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        let robust = GmmEstimator::fit(
            build_input(),
            GmmType::TwoStep {
                weight: WeightType::Robust,
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let diff = (*classical.params().get(1, 0) - *robust.params().get(1, 0)).abs();
        assert!(
            diff > 1e-4,
            "expected robust weighting to change the point estimate, diff={diff}"
        );
    }

    /// `WeightType::Robust`の点推定を、`GmmEstimator::fit`とは独立に`faer`演算で
    /// 手計算したオラクルと数値照合する（`two_sls/fit.rs`の
    /// `fit_matches_independently_recomputed_projection_formula_*`と同じ検証方針）。
    #[test]
    fn fit_computes_robust_weighted_estimate_matching_manual_formula() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Robust,
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        // オラクル: 手作業でZ・X・yを組み立て、W₀=(Z'Z)⁻¹で初期推定→残差→
        // S=Σêᵢ²zᵢzᵢ'→W₁=S⁻¹で最終推定、という同じ2段階を`gmm_point_estimate`とは
        // 別コードで再現する。
        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let ztz = z.transpose() * &z;
        let ztz_inv = ztz
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread0 = ztx.transpose() * &ztz_inv * &ztx;
        let meat0 = ztx.transpose() * &ztz_inv * &zty;
        let beta0 = bread0
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat0;
        let e0 = &y_mat - &x * &beta0;

        let z_scaled = Mat::from_fn(n, 3, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
        let s = z_scaled.transpose() * &z_scaled;
        let s_inv = s
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let bread1 = ztx.transpose() * &s_inv * &ztx;
        let meat1 = ztx.transpose() * &s_inv * &zty;
        let expected_beta = bread1
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat1;

        for j in 0..2 {
            assert!(
                (*estimator.params().get(j, 0) - *expected_beta.get(j, 0)).abs() < 1e-8,
                "param {j}: got {}, expected {}",
                *estimator.params().get(j, 0),
                *expected_beta.get(j, 0)
            );
        }
    }

    /// クラスター頑健版でグループ列未指定なら`MissingClusterColumn`。
    #[test]
    fn fit_returns_missing_cluster_column_error_when_cluster_weight_type_has_no_groups() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Cluster { groups: None },
            },
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::MissingClusterColumn {
                setting: ClusterSetting::GmmWeightType,
            })
        );
    }

    /// クラスターが1種類しかない場合`InsufficientClusters`。
    #[test]
    fn fit_returns_insufficient_clusters_error_when_only_one_group() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let groups = vec!["only_group".to_string(); n];
        let result = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&groups)),
                },
            },
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::InsufficientClusters {
                setting: ClusterSetting::GmmWeightType,
                g: 1
            })
        );
    }

    /// クラスター頑健版が、4グループの手計算オラクルと一致することを確認する。
    /// `robust`版のオラクル（`fit_computes_robust_weighted_estimate_matching_manual_
    /// formula`）と同様、`gmm_point_estimate`（SUT内部関数）を経由せず`llt`/`solve`を
    /// ゼロから書き下ろした完全に独立な再導出にする（`S`の構築だけでなく、
    /// bread/meat計算自体も独立検証する）。
    ///
    /// `S`（l×l、ここではl=3）はG個のランク1行列の和のため`rank(S) <= G`となる
    /// （`.claude/rules/testing-policy.md`「テスト用データセット」3.、`two_sls/cov.rs`の
    /// クラスターロバストWald検定のG境界と同じ構造的制約）。G=2だと`S`が必然的に
    /// 特異になる（`fit_returns_computation_error_when_cluster_count_is_less_than_
    /// instrument_count`で確認済み）ため、l=3に対してG=4を使う。
    #[test]
    fn fit_computes_cluster_weighted_estimate_matching_manual_formula() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let groups: Vec<String> = (0..n).map(|i| format!("g{}", i / (n / 4))).collect();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&groups)),
                },
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let ztz = z.transpose() * &z;
        let ztz_inv = ztz
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread0 = ztx.transpose() * &ztz_inv * &ztx;
        let meat0 = ztx.transpose() * &ztz_inv * &zty;
        let beta0 = bread0
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat0;
        let e0 = &y_mat - &x * &beta0;

        let mut s = Mat::<f64>::zeros(3, 3);
        for g in ["g0", "g1", "g2", "g3"] {
            let mut s_g = [0.0_f64; 3];
            for (i, group) in groups.iter().enumerate().take(n) {
                if group == g {
                    let e = *e0.get(i, 0);
                    for (a, s_g_a) in s_g.iter_mut().enumerate() {
                        *s_g_a += e * (*z.get(i, a));
                    }
                }
            }
            for a in 0..3 {
                for b in 0..3 {
                    *s.get_mut(a, b) += s_g[a] * s_g[b];
                }
            }
        }
        let s_inv = s
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let bread1 = ztx.transpose() * &s_inv * &ztx;
        let meat1 = ztx.transpose() * &s_inv * &zty;
        let expected_beta = bread1
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat1;

        for j in 0..2 {
            assert!(
                (*estimator.params().get(j, 0) - *expected_beta.get(j, 0)).abs() < 1e-8,
                "param {j}: got {}, expected {}",
                *estimator.params().get(j, 0),
                *expected_beta.get(j, 0)
            );
        }
    }

    /// `weight_type=Hac`の`lags`が範囲外（負・n以上）なら`InvalidHacLags`。
    #[test]
    fn fit_returns_invalid_hac_lags_error_when_out_of_range() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let build_input = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let result = GmmEstimator::fit(
            build_input(),
            GmmType::TwoStep {
                weight: WeightType::Hac {
                    lags: Some(-1),
                    time_order: row_time_order(n),
                },
            },
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::InvalidHacLags { hac_lags: -1, n })
        );

        let result = GmmEstimator::fit(
            build_input(),
            GmmType::TwoStep {
                weight: WeightType::Hac {
                    lags: Some(n as i64),
                    time_order: row_time_order(n),
                },
            },
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::InvalidHacLags {
                hac_lags: n as i64,
                n
            })
        );
    }

    /// `weight_type=Hac`（`lags=0`）は`Robust`と一致するはず（HAC0=HC0と同じ関係、
    /// `two_sls/cov.rs`の`fit_hac_with_zero_lags_matches_hc0`と同型の検証）。
    #[test]
    fn fit_hac_with_zero_lags_matches_robust() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let build_input = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let robust = GmmEstimator::fit(
            build_input(),
            GmmType::TwoStep {
                weight: WeightType::Robust,
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        let hac = GmmEstimator::fit(
            build_input(),
            GmmType::TwoStep {
                weight: WeightType::Hac {
                    lags: Some(0),
                    time_order: row_time_order(16),
                },
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        for j in 0..2 {
            assert!(
                (*robust.params().get(j, 0) - *hac.params().get(j, 0)).abs() < 1e-8,
                "param {j}: robust={}, hac={}",
                *robust.params().get(j, 0),
                *hac.params().get(j, 0)
            );
        }
    }

    /// `weight_type=Hac`（`lags=2`）の点推定を、`hac_moment_covariance`（`matmul`
    /// ベース）とは独立に、素朴なループでBartlettカーネル重み付き和を手計算したオラクルと
    /// 数値照合する（`fit_hac_with_zero_lags_matches_robust`は`lags=0`（ラグ項ループが
    /// 空になる自明ケース）のみで、ラグ重み付け本体は未検証だったため追加）。
    #[test]
    fn fit_computes_hac_weighted_estimate_matching_manual_formula_with_lags_two() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(
            input,
            GmmType::TwoStep {
                weight: WeightType::Hac {
                    lags: Some(2),
                    time_order: row_time_order(n),
                },
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let ztz = z.transpose() * &z;
        let ztz_inv = ztz
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread0 = ztx.transpose() * &ztz_inv * &ztx;
        let meat0 = ztx.transpose() * &ztz_inv * &zty;
        let beta0 = bread0
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat0;
        let e0 = &y_mat - &x * &beta0;

        // Bartlettカーネル（lags=2、time_order無指定なので行順=時系列順）を、
        // `hac_moment_covariance`のmatmulベース実装とは別に、要素ごとの
        // ループで素朴に計算する。
        let ze = Mat::from_fn(n, 3, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
        let mut s = Mat::<f64>::zeros(3, 3);
        for a in 0..3 {
            for b in 0..3 {
                let mut acc = 0.0;
                for t in 0..n {
                    acc += (*ze.get(t, a)) * (*ze.get(t, b));
                }
                *s.get_mut(a, b) = acc;
            }
        }
        for lag in 1..=2usize {
            let weight = 1.0 - (lag as f64) / 3.0;
            let mut s_l = Mat::<f64>::zeros(3, 3);
            for a in 0..3 {
                for b in 0..3 {
                    let mut acc = 0.0;
                    for t in lag..n {
                        acc += (*ze.get(t, a)) * (*ze.get(t - lag, b));
                    }
                    *s_l.get_mut(a, b) = acc;
                }
            }
            for a in 0..3 {
                for b in 0..3 {
                    *s.get_mut(a, b) += weight * (*s_l.get(a, b) + *s_l.get(b, a));
                }
            }
        }

        let s_inv = s
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let bread1 = ztx.transpose() * &s_inv * &ztx;
        let meat1 = ztx.transpose() * &s_inv * &zty;
        let expected_beta = bread1
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat1;

        for j in 0..2 {
            assert!(
                (*estimator.params().get(j, 0) - *expected_beta.get(j, 0)).abs() < 1e-8,
                "param {j}: got {}, expected {}",
                *estimator.params().get(j, 0),
                *expected_beta.get(j, 0)
            );
        }
    }

    /// `lags=None`（経験則自動計算`L = floor(4*(n/100)^(2/9))`）が、`heteroskedastic_test_
    /// columns()`の`n=16`では`L=2`と一致するため、`lags=Some(2)`を明示指定した場合と
    /// 数値的に一致するはず（`two_sls/cov.rs`の
    /// `fit_computes_hac_std_errors_with_auto_lags_matching_explicit_lags`と同じ検証方針）。
    #[test]
    fn fit_with_hac_weight_type_and_auto_lags_matches_explicit_lags_two() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let build_input = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let auto_estimator = GmmEstimator::fit(
            build_input(),
            GmmType::TwoStep {
                weight: WeightType::Hac {
                    lags: None,
                    time_order: row_time_order(16),
                },
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        let explicit_estimator = GmmEstimator::fit(
            build_input(),
            GmmType::TwoStep {
                weight: WeightType::Hac {
                    lags: Some(2),
                    time_order: row_time_order(16),
                },
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        for j in 0..2 {
            assert!(
                (*auto_estimator.params().get(j, 0) - *explicit_estimator.params().get(j, 0)).abs()
                    < 1e-8
            );
        }
    }

    /// `hac_lags_used()`は`cov_type=Hac`または`weight_type=Hac`のどちらかが該当すれば
    /// 解決後のラグ数（`n=16`の自動計算は2）、どちらも該当しなければ`None`。両方がHacで
    /// `lags`が異なる場合は`cov_type`側を優先する。
    #[test]
    fn hac_lags_used_covers_cov_type_and_weight_type() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let hac = |lags: Option<i64>| CovType::Hac {
            lags,
            time_order: row_time_order(16),
        };
        let hac_weight = |lags: Option<i64>| WeightType::Hac {
            lags,
            time_order: row_time_order(16),
        };
        let lags_used = |gmm_type: GmmType, cov_type: CovType| {
            let input = IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();
            // 収束は検証対象外（反復GMMが`max_iter`内に収束しなくてもラグは報告される）
            GmmEstimator::fit(input, gmm_type, false, cov_type, 0.95)
                .unwrap()
                .hac_lags_used()
        };

        // どちらもHacでない
        assert_eq!(lags_used(GmmType::OneStep, CovType::Classical), None);
        assert_eq!(
            lags_used(
                GmmType::TwoStep {
                    weight: WeightType::Robust
                },
                CovType::Hc1
            ),
            None
        );
        // `cov_type=Hac`のみ（1段階は`weight_type`を持たない）
        assert_eq!(lags_used(GmmType::OneStep, hac(None)), Some(2));
        assert_eq!(
            lags_used(
                GmmType::TwoStep {
                    weight: WeightType::Robust
                },
                hac(Some(3))
            ),
            Some(3)
        );
        // `weight_type=Hac`のみ
        assert_eq!(
            lags_used(
                GmmType::TwoStep {
                    weight: hac_weight(None)
                },
                CovType::Classical
            ),
            Some(2)
        );
        // 反復GMMでも`weight_type=Hac`のラグが報告される
        assert_eq!(
            lags_used(
                GmmType::Iterated {
                    weight: hac_weight(None),
                    max_iter: 5,
                    tol: 1e-6
                },
                CovType::Classical
            ),
            Some(2)
        );
        // 両方Hacで`lags`が異なる場合は`cov_type`側
        assert_eq!(
            lags_used(
                GmmType::TwoStep {
                    weight: hac_weight(Some(1))
                },
                hac(Some(3))
            ),
            Some(3)
        );
    }

    /// `time_order`を指定した場合、行順がシャッフルされていても時系列順に並べ替えてから
    /// ラグ付きモーメント共分散を計算することを確認する（`two_sls/cov.rs`の
    /// `fit_computes_hac_std_errors_respecting_time_order`と同じ検証方針）。
    #[test]
    fn fit_with_hac_weight_type_respects_time_order() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let shuffle: Vec<usize> = vec![3, 1, 6, 0, 5, 2, 7, 4, 11, 9, 14, 8, 13, 10, 15, 12];
        assert_eq!(shuffle.len(), n);
        let shuffled_time: Vec<f64> = shuffle.iter().map(|&i| i as f64).collect();
        let shuffled_y: Vec<f64> = shuffle.iter().map(|&i| y[i]).collect();
        let shuffled_x_endog: Vec<f64> = shuffle.iter().map(|&i| x_endog[i]).collect();
        let shuffled_z1: Vec<f64> = shuffle.iter().map(|&i| z1[i]).collect();
        let shuffled_z2: Vec<f64> = shuffle.iter().map(|&i| z2[i]).collect();

        let shuffled_input = IvInput::from_columns(
            &shuffled_y,
            &[],
            vec![],
            std::slice::from_ref(&shuffled_x_endog),
            vec!["endog1".to_string()],
            &[shuffled_z1, shuffled_z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let shuffled_estimator = GmmEstimator::fit(
            shuffled_input,
            GmmType::TwoStep {
                weight: WeightType::Hac {
                    lags: Some(2),
                    time_order: shuffled_time,
                },
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let unshuffled_input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let unshuffled_estimator = GmmEstimator::fit(
            unshuffled_input,
            GmmType::TwoStep {
                weight: WeightType::Hac {
                    lags: Some(2),
                    time_order: row_time_order(n),
                },
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        for j in 0..2 {
            assert!(
                (*shuffled_estimator.params().get(j, 0) - *unshuffled_estimator.params().get(j, 0))
                    .abs()
                    < 1e-8
            );
        }
    }

    #[test]
    fn validate_weight_type_returns_the_groups_only_for_cluster() {
        let n = 16;
        let cluster = WeightType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&labelled_groups(
                n, 4, "a",
            ))),
        };
        let codes = validate_weight_type(&cluster, n, 2, 3)
            .unwrap()
            .expect("Cluster weight must expose its groups");
        assert_eq!(codes.nobs(), n);
        assert_eq!(codes.n_groups(), 4);

        for weight in [WeightType::Classical, WeightType::Robust] {
            assert!(validate_weight_type(&weight, n, 2, 3).unwrap().is_none());
        }
    }

    /// 旧実装（`String`ラベルの`BTreeMap`で集計）をそのまま残したoracleと、
    /// `cluster_moment_covariance`（コードで行を1回走査して集計）がビット単位で一致する。
    #[test]
    fn cluster_moment_covariance_is_bit_identical_to_the_string_keyed_btreemap_implementation() {
        let n = 300;
        let l = 4;
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let labels: Vec<String> = (0..n)
            .map(|_| ((next() * 29.0) as usize).to_string())
            .collect();
        let z = Mat::from_fn(n, l, |_, _| next() * 20.0 - 10.0);
        let residuals = Mat::from_fn(n, 1, |_, _| next() * 4.0 - 2.0);

        let mut by_label: std::collections::BTreeMap<&str, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (i, label) in labels.iter().enumerate() {
            by_label.entry(label.as_str()).or_default().push(i);
        }
        let mut expected = Mat::<f64>::zeros(l, l);
        for rows in by_label.values() {
            let mut s_g = vec![0.0_f64; l];
            for &i in rows {
                let e = *residuals.get(i, 0);
                for (a, s_g_a) in s_g.iter_mut().enumerate() {
                    *s_g_a += e * (*z.get(i, a));
                }
            }
            for a in 0..l {
                for b in 0..l {
                    *expected.get_mut(a, b) += s_g[a] * s_g[b];
                }
            }
        }

        let codes = GroupCodes::from_labels_without_keys(&labels);
        let actual = cluster_moment_covariance(&z, &residuals, l, &codes);
        for a in 0..l {
            for b in 0..l {
                assert_eq!(actual.get(a, b).to_bits(), expected.get(a, b).to_bits());
            }
        }
    }
}
