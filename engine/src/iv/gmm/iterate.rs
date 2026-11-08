//! GMMの推定方式（`GmmType`）と、点推定・反復（`gmm_iterate`）・収束判定。
//!
//! 定義は`gmm.rs`のモジュールdoc「`GmmType`と点推定への影響」「収束条件」参照。

use faer::prelude::Solve;
use faer::{Mat, Side};

use super::weight::{WeightType, moment_covariance};
use crate::iv::common::IvError;
use crate::shared::covariance::{resolve_hac_lags, time_ordering};
use crate::shared::error::{CommonError, MatrixKind};
use crate::shared::group_codes::GroupCodes;

/// GMMの推定方式（`docs/spec/iv-spec.md`3.3節）。
///
/// 方式ごとに使う設定だけを持つenumにすることで、矛盾した組み合わせ（1段階GMMに反復設定、
/// 1段階GMMに`weight_type`等）を型で表現できないようにする。1段階GMMの点推定・Hansen Jは
/// 初期の重み`(Z'Z)⁻¹`だけで計算するため`WeightType`を持たない。
#[derive(Debug, Clone, PartialEq)]
pub enum GmmType {
    /// 1段階GMM。`W₀ = (Z'Z)⁻¹`のみで打ち切る（`weight_type`によらず2SLSと同じ点推定）。
    OneStep,
    /// 2段階の効率的GMM。1段階目の残差から`weight`に応じた重み行列を構築して再推定する。
    TwoStep { weight: WeightType },
    /// 反復GMM。`weight`に応じた重み行列の再構築を、係数が`tol`で収束するか、初回推定を
    /// 含めて`max_iter`回に達するまで繰り返す（`max_iter`は3以上）。
    Iterated {
        weight: WeightType,
        max_iter: usize,
        tol: f64,
    },
}

/// GMMの収束判定に使う絶対誤差の床。ユーザーには公開せず内部固定値とする
/// （`gmm.rs`のモジュールdoc「収束条件」参照）。`tests/api_tests`のクロスチェックが
/// 使う`ATOL`の既定値と同じ`1e-8`を踏襲する（`.claude/rules/testing-policy.md`
/// 「許容誤差は相対誤差1e-8を基本」）。
const GMM_CONVERGENCE_ATOL: f64 = 1e-8;

/// [`gmm_iterate`]の戻り値。
pub(super) struct GmmIteration {
    pub(super) beta: Mat<f64>,
    /// 最終推定の残差 `ê = y - Xβ̂`。
    pub(super) residuals: Mat<f64>,
    /// 最終推定`beta`に実際に使った重み行列`S`（l×l）。ループが1度も回らない場合は
    /// `classical_s`（σ̂²₀・Z'Z）。
    pub(super) s_used: Mat<f64>,
    pub(super) n_iter: i64,
    pub(super) converged: bool,
    /// `weight_type=Hac`のとき実際に使われたラグ数。それ以外では`None`。
    pub(super) hac_lags_used: Option<usize>,
}

/// 初期推定（`W₀ = (Z'Z)⁻¹`）から、方式（`gmm_type`）に応じた回数だけ「直前の残差から`S`を
/// 再構築→再推定」を繰り返す。1段階は1度も回らず打ち切り（`weight_type`を持たない）、2段階は
/// 2回目の推定まで、反復は収束するか`max_iter`回に達するまで回す（`gmm.rs`のモジュールdoc
/// 「`GmmType`と点推定への影響」「収束条件」参照）。`s_used`はHansen J検定に使う重み行列で、
/// ループが1度も回らない場合は常に`classical_s`（σ̂²₀・Z'Z）を使う。
#[allow(clippy::too_many_arguments)]
pub(super) fn gmm_iterate(
    gmm_type: &GmmType,
    weight_type: Option<&WeightType>,
    weight_cluster_codes: Option<&GroupCodes>,
    z: &Mat<f64>,
    x: &Mat<f64>,
    y: &Mat<f64>,
    ztz: &Mat<f64>,
    beta0: Mat<f64>,
    residuals0: Mat<f64>,
    classical_s: Mat<f64>,
) -> Result<GmmIteration, IvError> {
    let n = z.nrows();
    let mut beta = beta0;
    let mut residuals = residuals0;
    let mut s_used = classical_s;
    let mut n_iter: i64 = 1;
    // `Iterated`以外は収束判定自体を行わないため常にtrue。`Iterated`は
    // ループ内で実際に収束条件を満たした場合のみtrueに更新する（`max_iter>=3`のため
    // 比較対象となる前回推定値は必ず存在する）。
    let mut converged = !matches!(gmm_type, GmmType::Iterated { .. });
    let mut hac_lags_used: Option<usize> = None;
    let (max_steps, tol): (i64, Option<f64>) = match gmm_type {
        GmmType::OneStep => (1, None),
        GmmType::TwoStep { .. } => (2, None),
        GmmType::Iterated { max_iter, tol, .. } => (*max_iter as i64, Some(*tol)),
    };

    if let Some(weight_type) = weight_type {
        // Hacのlags解決・時系列順序（`O(n log n)`のソートを含む）はweight_typeに対して
        // 不変な前処理のため、ループの外で一度だけ計算し使い回す（ループ内で反復のたびに
        // 再計算すると、max_iterが大きいiterated GMMで無駄なコストが反復回数倍に
        // 膨らむ）。Clusterのgroups検証は`validate_weight_type`（`fit()`冒頭）で既に
        // 1回行っているため、ループ内では再検証しない。
        let hac_precomputed = match weight_type {
            WeightType::Hac { lags, time_order } => {
                let lags = resolve_hac_lags(*lags, n)?;
                hac_lags_used = Some(lags);
                let order = time_ordering(time_order, n);
                Some((lags, order))
            }
            _ => None,
        };

        while n_iter < max_steps {
            // 点推定は正のスカラー倍不変のため`classical_s`相当（σ̂²・Z'Z）を使っても
            // `Z'Z`単体を使った場合と`β̂`は変わらない。Hansen Jにそのまま使い回せる
            // よう、あらかじめ正しくスケーリングされたSを使う（`iv/CLAUDE.md`参照）。
            let s_next = moment_covariance(
                weight_type,
                weight_cluster_codes,
                hac_precomputed.as_ref(),
                z,
                &residuals,
                ztz,
            )?;
            let beta_next = gmm_point_estimate(z, x, y, &s_next)?;
            n_iter += 1;

            let just_converged = match tol {
                Some(tol) => gmm_coefficients_converged(&beta, &beta_next, tol),
                None => false,
            };

            beta = beta_next;
            residuals = y - x * &beta;
            s_used = s_next;

            if just_converged {
                converged = true;
                break;
            }
        }
    }

    Ok(GmmIteration {
        beta,
        residuals,
        s_used,
        n_iter,
        converged,
        hac_lags_used,
    })
}

/// 係数ベクトルが収束したかをelementwiseで判定する: 各係数`i`について
/// `|next_i - prev_i| <= max(rtol * |prev_i|, GMM_CONVERGENCE_ATOL)`を満たせば収束
/// （`tests/api_tests`の数値クロスチェックと同じ`tol = max(rtol * |ref|, atol)`の考え方、
/// `gmm.rs`のモジュールdoc「収束条件」参照）。全係数がこの条件を満たして初めて
/// 収束とする（ベクトルノルムの比ではなくelementwise maxを取る設計、いずれか1つの係数が
/// 収束していない状態を見逃さないため）。
fn gmm_coefficients_converged(prev: &Mat<f64>, next: &Mat<f64>, rtol: f64) -> bool {
    let k = prev.nrows();
    (0..k).all(|i| {
        let diff = (*next.get(i, 0) - *prev.get(i, 0)).abs();
        let tol = (rtol * prev.get(i, 0).abs()).max(GMM_CONVERGENCE_ATOL);
        diff <= tol
    })
}

/// 重み行列`W=S⁻¹`によるGMMの点推定 `β̂(W) = (X'ZWZ'X)⁻¹X'ZWZ'y` を求める。
///
/// `W`を明示的に逆行列として構築せず、`S`に対する連立方程式を`llt().solve(...)`で
/// 解くことで数値的に安定させる（`two_sls/cov.rs`の`xtx_inverse`と同様、明示的な逆行列は
/// 必要な箇所のみ`Mat::identity`を渡して求める設計だが、ここでは逆行列自体を後で
/// 再利用しないため`solve`のみで完結させる）。
pub(super) fn gmm_point_estimate(
    z: &Mat<f64>,
    x: &Mat<f64>,
    y: &Mat<f64>,
    s: &Mat<f64>,
) -> Result<Mat<f64>, IvError> {
    let zx = z.transpose() * x; // (l, k)
    let zy = z.transpose() * y; // (l, 1)

    let llt_s = s
        .llt(Side::Lower)
        .map_err(|_| CommonError::NearSingularMatrix {
            matrix: MatrixKind::GmmWeight,
        })?;
    let s_inv_zx = llt_s.solve(&zx); // (l, k) = S⁻¹Z'X
    let s_inv_zy = llt_s.solve(&zy); // (l, 1) = S⁻¹Z'y

    let bread = zx.transpose() * &s_inv_zx; // (k, k) = X'ZS⁻¹Z'X
    let meat = zx.transpose() * &s_inv_zy; // (k, 1) = X'ZS⁻¹Z'y

    let llt_bread = bread
        .llt(Side::Lower)
        .map_err(|_| CommonError::NearSingularMatrix {
            matrix: MatrixKind::GmmBread,
        })?;
    Ok(llt_bread.solve(&meat))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iv::common::{IvEstimate, IvInput};
    use crate::iv::gmm::GmmEstimator;
    use crate::iv::gmm::test_support::*;
    use crate::linear::ols::CovType;

    /// `Iterated`の`max_iter`が3未満（0・1・2）なら`InvalidGmmMaxIter`（上限2回の反復は
    /// 2段階GMMに収束判定を付けただけで紛らわしいため。2段階は`TwoStep`を使う）。
    #[test]
    fn fit_returns_invalid_gmm_max_iter_error_for_values_below_three() {
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

        for invalid in [0_usize, 1, 2, 10_001, 1_000_000] {
            let result = GmmEstimator::fit(
                build_input(),
                GmmType::Iterated {
                    weight: WeightType::Classical,
                    max_iter: invalid,
                    tol: 1e-6,
                },
                true,
                CovType::Classical,
                0.95,
            );
            assert_eq!(
                result.unwrap_err(),
                IvError::InvalidGmmMaxIter {
                    max_iter: invalid as i64
                },
                "max_iter={invalid}"
            );
        }
    }

    /// `Iterated`は許容誤差に収束しなければ`max_iter`回（初回推定を含む）まで反復し、
    /// `raise_on_non_convergence=false`なら`converged=false`のまま`n_iter=max_iter`で返す。
    #[test]
    fn fit_iterated_runs_up_to_max_iter_when_not_converged() {
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

        let estimator = GmmEstimator::fit(
            input,
            GmmType::Iterated {
                weight: WeightType::Robust,
                max_iter: 3,
                tol: 1e-300,
            },
            false,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert_eq!(estimator.n_iter(), 3);
        assert!(!estimator.converged());
    }

    /// 3回目以降の反復（`Iterated`の`max_iter=3`）は「2回目の反復と同じ手続き（直前の残差から
    /// Sを再構築して再推定）をもう一度繰り返すだけ」であることを、`max_iter=3`の
    /// 結果を`GmmEstimator::fit`とは独立に手計算したオラクル（3回分のS再構築を明示的に
    /// 書き下ろす）と数値照合して確認する（`gmm.rs`のモジュールdoc
    /// 「`GmmType`と点推定への影響」参照）。
    #[test]
    fn fit_computes_three_step_gmm_matching_manual_iteration() {
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
            GmmType::Iterated {
                weight: WeightType::Robust,
                max_iter: 3,
                tol: 1e-300,
            },
            false,
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

        let solve3 = |s: &Mat<f64>| -> Mat<f64> {
            let zty = z.transpose() * &y_mat;
            let ztx = z.transpose() * &x;
            let s_inv = s
                .llt(Side::Lower)
                .unwrap()
                .solve(Mat::<f64>::identity(3, 3));
            let bread = ztx.transpose() * &s_inv * &ztx;
            let meat = ztx.transpose() * &s_inv * &zty;
            bread
                .llt(Side::Lower)
                .unwrap()
                .solve(Mat::<f64>::identity(2, 2))
                * &meat
        };

        let ztz = z.transpose() * &z;
        let beta0 = solve3(&ztz);
        let e0 = &y_mat - &x * &beta0;
        let z_scaled0 = Mat::from_fn(n, 3, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
        let s1 = z_scaled0.transpose() * &z_scaled0;
        let beta1 = solve3(&s1);
        let e1 = &y_mat - &x * &beta1;
        let z_scaled1 = Mat::from_fn(n, 3, |i, j| (*e1.get(i, 0)) * (*z.get(i, j)));
        let s2 = z_scaled1.transpose() * &z_scaled1;
        let expected_beta = solve3(&s2);

        for j in 0..2 {
            assert!(
                (*estimator.params().get(j, 0) - *expected_beta.get(j, 0)).abs() < 1e-8,
                "param {j}: got {}, expected {}",
                *estimator.params().get(j, 0),
                *expected_beta.get(j, 0)
            );
        }
    }

    /// 1段階GMM（`OneStep`）は`weight_type`を持たず、`W₀=(Z'Z)⁻¹`による初期推定`β̂₀`
    /// （=2SLSの点推定）と一致する。`weight_type()`は`None`、`n_iter=1`、`converged=true`
    /// （`gmm.rs`のモジュールdoc参照）。
    #[test]
    fn fit_one_step_has_no_weight_type_and_matches_classical_two_step() {
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

        let one_step = GmmEstimator::fit(
            build_input(),
            GmmType::OneStep,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert_eq!(one_step.gmm_type(), &GmmType::OneStep);
        assert_eq!(one_step.weight_type(), None);
        assert_eq!(one_step.n_iter(), 1);
        assert!(one_step.converged());

        // 2-step（weight_type=Classicalなら1-stepと数値的に同じはず、上記参照）とも一致する。
        let two_step = GmmEstimator::fit(
            build_input(),
            GmmType::TwoStep {
                weight: WeightType::Classical,
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert_eq!(two_step.n_iter(), 2);
        assert!(two_step.converged());
        for j in 0..2 {
            assert!(
                (*one_step.params().get(j, 0) - *two_step.params().get(j, 0)).abs() < 1e-8,
                "param {j}"
            );
        }
    }

    /// `S`を正のスカラー倍しても点推定`β̂`が変わらないこと（`gmm.rs`のモジュールdoc
    /// 「`weight_type`と点推定への影響」の根拠となる代数的性質）を、`gmm_point_estimate`を
    /// 直接呼んで検証する。
    #[test]
    fn gmm_point_estimate_is_invariant_to_positive_scaling_of_s() {
        let n = 8;
        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => (i as f64) + 1.0,
            _ => ((i as f64) * 1.3).sin() * 4.0 + (i as f64),
        });
        let x = Mat::from_fn(n, 2, |i, j| {
            if j == 0 {
                1.0
            } else {
                (i as f64) * 0.7 + ((i as f64) * 0.9).cos()
            }
        });
        let y = Mat::from_fn(n, 1, |i, _| 1.0 + 2.0 * (*x.get(i, 1)) + (i as f64) * 0.1);

        let s = z.transpose() * &z;
        let beta_unscaled = gmm_point_estimate(&z, &x, &y, &s).unwrap();
        let s_scaled = Mat::from_fn(3, 3, |i, j| 37.5 * (*s.get(i, j)));
        let beta_scaled = gmm_point_estimate(&z, &x, &y, &s_scaled).unwrap();

        for j in 0..2 {
            assert!(
                (*beta_unscaled.get(j, 0) - *beta_scaled.get(j, 0)).abs() < 1e-8,
                "param {j}: unscaled={}, scaled={}",
                *beta_unscaled.get(j, 0),
                *beta_scaled.get(j, 0)
            );
        }
    }

    /// `gmm_tol`が`Some`かつ0以下（0・負）なら`InvalidGmmTol`。
    #[test]
    fn fit_returns_invalid_gmm_tol_error_for_non_positive_values() {
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

        for invalid in [0.0_f64, -1.0, -1e-8] {
            let result = GmmEstimator::fit(
                build_input(),
                GmmType::Iterated {
                    weight: WeightType::Classical,
                    max_iter: 3,
                    tol: invalid,
                },
                true,
                CovType::Classical,
                0.95,
            );
            assert_eq!(
                result.unwrap_err(),
                IvError::InvalidGmmTol { gmm_tol: invalid },
                "gmm_tol={invalid}"
            );
        }
    }

    /// `gmm_tol`を指定すると、上限（`max_iter`）に達する前でも収束条件を
    /// 満たした時点で早期終了する（`fit()`のdocコメント「収束条件」参照）。
    /// 極めて緩い許容誤差（`rtol=1.0`）を使うことで、実際の収束の速さに依存せず
    /// 「上限に達する前に打ち切られる」ことを決定的に検証する。
    #[test]
    fn fit_stops_early_when_gmm_tol_is_satisfied() {
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

        let estimator = GmmEstimator::fit(
            input,
            GmmType::Iterated {
                weight: WeightType::Robust,
                max_iter: 10,
                tol: 1.0,
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert!(estimator.converged());
        assert!(estimator.n_iter() < 10, "n_iter={}", estimator.n_iter());
    }

    /// 上記テストは`rtol=1.0`という極めて緩い許容誤差のため、ループの1回目
    /// （`n_iter=2`、`β̂₀`と`β̂₁`の比較）で必ず収束してしまい、`while`ループが
    /// 複数周する経路（2回目以降のS再構築後に収束）を検証できない。
    /// `heteroskedastic_test_columns()`・`weight_type=Robust`の
    /// 実測収束系列（`β̂₀→β̂₁`の相対誤差最大値が約`3.7e-3`、`β̂₁→β̂₂`が約`5.8e-5`、
    /// `GmmEstimator::fit`とは独立に`max_iter`を変えて`params()`を
    /// 比較して確認済み）から、`rtol=1e-3`は1回目では満たされず2回目で満たされることが
    /// 決定的に保証できる（両者の間には約2桁の余裕があり、境界的な値ではない）。
    #[test]
    fn fit_stops_early_after_multiple_iterations_when_gmm_tol_is_satisfied() {
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

        let estimator = GmmEstimator::fit(
            input,
            GmmType::Iterated {
                weight: WeightType::Robust,
                max_iter: 10,
                tol: 1e-3,
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert!(estimator.converged());
        assert_eq!(estimator.n_iter(), 3);
    }

    /// `weight_type=Classical`は点推定が正のスカラー倍不変のため、`gmm_tol`を
    /// 指定していても2回目のS再構築（`β̂₁`）は初期推定`β̂₀`と数値的に完全一致し
    /// （`fit_matches_two_sls_point_estimate_when_weight_type_is_classical`と同じ根拠）、
    /// どんなに厳しい許容誤差でも常に`n_iter=2`で収束する。
    #[test]
    fn fit_with_classical_weight_type_always_converges_at_second_iteration() {
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

        let estimator = GmmEstimator::fit(
            input,
            GmmType::Iterated {
                weight: WeightType::Classical,
                max_iter: 10,
                tol: 1e-300,
            },
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert!(estimator.converged());
        assert_eq!(estimator.n_iter(), 2);
    }

    /// `Iterated`が`max_iter`回（上限）以内に収束せず、
    /// `raise_on_non_convergence=true`（既定）なら`IvError::GmmNonConvergence`。
    /// `weight_type=Robust`は`Classical`と点推定が異なる（既存テスト
    /// `fit_with_robust_weight_type_differs_from_classical_when_heteroskedastic`で
    /// `diff > 1e-4`を確認済み）ため、極めて厳しい許容誤差（`rtol=1e-300`）と組み合わせれば
    /// `max_iter=3`（2回だけSを再構築）では収束しないことが決定的に保証できる。
    #[test]
    fn fit_returns_gmm_non_convergence_error_when_not_converged_within_max_iterations() {
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
            GmmType::Iterated {
                weight: WeightType::Robust,
                max_iter: 3,
                tol: 1e-300,
            },
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::GmmNonConvergence { n_iter: 3 }
        );
    }

    /// 上記と同じ非収束ケースで`raise_on_non_convergence=false`なら、`fit()`は
    /// エラーにせず`converged=false`のまま結果を返す（`nonlinear::common::run_solver`の
    /// `raise_on_non_convergence`と同じ設計、`engine/src/iv/CLAUDE.md`参照）。
    #[test]
    fn fit_returns_result_with_converged_false_when_raise_on_non_convergence_is_false() {
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

        let estimator = GmmEstimator::fit(
            input,
            GmmType::Iterated {
                weight: WeightType::Robust,
                max_iter: 3,
                tol: 1e-300,
            },
            false,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert!(!estimator.converged());
        assert_eq!(estimator.n_iter(), 3);
    }

    /// `TwoStep`は収束判定を行わないため、常に`converged=true`・`n_iter=2`
    /// （`GmmNonConvergence`にはならない）。`fit()`のdocコメント「反復本体」参照。
    #[test]
    fn fit_treats_two_step_as_always_converged() {
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
        assert!(estimator.converged());
        assert_eq!(estimator.n_iter(), 2);
    }
}
