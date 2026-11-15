//! GMMの推定方式（`GmmType`）と、点推定・反復（`gmm_iterate`）・収束判定。
//!
//! 定義は`iv/gmm.rs`のモジュールdoc「`GmmType`と点推定への影響」「収束条件」参照。IVのGMMと固定効果つきの
//! GMM（within変換後の行列を渡す）が共有する。行列だけを受け取り、`IvInput`・系統のエラー型には依存しない。
//!
//! 推定量を通すテストは`iv/gmm/fit/tests/`にある（`IvInput`・`GmmEstimator`が要るため）。

use faer::Mat;

use super::gmm::{GmmSolution, WhitenedInstruments, gmm_solve};
use super::gmm_weight::{WeightType, moment_covariance, moment_factor_rows};
use crate::shared::covariance::{resolve_hac_lags, time_ordering};
use crate::shared::error::CommonError;
use crate::shared::group_codes::GroupCodes;
use crate::shared::validation::{GMM_MAX_ITER_MIN, MAX_ITER_LIMIT};

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

impl GmmType {
    /// `Iterated`の`max_iter`・`tol`が範囲内かを検証する（他の方式は検証するものが無い）。
    ///
    /// # Errors
    /// - `max_iter`が`GMM_MAX_ITER_MIN`未満または`MAX_ITER_LIMIT`超: `CommonError::InvalidGmmMaxIter`
    /// - `tol`が正の有限値でない: `CommonError::InvalidGmmTol`
    pub(crate) fn validate(&self) -> Result<(), CommonError> {
        if let Self::Iterated { max_iter, tol, .. } = self {
            if *max_iter < GMM_MAX_ITER_MIN as usize || *max_iter as u64 > MAX_ITER_LIMIT as u64 {
                return Err(CommonError::InvalidGmmMaxIter {
                    max_iter: *max_iter as i64,
                });
            }
            if !(tol.is_finite() && *tol > 0.0) {
                return Err(CommonError::InvalidGmmTol { gmm_tol: *tol });
            }
        }
        Ok(())
    }

    /// 点推定に使う重みの種類。`OneStep`は`W₀ = (Z'Z)⁻¹`のみで打ち切るので`None`。
    pub(crate) fn weight_type(&self) -> Option<&WeightType> {
        match self {
            Self::OneStep => None,
            Self::TwoStep { weight } | Self::Iterated { weight, .. } => Some(weight),
        }
    }
}

/// GMMの収束判定に使う絶対誤差の床。ユーザーには公開せず内部固定値とする
/// （`gmm.rs`のモジュールdoc「収束条件」参照）。`tests/api_tests`のクロスチェックが
/// 使う`ATOL`の既定値と同じ`1e-8`を踏襲する（`.claude/rules/testing-policy.md`
/// 「許容誤差は相対誤差1e-8を基本」）。
const GMM_CONVERGENCE_ATOL: f64 = 1e-8;

/// 重み`S`（対角を1に揃えた`D S D`）の条件数がこれ以下なら、`S`のCholesky分解で白色化する。
/// 丸め誤差は条件数に比例して広がり、`1e6`で約`1e-10`（倍精度）と、係数・共分散・Wald統計量の
/// 目標精度（相対`1e-8`）に十分収まる。これを超えると、`S`を作らず`A`のQR分解から白色化する。
const WEIGHT_CHOLESKY_MAX_CONDITION: f64 = 1.0e6;

/// [`gmm_iterate`]の戻り値。
pub(crate) struct GmmIteration {
    pub(crate) beta: Mat<f64>,
    /// 最終推定の直交座標つきの解（共分散・全体Wald検定が使う）。
    pub(crate) solution: GmmSolution,
    /// 最終推定の残差 `ê = y - Xβ̂`。
    pub(crate) residuals: Mat<f64>,
    /// 最終推定`beta`に実際に使った重み行列`S`（l×l）。ループが1度も回らない場合は
    /// `classical_s`（σ̂²₀・Z'Z）。
    pub(crate) s_used: Mat<f64>,
    pub(crate) n_iter: i64,
    pub(crate) converged: bool,
    /// `weight_type=Hac`のとき実際に使われたラグ数。それ以外では`None`。
    pub(crate) hac_lags_used: Option<usize>,
}

/// [`gmm_initial_step`]の戻り値。
pub(crate) struct GmmInitialStep {
    /// `Z'Z`（l×l）。
    pub(crate) ztz: Mat<f64>,
    /// 初期推定の直交座標つきの解（`Z = Q_ZR_Z`で白色化した`Ĝ = Q_Z'X`のQR）。
    pub(crate) solution: GmmSolution,
    /// 初期推定 `β̂₀ = β̂(W₀)`（`W₀ = (Z'Z)⁻¹`。2SLSと同じ点推定）。
    pub(crate) beta: Mat<f64>,
    /// 初期推定の残差 `ê₀ = y - Xβ̂₀`。
    pub(crate) residuals: Mat<f64>,
    /// 等分散の重み行列 `σ̂²₀・Z'Z`（`σ̂²₀ = ê₀'ê₀/n`、中心化しない）。点推定には不要だが、
    /// 反復が回らない場合のHansen Jの`S`に必要（`iv/CLAUDE.md`「Hansen Jの`OneStep`」参照）。
    pub(crate) classical_s: Mat<f64>,
}

/// 初期推定（`W₀ = (Z'Z)⁻¹`、`weight_type`によらず共通）と、その残差・`Z'Z`・等分散の`S`を求める。
///
/// # Errors
/// `Z'Z`または`X'ZW₀Z'X`が数値的に特異のとき`CommonError::NearSingularMatrix`。
pub(crate) fn gmm_initial_step(
    z: &Mat<f64>,
    x: &Mat<f64>,
    y: &Mat<f64>,
) -> Result<GmmInitialStep, CommonError> {
    let (n, l) = (z.nrows(), z.ncols());
    let ztz = z.transpose() * z;
    let solution = gmm_solve(x, y, WhitenedInstruments::from_instruments(z)?)?;
    let beta = solution.beta().clone();
    let residuals = y - x * &beta;
    let sigma2_0: f64 = (0..n).map(|i| (*residuals.get(i, 0)).powi(2)).sum::<f64>() / (n as f64);
    let classical_s = Mat::from_fn(l, l, |i, j| sigma2_0 * (*ztz.get(i, j)));
    Ok(GmmInitialStep {
        ztz,
        solution,
        beta,
        residuals,
        classical_s,
    })
}

/// 初期推定（`W₀ = (Z'Z)⁻¹`）から、方式（`gmm_type`）に応じた回数だけ「直前の残差から`S`を
/// 再構築→再推定」を繰り返す。1段階は1度も回らず打ち切り（`weight_type`を持たない）、2段階は
/// 2回目の推定まで、反復は収束するか`max_iter`回に達するまで回す（`gmm.rs`のモジュールdoc
/// 「`GmmType`と点推定への影響」「収束条件」参照）。`s_used`はHansen J検定に使う重み行列で、
/// ループが1度も回らない場合は常に`classical_s`（σ̂²₀・Z'Z）を使う。
#[allow(clippy::too_many_arguments)]
pub(crate) fn gmm_iterate(
    gmm_type: &GmmType,
    weight_type: Option<&WeightType>,
    weight_cluster_codes: Option<&GroupCodes>,
    z: &Mat<f64>,
    x: &Mat<f64>,
    y: &Mat<f64>,
    ztz: &Mat<f64>,
    solution0: GmmSolution,
    beta0: Mat<f64>,
    residuals0: Mat<f64>,
    classical_s: Mat<f64>,
) -> Result<GmmIteration, CommonError> {
    let n = z.nrows();
    let mut solution = solution0;
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
            // 白色化: 古典的な重み`σ̂²Z'Z`は`Z`のQR分解、ロバスト・クラスターは`S`（対角を揃えた`D S D`）の
            // 条件数が小さければ`S`のCholesky分解、大きければ`S = A'A`の`A`のQR分解（`S`を作らない。
            // `A`の条件数の2乗で精度を失わない）。HACは`S`のCholesky分解。
            let whitening = match weight_type {
                WeightType::Classical => WhitenedInstruments::from_instruments(z)?,
                WeightType::Robust | WeightType::Cluster { .. } => {
                    match WhitenedInstruments::from_weight_if_well_conditioned(
                        z,
                        &s_next,
                        WEIGHT_CHOLESKY_MAX_CONDITION,
                    )? {
                        Some(whitening) => whitening,
                        None => {
                            let rows = moment_factor_rows(
                                weight_type,
                                weight_cluster_codes,
                                z,
                                &residuals,
                            )
                            .ok_or_else(|| {
                                CommonError::internal("the moment rows of the weight are missing")
                            })?;
                            WhitenedInstruments::from_factor_rows(z, &rows)?
                        }
                    }
                }
                WeightType::Hac { .. } => WhitenedInstruments::from_weight(z, &s_next)?,
            };
            let solution_next = gmm_solve(x, y, whitening)?;
            let beta_next = solution_next.beta().clone();
            n_iter += 1;

            let just_converged = match tol {
                Some(tol) => gmm_coefficients_converged(&beta, &beta_next, tol),
                None => false,
            };

            beta = beta_next;
            solution = solution_next;
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
        solution,
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
/// `S = L L'`で白色化した操作変数の最小二乗として解く（[`gmm_solve`]。`X'ZWZ'X`は作らない）。
/// 係数だけが要る呼び出し側（C統計量）の入口で、共分散・Wald検定の直交座標は返さない。
pub(crate) fn gmm_point_estimate(
    z: &Mat<f64>,
    x: &Mat<f64>,
    y: &Mat<f64>,
    s: &Mat<f64>,
) -> Result<Mat<f64>, CommonError> {
    let solution = gmm_solve(x, y, WhitenedInstruments::from_weight(z, s)?)?;
    Ok(solution.beta().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iterated(max_iter: usize, tol: f64) -> GmmType {
        GmmType::Iterated {
            weight: WeightType::Robust,
            max_iter,
            tol,
        }
    }

    /// `max_iter`は3以上`MAX_ITER_LIMIT`以下、`tol`は正の有限値。境界の内側は受理する。
    #[test]
    fn validate_accepts_the_boundaries_and_rejects_outside() {
        assert_eq!(GmmType::OneStep.validate(), Ok(()));
        assert_eq!(
            GmmType::TwoStep {
                weight: WeightType::Classical
            }
            .validate(),
            Ok(())
        );
        assert_eq!(iterated(3, 1e-12).validate(), Ok(()));
        assert_eq!(
            iterated(MAX_ITER_LIMIT as usize, f64::MIN_POSITIVE).validate(),
            Ok(())
        );
        for invalid in [0, 1, 2, MAX_ITER_LIMIT as usize + 1] {
            assert_eq!(
                iterated(invalid, 1e-6).validate(),
                Err(CommonError::InvalidGmmMaxIter {
                    max_iter: invalid as i64
                })
            );
        }
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = iterated(10, invalid).validate().unwrap_err();
            assert!(
                matches!(err, CommonError::InvalidGmmTol { .. }),
                "tol={invalid}: {err:?}"
            );
        }
    }

    /// `max_iter`が先に検証される（両方が不正なら`max_iter`のエラー）。
    #[test]
    fn validate_reports_max_iter_before_tol() {
        assert!(matches!(
            iterated(1, -1.0).validate(),
            Err(CommonError::InvalidGmmMaxIter { max_iter: 1 })
        ));
    }

    #[test]
    fn weight_type_is_none_only_for_one_step() {
        assert_eq!(GmmType::OneStep.weight_type(), None);
        assert_eq!(
            GmmType::TwoStep {
                weight: WeightType::Robust
            }
            .weight_type(),
            Some(&WeightType::Robust)
        );
        assert_eq!(iterated(5, 1e-6).weight_type(), Some(&WeightType::Robust));
    }

    /// 初期推定は`(X'Z(Z'Z)⁻¹Z'X)⁻¹X'Z(Z'Z)⁻¹Z'y`（2SLS）で、丁度識別では`Z'X`が正方なので`(Z'X)⁻¹Z'y`
    /// （IV推定）に一致する。`σ̂²₀・Z'Z`は残差の平方平均（中心化しない）を掛けたもの。
    #[test]
    fn initial_step_is_two_stage_least_squares_and_scales_ztz_by_the_residual_variance() {
        // y = 2 + 3x + 疑似ノイズ。操作変数は切片と`z`（丁度識別）。
        let n = 8;
        let z_col = [0.5, -1.0, 2.0, 0.0, 1.5, -0.5, 1.0, -2.0];
        let x_col = [0.6, -0.8, 2.2, 0.1, 1.4, -0.4, 1.1, -1.7];
        let noise = [0.1, -0.2, 0.05, 0.15, -0.1, 0.2, -0.05, 0.0];
        let y = Mat::from_fn(n, 1, |i, _| 2.0 + 3.0 * x_col[i] + noise[i]);
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_col[i] });
        let z = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { z_col[i] });

        let step = gmm_initial_step(&z, &x, &y).unwrap();

        // 丁度識別のIV推定を手で解く: Z'X b = Z'y（2×2）。
        let zx = z.transpose() * &x;
        let zy = z.transpose() * &y;
        let det = zx.get(0, 0) * zx.get(1, 1) - zx.get(0, 1) * zx.get(1, 0);
        let b0 = (zy.get(0, 0) * zx.get(1, 1) - zx.get(0, 1) * zy.get(1, 0)) / det;
        let b1 = (zx.get(0, 0) * zy.get(1, 0) - zy.get(0, 0) * zx.get(1, 0)) / det;
        assert!((step.beta.get(0, 0) - b0).abs() < 1e-10);
        assert!((step.beta.get(1, 0) - b1).abs() < 1e-10);

        let sigma2: f64 = (0..n)
            .map(|i| step.residuals.get(i, 0).powi(2))
            .sum::<f64>()
            / n as f64;
        for a in 0..2 {
            for b in 0..2 {
                let want = sigma2 * step.ztz.get(a, b);
                assert!((step.classical_s.get(a, b) - want).abs() < 1e-12 * want.abs().max(1.0));
            }
        }
    }
}
