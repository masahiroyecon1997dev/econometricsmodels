//! 予測・限界効果の対象（`MarginalEffectsTarget`）ごとの値`predicted_value`と、`w`・`s`
//! （限界効果のデルタ法に使う）。

use super::mills::{TruncationRatios, UncensoredTerms, boundary_terms};
use crate::nonlinear::common::MleError;
use statrs::distribution::Normal;

/// `marginal_effects`が評価する対象（McDonald-Moffitt 1980）。Logit/Probitの
/// `dydx_and_jacobian`型の共通化はしない（対象ごとに式が異なり、
/// 同型の`(w,s)`分解に無理に収める価値がないと判断した。
/// `docs/spec/nonlinear-common.md`6章参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarginalEffectsTarget {
    /// 潜在変数の期待値 `E[y*|x] = x'β`
    ExpectedLatent,
    /// 打ち切りを考慮した観測値の期待値 `E[y|x]`（既定、実測`y`と直接比較できる量）
    ExpectedObserved,
    /// 非打ち切り確率 `P(uncensored|x)`
    ProbUncensored,
    /// 非打ち切り領域での条件付き期待値 `E[y|lower<y<upper,x]`（切断正規分布の平均）。
    /// 下限0のみの左打ち切りなら `E[y|y>0,x]`。`E[y|x]`との関係は
    /// `E[y|x]=lower·P(下限)+upper·P(上限)+P(uncensored|x)·E[y|lower<y<upper,x]`
    ExpectedUncensored,
}

/// `target`ごとの限界効果`dydx_j=w*βⱼ`（`marginal_effects_from_tobit_w_s`のdocコメント
/// 「数式」参照）における`w`とその勾配`(s_beta, s_sigma)`を、評価点`x_point`
/// （`k`要素、`at="overall"`なら観測`i`の行、`at="mean"`/`"median"`なら代表点）で計算する。
///
/// ## 数式
///
/// `μ=x_point'β`とし、`za=(lower-μ)/σ`・`zb=(upper-μ)/σ`（`boundary_terms`、`None`の
/// 方向は`φ=0`かつ`Φ`は0/1の定数として扱う）とすると、非打ち切り確率は
/// `W(θ)=Φ(zb)-Φ(za)`（左のみ打ち切りなら`Φ(zb)=1`で`W=Φ(za)`相当、右のみなら
/// `Φ(za)=0`で`W=Φ(zb)`相当、両側打ち切りなら両方が効く——単一の式で3ケースとも
/// 表現できる）。
///
/// - `ExpectedLatent`（`E[y*|x]=x'β`）: `dydx_j=βⱼ`（`x_point`に依存しない自明な形。
///   `w=1`・`s_beta=0`・`s_sigma=0`）
/// - `ExpectedObserved`（`E[y|x]`）: `w=W(θ)`。`∂W/∂μ=(φ(za)-φ(zb))/σ`、
///   `∂W/∂σ=(za·φ(za)-zb·φ(zb))/σ`（`s_beta_m=∂W/∂μ・x_point[m]`、連鎖律`∂μ/∂βₘ=x_point[m]`）
/// - `ProbUncensored`（`P(uncensored|x)=W(θ)`自体）: `w=∂W/∂μ=(φ(za)-φ(zb))/σ`。
///   その勾配は`∂²W/∂μ²=(za·φ(za)-zb·φ(zb))/σ²`・
///   `∂(∂W/∂μ)/∂σ=(φ(za)(za²-1)-φ(zb)(zb²-1))/σ²`
/// - `ExpectedUncensored`（`E[y|uncensored,x]`）: `uncensored_w_and_s`
///
/// いずれも境界項がある方向のみ`φ≠0`となるため（`boundary_terms`のNone時`φ=0`）、
/// 左/右/両側打ち切りいずれでも同じ式で正しく計算できる（両側打ち切りは
/// McDonald-Moffitt(1980)の一般形、片側打ち切りはその特殊ケースとして導出済み、
/// `docs/spec/tobit-spec.md`3.5節参照）。
pub(super) fn target_w_and_s(
    target: MarginalEffectsTarget,
    x_point: &[f64],
    beta: &[f64],
    sigma: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    normal: &Normal,
) -> (f64, Vec<f64>, f64) {
    let k = beta.len();
    if target == MarginalEffectsTarget::ExpectedLatent {
        return (1.0, vec![0.0; k], 0.0);
    }

    let mu: f64 = (0..k).map(|m| x_point[m] * beta[m]).sum();
    if target == MarginalEffectsTarget::ExpectedUncensored {
        return uncensored_w_and_s(x_point, mu, sigma, lower, upper, normal);
    }
    let (za, phi_za, cdf_za) = boundary_terms(lower, mu, sigma, normal, true);
    let (zb, phi_zb, cdf_zb) = boundary_terms(upper, mu, sigma, normal, false);

    match target {
        MarginalEffectsTarget::ExpectedLatent | MarginalEffectsTarget::ExpectedUncensored => {
            unreachable!("handled by early return above")
        }
        MarginalEffectsTarget::ExpectedObserved => {
            let w = cdf_zb - cdf_za;
            let dw_dmu = (phi_za - phi_zb) / sigma;
            let s_beta: Vec<f64> = x_point.iter().map(|&x_m| dw_dmu * x_m).collect();
            let s_sigma = (za * phi_za - zb * phi_zb) / sigma;
            (w, s_beta, s_sigma)
        }
        MarginalEffectsTarget::ProbUncensored => {
            let w = (phi_za - phi_zb) / sigma;
            let dw_dmu = (za * phi_za - zb * phi_zb) / (sigma * sigma);
            let s_beta: Vec<f64> = x_point.iter().map(|&x_m| dw_dmu * x_m).collect();
            let s_sigma = (phi_za * (za * za - 1.0) - phi_zb * (zb * zb - 1.0)) / (sigma * sigma);
            (w, s_beta, s_sigma)
        }
    }
}

/// `ExpectedUncensored`を信頼できる精度で評価できる、非打ち切り区間の幅`(upper-lower)/σ`の下限。
/// `w`の絶対誤差は約`ε/δ²`、真値は`≈δ²/12`なので相対誤差は約`12ε/δ⁴`で、デルタ法に使う
/// `∂w/∂μ`・`∂w/∂σ`の誤差はさらに大きい（`ε/δ³`級。`|z|=10`で`δ=1e-3`の`s`は
/// 絶対`2e-8`で`w≈1e-7`の`1/5`）。`δ=1e-2`で`w`の相対誤差が全領域（`za∈[-10,10]`）で
/// `6e-7`以下（80桁参照の実測）、`δ=1e-3`では`w`の相対誤差が`3e-3`〜、`s`が`w`の1%超になる。
/// `σ̂`が区間幅の100倍以上（非打ち切りの確率が約0.4%未満）でほぼ識別できない当てはめのときだけ
/// 超える。
pub(super) const MIN_UNCENSORED_WIDTH_RATIO: f64 = 1e-2;

/// 両側打ち切りで非打ち切り区間が`σ`に比べて狭すぎる場合に
/// [`MleError::UncensoredIntervalTooNarrow`]を返す（`MIN_UNCENSORED_WIDTH_RATIO`）。
/// 片側打ち切りは区間が無限大なので常に`Ok`。幅`(upper-lower)/σ`はモデル（`σ̂`）で1つに
/// 決まり観測に依存しないので、呼び出しごとに1回検査すれば足りる。
pub(super) fn ensure_uncensored_interval_resolvable(
    lower: Option<f64>,
    upper: Option<f64>,
    sigma: f64,
) -> Result<(), MleError> {
    if let (Some(lower), Some(upper)) = (lower, upper)
        && (upper - lower) / sigma < MIN_UNCENSORED_WIDTH_RATIO
    {
        return Err(MleError::UncensoredIntervalTooNarrow {
            lower,
            upper,
            sigma,
        });
    }
    Ok(())
}

/// `ExpectedUncensored`（`m(μ,σ)=E[y|lower<y<upper,x]=μ+σ(Pa-Pb)`）の`(w, s_beta, s_sigma)`。
///
/// ## 数式
///
/// `m`は`x`に`μ=x'β`を通してしか依存しないので`dydx_j=w·βⱼ`、`w=∂m/∂μ`。
/// `Pa=φ(za)/W`・`Pb=φ(zb)/W`・`ρ1=Pa-Pb`・`ρ2=za·Pa-zb·Pb`とおくと
/// `w=1+ρ2-ρ1²`（切断正規分布の分散を`σ²`で割った値で、`(0,1)`に入る）。
/// `za`/`zb`についての偏微分は`W_a=-φa`・`W_b=φb`などから
///
/// - `∂ρ1/∂za=Pa(ρ1-za)`・`∂ρ1/∂zb=Pb(zb-ρ1)`
/// - `∂ρ2/∂za=Pa(1-za²+ρ2)`・`∂ρ2/∂zb=Pb(zb²-1-ρ2)`
/// - `w_a=∂ρ2/∂za-2ρ1∂ρ1/∂za`・`w_b=∂ρ2/∂zb-2ρ1∂ρ1/∂zb`
///
/// で、`∂za/∂μ=∂zb/∂μ=-1/σ`・`∂za/∂σ=-za/σ`・`∂zb/∂σ=-zb/σ`より
/// `∂w/∂μ=-(w_a+w_b)/σ`・`∂w/∂σ=-(za·w_a+zb·w_b)/σ`。左のみ打ち切り（`zb`なし）で
/// `Pa=λ(za)`とすると`w=1+za·λ-λ²`（McDonald-Moffittの`1-zλ-λ²`型の式）に一致する。
///
/// ## 数値安定性
///
/// 上の式をそのまま書くと、`|z|`が大きい裾（`Pa≈za`）では`za²`と`za·Pa`の大きな数同士の
/// 引き算になり、`w_a`の絶対誤差が`ε·z³`で増える（`z=72`の観測を含む実データで`s_sigma`の
/// 相対誤差が`5e-9`、限界効果の標準誤差が`1e-8`ずれた）。
/// `A=Pa-za`・`B=zb+Pb`（`truncation_pdf_ratios`が引き算なしに返す）で整理すると
///
/// - `w=1-Pa·A-Pb·B+2·Pa·Pb`
/// - `w_a=Pa[1+za·A-zb·Pb-2ρ1(A-Pb)]`
/// - `w_b=Pb[-1+zb·B-za·Pa-2ρ1(B-Pa)]`
///
/// となり、括弧内は`O(1)`同士の和なので絶対誤差は`ε·Pa`程度に収まる。`W→0`となる裾でも
/// `Pa`・`Pb`が直接計算されるため割り算は破綻しない。`w`は理論上`(0,1)`なので、丸め誤差で
/// 範囲外になった値が平均に混ざらないよう`[0,1]`にクランプする。
///
/// ## 既知の限界
///
/// - **遠い裾の相対精度**: 片側の裾で`|z|`が大きいと、括弧内の`O(1)`同士が`O(z⁻⁴)`まで
///   打ち消し合うため、絶対誤差は小さいままでも`w`・`w_a`・`w_b`の**相対**精度は
///   `|z|≳1e3`から落ち、`|z|≳1e4`で`w_a`は符号も保証されない（`w≈1/z²`・`w_a≈-2/z³`自体が
///   無視できる大きさなので、全観測の平均には影響しない）。
/// - **非打ち切り区間が極端に狭い場合**: 区間幅`δ=zb-za=(upper-lower)/σ`が小さいと、上側/下側の
///   裾の`q-r·p`や中央の`W`・`1-Pa·A-Pb·B+…`が桁落ちする（`MIN_UNCENSORED_WIDTH_RATIO`の
///   docコメント参照）。この関数は`δ`を検査しない。`TobitEstimator`の`predict`・
///   `predict_new_data`・`marginal_effects`が入口で`ensure_uncensored_interval_resolvable`
///   により`δ<1e-2`を`MleError::UncensoredIntervalTooNarrow`として拒否する。
pub(super) fn uncensored_w_and_s(
    x_point: &[f64],
    mu: f64,
    sigma: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    normal: &Normal,
) -> (f64, Vec<f64>, f64) {
    let UncensoredTerms {
        za,
        zb,
        ratios:
            TruncationRatios {
                pa,
                pb,
                a_gap,
                b_gap,
            },
    } = UncensoredTerms::new(mu, sigma, lower, upper, normal);
    let rho1 = pa - pb;
    let w = (1.0 - pa * a_gap - pb * b_gap + 2.0 * pa * pb).clamp(0.0, 1.0);

    let w_a = pa * (1.0 + za * a_gap - zb * pb - 2.0 * rho1 * (a_gap - pb));
    let w_b = pb * (-1.0 + zb * b_gap - za * pa - 2.0 * rho1 * (b_gap - pa));
    let dw_dmu = -(w_a + w_b) / sigma;
    let s_beta: Vec<f64> = x_point.iter().map(|&x_m| dw_dmu * x_m).collect();
    let s_sigma = -(za * w_a + zb * w_b) / sigma;
    (w, s_beta, s_sigma)
}

/// `predict`が返す予測量`target`の値を、線形予測子`mu=x'β`・`σ`・打ち切り境界から計算する
/// （McDonald-Moffitt 1980）。`target_w_and_s`と同じ`boundary_terms`（`lower`/`upper`の
/// `None`側を`Φ(∓∞)=0/1`・`φ(∓∞)=0`相当の定数として扱う統一的な扱い）を再利用するため、
/// 左のみ・右のみ・両側打ち切りいずれでも同じ式で正しく計算できる（`target_w_and_s`の
/// docコメント「数式」と同じ考え方。こちらは値そのもの、`target_w_and_s`はその微分）。
///
/// `ExpectedUncensored`は`E[y|lower<y<upper,x]=μ+σ(Pa-Pb)`（`UncensoredTerms::mean`、
/// 裾でも安定な`truncation_pdf_ratios`経由）。
///
/// `E[y|x]`の閉形式は`Φ(za)*lower + (1-Φ(zb))*upper + (Φ(zb)-Φ(za))*mu - σ(φ(zb)-φ(za))`
/// （両側打ち切りの一般形。`lower`/`upper`が`None`の側は`boundary_terms`の規約により
/// 係数が0になるため、ダミー値`lower.unwrap_or(0.0)`/`upper.unwrap_or(0.0)`を掛けても
/// 結果に影響しない）。`P(uncensored|x)`は`Φ(zb)-Φ(za)`そのもの。
pub(super) fn predicted_value(
    target: MarginalEffectsTarget,
    mu: f64,
    sigma: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    normal: &Normal,
) -> f64 {
    if target == MarginalEffectsTarget::ExpectedLatent {
        return mu;
    }
    if target == MarginalEffectsTarget::ExpectedUncensored {
        return UncensoredTerms::new(mu, sigma, lower, upper, normal).mean(mu, sigma);
    }
    let (_, phi_za, cdf_za) = boundary_terms(lower, mu, sigma, normal, true);
    let (_, phi_zb, cdf_zb) = boundary_terms(upper, mu, sigma, normal, false);
    match target {
        MarginalEffectsTarget::ExpectedLatent | MarginalEffectsTarget::ExpectedUncensored => {
            unreachable!("handled by early return above")
        }
        MarginalEffectsTarget::ExpectedObserved => {
            let lower_c = lower.unwrap_or(0.0);
            let upper_c = upper.unwrap_or(0.0);
            cdf_za * lower_c + (1.0 - cdf_zb) * upper_c + (cdf_zb - cdf_za) * mu
                - sigma * (phi_zb - phi_za)
        }
        MarginalEffectsTarget::ProbUncensored => cdf_zb - cdf_za,
    }
}

#[cfg(test)]
mod tests {

    use super::super::test_support::*;
    use super::*;

    use statrs::distribution::ContinuousCDF;
    use statrs::distribution::Normal;

    /// `(lower,upper)`の代表的な3構成（左のみ・右のみ・両側）。
    fn bounds_configs() -> [(Option<f64>, Option<f64>); 3] {
        [
            (Some(0.0), None),
            (None, Some(1.0)),
            (Some(-1.0), Some(1.5)),
        ]
    }

    /// 極端な入力（`z`が`1e8`、`w`が丸め誤差に支配される領域を含む）でも、`m`・`w`・`s`が
    /// 有限で、`w`は`[0,1]`に収まり`m`は非打ち切り領域`[lower,upper]`に収まる。
    #[test]
    fn expected_uncensored_stays_finite_and_bounded_for_extreme_inputs() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let sigma = 1.0;
        for (lower, upper) in bounds_configs() {
            for mu in [
                -1e8, -1e4, -100.0, -39.0, -8.0, 0.0, 0.5, 8.0, 39.0, 100.0, 1e4, 1e8,
            ] {
                let m = predicted_value(
                    MarginalEffectsTarget::ExpectedUncensored,
                    mu,
                    sigma,
                    lower,
                    upper,
                    &normal,
                );
                let (w, s_beta, s_sigma) = target_w_and_s(
                    MarginalEffectsTarget::ExpectedUncensored,
                    &[1.0],
                    &[mu],
                    sigma,
                    lower,
                    upper,
                    &normal,
                );
                let label = format!("lower={lower:?}, upper={upper:?}, mu={mu}");
                assert!(m.is_finite(), "{label}: m={m}");
                assert!(w.is_finite() && (0.0..=1.0).contains(&w), "{label}: w={w}");
                assert!(s_beta[0].is_finite(), "{label}: s_beta={}", s_beta[0]);
                assert!(s_sigma.is_finite(), "{label}: s_sigma={s_sigma}");
                // 非打ち切り領域への条件付き期待値は領域内（境界は丸め誤差分だけ許す）
                let eps = 1e-9 * (1.0 + m.abs());
                if let Some(l) = lower {
                    assert!(m >= l - eps, "{label}: m={m} < lower");
                }
                if let Some(u) = upper {
                    assert!(m <= u + eps, "{label}: m={m} > upper");
                }
            }
        }
    }

    /// 遠い裾では`E[y|uncensored]`は最寄りの境界に漸近する（`μ`が下限よりずっと下なら
    /// `m→lower`、上限よりずっと上なら`m→upper`）。素直な`φ/W`だと`W=0`で`NaN`になる領域。
    #[test]
    fn expected_uncensored_converges_to_nearest_bound_in_far_tail() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let m_left = predicted_value(
            MarginalEffectsTarget::ExpectedUncensored,
            -100.0,
            1.0,
            Some(0.0),
            None,
            &normal,
        );
        // 左のみ: m = μ+σλ(za), za=100 → λ=100.00999800099926 → m=0.00999800099926
        assert_relative_eq(m_left, 0.009_998_000_999_26, 1e-9, "left-only far tail");
        let m_right = predicted_value(
            MarginalEffectsTarget::ExpectedUncensored,
            100.0,
            1.0,
            None,
            Some(0.0),
            &normal,
        );
        assert_relative_eq(m_right, -0.009_998_000_999_26, 1e-9, "right-only far tail");
        let m_two_sided_above = predicted_value(
            MarginalEffectsTarget::ExpectedUncensored,
            1000.0,
            1.0,
            Some(-1.0),
            Some(2.0),
            &normal,
        );
        // 上限より遥か上（`zb=-998`）: m = μ - σ·λ(998) = 1000 - (998 + gap), gap = λ(998)-998 ≈ 1/(998+2/998)
        assert!(
            (m_two_sided_above - (2.0 - 1.0 / (998.0 + 2.0 / 998.0))).abs() < 1e-9,
            "m={m_two_sided_above}"
        );
    }

    /// 全領域（左のみ・右のみ・両側、`za=0`・`zb=0`の領域境界を含む）で、
    /// `predicted_value`が独立の閉形式`expected_uncensored_closed_form`と一致する。
    #[test]
    fn predicted_value_expected_uncensored_matches_independent_closed_form() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        for (lower, upper) in bounds_configs() {
            for sigma in [0.8, 1.0, 3.0] {
                for mu in [-3.5, -1.5, -1.0, 0.0, 0.3, 1.0, 1.5, 2.5, 4.5] {
                    let actual = predicted_value(
                        MarginalEffectsTarget::ExpectedUncensored,
                        mu,
                        sigma,
                        lower,
                        upper,
                        &normal,
                    );
                    let expected = expected_uncensored_closed_form(mu, sigma, lower, upper);
                    assert!(
                        (actual - expected).abs() < 1e-9 * (1.0 + expected.abs()),
                        "lower={lower:?}, upper={upper:?}, sigma={sigma}, mu={mu}: \
                         actual={actual}, expected={expected}"
                    );
                }
            }
        }
    }

    /// 分解恒等式 `E[y|x]=lower·P(下限)+upper·P(上限)+P(uncensored|x)·E[y|uncensored,x]`
    /// （下限0のみの左打ち切りなら`E[y|x]=P(y>0|x)·E[y|y>0,x]`）。既存の
    /// `ExpectedObserved`・`ProbUncensored`と新`ExpectedUncensored`が互いに整合することの確認。
    #[test]
    fn expected_observed_equals_censoring_mass_plus_prob_uncensored_times_expected_uncensored() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        for (lower, upper) in bounds_configs() {
            for sigma in [0.8, 1.0, 3.0] {
                for mu in [-3.5, -1.0, 0.0, 0.7, 1.5, 2.5, 4.5] {
                    let predict =
                        |target| predicted_value(target, mu, sigma, lower, upper, &normal);
                    let p_lower = lower.map_or(0.0, |l| normal.cdf((l - mu) / sigma));
                    let p_upper = upper.map_or(0.0, |u| 1.0 - normal.cdf((u - mu) / sigma));
                    let decomposed = lower.unwrap_or(0.0) * p_lower
                        + upper.unwrap_or(0.0) * p_upper
                        + predict(MarginalEffectsTarget::ProbUncensored)
                            * predict(MarginalEffectsTarget::ExpectedUncensored);
                    let observed = predict(MarginalEffectsTarget::ExpectedObserved);
                    assert!(
                        (observed - decomposed).abs() < 1e-9 * (1.0 + observed.abs()),
                        "lower={lower:?}, upper={upper:?}, sigma={sigma}, mu={mu}: \
                         observed={observed}, decomposed={decomposed}"
                    );
                }
            }
        }
    }

    /// `w=∂m/∂μ`・`s_beta=∂w/∂μ·x`・`s_sigma=∂w/∂σ`を、独立の閉形式`m`の中心差分
    /// （`w`は`m`の`μ`微分、`s`は`w`の微分なので`m`の2階・3階の差分）と突き合わせる。
    /// 3構成 × `μ`・`σ`の格子で、領域の切り替え（`za>=0`・`zb<=0`・中央）をまたぐ。
    /// `w`の比較に使うオラクル`expected_uncensored_closed_form`は`statrs`の`cdf`（絶対誤差
    /// 約`1.4e-11`）を使うため、中心差分（`h=1e-4`）経由で約`1e-7`・`h`で割る分`1e-9`級の誤差を
    /// 含む（実測の最大差`1.1e-9`）。許容`1e-8`はその約10倍。`s`の比較は本番の`w`の差分なので
    /// 中心差分の打ち切り誤差`h²·|w'''|/6`だけ（実測の最大`1.4e-8`）で、許容は`1e-7`。
    #[test]
    fn target_w_and_s_expected_uncensored_matches_finite_differences_of_closed_form_mean() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let w_of = |mu: f64, sigma: f64, lower, upper| {
            target_w_and_s(
                MarginalEffectsTarget::ExpectedUncensored,
                &[1.0],
                &[mu],
                sigma,
                lower,
                upper,
                &normal,
            )
            .0
        };
        for (lower, upper) in bounds_configs() {
            for sigma in [0.7, 1.0, 2.5] {
                for mu in [-3.0, -1.0, -0.4, 0.0, 0.4, 1.0, 1.5, 2.0, 4.0] {
                    let h = 1e-4;
                    let m = |mu: f64, sigma: f64| {
                        expected_uncensored_closed_form(mu, sigma, lower, upper)
                    };
                    let label = format!("lower={lower:?}, upper={upper:?}, sigma={sigma}, mu={mu}");

                    let (w, s_beta, s_sigma) = target_w_and_s(
                        MarginalEffectsTarget::ExpectedUncensored,
                        &[1.0],
                        &[mu],
                        sigma,
                        lower,
                        upper,
                        &normal,
                    );
                    let w_fd = (m(mu + h, sigma) - m(mu - h, sigma)) / (2.0 * h);
                    assert!((w - w_fd).abs() < 1e-8, "{label}: w={w}, fd={w_fd}");
                    assert!((0.0..=1.0).contains(&w), "{label}: w={w}");

                    let dw_dmu_fd = (w_of(mu + h, sigma, lower, upper)
                        - w_of(mu - h, sigma, lower, upper))
                        / (2.0 * h);
                    assert!(
                        (s_beta[0] - dw_dmu_fd).abs() < 1e-7,
                        "{label}: s_beta={}, fd={dw_dmu_fd}",
                        s_beta[0]
                    );
                    let dw_dsigma_fd = (w_of(mu, sigma + h, lower, upper)
                        - w_of(mu, sigma - h, lower, upper))
                        / (2.0 * h);
                    assert!(
                        (s_sigma - dw_dsigma_fd).abs() < 1e-7,
                        "{label}: s_sigma={s_sigma}, fd={dw_dsigma_fd}"
                    );
                }
            }
        }
    }

    /// 非打ち切り区間の幅`(upper-lower)/σ`が`MIN_UNCENSORED_WIDTH_RATIO`未満のときだけ
    /// `UncensoredIntervalTooNarrow`を返す。境界ちょうど（`=1e-2`）は通し、片側打ち切り
    /// （`lower`か`upper`が`None`）は`σ`がどれほど大きくても通す。
    #[test]
    fn ensure_uncensored_interval_resolvable_rejects_only_narrow_two_sided_intervals() {
        // 幅/σ = 1/100 = 0.01（境界ちょうど）は通す
        assert!(ensure_uncensored_interval_resolvable(Some(0.0), Some(1.0), 100.0).is_ok());
        // 幅/σ = 1/100.5 < 0.01 は拒否し、境界とσを持つ
        match ensure_uncensored_interval_resolvable(Some(0.0), Some(1.0), 100.5) {
            Err(MleError::UncensoredIntervalTooNarrow {
                lower,
                upper,
                sigma,
            }) => {
                assert_eq!((lower, upper, sigma), (0.0, 1.0, 100.5));
            }
            other => panic!("expected UncensoredIntervalTooNarrow, got {other:?}"),
        }
        // 片側打ち切りは区間が無限大なので`σ`が巨大でも通す
        assert!(ensure_uncensored_interval_resolvable(Some(0.0), None, 1e12).is_ok());
        assert!(ensure_uncensored_interval_resolvable(None, Some(0.0), 1e12).is_ok());
        // 広い区間は通す
        assert!(ensure_uncensored_interval_resolvable(Some(-5.0), Some(5.0), 1.0).is_ok());
    }

    /// `fit()`を通さず`predicted_value`・`target_w_and_s`を直接検証するproperty-basedテスト。
    mod proptests {
        use super::*;

        use proptest::prelude::*;

        // ── ExpectedUncensored（`E[y|lower<y<upper,x]`）の不変条件 ──────────
        // 以降は`fit()`を介さず`predicted_value`・`target_w_and_s`を直接検証する。
        // 乱数の`(μ, σ, lower, upper)`は、非打ち切り区間の幅/σが
        // `MIN_UNCENSORED_WIDTH_RATIO`以上になる範囲に限る（それ未満は`TobitEstimator`が
        // `UncensoredIntervalTooNarrow`で拒否する）。`μ`は`±60`まで振って`z`が数十に
        // なる裾（連分数・鏡像変換の分岐）も踏む。

        /// `(mu, sigma, lower, upper)`。`lower`/`upper`は片側・両側のいずれかで、
        /// 両側のときは幅が`σ`の`1/100`以上。
        type UncensoredCase = (f64, f64, Option<f64>, Option<f64>);

        fn uncensored_case_strategy() -> impl Strategy<Value = UncensoredCase> {
            (
                -60.0f64..60.0,
                0.2f64..5.0,
                -3.0f64..3.0,
                0.0f64..8.0,
                0u8..3,
            )
                .prop_map(|(mu, sigma, lower, width, kind)| {
                    // 幅の下限: `δ=width/σ>=0.01`を満たすよう`σ/100`以上にする
                    let width = width + sigma / 100.0;
                    let (lo, up) = match kind {
                        0 => (Some(lower), None),
                        1 => (None, Some(lower)),
                        _ => (Some(lower), Some(lower + width)),
                    };
                    (mu, sigma, lo, up)
                })
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(512))]

            /// 分解恒等式: `E[y|x] = lower·P(下限) + upper·P(上限) + P(uncensored|x)·E[y|uncensored,x]`
            /// （下限0のみの左打ち切りなら`E[y|x]=P(y>0|x)·E[y|y>0,x]`）。既存の`ExpectedObserved`・
            /// `ProbUncensored`（`statrs`の`cdf`、絶対誤差`1e-11`級）と新`ExpectedUncensored`
            /// （別の安定実装）が、乱数の`(μ,σ,境界)`全域で整合する。裾（`z`が数十）では
            /// `P(uncensored)≈0`なので右辺は`lower`/`upper`の項が支配する。
            #[test]
            fn decomposition_identity_holds_for_random_inputs(
                (mu, sigma, lower, upper) in uncensored_case_strategy(),
            ) {
                let normal = Normal::standard();
                let value = |t| predicted_value(t, mu, sigma, lower, upper, &normal);
                let p_lower = lower.map_or(0.0, |l| normal.cdf((l - mu) / sigma));
                let p_upper = upper.map_or(0.0, |u| 1.0 - normal.cdf((u - mu) / sigma));
                let decomposed = lower.unwrap_or(0.0) * p_lower
                    + upper.unwrap_or(0.0) * p_upper
                    + value(MarginalEffectsTarget::ProbUncensored)
                        * value(MarginalEffectsTarget::ExpectedUncensored);
                let observed = value(MarginalEffectsTarget::ExpectedObserved);
                prop_assert!(
                    (observed - decomposed).abs() <= 1e-8 * (1.0 + observed.abs()),
                    "observed={observed}, decomposed={decomposed}"
                );
            }

            /// 値域: `E[y|lower<y<upper,x]`は非打ち切り領域`[lower, upper]`に収まり
            /// （境界は丸め誤差分だけ許す）、重み`w`（切断分散/σ²）は`[0,1]`、`s_beta`・
            /// `s_sigma`は有限。裾（`z`が数十）でも破綻しない。
            #[test]
            fn expected_uncensored_stays_within_bounds_and_w_in_unit_interval(
                (mu, sigma, lower, upper) in uncensored_case_strategy(),
            ) {
                let normal = Normal::standard();
                let m = predicted_value(
                    MarginalEffectsTarget::ExpectedUncensored, mu, sigma, lower, upper, &normal,
                );
                let (w, s_beta, s_sigma) = target_w_and_s(
                    MarginalEffectsTarget::ExpectedUncensored,
                    &[1.0], &[mu], sigma, lower, upper, &normal,
                );
                prop_assert!(m.is_finite() && w.is_finite() && s_beta[0].is_finite() && s_sigma.is_finite());
                prop_assert!((0.0..=1.0).contains(&w), "w={w}");
                let eps = 1e-9 * (1.0 + m.abs());
                if let Some(l) = lower {
                    prop_assert!(m >= l - eps, "m={m} < lower={l}");
                }
                if let Some(u) = upper {
                    prop_assert!(m <= u + eps, "m={m} > upper={u}");
                }
            }

            /// 鏡像対称性: `(μ, lower, upper)→(-μ, -upper, -lower)`で、平均`m`は符号反転、
            /// 重み`w`は不変、`∂w/∂μ`は符号反転、`∂w/∂σ`は不変。左のみ↔右のみに入れ替わり、
            /// 両側の上側の裾（`za>=0`）↔下側の裾（`zb<=0`）が別の分岐（符号反転した
            /// `upper_tail_pdf_ratios`）を通るため、片方の分岐の符号・入れ替え（`Pa↔Pb`・`A↔B`）の
            /// バグを検出できる。
            #[test]
            fn mirror_symmetry_negates_mean_and_derivative_and_keeps_weight(
                (mu, sigma, lower, upper) in uncensored_case_strategy(),
            ) {
                let normal = Normal::standard();
                let (mirror_lower, mirror_upper) = (upper.map(|u| -u), lower.map(|l| -l));
                let m = predicted_value(
                    MarginalEffectsTarget::ExpectedUncensored, mu, sigma, lower, upper, &normal,
                );
                let m_mirror = predicted_value(
                    MarginalEffectsTarget::ExpectedUncensored, -mu, sigma, mirror_lower, mirror_upper, &normal,
                );
                let (w, s_beta, s_sigma) = target_w_and_s(
                    MarginalEffectsTarget::ExpectedUncensored, &[1.0], &[mu], sigma, lower, upper, &normal,
                );
                let (w_mirror, s_beta_mirror, s_sigma_mirror) = target_w_and_s(
                    MarginalEffectsTarget::ExpectedUncensored, &[1.0], &[-mu], sigma, mirror_lower, mirror_upper, &normal,
                );
                // `z`が数十の裾では`ε·z`〜`ε·z²`の丸め誤差があるので`z`に応じて許容を広げる
                let z_max = [lower, upper]
                    .iter()
                    .flatten()
                    .map(|b| ((b - mu) / sigma).abs())
                    .fold(0.0f64, f64::max);
                let tol = 1e-9 * (1.0 + z_max * z_max);
                prop_assert!((m + m_mirror).abs() <= 1e-9 * (1.0 + m.abs() + mu.abs()), "m={m}, mirror={m_mirror}");
                prop_assert!((w - w_mirror).abs() <= tol, "w={w}, mirror={w_mirror}");
                prop_assert!((s_beta[0] + s_beta_mirror[0]).abs() <= tol, "s_beta={}, mirror={}", s_beta[0], s_beta_mirror[0]);
                prop_assert!((s_sigma - s_sigma_mirror).abs() <= tol, "s_sigma={s_sigma}, mirror={s_sigma_mirror}");
            }
        }
    }
}
