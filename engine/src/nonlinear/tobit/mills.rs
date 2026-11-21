//! 正規分布の裾を安定して計算する部品（逆ミルズ比、`Φ(x)-1/2`、生存関数、切断正規の密度比、
//! `ExpectedUncensored`の項）。

use crate::nonlinear::common::clamped_pdf_cdf;
use statrs::distribution::{Continuous, Normal};

/// 逆ミルズ比・生存関数を連分数展開で評価する`a`の下限。連分数は`a>=3`で60段あれば相対誤差
/// `3e-17`以下に収束する。`a`がこれ未満では`normal_survival`（テイラー級数）で`Φ(-a)`を
/// 作って`φ(a)/Φ(-a)`を直接計算する。`a≈38`では`φ`・`Φ(-a)`がアンダーフローするため、
/// 裾は連分数でなければ評価できない。
const MILLS_CONTINUED_FRACTION_THRESHOLD: f64 = 3.0;

/// 連分数展開の段数（`a>=MILLS_CONTINUED_FRACTION_THRESHOLD`で倍精度に十分収束する）。
const MILLS_CONTINUED_FRACTION_DEPTH: u32 = 60;

/// `Φ(x)-1/2`（`x>=0`）を絶対誤差`≈5e-16`以下で計算する。`statrs`の`Normal::cdf`は絶対誤差が
/// 約`1.4e-11`（`Φ(-1)`で相対`8.6e-11`）あり、`W=Φ(zb)-Φ(za)`で割る量では限界効果の
/// 標準誤差に`100`倍程度に増幅されて現れる（`k=20`で相対`1e-8`）ため使わない。
/// `x<MILLS_CONTINUED_FRACTION_THRESHOLD`では`normal_cdf_minus_half_series`、それ以上では
/// `1/2-φ(x)/λ(x)`（`λ`は連分数、`Φ(x)`が1に近いので絶対誤差で足りる）。
fn normal_cdf_minus_half(normal: &Normal, x: f64) -> f64 {
    debug_assert!(x >= 0.0 || x.is_nan(), "requires x >= 0");
    if x >= MILLS_CONTINUED_FRACTION_THRESHOLD {
        let (lambda, _) = inverse_mills_ratio_continued_fraction(x, MILLS_CONTINUED_FRACTION_DEPTH);
        return 0.5 - normal.pdf(x) / lambda;
    }
    normal_cdf_minus_half_series(normal, x)
}

/// `Φ(x)-1/2=φ(x)·Σₙ x^{2n+1}/(2n+1)!!`（`x>=0`、全項が正で桁落ちなし）。`x<3`なら最大33項で
/// 収束する（打ち切り条件は項が和の`1e-17`以下）。上限500は収束しない入力（NaN等）での
/// 無限ループ防止。`x`が大きいほど項数が増えるため`x<MILLS_CONTINUED_FRACTION_THRESHOLD`で使う。
fn normal_cdf_minus_half_series(normal: &Normal, x: f64) -> f64 {
    let x2 = x * x;
    let mut term = x;
    let mut sum = x;
    for n in 1..=500u32 {
        term *= x2 / (2.0 * f64::from(n) + 1.0);
        sum += term;
        if term <= sum * 1e-17 {
            break;
        }
    }
    normal.pdf(x) * sum
}

/// 標準正規分布の生存関数`Φ(-a)`。`a<0`は`1/2+(Φ(-a)-1/2)`、`0<=a<3`は`1/2-(Φ(a)-1/2)`
/// （`Φ(-3)≈1.3e-3`で桁落ちは約3桁、`a→3`で相対誤差が最大`4e-13`）、`a>=3`は
/// `φ(a)/λ(a)`（連分数、相対`5e-16`以下）。`a>=3`は`a≈38`で`φ(a)/λ`が非正規化数になり
/// `a≳38.6`で0になるので、それより遠い裾は`inverse_mills_ratio_and_gap`（`φ`を作らない）を
/// 使う。本番経路（`inverse_mills_ratio_and_gap`）が呼ぶのは`a<3`のみ。
fn normal_survival(normal: &Normal, a: f64) -> f64 {
    if a < 0.0 {
        0.5 + normal_cdf_minus_half(normal, -a)
    } else if a < MILLS_CONTINUED_FRACTION_THRESHOLD {
        0.5 - normal_cdf_minus_half(normal, a)
    } else {
        let (lambda, _) = inverse_mills_ratio_continued_fraction(a, MILLS_CONTINUED_FRACTION_DEPTH);
        normal.pdf(a) / lambda
    }
}

/// 逆ミルズ比`λ(a)=φ(a)/Φ(-a)`（標準正規分布の右裾のハザード率）と、その`a`からの
/// 超過分`λ(a)-a`を返す。`a`が大きいと`λ≈a`で、引き算`λ-a`は桁落ちするため
/// （真値は`≈1/a`）、超過分は引き算せず直接計算して一緒に返す。
///
/// `a`が小さい（負を含む）ときは`normal_survival`の`Φ(-a)`を使い（`1-Φ(a)`は裾で0に潰れる
/// ため使わない）、`a>=MILLS_CONTINUED_FRACTION_THRESHOLD`では連分数展開
/// `λ(a)=a+1/(a+2/(a+3/(a+…)))`に切り替える（`φ`と`Φ(-a)`が`a≈38`でアンダーフローする
/// 領域でも有効）。
fn inverse_mills_ratio_and_gap(normal: &Normal, a: f64) -> (f64, f64) {
    if a < MILLS_CONTINUED_FRACTION_THRESHOLD {
        let lambda = normal.pdf(a) / normal_survival(normal, a);
        (lambda, lambda - a)
    } else {
        inverse_mills_ratio_continued_fraction(a, MILLS_CONTINUED_FRACTION_DEPTH)
    }
}

/// `inverse_mills_ratio_and_gap`の連分数展開（`depth>=2`段で打ち切り、内側から評価する）。
/// `a>0`が前提。`λ(a)-a=1/(a+2/(a+3/(a+…)))`は第2項以降の連分数の逆数として
/// 引き算なしに得られる。
fn inverse_mills_ratio_continued_fraction(a: f64, depth: u32) -> (f64, f64) {
    let mut t = a;
    for k in (2..=depth).rev() {
        t = a + f64::from(k) / t;
    }
    let gap = 1.0 / t;
    (a + gap, gap)
}

/// `0<=za<zb`の上側裾での`(Pa, Pb, A, B)`。`Pa=φ(za)/W`・`Pb=φ(zb)/W`
/// （`W=Φ(zb)-Φ(za)`）、`A=Pa-za`・`B=zb+Pb`（`ExpectedUncensored`の式で`za`・`zb`との
/// 大きな数同士の差を避けるための量。`truncation_pdf_ratios`参照）。
///
/// `W`を直接作ると両方が裾にあるとき桁落ち・アンダーフローするため、`W=Φ(-za)-Φ(-zb)`を
/// `Φ(-za)`で割った比で表す。`p=λ(za)`・`q=λ(zb)`・`r=φ(zb)/φ(za)=exp(-(zb²-za²)/2)`と
/// すると`φ(zb)/Φ(-za)=r·p`・`Φ(-zb)/Φ(-za)=r·p/q`なので、`W/Φ(-za)=1-r·p/q`、
/// `Pa=p·q/(q-r·p)`、`Pb=r·p·q/(q-r·p)`となる（`q>p>=r·p`で分母は正）。
/// `A=(p-za)+(Pa-p)=gap_p+r·p²/(q-r·p)`は正の項の和で、`B=zb+Pb`も正の項の和。
fn upper_tail_pdf_ratios(normal: &Normal, za: f64, zb: f64) -> (f64, f64, f64, f64) {
    debug_assert!(0.0 <= za && za < zb, "upper tail requires 0 <= za < zb");
    let (p, gap_p) = inverse_mills_ratio_and_gap(normal, za);
    let (q, _) = inverse_mills_ratio_and_gap(normal, zb);
    let r = (-0.5 * (zb - za) * (zb + za)).exp();
    let denom = q - r * p;
    let pa = p * q / denom;
    let pb = r * p * q / denom;
    (pa, pb, gap_p + r * p * p / denom, zb + pb)
}

/// `truncation_pdf_ratios`の返り値。`pa=φ(za)/W`・`pb=φ(zb)/W`（`W=Φ(zb)-Φ(za)`、
/// 境界が無い方向は0）、`a_gap=pa-za`・`b_gap=zb+pb`。
pub(super) struct TruncationRatios {
    pub(super) pa: f64,
    pub(super) pb: f64,
    pub(super) a_gap: f64,
    pub(super) b_gap: f64,
}

/// 非打ち切り領域`(za,zb)`（標準化した下限・上限。`None`はその方向の打ち切りなし）に
/// 切断した標準正規分布について、`W=Φ(zb)-Φ(za)`で割る量を`W→0`となる裾でも
/// 破綻しない形で返す（`TruncationRatios`）。`E[y|uncensored]`の`μ`からのずれ
/// `σ(Pa-Pb)`、切断分散比`w`など、`W`で割る量は全てこれで書ける。`A=Pa-za`・
/// `B=zb+Pb`は`|z|`が大きいと`Pa≈za`のように大きな数同士の差になるため、引き算せず
/// 直接計算して返す（`uncensored_w_and_s`が`z²`同士の桁落ちを避けるのに使う）。
///
/// - 片側のみ: 逆ミルズ比（左のみ`Pa=λ(za)`、`A=λ(za)-za`。右のみ`Pb=λ(-zb)`、
///   `B=λ(-zb)-(-zb)`）
/// - 両側で`za>=0`（上側の裾）: `upper_tail_pdf_ratios`
/// - 両側で`zb<=0`（下側の裾）: 符号反転`(za,zb)→(-zb,-za)`で上側の裾へ写し、
///   `(Pa,Pb)`と`(A,B)`をそれぞれ入れ替える
/// - 両側で`za<0<zb`: `W`が裾に潰れないので直接計算する（`A`・`B`は正の項の和）
fn truncation_pdf_ratios(normal: &Normal, za: Option<f64>, zb: Option<f64>) -> TruncationRatios {
    let (pa, pb, a_gap, b_gap) = match (za, zb) {
        (None, None) => (0.0, 0.0, 0.0, 0.0),
        (Some(a), None) => {
            let (lambda, gap) = inverse_mills_ratio_and_gap(normal, a);
            (lambda, 0.0, gap, 0.0)
        }
        (None, Some(b)) => {
            let (lambda, gap) = inverse_mills_ratio_and_gap(normal, -b);
            (0.0, lambda, 0.0, gap)
        }
        (Some(a), Some(b)) => {
            if a >= 0.0 {
                upper_tail_pdf_ratios(normal, a, b)
            } else if b <= 0.0 {
                let (pa_m, pb_m, a_gap_m, b_gap_m) = upper_tail_pdf_ratios(normal, -b, -a);
                (pb_m, pa_m, b_gap_m, a_gap_m)
            } else {
                // `Φ(b)-Φ(a)=(Φ(b)-1/2)+(1/2-Φ(a))`は正の項の和（`a<0<b`）
                let w = normal_cdf_minus_half(normal, b) + normal_cdf_minus_half(normal, -a);
                let pa = normal.pdf(a) / w;
                let pb = normal.pdf(b) / w;
                (pa, pb, pa - a, b + pb)
            }
        }
    };
    TruncationRatios {
        pa,
        pb,
        a_gap,
        b_gap,
    }
}

/// `ExpectedUncensored`の評価に使う、線形予測子`mu`での標準化境界と`TruncationRatios`。
/// `za`/`zb`は境界が無い方向を`0.0`のダミー値にする（対応する`pa`/`pb`が0なので
/// 以降の式に影響しない）。
pub(super) struct UncensoredTerms {
    pub(super) za: f64,
    pub(super) zb: f64,
    pub(super) ratios: TruncationRatios,
}

impl UncensoredTerms {
    pub(super) fn new(
        mu: f64,
        sigma: f64,
        lower: Option<f64>,
        upper: Option<f64>,
        normal: &Normal,
    ) -> Self {
        let za = lower.map(|l| (l - mu) / sigma);
        let zb = upper.map(|u| (u - mu) / sigma);
        Self {
            za: za.unwrap_or(0.0),
            zb: zb.unwrap_or(0.0),
            ratios: truncation_pdf_ratios(normal, za, zb),
        }
    }

    /// `E[y|uncensored]=μ+σ·(Pa-Pb)`
    pub(super) fn mean(&self, mu: f64, sigma: f64) -> f64 {
        mu + sigma * (self.ratios.pa - self.ratios.pb)
    }
}

/// 打ち切り境界`bound`を標準化した`z=(bound-μ)/σ`・`φ(z)`・`Φ(z)`を計算する。`None`は
/// 「その方向は打ち切りなし」を意味し、`Φ(∓∞)=0/1`・`φ(∓∞)=0`に相当する値を返す
/// （`z`自体はダミー値`0.0`を返すが、後続の計算では常に`φ=0.0`とセットでしか使わないため
/// 結果に影響しない。`is_lower`で下限側（`Φ(-∞)=0`）か上限側（`Φ(+∞)=1`）かを切り替える）。
pub(super) fn boundary_terms(
    bound: Option<f64>,
    mu: f64,
    sigma: f64,
    normal: &Normal,
    is_lower: bool,
) -> (f64, f64, f64) {
    match bound {
        Some(c) => {
            let z = (c - mu) / sigma;
            let (phi, big_phi) = clamped_pdf_cdf(normal, z);
            (z, phi, big_phi)
        }
        None => (0.0, 0.0, if is_lower { 0.0 } else { 1.0 }),
    }
}

#[cfg(test)]
mod tests {
    use super::super::target::uncensored_w_and_s;
    use super::super::test_support::*;
    use super::*;

    /// 逆ミルズ比`λ(a)=φ(a)/Φ(-a)`とその超過分`λ(a)-a`の参照値。`a<=1`は
    /// `scipy.special.erfcx`（`λ=√(2/π)/erfcx(a/√2)`）、`a>=3`は連分数を60桁の10進演算で
    /// 20000段評価した値（超過分`λ-a`も同じ演算で直接作った値）。`a∈[2.5,3)`の参照は
    /// 70桁の10進演算のテイラー級数（`scipy`の`erfcx`と一致を確認済み）。直接計算
    /// （`a<MILLS_CONTINUED_FRACTION_THRESHOLD=3`、テイラー級数の`Φ`）と連分数の両方の領域、
    /// および切り替え点の前後（`2.9999`・`3.0`）を含む。
    #[test]
    fn inverse_mills_ratio_and_gap_match_high_precision_reference_on_both_branches() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        // (a, lambda, gap)。`a<3`のgapは`λ-a`の引き算で、`λ/gap≈12`倍に誤差が増幅される
        let cases = [
            (-5.0, 1.486_719_940_904_908_6e-6, None),
            (0.0, 0.797_884_560_802_865_4, None),
            (1.0, 1.525_135_276_160_981_3, None),
            (2.5, 2.822_744_797_663_907, Some(0.322_744_797_663_907_2)),
            (2.9, 3.190_315_139_542_982, Some(0.290_315_139_542_982)),
            (2.99, 3.273_805_823_492_689, Some(0.283_805_823_492_688_8)),
            (2.9999, 3.283_005_711_006_472, Some(0.283_105_711_006_471_4)),
            (3.0, 3.283_098_654_930_436, Some(0.283_098_654_930_436_5)),
            (5.0, 5.186_503_967_125_842, Some(0.186_503_967_125_842_1)),
            (10.0, 10.098_093_233_962_512, Some(0.098_093_233_962_511_96)),
            (29.9, 29.933_370_411_296_167, None),
            (38.0, 38.026_279_466_575_87, Some(0.026_279_466_575_868_99)),
            (
                100.0,
                100.009_998_000_999_26,
                Some(0.009_998_000_999_260_705),
            ),
            (1000.0, 1_000.000_999_998, Some(0.000_999_998_000_01)),
        ];
        for (a, lambda, gap) in cases {
            let (actual, actual_gap) = inverse_mills_ratio_and_gap(&normal, a);
            // 級数側（`a<3`）は`a→3`で`Φ(-a)=1/2-(Φ(a)-1/2)`の桁落ちが増え、`λ`は最大`4e-13`、
            // `gap`は最大`4e-12`（実測）。連分数側（`a>=3`）は`5e-16`以下
            let (lambda_rtol, gap_rtol) = if a < MILLS_CONTINUED_FRACTION_THRESHOLD {
                (1e-12, 1e-11)
            } else {
                (1e-14, 1e-13)
            };
            assert_relative_eq(actual, lambda, lambda_rtol, &format!("lambda({a})"));
            if let Some(gap) = gap {
                assert_relative_eq(actual_gap, gap, gap_rtol, &format!("gap({a})"));
            }
        }
    }

    /// 切り替え点`MILLS_CONTINUED_FRACTION_THRESHOLD`の前後で`λ`・`Φ(x)-1/2`・`Φ(-a)`が
    /// 連続（飛びがない）。直接計算側（テイラー級数）と連分数側が同じ値を返すことの確認。
    #[test]
    fn mills_ratio_and_normal_cdf_are_continuous_across_the_series_to_continued_fraction_switch() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let t = MILLS_CONTINUED_FRACTION_THRESHOLD;
        // 切り替え点の直前の浮動小数点数（傾きによる変化を除き、飛びだけを見る）
        let below = f64::from_bits(t.to_bits() - 1);
        let (lambda_below, _) = inverse_mills_ratio_and_gap(&normal, below);
        let (lambda_at, _) = inverse_mills_ratio_and_gap(&normal, t);
        // 飛びの実測は`λ`で`1.3e-13`、`Φ(-a)`で`3.6e-13`（級数側の桁落ち）。許容は`1e-12`
        assert!(
            (lambda_at - lambda_below).abs() < 1e-12,
            "lambda: below={lambda_below}, at={lambda_at}"
        );
        assert_relative_eq(
            normal_survival(&normal, below),
            normal_survival(&normal, t),
            1e-12,
            "survival around switch",
        );
        assert_relative_eq(
            normal_cdf_minus_half(&normal, below),
            normal_cdf_minus_half(&normal, t),
            1e-12,
            "cdf-1/2 around switch",
        );
    }

    /// `statrs`の`Normal::cdf`（絶対誤差約`1.4e-11`）に代わる`normal_cdf_minus_half`・
    /// `normal_survival`の精度。参照は`scipy.special`（`0.5·erf(x/√2)`・`ndtr(-a)`）。
    /// `Φ(-8)≈6.2e-16`のような裾でも相対`1e-13`、`1/2`近傍も絶対`1e-16`級で一致する。
    #[test]
    fn normal_cdf_minus_half_and_survival_match_scipy_to_machine_precision() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        for (x, expected) in [
            (0.5, 0.191_462_461_274_013_06),
            (1.0, 0.341_344_746_068_542_9),
            (2.0, 0.477_249_868_051_820_8),
            (2.99, 0.498_605_112_764_507_75),
            (3.0, 0.498_650_101_968_369_9),
            (5.0, 0.499_999_713_348_428_13),
        ] {
            assert_relative_eq(
                normal_cdf_minus_half(&normal, x),
                expected,
                1e-14,
                &format!("Phi({x})-1/2"),
            );
        }
        for (a, expected) in [
            (-2.0, 0.977_249_868_051_820_8),
            (0.0, 0.5),
            (1.0, 0.158_655_253_931_457_07),
            (2.0, 0.022_750_131_948_179_195),
            (2.99, 0.001_394_887_235_492_246_8),
            (3.0, 0.001_349_898_031_630_093_3),
            (4.0, 3.167_124_183_311_986e-5),
            (8.0, 6.220_960_574_271_74e-16),
        ] {
            assert_relative_eq(
                normal_survival(&normal, a),
                expected,
                1e-13,
                &format!("Phi(-{a})"),
            );
        }
    }

    /// 連分数展開は、切り替え点より上（`a=3`・`3.5`・`4`）でも、テイラー級数
    /// （`normal_cdf_minus_half_series`、切り替え点より上へ外挿して`Φ(-a)=1/2-(Φ(a)-1/2)`を作る）
    /// 由来の`φ(a)/Φ(-a)`と一致する。段数`MILLS_CONTINUED_FRACTION_DEPTH`が切り替え点でも
    /// 収束していることの確認で、段数を落とす（20等）と失敗する。級数側は`a`が大きいほど
    /// `Φ(-a)=1/2-(Φ(a)-1/2)`の桁落ち（相対`ε/Φ(-a)`）で精度が落ちるため、許容は`a`に応じて
    /// `Φ(-3)=1.3e-3`で`5e-13`、`Φ(-3.5)=2.3e-4`で`2e-12`、`Φ(-4)=3.2e-5`で`1e-11`（実測`4e-12`）。
    #[test]
    fn inverse_mills_ratio_continued_fraction_agrees_with_series_based_survival_near_the_switch() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        for (a, rtol) in [(3.0, 5e-13), (3.5, 2e-12), (4.0, 1e-11)] {
            let series_survival = 0.5 - normal_cdf_minus_half_series(&normal, a);
            let series_lambda = normal.pdf(a) / series_survival;
            let (cf, _) = inverse_mills_ratio_continued_fraction(a, MILLS_CONTINUED_FRACTION_DEPTH);
            assert_relative_eq(cf, series_lambda, rtol, &format!("cf vs series at a={a}"));
        }
    }

    /// 補助関数は`NaN`・`±inf`・`±40`でもpanicや無限ループなしに有限時間で戻り、値は
    /// `Φ(-a)`の定義（`a=-inf`で1、`a=+inf`で0、`NaN`は`NaN`）に従う。
    #[test]
    fn normal_cdf_helpers_handle_non_finite_and_far_tail_inputs_without_panicking() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        assert!(normal_survival(&normal, f64::NAN).is_nan());
        assert!(normal_cdf_minus_half(&normal, f64::NAN).is_nan());
        assert_eq!(normal_survival(&normal, f64::INFINITY), 0.0);
        assert_eq!(normal_survival(&normal, f64::NEG_INFINITY), 1.0);
        assert_eq!(normal_survival(&normal, -40.0), 1.0);
        assert!(normal_survival(&normal, 40.0) >= 0.0);
        assert_eq!(normal_cdf_minus_half(&normal, 40.0), 0.5);
        // 遠い裾の逆ミルズ比は有限（`φ`を作らない連分数）
        for a in [-1e8, -40.0, 40.0, 1e8] {
            let (lambda, gap) = inverse_mills_ratio_and_gap(&normal, a);
            assert!(lambda.is_finite() && gap.is_finite(), "a={a}");
        }
    }

    /// `(Pa,Pb)`と`ExpectedUncensored`の`(w, ∂w/∂μ, ∂w/∂σ)`を、70〜80桁の10進演算で
    /// 直接評価した参照値（定義通りに`φ/W`・`w=1+ρ2-ρ1²`・偏微分を計算。裾は`W=Φ(-za)-Φ(-zb)`を
    /// 連分数の`Φ(-z)`から、中央・級数側は`Φ(x)-1/2`のテイラー級数から作る）と突き合わせる。
    /// 左のみ・右のみ・両側の上側の裾・下側の裾（符号反転）・`zb`が遥か先で片側と一致する場合に
    /// 加え、切り替え点の前後の両側`(2.9,3.1)`・`(-3.2,-2.9)`・`(2.99,3.5)`、中央`(-1,1.5)`、
    /// 級数側の上側`(0.5,1)`を含む。`z=40`では`Φ(-40)≈3.7e-350`が倍精度でアンダーフローし、
    /// 直接計算では`W=0`になる領域。許容は実測（`w`の絶対誤差`2e-16`、`Pa`/`Pb`の相対`1e-16`、
    /// `s_beta`は`z=40`で`3e-15`、`s_sigma`は`z=40`で`1e-13`・`z=1000`で`9e-12`）の約10倍。
    /// 整理前の式（`ε·z³`の桁落ち）は`z=40`以上で許容を超える。
    #[test]
    fn uncensored_terms_and_w_match_high_precision_reference_in_extreme_tails() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        // (label, za, zb, Pa, Pb, w, w_a, w_b)
        #[allow(clippy::type_complexity)]
        let cases: [(&str, Option<f64>, Option<f64>, f64, f64, f64, f64, f64); 15] = [
            (
                "left-only z=40",
                Some(40.0),
                None,
                40.024_968_847_207_26,
                0.0,
                6.226_683_785_913_888e-4,
                -3.101_744_039_648_625e-5,
                0.0,
            ),
            (
                "left-only z=100",
                Some(100.0),
                None,
                100.009_998_000_999_26,
                0.0,
                9.994_004_994_826_346e-5,
                -1.997_602_995_862_343e-6,
                0.0,
            ),
            (
                "left-only z=1000",
                Some(1000.0),
                None,
                1_000.000_999_998,
                0.0,
                9.999_940_000_499_995e-7,
                -1.999_976_000_299_996e-9,
                0.0,
            ),
            (
                "right-only z=-40",
                None,
                Some(-40.0),
                0.0,
                40.024_968_847_207_26,
                6.226_683_785_913_888e-4,
                0.0,
                3.101_744_039_648_625e-5,
            ),
            (
                "two-sided upper (40,45)",
                Some(40.0),
                Some(45.0),
                40.024_968_847_207_26,
                2.064_208_671_428_423_9e-91,
                6.226_683_785_913_888e-4,
                -3.101_744_039_648_625e-5,
                5.108_980_927_616_573e-90,
            ),
            (
                "two-sided upper (30,31)",
                Some(30.0),
                Some(31.0),
                30.033_259_667_435_328,
                1.704_593_283_822_866_3e-12,
                1.103_771_511_835_282_2e-3,
                -7.309_993_169_431_127e-5,
                1.591_209_021_285_825_6e-12,
            ),
            (
                "two-sided upper (40,1000)",
                Some(40.0),
                Some(1000.0),
                40.024_968_847_207_26,
                0.0,
                6.226_683_785_913_888e-4,
                -3.101_744_039_648_625e-5,
                0.0,
            ),
            (
                "two-sided lower (-45,-40)",
                Some(-45.0),
                Some(-40.0),
                2.064_208_671_428_423_9e-91,
                40.024_968_847_207_26,
                6.226_683_785_913_888e-4,
                -5.108_980_927_616_573e-90,
                3.101_744_039_648_625e-5,
            ),
            (
                "two-sided lower (-31,-30)",
                Some(-31.0),
                Some(-30.0),
                1.704_593_283_822_866_3e-12,
                30.033_259_667_435_328,
                1.103_771_511_835_282_2e-3,
                -1.591_209_021_285_825_6e-12,
                7.309_993_169_431_127e-5,
            ),
            (
                "two-sided upper (12,12.5)",
                Some(12.0),
                Some(12.5),
                12.107_653_550_680_133,
                0.026_485_384_604_148_263,
                6.151_043_601_970_409e-3,
                -5.293_800_087_694_461_5e-3,
                4.483_156_195_574_498e-3,
            ),
            (
                "two-sided middle (-1,1.5)",
                Some(-1.0),
                Some(1.5),
                0.312_406_707_904_292_9,
                0.167_219_260_751_766_74,
                0.415_685_006_157_389_35,
                -0.279_844_332_730_801_1,
                0.237_423_265_323_732_5,
            ),
            (
                "two-sided series side (0.5,1)",
                Some(0.5),
                Some(1.0),
                2.348_945_555_820_274,
                1.614_405_096_978_975_4,
                0.020_517_995_256_376_27,
                -0.081_018_025_127_821_73,
                0.080_640_782_102_733_44,
            ),
            (
                "two-sided across switch (2.9,3.1)",
                Some(2.9),
                Some(3.1),
                6.627_104_844_212_096,
                3.637_032_252_118_688_6,
                0.003_269_960_658_955_57,
                -0.032_095_805_611_538_79,
                0.032_057_072_683_910_62,
            ),
            (
                "two-sided lower across switch (-3.2,-2.9)",
                Some(-3.2),
                Some(-2.9),
                2.022_684_343_271_988,
                5.050_188_210_696_992,
                0.007_176_245_277_561_196,
                -0.045_669_523_263_236_76,
                0.045_860_713_340_026_13,
            ),
            (
                "two-sided just above switch (2.99,3.5)",
                Some(2.99),
                Some(3.5),
                3.929_066_816_439_395_4,
                0.750_850_996_568_809_9,
                0.018_875_495_487_299_693,
                -0.065_024_874_586_292_15,
                0.063_574_225_828_370_7,
            ),
        ];
        // `mu=1, sigma=2`で`za=(lower-mu)/sigma`が上の値になるよう境界を逆算する
        let (mu, sigma) = (1.0, 2.0);
        for (label, za, zb, pa, pb, w, w_a, w_b) in cases {
            let lower = za.map(|z| mu + sigma * z);
            let upper = zb.map(|z| mu + sigma * z);
            let terms = UncensoredTerms::new(mu, sigma, lower, upper, &normal);
            // 切り替え点の手前（`a<3`）を通る級数側は`λ`の相対誤差が最大`4e-13`
            let near_switch = label.contains("switch");
            let ratio_rtol = if near_switch { 1e-12 } else { 1e-14 };
            assert_relative_eq(terms.ratios.pa, pa, ratio_rtol, &format!("{label}: Pa"));
            assert_relative_eq(terms.ratios.pb, pb, ratio_rtol, &format!("{label}: Pb"));

            let x_point = [1.0, 2.0];
            let (w_actual, s_beta, s_sigma) =
                uncensored_w_and_s(&x_point, mu, sigma, lower, upper, &normal);
            let z_scale = za.unwrap_or(0.0).abs().max(zb.unwrap_or(0.0).abs());
            // `w`は`1-Pa·A-Pb·B+2·Pa·Pb`（`O(1)`同士の和）なので絶対誤差は数`ε`。切り替え点付近の
            // 級数側は`Pa`・`Pb`・`A`・`B`の誤差（`λ`で最大`4e-13`、`gap`で最大`4e-12`）が`O(10)`倍で効く
            assert!(
                (w_actual - w).abs() <= if near_switch { 2e-12 } else { 2e-15 },
                "{label}: w actual={w_actual:e}, expected={w:e}"
            );
            let dw_dmu = -(w_a + w_b) / sigma;
            let dw_dsigma = -(za.unwrap_or(0.0) * w_a + zb.unwrap_or(0.0) * w_b) / sigma;
            assert!(
                (s_beta[1] - dw_dmu * x_point[1]).abs()
                    <= if near_switch {
                        5e-12
                    } else {
                        1e-15 * z_scale + 1e-15
                    },
                "{label}: s_beta actual={:e}, expected={:e}",
                s_beta[1],
                dw_dmu * x_point[1]
            );
            assert!(
                (s_sigma - dw_dsigma).abs()
                    <= if near_switch {
                        2e-11
                    } else {
                        1e-15 * z_scale * z_scale + 1e-15
                    },
                "{label}: s_sigma actual={s_sigma:e}, expected={dw_dsigma:e}"
            );
        }
    }
}
