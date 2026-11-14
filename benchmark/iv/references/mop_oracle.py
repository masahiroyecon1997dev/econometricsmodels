"""Montiel Olea and Pflueger (2013)の有効F統計量と臨界値の独立実装（オラクル）。

論文 "A Robust Test for Weak Instruments", Journal of Business & Economic
Statistics 31(3), 358-369 の式だけから書き起こした参照実装で、依存は
numpy と scipy のみ。engine の Rust 実装を照合する参照値を作るために使う。
内生変数は1個（論文のモデル Eq. (1), (2)）。

式の根拠は論文の式番号で示す:

- 前処理（外生変数への直交化、操作変数の直交正規化）: 2.2.1節 Step 1
- 有効F統計量: Eq. (3), Lemma 1.5 の ``F_eff``
- 有効自由度: Eq. (4), (33)
- Patnaik臨界値: Eq. (32)、``x`` は単純版が 2.2.2節、一般化版が Eq. (5), (27)
- ``B_TSLS``: Theorem 1 の Eq. (19), (21)、数値実装は Remark 5

LIML は対象外（``B_LIML`` は実装しない）。

``B_TSLS`` の上限（Eq. (21)）は ``C0`` を解析的に、``beta`` を数値的に最大化する。
``beta`` の最大化は次の2つの独立した方法で行い、結果を突き合わせられる:

- ``method="beta"``: ``beta`` の軸で粗い格子＋局所探索。``|beta| -> inf`` は
  極限の式（``_b_tsls_limit``）で別に扱う。
- ``method="angle"``: ``beta = cos(phi)/sin(phi)`` と置いた角度 ``phi`` の周期関数
  として最大化する。``beta = inf`` は ``phi = 0`` の通常の評価点になり、極限の式を
  使わない。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from typing import Literal

import numpy as np
from scipy import optimize, stats

CovKind = Literal["classical", "hc0", "hc1", "hc2", "hc3", "cluster", "hac"]
BMethod = Literal["beta", "angle"]

# 論文の閾値 tau。x = 1/tau を厳密な有理数から作る（tau = 0.30 は 10/3）。
TAUS: tuple[Fraction, ...] = (
    Fraction(5, 100),
    Fraction(10, 100),
    Fraction(20, 100),
    Fraction(30, 100),
)


@dataclass(frozen=True)
class ReducedForm:
    """直交化・直交正規化後の誘導形と、その漸近共分散 ``W_hat``。

    Attributes:
        nobs: 観測数 ``S``。
        pi_y: ``y`` の誘導形係数 ``Z'y/S``（長さ ``K``）。
        pi: ``Y`` の誘導形係数 ``Z'Y/S``（長さ ``K``）。
        w: ``sqrt(S) * (pi_y; pi)`` の漸近共分散の推定値（``2K x 2K``）。
    """

    nobs: int
    pi_y: np.ndarray
    pi: np.ndarray
    w: np.ndarray

    @property
    def n_instruments(self) -> int:
        return self.pi.shape[0]

    @property
    def w1(self) -> np.ndarray:
        k = self.n_instruments
        return self.w[:k, :k]

    @property
    def w12(self) -> np.ndarray:
        k = self.n_instruments
        return self.w[:k, k:]

    @property
    def w2(self) -> np.ndarray:
        k = self.n_instruments
        return self.w[k:, k:]


@dataclass(frozen=True)
class CriticalValue:
    """Patnaik臨界値 Eq. (32) と、その入力。"""

    x: float
    k_eff: float
    value: float


@dataclass(frozen=True)
class MopResult:
    """1ケース分の有効F統計量と臨界値。"""

    f_eff: float
    trace_v2: float
    w2_eigenvalues: np.ndarray
    b_tsls: float
    simple: dict[Fraction, CriticalValue]
    generalized: dict[Fraction, CriticalValue]


def _thin_qr(a: np.ndarray) -> np.ndarray:
    """列が一次独立な行列の直交基底 ``Q``（``a`` の列空間）を返す。"""
    q, r = np.linalg.qr(a, mode="reduced")
    diag = np.abs(np.diag(r))
    if diag.min() <= 1e-12 * diag.max():
        raise ValueError("columns are (nearly) linearly dependent")
    return q


def reduced_form(
    y: np.ndarray,
    endog: np.ndarray,
    x_exog: np.ndarray,
    z: np.ndarray,
    cov: CovKind,
    *,
    n_absorbed: int = 0,
    groups: np.ndarray | None = None,
    lags: int | None = None,
) -> ReducedForm:
    """前処理（2.2.1節 Step 1）と誘導形の漸近共分散 ``W_hat`` の推定。

    Args:
        y: 被説明変数（長さ ``S``）。
        endog: 内生変数 ``Y``（長さ ``S``）。
        x_exog: 外生変数（定数列を含む。平均は引かず回帰で落とす）。
        z: 除外操作変数（``S x K``）。
        cov: 共分散の種類。
        n_absorbed: 吸収したパラメータ数。小標本係数 ``c`` にだけ入る。
        groups: ``cov="cluster"`` のグループ。
        lags: ``cov="hac"`` のラグ数 ``L``（Bartlett核）。

    Returns:
        直交正規化後の ``pi_hat`` と ``W_hat``。
    """
    y = np.asarray(y, dtype=float)
    endog = np.asarray(endog, dtype=float)
    x_exog = np.asarray(x_exog, dtype=float)
    z = np.asarray(z, dtype=float)
    s, k = z.shape
    l_total = k + x_exog.shape[1]

    if cov in ("hc2", "hc3") and n_absorbed != 0:
        raise ValueError(f"{cov} is undefined when n_absorbed > 0")
    if s - l_total - n_absorbed <= 0:
        raise ValueError(
            f"S - l - n_absorbed must be positive, got S={s}, l={l_total}, "
            f"n_absorbed={n_absorbed}"
        )
    if cov == "hac" and lags is not None and lags >= s:
        raise ValueError(f"lags must be less than S, got lags={lags}, S={s}")

    # Step 1: 外生変数への直交化と、Z'Z/S = I_K となる直交正規化。
    qx = _thin_qr(x_exog)
    y_t = y - qx @ (qx.T @ y)
    endog_t = endog - qx @ (qx.T @ endog)
    z_r = z - qx @ (qx.T @ z)
    z_t = np.sqrt(s) * _thin_qr(z_r)

    pi_y = z_t.T @ y_t / s
    pi = z_t.T @ endog_t / s
    e1 = y_t - z_t @ pi_y
    e2 = endog_t - z_t @ pi

    dof_c = s / (s - l_total - n_absorbed)

    if cov == "classical":
        e = np.column_stack([e1, e2])
        omega = e.T @ e / s
        w = dof_c * np.kron(omega, np.eye(k))
        return ReducedForm(s, pi_y, pi, w)

    scores = np.hstack([z_t * e1[:, None], z_t * e2[:, None]])

    if cov in ("hc0", "hc1", "hc2", "hc3"):
        if cov in ("hc2", "hc3"):
            q_full = _thin_qr(np.hstack([x_exog, z]))
            h = np.sum(q_full**2, axis=1)
            power = 1 if cov == "hc2" else 2
            scale = 1.0 / (1.0 - h) ** power
        else:
            scale = np.ones(s)
        meat = (scores * scale[:, None]).T @ scores
        c = dof_c if cov == "hc1" else 1.0
    elif cov == "cluster":
        if groups is None:
            raise ValueError("groups is required for cov='cluster'")
        _, inverse = np.unique(np.asarray(groups), return_inverse=True)
        n_groups = int(inverse.max()) + 1
        sums = np.zeros((n_groups, scores.shape[1]))
        np.add.at(sums, inverse, scores)
        meat = sums.T @ sums
        c = n_groups / (n_groups - 1) * (s - 1) / (s - l_total - n_absorbed)
    elif cov == "hac":
        if lags is None:
            raise ValueError("lags is required for cov='hac'")
        meat = scores.T @ scores
        for j in range(1, lags + 1):
            gamma_j = scores[j:].T @ scores[:-j]
            meat = meat + (1.0 - j / (lags + 1)) * (gamma_j + gamma_j.T)
        c = dof_c
    else:
        raise ValueError(f"unknown cov: {cov}")

    return ReducedForm(s, pi_y, pi, c * meat / s)


def effective_f(pi: np.ndarray, w2: np.ndarray, nobs: int) -> float:
    """有効F統計量 Eq. (3): ``S * pi'pi / tr(W2)``（``Z'Z/S = I`` の下）。"""
    return float(nobs * (pi @ pi) / np.trace(w2))


def effective_dof(w2: np.ndarray, x: float) -> float:
    """有効自由度 ``K_eff``（Eq. (4), (33)）。小数でよい。"""
    tr = np.trace(w2)
    lam_max = np.linalg.eigvalsh((w2 + w2.T) / 2).max()
    denom = np.trace(w2.T @ w2) + 2.0 * x * tr * lam_max
    return float(tr**2 * (1.0 + 2.0 * x) / denom)


def patnaik_critical_value(
    w2: np.ndarray, x: float, alpha: float = 0.05
) -> CriticalValue:
    """Patnaik臨界値 Eq. (32): ``ncx2`` の上側 ``alpha`` 分位点を ``K_eff`` で割る。"""
    k_eff = effective_dof(w2, x)
    quantile = stats.ncx2.ppf(1.0 - alpha, k_eff, x * k_eff)
    return CriticalValue(x=x, k_eff=k_eff, value=float(quantile / k_eff))


def _exact_sum(values: np.ndarray) -> Fraction:
    """浮動小数の和を、丸めずに有理数で返す。"""
    return sum((Fraction(float(v)) for v in values), Fraction(0))


class _BProblem:
    """``B_TSLS`` の最大化に使う ``W_hat`` の要約量と、比 Eq. (21) の評価。

    ``beta0 = tr(W12)/tr(W2)`` を中心に ``tr(S1)`` を展開して評価する。
    ``tr(S1)`` の最小値は ``tr(W1) - tr(W12)**2/tr(W2)`` で、誤差がほぼ比例する
    ときは ``tr(W1)`` に比べて極端に小さく、展開式 ``tr(W1) - 2*beta*tr(W12) +
    beta**2*tr(W2)`` や上の式を浮動小数のまま計算すると桁落ちする。係数
    ``q0 = tr(S1)(beta0)``、``e = tr(W12) - beta0*tr(W2)``、``S12(beta0)`` は入力の
    浮動小数を厳密な有理数として計算してから1回だけ丸める。恒等変形であり、
    式そのものは変わらない。
    """

    _CHUNK = 2048

    def __init__(self, rf: ReducedForm) -> None:
        t1 = _exact_sum(np.diag(rf.w1))
        t12 = _exact_sum(np.diag(rf.w12))
        t2 = _exact_sum(np.diag(rf.w2))
        b0 = Fraction(float(t12 / t2))
        self.tr_w2 = float(t2)
        self.beta0 = float(b0)
        self.q0 = float(t1 - 2 * b0 * t12 + b0 * b0 * t2)
        self.e = float(t12 - b0 * t2)
        k = rf.n_instruments
        sym_a = [
            [
                Fraction(float(rf.w12[i, j])) + Fraction(float(rf.w12[j, i]))
                for j in range(k)
            ]
            for i in range(k)
        ]
        sym_b = [
            [
                Fraction(float(rf.w2[i, j])) + Fraction(float(rf.w2[j, i]))
                for j in range(k)
            ]
            for i in range(k)
        ]
        # 2 * ((W12 + W12')/2 - beta0 * (W2 + W2')/2) と、2 * (W2 + W2')/2 の半分
        self.shift = np.array(
            [
                [float((sym_a[i][j] - b0 * sym_b[i][j]) / 2) for j in range(k)]
                for i in range(k)
            ]
        )
        self.w2 = np.array(
            [[float(sym_b[i][j] / 2) for j in range(k)] for i in range(k)]
        )

    def ratios(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """同次化した ``beta = u / v`` での Eq. (21) の比（``u``、``v`` は配列）。

        ``n_TSLS`` は ``C0' S12 C0`` について線形（Eq. (19)）なので、``C0`` を
        単位球面で動かしたときの ``|n_TSLS|`` の上限は、``(S12 + S12')/2`` の最小・
        最大固有値 ``lam`` に対する ``|tr(S12) - 2*lam|`` の大きい方（Remark 5）。
        分母は ``BM(beta) * tr(S2) = sqrt(tr(S1) * tr(S2))``（Theorem 1）。
        ``S12`` に ``v``、``S1`` に ``v**2`` を掛けると、分子・分母とも ``|v|`` 倍に
        なって比は変わらない（``beta = inf`` が ``v = 0`` で扱える）。
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        out = np.empty(u.shape[0])
        for lo in range(0, u.shape[0], self._CHUNK):
            sl = slice(lo, lo + self._CHUNK)
            uu, vv = u[sl], v[sl]
            w = uu - vv * self.beta0
            trace_s12 = vv * self.e - w * self.tr_w2
            sym = vv[:, None, None] * self.shift - w[:, None, None] * self.w2
            lam = np.linalg.eigvalsh(sym)
            numerator = np.maximum(
                np.abs(trace_s12 - 2.0 * lam[:, 0]),
                np.abs(trace_s12 - 2.0 * lam[:, -1]),
            )
            trace_s1 = (
                self.q0 * vv * vv - 2.0 * self.e * vv * w + self.tr_w2 * w * w
            )
            out[sl] = numerator / np.sqrt(trace_s1 * self.tr_w2)
        return out

    def ratio_beta(self, beta: float) -> float:
        return float(self.ratios(np.array([beta]), np.ones(1))[0])

    def ratio_angle(self, phi: float) -> float:
        return float(
            self.ratios(np.array([np.cos(phi)]), np.array([np.sin(phi)]))[0]
        )


def _b_tsls_limit(problem: _BProblem) -> float:
    """``|beta| -> inf`` での Eq. (21) の比の極限。

    ``S12 ~ -beta*W2``、``tr(S1) ~ beta**2 * tr(W2)`` から、``W2`` の最小・最大
    固有値 ``lam`` について ``|1 - 2*lam/tr(W2)|`` の大きい方。
    """
    lam = np.linalg.eigvalsh(problem.w2)
    ratios = np.abs(1.0 - 2.0 * lam / problem.tr_w2)
    return float(max(ratios[0], ratios[-1]))


def _negated_at_offset(
    evaluate: Callable[[np.ndarray], np.ndarray], sign: float, d: float
) -> float:
    return -float(evaluate(np.array([sign * d]))[0])


# 局所最大のうち、格子の値が大きい上位だけを磨く。平坦な領域（ノイズ程度の
# 変動で局所最大が数千個並ぶ）で、同値の候補を全部磨いて遅くなるのを避ける。
_MAX_REFINED = 40


def _top_local_maxima(values: np.ndarray, interior: np.ndarray) -> np.ndarray:
    """``interior`` の添字を、``values`` の降順で上位 ``_MAX_REFINED`` 件に絞る。"""
    order = np.argsort(values[interior])[::-1]
    return interior[order[:_MAX_REFINED]]


def _refine_centered(
    evaluate: Callable[[np.ndarray], np.ndarray],
    offsets: np.ndarray,
) -> float:
    """中心からのずれ ``t`` を対数間隔の格子で調べ、局所最大を磨いた最大値。

    誤差がほぼ比例すると、比 Eq. (21) は ``tr(S1)`` が最小になる
    ``beta* = tr(W12)/tr(W2)`` のごく近くに鋭い山を持ち、広い範囲の粗い格子では
    取りこぼす。磨く変数を ``beta`` そのものではなく中心からのずれ ``d`` にする
    のは、有界の Brent 法の分解能が ``|x|`` に比例する（相対約1.5e-8）ためで、
    ``|beta|`` が1のオーダーのままだと山の幅に届かない。
    """
    best = 0.0
    for sign in (1.0, -1.0):
        values = evaluate(sign * offsets)
        best = max(best, float(values.max()))
        interior = np.flatnonzero(
            (values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:])
        )
        for i in _top_local_maxima(values, interior + 1):
            res = optimize.minimize_scalar(
                partial(_negated_at_offset, evaluate, sign),
                bounds=(offsets[i - 1], offsets[i + 1]),
                method="bounded",
                options={"xatol": 1e-4 * offsets[i]},
            )
            best = max(best, -float(res.fun))
    return best


def _maximize_beta(problem: _BProblem) -> float:
    """``beta`` 軸での粗い格子＋局所探索。``|beta| -> inf`` は極限の式を併用。

    粗い格子（``|beta| <= 1e8``）に加えて、``beta* = tr(W12)/tr(W2)`` を中心にした
    対数間隔の格子（ずれ 1e-14〜1e8）も調べる。
    """
    scale = 1e8
    s_grid = np.linspace(-np.arcsinh(scale), np.arcsinh(scale), 40001)
    grid = np.sinh(s_grid)
    values = problem.ratios(grid, np.ones_like(grid))
    best = max(float(values.max()), _b_tsls_limit(problem))
    interior = np.flatnonzero(
        (values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:])
    )
    for i in _top_local_maxima(values, interior + 1):
        res = optimize.minimize_scalar(
            lambda b: -problem.ratio_beta(b),
            bounds=(grid[i - 1], grid[i + 1]),
            method="bounded",
            options={"xatol": 1e-14 * max(1.0, abs(grid[i]))},
        )
        best = max(best, -float(res.fun))

    def around_center(t: np.ndarray) -> np.ndarray:
        return problem.ratios(problem.beta0 + t, np.ones_like(t))

    return max(
        best, _refine_centered(around_center, np.logspace(-14, 8, 10001))
    )


def _maximize_angle(problem: _BProblem) -> float:
    """角度 ``phi`` の周期関数としての最大化。``beta = inf`` は ``phi = 0``。

    一様な格子に加えて、``beta*`` に対応する角度 ``cot(phi0) = beta*`` を中心にした
    対数間隔の格子（ずれ 1e-13〜pi）も調べる。
    """
    n = 20000
    phi = np.linspace(0.0, np.pi, n, endpoint=False)
    step = np.pi / n
    values = problem.ratios(np.cos(phi), np.sin(phi))
    best = float(values.max())
    left = np.roll(values, 1)
    right = np.roll(values, -1)
    peaks = np.flatnonzero((values >= left) & (values >= right))
    for i in _top_local_maxima(values, peaks):
        res = optimize.minimize_scalar(
            lambda p: -problem.ratio_angle(p),
            bounds=(phi[i] - step, phi[i] + step),
            method="bounded",
            options={"xatol": 1e-15},
        )
        best = max(best, -float(res.fun))

    phi0 = float(np.arctan2(1.0, problem.beta0))

    def around_center(t: np.ndarray) -> np.ndarray:
        return problem.ratios(np.cos(phi0 + t), np.sin(phi0 + t))

    offsets = np.logspace(-13, np.log10(np.pi), 10001)
    return max(best, _refine_centered(around_center, offsets))


def min_trace_s1_ratio(rf: ReducedForm) -> float:
    """``min_beta tr(S1) / tr(W1)``。1に比べて極小なほど誤差が比例に近い。

    ``tr(S1)`` は ``beta`` の2次式で、最小値は ``tr(W1) - tr(W12)**2/tr(W2)``。
    桁落ちしないよう、入力の浮動小数を有理数として計算する。
    """
    t1 = _exact_sum(np.diag(rf.w1))
    t12 = _exact_sum(np.diag(rf.w12))
    t2 = _exact_sum(np.diag(rf.w2))
    return float((t1 - t12 * t12 / t2) / t1)


def b_tsls(rf: ReducedForm, method: BMethod = "angle") -> float:
    """``B_TSLS(W_hat)``（Eq. (21)）。``W_hat`` のスケールには依らない。"""
    problem = _BProblem(rf)
    if method == "beta":
        return _maximize_beta(problem)
    if method == "angle":
        return _maximize_angle(problem)
    raise ValueError(f"unknown method: {method}")


def mop_effective_f(
    rf: ReducedForm,
    *,
    alpha: float = 0.05,
    method: BMethod = "angle",
) -> MopResult:
    """有効F統計量と、閾値 ``tau`` ごとの単純版・一般化版の臨界値。"""
    w2 = (rf.w2 + rf.w2.T) / 2.0
    b = b_tsls(rf, method)
    simple: dict[Fraction, CriticalValue] = {}
    generalized: dict[Fraction, CriticalValue] = {}
    for tau in TAUS:
        x_simple = float(1 / tau)
        simple[tau] = patnaik_critical_value(w2, x_simple, alpha)
        x_general = float(Fraction(b) / tau)
        generalized[tau] = patnaik_critical_value(w2, x_general, alpha)
    eigenvalues = np.sort(np.linalg.eigvalsh(w2))[::-1]
    return MopResult(
        f_eff=effective_f(rf.pi, w2, rf.nobs),
        trace_v2=float(np.trace(w2) / rf.nobs),
        w2_eigenvalues=eigenvalues,
        b_tsls=b,
        simple=simple,
        generalized=generalized,
    )
