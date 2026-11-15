"""傾きの全体Wald統計量を、データから有理数（`fractions`）で厳密に求める。

返された`cov_params`を使わず、データ（`X`・`y`）と共分散の種類（classical・hc0〜hc3・cluster）
から`V = c · B M B`を有理数で組み、傾き部分行列の二次形式`β_S' V_SS⁻¹ β_S`を求める。
ほぼ共線な傾きでは`cov_params`自体が桁落ちで不正確になりうるため、`cov_params`を介した
照合（`tests/cross_method/test_wald_ill_conditioned.py`）では確かめられない、統計量の
精度を確かめるための基準値。入力のf64の値をそのまま有理数にするので、丸めでデータが
変わることによる差は含まない。小さなデータ（数百行）向け（有理数演算のため遅い）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction as F


def _solve(a: list[list[F]], b: list[list[F]]) -> list[list[F]]:
    """`a x = b`をGauss-Jordan法で解く（`a`はd×d、`b`はd×m）。"""
    d = len(a)
    m = [list(a[i]) + list(b[i]) for i in range(d)]
    for i in range(d):
        pivot = next(r for r in range(i, d) if m[r][i] != 0)
        m[i], m[pivot] = m[pivot], m[i]
        m[i] = [v / m[i][i] for v in m[i]]
        for r in range(d):
            if r != i and m[r][i] != 0:
                factor = m[r][i]
                m[r] = [x - factor * p for x, p in zip(m[r], m[i])]
    return [row[d:] for row in m]


def exact_projected_design(
    z: Sequence[Sequence[float]], x: Sequence[Sequence[float]]
) -> list[list[F]]:
    """操作変数`Z`への射影 `X̂ = Z (Z'Z)⁻¹ Z'X`（有理数。2SLSの第二段階の設計行列）。"""
    n, kz, kx = len(z), len(z[0]), len(x[0])
    zf = [[F(float(v)) for v in row] for row in z]
    xf = [[F(float(v)) for v in row] for row in x]
    ztz = [
        [sum(zf[i][a] * zf[i][b] for i in range(n)) for b in range(kz)]
        for a in range(kz)
    ]
    ztx = [
        [sum(zf[i][a] * xf[i][b] for i in range(n)) for b in range(kx)]
        for a in range(kz)
    ]
    coef = _solve(ztz, ztx)
    return [
        [sum(zf[i][a] * coef[a][b] for a in range(kz)) for b in range(kx)]
        for i in range(n)
    ]


def _exact_fit(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    x_hat: Sequence[Sequence[F]] | None,
) -> tuple[
    list[list[F]], list[list[F]], list[list[F]], list[F], list[list[F]]
]:
    """有理数の最小二乗（2SLSなら`x_hat`をbread側に使う）。

    `(bread行列, 係数, (bread'bread)⁻¹, 構造残差, bread行)`を返す。bread行は、通常は`x`の
    各行、`x_hat`があれば`x_hat`の各行（ミートのスコアと、レバレッジに使う）。
    """
    n, k = len(x), len(x[0])
    xf = [[F(float(v)) for v in row] for row in x]
    yf = [F(float(v)) for v in y]
    rows = [list(r) for r in x_hat] if x_hat is not None else xf
    gram = [
        [sum(rows[i][a] * rows[i][b] for i in range(n)) for b in range(k)]
        for a in range(k)
    ]
    rhs = [[sum(rows[i][a] * yf[i] for i in range(n))] for a in range(k)]
    beta = _solve(gram, rhs)
    resid = [
        yf[i] - sum(xf[i][a] * beta[a][0] for a in range(k)) for i in range(n)
    ]
    inverse = _solve(
        gram, [[F(int(a == b)) for b in range(k)] for a in range(k)]
    )
    return gram, beta, inverse, resid, rows


@dataclass(frozen=True)
class ExactInference:
    """`exact_inference`の結果（浮動小数点に直した値）。"""

    beta: list[float]
    se: list[float]
    wald: float


def exact_inference(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    *,
    cov_type: str = "classical",
    groups: Sequence[object] | None = None,
    n_constants: int = 1,
    df_resid: int | None = None,
    cluster_k: int | None = None,
    x_hat: Sequence[Sequence[F]] | None = None,
    extra_leverage: Sequence[F | float] | None = None,
) -> ExactInference:
    """係数・標準誤差と、設計行列`x`（先頭`n_constants`列が切片）の傾きが全てゼロという仮説のWald統計量。

    `cov_type`は`"classical"`・`"hc0"`〜`"hc3"`・`"cluster"`（`groups`が必要）。小標本補正は
    classicalが`SSR/(n-k)`、hc1が`n/(n-k)`、clusterが`G/(G-1)·(n-1)/(n-k)`。hc2・hc3は
    行ごとに`e²/(1-h)`・`e²/(1-h)²`（`h`はレバレッジ）。`df_resid`を
    渡すと、classicalとhc1の`n - k`の代わりに使う（固定効果を吸収したパネルは
    `n - k - 吸収した効果の数`）。`cluster_k`を渡すとclusterの`n - K`の`K`に使う
    （既定は`k`。固定効果がクラスターにネストされたパネルは、fixest流で`k + 1`）。
    `x_hat`（`exact_projected_design`の戻り値）を渡すと2SLSになる（breadと係数を`x_hat`
    から、構造残差を`x`から作る）。`extra_leverage`は、行ごとに足すレバレッジ（固定効果の
    ダミー分。1-wayなら`1/T_i`）。Wald統計量は`F`ではなく`W`（`F = W / q`）。標準誤差は`cov`の対角の平方根。
    """
    n, k = len(x), len(x[0])
    dof = n - k if df_resid is None else df_resid
    _, beta, inverse, resid, rows = _exact_fit(x, y, x_hat)

    if cov_type == "classical":
        sigma2 = sum(e * e for e in resid) / dof
        cov = [[sigma2 * inverse[a][b] for b in range(k)] for a in range(k)]
    else:
        if cov_type in {"hc0", "hc1", "hc2", "hc3"}:
            weights = [e * e for e in resid]
            if cov_type in {"hc2", "hc3"}:
                leverage = [
                    sum(
                        rows[i][a] * inverse[a][b] * rows[i][b]
                        for a in range(k)
                        for b in range(k)
                    )
                    + (
                        F(0)
                        if extra_leverage is None
                        else F(extra_leverage[i])
                    )
                    for i in range(n)
                ]
                power = 1 if cov_type == "hc2" else 2
                weights = [
                    w / (1 - h) ** power for w, h in zip(weights, leverage)
                ]
            meat = [
                [
                    sum(rows[i][a] * rows[i][b] * weights[i] for i in range(n))
                    for b in range(k)
                ]
                for a in range(k)
            ]
            correction = F(n, dof) if cov_type == "hc1" else F(1)
        elif cov_type == "cluster":
            if groups is None:
                raise ValueError("cluster needs groups")
            labels = sorted(set(groups), key=str)
            meat = [[F(0)] * k for _ in range(k)]
            for label in labels:
                members = [i for i in range(n) if groups[i] == label]
                score = [
                    sum(rows[i][a] * resid[i] for i in members)
                    for a in range(k)
                ]
                for a in range(k):
                    for b in range(k):
                        meat[a][b] += score[a] * score[b]
            g = len(labels)
            kk = k if cluster_k is None else cluster_k
            correction = F(g, g - 1) * F(n - 1, n - kk)
        else:
            raise ValueError(f"unsupported cov_type: {cov_type}")
        bm = [
            [
                sum(inverse[a][c] * meat[c][b] for c in range(k))
                for b in range(k)
            ]
            for a in range(k)
        ]
        cov = [
            [
                correction * sum(bm[a][c] * inverse[c][b] for c in range(k))
                for b in range(k)
            ]
            for a in range(k)
        ]

    slopes = range(n_constants, k)
    v_slopes = [[cov[a][b] for b in slopes] for a in slopes]
    b_slopes = [[beta[a][0]] for a in slopes]
    solved = _solve(v_slopes, b_slopes)
    wald = float(sum(b[0] * s[0] for b, s in zip(b_slopes, solved)))
    # 有理数の平方根は無いので、分散を浮動小数点に直してから平方根を取る。
    se = [float(cov[a][a]) ** 0.5 for a in range(k)]
    return ExactInference(beta=[float(b[0]) for b in beta], se=se, wald=wald)


def exact_slope_wald(
    x: Sequence[Sequence[float]], y: Sequence[float], **kwargs: object
) -> float:
    """`exact_inference`のWald統計量だけ（引数は同じ）。"""
    return exact_inference(x, y, **kwargs).wald  # type: ignore[arg-type]
