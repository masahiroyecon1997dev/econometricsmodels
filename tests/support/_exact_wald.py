"""傾きの全体Wald統計量を、データから有理数（`fractions`）で厳密に求める。

返された`cov_params`を使わず、データ（`X`・`y`）と共分散の種類（classical・hc1・cluster）
から`V = c · B M B`を有理数で組み、傾き部分行列の二次形式`β_S' V_SS⁻¹ β_S`を求める。
ほぼ共線な傾きでは`cov_params`自体が桁落ちで不正確になりうるため、`cov_params`を介した
照合（`tests/cross_method/test_wald_ill_conditioned.py`）では確かめられない、統計量の
精度を確かめるための基準値。入力のf64の値をそのまま有理数にするので、丸めでデータが
変わることによる差は含まない。小さなデータ（数百行）向け（有理数演算のため遅い）。
"""

from __future__ import annotations

from collections.abc import Sequence
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


def exact_slope_wald(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    *,
    cov_type: str = "classical",
    groups: Sequence[object] | None = None,
    n_constants: int = 1,
    df_resid: int | None = None,
    cluster_k: int | None = None,
) -> float:
    """設計行列`x`（先頭`n_constants`列が切片）の傾きが全てゼロという仮説のWald統計量。

    `cov_type`は`"classical"`・`"hc1"`・`"cluster"`（`groups`が必要）。小標本補正は
    classicalが`SSR/(n-k)`、hc1が`n/(n-k)`、clusterが`G/(G-1)·(n-1)/(n-k)`。`df_resid`を
    渡すと、classicalとhc1の`n - k`の代わりに使う（固定効果を吸収したパネルは
    `n - k - 吸収した効果の数`）。`cluster_k`を渡すとclusterの`n - K`の`K`に使う
    （既定は`k`。固定効果がクラスターにネストされたパネルは、fixest流で`k + 1`）。戻り値は`F`ではなく`W`（`F = W / q`）。
    """
    n, k = len(x), len(x[0])
    dof = n - k if df_resid is None else df_resid
    xf = [[F(float(v)) for v in row] for row in x]
    yf = [F(float(v)) for v in y]
    xtx = [
        [sum(xf[i][a] * xf[i][b] for i in range(n)) for b in range(k)]
        for a in range(k)
    ]
    xty = [[sum(xf[i][a] * yf[i] for i in range(n))] for a in range(k)]
    beta = _solve(xtx, xty)
    resid = [
        yf[i] - sum(xf[i][a] * beta[a][0] for a in range(k)) for i in range(n)
    ]
    inverse = _solve(
        xtx, [[F(int(a == b)) for b in range(k)] for a in range(k)]
    )

    if cov_type == "classical":
        sigma2 = sum(e * e for e in resid) / dof
        cov = [[sigma2 * inverse[a][b] for b in range(k)] for a in range(k)]
    else:
        if cov_type == "hc1":
            meat = [
                [
                    sum(xf[i][a] * xf[i][b] * resid[i] ** 2 for i in range(n))
                    for b in range(k)
                ]
                for a in range(k)
            ]
            correction = F(n, dof)
        elif cov_type == "cluster":
            if groups is None:
                raise ValueError("cluster needs groups")
            labels = sorted(set(groups), key=str)
            meat = [[F(0)] * k for _ in range(k)]
            for label in labels:
                rows = [i for i in range(n) if groups[i] == label]
                score = [
                    sum(xf[i][a] * resid[i] for i in rows) for a in range(k)
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
    return float(sum(b[0] * s[0] for b, s in zip(b_slopes, solved)))
