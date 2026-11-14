"""MOPオラクル（``mop_oracle.py``）の照合用データとケース、値の表の書き出し。

D1・D2は式から決まる小さな決定論的データで、engine 側のテストも同じ式で
作る。値の表は ``VALUES_PATH`` に書き出す。

使い方（リポジトリルートから）:
    python -m benchmark.iv.references.mop_oracle_cases
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmark.iv.references import mop_oracle as mop

VALUES_PATH = Path(__file__).resolve().parent / "mop_oracle_values.md"

# ivDiagとの比較の記録の節の見出し。手で書いた節で、`main()`が値の表を再生成しても残す。
IVDIAG_MARKER = "## ivDiag 1.0.8との比較"

COV_KINDS: tuple[mop.CovKind, ...] = (
    "classical",
    "hc0",
    "hc1",
    "hc2",
    "hc3",
    "cluster",
    "hac",
)


@dataclass(frozen=True)
class Dataset:
    """``(y, Y, X_exog, Z)`` と、cluster・hac の設定。"""

    y: np.ndarray
    endog: np.ndarray
    x_exog: np.ndarray
    z: np.ndarray
    groups: np.ndarray
    lags: int


def dataset_d1() -> Dataset:
    """D1: S=12、k_exog=2（定数と x）、K=3。"""
    x = np.array(
        [0.3, -1.2, 0.8, 2.0, -0.5, 1.1, 0.7, -0.4, 1.6, 0.2, -1.0, 0.9]
    )
    z1 = np.array(
        [1.0, 0.2, -0.7, 1.5, 0.9, -1.1, 0.3, 0.8, -0.6, 1.2, -0.3, 0.5]
    )
    z2 = np.array(
        [0.4, -0.9, 1.3, 0.1, -0.2, 0.7, -1.4, 0.6, 0.0, -0.8, 1.1, 0.35]
    )
    z3 = np.array(
        [-0.5, 0.7, 0.2, -1.1, 0.8, 0.4, 1.0, -0.3, 0.9, -0.6, 0.15, -0.2]
    )
    u1 = np.array(
        [0.2, -0.5, 0.1, 0.4, -0.3, 0.6, -0.2, 0.05, -0.45, 0.3, 0.15, -0.1]
    )
    u2 = np.array(
        [-0.3, 0.4, 0.5, -0.2, 0.1, -0.6, 0.35, 0.2, -0.1, -0.4, 0.25, 0.3]
    )
    endog = 0.9 * z1 - 0.4 * z2 + 0.3 * x + u1
    y = endog - 0.5 * (0.2 * z1 + 0.7 * z3 - 0.2 * x + u2) + 0.1 * x + u1
    s = x.shape[0]
    return Dataset(
        y=y,
        endog=endog,
        x_exog=np.column_stack([np.ones(s), x]),
        z=np.column_stack([z1, z2, z3]),
        groups=np.arange(s) // 3,
        lags=2,
    )


def dataset_d2(n_instruments: int = 4) -> Dataset:
    """D2: S=240、k_exog=2（定数と x）、K=4（または3）。"""
    s = 240
    i = np.arange(s, dtype=float)
    x = np.sin(0.7 * i + 0.3) + 0.5 * np.cos(0.11 * i)
    z_full = np.column_stack(
        [
            np.sin(1.3 * i * (col + 1) + col)
            + 0.3 * np.cos(0.37 * i * (col + 2))
            for col in range(4)
        ]
    )
    h = 1.0 + 0.8 * np.abs(np.sin(0.05 * i))
    e1 = np.sin(2.1 * i + 1.0) * h
    coefs = [0.5, 0.3, -0.2, 0.1][:n_instruments]
    endog = z_full[:, :n_instruments] @ np.array(coefs) + 0.4 * x + e1
    y = endog + 0.5 * x + np.cos(0.9 * i) * h
    return Dataset(
        y=y,
        endog=endog,
        x_exog=np.column_stack([np.ones(s), x]),
        z=z_full[:, :n_instruments],
        groups=np.arange(s) // 12,
        lags=4,
    )


DATASETS = {
    "D1": dataset_d1,
    "D2_K4": lambda: dataset_d2(4),
    "D2_K3": lambda: dataset_d2(3),
}


@dataclass(frozen=True)
class Case:
    dataset: str
    cov: mop.CovKind
    n_absorbed: int = 0
    alpha: float = 0.05

    @property
    def label(self) -> str:
        parts = [self.dataset, self.cov, f"n_absorbed={self.n_absorbed}"]
        if self.alpha != 0.05:
            parts.append(f"alpha={self.alpha}")
        return ", ".join(parts)


def all_cases() -> list[Case]:
    cases = [Case(d, c) for d in DATASETS for c in COV_KINDS]
    cases += [
        Case("D2_K4", c, n_absorbed=3)
        for c in ("classical", "hc1", "cluster", "hac")
    ]
    cases.append(Case("D2_K4", "hc1", alpha=0.10))
    return cases


def reduced_form_of(
    data: Dataset, cov: mop.CovKind, n_absorbed: int = 0
) -> mop.ReducedForm:
    return mop.reduced_form(
        data.y,
        data.endog,
        data.x_exog,
        data.z,
        cov,
        n_absorbed=n_absorbed,
        groups=data.groups,
        lags=data.lags,
    )


def demean_by_group(a: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """グループごとに平均を引く（FWLで固定効果を吸収する側のデータ）。"""
    out = np.array(a, dtype=float)
    for g in np.unique(groups):
        out[groups == g] -= out[groups == g].mean(axis=0)
    return out


# 誤差がほぼ比例する誘導形の `scale`。`min tr(S1)/tr(W1)` が概ね
# 1e-5、1e-9、1e-13 になる。
NEAR_PROPORTIONAL_SCALES: tuple[float, ...] = (1e-2, 1e-4, 1e-6)


def near_proportional_reduced_form(scale: float) -> mop.ReducedForm:
    """``y`` の誘導形の誤差が ``Y`` の誤差のほぼ ``b`` 倍になる ``W_hat``。

    ``tr(S1)`` が ``beta* = tr(W12)/tr(W2)`` の近くで極端に小さくなり、
    ``B_TSLS`` の比 Eq. (21) が ``beta*`` のまわりに鋭い山を持つ。行数10、
    ``K = 3``、``b = 2.5``。``W = A'A`` を直接与える（``nobs`` は100）。
    """
    i = np.arange(10)[:, None]
    j = np.arange(3)[None, :]
    a_x = np.sin(1.7 * (i + 1) * (j + 1) + 0.3 * j)
    a_e = np.cos(2.3 * (i + 1) + 1.1 * (j + 1) ** 2)
    a = np.hstack([2.5 * a_x + scale * a_e, a_x])
    return mop.ReducedForm(
        nobs=100,
        pi_y=np.zeros(3),
        pi=np.zeros(3),
        w=a.T @ a,
    )


def _fmt(v: float) -> str:
    return f"{float(v):.17g}"


def render_values() -> str:
    lines = [
        "# MOP有効F統計量オラクルの参照値",
        "",
        (
            "`benchmark/iv/references/mop_oracle_cases.py`が生成する"
            "（`python -m benchmark.iv.references.mop_oracle_cases`）。"
            "実装は`benchmark/iv/references/mop_oracle.py`。"
            "数値は有効数字17桁（`%.17g`）。"
        ),
        "",
        (
            "- `B_TSLS`は2つの最大化の方法で求める。`beta`は`beta`軸の粗い格子＋"
            "局所探索（`|beta|→∞`は極限の式）、`angle`は"
            "`beta=cos(phi)/sin(phi)`の角度での周期関数の最大化"
            "（`beta=∞`は`phi=0`の通常の評価点）。"
            "本表の値は`angle`を使い、`beta`との相対差を併記する。"
        ),
        (
            "- 臨界値は`ncx2.ppf`による。単純版は`x=1/tau`"
            "（`tau=0.30`は`10/3`）、一般化版は`x=B_TSLS/tau`。"
        ),
        (
            "- D1・D2は`dataset_d1`・`dataset_d2`の式で作る。cluster・hacの"
            "設定はD1が`i//3`・`lags=2`、D2が`i//12`・`lags=4`。"
        ),
        (
            "- `near_proportional`は`near_proportional_reduced_form(scale)`が"
            "返す`W_hat`（`nobs=100`、`K=3`）を直接与えたケース。誤差がほぼ"
            "比例し、比 Eq. (21)が`beta*=tr(W12)/tr(W2)`のまわりに鋭い山を"
            "持つ。F_effは定義しない（`pi=0`）。`min tr(S1)/tr(W1)`は"
            "入力の浮動小数を有理数として計算した値。"
        ),
        "",
    ]
    for case in all_cases():
        data = DATASETS[case.dataset]()
        rf = reduced_form_of(data, case.cov, case.n_absorbed)
        lines += _render_block(
            f"## {case.label}",
            rf,
            mop.mop_effective_f(rf, alpha=case.alpha, method="angle"),
            f_eff=True,
        )
    for scale in NEAR_PROPORTIONAL_SCALES:
        rf = near_proportional_reduced_form(scale)
        lines += _render_block(
            f"## near_proportional, scale={scale:g}",
            rf,
            mop.mop_effective_f(rf, method="angle"),
            f_eff=False,
        )
    return "\n".join(lines)


def _render_block(
    title: str, rf: mop.ReducedForm, res: mop.MopResult, *, f_eff: bool
) -> list[str]:
    b_beta = mop.b_tsls(rf, "beta")
    rel = abs(b_beta - res.b_tsls) / res.b_tsls
    rows = [f"| F_eff | {_fmt(res.f_eff)} |"] if f_eff else []
    if not f_eff:
        ratio = mop.min_trace_s1_ratio(rf)
        rows.append(f"| min tr(S1)/tr(W1) | {ratio:.6e} |")
    lines = [
        title,
        "",
        "| 量 | 値 |",
        "|---|---|",
        *rows,
        f"| tr(V2_hat) | {_fmt(res.trace_v2)} |",
        "| eig(W2_hat)（降順） | "
        + ", ".join(_fmt(v) for v in res.w2_eigenvalues)
        + " |",
        f"| B_TSLS (angle) | {_fmt(res.b_tsls)} |",
        f"| B_TSLS (beta) | {_fmt(b_beta)} |",
        f"| 相対差 | {rel:.3e} |",
        "",
        (
            "| tau | 単純版 K_eff | 単純版 臨界値 | 一般化版 x | "
            "一般化版 K_eff | 一般化版 臨界値 |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for tau in mop.TAUS:
        sc, gc = res.simple[tau], res.generalized[tau]
        lines.append(
            f"| {float(tau):.2f} | {_fmt(sc.k_eff)} | {_fmt(sc.value)} | "
            f"{_fmt(gc.x)} | {_fmt(gc.k_eff)} | {_fmt(gc.value)} |"
        )
    lines.append("")
    return lines


def main() -> None:
    kept = ""
    if VALUES_PATH.exists():
        old = VALUES_PATH.read_text(encoding="utf-8")
        if IVDIAG_MARKER in old:
            kept = "\n" + IVDIAG_MARKER + old.split(IVDIAG_MARKER, 1)[1]
    VALUES_PATH.write_text(render_values() + kept, encoding="utf-8")
    print(f"wrote {VALUES_PATH}")


if __name__ == "__main__":
    main()
