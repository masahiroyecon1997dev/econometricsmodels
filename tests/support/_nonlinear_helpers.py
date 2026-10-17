"""`tests/nonlinear/`（Logit/Probit/Tobit）が共有するテストデータとアサーション。

- 分離・準分離・打ち切りの境界を作る合成データ（`separated_dataset`等）と、
  Wooldridge `apple`のロード（`load_apple_adoption_dataset`）。
- `check_margeff`: 限界効果6統計量を`at`×係数名の全組み合わせで比較する。

全系統で共通のデータ生成は`_helpers.py`、アサーションは`_assertions.py`を参照。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _assertions import assert_close
from _constants import MARGEFF_AT
from scipy.stats import norm


def separated_dataset() -> pl.DataFrame:
    """分離データ（`x1`の真の係数を極端に大きくし、全観測がx1の符号だけで
    完全に分類できるようにしたDGP）を生成する。

    `Logit`/`Probit`いずれの`ComputationError`（分離検出）テストにも
    使う（Probit側もsigmoidベースのDGPをそのまま流用する。
    `test_logit_validation.py`のLogit版のProbit版という位置づけ、DGP自体の
    正確なProbitリンクである必要はない）。

    `benchmark/`側のDGP（`benchmark/nonlinear/datasets.py`等）と同じ
    `numpy`（`np.random.default_rng`）ベースのベクトル化演算で書く。
    """
    rng = np.random.default_rng(42)
    n = 200
    beta = (0.0, 100.0, 0.5)
    x1 = rng.uniform(-2.0, 2.0, size=n)
    x2 = rng.uniform(-1.0, 1.0, size=n)
    z = beta[0] + beta[1] * x1 + beta[2] * x2
    p = 1.0 / (1.0 + np.exp(-z))
    y = rng.binomial(1, p).astype(np.float64)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


def censored_separation_dataset(
    noise_sd: float, seed: int = 42, n: int = 200
) -> pl.DataFrame:
    """`separated_dataset`の左打ち切り版（Tobit用）。`x1`の真の係数が100と極端に大きい。

    `y* = 100·x1 + 0.5·x2 + N(0, noise_sd²)`（`x1`~U(-2, 2)、`x2`~U(-1, 1)）を0で
    左打ち切りする（`noise_sd=1`、`seed=42`で約52%が打ち切り）。`noise_sd=0`は`x1`の符号だけで
    `y`が決まる完全分離で、Tobitでは係数の発散ではなく`σ→0`の退化として現れる。
    ノイズがあれば識別可能で、真値を回復できる。

    外部のデータファイルは持たず、numpyの`default_rng(seed)`で作る。数値照合を
    しない（エラーパスと収束の確認だけ）ため、ベンチマークCSVとしては固定しない。
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2.0, 2.0, size=n)
    x2 = rng.uniform(-1.0, 1.0, size=n)
    y_star = 100.0 * x1 + 0.5 * x2
    if noise_sd > 0.0:
        y_star = y_star + rng.normal(0.0, noise_sd, size=n)
    return pl.DataFrame({"y": np.maximum(y_star, 0.0), "x1": x1, "x2": x2})


def small_sample_separation_dataset() -> pl.DataFrame:
    """自由度が1（`n=k+1`、`k`は定数項を含む4）で完全分離になる5行のデータ。

    `benchmark.nonlinear.datasets.generate_binary_choice_dataset("baseline", n=5, k=3,
    seed=42)`の出力を固定した値（Logit/Probitで同じ`y`になる）。極小標本では
    パラメータの数と観測数がほぼ等しく、ほぼ必ず完全分離になる。標準化パラメータの
    ノルムによる旧来の検出は、この種のデータを無警告の「成功」として返していた
    （係数は有限だが標準誤差が数千になる）。
    """
    return pl.DataFrame(
        {
            "y": [1.0, 0.0, 1.0, 0.0, 0.0],
            "x1": [
                -1.9510351886538364,
                -0.3162425923435822,
                0.8793979748628286,
                1.1272412069680329,
                0.36875078408249884,
            ],
            "x2": [
                -1.302179506862318,
                -0.016801157504288795,
                0.7777919354289483,
                0.4675093422520456,
                -0.9588826008289989,
            ],
            "x3": [
                0.12784040316728537,
                -0.85304392757358,
                0.06603069756121605,
                -0.8592924628832382,
                0.8784503013072725,
            ],
        }
    )


def multi_dummy_separation_dataset(seed: int = 58) -> pl.DataFrame:
    """3つのダミーが同時に準完全分離するデータ（`n=600`、連続変数5本）。

    d0・d2が1なら必ず`y=1`、d1が1なら必ず`y=0`。分離方向が複数ある（錐の次元が2以上）ので、
    情報行列の近零固有空間が多次元になる。収束点の係数方向や最小固有ベクトルの方向だけで
    分離を判定する方式は、この種のデータを取りこぼす（lbfgsで実測）。
    """
    rng = np.random.default_rng(seed)
    n = 600
    d0 = (rng.random(n) > 0.8).astype(np.float64)
    d1 = (rng.random(n) > 0.8).astype(np.float64)
    d2 = (rng.random(n) > 0.9).astype(np.float64)
    c = rng.normal(size=(n, 5))
    base = (
        rng.random(n)
        < 1.0 / (1.0 + np.exp(-c @ np.array([0.5, -0.4, 0.3, 0.2, -0.3])))
    ).astype(np.float64)
    y = np.where((d0 == 1) | (d2 == 1), 1.0, np.where(d1 == 1, 0.0, base))
    data = {"y": y, "d0": d0, "d1": d1, "d2": d2}
    for j in range(5):
        data[f"c{j}"] = c[:, j]
    return pl.DataFrame(data)


def strongly_collinear_binary_dataset(
    link: str, seed: int = 6
) -> pl.DataFrame:
    """`x2 = x1 + 1e-3·ε`（相関がほぼ1）で、分離していない2値データを生成する。

    有限の最尤推定量が存在する（係数は`x1`と`x2`で符号が逆に大きく振れうるが、
    線形予測子への寄与は打ち消し合う）。標準化パラメータのノルムによる旧来の
    分離検出はこの種のデータを誤って棄却していた（`seed=6`のProbitで再現）。
    `link`は`"logit"`または`"probit"`。
    """
    rng = np.random.default_rng(seed)
    n = 400
    x1 = rng.normal(size=n)
    x2 = x1 + 1e-3 * rng.normal(size=n)
    x3 = rng.normal(size=n)
    z = 0.5 * x1 + 0.5 * x2 + 0.5 * x3
    p = 1.0 / (1.0 + np.exp(-z)) if link == "logit" else norm.cdf(z)
    y = (rng.random(n) < p).astype(np.float64)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})


def load_apple_adoption_dataset() -> pl.DataFrame:
    """Wooldridge `apple`に`ecolbs > 0`の2値`y`を足したDataFrameをロードする。

    `wooldridge_loader`と同じ理由でCSVとして固定せず都度ロードする。
    """
    pytest.importorskip("wooldridge")
    from benchmark.common.load_wooldridge import load_apple_adoption

    return load_apple_adoption()


def heavily_censored_tobit_dataset() -> pl.DataFrame:
    """打ち切りが重い（60観測中51観測＝85%が0で左打ち切り）Tobit用の合成データ。

    外部の乱数に依存しない線形合同法で作る。Rust側
    （`engine/src/nonlinear/tobit.rs`の`heavily_censored_input`）と同じ手順・同じ値で、
    `max_iter=1`の打ち切り点が全solver（newton/bfgs/lbfgs）で未収束の結果になる挙動を
    両側で同じ入力に対して固定するために使う（かつて`solver="lbfgs"`だけ`SingularHessian`だった）。潜在変数は
    `y* = -1.8 + x1 + 0.5*x2 + e`（`x1`・`x2`・`e`は12個の一様乱数の和-6で近似した
    正規乱数）。`g`は4グループのクラスター列。
    """
    mask = (1 << 64) - 1
    state = 167

    def unit() -> float:
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & mask
        return (state >> 11) / float(1 << 53)

    def normal() -> float:
        return sum(unit() for _ in range(12)) - 6.0

    n = 60
    x1 = [normal() for _ in range(n)]
    x2 = [normal() for _ in range(n)]
    e = [normal() for _ in range(n)]
    y = [max(-1.8 + a + 0.5 * b + d, 0.0) for a, b, d in zip(x1, x2, e)]
    # Rust側（`heavily_censored_input`）と同じ値になっていることの確認。
    assert sum(v == 0.0 for v in y) == 51
    return pl.DataFrame(
        {"y": y, "x1": x1, "x2": x2, "g": [str(i % 4) for i in range(n)]}
    )


def single_overlap_binary_dataset(overlap: float) -> pl.DataFrame:
    """完全分離データに`y=1`の観測を1つ足して、分離を壊す（または壊しかける）データ。

    `x`は`±0.01〜±1.00`の0.01刻み（`x>0`なら`y=1`の完全分離）で、`y=1`の観測を1つ
    `x=-0.01-overlap`に足す。`y=0`で最も`x=0`に近い観測（`x=-0.01`）より左に`y=1`が
    1つあるため、`overlap>0`ならちょうど1観測が分離を壊し、有限のMLEが存在する。
    Rust側（`engine/src/nonlinear/separation.rs`の`single_overlap_data`）と同じデータ。
    説明変数は`x1`1列（`include_intercept=True`）。
    """
    grid = [0.01 * i for i in range(1, 101)]
    x = [-g for g in grid] + grid + [-0.01 - overlap]
    y = [0.0] * 100 + [1.0] * 101
    return pl.DataFrame({"y": y, "x1": x})


def check_margeff(
    res,
    ref_margeff: dict,
    label: str,
    *,
    rtol: float,
    atol: float,
) -> None:
    """限界効果6統計量を`at`×係数名の全組み合わせでリファレンスと比較する。"""
    for at in MARGEFF_AT:
        effects = {
            row["param_name"]: row for row in res.marginal_effects(at=at)
        }
        for name, ref_stats in ref_margeff[at].items():
            row = effects[name]
            assert_close(
                row["effect"],
                ref_stats["effect"],
                f"{label}/{at}/{name}/effect",
                rtol=rtol,
                atol=atol,
            )
            assert_close(
                row["std_error"],
                ref_stats["std_error"],
                f"{label}/{at}/{name}/std_error",
                rtol=rtol,
                atol=atol,
            )
            assert_close(
                row["test_stat"],
                ref_stats["test_stat"],
                f"{label}/{at}/{name}/test_stat",
                rtol=rtol,
                atol=atol,
            )
            assert_close(
                row["p_value"],
                ref_stats["p_value"],
                f"{label}/{at}/{name}/p_value",
                rtol=rtol,
                atol=atol,
            )
            assert_close(
                row["conf_lower"],
                ref_stats["conf_lower"],
                f"{label}/{at}/{name}/conf_lower",
                rtol=rtol,
                atol=atol,
            )
            assert_close(
                row["conf_upper"],
                ref_stats["conf_upper"],
                f"{label}/{at}/{name}/conf_upper",
                rtol=rtol,
                atol=atol,
            )
