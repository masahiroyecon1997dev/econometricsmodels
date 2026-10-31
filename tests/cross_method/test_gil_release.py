"""`fit()`と推定後のメソッドが計算中にGILを手放すことのテスト。

GILを持ったまま推定すると、Webサーバー・GUIに組み込んだときに他のPythonスレッドが止まり、
複数のfitをスレッドで並行しても速くならない。境界の関数は、列の取り込み（GILが要る）の後の
純Rust部分を`py.detach`で囲んでいる。

判定は、`fit()`の実行中にバックグラウンドの空ループが止まっていた最長の時間で行う。GILを
持ったままだと、Rustの計算が終わるまで空ループは一度も進めず、最長の停止時間は`fit()`の
ほぼ全体になる（GILを握り続けるC関数で実測すると1.0）。手放していれば、GILを返して
もらうまでの待ち（切り替え間隔）程度に収まる（8手法の実測で0.006以下）。空ループの回数で
見ないのは、`fit()`の前後のPython側の処理中にも空ループが進むため。`sleep`は刻みがOSで
違う（Windowsは約15ms）ので、空ループにしている。

行数は、デバッグビルド（CIの`maturin develop`）で1回のfitが1〜3秒になる大きさにしてある。
リリースビルドで速すぎて測れないときは、行数を倍にして測り直す（測れないまま通さない）。
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from functools import cache, partial
from typing import Any

import econometricsmodels
import numpy as np
import polars as pl
import pytest
from econometricsmodels import (
    FE,
    IV,
    OLS,
    RE,
    WLS,
    Logit,
    Probit,
    Tobit,
)

X_NAMES = ["x1", "x2", "x3", "x4", "x5"]
PERIODS = 10

FITS: dict[str, Callable[[pl.DataFrame], object]] = {
    "OLS": lambda d: OLS(d, y="y", x=X_NAMES).fit(),
    "WLS": lambda d: WLS(d, y="y", x=X_NAMES, weight="w").fit(),
    "Logit": lambda d: Logit(d, y="y_bin", x=X_NAMES).fit(),
    "Probit": lambda d: Probit(d, y="y_bin", x=X_NAMES).fit(),
    "Tobit": lambda d: Tobit(d, y="y_cens", x=X_NAMES).fit(),
    "IV": lambda d: IV(
        d, y="y", x_exog=X_NAMES, x_endog=["endog"], instruments=["z1", "z2"]
    ).fit(),
    "FE": lambda d: FE(d, y="y", x=X_NAMES, entity="entity").fit(),
    "RE": lambda d: RE(d, y="y", x=X_NAMES, entity="entity").fit(),
}

# ケースごとの行数（`PERIODS`の倍数）。反復最適化・第一段階を持つ手法は、同じ行数でも
# 桁違いに遅い。
ROWS = {
    "OLS": 200_000,
    "WLS": 200_000,
    "Logit": 20_000,
    "Probit": 20_000,
    "Tobit": 20_000,
    "IV": 50_000,
    "FE": 200_000,
    "RE": 50_000,
    # 推定後のメソッド。White検定は補助回帰の項数が多く、同じ行数でも遅い。
    # `marginal_effects`は`fit()`より桁違いに速いので、`fit()`（測定の外）に時間がかかり
    # すぎない範囲で行数を増やしてある。
    "OLS.white_test": 40_000,
    "OLS.breusch_pagan_test": 200_000,
    "OLS.breusch_godfrey_test": 100_000,
    "Logit.marginal_effects": 40_000,
    "Probit.marginal_effects": 40_000,
    "Tobit.marginal_effects": 40_000,
}


def _fit_case(fit: Callable[[pl.DataFrame], object]):
    """`fit()`そのものを測るケース。"""
    return lambda data: partial(fit, data)


def _post_case(method: str, call: Callable[[Any], object]):
    """推定後のメソッド`call(results)`を測るケース。`fit()`は測定の外で済ませる。"""
    return lambda data: partial(call, FITS[method](data))


# 測るケース。キーは手法名（`fit()`）か`<手法名>.<メソッド名>`（推定後のメソッド）。
# 推定後のメソッドは、列の取り込み（GILが要る）の後の計算をGILを手放して走らせるもの。
CASES: dict[str, Callable[[pl.DataFrame], Callable[[], object]]] = {
    **{method: _fit_case(fit) for method, fit in FITS.items()},
    "OLS.white_test": _post_case("OLS", lambda r: r.white_test()),
    "OLS.breusch_pagan_test": _post_case(
        "OLS", lambda r: r.breusch_pagan_test()
    ),
    "OLS.breusch_godfrey_test": _post_case(
        "OLS", lambda r: r.breusch_godfrey_test(time="t", nlags=2)
    ),
    "Logit.marginal_effects": _post_case(
        "Logit", lambda r: r.marginal_effects()
    ),
    "Probit.marginal_effects": _post_case(
        "Probit", lambda r: r.marginal_effects()
    ),
    "Tobit.marginal_effects": _post_case(
        "Tobit", lambda r: r.marginal_effects()
    ),
}


# 最長の停止時間が`fit()`の所要時間に占める割合の上限。GILを持ったままだと1.0に近く、
# 手放していれば0.006以下。全体の1割を超えてGILを握る区間（例: 反復ごとのHessian計算）が
# 残っても検出できるよう、実測に対して十分な余裕を残して低くしてある。
MAX_STALL_FRACTION = 0.10

# GILを返してもらうまでの待ちを短くして、手放した場合の停止時間を安定して小さくする。
SWITCH_INTERVAL = 0.0002

# 所要時間がこの長さに届くまで行数を倍にして測り直す。`MAX_DOUBLINGS`回倍にしても届かない
# ときは、`MIN_MEASURABLE_SECONDS`以上なら測れたものとし、それ未満（切り替え間隔や
# スケジューラの揺らぎと区別できない）なら失敗にする。デバッグビルド（CI）では行数を
# 倍にする必要はない。
TARGET_SECONDS = 0.05
MIN_MEASURABLE_SECONDS = 0.01
MAX_DOUBLINGS = 6


@cache
def make_data(n: int) -> pl.DataFrame:
    """全手法で使える列を持つ`n`行のデータ。`y_bin`（二値）・`y_cens`（0で左打ち切り）・
    `w`（重み）・`entity`（10期のパネル）・時間順`t`・内生変数`endog`と操作変数`z1`/`z2`を含む。"""
    rng = np.random.default_rng(587)
    x = rng.normal(size=(n, len(X_NAMES)))
    z = rng.normal(size=(n, 2))
    u = rng.normal(size=n)
    endog = z.sum(axis=1) + 0.5 * u + rng.normal(size=n)
    latent = x @ np.arange(1.0, len(X_NAMES) + 1) + endog + u
    frame = {name: x[:, i] for i, name in enumerate(X_NAMES)}
    frame.update(
        y=latent,
        y_bin=(latent > 0).astype(int),
        y_cens=np.maximum(latent, 0.0),
        w=rng.uniform(0.5, 2.0, size=n),
        entity=np.arange(n) // PERIODS,
        t=np.arange(n),
        endog=endog,
        z1=z[:, 0],
        z2=z[:, 1],
    )
    return pl.DataFrame(frame)


def _longest_stall_during(fit: Callable[[], object]) -> tuple[float, float]:
    """`fit()`中にバックグラウンドの空ループが止まっていた最長の時間と、`fit()`の所要時間。

    空ループは、GILを取り戻してから停止時間を記録する。GILを握り続けた最後の1回は、`fit()`が
    戻って空ループが走るまで記録されないので、最後に動いた時刻（ハートビート）からの
    経過も停止時間に含める。
    """
    longest = 0.0
    heartbeat = 0.0
    started = threading.Event()
    stop = threading.Event()

    def spin() -> None:
        nonlocal longest, heartbeat
        last = time.perf_counter()
        started.set()
        while not stop.is_set():
            now = time.perf_counter()
            longest = max(longest, now - last)
            last = heartbeat = now

    previous = sys.getswitchinterval()
    sys.setswitchinterval(SWITCH_INTERVAL)
    thread = threading.Thread(target=spin)
    thread.start()
    try:
        started.wait()
        longest = 0.0
        t0 = time.perf_counter()
        fit()
        t1 = time.perf_counter()
        return max(longest, t1 - heartbeat), t1 - t0
    finally:
        stop.set()
        thread.join()
        sys.setswitchinterval(previous)


def _measure(case: str) -> tuple[float, float, int]:
    """`case`を測る。短すぎれば行数を倍にして測り直し、最後の測定の
    （停止時間、所要時間、行数）を返す。"""
    n = ROWS[case]
    for _ in range(MAX_DOUBLINGS + 1):
        stall, duration = _longest_stall_during(CASES[case](make_data(n)))
        if duration >= TARGET_SECONDS:
            break
        n *= 2
    return stall, duration, n


def test_every_method_is_covered():
    """公開されている手法（`*Options`）の`fit()`が`FITS`に載り、測るケース（`CASES`）と
    行数（`ROWS`）が揃っている。新しい手法を足したときの足し忘れを検出する。"""
    methods = {
        name.removesuffix("Options")
        for name in econometricsmodels.__all__
        if name.endswith("Options")
    }
    assert set(FITS) == methods
    assert set(ROWS) == set(CASES)
    assert methods <= set(CASES)
    for case in CASES.keys() - methods:
        assert case.partition(".")[0] in methods, case
    assert all(n % PERIODS == 0 for n in ROWS.values())


@pytest.mark.parametrize("case", CASES)
def test_other_python_threads_run_during(case):
    stall, duration, n = _measure(case)
    assert duration >= MIN_MEASURABLE_SECONDS, (
        f"{case} took only {duration:.4f}s at {n} rows, too short to measure"
    )
    assert stall < MAX_STALL_FRACTION * duration, (
        f"another thread was stalled for {stall:.3f}s of a {duration:.3f}s call"
    )
