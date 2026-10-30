"""`fit()`が推定中にGILを手放すことのテスト。

GILを持ったまま推定すると、Webサーバー・GUIに組み込んだときに他のPythonスレッドが止まり、
複数のfitをスレッドで並行しても速くならない。境界の関数は、列の取り込み（GILが要る）の後の
純Rust部分を`py.detach`で囲んでいる。

判定は、`fit()`の実行中にバックグラウンドの空ループが止まっていた最長の時間で行う。GILを
持ったままだと、Rustの計算が終わるまで空ループは一度も進めず、最長の停止時間は`fit()`の
大半を占める。手放していれば、GILを返してもらうまでの待ち（切り替え間隔）程度に収まる。
空ループの回数で見ないのは、`fit()`の前後のPython側の処理中にも空ループが進むため。
`sleep`は刻みがOSで違う（Windowsは約15ms）ので、空ループにしている。
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable

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

N = 200_000
X_NAMES = ["x1", "x2", "x3", "x4", "x5"]

# 最長の停止時間が`fit()`の所要時間に占める割合の上限。GILを持ったままだと0.8を超え、
# 手放していれば0.05未満（切り替え間隔を短くしてあるので、数msの待ちが全体に占める割合）。
MAX_STALL_FRACTION = 0.25

# GILを返してもらうまでの待ちを短くして、手放した場合の停止時間を安定して小さくする。
SWITCH_INTERVAL = 0.0002


PERIODS = 10

# 手法ごとの行数。デバッグビルド（CIの`maturin develop`）で1回のfitが1〜3秒になる大きさ。
# 反復最適化・第一段階を持つ手法は、同じ行数でも桁違いに遅い。
ROWS = {
    "OLS": 200_000,
    "WLS": 200_000,
    "FE": 200_000,
    "RE": 50_000,
    "IV": 50_000,
    "Logit": 20_000,
    "Probit": 20_000,
    "Tobit": 20_000,
}

# これより短いfit（リリースビルドで小さなデータを使ったとき等）は、切り替え間隔や
# スケジューラの揺らぎと区別できないので測らない。
MIN_MEASURABLE_SECONDS = 0.02


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    """全手法で使える列を持つデータ。`y_bin`（二値）・`y_cens`（0で左打ち切り）・
    `w`（重み）・`entity`（10期のパネル）・内生変数`endog`と操作変数`z1`/`z2`を含む。"""
    rng = np.random.default_rng(587)
    x = rng.normal(size=(N, len(X_NAMES)))
    z = rng.normal(size=(N, 2))
    u = rng.normal(size=N)
    endog = z.sum(axis=1) + 0.5 * u + rng.normal(size=N)
    latent = x @ np.arange(1.0, len(X_NAMES) + 1) + endog + u
    frame = {name: x[:, i] for i, name in enumerate(X_NAMES)}
    frame.update(
        y=latent,
        y_bin=(latent > 0).astype(int),
        y_cens=np.maximum(latent, 0.0),
        w=rng.uniform(0.5, 2.0, size=N),
        entity=np.arange(N) // PERIODS,
        endog=endog,
        z1=z[:, 0],
        z2=z[:, 1],
    )
    return pl.DataFrame(frame)


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


def _longest_stall_during(fit: Callable[[], object]) -> tuple[float, float]:
    """`fit()`の実行中にバックグラウンドの空ループが止まっていた最長の時間と、`fit()`の所要時間。"""
    longest = 0.0
    started = threading.Event()
    stop = threading.Event()

    def spin() -> None:
        nonlocal longest
        last = time.perf_counter()
        started.set()
        while not stop.is_set():
            now = time.perf_counter()
            longest = max(longest, now - last)
            last = now

    previous = sys.getswitchinterval()
    sys.setswitchinterval(SWITCH_INTERVAL)
    thread = threading.Thread(target=spin)
    thread.start()
    try:
        started.wait()
        longest = 0.0
        t0 = time.perf_counter()
        fit()
        return longest, time.perf_counter() - t0
    finally:
        stop.set()
        thread.join()
        sys.setswitchinterval(previous)


@pytest.mark.parametrize("method", FITS)
def test_fit_lets_other_python_threads_run(method, data):
    fit = FITS[method]
    frame = data.head(ROWS[method])
    stall, duration = _longest_stall_during(lambda: fit(frame))
    if duration < MIN_MEASURABLE_SECONDS:
        pytest.skip(f"fit finished in {duration:.4f}s, too short to measure")
    assert stall < MAX_STALL_FRACTION * duration, (
        f"another thread was stalled for {stall:.3f}s of a {duration:.3f}s fit"
    )
