"""FEの3次元以上の固定効果（`effects`が3〜4列）の合成データセットの生成。

1-way/2-wayのシナリオ（`datasets.py`の`generate_fe_dataset`）とは乱数列を共有しない
別モジュールにする（既存の凍結CSVを変えずに、3つ目以降の次元を持つシナリオを足すため）。
列は`entity`・`time`・`c`（3つ目の効果）・（4次元のみ`d`）・`region`（クラスター用）・`x1`・`x2`・`y`。
`y = x1·β1 + x2·β2 + Σ_d 効果_d[水準] + 誤差`で、`x1`・`x2`は個体と時点の効果と相関させる
（固定効果を落とすと係数が偏る）。

シナリオ（`MULTI_WAY_SCENARIOS`）:

- `balanced`: 個体×時点×`c`の全組合せが1回ずつ（格子）。
- `unbalanced`: 格子から行を無作為に落とす。
- `nested`: `c`が個体にネストする（`c`は個体の関数）。`c`の効果は個体の効果と分離できない。
- `sparse`: 格子の約15%だけを残す。
- `disconnected`: 水準が重ならない2つのブロック（連結成分が2つ）。
- `repeated`: 同じ組合せの行が複数回現れる（`x`・`y`は行ごとに別の値）。
- `four_way`: 個体×時点×`c`×`d`の4次元。
- `small`: 個体5×時点3×`c`2の小さな不均衡パネル（自由度が小さい成功パス）。
- `heteroskedastic`: `unbalanced`と同じ格子で、誤差の標準偏差が`|x1|`に比例して大きい。

全シナリオに、個体を6人ずつまとめたクラスター用の列`region`（個体の入れ子。`small`は個体を
2人ずつ）を付ける。`nested`の`c`は`region`と同じ区分（個体を6人ずつ）で、個体と`c`の両方が
`region`に入れ子になる（効果の外にあるクラスター列での、複数の効果の入れ子の判定）。

水準は文字列（`e00`・`t0`・`c0`・`d0`）で、各次元の各水準は少なくとも2行に現れる
（singletonを作らない）。行順は無作為に並べ替える。
"""

from __future__ import annotations

import numpy as np
import polars as pl

MULTI_WAY_SCENARIOS = [
    "balanced",
    "unbalanced",
    "nested",
    "sparse",
    "disconnected",
    "repeated",
    "four_way",
    "small",
    "heteroskedastic",
]

TRUE_BETA = np.array([1.5, -0.7])
_N_ENTITIES = 24
_N_PERIODS = 6
_N_THIRD = 4
_N_FOURTH = 3
_MIN_ROWS_PER_LEVEL = 2
_ENTITIES_PER_REGION = 6
_SMALL_ENTITIES_PER_REGION = 2
_NOISE_SD = 0.5
_HETEROSKEDASTIC_NOISE_SLOPE = 0.6


def _all_cells(sizes: list[int]) -> np.ndarray:
    """各次元の水準の全組合せ（行ごとに次元ごとの水準番号、辞書順）。"""
    grids = np.meshgrid(*[np.arange(s) for s in sizes], indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


def _thin(
    cells: np.ndarray,
    sizes: list[int],
    keep_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """確率`keep_prob`で行を残す。どの次元の水準も`_MIN_ROWS_PER_LEVEL`行を下回る
    ならその行は落とさない。"""
    counts = [
        np.bincount(cells[:, d], minlength=s) for d, s in enumerate(sizes)
    ]
    keep = np.ones(len(cells), dtype=bool)
    for row in rng.permutation(len(cells)):
        if rng.uniform() < keep_prob:
            continue
        if all(
            counts[d][cells[row, d]] > _MIN_ROWS_PER_LEVEL
            for d in range(len(sizes))
        ):
            keep[row] = False
            for d in range(len(sizes)):
                counts[d][cells[row, d]] -= 1
    return cells[keep]


def _cells(
    scenario: str, rng: np.random.Generator
) -> tuple[np.ndarray, list[int]]:
    """シナリオの行（次元ごとの水準番号）と各次元の水準数を返す。"""
    sizes = [_N_ENTITIES, _N_PERIODS, _N_THIRD]
    if scenario == "balanced":
        return _all_cells(sizes), sizes
    if scenario in ("unbalanced", "heteroskedastic"):
        return _thin(_all_cells(sizes), sizes, 0.5, rng), sizes
    if scenario == "small":
        small = [5, 3, 2]
        return _thin(_all_cells(small), small, 0.6, rng), small
    if scenario == "sparse":
        return _thin(_all_cells(sizes), sizes, 0.15, rng), sizes
    if scenario == "nested":
        base = _thin(_all_cells(sizes[:2]), sizes[:2], 0.6, rng)
        third = (base[:, 0] // _ENTITIES_PER_REGION)[:, None]
        return np.hstack([base, third]), sizes
    if scenario == "disconnected":
        half_e, half_t = _N_ENTITIES // 2, _N_PERIODS // 2
        half_c = _N_THIRD // 2
        blocks = []
        for block in range(2):
            local = _all_cells([half_e, half_t, half_c])
            offset = np.array([block * half_e, block * half_t, block * half_c])
            thinned = _thin(local, [half_e, half_t, half_c], 0.7, rng)
            blocks.append(thinned + offset)
        return np.vstack(blocks), sizes
    if scenario == "repeated":
        base = _thin(_all_cells(sizes), sizes, 0.4, rng)
        extra = base[rng.choice(len(base), size=len(base) // 4, replace=False)]
        twice = base[
            rng.choice(len(base), size=len(base) // 10, replace=False)
        ]
        return np.vstack([base, extra, twice, twice]), sizes
    if scenario == "four_way":
        four = [*sizes, _N_FOURTH]
        return _thin(_all_cells(four), four, 0.25, rng), four
    raise ValueError(f"unknown multi-way scenario: {scenario}")


_LEVEL_PREFIX = ["e", "t", "c", "d"]
_COLUMN_NAMES = ["entity", "time", "c", "d"]


def generate_fe_multi_way_dataset(
    scenario: str, seed: int = 42
) -> tuple[pl.DataFrame, np.ndarray]:
    """指定シナリオの3次元以上のパネルデータと真の係数を返す。

    Args:
        scenario: `MULTI_WAY_SCENARIOS`のいずれか。
        seed: 乱数シード。

    Returns:
        `(df, true_beta)`。列は`entity`・`time`・`c`（・`d`）・`region`・`x1`・`x2`・`y`。
    """
    rng = np.random.default_rng(seed)
    cells, sizes = _cells(scenario, rng)
    effects = [rng.normal(size=s) for s in sizes]
    n = len(cells)
    x = np.column_stack(
        [
            rng.normal(size=n) + 0.8 * effects[0][cells[:, 0]],
            rng.normal(size=n) + 0.6 * effects[1][cells[:, 1]],
        ]
    )
    noise_sd = _NOISE_SD
    if scenario == "heteroskedastic":
        noise_sd = _NOISE_SD + _HETEROSKEDASTIC_NOISE_SLOPE * np.abs(x[:, 0])
    y = x @ TRUE_BETA + rng.normal(size=n) * noise_sd
    for d, effect in enumerate(effects):
        y = y + effect[cells[:, d]]

    order = rng.permutation(n)
    data = {}
    for d, size in enumerate(sizes):
        width = len(str(size - 1))
        data[_COLUMN_NAMES[d]] = [
            f"{_LEVEL_PREFIX[d]}{level:0{width}d}" for level in cells[order, d]
        ]
    per_region = (
        _SMALL_ENTITIES_PER_REGION
        if scenario == "small"
        else _ENTITIES_PER_REGION
    )
    data["region"] = [f"r{level // per_region}" for level in cells[order, 0]]
    data["x1"] = x[order, 0].tolist()
    data["x2"] = x[order, 1].tolist()
    data["y"] = y[order].tolist()
    return pl.DataFrame(data), TRUE_BETA.copy()
