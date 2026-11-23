"""C統計量のオラクル（`benchmark/iv/references/c_statistic_oracle.py`）の自己検証。

オラクルは`IVResults.c_stat()`の参照値を作るnumpyの独立実装。ここでは独立なパッケージで
確かめる。

- 内生変数の外生性: linearmodelsの`IVGMMResults.c_stat`
- 除外操作変数の直交性、内生変数との同時の検定: linearmodelsに無いので、`IVGMM`の
  `weight_matrix`・`estimate_parameters`・`_j_statistic`を部品に`c_stat`と同じ手順で組み直した
  参照実装（`benchmark/iv/references/linearmodels_c_stat.py`）。J統計量も照合する。

内生変数が2個・操作変数が3個のモデルと、内生変数が1個・操作変数が4個のモデルを使う。後者は
操作変数を1〜2個外しても過剰識別のままで、外す列の選び方が結果に効く。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _assertions import assert_close
from _constants import DATA_DIR
from _tolerances import TOLERANCES
from linearmodels.iv import IVGMM

from benchmark.iv.references import linearmodels_c_stat
from benchmark.iv.references.c_statistic_oracle import c_statistic

_CLOSE = {"rtol": TOLERANCES["iv_c_statistic"]["rtol"], "atol": 0.0}
_LAGS = 2
_N_GROUPS = 10

_WEIGHTS = {
    "classical": {"weight_type": "unadjusted"},
    "robust": {"weight_type": "robust"},
    "cluster": {"weight_type": "clustered"},
    "hac": {
        "weight_type": "kernel",
        "kernel": "bartlett",
        "bandwidth": _LAGS,
    },
}

_TWO_ENDOG = {
    "exog": ["const", "x1"],
    "endog": ["endog1", "endog2"],
    "instruments": ["z1", "z2", "z3"],
}
# endog2を4本目の除外操作変数として使う（内生変数1個・除外操作変数4個）。
_ONE_ENDOG = {
    "exog": ["const", "x1"],
    "endog": ["endog1"],
    "instruments": ["z1", "z2", "z3", "endog2"],
}


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "iv_baseline_multi_endog.csv")
    df["const"] = 1.0
    return df


def _groups(frame: pd.DataFrame) -> np.ndarray:
    return np.arange(len(frame)) % _N_GROUPS


def _config(frame: pd.DataFrame, weight: str) -> dict:
    config = dict(_WEIGHTS[weight])
    if weight == "cluster":
        config["clusters"] = _groups(frame)
    return config


def _oracle(frame, spec, weight, *, endog, instruments):
    return c_statistic(
        frame["y"].to_numpy(),
        frame[spec["exog"]].to_numpy(),
        frame[spec["endog"]].to_numpy(),
        frame[spec["instruments"]].to_numpy(),
        tested_endog=[spec["endog"].index(name) for name in endog],
        tested_instruments=[
            spec["instruments"].index(name) for name in instruments
        ],
        weight=weight,
        groups=_groups(frame),
        lags=_LAGS,
    )


def _reference(frame, spec, weight, *, endog, instruments):
    return linearmodels_c_stat.c_statistic(
        frame["y"],
        frame[spec["exog"]],
        frame[spec["endog"]],
        frame[spec["instruments"]],
        tested_endog=endog,
        tested_instruments=instruments,
        **_config(frame, weight),
    )


# ── ライブ linearmodels との照合（内生変数の外生性: c_stat） ──────────


@pytest.mark.parametrize("weight", sorted(_WEIGHTS))
@pytest.mark.parametrize(
    ("spec", "names"),
    [
        (_TWO_ENDOG, ["endog1"]),
        (_TWO_ENDOG, ["endog2"]),
        (_TWO_ENDOG, ["endog1", "endog2"]),
        (_ONE_ENDOG, ["endog1"]),
    ],
    ids=["two/first", "two/second", "two/both", "one/first"],
)
def test_endogeneity_matches_linearmodels(frame, weight, spec, names) -> None:
    model = IVGMM(
        frame["y"],
        frame[spec["exog"]],
        frame[spec["endog"]],
        frame[spec["instruments"]],
        **_config(frame, weight),
    )
    reference = model.fit().c_stat(names)
    ours = _oracle(frame, spec, weight, endog=names, instruments=[])
    assert_close(ours.statistic, float(reference.stat), weight, **_CLOSE)
    assert_close(ours.p_value, float(reference.pval), weight, **_CLOSE)
    assert ours.df == reference.df


# ── ライブ linearmodels との照合（組み直した参照実装） ────────────────

_TESTED = [
    ({"endog": ["endog1"], "instruments": []}, "endogeneity"),
    ({"endog": [], "instruments": ["z3"]}, "z3"),
    ({"endog": [], "instruments": ["endog2"]}, "endog2"),
    ({"endog": [], "instruments": ["z1", "endog2"]}, "z1+endog2"),
    ({"endog": ["endog1"], "instruments": ["z3"]}, "endog1+z3"),
    (
        {"endog": ["endog1"], "instruments": ["z1", "endog2"]},
        "endog1+z1+endog2",
    ),
]


@pytest.mark.parametrize("weight", sorted(_WEIGHTS))
@pytest.mark.parametrize(
    "tested", [t for t, _ in _TESTED], ids=[name for _, name in _TESTED]
)
def test_the_oracle_matches_the_linearmodels_building_blocks(
    frame, weight, tested
) -> None:
    ours = _oracle(frame, _ONE_ENDOG, weight, **tested)
    reference = _reference(frame, _ONE_ENDOG, weight, **tested)
    assert_close(ours.statistic, reference.statistic, weight, **_CLOSE)
    assert_close(ours.j_with_tested, reference.j_with_tested, weight, **_CLOSE)
    assert_close(
        ours.j_without_tested, reference.j_without_tested, weight, **_CLOSE
    )
    assert ours.df == reference.df
    if tested["instruments"]:
        assert ours.j_without_tested > 1e-6  # 外した後も過剰識別


@pytest.mark.parametrize("weight", sorted(_WEIGHTS))
def test_the_building_blocks_reproduce_linearmodels_c_stat(
    frame, weight
) -> None:
    """組み直した参照実装は、`c_stat`が使える範囲では`c_stat`と一致する。"""
    spec = _TWO_ENDOG
    model = IVGMM(
        frame["y"],
        frame[spec["exog"]],
        frame[spec["endog"]],
        frame[spec["instruments"]],
        **_config(frame, weight),
    )
    expected = float(model.fit().c_stat(["endog1"]).stat)
    rebuilt = _reference(frame, spec, weight, endog=["endog1"], instruments=[])
    assert_close(rebuilt.statistic, expected, weight, **_CLOSE)


def test_the_statistic_is_the_difference_of_the_two_hansen_statistics(
    frame,
) -> None:
    result = _oracle(
        frame, _ONE_ENDOG, "robust", endog=["endog1"], instruments=["z1"]
    )
    assert result.df == 2
    assert_close(
        result.statistic,
        result.j_with_tested - result.j_without_tested,
        "difference",
        **_CLOSE,
    )
