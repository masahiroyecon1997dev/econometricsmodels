"""IVのC統計量（`c_stat()`）。

`fit()`では計算しない事後診断のメソッドで、結果が保持する元のデータから`fit()`と同じ手順で
入力を作り直し、検定するモーメント条件を含むモデルと含まないモデルを2段階GMMで推定して、
Hansen Jの差を`χ²`で検定する。参照は次のとおり。

- 内生変数の外生性: linearmodelsの`IVGMMResults.c_stat`。
- 除外操作変数の直交性、内生変数との同時の検定、J統計量: linearmodelsに無いので、`IVGMM`の
  部品で`c_stat`と同じ手順を組み直した参照実装（`benchmark/iv/references/linearmodels_c_stat.py`。
  `c_stat`が使える範囲では`c_stat`と一致する）。numpyの独立実装（`c_statistic_oracle.py`）が
  これらと一致することは`test_c_statistic_oracle.py`で確認する。

重みは`estimator="gmm"`（`gmm_type`が`two_step`・`iterated`）なら`gmm_weight_type`、それ以外
（2SLSと1段階GMM）なら`cov_type`の系統から決める（linearmodelsの`IVGMM`で対応する重みを使う）。
定数項なしの`classical`の重みはlinearmodelsが残差を中心化して`σ̂²`を作るので比べない
（`robust`・`hac`は定数項なしでも一致する）。
"""

from __future__ import annotations

import copy
import json
import pickle
from functools import partial

import pandas as pd
import polars as pl
import pytest
from _assertions import assert_close
from _constants import DATA_DIR
from _error_expectations import (
    COLUMN_LIST_ELEMENT_NOT_STR,
    COLUMN_LIST_NOT_LIST,
    escaped,
    fully_qualified_type_name,
)
from _errors import raises_error
from _helpers import (
    ROW_TIME,
    load_wooldridge_dataset,
    reorder_rows,
    with_row_time,
)
from _tolerances import TOLERANCES
from econometricsmodels import (
    IV,
    CStatisticResult,
    DiagnosticResult,
    IVOptions,
)
from linearmodels.iv import IVGMM
from linearmodels.iv.results import InvalidTestStatistic
from scipy import stats

from benchmark.common import imbalanced_cluster_groups
from benchmark.common.dgp import hac_auto_lag
from benchmark.iv.datasets import (
    scenario_instruments,
    scenario_x_exog,
    with_card_region,
)
from benchmark.iv.references import linearmodels_c_stat

_RTOL = TOLERANCES["iv_c_statistic"]["rtol"]
_ATOL_ZERO = TOLERANCES["iv_c_statistic"]["atol_just_identified"]
_RTOL_BOUNDARY = TOLERANCES["iv_c_statistic"]["rtol_boundary_cluster"]
_CLOSE = partial(assert_close, rtol=_RTOL, atol=0.0)
_LAGS = 2
_N_GROUPS = 10
_CLUSTER = "cluster_group"
_IMBALANCED = "cluster_imbalanced"

# linearmodelsの`IVGMM`の`weight_type`。2SLSの`cov_type`は同じ系統の重みに対応させる。
_LM_WEIGHT = {
    "classical": {"weight_type": "unadjusted"},
    "robust": {"weight_type": "robust"},
    "cluster": {"weight_type": "clustered"},
    "hac": {"weight_type": "kernel", "kernel": "bartlett"},
}
_COV_TYPE_FAMILY = {
    "classical": "classical",
    "hc0": "robust",
    "hc1": "robust",
    "hc2": "robust",
    "hc3": "robust",
    "cluster": "cluster",
    "hac": "hac",
}
_GMM_WEIGHTS = ["classical", "robust", "cluster", "hac"]

# 内生変数2個・除外操作変数3個と、内生変数1個・除外操作変数4個（endog2を4本目の操作変数に
# 使う。検定する操作変数を外しても過剰識別のままで、外す列の選び方が結果に効く）。
_TWO = {
    "x_exog": ["x1"],
    "x_endog": ["endog1", "endog2"],
    "instruments": ["z1", "z2", "z3"],
}
_ONE = {
    "x_exog": ["x1"],
    "x_endog": ["endog1"],
    "instruments": ["z1", "z2", "z3", "endog2"],
}
_ONE_NO_EXOG = {**_ONE, "x_exog": []}


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    df = pl.read_csv(DATA_DIR / "iv_baseline_multi_endog.csv")
    df = df.with_columns(
        (pl.int_range(pl.len()) % _N_GROUPS).alias(_CLUSTER),
        pl.Series(_IMBALANCED, imbalanced_cluster_groups(df.height)),
    )
    return with_row_time(df)


def _options(
    *,
    estimator: str,
    setting: str,
    cluster: str = _CLUSTER,
    hac_lags: int | None = _LAGS,
    **extra,
) -> IVOptions:
    """`setting`は、GMMなら`gmm_weight_type`、2SLSなら`cov_type`の値。"""
    kwargs = dict(extra)
    if estimator == "gmm":
        kwargs["gmm_weight_type"] = setting
        family = setting
    else:
        kwargs["cov_type"] = setting
        family = _COV_TYPE_FAMILY[setting]
    if family == "cluster":
        kwargs["cluster"] = cluster
    if family == "hac":
        kwargs["hac_lags"] = hac_lags
        kwargs["hac_time"] = ROW_TIME
    return IVOptions(estimator=estimator, **kwargs)


def _fit(
    df: pl.DataFrame,
    spec: dict,
    *,
    estimator: str = "2sls",
    setting: str = "classical",
    y: str = "y",
    **extra,
):
    return IV(
        df,
        y=y,
        options=_options(estimator=estimator, setting=setting, **extra),
        **spec,
    ).fit()


def _pandas(df: pl.DataFrame, spec: dict, y: str, *, constant: bool = True):
    """linearmodelsに渡す`(y, exog, endog, instruments)`。"""
    pdf: pd.DataFrame = df.to_pandas()
    exog = pdf[spec["x_exog"]]
    if constant:
        const = pd.Series(1.0, index=pdf.index, name="const")
        exog = pd.concat([const, exog], axis=1)
    return pdf[y], exog, pdf[spec["x_endog"]], pdf[spec["instruments"]]


def _weight_config(
    df: pl.DataFrame,
    family: str,
    *,
    cluster: str = _CLUSTER,
    lags: int = _LAGS,
) -> dict:
    config = dict(_LM_WEIGHT[family])
    if family == "cluster":
        config["clusters"] = df[cluster].to_numpy()
    if family == "hac":
        config["bandwidth"] = lags
    return config


def _lm_c_stat(
    df: pl.DataFrame,
    spec: dict,
    family: str,
    variables: list[str],
    *,
    y: str = "y",
    constant: bool = True,
    **config,
):
    """linearmodelsの`c_stat`（内生変数の外生性）。"""
    y_, exog, endog, instruments = _pandas(df, spec, y, constant=constant)
    weight = _weight_config(df, family, **config)
    return (
        IVGMM(y_, exog, endog, instruments, **weight).fit().c_stat(variables)
    )


def _rebuilt(
    df: pl.DataFrame,
    spec: dict,
    family: str,
    *,
    variables: list[str] = (),
    instruments: list[str] = (),
    y: str = "y",
    constant: bool = True,
    **config,
):
    """`IVGMM`の部品で組み直した参照実装（直交性・同時の検定、J統計量）。"""
    y_, exog, endog, instr = _pandas(df, spec, y, constant=constant)
    return linearmodels_c_stat.c_statistic(
        y_,
        exog,
        endog,
        instr,
        tested_endog=list(variables),
        tested_instruments=list(instruments),
        **_weight_config(df, family, **config),
    )


def _assert_matches(
    result: CStatisticResult, reference, label: str, *, rtol: float = _RTOL
) -> None:
    close = partial(assert_close, rtol=rtol, atol=0.0)
    close(result.statistic, reference.statistic, f"{label}: statistic")
    close(
        result.p_value,
        float(stats.chi2.sf(reference.statistic, reference.df)),
        f"{label}: p-value",
    )
    close(result.j_with_tested, reference.j_with_tested, f"{label}: J with")
    # 丁度識別のJ_cは本実装では厳密に0で、参照は丸め誤差（1e-17台）を持つ。
    assert_close(
        result.j_without_tested,
        reference.j_without_tested,
        f"{label}: J without",
        rtol=rtol,
        atol=_ATOL_ZERO,
    )
    assert result.df == reference.df


# ── ライブ linearmodels との照合（内生変数の外生性: c_stat） ──────────


@pytest.mark.parametrize("weight", _GMM_WEIGHTS)
@pytest.mark.parametrize(
    ("spec", "variables"),
    [
        (_TWO, ["endog1"]),
        (_TWO, ["endog2"]),
        (_TWO, ["endog1", "endog2"]),
        (_ONE, ["endog1"]),
        (_ONE_NO_EXOG, ["endog1"]),
    ],
    ids=["two/first", "two/second", "two/both", "one/first", "one/no-exog"],
)
def test_gmm_endogeneity_matches_linearmodels(
    frame, weight, spec, variables
) -> None:
    result = _fit(frame, spec, estimator="gmm", setting=weight).c_stat(
        variables=variables
    )
    reference = _lm_c_stat(frame, spec, weight, variables)
    assert isinstance(result, CStatisticResult)
    _CLOSE(result.statistic, float(reference.stat), weight)
    _CLOSE(result.p_value, float(reference.pval), weight)
    assert result.df == reference.df == len(variables)
    assert result.variables == variables
    assert result.instruments == []


@pytest.mark.parametrize("cov_type", sorted(_COV_TYPE_FAMILY))
@pytest.mark.parametrize(
    "variables", [["endog1"], ["endog1", "endog2"]], ids=["first", "both"]
)
def test_2sls_endogeneity_uses_the_weight_of_its_cov_type(
    frame, cov_type, variables
) -> None:
    result = _fit(frame, _TWO, setting=cov_type).c_stat(variables=variables)
    reference = _lm_c_stat(frame, _TWO, _COV_TYPE_FAMILY[cov_type], variables)
    _CLOSE(result.statistic, float(reference.stat), cov_type)
    _CLOSE(result.p_value, float(reference.pval), cov_type)


def test_endogeneity_of_all_variables_matches_the_default_of_c_stat(
    frame,
) -> None:
    result = _fit(frame, _TWO, estimator="gmm", setting="robust").c_stat(
        variables=["endog1", "endog2"]
    )
    y, exog, endog, instruments = _pandas(frame, _TWO, "y")
    model = IVGMM(y, exog, endog, instruments, weight_type="robust")
    _CLOSE(result.statistic, float(model.fit().c_stat().stat), "all")


# ── ライブ linearmodels との照合（組み直した参照実装: 直交性・同時・J） ─

_TESTED = {
    "endogeneity": {"variables": ["endog1"]},
    "z3": {"instruments": ["z3"]},
    "endog2": {"instruments": ["endog2"]},
    "z1+endog2": {"instruments": ["z1", "endog2"]},
    "endog1+z3": {"variables": ["endog1"], "instruments": ["z3"]},
    "endog1+z1+endog2": {
        "variables": ["endog1"],
        "instruments": ["z1", "endog2"],
    },
}


@pytest.mark.parametrize("weight", _GMM_WEIGHTS)
@pytest.mark.parametrize("name", sorted(_TESTED))
def test_gmm_orthogonality_and_joint_tests_match_the_rebuilt_reference(
    frame, weight, name
) -> None:
    tested = _TESTED[name]
    result = _fit(frame, _ONE, estimator="gmm", setting=weight).c_stat(
        **tested
    )
    _assert_matches(
        result, _rebuilt(frame, _ONE, weight, **tested), f"{weight}/{name}"
    )
    assert result.variables == tested.get("variables", [])
    assert result.instruments == tested.get("instruments", [])
    if "instruments" in tested:
        # 外した後も過剰識別で、外す列の選び方が結果に効く。
        assert result.j_without_tested > 1e-6


@pytest.mark.parametrize("cov_type", sorted(_COV_TYPE_FAMILY))
@pytest.mark.parametrize("name", ["z3", "z1+endog2", "endog1+z3"])
def test_2sls_orthogonality_and_joint_tests_match_the_rebuilt_reference(
    frame, cov_type, name
) -> None:
    tested = _TESTED[name]
    result = _fit(frame, _ONE, setting=cov_type).c_stat(**tested)
    reference = _rebuilt(frame, _ONE, _COV_TYPE_FAMILY[cov_type], **tested)
    _assert_matches(result, reference, f"{cov_type}/{name}")


def test_the_statistic_is_the_difference_of_the_two_hansen_statistics(
    frame,
) -> None:
    result = _fit(frame, _ONE, estimator="gmm", setting="robust").c_stat(
        variables=["endog1"], instruments=["z1"]
    )
    assert result.df == 2
    _CLOSE(
        result.statistic,
        result.j_with_tested - result.j_without_tested,
        "C = J_e - J_c",
    )


def test_j_without_tested_is_zero_when_the_other_model_is_just_identified(
    frame,
) -> None:
    res = _fit(frame, _TWO, estimator="gmm", setting="robust")
    result = res.c_stat(instruments=["z3"])
    assert result.j_without_tested == 0.0
    assert result.statistic == result.j_with_tested
    # Z_fullは元の操作変数なので、J_eは`fit()`の過剰識別検定（Hansen J）と一致する。
    _CLOSE(result.j_with_tested, res.overid_statistic, "J_e = Hansen J")


# ── ライブ linearmodels との照合（シナリオ） ──────────────────────────

_MULTI_ENDOG_SCENARIOS = [
    "baseline",
    "heteroskedastic",
    "autocorrelated",
    "small_n",
    "weak_instruments",
    "just_identified",
]
# `iv_<name>.csv`の単一の内生変数`endog1`のシナリオ。
_SINGLE_ENDOG_SCENARIOS = [
    "baseline",
    "just_identified",
    "weak_instruments",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
    "high_condition_number",
    "scale_variance_mild",
    "invalid_instrument",
    "many_regressors",
    "outlier_regressor",
]


def _scenario_frame(name: str) -> pl.DataFrame:
    df = pl.read_csv(DATA_DIR / name)
    return with_row_time(
        df.with_columns((pl.int_range(pl.len()) % _N_GROUPS).alias(_CLUSTER))
    )


def _multi_endog_spec(scenario: str) -> dict:
    return {
        "x_exog": ["x1"],
        "x_endog": ["endog1", "endog2"],
        "instruments": (
            ["z1", "z2"]
            if scenario == "just_identified"
            else ["z1", "z2", "z3"]
        ),
    }


@pytest.mark.parametrize("weight", _GMM_WEIGHTS)
@pytest.mark.parametrize("scenario", _MULTI_ENDOG_SCENARIOS)
def test_multi_endog_scenarios_match_linearmodels(scenario, weight) -> None:
    df = _scenario_frame(f"iv_{scenario}_multi_endog.csv")
    spec = _multi_endog_spec(scenario)
    result = _fit(df, spec, estimator="gmm", setting=weight).c_stat(
        variables=["endog1"]
    )
    reference = _lm_c_stat(df, spec, weight, ["endog1"])
    _CLOSE(result.statistic, float(reference.stat), f"{scenario}/{weight}")
    _CLOSE(result.p_value, float(reference.pval), f"{scenario}/{weight}")


@pytest.mark.parametrize("weight", ["classical", "robust", "hac"])
@pytest.mark.parametrize("scenario", _SINGLE_ENDOG_SCENARIOS)
def test_single_endog_scenarios_match_linearmodels(scenario, weight) -> None:
    df = _scenario_frame(f"iv_{scenario}.csv")
    spec = {
        "x_exog": scenario_x_exog(scenario),
        "x_endog": ["endog1"],
        "instruments": scenario_instruments(scenario),
    }
    result = _fit(df, spec, estimator="gmm", setting=weight).c_stat(
        variables=["endog1"]
    )
    reference = _lm_c_stat(df, spec, weight, ["endog1"])
    _CLOSE(result.statistic, float(reference.stat), f"{scenario}/{weight}")
    _CLOSE(result.p_value, float(reference.pval), f"{scenario}/{weight}")


def test_an_invalid_instrument_is_rejected() -> None:
    """操作変数が誤差と相関するシナリオでは、直交性が強く棄却される。"""
    df = _scenario_frame("iv_invalid_instrument.csv")
    spec = {
        "x_exog": scenario_x_exog("invalid_instrument"),
        "x_endog": ["endog1"],
        "instruments": scenario_instruments("invalid_instrument"),
    }
    for weight in ("classical", "robust"):
        res = _fit(df, spec, estimator="gmm", setting=weight)
        # 外した後は丁度識別なので、C統計量は`fit()`の過剰識別検定に一致する。
        result = res.c_stat(instruments=["z1"])
        assert result.p_value < 1e-4
        _CLOSE(result.statistic, res.overid_statistic, weight)


# ── ライブ linearmodels との照合（クラスター・HAC・定数項） ───────────


def test_cluster_boundary_needs_as_many_clusters_as_moment_conditions(
    frame,
) -> None:
    # 検定する内生変数を操作変数に加えた側は、定数・x1・endog1・操作変数4本の7列。
    few = frame.with_columns((pl.int_range(pl.len()) % 6).alias(_CLUSTER))
    exact = frame.with_columns((pl.int_range(pl.len()) % 7).alias(_CLUSTER))
    # linearmodelsも、6クラスターでは無効な統計量を返し、7クラスターでは計算できる。
    assert isinstance(
        _lm_c_stat(few, _ONE, "cluster", ["endog1"]), InvalidTestStatistic
    )
    reference = _lm_c_stat(exact, _ONE, "cluster", ["endog1"])
    result = _fit(exact, _ONE, estimator="gmm", setting="cluster").c_stat(
        variables=["endog1"]
    )
    _CLOSE(result.statistic, float(reference.stat), "G = columns")
    res_few = _fit(few, _ONE, estimator="gmm", setting="cluster")
    with raises_error("C_STATISTIC_UNDEFINED"):
        res_few.c_stat(variables=["endog1"])
    # 検定する対象を減らせば、`fit()`が通る6クラスターでも計算できる（6列）。
    _assert_matches(
        res_few.c_stat(instruments=["z3"]),
        _rebuilt(few, _ONE, "cluster", instruments=["z3"]),
        "G = 6",
    )


@pytest.mark.parametrize("estimator", ["gmm", "2sls"])
def test_unbalanced_clusters_match_the_rebuilt_reference(
    frame, estimator
) -> None:
    result = _fit(
        frame,
        _ONE,
        estimator=estimator,
        setting="cluster",
        cluster=_IMBALANCED,
    ).c_stat(variables=["endog1"], instruments=["z3"])
    reference = _rebuilt(
        frame,
        _ONE,
        "cluster",
        variables=["endog1"],
        instruments=["z3"],
        cluster=_IMBALANCED,
    )
    _assert_matches(result, reference, f"{estimator}/unbalanced")


@pytest.mark.parametrize("weight", ["robust", "hac"])
def test_without_an_intercept_robust_and_hac_weights_match_linearmodels(
    frame, weight
) -> None:
    res = _fit(
        frame,
        _ONE,
        estimator="gmm",
        setting=weight,
        include_intercept=False,
    )
    reference = _lm_c_stat(frame, _ONE, weight, ["endog1"], constant=False)
    _CLOSE(
        res.c_stat(variables=["endog1"]).statistic,
        float(reference.stat),
        weight,
    )
    _assert_matches(
        res.c_stat(instruments=["z3"]),
        _rebuilt(frame, _ONE, weight, instruments=["z3"], constant=False),
        f"{weight}/no intercept",
    )


@pytest.mark.parametrize("lags", [1, 3, 5])
def test_hac_lags_other_than_two_match_the_rebuilt_reference(
    frame, lags
) -> None:
    result = _fit(
        frame, _ONE, estimator="gmm", setting="hac", hac_lags=lags
    ).c_stat(variables=["endog1"], instruments=["z3"])
    reference = _rebuilt(
        frame,
        _ONE,
        "hac",
        variables=["endog1"],
        instruments=["z3"],
        lags=lags,
    )
    _assert_matches(result, reference, f"lags={lags}")


def test_hac_with_automatic_lags_equals_the_explicit_automatic_lag(
    frame,
) -> None:
    auto = _fit(
        frame, _ONE, estimator="gmm", setting="hac", hac_lags=None
    ).c_stat(variables=["endog1"])
    explicit = _fit(
        frame,
        _ONE,
        estimator="gmm",
        setting="hac",
        hac_lags=hac_auto_lag(frame.height),
    ).c_stat(variables=["endog1"])
    _CLOSE(auto.statistic, explicit.statistic, "automatic lags")


def test_hac_follows_the_time_column_not_the_row_order(frame) -> None:
    order = list(range(frame.height))[::-1]
    reference = _fit(frame, _ONE, estimator="gmm", setting="hac").c_stat(
        variables=["endog1"]
    )
    shuffled = _fit(
        reorder_rows(frame, order), _ONE, estimator="gmm", setting="hac"
    ).c_stat(variables=["endog1"])
    _CLOSE(shuffled.statistic, reference.statistic, "reordered rows")


# ── ライブ linearmodels との照合（実データ: Wooldridge card） ─────────

_CARD = {
    "x_exog": ["exper", "expersq", "black", "south", "smsa", "smsa66"],
    "x_endog": ["educ"],
    "instruments": ["nearc2", "nearc4"],
}


@pytest.fixture(scope="module")
def card() -> pl.DataFrame:
    return with_row_time(with_card_region(load_wooldridge_dataset("card")))


@pytest.mark.parametrize("weight", ["classical", "robust"])
def test_card_endogeneity_of_education_matches_linearmodels(
    card, weight
) -> None:
    result = _fit(
        card, _CARD, estimator="gmm", setting=weight, y="lwage"
    ).c_stat(variables=["educ"])
    reference = _lm_c_stat(card, _CARD, weight, ["educ"], y="lwage")
    _CLOSE(result.statistic, float(reference.stat), weight)
    _CLOSE(result.p_value, float(reference.pval), weight)


@pytest.mark.parametrize("weight", ["classical", "robust"])
def test_card_orthogonality_of_one_instrument_matches_the_rebuilt_reference(
    card, weight
) -> None:
    result = _fit(
        card, _CARD, estimator="gmm", setting=weight, y="lwage"
    ).c_stat(instruments=["nearc2"])
    reference = _rebuilt(
        card, _CARD, weight, instruments=["nearc2"], y="lwage"
    )
    _assert_matches(result, reference, f"card/{weight}")
    # 外した後は丁度識別（内生変数1個に操作変数1個）なのでJ_cは0。
    assert result.j_without_tested == 0.0


def test_card_regions_as_unbalanced_clusters(card) -> None:
    res = _fit(
        card,
        _CARD,
        estimator="gmm",
        setting="cluster",
        y="lwage",
        cluster="region",
    )
    reference = _rebuilt(
        card,
        _CARD,
        "cluster",
        instruments=["nearc2"],
        y="lwage",
        cluster="region",
    )
    _assert_matches(
        res.c_stat(instruments=["nearc2"]),
        reference,
        "card/region",
        rtol=_RTOL_BOUNDARY,
    )
    # 地域は9グループで、内生変数を加えると10列になるため定義できない。
    with raises_error("C_STATISTIC_UNDEFINED"):
        res.c_stat(variables=["educ"])


# ── 推定量・オプションへの依存 ────────────────────────────────────────


@pytest.mark.parametrize("weight", ["robust", "cluster", "hac"])
@pytest.mark.parametrize(
    "extra",
    [
        {"gmm_type": "two_step"},
        # 収束しない重みもある（収束しなくても統計量は`gmm_type`に依存しない）。
        {"gmm_type": "iterated", "raise_on_non_convergence": False},
        {
            "gmm_type": "iterated",
            "gmm_max_iter": 5,
            "gmm_tol": 1e-3,
            "raise_on_non_convergence": False,
        },
    ],
    ids=["two_step", "iterated", "iterated/few"],
)
def test_the_statistic_does_not_depend_on_gmm_type(
    frame, weight, extra
) -> None:
    reference = _fit(frame, _ONE, estimator="gmm", setting=weight).c_stat(
        instruments=["z3"]
    )
    result = _fit(
        frame, _ONE, estimator="gmm", setting=weight, **extra
    ).c_stat(instruments=["z3"])
    _CLOSE(result.statistic, reference.statistic, weight)


def _cov_type_extra(cov_type: str) -> dict:
    extra: dict = {"cov_type": cov_type}
    if cov_type == "cluster":
        extra["cluster"] = _CLUSTER
    if cov_type == "hac":
        extra.update(hac_lags=_LAGS, hac_time=ROW_TIME)
    return extra


@pytest.mark.parametrize("cov_type", ["classical", "hc1", "cluster", "hac"])
def test_a_gmm_fit_ignores_its_cov_type(frame, cov_type) -> None:
    reference = _fit(frame, _ONE, estimator="gmm", setting="robust").c_stat(
        instruments=["z3"]
    )
    result = IV(
        frame,
        y="y",
        options=IVOptions(
            estimator="gmm",
            gmm_weight_type="robust",
            **_cov_type_extra(cov_type),
        ),
        **_ONE,
    ).fit()
    _CLOSE(
        result.c_stat(instruments=["z3"]).statistic,
        reference.statistic,
        cov_type,
    )


@pytest.mark.parametrize("cov_type", ["hc1", "cluster", "hac"])
def test_a_one_step_gmm_fit_follows_its_cov_type_like_2sls(
    frame, cov_type
) -> None:
    one_step = IV(
        frame,
        y="y",
        options=IVOptions(
            estimator="gmm", gmm_type="one_step", **_cov_type_extra(cov_type)
        ),
        **_ONE,
    ).fit()
    reference = _fit(frame, _ONE, setting=cov_type)
    _CLOSE(
        one_step.c_stat(instruments=["z3"]).statistic,
        reference.c_stat(instruments=["z3"]).statistic,
        cov_type,
    )


def test_2sls_with_the_classical_covariance_equals_gmm_with_the_classical_weight(
    frame,
) -> None:
    sls = _fit(frame, _ONE).c_stat(instruments=["z3"])
    gmm = _fit(frame, _ONE, estimator="gmm", setting="classical").c_stat(
        instruments=["z3"]
    )
    _CLOSE(sls.statistic, gmm.statistic, "classical")


def test_the_order_of_the_names_does_not_matter(frame) -> None:
    res = _fit(frame, _ONE, estimator="gmm", setting="robust")
    forward = res.c_stat(instruments=["z1", "z3"])
    backward = res.c_stat(instruments=["z3", "z1"])
    _CLOSE(forward.statistic, backward.statistic, "instruments")
    assert backward.instruments == ["z1", "z3"]
    two = _fit(frame, _TWO, estimator="gmm", setting="robust")
    swapped = two.c_stat(variables=["endog2", "endog1"])
    _CLOSE(
        swapped.statistic,
        two.c_stat(variables=["endog1", "endog2"]).statistic,
        "variables",
    )
    assert swapped.variables == ["endog1", "endog2"]


# ── 結果の構造 ────────────────────────────────────────────────────────


def test_the_result_is_a_diagnostic_result_with_a_chi_squared_distribution(
    frame,
) -> None:
    result = _fit(frame, _ONE, estimator="gmm", setting="robust").c_stat(
        variables=["endog1"]
    )
    assert isinstance(result, DiagnosticResult)
    assert result.distribution == "chi2"
    assert result.df_denom is None
    assert result.statistic >= 0.0
    assert 0.0 <= result.p_value <= 1.0
    with pytest.raises(AttributeError):
        result.statistic = 0.0  # type: ignore[misc]


def test_to_dict_is_json_ready(frame) -> None:
    result = _fit(frame, _ONE, estimator="gmm", setting="robust").c_stat(
        variables=["endog1"], instruments=["z3"]
    )
    as_dict = result.to_dict()
    assert json.loads(json.dumps(as_dict)) == as_dict
    assert as_dict["variables"] == ["endog1"]
    assert as_dict["instruments"] == ["z3"]
    assert set(as_dict) == {
        "statistic",
        "p_value",
        "df",
        "df_denom",
        "distribution",
        "variables",
        "instruments",
        "j_with_tested",
        "j_without_tested",
    }


def test_the_method_survives_pickling_and_copying(frame) -> None:
    res = _fit(frame, _ONE, estimator="gmm", setting="robust")
    expected = res.c_stat(instruments=["z3"]).statistic
    for restored in (pickle.loads(pickle.dumps(res)), copy.deepcopy(res)):
        _CLOSE(restored.c_stat(instruments=["z3"]).statistic, expected, "copy")


def test_the_method_does_not_modify_the_result(frame) -> None:
    res = _fit(frame, _ONE, estimator="gmm", setting="robust")
    before = (res.params, res.std_errors, res.overid_statistic)
    first = res.c_stat(variables=["endog1"])
    assert (res.params, res.std_errors, res.overid_statistic) == before
    assert res.c_stat(variables=["endog1"]) == first


# ── ValidationError・ComputationError ─────────────────────────────────


def _result(frame):
    return _fit(frame, _ONE, estimator="gmm", setting="robust")


def test_nothing_to_test_is_a_validation_error(frame) -> None:
    with raises_error("C_STATISTIC_NOTHING_TO_TEST"):
        _result(frame).c_stat()


@pytest.mark.parametrize("argument", ["variables", "instruments"])
def test_an_empty_list_is_a_validation_error(frame, argument) -> None:
    with raises_error("ROLE_EMPTY", argument=argument):
        _result(frame).c_stat(**{argument: []})


@pytest.mark.parametrize(
    ("argument", "name"), [("variables", "endog1"), ("instruments", "z1")]
)
def test_a_repeated_name_is_a_validation_error(frame, argument, name) -> None:
    with raises_error("ROLE_DUPLICATE_COLUMN", column=name, argument=argument):
        _result(frame).c_stat(**{argument: [name, name]})


@pytest.mark.parametrize(
    ("argument", "name", "role"),
    [
        ("variables", "x1", "x_endog"),
        ("variables", "z1", "x_endog"),
        ("variables", "nope", "x_endog"),
        ("instruments", "endog1", "instruments"),
        ("instruments", "x1", "instruments"),
    ],
)
def test_a_name_that_is_not_in_the_model_is_a_validation_error(
    frame, argument, name, role
) -> None:
    with raises_error(
        "C_STATISTIC_UNKNOWN_VARIABLE",
        name=name,
        argument=argument,
        role=role,
    ):
        _result(frame).c_stat(**{argument: [name]})


def test_testing_too_many_instruments_is_a_validation_error(frame) -> None:
    res = _fit(frame, _TWO, estimator="gmm", setting="robust")
    expected = {"n_tested": 2, "n_instruments": 3, "n_endog": 2}
    with raises_error("C_STATISTIC_TOO_MANY_TESTED_INSTRUMENTS", **expected):
        res.c_stat(instruments=["z1", "z2"])
    # 内生変数の検定を同時に指定しても、外した操作変数の数だけで決まる。
    with raises_error("C_STATISTIC_TOO_MANY_TESTED_INSTRUMENTS", **expected):
        res.c_stat(variables=["endog1"], instruments=["z1", "z2"])


def test_a_just_identified_model_cannot_test_any_instrument(frame) -> None:
    just = _fit(
        frame,
        {**_TWO, "instruments": ["z1", "z2"]},
        estimator="gmm",
        setting="robust",
    )
    with raises_error(
        "C_STATISTIC_TOO_MANY_TESTED_INSTRUMENTS",
        n_tested=1,
        n_instruments=2,
        n_endog=2,
    ):
        just.c_stat(instruments=["z1"])
    # 内生変数の外生性は検定できる（J_cは0でC = J_e）。
    assert just.c_stat(variables=["endog1"]).j_without_tested == 0.0


@pytest.mark.parametrize("argument", ["variables", "instruments"])
def test_a_value_that_is_not_a_list_of_names_is_a_type_error(
    frame, argument
) -> None:
    res = _result(frame)
    for bad in ("endog1", ("endog1",), 3):
        message = escaped(
            COLUMN_LIST_NOT_LIST,
            name=argument,
            type_name=fully_qualified_type_name(bad),
        )
        with pytest.raises(TypeError, match=f"^{message}$"):
            res.c_stat(**{argument: bad})
    message = escaped(
        COLUMN_LIST_ELEMENT_NOT_STR,
        name=argument,
        index=0,
        type_name=fully_qualified_type_name(1),
    )
    with pytest.raises(TypeError, match=f"^{message}$"):
        res.c_stat(**{argument: [1]})


def test_the_types_are_checked_before_the_values(frame) -> None:
    """型の誤りは、もう一方の引数の値の誤り（空の一覧・重複）より先に報告される。"""
    res = _result(frame)
    with pytest.raises(TypeError):
        res.c_stat(variables=[], instruments=5)
    with pytest.raises(TypeError):
        res.c_stat(variables=["endog1", "endog1"], instruments="z1")


def test_arguments_are_keyword_only(frame) -> None:
    with pytest.raises(TypeError):
        _result(frame).c_stat(["endog1"])  # type: ignore[misc]
