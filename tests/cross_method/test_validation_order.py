"""全手法共通の、`ValidationError`が報告される順序のテスト。

誤りが2つ同時にある入力で、どちらのエラーが先に出るかを固定する。順序は次の段階で
決まり、先の段階の誤りが常に後の段階の誤りより先に報告される
（`docs/guide/validation.md`「Order in which errors are reported」）。

1. 型の誤り（`TypeError`。ここでは扱わない）
2. オプションの値: `estimator` → `cov_type` → `gmm_type` → `gmm_weight_type` →
   `solver` → `variance_method`。使われないオプションの未知の値も報告する
3. オプションの組合せ: 使われないオプション → 必須オプションの欠落
4. ロールの構造（列名の重複など）
5. `y`・`x`の列（存在・dtype・欠損値）
6. オプションの列（`cluster`・`hac_time`・`dk_time`）
7. エンジンの検査（`confidence_level`の範囲など）

故障を1つずつ`FAULTS`に定義し、手法ごとに「その手法に当てはまる故障」を総当たりで2つ
組み合わせる。段階が違う（同じ段階の故障は順序を決めない）組について、段階が早い
故障のエラーが出ることを確かめる。
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import polars as pl
import pytest
from _dtype_helpers import make_frame
from econometricsmodels import (
    FE,
    IV,
    OLS,
    RE,
    WLS,
    FEOptions,
    IVOptions,
    Logit,
    LogitOptions,
    OLSOptions,
    Probit,
    ProbitOptions,
    REOptions,
    Tobit,
    TobitOptions,
    ValidationError,
    WLSOptions,
)

Changes = tuple[dict[str, Any], dict[str, Any]]


@dataclass(frozen=True)
class Method:
    """手法ごとの呼び出し方と、当てはまる故障の違い。"""

    name: str
    # `(data, args, options_kwargs)`から`fit()`を呼ぶ。
    run: Callable[[pl.DataFrame, dict[str, Any], dict[str, Any]], object]
    base_args: dict[str, Any]
    # 当てはまる故障のうち、手法ごとに内容が変わるもの（時間順の列を取らない手法は
    # `None`）。
    time_cov_type: str | None = None
    time_option: str | None = None
    x_role: str = "x"
    # この手法が持つオプションの値の故障（`unknown_*`）。
    value_faults: tuple[str, ...] = ("unknown_cov_type",)
    extra_faults: tuple[str, ...] = field(default_factory=tuple)


def _simple(model_cls, options_cls):
    def run(df, args, opts):
        return model_cls(df, options=options_cls(**opts), **args).fit()

    return run


METHODS = [
    Method(
        "ols",
        _simple(OLS, OLSOptions),
        {"y": "y", "x": ["x1", "x2"]},
        time_cov_type="hac",
        time_option="hac_time",
        extra_faults=("unused_cluster",),
    ),
    Method(
        "wls",
        _simple(WLS, WLSOptions),
        {"y": "y", "x": ["x1", "x2"], "weight": "w"},
        time_cov_type="hac",
        time_option="hac_time",
        extra_faults=("unused_cluster",),
    ),
    Method(
        "iv",
        _simple(IV, IVOptions),
        {
            "y": "y",
            "x_exog": ["x2"],
            "x_endog": ["x1"],
            "instruments": ["z", "z2"],
        },
        time_cov_type="hac",
        time_option="hac_time",
        x_role="x_exog",
        value_faults=(
            "unknown_estimator",
            "unknown_cov_type",
            "unknown_gmm_type",
            "unknown_gmm_weight_type",
        ),
        extra_faults=("unused_cluster", "unused_gmm_max_iter"),
    ),
    Method(
        "logit",
        _simple(Logit, LogitOptions),
        {"y": "y01", "x": ["x1", "x2"]},
        value_faults=("unknown_cov_type", "unknown_solver"),
        extra_faults=("unused_cluster",),
    ),
    Method(
        "probit",
        _simple(Probit, ProbitOptions),
        {"y": "y01", "x": ["x1", "x2"]},
        value_faults=("unknown_cov_type", "unknown_solver"),
        extra_faults=("unused_cluster",),
    ),
    Method(
        "tobit",
        _simple(Tobit, TobitOptions),
        {"y": "yc", "x": ["x1", "x2"]},
        value_faults=("unknown_cov_type", "unknown_solver"),
        extra_faults=("unused_cluster",),
    ),
    Method(
        "fe",
        _simple(FE, FEOptions),
        {"y": "y", "x": ["x1", "x2"], "entity": "entity"},
        time_cov_type="dk",
        time_option="dk_time",
        extra_faults=("unused_dk_bandwidth",),
    ),
    Method(
        "re",
        _simple(RE, REOptions),
        {"y": "y", "x": ["x1", "x2"], "entity": "entity"},
        time_cov_type="dk",
        time_option="dk_time",
        value_faults=("unknown_cov_type", "unknown_variance_method"),
        extra_faults=("unused_dk_bandwidth",),
    ),
]


@dataclass(frozen=True)
class Fault:
    """入力に混ぜる誤り1つ。`rank`が小さいほど先に報告される。"""

    id: str
    rank: tuple[int, int]
    marker: str


# 段階2の第2要素は、仕様の報告順（estimator → cov_type → gmm_type →
# gmm_weight_type → solver → variance_method）。段階3は使われないオプション(0) →
# 必須オプションの欠落(1)。
FAULTS = {
    f.id: f
    for f in [
        Fault("unknown_estimator", (2, 0), r"unknown estimator"),
        Fault("unknown_cov_type", (2, 1), r"unknown cov_type"),
        Fault("unknown_gmm_type", (2, 2), r"unknown gmm_type"),
        Fault("unknown_gmm_weight_type", (2, 3), r"unknown gmm_weight_type"),
        Fault("unknown_solver", (2, 4), r"unknown solver"),
        Fault("unknown_variance_method", (2, 5), r"unknown variance_method"),
        Fault("unused_cluster", (3, 0), r"cluster is only used with"),
        Fault("unused_dk_bandwidth", (3, 0), r"dk_bandwidth is only used"),
        Fault("unused_gmm_max_iter", (3, 0), r"gmm_max_iter is only used"),
        Fault(
            "missing_time_option",
            (3, 1),
            r"requires the `(hac_time|dk_time)` option",
        ),
        Fault(
            "duplicate_x",
            (4, 0),
            r"column '(x1|x2)' is specified more than once in x(_exog)?",
        ),
        Fault("missing_y", (5, 0), r"column 'no_such_y' does not exist"),
        Fault(
            "missing_cluster_column",
            (6, 0),
            r"column 'no_such_cluster' does not exist",
        ),
        Fault(
            "missing_time_column",
            (6, 1),
            r"column 'no_such_time' does not exist",
        ),
        Fault(
            "bad_confidence_level",
            (7, 0),
            r"confidence_level must be in the range",
        ),
    ]
}


def fault_ids(method: Method) -> list[str]:
    """`method`に当てはまる故障の`id`。"""
    ids = list(method.value_faults)
    ids += method.extra_faults
    ids += ["duplicate_x", "missing_y", "missing_cluster_column"]
    if method.time_cov_type is not None:
        ids += ["missing_time_option", "missing_time_column"]
    ids.append("bad_confidence_level")
    return ids


def changes(method: Method, fault_id: str) -> Changes:
    """故障を入力に混ぜるための、`(引数の差分, オプションの差分)`。"""
    match fault_id:
        case "unknown_estimator":
            return {}, {"estimator": "bogus"}
        case "unknown_cov_type":
            return {}, {"cov_type": "bogus"}
        case "unknown_gmm_type":
            return {}, {"gmm_type": "bogus"}
        case "unknown_gmm_weight_type":
            return {}, {"gmm_weight_type": "bogus"}
        case "unknown_solver":
            return {}, {"solver": "bogus"}
        case "unknown_variance_method":
            return {}, {"variance_method": "bogus"}
        case "unused_cluster":
            # 既定の`cov_type`が`cluster`でない手法だけ（FE/REは`unused_dk_bandwidth`）。
            return {}, {"cluster": "cluster"}
        case "unused_dk_bandwidth":
            return {}, {"dk_bandwidth": 2}
        case "unused_gmm_max_iter":
            return {}, {"gmm_max_iter": 10}
        case "missing_time_option":
            return {}, {"cov_type": method.time_cov_type}
        case "duplicate_x":
            name = method.base_args[method.x_role][0]
            return {method.x_role: [name, name]}, {}
        case "missing_y":
            return {"y": "no_such_y"}, {}
        case "missing_cluster_column":
            return {}, {"cov_type": "cluster", "cluster": "no_such_cluster"}
        case "missing_time_column":
            return {}, {
                "cov_type": method.time_cov_type,
                method.time_option: "no_such_time",
            }
        case "bad_confidence_level":
            return {}, {"confidence_level": 2.0}
    raise AssertionError(fault_id)


def compose(
    method: Method, early: str, late: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """2つの故障を混ぜた`(args, options)`。混ぜられない組は`None`。

    同じキーに別の値を入れる故障同士は、先に報告される側が段階2（オプションの値）の
    ときだけ、先の側の値を優先して混ぜる（未知の値は他の設定の意味を変えない）。
    後の側のキーがすべて上書きされて故障が残らない組と、それ以外の衝突は省く。
    """
    e_args, e_opts = changes(method, early)
    l_args, l_opts = changes(method, late)
    conflicts = {
        ("args", k) for k in e_args if k in l_args and e_args[k] != l_args[k]
    } | {("opts", k) for k in e_opts if k in l_opts and e_opts[k] != l_opts[k]}
    if conflicts:
        late_keys = {("args", k) for k in l_args} | {
            ("opts", k) for k in l_opts
        }
        if FAULTS[early].rank[0] != 2 or late_keys <= conflicts:
            return None
    args = {**method.base_args, **l_args, **e_args}
    opts = {**l_opts, **e_opts}
    return args, opts


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    return make_frame(11)


def _method_params():
    return [pytest.param(m, id=m.name) for m in METHODS]


def _pair_params():
    params = []
    for m in METHODS:
        for a, b in itertools.permutations(fault_ids(m), 2):
            if FAULTS[a].rank >= FAULTS[b].rank:
                continue
            if compose(m, a, b) is None:
                continue
            params.append(pytest.param(m, a, b, id=f"{m.name}-{a}-before-{b}"))
    return params


def _single_params():
    return [
        pytest.param(m, f, id=f"{m.name}-{f}")
        for m in METHODS
        for f in fault_ids(m)
    ]


@pytest.mark.parametrize("method", _method_params())
def test_baseline_input_is_valid(df, method):
    """故障を混ぜる前の入力は、そのまま推定できる（以降のテストの前提）。"""
    assert method.run(df, dict(method.base_args), {}) is not None


@pytest.mark.parametrize(("method", "fault"), _single_params())
def test_each_fault_alone_raises_its_own_error(df, method, fault):
    """故障を1つだけ混ぜると、その故障のエラーが出る（組合せテストの前提）。"""
    args_changes, opts = changes(method, fault)
    args = {**method.base_args, **args_changes}
    with pytest.raises(ValidationError, match=FAULTS[fault].marker):
        method.run(df, args, opts)


@pytest.mark.parametrize(("method", "early", "late"), _pair_params())
def test_earlier_stage_error_is_reported_first(df, method, early, late):
    """誤りが2つあると、仕様上先の段階のエラーが報告される。"""
    composed = compose(method, early, late)
    assert composed is not None
    args, opts = composed
    with pytest.raises(ValidationError, match=FAULTS[early].marker):
        method.run(df, args, opts)


def test_pairs_cover_every_stage_boundary():
    """組合せが、段階2〜7の隣り合う境界をすべて含む（故障の定義を変えたときの保険）。"""
    stages = {
        (FAULTS[a].rank[0], FAULTS[b].rank[0])
        for _, a, b in (p.values for p in _pair_params())
    }
    for boundary in [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]:
        assert boundary in stages, boundary
