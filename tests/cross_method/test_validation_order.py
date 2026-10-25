"""全手法共通の、`ValidationError`が報告される順序のテスト。

誤りが2つ同時にある入力で、どちらのエラーが先に出るかを固定する。順序は次の段階で
決まり、先の段階の誤りが常に後の段階の誤りより先に報告される
（`docs/guide/validation.md`「Order in which errors are reported」）。

1. 型の誤り（`TypeError`。ここでは扱わない）
2. オプションの値: `estimator` → `cov_type` → `gmm_type` → `gmm_weight_type` →
   `solver` → `variance_method`。使われないオプションの未知の値も報告する
3. オプションの組合せ: 使われないオプション → 必須オプションの欠落
   （`cov_type`の`hac`/`dk`が先、`gmm_weight_type`の`hac`が後）
4. ロールの構造（列名の重複など）
5. `y`・`x`などモデルの列（存在・dtype・欠損値）
6. オプションの列（`cluster`・`hac_time`・`dk_time`・GMMの重み行列の列）
7. エンジンの検査（`confidence_level`の範囲など）

故障を1つずつ`FAULTS`に定義し、手法ごとに「その手法に当てはまる故障」を総当たりで
2つ組み合わせる。順位（段階と段階内の順）が違う組について、順位が早い故障のエラーが
出ることを確かめる。2つの故障が互いの成立を打ち消す組（片方が`cov_type`を書き換えて、
もう片方の故障が「使われている」状態になる等）と、同じキーに別の値を入れる組は、
組合せから除く（`compose`）。
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import polars as pl
import pytest
from _dtype_helpers import make_frame
from _errors import raises_error
from econometricsmodels import (
    FE,
    IV,
    OLS,
    RE,
    WLS,
    ErrorCode,
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

# ── 手法 ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Method:
    """手法ごとの呼び出し方と、当てはまる故障の違い。"""

    name: str
    # `(data, args, options_kwargs)`から`fit()`を呼ぶ。
    run: Callable[[pl.DataFrame, dict[str, Any], dict[str, Any]], object]
    base_args: dict[str, Any]
    # この手法に当てはまる故障の`id`。
    faults: tuple[str, ...]
    # 故障を混ぜる前から指定しているオプション（IVのGMMの`estimator`等）。
    base_opts: dict[str, Any] = field(default_factory=dict)
    # 時間順の列を取る`cov_type`（OLS/WLS/IVは`hac`、FE/REは`dk`）とそのオプション名。
    time_cov_type: str | None = None
    time_option: str | None = None
    # `x`にあたるロール名（IVは外生変数の`x_exog`）。
    x_role: str = "x"


def _simple(model_cls, options_cls):
    def run(df, args, opts):
        return model_cls(df, options=options_cls(**opts), **args).fit()

    return run


_X = ["x1", "x2"]
_LINEAR_FAULTS = (
    "uint128_y",
    "uint128_x",
    "uint128_cluster",
    "unknown_cov_type",
    "unused_cluster",
    "unused_hac_lags",
    "missing_time_option",
    "duplicate_x",
    "y_in_x",
    "const_in_x",
    "empty_x",
    "missing_y",
    "missing_x",
    "missing_cluster_column",
    "missing_time_column",
    "bad_confidence_level",
)
_NONLINEAR_FAULTS = (
    "uint128_y",
    "uint128_x",
    "uint128_cluster",
    "unknown_cov_type",
    "unknown_solver",
    "unused_cluster",
    "duplicate_x",
    "y_in_x",
    "const_in_x",
    "empty_x",
    "missing_y",
    "missing_x",
    "missing_cluster_column",
    "bad_confidence_level",
    "bad_max_iter",
)
_IV_FAULTS = (
    "uint128_y",
    "uint128_x",
    "uint128_cluster",
    "unknown_estimator",
    "unknown_cov_type",
    "unknown_gmm_type",
    "unknown_gmm_weight_type",
    "unused_cluster",
    "unused_hac_lags",
    "unused_gmm_max_iter",
    "unused_gmm_tol",
    "missing_time_option",
    "duplicate_x",
    "y_in_x",
    "const_in_x",
    "empty_x",
    "missing_y",
    "missing_x",
    "missing_instrument",
    "missing_cluster_column",
    "missing_time_column",
    "bad_confidence_level",
)
_PANEL_FAULTS = (
    "uint128_y",
    "uint128_x",
    "uint128_cluster",
    "unknown_cov_type",
    "unused_cluster",
    "unused_dk_bandwidth",
    "unused_dk_time",
    "missing_time_option",
    "duplicate_x",
    "y_in_x",
    "empty_x",
    "time_is_y",
    "dk_time_is_y",
    "missing_y",
    "missing_x",
    "missing_entity",
    "missing_time",
    "missing_cluster_column",
    "missing_time_column",
    "bad_confidence_level",
)
_IV_ARGS = {
    "y": "y",
    "x_exog": ["x2"],
    "x_endog": ["x1"],
    "instruments": ["z", "z2"],
}
_PANEL_ARGS = {"y": "y", "x": _X, "entity": "entity"}

METHODS = [
    Method(
        "ols",
        _simple(OLS, OLSOptions),
        {"y": "y", "x": _X},
        _LINEAR_FAULTS,
        time_cov_type="hac",
        time_option="hac_time",
    ),
    Method(
        "wls",
        _simple(WLS, WLSOptions),
        {"y": "y", "x": _X, "weight": "w"},
        (*_LINEAR_FAULTS, "weight_is_y", "missing_weight"),
        time_cov_type="hac",
        time_option="hac_time",
    ),
    Method(
        "logit",
        _simple(Logit, LogitOptions),
        {"y": "y01", "x": _X},
        (*_NONLINEAR_FAULTS, "non_binary_y"),
    ),
    Method(
        "probit",
        _simple(Probit, ProbitOptions),
        {"y": "y01", "x": _X},
        (*_NONLINEAR_FAULTS, "non_binary_y"),
    ),
    Method(
        "tobit",
        _simple(Tobit, TobitOptions),
        {"y": "yc", "x": _X},
        (*_NONLINEAR_FAULTS, "sigma_in_x", "bad_censoring_bounds"),
    ),
    Method(
        "iv",
        _simple(IV, IVOptions),
        _IV_ARGS,
        _IV_FAULTS,
        time_cov_type="hac",
        time_option="hac_time",
        x_role="x_exog",
    ),
    Method(
        "iv_gmm",
        _simple(IV, IVOptions),
        _IV_ARGS,
        (
            *_IV_FAULTS,
            "unused_gmm_weight_type",
            "missing_weight_time_option",
            "missing_weight_cluster_column",
        ),
        base_opts={"estimator": "gmm"},
        time_cov_type="hac",
        time_option="hac_time",
        x_role="x_exog",
    ),
    Method(
        "fe",
        _simple(FE, FEOptions),
        _PANEL_ARGS,
        _PANEL_FAULTS,
        time_cov_type="dk",
        time_option="dk_time",
    ),
    Method(
        "re",
        _simple(RE, REOptions),
        _PANEL_ARGS,
        (*_PANEL_FAULTS, "unknown_variance_method"),
        time_cov_type="dk",
        time_option="dk_time",
    ),
]

# ── 故障 ─────────────────────────────────────────────────────────────


def _arg_role(method: Method) -> str:
    """`empty_x`が指摘する、空のリストを渡した引数（IVは`x_endog`）。"""
    return "x_endog" if method.name.startswith("iv") else "x"


def _dup_x(method: Method) -> dict[str, Any]:
    return {
        "column": method.base_args[method.x_role][0],
        "argument": method.x_role,
    }


def _y_in_x(method: Method) -> dict[str, Any]:
    return {
        "column": method.base_args["y"],
        "argument": "y",
        "other_argument": method.x_role,
    }


def _time_cov_code(method: Method) -> ErrorCode:
    """`hac`は`HAC_TIME_REQUIRED`、`dk`は`DK_TIME_REQUIRED`。"""
    if method.time_cov_type == "dk":
        return ErrorCode.DK_TIME_REQUIRED
    return ErrorCode.HAC_TIME_REQUIRED


def _time_cov_params(method: Method) -> dict[str, Any]:
    if method.time_cov_type == "dk":
        return {}
    return {"setting": "cov_type"}


def _unknown(option: str) -> dict[str, Any]:
    # 入力は常に`bogus`。受け付ける値の一覧（`allowed`）は手法ごとに違い、
    # 各手法の`*_validation.py`が確かめる。
    return {"option": option, "value": "bogus"}


@dataclass(frozen=True)
class Fault:
    """入力に混ぜる誤り1つ。`rank`が小さいほど先に報告される。

    エラーは次のいずれかで照合する。

    - `code`と`params`（構造化されたエラー。`code`は手法によって変わるなら
      `Method`から`ErrorCode`を返す関数、`params`は`Method`から`dict`を返す
      関数でもよい。渡したキーだけ比較する）
    - `column`（と`argument`）: `COLUMN_NOT_FOUND`
    - `marker`: まだ構造化していないエラーの文面（正規表現）
    """

    id: str
    rank: tuple[int, int]
    marker: str = ""
    # 列が存在しない故障は、文面ではなく`COLUMN_NOT_FOUND`の`column`（と、その列を
    # 渡した引数`argument`。手法によって引数名が変わる故障は`None`）で照合する。
    column: str | None = None
    argument: str | None = None
    code: ErrorCode | Callable[[Method], ErrorCode] | None = None
    params: dict[str, Any] | Callable[[Method], dict[str, Any]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        # 文面も列もコードも指定しない故障は、`match=""`で何にでも一致して検査が空になる。
        if not self.marker and self.column is None and self.code is None:
            raise ValueError(
                f"fault {self.id!r} needs a marker, a column or a code"
            )


# 順位の第1要素が段階、第2要素が段階内の順。段階2の第2要素は仕様の報告順
# （estimator → cov_type → gmm_type → gmm_weight_type → solver → variance_method）。
# 段階3は使われないオプション(0) → `cov_type`の必須オプションの欠落(1) →
# `gmm_weight_type`の必須オプションの欠落(2)。段階6は`cov_type`の列(0, 1)が先、
# GMMの重み行列の列(2)が後。同じ順位の故障は、組にしない。
FAULTS = {
    f.id: f
    for f in [
        Fault(
            "unknown_estimator",
            (2, 0),
            code=ErrorCode.OPTION_UNKNOWN_VALUE,
            params=_unknown("estimator"),
        ),
        Fault(
            "unknown_cov_type",
            (2, 1),
            code=ErrorCode.OPTION_UNKNOWN_VALUE,
            params=_unknown("cov_type"),
        ),
        Fault(
            "unknown_gmm_type",
            (2, 2),
            code=ErrorCode.OPTION_UNKNOWN_VALUE,
            params=_unknown("gmm_type"),
        ),
        Fault(
            "unknown_gmm_weight_type",
            (2, 3),
            code=ErrorCode.OPTION_UNKNOWN_VALUE,
            params=_unknown("gmm_weight_type"),
        ),
        Fault(
            "unknown_solver",
            (2, 4),
            code=ErrorCode.OPTION_UNKNOWN_VALUE,
            params=_unknown("solver"),
        ),
        Fault(
            "unknown_variance_method",
            (2, 5),
            code=ErrorCode.OPTION_UNKNOWN_VALUE,
            params=_unknown("variance_method"),
        ),
        Fault(
            "unused_cluster",
            (3, 0),
            code=ErrorCode.OPTION_UNUSED,
            params={"option": "cluster"},
        ),
        Fault(
            "unused_hac_lags",
            (3, 0),
            code=ErrorCode.OPTION_UNUSED,
            params={"option": "hac_lags"},
        ),
        Fault(
            "unused_dk_bandwidth",
            (3, 0),
            code=ErrorCode.OPTION_UNUSED,
            params={"option": "dk_bandwidth"},
        ),
        Fault(
            "unused_dk_time",
            (3, 0),
            code=ErrorCode.OPTION_UNUSED,
            params={"option": "dk_time"},
        ),
        Fault(
            "unused_gmm_max_iter",
            (3, 0),
            code=ErrorCode.OPTION_UNUSED,
            params={"option": "gmm_max_iter"},
        ),
        Fault(
            "unused_gmm_tol",
            (3, 0),
            code=ErrorCode.OPTION_UNUSED,
            params={"option": "gmm_tol"},
        ),
        Fault(
            "unused_gmm_weight_type",
            (3, 0),
            code=ErrorCode.OPTION_UNUSED,
            params={"option": "gmm_weight_type"},
        ),
        Fault(
            "missing_time_option",
            (3, 1),
            code=_time_cov_code,
            params=_time_cov_params,
        ),
        Fault(
            "missing_weight_time_option",
            (3, 2),
            code=ErrorCode.HAC_TIME_REQUIRED,
            params={"setting": "gmm_weight_type"},
        ),
        Fault(
            "duplicate_x",
            (4, 0),
            code=ErrorCode.ROLE_DUPLICATE_COLUMN,
            params=_dup_x,
        ),
        Fault(
            "y_in_x",
            (4, 0),
            code=ErrorCode.ROLE_COLUMN_OVERLAP,
            params=_y_in_x,
        ),
        Fault(
            "const_in_x",
            (4, 0),
            code=ErrorCode.INTERCEPT_NAME_COLLISION,
            params=lambda m: {"argument": m.x_role},
        ),
        Fault(
            "empty_x",
            (4, 0),
            code=ErrorCode.ROLE_EMPTY,
            params=lambda m: {"argument": _arg_role(m)},
        ),
        Fault(
            "weight_is_y",
            (4, 0),
            code=ErrorCode.ROLE_COLUMN_OVERLAP,
            params={
                "column": "y",
                "argument": "weight",
                "other_argument": "y",
            },
        ),
        Fault("sigma_in_x", (4, 0), code=ErrorCode.TOBIT_SIGMA_NAME_COLLISION),
        Fault(
            "time_is_y",
            (4, 0),
            code=ErrorCode.ROLE_COLUMN_OVERLAP,
            params={
                "column": "y",
                "argument": "time",
                "other_argument": "y",
            },
        ),
        Fault(
            "dk_time_is_y",
            (4, 0),
            code=ErrorCode.ROLE_COLUMN_OVERLAP,
            params={
                "column": "y",
                "argument": "dk_time",
                "other_argument": "y",
            },
        ),
        Fault(
            "uint128_y",
            (5, 0),
            code=ErrorCode.COLUMN_UNSUPPORTED_DTYPE,
            params={"column": "u128", "dtype": "UInt128", "argument": "y"},
        ),
        Fault(
            "uint128_x",
            (5, 0),
            code=ErrorCode.COLUMN_UNSUPPORTED_DTYPE,
            params=lambda m: {
                "column": "u128",
                "dtype": "UInt128",
                "argument": m.x_role,
            },
        ),
        Fault("missing_y", (5, 0), column="no_such_y", argument="y"),
        Fault("missing_x", (5, 0), column="no_such_x"),
        Fault(
            "missing_weight",
            (5, 0),
            column="no_such_weight",
            argument="weight",
        ),
        Fault(
            "missing_instrument",
            (5, 0),
            column="no_such_z",
            argument="instruments",
        ),
        Fault("missing_entity", (5, 0), column="no_such_e", argument="entity"),
        Fault("missing_time", (5, 0), column="no_such_t", argument="time"),
        Fault(
            "missing_cluster_column",
            (6, 0),
            column="no_such_cluster",
            argument="cluster",
        ),
        Fault(
            "uint128_cluster",
            (6, 0),
            code=ErrorCode.COLUMN_UNSUPPORTED_DTYPE,
            params={
                "column": "u128",
                "dtype": "UInt128",
                "argument": "cluster",
            },
        ),
        Fault("missing_time_column", (6, 1), column="no_such_time"),
        Fault(
            "missing_weight_cluster_column",
            (6, 2),
            column="no_such_cluster",
            argument="cluster",
        ),
        Fault(
            "bad_confidence_level",
            (7, 0),
            code=ErrorCode.CONFIDENCE_LEVEL_OUT_OF_RANGE,
            params={"value": 2.0},
        ),
        Fault(
            "bad_max_iter",
            (7, 0),
            code=ErrorCode.MAX_ITER_OUT_OF_RANGE,
            params={"value": 0, "min": 1, "max": 10_000},
        ),
        Fault(
            "non_binary_y",
            (7, 0),
            code=ErrorCode.OUTCOME_NOT_BINARY,
        ),
        Fault(
            "bad_censoring_bounds",
            (7, 0),
            code=ErrorCode.CENSORING_BOUNDS_NOT_ORDERED,
            params={"lower": 1.0, "upper": 0.0},
        ),
    ]
}

# `y`の列名を使う故障。`y`を別の列名に変える故障と混ぜると、成立しなくなる。
_Y_FAULTS = frozenset({"y_in_x", "weight_is_y", "time_is_y", "dk_time_is_y"})
_Y_CHANGERS = frozenset({"missing_y", "non_binary_y", "uint128_y"})
# `hac`/`dk`を選ぶと使われてしまう、`hac_lags`/`dk_*`の未使用の故障。
_UNUSED_WITHOUT_TIME_COV = frozenset(
    {"unused_hac_lags", "unused_dk_bandwidth", "unused_dk_time"}
)


def changes(method: Method, fault_id: str) -> Changes:
    """故障を入力に混ぜるための、`(引数の差分, オプションの差分)`。"""
    x_role = method.x_role
    base_x = method.base_args[x_role]
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
            # FE/REの既定の`cov_type`は`cluster`なので、`cluster`が使われなくなるよう
            # `classical`にする。
            if method.time_cov_type == "dk":
                return {}, {"cov_type": "classical", "cluster": "cluster"}
            return {}, {"cluster": "cluster"}
        case "unused_hac_lags":
            return {}, {"hac_lags": 2}
        case "unused_dk_bandwidth":
            return {}, {"dk_bandwidth": 2}
        case "unused_dk_time":
            return {}, {"dk_time": "t"}
        case "unused_gmm_max_iter":
            return {}, {"gmm_max_iter": 10}
        case "unused_gmm_tol":
            return {}, {"gmm_tol": 1e-3}
        case "unused_gmm_weight_type":
            return {}, {"gmm_type": "one_step", "gmm_weight_type": "robust"}
        case "missing_time_option":
            return {}, {"cov_type": method.time_cov_type}
        case "missing_weight_time_option":
            return {}, {"gmm_weight_type": "hac"}
        case "duplicate_x":
            return {x_role: [base_x[0], base_x[0]]}, {}
        case "y_in_x":
            return {x_role: [method.base_args["y"], *base_x]}, {}
        case "const_in_x":
            return {x_role: ["const", *base_x]}, {}
        case "empty_x":
            if method.name.startswith("iv"):
                return {"x_endog": []}, {}
            return {x_role: []}, {}
        case "weight_is_y":
            return {"weight": method.base_args["y"]}, {}
        case "sigma_in_x":
            return {x_role: [*base_x, "sigma"]}, {}
        case "time_is_y":
            return {}, {"time": method.base_args["y"]}
        case "dk_time_is_y":
            return {}, {"cov_type": "dk", "dk_time": method.base_args["y"]}
        case "uint128_y":
            return {"y": "u128"}, {}
        case "uint128_x":
            return {x_role: [*base_x, "u128"]}, {}
        case "uint128_cluster":
            return {}, {"cov_type": "cluster", "cluster": "u128"}
        case "missing_y":
            return {"y": "no_such_y"}, {}
        case "missing_x":
            return {x_role: [*base_x, "no_such_x"]}, {}
        case "missing_weight":
            return {"weight": "no_such_weight"}, {}
        case "missing_instrument":
            instruments = method.base_args["instruments"]
            return {"instruments": [*instruments, "no_such_z"]}, {}
        case "missing_entity":
            return {"entity": "no_such_e"}, {}
        case "missing_time":
            return {}, {"time": "no_such_t"}
        case "missing_cluster_column":
            return {}, {"cov_type": "cluster", "cluster": "no_such_cluster"}
        case "missing_time_column":
            return {}, {
                "cov_type": method.time_cov_type,
                method.time_option: "no_such_time",
            }
        case "missing_weight_cluster_column":
            return {}, {
                "gmm_weight_type": "cluster",
                "cluster": "no_such_cluster",
            }
        case "bad_confidence_level":
            return {}, {"confidence_level": 2.0}
        case "bad_max_iter":
            return {}, {"max_iter": 0}
        case "non_binary_y":
            return {"y": "y"}, {}
        case "bad_censoring_bounds":
            return {}, {"lower": 1.0, "upper": 0.0}
    raise AssertionError(fault_id)


def compose(
    method: Method, early: str, late: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """2つの故障を混ぜた`(args, options)`。混ぜられない組は`None`。

    混ぜられないのは、次の組。どれも、混ぜると片方の故障が成立しなくなる。

    - 同じキーに別の値を入れる故障同士
    - 「時間順の列が無い」故障と、その列を指定する故障
    - `hac_lags`/`dk_*`の未使用の故障と、`hac`/`dk`を選ぶ故障
      （選ぶと`hac_lags`/`dk_*`は使われる）
    - `y`の列名を使う故障と、`y`を別の列に変える故障
    """
    e_args, e_opts = changes(method, early)
    l_args, l_opts = changes(method, late)
    if (
        early == "unused_cluster"
        and method.time_cov_type == "dk"
        and late in ("missing_time_option", "missing_time_column")
    ):
        # FE/REの`unused_cluster`は`cov_type`を`classical`にするが、`dk`の故障と混ぜる
        # ときは`dk`のまま`cluster`だけを残す（`dk`では`cluster`は使われない）。
        return {**method.base_args, **l_args}, {
            **method.base_opts,
            **l_opts,
            "cluster": "cluster",
        }
    conflicts = {k for k in e_args if k in l_args and e_args[k] != l_args[k]}
    conflicts |= {k for k in e_opts if k in l_opts and e_opts[k] != l_opts[k]}
    if conflicts:
        return None
    if (
        early in ("missing_time_option", "missing_weight_time_option")
        and method.time_option in l_opts
    ):
        return None
    if early in _UNUSED_WITHOUT_TIME_COV and (
        l_opts.get("cov_type") == method.time_cov_type
        or l_opts.get("gmm_weight_type") == "hac"
    ):
        return None
    if early in _Y_FAULTS and late in _Y_CHANGERS:
        return None
    args = {**method.base_args, **l_args, **e_args}
    opts = {**method.base_opts, **l_opts, **e_opts}
    return args, opts


# ── 組合せの列挙 ─────────────────────────────────────────────────────

BOUNDARIES = [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]


def _pairs(method: Method) -> list[tuple[str, str]]:
    """`method`で混ぜられる、順位の異なる故障の組（早い方, 遅い方）。"""
    return [
        (a, b)
        for a, b in itertools.permutations(method.faults, 2)
        if FAULTS[a].rank < FAULTS[b].rank and compose(method, a, b)
    ]


def _method_params():
    return [pytest.param(m, id=m.name) for m in METHODS]


def _pair_params():
    return [
        pytest.param(m, a, b, id=f"{m.name}-{a}-before-{b}")
        for m in METHODS
        for a, b in _pairs(m)
    ]


def _single_params():
    return [
        pytest.param(m, f, id=f"{m.name}-{f}")
        for m in METHODS
        for f in m.faults
    ]


def raises_fault(fault: Fault, method: Method) -> Any:
    """`fault`のエラーが`method`で送出されることを検査するコンテキストマネージャ。"""
    if fault.code is not None:
        code = fault.code(method) if callable(fault.code) else fault.code
        params = (
            fault.params(method) if callable(fault.params) else fault.params
        )
        return raises_error(code, **params)
    if fault.column is None:
        return pytest.raises(ValidationError, match=fault.marker)
    params = {"column": fault.column, "data_argument": "data"}
    if fault.argument is not None:
        params["argument"] = fault.argument
    return raises_error(ErrorCode.COLUMN_NOT_FOUND, **params)


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    frame = make_frame(11)
    # 取り込めないdtypeの列。指名されたときの報告順を確かめるための故障が使う。
    u128 = pl.Series("u128", range(frame.height), dtype=pl.Int64).cast(
        pl.UInt128
    )
    return frame.with_columns(u128)


# ── 前提（故障の定義と単独の故障）────────────────────────────────────


@pytest.mark.parametrize("method", _method_params())
def test_baseline_input_is_valid(df, method):
    """故障を混ぜる前の入力は、そのまま推定できる（以降のテストの前提）。"""
    result = method.run(df, dict(method.base_args), dict(method.base_opts))
    assert result is not None


@pytest.mark.parametrize("method", _method_params())
def test_every_fault_of_a_method_is_defined(method):
    """手法に当てはめた故障は、すべて`FAULTS`にあり、混ぜ方を持つ。"""
    for fault in method.faults:
        assert fault in FAULTS, fault
        changes(method, fault)


@pytest.mark.parametrize(("method", "fault"), _single_params())
def test_each_fault_alone_raises_its_own_error(df, method, fault):
    """故障を1つだけ混ぜると、その故障のエラーが出る（組合せテストの前提）。"""
    args_changes, opts = changes(method, fault)
    args = {**method.base_args, **args_changes}
    with raises_fault(FAULTS[fault], method):
        method.run(df, args, {**method.base_opts, **opts})


# ── 組合せ（順位の早い故障が先に報告される）──────────────────────────


@pytest.mark.parametrize(("method", "early", "late"), _pair_params())
def test_earlier_error_is_reported_first(df, method, early, late):
    """誤りが2つあると、仕様上先に来るエラーが報告される。"""
    composed = compose(method, early, late)
    assert composed is not None
    args, opts = composed
    with raises_fault(FAULTS[early], method):
        method.run(df, args, opts)


@pytest.mark.parametrize("method", _method_params())
def test_every_method_covers_every_stage_boundary(method):
    """どの手法も、段階2〜7の隣り合う境界のすべてを組合せで確かめている。

    故障の定義を変えて、ある手法の境界の組がすべて消えたときに気づくための検査。
    """
    covered = {
        (FAULTS[a].rank[0], FAULTS[b].rank[0]) for a, b in _pairs(method)
    }
    for boundary in BOUNDARIES:
        assert boundary in covered, boundary
