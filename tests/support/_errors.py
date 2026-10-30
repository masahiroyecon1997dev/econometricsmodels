"""構造化例外（`ValidationError`/`ComputationError`）を`code`と`params`で検査するヘルパー。

新規・更新するテストは、文面ではなく`code`と`params`で照合する（`message`/`hint`の
言い回しを直してもテストが壊れないようにするため）。文面そのものはRust側の単体テストと
`tests/cross_method/test_error_catalog.py`が固定する。

`raises_error`は検査した`code`を記録し、`--check-error-coverage`を付けた全体実行の終了時に
`MIGRATED_CODES`の全`code`が1度は送出されたことを`conftest.py`が検査する
（一度も送出されない`code`は、死んだ定義かテストの抜け）。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest
from econometricsmodels import (
    ComputationError,
    ErrorCode,
    ValidationError,
    _lib,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

_CATALOG = {c["code"]: c for c in _lib.error_catalog()["codes"]}
_STAGES = {s["name"]: s for s in _lib.error_catalog()["stages"]}
_CLASSES = {
    "ValidationError": ValidationError,
    "ComputationError": ComputationError,
}

# `raises_error`が検査して通った`code`（セッション中に積み上がる）。
RAISED_CODES: set[str] = set()

# 実際のエラー経路から送出されることをテストで保証済みの`code`。エラーの移行が進んだ
# 段階で、その`code`をここに足す（全体実行の終了時に、足した全`code`が1度は
# `raises_error`で検査されたかを調べる）。全`code`の移行が終わったら、カタログの
# 全`code`を対象にする。
MIGRATED_CODES: frozenset[str] = frozenset(
    {
        ErrorCode.COLUMN_NOT_FOUND,
        ErrorCode.COLUMN_CONTAINS_NULL,
        ErrorCode.COLUMN_NON_FINITE,
        ErrorCode.COLUMN_UNSUPPORTED_DTYPE,
        ErrorCode.COLUMN_TIMEZONE_NOT_SUPPORTED,
        ErrorCode.COLUMN_VALUES_TIED,
        ErrorCode.ROLE_EMPTY,
        ErrorCode.ROLE_DUPLICATE_COLUMN,
        ErrorCode.ROLE_COLUMN_OVERLAP,
        ErrorCode.INTERCEPT_NAME_COLLISION,
        ErrorCode.TOBIT_SIGMA_NAME_COLLISION,
        ErrorCode.AUGMENT_COLUMN_EXISTS,
        ErrorCode.OPTION_UNKNOWN_VALUE,
        ErrorCode.OPTION_UNUSED,
        ErrorCode.HAC_TIME_REQUIRED,
        ErrorCode.DK_TIME_REQUIRED,
        ErrorCode.COV_TYPE_NOT_AVAILABLE_FOR_MODEL,
        ErrorCode.CONFIDENCE_LEVEL_OUT_OF_RANGE,
        ErrorCode.HAC_LAGS_OUT_OF_RANGE,
        ErrorCode.TOLERANCE_NOT_POSITIVE_FINITE,
        ErrorCode.MAX_ITER_OUT_OF_RANGE,
        ErrorCode.GMM_MAX_ITER_OUT_OF_RANGE,
        ErrorCode.NLAGS_NOT_POSITIVE,
        ErrorCode.DK_BANDWIDTH_OUT_OF_RANGE,
        ErrorCode.INSUFFICIENT_OBSERVATIONS,
        ErrorCode.CLUSTER_REQUIRED,
        ErrorCode.CLUSTER_COUNT_TOO_SMALL,
        ErrorCode.CLUSTER_COUNT_NOT_ABOVE_SLOPES,
        ErrorCode.DESIGN_MATRIX_SINGULAR,
        ErrorCode.MATRIX_NEAR_SINGULAR,
        ErrorCode.NOT_CONVERGED,
        ErrorCode.SEPARATION_DETECTED,
        ErrorCode.UNCENSORED_INTERVAL_TOO_NARROW,
        ErrorCode.WEIGHT_NOT_POSITIVE,
        ErrorCode.INSUFFICIENT_OBSERVATIONS_FOR_DIAGNOSTIC,
        ErrorCode.DIAGNOSTIC_VARIABLES_CONSTANT,
        ErrorCode.DIAGNOSTIC_VARIABLE_DEGENERATE,
        ErrorCode.DIAGNOSTIC_STATISTIC_UNDEFINED,
        ErrorCode.TRAINING_DATA_NOT_RETAINED,
        ErrorCode.AUGMENT_REQUIRES_NEW_DATA,
        ErrorCode.OUTCOME_NOT_BINARY,
        ErrorCode.CENSORING_BOUNDS_MISSING,
        ErrorCode.CENSORING_BOUNDS_NOT_ORDERED,
        ErrorCode.CENSORING_BOUND_NOT_FINITE,
        ErrorCode.OUTCOME_OUTSIDE_CENSORING_BOUNDS,
        ErrorCode.NO_UNCENSORED_OBSERVATIONS,
        ErrorCode.INSUFFICIENT_INSTRUMENTS,
        ErrorCode.INSUFFICIENT_OBSERVATIONS_FOR_FIRST_STAGE,
        ErrorCode.GMM_WEIGHT_CLUSTER_COUNT_TOO_SMALL,
        ErrorCode.INSUFFICIENT_DEGREES_OF_FREEDOM,
        ErrorCode.SINGLETON_GROUP,
        ErrorCode.PANEL_NOT_BALANCED_FOR_TWO_WAY,
        ErrorCode.REGRESSOR_NO_WITHIN_VARIATION,
        ErrorCode.DK_PERIODS_TOO_FEW,
        ErrorCode.DK_PERIODS_NOT_ABOVE_COEFFICIENTS,
        ErrorCode.DK_TWO_PERIODS_DEGENERATE,
        ErrorCode.CLUSTER_TWO_GROUPS_DEGENERATE,
        ErrorCode.AMEMIYA_REQUIRES_WITHIN_VARIATION,
        ErrorCode.NON_POSITIVE_IDIOSYNCRATIC_VARIANCE,
        ErrorCode.OUTCOME_CONSTANT,
        ErrorCode.OUTCOME_CONSTANT_WITHIN_ENTITY,
        ErrorCode.PERFECT_FIT,
        ErrorCode.PICKLE_FORMAT_MISMATCH,
        ErrorCode.INTERNAL_ERROR,
    }
)


# 実際のエラー経路から構成できない`code`と、その理由。カタログの`code`は
# `MIGRATED_CODES`（送出をテストで保証）か、ここ（構成できない理由を添えて許可）の
# どちらかに入っていなければならない（`tests/cross_method/test_error_catalog.py`が検査する）。
# 文面・形式はカタログ全体の検査（`_lib.raise_error_for_test`）とRustの単体テストが固定する。
# `MATRIX_NEAR_SINGULAR`の`matrix`のうち、Pythonから構成できているのは
# `COEFFICIENT_COVARIANCE`だけ（`DESIGN_GRAM`・`HESSIAN`・`OPG`・`GMM_WEIGHT`は、前段の
# 検査や分解が先に失敗するため、極端なスケール差のデータでも構成できていない）。
UNREACHABLE_CODES: dict[str, str] = {
    ErrorCode.BREUSCH_GODFREY_MATRIX_TOO_LARGE: (
        "補助回帰の行列を確保できない大きさのときだけ。`n <= k_aux`の検査が先に"
        "働き、確保の失敗を起こせる入力はテストでは作れない"
    ),
    ErrorCode.VARIANCE_COMPONENTS_MOMENTS_SINGULAR: (
        "Amemiyaの不均衡1-wayのモーメント連立の行列式は、通常の入力では正になる。"
        "特異になる実入力は見つかっていない（防御として残している）"
    ),
    ErrorCode.LINE_SEARCH_FAILED: (
        "最適化の直線探索が上限回数で収束しないときだけ。病的な尤度でないと起きず、"
        "再現できる入力が作れない"
    ),
    ErrorCode.SEPARATION_CHECK_FAILED: (
        "分離の判定の線形計画が数値的に完了しないときだけ。病的なデータでないと"
        "起きず、再現できる入力が作れない"
    ),
}


def unraised_migrated_codes() -> list[str]:
    """`MIGRATED_CODES`のうち、まだ`raises_error`で検査されていない`code`。"""
    return sorted(MIGRATED_CODES - RAISED_CODES)


def _expected_param_names(code: str, params: dict[str, object]) -> set[str]:
    """`params`のキーとして期待される集合（`stage`があればその補助値も）。"""
    entry = _CATALOG[code]
    names = {p["name"] for p in entry["params"]}
    if "stage" in params:
        assert entry["stage_allowed"], f"{code} cannot carry a stage"
        stage = _STAGES.get(params["stage"])
        assert stage is not None, f"unknown stage {params['stage']!r}"
        names |= {"stage", *stage["params"]}
    return names


def _type_matches(type_name: str, value: object) -> bool:
    match type_name:
        case "int":
            return isinstance(value, int) and not isinstance(value, bool)
        case "float":
            return isinstance(value, float)
        case "float_or_none":
            return value is None or isinstance(value, float)
        case "str":
            return isinstance(value, str)
        case "str_list" | "column_list":
            return isinstance(value, list) and all(
                isinstance(v, str) for v in value
            )
        case "bool":
            return isinstance(value, bool)
    return False


def _check_error(
    err: ValidationError | ComputationError,
    expected: ErrorCode,
    params: dict[str, object],
) -> None:
    """`err`が`expected`の`code`で、`params`・形式がカタログと一致することを検査する。"""
    assert err.code == expected, (
        f"expected code {expected}, got {err.code}: {err}"
    )
    expected_class = _CLASSES[_CATALOG[expected]["class"]]
    assert type(err) is expected_class, (
        f"{expected} should be {expected_class.__name__}, "
        f"got {type(err).__name__}"
    )
    names = _expected_param_names(expected, err.params)
    assert set(err.params) == names, (
        f"{expected}: params keys {sorted(err.params)} do not match the "
        f"catalog (expected {sorted(names)})"
    )
    for spec in _CATALOG[expected]["params"]:
        value = err.params[spec["name"]]
        assert _type_matches(spec["type"], value), (
            f"{expected}: param {spec['name']!r} should be {spec['type']}, "
            f"got {value!r}"
        )
    for name, value in params.items():
        assert name in err.params, f"{expected}: no param {name!r}"
        assert err.params[name] == value, (
            f"{expected}: param {name!r} is {err.params[name]!r}, "
            f"expected {value!r}"
        )
    full = err.message + (f" {err.hint}" if err.hint else "")
    assert str(err) == full
    assert err.args == (full,)
    RAISED_CODES.add(str(expected))


@contextmanager
def raises_error(
    code: ErrorCode | str, **params: object
) -> Iterator[pytest.ExceptionInfo[ValidationError | ComputationError]]:
    """ブロック内で`code`の例外が送出され、`params`が一致することを検査する。

    `code`のカタログ上のクラス（`ValidationError`/`ComputationError`）で
    あること、`params`のキー集合（`stage`があればその補助値も）と各値の型がカタログと
    一致すること、`str(e)`が
    `message`と`hint`の連結で`args`が`(str(e),)`であることも毎回検査する。
    `params`に渡したキーだけ値を比較する（`pytest.approx`も渡せる）。

    Args:
        code: 期待する`code`。
        **params: 期待する`params`の値（一部のキーだけでよい）。

    Yields:
        送出された例外の`pytest.ExceptionInfo`。
    """
    expected = ErrorCode(code)
    with pytest.raises((ValidationError, ComputationError)) as info:
        yield info
    _check_error(info.value, expected, params)


@contextmanager
def raises_one_of_errors(
    expected: Mapping[ErrorCode | str, Mapping[str, object]],
) -> Iterator[pytest.ExceptionInfo[ValidationError | ComputationError]]:
    """ブロック内で`expected`のいずれかの`code`の例外が送出されることを検査する。

    solverの軌道によって、同じ入力でも別の`code`で失敗しうる場合（分離データを
    bfgs/lbfgsに渡すと、分離の検出か最大反復到達のどちらかになる等）に使う。
    送出された`code`について、`raises_error`と同じ検査（`params`の一致を含む）を行う。

    Args:
        expected: 許容する`code`から、その`code`で期待する`params`への対応。

    Yields:
        送出された例外の`pytest.ExceptionInfo`。
    """
    allowed = {ErrorCode(code): dict(p) for code, p in expected.items()}
    with pytest.raises((ValidationError, ComputationError)) as info:
        yield info
    err = info.value
    assert err.code in allowed, (
        f"expected one of {sorted(map(str, allowed))}, got {err.code}: {err}"
    )
    _check_error(err, err.code, allowed[err.code])
