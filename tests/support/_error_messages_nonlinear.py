"""nonlinear系統固有の`ValidationError`/`ComputationError`メッセージのテンプレート文字列。

Rust側の送出元: `engine/src/nonlinear/`・`engine_pybind/src/nonlinear/`（Logit/Probit/Tobit）。
全系統共通のメッセージと`escaped()`等のフォーマット関数は`_error_messages.py`を参照
（使い方は同じ。Rust側の文言が正本で、このファイルはそのコピー）。
"""

from __future__ import annotations


def format_bound(value: float | None) -> str:
    """打ち切り境界のメッセージ内での表記（`None`・`0.0`）。

    `MleError`の打ち切り境界系メッセージ（`InvalidCensoringBounds`等）が使う。
    設定されていない境界は`None`、設定されている境界はPythonの`repr(float)`
    （整数値でも常に小数点付き）と一致する。
    """
    return "None" if value is None else repr(value)


# `OPTION_UNKNOWN_VALUE`の`allowed`（受け付ける値の一覧）。
ALLOWED_COV_TYPES_NONLINEAR = ["classical", "opg", "hc0", "hc1", "cluster"]
ALLOWED_SOLVERS = ["newton", "bfgs", "lbfgs"]
ALLOWED_MARGINAL_EFFECTS_AT = ["overall", "mean", "median"]
# Tobitの`marginal_effects()`・`predict()`・`augment()`の`target`。
ALLOWED_MARGINAL_EFFECTS_TARGET = [
    "expected_latent",
    "expected_observed",
    "prob_uncensored",
    "expected_uncensored",
]

INVALID_BINARY_Y = (
    "y at row {row} must be coded as 0.0 or 1.0 (binary outcome), got {value}"
)

# Tobit固有（engine/src/nonlinear/common.rs・engine_pybind/src/nonlinear/tobit.rs）
INVALID_CENSORING_BOUNDS = (
    "invalid censoring bounds: lower={lower}, upper={upper} (at least one "
    "bound must be set, and lower must be less than upper when both are set)"
)

Y_OUT_OF_CENSORING_BOUNDS = (
    "y at row {row} is out of the censoring bounds (lower={lower}, "
    "upper={upper}): got {value}"
)

NO_UNCENSORED_OBSERVATIONS = (
    "no uncensored observations: at least one y value strictly between "
    "lower={lower} and upper={upper} is required to identify the model"
)

UNCENSORED_INTERVAL_TOO_NARROW = (
    "target 'expected_uncensored' cannot be computed reliably: the uncensored "
    "interval [{lower}, {upper}] is narrower than 1% of the estimated sigma "
    "({sigma}), so the model puts almost no probability on uncensored values. "
    "Use another target"
)
