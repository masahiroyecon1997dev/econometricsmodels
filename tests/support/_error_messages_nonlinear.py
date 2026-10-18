"""nonlinear系統固有の`ValidationError`/`ComputationError`メッセージのテンプレート文字列。

Rust側の送出元: `engine/src/nonlinear/`・`engine_pybind/src/nonlinear/`（Logit/Probit/Tobit）。
全系統共通のメッセージと`escaped()`等のフォーマット関数は`_error_messages.py`を参照
（使い方は同じ。Rust側の文言が正本で、このファイルはそのコピー）。
"""

from __future__ import annotations

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
