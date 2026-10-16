"""linear系統固有の`ValidationError`/`ComputationError`メッセージのテンプレート文字列。

Rust側の送出元: `engine/src/linear/common.rs`（OLS/WLS）。
全系統共通のメッセージと`escaped()`等のフォーマット関数は`_error_messages.py`を参照
（使い方は同じ。Rust側の文言が正本で、このファイルはそのコピー）。
"""

from __future__ import annotations

NON_POSITIVE_WEIGHT = "weight at row {row} must be positive, got {weight}"

# White検定（engine/src/linear/common.rs・engine/src/linear/diagnostics.rs・
# engine_pybind/src/linear/ols.rs）。
UNKNOWN_DIAGNOSTIC_STATISTIC = (
    "unknown statistic: '{other}'. Expected 'lm' or 'f'"
)

INSUFFICIENT_OBSERVATIONS_AUX_REGRESSION = (
    "insufficient observations for the auxiliary regression of the diagnostic "
    "test: n={n} must be greater than the number of auxiliary regressors "
    "including the intercept (k={k})"
)

WHITE_NO_TRAINING_DATA = (
    "white_test() requires the original training data, which is not "
    "retained for this result"
)

# 補助回帰が除外後も特異・推定不能など、`fit()`は通ったがWhite検定の補助回帰で
# 失敗した場合のメッセージの先頭（原因の文言は状況で変わるため先頭のみ固定）。
WHITE_AUX_REGRESSION_PREFIX = "computation failed: White test:"

# Breusch-Pagan検定。`variables`の型の誤りは`TypeError`（`extract_column_list`）。
BP_NO_TRAINING_DATA = (
    "breusch_pagan_test() requires the original training data, which is not "
    "retained for this result"
)

BP_AUX_REGRESSION_PREFIX = "computation failed: Breusch-Pagan test:"

VARIABLES_NOT_A_LIST = (
    "'variables' must be a list of column names (e.g. variables=[\"x1\"]), "
    "got {type_name}"
)

VARIABLES_ELEMENT_NOT_STR = (
    "'variables[{index}]' must be a str column name, got {type_name}"
)

# Breusch-Godfrey検定。
INVALID_NLAGS = "nlags must be a positive integer: got {nlags}"

BG_NO_TRAINING_DATA = (
    "breusch_godfrey_test() requires the original training data, which is "
    "not retained for this result"
)
