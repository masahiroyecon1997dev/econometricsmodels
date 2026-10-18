"""linear系統固有の`ValidationError`/`ComputationError`メッセージのテンプレート文字列。

Rust側の送出元: `engine/src/linear/common.rs`（OLS/WLS）。
全系統共通のメッセージと`escaped()`等のフォーマット関数は`_error_messages.py`を参照
（使い方は同じ。Rust側の文言が正本で、このファイルはそのコピー）。
"""

from __future__ import annotations

# White・Breusch-Pagan・Breusch-Godfrey検定の未知の`statistic`に対する
# `OPTION_UNKNOWN_VALUE`の`allowed`。
ALLOWED_DIAGNOSTIC_STATISTICS = ["lm", "f"]

# Breusch-Pagan検定。`variables`の型の誤りは`TypeError`（`extract_column_list`）。
VARIABLES_NOT_A_LIST = (
    "'variables' must be a list of column names (e.g. variables=[\"x1\"]), "
    "got {type_name}"
)

VARIABLES_ELEMENT_NOT_STR = (
    "'variables[{index}]' must be a str column name, got {type_name}"
)
