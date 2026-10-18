"""panel系統固有の`ValidationError`/`ComputationError`メッセージに使う定数。

panel系統のエラーは`code`と`params`で検査する（`_errors.py`の`raises_error`）。
ここに残すのは、`OPTION_UNKNOWN_VALUE`・`COV_TYPE_NOT_AVAILABLE_FOR_MODEL`の`allowed`
として照合する、FE/REが受け付ける値の一覧だけ。
"""

from __future__ import annotations

# `OPTION_UNKNOWN_VALUE`・`COV_TYPE_NOT_AVAILABLE_FOR_MODEL`の`allowed`
# （FE/REが受け付ける`cov_type`と`variance_method`。`hc0`は含まない）。
ALLOWED_COV_TYPES_PANEL = ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]
ALLOWED_VARIANCE_METHODS = ["swamy_arora", "wallace_hussain", "amemiya"]
