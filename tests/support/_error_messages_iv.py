"""IV系統固有の`ValidationError`/`ComputationError`メッセージのテンプレート文字列。

Rust側の送出元: `engine/src/iv/common.rs`・`engine_pybind/src/iv/common.rs`。
全系統共通のメッセージと`escaped()`等のフォーマット関数は`_error_messages.py`を参照
（使い方は同じ。Rust側の文言が正本で、このファイルはそのコピー）。
"""

from __future__ import annotations

# `OPTION_UNKNOWN_VALUE`の`allowed`（受け付ける値の一覧）。
ALLOWED_ESTIMATORS = ["2sls", "gmm"]
ALLOWED_GMM_TYPES = ["one_step", "two_step", "iterated"]
ALLOWED_GMM_WEIGHT_TYPES = ["classical", "robust", "cluster", "hac"]
