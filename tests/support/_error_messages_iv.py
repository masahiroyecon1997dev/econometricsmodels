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

INSUFFICIENT_INSTRUMENTS = (
    "insufficient instruments for identification: {n_instruments} "
    "instrument(s) provided but {n_endog} endogenous regressor(s) require "
    "at least {n_endog} (order condition: len(instruments) >= len(x_endog))"
)

INSUFFICIENT_CLUSTERS_FOR_WEIGHT_MATRIX = (
    "gmm_weight_type='cluster' requires at least l clusters (l+1 if exactly "
    "identified) for the moment weight matrix: got g={g} clusters for l={l} "
    "instruments (including exogenous regressors), but the cluster moment "
    "covariance has rank at most g (g-1 if exactly identified), so it is "
    "singular"
)

# 観測数のチェック（`validate_observation_counts`）は構造方程式の`n<=k`を先に
# 共通の`INSUFFICIENT_OBSERVATIONS`で、次に第一段階の`n<=l`を
# `INSUFFICIENT_OBSERVATIONS_FOR_FIRST_STAGE`で報告する。
INSUFFICIENT_OBSERVATIONS_FOR_FIRST_STAGE = (
    "insufficient observations for the first stage: n={n} must be greater "
    "than l={l} (number of instruments, including the intercept and the "
    "exogenous regressors); with n <= l the first stage leaves no residual "
    "degrees of freedom (at n = l it fits the endogenous variables "
    "perfectly, so the IV estimate equals OLS)"
)
