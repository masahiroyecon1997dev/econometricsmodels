"""IV系統固有の`ValidationError`/`ComputationError`メッセージのテンプレート文字列。

Rust側の送出元: `engine/src/iv/common.rs`・`engine_pybind/src/iv/common.rs`。
全系統共通のメッセージと`escaped()`等のフォーマット関数は`_error_messages.py`を参照
（使い方は同じ。Rust側の文言が正本で、このファイルはそのコピー）。
"""

from __future__ import annotations

UNKNOWN_IV_ESTIMATOR = (
    "unknown estimator: '{estimator}'. Expected one of '2sls' or 'gmm'"
)

UNKNOWN_WEIGHT_TYPE = (
    "unknown gmm_weight_type: '{other}'. Expected one of 'classical', "
    "'robust', 'cluster', or 'hac'"
)

INSUFFICIENT_INSTRUMENTS = (
    "insufficient instruments for identification: {n_instruments} "
    "instrument(s) provided but {n_endog} endogenous regressor(s) require "
    "at least {n_endog} (order condition: len(instruments) >= len(x_endog))"
)

INVALID_GMM_MAX_ITER = (
    "gmm_max_iter must be an integer between 3 (counting the initial "
    "estimate) and 10000, got {max_iter}; for fewer than 3 use "
    'gmm_type="two_step" for a two-step GMM'
)

UNKNOWN_GMM_TYPE = (
    "unknown gmm_type: '{other}'. Expected one of 'one_step', 'two_step', "
    "or 'iterated'"
)

INVALID_GMM_TOL = "gmm_tol must be a positive finite number, got {gmm_tol}"

INSUFFICIENT_CLUSTERS_FOR_WEIGHT_MATRIX = (
    "gmm_weight_type='cluster' requires at least l clusters (l+1 if exactly "
    "identified) for the moment weight matrix: got g={g} clusters for l={l} "
    "instruments (including exogenous regressors), but the cluster moment "
    "covariance has rank at most g (g-1 if exactly identified), so it is "
    "singular"
)

# `IvError::FirstStageFailed`（engine/src/iv/common.rs）。`engine_pybind::fit()`
# （engine_pybind/src/iv/common.rs）は`TwoSlsEstimator::fit`/`GmmEstimator::fit`
# を呼ぶより前に無条件で`compute_first_stage`（弱操作変数診断用）を呼ぶため、
# 第一段階回帰由来の`ValidationError`（`InsufficientObservations`等）はこの
# ラッパー経由で観測される。クラスター数（`G<=q`）は、構造方程式自身のqを使う
# `engine_pybind::fit()`のチェックが`compute_first_stage`より前にラッパー無しで
# 弾く。第一段階側のクラスター数不足（構造方程式では`G>q`）は失敗にならず、
# 第一段階の全体F検定が`None`になる。
# 観測数のチェック（`validate_observation_counts`）は構造方程式の`n<=k`を先に
# 共通の`INSUFFICIENT_OBSERVATIONS`で、次に第一段階の`n<=l`を
# `INSUFFICIENT_OBSERVATIONS_FOR_FIRST_STAGE`で報告する。どちらも`compute_first_stage`
# より前に弾くため、`FIRST_STAGE_FAILED`でラップされることはない。
FIRST_STAGE_FAILED = "first stage regression for endogenous variable '{endog_name}' failed: {source}"

INSUFFICIENT_OBSERVATIONS_FOR_FIRST_STAGE = (
    "insufficient observations for the first stage: n={n} must be greater "
    "than l={l} (number of instruments, including the intercept and the "
    "exogenous regressors); with n <= l the first stage leaves no residual "
    "degrees of freedom (at n = l it fits the endogenous variables "
    "perfectly, so the IV estimate equals OLS)"
)

# 設計行列が特異なときの`LeastSquaresError::SingularMatrix`の文言。第一段階では
# `FIRST_STAGE_FAILED`の`source`、第二段階では`SECOND_STAGE_FAILED`の`source`になる。
SINGULAR_DESIGN_MATRIX = (
    "design matrix is singular (perfect multicollinearity detected)"
)

# `IvError::SecondStageFailed`（engine/src/iv/common.rs）。2SLSのみ。
SECOND_STAGE_FAILED = "second stage regression failed: {source}"

# GMMは第二段階回帰を持たず、`X'ZWZ'X`の逆行列が取れないときに失敗する
# （`engine/src/iv/gmm.rs`）。第一段階が成功し、構造方程式の説明変数同士が
# 共線のときにこの経路になる。
GMM_BREAD_NOT_INVERTIBLE = (
    "computation failed: failed to invert X'ZWZ'X (GMM bread matrix)"
)
