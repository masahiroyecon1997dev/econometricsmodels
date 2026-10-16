"""panel系統固有の`ValidationError`/`ComputationError`メッセージのテンプレート文字列。

Rust側の送出元: `engine/src/panel/common.rs`・`engine_pybind/src/panel/`（FE/RE）。
全系統共通のメッセージと`escaped()`等のフォーマット関数は`_error_messages.py`を参照
（使い方は同じ。Rust側の文言が正本で、このファイルはそのコピー）。
"""

from __future__ import annotations

# `PanelError::InsufficientDegreesOfFreedom`。1-wayでは`n_periods_clause`が
# 空文字列、2-wayでは`, n_periods={n}`になる（Rust側の`n_periods_clause`
# 関数と同じ。`OptionのDebug表記が漏れるのを避けるための専用フォーマット、
# engine/src/panel/common.rs参照）。
INSUFFICIENT_DEGREES_OF_FREEDOM_PANEL = (
    "insufficient degrees of freedom for panel estimation: n_obs={n_obs}, "
    "n_entities={n_entities}{n_periods_clause}, k={k} (the panel-adjusted "
    "residual degrees of freedom must be positive)"
)


def n_periods_clause(n_periods: int | None) -> str:
    """`INSUFFICIENT_DEGREES_OF_FREEDOM_PANEL`の`{n_periods_clause}`用。"""
    return "" if n_periods is None else f", n_periods={n_periods}"


# `PanelError::SingletonGroup`（6.5節）。`{dimension}`は"entity"/"time"。
SINGLETON_GROUP = (
    "singleton {dimension} group detected: {dimension} '{group_id}' has "
    "only 1 observation. Singleton groups are not dropped automatically; "
    "remove them from the input"
)

# `PanelError::UnbalancedPanelForTwoWay`（6.4節）。
UNBALANCED_PANEL_FOR_TWO_WAY = (
    "two-way estimation (fixed or random effects) requires a balanced "
    "panel: got n_obs={n_obs} for n_entities={n_entities} x "
    "n_periods={n_periods} (expected {expected} observations)"
)

# `PanelError::ZeroVarianceAfterDemeaning`（6.7節）。
ZERO_VARIANCE_AFTER_DEMEANING = (
    "regressor '{column}' has zero variance after the within-"
    "transformation (it is time-invariant, or collinear with the fixed "
    "effects)"
)

# RE: 未知の`variance_method`。
UNKNOWN_VARIANCE_METHOD = (
    "unknown variance_method: '{value}'. Expected one of 'swamy_arora', "
    "'wallace_hussain', or 'amemiya'"
)

# RE: `variance_method="amemiya"`でwithin変動の無い列がある
# （`PanelError::AmemiyaRequiresWithinVariation`）。
AMEMIYA_NEEDS_WITHIN_VARIATION = (
    'variance_method="amemiya" needs every regressor to vary within the '
    "effect (within each entity; for two-way effects also across entities and "
    "across periods), but these have no within variation: {columns}. Use "
    'variance_method="swamy_arora" or "wallace_hussain", which accept such '
    "regressors"
)

# RE: `variance_method="wallace_hussain"`の1段目（プールドOLS）が特異
# （`PanelError::PooledRegressionFailed`、`source`は`LeastSquaresError::SingularMatrix`）。
POOLED_REGRESSION_FAILED = (
    "pooled least-squares estimation for the variance component failed: "
    "design matrix is singular (perfect multicollinearity detected)"
)

# RE: `variance_method`が推定した`σ_ε²`が正でない
# （`PanelError::NonPositiveIdiosyncraticVariance`）。`{sigma2_eps}`はRustのf64のDisplay。
NON_POSITIVE_IDIOSYNCRATIC_VARIANCE = (
    "the estimated idiosyncratic variance is not positive ({sigma2_eps}) "
    'with variance_method="{method}"; the weights of the quasi-demeaning '
    "are undefined"
)

# RE: `cov_type="dk"`で`dk_time`が未指定（`time`は借用されない。1-way・2-wayとも）。
DK_REQUIRES_TIME = (
    "cov_type='dk' requires the `dk_time` option: the column that defines the "
    "time periods of the Driscoll-Kraay estimator (it is not taken from "
    "`time`, which only sets the two-way random effects)"
)

# FE: `cov_type="dk"`で`dk_time`が未指定（`time`は借用されない。1-way・2-wayとも）。
FE_DK_REQUIRES_DK_TIME = (
    "cov_type='dk' requires the `dk_time` option: the column that defines the "
    "time periods of the Driscoll-Kraay estimator (it is not taken from "
    "`time`, which only sets the two-way fixed effects)"
)

# `PanelError::InvalidDkBandwidth`。`t`は時点数（観測数`n`ではない点に
# 注意、OLSの`INVALID_HAC_LAGS`とは上限の意味が異なる）。
INVALID_DK_BANDWIDTH = (
    "bandwidth must be in the range [0, t): got {bandwidth}, t={t}"
)

# `PanelError::InsufficientDkPeriodsForInference`（`INSUFFICIENT_CLUSTERS_FOR_
# INFERENCE`のDK版。`q`はFEでは傾き係数のF検定、REではハウスマン検定の対象数）。
INSUFFICIENT_DK_PERIODS_FOR_INFERENCE = (
    "cov_type='dk' requires more unique time periods than jointly tested "
    "coefficients: got t_periods={t_periods} for q={q} coefficient(s) (the "
    "slope F-test for FE, the Hausman test for RE), but the Driscoll-Kraay "
    "covariance has rank at most t_periods-1, so the q×q Wald submatrix is "
    "singular when t_periods <= q"
)

# `PanelError::DegenerateDkTwoPeriods` / `DegenerateClusterTwoGroups`（FEのみ）。
DEGENERATE_DK_TWO_PERIODS = (
    "cov_type='dk' with 2 unique time periods is degenerate for fixed "
    "effects when every entity (or, with two-way effects, every time period) "
    "is observed exactly once in each of the two periods: the within "
    "transformation makes the two per-period scores equal, and they sum to "
    "zero, so the Driscoll-Kraay covariance is identically zero. Use more "
    "time periods or another cov_type"
)

DEGENERATE_CLUSTER_TWO_GROUPS = (
    "cov_type='cluster' with 2 clusters is degenerate for fixed effects when "
    "every entity (or, with two-way effects, every time period) is observed "
    "exactly once in each cluster (e.g. clustering by time with 2 periods, "
    "or by entity with 2 entities and two-way effects): the within "
    "transformation makes the two cluster scores equal, and they sum to "
    "zero, so the cluster-robust covariance is identically zero. Use more "
    "clusters or another cov_type"
)

# FE用cov_type文字列パース（engine_pybind/src/panel/fe.rs::parse_fe_cov_type）。
# OLS/WLS/IVの`UNKNOWN_COV_TYPE_LINEAR`と異なりhc0を含まない一覧になる。
UNKNOWN_COV_TYPE_FE = (
    "unknown cov_type: '{other}'. Expected one of 'classical', 'hc1' "
    "through 'hc3', 'cluster', or 'dk'"
)

HC0_NOT_SUPPORTED_FE = (
    "cov_type='hc0' is not supported for FE (neither linearmodels nor "
    "fixest offer HC0 for panel/FE regressions); use 'hc1', 'hc2', or "
    "'hc3' instead"
)

# RE用cov_type文字列パース（engine_pybind/src/panel/re.rs::parse_re_cov_type）。
# 「unknown cov_type」文言自体はFEと一字一句同じ（UNKNOWN_COV_TYPE_FEを流用する）が、
# hc0専用メッセージはFEと文言が異なる（参照実装の主語が「no reference
# implementation」でlinearmodels/fixestを個別に挙げない）ため別定数にする。
HC0_NOT_SUPPORTED_RE = (
    "cov_type='hc0' is not supported for RE (no reference implementation "
    "offers HC0 for panel/RE regressions); use 'hc1', 'hc2', or 'hc3' "
    "instead"
)
