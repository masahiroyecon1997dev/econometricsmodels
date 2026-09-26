"""全系統で共有する`ValidationError`メッセージのテンプレート文字列。

`engine`（`CommonError`等）・`engine_pybind`（`column_extraction.rs`・
`validation.rs`・各系統の`cov_type`/`method`文字列パース）が実際に送出する
メッセージの正確な文字列をPythonのフォーマット文字列としてここに集約する。

Rust側のメッセージ文言が正本であり（`engine/src/error.rs`・
`engine_pybind/src/column_extraction.rs`・`engine_pybind/src/validation.rs`等）、
このファイルはそのコピー。Rust側で文言を変更した場合はこのファイルも同時に
更新すること（複数系統でメッセージが文字通り重複しているため直書きではなく共通化した）。

各テストは`pytest.raises(ValidationError, match=escaped(TEMPLATE, name=...))`の
形で使う（`escaped()`は`str.format(**kwargs)`した上で`re.escape`する。メッセージ中の
`()`・`.`等は正規表現の特殊文字のため、素の文字列を`match=`にそのまま渡すのは誤り）。
"""

from __future__ import annotations

import re


def escaped(template: str, /, **kwargs: object) -> str:
    """`template.format(**kwargs)`を`pytest.raises(match=...)`用に`re.escape`する。"""
    return re.escape(template.format(**kwargs))


def rust_f64(value: float) -> str:
    """RustのDisplay（`{}`）でのf64表示を模したフォーマット。

    Rustの`f64`のDisplayは、Pythonの`str(float)`/`repr(float)`と異なり、
    整数値でも小数点以下を付けない（例: `format!("{}", 1.0)` は`"1"`、Pythonの
    `str(1.0)`は`"1.0"`）。`CommonError::InvalidConfidenceLevel`等、f64を
    そのまま埋め込むメッセージの期待値を組み立てる際に使う。有限値のみ対応
    （NaN/無限大は`column_extraction.rs`のメッセージでのみ使われ、そちらは別途
    `"NaN"`/`"inf"`を直接指定する）。
    """
    return repr(float(value)).removesuffix(".0")


def fully_qualified_type_name(obj: object) -> str:
    """pyo3の`PyType::fully_qualified_name()`と同じ規則で型名を組み立てる。

    `NOT_A_POLARS_DATAFRAME`の`{type_name}`はRust側でこの関数
    （`engine_pybind/src/column_extraction.rs`の`extract_dataframe`）を
    使って組み立てているため、テスト側も同じ規則（`__module__`が
    `"builtins"`/`"__main__"`のときは`__qualname__`のみ、それ以外は
    `f"{{__module__}}.{{__qualname__}}"`）で期待値を作る。pandasの
    `__module__`の値はバージョンによって異なりうる（例:
    3.0系は`"pandas"`、古いバージョンは`"pandas.core.frame"`）ため、
    ハードコードせずこの関数で実行時に計算する。
    """
    cls = type(obj)
    module = cls.__module__
    if module in ("builtins", "__main__"):
        return cls.__qualname__
    return f"{module}.{cls.__qualname__}"


def rust_option_f64_debug(value: float | None) -> str:
    """RustのDebug（`{:?}`）での`Option<f64>`表示を模したフォーマット。

    `TobitError`の打ち切り境界系メッセージ（`InvalidCensoringBounds`等）が
    使う。RustのDebugはf64のDisplayと異なり整数値でも常に小数点を付ける
    （`None`→`"None"`、`Some(0.0)`→`"Some(0.0)"`）。Pythonの`repr(float)`は
    常に小数点付きのため、そのまま使えばRustのDebugと一致する。
    """
    return "None" if value is None else f"Some({value!r})"


# ── column_extraction.rs（全系統共通） ──────────────────────────────
#
# extract_dataframe: `data`/`new_data`にpolars以外のDataFrame（pandas等）が
# 渡された場合に使う（engine_pybind/src/column_extraction.rs）。
# `param_name`は呼び出し側で"data"（fit系）または"new_data"（predict/augment）
# を渡す。`type_name`はPythonオブジェクトの完全修飾クラス名
# （`type(obj).__module__ + "." + type(obj).__qualname__`相当）。
NOT_A_POLARS_DATAFRAME = (
    "'{param_name}' must be a polars.DataFrame, got {type_name}"
)
# `type_name`が`"polars."`で始まる（渡されたオブジェクト自体は本物のpolars
# DataFrameなのに抽出が失敗している）場合の文言。`pyo3`/`polars`/`pyo3-polars`
# のバージョンの組み合わせによるABI不整合等でのみ発生しうるため、通常の
# テスト環境では再現できず対応するテストは無い（`extract_dataframe`の
# ロジックそのものはRust側で経路を確認済み）。
DATAFRAME_EXTRACTION_FAILED = (
    "failed to read '{param_name}' as a polars.DataFrame: {error}"
)

# extract_f64_column: y/x/weight/time_col の抽出で使う（engine_pybind/src/
# column_extraction.rs:27-75）。

COLUMN_DOES_NOT_EXIST = "column '{name}' does not exist in the data"
COLUMN_HAS_MISSING_VALUES = (
    "column '{name}' contains {count} missing value(s). Missing values are "
    "not handled automatically; please impute or remove them before calling "
    "this function"
)
COLUMN_HAS_NON_FINITE_VALUE = (
    "column '{name}' contains a non-finite value ({value}) at row {row}. "
    "NaN and infinite values are not handled automatically; please impute "
    "or remove them before calling this function"
)
# `Series.cast(Float64)`自体が失敗する場合のメッセージ。polarsは数値として
# 解釈できない文字列（`"a"`等）を非strictキャストでnullに変換するため、通常の
# 非数値文字列テストはこの分岐ではなく`COLUMN_HAS_MISSING_VALUES`を通る
# （実測確認済み、tests/linear/test_ols_validation.py::test_non_numeric_dtype_raises
# 参照）。この分岐が実際にテストで踏まれるケースは現状無い。
COLUMN_NOT_CASTABLE_TO_NUMERIC = (
    "column '{name}' could not be cast to a numeric type (f64):"
)

# extract_group_key_column: cluster_col の抽出で使う（同ファイル86-111行）。
# 列が存在しない場合のメッセージは extract_f64_column と同文言だが、欠損値の
# メッセージはグループキー列専用の短い文言になる点に注意。
GROUP_KEY_COLUMN_HAS_MISSING_VALUES = "column '{name}' contains missing values"

# ── validation.rs（全系統共通、y/weight/x/x_exog/x_endog/instruments等の
#    ロール間検証） ────────────────────────────────────────────────────

X_EMPTY = "{role} must contain at least one column name"
DUPLICATE_WITHIN_ROLE = "column '{name}' is specified more than once in {role}"
CONST_COLLISION = (
    "when include_intercept=true, {role} cannot contain a column named 'const' "
    "(it collides with the automatically added intercept)"
)
EXISTING_COLUMN_COLLISION = (
    "the data already has a column named '{name}'; augment() would overwrite "
    "it, which is not allowed"
)

# `validate_no_duplicate_roles`のメッセージ（engine_pybind/src/validation.rs
# `duplicate_role_message`）。単一列ロール（y/weight）と複数列ロール（x/x_exog/
# x_endog/instruments）の組み合わせによって主語が変わるため3パターンに分ける
# （同ファイルのdocコメント「呼び出し側の契約」参照）。
ROLE_OVERLAP_SINGLE_IN_MULTI = "the column '{col}' specified as {single_role} is also included in {multi_role}"
ROLE_OVERLAP_SINGLE_EQUALS_SINGLE = "the column '{col}' specified as {later_role} is also specified as {earlier_role}"
ROLE_OVERLAP_MULTI_VS_MULTI = "the column '{col}' specified as {later_role} is also included in {earlier_role}"

# ── CommonError（engine/src/error.rs、OLS/WLS/Tobit/Logit/Probit/IV共通） ──

INSUFFICIENT_OBSERVATIONS = (
    "insufficient observations: n={n} must be greater than k={k} (number of "
    "independent variables, including the intercept)"
)
INVALID_CONFIDENCE_LEVEL = (
    "confidence_level must be in the range (0, 1): {confidence_level}"
)
MISSING_CLUSTER_COLUMN = (
    "cov_type='cluster' requires cluster identifiers to be provided"
)
INSUFFICIENT_CLUSTERS = (
    "cov_type='cluster' requires at least 2 clusters, got {g}"
)
INSUFFICIENT_CLUSTERS_FOR_INFERENCE = (
    "cov_type='cluster' requires more clusters than slope coefficients for "
    "joint inference: got g={g} clusters for q={q} slope coefficient(s), but "
    "the cluster-robust covariance has rank at most g-1, so the q×q "
    "Wald/F submatrix is singular when g <= q"
)

# ── linear系統固有（engine/src/linear/common.rs、OLS/WLS） ─────────────

INVALID_HAC_LAGS = (
    "hac_lags must be in the range [0, n): got {hac_lags}, n={n}"
)
NON_POSITIVE_WEIGHT = "weight at row {row} must be positive, got {weight}"

# cov_type文字列パース。OLS/WLS/IVは文言が完全一致（実装は別々、
# engine_pybind/src/linear/common.rs::parse_cov_type・
# engine_pybind/src/iv/common.rs::parse_iv_cov_type）。
UNKNOWN_COV_TYPE_LINEAR = (
    "unknown cov_type: '{other}'. Expected one of 'classical', 'hc0' "
    "through 'hc3', 'hac', or 'cluster'"
)

# ── nonlinear系統固有（Logit/Probit/Tobit） ─────────────────────────────

UNKNOWN_COV_TYPE_NONLINEAR = (
    "unknown cov_type: '{other}'. Expected one of 'classical' (or "
    "'nonrobust'), 'opg', 'hc0', 'hc1', or 'cluster'"
)
UNKNOWN_METHOD_NONLINEAR = (
    "unknown method: '{other}'. Expected one of 'newton', 'bfgs', or 'lbfgs'"
)
INVALID_TOL = "tol must be a positive number, got {tol}"
INVALID_MAX_ITER = "max_iter must be a positive integer, got {max_iter}"
INVALID_BINARY_Y = (
    "y at row {row} must be coded as 0.0 or 1.0 (binary outcome), got {value}"
)
UNKNOWN_MARGINAL_EFFECTS_AT = (
    "unknown at: '{other}'. Expected one of 'overall', 'mean', or 'median'"
)

# Tobit固有（engine/src/nonlinear/common.rs・engine_pybind/src/nonlinear/tobit.rs）
SIGMA_COLLISION = (
    "x cannot contain a column named 'sigma' (it collides with the error "
    "term's standard deviation, which TobitResult appends to param_names/params)"
)
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
UNKNOWN_MARGINAL_EFFECTS_TARGET = (
    "unknown target: '{other}'. Expected one of 'expected_latent', "
    "'expected_observed', or 'prob_uncensored'"
)

# ── IV系統固有（engine/src/iv/common.rs・engine_pybind/src/iv/common.rs） ──

UNKNOWN_IV_METHOD = (
    "unknown method: '{method}'. Expected one of '2sls' or 'gmm'"
)
UNKNOWN_WEIGHT_TYPE = (
    "unknown weight_type: '{other}'. Expected one of 'unadjusted' "
    "('homoskedastic'), 'robust' ('heteroskedastic'), 'cluster', or 'kernel'"
)
INSUFFICIENT_INSTRUMENTS = (
    "insufficient instruments for identification: {n_instruments} "
    "instrument(s) provided but {n_endog} endogenous regressor(s) require "
    "at least {n_endog} (order condition: len(instruments) >= len(x_endog))"
)
INVALID_GMM_ITERATIONS = (
    "gmm_iterations must be a positive integer: got {gmm_iterations}"
)
INVALID_GMM_TOL = "gmm_tol must be a positive number, got {gmm_tol}"
INSUFFICIENT_CLUSTERS_FOR_WEIGHT_MATRIX = (
    "weight_type='cluster' requires at least l clusters (l+1 if exactly "
    "identified) for the moment weight matrix: got g={g} clusters for l={l} "
    "instruments (including exogenous regressors), but the cluster moment "
    "covariance has rank at most g (g-1 if exactly identified), so it is "
    "singular"
)

# ── panel系統固有（FE、engine/src/panel/common.rs・
#    engine_pybind/src/panel/fe.rs） ─────────────────────────────────────

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
    "two-way fixed effects requires a balanced panel: got n_obs={n_obs} "
    "for n_entities={n_entities} x n_periods={n_periods} (expected "
    "{expected} observations)"
)

# `PanelError::ZeroVarianceAfterDemeaning`（6.7節）。
ZERO_VARIANCE_AFTER_DEMEANING = (
    "regressor '{column}' has zero variance after the within-"
    "transformation (it is time-invariant, or collinear with the fixed "
    "effects)"
)

# `PanelError::TwoWayRequiresTime`。
TWO_WAY_REQUIRES_TIME = (
    "two-way fixed effects requires the `time` option to be set"
)

# `PanelError::HacRequiresTime`（Driscoll-Kraay HAC、1-way限定で到達）。
HAC_REQUIRES_TIME = (
    "Driscoll-Kraay panel HAC requires the `time` option to be set"
)

# `PanelError::InvalidHacBandwidth`。`t`は時点数（観測数`n`ではない点に
# 注意、OLSの`INVALID_HAC_LAGS`とは上限の意味が異なる）。
INVALID_HAC_BANDWIDTH = (
    "bandwidth must be in the range [0, t): got {bandwidth}, t={t}"
)

# FE用cov_type文字列パース（engine_pybind/src/panel/fe.rs::parse_fe_cov_type）。
# OLS/WLS/IVの`UNKNOWN_COV_TYPE_LINEAR`と異なりhc0を含まない一覧になる。
UNKNOWN_COV_TYPE_FE = (
    "unknown cov_type: '{other}'. Expected one of 'classical', 'hc1' "
    "through 'hc3', 'cluster', or 'hac'"
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

# `IvError::FirstStageFailed`（engine/src/iv/common.rs）。`engine_pybind::fit()`
# （engine_pybind/src/iv/common.rs）が`TwoSlsEstimator::fit`/`GmmEstimator::fit`
# を呼ぶより前に無条件で`compute_first_stage`（弱操作変数診断用）を呼ぶため、
# 第一段階回帰由来の`ValidationError`（`InsufficientObservations`・
# `InsufficientClustersForInference`等）は常にこのラッパー経由で観測される。
# 構造方程式自身のqを使う`TwoSlsEstimator::fit`/`GmmEstimator::fit`冒頭の同種
# 事前チェックはPython APIからは実質到達不能（第一段階のqは
# 識別条件`instruments>=x_endog`により常に構造方程式のq以上のため、第一段階側の
# チェックが必ず先に発火する）。修正は別Issueで検討。
FIRST_STAGE_FAILED = "first stage regression for endogenous variable '{endog_name}' failed: {source}"
