"""全系統で共有する`ValidationError`メッセージのテンプレート文字列。

`engine`（`CommonError`等）・`engine_pybind`（`column_extraction.rs`・
`validation.rs`・各系統の`cov_type`/`method`文字列パース）が実際に送出する
メッセージの正確な文字列をPythonのフォーマット文字列としてここに集約する。

Rust側のメッセージ文言が正本であり（`engine/src/error.rs`・
`engine_pybind/src/column_extraction.rs`・`engine_pybind/src/validation.rs`等）、
このファイルはそのコピー。Rust側で文言を変更した場合はこのファイルも同時に
更新すること（`docs/planning/specs/test-coverage-candidates.md`項目26、
複数系統でメッセージが文字通り重複しているため直書きではなく共通化した）。

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
    "when include_intercept=true, x cannot contain a column named 'const' "
    "(it collides with the automatically added intercept)"
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
INVALID_GMM_CONVERGENCE = (
    "gmm_convergence must be a positive number, got {gmm_convergence}"
)

# `IvError::FirstStageFailed`（engine/src/iv/common.rs）。`engine_pybind::fit()`
# （engine_pybind/src/iv/common.rs）が`TwoSlsEstimator::fit`/`GmmEstimator::fit`
# を呼ぶより前に無条件で`compute_first_stage`（弱操作変数診断用）を呼ぶため、
# 第一段階回帰由来の`ValidationError`（`InsufficientObservations`・
# `InsufficientClustersForInference`等）は常にこのラッパー経由で観測される。
# 構造方程式自身のqを使う`TwoSlsEstimator::fit`/`GmmEstimator::fit`冒頭の同種
# 事前チェック（Issue #289）はPython APIからは実質到達不能（第一段階のqは
# 識別条件`instruments>=x_endog`により常に構造方程式のq以上のため、第一段階側の
# チェックが必ず先に発火する）——`docs/planning/specs/test-coverage-candidates.md`
# 項目31に記録済み、修正は別Issueで検討。
FIRST_STAGE_FAILED = "first stage regression for endogenous variable '{endog_name}' failed: {source}"
