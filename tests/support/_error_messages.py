"""全系統で共有する`ValidationError`メッセージのテンプレート文字列。

`engine`（`CommonError`等）・`engine_pybind`（`validation.rs`・各系統の`cov_type`/`solver`/
`estimator`文字列パース）が実際に送出するメッセージの正確な文字列をPythonのフォーマット
文字列としてここに集約する。

系統固有のメッセージは`_error_messages_<系統>.py`（`linear`・`nonlinear`・`iv`・
`panel`）に置き、ここには全系統（または複数の系統）が共有するものだけを置く。
テストからは`import _error_messages_<系統> as <系統>_msgs`で参照する。

Rust側のメッセージ文言が正本であり（`engine/src/shared/error.rs`・
`engine_pybind/src/shared/validation.rs`等）、このファイルはそのコピー。Rust側で
文言を変更した場合はこのファイルも同時に更新すること（複数系統でメッセージが
文字通り重複しているため直書きではなく共通化した）。

各テストは`pytest.raises(ValidationError, match=escaped(TEMPLATE, name=...))`の
形で使う（`escaped()`は`str.format(**kwargs)`した上で`re.escape`する。メッセージ中の
`()`・`.`等は正規表現の特殊文字のため、素の文字列を`match=`にそのまま渡すのは誤り）。

列の抽出で出るエラー（列が存在しない・欠損値・非有限値・許可外のdtype・同じ値・
タイムゾーン付きDatetime）は構造化例外（`code`/`params`）のため、ここにテンプレートを
置かず、`_errors.raises_error`で`code`と`params`を照合する。
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
    （`engine_pybind/src/shared/column_extraction.rs`の`extract_dataframe`）を
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


# ── column_extraction.rs（全系統共通） ──────────────────────────────
#
# extract_dataframe: `data`/`new_data`にpolars以外のDataFrame（pandas等）が
# 渡された場合に使う（engine_pybind/src/shared/column_extraction.rs）。
# `param_name`は呼び出し側で"data"（fit系）または"new_data"（predict/augment）
# を渡す。`type_name`はPythonオブジェクトの完全修飾クラス名
# （`type(obj).__module__ + "." + type(obj).__qualname__`相当）。
NOT_A_POLARS_DATAFRAME = (
    "'{param_name}' must be a polars.DataFrame, got {type_name}"
)
# `polars.LazyFrame`を渡した場合は、`NOT_A_POLARS_DATAFRAME`に`.collect()`の案内が付く。
NOT_A_POLARS_DATAFRAME_LAZY = (
    NOT_A_POLARS_DATAFRAME + "; call .collect() first"
)

# HAC(`cov_type="hac"`・IV GMMの`gmm_weight_type="hac"`)で`hac_time`が未指定の場合。
# 行順を時間順とみなす暗黙の既定は置かない。`{setting}`は`hac_time`を要求している設定名
# (`cov_type`または`gmm_weight_type`)。
HAC_REQUIRES_HAC_TIME = (
    "{setting}='hac' requires the `hac_time` option: the column that gives the "
    "time order of the observations (the row order of the data is not assumed "
    'to be the time order; add an explicit index column such as `df.with_row_index("t")` '
    "if the rows are already in time order)"
)

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

# `validate_no_duplicate_roles`のメッセージ（engine_pybind/src/shared/validation.rs
# `duplicate_role_message`）。単一列ロール（y/weight）と複数列ロール（x/x_exog/
# x_endog/instruments）の組み合わせによって主語が変わるため3パターンに分ける
# （同ファイルのdocコメント「呼び出し側の契約」参照）。
ROLE_OVERLAP_SINGLE_IN_MULTI = "the column '{col}' specified as {single_role} is also included in {multi_role}"
ROLE_OVERLAP_SINGLE_EQUALS_SINGLE = "the column '{col}' specified as {later_role} is also specified as {earlier_role}"
ROLE_OVERLAP_MULTI_VS_MULTI = "the column '{col}' specified as {later_role} is also included in {earlier_role}"

# ── CommonError（engine/src/shared/error.rs、OLS/WLS/Tobit/Logit/Probit/IV共通） ──

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


# ── 複数系統が使う（系統固有の節に由来するが、系統をまたいで共有） ──

INVALID_HAC_LAGS = (
    "hac_lags must be in the range [0, n): got {hac_lags}, n={n}"
)

# cov_type文字列パース。OLS/WLS/IVは文言が完全一致（3手法とも
# engine_pybind/src/linear/common.rs::parse_cov_kindが検査する）。
UNKNOWN_COV_TYPE_LINEAR = (
    "unknown cov_type: '{other}'. Expected one of 'classical', 'hc0' "
    "through 'hc3', 'hac', or 'cluster'"
)

# 選んだモードで使われないオプションが明示指定された場合の共通文言
# （`engine_pybind/src/shared/validation.rs`の`reject_unused_option`）。
UNUSED_OPTION = (
    "{option} is only used with {condition}, so it would be silently "
    "ignored; set {condition} or remove {option}"
)
