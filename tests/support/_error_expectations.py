"""構造化例外の照合に使う、期待値の定数とTypeErrorの文面。

`ValidationError`/`ComputationError`は文面ではなく`code`と`params`で照合する
（`_errors.raises_error`）。文面そのものはRust側の単体テストと
`tests/cross_method/test_error_catalog.py`が固定する。ここに置くのは次の2種類だけ。

- `OPTION_UNKNOWN_VALUE`・`COV_TYPE_NOT_AVAILABLE_FOR_MODEL`が`params`に持つ
  `allowed`（受け付ける値の一覧）の期待値（`ALLOWED_*`）。
- 構造化されない`TypeError`の文面（`data`・列名・オプションの型違い）のテンプレート。

`TypeError`の文面は`engine_pybind/src/shared/{column_extraction,option_values}.rs`が正本。
`escaped()`は`str.format(**kwargs)`した上で`re.escape`する（`pytest.raises(match=...)`は
正規表現のため、`()`・`.`等を含む素の文字列をそのまま渡すのは誤り）。
"""

from __future__ import annotations

import re


def escaped(template: str, /, **kwargs: object) -> str:
    """`template.format(**kwargs)`を`pytest.raises(match=...)`用に`re.escape`する。"""
    return re.escape(template.format(**kwargs))


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
    if module in ("builtins", "__main__") or module.startswith("polars"):
        # polarsの型は内部のモジュール経路を隠し、クラス名だけで示す。
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
    "{param_name} must be a polars.DataFrame, got {type_name}."
)
# pandasの`DataFrame`は`{type_name}`を使わず、変換方法の案内が付く。
NOT_A_POLARS_DATAFRAME_PANDAS = (
    "{param_name} must be a polars.DataFrame, got pandas.DataFrame. "
    "Convert it with polars.from_pandas(df)."
)
# `polars.LazyFrame`には`.collect()`の案内が付く。
NOT_A_POLARS_DATAFRAME_LAZY = (
    "{param_name} must be a polars.DataFrame, got LazyFrame. "
    "Call .collect() first."
)

# ── 構造化されない`TypeError`の文面（`engine_pybind/src/shared/`が正本） ──
#
# 引数名・オプション名は`{name}`、実際の型は`{type_name}`（完全修飾名）。
COLUMN_NAME_NOT_STR = "{name} must be a str column name, got {type_name}."
COLUMN_LIST_NOT_LIST = (
    "{name} must be a list of column names, got {type_name}. "
    'Pass a list, for example {name}=["x1"].'
)
COLUMN_LIST_ELEMENT_NOT_STR = (
    "{name}[{index}] must be a str column name, got {type_name}."
)
OPTION_NOT_INT = "{name} must be an int, got {type_name}."
OPTION_NOT_REAL = "{name} must be a real number, got {type_name}."
OPTION_NOT_STR = "{name} must be a str, got {type_name}."
OPTIONS_NOT_INSTANCE = (
    "{name} must be an instance of {class_name}, got {type_name}."
)

# ── 受け付ける値の一覧（`allowed`の期待値） ──

# `OPTION_UNKNOWN_VALUE`の`allowed`（未知の`cov_type`のとき、受け付ける値の一覧）。
# OLS/WLS/IVは同じ一覧（engine_pybind/src/linear/common.rs::parse_cov_kind）。
ALLOWED_COV_TYPES_LINEAR = [
    "classical",
    "hc0",
    "hc1",
    "hc2",
    "hc3",
    "hac",
    "cluster",
]


# White・Breusch-Pagan・Breusch-Godfrey検定の未知の`statistic`に対する
# `OPTION_UNKNOWN_VALUE`の`allowed`。
ALLOWED_DIAGNOSTIC_STATISTICS = ["lm", "f"]

# Breusch-Pagan検定。`variables`の型の誤りは`TypeError`（`extract_column_list`）。
VARIABLES_NOT_A_LIST = (
    "variables must be a list of column names, got {type_name}. "
    'Pass a list, for example variables=["x1"].'
)

VARIABLES_ELEMENT_NOT_STR = (
    "variables[{index}] must be a str column name, got {type_name}."
)

# `OPTION_UNKNOWN_VALUE`の`allowed`（受け付ける値の一覧）。
ALLOWED_ESTIMATORS = ["2sls", "gmm"]
ALLOWED_GMM_TYPES = ["one_step", "two_step", "iterated"]
ALLOWED_GMM_WEIGHT_TYPES = ["classical", "robust", "cluster", "hac"]

# `OPTION_UNKNOWN_VALUE`の`allowed`（受け付ける値の一覧）。
ALLOWED_COV_TYPES_NONLINEAR = ["classical", "opg", "hc0", "hc1", "cluster"]
ALLOWED_SOLVERS = ["newton", "bfgs", "lbfgs"]
ALLOWED_MARGINAL_EFFECTS_AT = ["overall", "mean", "median"]
# Tobitの`marginal_effects()`・`predict()`・`augment()`の`target`。
ALLOWED_MARGINAL_EFFECTS_TARGET = [
    "expected_latent",
    "expected_observed",
    "prob_uncensored",
    "expected_uncensored",
]

# `OPTION_UNKNOWN_VALUE`・`COV_TYPE_NOT_AVAILABLE_FOR_MODEL`の`allowed`
# （FE/REが受け付ける`cov_type`と`variance_method`。`hc0`は含まない）。
ALLOWED_COV_TYPES_PANEL = ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]
ALLOWED_VARIANCE_METHODS = ["swamy_arora", "wallace_hussain", "amemiya"]
