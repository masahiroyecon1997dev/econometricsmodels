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

構造化例外（`code`/`params`）になったエラーは、ここにテンプレートを置かず、
`_errors.raises_error`で`code`と`params`を照合する。列の抽出で出るエラー（列が存在しない・
欠損値・非有限値・許可外のdtype・同じ値・タイムゾーン付きDatetime）、ロールの検査
（空・列の重複・ロール間の重複・`const`との衝突・出力列の衝突）、オプション値の検査
（未知の値・使われないオプション・`hac_time`/`dk_time`の欠落・パネルで使えない
`cov_type`・`confidence_level`/`hac_lags`/`dk_bandwidth`/`max_iter`/`gmm_max_iter`/
`tol`/`gmm_tol`/`nlags`の範囲違反）がこれにあたる。未知の値のエラー（`OPTION_UNKNOWN_VALUE`）が持つ
`allowed`（受け付ける値の一覧）の期待値だけを、`ALLOWED_*`として各ファイルに置く。
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

# ── 複数系統が使う（系統固有の節に由来するが、系統をまたいで共有） ──

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
