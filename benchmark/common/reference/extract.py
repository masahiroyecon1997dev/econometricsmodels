"""リファレンス実装のfit結果から統計量を取り出す抽出ヘルパー。"""

from __future__ import annotations

from typing import Any


def extract_params_std_errors(model: Any) -> dict[str, dict[str, float]]:
    """statsmodelsのfit結果から係数・標準誤差をプレーンなdictで取り出す。

    `model.params`/`model.bse`（pandas Series）を`{パラメータ名: 値}`の
    dictに変換する。フィクスチャ生成スクリプトの結果辞書組み立てで
    `params`/`std_errors`の2キーとして`**`展開して使う。

    Args:
        model: statsmodelsのfit結果（`params`/`bse`属性を持つもの）。

    Returns:
        `{"params": {名前: 係数}, "std_errors": {名前: 標準誤差}}`。
    """
    return {
        "params": {
            str(k): float(v) for k, v in model.params.to_dict().items()
        },
        "std_errors": {
            str(k): float(v) for k, v in model.bse.to_dict().items()
        },
    }
