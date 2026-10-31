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


def cov_frame_to_dict(cov: Any) -> dict[str, dict[str, float]]:
    """分散共分散行列のpandas DataFrameを`{行の名前: {列の名前: 値}}`にする。

    statsmodelsの`cov_params()`のほか、linearmodelsの`res.cov`のように、行列が
    メソッドでなく属性のDataFrameで得られるリファレンスにも使う。
    """
    return {
        str(row): {str(col): float(value) for col, value in values.items()}
        for row, values in cov.iterrows()
    }


def extract_cov_params(model: Any) -> dict[str, dict[str, float]]:
    """statsmodelsのfit結果から係数の分散共分散行列を入れ子のdictで取り出す。

    `model.cov_params()`（pandas DataFrame）を`{行の名前: {列の名前: 値}}`にする。
    fit時に指定した`cov_type`の行列（robust/cluster/HACでも`bse`の2乗が対角に並ぶ）。
    名前は生のまま返す（切片名の`"const"`への正規化は呼び出し側
    〔`normalize_cov_params`〕が行う）。

    Args:
        model: statsmodelsのfit結果（`cov_params()`を持つもの）。

    Returns:
        `{名前: {名前: 値}}`。行・列とも`model.params`と同じ順。
    """
    return cov_frame_to_dict(model.cov_params())
