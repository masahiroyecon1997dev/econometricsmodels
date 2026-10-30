"""`tests/cross_method/test_pickle.py`が別プロセス（`spawn`）で実行する関数。

`spawn`の子プロセスは関数をモジュール名で再importするため、テストファイルではなく
`pythonpath`に載る`tests/support/`に置く。
"""

from __future__ import annotations

from econometricsmodels import OLS


def fit_ols(data, options):
    """子プロセスで推定し、結果を親へ返す（結果のpickleを通す）。"""
    return OLS(data, y="y", x=["x1", "x2"], options=options).fit()


def describe(result, options):
    """親から渡された結果とオプションを使い、子プロセスでの値を返す。"""
    return {
        "params": result.params,
        "predicted": result.predict(),
        "white": result.white_test().statistic,
        "cov_type": options.cov_type,
    }
