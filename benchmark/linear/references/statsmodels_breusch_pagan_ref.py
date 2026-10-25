"""statsmodelsでBreusch-Pagan検定（`het_breuschpagan`）のベンチマーク値を生成する。

`OLSResults.breusch_pagan_test()`の主リファレンス。`robust=True`（既定、Koenkerの
標準化版 `LM = n * R^2`）を使う。補助回帰の説明変数`exog_het`には常に定数列を足して渡す
（本実装は`include_intercept`に関わらず補助回帰に定数を含める。statsmodelsは`exog_het`に
定数が無いと補助回帰に定数を入れない）。

`het_breuschpagan`はLMのp値の自由度を`exog_het`の列数-1で数え、列のランクを見ない。
このため定数列・重複列を含む`Z`は、除いた後の列（`reference_variables`）を渡す。

使用例（リポジトリルートから）:
    python -m benchmark.linear.references.statsmodels_breusch_pagan_ref \\
        --dataset-source wooldridge --dataset hprice1 \\
        --formula "price ~ lotsize + sqrft + bdrms"
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import polars as pl
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

from benchmark.common.load_wooldridge import load as load_wooldridge


def run_breusch_pagan(
    df: pl.DataFrame,
    y_col: str,
    x_cols: list[str],
    variables: list[str],
    include_intercept: bool = True,
) -> dict:
    """OLSの残差に対する`het_breuschpagan(robust=True)`の結果を返す。

    Args:
        df: データ。
        y_col: 被説明変数の列名。
        x_cols: モデルの説明変数。
        variables: 補助回帰の変数`Z`（定数列は`het_breuschpagan`の側で足す）。
        include_intercept: Falseなら切片なしのモデルの残差を使う。補助回帰には
            常に定数を含める。
    """
    y = df[y_col].to_numpy()
    x = df.select(x_cols).to_numpy()
    if include_intercept:
        x = sm.add_constant(x, has_constant="add")
    resid = sm.OLS(y, x).fit().resid

    exog_het = sm.add_constant(
        df.select(variables).to_numpy(), has_constant="add"
    )
    lm, lm_p, f_value, f_p = het_breuschpagan(resid, exog_het, robust=True)

    return {
        "lm": float(lm),
        "lm_p_value": float(lm_p),
        "f": float(f_value),
        "f_p_value": float(f_p),
        "df": len(variables),
        "n_obs": len(y),
        "_meta": {
            "reference": "statsmodels",
            "function": "statsmodels.stats.diagnostic.het_breuschpagan",
            "robust": True,
            "statsmodels_version": statsmodels.__version__,
            "generated_at": datetime.now(UTC).isoformat(),
            "y": y_col,
            "x": x_cols,
            "variables": variables,
            "include_intercept": include_intercept,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--formula", required=True, help="例: 'y ~ x1 + x2'")
    parser.add_argument("--variables", nargs="*", default=None)
    parser.add_argument("--no-intercept", action="store_true")
    args = parser.parse_args()
    lhs, rhs = args.formula.split("~")
    model_x = [t.strip() for t in rhs.split("+")]
    print(
        json.dumps(
            run_breusch_pagan(
                load_wooldridge(args.dataset),
                lhs.strip(),
                model_x,
                args.variables or model_x,
                include_intercept=not args.no_intercept,
            ),
            indent=2,
        )
    )
