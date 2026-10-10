"""statsmodelsでBreusch-Godfrey検定（`acorr_breusch_godfrey`）のベンチマーク値を生成する。

`OLSResults.breusch_godfrey_test()`の主リファレンス。statsmodelsはサンプル前期間の
ラグを0で埋める（本実装・R `bgtest`の既定と同じ）。ただし**切片なしのモデルでは補助回帰に
定数を足す**ため、R・Greeneの定義（足さない）を採る本実装とは値が異なる。このため
切片ありのモデルだけを対象にする（切片なしはRのみで照合する）。

行の並びが時間順であることを前提にする（`generate_*`側でデータを時間順に並べて渡す）。

使用例（リポジトリルートから）:
    python -m benchmark.linear.references.statsmodels_breusch_godfrey_ref \\
        --dataset-source synthetic --dataset autocorrelated --nlags 1 4
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

from benchmark.common import load_frozen_dataset
from benchmark.common.load_wooldridge import load as load_wooldridge


def _quoted(formula: str) -> str:
    """`lhs ~ a + b`の各変数を`Q('name')`で包む（`inf`のようにpatsyがPythonの名前として
    解釈してしまう列名を、そのまま列名として扱わせるため）。"""
    lhs, rhs = formula.split("~")
    terms = [t.strip() for t in rhs.split("+")]
    return f"Q('{lhs.strip()}') ~ " + " + ".join(f"Q('{t}')" for t in terms)


def run_breusch_godfrey(
    dataset_source: str,
    dataset: str,
    formula: str | None,
    nlags: list[int],
    time_column: str | None = None,
) -> dict:
    """切片ありのOLS残差に対する`acorr_breusch_godfrey`の結果を`nlags`ごとに返す。

    Args:
        dataset_source: `"synthetic"`または`"wooldridge"`。
        dataset: データセット名。
        formula: 回帰式。`synthetic`でNoneなら`y ~ x1 + ...`を自動生成する。
        nlags: ラグ次数のリスト。
        time_column: 指定すればその列の昇順に並べてから推定する（`wooldridge`の時系列用）。
            合成データは行順が時間順。
    """
    if dataset_source == "synthetic":
        df, _ = load_frozen_dataset("synthetic", dataset)
        if formula is None:
            x_cols = [c for c in df.columns if c not in ("y", "weight")]
            formula = "y ~ " + " + ".join(x_cols)
    elif dataset_source == "wooldridge":
        df = load_wooldridge(dataset)
        if formula is None:
            raise ValueError("wooldridgeデータセットの場合はformulaが必須です")
    else:
        raise ValueError(f"unknown dataset_source: {dataset_source!r}")
    if time_column is not None:
        df = df.sort(time_column)

    model = smf.ols(formula=_quoted(formula), data=df.to_pandas()).fit()
    result: dict = {"n_obs": int(model.nobs), "nlags": {}}
    for m in nlags:
        lm, lm_p, f_value, f_p = acorr_breusch_godfrey(
            model, nlags=m, result_object=False
        )
        result["nlags"][str(m)] = {
            "lm": float(lm),
            "lm_p_value": float(lm_p),
            "f": float(f_value),
            "f_p_value": float(f_p),
        }
    result["_meta"] = {
        "reference": "statsmodels",
        "function": "statsmodels.stats.diagnostic.acorr_breusch_godfrey",
        "statsmodels_version": statsmodels.__version__,
        "generated_at": datetime.now(UTC).isoformat(),
        "formula": formula,
        "time_column": time_column,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-source", choices=["synthetic", "wooldridge"], required=True
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--formula", default=None)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--nlags", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()
    print(
        json.dumps(
            run_breusch_godfrey(
                args.dataset_source,
                args.dataset,
                args.formula,
                args.nlags,
                args.time_column,
            ),
            indent=2,
        )
    )
