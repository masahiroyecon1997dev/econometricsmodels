"""FE-2SLS（`FEIV`）のダミー変数を明示した2SLSによるクロスチェック用フィクスチャ
（tests/fixtures/benchmarks/fe_iv_dummy_crosscheck.json）を生成するスクリプト。

固定効果のダミー変数を外生変数に入れた2SLS（`linearmodels.iv.IV2SLS`）を参照にする。within変換
（交互射影）を使わない独立な計算で、係数・標準誤差・共分散・Sargan・Wu-Hausman・部分F・
Cragg-Donald・Kleibergen-Paap・有効F（MOPの論文の式だけから書いた独立実装）・R²を比べる。
linearmodelsにHC2・HC3が無いため、そのケースの標準誤差だけはnumpyで書いたダミー変数2SLSが
唯一の参照になる（`docs/guide/verification-details.md`の単一リファレンスの例外）。

定義と本実装との対応（Sargan・Wu-Hausmanの尺度、KP rk LMがHC0基準であること、クラスターの
`K`の数え方）は`benchmark/panel/references/dummy_2sls_ref.py`のモジュールdocが正本。
ケースは`benchmark/panel/fe_iv_cases.py`の`dummy_cases()`。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_iv_dummy_crosscheck_fixtures
"""

from __future__ import annotations

from datetime import UTC, datetime

import linearmodels
import pandas as pd

from benchmark.common import (
    BENCHMARKS_DIR,
    group_by_dataset_source,
    run_fixture_cli,
)
from benchmark.common.cov_fixtures import strip_cov_params
from benchmark.panel.fe_iv_cases import dummy_cases
from benchmark.panel.references.dummy_2sls_ref import run_dummy_2sls


def build_fixtures() -> dict:
    fixtures: dict = {}
    for case in dummy_cases():
        df = pd.read_csv(case.csv_path)
        node = fixtures
        for key in case.path[:-1]:
            node = node.setdefault(key, {})
        node[case.path[-1]] = run_dummy_2sls(df, case)
    fixtures["_meta"] = {
        "method": "fe_iv",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "linearmodels IV2SLS (dummy variables) / numpy (HC2, HC3)",
        "linearmodels_version": linearmodels.__version__,
        "note": (
            "固定効果のダミー変数（1つ目は全水準、2つ目以降は最初の水準を落とす）を外生変数に入れた"
            "2SLS。classicalはunadjusted、hc1はrobust（どちらもdebiased=True）、cluster_timeは1-wayの"
            "個体FEでの時点クラスター。hc2・hc3はlinearmodelsに無くnumpyの手計算（有効FはMOPの論文の式）。"
            "overid_statisticはlinearmodelsのsargan（e'e/n尺度）に(n-n_absorbed)/nを掛けた本実装の尺度。"
            "wu_hausman_statisticはwooldridge_regression.statを内生変数の数で割ったF。"
            "rk_lm_statisticはhc1ではlinearmodelsがHC0基準のため比べない。"
            "r_squared_*はPanelOLS._rsquared（構造残差ベース）、2-wayのwithinはダミー射影の手計算、"
            "3-wayは置かない。"
        ),
    }
    return group_by_dataset_source(fixtures, [])


if __name__ == "__main__":
    run_fixture_cli(
        lambda: strip_cov_params(build_fixtures()),
        BENCHMARKS_DIR / "fe_iv_dummy_crosscheck.json",
        description=__doc__,
    )
