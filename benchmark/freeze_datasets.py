"""ベンチマーク・テストで使う合成入力データセットを、CSVとして固定（凍結）するスクリプト。

`generate_linear_datasets.py`（seed固定・決定論的）は「データがどう作られたか」の
再現可能なコードとして残すが、フィクスチャ生成・pytestの実行時にこれを毎回呼び出す
設計だと、ジェネレータ側のコードが将来変わったときに、既に固定した
`tests/api_tests/fixtures/benchmarks/*.json`の期待値と無言で不整合になる
（同じseedでも呼び出し順序やパラメータが変われば出力が変わるため）。

このスクリプトは、現在frozen対象になっている合成データを一度だけCSVとして
`tests/api_tests/fixtures/benchmarks/data/`に書き出す。以後の通常運用では
このスクリプトは呼ばれない（フィクスチャJSON同様、意図的に更新する場合のみ
手動で再実行する）。

実際のシナリオ定義・生成処理は系統ごとに分割している（手法が増えるたびに本ファイルが
肥大化するのを避けるため）。このファイルは各系統の`freeze()`を呼び出す薄いディスパッチャに
徹する。

- `benchmark/linear/freeze_linear_datasets.py`: 連続y（OLS/WLS用、
  `generate_linear_datasets.py`）
- `benchmark/nonlinear/freeze_nonlinear_datasets.py`: 2値y（Logit/Probit用、
  真のlogit/probit DGP、`generate_nonlinear_datasets.py`）
- `benchmark/iv/freeze_iv_datasets.py`: IV（2SLS/GMM用、`generate_iv_datasets.py`）

**Wooldridgeデータセットはここでは固定しない**（`wooldridge`パッケージ自体は
MITライセンスだが、同梱される実データの著作権はWooldridge『Introductory
Econometrics』教科書側にある可能性があり、フィルタ後の部分集合であっても
MITライセンスの本リポジトリにCSVとしてコミットして再配布してよいか未確認の
ため。ユーザー確認済み）。Wooldridgeデータは引き続き`load_wooldridge.py`経由で
都度ロードする（`run_statsmodels_benchmark.py`・各`generate_*_crosscheck_fixtures.py`・
`tests/api_tests/test_ols_crosscheck.py`参照）。

使用例:
    python freeze_datasets.py --output-dir ../tests/api_tests/fixtures/benchmarks/data
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent / "linear")
)  # benchmark/linear/ を import path に追加（freeze_linear_datasets）
sys.path.insert(
    0, str(Path(__file__).resolve().parent / "nonlinear")
)  # benchmark/nonlinear/ を import path に追加（freeze_nonlinear_datasets）
sys.path.insert(
    0, str(Path(__file__).resolve().parent / "iv")
)  # benchmark/iv/ を import path に追加（freeze_iv_datasets）

from _common import run_freeze_cli  # noqa: E402
from freeze_iv_datasets import freeze as _freeze_iv  # noqa: E402
from freeze_linear_datasets import freeze as _freeze_linear  # noqa: E402
from freeze_nonlinear_datasets import freeze as _freeze_nonlinear  # noqa: E402


def freeze(output_dir: Path) -> None:
    _freeze_linear(output_dir)
    _freeze_nonlinear(output_dir)
    _freeze_iv(output_dir)


if __name__ == "__main__":
    run_freeze_cli(
        freeze,
        "../tests/api_tests/fixtures/benchmarks/data",
        "wrote frozen datasets",
        description=__doc__,
    )
