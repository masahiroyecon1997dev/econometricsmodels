# benchmark/

`tests/fixtures/benchmarks/`に固定するベンチマーク値（統計量の期待値）を、
リファレンス実装（statsmodels・R）を使って生成するための開発用ツール群です。
`tests/`とはライフサイクルが異なる（Rランタイムに依存し、随時手動実行するツールである）
ため、`tests/`とは分離しています。

**性能比較ツール（旧 `benchmark/performance/`）はリポジトリ直下の
[`performance/`](../performance/) に分離しました**（`benchmark_performance.yml`
専用で、pytest とは無関係な随時実行ツールであり、正確性検証用の本ディレクトリとは
性質が違うため。`docs/planning/specs/benchmark-restructure-design.md` D5）。

ディレクトリ構成・各スクリプトの役割分担・リファレンス実装の使い分けの詳細は
[`.claude/skills/reference-benchmark/SKILL.md`](../.claude/skills/reference-benchmark/SKILL.md)
を参照してください。

## 実行方法（Initiative A でパッケージ化）

`benchmark/` は `__init__.py` を持つ Python パッケージです。スクリプトは
**リポジトリルートから `-m` で**実行します（各ディレクトリへ `cd` して
`python foo.py` とは実行できません）。

```
python -m benchmark.linear.freeze                     # linear系のCSVのみ凍結
python -m benchmark.linear.fixtures.generate_ols_fixtures
python -m benchmark.regenerate_all                    # 全系統CSV + 全JSONフィクスチャ
python -m benchmark.regenerate_all --datasets-only    # 全系統CSVのみ（Rscript不要）
```

各系統は `datasets.py`（DGP）／`freeze.py`（CSV凍結）／`references/`（リファレンス
実装アダプタ・`.R`）／`fixtures/`（`generate_*_fixtures.py`）で構成。系統をまたぐ
共通ヘルパーは `benchmark/common/` に集約。再設計の全体像は
[`docs/planning/specs/benchmark-restructure-design.md`](../docs/planning/specs/benchmark-restructure-design.md)。

## ライセンスに関する注記

- 本リポジトリ本体（`engine` / `engine_pybind` / `python_package`。PyPIで配布される
  wheel/sdistの中身）は[MITライセンス](../LICENSE)です。
- `benchmark/`配下のRスクリプト（`*.R`）は、独立実装によるクロスチェック用に以下の
  Rパッケージを使用します。
  - `fixest` / `plm` / `ivreg` / `sandwich` / `lmtest`: GPL-2 / GPL-3（パッケージにより異なる）
  - `jsonlite`: MIT
- これらのRパッケージはPyPI配布物には一切含まれません。Pythonスクリプトから
  `subprocess`経由で別プロセスの`Rscript`を呼び出しているだけで、リンク・
  同梱・配布のいずれも行っていないため、本リポジトリのMITライセンスに対する
  ライセンス上の制約は生じません。ライセンス上必須ではありませんが、透明性の
  ため依存関係を明示しています。

## 実行に必要な環境

- **R本体（r-base）と上記Rパッケージの別途インストールが必要です**。
  devcontainer環境には`.devcontainer/Dockerfile`で導入済みですが、devcontainer外で
  直接これらのスクリプトを実行する場合は各自インストールしてください。
- `pytest tests/`の実行自体にはRは不要です。合成データのクロスチェック
  フィクスチャは`tests/fixtures/benchmarks/`にJSONとして固定済みで、
  pytestはそれを読むだけだからです。Rが必要になるのは、
  `benchmark/<系統>/fixtures/generate_*_crosscheck_fixtures.py`でこれらの
  フィクスチャを再生成する場合のみです。
