# benchmark/

`tests/fixtures/benchmarks/`に固定するベンチマーク値（統計量の期待値）を、
リファレンス実装（statsmodels・R・pyfixest）を使って生成するための開発用ツール群です。
`tests/`とはライフサイクルが異なる（Rランタイムに依存し、随時手動実行するツールである）
ため、`tests/`とは分離しています。

ディレクトリ構成・各スクリプトの役割分担・リファレンス実装の使い分けの詳細は
[`.claude/skills/reference-benchmark/SKILL.md`](../.claude/skills/reference-benchmark/SKILL.md)
を参照してください。

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
