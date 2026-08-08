---
name: cicd
description: CI/CDワークフローの作成・編集、またはCI失敗時の原因調査を支援する
argument-hint: [作成/編集したい内容 or 失敗調査の対象]
allowed-tools: Bash(gh:*), Bash(git:*), Read, Write, Edit, Grep, Glob
---

# CI/CD支援

対応するCLAUDE.mdの方針: 8章（バージョニング・CI/CD）

## 対象

$ARGUMENTS

## 手順

まず `$ARGUMENTS` の内容から、以下のどちらの依頼かを判断する。両方に該当する場合は順番に対応する。

### A. ワークフローファイルの新規作成・編集の場合

1. 対象が以下のどれかを確認する。
   - `ci_engine.yml`: `cargo test` / `clippy` / `fmt`（`-p engine`）・`cargo audit`。`engine/` 配下の変更をトリガーとする。
   - `ci_python.yml`: `pytest` / `Ruff`・`engine_pybind`のclippy/fmt・`pip-audit`。`python_package/` `engine_pybind/` 配下の変更をトリガーとする。
   - `cd_release.yml`: maturin-actionでのマルチOS（Linux/macOS/Windows）wheelビルド。タグpush + `workflow_dispatch`トリガー。
   - `benchmark_ols.yml`: `benchmark/compare_performance.py`の定期実行、job summaryへの結果出力。タグpush + `workflow_dispatch`トリガー。
   - `cd_docs.yml`: mkdocs → GitHub Pages（未実装）。
2. 既存のワークフローファイルがあれば内容を読み、既存の構成・命名規則に合わせる。
3. path-triggerを適切に設定し、無関係な変更でCIが走らないようにする（engine側とpython側のワークフローを混在させない）。
4. 変更内容を提示し、問題なければファイルに反映する。

### B. CI失敗の調査の場合

1. `gh run list` で直近の実行状況を確認する。
2. `gh run view --log-failed` 等で失敗したジョブのログを取得する。
3. ログから原因を特定する（コンパイルエラー、clippy/Ruff違反、テスト失敗、依存関係の問題等）。
4. 原因と修正案を提示する。実際のコード修正が必要な場合は `/implement-python` または `/implement-rust` の利用を提案する。

## 注意

- ワークフローファイルの変更内容は必ず提示し、ユーザーの確認を得てから反映する。
- push・force-push等、破壊的なgit操作は行わない。
