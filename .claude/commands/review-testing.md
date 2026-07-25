---
description: testing-completeness-reviewerサブエージェントに委譲してテストの網羅性をレビューする（カバレッジ率では見えない構造的な抜けの検出）
argument-hint: [対象の推定手法名（省略時は直近のgit diffから推定）]
allowed-tools: Bash(git diff:*), Bash(git log:*)
---

# テスト網羅性レビュー

`testing-completeness-reviewer` サブエージェント（`.claude/subagents/testing-completeness-reviewer.md`）にレビューを委譲する。
チェック観点・出力形式の詳細はサブエージェント側の定義を単一ソースとする（このコマンドでは重複定義しない）。

## レビュー対象

$ARGUMENTS が指定されていればその手法を対象にする。指定がなければ直近の `git diff` から対象手法を推定する。

## 手順

1. `testing-completeness-reviewer` サブエージェントに対象（$ARGUMENTS または直近の `git diff`）を渡してレビューを依頼する。
2. サブエージェントからの指摘結果をそのままユーザーに提示する。
3. 対応が必要な指摘があれば `/test-new`・`/implement-python`・`/implement-rust` の利用を提案する。このコマンド自体はコードを修正しない。
