---
name: review-performance
description: performance-reviewerサブエージェントに委譲して性能比較コードをレビューする（計測方法論の妥当性・恣意性の排除・Python文法/規約・ドキュメント整合）
argument-hint: "[レビュー対象ファイル（省略時はgit diff）]"
allowed-tools: Bash(git diff:*), Bash(git log:*)
---

# パフォーマンス比較コードのレビュー

`performance-reviewer` サブエージェント（`.claude/agents/performance-reviewer.md`）にレビューを委譲する。
チェック観点・出力形式の詳細はサブエージェント側の定義を単一ソースとする（このコマンドでは重複定義しない）。

## レビュー対象

$ARGUMENTS が指定されていればそのファイルを対象にする。指定がなければ `git diff` の変更内容（`performance/` 配下）を対象にする。

## 手順

1. `performance-reviewer` サブエージェントに対象（$ARGUMENTS または直近の `git diff`）を渡してレビューを依頼する。
2. サブエージェントからの指摘結果をそのままユーザーに提示する。
3. 対応が必要な指摘があれば、メインセッションでの修正（または `/implement-python`）を提案する。このコマンド自体はコードを修正しない。
