---
description: rust-reviewerサブエージェントに委譲してコードレビューを行う（規約準拠・設計方針整合性・パフォーマンス）
argument-hint: [レビュー対象ファイル（省略時はgit diff）]
allowed-tools: Bash(git diff:*), Bash(git log:*)
---

# Rustレビュー

`rust-reviewer` サブエージェント（`.claude/agents/rust-reviewer.md`）にレビューを委譲する。
チェック観点・出力形式の詳細はサブエージェント側の定義を単一ソースとする（このコマンドでは重複定義しない）。

## レビュー対象

$ARGUMENTS が指定されていればそのファイルを対象にする。指定がなければ `git diff` の変更内容を対象にする。

## 手順

1. `rust-reviewer` サブエージェントに対象（$ARGUMENTS または直近の `git diff`）を渡してレビューを依頼する。
2. サブエージェントからの指摘結果をそのままユーザーに提示する。
3. 修正が必要な指摘があれば `/implement-rust` の利用を提案する。このコマンド自体はコードを修正しない。
