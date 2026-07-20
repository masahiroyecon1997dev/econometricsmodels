---
description: 既存テスト（engine / api_tests）を実行し、失敗があれば原因を調査する
argument-hint: [テスト対象パターン（省略可）]
allowed-tools: Bash(cargo test:*), Bash(pytest:*), Read, Grep, Glob
---

# 既存テストの実行・失敗調査

## 対象

$ARGUMENTS が指定されていればそのパターンに絞る。指定がなければ全テストを実行する。

## 手順

1. `cargo test -p engine $ARGUMENTS` を実行する。
2. `pytest tests/api_tests $ARGUMENTS` を実行する（`api_tests`）。
3. 失敗があれば、以下の観点で原因を切り分ける。
   - 実装のバグか
   - リファレンス実装（pyfixest/R）との前提の違い（計算方法の差異）か
   - 許容誤差の設定が厳しすぎる／緩すぎるか
4. 原因の仮説と対応方針を提示する。
   - 実装修正が必要な場合は `/implement-python` または `/implement-rust` の利用を提案する。
   - 許容誤差の見直しが必要な場合は根拠とあわせて提案し、テストコード自体の修正はユーザーの確認を得てから行う。

## 出力形式

- 実行結果のサマリ（成功/失敗件数）
- 失敗したテスト一覧
- 各失敗の原因の仮説
- 対応方針の提案
