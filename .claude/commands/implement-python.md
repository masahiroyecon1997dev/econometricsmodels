---
description: python_package側の実装計画立案・実装・規約チェックまでを一括で行う
argument-hint: [実装したい内容]
allowed-tools: Read, Write, Edit, Bash(ruff check:*), Bash(ruff format:*), Bash(pytest:*), Grep, Glob
---

# Python実装

対応するCLAUDE.mdの方針: 2章（絶対に守るべき設計方針）、6章（コーディング規約: Python）

## 実装対象

$ARGUMENTS

## 手順

1. **実装計画の提示**
   - 変更・追加するファイル（`python_package/econometricsmodels/` 配下等）を列挙する。
   - `engine_pybind` からの呼び出し方、polarsラッパーとしてのインターフェースを確認する。
   - 2章の非交渉事項（データ入力はpolarsのみ、formula文字列パース不採用でList渡し＋オブジェクト渡し）に抵触しないか確認する。抵触する可能性がある場合は実装前にユーザーへ確認する。

2. **実装**
   - 全public関数・クラスに型ヒントを必須で付与する。
   - 全public関数・クラスにGoogleスタイルのdocstringを付与する。

3. **実装後の規約チェック**
   - `ruff check .` を実行し、違反があれば修正する。
   - `ruff format --check .` を実行し、フォーマット崩れがあれば `ruff format .` で修正する。
   - 関連する既存テストがあれば `pytest` で実行し、デグレがないか確認する。

## 完了条件

- `ruff check` / `ruff format --check` ともにエラーなし
- 型ヒント・docstringが揃っている
- CLAUDE.mdの設計方針からの逸脱がない
