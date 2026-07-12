---
paths:
  - python_package/**/*
---

# Pythonコーディング規約（python_package）

このファイルは `python_package/` 配下で作業する際に自動的に読み込まれる。CLAUDE.mdの非交渉事項（2章）を前提とした、Python実装の詳細規約。

## 型ヒント・docstring

- 全public関数・クラスに **型ヒントを必須** で付与する。
- 全public関数・クラスに **Googleスタイルのdocstringを必須** で付与する。

## Lint / フォーマット

- Ruffの **line-length は79（PEP8標準）**。
- `ruff check .` / `ruff format --check .` をエラーなしで通す。
- ルールセットの詳細（有効化するRuffルール群）はリポジトリ雛形作成時に`pyproject.toml`で確定する。

## 設計方針との整合性

- データ入力は **polarsのみ**。pandas等への変換・依存を持ち込まない。
- 変数は **List渡し**、推定オプションは **オブジェクト渡し**。formula文字列パースを実装しない。
- `engine_pybind` からの呼び出しを薄くラップする。計算ロジックをPython側に持たない。

## テスト

- 対応するリファレンス実装（pyfixest/R）との比較テストは `tests/api_tests/`（pytest）に置く。
- 許容誤差等のテスト方針の詳細は `testing-policy.md` を参照。
