---
paths:
  - python_package/**/*
---

# Pythonコーディング規約（python_package）

このファイルは `python_package/` 配下で作業する際に自動的に読み込まれる。CLAUDE.mdの非交渉事項（2章）を前提とした、Python実装の詳細規約。

## 型ヒント・docstring

- 全public関数・クラスに **型ヒントを必須** で付与する。
- 全public関数・クラスに **Googleスタイルのdocstringを必須** で付与する。
- **docstringは英語にする**（モジュール・クラス・関数・メソッド・プロパティの`"""docstring"""`全て）。`rust-style.md`「言語方針」が`engine_pybind`の`#[pyclass]`/`#[pyfunction]`のdocコメント（PyO3経由でPythonの`__doc__`になるもの）を英語にする方針と揃え、mkdocs（mkdocstrings）で同じページにレンダリングされるRust側・Python側のdocstringで言語が混在しないようにする（発覚した不整合の是正）。PyPI公開の独立パッケージであり、Pythonエコシステムの慣習（pandas/numpy/polars等）に合わせる、という`rust-style.md`と同じ理由。
- **日本語のままでよい**: docstring以外の非公開実装コメント（実装の背景説明、設計判断の理由、TODOコメント等）。`rust-style.md`と同じ区別。

## Lint / フォーマット

- Ruffの **line-length は79（PEP8標準）**。
- `ruff check .` / `ruff format --check .` をエラーなしで通す。
- ルールセットの詳細（有効化するRuffルール群）はリポジトリ雛形作成時に`pyproject.toml`で確定する。

## 設計方針との整合性

- データ入力は **polarsのみ**。pandas等への変換・依存を持ち込まない。
- 変数は **List渡し**、推定オプションは **オブジェクト渡し**。formula文字列パースを実装しない。
- `engine_pybind` からの呼び出しを薄くラップする。計算ロジックをPython側に持たない。

## 開発ツールの実行方針

- **`uvx`（`uvx <tool> ...`）は使わない**。`uvx`は実行のたびにPyPIから解決するため、バージョンが実行タイミングで変わりうる上、ハッシュ検証がない（サプライチェーン攻撃への露出面が広い）。
- maturin・statsmodels等、CLIとして実行する開発ツールは`pyproject.toml`の`[dependency-groups] dev`にバージョン固定（`==`）で追加し、`uv lock`でハッシュごと`uv.lock`に記録する。実行は`uv run <tool> ...`（例: `uv run maturin build`）を使う。
- `[build-system] requires`（PEP 517、`pip install .`時に使われる方）も、範囲指定（`>=1.0,<2.0`等）ではなく`==`で固定する。
- バージョンを上げる際は、`uv lock --upgrade-package <package>`を使い、事前に差分・changelogを確認してから行う（無警戒に最新へ追従しない）。

## テスト

- 対応するリファレンス実装（pyfixest/R）との比較テストは `tests/api_tests/`（pytest）に置く。
- 許容誤差等のテスト方針の詳細は `testing-policy.md` を参照。
