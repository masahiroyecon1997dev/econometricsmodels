---
description: 新しい推定手法（OLS, Logit等）を追加する際に、engine/engine_pybind/python_packageのボイラープレートをテンプレートから生成する。新手法の実装に着手する際に使用する。
argument-hint: [手法名]
allowed-tools: Read, Write, Edit, Grep, Glob
---

# 推定手法スキャフォールド

新しい推定手法を追加する際、`engine` → `engine_pybind` → `python_package` の3層に一貫したボイラープレートを生成する。

## 前提テンプレート

- `templates/engine_module.rs.template`: engine側の推定モジュール雛形
- `templates/engine_pybind_binding.rs.template`: PyO3バインディング雛形
- `templates/python_wrapper.py.template`: Pythonラッパー雛形

> **TODO（ユーザー提供待ち）**: 上記3ファイルは現時点でプレースホルダーです。実際のボイラープレート内容（命名規則、共通トレイト/基底クラスの形、エラー型の使い方の具体例等）をユーザーから提供してもらい次第、このSKILL.mdと合わせて更新する。

## 手順

1. `$ARGUMENTS`（手法名）を確認し、CLAUDE.md 4章のどのPhaseに属するか確認する。
2. `.claude/rules/rust-style.md` `.claude/rules/python-style.md` を踏まえ、テンプレートを対象手法向けに複製・置換する。
3. 生成先:
   - `engine/src/<method>.rs`
   - `engine_pybind/src/<method>.rs`（または`lib.rs`への追記）
   - `python_package/econometricsmodels/<method>.py`（または該当モジュールへの追記）
4. 生成後、テストの雛形が必要であれば `/test-new` の利用を提案する。
5. 生成したコードの実装の詰め（実際のロジック記述）は `/implement-rust` `/implement-python` に引き継ぐ。このスキルはあくまで雛形生成に留める。
