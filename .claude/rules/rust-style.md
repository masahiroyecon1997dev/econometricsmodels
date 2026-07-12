---
paths:
  - engine/**/*
  - engine_pybind/**/*
---

# Rustコーディング規約（engine / engine_pybind）

このファイルは `engine/` `engine_pybind/` 配下で作業する際に自動的に読み込まれる。CLAUDE.mdの非交渉事項（2章）を前提とした、Rust実装の詳細規約。

## 責務分離

- `engine`: 純粋Rustロジック。PyO3に依存しない。
- `engine_pybind`: PyO3の薄いバインディング層。`#[pymodule]`を定義し`engine`の関数を呼ぶだけに留める。計算ロジックを`engine_pybind`に書かない。

## エラーハンドリング

- 独自エラー型は **thiserror** で定義する。
- `engine_pybind`（PyO3境界）で、thiserrorのエラー型を`PyErr`に変換する。
- `unwrap` / `expect` はプロトタイプ段階を除き避ける。避けられない場合は理由をコメントで明記する。

## Lint / フォーマット

- `cargo clippy --all-targets -- -D warnings` で警告ゼロを基準とする。
- `cargo fmt --check` でフォーマット崩れを許容しない。
- 具体的なlintレベルの追加設定（`clippy.toml`等）はリポジトリ雛形作成時に確定する。

## パフォーマンス

- Arrowのゼロコピー原則（CLAUDE.md 2章）を壊す不要な`clone`・コピーを避ける。
- 大きなデータに対する計算量・メモリ使用量に注意する。

## テスト

- 対応する純粋ロジックの単体テストは `tests/engine_tests/`（`cargo test`）に置く。
- 許容誤差等のテスト方針の詳細は `testing-policy.md` を参照。
