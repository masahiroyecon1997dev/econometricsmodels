# econometricsmodels — Claude 開発ガイド

## プロジェクト概要

Polars DataFrame を **ゼロコピー（Arrow メモリ）** で Rust に渡し、省メモリ・高速に計量経済モデルを推定するライブラリ。
Python API は `statsmodels` / `linearmodels` に近い `fit()` → `Results` パターン。

## 基本方針

- **仕様書ファースト**: 実装前に必ず `doc/spec/` を確認し、変更内容をユーザーに提示・確認してから実装フェーズへ進む
- **ゼロコピー最優先**: `Series.to_arrow()` → `ArrayView` の経路を絶対に維持する
- **可読性 > マイクロ最適化**: ボトルネックはベンチマークで確認してから最適化する。unsafe は FFI 境界のみ
- **不明点はまとめて質問**: 曖昧な仕様は実装前に 1 メッセージにまとめてユーザーに確認する

## 標準開発フロー

1. `doc/spec/<module>.md` で仕様を確認する
2. 変更ファイル・実装方針・懸念点を箇条書きでユーザーに提示し、確認を取る
3. Rust コア (`crates/<module>/src/`) → PyO3 バインディング (`python/src/`) の順に実装する
4. 完了後に `cargo clippy --workspace -- -D warnings` と `ruff check` + `ruff format` を実行し、警告ゼロを確認する

## 参照インデックス

### Rules（変わらないルール）
- [Rust・Python コーディング規約](.claude/rules/rust-python-conventions.md)
- [仕様書整合ルール](.claude/rules/spec-alignment.md)
- [ベンチマーク・精度基準](.claude/rules/benchmark-standards.md)

### Skills（ツール操作ノウハウ）
- [cargo コマンド集・エラー対処](.claude/skills/cargo-workflow.md)
- [Python ツールチェーン（uv / maturin / ruff / pytest）](.claude/skills/python-toolchain.md)
- [R 参照実装でテストフィクスチャを作る](.claude/skills/r-reference.md)
- [ゼロコピー実装パターン集](.claude/skills/zero-copy-patterns.md)

### Subagents（タスク特化の詳細指示）
スラッシュコマンドから呼び出す。直接参照する場合は各ファイルを Read してコンテキストに加える。

| コマンド | 対応ファイル | 用途 |
|---------|------------|------|
| `/implement` | [subagents/implementation.md](.claude/subagents/implementation.md) | 実装タスク開始 |
| `/test-gen` | [subagents/testing.md](.claude/subagents/testing.md) | テスト生成タスク開始 |
| `/review` | [subagents/review.md](.claude/subagents/review.md) | コードレビュー（指摘のみ・編集なし） |
| `/cicd` | [subagents/cicd.md](.claude/subagents/cicd.md) | CI/CD 設計・修正 |
