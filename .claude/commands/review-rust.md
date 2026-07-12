---
description: engine/engine_pybind側のコードを規約準拠・設計方針整合性・パフォーマンス/アーキテクチャの観点でレビューする
argument-hint: [レビュー対象ファイル（省略時はgit diff）]
allowed-tools: Read, Grep, Glob, Bash(git diff:*), Bash(git log:*), Bash(cargo clippy:*)
---

# Rustレビュー

対応するCLAUDE.mdの方針: 2章（絶対に守るべき設計方針）、3章（リポジトリ構成）、6章（コーディング規約）、7章（テスト方針）

## レビュー対象

$ARGUMENTS が指定されていればそのファイルを対象にする。指定がなければ `git diff` の変更内容を対象にする。

## チェック観点

1. **コーディング規約準拠**
   - `thiserror` による独自エラー型が定義され、`engine_pybind`境界で`PyErr`に変換されているか
   - `unwrap` / `expect` が残っていないか（プロトタイプ段階の一時的なものを除く）
   - `cargo clippy` で警告が出ていないか

2. **設計方針・アーキテクチャとの整合性**
   - `engine`（純粋Rustロジック）と`engine_pybind`（PyO3バインディング）の責務が混在していないか
   - Arrowのゼロコピー原則を壊す不要なコピー・変換が発生していないか

3. **パフォーマンス**
   - 不要な`clone`や大きなデータのコピーが発生していないか
   - アルゴリズムの計算量・メモリ使用量に明らかな問題がないか

4. **テストの網羅性**
   - 対応するpyfixest/R比較テスト（`tests/api_tests/`）や、純粋ロジックの単体テスト（`tests/engine_tests/`）が用意されているか

## 出力形式

- 指摘事項を「規約」「設計方針/アーキテクチャ」「パフォーマンス」「テスト」の4カテゴリに分けてリストアップする。
- 各指摘に重要度（must fix / should fix / nice to have）を付ける。
- コードの修正はこのコマンドでは行わず、指摘のみ行う（修正は `/実装-rust` に依頼する）。
