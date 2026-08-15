---
name: implement-rust
description: engine/engine_pybind側の実装計画立案・実装・規約チェックまでを一括で行う
argument-hint: "[実装したい内容]"
allowed-tools: Read, Write, Edit, Bash(cargo build:*), Bash(cargo clippy:*), Bash(cargo fmt:*), Bash(cargo test:*), Grep, Glob
---

# Rust実装

対応するCLAUDE.mdの方針: 2章（絶対に守るべき設計方針）、3章（リポジトリ構成）、6章（コーディング規約: Rust）

## 実装対象

$ARGUMENTS

## 手順

1. **実装計画の提示**
   - 対象が `engine/`（純粋Rustロジック、PyO3非依存）か `engine_pybind/`（PyO3バインディング層）かを明確にし、責務を混在させない。
   - 変更・追加するファイルを列挙する。
   - 2章の非交渉事項（Arrowゼロコピーでのデータ受け渡し、List渡し＋オブジェクト渡しに対応するデータ構造）に抵触しないか確認する。抵触する可能性がある場合は実装前にユーザーへ確認する。

2. **実装**
   - エラーハンドリングは **thiserror** で独自エラー型を定義する。`engine_pybind`（PyO3境界）で `PyErr` に変換する。
   - `unwrap` / `expect` はプロトタイプ段階を除き避ける。

3. **実装後の規約チェック**
   - `cargo build` でコンパイルが通ることを確認する。
   - `cargo clippy --all-targets -- -D warnings` を実行し、警告ゼロにする。
   - `cargo fmt --check` を実行し、崩れていれば `cargo fmt` で修正する。
   - 関連する既存テストがあれば `cargo test` で実行し、デグレがないか確認する。

4. **コードレビュー**
   - `.claude/agents/rust-reviewer.md`（`rust-reviewer`サブエージェント）に、変更したファイルと実装の背景（対応するIssue番号・設計判断）を渡してレビューを依頼する。実装直後は明示的な指示がなくてもプロアクティブに呼び出す（このスキル自身のIssue完了条件には含めない、後続ステップとして必ず実施する）。
   - 指摘があれば内容を確認し、修正するかどうかを判断する（自明な指摘は反映、設計判断に関わる指摘はユーザーに確認してから対応する）。

## 完了条件

- `cargo build` / `cargo clippy` / `cargo fmt --check` すべて成功
- `engine` と `engine_pybind` の責務分離が保たれている
- CLAUDE.mdの設計方針からの逸脱がない
- `rust-reviewer`サブエージェントによるレビューを実施済み
