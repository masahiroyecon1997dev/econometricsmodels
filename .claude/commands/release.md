---
description: SemVerに基づくバージョンアップ・CHANGELOG作成・タグ付けを支援する（side-effectがあるため明示的な呼び出しのみ）
argument-hint: [patch/minor/major または具体的なバージョン番号]
allowed-tools: Read, Edit, Bash(git log:*), Bash(git tag:*), Bash(git status:*), Bash(cargo:*)
disable-model-invocation: true
---

# リリース支援

対応するCLAUDE.mdの方針: 9章（バージョニング・CI/CD）

## バージョンアップ種別

$ARGUMENTS （patch/minor/major、または明示的なバージョン番号）

## 手順

1. `Cargo.toml` / `pyproject.toml` 等から現在のバージョンを確認する。
2. 前回リリース（前回のtag）以降のコミットログ（Conventional Commits形式）を集計する。
   - `feat:` → Y（機能追加）
   - `fix:` → Z（バグ修正・性能改善）
   - `BREAKING CHANGE` を含むもの → X（破壊的変更）
   - ただし `0.x.x` のプレリリース期間中は、`Y`の変更でも破壊的変更を許容する例外に注意する。
3. コミット内容から適切なバージョン種別を判定し、`$ARGUMENTS`の指定と食い違いがあればユーザーに確認する。
4. CHANGELOGの更新案を作成する。
5. `Cargo.toml` / `pyproject.toml` 等のバージョン番号を更新する。
6. 変更内容一式（CHANGELOG案・バージョン番号）を提示し、明示的な確認を得てから、コミット・タグ付けを行う。

## 注意

- タグ付け・バージョンファイルの変更は必ず内容を提示し、確認を得てから実行する。
- push、PyPIへの公開（`cd_release.yml`のトリガーとなる操作）はこのコマンドでは行わない。
