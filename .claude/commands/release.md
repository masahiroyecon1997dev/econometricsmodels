---
description: SemVerに基づくバージョンアップ・CHANGELOG作成を支援する（side-effectがあるため明示的な呼び出しのみ。タグ付け・PR作成以降は/release-publish）
argument-hint: [patch/minor/major または具体的なバージョン番号]
allowed-tools: Read, Edit, Bash(git log:*), Bash(git status:*), Bash(cargo:*)
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
6. 変更内容一式（CHANGELOG案・バージョン番号）を提示し、明示的な確認を得てから、コミットする（**タグ付けはここでは行わない**。下記「タグ付けについて」参照）。

## 注意

- バージョンファイルの変更は必ず内容を提示し、確認を得てから実行する。
- バージョン番号は `Cargo.toml`（`[workspace.package] version`）・`pyproject.toml`（`[project] version`）・`python_package/econometricsmodels/__init__.py`（`__version__`）の3箇所を更新する（`Cargo.lock`/`uv.lock`は`cargo check`/`uv lock`等で同期する）。
- push、PyPIへの公開（`cd_release.yml`のトリガーとなる操作）はこのコマンドでは行わない。

## タグ付けについて

タグは、このコマンドのコミットが`dev`経由で`main`にマージされた**後**、`main`のマージコミットに対して付ける（v0.1.0・v0.2.0の実績、および`cd_release.yml`の設計上、tag pushがビルド→PyPI公開→GitHub Release作成の実トリガーであるため）。バージョンバンプのコミット自体に直接タグを付けない（PRマージで別コミットになり、タグの指す内容とmainの実態がずれるため）。`dev`へのPR作成からタグ付け・PyPI公開確認までの後工程は `/release-publish` を使う。
