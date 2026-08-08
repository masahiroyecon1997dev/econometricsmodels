---
description: SemVerに基づくバージョンアップ・CHANGELOG作成を支援する（side-effectがあるため明示的な呼び出しのみ。タグ付け・PR作成以降は/release-publish）
argument-hint: [patch/minor/major または具体的なバージョン番号]
allowed-tools: Read, Edit, Bash(git log:*), Bash(git status:*), Bash(cargo:*), Bash(uv lock:*), AskUserQuestion
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
4. CHANGELOGの更新案（`[X.Y.Z] - 日付`セクションの本文、Added/Changed/Fixed等）を作成する。
5. **バージョンバンプ・CHANGELOG案をユーザーに提示し、明示的な確認を得る**。この時点ではまだファイルを編集しない（`AskUserQuestion`でCHANGELOG本文全体をプレビュー表示し、「この内容で進める」で確認を得る形が実績あり。ファイルの差分ではなく完成形のプレビューを見せることで、確認が取りやすくなる）。
6. 確認が得られたら、以下を実施する。
   - `CHANGELOG.md`に確認済みの内容を追記する（`[Unreleased]`の下に新しいバージョンセクションを追加し、末尾の比較リンクも更新する）。
   - バージョン番号を更新する（`Cargo.toml`の`[workspace.package] version`・`pyproject.toml`の`[project] version`・`python_package/econometricsmodels/__init__.py`の`__version__`の3箇所）。
   - `cargo check --workspace`・`uv lock`を実行し、`Cargo.lock`/`uv.lock`を同期する。
7. 変更内容一式（`git status`・`git diff`で最終確認）をコミットする（**タグ付けはここでは行わない**。下記「タグ付けについて」参照）。

## 注意

- ステップ5の確認は、ファイルを編集する**前**に行う（プレビューで確認 → 確認後に編集、の順序を守る。編集してから確認を求めると、ユーザーが`git diff`を見に行く手間が生じる）。
- バージョン番号は `Cargo.toml`（`[workspace.package] version`）・`pyproject.toml`（`[project] version`）・`python_package/econometricsmodels/__init__.py`（`__version__`）の3箇所を更新する（`Cargo.lock`/`uv.lock`は`cargo check`/`uv lock`等で同期する）。
- push、PyPIへの公開（`cd_release.yml`のトリガーとなる操作）はこのコマンドでは行わない。

## タグ付けについて

タグは、このコマンドのコミットが`dev`経由で`main`にマージされた**後**、`main`のマージコミットに対して付ける（v0.1.0・v0.2.0の実績、および`cd_release.yml`の設計上、tag pushがビルド→PyPI公開→GitHub Release作成の実トリガーであるため）。バージョンバンプのコミット自体に直接タグを付けない（PRマージで別コミットになり、タグの指す内容とmainの実態がずれるため）。`dev`へのPR作成からタグ付け・PyPI公開確認までの後工程は `/release-publish` を使う。
