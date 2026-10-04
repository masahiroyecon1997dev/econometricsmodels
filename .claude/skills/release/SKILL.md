---
name: release
description: SemVerに基づくバージョンアップ・CHANGELOG作成を支援する（side-effectがあるため明示的な呼び出しのみ。タグ付け・PR作成以降は/release-publish）
argument-hint: "[patch/minor/major または具体的なバージョン番号]"
allowed-tools: Read, Edit, Bash(git log:*), Bash(git status:*), Bash(cargo:*), Bash(uv lock:*), AskUserQuestion
disable-model-invocation: true
---

# リリース支援

対応するCLAUDE.mdの方針: 8章（バージョニング・CI/CD）

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
7. **公開ページの性能数値を更新する**（下記「性能ページの更新」）。
8. 変更内容一式（`git status`・`git diff`で最終確認）をコミットする（**タグ付けはここでは行わない**。下記「タグ付けについて」参照）。

## 注意

- ステップ5の確認は、ファイルを編集する**前**に行う（プレビューで確認 → 確認後に編集、の順序を守る。編集してから確認を求めると、ユーザーが`git diff`を見に行く手間が生じる）。
- バージョン番号は `Cargo.toml`（`[workspace.package] version`）・`pyproject.toml`（`[project] version`）・`python_package/econometricsmodels/__init__.py`（`__version__`）の3箇所を更新する（`Cargo.lock`/`uv.lock`は`cargo check`/`uv lock`等で同期する）。
- push、PyPIへの公開（`cd_release.yml`のトリガーとなる操作）はこのコマンドでは行わない。

## タグ付けについて

タグは、このコマンドのコミットが`dev`経由で`main`にマージされた**後**、`main`のマージコミットに対して付ける（v0.1.0・v0.2.0の実績、および`cd_release.yml`の設計上、tag pushがビルド→PyPI公開→GitHub Release作成の実トリガーであるため）。バージョンバンプのコミット自体に直接タグを付けない（PRマージで別コミットになり、タグの指す内容とmainの実態がずれるため）。`dev`へのPR作成からタグ付け・PyPI公開確認までの後工程は `/release-publish` を使う。

## 性能ページの更新

`docs/guide/performance-results.md`と`docs/guide/performance.md`のサマリー表は、`benchmark_performance.yml`（CI）の計測値から生成する。公開するコードと同じ版の数値にするため、**リリースブランチの内容に対して**実行する（タグpush後の値だと、そのタグの次に入れた修正が反映されない）。

1. リリースブランチをpushし（原則、バージョンバンプ前の最新コミット。push・Actions実行はside-effectなので**ユーザーに確認してから**行う）、`gh workflow run benchmark_performance.yml --ref release/vX.Y.Z`で手動実行する。
2. run完了後、`gh run download <run-id> --dir <一時ディレクトリ>`でartifact（artifactは90日で失効する）を取得する。
3. `uv run python -m performance.render_docs_results <一時ディレクトリ> --tag vX.Y.Z --commit <sha> --run-id <run-id> --results-page docs/guide/performance-results.md --overview-page docs/guide/performance.md`で再生成する。
4. 差分を確認する。以前の版から大きく悪化した点（例: 速度比が1を割った手法、実行時間が桁で変わった点）があれば、`docs/guide/performance.md`の「Known limitations」と`docs/performance/<method>.md`（日本語ノート）の「既知の限界」に反映する。解消した課題は削除する。共有ランナーのノイズ（±数十%）と区別する。
5. `docs/guide/performance.md`の手書き部分に埋め込んだ数値（RE・Logit/Probitの倍率、サマリー表の条件の説明等）が、再生成後の表と食い違っていないか確認して直す。
6. `README.md`の「Performance」節（手法別の`fit()`時間の表、倍率の例、ピークメモリの例）を、再生成後のサマリー表の値（n=1,000,000、classical）に合わせて直す。有効数字2桁程度に丸めて転記し、比較対象の手法・向き（速い／遅い）が変わっていないかも確認する。
