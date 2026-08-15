---
name: release-publish
description: バージョンバンプ済みのコミットを、devへのPR〜mainへの反映〜タグ付け〜PyPI公開確認まで進める（side-effectがあるため明示的な呼び出しのみ）
argument-hint: "対象バージョン番号（例: 0.2.0）"
allowed-tools: Read, Bash(git:*), Bash(gh:*), Monitor, AskUserQuestion
disable-model-invocation: true
---

# リリース公開フロー支援

対応するCLAUDE.mdの方針: 8章（バージョニング・CI/CD）、5章（Git運用「マージはCIがgreenであることに加え、内容を確認してからmergeする。自動セルフマージはしない」）

## 前提

- `/release`でバージョンバンプ・CHANGELOGのコミットが完了していること（このコマンドはそこから先の、PR作成〜タグ付け〜PyPI公開確認を担う）。
- 対象バージョン: `$ARGUMENTS`（例: `0.2.0`）。以下`vX.Y.Z`と表記する。

## 手順

### 1. dev へのPR

1. 現在のブランチ（バージョンバンプのコミットがある想定。無ければユーザーに確認する）をpushし、`dev`へのPRを作成する。
2. PRのCIを`Monitor`で監視し、結果を報告する。CIが失敗した場合は原因を報告し、次に進む前にユーザーに確認する。
3. **ユーザーにPRのマージを依頼し、待つ**（自動セルフマージはしない）。マージ完了の連絡を受けてから次に進む。

### 2. dev → main のPR

4. `dev`→`main`のPRを作成する（過去のリリース同様「release: vX.Y.Zに向けてdevをmainに反映」のようなタイトルにする）。
5. CIを監視する。`main`向けPRは`dev`向けPRにない追加ジョブ（例: CodeQLの`Analyze`）が走る場合がある点に注意する。
6. **ユーザーにマージを依頼し、待つ**。

### 3. 公開前のビルド確認

7. マージ確認後、`gh workflow run cd_release.yml --ref main`で`workflow_dispatch`トリガーの事前ビルド確認を実行する（`cd_release.yml`のジョブ条件により`publish-pypi`以降はスキップされ、マルチOS（Linux/macOS×2/Windows/sdist）ビルドのみ検証できる）。
8. 全ビルドジョブの成功を`Monitor`で監視・確認する。失敗があれば原因を報告し、修正が必要な場合はユーザーに確認してから再実行する。

### 4. タグ付け・公開

9. `main`の最新コミット（dev→mainのマージコミット）のSHAを確認し、`vX.Y.Z`のannotated tagをローカルに作成する。
10. **タグの内容（対象コミット・メッセージ）を提示し、pushしてよいかユーザーに確認する**（タグpushは`cd_release.yml`の実トリガーであり、ビルド成功後はPyPI公開に進む）。
11. 確認が得られたらタグをpushする。
12. ビルドジョブの完了を`Monitor`で監視する。
13. `publish to PyPI`ジョブが`waiting`状態（`pypi`環境のRequired reviewersゲート）になったら、承認用のGitHub Actions実行URLを提示し、**ユーザーの承認を待つ**。
14. 承認確認後、`publish to PyPI`・`create GitHub Release`ジョブの完了を監視する。

### 5. 完了確認

15. `gh release view vX.Y.Z`でGitHub Releaseの内容（アセット一覧等）を確認する。
16. PyPI公開URL（`https://pypi.org/project/econometricsmodels/X.Y.Z/`）とGitHub ReleaseのURLをユーザーに報告する。

## 注意

- 各ステップの「ユーザーに依頼し、待つ」は省略しない。マージ・タグpush・PyPI公開承認はいずれもユーザー自身の操作／確認を経てから次に進む。
- タグpush・PyPI公開は実世界への副作用を伴う操作のため、手順3（事前ビルド確認）を省略しない。
- 想定外のCI失敗・承認待ち以外の異常（ジョブ失敗等）が起きた場合は、その場で報告し次のステップに進む前にユーザーに確認する。
