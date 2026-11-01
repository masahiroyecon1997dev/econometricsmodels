# コミット署名（Claude向けの注意）

`main`と`dev`は署名必須のrulesetで保護している。署名鍵（GPG）の秘密鍵はホストにだけあり、コンテナにはホストの`gpg-agent`の制限ソケットが転送される。ホスト側は署名のたびにパスフレーズの確認を求める設定（`ignore-cache-for-signing`）で、確認はホストの専用端末に出る。ホスト環境の手順は利用者が個人で管理しており、このリポジトリには置かない。

## 鍵の扱い

- 秘密鍵・ホストの`~/.gnupg`・`~/.ssh`は、読まない・exportしない・動作確認の目的でも見ない。`gpg --export-secret-keys`や`gpg --list-secret-keys`など秘密鍵を対象にするコマンドは実行しない。
- ホスト側の設定（`gpg-agent.conf`、キャッシュTTL、`ignore-cache-for-signing`）の変更は利用者の作業。緩める提案を自分から出さない。依頼されたときは手順を示すだけにする。

## 署名が走る操作の前後

- コミット・マージ・タグ作成など署名が走る操作の前に、ホスト端末で確認が出ることを一言伝える。署名待ちで止まるため、Bashの`timeout`は120秒以上にする。
- 署名に失敗した（`signing failed`、タイムアウト）ときは、エラーの内容を報告して止まる。連続でリトライしない。
- 署名を避けない。`--no-gpg-sign`、`-c commit.gpgsign=false`、`commit.gpgsign`の解除は使わない。

## 履歴の書き換え

- 先に退避ブランチを作り、書き換え後にツリー・author・メッセージ・署名（`%G?`が`G`）を新旧で照合してから、ブランチを動かす。
- 範囲内にマージコミットがあると`git rebase --rebase-merges`は古いマージの再現で衝突する。その場合は`git commit-tree -S`で元のツリーを再利用して作り直す。committerは署名鍵のUIDと同じ利用者に揃える（GitHubのcommitterのままだとVerifiedにならない）。
- force pushは`.claude/settings.json`で禁止しているため、利用者が実行する。
- 並行作業のステージ済み変更があるときは、stashしてから作業し、`git stash pop --index`で元に戻す。捨てない。
