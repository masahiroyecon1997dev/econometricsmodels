# リファクタリング候補メモ（続き）

`docs/planning/specs/refactoring-candidates.md`が別のエージェントによる
リファクタリング作業中（2026-08-22時点）のため、競合を避けるためにこちらへ
新規項目を追記する。番号は継続（`refactoring-candidates.md`の項目43までの続き、
44から開始）。フォーマット・運用方針は元ファイルと同一（コード解説
（`/explain-code`スキル等）や通常の実装作業の過程で気づいた、リファクタリングの
余地がある箇所を随時記録する場所）。

`refactoring-issue231-progress.md`との違い: あちらは
[#231](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/231)
としてスコープ・フェーズを確定させた上で実施する計画書だが、こちらは
Issue化する前の**気づいた時点での未整理のメモ**を溜める場所。ここに溜まった
項目は、着手時にIssue化するか`refactor`スキルの対象範囲として指定するかを
都度ユーザーが判断する。

両ファイルの統合（`refactoring-candidates.md`側の作業完了後）は別途ユーザー判断で行う。

## 記録フォーマット

各項目は以下を含める。

- **対象**: ファイルパス・行
- **内容**: 何が気になったか
- **気づいた経緯**: どの作業中に気づいたか（日付）
- **状態**: 未対応 / 対応済み（対応したIssue・PR等） / 対応不要と判断（理由）

---

## 一覧

### 44. `assert_dict_close`/`check_margeff`の`rename`引数が、非デフォルト値で呼ばれている箇所が無い

- **対象**: [tests/_assertions.py:37-53](../../../tests/_assertions.py#L37-L53)
  （`assert_dict_close`の`rename: Callable[[str], str] = rename_intercept`）・
  [tests/_assertions.py:56-64](../../../tests/_assertions.py#L56-L64)
  （`check_margeff`の同じく`rename`引数）
- **内容**: ユーザー指摘（2026-08-22）。`grep`で全呼び出し元
  （`test_ols_fixtures.py`・`test_wls_fixtures.py`・`test_logit_fixtures.py`・
  `test_probit_fixtures.py`・`test_iv_fixtures.py`・`test_iv_gmm_fixtures.py`、
  いずれも`functools.partial`で`rtol`/`atol`のみ束縛してから呼んでいる）を
  確認したところ、`rename=`を明示的に渡している箇所は1件も無かった。全呼び出しが
  デフォルト値の`rename_intercept`をそのまま使っている。
- **Claudeの所感**: 現状は「将来、切片名の変換ルールが異なるリファレンス実装が
  出てきた場合に備えた拡張ポイント」という位置づけだが、実際に使われたことが
  一度も無い。YAGNI（You Aren't Gonna Need It）の観点では引数自体を削除して
  `rename_intercept`を関数内に固定してしまう方が単純だが、拡張ポイントとして
  意図的に残しているだけの可能性もあり、削除するかどうかはユーザー判断に委ねる。
- **気づいた経緯**: 2026-08-22、`tests/_assertions.py`解説後のユーザー指摘・
  確認依頼。
- **状態**: 未対応（`rename`引数を削除するか、拡張ポイントとして維持するかは
  ユーザー判断待ち）

### 45. `separation_suspected_dataset`だけ標準ライブラリ`random`/素朴な`for`ループで、他のDGPと生成方法が異なる

- **対象**: [tests/_helpers.py:52-70](../../../tests/_helpers.py#L52-L70)
  （`separation_suspected_dataset`、`random.seed`/`random.uniform`/
  `math.exp`＋`for`ループ）
- **内容**: ユーザー指摘（2026-08-22）。`benchmark/`側の全DGP
  （`generate_*_datasets.py`）は`numpy`（`np.random.default_rng(seed)`）ベースの
  ベクトル化演算で書かれており、`tests/_helpers.py`の`with_cluster_groups`も
  polarsのベクトル演算だが、`separation_suspected_dataset`だけ標準ライブラリ
  `random`モジュール＋`for`ループで1件ずつ`y`を生成している。
- **Claudeの所感**: 実害はない（`n=200`程度の小規模データで、結果の正しさに
  影響しない）が、リポジトリ全体の乱数生成の一貫性という観点では`numpy`に
  揃える方が読み手の負担が減ると考える。優先度は低い。
- **気づいた経緯**: 2026-08-22、`tests/_helpers.py`解説後のユーザー指摘。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 46. `_helpers.py`/`_assertions.py`に定数と関数が混在している

- **対象**: [tests/_helpers.py:34](../../../tests/_helpers.py#L34)（`DATA_DIR`）・
  [tests/_helpers.py:73](../../../tests/_helpers.py#L73)（`MROZ_X`）・
  [tests/_assertions.py:19](../../../tests/_assertions.py#L19)（`MARGEFF_AT`）
- **内容**: ユーザー指摘（2026-08-22）。`_helpers`/`_assertions`という
  ファイル名は関数（ヘルパー・アサーション）を示唆するが、実際には
  `DATA_DIR`・`MROZ_X`・`MARGEFF_AT`という定数も同居している。
- **Claudeの所感**: ユーザー自身も「あまり問題になるものではない」とコメント
  している通り実害は乏しい。定数を`_constants.py`のような別ファイルへ分離する
  ほどの量ではなく、現状維持でも大きな支障は無いと考える。優先度は低い。
- **気づいた経緯**: 2026-08-22、`tests/_helpers.py`解説後のユーザー指摘。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 48. `tests/_assertions.py`/`tests/_helpers.py`のdocstringに「何箇所重複していたか」「フェーズ3.5で修正予定」等、経緯だけを説明する冗長な記述が残っている

- **対象**: [tests/_assertions.py:10](../../../tests/_assertions.py#L10)
  （「一部は他と異なる計算式を使っている、フェーズ3.5で別途調査・修正予定」）・
  [tests/_helpers.py:5](../../../tests/_helpers.py#L5)（「22箇所で重複」）・
  [tests/_helpers.py:12](../../../tests/_helpers.py#L12)（「4ファイルで重複」）
- **内容**: ユーザー指摘（2026-08-22）。これらはIssue #231フェーズ3実施時点の
  「なぜこの集約作業をしたか」という経緯説明だが、(1)「フェーズ3.5で別途調査・
  修正予定」は`tests/_tolerances.py`のdocstringで確認した通り**既に修正済み**
  であり、予定形のまま古い記述が残っている（現状と食い違っている）。
  (2)「22箇所で重複」「4ファイルで重複」は、集約が完了した現在では検証不能な
  過去の状態を指す数値で、今後コードが変わればこの数字自体が古くなる。
  一般的なコメント指針（「現在のタスク・修正・呼び出し元を参照しない、
  PR説明に属する情報でありコードベースの変化とともに腐る」）にも合致する。
- **Claudeの所感**: `_tolerances.py`のdocstringは既にフェーズ3.5の修正内容を
  過去形（「修正し...揃えた」）で正しく記述できているのに対し、
  `_assertions.py`は「予定」のまま取り残されている。関数の役割・設計判断の
  説明として必要な部分（「主リファレンスとの数値比較用に集約する」等）は
  残しつつ、経緯・件数への言及は削除するのが妥当と考える。
- **気づいた経緯**: 2026-08-22、`tests/_tolerances.py`解説前のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 49. `tests/_tolerances.py`の許容誤差が全て直書きで、共通する値（機械精度ベース等）が定数化されていない

- **対象**: [tests/_tolerances.py:18-113](../../../tests/_tolerances.py#L18-L113)
  （`TOLERANCES`辞書全体）
- **内容**: ユーザー指摘（2026-08-22）。`1e-8`（機械精度に近い基準、
  `ols_fixtures`/`wls_fixtures`/`iv_fixtures`/`iv_gmm_fixtures`/
  `logit_fixtures`/`probit_fixtures`の`rtol`、`ols_crosscheck`/`wls_crosscheck`/
  `iv_crosscheck`の`rtol_strict`で共通）や`1e-8`（`atol`、複数エントリで共通）
  のような、複数箇所で同じ値・同じ理由（機械精度、CLAUDE.mdの設計方針として
  リポジトリ全体で共通のはずの基準）が直書きで繰り返されている。
- **Claudeの所感**: `MACHINE_PRECISION_RTOL = 1e-8`のような名前付き定数に
  すれば、「なぜこの値なのか」が値そのものより名前で伝わり、複数箇所を
  同時に変更する際の一貫性も保ちやすくなる。ただし`rtol_hac`のように
  実測値に基づいて個別に決めた値まで無理に共通定数化する必要はなく
  （`testing-policy.md`「一律に緩めると本来検出できるはずのバグを見逃す」）、
  「機械精度」等の**設計方針レベルで共通の値だけ**を定数化するのが妥当と考える。
- **気づいた経緯**: 2026-08-22、`tests/_tolerances.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否・対象範囲の線引きはユーザー判断待ち）
