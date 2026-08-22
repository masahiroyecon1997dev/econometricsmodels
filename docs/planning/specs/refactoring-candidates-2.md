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
