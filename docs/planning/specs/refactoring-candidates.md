# リファクタリング候補メモ

コード解説（`/explain-code`スキル等）や通常の実装作業の過程で気づいた、
リファクタリングの余地がある箇所を随時記録する場所。

`refactoring-issue231-progress.md`との違い: あちらは
[#231](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/231)
としてスコープ・フェーズを確定させた上で実施する計画書だが、こちらは
Issue化する前の**気づいた時点での未整理のメモ**を溜める場所。ここに溜まった
項目は、着手時にIssue化するか`refactor`スキルの対象範囲として指定するかを
都度ユーザーが判断する。

## 記録フォーマット

各項目は以下を含める。

- **対象**: ファイルパス・行
- **内容**: 何が気になったか
- **気づいた経緯**: どの作業中に気づいたか（日付）
- **状態**: 未対応 / 対応済み（対応したIssue・PR等） / 対応不要と判断（理由）

---

## 一覧

### 1. `benchmark/load_wooldridge.py`の`SUGGESTED_DATASETS`が未使用

- **対象**: [benchmark/load_wooldridge.py:21-26](../../../benchmark/load_wooldridge.py#L21-L26)
- **内容**: 手法ごとの候補データセット名を持つ辞書`SUGGESTED_DATASETS`が、
  定義箇所以外どこからもimport・参照されていない（`grep`で確認済み）。
  コメントも「要検討・要確定」のまま更新されておらず、実際に採用された
  データセット（`mroz`, `401ksubs`等）は各`generate_*.py`側に個別に
  ハードコードされている。実質的にデッドコードの疑い。
- **気づいた経緯**: 2026-08-14、`load_wooldridge.py`のコード解説中に発見。
- **状態**: 未対応（残す/削除するかの方針をユーザーに確認待ち）
