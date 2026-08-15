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

### 2. `generate_linear_datasets.py`の`k`下限チェックが4箇所で同型パターン重複

- **対象**: [benchmark/linear/generate_linear_datasets.py:76-114](../../../benchmark/linear/generate_linear_datasets.py#L76-L114)
- **内容**: `moderate_multicollinearity`/`high_condition_number`（k>=2）・
  `perfect_multicollinearity`（k>=3）・`scale_variance`（k>=2）・
  `scale_variance_mild`（k>=2）の4箇所で、いずれも
  `if k < N: raise ValueError(f"{scenario} requires k >= N")`という
  同型の2行パターンを繰り返している。`_require_min_k(scenario, k, minimum)`
  のような小さなヘルパーに切り出せる余地はあるが、規模が小さく
  優先度は低い（nice to have）と判断。
- **気づいた経緯**: 2026-08-15、`generate_linear_datasets.py`のコード解説中に発見。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 3. `sys.path.insert`によるimportが静的解析（IDEの定義ジャンプ）と相性が悪い

- **対象**: `benchmark/`配下の各ファイル冒頭にある`sys.path.insert(0, str(Path(__file__)...))`
  パターン全般（例: [benchmark/freeze_datasets.py:41-49](../../../benchmark/freeze_datasets.py#L41-L49)）
- **内容**: ユーザー指摘（2026-08-15）。`Path(__file__).resolve()...`による動的なパス追加は
  実行時にしか解決されないため、VSCode（Pylance等）の静的解析は`sys.path.insert`の中身を
  実行せずに解析するので、`from generate_linear_datasets import ...`等の「定義へ移動」
  （Go to Definition）が効かず不便。
- **Claudeの所感**: `benchmark/`全体を正式なPythonパッケージ化する（`__init__.py`追加）と、
  実行方法が`python freeze_linear_datasets.py`のような直接実行から
  `python -m benchmark.linear.freeze_linear_datasets`等に変わってしまうトレードオフがある。
  一方、**`.vscode/settings.json`（または`pyrightconfig.json`）に
  `"python.analysis.extraPaths": ["benchmark", "benchmark/linear", "benchmark/nonlinear", "benchmark/iv"]`
  を追加する**方法であれば、実行時のimportの仕組み（`sys.path.insert`）自体は変えずに、
  IDEの静的解析にだけ「このパスも見てよい」と教えられるため、定義ジャンプの不便さだけを
  低リスクで解消できる可能性がある。
- **気づいた経緯**: 2026-08-15、`generate_linear_datasets.py`解説後の雑談から。
- **状態**: 未対応（`.vscode/settings.json`追加の要否をユーザー判断待ち）

### 4. `generate_linear_datasets.py`の`SCENARIOS`と`freeze_linear_datasets.py`の`SYNTHETIC_SCENARIOS`が完全重複

- **対象**: [benchmark/linear/generate_linear_datasets.py:23-34](../../../benchmark/linear/generate_linear_datasets.py#L23-L34)・
  [benchmark/linear/freeze_linear_datasets.py:30-41](../../../benchmark/linear/freeze_linear_datasets.py#L30-L41)
- **内容**: ユーザー指摘（2026-08-15）。両リストを比較したところ、順序・要素とも
  完全に同一（10シナリオ）だった。Issue #231フェーズ2で対応済みの
  「`NUMERIC_SCENARIOS`/`test_*_fixtures.py`側`SCENARIOS`の一元化」
  （`refactoring-issue231-progress.md`フェーズ2ステップ2項目5）と同種の重複だが、
  この`generate_linear_datasets.py`↔`freeze_linear_datasets.py`間のペアは
  その時の対応範囲に含まれていなかった模様。
- **Claudeの所感**: `freeze_linear_datasets.py`側で
  `from generate_linear_datasets import SCENARIOS as SYNTHETIC_SCENARIOS`と
  importする形に置き換えれば、単一定義元に統一できる（値が完全一致のため
  挙動を変えないリファクタリングとして低リスク）。nonlinear/iv系統の
  `freeze_*_datasets.py`にも同種の重複が無いか、着手時に合わせて確認する価値がある。
- **気づいた経緯**: 2026-08-15、`generate_linear_datasets.py`解説後の雑談から。
- **状態**: 未対応（着手要否はユーザー判断待ち）
