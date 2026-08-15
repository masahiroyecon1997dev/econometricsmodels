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

### 4. `generate_*_datasets.py`の`SCENARIOS`と`freeze_*_datasets.py`側リストが3系統とも完全重複

- **対象**:
  - [benchmark/linear/generate_linear_datasets.py:23-34](../../../benchmark/linear/generate_linear_datasets.py#L23-L34) ↔
    [benchmark/linear/freeze_linear_datasets.py:30-41](../../../benchmark/linear/freeze_linear_datasets.py#L30-L41)（`SCENARIOS`↔`SYNTHETIC_SCENARIOS`）
  - [benchmark/nonlinear/generate_nonlinear_datasets.py:60-68](../../../benchmark/nonlinear/generate_nonlinear_datasets.py#L60-L68) ↔
    [benchmark/nonlinear/freeze_nonlinear_datasets.py:32-40](../../../benchmark/nonlinear/freeze_nonlinear_datasets.py#L32-L40)（`SCENARIOS`↔`LOGIT_SCENARIOS`。`PROBIT_SCENARIOS`は`list(LOGIT_SCENARIOS)`で既に間接的に連動）
  - [benchmark/iv/generate_iv_datasets.py:41-52](../../../benchmark/iv/generate_iv_datasets.py#L41-L52) ↔
    [benchmark/iv/freeze_iv_datasets.py:30-41](../../../benchmark/iv/freeze_iv_datasets.py#L30-L41)（`SCENARIOS`↔`IV_SCENARIOS`）
- **内容**: ユーザー指摘（2026-08-15、linear系統で発覚）を受けてnonlinear/iv系統も
  Pythonスクリプトで機械的に比較したところ、**3系統とも**順序・要素完全一致だった
  （実測確認済み）。Issue #231フェーズ2で対応済みの「`NUMERIC_SCENARIOS`/
  `test_*_fixtures.py`側`SCENARIOS`の一元化」（`refactoring-issue231-progress.md`
  フェーズ2ステップ2項目5）と同種の重複だが、この`generate_*_datasets.py`↔
  `freeze_*_datasets.py`間のペアはその時の対応範囲に含まれていなかった模様。
- **Claudeの所感**: 3系統とも`freeze_*_datasets.py`側で
  `from generate_*_datasets import SCENARIOS as ...`の形にimportし直せば単一定義元に
  統一できる（値が完全一致のため挙動を変えないリファクタリングとして低リスク）。
- **気づいた経緯**: 2026-08-15、`generate_linear_datasets.py`解説後の雑談（linear分）→
  `generate_nonlinear_datasets.py`解説時に3系統横断で実測確認。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 5. `unknown scenario`検証（`ValueError`）が3系統の`generate_*_dataset`関数で完全重複

- **対象**: [benchmark/linear/generate_linear_datasets.py:62-65](../../../benchmark/linear/generate_linear_datasets.py#L62-L65)・
  [benchmark/nonlinear/generate_nonlinear_datasets.py:106-109](../../../benchmark/nonlinear/generate_nonlinear_datasets.py#L106-L109)・
  [benchmark/iv/generate_iv_datasets.py:106-109](../../../benchmark/iv/generate_iv_datasets.py#L106-L109)
- **内容**: ユーザー指摘（2026-08-15）。`if scenario not in SCENARIOS: raise ValueError(f"unknown
  scenario: {scenario!r}. choose from {SCENARIOS}")`という同型の検証が3ファイルで重複している
  （項目4の`SCENARIOS`重複と直接関連するが、こちらは検証ロジック自体の重複）。
  nonlinear側にはさらに同型の`unknown link`検証（`generate_nonlinear_datasets.py:119-122`）が
  同じ関数内にもう1つある。当初のコード解説時に見落としていた項目。
- **Claudeの所感**: `_common.py`に`validate_scenario(scenario, valid_scenarios)`のような
  小さなヘルパーを切り出せば3箇所（＋nonlinearのlink検証）を統合できる。ただし規模は小さく、
  優先度は`nice to have`程度。
- **気づいた経緯**: 2026-08-15、`generate_nonlinear_datasets.py`解説後のユーザー指摘で発覚
  （当初の解説時は見落とし）。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 6. 線形予測子の組み立て方（`column_stack`版 vs `beta[0]+X@beta[1:]`版）の書き方の不統一

- **対象**: [benchmark/linear/generate_linear_datasets.py:133](../../../benchmark/linear/generate_linear_datasets.py#L133)
  （`y = beta[0] + X @ beta[1:] + errors`）と
  [benchmark/nonlinear/generate_nonlinear_datasets.py:158-159](../../../benchmark/nonlinear/generate_nonlinear_datasets.py#L158-L159)
  （`x_const = np.column_stack([np.ones(n), X]); p = _LINK_CDF[link](x_const @ beta)`）
- **内容**: ユーザー指摘（2026-08-15）。同じ「切片＋線形結合」という数学的に同じ計算を、
  linear側とnonlinear側で異なる書き方（切片を別扱い vs 切片列を結合してから1回の行列積）で
  実装している。
- **Claudeの所感**: 統一するなら`_linear_predictor(X, beta)`のような小さな共通ヘルパーに
  切り出せるが、1〜2行の違いであり効果は小さい。優先度は低い。
- **気づいた経緯**: 2026-08-15、`generate_nonlinear_datasets.py`解説後のユーザー指摘。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 7. 説明変数X生成ロジック（multicollinearity/high_condition_number/perfect_multicollinearity/scale_variance）が3系統でほぼ同一

- **対象**: [benchmark/linear/generate_linear_datasets.py:76-114](../../../benchmark/linear/generate_linear_datasets.py#L76-L114)
  （`X`）・[benchmark/nonlinear/generate_nonlinear_datasets.py:132-156](../../../benchmark/nonlinear/generate_nonlinear_datasets.py#L132-L156)
  （`X`）・[benchmark/iv/generate_iv_datasets.py:141-156](../../../benchmark/iv/generate_iv_datasets.py#L141-L156)
  （`x_exog`）
- **内容**: ユーザー指摘（2026-08-15）。`rho=0.8/0.999`の相関構造を持つ`multivariate_normal`
  生成、`X[:,2]=2*X[:,0]+3*X[:,1]`という完全多重共線性の作り方が3系統でほぼ同一ロジック
  （IV側のdocstringにも「OLSと同じ発想をx_exogに適用」と明記済み）。
- **Claudeの所感**: `_common.py`に`_correlated_design_matrix(rng, scenario, k)`のような
  共通関数を切り出せる余地はあるが、変数名（`X` vs `x_exog`）・呼び出し文脈の違いもあり
  設計にやや検討が要る規模。ユーザー自身も「無理にすることではない」とコメント済みで、
  優先度は中程度・任意対応と位置づける。
- **気づいた経緯**: 2026-08-15、`generate_nonlinear_datasets.py`解説後のユーザー指摘。
- **状態**: 未対応（任意対応、着手要否はユーザー判断待ち）

### 8. `generate_logit_dataset`/`generate_probit_dataset`の後方互換用ラッパーが不要では

- **対象**: [benchmark/nonlinear/generate_nonlinear_datasets.py:179-206](../../../benchmark/nonlinear/generate_nonlinear_datasets.py#L179-L206)
- **内容**: ユーザー指摘（2026-08-15）。docstringに「既存の呼び出し元との互換のため名前付きで
  残している」とあるこの2つの薄いラッパー関数の実際の呼び出し箇所を`grep`で確認したところ、
  `freeze_nonlinear_datasets.py`と本ファイル自身の`__main__`ブロックの2箇所のみだった。
  後方互換性が不要なら、これらの呼び出し元を`generate_binary_choice_dataset(scenario,
  link="logit"/"probit")`に直接書き換えることでラッパー自体を削除できる。
- **Claudeの所感**: 呼び出し箇所が少なく（2箇所）、いずれも`benchmark/nonlinear/`内で
  完結しているため、削除は低リスクだと考える。
- **気づいた経緯**: 2026-08-15、`generate_nonlinear_datasets.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 9. `scale_variance`のスケール倍率（1e6/1e-3）が3系統ともマジックナンバーとして重複

- **対象**: [benchmark/linear/generate_linear_datasets.py:102-103](../../../benchmark/linear/generate_linear_datasets.py#L102-L103)
  （`X[:,0]*=1e6, X[:,1]*=1e-3`、直書き）・
  [benchmark/nonlinear/generate_nonlinear_datasets.py:77-78](../../../benchmark/nonlinear/generate_nonlinear_datasets.py#L77-L78)
  （`_SCALE_VARIANCE_X1_SCALE = 1e6`等、名前付き定数）・
  [benchmark/iv/generate_iv_datasets.py:231-232](../../../benchmark/iv/generate_iv_datasets.py#L231-L232)
  （`x_exog[:,0]*=1e6, x_exog[:,1]*=1e-3`、直書き）
- **内容**: ユーザー指摘（2026-08-15）。3系統とも`1e6`/`1e-3`という同じ倍率を使っており、
  nonlinear/iv側のコメントには「OLSと同じ倍率」と明記されている（意図的に値を揃えている）。
  にもかかわらず値の実体は3ファイルに分散しており（うち2ファイルはマジックナンバー直書き）、
  値を変えたくなった場合の追従漏れリスクがある。
- **Claudeの所感**: `_common.py`に`SCALE_VARIANCE_X1_SCALE = 1e6`・
  `SCALE_VARIANCE_X2_SCALE = 1e-3`として集約し、3ファイルともそこから参照する形に
  統一するのが妥当。「同じであるべき」という意図が既にコード上に書いてある以上、
  低リスクな改善だと考える。
- **気づいた経緯**: 2026-08-15、`generate_nonlinear_datasets.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 10. `heteroskedastic`/`autocorrelated`の誤差項生成式がlinear・IVの2系統でマジックナンバーとして重複

- **対象**: [benchmark/linear/generate_linear_datasets.py:121,124](../../../benchmark/linear/generate_linear_datasets.py#L121-L124)・
  [benchmark/iv/generate_iv_datasets.py:178,186](../../../benchmark/iv/generate_iv_datasets.py#L178-L186)
- **内容**: ユーザー指摘（2026-08-15）。`heteroskedastic`の分散式`sigma_i = 0.5 + 2.0 *
  np.abs(X[:,0]/x_exog[:,0])`と、`autocorrelated`のAR(1)係数`rho(_ar) = 0.7`が、
  linear・IVの2系統で全く同じ値のマジックナンバーとして直書きされている
  （nonlinearはこの2シナリオ自体を持たないため対象外、モジュールdocstringで
  理由明記済み）。項目9（`scale_variance`のスケール倍率重複）と同種の問題。
- **Claudeの所感**: `_common.py`に`HETEROSKEDASTIC_SIGMA_BASE = 0.5`・
  `HETEROSKEDASTIC_SIGMA_SLOPE = 2.0`・`AUTOCORRELATED_RHO = 0.7`のような
  名前付き定数として集約する余地がある。項目9と合わせて対応すると効率が良さそう。
- **気づいた経緯**: 2026-08-15、`generate_iv_datasets.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 11. `_run_cluster_case`（generate_ols_fixtures.py）が`run_statsmodels_benchmark.py`の`coef`/`se`抽出ロジックと重複

- **対象**: [benchmark/linear/fixtures/generate_ols_fixtures.py:154-158](../../../benchmark/linear/fixtures/generate_ols_fixtures.py#L154-L158)・
  [benchmark/linear/run_statsmodels_benchmark.py:104-106](../../../benchmark/linear/run_statsmodels_benchmark.py#L104-L106)
- **内容**: `_run_cluster_case`は、CSVに無い疑似グループ列`_group`を動的に追加する必要が
  あるため`run()`を再利用できず、statsmodelsを独自に直接呼んでいる（正当な理由）。
  ただし`{str(name): float(v) for name, v in model.params.to_dict().items()}`という
  `coef`/`se`抽出の辞書内包表記は、`run()`内の同種のコードとほぼ同じパターンで重複している。
- **Claudeの所感**: `_common.py`に`extract_coef_se(model) -> dict`のような小さな
  ヘルパーを切り出せる余地はあるが、規模は小さく優先度は低い。項目12でより根本的な
  代替案（`_run_cluster_case`自体の解消）を検討済み。
- **気づいた経緯**: 2026-08-15、`generate_ols_fixtures.py`解説中に発見。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 12. クラスター用の疑似グループ列をfreeze時にCSVへ焼き込み、`_run_cluster_case`を`run()`一律呼び出しに統合できないか

- **対象**: [benchmark/linear/fixtures/generate_ols_fixtures.py:82-105,124-166](../../../benchmark/linear/fixtures/generate_ols_fixtures.py#L82-L166)・
  [benchmark/linear/freeze_linear_datasets.py](../../../benchmark/linear/freeze_linear_datasets.py)
- **内容**: ユーザー提案（2026-08-15）。現状`_run_cluster_case`は、CSVに無い疑似グループ列
  `_group`をpandas DataFrameに動的に追加してから独自にstatsmodelsを呼んでいる（項目11参照）。
  3パターンの疑似グループ（既定=行番号%10、不均衡=`imbalanced_cluster_groups(n)`、
  G=2境界=行番号%2）はいずれも`n`のみから決まる決定論的なラベルで`X`/`y`/`weight`の値に
  依存しないため、`freeze_linear_datasets.py`が凍結CSV書き出し時にこれらの列を
  あらかじめ焼き込んでおけば、`generate_ols_fixtures.py`のメインループ
  （`for cov_type in COV_TYPES: result = run(...)`）に`cluster_col`引数付きで
  そのまま乗せられ、`_run_cluster_case`自体（および項目11のcoef/se抽出重複）を
  解消できる可能性がある。
- **Claudeの所感**: 技術的には実現できそうだが、「`generate_linear_dataset()`自体に
  テスト用ラベル列を混在させるか、`freeze_linear_datasets.py`側で後付けするか」という
  設計判断が必要（Claudeは後付け案を推奨、DGP関数を純粋に保てるため）。着手前に
  ユーザー確認が必要な設計判断を含む。
- **気づいた経緯**: 2026-08-15、`generate_ols_fixtures.py`解説後のユーザー提案。
- **状態**: 未対応（設計方針の確認・着手要否はユーザー判断待ち）

### 13. `_run_cluster_case`が`generate_ols_fixtures.py`と`generate_wls_fixtures.py`でほぼ完全に重複、`_run_401ksubs_case`も`run()`の結果辞書構築と大部分が重複

- **対象**: [benchmark/linear/fixtures/generate_wls_fixtures.py:138-179](../../../benchmark/linear/fixtures/generate_wls_fixtures.py#L138-L179)
  （`_run_cluster_case`）・[benchmark/linear/fixtures/generate_ols_fixtures.py:124-166](../../../benchmark/linear/fixtures/generate_ols_fixtures.py#L124-L166)
  （同名関数）・[benchmark/linear/fixtures/generate_wls_fixtures.py:182-260](../../../benchmark/linear/fixtures/generate_wls_fixtures.py#L182-L260)
  （`_run_401ksubs_case`）
- **内容**: `generate_wls_fixtures.py`の`_run_cluster_case`は、`smf.ols`→`smf.wls`（+重み引数）・
  `_meta`に`weight_col`が1行増える以外、`generate_ols_fixtures.py`の同名関数と
  **ほぼ完全に同一のコード**（関数まるごとのコピーに近い）。`_run_401ksubs_case`も、
  `run_statsmodels_benchmark.py`の`run()`が構築する結果辞書（`coef`/`se`/`t_stats`/
  `p_values`/`conf_int`/`r_squared`等13キー）とほぼ同一のロジックを再実装している。
  いずれも`fsize==1`フィルタや`inv_inc`派生列・疑似グループ列といった`run()`が
  対応しない前処理が必要なため`run()`をバイパスしている、項目11・12と同根の問題。
- **Claudeの所感**: 項目12の対応（`cluster_col`引数の拡張・列の事前焼き込み）に加えて、
  `run()`側に「読み込み後にDataFrameへ任意の前処理を挟めるフック」
  （例: `preprocess_fn: Callable[[pl.DataFrame], pl.DataFrame] | None`引数）を
  追加できれば、`_run_cluster_case`・`_run_401ksubs_case`とも`run()`一本化できる
  可能性がある。範囲が広がるため、項目12とまとめて検討する方が効率的。
- **気づいた経緯**: 2026-08-15、`generate_wls_fixtures.py`解説中に発見。
  `generate_logit_fixtures.py`（`smf.logit`+`disp=0`版）・`generate_probit_fixtures.py`
  （`smf.probit`+`disp=0`版）にも同型の`_run_cluster_case`があることを確認済み
  （linear 2ファイル＋nonlinear 2ファイルの計4ファイル全てに存在、2026-08-15）。
  さらに`benchmark/iv/fixtures/generate_iv_fixtures.py`にも同型の`_run_cluster_case`
  （加えて`_run_cluster_g2_case`というG=2境界専用のバリエーション）があることを確認済み
  （linear 2＋nonlinear 2＋iv 1の計5ファイルで確認、2026-08-15）。IVは`x_exog`/`x_endog`/
  `instruments`の3種の列グループを持つ分、他4ファイルより`run()`の引数がやや複雑だが、
  「CSVを読み疑似グループ列を追加→一時CSV書き出し→`run()`呼び出し→`finally`で削除」という
  骨格は完全に同型。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目12と合わせて検討）

### 14. `COV_TYPES`への`cluster`混入がOLS/WLSとLogitで不統一、メインループ自体の共通化余地

- **対象**: [benchmark/nonlinear/fixtures/generate_logit_fixtures.py:55,73-75](../../../benchmark/nonlinear/fixtures/generate_logit_fixtures.py#L55-L75)
  （`COV_TYPES = ["classical", "opg", "hc0", "cluster"]` + メインループ内`if cov_type ==
  "cluster": continue`）と、OLS/WLSの`COV_TYPES`（`cluster`を含まない、`continue`不要）
- **内容**: ユーザー指摘（2026-08-15）。実際の挙動（clusterはメインループで処理せず、
  専用の複数パターンとして後段で個別処理する）はOLS/WLS/Logit/Probit（Probit側も確認済み）
  で完全に同じだが、Logit・Probitだけ`COV_TYPES`に`cluster`を含めた上で`continue`
  スキップしている。意味のある設計差ではなく単なる書き方のブレ。`cluster`を`COV_TYPES`
  から除けば`continue`も不要になりOLS/WLSと統一できる。IVの`generate_iv_fixtures.py`
  （`COV_TYPES = ["classical", "hc0", "hc1", "hac", "cluster"]`）もLogit/Probit側
  （`cluster`を含めて`continue`でスキップ）と同じ書き方であることを確認済み
  （2026-08-15）。
- **Claudeの所感**: 統一後、メインループ本体（`for scenario: for cov_type: run(...)`の
  二重ループ＋辞書構築）自体を`_common.py`に`build_numeric_fixtures(run_fn, scenarios,
  cov_types, **extra_kwargs)`のような共通ヘルパーとして切り出せる可能性がある
  （`weight_col`/`model`等のオプション差分は`**extra_kwargs`で吸収）。OLS/WLS/Logit/Probit/IV
  の5ファイル分の重複を解消できる見込みで、影響範囲は大きい。ただしIVは`x_exog`/
  `x_endog`/`instruments`という3列グループ＋シナリオ別列構成（`X_EXOG_BY_SCENARIO`等）を
  持つ分、他4ファイルより`**extra_kwargs`の吸収範囲が広くなる点は考慮が必要。
- **気づいた経緯**: 2026-08-15、`generate_logit_fixtures.py`解説後のユーザー指摘。
  `generate_iv_fixtures.py`解説時にIVも同型であることを確認済み。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 15. `_run_cluster_case`を`_common.py`へ一般化して集約する案（項目13の発展形）

- **対象**: 項目13で列挙した`_run_cluster_case`の重複箇所全て
- **内容**: ユーザー提案（2026-08-15）。`_run_cluster_case`自体を`_common.py`に切り出し、
  「どの凍結CSVを使うか（prefix）」「どのstatsmodels関数を使うか（`smf.ols`/`smf.wls`/
  `smf.logit`/`smf.probit`）」をパラメータ化すれば、項目13の重複をより根本的に解消できる。
  ただしOLSのG=2境界ケースが`k1`（説明変数を1個に減らした専用データ）を必須とするのに対し、
  Logitは同じデータのままG=2境界ケースが成立する（項目16参照、F検定とLR検定の構造的な違いに
  起因する実質的な差）ため、**完全に同一のコードにはできず、`k1`相当のパラメータ化は必要**。
- **Claudeの所感**: 項目13・14と合わせて検討する価値がある。既存の`k1`引数の発想を
  さらに一般化する形が現実的。
- **気づいた経緯**: 2026-08-15、`generate_logit_fixtures.py`解説後のユーザー提案。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目13・14とまとめて検討）

### 16. `MROZ_FORMULA`が4ファイルで完全重複

- **対象**: [benchmark/nonlinear/fixtures/generate_logit_fixtures.py:63-65](../../../benchmark/nonlinear/fixtures/generate_logit_fixtures.py#L63-L65)・
  `generate_logit_crosscheck_fixtures.py:62-64`・`generate_probit_fixtures.py:67-69`・
  `generate_probit_crosscheck_fixtures.py:62-64`
- **内容**: ユーザーの「`MROZ_FORMULA`を`build_fixtures()`内に移してよいか」という質問を
  きっかけに`grep`で確認したところ、同一の文字列（`"inlf ~ nwifeinc + educ + exper + expersq
  + age + kidslt6 + kidsge6"`）が4ファイルに独立して重複定義されていた。
- **Claudeの所感**: 関数内へのローカル化は可読性向上の意図は理解できるが、逆に他ファイルからの
  import経路を断ってしまう。むしろ`_common.py`等に括り出し4ファイルともそこからimportする
  方向が、項目4（SCENARIOSリストの重複）と同じパターンで整合的。
- **気づいた経緯**: 2026-08-15、`generate_logit_fixtures.py`解説後のユーザー質問をきっかけに発見。
- **状態**: 未対応（着手要否はユーザー判断待ち）
