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

**完了項目の扱い（2026-08-22運用ルール）**: 「対応済み」になった項目はこのファイルから
削除する（コード自体とgit logが原本であり、詳細な対応済み記録を残す必要は無いと判断）。
削除した記録は`refactoring-issue231-progress.md`の進捗スナップショットに要点（対応内容・
コミットハッシュ）のみ残す。一方「対応不要と判断」の項目は、コード上に何も痕跡が
残らない却下判断のため、同じ提案の調査をやり直さずに済むよう1行程度に圧縮して残す
（項目38参照）。番号は削除後も詰め直さない（欠番があっても他項目からの「項目N」
表記自体は維持できる）。ただし**番号を維持しても参照先の内容は消える**ため、
削除対象の項目を他の項目が「項目N」で参照している場合は、削除前にその参照側へ
必要な文脈（何の話か・結論）を埋め込み、削除後も参照側だけで自己完結するようにする。

---

## 一覧

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

### 5. `unknown scenario`検証（`ValueError`）が3系統の`generate_*_dataset`関数で完全重複

- **対象**: [benchmark/linear/generate_linear_datasets.py:62-65](../../../benchmark/linear/generate_linear_datasets.py#L62-L65)・
  [benchmark/nonlinear/generate_nonlinear_datasets.py:106-109](../../../benchmark/nonlinear/generate_nonlinear_datasets.py#L106-L109)・
  [benchmark/iv/generate_iv_datasets.py:106-109](../../../benchmark/iv/generate_iv_datasets.py#L106-L109)
- **内容**: ユーザー指摘（2026-08-15）。`if scenario not in SCENARIOS: raise ValueError(f"unknown
  scenario: {scenario!r}. choose from {SCENARIOS}")`という同型の検証が3ファイルで重複している
  （`generate_*_datasets.py`の`SCENARIOS`リスト自体の重複＝項目4は
  2026-08-22対応済み・コミット済み`d23d9b7`だが、こちらは検証ロジック自体の重複）。
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
  （加えて`_run_cluster_g2_case`というG=2境界専用のバリエーション）・
  `benchmark/iv/fixtures/generate_iv_gmm_fixtures.py`にも同型の`_run_cluster_case`
  （`weight_type`引数が追加されている点のみGMM固有）があることを確認済み
  （linear 2＋nonlinear 2＋iv 2の計6ファイルで確認、2026-08-16）。IVは`x_exog`/`x_endog`/
  `instruments`の3種の列グループを持つ分、他4ファイルより`run()`/`run_gmm()`の引数が
  やや複雑だが、「CSVを読み疑似グループ列を追加→一時CSV書き出し→呼び出し→`finally`で
  削除」という骨格は完全に同型。
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
  （2SLS、`COV_TYPES = ["classical", "hc0", "hc1", "hac", "cluster"]`）もLogit/Probit側
  （`cluster`を含めて`continue`でスキップ）と同じ書き方であることを確認済み
  （2026-08-15）。一方`generate_iv_gmm_fixtures.py`（GMM）は`COV_TYPES = ["classical",
  "hc0", "hc1", "hac"]`と`cluster`を含めずOLS/WLS側のスタイルを採っており、**同じIV系統
  （2SLS/GMM）内でもスタイルが割れている**ことを確認済み（2026-08-16）。
- **Claudeの所感**: 統一後、メインループ本体（`for scenario: for cov_type: run(...)`の
  二重ループ＋辞書構築）自体を`_common.py`に`build_numeric_fixtures(run_fn, scenarios,
  cov_types, **extra_kwargs)`のような共通ヘルパーとして切り出せる可能性がある
  （`weight_col`/`model`等のオプション差分は`**extra_kwargs`で吸収）。OLS/WLS/Logit/Probit/
  2SLS/GMMの6ファイル分の重複を解消できる見込みで、影響範囲は大きい。ただしIV（2SLS/GMM）は
  `x_exog`/`x_endog`/`instruments`という3列グループ＋シナリオ別列構成
  （`X_EXOG_BY_SCENARIO`等）を持つ分、他4ファイルより`**extra_kwargs`の吸収範囲が
  広くなる点、GMMはさらに`weight_type`軸によるネスト（`fixtures[scenario][weight_type]
  [cov_type]`）が加わる点は考慮が必要。
- **気づいた経緯**: 2026-08-15、`generate_logit_fixtures.py`解説後のユーザー指摘。
  `generate_iv_fixtures.py`・`generate_iv_gmm_fixtures.py`解説時にIV系統（2SLS/GMM）が
  互いに異なるスタイルであることを確認済み。
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
  方向が、項目4（`SCENARIOS`リストの重複、2026-08-22対応済み・コミット済み
  `d23d9b7`）と同じパターンで整合的。
- **気づいた経緯**: 2026-08-15、`generate_logit_fixtures.py`解説後のユーザー質問をきっかけに発見。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 17. コメント中のIssue番号（`Issue #231`等）参照が10ファイルに散在

- **対象**: `grep -rn "Issue #[0-9]\+" benchmark/`で確認した以下10ファイル（計30箇所超）
  - [benchmark/_common.py:3](../../../benchmark/_common.py#L3)
  - [benchmark/iv/run_linearmodels_benchmark.py:156,292](../../../benchmark/iv/run_linearmodels_benchmark.py#L156)
  - [benchmark/iv/freeze_iv_datasets.py:55,60](../../../benchmark/iv/freeze_iv_datasets.py#L55)
  - [benchmark/iv/run_ivreg_benchmark.R:25,54,148,175](../../../benchmark/iv/run_ivreg_benchmark.R#L25)
  - [benchmark/iv/fixtures/generate_iv_fixtures.py:108,125,141,181,188,190](../../../benchmark/iv/fixtures/generate_iv_fixtures.py#L108)
  - [benchmark/iv/fixtures/generate_iv_gmm_fixtures.py:79,136,151,164,202](../../../benchmark/iv/fixtures/generate_iv_gmm_fixtures.py#L79)
  - [benchmark/iv/fixtures/generate_iv_crosscheck_fixtures.py:21,25,67,190,211,271,343,350,351,358,363](../../../benchmark/iv/fixtures/generate_iv_crosscheck_fixtures.py#L21)
  - [benchmark/nonlinear/fixtures/generate_logit_fixtures.py:58,160](../../../benchmark/nonlinear/fixtures/generate_logit_fixtures.py#L58)
  - [benchmark/nonlinear/fixtures/generate_probit_fixtures.py:63,170](../../../benchmark/nonlinear/fixtures/generate_probit_fixtures.py#L63)
  - [benchmark/nonlinear/run_glm_crosscheck_benchmark.R:115](../../../benchmark/nonlinear/run_glm_crosscheck_benchmark.R#L115)
- **内容**: ユーザー指摘（2026-08-15）。`generate_iv_datasets.py`解説時に「Issue番号は冗長なので削除したい」との指摘を受け該当1箇所を修正済み（本文はWHYを保持したままIssue番号のみ除去）だったが、同種の参照が上記の通り広範囲に残っている。Issue番号だけを見ても文脈が分からず、GitHub側の該当Issueが将来クローズ・番号体系変更等で参照として陳腐化するリスクがある一方、各コメント自体はWHY（なぜその実装・設計になっているか）を本文中に十分書けているため、Issue番号を削っても情報量は落ちない。
- **Claudeの所感**: `generate_iv_datasets.py`で行ったのと同じ要領（Issue番号を削り、WHYの実質的な記述は残す）で機械的に対応できる規模だが、30箇所超と件数が多いため一括対応が妥当かは要検討（`refactor`スキルの対象として一括実施する候補）。
- **気づいた経緯**: 2026-08-15、`run_linearmodels_benchmark.py`解説着手前のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 18. `fixtures["_meta"]`の構築パターンが11ファイルで共通の型を持つ

- **対象**: `benchmark/*/fixtures/generate_*_fixtures.py`全11ファイル（`generate_ols_fixtures.py`
  ・`generate_ols_crosscheck_fixtures.py`・`generate_wls_fixtures.py`・
  `generate_wls_crosscheck_fixtures.py`・`generate_logit_fixtures.py`・
  `generate_logit_crosscheck_fixtures.py`・`generate_probit_fixtures.py`・
  `generate_probit_crosscheck_fixtures.py`・`generate_iv_fixtures.py`・
  `generate_iv_crosscheck_fixtures.py`・`generate_iv_gmm_fixtures.py`）
- **内容**: ユーザー指摘（2026-08-16）。全ファイルで
  `fixtures["_meta"] = {"method": ..., "generated_at": datetime.now(UTC).isoformat(),
  "primary_reference": ..., "<ライブラリ>_version": <ライブラリ>.__version__, "note": (...)}`
  という共通の型を確認済み。`method`/`generated_at`/`primary_reference`/バージョン情報の
  4項目は機械的なボイラープレートだが、`note`は各ファイルで意味のある個別の文章（数十行に
  及ぶこともある）のため、`note`自体をテンプレート化する意味は無い。
- **Claudeの所感**: `_common.py`に`build_meta(method, primary_reference, version_field,
  version_value, note) -> dict`のような小さなヘルパーを切り出せば、ボイラープレート4項目の
  重複だけを解消できる。効果は限定的（項目19の`__main__`共通化ほどインパクトは無い）。
- **気づいた経緯**: 2026-08-16、`generate_iv_gmm_fixtures.py`解説後のユーザー指摘。
  `generate_ols_crosscheck_fixtures.py`の`_meta`には他ファイルに無い`"purpose"`キーが
  独自に追加されていることも確認済み（共通ヘルパー化の際は`purpose`のような
  ファイル固有キーの扱いも考慮が必要）。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 19. `fixtures/generate_*_fixtures.py`全11ファイルの`__main__`ブロックが完全に重複

- **対象**: 項目18と同じ11ファイル
- **内容**: ユーザー指摘（2026-08-16）。11ファイル全ての`__main__`ブロック
  （`argparse`で`--output`を受け取り→`build_fixtures()`→JSON書き出し→バイト数要約表示）を
  比較したところ、**`--output`のデフォルトパス文字列以外、一字一句完全に同一**だった
  （実測確認済み）。
- **Claudeの所感**: `_common.py`に`run_fixture_cli(build_fixtures_fn, default_output)`の
  ような関数を1つ用意すれば、各ファイルの`__main__`ブロック（8行程度）を
  `run_fixture_cli(build_fixtures, "../../../tests/fixtures/benchmarks/ols.json")`という
  1行にまで削減できる。項目18より効果が大きく、11ファイル分の見通し改善に直結する、
  優先度の高い候補だと考える。**裏付け**（2026-08-16、`freeze_iv_datasets.py`再解説時に
  確認）: `freeze_*.py`側では既に`_common.py`の`run_freeze_cli(freeze_fn,
  default_output_dir, success_message, description=...)`という同種の共通化が
  実現済みで、各`freeze_*.py`の`__main__`は`run_freeze_cli(freeze, ...)`という
  1回の呼び出しに削減されている。`fixtures/generate_*_fixtures.py`側でも同じ
  パターンが適用できる根拠になる。
- **気づいた経緯**: 2026-08-16、`generate_iv_gmm_fixtures.py`解説後のユーザー指摘。実際に
  11ファイル全ての`__main__`ブロックを比較し完全一致を確認済み。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 20. `NUMERIC_SCENARIOS`/`COV_TYPES`がmain/crosscheckフィクスチャファイルのペア間で重複

- **対象**:
  - [benchmark/linear/fixtures/generate_ols_fixtures.py:43-65](../../../benchmark/linear/fixtures/generate_ols_fixtures.py#L43-L65) ↔
    `generate_ols_crosscheck_fixtures.py`（`NUMERIC_SCENARIOS`9要素完全一致、
    `COV_TYPES`↔`R_COV_TYPES`も`["classical","hc0","hc1","hc2","hc3","hac"]`で完全一致）
  - `generate_wls_fixtures.py` ↔ `generate_wls_crosscheck_fixtures.py`（同様に完全一致）
  - `generate_logit_fixtures.py` ↔ `generate_logit_crosscheck_fixtures.py`（`NUMERIC_SCENARIOS`
    6要素完全一致。`COV_TYPES`は`hc1`の有無だけ意図的に非対称——statsmodels側は
    discrete modelでhc1が未実装のため含めず、R側はhc1を主リファレンスとして含める設計）
  - `generate_probit_fixtures.py` ↔ `generate_probit_crosscheck_fixtures.py`（同様）
- **内容**: ユーザー指摘（2026-08-16）。既存項目4（`generate_*_datasets.py`↔
  `freeze_*_datasets.py`間の`SCENARIOS`重複）とは別の、`fixtures/generate_*_fixtures.py`↔
  `fixtures/generate_*_crosscheck_fixtures.py`という**別ファイルペア**での重複を実測確認した。
  4系統・8ファイルにわたって同種の重複が存在する。
- **Claudeの所感**: 値が完全一致するペア（OLS/WLS）は`crosscheck`側で`main`側から
  `import`し直せば単一定義元に統一できる。Logit/Probitの`COV_TYPES`の非対称性
  （hc1の有無）は意味のある設計差のため、そのまま維持しつつ`NUMERIC_SCENARIOS`のみ
  統一する形が妥当。
- **気づいた経緯**: 2026-08-16、`generate_ols_crosscheck_fixtures.py`解説後のユーザー質問を
  きっかけに4系統・8ファイルを横断比較し発見。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 21. `benchmark/_common.py`が用途の異なるヘルパーの寄せ集めになりつつある

- **対象**: [benchmark/_common.py](../../../benchmark/_common.py)（`DATA_DIR`・
  `imbalanced_cluster_groups`・`hac_auto_lag`・`load_frozen_dataset`・`freeze_scenarios`・
  `run_freeze_cli`・`preview_dataset`の7項目が既に混在）
- **内容**: ユーザー指摘（2026-08-16）。現状`_common.py`には「DGP系ヘルパー」
  （`imbalanced_cluster_groups`/`hac_auto_lag`）・「データIO系ヘルパー」
  （`DATA_DIR`/`load_frozen_dataset`/`freeze_scenarios`）・「CLI系ヘルパー」
  （`run_freeze_cli`/`preview_dataset`）という異なる関心事が1ファイルに混在している。
  加えて本セッションで新規提案した項目9（スケール倍率定数）・10（誤差項生成定数）・
  18（`build_meta`）・19（`run_fixture_cli`）・`PREDICT_NEW_DATA`の共通化案（本セッションの
  ユーザー指摘、Issue #131/#132/#222の実装後に必要になる見込み）は、いずれも
  受け皿を素朴に`_common.py`と想定していたが、これらを全部積み増すと肥大化し見通しが
  悪化する。
- **Claudeの所感**: `benchmark/utils/`（ディレクトリ化）を作り、用途ごとに
  `data_io.py`（`DATA_DIR`/`load_frozen_dataset`/`freeze_scenarios`）・
  `cli.py`（`run_freeze_cli`/`run_fixture_cli`/`build_meta`）・
  `dgp.py`（`imbalanced_cluster_groups`/`hac_auto_lag`/スケール・誤差項定数）・
  `predict.py`（`PREDICT_NEW_DATA`系）のように分割する案は妥当だと考える。項目9・10・
  18・19の「`_common.py`に切り出す」という所感は、この分割案が採用された場合は
  それぞれ対応するファイルに置き換える形で読み替える。
- **気づいた経緯**: 2026-08-16、`generate_ols_crosscheck_fixtures.py`解説後のユーザー提案。
- **状態**: 未対応（項目9・10は2026-08-22、`_common.py`ではなく専用の
  [benchmark/_dgp_constants.py](../../../benchmark/_dgp_constants.py)を新設する形で
  対応済み——本項目の「肥大化させたくない」という懸念自体を踏まえた判断。
  項目18・19、および本項目自体〔`utils/`分割案〕は引き続き未対応、着手要否は
  ユーザー判断待ち）

### 22. Rスクリプト冒頭の引数パースパターンが3ファイルで重複（`_common.R`は後処理側のみ共通化済み）

- **対象**: [benchmark/linear/run_lm_crosscheck_benchmark.R:22-32](../../../benchmark/linear/run_lm_crosscheck_benchmark.R#L22-L32)・
  [benchmark/linear/run_lm_predict_crosscheck.R:13-22](../../../benchmark/linear/run_lm_predict_crosscheck.R#L13-L22)・
  [benchmark/iv/run_ivreg_benchmark.R:56-65](../../../benchmark/iv/run_ivreg_benchmark.R#L56-L65)・
  [benchmark/nonlinear/run_glm_crosscheck_benchmark.R:48-62](../../../benchmark/nonlinear/run_glm_crosscheck_benchmark.R#L48-L62)
- **内容**: `commandArgs(trailingOnly = TRUE)`→引数不足チェック（`stop(...)`）→
  `data_path <- args[1]`→`formula_str <- args[2]`→
  `read.csv(data_path, check.names = FALSE)`という冒頭5〜6行のパターンが4ファイルで
  同型。`benchmark/_common.R`は`extract_coef_se`/`wald_f_test`という**後処理側**の
  重複は既に解消済みだが、この**冒頭の引数パース**側は対象になっておらず残っている。
- **Claudeの所感**: `_common.R`に`parse_common_args(args, min_required=2)`のような
  関数を追加すれば解消できそうだが、Rには構造化された戻り値（複数の変数をまとめて
  返す）の慣用的な書き方がPython程スッキリしない（リストで返して`$`で分解する形に
  なる）ため、効果とのバランスは要検討。
- **気づいた経緯**: 2026-08-16、`run_lm_predict_crosscheck.R`解説中に発見。
  `run_glm_crosscheck_benchmark.R`にも同型（`link`引数の追加分岐はあるが冒頭部分は
  同じ）であることを確認済み。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 23. `run_lm_predict_crosscheck.R`を手法非依存の汎用スクリプトにできないか（設計判断候補）

- **対象**: [benchmark/linear/run_lm_predict_crosscheck.R](../../../benchmark/linear/run_lm_predict_crosscheck.R)
- **内容**: ユーザー提案（2026-08-16）。今後WLS（Issue #132）・Logit（#131）・Tobit（#222）
  にも`predict()`のRクロスチェックが必要になる見込みだが、現状の`run_lm_predict_crosscheck.R`
  は名前・置き場所（`benchmark/linear/`）ともにOLS専用に見える。`run_lm_crosscheck_benchmark.R`
  （cov_type/cluster_col/hac_lag/weight_colという複数の位置引数）と単純結合すると引数パースが
  複雑になる一方、`fitted()`/`predict(model, newdata=...)`というR関数自体は`lm`オブジェクトにも
  `glm`オブジェクト（Logit/Probit用）にも共通して使える。
- **Claudeの所感**: 「`run_lm_crosscheck_benchmark.R`に統合する」よりも、
  `run_lm_predict_crosscheck.R`自体を手法非依存の汎用スクリプト（`benchmark/`直下等に移動し、
  `weights`引数を追加すればWLSにもそのまま使える）にする方向の方が筋が良さそうだと考える。
  ただしTobit（打ち切りモデル）は`predict()`の意味自体が変わりうる（打ち切り前の潜在変数か、
  打ち切り後の観測値か）ため、そこだけ別途確認が必要。今すぐ決める話ではなく、Issue #131/
  #132/#222着手時の設計判断になる。
- **気づいた経緯**: 2026-08-16、`run_lm_predict_crosscheck.R`解説後のユーザー提案。
- **状態**: 未対応（Issue #131/#132/#222着手時に判断）

### 24. `_normalize_names`/`_write_csv`/`_run_cluster_case`が5つのcrosscheckフィクスチャファイルで重複

- **対象**: `generate_ols_crosscheck_fixtures.py`・`generate_wls_crosscheck_fixtures.py`・
  `generate_logit_crosscheck_fixtures.py`・`generate_probit_crosscheck_fixtures.py`・
  `generate_iv_crosscheck_fixtures.py`（5ファイル、`grep`で`_normalize_names`/
  `_write_csv`/`_run_cluster_case`の定義箇所を確認済み。**訂正**（2026-08-16）:
  `generate_iv_crosscheck_fixtures.py`のみ`_write_csv`が存在せず、CSV書き出しは
  `df.write_csv(...)`を直接呼ぶ形にとどまる。`_normalize_names`/`_run_cluster_case`の
  2つは5ファイル全てに存在）
- **内容**: 項目13・15（statsmodels側`generate_*_fixtures.py`の`_run_cluster_case`重複）
  とは別に、**Rクロスチェック側**の5ファイルにも同型のヘルパー
  （`_normalize_names`: 切片名を`"const"`に統一、`_write_csv`: 一時CSV書き出し（IV版のみ
  未切り出し）、`_run_cluster_case`: 疑似グループ付与→一時CSV→`_run_r`呼び出し）が
  重複していることを発見した。
- **Claudeの所感**: `_write_csv`は特に単純（3行）で`_common.py`（または項目21の
  `utils/`分割案が採用されるなら適切なファイル）に切り出す価値が高い。
  `_normalize_names`は切片名の正規化ルールが全ファイル共通（`"(Intercept)"`/
  `"Intercept"`→`"const"`）なので同様に共通化できそうだが、対象キー
  （`t_stats`/`p_values`/`conf_int`等の有無チェック）がファイルごとに微妙に
  違う可能性があり要確認。`_run_cluster_case`は項目13・15と同じ設計判断
  （凍結時焼き込み案・`_common.py`集約案）を、`_run_r`ベース版としても
  検討する余地がある。
- **気づいた経緯**: 2026-08-16、`generate_wls_crosscheck_fixtures.py`解説中に発見。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目13・15と合わせて検討）

### 25. `"weight"`列名がマジックストリングとして6箇所に散在

- **対象**: [benchmark/linear/generate_linear_datasets.py:144](../../../benchmark/linear/generate_linear_datasets.py#L144)・
  [benchmark/linear/fixtures/generate_ols_fixtures.py:147](../../../benchmark/linear/fixtures/generate_ols_fixtures.py#L147)・
  `generate_wls_fixtures.py`（81・161・165行目）・
  [benchmark/linear/run_statsmodels_benchmark.py:66](../../../benchmark/linear/run_statsmodels_benchmark.py#L66)・
  [benchmark/linear/fixtures/generate_wls_crosscheck_fixtures.py:82](../../../benchmark/linear/fixtures/generate_wls_crosscheck_fixtures.py#L82)
  （`WEIGHT_COL = "weight"`、ローカル定数）
- **内容**: ユーザー指摘（2026-08-16）。合成データセットの重み列名`"weight"`が、DGP自体
  （`generate_linear_datasets.py`が列を作る箇所）・statsmodels側フィクスチャ生成
  （2ファイル）・Rクロスチェック側（`WEIGHT_COL`というローカル定数）の計6箇所に
  マジックストリングとして独立に散らばっている（`WEIGHT_COL`は他5箇所と何の関連も
  無い独立定義）。
- **Claudeの所感**: `_common.py`（または項目21の`utils/`分割案採用時は適切なファイル）に
  `WEIGHT_COLUMN_NAME = "weight"`のような共有定数を1つ置き、6箇所全てから参照する形に
  統一できる。
- **気づいた経緯**: 2026-08-16、`generate_wls_crosscheck_fixtures.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 26. `build_synthetic_fixtures()`をOLS/WLS crosscheck間で共通化できないか

- **対象**: `generate_ols_crosscheck_fixtures.py`の`build_synthetic_fixtures()`・
  `generate_wls_crosscheck_fixtures.py`の同名関数
- **内容**: ユーザー提案（2026-08-16）。両者の違いは`weight_col`の有無（`_run_r`が既に
  `weight_col: str | None = None`という省略可能な設計のため、OLS側は渡さなければ
  そのまま流用できる）・`formula`・`NUMERIC_SCENARIOS`・`R_COV_TYPES`のみで、
  ループの骨格自体は完全に同型。
- **Claudeの所感**: OLS/WLSの2ファイルだけなら`formula`/`weight_col`/シナリオ/
  cov_typeを引数化するだけで現実的に共通化できそう。Logit/Probitまで含めるなら
  `COV_TYPES`の中身（`opg`の有無等）・シナリオの中身（`near_separation`/
  `scale_variance`は別物）が異なるため、項目14で提案した`build_numeric_fixtures
  (run_fn, scenarios, cov_types, **extra_kwargs)`と合わせた検討が必要。
- **気づいた経緯**: 2026-08-16、`generate_wls_crosscheck_fixtures.py`解説後のユーザー提案。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目14と合わせて検討）

### 27. `formula = "y ~ x1 + x2 + x3"`が6ファイルで重複

- **対象**: `generate_ols_crosscheck_fixtures.py`・`generate_wls_crosscheck_fixtures.py`・
  `generate_logit_fixtures.py`・`generate_logit_crosscheck_fixtures.py`（4箇所）・
  `generate_probit_fixtures.py`・`generate_probit_crosscheck_fixtures.py`（4箇所）
- **内容**: ユーザー指摘（2026-08-16）。合成データセット共通の回帰式`"y ~ x1 + x2 + x3"`が
  同一の文字列リテラルとして6ファイルに独立して重複していることを`grep`で確認済み
  （Logit/Probitのcrosscheck側は1ファイルにつき4回登場）。
- **Claudeの所感**: `_common.py`に`SYNTHETIC_FORMULA = "y ~ x1 + x2 + x3"`のような
  共有定数を置ける。項目16（`MROZ_FORMULA`の重複）と同じパターン。優先度は高くないが
  影響ファイル数は多め。
- **気づいた経緯**: 2026-08-16、`generate_wls_crosscheck_fixtures.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 28. `WOOLDRIDGE_COV_TYPES`相当のリストがOLS/WLS crosscheckで書き方だけ異なる

- **対象**: `generate_ols_crosscheck_fixtures.py`の`datasets`辞書内`hc_types`
  （`["classical", *hc_types]`で展開）・`generate_wls_crosscheck_fixtures.py`の
  `WOOLDRIDGE_COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3"]`（名前付き定数）
- **内容**: ユーザー指摘（2026-08-16）。両者とも最終的に生成される値は
  `["classical","hc0","hc1","hc2","hc3"]`で完全に同じだが、OLS側は
  インライン展開、WLS側は名前付き定数と書き方が異なる。
- **Claudeの所感**: 項目26（`build_synthetic_fixtures`共通化）と合わせて検討すると
  効率的。値が同じなら共有定数化は低リスク。
- **気づいた経緯**: 2026-08-16、`generate_wls_crosscheck_fixtures.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目26と合わせて検討）

### 29. `_add_age_bin`の置き場所、および`_run_wage1_region_cluster_case`相当の切り出し方がOLS/WLSで非対称

- **対象**: [benchmark/linear/fixtures/generate_wls_fixtures.py:263-275](../../../benchmark/linear/fixtures/generate_wls_fixtures.py#L263-L275)
  （`_add_age_bin`定義）・[benchmark/linear/fixtures/generate_wls_crosscheck_fixtures.py:58](../../../benchmark/linear/fixtures/generate_wls_crosscheck_fixtures.py#L58)
  （`from generate_wls_fixtures import _add_age_bin`）・
  [benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py:312-330](../../../benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py#L312-L330)
  （`_run_wage1_region_cluster_case`、独立関数）
- **内容**: ユーザー指摘（2026-08-16）。当初「既存の`freeze_*.py`→`generate_*.py`の
  import慣習と同じなので問題なし」と判断したが誤りだった。`freeze_*.py`→
  `generate_*.py`は**階層的な関係**（freezeがgenerateをデータソースとして使う、
  上流→下流の自然な一方向依存）だが、`generate_wls_fixtures.py`（statsmodels側）と
  `generate_wls_crosscheck_fixtures.py`（R側）は**同格・並列の役割**（どちらも
  凍結済みCSVを読んで異なるリファレンス実装を呼ぶ）であり、一方がもう一方の実装
  詳細に依存する構造は性質が異なる。`_add_age_bin`のdocstring自体もstatsmodels
  固有の理由を含まない（`testing-policy.md`の「実データでのグループ列も検証する」
  という手法非依存の一般方針が理由）ため、`generate_wls_fixtures.py`という
  「statsmodels側」のファイルに置かれていること自体が位置づけの誤り。
  さらに、OLS側は同種の処理（`wage1`の地域ダミーからクラスター列を作る）を
  `_run_wage1_region_cluster_case`という独立関数に切り出しているのに対し、
  WLS側の同種処理（`age_bin`によるクラスター確認）は`build_401ksubs_fixture()`
  内にインラインのままで、切り出し方が非対称。
- **Claudeの所感**: `_add_age_bin`を`_common.py`（または項目21の`utils/`分割案
  採用時は適切なファイル）に移し、OLS側と同じく独立関数として切り出す形に
  揃えるのが妥当。
- **気づいた経緯**: 2026-08-16、`generate_wls_crosscheck_fixtures.py`解説後の
  ユーザー指摘で発覚（当初の判断を訂正）。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 30. `run_glm_crosscheck_benchmark.R`内で列スケーリングによる反転ロジックが2回重複

- **対象**: [benchmark/nonlinear/run_glm_crosscheck_benchmark.R:91-101](../../../benchmark/nonlinear/run_glm_crosscheck_benchmark.R#L91-L101)
  （`observed_bread`）・[benchmark/nonlinear/run_glm_crosscheck_benchmark.R:134-138](../../../benchmark/nonlinear/run_glm_crosscheck_benchmark.R#L134-L138)
  （`opg`分岐）
- **内容**: `scale_variance`シナリオでの見かけ上の特異性を避けるため、「列を各々のノルムで
  正規化→反転→`Σ=D⁻¹(D⁻¹MD⁻¹)⁻¹D⁻¹`の恒等式でスケールを戻す」という同じテクニックが
  同一ファイル内で2回（`observed_bread`関数内、および`opg`分岐のインラインコード）
  ほぼ同じ形で書かれている。
- **Claudeの所感**: `scale_and_invert(M, X または scores) -> 行列`のような小さな
  ヘルパー関数に切り出せそうだが、他ファイルとの重複ではなく同一ファイル内の
  重複のため優先度は低め。
- **気づいた経緯**: 2026-08-16、`run_glm_crosscheck_benchmark.R`解説中に発見。
- **状態**: 未対応（着手要否はユーザー判断待ち、優先度低）

### 31. `generate_logit_crosscheck_fixtures.py`/`generate_probit_crosscheck_fixtures.py`が`_write_csv`の命名規則に暗黙依存

- **対象**: [benchmark/nonlinear/fixtures/generate_logit_crosscheck_fixtures.py:143-147](../../../benchmark/nonlinear/fixtures/generate_logit_crosscheck_fixtures.py#L143-L147)・
  [benchmark/nonlinear/fixtures/generate_probit_crosscheck_fixtures.py:149-153](../../../benchmark/nonlinear/fixtures/generate_probit_crosscheck_fixtures.py#L149-L153)
  （完全に同型のコードが2ファイルに存在することを確認済み、2026-08-16）
- **内容**: `build_synthetic_fixtures()`のメインループが終わった後、
  `baseline_csv = tmpdir / "baseline.csv"`という形で、ループ内で既に
  `_write_csv(df, tmpdir, "baseline")`により書き出し済みのファイルパスを
  **文字列結合で再構築**している。`generate_ols_crosscheck_fixtures.py`は同種の
  処理をループ**内側**（`if scenario == "baseline":`ブロック）に置き、ループ内の
  `df`/`csv_path`変数をそのまま再利用する設計だった。動作上は問題ないが、
  `_write_csv`の命名規則（`f"{name}.csv"`）が将来変わった場合にここだけ気づかれず
  壊れる暗黙の結合になっている。`_run_cluster_case`のシグネチャも、OLS版が`df`と
  `csv_path`の両方を受け取るのに対しLogit版は`csv_path`のみ受け取り内部で
  `pl.read_csv`し直す、という違いがある。
- **Claudeの所感**: OLS版と同じくループ内側で処理し、返り値の`Path`オブジェクトを
  直接使い回す形に統一する方が頑健。項目24（crosscheck側ヘルパー重複）と合わせて
  検討する規模の小さい改善。
- **気づいた経緯**: 2026-08-16、`generate_logit_crosscheck_fixtures.py`解説中に発見。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 32. `<系統>_DIR`/`R_SCRIPT`のパス組み立てパターンが5つのcrosscheckフィクスチャファイルで重複

- **対象**: `generate_ols_crosscheck_fixtures.py`（`LINEAR_DIR`/`R_SCRIPT`/
  `PREDICT_R_SCRIPT`）・`generate_wls_crosscheck_fixtures.py`（`LINEAR_DIR`/
  `R_SCRIPT`）・`generate_logit_crosscheck_fixtures.py`（`NONLINEAR_DIR`/
  `R_SCRIPT`）・`generate_probit_crosscheck_fixtures.py`（同様、要確認）・
  `generate_iv_crosscheck_fixtures.py`（`IV_DIR`/`R_SCRIPT`）
- **内容**: ユーザー指摘（2026-08-16）。`<系統>_DIR = Path(__file__).resolve().parent.
  parent` → `R_SCRIPT = <系統>_DIR / "<Rスクリプト名>"`という2行パターンが5ファイル
  全てで同型（ディレクトリ変数名とRスクリプト名だけが違う）であることを`grep`で
  確認済み。
- **Claudeの所感**: `_common.py`に`resolve_sibling_r_script(caller_file, script_name)
  -> Path`のような小さなヘルパーを切り出せる。効果は小さいが対象ファイル数は多い。
- **気づいた経緯**: 2026-08-16、`generate_logit_crosscheck_fixtures.py`解説後の
  ユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 33. `_normalize_names`の骨格（`fix()`クロージャ＋存在チェック付きキーコピー）は共通化できそうだが対象キーが手法ごとに異なる

- **対象**: 項目24で列挙した5つのcrosscheckフィクスチャファイルの`_normalize_names`
- **内容**: ユーザー提案（2026-08-16）。OLS版（`t_stats`/`p_values`/`conf_int`＋
  `aic`/`bic`/`log_likelihood`/`f_statistic`/`f_p_value`/`r_squared`/
  `r_squared_adj`）とLogit版（`z_stats`/`p_values`/`conf_int`は`conf_low`+
  `conf_high`から組み立て＋`log_likelihood`/`log_likelihood_null`/`aic`/`bic`/
  `lr_statistic`/`lr_p_value`/`pseudo_r_squared`＋`margeff`）を比較したところ、
  関数の**骨格**（`fix()`クロージャ定義→存在チェック付きでキーをコピー）は完全に
  同型だが、対象キーの集合自体はかなり異なる（`conf_int`の組み立て方も違う）。
- **Claudeの所感**: 単純に1関数へ統合するのは難しく、`normalize_names(raw,
  stat_key="t_stats"|"z_stats", extra_keys=[...])`のようなパラメータ化された
  ヘルパーにする形が現実的。
- **気づいた経緯**: 2026-08-16、`generate_logit_crosscheck_fixtures.py`解説後の
  ユーザー提案。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目24と合わせて検討）

### 34. `_run_r()`の`link`引数の扱いがLogit/Probit crosscheckで不統一

- **対象**: [benchmark/nonlinear/fixtures/generate_logit_crosscheck_fixtures.py:67-73](../../../benchmark/nonlinear/fixtures/generate_logit_crosscheck_fixtures.py#L67-L73)・
  [benchmark/nonlinear/fixtures/generate_probit_crosscheck_fixtures.py:67-72](../../../benchmark/nonlinear/fixtures/generate_probit_crosscheck_fixtures.py#L67-L72)
- **内容**: Logit版の`_run_r`は`link: str = "logit"`をデフォルト引数として持つが、
  Probit版は引数を持たず`"probit"`という文字列を`cmd`リスト内に直接ハードコード
  している。どちらのファイルも実際には`link`を呼び出し元から変えて渡すことは
  一度も無い（Logit版は常に`"logit"`、Probit版は常に`"probit"`のまま）ため実害は
  無いが、書き方が統一されていない。
- **Claudeの所感**: 些細な差だが、Logit版の`link`引数は実質使われていない
  デッドウェイトの可能性もある（両ファイルとも文字列直書きに統一するか、
  両ファイルとも引数化するかのどちらかに揃えるのが妥当）。
- **気づいた経緯**: 2026-08-16、`generate_probit_crosscheck_fixtures.py`解説中に発見。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 35. `_run_r()`自体が5つのcrosscheckフィクスチャファイルで個別定義されている

- **対象**: `generate_ols_crosscheck_fixtures.py`・`generate_wls_crosscheck_fixtures.py`・
  `generate_logit_crosscheck_fixtures.py`・`generate_probit_crosscheck_fixtures.py`・
  `generate_iv_crosscheck_fixtures.py`（`grep`で5ファイル全てに`def _run_r`が
  存在することを確認済み）
- **内容**: ユーザー指摘（2026-08-16）。項目24（`_run_cluster_case`等）は`_run_r`を
  土台にした関数の重複を指摘していたが、`_run_r`**自体**は候補として明記していな
  かった。シグネチャの違い（線形系は`weight_col`/`hac_lag`、非線形系は`link`、IVは
  未確認）はあるが、「コマンドライン組み立て→`subprocess.run`→JSON parse→
  `_normalize_names`呼び出し」という骨格は5ファイルで完全に共通。
- **Claudeの所感**: 項目33（`_normalize_names`のパラメータ化）と合わせて、
  `_run_r(r_script, csv_path, formula, cov_type, **extra_args)`のような
  共通版を`_common.py`に切り出せる可能性がある。
- **気づいた経緯**: 2026-08-16、`generate_probit_crosscheck_fixtures.py`解説後の
  ユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目24・33と合わせて検討）

### 37. `suppressMessages`が`run_ivreg_benchmark.R`にしか無く、他3ファイルにJSON破損リスクが残る

- **対象**: [benchmark/iv/run_ivreg_benchmark.R:67-72](../../../benchmark/iv/run_ivreg_benchmark.R#L67-L72)
  （`suppressMessages({library(...)...})`）と、`run_lm_crosscheck_benchmark.R`・
  `run_lm_predict_crosscheck.R`・`run_glm_crosscheck_benchmark.R`（いずれも素の
  `library(...)`のまま）
- **内容**: ユーザー指摘（2026-08-16）。`library()`実行時にRのバージョンや
  パッケージの警告等でメッセージが標準出力に出力されると、`toJSON`の出力に
  混ざってJSONパースが壊れる可能性がある。`run_ivreg_benchmark.R`のみこれを
  `suppressMessages({...})`で防いでいるが、他3ファイルには同じ対策が無く、
  単なるスタイルの不統一ではなく**潜在的な頑健性のギャップ**。
- **Claudeの所感**: 他3ファイルにも`suppressMessages({...})`を追加するのが
  低リスクな対策。
- **気づいた経緯**: 2026-08-16、`run_ivreg_benchmark.R`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 38. `script_dir`特定→`_common.R`の`source()`ブロックは構造的に共通化しにくい（対応不要と判断）

- **対応不要と判断**（2026-08-16）: `run_lm_crosscheck_benchmark.R`・`run_ivreg_benchmark.R`
  にある「自分の場所を特定して`_common.R`を`source()`する」3行を`_common.R`側の
  ヘルパーに切り出そうとすると、そのヘルパーを呼ぶために先に`_common.R`をsourceする
  必要がある循環（鶏と卵）が生じ、構造的に共通化できない（Rに`__file__`相当が無いため）。
  再提案しても同じ結論になる見込み。

### 39. `_normalize_names`の`if key in raw`存在チェックが、実際には常にTrueにしかならないデッドコード（IV版のみ直接アクセス）

- **対象**: [benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py:146-164](../../../benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py#L146-L164)・
  [benchmark/linear/fixtures/generate_wls_crosscheck_fixtures.py:121-137](../../../benchmark/linear/fixtures/generate_wls_crosscheck_fixtures.py#L121-L137)・
  [benchmark/nonlinear/fixtures/generate_logit_crosscheck_fixtures.py:93-117](../../../benchmark/nonlinear/fixtures/generate_logit_crosscheck_fixtures.py#L93-L117)（Probit版も同型）
  と対比した[benchmark/iv/fixtures/generate_iv_crosscheck_fixtures.py:109-130](../../../benchmark/iv/fixtures/generate_iv_crosscheck_fixtures.py#L109-L130)
- **内容**: ユーザー指摘（2026-08-16）。IV版の`_normalize_names`は`raw[key]`で
  全キーへ無条件に直接アクセスするが、OLS/WLS/Logit/Probit版は`if "t_stats" in
  raw:`のような存在チェックを経てからコピーする。差の理由を確認するため
  呼び出し元の`_run_r`・呼び出し先のRスクリプトを確認したところ、OLS/WLSが
  呼ぶ`run_lm_crosscheck_benchmark.R`、Logit/Probitが呼ぶ
  `run_glm_crosscheck_benchmark.R`はいずれも`cov_type`等の分岐に関わらず
  `result <- list(...)`で**毎回全キーを無条件に構築**しており（`t_stats`/
  `p_values`/`conf_int`/`aic`/`bic`/`f_statistic`等が欠ける分岐は存在しない）、
  IV側が呼ぶ`run_ivreg_benchmark.R`も同様に毎回全キーを構築する（該当しない
  値は欠落ではなく`NA_real_`で埋める設計、モジュールコメントに明記）。
  つまり4ファイルとも「Rスクリプトは常に全キーを返す」という同じ前提が
  成り立っており、OLS/WLS/Logit/Probit側の`if key in raw`は`_run_r`の呼び出し元を
  `grep`で確認した限り単一のRスクリプトしか呼んでおらず、実行時に`False`へ
  倒れることが無い（`_normalize_names`は`_run_r`からしか呼ばれていないことも確認済み）。
- **Claudeの所感**: 存在チェックが要る状況（同じ`_normalize_names`を複数の
  異なるRスクリプト・異なるキー集合の出力に対して使い回す等）には現状なって
  いないため、実質的にデッドコードと考えられる。IV版の「直接アクセス＋
  該当しない値は`NA_real_`で明示」という設計の方が、Rスクリプト側の
  「毎回全キーを返す」という実際の契約を素直に反映しており、揃えるなら
  IV版の書き方に統一する方向が妥当。ただし項目33・35（`_normalize_names`/
  `_run_r`自体の共通化）に着手する際にまとめて解消するのが自然で、この項目
  単体で先に着手する優先度は高くないと考える。
- **気づいた経緯**: 2026-08-16、`generate_iv_crosscheck_fixtures.py`解説後、
  `run_ivreg_benchmark.R`の設計との対比についてのユーザー指摘。
- **状態**: 未対応（着手要否・項目33/35との統合方針はユーザー判断待ち）

### 40. `compare_performance.py`が現状OLS専用に書かれており、他手法へ性能比較を拡張する際に共通化できる可能性が高い

- **対象**: [benchmark/performance/compare_performance.py](../../../benchmark/performance/compare_performance.py)
  全体（`_fit_once_engine`/`_fit_once_statsmodels`/`_fit_once_pyfixest`、
  `_build_dataframe`、`run_n_sweep`/`run_k_sweep`/`build_report`の`_meta`構築、
  `_worker`のディスパッチ構造等）
- **内容**: ユーザー指摘（2026-08-22）。現状のスクリプトはOLSの`fit()`呼び出し
  にほぼ全体が特化しており（`OLS`/`OLSOptions`のimport、`y ~ x1 + ...`という
  線形の説明変数構成、`generate_linear_dataset`の呼び出し等）、Logit/Probit/IV等
  他手法の性能比較を今後追加する際に、`_worker`のライブラリ分岐・
  `_run_isolated`のサブプロセス起動・`run_n_sweep`/`run_k_sweep`のスイープ
  構造・`build_report`の`_meta`組み立てパターンといった「OLS固有ではない
  骨格部分」を共通化できる可能性が高い。また`_meta`に`statsmodels_version`/
  `pyfixest_version`等のリファレンス実装バージョンが記録されていない点
  （`testing-policy.md`の`_meta`要件と対照的）も、他手法へ拡張する際に
  併せて見直す余地がある。
- **Claudeの所感**: 現時点ではOLSしか性能比較の実装が無く、共通化すべき
  「本当に変わらない部分」と「手法ごとに変わる部分」の境界が1サンプルからは
  判断しづらい。ユーザー方針として、この項目以外の細かい共通化候補
  （`_fit_once_*`のディスパッチ方式の統一等）は今回は深追いせず、実際に
  次の手法の性能比較を実装するタイミングでまとめて判断する。
- **気づいた経緯**: 2026-08-22、`compare_performance.py`解説（`build_report`
  まで）後のユーザー指摘。
- **状態**: 未対応（次の手法の性能比較スクリプト実装時に着手判断）

### 41. ピークRSSの小数点以下の桁数が`compare_performance.py`と`render_performance_summary.py`で不統一

- **対象**: [benchmark/performance/compare_performance.py:274](../../../benchmark/performance/compare_performance.py#L274)
  （`f"peak_rss={row['peak_rss_kb'] / 1024:.1f}MB"`、標準エラーの進捗ログ用）・
  [benchmark/performance/render_performance_summary.py:27](../../../benchmark/performance/render_performance_summary.py#L27)
  （`_format_rss`、`f"{peak_rss_kb / 1024:.0f}MB"`、Markdown表用）
- **内容**: どちらも同じ「ピークRSS（KB）をMB表示に整形する」処理だが、
  小数点以下の桁数が前者`.1f`・後者`.0f`で異なる。実行時間側（`_format_time`と
  `_measure_point`の進捗ログ）はどちらも`.4f`で揃っているため、RSS側だけの
  不統一に見える。
- **Claudeの所感**: 実害はない（一方は人間向けの進捗ログ、他方はJob Summaryの
  表という別用途で、桁数がずれていても実用上の支障はない）が、同じ量の表示
  精度が揃っていない点は気になる。優先度は低い。
- **気づいた経緯**: 2026-08-22、`render_performance_summary.py`解説後のユーザー指摘。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 43. `tests/`が21ファイル（手法別3〜4×6手法＋共通4ファイル）フラット構造のまま、手法別ディレクトリ分割の要否

- **対象**: `tests/`直下（`conftest.py`・`_assertions.py`・`_helpers.py`・
  `_tolerances.py`・`test_{ols,wls,logit,probit,iv,tobit}*.py`）
- **内容**: ユーザー指摘（2026-08-22）。`benchmark/`は`linear/`/`nonlinear/`/
  `iv/`/`performance/`という系統別ディレクトリ＋ルート直下の共通ファイルという
  構成だが、`tests/`は`docs/planning/specs/refactoring-issue231-progress.md`
  フェーズ3で`tests/api_tests/`という中間ディレクトリを「他のテスト種別が
  今後増える見込みが薄い」という理由でフラット化して以来、手法別のサブ
  ディレクトリを持たない。この決定は「カテゴリ単位の中間ディレクトリ」の
  要否についてのものであり、「手法別サブディレクトリ」の要否を検討した
  形跡はない。
- **Claudeの所感**: `benchmark/<系統>/`はRスクリプト・複数バリエーションの
  generator・`fixtures/`サブディレクトリまで含む質的に深い構造を持つのに対し、
  `tests/`側は`test_<手法>.py`/`test_<手法>_fixtures.py`/
  `test_<手法>_crosscheck.py`という命名規則だけで手法ごとに自然にグループ化
  できる平坦な集合（ファイル名プレフィックスによる整列・`pytest tests/test_ols*`
  的な絞り込みが既に機能する）で、`benchmark/`ほどディレクトリ分割の必然性は
  高くない。現状21ファイルの規模ではフラットのままで見通しの悪さは無いと
  考えるが、今後手法が増える（FE/RE・パネル・VAR等）につれてファイル数が
  線形に増え続けるため、着手基準（何ファイル・何手法を超えたら分割するか）
  を決めておく価値はある。
- **気づいた経緯**: 2026-08-22、`tests/`配下の解説着手前のユーザー質問。
- **状態**: 未対応（現時点では静観が妥当、着手基準の設定自体もユーザー判断待ち）
