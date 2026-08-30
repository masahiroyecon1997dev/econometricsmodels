# リファクタリング候補メモ（続き2）

`docs/planning/specs/refactoring-candidates-2.md`に対して並行タスクが
リファクタリング作業に着手した（2026-08-30時点）ため、競合を避けるために
こちらへ新規項目を追記する。番号は独立採番（`refactoring-candidates-2.md`
とは別の番号列として1から開始。統合時の付け替えは着手時にユーザー判断）。
フォーマット・運用方針は元ファイル群と同一（コード解説（`/explain-code`
スキル等）や通常の実装作業の過程で気づいた、リファクタリングの余地がある
箇所を随時記録する場所）。

`refactoring-issue231-progress.md`との違い: あちらは
[#231](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/231)
としてスコープ・フェーズを確定させた上で実施する計画書だが、こちらは
Issue化する前の**気づいた時点での未整理のメモ**を溜める場所。ここに溜まった
項目は、着手時にIssue化するか`refactor`スキルの対象範囲として指定するかを
都度ユーザーが判断する。

3ファイル（`refactoring-candidates.md`・`refactoring-candidates-2.md`・
本ファイル）の統合は別途ユーザー判断で行う。

## 記録フォーマット

各項目は以下を含める。

- **対象**: ファイルパス・行
- **内容**: 何が気になったか
- **気づいた経緯**: どの作業中に気づいたか（日付）
- **状態**: 未対応 / 対応済み（対応したIssue・PR等） / 対応不要と判断（理由）

---

## 一覧

### 1. `x2 = 2 * x1`が計算でなく直書き（IV版）

- **対象**: [tests/test_iv.py:700-707](../../../tests/test_iv.py#L700-L707)
  （`test_singular_first_stage_design_matrix_raises_computation_error`内の
  `x2`リスト）
- **内容**: 完全な多重共線性を作るための`x2 = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]`
  が、`x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]`の2倍を手で書き下したもの
  （コメント`# x2 = 2 * x1（完全な多重共線性）`で意図は明記されている）。
  `refactoring-candidates-2.md`項目82（`test_logit.py`の同型ケース）と
  全く同じ論点がIV側にも存在する。
- **Claudeの所感**: 項目82と同じく、`x2 = [v * 2 for v in x1]`等の計算式に
  すれば「2倍」という関係が値の一致で保証される。実害は小さいが直す場合は
  項目82とまとめて一括対応するのが効率的。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時。
- **状態**: 未対応

### 2. IVの変数集合重複チェック（`y`/`x_exog`/`x_endog`/`instruments`）が個別関数の羅列で組み合わせ数が多い

- **対象**: [tests/test_iv.py:522-618](../../../tests/test_iv.py#L522-L618)
  （`test_y_in_x_exog_raises`から`test_duplicate_x_endog_column_raises`までの
  9関数）
- **内容**: IVは`y`・`x_exog`・`x_endog`・`instruments`という4つの変数集合を
  持つため、「yが他集合に含まれる」（3パターン）・「集合間の重複」
  （3パターン、$\binom{4}{2}$のうちyを除く3組）・「集合内の重複」
  （3パターン、instruments/x_exog/x_endogそれぞれ）を計9個の独立関数として
  1つずつ書いている。各関数はほぼ同じ形（`IV(...).fit()`を`pytest.raises
  (ValidationError)`で包むだけ）で、変えているのは引数の組み立て方のみ。
- **Claudeの所感**: 網羅性自体は高く良い点だが、
  `@pytest.mark.parametrize`で「どの引数にどの重複を仕込むか」を
  タプルのリストとして渡し1関数に統合する余地がある（例:
  `[("x_exog", ["y", "x1"]), ("x_endog", ["y"]), ...]`のような形）。
  ただしOLS/Logit/Probitの`test_cov_type_is_case_insensitive`等、既存の
  parametrize済みテストと違い、ここは「どの引数キーワードに値を渡すか」
  自体が変数のため、素直な`parametrize`よりは`**kwargs`の組み立てが
  やや複雑になる可能性がある。実施するかはコード量と可読性のトレードオフ
  次第でユーザー判断が必要。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時。
- **状態**: 未対応

### 3. `IvResults`に`method`だけでなく`weight_type`も含まれておらず、正規化値を検証する手段が無い

- **対象**: [python_package/econometricsmodels/iv/iv.py:113-373](../../../python_package/econometricsmodels/iv/iv.py#L113-L373)
  （`IvResults`、`cov_type`プロパティはあるが`method`/`weight_type`が無い）
- **内容**: ユーザー指摘（2026-08-30）。`refactoring-candidates-2.md`項目78
  （`LogitResult`に`method`フィールドが無い）と同型の論点がIVにも存在する。
  IVはLogit/Probitと異なり`method`（`"2sls"`/`"gmm"`）に加えて`weight_type`
  （GMMの点推定重み行列の種類、`"unadjusted"`/`"robust"`/`"cluster"`/
  `"kernel"`、`"homoskedastic"`/`"heteroskedastic"`のエイリアスも受け付ける）
  という**もう1軸の入力オプションを持つが、こちらも結果に反映されない**。
  そのため`test_weight_type_is_case_insensitive_and_aliased`
  （[tests/test_iv.py:208-222](../../../tests/test_iv.py#L208-L222)）は
  `cov_type`の`test_cov_type_is_case_insensitive`のように「正規化後の
  ラベルを直接読んで検証する」のではなく、「点推定`params`が2つの呼び方で
  一致すること」という間接的な検証にとどまっている。
- **Claudeの所感**: ユーザー見解に同意。`res.method`・`res.weight_type`を
  追加すれば、(1) 実際にどちらのmethodで推定されたかが結果から確認できる、
  (2) `weight_type`についても`cov_type`と同様の「ラベル直接検証」テストが
  書けるようになる。項目78・67（`WLSOptions`検討）と合わせて設計変更として
  検討するのが良い。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目78と合わせて検討）

### 4. `test_cov_type_label`/`test_cluster_cov_type_label`/`test_nonrobust_is_alias_for_classical`が`test_cov_type_is_case_insensitive`と部分的に重複している

- **対象**: [tests/test_iv.py:144-192](../../../tests/test_iv.py#L144-L192)
- **内容**: ユーザー指摘（2026-08-30、「`test_cov_type_is_case_insensitive`
  があるなら`test_cov_type_label`と`test_cluster_cov_type_label`は冗長では
  ないか。`test_nonrobust_is_alias_for_classical`も含めると1つのテストで
  確かめられそう」）を受けて確認。
  - `test_cov_type_label`は`["classical", "hc0", "hc1", "hc2", "hc3",
    "hac"]`という**カノニカル表記そのまま**を1つずつ試しており、
    `test_cov_type_is_case_insensitive`（大文字混じりの表記→カノニカル値
    への変換を確認）とは厳密には検証内容が異なる（カノニカル表記自体を
    渡した場合の動作は`test_cov_type_is_case_insensitive`のparametrizeに
    含まれていない）。ただし`cov_type`のパース関数がまず正規化（lower化等）
    してから比較する設計であれば、カノニカル表記もその他の表記も同じ
    コードパスを通るはずで、実務上の独立した検証価値は薄い。
  - `test_cluster_cov_type_label`は`cluster_col`付きの`cluster_dataset`
    フィクスチャが必要なため、`test_cov_type_is_case_insensitive`に
    そのまま`"cluster"`/`"CLUSTER"`を追加するには、そのケースだけ
    データセット・追加オプションを出し分ける特別扱いが必要になる
    （現状`test_cov_type_is_case_insensitive`はそこまでしていない、
    かつ`"cluster"`/`"CLUSTER"`表記の大文字小文字非依存性自体は
    どのテストからも検証されていない。test-coverage-candidates.md
    項目53参照）。
  - `test_nonrobust_is_alias_for_classical`は「ラベル」ではなく
    「計算結果（標準誤差）の一致」を確認しており、`test_cov_type_
    is_case_insensitive`（ラベルの変換のみ確認）とは検証対象が異なる
    ため、完全な重複ではない。
- **Claudeの所感**: `test_cov_type_label`は`test_cov_type_is_case_
  insensitive`のparametrizeに`("classical", "classical")`等カノニカル
  表記のケースを追加すれば統合でき、重複解消の価値がある。
  `test_cluster_cov_type_label`は上記の理由で単純併合はしにくく、
  「クラスター専用のケース」として残すか、`test_cov_type_is_case_
  insensitive`に特別扱いの分岐を追加するかはトレードオフがある
  （後者を選ぶなら`"CLUSTER"`表記の大文字小文字非依存性も同時に
  埋められる）。`test_nonrobust_is_alias_for_classical`は検証対象が
  異なるため独立して残すのが妥当と考える。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 5. `test_hac_time_col_reorders_rows_before_computing_lags`のデータ生成スタイルが`test_ols.py`の同名テストと不整合

- **対象**: [tests/test_iv.py:358-403](../../../tests/test_iv.py#L358-L403)
  （IV版、`shuffled_df`を`ordered_df`から`perm`リストで機械的に構築）と
  [tests/test_ols.py:411-445](../../../tests/test_ols.py#L411-L445)
  （OLS版、`ordered_df`/`shuffled_df`とも値を直書き）
- **内容**: ユーザー指摘（2026-08-30）を受けて確認。IV版のdocstringは
  「`test_ols.py`の同名テストと同じ発想をIVに適用」と書いているが、実際の
  データ生成方法はOLS版（両方とも直書き）とIV版（`ordered_df`は直書き、
  `shuffled_df`は`perm = [3, 1, 5, 2, 4, 0, 7, 6]`を使ったリスト内包表記で
  `ordered_df`から計算）で異なっていた。
- **Claudeの所感**: 実害はない（`shuffled_df`が`ordered_df`の並べ替えである
  ことが明示的に保証される分、IV版の書き方の方がむしろ意図が読み取りやすい
  という見方もできる）が、「同じ発想」と明記しているdocstringとの整合性、
  および2つのファイル間のスタイル統一という観点では気になる点。どちらの
  スタイルに統一するかはユーザー判断（個人的には計算で導出するIV版の方が
  「本当に並べ替えただけ」であることが読み手に伝わりやすく好ましいと思う）。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時のユーザー指摘。
- **状態**: 未対応

### 6. `test_gmm_weight_type_options_run`が`test_weight_type_is_case_insensitive_and_aliased`と部分的に重複している

- **対象**: [tests/test_iv.py:415-432](../../../tests/test_iv.py#L415-L432)
  と[tests/test_iv.py:195-222](../../../tests/test_iv.py#L195-L222)
- **内容**: ユーザー指摘（2026-08-30）を受けて確認。
  `test_gmm_weight_type_options_run`は`["unadjusted", "robust", "cluster",
  "kernel"]`の4値それぞれが成功パスで動作することを確認する。
  `test_weight_type_is_case_insensitive_and_aliased`のparametrizeには
  `"ROBUST"→"robust"`・`"KERNEL"→"kernel"`という**大文字小文字変換の
  無い（実質恒等関数の）ケース**が含まれており、これらは間接的に
  「`weight_type="robust"`/`"kernel"`が成功パスで動作すること」も
  再確認してしまっている。ただし**`"cluster"`は`test_weight_type_
  is_case_insensitive_and_aliased`のparametrizeに一切含まれていない**
  ため、重複は`unadjusted`/`robust`/`kernel`の3値に限られ、`cluster`は
  `test_gmm_weight_type_options_run`だけがカバーする独自のケースであり
  完全な重複ではない。
- **Claudeの所感**: 3値分の軽微な重複はあるが、`cluster`のカバーという
  独自価値があるため、`test_gmm_weight_type_options_run`自体を削除する
  のではなく`cluster`のみに絞る、または現状維持のどちらかが妥当と考える。
  優先度は低い。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時のユーザー指摘。
- **状態**: 未対応（優先度低）

### 7.【要注意・バグ疑い】`const`名衝突チェックが`x_exog`のみに存在し、`instruments`/`x_endog`に含まれる場合は検証されず`first_stage()`・`params`辞書がサイレントに破損しうる

- **対象**: [tests/test_iv.py:621-636](../../../tests/test_iv.py#L621-L636)
  （`test_const_collision_with_include_intercept_raises`、`x_exog`のみ検証）、
  実装側は`engine`/`engine_pybind`のconst名衝突バリデーション箇所（未特定、
  `x_exog`側のみ実装されている可能性が高い）
- **内容**: ユーザー指摘（2026-08-30、「操作変数にconstが入っていたらと
  内生変数に入っていたらも検証したほうがいい（内生変数は2段階目のときに
  問題になるはず）」）を受けて、実際に構築済みパッケージで動作確認した。
  - **`instruments`に`"const"`という名前の列（実際の値は定数ではない）を
    含めた場合**、`fit()`自体はエラーにならず成功するが、
    `first_stage()[endog名].param_names`が`['const', 'x1', 'const', 'z1']`
    のように**`"const"`が2回出現**する。`OlsResults.params`は
    `dict(zip(param_names, params))`で辞書化するため、後から出てくる
    `"const"`（ユーザーの操作変数の係数値）が先の`"const"`（実際の
    切片の係数値）を**サイレントに上書き**する。実測で確認した具体例
    （`tests/fixtures/benchmarks/data/iv_baseline.csv`の`z2`列を
    `"const"`に改名し`instruments=["const", "z1"]`とした場合）:
    `fs.params["const"]`が真の切片係数`0.5193043061061868`ではなく
    `0.555951305782325`（本来`z2`＝改名後`"const"`の係数）を返す。
  - **`x_endog`に`"const"`という名前の列を含めた場合はさらに深刻**で、
    **構造方程式本体の`res.params`辞書からも真の切片の値が消える**。
    実測例（`endog1`列を`"const"`に改名し`x_endog=["const"]`とした場合）:
    `res.param_names`は`['const', 'x1', 'const']`（3要素）だが
    `res.params`辞書は`{'const': -0.1259924352302418, 'x1': ...}`の
    **2キーしか持たない**——真の切片の係数（本来`0.7602470494600461`）が
    完全に失われ、代わりに内生変数`endog1`（改名後`"const"`）の係数で
    上書きされている。
  - 一方、`instruments`に`"const"`という名前の**リテラルに定数な**列
    （全行同じ値）を含めた場合は、二重の定数列による完全な多重共線性で
    `ComputationError`（設計行列が特異）になり実害は無い（が、メッセージが
    `x_exog`側の衝突チェックのような明確な`ValidationError`ではなく
    分かりにくい`ComputationError`になる）。
- **Claudeの所感**: これは単なるテストカバレッジの抜けではなく、
  **`x_exog`側だけに実装されているconst名衝突バリデーションを
  `instruments`/`x_endog`にも拡張すべき、実装側の潜在バグ**だと考える
  （`x_endog`側の症状——構造方程式の主要な推定結果が説明もなく静かに
  消える/上書きされる——は特に深刻）。統計的には「たまたま説明変数の
  1つが`"const"`という列名を持っていた」という現実的にあり得る入力
  （ユーザーがデータの列名を制御できない場面、例えば外部データの
  結合等）で発生しうる。ユーザー指示により本セッションでは記録のみに
  留めるが、対応の優先度は本ファイル中では最も高いと考える。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時のユーザー指摘・
  実機検証で確認。
- **状態**: 未対応（優先度: 高。着手時は`x_exog`と同じ`ValidationError`を
  `instruments`/`x_endog`の`"const"`衝突にも拡張する方向で検討）

### 8. `test_const_collision_with_include_intercept_raises`のデータが手書きで、`refactoring-candidates-2.md`項目81（OLS/WLS/Logitのconst衝突データ共有）にIVも該当しうる

- **対象**: [tests/test_iv.py:625-632](../../../tests/test_iv.py#L625-L632)
- **内容**: ユーザー指摘（2026-08-30）。`refactoring-candidates-2.md`項目81
  （OLS/WLS/Logitの`const`衝突用DataFrameを`_helpers.py`のビルダー関数に
  切り出す提案）と同じ論点がIVにも当てはまるか確認した。ただしIVの
  `const`衝突テストは`x_exog`/`x_endog`/`instruments`の4列構成
  （`y`/`const`/`endog1`/`z1`）であり、OLS/WLS/Logitの2〜3列構成
  （`y`/`const`/`x1`等）より列数・役割が多い。加えて項目7の指摘を反映して
  `instruments`/`x_endog`側の衝突ケースを追加すると、IV側だけデータの
  组み立てパターンが増える。
- **Claudeの所感**: 単純に同じヘルパーを流用するのは列構成の違いから
  難しく、IV用に別途「`const`衝突用DataFrameビルダー」を用意するなら
  項目81のヘルパーとは別物になる可能性が高い。項目81の対応時にIVも
  範囲に含めるかどうかは、その時点でOLS/WLS/Logit側の対応方針が固まって
  から改めて判断するのが良い。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時のユーザー指摘。
- **状態**: 未対応（優先度低、項目81〔`refactoring-candidates-2.md`〕と
  合わせて検討）

### 9.【ドキュメント不整合】`iv-api-design.md`の「`x_endog`/`instruments`は最低1要素を要求する見込み」という記述が実装と食い違っている

- **対象**: [docs/planning/specs/iv-api-design.md:26-28](../../../docs/planning/specs/iv-api-design.md#L26-L28)
  と、[tests/test_iv.py:277-285](../../../tests/test_iv.py#L277-L285)
  （`test_weak_instrument_f_statistics_empty_when_no_endog`、
  `x_endog=[]`かつ`instruments=[]`が実際に成功パスとして存在する）
- **内容**: `tests/test_iv.py`解説時にユーザーから挙がった論点
  （項目10）の調査中に発見。設計ドキュメントは「`x_endog`/`instruments`は
  最低1要素を要求する**見込み**」（＝設計当時の予定、確定ではない書き方）
  としているが、実際に構築済みパッケージで確認したところ、
  `x_endog=[]`かつ`instruments=[]`は`ValidationError`にならず**成功する**
  （実質OLSとして完走する）。CLAUDE.md 14章「既存ドキュメント・issueの
  記述と、実装時に判明した事実が食い違う」に該当する典型例。
- **Claudeの所感**: ドキュメントが「見込み」という未確定表現のまま
  更新されずに残っていた可能性が高い。実装が意図的にこの制約を
  設けなかった（`x_endog=[]`を許容する設計にした）のであれば
  ドキュメント側を実態に合わせて修正すべきだし、逆に本来は制約を
  入れるはずだったのが実装時に漏れたのであれば実装側の検討が必要——
  どちらが正しい経緯かはこのセッションでは分からないため、著者
  （ユーザー）に確認したい。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時のユーザー指摘
  （項目10「`x_endog=[]`のケースはバリデーションエラーにすべきか」）の
  調査中に発見。
- **状態**: 未対応（**要ユーザー判断**: ドキュメント記述を実態に合わせて
  修正するか、実装側に制約を追加するか）

### 10. `x_endog=[]`（内生変数ゼロ、実質OLSへの意図的な縮退）を許容し続けるべきかは設計判断が必要

- **対象**: [tests/test_iv.py:277-285](../../../tests/test_iv.py#L277-L285)
  （`test_weak_instrument_f_statistics_empty_when_no_endog`）
- **内容**: ユーザー指摘（2026-08-30、「このケースはそもそもバリデーション
  チェックでエラーにしたほうがよいのか？確かにOLSに帰着するが、そもそも
  IVを使用することが誤りであるケースになる」）。現状は`x_endog=[]`は
  正常に受理され、`weak_instrument_f_statistics`/`overid_statistic`/
  `wu_hausman_statistic`が意味を持たないため`{}`/`None`になる、という
  設計（項目9のドキュメント不整合とも関連）。
- **Claudeの所感**: 一理あると思う一方、「`IV`クラスにわざわざ
  `x_endog=[]`を渡す」というのは、プログラムから動的に変数リストを
  組み立てる場面（CLAUDE.md 2章の設計方針が重視する使い方）では
  `x_endog`が実行時に空になりうるケースを`OLS`への切り替えなしに
  そのまま`IV`に渡せる、という実務上の利便性にもなりうる。
  「誤用を防ぐ」（`ValidationError`にする）か「柔軟性を許容する」
  （現状維持）かはトレードオフであり、`.claude/rules`にも明確な
  指針が無いためユーザー判断が必要と考える。項目9のドキュメント
  不整合の解消と合わせて、まず「意図的な設計か実装漏れか」を
  確認してから、必要なら本項目の要否を判断するのが良い。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時のユーザー指摘。
- **状態**: 未対応（**要ユーザー判断**、項目9と合わせて検討）
