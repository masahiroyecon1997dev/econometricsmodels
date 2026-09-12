# リファクタリング候補メモ（続き2）

元々は`docs/planning/specs/refactoring-candidates-2.md`に対して並行タスクが
リファクタリング作業に着手した（2026-08-30時点）ため、競合を避けるために
`refactoring-candidates-3.md`として新規項目を追記していたファイル。旧
`refactoring-candidates.md`は項目44〜46（バグ調査に近い内容のためIssue化、
`refactoring-issue231-progress.md`「Issue化した項目」参照）の削除により
項目が無くなったため削除し、本ファイルをリネームしてその名前を引き継いだ
（2026-09-05）。番号は`refactoring-candidates-2.md`とは別の独立採番
（1から開始。統合時の付け替えは着手時にユーザー判断）。フォーマット・
運用方針は元ファイル群と同一（コード解説（`/explain-code`スキル等）や
通常の実装作業の過程で気づいた、リファクタリングの余地がある箇所を
随時記録する場所）。

`refactoring-issue231-progress.md`との違い: あちらは
[#231](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/231)
としてスコープ・フェーズを確定させた上で実施する計画書だが、こちらは
Issue化する前の**気づいた時点での未整理のメモ**を溜める場所。ここに溜まった
項目は、着手時にIssue化するか`refactor`スキルの対象範囲として指定するかを
都度ユーザーが判断する。

2ファイル（本ファイル・`refactoring-candidates-2.md`）の統合は別途
ユーザー判断で行う。

## 記録フォーマット

各項目は以下を含める。

- **対象**: ファイルパス・行
- **内容**: 何が気になったか
- **気づいた経緯**: どの作業中に気づいたか（日付）
- **状態**: 未対応 / 対応済み（対応したIssue・PR等） / 対応不要と判断（理由）

---

## 一覧

### 3.【Issue化】`IvResults`に`method`だけでなく`weight_type`も含まれておらず、正規化値を検証する手段が無い

→ Issue #307として切り出し済み（2026-09-11）。`refactoring-candidates-2.md`
項目78（`LogitResult`/`ProbitResult`の`method`欠落）と合わせて1つのIssueに
統合した。詳細はIssueを参照。

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

### 7.【Issue化】`const`名衝突チェックが`x_exog`のみに存在し、`instruments`/`x_endog`に含まれる場合は検証されず`first_stage()`・`params`辞書がサイレントに破損しうる

→ Issue #305として切り出し済み（2026-09-11）。実測での再現例等の詳細はIssue
本文に転記済み。

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

### 9.【Issue化】`iv-api-design.md`の「`x_endog`/`instruments`は最低1要素を要求する見込み」という記述が実装と食い違っている

→ Issue #306として切り出し済み（2026-09-11）。項目10（`x_endog=[]`の
許容可否）と統合して1つのIssueにまとめた。詳細はIssueを参照。

### 10.【Issue化】`x_endog=[]`（内生変数ゼロ、実質OLSへの意図的な縮退）を許容し続けるべきかは設計判断が必要

→ Issue #306として切り出し済み（2026-09-11）。項目9と統合済み。詳細はIssueを
参照。

### 11. `test_iv_fixtures.py`と`test_iv_gmm_fixtures.py`の統合可否を検討した結果、完全統合は非推奨・部分的な共通化に留めるべき

- **対象**: [tests/test_iv_fixtures.py:100-155](../../../tests/test_iv_fixtures.py#L100-L155)
  （`_check_result`）と
  [tests/test_iv_gmm_fixtures.py:93-150](../../../tests/test_iv_gmm_fixtures.py#L93-L150)
  （同名の`_check_result`）
- **内容**: ユーザー指摘（2026-08-30、「可能ならtest_iv_gmm_fixtures.pyと
  test_iv_fixtures.pyを統合したい（logit/probitのmethod同様）。ただし
  データセットの性質が異なるなどで統合が難しい場合は無理に統合しない」）
  を受けて両ファイルを比較した。Logit/Probitのケース（コード完全一致、
  項目95）とは異なり、**この2ファイルはコード上も検証範囲上も相当に
  違う**ことを確認した。
  - `_check_result`の骨格（`params`/`std_errors`/`stats`/`p_values`/
    `conf_int`/`r_squared`/`r_squared_adj`/`f_statistic`/`f_p_value`/
    `n_obs`/`df_resid`/`weak_instrument_f_statistics`の検証）はほぼ同型
    だが、キー名が異なる（`ref["t_stats"]` vs `ref["z_stats"]`、
    `sargan_statistic` vs `hansen_j_statistic`）。
  - GMM版は`weight_type`という2SLSに存在しない軸を持ち、フィクスチャの
    ネスト構造自体が異なる（`fixtures[scenario][cov_type]` vs
    `fixtures[scenario]["unadjusted"][cov_type]`）。
  - GMM版だけに存在する専用テスト: `test_kernel_hac_matches_linearmodels`
    （`weight_type="kernel"`×`cov_type="hac"`の組み合わせ）・
    `test_gmm_iterations_matches_linearmodels`（`gmm_iterations`の
    非既定値）・`test_other_weight_types_match_linearmodels`
    （`weight_type`の残り3値）。
  - 2SLS版だけに存在する専用テスト: `test_cluster_g2_matches_linearmodels`
    （G=2境界バグの再現）・`test_card_matches_linearmodels`（実データ）・
    `test_df1_matches_linearmodels`（自由度1境界）・
    `test_perfect_multicollinearity_raises_computation_error`/
    `test_scale_variance_raises_computation_error`（`ComputationError`
    パス）。GMM版にこれらの対応物は無い。
  - `_check_result`自体も、GMM版は`check_overid`という2SLSに無い
    引数（`gmm_iterations=1`時のHansen J不一致という既知の未解明差異を
    避けるための抜け穴）を持つ。
- **Claudeの所感**: ユーザーの想定通り、無理に1ファイルへ統合するのは
  避けるべきと考える。`weight_type`という2SLSに存在しない軸、GMM固有の
  検定（Hansen J vs Sargan）、双方に存在しない専用テスト群があるため、
  Logit/Probitのような「コード完全一致→共通関数抽出」パターンは
  当てはまらない。ただし**`_check_result`の共通部分
  （`params`/`std_errors`/`conf_int`/適合度統計量/`weak_instrument_f_
  statistics`）だけを、キー名（t_stats/z_stats、sargan/hansen_j）を
  引数化した共通ヘルパーとして`_assertions.py`または専用モジュールに
  切り出す余地はある**。ただし現状の重複はコメント込みでも各50行程度と
  小さく、優先度は低いと考える。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_fixtures.py`解説時の
  ユーザー指摘、両ファイルの比較調査で確認。
- **状態**: 未対応（優先度低。完全統合は非推奨、`_check_result`共通部分の
  部分的なヘルパー化のみ検討の余地あり）

### 12.【重要な発見】IVのHC2/HC3がR `ivreg`+`sandwich`で実際に検証可能なことを実機確認した——「参照実装が無い」というドキュメント記述は現在のivregバージョンでは事実と異なる

- **対象**: [docs/planning/specs/iv-api-design.md:110-116](../../../docs/planning/specs/iv-api-design.md#L110-L116)
  （「`hc2`/`hc3`は引き続き外部の参照実装で検証できない...R `ivreg`も同様
  （`hatvalues.ivreg`の実装がソース上コメントアウトされている）」）、
  [tests/test_iv_fixtures.py:19-22](../../../tests/test_iv_fixtures.py#L19-L22)、
  [tests/test_iv_crosscheck.py:33-34](../../../tests/test_iv_crosscheck.py#L33-L34)、
  [engine/src/iv/two_sls.rs:1584-1590](../../../engine/src/iv/two_sls.rs#L1584-L1590)
  （いずれも同じ「参照実装が無い」という前提を踏襲している）
- **内容**: ユーザー指摘（2026-08-30、「HC2/HC3はR側かpythonならpyfixest
  などで検証できそう？Rust側だとパッケージ後の結合テストが未完のまま」）
  を受けて、devcontainerに導入済みのR（`ivreg` 0.6.8、CLAUDE.md 10章参照）
  で直接実機検証した。
  - `Rscript -e 'print(ivreg:::hatvalues.ivreg)'`で実際の関数定義を確認
    したところ、**コメントアウトされておらず、`type="stage2"`
    （既定）で第二段階OLS回帰（`lm(y ~ X̂)`、`X̂`は第一段階の予測値）の
    `hatvalues()`をそのまま返す、正常に動作する実装**だった（`ivreg`の
    `NEWS`ファイルにVersion 0.6-4で`hatvalues.ivreg()`のバグ修正記録が
    あり、少なくとも0.6-4時点で既に存在する関数）。
  - `sandwich::vcovHC(ivreg_fit, type="HC2")`/`type="HC3"`が実際に
    エラーなく計算できることを確認した。
  - **本実装（`X̂`から計算するレバレッジ、`two_sls.rs`の
    `hc_cov_params`）と同じ合成データ（`y ~ x1 + endog1 | x1 + z1 + z2`、
    n=200）で数値を突き合わせたところ、HC2/HC3とも6桁以上の精度で一致**
    した（実測: HC2の`endog1`のSE、本実装`0.13356919386622892`、R
    `0.1335692`。HC3も同様）。R側の`hatvalues.ivreg(type="stage2")`が
    第二段階回帰（`X̂`を設計行列とする`lm`）のレバレッジをそのまま使う
    実装であることをソースで確認しており、本実装が「`X̂`のみから
    レバレッジを計算する」（`two_sls.rs`のdocコメント）としている定義と
    完全に一致する。
  - **原因の推測**: `iv-api-design.md`の記述はIssue #166/#171時点の
    調査に基づくが、CLAUDE.md 10章に記録されている通り`ivreg`は当初
    Debian標準のr-baseでは依存関係を満たせず**インストール自体が
    サイレントに失敗していた**（CRAN APTリポジトリ追加で解消）経緯が
    ある。この調査がその失敗期間中、または古いivregバージョンに基づいて
    行われた可能性が高い。
- **Claudeの所感**: これは単なるテストカバレッジの話ではなく、
  **ドキュメント上の技術的前提そのものが現状のツールチェーンでは
  誤りになっている**、重要度の高い発見だと考える。現状のRust単体
  テスト（`fit_computes_hc2_std_errors_matching_manual_sandwich_
  formula`）は同じ開発者が書いた「手計算オラクル」と本実装を突き合わせる
  だけの**自己参照的な検証**（`testing-policy.md`が警告する「独立性が
  限定的」なパターン）に留まっていたが、今回の発見により**真に独立した
  R `ivreg`実装との数値一致**が確認でき、この懸念を解消できる。
  対応するなら: (1) `test_iv_crosscheck.py`に`hc2`/`hc3`のクロスチェック
  テストを追加する（`benchmark/iv/references/run_ivreg.R`に
  `vcovHC(type="HC2"/"HC3")`を追加）、(2) `iv-api-design.md`3.1節・
  関連するdocstring群（本項目「対象」に列挙した4箇所）の「参照実装が
  無い」という記述を訂正する、の2段階が必要になる。ユーザー指示により
  本セッションでは記録のみ。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_fixtures.py`解説時の
  ユーザー指摘、R実機検証で確認（`ivreg`/`sandwich`とも devcontainerに
  導入済みのものをそのまま使用）。
- **状態**: 対応済み（2026-09-12）。`benchmark/iv/references/run_ivreg.R`に
  hc2/hc3の`vcovHC(type="HC2"/"HC3")`分岐を追加し、
  `generate_iv_crosscheck_fixtures.py`のCOV_TYPESに追加して
  `iv_crosscheck.json`を再生成、`tests/iv/test_iv_crosscheck.py`に
  hc2/hc3をCOV_TYPESとして追加（既存のRTOL_STRICT=1e-8でdf1境界
  シナリオ含め全て通過、実測で許容誤差の追加緩和は不要だった）。
  `iv-api-design.md`3.1節・5.2節・冒頭未決着事項、`test_iv_reference.py`・
  `test_iv_crosscheck.py`のdocstring、`engine/src/iv/two_sls.rs`・
  `gmm.rs`のdocコメントの「参照実装が無い」という誤った記述も訂正した
  （GMM側はivreg非対応という結論は維持し根拠のみ訂正、ユーザー承認済み）。

### 13. `wu_hausman_statistic`の`cov_type="hac"`時のNone原因調査について、R側では既に独立検証済み（Issue #233）という事実が`test_iv_fixtures.py`側のドキュメントに反映されていない

- **対象**: [tests/test_iv_fixtures.py:23-31](../../../tests/test_iv_fixtures.py#L23-L31)
  （「原因未特定、次セッションで別途調査予定」）と対比した
  [tests/test_iv_crosscheck.py:38-44](../../../tests/test_iv_crosscheck.py#L38-L44)
  （「wu_hausman_statistic/wu_hausman_p_valueは全cov_typeでフィクスチャに
  実測値がある（Issue #233。summary(diagnostics=TRUE, vcov.=<関数>)で
  cov_type別のロバスト共分散を診断表に反映できることが判明）」）
- **内容**: ユーザー指摘（2026-08-30、「wu_hausman_statisticの`cov_type=
  "hac"`原因調査をリファクタリング時に行うことを明記＆必要なら別
  パッケージ等での検証をする」）を受けて`test_iv_crosscheck.py`（本セッ
  ションではまだ解説していないファイル）を先読みしたところ、**「別
  パッケージでの検証」は既にIssue #233でR `ivreg`側から実現済み**
  だったことが分かった。`linearmodels`の`wooldridge_regression`が
  `hac`（kernel）のときだけ本実装と一致しない、という原因はまだ未特定の
  ままだが、Rの`ivreg`は`hac`を含む全`cov_type`でWu-Hausman統計量を
  正しく計算でき、本実装の値と（クラスターのp値を除き）一致することが
  既に確認されている。つまり「値そのものが正しいか」という統計的な
  懸念は既に別ルートで解消済みで、残っているのは「なぜ`linearmodels`
  という特定の1パッケージだけ再現できないか」という限定的な原因調査
  のみである。
- **Claudeの所感**: `test_iv_fixtures.py`のモジュールdocstring（「次
  セッションで別途調査予定」）が、既に`test_iv_crosscheck.py`側で
  部分的に解決済みという事実を反映しておらず、今読むと誤解を招く
  （「まだ何も検証されていない」ように読める）。ドキュメントを更新し、
  「Rクロスチェック側で独立検証済み（Issue #233）、`linearmodels`固有の
  原因のみ未解明」という現状に揃えるべきと考える。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_fixtures.py`解説時の
  ユーザー指摘、`test_iv_crosscheck.py`の先読みで確認。
- **状態**: 未対応（ドキュメント更新のみで対応可能、優先度中）

### 14. OLS（およびWLS）のHAC自動ラグ選択が、statsmodels側で`maxlags=1`固定のまま「別途検討事項」として放置されている——IVの`hac_auto_lag()`方式に揃えられる可能性がある

- **対象**: [benchmark/linear/references/statsmodels_ref.py:84-87](../../../benchmark/linear/references/statsmodels_ref.py#L84-L87)
  （`fit_kwargs["cov_kwds"] = {"maxlags": 1}  # ラグ選択方法は別途検討
  事項（issue参照）`）、[tests/linear/test_ols_fixtures.py:60](../../../tests/linear/test_ols_fixtures.py#L60)
  （`HAC_LAG_IN_FIXTURE = 1`、テスト側も同じ固定値を明示的に渡す）と
  対比した[benchmark/common/dgp.py:116-122](../../../benchmark/common/dgp.py#L116-L122)
  （`hac_auto_lag(n)`、IVが使っている「本実装と同じ経験則
  `floor(4*(n/100)**(2/9))`をPython側で独立に計算し明示的に渡す」
  ヘルパー）
- **内容**: ユーザー指摘（2026-08-30、「HACのラグ数自動計算式が本実装と
  リファレンス実装側の両方で揃えられているためテスト側で明示的にラグ数を
  渡す必要が無いなら、OLSでも同様の修正を行えばテスト側で明示的にラグ数を
  渡さなくてもいい？」）を受けて調査した。`statsmodels_ref.py`の
  HAC分岐は`maxlags: 1`という**リテラルな固定値**で、`hac_auto_lag(n)`を
  計算して渡してはいない（コメント「ラグ選択方法は別途検討事項」が示す
  通り、意図的に先送りされたまま）。一方IV（`benchmark/iv/references/
  linearmodels_ref.py`）は`hac_auto_lag(n)`で本実装と同じ式を明示的に
  計算しリファレンス側に渡しており、この式の一致により`tests/
  test_iv_fixtures.py`はテスト実行時に`hac_lags`を渡さなくても
  （`None`＝自動計算）フィクスチャと一致する、という設計になっている
  （前回の解説で説明した通り）。
- **Claudeの所感**: ユーザーの見立て通り、OLS/WLSも`statsmodels_ref.py`
  のHAC分岐を`hac_auto_lag(n)`を使う形に変更すれば、`HAC_LAG_IN_
  FIXTURE`という固定値をテスト側で明示的に渡す仕組み自体が不要になる
  可能性が高い。ただしこれは**フィクスチャの再生成を伴う変更**
  （`maxlags=1`から`hac_auto_lag(n)`に変えると、`n`によっては異なる
  ラグ数になり、フィクスチャ内のHAC関連の数値が変わりうる）ため、
  実施する場合は`benchmark/regenerate_all.py`の再実行とコミットが
  必要になる。Logit/Probitは`cov_type="hac"`に対応していないため対象外
  （`grep`で確認済み）。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_fixtures.py`解説時の
  ユーザー指摘、`statsmodels_ref.py`で確認。
- **状態**: 未対応（フィクスチャ再生成を伴うため優先度中、着手要否は
  ユーザー判断待ち）

### 15. `_rename`使用に関する前提の訂正——IVは既に項目63が提案する「生成時に正規化する」設計を先取りしており、`test_iv_fixtures.py`の`_rename`はむしろ不要な残骸

- **対象**: [tests/test_iv_fixtures.py:107](../../../tests/test_iv_fixtures.py#L107)
  （`_check_result`内の`our_name = _rename(name)`）、
  [benchmark/iv/references/linearmodels_ref.py:236,492](../../../benchmark/iv/references/linearmodels_ref.py#L236)
  （`return "const" if name == "Intercept" else name`、生成時に既に
  正規化している）
- **内容**: ユーザー指摘（2026-08-30、「`_check_result`で`_rename`が
  使われているのでIVも改修できそう（OLSと同じ問題）」）を受けて調査した
  結果、**前提が実際とは逆だった**ことが分かった。`refactoring-
  candidates-2.md`項目63は「statsmodels主リファレンス側の生成スクリプト
  （`run_statsmodels_benchmark.py`）が`Intercept`→`const`正規化を
  **していない**ため、`test_ols_fixtures.py`等が実行時に毎回`_rename`を
  呼ぶ必要がある」という問題だったが、IVの主リファレンス生成スクリプト
  （`linearmodels_ref.py`）は**既に生成時点で`"const"`に正規化済み**
  であることを`tests/fixtures/benchmarks/iv.json`の実際のキー名
  （`["const", "x1", "endog1"]`、`"Intercept"`ではない）で確認した。
  つまり`test_iv_fixtures.py`の`_rename`呼び出しは、既に`"const"`に
  正規化済みの値に対して`rename_intercept("const")`（＝恒等関数として
  素通り）を呼んでいるだけの**実質的な無駄働き（無害だが不要な処理）**
  であり、項目63が指摘する「OLSと同じ問題」がIVにあるわけではない。
  むしろ**IVは項目63が理想としている設計（生成時点での正規化）を
  既に達成している**、逆の立ち位置だった。
- **Claudeの所感**: ユーザーへの訂正が必要な点。項目63の対応
  （`run_statsmodels_benchmark.py`側を生成時正規化に変更）を実施する
  際は、`linearmodels_ref.py`を実装の参考にするとよい。また項目63の
  対応が完了した暁には、`test_ols_fixtures.py`等の`_rename`呼び出しが
  IVと同じく無駄働きになるため、そのタイミングで一括除去を検討する
  価値がある（優先度は低い、実害が無いため）。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_fixtures.py`解説時の
  ユーザー指摘、`iv.json`のキー名で確認。
- **状態**: 未対応（優先度低。項目63の対応と合わせて`_rename`の
  一括除去を検討）

### 18. `COV_TYPES`の定義元が`test_iv_fixtures.py`（自前定義）と`test_iv_gmm_fixtures.py`（生成スクリプトからimport）で非対称

- **対象**: [tests/test_iv_fixtures.py:75](../../../tests/test_iv_fixtures.py#L75)
  （`COV_TYPES = ["classical", "hc0", "hc1", "hac"]`、このファイル自身で
  リテラル定義）と
  [tests/test_iv_gmm_fixtures.py:54-56](../../../tests/test_iv_gmm_fixtures.py#L54-L56)
  （`from benchmark.iv.fixtures.generate_iv_gmm_fixtures import COV_TYPES`、
  生成スクリプト側からimport）
- **内容**: ユーザー指摘（2026-08-30、「2SLS版ではCOV_TYPESをこのファイル
  自身で定義していましたが、GMM版は生成スクリプト側からCOV_TYPESもimport
  しています」も記録しておいてほしい、リファクタリング候補になりうる）。
  両ファイルとも`SCENARIOS`は生成スクリプトからimportして単一の定義元に
  しているが、`COV_TYPES`は2SLS側だけこの原則から外れてリテラルで
  再定義している。
- **Claudeの所感**: `benchmark/iv/fixtures/generate_iv_fixtures.py`側にも
  同名の`COV_TYPES`定義があるはずで（HC2/HC3を除く4値）、GMM版と同じ
  ように`test_iv_fixtures.py`もそちらからimportする形に揃えれば、
  「生成スクリプトでcov_typeの対象範囲を変更した際にテスト側の書き換えを
  忘れる」というリスクを塞げる。軽微な修正で済む部類。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_gmm_fixtures.py`解説時の
  ユーザー指摘。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 19.【Issue化】GMMの推論統計量がz/カイ二乗分布固定であること・過剰棄却問題への対処（CUE/bootstrap/Windmeijer 2005補正）は未検討

→ Issue #312として切り出し済み（2026-09-11）。詳細はIssueを参照。

### 20. `INSTRUMENTS_BY_SCENARIO`/`X_EXOG_BY_SCENARIO`が`test_iv_fixtures.py`と`test_iv_gmm_fixtures.py`に同一内容で重複定義されている

- **対象**: [tests/test_iv_fixtures.py:77-83](../../../tests/test_iv_fixtures.py#L77-L83)
  と[tests/test_iv_gmm_fixtures.py:74-78](../../../tests/test_iv_gmm_fixtures.py#L74-L78)
  （両方とも`INSTRUMENTS_BY_SCENARIO = {"just_identified": ["z1"]}`・
  `X_EXOG_BY_SCENARIO = {"moderate_multicollinearity": [...],
  "high_condition_number": [...]}`という同一内容）
- **内容**: ユーザー指摘（2026-08-30、「まとめてもよいかもしれないので
  記録しておいて」）。この2つの辞書は各シナリオのDGPが「`x_exog`が
  何列か」「`instruments`が何本か」というデータ生成側の性質そのものを
  表しており、テストコード固有のロジックではない。
- **Claudeの所感**: 妥当な指摘。`benchmark/iv/datasets.py`（またはその
  近くの共通モジュール）側に単一の定義を置き、`SCENARIOS`と同じように
  両テストファイルからimportする形に揃えるのが筋が良い。実質的に
  `SCENARIOS`のimportパターンをこの2つの辞書にも広げるだけの、
  リスクの低いリファクタリングだと考える。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_gmm_fixtures.py`解説時の
  ユーザー指摘。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 22. `test_iv_fixtures.py`解説時に指摘した項目のうち複数が`test_iv_gmm_fixtures.py`にも同様に該当する（一括注記）

- **対象**: [tests/test_iv_gmm_fixtures.py](../../../tests/test_iv_gmm_fixtures.py)
  全体
- **内容**: ユーザー指摘（2026-08-30、「実装に関しては`test_iv_fixtures.
  py`と同じ実装をしている個所の指摘はそのまま`test_iv_gmm_fixtures.py`
  にも適用できることを記録してほしい」）。個別に項目を複製すると項目数が
  倍増するため、該当箇所を1項目にまとめて記録する。
  - **項目14**（OLSのHAC自動ラグが`maxlags=1`固定のまま放置）: GMM版も
    `IvOptions.hac_lags`未指定（自動計算）でフィクスチャと一致する設計
    （`test_iv_gmm_fixtures.py:80-81`のコメント）であり、IV側
    （2SLS/GMM共通）は既にこの問題を回避できている、という文脈でOLS/WLS
    側の改善余地の参考になる点は同じ。
  - **項目15**（`_rename`が既に無用の残骸）: GMM版の`_check_result`
    （[tests/test_iv_gmm_fixtures.py:102](../../../tests/test_iv_gmm_fixtures.py#L102)）
    も`our_name = _rename(name)`を使っており、`iv_gmm.json`のキーが
    既に`"const"`に正規化済み（生成元の`linearmodels_ref.py`の
    `_normalize`が2SLS/GMM共通関数のため）なら同じく無用の残骸である
    可能性が高い。
- **Claudeの所感**: 対応する場合は2SLS/GMM両方をまとめて一度に確認・
  修正するのが効率的（同じ`linearmodels_ref.py`・`benchmark/common`の
  ヘルパーを共有しているため）。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_gmm_fixtures.py`解説時の
  ユーザー指摘。
- **状態**: 未対応（項目14・15への追記の代わりにこの1項目に集約、
  着手要否はユーザー判断待ち）

### 23. `weight_type`×`cov_type`の「両方とも非デフォルトで異なる種類」という組み合わせ（例: `cluster`×`hac`、`kernel`×`cluster`）が`kernel`×`hac`以外は未検証——ただし実装は特殊ケース分岐を持たない一般形サンドイッチのため構造的リスクは低いと判断

- **対象**: `engine/src/iv/gmm.rs:119-124`
  （「点推定に実際に使った重み`W`と`cov_type`が指定する`Ω̂`は一般に
  一致しない。そのため...特殊ケースを分岐させず、常に一般形の
  サンドイッチを使う（ユーザー確認済み）」）、
  [tests/test_iv_gmm_fixtures.py:259-278](../../../tests/test_iv_gmm_fixtures.py#L259-L278)
  （`test_kernel_hac_matches_linearmodels`、`kernel`×`hac`のみ専用テスト
  あり）
- **内容**: ユーザー指摘（2026-08-30、「`weight_type`と`cov_type`、
  `weight_type`とシナリオでどれかの組み合わせをやらない影響はない
  か？...問題が起きやすい組み合わせだけ別途やっておくと検出力が
  上がりそうなのだが」）を受けて`gmm.rs`のモジュールdocコメントを
  確認した。実装は「`weight_type=cov_type`相当（効率的GMM）のときに
  `Avar(β̂)=(X'ZΩ̂⁻¹Z'X)⁻¹`へ潰せる特殊ケースを分岐させず、常に一般形の
  サンドイッチ`Avar(β̂) = B⁻¹(X'ZWΩ̂WZ'X)B⁻¹`を使う」という設計
  （`weight_type`と`cov_type`が一致する場合・しない場合を同じコード
  パスで扱う）になっており、「一致するケースだけ動作確認して、
  一致しないケースだけ特別に壊れている」という種類のバグが起きにくい
  構造だと判断できる。とはいえ、現状のテストは実質的に以下の2群しか
  無い。
  1. `weight_type=unadjusted`固定×`cov_type`を4値スイープ
     （`test_matches_linearmodels`）
  2. `weight_type`を3値スイープ×`cov_type=classical`固定
     （`test_other_weight_types_match_linearmodels`）
  3. `weight_type=kernel`×`cov_type=hac`のみの単発テスト
     （`test_kernel_hac_matches_linearmodels`、「両方とも非自明な
     HAC/カーネル系」という組み合わせの唯一の例）
  「`weight_type=cluster`×`cov_type=hac`」「`weight_type=kernel`×
  `cov_type=cluster`」のような、**両方とも非自明かつ互いに異なる種類**
  の組み合わせは、`kernel`×`hac`以外は1つも検証されていない。
- **Claudeの所感**: 一般形サンドイッチという設計自体は良い判断であり
  過度に心配する必要は無いと思うが、ユーザーの懸念にも一理ある——
  「一般形だから安全」という主張自体を裏付けるテストが`kernel`×`hac`
  1点のみというのは心もとない。全組み合わせ（4`weight_type`×6`cov_type`
  ≈24通り）は過剰だが、`cluster`×`hac`のような、実務上もありうる
  もう1〜2パターンを追加すれば費用対効果良く検出力を上げられると考える。
  `weight_type`とシナリオ（DGP）の組み合わせについては、`weight_type`は
  純粋に点推定側の重み選択でありDGPのシナリオ性質（不均一分散・自己相関
  等）との相互作用は`cov_type`ほど強くないと考えられるため、そちらは
  優先度を下げてよいと考える。既定`weight_type="unadjusted"`を主軸に
  シナリオを広くスイープする現状の設計（`IvOptions().weight_type`の
  実際の既定値と一致）は妥当で、`robust`に差し替える積極的な理由は薄い。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_gmm_fixtures.py`解説時の
  ユーザー指摘、`gmm.rs`のモジュールdocコメントで設計を確認。
- **状態**: 未対応（`test-coverage-candidates.md`項目59に具体案を記録、
  着手要否はユーザー判断待ち）

### 24. GMMで`include_intercept=False`にした場合も`first_stage()`が影響を受けることを実機確認——ユーザーの「GMMにfirst_stageの概念が無いから問題にならない？」という予想は誤りで、2SLSと同じ懸念が当てはまる

- **対象**: `python_package/econometricsmodels/iv/iv.py`の`first_stage()`
  （2SLS/GMM共通の`compute_first_stage`経由）
- **内容**: ユーザー指摘（2026-08-30、「GMMのconstがfalseの場合の
  テストが無い。GMMにfirst_stageの概念がないから問題にならない？」）を
  受けて実機確認した。**予想に反し、GMMも`first_stage()`を持ち、
  `include_intercept=False`の影響を受ける**ことを確認した。
  `IV(..., options=IvOptions(method="gmm", include_intercept=False))
  .fit().first_stage()`は正常に動作し、`param_names`から`const`が
  正しく除外されていた。`engine/src/iv/CLAUDE.md`にも「（過去の
  `k_constant`取り違えバグの）影響を受けていたのは`first_stage()`が
  返す`OlsResults`...`method`によらず、`fit()`が常に`compute_first_
  stage`経由で構築するため2SLS/GMM両方に及んでいた」と明記されており、
  GMMも2SLSと全く同じ第一段階回帰の配線コードを共有している。
  つまり本項目は`test-coverage-candidates.md`項目50・51（2SLSの
  `include_intercept=False`数値照合・`first_stage()`への伝播が未検証）
  と**全く同種の懸念がGMM側にも当てはまる**、という追加確認である。
- **Claudeの所感**: ユーザーの初期の予想（「GMMには影響しないのでは」）
  は妥当な直感だったと思う（GMMは第一段階の解釈がやや異なるため）が、
  実装上は2SLSと全く同じ配線コードを共有しているため、実際には同程度に
  重要な確認事項だと考える。対応する場合は項目50・51の対応と同時に、
  GMM版（`test_iv_gmm_fixtures.py`）にも同種のテストを追加するのが
  効率的。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_gmm_fixtures.py`解説時の
  ユーザー指摘、実機検証で確認。
- **状態**: 未対応（`test-coverage-candidates.md`項目60参照、着手要否は
  ユーザー判断待ち）

### 25. `refactoring-candidates-2.md`項目51・79（Issue #231フェーズ4コメント残置）がIV関連4ファイル全てに該当する（18箇所）

- **対象**: `tests/test_iv.py`（6箇所: 175, 185, 214, 654, 674, 784行目）・
  `tests/test_iv_fixtures.py`（3箇所: 226, 252, 276行目）・
  `tests/test_iv_gmm_fixtures.py`（4箇所: 125, 237, 265, 285行目）・
  `tests/test_iv_crosscheck.py`（5箇所: 22, 283, 324, 351, 403行目）
  （`grep -n "Issue #231"`で確認、計18箇所）
- **内容**: ユーザー指摘（2026-08-31、「Issue #231フェーズ4などコミット
  番号が残っているので削除」）を受けて確認。`refactoring-candidates-2.md`
  項目51（OLS、3箇所）・項目79（Logit、11箇所）で既に確立済みの
  「Issue番号の経緯コメントが本文中に残置されている」パターンが、
  IV関連4ファイル全てにも一貫して存在する。項目79の追記でProbitにも
  同様の指摘があったことを踏まえると、この問題はIssue #231フェーズ4の
  テスト拡充作業で書かれた全ファイルに共通する残置パターンだと考えられる。
- **Claudeの所感**: 項目51・79で確立済みの方針（Issue番号のみ削除し、
  テストの意図の説明（「OLS/WLS/Logit/Probitの同名テストと同型」等）は
  残す）をそのまま適用できる。`refactoring-candidates-2.md`は現在別の
  並行タスクが対応中で凍結中のため、項目51・79とまとめて対応する
  タイミングは、その並行タスクの状況を踏まえてユーザーが判断するのが
  良いと考える。
- **気づいた経緯**: 2026-08-31、`tests/test_iv_crosscheck.py`解説時の
  ユーザー指摘、`grep`で確認。
- **状態**: 未対応（`refactoring-candidates-2.md`項目51・79と統合して
  対応するのが効率的、着手タイミングはユーザー判断待ち）

### 26. `test_iv_crosscheck.py`にGMMのRクロスチェックが無い件——v1時点では意図的な例外規定だったが、今後のGMM拡張（C統計量等）に合わせてテストも拡張予定

- **対象**: [docs/planning/specs/iv-api-design.md:207-212](../../../docs/planning/specs/iv-api-design.md#L207-L212)
  （「5.3 GMMのRクロスチェック省略（例外規定）」、「GMMは`ivreg`が
  対応していないため、Python（`linearmodels`）のみで検証しRクロス
  チェックを省略することを許容する。`panel-api-design.md`5.3の
  ハウスマン検定と同様...」）
- **内容**: ユーザー指摘（2026-08-31、「benchmarkで指摘したかもしれ
  ないが、`test_iv_crosscheck.py`にGMMの検証が抜けている」）を受けて
  確認したところ、v1実装時点では見落としではなく`iv-api-design.md`
  5.3節に明記された**意図的な設計判断（例外規定）**だった
  （`ivreg`パッケージ自体がGMM推定に対応していないというツール側の
  制約が根拠、パネルデータのHausman検定と同種の前例あり）。
  **追記（2026-08-31、ユーザーからの追加情報）**: ただし、IVの
  GMM実装は当初のスコープのまま固定されるわけではなく、今後C統計量
  （GMMの部分過剰識別検定、Hansen J検定を拡張した診断統計量）等の
  追加や、FEIV（パネル固定効果×IV）実装が予定されている
  （Issue #247・#255・#276に言及あり、とのこと）。`gh issue view`で
  実機確認したところ、**Issue #247**（「IV: 複数内生変数対応後、
  Cragg-Donald統計量のv1スコープ外判断を再検討する」）は直接該当を
  確認できた。**Issue #276**（実装ロードマップの「地図」issue）は
  今後の実装順序に「パネル（FE/RE）」を2番目（GLSの次）に位置づけて
  おり、これがFEIVに relatesすると考えられる。**Issue #255**
  （「Tobit: 性能比較ベンチマーク（vs R）を追加する」）は`gh issue
  view`で確認した限りTobitの性能比較ベンチマークに関する内容で、
  本項目のGMM拡張との直接的な関連は本文からは確認できなかった
  （番号の記憶違いの可能性がある旨、念のため申し添える）。
- **追記（2026-08-31、ユーザー訂正）**: 上記の「Issue #255」は
  ユーザーの記憶違いで、正しくは**Issue #249**（「IV(GMM): C統計量
  （difference-in-Hansen統計量）による内生性検定の追加を検討する」）
  だった。`gh issue view`で本文を確認したところ、GMMには現状
  2SLSの`wu_hausman_statistic`に相当する内生性検定が無いという課題
  意識から、C統計量（疑わしい変数を内生・外生それぞれとして扱った
  場合のHansen J統計量の差を取る手法、Eichenbaum-Hansen-Singleton
  1988が起源、Stata`ivreg2`の`C statistic`）の追加が検討されている
  ことを確認した。これは本項目（GMMのRクロスチェック拡張）と直接
  関連する将来拡張であり、C統計量が実装される際は当然その数値検証
  （linearmodels主リファレンス・Rクロスチェックの両方）も必要になる。
- **Claudeの所感**: v1時点の判断自体は妥当だったと考えるが、GMMの
  スコープが今後拡張される計画がある以上、**Rクロスチェックの省略も
  永続的な決定ではなく「v1時点のスコープに対する暫定的な例外」**と
  再定義すべきだと考える。C統計量・FEIV等の拡張が実装される際は、
  `ivreg`（またはR `gmm`パッケージ等の代替）でのクロスチェック追加も
  合わせて検討する必要がある。
- **気づいた経緯**: 2026-08-31、`tests/test_iv_crosscheck.py`解説時の
  ユーザー指摘・追加情報、`gh issue view`で関連Issueを確認。
- **状態**: 要対応（将来のGMM拡張〔C統計量・FEIV等、Issue #247・#276他〕
  実装時に、Rクロスチェックの追加を合わせて検討する。現時点では
  記録のみ・実装着手はしない）

### 27. `tests/test_tobit.py`が`tests/nonlinear/`ではなく`tests/`直下に残っている——CLAUDE.mdの「系統ディレクトリ未確定」という注記が、サブディレクトリ確定後も更新されていない

- **対象**: [tests/test_tobit.py](../../../tests/test_tobit.py)（配置場所）、
  [CLAUDE.md](../../../CLAUDE.md)3章のリポジトリ構成表（「`test_tobit.py`
  # 系統ディレクトリ未確定のため当面ルート据え置き」という注記）
- **内容**: ユーザー指摘（2026-08-31、解説前指摘として「ディレクトリ
  `tests/nonlinear`に入れるべきではないか」）を受けて確認した。CLAUDE.md
  の注記自体は意図的な暫定措置として記録されていたが、これは
  `tests/linear/`・`tests/nonlinear/`・`tests/iv/`という系統別
  サブディレクトリ自体がまだ無かった時点（あるいはその過渡期）に書かれた
  可能性が高い。既に以下の2点が確定している。
  1. `tests/`側のディレクトリ分割自体は完了済み（コミット`7e93052`、
     `refactoring-candidates-2.md`項目68関連）。
  2. 実装側（`python_package/econometricsmodels/nonlinear/tobit.py`）は
     既に`nonlinear`配下に置かれており、Tobitを`nonlinear`系統として
     扱うこと自体は実装レベルで既に確定している。
  この2点を踏まえると、CLAUDE.mdの「系統ディレクトリ未確定」という
  理由づけは現状と整合しておらず、`tests/test_tobit.py`を
  `tests/nonlinear/test_tobit.py`へ移動する障害は無いように見える。
- **Claudeの所感**: ユーザー指摘に同意する。CLAUDE.mdの注記を更新した
  上で、`test_tobit.py`を`tests/nonlinear/`へ移動するのが筋が良いと
  考える。ただし`tests/`のディレクトリ分割自体を進めている並行タスク
  （`refactoring-candidates-2.md`項目68関連）が存在するため、二重作業・
  競合を避けるためこのタスクとの調整（着手タイミング・担当）が必要。
- **気づいた経緯**: 2026-08-31、`tests/test_tobit.py`解説前のユーザー指摘。
- **状態**: 未対応（着手要否・タイミングはユーザー判断待ち、`tests/`
  ディレクトリ分割の並行タスクとの調整が必要）

### 28. `tests/test_tobit.py`のモジュールdocstringに「Issue #227」という経緯コメントが残っている

- **対象**: [tests/test_tobit.py:4](../../../tests/test_tobit.py#L4)
  （「主リファレンス...との厳密な数値比較は別途実施する
  （`test_logit_fixtures.py`/`test_logit_crosscheck.py`と同じ役割分担、
  Issue #227）」）
- **内容**: `refactoring-candidates-2.md`項目51・79・本ファイル項目25
  （Issue #231フェーズ4のコメント残置、OLS/Logit/IV関連ファイルで
  確認済み）と同一パターンが`test_tobit.py`にも存在する（Issue番号は
  #231ではなく#227だが、性質は同じ「経緯コメントの残置」）。
- **Claudeの所感**: 項目25・51・79とまとめて一括対応するのが効率的。
  番号のみ削除し、役割分担の説明自体（「`test_logit_fixtures.py`/
  `test_logit_crosscheck.py`と同じ役割分担」）は有用な情報のため残す
  という、既存の方針をそのまま適用できる。
- **気づいた経緯**: 2026-08-31、`tests/test_tobit.py`解説時に確認。
- **状態**: 未対応（項目25・51・79とまとめて対応するのが効率的、
  着手タイミングはユーザー判断待ち）

### 29. `rel=1e-4`という許容誤差がLogit/Probit/Tobitの`test_method_option_converges_to_same_params`に同一値で直書きされている（リポジトリ全体）

- **対象**: [tests/nonlinear/test_logit.py:59](../../../tests/nonlinear/test_logit.py#L59)・
  [tests/nonlinear/test_probit.py:59](../../../tests/nonlinear/test_probit.py#L59)・
  [tests/test_tobit.py:64](../../../tests/test_tobit.py#L64)
- **内容**: ユーザー指摘（2026-08-31、「`rel=1e-4`と閾値が直書きされて
  いる。可能なら`nonlinear`で統一したほうが良い」）を受けて`grep`で確認
  したところ、3ファイルとも全く同じ`rel=1e-4`が直書きされていた
  （Tobit固有の問題ではなくLogit/Probit解説時から既に存在していた
  リポジトリ全体のパターン）。
- **Claudeの所感**: `tests/_tolerances.py`（各ファイルの`RTOL`/`ATOL`
  定数の一元管理に使われている既存の仕組み）に`method_convergence_rtol`
  のような1つの定数を追加し、3ファイルから参照する形に揃えるのが
  筋が良い。実施コストは低い部類。
- **気づいた経緯**: 2026-08-31、`tests/test_tobit.py`解説時のユーザー
  指摘、`grep`でLogit/Probitにも同じ値があることを確認。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 30. TobitのAIC/BICが`k+1`（`σ`を含む）でペナルティを掛ける一方`df_model`は`σ`を含まない——バグではなく統計的に正しい設計だが、ドキュメント・テストが無い

- **対象**: [engine/src/nonlinear/tobit.rs:1328-1333](../../../engine/src/nonlinear/tobit.rs#L1328-L1333)
  （`let aic = -2.0 * llf + 2.0 * ((k + 1) as f64);`・
  `let df_model = k - k_constant;`、`k+1`は`β`（`k`個、切片込み）＋`σ`、
  `df_model`は`k`から`k_constant`（0か1）を引いた「傾き係数の数」のみ）
- **内容**: ユーザー指摘（2026-08-31、「`df_model`の個数に`sigma`は
  含めていないが、AIC, BICの計算には係数+1（sigma分）して計算している
  のか？」）を受けて`engine/src/nonlinear/tobit.rs`を確認した。
  ご指摘の通り、`aic`/`bic`は`k+1`（`σ`を含む、実際に最尤推定で
  推定されたパラメータの総数）でペナルティを計算する一方、`df_model`
  は`k - k_constant`（`σ`を含まない、Wald検定で同時に検定する傾き
  係数の数）で、**この2つは異なる目的のために異なる数を使っている**。
  これはバグではなく統計的に正しい: AIC/BICは「実際に推定した
  パラメータの数」でモデルの複雑さを罰するのに対し、`df_model`は
  「Wald検定で同時仮説検定する係数の数」であり、`σ`は検定対象の
  係数（`β`）ではないため含まれない。`testing-policy.md`が触れている
  「RのAIC()/BIC()標準関数は残差分散を推定パラメータ1個として追加で
  カウントする（k+1）が、statsmodels・本実装は回帰係数の数`k`のみを
  使う」というOLSの慣習差の話（Rが`k+1`、本実装が`k`）とは**別の
  文脈**であることに注意——OLSの`σ²`は最尤推定/最小二乗の対象になって
  いない事後計算量だが、Tobitの`σ`は最尤推定で`β`と同時に最適化される
  真のパラメータであるため、`k+1`でカウントするのが元々正しい
  （Rの慣習に合わせたわけではない）。
- **Claudeの所感**: 実装は正しいと判断するが、(1) この`k+1` vs
  `df_model`の使い分けを説明するコメント・ドキュメントが無い、
  (2) `test_tobit.py`に`aic`/`bic`の検証テストが1つも無い、という
  2点は気になる。前者はコメントを1行追加する程度で解消できる。
  後者は数値照合フィクスチャ（`test_tobit_fixtures.py`、Issue #227で
  今後実施予定）が整備されればそちらでカバーされる見込みだが、
  構造確認レベル（`res.aic`/`res.bic`が有限の値を返す、`k`が変われば
  値も変わる、程度）のテストは`test_tobit.py`側にあってもよいと考える。
- **気づいた経緯**: 2026-08-31、`tests/test_tobit.py`解説時のユーザー
  指摘、`engine/src/nonlinear/tobit.rs`で確認。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 32. `refactoring-candidates-2.md`項目76（テストファイル内のセクション見出し・順序の不統一）が`test_tobit.py`にも同様に該当する

- **対象**: [tests/test_tobit.py](../../../tests/test_tobit.py)全体
  （見出しは`成功パス・API構造`/`predict() / censoring_fit_check()`/
  `marginal_effects()`/`エラーハンドリング`の4つのみ）
- **内容**: ユーザー指摘（2026-08-31、「`test_marginal_effects_
  unknown_at_raises`と`test_marginal_effects_confidence_level_out_
  of_range_raises`はエラーハンドリングセクションだと思う」・
  「`test_cov_type_label`等はAPI構造セクションにあったほうがいい」）
  を受けて確認した。`marginal_effects()`関連の`ValidationError`テスト
  （`test_marginal_effects_unknown_at_raises`・`test_marginal_effects_
  unknown_target_raises`・`test_marginal_effects_confidence_level_
  out_of_range_raises`）が「`marginal_effects()`」見出し配下（＝
  「エラーハンドリング」見出しより前）に置かれている点、`cov_type`
  関連テスト（`test_cov_type_label`等）が「エラーハンドリング」見出し
  配下に置かれている点は、**`refactoring-candidates-2.md`項目76が
  `test_logit.py`について既に指摘していたのと全く同じパターン**
  だった（`grep`で確認したところLogit/Probitも同一構造）。
- **Claudeの所感**: Tobit固有の問題ではなく、項目76で確立済みの
  「機能単位の見出し配下に、その機能に関する成功パス・エラーパス両方の
  テストをまとめる」という設計（意図的か否かは項目76時点で結論が
  出ていない）が、Tobitにもそのまま引き継がれている。項目76の対応
  方針が決まった際に、Tobitも合わせて統一するのが良い。
- **気づいた経緯**: 2026-08-31、`tests/test_tobit.py`解説時のユーザー
  指摘、`grep`でLogit/Probitとの一致を確認。
- **状態**: 未対応（項目76〔`refactoring-candidates-2.md`〕と統合して
  対応するのが効率的、着手タイミングはユーザー判断待ち）

### 35.【完了】Logit/Probitの初期値・特異性検出をTobit方式（`method`共通のQRランクチェック＋OLSベース初期値）に統一 → Issue #279（2026-09-08 実施）

- **完了内容**: `nonlinear::common::checked_design_matrix_qr`（列ピボットQR＋相対閾値ランク
  チェック、logit/probit/tobit 共有）と `ols_based_initial_params`（標準化LPM解 + IRLS 1
  ステップ相当のスケール補正、logit/probit 共有）を新設。Logit/Probit の `fit()` を
  ゼロベクトル初期値 → この warm start に変更し、多重共線性検出を `method` 非依存の単一経路
  （前段QR → `MleError::SingularDesignMatrix`）に統一。過去に `bfgs` だけ検出漏れした
  バグクラスを構造的に排除。テストは Python `test_perfect_multicollinearity_raises_
  computation_error`（`method`×3 parametrize、CSVフィクスチャ）＋ engine
  `fit_returns_singular_design_matrix_error_for_perfectly_collinear_design_matrix`
  （`method`×`cov_type` ループ）へ集約。設計背景・数式は
  `docs/spec/logit-spec.md` 3.2・`docs/planning/specs/nonlinear-implementation-notes.md`
  「Issue #279」段落が正本。`refactoring-candidates-2.md` 項目82 も同時解消。
- **フォローアップ（未完・別Issue）**:
  - **#304**: warm start 導入で `method="bfgs"` 大標本が ~33% 悪化（既定 newton は逆に高速化）。
    quasi-Newton × warm start の相互作用の是正。あわせて下記の項目番号欠落を整理する。
  - **stale 参照（要調査）**: Issue #279 本文が参照する `refactoring-candidates-2.md` 項目54・
    `refactoring-candidates-3.md`、`docs/performance/{logit,probit}.md` が参照する項目44/45/46
    （マルチスレッド不安定・Probit Hessian 飽和・BFGS/L-BFGS 遅い）が、現行
    `refactoring-candidates*.md`（項目43 まで）に存在しない。番号ずれ or 過去の整理での欠落。

### 36. `test_separation_suspected_raises_computation_error_for_near_separation_data`のDGPがインライン生成で、Logit/Probitの`separation_suspected_dataset`共有ヘルパーを使っていない

- **対象**: [tests/test_tobit.py:460-471](../../../tests/test_tobit.py#L460-L471)
  （`random.Random(42)`でその場にDGPを生成）と対比した
  [tests/_helpers.py](../../../tests/_helpers.py)の`separation_
  suspected_dataset`（Logit/Probit共有のヘルパー関数、
  [tests/nonlinear/test_logit.py:14,396](../../../tests/nonlinear/test_logit.py#L14)・
  [tests/nonlinear/test_probit.py:14,368](../../../tests/nonlinear/test_probit.py#L14)
  で使用）
- **内容**: ユーザー指摘（2026-08-31、「先にbenchmarkでテストデータを
  作っておくのがいいと思うがどうか」）を受けて確認した。Tobitの
  分離疑いテストは`x1`の係数を100という極端な値にした準完全分離データを
  `random.Random(42)`（Python標準ライブラリの`random`モジュール、他の
  DGPで一般的な`numpy`ベースの乱数生成とも異なる）でその場で生成して
  いる。Logit/Probitは同種の目的のデータを`_helpers.py`の共有関数
  `separation_suspected_dataset`から取得している。ただしTobit版は
  `y_star`を`max(0.0, y_star)`で打ち切る処理が追加で必要なため、
  Logit/Probit版とDGP自体が完全に同一というわけではない。
- **Claudeの所感**: `testing-policy.md`「テスト用データセット」が
  求める`benchmark/`側でのDGP定義＋CSV固定は、この種の
  `ComputationError`パス専用データ（数値比較をしないデータ）には
  必須ではない（`engine`側の`fit_returns_singular_design_matrix_error_
  for_perfectly_collinear_design_matrix`等の小さい手書きデータも
  同様に固定CSV化されていない）ため、`benchmark/`フル対応は過剰だと考える。ただし`_helpers.py`に
  Tobit版の「打ち切り付き分離疑いデータセット」ヘルパーとして切り出し、
  `random`ではなく他のDGPと統一感のある`numpy`ベースの乱数生成に
  揃えるのは、再利用性・一貫性の両面で価値があると考える（実施コストは
  低い）。
- **気づいた経緯**: 2026-08-31、`tests/test_tobit.py`解説時のユーザー
  指摘。
- **状態**: 未対応（`_helpers.py`への切り出しを推奨、`benchmark/`側の
  フル対応は不要と判断。着手要否はユーザー判断待ち）

### 37. `test_cov_type_is_case_insensitive`と`test_nonrobust_is_alias_for_classical`の統合可能性（IV版項目4と同型の論点）

- **対象**: [tests/test_tobit.py:526-564](../../../tests/test_tobit.py#L526-L564)
- **内容**: ユーザー指摘（2026-08-31、「`test_cov_type_is_case_
  insensitive`と`test_nonrobust_is_alias_for_classical`は組み合わせの
  問題だと思うが、同じテストにまとめられそう（`NonRobust`〔大文字
  小文字が混じる〕を他の`cov_type`でもやったほうがいいのでは？）」）。
  本ファイル項目4（IVの同名テストに関する同種の指摘）と全く同じ論点
  がTobitにも当てはまる。
- **Claudeの所感**: 項目4と同じ所感——`test_cov_type_is_case_
  insensitive`は「ラベルの変換」、`test_nonrobust_is_alias_for_
  classical`は「計算結果（標準誤差）の一致」を見ており検証対象が
  異なるため完全な統合は難しいが、「大文字小文字が混じった表記
  （`NonRobust`のような）を`nonrobust`以外の`cov_type`（`HC0`の
  代わりに`Hc0`のような）でも一貫してテストする」という観点の拡充は
  価値があると考える。
- **気づいた経緯**: 2026-08-31、`tests/test_tobit.py`解説時のユーザー
  指摘。
- **状態**: 未対応（本ファイル項目4と合わせて検討、着手要否はユーザー
  判断待ち）

### 38.【Issue化】`marginal_effects()`/`predict()`に4つ目のtarget候補`E[y|y>0,x]`（打ち切られていないサブサンプルへの条件付き期待値）を追加すべきか

→ Issue #311として切り出し済み（2026-09-11）。詳細はIssueを参照。

### 39. `fit_iv`のdoc commentがGMM実装状況について古い記述のまま

- **対象**: [engine_pybind/src/lib.rs:156-157](../../../engine_pybind/src/lib.rs#L156-L157)
- **内容**: `fit_iv`関数のdoc commentに「`options.method`は`"2sls"`
  （現状唯一の実装）か`"gmm"`（未実装、`ValidationError`を送出）を
  選ぶ」と書かれているが、GMMは`tests/iv/test_iv_gmm_fixtures.py`で
  `linearmodels`とのクロスチェックが通っている通り既に実装済み。
  PyO3の`///`docコメントはそのままPython側の`__doc__`（`help()`や
  IDE補完）に反映されるため、ユーザーに見える形で古い情報が残っている。
- **Claudeの所感**: 実装が先行しdoc commentの更新が漏れた典型的な
  ケース。実害は小さいが、ユーザー向けAPIドキュメントの正確性の
  問題として修正対象になりうる。
- **気づいた経緯**: 2026-08-31、`engine_pybind/src/lib.rs`解説時に
  コード全体を確認して発見。
- **状態**: 未対応

### 40.【Issue化】結果クラスの命名が頭字語の大文字小文字で不統一（`OlsResults`/`WlsResults`/`IvResults` vs `OLS`/`WLS`/`IV`）

→ Issue #310として切り出し済み（2026-09-11）。詳細はIssueを参照。

### 41. `__version__`がバージョン文字列の3つ目の手書きソースになっている

- **対象**: [python_package/econometricsmodels/__init__.py:40](../../../python_package/econometricsmodels/__init__.py#L40)
- **内容**: `__version__ = "0.5.0"`は`pyproject.toml`の`version`・
  `Cargo.toml`の`version`と同じ値を独立に手書きしている（確認時点で
  3箇所とも`"0.5.0"`で一致）。`maturin`は`pyproject.toml`の`version`
  を使ってwheelバージョンを決めるため、この`__init__.py`の値は
  自動的には追従しない。バージョンアップ時にどれか1箇所を更新し
  忘れるとズレるリスクがある。
- **Claudeの所感**: 実害はまだ顕在化していない（現状3箇所とも一致）
  が、`X.Y.Z`をSemVerで手動管理する運用（CLAUDE.md 8章）である以上、
  更新手順書・チェックリスト化、または`importlib.metadata.version()`
  等で`pyproject.toml`の値を実行時に参照する方式への変更を検討する
  余地がある。
- **気づいた経緯**: 2026-08-31、`python_package/econometricsmodels/
  __init__.py`解説時に`pyproject.toml`・`Cargo.toml`と突き合わせて
  確認。
- **状態**: 未対応

### 42. `param_names`と他配列を`zip`して`dict`化するパターンが全6手法・多数箇所で重複

- **対象**: [python_package/econometricsmodels/linear/ols.py:104-131](../../../python_package/econometricsmodels/linear/ols.py#L104-L131)
  （`params`/`std_errors`/`t_stats`/`p_values`/`conf_int`の各
  プロパティ、および`coef_table()`）。同型のコードが
  `linear/wls.py`・`nonlinear/logit.py`・`nonlinear/probit.py`・
  `nonlinear/tobit.py`・`iv/iv.py`にも存在する
  （`dict(zip(self._raw.param_names, self._raw.xxx))`という形の
  記述だけで`grep`上24箇所、`coef_table()`相当の複数配列`zip`も
  含めるとさらに多い）。
- **内容**: `_lib`側の結果オブジェクトが係数名配列（`param_names`）
  と各統計量配列（`params`/`std_errors`等）を別々に持つ設計
  （PyO3の`#[pyclass(get_all)]`でRust構造体フィールドをそのまま
  公開する都合）のため、Python側の全ラッパークラスが同じ
  「`param_names`と対応する配列を`zip`して`dict`化する」処理を
  個別に書いている。共通ヘルパー関数（例:
  `_zip_to_dict(names, values)`）を`python_package/econometricsmodels`
  直下の共通モジュールに切り出せば重複を削減できる。
- **Claudeの所感**: 実装済みの6手法全てに渡って全く同一のパターンが
  重複しているため、リファクタリングの費用対効果は高いと考える。
  加えて、この実装パターン自体が、以前記録したIVの「`const`名衝突に
  よる`dict`キー上書きバグ」が発生する箇所そのものでもある
  （ヘルパーに一元化しておけば、将来的な修正が1箇所で全手法に
  波及するという副次的なメリットもある）。
- **気づいた経緯**: 2026-08-31、`python_package/econometricsmodels/
  linear/ols.py`解説時に、他の結果クラスと`grep`で突き合わせて確認。
- **状態**: 未対応

### 43. `ols.py`等のモジュールdocstringと`linear/CLAUDE.md`「実装パターン」節が同じ設計方針を2箇所に重複記述

- **対象**: [python_package/econometricsmodels/linear/ols.py:1-8](../../../python_package/econometricsmodels/linear/ols.py#L1-L8)
  と[python_package/econometricsmodels/linear/CLAUDE.md](../../../python_package/econometricsmodels/linear/CLAUDE.md)
  「実装パターン」節（`OLS`/`WLS`クラスはコンストラクタでは検証せず
  `fit()`呼び出し時に初めて`_lib.fit_ols`/`_lib.fit_wls`を呼ぶ、という
  記述）
- **内容**: `ols.py`のモジュールdocstringは「Validation and estimation
  logic live entirely on the Rust side; this module only provides the
  Python-facing API shape」と説明しており、`linear/CLAUDE.md`の
  「実装パターン」節も同じ設計方針（コンストラクタでは検証しない、
  `fit()`時に初めてRust側を呼ぶ）を別の言葉で改めて説明している。
  同じ設計方針が2箇所（docstring・ネストCLAUDE.md）に独立して
  記述されており、どちらかを更新した際にもう片方が古いままになる
  リスクがある。なお、同型のdocstring文言（"A thin wrapper
  around...Validation and estimation logic live entirely on the
  Rust side..."）自体は`wls.py`/`logit.py`/`probit.py`/`tobit.py`/
  `iv.py`の6ファイル全てにほぼ同一の形で存在するが、`linear/CLAUDE.md`
  が明示的に同じ内容を繰り返しているのはOLS/WLSの2ファイルのみ
  （`nonlinear/CLAUDE.md`はTobit固有差分の説明に特化しており、
  この一般論は繰り返していない）。
- **Claudeの所感**: 実害の大きいバグではないが、CLAUDE.mdの目的が
  「実装ノウハウ・設計判断の理由の集約」である以上、docstringに
  既にある内容をネストCLAUDE.mdでも全文近く繰り返す必要は薄いと
  考える。ネストCLAUDE.md側は「なぜそうしたか」の背景・経緯に
  絞り、「何をしているか」はdocstring（またはコード自体）を参照する
  形にする方が重複を減らせる。
- **気づいた経緯**: 2026-08-31、`python_package/econometricsmodels/
  linear/ols.py`解説時のユーザー指摘。
- **状態**: 未対応

### 44. `tests/nonlinear/test_tobit.py`の一部テストが`_binary_choice_checks.py`（Logit/Probit共有）と重複しており、部分的な共通化の余地がある

- **対象**: [tests/nonlinear/test_tobit.py](../../../tests/nonlinear/test_tobit.py)
  と[tests/nonlinear/_binary_choice_checks.py](../../../tests/nonlinear/_binary_choice_checks.py)
- **内容**: ユーザー指摘（2026-09-11、Issue #307対応でTobitに`test_method_label`を
  新規で書き下ろした際、「`test_tobit.py`の内容は`_binary_choice_checks.py`と
  同様な気がする、共通化したほうが良いか検討してほしい」）を受けて全件突き合わせた。
  - **完全に同じ挙動でそのまま流用できるもの（4件）**: `test_fit_succeeds_and_returns_tobit_results`
    ⇔`check_fit_succeeds_and_returns_results`、`test_method_option_converges_to_same_params`
    ⇔`check_method_option_converges_to_same_params`、`test_n_obs_matches_dataset_size`
    ⇔`check_n_obs_matches_dataset_size`、今回追加した`test_method_label`
    ⇔`check_method_label`。いずれも`res.converged`/`res.param_names`/`res.n_obs`/
    `res.method`など、Tobit固有の`"sigma"`追加パラメータの有無に依存しないフィールドしか
    見ておらず、コード変更なしでそのまま呼べる（Issue #307で`test_method_label`を
    独自に書き下ろしたのは、この既存の重複ヘルパーに気づかず車輪の再発明をした形になる）。
  - **一見似ているが実質的に異なるもの（大半、10件以上）**: `test_params_std_errors_z_stats_p_values_share_keys`/
    `test_conf_int_structure`/`test_coef_table_structure`/
    `test_param_names_include_const_first_and_sigma_last`/
    `test_include_intercept_false_omits_const_and_converges`は、いずれも`"sigma"`が
    末尾に追加される分だけ期待値（キー集合・件数・`df_model`の値）が異なり、単純な
    パラメータ化では吸収しきれない（`python_package/econometricsmodels/nonlinear/CLAUDE.md`
    「Tobit固有の設計」節が明記する設計差）。`predict()`/`marginal_effects()`の
    `target`引数・`censoring_fit_check()`（`pred_table()`と無関係な形状）・打ち切り境界
    関連のバリデーションはLogit/Probitに対応物が無いTobit固有機能。
- **Claudeの所感**: `refactoring-candidates.md`項目11（IV 2SLS/GMMの`_check_result`
  統合可否）と同じ構図で、全面統合は不適切（Tobit側の固有機能・フィールドが多く、
  無理に共通化すると可読性が落ちる）だが、上記4件の完全重複だけを共通化する価値は
  あると考える。対応する場合は`test_tobit.py`の該当4関数を削除し`_binary_choice_checks`
  をimportして呼ぶ薄いラッパーに置き換えることになるが、その場合`_binary_choice_checks.py`
  のモジュールdocstring（現在「Logit/Probitの重複を集約」とだけ記載）もTobitを含む
  形に更新する必要がある。
- **気づいた経緯**: 2026-09-11、Issue #307（Logit/Probit/Tobit/IVの結果オブジェクトに
  `method`フィールドを追加）対応中のユーザー指摘。
- **状態**: 未対応（ユーザー判断によりIssue化はせず本メモへの記録のみ。engineの修正を
  伴わないテストのみの変更のため）

### 45. `python_package/econometricsmodels/nonlinear/logit.py`と`probit.py`が実質1文字しか違わない（本体コード340行の重複）

- **対象**: [python_package/econometricsmodels/nonlinear/logit.py](../../../python_package/econometricsmodels/nonlinear/logit.py)
  と[python_package/econometricsmodels/nonlinear/probit.py](../../../python_package/econometricsmodels/nonlinear/probit.py)
- **内容**: `diff`を取ったところ、全340行中、`Logit`→`Probit`という名前の
  機械的な置換（クラス名・型名・docstring中の言及）を除いて実質的な差分は
  `predict()`のdocstring1箇所のみ（`p_i = Λ(x_i'β̂)` vs `p_i = Φ(x_i'β̂)`、
  リンク関数がロジスティック分布CDFか標準正規分布CDFかの違いを示すのみで、
  コードとしての差分ではない）だった。`nonlinear/CLAUDE.md`にも「`probit.py`
  は`logit.py`と完全に同型のパターン」と明記されている、既知・意図的な
  重複ではあるが、テスト側（`test_logit_*.py`/`test_probit_*.py`の共通化、
  項目95で対応済み）とは異なり、**この本体コード（`logit.py`/`probit.py`）
  自体の重複は未検討**だった。
- **Claudeの所感**: `_lib.fit_logit`/`_lib.fit_probit`という呼び出し先関数名、
  `LogitOptions`/`ProbitOptions`という型、`Λ`/`Φ`というdocstring中の記号の
  3点だけが実際の相違点であり、共有基底クラス（例:
  `_BinaryChoiceEstimator`/`_BinaryChoiceResults`）にこれらをパラメータ化
  して切り出せば、340行×2ファイルを実質1ファイル+数十行の差分に圧縮できる
  可能性がある。ただし、テスト側の項目11・44で「無理な統合は避けるべき」
  という判断が繰り返されている通り、コードの完全一致度が高くても
  **将来Tobitのように固有機能が増えた際に基底クラスの抽象化が破綻する
  リスク**（Tobit方式の初期値統一を`nonlinear-api-design.md`が既に検討して
  いる、項目35参照）もあるため、着手前にユーザー判断が必要と考える。
- **気づいた経緯**: 2026-08-31、`nonlinear/probit.py`解説時に`logit.py`と
  `diff`で突き合わせて確認。
- **状態**: 未対応（着手要否はユーザー判断待ち）
