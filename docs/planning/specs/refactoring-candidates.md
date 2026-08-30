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

※ 項目12・13・15・16・18〜21・24〜29・31〜35・39 は、`benchmark/` の構造変更（Initiative A、`benchmark-restructure-design.md`）が上位計画として吸収したため削除した（吸収項目の一覧は同ノート10章、対応の進捗は`refactoring-issue231-progress.md`）。番号は詰め直さない（記録フォーマット節参照）。

### 17. コメント中のIssue番号（`Issue #231`等）参照が10ファイルに散在

- **対象**: `grep -rn "Issue #[0-9]\+" benchmark/`で確認した以下10ファイル（計30箇所超）
  - [benchmark/common/reference/r.py:3](../../../benchmark/common/reference/r.py#L3)
  - [benchmark/iv/references/linearmodels_ref.py:156,292](../../../benchmark/iv/references/linearmodels_ref.py#L156)
  - [benchmark/iv/freeze.py:55,60](../../../benchmark/iv/freeze.py#L55)
  - [benchmark/iv/references/run_ivreg.R:25,54,148,175](../../../benchmark/iv/references/run_ivreg.R#L25)
  - [benchmark/iv/fixtures/generate_iv_fixtures.py:108,125,141,181,188,190](../../../benchmark/iv/fixtures/generate_iv_fixtures.py#L108)
  - [benchmark/iv/fixtures/generate_iv_gmm_fixtures.py:79,136,151,164,202](../../../benchmark/iv/fixtures/generate_iv_gmm_fixtures.py#L79)
  - [benchmark/iv/fixtures/generate_iv_crosscheck_fixtures.py:21,25,67,190,211,271,343,350,351,358,363](../../../benchmark/iv/fixtures/generate_iv_crosscheck_fixtures.py#L21)
  - [benchmark/nonlinear/fixtures/generate_logit_fixtures.py:58,160](../../../benchmark/nonlinear/fixtures/generate_logit_fixtures.py#L58)
  - [benchmark/nonlinear/fixtures/generate_probit_fixtures.py:63,170](../../../benchmark/nonlinear/fixtures/generate_probit_fixtures.py#L63)
  - [benchmark/nonlinear/references/run_glm_crosscheck.R:115](../../../benchmark/nonlinear/references/run_glm_crosscheck.R#L115)
- **内容**: ユーザー指摘（2026-08-15）。`benchmark/iv/datasets.py`解説時に「Issue番号は冗長なので削除したい」との指摘を受け該当1箇所を修正済み（本文はWHYを保持したままIssue番号のみ除去）だったが、同種の参照が上記の通り広範囲に残っている。Issue番号だけを見ても文脈が分からず、GitHub側の該当Issueが将来クローズ・番号体系変更等で参照として陳腐化するリスクがある一方、各コメント自体はWHY（なぜその実装・設計になっているか）を本文中に十分書けているため、Issue番号を削っても情報量は落ちない。
- **Claudeの所感**: `benchmark/iv/datasets.py`で行ったのと同じ要領（Issue番号を削り、WHYの実質的な記述は残す）で機械的に対応できる規模だが、30箇所超と件数が多いため一括対応が妥当かは要検討（`refactor`スキルの対象として一括実施する候補）。
- **気づいた経緯**: 2026-08-15、`benchmark/iv/references/linearmodels_ref.py`解説着手前のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 22. Rスクリプト冒頭の引数パースパターンが3ファイルで重複（`_common.R`は後処理側のみ共通化済み）

- **対象**: [benchmark/linear/references/run_lm_crosscheck.R:22-32](../../../benchmark/linear/references/run_lm_crosscheck.R#L22-L32)・
  [benchmark/linear/references/run_lm_predict_crosscheck.R:13-22](../../../benchmark/linear/references/run_lm_predict_crosscheck.R#L13-L22)・
  [benchmark/iv/references/run_ivreg.R:56-65](../../../benchmark/iv/references/run_ivreg.R#L56-L65)・
  [benchmark/nonlinear/references/run_glm_crosscheck.R:48-62](../../../benchmark/nonlinear/references/run_glm_crosscheck.R#L48-L62)
- **内容**: `commandArgs(trailingOnly = TRUE)`→引数不足チェック（`stop(...)`）→
  `data_path <- args[1]`→`formula_str <- args[2]`→
  `read.csv(data_path, check.names = FALSE)`という冒頭5〜6行のパターンが4ファイルで
  同型。`benchmark/common/_common.R`は`extract_coef_se`/`wald_f_test`という**後処理側**の
  重複は既に解消済みだが、この**冒頭の引数パース**側は対象になっておらず残っている。
- **Claudeの所感**: `_common.R`に`parse_common_args(args, min_required=2)`のような
  関数を追加すれば解消できそうだが、Rには構造化された戻り値（複数の変数をまとめて
  返す）の慣用的な書き方がPython程スッキリしない（リストで返して`$`で分解する形に
  なる）ため、効果とのバランスは要検討。
- **気づいた経緯**: 2026-08-16、`benchmark/linear/references/run_lm_predict_crosscheck.R`解説中に発見。
  `benchmark/nonlinear/references/run_glm_crosscheck.R`にも同型（`link`引数の追加分岐はあるが冒頭部分は
  同じ）であることを確認済み。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 23. `benchmark/linear/references/run_lm_predict_crosscheck.R`を手法非依存の汎用スクリプトにできないか（設計判断候補）

- **対象**: [benchmark/linear/references/run_lm_predict_crosscheck.R](../../../benchmark/linear/references/run_lm_predict_crosscheck.R)
- **内容**: ユーザー提案（2026-08-16）。今後WLS（Issue #132）・Logit（#131）・Tobit（#222）
  にも`predict()`のRクロスチェックが必要になる見込みだが、現状の`benchmark/linear/references/run_lm_predict_crosscheck.R`
  は名前・置き場所（`benchmark/linear/references/`）ともにOLS専用に見える。`benchmark/linear/references/run_lm_crosscheck.R`
  （cov_type/cluster_col/hac_lag/weight_colという複数の位置引数）と単純結合すると引数パースが
  複雑になる一方、`fitted()`/`predict(model, newdata=...)`というR関数自体は`lm`オブジェクトにも
  `glm`オブジェクト（Logit/Probit用）にも共通して使える。
- **Claudeの所感**: 「`benchmark/linear/references/run_lm_crosscheck.R`に統合する」よりも、
  `benchmark/linear/references/run_lm_predict_crosscheck.R`自体を手法非依存の汎用スクリプト（`benchmark/`直下等に移動し、
  `weights`引数を追加すればWLSにもそのまま使える）にする方向の方が筋が良さそうだと考える。
  ただしTobit（打ち切りモデル）は`predict()`の意味自体が変わりうる（打ち切り前の潜在変数か、
  打ち切り後の観測値か）ため、そこだけ別途確認が必要。今すぐ決める話ではなく、Issue #131/
  #132/#222着手時の設計判断になる。
- **気づいた経緯**: 2026-08-16、`benchmark/linear/references/run_lm_predict_crosscheck.R`解説後のユーザー提案。
- **状態**: 未対応（Issue #131/#132/#222着手時に判断）

### 30. `benchmark/nonlinear/references/run_glm_crosscheck.R`内で列スケーリングによる反転ロジックが2回重複

- **対象**: [benchmark/nonlinear/references/run_glm_crosscheck.R:91-101](../../../benchmark/nonlinear/references/run_glm_crosscheck.R#L91-L101)
  （`observed_bread`）・[benchmark/nonlinear/references/run_glm_crosscheck.R:134-138](../../../benchmark/nonlinear/references/run_glm_crosscheck.R#L134-L138)
  （`opg`分岐）
- **内容**: `scale_variance`シナリオでの見かけ上の特異性を避けるため、「列を各々のノルムで
  正規化→反転→`Σ=D⁻¹(D⁻¹MD⁻¹)⁻¹D⁻¹`の恒等式でスケールを戻す」という同じテクニックが
  同一ファイル内で2回（`observed_bread`関数内、および`opg`分岐のインラインコード）
  ほぼ同じ形で書かれている。
- **Claudeの所感**: `scale_and_invert(M, X または scores) -> 行列`のような小さな
  ヘルパー関数に切り出せそうだが、他ファイルとの重複ではなく同一ファイル内の
  重複のため優先度は低め。
- **気づいた経緯**: 2026-08-16、`benchmark/nonlinear/references/run_glm_crosscheck.R`解説中に発見。
- **状態**: 未対応（着手要否はユーザー判断待ち、優先度低）

### 37. `suppressMessages`が`benchmark/iv/references/run_ivreg.R`にしか無く、他3ファイルにJSON破損リスクが残る

- **対象**: [benchmark/iv/references/run_ivreg.R:67-72](../../../benchmark/iv/references/run_ivreg.R#L67-L72)
  （`suppressMessages({library(...)...})`）と、`benchmark/linear/references/run_lm_crosscheck.R`・
  `benchmark/linear/references/run_lm_predict_crosscheck.R`・`benchmark/nonlinear/references/run_glm_crosscheck.R`（いずれも素の
  `library(...)`のまま）
- **内容**: ユーザー指摘（2026-08-16）。`library()`実行時にRのバージョンや
  パッケージの警告等でメッセージが標準出力に出力されると、`toJSON`の出力に
  混ざってJSONパースが壊れる可能性がある。`benchmark/iv/references/run_ivreg.R`のみこれを
  `suppressMessages({...})`で防いでいるが、他3ファイルには同じ対策が無く、
  単なるスタイルの不統一ではなく**潜在的な頑健性のギャップ**。
- **Claudeの所感**: 他3ファイルにも`suppressMessages({...})`を追加するのが
  低リスクな対策。
- **気づいた経緯**: 2026-08-16、`benchmark/iv/references/run_ivreg.R`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 38. `script_dir`特定→`_common.R`の`source()`ブロックは構造的に共通化しにくい（対応不要と判断）

- **対応不要と判断**（2026-08-16）: `benchmark/linear/references/run_lm_crosscheck.R`・`benchmark/iv/references/run_ivreg.R`
  にある「自分の場所を特定して`_common.R`を`source()`する」3行を`_common.R`側の
  ヘルパーに切り出そうとすると、そのヘルパーを呼ぶために先に`_common.R`をsourceする
  必要がある循環（鶏と卵）が生じ、構造的に共通化できない（Rに`__file__`相当が無いため）。
  再提案しても同じ結論になる見込み。

### 40. `compare_performance.py`が現状OLS専用に書かれており、他手法へ性能比較を拡張する際に共通化できる可能性が高い

- **対象**: [performance/compare_performance.py](../../../performance/compare_performance.py)
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

- **対象**: [performance/compare_performance.py:274](../../../performance/compare_performance.py#L274)
  （`f"peak_rss={row['peak_rss_kb'] / 1024:.1f}MB"`、標準エラーの進捗ログ用）・
  [performance/render_performance_summary.py:27](../../../performance/render_performance_summary.py#L27)
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
