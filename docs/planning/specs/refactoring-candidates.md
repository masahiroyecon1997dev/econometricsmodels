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

**完了項目の扱い（2026-08-22運用ルール、2026-08-30更新）**: 「対応済み」「対応不要と
判断」「調査の上やらないと判断」のいずれになった項目もこのファイルから削除する
（コード自体とgit logが原本）。削除した記録は`refactoring-issue231-progress.md`の
「随時対応ログ」に残す——「対応済み」は要点（対応内容・コミットハッシュ）のみ、
却下・見送り判断は同じ提案の調査をやり直さずに済むよう根拠も含めて残す。番号は
削除後も詰め直さない（欠番があっても他項目からの「項目N」表記自体は維持できる）。
ただし**番号を維持しても参照先の内容は消える**ため、削除対象の項目を他の項目が
「項目N」で参照している場合は、削除前にその参照側へ必要な文脈（何の話か・結論）を
埋め込み、削除後も参照側だけで自己完結するようにする。

---

## 一覧

※ 項目12・13・15・16・18〜21・24〜29・31〜35・39 は、`benchmark/` の構造変更（Initiative A、`benchmark-restructure-design.md`）が上位計画として吸収したため削除した（吸収項目の一覧は同ノート10章、対応の進捗は`refactoring-issue231-progress.md`）。番号は詰め直さない（記録フォーマット節参照）。

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
