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

- **対象**: [performance/compare_ols.py](../../../performance/compare_ols.py)・
  [performance/_perf_harness.py](../../../performance/_perf_harness.py)
  （旧`performance/compare_performance.py`。当時はハーネスとOLSアダプタが1ファイルに
  同居していた。`_fit_once_engine`/`_fit_once_statsmodels`/`_fit_once_pyfixest`、
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
- **状態**: 一部対応済み。#250（`de0b4a7`）で手法非依存の骨格を
  `_perf_harness.py`（`PerfAdapter`/`FitContext`・`_worker`/`_run_isolated`・
  スイープ・`build_report`）へ切り出し、`_meta`にリファレンス実装バージョンも
  記録するようにした。pyfixest比較は廃止。残りは #251〜254 で各手法アダプタ
  （`compare_wls.py`等）を追加しながら随時。**全手法（WLS/Logit/Probit/IV）の
  組み込み完了後にこの項目を削除する**（ユーザー指示、2026-08-30）。

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

### 44. engine（faer/rayon）のマルチスレッド線形代数が、多コア機・負荷下でシングルスレッド時の20倍以上遅くなり不安定になる

- **対象**: `engine/src/linear/ols.rs`の`OlsEstimator::fit`（faer経由の
  QR分解・Gram行列構築）。他の線形代数を伴う手法（WLS/IV/GMM等）も同傾向の
  可能性。
- **内容**: 2026-08-30、Issue #250/#98のOLS性能比較の再計測中に発覚。
  12論理コアのdevcontainer（WSL2）上で`OLS(...).fit()`（classical, n=1,000,000,
  k=5）を計測すると、スレッド数無制限では実行時間が3〜4秒かつ試行ごとに
  0.6〜4.5秒と大きくばらつく。同じ条件で`RAYON_NUM_THREADS=1`等を設定して
  シングルスレッドに固定すると**0.14〜0.18秒で安定**し、statsmodels
  （0.33〜0.49秒）より2〜3倍速いという期待どおりの結果になる。
  faerは0.24.4のまま・`ols.rs`は2026-08-09以降変更なしのため、コード
  リグレッションではなく、スレッドプールが負荷下で競合する挙動の問題。
  比較対象のstatsmodels（numpy/OpenBLAS）は同条件でも安定して劣化しなかった。
- **ユーザーの懸念（2026-08-30）**: ベンチマークは`OLS(...).fit()`をそのまま
  呼んでおり、これは実利用と同じ経路。多コア機のユーザーが大規模データで
  `fit()`すると同じ現象が起きうる（ベンチマークで先に見つかったのは幸い）。
- **想定される調査の方向**: OLSの設計行列はtall-skinny（n大・k小）で、skinny
  行列のQR/Gram構築を多スレッドに分割してもスレッドプールのオーバーヘッドと
  メモリ帯域競合が支配的になりやすい。問題サイズに応じてシングルスレッドに
  留める閾値、あるいは明示的なスレッド数上限の導入を検討する。WSL2固有の
  スケジューラ挙動か、ネイティブの多コアLinuxでも再現するかの切り分けも要る。
  `.claude/rules/rust-style.md`「パフォーマンス」節のrayon採用は「実測してから
  決める」方針であり、本件はその実測データ点。
- **状態**: 未対応（engineのロジック挙動に関わるためリファクタリングの範囲外。
  別途Issue化して調査する。性能比較ハーネス側は`_SINGLE_THREAD_ENV`で
  スレッド数を1に固定し、この現象を計測から切り離す対応を実施済み＝
  `de0b4a7`の後続コミット）

### 45. engineのProbitが、statsmodelsが収束できる大標本条件でHessian特異エラーを出す

- **対象**: `engine/src/nonlinear/`のProbitのHessian（観測情報行列）構築・
  Newtonソルバ（`nonlinear/common.rs`の`run_solver`共有部分を含む）。
- **内容**: 2026-08-30、Probitの性能比較（#253）実装中に発覚。
  `generate_binary_choice_dataset("baseline", link="probit", n=1_000_000,
  k=5, seed=42)`を`Probit(...).fit()`すると
  `ComputationError: the Hessian is singular and cannot be inverted`。
  **同じデータで statsmodels の `smf.probit(...).fit()` は収束し Hessian も
  反転できる（1.87秒、engineは即エラー）**。n=100,000では engine も通り、
  k=3なら n=1,000,000 でも engine が通る。Φはロジスティック分布のΛより裾が
  薄く、n・kが大きいとXβの分散増大でΦ(Xβ)が0/1に飽和する観測が増え、
  Probitの観測情報行列の重み`φ(xβ)²/[Φ(xβ)(1−Φ(xβ))]`がアンダーフローして
  Hessianが数値的に特異化するのが原因と推測。Logit（Λ）では
  n=1,000,000, k=5 でも問題は起きない。
- **ユーザー指摘（2026-08-30）**: statsmodelsが同条件を捌けている以上、
  engine側に不具合の可能性がある。飽和に強い実装（重みのクリッピング・
  対数空間でのΦ(1−Φ)計算・Newtonのdamping/line search・step halving等、
  statsmodelsが持つ頑健化）の余地がないか調査する価値がある。
- **状態**: 未対応（engineのロジック挙動に関わるためリファクタリングの範囲外。
  別途Issue化して調査する。性能比較側はProbitのn軸上限を100,000に制限して
  回避＝`compare_probit.py`の`PROBIT_ADAPTER.n_sweep`、#253のコミット）
