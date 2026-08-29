# Issue #231 リファクタリング進捗管理

[#231](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/231)
（リファクタリング用スキルの作成とOLS/WLS/Probit/Logit/IVのリファクタリング）の
進捗・メモを記録するドキュメント。範囲が広いため7フェーズに分割し、1フェーズずつ
完了させてから次に進む。各フェーズの対象範囲・変更方針はユーザーが都度指示し、
このドキュメントはその結果・未解決事項を記録する。

**状態（2026-08-16更新）**: フェーズ1〜4完了時点で#231はクローズ済み。フェーズ5〜7
（`python_package`/`engine_pybind`/`engine`の実リファクタリング）は、具体的な変更
方針が未確定のまま残っていたことと、`refactoring-candidates.md`・
`test-coverage-candidates.md`という#231専用ではない汎用の候補メモの仕組みが
並行してできたことから、後継Issue
[#248](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/248)
に引き継いだ。本ドキュメントのフェーズ1〜4の記録は完了した実施内容の履歴として
そのまま残す（フェーズ5〜7以降の進捗は#248側で新たに記録する）。

## 全体方針

- フェーズは**順番に**進める（フェーズNの完了条件を満たしてから N+1 に着手）。
- 各フェーズ内で設計書（`docs/spec/`・`docs/planning/specs/`・各系統の
  ネストCLAUDE.md等）の修正が必要になった場合はそのフェーズ内で対応する。
- 疑問点・判断が分かれる点はCLAUDE.md 14章の方針通り、独自判断で埋めず
  都度ユーザーに確認する。確認事項は都度このドキュメントの該当フェーズに追記する。
- 各フェーズ完了時に `cargo test` / `pytest` がパスすることを確認してから
  次フェーズへ進む。

## フェーズ一覧

| # | フェーズ | 状態 |
|---|---|---|
| 1 | リファクタリング用スキル（or 既存スキル/エージェントの代用整理） | 完了 |
| 2 | `benchmark/`ディレクトリの整理とリファクタリング | 完了 |
| 3 | `tests/`ディレクトリの整理とリファクタリング | 完了 |
| 3.5 | crosscheckテストの許容誤差計算式バグ修正（`refactor`スキル範囲外） | 完了 |
| 4 | ロジック整理前のテスト拡充（OLS/WLS/Logit/Probitレビュー＋IV #232〜238） | 完了（linear系統〔OLS/WLS〕・nonlinear系統〔Logit/Probit〕・IV系統〔2SLS/GMM〕全て完了） |
| 5 | `python_package/`のリファクタリング | [#248](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/248)に引き継ぎ |
| 6 | `engine_pybind/`のリファクタリング | [#248](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/248)に引き継ぎ |
| 7 | `engine/`のリファクタリング | [#248](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/248)に引き継ぎ |

---

## `refactoring-candidates.md`駆動の随時対応ログ（フェーズ外、2026-08-22〜）

上記フェーズはIssue #231としてスコープ確定済みの計画だが、それとは別に
`docs/planning/specs/refactoring-candidates.md`に溜まった個別項目を`refactor`
スキルでその都度つまみ食い的に対応する運用も並行して行っている。この節は
その運用ルール・進捗をセッションをまたいで引き継ぐためのメモ。`refactoring-candidates.md`
に項目がまだ残っている間は同ファイルの各項目「状態」欄が一次情報だが、2026-08-22の
運用ルール（同ファイル冒頭「完了項目の扱い」参照）により「対応済み」項目は本体から
削除されるため、削除後はこちらのスナップショットが当該項目の唯一の記録になる。

**運用ルール（2026-08-22のセッションでユーザーと確立）**:
- `refactoring-candidates.md`の項目を**1件（または明示的に指定された小さいまとまり）ずつ**対応する。
  まとめて全部やらない。
- 実装 → 検証（構文・lintに加え、挙動を変えないリファクタリングでは可能な限り
  「リファクタリング前後で出力が完全一致すること」を実測確認する） → `/code-review`
  → `refactoring-candidates.md`の該当項目「状態」を更新、の順で1件を完了させる。
- 1件完了したら**そこで一旦止まり、ユーザーにコミット前確認を取ってから**コミットする
  （まとめて複数件コミットしない）。
- 設計判断が分かれる点（例: 定数の置き場所、sys.path問題の解決方式）は着手前に
  ユーザーに確認する（CLAUDE.md 14章）。

**進捗スナップショット（2026-08-22時点、ブランチ`release/v0.6.0`）**:
- 項目3（`sys.path.insert`とIDE静的解析）: 対応済み・コミット済み（`48727c9`）。
  PYTHONPATH環境変数方式を採用（`.devcontainer/devcontainer.json`の`remoteEnv`に
  `PYTHONPATH`を追加、`benchmark/`配下22ファイルから`sys.path.insert`を削除）。
  devcontainer再ビルド後、`echo $PYTHONPATH`でbenchmark関連パスが設定されていること・
  `python benchmark/linear/generate_linear_datasets.py baseline`等がimportエラー
  無く動くことを確認済み（2026-08-22、別セッションで再ビルド後に確認。ただし
  再ビルド直後は`test`/`benchmark`グループが`.venv`に未同期だったため
  `uv sync --all-groups`が別途必要だった）。
- 項目4（`SCENARIOS`重複）: 対応済み・コミット済み（`d23d9b7`）。
- 項目9・10（DGP用マジックナンバー集約、`benchmark/_dgp_constants.py`新設）:
  対応済み・コミット済み（`7d51802`）。
- 項目36（crosscheck側のpl.read_csv再実装をload_frozen_datasetに差し替え）:
  対応済み・コミット済み（`8f6f98e`）。ドキュメント編集時に項目37の見出しを
  誤って削除する事故が発生し、コミット後の`/code-review`で検知・修正した
  （`70bef2a`、`refactor`スキルに「変更した層に関わらず`/code-review`必須」を
  明記する再発防止も合わせて実施）。
- 上記5項目は`refactoring-candidates.md`本体からは削除済み（2026-08-22運用ルール、
  同ファイル冒頭「完了項目の扱い」参照）。このスナップショットが唯一の記録。
- 項目47（`refactoring-candidates-2.md`、`tests/_helpers.py`の`wooldridge_loader`が
  `conftest.py`と同一パスへ`sys.path.insert`を重複実行していた）: 対応済み・
  コミット済み（`835da85`）。`wooldridge_loader`から`sys.path.insert`呼び出しを
  削除（未使用になった`import sys`も削除）。`refactoring-candidates-2.md`本体からも
  同ポリシーで削除済み。
  **注**: 同ファイルの項目50（別セッションが追記、`1c70f63`までの間に
  コミット済み）が「影響範囲」として`tests/_helpers.py:90`・「項目47で指摘済みの
  重複分」に言及しているが、本対応で該当コードが削除されたためこの1点の記述が
  古くなっている（2026-08-23時点でも未修正）。項目50自体の本題（CI環境
  `ci_python.yml`にPYTHONPATHが伝播しておらず`tests/`側の`sys.path.insert`が
  今も必要という指摘）には影響しない。項目50を書いたセッション側の判断で
  追って更新される想定のため、こちらからは修正しない。
- 項目42（`tests/_assertions.py`のdocstringが現状と食い違っていた、
  「crosscheck系は計算式・シグネチャが異なるため対象外」という説明が
  OLS/WLS/IV crosscheckには既に当てはまらなくなっていた）・項目48
  （同docstringの「フェーズ3.5で修正予定」という古い記述、`tests/_helpers.py`の
  「22箇所で重複」等の陳腐化した件数）: いずれも対応済み・コミット済み
  （`10ef78e`）。`grep`で実際の呼び出し元を
  確認した結果、`test_ols_crosscheck.py`/`test_wls_crosscheck.py`/
  `test_iv_crosscheck.py`は`functools.partial`で`_assertions.py`の
  `assert_close`/`assert_dict_close`を再利用している一方、
  `test_logit_crosscheck.py`/`test_probit_crosscheck.py`は`_assert_dict_close`が
  `rtol`引数を取らない点でシグネチャが異なるだけで独立実装のまま
  （計算式自体は`tol = max(rtol*|ref|, atol)`で既に同一）と判明。この事実に
  合わせて`tests/_assertions.py`のdocstringを書き換え、`tests/_helpers.py`の
  陳腐化した件数言及も削除した。`refactoring-candidates.md`から項目42、
  `refactoring-candidates-2.md`から項目48を削除済み。**副次的な気づき**:
  Logit/Probit crosscheckを`_assertions.py`の`partial`パターンへ統合する
  コード変更自体（計算式は既に一致しているため統合の障害は無い見込み）は
  未実施。着手する場合は新規候補として別途起票する。
  `pytest tests`956件全件パス、`ruff check`／`ruff format --check`パス確認済み。
- 項目1（`benchmark/load_wooldridge.py`の`SUGGESTED_DATASETS`が未使用）:
  対応済み・コミット済み（`c0b78bf`）。
  削除する方針で確定（ユーザー判断。呼び出し側の形〔手法カテゴリ→リスト〕が
  実際の利用ニーズ〔個別データセット名〕と合わず、Wooldridgeデータセット名は
  外部パッケージ側の固定識別子で一元化の恩恵が薄いという判断）。削除に伴い、
  「仕様書に明記されていること」を代替の記録場所とする方針とし、記載が
  無かった2箇所を追加で埋めた。
  - `docs/spec/ols-spec.md`「3.6 テスト」に実データセット（`wage1`/`gpa2`、
    classical/HC0-3、`wage1`は地域クラスターも検証）の記載を追加
    （従来は完全に未記載だった）。
  - `docs/planning/specs/iv-api-design.md`に新規「5.5 実データセット」節を追加
    （`card`、全cov_typeで`linearmodels`/`ivreg`双方をクロスチェック）。
    **副次的な発見**: `SUGGESTED_DATASETS`の`"iv": ["mroz", "card"]`は
    不正確だった（`mroz`は実際にはIVで使われておらず、コメント中の一例と
    しての言及のみ。実際にIVが使う実データは`card`のみ）。
  - `.claude/skills/reference-benchmark/SKILL.md`の`SUGGESTED_DATASETS`言及
    2箇所も、「選定理由・変数構成を`docs/spec/<手法名>-spec.md`に明記する」
    という新しい運用に合わせて更新。
  - `refactoring-candidates.md`から項目1を削除済み。
  `pytest tests`956件全件パス、`ruff check`／`ruff format --check`パス確認済み、
  `python benchmark/load_wooldridge.py wage1`で動作確認済み。
- 項目2（`generate_linear_datasets.py`の`k`下限チェックが4箇所で同型パターン重複）:
  対応済み・コミット済み（`769eee8`）。
  `_require_min_k(scenario, k, minimum)`ヘルパーを新設し、
  `moderate_multicollinearity`/`high_condition_number`（k>=2）・
  `perfect_multicollinearity`（k>=3）・`scale_variance`（k>=2）・
  `scale_variance_mild`（k>=2）の4箇所を置き換えた（ファイル内ローカルの
  重複のため`_common.py`等への切り出しはせず、同一ファイル内のプライベート
  関数とした）。エラーメッセージの文言は変更前と完全に一致することを実測
  確認済み。`freeze_linear_datasets.py`の出力（synthetic系10シナリオ）が
  変更前後で完全一致することも確認済み。`refactoring-candidates.md`から
  項目2を削除済み。`pytest tests`956件全件パス、`ruff check`／
  `ruff format --check`パス確認済み。
- 項目5（`unknown scenario`/`unknown link`検証の重複が3系統＋nonlinearの
  link検証で計4箇所）: 対応済み・コミット済み（`8c71613`）。
  `benchmark/_common.py`に`validate_choice(value,
  valid_choices, label)`ヘルパーを新設し、linear/nonlinear/ivの
  `generate_*_datasets.py`3ファイルの`scenario`検証、nonlinearの`link`検証を
  置き換えた（計4箇所）。エラーメッセージが変更前と完全に一致することを
  4パターンとも実測確認済み。`freeze_datasets.py`の出力（linear/nonlinear/iv
  全系統）が変更前後で完全一致することも確認済み。`refactoring-candidates.md`
  から項目5を削除済み。`pytest tests`956件全件パス、`ruff check`／
  `ruff format --check`パス確認済み。
- 項目6（線形予測子の組み立て方が`column_stack`版とスカラー切片版で
  不統一）: 対応済み・コミット済み（`871f59b`）。
  `benchmark/_common.py`に`linear_predictor(X, beta) -> beta[0] + X @
  beta[1:]`ヘルパーを新設し、`generate_linear_datasets.py`（`y = beta[0] +
  X @ beta[1:] + errors`）・`generate_nonlinear_datasets.py`（`x_const =
  np.column_stack([np.ones(n), X]); p = _LINK_CDF[link](x_const @ beta)`）
  の両方を置き換えた。IVは`x_exog`/`x_endog`の2種の説明変数を持つ構造式
  （`beta0 + x_exog @ beta_exog + x_endog @ beta_endog + u`）で数学的に
  異なるため対象外（項目自体のスコープ通り）。`freeze_datasets.py`の出力
  （linear/nonlinear/iv全系統）が変更前後で完全一致することを実測確認済み。
  `refactoring-candidates.md`から項目6を削除済み。`pytest tests`956件全件
  パス、`ruff check`／`ruff format --check`パス確認済み。
- 項目7（説明変数X生成ロジックが3系統でほぼ同一）: 対応済み・コミット待ち
  （本セッションで実装、コミット前確認待ち）。**実装前の調査で判明した
  重要な事実**: 当初の項目記載は`scale_variance`/`scale_variance_mild`も
  対象に含んでいたが、実際にはlinear（Xをその場でスケーリングしてから
  yの計算に使う）とnonlinear/IV（y・x_endogは未スケーリングのXで計算し、
  出力直前にのみ列をスケーリングする、両ファイルのコメントに明記済み）で
  タイミングが意図的に異なると判明したため、ユーザー確認の上
  **`scale_variance`系は対象から除外**した。真に完全一致していた
  `moderate_multicollinearity`/`high_condition_number`（相関付き設計行列
  生成）・`perfect_multicollinearity`（3列目の上書き）の2ブロックのみを
  `benchmark/_common.py`の`correlated_design_matrix(rng, scenario, n, k)`・
  `apply_perfect_multicollinearity(X)`に集約した（linear/nonlinear/iv
  3ファイル、`k`下限チェックは各ファイル既存のタイミング・メッセージのまま
  変更していない）。エラーメッセージが変更前と完全一致することを実測確認済み。
  `freeze_datasets.py`の出力（linear/nonlinear/iv全系統）が変更前後で完全
  一致することも確認済み。`refactoring-candidates.md`から項目7を削除済み。
  `pytest tests`956件全件パス、`ruff check`／`ruff format --check`パス
  確認済み。
- 上記以外（`refactoring-candidates.md`項目8〜35・37・39〜41・43）は未着手。
  `refactoring-candidates-2.md`は項目47・48が対応済み（上記）、項目44〜46・49は
  未着手。項目50以降は別セッションが並行して追記・コミットしているため、
  このスナップショットでは網羅しない（同ファイルを直接参照すること）。

**並行作業についての注意**: 本セッションと並行して、別セッションが
`refactoring-candidates.md`を対象にした別の作業（コード解説中の気づき記録等）を
行っており、衝突を避けるため`docs/planning/specs/refactoring-candidates-2.md`
という新規ファイルに項目44以降を追記する運用に切り替えている（コミット`a197ba5`）。
次にこの随時対応を再開する際は、`refactoring-candidates.md`だけでなく
`refactoring-candidates-2.md`側の新規項目も合わせて確認すること。

**環境についての注意（2026-08-22判明）**: `~/.claude/`（Claude Codeのセッション
履歴・メモリ）は`.devcontainer/docker-compose.yml`のvolumeマウント対象外
（マウントされているのは`/workspaces/econometricsmodels`本体と`.cargo`関連のみ）
のため、**devcontainerの再ビルドで消える**。セッションをまたいで残したい情報は
Claude Codeのメモリ機能に頼らず、必ずリポジトリ内のファイル（本ドキュメント等）に
書くこと。

---

## `/explain-code`による`benchmark/`・`tests/`解説ウォークスルーの進捗（フェーズ外、2026-08-16〜）

上記の「`refactoring-candidates.md`駆動の随時対応」とは**別のセッション・別の目的**の
継続作業。ユーザーが`benchmark/`・`tests/`配下をファイル単位で`/explain-code`スキルに
沿って通読し、統計学的な意味・設計判断を確認しつつ、気づいた重複・設計上の疑問点を
その都度`refactoring-candidates.md`（またはブロック中は`refactoring-candidates-2.md`）・
`test-coverage-candidates.md`に記録し、必要に応じてGitHub Issueも作成する運用
（2026-08-22時点、devcontainer再ビルド直前にセッション履歴保全のため記録）。

**解説済み**:
- `benchmark/iv/`: `fixtures/generate_iv_fixtures.py`・`run_linearmodels_benchmark.py`・
  `fixtures/generate_iv_gmm_fixtures.py`・`run_ivreg_benchmark.R`・
  `fixtures/generate_iv_crosscheck_fixtures.py`・`freeze_iv_datasets.py`
- `benchmark/linear/`: `run_lm_crosscheck_benchmark.R`・
  `fixtures/generate_ols_crosscheck_fixtures.py`・`run_lm_predict_crosscheck.R`・
  `fixtures/generate_wls_crosscheck_fixtures.py`
- `benchmark/nonlinear/`: `run_glm_crosscheck_benchmark.R`・
  `fixtures/generate_logit_crosscheck_fixtures.py`・
  `fixtures/generate_probit_crosscheck_fixtures.py`
- `benchmark/_common.py`（`freeze_scenarios`/`run_freeze_cli`部分）・`benchmark/_common.R`
- `benchmark/performance/`: `compare_performance.py`・`render_performance_summary.py`
  （全区切り解説済み）
- `tests/`: `_assertions.py`・`_helpers.py`・`_tolerances.py`・`conftest.py`・
  `test_ols.py`・`test_ols_fixtures.py`・`test_ols_crosscheck.py`
  （全区切り解説済み。これでOLS関連ファイル一式が完了）

**次に解説予定**: `tests/test_wls*.py`等、残り15ファイル
（ユーザー指示によりOLS関連ファイルを一通り見終えたので次の手法に進む想定。
系統・手法の順序は次回セッションでユーザーに確認）。OLS3ファイル解説後の
ユーザーとの質疑で、数値比較の役割・許容誤差の一貫性・オプションのfixtures
カバレッジ・クラスターのシナリオ横断検証漏れ・命名の不統一・Intercept→const
正規化のタイミング不一致・Wooldridge実データ検証の非対称等について多数の
指摘を受け、`refactoring-candidates-2.md`項目51〜64・
`test-coverage-candidates.md`項目25〜33に記録済み（詳細下記）。特に
項目52・57（`test_ols.py`/`test_ols_fixtures.py`/`test_ols_crosscheck.py`の
3ファイルが互いに矛盾する役割分担の説明を持つ）・項目63（Intercept→const
正規化を生成時に統一する案、項目44を包含）・項目27/28/29/33（主リファレンス
statsmodels側がRクロスチェック側よりオプション・シナリオ・実データの
検証範囲が狭いという同型のパターンが4例見つかっている）は、他手法の
ファイルを解説する際も同じ観点で確認するとよい。

**このウォークスルー中に作成したGitHub Issue**: [#246](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/246)
（検定分布・診断統計量の運用ノート`docs/spec/inference-conventions.md`化）・
[#247](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/247)
（Cragg-Donald統計量の再検討）・[#249](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/249)
（GMMのC統計量実装）・[#256](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/256)
（GMM/Hansen JのRクロスチェック再検討）・[#264](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/264)
（Logit/Probitの最適化methodにFisher-scoring追加検討）・
[#266](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/266)
（polars以外のDataFrame（pandas等）を渡すと`ValidationError`ではなく
`AttributeError`が漏れる、`test_ols.py`解説中の質問を受けて実際に検証し発覚）・
[#267](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/267)
（Rとの計算慣習差〔HAC・AIC/BIC等〕に完全一致させる互換モードの要否、
優先度低、`test_ols_fixtures.py`解説中の議論から）。いずれもオープンのまま。

**`.claude/rules/testing-policy.md`への追記（2026-08-23）**: OLSクロスチェック
用Rスクリプト（`run_lm_crosscheck_benchmark.R`等）解説からの派生議論として、
ユーザーから「クロスチェックスクリプト内で本実装と同じ計算式を手計算している
箇所（AIC/BIC・ロバストWald検定・観測情報行列Hessian等）は、本実装と
同じ開発者が書いているため独立検証として機能しないリスクがあるのでは」
という指摘があり、対応方針を`testing-policy.md`「リファレンス実装」章に
ルールとして明文化した（独立パッケージでの確認を優先→無理ならformula非依存の
数値微分等で検証→それも無理なら理由をコメント明記、の優先順位）。Issue #267
にも関連コメントとして追記済み。Tobit実装時（主リファレンスがR単独で
第三者実装による三角測量が効かない）はこのルールが特に重要になる。

**候補メモの状態（2026-08-23時点）**:
- `refactoring-candidates.md`: 項目1〜43（このウォークスルー由来の最後の追記は項目43）。
  上記「`refactoring-candidates.md`駆動の随時対応」セッションが並行して対応中のため、
  このウォークスルーからの新規追記は`refactoring-candidates-2.md`（項目44〜64、
  直近の追記は`test_ols_crosscheck.py`解説時の`_assert_close`命名衝突
  〔スカラー/辞書で意味が逆〕・`NON_HAC_COV_TYPES`等cov_typeリストの3階層重複・
  `_check_result`/`_assert_fit_stats_close`命名不統一・Intercept→const正規化の
  タイミング不一致〔項目44を包含〕・`predict()`の`"fitted"`キー命名・項目57への
  追記で3ファイル目の役割分担矛盾を確認）に切り替えている。**両ファイルの統合は
  `refactoring-candidates.md`側の随時対応が一区切りついた後にユーザー判断で行う**
  （現時点では統合しない）。
- `test-coverage-candidates.md`: 項目1〜33（直近3件は`test_ols.py`/
  `test_ols_fixtures.py`/`test_ols_crosscheck.py`解説時、`time_col`存在チェック・
  `fit()`本体のNaN/無限大チェックのテスト欠如・Wooldridge実データが主リファレンス
  〔statsmodels〕側で未検証）。こちらはブロックされていないため引き続き
  直接追記してよい。

**再開時の確認事項**: 上記「並行作業についての注意」と同じく、`refactoring-candidates.md`
側の対応が進んでいれば項目1〜43の「状態」欄が更新されているはずなので、再開前に
`git log`・該当ファイルの差分を確認すること。

**devcontainer再ビルド直前に発見した未コミットの変更（本ウォークスルーとは無関係、
2026-08-22時点）**: `.devcontainer/Dockerfile`に未コミットの差分があり、
`fixest`/`plm`/`ivreg`/`jsonlite`/`marginaleffects`/`sandwich`/`lmtest`のRパッケージを
`remotes::install_version()`でバージョン固定するIssue #239対応と見られる変更が
入っている。このウォークスルーのセッションが行った変更ではない（別セッション・
別作業由来の可能性が高い）ため、内容の妥当性は未検証。再ビルド後、この変更が
意図したものか、コミットすべきかをユーザーに確認すること。

---

## フェーズ1: リファクタリング用スキル

**目的**: コード（Python/R/Rust）・コメント・ディレクトリ構造・ドキュメント
（README/CLAUDE.md/仕様書）を横断的にリファクタリングするための観点を固定した
スキル（or 既存スキル/エージェントの代用）を用意する。

**観点（Issue #231より）**:
- 重複したロジックの共通化（分離したままが適切な場合の判断基準を含む）
- パフォーマンス劣化箇所の指摘・修正
- コメント内のIssue番号への言及・冗長な説明の削除／要約
- 不要になったファイルの削除
- ロジックを壊さないためのテスト追加提案

**既存スキル/エージェントとの役割分担（要整理）**:
- `.claude/skills/simplify` — 変更済みコードの重複/簡潔化/効率化レビュー＋適用
- `.claude/skills/review-rust` / `review-python` — 規約準拠・設計整合性レビュー
- `.claude/skills/review-testing` — テスト網羅性レビュー
- 上記は「直近の差分」を対象にする傾向。本Issueは**手法横断・実装当時からの
  蓄積**を対象にするため、新規スキルが必要か、既存スキルの対象範囲指定
  （ファイル/ディレクトリ指定）で代用できるかを検討する。

**状態**: 完了

**メモ**:
- 既存スキル（`review-rust`/`review-python`/`review-testing`）は指摘のみ・`git diff`
  スコープが中心で、実装当時から蓄積した既存コード・ドキュメントの横断的な整理には
  非対応と判断。`simplify`（組み込みスキル）も重複/簡潔化/効率化の指摘→適用は行うが
  diffスコープ限定で、Issue番号コメント整理・不要ファイル削除・ディレクトリ構造整理・
  ドキュメント整理は対象外。
- ユーザー確認の結果、新規スキルは「指摘→適用まで一括」「コード＋コメント＋
  ディレクトリ構造＋ドキュメントを1つの統合スキルで扱う」方針で作成することに決定。
- `.claude/skills/refactor/SKILL.md` を新規作成。対象範囲・方針は都度ユーザーが
  `$ARGUMENTS`で指定する形（自動網羅ではない、Issue #231の前提通り）。
  観点はIssue #231 4章の5点＋ディレクトリ構造・ドキュメント整理。
  破壊的操作（削除・移動）は計画提示→合意→適用の順を必須にした。
  ロジックの挙動を変える変更（バグ修正含む）は範囲外と明記（特にフェーズ7のengine層）。
  テスト拡充自体は範囲外とし`review-testing`/`test-new`に委譲。
- 運用しながら実際の動作を見て改善していく方針（ユーザー確認済み）。

---

## フェーズ2: `benchmark/`ディレクトリの整理とリファクタリング

**現状の課題（Issue #231より）**:
- テスト用データ作成コード・参照用パラメータJSON作成コード・パフォーマンス測定
  コードが混在し、手法追加（WLS/Logit/Probit等へのパフォーマンス測定拡大）に
  対応できない構造になっている。
- pyfixestが実際には使われていない箇所のファイル・コメントが残存している疑い。

**進め方**:
1. ディレクトリ・ファイル配置の整理（構造変更のみ、ロジックは変更しない）
2. コードの整理（不要ファイル削除、pyfixest関連の削除、コメント整理）

**現状構成（着手前スナップショット）**:
```
benchmark/
├── README.md
├── generate_synthetic_datasets.py   # 系統非依存: 合成データ生成
├── freeze_datasets.py               # 系統非依存: 上記をCSV固定
├── load_wooldridge.py               # 系統非依存: Wooldridgeデータロード
├── compare_performance.py           # OLS専用パフォーマンス測定（root直下、系統構造と不整合）
├── render_performance_summary.py    # 上記の結果整形（同上）
├── linear/     (OLS/WLS) fixtures/ + run_statsmodels_benchmark.py +
│                run_lm_crosscheck_benchmark.R + run_lm_predict_crosscheck.R +
│                run_pyfixest_benchmark.py + run_fixest_benchmark.R（未使用、後述）
├── nonlinear/  (Logit/Probit) fixtures/ + generate_binary_choice_datasets.py +
│                run_statsmodels_benchmark.py + run_glm_crosscheck_benchmark.R
├── iv/         (2SLS/GMM) fixtures/ + generate_iv_datasets.py +
│                run_linearmodels_benchmark.py + run_ivreg_benchmark.R
└── panel/      run_plm_benchmark.R（FE/RE用スキャフォールド、engine側未実装＝Issue #231対象外）
```

`linear`/`nonlinear`/`iv`の3系統は既に「系統ディレクトリ＋`fixtures/`」で一貫した構造
（`reference-benchmark`スキルに文書化済み）。問題は`compare_performance.py`/
`render_performance_summary.py`がroot直下にありながら中身はOLS専用（`LIBRARIES`/
`COV_TYPES`/回帰式ハードコード）という不整合のみだった。

**状態**: 完了（ステップ1: ディレクトリ配置整理、ステップ2: コード整理とも完了）。

**メモ**:
- ユーザー確認の結果、`compare_performance.py`/`render_performance_summary.py`は
  `benchmark/performance/`という新規の系統横断ディレクトリへ移動することに決定
  （今後WLS/Logit/Probitへ性能測定を拡張する際、系統ごとにコードを複製せず
  1箇所に集約しやすくするため）。
- `panel/run_plm_benchmark.R`はFE/RE着手時（Issue #231とは別スコープ）まで触らず
  そのまま残すことに決定。
- 実施内容:
  - `git mv`で`benchmark/compare_performance.py`→`benchmark/performance/compare_performance.py`、
    `benchmark/render_performance_summary.py`→`benchmark/performance/render_performance_summary.py`。
  - `compare_performance.py`の`generate_synthetic_datasets`インポートが
    ディレクトリ移動で壊れるため、他の`benchmark/<系統>/fixtures/*.py`と同じ
    `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`パターンで修正
    （移動前は同一ディレクトリだったため素の`import`で通っていた）。
  - 移動に伴い参照更新: `.github/workflows/benchmark_ols.yml`（`working-directory`・
    artifactパス）、`docs/spec/ols-performance-notes.md`（パス・再現コマンドの相対パス、
    ついでに既存の誤記`../docs/planning/specs/_ols_performance_results.json`→
    正しくは`docs/spec/`配下、`.gitignore`の`docs/spec/_*.json`と整合させて修正）、
    `docs/spec/ci-cd-notes.md`、`.claude/skills/cicd/SKILL.md`、
    `.claude/skills/reference-benchmark/SKILL.md`（`benchmark/performance/`の説明を追加）。
  - 移動後、`compare_performance.py --worker`単体実行・`_run_isolated`経由の
    サブプロセス再帰呼び出し（実際のCIスイープが使う経路）の両方で動作確認済み。
### ステップ2（コード整理）の調査結果・決定事項（実装はまだ、次回着手）

ユーザーからの追加指摘を受けて`benchmark/`（一部`tests/api_tests/`との重複）を
再調査した。**以下は調査・方針決定のみで、コード変更は未実施**。

**1. `generate_synthetic_datasets.py`は2つの役割が同居していた**
`generate_dataset()`/`SCENARIOS`（linear専用DGP、`iv`/`nonlinear`の自前DGPからは
一切importされていないことを確認済み）と、`imbalanced_cluster_groups()`
（全系統のfixture生成・テストから使われる真の系統非依存ユーティリティ、
20ファイル超からimportされていることを確認済み）が1ファイルに混在。
「linearだけrootにある」という不整合の正体はこれだった。
→ **決定**: 分割する。linear専用DGPは`benchmark/linear/`へ移動、
`imbalanced_cluster_groups`はroot（`benchmark/_common.py`、後述）に残す。

**2. `freeze_datasets.py`の肥大化**
201行中、SYNTHETIC(linear)/LOGIT/PROBIT/IVの4ブロックがほぼ同一パターン
（シナリオループ→生成→CSV書き出し→true_beta収集→JSON書き出し）をベタ書き。
→ **決定**: 共通ヘルパー（`_freeze_family(generator_fn, scenarios, prefix,
output_dir, overrides=...)`相当）で圧縮した上で、**freeze処理自体も各系統
ディレクトリに分割**する（rootは薄いディスパッチャのみ残す）。

**3. `linear/run_pyfixest_benchmark.py`**
fixture生成には未接続（pyfixestは精度検証に使わない方針のため）、
`compare_performance.py`もpyfixestを直接importしていて経由していない。
他に依存箇所なし。→ **決定**: 削除する。

**4. `linear/run_fixest_benchmark.R`**
内容を確認したところ、固定効果構文（`| entity`等）を含まない素の
`feols(formula, data=df, weights=...)`呼び出しで、`panel-api-design.md`が
想定する将来のFEクロスチェック用スクリプト（固定効果項必須）とは中身が違う
ため、panelへ移動しても「そのまま使える叩き台」にはならないと判明。
→ **決定**: 今回削除する。FE着手時（Issue #231とは別スコープ）に
固定効果構文込みで新規作成する（CLAUDE.mdの「将来のための設計をしない」方針）。

**5. SCENARIOSの重複（3〜4階層）**
同じシナリオ名リストが、`generate_*_datasets.py`の`SCENARIOS`（全シナリオ）→
`fixtures/generate_*_fixtures.py`の`NUMERIC_SCENARIOS`（数値比較サブセット）→
`tests/api_tests/test_*_fixtures.py`の`SCENARIOS`（pytest parametrize用）
という3階層（＋`freeze_datasets.py`独自コピーで4階層目）で再定義されていた。
OLSで実測したところ、`generate_ols_fixtures.py`の`NUMERIC_SCENARIOS`と
`test_ols_fixtures.py`の`SCENARIOS`は**完全に同一リスト**（`COV_TYPES`も同様）。
`tests/`が`benchmark/`を参照する形になるが、依存関係を調査した結果、
`generate_ols_fixtures.py`はstatsmodels（既に`test`依存グループに含まれ
pytest実行時に既存）のみに依存し、Rサブプロセス呼び出しは関数呼び出し時のみ
（`subprocess`のimport自体はモジュールロード時に無害）と確認できたため、
`tests/`側の「Rランタイム非依存」という既存方針を壊さずにimport可能と判断。
→ **決定**: `tests/`も含めて一元化する。`benchmark/<系統>/fixtures/
generate_<手法>_fixtures.py`側の`NUMERIC_SCENARIOS`/`COV_TYPES`を正とし、
`tests/api_tests/test_*_fixtures.py`はそこからimportする形にする
（フルシナリオ⊃数値比較サブセットという包含関係の表現方法は実装時に設計）。
実装はフェーズ2・3どちらに属するかも含め次回着手時に決める。

**6. Issue番号の残存（`freeze_datasets.py`以外）**
5ファイルで確認: `iv/generate_iv_datasets.py`、
`iv/fixtures/generate_iv_crosscheck_fixtures.py`、`iv/run_linearmodels_benchmark.py`
（4箇所、いずれもIssue #171）、`linear/run_lm_crosscheck_benchmark.R`（Issue #107）、
`nonlinear/run_glm_crosscheck_benchmark.R`（Issue #84）。経緯の記録と非自明な
WHYの説明が混在しているため、一律削除ではなく`refactor`スキルの観点3
（要約して残すか削除か）に沿って個別判断する。

**7. 追加で見つかった共通化候補**
- `_hac_auto_lag(n) = int(4 * (n/100)**(2/9))`: 完全に同一の実装が**5ファイル**
  （`benchmark/performance/compare_performance.py`、
  `linear/fixtures/generate_wls_crosscheck_fixtures.py`、
  `linear/fixtures/generate_ols_crosscheck_fixtures.py`、
  `iv/fixtures/generate_iv_crosscheck_fixtures.py`、
  `iv/run_linearmodels_benchmark.py`）にコピペされている。
- `_load_synthetic`/`_load_iv_dataset`（ユーザー指摘）: `linear/nonlinear/iv`の
  各`run_statsmodels_benchmark.py`/`run_linearmodels_benchmark.py`で、
  「`{prefix}_{scenario}.csv`と`{prefix}_true_beta.json`を読む」という
  同一パターンが3箇所に実装されている。`prefix`引数化で統合可能。
- `DATA_DIR`のパス構築が上記3ファイルで重複（`parents[N]`の深さのみ違う）。
- `_meta`辞書の構築（13箇所）は`method`/`generated_at`/`primary_reference`/
  `{ref}_version`/`note`という共通の形はあるが、`note`は手法固有の文章のため
  優先度は上記2つより低いと判断。
→ **決定**: `benchmark/_common.py`を新設し、`_hac_auto_lag`・
`DATA_DIR`構築・`_load_frozen_dataset`相当（`_load_synthetic`/`_load_iv_dataset`の
統合版）・`imbalanced_cluster_groups`（項目1）を集約する。`_meta`辞書は
今回は見送り。

### ステップ2 実施結果（完了）

上記7項目の決定事項を全て実装した。

1. **`benchmark/_common.py`新設**: `imbalanced_cluster_groups`・`hac_auto_lag`・
   `DATA_DIR`・`load_frozen_dataset`・`freeze_scenarios`の5関数/定数を集約。
2. **`generate_synthetic_datasets.py`分割**: linear専用DGP部分は
   `benchmark/linear/generate_synthetic_datasets.py`へ`git mv`、
   `imbalanced_cluster_groups`は`_common.py`へ。importer側（`compare_performance.py`
   他）のパスを追従。
3. **`imbalanced_cluster_groups`のimport元統一**: `_common`からimportする形に
   22ファイル（`benchmark/`17・`tests/api_tests/`9、重複あり）を統一。
   `generate_logit_fixtures.py`/`generate_probit_fixtures.py`に**未申告だった
   ローカル再実装**（`_imbalanced_cluster_groups`、_common.pyと byte-for-byte
   同一）を発見し、合わせて統合（当初の想定外の追加成果）。
4. **`_hac_auto_lag`/`_load_synthetic`/`_load_iv_dataset`/`DATA_DIR`統合**:
   5+3ファイルを`_common.py`参照に統一。既存の`from run_statsmodels_benchmark
   import DATA_DIR`という間接import（8ファイル）を壊さないよう、各
   `run_*_benchmark.py`が`_common`からimportし直すことで透過的に維持しつつ、
   実際には11ファイル全てを`_common`直接参照に更新（再エクスポート依存を残さず
   単一定義元を明確化）。
5. **`freeze_datasets.py`の系統別分割**: `benchmark/{linear,nonlinear,iv}/
   freeze_<系統>_datasets.py`を新設、`freeze_scenarios`ヘルパーで各ブロックを
   圧縮。root`freeze_datasets.py`は3つの`freeze()`を呼ぶだけの薄い
   ディスパッチャに変更。**分割後の出力を一時ディレクトリに生成し、
   コミット済みの`tests/api_tests/fixtures/benchmarks/data/`と`diff -rq`で
   完全一致することを確認済み**（40ファイル、バイト単位で同一）。
6. **未使用ファイル削除**: `benchmark/linear/run_pyfixest_benchmark.py`・
   `run_fixest_benchmark.R`を`git rm`。参照していた
   `.claude/skills/reference-benchmark/SKILL.md`の該当記述も削除。
7. **SCENARIOS/NUMERIC_SCENARIOSの一元化**: `benchmark/<系統>/fixtures/
   generate_<手法>_fixtures.py`側のリストを正とし、`tests/api_tests/
   test_*_fixtures.py`・`test_*_crosscheck.py`はそこからimportする形に
   10ペア全て変更（COV_TYPESは値が一致するOLS/WLS/IV-GMMの3ペアのみ合わせて
   統一、Logit/Probit/IV(2SLS)/IV-crosscheckはgenerator側にのみ`cluster`が
   含まれ意図的に非対称なため据え置き）。
8. **Issue番号コメントの整理**: `benchmark/`配下で確認した6ファイル・
   10箇所すべてについて、`refactor`スキルの観点3
   （非自明なWHYが無ければ削除）に沿って個別判断し、全て「単なる経緯の記録」と
   判断して番号のみ削除（周辺の設計判断の説明文は保持）。
9. **ドキュメント整合**: `.claude/skills/reference-benchmark/SKILL.md`
   （ディレクトリ構成節・手順1/4）、`.claude/rules/testing-policy.md`、
   `docs/spec/wls-spec.md`の`benchmark/generate_synthetic_datasets.py`パス言及を
   `benchmark/linear/generate_synthetic_datasets.py`に修正。

**最終確認**: `ruff check .` / `ruff format --check .` 全件パス、
`pytest tests/api_tests/`（Rクロスチェック含む）670件全件パス、
freeze出力のバイト一致確認済み。

### 追加ラウンド: ユーザー再指摘への対応（完了）

ステップ2完了後、ユーザーから4点の追加指摘を受けて対応した。

1. **`freeze_datasets.py`のIssue #231記述削除**: モジュールdocstring内の
   唯一の言及を削除。
2. **命名規則の系統名統一**: `generate_synthetic_datasets.py`→
   `generate_linear_datasets.py`（関数`generate_dataset`→`generate_linear_dataset`）、
   `generate_binary_choice_datasets.py`→`generate_nonlinear_datasets.py`
   （中身が実際に表す内容を正確に表しているため関数名`generate_binary_choice_dataset`
   は維持、ファイル名のみ系統名に統一）。呼び出し元（`freeze_<系統>_datasets.py`・
   `fixtures/generate_*.py`・テスト・SKILL.md・`testing-policy.md`・
   `wls-spec.md`）を全て追従。
3. **`__main__`ブロックの共通化**: `_common.py`に`run_freeze_cli`
   （`freeze_datasets.py`・`freeze_<系統>_datasets.py`4ファイルで完全に同一だった
   argparse定義・出力先ディレクトリ作成・freeze呼び出し・完了printを集約）と
   `preview_dataset`（`generate_<系統>_datasets.py`3ファイルで同型だった、単体実行
   時のシナリオ1件プレビュー表示を集約）を追加。各`freeze()`関数からmkdir/print
   を除去し純粋化（ディレクトリ作成・printは`run_freeze_cli`側の責務に統一）。
4. **Rスクリプトの共通化**: `run_lm_crosscheck_benchmark.R`と
   `run_ivreg_benchmark.R`で完全に同一だった「`coeftest()`からの係数・標準誤差
   抽出」「ロバストWald F検定」の2ブロックを`benchmark/_common.R`に抽出し
   `source()`で読み込む形にした。cov_type→vcov分岐自体（lmはHC0-3・weight対応、
   ivregはHC0-1のみ対応という実質的な差分がある）は共通化していない（過剰な
   抽象化を避けるため）。`run_glm_crosscheck_benchmark.R`はProbit対応の観測情報
   行列計算が特殊で共通化できる箇所が見当たらず対象外とした。Rには`__file__`
   相当が無いため、各スクリプトが`commandArgs(trailingOnly=FALSE)`の`--file=`
   から自身のディレクトリを特定して`source()`する方式にした。

**最終確認**: `ruff check .` / `ruff format --check .` 全件パス、
`pytest tests/api_tests/`670件全件パス、freeze出力のバイト一致確認済み、
Rスクリプト（`run_lm_crosscheck_benchmark.R`・`run_ivreg_benchmark.R`）を
実際に実行し既存crosscheckフィクスチャとバイト一致することを確認済み。

さらにユーザー指示により、`benchmark/*/fixtures/generate_*.py`11本全てを
実行し`tests/api_tests/fixtures/benchmarks/*.json`と再照合した
（`_meta.generated_at`のみ差異を許容、それ以外は完全一致を要求）。
10本は一致を確認。**残り1本（`generate_logit_crosscheck_fixtures.py`）は
`near_separation`シナリオ×`classical`×logitの組み合わせでR側が
`stopifnot(isTRUE(all.equal(bread_obs, bread(model), tolerance=1e-6)))`
（`run_glm_crosscheck_benchmark.R`の不変条件チェック）で失敗し実行不能**。
`git stash`で今回の変更を全て戻した状態（今回のリファクタリング前のコード）
でも同じ入力・同じコマンドで同一エラーが再現することを確認済みのため、
**今回のリファクタリングが原因ではなく、既存の潜在バグ（環境のRパッケージ
バージョン差等による数値不安定性の可能性）**。フェーズ2のスコープ外のため
今回は修正しない。フェーズ4（テスト拡充、Logit/Probitレビュー）着手時に
要調査・要修正。

---

## フェーズ3: `tests/`ディレクトリの整理とリファクタリング

**現状の課題（Issue #231より）**:
- `tests/api_tests`という中間ディレクトリの要否
- `_assert_close` / `_assert_dict_close` 等の共通アサーションが各テストファイルに
  重複実装されている疑い
- 許容誤差の設定が分散しており見落としリスクがある
- Issue番号への言及が残存している箇所がある

**現状構成（着手前スナップショット）**:
```
tests/api_tests/
├── conftest.py
├── test_{ols,wls,logit,probit,iv}.py               # API/構造テスト
├── test_{ols,wls,logit,probit,iv,iv_gmm}_fixtures.py # 数値照合ベンチマーク
├── test_{ols,wls,logit,probit,iv}_crosscheck.py      # Rクロスチェック
└── fixtures/benchmarks/
```

**状態**: 完了

**メモ（現状調査・計画）**:

`Explore`エージェントによる17ファイル横断調査の結果、以下を計画として確定した
（ユーザー確認済み、実装はこれから）。

1. **ディレクトリ移動**: `tests/api_tests/` → `tests/`へフラット化（`git mv`）。
   Rust側テストは`engine`の`mod tests`に完全分離済みで、Python側に
   `api_tests`以外の種別が今後増える見込みが薄いため中間ディレクトリ不要と判断。
   参照更新対象: `CLAUDE.md`3章の構成図、`pyproject.toml`（コメント）、
   `.github/workflows/ci_python.yml`（pathsトリガー・実行コマンド）、
   `.claude/rules/testing-policy.md`、`.claude/rules/python-style.md`、
   `.claude/skills/test-new/SKILL.md`・`test-run/SKILL.md`・
   `reference-benchmark/SKILL.md`、`docs/spec/ci-cd-notes.md`。
2. **`tests/_assertions.py`新設**: `assert_close`/`assert_dict_close`/
   `check_result`（ols・wls用／logit・probit用／iv・iv_gmm用の3パターン）/
   `check_margeff`（fixtures用／crosscheck用の2パターン）を集約。
   **`test_ols_crosscheck.py`・`test_iv_crosscheck.py`の基本`_assert_close`は
   許容誤差の計算式が他と異なる（フェーズ3.5参照）ため統合対象から除外し、
   現状のファイル内実装のまま残す**（挙動を変えないため）。
3. **`tests/_tolerances.py`新設**: 許容誤差の値を手法ごとに辞書化して集約
   （計算式の統一はフェーズ3.5で別途対応、値の集約のみ今回実施）。
4. **`tests/_helpers.py`新設**: クラスター列付与ヘルパー
   （`with_cluster_groups`、「行番号%N」パターン22箇所）、
   `separation_suspected`DGP（logit/probit共通）、`MROZ_X`定数、
   Wooldridgeロードの一本化。クラスター列ヘルパーは`benchmark/_common.py`の
   `imbalanced_cluster_groups`とは役割が近いが、`benchmark/`と`tests/`は
   ライフサイクルが別（testing-policy.md）という既存方針を踏まえ、
   テスト専用ロジックのため`tests/`側に置くことをユーザー確認済み。
5. **`conftest.py`拡張**: `binary_dataset`フィクスチャ（logit/probit共通）、
   `DATA_DIR`定数、`sys.path`への`benchmark/`直下挿入（系統別サブディレクトリの
   挿入は各ファイルに残す）、`fixtures()`JSON読み込みの汎用化。
6. **COV_TYPESの一元化拡張**: logit/probit系（fixtures・crosscheck）にも、
   既にols/wls/ivで適用済み（フェーズ2）の「`generate_*_fixtures.py`から
   importして単一定義元にする」パターンを適用（値は現状と同一のため挙動不変）。
7. **IVの`fit()`呼び出し共通化**: `test_iv.py`に`_our_fit`ヘルパー
   （`test_ols.py`と同様の形）を追加し、40箇所以上の重複呼び出しを整理。
8. **Issue番号コメント整理**: `#153`（test_wls.py 2箇所、番号のみ削除・
   「OLSと共有される検証経路」という説明は残す）、`#171`（4ファイル5箇所、
   現状との記述一致を確認しつつ経緯のみ削除）、`#231`（4ファイル、
   「単一定義元にする」という設計判断の説明を残し番号のみ削除）。

**除外事項**: crosscheckテストの許容誤差計算式の不一致（`tol`計算式が
`test_ols_crosscheck.py`・`test_iv_crosscheck.py`のみ他と異なる）は
ロジックの挙動を変える修正のため、`refactor`スキルの範囲外としフェーズ3.5へ
切り出した（ユーザー指示）。

### 実施結果（完了）

計画の1〜8を全て実装した。

1. **ディレクトリ移動**: `git mv`で17ファイル＋`fixtures/`を`tests/api_tests/`
   から`tests/`へ移動。付随して各ファイル内の`sys.path.insert`の
   `parents[2]`（旧`tests/api_tests/test_x.py`基準）を`parents[1]`
   （新`tests/test_x.py`基準）に修正、docstring内のパス表記・
   `benchmark/_common.py`の`DATA_DIR`組み立て・`benchmark/`配下の
   フィクスチャ生成スクリプトのデフォルト出力パスも追従。参照更新した
   非コードファイル: `CLAUDE.md`（3章構成図・6章要点）、`docs/plan.md`、
   `pyproject.toml`（コメント）、`.github/workflows/ci_python.yml`
   （pathsトリガー・実行コマンド）、`.claude/settings.json`（`ask`パス）、
   `.claude/rules/testing-policy.md`・`python-style.md`、
   `.claude/skills/reference-benchmark/SKILL.md`・`test-run/SKILL.md`・
   `test-new/SKILL.md`、`.claude/agents/rust-reviewer.md`・
   `python-reviewer.md`・`testing-completeness-reviewer.md`、
   `docs/spec/ci-cd-notes.md`、`python_package/econometricsmodels/
   nonlinear/CLAUDE.md`、`engine_pybind/src/nonlinear/CLAUDE.md`、
   `engine/src/iv/CLAUDE.md`、`benchmark/README.md`、
   `docs/planning/specs/iv-api-design.md`。`docs/planning/specs/
   panel-iv-issue-breakdown.md`（完了済みIssueのチェックリスト、履歴的な
   記録）と本ドキュメントの「着手前スナップショット」節は、過去の状態を
   記述する文書のため意図的に更新対象から除外した。
   移動後、`freeze_datasets.py`の出力を一時ディレクトリに生成し
   コミット済み`tests/fixtures/benchmarks/data/`と`diff -rq`で完全一致
   することを確認済み。
2. **`tests/_assertions.py`新設**: `assert_close`/`assert_dict_close`/
   `rename_intercept`/`check_margeff`を集約し、fixtures系6ファイル
   （ols/wls/logit/probit/iv/iv_gmm）から`functools.partial`で
   `rtol`/`atol`を束縛する形で参照するよう変更。crosscheck系5ファイルは
   計算式が現状不統一（フェーズ3.5参照）のため、今回は対象外とし
   ファイル内実装のまま維持。
3. **`tests/_tolerances.py`新設**: 全11ファイル（fixtures 6・crosscheck 5）
   の許容誤差の値を手法名をキーにした辞書に集約。計算式は変更していないため
   挙動は不変（`ols_crosscheck`/`iv_crosscheck`の異なる式もそのまま）。
4. **`tests/_helpers.py`新設**: `with_cluster_groups`（「行番号%N」パターン、
   正規表現による一括置換で23箇所を集約）、`separation_suspected_dataset`
   （`test_logit.py`/`test_probit.py`で完全同一実装だったDGP）、`MROZ_X`
   （4ファイル）、`wooldridge_loader`/`load_wooldridge_dataset`
   （3種類あったWooldridgeロードの書き方を統一。`test_ols_crosscheck.py`の
   複数データセット対応fixtureは`wooldridge_loader()`、単発ロードは
   `load_wooldridge_dataset(name)`）、`DATA_DIR`（12ファイルで重複していた
   `Path(__file__).resolve().parent / "fixtures" / "benchmarks" / "data"`
   の組み立てを集約）を追加。
5. **`conftest.py`拡張**: `binary_dataset`フィクスチャ（`test_logit.py`/
   `test_probit.py`の完全同一実装を統合）、`benchmark/`直下への
   `sys.path.insert`（11ファイルが個別に行っていたものを一度だけに集約。
   系統別サブディレクトリの挿入は各ファイルに残置）を追加。
6. **COV_TYPESの一元化拡張**: 調査の結果、logit/probitの
   `generate_*_fixtures.py`側`COV_TYPES`には`"cluster"`が追加で含まれており
   （fixture生成用の全網羅リストのため）、テスト側の意図的に絞った
   リスト（cluster抜き、クラスターは専用テストで別途検証）とは値が
   一致しないと判明（フェーズ2でOLS等について確認済みの「意図的な非対称」と
   同じ構造）。単純importすると挙動が変わるため、ユーザー確認の結果
   **据え置き**（現状のハードコードのまま）で決定。
7. **IVの`fit()`呼び出し共通化**: `test_iv.py`に`_our_fit`ヘルパー
   （`x_exog=["x1"], x_endog=["endog1"], instruments=["z1", "z2"]`が
   既定、`options`含め呼び出し側で上書き可能）を追加。正規表現による
   一括変換で、既定値と完全一致していた36箇所の`IV(...).fit()`呼び出しを
   `_our_fit(...)`に置換（異なる`x_exog`/`x_endog`/`instruments`を使う
   残り約20箇所は、それぞれ意図的に異なるテストケースであり
   重複ではないため元のまま維持）。
8. **Issue番号コメント整理**: `#153`（`test_wls.py`2箇所）・`#171`
   （`test_iv.py`2箇所・`test_iv_fixtures.py`2箇所・`test_iv_crosscheck.py`
   1箇所・`test_iv_gmm_fixtures.py`1箇所）・`#231`（4ファイル）の
   全11箇所を番号のみ削除し周辺の説明文は保持。調査中に`test_iv_fixtures.py`
   のdocstringが「Rクロスチェックは別issueで保留中（`ivreg`未導入のため）」
   という**現状と矛盾する記述**（実際には`test_iv_crosscheck.py`として
   実装済み）であることを発見し、あわせて修正した。

**最終確認**: `pytest tests`670件全件パス、`ruff check .`／
`ruff format --check .`全件パス。`cargo test`は対象外
（`engine`/`engine_pybind`/`python_package`は今回変更していない）。

---

## フェーズ3.5: crosscheckテストの許容誤差計算式バグ修正

**背景**: フェーズ3の現状調査で発覚。`test_ols_crosscheck.py`の`_assert_close`
（dict版）・`_assert_scalar_close`と、`test_iv_crosscheck.py`の`_assert_close`
（スカラー版）・`_assert_dict_close`が
`tol = rtol * max(abs(ref_val), 1e-8)`という式を使っている。これは他の大半
（`*_fixtures.py`全6ファイル、`test_wls_crosscheck.py`、`test_logit_crosscheck.py`、
`test_probit_crosscheck.py`、`test_iv_crosscheck.py`の`_assert_p_value_close`のみ）が
使う`tol = max(rtol * abs(ref_val), atol)`と**数学的に別物**（前者は`|ref|`が
小さいとき許容誤差が`rtol*1e-8`という極小値まで縮み、本来検出すべきでない
誤差でテストが誤って失敗しうる／逆に緩すぎて見逃しうる）。

`test_wls_crosscheck.py:79-82`のコメントは、この不一致を認識した上で
`test_ols_crosscheck.py`と異なる（＝修正済みの）式を採用した旨を明記しており、
**`test_ols_crosscheck.py`と`test_iv_crosscheck.py`が未修正のまま取り残された
可能性が高い**（本来の意図は`max(rtol*|ref|, atol)`と推測されるが、要検証）。

**進め方（`refactor`スキルではなく通常の実装フローとして対応）**:
1. どちらの式が正しい意図か（`ATOL`をなぜ`1e-8`固定にしていたか等）を
   コミット履歴・関連Issueから確認する。
2. 該当2ファイルの許容誤差式を正しい式に修正し、修正後に既存のRクロスチェック
   フィクスチャに対して実際にテストがパスするか（許容誤差を厳しくする方向の
   修正であれば、逆に既存の実測乖離を超えて失敗するケースが無いか）確認する。
3. 修正により新たにテストが失敗する場合、それが「これまで見逃していた本物の
   数値不一致」か「許容誤差の詰めすぎ」かを切り分ける。
4. フェーズ3で新設する`tests/_assertions.py`へ、修正後の式で統合する
   （フェーズ3時点で除外していた2ファイルの`_assert_close`をここで合流させる）。

**状態**: 完了

**メモ（実施結果）**:

1. **コミット履歴による意図確認**: `git log --all -S`で`test_ols_crosscheck.pyと
   異なり`というコメント文言をpickaxe検索し、導入元コミット`4ac9b80`
   （`fix(wls): R²・対数尤度（→AIC/BIC）の計算式を修正し、tests/api_testsを
   作成する`）を特定。コミットメッセージ・当該diffのコメントから、
   `tol = rtol * max(|ref|, 1e-8)`という式（ols/iv crosscheckが使い続けていた
   もの）は、`|ref|`が0近傍のとき許容誤差が`rtol*1e-8`という極小値
   （例: `RTOL_STRICT=1e-8`なら`1e-16`）まで縮んでしまい、機械精度ノイズで
   偽陽性の失敗を起こすバグであることが、WLSのRクロスチェックテスト作成時
   （`cluster`/`f_p_value`が5e-13程度まで下がるケースで発覚）に判明済み
   だったと確認できた。正しい式は`tol = max(rtol * |ref|, atol)`
   （常に絶対誤差フロア`atol`以上を保証する）で、`test_wls_crosscheck.py`は
   この時点で修正済みだったが、`test_ols_crosscheck.py`・
   `test_iv_crosscheck.py`は追従していなかった。
   - 数学的な影響範囲も確認: 修正前の式をB、修正後の式をAとすると、
     `|ref| >= atol`の領域ではA=B、`|ref| < atol`の領域では常にA≥B
     （Aの方が緩い）となることを確認済み。つまり今回の修正は許容誤差を
     **緩める方向にのみ**作用し、修正によって新たにテストが失敗すること
     は理論上あり得ない（既存のRクロスチェックフィクスチャに対して
     ステップ3「本物の数値不一致か許容誤差の詰めすぎか」の切り分けは
     不要と判明）。
2. **修正**: `test_ols_crosscheck.py`（`_assert_close`のdict版・
   `_assert_scalar_close`に加え、同じ式をヘルパー経由せず直書きしていた
   `test_predict_none_matches_r_fitted_values`・
   `test_predict_new_data_matches_r`内のループも発見し、`_assert_scalar_close`
   呼び出しに統一）、`test_iv_crosscheck.py`（`_assert_close`のスカラー版・
   `_assert_dict_close`）の計算式を`max(rtol * abs(ref_val), atol)`に修正。
   絶対誤差フロアの値（`1e-8`、修正前と同じ数値）は`tests/_tolerances.py`の
   `TOLERANCES["ols_crosscheck"]["atol"]`・`TOLERANCES["iv_crosscheck"]["atol"]`
   として追加（値自体は変更していない）。
3. **`tests/_assertions.py`への統合**: 式が他ファイルと一致したため、
   `functools.partial`で`tests/_assertions.py`の`assert_close`/
   `assert_dict_close`を束縛する形に置き換え、ファイル内の独自実装を削除
   （フェーズ3で6ファイルのfixtures系に適用したのと同じパターン）。
   `test_iv_crosscheck.py`の`_assert_p_value_close`（元々正しい式だった）も、
   `atol`のみ`ATOL_F_PVALUE`に差し替えた`partial(assert_close,
   atol=ATOL_F_PVALUE)`に簡潔化。
4. **検証**: `pytest tests`670件全件パス（新規失敗0件、事前の数学的分析
   通り）。`ruff check .`／`ruff format --check .`全件パス。

**除外事項（今回のスコープ外）**: `test_wls_crosscheck.py`・
`test_logit_crosscheck.py`・`test_probit_crosscheck.py`は元々正しい式を
使っており今回のバグ修正の対象外だが、これで全5つのcrosscheckファイルが
同じ計算式を使う状態になったため、`tests/_assertions.py`へさらに統合する
余地がある（フェーズ3時点では計算式の不一致を理由に全crosscheckファイルを
統合対象から除外していたが、その前提が解消された）。当フェーズの計画
（ステップ4）はols/iv crosscheckの2ファイルのみを対象としていたため、
残り3ファイルの統合は今回実施せず、次回リファクタリング時の候補として
記録するに留める。

**メモ**: (着手後に記載)

---

## フェーズ4: ロジック整理前のテスト拡充

**目的**: フェーズ5〜7（python_package/engine_pybind/engineのロジック変更）で
既存実装を壊さないよう、着手前にテストを拡充する。

**進め方**:
0. **着手前に`benchmark/`を再生成する**: `benchmark/freeze_datasets.py`で
   合成データセットCSV（`tests/fixtures/benchmarks/data/`）を、
   `benchmark/*/fixtures/generate_*.py`各スクリプトで参照用パラメータJSON
   （`tests/fixtures/benchmarks/*.json`）をそれぞれ再生成し、
   コミット済みの状態と一致することを確認する（フェーズ2追加ラウンドの
   「最終確認」で発見した`generate_logit_crosscheck_fixtures.py`の
   `near_separation`ケース失敗も含め、この時点で一連の生成過程が整合的で
   あることを確定させる）。生成データ・参照用パラメータJSONとも生成過程の
   パラメータ・シード値は変えていないため、通常は差分が出ないはず。
1. `review-testing`スキルでOLS/WLS/Logit/Probitのテストをレビューし、
   不足があれば充足する。
2. IV分（[#232](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/232)〜
   [#238](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/238)）の
   テスト拡充も合わせて実施する。

**状態**: 完了（linear系統〔OLS/WLS〕・nonlinear系統〔Logit/Probit〕・IV系統〔2SLS/GMM、#232〜238〕全て完了）

**メモ（linear系統〔OLS/WLS〕実施結果）**:

- **ステップ0（前提確認）**: `benchmark/freeze_datasets.py`でCSVを、
  `generate_ols_fixtures.py`/`generate_wls_fixtures.py`（statsmodels）で
  `ols.json`/`wls.json`を再生成し、`generated_at`以外は完全一致を確認。
  Rクロスチェック側（`generate_ols_crosscheck_fixtures.py`/
  `generate_wls_crosscheck_fixtures.py`）も同様に確認したところ、数値は完全
  一致・`_meta.r_version`のみ差分（4.2.2 → 4.5.3、コンテナのRバージョンが
  記録時点から更新されているだけで実測値には影響なし）だった。
- **ステップ1（`review-testing`スキルでのレビュー、linear系統分）**:
  `testing-completeness-reviewer`に`engine/src/linear/`・
  `engine_pybind/src/linear/`（`mod tests`が0件だった）・
  `python_package/econometricsmodels/linear/`・`tests/test_ols*.py`・
  `tests/test_wls*.py`・`benchmark/linear/`のレビューを依頼し、
  must fix 1件・should fix 7件・nice to have 4件、計12件の指摘を得た。
  ユーザー確認の上、全12件をこのフェーズで対応する方針とした（未対応の
  指摘は無し）。
  1. **[must fix] Rクロスチェックに`r_squared`/`r_squared_adj`が欠落**:
     `run_lm_crosscheck_benchmark.R`に`r.squared`/`adj.r.squared`
     （cov_type非依存、`summary(model)`から取得）・`t_stats`/`p_values`
     （`coeftest()`の列を`_common.R::extract_coef_se`が返すよう拡張）・
     `conf_int`（`coefs ± qt(0.975, df_inference) * ses`の手計算、
     confidence_level=0.95固定）を追加。`generate_ols_crosscheck_fixtures.py`/
     `generate_wls_crosscheck_fixtures.py`の`_normalize_names`を対応する
     フィールドを抽出するよう拡張し、`test_ols_crosscheck.py`/
     `test_wls_crosscheck.py`の`_assert_fit_stats_close`にアサーションを
     追加。実装中、HAC/autocorrelatedシナリオのみ`p_values`が浮動小数点
     アンダーフロー近傍（0に丸まる/1e-24等の極小値）で相対誤差比較が
     破綻することが判明し（実測最大絶対乖離1.69e-7）、IV/Logit/Probit
     クロスチェックの`atol_f_pvalue`/`atol_p_value`と同じ「p値のみ絶対誤差
     フロアで比較する」パターンを適用（`TOLERANCES["ols_crosscheck"/
     "wls_crosscheck"]["atol_p_value"] = 1e-6`）。
  2. **[should fix→A統合] Rクロスチェックに`conf_int`/`t_stats`/`p_values`が欠落**:
     上記1と同時に対応。
  3. **[should fix] WLS実データ（401ksubs）がclassicalのみ検証**:
     `_run_401ksubs_case()`を`cov_type`引数対応に拡張し、
     `WOOLDRIDGE_COV_TYPES = ["classical","hc0","hc1","hc2","hc3"]`
     （401ksubsはクロスセクションデータのためHACは対象外、OLSのwage1/gpa2と
     同じ方針）でstatsmodels/R双方をcov_type別に生成。
  4. **[should fix] `scale_variance`に成功パスが無い**: ユーザー確認の上、
     `scale_variance_mild`シナリオ（x1を1e2倍・x2を1e-1倍、既存の
     `scale_variance`のx1×1e6・x2×1e-3より緩いスケール差）を
     `generate_linear_datasets.py`に追加し全cov_typeで成功パスになることを
     確認、`freeze_linear_datasets.py`のSYNTHETIC_SCENARIOSと4つの
     `NUMERIC_SCENARIOS`（ols/wls×fixtures/crosscheck）に追加。既存の
     `test_matches_statsmodels`等のparametrizeされたテストがSCENARIOSを
     importしているため、シナリオ追加だけで既存テストの対象に自動的に
     含まれた（新規テスト関数の追加は不要）。
  5. **[should fix] WLS実データでのクラスターロバストSEが無い**:
     ユーザー確認の上、401ksubsに地域等の自然なカテゴリ列が無いため
     （marr/maleのような2値変数のみ）、`age`を分位点でビン化した
     `age_bin`列（8分位、`_add_age_bin`）を疑似的なクラスター列として使う
     方針を採用。`generate_wls_fixtures.py`/`generate_wls_crosscheck_fixtures.py`
     ・`test_wls_fixtures.py::test_401ksubs_cluster_matches_statsmodels`・
     `test_wls_crosscheck.py::test_401ksubs_cluster_matches_r`を追加。
  6. **[should fix] WLSにOLS相当のValidationErrorテストが欠落**:
     `test_insufficient_clusters_raises`/`test_invalid_confidence_level_raises`/
     `test_invalid_hac_lags_raises`/`test_missing_column_raises`/
     `test_null_values_raise`/`test_non_numeric_dtype_raises`を
     `test_wls.py`に追加（`weight`列自体の検証は既存の`test_missing_weight_
     column_raises`等が別途担当、`y`/`x`側の検証が抜けていた）。
  7. **[should fix] WLSにHACオプション配線確認テストが欠落**:
     `test_hac_auto_lags_runs_and_returns_finite_std_errors`（`hac_lags=None`
     自動計算）・`test_hac_time_col_reorders_rows_before_computing_lags`
     （`time_col`配線）を`test_wls.py`に追加（OLSの同名テストと同一データ、
     重み=1でOLSと同じ結果になることを利用）。
  8. **[should fix] `include_intercept=False`×ロバストcov_typeが参照実装と
     未比較**: ユーザー確認の上、statsmodelsとの直接比較のみ（Rクロスチェック
     は対象外、`run_lm_crosscheck_benchmark.R`に切片なしformula組み立て
     ロジックが無いため今回は追加しない）とし、OLSの既存テスト
     （`test_include_intercept_false_matches_statsmodels`、cov_type固定）は
     そのまま残し、新規に`test_include_intercept_false_matches_statsmodels_
     robust_cov_types`（HC0-3/cluster/HACをparametrize）を追加。WLSには
     同等のテストが元々存在しなかったため、`test_wls_fixtures.py`に
     `test_include_intercept_false_matches_statsmodels`（cov_type
     parametrize込み）を新規追加。
  9. **[nice to have] `cov_type`大文字小文字非依存性・`"nonrobust"`エイリアス
     未テスト**: `engine_pybind/src/linear/common.rs`に同ファイル初の
     `#[cfg(test)] mod tests`を追加（`parse_cov_type_is_case_insensitive`/
     `parse_cov_type_accepts_nonrobust_as_classical_alias`/
     `parse_cov_type_returns_validation_error_for_unknown_value`、
     `OLSOptions`をstruct literalで直接構築しGIL不要で実行）。Python側も
     `test_ols.py`/`test_wls.py`に`test_cov_type_is_case_insensitive`・
     `test_nonrobust_is_alias_for_classical`を追加。
  10. **[nice to have] WLSのy/x列自体のValidationErrorテスト不足**:
      上記6で統合対応。
  11. **[nice to have] WLSに`cov_type`ラベル・`confidence_level`反映テストが
      無い**: `test_cov_type_label`・`test_confidence_level_changes_
      interval_width`を`test_wls.py`に追加（OLSの同名テストと同じ観点）。
  12. **[nice to have] `TOLERANCES["wls_crosscheck"]`に`atol`キーが無く
      ハードコード**: `test_wls_crosscheck.py`の独自`_assert_close`/
      `_assert_scalar_close`（`tol = max(rtol*|ref|, 1e-8)`をハードコード）を
      `tests/_assertions.py`の共有関数へ統合し（`test_ols_crosscheck.py`と
      同じ計算式のため、フェーズ3.5の「除外事項」注記の解消も兼ねる）、
      `TOLERANCES["wls_crosscheck"]["atol"]`として一元化。
- **検証**: `cargo test -p engine`（317件）・`cargo test -p engine_pybind`
  （68件）・`uv run pytest tests`（752件、フェーズ3の670件から82件増）・
  `ruff check`/`ruff format --check`・`cargo fmt --check`・
  `cargo clippy --all-targets -- -D warnings`（engine・engine_pybind）
  全てグリーン。
- **`rust-reviewer`によるレビューと対応**: `engine_pybind/src/linear/
  common.rs`の`parse_cov_type`テスト追加について`rust-reviewer`にレビュー
  依頼し、must fix 1件を検出・修正した。
  - **[must fix→対応済み]** 成功系2テスト（`parse_cov_type_is_case_insensitive`/
    `parse_cov_type_accepts_nonrobust_as_classical_alias`）が
    `parse_cov_type(...).unwrap()`を使っていたが、`PyErr`の`Debug`実装
    （`unwrap()`失敗時のpanicメッセージ生成に使われる）はGIL取得を要求し、
    GIL未初期化のこのテスト環境ではErrの場合に二重パニック（テストバイナリ
    全体がSIGABRTでクラッシュし、他のテスト結果も失われる）を起こす
    バグだった。レビュー中に実際に`"nonrobust"`分岐を一時的に壊して
    `thread panicked while processing panic. aborting.`の再現を確認済み。
    `.unwrap()`を`let-else`（Err値のDebug/Displayに触れない）に置き換えて
    修正し、同じ再現手順でPASS/FAILが正しく報告されること（アボートしない
    こと）を確認した。ついでに`cluster`/`hac`自体の大文字小文字混在ケース
    （`"CLUSTER"`/`"Hac"`、指摘のnice to have）も
    `parse_cov_type_is_case_insensitive`に追加した。
- **未対応・今後**: Logit/Probitの`review-testing`レビュー、IV分
  （#232〜238）のテスト拡充は本フェーズの別ラウンドとして未着手（→nonlinear系統・
  IV系統とも下記の通り完了、フェーズ4全体が完了）。

**メモ（nonlinear系統〔Logit/Probit〕実施結果）**:

- **ステップ0（前提確認）**: `benchmark/nonlinear/freeze_nonlinear_datasets.py`で
  CSVを、`generate_logit_fixtures.py`/`generate_probit_fixtures.py`
  （statsmodels）で`logit.json`/`probit.json`を再生成したところ、CSVは完全一致
  したが、statsmodels側は`_meta.note`のファイル名参照
  （`generate_logit_datasets.py`→`generate_nonlinear_datasets.py`、フェーズ2の
  リネームに追従していなかった）と`_meta.model`フィールド（`logit.json`のみ
  未追随だった）の差分のみで数値は完全一致と確認。
  Rクロスチェック側（`generate_logit_crosscheck_fixtures.py`）の再生成時に、
  フェーズ2で「今回のリファクタリングと無関係の既存バグ」と暫定判断していた
  `near_separation`シナリオ×logitの生成失敗を実際に調査した。
  - **原因判明**: `run_glm_crosscheck_benchmark.R`の不変条件チェック
    （logitでは期待情報行列と観測情報行列が理論上一致するはず、という
    `stopifnot(...tolerance=1e-6)`）が、`near_separation`シナリオ
    （500件中161件のfitted probabilityが浮動小数点上ちょうど0/1に潰れる
    強い準分離ケース）でFrobeniusノルム相対誤差約1.7e-4となり失敗していた。
    調査の結果、これは計算式のバグではなく浮動小数点精度の限界（IRLS内部の
    期待情報行列と直接計算する観測情報行列が数値的に微小にずれる）と判明。
    さらに、コミット済み`logit_crosscheck.json`の`near_separation`は
    Probit対応時の観測情報行列（`observed_bread`）採用（2026-08-02）より
    **前**（2026-08-01）に生成されたものであり、この不変条件チェックが
    実際には「今の計算式（`bread_obs`）に対して過去の陳腐化したフィクスチャが
    追従できていない」ことを正しく検知していたと確認できた（コミットされた
    SEの値が新計算式`bread_obs`ではなく旧計算式`bread(model)`と一致することを
    実測確認）。
  - **対応（ユーザー確認済み）**: `stopifnot`のtoleranceを実測値
    （1.7e-4）に対し約6倍のマージンを持つ1e-3に緩め、理由をコード内コメントに
    明記した上で、`near_separation`（logit）のclassical/opg/hc0/hc1を現在の
    `bread_obs`式で再生成した。`test_logit_crosscheck.py`の`rtol=2e-4`が
    この差分を十分にカバーしており、既存テストは新規失敗なくパス。
    probit側の`near_separation`はこの不変条件チェック自体が`link=="logit"`
    限定で対象外のため、再生成しても数値は完全一致（`r_version`のメタ情報のみ
    4.2.2→4.5.3に更新、OLS/WLSフェーズ4で見た同種の環境差と同じで実測値には
    影響なし）。
- **ステップ1（`review-testing`スキルでのレビュー、nonlinear系統分）**:
  `testing-completeness-reviewer`に`engine/src/nonlinear/`・
  `engine_pybind/src/nonlinear/`・`python_package/econometricsmodels/nonlinear/`・
  `tests/test_logit*.py`・`tests/test_probit*.py`・`benchmark/nonlinear/`の
  レビューを依頼し、must fix 0件・should fix 5件・nice to have 8件、計13件の
  指摘を得た。ユーザー確認の上、should fix全5件は対応、nice to haveは
  8件中4件を選んで対応（残り4件は`docs/planning/specs/test-coverage-candidates.md`
  へ記録し今回は見送り）。
  1. **[should fix] `method="bfgs"/"lbfgs"`が主リファレンスに対しフルの統計量
     （std_errors含む）で照合されていない**: `run_statsmodels_benchmark.py`に
     `--method`引数を追加（statsmodelsの`fit(method=...)`にそのまま渡せる、
     `"bfgs"`/`"lbfgs"`の文字列がRust側`Method`のバリアント名と一致するため
     変換不要）。`generate_logit_fixtures.py`/`generate_probit_fixtures.py`に
     `fixtures["method"]`（baselineシナリオ・classical cov_typeの1ケースのみ、
     bfgs/lbfgs）を追加し、`test_logit_fixtures.py`/`test_probit_fixtures.py`に
     `test_method_matches_statsmodels`を追加。bfgs（statsmodelsの`mle_retvals`に
     `"iterations"`キーが無い、`fmin_bfgs`の戻り値の都合）は`n_iter=None`で
     許容するようフィールド抽出側を修正。method間の実測最大相対誤差
     （~7.7e-5、係数・SEとも）に対し約13倍のマージンを持つ
     `TOLERANCES["logit_fixtures"/"probit_fixtures"]["rtol_method"]=1e-3`を新設。
  2. **[should fix] `method=bfgs/lbfgs`×完全多重共線性の`ComputationError`が
     API境界で未検証**: `engine`側には既存の回帰テスト
     （`fit_returns_singular_hessian_error_for_perfectly_collinear_design_matrix_
     with_bfgs_and_lbfgs`、過去にbfgsのみ検出漏れし桁違いに巨大なSEを含む`Ok`が
     返る実バグがあった経緯）があったが、`method`の文字列パース〜
     `engine_pybind`配線を経由するAPI境界の確認が無かった。
     `test_logit.py`/`test_probit.py`の`test_singular_hessian_raises_
     computation_error`を`method`でparametrize（newton/bfgs/lbfgs）して対応。
  3. **[should fix] `fit()`本体の`confidence_level`範囲外`ValidationError`が
     未検証**: `marginal_effects(confidence_level=1.5)`側は既存だったが
     `LogitOptions`/`ProbitOptions`側が無かった（OLS/WLSとの非対称）。
     `test_invalid_confidence_level_raises`を追加。
  4. **[should fix] 欠損値・非数値dtypeのValidationErrorがPython API境界で
     未検証**: `test_null_values_raise`/`test_non_numeric_dtype_raises`を
     `test_logit.py`/`test_probit.py`に追加（OLSと同型）。
  5. **[should fix] `include_intercept=False`の成功パスが構造テスト・数値
     照合テストとも一切未検証**: `df_model`が`include_intercept`の値に関わらず
     常に`k-1`になる・`log_likelihood_null`が常に切片のみモデルを参照する
     という特殊挙動が`engine`側の単体テストのみで、statsmodelsとの数値一致が
     未確認だった。構造テスト（`test_include_intercept_false_omits_const_and_
     converges`）を`test_logit.py`/`test_probit.py`に、数値照合テスト
     （`test_include_intercept_false_matches_statsmodels`、classical/opg/hc0を
     parametrize、`run_statsmodels_benchmark.run()`を`formula="y ~ ... - 1"`で
     直接呼ぶ方式）を`test_logit_fixtures.py`/`test_probit_fixtures.py`に追加。
  6. **[nice to have] `cov_type`大文字小文字非依存性・`"nonrobust"`エイリアスが
     未テスト**: `engine_pybind/src/nonlinear/{logit,probit}.rs`の`mod tests`に
     `build_{logit,probit}_input_cov_type_is_case_insensitive`/
     `_accepts_nonrobust_as_classical_alias`を追加。linear系統と異なり
     nonlinear系統の`parse_cov_type`は「呼び出し元が`to_lowercase()`済みの
     文字列を渡す」設計のため、`parse_cov_type`単体ではなく実際にPythonから
     渡された文字列を受ける`build_{logit,probit}_input`をテスト対象にした
     （`parse_cov_type`単体だと「小文字を渡せば小文字のまま通る」という
     トートロジーになるため）。Python側も`test_logit.py`/`test_probit.py`に
     同名テストを追加。
  7. **[nice to have] `cov_type`ラベル反映テスト欠如**: `test_cov_type_label`を
     `test_logit.py`/`test_probit.py`に追加（OLSと同型）。
  8. **[nice to have] `confidence_level`の信頼区間幅反映テスト欠如**:
     `test_confidence_level_changes_interval_width`を追加（OLSと同型）。
  9. **[nice to have] `max_iter<=0`のAPI境界テスト欠如**: `tol<=0`側は既存
     だったが対応する`max_iter`側が無かった。
     `test_non_positive_max_iter_raises`を追加。
  - **見送った4件**（`docs/planning/specs/test-coverage-candidates.md`
    3〜5番に記録）: n=k+1境界値で「ほぼ確実に完全分離する」という主張自体の
    未検証、`raise_on_non_convergence=False`がclassical cov_typeでしか
    未検証、`cov_type="cluster"`×`cluster_col`未指定のAPI境界未検証
    （OLS側にも同種の欠落がある既存パターン）。あわせて、実装当時から
    `docs/spec/logit-spec.md`/`probit-spec.md`4章に記載されていた
    `SEPARATION_PARAM_NORM_THRESHOLD`の多変量モデルでの誤検知リスク等の
    既知の未検証事項も、同ドキュメントの6〜10番として転記・集約した。
- **検証**: `cargo test -p engine`（317件）・`cargo test -p engine_pybind`
  （72件、should/nice to have対応前の68件から+4）・`uv run pytest tests`
  （810件、linear系統フェーズ4完了時点の752件から58件増）・
  `ruff check`/`ruff format --check`・`cargo fmt --check`・
  `cargo clippy --all-targets -- -D warnings`（engine・engine_pybind）
  全てグリーン。
- **`rust-reviewer`によるレビューと対応**: `engine_pybind/src/nonlinear/
  {logit,probit}.rs`への`mod tests`追加（項目6）について`rust-reviewer`に
  レビュー依頼し、should fix 1件・nice to have 1件を検出・対応した。
  - **[should fix→対応済み]** 今回追加した4テスト自体は`let-else`パターンで
    安全だったが、レビュー中に**同じ`mod tests`内の既存テスト**
    （`build_{logit,probit}_input_succeeds_for_well_formed_data`等、
    logit.rs 4箇所・probit.rs 4箇所の計8箇所）が`build_{logit,probit}_input
    (...).unwrap()`という、`linear/common.rs`で過去に修正したのと同種の
    GIL/SIGABRTリスクパターンを抱えたまま残っていることが発覚した
    （pyo3のソースで`PyErr`の`Debug`実装が`Python::attach`を要求することを
    確認済み）。今回のスコープ外の既存コードだったが、同じファイル・同じ
    `mod tests`を触っている機会のため、8箇所全てを`let-else`パターンに
    置き換えて修正した。
  - **[nice to have→対応済み]** 新規追加した`cov_type_is_case_insensitive`
    テストが、`Ok`であることのみを確認し実際に正しい`EngineCovType`
    バリアントにマッピングされているかを確認していなかった（誤った
    match分岐でも`Ok`である限り通ってしまう）指摘を受け、各入力に対応する
    期待バリアントを`matches!`で突き合わせる形に強化した。

**メモ（IV系統〔2SLS/GMM〕実施結果、Issue #232〜238）**:

- **ステップ0（前提確認）**: `benchmark/iv/freeze_iv_datasets.py`でCSVを、
  `generate_iv_fixtures.py`（linearmodels 2SLS）・`generate_iv_gmm_fixtures.py`
  （linearmodels GMM）で`iv.json`/`iv_gmm.json`を再生成したところ、数値・
  タイムスタンプ以外は完全一致（タイムスタンプのみの差分は`git checkout --`で
  戻した）。Rクロスチェック（`generate_iv_crosscheck_fixtures.py`）も同様に
  完全一致を確認。
- **ステップ1（`review-testing`スキルでのレビュー、IV系統分）**:
  `testing-completeness-reviewer`に`engine/src/iv/`・`engine_pybind/src/iv/`・
  `python_package/econometricsmodels/iv/`・`tests/test_iv*.py`・
  `benchmark/iv/`のレビューを依頼し、must fix 1件・should fix 5件・
  nice to have 3件、計9件の指摘を得た。ユーザー確認の上、must fix・should fix
  全6件、nice to have全3件を対応した（未対応の指摘は無し）。
  1. **[must fix] 複数内生変数（`k_endog>=2`）がPython API境界・フィクスチャ・
     linearmodels/Rクロスチェックのいずれにも一切存在しない**: 対応の過程で
     `generate_iv_datasets.py`のDGP自体の見落としを発見した。第一段階誤差`v`が
     全内生変数に単一列としてブロードキャストされており、`k_endog=2`だと
     第一段階回帰残差が事実上完全共線（相関~0.99999999999998を実測）になり、
     Wu-Hausman検定の拡張回帰が推定不能（`wu_hausman_statistic=None`）になる
     ことが判明した。ユーザー確認の上、`v`を内生変数ごとに独立な誤差
     （構造誤差`u`とはそれぞれ相関`_RHO_ENDOG`、`v_i`・`v_j`間は無相関）の
     `(n, k_endog)`行列に一般化する修正を実施（`k_endog=1`では従来と数学的に
     完全に一致する一般化のため、既存シナリオへの影響はゼロ、再生成CSVが
     バイト単位で一致することを確認済み）。`freeze_iv_datasets.py`に
     `iv_baseline_multi_endog.csv`（`k_endog=2, k_instruments=3`、過剰識別）を
     追加し、`generate_iv_fixtures.py`/`generate_iv_crosscheck_fixtures.py`/
     `generate_iv_gmm_fixtures.py`の3スクリプト全てにフィクスチャを追加、
     `test_iv_fixtures.py`/`test_iv_crosscheck.py`/`test_iv_gmm_fixtures.py`に
     数値照合テストを追加。Rクロスチェック側は`weak_instrument_f`が単一の
     内生変数を前提にスカラーで返す設計だったため、内生変数名をキーにした
     dict形式（本実装の`weak_instrument_f_statistics`と同じ形）に一般化した
     （`run_ivreg_benchmark.R`、既存の全synthetic/wooldridgeエントリにも遡及
     適用、`test_iv_crosscheck.py`の該当箇所も合わせて修正）。
  2. **[should fix] `weight_type="kernel"`×`cov_type="hac"`の組み合わせが
     構造テストにも一度も現れない**: `generate_iv_gmm_fixtures.py`に
     `kernel_hac`フィクスチャを追加、`test_iv_gmm_fixtures.py`に
     `test_kernel_hac_matches_linearmodels`を追加。
  3. **[should fix] `gmm_iterations`が1（1-step）・3以上（iterated）の成功パスが
     フィクスチャで数値照合されていない**: `generate_iv_gmm_fixtures.py`に
     `gmm_iterations`フィクスチャ（1・3）を追加。実装中、`weight_type=
     "unadjusted"`では係数が反復回数に依らず不変（重み`S=Z'Z`が残差に依存
     しないため）にも関わらず、linearmodelsの`iter_limit=1`のHansen J統計量
     （0.30086708530935663）が`iter_limit=2/3`の値（0.32832429087644643、
     2SLSのSarganと機械精度一致）と異なることを発見した。本実装は
     `gmm_iterations=1`でも一貫して後者を返す（`gmm.rs`のσ̂²・Z'Zスケーリング
     設計通り、内部的に自己無矛盾）。linearmodels側の`iter_limit=1`固有の
     計算式の違いが原因と考えられるが未特定のため、ユーザー確認の上
     `gmm_iterations=1`のときのみHansen J（`overid_statistic`/
     `overid_p_value`）の比較を除外する（係数・SE等の他統計量は引き続き
     比較する）方針とした（`test_iv_gmm_fixtures.py`の`_check_result`に
     `check_overid`引数を追加）。
  4. **[should fix] 「分散が大きい」データセットシナリオがIVに存在しない**:
     `generate_iv_datasets.py`に`high_variance`シナリオ（構造誤差`u`の分散を
     100倍、OLSの`high_variance`と同じ標準偏差10相当）を追加。既存シナリオへの
     影響はゼロ（`u_var=1.0`固定で数学的に同一）。追加時、Rクロスチェックの
     `high_variance`×`hac`の`f_p_value`が既存の`RTOL_HAC`（1%）をわずかに
     超える乖離（実測相対誤差2.37%）を示すことが判明したが、`f_statistic`
     自体は0.6%程度しか違わずF分布の裾で増幅されるだけの既知のパターン
     （`small_n`×`hac`と同種）と判断し、`RTOL_HAC_SMALL_N`（10%）を
     `high_variance`にも適用する形で解消した。
  5. **[should fix] 実データセットでの検証が皆無**: Wooldridge `card`
     （Card 1995、大学近接ダミー`nearc2`/`nearc4`を操作変数とする教育年数
     `educ`の内生性補正、教科書的定番例）を追加。`run_linearmodels_benchmark.py`
     の`run()`は元々合成データ専用（`_load_iv_dataset`が凍結CSV固定読み込み）
     だったため、OLS/Logit側の`dataset_source`パターンに倣い
     `dataset_source`/`y_col`引数を追加して一般化した（IVの合成データは
     常に`y`列固定のため、実データ対応には`y_col`パラメータ化が必要だった）。
     `generate_iv_fixtures.py`/`generate_iv_crosscheck_fixtures.py`双方に
     `card`フィクスチャを追加、`test_iv_fixtures.py`/`test_iv_crosscheck.py`に
     数値照合テストを追加（`cluster` cov_typeは対応する自然なカテゴリ列が
     無いため対象外）。
  6. **[should fix] `cov_type`/`weight_type`の大文字小文字非依存性のテストが
     無い**: `engine_pybind/src/iv/common.rs`の`parse_iv_cov_type`/
     `parse_weight_type`は既に`.to_lowercase()`・エイリアス
     （`nonrobust`↔`classical`、`homoskedastic`↔`unadjusted`、
     `heteroskedastic`↔`robust`）に対応済みだったが、Python API境界での
     確認テストが無かった。`test_iv.py`に`test_cov_type_is_case_insensitive`・
     `test_nonrobust_is_alias_for_classical`・
     `test_weight_type_is_case_insensitive_and_aliased`を追加（`IvResults`は
     `cov_type`ラベルは公開するが`weight_type`ラベルは公開しないため、後者は
     正準表記との`params`一致で確認）。
  7. **[nice to have] G=2クラスター境界の成功パスが構造確認のみで数値
     クロスチェックが無い（コメントとコードの乖離あり）**: `generate_iv_
     fixtures.py`/`generate_iv_crosscheck_fixtures.py`双方のコメントが
     「IVの第一段階回帰でComputationErrorが再現するため原因究明後に追加予定」
     という古い記述のままだったが、`engine/src/iv/CLAUDE.md`の記録によれば
     原因（`k_constant`取り違えバグ）は既に特定・修正済みと判明。両スクリプトに
     `cluster_g2`フィクスチャを追加し、`test_iv_fixtures.py`/
     `test_iv_crosscheck.py`に数値照合テストを追加、古いコメントも修正した。
  8. **[nice to have] 存在しない`cluster_col`列名の`ValidationError`パス未検証
     （OLS側にも同種の欠落があるとの指摘）**: 実際に確認したところ、5手法
     全て（OLS/WLS/Logit/Probit/IV）で`column_extraction`の既存の汎用チェック
     経由で正しく`ValidationError`を返すことを確認済み（挙動自体は正しい、
     テストが無いだけ）。ユーザー指示で対象をIVだけでなくOLS/WLS/Logit/Probit
     にも広げ、`test_cluster_col_nonexistent_column_raises`を5ファイル全てに
     追加した。
  9. **[nice to have] `x_exog`/`x_endog`/`instruments`列の欠損値・非数値dtype
     バリデーションが`y`列のみで未検証**: `test_iv.py`の`test_null_values_
     raise`/`test_non_numeric_dtype_raises`を`bad_col`（`y`/`x1`/`endog1`/
     `z1`）でparametrizeする形に拡張した。
- **検証**: `cargo test -p engine`（317件）・`cargo test -p engine_pybind`
  （72件、IV分の変更はテストコード・ベンチマークスクリプトのみで
  engine/engine_pybindには変更なし）・`uv run pytest tests`（878件、
  nonlinear系統フェーズ4完了時点の810件から68件増）・`ruff check`/
  `ruff format --check`・`cargo fmt --check`・
  `cargo clippy --all-targets -- -D warnings`（engine・engine_pybind）
  全てグリーン。今回はengine/engine_pybind/python_packageへの変更が無かった
  ため`rust-reviewer`/`python-reviewer`の呼び出しは対象外。

**追加ラウンド: GitHub Issue #232〜238の実施状況確認・残件対応（完了）**:

- **経緯**: 上記メモ完了後、ユーザーから「Issue #232〜238はすでに実施済みか」と
  確認依頼があった。実際にコードを確認したところ、上記の`testing-completeness-
  reviewer`レビュー（直近の`git diff`ベースで対象推定）とこれら7件のIssue本文が
  1対1対応しておらず、**完了していたのは#234（high_variance）・#236（Wooldridge
  card）・#238（複数内生変数）の3件のみ**（クローズ済み）で、
  #232・#233・#235・#237は未着手と判明した。ユーザー確認の上、残り4件も
  引き続き実装した。
- **#235（自由度1境界df=1シナリオ追加）**: `x_exog=[]`・`x_endog=['endog1']`・
  `instruments=['z1']`（丁度識別、n=3）の最小構成で追加
  （`freeze_iv_datasets.py`の`IV_BOUNDARY_DF1_SCENARIOS`）。実装中に2件の
  副次的なバグを発見・修正した。
  1. `run_linearmodels_benchmark.py`の`_nested_f_test`が`x_exog_cols=[]`の
     とき制限モデルのSSRを非中心化（`(y**2).sum()`）で計算していたが、
     本実装は常に切片を含む（`include_intercept=false`の退化ケース専用の式を
     誤って全`x_exog=[]`ケースに適用していた）。df1追加で顕在化（本実装の
     `weak_instrument_f_statistics`と2.6958 vs 11.607で不一致）、中心化SSRに
     修正して解消。この修正で`cluster_g2`フィクスチャの
     `weak_instrument_f_independent`も無言で誤っていたことが判明したが、
     この値を比較するテストはこれまで無かったため既存テストへの影響はゼロ。
  2. augmented regressionがsaturated（残差自由度0）になる境界のため、
     linearmodelsの`wooldridge_regression`が`ZeroDivisionError`、ivregの
     HC0診断が`solve()`エラーで落ちた。本実装が同じ状況で
     `wu_hausman_statistic`/`wu_hausman_p_value`を`None`にする設計
     （`engine/src/iv/CLAUDE.md`）と揃え、両ベンチマークスクリプト側も
     同じ状況を検出して`None`/`NA`を返すよう修正（ユーザー確認不要、
     本実装の既存設計への追従のみ）。
- **#232（Rクロスチェックにt値・p値・信頼区間を追加）・#237（nobs/df_resid
  追加）**: `run_ivreg_benchmark.R`に`extract_coef_se`の`t_stats`/`p_values`と
  手計算信頼区間、`nrow(df)`/`df.residual(model)`を追加（OLS/WLSクロスチェック
  の`run_lm_crosscheck_benchmark.R`と同じパターン）。`nobs`/`df_resid`は
  cov_type非依存の構造残差自由度（n-k）を返す必要があり、cluster時に
  `df_inference`をG-1へ上書きする既存変数をそのまま使うと誤った値になる
  （linearmodelsの`df_resid`もcluster時n-k固定であることを`iv.json`で確認）
  ためcluster分岐と独立に`df.residual(model)`を保持するよう実装した。
- **#233（wu_hausmanのcov_type=hacケースを外部リファレンスで検証）**:
  ユーザーは当初「R側でHAC augmented regressionを手動実装する」案（Issue本文の
  選択肢2）を、手動実装自体の正しさが別途検証されない点を理由に却下し、
  「別パッケージで参照できないか」を確認するよう指示。`ivreg:::ivdiag`の
  ソースを確認したところ、`summary.ivreg`の`vcov.`引数は**関数**として渡せば
  診断表（Wu-Hausman・弱操作変数F統計量）にも反映される仕様だったが、
  既存コードは**行列**を渡して警告付きでNULLにフォールバックしていたことが
  判明（コメント「vcov.に行列を渡すと警告付きでNULLにフォールバックする」は
  事実だが、関数として渡す代替を見落としていた）。手動実装は不要で、ivreg
  自身の診断機構をcov_type別のvcov関数（`vcov_fn`）で呼び分けるだけで全
  cov_typeのwu_hausmanクロスチェックが可能と判明（弱操作変数F統計量・Sargan
  は常にclassical固定を維持するため、`vcov.`無しのデフォルト呼び出しと
  使い分けた）。ただし**cluster cov_typeのみ**、`ivdiag`内の`wald()`が
  F分布の分母自由度に常に`obj1$df.residual`（n-k）を使い、本実装のG-1
  （標準的な慣行）に追従しない既知の制約を発見。統計量は高精度で一致する
  （実測: 112.32で完全一致）がp値は一致しない（R側8.5e-24 vs 本実装2.2e-06、
  df=9で計算すると本実装の値が再現できることを確認）ため、ユーザー確認の上
  cluster cov_typeのみp値をクロスチェック対象から除外した
  （`gmm_iterations=1`のHansen J除外と同型のパターン）。
  さらにcluster_g2（G=2境界）ケースでは、augmented regressionの傾き係数が
  q=2（endog1・第一段階残差）に対しG-1=1<qとなり構造的にクラスタロバスト
  共分散が特異になる（本実装のG≤qの罠と同じ原理、`engine/src/iv/CLAUDE.md`
  参照）ため本実装は正しく`None`を返すが、ivreg側はこの特異性を検出せず
  値を返すため、この1ケースのみwu_hausman比較自体を対象外にした。
  linearmodels側の`hac`不一致原因調査自体は次セッション送りとした
  （ユーザー確認済み）。
- **HAC許容誤差の拡張**: #232/#233で追加した新規フィールド（t_stats/p_values/
  conf_int/wu_hausman）は、既存のcoef/se向けの`RTOL_HAC`/`RTOL_HAC_SMALL_N`
  では収まらない乖離（p_values最大28.7%、conf_int最大1.66%、
  wu_hausman_statistic最大11.1%（small_n）等）を示した。実測値に基づき
  専用の許容誤差を追加（`_tolerances.py`の`atol_hac_pvalue`・
  `atol_hac_conf_int`・`rtol_hac_wu_hausman`・
  `rtol_hac_wu_hausman_small_n`、いずれもhacケースでのみ適用しclassical/
  hc0/hc1/clusterの厳密比較には影響しない）。df1×hac（n=3でのNewey-West
  ラグ選択が統計的にほぼ無意味、実測se最大42%乖離）はユーザー確認の上
  crosscheck対象外にした。
- **検証**: `uv run pytest tests`（885件、上記完了時点の878件から7件増）・
  `ruff check`/`ruff format --check`・`cargo test -p engine -p engine_pybind`
  （317+72件、変更なし）全てグリーン。engine/engine_pybind/python_packageへの
  変更は無かったため`rust-reviewer`/`python-reviewer`の呼び出しは対象外。
  完了条件を満たしたことを確認の上、Issue #232・#233・#235・#237をクローズ
  （#234・#236・#238は前段の完了時点で既にクローズ済み）。

---

## フェーズ5: `python_package/`のリファクタリング

**想定範囲**: Issue番号への言及の削除が主。タスクとしては軽量な見込み。

**状態**: [#248](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/248)に引き継ぎ（未着手のまま#231をクローズ）

**メモ**: (着手後に記載、以降は#248側で記録)

---

## フェーズ6: `engine_pybind/`のリファクタリング

**想定範囲**: 重複ロジックの共通化、Issue番号への言及の削除、コード整理。

**気になっている個所（Issue #231より）**:
- `engine_pybind`側の`cov_type`パース等、既にA2（[#153](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/153)）で
  共通化済みの範囲との重複有無を再確認する

**状態**: [#248](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/248)に引き継ぎ（未着手のまま#231をクローズ）

**メモ**: (着手後に記載、以降は#248側で記録)

---

## フェーズ7: `engine/`のリファクタリング

**注意**: ロジックを壊さないこと・無理な共通化をしないことを優先する、
最も慎重な設計判断が必要な領域。フェーズ4で拡充したテストで担保する。

**気になっている個所（Issue #231より）**:
- `engine/src/linear/ols.rs`の`from_columns`と同様の実装が各手法のメインロジックに
  あるが、共通化可能か（IV着手時のA章と同様、共通化が呼び出し箇所の性質上
  適さない場合は無理に統合しない）

**状態**: [#248](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/248)に引き継ぎ（未着手のまま#231をクローズ）

**メモ**: (着手後に記載、以降は#248側で記録)

---

## 未解決の確認事項

（フェーズ着手時に判断が分かれる点が出た場合はここに追記し、ユーザー確認後に
解消済みとして記録する）

- なし（フェーズ1着手前時点）
