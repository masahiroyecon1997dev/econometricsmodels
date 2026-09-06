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

**注記（2026-09-05、圧縮）**: 本ドキュメントが2282行まで肥大化したため、
完了済みで注記の必要性が薄い部分（検証コマンドの実行結果・byte単位一致確認・
移行手順の逐次記録等、`git log`や現在のコード自体から再現可能な情報）を削除・
要約した。外部ファイルから名指しで参照されている項目（Initiative A節、
項目44・50・58・63、フェーズ3実施結果の項目8）は参照が壊れないよう残してある。

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
`docs/planning/specs/refactoring-candidates*.md`に溜まった個別項目を`refactor`
スキルでその都度1件ずつ対応する運用も並行して行っている。対応済み項目は
候補メモ本体から削除する運用（2026-08-22確立）のため、削除後はこの
スナップショットが唯一の記録になる。

**運用ルール**:
- 候補メモの項目を1件（または明示的に指定された小さいまとまり）ずつ対応する。
  まとめて全部やらない。
- 実装 → 検証（挙動を変えないリファクタリングでは可能な限り出力の完全一致を
  実測確認） → `/code-review` → 候補メモの「状態」更新 → **ユーザーのコミット前
  確認を得てから**コミット、の順で1件ずつ完了させる。
- 設計判断が分かれる点（定数の置き場所等）は着手前にユーザーに確認する
  （CLAUDE.md 14章）。

**対応済み項目**（詳細な変更内容・検証手順は各コミットの差分・メッセージ参照）:

- 項目3: `sys.path.insert`→PYTHONPATH方式に変更（`48727c9`）。後にInitiative Aで
  `pyproject.toml`の`pythonpath` ini設定に置き換わり、この方式自体は不要になった。
- 項目4: `SCENARIOS`重複解消（`d23d9b7`）。
- 項目9・10: DGP用マジックナンバーを`_dgp_constants.py`へ集約（`7d51802`）。
- 項目36: crosscheck側の`pl.read_csv`再実装を`load_frozen_dataset`に統一（`8f6f98e`）。
- 項目47: `wooldridge_loader`の`sys.path.insert`重複呼び出しを削除（`835da85`）。
- 項目42・48: `tests/_assertions.py`/`_helpers.py`の陳腐化したdocstring記述
  （crosscheck対象外という古い説明・古い箇所数）を現状に合わせて修正（`10ef78e`）。
- 項目1: `benchmark/load_wooldridge.py`の未使用`SUGGESTED_DATASETS`を削除し、
  実データセットの選定理由を各`docs/spec/<手法>-spec.md`側に明記する運用に変更
  （`c0b78bf`）。
- 項目2: `_require_min_k`ヘルパーで`k`下限チェックの重複（4箇所）を解消（`769eee8`）。
- 項目5: `validate_choice`ヘルパーで`scenario`/`link`検証の重複（4箇所）を解消
  （`8c71613`）。
- 項目6: `linear_predictor`ヘルパーで線形予測子の組み立てを統一（`871f59b`）。
- 項目7: `correlated_design_matrix`/`apply_perfect_multicollinearity`で設計行列
  生成ロジックを3系統で統一（`scale_variance`系はタイミングが意図的に異なるため
  対象外と確認した上で除外、`35bcaa7`）。
- 項目8: Logit/Probitの後方互換ラッパー関数を削除し直接呼び出しに統一（`a48a2e5`）。
- 項目14: `COV_TYPES`への`cluster`混入をLogit/Probit/IV(2SLS)でOLS/WLS/GMMと同じ
  書き方に統一（`e99c13f`）。検証中に`benchmark/linear`/`benchmark/nonlinear`の
  同名`run_statsmodels_benchmark.py`がPYTHONPATH経由のimportで衝突する別バグを
  発見し、項目71として直後に緊急対応した。
- 項目71: 上記の同名ファイル衝突を、3ファイルとも系統サフィックス付きの名前へ
  `git mv`でリネームして解消（`a41dc38`）。
- 項目11: `extract_coef_se`ヘルパーを新設し、OLSの生成スクリプトと参照実装の
  重複した`coef`/`se`抽出コードを統合。
- 項目17: `benchmark/`配下9スクリプトのコメント・docstring・`_meta.note`から
  Issue番号（#231/#232/#233/#235/#237）への言及を除去（WHY自体の説明は保持）。
- 項目30: `run_glm_crosscheck.R`で2回重複していた列スケーリング反転ロジックを
  ローカル関数`scaled_gram_inverse`に集約。
- 項目40: `compare_performance.py`のOLS専用構造は、Issue #250〜254で
  `performance/_perf_harness.py`（手法非依存の骨格）＋`performance/compare_<method>.py`
  （手法別アダプタ）へ既に一般化済みと確認。
- 項目41: ピークRSS表示桁の不統一（進捗ログ`.1f`・サマリー`.0f`）を`.1f`に統一。
- 項目43: `tests/`ディレクトリ分割の要否は、より具体的な項目68に論点が一本化された
  ため項目68側へ統合。
- **項目68**（`tests/`を系統別ディレクトリ×関心事4分割 `_api`/`_validation`/
  `_reference`/`_crosscheck`に再編）: Phase 1（ディレクトリの系統別分割）→
  Phase 2 linear（OLS/WLS）→ Phase 2 nonlinear（Logit/Probit）→ Phase 2 iv
  （2SLS/GMM）の順で全系統完了。`_tolerances.py`の`*_fixtures`キーは全6系統
  `*_reference`に統一（項目55も同時解消）、セクション見出しも全系統で統一
  （項目76も同時解消）。各Phaseとも`pytest tests`は不変件数（移設のみ、
  カバレッジ減ゼロ）でパス。
- 項目53: OLSの独自絶対誤差定数（`ATOL_COEF`/`ATOL_SE`/`ATOL_STAT`等）を全廃し、
  `tests/_assertions.py`の`assert_close`系＋`tests/_tolerances.py`の
  `"ols_reference"`（凍結フィクスチャ照合と同一のrtol/atol）に統一。
- 項目54: 「完全な多重共線性→`ComputationError`」の手書き極小df版とCSV固定版の
  重複を整理。OLS・IVは追加検証価値が無いため手書き版を削除しCSV版へ一本化。
  Logit/Probitの手書き版は`method`×3 parametrize（過去の`bfgs`検出漏れバグの
  回帰テスト）のため維持し、Issue #279（Tobit方式のmethod共通QR検証を
  Logit/Probitに適用する提案）完了後に一本化する方針。
- 項目58: HAC自動ラグ数の固定値（`maxlags=1`）が生成側・消費側4ファイルに
  独立複製されていたのを単一の定義元`HAC_MAXLAGS`に統一。後日（2026-09-05）
  ユーザー指摘により、定義元を参照値生成スクリプトから独立した
  `benchmark/linear/constants.py`へ再移設し、`_meta.hac_maxlags`の記録
  （一度デッドコード扱いで削除したが、IV側`_meta.hac_lag`との前例整合・
  人間可読性目的だったことが判明し復元）も合わせて対応した。
- 項目50: `sys.path.insert`除去がdevcontainerのみでCI・`tests/`は据え置き、
  という項目の前提を確認したところ、`pyproject.toml`の`pythonpath` ini設定
  （Initiative Aで導入）により既に一律解消済みと判明（`sys.path.insert`は
  リポジトリ全体でgrep 0件）。コード変更は無し。
- 項目44・63: Intercept→const正規化のタイミングをRクロスチェック側（生成時）と
  statsmodels主リファレンス側（テスト実行時）で統一する方針で対応。
  `benchmark/common/reference/r.py`の`normalize_names`を系統横断の
  `benchmark/common/reference/normalize.py`へ移設し、`statsmodels_ref.py`
  （linear/nonlinear）の生成時にも適用。`tests/_assertions.py`の`rename`引数は
  ユーザー方針（「基本benchmarkに寄せる、デフォルト値は拡張ポイントとして残す」）
  により削除せず維持（`normalize.py`の`intercept_aliases`引数と同じ理由）。
- 項目45: `tests/_helpers.py`の`separation_suspected_dataset`を標準ライブラリ
  `random`から`benchmark/`側DGPと同じ`numpy`（`np.random.default_rng`）ベースの
  ベクトル化演算に統一（数値比較をしないテスト専用データセットのため乱数値の
  変化は無害）。

**Issue化した項目**（バグ調査に近く候補メモの範囲外と判断し、個別Issueへ切り出し）:

- 項目44: engine（faer/rayon）のマルチスレッド線形代数が多コア機・負荷下で
  シングルスレッド比20倍以上遅くなり不安定になる件 → Issue #283。
- 項目45: engineのProbitが、statsmodelsが収束できる大標本条件でHessian
  特異エラーを出す件 → Issue #284。
- 項目46: engineのLogit/ProbitのBFGS/L-BFGSがNewton・statsmodelsの同method
  より桁違いに遅い件 → Issue #285。

**現在対応不要と判断した項目**（調査の上、対応しないことをユーザー確認済み）:

- 項目22: Rスクリプト冒頭の引数パースパターン共通化 → 正味の削減が少なく
  間接層が増えるだけと判断し見送り。
- 項目23: `run_lm_predict_crosscheck.R`の手法非依存化 → 消費者がOLSのみで
  時期尚早（YAGNI）、Logit/Probit/Tobit着手時に判断し直す。
- 項目37: R側`library()`の`suppressMessages`不統一によるJSON破損リスク →
  `run_r()`はstdout/stderrを分離しJSONはstdoutのみパースするため実害無しと確認。
- 項目38: Rスクリプトの`script_dir`特定ロジック共通化 → `source()`前に
  ヘルパー自体をsourceする循環依存が生じるため対応不能。

**未着手**: `refactoring-candidates.md`項目12・13・15〜35・37・39、
`refactoring-candidates-2.md`項目46・49。項目51以降は各候補メモファイルを
直接参照すること（本ドキュメントでは網羅しない）。

**環境についての注意（2026-08-22判明）**: `~/.claude/`（Claude Codeのセッション
履歴・メモリ）は`.devcontainer/docker-compose.yml`のvolumeマウント対象外のため
**devcontainerの再ビルドで消える**。セッションをまたいで残したい情報は
Claude Codeのメモリ機能に頼らず、必ずリポジトリ内のファイル（本ドキュメント等）に
書くこと。

---

## Initiative A: `benchmark/`再設計（#231サブIssue、2026-08-29〜2026-08-30）

`refactoring-candidates.md`の単発項目を1つずつ潰す「随時対応ログ」とは別の、
`benchmark/`をパッケージとして構造化し(a)データセット/(b)リファレンスアダプタ/
(c)フィクスチャドライバの3層に分離し、手法ごとの重複を共有ヘルパーへ集約する
構造変更。設計ノート`benchmark-restructure-design.md`はInitiative A完了後に
削除済み（内容は本節・実コード・`benchmark/README.md`に集約済み）。

**決定済み（AskUserQuestionで確認）**:
- トップレベル兄弟構成（`tests/`配下への入れ子は却下）。
- `benchmark/`をパッケージ化し`pyproject.toml`の`[tool.pytest.ini_options]
  pythonpath=["."]`で`sys.path.insert`を全廃。
- `performance/`はトップレベルへ昇格（`benchmark/`の外）。
- フィクスチャドライバは軽量な共有ヘルパー方式（Spec + 汎用ドライバの深い抽象化は
  不採用、下記「不採用」参照）。
- `refactoring-candidates.md`の構造的重複項目（12・13・15・16・18〜21・24〜29・
  31〜35・39、約20項目）はInitiative Aが上位計画として吸収し同ファイルから削除。

**実施内容の要約**（詳細な移行手順・各ステップの検証結果はコミット`ea27117`〜
`f27d1c8`のgit log参照。各ステップとも生成JSON/CSVが`generated_at`等を除き
移行前と完全一致することを確認しながら進めた）:
- `benchmark/`を`__init__.py`付きパッケージ化し、`benchmark/common/`に
  系統横断の共有ヘルパー（`datasets_io.py`・`dgp.py`・`dgp_constants.py`・
  `reference/extract.py`・`reference/r.py`・`driver.py`・`constants.py`）を集約。
  `.devcontainer/devcontainer.json`のPYTHONPATHをリポジトリルート1つに縮小。
- OLS→WLS→Logit→Probit→2SLS/GMMの順に、各系統の`datasets.py`・`freeze.py`・
  `references/`（`statsmodels_ref.py`/`linearmodels_ref.py`/`r.py`）を新構造へ
  再配置し、生成スクリプトの`__main__`を共有の`run_fixture_cli`に統一。
  参照実装アダプタは`statsmodels`本体との名前衝突を避けるため`_ref`サフィックス
  付きにリネーム（旧`run_statsmodels_benchmark.py`同名衝突バグの再発防止）。
- `performance/`をトップレベルへ移動、`benchmark/freeze_datasets.py`を
  `benchmark/regenerate_all.py`（合成CSV＋全フィクスチャJSONの一括再生成
  オーケストレータ）として実体化。
- 最終ステップで全`generate_*_fixtures.py`の`_meta`文字列内の旧ファイル名参照を
  一括更新し対応するフィクスチャJSONを再凍結（数値・バージョンは不変、`_meta`
  文字列と`generated_at`のみ変化）、約40ファイルのdocstring・コメント・
  `CLAUDE.md`等のprose参照を新構造へ更新。

**不採用: Spec/汎用ドライバ層**（`MethodBenchmarkSpec`/`build_fixture_json`）:
設計ノート§5.3で「フィクスチャドライバをdataclass Spec＋データ駆動ループにする」
構想があったが、WLS以降の各ステップで毎回「rule of three未達」として見送り、
最終的に6手法すべてを軽量な共有ヘルパー（`run_fixture_cli`・
`reference/r.py::normalize_names`・`reference/extract.py::extract_coef_se`等）
だけで移行完了した。`_meta`は各`generate_*_fixtures.py`にインライン維持
（dataclass層は導入していない）。将来panel/時系列等で3個目以降の異なる形が
出て共通化余地が再燃したら、その時点で設計し直す。

**これでInitiative A（`benchmark/`パッケージ再設計）完了**。

---

## `/explain-code`による`benchmark/`・`tests/`解説ウォークスルーの進捗（フェーズ外、2026-08-16〜2026-08-31）

上記の随時対応ログ・Initiative Aとは**別セッション・別目的**の継続作業。
ユーザーが`benchmark/`・`tests/`配下をファイル単位で`/explain-code`スキルに
沿って通読し、統計学的な意味・設計判断を確認しつつ、気づいた重複・設計上の
疑問点をその都度`refactoring-candidates-2.md`/`-3.md`・
`test-coverage-candidates.md`に記録する運用。

**状態**: 完了。`benchmark/`（iv/linear/nonlinear/performance配下の主要スクリプト）・
`tests/`（OLS/WLS/Logit/Probit/IV/IV-GMM/Tobitの全ファイル）の解説が一巡した。
各回の質疑で見つかった具体的な指摘内容は候補メモ側（`refactoring-candidates-2.md`/
`-3.md`/`test-coverage-candidates.md`）に記録済みのため、ここでは重複して
記載しない（対応済み項目は各ファイルから削除され、本ドキュメントの「随時対応
ログ」節に集約される）。

**特に重要だった発見**（コード側のコメント・docstringに反映済み）:
- IV関連ドキュメント・テストが揃って「Rの`ivreg`にHC2/HC3の確立した参照実装が
  無い」としていたが、実機検証で誤りと判明（`hatvalues.ivreg`は正常に動作する
  関数で、`sandwich::vcovHC`と6桁以上の精度で一致）。CLAUDE.md 10章に記録された
  `ivreg`サイレントインストール失敗（Issue #171）当時の古い調査が残っていたと
  推測される。
- GMMの`gmm_iterations=1`におけるHansen J統計量の`linearmodels`との不一致は、
  `IVGMM.fit(iter_limit=1)`がループを1度も実行せず重み行列が生の`(Z'Z/n)⁻¹`の
  ままになる（本実装は`σ̂²`スケーリング済みの重みを常に使う）という規約の違いが
  原因と特定（バグではない）。
- Tobitが`method`によらず常にOLSベースの初期値・QR検証を行う設計は、
  Logit/Probitの過去の実バグ（`bfgs`のみ特異性検出漏れ）を構造的に解消できる
  可能性がある設計提案として記録（Issue #279）。

**このウォークスルー中に作成したGitHub Issue（2026-09-05時点、いずれもオープン）**:
[#246](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/246)
（検定分布・診断統計量の運用ノートのドキュメント化）・
[#247](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/247)
（Cragg-Donald統計量の再検討）・
[#249](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/249)
（GMMのC統計量実装）・
[#256](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/256)
（GMM/Hansen JのRクロスチェック再検討）・
[#264](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/264)
（Logit/Probitの最適化methodにFisher-scoring追加検討）・
[#266](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/266)
（非polars DataFrameを渡すと`ValidationError`ではなく`AttributeError`が漏れる）・
[#267](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/267)
（Rとの計算慣習差に完全一致させる互換モードの要否、優先度低）・
[#277](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/277)
（WLS: `weight`と同じ列を`x`に含めることを許容する、実施は未着手）。

---

## フェーズ1: リファクタリング用スキル

**目的**: コード（Python/R/Rust）・コメント・ディレクトリ構造・ドキュメント
（README/CLAUDE.md/仕様書）を横断的にリファクタリングするための観点を固定した
スキルを用意する。

**状態**: 完了。

**結果**: 既存スキル（`review-rust`/`review-python`/`review-testing`/`simplify`）は
いずれも「直近の差分」対象で、実装当時から蓄積した既存コード・ドキュメントの
横断的な整理には非対応と判断。`.claude/skills/refactor/SKILL.md`を新規作成し、
「指摘→適用まで一括」「コード＋コメント＋ディレクトリ構造＋ドキュメントを
1つの統合スキルで扱う」方針で運用開始（対象範囲は都度ユーザーが指定、
破壊的操作は計画提示→合意→適用を必須化、ロジックの挙動を変える変更は範囲外）。

---

## フェーズ2: `benchmark/`ディレクトリの整理とリファクタリング

**目的**: テスト用データ作成コード・参照用パラメータJSON作成コード・パフォーマンス
測定コードが混在し手法追加に対応できない構造、pyfixest関連の不要ファイル残存を
解消する。

**状態**: 完了（この時点の構成・ファイル名は、後のInitiative Aでさらに
`benchmark/common/`パッケージ構造へ再編されたため現在のコードには対応しない。
現状は`benchmark/README.md`・Initiative A節参照）。

**主な決定・実施内容**:
- パフォーマンス測定コードを系統横断の`benchmark/performance/`（後にInitiative Aで
  トップレベル`performance/`へ再移動）へ分離。
- 系統非依存の重複ロジック（`imbalanced_cluster_groups`・`hac_auto_lag`・
  `DATA_DIR`構築・データロード等）を`benchmark/_common.py`（後に`common/`へ細分化）
  へ集約。
- 未使用ファイル（`run_pyfixest_benchmark.py`・`run_fixest_benchmark.R`）を削除。
- `SCENARIOS`/`NUMERIC_SCENARIOS`/`COV_TYPES`の3〜4階層重複を、生成スクリプト側を
  正としテスト側がimportする形に一元化。
- Rスクリプトの重複ロジック（係数・標準誤差抽出、ロバストWald検定）を
  `benchmark/_common.R`へ抽出。
- Issue番号コメントの整理（非自明なWHYは残し番号のみ削除）。

**最終確認**: `ruff check`/`ruff format --check`全件パス、`pytest`670件全件パス、
freeze出力のバイト一致・Rスクリプト再実行結果のバイト一致を確認済み。

**発見した既知バグ（フェーズ2では対応せず、フェーズ4送り）**: `generate_logit_
crosscheck_fixtures.py`の`near_separation`シナリオ×classical×logitの組み合わせで
Rスクリプトの不変条件チェックが失敗（変更前コードでも再現するため今回の
リファクタリングが原因ではない潜在バグと確認）。原因はフェーズ4で判明・解消
（下記参照）。

---

## フェーズ3: `tests/`ディレクトリの整理とリファクタリング

**目的**: `tests/api_tests`という中間ディレクトリの要否、共通アサーションの
重複、許容誤差設定の分散、Issue番号コメント残置を解消する。

**状態**: 完了（この時点のディレクトリ名`tests/api_tests/`は、後に
`tests/`へフラット化 → さらに項目68で系統別×関心事別に再分割されたため
現在のコードには対応しない）。

**主な決定・実施内容**:
1. `tests/api_tests/` → `tests/`へフラット化（中間ディレクトリ不要と判断）。
2. `tests/_assertions.py`新設: `assert_close`/`assert_dict_close`/
   `rename_intercept`/`check_margeff`を集約（crosscheck系は許容誤差の計算式が
   一部異なるため対象外、フェーズ3.5で解消）。
3. `tests/_tolerances.py`新設: 許容誤差の値を手法ごとに辞書化。
4. `tests/_helpers.py`新設: `with_cluster_groups`・`separation_suspected_dataset`・
   `MROZ_X`・Wooldridgeロードの一本化。
5. `conftest.py`拡張: 共通フィクスチャ・`sys.path.insert`の集約。
6. `COV_TYPES`一元化をLogit/Probit系にも拡張。
7. IVの`fit()`呼び出し共通化（`_our_fit`ヘルパー）。
8. **Issue番号コメント整理**: `#153`（`test_wls.py`2箇所）・`#171`（4ファイル
   5箇所）・`#231`（4ファイル）の計11箇所を番号のみ削除し周辺の説明文は保持
   （`refactoring-candidates-2.md`項目51が参照する「フェーズ3実施結果項目8」は
   これを指す）。

**最終確認**: `pytest tests`670件全件パス、`ruff check`/`ruff format --check`
全件パス。

---

## フェーズ3.5: crosscheckテストの許容誤差計算式バグ修正

**背景**: フェーズ3の調査で、`test_ols_crosscheck.py`/`test_iv_crosscheck.py`が
`tol = rtol * max(abs(ref_val), 1e-8)`という式（`|ref|`が0近傍で許容誤差が
極小値まで縮み偽陽性の失敗を起こしうるバグ）を使い、他ファイルの正しい式
`tol = max(rtol * abs(ref_val), atol)`（`.claude/rules/testing-policy.md`
「許容誤差」記載の現行方針）と一致していないことが発覚。

**状態**: 完了。

**対応**: `git log -S`でバグの経緯（WLSクロスチェック作成時に発覚し
`test_wls_crosscheck.py`のみ修正済みだったが、OLS/IV版が追従していなかった）を
確認した上で、両ファイルの計算式を正しい式に修正（数学的にA≥Bの関係のため
既存フィクスチャに対して新規失敗は発生しないことを事前に確認済み）。
`tests/_assertions.py`の共有関数へ統合。

**最終確認**: `pytest tests`670件全件パス（新規失敗0件）、`ruff`パス。

---

## フェーズ4: ロジック整理前のテスト拡充

**目的**: フェーズ5〜7（ロジック変更）で既存実装を壊さないよう、着手前に
`review-testing`スキルでOLS/WLS/Logit/Probit・IV（#232〜238）のテストを
拡充する。

**状態**: 完了（linear系統〔OLS/WLS〕・nonlinear系統〔Logit/Probit〕・IV系統
〔2SLS/GMM、#232〜238〕全て完了）。

**進め方**: 各系統について、(0) `benchmark/`再生成でフィクスチャの整合性を
確認 → (1) `testing-completeness-reviewer`のレビューで指摘を洗い出し、
ユーザー確認の上で対応 → (2) 検証（`cargo test`/`pytest`/`ruff`/`clippy`）、
という手順を踏んだ。

**linear系統（OLS/WLS）**: must fix 1件・should fix 7件・nice to have 4件、
計12件の指摘に全て対応。主な内容: Rクロスチェックに`r_squared`等の欠落
フィールドを追加、WLS実データ（401ksubs）のcov_type別・クラスター別検証を
拡充、`scale_variance_mild`シナリオ追加、`cov_type`大文字小文字非依存性の
テスト追加。`rust-reviewer`のレビューで、`.unwrap()`使用箇所がGIL未初期化
環境でのpanic時に二重パニック（テストバイナリ全体がSIGABRT）を起こす
バグを検出・`let-else`パターンへ修正（`engine_pybind/src/linear/common.rs`）。
検証: `cargo test -p engine`（317件）・`-p engine_pybind`（68件）・
`pytest tests`（752件）・`ruff`/`clippy`全てグリーン。

**nonlinear系統（Logit/Probit）**: must fix 0件・should fix 5件・nice to have
8件中4件に対応（残り4件は`test-coverage-candidates.md`へ記録し見送り）。
主な内容: `method="bfgs"/"lbfgs"`のフル統計量照合・`method`×3の特異性検出
回帰テスト・`include_intercept=False`の数値照合を追加。フェーズ2で発覚していた
Rクロスチェックの`near_separation`不変条件チェック失敗は、浮動小数点精度の
限界（IRLS内部の期待情報行列と観測情報行列の微小なズレ）が原因であり、かつ
コミット済みフィクスチャがProbit対応時の観測情報行列採用より前の陳腐化した
値だったと判明、許容誤差を実測ベースで緩めて解消。`rust-reviewer`のレビューで
同種の`.unwrap()`パニックリスクが既存テスト8箇所にも見つかり合わせて修正。
検証: `cargo test -p engine`（317件）・`-p engine_pybind`（72件）・
`pytest tests`（810件）・`ruff`/`clippy`全てグリーン。

**IV系統（2SLS/GMM、Issue #232〜238）**: must fix 1件・should fix 5件・
nice to have 3件、計9件の指摘に全て対応。特に、複数内生変数
（`k_endog>=2`）のテストが皆無だった調査中に、DGPの第一段階誤差が全内生変数に
単一列としてブロードキャストされ`k_endog>=2`で第一段階回帰残差が実質完全共線に
なる設計バグを発見・修正（`k_endog=1`では従来と数学的に完全一致する一般化のため
既存シナリオへの影響ゼロ）。他に`weight_type="kernel"`×`cov_type="hac"`・
`gmm_iterations`の複数値・実データ（Wooldridge `card`）・`cov_type`/`weight_type`
大文字小文字非依存性のテストを追加。検証: `pytest tests`（878件）・`ruff`/
`clippy`全てグリーン。

続く追加ラウンドで、Issue #232〜238の実施状況を確認したところ#234/#236/#238の
3件のみ完了で#232/#233/#235/#237が未着手と判明し、残り4件も実装した
（自由度1境界シナリオ追加、Rクロスチェックへのt値・p値・信頼区間・nobs追加、
wu_hausmanのcov_type=hacケースをR `ivreg`の`vcov.`引数を関数として渡す方式で
検証可能と判明させ対応、等）。実装中に判明した副次的なバグ（`_nested_f_test`の
非中心化SSR誤用、augmented regression飽和時のエラーハンドリング不備）も合わせて
修正。HAC関連の新規フィールドは実測乖離に基づく専用許容誤差を追加。
検証: `pytest tests`（885件）・`ruff`/`cargo test`全てグリーン。完了条件を
満たしたことを確認の上、Issue #232・#233・#235・#237をクローズ。

---

## フェーズ5: `python_package/`のリファクタリング

**想定範囲**: Issue番号への言及の削除が主。タスクとしては軽量な見込み。

**状態**: [#248](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/248)に引き継ぎ（未着手のまま#231をクローズ）

---

## フェーズ6: `engine_pybind/`のリファクタリング

**想定範囲**: 重複ロジックの共通化、Issue番号への言及の削除、コード整理。
`cov_type`パース等、既にA2（[#153](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/153)）で
共通化済みの範囲との重複有無を再確認する。

**状態**: [#248](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/248)に引き継ぎ（未着手のまま#231をクローズ）

---

## フェーズ7: `engine/`のリファクタリング

**注意**: ロジックを壊さないこと・無理な共通化をしないことを優先する、
最も慎重な設計判断が必要な領域。フェーズ4で拡充したテストで担保する。
`engine/src/linear/ols.rs`の`from_columns`と同様の実装が各手法のメインロジックに
あるが、共通化可能か（無理な統合はしない）を検討する。

**状態**: [#248](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/248)に引き継ぎ（未着手のまま#231をクローズ）

---

## 未解決の確認事項

（フェーズ着手時に判断が分かれる点が出た場合はここに追記し、ユーザー確認後に
解消済みとして記録する）

- なし
