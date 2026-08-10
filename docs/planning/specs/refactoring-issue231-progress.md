# Issue #231 リファクタリング進捗管理

[#231](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/231)
（リファクタリング用スキルの作成とOLS/WLS/Probit/Logit/IVのリファクタリング）の
進捗・メモを記録するドキュメント。範囲が広いため7フェーズに分割し、1フェーズずつ
完了させてから次に進む。各フェーズの対象範囲・変更方針はユーザーが都度指示し、
このドキュメントはその結果・未解決事項を記録する。

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
| 2 | `benchmark/`ディレクトリの整理とリファクタリング | 未着手 |
| 3 | `tests/`ディレクトリの整理とリファクタリング | 未着手 |
| 4 | ロジック整理前のテスト拡充（OLS/WLS/Logit/Probitレビュー＋IV #232〜238） | 未着手 |
| 5 | `python_package/`のリファクタリング | 未着手 |
| 6 | `engine_pybind/`のリファクタリング | 未着手 |
| 7 | `engine/`のリファクタリング | 未着手 |

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

**状態**: ステップ1（ディレクトリ配置整理）完了。ステップ2（コード整理）は未着手。

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

**状態**: 未着手

**メモ**: (着手後に記載)

---

## フェーズ4: ロジック整理前のテスト拡充

**目的**: フェーズ5〜7（python_package/engine_pybind/engineのロジック変更）で
既存実装を壊さないよう、着手前にテストを拡充する。

**進め方**:
1. `review-testing`スキルでOLS/WLS/Logit/Probitのテストをレビューし、
   不足があれば充足する。
2. IV分（[#232](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/232)〜
   [#238](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/238)）の
   テスト拡充も合わせて実施する。

**状態**: 未着手

**メモ**: (着手後に記載)

---

## フェーズ5: `python_package/`のリファクタリング

**想定範囲**: Issue番号への言及の削除が主。タスクとしては軽量な見込み。

**状態**: 未着手

**メモ**: (着手後に記載)

---

## フェーズ6: `engine_pybind/`のリファクタリング

**想定範囲**: 重複ロジックの共通化、Issue番号への言及の削除、コード整理。

**気になっている個所（Issue #231より）**:
- `engine_pybind`側の`cov_type`パース等、既にA2（[#153](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/153)）で
  共通化済みの範囲との重複有無を再確認する

**状態**: 未着手

**メモ**: (着手後に記載)

---

## フェーズ7: `engine/`のリファクタリング

**注意**: ロジックを壊さないこと・無理な共通化をしないことを優先する、
最も慎重な設計判断が必要な領域。フェーズ4で拡充したテストで担保する。

**気になっている個所（Issue #231より）**:
- `engine/src/linear/ols.rs`の`from_columns`と同様の実装が各手法のメインロジックに
  あるが、共通化可能か（IV着手時のA章と同様、共通化が呼び出し箇所の性質上
  適さない場合は無理に統合しない）

**状態**: 未着手

**メモ**: (着手後に記載)

---

## 未解決の確認事項

（フェーズ着手時に判断が分かれる点が出た場合はここに追記し、ユーザー確認後に
解消済みとして記録する）

- なし（フェーズ1着手前時点）
