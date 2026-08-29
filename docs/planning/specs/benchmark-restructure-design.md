# benchmark/ 再設計ノート（Initiative A）

`benchmark/`（リファレンス実装によるフィクスチャ生成ツール群）を、パッケージとして
正しく構造化し、3つの関心事に分離し、手法ごとの重複を Spec + 汎用ドライバに集約する
ための設計ノート。実装は本ノートに沿って **OLS 先行 → 手法ごと**に進める。

- 位置づけ: `refactoring-candidates.md` の項目11〜41のうち構造的な重複（後述の一覧）を
  まとめて解消する上位計画。`refactoring-issue231-progress.md` の「随時対応ログ」
  （単発の候補項目を1つずつ潰す）とは別の、構造変更として **Issue #231 のサブ Issue**
  を立てて進める。
- 本ノートで扱わないもの: `benchmark/performance/`（性能比較。依存・ライフサイクルが
  別。項目40・41）、`tests/` の手法別ディレクトリ分割（項目43）、API 設計（CLAUDE.md
  2章の非交渉事項には一切触れない。これは `benchmark/` 内部ツールの構造の話）。

---

## 1. 背景と目的

### なぜ今か

- プレリリース期間（`0.x`）で、実装済みは6手法（OLS/WLS/Logit/Probit/IV/Tobit）。
  Phase4以降で FE/RE・パネル・DID/RDD・時系列（VAR 等）が追加され、手法数は
  線形に増える。**手法が増えてから再設計するより、今のうちに骨格を変える方が
  総コストが小さい**（ユーザー判断）。
- 現状の「手法ごとにフラットなスクリプトを並べ、必要な関数を各ファイルにコピペ」
  というスタイルは、`refactoring-candidates.md` 項目11〜41 が示すとおり
  **同型関数の重複が系統×手法の数だけ増える**構造になっている。

### 現状の5つの問題（ユーザー整理）

1. パフォーマンス評価用ベンチマークとテスト用フィクスチャ生成が `benchmark/` に混在。
2. `benchmark/<系統>/fixtures/` の "fixtures" が「期待値JSON生成」の意味で使われる
   一方、入力CSVを凍結する `freeze_*.py` は別階層にあり、命名と役割が食い違う。
3. 「検証用入力データの生成」と「パラメータ/統計量のリファレンスJSON生成」は
   分離できるはずだが、`generate_*_fixtures.py` に (b)(c) が同居している（後述）。
4. `benchmark/freeze_datasets.py`（系統ディスパッチャ）は、手法ごとにCSVを作る以上
   必須ではないのでは。
5. 関数スクリプトのまま手法ごとに分かれており、手法増加時に再設計が要る。
   今のうちに Spec/ドライバ型に変えておきたい。

---

## 2. 現状の構造と問題点（事実確認）

### 2.1 `benchmark/` ↔ `tests/` は既に import 時結合している

`pytest tests` は既に `benchmark/` のモジュールを **import 時に**使用している。

| tests 側 | import 元 | 用途 |
|---|---|---|
| `from generate_ols_fixtures import COV_TYPES, NUMERIC_SCENARIOS` | `benchmark/linear/fixtures/` | シナリオ/cov_type の単一定義元 |
| `from generate_wls_fixtures import _add_age_bin` | 同上 | **private 関数**の直接 import |
| `from generate_iv_fixtures import CARD_X_EXOG` | `benchmark/iv/fixtures/` | 定数 |
| `from generate_ols_crosscheck_fixtures import NUMERIC_SCENARIOS, PREDICT_NEW_DATA` | `benchmark/linear/fixtures/` | 定数 |
| `from _common import imbalanced_cluster_groups` | `benchmark/_common.py` | 約10テストファイル |

- `generate_*_fixtures.py` は先頭で `import statsmodels` する。**このため statsmodels は
  `test` 依存グループにある**（`benchmark` グループではない）。CLAUDE.md 10章の
  「重い依存を `tests/` から隔離」という前提は Python 依存については既に崩れている。
  R だけが真に隔離されている（pip 依存にできない）。
- `benchmark/<系統>/fixtures/generate_*_crosscheck_fixtures.py`（5ファイル）は
  `subprocess.run(["Rscript", ...])` を**関数内で**呼ぶため、import 自体は R 非依存で
  安全。pytest はこれらを定数取得のために import するだけで `build_fixtures()` は
  呼ばない。

### 2.2 import 解決が `sys.path` 継ぎ足しに依存し脆い

- `tests/conftest.py` が `benchmark/` を1回 insert。
- 12個のテストファイルが各々 `benchmark/<系統>/fixtures/` を個別 insert。
- さらに `benchmark/linear/` が別経路（editable `.pth` か実行順序依存の状態）で
  解決されており、**隔離した環境で `import generate_ols_fixtures` が再現しない**
  （`polars` すら見つからない）。
- `run_statsmodels_benchmark.py` の同名衝突バグ（`refactoring-issue231-progress.md`
  項目71）は、この「複数の `sys.path` エントリ + 同名モジュール」構造の症状。

### 2.3 1手法あたり3つの関心事が2〜3ファイルに絡んでいる

| 関心事 | 現在のファイル | 中身 |
|---|---|---|
| (a) 入力データ | `generate_<系統>_datasets.py` + `freeze_<系統>_datasets.py` | DGP、シナリオ、CSV 凍結 |
| (b) リファレンス呼び出し | `run_statsmodels_benchmark_<系統>.py` / `run_linearmodels_benchmark_iv.py` + `.R` | 1ケース → 結果 dict |
| (c) フィクスチャ組み立て | `generate_<手法>_fixtures.py` / `generate_<手法>_crosscheck_fixtures.py` | 全ケースを回して JSON + CLI |

(b)(c) が `generate_*_fixtures.py` に同居し、`_run_cluster_case` / `_run_401ksubs_case`
/ `_run_wage1_region_cluster_case` などの「グリッドに乗らない特殊ケース」がそこに
散在している。`refactoring-candidates.md` 項目11〜41 はほぼ全てこの (b)/(c) の
分離不足に起因する。

### 2.4 Rスクリプトの境界

- `run_*_benchmark.R` 6本 + `_common.R`。R は pip 依存にできず、`ci_python.yml` の
  pytest ジョブには R が入らない。だから R クロスチェック値は**事前生成して
  `tests/fixtures/benchmarks/*_crosscheck.json` をコミット**してある。
- Python アダプタ（`_run_r`）が `Path` を組み立てて `Rscript` を subprocess 起動する。
  `.R` は「R がある環境でしか実行できず、pytest 実行時は import されても実行されて
  はいけない」コード。

---

## 3. 決定事項（本ノートで確定）

| # | 決定 | 根拠 / 却下案 |
|---|---|---|
| D1 | **トップレベル兄弟構成**を採る。`performance/` をトップに昇格、`benchmark/` は名前維持で中身をパッケージ再編、`tests/` は場所も名前も不動。 | `tests/` 配下への入れ子は `import tests` 名前衝突、pytest 走査からの分離が逆に困難化、移設コスト大（全 `test_*.py` 移動 + 文書群のパス一括書換）。入れ子の利点（「1ツリー」）は `pythonpath` を1回通せば入れ子なしで得られる。 |
| D2 | `benchmark/` を **`__init__.py` を持つ本物のパッケージ**にする。`pyproject.toml` に `[tool.pytest.ini_options] pythonpath = ["."]` を1行足し、**per-file `sys.path.insert` を全廃**、ドット表記 import に統一。 | 同名モジュール衝突（項目71）が構造的に不可能になる。import 解決が環境状態に依存しなくなる。 |
| D3 | (b)(c) を分離する。**(b) リファレンスアダプタ**＝「凍結 df + spec + cov_type → 結果 dict」、**(c) フィクスチャドライバ**＝「Spec を受けて scenarios×cov_types + 特殊ケースを回し JSON 化 + CLE」。 | 項目11〜41 の構造的重複の大半がここで解消。 |
| D4 | (c) は **Spec（dataclass）+ 汎用ドライバ関数**。深い継承（template method）は採らない。 | ベンチマーク生成は制御フローが grep で追える利点を保ちたい。データ駆動の方が特殊ケースの追加が読みやすい。 |
| D5 | `benchmark/performance/` は **対象外**。`performance/`（トップ）に昇格し、性能比較ツールとしてそのまま残す。共通化の是非は項目40で別途。 | pyfixest 依存・`benchmark_ols.yml` 専用・pytest 無関係で性質が違う。1サンプル（OLS のみ）から共通境界を判断できない。 |
| D6 | R スクリプトは**各系統のアダプタ層の隣**（`benchmark/<系統>/references/*.R`）に置き、`run_r` は `Path(__file__).parent / "..."` で解決。`_common.R` は `benchmark/common/_common.R`。 | 項目32（`<系統>_DIR / R_SCRIPT` パターンの重複）が解消。`source()` のブートストラップ（項目38）は R に `__file__` 相当が無いため現状維持。 |
| D7 | 移行は **OLS 先行 → 手法ごと**。各手法で「移行前後の `<手法>.json` / `<手法>_crosscheck.json` / 凍結CSV が `_meta.generated_at` 除外で完全一致」を確認してからコミット。 | `refactoring-issue231-progress.md` で確立済みの不変性チェック手順を踏襲。CSV/JSON が壊れるリスクを最小化。 |

---

## 4. 目標ディレクトリ構造

```
<repo root>/
├── performance/                        # 旧 benchmark/performance/（トップに昇格）
│   ├── compare_performance.py
│   └── render_performance_summary.py
│
├── benchmark/                          # 名前維持・中身をパッケージ再編
│   ├── __init__.py
│   ├── common/                         # 旧 benchmark/_common.py / _dgp_constants.py / _common.R を分割
│   │   ├── __init__.py
│   │   ├── datasets_io.py              # DATA_DIR, load_frozen_dataset, freeze_scenarios, run_freeze_cli
│   │   ├── dgp.py                      # imbalanced_cluster_groups, hac_auto_lag, linear_predictor,
│   │   │                              #   correlated_design_matrix, apply_perfect_multicollinearity,
│   │   │                              #   validate_choice, preview_dataset
│   │   ├── dgp_constants.py            # scale/誤差項定数（旧 _dgp_constants.py。dgp.py へ統合せず据え置き）
│   │   ├── cluster_cases.py            # run_cluster_case（6コピーを1本化。項目13/15/24/35）
│   │   ├── reference/
│   │   │   ├── extract.py              # extract_coef_se（項目11・実装済み）等の抽出ヘルパー
│   │   │   ├── r.py                    # run_r, normalize_names（項目33/34/35/39）
│   │   │   └── meta.py                 # build_meta（項目18）
│   │   ├── driver.py                   # MethodBenchmarkSpec, build_fixture_json, run_fixture_cli（項目19/26）
│   │   ├── constants.py               # SYNTHETIC_FORMULA, MROZ_FORMULA, WEIGHT_COLUMN_NAME（項目16/25/27）
│   │   ├── load_wooldridge.py          # 旧 benchmark/load_wooldridge.py
│   │   └── _common.R                   # 旧 benchmark/_common.R
│   │
│   ├── linear/
│   │   ├── __init__.py
│   │   ├── datasets.py                 # 旧 generate_linear_datasets.py + freeze_linear_datasets.py 統合
│   │   ├── constants.py                # NUMERIC_SCENARIOS, COV_TYPES（main↔crosscheck 単一定義。項目20/28）
│   │   ├── references/
│   │   │   ├── statsmodels.py          # 旧 run_statsmodels_benchmark_linear.py（run() = アダプタ）
│   │   │   ├── r.py                    # run_lm_crosscheck.R / run_lm_predict_crosscheck.R を呼ぶ薄い層
│   │   │   ├── run_lm_crosscheck.R     # 旧 run_lm_crosscheck_benchmark.R
│   │   │   └── run_lm_predict_crosscheck.R
│   │   ├── ols.py                      # OLS の MethodBenchmarkSpec 定義（+ __main__ で run_fixture_cli）
│   │   └── wls.py                      # WLS の Spec 定義
│   │
│   ├── nonlinear/  … logit.py / probit.py / references/{statsmodels.py,r.py,run_glm_crosscheck.R}
│   ├── iv/         … iv.py / iv_gmm.py / references/{linearmodels.py,r.py,run_ivreg.R}
│   └── panel/      … references/run_plm_benchmark.R（Phase4 で肉付け）
│
└── tests/                              # 場所も名前も不動
    ├── conftest.py  _helpers.py  _assertions.py  _tolerances.py
    ├── fixtures/benchmarks/            # 固定 CSV + リファレンス JSON（コミット済み成果物・不動）
    │   ├── data/*.csv  *_true_beta.json
    │   └── *.json  *_crosscheck.json
    └── test_*.py
```

### 旧 → 新 対応（linear の例）

| 旧 | 新 |
|---|---|
| `benchmark/_common.py` | `benchmark/common/{datasets_io,dgp}.py` + `common/reference/extract.py`（＋後続で `constants.py` / `reference/{r,meta}.py` / `driver.py`） |
| `benchmark/_dgp_constants.py` | `benchmark/common/dgp_constants.py`（据え置き） |
| `benchmark/_common.R` | `benchmark/common/_common.R` |
| `benchmark/load_wooldridge.py` | `benchmark/common/load_wooldridge.py` |
| `benchmark/freeze_datasets.py` | 廃止。`benchmark/regenerate_all.py`（薄い「全手法再生成」スクリプト）に置換（項目4） |
| `benchmark/linear/generate_linear_datasets.py` | `benchmark/linear/datasets.py`（DGP 部分） |
| `benchmark/linear/freeze_linear_datasets.py` | `benchmark/linear/datasets.py`（freeze 部分。同一ファイルに DGP と freeze を隣接） |
| `benchmark/linear/run_statsmodels_benchmark_linear.py` | `benchmark/linear/references/statsmodels.py` |
| `benchmark/linear/run_lm_crosscheck_benchmark.R` | `benchmark/linear/references/run_lm_crosscheck.R` |
| `benchmark/linear/run_lm_predict_crosscheck.R` | `benchmark/linear/references/run_lm_predict_crosscheck.R` |
| `benchmark/linear/fixtures/generate_ols_fixtures.py` | `benchmark/linear/ols.py`（Spec 定義のみ。ループ/CLI はドライバへ） |
| `benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py` | `benchmark/linear/ols.py` の Spec に `crosscheck` アダプタとして畳み込み |
| `benchmark/linear/fixtures/generate_wls_fixtures.py` | `benchmark/linear/wls.py` |
| `benchmark/linear/fixtures/generate_wls_crosscheck_fixtures.py` | `benchmark/linear/wls.py` に畳み込み |

`fixtures/` サブディレクトリは消える（項目2の命名曖昧さが解消）。「fixtures」は
`tests/fixtures/benchmarks/` の**成果物**だけを指す語になる。

---

## 5. レイヤ設計

### 5.1 (a) データセット層 — `benchmark/<系統>/datasets.py`

- DGP（`generate_*` 関数）と freeze（`tests/fixtures/benchmarks/data/` へ CSV 書き出し）を
  **同一ファイルに隣接**させる。系統ディスパッチャ（旧 `freeze_datasets.py`）は廃止し、
  「全系統を順に呼ぶ」だけの `benchmark/regenerate_all.py` に縮小（項目4）。
- 疑似グループ列の凍結CSV焼き込み（項目12）は本レイヤの担当になるが、**設計判断を
  含むため本ノートでは方式を確定しない**（後述 8章「未解決」）。当面は (c) 側の
  `run_cluster_case` が実行時にラベルを付与する現行方式を維持し、Spec の
  `extra_cases` に載せる。

### 5.2 (b) リファレンスアダプタ層 — `benchmark/<系統>/references/`

- **アダプタの契約**: `call(frozen_df, *, formula, cov_type, **extra) -> dict`。
  戻り値は手法が公開する統計量のフラットな dict（`coef` / `se` / `t_stats` / … /
  `_meta` は付けず、`_meta` はドライバが `build_meta` で組み立てる）。
- `statsmodels.py` の `run()` は「1ケース = 1アダプタ呼び出し」に純化する
  （現状の `run()` はほぼこの形。`_meta` 組み立てをドライバに移すだけ）。
- `r.py` は `common/reference/r.py::run_r(script_path, csv_path, formula, cov_type,
  **extra)` を呼ぶ薄い系統別ラッパー。`run_r` / `normalize_names` は共通化
  （項目33/34/35/39）。`normalize_names` は
  `normalize_names(raw, *, stat_key="t_stats"|"z_stats", extra_keys=[...],
  conf_from_low_high=False)` のようにパラメータ化する。
- `.R` 本体は同ディレクトリに置き、`run_r` が `Path(__file__).parent` 基準で解決
  （項目32）。`suppressMessages` を全 `.R` に適用（項目37）。

### 5.3 (c) フィクスチャドライバ層 — `benchmark/common/driver.py` + `benchmark/<系統>/<手法>.py`

```python
@dataclass(frozen=True)
class ReferenceAdapter:
    name: str  # "statsmodels" | "r-lm" | "linearmodels" | ...
    call: Callable[..., dict]


@dataclass(frozen=True)
class ExtraCase:
    key: str  # "cluster" | "cluster_imbalanced" | "cluster_g2" | "wage1_region" | ...
    run: Callable[
        ["DriverContext"], dict
    ]  # グリッドに乗らないケースを1エントリ分生成


@dataclass(frozen=True)
class MethodBenchmarkSpec:
    method: str  # "ols"
    family: str  # "linear"
    dataset_prefix: str  # "synthetic"
    scenarios: list[str]
    x_cols: list[str]  # ["x1", "x2", "x3"]
    cov_types: list[str]
    primary: ReferenceAdapter  # statsmodels
    crosscheck: ReferenceAdapter | None  # r-lm（None の手法もありうる）
    extra_cases: list[ExtraCase]
    hac_lag: int | None = None
    weight_col: str | None = None
    note_primary: str = ""
    note_crosscheck: str = ""


def build_fixture_json(
    spec: MethodBenchmarkSpec, *, which: Literal["primary", "crosscheck"]
) -> dict:
    """scenarios × cov_types + extra_cases を回して 1 手法分の JSON を組み立てる。"""


def run_fixture_cli(
    spec: MethodBenchmarkSpec, *, which: str, default_output: str
) -> None:
    """旧 generate_*_fixtures.py 11ファイルで一字一句同じだった __main__ を1関数に（項目19）。"""
```

- **クラスターケース**は `ExtraCase` として表現し、`run` は共通ヘルパー
  `benchmark/common/cluster_cases.py::run_cluster_case(ctx, *, groups, x_cols,
  base_dataset, adapter)` を呼ぶ。これが現行の6コピー
  （linear/nonlinear/iv の main・crosscheck）を1本化する（項目13/15/24/35）。
  `k1`（OLS の G=2 境界が説明変数1個を要する差）は `x_cols` / `base_dataset` の
  引数差として吸収（項目15）。
- **実データケース**（`_run_401ksubs_case` / `_run_wage1_region_cluster_case`）も
  `ExtraCase`。派生列（`inv_inc` / `age_bin` / `region`）の組み立ては
  `ExtraCase.run` の中に閉じる。`_add_age_bin` は `benchmark/common/` へ移し
  独立関数化（項目29）。
- **main と crosscheck の共通化**（項目20/26/28）: `scenarios` / `cov_types` /
  `x_cols` は Spec の単一フィールド。`which="primary"|"crosscheck"` でアダプタだけ
  切り替える。Logit/Probit の `cov_types` の main↔crosscheck 非対称（`hc1` の有無）は
  Spec に `crosscheck_cov_types: list[str] | None` を追加して意味のある差として明示。

### 5.4 定数の単一定義元

| 定数 | 新しい置き場所 | 解消する項目 |
|---|---|---|
| `SYNTHETIC_FORMULA = "y ~ x1 + x2 + x3"` | `benchmark/common/constants.py` | 27 |
| `MROZ_FORMULA` | `benchmark/common/constants.py`（または `nonlinear/constants.py`） | 16 |
| `WEIGHT_COLUMN_NAME = "weight"` | `benchmark/common/constants.py` | 25 |
| `NUMERIC_SCENARIOS` / `COV_TYPES`（系統ごと） | `benchmark/<系統>/constants.py`、Spec が参照 | 20 |
| `WOOLDRIDGE_COV_TYPES` | `benchmark/linear/constants.py` | 28 |

`tests/` はこれらを `from benchmark.linear.constants import NUMERIC_SCENARIOS` の
ように**ドット表記で1経路 import**する（`pythonpath = ["."]` により per-file
`sys.path.insert` 不要）。

---

## 6. パッケージ化と import

- `benchmark/` 以下の全ディレクトリに `__init__.py` を置く。
- `pyproject.toml`:
  ```toml
  [tool.pytest.ini_options]
  pythonpath = ["."]
  ```
  これで `tests/` から `import benchmark.linear.ols` 等が通る。
- `tests/conftest.py` の `sys.path.insert(0, .../benchmark)` と、12テストファイルの
  個別 `sys.path.insert(...)` を**全削除**。import を `from benchmark.<...> import`
  に書き換える。
- `benchmark/` 内部の相対 import（`from _common import ...` → `from benchmark.common
  .dgp import ...` あるいは `from ..common.dgp import ...`）も統一。
- 同名モジュール（`run_statsmodels_benchmark_linear` / `_nonlinear`）は
  `benchmark.linear.references.statsmodels` / `benchmark.nonlinear.references.statsmodels`
  という別パッケージ配下の同名モジュールになり、衝突しない（項目71 の再発防止）。

---

## 7. CI・依存グループへの影響

- **`.github/workflows/ci_python.yml`**: トリガー `paths` に `benchmark/**` を追加
  （生成コードは `pytest tests` が import するため、壊れると pytest も落ちる。
  現状トリガーに無いのは潜在的ギャップ）。`tests/**` はそのまま。
- **`.github/workflows/benchmark_ols.yml`**: `working-directory: benchmark/performance`
  → `performance` に変更。
- **依存グループ（`pyproject.toml`）**: `statsmodels` / `linearmodels` は引き続き
  `test` グループ（`generate_*` 相当が import 時に読み込む構造は維持）。`benchmark`
  グループ（`pyfixest` / `wooldridge` / `pyarrow`）は変更なし。将来 `import
  statsmodels` を関数内に落として `test` グループから外す案は本ノートの範囲外
  （`tests/test_ols.py` 等が statsmodels を直接使うため外し切れない）。
- **ruff**: `benchmark/` がパッケージ化されても `ruff check .` / `ruff format --check .`
  の対象は変わらない（既に `.` 全体）。

---

## 8. 移行手順（OLS 先行）

各ステップ後に `pytest tests -k <手法>` がグリーンであることを確認する。

1. **足場（手法非依存・1コミット）**: 【2026-08-29 実施済み】
   `benchmark/__init__.py` 群（8ディレクトリ）、`benchmark/_common.py` →
   `benchmark/common/helpers.py`・`_dgp_constants.py` → `benchmark/common/dgp_constants.py`・
   `load_wooldridge.py` → `benchmark/common/load_wooldridge.py` の `git mv`、
   `benchmark/common/__init__.py` で公開 API を re-export、`pyproject.toml` に
   `[tool.pytest.ini_options] pythonpath = ["."]`、`benchmark/` 内 internal import と
   `tests/`（conftest + 12ファイル）の `sys.path.insert` を全廃してドット表記
   （`from benchmark.<...> import`）へ。**生成ロジックは不変**（純粋な移動と
   import 経路付け替え）。検証: `PYTHONPATH=` を空にして `pytest tests` 957件パス、
   `ruff check .`／`ruff format --check .` パス、`ols.json`・凍結 synthetic CSV を
   再生成し `_meta.generated_at` 除外でコミット済みと完全一致。
   **ノートからの差分（実施時の判断）**:
   - `_common.py` の細分化は Step 1 では後回しにし、**別コミットで実施済み
     （2026-08-29）**: `benchmark/common/helpers.py` を `datasets_io.py`（`DATA_DIR` /
     `load_frozen_dataset` / `freeze_scenarios` / `run_freeze_cli`）・`dgp.py`
     （`imbalanced_cluster_groups` / `linear_predictor` / `correlated_design_matrix` /
     `apply_perfect_multicollinearity` / `hac_auto_lag` / `validate_choice` /
     `preview_dataset`）・`reference/extract.py`（`extract_coef_se`）へ分割し
     `helpers.py` を削除。`__init__.py` の re-export で利用側 import は無変更。
     `dgp_constants.py` は `dgp.py` へ統合せず据え置き。検証は Step 1 と同じ
     （`pytest` 957件・`ruff`・`ols.json`/凍結CSV の不変性）。
   - `_common.R` は `benchmark/_common.R` に**据え置き**。`.R` 側の `source(".../_common.R")`
     と一体で動かす方が安全なため、ステップ3（`.R` を `references/` へ移す回）で
     一緒に移動する。
   - `.devcontainer/devcontainer.json` の `PYTHONPATH` は**削除ではなく単一の
     リポジトリルート**（`${containerWorkspaceFolder}`）へ縮小。5ディレクトリ
     並記（bare import と同名衝突の温床）を廃し、`pyproject` の `pythonpath=["."]`
     と対称にした。devcontainer 内で `python benchmark/.../foo.py` 直実行も維持できる。
   - `compare_performance.py` の `_run_isolated`（自己サブプロセス再実行）を
     ファイルパス直接起動から `python -m benchmark.performance.compare_performance`
     （`cwd`=リポジトリルート）へ変更（パッケージ import を PYTHONPATH 非依存で
     解決）。加えて `benchmark_ols.yml` に `env: PYTHONPATH: ${{ github.workspace }}`
     を追加（親プロセスは現状 `working-directory: benchmark/performance` +
     `python compare_performance.py` のままのため）。`performance/` のトップレベル
     移動と `benchmark_ols.yml` の `-m` 化はノート通りステップ8。
   - 11個の `generate_*_fixtures.py` の `--output` 既定値と 4個の `freeze_*` の
     出力先既定値を、cwd 相対（`../../../tests/...`）から `Path(__file__).parents[N]`
     アンカーへ修正（`python -m` 実行で既定値のまま正しい場所へ書けるように）。
   - スクリプトは今後 **`python -m benchmark.<...>`（リポジトリルートから）**で実行する
     （各ディレクトリへ `cd` して `python foo.py` は不可）。`benchmark/README.md` に明記。
     SKILL.md・CLAUDE.md 等のパス参照更新はノート通りステップ9でまとめて行う。
2. **ドライバ骨格（手法非依存・1コミット）**: `driver.py`（`MethodBenchmarkSpec` /
   `build_fixture_json` / `run_fixture_cli`）、`common/reference/{r,meta,extract}.py`、
   `common/cluster_cases.py` を新設。まだどの手法も接続しない。
3. **OLS 移行（1コミット）**: `benchmark/linear/{datasets,constants,ols}.py` と
   `benchmark/linear/references/{statsmodels,r}.py` + `.R` を作成。旧
   `generate_ols_fixtures.py` / `generate_ols_crosscheck_fixtures.py` /
   `run_statsmodels_benchmark_linear.py` / `run_lm_*_benchmark.R` /
   `generate_linear_datasets.py` / `freeze_linear_datasets.py` を新構造へ。
   **不変性チェック**: `ols.json` / `ols_crosscheck.json` /
   `tests/fixtures/benchmarks/data/synthetic_*.csv` を再生成し、`_meta.generated_at`
   除外で移行前のコミット済みと完全一致することを確認（コミット済み成果物自体は
   更新しない）。`tests/test_ols*.py` の import 書き換え。`pytest tests -k ols` パス。
4. **WLS 移行** → 5. **Logit** → 6. **Probit** → 7. **IV / IV-GMM**。各手法で
   ステップ3と同じ不変性チェック。
8. **後片付け**: 旧 `benchmark/<系統>/fixtures/` ディレクトリ・旧ファイルの削除
   （`git rm`、参照ゼロを `grep` 確認）、`benchmark/freeze_datasets.py` →
   `regenerate_all.py`、`performance/` への `git mv`。
9. **文書更新**（9章）。

途中の手法が未移行の間は、新旧が並存する（`driver.py` は追加のみ、旧
`generate_*_fixtures.py` はそのまま動く）。手法単位で独立にコミット・確認できる。

---

## 9. 影響を受ける文書

| 文書 | 更新内容 |
|---|---|
| `CLAUDE.md` 3章 | リポジトリ構成図（`benchmark/` の中身、`performance/` の新設） |
| `CLAUDE.md` 10章 | 「`benchmark/` は `tests/` と別ライフサイクル」の記述を、依存の実態
  （statsmodels は `test` グループ、R のみ真に隔離）に合わせて補正 |
| `CLAUDE.md` 13章 | `docs/planning/specs/` に本ノートを追加した旨 |
| `engine/src/linear/CLAUDE.md` 等ネスト CLAUDE.md | `benchmark/` 配下のパス参照 |
| `.claude/skills/reference-benchmark/SKILL.md` | ディレクトリ構成節を全面改訂（Spec/ドライバ、新パス） |
| `.claude/skills/reference-benchmark` の `allowed-tools` | `Bash(python3:*)` の対象パス感覚（変更不要だが記述確認） |
| `benchmark/README.md` | 新構成、`performance/` が別ディレクトリになった旨 |
| `docs/spec/ols-spec.md` ほか手法 spec の「テスト」節 | `benchmark/...` のパス参照 |
| `docs/spec/inference-conventions.md` / `docs/planning/specs/iv-api-design.md` 等 | 同上 |
| `refactoring-candidates.md` | 本ノートが吸収した項目を削除（10章の一覧に従う） |
| `refactoring-issue231-progress.md` | 本 Initiative を新セクションとして追跡、進捗スナップショット |

---

## 10. 本ノートが吸収する `refactoring-candidates.md` 項目

**構造的重複として本再設計で解消（削除対象）**:
12, 13, 15, 16, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 39
（項目11 は `extract_coef_se` 切り出しで対応済み・コミット `28186ed`。移行時に
`benchmark/common/reference/extract.py` へ再配置）。

**関連するが本ノートの主目的ではない（移行時に機会があれば同時対応、無理なら残す）**:
- 17（コメント中の Issue 番号参照 ×10ファイル）: `git mv` で触るファイルが多いので
  ついでに削れるが、必須ではない。
- 22（R 冒頭の引数パース重複 ×4）: `common/reference/r.py` 側で吸収余地。
- 23（`run_lm_predict_crosscheck.R` の手法非依存化）: Issue #131/#132/#222 着手時の
  判断のまま。新構造では `linear/references/` に置くが汎用化はしない。
- 30（`run_glm_crosscheck_benchmark.R` 内の `scale_and_invert` 重複）: 同一 R ファイル
  内の重複。優先度低。
- 37（`suppressMessages` の欠落 ×3 R ファイル）: 5.2 で全 `.R` に適用と明記済み。

**本ノートの対象外（別 Issue）**:
- 40, 41（`performance/` = 性能比較ツールの共通化・表示桁）: D5 で対象外。
- 43（`tests/` の手法別ディレクトリ分割）: テストスイート側のディレクトリ構成の話。
- 38（`script_dir` → `_common.R` の `source()`）: 既に「対応不要」判定済み。

---

## 11. 未解決・着手前に要確認

1. ~~**疑似グループ列の凍結CSV焼き込み（項目12）**~~: 決定済み。(c) の
   `run_cluster_case` が実行時付与、`ExtraCase` に載せる方式で進める。焼き込みは
   移行完了後に別途判断（DGP 関数にテスト用ラベル列を混ぜない方針とのトレードオフ）。
2. ~~**`benchmark/common/` の粒度**~~: 決定済み。4章の分割で進める。実装前後で
   統廃合が必要なら再考。
3. ~~**`ExtraCase` の表現力**~~: 決定済み。IV 移行（ステップ7）で実証する。書けなければ
   Spec に系統別の追加フィールドを許す。
4. ~~**Issue 番号**~~: 決定済み。**Issue #231 のサブ Issue**として起票する。
   `docs/plan.md` のフェーズ表には載せない（フェーズ外の保守作業）。

---

## 12. 期待される効果（まとめ）

- `sys.path.insert` 全廃・同名衝突の構造的排除（項目71 の再発防止）。
- 手法追加時に書くのは「`datasets.py` の DGP + `constants.py` + `<手法>.py` の Spec
  +（新規リファレンスなら）アダプタ1本」だけになり、ループ/CLI/`_meta`/クラスター
  ケース/`_run_r`/`normalize_names` は共通コードを再利用。
- `refactoring-candidates.md` の約20項目を1つの構造変更でまとめて解消。
- perf 比較ツールとフィクスチャ生成ツールが物理的にも分離。
- `tests/` は場所も名前も不動、固定成果物（CSV/JSON）も不動 → 既存テストと
  フィクスチャが壊れるリスクを最小化しつつ移行できる。
