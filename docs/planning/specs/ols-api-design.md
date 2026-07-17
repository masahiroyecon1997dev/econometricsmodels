# OLS API・オプション設計

GitHub Issue #2「OLS: API・オプション設計」の完了条件（設計をdocsに書き出す）を満たすための、
確定済み設計のまとめ。パラメータ以外の内部実装の決定事項は
[`ols-implementation-notes.md`](./ols-implementation-notes.md) を参照。

**ステータス**: 設計確定済み。Rust/PyO3境界（`engine_pybind`）までは実装済み
（[`engine_pybind/src/linear/ols.rs`](../../../engine_pybind/src/linear/ols.rs)）。
`python_package`側のラッパー（`OLS` / `OlsResults`クラス）は未実装（Issue #15）。

## 1. 全体構成（3層）

```
python_package (Issue #15, 未実装)      engine_pybind (実装済み)          engine (未実装)
┌─────────────────────────┐   ┌──────────────────────────┐   ┌─────────────────┐
│ OLS(data, y, x, options) │──▶│ fit_ols(data, y, x,      │──▶│ 正規方程式ソルバー│
│   .fit() -> OlsResults   │   │          options)        │   │ 標準誤差計算 等   │
│ OLSOptions（_lib再輸出） │   │ OLSOptions (#[pyclass])  │   │ (Issue #6〜#12)  │
└─────────────────────────┘   └──────────────────────────┘   └─────────────────┘
```

- CLAUDE.md 2章の非交渉事項（List渡し・オブジェクト渡し・formula不採用）は
  **python_package層のユーザー向けAPI**に対する規約。
- PyO3境界（`engine_pybind`）がオプションを`#[pyclass]`構造体で受け取るのは、
  同じ規約の延長であり抵触しない（性能差はほぼなく、手法が増えた際のシグネチャ肥大化を防ぐため）。

## 2. `y` / `x` のシグネチャ

- `y: str`（単一列名）、`x: list[str]`（複数列名）
- `y`を`list[str]`にしない理由: Phase1〜6（VAR等の一部時系列手法を除く）でyは常に1変数。
  `list[str]`にすると「長さ1であること」を全推定関数が実行時検証する必要が生じるため、型で表現する。
  真に多変量なyが必要な手法（VAR等）が出てきた場合は、その手法だけ`y: list[str]`にする。
- CLAUDE.md 2章の例は`y="y_col", x=["x1", "x2"]`に修正済み。

## 3. `OLSOptions`

`engine_pybind/src/linear/ols.rs`の`#[pyclass] OLSOptions`として実装済み
（`hac_lags` / `time_col`、および`cov_type`の`"hac"`はIssue #3で確定した追加分で、コードへの反映は
実装issue #11で行う）。
python_package層はこれを独自クラスとして再定義せず、`_lib`から再輸出する
（`engine_pybindからの呼び出しを薄くラップする。計算ロジックをPython側に持たない`、
`.claude/rules/python-style.md`参照）。

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `cov_type` | `str` | `"classical"` | 標準誤差の種別。`"classical"` / `"hc0"`〜`"hc3"` / `"cluster"` / `"hac"`（大小無視） |
| `include_intercept` | `bool` | `True` | `True`の場合、engine側が設計行列の先頭に定数列（全て1）を自動追加する |
| `confidence_level` | `float` | `0.95` | 信頼区間の信頼水準、`(0, 1)`の範囲。`alpha`（有意水準=0.05側を指す慣習語）ではなくこちらを正式名とする |
| `cluster_col` | `str \| None` | `None` | `cov_type="cluster"`のときのグループキー列名。`data`内の列を指す（別配列としては渡さない）。それ以外の`cov_type`では無視される |
| `hac_lags` | `int \| None` | `None` | `cov_type="hac"`のときのラグ数（バンド幅）。`None`なら経験則で自動計算。詳細は[`ols-standard-errors.md`](./ols-standard-errors.md)（Issue #3） |
| `time_col` | `str \| None` | `None` | `cov_type="hac"`のときの時系列順序列。`None`なら`data`の行順を使用。詳細は[`ols-standard-errors.md`](./ols-standard-errors.md)（Issue #3） |

補足:
- `include_intercept=True`のとき`x`に`"const"`という列名があるとエラー（自動追加する定数項と衝突するため）。
- `include_intercept=True`かつユーザーが`x`に自前で定数列を含めていた場合は、専用の重複検出はせず、
  結果として生じる多重共線性を既存の`SingularMatrix`（`ComputationError`）にそのまま乗せる。
- `confidence_level`は`fit()`実行時に一度だけ使われ、結果（coef_table等）に固定して含める。
  再計算用の可変引数（例: `conf_int(alpha=...)`）は提供しない。

## 4. Rust/PyO3境界のインターフェース

```rust
// engine_pybind/src/lib.rs
#[pyfunction]
fn fit_ols(
    data: PyDataFrame,
    y: String,
    x: Vec<String>,
    options: OLSOptions,
) -> PyResult<PyObject>
```

- `data`はpolars DataFrame（Arrow経由、ゼロコピー）。
- 現状の実装スコープ: 「パラメータの受け口」（`data`/`y`/`x`/`options`の検証とfaer行列への変換、
  `engine_pybind/src/linear/ols.rs::extract_ols_input`）のみ。実際の推定計算（正規方程式ソルバー・
  標準誤差計算等）は`engine`側の別issue（#6〜#12）に委ねており、現状は`ComputationError`を返して打ち切る。
- 検証エラーはすべて`ValidationError`、計算過程で発覚する問題（特異行列等）は`ComputationError`
  （`.claude/rules/rust-style.md`「エラーハンドリング」、詳細対応表は`ols-implementation-notes.md`参照）。

## 5. Python向け出力方針

- **structured onlyとし、`summary()`（テキスト整形）は作らない**。economiconのGUIエンジンという用途上、
  テキスト表示は不要。人間向けの見せ方が必要ならGUI側/Python側に委ねる。formula不採用の判断
  （対話的な使い方を最初から重視しない）とも一貫。
- 係数テーブルにpolars DataFrameは使わない。少数行のテーブルではDataFrameの利点がなく、
  REST APIでJSON化する前提だと変換の中間ステップが増えるだけのため。
- **Rust/engine_pybind側の責務は配列＋名前リストを返すところまで**
  （`params`, `std_errors`, `t_stats`, `p_values`, `conf_int`, `param_names`）。
  テーブル組み立てはpython_package層で行う。
- python_package層で用意する2形式:
  1. `coef_table`（`list[dict]`、行指向）: REST APIレスポンスにほぼそのまま使える。
     アプリ用途が主なので**先に実装**。
  2. `params` / `std_errors`等（統計量ごとの`dict`）: `results.params["educ"]`のような
     O(1)単一パラメータ取り出し用。**後回しでよい**。
- `__repr__`のみ最小限用意する案は出たが、未確定（採否は保留）。

## 6. その他の確定事項（API設計に直結するもの）

- 欠損値（NaN/無限大）は常にエラー。listwise deletion等の自動除外はしない。
- 検定分布は**t分布**（正規分布ではない）。
- `cov_type`が未指定の場合のデフォルトは**classical**。
- `cov_type`がHC系/clusterの場合、F検定も**ロバストWald検定**に切り替える（常に古典的F検定のままにはしない）。

## 7. 既知の不整合（本設計確定に伴い後続issueで解消が必要）

- `tests/api_tests/test_ols.py`は本設計確定前の草案段階のテストで、
  `OLS(df, y="y", x=["x1", "x2"], cov_type=cov_type)`のように`cov_type`等をフラットな
  キーワード引数として渡しており、本ドキュメントの「オプションはオブジェクト（`OLSOptions`）渡し」
  の方針と一致しない。また`res.summary()`を呼んでおり、5章の「summary()は作らない」方針とも
  一致しない。python_packageの`OLS`/`OlsResults`実装（Issue #15）・テスト整備（Issue #19）の際に
  書き直しが必要。
- `docs/spec/01_ols.md`は本ドキュメントより前に書かれた古い草案で、`add_constant`（→`include_intercept`）、
  `summary()`出力、`weights`（WLS用、OLS Phase1のスコープ外）等、現在の設計と整合しない箇所が
  複数ある。CLAUDE.md 3章のリポジトリ構成にも`docs/spec/`は現れず、現行の`docs/planning/`体系に
  統合されていない。取り扱い（統合・アーカイブ・削除）は別途ユーザー判断が必要。
