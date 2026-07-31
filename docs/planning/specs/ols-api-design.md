# OLS API・オプション設計

OLSのAPI・オプションに関する確定済み設計のまとめ。パラメータ以外の内部実装の決定事項は
[`ols-implementation-notes.md`](./ols-implementation-notes.md) を参照。

**ステータス**: 設計確定済み。`engine`・`engine_pybind`・`python_package`（`OLS` / `OlsResults`
クラス、[`python_package/econometricsmodels/linear/ols.py`](https://github.com/masahiroyecon1997dev/econometricsmodels/blob/main/python_package/econometricsmodels/linear/ols.py)）
まで実装済み。

## 1. 全体構成（3層）

```
python_package (実装済み)              engine_pybind (実装済み)          engine (実装済み)
┌─────────────────────────┐   ┌──────────────────────────┐   ┌─────────────────┐
│ OLS(data, y, x, options) │──▶│ fit_ols(data, y, x,      │──▶│ 正規方程式ソルバー│
│   .fit() -> OlsResults   │   │          options)        │   │ 標準誤差計算 等   │
│ OLSOptions（_lib再輸出） │   │ OLSOptions (#[pyclass])  │   │                  │
│                          │   │ OLSResult (#[pyclass])   │   │                  │
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

`engine_pybind/src/linear/ols.rs`の`#[pyclass] OLSOptions`として実装済み。
python_package層はこれを独自クラスとして再定義せず、`_lib`から再輸出する
（`engine_pybindからの呼び出しを薄くラップする。計算ロジックをPython側に持たない`、
`.claude/rules/python-style.md`参照）。

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `cov_type` | `str` | `"classical"` | 標準誤差の種別。`"classical"` / `"hc0"`〜`"hc3"` / `"cluster"` / `"hac"`（大小無視） |
| `include_intercept` | `bool` | `True` | `True`の場合、engine側が設計行列の先頭に定数列（全て1）を自動追加する |
| `confidence_level` | `float` | `0.95` | 信頼区間の信頼水準、`(0, 1)`の範囲。`alpha`（有意水準=0.05側を指す慣習語）ではなくこちらを正式名とする |
| `cluster_col` | `str \| None` | `None` | `cov_type="cluster"`のときのグループキー列名。`data`内の列を指す（別配列としては渡さない）。それ以外の`cov_type`では無視される |
| `hac_lags` | `int \| None` | `None` | `cov_type="hac"`のときのラグ数（バンド幅）。`None`なら経験則で自動計算。詳細は[`ols-standard-errors.md`](./ols-standard-errors.md) |
| `time_col` | `str \| None` | `None` | `cov_type="hac"`のときの時系列順序列。`None`なら`data`の行順を使用。詳細は[`ols-standard-errors.md`](./ols-standard-errors.md) |

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
) -> PyResult<OLSResult>
```

- `data`はpolars DataFrame（Arrow経由、ゼロコピー）。
- `engine_pybind/src/linear/ols.rs::fit`が「パラメータの受け口」
  （`data`/`y`/`x`/`options`の検証とfaer行列への変換）から`engine::linear::ols::OlsInput::from_columns`
  ＋`OlsEstimator::fit`の呼び出しまでを一気通貫で行う。返り値は`#[pyclass] OLSResult`
  （`params`/`std_errors`/`t_stats`/`p_values`/`conf_lower`/`conf_upper`/`param_names`/`residuals`/
  適合度統計量（`r_squared`等）を`#[pyo3(get_all)]`で公開する構造体）。
- 検証エラーはすべて`ValidationError`、計算過程で発覚する問題（特異行列等）は`ComputationError`
  （`.claude/rules/rust-style.md`「エラーハンドリング」、詳細対応表は`ols-implementation-notes.md`参照）。

## 5. Python向け出力方針

- **structured onlyとし、`summary()`（テキスト整形）は作らない**。economiconのGUIエンジンという用途上、
  テキスト表示は不要。人間向けの見せ方が必要ならGUI側/Python側に委ねる。formula不採用の判断
  （対話的な使い方を最初から重視しない）とも一貫。
- 係数テーブルにpolars DataFrameは使わない。少数行のテーブルではDataFrameの利点がなく、
  REST APIでJSON化する前提だと変換の中間ステップが増えるだけのため。
- **Rust/engine_pybind側の責務は配列＋名前リストを返すところまで**
  （`params`, `std_errors`, `t_stats`, `p_values`, `conf_lower`/`conf_upper`, `param_names`,
  `residuals`。`OLSResult`は`conf_int`を`conf_lower`/`conf_upper`に分割している
  （engine内部の表現と揃え、pyo3実装を簡潔にするため）。適合度統計量（`r_squared`等）も
  含める）。テーブル組み立てはpython_package層で行う。
- python_package層は以下の2形式を提供する:
  1. `coef_table()`（`list[dict]`、行指向）: REST APIレスポンスにほぼそのまま使える形。
  2. `params` / `std_errors`等（統計量ごとの`dict`）: `results.params["educ"]`のような
     O(1)単一パラメータ取り出し用。
- `__repr__`のみ最小限用意する案は出たが、未確定（採否は保留）。

## 6. その他の確定事項（API設計に直結するもの）

- 欠損値（NaN/無限大）は常にエラー。listwise deletion等の自動除外はしない。
- 検定分布は**t分布**（正規分布ではない）。
- `cov_type`が未指定の場合のデフォルトは**classical**。
- `cov_type`がHC系/clusterの場合、F検定も**ロバストWald検定**に切り替える（常に古典的F検定のままにはしない）。

## 7. `predict()`（Issue #86、設計確定・実装済み）

5章の「薄いラッパー」方針（`summary()`・`to_frame()`等は作らない）は維持したまま、実測値との比較・
新規データへの予測という実務上の需要に応えるため、これは例外的にスコープに含める。

**Logit（nonlinear系統）との命名整合**: `LogitEstimator::predict()`（`nonlinear-api-design.md`6章、
実装済み）は引数を取らず学習データに対する予測確率のみを返し、新規データ対応は将来
`predict()`と同じ名前のまま引数を追加する形で拡張する前提になっている（`statsmodels`の
`results.predict(exog=None)`と同じ、1メソッド+オプション引数という設計）。OLSも当初案
（`fitted_values`プロパティ + `predict(new_data)`必須引数メソッドという2メソッド分離案）から
変更し、この設計に揃える。Logit側は既に実装・マージ済みでリネームのコストが高い一方、OLSは
未実装のため、OLS側をLogitの命名に合わせる。

- **呼び出し場所**: `OlsResults.predict(new_data: pl.DataFrame | None = None) -> list[dict[str, float]]`
  （`OLS`側ではない）。`OLS`は「fit前の設定」を保持するだけのステートレスな値であり
  （`OLS.fit()`は`self`に何も書き込まず、都度新しい`OlsResults`を返す設計）、`predict`が必要と
  する学習済み係数は`OlsResults`だけが持つ。`OLS`側に置くと`OLS`が「fit前の設定」と「fit後の
  状態」の二重の責務を持つステートフルなオブジェクトに変わってしまうため採用しない。
- **`new_data=None`（デフォルト）**: 学習データに対する予測値 `ŷ = y - ε̂` を返す
  （旧案の`fitted_values`が担っていた役割）。`OLSResult`は効率化のため`fit()`時に計算した
  値を内部フィールドとして保持してよいが、Python側に独立したプロパティとしては公開しない
  （公開APIは`predict()`の1メソッドのみに統一する）。
- **`new_data`指定時**: 新規データに対する予測値を返す（out-of-sample）。
- **戻り値**: どちらの場合も行指向の`list[dict[str, float]]`（`coef_table()`と同じ形式方針）で
  統一する。点予測のみの現段階では各要素は1キー（予測値）のみだが、将来信頼区間・予測区間を
  追加する場合に`conf_lower`/`conf_upper`等のキーを追加できるようにするため、単純な`list[float]`
  ではなくこの形にする。
- **新規データの列の対応付け**: `new_data`はfit時に`x`で渡したのと同じ列名を持つ列を含む必要が
  ある（列名でマッチング、列順は問わない）。`include_intercept=True`でfitした場合、定数項の列は
  `new_data`に含める必要はなく、`fit()`時と同様に自動で付加される。
- **層構成**: `engine_pybind`側が、`new_data`が`Some`の場合はfit結果の`param_names`
  （`include_intercept`時は先頭の`"const"`を除いたもの）を使って`new_data`から必要な列を
  `column_extraction::extract_f64_column`で抽出し、`engine`側の新規関数（列データを渡して
  `new_X・params`を計算する想定、`const`列の自動付加を含む）を呼ぶ。`None`の場合は`fit()`時に
  計算済みの値をそのまま返す。`engine`が設計行列の組み立てを担い`engine_pybind`は列抽出のみに
  留める、という既存の責務分担（`OlsInput::from_columns`と同じパターン）をそのまま踏襲する。
  具体的な関数シグネチャは実装時に確定する。
- **エラーハンドリング**: 列不足・型不一致・NaN/無限大は、既存の`ValidationError`の枠組みに
  そのまま乗せる（`fit()`の列抽出と同じ検証パターンを再利用し、専用のエラーバリアントは
  新設しない）。
- **スコープ外**: 信頼区間・予測区間（点予測のみ）、WLSへの適用（`WlsEstimator`は内部で
  `OlsEstimator`を使う設計のため同じパターンを流用できる見込みだが、別issueで扱う）。
- **今後の課題**: Logit側の新規データ対応（同issueの別issueでトラッキング）も、この統一設計
  （`predict(new_data=None)`）に揃える形で実装することになる。
