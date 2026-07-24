# WLS API・オプション設計

WLS（Weighted Least Squares）のAPI・オプションに関する設計案。標準誤差・適合度統計量の
重み付き定義の確定は[`wls-standard-errors.md`](./wls-standard-errors.md)（Issue #34）に委ねる。

**ステータス**: 設計提案中（Issue #33）。本ドキュメントは2026-07-24時点の
[`OLS実装`](../../../engine/src/linear/ols.rs)（`engine/src/linear/ols.rs`,
`engine_pybind/src/linear/ols.rs`, `python_package/econometricsmodels/linear/ols.py`）を
出発点にしている。OLSは今後リファクタリングされる可能性が高いため（進行中のフォローアップ
Issue #27〜#31）、実装着手時には改めて最新のOLS実装を確認し、本ドキュメントとの差分があれば
本ドキュメント側を更新すること。

## 0. 前提として確定済みの3点（Issue #33起票時にユーザー承認済み）

1. **重みの指定方法**: `WLSOptions.weight: str`（`data`内の列名参照）。生ベクトル直接渡しは不採用。
2. **重みの意味論**: analytic weight（分散の逆数に比例、正規化不要）。frequency weight/probability
   weightはPhase1では対象外。
3. **engineでの共通化方針**: `X, y`を`sqrt(w)`倍してから既存のOLS正規方程式ソルバー
   （`OlsEstimator::fit`）にそのまま渡す変換方式。

## 1. 全体構成（3層）

```
python_package (未実装)                engine_pybind (未実装)            engine (未実装)
┌─────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────┐
│ WLS(data, y, x, options) │──▶│ fit_wls(data, y, x,      │──▶│ OlsInput::            │
│   .fit() -> WlsResults   │   │          options)        │   │  from_columns_weighted│
│ WLSOptions（_lib再輸出） │   │ WLSOptions (#[pyclass])  │   │ + 既存OlsEstimator::   │
│                          │   │ WLSResult (#[pyclass])   │   │  fit（無変更で再利用）  │
└─────────────────────────┘   └──────────────────────────┘   └──────────────────────┘
```

OLSと同じ3層構成・同じ「List渡し＋オブジェクト渡し」規約（CLAUDE.md 2章）に従う。
`engine_pybind`境界がオプションを`#[pyclass]`構造体で受け取る位置づけもOLSと同じ
（[`ols-api-design.md`](./ols-api-design.md) 1章参照）。

## 2. `y` / `x` / `weight` のシグネチャ

- `y: str`, `x: list[str]` はOLSと同一。
- `weight`は**トップレベル引数にはせず、`WLSOptions.weight: str`として`options`側に含める**。
  - 理由: `weight`は`data`内の列を参照するという性質上、既存の`cluster_col`/`time_col`
    （どちらも`OLSOptions`内で「dataの列名を指す」パターンを既に採用している。
    [`ols-api-design.md`](./ols-api-design.md) 3章）と同じ扱いにするのが一貫している。
    `y`/`x`のように「モデルの構造そのもの」を規定する引数ではなく、`cov_type`と同様に
    「推定方法の設定の一部」として扱う。

## 3. `WLSOptions`

`OLSOptions`（[`ols-api-design.md`](./ols-api-design.md) 3章）のフィールドをすべて引き継ぎ、
`weight`を追加する。

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `weight` | `str` | **なし（必須）** | 重み列の列名。`data`内の列を指す。analytic weight（分散の逆数に比例）として扱う |
| `cov_type` | `str` | `"classical"` | OLSと同じ。重み付きの計算式は[`wls-standard-errors.md`](./wls-standard-errors.md)で確定 |
| `include_intercept` | `bool` | `True` | OLSと同じ |
| `confidence_level` | `float` | `0.95` | OLSと同じ |
| `cluster_col` | `str \| None` | `None` | OLSと同じ |
| `hac_lags` | `int \| None` | `None` | OLSと同じ |
| `time_col` | `str \| None` | `None` | OLSと同じ |

補足:
- `weight`はデフォルト値を持たない必須フィールドとする（重みを使わない場合は`OLS`を使うべきであり、
  `WLS`という別クラスを用意している以上、暗黙のフォールバックは提供しない）。PyO3の
  `#[pyo3(signature = ...)]`ではデフォルトなし引数をデフォルトあり引数より前に置く必要があるため、
  `weight`をシグネチャの先頭に置く。
- 重みの検証（0以下・NaN/Inf）は`ValidationError`とする。NaN/Infは既存の
  `column_extraction::extract_f64_column`が既に検出するため、追加が必要なのは
  **0以下の値の検証のみ**（3.1節参照）。
- `include_intercept=True`のときの`"const"`列衝突チェック等、OLSの`fit()`受け口が行っている
  検証（[`ols-api-design.md`](./ols-api-design.md) 3章補足）はそのまま踏襲する。

### 3.1 重みの検証: 0以下の値はエラー

- **0以下（0を含む）・NaN・無限大の重みは常にエラー**とし、該当観測を自動的に落とす
  （ゼロ重み＝実質除外として許容する）ことはしない。OLSの欠損値ポリシー
  （常にエラー、自動除外はしない。[`ols-implementation-notes.md`](./ols-implementation-notes.md)
  「欠損値の扱い」）と同じ考え方を重みにも適用する。
- ゼロ重みの許容（観測を明示的に無視する手段としての活用）が将来必要になった場合は、
  別issueで検討する。

## 4. engineでの実装方針

### 4.1 「sqrt(w)変換→既存OLSソルバー」の具体化

`OlsEstimator::fit`（正規方程式ソルバー・標準誤差・適合度統計量の計算本体）は**無変更のまま
再利用**する。変換が必要なのは`OlsInput`の組み立て（`OlsInput::from_columns`）の部分のみ。

**注意点（切片列の重み付け）**: 単純に「`x_columns`とyを先に`sqrt(w)`倍してから既存の
`OlsInput::from_columns(weighted_y, weighted_x_columns, ...)`を呼ぶ」という実装は**誤り**。
`from_columns`が内部で追加する切片列（すべて1.0）は重み付け前の値のままになってしまい、
`include_intercept=true`のときの設計行列が数学的に不正になる（切片列も`sqrt(w_i)`倍されて
いなければならない）。

したがって、重み変換は`OlsInput`の行列組み立てそのものの中で行う必要がある。具体的には
`engine/src/linear/ols.rs`に以下のいずれかの形で手を入れる:

- `OlsInput::from_columns`の内部実装を、`weights: Option<&[f64]>`を取る非公開ヘルパーに
  委譲する形にリファクタリングする。既存の`from_columns`（公開シグネチャ・挙動は無変更）は
  内部で`weights=None`としてこのヘルパーを呼ぶ。
- WLS用に新しい公開コンストラクタ（例: `OlsInput::from_columns_weighted(y, x_columns, x_names,
  include_intercept, dep_var_name, weights: &[f64])`）を追加し、同じヘルパーを`Some(weights)`で呼ぶ。
- ヘルパーは、`weights`が`Some`の場合、設計行列・yの各行に`sqrt(w_i)`を掛けて組み立てる
  （切片列も含む）。`None`の場合は現状と全く同じ（`sqrt(1.0) = 1.0`と等価）。

この設計により、**「重みが全て1のときWLSはOLSと数値的に完全一致する」という不変条件が、
テストで検証するまでもなく構造的に保証される**（同じコードパスを通るため）。Issue #37の
不変条件テストは、この構造的保証が壊れていないことの回帰検知として位置づけられる。

### 4.2 エラー型: `OlsError`をそのまま再利用し、新しいバリアントを1つ追加する

WLSは`OlsEstimator::fit`をそのまま呼ぶため、`DimensionMismatch` /
`InsufficientObservations` / `SingularMatrix` 等、既存の`OlsError`バリアントがそのまま
WLSにも当てはまる。重み固有のエラー（0以下の重み）のためだけに別の`WlsError`型を新設すると、
既存バリアントの重複定義・`engine_pybind`側の変換関数の二重化が発生する。

**設計案**: `OlsError`に新バリアント`NonPositiveWeight { row: usize, weight: f64 }`を追加し、
WLS・OLS共通のエラー型として使い続ける。デメリット（「OLS専用のはずの型にWLSの都合が混入する」）
はあるが、後続でGLS等が増えたときに同じパターンが繰り返されるようなら、その時点で
`linear/common.rs`への切り出しを検討する（YAGNI、`rust-style.md`「ファイル・ディレクトリ構成」）。

### 4.3 残差の扱い: 元スケール（unweighted）の残差を公開する

`OlsEstimator::residuals()`は、渡された`OlsInput`がWLS用に変換済み（重み付き）であれば、
その残差 `ε̃_i = sqrt(w_i)(y_i - x_i'β̂)` = **重み付き残差**（statsmodelsでいう`.wresid`相当）を返す。

Phase1では、ユーザー向けに公開する`residuals`は**元スケール（unweighted）の残差
`ε_i = y_i - x_i'β̂`**（statsmodelsの`.resid`相当）とする。理由: 残差プロット等の診断用途では
元スケールの残差の方が直感的で、OLSの`residuals`（`y - Xβ̂`）とも定義が揃う。

実装上は、`OlsEstimator::residuals()`（重み付き）をそのまま使うのではなく、WLSのentry point
（`engine/src/linear/wls.rs`想定）側で元の（重み変換前の）`y`・`x_columns`と`estimator.params()`
から改めて`y_i - x_i'β̂`を計算する。重み付き残差（`.wresid`相当）はPhase1では公開しない
（将来、加重残差プロット等の需要が出た場合に追加を検討）。

### 4.4 適合度統計量・標準誤差: 変換後データに対するOLSの計算式がそのまま重み付き版になる

`r_squared` / `r_squared_adj` / `f_statistic` / `log_likelihood` / `aic` / `bic`、および
classical / HC0-3 / HAC / cluster の標準誤差は、すべて`OlsEstimator::fit`が変換後の
`OlsInput`（重み付き`X̃, ỹ`）に対して計算したものをそのまま使う。

statsmodelsの`WLS`も内部的に同じ変換方式（`wexog = sqrt(weights) * exog`,
`wendog = sqrt(weights) * endog`）で実装されており、`.rsquared`等の適合度統計量も変換後データの
`wexog`/`wendog`ベースで計算される。したがって、この「変換後データにOLSの計算式をそのまま
適用する」設計は、独自の重み付き公式を新たに導出する必要がなく、**そのままstatsmodelsとの
数値一致が期待できる**。[`wls-standard-errors.md`](./wls-standard-errors.md)（Issue #34）は、
新しい計算式の導出ではなく、この前提の確認とベンチマークでの実証が主な作業になる見込み。

## 5. Rust/PyO3境界のインターフェース

```rust
// engine_pybind/src/lib.rs
#[pyfunction]
fn fit_wls(
    data: PyDataFrame,
    y: String,
    x: Vec<String>,
    options: WLSOptions,
) -> PyResult<WLSResult>
```

`OLSOptions`/`OLSResult`と同様、`WLSOptions`/`WLSResult`は`engine_pybind/src/linear/wls.rs`に
`#[pyclass(module = "econometricsmodels._lib")]`として定義する
（`.claude/rules/rust-style.md`「言語方針」のmodule属性要件）。`WLSResult`のフィールド構成は
`OLSResult`と同一（4.3節の通り`residuals`は元スケール）。

## 6. Python向け出力方針

OLSと同じ（[`ols-api-design.md`](./ols-api-design.md) 5章）。`structured only`、
`coef_table()` / 統計量ごとの`dict`の2形式、DataFrame不使用。

## 7. その他の確定事項

- 重みが全て1のとき、`WLS`の推定値・標準誤差・適合度統計量は`OLS`と数値的に完全一致する
  （4.1節の構造的保証）。
- 欠損値（NaN/無限大）は常にエラー。検定分布はt分布。`cov_type`未指定時のデフォルトは
  classical。ロバストF検定への切替方針。これらはすべてOLSと同じ
  （[`ols-api-design.md`](./ols-api-design.md) 6章）。

## 8. 未確定・後続issueで扱う事項

- 標準誤差（classical/HC0-3/HAC/cluster）・適合度統計量の重み付き定義の最終確認とベンチマーク照合
  → Issue #34, #43
- `WLSOptions`と`OLSOptions`のフィールド重複を将来的に共通化するか（trait化・共通構造体化）
  → 本ドキュメントでは見送り、GLS実装時に3つ目の実例が揃った時点で判断する（YAGNI）
