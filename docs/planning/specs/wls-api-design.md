# WLS API・オプション設計

WLS（Weighted Least Squares）のAPI・オプションに関する設計案。標準誤差・適合度統計量の
重み付き定義の確定は[`wls-standard-errors.md`](./wls-standard-errors.md)（Issue #34）に委ねる。

**ステータス**: 設計提案中（Issue #33）。本ドキュメントは2026-07-24時点の
[`OLS実装`](../../../engine/src/linear/ols.rs)（`engine/src/linear/ols.rs`,
`engine_pybind/src/linear/ols.rs`, `python_package/econometricsmodels/linear/ols.py`）を
出発点にしている。OLSは今後リファクタリングされる可能性が高いため（進行中のフォローアップ
Issue #27〜#31）、実装着手時には改めて最新のOLS実装を確認し、本ドキュメントとの差分があれば
本ドキュメント側を更新すること。

## 0. 前提として確定済みの事項

1. **重みの指定方法**: `weight: str`という**`y`/`x`と同列のトップレベル引数**（`data`内の列名参照）。
   生ベクトル直接渡しは不採用。当初は`WLSOptions.weight`として`options`側に含める案だったが、
   レビューで「`weight`はOLSの`cluster_col`/`time_col`のような条件付き・任意設定ではなく、
   `y`/`x`と同様にモデルを規定する必須データである」という指摘を受け、トップレベル引数に変更した
   （1章参照）。
2. **重みの意味論**: analytic weight（分散の逆数に比例、正規化不要）。frequency weight/probability
   weightはPhase1では対象外。
3. **engineでの共通化方針**: `X, y`を`sqrt(w)`倍してから既存のOLS正規方程式ソルバー
   （`OlsEstimator::fit`）にそのまま渡す変換方式。
4. **オプション型の共有**: `weight`を切り離した結果、WLSに必要な設定項目
   （`cov_type`/`include_intercept`/`confidence_level`/`cluster_col`/`hac_lags`/`time_col`）は
   `OLSOptions`と完全に一致する。専用の`WLSOptions`型は新設せず、**`OLSOptions`をそのまま使う**
   （2章参照）。

## 1. `y` / `x` / `weight` の位置づけ: 「モデルを規定する必須データ」と「推定方法の設定」の分離

`OLSOptions`の既存フィールドは性質が異なる2種類が混在している。

- **条件付き・デフォルトあり**（`cluster_col`, `time_col`, `hac_lags`）: `cov_type`の選択に応じて
  使われたり無視されたりする、真の「推定方法の設定」。
- **常に使われる必須データ**（`y`, `x`）: モデルそのものを定義する列参照で、デフォルトを持たず、
  常に使われる。

`weight`はanalytic weightとしてWLSが常に必要とする値であり、後者（`y`/`x`と同格）に分類すべきで、
前者のパターン（`cluster_col`/`time_col`）を安易に流用すべきではなかった。「デフォルトを持たない
必須フィールドをOptionsクラスに置く」こと自体が、Optionsという概念（任意の設定）と矛盾している。

この分類は今後のFE（`entity_id`）・IV（`instruments`）にも同様に適用する。パネルの個体ID、
操作変数リストのような必須データ列は`y`/`x`と同列のトップレベル引数とし、条件付き・任意の設定
のみを各手法の`Options`型に置く。

## 2. 全体構成（3層）

```
python_package (未実装)                       engine_pybind (未実装)                engine (未実装)
┌────────────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────┐
│ WLS(data, y, x, weight, options)│──▶│ fit_wls(data, y, x,      │──▶│ OlsInput::            │
│   .fit() -> WlsResults          │   │   weight, options)       │   │  from_columns_weighted│
│ OLSOptions（_lib再輸出、       │   │ OLSOptions（ols.rsのものを│   │ + 既存OlsEstimator::   │
│  WLS専用のOptions型は新設しない）│   │  そのまま使う。新設しない）│   │  fit（無変更で再利用）  │
│                                  │   │ WLSResult (#[pyclass])   │   │                       │
└────────────────────────────────┘   └──────────────────────────┘   └──────────────────────┘
```

OLSと同じ「List渡し＋オブジェクト渡し」規約（CLAUDE.md 2章）に従う。`weight`は`y`と同じ
「dataの列名を指す必須のstr」という扱いであり、`options`の型自体はOLSと共有する。

## 3. `weight`引数とOptionsの共有

- `weight: str`は`y`と同じ位置づけの必須引数。デフォルトを持たない。PyO3の
  `#[pyo3(signature = ...)]`ではデフォルトなし引数をデフォルトあり引数より前に置く必要があるため、
  `fit_wls(data, y, x, weight, options)`のように`options`より前に置く。
- `options: OLSOptions`は`ols.rs`で定義済みの型をそのまま使う。`engine_pybind/src/linear/wls.rs`は
  新しい`#[pyclass]` Options型を定義しない（`use super::ols::OLSOptions;`で参照する）。
  python_package層も`WLSOptions`を新設せず、`OLSOptions`を再輸出してそのまま使う。
- 重みの検証（0以下・NaN/Inf）は`ValidationError`とする。NaN/Infは既存の
  `column_extraction::extract_f64_column`が`weight`列にもそのまま適用されることで検出されるため、
  追加が必要なのは**0以下の値の検証のみ**（3.1節参照）。
- `include_intercept=True`のときの`"const"`列衝突チェック等、OLSの`fit()`受け口が行っている
  検証（[`ols-api-design.md`](./ols-api-design.md) 3章補足）はそのまま踏襲する。加えて、
  `weight`列についても`y`/`x`との重複チェック（`weight == y`、`x.contains(weight)`）を
  `wls.rs`の受け口で行う（`y`/`x`間の重複チェックと同じパターン。誤って同じ列を複数の役割に
  指定してしまう典型的なミスを早期に分かりやすいエラーで防ぐ）。

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
    weight: String,
    options: OLSOptions,
) -> PyResult<WLSResult>
```

`options`は`ols.rs`の`OLSOptions`をそのまま使う（2章・3章参照）。`WLSResult`のみ
`engine_pybind/src/linear/wls.rs`に`#[pyclass(module = "econometricsmodels._lib")]`として新規定義する
（`.claude/rules/rust-style.md`「言語方針」のmodule属性要件）。フィールド構成は`OLSResult`と同一
（4.3節の通り`residuals`は元スケール）だが、型としては別に定義する
（`WLSResult`は将来、重み付き残差等WLS固有のフィールドが追加される可能性があり、`OLSResult`との
将来的な乖離リスクがOptionsより高いと判断したため。8章参照）。

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
- `OLSOptions`をWLSと共有する設計は、WLS固有の設定（例: frequency weight対応時の重みタイプ切替）が
  将来必要になった時点で見直す。その際は`OLSOptions`から分離するか、共通フィールドを
  `linear/common.rs`側の構造体に切り出すかを判断する（YAGNI、`rust-style.md`
  「ファイル・ディレクトリ構成」）。
- `WLSResult`と`OLSResult`は現時点ではフィールド構成が完全に一致するが、型としては分離した
  （5章）。重み付き残差（`.wresid`相当）等、WLS固有のフィールドが将来追加された場合はこの分離が
  効いてくる。逆に長期間フィールドが分岐しないようであれば、`OLSResult`との統合を再検討してもよい。
