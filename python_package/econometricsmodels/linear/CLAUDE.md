# python_package/econometricsmodels/linear/ 実装ノート（OLS/WLS）

このファイルは `python_package/econometricsmodels/linear/` 配下のファイルを読み書きするときだけ自動ロードされる。詳細は`docs/planning/specs/ols-api-design.md`5章・`ols-implementation-notes.md`7章が正本。

## 確定済みのスコープ（再提案しない）

以下は既にユーザー承認済みで見送りが確定している。「使いやすさ」目的で再提案しない（CLAUDE.md 2章の非交渉事項に準ずる運用）。

- `summary()` / `predict()` / `fitted_values` / `conf_int()`のDataFrame版は実装しない。「薄いラッパー」というスコープを優先する。
- `OLSOptions`（`WLSOptions`も同様）は独自クラスとして再定義せず、`_lib`からそのまま再輸出する。
- `params`/`std_errors`/`t_stats`/`p_values`は係数名→値の`dict[str, float]`（O(1)取り出し用）。行指向で欲しい場合は`coef_table()`（`list[dict]`、REST APIレスポンスにそのまま使える形）を使う。DataFrameには変換しない。
- `residuals`はそのまま`list[float]`を素通しする（polars Seriesへの変換等はしない）。

## 実装パターン

- `OLS`/`WLS`クラスは`data`/`y`/`x`（+`weight`）/`options`のコンストラクタ引数を保持するだけで、`fit()`呼び出し時に初めて`_lib.fit_ols`/`_lib.fit_wls`を呼ぶ（コンストラクタでは検証しない）。
- `OlsResults`/`WlsResults`は`_lib`の結果オブジェクト（`_lib.OLSResult`等）を`_raw`として保持する薄いラッパー。新しいプロパティを追加する際も、Rust側`#[pyclass(get_all)]`のフィールドをそのまま`dict`化する以上のロジックをPython側に持ち込まない。
