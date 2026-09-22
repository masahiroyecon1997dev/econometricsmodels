# python_package/econometricsmodels/linear/ 実装ノート（OLS/WLS）

このファイルは `python_package/econometricsmodels/linear/` 配下のファイルを読み書きするときだけ自動ロードされる。詳細は`docs/spec/ols-spec.md`が正本。

## 確定済みのスコープ（再提案しない）

以下は既にユーザー承認済みで見送りが確定している。「使いやすさ」目的で再提案しない（CLAUDE.md 2章の非交渉事項に準ずる運用）。

- `summary()` / `conf_int()`のDataFrame版は実装しない。「薄いラッパー」というスコープを優先する。
- **`predict()`は例外的に実装する**。`fitted_values`という別名のプロパティは作らず、`predict(new_data=None)`の1メソッドに統一する。`new_data=None`（デフォルト）で学習データの予測値、指定時は新規データの予測値を返す。Logitの`predict()`（引数なしで学習データの予測確率を返す設計）と命名を揃えるための判断（`ols-spec.md`「predict()」参照）。WLSにも同じ設計で実装済み（Issue #132、`wls-spec.md`「predict()」）。重みは予測値の計算に一切関与しない（学習データ・新規データいずれも`ŷ=x'β̂`）。戻り値の辞書キーは`"predicted"`（当初`"fitted"`固定だったが、out-of-sample予測に対して統計学的に不正確という指摘を受けIssue #309で統一、OLS/WLS共通）。
- **`augment(new_data=None)`も例外的に実装する**（Issue #295）。`predict()`と同じ引数・エラー
  意味論で、ソースデータ（学習データ or `new_data`）に予測値の列（`"predicted"`）を1列付加した
  polars DataFrameを返す。プロジェクト全体の「DataFrameは返さない」方針の**唯一の例外**（既存の
  `predict()`/`residuals`/`coef_table()`は変更しない、フラグで戻り値の型を変える設計は不採用、
  `ols-spec.md`「augment()」参照）。実装はRust側（`engine_pybind`、列名衝突を`ValidationError`
  として送出しやすいため）。Python側は`self._raw.augment(new_data)`を素通しするだけ。
- `OLSOptions`（`WLSOptions`も同様）は独自クラスとして再定義せず、`_lib`からそのまま再輸出する。
- `params`/`std_errors`/`t_stats`/`p_values`は係数名→値の`dict[str, float]`（O(1)取り出し用）。行指向で欲しい場合は`coef_table()`（`list[dict]`、REST APIレスポンスにそのまま使える形）を使う。DataFrameには変換しない（`augment()`を除く。上記参照）。
- `residuals`はそのまま`list[float]`を素通しする（polars Seriesへの変換等はしない）。

## 実装パターン

- `OLS`/`WLS`クラスは`data`/`y`/`x`（+`weight`）/`options`のコンストラクタ引数を保持するだけで、`fit()`呼び出し時に初めて`_lib.fit_ols`/`_lib.fit_wls`を呼ぶ（コンストラクタでは検証しない）。
- `OLSResults`/`WLSResults`は`_lib`の結果オブジェクト（`_lib.OLSResult`等）を`_raw`として保持する薄いラッパー。新しいプロパティを追加する際も、Rust側`#[pyclass(get_all)]`のフィールドをそのまま`dict`化する以上のロジックをPython側に持ち込まない。
