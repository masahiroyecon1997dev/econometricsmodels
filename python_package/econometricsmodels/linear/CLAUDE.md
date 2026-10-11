# python_package/econometricsmodels/linear/ 実装ノート（OLS/WLS）

このファイルは `python_package/econometricsmodels/linear/` 配下のファイルを読み書きするときだけ自動ロードされる。詳細は`docs/spec/ols-spec.md`が正本。

## 確定済みのスコープ（再提案しない）

以下は既にユーザー承認済みで見送りが確定している。「使いやすさ」目的で再提案しない（CLAUDE.md 2章の非交渉事項に準ずる運用）。

- `summary()` / `conf_int()`のDataFrame版は実装しない。「薄いラッパー」というスコープを優先する。
- **`predict()`は例外的に実装する**。`fitted_values`という別名のプロパティは作らず、`predict(new_data=None)`の1メソッドに統一する。`new_data=None`（デフォルト）で学習データの予測値、指定時は新規データの予測値を返す。Logitの`predict()`（引数なしで学習データの予測確率を返す設計）と命名を揃えるため。戻り値は観測順の`list[float]`（`residuals`と同じ形。区間を足すなら別メソッドにし、`predict()`にキーを足さない。`ols-spec.md`「predict()」参照）。WLSにも同じ設計で実装済み（`wls-spec.md`「predict()」）。重みは予測値の計算に一切関与しない（学習データ・新規データいずれも`ŷ=x'β̂`）。`augment()`が付加する列名は`"predicted"`（学習データ・新規データで呼び分けない、OLS/WLS共通）。
- **`augment(new_data=None)`も例外的に実装する**。`predict()`と同じ引数・エラー
  意味論で、ソースデータ（学習データ or `new_data`）に予測値の列（`"predicted"`）を1列付加した
  polars DataFrameを返す。プロジェクト全体の「DataFrameは返さない」方針の**唯一の例外**（既存の
  `predict()`/`residuals`/`coef_table()`は変更しない、フラグで戻り値の型を変える設計は不採用、
  `ols-spec.md`「augment()」参照）。実装はRust側（`engine_pybind`、列名衝突を`ValidationError`
  として送出しやすいため）。Python側は`self._raw.augment(new_data)`を素通しするだけ。
- **`white_test(statistic="lm" | "f")`は事後診断として実装する**（`fit()`では計算しない。`ols-spec.md`「white_test()」・`docs/spec/inference-conventions.md`6章）。結果は検定共通の`DiagnosticResult`を継承した`WhiteTestResult`（`econometricsmodels.diagnostics`）で、Rust側の`WhiteTestOutput`を詰め替えるだけ（計算・バリデーションはRust側）。補助回帰は常に定数を含み`aux_terms[0]`は常に`"const"`、重複・定数の項は`dropped_terms`。`WLS`・IVへの展開は未対応（WLSは残差の意味が変わるため別途）。
- **`breusch_pagan_test(variables=None, statistic="lm" | "f")`も事後診断として実装する**（`ols-spec.md`「breusch_pagan_test()」）。Koenkerの標準化版（`n·R²`）のみで、元の`ESS/2`版は扱わない。`variables`は不均一分散の変数の列名のlist（既定`None`はモデルの`x`。モデルに入っていない列・`y`列も指定でき、拒否しない）。補助回帰は常に定数を含み`aux_terms[0]`は常に`"const"`、定数・重複の列は`dropped_terms`（`WhiteTestResult`と同じ形の`BreuschPaganTestResult`）。Rust側の`BreuschPaganTestOutput`を詰め替えるだけで、`variables`の型検査（`list`以外・`str`以外の要素は`TypeError`）・空・重複・列の検査もRust側。
- **`breusch_godfrey_test(time, nlags, statistic="lm" | "f")`も事後診断として実装する**（`ols-spec.md`「breusch_godfrey_test()」）。`time`（時間順を決める列名）と`nlags`は**必須**（行順を時間順とみなす暗黙の既定も、恣意的な既定のラグ次数も置かない）。サンプル前期間のラグは0埋め、補助回帰は元の`X`をそのまま使い切片なしでも定数を足さない（R・Greeneの定義。statsmodelsは足すため切片なしはRのみで照合）。結果は`DiagnosticResult`を継承した`BreuschGodfreyTestResult`（追加フィールド`nlags`）で、Rust側の`BreuschGodfreyTestOutput`を詰め替えるだけ。型の検査（`nlags`が`int`、`bool`・`float`は`TypeError`）はRust側（`extract_strict_int`）。
- `OLSOptions`（`WLSOptions`も同様）は独自クラスとして再定義せず、`_lib`からそのまま再輸出する。
- `params`/`std_errors`/`test_stats`/`p_values`は係数名→値の`dict[str, float]`（O(1)取り出し用）。行指向で欲しい場合は`coef_table()`（`list[dict]`、REST APIレスポンスにそのまま使える形）を使う。DataFrameには変換しない（`augment()`を除く。上記参照）。
- `residuals`はそのまま`list[float]`を素通しする（polars Seriesへの変換等はしない）。

## 実装パターン

- `OLS`/`WLS`クラスは`data`/`y`/`x`（+`weight`）/`options`のコンストラクタ引数を保持するだけで、`fit()`呼び出し時に初めて`_lib.fit_ols`/`_lib.fit_wls`を呼ぶ（コンストラクタでは検証しない）。
- `OLSResults`/`WLSResults`は`_lib`の結果オブジェクト（`_lib.OLSResult`等）を`_raw`として保持する薄いラッパー。新しいプロパティを追加する際も、Rust側`#[pyclass(get_all)]`のフィールドをそのまま`dict`化する以上のロジックをPython側に持ち込まない。
