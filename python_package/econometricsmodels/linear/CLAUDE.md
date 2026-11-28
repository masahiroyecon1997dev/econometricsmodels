# python_package/econometricsmodels/linear/ 実装ノート（OLS/WLS）

このファイルは `python_package/econometricsmodels/linear/` 配下のファイルを読み書きするときだけ自動ロードされる。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠と確定済みの判断のみ。詳細は`docs/spec/ols-spec.md`・`docs/spec/wls-spec.md`が正本。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。

## 確定済みのスコープ（再提案しない）

ユーザー承認済みで見送り・決定が確定している。「使いやすさ」を理由に再提案しない（CLAUDE.md 2章の非交渉事項に準ずる運用）。

- **`summary()`・DataFrame版の`conf_int()`は実装しない。短い`__repr__`（被説明変数・`n_obs`・`cov_type`と、係数ごとの`param`・`std_error`の表）は別機能で、これを`summary()`へ育てない**: 「薄いラッパー」のスコープを優先する。詳細: `docs/spec/ols-spec.md`2章。
- **`predict()`は例外的に実装し、`predict(new_data=None)`の1メソッドに統一する（`fitted_values`の別名プロパティを作らない）。戻り値は観測順の`list[float]`で、区間を足すなら別メソッドにし`predict()`にキーを足さない。WLSの重みは予測値に関与しない（`ŷ=x'β̂`）**: Logitの`predict()`と命名を揃える。詳細: `docs/spec/ols-spec.md`3.4節、`docs/spec/wls-spec.md`。
- **`augment(new_data=None)`はDataFrameを返す唯一の例外。追加する列名は`"predicted"`（学習データ・新規データ、OLS・WLSで共通）。既存の`predict()`/`residuals`/`params_table()`は変えず、フラグで戻り値の型を変える設計にしない。Python側は`self._raw.augment(new_data)`を素通しするだけ**: 列名の衝突は`engine_pybind`が`ValidationError`にする。詳細: `docs/spec/ols-spec.md`3.5節。
- **事後診断（`white_test`・`breusch_pagan_test`・`breusch_godfrey_test`）は`fit()`で計算せず、検定共通の`DiagnosticResult`を継承した結果（`WhiteTestResult`等）にRust側の出力を詰め替えるだけ。計算・型の検査・バリデーションはRust側**: `breusch_godfrey_test`の`time`と`nlags`は必須（行順を時間順とみなす既定も恣意的な既定のラグ次数も置かない）。補助回帰の`aux_terms[0]`は常に`"const"`、除いた項は`dropped_terms`。Breusch-Paganの`variables`はモデル外の列・`y`も指定でき拒否しない。WLS・IVへの展開は未対応（WLSは残差の意味が変わる）。詳細: `docs/spec/ols-spec.md`3.9〜3.11節、`docs/spec/inference-conventions.md`6章。
- **`OLSOptions`/`WLSOptions`は独自クラスとして再定義せず`_lib`から再輸出する。`params`等は係数名→値の`dict[str, float]`、行指向は`params_table()`（`list[dict]`）、`residuals`は`list[float]`のまま。DataFrameに変換しない（`augment()`を除く）**。

## 実装パターン

- **`OLS`/`WLS`のコンストラクタは引数を保持するだけで検証せず、`fit()`で初めて`_lib.fit_ols`/`_lib.fit_wls`を呼ぶ。`*Results`は`_lib`の結果を`_raw`として持つ薄いラッパーで、`#[pyclass(get_all)]`のフィールドを`dict`化する以上のロジックをPython側に持ち込まない**: 詳細: `.claude/rules/python-style.md`「設計方針との整合性」。
- **係数名→値の`dict`化・`conf_int`・`params_table()`の行の組み立ては`econometricsmodels/_results_utils.py`のヘルパー（`by_param`・`conf_int_by_param`・`params_table_rows`・`marginal_effect_rows`・`results_repr`）を使う。プロパティ本体とdocstringは各クラスに残し、基底クラスにしない**: 手法ごとにdocstringやメンバーが分岐する。
