# python_package/econometricsmodels/panel/ 実装ノート（FE）

このファイルは `python_package/econometricsmodels/panel/` 配下のファイルを読み書きするときだけ自動ロードされる。詳細は`docs/planning/specs/panel-api-design.md`が正本。

## 確定済みのスコープ（再提案しない）

以下は既にユーザー承認済みで見送りが確定している。「使いやすさ」目的で再提案しない（CLAUDE.md 2章の非交渉事項に準ずる運用、`linear/CLAUDE.md`と同じ方針）。

- `summary()`は実装しない（structured onlyの出力方針）。
- `predict()`は実装しない（`panel-api-design.md`にpredict関連の合意が無く、OLSの`predict()`はあくまで例外扱い。必要になった時点で別issueで検討する）。
- `FeOptions`は`_lib`からそのまま再輸出する（独自クラスとして再定義しない、`OLSOptions`/`IvOptions`と同じ方針）。

## 実装パターン

- `FE`/`FeResults`は`IV`/`IvResults`と同型（`data`/`y`/`x`/`entity`/`options`を保持するだけのコンストラクタ、`fit()`呼び出し時に初めて`_lib.fit_fe`を呼ぶ。コンストラクタでは検証しない）。
- `params`/`std_errors`/`t_stats`/`p_values`は係数名→値の`dict[str, float]`（O(1)取り出し用）。行指向で欲しい場合は`coef_table()`（`OlsResults.coef_table()`と同じキー: `param`/`coef`/`std_err`/`t_stat`/`p_value`/`conf_lower`/`conf_upper`。FEはOLS同様t検定のためIVの`stats`/`stat`のような汎用命名は不要）。
- `fixed_effects()`はIVの`first_stage()`と同じ「追加結果は別メソッド」方針（`panel-api-design.md`6.6節）。ただし`first_stage()`と異なり結果の型変換は不要（`_lib.FeResult.fixed_effects()`が返す`dict`をそのまま素通しする）。

## テスト

`engine_pybind`側（Issue #186〜#188）は`PyDataFrame`引数を取る関数を`cargo test`で直接検証できない制約があるため（`nonlinear/CLAUDE.md`「テストの制約」参照）、`maturin develop` + Pythonスモークテストのみで検証されている。`tests/panel/`配下のpytest（Issue #190、`fixest`/`linearmodels`との数値照合ベンチマーク）が、このPython側ラッパーを含めた最初の本格的なテストになる。
