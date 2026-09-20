# python_package/econometricsmodels/panel/ 実装ノート（FE/RE）

このファイルは `python_package/econometricsmodels/panel/` 配下のファイルを読み書きするときだけ自動ロードされる。詳細は`docs/planning/specs/panel-api-design.md`が正本。

## 確定済みのスコープ（再提案しない）

以下は既にユーザー承認済みで見送りが確定している。「使いやすさ」目的で再提案しない（CLAUDE.md 2章の非交渉事項に準ずる運用、`linear/CLAUDE.md`と同じ方針）。

- `summary()`は実装しない（structured onlyの出力方針）。
- `predict()`は実装しない（`panel-api-design.md`にpredict関連の合意が無く、OLSの`predict()`はあくまで例外扱い。必要になった時点で別issueで検討する）。
- `FeOptions`/`ReOptions`は`_lib`からそのまま再輸出する（独自クラスとして再定義しない、`OLSOptions`/`IvOptions`と同じ方針）。

## 実装パターン

- `FE`/`FeResults`・`RE`/`ReResults`はいずれも`IV`/`IvResults`と同型（`data`/`y`/`x`/`entity`/`options`を保持するだけのコンストラクタ、`fit()`呼び出し時に初めて`_lib.fit_fe`/`_lib.fit_re`を呼ぶ。コンストラクタでは検証しない）。
- `params`/`std_errors`/`t_stats`/`p_values`は係数名→値の`dict[str, float]`（O(1)取り出し用）。行指向で欲しい場合は`coef_table()`（`OlsResults.coef_table()`と同じキー: `param`/`coef`/`std_err`/`t_stat`/`p_value`/`conf_lower`/`conf_upper`。FE/REともOLS同様t検定のためIVの`stats`/`stat`のような汎用命名は不要）。
- `fixed_effects()`（FE限定）はIVの`first_stage()`と同じ「追加結果は別メソッド」方針（`panel-api-design.md`6.6節）。ただし`first_stage()`と異なり結果の型変換は不要（`_lib.FeResult.fixed_effects()`が返す`dict`をそのまま素通しする）。
- **RE（Issue #202）にはFEの`fixed_effects()`に相当する追加メソッドが無い**: ハウスマン検定（`hausman_statistic`/`hausman_p_value`/`hausman_df`）は`fit()`内で自動計算済みの値をそのまま`ReResults`のプロパティとして公開するだけで済む（`panel-api-design.md`2.4節「RE: ハウスマン検定は`fit()`内で自動計算」）。3つとも`float | None`/`int | None`型（内部FE比較が不成立の場合`None`）。
- **REは切片を持つ**ため`param_names[0]`が常に`"const"`になる（FEはwithin変換で切片が構造的に消えるため無い）。`df_resid`の意味もFEと異なる（`n - k`、FEの`n - n_entities - k`とは別式、7.5節）——docstringに明記して混同を防ぐ。

## テスト

`engine_pybind`側（FE: Issue #186〜#188、RE: Issue #200〜#201）は`PyDataFrame`引数を取る関数を`cargo test`で直接検証できない制約があるため（`nonlinear/CLAUDE.md`「テストの制約」参照）、`maturin develop` + Pythonスモークテストのみで検証されている。`tests/panel/`配下のpytest（`fixest`/`linearmodels`との数値照合ベンチマーク）が、このPython側ラッパーを含めた最初の本格的なテストになる（FE: Issue #190、RE: 別issueで着手予定）。
