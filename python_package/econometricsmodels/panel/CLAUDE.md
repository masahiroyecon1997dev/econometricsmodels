# python_package/econometricsmodels/panel/ 実装ノート（FE/RE）

このファイルは `python_package/econometricsmodels/panel/` 配下のファイルを読み書きするときだけ自動ロードされる。詳細は`docs/spec/fe-spec.md` / `docs/spec/re-spec.md`（FE/RE共通の論点は`docs/spec/panel-common.md`）が正本。

## 確定済みのスコープ（再提案しない）

以下は既にユーザー承認済みで見送りが確定している。「使いやすさ」目的で再提案しない（CLAUDE.md 2章の非交渉事項に準ずる運用、`linear/CLAUDE.md`と同じ方針）。

- `summary()`は実装しない（structured onlyの出力方針）。
- `predict()`は実装しない（`panel-common.md`にpredict関連の合意が無く、OLSの`predict()`はあくまで例外扱い。必要になった時点で別issueで検討する）。
- `FEOptions`/`REOptions`は`_lib`からそのまま再輸出する（独自クラスとして再定義しない、`OLSOptions`/`IVOptions`と同じ方針）。

## 実装パターン

- `FE`/`FEResults`・`RE`/`REResults`はいずれも`IV`/`IVResults`と同型（`data`/`y`/`x`/`effects`/`options`を保持するだけのコンストラクタ、`fit()`呼び出し時に初めて`_lib.fit_fe`/`_lib.fit_re`を呼ぶ。コンストラクタでは検証しない）。
- **`effects`は必須の`list[str]`（1〜2要素）**: 1要素=one-way、2要素=two-way（`effects[0]`が個体、`effects[1]`が時点。two-wayはバランスパネル必須）。FE・REとも同じで、`FEOptions.time`・`REOptions.time`は廃止した。3要素以上は`EFFECTS_TOO_MANY`（`ValidationError`）。先頭要素がデフォルトのクラスターキー。`dk_time`は`effects`とは独立（`cov_type="dk"`で必須、`effects`の列を借用しない）。理由は`panel-common.md`1.1節。
- **件数と固定効果は列名キー**: `n_groups`（`dict[str, int]`、`effects`の列名→グループ数、`effects`の順。旧`n_entities`・`n_periods`の置き換え）、FEの`fixed_effects()`は常に`{列名: {水準: 効果}}`の入れ子（one-wayも入れ子）。一方、REの`sigma2_u`・`sigma2_lambda`・`theta`・`theta_entity`・`theta_time`・`theta_total`は意図的に現状維持（`panel-common.md`2.1節）。旧版のpickleは読めない（`FORMAT_VERSION`は上げていない）。
- `params`/`std_errors`/`test_stats`/`p_values`は係数名→値の`dict[str, float]`（O(1)取り出し用）。行指向で欲しい場合は`params_table()`（`OLSResults.params_table()`と同じキー: `param_name`/`param`/`std_error`/`test_stat`/`p_value`/`conf_lower`/`conf_upper`。FE/REともOLS同様t検定のためIVの`stats`/`stat`のような汎用命名は不要）。
- `fixed_effects()`（FE限定）はIVの`first_stage()`と同じ「追加結果は別メソッド」方針（`docs/spec/fe-spec.md`3.5節）。ただし`first_stage()`と異なり結果の型変換は不要（`_lib.FEResult.fixed_effects()`が返す`dict`をそのまま素通しする）。
- **REにはFEの`fixed_effects()`に相当する追加メソッドが無い**: ハウスマン検定（`hausman_statistic`/`hausman_p_value`/`hausman_df`）は`fit()`内で自動計算済みの値をそのまま`REResults`のプロパティとして公開するだけで済む（`panel-common.md`2.4節「RE: ハウスマン検定は`fit()`内で自動計算」）。3つとも`float | None`/`int | None`型（検定できる傾き係数が無い場合のみ`None`。補助回帰・ロバスト共分散の計算失敗は`fit()`が`ValidationError`/`ComputationError`で失敗する）。比較相手はRE本体と同じ個体効果構造のFE（1-wayは個体のみ、2-wayは個体+時点）で、補助回帰のWald検定の共分散はRE本体の`cov_type`に連動する。`REOptions.dk_time`（`cov_type="dk"`専用・必須）はDKの時点列にのみ使う。
- **`REOptions.variance_method`**（`"swamy_arora"`・`"wallace_hussain"`・`"amemiya"`）と`REResults.variance_method`（正規化した値）: 分散成分と重みがどの推定法の値かを示す。`REOptions`は`_lib`の再輸出のままで、docstringはRust側（`engine_pybind/src/panel/re.rs`）が正本。
- **REの分散成分と重み**: `sigma2_eps`・`sigma2_u`（`float`）、`sigma2_lambda`（2-wayのみ）、`theta`（1-wayのみ、エンティティID→θ_i、IDの辞書順の`dict`）、`theta_entity`/`theta_time`/`theta_total`（2-wayのみ）をプロパティで返す。使わない側は`None`（型安定。`re-spec.md`2章）。
- **2-way RE（`effects`が2要素、バランスパネル）**: REは`effects`の順序に意味がある（`effects[0]`が個体: `sigma2_u`・`theta_entity`、`effects[1]`が時点: `sigma2_lambda`・`theta_time`）。バランスパネルでは入れ替えてもモデルが対称なので検証では検出できず、ドキュメントとテストで保証する（FEは入れ替えても係数・SEは同じだが、デフォルトのクラスターキー、`fixed_effects()`のキー順、時点側の基準期間が変わる）。ハウスマン検定・パネルR²は2-wayでも1-wayと同じプロパティで返す（`hausman_df`は2-way within変換後に分散のある列の数、`re-spec.md`3.6・3.7節）。`effects`と`dk_time`は別物で、`cov_type="dk"`では`dk_time`が必須（`effects`の列は借用されない）。
- **REは切片を持つ**ため`param_names[0]`が常に`"const"`になる（FEはwithin変換で切片が構造的に消えるため無い）。`df_resid`の意味もFEと異なる（`n - k`、FEの`n - 個体数 - k`とは別式、`re-spec.md`3.3節）——docstringに明記して混同を防ぐ。

## テスト

`engine_pybind`側（FE・RE）は`PyDataFrame`引数を取る関数を`cargo test`で直接検証できない制約があるため（`nonlinear/CLAUDE.md`「テストの制約」参照）、`maturin develop` + Pythonスモークテストのみで検証されている。`tests/panel/`配下のpytest（`fixest`/`linearmodels`との数値照合ベンチマーク）が、このPython側ラッパーを含めた最初の本格的なテストになる。
