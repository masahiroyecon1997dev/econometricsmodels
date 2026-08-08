# python_package/econometricsmodels/nonlinear/ 実装ノート（Logit/Probit）

このファイルは `python_package/econometricsmodels/nonlinear/` 配下のファイルを読み書きするときだけ自動ロードされる。詳細は`docs/planning/specs/nonlinear-api-design.md`6章・`docs/spec/logit-spec.md`・`docs/spec/probit-spec.md`が正本。

`probit.py`は`logit.py`と完全に同型のパターン（`Probit`/`ProbitResults`、フィールド・メソッド構成も同一。`z_stat`ベースの検定等、下記の各節はLogit/Probit共通で成り立つ）。

## 確定済みのスコープ（再提案しない）

以下は既にユーザー承認済みで見送りが確定している。「使いやすさ」目的で再提案しない（CLAUDE.md 2章の非交渉事項に準ずる運用、`linear/CLAUDE.md`と同じ方針）。

- `summary()`は実装しない（structured onlyの出力方針、`linear/CLAUDE.md`のOLSと同じ）。
- `LogitOptions`/`ProbitOptions`は`_lib`からそのまま再輸出する（独自クラスとして再定義しない、`OLSOptions`と同じ方針）。
- `predict()`はout-of-sample（`new_data`引数）未対応。`OLS.predict(new_data=None)`とは異なり引数を取らない（engine側がまだ対応していないため。別issueでトラッキング）。

## 実装パターン

- `Logit`/`LogitResults`（`Probit`/`ProbitResults`も同様）は`OLS`/`OlsResults`と同型（`data`/`y`/`x`/`options`を保持するだけのコンストラクタ、`fit()`呼び出し時に初めて`_lib.fit_logit`/`_lib.fit_probit`を呼ぶ。コンストラクタでは検証しない）。
- `params`/`std_errors`/`z_stats`/`p_values`は係数名→値の`dict[str, float]`（O(1)取り出し用）。行指向で欲しい場合は`coef_table()`。
- `coef_table()`のキーは`OlsResults.coef_table()`と同じ形状だが、`t_stat`ではなく`z_stat`（Logit/Probitは正規分布ベースのz検定、`nonlinear-api-design.md`5章）。

## `marginal_effects()`/`pred_table()`のキー命名（混同注意）

- `marginal_effects()`の行指向キー（`param`/`dydx`/`std_err`/`z`/`p_value`/`conf_low`/`conf_high`）は`nonlinear-api-design.md`6章で確定済みの命名をそのまま使う。`coef_table()`の`conf_lower`/`conf_upper`とは**意図的に異なる**（statsmodelsの`get_margeff().summary_frame()`のカラム名に近い形を踏襲したもので、表記揺れではない）。
- `pred_table()`の返り値形状（`[{"actual": 0, "predicted_0": .., "predicted_1": ..}, {"actual": 1, ...}]`という行指向`list[dict]`）は仕様書に明記が無く、`coef_table()`/`predict()`との一貫性（このプロジェクトの行指向`list[dict]`慣習）を優先した実装時の判断（ユーザー確認済み）。`_lib.LogitResult.pred_table()`自体は`Vec<Vec<f64>>`（`table[actual][predicted]`の2×2）を返すだけで、ラベル付けはこのモジュール側の責務。

## テスト

`tests/api_tests/test_logit.py`/`test_probit.py`は構造・API・エラーパスのスモークテストのみ（`test_probit.py`は`test_logit.py`と同型）。statsmodels/R glmとの厳密な数値比較は`test_logit_fixtures.py`/`test_logit_crosscheck.py`、Probit側は`test_probit_fixtures.py`/`test_probit_crosscheck.py`で行う（OLSの`test_ols_fixtures.py`/`test_ols_crosscheck.py`と同じ役割分担）。
