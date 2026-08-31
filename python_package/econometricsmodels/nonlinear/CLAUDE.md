# python_package/econometricsmodels/nonlinear/ 実装ノート（Logit/Probit/Tobit）

このファイルは `python_package/econometricsmodels/nonlinear/` 配下のファイルを読み書きするときだけ自動ロードされる。詳細は`docs/planning/specs/nonlinear-api-design.md`6章・`docs/spec/logit-spec.md`・`docs/spec/probit-spec.md`・`docs/planning/specs/nonlinear-implementation-notes.md`（Tobit）が正本。

`probit.py`は`logit.py`と完全に同型のパターン（`Probit`/`ProbitResults`、フィールド・メソッド構成も同一。`z_stat`ベースの検定等、下記の各節はLogit/Probit共通で成り立つ）。`tobit.py`（Issue #226）は同じ骨格を踏襲しつつ、以下の点でLogit/Probitと異なる（詳細は「Tobit固有の設計」節参照）。

## Tobit固有の設計

- **`params`/`std_errors`/`z_stats`/`p_values`/`conf_int`は`"sigma"`（誤差項の標準偏差）を含む**: `engine_pybind`の`TobitResult`が`param_names`の末尾に`"sigma"`を付加して`(k+1)`長に統一しているため（`engine_pybind/src/nonlinear/tobit.rs`の`TobitResult`docコメント参照）、Python側もそのまま`dict(zip(param_names, ...))`すれば`"sigma"`が自然に含まれる。`coef_table()`も`"sigma"`の行を含む（R`summary.tobit`の`Log(scale)`行に相当）。`sigma: float`プロパティ（`params["sigma"]`と同値）も利便のため追加した。
- **`log_likelihood_null`/`lr_statistic`/`lr_p_value`/`pseudo_r_squared`は提供しない**。代わりに`wald_statistic`/`wald_p_value`（モデル全体の有意性検定）を提供する（`nonlinear-api-design.md`5章）。
- **`pred_table()`の代わりに`censoring_fit_check()`**: `y`が連続変数のため分類の的中表は意味を持たない（`nonlinear-api-design.md`6章）。返り値は`pred_table()`と同じ行指向`list[dict]`慣習に合わせ、`category`（`"lower"`/`"uncensored"`/`"upper"`のうち該当するもの）・`observed_rate`・`model_implied_rate`をキーに持つ（実装時の判断、`pred_table()`の`[{"actual":..., "predicted_0":...}]`という先例と同じ理由）。
- **`predict()`/`marginal_effects()`に`target`引数**（`"expected_latent"`/`"expected_observed"`/`"prob_uncensored"`、既定`"expected_observed"`）を追加。`predict()`の返り値の行は単一キー`"predicted"`（Logitの`"probability"`に相当する汎用キー、複数の予測対象があるため対象非依存の名前にした）。
- **打ち切り境界（`lower`/`upper`）関連の追加バリデーション**: `TobitOptions.lower`/`upper`が両方`None`（`InvalidCensoringBounds`）、`y`が境界外（`YOutOfCensoringBounds`）、非打ち切り観測が1件も無い（`NoUncensoredObservations`、Issue #223）はいずれも`engine`層で検証され`ValidationError`になる。`x`に`"sigma"`という列名がある場合も`ValidationError`（`"sigma"`合成パラメータ名との衝突、`engine_pybind`の`validate_no_sigma_collision`）。
- **完全な多重共線性の検出経路**: Logitは`method`（newton/bfgs/lbfgs）によって検出経路が異なる（`newton_step`のQR分解 vs 収束後の`observed_information_cov_params`）が、Tobitは`ols_initial_params`のQR検証が`method`に関わらず常に最初に実行されるため、`method`をparametrizeしなくても`ComputationError`（`SingularDesignMatrix`）を一貫して検出できる。

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

`tests/nonlinear/test_logit.py`/`test_probit.py`は構造・API・エラーパスのスモークテストのみ（`test_probit.py`は`test_logit.py`と同型）。statsmodels/R glmとの厳密な数値比較は`test_logit_fixtures.py`/`test_logit_crosscheck.py`、Probit側は`test_probit_fixtures.py`/`test_probit_crosscheck.py`で行う（OLSの`test_ols_reference.py`/`test_ols_crosscheck.py`と同じ役割分担）。

`tests/nonlinear/test_tobit.py`（Issue #226）も同じ位置づけ（構造・API・エラーパスのスモークテストのみ）。`tests/conftest.py`の共有`dataset`フィクスチャを`y=0.0`で左打ち切りした`censored_dataset`フィクスチャ（打ち切り率21%、`TobitOptions`の既定`lower=0.0`と一致）を使う。R`survival::survreg`/`AER::tobit`との数値比較は`test_tobit_fixtures.py`/`test_tobit_crosscheck.py`（Issue #227）で行う。
