# python_package/econometricsmodels/nonlinear/ 実装ノート（Logit/Probit/Tobit）

このファイルは `python_package/econometricsmodels/nonlinear/` 配下のファイルを読み書きするときだけ自動ロードされる。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠と確定済みの判断のみ。詳細は`docs/spec/nonlinear-common.md`6章・10章・`docs/spec/logit-spec.md`・`docs/spec/probit-spec.md`・`docs/spec/tobit-spec.md`が正本。`probit.py`は`logit.py`と同型（クラス・フィールド・メソッドの構成が同じ）で、`tobit.py`は同じ骨格に下の差分を足したもの。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。

## 確定済みのスコープ（再提案しない）

ユーザー承認済みで見送り・決定が確定している。「使いやすさ」を理由に再提案しない（`linear/CLAUDE.md`と同じ運用）。

- **`summary()`は実装しない。`LogitOptions`/`ProbitOptions`/`TobitOptions`は`_lib`から再輸出する**: OLSと同じ。
- **`predict(new_data=None)`はOLSと同じシグネチャ・`None`の意味で、戻り値は観測順の`list[float]`。ただし値は確率（Tobitは`target`ごとの量）で、OLSのような点予測ではないことをdocstring（Note節）に明記する**: OLSからの類推で誤解されやすい。`pred_table()`・`censoring_fit_check()`の新規データ対応は未対応。詳細: `docs/spec/logit-spec.md`3.6節、`docs/spec/tobit-spec.md`3.6節。
- **`augment(new_data=None)`の列名は、Logit/Probitが固定の`"predicted_probability"`、Tobitが`"predicted_{target}"`**: Tobitを固定名にすると複数の`target`を積み上げる2回目が列名の衝突で失敗する。詳細: `docs/spec/nonlinear-common.md`10章。

## Tobit固有

- **`params`・`std_errors`・`test_stats`・`p_values`・`conf_int`・`params_table()`は末尾に`"sigma"`を含む（`sigma`プロパティは`params["sigma"]`と同値）。`x`に`"sigma"`という列名があれば`ValidationError`**: `engine_pybind`の`TobitResult`が`param_names`に`"sigma"`を足して`(k+1)`長にしているので、Python側は`dict(zip(...))`するだけ。詳細: `docs/spec/tobit-spec.md`2章。
- **`log_likelihood_null`・`lr_statistic`・`pseudo_r_squared`は提供せず、`wald_statistic`/`wald_p_value`を提供する。`pred_table()`の代わりに`censoring_fit_check()`（`category`・`observed_rate`・`model_implied_rate`の`list[dict]`）**: `y`が連続で的中表は意味を持たない。詳細: `docs/spec/nonlinear-common.md`5章、`docs/spec/tobit-spec.md`3.6節。
- **`predict()`/`marginal_effects()`/`augment()`は`target`（`"expected_latent"`・`"expected_observed"`・`"prob_uncensored"`・`"expected_uncensored"`、既定`"expected_observed"`）を取り、`target`と`new_data`は独立したキーワード引数**: 詳細: `docs/spec/tobit-spec.md`3.5節・3.6節。
- **打ち切り境界の検査（両方`None`・`lower >= upper`・非有限・`y`が境界外・非打ち切り観測が0件）はengineが`ValidationError`にし、Python側で重複して検査しない**: 詳細: `docs/spec/tobit-spec.md`1章。
- **完全な多重共線性は`solver`によらず`fit()`冒頭のランクチェックが`ComputationError`にするので、検出経路を`solver`でparametrizeしなくてよい。ただし非既定の`solver`の文字列×特異な入力×`ComputationError`のAPI境界テストは残す**: `solver`の文字列のパースと配線は`solver`ごと。詳細: `docs/spec/nonlinear-common.md`1.4節。

## 結果

- **`LogitResults`/`ProbitResults`/`TobitResults`に共通の属性を足すときは3クラス全てに足し、`tests/cross_method/test_mle_result_consistency.py`の集合も更新する**: 公開属性の集合（共通＋手法固有、ラッパーと`_raw`の両方）をこのテストが固定している。
- **`params`等は係数名→値の`dict[str, float]`、行指向は`params_table()`で、統計量のキーは`test_stat`に統一する（z検定なので`stat_dist="normal"`・`stat_df=None`）**: 詳細: `docs/spec/nonlinear-common.md`4章。
- **`marginal_effects()`の行のキー（`param_name`・`effect`・`std_error`・`test_stat`・`p_value`・`conf_lower`・`conf_upper`）は`params_table()`と同じ統計量名にし、ベンチマークのフィクスチャJSON（`margeff`）のキーも同名で、テストで読み替えない**: `effect`だけ限界効果固有。
- **`pred_table()`は`[{"actual": 0, "predicted_0": .., "predicted_1": ..}, {"actual": 1, ...}]`の行指向`list[dict]`で返し、ラベル付けはこのモジュールの責務**: `_lib`は2×2の`Vec<Vec<f64>>`（`table[actual][predicted]`）を返すだけ。行指向の`list[dict]`慣習を優先した（ユーザー確認済み）。
- **コンストラクタは引数を保持するだけで検証せず、`fit()`で初めて`_lib.fit_*`を呼ぶ**: OLSと同型。

## テスト

- **各手法のテストは`test_<手法>_api.py`（構造・API）・`test_<手法>_validation.py`（エラーパス）・`test_<手法>_reference.py`（主リファレンス）・`test_<手法>_crosscheck.py`（R）に分ける。Tobitのapi・validationは`tests/conftest.py`の`censored_dataset`（`y=0`で左打ち切り）を使う**: OLS/WLSと同じ分担。
