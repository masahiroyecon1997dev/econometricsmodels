# python_package/econometricsmodels/panel/ 実装ノート（FE/FEIV/RE）

このファイルは `python_package/econometricsmodels/panel/` 配下のファイルを読み書きするときだけ自動ロードされる。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠と確定済みの判断のみ。詳細は`docs/spec/panel-common.md`・`docs/spec/fe-spec.md`・`docs/spec/re-spec.md`・`docs/spec/fe-iv-spec.md`が正本。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。

## 確定済みのスコープ（再提案しない）

ユーザー承認済みで見送り・決定が確定している。「使いやすさ」を理由に再提案しない（`linear/CLAUDE.md`と同じ運用）。

- **`summary()`は実装しない。`predict()`も実装しない（OLSの`predict()`は例外扱いで、パネルには合意が無い。必要になった時点で別途検討する）。`FEOptions`/`REOptions`/`FEIVOptions`は`_lib`から再輸出する**。

## FE・RE

- **コンストラクタは引数を保持するだけで検証せず、`fit()`で初めて`_lib.fit_fe`/`_lib.fit_re`を呼ぶ**: `IV`と同型。
- **`effects`は必須の`list[str]`（FEは1〜5要素、REは1〜2要素）で、`effects[0]`が個体（既定のクラスターキー）、`effects[1]`が時点。`dk_time`は`effects`とは独立で`cov_type="dk"`では必須**: 詳細: `docs/spec/panel-common.md`1.1節。
- **件数と固定効果は列名キーにする（`n_groups`は`dict[str, int]`、FEの`fixed_effects()`は1-wayでも`{列名: {水準: 効果}}`の入れ子）。REの`sigma2_u`・`theta*`等は意図的に列名キーにしていない**: 次元の数が変わっても同じ形で読めるようにする。詳細: `docs/spec/panel-common.md`2.1節。
- **`params`等は係数名→値の`dict[str, float]`、行指向は`params_table()`（キーはOLSと同じ）。FEの`fixed_effects()`は`_lib`の`dict`を素通しし、型変換しない**: FE・REはt検定なのでIVの汎用命名は要らない。詳細: `docs/spec/fe-spec.md`3.5節。
- **REにFEの`fixed_effects()`に相当する追加メソッドは足さない（ハウスマン検定は`fit()`で計算済みのプロパティ）**: `hausman_*`は検定できる係数が無いときだけ`None`で、計算の失敗は`fit()`が失敗する。詳細: `docs/spec/panel-common.md`2.4節、`docs/spec/re-spec.md`3.7節。
- **REの分散成分と重み（`sigma2_eps`・`sigma2_u`・`sigma2_lambda`・`theta`・`theta_entity`・`theta_time`・`theta_total`）は、使わない側を`None`にして型を安定させる。`variance_method`のdocstringはRust側（`engine_pybind/src/panel/re.rs`）が正本**: 詳細: `docs/spec/re-spec.md`2章。
- **REは`effects`の順序に意味がある（`[0]`が`sigma2_u`・`theta_entity`、`[1]`が`sigma2_lambda`・`theta_time`）が、バランスパネルでは入れ替えても検出できないので、ドキュメントとテストで保証する**: FEは入れ替えても係数・SEは同じだが、既定のクラスターキー・`fixed_effects()`のキー順・時点側の基準期間が変わる。詳細: `docs/spec/re-spec.md`1章。
- **REは切片を持ち`param_names[0]`が`"const"`、`df_resid`は`n - k`（FEの`n - 個体数 - k`とは別の式）。docstringに明記して混同を防ぐ**: 詳細: `docs/spec/re-spec.md`3.3節。
- **`engine_pybind`のFE・REの`fit`は`cargo test`で検証できないので、このラッパーを含めた検証は`tests/panel/`のpytest（fixest・linearmodelsとの照合）が担う**: 詳細: `docs/spec/nonlinear-common.md`10章「テストの制約」、`docs/spec/panel-common.md`4.7節。

## FEIV（`panel/fe_iv.py`）

- **2SLSとGMMは1つのクラス（`FEIVOptions.estimator`）にし、両推定量で項目・メソッドを同じにする。docstringに「2sls only」「gmm only」と書かない（`c_stat()`の重みの決め方だけ書く）**: `IV`と同じインターフェース。詳細: `docs/spec/fe-iv-spec.md`2.1節・2.4節。
- **`FEIVResults(raw, x_endog)`は`x_endog`を保持し、`weak_instrument_f_statistics`・`first_stage()`を`x_endog`の順の`dict`に直す**: `_lib`側は`HashMap`で順序が不定。pickleはラッパーごと往復する。
- **`first_stage()`は既存の`FEResults`の`dict`で返し、診断・`c_stat()`の結果への包み方は`IVResults`と共通の`_results_utils.py`の関数（`cragg_donald_result`・`c_statistic_result`等）を使う**: 詳細: `docs/spec/fe-iv-spec.md`2.4節〜2.6節。
- **統計量はIVの命名（`wald_*`・`stat_dist`）。2SLSは`stat_dist="t"`・`wald_dist="f"`、GMMは`"normal"`・`"chi2"`（`wald_df_denom`は`None`、Waldは`k`で割らない）。`log_likelihood`・`aic`・`bic`は出さない**: 詳細: `docs/spec/fe-iv-spec.md`2.4節。
- **`c_stat()`のクラスター数不足は`ComputationError`で、`fit`の`GMM_WEIGHT_CLUSTER_COUNT_TOO_SMALL`（`ValidationError`）とクラスが違う。2SLSの`dk`は`C_STATISTIC_COV_TYPE_NOT_SUPPORTED`**: docstringとmethodsに書いてある。詳細: `docs/spec/fe-iv-spec.md`3.5節。
- **横断テストへの登録を忘れない。登録先の一覧は`docs/spec/fe-iv-spec.md`6.1節。GMMは`FEIV_GMM`等の別のキーにし、`test_validation_order`の`feiv_gmm`は`base_opts={"estimator": "gmm"}`、`unused_cluster`には`gmm_weight_type="robust"`を足す**: 2SLSとGMMでキーが重複する。GMMの既定の重み`cluster`も`cluster`列を使うので、そのままでは未使用にならない。
