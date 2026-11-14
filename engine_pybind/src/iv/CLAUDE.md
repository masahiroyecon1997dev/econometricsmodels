# engine_pybind/src/iv/ 実装ノート（IV: 2SLS/GMM）

このファイルは `engine_pybind/src/iv/` 配下のファイルを読み書きするときだけ自動ロードされる。設計の背景は `docs/spec/iv-spec.md` が正本。ここは差分の索引のみ。

## 踏んだ罠（再発防止）

- **`engine`側で共有エラー型（`IvError`等）に新バリアントを追加するときは、`cargo build -p engine`だけでなく`cargo test --workspace`（または少なくとも`-p engine_pybind`）まで確認する**: `iv_error_to_pyerr`（`common.rs`）は`to_pyerr(&err)`を呼ぶだけで`IvError`を`match`しないため、`engine_pybind`側の更新は要らない。新バリアントの`code`は`engine`の`ErrorInfo::report()`（`_`なしで全バリアントを列挙するため、足し忘れは`engine`自体のコンパイルエラー）が決める。`cargo build -p engine`はパッケージ境界を跨ぐ破壊を検出できないため、ワークスペース全体で確かめる。

- **`#[cfg(test)] mod tests`からしか呼ばれない関数に`#[expect(dead_code, ...)]`を使うと`--all-targets`ビルドで`unfulfilled_lint_expectations`エラーになる**。`#[expect]`は「指定したlintが実際に発火する」ことを検証する属性のため、以下の非対称性が問題になる。
  - `cargo build`（テストコードを含まない）: 関数が本当に未到達 → `dead_code`が発火 → `#[expect]`が正しく警告を吸収する。
  - `cargo clippy --all-targets -- -D warnings` / `cargo test`（テストコードを含む）: `#[cfg(test)] mod tests`内のテストがその関数を実際に呼ぶため到達可能になる → `dead_code`が発火しない → `#[expect]`の期待が外れ`unfulfilled_lint_expectations`が`-D warnings`下でエラーになる。
  - テストから呼ばれる関数が呼ぶ先の関数にも伝播する（テストから間接的に到達可能になるため）。
  - **対処**: テストから実際に呼ばれる「本番未接続」関数には`#[allow(dead_code)]`（無条件に抑制）を使う。`#[expect(dead_code, ...)]`は「テストからも含めてどこからも一切呼ばれていない」関数にのみ適格。手法をデータ抽出段階とengine呼び出し段階の2段階に分けて実装するときに踏みやすい。IVは全関数が本番経路から呼ばれるため、現在この属性は使っていない。

## 構成

`IVOptions`/`IVResult`/`build_iv_input`/`fit`は`iv/common/`に責務ごとのファイル（`options.rs`・`option_values.rs`・`gmm_setup.rs`・`input.rs`・`result.rs`・`fit.rs`・`error.rs`。`mod.rs`は再エクスポートのみ）に分けて置き、`two_sls.rs`/`gmm.rs`のような手法ごとのファイル分割はしない。`fit_iv`という単一エントリポイントを`IVOptions.estimator`（`"2sls"`/`"gmm"`）で2SLS/GMMに振り分ける設計のため、これらは系統内で真に共有されるロジックであり、`<系統>/common.rs`に置くという既存方針にそのまま合致する。

`build_iv_input`は`cov_type`の値を`crate::linear::common::parse_cov_kind`、列の抽出を`resolve_cov_type`で行う（OLS/WLSと型・エラーメッセージを揃えるため）。組合せの検査はIV自身の`check_iv_options`（GMM専用オプションと`gmm_weight_type`を持つため`linear::common::check_cov_options`は使えない）。

## 診断統計量の配線

- `weak_instrument_f_statistics`: `compute_first_stage`が返す`(String, f64)`を`HashMap<String, f64>`に詰める。
- `wu_hausman_statistic`/`wu_hausman_p_value`: `TwoSlsEstimator`の`Option<f64>`をそのまま代入する。`engine`側の判断で、拡張回帰が特異な場合（第一段階残差の分散がゼロ等）も`None`になる。この場合も`fit()`自体は失敗しない（`engine/src/iv/CLAUDE.md`「Wu-Hausmanの拡張回帰が特異な場合は…」参照）。`estimator="gmm"`では常に`None`（`GmmEstimator`はWu-Hausman検定を実装しない）。
- `overid_statistic`/`overid_p_value`/`overid_df`: `estimator="2sls"`は`TwoSlsEstimator::sargan_*`（Sargan）、`estimator="gmm"`は`GmmEstimator::hansen_j_*`（Hansen J）から代入する。丁度識別では`None`。
- `converged: bool`/`n_iter: i64`: `raise_on_non_convergence=False`でもGMMが収束したかをPython側で確認できるようにするための`IVResult`のフィールド（`LogitResult`/`ProbitResult`と同じ位置づけ）。`estimator="2sls"`は閉形式・非反復なので常に`converged=true`・`n_iter=1`。

## 弱操作変数診断のメソッド（`cragg_donald()`・`kleibergen_paap()`・`effective_f()`）

`fit()`では計算しない事後診断。`IVResult`が非公開フィールドで`source_data`（`fit()`に渡した`DataFrame`への参照）・`options`（`IVOptions`の複製）・`x_exog`・`x_endog`・`instruments`の列名（`y`は`dep_var_name`）を持ち、`diagnostics.rs`の`Rebuilt`が`select_columns`で列を取り込み直して（GILが要る）、`py.detach`の中で`build_iv_input`→`engine`の`compute_*`を呼ぶ。`fit()`と同じ`build_iv_input`なので列の検査と並びが一致する。`source_data`と`options`は`fit`が結果を組み立てた後でGILを持って付ける（`OLSResult`と同じ）。`options`の`PickleField`は`result.rs`にある（`IVOptions`自身の`__reduce__`に任せる）。出力は`CraggDonaldOutput`・`KleibergenPaapOutput`・`EffectiveFOutput`（臨界値は`(水準, 値)`のタプルの列で、Python側が`dict`にする）。

**`c_stat()`（C統計量）も同じ仕組み**（`run_c_stat`、出力は`CStatisticOutput`）。名前（`variables`は`x_endog`、`instruments`は除外操作変数）は、GILを持つ側で一覧の検査（空は`ROLE_EMPTY`、重複は`ROLE_DUPLICATE_COLUMN`）と位置への変換（モデルに無い名前は`C_STATISTIC_UNKNOWN_VARIABLE`。`COLUMN_NOT_FOUND`は`hint`が`data`の列を指すので使わない）を済ませ、昇順の位置をengineへ渡す。両方未指定はengineの`C_STATISTIC_NOTHING_TO_TEST`に任せる。重みは`gmm_type`が`two_step`・`iterated`なら`BuiltIvInput.plan.gmm`の`weight`、それ以外（2SLS・1段階GMM）は`weight_type_for_cov_type(&built.cov_type)`（`c_statistic_weight`）。`tests/cross_method/test_pickle.py`の`TRANSIENT`・`test_gil_release.py`・`test_keyword_only_arguments.py`に登録済み。

## `first_stage()`

`IVResult`の非公開フィールド`first_stage: Vec<(String, OLSResult)>`から返す。`estimator`（2sls/gmm）に依存しない表現にするため、`LogitResult`/`ProbitResult`のように推定量そのものは保持しない。`fit`が第一段階の`OlsEstimator`（設計行列を持つ）を`linear::ols::ols_estimator_to_result`で`OLSResult`へ変換して保持し（設計行列を手放せる）、`first_stage()`は`OLSResult::clone_ref`で複製して返す。pickleでは各`OLSResult`がそのまま状態に入る。第一段階回帰はそれ自体が正しい（ナイーブな）通常のOLSであり（`engine::iv::two_sls`のモジュールdocコメント参照）、OLSとの共有を避ける理由が無いため`ols_estimator_to_result`を再利用する。返る各`OLSResults.f_statistic`/`f_p_value`は通常のOLS F検定（`x_exog`の寄与を含む）であり、弱操作変数診断の部分F統計量（`weak_instrument_f_statistics`）とは別物（`IVResult`のdocコメント参照）。

`IVResult`の組み立ては`fit.rs`の`build_result`が一度だけ行う。2SLSとGMMで同じ値の読み出しは`engine::iv::common::IvEstimate`（両推定量が実装するtrait）から、推定量ごとに違うフィールド（収束・反復、GMM専用の設定、Wald検定の分布、過剰識別検定、Wu-Hausman検定）は`EstimatorSpecific`（`Default`なしで全フィールド必須）から詰める。`IVResult`にフィールドを足すときは、`IvEstimate`にメソッドを足す（共通）か`EstimatorSpecific`に足す（推定量ごと）かを選ぶ。どちらもコンパイラが両分岐の書き漏れを検出する。

`first_stage`/`weak_instrument_f_statistics`は`estimator`によらず`engine::iv::common::compute_first_stage`（2SLS/GMM間で共有）から構築する。**`estimator="2sls"`では第一段階回帰が二重計算になる**（`fit`が明示的に1回、`TwoSlsEstimator::fit`が内部でもう1回）。`OlsEstimator`が`Clone`未実装のため`TwoSlsEstimator::first_stage_estimators()`の借用結果を`IVResult`へ所有権ごと移せず、OLS自体が軽量という前提で許容した設計判断。恒久対応する場合は`TwoSlsEstimator`に第一段階結果を外部注入する`fit`のバリエーションを追加する案がある（着手前にユーザー確認すること）。

## `fit`の事前チェックの順序

以下はいずれも`compute_first_stage`より前に`fit_frame`で行う。第一段階の回帰変数・クラスター係数の数は識別条件により構造方程式以上のため、先に第一段階を走らせると構造方程式側の検査が第一段階の`FirstStageFailed`に隠れてPythonから到達不能になるため（`compute_first_stage`自体はこれらを検証しない）。

- **識別の順序条件（`k_instruments < k_endog`）**（`engine::iv::common::validate_iv_fit_inputs`の先頭）: 過小識別な入力で無駄な第一段階回帰が走るのも防ぐ。`build_iv_input`が`x_endog`/`instruments`の非空を既に検証しているため、ここでの`<`判定は「両方指定されているが数が足りない」ケースのみを扱う。
- **観測数（`engine::iv::common::validate_iv_fit_inputs`が呼ぶ`validate_observation_counts`）**: 構造方程式の`n <= k`を先に`CommonError::InsufficientObservations`（構造方程式の`k`）で、次に`n <= l`（`l = k_exog + k_instruments`、定数項込みの第一段階の回帰変数の数）を`IvError::InsufficientObservationsForFirstStage`（`ValidationError`）で弾く。`n == l`では`X̂ = X`で2SLSがOLSに縮退し（GMMの`W₀ = (Z'Z)⁻¹`も同様）、`n < l`では`Z'Z`が特異のため、第一段階を診断としてdegradeする余地はなく拒否する（`tests/iv/test_iv_validation.py`の`test_observation_count_boundaries`で固定）。
- **`cov_type=Cluster`の構造方程式の`G <= q`**（`engine::iv::common::validate_iv_fit_inputs`が呼ぶ`validate_structural_cluster_count`。`q`の式はエンジン側に集約し、pybindに複製しない）: 構造方程式で`G>q`でも第一段階の`q`が`G`以下なら、`fit`は失敗せず、その第一段階の`OLSResult`（`first_stage()`）の`f_statistic`/`f_p_value`/`f_df_num`/`f_df_denom`だけが`None`になる（`test_cluster_count_between_structural_and_first_stage_slopes_leaves_first_stage_f_undefined`/`test_cluster_count_boundaries`で固定）。`OLSResult`のF系列は全手法（OLS/WLS/FE/RE）で`Option<f64>`。

## GMMのオプションと重み行列

- **GMMの方式は`IVOptions.gmm_type`（`"one_step"`/`"two_step"`（実効既定）/`"iterated"`）で選び、GMM専用オプション（`gmm_type`/`gmm_weight_type`/`gmm_max_iter`/`gmm_tol`/`raise_on_non_convergence`）は全て`Option`で既定`None`**（`resolve_gmm_setup`が`engine::iv::gmm::GmmType`に変換する）。`IVOptions`はpyclassで既定値と明示指定を区別できないため既定値を`None`にし、使われるモードのときだけ実効既定値（`"two_step"`/`"classical"`/`100`/`1e-6`/`true`）に解決する。使われないモードでの明示指定は`check_iv_options`が`ValidationError`にする（`estimator="2sls"`では全GMM専用オプション、`"one_step"`では`gmm_weight_type`、`"one_step"`/`"two_step"`では`gmm_max_iter`/`gmm_tol`/`raise_on_non_convergence`）。`gmm_type`/`gmm_weight_type`の文字列が未知の値のときは、使われるかどうかによらず値の検査（`parse_iv_option_values`）で`unknown ...`を報告する。`cluster`/`hac_lags`/`hac_time`は`cov_type`と`gmm_weight_type`の両方が参照するため「どちらか一方でも使えば有効」で判定する（`check_iv_options`が両方を見る）。`"one_step"`は`gmm_weight_type`をengineに渡さない（結果の`gmm_weight_type`も`None`）。`IVResult.gmm_type`は`estimator="gmm"`のとき正規形に直した値、`"2sls"`では`None`。
- **`resolve_weight_type`**（`cov_type`側の`linear::common::resolve_cov_type`と対になる関数。`gmm_weight_type`は`WeightType`という`cov_type`側の`CovType`とは異なる型を組み立てるため独立実装のまま）が`WeightKind`に対応する列を抽出して`engine::iv::gmm::WeightType`にする。**`gmm_weight_type="hac"`も`hac_time`が必須**（`cov_type="hac"`と同じ`linear::common::require_hac_time`、未指定は`ValidationError`で、メッセージは要求している設定名`gmm_weight_type`を示す。欠落の検査は`check_iv_options`で`cov_type`が先、`gmm_weight_type`が後）。`cluster`/`hac_lags`/`hac_time`は`cov_type`と共用する（`IVOptions`に別フィールドを増やさない設計。`gmm_weight_type`と`cov_type`で異なるクラスター変数を使いたいニーズが出てきたら別フィールド化を検討）。

## オプションの検査（値・組合せ・列）と報告順

`build_iv_input`は、DataFrameを見ない検査（`parse_iv_option_values`: `estimator` → `cov_type` → `gmm_type` → `gmm_weight_type`の値。`check_iv_options`: 使われないオプション → 必須の`hac_time`の欠落）→ ロール構造 → `y`/`x_exog`/`x_endog`/`instruments`の列 → オプション列（`cov_type`側は`resolve_cov_type`、GMMの重み行列側は`resolve_gmm_setup`）の順に呼ぶ。GMMの重み行列の列も`fit`ではなく`build_iv_input`で抽出し、戻り値`BuiltIvInput`の`plan`（`IvPlan`: `IvEstimator`と、`estimator="gmm"`のときの`GmmSetup`）で`fit`に渡す。`*Result`の`cov_type`/`estimator`/`gmm_type`/`gmm_weight_type`の正規形は`CovKind::name()`/`IvEstimator::name()`等から作り、呼び出し側で小文字化し直さない。この順序は`tests/cross_method/test_validation_order.py`が固定している。`gmm_max_iter`が負値のときの変換エラーは`resolve_gmm_setup`で起きるが、範囲（3以上・上限）と`gmm_tol`の検証はengineが行う。

## `IVResult.test_stats`と`stat_dist`/`stat_df`

`IVResult`は`estimator="2sls"`（t分布）・`estimator="gmm"`（z分布、`docs/spec/iv-spec.md`3.2節）の両方で共有される単一の型のため、統計量は全手法共通の名前`test_stats`とし、分布は`stat_dist`（`"t"`/`"normal"`）と`stat_df`（t分布の自由度、正規分布は`None`）で示す（他の手法のResultsと同じ形。IVを`IV2SLS`/`IVGMM`に分ける案や、`t_stats`/`z_stats`の両方を持たせて片方を`None`にする案は、推定量を変えただけで結果の形が変わるため不採用）。`stat_dist`/`stat_df`は`TwoSlsEstimator::stat_dist()`/`GmmEstimator::stat_dist()`（`engine::shared::inference::StatDist`）から配線する。2SLSの`cov_type="cluster"`は`df_resid`ではなく`G-1`を使うため、`stat_df`は`df_resid`と一致するとは限らない。
