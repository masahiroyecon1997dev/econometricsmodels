# engine_pybind/src/iv/ 実装ノート（IV: 2SLS/GMM）

このファイルは `engine_pybind/src/iv/` 配下のファイルを読み書きするときだけ自動ロードされる。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景は `docs/spec/iv-spec.md`（3.10節がバインディング）が正本（このファイルはその要約ではなく差分の索引）。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## 構成

- **`IVOptions`/`IVResult`/`build_iv_input`/`fit`は`iv/common/`に責務ごとのファイルで置き、`two_sls.rs`/`gmm.rs`のように手法ごとに分けない**: `fit_iv`が`IVOptions.estimator`で2SLSとGMMに振り分ける単一の入口。`cov_type`の値と列はOLS/WLSの`linear::common::parse_cov_kind`・`resolve_cov_type`を使い、組合せはGMMのオプションを持つ自前の`check_iv_options`。詳細: `docs/spec/iv-spec.md`3.10節。

## 検査の順序

- **構造方程式の検査（設定値→識別→観測数→クラスター数→定数`y`、`validate_iv_fit_inputs`）は`fit_frame`で`compute_first_stage`より前に呼ぶ。`q`・`k`・`l`の式をpybindに複製しない**: 第一段階の回帰変数・クラスター係数の数は構造方程式以上なので、先に第一段階を回すと構造方程式の検査が`FirstStageFailed`に隠れてPythonから到達できない。回帰ガード: `tests/iv/test_iv_validation.py`の`test_observation_count_boundaries`・`test_cluster_count_boundaries`・`test_cluster_count_between_structural_and_first_stage_slopes_leaves_first_stage_f_undefined`。詳細: `docs/spec/iv-spec.md`1.1節・3.1節。
- **`build_iv_input`の報告順（値→使われないオプション→`hac_time`の欠落→役割→列→オプションの列）を変えるときは`tests/cross_method/test_validation_order.py`を合わせる。GMMの重みの列も`build_iv_input`で抽出して`BuiltIvInput.plan`で`fit`に渡す。`gmm_max_iter`は`i64`のまま渡し、範囲の検査はengineに任せる**: 負値もengineが`GMM_MAX_ITER_OUT_OF_RANGE`にする。詳細: `docs/spec/iv-spec.md`3.10節。

## GMMのオプション

- **GMM専用オプション（`gmm_type`・`gmm_weight_type`・`gmm_max_iter`・`gmm_tol`・`raise_on_non_convergence`）は既定`None`にし、使われるモードのときだけ実効既定値（`two_step`・`classical`・`100`・`1e-6`・`true`）に解決する。使われないモードでの明示指定は`check_iv_options`で`ValidationError`にする**: pyclassは既定値と明示指定を区別できない。黙って無視すると設定の書き忘れに気づけない。`cluster`・`hac_lags`・`hac_time`は`cov_type`と`gmm_weight_type`のどちらか一方でも使えば有効。`"one_step"`は重みをengineに渡さない（結果の`gmm_weight_type`も`None`）。詳細: `docs/spec/iv-spec.md`3.3節。
- **`gmm_weight_type="hac"`も`hac_time`を必須にする（`require_hac_time`）。`cluster`・`hac_lags`・`hac_time`は`cov_type`と共用し、重み用の別フィールドを足さない**: 別のクラスター列が要るニーズが出たら別フィールドを検討する。詳細: `docs/spec/iv-spec.md`3.10節。

## 結果

- **`IVResult`は`fit.rs`の`build_result`が一度だけ組み立てる。フィールドを足すときは、2SLSとGMMで共通なら`IvEstimate`に、推定量ごとに違うなら`EstimatorSpecific`（`Default`なし）に足す**: どちらも両分岐の書き漏れをコンパイラが検出する。詳細: `docs/spec/iv-spec.md`3.10節。
- **統計量は全手法共通の名前`test_stats`にし、分布は`stat_dist`（2SLSは`"t"`、GMMは`"normal"`）と`stat_df`で示す。`IV2SLS`/`IVGMM`に型を分けたり`t_stats`/`z_stats`を両方持たせたりしない。2SLSのクラスターの`stat_df`は`G-1`で`df_resid`と一致しない**: 推定量を変えただけで結果の形が変わる。詳細: `docs/spec/iv-spec.md`3.2節、`docs/spec/inference-conventions.md`。
- **`first_stage()`は推定量を保持せず、`fit`が第一段階の`OlsEstimator`を`ols_estimator_to_result`ですぐ`OLSResult`に変えて手放し（`compute_first_stage_with`）、複製（`clone_ref`）を返す。2SLSの第一段階の二重計算をなくす変更は着手前にユーザーに確認する**: 設計行列を内生変数の数だけ同時に持たない。返す`OLSResults.f_statistic`は`x_exog`込みの通常のF検定で、部分F（`weak_instrument_f_statistics`）とは別物。詳細: `docs/spec/iv-spec.md`3.7節・3.10節。
- **GMMのWu-Hausmanには、`fit`が回した第一段階の`OLSResult`の残差を借用で`GmmEstimator::fit_with_first_stage_residuals`に渡す（第一段階のOLS残差でなければならない。engineは列の数と長さしか検査しない）**: 第一段階を二重に回さず、残差を複製しない。Wu-Hausmanが`None`でも`fit()`は失敗しない。詳細: `docs/spec/iv-spec.md`3.6節。
- **`converged`・`n_iter`は2SLSでも持つ（常に`true`・`1`）。`overid_*`は2SLSがSargan（`cov_type`に追従）、GMMがHansen J**: 結果の形を推定量で変えない。詳細: `docs/spec/iv-spec.md`2章・3.5節。

## 事後診断（`cragg_donald()`・`kleibergen_paap()`・`effective_f()`・`c_stat()`）

- **`fit()`では計算せず、`IVResult`が保持する`source_data`・`options`・列名から`diagnostics.rs`の`Rebuilt`が`fit()`と同じ`build_iv_input`で入力を作り直し、`py.detach`の中で`engine`の`compute_*`を呼ぶ**: 列の検査と並びが`fit()`と一致する。`source_data`と`options`は結果を組み立てた後でGILを持って付ける。詳細: `docs/spec/iv-spec.md`3.4.4節。
- **`c_stat()`の名前は、GILを持つ側で一覧の検査（`ROLE_EMPTY`・`ROLE_DUPLICATE_COLUMN`）と位置への変換を済ませ、モデルに無い名前は`COLUMN_NOT_FOUND`ではなく`C_STATISTIC_UNKNOWN_VARIABLE`にする。両方未指定はengineの`C_STATISTIC_NOTHING_TO_TEST`に任せる**: `COLUMN_NOT_FOUND`の`hint`は`data`の列を指すので合わない。重みは`two_step`・`iterated`なら`fit()`の重み、それ以外（2SLS・1段階GMM）は`weight_type_for_cov_type`。新しい事後診断のメソッドは`test_pickle.py`の`TRANSIENT`・`test_gil_release.py`・`test_keyword_only_arguments.py`に登録する。詳細: `docs/spec/iv-spec.md`3.9.1節。

## Rust側の罠

- **engine側で共有エラー型（`IvError`等）に新しいバリアントを足したら、`cargo build -p engine`だけでなく`cargo test --workspace`まで確かめる**: `iv_error_to_pyerr`は`to_pyerr`を呼ぶだけで`engine_pybind`の更新は要らないが、パッケージ境界を跨ぐ破壊は`-p engine`のビルドでは検出できない。
- **`#[cfg(test)]`のテストからしか呼ばれない関数（とそこから呼ばれる関数）に`#[expect(dead_code)]`を使わず、`#[allow(dead_code)]`にする。`#[expect]`は「テストからも含めてどこからも呼ばれない」関数にだけ使う**: `cargo build`では`dead_code`が発火するが、`cargo test`・`clippy --all-targets`ではテストが呼ぶので発火せず、`unfulfilled_lint_expectations`が`-D warnings`でエラーになる。手法をデータ抽出とengine呼び出しの2段階に分けて実装するときに踏みやすい。
