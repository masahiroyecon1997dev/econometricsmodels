# engine_pybind/src/iv/ 実装ノート（IV: 2SLS/GMM）

このファイルは `engine_pybind/src/iv/` 配下のファイルを読み書きするときだけ自動ロードされる。設計の背景は `docs/spec/iv-spec.md` が正本。ここは差分の索引のみ。

## 踏んだ罠（再発防止）

- **`engine`側で共有エラー型（`IvError`等）に新バリアントを追加するときは、`cargo build -p engine`だけでなく`cargo test --workspace`（または少なくとも`-p engine_pybind`）まで確認する**: `iv_error_to_pyerr`（本ファイル）は`to_pyerr(&err)`を呼ぶだけで`IvError`を`match`しないため、`engine_pybind`側の更新は要らない。新バリアントの`code`は`engine`の`ErrorInfo::report()`（`_`なしで全バリアントを列挙するため、足し忘れは`engine`自体のコンパイルエラー）が決める。`cargo build -p engine`はパッケージ境界を跨ぐ破壊を検出できないため、ワークスペース全体で確かめる。

- **`#[cfg(test)] mod tests`からしか呼ばれない関数に`#[expect(dead_code, ...)]`を使うと`--all-targets`ビルドで`unfulfilled_lint_expectations`エラーになる**。`#[expect]`は「指定したlintが実際に発火する」ことを検証する属性のため、以下の非対称性が問題になる。
  - `cargo build`（テストコードを含まない）: 関数が本当に未到達 → `dead_code`が発火 → `#[expect]`が正しく警告を吸収する。
  - `cargo clippy --all-targets -- -D warnings` / `cargo test`（テストコードを含む）: `#[cfg(test)] mod tests`内のテストがその関数を実際に呼ぶため到達可能になる → `dead_code`が発火しない → `#[expect]`の期待が外れ`unfulfilled_lint_expectations`が`-D warnings`下でエラーになる。
  - この罠は`build_iv_input`/`parse_iv_cov_type`が自分自身のテストから呼ばれる場合だけでなく、それらが呼ぶ先（`iv_error_to_pyerr`）にも伝播する（`build_iv_input`経由でテストから間接的に到達可能になるため）。
  - **対処**: テストから実際に呼ばれる「本番未接続」関数（Logitの`build_logit_input`、IVの`build_iv_input`/`parse_iv_cov_type`/`iv_error_to_pyerr`と同じパターン）には`#[allow(dead_code)]`（無条件に抑制、`cargo build`/`cargo test`どちらでも警告を出さない）を使う。`#[expect(dead_code, ...)]`は「テストからも含めてどこからも一切呼ばれていない」関数（`iv_error_to_pyerr`が当初はそうだった）にのみ適格。次に手法を2段階（データ抽出段階→engine呼び出し段階）に分けて実装するとき（GMM等）も同じ罠を踏む可能性が高いため注意する。

## 実装フェーズの分割方針

Logitと同じ2段階に分けた。

1. **データ抽出・pyclass定義段階**: `IVOptions`/`IVResult`のpyclass定義、列抽出・バリデーション・`engine::iv::common::IvInput`構築までを行う`build_iv_input`を実装した。この時点では`#[pymodule]`への登録・実際の`TwoSlsEstimator::fit`呼び出しは行わなかった。
2. **engine呼び出し・エラー変換段階**: `build_iv_input`を実際に呼び出す`fit`関数を追加し、`lib.rs`に`#[pyfunction] fit_iv`を新設して`#[pymodule]`に登録した。この時点で`iv_error_to_pyerr`/`parse_iv_cov_type`/`build_iv_input`の`#[allow(dead_code)]`属性はすべて削除済み（本番経路から呼ばれるようになったため）。

`IVOptions`/`IVResult`/`build_iv_input`/`fit`は`iv/common.rs`に置く（`two_sls.rs`/`gmm.rs`のような手法ごとのファイル分割はしない）。`fit_iv`という単一エントリポイントを`IVOptions.estimator`（`"2sls"`/`"gmm"`）で2SLS/GMMに振り分ける設計のため、これらは系統内で真に共有されるロジックであり、`<系統>/common.rs`に置くという既存方針にそのまま合致する。

**上記・下記の`parse_iv_cov_type`への言及は上記1・2段階目時点の実装経緯としてそのまま残しているが、この関数自体は2026-09-20対応で削除済み**。`linear::common::parse_cov_type`（OLS/WLS用）と型・matchアーム・エラーメッセージが完全同一だったため、`IVOptions`の同名フィールド（`cov_type`/`cluster`/`hac_lags`/`hac_time`）を個々の引数として渡す形でそちらへ統合した。現在`build_iv_input`は`cov_type`の値を`crate::linear::common::parse_cov_kind`、列の抽出を`resolve_cov_type`で行う（組合せの検査はIV自身の`check_iv_options`）。

`weak_instrument_f_statistics`（空`HashMap`）・`overid_statistic`/`overid_p_value`・`wu_hausman_statistic`/`wu_hausman_p_value`（いずれも`None`）は`fit`ではプレースホルダーのまま返す。実際の計算はそれぞれ別途行う。

**`weak_instrument_f_statistics`は後日配線済み**: `TwoSlsEstimator::weak_instrument_f_statistics()`（`&[(String, f64)]`）を`.iter().cloned().collect()`で`HashMap<String, f64>`に詰め替えるだけ（`fit`、`iv/common.rs`）。`overid_statistic`系はまだ未着手のため引き続きプレースホルダー。

**`wu_hausman_statistic`/`wu_hausman_p_value`も後日配線済み**: `TwoSlsEstimator::wu_hausman_statistic()`/`wu_hausman_p_value()`（どちらも`Option<f64>`）をそのまま代入するだけ（`weak_instrument_f_statistics`と異なり型変換が要らない）。`engine`側の判断で`x_endog=[]`だけでなく拡張回帰が特異な場合（第一段階残差の分散がゼロ等）も`None`になる——この場合も`fit()`自体は失敗しない（`engine/src/iv/CLAUDE.md`「Wu-Hausmanの拡張回帰が特異な場合は…」参照）。

3. **`first_stage()`メソッド段階**: 当初は`IVResult`に非公開フィールド`estimator: TwoSlsEstimator`を追加し（`LogitResult`/`ProbitResult`が`predict()`/`marginal_effects()`用に推定量そのものを保持するのと同じパターン）、`first_stage()`が`estimator.first_stage_estimators()`から`dict[str, OLSResults]`をオンデマンドに構築する設計だった（**GMM配線時にこの`estimator`フィールドは廃止、下記「GMM配線」節参照**）。`OlsEstimator → OLSResult`変換は新設した`linear::ols::ols_estimator_to_result`（`linear::ols::fit`本体から抽出、`pub(crate)`）を再利用する——第一段階回帰はそれ自体が正しい（ナイーブな）通常のOLS回帰であり（`engine::iv::two_sls`のモジュールdocコメント参照）、2SLSの第二段階（サンドイッチ型分散を独自実装）とは異なりOLSとの共有を避ける理由が無いため。`first_stage()`が返す各`OLSResults.f_statistic`/`f_p_value`は通常のOLS F検定（`x_exog`の寄与を含む）であり、弱操作変数診断の部分F統計量（`weak_instrument_f_statistics`）とは別物（`IVResult`のdocコメント参照）。

## GMM配線（本ファイル冒頭「実装フェーズの分割方針」に続く4段階目、engine側のGMM cov_type対応完了後に実施）

**`estimator="gmm"`は実装済み**（当初`GmmEstimator`が点推定のみ・engine側cov_type対応も無かったため`ValidationError`を返していたが、`engine::iv::gmm::GmmEstimator`にcov_type対応SEを実装したうえで本ファイルにも配線した）。

- **GMMの方式は`IVOptions.gmm_type`（`"one_step"`/`"two_step"`（実効既定）/`"iterated"`）で選び、GMM専用オプション（`gmm_type`/`gmm_weight_type`/`gmm_max_iter`/`gmm_tol`/`raise_on_non_convergence`）は全て`Option`で既定`None`**（`resolve_gmm_setup`が`engine::iv::gmm::GmmType`に変換する）。`IVOptions`はpyclassで既定値と明示指定を区別できないため既定値を`None`にし、使われるモードのときだけ実効既定値（`"two_step"`/`"classical"`/`100`/`1e-6`/`true`）に解決する。使われないモードでの明示指定は`check_iv_options`が`ValidationError`にする（`estimator="2sls"`では全GMM専用オプション、`"one_step"`では`gmm_weight_type`、`"one_step"`/`"two_step"`では`gmm_max_iter`/`gmm_tol`/`raise_on_non_convergence`）。`gmm_type`/`gmm_weight_type`の文字列が未知の値のときは、使われるかどうかによらず値の検査（`parse_iv_option_values`）で`unknown ...`を報告する。`cluster`/`hac_lags`/`hac_time`は`cov_type`と`gmm_weight_type`の両方が参照するため「どちらか一方でも使えば有効」で判定する（`linear::common::check_cov_options`ではなく`check_iv_options`が両方を見る）。`"one_step"`は`gmm_weight_type`をengineに渡さない（結果の`gmm_weight_type`も`None`）。`IVResult.gmm_type`は`estimator="gmm"`のとき正規形に直した値、`"2sls"`では`None`。
- **`resolve_weight_type`**（`cov_type`側の`linear::common::resolve_cov_type`と対になる関数。`gmm_weight_type`は`WeightType`という`cov_type`側の`CovType`とは異なる型を組み立てるため独立実装のまま）が`WeightKind`に対応する列を抽出して`engine::iv::gmm::WeightType`にする。**`gmm_weight_type="hac"`も`hac_time`が必須**（`cov_type="hac"`と同じ`linear::common::require_hac_time`、未指定は`ValidationError`で、メッセージは要求している設定名`gmm_weight_type`を示す。欠落の検査は`check_iv_options`で`cov_type`が先、`gmm_weight_type`が後）。`cluster`/`hac_lags`/`hac_time`は`cov_type`と共用（`IVOptions`に別フィールドを増やさない設計、`gmm_weight_type`と`cov_type`が異なるクラスター変数を使いたいニーズが出てきたら別フィールド化を検討）。
- **`IVResult`の非公開フィールドを`estimator: TwoSlsEstimator`から`first_stage`に置き換えた**（`estimator`非依存の表現にするため）。当初は`Vec<(String, OlsEstimator)>`だったが、pickle対応で`fit`が`OLSResult`へ変換して`Vec<(String, OLSResult)>`として保持する形にした（設計行列を持つ`OlsEstimator`を手放せる。`first_stage()`は`OLSResult::clone_ref`で複製して返し、pickleでは各`OLSResult`がそのまま状態に入る）。`first_stage`/`weak_instrument_f_statistics`は`estimator`によらず`engine::iv::common::compute_first_stage`（`engine/src/iv/CLAUDE.md`参照、2SLS/GMM間で共有するロジックとして抽出済み）から構築する。**`estimator="2sls"`では第一段階回帰が二重計算になる**（`fit`が明示的に1回、`TwoSlsEstimator::fit`が内部でもう1回）——`OlsEstimator`が`Clone`未実装のため`TwoSlsEstimator::first_stage_estimators()`の借用結果を`IVResult`へ所有権ごと移せず、OLS自体が軽量という前提で許容した設計判断（rust-reviewerの指摘で認識済み、恒久対応する場合は`TwoSlsEstimator`に第一段階結果を外部注入する`fit`のバリエーションを追加する案がある。着手前にユーザー確認すること）。
- **識別の順序条件（`k_instruments < k_endog`）チェックは`compute_first_stage`呼び出しより前に行う**（`fit`冒頭、`compute_first_stage`自体はこの条件を検証しないため、過小識別な入力で無駄な第一段階回帰が走るのを防ぐ、rust-reviewerの指摘で追加）。
- **観測数のチェック（`engine::iv::common::validate_observation_counts`）も`compute_first_stage`より前に`fit`で行う**: 構造方程式の`n <= k`を先に`CommonError::InsufficientObservations`（構造方程式の`k`）で、次に`n <= l`（`l = k_exog + k_instruments`、定数項込みの第一段階の回帰変数の数）を`IvError::InsufficientObservationsForFirstStage`（`ValidationError`）で弾く。`n == l`では`X̂ = X`で2SLSがOLSに縮退し（GMMの`W₀ = (Z'Z)⁻¹`も同様）、`n < l`では`Z'Z`が特異のため、第一段階を診断としてdegradeする余地はなく拒否のまま。以前は第一段階の`OlsEstimator::fit`の`n <= k_first`が先に`FirstStageFailed`として発火し、構造方程式の`n <= k_struct`がPythonから到達不能で、メッセージも第一段階のOLSの`k`になっていた（`tests/iv/test_iv_validation.py`の`test_observation_count_boundaries`で固定）。
- **`cov_type=Cluster`の構造方程式の`G <= q`チェックも`compute_first_stage`より前に`fit`でも行う**（`engine::iv::common::validate_structural_cluster_count`を呼ぶ。`q`の式はエンジン側に集約し、pybindに複製しない）。第一段階の`q`は識別条件により構造方程式以上のため、先に第一段階を走らせると常に`FirstStageFailed`（第一段階の`g`・`q`）に隠れ、エンジン側の構造方程式チェックがPythonから到達不能になっていた。構造方程式で`G>q`でも第一段階の`q`が`G`以下なら、`fit`は失敗せず、その第一段階の`OLSResult`（`first_stage()`）の`f_statistic`/`f_p_value`/`f_df_num`/`f_df_denom`だけが`None`になる（`test_cluster_count_between_structural_and_first_stage_slopes_leaves_first_stage_f_undefined`/`test_cluster_count_boundaries`で固定）。`OLSResult`のF系列は全手法（OLS/WLS/FE/RE）で`Option<f64>`（以前の「傾き係数が無いときNaN」は`None`に統一した）。
- **`wu_hausman_statistic`/`wu_hausman_p_value`は`estimator="gmm"`では常に`None`**（`GmmEstimator`はWu-Hausman検定を実装しない）。`overid_statistic`/`overid_p_value`は`estimator="gmm"`では`GmmEstimator::hansen_j_statistic()`/`hansen_j_p_value()`から構築する（Hansen J検定、`estimator="2sls"`のSargan検定と対）。
- **`IVResult`に`converged: bool`/`n_iter: i64`を追加**（rust-reviewerの指摘: `raise_on_non_convergence=False`を指定してもGMMが収束したかをPython側で確認する手段が元々無かった、`LogitResult`/`ProbitResult`の`converged`/`n_iter`と同じ位置づけ）。`estimator="2sls"`では常に`converged=true`・`n_iter=1`（2SLSは閉形式・非反復のため）。

## オプションの検査（値・組合せ・列）と報告順

`build_iv_input`は、DataFrameを見ない検査（`parse_iv_option_values`: `estimator` → `cov_type` → `gmm_type` → `gmm_weight_type`の値。`check_iv_options`: 使われないオプション → 必須の`hac_time`の欠落）→ ロール構造 → `y`/`x_exog`/`x_endog`/`instruments`の列 → オプション列（`cov_type`側は`resolve_cov_type`、GMMの重み行列側は`resolve_gmm_setup`）の順に呼ぶ。GMMの重み行列の列も`fit`ではなく`build_iv_input`で抽出し、戻り値の`IvPlan`（`IvEstimator`と、`estimator="gmm"`のときの`GmmSetup`）で`fit`に渡す。`*Result`の`cov_type`/`estimator`/`gmm_type`/`gmm_weight_type`の正規形は`CovKind::name()`/`IvEstimator::name()`等から作り、呼び出し側で小文字化し直さない。この順序は`tests/cross_method/test_validation_order.py`が固定している。`gmm_max_iter`が負値のときの変換エラーは`resolve_gmm_setup`で起きるが、範囲（3以上・上限）と`gmm_tol`の検証はengineが行う。

## `IVResult.test_stats`と`stat_dist`/`stat_df`

`IVResult`は`estimator="2sls"`（t分布）・`estimator="gmm"`（z分布、`docs/spec/iv-spec.md`3.2節）の両方で共有される単一の型のため、統計量は全手法共通の名前`test_stats`とし、分布は`stat_dist`（`"t"`/`"normal"`）と`stat_df`（t分布の自由度、正規分布は`None`）で示す（他の手法のResultsと同じ形。IVを`IV2SLS`/`IVGMM`に分ける案や、`t_stats`/`z_stats`の両方を持たせて片方を`None`にする案は、推定量を変えただけで結果の形が変わるため不採用）。`stat_dist`/`stat_df`は`TwoSlsEstimator::stat_dist()`/`GmmEstimator::stat_dist()`（`engine::shared::inference::StatDist`）から配線する。2SLSの`cov_type="cluster"`は`df_resid`ではなく`G-1`を使うため、`stat_df`は`df_resid`と一致するとは限らない。
