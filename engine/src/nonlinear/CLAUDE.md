# engine/src/nonlinear/ 実装ノート（Logit/Probit/Tobit）

このファイルは `engine/src/nonlinear/` 配下のファイルを読み書きするときだけ自動ロードされる。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景・数式・実測は `docs/spec/nonlinear-common.md`（最適化・収束・標準化・メモリ）/ `docs/spec/logit-spec.md`（分離検出）/ `docs/spec/probit-spec.md` / `docs/spec/tobit-spec.md` / `docs/spec/orthogonal-coordinates.md` が正本（このファイルはその要約ではなく差分の索引）。ソルバー・Tobit・分離検出だけに関わる罠はサブディレクトリの`CLAUDE.md`に分けてある（下の「サブディレクトリのノート」）。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## ファイル構成

`logit.rs`・`probit.rs`・`tobit.rs`・`separation.rs`は外から見える面（構造体・getter・`fit`の契約と委譲・再エクスポート）だけを持つファサードで、ロジックは同名のディレクトリ（`logit/`等）の子モジュールにある（入力・数式・目的関数・`*Fitted`・`fit`の本体）。共有部品の`common.rs`は再エクスポートだけで、中身は`common/`（`error`・`validation`・`options`・`goodness_of_fit`・`marginal_effects`・`predict`・`normal`・`numerics`・`solver`・`line_search`・`newton`・`bfgs`・`lbfgs`・`initial`・`standardize`・`cov`）に置く。テストはロジックと同じファイルの`mod tests`にあり（`fit`を通すテストと、テストが大きいファイルは`<ファイル名>/tests/`にテーマ別で置く）、複数ファイルのテストが使うフィクスチャは各ディレクトリの`test_support.rs`（`common/test_support.rs`は`logit`/`probit`のテストも使うので`pub(crate)`）。

## サブディレクトリのノート

次のファイルは、そのディレクトリ配下のファイルを読み書きするときに、このファイルと一緒に自動ロードされる。ファサード（`common.rs`・`tobit.rs`・`separation.rs`）を読むだけではロードされないので、ファサードだけを変えるときや、トピックをまたぐ変更では該当ファイルを明示的に読む。

- `common/CLAUDE.md`: ソルバー（`solver`・`newton`・`bfgs`・`lbfgs`・`line_search`。初期化・停止・副次収束・返却点・line searchの評価回数）と標準化（`standardize`）。
- `tobit/CLAUDE.md`: Tobitの`y`のスケーリング・`Q`の座標・裾の数値（`expected_uncensored`・`W`の桁落ち）・property-basedテスト。
- `separation/CLAUDE.md`: 分離検出（LPによる判定・`DenseSimplex`の数値の工夫・LPの行の作り方）。

## `*Estimator`と`*Fitted`（予測・限界効果に必要な最小の状態）

- **`predict`・`predict_new_data`・`pred_table`・`marginal_effects`・`censoring_fit_check`の実装は`*Fitted`に置き、`*Estimator`は委譲するだけ。新しいMLE手法で`predict`等を持つときも同じ形にする**: `engine_pybind`の結果は`into_fitted()`だけを持ち、pickleの復元でも`*Fitted`を使う。標準誤差・適合度・収束情報は`*Estimator`だけが持つ。
- **`*Fitted`の構築経路は`fit`と`from_parts`の2つだけ。新しい復元の経路も`from_parts`を通す**: Tobitの打ち切り境界の検査（`TobitInput::validate_censoring_bounds`）は`from_columns`ではなく`fit()`と`from_parts`が行う。最適性と`params`・`cov_params`の有限性は検査しない（壊れたpickleだけが`CommonError::Internal`→`PICKLE_FORMAT_MISMATCH`に到達する）。詳細: `docs/spec/pickle-support.md`。

## 新しいMLE手法の`fit()`に必ず入れるもの

- **各手法の`fit()`冒頭（`run_solver`ではなく）で`crate::shared::parallelism::ensure_serial()`を呼ぶ**: faerの暗黙の全コア並列はtall-skinnyな設計行列で20〜200倍遅く不安定になる。回帰ガード: `logit/fit/tests/basic.rs`の`fit_pins_faer_global_parallelism_to_seq`。詳細: `.claude/rules/rust-style.md`「パフォーマンス」。
- **切片ありの定数の`y`は、`y`の形の検査の後で`validate_outcome_not_constant`により最適化の前に`OutcomeConstant`（`ValidationError`）にする**: 弾かないとLogit/Probitは`SeparationDetected`、Tobitは`σ→0`の`NonConvergence`になり原因が分からない。Logit/Probitは`validate_fit_preconditions`の末尾、Tobitは`validate_has_uncensored_observations`の後（`NoUncensoredObservations`が先）。切片なしは対象外。詳細: `docs/spec/error-handling.md`5.4章。
- **標準化の後・`run_solver`の前に`ols_based_initial_params`（Tobitは`DesignCoordinates::new`）を通し、ランク落ちを`SingularDesignMatrix`で弾く**: solverごとに検出経路が分かれると、`bfgs`だけ見逃す実バグがあった。**共線な列の特定は標準化前の`x`で行う**（中心化した`x_std`ではダミーの罠や定数列が切片を含まない形に見え、他の手法と報告が食い違う）。QRは`x_std`をその場で壊すので、呼び出し側が元の`x`から標準化し直す。回帰ガード: `logit/fit/tests/validation.rs`・`probit/fit/tests/validation.rs`の`fit_returns_singular_design_matrix_error_for_perfectly_collinear_design_matrix`、`common/initial.rs`の`ols_based_initial_params_reports_the_intercept_for_collinearity_hidden_by_centering`。詳細: `docs/spec/nonlinear-common.md`1.4節。
- **`cost`（対数尤度の総和）は`compensated_sum`で書く（勾配・Hessianの和は素朴な加算でよい）**: Newtonの`BelowCostResolution`判定は評価誤差が約1 ULPであることが前提で、素朴な和（約100 ULP）だとLMラダーの空回りが再発する。回帰ガード: `compensated_sum_*`、`common/newton/tests/stall.rs`の`run_solver_newton_skips_cost_comparison_below_cost_resolution`。詳細: `docs/spec/nonlinear-common.md`1.2節「コスト（対数尤度の総和）は補償和で計算する」。
- **スコアを使う`cov_type`は`run_solver_keeping_problem`が返す問題でスコアを評価し、最適化の前に`problem`を複製しない**: 複製は標準化した設計行列を含む`n·k`級。詳細: `docs/spec/nonlinear-common.md`1.5節。
- **2値選択モデルは収束後に`separation::verify_no_separation`を1回呼ぶ（順序モデルは`verify_no_ordered_separation`。Tobitは呼ばない）**: Tobitの分離は`σ→0`の退化として現れ、`y∈{0,1}`前提の符号条件が使えない。分離検出の罠は`separation/CLAUDE.md`。
- **`Φ`・生存関数・z検定のp値は`shared::normal`（`normal_cdf`・`normal_sf`・`StandardNormal`）を通し、`statrs`の`Normal::cdf`・`sf`を使わない**: `statrs`の`erfc`は相対誤差が最大約`1e-10`。密度`φ`と`inverse_cdf`は`statrs`のままでよい。テストで独立に再計算する`Φ`も`statrs`を参照にしない（`1e-12`の比較が合わない）。回帰ガード: `shared/normal.rs`のテスト（130桁の参照値）。詳細: `docs/spec/nonlinear-common.md`4章。

## Probit/TobitのHessianの重み

- **Probit/TobitのHessianの重みは、`λ`と`z`（`zeta`）を同じクランプ済みの引数から作る。新しい重み（期待情報等）を足すときも同じ**: 混在させると`|z|>U_CLAMP`の誤分類観測で重みが負になり、大域凹性が数値的に破れる。回帰ガード: `probit/problem.rs`の`hessian_weight_is_non_negative_even_when_misclassified_observation_exceeds_u_clamp`・`hessian_weight_is_non_negative_at_extreme_params`、Tobitの`censored_hessian_weight_is_non_negative_at_extreme_params`。詳細: `docs/spec/probit-spec.md`4章、`docs/spec/tobit-spec.md`4章。

## fit中のメモリ（ピークRSS）

- **Python側のピークRSSに効くのは、確保そのものをなくす変更と、mmapされる大きい行列の解放だけ。変更は`resource.getrusage`で前後のピークRSSを測って確かめる（ライブヒープの`engine/examples/memory_probe.rs`は見積りで代わりにならない）**: glibcの動的なmmap閾値のため、8MB級の列を早く解放してもRSSは下がらない。詳細: `docs/spec/nonlinear-common.md`1.5節、`docs/performance/ols.md`「メモリ」。
- **設計行列の複製を、最適化に使い続ける行列と同時に生かさない**: Logit/Probitの`ols_based_initial_params`はその場でQRし（`col_piv_qr_in_place_lstsq`）、Tobitは`OrthonormalDesign::new_checked_owned`で`x_std`を消費する。スコアを使う`cov_type`は上の`run_solver_keeping_problem`。詳細: `docs/spec/nonlinear-common.md`1.5節。
