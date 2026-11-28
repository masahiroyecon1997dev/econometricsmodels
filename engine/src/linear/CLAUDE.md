# engine/src/linear/ 実装ノート（OLS/WLS）

このファイルは `engine/src/linear/` 配下のファイルを読み書きするときだけ自動ロードされる。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景・数式・実測は`docs/spec/ols-spec.md`・`docs/spec/wls-spec.md`・`docs/spec/orthogonal-coordinates.md`が正本（このファイルはその要約ではなく差分の索引）。`engine/src/shared/`の最小二乗の核（直交座標のQR・条件数のガード・全体Wald・共分散・クラスターのコード）の中身の罠は`.claude/rules/engine-least-squares-core.md`（このディレクトリのファイルを読むと一緒にロードされる）に、事後診断（`diagnostics/`）だけに関わる罠はサブディレクトリの`CLAUDE.md`に分けてある（下の「サブディレクトリのノート」）。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## モジュール構成

- `ols.rs`・`wls.rs`・`diagnostics.rs`は外から見える面（構造体・getter・`fit*`の契約と委譲・`wald_test_last_columns`・再エクスポート）だけを持つファサードで、ロジックは`ols/`（`fit`・`auxiliary`・`wald`・`cov_type`・`cov_params`・`input`・`predict`）・`wls/`（`fit`・`statistics`）・`diagnostics/`（`aux_regression`・`white`・`breusch_pagan`・`breusch_godfrey`）にある。`ols/cov_params.rs`は`shared::covariance`の部品をOLS固有の規則でつなぐ層で、`LeastSquaresError`を返すので`shared/`に置けない。`fit`を通すテストと`mod proptests`は`ols/fit/tests/`、`cov_params.rs`のテストは`ols/cov_params/tests/`にテーマ別。
- 共分散・Wald検定・直交座標のQR・適合度・グループ化の部品は`engine/src/shared/`（`covariance`・`wald`・`orthonormal`・`goodness_of_fit`・`group_codes`。`least_squares`はテスト用の独立な参照実装だけ）にあり、`linear`のエラー型に依存しない（`CommonError`・`RankDeficient`を返し、呼び出し側が`map_err`する）。`engine`は列名を知らないので、列名の重複など列名が要る検証は`engine_pybind`の責務（`engine_pybind/src/linear/CLAUDE.md`）。

## サブディレクトリのノート

次のファイルは、そのディレクトリ配下のファイルを読み書きするときに、このファイルと一緒に自動ロードされる。ファサードの`diagnostics.rs`を読むだけではロードされないので、ファサードだけを変えるときや、補助回帰（`ols/auxiliary.rs`）と診断をまたぐ変更では明示的に読む。

- `diagnostics/CLAUDE.md`: 事後診断（White・Breusch-Paganの補助回帰の標準化と項の選び方、`R²`の非有限の判定、Breusch-GodfreyのQRと行列の確保）。
- `.claude/rules/engine-least-squares-core.md`（`engine/src/linear/`と`engine/src/shared/`の`orthonormal.rs`・`linear_algebra.rs`・`wald.rs`・`covariance.rs`・`group_codes.rs`を読むとロードされる）: 直交座標のQRと条件数のガード、特異性・悪条件の判定と全体Wald、共分散の演算順序と`Par::Seq`、クラスターの蓄積、`GroupCodes`の順序とキー。

## `fit()`に入れるもの

- **全推定手法の`fit()`冒頭（と`engine_pybind`の`#[pymodule]`初期化）で`ensure_serial()`を呼び、グローバル並列度を引き上げるコードを入れない。並列化は実測で効く箇所だけ`Par::Rayon(_)`を対象APIへ明示的に渡す。`disable_global_parallelism()`は使わない**: tall-skinnyな設計行列の暗黙の全コア並列は、多コア機・負荷下で20〜200倍遅く不安定。`disable_global_parallelism()`は`get_global_parallelism()`をpanicさせる。回帰ガード: `ols/fit/tests/basic.rs`の`fit_pins_faer_global_parallelism_to_seq`。詳細: `.claude/rules/rust-style.md`「パフォーマンス」、`docs/performance/ols.md`。

## クラスターの推論

- **`cov_type=Cluster`は`G <= q`（`q = k - k_constant`）を`fit()`冒頭で`InsufficientClustersForInference`（`validate_cluster_count_covers_slopes`）にする。「クラスター数の境界の成功パス」のテストは`G > q`（厳密不等号）を保つ**: `rank(Ŝ) ≤ G - 1`で、`G = q`ちょうども常に特異（相対閾値が検出できるかがデータ配置次第だっただけ）。`G > q`の準多重共線性は`wald_f_test`の悪条件判定がbackstop。GMMの重み行列の`G < l`は別軸（`docs/spec/iv-spec.md`3.1節）。詳細: `docs/spec/ols-spec.md`3.2節「クラスター」。
- **検定の自由度は`cov_type=Cluster`のときだけ`G-1`。`df_resid`（σ̂²・調整済みR²・AIC/BIC）は常に`n-k`**: 詳細: 同上。
- **全体F（`f_statistic()`/`f_p_value()`/`f_df()`）は定義できないとき3つとも`None`（NaNにしない）。`G <= q`でF検定だけを`None`にする`fit_with_optional_f_test`はIVの第一段階専用で、`fit`のシグネチャは変えない**: `G < 2`・`groups`未指定・数値的なほぼ特異は`None`へ握りつぶさず失敗のまま。単体のOLS等で`G <= q`の拒否を緩めるかは別の設計論点。詳細: `docs/spec/ols-spec.md`2章、`docs/spec/iv-spec.md`3.1節。

## WLS

- **`WlsEstimator`のR²・調整済みR²・log_likelihood（→AIC/BIC）は、変換後データの素のOLS計算を使わず`weighted_fit_statistics`で計算し直す。重み付け系の手法（GLS等）を足すときも同じ見落としに注意する**: 重み付き平均のTSSとヤコビアン補正`+0.5*Σlog(w_i)`が要る。詳細: `docs/spec/wls-spec.md`3.4節。

## 推定量と補助回帰

- **公開の`OlsEstimator::fit`はk=0を`NoRegressors`で拒否する。0列を扱うのは`OrthonormalDesign`を直接使うFEだけ。新しい呼び出し元がk=0になりうるか迷ったら、「構造的に防がれている」（REのbetween・最終回帰）のか「`fit`の検査に守られているだけ」（WLS・IV）なのかを区別する**: 回帰ガード: `new_checked_accepts_a_design_without_columns`・`checked_col_piv_qr_accepts_matrix_without_columns`、WLSの`fit_propagates_no_regressors_error_when_k_is_zero`。詳細: `docs/spec/ols-spec.md`3.1節。
- **「一部の列だけを検定したい」ときは`OlsEstimator::wald_test_last_columns(q)`を再利用し、サンドイッチ計算を複製しない。外部で計算した共分散・自由度で検定したいときは`shared::wald::wald_f_test`を直接呼ぶ**: `wald_test_last_columns`はその推定量を`fit()`した`cov_type`の値（クラスターなら`G-1`）を前提にする。`cov_params`/`df_inference`は`OLSResult`に含めない。詳細: `docs/spec/ols-spec.md`2章。
- **被説明変数の定数（`OutcomeConstant`）・完全フィット（`PerfectFit`、`SSR <= k·ε·SST`）の判定は`fit`/`fit_with_optional_f_test`だけが行う。被説明変数が元のモデルの`y`でない回帰（診断の補助回帰、REの最終回帰）には`fit_auxiliary`か`AuxiliaryFit::fit(.., false)`を使う。新しい呼び出し元は「この回帰の結果を利用者にモデルの結果として見せるか」で選ぶ**: `SST`は`OlsInput`が元の`y`と重みから持つ（`outcome_sst`。WLSの変換後の`y`の平均で中心化した値は別の量）。ノイズなしの`y`をテストに使うと`PerfectFit`になる。詳細: `docs/spec/ols-spec.md`3.3節。
- **`fit_releasing_design`（設計行列をその場で分解して手放す）は`engine_pybind`のOLS・WLSだけが使い、後で`input().x()`を読む呼び出し元（REの最終回帰、IVの第一段階）は`fit`のまま。事前検査は`preflight`を`fit_ols`と`AuxiliaryFit::fit`で共有し、片方だけ変えない**: 手放した後に`OlsInput::x()`を呼ぶとpanicする。`fit`との違いは`fitted_values()`（`Q·b̃`）だけで、QR以降を変えたら`ols/fit/tests/release_design.rs`と`ols/auxiliary.rs`のテストが落ちる。詳細: `docs/spec/ols-spec.md`3.1節。
- **Python経由のピークRSSは、確保そのものをなくす変更と、mmapされる大きい行列の解放でしか下がらない。効果は`resource.getrusage`で変更前後を測って確かめる**: glibcの動的なmmap閾値のため、8MB級の列を早く`drop`してもRSSは下がらない。詳細: `docs/performance/ols.md`「メモリ」。
