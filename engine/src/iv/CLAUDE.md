# engine/src/iv/ 実装ノート（2SLS/GMM）

このファイルは `engine/src/iv/` 配下のファイルを読み書きするときだけ自動ロードされる。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景・数式・実測は`docs/spec/iv-spec.md`（GMMの白色化は`docs/spec/orthogonal-coordinates.md`4.2節）が正本（このファイルはその要約ではなく差分の索引）。GMMだけ、または`common/`の第一段階・診断だけに関わる罠はサブディレクトリの`CLAUDE.md`に分けてある（下の「サブディレクトリのノート」）。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## ファイル構成

- `two_sls.rs`・`gmm.rs`・`common.rs`は外から見える面（構造体・getter・`IvEstimate`実装・`fit`の契約と委譲・再エクスポート）だけを持つファサードで、ロジックは`two_sls/`（`fit`・`cov`・`overid`・`sargan`）・`gmm/`（`fit`・`hansen_j`・`sandwich`・`c_statistic`）・`common/`（`error`・`input`・`validation`・`wu_hausman`・`first_stage`・`estimate`・`weak_instruments`・`proptest_support`）にある。テストはロジックと同じファイルの`mod tests`（`fit`を通すテストと大きいファイルは`<ファイル名>/tests/`にテーマ別）。
- **GMMの点推定の反復（`GmmType`・`gmm_iterate`・収束判定）・重み行列`S`（`WeightType`・`moment_covariance`・`validate_weight_type`）・白色化した座標の解（`shared/gmm.rs`）・C統計量の核は`engine/src/shared/`にある**（`gmm_iteration.rs`・`gmm_weight.rs`・`gmm.rs`・`gmm_c_statistic.rs`）: 固定効果つきGMM（`panel/fe_iv/gmm`）がwithin変換後の行列で同じコアを呼ぶため、行列だけを受け取り`CommonError`を返す。2SLSの行列版の部品（点推定・Sargan・部分F・Wu-Hausmanの列）は`shared/instrumental_variables.rs`、弱操作変数診断の核は`shared/`の`weak_instruments.rs`・`kleibergen_paap.rs`・`effective_f.rs`・`score_covariance.rs`。これらを`fit`越しに確かめるテストは`gmm/fit/tests/`（`iteration.rs`・`weight_validation.rs`・`weight_robust_hac.rs`・`weight_cluster.rs`）。

## サブディレクトリのノート

次のファイルは、そのディレクトリ配下のファイルを読み書きするときに、このファイルと一緒に自動ロードされる。ファサード（`two_sls.rs`・`gmm.rs`・`common.rs`）や、核を置いた`engine/src/shared/`のファイル（`gmm*.rs`・`weak_instruments.rs`・`kleibergen_paap.rs`・`effective_f.rs`等）を読むだけではロードされないので、それらを変えるときや、トピックをまたぐ変更では該当ファイルを明示的に読む。

- `gmm/CLAUDE.md`: Hansen J（`n`で割らない・`OneStep`と`Classical`の`S`）とGMMのWu-Hausmanの順序。
- `.claude/rules/engine-gmm-core.md`（`gmm/`・`gmm.rs`と`engine/src/shared/gmm*.rs`・FE-GMMを読むとロードされる）: GMMの方式・収束判定・重み行列`S`・白色化した座標・サンドイッチ・`cov_type`の中心化と小標本補正・C統計量。
- `common/CLAUDE.md`: 第一段階の`OlsInput`、弱操作変数の診断（部分F・Cragg-Donald・Kleibergen-Paap・有効F）、Wu-Hausman。

## 新しいIV系手法の`fit()`に入れるもの

- **`fit()`冒頭で`crate::shared::parallelism::ensure_serial()`を呼ぶ**（第一段階の`OlsEstimator::fit`でも呼ばれるが冪等）: faerの暗黙の全コア並列はtall-skinnyな設計行列で20〜200倍遅く不安定になる。回帰ガード: `two_sls/fit/tests/basic.rs`の`fit_pins_faer_global_parallelism_to_seq`。詳細: `.claude/rules/rust-style.md`「パフォーマンス」。
- **入力の検証は`common/validation.rs`の`validate_iv_fit_inputs`に集約し、2SLS・GMMの`fit()`と`engine_pybind`（`compute_first_stage`より前）が共有する。`k`・`l`・`q`の式を呼び出し側に複製しない**: 順序は`confidence_level` → `gmm_max_iter` → `gmm_tol` → 識別 → 観測数 → クラスター数 → 定数`y`。`engine_pybind`が先に呼ばないと第一段階が先に失敗し、Pythonから構造方程式の検証に到達できない。詳細: `docs/spec/iv-spec.md`1.1節。

## 入力検証とクラスター

- **`cov_type=Cluster`のクラスター数は構造方程式の傾きの数`q = k - k_constant`で`G <= q`を弾く（`InsufficientClustersForInference`）。第一段階で`G <= q_first`になるだけなら第一段階の全体F検定だけを`None`にし、Wu-Hausmanの拡張回帰で`G <= q_aug`になるなら`wu_hausman_*`を`None`にする**: `rank(Ŝ) ≤ G-1`でWaldの`q×q`部分行列が構造的に特異。構造方程式は全体Waldが必須の出力、第一段階とWu-Hausmanは診断という線引き。回帰ガード: `two_sls/cov/tests/validation.rs`の`fit_returns_validation_error_when_cluster_count_at_most_slopes`、`common/first_stage.rs`の`compute_first_stage_leaves_overall_f_undefined_when_g_equals_first_stage_slopes`・`compute_first_stage_computes_overall_f_when_g_exceeds_first_stage_slopes`、`tests/iv/test_iv_validation.py`の`test_cluster_count_boundaries`。詳細: `docs/spec/iv-spec.md`3.1節・3.7節。
- **観測数は構造方程式の`n <= k`（`InsufficientObservations`）→第一段階の`n <= l`（`InsufficientObservationsForFirstStage`）の順に弾く。`n <= l`を診断の`None`へdegradeしない**: `n == l`で第一段階が完全に当てはまり`X̂ = X`（2SLS・GMMともOLSに縮退して「成功」する）、`n < l`で`Z'Z`が特異。`l >= k`なので構造方程式を先に報告しないと隠れる。回帰ガード: `common/validation.rs`の`validate_observation_counts_*`、`two_sls/fit/tests/validation.rs`・`gmm/fit/tests/validation.rs`の`fit_returns_insufficient_observations_for_first_stage_when_n_is_at_most_l`。詳細: `docs/spec/iv-spec.md`1.1節。
- **`gmm_weight_type=Cluster`は`cov_type`の`G <= q`とは別軸で、全操作変数の数`l`に対して過剰識別は`G < l`、丁度識別は`G <= l`を弾く（`CommonError::InsufficientClustersForWeightMatrix`、`validate_weight_type`。`OneStep`は対象外）**: `S`はG個のランク1行列の和で`rank(S) ≤ G`、丁度識別は`Z'ê=0`でさらに`G-1`。**クラスターの境界値テストを書くときは、常に`G ≥`対象の行列の次元を確かめる**。回帰ガード: `gmm/fit/tests/weight_validation.rs`の`fit_returns_insufficient_clusters_for_weight_matrix_*`・`fit_succeeds_when_*_g_equals_l`・`fit_succeeds_when_just_identified_and_g_exceeds_l`。詳細: `docs/spec/iv-spec.md`3.1節。
- **クラスター列は`engine_pybind`が一度だけ整数コード（`GroupCodes`）にして渡し、engineは`String`列を受け取ってコード化し直さない。`cov_type`と`gmm_weight_type`のコードは同じ列名でも別物として扱う**: 第一段階・Wu-Hausmanの拡張回帰・反復GMMで同じコードを借りる。詳細: `docs/spec/iv-spec.md`3.1節。
- **定数の`y`（切片あり）は第一段階より前に`validate_outcome_not_constant`で弾く。完全フィットは構造残差`y - Xβ̂`で判定する（2SLSは`ssr`の直後、GMMは初期推定の残差で`S`が0になる前）。第二段階（`X̂`への回帰）と部分Fの制限モデルでは判定しない**: 第二段階は別の量。テストで操作変数が内生変数を完全に予測するデータ（`x_endog = z`）を使うと第一段階が`PERFECT_FIT`（`stage=FIRST_STAGE`）で失敗する。詳細: `docs/spec/iv-spec.md`1.1節。

## 診断統計量の方針と過剰識別検定

- **新しい診断統計量では「`cov_type`に追従するか、固定の計算方式か」と「失敗したら`fit()`ごと失敗させるか、その統計量だけ`None`にするか（`None`にする理由をどこまで広げるか）」を独自に決めず、都度ユーザーに確認する**: 既存の判断は統計量ごとに違う（弱操作変数のFは常に等分散前提、Wu-Hausmanと2SLSの過剰識別検定は`cov_type`に追従、Sargan・GMMのHansen Jの失敗は伝播）。詳細: `docs/spec/iv-spec.md`3.4節〜3.6節。
- **Sargan・GMMのHansen Jの計算失敗は`None`にせず`IvError`として伝播する**（2SLSのロバストな過剰識別検定は別）: 使う行列は点推定で反転済みの行列の再利用で、特異は理論上到達不能。詳細: `docs/spec/iv-spec.md`3.5節。
- **2SLSの過剰識別検定（`two_sls/overid.rs`）は`cov_type`に追従する。`Classical`以外は2SLS残差から重み`S`を作ってGMMを再推定した解のHansen Jにし、2SLS残差のままの`(Z'e)'S⁻¹(Z'e)`は使わない。`None`にするのは丁度識別・`Cluster`で`G<l`・重みの白色化と`Ĝ`の分解の`NearSingularMatrix`の3つだけ**: 2SLS残差での形はχ²に従わず、不均一分散で棄却率が名目からずれる。診断1つのために`fit()`を失敗させない（近共線な外生変数×クラスターで実際に退行した）。回帰ガード: `two_sls/overid.rs`の`equals_the_two_step_gmm_hansen_j_of_the_same_weight`、`tests/cross_method/test_near_collinear_exact.py`。詳細: `docs/spec/iv-spec.md`3.5節。

## 2SLSとGMMの共有範囲

- **`TwoSlsEstimator`と`GmmEstimator`の`fit()`は独立に実装し、GMMを2SLSへ委譲しない。共通の読み出し口は`common/estimate.rs`の`IvEstimate` trait（`engine_pybind`が`IVResult`の共通フィールドを一度だけ詰める口）で、推定量ごとに意味が違うもの（`sargan_*`・`hansen_j_*`・`wu_hausman_*`・`n_iter`等）は入れない。traitを広げるのは3つ目の推定量が出てから**: `Classical`のGMMと2SLSは点推定が一致するが、持つ推論統計量の範囲が違う（理由は`gmm.rs`のモジュールdoc「2SLSとの共通化の判断」）。詳細: `docs/spec/iv-spec.md`3.1節。
- **共分散・検定の中立な計算（classicalの共分散・HCの行スケール`hc_scales`と`hc_meat`・`hac_meat`・`cluster_meat`と補正・HACラグの解決・Waldの二次形式`wald_quadratic_form`（対称化込み）と`wald_f_test`/`wald_chi2_test`）は`shared/`の共有関数を使い、推定量ごとに複製しない。スコアに依存する部分（HCのレバレッジ、小標本補正の分母）だけを推定量が決めて値として渡す（`HcScale`）**: 複製すると`Par::Seq`やWaldの対称化のような数値上の修正が片方に漏れる（IV・TobitだけWaldの対称化が無く、準共線で大きく外れた）。回帰ガード: `tests/cross_method/test_wald_ill_conditioned.py`。詳細: `docs/spec/iv-spec.md`3.1節。
- **レバレッジが1の行（HC2・HC3）は共有の`hc_scales`が`UnitLeverage`を返す。呼び出し側でレバレッジを検査しない**: 係数の標準誤差と第一段階は失敗させ、診断（KP・有効Fは`hc_row_scales`が`None`、Wu-Hausmanは`None`）は`fit()`を止めない。詳細: `docs/spec/iv-spec.md`3.6節。
- **2SLSの点推定・Sargan・部分F・Wu-Hausmanの列は`shared/instrumental_variables.rs`の行列版を使い、`OlsEstimator`で第二段階を回さない（`second_stage`フィールドを持たない）**: 共分散・全体FがQRを共有するので1本で済む。`n_absorbed`で固定効果つき（FEIV）も同じ関数を呼べる（個体ダミー入りとwithin変換が一致するFrisch-Waugh-Lovell）。回帰ガード: `dummies_in_the_regressors_equal_the_demeaned_data_with_absorbed_parameters`。詳細: `docs/spec/iv-spec.md`3.4.1節。

## テスト

- **HACのテストは`lags=None`・`lags=Some(正の値)`（Bartlettのループ本体）・`time_order`の3種を最初からそろえ、`cargo llvm-cov`の欠落行で確かめる**: `lags=Some(0)`とエラーパスだけではループ本体が通らず、%では気づかなかった。回帰ガード: `two_sls/cov/tests/hac.rs`の`manual_hac_std_errors`、`gmm/fit/tests/weight_robust_hac.rs`の`fit_computes_hac_weighted_estimate_matching_manual_formula_with_lags_two`・`fit_with_hac_weight_type_and_auto_lags_matches_explicit_lags_two`・`fit_with_hac_weight_type_respects_time_order`。詳細: `docs/spec/iv-spec.md`4章。
- **新しいgetterを足したら、`fit_succeeds_when_over_identified`（`two_sls`・`gmm`の`fit/tests/basic.rs`）のメタデータのテストに追記する**: 値のテストは主要な値しか読まず、単純なgetterが未カバーになった。
- **linearmodelsとの対応関係（`cov_type`↔`cov_type`/`debiased`、Wu-Hausmanは`res.wooldridge_regression`と照合すること等）は`benchmark/iv/references/linearmodels_ref.py`のモジュールdocstringが正本。IVのpytestを書く前・GMMの`cov_type`対応を変える前にまずそこを読む**: 詳細: `docs/spec/iv-spec.md`3.1節・3.2節・4章。
- **proptestの生成器は`common/proptest_support.rs`に一本化して2SLS・GMMで共有する。列順序の不変性へのバグ注入は`Z`ではなく`X`側の列で行い、GMMで`Z`の列の変換への不変性を安易に期待しない。`n > 50`のときだけ発現する注入でproptestだけが失敗することを確かめ、生成された`proptest-regressions/`はコミットしない**: 2段階GMMの重みは`Z`の変換に追従する。詳細: `docs/spec/iv-spec.md`4章「property-basedテスト」。
