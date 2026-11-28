# engine/src/linear/ 実装ノート（OLS/WLS）

このファイルは `engine/src/linear/` 配下のファイルを読み書きするときだけ自動ロードされる。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景・数式・実測は`docs/spec/ols-spec.md`・`docs/spec/wls-spec.md`・`docs/spec/orthogonal-coordinates.md`が正本（このファイルはその要約ではなく差分の索引）。事後診断（`diagnostics/`）だけに関わる罠はサブディレクトリの`CLAUDE.md`に分けてある（下の「サブディレクトリのノート」）。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## モジュール構成

- `ols.rs`・`wls.rs`・`diagnostics.rs`は外から見える面（構造体・getter・`fit*`の契約と委譲・`wald_test_last_columns`・再エクスポート）だけを持つファサードで、ロジックは`ols/`（`fit`・`auxiliary`・`wald`・`cov_type`・`cov_params`・`input`・`predict`）・`wls/`（`fit`・`statistics`）・`diagnostics/`（`aux_regression`・`white`・`breusch_pagan`・`breusch_godfrey`）にある。`ols/cov_params.rs`は`shared::covariance`の部品をOLS固有の規則でつなぐ層で、`LeastSquaresError`を返すので`shared/`に置けない。`fit`を通すテストと`mod proptests`は`ols/fit/tests/`、`cov_params.rs`のテストは`ols/cov_params/tests/`にテーマ別。
- 共分散・Wald検定・直交座標のQR・適合度・グループ化の部品は`engine/src/shared/`（`covariance`・`wald`・`orthonormal`・`goodness_of_fit`・`group_codes`。`least_squares`はテスト用の独立な参照実装だけ）にあり、`linear`のエラー型に依存しない（`CommonError`・`RankDeficient`を返し、呼び出し側が`map_err`する）。`engine`は列名を知らないので、列名の重複など列名が要る検証は`engine_pybind`の責務（`engine_pybind/src/linear/CLAUDE.md`）。

## サブディレクトリのノート

次のファイルは、そのディレクトリ配下のファイルを読み書きするときに、このファイルと一緒に自動ロードされる。ファサードの`diagnostics.rs`を読むだけではロードされないので、ファサードだけを変えるときや、補助回帰（`ols/auxiliary.rs`）と診断をまたぐ変更では明示的に読む。

- `diagnostics/CLAUDE.md`: 事後診断（White・Breusch-Paganの補助回帰の標準化と項の選び方、`R²`の非有限の判定、Breusch-GodfreyのQRと行列の確保）。

## 計算の経路と数値

- **係数・残差・共分散・全体Fは、ピボットなしQR 1本（`OrthonormalDesign::new_checked`）から求める。`X'Xβ=X'y`をCholeskyで解かない。n×kの列ピボットQRや`(X'X)⁻¹`を別に走らせない**: `X'X`は条件数が2乗になる。2本目のQRを足した版は+50〜60%遅かった。ピボットなしにするのは、検定対象が末尾`q`列という性質を保つため。回帰ガード: `new_checked_rejects_rank_deficient_designs_like_checked_col_piv_qr`・`new_checked_reports_exact_collinearity_as_columns_not_as_the_condition_guard`・`new_checked_boundary_does_not_move_with_the_column_scale`、`tests/support/_exact_wald.py`。詳細: `docs/spec/orthogonal-coordinates.md`1章〜3.1節、`docs/spec/ols-spec.md`3.1節。
- **全体Wald/F・標準誤差は`cov_params`（`B M B`）からではなく、直交座標の`(b̃, M̃)`から作る。`wald_f_test`/`wald_chi2_test`には`(b̃, M̃)`を渡す**: `B = (X'X)⁻¹`の明示的な逆行列は、傾きの相関が約`1 - 1e-11`を超えると桁落ちで`V`が正定値でなくなる。推定量は`Q`を持たず`b̃`と`M̃`だけを保持する。詳細: `docs/spec/orthogonal-coordinates.md`1章。
- **ランク判定と条件数のガードは、列を単位ノルムに揃えた`R`に対して行う（生の`R`の相対閾値で判定しない）。条件数`1e12`超は列名つきの`Columns`、`5e7`超は`GramMatrix`。`5e7`を厳しくしない（破壊的）**: 生の`R`だと列の単位で合否が動く。QRは`cond(X)`が`1e8`〜`1e15`でも解けてしまうが、係数が静かに大きくずれる。完全な共線は条件数が`1e14`〜`1e16`でランク判定の境界に来るので、`1e12`で列名つきの報告に揃える。詳細: `docs/spec/orthogonal-coordinates.md`3.1節。
- **相対閾値で特異性を判定する箇所は`shared::linear_algebra::checked_col_piv_qr`を使い、式を再実装しない。比較は`diag.is_nan() || diag <= threshold`**: 全ゼロの設計行列で`col_piv_qr`がR対角にNaNを生成しうる（faer 0.24.4）。NaNとの比較は常に`false`。以前は3箇所に重複した式でこの罠を別々に踏んだ。回帰ガード: `checked_col_piv_qr_detects_nan_diagonal_from_all_zero_matrix`。詳細: `docs/spec/ols-spec.md`3.1節。
- **対称正定値行列の条件数は`ensure_well_conditioned_symmetric_matrix`（固有値ベース）で判定し、非ピボットCholeskyのL因子対角に頼らない**: L因子対角は列間のスケール差によるほぼ特異を検出できない。詳細: `docs/spec/ols-spec.md`3.3節。
- **全体Wald/Fの悪条件判定は`Σ`ではなく相関スケール化した`C = D^{-1/2}ΣD^{-1/2}`に対して行い（`wald_quadratic_form`）、二次形式も同じ`C`で求める。`ensure_well_conditioned_symmetric_matrix`自体はスケール化しない（Wald用に呼ぶ側で行う）。「極端なスケール差で失敗する」ことを期待するテストは書かない（失敗させたいときは準多重共線性で作る）**: `Σ`のままだと、列の単位を変えるだけで計算できるデータが`MATRIX_NEAR_SINGULAR`になった。nonlinearのHessian・OPGは標準化済みの空間で判定するので関数の意味を変えない。回帰ガード: `wald_f_test_rejects_a_near_singular_covariance_submatrix`。詳細: `docs/spec/ols-spec.md`3.3節。
- **HC・HAC・クラスターの演算順序（`resid * x`を先に作ってから積み上げる、`bread * meat * bread`の結合順）を変えない**: 最終桁がずれ、`1e-8`の照合は通っても過去の出力とのビット一致が崩れる。HACとクラスターはスコアをクロージャで受け、n×kの行列を追加で確保しない。詳細: `docs/spec/ols-spec.md`3.2節。
- **HACの`k×k`の行列積は`Par::Seq`を明示する。小さい行列の頻繁な積を他手法で書くときも、並列化の要否は実測してから決める**: faerの並列ディスパッチのオーバーヘッドが計算本体を上回る（n=10,000・k=2で6倍悪化）。詳細: `docs/spec/ols-spec.md`3.2節「HAC」。
- **全推定手法の`fit()`冒頭（と`engine_pybind`の`#[pymodule]`初期化）で`ensure_serial()`を呼び、グローバル並列度を引き上げるコードを入れない。並列化は実測で効く箇所だけ`Par::Rayon(_)`を対象APIへ明示的に渡す。`disable_global_parallelism()`は使わない**: tall-skinnyな設計行列の暗黙の全コア並列は、多コア機・負荷下で20〜200倍遅く不安定。`disable_global_parallelism()`は`get_global_parallelism()`をpanicさせる。回帰ガード: `ols/fit/tests/basic.rs`の`fit_pins_faer_global_parallelism_to_seq`。詳細: `.claude/rules/rust-style.md`「パフォーマンス」、`docs/performance/ols.md`。

## クラスター

- **グループ化はキーの辞書順の`GroupCodes`で行い、`HashMap`は使わない**: ハッシュシードで反復順序が変わり、`fit()`を繰り返すと標準誤差が1 ULPぶれる非決定性バグを起こした（WLSで発覚）。回帰ガード: `fit_cluster_std_errors_are_deterministic_across_repeated_fits`。詳細: `docs/spec/ols-spec.md`3.2節「クラスター」。
- **`CovType::Cluster { groups }`（nonlinear・IV・panelも同じ）はコード化済みの`Option<GroupCodes>`で受け取り、`String`列を受け取って`fit()`内でコード化し直さない。コード化は`engine_pybind`の`extract_cluster_groups`で一度だけ行い、順序（辞書順）と同じ文字列になる値のまとめ方は`engine`の`GroupCodes`だけが決める。浮動小数の`-0.0`と`0.0`を別グループのまま保ち、polarsの`unique`/`group_by`に置き換えない**: 走査し直しはn=100万で1回0.2〜0.4秒かかった。polarsは`-0.0`と`0.0`を同一視する。回帰ガード: `extract_cluster_groups_agrees_with_the_string_label_path_for_every_key_dtype`。詳細: `docs/spec/ols-spec.md`3.2節「クラスター」。
- **クラスター用のコードはキーを持たない（`from_labels_without_keys`/`from_factorized_without_keys`）ので、検証・集計で`keys()`を呼ばない。コードの行数が`n`と違えば`ClusterDimensionMismatch`**: キー無しで`keys()`を呼ぶとpanicする。長さの検査が無いと、長いコードはpanic、短いコードは先頭の行だけの標準誤差を静かに返す。`GroupCodes`の`==`はキーを見ない。詳細: 同上。
- **クラスターの蓄積（`Σ_g S_g S_g'`）と小標本補正は`shared::covariance::cluster_meat`・`cluster_correction`を使い、新しいクラスター系の計算でこのループを書き直さない**: 加算順と最適化をここ1か所で保つ（行を観測順に1回走査する形で、旧実装とビット単位で一致する）。詳細: 同上。
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
