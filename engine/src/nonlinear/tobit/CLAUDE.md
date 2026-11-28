# engine/src/nonlinear/tobit/ 実装ノート（Tobit）

このファイルは `engine/src/nonlinear/tobit/` 配下のファイルを読み書きするときだけ自動ロードされる（親の`engine/src/nonlinear/CLAUDE.md`も一緒にロードされる）。ファサードの`tobit.rs`を読むだけではロードされない。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景・数式・実測は`docs/spec/tobit-spec.md`・`docs/spec/orthogonal-coordinates.md`が正本。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## スケーリングと座標

- **Tobitの`y_scale`は2の冪を保つ（`TobitScaling`は`x`側を`ColumnScale`に委ね、`y`のスケーリングと`logσ`だけを持つ）**: `c`を掛ける順序が値を変えないことに依存して、出力を丸めのレベルで安定させている。2の冪以外では`heavy_censoring`等の際どいデータで結果が変わりうる。回帰ガード: `tobit/scaling.rs`の`tobit_scaling_y_scale_is_an_exact_power_of_two`・`tobit_scaling_destandardize_matches_the_closed_form_bit_for_bit`、`common/standardize.rs`の`column_scale_fit_then_standardize_applies_the_fitted_mean_and_std_to_other_rows`。詳細: `docs/spec/tobit-spec.md`3.2節。
- **Tobitの共分散・全体Waldを`X`の座標の`H`で組まない。標準化した設計行列の`Q`の座標`b=Rβ̃`（`tobit/coordinates.rs`の`DesignCoordinates`）で最適化・共分散・Waldを作り、標準化空間へは三角行列の後退代入で戻す（`R⁻¹`を作らない）**: `-H`の条件数は設計行列の2乗で、傾きの相関が約`1-1e-11`以上だと丸めだけで統計量が崩れる。コレスキー因子の座標でも`H`の丸めは残り、受け入れ範囲の判定にも使えない。`Q`は約`√n`の2の冪倍する（そのままだと`tol`が`√n`倍ゆるむ）。受け入れ境界は他の手法と同じ。回帰ガード: `tobit/coordinates.rs`の`overall_wald_in_q_coordinates_matches_exact_rational_values`・`q_is_scaled_by_a_power_of_two_near_the_square_root_of_n`・`a_design_beyond_the_condition_limit_is_near_singular_design_gram`、`tests/cross_method/test_near_collinear_tobit.py`。詳細: `docs/spec/orthogonal-coordinates.md`3.2節・4.3節、`docs/spec/tobit-spec.md`3.3節。
- **反復の経路に依存するテスト（`max_iter=1`の打ち切り点で`SingularHessian`等）は座標で成否が変わるので、`σ→0`に退化するノイズなしの分離データ`noise_free_separation_input`を使う**: LM型の正則化`λI`と`bfgs`/`lbfgs`の初期逆Hessianは座標に依存する。

## 裾の数値（`expected_uncensored`）

- **`expected_uncensored`で`W=Φ(zb)-Φ(za)`で割る量は、`W`を作らずに`Pa=φ(za)/W`・`Pb=φ(zb)/W`を直接計算する（`tobit/mills.rs`の`truncation_pdf_ratios`）**: `μ`が境界から遠い観測（`za=72`）で`W→0`になり、素の`φ/W`は`0/0`。詳細: `docs/spec/tobit-spec.md`3.5節「数値安定性」。
- **`w`の偏微分を`1+ρ2-ρ1²`の教科書形のまま書かない。`A=Pa-za`・`B=zb+Pb`を引き算せずに作り、`w=1-Pa·A-Pb·B+2·Pa·Pb`等で評価する**: `z²`と`z·Pa`の引き算で絶対誤差が`ε·z³`になり、`z=72`で限界効果の標準誤差が`1e-8`ずれた。回帰ガード: `uncensored_terms_and_w_match_high_precision_reference_in_extreme_tails`（80桁の参照値）。詳細: 同上。
- **`W`の逆数にかかる経路では`normal_cdf_minus_half`・`normal_survival`を使う（`normal_cdf_minus_half`は`x>=0`が前提）。`W`で割らない既存のtarget（`ExpectedObserved`等）は`clamped_pdf_cdf`の`Φ`のままでよい**: `Φ`の誤差が標準誤差へ約100倍に増幅される。詳細: 同上。
- **非打ち切り区間が`σ̂`に比べて極端に狭い（`δ=(upper-lower)/σ̂<1e-2`）ときは`UncensoredIntervalTooNarrow`（`ComputationError`）にする（`ensure_uncensored_interval_resolvable`を`predict`・`predict_new_data`・`marginal_effects`の入口で1回）。級数への内部切り替えはしない。`ValidationError`にもしない**: `W`が桁落ちし`w`の相対誤差が約`12ε/δ⁴`になる。黙って近似に変わるのを避ける。入力が不正なのではなく`σ̂`で決まる条件。閾値の根拠は`MIN_UNCENSORED_WIDTH_RATIO`のdocコメント。詳細: `docs/spec/tobit-spec.md`3.5節。

## property-basedテスト（`fit/tests/proptests.rs`・`target.rs`・`problem.rs`の`mod proptests`）

- **左右の打ち切りは別のコードパスなので、境界は`censoring_strategy`で左・右・両側をランダムに選び、鏡像対称性（`(l,u)→(-u,-l)`で係数が符号反転、`σ`とSEは不変）を検証する**: 右打ち切りだけへのバグ注入は固定値のテストでは出ず、鏡像対称性と`score_is_near_zero`だけが失敗した。`expected_uncensored`も鏡像対称性で`(Pa,Pb)`・`(A,B)`の入れ替えを検出する。詳細: `docs/spec/tobit-spec.md`3.8節「property-basedテスト」。
- **クランプ領域（`|zeta|>U_CLAMP`）のHessianの重みは、`fit()`を通す生成器では検出できない。`fit()`を通さずクランプ領域の誤分類観測を毎ケース作ってHessianを直接評価し、符号に加えて閉形式と`hessian()`の組み立ても照合する**: 収束点の`|z|`は`U_CLAMP`に届かない。符号だけだと`|w|`で隠す変更や`h_ss`の誤りを見逃す。新しい重みを足したときも同じ形で検証する。詳細: 同上。
- **バグ注入で生成される`engine/proptest-regressions/`のシードはコミットしない**。
