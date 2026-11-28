---
paths:
  - engine/src/shared/gmm*.rs
  - engine/src/iv/gmm.rs
  - engine/src/iv/gmm/**/*
  - engine/src/panel/fe_iv/gmm.rs
  - engine/src/panel/fe_iv/gmm/**/*
---

# GMMの核（IVのGMM・FE-GMM共通）

このファイルは上の`paths`に一致するファイルを読み書きするときだけ自動ロードされる。IVのGMM（`engine/src/iv/gmm`）とFE-GMM（`engine/src/panel/fe_iv/gmm`）が、`engine/src/shared/`の点推定の反復（`gmm_iteration.rs`）・重み行列（`gmm_weight.rs`）・白色化した座標の解（`gmm.rs`）・C統計量の核（`gmm_c_statistic.rs`）を共有するので、系統のネストCLAUDE.mdではなくここに置く。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景は`docs/spec/iv-spec.md`3.1節・3.3節・3.9節（白色化は`docs/spec/orthogonal-coordinates.md`4.2節、FE-GMMとの違いは`docs/spec/fe-iv-spec.md`3.5節）が正本。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## 点推定・重み・共分散

- **方式は`GmmType`のenum（`OneStep`/`TwoStep{weight}`/`Iterated{weight,max_iter,tol}`）で選び、回数の整数に戻さない。3方式を`gmm_iterate`の1つのループで扱い、方式ごとにアルゴリズムを分岐させない**: 整数だと「2が2段階GMM」を知らないと読めず、矛盾した組み合わせ（1段階に反復設定・`weight_type`）を型で防げない。`OneStep`は`weight_type`を持たず検証もしない（`TwoStep`の`validate_weight_type`と本体の`match`の二重チェックは意図的）。詳細: `docs/spec/iv-spec.md`3.3節。
- **収束判定は係数ごとの`max(rtol·|前回値|, atol)`で全係数が満たして収束（ベクトルノルムの比にしない）。`atol`（`GMM_CONVERGENCE_ATOL=1e-8`）は公開しない。`Iterated`の`max_iter`は初回推定を含めて数え3以上（`InvalidGmmMaxIter`）。`OneStep`/`TwoStep`は収束判定を持たず常に`converged=true`、`n_iter`は1・2**: 1つでも収束していない係数を見逃さない。未収束は`raise_on_non_convergence`でMLEと同じ意味論。回帰ガード: `gmm/fit/tests/iteration.rs`の`fit_treats_two_step_as_always_converged`。詳細: `docs/spec/iv-spec.md`3.3節。
- **重み行列`S`（`shared/gmm_weight.rs`）にはHC1・クラスターの小標本補正を入れない**: `β̂`は`S`の正のスカラー倍で変わらない。HC2/HC3のレバレッジ補正は成分ごとに違うので、`gmm_weight_type`にHC2/HC3相当は無い。回帰ガード: `gmm/fit/tests/iteration.rs`の`gmm_point_estimate_is_invariant_to_positive_scaling_of_s`。詳細: `docs/spec/iv-spec.md`3.1節。
- **GMMの係数・共分散・全体Wald・Hansen Jは白色化した操作変数の直交座標（`shared/gmm.rs`の`WhitenedInstruments`・`GmmSolution`）で求め、`X'ZS⁻¹Z'X`も`B⁻¹`も作らない。ロバスト・クラスターの`S`は条件数が`WEIGHT_CHOLESKY_MAX_CONDITION`を超えたら`S = A'A`の`A`のQRにし、`S`を明示的に作らない。`Ω̂`は`Z`ではなく`Z̃`から組んで`Ω̃`を直接得る**: どれも条件数が2乗になり、近共線な`Z`で精度を失った。ガードは`Ĝ`が`GMM_BREAD`、`Z`と重みが`GMM_WEIGHT`。回帰ガード: `tests/cross_method/test_near_collinear_exact.py`。詳細: `docs/spec/orthogonal-coordinates.md`4.2節。
- **SEのサンドイッチは常に一般形`B⁻¹(X'ZWΩ̂WZ'X)B⁻¹`（`W`は点推定に使った重み）で、効率的GMMの特殊ケースでも`B⁻¹`だけに簡略化しない**: `gmm_weight_type`と`cov_type`は独立な選択。`OneStep`の`S_used`のスケールはサンドイッチ全体のスケール不変性で相殺する。回帰ガード: `gmm/sandwich/tests/reference.rs`の`fit_computes_hc1_std_errors_matching_manual_formula_with_one_step_gmm`。詳細: `docs/spec/iv-spec.md`3.1節。
- **GMMの`cov_type=Classical`は残差を中心化して`σ̂²`を作る（HC0-3・HAC・クラスターの`Ω̂`は中心化しない）。`gmm_weight_type≠Classical`のケースを独立に（実装の式を複製しないオラクルで）検証する**: GMMの一次条件は一般に`ē=0`を保証しない。linearmodelsと同じ。詳細: `docs/spec/iv-spec.md`3.1節。
- **HC1・クラスターの小標本補正の分母は構造パラメータの数`k`で、操作変数の数`l`にしない。2SLSの式を「`X̂`→`Z`、`k`→`l`」と機械的に置き換えない**: レバレッジ`z_i'(Z'Z)⁻¹z_i`は`Z`への置換が妥当だが、補正が表す消費した自由度は`k`次元の`β̂`のもの。新しい手法でも、各除算・各逆行列が実際にどの次元に依存するかを個別に確かめる（`gmm_hc_omega`/`gmm_cluster_omega`のdocコメント）。詳細: `docs/spec/iv-spec.md`3.1節。

## C統計量

- **C統計量（核は`shared/gmm_c_statistic.rs`）の`J_c`の重みは`S_e`の部分行列の逆行列`S_c⁻¹`にし、`S_e⁻¹`の部分行列にしない。`J_c`の点推定も`Z_c`で推定し直す**: 共通の`S`から作るので`C = J_e − J_c ≥ 0`になる（`S_e⁻¹`の部分行列だと負になりうる）。常に2段階で重みの種類だけを引き継ぎ、クラスターは`G < l_full`で未定義、`validate_weight_type`は呼ばない（文面が`gmm_weight_type`のもの）。詳細: `docs/spec/iv-spec.md`3.9節。
