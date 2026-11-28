---
paths:
  - engine/src/shared/orthonormal.rs
  - engine/src/shared/linear_algebra.rs
  - engine/src/shared/wald.rs
  - engine/src/shared/covariance.rs
  - engine/src/shared/group_codes.rs
  - engine/src/linear/**/*
---

# 直交座標のQR・条件数のガード・全体Wald・共分散・クラスターのコード（`engine/src/shared/`の最小二乗の核）

このファイルは上の`paths`に一致するファイルを読み書きするときだけ自動ロードされる。`engine/src/shared/`の`orthonormal.rs`・`linear_algebra.rs`・`wald.rs`・`covariance.rs`・`group_codes.rs`の中身の罠で、OLS/WLS（`engine/src/linear/`）を読み書きするときにもロードする。他の系統（iv・panel・nonlinear）もこれらを使うが、ロードする範囲は広げていない。部品を使う側の規則（`fit()`に入れるもの・クラスター数の検査等）は各系統の`CLAUDE.md`。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景・実測は`docs/spec/orthogonal-coordinates.md`・`docs/spec/ols-spec.md`3.1節〜3.3節が正本。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## 直交座標のQRと条件数のガード（`orthonormal`）

- **係数・残差・共分散・全体Fは、ピボットなしQR 1本（`OrthonormalDesign::new_checked`）から求める。`X'Xβ=X'y`をCholeskyで解かない。n×kの列ピボットQRや`(X'X)⁻¹`を別に走らせない**: `X'X`は条件数が2乗になる。2本目のQRを足した版は+50〜60%遅かった。ピボットなしにするのは、検定対象が末尾`q`列という性質を保つため。回帰ガード: `new_checked_rejects_rank_deficient_designs_like_checked_col_piv_qr`・`new_checked_reports_exact_collinearity_as_columns_not_as_the_condition_guard`・`new_checked_boundary_does_not_move_with_the_column_scale`、`tests/support/_exact_wald.py`。詳細: `docs/spec/orthogonal-coordinates.md`1章〜3.1節、`docs/spec/ols-spec.md`3.1節。
- **全体Wald/F・標準誤差は`cov_params`（`B M B`）からではなく、直交座標の`(b̃, M̃)`から作る。`wald_f_test`/`wald_chi2_test`には`(b̃, M̃)`を渡す**: `B = (X'X)⁻¹`の明示的な逆行列は、傾きの相関が約`1 - 1e-11`を超えると桁落ちで`V`が正定値でなくなる。推定量は`Q`を持たず`b̃`と`M̃`だけを保持する。詳細: `docs/spec/orthogonal-coordinates.md`1章。
- **ランク判定と条件数のガードは、列を単位ノルムに揃えた`R`に対して行う（生の`R`の相対閾値で判定しない）。条件数`1e12`超は列名つきの`Columns`、`5e7`超は`GramMatrix`。`5e7`を厳しくしない（破壊的）**: 生の`R`だと列の単位で合否が動く。QRは`cond(X)`が`1e8`〜`1e15`でも解けてしまうが、係数が静かに大きくずれる。完全な共線は条件数が`1e14`〜`1e16`でランク判定の境界に来るので、`1e12`で列名つきの報告に揃える。詳細: `docs/spec/orthogonal-coordinates.md`3.1節。

## 特異性・悪条件の判定と全体Wald（`linear_algebra`・`wald`）

- **相対閾値で特異性を判定する箇所は`shared::linear_algebra::checked_col_piv_qr`を使い、式を再実装しない。比較は`diag.is_nan() || diag <= threshold`**: 全ゼロの設計行列で`col_piv_qr`がR対角にNaNを生成しうる（faer 0.24.4）。NaNとの比較は常に`false`。以前は3箇所に重複した式でこの罠を別々に踏んだ。回帰ガード: `checked_col_piv_qr_detects_nan_diagonal_from_all_zero_matrix`。詳細: `docs/spec/ols-spec.md`3.1節。
- **対称正定値行列の条件数は`ensure_well_conditioned_symmetric_matrix`（固有値ベース）で判定し、非ピボットCholeskyのL因子対角に頼らない**: L因子対角は列間のスケール差によるほぼ特異を検出できない。詳細: `docs/spec/ols-spec.md`3.3節。
- **全体Wald/Fの悪条件判定は`Σ`ではなく相関スケール化した`C = D^{-1/2}ΣD^{-1/2}`に対して行い（`wald_quadratic_form`）、二次形式も同じ`C`で求める。`ensure_well_conditioned_symmetric_matrix`自体はスケール化しない（Wald用に呼ぶ側で行う）。「極端なスケール差で失敗する」ことを期待するテストは書かない（失敗させたいときは準多重共線性で作る）**: `Σ`のままだと、列の単位を変えるだけで計算できるデータが`MATRIX_NEAR_SINGULAR`になった。nonlinearのHessian・OPGは標準化済みの空間で判定するので関数の意味を変えない。回帰ガード: `wald_f_test_rejects_a_near_singular_covariance_submatrix`。詳細: `docs/spec/ols-spec.md`3.3節。

## 共分散の計算（`covariance`）

- **HC・HAC・クラスターの演算順序（`resid * x`を先に作ってから積み上げる、`bread * meat * bread`の結合順）を変えない**: 最終桁がずれ、`1e-8`の照合は通っても過去の出力とのビット一致が崩れる。HACとクラスターはスコアをクロージャで受け、n×kの行列を追加で確保しない。詳細: `docs/spec/ols-spec.md`3.2節。
- **HACの`k×k`の行列積は`Par::Seq`を明示する。小さい行列の頻繁な積を他手法で書くときも、並列化の要否は実測してから決める**: faerの並列ディスパッチのオーバーヘッドが計算本体を上回る（n=10,000・k=2で6倍悪化）。詳細: `docs/spec/ols-spec.md`3.2節「HAC」。
- **クラスターの蓄積（`Σ_g S_g S_g'`）と小標本補正は`shared::covariance::cluster_meat`・`cluster_correction`を使い、新しいクラスター系の計算でこのループを書き直さない**: 加算順と最適化をここ1か所で保つ（行を観測順に1回走査する形で、旧実装とビット単位で一致する）。詳細: 同上。

## クラスターのコード（`group_codes`）

- **グループ化はキーの辞書順の`GroupCodes`で行い、`HashMap`は使わない**: ハッシュシードで反復順序が変わり、`fit()`を繰り返すと標準誤差が1 ULPぶれる非決定性バグを起こした（WLSで発覚）。回帰ガード: `fit_cluster_std_errors_are_deterministic_across_repeated_fits`。詳細: `docs/spec/ols-spec.md`3.2節「クラスター」。
- **`CovType::Cluster { groups }`（nonlinear・IV・panelも同じ）はコード化済みの`Option<GroupCodes>`で受け取り、`String`列を受け取って`fit()`内でコード化し直さない。コード化は`engine_pybind`の`extract_cluster_groups`で一度だけ行い、順序（辞書順）と同じ文字列になる値のまとめ方は`engine`の`GroupCodes`だけが決める。浮動小数の`-0.0`と`0.0`を別グループのまま保ち、polarsの`unique`/`group_by`に置き換えない**: 走査し直しはn=100万で1回0.2〜0.4秒かかった。polarsは`-0.0`と`0.0`を同一視する。回帰ガード: `extract_cluster_groups_agrees_with_the_string_label_path_for_every_key_dtype`。詳細: `docs/spec/ols-spec.md`3.2節「クラスター」。
- **クラスター用のコードはキーを持たない（`from_labels_without_keys`/`from_factorized_without_keys`）ので、検証・集計で`keys()`を呼ばない。コードの行数が`n`と違えば`ClusterDimensionMismatch`**: キー無しで`keys()`を呼ぶとpanicする。長さの検査が無いと、長いコードはpanic、短いコードは先頭の行だけの標準誤差を静かに返す。`GroupCodes`の`==`はキーを見ない。詳細: 同上。
