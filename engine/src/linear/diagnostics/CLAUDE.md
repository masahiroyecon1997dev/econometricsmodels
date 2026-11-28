# engine/src/linear/diagnostics/ 実装ノート（事後診断）

このファイルは `engine/src/linear/diagnostics/` 配下のファイルを読み書きするときだけ自動ロードされる（親の`engine/src/linear/CLAUDE.md`も一緒にロードされる）。ファサードの`diagnostics.rs`を読むだけではロードされない。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景は`docs/spec/ols-spec.md`3.9〜3.11節、公開形は`docs/spec/inference-conventions.md`6章が正本。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## 事後診断（White・Breusch-Pagan・Breusch-Godfrey）

- **White・Breusch-Paganの補助回帰は列を標準化してから`OlsEstimator::fit`に渡し、`fit`のエラーは検定の文脈（`White test:`）に言い換える（`aux_fit_error`）**: 平均が標準偏差より桁違いに大きい変数の二乗項が元の列とほぼ共線になり、特異性判定・条件数チェックで誤って失敗した。詳細: `docs/spec/ols-spec.md`3.9節。
- **`R²`を`max(0.0)`で丸める前に`is_finite()`を判定する**: `f64::max`/`min`はNaNを無視し（`NAN.max(0.0) == 0.0`）、非有限の`R²`が統計量0.0として静かに返る。回帰ガード: `white_rejects_non_finite_residuals_instead_of_masking_nan`。詳細: 同上。
- **補助回帰の項から、定数の説明変数（`include_intercept=false`に入れた定数列）を含む項は値の比較を待たず除く（Whiteのみ。BPの`z`には要らない）**: `c*x_k`は`c != 1`だと`x_k`と数値的に同一にならず、重複判定をすり抜けてランク落ちになる。回帰ガード: `white_drops_terms_that_involve_a_constant_other_than_one`。詳細: 同上。
- **Breusch-Paganは補助回帰以降をWhiteと共有し（`aux_regression.rs`の`select_aux_terms`→`squared_residual_aux_test`）、違いは候補項だけにする。Koenker版（`n·R²`）のみ**: 詳細: `docs/spec/ols-spec.md`3.11節。
- **Breusch-Godfreyは`OlsEstimator::fit`を使わず、列ノルムでスケールした列ピボットQRで`SSR_u`だけを求める。列を中心化しない。Fの分子`Σû² - SSR_u`はOLS残差のときだけ成り立つ。補助回帰の行列は`try_reserve`してから埋める**: 切片なしのモデルは中心化すると列空間が変わる。WLS等の非直交な残差では過大になる。確保失敗は`BREUSCH_GODFREY_MATRIX_TOO_LARGE`。詳細: `docs/spec/ols-spec.md`3.10節。
