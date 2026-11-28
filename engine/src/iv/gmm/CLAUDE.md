# engine/src/iv/gmm/ 実装ノート（GMM。Hansen J・Wu-Hausman）

このファイルは `engine/src/iv/gmm/` 配下のファイルを読み書きするときだけ自動ロードされる（親の`engine/src/iv/CLAUDE.md`も一緒にロードされる）。ファサードの`gmm.rs`を読むだけではロードされない。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景は`docs/spec/iv-spec.md`3.5節・3.6節が正本。点推定・重み・共分散・C統計量の罠は、`engine/src/shared/`の核と共有するので`.claude/rules/engine-gmm-core.md`にある（このディレクトリのファイルを読むと一緒にロードされる）。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## Hansen J・Wu-Hausman

- **GMMのHansen J（`gmm/hansen_j.rs`）は`n`で割らない**: `S`は`n`で正規化していない生の和なので、標準形の`n`は相殺して`J=(Z'ê)'S⁻¹(Z'ê)`。`/n`にすると常に`1/n`倍に過小評価され、オラクルに同じ`/n`を複製する自己参照的な検証では検出できない。**`S`の正規化の有無を前提にする統計量では、自己参照的でない検証（Sarganとの一致等）を必ず用意する**。詳細: `docs/spec/iv-spec.md`3.5節。
- **`OneStep`と`gmm_weight_type=Classical`のHansen Jの`S`は`σ̂²₀・Z'Z`（`σ̂²₀ = ê₀'ê₀/n`）**: 点推定はスカラー倍で不変だが、Jは絶対スケールに依存する。これで`Classical`×`TwoStep`のJが2SLSのSarganに一致する。回帰ガード: `gmm/hansen_j.rs`の`fit_computes_hansen_j_statistic_matching_two_sls_sargan_when_weight_type_is_classical`、`gmm/fit/tests/basic.rs`の`fit_matches_two_sls_point_estimate_when_weight_type_is_classical`。詳細: `docs/spec/iv-spec.md`3.5節。
- **GMMのWu-Hausmanは、点推定・共分散・Wald・Hansen Jの後に回し、第一段階の残差は借用で受け取る（`fit_with_first_stage_residuals`。`engine_pybind`が先に回した残差を渡し、第一段階を二重に回さない）。補助回帰の前に`x`・`z`・`GmmSolution`を`drop`する**: 構造式のエラーを先に返し、第一段階の残差と同時に持つ時間を短くする。GMMの重みにも点推定にも依らないので、同じデータ・`cov_type`なら2SLSと同じ値（`gmm/fit/tests/wu_hausman.rs`）。詳細: `docs/spec/iv-spec.md`3.6節。
