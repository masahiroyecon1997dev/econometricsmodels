# engine/src/iv/common/ 実装ノート（第一段階・Wu-Hausman）

このファイルは `engine/src/iv/common/` 配下のファイルを読み書きするときだけ自動ロードされる（親の`engine/src/iv/CLAUDE.md`も一緒にロードされる）。ファサードの`common.rs`を読むだけではロードされない。弱操作変数の診断の罠は、`engine/src/shared/`の核と共有するので`.claude/rules/engine-weak-instruments.md`にある（`weak_instruments.rs`・`first_stage.rs`を読むと一緒にロードされる）。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景は`docs/spec/iv-spec.md`3.6節・3.7節が正本。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## 第一段階

- **第一段階の`OlsInput`には、焼き込み済みの定数項列を除いた素の`x_exog`と`include_intercept=input.has_intercept()`を渡す（`without_baked_in_intercept`）。`include_intercept=false`で二重切片を避けない**: `has_intercept`は`k_constant`も兼ねるので、`false`だと第一段階の`r_squared`が静かに誤り（参照実装との比較では検出できない）、`G=2`の最小ケースが`ComputationError`になる。Wu-Hausmanの拡張回帰も同じ。回帰ガード: `two_sls/cov/tests/validation.rs`の`fit_succeeds_with_cluster_g2_boundary_when_x_exog_is_empty`、`tests/iv/test_iv_api.py`の`test_cluster_g2_boundary_succeeds_when_x_exog_is_empty`、`tests/iv/test_iv_first_stage.py`。詳細: `docs/spec/iv-spec.md`3.7節。

## Wu-Hausman

- **Wu-Hausman（`common/wu_hausman.rs`、2SLS・GMM共通）は`fit()`の`cov_type`に追従し、想定内の失敗（拡張回帰の`PerfectFit`・`SingularMatrix`・`InsufficientObservations`・`UnitLeverage`）だけを`None`にする。それ以外（Wald側の`NearSingularMatrix`を含む）は`HausmanRegressionFailed`として伝播する**: 広すぎる`Err(_)`のキャッチは将来の実装バグを`None`で隠す。拾う失敗は`is_expected_failure`に明示する。結果は`?`ではなく`match`で振り分ける。詳細: `docs/spec/iv-spec.md`3.6節。
- **Wu-Hausmanのp値の分母自由度は拡張回帰自身の`df_resid`（主構造式の`df_resid - n_endog`）。統計量とp値は別々に照合する**: 主構造式の自由度を流用しても統計量は機械精度で一致し、p値だけがずれるので見逃しやすい。詳細: `docs/spec/iv-spec.md`3.6節。
