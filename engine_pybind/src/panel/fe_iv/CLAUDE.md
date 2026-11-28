# engine_pybind/src/panel/fe_iv/ 実装ノート（FEIV）

このファイルは `engine_pybind/src/panel/fe_iv/` 配下のファイルを読み書きするときだけ自動ロードされる（親の`engine_pybind/src/panel/CLAUDE.md`も一緒にロードされる）。`panel/fe_iv.rs`（モジュールの宣言）を読むだけではロードされない。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景は`docs/spec/fe-iv-spec.md`4章（FE・REと共通の部分は`docs/spec/panel-common.md`4.7節）が正本。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## FEIV（2SLS・GMM）

- **2SLSとGMMは1つのクラス（`FEIVOptions.estimator`）にし、両推定量で結果の項目・メソッドを同じにする（推定量ごとに`None`やエラーで差を作らない）**: IVと同じインターフェースにする。詳細: `docs/spec/fe-iv-spec.md`2.1節・2.4節。
- **`gmm_weight_type`の既定は`cluster`（IVの`classical`と逆。`cov_type`の既定に揃う）。重みの語彙は`classical`・`robust`・`cluster`の3つで、`hac`は`parse_choice`の`OPTION_UNKNOWN_VALUE`で弾き専用の`code`を足さない。`GmmKind`・`WeightKind`は`iv/common`のものを使い回さない**: `FeWeightType`が`Hac`を型で持たないのと揃える。`tol`・`max_iter`は交互射影用で`gmm_tol`・`gmm_max_iter`とは別。詳細: `docs/spec/fe-iv-spec.md`2.3節。
- **使われないオプションの明示指定は`check_fe_iv_options`で`OPTION_UNUSED`にする。`estimator="gmm"`の`cluster`は「`cov_type="cluster"`」か「重みを使う`gmm_type`で`gmm_weight_type="cluster"`」のどちらかが使えば有効**: 判定表は`option_values.rs`の`mod tests`。詳細: `docs/spec/fe-iv-spec.md`2.3節。
- **診断・`fixed_effects()`・`c_stat()`は呼び出し時に`source_data`と`FEIVOptions`から`build_fe_iv_input`で推定量を作り直して計算する（`with_estimator`、`Estimated`の両アームを書く）**: 結果は設計行列を持たない（1回呼ぶごとに推定が1回走る）。`c_stat`の重みはGMMなら`fit()`の重み（`one_step`は`Classical`）、2SLSなら`cov_type`の系統で、2SLSの`Dk`は再推定の前に`C_STATISTIC_COV_TYPE_NOT_SUPPORTED`で弾く。詳細: `docs/spec/fe-iv-spec.md`2.5節・3.5節・4章。
- **`cluster`・`dk`の標準誤差の数値はengineの単体テストに任せ、バインディングのテストは配線（`cluster`の共用・既定）だけを見る。新しい`#[pyclass]`は`tests/cross_method/test_pickle.py`の`SAMPLES`に足す**: 詳細: `docs/spec/fe-iv-spec.md`6.1節。
