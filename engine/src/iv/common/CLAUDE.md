# engine/src/iv/common/ 実装ノート（第一段階・弱操作変数の診断・Wu-Hausman）

このファイルは `engine/src/iv/common/` 配下のファイルを読み書きするときだけ自動ロードされる（親の`engine/src/iv/CLAUDE.md`も一緒にロードされる）。ファサードの`common.rs`と、診断の核を置いた`engine/src/shared/`の`weak_instruments.rs`・`kleibergen_paap.rs`・`effective_f.rs`・`score_covariance.rs`を読むだけではロードされない。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景は`docs/spec/iv-spec.md`3.4節・3.6節・3.7節が正本。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## 第一段階

- **第一段階の`OlsInput`には、焼き込み済みの定数項列を除いた素の`x_exog`と`include_intercept=input.has_intercept()`を渡す（`without_baked_in_intercept`）。`include_intercept=false`で二重切片を避けない**: `has_intercept`は`k_constant`も兼ねるので、`false`だと第一段階の`r_squared`が静かに誤り（参照実装との比較では検出できない）、`G=2`の最小ケースが`ComputationError`になる。Wu-Hausmanの拡張回帰も同じ。回帰ガード: `two_sls/cov/tests/validation.rs`の`fit_succeeds_with_cluster_g2_boundary_when_x_exog_is_empty`、`tests/iv/test_iv_api.py`の`test_cluster_g2_boundary_succeeds_when_x_exog_is_empty`、`tests/iv/test_iv_first_stage.py`。詳細: `docs/spec/iv-spec.md`3.7節。

## 弱操作変数の診断（部分F・Cragg-Donald・Kleibergen-Paap・有効F）

- **弱操作変数の部分F（`partial_f_statistic`）は`cov_type`によらず常に等分散前提のネストF検定**: Stock-Yogoの臨界値が等分散前提で較正されている。`x_exog=[]`かつ切片なしでは制限モデルが0列になり、SSRは`y_endog`自体の二乗和（`OrthonormalDesign::new_checked`は0列を受理する）。退化した制限モデルが0変数になるケースを見落とさない。制限モデルは制限なしモデルの列の部分集合なので、その`new_checked`の失敗は理論上到達不能として`?`で伝えるだけ。回帰ガード: `two_sls/fit/tests/basic.rs`の`fit_computes_weak_instrument_f_statistic_when_x_exog_is_empty_and_no_intercept`。詳細: `docs/spec/iv-spec.md`3.4節。
- **Cragg-Donaldの外生変数（定数列込み）は平均を引かず偏回帰で落とす。ランクの判定は列をノルム1にそろえてから行い、`Z`・`Y`が`X`の張る空間にほぼ入る場合は直交化後のノルムを元のノルムと比べて弾く**: 平均を引くと重みつきで切片が合わない。`checked_col_piv_qr`の閾値は自分自身の最大対角に対する相対値なので、`q=1`で`z∈span(X)`の丸め誤差の1列を見逃す。`fit()`はCD・KP・有効Fを呼ばない（バインディングが`compute_*`を呼ぶ）。詳細: `docs/spec/iv-spec.md`3.4.1節。
- **Kleibergen-Paapの`cov_type`対応は、linearmodelsの`robust`のFが`Hc1`、LMが`Hc0`。ランク落ちの共分散に一般化逆行列を使わず`None`にし、LMとWald Fの定義可否は独立に返す。固定効果を吸収したHC2・HC3は`ScoreCovariance::HcAbsorbed`（吸収した効果のレバレッジを足す）を使い、`n_absorbed > 0`の`Hc(..)`のHC2・HC3は`None`**: fixestの`kpr`は参照にならない。詳細: `docs/spec/iv-spec.md`3.4.2節。
- **有効F（`shared/effective_f.rs`）の一般化版TSLSの臨界値の探索で、`tr S1`の最小点`β*`まわりの`sinh`格子を落とさない。`tan`の格子にしない**: 誤差がほぼ比例する強い内生性で鋭い山を取りこぼし、`B_TSLS`が9%過小になった。`tan`の格子は端に埋もれる山を見落とす。`y_dep`は他の行列と同じ変換（`√w`・within）を済ませて渡す。詳細: `docs/spec/iv-spec.md`3.4.3節。

## Wu-Hausman

- **Wu-Hausman（`common/wu_hausman.rs`、2SLS・GMM共通）は`fit()`の`cov_type`に追従し、想定内の失敗（拡張回帰の`PerfectFit`・`SingularMatrix`・`InsufficientObservations`・`UnitLeverage`）だけを`None`にする。それ以外（Wald側の`NearSingularMatrix`を含む）は`HausmanRegressionFailed`として伝播する**: 広すぎる`Err(_)`のキャッチは将来の実装バグを`None`で隠す。拾う失敗は`is_expected_failure`に明示する。結果は`?`ではなく`match`で振り分ける。詳細: `docs/spec/iv-spec.md`3.6節。
- **Wu-Hausmanのp値の分母自由度は拡張回帰自身の`df_resid`（主構造式の`df_resid - n_endog`）。統計量とp値は別々に照合する**: 主構造式の自由度を流用しても統計量は機械精度で一致し、p値だけがずれるので見逃しやすい。詳細: `docs/spec/iv-spec.md`3.6節。
