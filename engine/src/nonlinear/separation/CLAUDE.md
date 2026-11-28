# engine/src/nonlinear/separation/ 実装ノート（分離検出）

このファイルは `engine/src/nonlinear/separation/` 配下のファイルを読み書きするときだけ自動ロードされる（親の`engine/src/nonlinear/CLAUDE.md`も一緒にロードされる）。ファサードの`separation.rs`と、LPの`engine/src/shared/linear_program.rs`を読むだけではロードされない。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景・実測は`docs/spec/logit-spec.md`3.2節が正本。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## 分離検出

- **閾値付きの指標（標準化パラメータのノルム。`n`依存にしたものも含む）に戻さない。判定はLP（`max Σuᵢ'θ s.t. uᵢ'θ≥0, |θⱼ|≤1`）の解の最大マージン（和ではない）`>1e-6`**: ノルムは小さい`n`で分離を見逃し、準完全分離（ダミー型）を全件見逃し、強い共線性で有限のMLEを棄却する。和は`n`に比例して雑音が積もる。行は単位長にそろえ、目的関数を`Σ‖z‖`で正規化しない。詳細: `docs/spec/logit-spec.md`3.2節「完全分離・準完全分離の検出」。
- **事前スクリーニング（`θ̂`や情報行列の最小固有ベクトルの方向でほぼ全観測が正側のときだけLPを走らせる）でLPを省かない**: 分離方向が2次元以上だと最小固有ベクトルが錐の外を向いて取りこぼす（一度実装して廃止）。詳細: 同上「常にLP（制約生成）で判定する」。
- **`DenseSimplex`（`shared/linear_program.rs`）のDantzig規則・退化30回でのBland規則への切り替え・右辺の`1e-11`級の決定的な摂動の3つを外さない。比検定の同点幅`RATIO_TIE_TOLERANCE`（`1e-15`）は摂動より十分小さく保ち、違反判定の許容幅（`1e-9`）は摂動による解のずれ（`2e-11`）より大きく保つ**: 右辺0の錐制約は原点で強く退化し、どれを外しても`k>=10`で失敗しうる。同点幅`1e-12`は混合スケールの`k>=14`で終わらず、摂動を`1e-8`に上げると全件失敗する。回帰ガード: `separation/lp.rs`の`handles_many_regressors_without_numerical_failure`・`does_not_fail_on_mixed_scale_data_without_separation`・`matches_brute_force_vertex_enumeration_on_random_problems`、`separation/ordered.rs`の`handles_about_fifty_variables_without_numerical_failure`、`tests/support/_nonlinear_helpers.py`の`mixed_scale_binary_dataset`。詳細: `docs/spec/logit-spec.md`3.2節「LPソルバーの差し替え」。
- **LPが数値的に完了しないとき（ピボット上限・作業集合の肥大・解が自分の制約を破る）は、`raise_on_non_convergence=true`なら`SeparationCheckFailed`（`ComputationError`）、`false`なら`converged=false`で結果を返す**: 分離が無いと確認できていない。詳細: 同上。
- **2値モデルの分離検出の行は標準偏差で割るだけ（平均を引かない）にし、標準化後の行列を複製せず元の`x`と`ColumnScale::stds()`から都度計算する（`separation/rows.rs`の`SignedRows`）。順序モデルは中心化した行を使い、`verify_no_ordered_separation`の`means`には列の平均を渡す**: 閾値`1e-6`はそれぞれの行で較正してある。複製すると大標本でメモリが倍になる。平均は要素ごとに引く（和の後に引くと平均の大きい列で桁落ちする）。順序モデルの行の作り方と前提は`separation/ordered.rs`のモジュールdoc。
