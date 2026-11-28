# engine/src/panel/fe_iv/ 実装ノート（FEIV。2SLSとGMM）

このファイルは `engine/src/panel/fe_iv/` 配下のファイルを読み書きするときだけ自動ロードされる（親の`engine/src/panel/CLAUDE.md`も一緒にロードされる）。ファサードの`fe_iv.rs`を読むだけではロードされない（`fe_iv/two_sls.rs`・`fe_iv/gmm.rs`はこのディレクトリにあるのでロードされる）。FE-GMMの`gmm.rs`・`gmm/`を読むと、IVのGMMと共有する核の罠（`.claude/rules/engine-gmm-core.md`）も一緒にロードされる。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景・実測は`docs/spec/fe-iv-spec.md`、定義式は`docs/methods/fe_iv.md`が正本。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## FEIV

- **全列（`y`・`x_exog`・`x_endog`・`instruments`）を`prepare_within`で1回だけwithin変換し、第一段階は変換済みの列を`FeEstimator::fit_from_within`に渡して内生変数ごとに回す。第一段階には`iv::two_sls`のコアを使わない**: 交互射影が高コスト。`TwoSlsEstimator`は自由度が`n − k`固定で吸収した効果を渡せない。前段の順（操作変数の数→singleton→定数列→変換→自由度（構造式`n > k + neffects`→操作変数`n > l + neffects`）→変換後の分散ゼロ）はFE-2SLSとFE-GMMで共有する。詳細: `docs/spec/fe-iv-spec.md`3.2節・3.3節。
- **第一段階は`tests_overall_f=false`（`PanelCovInput::tests_slopes`）で、`G`・`T`が`l`以下でも標準誤差は計算し傾きのFだけを`None`にする。構造式は`G <= k`でエラー**: IV本体の`first_stage()`と揃える。標準誤差に関わる検査（`G >= 2`・2グループ退化・完全適合）は残す。詳細: `docs/spec/fe-iv-spec.md`3.2節。
- **第一段階の`FeEstimator`を推定量に保持しない（`fit_with_first_stages(.., consume)`で取り出した直後に手放し、CD・KP・有効F用には`FirstStageWithin`だけを持つ）**: 内生変数の数に比例してピークメモリが増えた。詳細: `docs/spec/fe-iv-spec.md`3.2節。
- **FE-2SLSの`X̂ = [x̃_exog, x̃_endog − 第一段階の残差]`、残差は実際の`X̃`で作る構造残差。HC2/HC3のレバレッジは`X̂`の素のレバレッジにダミーのレバレッジを足す**: ダミー込みの`X̂_full`のハット行列の対角と一致する（LSDVの2SLSで確認）。構造残差の完全適合（`structural_fit_is_perfect`）は交互射影の誤差を`|β̂_j|`倍した各`x`の分まで見積もる。詳細: `docs/spec/fe-iv-spec.md`3.2節。
- **定数列のエラーは役割つき（`ColumnRole`: FEは`x`、FEIVは`x_exog`・`x_endog`・`instruments`）で、該当列は役割ごとに全て`columns`に入れ、最初に該当列がある役割だけを返す。第一段階の失敗は`stage=FIRST_STAGE`、構造式は`SecondStageFailed`（FE-GMMは段階なしの`CommonError::PerfectFit`）**: 詳細: `docs/spec/fe-iv-spec.md`3.6節。
- **SarganとFE-GMMの`Classical`・`OneStep`のHansen Jは`σ̂² = e'e/(n − neffects)`（`S`を`n/(n − neffects)`倍）。`Robust`・`Cluster`の`S`は補正しない**: within残差は`σ²(n − neffects)`程度しか持たず、`n`で割ると小さい`T`で過剰に棄却する。`Robust`・`Cluster`は参照実装に無く、`n/(n − neffects)`でも原理的に正しくない。C統計量の`S_e`の尺度も同じ規則（`C`は`(n − neffects)/n`倍）。詳細: `docs/spec/fe-iv-spec.md`3.4節・3.5節。
- **FE-GMMの重みの型`FeWeightType`に`Hac`を足さない（`Classical`・`Robust`・`Cluster{groups}`のみ。`groups`は必須）。既定の重み（`effects[0]`のクラスター）はバインディングの責務**: 行単位のNewey-Westはパネルに使えず、DK型の重みはミクロパネルで実用にならない。実行時に弾くより型で表せないようにする。詳細: `docs/spec/fe-iv-spec.md`3.5節。
- **FE-GMMは点推定・共分散・Waldの後に第一段階を回し、点推定に使った行列は第一段階の前に手放す。Wald χ²の失敗は`WaldTestFailed`（`stage=OVERALL_WALD_TEST`）でF検定の`FTestFailed`と分ける。推論は正規分布・χ²**: GMMのエラーを先に返す。詳細: `docs/spec/fe-iv-spec.md`3.5節・3.6節。
- **FE-GMMの`Ω̂`は`panel_moment_covariance`に白色化した`Z̃`を渡して`Ω̃`を直接得る。HC2・HC3のレバレッジは`Z̃`の素のレバレッジ＋吸収した効果のレバレッジ（`leverage_full`）で、`iv::gmm`の定義とは違う。`panel_cov_params`は触らず、`Cluster`・`Dk`の事前検証と`K`だけを`cluster_setup`・`dk_setup`で共有する**: 詳細: `docs/spec/fe-iv-spec.md`3.5節、`docs/spec/orthogonal-coordinates.md`4.2節。
- **弱操作変数診断の`n_absorbed`は診断・`cov_type`ごとに違う（部分F・CD・`Classical`・HC・DKは`neffects`、`Cluster`は`fe_cluster_k_correction`の`K − l`）。`Hc2`・`Hc3`は`ScoreCovariance::HcAbsorbed`、DKは`ScoreCovariance::Dk`**: 共有関数のシグネチャは変えず、呼び出し側が渡す数を決める。診断は第一段階だけに依存するので、同じデータ・`cov_type`ならGMMと2SLSで同じ値（`gmm/fit/tests/first_stage.rs`）。詳細: `docs/spec/fe-iv-spec.md`3.3節・3.4節。
- **Wu-Hausmanの補助FE回帰は`tests_slopes=false`で回し、`G`が`k+1`〜`k+k_endog`でも値を持たせる（IV本体と違う）。構造式の共分散・F検定の後に回す。自由度不足・`v̂`の分散ゼロ・完全適合・特異は`None`、それ以外は`stage=WU_HAUSMAN_REGRESSION`で返す**: 使うのは`v̂`の`k_endog × k_endog`部分行列。詳細: `docs/spec/fe-iv-spec.md`3.4節。
- **FE-GMMの`fixed_effects()`は構造式の`FeInput`を保持せず呼び出し時に`structural_fe_input`で組み立てる（`Result`を返す）。C統計量（`fe_gmm_c_statistic`）は推定量のメソッドにせず、呼び出し時に`prepare_within`で変換し直す。FE-2SLSの`c_stat()`用の関数はengineに足さない（バインディングが`FeWeightType`を決めて同じ関数を呼び、`Dk`は弾く）**: `fit()`後に`n(1+k)`個の複製を持たない。C統計量の検査は`prepare_within`が先、`target`・`weight`が後。詳細: `docs/spec/fe-iv-spec.md`3.5節。
- **FE-GMMをダミー入りの`iv::gmm`と照合（FWL）するのは`OneStep`と`Classical`重みの2段階だけにする**: ロバスト・クラスター重みの2段階は重みを作る残差がwithin残差と違い、反復は3回目の推定から一致しない。詳細: `docs/spec/fe-iv-spec.md`3.5節・6.1節。
