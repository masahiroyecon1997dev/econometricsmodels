# engine/src/panel/re/ 実装ノート（RE）

このファイルは `engine/src/panel/re/` 配下のファイルを読み書きするときだけ自動ロードされる（親の`engine/src/panel/CLAUDE.md`も一緒にロードされる）。ファサードの`re.rs`を読むだけではロードされない。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景・実測は`docs/spec/re-spec.md`（FE/RE共通は`docs/spec/panel-common.md`）、定義式は`docs/methods/re.md`が正本。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## RE

- **REの内部FE（σ_ε²用）は`FeInput`と同じsingleton検証を継承するので、`T_i=1`のエンティティは`SingletonGroup`。ベンチマークデータは全エンティティ`T_i>=2`にする**: linearmodelsの`RandomEffects`はsingletonを処理できる（本実装の方が厳格）。詳細: `docs/spec/re-spec.md`3.1節。
- **時間不変の説明変数は許容し、内部FEには時間変化のある列だけを渡す（`VarianceComponents.time_varying`で元の列に対応づける）。この対応を使う`re_r_squared_within`と`re_hausman_test`、`debug_assert_within_reuse_contract`は`k_tv`列で数える**: `fe.within_input().x()`の列数は`input.x()`ではなく`k_tv`。`plm::phtest(method="aux")`は時間不変の列があると係数を位置で選び間違えるので参照にしない。詳細: `docs/spec/re-spec.md`3.7節・5章。
- **1-wayのSwamy-Aroraは`plm`の`swar`のトレース形（`dfcor=3`）で、バランス・不均衡で式を切り替えない**: `T_i`が全て等しいと閉形式に一致するので不連続が無い。linearmodelsの既定（`small_sample=False`）とは別の推定量で、比べるときは`small_sample=True`。詳細: `docs/spec/re-spec.md`3.1節。
- **`variance_method`（`wallace_hussain`・`amemiya`）はバランスが`dfcor=1`の閉形式、不均衡の1-wayが`dfcor=3`のトレース形で、バランスでも両者の値は異なる。トレース形の式はproptestの不変性ではなく、観測ごとの行列に展開した参照実装との一致で守る**: 不変性のプロパティは式が一貫して誤っていても通る。回帰ガード: `trace_forms_match_the_observation_level_reference`。詳細: `docs/spec/re-spec.md`3.1.2節。
- **between回帰は平均が単位間で一定の列を落とし`K_b`にも数えず、別名の列（Mundlak型）は`independent_columns_in_order`で列順に後ろを落とす。`OlsEstimator`を経由しない**: 落とさないと切片と共線で特異。`n_entities > k+1`が必要。詳細: `docs/spec/re-spec.md`3.1節。
- **ハウスマン検定は`X̃`の識別できない列（`x*_j - x̃_j`が全観測で一定）を検定対象から外し、定数項は準偏差変換しない。`None`は検定できる係数が無いときだけで、補助回帰のランク落ち・クラスター数不足・ほぼ特異は`HausmanTestFailed`で`fit()`全体を失敗させる**: `plm::phtest(method="aux")`に合わせる。詳細: `docs/spec/re-spec.md`3.7節。
- **F統計量はWald二次形式にし、linearmodelsとは`f_statistic_robust`で比べる。`plm`のp値は使わない**: linearmodelsの`f_statistic`はSST/SSR方式で極端な不均衡パネルで負になる。`pwaldtest`の分母自由度は本実装の規約と一致する保証がない。詳細: `docs/spec/re-spec.md`3.5節。
- **HC2/HC3は`plm::vcovHC(method="white1")`、クラスター・DKは`plm`の`arellano`・`vcovSCC`の`type="sss"`と比べる**: linearmodelsはHC1相当しか持たず、クラスター/HACの小標本補正も一致しない。詳細: `docs/spec/re-spec.md`3.4節・4章。
- **`ReEstimator`は`std_errors`等を常に自前のフィールドで持ち、`estimator()`へ委譲しない。`estimator().f_statistic()`/`r_squared()`は使わない**: 一部の`cov_type`だけ委譲する非対称を避ける。`include_intercept=false`の委譲で`k_constant`が0扱いになり値が誤る。REは`extra_df=0`、HC2/HC3のレバレッジは素の`h_ii`。詳細: `docs/spec/re-spec.md`3.3節・3.4節。
- **σ_ε²=0の2つの場合を区別する**: σ_ε²=σ_u²=0の同時ゼロは`θ_i`が`0/0`でNaNが全観測に伝播する（未対応）。σ_ε²=0単独は内部FEが完全適合を先に`PerfectFit`（`stage=VARIANCE_COMPONENT_WITHIN_REGRESSION`）で止める。REの最終回帰（`fit_auxiliary`）とハウスマン補助回帰は完全フィット・定数の判定をしない（被説明変数が準偏差変換後の値）。詳細: `docs/spec/re-spec.md`3.1節。
- **`quasi_demean_two_way`を2段階の`quasi_demean_column`で代替しない**: `θ₃ = θ₁ + θ₂ - 1 + σ_ε/σ₄ ≠ θ₁θ₂`。`σ_u²=0`または`σ_λ²=0`なら`θ₃`は厳密に0にする。詳細: `docs/spec/re-spec.md`3.2節。
- **全エンティティの`ȳ_i.`が一致する退化データでは、between回帰は`σ_u²=0`で成功し、`OlsEstimator`の推論表は`0/0`のt統計量を`p_value=NaN`で返す（panicしない）。係数のSEが0になる退化データを`OlsEstimator`に渡すテストではこの挙動に注意する**: 回帰ガード: `swamy_arora_variance_components_does_not_panic_when_entity_means_are_all_zero`、`shared/inference.rs`のテスト。
- **RE（1-way・2-way）の参照値は`tests/fixtures/benchmarks/re_two_way.json`・`re_time_invariant.json`・`re_variance_method.json`、`plm`が失敗する入力の唯一の参照は`tests/support/_re_oracle.py`。engineの単体テストの小パネル（6×4）の`plm`値は`re_two_way.json`の`small_hand_*`と同じ値を埋め込んでいる**: engineはJSONに依存しない。詳細: `docs/spec/re-spec.md`4章。
