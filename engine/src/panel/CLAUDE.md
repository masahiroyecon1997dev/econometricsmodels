# engine/src/panel/ 実装ノート（FE/RE/FEIV）

このファイルは `engine/src/panel/` 配下のファイルを読み書きするときだけ自動ロードされる。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景・数式・実測は `docs/spec/fe-spec.md` / `docs/spec/re-spec.md` / `docs/spec/fe-iv-spec.md`（FEIV。2SLSとGMM）/ `docs/spec/panel-common.md`（FE/RE共通）が正本、定義式は `docs/methods/fe.md` / `re.md` / `fe_iv.md`。各部品の契約は `fe.rs` / `re.rs` / `fe_iv.rs` のモジュールdocにある（このファイルはその要約ではなく差分の索引）。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## ファイル構成

- `common.rs`はFE/RE/FEIVが共有する部品の再エクスポートだけで、中身は`common/`（`error`・`index`・`time_keys`・`validation`・`transform`・`cov_type`・`cov_params`・`fe_adjustment`・`dummy_leverage`・`panel_cov`・`moment_cov`（FE-GMMの`Ω̂`）・`test_support`）。`panel_cov_params`（`PanelCovType`/`PanelEffects`。`FeCovType`/`ReCovType`/`ReEffects`は型別名。`FeEffects`はFE専用の別enumで3次元以上の`MultiWay`を持つ）のFE固有の補正は`PanelStructure::FixedEffects`、REは`RandomEffects`で選ぶ。
- `fe.rs`・`re.rs`・`fe_iv/two_sls.rs`・`fe_iv/gmm.rs`はファサードで、ロジックは`fe/`（`input`・`within`・`fit`・`r_squared`・`fixed_effects`）・`re/`（`input`・`variance`・`variance_method`・`transform`・`fit`・`r_squared`・`hausman`）・`fe_iv/two_sls/`（`fit`・`sargan`）・`fe_iv/gmm/`（`fit`・`gmm_type`・`hansen_j`・`c_statistic`）にある。FE-2SLSとFE-GMMが共有する入力型（`FeIvInput`・`RoleColumns`・`PanelIndex`）・前段（`prepare_within`）・第一段階・弱操作変数診断・Wu-Hausmanは`fe_iv/common/`。`fit`を通すテストと`mod proptests`は各`fit/tests/`にテーマ別。
- 入力型の索引の持ち方は3通り: `FeIvInput`は`PanelIndex`（K次元の`Vec<GroupCodes>`＋列名）、`FeInput`は同じ形の自前の`groups`（`from_effect_groups`がK次元、`from_group_codes`は2次元の簡便版）、`ReInput`は2次元のまま`entity_codes`/`time`。空の役割は入力型では受理し、`x_endog`・`instruments`の空の拒否は`engine_pybind`、操作変数の不足は推定器が担う（`IvInput`と同じ層分け）。

## 全手法の`fit()`に共通

- **`fit()`冒頭で`crate::shared::parallelism::ensure_serial()`を呼ぶ**: faerの暗黙の全コア並列はtall-skinnyな設計行列で20〜200倍遅く不安定になる。回帰ガード: `fe_estimator_fit_pins_faer_global_parallelism_to_seq`等。詳細: `.claude/rules/rust-style.md`「パフォーマンス」。
- **設定値（`confidence_level` → `max_iter` → `tol`、GMMは続けて`gmm_max_iter` → `gmm_tol`）は`panel::common::validate_confidence_level`等で、singleton・定数列・within変換・自由度より前に検査する**: 入れ子の回帰（REの内部FE・FEIVの第一段階）にある同じ検査は備えとして残す。詳細: `docs/spec/error-handling.md`5.2節、`docs/spec/fe-spec.md`3.1節。
- **`engine`はpolars非依存で、グループはコード化済みの`GroupCodes`で受け取る。`String`キーの`HashMap`/`BTreeMap`で列ごと・統計量ごとに引き直さない。コードはキーの辞書順に振り（時点は`TimeKeys`の順）、グループ間の加算があるパネル系の実装はこのコード順で回す**: 引き直しはn=100万で1列約0.2秒かかった。辞書順はクラスターの和とbetween回帰の行の順を固定する。REの内部FEは`FeInput::from_re_input`で`ReInput`のコードごと作り、作り直さない。詳細: `docs/spec/panel-common.md`4.6節。
- **`quasi_demean_column`は列単位の関数で、FEは`θ_i = 1`の特殊ケース。グループ平均は返さない（`fixed_effects()`は保持した入力と`params()`から再計算する）**: 長さの不一致は内部の契約違反なので`assert!`でpanicしてよい。詳細: `docs/spec/panel-common.md`4.6節。
- **`cov_type`の計算は`panel_cov_params`に集約し、数式をFE/RE/FEIVで複製しない。FE固有の部分（HC1の`n/df_resid`、HC2/HC3のダミーのレバレッジ、クラスターの`K`、DKの`T/(T-1)`）だけここで決め、外積の和・カーネル和・クラスターの蓄積と補正式は`shared`を使う**: `shared::covariance`の係数共分散の関数はそのままでは使えない。詳細: `docs/spec/panel-common.md`4.6節。
- **`Dk`の`time`は必須の`TimeKeys`で、`FeInput`の2つ目の次元にフォールバックしない。`TimeKeys`の順序は呼び出し側が指定し（数値は数値順、日付は時系列順、`Enum`は定義順）、文字列の辞書順を時間順に使わない**: どの列がDKの時点か明示しないと意図と違う列が選ばれても気づけない。辞書順は`1, 10, 11, 2`となり標準誤差が黙って変わる（`TimeKeys::lexicographic`はISO 8601等向けの簡便版）。詳細: `docs/spec/fe-spec.md`3.3節。
- **クラスター数・時点数が検定する係数の数以下なら、共分散計算の前に`InsufficientClustersForInference`/`InsufficientDkPeriodsForInference`で弾く。数値的な特異性判定に任せない**: DK共分散のrankが`T-1`以下で、理論上0の固有値の丸め誤差が閾値をすり抜け`1e13`級の統計量を黙って返した。詳細: `docs/spec/fe-spec.md`3.3節。
- **2グループ退化（FEのDKで`t_periods=2`、クラスターで`G=2`、全エンティティが2グループに1観測ずつ）は`two_group_split_is_degenerate`で`k >= 1`のとき事前に弾く。2-wayでは時点方向も見る**: within変換でグループスコアが恒等的に0になり、`1×1`行列は相対閾値でも検出できずSE≈1e-16・F≈1e30を返す。REは対象外。`Dk.time`の上書き列は`FeInput`の検証を通らないので、DKアームの先頭で長さを検証する。詳細: `docs/spec/fe-spec.md`3.3節。

## FE

- **`FeEstimator`は`OlsEstimator`を経由せず、within変換後の設計行列のQR（`OrthonormalDesign::new_checked`）1本で係数・残差・共分散・F検定を求める。n×kの列ピボットQRと`(X̃'X̃)⁻¹`を足さない**: 2本目のQRで+25〜41%遅くなった。直交座標なので傾きの相関が極端でも`hc1`・`cluster`が通る。`x=[]`（k=0）はengineでは受理する（拒否は`engine_pybind`）。完全適合・準共線のエラーは`FTestFailed`。詳細: `docs/spec/fe-spec.md`3.1節。
- **2-wayはバランス（`covers_every_combination`: `n == N×T`かつ重複なし）のときだけ閉形式（entity→timeの2段階のquasi-demean）、それ以外は交互射影。件数の一致だけでバランスと判定しない**: 重複と欠落が相殺する入力を見逃す。FEの2-wayは同じ組合せの重複行を許す（必ず不均衡側に入る）。`validate_balanced_panel`はREの2-wayだけが使う。詳細: `docs/spec/fe-spec.md`3.1節。
- **不均衡な2-way・3次元以上（`MultiWay`）の交互射影は誤差が残るので、分散ゼロ・完全適合の判定に射影の誤差の見積もり（`Projected::error_bound`、`column_is_zero_variance_with_noise`）を足す**: 他の落とし穴: 自由度は連結成分の数によらず`k + N + T − 1`（fixest・linearmodelsと同じ）、HC2/HC3のダミーのレバレッジは小さい次元の密な行列で解く（`dummy_leverage`、水準数3000が上限。引き算の丸めはTwoSumで集める）、`fixed_effects()`は連結成分ごとに基準を0にする、鎖状のパネルは掃引が鎖長の2乗で増え`AbsorptionNotConverged`になる。詳細: `docs/spec/fe-spec.md`3.1節・3.5節・3次元以上の固定効果のテスト。
- **singletonはwithin変換の前に生の各次元（`effects`の順）から数えて`SingletonGroup`にし、下流の特異行列エラーとして偶発的に検出させない**: 2-wayは`time`の存在確認が先。詳細: `docs/spec/fe-spec.md`3.1節。
- **分散ゼロの閾値は「変換前の元の列の最大絶対値×`n`×`f64::EPSILON`」との比較（`column_is_zero_variance`）。変換後の値自身を基準にしない**: 自己参照になり閾値が機能しない。詳細: `docs/spec/fe-spec.md`3.1節。
- **生の値が全行で同じ説明変数は`RegressorConstant`、被説明変数は`OutcomeConstant`をsingleton検出の直後に、within変換後に分散の残らない`y`は`OutcomeConstantWithinEntity`を相対閾値で、完全フィットはQRの直後に`WithinRegressionFailed{PerfectFit}`で返す。厳密な等値で判定しない**: 固定効果を除く前に分かる事実と除いた後の分散ゼロを分ける。`0.1`を3回足して3で割った平均は`0.1`と一致しない。詳細: `docs/spec/fe-spec.md`3.1節・3.6節。
- **`df_model = k + neffects`（`absorbed_parameters`）。AIC/BICはペナルティの乗数だけ`df_model`にする。クラスターは`G-1`、DKは`t_periods-1`の自由度**: 詳細: `docs/spec/fe-spec.md`3.2節。
- **`r_squared_between`に`T_i`の重みを付けない（REは逆に付ける）**: linearmodelsの`_prepare_between`は重みを計算するが、サンプルウェイト未指定なら`1.0`に上書きする。バランスパネルでは`w_i=1`に退化して気づけない。**ループや分岐が一部の入力でしか通らないコードは、通らない入力でも検証する**。回帰ガード: `fe_estimator_fit_one_way_r_squared_between_matches_linearmodels_on_unbalanced_panel`。詳細: `docs/spec/fe-spec.md`3.4節。
- **`fixed_effects()`の2-wayの基準時点は`time`の時間順で最初の値に固定する（観測順に依存させない）**: fixestの`fixef()`は観測順で最初に現れた値に見え、一致は基準時点が同じ入力に限る。詳細: `docs/spec/fe-spec.md`3.5節。
- **fixestとの比較で、F検定のp値は`wald()`のものを使わず統計量と`degrees_freedom(model,"t",vcov)`から計算し直す。DKは`bandwidth <= T-2`で比べる。`aic`・`bic`と2-wayのwithin R²はfixestとだけ比べる**: `wald()`は分母自由度を`max(df2, df1+1)`に切り上げる。fixestの`cpp_driscoll_kraay`は`bandwidth == T-1`で最終ラグ項を落とすoff-by-oneのバグ（本実装は標準カーネルを維持）。linearmodelsは`aic`・`bic`を持たず、2-wayのwithin R²は個体だけのdemean。詳細: `docs/spec/fe-spec.md`3.3節・4章。
- **HC0は提供しない**: 仕様判断。他パッケージの有無（fixestは無い、plmは有る）は理由にしない。詳細: `docs/spec/panel-common.md`3.1節。

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

## FEIV（`fe_iv/`）

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

## PanelError

- **FE/RE/FEIVで`PanelError`を共有し、`FeError`/`ReError`を作らない。`report()`は全バリアントを`_`なしで列挙する（到達不能な`IdentifierDimensionMismatch`・`TwoWayRequiresTime`は`INTERNAL_ERROR`）**: `engine_pybind`の`panel_error_to_pyerr`は`to_pyerr`を呼ぶだけで更新は要らない。`code`の表は`docs/spec/fe-spec.md`3.6節。
- **回帰の失敗を包むバリアント（`WithinRegressionFailed`等）は`#[from]`でなく`.map_err`で包み、対象が異なる回帰は別バリアントにする。別の`PanelError`を文脈つきで包むときは`in_stage`（`stage_allowed`な`code`だけ）**: `CommonError`が2経路で`PanelError`になる曖昧さを避ける。詳細: `docs/spec/panel-common.md`4.4節。
- **`SingletonGroup`の`column`（`EffectNames`で運ぶ列名）はエラーに載せるためだけの値で、計算に使わない**: 既定は`"entity"`・`"time"`・`"effect3"`…。`n_dropped_rows`は取り除くことを繰り返したときの連鎖を含む。
- **REで`confidence_level`が不正なときは`QuasiDemeanedRegressionFailed`ではなく`WithinRegressionFailed`として伝播する**: 分散成分推定の最初の委譲が真っ先に検証するため。

## property-basedテスト（`mod proptests`）

- **バグ注入は`common/transform.rs`（FE・REの両方の土台）で、観測単位で値を汚すか行の順序に依存させる形にする。「グループ平均を定数倍ずらす」注入はしない**: `y = Σx + ノイズ`のDGPでは`y`と`x`で相殺して検出されない。詳細: `docs/spec/panel-common.md`5.5節。
