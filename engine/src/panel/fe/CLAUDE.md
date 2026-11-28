# engine/src/panel/fe/ 実装ノート（FE）

このファイルは `engine/src/panel/fe/` 配下のファイルを読み書きするときだけ自動ロードされる（親の`engine/src/panel/CLAUDE.md`も一緒にロードされる）。ファサードの`fe.rs`を読むだけではロードされない。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景・実測は`docs/spec/fe-spec.md`（FE/RE共通は`docs/spec/panel-common.md`）、定義式は`docs/methods/fe.md`が正本。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

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
