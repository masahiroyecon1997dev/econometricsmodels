# RE: 不均衡1-wayの`swamy_arora`と`r_squared_between`を`plm`に揃える（設計ノート・たたき台）

対象: GitHub Issue #526。v0.11.0向けで、`release/v0.11.0`で直接作業する。
状態: 方針はユーザー確認済み（3章）。実装はほぼ完了（5章の進捗参照）。

## 1. 調査で確定した事実

### 1.1 `plm`の`swar`（不均衡1-way）の式（数値で確認済み）

`plm` 2.6.7の`ercomp`のソースと、8個体・40個体の不均衡パネルでの数値照合による（相対誤差1.3e-15で一致）。不均衡では`dfcor = c(3, 3)`固定で、`models = c("within", "Between")`:

- `σ_ε² = Q_w / (O − N − K_tv)`。`Q_w`は内部FEの残差平方和、`K_tv`は時間変化のある列の数。**現在の実装と同じ**（変更なし）。
- between回帰は`[1, x̄_i]`への**`T_i`重み付き最小二乗**（観測レベルに展開したpooled OLSと同値）。
  `Q_id = Σ_i T_i ē_i²`（`ē_i`は重み付きbetween回帰の残差）。
- `σ_u² = max(0, (Q_id − σ_ε²(N − p)) / (O − tr(B⁻¹S)))`。
  `p = K_b + 1`（切片込みのbetween回帰の列数）、`B = Σ T_i z̄_i z̄_i'`、`S = Σ T_i² z̄_i z̄_i'`、`z̄_i = [1, x̄_i]`。
- **実装しやすい形**: `√T_i`でスケールしたbetween回帰のQRのレバレッジ`h_i`（`OrthonormalDesign::row_leverages`が既にある）を使うと`tr(B⁻¹S) = Σ_i T_i h_i`。分母は`N > p`なら正（`h_i ≤ 1`・`Σh_i = p`）。
- 現在の式（重みなしOLS＋調和平均`t_bar`）との差は40個体の例でσ_u² 1.565（`plm`）対1.363（現行）。

### 1.2 バランスとの連続性（重要）

- `T_i`が全て等しいとき、`O − tr(B⁻¹S) = T(N − p)`、`Q_id = T·SSR_b`となり、上の式は`SSR_b/(N−p) − σ_ε²/T`に**代数的に一致する**（`plm`のバランスの`dfcor = 2`の式。数値でも`dfcor`既定と`dfcor = 3`が一致）。
- つまり**`swar`はWallace-Hussain・Amemiyaと違って、閉形式とトレース形が不連続にならない**。バランス・不均衡で分岐せず、トレース形1本で書ける。「`T_i`が等しく時点集合が違うパネル」（`plm`はトレース形、本実装は観測数で閉形式）でも、`swar`なら値が同じになり、判定の食い違いは問題にならない。

### 1.3 エッジケース

- **不均衡＋時間不変の列**: `plm`は通常どおり推定できる（確認済み）。`p`の数え方は、現行どおり「個体平均が一定の列と別名の列を落とす」を維持する。
- **不均衡＋Mundlak型（`x`と`x̄_i`を両方入れる）**: `plm`は`system is computationally singular`で**失敗する**（確認済み）。本実装は現行どおり別名の列をランクで落として推定できる（`plm`の上位互換）。この入力は`plm`が参照にならず、独立な手計算だけが参照になる。
- `plm`のバランス判定は時点集合まで見る（`is.pbalanced`）。`swar`は1.2のとおり関係しない。

### 1.4 現状の記述との食い違い（要訂正）

- Issue本文が挙げる「Hausmanのrtol 0.15」「`rtol_unbalanced_f`」は**コードに存在しない**（旧い記述）。`tests/support/_tolerances_panel.py`のREの許容誤差は全て機械精度。実際に緩んでいるのは次の3つ:
  1. `README.md`の表「RE: unbalanced panels: 5e-3 to 5e-2」。
  2. `engine/src/panel/re/hausman/tests/reference.rs:168`の不均衡Hausmanテスト（`plm`と`5e-2`）。
  3. 不均衡を`plm`と直接比べず、`plm`に**本パッケージ（linearmodels式）の分散成分を与える**回避策（given-variance方式）。
- `engine/src/panel/CLAUDE.md`51行目の「`r_squared_between`に`T_i`ベースの重み付けをしてはいけない」は**FEについては有効**（FEは対象外）。REは今回逆になるのでREだけ書き換える。`re.rs:75`の「`T_i`ベースの重み付けは常に無効」もRE側は修正対象。

## 2. 実装方針（案）

### 2.1 Rust（`engine/src/panel/re/`）

| 箇所 | 変更 |
|---|---|
| `variance.rs` `between_ssr_and_df` | 観測数の重み（`Option<&[f64]>`）を受け、`√w`でスケールして最小二乗。戻り値を`(SSR_w, p, Σ w_i h_i)`の構造体にする。2-way（バランスのみ）は重みなしのまま（値は変わらない） |
| `variance.rs` `one_way_variance_components` | `t_bar`をやめ、`σ_u² = max(0, (Q_id − σ_ε²(N−p)) / (O − Σ T_i h_i))`。バランス・不均衡の分岐は作らない（1.2） |
| `r_squared.rs` `re_r_squared_between` | 重み`T_i`（`means.t`）、重み付きの全体平均（`Σ T_i ȳ_i / O`）で`SSR_b`・`TSS_b`を作る（3.1の未決項目） |
| doc | `variance.rs`・`variance_method.rs`・`re.rs`のモジュールdocと`VarianceMethod::SwamyArora`のdoc、`engine/src/panel/CLAUDE.md`（51・62・67行目付近のRE節）、`engine_pybind/src/panel/re.rs`と`python_package/.../panel/re.py`のdocstring（`sigma2_eps`・`sigma2_u`・`variance_method`・`r_squared_between`） |

`compute_theta`・`transform.rs`・Hausman・`fit.rs`のコードは変更不要（σ_u²が変わるだけで値が連鎖して変わる）。新しいエラーは増えない。

### 2.2 Rustのテスト（`engine/src/panel/re/**`）

- 値を差し替える（`plm`実測値）: `variance/tests/variance_components.rs`（7観測 a:3,b:2,c:2）、`fit/tests/reference.rs`（params・SE・Hausman・R²・hc2/hc3/cluster/hac。hc2/hc3の「1e-9のlinearmodels由来の手計算」と「`plm`実測の緩い比較」の二本立ては、`plm`値の1e-9比較に一本化できる）、`hausman/tests/reference.rs:154-170`（`5e-2`→機械精度）。
- 追加: (a)観測ごとの行列に展開した参照実装とのトレース形の一致（`variance_method`の`trace_forms_match_the_observation_level_reference`と同じ流儀）、(b)`T_i`が全て等しいときに旧式（閉形式）と一致する、(c)重み付き`r_squared_between`の定義どおりの手計算（分解`Σ(y−ȳ)² = within + Σ T_i(ȳ_i−ȳ)²`との整合）、(d)Mundlak型の不均衡で別名の列を落としても`σ_u²`が列順によらない。バグ注入（`T_i`重みを外す、調和平均に戻す）で失敗することを確認する。

### 2.3 Python・ベンチマーク

- **`plm`が不均衡1-way REの主参照になる**（FEIV・2-way REと同様の「`linearmodels`が参照にならない」例外）。`linearmodels`はバランス（全統計量）と`r_squared_within`の参照として残す。
- **クロスチェック**: `plm`の分散成分（`ercomp`の`sigma2`）から`Ω = σ_ε² I + σ_u² ZZ'`を密に組み、`statsmodels.GLS`で係数・標準誤差を求めるフィクスチャを足す（3.5）。
- `benchmark/panel/fixtures/generate_re_fixtures.py`: `NUMERIC_SCENARIOS`から不均衡を外し`re.json`/`re_cov.json`はバランスのみにする。`test_re_reference.py`の`test_matches_linearmodels`も追従。
- given-variance方式（`_given_variance.R`・`run_plm_re_given_variance_benchmark.R`・`run_plm_re_given_variance_hausman_benchmark.R`・`r.py`の`run_re_given_variance_plm_r`ほか・`generate_re_crosscheck_fixtures.py`の`_check_given_variance_*`・`_linearmodels_variance`）は撤去し、不均衡も`run_plm_benchmark.R`/`run_plm_hausman_benchmark.R`で`plm`自身のREと直接比べる。ただし`_given_variance.R`のコメントにある制約（θが全行同じでpoolingが定数列を落とす等）と、`cluster`の`effects`先頭以外の列などの参照（`export_plm_re_transformed.R`経由）が残る箇所は撤去前に個別確認する。
- 除外を外せる箇所: `generate_re_statsmodels_cluster_fixtures.py`の`UNBALANCED_SCENARIOS`、`test_re_time_invariant.py`の`BALANCED_SCENARIOS`分岐（σ_u²・θを不均衡でも`plm`と直接比較）、`generate_re_variance_method_fixtures.py`の`hand_only_methods=("swamy_arora",)`（`ow_mundlak_unbalanced`は`plm`が失敗するので手計算のみのまま。`ow_mundlak_time_invariant`は`plm`フィクスチャを持てるか確認）、`test_default_variance_method_is_swamy_arora`のdocstring。
- `tests/support/_re_oracle.py`: `hand_computed`のswamy_aroraにトレース形（重み付きbetween＋`dfcor=3`）を足し、`r_squared`のbetweenを`T_i`重み付きにする。
- 全フィクスチャの再生成は`benchmark/regenerate_all.py`（R・Pythonとも環境は`/reference-benchmark`参照）。再生成後、**バランスの値が変わっていない**（フィクスチャの差分が不均衡だけ）ことを確認する。
- `tests/support/_tolerances_panel.py`: 数値の緩和解除は不要で、コメント（不均衡は`plm`の分散成分が違うので・given-variance）の書き換えのみ。

### 2.4 ドキュメント

`docs/spec/re-spec.md`（3.1節「plmとの既知の実装差」・3.1.2節末尾「`swamy_arora`の不均衡1-way」・3.6節・4章・5章の`swar`の方針）、`docs/methods/re.md`（42-51・83・86行目: σ_u²の式、`r_squared_between`の定義）、`docs/guide/verification.md`・`verification-details.md`（given-variance方式の説明・不均衡の参照・時点集合の注記）、`docs/guide/inference-conventions.md:24`、`README.md`（表の`5e-3 to 5e-2`と293-297行目の説明）、`CHANGELOG.md`（Changed: 不均衡1-wayの分散成分・θ・係数・標準誤差・Hausman・`r_squared_between`が変わる破壊的変更。0.xなので許容）、`docs/performance/re.md`（確認のみ）。公開ページの変更後は`/review-docs`。

### 2.5 作業の順序（案）

1. oracle（`_re_oracle.py`）とRの`plm`実測値でトレース形を固定（テスト先行）。
2. Rust: `between_ssr_and_df`の重み対応→`one_way_variance_components`→`re_r_squared_between`→Rustテストの値差し替え。
3. フィクスチャ再生成・Pythonテスト更新・given-variance撤去。
4. ドキュメント・CHANGELOG・nested CLAUDE.md。
5. `rust-reviewer`・`python-reviewer`・`testing-completeness-reviewer`、`/review-docs`。

## 3. 決定事項（2026-11-28 ユーザー確認済み）

1. **`r_squared_between`の重み**: 案A。重み`T_i`・重み付き全体平均`ȳ = Σ T_i ȳ_i / O`で、`SSR_b = Σ T_i(ȳ_i − β̂₀ − x̄_i'β̂)²`、`TSS_b = Σ T_i(ȳ_i − ȳ)²`。分解`Σ(y−ȳ)² = within + Σ T_i(ȳ_i−ȳ)²`と整合し、`linearmodels`の`_prepare_between`の重み`T_i/mean(T)`ともR²は同値（スケール不変）。外部参照が無いため、検証は定義どおりの独立計算のみ（例外としてdocに明記）。
2. **FEの`r_squared_between`**: 今回はREのみ。FEは`linearmodels`と一致する重みなしのままで、REとの定義の差をdoc（`docs/methods/re.md`・`fe.md`、docstring）に明記する。FEを揃えるかは別Issue（#697）で検討する。
3. **2-wayの`r_squared_between`**: バランスのみなので数値は変わらない（同じ関数）。不均衡2-wayでの定義はIssue #527で決める。
4. **旧式の扱い**: 置き換えのみ。`linearmodels`式は別名で残さない。
5. **不均衡1-wayの参照**: `plm`を主参照（分散成分・係数・標準誤差・Hausman）とし、**独立のクロスチェックとして`plm`の分散成分を与えた`statsmodels.GLS`（密な`Ω`、小さい`N`）で係数・標準誤差を確かめる**（σ_u²自体の参照は`plm`のみ。手計算オラクルは補助で参照ではない）。GLSの`cov_type`は`statsmodels`が直接対応する範囲（classical・HC系・cluster）を確認して決める。
6. **実装の形**: バランスも含めてトレース形1本（分岐しない）。バランスは`linearmodels`と1e-8で一致させたまま。
7. **Issue本文の訂正**: 実装時に#526へ訂正コメントを入れる（「Hausman rtol 0.15・`rtol_unbalanced_f`」は旧記述、1.4）。

## 4. 想定リスク

- 不均衡の既存ユーザー向けの数値が変わる破壊的変更（`CHANGELOG`に明記）。
- given-variance撤去は、参照の質を下げないことを確認してから行う（撤去は最後。置き換え後の`plm`直接比較が機械精度で通ってから）。
- `statsmodels.GLS`の密な`Ω`はO×Oなので、クロスチェックのフィクスチャは小さいパネル（O ≲ 数百）に限る。

## 5. 進捗（2026-11-28）

- 済: Rust（`between_regression`の重み・トレース形の`one_way_sigma2_u`・重み付き`re_r_squared_between`・doc）、Rustテスト（`plm`値への差し替え、`variance/tests/trace_form.rs`、between R²の分解テスト、不均衡の切り上げ、T_i={2,2,15}の`plm`照合、proptestにSwamy-Arora、バグ注入で検出を確認）、pybind・Pythonのdocstring、オラクル、フィクスチャ（`re.json`・`re_cov.json`はバランスのみ、`re_crosscheck*.json`に不均衡を`plm`直接で＋不均衡の実データ`wagepan_unbalanced`、`re_statsmodels_cluster*.json`に不均衡、新規`re_gls.json`・`re_gls_cov.json`、`re_variance_method*.json`の不均衡1-wayにswamy_arora）、given-variance方式の撤去、許容誤差コメント、仕様書、公開ドキュメント、README・CHANGELOG、ネストCLAUDE.md、#697へのコメント（FEのdocstring）。
- レビュー: rust・python・testingの3エージェント実施、指摘は反映済み（テスト網羅性の拡張A〜Dはユーザー確認のうえ実施）。
- GLSで比べられるのはclassicalと個体クラスターだけ（Choleskyの白色化と準偏差変換は個体ごとの直交変換だけ違い、HC1〜HC3は変わる）。
- 全体テスト・#526への訂正コメント・コミット（`c22082d3`、push済み）は済み。
- `/review-docs`の指摘で判明: `linearmodels` 7.1の`RandomEffects.fit(small_sample=True)`は、不均衡パネルの1-wayのSwamy-Aroraを`plm`の`swar`と同じトレース形（`√T_i`で重み付けしたbetween回帰）で計算し、本実装と機械精度で一致する（既定の`small_sample=False`だけが重みなし＋調和平均）。時間不変の列もwithinの自由度に数えない（従来の「4桁目から食い違う」は7.1では成り立たない）。
- ユーザー確認のうえ（2026-11-28）、`linearmodels`（`small_sample=True`、classical/hc1）を不均衡パネルと時間不変の列の2つ目の参照として`re.json`に追加: `unbalanced`・`zero_variance_regressor`・`time_invariant_unbalanced`・`time_invariant_multi_unbalanced`・`wagepan_unbalanced`・`wagepan_time_invariant`。バランスの既存値はビット一致。不均衡パネルの`rsquared_between`は`linearmodels`が重み付けしないため比べない。
- `/review-docs`の指摘（must/should、一部nice）は公開ドキュメント・docstring・仕様書・ネストCLAUDE.mdに直接反映。
- 全体テスト再実行（23384 passed）・コミット（`6a226233`）・push・#526への追記コメントとクローズは済み。
- `/review-docs`の2回目の指摘（must 1件: Hausmanの`phtest`一致条件の書き漏れと相互参照の循環。ほかshould・nice）も全て直接反映した。
