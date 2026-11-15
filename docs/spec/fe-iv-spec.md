# FEIV 仕様書（FE-2SLS・FE-GMM）

FEIV（固定効果つきの操作変数法。`estimator`で2SLSとGMMを切り替える）の確定済み仕様。実体は「within変換＋2SLS/GMM」で線形。
`engine/src/panel/fe_iv.rs`（`common`・`two_sls`・`gmm`の3つの子モジュールを持つ親。`iv/`と同じ構成）・
`engine/src/panel/fe_iv/`（`common/`: 入力型と前段、`two_sls/`: `FeTwoSlsEstimator`、`gmm/`: `FeGmmEstimator`）・
`engine_pybind/src/panel/fe_iv.rs`と`fe_iv/`（`options`・`option_values`・`gmm_setup`・`input`・`fit`・`result`）・
`python_package/econometricsmodels/panel/fe_iv.py`として実装済み。
FE/REと共通の設計判断（`effects`の引数設計・`cov_type`の対応・共通化方針）は[`panel-common.md`](./panel-common.md)、
FEの内部実装（within変換・自由度・`cov_type`・R²・固定効果の復元）は[`fe-spec.md`](./fe-spec.md)、
IV本体（2SLS・GMM・診断）は[`iv-spec.md`](./iv-spec.md)を参照し、本ドキュメントにはFEIV固有の内容のみを記載する。
定義式（推定量・共分散・検定分布・診断）の正本は`docs/methods/fe_iv.md`で、ここには重複させない。

## 1. スコープ

- 対象: `effects`は1〜5列（FEと同じ。1-way・2-way（不均衡も可）・3列以上）。切片は持たない（固定効果が吸収する）。
- 対象外（確定）: First-differences IV、RE-IV（EC2SLS）、Hausman-Taylor。時間不変の内生変数＋時間不変の操作変数は識別されず、
  「固定効果を除くと一定」のエラーになる。
- 自由度は`fixest`に合わせる（`df_model = k + n_absorbed`、`df_resid = n − df_model`。`n_absorbed`の数え方は3.3）。

## 2. API

3層構成: `FEIV(data, y, x_exog, x_endog, instruments, effects, *, options)`（python_package）→
`fit_fe_iv`（engine_pybind）→`FeTwoSlsEstimator::fit`・`FeGmmEstimator::fit`（engine。`estimator`で振り分ける）。
必須の列は`data`の直後（`docs/spec/api-conventions.md`）。コンストラクタでは検証せず、`fit()`で初めて`_lib`を呼ぶ。
`FEIVOptions`は`_lib`の再輸出（独自クラスにしない）。`summary()`・`predict()`は持たない（FEと同じ）。

### 2.1 2SLSとGMMは1つのクラスにする（IVと同じ）

`FEIVOptions.estimator`（`"2sls"`既定・`"gmm"`）で切り替え、`FEIV`・`FEIVResults`は1組にする。GMMを別クラス（`FEGMM`）にしていた時期があったが、IVとの
インターフェースが大きく異なり利用者が分かりにくくなるため、IVと同じ形に統合した（IVで推定量ごとにクラスを分けなかったのも同じ理由）。
`FeGmmEstimator`は第一段階・弱操作変数診断・Wu-Hausman・`fixed_effects()`を持たないので、推定量が持たないものは次のとおり扱う
（IVの`wu_hausman_*`がGMMで`None`になるのと同じ形）:

- **プロパティ**は`None`（`weak_instrument_f_*`・`wu_hausman_*`）。型は`int | None`・`dict | None`。
- **メソッド**は`ValidationError`（`METHOD_NOT_AVAILABLE_FOR_ESTIMATOR`。`method`・`estimator`・`supported`を`params`に持つ）。GMMの`first_stage()`・`fixed_effects()`・
  `cragg_donald()`・`kleibergen_paap()`・`effective_f()`、2SLSの`c_stat()`。
- 将来GMMに第一段階・弱操作変数診断・`fixed_effects()`を足せば、このエラーを外すだけで済む（engine側の追加が要る）。FE-2SLSの`c_stat()`も同様。

### 2.2 `FEIVOptions`

`estimator`（`"2sls"`既定・`"gmm"`）、`cov_type`（既定`"cluster"`、`classical`・`hc1`〜`hc3`・`cluster`・`dk`。`hc0`は無い）・`cluster`・`dk_time`・`dk_bandwidth`・`confidence_level`（既定0.95）・
`tol`・`max_iter`（交互射影。FEと同じ）。`cluster`の既定は`effects[0]`。`dk_time`は`effects`と独立で、`cov_type="dk"`のとき必須
（`effects`の列は借用しない）。`include_intercept`は持たない。

### 2.3 GMMのオプション

上の項目に、`gmm_type`・`gmm_weight_type`・`gmm_max_iter`・`gmm_tol`・`raise_on_non_convergence`を足したもの。名前と規則は`IVOptions`と同じ
（既定`None`→使われるモードの実効既定値。使われないモードでの明示指定は`ValidationError`。`estimator="2sls"`での明示指定も`OPTION_UNUSED`）。決定:

- **`gmm_weight_type`の語彙は`classical`・`robust`・`cluster`の3つ**。`hac`は型に無い（`OPTION_UNKNOWN_VALUE`）。IVの`hac`重みは行単位のNewey-Westで、
  `hac_time`が全行で異なる値を要するためパネルに使えない。DK型の重み（時点ごとに集計したモーメントにカーネル）は、重み行列のランクが時点数`T`以下になり、
  `l`より大きい`T`を要するので、`T`が6前後のミクロパネルでは実用にならない。必要になれば値を足せる（互換）。
- **重みの既定は`cluster`**（IVの`classical`と違う）。within変換した誤差は個体内で相関する（個体内で和が0）ので、`robust`は小`T`で妥当でなく
  （3.5）、`classical`の2段階GMMは2SLSと同じ結果になる。`cov_type`の既定が`cluster`であることとも揃う。
- `cluster`は`cov_type`と重みで共用し、指定が無ければどちらも`effects[0]`。どちらも使わないときに指定すると`ValidationError`。
- `gmm_type="one_step"`は重み`(Z̃'Z̃)⁻¹`を使い、重みの種類を持たない（`gmm_weight_type`は`None`。指定すると`ValidationError`）。
- `cov_type`は重みと独立のサンドイッチ一般形で決まる（`gmm_weight_type`と自由に組み合わせられる）。`hc0`は無い。

### 2.4 結果

`FEIVResults(raw, x_endog)`は2SLSの項目に加えて`estimator`・`gmm_type`・`gmm_weight_type`（`one_step`と2SLSは`None`）・`n_iter`・`converged`を持つ。
`x_endog`を保持する（`_lib`の`weak_instrument_f_statistics`・`first_stage()`は`HashMap`で順序が不定なので、
`x_endog`の順の`dict`に直す。pickleはラッパーごと往復する）。係数・標準誤差・推論・`df_*`・`n_groups`・`dk_bandwidth_used`・`residuals`、
Wald F（`wald_*`）、`first_stage()`（`dict[str, FEResults]`）、`weak_instrument_f_*`・`overid_*`（Sargan）・`wu_hausman_*`、R²3つ、`fixed_effects()`、
診断メソッド`cragg_donald()`・`kleibergen_paap()`・`effective_f(*, alpha)`（3.4）を持つ。`log_likelihood`・AIC・BICは出さない。

`estimator="gmm"`の結果の公開面は**engineが持つものだけ**: 係数・推論（`stat_dist="normal"`、`stat_df`は`None`）・`wald_*`（`wald_dist="chi2"`、`wald_df_denom`は`None`、`wald_df_num`は`k`。
Waldは`k`で割らない）・R²3つ・`overid_*`（Hansen J）・`gmm_type`・`gmm_weight_type`・`n_iter`・`converged`・`n_groups`・`c_stat()`。
第一段階・弱操作変数診断・Wu-Hausman・`fixed_effects()`は持たない（engineの`FeGmmEstimator`に無い。上の2.1の扱い）。

統計量の命名はIVに合わせる（`wald_*`＋`stat_dist`）。FEの`f_statistic`系ではない。`estimator="2sls"`は`stat_dist="t"`・`wald_dist="f"`、`"gmm"`は
`"normal"`・`"chi2"`（型は`Literal["t", "normal"]`・`Literal["f", "chi2"]`）。

**R²**は構造残差ベースの`r_squared_within`・`_between`・`_overall`（FEと同じ3つ）。係数に`β̂`、`X`に実際の内生変数を使い、
within=`1 − e'e/ỹ'ỹ`、between=個体平均に`β̂`を当てはめた残差、overall=元の`y`・`X`に`β̂`だけを当てはめた残差。負や1超になりうる。
`adj_r_squared`は出さない。IVの`r_squared`（`1 − SSR構造残差/SST`）と揃う。`fixest`の`wr2`は第二段階の残差ベースで定義が違い、照合に使えない（6.2）。

**`residuals`**はwithin変換後の空間での構造残差`ỹ − X̃β̂`（`FEResults.residuals`と同じ空間）。第二段階の残差ではない。

### 2.5 事後診断と`source_data`

診断の公開形は`CLAUDE.md`の規約（推定量の妥当性・識別に属するものは`fit()`時に計算してプロパティ、利用者の選択が入るものは独立メソッド）に従う。

- `fit()`時に計算してプロパティ: 部分F（弱操作変数）・Sargan・Wu-Hausman（2SLS）、Hansen J（GMM）。
- 事後診断のメソッド: `cragg_donald()`・`kleibergen_paap()`・`effective_f()`（2SLS）、`c_stat()`（GMM）、`fixed_effects()`（2SLS）。
  `fit()`では計算しない（`fit()`の計算量を増やさない）。`cov_type`は`fit()`のものを使い、メソッドの引数にしない。
- メソッドのために結果が`fit()`に渡した`DataFrame`への参照（`source_data`。コピーしない）・列の役割・解決済みのオプションを保持し、**呼び出しのたびに再推定する**
  （設計行列を結果に持たせない）。呼び出しの間にそのデータを変えないことをdocstringに書く。
- 戻り値は診断ごとの専用frozen dataclass（`CraggDonaldResult`・`KleibergenPaapResult`・`EffectiveFResult`。FEIVはIVと共有、組み立ては`_results_utils.py`）。
  p値の無い診断があるので`DiagnosticResult`は継承しない。`CStatisticResult`はIVと**同じ型を使い回す**（`IVResults.c_stat()`・`FEIVResults.c_stat()`）。
  臨界値は水準をキーにした`dict[float, float]`。
- 定義できないときは`None`でなく理由つきのエラー: `ALPHA_OUT_OF_RANGE`・`EFFECTIVE_F_ENDOGENOUS_COUNT`（`ValidationError`）、
  `WEAK_INSTRUMENT_STATISTIC_UNDEFINED`（`ComputationError`）、`C_STATISTIC_UNDEFINED`（`ComputationError`）。原因（共線・完全フィット・共分散のランク落ち・レバレッジ1）は
  1つのコードにまとめ、文面の括弧内に列挙するだけで区別しない。

### 2.6 `first_stage()`と`fixed_effects()`（`estimator="2sls"`）

- 第一段階は内生変数ごとに`FeEstimator`の後半（変換済みの列を受け取る内部経路）で1回だけ推定し、`X̂`の作成と`first_stage()`の両方に使う
  （IVの第一段階が二重に計算される問題を最初から避ける）。
- `first_stage()`は既存の`FEResults`を返す（`f_statistic`等は`None`）。結果を作るとき、**固定効果の辞書をその場で計算して`LazyFixedEffects::Computed`にする**
  （`FeEstimator`を保持しない。辞書は個体数に比例する小さなもの）。残るのは内生変数ごとの残差（8nバイト）とスカラー・係数類だけで、第一段階側でも
  `fixed_effects()`が使える。`FeEstimator`から`FEResult`を作る変換は`engine_pybind`内の`pub(crate)`関数で、Pythonから見える面は増えない。
- `fixed_effects()`は`α_i = ȳ_i − x̄_i'β̂`（`β̂`は構造の係数、`X`は実際の内生変数）。`fixest`の`fixef`は`mean(y − Xβ̂)`と中心化の差を除いて一致する。
  基準化はFEと同じ（連結成分ごと）。

### 2.7 pickle

`source_data`（使った列だけ）を状態に含め、`rebuilt_from`で作り直せる部分は状態に入れない。第一段階の`FEResult`は`Computed`なのでそのままpickleできる。
`FEIVResults`は`DataFrame`とオプションを保持する（診断・`fixed_effects()`・`c_stat()`が再推定する）。

## 3. engine

### 3.1 入力型

`PanelIndex`は`Vec<GroupCodes>`＋効果の列名のK次元。DKの時点キーは効果の次元とは別に持つ（`dk_time`は`effects`と独立）。
`FeIvInput { y, dep_var_name, x_exog, x_endog, instruments, index }`は役割ごとの列と名前を`RoleColumns`にまとめて渡す。列は`faer::Mat`ではなく
列スライス（`WithinTransform`が列単位のため）。`x_exog`は空でもよい。`x_endog`・`instruments`は非空（`ROLE_EMPTY`）。

効果の種類は`FeEffects`（`OneWay`・`TwoWay`・`MultiWay`）。1-wayと均衡な2-wayは閉形式、それ以外（不均衡な2-way・3列以上）は交互射影。
`effects`はentity/timeの意味を持たず、同じ組合せの複数行を許す（バランスの判定は「`n == N×T`かつ重複なし」）。

### 3.2 `FeTwoSlsEstimator::fit`

1. `ensure_serial()`、`InsufficientInstruments`（`len(instruments) < len(x_endog)`）。
2. singleton検出（`ensure_no_singleton_groups`。FEの事前検証と同じ関数。FEIV側で複製しない）。観測が1つしかないグループはエラーのまま（欠測値を自動除外しない方針と揃える。除去すると行の対応（`residuals`の長さ・クラスターとDKのキー・`fixed_effects()`）が入力とずれ、役割列が4種あるFEIVでは全列で同じ行を落とす仕組みも要る）。
3. 全列（`y`・`x_exog`・`x_endog`・`instruments`）を**1回だけ**within変換する（`prepare_within`。FE-2SLSとFE-GMMで共有。`noise`・グループ索引・連結成分も共有）。
   反復法のため変換が高コストで、第一段階を内生変数ごとに回すと変換が繰り返されるので、変換済みの列を渡す内部経路を最初から設ける。
4. 変換後の分散ゼロ検出（全役割）、`y`の個体内定数、自由度の検証（3.3）。
5. 第一段階: 内生変数ごとに`FeEstimator`の後半で回す（`y=x_endog[j]`、`x=x_exog ++ instruments`、同じ`effects`・`cov_type`）。失敗は`stage=FIRST_STAGE`付き
   （`PanelError::InStage`）。`X̂ = [x̃_exog, x̂̃_endog]`（内生列は第一段階の当てはめ値）。
6. 点推定`β̂ = (X̂'X̂)⁻¹X̂'ỹ`（共有の`two_sls_point_estimate`）、構造残差`e = ỹ − X̃β̂`（`X̃`は実際の内生変数）。完全フィット検査は構造残差で行う
   （`SecondStageFailed`の`stage=SECOND_STAGE`）。
7. `panel_cov_params`に`X̂`と構造残差を渡す。`inference_table`（t分布）、`wald_f_test`。

**第一段階は`iv::two_sls`のコアではなく`FeEstimator`で推定する**。理由: `TwoSlsEstimator`・`compute_first_stage`は自由度が`n − k`固定で、吸収した固定効果の数を
渡せない。`FeEstimator`なら自由度・`cov_type`・F統計量が自動的にFE準拠になり、`first_stage()`も`FEResults`で返せる。

第一段階の共分散はIV本体と揃える: 標準誤差が計算できないときはエラー、それ以外は第一段階のFだけを`None`にする。FEの`panel_cov_params`は
クラスター数・時点数が検定する傾きの数以下だとWald Fのために弾くので、`PanelCovInput::tests_slopes`（`false`で`G > q`・`T > q`の検査を省き`f_statistic`を`None`にする）で切り替える。
標準誤差に関わる検査（`G >= 2`、2グループ退化、完全適合）は残す。構造式は従来どおり`G <= k`でエラー。

### 3.3 自由度と`n_absorbed`

`neffects`（`n_absorbed`）は`FeEstimator`と同じ`Σn_d − (K−1)`（連結成分の数やネストによるランク落ちは引かない。`fixest`流）。検証は構造式（`n > k + neffects`）が先、
第一段階（`n > l + neffects`、`l = k_exog + k_instruments`）が後（IVの`validate_observation_counts`と同じ層分け）。

共有関数のシグネチャ（単一の`n_absorbed`）は変えず、診断ごとに渡す数を呼び出し側が決める:

| 診断・`cov_type` | `n_absorbed` |
|---|---|
| 古典の部分F・CD・`Classical`・HC系・DK | `neffects`（全部の効果。DKは`fixest`の`K.fixef="full"`） |
| `Cluster` | `fe_cluster_k_correction`の`K − l`（入れ子の次元を除く） |

`lfe`・`ivDiag`は連結成分が複数のパネルで本実装・`fixest`と1ずれる（`lfe`は階数を正確に数える）。入れ子の第3の効果では3者とも真のランクより多く数える。
この差は公開ドキュメントに書く。クラスターの`K`の規則（`fixest`の`ssc(fixef.K="nested")`）はFEと同じで、効果の列そのものがクラスターのときの扱いは
`fe-spec.md`を参照。

### 3.4 診断（FEIV）

- **Sargan**: 構造残差を`Z̃ = [x̃_exog, z̃]`に回帰した`n·R²`相当。**`σ̂² = e'e/(n − n_absorbed)`**（`fixest`の`sargan`は`e'e/n`で、小`T`で過剰に棄却する。
  実測の棄却率（名目5%、`N=100`）は`e'e/n`で`T=3`が10.5%・`T=6`が7.3%・`T=20`が5.2%、`e'e/(n−N)`で4.8%・5.1%・4.7%。within残差の二乗和の期待値が`σ²(n−N)`なので、
  `n`で割ると統計量が`T/(T−1)`倍になる）。自由度は`len(instruments) − len(x_endog)`で、丁度識別と内生変数なしは`None`（IV本体は内生変数なしで`df = len(instruments)`だが、
  FEIVは第一段階が無い構成を対象にしない）。等分散前提で`cov_type`に依らない。`fixest`との差は文書で説明する。
- **Wu-Hausman**: 補助FE回帰`ỹ ~ [x̃_exog, x̃_endog, v̂]`（`v̂`は第一段階の残差）を`cov_type`と同じ共分散で回し、`v̂`列のWald F。分母自由度は補助回帰のもの
  （Clusterは`G−1`、DKは`T−1`、それ以外は`n − k − k_endog − neffects`）。`fixest`の`wh`はiid前提なので、classical以外は一致しない（参照の作り方は6.2）。
  補助回帰は`tests_slopes=false`なので、`G`が`k+1`〜`k+k_endog`でも`v̂`の`k_endog × k_endog`部分行列で検定でき値を持つ（IV本体との違い）。
  `None`になる条件: 内生変数なし・補助回帰の自由度不足・`v̂`の分散ゼロ・完全適合・特異。それ以外の失敗は`stage=WU_HAUSMAN_REGRESSION`つきで返す。
  構造式の共分散・F検定の後に呼ぶ（構造式のエラーが補助回帰のエラーに隠れないように）。
- **弱操作変数**: 部分Fは`fit()`時のプロパティ。**`cov_type`によらず等分散前提**で、分母自由度は`n − l − n_absorbed`（ロバスト版はKP・MOP）。CD・KP・有効Fは
  推定後のメソッドで、行列を受け取る共有関数（`shared/weak_instruments.rs`・`kleibergen_paap.rs`・`effective_f.rs`・`score_covariance.rs`）を第一段階の設計行列から取り出した行列で呼ぶ。
  - `ScoreCovariance`に`Dk{time, bandwidth}`（時点ごとのスコアの和にBartlett核）と`HcAbsorbed{variant, leverage}`（`n_absorbed > 0`のHC2/HC3。吸収した効果のレバレッジを含む）を持たせる。
    DKのKP・有効Fは実装する。HC2/HC3の吸収レバレッジは`leverage_full(0, levels)`（within変換後の素のレバレッジを0にしたダミー分）。
  - **DKで`T ≤ d`（`d = 除外操作変数の数 − 内生変数の数 + 1`）のときrk LMは自明な値になる**（時点ごとのスコアが`d`次元を張るのに足りず、`T = d`・`bandwidth = 0`でLMがちょうど`T`になる）。
    KPのLMで`Dk`かつ`T ≤ d`は未定義にした。rk Wald Fも未定義になり（スコアの和が0でランク落ち）、`kleibergen_paap()`・`effective_f()`は`ComputationError`になる。
    有効Fのスコアは`2q`次元だが使うのは`q × q`の`Ŵ2`だけで、ランク落ちは既存の閾値が未定義にするので、同じ検査は要らない。
  - 有効Fは`Ŵ2`のランク落ち（最小固有値が最大の`1e-10`倍以下）を未定義にする（KPに揃える）。内生変数が2個以上は未定義（原論文は1個）。
  - Stock-Yogoの臨界値は表をそのまま使う。吸収自由度を統計量の自由度に反映することが条件（反映しないと、10%基準で臨界値を超える確率が`T=2`で77.5%・`T=3`で45.6%・`T=6`で19.2%に膨らむ。
    反映すれば`T=2〜20`で依らず、横断面IVとほぼ同じ）。検証した範囲は内生変数1個・等分散・正規・`K2=3`。

### 3.5 FE-GMM

`FeGmmEstimator`は`FeTwoSlsEstimator`とは別の型（`fe_iv/two_sls/`と`fe_iv/gmm/`。どちらも`fe_iv/common/`の入力型と前段を共有する）。`fit(input, effects, gmm_type: FeGmmType, raise_on_non_convergence, cov_type, confidence_level)`／`fit_with_absorption`。
前段（入力検査・全列のwithin変換・自由度の検証・変換後の分散ゼロの検査）は`prepare_within`でFE-2SLSと共有する。

- **GMMの反復コア・重み行列は`shared/`へ**（`shared/gmm_iteration.rs`・`gmm_weight.rs`・`gmm.rs`・`gmm_c_statistic.rs`。`iv::gmm`も同じコアを呼ぶ）。`GmmEstimator`が自由度`n − k`固定で吸収した効果の数を渡せないため、
  `gmm_iterate`・`sandwich_covariance`を行列受け取り・自由度引数付きに改めて共有する（CUE・Windmeijer補正を二重に書かない）。返すエラーは`IvError`でなく`CommonError`
  （`code`・文面・`params`は不変）。FE-GMMは自前の`Ω̂`（FEのクラスター補正・DK・FEのレバレッジ）で`gmm_sandwich_covariance`を呼ぶ。
- 重みの型は`FeWeightType`（`Classical`・`Robust`・`Cluster{groups}`）、方式は`FeGmmType`。`iv::gmm`の`WeightType`とは別にした（`Hac`を型で表せない。`INTERNAL_ERROR`で弾くより、到達できない型にする）。
  重みの既定（`effects[0]`でのクラスター）はバインディングの責務で、engineは`Cluster{groups}`を明示で受ける。
- 点推定: `β̂(W) = (X̃'Z̃ W Z̃'X̃)⁻¹X̃'Z̃ W Z̃'ỹ`。`Z̃ = [x̃_exog, z̃]`（`l = k_exog + k_instruments`列）。構造残差の完全適合は段階なしの`CommonError::PerfectFit`（GMMに第二段階は無い）。
- 反復GMMはダミー入りとwithin変換で3回目の推定から一致しない。FE-GMMの反復は**within残差で重みを作る**定義（ダミーのモーメントは`α`で厳密に0にできるので2段階目はwithin変換と一致するが、
  2段階目の最適解は`D'ê ≠ 0`になり、次の重みを作る残差がwithin残差と変わる）。`Classical`重みの反復は2回目で収束し`n_iter=2`。
- **共分散**: 重みと独立のサンドイッチ一般形（`panel_moment_covariance`）。`Ω̂`の小標本補正はFE-2SLSと同じ`fixest`流。HC2・HC3のレバレッジは`Z̃`の素のレバレッジ＋吸収した効果のレバレッジ
  （ダミーを操作変数に入れたときのハット行列の対角と一致）。`cov_params`は丸めの非対称を対称化する。`Wald χ²`の失敗は`Stage::OverallWaldTest`・`PanelError::WaldTestFailed`
  （FE・RE・FE-2SLSのF検定の`FTestFailed`・`OVERALL_F_TEST`とは別の段階）。
- **Hansen J**: `fit()`時に計算してプロパティ。`S`は点推定に使ったもの。**`Classical`・`OneStep`のときだけ`S = σ̂²Z̃'Z̃`の`σ̂²`の分母を`n − neffects`にする**（`S`を`n/(n−neffects)`倍。
  FE-2SLSのSarganと同じ理由で、`Classical`のJはSarganに一致する）。`Robust`・`Cluster`は補正しない。理由: 参照実装に無く、`Cluster`は効果がクラスターに入れ子なら
  `Σ_t z̃ẽ = Σ_t z̃ε`で補正が要らず（効果がクラスターを横切るときは近似、個体を横切るクラスター（時点）では成り立たない）、`Robust`はwithin残差の分散が`σ²(1−1/T)`に縮むので
  小`T`で過剰棄却するが、`n/(n−neffects)`倍では不均一分散で直らず原理的に正しい補正でもない。自由度は`k_instruments − k_endog`で、丁度識別・内生変数なしは`None`。
- **`c_stat`**: 行列を受け取る核`gmm_c_statistic(&CStatisticMatrices, &CStatisticTarget, &CStatisticSettings)`を`shared`に置き、IV本体（`IvInput`を渡す薄い層）とFE-GMMが呼ぶ。
  `Z_full = [x̃_exog, x̃_{2B}, z̃]`と`Z_c`（検定する列を外す）で、`Z_full`の2段階GMMの`S_e`・`J_e`、`S_e`の部分行列`S_c`を重みにした`Z_c`の`J_c`、`C = J_e − J_c`を`χ²`で検定する。
  重みは`fit()`の`gmm_type`・`cov_type`に依存せず、`two_step`・`iterated`は`gmm_weight_type`、`one_step`は`Classical`。`S`の尺度の規則はHansen Jと同じ
  （`Classical`だけ`S_e`を`n/(n−neffects)`倍し、`C`は`(n−neffects)/n`倍になる）。`J_e`は同じ重みの2段階`fit`のHansen Jと一致する（除外操作変数だけを検定するとき）。
  公開の形は事後診断で、呼び出し時に`prepare_within`で変換し直す（設計行列を保持しない）。クラスター数が`Z_full`の列数に足りないと`CStatisticUndefined`（`ComputationError`）。
  `fit()`の`GMM_WEIGHT_CLUSTER_COUNT_TOO_SMALL`（`ValidationError`、過剰識別は`G < l`、丁度識別は`G ≤ l`）とはクラスが違う。検査の順は`fit`と同じ
  （`prepare_within`が先、`target`・`weight`の検査が後）。

### 3.6 エラー

`PanelError`にバリアントを足し、既存のカタログの`code`を再利用する（同じ`code`は同じ`hint`）。第一段階の失敗は原因の`code`に`stage=FIRST_STAGE`を付ける。新しい`code`は
`rust-style.md`のエラー追加手順に従う。

**定数列のエラーは2分類**: 生データで定数（`REGRESSOR_CONSTANT`・`OUTCOME_CONSTANT`）と、固定効果を除くと一定（`REGRESSOR_NO_WITHIN_VARIATION`・`OUTCOME_CONSTANT_WITHIN_ENTITY`）。
説明変数側の2つに`role`パラメータ（`x`（FE）・`x_exog`・`x_endog`・`instruments`）を持たせ、文面とhintを`{role}`で書く。個体内で一定の操作変数（時間不変）が典型的な誤りなので、
hintで指摘する。変換後の分散ゼロ検出は全役割に対しこの2分類で報告する。

GMMに第一段階は無いので、共線な外生変数・操作変数は`MATRIX_NEAR_SINGULAR`（`DESIGN_MATRIX_SINGULAR`＋`stage`ではない）。エラーの`model`は`"FEIV"`（推定量によらない。IVの`"IV"`と同じ）。
`C_STATISTIC_UNKNOWN_VARIABLE`・`C_STATISTIC_UNDEFINED`のhintはIV・FEIVで通じる中立な文面にしてある。

`test_validation_order`での検査順（FEIV。`feiv`・`feiv_gmm`）: `effects`の列数 → `x_endog`の空 → 役割の重複 → 内の重複 → `dk_time`の役割。`INSUFFICIENT_INSTRUMENTS`はエンジンの`fit`の中
（列の抽出の後）で報告されるので、順位は固定していない。

## 4. engine_pybind

- `FEIVOptions`・`FEIVResult`、`fit_fe_iv`（`estimator`で2SLSとGMMに振り分ける）。`PanelModel::FeIv`、`validate_fe_iv_roles`、`FeIvColumns`。
  `FEResult`は`Clone`（`LazyFixedEffects`が`FeEstimator`を`Arc`で共有）。`fe_result_fields`（借用で`FEResult`を作る）・`fixed_effects_dict`・`resolve_fe_cov_type`を`panel/fe.rs`から切り出して共有する。
- IVの診断の出力型（`diagnostics.rs`の`from_engine`）・`c_stat`の引数検査と結果の組み立て（`iv::common::diagnostics`の`resolve_c_stat_target`・`CStatisticOutput::from_engine`）はFEIVが再利用する。
- `estimator="gmm"`の`c_stat`は`fit`の重み（`one_step`は`Classical`）を`FeWeightType`に組み立て、検定する変数の名前を`x_endog`・`instruments`の位置に写す。空の`target`の検査はバインディングとengineの両方にある（IVと同じ）。

## 5. 公開ドキュメント

`docs/usage`・`docs/api`・`docs/methods`に各1ページ（navはPanel models配下）。`methods`は差分のみ（FE・IVのDefinitionsへリンクし、固有の式だけ書く）。`api/fe_iv.md`は`CraggDonaldResult`等を
再描画せずIVのページへリンクする（再描画するとautorefsの重複警告が出る）。公開時に書くこと: `Robust`重みの`J`・`C`は小`T`で過剰棄却すること、GMMのHC2・HC3は他の実装と比較していないこと、
クラスターの`K`が`fixest`流であること、`fit`と`c_stat`のクラスター数不足のクラス差、`n_absorbed`の診断別の違い、DKの`T ≤ d`。

## 6. テスト

### 6.1 構成

`tests/panel/test_fe_iv_{api,binding,reference,dummy_crosscheck,plm_crosscheck,weak_diagnostics_crosscheck,validation}.py`、
`test_fe_iv_gmm_{api,binding,reference,crosscheck,c_statistic,reduction,validation}.py`、`tests/support/_fe_iv_helpers.py`・`_fe_iv_gmm_helpers.py`、許容誤差は`_tolerances_panel.py`のキー。
`test_*_binding.py`は`_lib`を直接見る（ダミー変数を明示したnumpy実装との照合・配線）、`test_*_validation.py`は入力検証。engineの単体テストは`engine/src/panel/fe_iv/`の`mod tests`
（`fe_iv/fit/tests/`の`lsdv`・`sargan`・`wu_hausman`・`weak_instruments`、`gmm/`）。

engineの単体テストの軸（FE-GMM）: FE-2SLSへの還元、ダミー変数を明示した`iv::gmm`との照合（FWL）、手計算（素朴な平均除去とダミーへの回帰の残差。1-wayと2-way・不均衡・3次元）、検出力。
変異（HC2・HC3のダミーのレバレッジを外す、再スケールを外す・全重みに適用する、`S_c`の行の取り違え等）で失敗することを確認してある。

横断テストへの登録先（足し忘れ注意）: `test_call_signatures`・`test_gil_release`（`FITS`・`ROWS`・`CASES`。メソッドは自動検出されないので明示）・`test_results_repr`・
`test_keyword_only_arguments`・`test_validation_order`・`test_effects_argument`・`test_argument_types`・`test_data_input_types`・`test_numeric_column_values`・`test_key_columns`・
`test_cov_params`・`test_stat_dist`・`test_pickle`・`test_option_types`・`test_options_repr`・`test_option_value_echo`・`test_results_docstrings`（「Fields by purpose」）・
`test_doc_examples`・`test_docstring_examples`・`test_public_docs`、`tests/support/_errors.py`の`MIGRATED_CODES`。

### 6.2 リファレンス

**FE-2SLS**: 主は`fixest`（`feols`に操作変数。係数・classical/hc1/cluster/dkの標準誤差・Wald F・Wu-Hausman・Sargan・部分F・rk Wald F）。独立クロスチェックは
`plm(y ~ x | x2 + z, model = "within")`（1-wayの標準誤差）、`linearmodels.IV2SLS`にダミーを明示した2SLS（LSDV。係数・古典/HC1の標準誤差・Sargan・Wu-Hausman・部分F・CD・KP・R²。
`PanelOLS._rsquared(params)`でR²）、`ivDiag`（1-wayのclassical・hc1・クラスターと2-wayのclassical・hc1のCD・rk Wald F・有効F）。ダミー変数2SLSの有効Fは
`benchmark/iv/references/mop_oracle.py`にダミーを外生変数として渡す。

**FE-GMM**: 主は`linearmodels.IVGMM`をwithin変換後のデータ（切片なし）に適用（係数・重み・Hansen J・反復回数）。標準誤差は漸近共分散×小標本補正のスカラー（手書き）。
独立クロスチェックは`momentfit`（baseline系の全重み・全`gmm_type`とC統計量）。`fixest`・`plm`・`PanelOLS`にGMMは無い（`plm`には動的パネル用の`pgmm`のみ）。自前の部分は2つで、
(1)within変換（群の平均、または2列以上でダミー変数行列への`numpy.linalg.lstsq`。本体の交互射影とは別の方法）、(2)標準誤差の小標本補正のスカラー倍。(2)は還元で独立に押さえる:
`one_step`と`two_step`の`classical`重みは過剰識別でもFE-2SLSと同じ係数・標準誤差・共分散を返し（サンドイッチは重みの定数倍に依らない）、Hansen JはFE-2SLSのSarganになる。
還元をFE-2SLSの全ケースに広げ、`fixest`・`plm`で小標本補正・`K`・Jの分子を確かめる。未確認で残るのは、過剰識別のrobust・cluster重みとの組合せ。

**`fixest`の落とし穴（参照スクリプトの設定）**:
- `cd`は本実装と一致しない（baselineで206.9対172.7。原因は追っていない）。`kpr`はiid、または内生変数と操作変数が同数のときだけ実装され、rk LM・rk Wald Fのどちらにも一致しないので使わない。
- `wr2`は第二段階の残差ベースでR²の定義が違う。係数は内生変数が先頭（`fit_endog1`）なので`x_exog`・`x_endog`の順に直す。
- `wh`はiid前提。classical以外は、第一段階の残差`v̂`を足した補助FE回帰を同じ`vcov`で回し`v̂`のWaldを取る。
- 不均衡パネルでは`fixef.tol`を厳しくする（既定の`1e-6`だと参照値が約5e-9ずれる。下限は`2.2e-12`）。
- `wald()`は自由度1の境界で分母自由度を`max(df2, df1+1)`に切り上げるので、部分Fの分母自由度は第一段階の残差自由度を直接使う（統計量は同じ）。
- 内生変数が1個のとき、rk Wald Fは同じ`vcov`での第一段階の頑健Wald Fに等しい。`fixest`の第一段階（`wald(第一段階, keep=操作変数, vcov=vc)`）を参照にすると、DK・2-way・3-way・不均衡も含めて照合できる。

**`plm`**: `vcovHC(white1, HC0)`×`n/(n−k−G)`がHC1、`vcovHC(arellano, HC0)`×`G/(G−1)·(n−1)/(n−k−1)`が個体クラスター、`vcovSCC(HC0, maxlag)`×`T/(T−1)·(n−1)/(n−k−G)`がDKに一致する
（構造残差と`X̂`のサンドイッチ。第二段階の残差を使うと一致しない）。`plm`のHC2・HC3はレバレッジに固定効果の分を含まず本実装と定義が違うので、参照にしない。
HC2・HC3は手計算のダミー変数2SLS（numpy）だけが参照（`verification-details.md`の単一リファレンスの例外に載せる）。

**`linearmodels.IV2SLS`にダミーを入れた照合**: クラスターはダミーを全て`K`に数えるので、一致するのは**1-wayの個体FE × 時点クラスター**だけ。hc1のrk LMはlinearmodelsがHC0基準なので比べない。
`PanelOLS._rsquared(params)`の返り値の順は`(overall, within, between)`。2-wayのwithinはlinearmodelsが個体のみの別定義なので、ダミー射影の手計算にする。3-wayは`PanelOLS`が扱えずR²の参照なし。

**単一リファレンスの統計量**（`verification-details.md`に載せる）: HC2・HC3、DKのKP・有効F、HC2・HC3のKP・有効F、複数内生変数のKP・有効F、R²・`df_*`、GMMのHC2・HC3。

### 6.3 シナリオ

FEIV: 13シナリオに、小標本・境界・悪条件を足した20シナリオ（凍結CSV。丁度識別・過剰識別、内生変数1〜3個、`x_exog`が空、弱い操作変数、無効な操作変数（Sargan棄却）、内生性なし、時点数、
不均一分散・自己相関、極端な標準偏差、短い・長いパネル、自由度1の境界（4個体×2時点）、不均衡、3列の固定効果、相関0.9・0.995の外生変数、スケール差、多数の外生変数、外れ値のある外生変数）。
DKは時点数が傾き係数の数を超えるケースだけ作る。時点ラベルはゼロ埋め（DKの時点列は文字列のバイト順）。外生変数を複数列にする追加の列・外れ値は別の乱数生成器
（`default_rng([seed, 7])`）から作り、既存シナリオの凍結CSVを変えない。

`estimator="gmm"`: FEIVの20シナリオ（凍結CSVを流用）の既定の推定（`two_step`・`cluster`重み）に加え、baseline・多内生変数・不均衡・2-way・不均衡な2-way・3-wayの6構造で他の重み・`gmm_type`
（iteratedはrobust・cluster重みだけ。classical重みの反復は2SLSと同じ）。実データは含めない。

**許容誤差**: FEIV `1e-8`（絶対`1e-10`）が基本。実測（FE-2SLS）: fixestに係数4e-15・標準誤差1e-12・Sargan 3e-12・rk Wald F 4e-11、ダミー変数2SLSに係数7e-15・有効F（MOP）5e-11、
ivDiagにrk Wald F 1.1e-9・有効F 5e-10。`estimator="gmm"`: `IVGMM`に係数8e-14・標準誤差9e-13・共分散2e-12・信頼区間3e-10（悪条件のHC3）・Wald/J 5e-13、momentfitに係数7e-15。
`df1_boundary`（4個体×2時点）のHC3は`1/(1−h)²`が丸め誤差を増幅してWaldが2e-7ずれるので、HC2・HC3の追加シナリオから外す（標準誤差は3e-11）。
fixestの`sargan`と本実装は`(n − n_abs)/n`倍の差があるので、参照値に同じ倍率を掛けて比べる。

## 7. 性能

比較パッケージは置かない（engine単独。`PerfAdapter.libraries=("engine",)`）。理由: `linearmodels`にFEIVは無く、pandasのgroupby demean＋`IV2SLS`は計測の大部分をgroupbyが占め、不均衡2-wayや高次元で
参照側に交互射影を手書きする必要も出て恣意的になる。`performance/compare_fe_iv.py`の1本で、GMMはmethod軸（`extra_methods`）で測る。

- `k`は傾き係数の総数（`x_exog = k − 内生変数の数`。FEの`k`と揃い、`T=6`固定でcov軸のDKが`T > k`を満たす。IVの`k = x_exog`とは定義が違う）。Tを6に固定してNを掃引。
- 構造と識別の数は`extra_methods`の代表点（`two_way`・`two_way_unbalanced`・`three_way`・`endog3`（内生3・操作変数5）・`inst6`（内生1・操作変数6）、`n=1,000,000`・`k=5`・classical）。
- cov軸はclassical・hc1・hc2・hc3・cluster・dk。within変換は単独で計測せず、列数を変える軸と交互射影が必要な構造の比から間接的に見る（変換だけを取り出す公開APIは足さない）。
- 計測時間には`fit()`が一括計算する診断（Sargan・Wu-Hausman・弱操作変数の部分F）が含まれる。数値は`benchmark_performance.yml`のCI計測値を正とし、リリース準備時に公開ページへ転記する。
- データは`generate_fe_iv_perf_dataset`（規模を指定できる均衡パネル。凍結しない）。
