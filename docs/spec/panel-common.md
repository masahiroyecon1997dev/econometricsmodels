# パネルモデル（FE/RE）共通仕様

FE（固定効果）/RE（変量効果）が共有する基盤の確定済み仕様。手法固有の内容（within変換・
Swamy-Arora分散成分推定の具体式、固定効果の復元等）は[`fe-spec.md`](./fe-spec.md) /
[`re-spec.md`](./re-spec.md)を参照し、本ドキュメントには2手法が共有する設計判断のみを
記載する。

## 1. 引数設計

### 1.1 `y` / `x` / `effects` のシグネチャ

- `fit_fe(data, y, x, effects, options)` / `fit_re(data, y, x, effects, options)`
- `y: str`、`x: list[str]`はOLSと同じ。
- `effects: list[str]`（固定効果・変量効果の次元を指す列名）は**独立の必須引数**とする。FE/REいずれも
  パネル構造が無ければモデルとして成立しないため、`y`/`x`と同格に扱う。1〜2要素で、1要素=one-way、
  2要素=two-way（`effects[0]`が個体、`effects[1]`が時点。two-wayはバランスパネル必須）。
  空は`ROLE_EMPTY`、3要素以上は`EFFECTS_TOO_MANY`（`ValidationError`、params: `model`・`n_columns`・
  `max_columns`）。上限は当面FE/REとも2で、文面は上限だけを示す中立の形にしている（FEは将来上限を
  上げる予定、REは交差ランダム効果が標準的な推定量でないため上限が変わらない見込みと理由は違うが、
  同じcodeは同じhintという規則のため）。
- `FEOptions.time`・`REOptions.time`は廃止した。`dk_time`（Driscoll-Kraay型HACの時点列、`str | None`）は
  `effects`とは独立のまま`Options`に置く。**`cov_type="dk"`では`dk_time`が必須で、`effects`の列を
  借用しない**（未指定は`ValidationError`、one-way・two-wayとも同じ）。どの列がDKの時点かを利用者が
  明示しない設計は、意図と違う列が選ばれても気づけないため。`dk_time`は2-wayでも常にDK計算に使われ、
  「2-wayの固定効果構造に使う時点粒度」と「DK HACカーネルに使う時系列粒度」が異なるケースにも
  対応する。engine側は`FeCovType::Dk { bandwidth, time: TimeKeys }`（`time`は必須）として実装
  （`engine/src/panel/fe.rs`モジュールdoc「Driscoll-Kraay型パネルHAC対応」参照）。REも同じ規約
  （`ReInput.time`が2-wayの時間次元、DKの時点列は`ReCovType::Dk { time }`で別に渡す）。
- **`effects`にまとめた理由（個体と時点を1つの必須引数にする）**（全手法共通の基準は
  [`api-conventions.md`](./api-conventions.md)）:
  - 基準は「必須の列は引数、任意または条件付き必須の設定は`Options`」の1本。個体と時点は
    「効果の次元」として同じ役割なので、1つの必須引数にまとめた。`entity`が引数で`time`が`Options`と
    いう非対称は利用者に分かりにくく、`time`はone-wayかtwo-wayかという推定量そのものを変えるのに、
    推論の設定と同じ`Options`に置かれていた。`dk_time`との混同も避けられる。
  - 3次元以上の固定効果（高次元FE）は別APIにせず、同じ`effects`の上限を上げるだけで足りる。
    以前の「必須の`entity`以外の次元は`Options`側に並べる想定」とは逆の判断になった。
  - 先頭要素がデフォルトのクラスターキー（fixestの慣習と同じ。3.2節）。REは2要素の順序に意味があり
    （`effects[0]`が個体: `sigma2_u`・`theta_entity`、`effects[1]`が時点: `sigma2_lambda`・
    `theta_time`）、バランスパネルでは入れ替えてもモデルが対称なので検証では検出できず、
    ドキュメントとテストで保証する。FEは入れ替えても係数・SEは同じだが、デフォルトのクラスターキー、
    `fixed_effects()`のキー順、時点側の基準期間が変わる。
  - 列の型の規則は位置で決まる: `effects[0]`は従来の`entity`と同じ規則（同一性だけのキー）、
    `effects[1]`は従来の`time`と同じ規則（時点のキー）。推定値と受け付ける入力を変えないため。
    規則の統一は3次元以上を扱うときに検討する。
  - エラーの`argument`は`"effects"`に統一し、どの列かは`column`で示す。`SINGLETON_GROUP`のparamsは
    `{argument, column, group}`（`column`は`effects`の該当要素の列名。エンジンは`FeInput`/`ReInput`の
    `with_effect_names`で列名を受け取り、計算には使わない）。
- **命名規則**: 列名を受け取る引数は`effects`/`cluster`/`dk_time`のいずれも**`_col`サフィックスを
  付けない**（`y`/`x`/`weight`と同じ規約）。`dk_time`はDriscoll-Kraay型HACの時系列順序で、
  `cov_type`の値`"dk"`を接頭辞にした名前（OLS/WLS/IVの`hac_time`と対応）。複数列を取る`effects`は
  `x`と同じく`list[str]`の複数形。
- FE/RE間で`effects`の命名・意味は統一する。IV側の対応する引数は性質が異なるため
  [`iv-spec.md`](./iv-spec.md)を参照。

### 1.2 モデル固有オプションの置き場所

`FEOptions`/`REOptions`という別々の`#[pyclass]`構造体に含める（`OLSOptions`/`LogitOptions`の
前例を踏襲）。共有可能なフィールド（`effects`/`dk_time`等）を内部実装上どこまで共通化するか
（トレイト・共通struct等）はAPI設計とは別問題として扱う。

### 1.3 `weights` / `offset` の扱い

- `offset`: 線形モデル（link functionを前提としない）のため**該当なし**。今後も追加予定なし。
- `weights`: 汎用の頻度/分析重みオプションとしては**見送り**（Logit/Probit/Tobitにおける判断を
  踏襲、Phase6で再検討）。
  - **注**: この判断はGLS（FE/RE/IVの後に着手予定）の引数設計を拘束しない。GLSは重み・共分散
    構造の指定自体がモデルの本質であり、WLSの`weight: str`（必須の独立引数）に近い位置づけに
    なる見込み。GLS着手時に別途設計する。

## 2. 結果（Return）設計

### 2.1 共通コア項目

OLS（`OLSResult`、`engine_pybind/src/linear/ols.rs:137-191`）の項目を土台にするが、以下の点で
機械的な流用ではなく調整する。

| フィールド | 由来 | 備考 |
|---|---|---|
| `params` / `std_errors` / `test_stats` / `p_values` / `conf_lower` / `conf_upper` / `param_names` | OLS共通 | 検定分布はt分布で統一（3.3節） |
| `residuals` / `dep_var_name` | OLS共通 | そのまま踏襲 |
| `n_obs` | Logit由来の表記 | OLS/WLSも`n_obs`に統一済み |
| `df_resid` / `df_model` | Logit由来、パネル向けに新規追加 | OLSには無いが、FEは自由度調整が`n - 個体数 - k`という非自明な式になるため明示的に返す価値が高い |
| `n_groups` | 新規追加（FE/RE限定） | `dict[str, int]`（`effects`の列名→グループ数、`effects`の順）。次元ごとの件数は、次元が増えたり形が変わる（高次元FE・不均衡パネル）ときに構造が変わる標本の構造に属するので、固定の`n_entities`/`n_periods`ではなく列名キーにした |
| `sigma2_eps` / `sigma2_u` / `sigma2_lambda` / `theta` / `theta_entity` / `theta_time` / `theta_total` | 新規追加（REのみ） | 分散成分（σ_ε²・σ_u²、2-wayのσ_λ²）と準偏差変換の重み（1-wayは個体ごとのθ、2-wayは3つの重み）。詳細は`re-spec.md`2章 |
| `cov_type` | OLS共通 | サポート対象は3.1節 |
| `f_statistic` / `f_p_value` | OLS共通 | そのまま踏襲（ただしengine側の実装はOLSの単純な流用ではない。傾き係数`k`個の同時Wald検定をFE/RE独自の`cov_type`別`cov_params`・パネル自由度調整済み`df_resid`で行う） |
| `log_likelihood` / `aic` / `bic` | OLS共通 | FE/REは最小二乗族で正規性下の尤度が定義できるため含める |
| `r_squared_within` / `r_squared_between` / `r_squared_overall` | 新規追加（OLSの`r_squared`/`adj_r_squared`を置き換え） | 詳細は2.3節 |

**列名キーにしたもの・しなかったもの**: 次元ごとの件数（`n_groups`）と固定効果の水準（FEの`fixed_effects()`は
one-wayでも常に`{列名: {水準: 効果}}`の入れ子）は、次元が増えたり形が変わる（高次元FE・不均衡パネル）ときに
構造が変わる標本の構造に属するので列名キーにした。一方、**`sigma2_u`・`sigma2_lambda`・`theta`・`theta_entity`・
`theta_time`・`theta_total`は意図的に現状維持**する。分散成分はモデルのパラメータで、バランスの有無によらず
意味が変わらない。`theta_*`はバランスパネルの閉形式の3つの係数に紐づいたスカラーで、不均衡な2-way REでは
この構造自体が成り立たない見込みのため、列名キーにしても備えにならず、1-wayの`theta`（個体の水準がキーの
dict）と名前が衝突する。不均衡な2-way RE対応時に形が決まってから再検討する。

**pickle**: `FEOptions`/`REOptions`のフィールド集合が変わり、`FEResult`/`REResult`の状態キー
（`n_entities`・`n_periods`が`effects`・`group_counts`に）も変わるため、旧版のpickleは読めない
（キー集合の検査で検出される。`FORMAT_VERSION`は上げていない）。

### 2.2 OLSの`nobs`→`n_obs`リネーム

既存`OLSResult.nobs`（`ols.rs:157`）とLogitの`LogitResult.n_obs`（`logit.rs:206`）で表記が
不一致だったため、`n_obs`へ統一した（OLS・WLS双方のResult型・Python側プロパティを
リネーム済み）。

### 2.3 R²の種類（within/between/overall）

- `r_squared_within` / `r_squared_between` / `r_squared_overall`の3フィールドを`fit()`の
  戻り値に含める。
- **bareの`r_squared`/`adj_r_squared`は廃止する**（OLSの`r_squared`をそのまま流用しない）。
  パネルモデルでは「どのR²か」が一意に決まらないため、曖昧な単一フィールドを残さず
  明示的な3フィールドのみとする。
- 修正済み（adjusted）版の3種展開はv1スコープ外とし、必要になった時点で別途検討する
  （v1は非修正の3種のみ）。

### 2.4 モデル固有の追加結果の配置

- **RE: ハウスマン検定は`fit()`内で自動計算**し、`REResult`に`hausman_statistic` /
  `hausman_p_value` / `hausman_df`として含める（詳細は[`re-spec.md`](./re-spec.md)3.7節）。
  - **回帰ベース（補助回帰）版**（`plm::phtest(method = "aux")`相当）で、比較相手はRE本体と同じ個体効果構造のFE（1-wayはentity、2-wayはentity+time）。
    補助回帰のWald検定の共分散はRE本体の`cov_type`に連動する（`re-spec.md`3.7節）。
  - **`hausman_*`が`None`になるのは、検定できる傾き係数が無い場合のみ**（1-wayは時間変化のある列が無い、2-wayは2-way within変換後に分散のある列が無い、または残りが識別できない列）。補助回帰のランク落ち・
    Wald検定の特異性・クラスター数不足など、計算自体が成立しない場合は`hausman_*`を`None`にせず
    `fit()`が失敗する（設計行列の多重共線性でエラーにするのと同じ方針。`re-spec.md`3.7節）。
- **パネル固有R²（2.3節）**: `fit()`の結果本体に含める（別メソッド化しない）。
- **IV: 第一段階回帰結果は別メソッド**（[`iv-spec.md`](./iv-spec.md)2章参照）。

## 3. 標準誤差・検定

### 3.1 `cov_type`のサポート対象

`classical` / `hc0`〜`hc3` / `cluster` / `dk`（当初の名前は`hac`。OLSの`hac`と別の推定量のため改名）をすべて実装する予定だったが、**`hc0`は
スコープ外とした**（仕様判断。他パッケージの有無は理由にしない: fixestは`vcov = "HC0"`を受け付けないが、plm 2.6.7は提供する）ため、`FeCovType` enumは`Hc1`/`Hc2`/`Hc3`のみを持つ。詳細・数式は
`engine/src/panel/fe.rs`モジュールdoc「`cov_type`対応」参照。以下`hc0`を除く
`classical`/`hc1`〜`hc3`/`cluster`/`dk`が実装対象。ただし**`dk`はOLSの`hac`の実装を
そのまま流用しない**。OLSの`hac`はグローバルな時系列順序（`dk_time`）に対する単純な
Newey-West型HACだが、これをパネルにそのまま適用すると異なるエンティティの観測を単一の
時系列カーネルに混ぜてしまい、経済学的に不正確になる。パネル用に**Driscoll-Kraay型の
パネルHAC**（fixestの`vcov="DK"`、Stataの`xtscc`相当。時間方向にクロスセクション平均を
取ってからHACカーネルを適用し、エンティティ間・エンティティ内の両方の相関にロバストにする）を
別アルゴリズムとして実装する（具体的な実装は[`fe-spec.md`](./fe-spec.md)3.3節）。

### 3.2 `cov_type`のデフォルト

- `FEOptions`/`REOptions`の`cov_type`デフォルト値は**`"cluster"`（`effects`の先頭の列＝個体単位）**とする。
  OLSの`"classical"`デフォルトから**意図的に逸脱する**。
  - 理由: fixestは実際にこの挙動（FE指定時は自動的に最初のFE変数でクラスターする）を
    デフォルトにしている前例がある。パネルデータでは異分散だけでなくエンティティ内の
    系列相関がほぼ常に存在し、`hc0`〜`hc3`（クラスタリングなしの異分散ロバスト）だけでは
    標準誤差を過小評価するリスクが高い（Cameron & Miller 2015）。
  - FE/REは`effects`が必須引数（1章）のため、OLSと違いクラスター対象列が常に確実に
    存在し、デフォルト化の実装上の障害がない。
- `cov_type="cluster"`時、`cluster`省略なら`effects`の先頭の列を自動的にクラスターキーとして
  使う（fixestの慣習と同じ）。`cluster`を明示指定すれば任意の列（例: 個体より粗い粒度の`state`等）でも
  クラスター可能（OLSの`cluster`と同じ任意指定パターン）。
- **2-way clustering（個体+時点同時）はv1スコープ外**。2-way FEのスコープ（2-way FEでも
  クラスターのデフォルトは`effects[0]`（個体）単位のまま維持する。詳細は[`fe-spec.md`](./fe-spec.md)参照）と
  合わせて別途検討する。

### 3.3 検定分布

**t分布**（OLS準拠）。自由度（`df_inference`）は`classical`/`hc1`〜`hc3`のとき
FE/REそれぞれのパネル調整済み`df_resid`を使うが、**`cluster`のとき`G-1`、`dk`のとき
`t_periods-1`に切り替える**（fixestの`ssc()`既定`t.df="min"`に合わせた。
OLS自身が`cov_type=Cluster`のときだけ`n_groups-1`に切り替える既存パターンを、
FE/REのClassical/HC1-3以外の全cov_typeに広げたもの）。`df_resid`自体（σ̂²・
調整済みR²・AIC/BIC等）は`cov_type`によらず常に元の値のまま。詳細な式は
[`fe-spec.md`](./fe-spec.md)3.3節・[`re-spec.md`](./re-spec.md)3.4節参照。

## 4. 内部実装・共通化

### 4.1 既存の共通化パターン（前提）

- `CommonError`（`engine/src/shared/error.rs`）、`ensure_well_conditioned_symmetric_matrix`
  （`engine::shared::linear_algebra`）、`engine_pybind/src/shared/validation.rs`の列名検証4関数
  （`validate_x_non_empty`等）は既に系統横断で共有済み。
- **WLSはOLSを「並行輸入」ではなく「委譲」で再利用している**: 重み変換
  （`sqrt(weight)`）したデータをそのまま`OlsEstimator::fit`に渡し、その後で重み付き用に
  補正が必要な統計量（R²・調整済みR²・log_likelihood）だけ`weighted_fit_statistics`で
  計算し直す設計。「無理のない共通化」の実例として以降の方針の土台にする。

### 4.2 系統横断で切り出した共通化（FE/RE/IV着手前に実施済み）

1. **t/z検定の後処理の共通関数化**: OLS（t分布、`ols.rs:396-420`）とLogit（z分布、
   `logit.rs:694-717`）で、`std_err`/`stat`/`p_value`/`conf_low`/`conf_high`を計算する
   ループがほぼ同型のまま系統ごとに独立実装されていた。`statrs::distribution::ContinuousCDF`を
   ジェネリックに取る関数として`engine/src/shared/inference.rs`（系統をまたいで共有する位置）に
   切り出した。OLS・Logitに加えて、同型の重複がある`probit.rs`（z分布）・`nonlinear/common.rs`の
   `marginal_effects_from_w_s`（限界効果のSE/z値/CI、Logit/Probit共通）も対象に含めた。
   FE/RE（t分布）・IVの2SLS（t分布）・GMM（z分布、[`iv-spec.md`](./iv-spec.md)3.3節）もこの
   関数を使う。
2. **`engine_pybind`の`cov_type`文字列パース＋`cluster`/`dk_time`抽出ブロックの共通化**:
   `ols.rs:297-316`と`wls.rs:124-143`がほぼ完全一致で重複していたため共通関数化した。
3. **`validate_no_duplicate_roles`の複数列ロール対応拡張**: IVの`instruments`（複数列
   ロール）に必要（詳細は[`iv-spec.md`](./iv-spec.md)3.1節）。

### 4.3 FEの内部実装方針

FEは**まずOlsEstimatorへの委譲を試す**（within変換したデータを`OlsEstimator::fit`に渡す、
WLSと同型のパターン）。ただし以下はFE固有の再計算・補正が必要になる見込みで、無理に
OLSの計算をそのまま使わない（WLSがR²等を素のOLS計算のまま使わなかったのと同じ教訓）。

- 自由度（`n - 個体数 - k`、単純な`n-k`ではない）→ 検定統計量・adjusted R²・
  AIC/BICすべてに波及
- パネル固有R²（within/between/overall、2章）はOLSに存在しない新規計算
- `cov_type`デフォルトの`effects[0]`単位cluster化、HACのDriscoll-Kraay別実装（3章）

この委譲パターンが実際にうまくいくかは、within変換の実装方法（[`fe-spec.md`](./fe-spec.md)
3.1節）次第のため、結論を固定せず「まず委譲を試して、補正が管理可能な範囲に収まるか実装時に
判断する」という緩い方針とした（実装の結果、委譲パターンのまま実現できた。詳細は
[`fe-spec.md`](./fe-spec.md)3章）。

### 4.4 新規エラー型の設計

`LeastSquaresError`（OLS/WLS共有、`engine/src/linear/common.rs`）・`MleError`（nonlinear共有、
`engine/src/nonlinear/common.rs`）の前例に倣い、**`PanelError`をFE/REで共有する**
（`engine/src/panel/common.rs`に定義）。個別に`FeError`/`ReError`は作らない。`CommonError`
（`DimensionMismatch`等）は`#[from]`でラップする既存パターンを踏襲し、FE/RE固有のバリアント
（自由度計算失敗、singleton関連等）は`PanelError`に直接追加する。

### 4.5 共通化しない（意図的に見送り）

- OLSの`X'X`ベース分散計算とnonlinearのHessianベース分散計算は数式の前提が異なるため
  統一しない（[`nonlinear-common.md`](./nonlinear-common.md)でも同じ判断が既にされている）。
  IVのサンドイッチ型分散も無理にどちらかに寄せず独自実装でよい（[`iv-spec.md`](./iv-spec.md)
  3.1節）。
- `engine_pybind`の`fit()`関数全体のマクロ・テンプレート化はしない。手法ごとの抽出列・
  結果フィールドの差が大きく、無理に共通化すると可読性が落ちる。「抽出→バリデーション→
  engine呼出→結果構築」という大枠の流れだけ踏襲し、実装は個別に書く。

## 5. リファレンス実装・テスト方針

### 5.1 Python主リファレンス

**`linearmodels`をFE/RE共通の主リファレンスとする**（`PanelOLS`＝FE、`RandomEffects`＝RE）。
点推定・パネル固有R²・AIC/BIC等はこの方針のまま。
`pyfixest`は既存方針（`docs/spec/ols-spec.md`／`testing-policy.md`）を踏襲し、リファレンスに使わない。

**【例外】** 標準誤差・推論統計量（`std_errors`/`test_stats`/`p_values`/
`conf_lower`/`conf_upper`）については、`linearmodels`ではなくR（FEは`fixest`、REは
`plm`・plm相当のStata型補正）を正とする。FE/REの`cluster`/`dk`の小標本補正を
fixest（R）・Stata型に変更したため（`linearmodels`は独自の`n/(n-extra_df-k)`補正を
使い、fixest・Stataの利用者が期待する値と一致しない）、この2つのcov_typeについては
`linearmodels`との数値一致を意図的に崩している。`classical`/`hc1`の標準誤差は
引き続き`linearmodels`と数値一致する（変更なし。`hc2`/`hc3`は`linearmodels`に無くR（FEは`fixest`、
REは`plm`）のみ）。

### 5.2 Rクロスチェックパッケージ

- **RE**: `plm`（`model = "random"`）。`hc2`/`hc3`・`cluster`（`vcovHC(method = "arellano",
  type = "sss")`）・`dk`（`vcovSCC(type = "sss")`）の標準誤差を検証する（`benchmark/panel/`配下の
  `run_plm_benchmark.R`）。
- **FE**: `fixest`（`benchmark/panel/`配下の`run_fixest_benchmark.R`）。`fixest`自体は
  `.devcontainer/Dockerfile`に既にインストール済み。`hc2`/`hc3`・`cluster`・`dk`は
  `fixest`の`ssc()`既定のまま機械精度で一致する。
- **FEの第2リファレンス**: 1-way FEの`cluster`（entityクラスター）・`dk`は`plm`
  （`model = "within"`）＋`sandwich`でも検証する（`benchmark/panel/references/
  run_plm_fe_benchmark.R`）。clusterはplmのwithin変換後データへの定数項付き`lm`に
  `sandwich::vcovCL(type = "HC1", cadjust = TRUE)`を当てる（`K = k+1`、手計算の補正
  係数なし）。dkは`plm::vcovSCC(type = "HC0")`にfixestの補正`T/(T-1)·(n-1)/(n-k-G)`を
  手計算で掛けるため、plmが検証するのはカーネル・バンド幅の規約までで補正係数は
  fixestのみが独立に検証する。2-way・entity以外のクラスター列はplmが扱えずfixestのみ。
  fixestが一致しない`bandwidth == t_periods - 1`（fixest側のoff-by-oneで最終ラグ項が落ちる。
  `fe-spec.md`3.3節7.）はplmのみが参照値になる。
  fixestの境界ケースのうち、`G = q+1`・`G = 2`（先頭3・2 entityへの絞り込み）、
  `T = q+1`のdk、dkのバンド幅0・1、entityあたりの観測数の偏り（[2,3,5,10]期）も
  plmで検証する。plmは非entityのクラスター列を扱えないため、`G = q+1`・`G = 2`・観測数の
  偏りはクラスター列を使うfixestの同名ケースとはデータが異なる（dkの`T = q+1`は同一
  データ）。within R²もplmと比較する。
- **REのclusterの自由度`G-1`**: plmの準偏差変換済みデータにstatsmodelsの
  `OLS(...).fit(cov_type="cluster", use_t=True)`を当て、標準誤差・p値・信頼区間・
  自由度をstatsmodelsにネイティブに計算させて検証する（`benchmark/panel/references/
  statsmodels_ref.py`、バランスパネルのみ）。entity以外のクラスター列（サイズ
  [2,3,5,10,30,50]のタイル、k=1・G=3の境界）も同じ経路で検証する。DKの`T-1`はstatsmodelsに同じ規約がなく
  第2リファレンスがない。

### 5.3 ハウスマン検定の参照値（例外規定）

`linearmodels`にはハウスマン検定の専用メソッドが無い。この場合、**通常の「Python主リファレンス
＋Rクロスチェック」の2系統検証の例外**として、**Rの`plm::phtest(method = "aux", effect = "individual")`のみを参照値とする**ことを
許容する。`testing-policy.md`の「一部の統計量だけRクロスチェックを省略しない」という原則から
意図的に外れる例外であることをテスト実装時のコメント・ドキュメントに明記する。
不均衡パネルではplm自身の分散成分（Swamy-Arora）が`linearmodels`準拠の本実装と異なり
`phtest`の統計量が1〜2%ずれるため、不均衡パネルの参照値は、`linearmodels`の分散成分から
補助回帰を組み立ててplmの`vcov`系を当てたもの（`phtest`と同じ手順をplm自身の分散成分で
再現すると`phtest`と一致することを生成のたびに確認）にする。

### 5.4 許容誤差

既存方針（相対誤差1e-8を基本）を維持する。反復計算を伴う手法で乖離が大きい場合は、
OLSのHAC（実測乖離に基づき1e-2に緩和した前例、`testing-policy.md`）と同様、実装・テスト後の
実測値に基づいて個別に緩和を検討する（先に緩めない）。
