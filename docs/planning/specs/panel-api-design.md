# FE / RE API・オプション設計

[`panel-iv-design-points.md`](../panel-iv-design-points.md)の論点をモデルごとにIssue化したうちの、
FE/RE共通部分の確定事項をまとめる。IV固有の設計は[`iv-api-design.md`](./iv-api-design.md)を参照。

**ステータス**: 確定（1章: 引数設計／2章: 結果設計／3章: 標準誤差・検定／
4章: 内部実装・共通化／5章: リファレンス実装・テスト方針／
6章: FE固有論点／7章: RE固有論点）。FE/RE側の論点はすべて確定。
IV固有論点は`iv-api-design.md`参照。

## 1. 引数設計（確定）

### 1.1 `y` / `x` / `entity` / `time` のシグネチャ

- `fit_fe(data, y, x, entity, options)` / `fit_re(data, y, x, entity, options)`
- `y: str`、`x: list[str]`はOLSと同じ。
- `entity: str`（エンティティID列名）は**独立の必須引数**とする。FE/REいずれもパネル構造が
  無ければモデルとして成立しないため、`y`/`x`と同格に扱う。
- `time`（時点ID列名）は必須ではないため`Options`内に置く（`FeOptions.time` /
  `ReOptions.time`、`str | None`、デフォルト`None`）。
  - FEの2-way（entity + time FE）を指定する場合は`time`が実質的に必須になるが、これは
    `OLSOptions.cluster_col`が`cov_type="cluster"`のときのみ必須になるのと同じ「条件付き必須」
    パターンであり、`Options`に置くという判断自体は変えない。未指定時のバリデーションエラーで
    担保する。
- **命名規則**: `entity`/`time`は**bareネーミング**（`_col`サフィックスなし）を採用する。
  `y`/`x`/`weight`（WLS、`engine_pybind/src/linear/wls.rs`）と同じく、モデルを構成する中核的な
  変数という位置づけのため。既存`OLSOptions.cluster_col`/`time_col`
  （`engine_pybind/src/linear/ols.rs:58-74`）は「診断・ロバストSE計算のための補助列」という
  別の位置づけであり`_col`サフィックスを持つが、これを遡って改名することはしない
  （新規追加分から新しい命名規則を適用する）。
- FE/RE間で`entity`/`time`の命名は統一する（`panel-iv-design-points.md`1.1「3モデルで表記を
  揃えるか」への回答）。IV側の対応する引数は性質が異なるため`iv-api-design.md`を参照。

### 1.2 モデル固有オプションの置き場所

`FeOptions`/`ReOptions`という別々の`#[pyclass]`構造体に含める（`OLSOptions`/`LogitOptions`の
前例を踏襲）。共有可能なフィールド（`entity`/`time`等）を内部実装上どこまで共通化するか
（トレイト・共通struct等）はAPI設計とは別問題として#122（内部実装・共通化方針）で扱う。

### 1.3 `weights` / `offset` の扱い

- `offset`: 線形モデル（link functionを前提としない）のため**該当なし**。今後も追加予定なし。
- `weights`: 汎用の頻度/分析重みオプションとしては**見送り**
  （`nonlinear-api-design.md`のLogit/Probit/Tobitにおける判断を踏襲、Phase6で再検討）。
  - **注**: この判断はGLS（直近の実装順序でFE/RE/IVの後に着手予定）の引数設計を拘束しない。
    GLSは重み・共分散構造の指定自体がモデルの本質であり、WLSの`weight: str`（必須の独立引数）
    に近い位置づけになる見込み。GLS着手時に別途設計する。

## 2. 結果（Return）設計（確定）

### 2.1 共通コア項目

OLS（`OLSResult`, `engine_pybind/src/linear/ols.rs:137-191`）の項目を土台にするが、以下の点で
機械的な流用ではなく調整する。

| フィールド | 由来 | 備考 |
|---|---|---|
| `params` / `std_errors` / `t_stats` / `p_values` / `conf_lower` / `conf_upper` / `param_names` | OLS共通 | 検定分布（t/z）は#121で確定 |
| `residuals` / `dep_var_name` | OLS共通 | そのまま踏襲 |
| `n_obs` | Logit由来の表記 | OLS/WLSも`n_obs`に統一済み（[後述](#22-olsのnobsn_obsリネーム完了)） |
| `df_resid` / `df_model` | Logit由来、**新規追加** | OLSには無いが、FEは自由度調整が
  `n - n_entities - k`という非自明な式になるため明示的に返す価値が高い |
| `n_entities` | **新規追加**（FE/RE限定） | パネルユニット数。pyfixest/plmの前例に倣う |
| `cov_type` | OLS共通 | サポート対象は#121で確定 |
| `f_statistic` / `f_p_value` | OLS共通 | そのまま踏襲 |
| `log_likelihood` / `aic` / `bic` | OLS共通 | FE/REは最小二乗族で正規性下の尤度が定義できるため含める |
| `r_squared_within` / `r_squared_between` / `r_squared_overall` | **新規追加**（OLSの`r_squared`/`r_squared_adj`を置き換え） | 詳細は2.3 |

### 2.2 OLSの`nobs`→`n_obs`リネーム（完了）

既存`OLSResult.nobs`（`ols.rs:157`）とLogitの`LogitResult.n_obs`（`logit.rs:206`）で表記が
不一致だったため、`n_obs`へ統一した（OLS・WLS双方のResult型・Python側プロパティを
リネーム済み）。

### 2.3 R²の種類（within/between/overall）

- `r_squared_within` / `r_squared_between` / `r_squared_overall`の3フィールドを`fit()`の
  戻り値に含める。
- **bareの`r_squared`/`r_squared_adj`は廃止する**（OLSの`r_squared`をそのまま流用しない）。
  パネルモデルでは「どのR²か」が一意に決まらないため、曖昧な単一フィールドを残さず
  明示的な3フィールドのみとする。
- 修正済み（adjusted）版の3種展開は本Issueのスコープ外とし、必要になった時点で別途検討する
  （v1は非修正の3種のみ）。

### 2.4 モデル固有の追加結果の配置

- **RE: ハウスマン検定は`fit()`内で自動計算**し、`ReResult`に`hausman_statistic` /
  `hausman_p_value` / `hausman_df`として含める。
  - v1は**classical Hausman検定のみ**実装する（`cov_type`に依存しない、常にclassical SE
    前提での計算）。`cov_type="cluster"`等でfitした場合でも、ハウスマン検定自体は内部で
    classical前提のまま計算する（整合性の注記をdocstringに明記する）。
  - 将来的に`cov_type`と連動するrobust版（Wooldridgeの回帰ベース検定等）を追加できるよう、
    フィールド名・置き場所には拡張余地を残す（v1では実装しない）。
  - `RE.fit()`は内部でFE推定を実行してハウスマン検定の比較対象を得る（`entity`/`time`/`x`は
    RE呼び出し時と同一の指定を使う）。**FE推定が失敗した場合**（singleton除外後の変動不足等）
    は、`hausman_statistic`等を`None`にしたうえで、RE本体の結果は正常に返す
    （REの主要な結果自体は有効なため、診断情報の欠落だけに留める）。
  - REとFEの内部設計共有範囲（同じ推定ルーチンを呼ぶか等）は4章/7章で扱う。
- **パネル固有R²（2.3）**: `fit()`の結果本体に含める（別メソッド化しない）。
- **IV: 第一段階回帰結果は別メソッド**（`iv-api-design.md`2章参照）。

## 3. 標準誤差・検定（確定）

### 3.1 `cov_type`のサポート対象

`classical` / `hc0`〜`hc3` / `cluster` / `hac`をすべて実装する。ただし**`hac`はOLSの実装を
そのまま流用しない**。OLSの`hac`はグローバルな時系列順序（`time_col`）に対する単純な
Newey-West型HACだが、これをパネルにそのまま適用すると異なるエンティティの観測を単一の
時系列カーネルに混ぜてしまい、経済学的に不正確になる。パネル用に
**Driscoll-Kraay型のパネルHAC**（fixestの`vcov="DK"`、Stataの`xtscc`相当。時間方向にクロス
セクション平均を取ってからHACカーネルを適用し、エンティティ間・エンティティ内の両方の相関に
ロバストにする）を別アルゴリズムとして実装する。具体的な実装（`time`引数の利用、バンド幅
パラメータの設計等）は4章（内部実装・共通化）で扱う。

### 3.2 `cov_type`のデフォルト

- `FeOptions`/`ReOptions`の`cov_type`デフォルト値は**`"cluster"`（entity単位）**とする。
  OLSの`"classical"`デフォルトから**意図的に逸脱する**。
  - 理由: fixestは実際にこの挙動（FE指定時は自動的に最初のFE変数でクラスターする）を
    デフォルトにしている前例がある。パネルデータでは異分散だけでなくエンティティ内の
    系列相関がほぼ常に存在し、`hc0`〜`hc3`（クラスタリングなしの異分散ロバスト）だけでは
    標準誤差を過小評価するリスクが高い（Cameron & Miller 2015）。
  - FE/REは`entity`が必須引数（1章）のため、OLSと違いクラスター対象列が常に確実に
    存在し、デフォルト化の実装上の障害がない。
- `cov_type="cluster"`時、`cluster_col`省略なら`entity`引数の列を自動的にクラスターキーとして
  使う。`cluster_col`を明示指定すれば任意の列（例: `entity`より粗い粒度の`state`等）でも
  クラスター可能（OLSの`cluster_col`と同じ任意指定パターン）。
- **2-way clustering（entity+time同時）はv1スコープ外**。2-way FEのスコープ確定
  （6章）と合わせて別途検討する。

### 3.3 検定分布

**t分布**（OLS準拠）。自由度は#120で新規追加した`df_resid`（`n - n_entities - k`調整済み）を
使う。

## 4. 内部実装・共通化（確定）

### 4.1 既存の共通化パターン（前提）

- `CommonError`（`engine/src/error.rs`）、`ensure_well_conditioned_symmetric_matrix`
  （`engine::linear_algebra`）、`engine_pybind/src/validation.rs`の列名検証4関数
  （`validate_x_non_empty`等）は既に系統横断で共有済み。
- **WLSはOLSを「並行輸入」ではなく「委譲」で再利用している**: 重み変換
  （`sqrt(weight)`）したデータをそのまま`OlsEstimator::fit`に渡し、その後で重み付き用に
  補正が必要な統計量（R²・調整済みR²・log_likelihood）だけ`weighted_fit_statistics`で
  計算し直す設計。「無理のない共通化」の実例として以降の方針の土台にする。

### 4.2 新規に切り出す共通化（FE/RE/IV着手前に実施）

1. **t/z検定の後処理の共通関数化**: OLS（t分布、`ols.rs:396-420`）とLogit（z分布、
   `logit.rs:694-717`）で、`std_err`/`stat`/`p_value`/`conf_low`/`conf_high`を計算する
   ループがほぼ同型のまま系統ごとに独立実装されている。`statrs::distribution::
   ContinuousCDF`をジェネリックに取る関数としてcrate直下（`engine`直下、系統をまたぐ
   位置）に切り出す。FE/RE/2SLS（t分布）・GMM（z分布、3章）もこの関数を使う。
2. **`engine_pybind`の`cov_type`文字列パース＋`cluster_col`/`time_col`抽出ブロックの共通化**:
   `ols.rs:297-316`と`wls.rs:124-143`がほぼ完全一致で重複している。FE/RE/IVで重複を
   増やす前に共通関数化する。
3. **`validate_no_duplicate_roles`の複数列ロール対応拡張**: IVの`instruments`（複数列
   ロール）に必要（詳細は`iv-api-design.md`4章）。

### 4.3 FEの内部実装方針（緩い方針、確定）

FEは**まずOlsEstimatorへの委譲を試す**（within変換したデータを`OlsEstimator::fit`に渡す、
WLSと同型のパターン）。ただし以下はFE固有の再計算・補正が必要になる見込みで、無理に
OLSの計算をそのまま使わない（WLSがR²等を素のOLS計算のまま使わなかったのと同じ教訓）。

- 自由度（`n - n_entities - k`、単純な`n-k`ではない）→ 検定統計量・adjusted R²・
  AIC/BICすべてに波及
- パネル固有R²（within/between/overall、2章）はOLSに存在しない新規計算
- `cov_type`デフォルトのentity単位cluster化、HACのDriscoll-Kraay別実装（3章）

この委譲パターンが実際にうまくいくかは、within変換の実装方法（6章）次第のため、
今は結論を固定せず「まず委譲を試して、補正が管理可能な範囲に収まるか実装時に判断する」
という緩い方針とする。うまくいかない場合はFE専用実装に切り替えてよい。

### 4.4 新規エラー型の設計

`LeastSquaresError`（OLS/WLS共有、`engine/src/linear/common.rs`）・`MleError`（nonlinear共有、
`engine/src/nonlinear/common.rs`）の前例に倣い、**`PanelError`をFE/REで共有する**
（`engine/src/panel/common.rs`に定義）。個別に`FeError`/`ReError`を作らない。`CommonError`
（`DimensionMismatch`等）は`#[from]`でラップする既存パターンを踏襲し、FE/RE固有のバリアント
（自由度計算失敗、singleton関連等）は`PanelError`に直接追加する。

### 4.5 共通化しない（意図的に見送り）

- OLSの`X'X`ベース分散計算とnonlinearのHessianベース分散計算は数式の前提が異なるため
  統一しない（`nonlinear/common.rs`のコメントでも同じ判断が既にされている）。IVの
  サンドイッチ型分散も無理にどちらかに寄せず独自実装でよい。
- `engine_pybind`の`fit()`関数全体のマクロ・テンプレート化はしない。手法ごとの抽出列・
  結果フィールドの差が大きく、無理に共通化すると可読性が落ちる。「抽出→バリデーション→
  engine呼出→結果構築」という大枠の流れだけ踏襲し、実装は個別に書く。

## 5. リファレンス実装・テスト方針（確定）

### 5.1 Python主リファレンス

**`linearmodels`をFE/RE共通の主リファレンスとする**（`PanelOLS`＝FE、`RandomEffects`＝RE）。
`PanelOLS`の`cov_type="kernel"`でDriscoll-Kraay型SE（3章）の検証もカバーできる。
`pyfixest`は既存方針（`docs/spec/ols-spec.md`／`testing-policy.md`、HC2/HC3の実装バグにより
精度検証には使わない）を踏襲し、性能比較（実行時間・メモリ）のみに使う。

### 5.2 Rクロスチェックパッケージ

- **RE**: `plm`（`model = "random"`）。`benchmark/panel/run_plm_benchmark.R`は現状
  `model = "within"`（FE相当）呼び出しになっているため、RE専用に変更する（実装時対応）。
- **FE**: `fixest`。`benchmark/panel/`配下に新規スクリプト（`run_fixest_benchmark.R`）の
  作成が必要（実装時対応。`fixest`自体は`.devcontainer/Dockerfile`に既にインストール済みで
  追加インストール不要）。

### 5.3 ハウスマン検定の参照値（例外規定）

`linearmodels`にはハウスマン検定の専用メソッドが無い可能性が高い（要実装時再確認）。この場合、
**通常の「Python主リファレンス＋Rクロスチェック」の2系統検証の例外**として、
**Rの`plm::phtest`のみを参照値とする**ことを許容する。`testing-policy.md`の「一部の統計量だけ
Rクロスチェックを省略しない」という原則から意図的に外れる例外であることをテスト実装時の
コメント・ドキュメントに明記する。

### 5.4 許容誤差

既存方針（相対誤差1e-8を基本）を維持する。GMM等の反復計算を伴う手法で乖離が大きい場合は、
OLSのHAC（実測乖離に基づき1e-2に緩和した前例、`testing-policy.md`）と同様、実装・テスト後の
実測値に基づいて個別に緩和を検討する（先に緩めない）。

## 6. FE固有論点（確定）

### 6.1 within変換の実装方法

polarsの`group_by`機能で実装する。2-way（後述6.2）の場合はエンティティ平均・時点平均を引き
グランド平均を足し戻す閉形式の二重デミーニング
`ỹ_it = y_it - ȳ_i. - ȳ_.t + ȳ..`で計算する。**この閉形式はバランスパネルでのみ正確**であり、
「group_byで素直にやる」という実装方針自体が6.4のバランスパネル制約（2-way限定）と表裏一体で
ある。不均衡パネルで2-wayを正確に行うには反復的な交互射影（fixest/lfe方式）が必要になるが、
v1では扱わない。

### 6.2 1-way / 2-wayのスコープ

**v1から2-way（entity + time FE）までサポートする**（entity onlyに絞らない）。

### 6.3 自由度調整

- **1-way**: `n - n_entities - k`
- **2-way**: `n - n_entities - n_periods + 1 - k`
  - entityダミーとtimeダミーの間に定数項ぶんの重複（ランク落ち）が1つ生じるため、単純な
    `n - n_entities - n_periods - k`ではなく`+1`の補正が必要（LSDVでの有効パラメータ数が
    `n_entities + n_periods - 1`になることに対応）。

### 6.4 バランスパネルの前提

- **2-wayのみバランスパネルを必須とする。1-wayは不均衡パネルもv1からサポートする**
  （1-wayはエンティティごとの平均を引くだけで数学的に不均衡でも正確に成立するため、
  不要な制約を課さない）。
- 2-wayでバランスパネルでない場合は**常に`ValidationError`**とする。「不均衡を明示して
  チェックを回避する」ようなオプションは用意しない（回避可能にすると、閉形式の二重
  デミーニングが数学的に不正確なまま計算されてしまうリスクがあるため）。
- 不均衡2-way FEへの対応（反復的交互射影）は将来の別Issueとする。

### 6.5 singletonの扱い

- **自動除外せず、常に`ValidationError`とする**（欠損値を常にエラーとする既存の全体方針を
  踏襲。listwise deletion等の自動除外はしない）。
- 実装は「観測数1のグループを検出して明示的な`ValidationError`にする」方式にする
  （下流の特異行列エラーとして偶発的に検出される形にはしない）。
- **2-wayの場合、entityだけでなくtimeのsingleton（観測1件のみの時点）も同様に検出対象と
  する**（対称に扱う）。

### 6.6 固定効果自体（α_i）の取得方法

**別メソッド`fixed_effects()`を用意する**（`fit()`の戻り値本体には含めない。IVの
`first_stage()`と同じ「追加結果は別メソッド」方針を踏襲）。

- 1-way: `fixed_effects() -> dict[str, float]`（エンティティIDをキーとする）。
- 2-way: `fixed_effects() -> dict[str, dict[str, float]]`（トップレベルキーは`"entity"`/
  `"time"`、それぞれのvalueがID→効果のdict）。
- α_iは`within`変換時に保持しておいたグループ平均と推定済み係数から事後的に復元する
  （`α_i = ȳ_i - x̄_i'β̂`、2-wayも同様の考え方を時点効果に拡張）。within変換の実装
  （6.1）でグループ平均を捨てずに保持しておく必要がある。

### 6.7 within変換後に分散ゼロになる説明変数の検出

新規バリデーションとして実装する。デミーニング後の設計行列の各列の分散を確認し、ゼロの列が
あれば`ValidationError`にする（1-way/2-wayで同じロジックを共有できる。時間不変変数だけで
なく、2-wayでtime FEと完全共線な「エンティティ間で変動しない列」も同じチェックで検出できる）。

### 6.8 2-way FEとクラスター標準誤差のデフォルト（3章との接続）

3章で「2-way clustering（entity+time同時）はv1スコープ外、2-way FEのスコープ確定と
合わせて別途検討する」としていた保留事項を、6.2の決定を受けて確定する。**2-way FEでも
クラスターのデフォルトはentity単位のまま維持する**（`cluster_col`で上書き可能な既存挙動を
変えない）。2-way clustering自体はv1スコープ外のまま据え置く。

## 7. RE固有論点（確定）

### 7.1 分散成分の推定方法

**Swamy-Arora法を採用する**。R `plm`のデフォルト（`random.method="swar"`）、Python
`linearmodels.RandomEffects`の実装、いずれもSwamy-Arora相当であり、5章で確定した
2つの参照実装（主リファレンスlinearmodels、クロスチェックplm）と自然に整合する
（`linearmodels/panel/model.py`のソースで実装を確認済み）。

- **σ_ε²（idiosyncratic variance）**: within変換済み残差の分散、分母は`n - k - n_entities + 1`
  （FEのdf調整に類似した式。linearmodelsのソースで確認済み）。
- **σ_u²（individual variance）**: between回帰（エンティティ平均に対するOLS）から、調和平均
  `t_bar = n_entities / Σ(1/T_i)`を使う標準式`max(0, ssr/(n_entities - k) - σ_ε²/t_bar)`。
- linearmodelsが提供する`small_sample`補正（不均衡パネル向けのtraceベースの追加調整、
  デフォルト`False`）はv1では実装しない（linearmodelsのデフォルト挙動に合わせる）。

### 7.2 θ（準偏差変換の重み）計算

`θ_i = 1 - sqrt(σ_ε² / (T_i・σ_u² + σ_ε²))`（linearmodelsのソースで確認済み、Baltagiの教科書
通りの式）。

- **REはentity方向のみ（2-way REはv1スコープ外）**なので、6章のFE・1-wayと同じ扱いで
  **不均衡パネルもv1から無条件でサポートする**。`T_i`（エンティティごとの観測数）を直接使う
  この式は教科書レベルで不均衡対応済みであり、FEの2-wayのような反復アルゴリズムは不要。

### 7.3 ハウスマン検定の実装場所・インターフェース

- 2章の決定通り、**`RE.fit()`内で自動計算し`ReResult`にのみ含める**（`FeResult`には
  追加しない）。
- **計算部分（カイ二乗統計量そのもの）は共通関数化する**: `engine/src/panel/common.rs`に
  `hausman_statistic(beta_fe, cov_fe, beta_re, cov_re) -> (stat, df, p_value)`を実装し、
  RE側からのみ呼ぶ。
- **比較対象はFE/RE間で重なりのあるスロープ係数のみ**（REの切片は比較から除外する。FEには
  切片が存在しないため、次元を揃える必要がある）。
- REが時間不変変数を含む場合、内部FE推定は6章の分散ゼロ検証
  （[6.7](#67-within変換後に分散ゼロになる説明変数の検出)）で失敗する。この場合は2章で
  決めた「FE推定失敗時はハウスマン関連フィールドを`None`にする」フォールバックがそのまま
  適用される。
- linearmodelsのソースコードを確認したところ`hausman`という文字列は一切登場せず、専用実装が
  無いことを確定した。5章で決めた「`plm::phtest`のみを参照値とする例外」の妥当性を
  裏付ける。

### 7.4 FEとの内部設計共有範囲

- **θでパラメータ化した共通の準偏差変換関数を実装する**:
  `quasi_demean(data, entity, theta: &[f64]) -> transformed_data`。FEは全エンティティに
  `θ_i = 1.0`を渡すことでこの関数の特殊ケースとして扱える（6章の`OlsEstimator`委譲
  方針とあわせ、FE/RE双方がこの関数の出力を`OlsEstimator::fit`に渡す設計にできる）。
- **σ_ε²の推定（7.1）はFEのwithin回帰の残差分散をそのまま利用する**。`RE.fit()`は内部で
  FE推定を呼び出し、その残差分散を再利用する（RE → FE → `OlsEstimator`という委譲チェーンとして
  整理する）。
- ハウスマン統計量の計算関数（7.3）も共有する。
- 上記以外（between回帰によるσ_u²推定、θ計算そのもの）はRE固有のロジックとし、無理に
  共通化しない。

### 7.5 REのdf_resid

**`n - k`**（linearmodelsのソースで`df_resid = wy.shape[0] - wx.shape[1]`と確認済み）。
FEの`n - n_entities - k`（6.3）とは異なる式であることに注意する。REはGLS変換であり
FEのように個体ダミー相当の自由度を消費しないため、通常のOLSと同じ式になる。7.1の
σ_ε²推定で使う中間的な自由度調整（`n - k - n_entities + 1`）とは別物なので混同しないこと。
