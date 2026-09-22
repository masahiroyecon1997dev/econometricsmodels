# パネルモデル（FE/RE）共通仕様

FE（固定効果）/RE（変量効果）が共有する基盤の確定済み仕様。手法固有の内容（within変換・
Swamy-Arora分散成分推定の具体式、固定効果の復元等）は[`fe-spec.md`](./fe-spec.md) /
[`re-spec.md`](./re-spec.md)を参照し、本ドキュメントには2手法が共有する設計判断のみを
記載する。

## 1. 引数設計

### 1.1 `y` / `x` / `entity` / `time` のシグネチャ

- `fit_fe(data, y, x, entity, options)` / `fit_re(data, y, x, entity, options)`
- `y: str`、`x: list[str]`はOLSと同じ。
- `entity: str`（エンティティID列名）は**独立の必須引数**とする。FE/REいずれもパネル構造が
  無ければモデルとして成立しないため、`y`/`x`と同格に扱う。
- `time`（時点ID列名）は必須ではないため`Options`内に置く（`FEOptions.time` /
  `REOptions.time`、`str | None`、デフォルト`None`）。
  - FEの2-way（entity + time FE）を指定する場合は`time`が実質的に必須になるが、これは
    `OLSOptions.cluster_col`が`cov_type="cluster"`のときのみ必須になるのと同じ「条件付き必須」
    パターンであり、`Options`に置くという判断自体は変えない。未指定時のバリデーションエラーで
    担保する。
  - **`FEOptions.time`と`FEOptions.time_col`は別物**: `time`は2-way FE（固定効果構造）を
    指定するbareフィールドで、`Some`なら常に2-way・`None`なら1-way。Driscoll-Kraay型HAC
    （`cov_type="hac"`）専用の時系列順序は別フィールド`time_col`（`str | None`、
    `OLSOptions.cluster_col`/`time_col`と同じ「補助列」命名規則）で指定する。1-way FE +
    DK HAC（`time`未指定だが時系列順序だけ要る）という組み合わせを表現するために導入した
    ——`time`の有無だけで1-way/2-wayを決める設計（ブールフラグ等の追加無し）にすると、
    時系列順序と2-way構造を1つの`time`フィールドに詰め込めなくなるため分離が必要になった。
    **`time_col`は2-way（`time`指定あり）でも常に優先される**——「2-way FEの固定効果構造に
    使う時点粒度」と「DK HACカーネルに使う時系列粒度」が異なるケースにも対応するための設計。
    `time_col`未指定なら`time`（2-way）にフォールバックし、どちらも`None`なら（1-way FEで
    `cov_type="hac"`のとき）`PanelError::HacRequiresTime`。engine側は`FeCovType::Hac {
    bandwidth, time: Option<Vec<String>> }`（`time`が優先の上書き値）として実装
    （`engine/src/panel/fe.rs`モジュールdoc「Driscoll-Kraay型パネルHAC対応」参照）。
- **命名規則**: `entity`/`time`は**bareネーミング**（`_col`サフィックスなし）を採用する。
  `y`/`x`/`weight`（WLS、`engine_pybind/src/linear/wls.rs`）と同じく、モデルを構成する中核的な
  変数という位置づけのため。既存`OLSOptions.cluster_col`/`time_col`
  （`engine_pybind/src/linear/ols.rs:58-74`）は「診断・ロバストSE計算のための補助列」という
  別の位置づけであり`_col`サフィックスを持つが、これを遡って改名することはしない
  （新規追加分から新しい命名規則を適用する）。
- FE/RE間で`entity`/`time`の命名は統一する。IV側の対応する引数は性質が異なるため
  [`iv-spec.md`](./iv-spec.md)を参照。

### 1.2 モデル固有オプションの置き場所

`FEOptions`/`REOptions`という別々の`#[pyclass]`構造体に含める（`OLSOptions`/`LogitOptions`の
前例を踏襲）。共有可能なフィールド（`entity`/`time`等）を内部実装上どこまで共通化するか
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
| `params` / `std_errors` / `t_stats` / `p_values` / `conf_lower` / `conf_upper` / `param_names` | OLS共通 | 検定分布はt分布で統一（3.3節） |
| `residuals` / `dep_var_name` | OLS共通 | そのまま踏襲 |
| `n_obs` | Logit由来の表記 | OLS/WLSも`n_obs`に統一済み |
| `df_resid` / `df_model` | Logit由来、パネル向けに新規追加 | OLSには無いが、FEは自由度調整が`n - n_entities - k`という非自明な式になるため明示的に返す価値が高い |
| `n_entities` | 新規追加（FE/RE限定） | パネルユニット数。pyfixest/plmの前例に倣う |
| `cov_type` | OLS共通 | サポート対象は3.1節 |
| `f_statistic` / `f_p_value` | OLS共通 | そのまま踏襲（ただしengine側の実装はOLSの単純な流用ではない。傾き係数`k`個の同時Wald検定をFE/RE独自の`cov_type`別`cov_params`・パネル自由度調整済み`df_resid`で行う） |
| `log_likelihood` / `aic` / `bic` | OLS共通 | FE/REは最小二乗族で正規性下の尤度が定義できるため含める |
| `r_squared_within` / `r_squared_between` / `r_squared_overall` | 新規追加（OLSの`r_squared`/`r_squared_adj`を置き換え） | 詳細は2.3節 |

### 2.2 OLSの`nobs`→`n_obs`リネーム

既存`OLSResult.nobs`（`ols.rs:157`）とLogitの`LogitResult.n_obs`（`logit.rs:206`）で表記が
不一致だったため、`n_obs`へ統一した（OLS・WLS双方のResult型・Python側プロパティを
リネーム済み）。

### 2.3 R²の種類（within/between/overall）

- `r_squared_within` / `r_squared_between` / `r_squared_overall`の3フィールドを`fit()`の
  戻り値に含める。
- **bareの`r_squared`/`r_squared_adj`は廃止する**（OLSの`r_squared`をそのまま流用しない）。
  パネルモデルでは「どのR²か」が一意に決まらないため、曖昧な単一フィールドを残さず
  明示的な3フィールドのみとする。
- 修正済み（adjusted）版の3種展開はv1スコープ外とし、必要になった時点で別途検討する
  （v1は非修正の3種のみ）。

### 2.4 モデル固有の追加結果の配置

- **RE: ハウスマン検定は`fit()`内で自動計算**し、`REResult`に`hausman_statistic` /
  `hausman_p_value` / `hausman_df`として含める（詳細は[`re-spec.md`](./re-spec.md)3.7節）。
  - v1は**classical Hausman検定のみ**実装する（`cov_type`に依存しない、常にclassical SE
    前提での計算）。`cov_type="cluster"`等でfitした場合でも、ハウスマン検定自体は内部で
    classical前提のまま計算する（整合性の注記をdocstringに明記する）。
  - 将来的に`cov_type`と連動するrobust版（Wooldridgeの回帰ベース検定等）を追加できるよう、
    フィールド名・置き場所には拡張余地を残す（v1では実装しない）。
  - `RE.fit()`は内部でFE推定を実行してハウスマン検定の比較対象を得る（`entity`/`time`/`x`は
    RE呼び出し時と同一の指定を使う）。**FE推定が失敗した場合**（singleton除外後の変動不足等）
    は、`hausman_statistic`等を`None`にしたうえで、RE本体の結果は正常に返す
    （REの主要な結果自体は有効なため、診断情報の欠落だけに留める）。
- **パネル固有R²（2.3節）**: `fit()`の結果本体に含める（別メソッド化しない）。
- **IV: 第一段階回帰結果は別メソッド**（[`iv-spec.md`](./iv-spec.md)2章参照）。

## 3. 標準誤差・検定

### 3.1 `cov_type`のサポート対象

`classical` / `hc0`〜`hc3` / `cluster` / `hac`をすべて実装する予定だったが、**`hc0`は
スコープ外とすることが判明した**（linearmodels・fixestともにパネル/FE向けの`hc0`オプションが
存在しないため、`FeCovType` enumは`Hc1`/`Hc2`/`Hc3`のみを持つ。詳細・数式は
`engine/src/panel/fe.rs`モジュールdoc「`cov_type`対応」参照）。以下`hc0`を除く
`classical`/`hc1`〜`hc3`/`cluster`/`hac`が実装対象。ただし**`hac`はOLSの実装を
そのまま流用しない**。OLSの`hac`はグローバルな時系列順序（`time_col`）に対する単純な
Newey-West型HACだが、これをパネルにそのまま適用すると異なるエンティティの観測を単一の
時系列カーネルに混ぜてしまい、経済学的に不正確になる。パネル用に**Driscoll-Kraay型の
パネルHAC**（fixestの`vcov="DK"`、Stataの`xtscc`相当。時間方向にクロスセクション平均を
取ってからHACカーネルを適用し、エンティティ間・エンティティ内の両方の相関にロバストにする）を
別アルゴリズムとして実装する（具体的な実装は[`fe-spec.md`](./fe-spec.md)3.3節）。

### 3.2 `cov_type`のデフォルト

- `FEOptions`/`REOptions`の`cov_type`デフォルト値は**`"cluster"`（entity単位）**とする。
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
- **2-way clustering（entity+time同時）はv1スコープ外**。2-way FEのスコープ（2-way FEでも
  クラスターのデフォルトはentity単位のまま維持する。詳細は[`fe-spec.md`](./fe-spec.md)参照）と
  合わせて別途検討する。

### 3.3 検定分布

**t分布**（OLS準拠）。自由度はFE/REそれぞれのパネル調整済み`df_resid`を使う。

## 4. 内部実装・共通化

### 4.1 既存の共通化パターン（前提）

- `CommonError`（`engine/src/error.rs`）、`ensure_well_conditioned_symmetric_matrix`
  （`engine::linear_algebra`）、`engine_pybind/src/validation.rs`の列名検証4関数
  （`validate_x_non_empty`等）は既に系統横断で共有済み。
- **WLSはOLSを「並行輸入」ではなく「委譲」で再利用している**: 重み変換
  （`sqrt(weight)`）したデータをそのまま`OlsEstimator::fit`に渡し、その後で重み付き用に
  補正が必要な統計量（R²・調整済みR²・log_likelihood）だけ`weighted_fit_statistics`で
  計算し直す設計。「無理のない共通化」の実例として以降の方針の土台にする。

### 4.2 系統横断で切り出した共通化（FE/RE/IV着手前に実施済み）

1. **t/z検定の後処理の共通関数化**: OLS（t分布、`ols.rs:396-420`）とLogit（z分布、
   `logit.rs:694-717`）で、`std_err`/`stat`/`p_value`/`conf_low`/`conf_high`を計算する
   ループがほぼ同型のまま系統ごとに独立実装されていた。`statrs::distribution::ContinuousCDF`を
   ジェネリックに取る関数として`engine/src/inference.rs`（crate直下、系統をまたぐ位置）に
   切り出した。OLS・Logitに加えて、同型の重複がある`probit.rs`（z分布）・`nonlinear/common.rs`の
   `marginal_effects_from_w_s`（限界効果のSE/z値/CI、Logit/Probit共通）も対象に含めた。
   FE/RE（t分布）・IVの2SLS（t分布）・GMM（z分布、[`iv-spec.md`](./iv-spec.md)3.3節）もこの
   関数を使う。
2. **`engine_pybind`の`cov_type`文字列パース＋`cluster_col`/`time_col`抽出ブロックの共通化**:
   `ols.rs:297-316`と`wls.rs:124-143`がほぼ完全一致で重複していたため共通関数化した。
3. **`validate_no_duplicate_roles`の複数列ロール対応拡張**: IVの`instruments`（複数列
   ロール）に必要（詳細は[`iv-spec.md`](./iv-spec.md)3.1節）。

### 4.3 FEの内部実装方針

FEは**まずOlsEstimatorへの委譲を試す**（within変換したデータを`OlsEstimator::fit`に渡す、
WLSと同型のパターン）。ただし以下はFE固有の再計算・補正が必要になる見込みで、無理に
OLSの計算をそのまま使わない（WLSがR²等を素のOLS計算のまま使わなかったのと同じ教訓）。

- 自由度（`n - n_entities - k`、単純な`n-k`ではない）→ 検定統計量・adjusted R²・
  AIC/BICすべてに波及
- パネル固有R²（within/between/overall、2章）はOLSに存在しない新規計算
- `cov_type`デフォルトのentity単位cluster化、HACのDriscoll-Kraay別実装（3章）

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
`PanelOLS`の`cov_type="kernel"`でDriscoll-Kraay型SE（3章）の検証もカバーできる。
`pyfixest`は既存方針（`docs/spec/ols-spec.md`／`testing-policy.md`、HC2/HC3の実装バグにより
精度検証には使わない）を踏襲し、性能比較（実行時間・メモリ）のみに使う。

### 5.2 Rクロスチェックパッケージ

- **RE**: `plm`（`model = "random"`）。
- **FE**: `fixest`（`benchmark/panel/`配下の`run_fixest_benchmark.R`）。`fixest`自体は
  `.devcontainer/Dockerfile`に既にインストール済み。

### 5.3 ハウスマン検定の参照値（例外規定）

`linearmodels`にはハウスマン検定の専用メソッドが無い。この場合、**通常の「Python主リファレンス
＋Rクロスチェック」の2系統検証の例外**として、**Rの`plm::phtest`のみを参照値とする**ことを
許容する。`testing-policy.md`の「一部の統計量だけRクロスチェックを省略しない」という原則から
意図的に外れる例外であることをテスト実装時のコメント・ドキュメントに明記する。

### 5.4 許容誤差

既存方針（相対誤差1e-8を基本）を維持する。反復計算を伴う手法で乖離が大きい場合は、
OLSのHAC（実測乖離に基づき1e-2に緩和した前例、`testing-policy.md`）と同様、実装・テスト後の
実測値に基づいて個別に緩和を検討する（先に緩めない）。
