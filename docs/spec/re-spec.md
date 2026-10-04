# RE 仕様書

RE（Swamy-Arora GLSによる変量効果パネル回帰）の確定済み仕様。`engine/src/panel/re.rs`（共通基盤
は`engine/src/panel/common.rs`）・`engine_pybind/src/panel/re.rs`・
`python_package/econometricsmodels/panel/re.py`として実装済み。FE/RE共通の設計判断（`entity`/
`time`の引数設計、結果フィールドの共通コア、`cov_type`のサポート対象・デフォルト、内部実装の
共通化方針、リファレンス実装・テスト方針）は
[`panel-common.md`](./panel-common.md)を参照し、本ドキュメントには
RE固有の内容のみを記載する。FEとの共有範囲は[`fe-spec.md`](./fe-spec.md)も参照。

## 1. API引数

3層構成: `RE(data, y, x, entity, options).fit() -> REResults`（python_package）→
`fit_re(data, y, x, entity, options) -> REResult`（engine_pybind）→ `ReEstimator::fit`
（engine、準偏差変換したデータを`OlsEstimator`へ委譲）。

- `y: str`、`x: list[str]`はOLSと同じ。`entity: str`は独立の必須引数。
- `REOptions`（`#[pyclass]`）:

  | フィールド | 型 | デフォルト | 説明 |
  |---|---|---|---|
  | `cov_type` | `str` | `"cluster"` | `"classical"` / `"hc1"`〜`"hc3"` / `"cluster"` / `"dk"`（大小無視）。`"hc0"`は非対応 |
  | `confidence_level` | `float` | `0.95` | |
  | `dk_time` | `str \| None` | `None` | `cov_type="dk"`時のDK時点列（`FEOptions.dk_time`と同名・同じ意味。`"dk"`では必須）。RE自身の準偏差変換・ハウスマン検定には使わない。`"dk"`以外で指定すると`ValidationError` |
  | `cluster` | `str \| None` | `None` | `cov_type="cluster"`時のグループキー列名。省略時は`entity`をそのまま使う。他の`cov_type`で指定すると`ValidationError` |
  | `dk_bandwidth` | `int \| None` | `None` | DK HACのバンド幅。省略時は自動計算。`cov_type="dk"`以外で指定すると`ValidationError` |

- **`FEOptions.dk_time`と同名・同じ意味の`dk_time`を持つ。`time`という名前のオプションは
  持たない**: REは2-way構造自体を持たず、FEの`time`（2-way固定効果の時間次元）に相当する
  ものが無いため、`time`と呼ぶと誤解を招く。DKの時点列は`REOptions.dk_time`の1フィールドで
  受け取る（ハウスマン検定は常に1-way比較で`dk_time`に依存しない、3.7節）。将来2-way REを
  実装するときは、FEと同じ意味の`time`を改めて導入する。
- **`x`は空リストを許容しない**: `x=[]`は「説明変数を一切投入しない、分散成分（σ_ε²・
  σ_u²、ICC）のみを推定するnullモデル」として単独で意味を持つ標準的なユースケース
  （マルチレベルモデルの"null model"）だが、他手法（FE・OLS/WLS/Logit/Probit/IV）
  との一貫性を優先し`validate_x_non_empty`で拒否する。nullモデル・ICC推定のサポート自体は
  別途検討中（5章参照）。
- **REはentity方向のみ（2-way REはv1スコープ外）**なので、FEの1-wayと同じ扱いで不均衡パネル
  も無条件でサポートする。
- **singletonエンティティの扱いがlinearmodelsより厳格**: REのσ_ε²推定は内部で1-way FE推定を
  呼び出しその残差分散を再利用する設計（3.4節）のため、`FeInput`と同じsingleton検証を
  継承し`T_i=1`のエンティティを含むデータは`PanelError::SingletonGroup`で失敗する。一方
  `linearmodels.RandomEffects`自身はsingletonエンティティを問題なく処理できる（該当行の
  entity-demean値が単に0になるだけで、REの数学的定義自体はsingletonを許容するため）。この
  相違は「σ_ε²の推定はFEのwithin回帰の残差分散をそのまま利用する」という設計の意図的な
  帰結であり、ベンチマーク/テストフィクスチャは全エンティティ`T_i>=2`を確保して作成する。
- between回帰は切片+傾き`k`個で`k+1`パラメータのため`n_entities > k+1`が必要（`n_entities <=
  k+1`だと`PanelError::BetweenRegressionFailed`。`linearmodels`自身も`n_entities=k+1`ちょうど
  で`ZeroDivisionError`になることを実地確認済み）。
- 欠損値（null・NaN/無限大）は常にエラー（方針は[`docs/guide/validation.md`](../guide/validation.md)）。

## 2. 結果構造体

`REResult`（`#[pyclass]`）が公開する項目: `params` / `std_errors` / `test_stats`（**t検定**） /
`p_values` / `conf_lower` / `conf_upper` / `param_names`（`param_names[0]`は常に`"const"`） /
`residuals` / `dep_var_name` / `n_obs` / `df_resid` / `df_model` / `n_entities` / `cov_type` / `dk_bandwidth_used` /
`f_statistic` / `f_p_value` / `log_likelihood` / `aic` / `bic` / `r_squared_within` /
`r_squared_between` / `r_squared_overall` / `hausman_statistic` / `hausman_p_value` /
`hausman_df`。

- **`dk_bandwidth_used`**: `cov_type="dk"`のとき実際に使われたバンド幅（FEと同じ意味、`fe-spec.md`
  「結果構造体」参照）。`dk`以外は`None`。ハウスマン検定の補助回帰が内部で解決するバンド幅は
  本体と同じ値だが、この項目はRE本体の`fit()`で解決した値を報告する。
- **REは切片を持つ**ため`param_names[0]`が常に`"const"`になる（FEはwithin変換で切片が構造的
  に消えるため無い）。
- **`estimator()`（内部委譲した`OlsEstimator`）とRE自身のgetterの使い分けはFEと同型だが
  `aic`/`bic`の扱いだけ異なる**: `params`/`param_names`/`residuals`/`dep_var_name`/`n_obs`/
  `log_likelihood`/`aic`/`bic`は`estimator()`からそのまま取得する——`ReEstimator`自身は
  `aic()`/`bic()`メソッドを持たない（REの`df_model`が`OlsInput::k()`と自動的に一致する設計
  のため、FEのような独自再計算が不要、後述「`df_resid`」参照）。`std_errors`/`test_stats`/
  `p_values`/`conf_lower`/`conf_upper`/`df_resid`/`df_model`/`f_statistic`/`f_p_value`/
  `r_squared_*`/`hausman_*`はRE自身のgetterから取得する。
- **RE自身に`fixed_effects()`のような追加メソッドは無い**: ハウスマン検定は`fit()`内で
  自動計算済みの値をそのまま`REResult`のフィールドとして持つだけで済む（FEの
  `fixed_effects()`のような別メソッド化は不要）。`REResult`に`estimator`のような非公開
  フィールドも無く、`#[derive(Clone)]`を問題なく維持できる。
- `summary()`は実装しない。python_package層（`REResults`）の`coef_table()`はFEと同じキー
  構成（`param`/`coef`/`std_err`/`test_stat`/`p_value`/`conf_lower`/`conf_upper`）。

## 3. 内部実装の計算仕様

### 3.1 分散成分の推定（Swamy-Arora法）

Python `linearmodels.RandomEffects`の実装（Swamy-Arora）に準拠する。R `plm`の
デフォルト（`random.method="swar"`）も同名だが、不均衡パネルでは別の推定量になる
（下記「plmとの既知の実装差」）。

- **σ_ε²（idiosyncratic variance）**: 内部1-way FE推定（`FeEstimator::fit(OneWay,
  Classical)`）のwithin回帰残差平方和／`FeEstimator::df_resid()`。
- **σ_u²（individual variance）**: between回帰（エンティティ平均に対する`OlsEstimator::
  fit(include_intercept=true)`）のSSR、調和平均`t_bar = n_entities / Σ(1/T_i)`を使う標準式
  `max(0, ssr/df_resid - σ_ε²/t_bar)`。
- **`k`規約についての実装上の注記**: `linearmodels`ソースの式は`nvar`（切片を含む列数）
  表記だが、このプロジェクトの`k`規約（傾き係数のみ、切片を含まない）では、内部1-way FE
  推定が返す`df_resid()`（σ_ε²用）とbetween回帰が返す`nobs()-k()`（σ_u²用）をそのまま
  使えば`+1`/`-1`の手計算なしに自動的に一致する（乱数・手動データ複数ケースで
  `linearmodels.RandomEffects`との数値完全一致を実地検証済み）。
- **plmとの既知の実装差（不均衡パネル）**: バランスパネルでは`plm`と分散成分が一致する
  （機械精度）。不均衡パネルでは、`plm`は`dfcor=3`固定（他の値を指定すると
  `dfcor should equal 3 for unbalanced panels`のエラーになり、オプションでは変更できない）で、
  `T_i`のトレース項（`Σ T_i²`等）を含むモーメント方程式の連立を解いてσ_ε²・σ_u²を同時に
  求める。本実装（linearmodels準拠）はbetween回帰のSSRと調和平均`t_bar`の単純な式でσ_u²を
  求める。σ_ε²は両者で一致するが、σ_u²は不均衡で数％〜10％程度異なり、θ・係数・標準誤差・
  ハウスマン統計量に伝播する（実測: 8個体・21観測でσ_u² 0.819対1.129）。plm側の
  オプションで本実装の式を再現することはできない。本実装は`linearmodels`との
  機械精度一致（Classical/HC1/Cluster/HAC）を優先し、plmの式は採用しない。plm互換の
  推定法の提供は、具体的な要望が出た時点で改めて検討する。
- linearmodelsが提供する`small_sample`補正（不均衡パネル向けのtraceベースの追加調整、
  デフォルト`False`）はv1では実装しない（linearmodelsのデフォルト挙動に合わせる）。

### 3.2 θ（準偏差変換の重み）

`θ_i = 1 - sqrt(σ_ε² / (T_i・σ_u² + σ_ε²))`（Baltagiの教科書通りの式、`compute_theta`が
エンティティごとに計算）。REはentity方向のみのため`T_i`を直接使うこの式は教科書レベルで
不均衡対応済みであり、FEの2-wayのような反復アルゴリズムは不要。

`quasi_demean_transform`（`quasi_demean_column`をθパラメータ化した共通の準偏差変換関数、FEは
全エンティティに`θ_i = 1.0`を渡す特殊ケースとして扱う）を`y`・各`x`列に適用する。**REは
切片を持つ**ため、切片復元用の定数列（すべて1.0）にも同じθを適用してから
`OlsEstimator::fit(include_intercept=false)`に渡す（`linearmodels.RandomEffects.fit()`の
ソースで、`exog`に含まれる定数列自体も他の説明変数と同じ`quasi_demean`処理を受けている
ことを確認済み）。単純に`include_intercept=true`を使うと変換されない生の`1.0`列になって
しまうため、この手動追加が必要。

### 3.3 `df_resid`/`df_model`

**`df_resid = n - k`**（FEの`n - n_entities - k`とは異なる式。REはGLS変換でありFEのように
個体ダミー相当の自由度を消費しない）。`df_model = k`（変換済み定数列を含む設計行列の全列数、
`OlsInput::k()`）。

`OlsInput::k()`は`include_intercept`フラグの値に関わらず設計行列の実際の列数を返すため、
`ReEstimator::fit`が`include_intercept=false`で変換済み定数列を手動追加していても、
`estimator().input().k()`は`linearmodels`の`wx.shape[1]`（`df_resid = wy.shape[0] -
wx.shape[1]`）と自動的に一致する。この副産物として`estimator().std_errors()`/`test_stats()`/
`p_values()`/`conf_lower()`/`conf_upper()`/`aic()`/`bic()`は委譲した時点で既に正しいRE
推定量になっている（`linearmodels.HomoskedasticCovariance`の`cov_type="unadjusted"`実装を
確認済み）ため、FEのように`cov_params`を独自に作り直す必要は無い。

一方`estimator().f_statistic()`/`f_p_value()`・`r_squared()`/`adj_r_squared()`はこの時点でも
正しくない（`has_intercept()==false`扱いになるため）。正しいF統計量・パネル固有R²は
下記3.4節・3.5節でRE独自に計算し直す。

### 3.4 `cov_type`対応

`linearmodels.RandomEffects.fit()`のソース確認で判明した重要な事実——**REは`cov_type`に
よらず常に`extra_df=0`**（FEのような`neffects`・「クラスター変数がentityを包含するか」の
条件分岐が一切不要）。REの変換済み設計行列には「省略された固定効果ダミー」が無く、切片も
含め全パラメータが実際に列として含まれているため、HC2/HC3のレバレッジもFEのLSDV相当の
フルレバレッジ補正ではなく、素の`h_ii = x_i(X'X)⁻¹x_i'`（実際の設計行列に対して直接計算する
だけ）で足りる。

FE実装時（`panel::fe`）のcov_type計算関数（`design_matrix_from_columns`・`xtx_inverse`・
`leverage_within`・分類/HC/cluster/DriscollKraay計算）は`common.rs`にFE/RE共有で移設済み——
数式自体はFE実装時から変更しておらず、`k_correction`・レバレッジ配列を引数で受け取る汎用実装の
ため呼び出し側（REは常に`k_correction=df_model`・`leverage_within`）を差し替えるだけで
再利用できる。

`ReEstimator`は`cov_type`に関わらず常に自前のフィールド（`std_errors`/`test_stats`/`p_values`/
`conf_lower`/`conf_upper`/`cov_type`）を保持する（Classical/HC1のような「`OlsEstimator`
委譲でも数値的に正しい」cov_typeであっても、Cluster/HACとの非対称なAPIを避けるため）。

- **HC2/HC3の参照実装が無い**: `linearmodels`は`RandomEffects`・`PanelOLS`どちらも
  "heteroskedastic"（HC1相当）しか持たない。REは`plm::vcovHC(fit, method="white1",
  type="HC2"/"HC3")`をクロスチェックに使う。`plm`は変量効果の分散成分推定法が
  `linearmodels`と僅かに異なる（点推定自体が僅かに違う）ため、Classical/HC1/Cluster/HAC
  ほどの精度ではなくクロスチェック水準で検証する（4章参照）。
- `ReCovType::Cluster`の`q`（傾き係数の数、切片を除く）は`df_model - 1`。`ReCovType::Dk`は
  `time`オーバーライドフィールドを持たない（RE自身が2-way構造を持たないため）。
- **【linearmodels方式からfixest（R）・Stata型に変更】**: `Cluster`は当初
  `linearmodels`/`plm`に合わせStata流`(G/(G-1))×((n-1)/(n-k))`補正を使わずに独自計算して
  いたが、fixest・Stataの`xtreg,re vce(cluster)`利用者が期待する値と一致しないため変更した。
  REは`extra_df`が常に`0`（固定効果ダミーが設計行列に無くFEのようなネスト判定が不要）
  なため、`K`（`panel_cluster_cov_params`の`(n-1)/(n-K)`）は単純に`df_model`をそのまま
  渡せばよい——`OlsEstimator`自身の`cluster_cov_params`と数式的に同一になる
  （`plm::vcovHC(type="sss")`と同じ式）。`Dk`も同様にfixestの`vcov="DK"`に合わせ、
  `K=df_model`・`G`相当は`t_periods`を使う補正に変更した（FEの`Dk`と同じ式、3.3節参照）。
- **t検定・信頼区間の自由度（`df_inference`）は`cov_type=Cluster`のとき`G-1`、`Dk`のとき
  `t_periods-1`に切り替える**（fixestの`ssc()`既定`t.df="min"`、FEと同じ
  パターン）。それ以外（Classical/HC1-3）は`df_resid`のまま。**F統計量（3.5節）も
  同じ`cov_params`・`df_inference`のWald検定**（FEと同じ）。

### 3.5 F統計量

傾き係数`df_model - 1`個（定数項を除く）が同時にゼロという帰無仮説のWald F検定
（`F = β'V⁻¹β / q`、`q = df_model - 1`）。**FEと同じく`cov_type`に連動する**: `cov_type`別に
計算した`cov_params`（3.4節）の傾き係数部分行列を`wald_f_test`（OLS本体の検定の再利用、
サンドイッチ計算を複製しない）で検定し、分母自由度`f_df_denom`は`df_inference`
（`cluster`で`G-1`、`dk`で`t_periods-1`、それ以外は`df_resid`）で、t検定・信頼区間の
`stat_df`と常に一致する（`tests/test_test_dfs.py`が全`cov_type`で固定している）。`classical`のとき
`plm::pwaldtest(test = "F")`の既定（古典的分散共分散行列・`df.residual`）、ロバストの
ときは`pwaldtest(test = "F", vcov = ...)`の統計量と一致する。傾き係数が0個（`df_model==1`）なら
NaN。失敗（共分散部分行列のほぼ特異性）は`PanelError::FTestFailed`。

**`linearmodels`のSST/SSR方式ではなくplm定義を採用した理由**: `linearmodels.RandomEffects.
fit().f_statistic`（`_PanelModelBase._f_statistic`）は、「定数項を除く」際の比較対象
（`weps_const`）を、実際にモデルに含まれる変換済み定数列（`1-θ_i`、エンティティごとに
異なる）ではなく**変換済みyの単純平均**（`y - mean(y)`）で計算する。このため教科書的な
入れ子モデル比較（`total_ss >= residual_ss`）の保証が無く、**極端に不均衡なパネル
（`T_i`の差が大きい）で負値になる**（`linearmodels`自身でも実地確認済み、T_i={2,2,15}等）。
Wald二次形式は`V`が正定値である限り負値にならず、FEのF統計量（Wald検定）とも揃う。
バランスパネル（`θ`が全エンティティ共通）では両定義が一致し、不均衡パネルでは一致しない。

- 検証は`plm::pwaldtest(test = "F", vcov = ...)`の統計量（hc2/hc3/cluster/dk、
  `tests/panel/test_re_crosscheck.py`）と`linearmodels`の`f_statistic_robust`
  （classical/hc1、`tests/panel/test_re_reference.py`）との数値比較。バランスパネルは機械精度で
  一致する。`linearmodels`の`f_statistic_robust`は不均衡パネルでも機械精度で一致する
  （点推定が一致するため）。`plm`との不均衡パネルは、Swamy-Arora分散成分の推定差
  （`θ`の差）で`cov_type`別に最大約0.1〜0.7%の差が出るため専用の許容誤差を使う
  （`tests/_tolerances.py`の`re_crosscheck.rtol_unbalanced_f`）。
  `linearmodels`の`res.f_statistic`（SST/SSR方式、`cov_type`非依存）はバランスパネルの
  classicalでのみ一致するため比較に使わない。
- 傾き係数が1個のケースは「1自由度のF検定は両側t検定と代数的に等価」
  （`f_statistic = test_stat²`）を全`cov_type`でエンジンの単体テストが固定している。

### 3.6 パネル固有R²

FEの`r_squared_between`/`r_squared_overall`をそのまま流用できず、RE独自に実装している
（無理な共通化はしない方針）。差異は2点:

- **TSSの中心化有無**: REは`has_constant=True`のため中心化TSS（`Σ(y-ȳ)²`）を使うが、FEは
  `has_constant=False`扱いのため非中心化TSSを使う。
- **当てはめ値への切片`β0`の有無**: FEは固定効果自体を含めない「弱いR²」を意図的に採用する
  が、REは真の切片係数`β0`を含めて当てはめる（`fitted = β0 + Σ_j x_j・β_j`）。

`r_squared_within`はFEと同じ定義（θ=1固定の通常のwithin変換、REの`θ_i`とは無関係）。
`linearmodels`の早期リターン（`has_constant`かつ傾き係数0個なら3種とも`0.0`）に倣い、
`df_model==1`なら3種とも`0.0`とする。`r_squared_between`が負値になりうる（`linearmodels`のSST/SSR方式のF統計量と
同型の性質。F統計量自体は3.5節のWald形式で負値にならない）ことも実地確認済み。検証は`linearmodels`（`cov_type="unadjusted"`）の
`rsquared_within`/`rsquared_between`/`rsquared_overall`と直接数値比較。

### 3.7 ハウスマン検定

`hausman_statistic`/`hausman_p_value`/`hausman_df`は`RE.fit()`内で自動計算され、`REResult`
にのみ含まれる（`FEResult`には追加しない）。**回帰ベース（補助回帰）版**（Wooldridge (2010)
10.7.3節、`plm::phtest(method = "aux", effect = "individual")`相当）。

- **方式**: 準偏差変換済みの`y*`を、定数項（**準偏差変換前の`1`**）・準偏差変換済みの傾き
  `X*`・within変換済みの`X̃`にpooled OLSで回帰し、`X̃`の係数`k`個が同時に
  ゼロというWald検定を行う（共分散はRE本体の`cov_type`に連動、下記）。統計量はWald統計量
  （`k × F`、χ²版、`plm`と一致させるため）、p値は
  `χ²_k.sf(stat)`、`hausman_df = k`（傾き係数の数）。統計量は構造的に非負で、二次形式版の
  `abs()`や非正定値の問題が生じない。バランスパネルでは共通σ²を使った古典的ハウスマン
  検定と数値的に一致する。
- **定数項の扱い**: `plm`は`reX`から定数列を除き補助回帰に未変換の`1`を足す。バランス
  パネルでは`θ_i`が全個体共通のため変換済み定数列を使う版と同値だが、不均衡パネルでは
  値が異なる。本実装は`plm`に合わせる。
- **比較は常に1-way**（個体効果のみ、RE本体と同じ構造）。`X̃`はRE本体が分散成分推定に
  使う1-way FEのwithin変換から得る。`REOptions.dk_time`（`cov_type="dk"`専用）の有無は
  結果に影響しない（`dk_time`は`cov_type="dk"`の時点列にのみ使う）。2-wayのハウスマン検定は2-way REの実装時に
  改めて検討する。
- **`cov_type`連動**: 補助回帰のWald検定の共分散はRE本体の`cov_type`に対応させる
  （専用オプションは設けない）。既定の`cov_type="cluster"`ではcluster-robust版（Wooldridgeの
  robust Hausman検定）になり、`cov_type="classical"`では帰無仮説のもとでREが完全に効率的
  （等分散・系列無相関）という前提の古典版になる。前提が崩れる場合にclassical版は
  サイズが歪みうる。

  | RE本体の`cov_type` | 補助回帰の共分散 | `plm`の参照 |
  |---|---|---|
  | `classical` | `OlsEstimator`のClassical | `phtest(method = "aux")` |
  | `hc1`/`hc2`/`hc3` | `OlsEstimator`のHc1/Hc2/Hc3 | `vcovHC(method = "white1", type = "HC1"/"HC2"/"HC3")` |
  | `cluster` | `OlsEstimator`のCluster（`cluster`省略時はentity） | `vcovHC(method = "arellano", type = "sss")` |
  | `dk` | Driscoll-Kraay（`panel_driscoll_kraay_cov_params`を補助回帰に適用） | `vcovSCC(maxlag = dk_bandwidth, type = "HC1")` |

  - **【補正式の混在を解消済み】**: 当初、補助回帰は`OlsEstimator`の補正式
    （Cluster: `G/(G-1)·(n-1)/(n-k)`、Stata・R型）を使う一方、RE本体のcluster標準誤差は
    `linearmodels`型の`n/(n-k)`のみだったため、同一の`REResult`内で補正式が混在していた
    （当時の既知の制約）。RE本体のCluster/Dkもfixest（R）・Stata型
    （`K=df_model`を使う`G/(G-1)·(n-1)/(n-K)`、3.4節参照）に揃えたため、この混在は
    解消済み——補助回帰・RE本体のいずれもStata/fixest型の補正を使う（補助回帰は
    RE本体とは別の回帰（説明変数`2k+1`個）である点は変わらない）。
  - **`cluster`にentity以外を指定した場合**: `plm`がクラスターにできるのはgroup/timeのみで
    リファレンスが無い。式自体はOLS本体の`CovType::Cluster`（statsmodelsで検証済み）と同じ
    ため計算して値を返す（`plm`との照合はentityクラスターのみ）。
  - **`dk`のバンド幅**: RE本体と同じ解決規則（`dk_bandwidth`、省略時
    `floor(4*(T/100)^(2/9))`）。スケールは補助回帰の`(t_periods/(t_periods-1))×
    ((n-1)/(n-k_aux))`（RE本体のDkと同じfixest型補正に変更、Wald検定の分母自由度も
    `t_periods-1`に揃える）、Bartlett重みは同形。
  - 統計量の定義（Wald統計量）は`cov_type`によらず一定で、Wu-Hausman（IV、F版）とは異なる。
  - DKは時点数`T`→∞の漸近論に基づくため、`T`が短いと検定サイズが歪みうる。
  - `cluster`（既定）ではクラスター数`G`が補助回帰の傾き係数の数`2k`を超える必要がある
    （RE本体は`G > k`で足りるため、RE本体が成功してもfitが失敗しうる）。
- **`None`になるのは比較対象の傾き係数が0個の場合のみ**（`x=[]`は拒否されるため通常発生
  しない）。補助回帰のランク落ち・Wald検定の特異性など、計算自体が成立しない場合は
  `hausman_*`を`None`にせず`RE.fit()`が失敗する（設計行列の多重共線性でエラーにするのと
  同じ方針）。ロバスト共分散が構造的に特異になる場合が典型で、`cluster`ではクラスター数`G`が
  補助回帰の傾き係数の数`2k`以下（`ValidationError`、OLSの`InsufficientClustersForInference`と
  同じ事前検証）、`dk`では時点数`T`が検定対象の`k`個以下（`T <= k`、`ValidationError`。
  DK共分散のrankは`T-1`以下のため。FEのF検定と同じ事前検証
  `PanelError::InsufficientDkPeriodsForInference`）。RE本体は成功する入力でも、`cov_type="cluster"`（既定）で
  `G <= 2k`の場合はfitが失敗するため、`cov_type="classical"`等を指定するか
  クラスター数を増やす。時間不変変数・singleton entity等による内部1-way FE推定の失敗は、
  分散成分（σ_ε²）推定が先に失敗するため`RE.fit()`自体が失敗する。
- **旧実装（削除済み）**: `Var(β_FE)-Var(β_RE)`の二次形式に`abs()`を適用する版は、
  非正定値の問題を隠す（負の二次形式が「大きな正の統計量」に化ける）ため置き換えた。

### 3.8 engine_pybind: エラー変換

`fe-spec.md`「engine_pybind: エラー変換」と同じ`PanelError` → `PyErr`変換（`panel_error_to_
pyerr`、FE/RE共有）を使う。RE固有の追加バリアントは`BetweenRegressionFailed`（between回帰の
失敗）・`QuasiDemeanedRegressionFailed`（最終的な準偏差変換後の委譲回帰の失敗）で、いずれも
FEの`WithinRegressionFailed`と同じく内側の`LeastSquaresError`の分類に従う（特異行列等なら
`ComputationError`、それ以外は`ValidationError`）。例えばbetween回帰は`n_entities <= k + 1`
だと内側が観測数不足になり`ValidationError`になる。

## 4. テスト

- Python主リファレンス: `linearmodels.RandomEffects`（点推定・`cov_type="unadjusted"`等）。
  Rクロスチェック: `plm`（`model="random"`、`benchmark/panel/run_plm_benchmark.R`）。
- **【例外】** `cov_type="cluster"`/`"dk"`の標準誤差・推論統計量は
  `linearmodels`ではなく`plm`を正とする（`linearmodels`独自の`n/(n-k)`補正から
  Stata・R型の補正（3.4節参照）に変更したため）。参照値は`plm::vcovHC(method = "arellano",
  type = "sss")`（cluster）・`plm::vcovSCC(maxlag = <bandwidth>, type = "sss")`（dk。バンド幅は
  本実装の既定式`floor(4*(T/100)^(2/9))`で求めた値を明示的に渡す）で、`plm`はz検定のため
  t統計量・p値・信頼区間は`plm`の標準誤差から本実装と同じt分布（自由度clusterは`G-1`、
  dkは`T-1`）で計算し直す（この自由度の選択自体は本実装と同じ規約の手計算で、独立検証が
  及ぶのは補正係数込みの標準誤差まで）。clusterの`G-1`のみ、plmの準偏差変換済みデータに
  statsmodelsのクラスターロバストOLS（`use_t=True`）を当てた値でも検証する
  （`re_statsmodels_cluster.json`、バランスパネルのみ）。dkの`T-1`はstatsmodelsに
  同じ規約がなく第2リファレンスがない。`classical`/`hc1`は引き続き`linearmodels`と数値一致で
  検証する。
  `cluster`にentity以外の列を指定する場合は、`plm::vcovHC`がgroup/timeしかクラスターに
  できないため、`plm`の準偏差変換済み設計行列・応答に`lm` + `sandwich::vcovCL(type = "HC1",
  cadjust = TRUE)`を当てた値を参照値にする（entityクラスターでは`vcovHC(arellano, sss)`と
  機械精度で一致を確認済み）。クラスター不均衡（サイズ[2,3,5,10,30,50]）と、境界の成功パス
  （ハウスマン検定の補助回帰が`G > 2k`を要するため`k=1`・`G=3`）を持つ。
- 許容誤差: Classical/HC1は`linearmodels`と相対誤差`1e-8`（`tests/_tolerances.py`の
  `re_reference`。実測は`1e-9`〜`1e-14`程度）で数値完全一致。Cluster/DK・
  HC2/HC3の`plm`クロスチェックは、バランスパネルでは機械精度で一致する（`1e-8`）。不均衡
  パネルのみ分散成分推定（Swamy-Arora）が`plm`とlinearmodels準拠の本実装で僅かに異なるため、
  統計量・cov_type別に実測へマージンを載せて緩める（係数`5e-3`、se・t・p値・信頼区間は
  cluster `5e-3`・hc2/hc3 `2e-2`・dk `5e-2`。`tests/_tolerances.py`の`re_crosscheck`参照。
  一律に緩めるとclusterの`G/(G-1)`欠落（seに約1.3%）を見逃すため分けている）。
- **ハウスマン検定は`plm::phtest(method = "aux", effect = "individual", vcov = ...)`のみを参照値と
  する例外規定**（`linearmodels`のソースに`hausman`という文字列が一切登場せず専用実装が
  無いことを確認済み。通常の「Python主リファレンス＋Rクロスチェック」の2系統検証の例外）。
  バランスパネルでは機械精度で一致し、不均衡パネルではSwamy-Arora分散成分（σ_u²）の
  推定式の差で数％ずれるため、シナリオ別の許容誤差で比較する（3.7節）。`cov_type`別
  （`benchmark/panel/run_plm_hausman_benchmark.R`）に全6種を照合し、ロバスト共分散が構造的に
  特異なシナリオ（`many_regressors`のcluster/dk）は`fit()`が`ValidationError`/`ComputationError`で
  失敗することを確認する。
- `aic`/`bic`は`estimator()`（内部`OlsEstimator`）からそのまま取得するため、FEのような
  Rクロスチェック限定の例外は無く`linearmodels`と直接比較できる。

## 5. 未実装・未対応

- **nullモデル・ICC推定のサポート**（`x=[]`、検討中）: 分散成分のみを推定する
  マルチレベルモデルの標準的なユースケースだが、v1は他手法との一貫性を優先し`x`の空リストを
  拒否している。
- **2-way RE**（v1スコープ外）: バランスパネル限定の2-way RE（Wallace-Hussain法・Amemiya法
  等のANOVA型閉形式推定量）・アンバランスパネルの2-way RE（教科書レベルの
  閉形式が存在せず、Wansbeek and Kapteyn (1989)の推定量が候補）は別issueで
  検討する。
