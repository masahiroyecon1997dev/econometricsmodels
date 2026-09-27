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
  | `time` | `str \| None` | `None` | `cov_type="dk"`時のDK時系列順序専用（`"dk"`では必須）。RE自身の準偏差変換・ハウスマン検定には使わない。`"dk"`以外で指定すると`ValidationError` |
  | `cluster` | `str \| None` | `None` | `cov_type="cluster"`時のグループキー列名。省略時は`entity`をそのまま使う。他の`cov_type`で指定すると`ValidationError` |
  | `dk_bandwidth` | `int \| None` | `None` | DK HACのバンド幅。省略時は自動計算。`cov_type="dk"`以外で指定すると`ValidationError` |

- **`FEOptions`と異なり`dk_time`が無い**: `ReCovType::Dk`は`FeCovType::Dk`と違い`time`
  オーバーライドフィールドを持たない。REは2-way構造自体を持たないため、「2-way FEの固定効果
  構造」と「DK HACの時系列粒度」を分離する必要が無く、`REOptions.time`1フィールドが
  「HAC時系列順序」専用になる（ハウスマン検定は常に1-way比較で`time`に依存しない、3.7節）。
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
- 欠損値（NaN/無限大）は常にエラー。

## 2. 結果構造体

`REResult`（`#[pyclass]`）が公開する項目: `params` / `std_errors` / `test_stats`（**t検定**） /
`p_values` / `conf_lower` / `conf_upper` / `param_names`（`param_names[0]`は常に`"const"`） /
`residuals` / `dep_var_name` / `n_obs` / `df_resid` / `df_model` / `n_entities` / `cov_type` /
`f_statistic` / `f_p_value` / `log_likelihood` / `aic` / `bic` / `r_squared_within` /
`r_squared_between` / `r_squared_overall` / `hausman_statistic` / `hausman_p_value` /
`hausman_df`。

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

R `plm`のデフォルト（`random.method="swar"`）・Python `linearmodels.RandomEffects`の実装、
いずれもSwamy-Arora相当。

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
数式自体はFE実装時から変更しておらず、`extra_df`・レバレッジ配列を引数で受け取る汎用実装の
ため呼び出し側（REは`extra_df=0`固定・`leverage_within`）を差し替えるだけで再利用できる。

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

### 3.5 F統計量

傾き係数`df_model - 1`個（定数項を除く）が同時にゼロという帰無仮説の検定。

**`wald_test_last_columns`（`cov_params`の部分行列を反転するWald検定、FEの`wald_f_test`
再利用と同型の発想）は使えない**: 不均衡パネル（θ_iがエンティティごとに異なる）データで
`linearmodels.RandomEffects.fit().f_statistic`と数値が一致しない。原因は`linearmodels`の
`_PanelModelBase._f_statistic`のソース確認で判明——「定数項を除く」際の比較対象
（`weps_const`）を、実際にモデルに含まれる変換済み定数列（`1-θ_i`、エンティティごとに
異なる）ではなく**変換済みyの単純平均**（`y - mean(y)`）で計算している。この定義は、
定数列が全観測で同一の値（バランスパネルでθが全エンティティ共通）でない限りWald検定
（部分行列反転）とは一致しない。`linearmodels`が主リファレンスのため、`ReEstimator::fit`は
この定義（変換済みyの単純平均を基準にした古典的SST/SSR比較）を直接実装している。

- `residual_ss<=0.0`（完全な当てはめ）なら`linearmodels`と同じくF統計量を`0.0`とする。
  傾き係数が0個（`df_model==1`）ならNaN。
- **F統計量は負値になりうる**（`linearmodels`自身でも極端な不均衡パネルで実地確認済み）。
  `total_ss`（変換済みyの単純平均基準）は実際にモデルに含まれる変換済み定数列に対する
  直交性を持たないため、教科書的な入れ子モデル比較（`total_ss >= residual_ss`保証）とは
  異なり`total_ss < residual_ss`になりうる。
- 検証は`linearmodels`（`cov_type="unadjusted"`）の`f_statistic`と直接数値比較。

### 3.6 パネル固有R²

FEの`r_squared_between`/`r_squared_overall`をそのまま流用できず、RE独自に実装している
（無理な共通化はしない方針）。差異は2点:

- **TSSの中心化有無**: REは`has_constant=True`のため中心化TSS（`Σ(y-ȳ)²`）を使うが、FEは
  `has_constant=False`扱いのため非中心化TSSを使う。
- **当てはめ値への切片`β0`の有無**: FEは固定効果自体を含めない「弱いR²」を意図的に採用する
  が、REは真の切片係数`β0`を含めて当てはめる（`fitted = β0 + Σ_j x_j・β_j`）。

`r_squared_within`はFEと同じ定義（θ=1固定の通常のwithin変換、REの`θ_i`とは無関係）。
`linearmodels`の早期リターン（`has_constant`かつ傾き係数0個なら3種とも`0.0`）に倣い、
`df_model==1`なら3種とも`0.0`とする。`r_squared_between`が負値になりうる（`f_statistic`と
同型の性質）ことも実地確認済み。検証は`linearmodels`（`cov_type="unadjusted"`）の
`rsquared_within`/`rsquared_between`/`rsquared_overall`と直接数値比較。

### 3.7 ハウスマン検定

`hausman_statistic`/`hausman_p_value`/`hausman_df`は`RE.fit()`内で自動計算され、`REResult`
にのみ含まれる（`FEResult`には追加しない）。**回帰ベース（補助回帰）版**（Wooldridge (2010)
10.7.3節、`plm::phtest(method = "aux", effect = "individual")`相当）。

- **方式**: 準偏差変換済みの`y*`を、定数項（**準偏差変換前の`1`**）・準偏差変換済みの傾き
  `X*`・within変換済みの`X̃`にpooled OLS（classical共分散）で回帰し、`X̃`の係数`k`個が同時に
  ゼロというWald検定を行う。統計量は`k × F`（χ²版、`plm`と一致させるため）、p値は
  `χ²_k.sf(stat)`、`hausman_df = k`（傾き係数の数）。統計量は構造的に非負で、二次形式版の
  `abs()`や非正定値の問題が生じない。バランスパネルでは共通σ²を使った古典的ハウスマン
  検定と数値的に一致する。
- **定数項の扱い**: `plm`は`reX`から定数列を除き補助回帰に未変換の`1`を足す。バランス
  パネルでは`θ_i`が全個体共通のため変換済み定数列を使う版と同値だが、不均衡パネルでは
  値が異なる。本実装は`plm`に合わせる。
- **比較は常に1-way**（個体効果のみ、RE本体と同じ構造）。`X̃`はRE本体が分散成分推定に
  使う1-way FEのwithin変換から得る。`REOptions.time`（`cov_type="dk"`専用）の有無・
  `cov_type`の選択は結果に影響しない。2-wayのハウスマン検定は2-way REの実装時に
  改めて検討する。
- **classical固定（`cov_type`非連動）**: 補助回帰は常にclassical共分散を使う（帰無仮説の
  もとでREが完全に効率的＝等分散・系列無相関を前提とする古典的ハウスマン検定）。RE本体の
  `cov_type`既定は`"cluster"`だがハウスマン検定だけはclassicalである点に注意。robust版
  （補助回帰にcluster-robust共分散を使う等）は別途検討する。
- **`None`フォールバック**: 比較対象の傾き係数が0個（`x=[]`は拒否されるため通常発生
  しない）、補助回帰のランク落ち、Wald検定の特異性。いずれも`hausman_statistic`等を
  `None`にしたうえでRE本体の結果は正常に返す（診断情報の欠落だけに留める）。なお
  時間不変変数・singleton entity等による内部1-way FE推定の失敗は、分散成分（σ_ε²）推定が
  先に失敗するため`RE.fit()`自体が失敗する。
- **旧実装（削除済み）**: `Var(β_FE)-Var(β_RE)`の二次形式に`abs()`を適用する版は、
  非正定値の問題を隠す（負の二次形式が「大きな正の統計量」に化ける）ため置き換えた。

### 3.8 engine_pybind: エラー変換

`fe-spec.md`「engine_pybind: エラー変換」と同じ`PanelError` → `PyErr`変換（`panel_error_to_
pyerr`、FE/RE共有）を使う。RE固有の追加バリアントは`BetweenRegressionFailed`（between回帰の
失敗）・`QuasiDemeanedRegressionFailed`（最終的な準偏差変換後の委譲回帰の失敗）で、いずれも
`ComputationError`に分類される。

## 4. テスト

- Python主リファレンス: `linearmodels.RandomEffects`（`cov_type="unadjusted"`/`"clustered"`/
  `"kernel"`等）。Rクロスチェック: `plm`（`model="random"`、
  `benchmark/panel/run_plm_benchmark.R`）。
- 許容誤差: Classical/HC1/Cluster/HACは`linearmodels`と相対誤差`1e-9`で数値完全一致。
  HC2/HC3は`plm::vcovHC`とのクロスチェック水準（分散成分推定法が僅かに異なるため）。
- **ハウスマン検定は`plm::phtest(method = "aux", effect = "individual")`のみを参照値と
  する例外規定**（`linearmodels`のソースに`hausman`という文字列が一切登場せず専用実装が
  無いことを確認済み。通常の「Python主リファレンス＋Rクロスチェック」の2系統検証の例外）。
  バランスパネルでは機械精度で一致し、不均衡パネルではSwamy-Arora分散成分（σ_u²）の
  推定式の差で数％ずれるため、シナリオ別の許容誤差で比較する（3.7節）。
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
- 将来的に`cov_type`と連動するrobust版Hausman検定（Wooldridgeの回帰ベース検定等）は
  未実装（v1はフィールド名・置き場所に拡張余地を残すのみ）。
