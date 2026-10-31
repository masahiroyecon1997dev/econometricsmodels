# FE 仕様書

FE（固定効果パネル回帰、within推定）の確定済み仕様。`engine/src/panel/fe.rs`（共通基盤は
`engine/src/panel/common.rs`）・`engine_pybind/src/panel/fe.rs`・
`python_package/econometricsmodels/panel/fe.py`として実装済み。FE/RE共通の設計判断（`effects`の
引数設計、結果フィールドの共通コア、`cov_type`のサポート対象・デフォルト、内部実装の
共通化方針、リファレンス実装・テスト方針）は
[`panel-common.md`](./panel-common.md)を参照し、本ドキュメントには
FE固有の内容のみを記載する。

## 1. API引数

3層構成: `FE(data, y, x, effects, options=options).fit() -> FEResults`（python_package）→
`fit_fe(data, y, x, effects, options) -> FEResult`（engine_pybind）→ `FeEstimator::fit`
（engine、within変換したデータを`OlsEstimator`へ委譲）。

- `y: str`、`x: list[str]`はOLSと同じ。`effects: list[str]`は独立の必須引数（`y`/`x`と同格、1〜2要素。1要素=1-way、
  2要素=2-way。`effects[0]`が個体、`effects[1]`が時点）。3要素以上は`EFFECTS_TOO_MANY`
  （`ValidationError`）。`FEOptions.time`は廃止した（理由は`panel-common.md`1.1節）。
- `FEOptions`（`#[pyclass]`）:

  | フィールド | 型 | デフォルト | 説明 |
  |---|---|---|---|
  | `cov_type` | `str` | `"cluster"` | `"classical"` / `"hc1"`〜`"hc3"` / `"cluster"` / `"dk"`（大小無視）。OLSと異なり`"hc0"`は非対応（`FeCovType` enum自体が持たない、専用エラーメッセージで弾く） |
  | `confidence_level` | `float` | `0.95` | |
  | `cluster` | `str \| None` | `None` | `cov_type="cluster"`時のグループキー列名。省略時は`effects`の先頭の列をそのまま使う。他の`cov_type`で指定すると`ValidationError` |
  | `dk_time` | `str \| None` | `None` | DK HACの時点列（`cov_type="dk"`では必須、未指定は1-way・2-wayとも`ValidationError`）。`effects`とは独立で、2-wayでも常にこちらがDK計算に使われる（`effects`の列は借用しない）（固定効果は四半期、DKは年次、等も指定できる）。`cov_type="dk"`以外で指定すると`ValidationError` |
  | `dk_bandwidth` | `int \| None` | `None` | DK HACのバンド幅（時点数`t`ベース、OLSの`hac_lags`とは意味が異なるため別名）。省略時は`floor(4*(t/100)^(2/9))`で自動計算。`cov_type="dk"`以外で指定すると`ValidationError` |

- **`include_intercept`は無い**: withinの変換で切片が構造的に消えるため、OLS/WLS/IVと異なり
  このオプション自体が意味を持たない（`FeEstimator::fit`は常に`include_intercept=false`で
  `OlsEstimator`に委譲する）。
- **`x`は空リストを許容しない**: v1では固定効果のみのモデル（`k=0`）を許容して
  いたが、独立して吟味された設計記録が無く、説明変数ゼロは因果推論として意味を持たない
  （個体・時間固定効果によるyの分解という別の操作になる）ためユーザー指摘を受けて拒否に
  変更した。**この検証は`engine_pybind`層のみ**（`build_fe_input`の`validate_x_non_empty`）。
  `engine`側（`FeInput::from_columns`・`FeEstimator::fit`）はk=0を引き続き受理する設計を
  維持している（`FeEstimator::fit`は`OlsEstimator`を経由せず、0列も受理する
  `shared::least_squares::least_squares`を直接呼ぶ。RE自身の内部`OlsEstimator::fit`呼び出しは
  between回帰・最終回帰どちらも構造的にk=0にならないため無変更）。
- 列名を受け取る引数は`effects`/`cluster`/`dk_time`のいずれも`_col`サフィックスを
  付けない（`y`/`x`と同じ規約）。`dk_time`は`cov_type="dk"`の値を接頭辞にした名前。
- **2-wayはバランスパネルを必須とする**。`(effects[0], effects[1])`のユニークペア数が`n_obs`と一致しない
  （重複・欠落がある）場合も含め`ValidationError`（`validate_balanced_panel`が
  `n_obs == 個体数 * 時点数`のカウント一致だけでなくユニークペア数も検証する。ペアの
  重複と欠落が相殺するケースを見逃さないため）。回避オプションは用意しない。1-wayは不均衡
  パネルもv1から無条件でサポートする。
- **1-wayは個体と時点の組の重複を検証しない**。1-wayには時点の列が無く、組という概念自体が
  ないため（個体内の観測の並びと数だけが意味を持つ。同じ行が重複していても各行は独立な
  観測として扱う）。`cov_type="dk"`と`dk_time`を併用する場合も、重複する`(effects[0], dk_time)`は
  検証せず、DKの時点ごとのクロスセクション和でそのまま二重に数える。重複を弾くかどうかは
  利用者側の判断とし、検証は追加しない（意図した挙動）。
- **singletonは自動除外せず常に`ValidationError`**（`validate_no_singleton_groups_one_way`/
  `_two_way`。within変換の前に生の`effects`の列を直接見て検出し、下流の特異行列エラーと
  して偶発的に検出される形にはしない）。2-wayは個体・時点双方を対称に検出する（両方
  該当時は個体側を報告）。エラーの`argument`は`"effects"`で、どの列かは`column`に入る
  （`SINGLETON_GROUP`の`group`はsingletonの値）。
- **within変換後に分散ゼロになる説明変数は`ValidationError`**（`validate_no_zero_variance_
  regressors`。時間不変変数（1-way）・2-wayでtime FEと完全共線な列を同じロジックで検出する。
  閾値は絶対値ではなく相対値: 変換前の元の列の最大絶対値×観測数×`f64::EPSILON`と比較する）。
- **被説明変数にwithin変動が残らなければ`ValidationError`（`OUTCOME_CONSTANT_WITHIN_ENTITY`）**:
  `y`がどの個体内でも定数（1-way）、または個体効果と時点効果だけで説明される（2-way）と、
  within変換後の`y`は0になり固定効果が説明するものが残らない。説明変数と同じ
  `column_is_zero_variance`の相対閾値（変換前の`y`の最大絶対値×観測数×`f64::EPSILON`）で判定する
  （`0.1`を3回足して3で割ると丸め誤差が残るため、厳密な等値では見逃す）。説明変数の検査の後に行う。
- **完全フィットは`ComputationError`（`PERFECT_FIT`）**: within回帰の残差平方和が、within変換後の
  `y`の総平方和（非中心化の$\sum\tilde y^2$）の`k·ε`倍以下だと、標準誤差とF検定が定義できない
  （`ols-spec.md`3.3節と同じ判定、`is_perfect_fit`）。`PanelError::WithinRegressionFailed`に包んで
  共分散・F検定より前に止める（`stage`は`WITHIN_REGRESSION`）。
- 欠損値（null・NaN/無限大）は常にエラー（方針は[`docs/guide/validation.md`](../guide/validation.md)）。`include_intercept`が無いため`"const"`列との衝突検証も
  無い。

## 2. 結果構造体

`FEResult`（`#[pyclass]`）が公開する項目: `params` / `std_errors` / `test_stats`（**t検定**） /
`p_values` / `conf_lower` / `conf_upper` / `param_names`（切片なし） / `residuals` /
`dep_var_name` / `n_obs` / `df_resid` / `df_model` / `n_groups`（`dict[str, int]`、`effects`の列名→グループ数、`effects`の順） / `cov_type` / `dk_bandwidth_used` / `f_statistic` /
`f_p_value` / `f_df_num` / `f_df_denom`（`(k, df_resid)`、`k=0`で`f_statistic`/`f_p_value`ともども`None`） / `log_likelihood` / `aic` / `bic` / `r_squared_within` / `r_squared_between` /
`r_squared_overall`。

- **`dk_bandwidth_used`**: `cov_type="dk"`のとき実際に使われたバンド幅（`dk_bandwidth`明示指定ならその値、
  未指定なら`resolve_dk_bandwidth`が`floor(4*(t/100)^(2/9))`で解決した値、`t`は`dk_time`のユニーク数）。
  `dk`以外は`None`。OLS/IVの`hac_lags_used`とは基準が観測数`n`でなく時点数`t`のため別名にしている。
- **`n_groups`は`effects`の列名をキーにした`dict[str, int]`**（`effects`の順。one-wayは1項目、
  two-wayは2項目）: `FEResult`は`effects`（列名）と`group_counts`（件数）を保持し、`n_groups`
  getterで`dict`にする。1つ目は`FeInput::n_entities()`（構築時に一度だけ作ったエンティティ
  コード`GroupCodes`のユニーク数）、2つ目は`FeEstimator::n_periods()`（`fit`内で自由度計算の
  ために既に求めている時点のユニーク数。二重計算しない）から取る。2-wayはバランスパネル必須の
  ため、2つ目の件数は各個体の観測数と一致する（将来2-wayの不均衡パネルに対応する場合は
  定義を再検討する）。固定の`n_entities`/`n_periods`ではなく列名キーにした理由は
  `panel-common.md`2.1節。REの`n_groups`も同じ定義（`re-spec.md`2章）。
- **`estimator()`（内部委譲した`OlsEstimator`）とFE自身のgetterの使い分け**: `params`/
  `param_names`/`residuals`/`dep_var_name`/`n_obs`/`log_likelihood`は`estimator()`から、
  `std_errors`/`test_stats`/`p_values`/`conf_lower`/`conf_upper`/`df_model`/`df_resid`/
  `f_statistic`/`f_p_value`/`aic`/`bic`/`r_squared_*`はFE自身から取得する。後者はFEが
  `cov_type`・パネル自由度調整を反映して計算し直した値であり、`estimator()`側は常に
  `CovType::Classical`で委譲した内部OLSの生の値のため取り違えるとcov_type非対応の値を
  返してしまう。
- **`fixed_effects()`（固定効果自体）は`fit()`の戻り値本体に含めず別メソッド**（IVの
  `first_stage()`と同じ「追加結果は別メソッド」方針）。`FEResult`は非公開フィールド
  `estimator: FeEstimator`として推定量本体を保持し、呼び出し時にオンデマンドで
  `FeEstimator::fixed_effects()`を呼ぶ（`FeEstimator`は`Clone`未実装のため`FEResult`から
  `#[derive(Clone)]`は外している）。
- `summary()`は実装しない（structured-data-only出力方針）。
- python_package層（`FEResults`）: `params`/`std_errors`/`test_stats`/`p_values`/`conf_int`は
  係数名→値の`dict`。`params_table()`は行指向`list[dict]`（キーは`param_name`/`param`/`std_error`/
  `test_stat`/`p_value`/`conf_lower`/`conf_upper`、OLSと同じ）。

## 3. 内部実装の計算仕様

### 3.1 within変換

polarsではなく`engine`側は抽出済み配列（`entity: &[String]`等）を直接扱う。1-way/2-wayとも
`quasi_demean_column`（θ=1固定、[`re-spec.md`](./re-spec.md)3.2節参照）を使う:

- **1-way**: `quasi_demean_column`を1回呼ぶだけ（`ỹ_i = y_i - ȳ_i.`）。
- **2-way**: 閉形式の二重デミーニング`ỹ_it = y_it - ȳ_i. - ȳ_.t + ȳ..`を直接計算するのでは
  なく、「entityでquasi-demean（θ=1）→ その結果をtimeでquasi-demean（θ=1）」という2段階の
  `quasi_demean_column`呼び出しで実装している。バランスパネルではこの2段階が閉形式と厳密に
  一致することを導出済み（独立検算のproptestでも検証済み）。バランスパネル検証
  （`validate_balanced_panel`）は時点のキー未指定なら`PanelError::TwoWayRequiresTime`。

`FeEstimator::fit`のパイプラインは「singleton検出 → within変換（2-wayはバランスパネル検証を
内包） → 自由度検証 → 分散ゼロ検出 → `OlsEstimator::fit`への委譲（`include_intercept=false`
固定、Frisch-Waugh-Lovell定理によりwithin変換後のOLS点推定はFE推定量と数学的に一致） →
`cov_type`別の共分散行列計算 → 自由度調整後の統計量再計算」の順。委譲先の`OlsEstimator::fit`
自体は常に`cov_type=Classical`固定で呼ぶ（`β̂`・残差の取得のみが目的で、`cov_type`ごとの
標準誤差はFE自身が独立に計算し直すため委譲先のcov_type選択は結果に影響しない）。

### 3.2 自由度調整

- **1-way**: `df_model = k + N`（`N`=個体数）、`df_resid = n - df_model`。
- **2-way**: `df_model = k + N + T - 1`（`T`=時点数。個体/時点ダミー間の定数項ぶんの
  重複を`+1`補正、LSDVでの有効パラメータ数が`N + T - 1`になることに対応）。
- `n <= df_model`は`PanelError::InsufficientDegreesOfFreedom`。`neffects>=1`のため
  `df_model>k`が恒常的に成り立ち、`OlsEstimator::fit`自身の`n<=k`チェックより常に厳格
  （FE経由でOLS側の`InsufficientObservations`が発生することは構造的にない）。
- **t値・p値・信頼区間・F検定の自由度（`df_inference`）は`cov_type=Cluster`のとき`G-1`、
  `Dk`のとき`t_periods-1`に切り替える**（fixestの`ssc()`既定`t.df="min"`。
  それ以外（Classical/HC1-3）は`df_resid`のまま）。AIC/BICは`log_likelihood`
  自体（`SSR/n`のみに依存しdf非依存）を`OlsEstimator::log_likelihood()`からそのまま再利用し、
  ペナルティ項の乗数だけ`k`から`df_model`に差し替える（`df_resid`ベース、`df_inference`とは
  無関係）。
- **F統計量（`f_statistic`/`f_p_value`）**: 傾き係数`k`個が同時にゼロという帰無仮説のWald
  F検定（`linearmodels.PanelOLS.f_statistic`「H0: All parameters ex. constant are zero」と
  同じ定義。固定効果ダミー自体は検定対象に含めない——`fixest`の`fitstat(m, "f")`はFEダミーも
  含めたモデル全体のF検定で定義が異なるためクロスチェックに使わず、同じ傾き係数のみの検定である
  `fixest::wald()`を使う）。`wald_f_test`
  （OLS本体の関数を`pub(crate)`化して再利用）をFEの`cov_type`別`cov_params`・`df_inference`で
  呼ぶ形でサンドイッチ計算の複製を避ける（`k_constant=0`固定）。失敗は
  `PanelError::FTestFailed`。

### 3.3 `cov_type`対応

`FeCovType` enum（`Classical`/`Hc1`/`Hc2`/`Hc3`/`Cluster{groups}`/`Dk{bandwidth, time}`）を
`OlsEstimator`の`CovType`とは別に新設（HC0を含まない、無効な組み合わせを型で表現不可能にする
設計）。`OlsEstimator`の既存cov_type計算式はそのまま流用できない——linearmodels/fixestの
ソース確認・実地数値検証で判明した3点の相違:

1. **HC0はスコープ外**（仕様判断。fixest 0.14.2は`vcov = "HC0"`を受け付けないが、plm 2.6.7は提供するため、他パッケージの有無は理由にしない）。
2. **HC1〜HC3は独自計算**: HC1は小標本補正係数が`n/(n-k)`ではなく`n/df_resid`。HC2/HC3の
   レバレッジは、within変換後の設計行列ではなくLSDV相当のフルレバレッジ
   `h_ii_full = 1/T_entity(i) + h_ii_within`（1-wayは分割回帰＝Frisch-Waugh-Lovellのレバレッジ
   分解則、2-wayはさらに`+ 1/N_time(i) - 1/n`）を使う（fixestと数値一致確認済み）。
3. **Clusterも独自計算**（**【linearmodels方式からfixest方式へ変更】**）:
   補正係数はOLSと同じ`(G/(G-1))×((n-1)/(n-K))`だが、**`K`（`(n-1)/(n-K)`の分母）は
   fixestの`ssc()`小標本補正（既定`K.fixef="nonnested"`）に合わせたFE固有ロジック**
   （`fe_cluster_k_correction`、`engine/src/panel/fe.rs`）で決める——linearmodelsが
   使っていた単純な`extra_df`分岐（`n/(n-extra_df-k)`のみ、`G/(G-1)`補正なし）とは
   異なる。fixest 0.14.2のRソース（`fixest:::ssc_compute_K`）と実地数値実験で確定した式:
   FE次元（1-wayはentityのみ、2-wayはentity+time）それぞれについて、その次元が
   クラスター変数に「ネスト」しているか（各水準がクラスターの単一の値にしか対応しないか）
   で場合分けし、全次元がネスト（1-way・`cluster`省略時の既定ケース）なら
   `K = df_model - nested_size_sum + m`（`m`=FE次元数、`nested_size_sum`=ネストした
   次元の生の水準数の合計）、一部の次元だけネスト（2-way FEで典型）なら
   `K = df_model - (nested_size_sum - count_nested)`、どの次元もネストしない
   （Stataの`xtreg,fe`型）なら`K = df_model`（フルカウント）。最後にfixest自身の
   安全弁`K = max(K, k+1)`を適用する。詳細な導出・具体例は`fe_cluster_k_correction`
   関数doc参照。
   **2クラスターの退化**: `G=2`かつ吸収したFE次元（常にentity、2-wayではtimeも）のいずれかで
   全水準が2つのクラスターに1観測ずつで`k >= 1`なら、`PanelError::DegenerateClusterTwoGroups`
   として拒否する。典型例は2時点のパネルを時点の列でクラスタリング（個体方向）と、2-way FEで
   個体2つのパネルを`effects[0]`でクラスタリング——Clusterの既定——（時点方向）。within変換は
   吸収した各次元の水準内で和をゼロにするため、そうした水準の2観測で`x̃`・`ẽ`が符号反転し
   （entity方向なら`x̃_i1 = -x̃_i2`・`ẽ_i1 = -ẽ_i2`）、クラスタースコアが`s_1 = s_2`、
   正規方程式`s_1 + s_2 = 0`と合わせて`s_1 = s_2 = 0`、共分散が恒等的にゼロになるため
   （`G > k`の検証は`k=1`で通ってしまう）。1水準でもパターンが崩れれば退化しないため通す。

`cov_type`のデフォルト（`"cluster"`、`effects`の先頭の列＝個体単位）・`cluster`文字列パースは
`engine_pybind`層の責務。`FeEstimator::fit`自体はデフォルトを持たない。**2-way FEでも
クラスターのデフォルトは`effects[0]`（個体）単位のまま**（`cluster`で上書き可能）。2-way clustering
（個体+時点同時）はv1スコープ外。

**Driscoll-Kraay型パネルHAC（`FeCovType::Dk { bandwidth, time }`）**: カーネル本体
（Bartlett重み・ラグ項の積算）は`linearmodels.panel.covariance.DriscollKraay`のソース
確認に基づく実装のまま。**小標本補正のみfixestの`vcov="DK"`（`ssc()`に
従う）に変更した**。

1. **カーネルはv1でBartlett限定**（OLSの`CovType::Hac`と平仄を合わせる。Parzen/QSは未対応、
   4章参照）。
2. **バンド幅**は`linearmodels`のデフォルトルール`floor(4*(t/100)^(2/9))`（`t`=ユニークな
   時点数、OLSの`hac_lags`が観測数`n`ベースなのと違う点に注意）。明示指定は`[0, t)`範囲検証
   （`PanelError::InvalidDkBandwidth`）。`t<2`は`G=1`のクラスターと同じ理由
   （`t/(t-1)`が発散する）で`PanelError::InsufficientDkPeriods`として拒否する。
   また`t <= k`（時点数が傾き係数の数以下）も`PanelError::InsufficientDkPeriodsForInference`
   として拒否する: `Ŝ`は時点ごとのスコア`h_t = Σ_i x̃_it·ẽ_it`の外積（とラグ項）の和で、
   正規方程式`Σ_t h_t = X̃'ẽ = 0`により`rank(Ŝ) ≤ t-1`となり、F検定の`k×k`部分行列が
   構造的に特異になるため（クラスターの`G <= k`と同じ構造。数値的な特異性判定に任せると
   丸め誤差次第で巨大な無意味な統計量を返しうるため、入力から事前に判定する）。
   さらに`t=2`かつ全エンティティ（2-wayでは全時点でも）が2時点に1観測ずつで`k >= 1`なら、
   Clusterの2クラスターの
   退化（上記3.）と同じ理由で`Ŝ`が恒等的にゼロになるため
   `PanelError::DegenerateDkTwoPeriods`として拒否する（REは残差がwithin変換されないため対象外）。
3. **時系列順序は列の値の順序で決める**（文字列の辞書順ではない）。整数・浮動小数は数値順、
   `Date`・`Datetime`は時系列順、`Enum`はカテゴリの定義順、文字列・`Categorical`はラベルの
   辞書順。`engine_pybind`が列のdtypeから時間順のコードを作り、`engine`には時点のラベルと
   時間順のコード（`TimeKeys`）で渡す。辞書順を使うと、ゼロ埋めのない整数
   （`1, 10, 11, 2, ...`）等でラグ項が別の時点同士を組み合わせ、標準誤差が黙って変わる。
4. **1-way/2-way両対応**。DKの時点列は`dk_time`（engineでは`FeCovType::Dk.time`）として
   常に明示的に受け取る。未指定は`ValidationError`（`engine_pybind`が弾く）。2-wayの
   `effects[1]`は借用しない。
5. **スケールは`(t_periods/(t_periods-1)) × ((n-1)/(n-K)) × (X̃'X̃)⁻¹ Ŝ (X̃'X̃)⁻¹`**
   （fixestの`ssc()`は時点数`t_periods`をクラスターの`G`と同じ役割で使う。`K`は
   fixestの既定`K.fixef="full"`——DKにはクラスター変数という概念が無くネスト判定
   自体が発生しないため——により常に`K=df_model`）。
6. **`FeCovType::Dk.time`は必須で、`FeInput.time_keys()`にはフォールバックしない**: 当初は
   `Option`で、`None`なら2-wayの時点キーを暗黙に借用していたが、どの列がDKの時点かを利用者が
   明示しない設計は、意図と違う列が選ばれても気づけないため廃止した。`FeInput.time_keys()`は
   2-wayの固定効果の時間次元（`effects[1]`）専用で、DKの時点列とは独立。
7. **`bandwidth == t_periods - 1`ではfixestと一致しない（fixest側の実装誤り、本実装は
   標準のBartlettカーネルを維持）**: `bandwidth`が許容範囲`[0, t_periods)`の上限ちょうど
   のとき、fixest 0.14.2の`vcov="DK"`は最後のラグ項（`l = bandwidth`）を落とした値を
   返し、本実装（最後のラグ項まで含める）とこの境界値でのみ数値が一致する。
   `bandwidth <= t_periods - 2`では一致する。
   - **原因**: fixestのC++実装`cpp_driscoll_kraay`（`src/vcov_related.cpp`）の
     off-by-one。R側はBartlett重みの長さ`bandwidth + 2`のベクトル（末尾は0）を渡し、
     C++側は末尾の0を除いた「ラグ`0..=bandwidth`の個数」`L = bandwidth + 1`を数える。
     続けて`if (L > T - 1) L = T - 1;`で上限を課すが、`L`は最大ラグではなく個数なので、
     ラグの上限`T - 1`に対応する上限は`L <= T`でなければならない。`T - 1`と書かれて
     いるため`bandwidth == T - 1`（`L = T`）だけ`L = T - 1`に切り詰められ、最大ラグが
     `T - 2`になる。内側のループが`t < T - l`で境界を自前で処理しているためこの上限は
     不要で、意図的な切り捨てとは考えにくい。`l = T - 1`の項`Γ_{T-1} = ξ_{T-1}'ξ_0`は
     一般に非ゼロで、落とす数学的根拠もない。T=5の乱数スコアで`cpp_driscoll_kraay`を
     直接呼ぶと、`bandwidth = 0..=3`は標準カーネルと機械精度で一致し、`bandwidth = 4`だけ
     「`l = 4`の項を除いた標準カーネル」と機械精度で一致した（標準カーネルとの差は約0.3）。
   - **方針**: fixestのこの挙動は再現しない。標準のBartlettカーネル（linearmodelsの
     `DriscollKraay`・`plm::vcovSCC(maxlag = T-1)`と同じ）を維持する。`plm::vcovSCC`は
     最後のラグ項を落とさず、補正係数を除いて本実装と一致する（第2リファレンスとして
     `fe_plm_crosscheck.json`の`dk_max_bandwidth`が検証する）。エンジンの
     `panel_driscoll_kraay_cov_params`の単体テストは、境界を含む全バンド幅を定義式の
     手計算と照合する。fixestの`DK(lag)`とは`lag <= T - 2`で比較すること。

### 3.4 パネル固有R²（`r_squared_within`/`between`/`overall`）

`linearmodels==7.0`のソース確認・実地数値検証で判明した3点（素朴に「実際に使ったFE構造で
demeanしたR²」を3種とも定義すると誤る）:

1. **`r_squared_within`は「実際に使ったFE構造でdemeanした残差」を採用**（1-wayはentityの
   み、2-wayはentity+time）。**`linearmodels`自身の`rsquared_within`は常にentityのみの
   demeanで固定**（2-wayモデルでも時間効果を含めない）という別定義のため、2-wayでは意図的に
   数値が食い違う。2-wayの検証は`linearmodels`ではなく`fixest`の`fitstat(model, "wr2")`
   （AIC/BICと同型のRクロスチェック例外）。
2. **`r_squared_between`/`r_squared_overall`は`linearmodels`の`_rsquared`と完全一致**させる。
   FEのxには定数列を含められないため`has_constant=False`分岐（非中心化TSS）が常に適用される。
   `r_squared_overall`は固定効果の切片項を一切含めない元の`y`・`x`に`β̂`だけを当てはめた
   残差ベース。`r_squared_between`はエンティティ平均`ȳ_i.`・`x̄_i.`に`β̂`を当てはめた
   残差ベース。
3. **`r_squared_between`にエンティティ観測数`T_i`による重み付けをしてはいけない**:
   `linearmodels`の`_prepare_between`自体は不均衡パネル用の重み`w_i=T_i/mean(T)`を計算する
   が、`_rsquared`側でサンプルウェイトが全て`1.0`（未指定）なら`w`を無条件に`1.0`へ
   上書きする。FEは`weights`引数をサポートしないためこの重み付けは常に無効（バランスパネルの
   テストのみだと`w_i=1`に退化するため気づかず、不均衡パネルで初めて数値不一致が発覚した
   経緯あり）。

### 3.5 固定効果自体（α_i）の復元

`FeEstimator::fixed_effects()`（`FixedEffects::OneWay(BTreeMap<String, f64>)`/
`TwoWay { entity, time }`）。推定済み係数から事後的に復元する（`within`変換時のグループ平均を
保持する形は取らず、`FeInput`が保持する変換前の元の`y`/`x`/`entity`/`time`から都度
再計算する）。

- **1-way**: `α_i = ȳ_i. - x̄_i.'β̂`（一意。切片が全てentityに吸収される設計のため正規化の
  任意性が無い）。
- **2-way**: この式をそのまま時点効果に拡張できない——`α_i`に定数`c`を足し`γ_t`から`c`を
  引いても同じ予測値になるため正規化の任意性があり、単純に両方へ当てはめると大域平均が
  二重計上されるバグになる。採用した規約: **`effects[1]`の時間順（DKの時系列順序と同じ、
  上記3.）で最初の値`t_ref`を基準に`γ_{t_ref}=0`とし、`α_i`に大域的な水準を吸収させる**（Stata `areg`・R `fixest`/`lfe`等と
  同型の「片方の基準カテゴリを0にする」慣行）。**`fixest::fixef()`との数値一致は`t_ref`の
  選び方が一致する入力に限られる**（`fixest`自身の基準時点選択は時点列の時間順ではなく
  観測順で最初に現れた値であるため、行の並び順を変えると`fixest`側の基準時点も変わる。
  2-wayの正規化はどの`t_ref`を選んでも数学的に等価なため、本実装は`fixest`の観測順依存の
  挙動は再現せず時間順の規約を優先する）。`fixed_effects()`は常に`{列名: {水準: 効果}}`の入れ子の`dict`（one-wayも1項目の入れ子。キーは
  `effects`の列名で、`effects`の順）で、時点側の`dict`は時間順に並べる。入れ替えると
  キー順と基準期間が変わる（`panel-common.md`1.1節）。

### 3.6 engine_pybind: エラー変換

`engine::panel::common::PanelError` → `PyErr`（`panel_error_to_pyerr`、`common.rs`）。`ErrorInfo::report()`が
`code`・`params`に変換し、`ValidationError`/`ComputationError`の別はカタログの`class`で決まる:

| `PanelError` | `code` | 例外 |
|---|---|---|
| `Common(CommonError)` | `CommonError`と同じ | `CommonError`と同じ |
| `InsufficientDegreesOfFreedom` | `INSUFFICIENT_DEGREES_OF_FREEDOM`（`n_effects`は1-wayで個体数、2-wayで個体数+時点数-1） | `ValidationError` |
| `SingletonGroup` | `SINGLETON_GROUP`（params: `argument`=`"effects"`・`column`・`group`。`column`は`effects`の該当要素の列名で、エンジンは`with_effect_names`で受け取り計算には使わない） | `ValidationError` |
| `UnbalancedPanelForTwoWay` | `PANEL_NOT_BALANCED_FOR_TWO_WAY` | `ValidationError` |
| `ZeroVarianceAfterDemeaning` | `REGRESSOR_NO_WITHIN_VARIATION` | `ValidationError` |
| `OutcomeConstantWithinEntity` | `OUTCOME_CONSTANT_WITHIN_ENTITY` | `ValidationError` |
| `WithinRegressionFailed { source: Common(PerfectFit) }` | `PERFECT_FIT`（`stage`は`WITHIN_REGRESSION`） | `ComputationError` |
| `InvalidDkBandwidth` / `InsufficientDkPeriods` / `InsufficientDkPeriodsForInference` | `DK_BANDWIDTH_OUT_OF_RANGE` / `DK_PERIODS_TOO_FEW` / `DK_PERIODS_NOT_ABOVE_COEFFICIENTS` | `ValidationError` |
| `DegenerateDkTwoPeriods` / `DegenerateClusterTwoGroups` | `DK_TWO_PERIODS_DEGENERATE` / `CLUSTER_TWO_GROUPS_DEGENERATE` | `ValidationError` |
| `AmemiyaRequiresWithinVariation` | `AMEMIYA_REQUIRES_WITHIN_VARIATION` | `ValidationError` |
| `NonPositiveIdiosyncraticVariance` / `VarianceComponentsMomentsSingular` | `NON_POSITIVE_IDIOSYNCRATIC_VARIANCE` / `VARIANCE_COMPONENTS_MOMENTS_SINGULAR` | `ComputationError` |
| `InStage` | 包んだ`source`の`code`に、起きた計算の`stage`を付ける。REの内部FE（分散成分の推定）の自由度不足・最小二乗・F検定の失敗は`VARIANCE_COMPONENT_WITHIN_REGRESSION`、FE・REのDK時点数不足は`OVERALL_F_TEST` | `source`と同じ |
| `IdentifierDimensionMismatch` / `TwoWayRequiresTime` | `INTERNAL_ERROR`（Python APIからは起こらない） | `ComputationError` |
| `WithinRegressionFailed` / `FTestFailed`（RE: `BetweenRegressionFailed` / `QuasiDemeanedRegressionFailed` / `PooledRegressionFailed` / `HausmanTestFailed`） | 内側の`LeastSquaresError`の`code`に、どの回帰かを示す`stage`を付ける（`stage`を付けてよい`code`のみ） | `code`のクラスで決まる |

`PanelError`はFE/REで共有し、`FeError`/`ReError`は個別に作らない。engine側に新バリアントを
追加したら`ErrorInfo::report()`が対応する`code`を返すようにする（`match`が網羅的なので、
更新漏れはコンパイルエラーで検出される）。

## 4. テスト

- Python主リファレンス: `linearmodels.PanelOLS`（点推定・`cov_type="unadjusted"`/
  パネル固有R²・AIC/BIC等）。Rクロスチェック: `fixest`（`benchmark/panel/
  run_fixest_benchmark.R`）。
- **【例外】** `cov_type="cluster"`/`"dk"`の標準誤差・推論統計量
  （`std_errors`/`test_stats`/`p_values`/`conf_lower`/`conf_upper`）は`linearmodels`
  ではなく`fixest`を正とする（3.3節参照、`linearmodels`独自の`extra_df`補正から
  fixest・Stata型`G/(G-1)×(n-1)/(n-K)`補正に変更したため）。fixestの`ssc()`既定のまま
  1-way・2-wayとも機械精度で一致する（p値・信頼区間を含む）。DKはfixestの既定バンド幅が
  本実装と異なるため、本実装の既定式`floor(4*(T/100)^(2/9))`で求めた値を`DK(lag)`へ明示的に
  渡す。`linearmodels`との比較は`classical`（`unadjusted`）・`hc1`（`robust`）のみ、
  `hc2`/`hc3`は`linearmodels`に無いためfixestのみ。
- 許容誤差: 相対誤差`1e-8`（`tests/support/_tolerances_panel.py`の`fe_reference`・`fe_crosscheck`、
  `.claude/rules/testing-policy.md`の基本方針通り）。実測の一致は`1e-14`程度で、許容誤差は
  それにマージンを載せた基本値のまま。
- **`aic`/`bic`はRクロスチェック（`fixest`）のみで検証する**: `linearmodels.PanelOLS`は
  `aic`/`bic`を一切提供しないため（`rsquared_within`/`between`/`overall`/`inclusive`・
  `loglik`のみ）、通常の「Python主リファレンス＋Rクロスチェック」の2系統検証の例外
  （ハウスマン検定と同型）。
- **2-wayの`r_squared_within`もRクロスチェック（`fixest`）のみ**（3.4節参照、`linearmodels`
  自身が2-wayでも常にentityのみdemeanという別定義のため）。
- F統計量: `linearmodels`と直接比較（`cov_type="unadjusted"`・HC1相当）。加えて
  `fixest::wald(m, keep = <全傾き係数>, vcov = <cov_typeと同じ>)`と、全`cov_type`
  （classical/HC1/HC2/HC3/cluster/dk）×1-way・2-wayで機械精度の一致を確認する
  （`tests/panel/test_fe_crosscheck.py`）。p値はfixestの統計量と`summary`のt検定と同じ分母自由度
  （`degrees_freedom(model, "t", vcov)`）から`pf()`で計算し直す——`wald()`は分母自由度を
  `max(df2, df1 + 1)`に切り上げるため、`G-1 <= q`・`df_resid <= q`の境界（G=2、df_resid=1等）で
  t検定と食い違う自由度を使うため（fixest 0.14.2で確認）。k=1のケースは「1自由度の
  F検定は両側t検定と代数的に等価」という恒等式（`f_statistic = test_stat²`）でHC1/HC2/HC3/
  Cluster/HACを横断検証する。

## 5. 未実装・未対応

- **不均衡パネルでの2-way FE**: 反復的な交互射影（fixest/lfe方式）が必要で、v1では扱わない
  （閉形式の二重デミーニングはバランスパネルでのみ正確なため、6.4節相当の制約として
  `ValidationError`にする）。
- **2-way clustering（個体+時点同時）**: v1スコープ外。
- **Driscoll-Kraay HACのカーネル拡張**: Parzen/QSカーネルへの拡張は別issue。
- **DKの`bandwidth == t_periods - 1`（許容範囲の上限ちょうど）でfixestと数値不一致**:
  fixestの`cpp_driscoll_kraay`のoff-by-oneが原因で、本実装は標準カーネルを維持する
  （3.3節7.参照）。未対応ではなく意図した差異。
