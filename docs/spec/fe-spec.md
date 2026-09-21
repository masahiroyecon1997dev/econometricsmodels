# FE 仕様書

FE（固定効果パネル回帰、within推定）の確定済み仕様。`engine/src/panel/fe.rs`（共通基盤は
`engine/src/panel/common.rs`）・`engine_pybind/src/panel/fe.rs`・
`python_package/econometricsmodels/panel/fe.py`として実装済み。FE/RE共通の設計判断（`entity`/
`time`の引数設計、結果フィールドの共通コア、`cov_type`のサポート対象・デフォルト、内部実装の
共通化方針、リファレンス実装・テスト方針）は
[`panel-common.md`](./panel-common.md)を参照し、本ドキュメントには
FE固有の内容のみを記載する。

## 1. API引数

3層構成: `FE(data, y, x, entity, options).fit() -> FEResults`（python_package）→
`fit_fe(data, y, x, entity, options) -> FEResult`（engine_pybind）→ `FeEstimator::fit`
（engine、within変換したデータを`OlsEstimator`へ委譲）。

- `y: str`、`x: list[str]`はOLSと同じ。`entity: str`は独立の必須引数（`y`/`x`と同格）。
- `FEOptions`（`#[pyclass]`）:

  | フィールド | 型 | デフォルト | 説明 |
  |---|---|---|---|
  | `cov_type` | `str` | `"cluster"` | `"classical"` / `"hc1"`〜`"hc3"` / `"cluster"` / `"hac"`（大小無視）。OLSと異なり`"hc0"`は非対応（`FeCovType` enum自体が持たない、専用エラーメッセージで弾く） |
  | `confidence_level` | `float` | `0.95` | |
  | `time` | `str \| None` | `None` | `Some`なら2-way（entity+time）、`None`なら1-way。`cov_type="hac"`時のDK時系列順序としても使われる（`time_col`未指定の場合） |
  | `cluster_col` | `str \| None` | `None` | `cov_type="cluster"`時のグループキー列名。省略時は`entity`をそのまま使う |
  | `time_col` | `str \| None` | `None` | DK HAC専用の時系列順序。`time`とは独立に指定でき、指定時は2-wayでも常にこちらが優先される |
  | `dk_bandwidth` | `int \| None` | `None` | DK HACのバンド幅（時点数`t`ベース、OLSの`hac_lags`とは意味が異なるため別名）。省略時は`floor(4*(t/100)^(2/9))`で自動計算 |

- **`include_intercept`は無い**: withinの変換で切片が構造的に消えるため、OLS/WLS/IVと異なり
  このオプション自体が意味を持たない（`FeEstimator::fit`は常に`include_intercept=false`で
  `OlsEstimator`に委譲する）。
- **`x`は空リストを許容しない（Issue #320）**: v1では固定効果のみのモデル（`k=0`）を許容して
  いたが、独立して吟味された設計記録が無く、説明変数ゼロは因果推論として意味を持たない
  （個体・時間固定効果によるyの分解という別の操作になる）ためユーザー指摘を受けて拒否に
  変更した。**この検証は`engine_pybind`層のみ**（`build_fe_input`の`validate_x_non_empty`）。
  `engine`側（`FeInput::from_columns`・`FeEstimator::fit`）はk=0を引き続き受理する設計を
  維持している（`OlsEstimator::fit_allowing_no_regressors`という、通常の`fit`からk=0拒否
  ガードだけを外した別関数への切り替えで実現。RE自身の内部`OlsEstimator::fit`呼び出しは
  between回帰・最終回帰どちらも構造的にk=0にならないため無変更）。
- `entity`/`time`はbareネーミング（`_col`サフィックスなし、`y`/`x`と同格の中核変数という
  位置づけ）。`cluster_col`/`time_col`は「補助列」という別の位置づけのため`_col`サフィックスを
  持つ（既存の`OLSOptions.cluster_col`/`time_col`と同じ規約）。
- **2-wayはバランスパネルを必須とする**。`(entity, time)`のユニークペア数が`n_obs`と一致しない
  （重複・欠落がある）場合も含め`ValidationError`（`validate_balanced_panel`が
  `n_obs == n_entities * n_periods`のカウント一致だけでなくユニークペア数も検証する。ペアの
  重複と欠落が相殺するケースを見逃さないため）。回避オプションは用意しない。1-wayは不均衡
  パネルもv1から無条件でサポートする。
- **singletonは自動除外せず常に`ValidationError`**（`validate_no_singleton_groups_one_way`/
  `_two_way`。within変換の前に生の`entity`/`time`列を直接見て検出し、下流の特異行列エラーと
  して偶発的に検出される形にはしない）。2-wayはentity・time双方を対称に検出する（両方
  該当時はentity側を報告）。
- **within変換後に分散ゼロになる説明変数は`ValidationError`**（`validate_no_zero_variance_
  regressors`。時間不変変数（1-way）・2-wayでtime FEと完全共線な列を同じロジックで検出する。
  閾値は絶対値ではなく相対値: 変換前の元の列の最大絶対値×観測数×`f64::EPSILON`と比較する）。
- 欠損値（NaN/無限大）は常にエラー。`include_intercept`が無いため`"const"`列との衝突検証も
  無い。

## 2. 結果構造体

`FEResult`（`#[pyclass]`）が公開する項目: `params` / `std_errors` / `t_stats`（**t検定**） /
`p_values` / `conf_lower` / `conf_upper` / `param_names`（切片なし） / `residuals` /
`dep_var_name` / `n_obs` / `df_resid` / `df_model` / `n_entities` / `cov_type` / `f_statistic` /
`f_p_value` / `log_likelihood` / `aic` / `bic` / `r_squared_within` / `r_squared_between` /
`r_squared_overall`。

- **`n_entities`はengine側にgetterが無い**: `FeEstimator`内部のprivateな`count_unique`のみで
  外部公開されていないため、`engine_pybind`側で`FeInput::entity()`を`HashSet`に集めて独立に
  計算する（`fit()`実装の一部）。
- **`estimator()`（内部委譲した`OlsEstimator`）とFE自身のgetterの使い分け**: `params`/
  `param_names`/`residuals`/`dep_var_name`/`n_obs`/`log_likelihood`は`estimator()`から、
  `std_errors`/`t_stats`/`p_values`/`conf_lower`/`conf_upper`/`df_model`/`df_resid`/
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
- python_package層（`FEResults`）: `params`/`std_errors`/`t_stats`/`p_values`/`conf_int`は
  係数名→値の`dict`。`coef_table()`は行指向`list[dict]`（キーは`param`/`coef`/`std_err`/
  `t_stat`/`p_value`/`conf_lower`/`conf_upper`、OLSと同じ）。

## 3. 内部実装の計算仕様

### 3.1 within変換

polarsではなく`engine`側は抽出済み配列（`entity: &[String]`等）を直接扱う。1-way/2-wayとも
`quasi_demean_column`（θ=1固定、[`re-spec.md`](./re-spec.md)3.2節参照）を使う:

- **1-way**: `quasi_demean_column`を1回呼ぶだけ（`ỹ_i = y_i - ȳ_i.`）。
- **2-way**: 閉形式の二重デミーニング`ỹ_it = y_it - ȳ_i. - ȳ_.t + ȳ..`を直接計算するのでは
  なく、「entityでquasi-demean（θ=1）→ その結果をtimeでquasi-demean（θ=1）」という2段階の
  `quasi_demean_column`呼び出しで実装している。バランスパネルではこの2段階が閉形式と厳密に
  一致することを導出済み（独立検算のproptestでも検証済み）。バランスパネル検証
  （`validate_balanced_panel`）は`time`未指定なら`PanelError::TwoWayRequiresTime`。

`FeEstimator::fit`のパイプラインは「singleton検出 → within変換（2-wayはバランスパネル検証を
内包） → 自由度検証 → 分散ゼロ検出 → `OlsEstimator::fit`への委譲（`include_intercept=false`
固定、Frisch-Waugh-Lovell定理によりwithin変換後のOLS点推定はFE推定量と数学的に一致） →
`cov_type`別の共分散行列計算 → 自由度調整後の統計量再計算」の順。委譲先の`OlsEstimator::fit`
自体は常に`cov_type=Classical`固定で呼ぶ（`β̂`・残差の取得のみが目的で、`cov_type`ごとの
標準誤差はFE自身が独立に計算し直すため委譲先のcov_type選択は結果に影響しない）。

### 3.2 自由度調整

- **1-way**: `df_model = k + n_entities`、`df_resid = n - df_model`。
- **2-way**: `df_model = k + n_entities + n_periods - 1`（entity/timeダミー間の定数項ぶんの
  重複を`+1`補正、LSDVでの有効パラメータ数が`n_entities + n_periods - 1`になることに対応）。
- `n <= df_model`は`PanelError::InsufficientDegreesOfFreedom`。`neffects>=1`のため
  `df_model>k`が恒常的に成り立ち、`OlsEstimator::fit`自身の`n<=k`チェックより常に厳格
  （FE経由でOLS側の`InsufficientObservations`が発生することは構造的にない）。
- t値・p値・信頼区間の自由度は`cov_type`によらず常に`df_resid`。AIC/BICは`log_likelihood`
  自体（`SSR/n`のみに依存しdf非依存）を`OlsEstimator::log_likelihood()`からそのまま再利用し、
  ペナルティ項の乗数だけ`k`から`df_model`に差し替える。
- **F統計量（`f_statistic`/`f_p_value`）**: 傾き係数`k`個が同時にゼロという帰無仮説のWald
  F検定（`linearmodels.PanelOLS.f_statistic`「H0: All parameters ex. constant are zero」と
  同じ定義。固定効果ダミー自体は検定対象に含めない——`fixest`の`fitstat(m, "f")`はFEダミーも
  含めたモデル全体のF検定で定義が異なるためクロスチェックに使わない）。`wald_f_test`
  （OLS本体の関数を`pub(crate)`化して再利用）をFEの`cov_type`別`cov_params`・`df_resid`で
  呼ぶ形でサンドイッチ計算の複製を避ける（`k_constant=0`固定）。失敗は
  `PanelError::FTestFailed`。

### 3.3 `cov_type`対応

`FeCovType` enum（`Classical`/`Hc1`/`Hc2`/`Hc3`/`Cluster{groups}`/`Hac{bandwidth, time}`）を
`OlsEstimator`の`CovType`とは別に新設（HC0を含まない、無効な組み合わせを型で表現不可能にする
設計）。`OlsEstimator`の既存cov_type計算式はそのまま流用できない——linearmodels/fixestの
ソース確認・実地数値検証で判明した3点の相違:

1. **HC0はスコープ外**（linearmodels・fixestともにパネル/FE向けのHC0オプションが存在しない）。
2. **HC1〜HC3は独自計算**: HC1は小標本補正係数が`n/(n-k)`ではなく`n/df_resid`。HC2/HC3の
   レバレッジは、within変換後の設計行列ではなくLSDV相当のフルレバレッジ
   `h_ii_full = 1/T_entity(i) + h_ii_within`（1-wayは分割回帰＝Frisch-Waugh-Lovellのレバレッジ
   分解則、2-wayはさらに`+ 1/N_time(i) - 1/n`）を使う（fixestと数値一致確認済み）。
3. **Clusterも独自計算**: OLSは`(G/(G-1))×((n-1)/(n-k))`というStata流の小標本補正を常に適用
   するが、linearmodels（FEの主リファレンス）はこの補正を使わず`n/(n-extra_df-k)`のみを
   使う。`extra_df`はcluster変数とFEの関係で決まる（linearmodelsの`_determine_df_adjustment`
   と数値一致確認済み）: 1-way FEで「クラスター変数がentityと同じか、entityを包含するより
   粗い分割」なら`extra_df=0`（`cluster_col`省略時のデフォルト、すなわちentity自体は常に
   この条件を満たす）、それ以外（1-way FEでentityと無関係なクラスター変数、または2-way FE）
   は`extra_df=neffects`。

`cov_type`のデフォルト（`"cluster"`、entity単位）・`cluster_col`文字列パースは
`engine_pybind`層の責務。`FeEstimator::fit`自体はデフォルトを持たない。**2-way FEでも
クラスターのデフォルトはentity単位のまま**（`cluster_col`で上書き可能）。2-way clustering
（entity+time同時）はv1スコープ外。

**Driscoll-Kraay型パネルHAC（`FeCovType::Hac { bandwidth, time }`）**: `linearmodels.panel.
covariance.DriscollKraay`のソース確認に基づく実装。

1. **カーネルはv1でBartlett限定**（OLSの`CovType::Hac`と平仄を合わせる。Parzen/QSは未対応、
   4章参照）。
2. **バンド幅**は`linearmodels`のデフォルトルール`floor(4*(t/100)^(2/9))`（`t`=ユニークな
   時点数、OLSの`hac_lags`が観測数`n`ベースなのと違う点に注意）。明示指定は`[0, t)`範囲検証
   （`PanelError::InvalidHacBandwidth`）。
3. **時系列順序は`time: Vec<String>`の辞書順とみなす**（ISO 8601日付・ゼロ埋め年度等、
   辞書順=時系列順になる形式で渡すことが呼び出し側の契約。`engine`側にこの契約の
   バリデーションは無い）。
4. **1-way/2-way両対応**。1-way FEで`FeCovType::Hac`を指定したのに時系列順序が一切ない
   （`time`も`time_col`も未指定）なら`PanelError::HacRequiresTime`。
5. スケールは`(n/df_resid) × (X̃'X̃)⁻¹ Ŝ (X̃'X̃)⁻¹`（linearmodelsは`cov_type="kernel"`で常に
   `extra_df=neffects`かつデフォルト`debiased=True`のため、素直に`df_resid`と一致する）。
6. **`FeCovType::Hac.time`による明示的な上書き**: `time`が`Some`（`FEOptions.time_col`由来）
   なら`FeInput.time()`より優先してDK計算に使う（`time`未指定の1-way FEでもこれだけでDK HAC
   が成立する）。

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
  二重計上されるバグになる。採用した規約: **`time`の辞書順で最初の値`t_ref`を基準に
  `γ_{t_ref}=0`とし、`α_i`に大域的な水準を吸収させる**（Stata `areg`・R `fixest`/`lfe`等と
  同型の「片方の基準カテゴリを0にする」慣行）。**`fixest::fixef()`との数値一致は`t_ref`の
  選び方が一致する入力に限られる**（`fixest`自身の基準時点選択は`time`列の辞書順ではなく
  観測順で最初に現れた値であるため、行の並び順を変えると`fixest`側の基準時点も変わる。
  2-wayの正規化はどの`t_ref`を選んでも数学的に等価なため、本実装は`fixest`の観測順依存の
  挙動は再現せず辞書順の規約を優先する）。

### 3.6 engine_pybind: エラー変換

`engine::panel::common::PanelError` → `PyErr`（`panel_error_to_pyerr`、`common.rs`）:

| `PanelError` | Python例外 |
|---|---|
| `Common(...)` / `TwoWayRequiresTime` / `UnbalancedPanelForTwoWay` / `SingletonGroup` / `ZeroVarianceRegressor` / `InvalidHacBandwidth` / `HacRequiresTime` | `ValidationError` |
| `InsufficientDegreesOfFreedom` / `WithinRegressionFailed` / `FTestFailed` | `ComputationError` |

`PanelError`はFE/REで共有し、`FeError`/`ReError`は個別に作らない。**engine側に新バリアントを
追加したら`panel_error_to_pyerr`の網羅的`match`も必ず更新すること**（更新漏れは
`cargo build --workspace`の`E0004`で検出される。`cargo test -p engine`だけでは検出できない）。

## 4. テスト

- Python主リファレンス: `linearmodels.PanelOLS`（`cov_type="unadjusted"`/`"clustered"`/
  `"kernel"`等）。Rクロスチェック: `fixest`（`benchmark/panel/run_fixest_benchmark.R`）。
- 許容誤差: 相対誤差`1e-9`を基本（`.claude/rules/testing-policy.md`の基本方針`1e-8`より
  厳しく、実測で機械精度一致が確認できたため）。
- **`aic`/`bic`はRクロスチェック（`fixest`）のみで検証する**: `linearmodels.PanelOLS`は
  `aic`/`bic`を一切提供しないため（`rsquared_within`/`between`/`overall`/`inclusive`・
  `loglik`のみ）、通常の「Python主リファレンス＋Rクロスチェック」の2系統検証の例外
  （ハウスマン検定と同型）。
- **2-wayの`r_squared_within`もRクロスチェック（`fixest`）のみ**（3.4節参照、`linearmodels`
  自身が2-wayでも常にentityのみdemeanという別定義のため）。
- F統計量: `linearmodels`と直接比較（`cov_type="unadjusted"`）。k=1のケースは「1自由度の
  F検定は両側t検定と代数的に等価」という恒等式（`f_statistic = t_stat²`）でHC1/HC2/HC3/
  Cluster/HACを横断検証する。

## 5. 未実装・未対応

- **不均衡パネルでの2-way FE**: 反復的な交互射影（fixest/lfe方式）が必要で、v1では扱わない
  （閉形式の二重デミーニングはバランスパネルでのみ正確なため、6.4節相当の制約として
  `ValidationError`にする）。
- **2-way clustering（entity+time同時）**: v1スコープ外。
- **Driscoll-Kraay HACのカーネル拡張**: Parzen/QSカーネルへの拡張は別issue（Issue #313）。
