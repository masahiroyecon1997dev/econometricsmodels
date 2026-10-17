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
  | `time` | `str \| None` | `None` | 2-way RE（entity + time）の時間次元（`FEOptions.time`と同じ意味）。`None`なら1-way。バランスパネルのみ（不均衡は`ValidationError`）。DKの時点列には使わない（`dk_time`が別に必須） |
  | `dk_time` | `str \| None` | `None` | `cov_type="dk"`時のDK時点列（`FEOptions.dk_time`と同名・同じ意味。`"dk"`では必須で、`time`は借用されない）。RE自身の準偏差変換には使わない。`"dk"`以外で指定すると`ValidationError` |
  | `cluster` | `str \| None` | `None` | `cov_type="cluster"`時のグループキー列名。省略時は`entity`をそのまま使う。他の`cov_type`で指定すると`ValidationError` |
  | `dk_bandwidth` | `int \| None` | `None` | DK HACのバンド幅。省略時は自動計算。`cov_type="dk"`以外で指定すると`ValidationError` |
  | `variance_method` | `str` | `"swamy_arora"` | 分散成分の推定法: `"swamy_arora"`（既定、3.1節）/ `"wallace_hussain"` / `"amemiya"`（3.1.2節。`plm`の`random.method`の`swar`・`walhus`・`amemiya`）。大小無視。1-way・2-wayとも指定でき、末尾に置く（先頭6個の位置引数の順は`FEOptions`と同じ）。未知の値は`ValidationError` |

- **`time`と`dk_time`は別物（`FEOptions`と同じ規約、`panel-common.md`1.1節）**: `time`は2-way REの
  時間次元で、`dk_time`はDriscoll-Kraay HAC（`cov_type="dk"`）専用の時点列。`cov_type="dk"`では
  `dk_time`が必須で、2-wayの`time`を借用しない（意図と違う列が選ばれても気づけないため）。
  2-wayでも`dk_time`は`time`と別の時間粒度を指定できる。エンジンでは`ReInput.time`が2-wayの時間
  次元、DKの時点列は`ReCovType::Dk { time }`（必須の`TimeKeys`）で別に渡す（FEの
  `FeCovType::Dk.time`と同じ）。`time`は`y`・`entity`・`x`と重複できない。
- **`x`は空リストを許容しない**: `x=[]`は「説明変数を一切投入しない、分散成分（σ_ε²・
  σ_u²、ICC）のみを推定するnullモデル」として単独で意味を持つ標準的なユースケース
  （マルチレベルモデルの"null model"）だが、他手法（FE・OLS/WLS/Logit/Probit/IV）
  との一貫性を優先し`validate_x_non_empty`で拒否する。nullモデル・ICC推定のサポート自体は
  別途検討中（5章参照）。
- **1-way REは不均衡パネルを無条件でサポートする**（FEの1-wayと同じ扱い）。**2-way RE
  （`time`指定）はバランスパネルのみ**で、不均衡パネルは`ValidationError`（`validate_balanced_panel`を
  FEの2-wayと共有。不均衡の2-way REは別問題、5章）。検証の順序はFEの2-wayと同じで、singleton検出
  （entity・time）→バランスパネルの検証。
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
`residuals` / `dep_var_name` / `n_obs` / `df_resid` / `df_model` / `n_entities` / `n_periods`（2-wayのみ、1-wayは`None`） / `sigma2_eps` / `sigma2_u` / `sigma2_lambda`（2-wayのみ） / `theta`（1-wayのみ） / `theta_entity` / `theta_time` / `theta_total`（2-wayのみ） / `cov_type` / `dk_bandwidth_used` /
`f_statistic` / `f_p_value`（傾き係数が無ければ`None`） / `log_likelihood` / `aic` / `bic` / `r_squared_within` /
`r_squared_between` / `r_squared_overall` / `hausman_statistic` / `hausman_p_value` /
`hausman_df`。

- **`n_periods`**: `time`を指定した2-wayのときだけユニークな時点数（FEと同じ定義、`fe-spec.md`
  「結果構造体」参照）。1-wayは`None`で、`dk_time`だけの指定でも`None`。2-wayはバランスパネル
  必須のため各エンティティの観測数と一致する。
- **`hausman_*`・`r_squared_*`は1-way・2-wayとも計算される**（3.6・3.7節）。`hausman_*`が`None`に
  なるのは、検定できる傾き係数が無いとき（1-wayは時間変化のある列が無い、2-wayは2-way within変換後に
  分散のある列が無い、または残りがすべて識別できない列）だけ。
- **`variance_method`**: 実際に使った推定法（正規化した小文字の`str`。`sigma2_*`・θが、どの推定法の値かを示す）。
- **分散成分と準偏差変換の重み**: `sigma2_eps`（σ_ε²）・`sigma2_u`（σ_u²、負の推定値は0に切り上げた値）は1-way・
  2-wayとも`float`。`sigma2_lambda`（σ_λ²、同じく0に切り上げ）は2-wayのみで1-wayは`None`。θは、1-wayは
  `theta`（`dict[str, float]`、キーはエンティティIDの文字列表現を辞書順に並べたもの（整数IDでも`"10"`が`"2"`より前）→θ_i。`θ_i`は観測数`T_i`で決まるため不均衡パネル
  でだけ個体間で異なる）、2-wayは`theta_entity`（θ₁）・`theta_time`（θ₂）・`theta_total`（θ₃、θ₁θ₂ではない）
  の`float`で、使わない側は`None`（`n_periods`・`hausman_*`と同じ「該当しなければ`None`」の型安定な形）。
  値は`plm::ercomp`（バランスパネルの`σ²`・`theta`）、1-wayは`linearmodels`の`variance_decomposition`・
  `theta`と一致する。不均衡パネルの`σ_u²`は`plm`と定義が異なる（3.1節）。
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
- `summary()`は実装しない。python_package層（`REResults`）の`params_table()`はFEと同じキー
  構成（`param_name`/`param`/`std_error`/`test_stat`/`p_value`/`conf_lower`/`conf_upper`）。

## 3. 内部実装の計算仕様

### 3.1 分散成分の推定（Swamy-Arora法）

Python `linearmodels.RandomEffects`の実装（Swamy-Arora）に準拠する。R `plm`の
デフォルト（`random.method="swar"`）も同名だが、不均衡パネルでは別の推定量になる
（下記「plmとの既知の実装差」）。

- **σ_ε²（idiosyncratic variance）**: 内部1-way FE推定（`FeEstimator::fit(OneWay,
  Classical)`）のwithin回帰残差平方和／`FeEstimator::df_resid()`。内部FE推定に渡す説明変数は
  within変換後に分散がある列（時間変化のある列、`k_tv`個）だけで、時間不変の列は渡さない
  （FEは分散ゼロの列を拒否するため。REは時間不変の説明変数を許容する、5章）。そのため
  `df_resid() = n - k_tv - n_entities`になる（`plm`と同じ。`linearmodels`は時間不変の列も
  自由度に数えるため`σ_ε²`が食い違う）。全説明変数が時間不変でも成立する（FEは説明変数0個を
  受け付ける）。分散がゼロかどうかの判定はFEの分散ゼロ検出と同じ相対閾値。
- **σ_u²（individual variance）**: between回帰（エンティティ平均`[1, x̄ᵢ.]`への最小二乗）のSSR、
  調和平均`t_bar = n_entities / Σ(1/T_i)`を使う標準式`max(0, ssr/df_resid - σ_ε²/t_bar)`。
  `df_resid = n_entities - K_b - 1`。平均が個体間で一定の列（バランスパネルで時点間でのみ変わる
  列等）は切片と共線になるため、between回帰から落とし`K_b`にも数えない（`plm`と同じ。最終回帰には
  全列が残る。2-wayの規則と同じ、3.1.1節）。さらに、切片・前の列の線形結合になる列も落とす
  （次の「between回帰から落とす列」）。
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

### 3.1.1 2-wayの分散成分（Swamy-Arora、バランスパネル）

`time`を指定した2-wayでは、`plm::ercomp(effect = "twoways", method = "swar")`のバランスパネルの
式と同じ、3つの独立な式で3つの成分を決める（数値照合済み。不均衡は対象外）。`N`はエンティティ数、
`T`は時点数、`n = NT`。

- **σ_ε²**: 内部の2-way FE推定（`FeEstimator::fit(TwoWay, Classical)`、説明変数は2-way within変換後に
  分散のある列`k_tv`個だけ）の残差平方和／`FeEstimator::df_resid()`（`n - N - T + 1 - k_tv`）。
- **σ_u²**: entity平均へのbetween回帰（`[1, x̄ᵢ.]`）の残差平方和を`N - K_b - 1`で割った値から
  `σ_ε²/T`を引く（`max(0, ·)`）。
- **σ_λ²**: time平均へのbetween回帰（`[1, x̄.ₜ]`）の残差平方和を`T - K_b - 1`で割った値から
  `σ_ε²/N`を引く（`max(0, ·)`）。
- **between回帰から落とす列**: 平均が単位間で一定の列（entity平均の回帰では時点間でのみ変わる列、
  time平均の回帰では個体間でのみ変わる列）は切片と共線になるため、その回帰から除き`K_b`にも数えない
  （`plm`と同じ）。一定かどうかは平均の標準偏差が`単位数·ε·max|x̄|`以下かで判定する。
  その後、残りの列のうち切片・前の列の線形結合で表せる列も落とし、`K_b`にも数える対象から外す
  （Rの`lm`と同じ規則: 列順に見て、前の独立な列への射影残差のノルムが元の列のノルムの`1e-7`倍以下
  なら従属。列順で後ろの列を落とす）。個体平均`x̄`を説明変数に加える相関ランダム効果（Mundlak型）
  では、個体平均の回帰で`x`と`x̄`の個体平均が同じ列になり、後ろの列が落ちる。残差平方和とランクは
  どちらを落としても変わらないため`σ_u²`は列順によらない（`plm`の`ercomp`と一致）。時点平均を加える
  2-wayでは時点平均の回帰で同様に別名になる。
  単位数が`K_b + 1`以下なら`BetweenRegressionFailed`（`ComputationError`ではなく`ValidationError`。
  個体平均・時点平均のどちらの回帰かがメッセージに入る。`K_b`は別名を落とした後の数）。別名を落とした後の設計行列は
  通常特異にならない（非有限値などでは同じバリアントで失敗する）。最終回帰の設計行列は全列を残すため、
  完全共線は`QuasiDemeanedRegressionFailed`（`ComputationError`）のまま。
- **θ**: 3つの重み`θ₁ = 1 - σ_ε/σ₁`・`θ₂ = 1 - σ_ε/σ₂`・`θ₃ = θ₁ + θ₂ - 1 + σ_ε/σ₄`
  （`σ₁² = σ_ε² + Tσ_u²`・`σ₂² = σ_ε² + Nσ_λ²`・`σ₄² = σ_ε² + Tσ_u² + Nσ_λ²`）。`θ₃ ≠ θ₁θ₂`で、
  `σ_u² = 0`または`σ_λ² = 0`なら`θ₃ = 0`。`plm`の`ercomp`のθ（id・time・total）と機械精度で一致する。

### 3.1.2 `variance_method`: Wallace-HussainとAmemiya

`variance_method`は分散成分（σ_ε²・σ_u²、2-wayはσ_λ²も）の推定法を選ぶ。GLS変換・`cov_type`・
ハウスマン検定・パネル固有R²の式は推定法によらず同じで、変わるのは分散成分と、そこから決まるθ・
係数・標準誤差・F統計量だけ（内部のFE推定は全推定法で作り、ハウスマン検定とwithin R²が再利用する）。
参照実装は`plm`（`random.method = "walhus"`・`"amemiya"`）だけで、`linearmodels`にはSwamy-Arora
しか無い。

3法は同じ枠組みで、1段目の回帰の残差`e`から二乗和を作り、期待値の行列`M`と連立させて解く。負の
σ_u²・σ_λ²は0に切り上げる。

- **1段目の回帰**: `wallace_hussain`は切片と全説明変数へのプールドOLS（時間不変の列も入れる）、
  `amemiya`はwithin推定（内部FEの係数。切片は`ȳ - x̄'β̂_within`）。
- **二乗和**: `Q_w = Σ(within変換した e)²`、`Q_id = Σ_i T_i ē_i²`、2-wayは`Q_ts = Σ_t N_t ē_t²`。
  within変換は1-wayがentity、2-wayがentityとtime。
- **バランス**（1-wayは全個体の観測数が等しい、2-wayは常に）は`plm`の`dfcor = 1`の閉形式:
  `σ_ε² = Q_w/(O - N)`（2-wayは`O - N - T + 1`、`K`は引かない）、`σ_u² = Q_id/(T·N) - σ_ε²/T`、
  `σ_λ² = Q_ts/(N·T) - σ_ε²/N`（`O`は総観測数、`N`は個体数、`T`は時点数）。
- **不均衡の1-way**は`plm`の`dfcor = 3`（固定）のトレース項の式で、`M·(σ_ε², σ_u²)ᵀ = (Q_w, Q_id)ᵀ`を
  2×2の連立として解く。`Z = [1, X]`、`B = Σ_i T_i z̄_i z̄_i'`、`S = Σ_i T_i² z̄_i z̄_i'`、
  `W = Z'Z - B`として、`wallace_hussain`は`M_w,ν = O - N - tr((Z'Z)⁻¹W)`、
  `M_w,η = tr((Z'Z)⁻¹W(Z'Z)⁻¹S)`、`M_id,ν = N - tr((Z'Z)⁻¹B)`、
  `M_id,η = O - 2tr((Z'Z)⁻¹S) + tr((Z'Z)⁻¹B(Z'Z)⁻¹S)`。`amemiya`は`M_w,ν = O - N - K`、`M_w,η = 0`、
  `M_id,ν = N - 1 + tr((X̃'X̃)⁻¹C)`（`C = Σ_i T_i(x̄_i - x̄)(x̄_i - x̄)'`）、`M_id,η = O - ΣT_i²/O`。
  不均衡の2-wayは対象外（`ValidationError`のまま）。

閉形式とトレース形は、同じバランスパネルでも値が異なる（`plm`が切り替えている）。つまり、1行落として
不均衡にすると`wallace_hussain`・`amemiya`の値は連続でなくなる。1-wayは時点列を持たないため、`plm`の
`is.pbalanced`（同じ時点集合も要求する）ではなく「全個体の観測数が等しい」でバランスを判定する。
観測数が等しく時点がずれたパネルでは、`plm`がトレース形を使う一方で本実装は閉形式を使うため、値が
異なる（テストのパネルでσ_u²が約5〜11%）。

**`amemiya`が拒否する入力**: 1段目がwithinだけなので、within変動の無い説明変数（時間不変の列。2-wayでは
個体間でのみ・時点間でのみ変わる列も）があると、その効果が残差に残ってσ_u²が偏る。`plm`は不均衡
パネルでは拒否し、バランスパネルでは時間不変の列を無視して推定する。本実装は一律に`ValidationError`で
拒否し、列の名前を示して`swamy_arora`・`wallace_hussain`を案内する（時点間でのみ変わる1-wayの列は
個体内で変動するため使える）。`wallace_hussain`は時間不変の列・全列が時間不変でも推定できる。

**失敗する入力**: `wallace_hussain`のプールドOLSが特異（`ComputationError`）、σ_ε²が正でない
（`ComputationError`。`plm`は0に切り上げてθが`NaN`になる）、トレース形の連立が特異
（`ComputationError`）。内部のFE推定の失敗（singleton・自由度不足）は`swamy_arora`と同じ。

**`swamy_arora`の不均衡1-way**は3.1節の式（`linearmodels`準拠）のままで、`plm`の`swar`のトレース形には
切り替えない（`plm`互換の推定法を別の値として足すかは要望が出てから検討する）。

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

**2-wayのGLS変換**は`y - θ₁ȳᵢ. - θ₂ȳ.ₜ + θ₃ȳ..`で、`y`・各`x`列と切片復元用の定数列（変換後は
`1 - θ₁ - θ₂ + θ₃`倍）に適用してから`OlsEstimator::fit(include_intercept=false)`に渡す。
`quasi_demean_column`は1つのグルーピングしか扱えず、FEの2-wayのように「entityでquasi-demean→
その結果をtimeでquasi-demean」の2段階適用（θ₁・θ₂のみ使う）では`θ₃ ≠ θ₁θ₂`のためGLS変換に
ならない。そこで3つの平均を同時に引く`quasi_demean_two_way`（`panel/common.rs`、バランスパネル前提）を
用意した。θ=(1,1,1)で2-way within変換、θ=(θ₁,0,0)・(0,θ₂,0)で1-wayのRE（entity方向・time方向）、
θ=(0,0,0)で変換なし（pooled OLS）になる。

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
分類/HC/cluster/DriscollKraay計算）は`common.rs`にFE/RE共有で移設済み（素のレバレッジは
`shared::covariance::leverages`）——
数式自体はFE実装時から変更しておらず、`k_correction`・レバレッジ配列を引数で受け取る汎用実装の
ため呼び出し側（REは常に`k_correction=df_model`・素のレバレッジ）を差し替えるだけで
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
  `time`（必須の`TimeKeys`、`REOptions.dk_time`由来）を持つ。2-wayの`time`（`ReInput`）とは独立で借用しない。
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
`stat_df`と常に一致する（`tests/cross_method/test_test_dfs.py`が全`cov_type`で固定している）。`classical`のとき
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
  （`tests/support/_tolerances_panel.py`の`re_crosscheck.rtol_unbalanced_f`）。
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
`df_model==1`なら3種とも`0.0`とする。**2-wayでも定義は同じで、withinだけがFEの2-wayと揃えて個体と時点の
両方でdemeanする**: `r_squared_within`は2-way within変換した`y`・`x`に傾きを当てはめたR²（時間変化のある
列が無ければ`0.0`）、`r_squared_between`は個体平均への当てはめ、`r_squared_overall`は生の`y`・`x`への
当てはめ（どちらも切片込みの中心化TSS）。`plm`の`r.squared`は変換後の回帰のR²を返すだけで対応する参照が
無く、検証は`plm`の係数から定義どおりに計算した独立な値との照合になる。`r_squared_between`が負値になりうる（`linearmodels`のSST/SSR方式のF統計量と
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
  `χ²_k.sf(stat)`、`hausman_df = k`。`k`は傾き係数のうち時間変化のある列の数`k_tv`で、
  `X*`は全傾き、`X̃`は時間変化のある列だけ（時間不変の列のwithin変換後は全ゼロ列で、比較に
  寄与しない）。さらに、**`X̃`の列のうち識別できない列は検定対象から外す**: バランスパネルで時点間で
  のみ変わる列（年トレンド等）は個体平均が個体間で一定になり、`x*_j = x̃_j + 定数`で未変換の定数項と
  共線になるため、その係数は識別できない（`x*_j - x̃_j`が全観測で一定かで判定する）。外した分だけ
  `hausman_df`は小さくなり、検定できる列が残らなければ`None`。時点間でのみ変わる列が無い入力の結果は
  変わらない（`plm`の`phtest(method = "aux")`はこの入力で添字エラーになる）。個体平均・時点平均を
  説明変数に加える相関ランダム効果（Mundlak型）では、`x̃_j`が定数項・`X*`・前の`x̃`の線形結合になる
  列（個体平均を加えた`x_j`の`x̃_j`）も同様に外す（次の「識別できない`X̃`の列の判定」）。統計量は構造的に非負で、二次形式版の
  `abs()`や非正定値の問題が生じない。バランスパネルでは共通σ²を使った古典的ハウスマン
  検定と数値的に一致する。
- **定数項の扱い**: `plm`は`reX`から定数列を除き補助回帰に未変換の`1`を足す。バランス
  パネルでは`θ_i`が全個体共通のため変換済み定数列を使う版と同値だが、不均衡パネルでは
  値が異なる。本実装は`plm`に合わせる。
- **比較相手はRE本体と同じ個体効果構造のFE**: 1-wayはentityのみ、`time`を指定した2-wayは
  entity+timeの2-way FE。`X̃`はRE本体が分散成分推定に使う内部FE（1-wayは1-way、2-wayは2-way）の
  within変換から得る（2-wayでは2-way within変換後に分散のある列だけ、`hausman_df`はその数）。
  `REOptions.dk_time`（`cov_type="dk"`専用）の有無は結果に影響しない（`dk_time`は`cov_type="dk"`の
  時点列にのみ使う）。2-wayの補助回帰の`X*`は3つのθで変換した全傾き、`y*`も同じ変換で、定数項は
  未変換の`1`（1-wayと同じ）。
- **識別できない`X̃`の列の判定**: まず列ごとに`x*_j - x̃_j`が全観測で一定の列を外す。1-wayのバランス
  パネルの時点間でのみ変わる列のほか、2-wayでは個体平均も時点平均も一定の交互作用列（`x* = x̃`）が
  該当する。次に、残った`x̃_j`のうち、定数項・`X*`・前の`x̃`の線形結合になる列を外す（Rの`lm`と
  同じ規則と許容誤差`1e-7`、列順で後ろの列を落とす）。個体平均`x̄ᵢ`を説明変数に加える相関ランダム
  効果（Mundlak型）では、`x*_j - x̃_j = (1-θ)·x̄ᵢ`が加えた個体平均の列で表せるため`x̃_j`が別名になる
  （2-wayでは個体平均と時点平均の両方を加えたとき）。落とすのは検定対象の`x̃`側だけ（最終回帰が
  通っていれば`[1, X*]`は通常独立で、`x̃`は補助回帰の列順で最後のため`X*`の列は落ちない）で、`X*`の列同士が従属なら
  補助回帰が特異になり`fit()`が失敗する。複数の`x̃`が別名になる入力では、どの`x̃`が落ちるかが列順に
  依存する。別名を落とすと検定する帰無仮説は、加えた平均の列を許した条件付きの「残りの`x̃`の係数がゼロ」になる
  （落とした`x̃`の係数は、加えた平均の列の係数と区別できない）。許容誤差`1e-7`は非中心化ノルムに対する相対値で、
  平均に対して変動が`1e-7`倍以下の列は定数に近い列として落ちる。許容誤差を超えていても相対`1e-4`程度までの近共線
  （別名ではないが近い列）は、ロバスト共分散の補助回帰が数値的にほぼ特異になって`ComputationError`になることがある。`plm`の`phtest(method = "aux")`は検定する係数を位置で選ぶため、この
  入力で別の係数を検定し、統計量が式の列順で変わる（参照値には使わない。テストは名前で選ぶ）。
  `Amemiya`は時間不変の個体平均の列を拒否するため、相関ランダム効果とは併用できない
  （`swamy_arora`・`wallace_hussain`で使う）。
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
- **`None`になるのは検定できる傾き係数が無い場合のみ**（時間変化のある列が無い、または残りがすべて
  別名の列のとき。説明変数がすべて時間不変の場合や、全ての時間変化列の個体平均を加えた場合に
  発生する。`x=[]`は拒否される）。補助回帰のランク落ち・Wald検定の特異性など、計算自体が成立しない場合は
  `hausman_*`を`None`にせず`RE.fit()`が失敗する（設計行列の多重共線性でエラーにするのと
  同じ方針）。ロバスト共分散が構造的に特異になる場合が典型で、`cluster`ではクラスター数`G`が
  補助回帰の傾き係数の数（`X*`の全傾きと、別名を除いた`X̃`の和。時間不変の列も別名も無ければ`2k`）以下
  （`ValidationError`、OLSの`InsufficientClustersForInference`と
  同じ事前検証）、`dk`では時点数`T`が検定対象の`k`個以下（`T <= k`、`ValidationError`。
  DK共分散のrankは`T-1`以下のため。FEのF検定と同じ事前検証
  `PanelError::InsufficientDkPeriodsForInference`）。RE本体は成功する入力でも、`cov_type="cluster"`（既定）で
  `G <= 2k`の場合はfitが失敗するため、`cov_type="classical"`等を指定するか
  クラスター数を増やす。singleton entity等による内部1-way FE推定の失敗は、
  分散成分（σ_ε²）推定が先に失敗するため`RE.fit()`自体が失敗する（時間不変変数は
  内部FE推定に渡さないので失敗の原因にならない）。
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
- 許容誤差: Classical/HC1は`linearmodels`と相対誤差`1e-8`（`tests/support/_tolerances_panel.py`の
  `re_reference`。実測は`1e-9`〜`1e-14`程度）で数値完全一致。Cluster/DK・
  HC2/HC3の`plm`クロスチェックは、バランスパネルでは機械精度で一致する（`1e-8`）。不均衡
  パネルのみ分散成分推定（Swamy-Arora）が`plm`とlinearmodels準拠の本実装で僅かに異なるため、
  統計量・cov_type別に実測へマージンを載せて緩める（係数`5e-3`、se・t・p値・信頼区間は
  cluster `5e-3`・hc2/hc3 `2e-2`・dk `5e-2`。`tests/support/_tolerances_panel.py`の`re_crosscheck`参照。
  一律に緩めるとclusterの`G/(G-1)`欠落（seに約1.3%）を見逃すため分けている）。
- **ハウスマン検定は`plm::phtest(method = "aux", effect = "individual", vcov = ...)`のみを参照値と
  する例外規定**（`linearmodels`のソースに`hausman`という文字列が一切登場せず専用実装が
  無いことを確認済み。通常の「Python主リファレンス＋Rクロスチェック」の2系統検証の例外）。
  バランスパネルでは機械精度で一致し、不均衡パネルではSwamy-Arora分散成分（σ_u²）の
  推定式の差で数％ずれるため、シナリオ別の許容誤差で比較する（3.7節）。`cov_type`別
  （`benchmark/panel/run_plm_hausman_benchmark.R`）に全6種を照合し、ロバスト共分散が構造的に
  特異なシナリオ（`many_regressors`のcluster/dk）は`fit()`が`ValidationError`/`ComputationError`で
  失敗することを確認する。
- **`variance_method`（Wallace-Hussain・Amemiya）**: 参照実装は`plm`（`random.method = "walhus"`・
  `"amemiya"`）のみで、`linearmodels`には無い（単一リファレンスの例外）。`tests/panel/test_re_variance_method.py`が、
  `re_variance_method.json`（`benchmark/panel/fixtures/generate_re_variance_method_fixtures.py`、1-wayは
  バランス・不均衡（トレース項の式）・準バランス・時間不変の列・時点間のみ変わる列・実データ、2-wayは
  バランス）の分散成分・θ・係数・標準誤差・F統計量・ハウスマン検定を代表ケースで全`cov_type`照合し、
  plmのソースどおりに観測ごとの行列で書いた手計算（`tests/support/_re_oracle.py`）とも比べる。許容誤差は
  `tests/support/_tolerances_panel.py`の`re_variance_method`（機械精度。絶対年の列を含む入力のみ丸め誤差が拡大する統計量で
  `1e-6`）。手計算との照合は全ケース×全`cov_type`で行い、plmの準偏差変換済みデータへのstatsmodels
  OLSで対数尤度・AIC・BICとクラスターロバスト推論（`G-1`の自由度）を全ケースで照合する（パネル固有R²は
  手計算だけが参照）。個体効果なし・時点効果なしのDGPで`σ_u²`・`σ_λ²`が0に切り上げられる場合
  （θ=0、GLSがプールドOLSに退化）も、バランス・不均衡の1-wayと2-wayでplmと照合する。観測数が
  等しく時点がずれたパネルは、plmと値が異なること（本実装は閉形式、plmはトレース形）自体を固定する。
  拒否する入力（Amemiyaとwithin変動の無い列。複数列・全列・実データ・2-wayを含む）はフィクスチャを持たず
  例外だけを確かめる。不均衡の1-wayのWallace-Hussainで`σ_ε²`が負になる入力（`ComputationError`）と、
  個体数が傾き係数の数+1のとき`swamy_arora`だけが失敗することも固定している。
- `aic`/`bic`は`estimator()`（内部`OlsEstimator`）からそのまま取得するため、FEのような
  Rクロスチェック限定の例外は無く`linearmodels`と直接比較できる。

## 5. 未実装・未対応

- **nullモデル・ICC推定のサポート**（`x=[]`、検討中）: 分散成分のみを推定する
  マルチレベルモデルの標準的なユースケースだが、v1は他手法との一貫性を優先し`x`の空リストを
  拒否している。
- **2-way RE**（バランスパネルの本体は実装済み: `time`オプション・Swamy-Arora分散成分・3つのθの
  GLS変換・F統計量・`cov_type`・`n_periods`・ハウスマン検定・パネル固有R²。**未実装**:
  不均衡パネル。分散成分の公開と`variance_method`は実装済み。以下は設計検討で確定した方針）:
  - **対象はバランスパネルのみ**。不均衡パネルは、FEの2-wayと同じく`ValidationError`で拒否する
    （`validate_balanced_panel`相当の検証を共有する）。アンバランスパネルの2-way REは教科書レベルの
    閉形式が無く、Wansbeek and Kapteyn (1989)の推定量が候補になる別問題として切り分ける。
  - **分散成分の推定は、Swamy-Arora法の2-way拡張を既定とする**（1-wayと推定量の系統を揃え、
    `plm`の`random.method`既定値とも一致する）。Wallace-Hussain法・Amemiya法はオプションで選べるように
    するが、実装はSwamy-Aroraの後に回す。オプションの名前・値の表記は実装時に決める。Nerlove法は
    対象にしない。`plm`の`ercomp`で、3法は別々の分散成分・θを返すことを確認済み。
  - **分散成分の推定法を選ぶオプション`variance_method`は、Swamy-Arora以外の推定法を実装する時点で
    追加する**（最初のリリースでは設けない。選択肢が1つのオプションを公開しないため）。追加するときは
    1-wayにも2-wayにも付け（片方だけだと非対称になるため）、既定は`"swamy_arora"`として、既定の挙動
    （3.1節、`linearmodels`準拠）は変えない（追加は破壊的変更にならない）。名前は`variance_method`
    （`cov_type`と取り違えず、`plm`の`random.method`と同じ語）、値は省略しない小文字のスネークケース
    （`"swamy_arora"`・`"wallace_hussain"`・`"amemiya"`、`cov_type`と同じく大文字小文字は区別しない）。
    Wallace-Hussain・Amemiyaの参照実装は`plm`のみで、`linearmodels`には無い。
  - **`plm`の分散成分推定の構造**（`plm` 2.6.7の`ercomp`のソース確認と、1-wayの3法をRで再実装した
    数値照合による。バランス・不均衡とも`ercomp`の出力と小数8桁まで一致）。3法は同じ枠組みで、
    1段目の回帰の残差から2つの二乗和（within型の`Q_w`、個体平均型の`Q_id`）を作り、期待値の行列`M`と
    連立させて`M·(σ_ε², σ_u²)ᵀ = (Q_w, Q_id)ᵀ`を解く。負の分散成分は0に切り上げる。θは`1 - (1 + T_i σ_u²/σ_ε²)^(-1/2)`
    （3.2節と同じ式）。1段目の回帰は、`swar`がwithin＋between、`walhus`がpooled OLS、`amemiya`が
    within（切片は`ȳ - x̄'β̂_within`）である。
    - **バランスパネル**（`T`は全個体共通、`O`は総観測数、`N`は個体数、`K`は傾き係数の数）は簡約形で
      解ける。`swar`は`σ_ε² = Q_w/(O-N-K)`・`σ_u² = SSR_b/(N-K-1) - σ_ε²/T`で、3.1節の式
      （`linearmodels`）と一致する。`walhus`・`amemiya`は`σ_ε² = (残差のwithin二乗和)/(O-N)`・
      `σ_u² = ((個体平均残差の二乗和)/N - σ_ε²)/T`で、残差の出所だけが異なる。
    - **不均衡パネル**では`plm`は自由度の補正種別を3に固定し（他の値を指定するとエラー）、`(Z'Z)⁻¹`を含む
      トレース項でモーメント方程式を組む。`swar`と`amemiya`はσ_ε²が同じ値になる（どちらもwithin残差を
      `O-N-K`で割る）。`walhus`はpooled残差を使うため別の値になる。σ_u²は3法とも異なる。
    - **不均衡の`swar`と3.1節の違いの正体**: `plm`のbetween回帰は個体平均を観測レベルに展開して回帰する
      ため、`T_i`で重み付けしたWLSになる。3.1節（`linearmodels`準拠）は個体平均への重みなしOLSと調和平均
      `t_bar`を使う。`plm`のソースには調和平均`t̄`を使う分岐が残っているが、不均衡パネルでは補正種別が
      3に固定されるため到達しない。
    - **方針**: 不均衡パネルの`"swar"`は3.1節の式（`linearmodels`準拠）を維持し、`plm`式への切り替えは
      行わない。`plm`の式を別の値として足すかどうかは、Rの計算慣習への互換モードの要否と合わせて、
      具体的な要望が出た時点で検討する。一方、`walhus`・`amemiya`は参照実装が`plm`のみなので、
      不均衡パネルを含めて`plm`の式に従う。
    - **実装**（3.1.2節）: 「1段目の回帰」と「行列`M`の組み立て」だけを推定法ごとに分け、バランス用の閉形式と
      不均衡の1-way用のトレース形の2系統を持つ（`engine/src/panel/re_variance_method.rs`）。`plm`は2-wayの
      不均衡パネルにも3法のモーメント行列を持つが、数値検証はしておらず、対象外のまま。
  - **θは3つの重みが要る**。2-wayのGLS変換は`y - θ₁ȳᵢ. - θ₂ȳ.ₜ + θ₃ȳ..`の形で、`plm`の
    `ercomp`が返す`theta$total`は`theta$id × theta$time`と一致しない（3法とも別の値、確認済み）。
    このため、FEの2-way within変換と同じ「entityでquasi-demean→その結果をtimeでquasi-demean」の
    2段階適用（θ₁・θ₂のみ使う）をそのまま流用すると、θ₃が合わずGLS変換にならない。現行の
    `quasi_demean_column`（1つの`GroupCodes`とそのθ配列を取る）は、3つの重みを扱える変換を
    新設する形で拡張する。
  - **`time`オプションを導入する**（実装済み）。FEの`FEOptions.time`と同じ意味（2-wayの時間次元）で
    `REOptions.time`を新設する。`dk_time`（DK HACの時点列）は引き続き別フィールドとして残し、
    2-wayの`time`をDKの時点に借用しない（FEと同じ規約、`panel-common.md`1.1節）。
  - **参照実装は`plm`のみ**（`model = "random"`・`effect = "twoways"`）。`linearmodels`は2-way REを
    持たないため、「Python主リファレンス＋Rクロスチェック」の2系統検証の例外になる
    （ハウスマン検定・FEのAIC/BICと同型）。許容誤差はバランスパネルの機械精度（`1e-8`）を基本とする。
  - **変換後の回帰・自由度・F統計量・`cov_type`は1-wayの規約をそのまま使える**（`plm` 2.6.7の
    `effect = "twoways"`、バランスパネルで数値確認済み）。3つのθで変換した`y`・`x`・定数列
    （定数列は`1 - θ₁ - θ₂ + θ₃`倍になり、`σ_ε² > 0`なら0にならない）を`include_intercept=false`で
    最小二乗に当てると、係数・標準誤差が`plm`と機械精度で一致する。
    - **`df_resid = n - k`・`df_model = k`**（`k`は変換済み定数列を含む列数）。`plm`の`df.residual`
      も同じ値。FEのように個体・時点ダミーぶんの自由度は消費しない。
    - **F統計量はWald形式**（傾き係数`k-1`個、`F(k-1, df_resid)`、`cov_type`連動）。`plm::pwaldtest(test =
      "F")`の統計量と手計算の二次形式`β'V⁻¹β/(k-1)`が一致する。`linearmodels`のSST/SSR型は2-wayに参照が
      無く、3.5節の理由（負値になりうる）からも採用しない。
    - **`cov_type`の小標本補正は1-wayと同じ**（REは`extra_df=0`）。変換後データの素のレバレッジ
      `h_ii = x_i(X'X)⁻¹x_i'`でHC2/HC3を計算し、`plm::vcovHC`のwhite1（HC1/HC2/HC3）・arellano（sss、
      entityクラスター）が、変換後データの手計算サンドイッチと機械精度で一致する。`vcovSCC`も2-wayで
      動く（数値照合は実装時）。クラスターの既定はentityのまま、2-way clustering（entity+time同時）は
      v1スコープ外。
    - **分散成分が0になる場合**: `σ_u² = 0`または`σ_λ² = 0`のとき、`θ₃`は定義式から0になる
      （`plm`も同じ扱い）。片方が0なら残りの成分だけの1-way REに、両方0ならpooled OLSに退化するため、
      1-wayの「`σ_u² = 0`でpooled OLSと一致する」性質と同型の回帰ガードにできる。
  - **2-wayのハウスマン検定は、`time`を指定したとき2-wayで行う（補助回帰版、1-wayと同じ方式）**:
    「3つのθで変換した`y*`を、変換前の定数項・変換済みの全傾き`X*`・**2-way within変換済み**`X̃`
    （2-way within変換後に分散のある列のみ）に回帰し、`X̃`の係数をWald検定する」。比較相手はRE本体の
    構造に合わせた2-way FEで、1-way within`X̃`を使うと別の値になる。`time`を指定しない（1-way）ときの
    挙動は変えない。
    - **`cov_type`連動も1-wayと同じ**（補助回帰の共分散はRE本体の`cov_type`に対応、3.7節の表）。
      `plm::phtest(method = "aux", effect = "twoways", vcov = ...)`が、`X`の全列が両次元で変動する
      データでclassical・HC1・HC2・HC3（`vcovHC`のwhite1）・cluster（arellano・sss、entity）の
      全てについて、補助回帰の手計算と機械精度（相対`1e-14`）で一致することを確認済み。DK
      （`vcovSCC`、`type = "sss"`・Bartlett重み）も同じ入力で手計算と相対`7e-14`で一致する
      （スケールは`T/(T-1)·(n-1)/(n-K)`、`K`は補助回帰の説明変数の数）。
    - **検定対象は`X̃`の列数**（2-way within変換後に分散のある列の数、以下`k_tv`）。時点間でのみ・個体間で
      のみ変わる列は`X*`には入るが`X̃`には入らない。`k_tv = 0`なら`None`。クラスター数の
      事前検証は補助回帰の傾き係数の数（`X*`と`X̃`）に対して、DKの時点数の事前検証は1-wayと同じく
      RE本体の傾き係数の数に対して行う（検定対象の係数の数より保守的）。
    - **参照**: 全列が両次元で変動するシナリオは`plm`で照合する。時点間でのみ・個体間でのみ変わる列を含む
      シナリオは、`plm`が係数を位置で選ぶため`X̃`ではない係数を検定してしまい、手計算（独立な補助回帰）
      との照合になる（plmとの食い違いを実測: 同じデータで`plm` 5.388・手計算3.291）。
    - **実装**: `re_hausman_test`は、`X̃`を内部FE推定量（分散成分用）が持つwithin変換済み設計行列から
      取る。2-wayでは分散成分用に作る2-way FE推定量（`k_tv`列のみ）がそのまま渡り、補助回帰以降の計算は
      1-wayと共通。`dk_time`は引き続きDKの時点列として
      `time`とは独立に扱う（FEと同じ規約）。
  - **パネル固有R²は1-wayと同じ3種を持つ**。定義はFEの2-wayと揃える: `r_squared_within`は2-way
    within（個体＋時点でdemean）した`y`・`x`に`β̂`を当てはめたR²、`r_squared_between`は個体平均に
    対する当てはめ、`r_squared_overall`は生の`y`・`x`に切片込みで当てはめたR²（3.6節と同じ、切片`β0`を
    含める中心化TSS）。`plm`の`r.squared`は変換後回帰のR²を1つ返すだけで、この3種に対応する参照が
    無い。そのため検証は外部パッケージとの数値比較ではなく、ベンチマーク生成スクリプトで`plm`の係数から
    定義どおりに計算した独立な値との照合になる（外部リファレンスが無い統計量の例外扱い。テスト実装時に
    コメントとドキュメントへ明記する）。
  - **時間不変変数（および片方の次元でしか変動しない説明変数）は許容する**。REは個体効果・時点効果が
    説明変数と無相関という前提を置くため、時間不変変数や、時点間でのみ（個体間でのみ）変わる変数の
    係数も識別できる（時点効果と共線になるのはFEの話）。1-way・2-wayとも実装済み（3.1・3.1.1・3.7節）で、
    2-wayは片方の次元でしか変動しない変数も許容する（内部の2-way FE推定には2-way within変換後に
    分散のある列だけを渡し、between回帰からは平均が単位間で一定の列を落とす）。
    - **分散成分（σ_ε²）**: 内部のFE推定には、within変換後の分散がある列（時間変化のある列、以下
      `k_tv`個）だけを渡す。FEは説明変数0個も受け付けるため、全列が時間不変でも成立する。
      `σ_ε² = SSR_within / FeEstimator::df_resid()`の式は変わらず、自由度は実際に推定したwithin係数の
      数`k_tv`を反映する（`plm`と同じ。`linearmodels`は時間不変の列も自由度に数えるため値が食い違う）。
    - **最終回帰・`df_model`・F統計量・`cov_type`・between/overall R²**: 全列を使い、変更なし。
      `r_squared_within`は時間変化のある列の係数だけを使う（時間不変の列のwithin変換後の値は0）。
    - **ハウスマン検定**: `y*`を`[1, X*（全傾き）, X̃（時間変化のある列のみ）]`に回帰し、`X̃`の係数だけを
      Wald検定する。`hausman_df = k_tv`（3.7節の識別できない列を外した場合はその分減る）で、
      検定できる列が無ければ`None`。事前検証（クラスター数・DKの時点数）は
      検定対象の係数の数`k_tv`と補助回帰の説明変数の数のどちらを使うかを整理して直す。`plm`の
      `phtest(method = "aux")`は、時間不変変数があると係数を位置で選ぶため`X̃`ではなく`X*`の係数を検定
      してしまい（`reX`と`feX`の列数が等しい前提）、参照として使えない。この場合の参照値は、
      ベンチマーク生成スクリプトでの独立な手計算になる。
    - **参照実装**: 分散成分・係数・`cov_type`は`plm`が基準（時間変化のある変数だけなら`linearmodels`と
      一致する。時間不変変数を含むシナリオだけ`linearmodels`は使えず、「Python主リファレンス＋Rクロス
      チェック」の例外になる）。
    - 切片と共線な定数列・時間不変変数同士の共線など、設計行列が特異になる入力は特異性検出
      （`ComputationError`）に任せる。これらの列は、between回帰（σ_u²の推定）では別名として落ちるため、
      最終回帰（準偏差変換後、全列を残す）の特異性（`QuasiDemeanedRegressionFailed`）で検出される。
  - **`REResult`に`n_periods`を追加する**（実装済み。FEと同じ定義: `time`を指定した2-wayのときだけ
    ユニークな時点数、1-wayは`None`で、`dk_time`だけの指定でも`None`）。`fe-spec.md`2章・
    `panel-common.md`2.1節の記述も直した。
  - **分散成分を結果として公開する**（実装済み、2章）。`variance_method`を追加するときも、この項目が
    「どの推定法の値か」を示す。
