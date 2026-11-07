# IV 仕様書

IV（操作変数法: 2SLS/GMM）の確定済み仕様。`engine/src/iv/`（`two_sls.rs`/`gmm.rs`、共通基盤は
`engine/src/iv/common.rs`と`common/`）・`engine_pybind/src/iv/common/`・
`python_package/econometricsmodels/iv/iv.py`として実装済み。`estimator="2sls"`/`estimator="gmm"`を
単一の`IV`/`IVResults`ペアで扱う設計のため、本ドキュメントは2SLS/GMM両方を扱う。

## 1. API引数

3層構成: `IV(data, y, x_exog, x_endog, instruments, options=options).fit() -> IVResults`
（python_package）→ `fit_iv(data, y, x_exog, x_endog, instruments, options) -> IVResult`
（engine_pybind）→ `TwoSlsEstimator::fit` / `GmmEstimator::fit`（engine、`IVOptions.estimator`で
振り分け）。

### 1.1 `y` / `x_exog` / `x_endog` / `instruments`

- 引数とOptionsの置き場所の基準と、`x`ではなく`x_exog`/`x_endog`にする理由は[`api-conventions.md`](./api-conventions.md)を参照。
- `y: str`、`x_exog: list[str]`（外生説明変数）、`x_endog: list[str]`（内生説明変数）、
  `instruments: list[str]`（操作変数）をすべて独立の引数として渡す（OLSの`y`/`x`と同格、
  bareネーミング）。
- **`instruments`は除外操作変数（excluded instruments）のみ**を指定する。`x_exog`に含めた
  列を`instruments`に重複して渡す必要はない（渡すと`ValidationError`）。第一段階の設計行列
  （全操作変数`Z`）は内部で`x_exog ++ instruments`をunionして構築する（Stata
  `ivregress`・`linearmodels.IV2SLS(dependent, exog, endog, instruments)`と同じ方式）。
  過剰識別検定の自由度が`len(instruments) - len(x_endog)`とそのまま一致し、`x_exog`分を
  差し引く補正が不要になる。
- `x_exog`は空リストを許容する（内生変数のみのモデルも成立するため）。**`x_endog`/
  `instruments`はいずれも独立に最低1要素を要求し、空リストは`ValidationError`**。
  `x_endog=[]`は実質OLSと等価な退化ケースであり「そもそもIVを使用すること自体が誤り」と
  判断し、`OLS`への切り替えなしにそのまま`IV`に渡せる利便性よりも誤用防止を優先した。
  `x_endog`/`instruments`は独立に検証するため、「操作変数はあるが対応する内生変数が無い」
  誤用も検出できる。
- **識別の順序条件**（`len(instruments) >= len(x_endog)`）を満たさない場合は`fit()`冒頭
  （第一段階回帰の前）で`IvError::InsufficientInstruments`（`ValidationError`）にする。
- **観測数**（`engine::iv::common::validate_iv_fit_inputs`が呼ぶ`validate_observation_counts`）は`fit()`冒頭（第一段階回帰の前）で、
  構造方程式の`n <= k`（`k = k_exog + k_endog`、定数項込み）を共通の`InsufficientObservations`で、
  次に第一段階の`n <= l`（`l = k_exog + len(instruments)`、定数項込み）を
  `IvError::InsufficientObservationsForFirstStage`（`ValidationError`）で拒否する。`l >= k`
  のため構造方程式を先に報告しないと第一段階の`k`に隠れる。`n <= l`は診断のdegradeでは済まない:
  `n = l`では第一段階が完全に当てはまり`X̂ = X`となって2SLSがOLSと一致し（GMMも
  `W₀ = (Z'Z)⁻¹`でOLSに縮退）、`n < l`では`Z'Z`が特異になる。クラスター数が第一段階の
  傾き係数の数以下のとき全体F検定だけを`None`にできる（3.1節）のとは異なり、`X̂`の元になる推定
  そのものが成立しない。
- **被説明変数が定数・完全フィット**: 切片があり`y`が全行で同じ値なら`OUTCOME_CONSTANT`
  （`ValidationError`、`validate_iv_fit_inputs`が呼ぶ`validate_outcome_not_constant`。第一段階より前に弾く）。構造残差
  $y - X\hat\beta$（第二段階回帰の残差ではない）の平方和が`y`の総平方和の`k·ε`倍以下なら
  `PERFECT_FIT`（`ComputationError`、判定は`ols-spec.md`3.3節と同じ`is_perfect_fit`）。GMMは
  重み行列`S`が0になって逆行列が特異になる前に、初期推定（2SLSと同じ点推定）の構造残差で判定する。
  第一段階は内生変数ごとの`OlsEstimator::fit_with_optional_f_test`なので、操作変数が内生変数を
  完全に予測する（第一段階の残差が0）と、その内生変数の文脈（`stage=FIRST_STAGE`）で
  `PERFECT_FIT`になる（診断ではなく`fit()`全体を拒否する。2SLSがOLSに一致する退化も含め、
  標準誤差が定義できない回帰を結果に残さない）。第二段階・部分F統計量の制限モデルは判定しない
  （行列版の関数で、`OlsEstimator`を使わない。第二段階は`X̂`への回帰で、判定は構造残差で行う）。Wu-Hausman拡張回帰は
  元の`y`の回帰なので判定し、完全フィットなら3.6節のとおり検定統計量だけ`None`にする。
- **バリデーション**（`engine_pybind::iv::common::build_iv_input`）:
  - `y`/`x_exog`/`x_endog`/`instruments`間の重複列名は`ValidationError`。
  - 各ロール内部（`x_exog`/`x_endog`/`instruments`それぞれ）の重複列名も`ValidationError`。
  - `include_intercept=True`のとき、`x_exog`だけでなく`x_endog`/`instruments`のいずれかに
    `"const"`列が含まれていても`ValidationError`。`x_exog`側でのみ自動追加
    される切片列と、構造方程式本体・`first_stage()`双方の`param_names`が衝突し、
    `dict(zip(param_names, params))`構築時に真の切片係数が後勝ちでサイレントに
    上書きされるため、衝突源が`x_endog`/`instruments`側でも同じ実害が生じる。

### 1.2 `IVOptions`（`#[pyclass]`）

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `estimator` | `str` | `"2sls"` | `"2sls"` / `"gmm"`（大小無視） |
| `cov_type` | `str` | `"classical"` | `"classical"` / `"hc0"`〜`"hc3"` / `"cluster"` / `"hac"`（大小無視）。`estimator="gmm"`でも`gmm_weight_type`とは独立の軸（最終的な報告用SE計算） |
| `include_intercept` | `bool` | `True` | `x_exog`側の設計行列にのみ定数列を自動追加する。`x_endog`/`instruments`には自動追加しない |
| `confidence_level` | `float` | `0.95` | |
| `cluster` | `str \| None` | `None` | `cov_type="cluster"`時（`gmm_weight_type="cluster"`とも共用）のグループキー列名。どちらからも使われないモードで指定すると`ValidationError` |
| `hac_lags` | `int \| None` | `None` | `cov_type="hac"`時（`gmm_weight_type="hac"`とも共用）のラグ数。`None`なら自動計算。どちらからも使われないモードで指定すると`ValidationError` |
| `hac_time` | `str \| None` | `None` | `cov_type="hac"`時（`gmm_weight_type="hac"`とも共用）の時系列順序列（**どちらかがHACなら必須**、未指定は`ValidationError`。値は全行で互いに異なること。同値があれば`ValidationError`、`ols-spec.md`「標準誤差」のHAC参照）。どちらからも使われないモードで指定すると`ValidationError` |
| `gmm_weight_type` | `str \| None` | `None` | GMMの点推定に使う重み行列（`gmm_type="two_step"`/`"iterated"`のみ。`None`は実効既定値`"classical"`）: `"classical"` / `"robust"` / `"cluster"` / `"hac"`（`cov_type`と同じ語彙。旧名`unadjusted`/`kernel`と別名`homoskedastic`/`heteroskedastic`は受け付けない）。`estimator="2sls"`と`gmm_type="one_step"`で指定すると`ValidationError` |
| `gmm_type` | `str \| None` | `None` | GMMの推定方式（`estimator="gmm"`のみ、`None`は実効既定値`"two_step"`。`estimator="2sls"`で指定すると`ValidationError`）: `"one_step"`（1段階、重み`(Z'Z)⁻¹`のみ）/ `"two_step"`（2段階の効率的GMM）/ `"iterated"`（収束まで反復）。大文字小文字は区別しない |
| `gmm_max_iter` | `int \| None` | `None` | `"iterated"`の最大推定回数（初回推定を含む、3以上10000以下）。`None`は実効既定値`100`。`"one_step"`/`"two_step"`/`estimator="2sls"`で指定すると`ValidationError` |
| `gmm_tol` | `float \| None` | `None` | `"iterated"`の収束許容誤差。`None`は実効既定値`1e-6`。`"one_step"`/`"two_step"`/`estimator="2sls"`で指定すると`ValidationError` |
| `raise_on_non_convergence` | `bool \| None` | `None` | `gmm_type="iterated"`で収束しなければ`True`でエラー、`False`で`converged=False`のまま結果を返す。`None`は実効既定値`True`。それ以外の`gmm_type`・`estimator="2sls"`で指定すると`ValidationError` |

- **`cluster`/`hac_lags`/`hac_time`は`cov_type`と`gmm_weight_type`（GMM）で共用する**
  （`IVOptions`に別フィールドを増やさない設計。異なるクラスター変数を使い分けたいニーズが
  出てきたら別フィールド化を検討）。
- **`gmm_weight_type`（GMMの点推定に使う重み行列）と`cov_type`（最終的な報告用SE計算）を
  分離する**（`linearmodels.IVGMM`と同じ構造）。GMMは他のモデルと異なり`cov_type`相当の
  選択（重み行列の仮定する誤差構造）が点推定自体に影響するため、この分離をしないと
  「SEを変えたら係数も変わる」という他モデルには無い挙動を`cov_type`の名の下に隠すことに
  なり紛らわしい。
- **丁度識別（`len(instruments) == len(x_endog)`）では、GMMの点推定は`gmm_weight_type`に
  よらず2SLSと数値的に一致する**（GMMの一般的性質: モーメント条件を正確に0にできるため
  重み行列が点推定に影響しない）。共通GMM推定コアで自然に吸収され、特別な分岐は不要。
- **2SLSはGMMの特殊ケース**（`gmm_weight_type="classical"`、`gmm_type="one_step"`）として点推定は
  数値的に一致するが、実装（`TwoSlsEstimator`/`GmmEstimator`）は意図的に独立させている
  （`TwoSlsEstimator`は`cov_type`対応の推論統計量一式・Sargan・Wu-Hausmanを持つのに対し
  `GmmEstimator`はそれらを持たないため、委譲すると過剰設計になる）。
- 欠損値（null・NaN/無限大）は常にエラー（方針は[`docs/guide/validation.md`](../guide/validation.md)）。

## 2. 結果構造体

`IVResult`（`#[pyclass]`）が公開する項目: `params` / `std_errors` / `test_stats` / `stat_dist` / `stat_df` / `p_values` /
`conf_lower` / `conf_upper` / `param_names` / `residuals` / `dep_var_name` / `n_obs` /
`df_resid` / `df_model` / `converged` / `n_iter` / `cov_type` / `hac_lags_used` / `estimator` / `gmm_weight_type` /
`wald_statistic` / `wald_p_value` / `wald_dist` / `wald_df_num` / `wald_df_denom` /
`r_squared` / `adj_r_squared` / `weak_instrument_f_statistics` / `weak_instrument_f_df_num` /
`weak_instrument_f_df_denom` / `overid_df` / `wu_hausman_df_num` / `wu_hausman_df_denom` /
`overid_statistic` / `overid_p_value` / `wu_hausman_statistic` / `wu_hausman_p_value`。

- **`test_stats`と`stat_dist`/`stat_df`**: 1つの`IVResult`型を`estimator="2sls"`（t分布）・
  `estimator="gmm"`（z分布）の両方が共有するため、統計量は全手法共通の`test_stats`とし、
  分布は`stat_dist`（`"t"`/`"normal"`）と`stat_df`（t分布の自由度。正規分布は`None`）で示す。
  2SLSの`cov_type="cluster"`は`df_resid`ではなく`G-1`を使うため、`stat_df`は実際に使った
  自由度である（`df_resid`とは限らない）。
- **`hac_lags_used`**: 実際に使われたHACラグ数（`hac_lags`明示指定ならその値、未指定なら経験則
  `floor(4*(n/100)^(2/9))`の自動計算値）。`cov_type="hac"`、または`estimator="gmm"`かつ
  `gmm_type`が`two_step`/`iterated`で`gmm_weight_type="hac"`のとき`Some`、それ以外は`None`
  （`hac_lags`は`cov_type`と`gmm_weight_type`で共用のため、両方がHacでも値は同じ。engine層では
  両者のラグが異なる場合`cov_type`側を優先する）。`first_stage()`の各`OLSResult`も同じ`cov_type`で
  推定されるため、HACなら同じ値を持つ。
- **`estimator`**: `IVOptions.estimator`を正規化した小文字文字列（`"2sls"`/`"gmm"`）。
  常に反映される。
- **`gmm_weight_type`**: 型は`Option<String>`。`estimator="gmm"`のときだけ
  `Some(String)`（`IVOptions.gmm_weight_type`を正規化した小文字文字列）、
  `estimator="2sls"`では概念自体が存在しないため常に`None`。
- **`converged`/`n_iter`**: `estimator="2sls"`では常に`converged=true`・`n_iter=1`
  （2SLSは閉形式・非反復のため）。`estimator="gmm"`では実際の反復回数・収束判定結果を返す
  （`gmm_type="iterated"`以外は収束判定を行わないため常に`converged=true`、`n_iter`は1・2）。
- **`gmm_type`**: `estimator="gmm"`では実際に使った推定方式（小文字に正規化）、`estimator="2sls"`では常に`None`。
  `gmm_type="one_step"`では`gmm_weight_type`は使われないため結果の`gmm_weight_type`も`None`。
- **`n_groups`（FE/REの`effects`ごとのグループ数）は含めない**（IVはパネル構造を前提としない）。
- **`log_likelihood`/`aic`/`bic`は除外する**（2SLS/GMMは尤度ベースの推定法ではなく、
  Stataの`ivregress`もデフォルトでは出力しない。正規性を仮定した疑似尤度を計算して
  OLS/FE/REと同じフィールド名で返すと、異なる推定基準の値を同列に比較できるかのように
  誤解させるため統計的な誠実さを優先して含めない）。
- **`r_squared`/`adj_r_squared`はFE/REのような3分割はせず、OLSと同じ単一フィールド**
  （IVはパネルのwithin/between区別を持たない）。
- **全体検定は`wald_statistic`/`wald_p_value`/`wald_dist`/`wald_df_num`/`wald_df_denom`**
  （旧`f_statistic`/`f_p_value`）。2SLSはF型（Wald統計量を傾き係数の数で割ったもの、
  `wald_dist="f"`、`wald_df_denom`は`df_resid`で`cov_type="cluster"`のときだけ`G-1`）、
  GMMは割らないχ²型（`wald_dist="chi2"`、`wald_df_denom=None`）。どちらもWald検定なので
  `wald`という名前が正確で、OLSの`f_statistic`に相当するのは2SLSの`wald_statistic`。
  OLSが`cov_type`がHC系/clusterのときF検定をロバストWald検定に切り替える挙動は2SLSも
  同じ、GMMは常にロバストWald（χ²）（GMMはz分布と決定済みで古典的F検定の正当化が無い
  ため）。
- **検定統計量の自由度**: `overid_df`（Sargan/Hansen J、χ²、`len(instruments) - len(x_endog)`、
  丁度識別は`None`）、`wu_hausman_df_num`/`wu_hausman_df_denom`（F、分子は内生変数の数、
  分母は拡張回帰の`df_inference`）、`weak_instrument_f_df_num`/`weak_instrument_f_df_denom`
  （分子は除外操作変数の数、分母は第一段階回帰の残差自由度。内生変数ごとの辞書だが自由度は
  共通のためスカラー）。統計量が`None`/NaNのときは自由度も`None`。
- **`cov_params`（k×kの分散共分散行列）は`cov_type`に応じた値をPython側に公開する**（OLSと同じ方針、`ols-spec.md`「結果構造体」参照）。`TwoSlsEstimator`/`GmmEstimator`が`fit()`で計算した行列を保持し、`std_errors`はその対角成分の平方根。
- **第一段階回帰結果は`first_stage()`という別メソッド**に切り出す（`fit()`の戻り値本体には
  含めない。非線形モデルの`marginal_effects()`分離方針を踏襲）。`first_stage() -> dict[str,
  OLSResults]`。キーは`x_endog`の変数名、値は既存の`OLSResults`型（新規のIV専用型は
  作らない）。第一段階回帰は`x_endog[i] ~ x_exog + instruments`、`estimator`によらず同じ
  （`engine::iv::common::compute_first_stage`を共有）。
- 弱操作変数診断・過剰識別検定・Wu-Hausman検定はいずれも`fit()`の結果本体に含める
  （別メソッド化しない）。
- `summary()`は実装しない。python_package層（`IVResults`）の`params_table()`は行指向
  `list[dict]`で、キーは`param_name`/`param`/`std_error`/`test_stat`/`p_value`/`conf_lower`/`conf_upper`。

## 3. 内部実装の計算仕様

### 3.1 `cov_type`（標準誤差）

`classical` / `hc0`〜`hc3` / `cluster` / `hac`をサポートする。IVはパネル構造を前提としない
ため、`hac`はOLSの実装（グローバルな時系列順序に対する通常のNewey-West型）をそのまま
踏襲する（Driscoll-Kraay型は不要）。デフォルトは`"classical"`（OLS踏襲。FE/REと異なり
`effects`のような常在するグルーピング列が無いため`"cluster"`をデフォルトにする根拠が無い）。

- **2SLSの分散はサンドイッチ型**: `(X'PzX)^-1 X'Pz Ω Pz X (X'PzX)^-1`（`Ω`の推定方法が
  `cov_type`で変わる）。`X̂`（射影後の予測値）と構造残差`e`をOLSの対応する公式に当てはめた形で、
  `covariance_stage`（`engine/src/iv/two_sls/cov.rs`）が`shared::covariance`の
  `classical_cov_params`・`hc_cov_params`・`hac_cov_params`・`cluster_cov_params`を`X̂`・`e`で
  呼ぶ。スコアに依存する部分（HCのレバレッジ定義、小標本補正の分母）は推定量ごとに決めて共有関数へ
  値として渡し、中立な計算（古典の共分散、外積の和、Newey-WestのBartlettカーネル和、クラスターの
  蓄積と小標本補正、HACラグ数の解決、Wald二次形式）は`shared::covariance`・`shared::wald`に
  一本化する（加算順・`Par::Seq`の指定・対称化といった数値上の判断を1か所に固定するため）。
  Wald二次形式（対称化込み）も共有版（`wald_f_test`・`wald_chi2_test`）を使う。
  OLS固有の調整はここに持ち込まない。GMMのモーメント共分散行列（`gmm/weight.rs`）と`cov_type`用の
  `Ω̂`（`gmm/sandwich.rs`）も同じ共有関数を`Z`で呼ぶ（`iv/CLAUDE.md`「共分散・検定の共有範囲」参照）。
- **GMMのSEサンドイッチは常に一般形**: `Avar(β̂) = B⁻¹(X'ZWΩ̂WZ'X)B⁻¹`
  （`B=X'ZWZ'X`、`W=S_used⁻¹`は点推定に実際に使った重み）。`gmm_weight_type`と`cov_type`は
  独立な選択のため一般に一致せず、「効率的GMM」の特殊ケースでも`B⁻¹`のみへの簡略化分岐は
  しない（実装が単一経路になり単純）。この代数と、Hansen J統計量は、`Ω̂`・自由度を引数に取る
  共有関数（`shared/gmm.rs`）で、固定効果つきのGMMも同じ関数を呼べる形にしてある。
- **HC2/HC3のレバレッジ補正**: `hc2_cov_params`/`hc3_cov_params`は`h_ii = x̂_i'(X'PzX)^-1 x̂_i`
  （`X̂`ベース、実際の設計行列に対して直接計算する）。GMMの`gmm_weight_type`にはHC2/HC3相当の
  区分が無い（成分ごとに異なるレバレッジ補正はGMMの重み行列のスカラー不変性の議論と
  相性が悪いため）。
- **クラスター系の小標本補正は構造方程式のパラメータ数`k`に対して適用する**（操作変数の
  本数`l`（過剰識別なら`l>k`）ではない）。「外積を取る対象の次元（`l`）」と「自由度として
  消費した数（`k`）」を混同しないよう区別する。
- **`cov_type="cluster"`はクラスター数`G`が構造方程式の傾き係数の数`q`（`k - k_constant`）
  より多くなければならない**（`G <= q`は`CommonError::InsufficientClustersForInference`＝
  `ValidationError`。`rank(Ŝ) ≤ G-1`のためロバストWald/F（χ²）検定の`q×q`
  部分行列が構造的に特異。`fit()`冒頭で構造方程式の`q`を使って弾く。判定は
  `engine::iv::common::validate_iv_fit_inputs`（内部の`validate_structural_cluster_count`）に集約し、`TwoSlsEstimator::fit`/
  `GmmEstimator::fit`冒頭と`engine_pybind`の`fit`（`compute_first_stage`より前）が共有する。
  第二段階は行列版の点推定で、共分散の検証は走らない（第一段階の`OlsEstimator::fit`は第一段階の
  `q`で検証し、`FirstStageFailed`にラップされるため区別される）。**構造方程式では
  `G > q`でも、第一段階の傾き係数の数`q_first = k_exog傾き + k_instruments (>= q)`が`G`以下の
  ときは、第一段階の全体F検定（`first_stage()`の`f_statistic`/`f_p_value`/`f_df_num`/
  `f_df_denom`）だけを`None`にして`fit()`は成功する**（第一段階のロバストWald/F検定が構造的に
  特異になるが、診断でしかなく、係数・標準誤差・弱操作変数診断は有効なため。構造方程式の結果は
  全体Wald検定を必須の出力に持つので、`G <= q`は上のとおり拒否したまま）。第一段階の推定は
  `OlsEstimator::fit_with_optional_f_test`（`G <= q`でも`G >= 2`ならF検定だけ定義なしにして続行）
  を使う（`compute_first_stage`）。OLS/WLS/Tobit/Logit/Probit/IVの主結果は横断
  統一。Wu-Hausman拡張回帰は`q_aug = q + k_endog`で`G <= q_aug`になりうるが、実際に使う
  末尾`k_endog`列の部分行列は`rank(Ŝ) ≤ G-1 ≥ k_endog`なら計算可能なので`wu_hausman_*`を
  `None`へdegradeする（3.6節）。
- **GMMの`gmm_weight_type="cluster"`の重み行列`S`（l×l、全操作変数の本数`l`）が`G<l`で
  特異になる問題は別軸**（`cov_type=Cluster`の`G<=q`とは対象・閾値が異なる）。
  過剰識別（`l>k`）は`G<l`、丁度識別（`l==k`）は`G<=l`（`Z'ê=0`により`rank(S)≤G-1`）で
  `IvError::InsufficientClustersForWeightMatrix`＝`ValidationError`。`fit()`冒頭で
  `gmm_type="two_step"`/`"iterated"`のとき検証する（`"one_step"`は`gmm_weight_type`を
  使わないため検証しない）。悪条件の`ComputationError`はbackstopとして残る。

### 3.2 検定分布

**2SLSとGMMで分ける**。

- **2SLS**: t分布（OLS系）。自由度は`df_resid`。
- **GMM**: z分布。2-step efficient GMMはM推定量としての漸近正規性が根拠であり、有限標本の
  t分布としての正当化がない（非線形モデル・MLE系のz分布判断と同じ理由）。
- `linearmodels`の`IV2SLS`/`IVGMM`は`fit(debiased=False)`（既定）で正規分布・
  `fit(debiased=True)`でt分布/F分布を返す（Stataの`ivregress`の`small`オプションと同じ
  発想）。本実装の2SLSは`cov_type`によらず常にt分布/F分布、GMMは常にz分布という設計の
  ため、`linearmodels`のデフォルト（`debiased=False`）とは2SLS側で異なる（ベンチマーク
  照合時は`debiased=True`を明示指定する必要がある）。GMM側は`linearmodels`の既定
  （`debiased=False`→z分布）と一致する。

### 3.3 GMMの推定方式（`gmm_type`/`gmm_max_iter`/`gmm_tol`）

- **用語**: 「1-step GMM」（`gmm_type="one_step"`）は残差に基づく重みの再構築を一切行わず
  アドホックな`W₀=(Z'Z)⁻¹`のみで打ち切る推定（2SLSと同じ結果）。「2-step efficient GMM」
  （`"two_step"`、既定）は「初期推定→残差からS構築→S⁻¹で再推定」の2段階手続き。
  「反復GMM」（`"iterated"`）はこの手続きを収束するまで（または上限まで）繰り返す。
  実装は「直前の残差からSを再構築→再推定」を`while`ループで繰り返すだけで、方式間で
  アルゴリズムを分岐させる必要はない。
- **選択肢を方式名にした理由**: 利用者が選びたいのは推定方式であり回数ではない。回数の整数で
  表すと「2が2段階GMM」という対応を知らないと読めず、許容誤差の有無で意味が切り替わる。
  将来CUEを足す場合も`gmm_type="cue"`を足すだけで済む。「ちょうどn回で打ち切る反復」は
  実務上の需要がないため表現しない。
- **`gmm_max_iter`/`gmm_tol`は`"iterated"`専用**: `"one_step"`/`"two_step"`で指定すると
  `ValidationError`。`IVOptions`は既定値と明示指定を区別できないため既定値は`None`とし、
  `"iterated"`のときだけ実効既定値（`gmm_max_iter=100`、`gmm_tol=1e-6`）に解決する。
  `estimator="2sls"`で指定しても`ValidationError`。
- **選んだモードで使われないオプションは黙って無視せず`ValidationError`にする**:
  `cov_type="cluster"`の書き忘れでclassicalの標準誤差が黙って返るのを防ぐため。
  `gmm_type`/`gmm_weight_type`/`gmm_max_iter`/`gmm_tol`/`raise_on_non_convergence`は
  `estimator="gmm"`のときのみ、`gmm_weight_type`はさらに`gmm_type`が`"two_step"`/
  `"iterated"`のとき、`gmm_max_iter`/`gmm_tol`/`raise_on_non_convergence`は`"iterated"`
  のときのみ有効。`cluster`/`hac_lags`/`hac_time`は`cov_type`と`gmm_weight_type`の
  どちらか一方でも使えば有効。既定値が非`None`だったオプションは、指定有無を区別する
  ため既定値を`None`にして使われるモードでのみ実効既定値に解決する。
- **`gmm_max_iter`は初回推定を含めて数え、3以上10000以下を必須とする**（linearmodelsの`iter_limit`と
  同じ数え方。上限は`max_iter`と共通の`MAX_ITER_LIMIT`で、収束しない問題が実質的に終わらなく
  なるのを防ぐ）。上限2回の反復は「two_stepに収束判定を付けたもの」になり紛らわしいため、
  2段階が欲しい場合は`gmm_type="two_step"`を使う（エラーメッセージでも案内する）。
- **収束判定**: 係数のelementwise・絶対誤差と相対誤差の併用（`tol = max(rtol * |前回値|,
  atol)`、`atol`は内部固定値`1e-8`）。全係数が満たして初めて収束とする。
- **未収束時の挙動**: `raise_on_non_convergence=true`（既定）なら`IvError::
  GmmNonConvergence`（`ComputationError`）、`false`なら`converged=false`のまま結果を返す
  （MLEの`raise_on_non_convergence=false`→`converged=False`と同じ意味論）。
  `raise_on_non_convergence`・`converged`が意味を持つのは`"iterated"`のときだけで、
  `"one_step"`/`"two_step"`は常に`converged=true`、`n_iter`は実際の推定回数（1・2）。
- **engine層の型**: 矛盾した組み合わせを型で表現できないよう、`GmmType::{OneStep,
  TwoStep { weight }, Iterated { weight, max_iter, tol }}`のenumにし、`weight_type`も
  バリアントに含める。`OneStep`は`weight_type`を持たない（検証もしない）。

### 3.4 弱操作変数診断（`weak_instrument_f_statistics`）

- **x_exogを直交化した後の操作変数係数のみを検定する「部分F統計量」として専用計算する**
  （`linearmodels.iv.results.FirstStageResults.diagnostics`と同じ方式）。`first_stage()`が
  返す`OLSResults.f_statistic`（x_exog込みの全回帰係数に対する検定）とは別物。
- 内生変数ごとに計算し`dict[str, float]`で`fit()`の主結果に含める。
- **常に等分散前提、`cov_type`には依存しない**。理由: (1) Stock-Yogoの臨界値
  表自体が等分散前提でキャリブレーションされている（v1では臨界値照合はしないが意味合いは
  引き継ぐ）、(2) `OlsEstimator`が係数の分散共分散行列全体を公開していないため。
  `estimator="2sls"`/`estimator="gmm"`ともに同じ計算方式（`engine::iv::common::
  compute_first_stage`を共有）。
- **v1のスコープ**: 生の部分F統計量のみ返す。Cragg-Donald統計量とStock-Yogo臨界値はエンジンに
  実装済み（3.4.1節）だが、Pythonには未公開（バインディングは別途）。
- `x_exog=[]`かつ`include_intercept=false`（制限モデルに回帰変数が1つも無い）の退化ケース
  はSSRを`y_endog`自体の二乗和として直接計算する特別扱いをしている。

#### 3.4.1 Cragg-Donald統計量とStock-Yogo臨界値（エンジンのみ）

内生変数が複数あるとき、内生変数ごとの部分Fでは見えない、操作変数群全体としての弱さを調べる診断。

- **定義**: 外生変数`X`（定数列を含む。平均を引かず、他の列と同じく偏回帰で落とす）で内生変数`Y`と
  除外操作変数`Z`を直交化し、`Ỹ`を`Z̃`に回帰した当てはめ値`Ŷ`・残差`E`から`A=Ŷ'Ŷ`、`S=E'E`、
  `G=S^{-1/2}'AS^{-1/2}`（`S=LL'`のCholeskyで`L⁻¹AL⁻ᵀ`）を作り、
  `CD = λ_min(G)·(n − k_exog − q − n_absorbed)/q`（`q`は除外操作変数の数）。Stataの`ivreg2`の
  "Cragg-Donald Wald F statistic"と同じ定義で（リポジトリで数値を照合したのはlinearmodelsだけ）、
  内生変数が1個なら部分F統計量に一致する。`G`は半正定値なので、丸めで最小固有値が負になったら0に
  打ち切る。ランクの判定の前に各列をノルム1にそろえるので、列の単位に依存しない。
  `n_absorbed`は固定効果で吸収したパラメータ数（固定効果が無ければ0）。p値は無い（検定ではない）。
- **`cov_type`に依存しない**（等分散前提。3.4節と同じ判断）。ロバスト版はKleibergen-Paap（3.4.2節）と
  Montiel Olea-Pflueger（3.4.3節）。
- **定義できないとき`None`**（`fit()`は失敗させない診断として扱う。理由は区別しない）: 内生変数が
  無い、識別不足（`q < k`）、自由度が1未満、`X`がフルランクでない、`Z`または`Y`の列が`X`の張る空間に
  ほぼ入っている（直交化で元のノルムの`√ε`未満）、`Z̃`がフルランクでない、`S`が数値的に特異
  （内生変数が共線。固有値で確かめる）、いずれかの内生変数の第一段階が完全フィット
  （`SSR <= q·ε·SST`）、入力に非有限値がある。理由を利用者に示す必要が出たら、`None`を理由つきの
  `Err`に変える。正本はここ（`cragg_donald`のdocと`iv/CLAUDE.md`は要点だけ）。
- **臨界値**（`shared/stock_yogo.rs`）: 2SLSの相対バイアス（`b`=0.05/0.10/0.20/0.30。内生変数1〜3個、
  `K2`が`n+2`〜30）と、5%のWald検定のサイズ歪み（`r`=0.10/0.15/0.20/0.25。内生変数1〜2個、
  `K2`が`n`〜30）の2表のみ。表に無い組み合わせは`None`。統計量が臨界値を超えれば5%水準で
  「操作変数は弱い」を棄却する。LIMLとFullerの表は、そのような推定量が無いので持たない。
- **値の出典と検証**: 表は著者のワーキングペーパー（NBER Technical Working Paper 284、2001年8月、
  2002年10月改訂版）のTable 1・Table 2を、PDFから機械的に取り出して転記した。第二の独立な出典
  （`ivreg2`のソース）は取得できなかったため、全セルの独立な照合はしていない。代わりに、欠損セルの
  位置・水準と`K2`と内生変数の数に対する単調性・隣り合う`K2`の差の滑らかさ・広く引用される値との
  一致を単体テストで固定した（`shared/stock_yogo.rs`）。
- **統計量の検証**: linearmodels 7.1の`cragg_donald_f`と、エンジンの単体テストと同じデータで
  相対差1e-15（内生変数1個・2個）で一致する。その値をテストに定数として埋めて回帰ガードにした
  （`weak_instruments.rs`）。単体テストは、閉形式（2×2の固有値）、
  内生変数1個での部分Fとの一致、`n_absorbed`の自由度、Frisch-Waugh-Lovell（個体ダミー入りと
  within変換+`n_absorbed`）、列のスケール不変性（単位が1e-14〜1e9違う列を含む）、操作変数が
  無関係でも非負、`q=1`で除外操作変数が`X`の張る空間にある場合、内生変数の一方だけ完全フィット、
  非有限値、`None`になる各条件を固定する。Pythonからの数値照合はバインディングの後（別途）。

出典:

- Stock, J. H. and Yogo, M. (2005), "Testing for Weak Instruments in Linear IV Regression", in
  Andrews, D. W. K. and Stock, J. H. (eds.), *Identification and Inference for Econometric Models:
  Essays in Honor of Thomas Rothenberg*, Cambridge University Press（臨界値表とg_min統計量の定義）。
  ワーキングペーパー版はNBER Technical Working Paper 284。
- Cragg, J. G. and Donald, S. G. (1993), "Testing Identifiability and Specification in Instrumental
  Variable Models", *Econometric Theory* 9(2), 222-240（ランク統計量）。
- Baum, C. F., Schaffer, M. E. and Stillman, S. (2007), "Enhanced routines for instrumental
  variables/generalized method of moments estimation and testing", *Stata Journal* 7(4), 465-506
  （`ivreg2`の"Cragg-Donald Wald F statistic"との対応）。

#### 3.4.2 Kleibergen-Paap統計量（エンジンのみ）

Cragg-Donaldを、不均一分散・クラスター・自己相関に頑健にした診断。`shared/kleibergen_paap.rs`の
`kleibergen_paap_lm`（rk LM。過小識別の検定）と`kleibergen_paap_f`（rk Wald F）を、
`iv/common/weak_instruments.rs`の`compute_kleibergen_paap(&IvInput, &CovType)`が`IvInput`から呼ぶ。

- **定義**: 外生変数で直交化した`Ỹ`・`Z̃`（列は二乗平均1にそろえる）の`Θ̂ = Q_zz^{-1/2}'Q_zx Q_xx^{-1/2}`
  （`Q`は`Z̃'Z̃/n`等）を特異値分解し、最小特異値に属する右特異ベクトル`B⊥`（長さ`k`）と、
  それ以降の左特異ベクトルの基底`A⊥`（`q−k+1`本）で`λ = A⊥'Θ̂B⊥`を作る。
  統計量は`n·λ'V̂_λ⁻¹λ`で、帰無仮説（`Θ`のランクが`k−1`）の下で`χ²(q−k+1)`に従う（Kleibergen and Paap
  2006）。`V̂_λ`は一次段階のスコア`v_i⊗z̃_i`（`vec(Q_zx)`の標本平均の被積分）を`λ`の方向へ射影した
  `q−k+1`次元のスコア`ψ_i = (v_i'·B⊥'Q_xx^{-1/2}')·(z̃_i'·A⊥'Q_zz^{-1/2}')`の共分散（平均を引かず、
  自由度補正なし）。**射影するとクロネッカー積の`qk×qk`行列を作らずに済み**、meatは共有の
  `hc_meat`・`cluster_meat`・`hac_meat`をそのまま使える。LMのスコアの`v_i`は`Ỹ`、rk Wald Fでは
  `Ỹ`を`Z̃`に回帰した残差。
- **`cov_type`ごとの共分散**: `Classical`（射影後は`ω·I`。LMはAndersonの`n·r²_min`、FはCragg-Donaldに
  一致）、`Hc0`〜`Hc3`（スコア全体に行スケールを掛ける。`Hc1`は`√(n/(n−q−k_exog−n_absorbed))`、
  `Hc2`・`Hc3`は`[X_exog, Z]`のレバレッジ`h_i`による`1/√(1−h_i)`・`1/(1−h_i)`）、`Cluster`、`Hac`（Bartlett核。
  `lags`は共有の`hac_meat`と同じ定義）。`1−h_i ≤ 1e-10`の行があるHC2・HC3は`None`。レバレッジは`[X_exog, Z]`だけから計算し、吸収した効果の分を
  含まないので、`n_absorbed > 0`のHC2・HC3も`None`（個体ダミーを回帰子に入れた定義と食い違うため。固定効果つき
  2SLSで吸収した効果のレバレッジを受け取る口を足すまでの扱い）。
- **rk Wald Fの小標本の補正**（`l = q + k_exog`）: `F = W/q`に、`Classical`・`Hac`は`(n−l−n_absorbed)/n`、
  `Cluster`は`(n−l−n_absorbed)(G−1)/(G(n−1))`を掛ける。`Hc1`は行スケールの中に同じ補正を含むので
  追加の係数が無く、`Hc0`・`Hc2`・`Hc3`は補正なし。これはStataの`ivreg2`・linearmodelsの定義（`robust`は
  HC0のWald統計量`W`に`(n−l)/n`を掛ける。meatに`n/(n−l)`を掛けるHC1と同じ）と同じ値になるよう置いたもので、**linearmodelsの`robust`のFは
  このプロジェクトの`Hc1`に、`robust`のLMは`Hc0`に対応する**。LMには自由度補正が無い
  （`Hc1`のLMだけ行スケールで`√(n/(n−l−n_absorbed))`が入る）。
- **参照が無い範囲**: linearmodelsとivreg2が持つのは`robust`（HC0）・クラスター（一元）・Bartlett HACだけ
  なので、`Hc2`・`Hc3`、`Hc1`のLM、`n_absorbed`つきのロバスト版はこのプロジェクトの拡張で、
  linearmodelsの`_rk_chi2`に行スケールを掛けたスコアを渡す独立な計算（クロネッカー積をそのまま使う形）と
  照合した。DK（Driscoll-Kraay）は未対応で、固定効果つき2SLSで時点ごとに合計したスコアを`hac_meat`に
  渡す分岐を足す（`ScoreCovariance`に1バリアント）。
- **定義できないとき`None`**（LMとWald Fは独立に判定する）: 3.4.1節と同じ条件に加えて、最小の正準相関が1
  （完全フィット）、スコアの共分散が正定値でない（クラスター・観測が少なすぎる。linearmodelsは一般化逆行列と
  自由度の減少で警告つきの値を返すが、ここでは使わない）、クラスターが2未満、クラスターや時間順の長さが行数と
  違う。LMの共分散は`q−k+1`次元でクラスター数が階数の上限、Wald Fのスコアは合計が0なので`q−k+2`個
  以上のクラスターを要する。
- **fixestの`kpr`は参照にしない**: `fitstat(m, "kpr")`はiid、または内生変数と操作変数が同数のときだけ実装
  され、実測ではrk LM・rk Wald Fのどちらにも一致しない（n=240で内生変数1個・操作変数3個のiid:
  `kpr`=0.904に対しLM=35.77・F=13.72。同数でiid: 1.64に対しLM=22.99・F=25.10。同数のheteroは24.51で、
  rk Wald Fの24.82に`(n−l)/n`を掛けた値に近い）。参照はlinearmodelsを使う。
- **検証**: linearmodels 7.1の`kleibergen_paap`・`kleibergen_paap_f`と、unadjusted・robust・clustered・
  kernelで相対差1e-9以内に一致する（n=12の2種類と、n=240の合成データ。値を定数として埋めて回帰ガードにした）。
  `Classical`のWald FはCragg-Donald、LMは`n·λ/(1+λ)`（`λ`はCragg-Donaldの最小固有値）に一致する。単体テストは、
  列のスケール・操作変数の正則な線形結合への不変性、行の並べ替えと`Hac`の時間順、`n_absorbed`が入る場所、
  個体ダミー入り（`n_absorbed=0`）とwithin変換＋`n_absorbed`の一致（Frisch-Waugh-Lovell。等分散・HC0・HC1・
  クラスター・HAC）、`q = k`、`None`になる各条件を固定する。最小の2つの特異値が重なるとき`λ`の方向が一意でなく、
  ロバスト版の値が不安定になりうるが、検出はしていない（既知の限界）。Pythonからの数値照合はバインディングの後（別途）。

出典（3.4.1節に加えて）:

- Kleibergen, F. and Paap, R. (2006), "Generalized reduced rank tests using the singular value
  decomposition", *Journal of Econometrics* 133(1), 97-126（rkランク統計量）。
- Kleibergen, F. and Schaffer, M. E. (2007), "ranktest: Stata module to test the rank of a matrix using
  the Kleibergen-Paap rk statistic", Statistical Software Components S456865, Boston College
  （`ivreg2`が使うrk LMとrk Wald Fの実装）。

#### 3.4.3 Montiel Olea-Pflueger有効F統計量（エンジンのみ）

内生変数が1個のときの、ロバストな弱操作変数の診断。`shared/effective_f.rs`の`effective_f`（行列版）を、
`iv/common/weak_instruments.rs`の`compute_effective_f(&IvInput, &CovType, alpha)`が呼ぶ。

- **定義**（Montiel Olea and Pflueger 2013）: 外生変数（定数列を含む。平均は引かない）で内生変数`Y`と除外操作変数`Z`を
  直交化し、`Z̃'Z̃/n = I`となるように直交正規化する（統計量はこの変換に依らない）。`π̂ = Z̃'Y/n`、`V̂2`を`π̂`の
  共分散の推定値（`Ŵ2 = n·V̂2`）として、
  `F_eff = π̂'π̂/tr(V̂2)`（論文 Eq. (3)）。`Classical`なら`Ŵ2 = ω̂_22·I`でCragg-Donaldの`F`に一致し、操作変数1個なら
  頑健な`t²`（Kleibergen-PaapのWald F）に一致する。
- **`cov_type`と小標本の係数**: Kleibergen-Paap（3.4.2節）と同じ規約（`shared/score_covariance.rs`）。`Ŵ2 = c·meat/n`で、
  `Classical`・`Hac`は`c = n/(n−l−n_absorbed)`、`Hc1`は行スケール`√(n/(n−l−n_absorbed))`に含み、`Hc0`・`Hc2`・`Hc3`は補正なし、
  `Cluster`は`G/(G−1)·(n−1)/(n−l−n_absorbed)`（`l`は除外操作変数と外生変数の数）。`Hc2`・`Hc3`のレバレッジは`[X_exog, Z]`
  だけから計算し、吸収した効果の分を含まないので、`n_absorbed > 0`では`None`。
- **臨界値**（Patnaik近似）: 有効自由度
  `K_eff(x) = tr(Ŵ2)²(1+2x)/[tr(Ŵ2²) + 2x·tr(Ŵ2)·λ_max(Ŵ2)]`（Eq. (4), (33)。`Ŵ2`のスケールに依らず`1 ≤ K_eff ≤ K`）、
  臨界値は`χ²_{K_eff}(x·K_eff)`の上側`α`点を`K_eff`で割ったもの（Eq. (32)）。有効Fがこれを超えれば、水準`α`で
  「Nagarバイアスがベンチマークの`τ`倍を超える」を棄却する。**単純版TSLS**は`x = 1/τ`（`τ`は5/10/20/30%）。
  非心カイ二乗分布の分位点は`statrs`に無いので、ポアソン混合のCDFと二分法で自前に実装した
  （`shared/noncentral_chi_squared.rs`）。ポアソンの重みは最頻値を1とした相対値を漸化式でたどって正規化する
  （`ln Γ`を使わず、非心度が大きくても桁落ちが共通の係数になって消える）。非心度の上限は`ncp/2 ≤ 1e9`（超えると`None`）。
  水準`1 − α`が1に近すぎる（`α`が1e-8より小さい）と、CDFの丸め誤差が水準に対して無視できず、分位点の精度が落ちる。
- **一般化版（非単純版）TSLS**: 被説明変数`y`を渡すと、`y`の誘導形の残差もスコアに加えた`2K×2K`の`Ŵ`（ブロックは`W1`（`y`）・`W12`・
  `W2`（内生変数））から、`x = B_TSLS(Ŵ)/τ`で臨界値を作る（Eq. (5)）。`B_TSLS`は`β`ごとに`S1 = W1 − 2βW12 + β²W2`、
  `S12 = W12 − βW2`、`S2 = W2`として、`sup_{β, C0} |tr(S12) − 2C0'S12C0| / √(tr(S1)·tr(S2))`（Eq. (19), (21)）。
  `C0`についての上限は`S12`の対称部分の最小・最大固有値で解析的に取れ、`β`についての上限は、`β = c + s·sinh u`
  （`u ∈ [−30, 30]`の3000区間の等間隔の格子。`c`の近くは線形、遠くは対数間隔）で局所最大を探し、左隣より丸め誤差を超えて
  大きい点のうち値の大きい上位16個を黄金分割で詰め、`β → ±∞`の極限（`max_λ |1 − 2λ/tr W2|`）も候補にする。
  `(c, s)`は2通り: `(0, β₀)`（`β₀ = √(tr W1/tr W2)`）と、`tr S1(β)`の最小点まわり`(β*, δ)`（`β* = tr W12/tr W2`、
  `δ = √(min tr S1/tr W2)`、`tr S1(β) = min tr S1 + tr W2 (β − β*)²`の形で桁落ちを避ける）。**後者が要る理由**: 被説明変数と
  内生変数の誘導形の誤差がほぼ比例する（`min tr S1/tr W1`が小さい。内生性が強いと起こる）と、`β*`の近くに幅`δ`の鋭い山ができ、
  前者の格子の刻みでは取りこぼして`B_TSLS`を過小評価する（反保守的）。`min tr S1 ≤ 1e-12·tr W1`は未定義として`None`にする。
  格子は`tan`ではなく`sinh`にした（`tan`だと`|β|`の非常に大きいところの山が端の1区間に埋もれた）。1回の計算で`K×K`の固有値分解を
  約6000回＋精緻化に行うので、`K`が数十を超えると秒単位になりうる（公開時に計測して格子の粗さを見直す）。
  同次分散（`Ŵ = Ω⊗I_K`、`classical`）では`B_TSLS = |1 − 2/K|`で、論文の公表値（`τ = 0.10`、`α = 0.05`で`K = 3`が8.53、
  `K = 30`が12.27）を再現する。被説明変数の形が不正・`Ŵ`が不定値・`v1 = βv2`の完全な比例などでは、一般化版だけが`None`
  （統計量と単純版は返す）。
- **未実装**: LIML（このプロジェクトに推定量が無い）、内生変数が2個以上（原論文は1個。拡張を実装した参照実装が無く、
  検証手段が無いので`None`）。
- **`Ŵ2`が特異のとき**: トレースと固有値だけを使うので、半正定値なら値を返す（Kleibergen-Paapは逆行列を使うので`None`）。
  クラスターが操作変数の数より少ないと推論が不安定になる点は同じ。公開時の扱いは別途決める。丸め誤差の範囲の負の固有値は0に
  そろえ、最小固有値が最大固有値の`-1e-10`倍より小さい不定値は`None`。
- **定義できないとき`None`**（診断なので`fit()`は失敗させない）: 内生変数が1個でない、操作変数が無い、`alpha`が(0,1)の外、
  3.4.1節と同じ前処理の条件（`X`がフルランクでない、`Z`が`X`の張る空間に入る、`Z̃`が共線）、一次段階が完全フィット
  （`1 − R² ≤ 1.4e-14`。Kleibergen-Paapと同じ基準で、Cragg-Donaldの`q·ε`に合わせて境界を確かめる）、
  HC2・HC3でレバレッジが1・`n_absorbed > 0`、クラスターが2未満、`Hac`の`lags >= n`・時間順の長さの不一致、
  自由度`n − l − n_absorbed`が1未満、非有限値。
- **検証**: 論文のTable 1（`K_eff = 1..30`、`x = 3.33・5・10・20`の上側5%点）の120セルを、公表値の小数第2位まで
  再現する（`x = 3.33`の列は`10/3`）。非心カイ二乗分布はscipy 1.18と照合した。単体テストは、`Classical`でCragg-Donald、
  操作変数1個でKleibergen-PaapのWald Fとの一致（`Classical`・HC0〜HC3・クラスター・HAC）、固有値(3, 1)の手計算、
  `1 ≤ K_eff ≤ K`、列のスケール・操作変数の線形結合・行の並べ替えへの不変性、個体ダミー入りとwithin変換＋`n_absorbed`の
  一致（Frisch-Waugh-Lovell）、`None`になる各条件を固定する。第三者のソフトウェアを実行した値との照合は、
  ivDiagの`eff_F`（有効Fの統計量のみ。臨界値は返さない）で行う（Pythonへの公開後。オラクル側では実行して比較した）。

出典（3.4.1節に加えて）:

- Montiel Olea, J. L. and Pflueger, C. (2013), "A Robust Test for Weak Instruments", *Journal of Business &
  Economic Statistics* 31(3), 358-369（有効F統計量とPatnaik臨界値）。

### 3.5 過剰識別検定（`overid_statistic`/`overid_p_value`）

Sargan検定（2SLS）／Hansen J検定（GMM）を`fit()`の結果本体に含める（別メソッド化しない）。

- 自由度は`len(instruments) - len(x_endog)`。**丁度識別（自由度0）の場合は`None`**。
- **Sargan検定**（`two_sls/sargan.rs`）は常に古典的（`e'Z(Z'Z)⁻¹Z'e/σ̂²`、`σ̂²=e'e/n`）な計算式を
  使い`cov_type`には依存しない（定義自体が等分散前提の検定であり、不均一分散に頑健な版が
  欲しい場合はGMM＋Hansen Jを使うのが標準的な使い分けのため）。
- **Hansen J検定**（`gmm/hansen_j.rs`）は点推定に使った重み行列`S`（`gmm_weight_type`依存）をそのまま
  流用するのが定義そのもの（`J=(Z'ê)'S⁻¹(Z'ê)`）。`S`は`n`で正規化していない生の和のため
  `n`で割ってはならない（標準形`J=n·ḡₙ'Ŝ⁻¹ḡₙ`に代入すると`n`は完全に相殺する）。
  `one_step`・`gmm_weight_type=Classical`時の`S`は`σ̂²・Z'Z`（`σ̂²`スケーリング必須、
  Classical以外の`Robust`/`Cluster`/`Hac`と絶対スケールを揃えるため）。
  `gmm_weight_type=Classical`かつ`two_step`のHansen Jは2SLSのSargan統計量と数値的に
  一致する。
- どちらも計算失敗は`None`にせず`IvError`として伝播する（使う行列はいずれも点推定計算で
  既に反転成功済みの行列の再利用であり、理論上ここでの特異性は到達不能なため）。

### 3.6 内生性検定（`wu_hausman_statistic`/`wu_hausman_p_value`）

「Wu-Hausman検定（回帰ベース）」は、**第一段階残差を構造式に追加回帰し係数のジョイント
有意性を検定する方式**（`linearmodels.iv.results.IVResults.wooldridge_regression`相当）で
実装する（SSR差に基づく古典公式の`wu_hausman`とは別物）。

- **`fit()`に渡された`cov_type`に対応させる**（弱操作変数診断とは対照的な判断）。
  `linearmodels`の`wooldridge_regression`が「fit時と同じcovarianceでのWald検定」という
  仕様のため。
- `fit()`の結果本体に含める（内生変数全体のジョイント検定のみ、変数ごとのサブセット検定は
  v1スコープ外）。
- **`estimator="gmm"`では常に`None`**（`GmmEstimator`はWu-Hausman検定を実装しない）。
- **想定内の理由で失敗した場合は`fit()`全体を失敗させず`None`にする**（`FirstStageFailed`/
  `SecondStageFailed`の"all-or-nothing"方針とは意図的に異なる）。想定内の理由は2つ:
  (1) 拡張回帰が完全フィット（`y`が第二段階の説明変数と第一段階残差で丸め誤差の範囲で
  厳密に説明される。`sigma^2 = 0`でワルド統計量が定義できない）、第一段階残差が他の列と共線に
  なり設計行列が特異になる（第一段階の残差が0になる退化ケースは第一段階が`PERFECT_FIT`で
  先に止めるので、この経路には来ない）、(2) 拡張回帰は第二段階より内生変数の数だけ
  列が多い（`k_exog+2*k_endog`列）ため、境界的なサンプルサイズでは第二段階は成功するが
  拡張回帰は観測数不足になりうる。**それ以外の失敗（数値的なほぼ特異性等）は`None`へ
  握りつぶさず`IvError::HausmanRegressionFailed`として伝播する**（広すぎる`Err(_)`
  キャッチで実装バグを隠さないため）。
- `x_endog=[]`のときも`None`（同じ意味論に統合）。

### 3.7 `first_stage()`

- 各内生変数`x_endog[i]`について`x_endog[i] ~ x_exog + instruments`を通常のOLSで推定
  （`OlsEstimator → OLSResult`変換は`linear::ols::ols_estimator_to_result`を再利用）。
- `estimator`によらず`engine::iv::common::compute_first_stage`から構築する共通ロジック
  （GMMでも2SLSと同じ診断情報を提供する）。
- **`estimator="2sls"`では第一段階回帰が二重計算になる**（`fit`が明示的に1回、
  `TwoSlsEstimator::fit`が内部でもう1回）。`OlsEstimator`が`Clone`未実装のため
  `TwoSlsEstimator::first_stage_estimators()`の借用結果を`IVResult`へ所有権ごと移せず、
  OLS自体が軽量という前提で許容した設計判断。
- 返す`OLSResults.f_statistic`/`f_p_value`は`x_exog`の寄与を含む通常のOLS F検定であり、
  弱操作変数診断の部分F統計量（3.4節）とは別物。
- **`cov_type="cluster"`でクラスター数`G`が第一段階の傾き係数の数`q_first`以下（構造方程式は
  `G > q_struct`）のときは、その第一段階回帰の`f_statistic`/`f_p_value`/`f_df_num`/`f_df_denom`が
  `None`になる**（2章・3.1節）。標準誤差・t検定・信頼区間は`G-1`自由度で計算され、弱操作変数F
  （3.4節）は古典的なネストF検定で`cov_type`・`G`に依存しないため値が出る。ただしこの
  弱操作変数Fは等分散前提の統計量で、クラスターが少ないときにクラスターロバストな関連性の
  根拠にはならない（`G`の大小によらない既存の限界）。なお`G`が小さいときのクラスターロバスト
  標準誤差は下方バイアスが大きく信頼性が低い（`G-1`自由度のt分布を使うのはその緩和にすぎない）。
  この領域の第一段階は診断として読み、F検定が`None`になる＝標準誤差が十分に信頼できる、
  という意味ではない。

### 3.8 engine_pybind: エラー変換

`engine::iv::common::IvError` → `PyErr`対応（`iv_error_to_pyerr`、`engine_pybind/src/iv/common/
error.rs`）:

| `IvError` | `code`（カタログ） | 例外 |
|---|---|---|
| `Common(CommonError)`（`InvalidHacLags`を含む） | `CommonError`と同じ | `CommonError`と同じ |
| `InsufficientInstruments` | `INSUFFICIENT_INSTRUMENTS` | `ValidationError` |
| `InsufficientObservationsForFirstStage` | `INSUFFICIENT_OBSERVATIONS_FOR_FIRST_STAGE` | `ValidationError` |
| `InsufficientClustersForWeightMatrix` | `GMM_WEIGHT_CLUSTER_COUNT_TOO_SMALL`（`min_clusters`は丁度識別で`l+1`、過剰識別で`l`） | `ValidationError` |
| `InvalidGmmMaxIter` / `InvalidGmmTol` | `GMM_MAX_ITER_OUT_OF_RANGE` / `TOLERANCE_NOT_POSITIVE_FINITE` | `ValidationError` |
| `GmmNonConvergence` | `NOT_CONVERGED`（`iter_option="gmm_max_iter"`。`MleError::NonConvergence`と同じ分類: パラメータの不正ではなく計算過程で発覚した問題） | `ComputationError` |
| `FirstStageFailed` / `SecondStageFailed` / `HausmanRegressionFailed` | 内部の`LeastSquaresError`の`code`に`stage`（`FIRST_STAGE`・`SECOND_STAGE`・`WU_HAUSMAN_REGRESSION`）を付ける（`stage`を付けてよい`code`のみ） | `code`のクラスで決まる |

`IvError`（`engine`）・`PyErr`（`pyo3`）はどちらもこのクレート外定義の型のためorphan ruleに
より`impl From`は書けず、関数として実装し`.map_err(iv_error_to_pyerr)?`で変換する
（`ErrorInfo::report()`の`code`・`params`から例外を作る）。

## 4. テスト

- **Python主リファレンス**: `linearmodels`（`IV2SLS`＝2SLS、`IVGMM`＝GMM）。
- **Rクロスチェック（2SLS）**: `ivreg`。`classical`/`hc0`〜`hc3`/
  `cluster`/`hac`の`vcov`を`coeftest()`経由でそのまま使える。`summary(model,
  diagnostics=TRUE)`は`vcov.`に行列を渡すと常にclassical（iid）vcovにフォールバックする
  仕様のため、`weak_instrument_f_statistics`/`overid_statistic`（設計自体が常にclassical）
  はcov_typeによらず一律クロスチェックできる。`wu_hausman_statistic`/`wu_hausman_p_value`も
  `summary(diagnostics=TRUE, vcov.=<関数>)`でcov_type別のロバスト共分散を診断表に反映できるため、
  全cov_typeでクロスチェックする（`benchmark/iv/references/run_ivreg.R`参照）。ただしclusterの
  p値のみ、`ivreg`のWald検定がF分布の分母自由度にクラスター数を反映しないため一致しない。
  統計量はそのまま比較し、p値は`ivreg`の統計量とF(`n_endog`, `G-1`)から再計算して本実装と比較する
  （`tests/iv/test_iv_crosscheck.py`の`_check_result`の`n_clusters`）。
- **Rクロスチェック（GMM）**: `ivreg`はGMM非対応のため、R `momentfit`（1.0、`gmm`パッケージの
  後継）を使う（`benchmark/iv/references/run_momentfit.R`、フィクスチャ
  `iv_gmm_crosscheck.json`、テスト`tests/iv/test_iv_gmm_crosscheck.py`）。`gmm`パッケージは
  1段階目の重みが単位行列で固定され、linearmodels・本実装の`(Z'Z)⁻¹`と揃えられないため
  採用しない。`iv_gmm.json`（linearmodels）と同じ構成（`NUMERIC_SCENARIOS`×classical重み×
  cov_type、baselineの他の重み・重みと共分散の組み合わせ、過剰識別の棄却シナリオ（`invalid_instrument`）の
  robust重み、複数内生変数（`MULTI_ENDOG_SCENARIOS`、classical以外の重みは`MULTI_ENDOG_OTHER_WEIGHT_TYPES`）、
  Wooldridge `card`）を、
  係数・標準誤差・z値・p値・信頼区間・`n_obs`/`df_resid`・ロバストWald・Hansen Jで照合する。
  **対象外**: hc2/hc3（linearmodels・momentfitとも非対応で、GMMのHC2/HC3は外部参照なし）、
  反復GMM・1段階GMM（フィクスチャがclassical重みのみで、反復しても2SLSと同じ結果になる。
  収束判定も実装ごとに異なる）、弱操作変数F・R²（2SLSのivregクロスチェックと重複、
  momentfitは出さない）。
  - **momentfitを揃えるための設定と、揃えないと一致しない原因**（`run_momentfit.R`のヘッダ
    コメントが実装の正本。実測で確定し、全ケース相対誤差1e-10以下で一致）:

    | 現象 | 原因 | Rスクリプトでの対処 |
    |---|---|---|
    | HAC・クラスター重みで係数が合わない（HACで約1e-2） | momentfit 1.0のバグ。`evalWeights`はHAC・クラスターで`chol(pivot=TRUE)`の因子を保持し、`quadra`が`attr(w, "pivot")`でピボットを取ろうとするが、実際の属性は`w@w`側に付いているためピボットが失われる（qr型のiid・MDSは`w@w$pivot`を使うため影響しない） | 2SLS → `S = vcov(model, θ̂_2SLS)` → `gmmFit(model, weights = solve(S), type = "onestep")`と重み行列を明示的に渡す |
    | 頑健な重みで係数が約1e-6ずれる | 既定の`initW = "ident"`（1段階目が単位行列）、`centeredVcov = TRUE`（モーメントを中心化） | `initW = "tsls"`、`centeredVcov = FALSE` |
    | Hansen Jが約0.2%ずれる | `specTest`の既定は最終推定値の残差で重みを作り直す。linearmodels・本実装は推定に使った重みを使う | `specTest(fit, wObj = <推定時の重み>)` |
    | HAC重みのカーネルが合わない | momentfitの`bw`はBartlettの分母で`bw = ラグ数 + 1`。既定は`prewhite = 1`、QSカーネル、`adjust = TRUE`（Sを`n/(n-q)`倍。係数は不変だがJが変わる） | `kernel = "Bartlett"`、`bw = lag + 1`、`prewhite = FALSE`、`adjust = FALSE` |
    | classical共分散の標準誤差が約0.3%ずれる | momentfitの`iid`は`σ̂² = sd(e)²`（中心化して`n-1`で割る）、本実装・linearmodelsは中心化した残差二乗和を`n-k`で割る（`debiased=True`） | 一般サンドイッチ（`sandwich = TRUE`、meatを`iid`モデルで計算）の標準誤差に`√((n-1)/(n-k))`を掛ける |
    | cluster共分散の標準誤差が約5.6%ずれる | linearmodelsの補正は`G/(G-1)·(n-1)/(n-k)`（Stata流）、momentfitは`G/(G-1)`のみ | `cadjust = TRUE`に加え`√((n-1)/(n-k))`を掛ける |
    | hc1・hac共分散 | 差なし | hc1は`df.adj = TRUE`、hacは`adjust = FALSE`（同じ帯域幅） |

    補正係数はスクリプト内のスカラー手計算で、独立検証として効くのは重み行列・ブレッド・
    ミート行列本体（momentfitの計算）。z値・p値・信頼区間・ロバストWaldは係数・標準誤差・
    共分散行列からの計算（momentfitがネイティブに出す統計量ではない）。
  - 許容誤差は`tests/support/_tolerances_iv.py`の`iv_gmm_crosscheck`（rtol 1e-8、atol 1e-8）。
- **許容誤差**: 相対誤差1e-8＋絶対誤差フロア1e-10（`tests/support/_tolerances_iv.py`の`iv_reference`）を
  基本。`classical`/`hc0`〜`hc1`/`cluster`/`hac`は
  `linearmodels`と(`cov_type`, `debiased`)の対応（`classical`↔(`unadjusted`,
  `debiased=True`)、`hc0`↔(`robust`, `debiased=False`)、`hc1`↔(`robust`,
  `debiased=True`)、`cluster`↔(`clustered`, `debiased=True`)、`hac`↔(`kernel`(bartlett),
  `debiased=False`)）で`params`/`std_errors`が実質機械精度（実測1e-10以下）で一致する。`hc2`/
  `hc3`は`linearmodels`では検証できない（`linearmodels.iv.covariance`にhc2/hc3相当が
  無い）ため、R `ivreg`+`sandwich::vcovHC(type="HC2"/"HC3")`で検証する。`f_p_value`は
  浮動小数点アンダーフローに近い極小値（1e-9〜1e-12オーダー）のケースで相対誤差比較が
  意味を持たないため絶対誤差フローを使う。
- **実データセット**: Wooldridge `card`（Card 1995、大学近接操作変数`nearc2`/`nearc4`に
  よる教育の収益率推定`lwage ~ CARD_X_EXOG + educ`）。`test_iv_reference.py`
  （linearmodels）・`test_iv_crosscheck.py`（ivreg）の両方で全`cov_type`をクロス
  チェックする。GMMも同じ`card`で、`test_iv_gmm_reference.py`（linearmodels、classical重み×
  全cov_type＋robust/hac重み×classical共分散）・`test_iv_gmm_crosscheck.py`（momentfit）の
  両方を照合する（`card`にはクラスター列が無いためcluster重み・cluster共分散は含めない）。
- テストファイル: `tests/iv/test_iv_api.py`（成功パスの構造・API・オプション反映）/
  `test_iv_validation.py`（`ValidationError`/`ComputationError`パス）/
  `test_iv_reference.py`（linearmodels、2SLS）/ `test_iv_gmm_reference.py`（linearmodels、
  GMM）/ `test_iv_crosscheck.py`（R ivreg、2SLS）/ `test_iv_gmm_crosscheck.py`（R momentfit、GMM）。

## 5. 未実装・未対応

- Cragg-Donald統計量とStock-Yogo臨界値のPython公開（エンジンは実装済み、3.4.1節）
- Kleibergen-Paap統計量のPython公開（エンジンは実装済み、3.4.2節）
- Montiel Olea-Pflueger有効FのPython公開（エンジンは実装済み、3.4.3節）、複数内生変数への拡張
- Wu-Hausman検定のGMM対応（`estimator="gmm"`では常に`None`）
- Wu-Hausman検定の変数ごとのサブセット検定（現状は内生変数全体のジョイント検定のみ）
