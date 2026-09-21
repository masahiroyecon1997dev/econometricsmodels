# IV 仕様書

IV（操作変数法: 2SLS/GMM）の確定済み仕様。`engine/src/iv/`（`two_sls.rs`/`gmm.rs`、共通基盤は
`engine/src/iv/common.rs`）・`engine_pybind/src/iv/common.rs`・
`python_package/econometricsmodels/iv/iv.py`として実装済み。`method="2sls"`/`method="gmm"`を
単一の`IV`/`IVResults`ペアで扱う設計のため、本ドキュメントは2SLS/GMM両方を扱う。

## 1. API引数

3層構成: `IV(data, y, x_exog, x_endog, instruments, options).fit() -> IVResults`
（python_package）→ `fit_iv(data, y, x_exog, x_endog, instruments, options) -> IVResult`
（engine_pybind）→ `TwoSlsEstimator::fit` / `GmmEstimator::fit`（engine、`IVOptions.method`で
振り分け）。

### 1.1 `y` / `x_exog` / `x_endog` / `instruments`

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
  `instruments`はいずれも独立に最低1要素を要求し、空リストは`ValidationError`**（Issue #306）。
  `x_endog=[]`は実質OLSと等価な退化ケースであり「そもそもIVを使用すること自体が誤り」と
  判断し、`OLS`への切り替えなしにそのまま`IV`に渡せる利便性よりも誤用防止を優先した。
  `x_endog`/`instruments`は独立に検証するため、「操作変数はあるが対応する内生変数が無い」
  誤用も検出できる。
- **識別の順序条件**（`len(instruments) >= len(x_endog)`）を満たさない場合は`fit()`冒頭
  （第一段階回帰の前）で`IvError::InsufficientInstruments`（`ValidationError`）にする。
- **バリデーション**（`engine_pybind::iv::common::build_iv_input`）:
  - `y`/`x_exog`/`x_endog`/`instruments`間の重複列名は`ValidationError`。
  - 各ロール内部（`x_exog`/`x_endog`/`instruments`それぞれ）の重複列名も`ValidationError`。
  - `include_intercept=True`のとき、`x_exog`だけでなく`x_endog`/`instruments`のいずれかに
    `"const"`列が含まれていても`ValidationError`（Issue #305）。`x_exog`側でのみ自動追加
    される切片列と、構造方程式本体・`first_stage()`双方の`param_names`が衝突し、
    `dict(zip(param_names, params))`構築時に真の切片係数が後勝ちでサイレントに
    上書きされるため、衝突源が`x_endog`/`instruments`側でも同じ実害が生じる。

### 1.2 `IVOptions`（`#[pyclass]`）

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `method` | `str` | `"2sls"` | `"2sls"` / `"gmm"`（大小無視） |
| `cov_type` | `str` | `"classical"` | `"classical"` / `"hc0"`〜`"hc3"` / `"cluster"` / `"hac"`（大小無視）。`method="gmm"`でも`weight_type`とは独立の軸（最終的な報告用SE計算） |
| `include_intercept` | `bool` | `True` | `x_exog`側の設計行列にのみ定数列を自動追加する。`x_endog`/`instruments`には自動追加しない |
| `confidence_level` | `float` | `0.95` | |
| `cluster_col` | `str \| None` | `None` | `cov_type="cluster"`時（`weight_type="cluster"`とも共用）のグループキー列名 |
| `hac_lags` | `int \| None` | `None` | `cov_type="hac"`時（`weight_type="kernel"`とも共用）のラグ数。`None`なら自動計算 |
| `time_col` | `str \| None` | `None` | `cov_type="hac"`時（`weight_type="kernel"`とも共用）の時系列順序列 |
| `weight_type` | `str` | `"unadjusted"` | GMMの点推定に使う重み行列（`method="gmm"`のみ）: `"unadjusted"`（別名`"homoskedastic"`）/ `"robust"`（別名`"heteroskedastic"`）/ `"cluster"` / `"kernel"`。`method="2sls"`では無視 |
| `gmm_iterations` | `int` | `2` | GMM反復回数（`method="gmm"`のみ）: `2`＝efficient two-step、`1`＝1-step、`3`以上＝iterated GMM |
| `gmm_convergence` | `float \| None` | `None` | `Some`のとき`gmm_iterations`を「収束判定の上限反復回数（安全弁）」として扱う（併用方式） |
| `raise_on_non_convergence` | `bool` | `True` | `gmm_convergence`設定時、収束しなければ`True`でエラー、`False`で`converged=False`のまま結果を返す |

- **`cluster_col`/`hac_lags`/`time_col`は`cov_type`と`weight_type`（GMM）で共用する**
  （`IVOptions`に別フィールドを増やさない設計。異なるクラスター変数を使い分けたいニーズが
  出てきたら別フィールド化を検討）。
- **`weight_type`（GMMの点推定に使う重み行列）と`cov_type`（最終的な報告用SE計算）を
  分離する**（`linearmodels.IVGMM`と同じ構造）。GMMは他のモデルと異なり`cov_type`相当の
  選択（重み行列の仮定する誤差構造）が点推定自体に影響するため、この分離をしないと
  「SEを変えたら係数も変わる」という他モデルには無い挙動を`cov_type`の名の下に隠すことに
  なり紛らわしい。
- **丁度識別（`len(instruments) == len(x_endog)`）では、GMMの点推定は`weight_type`に
  よらず2SLSと数値的に一致する**（GMMの一般的性質: モーメント条件を正確に0にできるため
  重み行列が点推定に影響しない）。共通GMM推定コアで自然に吸収され、特別な分岐は不要。
- **2SLSはGMMの特殊ケース**（`weight_type="unadjusted"`、`gmm_iterations=1`）として点推定は
  数値的に一致するが、実装（`TwoSlsEstimator`/`GmmEstimator`）は意図的に独立させている
  （`TwoSlsEstimator`は`cov_type`対応の推論統計量一式・Sargan・Wu-Hausmanを持つのに対し
  `GmmEstimator`はそれらを持たないため、委譲すると過剰設計になる）。
- 欠損値（NaN/無限大）は常にエラー。

## 2. 結果構造体

`IVResult`（`#[pyclass]`）が公開する項目: `params` / `std_errors` / `stats` / `p_values` /
`conf_lower` / `conf_upper` / `param_names` / `residuals` / `dep_var_name` / `n_obs` /
`df_resid` / `df_model` / `converged` / `n_iterations` / `cov_type` / `method` / `weight_type` /
`f_statistic` / `f_p_value` / `r_squared` / `r_squared_adj` / `weak_instrument_f_statistics` /
`overid_statistic` / `overid_p_value` / `wu_hausman_statistic` / `wu_hausman_p_value`。

- **`t_stats`ではなく`stats`という分布非依存の名前**（Issue #159）: 1つの`IVResult`型を
  `method="2sls"`（t分布）・`method="gmm"`（z分布）の両方が共有するため、`OLSResult.t_stats`・
  `LogitResult.z_stats`のような分布固定の名前は使えない。`engine::inference::InferenceStat`が
  同じ理由で`stat`という名前を使っている前例に倣った。
- **`method`**（Issue #307）: `IVOptions.method`を正規化した小文字文字列（`"2sls"`/`"gmm"`）。
  常に反映される。
- **`weight_type`**（Issue #307）: 型は`Option<String>`。`method="gmm"`のときだけ
  `Some(String)`（`IVOptions.weight_type`を正規化した小文字文字列。エイリアス入力
  （`"homoskedastic"`/`"heteroskedastic"`）は正規化されずそのままechoされる）、
  `method="2sls"`では概念自体が存在しないため常に`None`。
- **`converged`/`n_iterations`**: `method="2sls"`では常に`converged=true`・`n_iterations=1`
  （2SLSは閉形式・非反復のため）。`method="gmm"`では実際の反復回数・収束判定結果を返す
  （`gmm_convergence=None`のときは固定回数モードのため常に`converged=true`）。
- **`n_entities`は含めない**（IVはパネル構造を前提としない）。
- **`log_likelihood`/`aic`/`bic`は除外する**（2SLS/GMMは尤度ベースの推定法ではなく、
  Stataの`ivregress`もデフォルトでは出力しない。正規性を仮定した疑似尤度を計算して
  OLS/FE/REと同じフィールド名で返すと、異なる推定基準の値を同列に比較できるかのように
  誤解させるため統計的な誠実さを優先して含めない）。
- **`r_squared`/`r_squared_adj`はFE/REのような3分割はせず、OLSと同じ単一フィールド**
  （IVはパネルのwithin/between区別を持たない）。
- **`f_statistic`/`f_p_value`はGMMでは常にロバストWald検定（χ²）**。OLSが`cov_type`が
  HC系/clusterのときF検定をロバストWald検定に切り替える既存挙動をGMMにも一貫適用する
  （GMMはz分布と決定済みで古典的F検定の正当化が無いため）。2SLSはOLSと同じ切り替え
  ロジック（classical時はF検定、HC/cluster/hac時はロバストWald検定）。
- **`cov_params`（k×kの分散共分散行列）はPython側に公開しない**（OLSと同じ方針）。
- **第一段階回帰結果は`first_stage()`という別メソッド**に切り出す（`fit()`の戻り値本体には
  含めない。非線形モデルの`marginal_effects()`分離方針を踏襲）。`first_stage() -> dict[str,
  OLSResults]`。キーは`x_endog`の変数名、値は既存の`OLSResults`型（新規のIV専用型は
  作らない）。第一段階回帰は`x_endog[i] ~ x_exog + instruments`、`method`によらず同じ
  （`engine::iv::common::compute_first_stage`を共有）。
- 弱操作変数診断・過剰識別検定・Wu-Hausman検定はいずれも`fit()`の結果本体に含める
  （別メソッド化しない）。
- `summary()`は実装しない。python_package層（`IVResults`）の`coef_table()`は行指向
  `list[dict]`で、キーは`param`/`coef`/`std_err`/`stat`（`t_stat`/`z_stat`ではなく`stats`
  プロパティと同じ理由）/`p_value`/`conf_lower`/`conf_upper`。

## 3. 内部実装の計算仕様

### 3.1 `cov_type`（標準誤差）

`classical` / `hc0`〜`hc3` / `cluster` / `hac`をサポートする。IVはパネル構造を前提としない
ため、`hac`はOLSの実装（グローバルな時系列順序に対する通常のNewey-West型）をそのまま
踏襲する（Driscoll-Kraay型は不要）。デフォルトは`"classical"`（OLS踏襲。FE/REと異なり
`entity`のような常在するグルーピング列が無いため`"cluster"`をデフォルトにする根拠が無い）。

- **2SLSの分散はサンドイッチ型**: `(X'PzX)^-1 X'Pz Ω Pz X (X'PzX)^-1`（`Ω`の推定方法が
  `cov_type`で変わる）。`hc_cov_params`/`cluster_cov_params`/`hac_cov_params`
  （`engine/src/iv/two_sls.rs`）は`X̂`（射影後の予測値）ベースのレバレッジ・スコアで
  OLSの対応する実装と同型に計算する。
- **GMMのSEサンドイッチは常に一般形**: `Avar(β̂) = B⁻¹(X'ZWΩ̂WZ'X)B⁻¹`
  （`B=X'ZWZ'X`、`W=S_used⁻¹`は点推定に実際に使った重み）。`weight_type`と`cov_type`は
  独立な選択のため一般に一致せず、「効率的GMM」の特殊ケースでも`B⁻¹`のみへの簡略化分岐は
  しない（実装が単一経路になり単純）。
- **HC2/HC3のレバレッジ補正**: `hc2_cov_params`/`hc3_cov_params`は`h_ii = x̂_i'(X'PzX)^-1 x̂_i`
  （`X̂`ベース、実際の設計行列に対して直接計算する）。GMMの`weight_type`にはHC2/HC3相当の
  区分が無い（成分ごとに異なるレバレッジ補正はGMMの重み行列のスカラー不変性の議論と
  相性が悪いため）。
- **クラスター系の小標本補正は構造方程式のパラメータ数`k`に対して適用する**（操作変数の
  本数`l`（過剰識別なら`l>k`）ではない）。「外積を取る対象の次元（`l`）」と「自由度として
  消費した数（`k`）」を混同しないよう区別する。
- **`cov_type="cluster"`はクラスター数`G`が構造方程式の傾き係数の数`q`（`k - k_constant`）
  より多くなければならない**（`G <= q`は`CommonError::InsufficientClustersForInference`＝
  `ValidationError`、Issue #289。`rank(Ŝ) ≤ G-1`のためロバストWald/F（χ²）検定の`q×q`
  部分行列が構造的に特異。`fit()`冒頭で構造方程式の`q`を使って弾く。第一段階・第二段階
  回帰の`OlsEstimator::fit`内でも同じ検証が走るが、そちらは`FirstStageFailed`/
  `SecondStageFailed`にラップされるため区別される）。OLS/WLS/Tobit/Logit/Probit/IVで横断
  統一。Wu-Hausman拡張回帰は`q_aug = q + k_endog`で`G <= q_aug`になりうるが、実際に使う
  末尾`k_endog`列の部分行列は`rank(Ŝ) ≤ G-1 ≥ k_endog`なら計算可能なので`wu_hausman_*`を
  `None`へdegradeする（3.6節）。
- **GMMの`weight_type="cluster"`の重み行列`S`（l×l、全操作変数の本数`l`）が`G<l`で
  特異になる問題は別軸**（`cov_type=Cluster`の`G<=q`とは対象・閾値が異なる。現状
  `ComputationError`、`ValidationError`への再分類はIssue #290で未着手）。

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

### 3.3 GMM反復（`gmm_iterations`/`gmm_convergence`）

- **用語**: 「1-step GMM」（`gmm_iterations=1`）は残差に基づく重みの再構築を一切行わず
  アドホックな`W₀=(Z'Z)⁻¹`のみで打ち切る推定（`weight_type`によらず常に2SLSと同じ
  結果）。「2-step efficient GMM」（`gmm_iterations=2`、既定）は「初期推定→残差からS構築
  →S⁻¹で再推定」の2段階手続き。`gmm_iterations`はこの反復を`while`ループでN回まで
  繰り返すだけの実装で、1-step/2-step/iterated間でアルゴリズムを分岐させる必要はない。
- **`gmm_convergence`設定時**: `gmm_iterations`は「収束判定の上限反復回数（安全弁）」に
  なる。収束判定は係数のelementwise・絶対誤差と相対誤差の併用（`tol = max(rtol * |前回値|,
  atol)`、`atol`は内部固定値`1e-8`）。全係数が満たして初めて収束とする。
- **未収束時の挙動**: `raise_on_non_convergence=true`（既定）なら`IvError::
  GmmNonConvergence`（`ComputationError`）、`false`なら`converged=false`のまま結果を返す
  （MLEの`raise_on_non_convergence=false`→`converged=False`と同じ意味論）。
- **`gmm_iterations=1`は比較対象となる前回推定値が無いため、`gmm_convergence`の指定有無に
  よらずトリビアルに`converged=true`**。
- **`gmm_iterations=1`でも`weight_type`引数自体の妥当性は常に検証する**（点推定には
  影響しなくても、`Cluster`の`groups`未指定等の設定ミスは黙って成功させない）。

### 3.4 弱操作変数診断（`weak_instrument_f_statistics`）

- **x_exogを直交化した後の操作変数係数のみを検定する「部分F統計量」として専用計算する**
  （`linearmodels.iv.results.FirstStageResults.diagnostics`と同じ方式）。`first_stage()`が
  返す`OLSResults.f_statistic`（x_exog込みの全回帰係数に対する検定）とは別物。
- 内生変数ごとに計算し`dict[str, float]`で`fit()`の主結果に含める。
- **常に等分散前提、`cov_type`には依存しない**（Issue #163）。理由: (1) Stock-Yogoの臨界値
  表自体が等分散前提でキャリブレーションされている（v1では臨界値照合はしないが意味合いは
  引き継ぐ）、(2) `OlsEstimator`が係数の分散共分散行列全体を公開していないため。
  `method="2sls"`/`method="gmm"`ともに同じ計算方式（`engine::iv::common::
  compute_first_stage`を共有）。
- **v1のスコープ**: 生の部分F統計量のみ返す。Stock-Yogo臨界値テーブルとの照合（弱操作変数の
  合否判定）・複数内生変数の同時検定（Cragg-Donald統計量等）はv1スコープ外。
- `x_exog=[]`かつ`include_intercept=false`（制限モデルに回帰変数が1つも無い）の退化ケース
  はSSRを`y_endog`自体の二乗和として直接計算する特別扱いをしている。

### 3.5 過剰識別検定（`overid_statistic`/`overid_p_value`）

Sargan検定（2SLS）／Hansen J検定（GMM）を`fit()`の結果本体に含める（別メソッド化しない）。

- 自由度は`len(instruments) - len(x_endog)`。**丁度識別（自由度0）の場合は`None`**。
- **Sargan検定**（`two_sls.rs`）は常に古典的（`e'Z(Z'Z)⁻¹Z'e/σ̂²`、`σ̂²=e'e/n`）な計算式を
  使い`cov_type`には依存しない（定義自体が等分散前提の検定であり、不均一分散に頑健な版が
  欲しい場合はGMM＋Hansen Jを使うのが標準的な使い分けのため）。
- **Hansen J検定**（`gmm.rs`）は点推定に使った重み行列`S`（`weight_type`依存）をそのまま
  流用するのが定義そのもの（`J=(Z'ê)'S⁻¹(Z'ê)`）。`S`は`n`で正規化していない生の和のため
  `n`で割ってはならない（標準形`J=n·ḡₙ'Ŝ⁻¹ḡₙ`に代入すると`n`は完全に相殺する）。
  `gmm_iterations=1`・`weight_type=Unadjusted`時の`S`は`σ̂²・Z'Z`（`σ̂²`スケーリング必須、
  Unadjusted以外の`Robust`/`Cluster`/`Kernel`と絶対スケールを揃えるため）。
  `weight_type=Unadjusted`かつ`gmm_iterations=2`のHansen Jは2SLSのSargan統計量と数値的に
  一致する。
- どちらも計算失敗は`None`にせず`IvError`として伝播する（使う行列はいずれも点推定計算で
  既に反転成功済みの行列の再利用であり、理論上ここでの特異性は到達不能なため）。

### 3.6 内生性検定（`wu_hausman_statistic`/`wu_hausman_p_value`）

「Wu-Hausman検定（回帰ベース）」は、**第一段階残差を構造式に追加回帰し係数のジョイント
有意性を検定する方式**（`linearmodels.iv.results.IVResults.wooldridge_regression`相当）で
実装する（SSR差に基づく古典公式の`wu_hausman`とは別物）。

- **`fit()`に渡された`cov_type`に対応させる**（弱操作変数診断とは対照的な判断、Issue #164）。
  `linearmodels`の`wooldridge_regression`が「fit時と同じcovarianceでのWald検定」という
  仕様のため。
- `fit()`の結果本体に含める（内生変数全体のジョイント検定のみ、変数ごとのサブセット検定は
  v1スコープ外）。
- **`method="gmm"`では常に`None`**（`GmmEstimator`はWu-Hausman検定を実装しない）。
- **想定内の理由で失敗した場合は`fit()`全体を失敗させず`None`にする**（`FirstStageFailed`/
  `SecondStageFailed`の"all-or-nothing"方針とは意図的に異なる）。想定内の理由は2つ:
  (1) 第一段階残差の分散がゼロ（操作変数が内生変数を完全予測する退化ケース）で拡張回帰の
  設計行列に分散ゼロの列が混入し特異になる、(2) 拡張回帰は第二段階より内生変数の数だけ
  列が多い（`k_exog+2*k_endog`列）ため、境界的なサンプルサイズでは第二段階は成功するが
  拡張回帰は観測数不足になりうる。**それ以外の失敗（数値的なほぼ特異性等）は`None`へ
  握りつぶさず`IvError::HausmanRegressionFailed`として伝播する**（広すぎる`Err(_)`
  キャッチで実装バグを隠さないため）。
- `x_endog=[]`のときも`None`（同じ意味論に統合）。

### 3.7 `first_stage()`

- 各内生変数`x_endog[i]`について`x_endog[i] ~ x_exog + instruments`を通常のOLSで推定
  （`OlsEstimator → OLSResult`変換は`linear::ols::ols_estimator_to_result`を再利用）。
- `method`によらず`engine::iv::common::compute_first_stage`から構築する共通ロジック
  （GMMでも2SLSと同じ診断情報を提供する）。
- **`method="2sls"`では第一段階回帰が二重計算になる**（`fit`が明示的に1回、
  `TwoSlsEstimator::fit`が内部でもう1回）。`OlsEstimator`が`Clone`未実装のため
  `TwoSlsEstimator::first_stage_estimators()`の借用結果を`IVResult`へ所有権ごと移せず、
  OLS自体が軽量という前提で許容した設計判断。
- 返す`OLSResults.f_statistic`/`f_p_value`は`x_exog`の寄与を含む通常のOLS F検定であり、
  弱操作変数診断の部分F統計量（3.4節）とは別物。

### 3.8 engine_pybind: エラー変換

`engine::iv::common::IvError` → `PyErr`対応（`iv_error_to_pyerr`、`engine_pybind/src/iv/
common.rs`）:

| `IvError` | Python例外 |
|---|---|
| `Common(CommonError)` | `common_error_to_pyerr`に委譲 |
| `InsufficientInstruments` / `InvalidHacLags` / `InvalidGmmIterations` / `InvalidGmmConvergence` | `ValidationError` |
| `GmmNonConvergence` | `ComputationError`（`MleError::NonConvergence`と同じ分類: パラメータの不正ではなく計算過程で発覚した問題） |
| `FirstStageFailed` / `SecondStageFailed` / `HausmanRegressionFailed` | 内部の`LeastSquaresError`が`ComputationError`相当かどうかで`ComputationError`/`ValidationError`に分岐（`least_squares_error_is_computation_error`） |

`IvError`（`engine`）・`PyErr`（`pyo3`）はどちらもこのクレート外定義の型のためorphan ruleに
より`impl From`は書けず、関数として実装し`.map_err(iv_error_to_pyerr)?`で変換する。

## 4. テスト

- **Python主リファレンス**: `linearmodels`（`IV2SLS`＝2SLS、`IVGMM`＝GMM）。
- **Rクロスチェック**: `ivreg`（2SLSのみ、GMMは非対応）。`classical`/`hc0`〜`hc3`/
  `cluster`/`hac`の`vcov`を`coeftest()`経由でそのまま使える。`summary(model,
  diagnostics=TRUE)`は`vcov.`に行列を渡すと常にclassical（iid）vcovにフォールバックする
  仕様のため、`weak_instrument_f_statistics`/`overid_statistic`（設計自体が常にclassical）
  はcov_typeによらず一律クロスチェックできるが、`wu_hausman_statistic`はclassical
  cov_typeのときのみ`ivreg`側でクロスチェックする（hc0/hc1/clusterは`linearmodels`側の
  クロスチェックに委ねる）。
- **GMMのRクロスチェックは例外的に省略する**（`ivreg`が対応していないため）。
  「Python主リファレンス＋Rクロスチェック」の2系統検証の例外であることをテスト実装時に
  明記する（RE のハウスマン検定と同型の例外規定）。
- **許容誤差**: 相対誤差1e-8を基本。`classical`/`hc0`〜`hc1`/`cluster`/`hac`は
  `linearmodels`と(`cov_type`, `debiased`)の対応（`classical`↔(`unadjusted`,
  `debiased=True`)、`hc0`↔(`robust`, `debiased=False`)、`hc1`↔(`robust`,
  `debiased=True`)、`cluster`↔(`clustered`, `debiased=True`)、`hac`↔(`kernel`(bartlett),
  `debiased=False`)）で`coef`/`se`が相対誤差1e-10以下（実質機械精度）で一致する。`hc2`/
  `hc3`は`linearmodels`では検証できない（`linearmodels.iv.covariance`にhc2/hc3相当が
  無い）ため、R `ivreg`+`sandwich::vcovHC(type="HC2"/"HC3")`で検証する。`f_p_value`は
  浮動小数点アンダーフローに近い極小値（1e-9〜1e-12オーダー）のケースで相対誤差比較が
  意味を持たないため絶対誤差フローを使う。
- **実データセット**: Wooldridge `card`（Card 1995、大学近接操作変数`nearc2`/`nearc4`に
  よる教育の収益率推定`lwage ~ CARD_X_EXOG + educ`）。`test_iv_reference.py`
  （linearmodels）・`test_iv_crosscheck.py`（ivreg）の両方で全`cov_type`をクロス
  チェックする。GMMは実データセットでのRクロスチェックも対象外。
- テストファイル: `tests/iv/test_iv_api.py`（成功パスの構造・API・オプション反映）/
  `test_iv_validation.py`（`ValidationError`/`ComputationError`パス）/
  `test_iv_reference.py`（linearmodels、2SLS）/ `test_iv_gmm_reference.py`（linearmodels、
  GMM）/ `test_iv_crosscheck.py`（R ivreg、2SLSのみ）。

## 5. 未実装・未対応

- Stock-Yogo臨界値テーブルとの照合（弱操作変数の合否判定）
- 複数内生変数の同時検定（Cragg-Donald統計量等）
- Wu-Hausman検定のGMM対応（`method="gmm"`では常に`None`）
- Wu-Hausman検定の変数ごとのサブセット検定（現状は内生変数全体のジョイント検定のみ）
- GMMの`weight_type="cluster"`が`G<l`で特異になる場合の`ComputationError`→
  `ValidationError`への再分類（Issue #290、未着手）
