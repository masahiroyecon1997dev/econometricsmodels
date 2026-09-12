# Tobit 仕様書

Tobit（打ち切り正規回帰、最尤推定）の確定済み仕様。`engine/src/nonlinear/tobit.rs`（共通基盤は
`engine/src/nonlinear/common.rs`）・`engine_pybind/src/nonlinear/tobit.rs`・
`python_package/econometricsmodels/nonlinear/tobit.py`として実装済み。nonlinear系統共通の設計判断
（ソルバー実行の共通化、`cov_type`共通行列演算、検定分布＝標準正規分布、標準化の基本方針等）は
[`logit-spec.md`](./logit-spec.md)・
[`nonlinear-api-design.md`](../planning/specs/nonlinear-api-design.md)・
[`nonlinear-implementation-notes.md`](../planning/specs/nonlinear-implementation-notes.md)を参照し、
本ドキュメントにはTobit固有の内容のみを記載する。Tobitは連続の潜在変数 `y* = Xβ + ε`
（`ε ~ N(0, σ²)`）を左/右/両側に打ち切った `y` を観測するモデルで、`y` が二値の
Logit/Probitとは尤度・予測量・内部パラメータ化が異なる。

## 1. API引数

3層構成: `Tobit(data, y, x, options).fit() -> TobitResults`（python_package）→
`fit_tobit(data, y, x, options) -> TobitResult`（engine_pybind）→ `TobitEstimator::fit`（engine、
Newton-Raphson/BFGS/L-BFGSによる対数尤度最大化）。

- `y: str`（単一列名、**連続変数**）、`x: list[str]`。`y`の値域検証（`{0.0, 1.0}`）は行わない。
- `TobitOptions`（`#[pyclass]`）は`LogitOptions`の8フィールド（`cov_type` / `include_intercept` /
  `confidence_level` / `cluster_col` / `method` / `max_iter` / `tol` / `raise_on_non_convergence`、
  型・デフォルト値とも[`logit-spec.md`](./logit-spec.md)1章の表と同一）に、打ち切り境界の2フィールドを
  追加する:

  | フィールド | 型 | デフォルト | 説明 |
  |---|---|---|---|
  | `lower` | `float \| None` | `0.0` | 下側打ち切り境界。`None`＝下側は打ち切りなし |
  | `upper` | `float \| None` | `None` | 上側打ち切り境界。`None`＝上側は打ち切りなし |

- デフォルト（`lower=0.0, upper=None`）は標準的な左打ち切り0のTobit。右打ち切りのみにしたい場合は
  `lower=None`を明示的に渡す（Pythonのキーワード引数デフォルトと明示的`None`渡しの区別を利用）。
- **打ち切り境界のバリデーション**（engine層 `TobitInput::from_columns`）:
  - `lower`/`upper`が両方`None`、または両方`Some`で`lower >= upper`は`InvalidCensoringBounds`
    （「境界設定自体が不正」、`ValidationError`）。
  - `y`の実測値が指定境界と矛盾する場合（`lower`指定時に`y < lower`の行、または`upper`指定時に
    `y > upper`の行がある）は`YOutOfCensoringBounds { row, value }`（「境界設定は妥当だがデータと
    矛盾」、`ValidationError`）。`InvalidBinaryY`と同型の「行番号＋値」パターン。
  - 検証は`fit()`冒頭のO(n)スキャン1回（非有限値チェックと同オーダー、Newton反復のO(n·k)に対して
    無視できる）。
- **非識別データの検出**: `y`が`lower`/`upper`いずれの境界にも一致しない観測（厳密に内部の観測）が
  1件も無い場合は`NoUncensoredObservations { lower, upper }`（`ValidationError`）。Tobitのこの退化は
  `β`が発散するのではなく`σ→0`収束として現れ、標準化パラメータノルム基準の`SeparationSuspected`
  では捕捉できないため、事後検知ではなく決定的な入力バリデーションを採用した（打ち切り判定は
  `y==lower`/`y==upper`の完全一致比較、`censoring_fit_check`と同じ規約）。「非打ち切り観測0件」は
  非識別の十分条件であり必要条件ではない（厳密には「打ち切りカテゴリが`x`の線形結合で完全分離
  可能」であること）が、連続`x`で非分離配置が実データに現れることは考えにくいため保守的に単純化
  している。
- `n<=k`（`InsufficientObservations`）・`k==0`（`NoRegressors`）・`"const"`列衝突・欠損値
  （NaN/無限大）・`x`列のnull・`cluster_col`のnull検証はLogitと共通（共有インフラ）。`x`に`"sigma"`
  列があると、`TobitResult`が`param_names`末尾に付ける合成名`"sigma"`と衝突するためエラー
  （`"const"`列衝突と同型、engine_pybind境界の`validate_no_sigma_collision`）。

## 2. 結果構造体

`TobitResult`（`#[pyclass]`）が公開する配列＋名前リスト: `params` / `std_errors` / `z_stats`
（**z検定**） / `p_values` / `conf_lower` / `conf_upper` / `param_names` / `sigma` /
`log_likelihood` / `aic` / `bic` / `wald_statistic` / `wald_p_value` / `n_obs` / `df_model` /
`df_resid` / `converged` / `n_iter` / `cov_type` / `method`（実際に使われたソルバーの小文字文字列、
Issue #307） / `lower` / `upper`。

- **`σ`を含めた`k+1`長への統一**: engine層の`TobitEstimator`は`params()`が`k`長（`β`のみ）だが
  `std_errors()`等は`(k+1)`長（末尾が`σ`）という非対称設計（`cov_params`が`(β,σ)`空間の
  `(k+1)×(k+1)`行列のため）。engine_pybind層でこの非対称性を解消し、`params`/`param_names`/
  `std_errors`/`z_stats`/`p_values`/`conf_lower`/`conf_upper`を全て`(k+1)`長に統一する
  （`param_names`末尾に`"sigma"`、`params`末尾に`sigma()`の値を追加）。利便のため`sigma: f64`
  フィールド（`params[-1]`と同値）も持つ。`param_names`は`["const", <x...>, "sigma"]`。
- **`log_likelihood_null` / `pseudo_r_squared`は提供しない**（Logit/Probitの`llnull`は
  `link(θ̂)=ȳ`の閉形式解だが、Tobitの切片のみモデルはΦ・φを含む非線形方程式で閉形式が存在しない。
  主リファレンスの`AER::tobit`（`summary.tobit`）自体もpseudo R²を実装していない）。
- **`lr_statistic` / `lr_p_value`は`wald_statistic` / `wald_p_value`に置き換える**
  （`AER::tobit`の`summary.tobit`と同じ方式。切片以外の係数が同時にゼロという帰無仮説を
  `cov_params`から直接計算し、`llnull`のための追加最適化が不要）。
- `df_model = k - k_constant`（OLSと同じ定義。Logit/Probitの`k-1`固定とは異なる）。
  `df_resid = n - (k+1)`（`AER::tobit`/`survreg`の`df.residual`と同じく`σ`を含む総パラメータ数を
  差し引く）。`aic`/`bic`も総パラメータ数`k+1`を使う（`σ`は真に推定されたパラメータ）。
- `cov_params`（`(k+1)×(k+1)`）はPython側に公開しないが、`predict()`/`marginal_effects()`/
  `censoring_fit_check()`用に非公開フィールド`estimator: TobitEstimator`として保持する。
- python_package層（`TobitResults`）: `params`/`std_errors`/`z_stats`/`p_values`/`conf_int`は
  係数名→値の`dict`。`coef_table()`は行指向`list[dict]`（`param`/`coef`/`std_err`/`z_stat`/
  `p_value`/`conf_lower`/`conf_upper`）。`summary()`は作らない。

## 3. 内部実装の計算仕様

### 3.1 尤度・スコア・Hessian

`z_i = (y_i - x_i'β)/σ`、`Φ`・`φ`を標準正規分布のCDF・PDFとする。観測`i`の対数尤度は打ち切り
状態で分岐する:

- 非打ち切り（`lower < y_i < upper`）: `log( (1/σ)·φ(z_i) ) = -log σ + log φ(z_i)`
- 左打ち切り（`y_i == lower`）: `log Φ((lower - x_i'β)/σ)`
- 右打ち切り（`y_i == upper`）: `log( 1 - Φ((upper - x_i'β)/σ) )`

左のみ・右のみ・両側いずれの打ち切りでも同一の式で評価できる（境界項は`boundary_terms`が
`lower`/`upper`の`None`を`Φ(∓∞)=0/1`・`φ(∓∞)=0`相当の定数として扱う）。

- **内部最適化変数は`(β, log σ)`の`k+1`次元ベクトル**。`σ`ではなく`log σ`を最適化することで
  正値制約を回避する（`AER::tobit`の`summary.tobit`が`Log(scale)`をそのまま報告するのと同じ流儀。
  Olsen(1978)の`(β/σ, 1/σ)`変換による大域凹性の保証は採用しない）。
- 符号規約はLogit/Probitと同じ（`CostFunction::cost = -ℓ(θ)`、`Gradient`/`Hessian`トレイトも
  同じ符号、`scores()`は符号反転しない生のスコアを返す）。
- Tobitの`(β, log σ)`尤度は**Hessianが不定符号になる領域を持つ**（Logit/Probitの大域凹な尤度とは
  異なる）。そこでは生のNewtonステップが降下方向ですらなくなるため、共有`FaerNewton`
  （`nonlinear/common.rs`）にLevenberg-Marquardt型の減衰ステップ`regularized_newton_step`を追加した:
  `H + λI`で`cost`が減少する候補が見つかるまで`λ`を段階的に増やす。大域凹な問題では`λ=0`の生の
  ステップが常に最初の試行で受理されるため、Logit/Probitの既存の収束挙動と完全に一致する。

### 3.2 最適化・収束判定・病理ケース

- **初期値はOLS推定値**（打ち切りを無視した単純なOLSの`β`とその残差の標本標準偏差、
  `ols_initial_params`）。ゼロベクトル初期値ではNewtonが`SingularHessian`で失敗するケースが
  あったため。`ols_initial_params`のQRベースの階数検定が`method`に関わらず最初に走るため、完全な
  多重共線性は最適化前に`SingularDesignMatrix`（`ComputationError`）で検出される
  （`method`をparametrizeする必要が無い）。
- **`TobitScaling`（`tobit.rs`局所の標準化。共有`standardize_columns`は未使用）**: Logit/Probitが
  使う`common.rs`の`standardize_columns`は`x`列を分散1へスケーリングするだけで`y`をスケーリング
  しないため、`y`のスケールが大きいデータ（例: Wooldridge mroz `hours`、`σ̂≈1122`）で健全なMLE解
  でも標準化係数のノルムが大きくなり、勾配ノルム基準の収束判定や分離ヒューリスティックが不安定に
  なる。対策として`y`のスケーリングと`x`の平均センタリングを行う`TobitScaling`を新設した:
  - `y`・打ち切り境界を`y_scale = 2^round(log2(母集団std(y)))`（**2の冪に丸める**）で一律
    スケーリング。2の冪での除算は倍精度で仮数部不変＝厳密なので、程よいスケールの`y`
    （`std ∈ [1/√2, √2)`）では`y_scale=1`の完全恒等変換になり、反復軌道がスケーリング前と厳密に
    相似に保たれる（生のstdで割ると丸め誤差だけで打ち切り率の高い悪条件データが`SingularHessian`
    に倒れることを実測確認したため丸める）。
  - `x`列は切片ありのとき`(x-x̄)/std`とセンタリング、切片なしのとき`x/std`のみ（吸収先の切片が
    無いと逆変換が壊れるため）。
  - 逆変換: `βⱼ = c·β̃ⱼ/stdⱼ`（`j≥1`）、`β₀ = c·β̃₀ - Σ_{j≥1} x̄ⱼ·βⱼ`、`σ = c·exp(s̃)`。
    `cov_params`は`Cov(β,σ) = J·Cov(β̃,s̃)·Jᵀ`（`J`は上記線形部分の微分に`∂σ/∂s̃=σ`のデルタ法を
    末尾行へ折り込んだもの、`TobitScaling::param_jacobian`）。
- **収束判定`tol`の既定値`1e-6`**はLogit/Probitと同じ結論（通常データでは高精度一致、境界ケース
  のみ`tol`を明示的に締める運用）。
  - **大標本での`tol`スケール問題と副次的な収束判定（Issue #291）**: `terminate`の主判定
    `l2_norm(gradient) < tol`は**総和勾配に対する絶対閾値**で観測数`n`でスケールしない。大 `n`
    （実測で `n≳2·10⁵`、`β`の引き次第）では収束点近傍で勾配の丸め誤差の床が`tol`を上回り、
    コスト関数が浮動小数点の底に達しても主判定が発火しないことがある。この状態で
    `regularized_newton_step`が`MAX_LM_ATTEMPTS`回すべてコスト減少に失敗し、以前は誤って
    `SingularHessian`（`ComputationError`）を返していた（`moderate_censoring, n=10⁶, seed=42` /
    `n=2·10⁵, seed=1`で再現）。現在は`common.rs`共有の`FaerNewton`が、`λ=0`のHessianが可逆
    （＝真に特異ではない）かつ次の3条件——(1) 生Newtonステップを1回進めても勾配ノルムが
    減らない（`≥0.9·前反復`）、(2) 勾配ノルムが収束目標近傍（`<10⁴·tol`）、(3) **コスト関数
    （負の対数尤度）のHessianが正定値**（`llt`成功＝内点最大の2階条件。`(β, logσ)`尤度は
    大域凹でなく鞍点で `NoProgress` が返りうるため必須）——を確認して**収束扱い**にする
    （`FaerNewton::stalled_at_optimum`、`RegularizedStep::NoProgress`。現在点をそのまま返し
    生ステップは適用しない）。3条件が揃わなければ生ステップを適用して反復継続し、
    `max_iter`到達で`NonConvergence`。Logit/Probitの大域凹な尤度ではこの経路（LMラダーの
    全失敗）に入らないため挙動は不変。真の特異性（完全な多重共線性等、`λ=0`のHessianが
    可逆でない）は従来どおり`SingularHessian`。
  - **同じ`tol`のn非スケール性は、大標本での`bfgs`/`lbfgs`の実行時間にも影響する（Issue #285）**:
    `newton`は2次収束のためこの影響をほぼ無償で吸収するが（Tobitでも反復回数は`n`によらずほぼ
    一定）、`bfgs`/`lbfgs`は超1次収束のため同じ絶対勾配閾値を満たすのに`n`が大きいほど反復・
    関数評価が増える。statsmodelsは`n`で正規化してから最適化するためこの影響を受けない。
    実測・小標本での精度検証テストへの影響・運用上の推奨（大標本では`tol`を`n`にほぼ比例させる）
    は[`logit-spec.md`](./logit-spec.md)3.2節参照（既定値・実装は変更していない）。
- **Tobitの「真の」分離は`σ→0`退化として現れる**（Logit/Probitの「係数が±∞へ発散」とは異なる）。
  そのため`run_solver`共有の`SeparationSuspected`（標準化パラメータノルム基準、`y∈{0,1}`で較正）は
  `run_solver`の`separation_norm_check: SeparationNormCheck`引数で**Tobitは`Disabled`**にし、この
  事後チェック自体を通らないようにする（`#286`のスケール由来偽陽性を回避。多変量モデルでノルムが
  `√k`オーダーに増える偽陽性の潜在リスクも排除）。Tobitの分離の現れ方:
  - 非打ち切り観測ゼロ（`σ→0`）→ `fit()`冒頭の`NoUncensoredObservations`（`ValidationError`）で
    先に弾く（1章）。
  - 部分的な準完全分離 → `NonConvergence`（`max_iter`到達、`ComputationError`）。BFGS/L-BFGSも
    `ComputationError`。軽度の準分離＋ごく小さいノイズでは真値自体は回復するが収束判定は満たさず、
    `raise_on_non_convergence=false`なら`converged=false`が返る（「巨大な誤った`β̂`で収束扱いに
    なる」病理ではない）。

### 3.3 標準誤差

`CovType`（Logit/Probit/Tobit共通、`engine::nonlinear::common`）: `Classical` / `Opg` / `Hc0` /
`Hc1` / `Cluster { groups }`。計算式・エラー型（`SingularHessian` / `SingularOpgMatrix` /
`MissingClusterColumn` / `InsufficientClusters` / `InsufficientClustersForInference`＝クラスター数
`G <= 傾き係数の数q`、Issue #289）は[`logit-spec.md`](./logit-spec.md)3.3節と共通
（`observed_information_cov_params`/`opg_cov_params`/`sandwich_cov_params`/`cluster_cov_params`を
共有インフラとして再利用。`H`＝負の対数尤度のHessian`(k+1)×(k+1)`、`scores`＝`n×(k+1)`を渡すだけ）。
Tobit固有の差分:

- **`(β, log σ)` → `(β, σ)` 空間への変換**: 内部最適化は`(β, log σ)`空間で行うため、`cov_params`
  全体（`(k+1)×(k+1)`）にヤコビアン`diag(1,…,1, σ)`（`dσ/d(log σ) = σ`、`β`部分は恒等写像）を
  両側から適用し、`β`-`σ`間の共分散も含めて`(β, σ)`空間へ変換する（対角成分＝`Var(σ)≈σ²·Var(log σ)`
  だけの変換に留めない。限界効果等での`cov_params`再利用を見据えた設計）。`std_errors()`等は
  `β∪{σ}`の`k+1`長ベクトル（`k`番目が`σ`）として公開する。
- **`cov_type="opg"`が特異になるテストケースの構築**: Logit/Probitは`x`の完全な多重共線性でOPG行列を
  特異にできるが、Tobitは`ols_initial_params`のQR階数検定が最適化前に走るため同じ手法が使えない。
  実際に`SingularOpgMatrix`を踏むのは`n`が総パラメータ数`k+1`ぎりぎりの小標本での収束点の数値配置
  に依存するケースで、多重共線性とは無関係（実測ベースで検証）。

### 3.4 適合度統計量

- `log_likelihood`はTobit固有（打ち切り状態で分岐する`Contribution::log_lik`の総和、元のスケールで
  評価）。`aic = -2·ℓ + 2(k+1)`、`bic = -2·ℓ + log(n)·(k+1)`（`σ`を含む総パラメータ数）。
- **モデル全体の有意性検定はWald検定**（`wald_chi2_test`、Tobit専用で`tobit.rs`内に閉じる）。
  OLSの`wald_f_test`と同型の構成（`ensure_well_conditioned_symmetric_matrix`による悪条件検出→
  Cholesky分解→二次形式）だが、`F`分布ではなく**標準正規分布に基づくカイ二乗分布**を使う
  （自由度で正規化する`F = W/df_model`の変換を行わない）。帰無仮説は「切片以外の係数が同時にゼロ」、
  自由度は`df_model = k - k_constant`。`df_model == 0`のときのみスキップ。
- **`wald_statistic`は`cov_type`依存**: `fit()`が計算済みの`cov_params`（要求した`cov_type`の
  ロバスト分散）の傾き部分行列`q×q`をそのまま使う（classicalのときのみ`AER:::summary.tobit`の
  `wald`と一致）。
- **Wald検定とクラスターロバストSEの構造的な相互作用**: `cov_type=Cluster`のとき`Ŝ`は
  `rank(Ŝ) ≤ G - 1`（クラスター寄与スコアの総和がゼロ）のため、`G <= q`だと`q×q`部分行列が
  数学的に常に特異になる。`G`・`q`は入力だけから判定できるため`fit()`冒頭の
  `InsufficientClustersForInference`（`ValidationError`）で弾く（OLS/WLS/Logit/Probit/Tobit/IV
  横断で統一、Issue #289）。`G > q`でも傾き係数間の悪条件で`q×q`部分行列が数値的にほぼ特異になる
  ケースは`wald_chi2_test`内の`ensure_well_conditioned_symmetric_matrix`（`ComputationFailed`）が
  backstop。

### 3.5 限界効果

`marginal_effects(at, target, confidence_level)`は`fit()`とは独立したメソッド。`at`は
`"overall"`（既定、AME）/`"mean"`/`"median"`。`target`は3種:

| `target` | 予測量 | 重み `w`（`dydx_j = w·βⱼ`） |
|---|---|---|
| `expected_latent` | `E[y*\|x] = x'β` | `w = 1`（`s_beta = 0`, `s_sigma = 0`の自明形） |
| `expected_observed`（既定） | `E[y\|x]`（打ち切り考慮の条件付き期待値） | `w = Φ(z_b) - Φ(z_a)`（McDonald-Moffitt 1980） |
| `prob_uncensored` | `P(uncensored\|x) = Φ(z_b) - Φ(z_a)` | `w = (φ(z_a) - φ(z_b))/σ` |

（`z_a = (lower - x'β)/σ`、`z_b = (upper - x'β)/σ`。左のみ・右のみ・両側いずれでも`boundary_terms`
経由で同一の式が正しい値を返す。右打ち切りのみの場合`prob_uncensored`の限界効果は符号が反転する。）

- 3対象とも`dydx_j = w(θ)·βⱼ`という同一の形（Logit/Probitと同じ骨格、`w`の中身のみ異なる）に帰着
  するため、`target_w_and_s`（`w`とその勾配`(s_beta, s_sigma)`を計算）→
  `marginal_effects_from_tobit_w_s`という2段構成で実装する。ただしパラメータ次元が`k`ではなく
  `k+1`（`β∪{σ}`）である点が`common.rs`の`dydx_and_jacobian`と異なるため**Logit/Probitとは共有
  せず独立実装する**（`w`/`s`の計算式そのものが共有できないため。Issue #211の結論）。
- デルタ法のヤコビアン: `∂g_j/∂β_m = βⱼ·s_beta[m] + [j==m]·w`、`∂g_j/∂σ = βⱼ·s_sigma`。分散は
  `jac_j·cov_params·jac_j'`（`(k+1)`次元の二次形式）、標準誤差はその平方根、検定分布は標準正規分布。
  `fit()`時の`cov_params`をそのまま再利用し再最適化しない。
- 定数項は出力から除外する。

### 3.6 predict() / censoring_fit_check()

- `predict(target)`は`marginal_effects`と同じ`MarginalEffectsTarget`を再利用し、`E[y*|x]=x'β`・
  `E[y|x]`（既定）・`P(uncensored|x)`の3種を返す。値の計算は`predicted_value`（`target_w_and_s`と
  同じ`boundary_terms`を再利用、左/右/両側いずれでも単一の式）。学習データの各行のみ対象
  （**in-sample限定**、out-of-sample対応は4章）。
- **`pred_table()`は廃止し`censoring_fit_check()`に置き換える**。単一集約値ではなく`lower` /
  `uncensored` / `upper`の**方向別内訳**（`CensoringFitCheck`、該当方向の打ち切りが無ければその
  カテゴリは出力されない）。各カテゴリは`observed_rate`（`y`がちょうど境界値に一致する観測の割合）と
  `model_implied_rate`（各観測の該当カテゴリ確率の平均、`Φ`の組み合わせ）を持つ。理由: 両側打ち切りで
  どちらの境界に不整合があるか区別できるようにするため。
- **打ち切り判定は`yᵢ==lower`/`yᵢ==upper`の浮動小数点完全一致比較**（`TobitInput::from_columns`が
  `y`を変換せず保持する設計、およびTobitの定義自体と整合）。呼び出し側が渡す`y`と`lower`/`upper`が
  ビット単位で一致する前提（CSV/Parquet経由や`Float32`→`Float64`変換で丸め誤差が生じる経路がある
  場合は要注意、`censoring_fit_check`のdocコメント参照）。許容誤差付き比較は未導入。

### 3.7 engine_pybind: エラー変換

`MleError` → `PyErr`は[`logit-spec.md`](./logit-spec.md)3.7節のマッピング表と同一
（`mle_error_to_pyerr`はLogit/Probit/Tobit共通）。Tobit固有バリアントの追加:

| `MleError` | Python例外 |
|---|---|
| `Common(NoUncensoredObservations)` / `InvalidCensoringBounds` / `YOutOfCensoringBounds` | `ValidationError` |

- `predict()`/`marginal_effects()`の`target`引数のPython文字列は`"expected_latent"` /
  `"expected_observed"` / `"prob_uncensored"`（Rust enum名のsnake_case版。`at`のような単語1つの
  慣習が無いためenum名との対応を優先）。パース関数`parse_marginal_effects_target`はTobit専用
  （`tobit.rs`）。

### 3.8 テスト

- **主リファレンス**: R `AER::tobit`（`survival::survreg`エンジン）。`survreg`は内部で`(β, log σ)`を
  独自のNewton-Raphsonで最適化するが、本実装との一致は実測で係数 ~3e-9・標準誤差 ~1e-9・対数尤度
  ~1e-12 と`RTOL=1e-8`を満たす（Logit/Probitのstatsmodels比較を踏襲）。
- **交差検証**: R `censReg`（`maxLik`エンジン）。`survreg`と`maxLik`は最適化実装が完全に独立。
  `censReg`側の`maxLik`収束を`reltol=1e-14 / gradtol=1e-10`まで詰めた上で、合成シナリオは点推定・
  SE・限界効果とも ~2e-9 で一致し`RTOL=1e-8`。実測乖離に基づき個別に緩めた項目
  （`tests/_tolerances.py`）:
  - `high_condition_number`（x1,x2 相関 0.999）の hc0/hc1: SE・z・信頼区間・限界効果SEが ~1.9e-8
    まで増幅するため`5e-8`。
  - `mroz`（`hours`生スケール）: `censReg`の`maxLik`が`survreg`ほど収束が詰まらず、SE系・Wald・
    信頼区間・限界効果SEが ~1e-7〜3e-5 乖離するため`1e-4`（点推定・σ・対数尤度・限界効果dydx・
    予測値・打ち切り適合度は ~3e-9 で一致。engineと`survreg`は同データで ~3e-10 一致するため
    `censReg`側の収束限界であって本実装の問題ではない）。
  - `method="bfgs"/"lbfgs"`: `newton`と異なる最適化経路でリファレンス（method非依存）から僅かに
    ずれた点に収束するため全フィールド`1e-7`（予測値で最大 ~2.2e-8）。
- **手計算箇所のformula非依存検証**（主・交差ともR実装で第三者三角測量が効かないため、
  `.claude/rules/testing-policy.md`「リファレンス実装」2.）: `run_tobit_crosscheck.R`内で
  `stopifnot`により以下を検証する — スコア（`estfun`）↔ Tobit対数尤度の`numDeriv::grad`、
  McDonald-Moffitt閉形式（重み`w = dE[y|x]/dμ`・デルタ法ヤコビアン）↔ `numDeriv::grad`および
  `boundary_terms`を使わない独立実装、AIC/BIC ↔ R標準の`AIC()`/`BIC()`ジェネリック、classicalの
  全体Wald ↔ `AER:::summary.tobit(fit)$wald`。ロバスト共分散の meat（`sandwich::estfun`）と bread
  （`sandwich::bread`）は`sandwich`パッケージ由来で本実装から独立だが、`(β,logσ)→(β,σ)`ヤコビアン
  変換と hc1 小標本補正は本スクリプトの手書きで、`clubSandwich`等との三角測量は行っていない
  （`estfun`自体は`numDeriv`検証済みのため共有の手書き部分は変換のみに限定される）。
- **合成データセット**（`benchmark/nonlinear/datasets.py`、
  `generate_censored_regression_dataset`）: 打ち切り比率違い（light/moderate/heavy ≈ 左打ち切り
  15/35/60%）、`right_censoring`、`interval_censoring`、誤差構造（`high_variance`＝`ε~N(0,10²)`、
  `heteroskedastic`＝`σ_i = exp(0.5·x1)`の乗法的不均一分散で Tobit MLE の等分散仮定に対する誤設定
  → ロバスト共分散 opg/hc0/hc1 と classical が乖離、点推定は擬似真値に一致）、悪条件
  （`small_n`、`moderate_multicollinearity`、`high_condition_number`、`scale_variance_mild`）。
  `scale_variance`（比 1e6）・`perfect_multicollinearity`は`ComputationError`パス専用
  （`scale_variance`は Tobit の全体 Wald 検定が傾き部分行列の反転を要求し全 cov_type で特異になる
  ため成功パスにできず、`scale_variance_mild`（比 1e3）を数値リグレッション検知用の成功パスとする。
  OLS の precedent に従う）。
- **実データ**: Wooldridge `mroz`の`hours`（生スケール、左打ち切り約43%、Example 17.2）。非クラスターの
  4 cov_type で数値照合する。`TOBIT_MROZ_FORMULA`は RHS 7変数で`G=2 <= q=7`のためクラスターケース
  （`city`列）は`InsufficientClustersForInference`（`ValidationError`）になり成功パスを持たない。
- **フィクスチャ**（`tobit.json` / `tobit_crosscheck.json`）は`run_tobit_crosscheck.R`の`engine`
  引数（`survreg` / `censReg`）違いで構造が完全に同一のため、pytest本体は`_tobit_checks.py`に
  集約している。

## 4. 未実装・未対応

- `predict()`/`censoring_fit_check()`のout-of-sample対応（`new_data`引数、Logit/Probitと同じ理由で
  別issueトラッキング）。
- `start_params`（ユーザー指定初期値）。
- **尤度比検定（LR statistic/p-value）**: v1では`llnull`のためのintercept-only再最適化を避けて
  Wald検定を採用した。実装コストは`TobitInput`を`k=1`（切片のみ）で構築し既存のNewton/BFGS/L-BFGS
  基盤にそのまま渡せるため軽微で、計量経済学の実務で好まれる場面もあるため将来拡張候補。
- `dist`（誤差分布）は**Gaussian固定**。`survival::survreg`が選べるlogistic/extreme value等は対象外。
- `censoring_fit_check`の許容誤差付き打ち切り判定（現状は浮動小数点完全一致）。実データで丸め誤差に
  よる誤分類が問題になった時点で検討する。
- **`SeparationNormCheck`を無効化したことによる「有限だが統計的に無意味なほど巨大な`β̂`」の検知**:
  標準化ノルムとは別の指標（Hessianの条件数、SEの発散、`σ̂/σ_y`比の下限等）は未着手。実データで
  問題が顕在化した時点で再検討する。
- **`wald_chi2_test`とOLSの`wald_f_test`の重複**: 部分行列抽出→悪条件検出→Cholesky→二次形式の
  構成が同型で、異なるのは検定分布（カイ二乗 vs F）のみ。現時点ではTobit1箇所のみの利用のため
  共通化は見送り。IV等で3箇所目の重複が生まれる場合に二次形式計算のコア部分の共通化を検討する。
