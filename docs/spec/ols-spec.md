# OLS 仕様書

OLS（最小二乗法）の確定済み仕様。`engine/src/linear/ols/`・`engine/src/shared/`（共分散・Wald検定・最小二乗・適合度の部品）・`engine_pybind/src/linear/ols.rs`・
`python_package/econometricsmodels/linear/ols.py`として実装済み。パフォーマンス比較の詳細は
[`../performance/ols.md`](../performance/ols.md)、CI/CD・セキュリティはmethod非依存のため
[`ci-cd-notes.md`](./ci-cd-notes.md)を参照。

## 1. API引数

3層構成: `OLS(data, y, x, options).fit() -> OLSResults`（python_package）→
`fit_ols(data, y, x, options) -> OLSResult`（engine_pybind、PyO3境界）→
正規方程式ソルバー・標準誤差計算（engine）。

- `y: str`（単一列名）、`x: list[str]`（複数列名）。`y`を`list[str]`にしない理由:
  Phase1〜6（VAR等の一部時系列手法を除く）でyは常に1変数であり、型で単一性を保証する。
- `OLSOptions`（`#[pyclass]`、python_packageは再輸出のみ）:

  | フィールド | 型 | デフォルト | 説明 |
  |---|---|---|---|
  | `cov_type` | `str` | `"classical"` | `"classical"` / `"hc0"`〜`"hc3"` / `"cluster"` / `"hac"`（大小無視） |
  | `include_intercept` | `bool` | `True` | `True`なら設計行列の先頭に定数列を自動追加する |
  | `confidence_level` | `float` | `0.95` | 信頼区間の信頼水準、`(0, 1)` |
  | `cluster` | `str \| None` | `None` | `cov_type="cluster"`時のグループキー列名（`data`内の列）。他の`cov_type`で指定すると`ValidationError` |
  | `hac_lags` | `int \| None` | `None` | `cov_type="hac"`時のラグ数。`None`なら`L=floor(4*(n/100)^(2/9))`で自動計算。他の`cov_type`で指定すると`ValidationError` |
  | `hac_time` | `str \| None` | `None` | `cov_type="hac"`時の時系列順序列（**必須**。未指定は`ValidationError`）。列の値は全行で互いに異なること（同値が1組でもあれば`ValidationError`）。他の`cov_type`で指定すると`ValidationError` |

- `include_intercept=True`のとき`x`に`"const"`列があるとエラー（自動追加する定数項と衝突）。
  `x`に自前の定数列を含める重複検出は行わず、生じる多重共線性は`SingularMatrix`に委ねる。
- 欠損値（null・NaN/無限大）は常にエラー。listwise deletionはしない。理由と全手法共通のエラー条件は公開ページ[`docs/guide/validation.md`](../guide/validation.md)を参照。
- 検定分布は**t分布**（正規分布ではない）。`cov_type`がHC系/clusterでもF検定はロバストWald検定に切り替える。
- `confidence_level`は`fit()`時に一度だけ使用し、結果に固定して含める（再計算用の可変引数は提供しない）。

## 2. 結果構造体

`OLSResult`（`#[pyclass]`、`skip_from_py_object`）が公開する配列＋名前リスト:
`params` / `std_errors` / `test_stats` / `p_values` / `conf_lower` / `conf_upper` / `param_names` /
`residuals` / `dep_var_name` / `n_obs` / `cov_type`（実際に使われた種別の小文字文字列） / `hac_lags_used` /
`r_squared` / `adj_r_squared` / `f_statistic` / `f_p_value` / `f_df_num` / `f_df_denom` /
`df_resid` / `df_model` / `log_likelihood` / `aic` / `bic`。

- `df_resid = n - k`、`df_model`は定数項を除く傾き係数の数。`f_df_num = df_model`、`f_df_denom`は
  検定に使った自由度（`df_resid`、`cov_type="cluster"`のときだけ`G-1`）。傾き係数が無く
  `f_statistic`がNaNのときは`f_df_num`/`f_df_denom`も`None`。

- `hac_lags_used`: `cov_type="hac"`のとき実際に使われたラグ数`L`（`hac_lags`明示指定ならその値、
  未指定なら経験則で自動計算した値。engineの`resolve_hac_lags`の戻り値をそのまま保持する）。
  `hac`以外は`None`。入力`hac_lags`（`None`のまま）とは別フィールドで、ユーザー指定値は書き換えない。
- `conf_int`は`conf_lower`/`conf_upper`の2配列に分割（engine内部表現・pyo3実装の簡潔さを優先）。
- `k×kの分散共分散行列（cov_params）はPython側に公開しない`。`OlsEstimator`自体は非公開
  フィールドとして保持する（クレート内の他系統からの部分Wald検定の再利用のため、
  `engine/src/linear/CLAUDE.md`参照。IVのWu-Hausman検定用に追加するまでは
  `fit()`内のローカル変数として使い切っていた）が、`engine_pybind`側に公開する`OLSResult`
  には引き続き含めない。
- `summary()`（テキスト整形）・DataFrame版の`coef_table()`/`conf_int()`は作らない
  （プログラムから呼び出して使う設計方針上、テキスト表示・対話的操作を前提に
  しないため）。
- python_package層（`OLSResults`）:
  - `params`/`std_errors`/`test_stats`/`p_values`/`conf_int`: 係数名→値の`dict`（O(1)取り出し用）。
  - `coef_table()`: 行指向`list[dict]`（REST APIレスポンスにそのまま使える形）。
  - `residuals`: `list[float]`をそのまま素通し。

## 3. 内部実装の計算仕様

### 3.1 設計行列・係数計算

- 係数は列ピボットQR分解（`col_piv_qr().solve_lstsq()`）で求める。`X'Xβ=X'y`をCholeskyで解く方式は
  不採用（`X'X`の明示計算で条件数が2乗になり不利な上、QRなら特異性検出と計算を同時に行える）。
- 特異性判定は相対閾値: `col_piv_qr`の`R`対角成分のうち`threshold = k * f64::EPSILON * |R[0,0]|`
  未満のものがあればランク落ち（絶対閾値は不採用、データスケール依存を避けるため）。この比較は
  `diag.is_nan() || diag <= threshold`という形で、NaNも明示的に検出する（`include_intercept=false`
  かつ全説明変数列がゼロという設計行列全体が完全にゼロのケースで、`col_piv_qr`が列選択時の0除算に
  よりR対角成分にNaNを生成しうるため。単純な`<=`比較だとNaNとの比較が常にfalseになりすり抜ける）。
- 標準誤差計算用の`(X'X)⁻¹`（`shared::covariance::xtx_inverse`）は`X'X`自体のCholesky分解で求める（QR分解の`R`因子から
  導出する案は実測で高速化しないことを確認済み）。

### 3.2 標準誤差

**classical**（デフォルト）: `σ̂²(X'X)⁻¹`の対角成分の平方根、`σ̂² = SSR/(n-k)`。t統計量・p値は
自由度`n-k`のt分布（**statsmodelsは`cov_type`がclassical以外だと既定で正規分布(`use_t=False`)
を使うが、本プロジェクトは全`cov_type`でt分布に統一する**。ベンチマーク照合時は`use_t=True`を
明示指定する必要がある）。

**HC0〜HC3**: $\widehat{\mathrm{Var}}_{HC}(\hat\beta) = (X^\top X)^{-1} \hat\Psi (X^\top X)^{-1}$

| タイプ | $\hat\Psi$ |
|---|---|
| HC0 | $\sum_i \hat\varepsilon_i^2\, x_i x_i^\top$ |
| HC1 | $\frac{n}{n-k}\cdot$ HC0 |
| HC2 | $\sum_i \frac{\hat\varepsilon_i^2}{1-h_{ii}}\, x_i x_i^\top$ |
| HC3 | $\sum_i \frac{\hat\varepsilon_i^2}{(1-h_{ii})^2}\, x_i x_i^\top$ |

$h_{ii} = x_i^\top (X^\top X)^{-1} x_i$（レバレッジ、HC2/HC3のみ必要）。HC2/HC3は`h_ii`が1に極めて
近い退化した設計だと発散しうるが、これはHC2/HC3自体の数学的性質でありengine固有のバグではない。

**HAC（Newey-West、Bartlettカーネル）**:
$$
\widehat{\mathrm{Var}}_{HAC}(\hat\beta) = (X^\top X)^{-1}\, \hat S \,(X^\top X)^{-1}, \quad
\hat S = \hat S_0 + \sum_{l=1}^{L} w_l (\hat S_l + \hat S_l^\top), \quad w_l = 1 - \frac{l}{L+1}
$$
- ラグ数`L`: `hac_lags`指定時はその値（`0 <= L < n`を検証）、未指定時は経験則
  `L = floor(4*(n/100)^(2/9))`で自動計算（EViews等でも使われるデータ非依存の式。完全な
  データ依存の自動バンド幅選択は主リファレンスのstatsmodelsに同等機能がなく未実装）。
- `hac_time`は必須（未指定は`ValidationError`）。行順を時系列順とみなす暗黙の既定は置かない:
  データが時系列順に並んでいなくてもエラーにならず、時系列順のHACに見える誤った結果が黙って
  返るため、時間順は常に列で明示させる（行順がそのまま時間順なら、`df.with_row_index("t")`
  等で行番号の列を足して渡す。statsmodelsやRの`sandwich`は行順を使い時点列を取らない、
  意図的な差）。`engine`の`CovType::Hac.time_order`も`Option`ではなく必須の`Vec<f64>`で、
  行順を既定とする経路はengineにも無い。昇順ソートしたインデックスで
  ラグ付き自己共分散を計算する（`OlsInput`自体は並べ替えない。Python側に返す残差配列と
  元DataFrameの行対応を保つため）。
- `hac_time`の値は全行で互いに異なることを要求する。同値があると順序が定まらず、engineの
  ソートは同値の行を行順で黙って並べてしまい、時系列順のHACに見える結果を返すため、
  `engine_pybind`が`ValidationError`にする（全値同一も同じ）。比較は元のdtypeの値で行い
  （整数・`Date`・`Datetime`・`Decimal`は物理表現の`i128`、浮動小数は値。`Boolean`は値が2種類しかなく
  順序が定まらないため、dtypeの時点で拒否する）、engineには
  値ではなく昇順の位置（順位）を渡す。f64へ変換すると2^53超の整数やナノ秒の`Datetime`が
  同値に潰れて、この検査をすり抜けるため。
- **パフォーマンス上の罠**: `k×k`という小さい出力サイズの行列積で、faer既定の並列実行は
  ディスパッチオーバーヘッドが計算本体を上回り逐次より遅くなる（実測n=10,000,k=2で6倍悪化）。
  `shared::covariance::hac_meat`内でのみ`Par::Seq`を明示指定して回避している。他手法で同様の小さい行列の
  頻繁な積を書く場合も並列化の要否を実測してから決めること。
- statsmodelsとの照合は`cov_kwds={"maxlags": L}, use_t=True`。`use_correction`（小標本補正）は
  既定の`False`のままで一致することを確認済み。

**クラスター**: $\hat S = \sum_g S_g S_g^\top$（$S_g = \sum_{i\in g}\hat\varepsilon_i x_i$）。
- グループ化は**キーの辞書順に整数コードを振る`GroupCodes`（`BTreeMap`と同じ反復順）を使う
  （`HashMap`は禁止）**: `HashMap`はプロセスごとのハッシュシードで
  反復順序が変わり、浮動小数点加算の非結合性により`fit()`を複数回呼ぶと標準誤差が1 ULP程度ぶれる
  非決定性バグを起こす（`fit_cluster_std_errors_are_deterministic_across_repeated_fits`で固定）。
  クラスター系の実装を今後増やす場合も同じ罠がある。
- 小標本補正`G/(G-1) * (n-1)/(n-k)`は常に適用し、無効化オプションは設けない（statsmodels
  `cov_cluster`の既定`use_correction=True`と一致）。
- t検定・信頼区間・F検定の自由度は`cov_type="cluster"`のときのみ`n-k`ではなく**`G-1`**に切り替える
  （statsmodelsの既定`df_correction=True`、計量経済学の標準的慣行）。`df_resid`自体（σ̂²・調整済み
  R²・AIC/BIC）は常に`n-k`のまま。
- **`G ≤ q`の境界（`ValidationError`）**: $\hat S = \sum_g S_g S_g'$は、クラスター
  寄与スコアの総和がゼロ（正規方程式$X'e = 0$）になるため`rank(Ŝ) ≤ G - 1`。F検定が使う`q×q`
  （`q = k - k_constant` = 傾き係数の数）部分行列は`G ≤ q`のとき構造的に特異になる（`G = q`
  ちょうども数学的には常に特異。`rank(Ŝ) ≤ G`という緩い上限で考えると`G = q`は「境界」に見えるが、
  実際の上限は`G - 1`）。`G`（クラスター列のユニーク数）も`q`（説明変数の列数）も入力だけから
  判定できるため、行列計算を待たず`fit()`冒頭で`InsufficientClustersForInference`
  （`ValidationError`）として弾く。「クラスタ数境界の成功パス」のテストは`G > q`（厳密不等号）を
  保つ必要がある。OLS/WLS/Tobit/Logit/Probit/IV(2SLS,GMM)横断で統一。
  - `G > q`でも傾き係数間の悪条件（極端なスケール差・準多重共線性等）で`q×q`部分行列が数値的に
    ほぼ特異になるケースは事前判定できないため、`ensure_well_conditioned_symmetric_matrix`による
    `ComputationError`（次節）がbackstopとして残る。
- `G < 2`は`InsufficientClusters`で検証（0除算によるNaN伝播・パニックを防ぐため）。

### 3.3 適合度統計量

- R²・調整済みR²: `include_intercept`により centered TSS（`Σ(y_i-ȳ)²`）/ uncentered TSS
  （`Σy_i²`）を切り替える（statsmodelsの`k_constant`分岐と一致）。調整済みR²は
  `1 - ((n-k_constant)/df_resid)*(1-R²)`。
- 対数尤度: `llf = -(n/2)*(ln(2π) + ln(SSR/n) + 1)`（分散は最尤推定量`SSR/n`。classical標準誤差の
  不偏推定量`SSR/(n-k)`とは異なる）。`aic = -2*llf + 2k`、`bic = -2*llf + ln(n)*k`。
- F統計量: `cov_type`によらず単一の式`F = (β_slopes' Σ⁻¹ β_slopes) / q`（`Σ`は`cov_params`の
  傾き係数部分行列、`q = k - k_constant`）。`cov_type=Classical`のとき古典的F検定と代数的に一致し、
  HC0-3・HAC・clusterではそのままロバストWald検定になる。`q=0`は`f64::NAN`（0除算回避）。
- **`Σ`が数値的にほぼ特異な場合の検出**: 変数間のスケールが極端に異なる設計行列では、`Σ`の条件数が
  倍精度の限界を超えるが非ピボットCholesky分解自体は失敗せず無意味なF統計量を返しうる
  （実測でstatsmodelsとの相対誤差5e10程度）。`ensure_well_conditioned_symmetric_matrix`
  （`crate::shared::linear_algebra`、nonlinear系統とも共有）が`SelfAdjointEigen`で実際の固有値を求め、
  最大固有値との相対比で判定しCholesky分解前に`ComputationFailed`で止める。

### 3.4 `predict()`

- `OLSResults.predict(new_data: pl.DataFrame | None = None) -> list[float]`
  （`OLS`側ではなく`OLSResults`側。`OLS`はfit前の設定を保持するだけのステートレスな値のため）。
- `new_data=None`（デフォルト）: 学習データに対する予測値`ŷ = Xβ̂`を返す（`fit()`時に計算し
  内部に保持。独立したプロパティとしては公開せず`predict()`経由のみ）。
- `new_data`指定時: 新規データに対する予測値（out-of-sample）。`x`と同じ列名を持つ列を含む必要が
  ある（列名でマッチング、列順不問）。`include_intercept=True`でfitした場合、定数項の列は
  `new_data`に含めない（自動付加される）。
- 戻り値は観測順の`list[float]`（`residuals`と同じ形）。学習データ・新規データのどちらでも
  同じ型で返す（統計学の慣習では学習データへの予測を「fitted values」、新規データへの予測を
  「predicted values」と呼び分けるが、本APIは`new_data`の有無で戻り値の型・構造を変えない
  設計方針のため呼び分けない）。点予測だけを返すので単一キーの`dict`にする意味がなく、
  1要素ごとに`dict`を作るコストも避けられる。将来、信頼区間・予測区間を足す場合は
  `predict()`にキーを足さず（引数で戻り値の型が変わるのを避けるため）、別メソッドとして追加する
  （statsmodelsの`get_prediction()`と同じ分け方）。
- **Logitとの命名整合**: `LogitEstimator::predict()`（学習データの予測確率のみを返す設計、
  statsmodelsの`results.predict(exog=None)`と同型）が先に実装・マージ済みだったため、OLS側を
  この命名（`fitted_values`プロパティを作らず`predict(new_data=None)`に一本化）に揃えた。
- エラーハンドリングは列不足・型不一致・NaN/無限大とも既存の`ValidationError`の枠組みをそのまま使う
  （専用のエラーバリアントは新設しない）。

### 3.5 `augment()`

- `OLSResults.augment(new_data: pl.DataFrame | None = None) -> pl.DataFrame`。
  `new_data`の意味・エラーハンドリングは`predict()`と完全に同じ。戻り値が
  `list[dict[str, float]]`ではなく、ソースデータ（`new_data=None`なら学習データ、
  指定時は`new_data`）に予測値の列（`"predicted"`）を1列付加したpolars DataFrameを返す点のみ異なる。
- **プロジェクト全体の「DataFrameは返さない」方針（2章）の唯一の例外**。予測値と元データの行対応を
  分かりやすくしたいというユーザー要望（R `broom::augment()`が先行事例）に応えるため。既存の
  `predict()`/`residuals`/`coef_table()`は変更せず、この用途専用の新規メソッドとして追加した
  （フラグで戻り値の型を変える設計はboolean trapのため不採用）。
- **スコープは予測列のみ**（残差列の同時付加は見送り、必要になれば別issueで拡張検討）。
- **実装層は`engine_pybind`**（`python_package`側で`predict()`の結果を`with_columns()`するだけでも
  実現できるが、列名衝突のエラー送出のしやすさを優先しユーザー判断でRust側に置いた）。
  `OLSResult`（Rust）に`fit()`時の元DataFrameを`training_data: Option<DataFrame>`として非公開保持する
  （polarsの列は内部で参照カウント方式のため、このクローン自体は実質コピーを伴わない）。
  `new_data`指定時は`predict()`と同じ`x`列抽出＋`engine::linear::ols::predict_new_data`を再利用し、
  `new_data`自体をソースにする。
- **列名衝突は`ValidationError`**（ソースデータに既に`"predicted"`列がある場合、`include_intercept=True`
  時の`"const"`列衝突と同じ発想で黙って上書きしない。`engine_pybind::validation::validate_no_existing_column`）。
- `training_data`が`None`（`IVResult.first_stage()`が`OLSResult`を構築する経路——各内生変数の
  第一段階回帰は単一のソースDataFrameを持たないため）の`OLSResults`に対して`augment(new_data=None)`を
  呼ぶと`ValidationError`になる（`new_data`を指定した呼び出しは通常どおり動作する）。

### 3.6 engine/engine_pybind間のデータ受け渡し・エラー変換

- Arrowゼロコピーは Python→Rust境界（`pyo3-polars`の`PyDataFrame`）の受け渡しを指す。
  polars DataFrame→`faer::Mat<f64>`は2段階: `engine_pybind`が列ごとに`Vec<f64>`へ抽出
  （`shared::column_extraction::extract_f64_column`）→`engine`（`OlsInput::from_columns`）が
  `faer::Mat`を組み立てる。この2回のコピー自体は許容する（QR分解本体のコストに対して無視できる）。
- `engine`はpolars/PyO3を知らない。列名が要る検証（`y`/`x`重複、`"const"`衝突、`x`空リスト）は
  `engine_pybind`側の責務。`confidence_level`範囲・`cluster`未指定は`engine`側が検知するため
  `engine_pybind`側で重複チェックしない。
- `engine::linear::common::LeastSquaresError` → `PyErr`対応表:

  | `LeastSquaresError` | Python例外 |
  |---|---|
  | `Common(DimensionMismatch \| InsufficientObservations \| MissingClusterColumn \| InvalidConfidenceLevel \| InsufficientClusters \| InsufficientClustersForInference \| NoRegressors)` | `ValidationError` |
  | `WeightDimensionMismatch \| NonPositiveWeight`（WLS） | `ValidationError` |
  | `InvalidHacLags` | `ValidationError` |
  | `SingularMatrix` | `ComputationError` |
  | `Common(ComputationFailed)` | `ComputationError` |

  `impl From<LeastSquaresError> for PyErr`は書けない（`LeastSquaresError`・`PyErr`ともこのクレート
  外定義の型でorphan ruleに抵触）。関数`least_squares_error_to_pyerr`として実装し
  `.map_err(...)?`で変換する。
- バージョン固定: `pyo3=0.29.2` / `polars=0.55.2` / `pyo3-polars=0.28.0`（すべて`=`固定）。
  `pyo3-polars=0.28.0`が`pyo3="^0.29"`・`polars="^0.55.1"`を要求するための組み合わせ。互換性は数字ではなく
  `pyo3-polars`が使う`polars_ffi::version_0`という安定版FFIプロトコルで担保される。

### 3.7 テスト

- 許容誤差: classical/HC0-3/cluster/係数はRとの実測で相対誤差1e-14程度のため`rtol_strict=1e-8`（`tests/_tolerances.py`の`ols_crosscheck`）。
  HACはRとの`prewhite`/`adjust`慣習差により実測0.4%程度のため`rtol_hac=1e-2`。
- `tests/linear/` に4ファイルで役割分担する:
  `test_ols_api.py`（成功パスの構造・API・オプション反映・`predict()`/`augment()`）/
  `test_ols_validation.py`（`ValidationError`/`ComputationError`パス）/
  `test_ols_reference.py`（statsmodels主リファレンスとの数値照合、`ols.json`＋ライブ照合）/
  `test_ols_crosscheck.py`（R独立実装、`ols_crosscheck.json`）。一般的なテスト方針は
  `.claude/rules/testing-policy.md`を参照。
- `hac_lags_used`（自動選択ラグ数）は、`tests/linear/test_ols_api.py`が複数の標本サイズ
  （境界`n=51200`を含む）でPython側の独立実装`benchmark.common.hac_auto_lag`と直接比較する
  （Rust側`resolve_hac_lags`との式の一致の直接検証。WLS・IV・FE/REのDKも各`*_api.py`で同様）。
- pyfixestはOLSの正確性検証には使わない（HC2/HC3にHC1用の小標本補正を誤って適用する既知の
  実装バグがあるため）。性能比較専用（[`../performance/ols.md`](../performance/ols.md)）。
- 実データセット: `wage1`（`lwage ~ educ + exper + tenure`）・
  `gpa2`（`colgpa ~ sat + hsperc + tothrs`）のWooldridgeデータセット2つ、
  classical/HC0-3で主リファレンス（statsmodels、`test_ols_reference.py`）・
  独立実装（R、`test_ols_crosscheck.py`）の両方と照合する（従来Rクロスチェック側にしか
  無かった実データ検証をstatsmodels側にも追加）。
  `wage1`はさらに地域ダミー（northcen/south/west、基準northeast）から合成したregion列
  でのクラスターロバストSE（実データでのグループ列、4グループ・不均衡サイズ）も両方で検証する。
- `engine`側は上記の固定シナリオ単体テストに加え、property-basedテスト（`proptest`、
  `engine/src/linear/ols/estimator.rs`の`mod proptests`）で不変条件を検証する（詳細な方針は
  `testing-policy.md`「property-basedテスト」参照）。対象プロパティ: 定数項ありなら残差和は常に0、
  yのスカラー倍で係数（切片含む）も同じ倍率でスケールする、xの列順序を入れ替えても係数名で
  対応付ければ値は変わらない、HC0の標準誤差は常にHC1以下。いずれも意図的なバグ注入により
  実際に検出できることを確認済み。
- `white_test()`（3.9節）・`breusch_godfrey_test()`（3.10節）は、診断検定1つで1ファイルにまとめた
  `test_ols_white.py`・`test_ols_breusch_godfrey.py`に置く（上記4分割と同じ区分を1ファイル内に持つ）。
- 上記4ファイルの役割分担（リファレンス実装との数値照合）とは別に、`test_ols_api.py`末尾に
  クラスターロバストSEの統計的健全性チェックを1本持つ
  （`test_cluster_std_error_exceeds_classical_under_true_intra_cluster_correlation`）。
  既存のクラスター系テストは誤差i.i.d.なデータに疑似グループラベルを後付けしたもので、
  「クラスターロバストSEが真のクラスター内相関がある状況で意図通り機能するか」は未検証
  だった。説明変数・誤差の両方にクラスター内相関を
  持たせたMoulton型DGPを使い、クラスターSEが古典的SEより明確に大きくなることを確認する
  （seed固定、実測レンジに対し十分なマージンを持たせた閾値で判定）。リファレンス実装との
  数値比較ではなく本実装内で完結した健全性チェックのため、`freeze.py`の固定CSVパイプラインは
  経由せずテスト内でDGPを都度生成する。

### 3.8 パフォーマンス（要約）

releaseビルド（`maturin develop --release`）必須（debugビルドは最大140倍遅い）。
classical/HC1/clusterはstatsmodels/pyfixest以上に高速、HACも大規模データではほぼ互角。
メモリはengineが一貫して最小。詳細な実測データは[`../performance/ols.md`](../performance/ols.md)参照。
faerのグローバル並列度は`engine::shared::parallelism::ensure_serial()`で常時`Par::Seq`に固定
している（tall-skinnyな設計行列では暗黙の全コア並列化が高速化せず、多コア機・負荷下で
不安定になったため。`engine/src/linear/CLAUDE.md`「faerのグローバル並列度」）。

### 3.9 `white_test()`（事後診断）

- `OLSResults.white_test(statistic: Literal["lm", "f"] = "lm") -> WhiteTestResult`。事後診断なので
  `fit()`では計算せず、利用者が選んで呼ぶ（`docs/spec/inference-conventions.md`6章）。結果型は検定共通の
  frozen dataclass `DiagnosticResult`（`statistic`/`p_value`/`df`/`df_denom`/`distribution`、`to_dict()`）
  を継承した`WhiteTestResult`（追加フィールド`aux_terms`/`dropped_terms`）。`distribution`は
  `Literal["chi2", "f"]`の文字列（`stat_dist`と同じ流儀）。
- **定義**: 残差の二乗を、説明変数・その二乗・説明変数同士の交差項に回帰する補助回帰を行い、
  LM版は`LM = n·R²`（帰無分布`χ²(q)`）、F版は古典的なF検定（`F(q, n-q-1)`）。`q`は補助回帰の
  定数以外の列数。LM/F版は戻り値を分けず`statistic`引数で選ぶ（型を単純にするため）。
  `cov_type`には依存しない（古典的な等分散を仮定する検定）。
- **補助回帰は常に定数を含める**（元のモデルが`include_intercept=False`でも同じ）。`aux_terms`の
  先頭は常に補助回帰の定数`"const"`。元のモデルに定数が無くても補助回帰の定数があることを結果で
  読めるようにするため。元のモデルが`include_intercept=False`で`x`に`"const"`という名前の列を
  持つ場合は`"const"`が2つ現れるが、先頭が補助回帰の定数である（位置で区別できる）。
- **項と除外ルール**: 項は`x`の各列・各列の二乗・列の組の積の順（ラベルは`"x1"`/`"x1^2"`/`"x1:x2"`）。
  定数列（全て0を含む）と、先に採用した項と数値的に同一の列（相対許容誤差1e-12。ダミーの二乗
  `d*d == d`、排他的ダミー同士の積が全て0になる場合等）を除き、除いた項を`dropped_terms`に返す。
  定数の説明変数（`include_intercept=False`のモデルに入れた定数列）を含む項は、その定数倍の別の項
  なので値の比較を待たず全て除く。**`df`は除外後の項の数（ランクに基づく）**。`fit()`は完全な
  多重共線性を`ComputationError`で弾くが、補助回帰の重複はテスト自身の構成（`d² = d`）から機械的に
  生じ利用者のモデル指定の誤りではないため、ここでは除いて続行する（何を使ったかは結果で見える）。
  ラベルは出力専用で、式としてパースしない・引数に受け付けない（formula方式を採らない方針
  〔CLAUDE.md 2章〕は入力の設計の話）。
- **数値の扱い**: 補助回帰の各列は標準化してから回帰する。定数ありの`R²`は列の平行移動・スケールで
  変わらないが、生の値のままだと平均が標準偏差より桁違いに大きい変数（賃金・人口等）の二乗項が
  元の列とほぼ共線になり、`fit()`の特異性判定・条件数チェックで誤って失敗するため（実測: 平均5e3、
  SD1e3の列で失敗）。`R²`が非有限、または1以上ならF統計量が定義できず`ComputationError`
  （`f64::max`はNaNを無視して0.0を返すため、丸める前に非有限を判定する）。
- **エラー**: `statistic`が不正・観測数が補助回帰の列数（定数込み）以下
  （`LeastSquaresError::InsufficientObservationsForAuxRegression`、`CommonError::
  InsufficientObservations`は元のモデルの`k`を指すためメッセージを分けた）・`training_data`が
  無い結果（`IVResults.first_stage()`由来）は`ValidationError`。全ての`x`が定数、補助回帰が除外後も
  特異（既知の制限: 全カテゴリのダミーを`include_intercept=False`で入れると補助回帰の定数と
  共線になる）、または補助回帰が推定できない場合は`ComputationError`（メッセージに
  `White test`と補助回帰であることを含め、元のモデルの共線性と取り違えないようにする）。
- **実装**: engineの`linear::diagnostics::white_test`（`x`の列・列名・残差を受け取り、補助回帰に
  `OlsEstimator::fit`〔`CovType::Classical`〕を再利用）。`OLSResult`は`OlsEstimator`を保持しないため
  `engine_pybind`が`training_data`と`param_names`から`x`を再抽出して渡す（`predict_for`と同じ経路、
  `augment()`と同様`training_data`が無ければ`ValidationError`）。LM/Fの両方をengineが計算し、
  `statistic`の選択は`engine_pybind`。p値は`statrs`の`sf`（`1 - cdf`は裾で潰れるため使わない）。
- **リファレンスとの関係**: statsmodels `het_white`は補助回帰の項の重複を除かないが、補助回帰を`OLS`で
  当てはめ自由度をランク（`df_model = rank - k_constant`）で数えるため、ダミーの二乗のような重複列
  があっても本実装と一致する（`SingularMatrixWarning`は出る）。Rは`lmtest::bptest(studentize = TRUE)`
  （Koenker版、`n·R²`）。F版はRに専用関数が無いため同じ補助回帰の`lm`から計算する。いずれも
  `lm`のエイリアス処理でランクに基づく自由度になる。
- **テスト**: `tests/linear/test_ols_white.py`の1ファイルにまとめる（構造・エラーパス・statsmodels凍結
  フィクスチャ`ols_white.json`・Rクロスチェック`ols_white_crosscheck.json`）。合成データの
  `WHITE_SYNTHETIC_SCENARIOS`（`baseline_df1`〔n=5で補助回帰に足りない〕・`scale_variance`・
  `perfect_multicollinearity`〔元の`fit()`が失敗〕を除く、説明変数1個の`baseline_k1`を含む）、
  baselineの`include_intercept=False`、Wooldridge実データ（`wage1`・`gpa2`・ダミーを含む
  `wage1_dummies`・二乗列を含む`wage1_polynomial`・排他的ダミーの`wage1_region`）を、statsmodels・R
  の両方と照合する。観測数の境界（`n = k`で拒否、`n = k + 1`で成功し`df_denom = 1`）と、重複除外後の
  列数での境界判定も確認する。許容誤差は`ols_white_reference`/`ols_white_crosscheck`（`rtol=1e-8`、
  実測最大相対誤差約2e-12）。p値は絶対誤差フロアを使わず相対誤差のみで比較する（裾の1e-39級の
  p値が`sf`ではなく`1 - cdf`で0に潰れる回帰を検出するため）。engine側は`diagnostics.rs`の`mod tests`（statsmodels値との照合、ダミー・定数・スケール・
  NaN残差・共線等）。

### 3.10 `breusch_godfrey_test()`（事後診断）

- `OLSResults.breusch_godfrey_test(time: str, nlags: int, statistic: Literal["lm", "f"] = "lm")
  -> BreuschGodfreyTestResult`。事後診断なので`fit()`では計算しない（`docs/spec/inference-conventions.md`
  6章）。結果型は`DiagnosticResult`を継承した`BreuschGodfreyTestResult`（追加フィールド`nlags`。
  `df`と同じ値だがラグ次数を直接読めるよう別に持つ）。
- **`time`と`nlags`は必須引数**。行順を時間順とみなす暗黙の既定は置かない（`hac_time`と同じ理由。
  横断面データでは順序に意味が無く、この検定自体が意味を持たない。パッケージは警告を出さない）。
  `nlags`も、statsmodelsの既定`min(10, n//5)`・Rの`order = 1`のどちらも恣意的な値のため既定を置かない。
  `nlags`は`int`（`bool`・`float`は`TypeError`）、1以上（未満は`ValidationError`）。
- **定義**（Greene・R `lmtest::bgtest`と同じ）: 残差`û`を、元のモデルの説明変数`X`と`û`自身の1〜`nlags`次の
  ラグ（時間列の昇順に並べた残差の遅れ）に回帰する補助回帰を行う。LM版は`LM = n·R²`（`χ²(nlags)`）、
  F版は`F = ((Σû² - SSR_u)/m)/(SSR_u/(n - k - m))`（`F(m, n - k - m)`、`m = nlags`、`k`は元のモデルの
  係数の数）。`û`は`X`と直交するOLS残差なので、補助回帰の残差二乗和`SSR_u`だけから
  `LM = n·(1 - SSR_u/Σû²)`と書ける。`cov_type`には依存しない。
- **サンプル前期間のラグは0で埋める**（statsmodels・R `bgtest`の既定・Stataと同じ）。補助回帰の観測数は
  常に`n`。観測を落とす版（R `fill = NA`）は対応する主リファレンスが無いため提供しない。
- **補助回帰は元のモデルの`X`をそのまま使い、`include_intercept=False`でも定数を足さない**（R・Greeneの定義）。
  statsmodelsは切片なしのモデル（`k_constant == 0`）でだけ補助回帰に定数を足すため定義が異なり、
  このケースはRのみで照合する（切片ありはstatsmodels・Rの両方）。
- **時間順**: `time`はデータ（`fit()`に渡したDataFrame）の列名。値の昇順が時間順で、`hac_time`と同じ
  `extract_time_order_ranks`で順位にする（整数・浮動小数・`Decimal`・`Date`・`Datetime`、同値・欠損値・
  NaN・無限大は`ValidationError`）。**値の間隔（欠番）は見ず、並べた順にラグを取る**。
- **数値の扱い**: 補助回帰は`OlsEstimator::fit`を使わず、列ノルムでスケールした列ピボットQRで
  `SSR_u`を直接求める（必要なのは`SSR_u`だけ）。Whiteと違い列を標準化（中心化）しないのは、切片なしの
  モデルを元の`X`のまま扱うため中心化すると列空間が変わるから（`SSR_u`は列の正のスケールでは変わらない）。
  完全適合（`SSR_u`が`k_aux·ε·Σû²`以下）はFの分母が0になるため`ComputationError`。
- **エラー**: `statistic`不正・`nlags < 1`・観測数が補助回帰の列数`k + nlags`以下
  （`InsufficientObservationsForAuxRegression`）・`time`列の不備・`training_data`が無い結果は
  `ValidationError`、型違い（`time`/`statistic`が`str`でない・`nlags`が`int`でない）は`TypeError`。
  補助回帰が特異（ラグが`X`と共線）・残差が全て0・完全適合・非有限は`ComputationError`
  （メッセージに`Breusch-Godfrey test`を含め、元のモデルの共線性と取り違えないようにする）。
  残差が全て0・完全適合は元の`fit()`が先に`ComputationError`にするためPython経由では
  到達しにくく、Rustの単体テストで担保する。
- **実装**: engineの`linear::diagnostics::breusch_godfrey_test`（説明変数の列・`has_intercept`・残差・
  時間の順位・`nlags`）。`engine_pybind`が`training_data`から`x`と時間列を再抽出して渡す
  （`white_test()`と同じ経路、`training_data`が無ければ`ValidationError`）。LM/Fの両方をengineが
  計算し、`statistic`の選択は`engine_pybind`。
- **リファレンス**: 切片ありはstatsmodels `acorr_breusch_godfrey`（主）とR `bgtest`（`type = "Chisq"/"F"`、
  `fill = 0`）、切片なしはR `bgtest`のみ。
- **テスト**: `tests/linear/test_ols_breusch_godfrey.py`の1ファイル（構造・エラーパス・statsmodels凍結
  フィクスチャ`ols_breusch_godfrey.json`・Rクロスチェック`ols_breusch_godfrey_crosscheck.json`）。
  合成データ（`BG_SYNTHETIC_SCENARIOS`、行番号を時間列にする）×`nlags = 1, 4`、切片なし（baseline・
  autocorrelated、Rのみ）、Wooldridge実データ`phillips`（`inf ~ unem`、行を無作為に並べ替えて`year`を
  時間列として渡す）を照合する。許容誤差は`ols_breusch_godfrey_reference`/`_crosscheck`（`rtol=1e-8`、
  実測最大相対誤差約6e-12、p値は相対誤差のみ）。観測数の境界（`n = k + nlags`で拒否、
  `n = k + nlags + 1`で成功し`df_denom = 1`）、時間列による並べ替え、時間列のdtype、`cov_type`非依存も確認する。
- **`nlags`の上限とメモリ**: 上限は数理的な条件`n > k + nlags`（`df_denom >= 1`）だけで、観測数に対する
  割合や固定値の上限は置かない（統計的に正当化できる線がなく、R・statsmodelsも置かない）。補助回帰は
  `n × (k + nlags)`の行列を作るため、`nlags`が`n`に近く`n`が大きいと確保できないことがある。行列の
  サイズ計算のオーバーフローと確保の失敗（QR分解が内部で確保する同サイズの複製を含む）は、プロセスを
  異常終了させず`ComputationError`（`nlags`を小さくするよう促す）にする。ただしLinuxのメモリの
  オーバーコミットにより、確保には成功しても実際に使うと終了させられる場合は検知できない。

### 3.11 `breusch_pagan_test()`（事後診断）

- `OLSResults.breusch_pagan_test(variables: list[str] | None = None, statistic: Literal["lm", "f"] = "lm")
  -> BreuschPaganTestResult`。`fit()`では計算しない事後診断（`docs/spec/inference-conventions.md`
  6章）。結果型は`DiagnosticResult`を継承した`BreuschPaganTestResult`（追加フィールドは
  `WhiteTestResult`と同じ`aux_terms`・`dropped_terms`）。`cov_type`には依存しない。
- **定義**: Koenkerの標準化版（`LM = n·R²`、`R²`は`û²`を定数と`variables`の列に回帰した補助回帰の
  決定係数、帰無分布`χ²(q)`）。F版は補助回帰の全傾きがゼロという古典的なF検定`F(q, n - q - 1)`。
  `q`は除外後の`variables`の列数。元のBreusch-Pagan（1979）の`ESS/2`版は誤差の正規性を仮定するため
  扱わない（R `bptest(studentize = TRUE)`・statsmodels `het_breuschpagan(robust = True)`の既定と同じ版）。
  後から`studentize`引数を足す場合は後方互換に追加できる。
- **`variables`**: 不均一分散の原因と疑う変数の列名のリスト（`x`と同じくlist渡し、式は使わない）。
  既定（`None`）はモデルの説明変数。**モデルに入っていない列や`y`列も指定できる**（拒否しない。
  BP検定の本質は`Z`を利用者が選ぶことで、R `varformula`・statsmodels `exog_het`も同じ）。型・検査は`x`と
  同じ（`list`以外・`str`以外の要素は`TypeError`、空リスト・同じ列名の重複・存在しない列・数値として
  使えないdtype・欠損値/NaN/無限大は`ValidationError`）。ただし`"const"`という列名は拒否しない
  （`aux_terms`の先頭の定数と紛れるが位置で区別できる）。
- **補助回帰は常に定数を含める**（`include_intercept=False`でも同じ）。`aux_terms`の先頭は常に`"const"`、
  項のラベルは列名そのまま。White検定と同じ除外ルール: 定数列と、先に指定した列と数値的に同一の列
  （相対許容誤差1e-12）を除いて`dropped_terms`に返し、**`df`は除外後の列数（ランクに基づく）**。
  除くのは定数と完全に同一な列だけで、`2 * x1`のような同一ではない共線列は除かず`ComputationError`
  （全カテゴリのダミーと定数の組み合わせも同様。`white_test()`と同じ制限）。補助回帰の各列は標準化して
  から回帰する（`R²`は変わらず、平均の大きい変数でも失敗しない）。
- **エラー**: `statistic`が不正・観測数が補助回帰の列数（定数込み`q + 1`）以下・`training_data`が無い結果
  （`IVResults.first_stage()`由来）は`ValidationError`。全ての`variables`が定数、補助回帰が除外後も特異、
  または`R²`が定義できない（残差の二乗が定数等）場合は`ComputationError`（メッセージに
  `Breusch-Pagan test`と補助回帰であることを含める）。
- **実装**: engineの`linear::diagnostics::breusch_pagan_test`（`z`の列・列名・残差を受け取る）。
  `white_test`と補助回帰以降の処理（項の除外・標準化・統計量・p値、`squared_residual_aux_test`）を共有し、
  違いは補助回帰の説明変数（`x`・二乗・交差項か、`variables`そのものか）だけ。`engine_pybind`が
  `training_data`から`variables`（既定は`param_names`から取り出した`x`）の列を再抽出して渡す。
  p値は`statrs`の`sf`。
- **リファレンスとの関係**: 主リファレンスはstatsmodels `het_breuschpagan(robust=True)`。`exog_het`に
  定数が無いと補助回帰に定数を入れないため、常に定数列を足して渡す。**LMのp値の自由度を列数-1で数え
  列のランクを見ない**ため、定数列・重複列を含む`Z`は除いた後の列を渡す（`reference_variables`）。
  定数・重複を落とす挙動そのものはR `lmtest::bptest(studentize = TRUE)`（自由度は補助回帰の
  ランク-1）にそのまま渡して照合する。F版はRに専用関数が無いため同じ補助回帰の`lm`から計算する。
- **テスト**: `tests/linear/test_ols_breusch_pagan.py`の1ファイルにまとめる（構造・エラーパス・
  statsmodels凍結フィクスチャ`ols_breusch_pagan.json`・Rクロスチェック`ols_breusch_pagan_crosscheck.json`）。
  合成データの`BP_SYNTHETIC_CASES`（`Z`＝モデルのxで`WHITE_SYNTHETIC_SCENARIOS`、`baseline`の`Z`をモデルの
  一部・モデル外の列にしたケース、切片なし、定数列・重複列を含むケース、`baseline_df1`〔n=5で`df_denom = 1`の成功パス。White検定では列数不足〕、誤差分散が`|x1|`に比例する`heteroskedastic`で`Z = |x1|`にして実際に棄却する裾のp値の経路）と、Wooldridge実データ
  （`hprice1`・`hprice1_log`〔教科書の例8.4の公表値とも照合〕・`wage1`・モデル外の列を`Z`にした
  `wage1_outside_model`）を、statsmodels・Rの両方と照合する。観測数の境界（`n = q + 1`で拒否、
  `n = q + 2`で成功し`df_denom = 1`）も確認する。許容誤差は`ols_breusch_pagan_reference`/
  `ols_breusch_pagan_crosscheck`（`rtol=1e-8`、実測最大相対誤差約2e-12）。p値は相対誤差のみで比較する。

## 4. 未実装・未対応

- 診断検定のうち、RESET・Jarque-Bera等（別メソッドとして順次追加。時間順が必要な検定は
  時間列を必須引数とする、`docs/spec/inference-conventions.md`6章）
- `breusch_pagan_test()`の元のBreusch-Pagan（1979）の`ESS/2`版（正規性を仮定する版）。必要になれば`studentize`引数で後方互換に追加できる
- `white_test()`・`breusch_pagan_test()`・`breusch_godfrey_test()`のWLS・IV（`first_stage()`以外）への展開（WLSは残差が重み付きかどうかで意味が変わるため別途検討）
- `predict()`の信頼区間・予測区間（点予測のみ対応。追加する場合は別メソッド、3.4節参照）
- HACの完全なデータ依存バンド幅自動選択（Newey & West 1994）: 参照実装がなく数値照合手段がないため見送り
- `SingularMatrix`のエラーメッセージを状況に応じて分岐させる（優先度低）
