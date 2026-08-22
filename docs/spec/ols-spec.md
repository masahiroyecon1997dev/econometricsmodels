# OLS 仕様書

OLS（最小二乗法）の確定済み仕様。`engine/src/linear/ols.rs`・`engine_pybind/src/linear/ols.rs`・
`python_package/econometricsmodels/linear/ols.py`として実装済み。パフォーマンス比較の詳細は
[`ols-performance-notes.md`](./ols-performance-notes.md)、CI/CD・セキュリティはmethod非依存のため
[`ci-cd-notes.md`](./ci-cd-notes.md)を参照。

## 1. API引数

3層構成: `OLS(data, y, x, options).fit() -> OlsResults`（python_package）→
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
  | `cluster_col` | `str \| None` | `None` | `cov_type="cluster"`時のグループキー列名（`data`内の列） |
  | `hac_lags` | `int \| None` | `None` | `cov_type="hac"`時のラグ数。`None`なら`L=floor(4*(n/100)^(2/9))`で自動計算 |
  | `time_col` | `str \| None` | `None` | `cov_type="hac"`時の時系列順序列。`None`なら`data`の行順を使用 |

- `include_intercept=True`のとき`x`に`"const"`列があるとエラー（自動追加する定数項と衝突）。
  `x`に自前の定数列を含める重複検出は行わず、生じる多重共線性は`SingularMatrix`に委ねる。
- 欠損値（NaN/無限大）は常にエラー。listwise deletionはしない。
- 検定分布は**t分布**（正規分布ではない）。`cov_type`がHC系/clusterでもF検定はロバストWald検定に切り替える。
- `confidence_level`は`fit()`時に一度だけ使用し、結果に固定して含める（再計算用の可変引数は提供しない）。

## 2. 結果構造体

`OLSResult`（`#[pyclass]`、`skip_from_py_object`）が公開する配列＋名前リスト:
`params` / `std_errors` / `t_stats` / `p_values` / `conf_lower` / `conf_upper` / `param_names` /
`residuals` / `dep_var_name` / `n_obs` / `cov_type`（実際に使われた種別の小文字文字列） /
`r_squared` / `r_squared_adj` / `f_statistic` / `f_p_value` / `log_likelihood` / `aic` / `bic`。

- `conf_int`は`conf_lower`/`conf_upper`の2配列に分割（engine内部表現・pyo3実装の簡潔さを優先）。
- `k×kの分散共分散行列（cov_params）はPython側に公開しない`。`OlsEstimator`自体は非公開
  フィールドとして保持する（クレート内の他系統からの部分Wald検定の再利用のため、
  `engine/src/linear/CLAUDE.md`参照。Issue #164でIVのWu-Hausman検定用に追加するまでは
  `fit()`内のローカル変数として使い切っていた）が、`engine_pybind`側に公開する`OLSResult`
  には引き続き含めない。
- `summary()`（テキスト整形）・DataFrame版の`coef_table()`/`conf_int()`は作らない
  （economiconのGUIエンジンという用途上、テキスト表示・対話的操作を前提にしないため）。
- python_package層（`OlsResults`）:
  - `params`/`std_errors`/`t_stats`/`p_values`/`conf_int`: 係数名→値の`dict`（O(1)取り出し用）。
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
- 標準誤差計算用の`(X'X)⁻¹`（`xtx_inverse`）は`X'X`自体のCholesky分解で求める（QR分解の`R`因子から
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
- `time_col`未指定なら`data`の行順を時系列順とみなす。指定時は昇順ソートしたインデックスで
  ラグ付き自己共分散を計算する（`OlsInput`自体は並べ替えない。Python側に返す残差配列と
  元DataFrameの行対応を保つため）。
- **パフォーマンス上の罠**: `k×k`という小さい出力サイズの行列積で、faer既定の並列実行は
  ディスパッチオーバーヘッドが計算本体を上回り逐次より遅くなる（実測n=10,000,k=2で6倍悪化）。
  `hac_cov_params`内でのみ`Par::Seq`を明示指定して回避している。他手法で同様の小さい行列の
  頻繁な積を書く場合も並列化の要否を実測してから決めること。
- statsmodelsとの照合は`cov_kwds={"maxlags": L}, use_t=True`。`use_correction`（小標本補正）は
  既定の`False`のままで一致することを確認済み。

**クラスター**: $\hat S = \sum_g S_g S_g^\top$（$S_g = \sum_{i\in g}\hat\varepsilon_i x_i$）。
- グループ化は**`BTreeMap`を使う（`HashMap`は禁止）**: `HashMap`はプロセスごとのハッシュシードで
  反復順序が変わり、浮動小数点加算の非結合性により`fit()`を複数回呼ぶと標準誤差が1 ULP程度ぶれる
  非決定性バグを起こす（`fit_cluster_std_errors_are_deterministic_across_repeated_fits`で固定）。
  クラスター系の実装を今後増やす場合も同じ罠がある。
- 小標本補正`G/(G-1) * (n-1)/(n-k)`は常に適用し、無効化オプションは設けない（statsmodels
  `cov_cluster`の既定`use_correction=True`と一致）。
- t検定・信頼区間・F検定の自由度は`cov_type="cluster"`のときのみ`n-k`ではなく**`G-1`**に切り替える
  （statsmodelsの既定`df_correction=True`、計量経済学の標準的慣行）。`df_resid`自体（σ̂²・調整済み
  R²・AIC/BIC）は常に`n-k`のまま。
- **G≤qの境界**: $\hat S$はG個のランク1行列の和のため`rank(Ŝ) ≤ G`。F検定が使う`q×q`
  （`q`=傾き係数の数）部分行列はG<qのとき構造的に特異になり`fit()`全体が`ComputationError`
  になる（係数・標準誤差自体はG≥2なら計算できる）。「クラスタ数境界の成功パス」のテストは
  qをG以下に保つ必要がある。他のクラスターロバストSEを持つ手法（IV等）にも同様に当てはまる。
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
  （`crate::linear_algebra`、nonlinear系統とも共有）が`SelfAdjointEigen`で実際の固有値を求め、
  最大固有値との相対比で判定しCholesky分解前に`ComputationFailed`で止める。

### 3.4 `predict()`

- `OlsResults.predict(new_data: pl.DataFrame | None = None) -> list[dict[str, float]]`
  （`OLS`側ではなく`OlsResults`側。`OLS`はfit前の設定を保持するだけのステートレスな値のため）。
- `new_data=None`（デフォルト）: 学習データに対する予測値`ŷ = Xβ̂`を返す（`fit()`時に計算し
  内部に保持。独立したプロパティとしては公開せず`predict()`経由のみ）。
- `new_data`指定時: 新規データに対する予測値（out-of-sample）。`x`と同じ列名を持つ列を含む必要が
  ある（列名でマッチング、列順不問）。`include_intercept=True`でfitした場合、定数項の列は
  `new_data`に含めない（自動付加される）。
- 戻り値は行指向`list[dict[str, float]]`で統一（点予測のみの現段階では1キーのみだが、将来
  信頼区間・予測区間を追加する場合にキーを追加できる形にするため）。
- **Logitとの命名整合**: `LogitEstimator::predict()`（学習データの予測確率のみを返す設計、
  statsmodelsの`results.predict(exog=None)`と同型）が先に実装・マージ済みだったため、OLS側を
  この命名（`fitted_values`プロパティを作らず`predict(new_data=None)`に一本化）に揃えた。
- エラーハンドリングは列不足・型不一致・NaN/無限大とも既存の`ValidationError`の枠組みをそのまま使う
  （専用のエラーバリアントは新設しない）。

### 3.5 engine/engine_pybind間のデータ受け渡し・エラー変換

- Arrowゼロコピーは Python→Rust境界（`pyo3-polars`の`PyDataFrame`）の受け渡しを指す。
  polars DataFrame→`faer::Mat<f64>`は2段階: `engine_pybind`が列ごとに`Vec<f64>`へ抽出
  （`column_extraction::extract_f64_column`）→`engine`（`OlsInput::from_columns`）が
  `faer::Mat`を組み立てる。この2回のコピー自体は許容する（QR分解本体のコストに対して無視できる）。
- `engine`はpolars/PyO3を知らない。列名が要る検証（`y`/`x`重複、`"const"`衝突、`x`空リスト）は
  `engine_pybind`側の責務。`confidence_level`範囲・`cluster_col`未指定は`engine`側が検知するため
  `engine_pybind`側で重複チェックしない。
- `engine::linear::common::LeastSquaresError` → `PyErr`対応表:

  | `LeastSquaresError` | Python例外 |
  |---|---|
  | `Common(DimensionMismatch \| InsufficientObservations \| MissingClusterColumn \| InvalidConfidenceLevel \| InsufficientClusters)` | `ValidationError` |
  | `InvalidHacLags` | `ValidationError` |
  | `SingularMatrix` | `ComputationError` |
  | `Common(ComputationFailed)` | `ComputationError` |

  `impl From<LeastSquaresError> for PyErr`は書けない（`LeastSquaresError`・`PyErr`ともこのクレート
  外定義の型でorphan ruleに抵触）。関数`least_squares_error_to_pyerr`として実装し
  `.map_err(...)?`で変換する。
- バージョン固定: `pyo3=0.29.2` / `polars=0.55.2` / `pyo3-polars=0.28.0`（すべて`=`固定、Issue #49で更新）。
  `pyo3-polars=0.28.0`が`pyo3="^0.29"`・`polars="^0.55.1"`を要求するための組み合わせ。互換性は数字ではなく
  `pyo3-polars`が使う`polars_ffi::version_0`という安定版FFIプロトコルで担保される。

### 3.6 テスト

- 許容誤差: classical/HC0-3/cluster/係数はRとの実測で相対誤差1e-14程度のため`RTOL_STRICT=1e-8`。
  HACはRとの`prewhite`/`adjust`慣習差により実測0.4%程度のため`RTOL_HAC=1e-2`。
- `test_ols.py`（構造・API・エラーパス）/ `test_ols_fixtures.py`（statsmodels主リファレンス、
  `ols.json`）/ `test_ols_crosscheck.py`（R独立実装、`ols_crosscheck.json`）の3ファイルで
  役割分担する。一般的なテスト方針は`.claude/rules/testing-policy.md`を参照。
- pyfixestはOLSの正確性検証には使わない（HC2/HC3にHC1用の小標本補正を誤って適用する既知の
  実装バグがあるため）。性能比較専用（[`ols-performance-notes.md`](./ols-performance-notes.md)）。
- `engine`側は上記の固定シナリオ単体テストに加え、property-basedテスト（`proptest`、
  `engine/src/linear/ols.rs`の`mod proptests`）で不変条件を検証する（詳細な方針は
  `testing-policy.md`「property-basedテスト」参照）。対象プロパティ: 定数項ありなら残差和は常に0、
  yのスカラー倍で係数（切片含む）も同じ倍率でスケールする、xの列順序を入れ替えても係数名で
  対応付ければ値は変わらない、HC0の標準誤差は常にHC1以下。いずれも意図的なバグ注入により
  実際に検出できることを確認済み。

### 3.7 パフォーマンス（要約）

releaseビルド（`maturin develop --release`）必須（debugビルドは最大140倍遅い）。
classical/HC1/clusterはstatsmodels/pyfixest以上に高速、HACも大規模データではほぼ互角。
メモリはengineが一貫して最小。詳細な実測データは[`ols-performance-notes.md`](./ols-performance-notes.md)参照。

## 4. 未実装・未対応

- `predict()`の信頼区間・予測区間（点予測のみ対応）
- WLSへの`predict()`適用（Issue #132）
- HACの完全なデータ依存バンド幅自動選択（Newey & West 1994）: 参照実装がなく数値照合手段がないため見送り
- `SingularMatrix`のエラーメッセージを状況に応じて分岐させる（優先度低）
