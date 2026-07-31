# OLS 内部実装ノート（パラメータ設計以外）

OLSのAPI・オプション設計（[`ols-api-design.md`](./ols-api-design.md) / [`ols-standard-errors.md`](./ols-standard-errors.md)）とは別に、**パラメータ以外の内部実装で決めたこと・まだ決まっていないこと**をまとめる。トピック別に構成しており、実装の詳細はコード（`engine/src/linear/ols.rs`等）のコメントを参照。ここには設計判断とその理由（rationale）のみを記載する。

## 1. エラーハンドリング

`engine::linear::common::LeastSquaresError`のバリアントとPython例外の対応（`ValidationError` / `ComputationError`の2階層、`.claude/rules/rust-style.md`「エラーハンドリング」参照）:

**改名の経緯**: 元々`OlsError`という名前で`engine/src/linear/ols.rs`にOLS単体のエラー型として定義していたが、WLSが`WeightDimensionMismatch`/`NonPositiveWeight`バリアントを追加する形で同じ型をそのまま再利用しており（4.2節）、「OLS単体のエラー型」という名前と実態が食い違っていた。実態（OLS/WLS/将来のGLS・区分回帰で共有する最小二乗法系エラー型）に合わせて`engine/src/linear/common.rs`に切り出し、`LeastSquaresError`に改名した（nonlinear系統の`MleError`と同じ、系統名ではなく推定方式名で命名する方針）。

**系統をまたぐ共通化**: `DimensionMismatch`/`InsufficientObservations`/`InvalidConfidenceLevel`/`MissingClusterColumn`/`InsufficientClusters`/`ComputationFailed`の6バリアントは、nonlinear系統の`MleError`と文言まで完全に重複していたため`engine::error::CommonError`に切り出した。`LeastSquaresError`はこれを`#[error(transparent)] Common(#[from] CommonError)`バリアントとして保持する（下表の6バリアントは実体としては`LeastSquaresError::Common(CommonError::X)`）。`WeightDimensionMismatch`/`NonPositiveWeight`/`InvalidHacLags`/`SingularMatrix`はOLS/WLS固有のため`LeastSquaresError`に残る。詳細は`nonlinear-implementation-notes.md`「系統をまたぐ重複バリデーションエラーの共通化」参照。

| `LeastSquaresError`のバリアント | Python例外 |
|---|---|
| `Common(CommonError::DimensionMismatch)` | `ValidationError` |
| `Common(CommonError::InsufficientObservations)` | `ValidationError` |
| `Common(CommonError::MissingClusterColumn)` | `ValidationError` |
| `Common(CommonError::InvalidConfidenceLevel)` | `ValidationError` |
| `Common(CommonError::InsufficientClusters)` | `ValidationError` |
| `InvalidHacLags` | `ValidationError` |
| `SingularMatrix` | `ComputationError` |
| `Common(CommonError::ComputationFailed)`（分布計算等の失敗） | `ComputationError` |

- **`InsufficientClusters { g: usize }`はクラスター数`g < 2`を`OlsEstimator::fit`の入口で検証する**。検証しないと0除算からのNaN伝播でパニックする（`correction`計算の`n_groups - 1`が0除算になるため）。
- **`InvalidTimeColumn`（`time_col`のf64キャスト失敗）は`LeastSquaresError`に含めない**。`engine`は`&[f64]`等クリーンな値のみを受け取る前提（モジュール冒頭のdocコメント参照）で、キャスト失敗の検出は`engine_pybind::column_extraction`側の責務。同じ理由で、欠損値（null）検出も`engine`のエラー型には含めない。
- **`OlsInput::from_columns`は`x_names.len() != x_columns.len()`を`debug_assert!`のままにする**（`Result`化しない）。これは`engine_pybind`側の実装バグでしか起こり得ない内部契約であり、実データに起因する`DimensionMismatch`（こちらは`Result`で返す）とは性質が異なるため区別している。ただし`engine_pybind::fit`は常に同一のpolars DataFrameから列を抽出するため、`DimensionMismatch`自体もPython API境界からは実質到達不能の可能性が高い（テストレビューで確認、8章参照）。

### `engine_pybind`受け口レベルの追加バリデーション

`engine`が列名を知らないため検知できず、`engine_pybind::fit`側で`ValidationError`として弾く項目:

- `y`と`x`に同じ列名が含まれる場合（完全な多重共線性になるため、engine側の一般的な`SingularMatrix`より先に、分かりやすいメッセージで弾く）
- `x`内に重複した列名がある場合
- `include_intercept=true`のとき、`x`に`"const"`という列名がある場合（自動追加する定数項の名前と衝突する）
- `x`が空リストの場合

`confidence_level`の範囲チェック、`cov_type="cluster"`なのに`cluster_col`未指定のチェックは、`engine::OlsEstimator::fit`（`InvalidConfidenceLevel`）・`CovType::Cluster`（`MissingClusterColumn`）が既に検知するため、`engine_pybind`側では重複してチェックしない。

## 2. engineの設計判断

- **`y`は`list[str]`ではなく`str`**: Phase1〜6（VAR等の一部時系列手法を除く）でyは常に1変数。`list[str]`にすると「長さ1であること」を全推定関数が実行時検証する必要が生じるため、型で表現する。真に多変量なyが必要な手法（VAR等）が出てきた場合は、その手法だけ`y: list[str]`にする。
- **推定量構造体（`OlsInput`/`OlsEstimator`）はフィールドをprivateにする**: 詳細・理由は`.claude/rules/rust-style.md`「推定量構造体の設計（全手法共通）」参照。OLSに限らず全手法共通のルール。
- **特異性判定は相対閾値**: `engine::linear::ols::ensure_full_rank`が、`col_piv_qr`の`R`の対角成分（列ピボットにより絶対値降順）のうち最大値`|R[0,0]|`を基準に、`threshold = k * f64::EPSILON * |R[0,0]|`未満の対角成分があればランク落ちと判定する。絶対閾値（データのスケールに依存する固定値）は不採用（`.claude/rules/rust-style.md`「線形代数」参照）。
- **係数の求め方: 列ピボットQR（Cholesky不採用）**: `X'Xβ=X'y`をCholesky分解で解く方式は不採用。理由: `X'X`を明示的に作ると条件数が2乗になり数値的に不利な上、`col_piv_qr`なら特異性検出（上記）と係数計算を同じ分解で一度に行える。`OlsEstimator::fit`は`qr.solve_lstsq(input.y())`で係数を得る（列ピボットの並べ替えは`solve_lstsq`内部で吸収され、返り値は元の列順のまま）。`n <= k`（観測数不足）は`OlsEstimator::fit`側で検証する。

## 3. 標準誤差の実装

### classical

- `σ̂²(X'X)⁻¹`の対角成分の平方根（`σ̂² = SSR/(n-k)`）。`X'X`は対称正定値であることが既に特異性検出（`ensure_full_rank`）で保証されているため、`X'X`自体のCholesky分解（`Llt`）で逆行列を求める（`xtx_inverse`関数、列ピボットQR分解を再利用する方式ではない）。
- t統計量: `params / std_errors`。p値: 両側検定、自由度`n-k`のt分布のCDFから計算（正規分布ではない、`ols-api-design.md`「検定分布」参照）。
- t分布のCDF・逆CDFには**statrs**クレート（`=0.18.0`固定、`default-features = false`でnalgebra/rand機能を除外）を使用。

### HC0〜HC3ロバスト標準誤差

- `engine::linear::ols::CovType`（`Classical`/`Hc0`/`Hc1`/`Hc2`/`Hc3`/`Hac`/`Cluster`）で分岐する。
- `(X'X)⁻¹`の計算をclassicalと共通化（`xtx_inverse`関数）。HC0-3は`(X'X)⁻¹Ψ̂(X'X)⁻¹`の対角成分の平方根。`Ψ̂ = Σ w_i ε̂_i² x_i x_i'`は、各行を`sqrt(w_i)*ε̂_i`でスケーリングした行列`Xw`を使い`Ψ̂ = Xw'Xw`として計算する（外積を手動で積み上げるより既存の行列積を再利用でき簡潔なため）。
- レバレッジ`h_ii = x_i'(X'X)⁻¹x_i`はHC2/HC3でのみ必要（`n×n`の帽子行列は作らず、`X(X'X)⁻¹`の行ごとの内積で計算）。
- HC2/HC3は`h_ii`が1に極めて近い観測（単一の観測だけを識別するダミー変数と切片の組み合わせ等、退化したデザイン）があると発散しうる。ただしこれはHC2/HC3という統計手法自体の数学的性質であり、statsmodels等の参照実装も同様に無防備なため、engine固有のバグとは言えない（対応の要否は未定）。

**重要な発見（statsmodelsの`use_t`既定値）**: statsmodelsは`cov_type`が`"nonrobust"`（classical）以外（HC0-3・cluster・HAC）の場合、`use_t=False`が既定で、p値・信頼区間に**正規分布**を使う（`use_t=True`にしない限りt分布にならない）。本プロジェクトは`cov_type`によらずt分布で統一する方針（`ols-api-design.md`「検定分布」）のため、`benchmark/run_statsmodels_benchmark.py`は`use_t=True`を明示的に指定している。

### HAC（Newey-West）

- `CovType::Hac { lags: Option<i64>, time_order: Option<Vec<f64>> }`をフィールド付きバリアントとして持つ。他のバリアント（`Classical`/`Hc0`〜`Hc3`）と違い、HACだけがラグ数・時間順序という追加パラメータを持つため。`OlsEstimator::fit`のシグネチャに`hac_lags`/`time_order`を常に生える引数として追加すると、HAC以外の`cov_type`では常に無意味な引数になってしまうため、`CovType`自身にデータを持たせる設計にしている。`Vec<f64>`を持つため`Copy`は付与できず`Clone`のみ。
- `(X'X)⁻¹Ŝ(X'X)⁻¹`の対角成分の平方根として計算する（`hac_cov_params`関数）。`Ŝ = Ŝ₀ + Σ_{l=1}^{L} w_l(Ŝ_l + Ŝ_l')`（Bartlett重み`w_l = 1 - l/(L+1)`）。残差でスケールした行列`Xe`（`Xe[t,a] = ε̂_t・x_t[a]`、時系列順）を使い、`Ŝ₀ = Xe'Xe`、`Ŝ_l = Xe[l:,:]'Xe[:n-l,:]`という`faer`の行列積で計算する（`Ŝ_l'`は転置を取るだけで再計算不要）。
  - ラグごとの行列積は`faer::linalg::matmul::matmul`を`Par::Seq`明示指定で呼ぶ。**理由**: 1回あたりの行列積は`k×k`という小さい出力サイズのため、faer既定の並列実行（グローバルスレッドプールへのディスパッチ）のオーバーヘッドが計算本体を上回り、素朴に行列積へ置き換えるだけだと手書きループより遅くなることを実測した（n=10,000, k=2で0.13倍＝約6倍の悪化）。`Par::Seq`をこの関数内だけにスコープを切ることで、他のcov_type計算・将来手法のグローバル並列化設定に影響を与えずにこの罠を回避している。書き換え後はk=2〜20の全域で3〜22倍高速化した（エンドツーエンド計測とマイクロベンチマークで倍率が異なる。詳細は11章「パフォーマンス」参照）。
- ラグ数の解決（`resolve_hac_lags`）: `Some(l)`なら`0 <= l < n`を検証（`LeastSquaresError::InvalidHacLags`）。`None`なら経験則`L = floor(4*(n/100)^(2/9))`で自動計算（`ols-standard-errors.md`3.2節）。
- 時間順序の解決（`time_ordering`）: `time_order`（`OlsInput`の行と対応する長さnの配列）が`Some`なら昇順ソートした行インデックス列を返し、`None`なら恒等順序（`OlsInput`の行順をそのまま時系列順とみなす）。実際の`X`/残差の並べ替え自体は`hac_cov_params`内でこのインデックス列を使って行う（`OlsInput`自体は並べ替えない。理由: `OlsInput`は`fit`の全`cov_type`で共有されるため、ここで恒久的に行を並べ替えるとPython側に返す残差配列の行と元のDataFrameの行の対応が崩れてしまう）。
- `partial_cmp().unwrap()`（時間順のソート）は、NaN/無限大が含まれないことが`engine_pybind`側の列抽出（`column_extraction::extract_f64_column`）で既に保証されている前提でパニックしない。
- statsmodelsとの数値照合は`sm.OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": L}, use_t=True)`。`use_correction`は明示していない（既定の`False`＝小標本補正なしで、本実装の式と一致することを確認済み）。

### クラスター標準誤差

- `CovType::Cluster { groups: Option<Vec<String>> }`。`groups`が`None`なら`fit()`が`CommonError::MissingClusterColumn`を返す（`CovType::Hac`の`lags: Option<i64>`と同じ設計パターン）。
- `cluster_col`で指定する列は`i64`固定にしない。州名・企業ID等、実務では文字列/カテゴリカルなクラスター変数の方が多いため、内部では`Vec<String>`として扱う（`BTreeMap<&str, Vec<usize>>`でグループ化）。
- **グループ化に`HashMap`ではなく`BTreeMap`を使う（統合PRでCI発覚、非決定性バグの修正）**: `cluster_cov_params`は`Ŝ = Σ_g S_g S_g'`をグループ順に加算するが、`HashMap`は反復順序がプロセスごとのランダムなハッシュシードに依存し非決定的（同一プロセス内でも`HashMap::new()`のたびに異なるキーを使うため、同じ入力に対する`fit()`の2回の呼び出し同士でも順序が変わりうる）。浮動小数点加算は結合則が成り立たないため、順序が変わると最終的な標準誤差が1 ULP程度ぶれる。`test_wls.py::test_weight_one_matches_ols[cluster]`（`OLS(...).fit()`と`WLS(...).fit()`という独立な2回の`fit()`呼び出しの結果をexact `==`で比較するテスト）がCI（Python 3.13/3.14ジョブ、3.12では非再現）で断続的に失敗し発覚した。`BTreeMap`（クラスター名の辞書順）に変更し、`fit_cluster_std_errors_are_deterministic_across_repeated_fits`（同一入力で`fit()`を21回呼びビット単位で一致することを検証）で固定した。
- `cluster_cov_params`関数: `Ŝ = Σ_g S_g S_g'`（`S_g = Σ_{i∈g} ε̂_i x_i`、クラスター内の観測を先に合計してから外積を取ることでクラスター内相関を許容する）。
- クラスター数`G`の検証（`validate_cluster_groups`関数）: `G < 2`なら`CommonError::InsufficientClusters`。`groups.len() != n`は`engine_pybind`側の実装バグでしか起こらない内部契約として`debug_assert_eq!`で検証。
  - **`engine::validation::validate_cluster_groups`への共有化**: Logit（nonlinear系統）のクラスターロバストSE実装時に、この検証ロジックがLogit側でも文言まで完全に同一で必要になったため、`engine/src/validation.rs`（新設。`engine::linear_algebra`と同じ位置付けの、系統をまたぐ純粋な入力バリデーションユーティリティ）に切り出した。`ols.rs`側はこの共有関数を呼ぶだけに変更し、挙動・受け入れ条件（`G<2`で`InsufficientClusters`）は変わらない。
- **小標本補正（`G/(G-1) * (n-1)/(n-k)`）は常に適用し、無効化するオプションは設けない**（`OLSOptions`に対応するフィールドを追加しない）。statsmodelsのソース（`statsmodels.stats.sandwich_covariance.cov_cluster`）を確認し、`use_correction=True`がデフォルトで`ols-standard-errors.md`5章の式と完全に一致することを確認済み。
- **自由度の切り替え**: statsmodelsは`cov_type="cluster"`のとき、デフォルト（`df_correction=True`）でt検定・信頼区間・F検定の自由度を`n-k`ではなく**`G-1`（クラスター数-1）に切り替える**（計量経済学の標準的な慣行、Cameron-Miller等）。標準誤差自体の値は変わらないが、p値・信頼区間・F検定のp値が大きく変わる（クラスター数が小さいとき特に顕著）。本実装も`cov_type=Cluster`のときのみ自由度を`G-1`に切り替える（他のcov_typeは引き続き`n-k`）。`fit()`内で`(cov_params, df_inference)`のタプルを`cov_type`ごとのmatchから返す設計にしている。`df_resid`自体（`σ̂²`・調整済みR²・AIC/BIC等で使う）は影響を受けず、常に`n-k`のまま。
- **G≤qの境界**: クラスターロバスト共分散`Ŝ = Σ_g S_g S_g'`はG個のランク1行列（外積）の和のため、`rank(Ŝ) ≤ G`。`wald_f_test`（4章）が使う傾き係数の部分行列（`q × q`、`q`は傾き係数の数）はG < qのとき構造的に特異になる（浮動小数点丸めではなく数学的な必然）。係数・標準誤差自体は`Ŝ`全体の対角成分から計算されるため問題なく求まるが、F検定の共分散部分行列でこの特異性が検出され`fit()`全体が`CommonError::ComputationFailed`になる（検出の仕組み自体は4章`ensure_well_conditioned_symmetric_matrix`参照）。「G=2ちょうどの成功パス」を検証する場合は、q（傾き係数の数）をG以下に保つ必要がある（`tests/api_tests/test_ols_fixtures.py::test_cluster_g2_matches_statsmodels`はq=1で検証、`test_cluster_g2_with_multiple_slopes_raises_computation_error`はq=3でComputationErrorになることを確認）。

### 信頼区間

- `OLSOptions.confidence_level: f64`（デフォルト`0.95`）。`alpha`という名前は使わない（統計慣習上「有意水準」を指すため紛らわしい）。
- 内部で`alpha = 1.0 - confidence_level`を計算し、`fit()`実行時に一度だけCIを計算して結果に固定して含める。実行時可変引数（例: `conf_int(alpha=...)`）は提供しない。

## 4. 適合度統計量

- `OlsInput`は`has_intercept: bool`フィールドを持つ（`from_columns`の`include_intercept`引数をそのまま保持）。R²・調整済みR²のcentered/uncentered TSS切り替え、F検定の自由度（`k_constant`）の判定に使う。
- **`(X'X)⁻¹`ベースの標準誤差計算関数は「k×kの分散共分散行列（cov_params）を返す」設計**（`classical_cov_params`・`hc_cov_params`・`hac_cov_params`・`cluster_cov_params`）。ロバストWald検定（後述）に完全な共分散行列が必要なため。`std_errors`は`fit()`側で対角成分の平方根を取って求める。**`cov_params`自体はPython側に公開しない**（`ols-api-design.md`5章の「Rust/engine_pybind側の責務は配列＋名前リストを返すところまで」に`cov_params`は含まれないため。`fit()`内のローカル変数として使い切る）。
- **R²・調整済みR²**: `include_intercept=true`ならcentered TSS（`Σ(y_i-ȳ)²`）、`false`ならuncentered TSS（`Σy_i²`）を使う（statsmodelsの`k_constant`による分岐と一致）。調整済みR²は`1 - ((n-k_constant)/df_resid)*(1-R²)`という一般形（`k_constant`を明示的に使うことで、切片あり/なし両方に同じ式で対応する）。
- **対数尤度・AIC・BIC**: `llf = -(n/2)*(ln(2π) + ln(SSR/n) + 1)`（分散は最尤推定量`SSR/n`であり、classical標準誤差の不偏推定量`SSR/(n-k)`とは異なる点に注意）。`aic = -2*llf + 2*k`、`bic = -2*llf + ln(n)*k`（`k`は切片を含む全パラメータ数）。
- **F統計量**: `cov_type`によらず単一の式`F = (β_slopes' Σ⁻¹ β_slopes) / q`（`Σ`は`cov_params`のうち切片以外の係数に対応する部分行列、`q`はその次元＝`k - k_constant`）で計算する（`wald_f_test`関数）。この式は`cov_type=Classical`のとき代数的に古典的F検定`((SST-SSR)/q)/(SSR/df_resid)`と完全に一致するため、分岐を分けていない。HC0-3・HAC・clusterでは`cov_params`がロバストな分散共分散行列になるため、この式がそのままロバストWald検定になる。p値はF分布（自由度`(q, df_inference)`、statrsの`FisherSnedecor`）の上側確率。
  - `q=0`（説明変数が定数項のみ）の場合はstatsmodels同様`f64::NAN`を返す（0除算回避）。
  - `Σ`の逆行列はCholesky分解（`Llt`）で求める。正定値行列の主小行列は必ず正定値という定理により理論上失敗しないはずだが、`xtx_inverse`と同様`ComputationFailed`に変換する境界ケース対応をしている。
  - **`Σ`が数値的にほぼ特異な場合の検出**: 変数間のスケールが極端に異なる設計行列（例: ある説明変数が1e6オーダー、別の説明変数が1e-3オーダー）では、`Σ`の条件数がスケール比の2乗（≈1e18）相当になり倍精度の限界を超えるが、Cholesky分解自体は（非ピボットのため）失敗せず数値的に無意味なF統計量を返してしまう（実測でstatsmodelsとの相対誤差5e10程度、`SingularMatrix`検出に使う`ensure_full_rank`と同じ発想をCholeskyのL因子対角成分に適用しても検出できないことも実測確認済み）。`ensure_well_conditioned_symmetric_matrix`関数（`crate::linear_algebra`に汎化済み、旧名`ensure_well_conditioned_cov_submatrix`。nonlinear系統の`observed_information_cov_params`/`opg_cov_params`が同じ非ピボットCholeskyの限界を抱えていたため系統をまたいで共有している）が`SelfAdjointEigen`（faerの対称行列固有値分解）で実際の固有値を求め、最大固有値との相対比（`q * f64::EPSILON * max_abs_eigenvalue`）で判定し、Cholesky分解の前に`ComputationFailed`で止める。この経路は理論上到達不能ではなく実際に到達する（下記「5. engine単体テストのカバレッジ」参照）。

## 5. engine単体テストのカバレッジ

`cargo-llvm-cov`で実測。100%は目指さず、理論上到達不能な防御的エラーパスはドキュメント化して受け入れる方針（`.claude/rules/rust-style.md`「テスト」参照）。

実測結果（25テスト時点）: Region 98.94%・Line 98.88%・Function 96.34%。未カバー箇所は以下の2種類。

- **理論上到達不能な防御的エラーパス**（受け入れて未カバーのまま。いずれも「事前に検証済みの不変条件により、理論上失敗し得ないはずだが、浮動小数点の丸めに備えて`Result`化してある」という同じ性質）:
  1. `xtx_inverse`の`Llt`失敗→`SingularMatrix`（特異性は`ensure_full_rank`で先に検出済みのため通常到達しない）
  2. `StudentsT::new`/`FisherSnedecor::new`の失敗→`ComputationFailed`（自由度は`n>k`・`G>=2`・`df_model>=1`の事前検証により常に有効な正の値になる）
  3. `wald_f_test`内の`Llt`失敗→`ComputationFailed`（正定値行列の主小行列は必ず正定値という線形代数の定理により理論上到達不能）
  - これらを実際に踏ませるには丸め誤差でギリギリ破綻する敵対的な浮動小数点データを人為的に作る必要があり、プラットフォーム依存で壊れやすく、実装の振る舞いというより浮動小数点ノイズの検証になるため見送っている。`cargo-llvm-cov`の除外マーカーも導入せず、コード側のdocコメントで理由を説明する方針にしている。
  - **理論上到達不能ではなく実際に到達するケース（上記に追加）**: `ensure_well_conditioned_symmetric_matrix`の`ComputationFailed`（`Σ`の数値的なほぼ特異性）は、上記3件とは異なり実データで実際に到達する経路であり、`fit_returns_computation_failed_for_extreme_scale_difference_in_f_test`テストでカバー済み。
- 「missed lines」表示の一部は`assert!`マクロのメッセージ引数（アサーション失敗時のみ評価される）による分析ツールの誤検知で、実際のギャップではない。

その後のテストレビュー（8章参照）で境界値・内部整合性テストを4件追加し、現在は29テスト。

## 6. engine_pybind実装

### Arrowゼロコピーデータ受け渡し

`polars DataFrame → faer::Mat<f64>`の変換は2段階（`.claude/rules/rust-style.md`「Python境界でのデータ受け渡し」参照）:
1. `engine_pybind`: polars DataFrameから列ごとに`Vec<f64>`へ抽出する（`column_extraction::extract_f64_column`）。
2. `engine`: 抽出済みの列から`faer::Mat`を組み立てる（`OlsInput::from_columns`）。

「Arrow経由のゼロコピー」は、Python→Rust境界（`pyo3-polars`の`PyDataFrame`）の受け渡し自体を指す。polars Seriesから`faer::Mat`への詰め替え自体は2回のコピーを許容する設計のまま（想定データ規模ではQR分解本体のコストに対して無視できるため）。

### polars（Rustクレート）とpyo3-polarsのバージョン選定

**重要な発見**: Rustの`polars`クレート（crates.io公開）とPython側`polars`パッケージ（PyPI）は既にバージョン体系が分離している（2026年時点のpolarsモノレポで、`py-polars/pyproject.toml`は`1.4x`系、内部Rustワークスペースクレートは`[workspace.package] version = "0.54.4"`）。したがってPython側`pyproject.toml`の`polars`バージョンとRust側`Cargo.toml`の`polars`バージョンを数字で合わせる必要はない。

実際の互換性は数字ではなく、`pyo3-polars`の`PyDataFrame`/`PySeries`変換が使う`polars_ffi::version_0`という安定版FFIプロトコル（Arrow C Data Interfaceに近い、バージョン非依存のインターフェース）によって担保される。

最終的な組み合わせ: `pyo3=0.28.2`, `polars=0.54.4`, `pyo3-polars=0.27.0`（すべて`=`で完全固定、`engine_pybind/Cargo.toml`）。**`pyo3-polars=0.27.0`は`pyo3="^0.28"`を要求する**ため、`pyo3`は`0.28.2`に固定している（`0.28.0`/`0.28.1`はyanked済みのため除外）。`pyo3-polars`は`pola-rs/pyo3-polars`が2025年7月にアーカイブ済みでpolars本体リポジトリに統合されており、crates.io公開版と本体リポジトリ内のバージョンにズレが生じうる既知のリスクがある（`.claude/rules/rust-style.md`「既知のリスク」参照）。`pyo3`を上げる場合は対応する`pyo3-polars`の新版公開を待つ必要がある（現状の上流待ち状況は12章参照）。

### 実装時に踏んだpolars 0.54.4特有の差異

- `ChunkedArray::rechunk()`が`Cow<'_, ChunkedArray<T>>`を返す（既に単一チャンクの場合の不要なcloneを避けるため）。`Cow`は`IntoIterator`を実装しないため、`.into_iter()`ではなく`ChunkedArray::iter()`（`Cow`はDerefで透過的に呼べる）を使う。
- pyo3 0.28では`PyObject`という型エイリアス（`Py<PyAny>`の別名）がpreludeから削除されている。`Py<PyAny>`を直接使う。
- pyo3 0.28以降、`Clone`を実装する`#[pyclass]`の`FromPyObject`自動導出はopt-inになった。Python側から`OLSOptions`インスタンスを引数として受け取るには`#[pyclass(from_py_object)]`を明示する必要がある。

### engine呼び出し・エラー変換

- `OLSResult`（`#[pyclass(get_all, skip_from_py_object)]`）: `params`/`std_errors`/`t_stats`/`p_values`/`conf_lower`/`conf_upper`/`param_names`/`residuals`/`dep_var_name`/`nobs`/`cov_type`（実際に使われた種別を小文字文字列でecho）/`r_squared`/`r_squared_adj`/`f_statistic`/`f_p_value`/`log_likelihood`/`aic`/`bic`を公開。`conf_int`は`conf_lower`/`conf_upper`の2配列に分割している（engineの内部表現・実装の簡潔さを優先）。`skip_from_py_object`なのは、`OLSResult`がRust側で組み立ててPythonに返すだけの型で、Python側から構築されることを想定しないため（`OLSOptions`の`from_py_object`とは対照的）。
- `engine::linear::common::LeastSquaresError` → `PyErr`の変換は、`impl From<LeastSquaresError> for PyErr`ではなく関数`fn least_squares_error_to_pyerr(err: LeastSquaresError) -> PyErr`として実装する。`LeastSquaresError`（`engine`クレート）・`PyErr`（`pyo3`クレート）のどちらもこのクレート外で定義された型のため、`impl From`はorphan ruleに抵触する。呼び出し側で`.map_err(least_squares_error_to_pyerr)?`する。
- `cov_type`固有の追加列（`cluster_col`/`time_col`）の抽出は、該当する`cov_type`のときのみ行う（`cov_type != "hac"`のとき`time_col`は無視、`cov_type != "cluster"`のとき`cluster_col`は無視）。誤って無関係な列を要求してエラーになることを避ける。

## 7. python_package実装

- `python_package/econometricsmodels/linear/ols.py`（`engine`/`engine_pybind`と同じ`linear/`系統ディレクトリ構成をpython_package側にも揃えている）。
- `OLS`クラス（`data`/`y`/`x`/`options: OLSOptions | None`のオブジェクト渡し）と`OlsResults`クラス（`_lib.OLSResult`の薄いラッパー）。`OLSOptions`自体は独自定義せず`_lib`から再輸出する（`ols-api-design.md`3章の既存方針通り）。
- `OlsResults`のプロパティ設計:
  - `params`/`std_errors`/`t_stats`/`p_values`は`dict[str, float]`（係数名→値。O(1)取り出し用）。
  - `conf_int`は`dict[str, tuple[float, float]]`（`_lib.OLSResult`の`conf_lower`/`conf_upper`2配列から組み立て）。
  - `coef_table()`メソッドで行指向の`list[dict]`（キー: `param`/`coef`/`std_err`/`t_stat`/`p_value`/`conf_lower`/`conf_upper`）を提供。REST APIレスポンスにそのまま使える形。
  - `residuals`はそのまま`list[float]`（`_lib.OLSResult`の値を素通しするだけの薄いラッパーという位置づけを優先し、polars Seriesへの変換等は行わない）。
- **`summary()`・`to_frame()`/`conf_int()`のDataFrame版は実装しない**（`ols-api-design.md`5章の確定方針、「薄いラッパー」というスコープに照らして見送っている）。**`predict()`は例外的にスコープに含める**（`new_data=None`で学習データの予測値、指定時は新規データの予測値を返す1メソッドに統一。Issue #86、設計は`ols-api-design.md`7章参照）。

## 8. テスト

### テストデータ

- **合成データセット**（`benchmark/generate_synthetic_datasets.py`の7シナリオ）: `testing-policy.md`が要求する全バリエーション（小標本`small_n`、高分散`high_variance`、不均一分散`heteroskedastic`、自己相関`autocorrelated`、中程度の多重共線性`moderate_multicollinearity`、完全な多重共線性`perfect_multicollinearity`）に`baseline`を加えた7種で網羅している。
- **実データセット**: `benchmark/load_wooldridge.py`の`wage1`（n=526、`lwage ~ educ + exper + tenure`）・`gpa2`（n=4137、`colgpa ~ sat + hsperc + tothrs`）を使用。いずれも欠損値なし。列の型はInt64/Float64混在だが`column_extraction::extract_f64_column`が`cast(&DataType::Float64)`で吸収する。実データは真の係数と比較できないため、リファレンス実装との一致のみで検証する（`testing-policy.md`）。

### リファレンス実装によるクロスチェックの役割分担

- **statsmodels**（`benchmark/run_statsmodels_benchmark.py`）: 主リファレンス。classical/HC0-3/cluster/HAC、AIC/BIC/log-likelihood、ロバストWald検定まで一貫して対応。
- **R（`lm` + `sandwich`/`lmtest`パッケージ）**: 独立実装によるクロスチェック。`benchmark/run_r_benchmark.R`の`lm`分岐で、classical/HC0-3/cluster/HACを確認する。`read.csv()`はデフォルトで列名を`make.names()`により書き換える（例: `_group`→`X_group`）ため、クラスター列等を渡す場合は`check.names = FALSE`を指定する必要がある。
- **pyfixest**: OLSの正確性検証には使わない。性能比較専用（`docs/planning/specs/ols-performance-notes.md`参照）。
  - **pyfixestのHC2/HC3に関する既知の差異**: fixest（R）本体のソース（`vcov_hc2_hc3_internal()`）を確認したところ、HC2/HC3にはssc（`n/(n-k)`の小標本補正）を一切適用しない設計だった（適用されるのはHC0/HC1のみ）。一方pyfixest（Python、v0.60.0タグで確認）はHC1/HC2/HC3を同一分岐で扱っており、HC1用の`N/(N-k)`補正をHC2/HC3にも誤って適用している（`sqrt(N/(N-k))`がSEに掛かり、nが小さいほど乖離が拡大する。`small_n`シナリオ n=20, k=4で約11.8%）。**fixestの仕様ではなくpyfixest自身の実装バグ**であり、この理由でOLSの正確性検証から除外している（upstreamへの報告は見送り）。固定効果が絡むPhase4（FE/RE）以降での採否はその時点で個別に判断する。

### 許容誤差

- classical/HC0-3/cluster: Rとの実測で最大でも相対誤差**1e-14程度**（機械精度レベル）。`RTOL_STRICT = 1e-8`（`testing-policy.md`の基本方針）を採用。
- HAC: Rとの実測で相対誤差**0.40%程度**（`prewhite`/`adjust`オプションの慣習差が原因）。`RTOL_HAC = 1e-2`を使う。係数（`coef`）比較は`RTOL_STRICT`のまま（cov_typeはSE計算方法にのみ影響し、係数自体はcov_typeに依存しないため）。

### `tests/api_tests/`の構成と役割分担

- `test_ols.py`: 構造・API・エラーパスの検証（statsmodelsとの部分的な数値比較も含む）。
- `test_ols_fixtures.py`: `tests/api_tests/fixtures/benchmarks/ols.json`（statsmodels主リファレンス）を読み込み、合成データ6シナリオ×classical/HC0-3/HAC + baselineのみclusterで、係数・標準誤差・検定統計量・適合度統計量を相対誤差1e-8で厳密比較する。HACはフィクスチャ生成時と同じ`hac_lags=1`を明示指定し、自動ラグ選択式の違いを比較対象から除外している。p値等0に近い極小値向けに絶対誤差フロア`ATOL=1e-10`を`max(RTOL*|ref|, ATOL)`の形で組み合わせている。
- `test_ols_crosscheck.py`: `tests/api_tests/fixtures/benchmarks/ols_crosscheck.json`（R）を読み込み、独立実装との一致を確認する。フィクスチャ生成スクリプト（`benchmark/fixtures/generate_ols_crosscheck_fixtures.py`）と生成物は別管理（`.claude/rules/testing-policy.md`「ベンチマーク値のフィクスチャ化」参照）。

### テストレビューでの追加

OLS関連の全テスト（Rust側`#[cfg(test)]`、Python側`tests/api_tests/`の3ファイル）を有効性・網羅性・ロジック検証としての妥当性の観点でレビューした。既存テストに実際の誤り（バグ）は見つからなかった。

- **Rust側**: `hac_lags`の上限側境界（`lags=n`が範囲外、`n-1`が最大許容値）、`confidence_level`の厳密な境界（0.0・1.0ちょうど）、`CovType::Hac { lags: Some(0), .. }`と`HC0`が数学的に同一の式になるという内部整合性、の3点が未検証だったため追加した（25件→29件）。
- **Python側**: `engine_pybind`の`OLSOptions`と突き合わせたところ、cov_type以外のオプション（`include_intercept`・`confidence_level`・`hac_lags=None`・`time_col`）や複数の`ValidationError`経路がPython API境界からほとんど検証されていなかった。特に**`time_col`（HACの時系列順序指定）は3ファイルのどこからも一度もテストされていなかった**（engine内部のロジックはRust単体テストで検証済みだったが、`engine_pybind`の列抽出経路が未検証だった）。`test_ols.py`に19件追加した（44件→63件）: エラーパス8件（観測数不足・クラスター数不足・confidence_level境界・hac_lags境界・y/x重複・const衝突・空x）、オプション反映確認4件（`include_intercept=False`・confidence_levelによる信頼区間幅の変化・hac_lags自動計算・time_col指定時の時系列順序復元）。
- **参考（対応不要と判断）**: `engine_pybind::fit`の「行数がyと一致しない」という`ValidationError`分岐が3箇所ある（`y`列・各`x`列・`cluster_col`・`time_col`それぞれ）。これらは全て同一のpolars DataFrameから抽出しており、polars DataFrameは構造上すべての列が同じ行数を持つため、実質的に到達不能（dead code）である可能性が高い（1章の`DimensionMismatch`と同種）。対応（削除するか、到達不能である理由をコメントで明記するか）はengine_pybind側の設計判断として未着手のまま残している。

## 9. ドキュメント

`docs/mkdocs.yml`をmkdocs-materialテーマで構築。

- **`docs_dir: .`**: `mkdocs.yml`自体を`docs/`直下に置く構成（CLAUDE.md 3章）のため、デフォルトの`docs/mkdocs.yml`相対`docs/`（実質`docs/docs/`）ではなく、`docs/`自身をdocs_dirとして明示指定している。これにより`docs/planning/`・`docs/spec/`もビルド対象に含まれるが、`nav`には載せていない（CLAUDE.md 10章「ソースとしては誰でも閲覧可能という前提で運用する」）。
- **APIリファレンスはmkdocstringsで自動生成**: `docs/api/ols.md`は`::: econometricsmodels.OLS`等のディレクティブのみで構成し、docstringから自動生成する（二重管理を避けるため）。
- **PyO3コンパイル済みクラスの`__module__`に注意**: `#[pyclass]`はデフォルトで`__module__ == "builtins"`になり、mkdocstrings（griffe）が再エクスポートのalias連鎖を解決できず`AliasResolutionError`でビルドが失敗する。`#[pyclass(module = "econometricsmodels._lib")]`を明示することで解消する（`.claude/rules/rust-style.md`「言語方針」に一般化済みのルール）。
- **mkdocstringsの`filters`設定**: デフォルトでは`__new__`/`__repr__`等のdunderメソッドがPyO3クラスの動的検査で「メンバー」として拾われノイズになるため、`filters: ["!^_"]`で除外している。
- **ページ構成**: `index.md`（トップ）・`getting-started.md`（インストール・使用例・エラーハンドリング）・`api/ols.md`（APIリファレンス）。
- `site/`（ビルド出力）は`.gitignore`対象。GitHub Pagesへの自動デプロイ（`cd_docs.yml`）は別途対応する。

## 10. CI/CD

- **`ci_engine.yml`**（`engine`＝純粋Rustの継続的な品質検証）:
  - トリガー: `engine/**`・`Cargo.toml`・`Cargo.lock`・ワークフローファイル自体。
  - `test`ジョブ: `cargo fmt -p engine --check` → `cargo clippy -p engine --all-targets -- -D warnings` → `cargo test -p engine`。`engine_pybind`はclippy/fmt対象外（`engine`はPyO3非依存の純粋Rustという責務分離を維持し、Python/PyO3環境を一切必要とせず完結させるため。`engine_pybind`側は`ci_python.yml`が担当）。
  - `audit`ジョブ: workspace全体の`Cargo.lock`（`Cargo.lock`はworkspaceで1つのため分割できない）を`taiki-e/install-action`でインストールした`cargo-audit`で検証する。`rustsec/audit-check`アクションは不採用（`cargo audit --json`の出力を`JSON.parse()`しており、出力にANSI制御文字等が混ざると失敗する既知の不具合が長期未解決のため。テキスト出力のまま`cargo audit`を直接実行する）。既知の脆弱性の扱いは12章参照。
  - ツールチェーンセットアップは`actions-rust-lang/setup-rust-toolchain`（キャッシュ内蔵、PRへのインライン警告注釈）を使用。全アクションをコミットSHAで固定（サプライチェーン攻撃対策）。
- **`ci_python.yml`**（`python_package`/`engine_pybind`の継続的な品質検証、3ジョブ）:
  - `test`ジョブ（Python 3.12/3.13/3.14マトリクス）: `uv sync --locked --group test` → `uv run maturin develop` → `pytest tests/api_tests` → `ruff check .` → `ruff format --check .`。`--locked`（`pyproject.toml`と`uv.lock`の不整合を検知）を`--frozen`より優先。`engine_pybind`はabi3を使っていないためPythonマイナーバージョンごとに別ビルドが必要（マトリクス化必須）。
  - `engine_pybind-lint`ジョブ: `cargo fmt -p engine_pybind --check` → `cargo clippy -p engine_pybind --all-targets -- -D warnings`。
  - `pip-audit`ジョブ（Python 3.12固定）: `test`グループのみを対象（ワークフローが実際にインストールする依存と一致させる）。
  - トリガー: `python_package/**`・`engine_pybind/**`・`pyproject.toml`・`uv.lock`・`tests/api_tests/**`・ワークフローファイル自体。`engine/**`は含めない（`engine`単体の変更は`ci_engine.yml`側でカバーされる前提）。
- **`cd_release.yml`**（Linux/macOS/Windows向けwheelビルドの継続的な成功確認）:
  - 0.1.0のプレリリース段階のため、PyPI公開ステップは含めない（ビルドとartifactアップロードまで）。
  - `maturin generate-ci github`の出力を土台に、CLAUDE.md 12章のサポート対象（Linux x86_64・Windows x64・macOS両アーキ）に絞り込んだ。
  - ビルド対象Pythonバージョンは`-i python3.12 -i python3.13 -i python3.14`を明示指定（`--find-interpreter`は未サポートバージョンまで検出してしまうため不採用）。
  - トリガー: タグpush（`v*`）+ `workflow_dispatch`のみ。マルチOSビルドはCI時間コストが高いため、PR毎には回さない。
  - 将来のPyPI公開ステップは本ワークフローに`release`ジョブとして追加する方針。
- **`dependabot.yml`**（`cargo`・`uv`・`github-actions`の3エコシステム）:
  - `"pip"`ではなく**`"uv"`エコシステム**を採用（uv専用の`package-ecosystem`。`dependabot-core`にuv専用gemが存在し、`test`/`benchmark`/`dev`/`docs`全依存グループが更新対象になる）。
  - cooldownは`default-days: 10`のみ指定（全エコシステム共通の設定キーのため）。セキュリティアップデートPRはGitHub側の既定で常にcooldown対象外（設定不要）。
  - `cargo audit`/`pip-audit`（CI実行時点のロックファイル検証）とDependabot（レジストリの継続的な監視・PR自動生成）は補完関係にあり、どちらかへの統合・置き換えは行わない。
- **`benchmark_ols.yml`**（`benchmark/compare_performance.py`の定期実行、11章参照）:
  - トリガー: タグpush（`v*`）+ `workflow_dispatch`のみ。フルスイープが数分かかり重いこと、ソロ開発でコード変更頻度が低いことから、毎PR/push・週次スケジュールは見送った。
  - 結果整形は`benchmark/render_performance_summary.py`として分離（`compare_performance.py`自体はJSON出力に専念する既存方針を維持）。`>> "$GITHUB_STEP_SUMMARY"`でjob summaryに出力する。
  - 依存グループは`test`＋`benchmark`＋`dev`の3つ（pyfixestとの比較・release buildが必要なため）。
  - 結果JSONは`actions/upload-artifact`でアーティファクト保存する。パフォーマンスregressionの自動検知・アラートはスコープ外（将来のstretch）。

## 11. パフォーマンス

`benchmark/compare_performance.py`で、`OLS(...).fit()`全体（Python API呼び出し込み）の実行時間・メモリ使用量をstatsmodels/pyfixestと比較している。詳細な結果・考察は[`ols-performance-notes.md`](./ols-performance-notes.md)参照。

**最重要の教訓: engineは必ずreleaseビルド（`maturin develop --release`）で計測する**。デフォルトの`maturin develop`（debugビルド）のままだと最大140倍遅い誤った結果になる（`.so`サイズが924MB→32.7MBの差）。この罠を防ぐため、スクリプトに`.so`ファイルサイズからdebugビルドを検知する警告機構（`_warn_if_debug_build()`）を組み込んでいる。

releaseビルドでの正しい計測では、classical/HC1/clusterでengineがstatsmodels/pyfixestと同等以上に高速、HACも大規模（n=1,000,000）ではほぼ互角という結果になった。メモリはengineが一貫して最小（pyfixestの約1/3）。

### engineコードレビューでの発見

`engine/src/linear/ols.rs`をパフォーマンス・ロジックの両観点でレビューした際の発見（実測はすべて`cargo test -p engine --release`のスクラッチベンチマークで行った）。

- **`hac_cov_params`の三重ループ**: 3章「HAC」に記載の通り、faerの行列積への書き換えと`Par::Seq`のスコープ限定で対応済み。
- **`xtx_inverse`をQR分解の`R`因子から導出する案（探索したが不採用）**: `fit()`は既に`col_piv_qr()`でXのQR分解を計算しているが、`xtx_inverse`は`x.transpose() * x`で`X'X`を計算し直している。`R`因子から`(X'X)⁻¹`を導出すれば`X'X`の明示計算（O(n·k²)）を省けるのではと仮説を立てたが、実測で否定された。n=1,000,000, k=5で、現状（`x.transpose() * x` + Cholesky）が25.1ms、`col_piv_qr()`単体が103.2ms（faerのQR分解はピボット選択のため逐次依存が強く、単純な行列積ほど並列化・SIMD化の恩恵を受けにくいと考えられる）、QR再利用の限界費用が21.7ms——現状と大差なく、明確な高速化余地はない。
- **`cluster_cov_params`の二重ループ**: `hac_cov_params`と同型の手書きループだが、計算量がO(G·k²)（Gはクラスター数、通常nよりずっと小さい）でホットスポットではないため対応不要と判断。

## 12. セキュリティ

`cargo audit`が検知した既知の脆弱性・非メンテナンス依存について調査した結果、**4件すべて現時点では上流（`pyo3-polars`/`polars`）の新バージョン公開待ちで、コード側からは修復不能**と判明した。継続監視の対象として保持している。

- **`pyo3 0.28.2`（RUSTSEC-2026-0176/0177）**: `pyo3-polars`の最新公開版（`0.27.0`）が`pyo3 = "^0.28"`を要求する。`pyo3>=0.29.0`へ上げるには対応する`pyo3-polars`の新版公開を待つ必要がある。
- **`quick-xml 0.39.4`（RUSTSEC-2026-0194/0195、severity 7.5 high）**: 経路は`polars → polars-error → object_store(^0.13.1) → quick-xml(^0.39.0)`。修正版（`object_store 0.14.1`が要求する`quick-xml ^0.41.0`）に到達するには`polars`自体の新バージョンが必要（`polars-error`は`object_store`を`^0.13.1`にしか許容しない）。
- **`bincode`（unmaintained警告）**: 同じく`polars`（`polars-utils`の`serde`機能経由）待ち。
- **`paste`（unmaintained警告）**: `faer`待ちだが、`faer`は`0.24.4`（`.claude/rules/rust-style.md`で明示的に固定しているバージョン）が現状最新で新版なし。

**重要な副次的発見**: `quick-xml`/`bincode`は実際にはビルドに一切含まれていない。`cargo build`のログに該当crateのコンパイルが一度も出現せず、`cargo tree -p object_store`が空を返すことを確認した。これらは`polars-error`/`polars-utils`のオプション依存（クラウドストレージ機能・serde機能）であり、本プロジェクトはpolars DataFrameをローカルでしか扱わずどちらも有効化していない。一方`cargo audit`は機能フラグを考慮せず`Cargo.lock`を丸ごとスキャンするため、実際にコンパイルされない依存でも警告に含まれる。実運用上のリスクは実質ゼロだが、`polars`の`default-features`を絞っても`cargo audit`の検知からは除外できない（`Cargo.lock`はワークスペース全体で到達しうる全オプション依存のバージョンを、有効化の有無に関わらず一括して固定する設計のため）。

`.cargo/audit.toml`のignore listは、上記4件のRUSTSEC IDを上流待ちとして保持している（解消できたら削除する）。

## 未確定（実装時判断でよい）

- 特異性判定の相対閾値の具体式
- `SingularMatrix`のエラーメッセージを「定数項が重複している可能性があります」等、状況に応じて分岐させるか（優先度低、後回し可）
