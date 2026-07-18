# OLS 内部実装ノート（パラメータ設計以外）

`docs/planning/specs/`配下。OLSのAPI・オプション設計（Issue「OLS: API・オプション設計」「OLS: 標準誤差の技術仕様確定」）とは別に、**パラメータ以外の内部実装で決めたこと・まだ決まっていないこと**をまとめる。実装issue（engine関連）着手時に必ず参照すること。

## 確定事項

### `y`は`list[str]`ではなく`str`

- Phase1〜6（VAR等の一部時系列手法を除く）でyは常に1変数。`list[str]`だと「長さ1であること」を全推定関数が実行時検証する必要が生じるため、型で表現する
- **`plan.md` / `CLAUDE.md` 2章の例（`y=["y_col"]`）を`y="y_col"`に修正済み**
- 真に多変量なyが必要な手法（VAR等）が出てきた場合は、その手法だけ`y: list[str]`にする

### 推定量構造体はフィールドをprivateにする

- 詳細・理由は `.claude/rules/rust-style.md`「推定量構造体の設計（全手法共通）」を参照。OLSに限らず全手法共通のルールとして横展開済み

### 特異性判定は相対閾値

- 詳細は `.claude/rules/rust-style.md`「線形代数」を参照。絶対閾値（草案コードの`1e-10`固定）は不採用。具体的な閾値式は実装時に判断
- Issue #8で確定した具体式: `engine::linear::ols::ensure_full_rank`。`col_piv_qr`の`R`の対角成分（列ピボットにより絶対値降順）のうち最大値`|R[0,0]|`を基準に、`threshold = k * f64::EPSILON * |R[0,0]|`未満の対角成分があればランク落ちと判定する

### 係数の求め方: 列ピボットQR（Cholesky不採用）

- Issue #8で確定。`X'Xβ=X'y`をCholesky分解で解く方式は不採用。理由: `X'X`を明示的に作ると条件数が2乗になり数値的に不利な上、`col_piv_qr`なら特異性検出（上記）と係数計算を同じ分解で一度に行える
- `engine::linear::ols::OlsEstimator::fit(input: OlsInput) -> Result<Self, OlsError>`で実装。`qr.solve_lstsq(input.y())`で係数を得る（列ピボットの並べ替えは`solve_lstsq`内部で吸収され、返り値は元の列順のまま）
- `n <= k`（観測数不足）は`OlsEstimator::fit`側で検証する（`OlsInput`自体は`n<=k`でも構築できる。次の項目参照）

### `OlsInput::from_columns`は`Result<Self, OlsError>`を返す（Issue #8で変更）

- Issue #6時点では次元不一致を`debug_assert!`にしていたが、`OlsError`（Issue #7）確定後のIssue #8で`Result`化した。`debug_assert!`だと失敗時にRustのpanicとしてしか表面化せず、PyO3境界で`ValidationError`に変換できないため
- `y`と各`x_columns`の長さ不一致 → `OlsError::DimensionMismatch`（`Result`で返る、実データで起こりうるため）
- `x_names.len() != x_columns.len()` → 引き続き`debug_assert!`のまま（`engine_pybind`側の実装バグでしか起こらない内部契約であり、実データに起因する検証とは性質が異なるため区別）

### classical標準誤差・t統計量・p値・信頼区間（Issue #9で実装済み）

- `engine::linear::ols::OlsEstimator::fit(input: OlsInput, confidence_level: f64) -> Result<Self, OlsError>`。
  `confidence_level`の範囲検証（`(0, 1)`）もここで行う（`OlsError::InvalidConfidenceLevel`）
- classical標準誤差: `σ̂²(X'X)⁻¹`の対角成分の平方根（`σ̂² = SSR/(n-k)`）。`X'X`は対称正定値であることが
  既に特異性検出（Issue #8の`ensure_full_rank`）で保証されているため、`X'X`自体のCholesky分解（`Llt`）で
  逆行列を求める（`classical_std_errors`関数）。列ピボットQR分解を再利用する方式ではない
- t統計量: `params / std_errors`。p値: 両側検定、自由度`n-k`のt分布のCDFから計算（正規分布ではない、
  `ols-api-design.md`「検定分布」参照）
- 信頼区間: `confidence_level`から求めたt分布の臨界値`t_crit`を使い、`coef ± t_crit * std_err`
- t分布のCDF・逆CDFには**statrs**クレート（`=0.18.0`固定、`default-features = false`でnalgebra/rand機能を除外）を使用。engineへの追加はfaer/thiserror同様、ルートCargo.tomlの`[workspace.dependencies]`で`=`により完全固定
- **スコープ外（Issue #9時点）**: `cov_type`によるHC/cluster/HACへの分岐（Issue #10・#11）。R²・調整済みR²・F統計量・AIC/BIC・対数尤度等の**適合度統計量はIssue #21**（Issue #11の後に着手）で実装する

### HC0〜HC3ロバスト標準誤差、および`use_t`に関する重要な発見（Issue #10で実装済み）

- `engine::linear::ols::CovType`（`Classical`/`Hc0`/`Hc1`/`Hc2`/`Hc3`）を新設し、
  `OlsEstimator::fit(input, cov_type, confidence_level)`で分岐する。文字列パース
  （Python文字列 → `CovType`）は`engine_pybind`側の責務のまま（後続issueで配線）
- `(X'X)⁻¹`の計算をclassicalと共通化（`xtx_inverse`関数に分離）。HC0-3は
  `(X'X)⁻¹Ψ̂(X'X)⁻¹`の対角成分の平方根。`Ψ̂ = Σ w_i ε̂_i² x_i x_i'`は、各行を
  `sqrt(w_i)*ε̂_i`でスケーリングした行列`Xw`を使い`Ψ̂ = Xw'Xw`として計算する
  （外積を手動で積み上げるより既存の行列積を再利用でき簡潔なため）
- レバレッジ`h_ii = x_i'(X'X)⁻¹x_i`はHC2/HC3でのみ必要（`n×n`の帽子行列は作らず、
  `X(X'X)⁻¹`の行ごとの内積で計算）

**重要な発見（statsmodelsの`use_t`既定値）**: statsmodelsは`cov_type`が`"nonrobust"`
（classical）以外（HC0-3・cluster・HAC）の場合、`use_t=False`が既定で、p値・信頼区間に
**正規分布**を使う（`use_t=True`にしない限りt分布にならない）。本プロジェクトは
`cov_type`によらずt分布で統一する方針（`ols-api-design.md`「検定分布」）のため、
`benchmark/run_statsmodels_benchmark.py`に明示的に`use_t=True`を追加した
（Issue #10で発見・修正）。

**既存フィクスチャへの影響**: `tests/api_tests/fixtures/benchmarks/ols.json`は
`use_t=True`修正前に生成されたもので、HC0-3・cluster・HACの`t_stats`/`p_values`/
`conf_int`が正規分布ベースの値になっている可能性が高い（`coef`/`se`自体は`use_t`に
依存しないため影響なし）。**Issue #18/#19着手時に再生成が必要**。
- テストは`tests/engine_tests`ではなく`engine`クレート内の`#[cfg(test)]`（`cargo test -p engine`）に実装。既知の厳密解データに加え、scipy.stats.tで独立に検算した教科書的データセット（x=[1..5], y=[2,4,5,4,5]）で標準誤差・t値・p値・信頼区間を1e-6〜1e-9の許容誤差で検証

### Python側の例外はカテゴリ別に分ける

- 詳細・理由は `.claude/rules/rust-style.md`「エラーハンドリング」を参照（`ValidationError` / `ComputationError`の2階層、`pyo3::create_exception!`で定義）
- OLSでの`engine::linear::ols::OlsError`との対応（Issue #7で確定・実装済み）:

| `OlsError`のバリアント | Python例外 |
|---|---|
| `DimensionMismatch` | `ValidationError` |
| `InsufficientObservations` | `ValidationError` |
| `MissingClusterColumn` | `ValidationError` |
| `InvalidConfidenceLevel` | `ValidationError` |
| `InsufficientClusters` | `ValidationError` |
| `InvalidHacLags`（Issue #3） | `ValidationError` |
| `SingularMatrix` | `ComputationError` |
| `ComputationFailed`（分布計算等の失敗） | `ComputationError` |

**Issue #7での確定事項（スコープの絞り込み）**: 当初案の`InvalidInput`（欠損値・次元不一致を1バリアントにまとめたもの）は、Issue #7実装時に`DimensionMismatch`（次元不一致のみ）に改名した。欠損値検出は`engine`が受け取る時点で既に`&[f64]`（クリーン値のみ）である前提のため、`engine`側のエラーではなく`engine_pybind::column_extraction`側の責務。同じ理由で`InvalidTimeColumn`（`time_col`の数値キャスト失敗）も`engine`のエラー型には含めず、`engine_pybind`側でのみ扱う。

## エラー型（`engine::linear::ols::OlsError`、Issue #7で実装済み）

草案コードの`OlsError`には無かったが、今回の設計決定に伴い追加したバリアント。

- **`InvalidConfidenceLevel`**: `confidence_level`が`(0, 1)`の範囲外だった場合
- **`InsufficientClusters { g: usize }`**: クラスター数`g < 2`の場合。草案コードでは未検証で、実際に0除算からのNaN伝播でパニックすることを確認済み（`docs/planning/draft-reference/ols-draft-consolidated.md`参照）。`new()`の時点で検証し、`fit()`まで到達させない
- **`InvalidHacLags { hac_lags: i64, n: usize }`**（Issue #3）: `hac_lags`が負、または`n`以上の場合

`InvalidTimeColumn`（Issue #3で検討した、`time_col`のf64キャスト失敗）は`OlsError`には含めない。上記「Python側の例外はカテゴリ別に分ける」の注記通り、`engine_pybind::column_extraction`側のみで扱う。

`OlsInput::from_columns`（Issue #6）自体は次元不一致を`debug_assert!`でしか検証しない（呼び出し側との内部契約）。`DimensionMismatch`/`InsufficientObservations`等を実際に`Result`として返す検証は、`OlsEstimator`のコンストラクタ（未実装、正規方程式ソルバー実装issue）で行う。

## 実装時に見落としやすい点（要注意）

- **ロバストF検定への切り替え**: `cov_type`がHC系/clusterの場合、F検定（適合度統計量）もロバストWald検定に切り替える方針が決定済み。SE計算の実装issueとは別に、適合度統計量計算の実装issueでも見落とされないよう明記すること
- **`log_likelihood` / `AIC` / `BIC`の計算式**: 草案コードはここにバグがあった（対数尤度の分散に不偏推定量`SSR/(n-k)`を誤用。正しくは最尤推定量`SSR/n`。AIC/BIC式も`n·ln(2π)+n`の定数項が欠落）。詳細・検証手順は`docs/planning/draft-reference/ols-draft-consolidated.md`の「検証詳細」を参照。**この部分は草案をそのまま移植しない**
- **草案のテストデータ自体のバグ**: `test_coefficients_and_r_squared`は、テストデータ`y=[2.1,3.9,6.2,8.1]`に対する真のOLS解が切片=0.0であるにもかかわらず、アサーションが切片≈0.2を期待しており失敗する。草案のテストを参考にする場合はこのテストの数値を先に直すこと

## 信頼区間

- `OLSOptions`に`confidence_level: f64`（デフォルト`0.95`）を追加。`alpha`という名前は使わない（統計慣習上「有意水準」を指すため紛らわしい）
- 内部で`alpha = 1.0 - confidence_level`を計算し、`fit()`実行時に一度だけCIを計算してcoef_table等に含めて返す。実行時可変引数にはしない

## クラスター変数は文字列として扱う

- `cluster_col`で指定する列は`i64`固定にしない。州名・企業ID等、実務では文字列/カテゴリカルなクラスター変数の方が多いため、内部では`Vec<String>`（Utf8にキャストしたグループキー）として扱う
- `engine`側の`cluster_sandwich`実装（未着手）も、`HashMap<i64, ...>`ではなく`HashMap<String, ...>`でグループ化する前提にすること

## 受け口レベルで検証すべき追加項目

実装時に気づいた、当初のエラー型一覧に無かった検証。`ValidationError`として扱う。

- `y`と`x`に同じ列名が含まれる場合（完全な多重共線性になるため、engine側の一般的な特異行列エラーより先に、分かりやすいメッセージで弾く）
- `x`内に重複した列名がある場合
- `include_intercept=true`のとき、`x`に`"const"`という列名がある場合（自動追加する定数項の名前と衝突する）

## テストフィクスチャ関連の決定事項

- フィクスチャ生成スクリプト（`benchmark/fixtures/generate_ols_fixtures.py`、動作確認済み）と生成物（`tests/api_tests/fixtures/ols.json`）は別管理。詳細は`.claude/rules/testing-policy.md`「ベンチマーク値のフィクスチャ化」参照
- `pyproject.toml`の`[dependency-groups]`に`statsmodels==0.14.6`（`test`グループ）を追加済み（Issue #5で対応）。既存の`ols.json`フィクスチャがこのバージョンで生成されているため、上げる場合はフィクスチャの再生成とセットで行うこと

## 未確定（実装時判断でよい）

- 特異性判定の相対閾値の具体式
- `SingularMatrix`のエラーメッセージを「定数項が重複している可能性があります」等、状況に応じて分岐させるか（優先度低、後回し可）
