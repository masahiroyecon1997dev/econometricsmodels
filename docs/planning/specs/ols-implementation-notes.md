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

### Python側の例外はカテゴリ別に分ける

- 詳細・理由は `.claude/rules/rust-style.md`「エラーハンドリング」を参照（`ValidationError` / `ComputationError`の2階層、`pyo3::create_exception!`で定義）
- OLSでの`OlsError`との対応（案）:

| `OlsError`のバリアント | Python例外 |
|---|---|
| `InvalidInput`（欠損値・次元不一致） | `ValidationError` |
| `InsufficientObservations` | `ValidationError` |
| `MissingClusterColumn` | `ValidationError` |
| `InvalidConfidenceLevel`（新規） | `ValidationError` |
| `InsufficientClusters`（新規） | `ValidationError` |
| `SingularMatrix` | `ComputationError` |
| `ComputationError`（分布計算等の失敗） | `ComputationError` |

## エラー型に追加が必要なもの

草案コードの`OlsError`には無かったが、今回の設計決定に伴い追加が必要なバリアント。

- **`InvalidConfidenceLevel`**: `confidence_level`が`(0, 1)`の範囲外だった場合
- **`InsufficientClusters { g: usize }`**: クラスター数`g < 2`の場合。草案コードでは未検証で、実際に0除算からのNaN伝播でパニックすることを確認済み（`docs/planning/draft-reference/ols-draft-consolidated.md`参照）。`new()`の時点で検証し、`fit()`まで到達させない

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
- `pyproject.toml`のdev dependenciesに`statsmodels`をバージョン固定（`==`）で追加する必要がある。**Issue #4（python_packageの依存追加）はpolars等の本番依存のみを対象としていたため、statsmodelsの追加はどこにも入っていない**。着手時にIssue #4かIssue #17（ベンチマーク作成）のどちらかに追記すること

## 未確定（実装時判断でよい）

- 特異性判定の相対閾値の具体式
- `SingularMatrix`のエラーメッセージを「定数項が重複している可能性があります」等、状況に応じて分岐させるか（優先度低、後回し可）
