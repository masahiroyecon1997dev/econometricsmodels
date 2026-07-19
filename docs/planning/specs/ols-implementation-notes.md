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
- **スコープ外（Issue #9時点）**: `cov_type`によるHC/cluster/HACへの分岐（Issue #10・#11、いずれも実装済み）。R²・調整済みR²・F統計量・AIC/BIC・対数尤度等の**適合度統計量はIssue #21**（Issue #11の後に着手）で実装する

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

### Newey-West HAC標準誤差（Issue #11で実装済み）

- `CovType::Hac { lags: Option<i64>, time_order: Option<Vec<f64>> }`をフィールド付きバリアントとして追加した。
  他のバリアント（`Classical`/`Hc0`〜`Hc3`）と違い、HACだけがラグ数・時間順序という追加パラメータを持つ。
  `OlsEstimator::fit`のシグネチャに`hac_lags`/`time_order`を常に生える引数として追加すると、HAC以外の
  `cov_type`では常に無意味な引数になってしまうため、`CovType`自身にデータを持たせる設計にした
  （この設計判断はIssue #3で確定した`OLSOptions`側の`hac_lags`/`time_col`フィールドの存在は前提にしつつ、
  それを`engine`内部でどう表現するかという実装issue #11のスコープ内の判断）。
  `Vec<f64>`を持つため`Copy`は付与できず、`Clone`のみ（`Hc*`等は元々`Copy`だったが、
  enum全体で外した。呼び出し側で必要なら`clone()`する）
- `(X'X)⁻¹Ŝ(X'X)⁻¹`の対角成分の平方根として計算する（`hac_std_errors`関数）。
  `Ŝ = Ŝ₀ + Σ_{l=1}^{L} w_l(Ŝ_l + Ŝ_l')`（Bartlett重み`w_l = 1 - l/(L+1)`）。
  `Ŝ_l`（l≥1）は対称でない外積の和（`x_t x_{t-l}'`）のため、HC0-3の`Xw'Xw`のような単純な行列積には
  落とし込めず、素直な三重ループ（ラグ×観測×`k²`）で計算する。`k`（説明変数の数）は通常小さいため許容
- ラグ数の解決（`resolve_hac_lags`）: `Some(l)`なら`0 <= l < n`を検証（`OlsError::InvalidHacLags`）。
  `None`なら経験則`L = floor(4*(n/100)^(2/9))`で自動計算（`ols-standard-errors.md`3.2節）
- 時間順序の解決（`time_ordering`）: `time_order`（`OlsInput`の行と対応する長さnの配列）が`Some`なら
  昇順ソートした行インデックス列を返し、`None`なら恒等順序（`OlsInput`の行順をそのまま時系列順とみなす）。
  実際の`X`/残差の並べ替え自体は`hac_std_errors`内でこのインデックス列を使って行う（`OlsInput`自体は
  並べ替えない。理由: `OlsInput`は`fit`の全`cov_type`で共有されるため、ここで恒久的に行を並べ替えると
  Python側に返す残差配列の行と元のDataFrameの行の対応が崩れてしまう）
- `partial_cmp().unwrap()`（時間順のソート）は、NaN/無限大が含まれないことが`engine_pybind`側の
  列抽出（`column_extraction::extract_f64_column`）で既に保証されている前提でパニックしない
- statsmodelsとの数値照合: `sm.OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": L}, use_t=True)`。
  `cov_kwds`に`use_correction`は明示していない（既定の`False`＝小標本補正なしで、本実装の式と一致することを
  確認済み）。テストは3種類: `maxlags`固定値指定、`hac_lags=None`（自動計算L=2、n=5のケース）、
  `time_order`指定（行をシャッフルした入力から`time_order`で正しく時系列順に復元できることを確認）

### 適合度統計量（Issue #21で実装済み）

- Issue #21の本文は根拠として`docs/planning/specs/ols-standard-errors.md`「ロバストな標準誤差選択時のF検定」を挙げているが、該当節はこのファイルには存在しない。実際の決定箇所は`ols-api-design.md`6章の1文（「`cov_type`がHC系/clusterの場合、F検定も**ロバストWald検定**に切り替える」）のみで、HACへの言及もない。Issue #21着手時にユーザーに確認し、**HACも同様にロバストWald検定に切り替える**方針で確定した（statsmodelsの実際の挙動が`cov_type != "nonrobust"`なら常にロバストWald検定、という単純な条件分岐であることとも一致する）
- `OlsInput`に`has_intercept: bool`フィールドを追加（`from_columns`の`include_intercept`引数をそのまま保持）。R²・調整済みR²のcentered/uncentered TSS切り替え、F検定の自由度（`k_constant`）の判定に使う
- **`(X'X)⁻¹`ベースの標準誤差計算関数を「対角成分の平方根（std_errors）を返す」から「k×kの分散共分散行列（cov_params）を返す」設計に変更**（`classical_std_errors`→`classical_cov_params`、`hc_std_errors`→`hc_cov_params`、`hac_std_errors`→`hac_cov_params`とリネーム）。ロバストWald検定（下記）に完全な共分散行列が必要になったため。`std_errors`は`fit()`側で対角成分の平方根を取って求める。**`cov_params`自体はPython側に公開しない**（`ols-api-design.md`5章の「Rust/engine_pybind側の責務は配列＋名前リストを返すところまで（params, std_errors, t_stats, p_values, conf_int, param_names）」に`cov_params`は含まれないため。`fit()`内のローカル変数として使い切り、`OlsEstimator`のフィールドにもしない）
- **R²・調整済みR²**: `include_intercept=true`ならcentered TSS（`Σ(y_i-ȳ)²`）、`false`ならuncentered TSS（`Σy_i²`）を使う（statsmodelsの`k_constant`による分岐と一致）。調整済みR²は`1 - ((n-k_constant)/df_resid)*(1-R²)`（草案の`1 - [SSR/(n-k)]/[SST/(n-1)]`という式は`include_intercept=true`のとき代数的に同じ値になるが、`k_constant`を明示的に使う一般形にした）
- **対数尤度・AIC・BIC**: `llf = -(n/2)*(ln(2π) + ln(SSR/n) + 1)`（分散は最尤推定量`SSR/n`。`ols-implementation-notes.md`「実装時に見落としやすい点」に記載の草案バグ＝不偏推定量`SSR/(n-k)`の誤用を回避）。`aic = -2*llf + 2*k`、`bic = -2*llf + ln(n)*k`（`k`は切片を含む全パラメータ数。草案にあった`n·ln(2π)+n`の欠落を修正）
- **F統計量**: `cov_type`によらず単一の式`F = (β_slopes' Σ⁻¹ β_slopes) / q`（`Σ`は`cov_params`のうち切片以外の係数に対応する部分行列、`q`はその次元＝`k - k_constant`）で計算する（`wald_f_test`関数）。この式は`cov_type=Classical`のとき代数的に古典的F検定`((SST-SSR)/q)/(SSR/df_resid)`と完全に一致することを手計算・statsmodelsとの数値照合の両方で確認済みのため、分岐を分けていない。HC0-3・HACでは`cov_params`がロバストな分散共分散行列になるため、この式がそのままロバストWald検定になる。p値はF分布（自由度`(q, df_resid)`。statrsの`FisherSnedecor`を追加使用）の上側確率
  - `q=0`（説明変数が定数項のみ）の場合はstatsmodels同様`f64::NAN`を返す（0除算回避）
  - `Σ`の逆行列はCholesky分解（`Llt`）で求める。正定値行列の主小行列は必ず正定値という定理により理論上失敗しないはずだが、`xtx_inverse`と同様`ComputationFailed`に変換する境界ケース対応をしている
- テストは`fit_computes_r_squared_and_information_criteria_with_intercept`（classical、切片あり）、`fit_computes_r_squared_without_intercept_uses_uncentered_tss`（切片なし、uncentered TSSの確認）、`fit_computes_robust_wald_f_test_for_hc_and_hac`（HC1・HACのロバストWald F検定）の3つを追加。全てstatsmodels 0.14.6（`use_t=True`）と1e-6〜1e-9の許容誤差で数値照合済み

### クラスター標準誤差（Issue #22で実装済み）

- Issue #22着手時にユーザーに2点確認して確定した:
  1. 小標本補正（`G/(G-1) * (n-1)/(n-k)`）は常に適用し、無効化するオプションは設けない（`OLSOptions`に新しいフィールドを追加しない）
  2. `cov_type="cluster"`なのにクラスターキー未指定というエラー（`OlsError::MissingClusterColumn`）は`engine`自体に検知させる。`CovType::Cluster { groups: Option<Vec<String>> }`とし、`None`なら`fit()`が`MissingClusterColumn`を返す（`CovType::Hac`の`lags: Option<i64>`と同じ設計パターン）
- statsmodelsのソース（`statsmodels.stats.sandwich_covariance.cov_cluster`）を確認したところ、小標本補正`use_correction=True`がデフォルトで、`ols-standard-errors.md`5章に既に書かれていた式と完全に一致することを数値照合で確認した（追加のドキュメント変更は不要だった）
- `cluster_cov_params`関数: `Ŝ = Σ_g S_g S_g'`（`S_g = Σ_{i∈g} ε̂_i x_i`、クラスター内の観測を先に合計してから外積を取ることでクラスター内相関を許容する）。グループ化は`HashMap<&str, Vec<usize>>`で行う（`ols-implementation-notes.md`「クラスター変数は文字列として扱う」の既存方針通り）
- クラスター数`G`の検証（`validate_cluster_groups`関数）: `G < 2`なら`OlsError::InsufficientClusters`。`groups.len() != n`は`engine_pybind`側の実装バグでしか起こらない内部契約として`debug_assert_eq!`で検証（`OlsInput::from_columns`の`x_names`/`x_columns`長さ検証と同じパターン）
- **重要な発見（自由度の切り替え）**: statsmodelsは`cov_type="cluster"`のとき、デフォルト（`df_correction=True`）でt検定・信頼区間・F検定の自由度を`n-k`ではなく**`G-1`（クラスター数-1）に切り替える**（`RegressionResults.get_robustcov_results`のdocstring参照）。これは計量経済学の標準的な慣行（Cameron-Miller等）であり、statsmodels固有の癖ではない。標準誤差自体の値は変わらないが、p値・信頼区間・F検定のp値が大きく変わる（小さいクラスター数のとき特に顕著）。ユーザーに確認し、**`cov_type=Cluster`のときのみ自由度を`G-1`に切り替える**方針で確定した（他のcov_typeは引き続き`n-k`で統一）
  - `fit()`内で`(cov_params, df_inference)`のタプルを`cov_type`ごとのmatchから返す設計にした。`df_inference`は通常`df_resid`（n-k）と同じだが、`Cluster`のときだけ`G-1`になる。`df_resid`自体（σ̂²・調整済みR²・AIC/BIC等で使う）は影響を受けず、常に`n-k`のまま
  - `wald_f_test`のF分布の自由度（分母側）も`df_resid`ではなく`df_inference`を受け取るようにシグネチャを変更した
- テストは`fit_computes_cluster_std_errors_t_stats_p_values_conf_int_and_f_test`（2クラスターのデータセットで標準誤差・t値・p値・信頼区間・F統計量を検証。自由度がG-1=1になることも含めてstatsmodelsと数値照合）、`fit_returns_missing_cluster_column_when_groups_not_provided`、`fit_returns_insufficient_clusters_when_only_one_group`の3つを追加

### engine単体テストのカバレッジ（Issue #12で確認・実装済み）

Issue #12は「実装（#9〜#11、実際には#21・#22も含む）と並行して書き足したテストの集約・完了確認用」という位置づけ。着手時に2点確認した。

1. **テスト配置**: `.claude/rules/rust-style.md`「テスト」・`testing-policy.md`「テストの分離」は共に「`tests/engine_tests/`に置く」と書かれていたが、Issue #9以降実際には`engine/src/linear/ols.rs`内の`#[cfg(test)] mod tests`に一貫してインライン実装してきた（`tests/engine_tests/`は`.gitkeep`のみで空）。この乖離を解消するため、両ルールファイルを実態（インライン方式が正式方針）に更新した。`tests/engine_tests/`は削除せず、将来モジュール横断の統合テストが必要になった場合の予約として残す
2. **カバレッジ目標**: `cargo-llvm-cov`で実測。100%は目指さず、理論上到達不能な防御的エラーパスはドキュメント化して受け入れる方針で確定（詳細は`rust-style.md`「テスト」参照）

実測結果（25テスト時点）: Region 98.94%・Line 98.88%・Function 96.34%。未カバー箇所は以下の3種類。

- **すぐ埋められたギャップ**（このissueで追加): `fit()`の`df_model == 0`分岐（`f64::NAN`を返す経路。`fit_returns_nan_f_statistic_when_model_has_no_slope_regressors`で解消）、`OlsEstimator::input()`/`cov_type()`/`residuals()`の3getter（`fit_exposes_input_cov_type_and_residuals_via_getters`で解消）、`validate_cluster_groups`の`debug_assert_eq!`（`fit_panics_when_cluster_groups_length_does_not_match_nobs`で解消）
- **理論上到達不能な防御的エラーパス**（受け入れて未カバーのまま。いずれも「事前に検証済みの不変条件により、理論上失敗し得ないはずだが、浮動小数点の丸めに備えて`Result`化してある」という同じ性質）:
  1. `xtx_inverse`の`Llt`失敗→`SingularMatrix`（特異性は`ensure_full_rank`で先に検出済みのため通常到達しない）
  2. `StudentsT::new`/`FisherSnedecor::new`の失敗→`ComputationFailed`（自由度は`n>k`・`G>=2`・`df_model>=1`の事前検証により常に有効な正の値になる）
  3. `wald_f_test`内の`Llt`失敗→`ComputationFailed`（正定値行列の主小行列は必ず正定値という線形代数の定理により理論上到達不能）
  - これらを実際に踏ませるには丸め誤差でギリギリ破綻する敵対的な浮動小数点データを人為的に作る必要があり、プラットフォーム依存で壊れやすく、実装の振る舞いというより浮動小数点ノイズの検証になるため見送った（`cargo-llvm-cov`の除外マーカーも導入しないことにした。ツール依存設定を増やすより、コード側のdocコメントで理由を説明する方針を優先）
- 残りの「missed lines」表示（981, 986等）は`assert!`マクロのメッセージ引数（アサーション失敗時のみ評価される）による分析ツールの誤検知で、実際のギャップではない

### engine_pybind: Arrowゼロコピーデータ受け渡し（Issue #13で実装済み）

Issue #13着手時に2点確認した。

1. **ファイル構成**: issueの追記は`engine_pybind/src/`直下に`lib.rs, ols_data.rs, ols_options.rs, errors.rs`という古い草案（`docs/planning/draft-reference/engine_pybind_data_layer/`、現存しない）に基づく指示をしていたが、既に`rust-style.md`「ファイル・ディレクトリ構成」の確立済みルール（系統ディレクトリ`linear/`＋YAGNI）に従って`engine_pybind/src/linear/ols.rs`が実装済みだったため、そちらを維持することにした
2. **依存クレートのバージョン**: 詳細は次項

#### polars（Rustクレート）とpyo3-polarsのバージョン選定（重要な発見）

issueの追記は「Python側pyproject.tomlのpolars（`1.42.1`）とバージョンを合わせる」としていたが、これは実行不可能だった。**Rustの`polars`クレート（crates.io公開）とPython側`polars`パッケージ（PyPI）は既にバージョン体系が分離している**（polarsモノレポの2026年時点のmainブランチを確認: `py-polars/pyproject.toml`は`version = "1.43.0"`なのに対し、同じコミットの内部Rustワークスペースクレートは`[workspace.package] version = "0.54.4"`）。

実際の互換性は数字ではなく、`pyo3-polars`の`PyDataFrame`/`PySeries`変換が使う`polars_ffi::version_0`という安定版FFIプロトコル（Python側`Series._export(ptr)`メソッドが書き出したバッファを、こちら側の`import_series`で読み取る。Arrow C Data Interfaceに近い、バージョン非依存のインターフェース）によって担保される（`pyo3-polars`のソース、`pyo3-polars/pyo3-polars/src/types.rs`の`PyDataFrame`/`PySeries`の`FromPyObject`実装を確認）。

バージョン選定は以下の手順で行った:
1. `cargo add polars/pyo3-polars --dry-run`でcrates.io最新版を確認: `polars=0.54.4`, `pyo3-polars=0.27.0`
2. `pyo3-polars=0.27.0`は`pyo3="^0.28"`を要求することが判明。既に固定済みの`pyo3=0.29.0`と衝突した（rust-style.md「既知のリスク」が警告していた、crates.io公開版とpolars本体リポジトリ内バージョンのズレそのもの。polarsモノレポのmainブランチは内部で既に`pyo3=0.29`に上げているが、対応する`pyo3-polars`の新版はまだcrates.ioに未公開）
3. ユーザーに確認し、**`pyo3`を`=0.29.0`から`=0.28.2`にダウングレード**して解消することにした（`0.28.0`/`0.28.1`はyanked済みのため除外。`=0.28.2`が最初の非yanked安定版）

最終的な組み合わせ: `pyo3=0.28.2`, `polars=0.54.4`, `pyo3-polars=0.27.0`（すべて`=`で完全固定、`engine_pybind/Cargo.toml`）。

#### 実装時に必要だった修正（草案コードとpolars 0.54.4の差異）

`column_extraction.rs`は当初「polarsを実際にビルドして検証できていない」草案だった。実際にビルドして判明した差異:

- `ChunkedArray::rechunk()`が`Cow<'_, ChunkedArray<T>>`を返すようになり（既に単一チャンクの場合の不要なcloneを避けるため）、`Cow`は`IntoIterator`を実装しないため`.into_iter()`が使えなくなっていた。`ChunkedArray::iter()`メソッド（`Cow`はDerefで透過的に呼べる）に置き換えて解消（`extract_f64_column`・`extract_group_key_column`両方）
- `lib.rs`の`fit_ols`戻り値型`PyObject`が見つからないエラー。pyo3 0.28では`PyObject`という型エイリアス（`Py<PyAny>`の別名）自体がpreludeから削除されていたため、`Py<PyAny>`に置き換えて解消
- `#[pyclass]`（`OLSOptions`）でdeprecation警告。pyo3 0.28以降、`Clone`を実装する`#[pyclass]`の`FromPyObject`自動導出がopt-inに変更されたため、`#[pyclass(from_py_object)]`に変更して明示化（`fit_ols`がPython側から`OLSOptions`インスタンスを引数として受け取るため、`FromPyObject`実装自体は必要）

#### 検証方法

`cargo build/test/clippy/fmt --workspace`に加え、`uv run maturin develop`で実際にビルド・インストールし、実際のPython polars 1.42.1のDataFrameを使った手動スモークテスト（`/tmp`配下、コミット対象外）で以下を確認した:
- 通常の数値データでの`fit_ols`呼び出しが、データ抽出を完了した上で想定通り`ComputationError`（Issue #14未実装のため）に到達すること
- 欠損値（null）・NaN・infinityを含む列が正しく`ValidationError`になること
- クラスターのグループキー列（文字列型）の抽出が正しく動作すること

いずれも成功。「Arrow経由のゼロコピーデータ受け渡し」は、Python→Rust境界の受け渡し自体を指す（`.claude/rules/rust-style.md`「Python境界でのデータ受け渡し」で既に確定済みの通り、polars Seriesからfaer::Matへの詰め替え自体は2回のコピーを許容する設計のまま。ここが変わったわけではない）。

### engine_pybind: engine呼び出し・エラー変換実装（Issue #14で実装済み）

Issue #13までの`extract_ols_input`（受け口の検証・変換のみ、`OlsFitInput`という暫定型を返して`fit_ols`側でエラーに打ち切る）を、`engine::linear::ols::OlsInput::from_columns` + `OlsEstimator::fit`を実際に呼び出す形に差し替えた。着手時に2点確認した。

1. **返り値に適合度統計量を含めるか**: Issue #21で実装済みの`r_squared`等を含める方針で確定（`ols-api-design.md`5章の「`params, std_errors, t_stats, p_values, conf_int, param_names`」というリストはIssue #21着手前の記述だったため、Issue #14で更新した）
2. **返り値の形式**: `OLSOptions`と同じ`#[pyclass]`パターンで`OLSResult`を新設。`conf_int`は`conf_lower`/`conf_upper`の2配列に分割（engineの内部表現・`Vec<(f64,f64)>`より実装が簡潔なため）

#### 実装内容

- `OLSOptions`に`hac_lags: Option<i64>`・`time_col: Option<String>`を追加（Issue #3で確定していたが、Issue #11時点ではengine側の`CovType::Hac`のみ実装され、engine_pybind側への反映はIssue #14に持ち越されていた）
- `OLSResult`（`#[pyclass(get_all, skip_from_py_object)]`）: `params`/`std_errors`/`t_stats`/`p_values`/`conf_lower`/`conf_upper`/`param_names`/`residuals`/`dep_var_name`/`nobs`/`cov_type`（実際に使われた種別を小文字文字列で echo）/`r_squared`/`r_squared_adj`/`f_statistic`/`f_p_value`/`log_likelihood`/`aic`/`bic`を公開。`skip_from_py_object`なのは、`OLSResult`がRust側で組み立ててPythonに返すだけの型で、Python側から構築されることを想定しないため（`OLSOptions`の`from_py_object`とは対照的）
- `engine::linear::ols::OlsError` → `PyErr`の変換: 当初`impl From<OlsError> for PyErr`を試みたが、`OlsError`（`engine`クレート）・`PyErr`（`pyo3`クレート）のどちらもこのクレート外で定義された型のため**orphan ruleに抵触してコンパイルエラー**になった。関数`fn ols_error_to_pyerr(err: OlsError) -> PyErr`として実装し、呼び出し側で`.map_err(ols_error_to_pyerr)?`する形に変更して解消
- `cov_type`固有の追加列（`cluster_col`/`time_col`）の抽出は、該当する`cov_type`のときのみ行う（`cov_type != "hac"`のとき`time_col`は無視、`cov_type != "cluster"`のとき`cluster_col`は無視、`ols-standard-errors.md`3.2/3.3節の既存方針通り）。誤って無関係な列を要求してエラーになることを避ける
- **早期バリデーションの重複排除**: `confidence_level`の範囲チェック、`cov_type="cluster"`なのに`cluster_col`未指定のチェックは、Issue #13までの暫定コードではengine_pybind側でも行っていたが、`engine::OlsEstimator::fit`（`InvalidConfidenceLevel`）・`CovType::Cluster`（`MissingClusterColumn`、Issue #22で確定した設計）が既に検知するため、engine_pybind側の重複チェックを削除した。`y`/`x`の重複列名・`"const"`列との衝突等、engineが検知できない（列名を知らない）ものはengine_pybind側の責務として残した
- `OlsFitInput`・engine_pybind独自の`CovType`列挙型（`Hac`を持たない暫定版）は削除。`faer::Mat`の直接構築もengine_pybindから完全になくなった（`engine::linear::ols::OlsInput::from_columns`に一本化）

#### 検証方法

`cargo build/test/clippy/fmt --workspace`に加え、`uv run maturin develop`で実ビルド・インストールし、実際のPython polars 1.42.1のDataFrameで以下を確認した:
- classical cov_typeの`fit_ols`結果が、`engine`のテストで使っている教科書的データセット（x=[1..5], y=[2,4,5,4,5]）のオラクル値（`params`/`std_errors`/`r_squared`/`aic`/`bic`等）と完全一致すること
- HAC（`hac_lags=1`）・cluster（2クラスター）の結果も、対応するengineテストのオラクル値と浮動小数点誤差（1e-13〜1e-14程度）の範囲で一致すること
- `MissingClusterColumn`・`InvalidHacLags`・未知の`cov_type`文字列・`SingularMatrix`のそれぞれが、想定通り`ValidationError`/`ComputationError`として送出されること

### python_package: OLSラッパー実装（Issue #15で実装済み）

`/implement-python`スキル経由で実装。着手前に2点確認した。

1. **モジュール構成**: `python_package/econometricsmodels/linear/ols.py`を新設（`engine`/`engine_pybind`と同じ`linear/`系統ディレクトリ構成をpython_package側にも揃える方針。CLAUDE.md 3章の当時のリポジトリ構成図はまだ単一ファイルのままだったが、20〜30手法規模を見据えて先に揃えた）
2. **`tests/api_tests/test_ols.py`（設計確定前の草案）の扱い**: 今回確定済み設計に合わせて全面的に書き直すことにした（Issue #19「tests/api_tests作成」に先送りしない）。旧版は`OLS(df, y="y", x=[...], cov_type=cov_type)`というフラットなキーワード引数渡し、`res.summary()`（5章で「作らない」と確定済み）、`res.to_frame()`/`res.conf_int()`がpolars DataFrameを返す（同章「係数テーブルにpolars DataFrameは使わない」と矛盾）等、`ols-api-design.md`7章に記載の不整合を抱えていた

#### 実装内容

- `OLS`クラス（`data`/`y`/`x`/`options: OLSOptions | None`のオブジェクト渡し）と`OlsResults`クラス（`_lib.OLSResult`の薄いラッパー）を実装。`OLSOptions`自体は独自定義せず`_lib`から再輸出（`ols-api-design.md`3章の既存方針通り）
- `OlsResults`のプロパティ設計:
  - `params`/`std_errors`/`t_stats`/`p_values`は`dict[str, float]`（係数名→値。O(1)取り出し用）
  - `conf_int`は`dict[str, tuple[float, float]]`（`_lib.OLSResult`の`conf_lower`/`conf_upper`2配列から組み立て）
  - `coef_table()`メソッドで行指向の`list[dict]`（キー: `param`/`coef`/`std_err`/`t_stat`/`p_value`/`conf_lower`/`conf_upper`）を提供。REST APIレスポンスにそのまま使える形（5章の2形式のうち先に実装すべきとされていたもの）
  - `residuals`はそのまま`list[float]`（`_lib.OLSResult`の値を素通しするだけの薄いラッパーという位置づけを優先し、polars Seriesへの変換等は行わない。design docに明記のない拡張のため見送った）
  - `summary()`・`predict()`・`fitted_values`・`to_frame()`/`conf_int()`のDataFrame版は実装しない（5章の確定方針、および「薄いラッパー」というissue本文のスコープに照らして見送った）
- **`pyproject.toml`に`[tool.ruff] line-length = 79`を追加**: `.claude/rules/python-style.md`は元々この値を明記していたが、`[tool.ruff]`セクション自体がリポジトリに存在せず、これまで未設定（デフォルトの88文字）だったことが判明。python_packageの最初の実コードを書くタイミングで顕在化したため追加した
- 上記追加により`benchmark/`配下の既存スクリプト（Issue #15のスコープ外）でlint/format違反が新たに表面化した（未使用import1件、フォーマット崩れ4ファイル）。CIの`ruff check .`/`ruff format --check .`はリポジトリ全体が対象のため、ユーザーに確認の上、この場で`ruff check --fix`・`ruff format`により機械的に修正した（ロジック変更なし）

#### 検証方法

`uv run maturin develop`で実ビルド・インストールし、`uv run pytest tests/api_tests/`で書き直した29件のテストを実行（全通過）。`ruff check .`・`ruff format --check .`もリポジトリ全体でクリーン。テストはstatsmodels（`use_t=True`）とclassical/HC0-3/clusterの係数・標準誤差・R²・F統計量を数値照合し、HACは動作確認（statsmodelsとの数値照合はしない。既存のengine単体テストで数値照合済みのため）。

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

- **ロバストF検定への切り替え**（Issue #21で解消済み）: `cov_type`がHC系/HAC/clusterの場合、F検定（適合度統計量）もロバストWald検定に切り替える方針が決定済み（HACも含めることをIssue #21着手時に確認）。SE計算の実装issueとは別に、適合度統計量計算の実装issueでも見落とされないよう明記すること
- **`log_likelihood` / `AIC` / `BIC`の計算式**（Issue #21で解消済み）: 草案コードはここにバグがあった（対数尤度の分散に不偏推定量`SSR/(n-k)`を誤用。正しくは最尤推定量`SSR/n`。AIC/BIC式も`n·ln(2π)+n`の定数項が欠落）。当時参照していた`docs/planning/draft-reference/ols-draft-consolidated.md`は現在リポジトリに存在しないため、`docs/spec/01_ols.md`4章の式を代わりに参照した。**この部分は草案をそのまま移植しない**
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

## テストデータの確認・整備（Issue #17で確認済み）

- **合成データセット**（`benchmark/generate_synthetic_datasets.py`の7シナリオ）: `testing-policy.md`が要求する全バリエーション（小標本`small_n`、高分散`high_variance`、不均一分散`heteroskedastic`、自己相関`autocorrelated`、中程度の多重共線性`moderate_multicollinearity`、完全な多重共線性`perfect_multicollinearity`）に`baseline`を加えた7種で網羅済み。追加実装は不要と確認した。
  - `maturin develop`でビルドした実バイナリに対し、7シナリオ全てを`OLS(...).fit()`に通して動作確認済み。`perfect_multicollinearity`のみ想定通り`ComputationError`（設計行列が特異）を送出し、それ以外は正常に収束することを確認した。
- **実データセット**: `benchmark/load_wooldridge.py`の`SUGGESTED_DATASETS["ols"]`候補（`wage1`, `gpa2`）を確定した。
  - 両データセットとも欠損値なし（`null_count() == 0`）。
  - 列の型はInt64/Float64混在（例: `wage1`の`educ`/`exper`/`tenure`はInt64、`wage`/`lwage`はFloat64）だが、`column_extraction::extract_f64_column`が`cast(&DataType::Float64)`で吸収するため問題なし。
  - `wage1`（n=526、`lwage ~ educ + exper + tenure`）・`gpa2`（n=4137、`colgpa ~ sat + hsperc + tothrs`）ともに実バイナリで最終フィットを確認済み。`wage1`の`educ`係数≈0.092はWooldridge教科書の既知値と整合。
- テストフィクスチャ（`generate_ols_fixtures.py`）は現状合成データのみを対象にしており、実データ（wage1/gpa2）はフィクスチャ化していない。実データは「真の係数と比較できないため、リファレンス実装との一致のみで検証する」（`testing-policy.md`）用途であり、フィクスチャへの組み込みはIssue #19（`tests/api_tests`整備）以降のスコープとする。

## リファレンス実装によるクロスチェック（Issue #18で実装済み）

`tests/api_tests/fixtures/benchmarks/ols.json`（statsmodels、主リファレンス）はIssue #21以前から存在していたが、`testing-policy.md`が定める「新しいcov_type追加時はstatsmodelsとRの一致を先に確認してからフィクスチャを固定する」というクロスチェック自体は、これまで一度も実施されていなかった（HAC含む全cov_typeが未検証のまま）。Issue #18でこのギャップを解消した。

- **役割分担の確定**: `testing-policy.md`が既に定めている「Rが正式なクロスチェック、pyfixestはOLSでは主役ではない」という分担を踏襲した。pyfixestはv0.60.0でHAC(`vcov="NW"`)・クラスター(`vcov={"CRV1": ...}`)にも対応していることが分かったが、**fixestベースのpyfixestは独立実装とは言えない**（fixestの実装元とほぼ同じ設計思想）ため、正式なクロスチェックには使わないという既存方針を維持することにした。
  - R側は`lm() + sandwich::vcovHC/vcovCL/NeweyWest + lmtest::coeftest`（base Rの独立実装）で、classical/HC0-3/cluster/HACの全cov_typeを確認する。
  - pyfixestはclassical/HC1-3のみ補助的に確認する（HC0はpyfixestが公開しておらず、`"hetero"`が`"HC1"`と同一の値を返す仕様のため対象外）。cluster/HACはpyfixestでは確認しない。
- **`benchmark/run_r_benchmark.R`の拡張**: 従来`fixest`/`plm`/`ivreg`のみだった`package`引数に`lm`を追加し、`sandwich`/`lmtest`ベースのクロスチェック機能を実装した（当初「Rはこのサンドボックス環境に無いため未検証」と記載されていたが、devcontainerには`fixest`/`sandwich`/`lmtest`/`jsonlite`が導入済みで、動作確認できた）。
  - **`read.csv()`の罠**: デフォルトでは列名が`make.names()`により書き換わる（例: `_group`→`X_group`）ため、クラスター列を正しく渡せていなかった。`check.names = FALSE`を追加して修正した。
  - クラスターSEは`vcovCL(model, cluster=..., type="HC1", cadjust=TRUE)` + 自由度`G-1`で、statsmodels（≒本実装）と厳密一致することを数値照合で確認した。
  - HACは`NeweyWest(model, lag=<本実装と同じ明示ラグ>, prewhite=FALSE, adjust=TRUE)`。厳密一致は期待しない設計だが、statsmodelsとの差は相対誤差0.4%程度で「大きな乖離ではない」ことを確認した。
- **pyfixestのHC2/HC3に関する既知の差異**: pyfixest（fixest）はHC2/HC3に対し、標準的なMacKinnon-White公式に加えて**`sqrt(n/(n-k))`倍の追加小標本補正（ssc）**を掛ける仕様であることを検証中に発見した。R（sandwich）・statsmodels・本実装はこの追加補正を行わないため、`n`が小さいほど乖離が大きくなる（`small_n`シナリオ n=20, k=4 では約11.8%）。バグではなく実装間の既知の設計差のため、クロスチェックテストではpyfixest比較のみ許容誤差を緩めている（`test_ols_crosscheck.py`の`RTOL_PYFIXEST=0.15`）。
- **成果物**:
  - `benchmark/fixtures/generate_ols_crosscheck_fixtures.py`（新規）: 合成データ6シナリオ（`perfect_multicollinearity`除く）×R全cov_type、classical/HC1-3×pyfixest、baselineのみcluster、autocorrelatedのみHAC、Wooldridge実データ（wage1/gpa2）でも同様に生成。
  - `tests/api_tests/fixtures/benchmarks/ols_crosscheck.json`（新規、コミット対象）。
  - `tests/api_tests/test_ols_crosscheck.py`（新規）: 上記フィクスチャを読み込み、本実装の実際の出力と緩い許容誤差（R: 相対誤差1%、pyfixest: 相対誤差15%）で比較する74ケースのテスト。`wooldridge`パッケージが無い環境では`pytest.importorskip`でskipする（`tests/api_tests`本体は`test`依存グループのみで完結させる方針、benchmark依存グループはCIでは使わない想定のため）。
- **`pyproject.toml`に`pyarrow==25.0.0`を`benchmark`グループへ追加**: `polars.DataFrame.to_pandas()`が内部で要求するが依存関係に漏れていた（`run_pyfixest_benchmark.py`・`run_statsmodels_benchmark.py`とも、これが無いと実行時エラーになる状態だった）。

## tests/api_tests作成（Issue #19で実装済み）

`tests/api_tests/fixtures/benchmarks/ols.json`（statsmodels主リファレンス、Issue #21実装時に作成）は、これまでどのテストからも読み込まれておらず、`test_ols.py`（Issue #15）は都度statsmodelsを直接呼び出す方式で1データセットのみを検証していた。Issue #19で、`ols.json`を読み込んで`testing-policy.md`の基本方針（相対誤差1e-8）に沿った厳密比較を行う`tests/api_tests/test_ols_fixtures.py`（新規、38テスト）を追加した。

- 対象: 合成データ6シナリオ（`perfect_multicollinearity`除く）×classical/HC0-3/HAC + baselineのみcluster。係数・標準誤差・t統計量・p値・信頼区間・R²・調整済みR²・F統計量・F検定p値・AIC/BIC/対数尤度を網羅的に比較する。
- HACはフィクスチャ生成時と同じ`hac_lags=1`（`generate_ols_fixtures.py`のstatsmodels呼び出しが`maxlags=1`固定のため）を明示的に指定し、自動ラグ選択式の違いを比較対象から除外した。
- p値等、0に近い極小値は`RTOL * ref`だと許容誤差が過度に厳しくなる（`ref=0`付近で許容誤差もほぼ0になる）ため、絶対誤差フロア`ATOL=1e-10`を`max(RTOL*|ref|, ATOL)`の形で組み合わせている。

### 発見: `ols.json`フィクスチャの陳腐化（信頼区間が正規分布ベースの古い値のまま）

実装したテストを最初に実行したところ、HC0-3/HACの`conf_int`（信頼区間）が本実装の出力と一致しなかった（係数・標準誤差・t統計量は一致）。調査の結果、**`ols.json`に記録されていた信頼区間の臨界値が、t分布ではなく正規分布（`z=1.95996...`）ベースの値になっていた**ことが判明した。

- `run_statsmodels_benchmark.py`を今すぐ再実行すると、`model.fit(cov_type="HC0", use_t=True)`は正しくt分布ベースの信頼区間を返す（本実装の出力と一致）。つまり**現在のスクリプト自体にバグはない**。
- 一方コミット済みの`ols.json`はこの現在の実行結果と食い違っており、`git log`で確認したところ元は`aaaebd3`「テストのベンチマーク作成用コードの作成、比較用json作成（仮）」で仮生成されたまま、Issue #10（HC-robustでのt分布使用の再確認）以降一度も再生成されていなかったと考えられる（`testing-policy.md`「フィクスチャは一度固定したら手動更新が前提（自動追従しない）」通りの状態）。
- `benchmark/fixtures/generate_ols_fixtures.py`を再実行し、`ols.json`を現在のスクリプトの出力で上書き・再コミットして解消した。係数・標準誤差は元々一致していたため影響なし、信頼区間のみ修正された。
- 教訓: フィクスチャを実際に読み込んで検証するテスト（本Issueで初めて追加）がなければ、この陳腐化は発見できなかった。今後cov_type・統計量を追加した際は、フィクスチャ再生成を忘れずに行うこと。

## mkdocsドキュメント作成（Issue #20で実装済み）

`docs/mkdocs.yml`が存在せずmkdocsの土台自体が未構築だったため、OLSのページ追加と合わせて土台構築も本Issueで行った（ユーザー承認済み）。

- **`docs_dir: .`**: `mkdocs.yml`自体を`docs/`直下に置く（CLAUDE.md 3章の想定構成）ため、デフォルトの`docs/mkdocs.yml`相対`docs/`（実質`docs/docs/`）ではなく、`docs/`自身をdocs_dirとして明示指定した。これにより`docs/planning/`・`docs/spec/`・`docs/plan.md`もビルド対象に含まれるが、`nav`には載せていない（CLAUDE.md 10章「ソースとしては誰でも閲覧可能という前提で運用する」通りの状態）。
- **テーマ**: mkdocs-material（`docs`依存グループに`mkdocs==1.6.1`・`mkdocs-material==9.7.7`・`mkdocstrings[python]==1.0.6`を追加）。
- **API リファレンスはmkdocstringsで自動生成**: `docs/api/ols.md`は`::: econometricsmodels.OLS`等のディレクティブのみで構成し、docstring（Issue #16でGoogleスタイル整備済み）から自動生成する。二重管理を避けるため。
- **発見・修正: PyO3コンパイル済みクラスの`__module__`がmkdocstringsのalias解決を壊す**: `OLSOptions`/`OLSResult`（`engine_pybind/src/linear/ols.rs`の`#[pyclass]`）はデフォルトで`__module__ == "builtins"`になる（PyO3の既定動作）。これにより、mkdocstringsの静的解析器（griffe）が`econometricsmodels.OLSOptions → econometricsmodels.linear.ols.OLSOptions(再エクスポート) → econometricsmodels._lib.OLSOptions`というalias連鎖を解決できず、`AliasResolutionError`でビルドが失敗した。`#[pyclass(module = "econometricsmodels._lib")]`を明示することで解消した（`maturin develop`での再ビルドが必要）。今後PyO3で新しい`#[pyclass]`を追加する際は、Python側から`__init__.py`経由で再エクスポートしdocstringを自動生成する予定であれば、最初から`module`属性を明示しておくとよい。
- **mkdocstringsの`filters`設定**: デフォルトでは`__new__`/`__repr__`等のdunderメソッドがPyO3クラスの動的検査で「メンバー」として拾われ、ノイズになる。`filters: ["!^_"]`で除外した。
- **ページ構成**: `index.md`（トップ）・`getting-started.md`（インストール・使用例・エラーハンドリング）・`api/ols.md`（APIリファレンス）。`getting-started.md`のコード例は実際に`uv run python`で実行し動作確認済み。
- **`site/`（ビルド出力）は`.gitignore`に追加**しコミット対象外とした。GitHub Pagesへの自動デプロイ（`cd_docs.yml`）は別issueとし、本Issueではローカルビルド（`uv run mkdocs build -f docs/mkdocs.yml`）の確認までに留めた。

## CI: ci_engine.yml作成（Issue #23で実装済み）

`.github/workflows/ci_engine.yml`を新規作成し、`engine`（純粋Rust）の継続的な品質検証を自動化した。

- **トリガー**: `engine/**`・`Cargo.toml`・`Cargo.lock`（workspaceルート）・ワークフローファイル自体の変更。`Cargo.lock`は`engine/`配下ではないが、依存バージョンの変更が`cargo test`/`cargo audit`双方に影響するため対象に含めた。
- **`test`ジョブ**: `cargo fmt -p engine --check` → `cargo clippy -p engine --all-targets -- -D warnings` → `cargo test -p engine`。全て`-p engine`で明示的にスコープした。
  - **判断: `engine_pybind`はclippy/fmt対象外**。rust-style.mdの責務分離（`engine`はPyO3非依存の純粋Rust）と一貫させ、`ci_engine.yml`がPython/PyO3環境を一切必要とせず完結するようにした。`engine_pybind`側のclippy/fmtは`ci_python.yml`（Issue #24）側で追加担当する（`maturin develop`によるビルドとは別に、明示的にカバーする必要がある）。
- **`audit`ジョブ**: `rustsec/audit-check`でworkspace全体の`Cargo.lock`（`engine`+`engine_pybind`両方の依存）をRustSecアドバイザリDBと突き合わせる。
  - **判断: workspace全体を`ci_engine.yml`側の責務とする**。`Cargo.lock`はworkspaceで1つしかなく分割できないため、`ci_python.yml`のPython依存脆弱性検知（`pip-audit`、`uv.lock`対象）と役割を分けた。
- **ツールチェーンセットアップアクション**: `dtolnay/rust-toolchain`ではなく`actions-rust-lang/setup-rust-toolchain`を採用した。キャッシュ（`Swatinem/rust-cache`相当）を内蔵し別ステップが不要なこと、rustc/clippyの警告をPRの差分行にインライン注釈（problem matcher）できることが決め手（ソロ開発でPRを自分でレビューする運用と相性が良い）。デフォルトで`RUSTFLAGS="-D warnings"`が付与される（コンパイラ警告もエラー化）が、`cargo test -p engine`をこの設定で実行しても警告は出ないことをローカルで確認済み。
- **全アクションをコミットSHAで固定**（サプライチェーン攻撃対策）。タグ名はコメントとして併記し、`gh api repos/<owner>/<repo>/git/refs/tags/<tag>`で実際に解決したSHAを使用した（mutableなタグそのものは参照しない）。
- **`cargo-about`（ライセンス確認）は本issueに含めず、Issue #32（公開前チェックリスト）側に追記**した。依存追加のたびに全PRで回すものではなく、リリース前の1回性の確認作業（ライセンス一覧レポート生成＋配布物へのNOTICE同梱）と位置づけた。
- **`rustsec/audit-check`アクションは不採用**: `cargo audit --json`の出力をアクション内で`JSON.parse()`しており、出力にANSI制御文字等が混ざると`Unexpected token ... in JSON`で失敗する既知の不具合が未解決のまま残っている（`rustsec/audit-check`のIssue #18・#33、いずれも1年以上オープン）。実際の脆弱性と無関係にCIが赤くなる偽陽性を招くため、`taiki-e/install-action`（RustSec公式のビルド済みバイナリを取得するだけでソースコンパイル不要）で`cargo-audit`をインストールし、テキスト出力のまま`cargo audit`を直接実行する方式に変更した。
- **発見: `cargo audit`が実際に4件の脆弱性を検知**（実装検証中に判明）。`pyo3 0.28.2`（直接依存、RUSTSEC-2026-0176/0177）、`quick-xml 0.39.4`（`polars`→`object_store`経由の推移依存、RUSTSEC-2026-0194/0195、severity 7.5 high）。加えて`bincode`（`polars`→`polars-utils`経由）・`paste`（`faer`→`gemm`経由）がunmaintained警告（非ブロッキング）。
  - `quick-xml`/`bincode`は`cargo tree`のデフォルト出力に現れず、`cargo metadata`のresolveノードを直接確認して発見した。`object_store`はpolarsのクラウドストレージ連携機能で、本プロジェクトはローカルDataFrameしか扱わないため未使用。`polars`の`default-features`を絞れば除外できる可能性がある（未検証）。
  - 恒久対応（`pyo3`アップグレード等、`pyo3-polars`とのバージョン整合検証が必要で規模が不明）はIssue #46に切り出した。
  - `ci_engine.yml`のマージ自体をブロックしないよう、`.cargo/audit.toml`に該当4件のRUSTSEC IDを一時的にallow-listした（cargo-auditの設定ファイルは`.cargo/audit.toml`固定パスでのみ自動検出される。workspaceルート直下の`audit.toml`は読み込まれないことを実機検証で確認した）。Issue #46完了時にこのallow-listごと削除する。

## CI: ci_python.yml作成（Issue #24で実装済み）

`.github/workflows/ci_python.yml`を新規作成し、`python_package`/`engine_pybind`の継続的な品質検証を自動化した。3ジョブ構成。

- **`test`ジョブ**（Python 3.12/3.13/3.14マトリクス、`fail-fast: false`）: `uv sync --locked --group test` → `uv run maturin develop` → `pytest tests/api_tests` → `ruff check .` → `ruff format --check .`。
  - **`--locked`を採用**（`--frozen`ではなく）。`--locked`は`pyproject.toml`と`uv.lock`の不整合（ロック更新忘れ）をCIで検知できるのに対し、`--frozen`は検証なしに`uv.lock`をそのまま使うため、不整合を見逃す。
  - **`test`グループのみインストール**（`dev`はuvが既定で含めるためmaturinも入る。`benchmark`/`docs`はCIでは使わない方針、`pyproject.toml`のコメント参照）。ローカル検証で`wooldridge`パッケージ依存の18テストが`pytest.importorskip`により問題なくskipされることを確認した（123 passed, 18 skipped）。
  - `engine_pybind`はPyO3の`extension-module`機能のみでabi3を使っていないため、Pythonマイナーバージョンごとに別ビルドが必要（マトリクス化が必須）。
- **`engine_pybind-lint`ジョブ**（マトリクス化不要、単一ジョブ）: `cargo fmt -p engine_pybind --check` → `cargo clippy -p engine_pybind --all-targets -- -D warnings`。Issue #23で「`ci_engine.yml`は`-p engine`のみが対象で`engine_pybind`のclippy/fmtは空白になる」と分かっていた分をここで埋めた。バージョン限定`cfg`（`Py_3_XX`等）を使っていないためPythonマトリクス不要と確認済み（`grep`で該当パターンなし）。
- **`pip-audit`ジョブ**（Python 3.12固定、マトリクス不要）: `uv sync --locked --group test` → `uv run --with pip-audit pip-audit`。
  - **判断: `test`グループのみを対象**。本ワークフローが実際にインストールする依存と完全一致させる方針（`benchmark`/`docs`はCIでは使わない）。ローカル検証時点（2026-07-19）でどちらの範囲でも脆弱性は検出されなかった。
  - `maturin develop`（Rustツールチェーンのセットアップ含む）はこのジョブでは省略した。pip-auditは依存パッケージの監査のみが目的で、`econometricsmodels`自体はPyPI未公開のため常にスキップされ、extension moduleがビルド済みかどうかは監査結果に影響しないことをローカルで確認した。
- **トリガー**: `python_package/**`・`engine_pybind/**`・`pyproject.toml`・`uv.lock`・`tests/api_tests/**`・ワークフローファイル自体。
  - **`engine/**`は含めない**（issueの記載通り）。`maturin develop`は`engine`もコンパイルするため`engine`単体の変更がPython統合テストに影響しうるが、`engine`単体の変更は`ci_engine.yml`の`cargo test -p engine`である程度カバーされる前提とし、CLAUDE.md 9章の「対応するパス配下の変更のみでトリガー」方針を文字通り優先した。
  - **`tests/api_tests/**`はissueに明記されていなかったが追加した**。このワークフローの目的が`pytest tests/api_tests`の実行である以上、テストファイル自体の変更でトリガーされないのは明確な抜け穴（新規テスト追加がCIで一度も走らない状態になる）と判断し、確認なしで含めた。
- Issue #23と同様、ツールチェーンセットアップは`actions-rust-lang/setup-rust-toolchain`、Python/uvセットアップは`astral-sh/setup-uv`（`python-version`入力でマトリクス対応）を使用。全アクションをコミットSHAで固定。

## CI: cd_release.yml作成（Issue #25で実装済み）

`.github/workflows/cd_release.yml`を新規作成し、Linux(manylinux x86_64)・macOS(Apple Silicon/Intel)・Windows(x64)向けwheelビルドの継続的な成功確認を自動化した。0.1.0のプレリリース段階のため、PyPI公開ステップは含めない（ビルドとartifactアップロードまで）。

- **土台に`maturin generate-ci github`の出力を使用**: `uv run maturin generate-ci github -m engine_pybind/Cargo.toml --platform manylinux --platform windows --platform macos`で公式テンプレートを生成し、それを元にCLAUDE.md 12章のサポート対象（Linux x86_64のみ・Windows x64のみ・macOS両アーキ）に絞り込んだ。生成テンプレートにはLinuxのx86/aarch64/armv7/s390x/ppc64le、Windowsのx86/aarch64も含まれていたが、CLAUDE.mdが明示していない組み合わせのため削除した。
- **ビルド対象Pythonバージョンを明示指定**: `--find-interpreter`ではなく`-i python3.12 -i python3.13 -i python3.14`を使用。`--find-interpreter`はmanylinuxコンテナ内にある未サポートバージョン・free-threaded版まで全て検出してビルドしてしまうため、CLAUDE.md 12章がサポート対象と定める3バージョンに限定した。`-i python3.14`単体でのローカルビルドを実際に行い、`econometricsmodels-0.1.0-cp314-cp314-manylinux_2_34_x86_64.whl`が生成されることを確認済み。
- **macOSランナー名は生成テンプレートの値をそのまま採用**（`macos-15-intel`=x86_64、`macos-latest`=aarch64）。GitHub Actionsのランナー命名・世代交代はmaturin側が追従しているため、自前で決め打ちしない方針とした。
- **トリガー: タグpush（`v*`）+ `workflow_dispatch`のみ**（ユーザー確認済み）。マルチOSビルド（Windows/macOS含む）はCI時間コストが高いため、`ci_python.yml`のようにPR毎には回さず、リリース直前の確認用として位置づけた。
- **将来のPyPI公開ステップは本ワークフローに`release`ジョブとして追加する方針**（ユーザー確認済み）。`maturin generate-ci`の生成テンプレートに同梱されていた`release`ジョブ（`needs: [linux, windows, macos, sdist]`、タグpush時のみ実行、trusted publishing）をそのまま参考にできる。
- **`sdist`ジョブも追加**（issue本文には明記されていなかったが、`maturin generate-ci`の標準構成であり、将来の`release`ジョブが依存する前提のビルド成果物のため含めた）。ローカルで`maturin sdist`を実行し正常に生成されることを確認済み。
- **発見: `Cargo.toml`にパッケージメタデータ不足の警告**。`maturin sdist`実行時に`engine`・`engine_pybind`両方で「manifest has no description, documentation, homepage or repository」という警告が出た。Issue #32（公開前チェックリスト）に追記した。
- **副次的な発見・修正: `actions/checkout`のバージョン不一致**。Issue #23・#24実装時に確認せず`v4`を使っていたが、実際の最新は`v7.0.0`だった。本issue実装時に確認したうえで、`ci_engine.yml`・`ci_python.yml`・`cd_release.yml`全てで`v7.0.0`に統一した（ユーザー確認済み）。他の固定アクション（`actions-rust-lang/setup-rust-toolchain`・`taiki-e/install-action`・`astral-sh/setup-uv`）は再確認の結果、既に最新だった。

## 未確定（実装時判断でよい）

- 特異性判定の相対閾値の具体式
- `SingularMatrix`のエラーメッセージを「定数項が重複している可能性があります」等、状況に応じて分岐させるか（優先度低、後回し可）
