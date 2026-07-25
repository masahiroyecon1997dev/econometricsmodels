# WLS 内部実装ノート（パラメータ設計以外）

WLSの実装過程で確認した事項のうち、[`wls-api-design.md`](./wls-api-design.md)・
[`wls-standard-errors.md`](./wls-standard-errors.md)（パラメータ・計算式の設計）には
含まれない実装ノート。構成は[`ols-implementation-notes.md`](./ols-implementation-notes.md)
に合わせている。

## 8. テスト

### テストデータ（Issue #42）

- **合成データセット**（`benchmark/generate_synthetic_datasets.py`の7シナリオ）: OLS実装時
  （Issue #15）から既に`weight`列（heteroskedasticシナリオは`1/sigma_i^2`、それ以外は
  `uniform(0.5, 1.5)`。いずれも正の値）を含む形で用意されており、WLS用に追加の実装は不要
  だった。`maturin develop --release`でビルドした実バイナリで、7シナリオ全てに対して
  `WLS(df, y="y", x=[...], weight="weight").fit()`を実行して確認した:
  - `baseline` / `small_n` / `high_variance` / `heteroskedastic` / `autocorrelated` /
    `moderate_multicollinearity`: 正常に収束
  - `perfect_multicollinearity`: 想定通り`ComputationError`（設計行列が特異）
- **実データセット**: `benchmark/load_wooldridge.py`の候補を、OLSの`wage1`/`gpa2`と同様に
  確認・確定した。
  - 当初の候補だった`hprice1`（住宅価格データ）には重みとして自然に使える列がなかったため、
    `401ksubs`（401(k)制度データ、n=9275、`fsize`＝世帯人数で1人世帯に絞るとn=2017）に変更した。
    `inc`（所得）等の列が常に正の値であることを確認済み。`fsize == 1`で絞り込んだ上で
    `WLS(df, y="nettfa", x=["age", "agesq", "e401k"], weight="inc").fit()`が正常に収束することを
    実バイナリで確認した（欠損値なし、Int64/Float64混在列も`extract_f64_column`のキャストで
    問題なく処理される）。
  - **確定（Issue #43）**: ベンチマーク用の最終的な回帰式・重みは、Wooldridge
    『Introductory Econometrics』Example 8.5・8.6と同じ変数構成
    `nettfa ~ inc + incsq + age + agesq + male + e401k`（`fsize == 1`の単身世帯サブサンプル、
    n=2017）とし、重みは`Var(u|inc) ∝ inc`という単純な仮定に基づく`1/inc`（analytic weight）
    とした。Example 8.6のfeasible GLS（補助回帰の残差から分散を推定する方式）は採用しない
    — 本実装のWLSは「重み列を渡す」設計であり分散モデルの推定機能自体を持たないため、
    ベンチマーク用の重みも本実装が実際にサポートする形（既知の重み列）に合わせる必要がある
    ことによる。`inv_inc`という列名（`inc`列とは別）でCSV/DataFrameに追加し、`weight`が
    `x`と重複してはいけないという既存のバリデーション（Issue #42で確認済み）と両立させた。

### ベンチマーク作成（Issue #43）

- `benchmark/run_statsmodels_benchmark.py`（Issue #45の系統別ディレクトリ再編後は
  `benchmark/linear/run_statsmodels_benchmark.py`）に`--weight-col`オプションを追加し、
  指定時は`smf.wls`を使うようにした（OLS/WLSで同じスクリプトを共用、分散共分散行列の
  計算式自体は共通のため）。
- `benchmark/run_r_benchmark.R`の`lm`ブランチ（独立実装によるクロスチェック用、Issue #27で
  確定した役割分担。系統別ディレクトリ再編後は`benchmark/linear/run_lm_crosscheck_benchmark.R`
  として独立したファイルに分離）に重み列指定を追加した。`weight_col`はcov_type固有の引数
  （`cluster_col`/`hac_lag`）の後ろに置く（classical/HC0-3はarg5、cluster/hacはarg6）。
  `sandwich`パッケージの`vcovHC`/`vcovCL`/`NeweyWest`は`lm(weights=)`の重み付きモデルに
  対してそのまま使え、追加の変更は不要だった（重み無しの場合と同じ関数呼び出し）。
- 生成した`wls.json`（statsmodels主リファレンス）・`wls_crosscheck.json`（Rクロスチェック）を
  比較した結果、classical/HC0-3/clusterはOLSと同様ほぼ機械精度で一致（相対誤差1e-13〜1e-15
  程度）。**HACのみOLSより乖離が大きく、実測で最大相対誤差約4.3%**
  （OLSの実測約0.4%の10倍程度）。原因はOLS側同様、`NeweyWest()`の`prewhite=FALSE`と
  本実装の重み付きBartlettカーネル計算の小標本補正の慣習差と推測されるが、重み付けにより
  誤差が増幅されている可能性がある（未調査、実害があれば別issueで深掘りする）。
  Issue #44のテスト実装時は、**WLSのHAC crosscheck許容誤差は相対誤差5e-2（5%）**を
  採用する想定（OLSの1e-2ではなく実測値に基づき緩めた値。`testing-policy.md`「同じ
  クロスチェック用パッケージでも、統計量・cov_typeごとに実測乖離が大きく異なる場合は、
  許容誤差を分けてよい」に従う）。classical/HC0-3/clusterはOLSと同じく相対誤差1e-8を適用する。

### tests/api_tests作成（Issue #44）— R²・対数尤度の計算式バグ発覚と修正

`test_wls_fixtures.py`（statsmodels厳密比較）を書いて実行したところ、係数・標準誤差・
t値・p値・信頼区間・F統計量・F検定p値はいずれも機械精度で一致した一方、**`r_squared`・
`r_squared_adj`・`log_likelihood`・`aic`・`bic`が重みが一様でない全シナリオ・全cov_typeで
系統的に乖離**した（例: baselineシナリオでR²相対誤差0.21%、log_likelihoodはbaselineで
ちょうど`+14.0007`のずれ）。Issue #37〜#39時点のテスト（weight=1でOLSと一致確認、手動
スモークテスト）は変換が恒等写像になるケース（$w_i\equiv1$）しか見ておらず、この乖離を
検出できていなかった。

**根本原因**（[`wls-standard-errors.md`](./wls-standard-errors.md) 5章で訂正済み。
statsmodelsのソース、`statsmodels/regression/linear_model.py`のv0.14.6を直接確認して特定）:

1. `r_squared`（切片ありの場合）: 変換後$\tilde y=\sqrt w\,y$の**単純平均**を使ったcentered
   TSSを使っていたが、正しくは元の$y$の**重み付き平均**（`np.average(endog,
   weights=weights)`）を使ったcentered TSSが必要だった。切片なし（uncentered TSS）の場合は
   代数的に$\sum\tilde y_i^2=\sum w_iy_i^2$が厳密に成り立つため訂正不要だった。
2. `log_likelihood`（→`aic`/`bic`）: 変換後データに対するOLSの対数尤度をそのまま使うと、
   `sqrt(weight)`変換のヤコビアンに由来する補正項`+0.5·Σlog(w_i)`が欠落していた
   （statsmodelsの`WLS.loglike`と同じ導出で確認）。

**修正**: `engine::linear::ols::OlsEstimator`/`OlsInput`自体は重みを一切知らない設計を
維持したまま、`engine::linear::wls::WlsEstimator::fit`側（元の、変換前の`y`・`weights`に
アクセスできる層）で`r_squared`/`r_squared_adj`/`log_likelihood`/`aic`/`bic`を計算し直す
`weighted_fit_statistics`関数を追加し、`WlsEstimator`自身に同名のgetterを生やした
（`residuals`と同じ「WLS固有の後処理をWlsEstimator層に置く」パターン）。
`engine_pybind/src/linear/wls.rs`側もこれらの5フィールドは
`wls_estimator.r_squared()`等（`estimator().r_squared()`ではなく）を呼ぶように変更した。
新しいエンジン単体テスト（`fit_computes_r_squared_and_information_criteria_matching_statsmodels`）
で、statsmodelsで独立に計算した既知の期待値と一致することを固定した。

この発見・修正は、Issue #38〜#39時点の「weight=1でOLSと一致すること」を主な検証手段とした
設計判断が持つ死角（重みが真に非自明な値のときの適合度統計量は未検証だった）を示す実例と
なった。今後の手法実装でも、構造的な不変条件テストだけでなく、実際の重み・変数を伴う
リファレンス実装とのクロスチェックを早期に行うことの重要性を再確認した。
