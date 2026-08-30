# WLS 仕様書

WLS（Weighted Least Squares）の確定済み仕様。`engine/src/linear/wls.rs`・
`engine_pybind/src/linear/wls.rs`・`python_package/econometricsmodels/linear/wls.py`として
実装済み。OLSに強く依存する設計のため、共通する式・理由は[`ols-spec.md`](./ols-spec.md)を参照し、
本ドキュメントではWLS固有の差分のみを記載する。

## 1. API引数

3層構成: `WLS(data, y, x, weight, options).fit() -> WlsResults`（python_package）→
`fit_wls(data, y, x, weight, options) -> WLSResult`（engine_pybind）→
`OlsInput::from_columns_weighted` + 既存`OlsEstimator::fit`（engine、無変更で再利用）。

- `weight: str`は`y`/`x`と同格の**必須のトップレベル引数**（`OLSOptions`側には置かない）。
  理由: `cluster_col`/`time_col`は`cov_type`に応じた条件付き・デフォルトありの「推定方法の設定」
  だが、`weight`はモデルそのものを規定する必須データであり、性質が異なる。この分類はFE
  （`entity_id`）・IV（`instruments`）等、今後の必須データ列にも適用する。
- 専用の`WLSOptions`型は新設せず、`OLSOptions`をそのまま使う（`cov_type`/`include_intercept`/
  `confidence_level`/`cluster_col`/`hac_lags`/`time_col`の意味論はOLSと完全に同じ）。
- 重みはanalytic weight（分散の逆数に比例、正規化不要）。frequency weight/probability weightは
  対象外。
- **重みの検証**: 0以下（0を含む）・NaN・無限大は常にエラー（`ValidationError`）とし、該当観測を
  自動的に落とすことはしない（OLSの欠損値ポリシーと同じ考え方、[`ols-spec.md`](./ols-spec.md)
  「API引数」）。NaN/Infは既存の`column_extraction::extract_f64_column`が`weight`列にも適用される
  ことで検出されるため、追加実装が必要なのは0以下の値の検証のみ。ゼロ重みの許容（観測除外の手段
  としての活用）は将来の別issue。
- `weight`は`y`/`x`と重複してはならない（`weight == y`、`x.contains(weight)`はエラー）。
  `include_intercept=True`時の`"const"`列衝突チェック等、OLSの検証はそのまま踏襲する。

## 2. 結果構造体

`WLSResult`は`OLSResult`とフィールド構成が同一（`params`/`std_errors`/.../`aic`/`bic`）だが、
型としては別に定義する（重み付き残差等、WLS固有フィールドが将来追加される可能性があり、
`OLSResult`との乖離リスクをOptions（共有）より高く見積もったため）。

- `residuals`は**元スケール（unweighted）の残差** `ε_i = y_i - x_i'β̂`を公開する（statsmodelsの
  `.resid`相当）。理由: 残差プロット等の診断用途では元スケールの方が直感的で、OLSの`residuals`
  とも定義が揃う。重み付き残差（`.wresid`相当、`ε̃_i = sqrt(w_i)ε_i`）はPhase1では公開しない。

## 3. 内部実装の計算仕様

### 3.1 `sqrt(w)`変換によるOLS計算式の再利用

`OlsEstimator::fit`（正規方程式ソルバー・標準誤差・適合度統計量の計算本体）は無変更のまま再利用
する。変換が必要なのは設計行列の組み立て（`OlsInput::from_columns_weighted`）のみ。

**罠（切片列の重み付け）**: `x_columns`とyを先に`sqrt(w)`倍してから既存の`from_columns`を呼ぶ実装
は誤り。`from_columns`が内部で追加する切片列（すべて1.0）が重み付け前のまま残ってしまい、
`include_intercept=true`時の設計行列が数学的に不正になる（切片列も`sqrt(w_i)`倍が必要）。
重み変換は設計行列の組み立てそのものの中で行う（`weights: Option<&[f64]>`を取るヘルパーに
`from_columns`本体を委譲し、`None`のときは現状と全く同じ、`Some`のときは切片列も含め各行に
`sqrt(w_i)`を掛ける）。

この設計により、**「重みが全て1のときWLSはOLSと数値的に完全一致する」という不変条件が同じ
コードパスを通ることで構造的に保証される**（テストは回帰検知として位置づけられる）。

### 3.2 エラー型

WLSは`OlsEstimator::fit`をそのまま呼ぶため、既存の`LeastSquaresError`（`engine::linear::common`）
バリアントがそのまま当てはまる。重み固有のエラーとして以下の2バリアントを追加し、OLS/WLS共通の
エラー型として使い続ける（専用の`WlsError`型は新設しない）。

- `WeightDimensionMismatch { y_rows, weight_rows }`: 重み配列と`y`の行数不一致（防御的チェック）。
- `NonPositiveWeight { row, weight }`: 重みが0以下（NaN含む）。

### 3.3 標準誤差

classical / HC0〜HC3 / HAC / cluster とも、変換後データ（$\tilde x_i=\sqrt{w_i}x_i$,
$\tilde y_i=\sqrt{w_i}y_i$, $\tilde\varepsilon_i=\sqrt{w_i}\hat\varepsilon_i$）に
[`ols-spec.md`](./ols-spec.md)「標準誤差」と同じ式をそのまま適用する（新しい計算式の導出は不要。
statsmodelsの`WLS`も内部的に同じ変換方式`wexog=sqrt(weights)*exog`で実装されており、この設計は
そのままstatsmodelsとの数値一致が期待できる）。

- HC0の$\hat\Psi=\sum_i\tilde\varepsilon_i^2\tilde x_i\tilde x_i^\top=\sum_i w_i^2\hat\varepsilon_i^2 x_i x_i^\top$
  のように、重みが2乗で効く（残差と設計行列の両方に$\sqrt{w_i}$がかかるため）。
- クラスターのグループ分け自体（`cluster_col`によるグルーピング）は重み変換の影響を受けない
  （グループ内で合計する対象が変換後の値になるだけ）。小標本補正・自由度の扱いもOLSと同じ。
- HAC・cluster・時間順序（`time_col`）を含め、ラグ選択式・小標本補正・自由度切替はすべて
  観測数`n`・クラスター数`G`のみに依存し重みには依存しないため、OLSと同じ式・同じオプションを
  そのまま使う。

### 3.4 適合度統計量（OLSと異なり要注意）

`f_statistic`・`f_p_value`・係数・標準誤差・t値・p値・信頼区間は変換後データへの代入のままで
正しい。一方、**`r_squared`・`r_squared_adj`・`log_likelihood`（→`aic`/`bic`）は変換後データに
OLSの計算式をそのまま適用するだけでは誤りになる**（statsmodelsとのクロスチェックで、R²相対誤差
0.2〜1%程度、対数尤度に加法的なずれが実際に発生することを確認済み）。

**R²**: `SST`（切片ありの場合、centered）は変換後$\tilde y$の単純平均ではなく、**元の$y$の重み付き
平均** $\bar y_w=\sum w_i y_i/\sum w_i$ を使う。

$$
\mathrm{SST} = \sum_i w_i (y_i - \bar y_w)^2 \quad\text{（切片あり）}, \qquad
\mathrm{SST} = \sum_i w_i y_i^2 \quad\text{（切片なし、uncentered。こちらは代入のままで正しい）}
$$

**対数尤度**: `sqrt(weight)`変換のヤコビアンに由来する補正項が必要。

$$
\ell = -\frac{n}{2}\Big(\ln(2\pi) + \ln(\mathrm{SSR}/n) + 1\Big) + \frac{1}{2}\sum_i \ln w_i
$$

（第1項はOLSの対数尤度の式に変換後の`SSR`・`n`を代入したもの、第2項$\frac12\sum_i\ln w_i$が
追加の補正項）。AIC/BICはこの$\ell$から通常通り計算する（式自体は不変、$\ell$の値が変わる）。

**実装**: `OlsEstimator`/`OlsInput`自体は重みを一切知らない設計を維持したまま、
`WlsEstimator::fit`側（元の、変換前の`y`・重みにアクセスできる層）で上記5フィールドを計算し直す
`weighted_fit_statistics`関数を`WlsEstimator`に持たせている（`residuals`と同じ「WLS固有の後処理を
`WlsEstimator`層に置く」パターン）。

### 3.5 テスト

- 許容誤差: classical/HC0-3/clusterはOLSと同じ`RTOL_STRICT=1e-8`（Rとの実測でほぼ機械精度）。
  **HACのみOLSより緩い`RTOL_HAC=5e-2`**（OLSは1e-2。実測最大相対誤差約4.3%、重み付けによる
  小標本補正の慣習差の増幅が原因と推測、未調査）。
- 実データセット: `401ksubs`（`fsize==1`の単身世帯サブサンプル、n=2017）、Wooldridge Example
  8.5・8.6と同じ変数構成`nettfa ~ inc + incsq + age + agesq + male + e401k`、重みは`1/inc`
  （`inv_inc`列）。Example 8.6のfeasible GLS（分散モデル自体の推定）は本実装のスコープ外のため
  不採用、既知の重み列を渡す設計に合わせた。
- 合成データセット（`benchmark/linear/datasets.py`の7シナリオ）はOLS実装時から
  `weight`列（heteroskedasticシナリオは`1/sigma_i^2`、他は`uniform(0.5, 1.5)`）を含むため、
  WLS用の追加実装は不要だった。
- `test_wls_fixtures.py`（statsmodels主リファレンス）/ `test_wls_crosscheck.py`（Rクロスチェック）
  の役割分担はOLSと同じ。

## 4. 未実装・未対応

- `predict()`（Issue #132。OLSの`predict(new_data=None)`と同じ設計を適用予定）
