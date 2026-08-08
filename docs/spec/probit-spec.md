# Probit 仕様書

Probit（二項プロビット回帰、最尤推定）の確定済み仕様。`engine/src/nonlinear/probit.rs`（共通基盤は
`engine/src/nonlinear/common.rs`）・`engine_pybind/src/nonlinear/probit.rs`・
`python_package/econometricsmodels/nonlinear/probit.py`として実装済み。Logitと共通基盤・API面を
共有する設計のため、共通する式・理由は[`logit-spec.md`](./logit-spec.md)を参照し、本ドキュメントでは
Probit固有の差分のみを記載する。

## 1. API引数

3層構成: `Probit(data, y, x, options).fit() -> ProbitResults` → `fit_probit(...) -> ProbitResult` →
`ProbitEstimator::fit`。`ProbitOptions`は`LogitOptions`とフィールド単位で完全に同一（型・デフォルト値
とも[`logit-spec.md`](./logit-spec.md)1章の表を参照）。

- `y`の値域検証（`{0.0, 1.0}`の完全一致）・`n<=k`/`k==0`の検証・`"const"`列衝突検証はLogitと同じ
  （共有インフラ、`nonlinear::common::validate_binary_y`等）。

## 2. 結果構造体

`ProbitResult`はフィールド構成が`LogitResult`と同一（`params`/`std_errors`/`z_stats`/.../`cov_type`）。
`df_model=k-1`固定・`log_likelihood_null`の非入れ子性等、[`logit-spec.md`](./logit-spec.md)2章の設計
判断をそのまま踏襲する。

## 3. 内部実装の計算仕様

### 3.1 尤度・スコア・Hessian

`z_i=x_i'θ`、`Φ`・`φ`を標準正規分布のCDF・PDFとする。`q_i=2y_i-1∈{-1,+1}`とおくと対数尤度は
`ℓ_i(θ) = log Φ(q_i z_i)`という同値な形に書ける。一般化残差（逆ミルズ比に相当）
`λ_i = q_i φ(q_i z_i)/Φ(q_i z_i)`を使うと:

- スコア: `∂ℓ/∂θ = X'λ`。Hessian: `∂²ℓ/∂θ∂θ' = -X'WX`（`W=diag(λᵢ(λᵢ+zᵢ))`）。`λᵢ(λᵢ+zᵢ)>0`が
  常に成り立つ（対数尤度が大域的に凹であることの根拠）。

符号規約はLogitと同じ（`CostFunction::cost=-ℓ(θ)`、`scores()`は符号反転しない生のスコア）。

**数値安定化（`U_CLAMP`、Logitとの重要な違い）**: `λ_i=φ(u)/Φ(u)`（`u=q_i z_i`）は`|u|≳39`で`φ`・`Φ`
がともに0にアンダーフローし`0.0/0.0`のNaNになる。Logitの`logistic`/`softplus`にはこのリスクが無い
（有限の`z`ではどれだけ極端でもNaNを産まない）ため、Probit固有の対策が必要になった。R
（`stats::binomial(link="probit")`）の方式を採用し、`u`を`φ`/`Φ`評価前に
`[-U_CLAMP, U_CLAMP]`（`U_CLAMP=8.125_890_664_701_908 = -Φ⁻¹(f64::EPSILON)`、コンパイル時定数）に
クランプする単一の関所`clamped_pdf_cdf`を実装し、`cost`/`gradient`/`hessian`/`scores`すべてに
一貫して適用する（statsmodelsは`score`/`loglike`のみクリップし`hessian`は非対称にクリップしていない
既知の実装ギャップがあるが、本実装はこの非対称性を避ける）。

### 3.2 最適化・収束判定

`LogitEstimator::fit`と同じ設計（標準化空間での最適化、`method`に関わらず収束点のHessianを解析的に
評価、`SeparationSuspected`による完全分離下のアンダーフロー対策を共有）。近似解析解
（切片のみモデル、`Φ(θ̂)=ȳ`すなわち`θ̂=Φ⁻¹(ȳ)`）で検証している。

収束判定`tol`の既定値`1e-6`は、Logitと同じ結論（通常データでは高精度に一致、`near_separation`
境界ケースのみ`tol=1e-6`だと相対誤差最大4.4e-8とわずかに超過し`tol=1e-8`で解消。既定値は変更しない）
に至った。`near_separation`の較正値はリンク関数ごとに異なる（`Φ`は`Λ`より裾が薄く同じベータ値でも
速く飽和するため、Logitの`beta1=20`ではなく`beta1=10`を採用）。

### 3.3 標準誤差

`CovType`（`Classical`/`Opg`/`Hc0`/`Hc1`/`Cluster`）・計算式・エラー型（`SingularHessian`/
`SingularOpgMatrix`/`MissingClusterColumn`/`InsufficientClusters`）は[`logit-spec.md`](./logit-spec.md)
3.3節と共通（`opg_cov_params`/`sandwich_cov_params`/`cluster_cov_params`を共有インフラとしてそのまま
再利用、Probit固有の新規計算は無い）。

**切片のみモデルの閉じた形（Fisher情報量）**: MLEの一階条件`Σᵢλᵢ=0`を使うと、収束点での観測情報行列は
`H(θ̂) = Σᵢλᵢ² = n・φ(θ̂)²/(ȳ(1-ȳ))`に単純化でき、`Var(θ̂) = ȳ(1-ȳ)/(n・φ(θ̂)²)`という解析解を持つ
（テストでこの式を独立に検算している）。

### 3.4 適合度統計量

`log_likelihood`はProbit固有（`Σᵢ log Φ(qᵢzᵢ)`、`clamped_pdf_cdf`経由で数値安定化）だが、
`log_likelihood_null`・`lr_statistic`・`lr_p_value`・`pseudo_r_squared`・`aic`/`bic`は
[`logit-spec.md`](./logit-spec.md)3.4節の共通実装（`goodness_of_fit`）をそのまま使う（切片のみモデルの
MLEが`Φ(θ̂)=ȳ`を満たすため、この計算がリンク関数に依存しないという性質はLogitと共通）。

### 3.5 限界効果

`dy/dx_j = φ(x_i'θ)θ_j`（Logitの`p(1-p)θ_j`とは異なる）。`w`の勾配は`φ'(z)=-zφ(z)`を使い
`s_m=-zφ(z)xₘ`と導出する。`at="overall"`/`"mean"`/`"median"`の代表点方式、デルタ法標準誤差、
`column_means`/`column_medians`/ヤコビアン計算・共通化（`nonlinear/common.rs`）は
[`logit-spec.md`](./logit-spec.md)3.5節と同一。`w=φ(z)`は`Φ`で割らない単独のPDF評価のため、`z`が
極端でも滑らかに0へ収束するのみで`U_CLAMP`によるクランプは不要（`cost`/`gradient`/`hessian`が使う
`λ=φ(u)/Φ(u)`とは異なり0除算のリスクが無い）。

### 3.6 predict() / pred_table()

`predict()`は`p_i=Φ(x_i'θ)`をそのまま計算する（Logitの`Λ`を`Φ`に置き換えたのみ）。`pred_table`の
計算本体はリンク関数を参照しないため`common.rs`の共有関数をそのまま使う（[`logit-spec.md`](./logit-spec.md)
3.6節参照）。in-sample限定、out-of-sample非対応もLogitと同じ。

### 3.7 engine_pybind: エラー変換

[`logit-spec.md`](./logit-spec.md)3.7節のマッピング表と同一（`MleError` → `PyErr`はLogit/Probit/Tobit
共通の`mle_error_to_pyerr`を使う）。

### 3.8 テスト

- 許容誤差はLogitと同じ基本方針（statsmodels主リファレンス`RTOL=1e-8`、Rクロスチェック
  `RTOL=2e-4`）。個別に緩めた項目もLogitと同種の性質: 限界効果`std_err`（`RTOL=1e-3`）・p値
  （`ATOL=5e-5`）に加え、Wooldridge実データ（`mroz`）のクラスターロバストSE（`cluster_col="city"`、
  G=2）は`RTOL=2e-3`（合成データのクラスターケースより大きいが、実データ・クラスタ数境界・相関の
  強い説明変数が重なる境界的なケースのため）。
- **`cov_type="hc1"`/`"opg"`の既知の欠落はLogitと同様**（statsmodelsのdiscrete modelでの非対応、
  対処もRを主リファレンスにする点まで同じ、[`logit-spec.md`](./logit-spec.md)3.8節参照）。
- **Rの`glm()`既定共分散が非正準リンクで「期待情報行列」を返す問題（重要）**: Rの`glm()`の
  `vcov()`（および`sandwich::vcovHC`/`vcovCL`が内部で使う`bread.glm()`）はIRLS（Fisher scoring）の
  作業重みに基づく**期待情報行列**を計算する。Logit（binomial族の正準リンク）では期待情報行列と
  観測情報行列（真の対数尤度のHessian）が理論上一致するためこの違いは表面化しないが、Probit
  （非正準リンク）では一致せず、`classical`で最大約2-3%・`hc0`/`hc1`で最大約8%の乖離が生じる。
  本実装・statsmodelsはいずれも観測情報行列を使うため、Rクロスチェックは`glm()`の既定`vcov()`を
  そのまま使わず、本実装と同じ解析式（`W=diag(λᵢ(λᵢ+zᵢ))`）で観測情報行列を明示的に計算し
  （`numDeriv`による数値微分Hessianとの一致で検証済み）、`sandwich(bread.=...)`に渡す方式を採る。
- 合成データセット生成は`generate_binary_choice_dataset(scenario, link)`としてLogitと共有
  （`near_separation`のみリンク関数ごとに`beta1`の較正値が異なる、3.2節参照）。実データは
  Logitと同じ`mroz`。

## 4. 未実装・未対応

- `predict()`/`pred_table()`のout-of-sample対応（Logitと同じ、[`logit-spec.md`](./logit-spec.md)4章）
- `start_params`（ユーザー指定初期値）
- **`U_CLAMP`とNewton法（line searchなし）の相互作用は未検証**: `U_CLAMP`は一般化残差`λᵢ`の
  NaN化のみを防ぐ局所的な保護で、Hessianが使う線形予測子`zᵢ`自体は無制限のまま。理論上は
  悪条件な中間反復でパラメータが大きくジャンプし発散的に増幅する経路がありうるが、最終的に
  NaN化すれば`newton_step`のNaNチェックが`SingularHessian`として偶発的に捕捉する見込み。
  実データで踏むかどうかは未検証。
- **`U_CLAMP`領域での`cost()`/`gradient()`の数学的非整合とBFGS/L-BFGSのline searchへの影響は
  未検証**: クランプ領域では`cost()`は`θ`に対して定数（微分ゼロ）のはずだが、`gradient()`は
  クランプ後の`λᵢ`（有限だが非ゼロ）を返すため真の微分と一致しない。この非整合を解消する
  「修正」（クランプ領域で`gradient`もゼロにする）は、完全分離に近いデータで勾配ノルム基準の
  収束判定を誤検知させる別のバグを誘発しうるため、あえて行わない設計上の判断（意図的に維持）。
  line searchが受理可能なステップを見つけられない、または不適切なステップを受理する可能性は
  理論上あるが未検証。
- `SEPARATION_PARAM_NORM_THRESHOLD=100.0`（Logitの実測に基づく較正値）がProbitのリンク関数
  （テイルの減衰特性が異なる）でも同程度に適切かは未較正
