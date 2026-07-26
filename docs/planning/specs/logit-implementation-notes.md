# Logit 内部実装ノート（数式・実装判断）

`docs/planning/specs/`配下。`nonlinear-api-design.md`・`nonlinear-implementation-notes.md`（nonlinear系統共通の設計・実装判断）とは別に、**Logit固有の数式導出・実装判断**をまとめる。OLSの`ols-implementation-notes.md`と同じ位置づけ。

## データ構造（Issue #54で実装済み）

`engine/src/nonlinear/logit.rs`の`LogitInput::from_columns`。詳細は`nonlinear-implementation-notes.md`「Logitのデータ構造」参照。

## 尤度・スコア・Hessian（Issue #55で実装済み）

### 数式

`p_i = Λ(x_i'θ) = 1/(1+exp(-x_i'θ))`（ロジスティック関数）。観測`i`の対数尤度への寄与は

```
ℓ_i(θ) = y_i log(p_i) + (1-y_i) log(1-p_i)
```

`z_i = x_i'θ`とおくと、`log(p_i) = z_i - softplus(z_i)`・`log(1-p_i) = -softplus(z_i)`（`softplus(z) = log(1+exp(z))`）が成り立つため、同値な形

```
ℓ_i(θ) = y_i z_i - softplus(z_i)
```

に書き換えられる。`softplus`形の方が指数関数のオーバーフローを避けやすいため、`CostFunction::cost`はこちらを実装している。

- スコア（対数尤度の1階微分）: `∂ℓ/∂θ = Σᵢ (yᵢ-pᵢ)xᵢ = X'(y-p)`
- Hessian（対数尤度の2階微分）: `∂²ℓ/∂θ∂θ' = -Σᵢ pᵢ(1-pᵢ)xᵢxᵢ' = -X'WX`（`W = diag(pᵢ(1-pᵢ))`）。ロジットの対数尤度は大域的に凹なので、`-X'WX`は（厳密な多重共線性がなければ）常に負定値、`X'WX`は常に正定値。

### 符号規約（`nonlinear/common.rs`の`Hessian`トレイト符号規約修正、Issue #55着手時に発覚）

`argmin`は最小化フレームワークのため`CostFunction::cost = -ℓ(θ)`。`Gradient`/`Hessian`トレイトも`CostFunction`と同じ符号（`-ℓ`の1階・2階微分）で実装する（`run_solver`のdocコメント「`Hessian`トレイトの符号規約」参照、`nonlinear-implementation-notes.md`に詳細な経緯を記録済み）。

- `Gradient::gradient` = `Σᵢ (pᵢ-yᵢ)xᵢ = X'(p-y)`（スコアの符号反転）
- `Hessian::hessian` = `X'WX`（対数尤度のHessian`-X'WX`の符号反転）

`LogitProblem::scores()`（argminのトレイトではない独自メソッド、`cov_type`共通行列演算向け）は符号反転しない生のスコア`sᵢ=(yᵢ-pᵢ)xᵢ`を返す（`nonlinear/common.rs`の`SolverOutput.hessian`と同じく対数尤度そのものの符号。`observed_information_cov_params`等が前提とする符号と一致させる）。

### 数値安定化

- `softplus(z) = z.max(0) + log1p(exp(-|z|))`: `z`が大きい正の値でも`exp(z)`がオーバーフローしない標準的な安定化式。
- `logistic(z)`: `z>=0`なら`1/(1+exp(-z))`、`z<0`なら`exp(z)/(1+exp(z))`と分岐し、`exp`の引数を常に非正に保つ（オーバーフロー回避）。`z`が極端に大きい/小さい場合でも`exp`の結果が0にアンダーフローするだけで、NaN/Infは発生しない。

### `y`の値域検証（単位区間`[0,1]`）は未実装、Issue #56に持ち越し

`nonlinear-implementation-notes.md`「Logitのデータ構造」節でIssue #54時点では「B2（本Issue #55）で追加予定」としていたが、Issue #55の本文スコープ（`LogitProblem`のCostFunction/Gradient/Hessian実装、完了条件は尤度・スコア・Hessianの正しさの検証のみ）には含まれていなかったため、`LogitEstimator::fit`実装時（Issue #56）に見送った。OLSが`InsufficientObservations`等のバリデーションを`OlsInput::from_columns`ではなく`OlsEstimator::fit`側で行っているのと同じ役割分担（データ構造自体はバリデーションを最小限にし、`fit()`の入口でまとめて検証する）。

### テスト

`engine/src/nonlinear/logit.rs`の`#[cfg(test)] mod tests`に実装。

- **閉じた形の解析解**: `θ=0`のとき全観測で`p_i=0.5`となり、`cost`（`4*ln(2)`相当）・`gradient`（`Σ(0.5-y_i)x_i`）・`hessian`（`0.25*X'X`）・`scores`（`(y_i-0.5)x_i`）が指数関数の評価を経ずに手計算できる。この性質を使い、実装から独立した期待値で検証した。
- **`scores`の総和が`-gradient`に一致すること**: `scores`（対数尤度の生のスコア）と`gradient`（`-ℓ`の勾配）は符号が逆なので、観測方向に合計すると`Σsᵢ = -gradient(θ)`が成り立つはず。これを`θ=0`でない一般の点（`θ=[0.3,-0.2]`）で検証し、符号規約の実装漏れ・取り違えを検出できるようにした。
- **数値微分との比較**: `θ=[0.3,-0.2]`（非自明な点）で、`gradient`を`cost`の中心差分（`h=1e-6`）と比較、`hessian`を`gradient`の中心差分（`h=1e-5`）と比較。解析解が閉じた形で書けない一般の点でも導関数の実装が正しいことを確認する。
- **数値安定化のテスト**: `logistic`/`softplus`を`z=±1000`で評価し、NaN/Infにならず有限の値を返すことを確認。
