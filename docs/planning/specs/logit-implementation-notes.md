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

### `y`の値域検証（単位区間`[0,1]`）は未実装、担当Issue未定

`nonlinear-implementation-notes.md`「Logitのデータ構造」節でIssue #54時点では「B2（Issue #55）で追加予定」としていたが、Issue #55の本文スコープ（`LogitProblem`のCostFunction/Gradient/Hessian実装、完了条件は尤度・スコア・Hessianの正しさの検証のみ）には含まれていなかったため見送った。続くIssue #56（`LogitEstimator::fit`の骨格）でも、本文スコープ（Newton-Raphsonでの最適化・収束判定・`confidence_level`検証のみ）には含まれておらず、`logit-probit-issue-breakdown.md`のB3〜B16のいずれにも明示的な担当Issueが無いことを確認済み（見送ったまま）。OLSが`InsufficientObservations`等のバリデーションを`OlsInput::from_columns`ではなく`OlsEstimator::fit`側で行っているのと同じ役割分担（データ構造自体はバリデーションを最小限にし、`fit()`の入口でまとめて検証する）で、`fit()`側に追加する想定。

### テスト

`engine/src/nonlinear/logit.rs`の`#[cfg(test)] mod tests`に実装。

- **閉じた形の解析解**: `θ=0`のとき全観測で`p_i=0.5`となり、`cost`（`4*ln(2)`相当）・`gradient`（`Σ(0.5-y_i)x_i`）・`hessian`（`0.25*X'X`）・`scores`（`(y_i-0.5)x_i`）が指数関数の評価を経ずに手計算できる。この性質を使い、実装から独立した期待値で検証した。
- **`scores`の総和が`-gradient`に一致すること**: `scores`（対数尤度の生のスコア）と`gradient`（`-ℓ`の勾配）は符号が逆なので、観測方向に合計すると`Σsᵢ = -gradient(θ)`が成り立つはず。これを`θ=0`でない一般の点（`θ=[0.3,-0.2]`）で検証し、符号規約の実装漏れ・取り違えを検出できるようにした。
- **数値微分との比較**: `θ=[0.3,-0.2]`（非自明な点）で、`gradient`を`cost`の中心差分（`h=1e-6`）と比較、`hessian`を`gradient`の中心差分（`h=1e-5`）と比較。解析解が閉じた形で書けない一般の点でも導関数の実装が正しいことを確認する。
- **数値安定化のテスト**: `logistic`/`softplus`を`z=±1000`で評価し、NaN/Infにならず有限の値を返すことを確認。

## Newton-Raphsonでの最適化・収束判定（Issue #56で実装済み）

`engine/src/nonlinear/logit.rs`の`LogitEstimator::fit`。A2（Issue #52）の`run_solver`を`Method::Newton`固定で呼び出し、`converged`/`n_iter`を含む結果を返す骨格。

- **`start_params`（ユーザー指定初期値）は見送り**: `nonlinear-api-design.md`7章では確定オプションだが、`logit-probit-issue-breakdown.md`のB3〜B16のどのIssueにも実装担当が明示されていなかったため、ユーザーに確認の上、本Issueでは実装しない方針とした（初期値は常にゼロベクトル固定）。標準化空間への変換ロジック（`destandardize_params`の逆方向）が追加で必要になる点も判断材料にした。将来実装する場合は別Issueで対応する。
- **`max_iter`の検証を`fit()`に追加**: `MleError::InvalidMaxIter`はIssue #51で型としては定義済みだったが、実際に検証する箇所が無かった。Issue #56本文の「`confidence_level`の範囲検証等」の「等」に該当すると判断し、`max_iter <= 0`の検証をここで実装した。
- **設計行列の標準化**: `nonlinear-implementation-notes.md`「収束判定のtol」で確定済みの方針（`standardize_columns`で標準化した空間で最適化し、`destandardize_params`で元のスケールへ逆変換）をそのまま適用。
- **標準誤差・z値・p値・信頼区間・適合度統計量は未実装**（`confidence_level`は範囲検証のみ行い、値自体はまだ使わない。B5以降で`fit()`を拡張してこれらを追加する想定）。
- **`n<=k`の検証根拠はOLSと異なる**: 閾値の式（`n<=k`で`CommonError::InsufficientObservations`）自体はOLSと同じだが、根拠は異なる。OLSでは残差自由度`n-k`が0以下だと分散推定が原理的に不可能という数学的必要条件だが、LogitのようなMLEベースのモデルでは`n<=k`はほぼ確実に完全分離（perfect separation、ある説明変数の値で`y`が完全に分かれてしまいMLEが発散して存在しなくなる状態）を引き起こす経験則としての安全側の判断（rust-reviewer指摘、Issue #56で対応）。後続のProbit/Tobit実装でもこの閾値をそのまま踏襲する方針とする。

### テスト

- **既知の解析解**: 説明変数なし（切片のみ）のLogitは一階条件`Σy_i - n*p = 0`から`p = ȳ`、`θ̂ = ln(ȳ/(1-ȳ))`という閉じた形の解析解を持つ。この性質を使い、Newton法が実際にこの値へ収束すること（`converged=true`、反復回数が妥当）を検証した。
- **バリデーションエラー**: `confidence_level`範囲外・`max_iter<=0`・`n<=k`のそれぞれで対応するエラーを返すことを検証。
- **収束判定の分岐**: `max_iter`を極端に小さくした場合に`raise_on_non_convergence=true`で`NonConvergence`を返すこと、`false`で`converged=false`の結果を返すことを検証。
- **`fit()`経由での`SingularHessian`伝播**: 完全分離（収束の挙動に依存し決定的に再現しづらい）ではなく、完全な多重共線性（`x2=2*x1`）を使うことで、θ=0時点のHessianが構造的に特異になり決定的に`MleError::SingularHessian`を再現できるデータセットで検証した（rust-reviewer指摘、Issue #56で対応）。
