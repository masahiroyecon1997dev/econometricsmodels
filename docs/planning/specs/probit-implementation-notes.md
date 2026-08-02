# Probit 内部実装ノート（数式・実装判断）

`docs/planning/specs/`配下。`nonlinear-api-design.md`・`nonlinear-implementation-notes.md`（nonlinear系統共通の設計・実装判断）とは別に、**Probit固有の数式導出・実装判断**をまとめる。`logit-implementation-notes.md`と同じ位置づけ。

## データ構造（Issue #70で実装済み）

`engine/src/nonlinear/probit.rs`の`ProbitInput::from_columns`。`engine::nonlinear::logit::LogitInput::from_columns`と同型（フィールド構成・切片列自動追加・次元検証ロジックとも同一）。`MleError`（`nonlinear/common.rs`）をそのまま再利用し、Probit固有のエラーバリアントは追加していない。

`y`が{0.0, 1.0}の二値であることの検証は、`LogitInput`と同様このIssueのスコープ外（次元検証のみ）。尤度・スコア・Hessianを実装する後続Issueで`validate_binary_y`（`nonlinear/common.rs`、Logit/Probit共通）による検証を追加する予定。

現時点で`ProbitInput`は`LogitInput`と完全に同型（フィールド・ロジックとも差分なし）。後続Issue（尤度・スコア・Hessian）でリンク関数がロジスティック関数`Λ(z)`から標準正規分布のCDF`Φ(z)`・PDF`φ(z)`に置き換わるが、これは`ProbitProblem`（`LogitProblem`相当）側の計算ロジックの差であり、`ProbitInput`自体の構造には影響しない見込み。仮に構造にも差分が必要と判明した場合は、この節を更新すること。

## 尤度・スコア・Hessian（Issue #71で実装済み）

### 数式

`z_i = x_i'θ`、`Φ`・`φ`を標準正規分布のCDF・PDFとする。観測`i`の対数尤度への寄与は

```
ℓ_i(θ) = y_i log Φ(z_i) + (1-y_i) log Φ(-z_i)
```

（`Φ(-z)=1-Φ(z)`を使用）。`q_i = 2y_i-1 ∈ {-1,+1}`とおくと`ℓ_i(θ) = log Φ(q_i z_i)`という同値な形に書き換えられる（`y_i=1`なら`q_i=1`で`logΦ(z_i)`、`y_i=0`なら`q_i=-1`で`logΦ(-z_i)`に一致）。

`λ_i = q_i φ(q_i z_i)/Φ(q_i z_i)`（一般化残差、逆ミルズ比に相当）とおくと:

- スコア（対数尤度の1階微分）: `∂ℓ/∂θ = Σᵢ λᵢxᵢ = X'λ`
- Hessian（対数尤度の2階微分）: `∂²ℓ/∂θ∂θ' = -Σᵢ λᵢ(λᵢ+zᵢ)xᵢxᵢ' = -X'WX`（`W = diag(λᵢ(λᵢ+zᵢ))`）

導出は`probit.rs`のモジュールdocコメント参照（`g(u)=φ(u)/Φ(u)`の微分`g'(u)=-ug(u)-g(u)²`を経由）。`λᵢ(λᵢ+zᵢ) > 0`が常に成り立つ（プロビットの対数尤度が大域的に凹であることの根拠）ため、`X'WX`は常に正定値。Logitの`-X'WX`（`W=diag(pᵢ(1-pᵢ))`）と同じ形だが、`W`の中身がLogitより複雑（Issue本文が警告していた「逆ミルズ比に類する項」に相当）。

### 符号規約

Logit（`logit-implementation-notes.md`「符号規約」節）と同じ。`CostFunction::cost = -ℓ(θ)`、`Gradient`/`Hessian`も`-ℓ`の1階・2階微分。`ProbitProblem::scores()`は符号反転しない生のスコア`sᵢ=λᵢxᵢ`を返す。

### 数値安定化について（Logitとの違い）

`Φ(q_i z_i)`・`φ(q_i z_i)`は`statrs::distribution::Normal`の`cdf`/`pdf`をそのまま使う。`1-Φ(z)`を手動計算せず常に`Φ(q_i z_i)`の形（`q_i`で符号を吸収）で評価するため、`statrs`の`cdf`実装（`erfc`ベース）が両裾で提供する精度をそのまま活かせる（手動`1.0 - cdf(z)`で生じる桁落ちを回避）。

Logitの`softplus`のような「対数を経由してもアンダーフローしない」変形は用意していない（`statrs`に`Normal`用の`ln_cdf`が存在しないため）。`z_i`が極端な値（完全分離に近いデータ等）で`Φ(q_i z_i)`が0にアンダーフローすると`cost`が`+inf`になりうる。Logitの完全分離対応（勾配ノルム収束判定のアンダーフロー対応）が`LogitEstimator::fit`実装（Issue #56）より後の別Issueで対応された経緯があるのと同様、本Issue（#71）のスコープは尤度・スコア・Hessianの数式が閉じた形・数値微分と一致することの検証までとし、極端な入力での頑健性は`ProbitEstimator::fit`実装以降の別Issueで必要に応じて対応する（ユーザーへの確認は行わず、Logitの前例に倣ってスコープを決定した）。

### テスト

`engine/src/nonlinear/probit.rs`の`#[cfg(test)] mod tests`に実装。Logitの対応するテスト（5種）と同じ構成。

- **閉じた形の解析解**: `θ=0`のとき全観測で`z_i=0`となり`Φ(0)=0.5`・`φ(0)=1/√(2π)`から`cost`（`4*ln(2)`、`Φ(0)=0.5`がLogitの`logistic(0)=0.5`と同じ値のため一致）・`gradient`・`hessian`・`scores`が閉じた形（`c=√(2/π)`を使った式）で手計算できる。
- **`scores`の総和が`-gradient`に一致すること**: Logitと同じ理由で`θ=[0.3,-0.2]`で検証。
- **数値微分との比較**: `gradient`を`cost`の中心差分（`h=1e-6`）、`hessian`を`gradient`の中心差分（`h=1e-5`）と比較。
