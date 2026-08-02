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

### 数値安定化について（Logitとの違い、`U_CLAMP`によるクランプ、Issue #71＋#72着手前の追加対応）

`Φ(q_i z_i)`・`φ(q_i z_i)`は`statrs::distribution::Normal`の`cdf`/`pdf`をそのまま使う。`1-Φ(z)`を手動計算せず常に`Φ(q_i z_i)`の形（`q_i`で符号を吸収）で評価するため、`statrs`の`cdf`実装（`erfc`ベース）が両裾で提供する精度をそのまま活かせる（手動`1.0 - cdf(z)`で生じる桁落ちを回避）。

**当初（Issue #71実装時点）**は、Logitの`softplus`のような「対数を経由してもアンダーフローしない」変形を用意せず（`statrs`に`Normal`用の`ln_cdf`が存在しないため）、極端な入力での頑健性を`ProbitEstimator::fit`実装以降の別Issueに先送りする判断をした（Logitの完全分離対応が`LogitEstimator::fit`実装より後の別Issueで対応された前例に倣った）。

しかし、Issue #72（`ProbitEstimator::fit`実装）着手前にrust-reviewerのレビューで再検証したところ、この前例は不正確だったことが判明した。実測では`λ_i = φ(u)/Φ(u)`（`u=q_i z_i`）は`u`が`|u|≳39`で`φ(u)`・`Φ(u)`がともに0にアンダーフローし**`0.0/0.0`のNaN**になる。Logitの`logistic`/`softplus`は有限の`z`ではどれだけ極端でもNaNを産まない設計だったため、これはLogitには無かった質的に異なるリスクだった。加えて、既定手法`Method::Newton`（`FaerNewton`）はline searchなしで`gradient`/`hessian`を直接使うため、Logitで問題になった「(準)完全分離データでの収束判定誤検知」（勾配ノルムのアンダーフロー）よりも緩い条件でこのNaN汚染に到達しうる。

**対応**: statsmodels・R双方の参照実装を調査した上でユーザーに確認し、Rの`stats::binomial(link="probit")$linkinv`方式を採用した。

- **statsmodels** `Probit`: `Φ`の**出力**を`np.clip(cdf, FLOAT_EPS, 1-FLOAT_EPS)`（`FLOAT_EPS=np.finfo(float).eps`）でクリップ。ただし`score`/`loglike`にのみ適用され、**`hessian`には適用されていない**（非対称、statsmodels自体の実装ギャップ）。
- **R** `stats::binomial(link="probit")$linkinv`: `thresh <- -qnorm(.Machine$double.eps)`（`≈8.1259`）で線形予測子`eta`自体を`pnorm`評価**前**に`[-thresh, thresh]`にクランプ。`mu.eta`（density）側も`pmax(dnorm(eta), .Machine$double.eps)`で下駄を履かせる二重の防御。

採用した方式（`probit.rs`の`U_CLAMP`定数・`clamped_pdf_cdf`関数）:

- `u=q_i z_i`を`φ`/`Φ`評価**前**に`[-U_CLAMP, U_CLAMP]`（`U_CLAMP=8.125_890_664_701_908`、Rと同じ`-Φ⁻¹(f64::EPSILON)`。Rとscipy両方で算出し一致を確認済み）にクランプする単一の関所`clamped_pdf_cdf`を実装し、`cost`/`linear_predictor_and_residual`（`gradient`/`hessian`/`scores`が経由）の両方から呼ぶ。
- `Normal::inverse_cdf`は反復計算のためホットパスで毎回呼ばず、コンパイル時定数としてハードコード。
- statsmodelsの非対称性（`hessian`未クリップ）を避け、`cost`/`gradient`/`hessian`/`scores`すべてに同じ閾値を一貫して適用する。

`cost_gradient_hessian_stay_finite_for_extreme_linear_predictor`テスト（`z=1000`相当）でNaN化しないことを直接検証済み。

### テスト

`engine/src/nonlinear/probit.rs`の`#[cfg(test)] mod tests`に実装。

- **閉じた形の解析解**: `θ=0`のとき全観測で`z_i=0`となり`Φ(0)=0.5`・`φ(0)=1/√(2π)`から`cost`（`4*ln(2)`、`Φ(0)=0.5`がLogitの`logistic(0)=0.5`と同じ値のため一致）・`gradient`・`hessian`・`scores`が閉じた形（`c=√(2/π)`を使った式）で手計算できる。
- **`scores`の総和が`-gradient`に一致すること**: Logitと同じ理由で`θ=[0.3,-0.2]`で検証。
- **数値微分との比較**: `gradient`を`cost`の中心差分（`h=1e-6`）、`hessian`を`gradient`の中心差分（`h=1e-5`）と比較。
- **極端な線形予測子でも有限であること**: `z=1000`相当（`U_CLAMP`が無ければ`λ`がNaNになる領域）で`cost`/`gradient`/`hessian`/`scores`が有限値を返すことを確認。

## Newton-Raphsonでの最適化・収束判定（Issue #72で実装済み）

`ProbitEstimator::fit`。`LogitEstimator::fit`の骨格実装（Issue #56、`Method::Newton`固定・`cov_type`/`method`分岐なし）と同じスコープ・同じ設計。`params`/`converged`/`n_iter`のみを保持し、標準誤差・適合度統計量等は後続Issueに持ち越す。

`LogitEstimator::fit`と異なり、以下は当初から実装している（Logitでは事後的な別Issue・別コミットで追加されたが、既に確立済みのパターン・共有インフラのため今回は最初から含めた。判断が分かれる新規の設計選択ではなく、既知のバグの再現を避けるための対応）:

- `tol <= 0.0`の検証（`MleError::InvalidTol`）
- `k == 0`（定数項も説明変数も無い病的な入力）の検証（`CommonError::NoRegressors`）。Logitでは当初漏れており、`n<=k`チェックをすり抜けて後段の分散共分散行列計算で`faer`が内部panicする実バグ（Issue #130）として顕在化し、Issue #118で修正された経緯がある。
- `validate_binary_y`によるyの値域検証（`MleError::InvalidBinaryY`）。Logitでは「Issue #54時点でB2に持ち越しと明記されながら未実装のまま残る」形でIssue #135まで放置されていたが、`nonlinear/common.rs`に既に共有実装があるため今回はゼロコストで含めた。
- `ProbitProblem`の構築を`from_standardized`コンストラクタ経由に統一（`new`はテスト専用として`#[cfg(test)]`化）。

### 既知の解析解によるテスト

切片のみ（説明変数なし）のProbitは、MLEの一階条件`Σ(y_i-Φ(θ))=0`（全観測で`z_i=θ`共通）から`Φ(θ̂)=ȳ`、すなわち`θ̂=Φ⁻¹(ȳ)`という閉じた形の解析解を持つ（Logitの`θ̂=ln(ȳ/(1-ȳ))`に相当）。`fit_newton_converges_to_closed_form_solution_for_intercept_only_model`テストで検証。

### テスト

`engine/src/nonlinear/probit.rs`の`#[cfg(test)] mod tests`に実装。10種: 切片のみモデルでの解析解一致、`confidence_level`範囲外エラー、`max_iter`非正エラー、`tol`非正エラー、`y`の値域エラー、`k=0`（`NoRegressors`）エラー、`n<=k`（`InsufficientObservations`）エラー、完全な多重共線性による`SingularHessian`エラー、`raise_on_non_convergence=true`での`NonConvergence`エラー、`raise_on_non_convergence=false`での未収束結果取得。Logitの`LogitEstimator::fit`骨格実装（Issue #56、コミット`c5e54f7`）時点の7種＋直後の修正（コミット`7525793`）で追加された`tol`/`k=0`検証テストに、Probit固有の`InvalidBinaryY`テストを加えた構成（rust-reviewerの指摘を受けて`NonConvergence`系2件を追加済み）。

### 未検証のリスク: `U_CLAMP`とNewton法（line searchなし）の相互作用

rust-reviewerのレビューで指摘され、ユーザー確認の上でテストは見送り、記録のみ残す。

`U_CLAMP`は一般化残差`λᵢ`（`φ(u)/Φ(u)`のNaN化）のみを防ぐ局所的な保護であり、Hessianの`w=λᵢ(λᵢ+zᵢ)`に使う`zᵢ`自体（生の線形予測子、クランプ前の値）は無制限のままである。`run_solver`（`nonlinear/common.rs`）のNewton実装はline searchなしで生のステップ`Δθ=H⁻¹g`をそのまま適用するため、理論上は次の経路が存在する: Hessianがまだ厳密特異ではないが悪条件な中間反復で`θ`（延いては標準化空間の`z`）が大きくジャンプし、次の反復でHessian要素が極端な値になり、さらに次の反復で発散的に増幅する。

最終的にNaN化すれば`newton_step`のNaNチェックが`SingularHessian`として捕捉する見込みだが、これは`U_CLAMP`自体が意図した保護機構ではなく「NaNチェックによる偶発的な保護」である。この経路を実データで踏むかどうかは未検証。

また、Logitの`SEPARATION_PARAM_NORM_THRESHOLD=100`（`SeparationSuspected`判定の閾値、`nonlinear/common.rs`）はLogitのロジスティック関数の飽和特性から較正された値であり、Probitのリンク関数（正規分布CDF、テイルの減衰特性が異なる）で同程度に適切かは未検証・未較正。

Logitの(準)完全分離対応（`SeparationSuspected`、Issue #138）は`LogitEstimator::fit`骨格実装（Issue #56）よりずっと後に、実運用で問題が顕在化してから発見・対応された別Issueであり、本Issue（#72、Issue #56相当のスコープ）には元々含まれていない。Probitでも同様に、method分岐（BFGS/L-BFGS、Issue #73）実装時、または実際に収束の問題が観測された時点で改めて検討する。

## BFGS/L-BFGSソルバー対応（Issue #73で実装済み）

`ProbitEstimator::fit`に`method: Method`引数を追加し、`run_solver`（既存のnewton/bfgs/lbfgs 3分岐）へ素通しする。`LogitEstimator::fit`（Issue #57、コミット`ef69aba`）と全く同じ変更（シグネチャへの引数追加のみ、バリデーション・後処理ロジックは無変更）。

`method`の選択に関わらず、収束点でのHessian評価（SE計算用、後続Issueで使用）は常に解析的に行う（`run_solver`の実装方針）。BFGS/L-BFGSが最適化中に内部で保持する近似Hessianは使い回さない。

### テスト

Logitの対応するテスト2種と同じ構成:

- **`fit_bfgs_and_lbfgs_converge_to_same_solution_as_newton`**: 切片のみモデル（既知の解析解`θ̂=Φ⁻¹(ȳ)`を持つ）で`bfgs`/`lbfgs`を実行し、解析解に収束することを確認。
- **`fit_bfgs_and_lbfgs_agree_with_newton_when_design_matrix_has_nontrivial_scale`**: 切片のみモデルは`x`が定数列だけのため`standardize_columns`のスケーリングが実質no-opになり、標準化・逆標準化の往復ロジックを通らない（Logitでrust-reviewerが指摘した点）。非自明なスケール（`x1=[10,20,30,40]`）を持つデータセットで`newton`/`bfgs`/`lbfgs`が同じ解に収束することを、閉じた形の解析解ではなく`newton`の結果を参照値として検証する。

いずれも「よく分離された」通常のデータのみを使っており、`U_CLAMP`が実際に発火する境界データ（Logitの`near_separation`シナリオに相当するもの）でのBFGS/L-BFGSの挙動は次節の理由により未検証のまま。

### 未検証のリスク: `U_CLAMP`領域での`cost()`/`gradient()`の数学的非整合とline search（BFGS/L-BFGS固有）

rust-reviewerのレビューで指摘され、ユーザー確認の上でテストは見送り、記録のみ残す（Issue #72の「未検証のリスク」節と同じ扱い）。

`u`が`U_CLAMP`でクランプされる領域では、`cost() = -log Φ(clamp(u(θ)))`はその領域内で`θ`に対して定数（`clamp`の微分が0のため）になるはずだが、`gradient()`はクランプ後の`λᵢ`（有限だが非ゼロ）をそのまま`xᵢ`に掛けた値を返しており、`cost()`の真の微分（0）と一致しない。この非整合はNewton法（line searchなし、`cost()`を呼ばない設計）には影響しにくいが、BFGS/L-BFGSが使うline search（Strong Wolfe条件、勾配から予測する減少量と実際の`cost()`の減少量を突き合わせる）は影響を受けうる。line searchが受理可能なステップを見つけられずエラーになる、あるいは逆に不適切なステップを受理する可能性が理論上ある。

**あえて修正しない判断（ユーザー確認済み）**: `gradient()`を`cost()`の真の微分（クランプ領域で0）に揃える「修正」は、実は逆に危険である。完全分離に近い強く誤分類された点で`gradient`が0になると、勾配ノルム基準の収束判定が誤検知しうる（Logitの完全分離問題、Issue #138と同じクラスのバグをU_CLAMP経由で再現することになる）。現状の実装（クランプ済みλをそのまま使う、有限だが大きな値を返す）は、`u→-∞`での真の（クランプ無し）Millsの比の漸近的な発散（`λ~|u|`）と整合する挙動に近く、意図的な設計判断として維持する。

BFGS/L-BFGS×分離データでの実際の挙動（line search失敗の有無等）は未検証。method分岐の後続issue、または実際に収束の問題が観測された時点で改めて検討する。

## 観測情報行列でのSE・z値・p値・信頼区間（Issue #74で実装済み）

`ProbitEstimator::fit`を拡張し、`run_solver`が返す収束点のHessianから観測情報行列（`Σ=-H⁻¹`、`cov_type="classical"`/`"nonrobust"`相当）による`std_errors`/`z_stats`/`p_values`/`conf_lower`/`conf_upper`を算出する。`LogitEstimator::fit`（Issue #58、コミット`75cc31d`）と同じ設計。検定分布は標準正規分布。

`destandardize_cov_params`・`observed_information_cov_params`はLogit実装時（Issue #58）に`nonlinear/common.rs`へ新設された系統共通インフラであり、Probitでは新規実装不要でそのまま再利用した。非ピボットCholeskyの特異性検出漏れ対策（Issue #129、`ensure_well_conditioned_symmetric_matrix`）も同じ共通関数経由のため、Probitは当初から恩恵を受けている。

`Normal::new(0.0, 1.0).map_err(...)`（Logitが使う、理論上到達不能な`Result`分岐を明示する書き方）ではなく`Normal::standard()`（`Result`を返さない）を使った。`probit.rs`内の他の箇所（`ProbitProblem`のcost/gradient/hessian/scores、`clamped_pdf_cdf`）で既に`Normal::standard()`に統一しているため、ファイル内の一貫性を優先した（Logitとの差分だが、機能的には同一）。

### 閉じた形の解析解（切片のみモデル、Fisher情報量の導出）

切片のみモデルは観測情報行列も閉じた形で書ける。MLEの一階条件`Σᵢλᵢ=0`（収束点で厳密に成立、`gradient(-ℓ)=-Σλᵢxᵢ`が0になる条件そのもの）を使うと、Hessian`H(θ)=Σᵢλᵢ(λᵢ+θ)`（`x_i=1`のため`xᵢxᵢ'=1`）は収束点`θ̂`で

```
H(θ̂) = Σᵢλᵢ² + θ̂·Σᵢλᵢ = Σᵢλᵢ²
```

（第2項が`Σλᵢ=0`で消える）に単純化できる。`y=1`の観測（`n1=n·ȳ`件）では`λ₁=φ(θ̂)/ȳ`、`y=0`の観測（`n0=n·(1-ȳ)`件）では`λ₀=-φ(θ̂)/(1-ȳ)`（`Φ(θ̂)=ȳ`を使用）なので、

```
H(θ̂) = n1·λ₁² + n0·λ₀² = n·φ(θ̂)²/ȳ + n·φ(θ̂)²/(1-ȳ) = n·φ(θ̂)²/(ȳ(1-ȳ))
```

これはprobitの切片のみモデルにおけるFisher情報量の標準的な結果と一致する。`Var(θ̂) = H(θ̂)⁻¹ = ȳ(1-ȳ)/(n·φ(θ̂)²)`。`fit_computes_std_errors_z_stats_p_values_and_ci_matching_closed_form_for_intercept_only_model`テストでこの式を実装から独立に検算した（自分で導出した式のため、rust-reviewerに数式検証を依頼済み）。

### テスト

Logitの対応するテスト2種と同じ構成:

- **`fit_computes_std_errors_z_stats_p_values_and_ci_matching_closed_form_for_intercept_only_model`**: 上記の閉じた形`Var(θ̂)=ȳ(1-ȳ)/(n·φ(θ̂)²)`から`cov_params`/`std_errors`/`z_stats`/`p_values`/`conf_lower`/`conf_upper`を検算（許容誤差`1e-6`、Newtonの収束判定`tol=1e-6`に起因する数値誤差を考慮）。
- **`fit_cov_params_is_symmetric_and_stats_are_internally_consistent`**: 多変量（説明変数2つ）データで`cov_params`の対称性・対角成分の正値性、および`std_errors`/`z_stats`/信頼区間幅の内部整合性（定義式通りの関係）を検証する回帰テスト。`destandardize_cov_params`の非対角成分の逆変換（`stds[i]*stds[j]`の掛け違い等）を対角成分だけのテストでは検出できないため。

## OPG（BHHH）・サンドイッチ型（HC0/HC1）・クラスターロバストSE（Issue #75/#76で実装済み）

`ProbitEstimator::fit`に`cov_type: CovType`引数を追加し、`Classical`/`Opg`/`Hc0`/`Hc1`/`Cluster`を選べるようにした。`LogitEstimator::fit`（Issue #59、コミット`0c485dc`、およびクラスター対応Issue #60、コミット`a91888e`）と同じ設計。`opg_cov_params`/`sandwich_cov_params`/`SandwichVariant`/`cluster_cov_params`はいずれもLogit実装時に`nonlinear/common.rs`へ新設された共通インフラで、新規実装は不要だった。

### Issue #76（クラスター）を同時に実装した理由

`CovType`は`Logit`/`Probit`/`Tobit`共通の1つの列挙型として`nonlinear/common.rs`に定義されており、既に`Cluster`バリアントを含んでいる（Logitのクラスター実装、Issue #60時点で追加済み）。`ProbitEstimator::fit`の`match cov_type`をコンパイルするには全バリアントを網羅する必要があるため、Issue #75（OPG/HC0/HC1のみ）に着手した時点で`Cluster`アームを何らかの形で書かないとビルドが通らない状態になった。

対応方針をユーザーに確認し、「今回（#75）でClusterも一緒に実装し、#76もクローズする」を選択した。理由: `Cluster`に必要な共有インフラ（`validate_cluster_groups`＝`engine/src/validation.rs`、`cluster_cov_params`＝`nonlinear/common.rs`）はLogit実装時（Issue #60）に既に完成・テスト済みで、Probit側は新規設計判断を伴わない配線作業のみだったため。仮の`NotYetImplemented`のようなエラー型を新設して`Cluster`を一時的に弾く案は、後で使われなくなる型を増やすだけで不自然と判断し採用しなかった。

`Logit`は反復最適化のため、グループキー未指定・クラスター数不足の検証を`fit()`冒頭（最適化実行前）で行う設計だった（OLSは閉形式解のため事後検証でもコストが変わらない）。Probitも同じ理由でこの設計をそのまま踏襲した。

### テスト

Logitの対応するテスト（Issue #59で2種、Issue #60で4種の計6種）と同じ構成:

- **`fit_cov_type_opg_hc0_hc1_match_independently_recomputed_values`**: `fit()`と同じ手順（標準化→収束点でのscores/Hessian評価→共通行列演算→`destandardize_cov_params`）をテスト側で独立に再現し、`opg`/`hc0`/`hc1`それぞれの`cov_params`と突き合わせる。多変量（k=3）データセットが必須な理由: 切片のみモデルでは情報行列の等式（`Σᵢsᵢsᵢ' = -H`）が有限標本で厳密に成り立ってしまい、`classical`/`opg`/`hc0`が偶然同じ値になるため、`match cov_type`の配線ミスを検出できない。
- **`fit_cov_type_cluster_matches_independently_recomputed_values`**（2:2の均等グループ）・**`fit_cov_type_cluster_matches_independently_recomputed_values_with_unbalanced_groups`**（3:2の不均衡グループ）: 同じ独立再現の技法で`cluster`を検証。不均衡ケースは`testing-policy.md`の指摘（均等サイズのみのテストは実務で起こりやすい偏ったグループサイズを見逃しうる）に倣う。
- **`fit_returns_missing_cluster_column_error_when_groups_not_provided`**・**`fit_returns_insufficient_clusters_error_when_only_one_group`**: `cov_type=Cluster`のグループキー未指定・クラスター数不足エラーを検証。
- **`fit_non_classical_cov_types_work_with_bfgs_and_lbfgs`**: `method`（`bfgs`/`lbfgs`）と`cov_type`（`Opg`/`Hc0`/`Hc1`/`Cluster`）の組み合わせを検証（Logitでrust-reviewerが指摘した「method横断・cov_type横断が別々にしか検証されておらず組み合わせが未検証」という穴を踏まえた構成）。

### 既知のテストギャップ（未対応、後続issueで拾う）

Logit側には、Issue #59/#60より後の「engine単体テストのカバレッジ確認」issue（コミット`25c4529`、`cargo-llvm-cov`で`fit()`内の`CovType::Hc0`/`Hc1`/`Opg`分岐の`?`（エラー伝播）を実際に通るテストが無いというギャップを検出）で追加された、完全な多重共線性データセットに対する`cov_type`ごとのエラー伝播テストが3種ある（`fit_returns_singular_hessian_error_for_perfectly_collinear_design_matrix_with_bfgs_and_lbfgs`／`..._with_hc0_and_hc1`／`fit_returns_singular_opg_matrix_error_for_perfectly_collinear_design_matrix`）。Probit側には対応するテストがまだ無い（rust-reviewer指摘）。

「Logitで既に判明済みのギャップパターン」であることが分かっているため、Probitのengine単体テストカバレッジ確認issue（`logit-probit-issue-breakdown.md`のC11相当、#80）着手時に確実に拾うこと。
