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

## 適合度統計量（Issue #77で実装済み）

`ProbitEstimator`に`log_likelihood`/`log_likelihood_null`/`lr_statistic`/`lr_p_value`/`pseudo_r_squared`（McFadden）/`aic`/`bic`/`n_obs`/`df_model`/`df_resid`を追加した（`nonlinear-api-design.md`5章の仕様通り）。`LogitEstimator`の対応する実装（Issue #61、コミット`6ae9d69`）とほぼ同じ設計をそのまま踏襲した。

- **`log_likelihood`**: `ProbitProblem::cost`（`-ℓ(θ)`、argminの`CostFunction`）から自由関数`log_likelihood(x, y, params)`（`Σᵢ log Φ(qᵢzᵢ)`、既存の`clamped_pdf_cdf`を経由して数値安定化する）を切り出し、`cost`はこれを符号反転して呼ぶ形にリファクタリングした。Logitと同様、`fit()`側は元スケールの`input.x()`/`input.y()`と`destandardize_params`済みの`params`を渡す。この関数はリンク関数固有（`Φ`ベース）のため`probit.rs`内に留まる。
- **`log_likelihood_null`（切片のみモデルのllf）・`lr_statistic`/`lr_p_value`/`pseudo_r_squared`/`aic`/`bic`/`df_model`/`df_resid`は`nonlinear/common.rs`の共通実装を使う**: 当初はLogitの実装（Issue #61、コミット`6ae9d69`）とほぼ同一のコードをそのまま移植する形で書いたが、rust-reviewerのレビューで「Logit/Probit間でこの計算がほぼ完全に重複している」との指摘を受け、ユーザー確認の上で`nonlinear/common.rs`へ集約した（`log_likelihood_null(y: &Mat<f64>) -> f64`、`GoodnessOfFit`構造体、`goodness_of_fit(llf, llnull, n, k) -> Result<GoodnessOfFit, MleError>`）。この計算がリンク関数に依存しないのは、切片のみモデルのMLEが「`link(θ̂) = ȳ`を満たす」という性質を持つため（Logitの`logistic(θ̂)=ȳ`、Probitの`Φ(θ̂)=ȳ`。どちらも既存テスト`fit_newton_converges_to_closed_form_solution_for_intercept_only_model`で検証済み）で、ベルヌーイ尤度が`p=link(θ)`のみに依存する形に落ちることに由来する。共通化の詳細な経緯・Logit側への影響は`logit-implementation-notes.md`「追記（Probit Issue #77時点）」節参照。Tobit着手時にも同じ計算が必要になる見込みのため、先に共通化しておいた。
- **`include_intercept=false`のときの非入れ子性・`df_model=k-1`固定・`df_model==0`時の`lr_p_value=NaN`**: Logitと同じ設計判断（詳細は`logit-implementation-notes.md`「適合度統計量」節参照、ユーザー確認済みの経緯を含め重複記載しない）。
- **Logit側Issue #130の`k=0`パニックはProbitでは再現しない**: Logit実装時に発覚した「`k=0`（`include_intercept=false`かつ説明変数も無い）で`cov_params`計算経路がfaerの`attempt to subtract with overflow`でパニックする」問題（`logit-implementation-notes.md`参照）は、Probitでは`fit()`冒頭で`k==0`のとき`CommonError::NoRegressors`を返すチェックが当初（Issue #72）から入っているため、そもそも到達しない。したがって本Issueでも追加の対応は不要だった。

### テスト

Logitの対応するテスト（Issue #61で3種）と同じ構成:

- **`fit_computes_goodness_of_fit_statistics_for_intercept_only_model`**: 切片のみモデル（`df_model=0`の境界ケース）。`log_likelihood`と`log_likelihood_null`が定義上一致すること、`lr_statistic≈0`・`pseudo_r_squared≈0`・`lr_p_value`がNaNになることを検証。
- **`fit_computes_goodness_of_fit_statistics_matching_independently_recomputed_values`**: 多変量（k=3）モデル。実装（`clamped_pdf_cdf`ベース）とは異なる式（標準正規CDF`Φ`から直接`Σ[y ln Φ(z) + (1-y) ln(1-Φ(z))]`を計算するベルヌーイ対数尤度の定義式そのもの）で`log_likelihood`を独立に再計算し、`log_likelihood_null`・`lr_statistic`・`pseudo_r_squared`・`df_model`・`df_resid`・`lr_p_value`（`statrs::ChiSquared`で独立に検算）・`aic`/`bic`を突き合わせた（`fit_cov_params_is_symmetric_and_stats_are_internally_consistent`と同じデータセットを再利用）。
- **`fit_lr_statistic_can_be_negative_when_include_intercept_is_false`**: `include_intercept=false`での非入れ子挙動。`lr_statistic`が負になりうること（NaN/Infにはならないこと）、`df_model`/`df_resid`/`aic`/`bic`は`include_intercept`の値に関わらず同じ式で計算されることを回帰テストとして固定した。

## 限界効果（Issue #78で実装済み）

`ProbitEstimator::marginal_effects(at, confidence_level)`を追加した（`fit()`とは独立した別メソッド、`nonlinear-api-design.md`6章）。着手前にユーザーへ方針を確認し、Issue #77（適合度統計量）でgoodness-of-fit計算を事後的に`nonlinear/common.rs`へ集約した経緯を踏まえ、**今回は最初から共通化を組み込んで設計した**（「dydx_and_jacobian・column_means・column_medianはリンク関数に依存せず、Logit/Probitで完全に同一の式になる」という判断、ユーザー確認済み）。

- **数式**: `dy/dx_j = φ(x_i'θ)θ_j`（`φ`は標準正規PDF）。Logitの`p(1-p)θ_j`とは異なるが、`at="overall"`（AME）・`"mean"`・`"median"`いずれも`g_j(θ)=w(θ)*θⱼ`という同じ形に帰着する性質はLogitと共通（`w`はAMEなら全観測平均の`φ(z)`、mean/medianなら代表点で評価した`φ(z̄)`）。`w`の勾配`s_m=∂w/∂θ_m`は`φ'(z)=-zφ(z)`（標準正規PDFの導関数）を使って`s_m=-zφ(z)xₘ`（AMEは全観測平均）と導出した。
- **`nonlinear/common.rs`への共通化**: `column_means`/`column_medians`/`MarginalEffects`構造体/`dydx_and_jacobian`（`w`・`sから`dydx`とヤコビアンを計算する部分、リンク関数に依らない）/`marginal_effects_from_w_s`（confidence_level検証・デルタ法標準誤差・定数項除外を含む、`dydx_and_jacobian`の呼び出しを含む一連の処理）をLogit/Probit共通として`common.rs`に配置した。Probit固有なのは`overall_w_and_s`/`at_point_w_and_s`（`w`/`s`を`φ`ベースで計算する部分）のみで、`probit.rs`に残している。これに伴いLogit側（Issue #62）の`marginal_effects`実装もリファクタリングしたが、数式・エラー型・テストカバレッジに変更はない。
- **`marginal_effects_from_w_s`の引数を構造体に束ねた**: 当初`param_names`/`has_intercept`/`k`/`params`/`cov_params`/`w`/`s`/`confidence_level`の8引数だったが、`clippy::too_many_arguments`（既定閾値7）に抵触したため、フィット済みモデルの情報（`w`/`s`/`confidence_level`を除く5つ）を`FittedModelForMarginalEffects<'a>`構造体に束ねた。
- **`dydx_and_jacobian`を`pub`にした理由**: `marginal_effects_from_w_s`からの呼び出しに加え、Logit/Probitそれぞれのテスト（モデル固有の`overall_w_and_s`等が返す実際の`w(θ)`を使った数値微分によるヤコビアンの独立検証）が直接呼ぶ必要があるため。
- **`U_CLAMP`を`overall_w_and_s`/`at_point_w_and_s`には適用していない**: 既存の`cost`/`gradient`/`hessian`は`λ=φ(u)/Φ(u)`という`Φ`で割る計算のため`U_CLAMP`によるクランプが必要だった（モジュール冒頭「数値安定化について」参照）が、限界効果の`w=φ(z)`は`Φ`で割らない単独の`φ`評価のみで、`z`が極端でも滑らかに0へ収束するだけで`0.0/0.0`のNaN化リスクが無いため、クランプ不要と判断した（rust-reviewerでも確認済み）。

### テスト

Logitの対応するテスト（Issue #62で7種）と同じ構成:

- **`dydx_and_jacobian_matches_numerical_differentiation_for_overall_w_and_s`**・**`..._for_at_point_w_and_s`**: `dydx_and_jacobian`のヤコビアンを、Probit固有の`overall_w_and_s`/`at_point_w_and_s`が返す`w(θ)`の中心差分数値微分と突き合わせる。
- **`marginal_effects_returns_empty_result_for_intercept_only_model`**: 切片のみモデル（`k_constant=1`、出力対象の説明変数が無い）で空の結果を返す境界ケース。
- **`marginal_effects_overall_matches_independently_recomputed_dydx_and_delta_method_se`**: `at="overall"`の`dydx`を`Normal::standard().pdf`から直接計算する式で独立に再計算し、標準誤差も`dydx_j`の数値微分によるヤコビアンから独立に導出して突き合わせる。
- **`marginal_effects_at_mean_differs_from_overall_and_matches_independent_recomputation`**・**`..._at_median_..._and_overall_..._`**: `at="mean"`/`"median"`が`at="overall"`と異なる代表点で評価されること、`dydx`を`column_means`/`column_medians`から独立に再計算した値と突き合わせる。
- **`marginal_effects_returns_invalid_confidence_level_error_out_of_range`**: `confidence_level`範囲外エラーの検証。

## predict() / pred_table()（Issue #79で実装済み）

`ProbitEstimator::predict()`（引数なし、`Vec<f64>`を直接返す。エラーなし）と`ProbitEstimator::pred_table(threshold)`（`Mat<f64>`の2×2的中表を直接返す。エラーなし）を追加した（`fit()`とは独立した別メソッド、`nonlinear-api-design.md`6章）。LogitのIssue #63（コミット`ad46976`）を踏襲。

- **`predict()`は`p_i=Φ(x_i'θ)`をそのまま計算する**（Logitの`Λ(x_i'θ)`を`Φ`に置き換えたのみ）。バリデーションを要する引数が無いためエラーなし。
- **`pred_table()`は`nonlinear/common.rs`へ最初から共通化して実装した**: `marginal_effects`（Issue #78）の`dydx_and_jacobian`等とは異なり、`pred_table`の計算そのもの（2×2カウント、`predicted: &[f64]`と`y: &Mat<f64>`のみに依存）は**リンク関数を一切参照しない**（`φ`/`Φ`/`logistic`のいずれも登場しない）ため、共通化の判断に曖昧さが無く、ユーザーに確認を求めずそのまま`common.rs`の`pred_table`関数へ移設した（`predict()`自体はリンク関数依存のため各モデルファイルに残す）。Logit側（Issue #63）の`pred_table`実装もこれに伴いリファクタリングしたが、数式・テストカバレッジに変更はない。
- **`pred_table`の`actual`側二値化の仕様**（`threshold`に関わらず常に`0.5`固定分割）は、Logit実装時にrust-reviewerの指摘・statsmodelsとの数値照合で発覚した実装ミス（初版は`actual`も`threshold`で二値化していた）の修正を踏まえたもの。共通化した`pred_table`関数のdocコメントに、この経緯を含めて記載している（`common.rs`参照）。
- **`in-sample`限定**: `predict()`/`pred_table()`ともに、`fit()`に使った`self.input.x()`/`self.input.y()`に対してのみ計算する。新規データ（out-of-sample）対応はLogit実装時に別issue化されたスコープ外（`logit-implementation-notes.md`参照）と同じ扱い。

### テスト

Logitの対応するテスト（Issue #63で5種）と同数・同構成:

- **`predict_matches_closed_form_for_intercept_only_model`**: 切片のみモデルでは全観測`p_i=ȳ`（closed form）になることを検証。
- **`predict_matches_independently_recomputed_normal_cdf_of_linear_predictor`**: 多変量モデルで`Normal::standard().cdf`から直接計算した`p_i=Φ(x_i'θ)`と突き合わせる。
- **`pred_table_matches_hand_computed_counts_for_intercept_only_model`**: 切片のみモデル（`p_i=ȳ=4/7`固定）で、`threshold`により全観測が一方のクラスに分類される自明なケースを手計算で検証。
- **`pred_table_matches_independently_recomputed_classification`**: 多変量モデルで、`predict()`の出力から独立に再計算した分類結果と突き合わせる（`threshold≠0.5`を使い、`actual`側が`threshold`に依存しないことも間接的に確認）。
- **`pred_table_actual_class_counts_are_invariant_to_threshold`**: 初回実装では`common.rs`側の合成データによる一般テストのみでこの性質をカバーし、`ProbitEstimator`経由の配線テストは省略していたが、rust-reviewerの指摘（Logit側には`predict()`→`pred_table()`という実際の呼び出し経路を通した対称のテストが既存のまま残っており、Probit側だけこの層のカバレッジが薄い）を受けて追加した。

### rust-reviewer指摘への対応

- **`common.rs::pred_table`に`debug_assert_eq!(predicted.len(), y.nrows())`を追加**: `enumerate().take(n)`が契約違反（長さ不一致）時にサイレントに不正確な集計を返すのを防ぐ防御（実際には長さが異なることはない内部契約だが、テスト実行時に即座に検知できるようにするため）。
- 上記「実測クラスカウントの`threshold`不変性」の配線テストをProbit側にも追加。

## engine単体テストのカバレッジ（Issue #80で確認・実装済み）

`cargo-llvm-cov -p engine --lib`で実測。OLS/Logitと同じ方針（100%は目指さず、理論上到達不能な防御的エラーパスはドキュメント化して受け入れる、`.claude/rules/testing-policy.md`「engine（Rust）のカバレッジ方針」参照）。

**実測結果（199テスト時点、修正前）**: `nonlinear/probit.rs` Region 97.80%・Line 98.69%・Function 100.00%。`--show-missing-lines`で未カバー行を確認したところ、`mod tests`（901行目）より前（本体コード）にあったのは`fit()`の`CovType::Hc0`/`Hc1`分岐内の`sandwich_cov_params(...)?`（エラー伝播、616・628行目）のみで、それ以外はすべてテストコード内（`assert!(cond, "fmt {}", expr)`のフォーマット引数がテスト成功時は評価されないことによる既知のノイズ、Logit側の`nonlinear-implementation-notes.md`・`logit-implementation-notes.md`「Issue #64」節で確認済みと同じパターン）。

### 実データで起こりうる真のギャップ（テスト追加で対応済み）

`probit-implementation-notes.md`「既知のテストギャップ」節（Issue #75/#76時点で記録済み）で予告していた通り、Logit側のIssue #64（コミット`25c4529`）で発見済みの`cov_type`ごとの特異行列エラー伝播ギャップを、ほぼそのままProbit版として移植した:

- **`fit_returns_singular_hessian_error_for_perfectly_collinear_design_matrix_with_bfgs_and_lbfgs`**: `Method::Newton`は`newton_step`内の特異性検出（ピボット付きQR）が最適化中に検出してしまうため、`bfgs`/`lbfgs`（`newton_step`を経由しない準ニュートン法）でのみ通る`observed_information_cov_params`経由の検出経路を検証。
- **`fit_returns_singular_hessian_error_for_perfectly_collinear_design_matrix_with_hc0_and_hc1`**: `sandwich_cov_params`内の`neg_hessian_inverse`経由。上記と同じ理由で`Method::Bfgs`を使う。
- **`fit_returns_singular_opg_matrix_error_for_perfectly_collinear_design_matrix`**: `opg_cov_params`が返す別のエラー型`SingularOpgMatrix`（`x2=2*x1`によりスコア行列`sᵢ=λᵢxᵢ`も構造的に多重共線性を持ち、OPG行列`Σsᵢsᵢ'`も特異になる）。
- **`fit_returns_singular_hessian_error_for_perfectly_collinear_design_matrix_with_cluster`**: `cluster_cov_params`も内部で`neg_hessian_inverse`を呼ぶため`SingularHessian`。

いずれもデータセット（`x2=2*x1`の完全な多重共線性）・アサーション構造はLogit版と同一。Probitの`W=diag(λᵢ(λᵢ+zᵢ))`はLogitの`W=diag(pᵢ(1-pᵢ))`と式は異なるが、`X'WX`の特異性は設計行列`X`自体の構造的な多重共線性（`W`の対角成分の値によらず`X`の列が線形従属である限り常に特異）に由来するため、同じデータセットで同様に検出できる。

修正後の`nonlinear/probit.rs`カバレッジ: Region 97.93%・Line 98.88%・Function 100.00%（Logit側のRegion 97.82%・Line 98.85%と同水準）。

### `common.rs`・`logit.rs`側の再確認

Issue #77/#78/#79で`common.rs`へ新規追加したコード（`goodness_of_fit`/`marginal_effects_from_w_s`/`pred_table`）に起因する新しいカバレッジギャップが無いことを確認した。`common.rs`・`logit.rs`の未カバー行は、いずれも既存のLogit Issue #64で「理論上到達不能な防御的エラーパス」としてdocコメント付きで受け入れ済みのもの（`convert_optimizer_error`の`Err(other)`分岐、`FaerNewton::name`等）のみだった。

### 完了条件

カバレッジ実測結果（修正前後の数値）・未カバー箇所の扱い（実データで起こりうるギャップ4件はテスト追加、それ以外は理論上到達不能または既存の許容済みパターンとして受け入れ）を本節にまとめた。

## engine_pybind: データ抽出・ProbitOptions/ProbitResult pyclass定義（Issue #81で実装済み）

`engine_pybind/src/nonlinear/probit.rs`を新設した（`mod.rs`に`pub mod probit;`を追加）。Logitの対応するIssue（#65、コミット`ee813b0`）とほぼ完全に同型のパターンで実装した。ただし現在の`logit.rs`はさらに後続issue（#66・#67・#133・#134）で進化しているため、着手前にその変遷を確認した上で、Issue #81本文が明示する`ProbitResult`のフィールド一覧（`estimator`フィールドを含まない、スカラー・ベクトルのみ）がIssue #65時点のスコープと一致することを確認して踏襲した。

- **スコープの区切り方（Logitの前例をそのまま踏襲）**: `ProbitEstimator::fit()`の呼び出し・`ProbitResult`の実際の構築・`#[pymodule]`への登録（`lib.rs`の変更）はIssue #82（Logitの#66に相当）に送り、本Issueでは行わない。データ抽出・バリデーション・`engine::nonlinear::probit::ProbitInput`構築までを行う`pub(crate) fn build_probit_input(df: &DataFrame, y: String, x: Vec<String>, options: &ProbitOptions) -> PyResult<(ProbitInput, EngineCovType, EngineMethod)>`を切り出した。
- **`ProbitOptions`はLogitOptionsとフィールド単位で完全に同一**（`nonlinear-api-design.md`7章、Logit/Probitが同じオプション面を共有する設計通り）。`start_params`は同じ理由（engine側`ProbitEstimator::fit`が未対応）で除外。
- **バリデーション・エラー変換は既存の共有インフラをそのまま再利用**: `engine_pybind/src/validation.rs`の`validate_x_non_empty`/`validate_no_duplicate_roles`/`validate_no_duplicate_x`/`validate_no_const_collision`、`nonlinear/common.rs`の`mle_error_to_pyerr`。Logit実装時（Issue #65→#134）に発見・解消済みの問題（`cargo test -p engine_pybind`が実行できない構造的制約、y/x列間の行数不一致チェックが理論上到達不能と判明し削除、バリデーションの重複実装をvalidation.rsへ集約）はいずれもクレート全体・共有インフラに対する修正のため、Probit側で再発することはない（そのまま恩恵を受ける）。`parse_cov_type`/`parse_method`も現在の`logit.rs`の対応する実装（`n: usize`引数を持たない`Option::map().transpose()?`パターン）をそのまま踏襲した。
- **`#[allow(dead_code)]`を`parse_cov_type`/`parse_method`/`build_probit_input`に付与**: Issue #81時点では`#[cfg(test)] mod tests`以外に呼び出し元が無いため。Logit実装時（Issue #65）と全く同じ理由・同じ対応方針（`engine_pybind`はPython拡張モジュール専用の薄いバインディング層でありクレート外にRust APIを公開する設計ではないため`pub`化での回避は見送り）。Issue #82で`fit`が実際に呼ぶようになった時点でこれらの属性は不要になる（削除すること）。

### テスト

`engine_pybind/src/nonlinear/probit.rs`の`#[cfg(test)] mod tests`に、`build_probit_input`の検証をLogitの対応するテスト（12件）と同数・同構成で実装した（`Series`/`DataFrame`を直接組み立てる、`polars::df!`マクロ利用。Pythonインタプリタ（GIL）を起動せずに検証できる設計）。

- 成功パス: 切片あり・切片なしそれぞれでの`ProbitInput`構築、`cov_type`/`method`の正しいパース
- `ValidationError`パス: 空の`x`、y/xの重複、x内の重複、`"const"`列衝突、存在しない列、欠損値、未知の`cov_type`文字列、未知の`method`文字列
- クラスター系: `cluster_col`指定時のグループキー抽出、未指定時に`groups=None`のまま返す（`engine`側の`MissingClusterColumn`検証に委ねる設計）

`cargo build -p engine_pybind`/`cargo clippy -p engine_pybind --all-targets -- -D warnings`/`cargo fmt --check`/`cargo test -p engine_pybind`すべて成功（43件pass、変更前31件から+12、デグレなし）。

## engine_pybind: engine呼び出し・`fit_probit`登録（Issue #82で実装済み）

`build_probit_input`の後段として`engine_pybind/src/nonlinear/probit.rs`に`pub(crate) fn fit(data: PyDataFrame, y, x, options: &ProbitOptions) -> PyResult<ProbitResult>`を追加した。`build_probit_input`→`engine::nonlinear::probit::ProbitEstimator::fit`（`MleError`は`mle_error_to_pyerr`で変換）→`ProbitResult`構築、という流れ（`logit.rs`の`fit`と完全に同じ構造）。`engine_pybind/src/lib.rs`に`#[pyfunction] fit_probit`を追加してこれに委譲し、`#[pymodule]`に`fit_probit`/`ProbitOptions`/`ProbitResult`を登録した。Issue #81時点で残していた`#[allow(dead_code)]`（`build_probit_input`/`parse_cov_type`/`parse_method`）は、これらが実際に呼ばれるようになったため全て削除した。

- **`mle_error_to_pyerr`は変更不要だった**: Issue本文は「共通化できる可能性がある」と示唆していたが、`nonlinear/common.rs`の`mle_error_to_pyerr`は既にLogit実装時（Issue #66）の時点でLogit/Probit/Tobit共有の`engine::nonlinear::common::MleError`全バリアント（`InvalidBinaryY`を含む）を対象にした実装になっており、Probit固有の変更は不要だった。
- **数値検証**: `engine`のユニットテスト`fit_cov_type_opg_hc0_hc1_match_independently_recomputed_values`・`fit_cov_type_cluster_matches_independently_recomputed_values`と同じ入力（`y=[0,1,0,1]`、`x1=[10,20,30,40]`、`x2=[-5,2,8,-1]`、cluster`=[a,a,b,b]`、`tol=1e-8`）で、一時的なRustサンプル（`engine/examples/probit_oracle.rs`、コミット対象外、検証後削除）から`params`/`std_errors`のオラクル値を出力させ、`uv run maturin develop --release`でビルドした`fit_probit`をPythonから同じデータで呼び出した結果と突き合わせ、classical/opg/hc0/hc1/clusterの全cov_typeで許容誤差1e-9で一致することを確認した（完了条件通り）。
- **テストは追加していない（Logitの前例通り）**: `fit`/`fit_probit`は`PyDataFrame`を引数に取るため、`build_probit_input`と異なりGILなしの`#[cfg(test)]`では直接呼べない（`engine_pybind/src/nonlinear/CLAUDE.md`「テストの制約: `PyDataFrame`引数の関数はcargo testから直接呼べない」参照）。
- **`uv run pytest tests/api_tests`（398件）で既存OLS/WLS/Logit機能への回帰が無いことも確認済み**。

## engine_pybind配線の追加・python_packageラッパー実装（Issue #83で実装済み）

Issue #83は当初「python_packageラッパー実装のみ」（依存: #82）というスコープだったが、着手前の調査で`ProbitResult`（Issue #81/#82時点）に`predict()`/`pred_table()`/`marginal_effects()`のpymethodsが1つも無いことが判明した（`engine::nonlinear::probit::ProbitEstimator`自体はIssue #77-79で実装済みだったが、engine_pybind層の配線が無かった）。Logit Issue #67で全く同じギャップが発覚し、OLSのpredict実装（コミット`c6caed7`、engine拡張・engine_pybind配線・python_packageラッパーを1つのIssueにまとめた前例）に倣ってengine_pybind配線もまとめて実装した経緯（コミット`cd222c5`）を確認し、ユーザー確認の上で同じ方針を踏襲。engine_pybind配線もIssue #83にまとめて実装した（`engine_pybind/src/nonlinear/CLAUDE.md`「Probitの実装状況」節にも同じ経緯を記録）。

### engine_pybind: `ProbitResult`への`estimator`フィールド・pymethods追加

`LogitResult`（Issue #67）と完全に同じ設計。`ProbitResult`に非公開`estimator: ProbitEstimator`フィールドを追加し（`#[derive(Clone)]`は`ProbitEstimator`がCloneを実装していないため外す、`LogitResult`と同じ理由）、`predict()`/`pred_table(threshold=0.5)`/`marginal_effects(at="overall", confidence_level=0.95)`をpymethodsとして実装した（いずれも`self.estimator`への単純委譲）。`fit()`が返す`ProbitResult`構築時に`estimator`（ムーブ）を最後のフィールドとして渡す。

**`MarginalEffectsResult`と`parse_marginal_effects_at`を`logit.rs`から`nonlinear/common.rs`へ移動**: `engine::nonlinear::common::MarginalEffects`（marginal_effectsの返り値の元となるengine側の型）は元々Logit/Probit共有だったため、対応するengine_pybind側のpyclass・パース関数もLogit専用の位置に置いたままにせず、`rust-style.md`「系統内で共有するロジックは`<系統>/common.rs`に置く」に従って共有化した（ユーザー確認済み、Probit専用に重複定義する代替案・移動せず`super::logit::MarginalEffectsResult`をそのまま参照する代替案も検討した上で選択）。`logit.rs`は`super::common::{MarginalEffectsResult, parse_marginal_effects_at}`をimportするだけに変更、テスト（`parse_marginal_effects_at_*`の2件）も`common.rs`側の`#[cfg(test)] mod tests`に移動した。`engine_pybind/src/lib.rs`の`#[pymodule]`登録も`nonlinear::logit::MarginalEffectsResult`から`nonlinear::common::MarginalEffectsResult`に変更（登録自体は1回のまま、Logit/Probit両方がこの1つのpyclassを共有する）。

`cargo build`/`clippy -D warnings`/`fmt --check`/`test -p engine_pybind`すべて成功（43件pass、件数はテスト移動のみで変わらず）。

### python_package: `Probit`/`ProbitResults`

`python_package/econometricsmodels/nonlinear/probit.py`を新設した。`logit.py`と完全に同型（`Probit`/`ProbitResults`のフィールド・メソッド構成、`coef_table()`の`z_stat`キー、`predict()`の`{"probability": p}`、`pred_table()`の行指向`list[dict]`、`marginal_effects()`のキー命名`param`/`dydx`/`std_err`/`z`/`p_value`/`conf_low`/`conf_high`まですべて一致）。`ProbitOptions`は`_lib`からの再輸出、`summary()`は未実装（いずれもLogitと同じ確定済み方針）。トップレベル`__init__.py`に`Probit`/`ProbitOptions`/`ProbitResults`を追加した。

### テスト

`tests/api_tests/test_probit.py`を新設した（`test_logit.py`と同型、36件）。統計量・API構造・`ValidationError`/`ComputationError`の各パスを検証するスモークテストのみで、statsmodels/R glmとの厳密な数値照合はIssue #84で別途実施する。`SeparationSuspected`検出（Issue #138由来）は`engine::nonlinear::common::run_solver`（Logit/Probit共有の最適化ループ）に実装されているため、Probit側でも同じ准完全分離DGPで`ComputationError`になることを確認した。

`uv run ruff check .`/`uv run ruff format --check .`（リポジトリ全体）、`uv run maturin develop --release`でのビルド後`uv run pytest tests/api_tests`（434件、398件+36件、既存機能への回帰なし）すべて成功（完了条件通り）。

## statsmodels/R glmとの数値照合ベンチマーク作成（Issue #84で実装済み）

`/test-new`スキル・`.claude/skills/reference-benchmark/`の手順に沿い、Logitの`benchmark/nonlinear/`資産（Issue #68）をProbit向けに拡張した。合成データ生成・`run_statsmodels_benchmark.py`・`run_glm_crosscheck_benchmark.R`はLogit/Probitで共有できるロジックが大半だったため、重複実装ではなく一般化する方針をユーザー確認の上で採用した（`--weight-col`でOLS/WLSを共有する`linear`系統の`run_statsmodels_benchmark.py`と同じ設計）。

### 合成データセット: `generate_logit_datasets.py`を`generate_binary_choice_datasets.py`へ一般化（ユーザー確認済み）

シナリオ構成（baseline/small_n/moderate_multicollinearity/high_condition_number/near_separation/perfect_multicollinearity/scale_variance）とX生成ロジックはリンク関数に一切依存せず完全に共有できることが分かったため、`generate_logit_dataset(scenario, ...)`に`link: "logit"|"probit"`引数を追加した`generate_binary_choice_dataset(scenario, link, ...)`へ一般化し、ファイルも`generate_binary_choice_datasets.py`にリネームした（`generate_logit_dataset`/`generate_probit_dataset`という名前付きエイリアスは既存呼び出し元（`freeze_datasets.py`）との互換のため維持）。リファクタリング後、既存の`logit_*.csv`（7シナリオ）が旧実装とバイト単位で完全に一致することを確認済み（既存のフィクスチャJSON・凍結CSVへの影響が無いことの検証）。

**`near_separation`の較正値はリンク関数ごとに異なる**（重要な実装判断）: 標準正規分布のΦはロジスティック分布のΛより裾が薄く、同じベータ値でもΦの方が0/1に速く飽和するため、Logitと同じ`beta1=20`をProbitにそのまま使うと収束後の標準誤差が過大（またはengine・statsmodelsで不安定）になる懸念があった。実測較正の結果、Probitは`beta1=10`で「収束するが標準誤差が大きく膨らむ」という同種の挙動になることを確認した（`beta1=6/8/10/12`で試行し、engine・statsmodelsのclassical SEが完全一致することも確認済み）。

### `cov_type="hc1"`/`"opg"`の既知の欠落はLogitと同様

statsmodelsのdiscrete model（`Probit.fit`）でも、Logitと全く同じ欠落が実機確認された: `hc1`はn/(n-k)小標本補正が未実装でHC0と同一値を返す（`hasattr(ProbitResults, "cov_HC1")`が`False`）、`opg`は"cov_type not recognized"エラーになる。対処もLogitと同じ（`hc1`はRを主リファレンスに、`opg`は`model.score_obs(params)`から手計算）。

### R側のクロスチェックで発覚した重要な問題: `glm()`の既定共分散は非正準リンクで「期待情報行列」を返す（ユーザー確認不要、実装で対応済み・重要）

Probitのクロスチェック値生成時、`classical`で最大約2-3%・`hc0`/`hc1`で最大約8%という無視できない乖離が発覚した（`opg`は誤差~1e-5と正常）。原因を`numDeriv::hessian()`による数値微分Hessianとの比較で特定した:

- R標準の`glm()`は`vcov(model)`（および`sandwich::vcovHC`/`vcovCL`が内部で使う`bread.glm()`）を、IRLS（Fisher scoring）の作業重みに基づく**期待情報行列**で計算する。
- **Logit（binomial族の正準リンク）では期待情報行列と観測情報行列（真の対数尤度のHessian）が理論上一致する**ため、この違いは表面化しなかった（Issue #68時点で発覚しなかった理由）。
- **Probit（非正準リンク）では両者が一致しない**。本実装（`nonlinear/probit.rs`）・statsmodelsはどちらも観測情報行列（`Σ=-H⁻¹`、`H`の重みは`λᵢ(λᵢ+zᵢ)`）を使うため、Rの`glm()`の既定`vcov()`をそのまま参照値にすると本実装が「間違っている」ように見えるが、実際にはRの既定値が異なる量を計算しているだけだった（`numDeriv`の数値微分Hessianが本実装の`classical`標準誤差と機械精度で一致することを確認し、本実装側が正しいことを検証済み）。

対処として`run_glm_crosscheck_benchmark.R`に`observed_hessian_weights()`/`observed_bread()`を新設し、本実装と同じ解析式（Logit: `pᵢ(1-pᵢ)`、Probit: `λᵢ(λᵢ+zᵢ)`）で観測情報行列を明示的に計算し、`sandwich::sandwich(bread.=observed_bread, meat.=...)`でclassical/hc0/hc1/clusterすべてに適用するよう変更した。Logitでは正準リンクの性質によりこの変更前後で数学的に完全に同じ値になることを実機確認済み（凍結済みの`logit_crosscheck.json`と新実装の出力が一致、既存フィクスチャの再生成は不要と判断）。`scale_variance`シナリオでは既存の`opg`ブランチと同じ理由（列スケール差による見かけ上の特異性）で`observed_bread()`も列正規化してから反転する対応が必要だった（同じ`Σ=D⁻¹(D⁻¹MD⁻¹)⁻¹D⁻¹`の恒等式を使用）。

この発見はLogitの「statsmodelsの`hc1`未実装」（Issue #68）と同種の、ベンチマーク作成時に初めて顕在化する参照実装側の落とし穴であり、独立した検証（`numDeriv`によるHessianの数値微分）で原因を特定してから対処した。

### near_separationの`tol`妥当性検証: Logitと同じ結論

`beta1=10`のnear_separationデータで、既定`tol=1e-6`とstatsmodelsとの相対誤差は最大約4.4e-8（`RTOL=1e-8`の基本方針をわずかに超過）、`tol=1e-8`まで締めると約6.6e-11まで改善することを実測確認した。Logit（Issue #68）と同じ結論（既定値`1e-6`は変更しない、near_separationの数値比較テストのみ`tol=1e-8`を明示指定）を踏襲した（`nonlinear-implementation-notes.md`「収束判定の`tol`」に追記済み）。

### テスト・フィクスチャ

- `tests/api_tests/fixtures/benchmarks/probit.json`（statsmodels主リファレンス）・`probit_crosscheck.json`（Rクロスチェック）を新規作成。`tests/api_tests/fixtures/benchmarks/data/probit_*.csv`（7シナリオ）・`probit_true_beta.json`も`freeze_datasets.py`で新規凍結。
- `tests/api_tests/test_probit_fixtures.py`（26件）・`test_probit_crosscheck.py`（32件）を新規作成。`test_logit_fixtures.py`/`test_logit_crosscheck.py`と完全に同型（件数もLogitと一致）。
- 許容誤差は`test_probit_fixtures.py`（statsmodels）に`RTOL=1e-8`、`test_probit_crosscheck.py`（R）に基本方針`RTOL=2e-4`を適用（Logitと同じ基本方針）。個別に緩めた項目（Logitと同じ理由・同じ性質の乖離）:
  - 限界効果の`std_err`: `RTOL=1e-3`（実測最大~7e-4、Logitの`5e-3`より小さい実測値だがマージンを持たせた）
  - p値: `ATOL=5e-5`（実測最大絶対誤差~2.9e-5）
  - Wooldridge mrozのクラスターロバストSE（`cluster_col="city"`、G=2）: `RTOL=2e-3`（実測最大相対誤差~1.1e-3、const）。合成データのクラスターケース（G=2/G=10いずれも~5e-5水準）より大きいが、実データ・クラスタ数境界（G=2）・相関の強い説明変数（exper/expersq等）が重なる境界的なケースであるため個別の許容誤差とした（`testing-policy.md`「許容誤差」の「統計量・cov_typeごとに実測乖離が大きく異なる場合は許容誤差を分けてよい」方針通り）。
- Wooldridge実データは`mroz`（Logitと同じ、労働参加モデル）を採用。probit_logitは経済学の教科書でも定番の比較対象であり、同じデータ・同じformulaを使うことが自然と判断した。

`uv run maturin develop --release`でのビルド後、`uv run ruff check .`/`uv run ruff format --check .`（リポジトリ全体）、`uv run pytest tests/api_tests`（492件、434件+26件+32件、既存機能への回帰なし）すべて成功。

### testing-completeness-reviewerレビューで対応した指摘

must-fixなし。should-fix 2件のうち1件を対応、1件は対応見送り（理由下記）。

- **対応済み**: `run_glm_crosscheck_benchmark.R`の`observed_bread()`が、logit（正準リンク）では`glm()`既定の`bread()`と数学的に一致するという不変条件を、目視確認のみに頼っていた（将来このスクリプトの計算式を変更した際に、Logit側のクロスチェック値が気づかれずに壊れるリスクが残っていた）。`link == "logit"`のときのみ`stopifnot(isTRUE(all.equal(bread_obs, bread(model), tolerance = 1e-6)))`を追加し、自動検証するようにした。フィクスチャの再生成結果がバイト単位で不変であることを確認済み（副作用なし）。
- **対応見送り**: `test_probit_fixtures.py`/`test_probit_crosscheck.py`が`method`（bfgs/lbfgs）をパラメータ化しておらず、常にデフォルト（Newton）でのみリファレンス実装と数値比較している点。ただしこれは`test_logit_fixtures.py`/`test_logit_crosscheck.py`（Issue #68）から完全に踏襲した既存の設計であり、Probitで新たに生じたギャップではない。レビュー自体も「Logit/Probit双方に一括で追加するかどうかをユーザーに確認してから着手するのが望ましい」と明記しており、Issue #84単体のスコープ外と判断した。
