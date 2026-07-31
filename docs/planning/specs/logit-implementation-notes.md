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

`engine/src/nonlinear/logit.rs`の`LogitEstimator::fit`。A2（Issue #52）の`run_solver`を呼び出し、`converged`/`n_iter`を含む結果を返す骨格。当初（Issue #56時点）は`Method::Newton`固定だったが、Issue #57で`method: Method`引数を追加し、呼び出し側が`newton`/`bfgs`/`lbfgs`を選べるようにした。

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

## BFGS/L-BFGSソルバー対応（Issue #57で実装済み）

`LogitEstimator::fit`に`method: Method`引数を追加し、`run_solver`へそのまま渡すだけ（`run_solver`自体は既にIssue #52で`newton`/`bfgs`/`lbfgs`の3分岐を実装済みで、収束点でのHessian評価も`method`の選択に関わらず常に解析的に行う設計になっていたため、Logit側での追加のロジックは不要だった）。

### テスト

- **`newton`との結果一致（切片のみモデル）**: 既知の解析解を持つ切片のみモデルで`bfgs`/`lbfgs`をそれぞれ実行し、`newton`（Issue #56のテスト）と同じ解析解へ収束することを検証（`fit_bfgs_and_lbfgs_converge_to_same_solution_as_newton`）。`newton`は許容誤差`1e-6`だが、`bfgs`/`lbfgs`は準ニュートン法で収束が緩やかなため`common.rs`の既存テスト（`run_solver_bfgs_converges_to_known_minimum`等）に倣い`1e-4`とした。
- **`newton`との結果一致（非自明なスケールを持つ説明変数）**: 上記の切片のみモデルは`x`が定数列（切片）だけのため`standardize_columns`のスケーリングが実質no-op（`stds`が全て`1.0`のまま）になり、標準化・逆標準化の往復ロジックを一度も通らない（rust-reviewer指摘、Issue #57で対応）。`x1=[10,20,30,40]`という非自明なスケールの説明変数を持つデータセット（閉じた形の解析解は無い）で`newton`/`bfgs`/`lbfgs`を実行し、3手法が同じ解へ収束することをクロスメソッド一致検証で確認した（`fit_bfgs_and_lbfgs_agree_with_newton_when_design_matrix_has_nontrivial_scale`）。標準化空間でのBFGSの初期逆Hessian（`identity_matrix(k)`）・`destandardize_params`が正しく機能していることの間接的な検証になる。

## 観測情報行列でのSE・z値・p値・信頼区間（Issue #58で実装済み）

`LogitEstimator::fit`を拡張し、`run_solver`が返す収束点のHessian（標準化空間、θ_std基準）から`cov_type="classical"`/`"nonrobust"`相当の分散共分散行列を計算する。`cov_type`の選択オプション自体はまだ無く（OPG/サンドイッチ/クラスターはB6・B7）、常に観測情報行列を使う。

- **標準化空間の`cov_params`をどう元のスケールに戻すか（重要な設計判断）**: `run_solver`が返すHessianは標準化された設計行列`x_std`基準（θ_std空間）で評価されている。`destandardize_params`はパラメータベクトルの逆変換（`θ_orig_j = θ_std_j/std_j`）だが、分散共分散行列には別の変換則が必要になる。`θ_orig = D⁻¹θ_std`（`D=diag(stds)`）とすると、連鎖律から`H_std = D⁻¹H_origD⁻¹`が成り立ち、これを`H_orig`について解くと`H_orig = D H_std D`。分散共分散行列はHessianの逆行列に比例する（`Σ=-H⁻¹`等）ため、`Σ_orig = D⁻¹Σ_stdD⁻¹`となる（`D`が対角行列であることから、成分ごとに`Σ_orig[i,j] = Σ_std[i,j]/(stds[i]*stds[j])`という単純な式に帰着する）。この関係はOPG・サンドイッチ・クラスターのいずれの`cov_type`でも同様に成り立つ（`Σ`の式がいずれも`H⁻¹`を両側から掛ける、または`H⁻¹`の逆行列を取る形のため）。この変換を`destandardize_cov_params`として`nonlinear/common.rs`に実装した（`destandardize_params`と対になる関数、Probit実装時にも再利用する想定）。
- **`cov_params`（k×k行列）をフィールドとして保持**: `nonlinear-api-design.md`のB9（限界効果）が「`fit()`時の`cov_params`を再利用する（再最適化不要）」と明記しているため、対角成分（`std_errors`）だけでなく行列全体を`LogitEstimator`のフィールドとして保持する（OLSの`OlsEstimator`は`cov_params`を`fit()`内のローカル変数としてのみ使い、フィールドとしては保持していないが、OLSには限界効果のような事後的な再利用箇所が無いための違い）。
- **`Normal::new(0.0, 1.0)`のエラー分岐は理論上到達不能**: 標準正規分布は標準偏差が正であることを要求するstatrsの検証を常に満たすため、`.claude/rules/rust-style.md`「テスト」のカバレッジ方針に従い、docコメントに理由を明記した上で`cargo-llvm-cov`のカバレッジ対象外として許容する。

### テスト

- **既知の解析解（切片のみモデル）**: 切片のみモデルでは全観測で`p_i=ȳ`となるため、観測情報行列も`H=-n*ȳ*(1-ȳ)`という閉じた形になり、`Var(θ̂)=1/(n*ȳ*(1-ȳ))`という解析解を持つ。この値との一致（許容誤差`1e-6`。Newtonの収束判定`tol=1e-6`由来の数値誤差があるため、他の閉じた形テストと同じ桁にした）と、z値・p値・信頼区間が標準正規分布（`statrs::Normal`で独立に検算）の定義式通りであることを検証した。
- **多変量モデルでの内部整合性**: 多変量（k=3）では標準誤差に閉じた形の解析解が無いため、`cov_params`の対称性・対角成分が正であること、および`std_errors`/`z_stats`/`conf_lower`/`conf_upper`が定義式通りの関係を満たすことを検証した。対角成分だけでなく非対角成分の対称性も確認することで、`destandardize_cov_params`の`stds[i]*stds[j]`の掛け違い（添字の転置ミス等）を検出できるようにしている。

### `bfgs`/`lbfgs`経由での特異性検出漏れ（Issue #129で修正済み）

rust-reviewerの指摘（`bfgs`/`lbfgs`×完全な多重共線性でのテスト欠落、Issue #58時点）に対応しようとしたところ、テストの欠落ではなく実際のバグを発見した。`Method::Bfgs`で完全な多重共線性のあるデータセットを`fit()`すると、`MleError::SingularHessian`にならず桁違いに巨大な値を含む`Ok`が返っていた。原因は`observed_information_cov_params`（`neg_hessian_inverse`）が使う非ピボットCholesky分解が特異性を確実に検出できないため（`engine/src/linear/CLAUDE.md`に記録済みのOLSと同じ既知の限界、Issue #107の再発）。`Method::Newton`は`newton_step`内の別の検出経路（ピボット付きQR）でたまたま検出できているだけで、`bfgs`/`lbfgs`はこの経路を経由しないため無防備だった。

Issue #58時点ではIssue #58本来のスコープを超えるためユーザー確認の上でIssue #129として切り出し、Issue #129で対応した。修正内容は`nonlinear-implementation-notes.md`「`cov_type`共通行列演算の特異性検出（Issue #129で修正済み）」参照。`fit_returns_singular_hessian_error_for_perfectly_collinear_design_matrix_with_bfgs_and_lbfgs`テスト（`bfgs`・`lbfgs`両方で正しく`SingularHessian`になることを確認）で検証済み。

## OPG（BHHH）・サンドイッチ型（HC0/HC1）でのSE（Issue #59で実装済み）

`LogitEstimator::fit`に`cov_type: CovType`引数を追加した。`CovType`（`Classical`/`Opg`/`Hc0`/`Hc1`。`Cluster`はB7で追加）は`Method`と同じ理由で`nonlinear/common.rs`に定義し、Probit/Tobitでも再利用する想定。

- **収束点でのスコア評価に`LogitProblem`のクローンが必要**: `Opg`/`Hc0`/`Hc1`は収束点での観測ごとのスコア（`LogitProblem::scores`）が必要だが、`run_solver`は`problem`の所有権を取り込み、内部で保持していたモデルを呼び出し元へ返さない設計になっている（`SolverOutput`に`model`フィールドが無い）。`run_solver`のシグネチャを変更して`model`を返す設計も検討したが、`LogitProblem`は元々`argmin::core::Executor`向けに`Clone`を要求しているため、`run_solver`に渡す前に`problem.clone()`しておく方が`run_solver`（Logit/Probit/Tobit共通のユーティリティ）のシグネチャを変えずに済み、影響範囲が小さい。
  - **クローンは`cov_type=Classical`のときは行わない**（rust-reviewer指摘）: 初回実装では`cov_type`に関わらず常に`problem.clone()`していたが、`Classical`はスコアを一切使わないため設計行列を含む無駄な複製になる。`cov_type`に応じて`Option<LogitProblem>`で条件付きにクローンする形に修正した。
- **`cov_params_std`の計算はいずれも標準化空間で行ってから`destandardize_cov_params`で逆変換**: Issue #58で確立した「標準化空間で`Σ_std`を計算し、最後に一度だけ`destandardize_cov_params`で元のスケールへ変換する」設計をそのまま踏襲する。`opg_cov_params`/`sandwich_cov_params`（`nonlinear/common.rs`、Issue #53で実装済み）はいずれも標準化空間の`scores_std`・`hessian_std`を受け取ってΣ_stdを返すため、`cov_type`ごとの分岐は「どの共通関数を呼ぶか」の違いのみで済む。
- **`ColumnScale::stds()`ゲッターを追加、可視性は`pub`のまま**: テストで`fit()`と同じ標準化・逆標準化の手順を独立に再現するために必要になった（元は`nonlinear/common.rs`内部でのみ使うprivateフィールドだったが、`destandardize_params`の逆方向の変換をテスト側で書くために公開した）。rust-reviewerからは「engine内部の実装詳細なので`pub(crate)`に絞るべき」という指摘があったが、実際に`pub(crate)`にすると`cargo clippy --all-targets -- -D warnings`の`lib`ターゲット（テストコードを含まないビルド）で`dead_code`エラーになった（唯一の呼び出し元が`logit.rs`の`#[cfg(test)] mod tests`のみで、`pub`アイテムはdead_code検出対象外という言語仕様上の扱いの違いによる）。ビルドを壊すため`pub`のまま据え置いた。

### テスト

- **`cov_type`ごとの独立再計算との一致**: `fit()`が内部で行う手順（標準化→収束点でのscores/Hessian評価→`common.rs`の共通行列演算→`destandardize_cov_params`）をテスト側で独立に再現し、`Opg`/`Hc0`/`Hc1`それぞれで`fit()`が返す`cov_params`と一致することを確認した（`fit_cov_type_opg_hc0_hc1_match_independently_recomputed_values`）。
  - **多変量（k=3）データセットが必須な理由**: 切片のみモデルでは情報行列の等式`Σᵢsᵢsᵢ' = -H`が有限標本で厳密に成り立ってしまい（`y_i∈{0,1}`かつ全観測で`p_i=ȳ`となる特殊性から`Σ(y_i-ȳ)² = n*ȳ(1-ȳ) = -H`が代数的に導ける）、`classical`/`opg`/`hc0`が偶然同じ値になる。そのため切片のみデータセットでは`fit()`の`match cov_type`の配線ミス（例えば`Opg`の枝で誤って`observed_information_cov_params`を呼ぶ等）を検出できない。実際に`Opg`の枝を意図的に壊して（`observed_information_cov_params`を呼ぶよう改変）このテストが失敗することを確認した上で、多変量データセットを採用した。
  - Hessianの符号規約に注意: `LogitProblem::hessian`（argminトレイト）はコスト関数（負の対数尤度）のHessianを返すため、`run_solver`が`SolverOutput.hessian`に格納する対数尤度そのもののHessianに合わせて、テスト側でも1回符号反転する必要がある（`run_solver`のdocコメント「Hessianトレイトの符号規約」と同じ変換）。
- **`method`×`cov_type`の組み合わせ**: 既存テストは`method`横断が`CovType::Classical`のみ、`cov_type`横断が`Method::Newton`のみで、両方を同時に変える組み合わせが未検証だった（rust-reviewer指摘）。`fit_non_classical_cov_types_work_with_bfgs_and_lbfgs`で、`Opg`/`Hc0`/`Hc1`それぞれについて`bfgs`/`lbfgs`の`cov_params`が`newton`の結果と一致することを確認した。
