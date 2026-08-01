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

`LogitEstimator::fit`に`cov_type: CovType`引数を追加した。`CovType`（`Classical`/`Opg`/`Hc0`/`Hc1`。`Cluster`はB7/Issue #60で追加）は`Method`と同じ理由で`nonlinear/common.rs`に定義し、Probit/Tobitでも再利用する想定。

- **収束点でのスコア評価に`LogitProblem`のクローンが必要**: `Opg`/`Hc0`/`Hc1`は収束点での観測ごとのスコア（`LogitProblem::scores`）が必要だが、`run_solver`は`problem`の所有権を取り込み、内部で保持していたモデルを呼び出し元へ返さない設計になっている（`SolverOutput`に`model`フィールドが無い）。`run_solver`のシグネチャを変更して`model`を返す設計も検討したが、`LogitProblem`は元々`argmin::core::Executor`向けに`Clone`を要求しているため、`run_solver`に渡す前に`problem.clone()`しておく方が`run_solver`（Logit/Probit/Tobit共通のユーティリティ）のシグネチャを変えずに済み、影響範囲が小さい。
  - **クローンは`cov_type=Classical`のときは行わない**（rust-reviewer指摘）: 初回実装では`cov_type`に関わらず常に`problem.clone()`していたが、`Classical`はスコアを一切使わないため設計行列を含む無駄な複製になる。`cov_type`に応じて`Option<LogitProblem>`で条件付きにクローンする形に修正した。
- **`cov_params_std`の計算はいずれも標準化空間で行ってから`destandardize_cov_params`で逆変換**: Issue #58で確立した「標準化空間で`Σ_std`を計算し、最後に一度だけ`destandardize_cov_params`で元のスケールへ変換する」設計をそのまま踏襲する。`opg_cov_params`/`sandwich_cov_params`（`nonlinear/common.rs`、Issue #53で実装済み）はいずれも標準化空間の`scores_std`・`hessian_std`を受け取ってΣ_stdを返すため、`cov_type`ごとの分岐は「どの共通関数を呼ぶか」の違いのみで済む。
- **`ColumnScale::stds()`ゲッターを追加、可視性は`pub`のまま**: テストで`fit()`と同じ標準化・逆標準化の手順を独立に再現するために必要になった（元は`nonlinear/common.rs`内部でのみ使うprivateフィールドだったが、`destandardize_params`の逆方向の変換をテスト側で書くために公開した）。rust-reviewerからは「engine内部の実装詳細なので`pub(crate)`に絞るべき」という指摘があったが、実際に`pub(crate)`にすると`cargo clippy --all-targets -- -D warnings`の`lib`ターゲット（テストコードを含まないビルド）で`dead_code`エラーになった（唯一の呼び出し元が`logit.rs`の`#[cfg(test)] mod tests`のみで、`pub`アイテムはdead_code検出対象外という言語仕様上の扱いの違いによる）。ビルドを壊すため`pub`のまま据え置いた。

### テスト

- **`cov_type`ごとの独立再計算との一致**: `fit()`が内部で行う手順（標準化→収束点でのscores/Hessian評価→`common.rs`の共通行列演算→`destandardize_cov_params`）をテスト側で独立に再現し、`Opg`/`Hc0`/`Hc1`それぞれで`fit()`が返す`cov_params`と一致することを確認した（`fit_cov_type_opg_hc0_hc1_match_independently_recomputed_values`）。
  - **多変量（k=3）データセットが必須な理由**: 切片のみモデルでは情報行列の等式`Σᵢsᵢsᵢ' = -H`が有限標本で厳密に成り立ってしまい（`y_i∈{0,1}`かつ全観測で`p_i=ȳ`となる特殊性から`Σ(y_i-ȳ)² = n*ȳ(1-ȳ) = -H`が代数的に導ける）、`classical`/`opg`/`hc0`が偶然同じ値になる。そのため切片のみデータセットでは`fit()`の`match cov_type`の配線ミス（例えば`Opg`の枝で誤って`observed_information_cov_params`を呼ぶ等）を検出できない。実際に`Opg`の枝を意図的に壊して（`observed_information_cov_params`を呼ぶよう改変）このテストが失敗することを確認した上で、多変量データセットを採用した。
  - Hessianの符号規約に注意: `LogitProblem::hessian`（argminトレイト）はコスト関数（負の対数尤度）のHessianを返すため、`run_solver`が`SolverOutput.hessian`に格納する対数尤度そのもののHessianに合わせて、テスト側でも1回符号反転する必要がある（`run_solver`のdocコメント「Hessianトレイトの符号規約」と同じ変換）。

## クラスターロバストSE（Issue #60で実装済み）

`CovType`に`Cluster { groups: Option<Vec<String>> }`を追加した（OLSの`CovType::Cluster`と同じ、フィールド付きバリアントの設計パターン）。クラスター単位の集約→サンドイッチ計算そのもの（`cluster_cov_params`）は`nonlinear/common.rs`にIssue #53時点で既に実装・テスト済みだったため、本Issueで新規に実装したのは(1)`CovType::Cluster`バリアントの追加、(2)`LogitEstimator::fit`への配線、(3)クラスターキー未指定・クラスター数不足のバリデーションの3点のみ。

- **クラスターグループの検証ロジックをOLSと共有**: `groups.len()==n`の内部契約チェック＋distinct count`>=2`の検証（`CommonError::MissingClusterColumn`/`InsufficientClusters`）は、OLSの`validate_cluster_groups`（`engine/src/linear/ols.rs`）と完全に同一のロジックだった。ユーザー確認の上、`engine::validation::validate_cluster_groups(groups: &[String], n: usize) -> Result<usize, CommonError>`として共有化し、OLS側もこの共有関数を呼ぶよう変更した（モデル固有の計算に一切依存しない純粋な検証ロジックのため）。
  - **配置場所を`engine/src/error.rs`から`engine/src/validation.rs`（新設）へ**: 当初`CommonError`と同じ`error.rs`に置いたが、rust-reviewerの指摘（`error.rs`冒頭のdocコメントは「エラー**型**の定義」とスコープを明記しており、検証**関数**を置く設計ではない。Issue #129で`ensure_well_conditioned_symmetric_matrix`を独立モジュール`engine::linear_algebra`に切り出した前例と一貫しない）を受けて、`engine::linear_algebra`と同じ位置付けの独立モジュール`engine/src/validation.rs`（系統をまたぐ入力バリデーションロジック集約）に移設した。
- **検証は`fit()`冒頭で早期に行う（OLSとは異なるタイミング）**: OLSは`cov_type=Cluster`の検証を残差計算後（事後）に行っている（OLSの`fit()`が閉形式解のため、検証タイミングを変えてもコストが変わらない）。Logitは反復最適化（Newton/BFGS/L-BFGS）のため、グループキー未指定・クラスター数不足を最適化の実行前（`n<=k`チェックの直後）に検証し、無駄な最適化計算を避ける設計にした。
- **`problem_for_scores`のクローン対象に`Cluster`を追加**: `Cluster`も収束点でのスコア（`LogitProblem::scores`）が必要なため、Issue #59で導入した「`cov_type`に応じた条件付きクローン」の対象に含めた。
- **`CovType`は`Copy`を外して`Clone`のみに変更**: `Cluster`が`Vec<String>`を持つフィールド付きバリアントになったため、既存の`Copy`実装が使えなくなった。同じ`cov_type`値を複数箇所で使うテストコードは`cov_type.clone()`で明示的に複製するよう修正した（本体側の`fit()`は`cov_type`を1回受け取って内部で使い切るのみのため影響なし）。

### テスト

- **独立再計算との一致**: `Opg`/`Hc0`/`Hc1`と同じ技法（`fit_cov_type_opg_hc0_hc1_match_independently_recomputed_values`）で、多変量（k=3）データセット・`cluster_cov_params`の直接呼び出しによる独立再計算と`fit()`の結果を突き合わせた（`fit_cov_type_cluster_matches_independently_recomputed_values`、2:2の均等サイズグループ）。`Cluster`の枝を意図的に`sandwich_cov_params`（Hc0）に差し替えてこのテストが失敗することを確認済み（配線ミスに対する検出力の確認）。
  - **不均衡なグループサイズのケースを追加**: rust-reviewerの指摘（`testing-policy.md`が指摘する通り、均等サイズのグループのみのテストは実務で起こりやすい偏ったグループサイズを見逃しうる。OLS側の対応するテストは2:3の不均衡を使っている）を受けて、3:2の不均衡なグループでも同じ独立再計算の技法で検証するテストを追加した（`fit_cov_type_cluster_matches_independently_recomputed_values_with_unbalanced_groups`）。
- **エラーハンドリング**: グループキー未指定（`fit_returns_missing_cluster_column_error_when_groups_not_provided`）・クラスター数1（`fit_returns_insufficient_clusters_error_when_only_one_group`）の2ケースを検証。
- **`bfgs`/`lbfgs`との組み合わせ**: Issue #59で追加した`fit_non_classical_cov_types_work_with_bfgs_and_lbfgs`のcov_typeの配列に`Cluster`を追加し、既存の`Opg`/`Hc0`/`Hc1`と同じ枠組みで検証した。
- **`method`×`cov_type`の組み合わせ**: 既存テストは`method`横断が`CovType::Classical`のみ、`cov_type`横断が`Method::Newton`のみで、両方を同時に変える組み合わせが未検証だった（rust-reviewer指摘）。`fit_non_classical_cov_types_work_with_bfgs_and_lbfgs`で、`Opg`/`Hc0`/`Hc1`それぞれについて`bfgs`/`lbfgs`の`cov_params`が`newton`の結果と一致することを確認した。

## 適合度統計量（Issue #61で実装済み）

`LogitEstimator`に`log_likelihood`/`log_likelihood_null`/`lr_statistic`/`lr_p_value`/`pseudo_r_squared`（McFadden）/`aic`/`bic`/`n_obs`/`df_model`/`df_resid`を追加した（`nonlinear-api-design.md`5章の仕様通り）。

- **`log_likelihood`**: `LogitProblem::cost`（`-ℓ(θ)`、argminの`CostFunction`）から自由関数`log_likelihood(x, y, params)`（`Result`を経由しない、収束後のパラメータで1回だけ評価する内部専用の計算）を切り出し、`cost`はこれを符号反転して呼ぶ形にリファクタリングした。`fit()`側は元スケールの`input.x()`/`input.y()`と`destandardize_params`済みの`params`を渡す（標準化空間を経由しない。`z_i=x_i'θ`は再パラメータ化に対して不変なため、標準化空間で評価しても値は変わらないが、`LogitProblem`のクローン（`cov_type`によっては行わない設計、Issue #59）に依存せず常に計算できる元スケール側を使う設計にした）。
- **`log_likelihood_null`（切片のみモデルのllf）**: `nonlinear-implementation-notes.md`「Logitのデータ構造」節で言及されていた「ソルバーの再フィット」ではなく、**閉じた形の解析解を直接計算する方式を採用した（ユーザー確認の上、`logit-probit-issue-breakdown.md`の当初案から変更）**。理由: 切片のみLogitは`p̂=ȳ`という閉じた形のMLEを持つ（既存テスト`fit_newton_converges_to_closed_form_solution_for_intercept_only_model`で検証済みの性質）ため、対数尤度も`ℓ_null = n1*ln(ȳ) + n0*ln(1-ȳ)`（`n1`/`n0`はy=1/0の観測数）という閉じた形で書け、再最適化を経由する必要がない。`n1`または`n0`が0のときの`0*ln(0)`（NaN）を避けるため、該当項を明示的に0として扱う（情報理論の`0 log 0 = 0`規約）。
- **`include_intercept=false`のときの非入れ子性**: `log_likelihood_null`は`include_intercept`の値に関わらず常に「切片のみ」モデルを参照する（`nonlinear-api-design.md`5章の定義通り、statsmodelsも`k_constant`の有無に関わらず同じ挙動）。そのため`include_intercept=false`でフィットしたモデルは、この「切片のみ」nullモデルの上位集合（入れ子）にならない。この場合`lr_statistic`が負になったり`lr_p_value`が統計的に意味の薄い値（ほぼ1.0）になったりしうるが、これはstatsmodels準拠の仕様上の挙動でありバグではない（rust-reviewerの指摘を受けてdocコメント・回帰テスト`fit_lr_statistic_can_be_negative_when_include_intercept_is_false`で明記・固定した）。
- **`df_model`の定義（`k-1`固定、OLSの`k-k_constant`とは異なる）**: `include_intercept`の値に関わらず常に`k-1`とする（ユーザー確認済み、statsmodels準拠）。`log_likelihood_null`が常に1パラメータ（切片）のnullモデルを参照するため、LR検定の自由度（本来の意味＝パラメータ数差）は`k-1`で統一するのが自然という判断。OLSの`df_model = k - k_constant`（`include_intercept=false`なら`k`）とは`include_intercept=false`のときに式が異なる点に注意（OLSのdf_modelは同一モデル内の「傾き係数の数」を表すのに対し、Logitのこのdf_modelは「フィット対象モデルと外部の別モデル（null）とのパラメータ数差」を表しており、意味論が異なるため）。
- **`df_model==0`時の`lr_p_value`**: OLSの`f_p_value`（`df_model==0`時にNaN、検定対象の傾き係数が存在しないため）と同じ扱いをユーザー確認の上で採用した。`ChiSquared::new`の`map_err`分岐は`df_model>0`が保証されているため理論上到達不能（`.claude/rules/rust-style.md`「テスト」のカバレッジ方針参照）。
- **`aic`/`bic`**: OLSと同じ式（`aic=-2ℓ+2k`、`bic=-2ℓ+ln(n)k`、`k`は定数項を含む全パラメータ数）。
- **`n_obs`はフィールドとして保持せず`self.input.nobs()`への委譲**: 当初`fit()`内のローカル変数`n`をそのままフィールドに複製していたが、rust-reviewerの指摘（`OlsEstimator`は同種の値を独自フィールドに持たず`input.nobs()`経由でアクセスさせる設計であり、同じ値の出どころが2つになるのは将来の不整合リスク）を受けて、`LogitEstimator::n_obs()`を`self.input.nobs()`へのdelegateに変更した（`n_obs`フィールド自体を削除）。

**発見した既存の問題（Issue #61のスコープ外、[Issue #130](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/130)としてトラッキング）**: `df_model = k.saturating_sub(1)`というアンダーフロー対策コードを書く過程で、`k=0`（`include_intercept=false`かつ説明変数も無い病的な入力）だと、この防御コードに到達する**前の段階**（`cov_params`計算経路、`ensure_well_conditioned_symmetric_matrix`周辺と推測）で`faer`が`attempt to subtract with overflow`というpanicを起こすことが判明した。`fit()`冒頭の`n<=k`チェックは`n>=1`であれば`k=0`でも通過してしまうため、`k>=1`の検証が欠けている。本Issueの差分（適合度統計量）とは無関係の既存コードの問題のため、ユーザー確認の上で別issue化し、`k=0`を直接検証するテストは追加していない。

### テスト

- **切片のみモデル（`df_model=0`の境界ケース）**: `fit_computes_goodness_of_fit_statistics_for_intercept_only_model`。`log_likelihood`と`log_likelihood_null`が定義上一致すること（同じ「切片のみ」モデルを参照するため）、`lr_statistic≈0`・`pseudo_r_squared≈0`・`lr_p_value`がNaNになることを検証。
- **多変量モデルでの独立再計算**: `fit_computes_goodness_of_fit_statistics_matching_independently_recomputed_values`。実装の`softplus`ベースの式とは異なる式（`logistic`から直接`Σ[y ln(p) + (1-y) ln(1-p)]`を計算するベルヌーイ対数尤度の定義式そのもの）で`log_likelihood`を独立に再計算し、`log_likelihood_null`・`lr_statistic`・`pseudo_r_squared`・`df_model`・`df_resid`・`lr_p_value`（`statrs::ChiSquared`で独立に検算）・`aic`/`bic`を突き合わせた（`fit_cov_params_is_symmetric_and_stats_are_internally_consistent`と同じデータセットを再利用）。
- **`include_intercept=false`での非入れ子挙動**: `fit_lr_statistic_can_be_negative_when_include_intercept_is_false`。`lr_statistic`が負になりうること（NaN/Infにはならないこと）、`df_model`/`df_resid`/`aic`/`bic`は`include_intercept`の値に関わらず同じ式で計算されることを回帰テストとして固定した。

## 限界効果（Issue #62で実装済み）

`LogitEstimator::marginal_effects(at, confidence_level)`を追加した（`fit()`とは独立した別メソッド、`nonlinear-api-design.md`6章）。設計方針の詳細（離散変数の自動判定なし・切片除外・代表点の定義・デルタ法ヤコビアンの統一形）は`nonlinear-implementation-notes.md`「限界効果」節（Logit/Probit/Tobit共通の方針）を参照。本節はLogit固有の実装判断のみ記す。

- **`w`・`s`の計算（`overall_w_and_s`/`at_point_w_and_s`）**: Logitのリンク関数`Λ`の微分`p(1-p)`を使い、`at="overall"`は全観測を1回走査した平均、`at="mean"`/`"median"`は代表点`x̄`での1点評価。いずれも`(w, s)`という同じ形の戻り値にまとめ、`dydx_and_jacobian`（`at`に依存しない共通関数、`g_j(θ)=w*θⱼ`・`∂g_j/∂θ_m=θⱼ*s_m+[j==m]*w`）に渡す設計にした。
- **分散の計算はk×kのヤコビアン行列を都度構築せず、変数`j`ごとに行ベクトルを取り出して二次形式`jac_j·Σ·jac_j'`を計算**（`Σ=cov_params`）。`Mat::from_fn`で一度k×kの`jacobian`を構築してから行を取り出す実装にした（k×k全体を毎回構築するオーバーヘッドはkが小さい想定のため許容、`.claude/rules/rust-style.md`「パフォーマンス」節の並列化検討対象（反復最適化中に繰り返し呼ばれる計算）には該当しない。呼び出し1回につき1回のみの評価のため）。
- **`column_medians`のNaN比較**: `partial_cmp().unwrap()`を使う（OLSの`time_ordering`と同じ正当化、`x`の値はNaN/無限大を含まないことが`engine_pybind::column_extraction`側で保証されている契約）。
- **statsmodelsとの数値照合（rust-reviewerが実施）**: `get_margeff(at='overall'/'mean', method='dydx', dummy=False)`とdydx・std_errが機械精度で一致することを確認済み。`dummy=False`がstatsmodelsの既定値であることも`inspect.signature`で確認済み。

### テスト

- **デルタ法ヤコビアンの数値微分検証**: `dydx_and_jacobian_matches_numerical_differentiation_for_overall_w_and_s`/`_for_at_point_w_and_s`。`hessian_matches_numerical_differentiation_of_gradient`と同じ技法（中心差分）で、`overall_w_and_s`/`at_point_w_and_s`が返す`(w,s)`から`dydx_and_jacobian`が計算するヤコビアンを、`w(θ)*θⱼ`を`θ`の関数として直接数値微分した値と比較した。
- **切片のみモデルでの空結果**: `marginal_effects_returns_empty_result_for_intercept_only_model`。定数項のみ（k=1、出力対象の説明変数が0個）でパニックしないことを確認する境界ケース。
- **独立再計算によるdydx・SEの検証**: `marginal_effects_overall_matches_independently_recomputed_dydx_and_delta_method_se`。`logistic`から直接計算した定義式（`overall_w_and_s`とは別の式）でdydxを再計算し、標準誤差も`dydx_j`をfit済みパラメータの周りで数値微分して得たヤコビアン行と`cov_params`の二次形式から独立に求めて突き合わせた（`dydx_and_jacobian`内の配線ミスを検出できる設計）。
- **`at="mean"`/`"median"`が`at="overall"`と異なる値になることの確認**: `marginal_effects_at_mean_differs_from_overall_and_matches_independent_recomputation`・`marginal_effects_at_median_differs_from_mean_and_overall_and_matches_independent_recomputation`。後者は`column_medians`（奇数・偶数nの両方を`column_medians_matches_expected_for_odd_and_even_n`で直接検証済み）が返す中央値を使い、平均・中央値が異なる非対称データセットで代表点が正しく切り替わることを確認した（rust-reviewer指摘、初回実装では`at="median"`のテストが皆無だった）。
- **`confidence_level`範囲外エラー**: `marginal_effects_returns_invalid_confidence_level_error_out_of_range`。`fit()`と同じ`CommonError::InvalidConfidenceLevel`を返すことを確認。

## predict() / pred_table()（Issue #63で実装済み）

`LogitEstimator::predict()`（引数なし、`Vec<f64>`を直接返す。エラーなし）と`LogitEstimator::pred_table(threshold)`（`Mat<f64>`の2×2的中表を直接返す。エラーなし）を追加した。いずれも`fit()`とは独立した別メソッド（`nonlinear-api-design.md`6章）。

- **対象データ範囲は学習データのみ（in-sample限定）**: `predict()`/`pred_table()`ともに、`fit()`に使った`self.input.x()`/`self.input.y()`に対してのみ計算する。新規データ（未知のX）を受け付けるout-of-sample対応は、着手前にユーザーへ確認し、本Issueのスコープ外として見送った（別GitHub issueとして追加作成予定）。
- **`predict()`は`p_i=Λ(x_i'θ)`をそのまま計算するのみ**で、バリデーションを要する引数が無いためエラーなし（`Result`を返さない設計）。
- **`threshold`の値域は検証しない**: `[0,1]`の範囲外でも`predicted`側が単に自明な分類結果（全て一方のクラスに分類される）になるだけで計算上破綻しないため（`confidence_level`とは異なり、範囲外でも統計的に無意味な値やNaN/panicを生まない）。statsmodelsも検証していない。

### バグ修正（rust-reviewerの指摘・statsmodelsとの数値照合で発覚）

初版実装は`pred_table`の実測クラス（`actual`）も`predicted`と同じ`threshold`で二値化していたが、statsmodelsの`BinaryResults.pred_table(threshold)`の実際のソース（`pred = (self.predict() > threshold)`で予測確率のみを`threshold`で二値化した**後**、`histogram2d(actual, pred, bins=[0, 0.5, 1])`で固定の0.5分割によりクロス集計する）をPythonで数値照合したところ、`threshold≠0.5`のとき初版実装がstatsmodelsと乖離することが判明した。`actual`は`threshold`に一切依存せず常に`0.5`で二値化するのが正しい仕様のため、`actual = if y_i >= 0.5 { 1 } else { 0 }`（`threshold`ではなく固定`0.5`）に修正した。`>=`（`>`ではなく）を使うのは、numpyの`histogram2d`のビン境界規約（最後のビンのみ右端を含む半開区間、`0.5`ちょうどの値は上側ビンに入る）に合わせるため。`y`が厳密に0/1でない場合（値域検証は未実装、`nonlinear-implementation-notes.md`参照）もこの規約で扱われる。

修正前の初回実装のテスト（4件）は全て`threshold=0.5`のみを使っていたため、この乖離を検出できていなかった（`y∈{0,1}`かつ`threshold=0.5`では`y_i>threshold`と`y_i>=0.5`が偶然一致するため）。

### テスト

- **`predict()`の閉じた形検証・独立再計算**: `predict_matches_closed_form_for_intercept_only_model`（切片のみモデルは全観測で`p_i=ȳ`）・`predict_matches_independently_recomputed_logistic_of_linear_predictor`（多変量モデルで`logistic`から直接計算した値と1e-12精度で突き合わせ）。
- **`pred_table`の手計算検証**: `pred_table_matches_hand_computed_counts_for_intercept_only_model`。切片のみモデル（全観測で`p_i=ȳ≈0.571`）を使い、`threshold=0.5`（全観測が予測クラス1）・`threshold=0.99`（全観測が予測クラス0）の2パターンで、手計算した期待値と一致することを確認。
- **`pred_table`の独立再計算**: `pred_table_matches_independently_recomputed_classification`。`threshold=0.2`（`0.5`以外の値、上記バグを検出できるようにするため）で、`predict()`の出力から独立に再計算した分類結果と突き合わせた。
- **`actual`クラスのカウントが`threshold`に対して不変であることの回帰テスト**: `pred_table_actual_class_counts_are_invariant_to_threshold`。`threshold∈{0.1,0.3,0.5,0.7,0.9}`の5パターンで、`actual0`/`actual1`の行合計が常に一定（`y=[0,1,0,1]`なので各2件）であることを確認し、上記バグの再発を防止する。

## engine単体テストのカバレッジ（Issue #64で確認・実装済み）

`cargo-llvm-cov -p engine --lib`で実測。OLSと同じ方針（100%は目指さず、理論上到達不能な防御的エラーパスはドキュメント化して受け入れる、`ols-implementation-notes.md`5章参照）。

**実測結果（129テスト時点）**: `nonlinear/logit.rs` Region 97.76%・Line 98.74%・Function 98.01%。`nonlinear/common.rs`（現時点でLogitのみが利用者のため、こちらも合わせて確認）はRegion 94.85%・Line 95.88%。

未カバー箇所を精査し、以下の対応を行った。

### 実データで起こりうる真のギャップ（テスト追加で対応済み）

- **`cov_type=Hc0`/`Hc1`/`Opg`/`Cluster`での`SingularHessian`/`SingularOpgMatrix`エラー伝播が一度も検証されていなかった**: 既存の`fit_returns_singular_hessian_error_for_perfectly_collinear_design_matrix_with_bfgs_and_lbfgs`（`CovType::Classical`のみ）に倣い、`fit_returns_singular_hessian_error_for_perfectly_collinear_design_matrix_with_hc0_and_hc1`・`fit_returns_singular_opg_matrix_error_for_perfectly_collinear_design_matrix`（`CovType::Opg`、`opg_cov_params`由来の別エラー型`SingularOpgMatrix`）・`_with_cluster`（`CovType::Cluster`）の3テストを追加した。
  - **`method=Newton`は使えない**: `newton_step`内の特異性検出（ピボット付きQR）が`cov_type`の分岐に到達する前（最適化中）に`SingularHessian`を返してしまうため、`Method::Bfgs`を使う必要がある（Issue #129で確立済みのClassical版と同じ理由）。**当初Hc0/Hc1のテストを`Method::Newton`で書いてしまい、実際にはこの経路を通れていなかったことが`cargo-llvm-cov`の再計測で発覚**した（テストは`Err(SingularHessian)`を返すには返すが、それは`cov_type`の分岐ではなくNewtonの最適化中に発生したものだった）。`Method::Bfgs`に修正して解決した。
  - **rust-reviewerの指摘でOpg/Clusterの同種ギャップを追加発見**: Hc0/Hc1の修正だけで一旦完了としたところ、rust-reviewerが`cargo-llvm-cov`のHTMLレポート（region単位のハイライト）まで確認し、`CovType::Opg`（`logit.rs`の`opg_cov_params(...)？`呼び出し）・`CovType::Cluster`（`cluster_cov_params(...)?`呼び出し）にも全く同じ構造の未検証パスが残っていることを指摘した。`opg_cov_params`は`neg_hessian_inverse`ではなく別の特異性検出（OPG行列自体の固有値判定）を使うため、返るエラー型は`SingularHessian`ではなく`SingularOpgMatrix`である点に注意（`common.rs`「OPG行列特異時のエラー型を分離」参照）。同じ完全な多重共線性データセット（`x2=2*x1`）を使えば、スコア行列`scores_i=(y_i-p_i)x_i`も同じ構造的多重共線性を持つ（列2=2×列1）ため、`Opg`側もこのデータセットで再現できた。
- **`log_likelihood_null`の`0*ln(0)`回避分岐（`n1`または`n0`が0の退化ケース）が未カバー**: `fit()`にインラインで書かれていたため、反復最適化を経由せずには独立にテストできなかった（全観測が同じyの完全分離データセットで`fit()`自体の収束が不安定になりうるため、`fit()`経由のテストは避けたい）。`log_likelihood`と同じ理由で`log_likelihood_null(y: &Mat<f64>) -> f64`という独立関数に切り出し（`fit()`はこれを呼ぶだけに変更、計算式・挙動は完全に同値）、全観測y=1・全観測y=0の両方の退化ケースを`fit()`を経由せず直接テストできるようにした（`log_likelihood_null_returns_zero_for_degenerate_all_same_y`）。

### 理論上到達不能な防御的エラーパス（受け入れて未カバーのまま、`common.rs`にdocコメントで明記済み）

`nonlinear/common.rs`（現時点でLogitのみが利用者）の以下の箇所。いずれも「argmin内部の契約により理論上失敗し得ないはずだが、防御的に`Result`化してある」という同じ性質（OLSの`xtx_inverse`等と同じカテゴリ）。

- `extract_outcome`の`state.get_best_param()`/`problem.take_problem()`の`None`分岐: argmin 0.11.0のソース（`IterState::update()`）を実際に確認した上で、`FaerNewton::init`が必ず初期パラメータを設定すること・`take_problem()`は1回しか呼ばないことから理論上到達不能と判断した。
- `convert_optimizer_error`の`Err(other)`分岐: 本プロジェクトが制御する全エラー経路は`MleError`のみを送出するため、`downcast`は常に成功する。argmin自体の内部エラーに備えた防御的フォールバック。
- `FaerNewton::name()`: argminの`Observer`（進捗ロギング機構）を使っていないため呼ばれない。分岐を持たない定型実装。
- `FaerNewton::init()`の`state.take_param()`の`None`分岐: `run_solver`が`Executor::configure`で`init()`実行前に必ず初期パラメータを設定するため到達不能。

### その他（対象外）

- `assert!`マクロのメッセージ引数（アサーション失敗時のみ評価される）は、`cargo-llvm-cov`上「未カバー」に見えるが実際のギャップではない（OLSと同じ誤検知パターン、`ols-implementation-notes.md`5章参照）。
- `linear/ols.rs`・`linear_algebra.rs`側の未カバー箇所はOLS側の既存スコープ（`ols-implementation-notes.md`参照）であり、本Issueの対象外。

## engine_pybind: データ抽出・LogitOptions/LogitResult pyclass定義（Issue #65で実装済み）

`engine_pybind/src/nonlinear/{mod.rs, common.rs, logit.rs}`を新設した。`engine_pybind/src/discrete_choice/`（未使用の`.gitkeep`のみのプレースホルダー）は削除し、`.claude/rules/rust-style.md`が想定する`nonlinear/`（`engine`側と同じ名称）に統一した（ユーザー確認済み）。

- **スコープの区切り方（ユーザー確認済み）**: `LogitEstimator::fit()`の呼び出し・`LogitResult`の実際の構築・`#[pymodule]`への`fit_logit`関数登録はIssue #66（engine呼び出し・エラー変換）に送り、本Issueでは行わない。代わりに、データ抽出・バリデーション・`engine::nonlinear::logit::LogitInput`構築までを行う`pub(crate) fn build_logit_input(df: &DataFrame, y: String, x: Vec<String>, options: &LogitOptions) -> PyResult<(LogitInput, EngineCovType, EngineMethod)>`を切り出した。OLSの`fit()`と同じ検証（y/x重複、`include_intercept=true`時の`"const"`列衝突、列抽出、`cov_type`/`method`文字列パース）をここに集約している。
- **`LogitOptions`から`start_params`を除外**: Issue本文のフィールド一覧には`start_params`が含まれていたが、`LogitEstimator::fit()`（engine側）は`start_params`引数を受け付けない（Issue #56で見送り済み、初期値は常にゼロベクトル固定）。ユーザー確認の上、今回は`LogitOptions`から完全に除外した。将来Issue #56の見送りが解除された時点で別issueとして追加する。
- **`LogitResult`は個別`#[pyo3(get)]`方式**: Issue本文は`get_all`と書いていたが、`OLSResult`の既存実装（フィールドごとに個別`#[pyo3(get)]`）と一貫性を持たせるため、そちらに揃えた。将来`marginal_effects`/`predict`/`pred_table`用の内部専用フィールド（`cov_params`等）を追加する際、`OLSResult`の`fitted_values`/`has_intercept`と同様に`#[pyo3(get)]`を付けずに済ませられるようにするため。
- **`mle_error_to_pyerr`のエラー型マッピング**: `nonlinear-implementation-notes.md`の対応表通り、`NonConvergence`/`SingularHessian`/`SingularOpgMatrix`→`ComputationError`、`InvalidMaxIter`/`InvalidCensoringBounds`→`ValidationError`、`Common`は`common_error_to_pyerr`に委譲。

### 構造的な発見・修正（`engine_pybind`クレート全体に影響）

- **`cargo test -p engine_pybind`が実行できない構造的制約を発見・修正した**: 当初`engine_pybind/Cargo.toml`は`pyo3 = { features = ["extension-module"] }`・`crate-type = ["cdylib"]`だった。`extension-module`はPythonインタプリタにdlopenされる前提でlibpythonへの静的リンクを意図的に省く仕様のため、`cargo test`が生成する単体テストバイナリ（dlopenされず自身で起動する）とリンクできず、`PyExc_TypeError`等のCPython C APIシンボル未定義でビルド失敗する（これがOLS側に`#[cfg(test)]`が1件も無かった実際の理由と推測される）。ユーザー確認の上、以下の構造的修正を行った:
  - `crate-type`に`"rlib"`を追加（`["cdylib", "rlib"]`。`cargo test`が生成する単体テストバイナリがクレートをリンクするために必要）
  - `pyo3`依存から`features = ["extension-module"]`を削除。`pyproject.toml`の`[tool.maturin] features = ["pyo3/extension-module"]`が`maturin build`/`maturin develop`実行時に外部からこのフィーチャを有効化する既存の仕組みがあるため、配布物（wheel）には引き続き反映される（`cargo`単体のビルド・テストとは別経路）。
  - 修正後、`uv run maturin build --release`でのwheelビルド成功・`uv run maturin develop --release`後の`uv run pytest tests/api_tests`（271件）全通過を確認し、OLS/WLSの既存機能に回帰が無いことを確認済み。
  - `.github/workflows/ci_python.yml`の`engine_pybind-lint`ジョブに`cargo test -p engine_pybind`ステップを追加した（rust-reviewerの指摘: 構造的修正をしても、CIがこれを実行しなければリグレッション検知に活かせないため）。
- **`dead_code`警告への対処は`#[allow(dead_code)]`を採用**: `build_logit_input`・`parse_cov_type`・`parse_method`・`mle_error_to_pyerr`はIssue #65時点では`#[cfg(test)] mod tests`からのみ呼ばれ、`cargo build`（テストコードを含まないlibターゲットのビルド）からは到達不能に見えるため`dead_code`警告になる。当初`engine`側の`ColumnScale::stds()`と同じ手法（`pub`にして回避）を試みたが、rust-reviewerの指摘（`engine_pybind`はPython拡張モジュール専用の薄いバインディング層であり、クレート外に`pub`なRust APIを公開する設計ではない。`engine`とは事情が異なる）を受けて、各関数に`#[allow(dead_code)]`＋理由コメント（Issue #66で実際に呼ばれるようになったら属性を削除する旨）を付ける方式に変更した。
- **y/x列間の行数不一致チェックは理論上到達不能と判明（Issue #65時点は防御的に残す判断だったが、Issue #133で削除に方針転換）**: `build_logit_input`（およびOLSの`fit()`）の`s.len() != n`チェックについて、polars 0.54.4の`DataFrame::new(height, columns)`が構築時に全列の長さが`height`と一致することを強制する（`validate_columns_slice`）ため、同一`DataFrame`内の列同士で行数が食い違う状態はAPI上構築できないことが判明した（Python側の`polars.DataFrame`も同じ不変条件）。Issue #65時点ではユーザー確認の上、検証コードを防御的に残しつつdocコメントで理由を明記しテストは作成しない方針だったが、Issue #133で再検証（`with_column`/`hstack`/`replace`等の変更系メソッドも同じ不変条件を強制することをソースで確認）した上でユーザー承認を得て、この種のチェックはOLS/WLS/Logitの3ファイルすべてから削除した（`parse_cov_type`の`n: usize`引数も不要になったため削除。`Option::map().transpose()?`パターンに簡略化）。

### テスト

`engine_pybind/src/nonlinear/logit.rs`の`#[cfg(test)] mod tests`に、`build_logit_input`の検証を12件実装した（`Series`/`DataFrame`を直接組み立てる、`polars::df!`マクロ利用。Pythonインタプリタ（GIL）を起動せずに検証できる設計、ファイル冒頭のdocコメント参照）。

- 成功パス: 切片あり・切片なしそれぞれでの`LogitInput`構築、`cov_type`/`method`の正しいパース
- `ValidationError`パス: 空の`x`、y/xの重複、x内の重複、`"const"`列衝突、存在しない列、欠損値、未知の`cov_type`文字列、未知の`method`文字列
- クラスター系: `cluster_col`指定時のグループキー抽出、未指定時に`groups=None`のまま返す（`engine`側の`MissingClusterColumn`検証に委ねる設計）

### バリデーションロジックの共通化（Issue #134で実装済み）

`build_logit_input`冒頭の`x`空チェック・y/xの重複チェック・x内重複チェック・`"const"`列衝突チェックは、OLS/WLSの`fit()`とメッセージ文言まで完全に重複していた。`engine_pybind/src/validation.rs`（クレート直下、`column_extraction.rs`と同じ位置づけ）に集約し、`validate_x_non_empty`/`validate_no_duplicate_x`/`validate_no_const_collision`/`validate_no_duplicate_roles`の4関数に切り出した。

- **`validate_no_duplicate_roles`**: `y`（Logitは`y`のみ、WLSは`y`+`weight`）のように`x`とは別に単一列名を指定する「ロール」同士、および各ロールと`x`との重複を検証する汎用関数。`roles: &[(&str, &str)]`というペアのリストを受け取る設計にした（将来IV等でロールが増えても流用できるようにするため。ただし`x`のように複数列を取るロール、例えば将来のIVの`instrument`にはそのままでは使えない。着手時に再検討する）。
- **判定順序のリグレッションと修正**: 初回実装では「ロール同士の全ペアを先に判定→`x`との重複を後で判定」という2段構成にしたが、これはWLSの旧実装（`x.contains(&y)` → `weight == y` → `x.contains(&weight)`の順に1つずつ判定）と異なる優先順位になってしまう（`weight == y`かつ`x`に`y`も含むような複合違反で返るメッセージが変わる）ことがrust-reviewerのレビューで判明した。ロールを先頭から順に処理し、各ロールについて「それより前のロールとの重複」→「`x`との重複」の順に確認する実装に修正し、旧実装と完全に同じ優先順位・メッセージになることを確認した。
- **`PyErr`はGILがないとメッセージを文字列化できない**: `PyErr::to_string()`（`Display`実装）はPython interpreterのGIL取得を要求するため、`#[cfg(test)]`（Python未初期化）で呼ぶとpanicする。判定ロジック・メッセージ文言そのものを`PyErr`に依存しない純粋関数`find_duplicate_role_message(roles, x) -> Option<String>`に切り出し、これをテストすることでGILなしにメッセージ文言・優先順位を単体テストで固定した（`validate_no_duplicate_roles`はこれを`ValidationError`でラップするだけの薄い関数）。
- `cov_type`/`method`文字列パースは対象外（戻り値の型が系統ごとに異なるため。`engine_pybind/src/linear/CLAUDE.md`参照）。

### engine呼び出し・`fit_logit`登録（Issue #66で実装済み）

`build_logit_input`の後段として`engine_pybind/src/nonlinear/logit.rs`に`pub(crate) fn fit(data: PyDataFrame, y, x, options: &LogitOptions) -> PyResult<LogitResult>`を追加した。`build_logit_input`→`engine::nonlinear::logit::LogitEstimator::fit`（`MleError`は`mle_error_to_pyerr`で変換）→`LogitResult`構築、という流れ（`ols.rs`の`fit`と同じ構造）。`engine_pybind/src/lib.rs`に`#[pyfunction] fit_logit`を追加してこれに委譲し、`#[pymodule]`に`fit_logit`/`LogitOptions`/`LogitResult`を登録した。Issue #65時点で残していた`#[allow(dead_code)]`（`build_logit_input`/`parse_cov_type`/`parse_method`/`mle_error_to_pyerr`）は、これらが実際に呼ばれるようになったため全て削除した。

- **テストは追加していない（OLSの前例通り）**: `fit`/`fit_logit`は`PyDataFrame`を引数に取るため、`build_logit_input`と異なりGILなしの`#[cfg(test)]`では直接呼べない。OLSの同等Issue（`cd2f662`「OLSのengine呼び出し・エラー変換を実装」）も、この段階ではテストを追加せず、後続の`python_package`ラッパー・statsmodelsベンチマークIssue（`d412c53`、Issue #19相当）で初めてpytestのカバレッジが付いたことを`git log`で確認済み。Logitも同じ流れ（Issue #66→#67→#68）を踏襲する。
- **数値検証**: `engine`のユニットテスト`fit_cov_type_opg_hc0_hc1_match_independently_recomputed_values`・`fit_cov_type_cluster_matches_independently_recomputed_values`と同じ入力（`y=[0,1,0,1]`、`x1=[10,20,30,40]`、`x2=[-5,2,8,-1]`、cluster`=[a,a,b,b]`、`tol=1e-8`）で、一時的なRustテスト（コミット対象外）から`params`/`std_errors`のオラクル値を出力させ、`uv run maturin develop --release`でビルドした`fit_logit`をPythonから同じデータで呼び出した結果と突き合わせ、classical/opg/hc0/hc1/clusterの全cov_typeで一致することを確認した（完了条件通り）。
- **`y`の値域検証（単位区間`[0,1]`）が未実装であることが判明**: rust-reviewerの指摘で、`fit_logit`のdocstringに「yは0.0/1.0でなければならない」と書いたところ、実際には`LogitInput::from_columns`にその検証が無いことが分かった。これはIssue #54時点で「B2（尤度・スコア・Hessian実装）に持ち越し」と明記されていたにもかかわらず、B2以降も未実装のまま残っていた既存のギャップ（本Issueが原因ではない）。ユーザー確認の上、docstringから未実装の保証を外し、検証実装自体はIssue #135として切り出した。

## engine_pybind: predict()/pred_table()/marginal_effects()の公開、python_packageラッパー（Issue #67で実装済み）

Issue #67は当初「python_packageラッパー実装」というタイトルだったが、着手前の調査で`LogitResult`（Issue #65/#66）に`predict`/`pred_table`/`marginal_effects`のpymethodsが1つも無いことが判明した（engine側の`LogitEstimator::predict()`/`pred_table()`/`marginal_effects()`はIssue #62/#63で実装済みだったが、engine_pybind層の配線が無かった）。OLSの`predict()`実装（engine拡張・engine_pybind配線・python_packageラッパーを1つのIssueにまとめたコミット`c6caed7`、Issue #86）と同じ方針で、engine_pybind配線をIssue #67のスコープに含めてまとめて実装することをユーザー確認済み。

### engine_pybind側

- `LogitResult`に非公開フィールド`estimator: LogitEstimator`を追加し、`#[pymethods] impl LogitResult`で`predict()`/`pred_table(threshold=0.5)`/`marginal_effects(at="overall", confidence_level=0.95)`を実装（詳細な設計判断は`engine_pybind/src/nonlinear/CLAUDE.md`参照）。
- `marginal_effects`の結果用に新規pyclass`MarginalEffectsResult`を追加。`at`文字列パースは`parse_marginal_effects_at`。
- `pred_table`の2×2`Mat<f64>`をPython向けの`Vec<Vec<f64>>`に変換する`mat_to_nested_vec`を`nonlinear/common.rs`に追加。
- 数値検証: `y=[0,1,0,1]`・`x1=[10,20,30,40]`・`x2=[-5,2,8,-1]`・`tol=1e-8`のデータで、一時的なRustテスト（コミット対象外）から`predict()`/`pred_table(0.5)`/`marginal_effects(at="overall")`のオラクル値を取得し、`uv run maturin develop --release`でビルドした結果と突き合わせて一致を確認した。`at="mean"`/`"median"`が`"overall"`と異なる値になること、`at`不正値・`confidence_level`範囲外で`ValidationError`になることも確認した。

### python_package側

- `python_package/econometricsmodels/nonlinear/logit.py`に`Logit`/`LogitResults`を実装（`OLS`/`OlsResults`と同型）。`LogitOptions`は`_lib`から再輸出。`coef_table()`は`OlsResults`と同形状だが`t_stat`ではなく`z_stat`。
- `predict()`は`[{"probability": p}, ...]`（OLSの`{"fitted": v}`に倣った命名）。
- `pred_table()`は`[{"actual": 0, "predicted_0": .., "predicted_1": ..}, {"actual": 1, ...}]`という行指向`list[dict]`（仕様書に明記の無い実装判断、ユーザー確認済み。詳細は`python_package/econometricsmodels/nonlinear/CLAUDE.md`参照）。
- `marginal_effects()`のキー（`param`/`dydx`/`std_err`/`z`/`p_value`/`conf_low`/`conf_high`）は`nonlinear-api-design.md`6章の確定済み命名をそのまま使用（`coef_table()`の`conf_lower`/`conf_upper`とは意図的に異なる）。

### テストに関する方針転換

Issue #66では「OLSの`engine呼び出し`Issue（`cd2f662`）もテスト無しでマージされた」という前例を根拠にテスト追加を見送ったが、Issue #67のレビュー（rust-reviewer・python-reviewer双方）で、この前例の理解が不正確だったことが判明した。実際には`2d554e4`（OLSのpython_packageラッパー実装コミット）は`tests/api_tests/test_ols.py`を「確定済み設計に合わせて全面的に書き直し」ており、ラッパー実装時点でテストが完全にゼロだったわけではない（統計的な数値比較=Issue #19相当は別issueだが、構造・APIレベルのテストはラッパーと同時だった）。Logitは`test_logit.py`自体が存在しない状態だったため、ユーザー確認の上、`tests/api_tests/test_logit.py`（構造・API・エラーパスのスモークテスト25件）を本Issueで追加した。statsmodelsとの厳密な数値比較（フィクスチャ化）は引き続きIssue #68で行う。

### レビューで対応した軽微な指摘

- `parse_marginal_effects_at`に直接の単体テストが無かった（`parse_cov_type`/`parse_method`と異なり`build_logit_input`経由の間接テストも無い）ため、`#[cfg(test)]`ユニットテストを2件追加。
- `python_package/econometricsmodels/nonlinear/logit.py`の新規docstring1行が79文字制限（`.claude/rules/python-style.md`）を超過していたため修正。
- `engine::nonlinear::logit::LogitEstimator`のdocコメントに「`predict`/`pred_table`は未実装」という古い記述が残っていたため修正（diff範囲外だが機会があったため合わせて対応）。
- `engine_pybind/src/nonlinear/CLAUDE.md`・`python_package/econometricsmodels/nonlinear/CLAUDE.md`を新規作成し、本Issueまでに蓄積した設計判断を集約した（Probit実装時の調査コスト削減のため、両レビュアーからの提案）。

## statsmodels/R glmとの数値照合ベンチマーク作成（Issue #68で実装済み）

`/test-new`スキル・`.claude/skills/reference-benchmark/`の手順に沿い、`benchmark/nonlinear/`を新設した（`engine`/`engine_pybind`と同じ系統ディレクトリ構成、`nonlinear`系統のベンチマークは本Issueが初）。

### 合成データセット: 真のlogit DGPで新規設計（ユーザー確認済み）

OLSの`generate_synthetic_datasets.py`の9シナリオは誤差項の分散構造（不均一分散・自己相関）に依存するものが多く、そのままlogitの2値DGPに転用できない。`benchmark/nonlinear/generate_logit_datasets.py`を新設し、`sigmoid(Xβ)`からのベルヌーイ乱数で2値yを生成する、logit向けに再設計した7シナリオを採用した。

- **踏襲**: `baseline`/`small_n`/`moderate_multicollinearity`/`high_condition_number`/`perfect_multicollinearity`（OLSと同じ意図）
- **除外**: `heteroskedastic`/`autocorrelated`/`high_variance`（OLSの誤差項構造に依存し、logitのDGPには直接対応しない）
- **新規追加**: `near_separation`（logit特有の病理、準完全分離。`x1`の係数を極端に大きくし、`x1`の値域のほとんどでp≈0/1になる状況を作る。収束はするが標準誤差が大きく膨らむ、成功パスの数値比較対象）

当初「完全分離でNonConvergenceになるエラーパス」シナリオ（`complete_separation`）も検討したが、下記「tolの妥当性検証」で判明した既知の限界により本実装では狙い通りに動作しないため不採用（ユーザー確認済み、Issue #138としてフォローアップ）。NonConvergence自体の発生確認は`tests/api_tests/test_logit.py`で`LogitOptions(max_iter=1)`により人為的に確認する。

実データは`load_wooldridge.py`の`SUGGESTED_DATASETS["probit_logit"]`で候補指定済みの`mroz`（労働参加モデル、Wooldridge教科書の定番例）を採用（ユーザー確認済み）。回帰式は`inlf ~ nwifeinc + educ + exper + expersq + age + kidslt6 + kidsge6`。クラスターロバストSEの実データ検証（`testing-policy.md`「テスト用データセット」3.）は、mrozの`city`（都市部居住ダミー、484/269の2値）を使う（OLSのwage1/regionクラスターと同じ趣旨）。

#### `scale_variance`: OLSからの単純な移植では動作しないと判明（レビューで発覚・対応済み）

`heteroskedastic`/`autocorrelated`/`high_variance`とは異なり、`scale_variance`（変数間のスケールが極端に異なるケース、`x1*=1e6, x2*=1e-3`）は誤差項構造ではなく設計行列のスケールの問題であり、当初「誤差項構造に依存するため転用不可」として一括除外していたのは技術的に不正確だった（python-reviewer・testing-completeness-reviewerの双方から指摘、ユーザー確認の上で再検討）。

素直にOLSの実装（p計算前に`X[:,0]*=1e6, X[:,1]*=1e-3`）を移植すると、sigmoidの非線形性によりスケールの大きいx1が線形予測子を支配し、ほぼ完全分離を起こしてしまい（`near_separation`と交絡し、設計行列のスケール自体を検証する意図が果たせない）、本来の目的を果たせないことが実装時に判明した。このため、**真のDGP（p・yの生成）は未スケーリングのXで行い、出力直前にのみ列とtrue_betaをスケーリングする**設計にした（`x_scaled @ beta_scaled == x_raw @ beta_raw`が成立するように、真の係数も逆スケーリングして返す）。この設計により、yはスケールの影響を受けず、推定側が読む設計行列のみが極端なスケール差を持つ、意図通りの数値比較シナリオになる（statsmodels・本実装とも収束し、baselineと同じ元データから復元される係数がスケールに応じて正しく変換されることを確認済み）。

Rクロスチェック（`run_glm_crosscheck_benchmark.R`）では、`opg`の手計算（`solve(t(scores) %*% scores)`）がこのシナリオで"computationally singular"エラーになることが判明した（スコア行列の列スケール差がそのまま見かけの条件数を悪化させるため、実際の条件数は1e18程度だが真に悪条件ではない）。列を各々のノルムで正規化してから反転し、`Σ=D⁻¹(D⁻¹MD⁻¹)⁻¹D⁻¹`の恒等式で元のスケールに戻す方式（本実装の`standardize_columns`/`destandardize_params`と同じ発想）に変更し、numpyの素の反転結果と完全一致することを確認した上で採用した。

#### `n=k+1`境界ケース（OLSの`baseline_df1`相当）は非採用（レビューで検討・ユーザー確認済み）

OLSには自由度がちょうど1（`n=k+1`）の境界値ケースがあるが、Logitには同型のシナリオを追加しなかった。実機確認により、5観測4パラメータ（`k=3`、切片含む）のデータでは、乱数シードを変えても常に完全ないし準完全分離が発生し、係数が数十〜百のオーダーまで発散することが判明した。これはOLSの閉形式解（`n=k+1`でも一意に定まる）とは根本的に事情が異なり、**ロジスティック回帰のMLEは`n`がパラメータ数`k`以下（またはそれに近い）だと構造的にほぼ確実に完全分離を起こす**という統計理論上の一般的な事実に起因する（`near_separation`シナリオで既に同種の境界的性質を検証済み）。意味のある「小標本の成功パス」を作れないため、構造的な差異として本ノートに記録し追加しない。

### `cov_type="hc1"`はstatsmodelsのdiscrete modelで未実装と判明（ユーザー確認済み・重要）

ベンチマーク作成中、`Logit.fit(cov_type="HC0")`と`"HC1"`が完全に同一の`bse`を返すことが発覚した。原因を`statsmodels.base.covtype.get_robustcov_results`のソースで追跡したところ、`HC1`指定時は`getattr(self, "cov_HC1", None)`でモデル固有の補正済みプロパティ（`n/(n-k)`小標本補正）を探すが、これは`RegressionResults`（OLS等）にしか定義されておらず、`LogitResults`は未定義のため、補正なしの`cov_white_simple(use_correction=False)`（`HC0`と同じ計算）に暗黙にフォールバックすることを確認した（`hasattr(LogitResults, "cov_HC1")`が`False`であることを実機確認）。

一方R（`sandwich::vcovHC(glm_obj, type="HC1")`）は`n/(n-k)`補正を正しく適用し、本実装が既に実装済みの式（`nonlinear-implementation-notes.md`「標準誤差の技術仕様」、`hc0`の`Σ`に`n/(n-k)`を乗じる）と一致することを実機確認した（本実装の`hc1`標準誤差はRの値と機械精度で一致、statsmodelsの`hc1`＝`hc0`値とは一致しない）。

このため`cov_type="hc1"`に限り、Rを主リファレンスとして扱う（ユーザー確認済み）。`benchmark/nonlinear/fixtures/generate_logit_fixtures.py`（statsmodels）はCOV_TYPESから`hc1`を除外し、`generate_logit_crosscheck_fixtures.py`（R）が`hc1`の数値比較の主リファレンスを担う。`test_logit_fixtures.py`は`hc1`を検証対象に含めず、`test_logit_crosscheck.py`が担う。

### `cov_type="opg"`はstatsmodelsのdiscrete modelでネイティブ非対応

`Logit.fit(cov_type="opg")`は"cov_type not recognized"エラーになる（`opg`/`oim`は`GenericLikelihoodModel`系専用で、組み込み`DiscreteModel`では非対応）。このため`opg`の係数標準誤差は`model.score_obs(params)`（statsmodels自身が検証済みのスコア計算）から`Σ = (Σᵢ sᵢsᵢ')⁻¹`を手計算する方式にした。

`opg`の限界効果（`get_margeff()`）はさらに難しく、fit済み結果の`cov_params_default`を事後的に上書きしてもキャッシュされた内部プロパティ（`bse`/`cov_params()`双方）に反映されないことが実装時に判明した（原因はstatsmodels内部のキャッシュ機構の詳細に依存）。このため`opg`の限界効果はstatsmodels側では算出せず、R（`marginaleffects`パッケージ、`vcov=`引数でカスタム共分散行列を直接渡せる）を唯一の数値照合対象とした。

### R側の限界効果リファレンス: `margins`ではなく`marginaleffects`を採用（ユーザー確認済み）

両パッケージを実機比較し、AME（overall）・at meanの数値がstatsmodelsの`get_margeff()`と一致することを確認した上で、より活発にメンテナンスされている`marginaleffects`（v0.32.0、2026-02-14公開）を採用した（`margins`はv0.3.28、2024-07-31が最新でメンテナ交代済み）。`marginaleffects`はtidyなdata.frameを返しJSON化しやすい点も採用理由。devcontainerの`Dockerfile`に`marginaleffects`を追加した。

**`datagrid()`の"mean"/"median"ショートカット文字列は使わない（重要なバグ発見）**: `slopes(model, newdata="mean")`のようなショートカット文字列は、整数のみを値に持つ数値列（Wooldridge `mroz`データの`age`/`educ`/`exper`等）を`FUN_integer`（既定で`round(mean(x))`相当）により丸めてしまい、本実装・statsmodelsが使う「生の標本平均・中央値」（`nonlinear-implementation-notes.md`「限界効果」節の`at="mean"`/`"median"`の代表点の定義）と評価点がずれることが判明した（mrozデータの`age`のdydxで最大約7%の乖離として顕在化）。`datagrid(model=model, FUN_numeric=mean, FUN_integer=mean)`のように`FUN_numeric`・`FUN_integer`を両方明示することで、全列を統一的に生の平均・中央値にできる（`run_glm_crosscheck_benchmark.R`参照）。

### 収束判定`tol`の妥当性検証（結論: 既定値`1e-6`を維持）

Issue #52策定時点で「最終的な妥当性判断はLogit/Probit実装・テスト段階に持ち越す」とされていた事項（`nonlinear-implementation-notes.md`「収束判定の`tol`」）。本Issueでの数値照合結果から、以下の結論に至った。

1. **通常のデータでは`tol=1e-6`で高精度に一致**: `baseline`/`small_n`/`moderate_multicollinearity`/`high_condition_number`シナリオおよびWooldridge実データ（`mroz`）では、statsmodelsとの係数・標準誤差の相対誤差は最大でも約6e-7（実測）で、`RTOL=1e-8`の基本方針を余裕をもって満たす。
2. **`near_separation`（準完全分離の境界ケース）では`tol=1e-6`だと精度不足**: statsmodelsとの相対誤差が最大約7e-8となり、`RTOL=1e-8`をわずかに超える。`tol=1e-8`まで締めると一致精度が大幅に改善する（実測diff: 2e-8→4e-13）。
3. **既定値は`1e-6`のまま変更しない（ユーザー確認済み）**: `tol=1e-8`に締めると、`bfgs`法が`max_iter=35`のうち34回を要するようになることを実測確認した。より難しいデータセットでは`bfgs`/`lbfgs`が`max_iter`不足によるNonConvergenceを起こすリスクが上がるため、全ユーザー向けの既定値としては`1e-6`の方が安全側と判断した。`near_separation`の数値比較テスト（`test_logit_fixtures.py`/`test_logit_crosscheck.py`）に限り、`LogitOptions(tol=1e-8)`を明示指定することで対応する。
4. **既知の限界（未修正、Issue #138）**: `tol`の値によらず、完全分離に近いデータでは勾配ノルム基準がスコア項`p(1-p)`の浮動小数点アンダーフローにより「収束済み」と誤判定しうる（実機確認: `x1`係数を100〜2000まで極端化しても常に`converged=True`、有限だが非常に大きい値を返す。statsmodelsは同程度のデータで非収束またはHessian特異エラーになる）。この問題は`tol`の調整では解決しない（アンダーフローは任意の`tol>0`で発生し得る）ため、既知の限界として記録し別issueで対処を検討する。

### テスト・フィクスチャ

- `tests/api_tests/fixtures/benchmarks/logit.json`（statsmodels主リファレンス）・`logit_crosscheck.json`（Rクロスチェック）を新規作成。
- `tests/api_tests/test_logit_fixtures.py`（26件）・`test_logit_crosscheck.py`（32件）を新規作成。許容誤差はOLSの基本方針（`RTOL=1e-8`）を`test_logit_fixtures.py`（statsmodels）に適用、`test_logit_crosscheck.py`（R、反復最適化同士の比較のため機械精度一致は期待できない）は実測最大相対誤差（約9.5e-5、near_separation）に基づき`RTOL=2e-4`を基本としつつ、以下は実測値に基づき個別に緩めた（`testing-policy.md`「許容誤差」の方針通り）:
  - 限界効果の`std_err`: `RTOL=5e-3`（デルタ法のヤコビアン経由でノイズが1桁大きい、実測最大~1.8e-3）
  - p値: `ATOL=3e-5`（標準正規分布CDFの裾で係数のわずかな差が増幅される、実測最大絶対誤差~1.19e-5）
  - `near_separation`の信頼区間: `RTOL=6e-4`（実測最大相対誤差~4.05e-4）
- `test_logit.py`に`test_non_convergence_raises_computation_error_with_tiny_max_iter`（`LogitOptions(max_iter=1)`によるNonConvergence確認）・`test_raise_on_non_convergence_false_returns_result_without_raising`（同じ設定で`raise_on_non_convergence=False`なら例外を投げず`converged=False`を返すことの確認）・`test_method_option_converges_to_same_params`（`method="newton"/"bfgs"/"lbfgs"`のAPIレベル成功パス、3手法が同じ解に収束することを確認。engine単体テストは3手法の一致を検証済みだが、engine_pybindの文字列パース・python_packageラッパーの配線を検出するAPIレベルのテストが無かったため追加）を追加。
- クラスターロバストSEの境界ケース（G=2ちょうど）は、OLSの`wald_f_test`と異なりLogitの`cluster_cov_params`がq×q部分行列の反転を要求しないため、説明変数を1個に絞る必要がないことを実機確認した（`k=3`のままG=2で正常に計算できる）。
- Rクロスチェックはz値・p値・信頼区間（`coeftest`の全4列＋標準正規分布ベースの手計算信頼区間）も比較対象に追加した（当初は係数・標準誤差のみで、`testing-policy.md`「公開する統計量は全て独立実装でもcrosscheckする」の要求を満たしていなかった。python-reviewer・testing-completeness-reviewerの指摘を受けて追加）。
- `test_logit_crosscheck.py`は当初`lr_p_value`をフィクスチャに含めながらテストで未検証だった（Rスクリプトは`lr_p_value`を出力していたが、テスト側でassertが漏れていた）。レビュー指摘を受けて追加。

## mkdocsドキュメント作成（Issue #69で実装済み）

- `docs/api/logit.md`を新規作成（`docs/api/ols.md`/`wls.md`と同じ構成: mkdocstringsディレクティブによる`Logit`/`LogitOptions`/`LogitResults`の自動APIリファレンスに加え、Logit固有の補足セクション（標準誤差の種類、ソルバーオプション、限界効果）を手書きで追加）。`docs/mkdocs.yml`のnavに`API Reference > Logit`を追加。
- `docs/getting-started.md`にLogitセクションを追加（WLSセクションの後、エラーハンドリングの前）。基本的な`fit()`、`predict()`/`pred_table()`、`marginal_effects()`の使用例を含む。`docs/index.md`の「Supported methods」もLogit追加を反映。
- Python側の全public docstring（`python_package/econometricsmodels/nonlinear/logit.py`）は既にGoogleスタイル・英語で書かれており（Issue #65〜#67で整備済み）、本Issueでの追加修正は不要だった。
- `mkdocs build`（`--config-file docs/mkdocs.yml`、CI（`cd_docs.yml`）と同じ非strictモード）で正常にビルドできることを確認済み。`--strict`は既存の未解消警告（Issue #111、`docs/planning/specs/`配下の相対リンク）のため使わない方針を踏襲。
