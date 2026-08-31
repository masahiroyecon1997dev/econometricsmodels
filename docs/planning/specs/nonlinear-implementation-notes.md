# 非線形モデル（MLEベース） 内部実装ノート（パラメータ設計以外）

`docs/planning/specs/`配下。非線形モデルのAPI・オプション設計（`nonlinear-api-design.md`）とは別に、**パラメータ以外の内部実装で決めたこと・まだ決まっていないこと**をまとめる。OLSの`docs/spec/ols-spec.md`と同じ位置づけ。実装issue（engine関連）着手時に必ず参照すること。

**現状**: Logit/Probit/Tobitともに実装未着手。本ノートはOLSのように実装issueごとの記録を積み上げていく形ではなく、設計段階で先に決まった内部実装レベルの事項を記録する。実装着手後は決定事項を同様に追記していく。

## 確定事項

### エラー型: nonlinear系統で共有（`MleError`）

- `engine/src/nonlinear/common.rs`に**Logit/Probit/Tobit共有のエラー型**（仮称`MleError`）を1つ定義し、3手法の`fit()`がこれを使う。OLSの「1手法1エラー型（`LeastSquaresError`、当時は`OlsError`という名前だったが後に改名）」パターンをそのまま横展開せず、共有型にする
- 理由: `raise_on_non_convergence`未収束・観測数不足・`confidence_level`範囲外等、3手法でほぼ共通のバリアントが多く、手法ごとに別エラー型にすると同じ意味のバリアントが3箇所に重複する。`.claude/rules/rust-style.md`「系統内で共有するロジックは`<系統>/common.rs`に置く」という既存ルールに合致する
- Tobit固有のバリアント（打ち切り境界の検証等）は、別のenumに分離せず**同じ`MleError`にバリアントとして追加する**（Logit/Probitの`fit()`はそのバリアントを構築しないだけで、型を分ける必要はない。OLSの`CovType::Hac`/`CovType::Cluster`がフィールド付きバリアントとして共存しているのと同じ考え方）

現時点で想定されるバリアント（`nonlinear-api-design.md`で確定済みの仕様から逆算。実装issue着手時に確定させる）:

| バリアント | 対応するPython例外 | 由来 |
|---|---|---|
| `NonConvergence { n_iter: usize }` | `ComputationError` | 3章: `raise_on_non_convergence=true`かつ`max_iter`到達時 |
| `InsufficientObservations { n: usize, k: usize }` | `ValidationError` | OLSの`InsufficientObservations`と同型 |
| `InvalidConfidenceLevel { confidence_level: f64 }` | `ValidationError` | OLSと同型 |
| `InvalidMaxIter { max_iter: i64 }` | `ValidationError` | `max_iter <= 0`等 |
| `MissingClusterColumn` | `ValidationError` | `cov_type="cluster"`なのにクラスターキー未指定。OLSと同型 |
| `InsufficientClusters { g: usize }` | `ValidationError` | OLSと同型 |
| `SingularHessian` | `ComputationError` | 収束点のHessianが特異で、観測情報行列（`cov_type="classical"`既定）の逆行列が計算できない |
| `ComputationFailed(String)` | `ComputationError` | 正規分布のCDF計算失敗等、OLSの`ComputationFailed`と同型 |
| `InvalidCensoringBounds { lower: Option<f64>, upper: Option<f64> }`（Tobit専用） | `ValidationError` | 下限≧上限等の不正な指定 |

### 標準誤差の技術仕様（`cov_type`の計算式）

`nonlinear-api-design.md`4章で確定した5種類（`classical`/`nonrobust`/`opg`/`hc0`/`hc1`/`cluster`）の具体式。`s_i`を観測`i`のスコアベクトル（対数尤度の1階微分）、`H`を収束点で評価した対数尤度のHessianとする。

- **観測情報行列**（`"classical"` / `"nonrobust"`、既定）: `Σ = -H⁻¹`
- **OPG/BHHH**（`"opg"`）: `Σ = (Σᵢ sᵢsᵢ')⁻¹`
- **サンドイッチ**（`"hc0"`）: `Σ = H⁻¹ (Σᵢ sᵢsᵢ') H⁻¹`（misspecification-robust、quasi-MLEサンドイッチ）
- **サンドイッチ+小標本補正**（`"hc1"`）: `hc0`の`Σ`に`n/(n-k)`を乗じる（OLSのHC1と同じ発想）
- **クラスター**（`"cluster"`）: `Σ = H⁻¹ (Σ_g S_g S_g') H⁻¹`、`S_g = Σ_{i∈g} sᵢ`（OLSの`cluster_cov_params`と同型の導出）

`H`・`s_i`はモデルごとに異なるが、上記5つの行列演算自体（`-H⁻¹`、外積和、サンドイッチ積）はLogit/Probit/Tobitで完全に共通のため、`nonlinear/common.rs`に共通関数として実装する（各モデル側は`s_i`・`H`を渡すだけでよい設計。詳細は後述「engine内のtrait設計」）。

**クラスターの小標本補正**: OLSと同じ規約（`Σ_cluster = correction * H⁻¹ (Σ_g S_g S_g') H⁻¹`、`correction = G/(G-1) * (n-1)/(n-k)`を常に適用し、無効化オプションを設けない）をそのまま踏襲する。根拠は`docs/spec/ols-spec.md`が確認済みの通り、statsmodelsの`sandwich_covariance.cov_cluster`がOLS専用ではなく線形モデル・MLEモデル共通の汎用関数であること。実装issue着手時にstatsmodelsソースで再確認する。

**OLSとの相違点（自由度切り替え不要）**: OLSは`cov_type=Cluster`のとき検定の自由度を`n-k`から`G-1`に切り替える処理があったが、非線形モデルはz検定（標準正規分布、自由度という概念がない）のため、この切り替え自体が不要。分散共分散行列のスケーリング（`correction`）だけ気にすればよい。

### 検定分布: 標準正規分布

- `nonlinear-api-design.md`5章で確定した「z検定」の実装詳細。MLEの漸近理論`θ̂ ~ N(θ, Σ)`に基づき、`z = θ̂ⱼ / se(θ̂ⱼ)`、p値は標準正規分布の両側確率、信頼区間は標準正規分布の臨界値（例: 95%信頼区間なら`z_crit ≈ 1.959964`）を使う
- OLSの`StudentsT`（statrs）に対応する形で、**`statrs::distribution::Normal`**を使う

### 収束判定の実装フロー

1. ソルバー（`method`＝`newton`/`bfgs`/`lbfgs`）を`max_iter`を上限に実行し、収束フラグ・反復回数を得る
2. `raise_on_non_convergence=true`（既定）かつ未収束 → `MleError::NonConvergence { n_iter }`を返す（`fit()`はここで打ち切り、パラメータ等は返さない）
3. それ以外（収束した、または`raise_on_non_convergence=false`） → 結果を返す。`converged`/`n_iter`フィールドに実際の値を格納する

### engineのモジュール構成

`.claude/rules/rust-style.md`の既存規約（系統＝ディレクトリ、手法＝最初は1ファイル）通り、`engine/src/nonlinear/{common.rs, logit.rs, probit.rs, tobit.rs}`とする。`common.rs`には`MleError`（上記）と、`cov_type`の共通行列演算（観測情報行列/OPG/サンドイッチ/クラスター）を置く候補。`engine_pybind`側も同じ系統名で対応させる（`engine_pybind/src/nonlinear/{logit,probit,tobit}.rs`）。

### engine内のtrait設計

argminの`CostFunction`/`Gradient`/`Hessian`トレイト実装と、`nonlinear`系統内の共通化範囲を以下のように分ける。

| 置き場所 | 内容 |
|---|---|
| 各モデルファイル（`logit.rs`/`probit.rs`/`tobit.rs`） | `{Logit,Probit,Tobit}Problem`構造体（`X`/`y`/Tobit境界値を保持）に対する`CostFunction`（負の対数尤度）/`Gradient`（スコアの符号反転）/`Hessian`（**`CostFunction`/`Gradientと同じ符号＝負の対数尤度のHessian`**。「収束点のHessian評価」節参照。対数尤度そのもののHessianではない点に注意）トレイト実装。加えてargminのトレイトではない独自メソッド`scores(&self, params) -> Mat<f64>`（n×k、観測ごとのスコア行列。OPG/サンドイッチ/クラスターSEの計算に必須。argminの`Gradient`は合計済みの1本のベクトルしか返さないため別途必要） |
| `nonlinear/common.rs` | (a) `Method`（`Newton`/`Bfgs`/`Lbfgs`）に応じたargminソルバーへのディスパッチ（`run_solver`関数、収束フラグ・反復回数を返す）。(b) `cov_type`ごとの共通行列演算（`H`と`scores`さえ受け取れば手法に依らず同じ計算） |

**Hessianは`method`の選択に関わらず常に解析的に実装する**: `bfgs`/`lbfgs`は最適化中にHessianを使わない（内部で近似する）が、`cov_type="classical"`（観測情報行列）には収束点でのHessianが必要。対象3手法はいずれも解析的Hessianが書けるため、`Hessian`トレイトは常に実装し、収束点で1回評価してSE計算に使う（BFGSの内部近似Hessianは使い回さない。手法間の結果の一貫性を優先する）。

### ソルバー実行の共通化（実装済み）

`engine/src/nonlinear/common.rs`に`Method`（enum、文字列パースはOLSの`CovType`と同じくengine_pybind側の責務）・`SolverOutput`・`run_solver()`を実装した。

**重要な発見（`argmin-math`はfaerバックエンドを使わない）**: 当初`nonlinear-api-design.md`2章は「argmin-math経由でfaerバックエンドに公式対応」としていたが、実装時に確認したところ`argmin-math`（0.5.1、crates.io最新かつargmin-rsのGitHub mainブランチも同じ）のfaerバックエンドは**faer 0.23までしか対応しておらず**、本プロジェクトの`faer = "=0.24.4"`ピンとは噛み合わないことが判明した。ユーザーに確認の上、以下の方針で解決した。

- argminには`argmin-math`の`vec`機能（追加の線形代数クレート不要、`Vec<f64>`/`Vec<Vec<f64>>`への実装のみ）を使わせる。`Param = Vec<f64>`、`Gradient = Vec<f64>`、`Hessian = Vec<Vec<f64>>`に固定
- モデル固有の尤度・勾配・Hessian計算、および`cov_type`の行列演算は引き続きfaer 0.24.4で行う。argminに値を渡す境界（`run_solver`関数の内部）でのみ`Vec<Vec<f64>>`⇔`faer::Mat`の変換を行う（k×k、パラメータ数は小さいのでコストは無視できる）
- Cargo.tomlの依存: `argmin = "=0.11.0"`、`argmin-math = { version = "=0.5.1", default-features = false, features = ["vec"] }`（`workspace.dependencies`で固定、他の依存と同じ方針）

**重要な発見（Newton法はargmin組み込みソルバーを使えない）**: argmin組み込みの`Newton`ソルバーは`H: ArgminInv<H>`（Hessianの逆行列）を要求するが、`argmin-math`の`vec`機能には`ArgminInv`の実装が存在しない（faer/nalgebra/ndarrayの行列型にしか実装されていない）。BFGS/L-BFGSは逆Hessianを直接更新していく方式のため`ArgminInv`不要で問題なく使えるが、Newtonだけこの制約に引っかかる。

- 対処: Newton法は独自の`Solver`実装（`FaerNewton`構造体、`argmin::core::Solver`トレイトを直接実装）とした。argminの`Solver`トレイトは`next_iter`/`terminate`が拡張ポイントとして用意されており、これ自体はargminの正規の使い方。Newtonステップの求解（`H·Δθ = g`）はfaerの列ピボットQR（`col_piv_qr`）で行う。OLSの`ensure_full_rank`と同じ相対閾値での特異性検出を行い、特異なら`MleError::SingularHessian`を返す
- argmin組み込みの`Newton`ソルバーは収束判定を一切行わず（`max_iters`に達するまで無条件に反復する）、`terminate()`をオーバーライドしていない。`FaerNewton`は`terminate()`を実装し、`next_iter`で計算した勾配を`state.gradient(...)`で状態に保存した上で、その勾配のノルムが`tol`未満なら`TerminationReason::SolverConverged`で早期終了する
- BFGS/L-BFGSは組み込みソルバーの`.with_tolerance_grad(tol)`（勾配のL2ノルムがこの値未満で収束と判定、`ArgminL2Norm`トレイト）をそのまま使う。線形探索は`MoreThuenteLineSearch`

**収束点のHessian評価**: `Method`の3分岐で`Executor::run()`実行後、`OptimizationResult.problem.take_problem()`でモデル（`O`）を取り出し、最終パラメータで`.hessian()`を1回呼び直す（Newtonの最後のイテレーションで計算済みのHessianを使い回すのではなく、常に独立して再評価する。3手法で同じコードパスにできて実装がシンプルになるため）。

**`Hessian`トレイトの符号規約（発覚・修正済み）**: `Hessian`トレイトの符号規約は当初明文化されていなかったが、Logitの尤度・スコア・Hessian実装に着手する際、2つの用途が矛盾する符号を要求していることが判明した。

- **Newton法の内部**（`FaerNewton::next_iter`）: `Δθ = H⁻¹g`（`g`は`Gradient`トレイトが返す「スコアの符号反転」、すなわち`CostFunction`＝負の対数尤度の勾配）。Newtonステップが正しい方向に進むには、`H`も`CostFunction`/`Gradient`と同じ符号（負の対数尤度のHessian）でなければならない。
- **収束点でのSE計算**（`SolverOutput.hessian`の`observed_information_cov_params`等）: `Σ_classical = -H⁻¹`という式は、`H`が対数尤度そのもののHessian（真の最大点で負定値、`neg_hessian_inverse`のdocコメント「真のMLE最大点では`-H`が正定値になるはず」の前提）であることを要求する。

両者は符号が逆（負の対数尤度のHessian＝対数尤度のHessianの符号反転）だが、当初の実装は`SolverOutput.hessian`を`model.hessian(&params)`の戻り値そのまま（符号変換なし）で構築しており、どちらか一方の用途で符号を取り違える状態だった。

**対処（ユーザー確認済み）**: `Hessian`トレイトの契約を「`CostFunction`/`Gradient`と同じ符号（負の対数尤度のHessian）」に統一する（オプティマイザライブラリとして自然な規約で、Newtonの正しさもこれで保証される）。`run_solver`内で`model.hessian(&params)`を呼んだ直後に1回だけ符号反転し、`SolverOutput.hessian`は対数尤度そのもののHessianとして返す（cov_type共通行列演算が前提とする符号と一致させる）。各モデルの`fit()`実装は符号変換を意識しなくてよい。

- 代替案（不採用）: 各モデルの`fit()`側でcov_type関数群に渡す直前に符号反転する。`run_solver`は変更不要だが、Probit/Tobit含め今後実装する全モデルで反転を忘れるリスクがあるため不採用とした。
- 影響範囲: `run_solver`関数本体（1箇所）、`SolverOutput.hessian`のdocコメント、`QuadraticProblem`を使った既存テスト（`run_solver_newton_converges_to_known_minimum`が検証する`output.hessian`の期待値を`2.0`/`5.0`から`-2.0`/`-5.0`に修正。`QuadraticProblem::hessian`自体はコスト関数のHessian`diag_a`をそのまま返す実装のまま変更していない）。

**エラー変換**: `FaerNewton::next_iter`内で`newton_step`が返す`MleError`は、`?`演算子で`argmin::core::Error`（`anyhow::Error`のエイリアス）に自動変換される（thiserrorが`std::error::Error + Send + Sync + 'static`を実装するため、anyhowの`From`実装が効く）。`Executor::run()`が`Err`を返した場合は`e.downcast::<MleError>()`で元の型を復元し、復元できない場合（argmin自体の内部エラー等）は`CommonError::ComputationFailed`（`MleError::Common`経由、`MleError::ComputationFailed`から移動）にまとめる（`convert_optimizer_error`関数）。

**バグ修正（rust-reviewerのレビューで発見・修正済み）**: `FaerNewton::next_iter`の初版実装は、`terminate()`の収束判定に「更新前のパラメータ」での勾配を使っていた（`next_iter`が`problem.gradient(&param)`（更新前）を計算し、`new_param`と一緒に`state`へ格納していたため）。argminの`Executor`は「`next_iter`実行後の`state`」に対して次のループの先頭で`terminate()`を呼ぶため、`terminate()`が見る勾配は返却される`params`（更新後の点）のものではなく1つ前の点のものになり、局所的に曲率（Hessian）が勾配より極端に小さい病的な点でNewtonステップが大きくオーバーシュートすると、「収束した」と誤って報告しつつ実際には勾配が全く小さくない推定値を返してしまうバグがあった。`FaerNewton::init`を新設して初期パラメータでの勾配も`state`に格納し、`next_iter`は`new_param`算出後に`problem.gradient(&new_param)`を改めて評価してから`state`に格納するよう修正した（`terminate()`が常に「返却するparamと対応する勾配」を見られるようにする）。回帰テスト（`faer_newton_terminate_reflects_gradient_at_returned_params_not_previous_params`）を追加済み。

**バグ修正（上記の回帰テスト作成中に発覚）**: `newton_step`の特異性検出（列ピボットQRのR対角成分を相対閾値と比較）は、`diag.abs() <= threshold`という比較を使っていたが、Hessianが全ゼロ行列のとき`col_piv_qr`が列選択時の0除算によりR対角成分に**NaN**を生成することがfaer 0.24.4で実機確認された。NaNとの比較は常に`false`になるため、この比較はNaNをすり抜けてしまい、特異なHessianが検出されないまま`solve_lstsq`に渡っていた。`diag.is_nan() || diag <= threshold`という形に修正した。**OLSの`ensure_full_rank`（`engine/src/linear/ols.rs`）も同じ比較パターンを使っており、理論上同じ弱点を抱えている**。ただしrust-reviewerの再検証により、実際にNaNが出るのは設計行列`X`全体が完全にゼロ（`include_intercept=false`かつ全説明変数列がゼロ）の場合のみで、単一の全ゼロ列（他に非ゼロ列がある通常の多重共線性ケース）は既存の閾値比較で正しく検出できることを確認済み（該当列のR対角成分が`NaN`ではなく厳密に`0.0`になるため）。リスクは実在するが限定的。今回のスコープ外のため未修正だったが、別途対応し、`ensure_full_rank`も同じ`diag.is_nan() || diag <= threshold`の形に修正済み（`docs/spec/ols-spec.md`「内部実装の計算仕様」参照）。

### 収束判定の`tol`（実装済み、妥当性はLogit・Probitで検証済み）

- **判定基準**: 勾配ノルム（L2ノルム、`‖∇ℓ(θ)‖ < tol`）。`newton`/`bfgs`/`lbfgs`の3手法すべてで同じ基準を使う（Newtonは独自実装、BFGS/L-BFGSは組み込みの`with_tolerance_grad`）。
- **デフォルト値**: `tol = 1e-6`で確定・維持（Logit実装・statsmodels/R glmとの数値照合で妥当性を検証済み。結論・根拠は`docs/spec/logit-spec.md`「最適化・収束判定」参照。要点: 通常データでは高精度に一致するが、準完全分離の境界ケースでは`1e-8`程度が必要。ただし`1e-8`に締めると`bfgs`が`max_iter`を使い切りやすくなるリスクがあるため、既定値は`1e-6`を維持し、境界ケースの数値比較テストのみ`tol`を明示的に締める運用とした）。**Probit実装時に同じ結論であることを実測確認済み**（通常シナリオはRTOL=1e-8で一致、near_separation境界ケースのみtol=1e-6だと相対誤差~4.4e-8とわずかに超過しtol=1e-8で解消、既定値は据え置き。詳細は`docs/spec/probit-spec.md`参照）。Tobit実装時も同じ結論を踏襲する想定(モデル固有の事情があれば個別に再検証する)。
- **既知の限界とその対処**: 完全分離に近いデータでは、係数発散の過程でスコア項が浮動小数点アンダーフローし、`tol`の値によらず「収束済み」と誤判定しうる。`tol`の調整では解決しない構造的な限界のため、`run_solver`の後処理として標準化パラメータ空間でのノルムを事後チェックし、異常に大きい場合は収束判定を取り消して`MleError::SeparationSuspected`を返す対処を追加した（勾配ノルム基準自体は`tol`のまま維持し、別の判定軸を追加する形。詳細は`docs/spec/logit-spec.md`「最適化・収束判定」参照）。
- **Options化**: `max_iter`と同じ扱いで`tol: f64 = 1e-6`をOptionsに追加する（`run_solver`関数の引数として実装済み。engine_pybind側の配線はLogitで実装済み）。

**スケール依存への対処（実装内容が確定）**: 勾配の絶対閾値は説明変数のスケールに依存する（`スコア = Σ 残差項 × x_i`のため、xが大きいスケールの列を含むと勾配も大きくなり、真に収束していても`tol`を割らない事態が起きうる）。

- `nonlinear/common.rs`に`standardize_columns(x: &Mat<f64>, has_intercept: bool) -> (Mat<f64>, ColumnScale)`・`destandardize_params(params_std: &[f64], scale: &ColumnScale) -> Vec<f64>`を実装した
- **設計変更（重要）**: 当初合意していた「平均0・分散1」への標準化（`x_std = (x-mean)/std`）は、実装時に**切片なし（`include_intercept=false`）のとき数式が壊れる**ことが判明した。平均を引く変換は、切片が「平均分のズレ」を吸収する前提（逆変換時に`θ_orig_intercept = θ_std_intercept - Σ θ_orig_j * mean_j`という補正が必要）で、切片が無いとこの補正先が存在せず、逆変換後のパラメータが数学的に不正になる
- ユーザーに確認の上、**「分散1のみ（スケーリングのみ、平均は引かない）」に変更した**（`x_std = x/std`）。この場合`θ_orig_j = θ_std_j/std_j`で完結し、切片の有無に関係なく成立する。当初の懸念（勾配ノルムの絶対閾値がxのスケールに依存する）は分散のスケーリングだけで解消できるため、目的は変わらず達成できる
- 却下案: Newton減少量`λ² = g'H⁻¹g`（スケール不変な収束基準）は理論的にはより厳密だが、`bfgs`/`lbfgs`では準ニュートン法が内部で保持する近似逆Hessianへのargmin API経由でのアクセスが必要で、実装できるか不確実なため採用しない
- 標準偏差が0の列（定数列）はスケーリング対象から除外する（0除算回避、`stds`をそのまま`1.0`にする）
- テストは`nonlinear/common.rs`内の`#[cfg(test)] mod tests`に実装。ダミーの2次関数（`f(θ) = 0.5(θ-target)'A(θ-target)`、`A`は対角正定値）でnewton/bfgs/lbfgsそれぞれの収束・既知の最小値への到達・Hessianの正しさを検証、`raise_on_non_convergence`の両分岐、`standardize_columns`/`destandardize_params`の往復変換を検証

### `cov_type`共通行列演算（実装済み）

`engine/src/nonlinear/common.rs`に`observed_information_cov_params`/`opg_cov_params`/`sandwich_cov_params`（`SandwichVariant::Hc0`/`Hc1`）/`cluster_cov_params`を実装した。いずれも`H`（収束点のHessian）・`scores`（n×k）のみを受け取り、モデル固有の尤度計算に依存しない（`docs/planning/specs/nonlinear-api-design.md`4章・本ファイル「標準誤差の技術仕様」の数式通り）。

**符号反転を1回の計算に集約**: `neg_hessian_inverse(H) = (-H)⁻¹`をコレスキー分解（OLSの`xtx_inverse`と同じ発想、`-H`は真のMLE最大点で正定値になるはず）で1回だけ計算し、`observed_information_cov_params`（`Σ = -H⁻¹`）・`sandwich_cov_params`・`cluster_cov_params`で同じ戻り値をそのまま再利用する。`-H⁻¹ = (-H)⁻¹`（逆行列の符号反転恒等式）、かつ`H⁻¹ΨH⁻¹ = (-H)⁻¹Ψ(-H)⁻¹`（符号が2回打ち消しあう）が成り立つため、サンドイッチ型・クラスターロバストの計算でも追加の符号反転が不要（rust-reviewerによる代数検証済み）。

**OPG行列特異時のエラー型を分離**: `cov_type="opg"`は`Σsᵢsᵢ'`（OPG行列、Hessianではない）の逆行列を計算するため、これが特異な場合は`MleError::SingularOpgMatrix`という別バリアントを新設して区別した（`MleError::SingularHessian`をそのまま流用する案もあったが、エラーメッセージ「the Hessian is singular」がOPG行列の特異性には不正確になるため、ユーザーに確認の上バリアントを追加する方針を採用）。

**クラスターの`MissingClusterColumn`/`InsufficientClusters`検証はこの共通関数のスコープ外**: `cluster_cov_params`はグループ数`G>=2`であることを検証しない（呼び出し側の責務、OLSの`validate_cluster_groups`と同じ役割分担）。この検証は各モデルの`fit()`実装で行う。

**追記（Logit）**: 実際に`fit()`側の検証を実装する段階で、この検証ロジック（`groups.len()==n`の内部契約チェック＋distinct count`>=2`の検証）がOLSの`validate_cluster_groups`と文言まで完全に同一だったため、ユーザー確認の上で`engine::validation::validate_cluster_groups`（`engine/src/validation.rs`、新設）に共有化し、OLS側もこれを呼ぶよう変更した。モデル固有の計算に一切依存しない純粋な検証ロジックである点が、`ensure_well_conditioned_symmetric_matrix`を共有化した判断根拠と同じ。当初は`CommonError`と同じ`error.rs`に置いたが、rust-reviewerの指摘（`error.rs`はエラー型定義専用のスコープ）を受けて独立モジュール`engine/src/validation.rs`に移設した。Logit側は反復最適化のコストを避けるため`fit()`冒頭（最適化の実行前）でこの検証を行う設計にした（OLSは閉形式解のため検証タイミングによるコスト差がなく、事後検証のまま）。

**テスト**: 対角Hessian・列間で観測ごとに片方が常にゼロになるスコア行列（対角のみのΨ）に加えて、列間に相関を持たせたスコア行列（非対角成分を持つΨ）でも検証し、転置・スケーリングの順序の取り違えを対角のみのテストより厳密に検出できるようにした（rust-reviewerの指摘を反映）。5種類（classical/opg/hc0/hc1/cluster）それぞれの正常系・特異行列時のエラー系を検証済み。

### `cov_type`共通行列演算の特異性検出（修正済み）

Logitのfit()に観測情報行列SEを実装するテスト追加中、`Method::Bfgs`で完全な多重共線性のあるデータセットを`fit()`すると、`MleError::SingularHessian`にならず桁違いに巨大な値（標準誤差が数千万オーダー等）を含む`Ok`が返ることが発覚した。

**原因**: `neg_hessian_inverse`・`opg_cov_params`（上記「`cov_type`共通行列演算」）は非ピボットCholesky分解（`Llt`）の失敗で特異性を検出していたが、これはOLSの`wald_f_test`で既に発覚していた既知の限界（非ピボットCholeskyは構造的な特異性・悪条件を確実には検出できない）の再発だった。`Method::Newton`は`newton_step`内の別の検出経路（ピボット付きQR、相対閾値）が最適化ステップ計算の途中で必ず通るため、たまたま検出できていた。`Method::Bfgs`/`Method::Lbfgs`は`newton_step`を一切経由しない（準ニュートン法は内部の近似逆Hessianで降下方向を決めるため、モデルの解析的Hessianの特異性に依存しない）ため、収束後の`observed_information_cov_params`/`opg_cov_params`呼び出しが唯一の検出経路になるが、そこが信頼できなかった。

**修正（OLSとの共通化）**: OLSの`wald_f_test`用に実装した`ensure_well_conditioned_cov_submatrix`（`SelfAdjointEigen`による固有値分解ベースの相対閾値判定）を、`context: &str`引数を追加した上で`engine/src/linear_algebra.rs`（新設。系統をまたいで共有する純粋な線形代数ユーティリティ、`.claude/rules/rust-style.md`「全手法で共有するロジック」）に`ensure_well_conditioned_symmetric_matrix`として切り出した。戻り値は`CommonError`にし、呼び出し側（`ols.rs`・`nonlinear/common.rs`）の`?`が`#[from]`経由で`LeastSquaresError`/`MleError`へ自動変換する（`CommonError`集約パターンをそのまま踏襲）。

`neg_hessian_inverse`・`opg_cov_params`はそれぞれCholesky分解の**前**にこの関数を呼び、エラー時は既存の`MleError::SingularHessian`/`MleError::SingularOpgMatrix`にマップする（`ensure_well_conditioned_symmetric_matrix`自体は特異性の種類を区別しない汎用関数のため、どちらのバリアントにするかは呼び出し側の`map_err`で決める）。`sandwich_cov_params`・`cluster_cov_params`は内部で`neg_hessian_inverse`を呼ぶため、修正を意識せず自動的に恩恵を受ける。

**テスト**: `fit_returns_singular_hessian_error_for_perfectly_collinear_design_matrix_with_bfgs_and_lbfgs`（Logit、`bfgs`/`lbfgs`両方での特異性検出を確認。`newton_step`を経由しないという主張通り両ソルバーで同じコードパスを通ることを個別に確認するため）・`opg_cov_params_returns_singular_opg_matrix_error_for_extreme_scale_difference`（`opg_cov_params`が構造的なゼロ行列だけでなく極端なスケール差による悪条件も検出できることを確認）・`linear_algebra`モジュール自体の単体テスト（良条件行列・完全特異行列・極端なスケール差行列の3ケース）を追加。`ensure_well_conditioned_symmetric_matrix`には`v`が`k×k`であるという呼び出し元の内部契約を検証する`debug_assert_eq!`も追加した（rust-reviewer指摘）。OLS側は挙動・受け入れ条件を変えない移設のため、既存テストがそのままリグレッションガードになる。

### Logitのデータ構造（実装済み）

`engine/src/nonlinear/logit.rs`に`LogitInput::from_columns`を実装した。`OlsInput::from_columns`（`weights=None`パス）と1:1で対応する設計（次元検証・切片列自動追加・`param_names`構築のロジックが同一）。weights/offsetはPhase2で見送り済みのため`from_columns_weighted`に相当するものはない。

`MleError`に`DimensionMismatch { y_rows, x_rows }`（OLSの`LeastSquaresError::DimensionMismatch`と同型）を追加した。当初のバリアント一覧には含まれていなかった（OLSには元々あったが、当時の設計メモへの転記漏れ）。Logit以降Probit/Tobitでも同じ形で使う。

**`y`の値域検証（単位区間`[0,1]`）は次Issue（B2、尤度・スコア・Hessian実装）に持ち越し**: statsmodelsの`Logit`はコンストラクタ時点で`endog`が単位区間`[0,1]`に収まることを検証する（範囲外は`ValueError: endog must be in the unit interval.`）。当初は次元検証のみがスコープのため`LogitInput::from_columns`では実装していないが、B2で同等の検証を追加する（rust-reviewerの指摘: 当初「statsmodelsも検証していない」という誤った理由でスコープ外としていたdocコメントを訂正済み）。

### 系統をまたぐ重複バリデーションエラーの共通化（実装済み）

上記「Logitのデータ構造」節で`MleError`に追加した`DimensionMismatch`をはじめ、`InsufficientObservations`/`InvalidConfidenceLevel`/`MissingClusterColumn`/`InsufficientClusters`/`ComputationFailed`の6バリアントが、linear系統の`LeastSquaresError`（`OlsError`から改名）と文言まで完全に重複していることが判明した。IV/panel/causal/io/time_series等、今後7系統・20〜30手法に増える前提のため、`engine/src/error.rs`に`CommonError`として切り出し、`MleError`・`LeastSquaresError`はいずれもthiserrorの`#[error(transparent)] Common(#[from] CommonError)`バリアントで包む設計にした。

- **`MleError`の該当6バリアントは削除し、`Common(CommonError)`に置き換えた**。`?`演算子は`#[from]`により`CommonError`→`MleError`へ自動変換されるため、`.map_err(|e| CommonError::ComputationFailed(e.to_string()))?`のように呼び出し側の変更は最小限で済む。直接`Err(...)`を返す箇所（`?`を経由しない`return Err(...)`）は`.into()`を明示する。
- **各系統固有のバリアント（`NonConvergence`/`InvalidMaxIter`/`SingularHessian`/`SingularOpgMatrix`/`InvalidCensoringBounds`）は`CommonError`に含めず、従来通り`MleError`に直接定義する**。「意味は同じだが系統固有の追加フィールドが要る」ケースが将来出てきた場合も、`CommonError`を拡張せずその系統独自のバリアントとして追加すればよい（構造上のブロッカーはない）。
- **`engine_pybind`側**: `engine_pybind/src/errors.rs`に`common_error_to_pyerr(CommonError) -> PyErr`を新設し、`least_squares_error_to_pyerr`（`linear`系統）はこれに委譲する形に変更済み。`MleError`用の変換関数（`mle_error_to_pyerr`相当）はLogitのpybind実装時（B13）にこのヘルパーを再利用する。
- **テストの重複排除**: 6バリアントのメッセージ・`PartialEq`のテストは`engine::error`側の`common_error_messages_are_human_readable`/`common_error_implements_partial_eq`に集約し、`MleError`・`LeastSquaresError`側のテストは各系統固有のバリアントと`Common`のtransparent転送の確認のみに絞った。

## 限界効果（Logit実装済み。Probit/Tobit実装時に同じ方針を踏襲する想定）

`marginal_effects(at="overall"|"mean"|"median", ...)`は`nonlinear-api-design.md`6章で確定済みの「`fit()`のReturn本体には含めない別メソッド」方針に従う。`at`の型（`MarginalEffectsAt`、文字列パースはengine_pybind側の責務）は`Method`/`CovType`と同じ理由で`nonlinear/common.rs`に定義し、Probit/Tobitでも再利用する。実際の限界効果・デルタ法ヤコビアンの計算式（リンク関数`Λ`/`Φ`等に依存する）は`CostFunction`/`Gradient`/`Hessian`と同様、モデルごとの実装（`logit.rs`等）に置く（`common.rs`には汎用の行列演算は置かない）。

- **離散変数（0/1のダミー変数）の自動判定は行わない**: statsmodelsの`get_margeff()`は既定（`dummy=False`）では0/1値の列も含め全ての説明変数を連続変数として扱い、解析的な偏微分`dy/dx=p(1-p)θⱼ`で計算する（`dummy=True`を明示指定したときのみ、値が0/1のみの列を検出して離散差分に切り替える特殊挙動）。本プロジェクトはデータ入力がlist渡し（列の型情報を持たない、CLAUDE.md 2章の非交渉事項）のため、離散変数の自動判定は実装せず、statsmodelsのデフォルト挙動（`dummy=False`相当）に合わせて常に連続変数として扱う方針とした（ユーザー確認済み、Logit実装時にPythonで`inspect.signature`と実際の数値照合により`dummy=False`が既定であることを確認済み）。
- **定数項（切片）は出力から除外する**: 切片の限界効果は経済学的に意味を持たないため（statsmodelsも同様に除外する）。`include_intercept`の値に関わらず`LogitInput::from_columns`の不変条件（`include_intercept=true`なら列0が常に定数項）を前提に、出力側で先頭`k_constant`列をスキップする設計。
- **`at="mean"`/`"median"`の代表点**: `fit()`に使った学習データ（`LogitInput::x()`）の各列の標本平均・標本中央値からなる固定ベクトル（statsmodelsの`atexog=None`のデフォルト挙動と同じ、外部から任意の評価点を指定するオプションは見送り）。
- **デルタ法のヤコビアンの統一形**: `at="overall"`（AME）・`"mean"`・`"median"`はいずれも`g_j(θ)=w(θ)*θⱼ`という同じ形に帰着する（`w`はAMEなら全観測平均の`p(1-p)`、mean/medianなら代表点で評価した`p̄(1-p̄)`）。この性質を使い、`w`とその勾配`s_m=∂w/∂θ_m`の計算（`at`ごとに異なる）と、そこから`dydx`・ヤコビアン`∂g_j/∂θ_m=θⱼ*s_m+[j==m]*w`を計算する部分（`at`に依らず共通）を分離する設計にした（Logitの`overall_w_and_s`/`at_point_w_and_s`と`dydx_and_jacobian`）。Probit/Tobit実装時、`w`/`s`の計算式（リンク関数の微分`p(1-p)`に相当する部分）はモデルごとに異なるが、`dydx_and_jacobian`と同型の共通化ができないか検討する。
- **分散**: `Var(g_j) = jac_jの行ベクトル · cov_params · jac_jの行ベクトル'`（二次形式）。標準誤差はこの平方根、検定分布は標準正規分布（`fit()`本体と同じ、`nonlinear-api-design.md`5章）。`fit()`時の`cov_params`をそのまま再利用し、再最適化は行わない。

## Tobit固有の設計判断（確定、実装ノート）

`nonlinear-api-design.md`5〜7章のユーザー確認済み方針を、内部実装レベルまで具体化した記録。実装issue着手時に参照すること。

### 打ち切り境界のバリデーション

- `TobitOptions.lower: Option<f64>`（既定`Some(0.0)`）/ `upper: Option<f64>`（既定`None`）。両方`None`、または両方`Some`で`lower >= upper`は既存の`MleError::InvalidCensoringBounds { lower, upper }`（`nonlinear-api-design.md`7章）
- **新規エラーバリアント**: `y`の実測値が指定した境界と矛盾する場合（`lower`指定時に`y < lower`の行がある、または`upper`指定時に`y > upper`の行がある）用に、`InvalidCensoringBounds`とは別のバリアントを新設する（暫定名`YOutOfCensoringBounds { row: usize, value: f64 }`。`InvalidBinaryY`と同型の「行番号+値」を持つエラーパターンを踏襲）。`InvalidCensoringBounds`は「境界設定自体が不正」、新バリアントは「境界設定は妥当だがデータと矛盾」という意味の違いを明確に分ける
- 検証は`fit()`冒頭、`LogitInput`/`ProbitInput`の`from_columns`に相当する`TobitInput::from_columns`内でO(n)の1回スキャンとして実装する（`extract_f64_column`の非有限値チェックと同オーダーのコストで、Newton反復本体のO(n·k)コストに対して無視できる）

### パラメータ化（内部最適化変数）

- `TobitProblem`の`params`は`(β, log σ)`という`k+1`次元ベクトルとして扱う。`σ`ではなく`log σ`を最適化変数にすることで正値制約を回避する（AER::tobitの`summary.tobit`が`Log(scale)`をそのまま報告しているのと同じ流儀）
- 収束後、`σ = exp(log σ)`へ逆変換し、そのSEはデルタ法（`Var(σ) ≈ σ² · Var(log σ)`）で計算する。**Issue #217の実装時に確定**: 対角成分（`Var(σ)`）だけでなく`cov_params`行列全体（`(k+1)×(k+1)`）にデルタ法のヤコビアン`diag(1,...,1,σ)`（`s→σ=exp(s)`、`dσ/ds=σ`、`β`部分は恒等写像）を両側から適用し、`β`-`σ`間の共分散も含めて`(β,σ)`空間へ変換する設計を採用した（ユーザー確認済み。将来の限界効果等での`cov_params`再利用を見据え、対角成分のみの変換に留めない）。`TobitEstimator::cov_params()`は`(k+1)×(k+1)`、`std_errors()`等は`β∪{σ}`の`k+1`長ベクトル（`k`番目の要素が`σ`）として公開する
- Olsen(1978)の`(δ=β/σ, γ=1/σ)`変換（大域凹性が数学的に保証される）は**採用しない**。当初は「ゼロベクトル初期値からのNewton収束はLogit/Probitで実績があり、`(β, log σ)`パラメータ化でもまず同様に運用し、収束性に問題が出た場合に再検討する」方針だったが、**Issue #215の実装時に実際に問題が発生**したため以下の2段階の対策を追加した（Olsen変換への回帰はせず、ユーザー確認済み）:
  1. **初期値をゼロベクトルではなくOLS推定値にする**: 打ち切りを無視した単純なOLS（`β`とその残差の標本標準偏差）を初期値にする（`tobit.rs`の`ols_initial_params`）。R `survreg`/`censReg`等の標準的なTobit実装と同じ方針。ただしこれだけでは、実際に打ち切りが発生するデータで依然としてNewtonが`SingularHessian`で失敗するケースが残った（打ち切りが皆無・無視できるデータでは1.のみで十分）
  2. **共有`FaerNewton`（`nonlinear/common.rs`、Logit/Probitと共有）にLevenberg-Marquardt型の減衰ステップを追加**: `(β, log σ)`パラメータ化のTobit尤度はHessianが不定符号になる領域を持ち、そこでは生のNewtonステップが降下方向ですらなくなることを実測で確認した。`H+λI`で`cost`が減少する候補が見つかるまで`λ`を段階的に増やす`regularized_newton_step`を追加。Logit/Probitのように尤度が大域凹な問題では`λ=0`の生のステップが常に最初の試行で受理されるため、既存の収束挙動と完全に一致する（既存テスト無変更で全通過を確認済み）

  詳細な発見の経緯・数式は`engine/src/nonlinear/tobit.rs`のモジュールdocコメント「Newton法の初期値」「Newtonステップの正則化」節、`regularized_newton_step`のdocコメント参照。

### `standardize_columns`とσの扱い

- 既存の`standardize_columns`/`destandardize_params`/`transform_cov_params_to_original_scale`（`nonlinear/common.rs`）は`params.len() == x.ncols() == ColumnScale.stds.len()`の1:1対応（`zip`ベース）が前提。Tobitは`params`がk+1次元（`β`のk個+`log σ`）になるため、そのままでは対応が崩れる
- `log σ`（`k+1`番目の要素）はXの列スケーリングと無関係な量（yのスケールに定義される）なので、**標準化対象に含めない**。実装は`ColumnScale.stds`にσ用の`1.0`（スケーリングなし）を末尾に追加する形にし、既存の`zip`ベースのロジックをそのまま再利用する

### `llnull`・GOF・有意性検定

- `log_likelihood_null`・`pseudo_r_squared`は実装しない（`nonlinear-api-design.md`5章で確定。理由: 閉形式が存在せず、主リファレンスのAER::tobitもpseudo R2を実装していないため）
- モデル全体の有意性検定は**Wald検定**を採用する（AER::tobitの`summary.tobit`と同じ方式:切片以外の係数が同時にゼロという帰無仮説を`cov_params`から直接計算。`linearHypothesis`相当のロジックを自前実装する）。尤度比検定（LR）は計量経済学の実務で好まれる場合があるため、v1では見送るが**将来拡張のTODOとして明記する**（`llnull`のためのintercept-only再最適化が必要になる。実装コストは`TobitInput`をk=1（切片のみ）で構築し既存のNewton/BFGS/L-BFGS基盤にそのまま渡せるため軽微）
- **Issue #220で実装済み**: `log_likelihood`（`Contribution::log_lik`の総和、`TobitProblem::from_standardized`は標準化済みかどうかを問わないため元のスケールの`x`・`params`をそのまま渡して評価）・`aic`/`bic`（総パラメータ数`k+1`を使う。`σ`も真に推定されたパラメータのため`k`ではなく`k+1`）・`n_obs`/`df_model`（`k-k_constant`）/`df_resid`（`n-(k+1)`、AER::tobit/survregの`df.residual`と同じく`σ`を含む総パラメータ数を差し引く）。Wald検定（`wald_chi2_test`）はOLSの`wald_f_test`と同型の構成（`ensure_well_conditioned_symmetric_matrix`による悪条件検出→Cholesky分解）だが、`F`分布ではなく標準正規分布に基づくカイ二乗分布を使う（自由度で正規化する`F=W/df_model`の変換を行わない）。Tobit専用（`llnull`を使わない検定方式のため`common.rs`には置かず`tobit.rs`内に閉じる）
- **Wald検定とクラスターロバストSEの構造的な相互作用（実装時に発覚）**: `fit()`はWald検定を`cov_type`の種類に関わらず常時実行する（`df_model==0`のときのみスキップ）ため、`cov_type=Cluster`のクラスターロバスト共分散`Ŝ=Σ_g S_gS_g'`が持つ構造的な制約（`rank(Ŝ)≤G`、`engine/src/linear/CLAUDE.md`「クラスター数`G`と傾き係数の数`q`の関係」参照、元々はOLSの`wald_f_test`で発見された制約）がWald検定の`q×q`部分行列にも及ぶ。Issue #219で書いたテスト（`multivariate_censored_input`、傾き係数`q=2`に対し`G=2`のクラスター境界値テスト）がこの制約に抵触し（`G=q`で部分行列が特異）、Issue #220の実装時に`ComputationFailed`で失敗するようになった。OLSの既存ガイドライン「境界の成功パスのテストでは`q`を`G`以下に保つ」に倣い、`q=1`のデータセット（`censored_regression_input`）に切り替えて`G=2>q=1`を保ちながら「G=2の境界値」の検証意図を維持し、多変量（`q=2`）のcov_type比較テストは`G=4`に増やして対応した（`tobit.rs`のcov_typeテスト群のdocコメント参照）。rust-reviewerレビューを受け、この`G=q`失敗パス自体を実際に踏む回帰テスト（`fit_returns_computation_failed_when_wald_submatrix_is_singular_for_cluster_with_g_equals_q`）と`df_model=1`の代数的恒等式（`W=z²`）テストを追加し、`engine/src/linear/CLAUDE.md`の記述も「`G<q`」の厳密不等号ではなく「`G=q`ちょうどでも安全とは限らない」と補強した
- **`wald_chi2_test`とOLSの`wald_f_test`の重複（既知、対応見送り）**: 両者は部分行列抽出→`ensure_well_conditioned_symmetric_matrix`→Cholesky分解→二次形式という構成が同型で、異なるのは検定分布（カイ二乗 vs F）のみ。`linear`系統と`nonlinear`系統をまたぐ共通化になるため`.claude/rules/rust-style.md`「全手法で共有するロジックはクレート直下」の対象になりうるが、現時点ではTobit1箇所のみの利用のため見送った（rust-reviewer指摘、should fix扱いだが本Issueのスコープ外と判断）。IV等で同型のWald検定が必要になり3箇所目の重複が生まれる場合は、二次形式計算のコア部分（分布変換を除く）の共通化を検討する

### `predict()` / `marginal_effects()` / `pred_table()`

- `predict()`は`E[y*|x]=x'β`・`E[y|x]`（打ち切り考慮の条件付き期待値）・`P(uncensored|x)=Φ(z)`の3種を返す。デフォルトは`E[y|x]`
- **Issue #222で実装済み**: `predict(target)`は`marginal_effects`の`target`引数と同じ`MarginalEffectsTarget`を再利用する（「予測値の対象」と「限界効果の対象」は同じ3種を指すため、専用enumを別途新設せず共有。DRY目的の実装判断であり、Issue #211「共通化しない」の対象——`w`/`s`の計算式そのもの——とは別の話）。値そのものの計算は新設した`predicted_value`（`target_w_and_s`と同じ`boundary_terms`を再利用し、左/右/両側打ち切りいずれでも単一の式で正しく計算できる）で行い、`marginal_effects`と同様に学習データの各行についてのみ返す（out-of-sample予測は未対応、Logit/Probitと同じ理由で別issueトラッキング）
- **打ち切り適合度チェックの出力形式（Issue #222着手時にユーザー確認）**: 単一集約値ではなく、`lower`/`uncensored`/`upper`の**方向別内訳**を採用した（`CensoringFitCheck`、`lower`/`upper`は該当方向の打ち切りが無ければ`None`）。各カテゴリは`observed_rate`（`y`がちょうど境界値に一致する観測の割合）と`model_implied_rate`（各観測の該当カテゴリ確率の平均、`Φ`の組み合わせは`predicted_value`の`ProbUncensored`と同型）を持つ。理由: 単一集約値だと両側打ち切りでどちらの境界に不整合があるか区別できないため。メソッド名は`censoring_fit_check`（`pred_table`の代替）
- **`censoring_fit_check`の打ち切り判定は`yᵢ==lower`/`yᵢ==upper`という浮動小数点の完全一致比較**（`TobitInput::from_columns`が`y`を変換せず保持する設計と整合し、`y`が実際に境界値ちょうどであることを前提とするTobitの定義自体とも整合する）。ただし、これは呼び出し側（`engine_pybind`、将来のPython層）が渡す`y`と`lower`/`upper`がビット単位で完全に一致することを前提とする（CSV/Parquet経由の読み込みや`Float32`→`Float64`変換等で丸め誤差が生じる経路がある場合は要注意、`censoring_fit_check`のdocコメント参照）。許容誤差付き比較は現時点では未導入（rust-reviewer指摘、少なくとも前提をdocコメントに明記する形で対応。実データで丸め誤差による誤分類が実際に問題になった場合に改めて検討する）
- `marginal_effects()`はLogit/Probitの`dydx_and_jacobian`共通化パターンを流用せず独立実装する（Issue #211「限界効果のw_and_s計算の共通化検討」の結論。対象ごとに式が異なり同型の`(w,s)`分解に無理に収める価値がないため、Tobit実装時に`overall_w_and_s`/`at_point_w_and_s`相当のTobit版を独自に書く）
- **Issue #221で実装済み**: `MarginalEffectsTarget`（`ExpectedLatent`=`E[y*|x]`・`ExpectedObserved`=`E[y|x]`（既定）・`ProbUncensored`=`P(uncensored|x)`）を新設し、`marginal_effects(at, target, confidence_level)`として提供。3対象とも実装時に`dydx_j=w(θ)*βⱼ`という同一の形（Logit/Probitと同じ骨格、`w`の中身のみ対象ごとに異なる）に帰着することが判明したため、`target_w_and_s`（`w`とその勾配`(s_beta,s_sigma)`を計算、`ExpectedLatent`は`w=1`・`s=0`の自明形）→`marginal_effects_from_tobit_w_s`（`common.rs`の`dydx_and_jacobian`/`marginal_effects_from_w_s`と同型だがパラメータ次元が`k`ではなく`k+1`=`β∪{σ}`である点が異なるため独立実装）という2段構成で実装した。**Issue #211の「共通化しない」という結論自体は変えていない**（`w`/`s`の計算式そのものはLogit/Probitと共有できない。あくまでTobit内部での実装パターンがLogit/Probitと相似形だったというだけ）
- **左/右/両側打ち切りいずれでも同じ式で正しく計算できることを確認済み**（ユーザー確認済み、v1スコープを両側打ち切りまで含めることを含めて確認）: `boundary_terms`が`lower`/`upper`の`None`を`Φ(∓∞)=0/1`・`φ(∓∞)=0`相当の定数として扱うことで、`W(θ)=Φ(zb)-Φ(za)`という単一の式が左のみ・右のみ・両側いずれの打ち切りでも正しい`P(uncensored|x)`を返す。`∂E[y|x]/∂xⱼ=P(uncensored|x)*βⱼ`という恒等式（McDonald-Moffitt 1980の一般形、境界項が積分の端点で相殺するため打ち切りパターンに依らず成立、手計算で導出・確認済み）も同様に統一的に扱える。テストでは`right_censored_regression_input`・`two_sided_censored_regression_input`という新規フィクスチャを追加し、`target_w_and_s`とは独立に書いた閉形式（`expected_observed_closed_form`/`prob_uncensored_closed_form`/`dprob_uncensored_dmu_closed_form`、`(lower,upper)`のパターンごとに別の`match`分岐で教科書通りの式を書く）と突き合わせて検証した。右打ち切りのみの場合`P(uncensored|x)`の限界効果は符号が反転する（`x`の増加が上側打ち切り確率を高めるため）ことも回帰テストで固定した
- **rust-reviewerレビュー結果**: must-fixなし。should-fix 1件（`MarginalEffects::from_parts`に7本の`Vec`の長さ整合性チェックが無かった。`TobitInput::from_columns`の`x_columns.len()==x_names.len()`と同じ方針で`debug_assert_eq!`を追加して対応）。nice-to-haveのうち`overall_target_w_and_s`が観測ごとに`x_row`をヒープ確保していた点（Logitの`overall_w_and_s`は`x.get(i,j)`を直接使いコピーなし）は、ループ外でバッファを1回確保し使い回す形に直して対応した。残りのnice-to-have（右打ち切り/両側打ち切りでのSE独立検証の追加、`cov_type`非Classicalとの組み合わせテスト、`at="mean"/"median"`×`ProbUncensored`等の組み合わせテスト、`target_w_and_s`内の`unreachable!`アームの構造的な冗長性）は、レビュアー自身が「大きなリスクは無い」「優先度は低い」と評価した項目であり、`marginal_effects`は`fit()`が計算済みの`cov_params`を打ち切りパターン・`cov_type`に依存しないコードパスで再利用するだけ（ヤコビアン組み立て自体は`marginal_effects_from_tobit_w_s`で一本化されており分岐しない）という設計上の理由から対応を見送った（`wald_chi2_test`/`wald_f_test`の重複を見送ったIssue #220と同じ判断基準）
- `MarginalEffects`構造体（`common.rs`）はフィールドがprivateのため、Logit/Probitの`marginal_effects_from_w_s`を経由しないTobitから構築できるよう`pub(crate) fn from_parts`コンストラクタを追加した（出力構造体の形自体は`coef_table`と同じ行指向でLogit/Probitと共通のまま、`nonlinear-api-design.md`6章）
- `pred_table()`は廃止し、打ち切り予測の適合度チェック（観測打ち切り比率 vs モデル含意の平均`Φ(z)`）に置き換える。出力の具体形式は実装issue着手時に決定する

### `cov_type`共通行列演算・バリデーション

- `observed_information_cov_params`/`opg_cov_params`/`sandwich_cov_params`/`cluster_cov_params`（`nonlinear/common.rs`）はモデル非依存のため、Tobitの`H`（負の対数尤度のHessian、`(k+1)×(k+1)`）・`scores`（n×(k+1)）を渡すだけでそのまま再利用できる
- 不均一分散下でのMLE非一致性の限界（ロバストSEはこれを解決しない）はLogit/Probitと同じ既存の前提を踏襲し、Tobit固有の新たな設計判断は不要
- Issue #212「fit()共通バリデーション関数のTobit対応拡張検討」の結論: `validate_fit_preconditions`の`validate_binary_y`呼び出しはTobitでは行わない（`y`は連続変数のため）。上記「打ち切り境界のバリデーション」の検証はTobit専用の追加ステップとして`validate_fit_preconditions`とは別に呼ぶ（具体的な関数分割は実装issue着手時に決定）
- **Issue #218（OPG/HC0/HC1）着手時に判明**: `CovType`はLogit/Probit/Tobit共有の1つのenumで、Logitの#60（クラスターロバストSE）実装時に既に`Cluster`バリアントを含んでいる。そのため`TobitEstimator::fit`に`cov_type: CovType`引数を追加すると`match cov_type`を網羅的にする必要があり、#218（本来OPG/HC0/HC1のみ）単独では実装できない。Probitの前例（コミット`c868912`、Issue #75着手時に同じ状況に遭遇し#76＝クラスターも同時実装）を踏襲し、ユーザー確認の上で#218・#219（クラスターロバストSE）をまとめて実装した
- **`cov_type=Opg`が特異になるテストケースの構築（Tobit固有の制約）**: Logit/Probitは`x`の完全な多重共線性（例: `x2=2*x1`）でOPG行列`Σsᵢsᵢ'`を特異にできる（`fit_returns_singular_opg_matrix_error_for_perfectly_collinear_design_matrix`）が、Tobitは`ols_initial_params`が最適化前に`x`のQRベースの階数検定を行うため、`x`が完全に多重共線だと最適化に入る前に`MleError::SingularDesignMatrix`になってしまい同じ手法が使えない。当初`x2=2*x1+摂動`（多重共線性から小さく崩す設計）で`SingularOpgMatrix`を再現するテストを書いたが、rust-reviewerのレビューで**多重共線性は無関係**と判明した（`x2`をxと無関係な値に変えても同じ現象が再現され、収束点でのOPG行列の最小固有値が機械精度オーダーまで落ちることを独立に確認）。実際の原因は観測数`n=6`が総パラメータ数`k+1=4`ぎりぎりで左打ち切り観測2件という小標本特有の収束点の数値配置に依存すると推測されるが、厳密な理論的必然性は未解明（`n=7`に増やすと再現しない）。テスト自体は実測ベースで正しく動作するため`fit_returns_singular_opg_matrix_error_for_small_sample_degenerate_case`という名前・docコメントに修正し、「多重共線性が原因」という誤った説明を残さないようにした

### テスト参照実装

- 主リファレンス: R `AER::tobit`（`survival::survreg`エンジン）。クロスチェック: R `censReg`（`maxLik`エンジン）
- `RTOL`は実測してから決定する（`survreg`は内部で`(β, log σ)`パラメータ化・独自のNewton-Raphson実装のため、Logit/Probitのstatsmodels比較ほど高精度で一致するとは限らない）
- Wooldridge `mroz`データセットの`hours`（左打ち切り、多くの0値。Wooldridge Example 17.2相当）をベンチマークデータセットに使う。`inlf`列を使うLogit/Probitと同じデータセットの別列を再利用できる

## 未確定（実装issue着手時、または追加相談が必要）

- **Tobitの尤度・勾配・Hessianの閉形式の書き下し**: 標準的な打ち切り正規回帰の尤度（打ち切り観測はΦ、非打ち切り観測はφ）で導出可能という方向性のみ確認済み。実際の数式・実装は着手時に行う
- **`YOutOfCensoringBounds`（暫定名）等、新規エラーバリアントの正式な命名・メッセージ文言**: 実装issue着手時に確定する

## engine単体テストのカバレッジ確認（Issue #223、完了）

- `cargo llvm-cov -p engine`で計測。`nonlinear/tobit.rs`はRegion 97.97%・Line 98.22%・Function 98.70%（Logit 97.78%/98.83%/99.26%・Probit 97.89%/98.86%/100.00%と同水準）。未カバー箇所は全て既知の許容パターン（`--show-missing-lines`で機能単位に確認済み）:
  - `wald_chi2_test`のCholesky分解失敗の防御的`map_err`分岐（`ensure_well_conditioned_symmetric_matrix`が事前に悪条件を検出するため理論上到達不能、OLSの`wald_f_test`と同じ扱い）
  - `target_w_and_s`/`predicted_value`の`MarginalEffectsTarget::ExpectedLatent => unreachable!(...)`（関数冒頭の早期returnで既に処理済みの分岐、Issue #221で導入）
  - テストファイル内の`assert!`/`panic!`引数の文字列補間式（アサーション失敗時のみ評価されるため、テストが全て通過する限り構造的にカバーされない。テストコード自体の欠落ではない）
  - テストヘルパー（`expected_observed_closed_form`等）内の`(None, None) => unreachable!(...)`（`TobitInput::from_columns`が両方`None`を弾くため、テスト側でも構造的に到達不能）
  - `n_iter()`ゲッターが未使用だったギャップは本Issueで発見・修正済み（`fit_newton_converges_for_data_with_actual_censoring`に`n_iter()`のアサーションを追加）
  - `sandwich_cov_params`/`cluster_cov_params`経由の`SingularHessian`伝播（`Hc0`/`Hc1`/`Cluster`）が一度もテストされていなかったギャップも本Issueで発見・修正済み（Logit/Probitの同種ギャップ＝Issue #64/#80と同じパターン。`fit_returns_singular_hessian_error_when_cov_params_computation_fails_at_truncated_point_with_hc0_and_hc1`/`..._with_cluster`を追加）
  - `nonlinear/common.rs`の残りの未カバー箇所（`run_solver`のargmin内部エラー変換等）はTobit導入前から存在する既存の防御的分岐で、Logit/Probitの過去のカバレッジissue（#64/#80）で既に確認済み。Tobit固有の新規ギャップではない

- **Tobit固有の病理ケース: 全件打ち切り（非識別）データの検出（実測で発見、ユーザー確認済み）**: `y`が全て`lower`（または`lower`/`upper`のいずれか）に一致する非打ち切り観測ゼロのデータでは、既存の`SeparationSuspected`検出条件（標準化パラメータ空間のノルムが閾値超）では捕捉できないことが実測で判明した——Tobitのこの退化は`β`が発散するのではなく`σ`が0に収束する形で起こるため、標準化パラメータノルムは閾値を超えないまま`fit()`が`converged=true`のまま統計的に無意味な巨大SE（実測で標準誤差が100万倍オーダー）を返してしまう。参照実装`survival::survreg`（`AER::tobit`のエンジン）は同種のデータで初期反復に失敗しエラーを返すことをdevcontainer内で確認した。
  - **対応（ユーザー確認済み、入力バリデーションを採用）**: `fit()`冒頭に`validate_has_uncensored_observations`を追加し、`y`が`lower`/`upper`いずれの境界にも一致しない観測（厳密に内部の観測）が1件も無い場合は`MleError::NoUncensoredObservations { lower, upper }`（`ValidationError`）を返す。事後検知（`SeparationSuspected`型のヒューリスティックを新設し閾値を経験的に較正する案）よりも、判定が決定的で閾値校正が不要な入力バリデーション方式を選んだ
  - 打ち切り判定は`censoring_fit_check`と同じ`yᵢ==lower`/`yᵢ==upper`の完全一致比較（一貫した規約）
  - **非全件打ち切りだが非常に高い打ち切り率**（実測: 8件中7件打ち切り）は本チェックの対象外（非打ち切り観測が1件でもあれば通過する）。この場合はNewtonが単に収束しない（`converged=false`）か、BFGSの`MoreThuenteLineSearch`が降下方向でないと判定し`ComputationFailed`を返すことを実測で確認済みで、いずれも既存のエラー経路で安全に失敗する（新規の検出機構は不要と判断）
  - **厳密な非識別条件の精緻化（rust-reviewer指摘、`MleError::NoUncensoredObservations`のdocコメント参照）**: 「非打ち切り観測0件→識別不能」という記述は正確には十分条件であり必要条件ではない。厳密な非識別条件は「打ち切りカテゴリ（`y==lower` vs `y==upper`）が`x`の線形結合で完全分離可能」であること（Logit/Probitの完全分離と同型）で、非打ち切り観測0件はこの分離を妨げる要因が無いため必ず分離可能になる（十分条件）が、逆に0件でなくても分離不能な`x`の配置であれば理論上は有限のMLEが存在しうる。連続変数`x`でこの非分離配置が実務データに現れることは考えにくいため、分離可能性を厳密に判定する複雑なロジック（線形計画法的な実行可能性判定に相当）は導入せず、「0件なら一律エラー」という保守的な単純化を採用している（ユーザー確認済み、OLSが完全な多重共線性のみ弾き条件数が大きいだけの悪条件行列はそのまま数値計算に委ねるのと同じ設計哲学）

- **rust-reviewerレビュー結果**: must-fixなし。should-fix 1件（`fit()`の`# Errors`docコメントに`NoUncensoredObservations`の記載漏れ、追加して対応）。計量経済学的妥当性の指摘（上記「厳密な非識別条件の精緻化」）に対応し、`MleError::NoUncensoredObservations`・`validate_has_uncensored_observations`双方のdocコメントを「十分条件であり必要条件ではない」ことが分かる表現に修正した。nice-to-have（片側打ち切り`upper`のみのケースの独立テスト追加、`validate_has_uncensored_observations`を`from_columns`ではなく`fit()`で検証する理由のdocコメント明記）にも対応。`nonlinear-implementation-notes.md`の「現時点で想定されるバリアント」テーブルの陳腐化、`tests/`配下のTobit用pytest未整備は本Issueのスコープ外（前者は過去のIssueから慢性的に陳腐化、後者は別Issue＝#226以降で対応予定）として見送った。

## engine_pybind実装（Issue #224+#225、まとめて実装・完了）

`build_tobit_input`（データ抽出・バリデーション）は`fit_tobit`（`TobitEstimator::fit`呼び出し・`TobitResult`構築・`lib.rs`登録）と1コミットで実装した。`build_tobit_input`/`parse_cov_type`/`parse_method`は`fit()`が無いと`#[cfg(test)]`以外から一度も呼ばれず、非testビルドの`cargo clippy -D warnings`がdead_code lintで失敗する（ユーザー確認済み、Issue #218+#219と同じ理由でのまとめ判断）。

- `TobitOptions`はLogit/Probitの8フィールドに`lower: Option<f64>`（既定`Some(0.0)`）・`upper: Option<f64>`（既定`None`）を追加（`nonlinear-api-design.md`7章）。打ち切り境界自体のバリデーション（両方`None`・`lower>=upper`等）はここでは行わず`TobitInput::from_columns`（engine層）に委ねる（Issue #212の結論通り）。
- **`TobitResult`のフィールド設計（ユーザー確認済み）**: `engine::nonlinear::tobit::TobitEstimator`は`params()`が`k`長（`β`のみ）である一方`std_errors()`等は`(k+1)`長（`σ`を含む）という非対称な設計（`cov_params`が`(β,σ)`空間の`(k+1)×(k+1)`行列であるため）。Python側ではこの非対称性を解消し、`params`/`param_names`/`std_errors`/`z_stats`/`p_values`/`conf_lower`/`conf_upper`を全て`(k+1)`長に統一した（`param_names`の末尾に`"sigma"`を追加、`params`に`sigma()`の値を追加）。これによりPython側で`zip(param_names, std_errors)`のような素朴な利用ができる。`sigma: f64`フィールド（`params[-1]`と同値）も利便のため追加。`log_likelihood_null`/`lr_statistic`/`lr_p_value`/`pseudo_r_squared`は提供しない（`nonlinear-api-design.md`5章の既存方針通り）。`cov_type`に加え`lower`/`upper`も`TobitOptions`からのエコーとして`TobitResult`に含めた（`cov_type`と同じ「実際に使われた設定値を結果から確認できるようにする」目的）。
- **`predict()`/`marginal_effects()`の`target`引数（ユーザー確認済み）**: Rust側`MarginalEffectsTarget`（`ExpectedLatent`/`ExpectedObserved`/`ProbUncensored`）に対応するPython文字列を`"expected_latent"`/`"expected_observed"`/`"prob_uncensored"`（enum名のsnake_case版）とした。`at`（`"overall"`/`"mean"`/`"median"`）のような単語1つの慣習が無い（3つとも複数統計概念の組み合わせ名）ため、Rust enum名との対応が一目瞭然な形を優先した。パース関数`parse_marginal_effects_target`はTobit専用（Logit/Probitに無い概念のため`nonlinear/common.rs`ではなく`tobit.rs`に定義）。
- `censoring_fit_check()`は`pred_table()`の代替として提供（`CensoringFitCategoryResult`/`CensoringFitCheckResult`という新規pyclassでラップ、`lower`/`upper`は該当方向の打ち切りが無ければ`None`）。
- `mle_error_to_pyerr`（`nonlinear/common.rs`）は`InvalidCensoringBounds`/`YOutOfCensoringBounds`/`SingularDesignMatrix`/`NoUncensoredObservations`を含め既に網羅済みだった（`MleError`への追加時にRustコンパイラの網羅性チェックにより既存issue（#213/#217/#223）で自動的に配線済みのため、本Issueでの追加変更は不要だった）。
- `parse_cov_type`/`parse_method`はLogit/Probitと同じロジックをファイルごとに複製する既存方針（`probit.rs`と同じ）を踏襲した。
- `maturin develop`後、Python側から`fit_tobit`・`TobitOptions`・`predict()`/`marginal_effects()`/`censoring_fit_check()`・主要なエラーパス（`cov_type`不正・打ち切り境界不正・`target`不正・`confidence_level`範囲外・`NoUncensoredObservations`）を実際に呼び出して動作確認済み（`engine_pybind/src/nonlinear/CLAUDE.md`の既知の制約通り、`fit`本体は`PyDataFrame`引数のため`#[cfg(test)]`から直接呼べない）。
- テスト: `build_tobit_input`系はLogitの対応テストを移植し、打ち切り境界（`lower`/`upper`カスタム値・右打ち切りのみ・両方`None`のエラー伝播）・`parse_marginal_effects_target`のテストを追加。

**rust-reviewerレビュー結果**: must-fix 2件、いずれも対応済み。
- **`"sigma"`列名衝突の未検証（バグ）**: `param_names`末尾に無条件で追加する合成パラメータ名`"sigma"`が、ユーザーの`x`に`"sigma"`という列があると重複し、`zip(param_names, params)`のような素朴な利用でエラーにならず静かに誤った対応になる（`"const"`列衝突と同型の問題）。`validate_no_const_collision`と同じパターンで`validate_no_sigma_collision`を追加し`build_tobit_input`で呼ぶ形で対応した。
- **公開`#[pyclass]`のdocコメントへの日本語混入**（`.claude/rules/rust-style.md`「言語方針」違反）: `TobitResult`本体・`sigma`フィールド・`CensoringFitCategoryResult`・`CensoringFitCheckResult`の各docコメントを英語に書き直した。

should-fix 2件も対応済み: `TobitOptions.include_intercept`/`cluster_col`のdocコメントにLogit/Probitにあった説明文を追加（Tobitは完全な多重共線性が`ols_initial_params`のQR検証で先に検出されるため「singular Hessian」ではなく「singular design matrix」に文言修正）。`TobitResult.lower`/`.upper`の取得元を`estimator.input()`経由から`options.lower`/`options.upper`直接参照に変更し、`cov_type`と同じ「エコー元は`options`」というパターンに揃えた。

nice-to-have 2件も対応済み: `build_tobit_input_cov_type_is_case_insensitive`に`"Classical"`（先頭大文字）のケースとIssue #231の説明コメントを追加。`censoring_fit_category_to_result`/`censoring_fit_check_to_result`（フィールドの詰め替えロジック）の単体テストが無かった点は、`build_tobit_input`が返す`TobitInput`に対して`TobitEstimator::fit`をGIL無しで直接呼び出すテストを新設して対応した。

修正後: engine_pybind全体で93件（Logit/Probit/OLS/WLS/IV等の既存分含む、Tobit分は21件）、clippy/fmt警告ゼロ。`maturin develop`での動作確認も再実施済み。

## python_package実装（Issue #226、完了）

`Tobit`/`TobitResults`（`python_package/econometricsmodels/nonlinear/tobit.py`）を`Logit`/`LogitResults`と同型の薄いラッパーとして実装した。`TobitOptions`は`_lib`からそのまま再輸出（独自クラス再定義しない、既存方針通り）。

- `params`/`std_errors`/`z_stats`/`p_values`/`conf_int`は`_lib.TobitResult`の`(k+1)`長設計（`"sigma"`を含む）をそのまま反映し、`dict(zip(param_names, ...))`で自然に`"sigma"`エントリを含む。`coef_table()`も`"sigma"`の行を含む（R`summary.tobit`の`Log(scale)`行に相当）。`sigma: float`プロパティも追加。
- `log_likelihood_null`/`lr_statistic`/`lr_p_value`/`pseudo_r_squared`は提供しない。`wald_statistic`/`wald_p_value`を提供。
- `pred_table()`の代わりに`censoring_fit_check()`（`category`/`observed_rate`/`model_implied_rate`をキーに持つ行指向`list[dict]`、既存の`pred_table()`の行指向`list[dict]`慣習を踏襲）。
- `predict()`/`marginal_effects()`に`target`引数を追加。`predict()`の返り値キーは対象非依存の`"predicted"`（3つの`target`で意味が変わるため、Logitの固定名`"probability"`のような単一の意味を持つ名前にできない）。
- `tests/conftest.py`に`censored_dataset`フィクスチャ（共有`dataset`の`y`を`0.0`で左打ち切り、打ち切り率21%）を追加。`tests/nonlinear/test_tobit.py`（70→71件、`test_logit.py`の構造・API・エラーパススモークテストを移植し、打ち切り境界固有のエラー（`InvalidCensoringBounds`/`YOutOfCensoringBounds`/`NoUncensoredObservations`）・`"sigma"`列衝突・`target`引数のテストを追加）。

**python-reviewerレビュー結果**: must-fixなし。should-fix 1件: `test_probit.py`にある`SeparationSuspected`（准完全分離）のAPI境界テストがTobit版に無く、`nonlinear-api-design.md`10章「Tobitの分離相当の病理ケース」が未確定のままだった点を指摘された。実測調査の結果、**2種類の異なる退化パターンが存在する**ことが判明した:
- 非打ち切り観測ゼロによる`σ→0`退化（Issue #223で発見・`MleError::NoUncensoredObservations`で対応済み。標準化パラメータノルムは大きくならないため既存の`SeparationSuspected`機構では捕捉できない）
- **極端な`β`による分離**（本Issueで新規発見）: Logitの`near_separation`DGPと同じ発想（`x1`の真の係数=100）を打ち切り正規回帰に適用したデータで、既存の`SeparationSuspected`機構（`run_solver`共有、標準化パラメータノルム基準）がLogit/Probitと**同じ閾値でそのまま**Tobitの分離も検出できることを、Python API境界のテストで実測確認した（`test_separation_suspected_raises_computation_error_for_near_separation_data`として追加）。Tobit専用の閾値較正は不要だった。`nonlinear-api-design.md`10章の該当項目を`[x]`に更新した。

修正後: pytest全体で956件（Tobit分71件）、ruffエラーゼロ。
