# 非線形モデル（MLEベース） 内部実装ノート（パラメータ設計以外）

`docs/planning/specs/`配下。非線形モデルのAPI・オプション設計（`nonlinear-api-design.md`）とは別に、**パラメータ以外の内部実装で決めたこと・まだ決まっていないこと**をまとめる。OLSの`ols-implementation-notes.md`と同じ位置づけ。実装issue（engine関連）着手時に必ず参照すること。

**現状**: Logit/Probit/Tobitともに実装未着手（Issue化前）。本ノートはOLSのように「Issue #Nで実装済み」を積み上げていく形ではなく、設計段階で先に決まった内部実装レベルの事項を記録する。実装着手後は同じ形式（`Issue #Nで実装済み`）で追記していく。

## 確定事項

### エラー型: nonlinear系統で共有（`MleError`）

- `engine/src/nonlinear/common.rs`に**Logit/Probit/Tobit共有のエラー型**（仮称`MleError`）を1つ定義し、3手法の`fit()`がこれを使う。OLSの「1手法1エラー型（`OlsError`）」パターンをそのまま横展開せず、共有型にする
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

**クラスターの小標本補正**: OLSと同じ規約（`Σ_cluster = correction * H⁻¹ (Σ_g S_g S_g') H⁻¹`、`correction = G/(G-1) * (n-1)/(n-k)`を常に適用し、無効化オプションを設けない）をそのまま踏襲する。根拠は`ols-implementation-notes.md`が確認済みの通り、statsmodelsの`sandwich_covariance.cov_cluster`がOLS専用ではなく線形モデル・MLEモデル共通の汎用関数であること。実装issue着手時にstatsmodelsソースで再確認する。

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
| 各モデルファイル（`logit.rs`/`probit.rs`/`tobit.rs`） | `{Logit,Probit,Tobit}Problem`構造体（`X`/`y`/Tobit境界値を保持）に対する`CostFunction`（負の対数尤度）/`Gradient`（スコアの符号反転）/`Hessian`トレイト実装。加えてargminのトレイトではない独自メソッド`scores(&self, params) -> Mat<f64>`（n×k、観測ごとのスコア行列。OPG/サンドイッチ/クラスターSEの計算に必須。argminの`Gradient`は合計済みの1本のベクトルしか返さないため別途必要） |
| `nonlinear/common.rs` | (a) `Method`（`Newton`/`Bfgs`/`Lbfgs`）に応じたargminソルバーへのディスパッチ（`run_solver`関数、収束フラグ・反復回数を返す）。(b) `cov_type`ごとの共通行列演算（`H`と`scores`さえ受け取れば手法に依らず同じ計算、Issue #53で実装） |

**Hessianは`method`の選択に関わらず常に解析的に実装する**: `bfgs`/`lbfgs`は最適化中にHessianを使わない（内部で近似する）が、`cov_type="classical"`（観測情報行列）には収束点でのHessianが必要。対象3手法はいずれも解析的Hessianが書けるため、`Hessian`トレイトは常に実装し、収束点で1回評価してSE計算に使う（BFGSの内部近似Hessianは使い回さない。手法間の結果の一貫性を優先する）。

### ソルバー実行の共通化（Issue #52で実装済み）

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

**エラー変換**: `FaerNewton::next_iter`内で`newton_step`が返す`MleError`は、`?`演算子で`argmin::core::Error`（`anyhow::Error`のエイリアス）に自動変換される（thiserrorが`std::error::Error + Send + Sync + 'static`を実装するため、anyhowの`From`実装が効く）。`Executor::run()`が`Err`を返した場合は`e.downcast::<MleError>()`で元の型を復元し、復元できない場合（argmin自体の内部エラー等）は`MleError::ComputationFailed`にまとめる（`convert_optimizer_error`関数）。

### 収束判定の`tol`（Issue #52で実装済み）

- **判定基準**: 勾配ノルム（L2ノルム、`‖∇ℓ(θ)‖ < tol`）。`newton`/`bfgs`/`lbfgs`の3手法すべてで同じ基準を使う（Newtonは独自実装、BFGS/L-BFGSは組み込みの`with_tolerance_grad`）。**最終的な妥当性判断はLogit/Probit実装・テスト段階に持ち越す**: statsmodels/R glmとの数値照合（`test-new`スキル）の結果次第で、閾値を見直す可能性がある
- **デフォルト値**: `tol = 1e-6`（暫定値。実装後の数値照合結果次第で調整）
- **Options化**: `max_iter`と同じ扱いで`tol: f64 = 1e-6`をOptionsに追加する（`run_solver`関数の引数として実装済み。engine_pybind側の配線は後続issue）

**スケール依存への対処（Issue #52で実装内容が確定）**: 勾配の絶対閾値は説明変数のスケールに依存する（`スコア = Σ 残差項 × x_i`のため、xが大きいスケールの列を含むと勾配も大きくなり、真に収束していても`tol`を割らない事態が起きうる）。

- `nonlinear/common.rs`に`standardize_columns(x: &Mat<f64>, has_intercept: bool) -> (Mat<f64>, ColumnScale)`・`destandardize_params(params_std: &[f64], scale: &ColumnScale) -> Vec<f64>`を実装した
- **設計変更（重要）**: 当初合意していた「平均0・分散1」への標準化（`x_std = (x-mean)/std`）は、実装時に**切片なし（`include_intercept=false`）のとき数式が壊れる**ことが判明した。平均を引く変換は、切片が「平均分のズレ」を吸収する前提（逆変換時に`θ_orig_intercept = θ_std_intercept - Σ θ_orig_j * mean_j`という補正が必要）で、切片が無いとこの補正先が存在せず、逆変換後のパラメータが数学的に不正になる
- ユーザーに確認の上、**「分散1のみ（スケーリングのみ、平均は引かない）」に変更した**（`x_std = x/std`）。この場合`θ_orig_j = θ_std_j/std_j`で完結し、切片の有無に関係なく成立する。当初の懸念（勾配ノルムの絶対閾値がxのスケールに依存する）は分散のスケーリングだけで解消できるため、目的は変わらず達成できる
- 却下案: Newton減少量`λ² = g'H⁻¹g`（スケール不変な収束基準）は理論的にはより厳密だが、`bfgs`/`lbfgs`では準ニュートン法が内部で保持する近似逆Hessianへのargmin API経由でのアクセスが必要で、実装できるか不確実なため採用しない
- 標準偏差が0の列（定数列）はスケーリング対象から除外する（0除算回避、`stds`をそのまま`1.0`にする）
- テストは`nonlinear/common.rs`内の`#[cfg(test)] mod tests`に実装。ダミーの2次関数（`f(θ) = 0.5(θ-target)'A(θ-target)`、`A`は対角正定値）でnewton/bfgs/lbfgsそれぞれの収束・既知の最小値への到達・Hessianの正しさを検証、`raise_on_non_convergence`の両分岐、`standardize_columns`/`destandardize_params`の往復変換を検証

## 未確定（実装issue着手時、または追加相談が必要）

- **Tobit固有エラーの正確な検証条件**: 下限<上限の検証、両側打ち切り時の整合性チェック等の詳細。Tobit実装時に決定する
- **モデルごとの尤度・勾配・Hessian導出**（数式そのもの）: Logit/Probit/Tobitそれぞれ着手時に別ノート（`logit-implementation-notes.md`等、または本ファイルへの追記）を作成
