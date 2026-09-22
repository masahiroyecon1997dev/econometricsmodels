# 非線形モデル（MLEベース）共通仕様

Logit/Probit/Tobit（最尤推定ベースの非線形モデル）が共有する基盤の確定済み仕様。
`engine/src/nonlinear/common.rs`・`engine_pybind/src/nonlinear/`として実装済み。手法固有の
内容（尤度・スコア・Hessianの具体式、限界効果の`w`の式等）は
[`logit-spec.md`](./logit-spec.md) / [`probit-spec.md`](./probit-spec.md) /
[`tobit-spec.md`](./tobit-spec.md)を参照し、本ドキュメントには3手法が共有する設計判断のみを
記載する。将来実装予定の多項ロジット・順序ロジット/プロビット等もこの基盤を前提とする。

## 1. 数値最適化基盤

### 1.1 ライブラリ: `argmin`

`argmin`（Apache-2.0/MIT）を採用。`CostFunction`/`Gradient`/`Hessian`トレイトでモデルごとに
尤度・勾配・Hessianを実装する方式。候補として検討した`ipopt-rs`（システムのBLAS/LAPACKに
依存し、faer方針・マルチOS wheel配布と両立しない）・`cobyla`（微分不要で収束が遅く精度も
劣る）は却下した。

- **`argmin-math`はfaerバックエンドを使わない**: `argmin-math`（0.5.1）のfaerバックエンドは
  faer 0.23までしか対応しておらず、本プロジェクトの`faer = "=0.24.4"`ピンとは噛み合わない。
  `argmin-math`の`vec`機能（`Vec<f64>`/`Vec<Vec<f64>>`のみ）を使い、モデル固有の尤度・勾配・
  Hessian計算や`cov_type`の行列演算は引き続きfaer 0.24.4で行う。argminとの境界
  （`run_solver`関数の内部）でのみ`Vec<Vec<f64>>`⇔`faer::Mat`の変換を行う（k×kで
  パラメータ数は小さくコストは無視できる）。

### 1.2 ソルバー（`method`）

`method`引数を公開する。値は`"newton"`（既定）/`"bfgs"`/`"lbfgs"`（statsmodelsの`method`
引数の値に揃えた文字列、大小無視）。3手法いずれも解析的スコアが書け、Newton-Raphsonを
既定にできる（statsmodelsの既定とも一致）。BFGS/L-BFGSはHessian計算が重い・不安定な
ケースのフォールバックとして用意する。Nelder-Mead/SANN等の勾配不要法はv1では対象外。

**Newton法・BFGS・L-BFGSはいずれも自前実装**（`FaerNewton`/`FaerBfgs`/`FaerLbfgs`、
`argmin::core::Solver`トレイトを直接実装し`Executor`/`MoreThuenteLineSearch`自体は流用する
設計）。argmin組み込みソルバーがモデル固有に必要な制御点を公開していないため。

- **Newton**: 組み込み`Newton`ソルバーは`H: ArgminInv<H>`（Hessianの逆行列）を要求するが、
  `argmin-math`の`vec`機能には`ArgminInv`の実装が存在しない。Newtonステップの求解
  （`H·Δθ = g`）はfaerの列ピボットQR（`col_piv_qr`、OLSの`ensure_full_rank`と同じ相対閾値
  での特異性検出）で行う。特異なら`MleError::SingularHessian`。
- **BFGS**: 組み込みBFGSの初期逆Hessーは単位行列（スケール`O(1)`）固定で、尤度Hessianの
  スケール（`O(n)`、観測数`n`個のスコアの和）との乖離が`n`が大きいほど開き、line searchが
  1反復あたり多数の関数評価を消費する（実測: n=1,000,000でnewton 0.65s・組み込みbfgs
  11.21s）。Nocedal & Wright 6.1節のself-scaling初期化・line searchの初期ステップ幅の反復間
  調整を行うための制御点（`linesearch`フィールド等）が組み込みBFGSに公開されていないため
  自前実装した。1回目の反復専用のline search初期ステップ幅を`min(1,1/‖g₀‖)`に調整し、
  2回目以降は標準の`alpha=1.0`に戻す。secant条件`yᵀs>0`を満たさない場合はrank-2更新を
  スキップする（閾値は絶対値`f64::EPSILON`固定、相対閾値は実測で反復回数・実行時間が
  かえって悪化したため不採用）。
- **L-BFGS**: 同じ理由（`s`/`y`履歴・初期`γ`を注入する公開APIが無い）で自前実装
  （`two_loop_recursion`、Nocedal & Wright Algorithm 7.4）。line search内側`Executor`に
  明示的な`max_iters`（`LINE_SEARCH_MAX_ITERS=100`）を設定し、到達時は`ComputationFailed`
  として明示的にエラー化する（`FaerBfgs`には無い追加ガード）。`two_loop_recursion`の
  逆向きループには`yᵀs<=f64::EPSILON`のペアをゼロ除算回避のため無視するガードを持つが、
  secant条件によるhistory admissionガード自体は**不採用**（履歴凍結による収束の悪化を実測で
  確認したため。`MoreThuenteLineSearch`のstrong Wolfe条件により受理されたペアは理論上
  `yᵀs>0`を自然に満たす）。
- **line searchの評価回数バジェット**: `MoreThuenteLineSearch`は内側`Executor`に反復上限
  （既定`u64::MAX`）・ステップ幅上限（既定`f64::INFINITY`）が無いため、退化した入力で理論上
  終了しないケースがある。`run_solver`は`problem`を`BudgetedProblem`でラップし、目的関数・
  勾配の評価回数に総枠`(max_iter + 1) * 2000`を設ける。枠を使い切ると
  `MleError::EvaluationBudgetExceeded`。

**収束点のHessian評価**: `method`の3分岐で`Executor::run()`実行後、最終パラメータで
Hessianを1回評価し直す（Newtonの最後のイテレーションで計算済みのものを使い回さない。
3手法で同じコードパスにできるため）。BFGS/L-BFGSの内部近似逆Hessianは使い回さず、
`cov_type="classical"`（観測情報行列）には常に解析的Hessianを使う。

**`Hessian`トレイトの符号規約**: `CostFunction`/`Gradient`と同じ符号（負の対数尤度の
Hessian）に統一する。`run_solver`内で`model.hessian(&params)`を呼んだ直後に1回だけ符号
反転し、以降（`SolverOutput.hessian`、`cov_type`共通行列演算）は対数尤度そのものの
Hessianとして扱う（`Σ_classical = -H⁻¹`が成り立つ前提と一致させる）。各モデルの`fit()`
実装は符号変換を意識しなくてよい。

### 1.3 収束判定

**Options**:

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `max_iter` | `int` | `35` | 最大反復回数。`method`に関わらず単一の値（statsmodelsもdiscreteモデルで`method`に依らず`maxiter=35`を一律適用） |
| `tol` | `float \| None` | `None` | 勾配ノルム収束判定の閾値。`None`時は`method`により既定値が異なる（下記） |
| `raise_on_non_convergence` | `bool` | `True` | `True`なら`max_iter`到達時に`ComputationError`。`False`なら最終反復時点のパラメータを`converged=False`として返す |

**Return**: `converged: bool` / `n_iter: int`。

statsmodels（`ConvergenceWarning`を出しつつ結果は必ず返す＝常に寛容）とは異なり、本
プロジェクトは**デフォルトで厳格（例外を投げる）**とする。未収束の結果をそれと知らず
使ってしまうリスクを避けるため。

**判定基準・既定値は`method`により異なる**:

- **`newton`**: 総和勾配に対する絶対閾値`‖∇ℓ(θ)‖ < tol`。既定`tol=1e-6`。2次収束のため
  観測数`n`が増えても追加反復はごく僅かで済む。
- **`bfgs`/`lbfgs`**: 観測数`n`で正規化した「観測あたり平均勾配」基準
  `‖∇ℓ(θ)‖ / n < tol`。既定`tol=1e-8`（実装上は`run_solver`が`tol * n_obs`を実効的な
  絶対閾値として渡す）。超1次収束のため、正規化しないと`n`が大きいほど同じ絶対閾値を
  満たすのに必要な反復・関数評価が増える（statsmodels/scipyが対数尤度・スコア・Hessianを
  `n`で割ってから最適化する設計に倣った）。
- **共有の単一既定値は採れない**: `newton`の既定を`1e-8`に締めると大標本で無視できない
  速度低下（実測n=1,000,000で0.98s→3.15s）が生じ、`bfgs`/`lbfgs`の既定を`1e-6`に緩めると
  `n=500`の精度検証テストを壊す。`newton`にも正規化を適用する案は、`near_separation`・
  `heavy_censoring`等の境界シナリオで`RTOL=1e-8`精度検証テストが28件失敗したため不採用
  （`newton`は2次収束による高精度前提でテストが組まれているため、正規化で実効閾値が
  `n`倍緩むと精度マージンを食いつぶす）。
- **大標本Newtonの副次収束判定**: `tol`は総和勾配に対する絶対閾値でスケールしないため、
  大標本ではコスト関数が浮動小数点の底に達しても勾配ノルム基準が発火しないことがある。
  LMラダー（正則化）が全失敗し、かつ`λ=0`のHessianが可逆（真に特異ではない）な場合、
  (1) 勾配の停滞（前反復比`≥0.9`）・(2) 収束目標近傍（`<1e4・tol`）・(3) コストHessianの
  正定値性（鞍点除外）の3条件が揃えば収束扱いにする（`stalled_at_optimum`）。Logit/Probitは
  尤度が大域凹でこの経路に入らず挙動不変（Tobitで顕在化、[`tobit-spec.md`](./tobit-spec.md)
  3.2節参照）。

**スケール依存への対処（標準化）**: 説明変数のスケールが異なると勾配の絶対閾値の妥当性が
崩れるため、`nonlinear/common.rs`の`standardize_columns`/`destandardize_params`で
標準化空間で最適化する。**「分散1のみ（スケーリングのみ、平均は引かない）」**
（`x_std = x/std`）——「平均0・分散1」への標準化は`include_intercept=false`のとき
逆変換の数式が壊れる（切片が「平均分のズレ」を吸収する前提が成り立たないため）ことが
判明し不採用にした。標準偏差が0の列（定数列）はスケーリング対象から除外する。

### 1.4 設計行列のランクチェック・初期値

**多重共線性の検出は`fit()`冒頭の列ピボットQRランクチェックに一本化**: `method`に関わらず、
`run_solver`を呼ぶ前に`nonlinear::common::checked_design_matrix_qr(x_std)`を必ず通し、
ランク落ちを`MleError::SingularDesignMatrix`（`ComputationError`）で弾く。旧来のゼロベクトル
初期値では検出経路が`method`ごとに分かれ（newtonは`newton_step`内QR、bfgs/lbfgsは収束後の
`observed_information_cov_params`）、bfgs/lbfgsのみ検出漏れする構造的リスクがあった。

Logit/Probitはこの列ピボットQR解（標準化空間のLPM最小二乗解）を、nullモデル
`p≡p̄`起点のIRLS 1反復目に相当するスケール補正を施した値で初期値（warm start）にも使う
（`ols_based_initial_params`）。Tobitは`ols_initial_params`がQR解を`β`初期値に、残差を
`σ`初期値にする。`start_params`（ユーザー指定初期値）は現状未対応。

## 2. エラー型（`MleError`、共有）

Logit/Probit/Tobit共有の1つのエラー型を`engine/src/nonlinear/common.rs`に定義する
（OLSの「1手法1エラー型」パターンを横展開せず共有型にする。`raise_on_non_convergence`
未収束・観測数不足・`confidence_level`範囲外等、3手法でほぼ共通のバリアントが多いため）。
Tobit固有のバリアント（打ち切り境界の検証等）も同じ`MleError`にバリアントとして追加する
（型を分けない）。

| バリアント | Python例外 | 由来 |
|---|---|---|
| `Common(CommonError)` | `ValidationError`（大半）/ケースによる | `DimensionMismatch`/`InsufficientObservations`/`InvalidConfidenceLevel`/`MissingClusterColumn`/`InsufficientClusters`/`InsufficientClustersForInference`/`NoRegressors`/`ComputationFailed`。系統横断で重複するバリアントは`CommonError`（`engine::error`）に切り出し、`MleError`は`#[error(transparent)] Common(#[from] CommonError)`で包む |
| `InvalidMaxIter` / `InvalidTol` | `ValidationError` | `max_iter<=0`等 |
| `NonConvergence { n_iter }` | `ComputationError` | `raise_on_non_convergence=true`かつ`max_iter`到達 |
| `SingularDesignMatrix` | `ComputationError` | 最適化前の列ピボットQRランクチェックでのランク落ち（1.4節） |
| `SingularHessian` | `ComputationError` | 収束点のHessianが特異で観測情報行列の逆行列が計算できない |
| `SingularOpgMatrix` | `ComputationError` | OPG行列（`Σsᵢsᵢ'`、`cov_type="opg"`）が特異。`SingularHessian`と原因が異なるため別バリアントに分離 |
| `SeparationSuspected { n_iter }` | `ComputationError` | 完全分離下のアンダーフローによる誤収束判定を検出（Logit/Probitのみ、3.4節） |
| `EvaluationBudgetExceeded { budget }` | `ComputationError` | line searchの評価回数バジェット超過（1.2節） |
| `InvalidCensoringBounds` 等（Tobit専用） | `ValidationError` | 下限≧上限等の不正な指定 |

`Common`以外の各バリアントは`engine_pybind`側で`MleError` → `PyErr`変換
（`mle_error_to_pyerr`、`common_error_to_pyerr`に委譲）を行う。

## 3. `cov_type`共通行列演算

`s_i`を観測`i`のスコアベクトル（対数尤度の1階微分）、`H`を収束点で評価した対数尤度の
Hessianとする。いずれも標準化空間で`Σ_std`を計算した後、`destandardize_cov_params`
（`Σ_orig = D⁻¹Σ_stdD⁻¹`、`D=diag(stds)`）で元のスケールに戻す。

| `cov_type` | 式 |
|---|---|
| `classical`（別名`nonrobust`、既定） | 観測情報行列 `Σ = -H⁻¹` |
| `opg` | outer product of gradients `Σ = (Σᵢ sᵢsᵢ')⁻¹`（BHHH） |
| `hc0` | サンドイッチ型 `Σ = H⁻¹(Σᵢ sᵢsᵢ')H⁻¹`（misspecification-robust） |
| `hc1` | `hc0`に小標本補正`n/(n-k)`を乗じる |
| `cluster` | `Σ = correction・H⁻¹(Σ_g S_gS_g')H⁻¹`、`S_g = Σ_{i∈g} sᵢ`（OLSの`cluster_cov_params`と同型） |

- HC2/HC3は対象外（レバレッジ・hat行列に依存した補正で線形回帰特有の概念のため）。HACも
  対象外（時系列拡張として保留）。
- **符号反転を1回の計算に集約**: `neg_hessian_inverse(H) = (-H)⁻¹`をCholesky分解で1回だけ
  計算し、`observed_information_cov_params`（`Σ=-H⁻¹`）・`sandwich_cov_params`・
  `cluster_cov_params`で同じ戻り値を再利用する（`-H⁻¹=(-H)⁻¹`、`H⁻¹ΨH⁻¹=(-H)⁻¹Ψ(-H)⁻¹`が
  成り立つため追加の符号反転が不要）。
- **クラスターの小標本補正**はOLSと同じ規約（`correction = G/(G-1) * (n-1)/(n-k)`を常に
  適用、無効化オプションなし）を踏襲する。
- **OLSとの相違点（自由度切り替え不要）**: 非線形モデルはz検定（自由度という概念がない）
  のため、OLSの`cov_type=Cluster`時の自由度切り替え（`n-k`→`G-1`）に相当する処理が不要。
  分散共分散行列のスケーリング（`correction`）だけ気にすればよい。
- **クラスター数`G`は傾き係数の数`q`（`k - k_constant`）より多くなければならない**
  （`G <= q`は`InsufficientClustersForInference`、`ValidationError`、Issue #289）。
  クラスターロバスト共分散`Ŝ`はクラスター寄与スコアの総和がゼロ（MLEの一次条件
  `Σᵢsᵢ=0`）のため`rank(Ŝ) ≤ G-1`で、`G<=q`だと退化する。Logit/Probitは全体検定がLR
  （`q×q`部分行列の反転を要求しない）だが、退化した共分散から読んだSEを黙って返すのは
  識別失敗の隠蔽になるため`fit()`冒頭で弾く（OLS/WLS/Tobit/IVと横断統一）。少数クラスタ
  一般の漸近的信頼性（`G=5, q=2`等、計算は通るケース）は別軸で弾かない。
- **クラスターキー未指定・クラスター数不足の検証はこの共通関数のスコープ外**（呼び出し側の
  責務、OLSの`validate_cluster_groups`を共有し`engine::validation::validate_cluster_groups`
  に集約済み）。反復最適化の無駄を避けるため、この検証は全手法`fit()`冒頭・最適化実行前に
  行う（OLS/WLSも閉形式解だが位置を統一）。
- **特異性検出は固有値分解ベースの相対閾値判定を経由する**: 非ピボットCholesky分解の失敗
  だけでは構造的な特異性・悪条件を確実には検出できない（`method=Bfgs`/`Lbfgs`は
  `newton_step`のピボット付きQRを経由しないため顕在化した）。`ensure_well_conditioned_
  symmetric_matrix`（`engine/src/linear_algebra.rs`、`SelfAdjointEigen`ベース、OLSの
  `wald_f_test`用実装を系統横断で共有）をCholesky分解の前に呼び、エラー時は
  `SingularHessian`/`SingularOpgMatrix`にマップする。

## 4. 検定分布

**標準正規分布（z検定）**を採用する。MLEの漸近理論`θ̂ ~ N(θ, Σ)`に基づき、
`z = θ̂ⱼ / se(θ̂ⱼ)`、p値は標準正規分布の両側確率、信頼区間は標準正規分布の臨界値を使う
（`statrs::distribution::Normal`）。OLSは「`cov_type`に関わらずt分布で統一」という方針だが、
MLEベースの非線形モデルは漸近理論が正規分布に基づいており、OLSの`n-k`に相当する自然な
自由度が存在しないためt分布統一方針は踏襲しない（statsmodels/R glmともにz検定が標準
であることとも一致）。

## 5. Return共通コア項目

- `params` / `std_errors` / `z_stats` / `p_values` / `conf_lower` / `conf_upper` /
  `param_names`
- `log_likelihood`（llf）/ `log_likelihood_null`（切片のみモデルのllf）
- `lr_statistic` / `lr_p_value`（尤度比検定、カイ二乗分布。OLSのF検定に相当する全体の
  有意性検定）
- `pseudo_r_squared`（McFadden方式）
- `aic` / `bic`
- `n_obs` / `df_model` / `df_resid`
- `converged` / `n_iter`（1.3節）
- `cov_type`（3章）

`k×kの分散共分散行列（cov_params）はPython側に公開しない`が、`predict()`/`pred_table()`/
`marginal_effects()`用に非公開フィールド`estimator`として結果オブジェクト内部に保持する
（`fit()`時の計算を再利用し再最適化を避けるため）。`summary()`は作らない。

**Tobitはこの共通コアから2点を意図的に外す**（詳細は[`tobit-spec.md`](./tobit-spec.md)2章）:
`log_likelihood_null`/`pseudo_r_squared`は実装しない（切片のみTobitに閉形式が無く、
`AER::tobit`自体もpseudo R²を持たない）。`lr_statistic`/`lr_p_value`は`wald_statistic`/
`wald_p_value`に置き換える（`llnull`の再最適化が不要な`summary.tobit`と同じ方式）。LR検定は
将来拡張候補として9章に残す。

## 6. 限界効果・予測確率・的中表（別メソッド方針）

いずれも`fit()`のReturn本体には含めず、**Resultオブジェクトの別メソッド**として提供する。

- `marginal_effects(at="overall" | "mean" | "median", ...)`: 限界効果。既定は`at="overall"`
  （AME、average marginal effects）。標準誤差はデルタ法で計算し、`fit()`時の`cov_params`を
  再利用する（再最適化不要）。Return形式は`coef_table`と同じ行指向
  （`dydx`/`std_err`/`z`/`p_value`/`conf_low`/`conf_high`）。「見る/見ない」を切り替える
  フラグは設けない（可変なのは`at`のみ）。
- `predict()`: 予測確率（Logit/Probit）。
- `pred_table()`: 分類の的中表（閾値依存のため、コアのReturnには含めない。Logit/Probit
  のみ）。

**限界効果の共通骨格**: `at="overall"`（AME）・`"mean"`・`"median"`はいずれも
`g_j(θ)=w(θ)*θⱼ`という同じ形に帰着する（`w`はAMEなら全観測平均、mean/medianなら代表点
評価。式自体はリンク関数依存でモデルごとに異なる）。この性質を使い、`w`とその勾配
`s_m=∂w/∂θ_m`の計算（`at`ごとに異なる、モデルファイル側に実装）と、そこから`dydx`・
ヤコビアン`∂g_j/∂θ_m=θⱼ*s_m+[j==m]*w`を計算する部分（`at`に依らず共通、
`dydx_and_jacobian`/`marginal_effects_from_w_s`として`nonlinear/common.rs`に実装）を
分離する。分散は`Var(g_j) = jac_jの行ベクトル・cov_params・jac_jの行ベクトル'`（二次形式）。

- **離散変数（0/1のダミー変数）の自動判定は行わない**: データ入力がlist渡し（列の型情報を
  持たない、CLAUDE.md非交渉事項）のため、statsmodelsの`dummy=True`相当の離散差分は実装
  せず、常に連続変数として扱う（`dummy=False`相当）。
- **定数項は出力から除外する**（経済学的に意味を持たないため。`include_intercept`の値に
  関わらず先頭`k_constant`列をスキップ）。
- **`at="mean"`/`"median"`の代表点**は学習データの各列の標本平均・標本中央値からなる固定
  ベクトル（外部から任意の評価点を指定するオプションは見送り）。
- **Tobitは`predict()`/`marginal_effects()`/`pred_table()`のいずれも独自の形になる**
  （詳細は[`tobit-spec.md`](./tobit-spec.md)3.5〜3.6節）: `predict(target)`/
  `marginal_effects(target)`は`E[y*|x]=x'β`・`E[y|x]`（既定）・`P(uncensored|x)`の3対象を
  取る（McDonald-Moffitt 1980）。3対象とも`dydx_j = w(θ)·βⱼ`に帰着するが`w`の式が対象
  ごとに異なるためLogit/Probitの`dydx_and_jacobian`とは共有せず独立実装する。`pred_table()`
  は廃止し`censoring_fit_check()`（方向別の観測打ち切り率 vs モデル含意率）に置き換える。

## 7. モデル固有オプション

| モデル | オプション | 状態 |
|---|---|---|
| Logit / Probit | `start_params: Option<Vec<f64>>`（ユーザー指定初期値） | 未実装（見送り）。`None`時の内部初期値は1.4節のwarm start |
| Tobit | 打ち切り方向（左/右/両側）・下限/上限値 | 確定（`TobitOptions.lower: Option<f64>`既定`Some(0.0)`・`upper: Option<f64>`既定`None`。両方`None`・`lower>=upper`は`InvalidCensoringBounds`、`y`が境界と矛盾する場合は`YOutOfCensoringBounds`） |
| Tobit | `dist`（誤差分布） | Gaussian固定。他分布はv1では対象外 |
| 多項ロジット | 参照カテゴリ（`base_category`等） | 着手時に決定 |
| 順序ロジット/プロビット | 閾値パラメータ数 | オプション化しない。yのカテゴリ数（K個）から`K-1`個を自動導出する |
| 全般 | `weights`（頻度/分析重み）、`offset` | 見送り。Phase6のIO手法で必要になった時点で追加検討 |

## 8. リファレンス実装・テスト比較ライブラリ

| モデル | 主リファレンス | 交差検証 |
|---|---|---|
| Logit / Probit | statsmodels | R `glm()` |
| Tobit | R `AER::tobit`（`survival::survreg`エンジン） | R `censReg`（`maxLik`エンジン。`survreg`と`maxLik`は最適化実装が完全に独立しているため交差検証として組み合わせる価値が高い） |
| 多項ロジット | R `nnet::multinom`、statsmodels `MNLogit` | ― |
| 順序ロジット/プロビット | R `MASS::polr` | ― |

- Python製`py4etrics`（Tobit/Truncreg/Heckit/probit）は教材付随パッケージで査読・保守体制が
  無いため、`censReg`を本命、`py4etrics`は余力があれば追加で見る程度の位置づけとする。
- **アーキテクチャの参考: R `maxLik`パッケージ**。モデル別実装ではなく汎用MLEエンジンとして
  設計され、`censReg`（Tobit）等がこれをエンジン層として利用している。「共通engine +
  モデル固有のloglike/gradient/hessian」という本プロジェクトの構造と近い。`finalHessian`
  パターン（最適化に使ったソルバーと標準誤差算出に使う情報行列の種類を分離できる設計）は
  「収束点のHessian評価」（1.2節）の設計判断の参考にした。

## 9. 未実装・未対応・将来課題

- 多項ロジット・順序ロジット/プロビットの参照カテゴリ等の詳細仕様（着手時に決定）
- `start_params`（ユーザー指定初期値）
- Tobitの尤度比検定（LR statistic/p-value）の追加実装（v1では`wald_statistic`のみ）
- `NEWTON_STALL_GRAD_FACTOR`（1.3節の副次収束判定が使う絶対閾値）を、スケール不変な
  Newton減少量`√(gᵀH⁻¹g)`ベースの停止基準に置き換える検討（スコープ超で保留）
- 手法固有の未実装・既知の限界（分離検出の較正等）は各手法のspec4章を参照
