# Logit 仕様書

Logit（二項ロジスティック回帰、最尤推定）の確定済み仕様。`engine/src/nonlinear/logit.rs`（共通基盤は
`engine/src/nonlinear/common.rs`）・`engine_pybind/src/nonlinear/logit.rs`・
`python_package/econometricsmodels/nonlinear/logit.py`として実装済み。nonlinear系統共通の設計判断
（ソルバー実行の共通化、`cov_type`共通行列演算、標準化の方針等）は
[`nonlinear-common.md`](./nonlinear-common.md)を参照し、
本ドキュメントにはLogit固有の内容のみを記載する。

## 1. API引数

3層構成: `Logit(data, y, x, options=options).fit() -> LogitResults`（python_package）→
`fit_logit(data, y, x, options) -> LogitResult`（engine_pybind）→ `LogitEstimator::fit`（engine、
Newton-Raphson/BFGS/L-BFGSによる対数尤度最大化）。

- `y: str`（単一列名）、`x: list[str]`。`y`の値は`{0.0, 1.0}`の完全一致のみ許容する（statsmodelsが
  許容する単位区間`[0,1]`の比率データは対象外。本実装は常に真の2値アウトカムのみを扱うため、
  より厳格にした）。範囲外の値は`InvalidBinaryY { row, value }`。
- `LogitOptions`（`#[pyclass]`）:

  | フィールド | 型 | デフォルト | 説明 |
  |---|---|---|---|
  | `cov_type` | `str` | `"classical"` | `"classical"`/ `"opg"` / `"hc0"` / `"hc1"` / `"cluster"`（大小無視） |
  | `include_intercept` | `bool` | `True` | |
  | `confidence_level` | `float` | `0.95` | |
  | `cluster` | `str \| None` | `None` | `cov_type="cluster"`時のグループキー列名。他の`cov_type`で指定すると`ValidationError` |
  | `solver` | `str` | `"newton"` | `"newton"` / `"bfgs"` / `"lbfgs"`（大小無視） |
  | `max_iter` | `int` | `35` | `1`以上`10000`以下の整数（上限は`engine::shared::validation::MAX_ITER_LIMIT`、収束しない問題が実質的に終わらなくなるのを防ぐ）。範囲外は`InvalidMaxIter` |
  | `tol` | `float` | `1e-6` | 勾配ノルム収束判定の閾値、以下は`InvalidTol` |
  | `raise_on_non_convergence` | `bool` | `True` | `False`なら未収束時も`converged=False`の結果を返す |

- `start_params`（ユーザー指定初期値）は提供しない。反復最適化の初期値（warm start）は、標準化空間の
  設計行列に対する線形確率モデル（LPM）最小二乗解に、logitのIRLS 1ステップ相当のスケール補正を
  施したものを内部で自動生成する（3.2節、`solver`に依らず共通）。
- `n<=k`は`InsufficientObservations`（OLSと同じ閾値だが根拠は異なる: OLSは残差自由度がゼロ以下という
  数学的必要条件、Logitは`n<=k`がほぼ確実に完全分離を引き起こすという経験則的な安全側の判断）。
  `k==0`（`include_intercept=false`かつ`x`が空、`n<=k`チェックをすり抜けうる病的な入力）は別途
  `NoRegressors { n }`で検証する。
- `include_intercept=True`のとき`x`に`"const"`列があるとエラー（OLSと同じ、自動追加する定数項と衝突）。
  欠損値（null・NaN/無限大）は常にエラー（方針は[`docs/guide/validation.md`](../guide/validation.md)）。

## 2. 結果構造体

`LogitResult`（`#[pyclass]`）が公開する配列＋名前リスト: `params` / `std_errors` / `test_stats`
（**z検定**、t検定ではない） / `p_values` / `conf_lower` / `conf_upper` / `param_names` /
`dep_var_name` / `log_likelihood` / `log_likelihood_null` / `lr_statistic` / `lr_p_value` / `lr_df`（χ²の自由度、`df_model==0`で`None`） / `pseudo_r_squared`
（McFadden） / `aic` / `bic` / `n_obs` / `df_model` / `df_resid` / `converged` / `n_iter` /
`cov_type`（実際に使われた種別の小文字文字列） / `solver`（実際に使われたソルバーの小文字文字列）。

- `k×kの分散共分散行列（cov_params）`は`cov_type`に応じた値をPython側に公開する（`ols-spec.md`
  「結果構造体」参照。Probit・Tobitも同様で、Tobitは`β∪{σ}`の`(k+1)×(k+1)`）。加えて、`predict()`/
  `pred_table()`/`marginal_effects()`用に非公開フィールド`estimator: LogitEstimator`として結果
  オブジェクト内部に保持する（`fit()`時の計算を再利用し再最適化を避けるため）。
- `df_model`は`include_intercept`の値に関わらず常に`k-1`固定（OLSの`df_model = k - k_constant`とは
  定義が異なる）。`log_likelihood_null`が常に「切片のみ」モデルを参照するため、LR検定の自由度
  （フィット対象モデルと切片のみnullモデルとのパラメータ数差）として`k-1`に統一している。
  `include_intercept=false`でフィットした場合、この「切片のみ」nullモデルの上位集合（入れ子）には
  ならないため、`lr_statistic`が負になったりp値が統計的に意味の薄い値になったりしうる
  （statsmodels準拠の仕様上の挙動）。`df_model==0`のとき`lr_p_value`はNaN。
- `summary()`（テキスト整形）は作らない（OLSと同じ理由、`ols-spec.md`参照）。
- python_package層（`LogitResults`）: `params`/`std_errors`/`test_stats`/`p_values`/`conf_int`は
  係数名→値の`dict`。`params_table()`は行指向`list[dict]`（キーは`param_name`/`param`/`std_error`/`test_stat`/`p_value`/`conf_lower`/`conf_upper`）。

## 3. 内部実装の計算仕様

### 3.1 尤度・スコア・Hessian

`p_i = Λ(x_i'θ) = 1/(1+exp(-x_i'θ))`（ロジスティック関数）。対数尤度は`z_i=x_i'θ`とおくと
`ℓ_i(θ) = y_i z_i - softplus(z_i)`（`softplus(z)=log(1+exp(z))`）という同値な形に書き換えられる
（オーバーフローを避けやすいため実装はこちらを使う）。

- スコア: `∂ℓ/∂θ = X'(y-p)`。Hessian: `∂²ℓ/∂θ∂θ' = -X'WX`（`W=diag(pᵢ(1-pᵢ))`）。対数尤度は大域的に
  凹なので、厳密な多重共線性がなければ`X'WX`は常に正定値。
- **符号規約**: `argmin`クレートは最小化フレームワークのため`CostFunction::cost = -ℓ(θ)`、
  `Gradient`/`Hessian`トレイトも同じ符号（`-ℓ`の1階・2階微分）で実装する。`LogitProblem::scores()`
  （`cov_type`共通行列演算向けの独自メソッド）は符号反転しない生のスコア`sᵢ=(yᵢ-pᵢ)xᵢ`を返す。
- **数値安定化**: `softplus(z) = z.max(0) + log1p(exp(-|z|))`、`logistic(z)`は`z>=0`/`z<0`で分岐し
  `exp`の引数を常に非正に保つ（いずれも有限の`z`ではどれだけ極端でもNaN/Infを産まない設計）。

### 3.2 最適化・収束判定

`LogitEstimator::fit`は設計行列を標準化した空間（`standardize_columns`）で最適化し、収束後に
`destandardize_params`で元のスケールへ逆変換する。`solver`（`newton`/`bfgs`/`lbfgs`）に関わらず、
収束点でのHessian評価（SE計算用）は常に解析的に行う。

- **初期値（warm start）と設計行列のランクチェック**: `solver`に関わらず、最適化を開始する前に
  標準化空間の設計行列を列ピボットQR分解する（`nonlinear::common::checked_design_matrix_qr`）。
  `R`の対角成分の相対閾値`k·ε·max|R_ii|`（OLS・Newton法と共通の`linear_algebra::checked_col_piv_qr`）でランク落ちを
  検出し、完全な多重共線性等は`MleError::SingularDesignMatrix`（`ComputationError`）で弾く。この
  QR解＝標準化空間のLPM最小二乗解 `b_lpm` に、nullモデル `p≡p̄`（`p̄=ȳ`）を起点にしたIRLSの
  1反復目に相当するスケール補正を施したものを初期値にする: 全成分を `1/w` 倍し（`w`はlogitの
  IRLS重み `p̄(1-p̄)`）、切片成分にのみ `η₀ - p̄/w`（`η₀ = ln(p̄/(1-p̄))`）を加える
  （`β⁽¹⁾ = b_lpm/w + (η₀ - p̄/w)·e₀`）。`p̄`が0/1の近傍（全観測で`y`が同一）では補正が発散する
  ため素の`b_lpm`にフォールバックする。従来のゼロベクトル初期値から変更した。この変更で
  多重共線性の検出経路が`solver`非依存の単一経路（前段QR）に統一され、従来`bfgs`/`lbfgs`のみ
  検出が収束後の`observed_information_cov_params`に依存していた構造的な差が解消された。収束先の
  推定値・既存のクロスチェック数値は不変（尤度が大域凹で無制約のためMLEは一意）。

- **完全分離・準完全分離の検出（`nonlinear::separation`、Probitと共有）**: ある`θ`について全観測で
  `zᵢ'θ ≥ 0`（`zᵢ=(2yᵢ-1)xᵢ`）かつどれかが厳密に正なら、`θ`の方向へ係数を大きくするほど対数尤度が
  増え続け、有限の最尤推定量は存在しない（Albert & Anderson 1984）。全観測で`zᵢ'θ>0`なら完全分離、
  `≥0`で一部が0なら準完全分離（ダミー変数が1の観測は全て`y=1`、など）。この条件は閾値を持たない
  厳密な判定で、行を単位長にそろえた`uᵢ=zᵢ/‖zᵢ‖`について線形計画
  `max Σᵢuᵢ'θ  s.t.  uᵢ'θ≥0, |θⱼ|≤1`に帰着する。正常なデータでは許容解が原点だけで解の最大マージンは0、
  分離していれば分離した観測に`O(1)`のマージンが付く。
  収束後に`fit()`が`verify_no_separation`を1回呼ぶ（ソルバーが収束していないときは呼ばない）。
  分離なら`raise_on_non_convergence=true`で`MleError::SeparationDetected`（`ComputationError`）、
  `false`なら`converged=false`のまま結果を返す。標準化後の設計行列は複製せず、元の`x`と
  `ColumnScale`から行を都度計算する。
  - **常にLP（制約生成）で判定する。事前スクリーニングは採らない**: LPは全観測を一度に制約に
    入れず、小さな作業集合の制約で解き、違反した制約（最大64本/ラウンド）だけを足して繰り返す。
    実測では3〜4ラウンド・作業集合は数百本以下（上限2000本、超えたら数値的な異常として扱う）。
    当初は、収束点の係数方向`θ̂`とHessianの最小固有ベクトル方向でほぼ全観測が正側のときだけLPを
    走らせる事前スクリーニングを置いていたが、分離方向が複数ある（錐の次元が2以上）と、情報行列の
    近零固有空間が多次元になり最小固有ベクトルが錐の外を向いて、構造的に取りこぼすと分かった
    （3つのダミーが同時に準完全分離するデータで、lbfgsの200件中2件が`converged=True`のまま
    返った）。取りこぼしは分離を無警告の「成功」として返す検出漏れになるため廃止し、常にLPで
    確かめる。判定は、LPの解`θ`が全制約を満たす（違反1e-9以内）うえで最大マージン
    `maxᵢ uᵢ'θ`が`1e-6`を超えるかで行う（和ではなく最大値を使うのは、和は`n`に比例して雑音が
    積もり、大標本で正常データが誤検知されうるため）。
  - **LPソルバーの差し替え**: LPは`shared::linear_program::LinearProgramSolver`トレイトの裏にあり、
    自前の密な単体法`DenseSimplex`を使う。変数を`x=p-q`に分け箱制約を行として足す標準形で、原点が
    許容解なのでPhase 1が要らない。右辺0の錐制約は原点で強く退化するため、(1)入る変数はDantzig規則、
    比検定の同点は絶対値最大のピボット要素、(2)退化ピボットが30回続いたらBland規則に切り替えて
    巡回を断つ、(3)制約の右辺に行ごとに異なる`1e-11`級の決定的な摂動を足して退化そのものを解く。
    Bland規則だけの初版は`k>=10`でピボット上限に達して有効なデータを失敗させ、同じ`x`で結果が
    逆の行対のようにDantzig/Blandでも終了しない入力（`k=30`）があった。将来LPクレートに替える場合は、
    トレイトの実装を1つ足して`verify_no_separation`の既定実装を切り替えるだけでよい。
  - **LPが数値的に完了しないとき**（ピボット上限・作業集合の肥大・解が自分の制約を大きく破る）:
    分離が無いと確認できなかったので、`raise_on_non_convergence=true`なら`MleError::SeparationCheckFailed`
    （`code`は`SEPARATION_CHECK_FAILED`、`ComputationError`）、`false`なら推定結果は返して
    `converged=false`にする。実測のデータでは発生していない。
  - **実測（合成データ、scipy/HiGHSの全制約LPを正解とする）**: 分離データ883件（`n=5〜2000`・
    Logit/Probit・完全/準完全分離）を`newton`で全件検出、見逃し0・誤検知0。`bfgs`/`lbfgs`は
    分離データの一部でソルバー自身が`NonConvergence`等になるが、いずれも`ComputationError`で無警告の
    「成功」は0。`k=10/20/30`（`n=2000`、完全分離と少数ラベル反転）225件で正解と全件一致、LP失敗0。
    3つのダミーが同時に準完全分離するデータは3 solverとも200/200件で検出。`tol`が`1e-8`〜`1e-3`でも
    結果は変わらない。
  - **境界の実測（1観測だけが分離を壊すデータ）**: 完全分離データ（`x`が`±0.01〜±1.00`、
    `n=201`）に`y=1`の観測を1つ、分離を壊す側へ`overlap`だけずらして足し、`overlap`を変えて
    判定を測った（`tests/support/_nonlinear_helpers.py`の`single_overlap_binary_dataset`）。`overlap=1e-4`以上は
    分離とみなさず収束する（係数は`x1`で500前後と大きいが有限のMLE、`1e-3`で約300、`1e-4`で約530）。
    `overlap=1e-5`以下は分離として`ComputationError`（`raise_on_non_convergence=false`なら
    `converged=false`、係数は760〜990に達している）。境界はこのデータで`overlap`が`1e-5`〜`3e-5`
    の間にあり、標準化後の行の幾何とマージン閾値`1e-6`で決まるためデータごとに変わる。このため
    境界そのものは固定せず、明らかに両側にある値（`overlap=1e-3,1e-4`は分離とみなさない、
    `1e-7,1e-9,1e-12`は分離とみなす）を、Rust（`separation.rs`の
    `single_overlapping_observation_is_flagged_only_when_the_overlap_is_tiny`）とPython
    （Logit/Probit × `newton`/`bfgs`/`lbfgs`）で固定している。有限のMLEが存在するのに分離と
    扱われる領域（`overlap≲1e-5`、係数が1000近くまで大きくなる）が存在するが、そこでは係数・
    標準誤差が数値的に意味を持たないため、「分離として止める」側に倒す設計。
  - **性能**: 分離検査は`n=1,000,000`・`k=6`（切片込み）で約68ms（Newtonの`fit()`約620msの約11%）、
    `n=100,000`で約4ms（約8%）。行列を数パス走査するだけでメモリ帯域が律速。LPが走る準完全分離
    データでも`fit()`全体は2〜3秒。
  - **旧方式（標準化パラメータのL2ノルムが閾値`100`を超えたら収束を取り消す）は廃止した**。
    当初は「完全分離下のスコア項`p(1-p)`のアンダーフローで勾配ノルム基準が誤って収束と判定する」
    と考えていたが、実測で機序が違うと分かった: 勾配の絶対閾値`tol`は、分離方向で最も飽和していない
    観測の残差が`tol`を下回れば満たされるため、停止時のマージンは`ln(1/tol)`程度（Probitは約5）で
    `n`にほぼ依存しない。ノルムは「マージン÷点と分離超平面の幾何学的な隙間」で決まり、`n`が小さいと
    隙間が大きく、閾値`100`に届く前に止まる。`n=5`（`n=k+1`）では分離したLogit 500件中415件が
    `converged=True`のまま標準誤差が数千の結果を返し、`n≲15〜20`まで見逃しが続いた。`n`依存の閾値に
    しても、`n=8〜15`では分離解と正常解のノルム分布が重なり判別できない。準完全分離（ダミー型）は
    `n=1000`でもノルムが10程度で、全て見逃した。逆に強い多重共線性では、有限のMLEが存在するのに
    係数が符号反対に大きく振れてノルムだけが膨張し、Probitで`20`件中`8`件が誤って棄却された。
- **`bfgs`/`lbfgs`のline searchの評価回数バジェット（実装済み）**: `MoreThuenteLineSearch`
  （argmin組み込み）は、`bfgs`の自前実装（`FaerBfgs`）・`lbfgs`の自前実装（`FaerLbfgs`）いずれの
  経路でも内側`Executor`に反復上限が設定されておらず（argmin既定`u64::MAX`）、ステップ幅の上限
  （`stpmax`）も未設定（既定`f64::INFINITY`）。探索方向・勾配が退化しline search内部の収束判定が
  一度も発火しない入力に嵌ると、この内側ループは理論上終了しない（devビルドで80分超のCPU時間を
  消費し続けるケースを実測で確認、`nonlinear::tobit::tests::proptests`で発覚）。対策として
  `run_solver`は`problem`を`BudgetedProblem`（`common.rs`）でラップし、目的関数・勾配の評価回数に
  総枠（`(max_iter + 1) * 2000`。`+1`は`Executor::init()`の初回評価分の枠を
  `max_iter=0`でも確保するため）を設ける。枠を使い切ると`MleError::EvaluationBudgetExceeded { budget }`
  を返す。argminのソースを変更せずに`bfgs`/`lbfgs`両方を保護できるのは、
  `MoreThuenteLineSearch`が呼び出し元によらず必ず`Problem<O>`経由で`cost`/`gradient`を呼ぶため
  （詳細は`common.rs`の`BudgetedProblem`docコメント参照）。
- 収束判定`tol`の既定値`1e-6`は、通常データでは高精度（statsmodelsとの相対誤差最大1e-7程度）に
  一致するが、準完全分離の境界ケースではやや不足する（相対誤差最大7e-8）。`tol=1e-8`まで締めると
  改善するが、`bfgs`が`max_iter`を使い切りやすくなるリスクが上がるため、既定値は`1e-6`のまま維持し、
  境界ケースの数値比較テストのみ`tol`を明示的に締める運用とした。
- **`tol`は総和勾配に対する絶対閾値で観測数`n`でスケールしない**。`FaerNewton`
  （Newton）には、`regularized_newton_step`のLMラダーが全失敗したとき`λ=0`のHessianが可逆かつ
  勾配ノルムが停滞（前反復比`≥0.9`）かつ収束目標近傍（`<1e4·tol`）なら収束扱いにする副次判定を
  追加した。Logit/Probitは尤度が大域凹でこの経路に入らず挙動不変（大標本Tobitで顕在化。
  詳細は[`tobit-spec.md`](./tobit-spec.md)3.2節）。
- **`bfgs`/`lbfgs`は`tol`を観測数`n`で正規化した基準を使う（実装済み）**:
  `newton`は上記の通り総和勾配に対する絶対閾値のまま（既定`1e-6`）だが、`bfgs`/`lbfgs`は
  「観測あたり平均勾配」基準`‖∇ℓ(θ)‖ / n < tol`（既定`1e-8`）を使う。実装上は`run_solver`
  （`nonlinear/common.rs`）が`tol * n_obs`を実効的な絶対閾値としてソルバーへ渡す形で、
  `FaerBfgs`/`FaerLbfgs`自体は正規化を知らない。`n_obs`は各モデルの`fit()`が観測数
  （`y.nrows()`）をそのまま渡す。statsmodels（scipy）が対数尤度・スコア・Hessianを観測数`n`で
  割ってから最適化する設計（`statsmodels.base.model.LikelihoodModel.fit`の`f=-loglike/nobs`等）に
  倣ったもの。`newton`は2次収束のため`tol`の絶対閾値のままでも大標本での追加反復はごく僅かで
  済むが、`bfgs`/`lbfgs`は超1次収束のため、正規化前は同じ絶対閾値を満たすのに`n`が大きいほど
  反復・関数評価が増えていた（実測: `n`=100,000→1,000,000で`bfgs`の反復回数が16→17〜22）。
  - **`newton`にも同じ正規化を適用する案は検証の上で不採用**: `newton`の絶対閾値`tol`を観測数
    正規化に統一すれば既定値を1つに揃えられるため検討したが、実測したところ`near_separation`
    （Logit/Probit）・`heavy_censoring`/`interval_censoring`（Tobit）等、境界条件に近いシナリオで
    `RTOL=1e-8`の精度検証テストが28件失敗した（例: `interval_censoring/hc1/wald_statistic`で
    相対誤差6.5e-6）。`newton`は元々2次収束による高精度を前提にテストが組まれているため、
    `n`が小さいケースで正規化により実効閾値が緩む（`tol*n < tol`となる`n<1`は無いが、`n`倍する
    こと自体で絶対閾値`1e-8`が`n*1e-8`まで緩む）とこの精度マージンを食いつぶす。この結果を
    ユーザーに報告の上、`newton`は正規化せず既定値`1e-6`の絶対閾値のまま維持する方針を確定した。
  - **既定値は`solver`により異なる**: `newton`（既定`tol=1e-6`、絶対閾値）と`bfgs`/`lbfgs`
    （既定`tol=1e-8`、正規化基準）で共有の単一既定値は採れない——`newton`の既定を`1e-8`
    （絶対閾値のまま）に締めると大標本で無視できない速度低下（実測: `n`=1,000,000で
    0.98秒→3.15秒）が生じる一方、`bfgs`/`lbfgs`の既定を`1e-6`（正規化基準）に緩めると
    `n`=500の精度検証テストを壊す（後述）。`tol`を明示的に指定した場合は、選択した`solver`の
    意味論（`newton`なら絶対、`bfgs`/`lbfgs`なら正規化）がそのまま適用される。
  - **実測（`generate_binary_choice_dataset("baseline", link="logit", n=1_000_000, k=5, seed=42)`、
    公式計測ハーネス`performance.compare_logit --worker`、単一スレッド）**: `bfgs`は
    実行時間15.0秒→0.73秒（既定`tol=1e-8`、正規化後の実効閾値`1e-2`相当）。`newton`は
    0.63秒（既定`tol=1e-6`のまま、旧実装から不変）。`lbfgs`は17.5秒で、旧実装（18.2秒）から
    ほぼ改善しない——同じ実効閾値でも`bfgs`より収束に必要な反復数がずっと多い（実測:
    実効閾値`1e-2`相当で`bfgs`は6反復・`lbfgs`は17反復）ため、`bfgs`に効いた既定値の正規化
    だけでは`lbfgs`は速くならない。`lbfgs`固有により緩い既定値が必要かどうかは別Issueで検討する
    （今回のスコープは正規化という仕組みの導入のみ、ユーザー確認済み）。
  - **小標本での精度は既定値`1e-8`で維持される**: `tests/fixtures/benchmarks/logit.json`の
    `baseline`シナリオ（`n=500`、`.claude/rules/testing-policy.md`の基本方針`相対誤差1e-8`と
    厳密比較する）で、`bfgs`の`tol=n*1e-8=5e-6`（既定値と同じ実効閾値）は`tol=1e-6`（旧既定の
    絶対閾値）と完全に同じ反復回数（7）・精度（相対誤差1.77e-9）だった。一方`tol=n*1e-6`
    （実効閾値100倍緩い設定）を試すと相対誤差3.84e-6まで悪化する——既定値を`1e-8`にした
    理由はこの安全マージンによる（`1e-6`ではなく`1e-8`を選んだ根拠）。

### 3.3 標準誤差

`CovType`（Logit/Probit/Tobit共通、`engine::nonlinear::common`）: `Classical` / `Opg` / `Hc0` / `Hc1` /
`Cluster { groups }`。いずれも標準化空間で`Σ_std`を計算した後、`destandardize_cov_params`
（`Σ_orig[i,j] = Σ_std[i,j]/(stds[i]*stds[j])`、`D=diag(stds)`とした`Σ_orig = D⁻¹Σ_stdD⁻¹`の成分表示）
で元のスケールに戻す。

| `cov_type` | 式 |
|---|---|
| `classical`（既定） | 観測情報行列 `Σ = -H⁻¹` |
| `opg` | outer product of gradients `Σ = (Σᵢ sᵢsᵢ')⁻¹` |
| `hc0` | サンドイッチ型 `Σ = H⁻¹(Σᵢ sᵢsᵢ')H⁻¹` |
| `hc1` | `hc0`に小標本補正 `n/(n-k)` を乗じる |
| `cluster` | `Σ = correction・H⁻¹(Σ_g S_gS_g')H⁻¹`（`S_g`はグループ内スコア和、小標本補正込み） |

検定分布は**標準正規分布**（z検定、statrs `Normal`）。

- クラスターのグループキー未指定は`MissingClusterColumn`、クラスター数`<2`は`InsufficientClusters`
  （検証ロジックはOLSの`validate_cluster_groups`と共有、`engine::shared::validation`）。反復最適化・多段
  推定の無駄を避けるため、この検証は全手法で`fit()`冒頭・最適化実行前に行う（OLS/WLS も他手法に揃えた。
  OLSは閉形式解のため事後検証でもコストは変わらないが、位置を統一）。
- クラスター数`G <= 傾き係数の数q`（`k - k_constant`）は`InsufficientClustersForInference`
  （`ValidationError`）。クラスターロバスト共分散`Ŝ`はクラスター寄与スコアの総和が
  ゼロ（MLEの一次条件`Σᵢsᵢ = 0`）のため`rank(Ŝ) ≤ G - 1`で、`G <= q`だと退化する。Logit/Probitは
  全体検定がLR（`lr_statistic`）のため`q×q`部分行列の反転こそ通らないが、退化した共分散から
  読んだSEを黙って返すのは識別失敗の隠蔽（fail-fast方針・多重共線性をエラーで止めるのと整合）
  のため、`fit()`冒頭で弾く（従来はsilent-passだった、実質バグ。OLS/WLS/Tobit/IVと横断で統一）。
  少数クラスタ一般の漸近的信頼性（`G=5, q=2`等、計算は通るケース）は別軸で、これは弾かない。
- 完全な多重共線性等の設計行列のランク落ちは、`solver`に依らず最適化前の列ピボットQR
  （`checked_design_matrix_qr`、3.2節）で`SingularDesignMatrix`として弾く。前段で弾かれた後に
  なお発生しうるHessianのランクエラーはHessian自体が特異な場合は`SingularHessian`、OPG行列
  （`Σᵢsᵢsᵢ'`）が特異な場合は`SingularOpgMatrix`（原因が異なるため区別）。

### 3.4 適合度統計量

`log_likelihood_null`（切片のみモデルのllf）はリンク関数に依存しない閉じた形
`ℓ_null = n1・ln(ȳ) + n0・ln(1-ȳ)`（`n1`/`n0`はy=1/0の観測数、`0*ln(0)`となる項は0として扱う）で
直接計算する（再最適化しない。切片のみモデルのMLEが`link(θ̂)=ȳ`を満たすことに由来し、Probit/Tobitでも
同じ式を共有する）。`lr_statistic`/`lr_p_value`（カイ二乗分布）/`pseudo_r_squared`/`aic`/`bic`は
`goodness_of_fit(llf, llnull, n, k)`（`nonlinear/common.rs`、Logit/Probit共通）が計算する。

### 3.5 限界効果

`marginal_effects(at, confidence_level)`は`fit()`とは独立したメソッド。`at="overall"`（既定、AME）/
`"mean"`/`"median"`いずれも`dy/dx_j = w(θ)*θⱼ`という同じ形に帰着する（`w`はLogitでは`p(1-p)`、
`at="overall"`は全観測平均、`mean`/`median`は代表点での1点評価）。デルタ法の標準誤差は
`w`の勾配`s`から求めるヤコビアンと`fit()`時の`cov_params`の二次形式（`jac_j・Σ・jac_j'`）で計算する。
定数項は出力から除外する。`column_means`/`column_medians`・ヤコビアン計算・デルタ法本体
（`dydx_and_jacobian`/`marginal_effects_from_w_s`）はリンク関数に依存しないためProbitと共有する
（`nonlinear/common.rs`）。

### 3.6 predict() / augment() / pred_table()

`predict(new_data=None)`は`p_i=Λ(x_i'θ)`を返す。`new_data`が`None`（既定）なら学習データ
（`fit()`に使ったデータ）に対する予測確率、指定すれば新規データ（out-of-sample）に対する予測確率を
返す（`new_data`の列名マッチング・`include_intercept`時の定数項自動付加はOLSの`predict(new_data)`
（[`ols-spec.md`](./ols-spec.md)「predict()」）と同じ規約）。`pred_table(threshold)`は
2×2的中表（`table[actual][predicted]`）を返す。こちらは学習データのみを対象とする
**in-sample限定**のまま（out-of-sample対応は4章、別issueでトラッキング）。

- `predict(new_data)`のengine側実装は`nonlinear::common::predict_new_data`（`engine::linear::ols::
  predict_new_data`のLogit/Probit共有版、`link`関数を差し替えられるようにしたもの）。新規データの
  設計行列組み立て（`has_intercept`時の定数項自動付加）は、OLS/WLSと共有する`crate::shared::design_matrix::
  design_matrix_element`ヘルパーに委ねている。
- `pred_table`の計算そのもの（`predicted`と`y`のみに依存、リンク関数を参照しない）は`common.rs`の
  `pred_table`関数としてProbitと共有する。`actual`側は`threshold`に関わらず常に**固定0.5**で二値化
  する（`predicted`側のみ`threshold`依存）。これはstatsmodelsの`BinaryResults.pred_table(threshold)`
  の実際の実装（`histogram2d`が常に`[0, 0.5, 1]`でクロス集計する）に合わせた仕様。
- `threshold`自体の値域は検証しない（範囲外でも自明な分類結果になるだけで破綻しないため）。
- **戻り値とOLSの`predict()`との意味の違い**: `predict()`は観測順の`list[float]`（OLS/WLS/Tobitと
  同じ形）で、各値は確率（`[0, 1]`の連続値）。OLSの`predict()`が返す`y`自体の点予測とは意味が
  異なる。0/1の分類結果が欲しい場合は`predict()`の出力に自前でしきい値を適用するか、学習データに限り
  `pred_table()`を使う。これはstatsmodelsの`predict()`と同じ標準的な慣習であり、docstringの
  注記で明示する。
- 検証: Rクロスチェック（`run_glm_crosscheck.R`）が全エントリで`pred_table`（確率>0.5で1、既定のしきい値のみ）・標本内予測確率（先頭10行と全行の合計）・新規データ（学習データの各傾きの平均±1標準偏差の2行）の予測確率を出し、`predict()`と`augment()`の両方を`RTOL=2e-7`で比べる。statsmodels側は`pred_table`を全ケースで、`predict(new_data)`をライブで比べる（切片あり・なし）。
- **`augment(new_data=None)`は`predict()`と同じ`new_data`意味論**で、ソースデータ（学習データまたは
  `new_data`）に予測確率の列（`"predicted_probability"`）を1列付加したpolars DataFrameを返す（OLSの
  `augment()`と同型）。列名衝突（ソースデータに既に`"predicted_probability"`列が
  ある場合）は`ValidationError`（`engine_pybind::shared::augment::append_predicted_column`）。
  `LogitResult`は`fit()`に渡された元の`polars.DataFrame`を非公開で保持し、
  `augment(new_data=None)`はそれに予測値の列を足して返す（推定に使わない列も残る。
  `IVResult.first_stage()`のような単一DataFrameを持たない構築経路が無いため、OLSと異なり
  `Option`にせず常に保持する。`WLSResult`と同じ設計）。

### 3.7 engine_pybind: エラー変換

`engine::nonlinear::common::MleError` → `PyErr`（バリアントごとの`code`・Python例外は
[`nonlinear-common.md`](./nonlinear-common.md)2章の表が正本。`MleError::report()`が`code`を決め、
クラスはカタログの`code`で決まる）。Logit/Probitに関わるバリアントの要点:

| `MleError` | `code` | Python例外 |
|---|---|---|
| `Common(InsufficientObservations \| InvalidConfidenceLevel \| MissingClusterColumn \| InsufficientClusters \| InsufficientClustersForInference)`・`InvalidMaxIter` / `InvalidTol` / `InvalidBinaryY` | `INSUFFICIENT_OBSERVATIONS`・`CONFIDENCE_LEVEL_OUT_OF_RANGE`・`CLUSTER_REQUIRED`・`CLUSTER_COUNT_TOO_SMALL`・`CLUSTER_COUNT_NOT_ABOVE_SLOPES`・`MAX_ITER_OUT_OF_RANGE`・`TOLERANCE_NOT_POSITIVE_FINITE`・`OUTCOME_NOT_BINARY` | `ValidationError` |
| `Common(NearSingularMatrix { matrix })`・`SingularHessian` / `SingularOpgMatrix` | `MATRIX_NEAR_SINGULAR`（`matrix`が`COEFFICIENT_COVARIANCE` / `HESSIAN` / `OPG`） | `ComputationError` |
| `SingularDesignMatrix` | `DESIGN_MATRIX_SINGULAR` | `ComputationError` |
| `NonConvergence` | `NOT_CONVERGED` | `ComputationError` |
| `SeparationDetected` / `SeparationCheckFailed` | `SEPARATION_DETECTED` / `SEPARATION_CHECK_FAILED` | `ComputationError` |
| `EvaluationBudgetExceeded` / `LineSearchFailed` | `LINE_SEARCH_FAILED` | `ComputationError` |

### 3.8 テスト

- 許容誤差: statsmodels主リファレンス（`test_logit_reference.py`）は`RTOL=1e-8`。Rクロスチェック
  （`test_logit_crosscheck.py`、反復最適化同士の比較のため機械精度一致は期待できない）は
  `RTOL=2e-7`（係数・SE・p値・信頼区間・限界効果の全統計量（`effect`/`std_error`/`test_stat`/`conf_lower`/`conf_upper`）・cluster 3ケース（均等・不均衡・境界G=q+1）の全統計量で共通、実測最大~3.5e-8。切片なし（`baseline.no_intercept`、全cov_type・均等/不均衡クラスター）も同じ許容誤差で通る（実測~1e-9以下）。クラスター数の境界`G = q + 1`（`baseline.cluster_g4`、q=3でG=4）も同じ許容誤差で通る（`G <= q`のエラーはbaselineでG=2・G=3を確認）。非既定の`confidence_level`（0.90、`baseline.confidence_level`、全cov_type＋クラスター）も同じ許容誤差で、係数の信頼区間（`Options`の水準）と限界効果の信頼区間（`marginal_effects(confidence_level=)`の水準）の両方を比べる（statsmodels側のopgの限界効果は算出不可）。`ATOL=1e-12`で
  `|ref|`が小さい項目も純粋な相対誤差で比較する）を全統計量に共通で使う（p値の専用許容誤差は置かない。裾のp値（`|ref|`が約5e-6未満）は`ATOL`の下限で比較されるため、同じ係数のz値の厳密な比較で担保している）。R側の参照値は
  `glm()`の収束判定を`epsilon=1e-14`に厳しくした上で、収束後の係数から再fitを係数が動かなくなるまで繰り返し
  （`glm()`の収束判定はデビアンスの変化量だけなので係数が~1e-7ずれて止まりうる。また
  `sandwich::estfun()`は1反復前の係数で作った作業重みを使うため、既定の1e-8だとロバストSEの
  参照値に~3e-5、`epsilon=1e-14`でもk=20では~1e-7のノイズが乗る）、`marginaleffects`の有限差分の
  刻み幅を変数ごとに標準偏差の1e-5倍にして生成する（`benchmark/nonlinear/references/run_glm_crosscheck.R`）。
- **statsmodelsのdiscrete modelにおける既知の欠落**: `cov_type="hc1"`は`LogitResults`に
  `cov_HC1`が未定義のためstatsmodelsが暗黙に`hc0`と同じ値を返す（Rの`n/(n-k)`補正版とは一致しない）。
  このためRを主リファレンスとし、`test_logit_reference.py`は`hc1`を検証対象から除外する。
  `cov_type="opg"`もstatsmodelsのdiscrete modelはネイティブ非対応（`model.score_obs(params)`から
  手計算する必要がある）。限界効果の`opg`はさらにstatsmodels内部のキャッシュ機構によりRのみが
  参照値になる。
- R側の限界効果リファレンスは`margins`ではなく`marginaleffects`パッケージを採用（メンテナンス状況、
  tidyな出力形式）。`datagrid()`/`slopes(newdata=...)`の`"mean"`/`"median"`ショートカット文字列は
  整数列を丸めてしまうため使わず、`FUN_numeric=mean, FUN_integer=mean`を明示する。
- 合成データセット（`benchmark/nonlinear/datasets.py`）はOLSの9シナリオを
  ベースに、誤差項構造に依存するもの（不均一分散・自己相関等）を除外し、Logit特有の病理シナリオ
  `near_separation`（準完全分離、係数を大きくしてp≈0/1が支配的になる状況）・`complete_separation`・
  `quasi_complete_separation`を追加した11シナリオ（数値比較8、`ComputationError`確認のみ3:
  `complete_separation`・`quasi_complete_separation`・`perfect_multicollinearity`）。
  実データは`mroz`（Wooldridge、労働参加モデル）と、多クラスターの`apple`（Wooldridge、
  `y = ecolbs > 0`、州で`G=49`・グループサイズ1〜66、説明変数は`ecoprc`・`regprc`・`faminc`・`hhsize`・
  `educ`・`age`・`reglbs`）。`apple`は`mroz`の`city`（`G=2`）が`G<=q`で成功パスを持てないため、
  実データのクラスターロバストSEの数値照合を担う（statsmodelsとRの両方で全統計量。
  Wooldridgeデータは固定CSVにしないため、`y`はロードの都度導出する）。
  `apple`は平均が標準偏差に比べて大きい列（`ecoprc`・`regprc`・`educ`等）を含み、標準化で平均を
  引かないと`lbfgs`が既定の`max_iter=35`では収束しなかった（約95〜96反復）。切片ありでは平均も
  引く標準化（`nonlinear-common.md`1.3節）により19・14反復になり、`solver`の組み合わせテストは
  `max_iter`を指定せず既定値で収束を確認する。
- **`solver`の組み合わせ**: `newton`は全シナリオ×`cov_type`で数値照合する。`bfgs`/`lbfgs`は
  baseline×classicalに加え、代表的な組み合わせ（`near_separation`×classical/opg/hc0、cluster・
  cluster_imbalanced、`apple`のcluster）を、statsmodelsの`newton`で生成した同じシナリオ・
  `cov_type`の全統計量に対して`rtol=2e-5`で照合する（`tests/support/_tolerances_nonlinear.py`の
  `logit_reference.rtol_solver_vs_newton`、実測最大~3.6e-6）。statsmodels自身の`bfgs`/`lbfgs`を
  参照にしないのは、`near_separation`で収束許容が緩く本実装とLogitで最大~7e-3（Probitは~8e-4）乖離する
  （statsmodels側の誤差）ため。
- **クラスターのstatsmodels照合は全統計量**（baselineの均等・不均衡疑似グループとも、係数・SE・
  z値・p値・信頼区間・適合度統計量・限界効果）。
- クラスターロバストSEの境界ケース（クラスター数G=2ちょうど）は、OLSの`wald_f_test`と異なり
  Logitの`cluster_cov_params`がq×q部分行列の反転を要求しないため、説明変数の数を絞る必要がない。

## 4. 未実装・未対応

- `predict()`のout-of-sample対応（`new_data`引数）は実装済み（3.6参照）。
  `pred_table()`のout-of-sample対応は引き続き未実装（別issueでトラッキング）。
- `augment()`は実装済み（3.6参照）。
- `start_params`（ユーザー指定初期値）
- **分離検出の既知の限界**: (1)`bfgs`/`lbfgs`は分離データで分離検出に至る前にソルバー自身が
  `NonConvergence`や`SingularHessian`で失敗することがある（いずれも`ComputationError`で、無警告の
  「成功」にはならない）。(2)LPが数値的に完了しない異常（実測のデータでは発生していない）は
  `SEPARATION_CHECK_FAILED`の`ComputationError`または`converged=false`になる。(3)判定は単位長にそろえた行の余弦で、`1e-6`程度の許容を持って「全制約を満たす非自明な方向」の有無を
  見る。そのため重なりが十分小さい「ほぼ分離」は有限のMLEが存在しても分離と扱われ、境界は重なりの
  大きさ（1観測だけが分離を壊すデータで`1e-5`前後、3.2節の実測）と標準化後の行の幾何で決まる。
  境界の外側（重なりが大きい）では分離とみなさず推定が進み、標準誤差が大きくなる。
  (4)制約を作業集合に足す上限（2000本）を超える病的な入力は数値的な異常として扱う。
- **完全分離でNonConvergenceになるシナリオ（`complete_separation`）・準完全分離（`quasi_complete_separation`）**:
  `benchmark/nonlinear/datasets.py`にシナリオとして追加済み（`perfect_multicollinearity`と同型、
  数値比較の対象外）。サンプルサイズに依らず、newton/bfgs/lbfgsいずれでも`ComputationError`になる。
  `raise_on_non_convergence=False`では、newtonは例外を送出せず`converged=False`で結果を返す
  （`tests/nonlinear/`で固定）。bfgs/lbfgsはソルバー自身の失敗経路に入ることがあり、結果を返した
  場合は`converged=False`で標準誤差が有限であることのみを保証する。
