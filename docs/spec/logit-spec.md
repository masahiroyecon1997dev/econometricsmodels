# Logit 仕様書

Logit（二項ロジスティック回帰、最尤推定）の確定済み仕様。`engine/src/nonlinear/logit.rs`（共通基盤は
`engine/src/nonlinear/common.rs`）・`engine_pybind/src/nonlinear/logit.rs`・
`python_package/econometricsmodels/nonlinear/logit.py`として実装済み。nonlinear系統共通の設計判断
（ソルバー実行の共通化、`cov_type`共通行列演算、標準化の方針等）は
[`nonlinear-api-design.md`](../planning/specs/nonlinear-api-design.md)・
[`nonlinear-implementation-notes.md`](../planning/specs/nonlinear-implementation-notes.md)を参照し、
本ドキュメントにはLogit固有の内容のみを記載する。

## 1. API引数

3層構成: `Logit(data, y, x, options).fit() -> LogitResults`（python_package）→
`fit_logit(data, y, x, options) -> LogitResult`（engine_pybind）→ `LogitEstimator::fit`（engine、
Newton-Raphson/BFGS/L-BFGSによる対数尤度最大化）。

- `y: str`（単一列名）、`x: list[str]`。`y`の値は`{0.0, 1.0}`の完全一致のみ許容する（statsmodelsが
  許容する単位区間`[0,1]`の比率データは対象外。本実装は常に真の2値アウトカムのみを扱うため、
  より厳格にした）。範囲外の値は`InvalidBinaryY { row, value }`。
- `LogitOptions`（`#[pyclass]`）:

  | フィールド | 型 | デフォルト | 説明 |
  |---|---|---|---|
  | `cov_type` | `str` | `"classical"` | `"classical"`（alias `"nonrobust"`）/ `"opg"` / `"hc0"` / `"hc1"` / `"cluster"`（大小無視） |
  | `include_intercept` | `bool` | `True` | |
  | `confidence_level` | `float` | `0.95` | |
  | `cluster_col` | `str \| None` | `None` | `cov_type="cluster"`時のグループキー列名 |
  | `method` | `str` | `"newton"` | `"newton"` / `"bfgs"` / `"lbfgs"`（大小無視） |
  | `max_iter` | `int` | `35` | 正整数、以下は`InvalidMaxIter` |
  | `tol` | `float` | `1e-6` | 勾配ノルム収束判定の閾値、以下は`InvalidTol` |
  | `raise_on_non_convergence` | `bool` | `True` | `False`なら未収束時も`converged=False`の結果を返す |

- `start_params`（ユーザー指定初期値）は提供しない。初期値は常にゼロベクトル固定（標準化空間への
  変換ロジックが逆方向にも必要になる点を含め、需要が顕在化してから対応する）。
- `n<=k`は`InsufficientObservations`（OLSと同じ閾値だが根拠は異なる: OLSは残差自由度がゼロ以下という
  数学的必要条件、Logitは`n<=k`がほぼ確実に完全分離を引き起こすという経験則的な安全側の判断）。
  `k==0`（`include_intercept=false`かつ`x`が空、`n<=k`チェックをすり抜けうる病的な入力）は別途
  `NoRegressors { n }`で検証する。
- `include_intercept=True`のとき`x`に`"const"`列があるとエラー（OLSと同じ、自動追加する定数項と衝突）。
  欠損値（NaN/無限大）は常にエラー。

## 2. 結果構造体

`LogitResult`（`#[pyclass]`）が公開する配列＋名前リスト: `params` / `std_errors` / `z_stats`
（**z検定**、t検定ではない） / `p_values` / `conf_lower` / `conf_upper` / `param_names` /
`log_likelihood` / `log_likelihood_null` / `lr_statistic` / `lr_p_value` / `pseudo_r_squared`
（McFadden） / `aic` / `bic` / `n_obs` / `df_model` / `df_resid` / `converged` / `n_iter` /
`cov_type`（実際に使われた種別の小文字文字列）。

- `k×kの分散共分散行列（cov_params）はPython側に公開しない`が、`predict()`/`pred_table()`/
  `marginal_effects()`用に非公開フィールド`estimator: LogitEstimator`として結果オブジェクト内部に
  保持する（`fit()`時の計算を再利用し再最適化を避けるため）。
- `df_model`は`include_intercept`の値に関わらず常に`k-1`固定（OLSの`df_model = k - k_constant`とは
  定義が異なる）。`log_likelihood_null`が常に「切片のみ」モデルを参照するため、LR検定の自由度
  （フィット対象モデルと切片のみnullモデルとのパラメータ数差）として`k-1`に統一している。
  `include_intercept=false`でフィットした場合、この「切片のみ」nullモデルの上位集合（入れ子）には
  ならないため、`lr_statistic`が負になったりp値が統計的に意味の薄い値になったりしうる
  （statsmodels準拠の仕様上の挙動）。`df_model==0`のとき`lr_p_value`はNaN。
- `summary()`（テキスト整形）は作らない（OLSと同じ理由、`ols-spec.md`参照）。
- python_package層（`LogitResults`）: `params`/`std_errors`/`z_stats`/`p_values`/`conf_int`は
  係数名→値の`dict`。`coef_table()`は行指向`list[dict]`（キーは`param`/`coef`/`std_err`/`z_stat`
  （`t_stat`ではない）/`p_value`/`conf_lower`/`conf_upper`）。

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
`destandardize_params`で元のスケールへ逆変換する。`method`（`newton`/`bfgs`/`lbfgs`）に関わらず、
収束点でのHessian評価（SE計算用）は常に解析的に行う。

- **完全分離下でのアンダーフロー対策**: 完全分離に近いデータでは、係数が発散する過程でスコア項
  `p(1-p)`が浮動小数点アンダーフローし、勾配ノルム基準が`tol`の値によらず「収束済み」と誤判定しうる
  （`tol`の調整では解決しない構造的な限界）。対策として、`run_solver`の後処理で標準化パラメータ空間
  でのL2ノルムが閾値`SEPARATION_PARAM_NORM_THRESHOLD=100.0`を超える場合は収束判定を取り消し、
  `MleError::SeparationSuspected { n_iter }`を返す（`raise_on_non_convergence=false`なら
  `converged=false`のまま結果を返す）。閾値はn=200・k=3の単一データセットでの実測較正値であり、
  パラメータ数`k`が大きい多変量モデルでの誤検知リスクは未検証（4章参照）。
- 収束判定`tol`の既定値`1e-6`は、通常データでは高精度（statsmodelsとの相対誤差最大1e-7程度）に
  一致するが、準完全分離の境界ケースではやや不足する（相対誤差最大7e-8）。`tol=1e-8`まで締めると
  改善するが、`bfgs`が`max_iter`を使い切りやすくなるリスクが上がるため、既定値は`1e-6`のまま維持し、
  境界ケースの数値比較テストのみ`tol`を明示的に締める運用とした。

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
  （検証ロジックはOLSの`validate_cluster_groups`と共有、`engine::validation`）。反復最適化のため、
  この検証は`fit()`冒頭・最適化実行前に行う（OLSは閉形式解のため事後検証で足りるのと対照的）。
- Hessianが特異な場合は`SingularHessian`、OPG行列（`Σᵢsᵢsᵢ'`）が特異な場合は`SingularOpgMatrix`
  （原因が異なるため区別）。`method=newton`はステップ求解中の別経路（ピボット付きQR）で先に
  特異性を検出してしまうため、`cov_type`側の特異性検出パスを実際に通すテストには`bfgs`/`lbfgs`が
  必要になる。

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

### 3.6 predict() / pred_table()

`predict()`（引数なし）は`p_i=Λ(x_i'θ)`を返す。`pred_table(threshold)`は2×2的中表（`table[actual][predicted]`）を返す。いずれも学習データのみを対象とする**in-sample限定**（out-of-sample対応は4章）。

- `pred_table`の計算そのもの（`predicted`と`y`のみに依存、リンク関数を参照しない）は`common.rs`の
  `pred_table`関数としてProbitと共有する。`actual`側は`threshold`に関わらず常に**固定0.5**で二値化
  する（`predicted`側のみ`threshold`依存）。これはstatsmodelsの`BinaryResults.pred_table(threshold)`
  の実際の実装（`histogram2d`が常に`[0, 0.5, 1]`でクロス集計する）に合わせた仕様。
- `threshold`自体の値域は検証しない（範囲外でも自明な分類結果になるだけで破綻しないため）。

### 3.7 engine_pybind: エラー変換

`engine::nonlinear::common::MleError` → `PyErr`:

| `MleError` | Python例外 |
|---|---|
| `Common(InsufficientObservations \| InvalidConfidenceLevel \| MissingClusterColumn \| InsufficientClusters \| NoRegressors)` | `ValidationError` |
| `InvalidMaxIter` / `InvalidTol` / `InvalidBinaryY` | `ValidationError` |
| `NonConvergence` / `SingularHessian` / `SingularOpgMatrix` / `SeparationSuspected` | `ComputationError` |

### 3.8 テスト

- 許容誤差: statsmodels主リファレンス（`test_logit_fixtures.py`）は`RTOL=1e-8`。Rクロスチェック
  （`test_logit_crosscheck.py`、反復最適化同士の比較のため機械精度一致は期待できない）は
  `RTOL=2e-4`を基本としつつ、限界効果の`std_err`（デルタ法のヤコビアン経由でノイズが1桁大きい、
  `RTOL=5e-3`）・p値（標準正規分布CDFの裾での増幅、`ATOL=3e-5`）・`near_separation`シナリオの
  信頼区間（`RTOL=6e-4`）を実測に基づき個別に緩めている。
- **statsmodelsのdiscrete modelにおける既知の欠落**: `cov_type="hc1"`は`LogitResults`に
  `cov_HC1`が未定義のためstatsmodelsが暗黙に`hc0`と同じ値を返す（Rの`n/(n-k)`補正版とは一致しない）。
  このためRを主リファレンスとし、`test_logit_fixtures.py`は`hc1`を検証対象から除外する。
  `cov_type="opg"`もstatsmodelsのdiscrete modelはネイティブ非対応（`model.score_obs(params)`から
  手計算する必要がある）。限界効果の`opg`はさらにstatsmodels内部のキャッシュ機構によりRのみが
  参照値になる。
- R側の限界効果リファレンスは`margins`ではなく`marginaleffects`パッケージを採用（メンテナンス状況、
  tidyな出力形式）。`datagrid()`/`slopes(newdata=...)`の`"mean"`/`"median"`ショートカット文字列は
  整数列を丸めてしまうため使わず、`FUN_numeric=mean, FUN_integer=mean`を明示する。
- 合成データセット（`benchmark/nonlinear/generate_nonlinear_datasets.py`）はOLSの9シナリオを
  ベースに、誤差項構造に依存するもの（不均一分散・自己相関等）を除外し、Logit特有の病理シナリオ
  `near_separation`（準完全分離、係数を大きくしてp≈0/1が支配的になる状況）を追加した7シナリオ。
  実データは`mroz`（Wooldridge、労働参加モデル）。
- クラスターロバストSEの境界ケース（クラスター数G=2ちょうど）は、OLSの`wald_f_test`と異なり
  Logitの`cluster_cov_params`がq×q部分行列の反転を要求しないため、説明変数の数を絞る必要がない。

## 4. 未実装・未対応

- `predict()`/`pred_table()`のout-of-sample対応（`new_data`引数）
- `start_params`（ユーザー指定初期値）
- `SEPARATION_PARAM_NORM_THRESHOLD`の多変量モデル（k大）での誤検知リスク: L2ノルムは`k`が増えるほど
  各成分が中程度でも合計が大きくなりやすく、真に分離していないケースでの誤検知は未検証
- `SeparationSuspected`検出が使う量（標準化パラメータのL2ノルム）と実際にアンダーフローを
  引き起こす量（線形予測子`|x_std_i・θ_std|`の最大値）は相関的な関係に過ぎず、数学的に保証された
  関係ではない（例: 特定の1列のみが分離に寄与するケースでは検出漏れがありうる）
- 完全分離でNonConvergenceになるシナリオ（`complete_separation`）のベンチマーク: 3.2の既知の限界
  （アンダーフローによる誤収束判定）により意図通りに動作しないため見送り
