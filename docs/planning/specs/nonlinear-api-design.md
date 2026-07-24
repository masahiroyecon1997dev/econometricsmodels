# 非線形モデル（MLEベース）設計検討メモ

対象: Logit / Probit / Tobit / 多項ロジット・プロビット / 順序ロジット・プロビット
関連: CLAUDE.md 2章（非交渉事項）、4章（実装フェーズ）、13章（今後の検討事項）

---

## 1. 実装対象・着手順序

- **優先順位**: Logit / Probit（最優先）→ Tobit → 多項・順序モデル（難易度が易しいものから）
- 多項・順序モデルはCLAUDE.md 4章のPhase分類に明示的に含まれていないため、フェーズ再整理が必要になる可能性がある（未対応）

---

## 2. 数値最適化ライブラリ・ソルバー

### 2.1 ライブラリ: `argmin`（確定）

- **ライセンス**: Apache-2.0 / MIT のデュアルライセンス（MIT採用で問題なし）
- `argmin-math`経由で**faerバックエンドに公式対応**しており、CLAUDE.mdの「faer採用・システムBLAS/LAPACK非依存」方針と整合する
- `CostFunction` / `Gradient` / `Hessian`トレイトでモデルごとに尤度・勾配・Hessianを実装する方式

比較検討した候補と却下理由:

| 候補 | 却下・留保理由 |
|---|---|
| `ipopt-rs` | システムのIpopt本体（+ BLAS/LAPACK系）に依存。マルチOS wheel配布（`cd_release.yml`）とfaer方針の両方に反する |
| `cobyla` | 微分不要（COBYLA法）で収束が遅く精度も劣る。補助的選択肢に留める |
| `basin` | argmin類似・faer統合済みの新興クレートだが、実績・エコシステム成熟度でargminに劣るため見送り |

### 2.2 ソルバー選択（確定）

- `method`引数をユーザーに公開する。値は`"newton"`（既定）/ `"bfgs"` / `"lbfgs"`の3つ
- 対象3手法（Logit/Probit/Tobit）はいずれも解析的スコア（勾配）が書け、Newton-Raphsonを既定にできる（statsmodelsの既定とも一致）。BFGS/L-BFGSはHessian計算が重い・不安定なケースのフォールバックとして用意する
- Nelder-Mead/SANN等の勾配不要法は**v1では対象外**（4.2節のmaxLikのエスカレーション梯子のうち、下2段（勾配なし→NM/SANN）は今回の対象手法では発生しないため不要と判断）
- 文字列はstatsmodelsの`method`引数の値（`'newton'`/`'bfgs'`/`'lbfgs'`）に揃え、ユーザーが馴染みやすいようにする

### 2.3 Tobitの打ち切り構造について

境界制約付き最適化（ipopt-rsが得意とする領域）が必要に見えるが、Tobitの打ち切りは標準的には**尤度関数自体で表現**でき、制約なし最適化（argminのNewton/BFGS/L-BFGS）で対応可能と判断。

---

## 3. 収束判定（確定）

**Options**:

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `max_iter` | `int` | `35` | 最大反復回数。ソルバー（`method`）に関わらず単一の値。statsmodelsもdiscreteモデルの`fit()`で`method`に関わらず`maxiter=35`を一律適用しており、ソルバーごとに値を分ける慣習が無いことを確認した上でこれに倣う |
| `raise_on_non_convergence` | `bool` | `True` | `True`の場合、`max_iter`回で収束しなければ`ComputationError`を送出する。`False`の場合、最終反復時点のパラメータを結果として返す（`converged=False`として） |

**Return**:

| フィールド | 型 | 説明 |
|---|---|---|
| `converged` | `bool` | 収束したかどうか |
| `n_iter` | `int` | 実際に使った反復回数 |

statsmodels（`ConvergenceWarning`を出しつつ結果は必ず返す＝常に寛容）とは異なり、本プロジェクトは**デフォルトで厳格（例外を投げる）**とする。未収束の結果をそれと知らず使ってしまうリスクを避けるため。探索的に緩めたいユーザーは`raise_on_non_convergence=False`を明示する。

---

## 4. 標準誤差（`cov_type`、確定）

| 文字列 | 内容 |
|---|---|
| `"classical"` / `"nonrobust"`（エイリアス、既定） | 観測情報行列（Hessianの負の逆行列） |
| `"opg"` | BHHH（期待情報行列近似。各観測のスコアの外積和） |
| `"hc0"` | サンドイッチ型ロバスト |
| `"hc1"` | サンドイッチ型ロバスト（小標本補正） |
| `"cluster"` | クラスターロバスト |

- HC2/HC3は対象外（レバレッジ・hat行列に依存した補正で線形回帰特有の概念のため）
- HACも対象外（今回のスコープ外の時系列拡張として保留）
- `"classical"` / `"nonrobust"`のエイリアス化はOLSの既存実装（[`engine_pybind/src/linear/ols.rs:102`](../../../engine_pybind/src/linear/ols.rs#L102)）に倣った

---

## 5. Return内容（共通コア、確定）

- `params` / `std_errors` / `z_stats` / `p_values` / `conf_lower` / `conf_upper` / `param_names`
- `log_likelihood`（llf）/ `log_likelihood_null`（切片のみモデルのllf）
- `lr_statistic` / `lr_p_value`（尤度比検定、カイ二乗分布。OLSのF検定に相当する全体の有意性検定）
- `pseudo_r_squared`（McFadden方式）
- `aic` / `bic`
- `n_obs` / `df_model` / `df_resid`
- `converged` / `n_iter`（3章）
- `cov_type`（4章）

**検定分布はz検定（正規分布）**を採用する。OLSは「`cov_type`に関わらずt分布で統一」という本プロジェクト独自の方針を取っているが、MLEベースの非線形モデルは漸近理論が正規分布に基づいており、OLSの`n-k`に相当する自然な自由度が存在しないため、t分布統一方針はここでは踏襲しない。statsmodels/R glmともにz検定が標準であることとも一致する。

---

## 6. 限界効果・予測確率・的中表（別メソッド方針、確定）

いずれも`fit()`のReturn本体には含めず、**Resultオブジェクトの別メソッド**として提供する。

- `marginal_effects(at="overall" | "mean" | "median", ...)`: 限界効果。デフォルトは`at="overall"`（AME、average marginal effects）。statsmodelsの`get_margeff(at='overall', ...)`に相当。標準誤差はデルタ法で計算し、`fit()`時の`cov_params`を再利用する（再最適化不要）。Return形式は`coef_table`と同じ行指向（`dydx` / `std_err` / `z` / `p_value` / `conf_low` / `conf_high`）
- `predict()`: 予測確率
- `pred_table()`: 分類の的中表（閾値依存のため、コアのReturnには含めない）

限界効果に関しては「見る/見ない」を切り替えるフラグは設けない（実証分析で見ないことは考えにくいため）。可変なのは`at`（どの代表点で見るか）のみ。

---

## 7. モデル固有オプション（確定・一部先送り）

| モデル | オプション | 状態 |
|---|---|---|
| Probit / Logit | `start_params: Option<Vec<f64>>`（初期値。`None`ならゼロベクトル、statsmodels方式） | 確定 |
| Tobit | 打ち切り方向（左/右/両側）・下限/上限値 | 確定 |
| Tobit | `dist`（誤差分布。`survival::survreg`はgaussian/logistic/extreme value等を選べる） | **Gaussian固定**。他分布はv1では対象外、将来拡張として保留 |
| 多項ロジット | 参照カテゴリ（`base_category`等） | 着手時に決定 |
| 順序ロジット/プロビット | 閾値パラメータ数 | オプション化しない。yのカテゴリ数（K個）から`K-1`個を自動導出する |
| 全般 | `weights`（頻度/分析重み）、`offset`（GLM系で一般的） | **見送り**。Phase2（Logit/Probit/Tobit）では不要。Phase6のIO手法（count model等）で必要になった時点で追加検討 |

---

## 8. 他パッケージの実装調査（アーキテクチャの参考）

### アーキテクチャの参考: R `maxLik`パッケージ

- モデル別実装ではなく**汎用MLEエンジンとして設計**されたパッケージ。`censReg`（Tobit）、`mlogit`系（多項ロジット）等がこれをエンジン層として利用しており、「共通engine + モデル固有のloglike/gradient/hessian」という今回作りたい構造そのものに近い
- **2層構造**: 最適化層（NR/BHHH/BFGS/NM/SANNを統一インターフェースで切替）＋ MLE専用の便利層（標準誤差抽出、最終Hessian計算方法の選択）
- **`finalHessian`パターン**: 「最適化に使ったソルバー」と「標準誤差算出に使う情報行列の種類」を分離できる設計。engine内のMLE共通の型・構造を検討する際（10章の未決定事項）に参考にする

---

## 9. リファレンス実装・テスト比較ライブラリ（確定）

| モデル | 主リファレンス | 交差検証 |
|---|---|---|
| Logit / Probit | statsmodels | R `glm()` |
| Tobit | R `AER::tobit`（`survival::survreg`エンジン） | R `censReg`（`maxLik`エンジン）。`survreg`と`maxLik`は最適化実装が完全に独立しているため交差検証として組み合わせる価値が高い |
| 多項ロジット | R `nnet::multinom`、statsmodels `MNLogit` | ― |
| 順序ロジット/プロビット | R `MASS::polr` | ― |

- Python製の`py4etrics`（PyPI公開、Tobit/Truncreg/Heckit/probitを実装）も候補に上がったが、教材付随パッケージでCRANのような査読・保守体制がないため、**censRegを本命、py4etricsは余力があれば追加で見る程度の位置づけ**とする。

---

## 10. 未決定・次に詰めるべき論点

- [ ] engine内のMLE共通の型・構造（`CostFunction`/`Gradient`/`Hessian`トレイトの実装パターン、観測ごとスコアの扱いを含む）の具体設計（8章のmaxLikアーキテクチャを参考にする）
- [ ] 収束判定の具体的な閾値（勾配ノルム等の`tol`のデフォルト値）
- [ ] モデル固有の尤度・勾配・Hessian導出（Probit/Logit/Tobitそれぞれの実装ノート。手法着手時に`*-implementation-notes.md`として作成）
- [ ] 多項・順序モデルの参照カテゴリ等の詳細仕様（着手時に決定）
- [ ] Issue分解の方針・粒度（最終ゴール。Probitだけで15〜20 Issueを想定）
