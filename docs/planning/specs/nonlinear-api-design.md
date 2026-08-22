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
- `"classical"` / `"nonrobust"`のエイリアス化はOLSの既存実装（[`engine_pybind/src/linear/ols.rs:102`](https://github.com/masahiroyecon1997dev/econometricsmodels/blob/main/engine_pybind/src/linear/ols.rs#L102)）に倣った

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

**Tobitはこの共通コアから2点を意図的に外す（確定、ユーザー確認済み）**:

- `log_likelihood_null` / `pseudo_r_squared`: 実装しない。Logit/Probitの`llnull`は`n1 ln ȳ + n0 ln(1-ȳ)`の閉形式解だが、Tobit（打ち切り回帰）のintercept-onlyモデルはΦ・φを含む非線形方程式になり閉形式が存在しない。主リファレンスのR`AER::tobit`（`summary.tobit`のソース確認済み）自体もpseudo R2を実装していない。不正確な指標を独自に定義するより、参照実装に無いものは実装しない方針を優先する
- `lr_statistic` / `lr_p_value`: **Wald検定に置き換える**（`AER::tobit`の`summary.tobit`が採用する方式に合わせる。切片以外の係数が同時にゼロという帰無仮説を`cov_params`から直接計算でき、`llnull`のための追加最適化が不要）。尤度比検定（LR）の方が計量経済学の実務では好まれる場面もあるため、**v1では見送りTODOとして残し、`llnull`の再最適化実装込みで将来追加を検討する**（10章参照）

---

## 6. 限界効果・予測確率・的中表（別メソッド方針、確定）

いずれも`fit()`のReturn本体には含めず、**Resultオブジェクトの別メソッド**として提供する。

- `marginal_effects(at="overall" | "mean" | "median", ...)`: 限界効果。デフォルトは`at="overall"`（AME、average marginal effects）。statsmodelsの`get_margeff(at='overall', ...)`に相当。標準誤差はデルタ法で計算し、`fit()`時の`cov_params`を再利用する（再最適化不要）。Return形式は`coef_table`と同じ行指向（`dydx` / `std_err` / `z` / `p_value` / `conf_low` / `conf_high`）
- `predict()`: 予測確率（Logit/Probit）
- `pred_table()`: 分類の的中表（閾値依存のため、コアのReturnには含めない。Logit/Probitのみ）

限界効果に関しては「見る/見ない」を切り替えるフラグは設けない（実証分析で見ないことは考えにくいため）。可変なのは`at`（どの代表点で見るか）のみ。

**Tobitは`predict()`/`marginal_effects()`/`pred_table()`のいずれも独自の形になる（確定、ユーザー確認済み）**:

- `predict()`: Tobitは打ち切りの有無で意味の異なる複数の予測量が定義される（McDonald-Moffitt 1980）。`E[y*|x]=x'β`（潜在変数の期待値・線形予測）、`E[y|x]`（打ち切りを考慮した観測値の期待値）、`P(uncensored|x)=Φ(z)`の3種を提供し、**デフォルトは`E[y|x]`**（実測`y`と直接比較できる量のため）とする
- `marginal_effects()`: Logit/Probitと同じ`dydx_and_jacobian`型の共通化はしない（独自実装。`∂E[y*|x]/∂xⱼ=βⱼ`・`∂E[y|x]/∂xⱼ=Φ(z)βⱼ`・`∂P(uncensored|x)/∂xⱼ=φ(z)βⱼ/σ`と対象ごとに式が異なるため。Issue #211の結論として記録）
- `pred_table()`: 廃止し、打ち切り予測の適合度チェック（観測された打ち切り比率 vs モデル含意の平均`Φ(z)`）に置き換える。具体的な出力形式は実装Issue着手時に決定する

---

## 7. モデル固有オプション（確定・一部先送り）

| モデル | オプション | 状態 |
|---|---|---|
| Probit / Logit | `start_params: Option<Vec<f64>>`（初期値。`None`ならゼロベクトル、statsmodels方式） | 確定 |
| Tobit | 打ち切り方向（左/右/両側）・下限/上限値 | 確定（詳細下記） |
| Tobit | `dist`（誤差分布。`survival::survreg`はgaussian/logistic/extreme value等を選べる） | **Gaussian固定**。他分布はv1では対象外、将来拡張として保留 |
| 多項ロジット | 参照カテゴリ（`base_category`等） | 着手時に決定 |
| 順序ロジット/プロビット | 閾値パラメータ数 | オプション化しない。yのカテゴリ数（K個）から`K-1`個を自動導出する |
| 全般 | `weights`（頻度/分析重み）、`offset`（GLM系で一般的） | **見送り**。Phase2（Logit/Probit/Tobit）では不要。Phase6のIO手法（count model等）で必要になった時点で追加検討 |

**Tobitの打ち切り境界オプション（確定、ユーザー確認済み）**:

- `TobitOptions`に`lower: Option<f64>`（既定値`Some(0.0)`）・`upper: Option<f64>`（既定値`None`）を追加。フィールド名は既存の`MleError::InvalidCensoringBounds { lower, upper }`（`engine/src/nonlinear/common.rs`に定義済み）に揃える
- `None`は「その方向は打ち切りなし」を意味する（左のみ打ち切り＝標準的なTobit、右のみ＝右打ち切り、両方`Some`＝両側打ち切り）。デフォルトが`lower=Some(0.0)`なのは標準的なTobit（左打ち切り0）に合わせるためだが、右打ち切りのみのモデルにしたい場合は明示的に`lower=None`を指定できるようにする（Pythonのキーワード引数デフォルトと明示的`None`渡しの区別を利用）
- バリデーション: 両方`None`はエラー（`InvalidCensoringBounds`）。両方`Some`のとき`lower >= upper`もエラー（同バリアント）
- **`y`の実測値と境界の整合性検証を追加**（新規、OLSの`n<=k`等engine層バリデーションと同じ位置づけ）: `lower`指定時に`y`が`lower`未満の値を含む、または`upper`指定時に`y`が`upper`超の値を含む場合はエラーとする。既存の`InvalidCensoringBounds`（境界設定自体の不正）とは意味が異なるため、**別のエラーバリアントを新設する**（例: `YOutOfCensoringBounds { row: usize, value: f64 }`、具体名は実装Issue着手時に確定）。パフォーマンス影響はO(n)の追加スキャン1回のみで、既存の`extract_f64_column`のNaN/無限大チェック（同じくO(n)）と同オーダーのため無視できる

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

- [x] engine内のMLE共通の型・構造 → Logit/Probit実装により確定・実装済み（`nonlinear-implementation-notes.md`参照）
- [x] 収束判定の具体的な閾値 → `tol=1e-6`で確定（同上）
- [x] Tobitの打ち切り境界API・GOF指標・predict/marginal_effects/pred_table方針 → 本ファイル5〜7章に確定事項として記載済み
- [ ] モデル固有の尤度・勾配・Hessian導出（Tobit）: 標準的な打ち切り正規回帰の尤度で導出可能（左右打ち切りは分布関数、非打ち切りは密度関数）と方向性は確認済みだが、実際の閉形式の書き下し・実装は着手時に行う。内部最適化パラメータ化は`(β, log σ)`とする方針（Olsen(1978)の`(β/σ, 1/σ)`変換による大域凹性の保証までは採用しない。ゼロベクトル初期値からのNewton収束はLogit/Probitで実績があり、Tobitでも同様の運用でまず試す）
- [x] Tobitの分離相当の病理ケース → 2種類の異なる退化パターンが存在することが判明し、いずれも対応済み（Issue #223・#226、詳細は`nonlinear-implementation-notes.md`「Tobit固有の病理ケース」参照）。(1) 非打ち切り観測ゼロによる`σ→0`退化は既存の`SEPARATION_PARAM_NORM_THRESHOLD`（標準化パラメータノルム基準）では捕捉できないため`MleError::NoUncensoredObservations`という専用の入力バリデーションを新設（Issue #223）。(2) 極端な`β`による分離は`run_solver`共有の既存`SeparationSuspected`機構がLogit/Probitと同じ閾値でそのまま捕捉できることを実測確認済み（Issue #226、python_packageラッパーのAPI境界テストで再現）
- [ ] Tobitのテスト許容誤差（`RTOL`）: `AER::tobit`（survreg、独立実装）との比較のため、Logit/Probitのstatsmodels比較（`RTOL=1e-8`）をそのまま踏襲できるかは未検証。実装・テスト作成段階で実測してから決定する
- [ ] Tobitの将来拡張候補（v1では見送り、backlog）: 尤度比検定（LR statistic/p-value）の追加実装
- [ ] 多項・順序モデルの参照カテゴリ等の詳細仕様（着手時に決定）
