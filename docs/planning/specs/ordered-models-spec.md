# OrderedLogit / OrderedProbit 設計ノート（実装途中）

順序ロジット・順序プロビット（潜在変数＋閾値モデル、最尤推定）の仕様を詰めるための設計ノート。
実装完了後は`docs/spec/`へ集約する（共通部分を`nonlinear-common.md`へ、リンク固有部分を
`ordered-logit-spec.md`・`ordered-probit-spec.md`へ。経緯は削除し理由のみ残す）。

状態の凡例: **未決**（ユーザー確認待ち）／**確定**。推奨案は調査に基づく提案であり、確定ではない
（CLAUDE.md 14章）。

## 0. モデルの定義（前提・確定）

`y*_i = x_i'β + ε_i`、`y_i = j ⇔ τ_{j-1} < y*_i ≤ τ_j`（`τ_0=-∞`、`τ_K=+∞`、`τ_1<…<τ_{K-1}`）。
`P(y_i=j|x_i) = F(τ_j - x_i'β) - F(τ_{j-1} - x_i'β)`。`F`はロジスティックCDF（OrderedLogit）または
標準正規CDF（OrderedProbit）。切片は閾値と同時に識別できないので`include_intercept`は提供しない。
閾値の数は`K-1`で、`y`のカテゴリ数`K`から自動導出する（オプション化しない）。
対数尤度は`(β, τ)`について大域的に凹（`τ`が単調増加の領域上）。

## 1. 論点一覧

### A. スコープ・依存関係・進め方

| # | 論点 | 選択肢 | 推奨案 | 状態 |
|---|---|---|---|---|
| A1 | 共通基盤が前提とする`validate_categorical_y`（多項ロジット向け）は**未実装**（多項ロジットは未着手） | (a) 順序モデル側で先に実装し、パラメータ化（最小/最大カテゴリ数）して多項ロジットが後から再利用する／(b) 多項ロジットを先に実装する | (a)。順序モデルを先行させる。上限が多項ロジット10・順序30と異なるので、上限を引数にした形で作る | **確定**: `nonlinear/common/validation.rs`（`validate_binary_y`・`validate_outcome_not_constant`と同じ場所）に置く。engineには`y`が`f64`で届くので整数性もここで検査する |
| A2 | 多項ロジットを同じリリース（v0.11.0）に含めるか | 含める／含めない | 本ノートの範囲外（順序モデルのみ） | 未決 |
| A3 | 実装順序 | OrderedLogit→OrderedProbit（Issueの推奨）／共通エンジンを先に作り両方を同時に通す | OrderedLogitを先に通し、共通化できる部分を抽出してからProbit（Issueどおり） | 未決 |
| A4 | リンク関数の共通化の単位 | (a) `Link`トレイト（または関数対）を取る1つの汎用エンジン＋薄い`ordered_logit.rs`/`ordered_probit.rs`ファサード／(b) Logit/Probitと同様に完全に別ディレクトリ（`ordered_logit/`・`ordered_probit/`）で、共通部分だけ`nonlinear/common/`へ | 既存のLogit/Probitは別ディレクトリだが、順序モデルは閾値の扱い・分離検出・限界効果が重く重複が大きい。**確定**: Logit/Probit式の別ディレクトリ（`ordered_logit/`・`ordered_probit/`）。Logit/Probitの汎用化は順序モデルと独立したリファクタとして別Issueで扱い、先行させない（二値モデルの`Fitted`・`fit`の骨格は順序モデルが再利用できない）。リンク非依存の部分（閾値のパラメータ化・確率からの尤度の骨格）は、OrderedLogitを通した後、Probitを書く段階で重複を見て`common/`へ抽出する |
| A5 | Pythonの公開形 | 別クラス`OrderedLogit`/`OrderedProbit`（Logit/Probitと同じ）／1クラス`OrderedModel(link=...)`（「系統内の手法ファミリは1クラス」の前例＝`estimator`オプション） | 別クラス。Logit/Probitが別クラスの前例であり、リンクの違いは「推定量の違い」ではなくモデルの違いなので | **確定**: 別クラス`OrderedLogit`/`OrderedProbit` |
| A6 | 設計ノート・仕様書の単位 | 1ファイル（本ノート）／手法ごとに2ファイル | 計画中は本ノート1ファイル。集約時に共通→`nonlinear-common.md`、固有→2ファイル | 未決 |
| A7 | ブランチ | `release/v0.11.0`を`release/v0.10.0`のHEADから作成済み（worktree: `/workspaces/econometricsmodels-v0.11.0`）。`release/v0.10.0`のリリース作業で`main`/`dev`に入る変更の取り込み方 | v0.10.0のリリース完了後に`git merge`で取り込む（リベースしない） | 未決 |

### B. 入力（`y`・`x`）と検証

| # | 論点 | 選択肢 | 推奨案 | 状態 |
|---|---|---|---|---|
| B1 | `y`で受け付けるdtypeと順序の与え方 | (a)数値で値が整数のもの。順序は整数コードの大小／(b)`Enum`・`Categorical`の順序も使う | (a)。`accepted-data.md`の数値の役割のdtype（整数・浮動小数点・`Boolean`・`Decimal`）をそのまま使い、値が整数かはengineが検査する。`Boolean`は2カテゴリなのでカテゴリ数の検査で弾かれる。`Enum`対応は将来拡張。11章 | **確定**: 11章 |
| B2 | コードの開始値 | 任意の開始値を許す／`0`始まり固定 | 任意（負の値も可）。連続であることだけを要求し、結果に`categories`を持つ。11章 | **確定**: 11章 |
| B3 | カテゴリ数`K`の範囲 | 3以上30以下（Issue）。`K=2`は`Logit`/`Probit`へ誘導 | 3以上30以下。新しい`code`は1つ（下限・上限を分けない規約）で、`hint`で`K=2`の誘導と30超の統合を両方示す。値が1種類（定数の`y`）は手法横断の`OUTCOME_CONSTANT`に任せる。11章 | **確定**: 11章 |
| B4 | 値が整数でない`y`・欠番の扱いとエラー | 新しい`code`を足す | `OUTCOME_NOT_INTEGER`（最初の行の三つ組＋`row`・`value`）と`OUTCOME_CATEGORIES_NOT_CONSECUTIVE`（三つ組＋欠けた最小のコード・最小・最大）。欠番を自動で詰めない（空のカテゴリを隠すため）。11章 | **確定**: 11章 |
| B5 | 観測が少ないカテゴリの扱い | 欠番（観測0）はエラー。観測1件のカテゴリ等を弾くか | 欠番だけをエラーにし、少数セルは弾かない（分離・非収束としてC5の検出に任せる）。`sparse_category`・`beauty`で検証（F4） | **確定**: 11章 |
| B6 | `x`の定数列 | (a)ランクチェックに任せる（`DESIGN_MATRIX_SINGULAR`、`ComputationError`）／(b)専用の`code`（`ValidationError`）で先に弾く／(c)FEの`REGRESSOR_CONSTANT`の文面を一般化して再利用 | **(b)**。入力だけで分かる（3.1章の分類）、statsmodelsのように定数列を足す利用者の誤りが多い、FEの前例（固定効果が吸収する定数列を専用の`code`で先に弾く）と同じ構造。既存の`REGRESSOR_CONSTANT`は文面がFE専用（固定効果と共線）で、`params`に英語の句を入れられないので流用できない。ダミー変数の罠などの残りの共線性は`DESIGN_MATRIX_SINGULAR`（暗黙の定数を除いた列を報告）。11章 | **確定**: 11章 |
| B7 | `x`が空（`x=[]`）のモデル（閾値のみ） | 許す／`NoRegressors`で弾く（Logit/Probitの`k==0`と同じ） | 許す案: 帰無モデルそのもので、閉形式（周辺累積比率）で解ける。ただし`df_model=0`でLR検定・擬似R²が定義できず、結果の組み立て・限界効果が特殊になる。**弾く**案が実装は単純 | **確定**: 弾く。Logit/Probit/OLSは既に`x=[]`を`ValidationError`で弾いており（`x must contain at least one column name`）、それに揃える。帰無モデルの対数尤度は内部で閉形式で使う |
| B8 | 検証の順序 | `fit()`冒頭の検証順 | 11章の順序（設定値 → `y`の形 → 観測数・クラスター数 → 定数の`y` → 定数列 → 特異行列 → 最適化 → 分離）。`test_validation_order.py`の`FAULTS`に足す | **確定**: 11章 |
| B9 | 観測数の下限 | `n <= k + (K-1)`（総パラメータ数）で`InsufficientObservations` | 既存の`INSUFFICIENT_OBSERVATIONS`を再利用（`n_params=k+K-1`、`min_obs=n_params+1`）。11章 | **確定**: 11章 |
| B10 | `y`の定数 | `OUTCOME_CONSTANT`かカテゴリ数のエラーか | `OUTCOME_CONSTANT`（`error-handling.md`5.4章の手法横断の表に揃える。閾値が切片の役割をするので「切片あり」の扱い）。カテゴリ数の検査は値が1種類のときは対象外にする。11章 | **確定**: 11章 |

### C. 推定（尤度・パラメータ化・最適化）

| # | 論点 | 選択肢 | 推奨案 | 状態 |
|---|---|---|---|---|
| C1 | 最適化のパラメータ化 | (a) 直接`τ`で最適化し、単調性は実行可能領域の外を`cost=+∞`で棄却／(b) 無制約化: `τ_1`と`log(τ_j-τ_{j-1})`（statsmodelsと同じ）／(c) 初期値を直接`τ`にしてLM付きNewtonで進める | **最重要の論点**。(b)は`bfgs`/`lbfgs`のline searchが素直だが、尤度が非凹になり、Tobitで踏んだ「鞍点での停滞収束」の罠が出る。(a)は大域凹性が使えるがline searchが棄却に弱い。**実験済み（4章）。推奨は(a)直接`τ`で最適化**: Newtonは棄却ではなく「境界までの割合」で`τ`の順序を保つ。`log`増分は非凹になり悪い初期値で失敗が増える（`log`増分は尤度の曲率が壊れ、Hessianが不定になる） | **確定**: 直接`τ`で最適化。準ニュートン法（`bfgs`/`lbfgs`）は`stpmax`で順序を保つ（必須）。Newtonの境界までの割合は任意（悪い初期値で評価回数を減らす最適化）。詳細は4章 |
| C2 | 標準化 | 切片が無いので、平均を引く標準化は`β̃=s⊙β`・`τ̃=τ+x̄'β`と閾値の平行移動で**厳密に表せる**。引くか、スケールだけにするか | 平均も引く（切片ありのLogit/Probitと同じ理由で条件数が下がる）。逆変換と共分散のヤコビアンを`common/standardize.rs`に足す | 未決 |
| C3 | 初期値 | `β=0`、`τ_j=F⁻¹(累積比率)`（周辺分布からの閉形式。nullモデルの厳密解） | 左のとおり。`ols_based_initial_params`のランクチェック（切片前提）は流用できないので、ランク落ちの検出を別途どこで行うか（中心化した`x_std`の列ピボットQR）を決める | 未決 |
| C4 | 確率の数値安定化 | `P_j=F(a)-F(b)`は裾でほぼ等しい2項の引き算になり桁落ちし、深い裾では`P_j`がアンダーフローする。(A)対数空間＋対称性による反転／(B)反転のみで倍精度領域の差を取り、アンダーフローは`cost=+∞`で棄却／(C)Probitの`U_CLAMP`型クランプ | **推奨は(A)**。(C)は`a`・`b`が同じ裾で両方クランプされると`F(a)-F(b)=0`になるので使えない。式・実測・Rustに足す部品は5章 | **確定**: (A)。`shared::normal`への`ln_normal_cdf`の追加も含む |
| C5 | 完全分離・準完全分離（MLEが存在しない場合） | (a) v1は非収束（`NOT_CONVERGED`）や巨大な閾値に任せる／(b) 既存のLP（`separation/lp.rs`）を拡張: 全`i`に`c_{y_i-1} ≤ d'x_i ≤ c_{y_i}`（`d≠0`）を満たす`(d,c)`が存在するかを判定 | 標準化ノルムの閾値方式は廃止済み（偽陽性・偽陰性がある）なので(a)で黙って誤った結果を返さないことが条件。**実験済み（6章）。推奨は(b)をv1に含める**: 分離の条件は`η_{y_i-1} ≤ x_i'δ ≤ η_{y_i}`を満たす`(δ,η)≠0`の存在で、既存のLPと同じ錐の形。(a)だと分離データで収束フラグが立ったまま返る（小標本の600件×2リンクで分離203件が全件「収束」） | **確定**: (b)をv1に含める（6章） |
| C6 | ソルバー | `newton`/`bfgs`/`lbfgs`の3種（既存の`run_solver`を共有）。`tol`の既定値（`newton`: `1e-6`、`bfgs`/`lbfgs`: `1e-8`）も共通 | 既存と同じ。C1の結果次第で`newton`の扱いを再検討 | 未決 |
| C7 | 直交座標（Tobitの`DesignCoordinates`）の採用 | Tobitは相関の高い傾きでのHessianの桁落ちを避けるために直交座標で最適化。順序モデルでも必要か | 標準化＋ウォームスタートで足りるかを実験で確認してから判断（最初は採用しない） | 未決 |
| C8 | `fit()`冒頭の共通処理 | `ensure_serial()`、`py.detach`、`guard`、`BudgetedProblem`（評価回数の枠）、`OutcomeConstant`相当 | 既存手法と同じ。`test_gil_release.py`の`FITS`・`CASES`・`ROWS`に足す | 確定 |

### D. 標準誤差・検定・適合度

| # | 論点 | 選択肢 | 推奨案 | 状態 |
|---|---|---|---|---|
| D1 | `cov_type` | `classical`（観測情報）・`opg`・`hc0`・`hc1`・`cluster`（Logit/Probitと同じ集合） | 同じ。スコアは最適化のパラメータ化に依らず、サンドイッチは再パラメータ化で不変（デルタ法で戻す）。`hc1`の補正`n/(n-k)`の`k`は総パラメータ数（`k_x+K-1`） | 未決 |
| D2 | クラスター数の条件 | Logit/Probitは`G <= q`（`q`は傾きの数）でエラー。順序モデルの`q`は`k_x`（閾値は傾きではないが、`β`ブロックの`q×q`部分行列が退化する条件は同じ） | `q=k_x`で`validate_cluster_count_covers_slopes`を再利用 | 未決 |
| D3 | 閾値の標準誤差 | 元の`τ`の尺度でデルタ法（C1(b)を採る場合）。`z`検定・信頼区間はパラメータ全体に共通（標準正規分布） | `params`の`τ_j`に対する`std_errors`・`test_stats`・`p_values`・`conf_int`を`β`と同じ表に載せる | 未決 |
| D4 | 全体検定 | LR検定（`lr_statistic`・`lr_p_value`、`df=k_x`）。帰無モデルは閾値のみで**入れ子**（Logit/Probitの切片なしと違い常に成り立つ）。`log_likelihood_null`は閉形式`Σ n_j log(n_j/n)` | 提供する。`pseudo_r_squared`（McFadden）・`aic`・`bic`も既存の`goodness_of_fit`を共有（`k`は総パラメータ数） | 未決 |
| D5 | `df_model`・`df_resid` | `df_model=k_x`、`df_resid=n-(k_x+K-1)` | 左のとおり | 未決 |
| D6 | 平行線（比例オッズ）仮定の検定（Brant検定等） | v1スコープ外（Issueで確定）。将来はオプトインの事後診断メソッド | 対象外。独立メソッドとして公開する方針（CLAUDE.md 2章）を`docs`の「Not implemented」に明記 | 確定 |

### E. 結果・API

| # | 論点 | 選択肢 | 推奨案 | 状態 |
|---|---|---|---|---|
| E1 | `params`等のキー名: 閾値の名前 | (a)`"1\|2"`（隣り合う2カテゴリの`y`のコード。`MASS::polr`と同じ）／(b)`"1/2"`（statsmodelsと同じ表記）／(c)`"cut1"`・`"cutpoint_1"`（連番）。`x`の列名との衝突（Tobitの`"sigma"`の前例）をどう扱うか | **推奨は(a)**。`polr`の`zeta`と値の意味が同じ（直接の`τ`）で、statsmodelsの`"1/2"`は`log`増分を指す名前なので表記だけ合わせると誤解を招く。衝突は`fit()`で専用の`code`により拒否する。7章 | **確定**: (a)`"1\|2"`、接頭辞なし。衝突は専用の`code`で拒否。構造は`categories`・`cutpoints`・`cutpoint_names`で公開（7章） |
| E2 | 閾値・カテゴリの公開面 | `params`とは別に閾値のリストを公開するか、名前は`cutpoints`か`thresholds`か。キーの文字列の分解に頼らずに使えるか | `categories: list[int]`（`y`のコード・昇順）・`cutpoints: list[float]`（`K-1`個。`cutpoints[j]`は`categories[j]`と`categories[j+1]`の間）・`cutpoint_names: list[str]`（`params`のうち閾値のキー）を公開し、閾値は`params`にも含める | **確定**: 左の3つ（E1の議論で決定）。`params_table()`の行にキーは足さない |
| E3 | 表の関数名 | Issueは`coef_table()`と書いているが、既存は**`params_table()`**（Logit/Probit/Tobit/IV） | `params_table()`に統一（Issueの記述は古い） | 未決 |
| E4 | `predict()`の返り値 | カテゴリ別確率は`n×K`。(P1)`list[list[float]]`（行ごと）／(P2)`dict[str, list[float]]`（カテゴリのコード→確率の列）／(P3)polars `DataFrame`／(P4)Tobit式の`target`切り替え | **推奨はP2、`target`は付けない**（8章）。CLAUDE.md 2章の公開形（キーがデータで決まる結果は`dict`、値は`list[float]`）の範囲に収まる | **確定**: P2（`dict[str, list[float]]`）、`target`は付けない。CLAUDE.md 2章の公開形の拡張として承認済み（8章） |
| E5 | `augment()` | カテゴリ別確率の列名と元の列との衝突検査 | **推奨**: 列は`predicted_probability_<コード>`（`K`列、昇順）。`K`列のどれかが元のデータにあれば既存の`AUGMENT_COLUMN_EXISTS`（最初に衝突した名前）。`new_data`は`predict()`と同じ意味（`y`は不要）。8章 | **確定**: 推奨どおり（8章） |
| E6 | `marginal_effects()` | **Issueのスコープに無い**。順序モデルでは限界効果が中心的な解釈量（`∂P(y=j)/∂x_k=[f(τ_{j-1}-xβ)-f(τ_j-xβ)]β_k`、端のカテゴリは片側）。Logit/Probitは提供済み | v1に含めるか、別タスクにするか。含める場合は`(変数, カテゴリ)`ごとの行、`at`（`overall`/`mean`/`median`）、デルタ法（`β`と`τ`のヤコビアン）、離散変数の自動判定はしない（既存方針） | **確定**: 実装する。独立したタスクとして起票済み（v0.11.0の対象。詳細設計は本ノートで詰める） |
| E7 | `pred_table()` | `K×K`の的中表（最大確率のカテゴリで分類）。Logit/Probitは提供 | **推奨は提供する**。`threshold`引数は無し。statsmodelsの`OrderedResults.pred_table()`で照合できる（実測）。8章 | **確定**: v1に含める（8章） |
| E8 | `*Options` | `cov_type`・`cluster`・`solver`・`max_iter`・`tol`・`raise_on_non_convergence`・`confidence_level`（`LogitOptions`から`include_intercept`を除いたもの）。全引数キーワード専用。`options`は末尾のキーワード専用引数。`start_params`・`weights`・`offset`は無し | 左のとおり。`api-conventions.md`の表に1行足し`test_call_signatures.py`が検査する | 未決 |
| E9 | pickle・`copy` | `*Fitted`（`β`・`τ`・共分散・カテゴリのコード・`x`の列名）を持つ。`from_parts`で`τ`が狭義単調増加・次元の整合を検査する | 既存パターンに従う。`FORMAT_VERSION`・`SAMPLES`への追加 | 確定 |

### F. テスト・ベンチマーク・性能

| # | 論点 | 選択肢 | 推奨案 | 状態 |
|---|---|---|---|---|
| F1 | 主リファレンスと交差検証 | 主=statsmodels`OrderedModel`（Issue）か主=R`MASS::polr`（`nonlinear-common.md`8章）か。statsmodelsの限界（`params`が`log`増分・既定`nm`・`hc1`の補正なし・限界効果なし）と、両者のHessianが数値微分である点 | **9章**。主=statsmodels（`newton`）、交差検証=R（`polr`＋`sandwich`）。statsmodelsに無い・合わない量（`hc1`・`opg`・限界効果）はRが主。SE等は参照実装間の実測の差から許容誤差を決める。自前のオラクルは参照にしない | **確定**: 9章の表のとおり |
| F2 | 比較する量 | `params`（`β`・`τ`）・`std_errors`・LR・対数尤度・AIC/BIC・カテゴリ別確率・（E6を含めるなら）限界効果。statsmodelsの`log`増分尺度の共分散はデルタ法で`τ`の尺度へ変換して比べる | 左のとおり | 未決 |
| F3 | 許容誤差 | 既定`RTOL=1e-8`。最適化器の許容誤差の差で`polr`は`~1e-6`ずれうる（Probitの`glm`の前例）。再fitで消す手順を使うか | 実測してから`tests/support/_tolerances_nonlinear.py`に例外を足す | 未決 |
| F4 | 合成データ | `generate_ordered_choice_dataset(scenario, link)`を`benchmark/nonlinear/datasets.py`に足す。2値モデルの共通シナリオに順序モデル固有のシナリオを足す | **10章**。共通11＋固有7シナリオと、`baseline`の派生（クラスター・信頼水準・`solver`・コードのずらし）。全シナリオで試作の生成関数の成立を確認済み | **確定**: 10章のとおり |
| F4' | 実データ | 順序の被説明変数を持つWooldridgeのデータ。候補を走査（118データセット） | **10章**。`happiness`（`happy`、K=3、n=16991、地域G=9でクラスター）と`beauty`（`looks`、K=5、両端が少数）を推奨。`beauty`に`belavg`を入れると実データの分離の例になる。`affairs`（`ratemarr`、K=5）は予備 | **確定**: 10章のとおり |
| F5 | テストの網羅性 | `testing-policy.md`の網羅性要件（`cov_type`×`solver`×切片なしの扱い・クラスター境界・信頼水準・検証順序・pickle・GIL・panic境界・エラー`code`）。`/test-new`と`/review-testing`で確認 | 左のとおり | 確定 |
| F6 | 性能比較 | `performance/compare_ordered_logit.py`・`compare_ordered_probit.py`。statsmodelsの最適化器（`bfgs`か`newton`）をどれにするか（`nm`は不公平）。`n`の掃引に加え`K`の掃引 | `solver`軸と`n_sweep_engine_only=(200_000, 1_000_000)`の大標本ガードを既存手法と同型に | 未決 |

### G. ドキュメント

| # | 論点 | 推奨案 | 状態 |
|---|---|---|---|
| G1 | 公開ページ | `usage/`・`api/`・`methods/`の各1ページをOrderedLogit・OrderedProbitそれぞれに（`methods/`は2ページか1ページか）。mkdocs navのNonlinear models配下。「Options at a glance」「Fields by purpose」「Options by purpose」を漏れなく | 未決 |
| G2 | 横断ガイドの更新 | `api-conventions.md`・`inference-conventions.md`の表に1行、`accepted-data.md`に`y`（順序コード）の行、`validation.md`に新しい検証の行、`verification.md`、`performance.md`、`error-codes.md`（生成）、`README.md` | 確定 |
| G3 | 数式の正本 | 尤度・スコア・Hessian・限界効果・確率の定義式は`docs/methods/`に置き、`docs/spec/`には理由のみ | 確定 |

## 2. 実装順序の案（確認後に確定する）

1. 共通基盤（カテゴリ検証・コード検証・新しいエラー`code`・標準化の拡張）
2. OrderedLogit: 尤度・スコア・Hessian → 点推定 → `cov_type` → 適合度 → カテゴリ別確率
3. OrderedLogit: `engine_pybind` → Pythonラッパー → テスト・ベンチマーク → ドキュメント → 性能
4. OrderedProbit: 共通化できる部分の抽出 → リンクの差し替え → 以降は3と同じ

## 3. 確認済みの事実（調査メモ）

- `nonlinear/common/validation.rs`に`validate_binary_y`はあるが`validate_categorical_y`は無い。
- `nonlinear/`には`logit`・`probit`・`tobit`・`separation`のみ（多項ロジットは未着手）。
- Logit/Probitの`fit()`が使う`ols_based_initial_params`・標準化（`standardize_columns`）は切片の存在を前提にしている。
- `nonlinear-common.md`7章は「順序モデルの閾値パラメータ数はオプション化せず`K-1`を自動導出」と既に決めている。
- 既存の表メソッドは`params_table()`（`coef_table()`は存在しない）。

## 4. C1の実験結果（閾値のパラメータ化）

**方法**: Pythonのプロトタイプ（numpy・scipy）。尤度・スコア・Hessianは解析式で、有限差分と`1e-9`、statsmodelsの対数尤度・最適点と9〜10桁で一致することを確認した。比べた方式は、直接`τ`（`θ=(β,τ)`、順序を破る点は`cost=+∞`）、直接`τ`＋境界までの割合（`direct_ftb`、Newtonのステップを`0.9·α_max`に縮めて順序を保つ）、`log`増分（`θ=(β,τ₁,log(τⱼ-τⱼ₋₁))`、statsmodelsと同じ）。ソルバーは、エンジン相当のNewton（LM梯子・補償和・`BelowCostResolution`）、BFGS、L-BFGS（scipyの`L-BFGS-B`と、warm start点の厳密な逆Hessianで前処理したもの）。`n=10⁵`（ストレスは`2·10⁴`）、`K=5`/`6`/`30`、`k=5`/`20`（相関0.9）、少数セル（0.05%）、強い効果、平均が大きい非中心化の`x`、logitとprobit。

**結果**:

1. **warm startからのNewtonは、どの方式も差が出ない**: 全シナリオで4〜6反復・評価回数4〜6で収束（`direct`・`direct_ftb`・`reparam`とも）。
2. **悪い初期値（`β~N(0,1)`・`N(0,3)`＋ランダムな`τ`、`τ`を±6に広げた初期値、各12回）で差が出る**:
   - `direct`: コストのHessianは常に正定値（不定0回）。全216回で成功。ただし棄却で梯子を上げるので評価回数が反復数の2〜4倍に膨らむ。
   - `direct_ftb`: 全216回で成功。評価回数が反復数とほぼ等しい（梯子を使わない）。**最良**。
   - `reparam`: **Hessianが不定になる反復が頻発する**（1つの条件の12回の実行の合計で最大13回）。失敗が出る（logit K5・K30、probit K30で12本中10〜11本が成功）。非凹性がTobitで踏んだ鞍点の罠と同じ形で出る。
   - （訂正）当初、probitの極端な初期値（`β~N(0,3)`、`k=20`）で全方式が失敗していたが、原因はプロトタイプがスコアの比を`P=exp(logP)`経由で作っていて、`P`が0にアンダーフローして`0/0`になったこと（初期点で`min logP≈-5000`）。比を`exp(log f - log P)`に直して再実験すると、`direct`・`direct_ftb`は全216回（3シナリオ×3初期値×12回×logit/probit）で成功、`reparam`は同じ条件で10〜11/12の失敗が残った（上の数字は再実験後のもの）。C4の5章を参照。
3. **L-BFGSは前処理しない場合に差が出る**: `K=30`で`direct`が102反復・`reparam`が26反復。ただし**warm start点の厳密な逆Hessianで初期化すると差が消える**（`direct`は4〜5反復、`reparam`は5〜7反復）。エンジンの`FaerBfgs`・`FaerLbfgs`は同じ初期化をしているので、この差は実装には持ち込まれない。直接`τ`ではコストのHessianが常に正定値なので、初期化が「正定値でなければ元に戻す」分岐に落ちることもない。
4. **BFGS**は両方式とも5〜13反復で、差は小さい。
5. **中心化**: 平均が大きい非中心化の`x`では、前処理なしのL-BFGSが`direct`で54〜75反復（中心化済みの`baseline`は15反復）。中心化は引き続き必要（C2）。

**決定（確定）**:

- 直接`τ`で最適化する。Newtonは、ステップが順序を破るときに`τ`の部分を境界までの割合（`0.9·α_max`）で縮めてよい（任意。LM梯子で全ステップを棄却せずに済み、悪い初期値で評価回数が減る。warm startでは境界がほとんど効かないので、無くても正しく動く）。
- BFGS・L-BFGSは、探索方向に沿った実行可能な最大ステップ幅`α_max`を計算し、line searchの上限`stpmax`（`MoreThuenteLineSearch::with_bounds(step_min, step_max)`）に`0.9·α_max`を渡す。**順序を破る点で`cost=+∞`を返す案は使えない**: argminのline search（0.11.0）は、補間（3次・2次）で`Inf`/`NaN`が出ると`Err`（`ConditionViolated`）を返す実装で、`+∞`の試行点があると後退ではなく`fit()`の失敗になる（scipyのline searchは`+∞`でも動いたが、プロトタイプの結果はエンジンに持ち込めない）。`α_max`の計算は`O(K)`。**なお、warm start点の最初のNewtonステップが順序を破ったデータは96件中0件**（`α_max=∞`）で、境界が効くのは悪い初期値とline searchの外挿に限られる。安全網として必須だが、頻度は低い。
- 標準誤差は、収束点の観測情報行列を直接`τ`で評価するので、デルタ法は要らない（statsmodelsとの比較ではstatsmodels側を元の`τ`の尺度に変換する）。
- 副産物: 補償和なしのコストは丸め誤差の床でNewtonを止める（`n=10⁵`で`|g|≈1e-3`以下）。順序モデルでも、既存の補償和と`BelowCostResolution`を使うこと。

**限界**: プロトタイプ（scipyの`L-BFGS-B`はエンジンの`FaerLbfgs`とは別物）、1つのseed（ストレステストは初期値だけ12通り）、`n=10⁶`と完全分離は未実施、境界までの割合の係数`0.9`は未調整。実験のスクリプトはscratchpad（`/tmp/claude-1000/.../scratchpad/c1/`）にあり、永続化していない。

## 5. C4の検討結果（確率の数値安定化）

**記法**: `a=τ_y-x'β`、`b=τ_{y-1}-x'β`（端のカテゴリは`a=+∞`または`b=-∞`）、`P=F(a)-F(b)`、`f=F'`、`A=f(a)/P`、`B=-f(b)/P`。対数尤度の観測`i`の項`ℓ=ln P`について、`∂ℓ/∂a=A`、`∂ℓ/∂b=B`、`∂²ℓ/∂a²=A·(f'/f)(a)-A²`、`∂²ℓ/∂b²=B·(f'/f)(b)-B²`、`∂²ℓ/∂a∂b=-AB`（`(f'/f)(x)`はlogitで`1-2F(x)`、probitで`-x`）。`(a,b)`は線形予測子と閾値のアフィン関数なので、`β`・`τ`への微分は`∂a/∂θ=(-x, e_y)`・`∂b/∂θ=(-x, e_{y-1})`の連鎖律で作る。

**方法**: 基準値は標準ライブラリの`decimal`（90桁。誤差関数は級数と連分数。scipyと15〜16桁で一致）。`a,b∈{±1000,±100,±38,±10,±3,±1,0}`の`a>b`の全組み合わせ、隣接閾値の接近（ギャップ`1e-3`〜`1e-9`）、端のカテゴリを合わせた119ケースで、logit・probitの3方式を比べた。(naive)差を直接取る。(flipnaive)`a+b>0`なら対称性で`P=F(-b)-F(-a)`に反転して倍精度領域で差を取る。(logspace)反転した上で`ln P = ln F(u)+log1mexp(ln F(v)-ln F(u))`（`log1mexp(x)=ln(1-e^x)`を`x>-ln2`で`ln(-expm1(x))`、それ以外で`ln1p(-exp(x))`）、スコアの比は`exp(ln f - ln P)`。

**結果**:

| 方式 | 非有限になったケース（logit／probit） | 有限なケースの精度 |
|---|---|---|
| naive | 10／19（正の側での`1-1=0`、深い裾） | 反転しないぶん正の側で桁落ち |
| flipnaive | 2／12（深い裾のアンダーフロー。probitは`x`の絶対値が約38超、logitは約745超） | logspaceと同等 |
| **logspace** | **0／0** | `A`・`B`の相対誤差は約`1e-14`以下。観測ごとの2×2のHessianの最小固有値／スケールは`-1.5e-17`以上（半正定値が丸めの範囲で保たれる） |

- **深い裾**: スコアの比を対数空間で作ると、probitで`z`が約100・確率が`e^-5000`の悪い初期値からでもNewtonが収束した（`P=exp(ln P)`経由だと初期点の勾配が`0/0`でNaN）。
- **ギャップ**: Hessianの相対誤差は約`ε/(カテゴリの確率質量)`で増える（ギャップ`1e-4`で約`1e-12`、`1e-6`で`1e-10`、`1e-8`で`1e-8`、`1e-10`で`1e-6`）。最適点では`n_j≥1`なので確率質量は`≳1/n`で、`n=10⁶`でも約`1e-10`に収まる。
- **クランプ**: `U_CLAMP`型のクランプは、`a`・`b`が同じ裾で両方クランプされると`F(a)-F(b)=0`になるので順序モデルには使えない（Probitの`λ=φ(u)/Φ(u)`のように片側だけを見る量とは性質が違う）。対数空間なら不要。

**決定（(A)で確定）**:

1. 式は「反転＋対数空間」。反転は`a+b>0`のとき（端は有限の側で判定）に`(u,v)=(-b,-a)`、それ以外は`(a,b)`。端のカテゴリは`ln F(v)=-∞`（`b=-∞`）または`ln F(u)=0`（`a=+∞`）として同じ式で扱う。
2. スコアは`A=exp(ln f(a)-ln P)`・`B=-exp(ln f(b)-ln P)`、Hessianは上の記法の式。`U_CLAMP`は使わない。
3. `ln P`が非有限（`F(a)-F(b)`が丸めで0になる）ときは、`cost=+∞`にして棄却する。勾配・Hessianは有限のコストの点でだけ評価する。Newtonは境界までの割合の余裕（`0.9`）で順序を保つので、反復点では起きない（line searchの試行点は、`stpmax`（4章）で常に実行可能にするので、argminに`+∞`を返さない。`+∞`が使えるのは自前のNewton（LM梯子）だけ）。
4. Rustに足す部品: `ln F`。logit: `x≥0`で`-ln1p(e^-x)`、`x<0`で`x-ln1p(e^x)`。probit: `shared::normal`に`ln_normal_cdf(x)`を足す（現状は`normal_cdf`・`normal_sf`のみで、`erfc`は約`z<-37.5`で0になる）。`x>-20`は`libm::erfc`（`x>0`では`ln1p(-½erfc(x/√2))`）、`x≤-20`は漸近展開`-x²/2-ln(-x)-½ln2π+ln(1-1/x²+3/x⁴-15/x⁶+…)`。継ぎ目（`-20`）の連続性と、130桁の参照値との比較を単体テストにする（`normal.rs`の既存の方針と同じ）。`f`・`ln f`は、logitは`ln F(x)+ln F(-x)`、probitは`-x²/2-½ln2π`。
5. v1で行わないこと: 小さなギャップ（確率質量が`1e-6`未満）の精度改善（logitは`sinh((a-b)/2)/(2cosh(a/2)cosh(b/2))`の恒等式、probitは級数）。最適点では上のとおり`≈1e-10`なので不要と判断した。

**限界**: 基準値は`decimal`の自前実装（scipyとの突き合わせ済み）、実験はPythonで、Rustの`libm::erfc`の精度（相対`z²·1e-16`）は反映していない。

## 6. C5の検討結果（完全分離・準完全分離）

**条件（導出）**: 対数尤度`ℓ(β,τ)=Σ ln[F(τ_{yᵢ}-xᵢ'β)-F(τ_{yᵢ-1}-xᵢ'β)]`は凹で、上に有界。方向`d=(δ,η)`に沿って観測`i`の確率が減らないのは、上端`τ_{yᵢ}-xᵢ'β`が下がらず、下端`τ_{yᵢ-1}-xᵢ'β`が上がらないとき（区間が広がるとき）だけで、どれかが厳密なら確率は真に増える。したがって、有限のMLEが存在しない必要十分条件は、次を満たす`(δ,η)≠0`の存在。

```text
η_{yᵢ-1} ≤ xᵢ'δ ≤ η_{yᵢ}     （全観測i。η₀=-∞、η_K=+∞）
```

言い換えると「線形結合`x'δ`で、カテゴリが弱い意味で順序どおりに並ぶ」。`K=2`では2値モデルの分離条件と一致する。各カテゴリに観測があれば、閾値の順序`η_1≤…≤η_{K-1}`は条件から導かれる（中間カテゴリの観測が両側から挟む）。`δ=0`なら`η=0`が強制されるので、自明な方向は無い。`x`の中心化後の列がランク落ちだと`x'δ`が定数になる非自明な解が出るので、ランクチェック（C3）を先に通すこと。

**LPの定式化（既存の`separation`と同じ形）**: 変数`(δ,η)`（`k+K-1`個、`|·|≤1`）。観測`i`（カテゴリ`m`）の行は`m<K`で`(-xᵢ, e_m)`、`m>1`で`(xᵢ, -e_{m-1})`（それぞれ単位長にそろえる。中間カテゴリは2行、端は1行）。制約は`行·(δ,η) ≥ 0`、目的は行の値の和の最大化、判定は**最大マージン**が`1e-6`超か。右辺は0の錐なので、`shared::linear_program`の`BoxBoundedLp`（`b=0`）にそのまま渡せる。

**実験**（Pythonプロトタイプ。LPはHiGHS）:

- **ランダムな小標本**（`n=10〜40`、`k=1〜2`、`K=3〜5`、連続とダミーの混在、効果の大きさ3水準）: logit 600件・probit 600件の全件で、LPの判定と実際のMLEの存在が一致した。「分離」と判定されたのはlogit 80件・probit 123件で、判定の根拠は、Newtonの停止点から**LPの方向へ進むと対数尤度がまだ増える**こと（最大点ではない）。「分離なし」の997件（logit 520件・probit 477件）は、`converged`で、Hessianの条件数が`1e10`未満。
- **分離データでNewtonの収束フラグは立つ**: 分離と判定された203件のすべてで、Newtonは勾配が`tol`未満になって「収束」した（停止点の`max|θ|`の中央値はlogit 24、probit 13。分離なしは約2）。LPが無ければ分離を無警告の成功として返す。**v1でLPを省く選択肢(a)は不可**。
- **境目**（`n=1500`、`K=3/5/12`、最上位カテゴリにだけ現れるダミー・最下位だけに現れるダミー・「最上位だけ」だが1観測を隣のカテゴリに足したもの）: 最大マージンは分離なしで厳密に0、分離ありで0.66以上と大きく離れ、1観測の違いで正しく反転する。閾値`1e-6`で十分に分離できる。分離データでNewtonは`|θ|≈25`で「収束」した。

**決定（確定）**:

1. v1に含める。`fit()`はソルバーの収束後に`verify_no_separation`と同様の判定を1回呼ぶ（Logit/Probitと同じ位置）。`converged=false`ならチェックせず、分離を検出したら`raise_on_non_convergence`で`SeparationDetected`か`converged=false`、LPが完了しなければ`SeparationCheckFailed`か`converged=false`（既存の契約）。
2. エラーの`code`は既存の`SEPARATION_DETECTED`・`SEPARATION_CHECK_FAILED`を再利用する（文面「結果が回帰変数のある組み合わせで完全に予測される（完全または準完全な分離）」は順序モデルでも成り立つ。同じ`code`はいつでも同じ`hint`）。
3. 行は、フィットと同じ中心化・標準化した`x`から都度計算する（複製しない）。中心化は錐を線形同型で写すだけで分離の有無を変えず（`η`が`c'δ`だけずれる）、`η`の尺度との釣り合いが良くなる。2値モデルの「平均は引かない」は閾値の較正を保つための選択で、順序モデルは較正し直す（実験は中心化したデータで行った）。
4. 回帰ガード: 上の「最上位だけ」「最下位だけ」「1観測の違い」と、ランダムな小標本でのLPと「停止点からの尤度の改善」の一致（600件）を`engine`の単体テストとpytestにする。再現データは`benchmark/nonlinear/datasets.py`に順序モデル版の`quasi_complete_separation`・`complete_separation`を足す（F4）。

**リスク・限界**:

- **自前の`DenseSimplex`の次元**: 変数は`k+K-1`個で、`K`が上限30・`k`が20なら約50。既存のテストは`k=30`まで。右辺0の錐は原点で強く退化し、`k≥10`・混合スケールで過去に破綻した（`docs/spec/logit-spec.md`3.2節「LPソルバーの差し替え」）ので、**最初の実装タスクで`lp.rs`の回帰テストを変数50個まで広げて確かめる**。足りなければ、既存のフォールバック（`SeparationCheckFailed`）で止まる。
  **確認済み（2026-11-28）**: 2値の回帰テストを`k=40・50`（切片込みで変数51個）、混合スケールを`k=36・48`まで広げて通った。順序モデルの行でも
  `k=20`・`K=30`（変数49個、ダミーを足して50個）で、分離なし・最上位だけのダミー・線形結合の順に30等分した`y`（最悪の形）を正しく判定する。
  最悪の形は作業集合が約800行・12〜13ラウンド（上限2000行の範囲内、releaseで1件約0.7秒）。分離なしのデータは約190行・3ラウンド。
- 実験のLPはHiGHSで、`DenseSimplex`ではない。マージンの閾値`1e-6`・摂動`1e-11`との相互作用は未検証。
- 大標本でのコストは未計測（行は端のカテゴリで1行、中間で2行になるので2値の最大2倍の走査）。2値の実測は`n=10⁶`で約68ms。
- 「分離なしで有限のMLEが存在する」ことは、中心化後の`x`が列フルランクのときに限る（C3）。

## 7. E1の検討結果（閾値のキー名）

**名前が現れる場所**: `params`・`std_errors`・`test_stats`・`p_values`・`conf_int`・`cov_params`のキーと`param_names`、`params_table()`の`param_name`。`x`の列名の後ろに閾値を`K-1`個並べる（`polr`・statsmodelsと同じ順序）。

**参照実装の命名（実測）**:

| 実装 | 名前 | 値の意味 |
|---|---|---|
| `MASS::polr` | `"0\|1"`, `"1\|2"`, `"2\|3"`（`y`が因子のラベルならそのラベル、例`"low\|mid"`） | 閾値`τ_j`そのもの |
| statsmodels `OrderedModel`（`y`がpandas） | `"0/1"`, `"1/2"`, `"2/3"` | `"0/1"`は`τ₁`、`"1/2"`以降は**`log(τ_j-τ_{j-1})`**（`τ_j`ではない）。`y`がnumpyだと名前自体が無い |

**候補の比較**:

| 候補 | 利点 | 欠点 |
|---|---|---|
| (a)`"1\|2"`（`y`のコードを`\|`で結ぶ） | `polr`の`zeta`と名前も値の意味も同じ。どのカテゴリの間の閾値かが名前で分かる。`K`によらず安定 | データ（コード）で決まるので衝突検査が要る。statsmodelsの`"1/2"`とは区切りが違う |
| (b)`"1/2"` | statsmodelsと表記が同じ | 同じ表記で**別の量**（`log`増分）を指すので、比較する利用者が取り違えやすい |
| (c)`"cut1"` | データによらず固定。衝突はほぼ起きない | カテゴリとの対応が名前から分からない。どの参照実装とも違う |

**決定（確定）**:

1. 名前は`"{下側のコード}|{上側のコード}"`（例: コードが`1..5`なら`"1|2"`〜`"4|5"`）。コードは検証済みの整数で、整数として書く（`1.0`ではなく`1`）。ラベルの付いた`y`は受け付けない（B1）ので`polr`のラベル名には対応しない。
2. `x`の列名が閾値の名前のどれかと一致するときは、`fit()`で`ValidationError`にする。新しい`code`を足す（候補`CUTPOINT_NAME_COLLISION`。`INTERCEPT_NAME_COLLISION`・`TOBIT_SIGMA_NAME_COLLISION`と同じ命名で、`params`は衝突した名前`name`、`hint`は「列の名前を変える」）。コードが分かるのは`y`の検証の後なので、検査はengineの入力の組み立て（`y`のコードの検証の直後）に置く。Tobitの`"sigma"`は静的なので`engine_pybind`の受け口にあるが、ここは静的に決まらない。
3. 閾値を`params`と同じ表（`std_errors`・`conf_int`・`cov_params`・`params_table()`）に載せる。`β`だけの辞書と閾値だけの辞書に分ける案は採らない（表の面がモデルごとに変わり、Tobitの`"sigma"`の前例とも揃わない）。
4. キーの文字列は分解して使わせない。閾値かどうかは名前からは判定できない（`K=3`で`x`に`"3|4"`という列があっても、閾値の名前と一致しなければ許される）ので、構造を別の面で公開する（E2）: `categories`（`y`のコード）・`cutpoints`（`cutpoints[j]`が`categories[j]`と`categories[j+1]`の間）・`cutpoint_names`（`params`のうち閾値のキー）。例: `{(lo, hi): c for lo, hi, c in zip(r.categories, r.categories[1:], r.cutpoints)}`。接頭辞つきの名前（`"cutpoint:1|2"`）は採らない。名前を作る処理は`engine`に置き（`param_names`の一部）、pickleの復元でも同じ名前になるよう`*Fitted`に名前を持たせる。
5. 文書化: statsmodelsとの違い（名前は同じ区切りでも値の意味が違う）を`docs/guide/inference-conventions.md`とAPIページに書く。ベンチマークのアダプタは名前ではなく位置で対応づけ、statsmodelsの`log`増分を直接の`τ`に直してから比べる（F2）。

**未決として残す点**: `predict()`・`augment()`の列名（コードを使う。E4/E5）。

## 8. E4・E5・E7の検討結果（`predict()`・`augment()`・`pred_table()`）

**既存の形（前例）**:

| | Logit/Probit | Tobit |
|---|---|---|
| `predict(*, new_data=None)` | `list[float]`（確率。観測順） | `+ target`（4種）。値は`list[float]` |
| `augment(*, new_data=None)` | 元データ＋`"predicted_probability"`（polars） | 元データ＋`"predicted_{target}"` |
| `pred_table(*, threshold=0.5)` | `[{"actual": 0, "predicted_0": n, "predicted_1": n}, …]` | 無し（`censoring_fit_check()`） |

`new_data`は、`fit()`の`x`と同名の列を名前で対応づける（`y`は不要）。`augment()`は元データ（または`new_data`）に列を足す。既存列と衝突すると`AUGMENT_COLUMN_EXISTS`。`predict()`の`list[float]`はCLAUDE.md 2章の公開形で、Series化は別に検討中（`n=10⁶`でメモリが約4倍、優先度は低い）。

**参照実装の実測**（statsmodels 0.15.0・R）:

| 実装 | 返り値 |
|---|---|
| statsmodels `predict()` | `ndarray (n, K)`。行和は1。`which="linpred"`（`x'β`、1次元）・`"cumprob"`（`P(y≤k)`）もある |
| R `polr` `predict(type="probs")` | `n×K`の行列（列名はレベル）。`type="class"`は因子 |
| statsmodels `OrderedResults.pred_table()` | **動く**。最大確率で分類した`K×K`（手計算の`argmax`のクロス集計と一致。ただし予測されなかったカテゴリの列は出ない） |
| statsmodels `get_margeff` | **無い**（順序モデルには未提供）。限界効果（E6）はRの`marginaleffects::avg_slopes(polr)`で出る（カテゴリごとの行）。statsmodelsが主リファレンスにならない |

**E4の選択肢**:

| 案 | 利点 | 欠点 |
|---|---|---|
| (P1)`list[list[float]]`（行ごと） | statsmodelsの配列に近い | `n`個のリスト。列の取り出しが面倒。大標本で重い |
| **(P2)`dict[str, list[float]]`**（キーはコードの文字列、値は確率の列） | CLAUDE.md 2章の公開形のまま（データで決まるキーは`dict`、値は`list[float]`）。`pl.DataFrame(res)`で表にできる。JSON化しやすい。Series化（別検討）の対象に素直に入る | 行ごとの取り出しは`zip`が要る |
| (P3)polars `DataFrame` | polarsのAPIと揃う。コピーが少ない | 2章の公開形から外れる（承認が要る）。他の手法の`predict()`と型が違う |
| (P4)`target`切り替え | Tobitと同じ形 | 返り値の形が`target`で変わる（確率は`K`列、クラスは1列）。型が不安定 |

**決定（確定）**:

1. **`predict(*, new_data=None) -> dict[str, list[float]]`**。キーは`categories`の`str`（`"1"`〜`"K"`、昇順）、値は観測順の確率。各行の和は1。`new_data`の意味はLogit/Probitと同じ（カテゴリは学習時の`K`個に固定）。
2. **`target`は付けない**。確率だけ（Logit/Probitと同じ範囲）。クラス・線形予測子・累積確率が要るときは、`target`で形を変えず、別メソッド（`predict_class()`等）として足す（足すのは非破壊）。確率からの導出は容易（`argmax`・累積和）で、線形予測子は`params`と`x`から計算できる。
3. **`augment(*, new_data=None) -> pl.DataFrame`**。元データに`predicted_probability_<コード>`を`K`列（Float64、昇順）足す。名前の前置き`predicted_`はOLS/Logit/Probit/Tobitと揃える。`K`列のどれかが元のデータにあれば、既存の`AUGMENT_COLUMN_EXISTS`（`column`は最初に衝突した名前）。`predict()`の辞書のキーと`augment()`の列名の対応は`predicted_probability_`＋キー。
4. **`pred_table() -> list[dict[str, float]]`**（引数なし）。`[{"actual": 1, "predicted_1": n, …, "predicted_K": n}, …]`（Logit/Probitの形に合わせ、`actual`はコード、`predicted_<コード>`は観測数）。分類は最大確率のカテゴリで、同点は小さいコード。全カテゴリに観測があるので`K`行`K`列が必ず揃う。学習データのみ（`new_data`は取らない）。注: 最大確率での分類は、中間のカテゴリや小さいカテゴリをほとんど予測しない（実測の例: 4カテゴリのうち1つが2000件中5件）。docstringに書く。
5. **テスト**: `predict()`は行和が1、statsmodelsの`predict()`（`n×K`）と`RTOL=1e-8`、`new_data`で学習データと同じ結果。`pred_table()`はstatsmodelsの`pred_table()`を`K×K`にreindexして整数で一致。`augment()`は構造（列名・順序・衝突）。ベンチマークのアダプタは、statsmodelsの`predict()`の列をコードに対応づける。
6. **pickle・`copy`**: `*Fitted`が`categories`と閾値・係数・共分散を持つ（列名が復元でも同じになる）。

**限界・注意**: `n×K`の辞書は`K`本の`list[float]`なので、`n=10⁶`・`K=5`でメモリは`predict()`5回分（Series化の検討と同じ課題。`K`の上限30なら30本）。この形は、2章の公開形（`predict()`は`list[float]`）の「データで決まるキーは`dict`」を拡張して使うもので、ユーザーの承認を得た（返り値が`list[float]`ではなくなるため）。実装時にCLAUDE.md 2章・`docs/spec/inference-conventions.md`6章の記述を、順序モデルの`predict()`を含む形に更新する。

## 9. F1の最終案（リファレンスと許容誤差）

**実測**（合成データ`n=2000`・`K=4`・`k=3`・40クラスター、logitとprobit。本実装の式はPythonプロトタイプの解析スコア・解析Hessianで計算）:

| `cov_type` | 本実装の式 | statsmodels（`newton`）との最大相対差 | R（`polr`＋`sandwich`）の対応する呼び出しと最大相対差 |
|---|---|---|---|
| `classical` | `H⁻¹` | 1.7e-7 | `vcov(polr)`: 1.6e-7 |
| `hc0` | `H⁻¹(ΣsᵢsᵢT)H⁻¹` | 3.5e-7 | `sandwich()`: 3.4e-7 |
| `hc1` | `hc0`×`n/(n-k)`（`k`は総パラメータ数） | **1.5e-3**（statsmodelsの`HC1`は`HC0`と同値で補正が無い） | `sandwich(adjust=TRUE)`: 3.4e-7 |
| `opg` | `(ΣsᵢsᵢT)⁻¹` | 無し（`cov_type`に無い） | `vcovOPG()`: **5.9e-10** |
| `cluster` | `G/(G-1)·(n-1)/(n-k)·H⁻¹(ΣS_gS_gT)H⁻¹` | 4.4e-7（**補正式も一致**） | `vcovCL(type="HC1", cadjust=TRUE)`: 4.1e-7（`type="HC0"`は`(n-1)/(n-k)`を含まず1.3e-3ずれる） |

- 約1e-7の差は、statsmodels・`polr`ともにHessianが数値微分であることによる（statsmodelsと`polr`の`classical`の差も2.7e-7）。Hessianを使わない`opg`は1e-9以下で一致する。
- **訂正**: 1章F1の調査時に「`cluster`の補正はstatsmodelsと異なり、Rの`cadjust=TRUE`が本実装と同じ」と書いたのは誤り。statsmodelsの`cluster`は本実装と同じ式で、Rは`type="HC1"`を指定したときに一致する。
- 点推定・対数尤度は、statsmodels（`newton`）・`polr`（`reltol=1e-15`）とも約1e-9で一致（1章F1）。

**決定（確定）**:

| 比べる量 | 主リファレンス | 交差検証 | 許容誤差 |
|---|---|---|---|
| `params`（`β`・`τ`）・対数尤度・LR・AIC/BIC・擬似R² | statsmodels（`method="newton"`。既定の`nm`は使わない） | `polr`（`reltol`を締める） | `RTOL=1e-8`（既存の基本） |
| カテゴリ別確率（`predict()`）・`pred_table()` | statsmodels（`predict()`・`pred_table()`） | `polr`の`predict(type="probs")` | 確率は`1e-8`、`pred_table()`は整数で完全一致 |
| SE・`z`・p値・信頼区間（`classical`・`hc0`・`cluster`） | statsmodels | `polr`＋`sandwich`（`cluster`は`vcovCL(type="HC1", cadjust=TRUE)`） | 参照実装間の実測の差から決める（今回の1データで最大4.4e-7。全シナリオで測り、最大値に余裕を持たせた名前付きの値を`tests/support/_tolerances_nonlinear.py`に置く） |
| SE等（`hc1`） | R`sandwich(adjust=TRUE)`（statsmodelsは補正が無いので使わない） | 無し（`hc0`×`n/(n-k)`の関係をstatsmodelsの`hc0`で確かめる） | 同上 |
| SE等（`opg`） | R`vcovOPG()` | 無し | Hessianを使わないので`1e-8`級で比べられる見込み（全シナリオで確認） |
| 限界効果（E6） | R`marginaleffects::avg_slopes(polr)`（statsmodelsに`get_margeff`が無い） | 無し（カテゴリ方向の和が0の不変条件を併用） | E6の設計で決める |

- 閾値は名前ではなく位置で対応づける。statsmodelsの閾値（`τ₁`と`log`増分）と共分散は、ベンチマークのアダプタで直接の`τ`にデルタ法で直してから比べる。
- **自前実装のオラクルは参照にしない**（ユーザーの方針: 実装者と同じ理解を共有するので客観性が弱い）。使うのは、参照実装同士が食い違ったときにどちらが真値に近いかを確かめる補助に限る。
- Probitの前例（Rの`glm()`の既定`vcov`が非正準リンクで期待情報行列になるため、R側で観測情報行列を解析式で組んだ）は、順序モデルでは不要。`polr`の`vcov`は最適化の数値Hessian（観測情報行列）なので種類は一致し、差は数値微分の誤差だけ。
- `nonlinear-common.md`8章（順序モデルの主リファレンスを`MASS::polr`としている）は、`docs/spec/`への集約時にこの表に合わせて直す。

## 10. F4の検討結果（合成データと実データ）

### 10.1 合成データ

`benchmark/nonlinear/datasets.py`に`generate_ordered_choice_dataset(scenario, link, n=500, k=3, K=4, seed)`を足す。潜在変数`y*=Xβ+ε`（`ε`はロジスティックまたは標準正規）を閾値で切って`y=1..K`を作る。2値モデル（`generate_binary_choice_dataset`）と同じく`tests/fixtures/benchmarks/data/`にCSVで固定し、テストはCSVを読む。閾値は真の値として固定する（真の係数と比べられるように）。

**試作の生成関数で成立を確認した**（seed 0〜19、logitとprobit。LPで分離が無いこと、statsmodels（`newton`）が収束すること）: `baseline`・`small_n`・`k_min`・`k_max`・`sparse_category`・`near_separation`・`many_regressors`は20/20で成立。`boundary_df1`（`n=k+K`）は20本中16〜17本が分離せず成立するので、成立するseedを選ぶ。

| シナリオ | 内容 | 目的 | 結果の扱い |
|---|---|---|---|
| `baseline` | `n=500`・`k=3`・`K=4` | 標準 | 参照と照合 |
| `small_n` | `n=40`・`K=3` | 小標本 | 参照と照合 |
| `moderate_multicollinearity` | 2値と同じ | ある程度の多重共線性 | 参照と照合 |
| `high_condition_number` | 相関0.99超 | 悪条件 | 参照と照合 |
| `scale_variance` | 列のスケールが`1e6`・`1e-3` | 極端なスケール差 | 参照と照合 |
| `many_regressors` | `k=20`、係数を小さく較正 | 列数依存のバグ | 参照と照合 |
| `outlier_regressor` | 5%の観測が極端な値 | 高レバレッジ行 | 参照と照合 |
| `near_separation` | `x1`の係数を大きく（logit 8・probit 4程度、実装時に較正） | 分離に近いが有限のMLE | 参照と照合 |
| `perfect_multicollinearity` | 完全な共線 | ランク落ち | `ComputationError`のみ確認 |
| `complete_separation` | カテゴリが`x1`の区間で決定論的に決まる | 完全分離 | `SEPARATION_DETECTED`のみ確認 |
| `quasi_complete_separation` | 最上位カテゴリにだけ現れるダミー | 準完全分離 | 同上。最下位だけ・1観測違い（`near_miss`、分離なし）もLPの回帰ガードとして使う（6章） |
| **`k_min`**（固有） | `K=3` | カテゴリ数の下限 | 参照と照合 |
| **`k_max`**（固有） | `K=30`（約17観測/カテゴリ） | カテゴリ数の上限。閾値29個・LPの変数32個 | 参照と照合 |
| **`sparse_category`**（固有） | 最上位カテゴリが約1%（5観測） | 閾値が近い・小さい確率（C4） | 参照と照合 |
| **`boundary_df1`**（固有） | `n=k+K`（`df_resid=1`）、`k=1`・`K=3` | 自由度ちょうど1の成功パス | 参照と照合（分離しないseed） |
| **`large_means`**（固有） | 平均が標準偏差より大きい列（中心化しないと悪条件） | 中心化（C2）の回帰ガード | 参照と照合 |
| **`heteroskedastic`**（固有） | 潜在誤差の尺度が`x`に依存（誤設定） | ロバストSEが`classical`と実質的に異なる状況での`cov_type`の検証 | 参照と照合 |

**`baseline`の派生**（2値モデルと同じ形）: `cluster`（均等）・`cluster_imbalanced`（不均衡）・`cluster_g4`（`G=q+1`の境界、`q=k=3`。`G≤q`は`ValidationError`のテスト）・`confidence_level`（0.90）・`solver`（`bfgs`・`lbfgs`）。順序モデル固有の派生として**`shifted_codes`**（`y`のコードを`0..K-1`や負の値にずらす。推定値は`baseline`と同じで、閾値の名前（`"-1|0"`等）・`categories`・`predict()`のキーを確かめる）。`no_intercept`は切片が無いので無い。

**テスト方針の観点との対応**: 「分散が大きい」は、順序モデルでは潜在誤差の尺度が識別されない（`β/σ`しか推定できない）ので、小さい係数（`many_regressors`）と同じで別のシナリオは作らない。「不均一分散」は`heteroskedastic`。「自己相関」はHACが無いので対象外。

### 10.2 実データ（Wooldridge）

118データセットを走査し、整数で3〜10個の連続した値をとる列のうち、順序の意味を持つものを候補にした。`happiness`の`happy`は文字列（3水準）なので、読み込みで`1..3`に直す。

| データ | `y` | `K` | `n` | カテゴリの観測数 | 説明変数（案） | クラスター | statsmodels・`polr` |
|---|---|---|---|---|---|---|---|
| **`happiness`**（GSS） | `happy`（1=not too happy, 2=pretty happy, 3=very happy） | 3 | 16991（使う列の欠損を除く） | 2076・9707・5208 | `educ`・`female`・`black`・`babies`・`preteen`・`teens` | `region`（9地域、`G=9>q=6`） | 両方収束、対数尤度一致 |
| **`beauty`** | `looks`（1〜5） | 5 | 1260 | **13**・142・722・364・**19** | `lwage`・`educ`・`exper`・`female`・`married`・`black`・`goodhlth` | 無し | 両方収束、対数尤度一致 |
| `affairs`（予備） | `ratemarr`（結婚生活の評価1〜5） | 5 | 601 | **16**・66・93・194・232 | `age`・`yrsmarr`・`kids`・`relig`・`educ`・`male` | 無し | 両方収束、対数尤度一致 |

- **`beauty`の`belavg`・`abvavg`は`looks`から作られた列**（`looks≤2`、`looks≥4`）で、説明変数に入れると準完全分離になる。実測でLPは分離と判定し（マージン0.87）、statsmodelsはlogitで`LinAlgError`、probitで`|β|=29`の未収束になった。**実データでの`SEPARATION_DETECTED`の回帰ガード**に使う（参照との照合はしない）。成功パスの説明変数からは必ず除く。
- `happiness`は実データのクラスターロバストSEを担う（Logit/Probitの`apple`に相当）。`region`は文字列で、クラスター列として受け付ける型。
- その他の候補（不採用）: `loanapp.action`（承認・条件変更・却下。順序の解釈に議論の余地がある）、`alcohol.status`・`injury.injtype`・`discrim.chain`（名義尺度）、`apple.hhsize`等（個数）。

**決定（確定）**: 実データは`happiness`（全`cov_type`、`cluster`は`region`）と`beauty`（`cluster`以外の全`cov_type`、`belavg`入りは分離のエラーパス）。`affairs`は予備（中くらいの`n`と小さいカテゴリの組み合わせが必要になったら足す）。Wooldridgeのデータは固定化の対象外なので、既存と同じく`load_wooldridge.py`経由で読み、`happy`の数値化と欠損行の除外はローダー側で行う。

## 11. B節の検討結果（入力の検証とエラー）

### 11.1 既存の規約から決まること

- `y`の値の問題（`OUTCOME_*`）は、`y`の列を三つ組（`column`・`argument`・`data_argument`）で特定する（`error_catalog.rs`のテストが`OUTCOME_`で始まる`code`に要求する）。行を持つものは`row`（0始まり）・`value`も持つ。
- 範囲違反は対象ごとに1つの`code`で、下限・上限の側は分けない。しきい値は計算済みの値で`params`に渡す。`params`に英語の句は入れない。
- 入力を見るだけで分かる問題は`ValidationError`、数値ルーチンが要る問題は`ComputationError`（`error-handling.md`3.1章）。
- 定数の`y`は手法横断で`OUTCOME_CONSTANT`（5.4章）。

### 11.2 新しい`code`（案）

**確定（2026-11-28、実装済み）**: 名前は表のとおり。文面はカタログ（`error_catalog.rs`）が正本で、下の下書きから
変えている（公開ページの校閲を反映済み）。`OUTCOME_NOT_INTEGER`・`OUTCOME_CATEGORY_COUNT_OUT_OF_RANGE`は多項ロジットが同じ`code`を
使えるよう「ordered」と書かない。`OUTCOME_CATEGORY_COUNT_OUT_OF_RANGE`の`hint`（Logit/Probitへの誘導）は最小カテゴリ数3が前提で、
最小が3でない手法が再利用するなら別の`code`にする。範囲（`2^53-1`）は文面に書かない。

| `code` | クラス | `params` | 文面（下書き） | `hint`（下書き） |
|---|---|---|---|---|
| `OUTCOME_NOT_INTEGER` | Validation | `column`・`argument`・`data_argument`・`row`・`value` | The outcome at row {row} (0-based) of column '{column}' (given as {argument}) in {data_argument} must be a whole number that codes an ordered category, got {value}. | Code the categories as integers, from the lowest category to the highest. |
| `OUTCOME_CATEGORY_COUNT_OUT_OF_RANGE` | Validation | 三つ組・`n_categories`・`min_categories`・`max_categories` | Column '{column}' (given as {argument}) in {data_argument} has {n_categories} distinct values, but an ordered model needs between {min_categories} and {max_categories} categories. | With two categories, use Logit or Probit; with more than the maximum, merge adjacent categories. |
| `OUTCOME_CATEGORIES_NOT_CONSECUTIVE` | Validation | 三つ組・`missing`・`lowest`・`highest` | The categories in column '{column}' (given as {argument}) in {data_argument} must be consecutive integers from {lowest} to {highest}, but {missing} does not occur. | Recode the outcome so that the observed values are consecutive integers. |
| `CUTPOINT_NAME_COLLISION` | Validation | 三つ組（`column`は衝突した`x`の列） | Column '{column}' of {argument} in {data_argument} has the same name as a cutpoint, which the result names after the two adjacent categories. | Rename the column. |
| `REGRESSOR_CONSTANT_WITH_CUTPOINTS` | Validation | `argument`・`data_argument`・`columns`（該当列を全て、5.3章） | These columns of {argument} in {data_argument} have the same value in every row up to rounding, so they are collinear with the cutpoints, which take the place of an intercept: {columns}. | Remove these columns from {argument}. |

- `OUTCOME_NOT_INTEGER`の「整数」は、元の整数を一意に決められる範囲（絶対値`2^53-1`以下）に限る。`2^53`は`f64`で表せるが、`Int64`の`2^53+1`も変換で`2^53`に丸まるので、元のコードを特定できない。それを超える値は連続性も判定できないので同じ`code`にする（範囲は文面に書かない）。
- 欠番を自動で詰めない理由: `{1, 2, 4}`を黙って`1..3`に直すと、空のカテゴリ（データの誤りの可能性）を隠し、閾値の名前・`categories`も利用者のコードとずれる。
- `CUTPOINT_NAME_COLLISION`: 閾値の名前は`y`のコードから決まるので、静的に検査できるTobitの`"sigma"`（`TOBIT_SIGMA_NAME_COLLISION`、`params`なし）と違い、衝突した列名を`column`で返す。`column`を持つので三つ組にする。
- 定数列の`code`: FEの`REGRESSOR_CONSTANT`と同じ形（`columns`で全列）だが、文面が「固定効果と共線」でFE専用のため別の`code`にする。名前は`REGRESSOR_CONSTANT`系で「閾値と共線」を表すもの（`REGRESSOR_CONSTANT_WITH_CUTPOINTS`は仮）。判定は`has_no_variation`（定数の`y`・FEと同じ相対閾値）。
- **既存の手法での定数列の扱い（実測）と、順序モデルが専用の`code`を持つ理由**:

  | 手法 | `x`に定数列`c` | 結果 |
  |---|---|---|
  | OLS・Logit（`include_intercept=True`） | 明示の切片`const`と共線 | `DESIGN_MATRIX_SINGULAR`（`ComputationError`）、`columns=['const', 'c']` |
  | OLS・Logit（`include_intercept=False`） | `c`が切片の役割をする | 通常どおり推定（エラーではない） |
  | IV・RE（切片あり） | 同上 | `DESIGN_MATRIX_SINGULAR`（`stage`つき）、`columns=['const', 'c']` |
  | FE | 固定効果が吸収する | `REGRESSOR_CONSTANT`（`ValidationError`）、`columns=['c']` |

  線形モデル（とLogit等）では、定数列そのものは不正な入力ではない（切片なしなら正当な定数項）。切片との**組み合わせ**で共線になり、ランクチェックが相手の`const`を名指しして報告する。順序モデルは、閾値が常に切片の役割をして外せない（`include_intercept`が無い）ので、定数列は**常に**推定できない。共線の相手は列ではなく閾値なので、ランクチェックでは`columns=['c']`だけになり原因が分からない。これはFEと同じ状況（相手が列ではない・常に推定できない）なので、FEに揃えて専用の`ValidationError`にする。クラスがOLS（`ComputationError`）とFE（`ValidationError`）で違うのは既存の分け方で、「入力だけで常に不正と分かるか」の違いによる（OLSの定数列は切片の有無次第で正当）。
- 再利用する既存の`code`: `INSUFFICIENT_OBSERVATIONS`（`n_params=k+K-1`）、`INSUFFICIENT_CLUSTERS_FOR_INFERENCE`（`q=k`。閾値は傾きではない）、`OUTCOME_CONSTANT`、`DESIGN_MATRIX_SINGULAR`、`SEPARATION_DETECTED`・`SEPARATION_CHECK_FAILED`、`AUGMENT_COLUMN_EXISTS`、`ROLE_EMPTY`（`x=[]`、B7）。

### 11.3 `DESIGN_MATRIX_SINGULAR`の列の報告

順序モデルの設計行列に切片の列は無いが、閾値が切片の役割をする。ランクチェックは中心化した`x`で行い（C3）、報告する列は「暗黙の切片を加えた`[1|x]`で共線な列を探し、暗黙の切片を除いたもの」にする。ダミー変数の罠（`d1+d2=1`）は、OLS等で`[const, d1, d2]`と報告されるところ`[d1, d2]`になる。定数列は11.2の専用の`code`が先に弾くので、ここには来ない。`hint`の「切片と定数列」は順序モデルでは「閾値と定数列」と読み替えることになるので、`docs/guide/validation.md`に注記する。

### 11.4 検査の順序

`docs/guide/validation.md`のstep 1〜7（型・オプション・ロール・列の存在とdtype・欠損値）は既存の手法と同じ。step 8（engine）は次の順にする。

1. `confidence_level` → `max_iter` → `tol`（MLE共通）
2. `y`の形: `OUTCOME_NOT_INTEGER` → `OUTCOME_CATEGORY_COUNT_OUT_OF_RANGE`（異なる値の数で判定。1種類のときは対象外） → `OUTCOME_CATEGORIES_NOT_CONSECUTIVE` → `CUTPOINT_NAME_COLLISION`（コードが分かった後）
3. `INSUFFICIENT_OBSERVATIONS`（`n ≤ k+K-1`） → `INSUFFICIENT_CLUSTERS_FOR_INFERENCE`（`G ≤ k`）
4. `OUTCOME_CONSTANT`（値が1種類）
5. 定数列（11.2）
6. `DESIGN_MATRIX_SINGULAR`（中心化した`x`のランクチェック）
7. 最適化（`NOT_CONVERGED`等） → 分離（`SEPARATION_DETECTED`・`SEPARATION_CHECK_FAILED`）

- **カテゴリ数を欠番より先に調べる理由**: 異なる値の数はコードの付け方によらない。`{1, 3}`は欠番より「2カテゴリなのでLogit/Probitを使う」が役に立つ。欠番は付け直せば直る問題で、付け直しても数は変わらない。
- **値が1種類のときにカテゴリ数の検査を外す理由**: 定数の`y`を手法横断で`OUTCOME_CONSTANT`にそろえる（`error-handling.md`5.4章。Logit/Probitも「`y`が許される形か → 観測数 → 定数の`y`」の順）。
- **定数列を特異行列の前に置く理由**: 中心化すると定数列はゼロ列になり、`DESIGN_MATRIX_SINGULAR`の報告では原因（閾値と共線）が分からない。FEが固定効果の除去の前に`REGRESSOR_CONSTANT`を調べるのと同じ。

### 11.5 更新する文書・テスト（実装時）

- `docs/spec/error-handling.md`: 5.2章（step 8の`y`の形の項目に順序モデルを足す）、5.4章の表に順序モデルの行（「切片あり」の扱い）。
- `docs/guide/validation.md`: Outcomeの節（整数・カテゴリ数・連続性・名前の衝突）、定数列、`DESIGN_MATRIX_SINGULAR`の注記、step 8の順序。
- `docs/guide/accepted-data.md`: 順序モデルの`y`（整数コード、任意の開始値）。
- `docs/guide/error-codes.md`: 生成物（`tools/render_error_catalog.py`）。
- テスト: `tests/cross_method/test_validation_order.py`の`FAULTS`（新しい`code`の段階と順位）、`test_perfect_fit.py`（定数の`y`）、`tests/support/_errors.py`の`MIGRATED_CODES`。

## 12. 実装の進め方（次のセッションへの引き継ぎ）

### 12.1 タスクの順序

1. **共通基盤**（Issue「Ordered models: 入力型・カテゴリバリデーション共通基盤」）: `y`の検証（整数・カテゴリ数・欠番）、閾値名の衝突・定数列の検査、新しいエラー`code`（11.2）、順序モデルの分離のLP（6章）と`DenseSimplex`の変数50個までの回帰テスト。
2. **OrderedLogit engine**: 尤度・スコア・Hessian（5章の対数空間の式）→ `fit`（直接`τ`・中心化・初期値・`stpmax`・分離チェック）→ `cov_type` → 適合度 → カテゴリ別確率・`pred_table`。
3. **OrderedLogit engine_pybind → python_package**（`params_table()`・閾値の名前・`categories`/`cutpoints`/`cutpoint_names`・`predict()`の`dict`・`augment()`・pickle・GIL）。新しい`code`を`UNREACHABLE_CODES`から`MIGRATED_CODES`へ移す。
4. **OrderedLogit ベンチマーク・テスト**（`/test-new`。9章の表・10章のデータ）→ ドキュメント → 性能比較。
5. **OrderedProbit**: `ln_normal_cdf`（`shared::normal`）→ リンク非依存の部分を`nonlinear/common/`へ抽出 → 2〜4と同じ。
6. **限界効果**（`marginal_effects()`の追加のIssue）: Rの`marginaleffects::avg_slopes(polr)`が主リファレンス。

### 12.2 最初のセッション（共通基盤）で行うこと

1. **作業場所**: `/workspaces/econometricsmodels`の`release/v0.11.0`。並行する別作業の未コミットの変更がありうるので、コミットは自分のファイルだけをステージする。署名のたびにホスト端末で確認が出る（`.claude/rules/commit-signing.md`）。
2. **読むもの**: 本ノートの1〜11章（特に1章の表、6章、11章）。`engine/src/nonlinear/CLAUDE.md`（自動で読まれる）と、ソルバー・分離検出を触るときは`engine/src/nonlinear/common/CLAUDE.md`・`.claude/rules/engine-separation.md`（該当するファイルを読むと自動で読まれる）。`docs/spec/error-handling.md`の3.1・4・5.2〜5.4章と`docs/spec/error-style-guide.md`。
3. **既存実装の調査**（CLAUDE.md 4章。Exploreエージェントに委ねる）: `nonlinear/common/validation.rs`（`validate_binary_y`・`validate_fit_preconditions`・`validate_outcome_not_constant`）、`nonlinear/separation.rs`・`separation/lp.rs`・`separation/rows.rs`、`shared/linear_program.rs`（`BoxBoundedLp`・`DenseSimplex`）、FEの定数列の検査（`panel/common/validation.rs`の`RegressorConstant`）、`shared/error_catalog.rs`へのエラーの足し方（`.claude/rules/rust-style.md`のエラーハンドリング節の手順1〜5）。
4. **計画**: `/implement-rust`で計画を立て、着手前に次の未決事項をユーザーに確認する（CLAUDE.md 14章）。**確定（2026-11-28）**:
   名前は`REGRESSOR_CONSTANT_WITH_CUTPOINTS`、文面は11.2のとおり（中立の文面）。新しい`code`は`UNREACHABLE_CODES`に一時登録。
   `validate_categorical_y(y, dep_var_name, min, max) -> Vec<i64>`（整数・カテゴリ数。値が1種類以下は数の検査をしない）と
   `validate_consecutive_categories`（欠番）の2関数に分け、行ごとのカテゴリ番号は返さない（`fit`が`y - lowest`で求める）。
   - 新しい`code`の最終的な名前（特に`REGRESSOR_CONSTANT_WITH_CUTPOINTS`は仮）と文面（`error-style-guide.md`のチェックリスト）。
   - 新しい`code`をPythonから送出できるのはOrderedLogitのバインディングの後なので、それまで`tests/support/_errors.py`の`UNREACHABLE_CODES`に理由つきで登録する（`tests/cross_method/test_error_catalog.py`がカタログの全`code`の登録を検査する）。この進め方でよいか。
   - `validate_categorical_y`の返り値（コードの一覧・最小値等）と、多項ロジットが後で再利用できる引数の形。
5. **実装の注意**:
   - 分離のLPは、制約の行を中心化・標準化した`x`から都度計算し（複製しない）、`K`に応じて端のカテゴリは1行、中間は2行。判定は最大マージン`>1e-6`。**`DenseSimplex`は変数30個までしかテストされていない**ので、`k+K-1`が約50になる場合の回帰テストを最初に足して確かめる（右辺0の錐の退化に注意。`engine/src/nonlinear/CLAUDE.md`の分離検出の節）。
   - 分離のLPの単体テスト: 最上位カテゴリだけに現れるダミー（分離）・最下位だけ（分離）・1観測を隣のカテゴリに足したもの（分離なし）。ランダムな小標本でLPとMLEの存在の一致を見る試験は、`fit`ができる段階（OrderedLogit）で足す。
   - 新しい`code`を足したら、`UPDATE_ERROR_CATALOG_GOLDEN=1 cargo test -p engine error_catalog`、`maturin develop`の後に`python tools/render_error_catalog.py`（`docs/guide/error-codes.md`と`error_codes.py`を生成し直す）。
6. **確認**: `cargo test -p engine`・`cargo clippy --all-targets -- -D warnings`・`cargo fmt --check`、`pytest tests/cross_method`（カタログ・検査順の検査）。コミット前に`/review-rust`。

### 12.2.1 共通基盤の実装後、OrderedLogitで配線するときの注意

- 共通基盤の関数（`validate_categorical_y`・`validate_consecutive_categories`・`cutpoint_names`・
  `validate_no_cutpoint_name_collision`・`validate_no_constant_regressors`・`verify_no_ordered_separation`）は
  `#[cfg_attr(not(test), expect(dead_code, ...))]`付き。`fit`から呼んだらビルドが失敗するので、属性を外し`common.rs`で再エクスポートする。
- 定数の`y`は`categories.len() == 1`で判定し、`CommonError::OutcomeConstant`を返す（11.4の位置）。`validate_outcome_not_constant`の
  相対閾値（`n·ε·max|y|`）は、コードの絶対値が大きいと（例: `n=10^6`で`{1e10, 1e10+1, 1e10+2}`）3カテゴリ以上の`y`も定数とみなす。
  `error-handling.md`5.4章の「判定は`has_no_variation`の1か所」に順序モデルの例外として書き足す。
- 空の`y`では`categories`が空。`y`の検査は観測数より先なので、`K-1`を先に計算すると`usize`の下位あふれになる（`k + K - 1`の順か`saturating_sub`）。
- `verify_no_ordered_separation`の`means`には、`fit`が最適化で平均を引くか（C2）によらず列の平均を渡す。
- 分離のLPは1ラウンドに長さ`3n`の一時領域を確保する（2値は`n`）。`n=10^6`のコストを性能比較（F6）で計る。
- `OUTCOME_*`等の新しい`code`を`UNREACHABLE_CODES`から`MIGRATED_CODES`へ移す（バインディングの後）。
- 既存の`OUTCOME_CONSTANT`の文面「so a model with an intercept has nothing to explain」は、切片の無い順序モデルにそのまま当てはまらない。
  配線時に「an intercept or cutpoints」のように一般化する（`params`は変えないので非破壊。`error-handling.md`5.4章の表も更新）。

### 12.3 プロトタイプについて

C1・C4・C5・F1・F4の実験は、セッションのscratchpad（`/tmp/claude-1000/.../scratchpad/`）のPythonプロトタイプで行い、永続化していない。結果と数値は本ノートの各章に記録してあるので、実装では本ノートを根拠にする（プロトタイプが消えていても困らない）。
