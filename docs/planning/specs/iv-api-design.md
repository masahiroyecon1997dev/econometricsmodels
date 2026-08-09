# IV（操作変数法）API・オプション設計

[`panel-iv-design-points.md`](../panel-iv-design-points.md)の論点をモデルごとにIssue化したうちの、IV固有の確定事項をまとめる。
FE/RE共通の設計は[`panel-api-design.md`](./panel-api-design.md)を参照。

**ステータス**: 確定（1章: 引数設計／2章: 結果設計／3章: 標準誤差・検定／
4章: 内部実装・共通化／5章: リファレンス実装・テスト方針／
6章: IV固有論点）。IV側の論点はすべて確定。ただし以下2点の細部は、
Issue #171（`linearmodels`/`ivreg`とのベンチマーク作成）でリファレンス実装を確認して
から最終判断する未決着事項として残っている。
- 3.2節のGMMの検定分布（z分布のまま確定とするか、実務慣行に合わせてt分布・
  切り替えオプションにすべきか。Issue #159時点で追記）
- 3.1節のIV版HC2/HC3（レバレッジ算出式に確立した参照実装が無い。Issue #166時点で追記）

## 1. 引数設計（確定）

### 1.1 `y` / `x_exog` / `x_endog` / `instruments` のシグネチャ

- `fit_iv(data, y, x_exog, x_endog, instruments, options)`
- `y: str`、`x_exog: list[str]`（外生説明変数）、`x_endog: list[str]`（内生説明変数）、
  `instruments: list[str]`（操作変数）を**すべて独立の引数**として渡す
  （`panel-iv-design-points.md`1.1節の「案B」を採用）。
- 理由: いずれもIVモデルの成立に必須（内生変数が無ければOLSと同じ、操作変数が無ければ識別
  不可）であり、`OLSOptions.cluster_col`のような「機能をopt-inするための任意列」とは性質が
  異なるため。`panel-api-design.md`の`entity`（FE/REで必須→独立引数）と同じ原則。
- `x_exog`は空リストを許容する（内生変数のみのモデルも成立するため）。`x_endog`/
  `instruments`は最低1要素を要求する見込み（丁度識別・過剰識別の判定を含む具体的な
  バリデーションルールは確定）。
- **命名規則**: bareネーミング（`_col`サフィックスなし）。`panel-api-design.md`の`entity`/
  `time`と同じ考え方（モデルを構成する中核的な変数）。

### 1.1.1 `x_exog` と `instruments` の重複可否（確定）

- **`instruments`は除外操作変数（excluded instruments）のみを指定する**。`x_exog`に含めた
  列を`instruments`に重複して渡す必要はない（渡すとバリデーションエラーになる、後述）。
  第一段階の設計行列（全操作変数）は内部で`x_exog ++ instruments`をunionして構築する。
  Stata `ivregress 2sls y x1 x2 (x3 = z1 z2)`、Python `linearmodels.IV2SLS(dependent, exog,
  endog, instruments)`と同じ方式。
- 採用理由:
  1. 計量経済学の教科書的な語彙（「instruments」は識別のソースとなる除外操作変数を指す）
     と一致する。
  2. CLAUDE.md 2章の非交渉事項（formula不採用・プログラムから動的に組み立てやすいAPI）と
     整合する。`x_exog`を変更するたびに`instruments`側も手動で同期する重複入力は、
     プログラムからの動的構築とヒューマンエラー防止の両面で相性が悪い。
  3. 過剰識別検定（Sargan/Hansen）の自由度が`len(instruments) - len(x_endog)`
     とそのまま一致し、`x_exog`分を差し引く補正が不要になる。
- **パラメータ名は`instruments`のまま維持する**（`excluded_instruments`等への改名はしない）。
  linearmodelsの命名を踏襲し、意味は本節にドキュメント化することで対応する。
- **バリデーション**: `instruments`に`x_exog`または`x_endog`と重複する列名が含まれていた場合は
  `ValidationError`にする（R `ivreg`のformula流儀に慣れたユーザーが誤って重複入力した場合に
  黙って通さないため）。

### 1.2 モデル固有オプションの置き場所

`IvOptions`という独立の`#[pyclass]`構造体に含める（`OLSOptions`/`LogitOptions`の前例を踏襲）。
2SLS/GMMの方式切り替え等、IV固有の詳細は6章で確定。

### 1.3 `weights` / `offset` の扱い

`panel-api-design.md`1.3と同じ（`offset`は該当なし、`weights`は汎用オプションとしては見送り）。

## 2. 結果（Return）設計（確定）

### 2.1 共通コア項目（`panel-api-design.md`2.1との差分）

`panel-api-design.md`2.1のFE/RE共通コアを土台にするが、IVは以下の点で異なる。

| フィールド | FE/REとの違い |
|---|---|
| `params` / `std_errors` / `stats` / `p_values` / `conf_lower` / `conf_upper` / `param_names` / `residuals` / `dep_var_name` / `n_obs` / `df_resid` / `df_model` / `cov_type` / `f_statistic` / `f_p_value` | `t_stats`ではなく**`stats`**という分布非依存の名前にする（Issue #159で確定）。1つの`IvResult`型を2SLS（t分布）・GMM（z分布、3章参照）の両方が共有するため、`OLSResult.t_stats`/`LogitResult.z_stats`のような分布固定の名前は使えない。`engine::inference::InferenceStat`が同じ理由で`stat`という分布非依存の名前を使っている前例に倣った。それ以外は共通（そのまま踏襲） |
| `n_entities` | **含めない**（IVはパネル構造を前提としない） |
| `log_likelihood` / `aic` / `bic` | **除外する**。2SLS/GMMは尤度ベースの推定法ではなく
  （Stataの`ivregress`もデフォルトでは出力しない）、正規性を仮定した疑似尤度を計算して
  OLS/FE/REと同じフィールド名で返すと、異なる推定基準の値を同列に比較できるかのように
  誤解させるため、統計的な誠実さを優先して含めない |
| `r_squared` / `r_squared_adj` | FE/REのような3分割はせず、**OLSと同じ単一フィールド**を
  維持する（IVはパネルのwithin/between区別を持たないため） |
| `f_statistic` / `f_p_value` | **GMMは常にロバストWald検定（χ²）とする**。OLSが
  `cov_type`がHC系/clusterのとき`f_statistic`/`f_p_value`を古典的F検定からロバストWald検定に
  切り替える既存挙動（`ols-api-design.md`6章）を、GMMにも一貫適用する。GMMは3章で
  z分布と決定済みで古典的F検定の正当化が無いため、フィールド名はそのまま流用しつつ常に
  Wald版にする（新規フィールドは追加しない）。2SLSはOLSと同じ切り替えロジック（classical時は
  F検定、HC/cluster/hac時はロバストWald検定）。 |

### 2.2 モデル固有の追加結果の配置

- **第一段階回帰結果は`first_stage()`という別メソッド**に切り出す
  （`fit()`の戻り値本体には含めない）。非線形モデルの`marginal_effects()`分離方針を踏襲。
- **戻り値の構造**: `first_stage() -> dict[str, OlsResults]`。キーは`x_endog`の変数名、値は
  既存の`OlsResults`型（新規のIV専用型は作らない）。内生変数が複数ある場合、内生変数の数だけ
  第一段階回帰（`x_endog[i] ~ x_exog + instruments`）が存在するため、変数名キーのdictで
  複数の完全な回帰結果を返す。
- 弱操作変数診断（第一段階F統計量）・Sargan/Hansen J（過剰識別）・Wu-Hausman（内生性）は
  いずれも`fit()`の結果本体に含める（別メソッド化しない）。詳細は6章で確定。

## 3. 標準誤差・検定（確定）

### 3.1 `cov_type`のサポート対象

`panel-api-design.md`3.1と同じ範囲（`classical` / `hc0`〜`hc3` / `cluster` / `hac`）を
サポートする。IVはパネル構造を前提としないため、`hac`はOLSの実装（グローバルな時系列順序に
対する通常のNewey-West型）をそのまま踏襲してよい（Driscoll-Kraay型は不要）。

- **デフォルトは`"classical"`のまま**（OLS踏襲）。FE/REと異なり`entity`のような常に存在する
  グルーピング列が無いため、`"cluster"`をデフォルトにする実装上の根拠がない。
- 2SLSの分散はサンドイッチ型（`(X'PzX)^-1 X'Pz Ω Pz X (X'PzX)^-1`、`Ω`の推定方法が
  `cov_type`で変わる）。**GMMは`cov_type`（最終SEの計算方法）と`weight_type`（点推定に使う
  重み行列）を分離する**（詳細は6.2）。
- **`hc2`/`hc3`（レバレッジ`h_ii`によるスケーリング）は引き続き外部の参照実装で検証できない**
  （Issue #171で再確認、`benchmark/iv/run_linearmodels_benchmark.py`のモジュールdocstring
  参照）: `linearmodels.iv.covariance`は`Homoskedastic`/`Heteroskedastic`/`Kernel`/
  `Clustered`のみで、R `ivreg`も同様（`hatvalues.ivreg`の実装がソース上コメントアウトされて
  いる）。実装（`engine/src/iv/two_sls.rs`）はOLSのHC2/HC3を代数的に拡張した自作の拡張の
  ままで、外部パッケージとの数値照合はできない（`engine`のRust単体テストによる独立な
  手計算クロスチェックのみで検証を継続する）。
- **`classical`/`hc0`〜`hc1`/`cluster`/`hac`は`linearmodels`の`cov_type`/`debiased`との対応が
  実測で確定した**（Issue #171）: `classical`↔(`unadjusted`, `debiased=True`)、
  `hc0`↔(`robust`, `debiased=False`)、`hc1`↔(`robust`, `debiased=True`)、
  `cluster`↔(`clustered`, `debiased=True`)、`hac`↔(`kernel`(bartlett), `debiased=False`)。
  いずれも`coef`/`se`が相対誤差1e-10以下（実質機械精度）で一致することを確認した
  （詳細な対応表・検証根拠は`run_linearmodels_benchmark.py`のモジュールdocstring参照）。

### 3.2 検定分布

**2SLSとGMMで分ける**。
- **2SLS**: t分布（OLS系、`panel-api-design.md`3.3と同じ理由）。自由度は`df_resid`を使う。
- **GMM**: z分布。2-step efficient GMMはM推定量としての漸近正規性が根拠であり、有限標本の
  t分布としての正当化がない（非線形モデル・MLE系のz分布判断と同じ理由）。
- 2SLSとGMMの実装方針の違いと接続する決定であり、実装の詳細（GMM目的関数・
  重み行列の設計）は6章で確定する。
- **`linearmodels`の分布切り替え方式が判明した（Issue #171で確認）**: `IV2SLS`/`IVGMM`とも
  `fit(debiased=False)`（既定）では正規分布（z）・`f_statistic`もカイ二乗形式（qで割らない生の
  二次形式）、`fit(debiased=True)`ではt(df_resid)分布・F分布（`f_statistic`もqで割った形式）を
  返す（`linearmodels.iv.results.IVResults.pvalues`/`f_statistic`のソースで確認済み。Stataの
  `ivregress`の`small`オプションと同じ発想）。本実装の2SLSは`cov_type`によらず常にt分布/F分布
  という設計だが、`linearmodels`は`debiased`という別軸で切り替わるため、`hc0`/`hac`
  （`debiased=False`で`coef`/`se`が一致する組み合わせ）をベンチマークする際は
  `linearmodels`の`pvalues`/`tstats`/`f_statistic`をそのまま使わず、`coef`/`se`のみ借りて
  t分布・F分布で独自に計算し直す必要があった（`run_linearmodels_benchmark.py`参照）。
  GMMについては、`linearmodels`の既定（`debiased=False`→z分布）が本実装の設計
  （GMMは常にz分布）と一致することも合わせて確認できた。GMM自体のcov_type対応・SE計算は
  Issue #171の作業中に実装した（当初Issue #166でクローズ済みの完了条件だったが、実際には
  GMM側が未実装のまま残っていたことが発覚したため。`engine/src/iv/gmm.rs`のモジュール
  docコメント「標準誤差・検定統計量（cov_type対応）」・`engine/src/iv/CLAUDE.md`参照）。
  `linearmodels`（`IVGMM`）との数値照合は`method="gmm"`のPython配線が別途必要なため、
  そちらの完了後に改めて行う。

## 4. 内部実装・共通化（確定）

`panel-api-design.md`4章の方針をそのまま踏襲する。IV固有の追加点は以下。

- **`instruments`の複数列ロール対応**（`panel-api-design.md`4.2の3）がIV実装の前提になる。
  `validate_no_duplicate_roles`の拡張なしに`x_exog`/`x_endog`/`instruments`間の重複検証は
  実装できない。
- IVのサンドイッチ型分散計算は独自実装でよい（`panel-api-design.md`4.4、OLS/nonlinear
  どちらの既存計算にも寄せない）。
- GMMが数値最適化を要する場合、`nonlinear/common.rs`の`run_solver`
  （Newton/BFGS/L-BFGS、`CostFunction`/`Gradient`/`Hessian`トレイトのみに依存するモデル
  非依存の設計）を転用できる可能性がある。効率的2-step GMMは閉形式で済むことが多いため
  必須ではないが、選択肢として残す（詳細は6章）。
- **新規エラー型**: `LeastSquaresError`/`MleError`/`PanelError`（`panel-api-design.md`4.4）の
  前例に倣い、**`IvError`を2SLS/GMMで共有する**（`engine/src/iv/common.rs`に定義）。個別に
  `TwoSlsError`/`GmmError`を作らない。

## 5. リファレンス実装・テスト方針（確定）

### 5.1 Python主リファレンス

**`linearmodels`を主リファレンスとする**（`IV2SLS`＝2SLS、`IVGMM`＝GMM）。6章で
挙がっている診断（Sargan/Hansen J、first-stage F統計量、Wu-Hausman系）も概ね
`linearmodels`でカバーできる見込み。

### 5.2 Rクロスチェックパッケージ

**`ivreg`**。ただし`ivreg`は2SLSのみ対応でGMMには対応していない見込み（要実装時再確認）。

**現状の既知の問題（Issue #171で発覚）**: devcontainerのRには`ivreg`が未導入で、インストールも
できない状態（`ivreg`が要求する`car`パッケージの依存先`MatrixModels`が`Matrix>=1.6.0`
（→R>=4.4）を要求するが、devcontainerのRは4.2.2で更新できない）。CLAUDE.md §10の
「導入済み」という記述は実態と食い違っている（要修正）。当面はPython
（`linearmodels`）側のみで2SLS/GMMともに検証し、Rクロスチェックは実装を保留する
（ユーザー確認済み。代替候補として、AERパッケージの`ivreg()`関数や`gmm`パッケージは
依存が軽く導入できる可能性がある。Rクロスチェックに実際に着手する際に確認する）。

### 5.3 GMMのRクロスチェック省略（例外規定）

GMMは`ivreg`が対応していないため、**Python（`linearmodels`）のみで検証しRクロスチェックを
省略する**ことを許容する。`panel-api-design.md`5.3のハウスマン検定と同様、
「Python主リファレンス＋Rクロスチェック」の2系統検証の例外であることをテスト実装時の
コメント・ドキュメントに明記する。R側で別パッケージ（`gmm`等）を新規導入するコストは
掛けない。

### 5.4 許容誤差

`panel-api-design.md`5.4と同じ（既存方針の相対誤差1e-8を基本、GMM等で乖離が大きい場合は
実測値に基づき個別に緩和を検討）。

## 6. IV固有論点（確定）

`linearmodels`（`linearmodels/iv/model.py`・`results.py`）のソースコードを確認しながら
確定した。

### 6.1 引数の切り分け

1.1.1節で確定済み。`x_exog`/`x_endog`/`instruments`をすべて独立引数化、
`instruments`は除外操作変数のみ。

### 6.2 2SLSとGMMの実装方針の違い

- **`weight_type`（GMMの点推定に使う重み行列）と`cov_type`（最終的な報告用SE計算）を分離
  する**。`linearmodels.IVGMM`と同じ構造（`weight_type`はコンストラクタ/Options側、
  `cov_type`は`fit()`側という区別ではなく、`IvOptions`に両方のフィールドを持たせる）。
  他のモデルと違い、**GMMは`cov_type`相当の選択が点推定自体に影響する**
  （効率的GMMの重み行列は仮定する誤差構造に依存するため）。この分離をしないと、GMMだけ
  「SEを変えたら係数も変わる」という他モデルには無い挙動を`cov_type`の名の下に隠すことに
  なり紛らわしい。
  - `weight_type`の取りうる値: `unadjusted`/`homoskedastic`、`robust`/`heteroskedastic`、
    `cluster`、`kernel`（Driscoll-Kraayではなく通常のHAC、IVはパネル構造を前提としないため
    3.1と同じ理由）。
- **GMMのstep数（1-step/2-step efficient/iterated）を選択可能にする**。`IvOptions`に
  `gmm_iterations: int`（デフォルト`2`＝efficient two-step、`1`で1-step GMM）を追加する。
  `linearmodels.IVGMM.fit(iter_limit=2, ...)`と同じ考え方。当初は1・2の2値のみ許容していた
  が（Issue #165）、Issue #229で3以上（iterated GMM）・収束条件に一般化した:
  - `gmm_iterations`は1以上の任意の整数を受け付ける。`gmm_convergence: float | None`
    （既定`None`）を追加し、`Some`のときは`gmm_iterations`を「固定反復回数」ではなく
    「収束判定の上限反復回数（安全弁）」として扱う。両者は排他ではなく併用方式
    （ユーザー確認済み）。
  - 収束判定は係数のelementwise・絶対誤差と相対誤差の併用（`tol = max(rtol * |前回値|,
    atol)`、`atol`は内部固定値`1e-8`でユーザーには公開しない、ユーザー確認済み）で行う。
  - 収束しない場合の挙動は`nonlinear`のMLE実装（`LogitOptions`/`ProbitOptions`の
    `max_iter`/`tol`/`raise_on_non_convergence`/`converged`）と同じ設計に揃える:
    `raise_on_non_convergence: bool`（既定`true`）が`true`なら未収束時にエラー、
    `false`なら`converged=False`のまま結果を返す。
  - `gmm_iterations`を「methodそのもの」（例: `"1step_gmm"`/`"2step_gmm"`/
    `"iterated_gmm"`）に分割する案も検討したが不採用。反復回数はGMMという1つの
    `method="gmm"`のパラメータとして扱う（`linearmodels`等の慣行と同じ、詳細は
    `engine/src/iv/CLAUDE.md`参照）。
  - `engine`側の実装は`engine::iv::gmm::GmmEstimator::fit`のみ完了（Issue #229）。
    `IvOptions`/`IvResult`（`engine_pybind`）・`python_package`への配線は
    `method="gmm"`自体がまだ未実装（本節冒頭、6章の前提）のため別issueで行う。
- **2SLSはGMMの特殊ケース（`weight_type="unadjusted"`、`gmm_iterations=1`）として実装できる**
  ことを踏まえ、共通のGMM推定コアを実装し、2SLSはそのコアを固定パラメータで呼び出す設計に
  する。ただし無理な共通化はしない方針（4章）に従い、実際にどこまで一体化できるかは
  実装時に判断する（うまく一体化できない場合は2SLS側を素直に閉形式で実装してよい）。

### 6.3 丁度識別（just-identified）の場合の扱い

丁度識別（`len(instruments) == len(x_endog)`）では、GMMの点推定は`weight_type`によらず2SLSと
数値的に一致する（GMMの一般的性質: 丁度識別ではモーメント条件を正確に0にできるため重み行列が
点推定に影響しない）。6.2の共通GMM推定コアで自然に吸収され、特別な分岐コードは不要。

### 6.4 弱操作変数診断（第一段階F統計量）

- **x_exogを直交化した後の操作変数係数のみを検定する「部分F統計量」として専用計算する**
  （`linearmodels.iv.results.FirstStageResults.diagnostics`と同じ方式）。`first_stage()`が
  返す`OlsResults.f_statistic`（x_exog込みの全回帰係数に対する検定）をそのまま使うと、
  x_exogの寄与が混ざり弱操作変数診断として不正確になるため、**別計算が必要**。
- 内生変数ごとに計算し、**`fit()`の主結果にサマリーとして含める**
  （フィールド名は実装時に確定、例: 内生変数名キーの`dict[str, float]`）。詳細な内訳
  （各内生変数のOLS回帰そのもの）は`first_stage()`側に残す。
- **v1のスコープ**: 生の部分F統計量のみ返す。Stock-Yogo臨界値テーブルとの照合（弱操作変数の
  合否判定）はテーブルが経験的なシミュレーション値でクローズドフォームでないため実装コストが
  高く、v1では実装しない。複数内生変数の同時検定（Cragg-Donald統計量等）も同様にv1スコープ外
  とし、各内生変数ごとの部分F統計量のみ返す。

### 6.5 過剰識別検定（Sargan / Hansen J）

- **Sargan検定（2SLS）／Hansen J検定（GMM）を`fit()`の結果本体に含める**（別メソッド化
  しない。OLS/REの適合度統計量と同じeager計算方針を踏襲。`linearmodels`は遅延プロパティだが
  本プロジェクトは一貫してeager計算とする）。
- 自由度は`len(instruments) - len(x_endog)`（1.1.1節で確定した`instruments`＝除外操作変数
  のみという定義とそのまま整合）。
- **丁度識別（自由度0）の場合は`None`を返す**（`linearmodels`も`InvalidTestStatistic`相当の
  扱いをしている）。

### 6.6 内生性検定（Wu-Hausman）

- 「Wu-Hausman検定（回帰ベース）」は、**第一段階残差を構造式に追加回帰し係数のジョイント
  有意性を検定する方式**（`linearmodels.iv.results.IVResults.wooldridge_regression`相当）で
  実装する。`linearmodels`には数式が異なる`wu_hausman`（SSR差に基づく古典公式）という別
  メソッドも存在するが、「回帰ベース」という表現とv1の実装コストを踏まえ
  `wooldridge_regression`相当を採用する。
- `first_stage()`で計算する残差をそのまま再利用できる。
- **`fit()`の結果本体に含める**（内生変数全体のジョイント検定のみ、変数ごとのサブセット検定
  はv1スコープ外）。

### 6.7 標準誤差の計算方法

- **2SLS**: `classical`/`hc0`〜`hc3`/`cluster`/`hac`（3.1で確定済み）。
- **GMM**: 6.2の`weight_type`とは独立に`cov_type`を選択できる。サポート対象は2SLSと同じ範囲を
  踏襲する。

- IV固有論点（2SLS/GMM方式・丁度識別/過剰識別・弱操作変数診断・内生性検定等）: 本章（6章）で確定。
