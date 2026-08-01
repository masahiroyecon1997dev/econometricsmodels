# IV（操作変数法）API・オプション設計

[`panel-iv-design-points.md`](../panel-iv-design-points.md)の論点をモデルごとにIssue化したうちの、IV固有の確定事項をまとめる。
FE/RE共通の設計は[`panel-api-design.md`](./panel-api-design.md)を参照。

**ステータス**: 一部確定（1章: 引数設計、Issue #119／2章: 結果設計、Issue #120／3章: 標準誤差・検定、
Issue #121）。他は未確定（4章参照）。

## 1. 引数設計（Issue #119、確定）

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
  バリデーションルールはIssue #126で確定）。
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
  3. 過剰識別検定（Sargan/Hansen、Issue #126）の自由度が`len(instruments) - len(x_endog)`
     とそのまま一致し、`x_exog`分を差し引く補正が不要になる。
- **パラメータ名は`instruments`のまま維持する**（`excluded_instruments`等への改名はしない）。
  linearmodelsの命名を踏襲し、意味は本節にドキュメント化することで対応する。
- **バリデーション**: `instruments`に`x_exog`または`x_endog`と重複する列名が含まれていた場合は
  `ValidationError`にする（R `ivreg`のformula流儀に慣れたユーザーが誤って重複入力した場合に
  黙って通さないため）。

### 1.2 モデル固有オプションの置き場所

`IvOptions`という独立の`#[pyclass]`構造体に含める（`OLSOptions`/`LogitOptions`の前例を踏襲）。
2SLS/GMMの方式切り替え等、IV固有の詳細はIssue #126で確定。

### 1.3 `weights` / `offset` の扱い

`panel-api-design.md`1.3と同じ（`offset`は該当なし、`weights`は汎用オプションとしては見送り）。

## 2. 結果（Return）設計（Issue #120、確定）

### 2.1 共通コア項目（`panel-api-design.md`2.1との差分）

`panel-api-design.md`2.1のFE/RE共通コアを土台にするが、IVは以下の点で異なる。

| フィールド | FE/REとの違い |
|---|---|
| `params` / `std_errors` / `t_stats` / `p_values` / `conf_lower` / `conf_upper` / `param_names` / `residuals` / `dep_var_name` / `n_obs` / `df_resid` / `df_model` / `cov_type` / `f_statistic` / `f_p_value` | 共通（そのまま踏襲） |
| `n_entities` | **含めない**（IVはパネル構造を前提としない） |
| `log_likelihood` / `aic` / `bic` | **除外する**。2SLS/GMMは尤度ベースの推定法ではなく
  （Stataの`ivregress`もデフォルトでは出力しない）、正規性を仮定した疑似尤度を計算して
  OLS/FE/REと同じフィールド名で返すと、異なる推定基準の値を同列に比較できるかのように
  誤解させるため、統計的な誠実さを優先して含めない |
| `r_squared` / `r_squared_adj` | FE/REのような3分割はせず、**OLSと同じ単一フィールド**を
  維持する（IVはパネルのwithin/between区別を持たないため） |

### 2.2 モデル固有の追加結果の配置

- **第一段階回帰結果は`first_stage()`という別メソッド**に切り出す
  （`fit()`の戻り値本体には含めない）。非線形モデルの`marginal_effects()`分離方針を踏襲。
- **戻り値の構造**: `first_stage() -> dict[str, OlsResults]`。キーは`x_endog`の変数名、値は
  既存の`OlsResults`型（新規のIV専用型は作らない）。内生変数が複数ある場合、内生変数の数だけ
  第一段階回帰（`x_endog[i] ~ x_exog + instruments`）が存在するため、変数名キーのdictで
  複数の完全な回帰結果を返す。
- 弱操作変数診断（第一段階F統計量、Stock-Yogo基準）を`first_stage()`の出力にどう含めるか
  （各`OlsResults`に追加情報を添えるか等）はIssue #126で確定する。
- Sargan/Hansen検定・Wu-Hausman内生性検定は`fit()`の結果本体に含めるか別メソッドにするかを
  含め、Issue #126で確定する（本Issueのスコープ外）。

## 3. 標準誤差・検定（Issue #121、確定）

### 3.1 `cov_type`のサポート対象

`panel-api-design.md`3.1と同じ範囲（`classical` / `hc0`〜`hc3` / `cluster` / `hac`）を
サポートする。IVはパネル構造を前提としないため、`hac`はOLSの実装（グローバルな時系列順序に
対する通常のNewey-West型）をそのまま踏襲してよい（Driscoll-Kraay型は不要）。

- **デフォルトは`"classical"`のまま**（OLS踏襲）。FE/REと異なり`entity`のような常に存在する
  グルーピング列が無いため、`"cluster"`をデフォルトにする実装上の根拠がない。
- 2SLSの分散はサンドイッチ型（`(X'PzX)^-1 X'Pz Ω Pz X (X'PzX)^-1`、`Ω`の推定方法が
  `cov_type`で変わる）。GMMは重み行列の選択（1-step/2-step efficient、Issue #126）と
  `cov_type`の計算が密接に連動するため、具体的な対応表はIssue #126で確定する。

### 3.2 検定分布

**2SLSとGMMで分ける**。
- **2SLS**: t分布（OLS系、`panel-api-design.md`3.3と同じ理由）。自由度は`df_resid`を使う。
- **GMM**: z分布。2-step efficient GMMはM推定量としての漸近正規性が根拠であり、有限標本の
  t分布としての正当化がない（非線形モデル・MLE系のz分布判断と同じ理由）。
- Issue #126（2SLSとGMMの実装方針の違い）と接続する決定であり、実装の詳細（GMM目的関数・
  重み行列の設計）は同Issueで確定する。

## 4. その他の論点（未確定）

- 内部実装・共通化: Issue #122
- リファレンス実装・テスト方針: Issue #123
- IV固有論点（2SLS/GMM方式・丁度識別/過剰識別・弱操作変数診断・内生性検定等）: Issue #126
