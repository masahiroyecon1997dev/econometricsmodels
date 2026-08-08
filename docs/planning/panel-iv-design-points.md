# FE / RE / IV API設計 論点整理（Issue分解前のドラフト）

`docs/spec/ols-spec.md` / `nonlinear-api-design.md`に続く設計ドキュメントの前段として、Issue分解（各モデル20個程度を想定）に入る前に詰めておくべき論点を洗い出す。各論点は「確定 / 未決定」のステータスを埋めていく形で運用する想定。

対象: Phase 3（IV: 2SLS, GMM）、Phase 4（FE, RE）。既存実装（OLS/WLS）の規約（List渡し、`ValidationError`/`ComputationError`の2階層、`cov_type`パターン等）を踏襲することを前提とする。

---

## 1. 共通論点（FE / RE / IV）

### 1.1 引数設計（確定・Issue #119）

- [x] モデル固有の識別子列（エンティティID・時点ID・操作変数列）の渡し方
  - 必須なもの（FE/REの`entity`、IVの`x_exog`/`x_endog`/`instruments`）は`y`/`x`と同格の
    独立引数（案B）。任意なもの（FE/REの`time`）は`Options`内（案A）という使い分けで確定。
    3モデルとも命名はbareネーミング（`_col`サフィックスなし）で揃える。
  - 詳細: [`panel-api-design.md`](./specs/panel-api-design.md)1章、
    [`iv-api-design.md`](./specs/iv-api-design.md)1章
- [x] モデル固有オプションの置き場所（`XxxOptions`構造体に含めるか、別構造体に分離するか）
  - `FeOptions`/`ReOptions`/`IvOptions`という別々の`#[pyclass]`構造体に含める（`OLSOptions`/
    `LogitOptions`の前例を踏襲）。内部実装上の共通化は1.4（#122）で別途検討。
- [x] `weights`/`offset`の扱い（非線形モデルではv1見送り・Phase6再検討という判断を踏襲するか）
  - `offset`は線形モデルのため該当なし。`weights`は汎用オプションとしては見送り（Phase6再検討
    の判断を踏襲）。ただしGLSの共分散構造引数はこの判断の対象外（GLS着手時に別途設計）。

### 1.2 結果（Return）設計（確定・Issue #120）

- [x] 共通コア項目の再定義（`params`/`std_errors`/`t_stats`/`p_values`/`conf_lower`/`conf_upper`/`param_names`はOLSと共通のはず。追加が必要な項目の洗い出し）
  - OLSの項目を土台にしつつ、`n_obs`表記への統一（既存OLSの`nobs`はIssue #139でリネーム済み）、
    `df_resid`/`df_model`・`n_entities`（FE/RE限定）の新規追加、IVでは`log_likelihood`/
    `aic`/`bic`を除外、を確定。詳細: [`panel-api-design.md`](./specs/panel-api-design.md)2.1節、
    [`iv-api-design.md`](./specs/iv-api-design.md)2.1節
- [x] モデル固有の追加結果をどこまで`fit()`の戻り値本体に含めるか、別メソッドに切り出すか
  - IVの第一段階回帰結果は別メソッド（`marginal_effects()`分離方針を踏襲）。FE/REのパネル
    R²・REのハウスマン検定は`fit()`結果本体に含める。
- [x] IV: 第一段階回帰結果の粒度（フルの回帰結果オブジェクトか、係数＋F統計量のみか）
  - `first_stage() -> dict[str, OlsResults]`（内生変数名キー、既存`OlsResults`型を再利用）。
    詳細: [`iv-api-design.md`](./specs/iv-api-design.md)2.2節
- [x] FE/RE: R²の種類（within/between/overall）をどれだけ含めるか、命名規則
  - `r_squared_within`/`r_squared_between`/`r_squared_overall`の3種のみ（bareの`r_squared`は
    廃止）。詳細: [`panel-api-design.md`](./specs/panel-api-design.md)2.3節
- [x] RE: ハウスマン検定を`fit()`内で自動計算するか、独立関数（`hausman_test(fe_result, re_result)`）にするか
  - `fit()`内で自動計算。v1はclassical Hausman検定のみ（`cov_type`非連動）、将来robust版を
    追加できる拡張余地を残す。内部FE推定が失敗した場合はハウスマン関連フィールドを`None`にし
    RE本体の結果は返す。詳細: [`panel-api-design.md`](./specs/panel-api-design.md)2.4節

### 1.3 標準誤差・検定（確定・Issue #121）

- [x] `cov_type`のサポート対象をモデルごとに確定（OLSの`classical`/`hc0-3`/`hac`/`cluster`のうちどれを含める・除外するか）
  - FE/RE/IVとも`classical`/`hc0-3`/`cluster`/`hac`をすべて実装。ただしFE/REの`hac`はOLS流用
    ではなく**Driscoll-Kraay型のパネルHAC**を別途実装する（エンティティを跨いだ時系列相関の
    混同を避けるため）。IVの`hac`はパネル構造を前提としないためOLS流用でよい。詳細:
    [`panel-api-design.md`](./specs/panel-api-design.md)3.1節、
    [`iv-api-design.md`](./specs/iv-api-design.md)3.1節
- [x] 検定分布（t分布 or z分布）の統一方針。OLS系はt分布、非線形（MLE）はz分布という既存判断をFE/RE/IVにどう当てはめるか
  - FE/RE/IVの2SLSはt分布（OLS系）。**IVのGMMのみz分布**（M推定量としての漸近正規性が根拠、
    有限標本のt分布としての正当化がないため）。詳細:
    [`panel-api-design.md`](./specs/panel-api-design.md)3.3節、
    [`iv-api-design.md`](./specs/iv-api-design.md)3.2節
- [x] クラスター標準誤差のデフォルト挙動（パネルではエンティティ単位クラスターが慣行だが、デフォルトにするか明示指定必須にするか）
  - **FE/REは`cov_type`のデフォルト自体を`"cluster"`（entity単位）にする**（OLSの`"classical"`
    デフォルトから意図的に逸脱、fixest前例踏襲）。IVは`entity`のような常在するグルーピング列が
    無いため`"classical"`のまま。詳細: [`panel-api-design.md`](./specs/panel-api-design.md)3.2節

### 1.4 内部実装・共通化（確定・Issue #122）

- [x] OLSエンジンとの共通化の粒度（内部で`OlsEstimator`をそのまま呼ぶか、行列演算のみ共有するか）
  - FEは**まずOlsEstimatorへの委譲を試す**（within変換データを`OlsEstimator::fit`に渡す、
    WLSと同型のパターン）緩い方針で確定。自由度調整・パネルR²・cov_typeデフォルト等の
    FE固有補正は別途必要。詳細: [`panel-api-design.md`](./specs/panel-api-design.md)4.3節
- [x] `rust-style.md`のディレクトリ方針（`panel/`, `iv/`）に対応する`common.rs`の設計（FE/RE間、OLS/IV間でどこまで共有するか）
  - 系統横断で新規に切り出す共通化（t/z検定後処理のジェネリック関数化、`engine_pybind`の
    `cov_type`パース＋列抽出の共通化、`validate_no_duplicate_roles`の複数列ロール対応拡張）を
    決定。OLSのX'Xベース分散計算とnonlinearのHessianベース分散計算は統一しない。詳細:
    [`panel-api-design.md`](./specs/panel-api-design.md)4.2節・4.5節、
    [`iv-api-design.md`](./specs/iv-api-design.md)4章
- [x] 新規エラー型の設計（`OlsError`同様にモデル別`FeError`/`ReError`/`IvError`を作るか、共通型に寄せるか）
  - `LeastSquaresError`/`MleError`の前例に倣い、`PanelError`（FE/RE共有）・`IvError`
    （2SLS/GMM共有）という系統単位の共有エラー型にする。詳細:
    [`panel-api-design.md`](./specs/panel-api-design.md)4.4節

### 1.5 リファレンス実装・テスト方針（確定・Issue #123）

- [x] Python主リファレンスの確定（FE/RE: pyfixest、IV: linearmodels等の候補を検討）
  - **`linearmodels`をFE/RE/IV共通の主リファレンス**とする（`PanelOLS`/`RandomEffects`/
    `IV2SLS`/`IVGMM`）。`pyfixest`は既存方針通り性能比較のみ。詳細:
    [`panel-api-design.md`](./specs/panel-api-design.md)5.1節、
    [`iv-api-design.md`](./specs/iv-api-design.md)5.1節
- [x] R側交差検証パッケージの確定（`run_r_benchmark.R`に`plm`/`ivreg`の枠は用意済み、詳細オプションの対応付けが必要）
  - **RE: `plm`（`model="random"`）、FE: `fixest`（新規スクリプト作成要）、IV: `ivreg`**
    （2SLSのみ）。ハウスマン検定はlinearmodelsに専用メソッドが無い可能性が高いため
    `plm::phtest`のみを参照値とする例外、GMMは`ivreg`が非対応のためPython単独検証の例外を
    それぞれ許容する。詳細: [`panel-api-design.md`](./specs/panel-api-design.md)5.2〜5.3節、
    [`iv-api-design.md`](./specs/iv-api-design.md)5.2〜5.3節
- [x] 許容誤差（既存方針は相対誤差1e-8基本）がFE/RE/IVでも維持できるか（反復計算を伴うGMM等は要検討）
  - 既存方針（相対誤差1e-8基本）を維持し、テスト実装・実測値を見てから個別に緩和を検討する
    （先に緩めない）。詳細: [`panel-api-design.md`](./specs/panel-api-design.md)5.4節

---

## 2. FE（固定効果）固有論点（確定・Issue #124）

- [x] within変換（固体内偏差）の実装方法：polarsのgroup_by機能で固体内平均を引く方式でよいか
  - polarsの`group_by`で実装。2-wayは閉形式の二重デミーニング（バランスパネル限定）。
- [x] 1-way（entity FE）と2-way（entity + time FE）のスコープ（v1をentity onlyにするか）
  - v1から2-wayまでサポート。
- [x] 自由度調整：`n - n_entities - k`への調整方針（2-wayの場合の一般化も含む）
  - 2-wayは`n - n_entities - n_periods + 1 - k`（entity/timeダミー間の1自由度重複を補正）。
- [x] singleton（観測数1のエンティティ）の扱い：自動除外（fixest方式）か、エラーにするか
  - 自動除外せず常に`ValidationError`。2-wayの場合timeのsingletonも同様に検出。
- [x] 不均衡パネルのサポート範囲（v1からサポートするか、バランスパネル前提にするか）
  - **2-wayのみバランスパネル必須（回避オプションなしの常時ハードエラー）。1-wayは不均衡も
    v1からサポート**（1-wayは数学的に不均衡でも正確に成立するため制約しない）。
- [x] 固定効果自体（α_i）の取得方法：`fit()`戻り値に含めるか、別メソッドにするか
  - 別メソッド`fixed_effects()`。1-wayは`dict[str, float]`、2-wayは
    `dict[str, dict[str, float]]`（`"entity"`/`"time"`キー）。
- [x] within変換後に分散ゼロになる説明変数（時間不変変数）の検出・エラーメッセージ
  - 新規バリデーションとして実装（デミーニング後の列分散チェック、1-way/2-way共通ロジック）。

詳細: [`panel-api-design.md`](./specs/panel-api-design.md)6章

## 3. RE（変量効果）固有論点（確定・Issue #125）

- [x] 分散成分の推定方法（Swamy-Arora法 / Wallace-Hussain法など、方式の選定）
  - Swamy-Arora法。R plmのデフォルト・Python linearmodelsの実装いずれも相当し、#123の
    参照実装選定と整合（linearmodelsのソースコードで実装式を確認済み）。
- [x] θ（準偏差変換の重み）計算：バランス/不均衡パネルでの扱いの違い
  - `θ_i = 1 - sqrt(σ_ε² / (T_i・σ_u² + σ_ε²))`。REはentity方向のみ（2-wayスコープ外）なので
    FEの1-wayと同様、不均衡パネルもv1から無条件でサポート。
- [x] ハウスマン検定の実装場所・インターフェース（1.2参照）
  - `RE.fit()`内で自動計算し`ReResult`にのみ含める（FEには追加しない）。計算部分
    （カイ二乗統計量）は`hausman_statistic`として`engine/src/panel/common.rs`に共通関数化。
    linearmodelsに専用実装が無いことをソースコードで確認済み（#123の例外判断の裏付け）。
- [x] FEとの内部設計共有範囲（within/between変換ロジックの共通化）
  - θでパラメータ化した準偏差変換関数`quasi_demean`をFE/REで共有（FEは`θ_i=1.0`の特殊
    ケース）。σ_ε²推定もFEのwithin回帰残差分散を再利用（RE→FE→`OlsEstimator`の委譲チェーン）。
- [x]（追加決定）REのdf_resid: `n - k`（FEの`n - n_entities - k`とは別式、linearmodelsソースで
  確認済み）。

詳細: [`panel-api-design.md`](./specs/panel-api-design.md)7章

## 4. IV（操作変数法）固有論点（確定・Issue #126）

- [x] 引数の切り分け：内生変数（endogenous x）／外生変数（exogenous x）／操作変数（instruments）の3区分をどう引数に落とすか
  - `x_exog`/`x_endog`/`instruments`をすべて独立引数化（Issue #119、1.1節参照）。加えて
    `instruments`は除外操作変数のみとし`x_exog`との重複入力は不可（バリデーションエラー）、
    第一段階設計行列は内部で`x_exog ++ instruments`をunion。詳細:
    [`iv-api-design.md`](./specs/iv-api-design.md)1.1.1節
- [x] 2SLSとGMMの実装方針の違い（GMMの重み行列: 1-step か 2-step efficient GMM か）
  - `weight_type`（点推定の重み行列）と`cov_type`（報告用SE）を分離。`gmm_iterations`
    （デフォルト2＝efficient two-step、1で1-step）を追加。2SLSはGMMの特殊ケース
    （`weight_type="unadjusted"`, `gmm_iterations=1`）として無理のない範囲で共通コード化。
- [x] 丁度識別（just-identified）の場合の扱い：GMMが2SLSに一致するケースを分岐するか共通コードで吸収するか
  - 共通GMM推定コアで自然に吸収（丁度識別では重み行列が点推定に影響しないGMMの一般的性質）、
    特別分岐は不要。
- [x] 弱操作変数診断：第一段階F統計量（Stock-Yogo基準）を結果に含めるか
  - x_exogを直交化した「部分F統計量」として専用計算し`fit()`の結果本体に含める（単純に
    `first_stage()`のOlsResults.f_statisticを流用すると不正確になるため注意）。Stock-Yogo
    臨界値照合・複数内生変数の同時検定はv1スコープ外。
- [x] 過剰識別検定：Sargan検定（2SLS）／Hansen J検定（GMM）を含めるか
  - `fit()`の結果本体に含める。自由度`len(instruments) - len(x_endog)`、丁度識別時は`None`。
- [x] 内生性検定：Wu-Hausman検定（回帰ベース）を含めるか
  - 含める。「回帰ベース」は第一段階残差を構造式に追加回帰する方式
    （linearmodelsの`wooldridge_regression`相当）で実装、`fit()`の結果本体に含める。
- [x] 標準誤差の計算方法（1.3の`cov_type`対応表と連動、2SLS特有のサンドイッチ型分散の扱い）
  - 2SLSは`classical`/`hc0-3`/`cluster`/`hac`（#121で確定済み）。GMMは`weight_type`と独立に
    `cov_type`を選択可能、サポート対象は2SLSと同じ範囲。

詳細: [`iv-api-design.md`](./specs/iv-api-design.md)6章

---

## 5. 次のステップ

1. ~~上記論点を1つずつ「確定」させ、`panel-api-design.md`（FE/RE共通）・`iv-api-design.md`
   （IV単独）としてドキュメント化する~~ **完了**（Issue #119〜#126、全論点確定・ドキュメント化済み）
2. ~~`nonlinear-api-design.md`同様、他パッケージ調査（pyfixest, plm, linearmodels等）のセクションを追加する~~
   各論点の確定時にlinearmodels/plm/fixest/ivregのソースコード・ドキュメントを都度確認する形で
   実施済み（独立セクションとしては追加していないが、`panel-api-design.md`・`iv-api-design.md`
   各所に確認内容を反映済み）
3. **次のステップ**: 設計確定を受け、Issue分解（各モデル20個程度を想定）に着手する
