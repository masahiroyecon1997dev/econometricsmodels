# FE / RE / IV API設計 論点整理（Issue分解前のドラフト）

`ols-api-design.md` / `nonlinear-api-design.md`に続く設計ドキュメントの前段として、Issue分解（各モデル20個程度を想定）に入る前に詰めておくべき論点を洗い出す。各論点は「確定 / 未決定」のステータスを埋めていく形で運用する想定。

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
  - OLSの項目を土台にしつつ、`n_obs`表記への統一（既存OLSの`nobs`は別Issueでリネーム）、
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

### 1.3 標準誤差・検定

- [ ] `cov_type`のサポート対象をモデルごとに確定（OLSの`classical`/`hc0-3`/`hac`/`cluster`のうちどれを含める・除外するか）
- [ ] 検定分布（t分布 or z分布）の統一方針。OLS系はt分布、非線形（MLE）はz分布という既存判断をFE/RE/IVにどう当てはめるか
- [ ] クラスター標準誤差のデフォルト挙動（パネルではエンティティ単位クラスターが慣行だが、デフォルトにするか明示指定必須にするか）

### 1.4 内部実装・共通化

- [ ] OLSエンジンとの共通化の粒度（内部で`OlsEstimator`をそのまま呼ぶか、行列演算のみ共有するか）
- [ ] `rust-style.md`のディレクトリ方針（`panel/`, `iv/`）に対応する`common.rs`の設計（FE/RE間、OLS/IV間でどこまで共有するか）
- [ ] 新規エラー型の設計（`OlsError`同様にモデル別`FeError`/`ReError`/`IvError`を作るか、共通型に寄せるか）

### 1.5 リファレンス実装・テスト方針

- [ ] Python主リファレンスの確定（FE/RE: pyfixest、IV: linearmodels等の候補を検討）
- [ ] R側交差検証パッケージの確定（`run_r_benchmark.R`に`plm`/`ivreg`の枠は用意済み、詳細オプションの対応付けが必要）
- [ ] 許容誤差（既存方針は相対誤差1e-8基本）がFE/RE/IVでも維持できるか（反復計算を伴うGMM等は要検討）

---

## 2. FE（固定効果）固有論点

- [ ] within変換（固体内偏差）の実装方法：polarsのgroup_by機能で固体内平均を引く方式でよいか
- [ ] 1-way（entity FE）と2-way（entity + time FE）のスコープ（v1をentity onlyにするか）
- [ ] 自由度調整：`n - n_entities - k`への調整方針（2-wayの場合の一般化も含む）
- [ ] singleton（観測数1のエンティティ）の扱い：自動除外（fixest方式）か、エラーにするか
- [ ] 不均衡パネルのサポート範囲（v1からサポートするか、バランスパネル前提にするか）
- [ ] 固定効果自体（α_i）の取得方法：`fit()`戻り値に含めるか、別メソッドにするか
- [ ] within変換後に分散ゼロになる説明変数（時間不変変数）の検出・エラーメッセージ

## 3. RE（変量効果）固有論点

- [ ] 分散成分の推定方法（Swamy-Arora法 / Wallace-Hussain法など、方式の選定）
- [ ] θ（準偏差変換の重み）計算：バランス/不均衡パネルでの扱いの違い
- [ ] ハウスマン検定の実装場所・インターフェース（1.2参照）
- [ ] FEとの内部設計共有範囲（within/between変換ロジックの共通化）

## 4. IV（操作変数法）固有論点

- [x] 引数の切り分け：内生変数（endogenous x）／外生変数（exogenous x）／操作変数（instruments）の3区分をどう引数に落とすか
  - `x_exog`/`x_endog`/`instruments`をすべて独立引数化（Issue #119、1.1節参照）。加えて
    `instruments`は除外操作変数のみとし`x_exog`との重複入力は不可（バリデーションエラー）、
    第一段階設計行列は内部で`x_exog ++ instruments`をunion。詳細:
    [`iv-api-design.md`](./specs/iv-api-design.md)1.1.1節
- [ ] 2SLSとGMMの実装方針の違い（GMMの重み行列: 1-step か 2-step efficient GMM か）
- [ ] 丁度識別（just-identified）の場合の扱い：GMMが2SLSに一致するケースを分岐するか共通コードで吸収するか
- [ ] 弱操作変数診断：第一段階F統計量（Stock-Yogo基準）を結果に含めるか
- [ ] 過剰識別検定：Sargan検定（2SLS）／Hansen J検定（GMM）を含めるか
- [ ] 内生性検定：Wu-Hausman検定（回帰ベース）を含めるか
- [ ] 標準誤差の計算方法（1.3の`cov_type`対応表と連動、2SLS特有のサンドイッチ型分散の扱い）

---

## 5. 次のステップ

1. 上記論点を1つずつ「確定」させ、`panel-api-design.md`（FE/RE共通）・`iv-api-design.md`（IV単独）としてドキュメント化する
2. `nonlinear-api-design.md`同様、他パッケージ調査（pyfixest, plm, linearmodels等）のセクションを追加する
3. 設計確定後、Issue分解（各モデル20個程度）に着手する
