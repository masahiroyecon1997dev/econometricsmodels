# 固定効果IV（FE-2SLS / FE-GMM）設計ノート

FEIV（固定効果IV）の入力型・API・実装方針の設計。実装途中の設計ノートで、実装が完了したら
`docs/spec/fe-iv-spec.md`へ集約する（CLAUDE.md 13章）。ロードマップの決定事項は#276、IV本体側の
先行作業（弱操作変数診断・C統計量）は#363〜#369、本ノートは#370の成果物。

## 0. 前提と今回の改訂点

#276で確定済み（変えない）:

- 実体は「within変換 + 2SLS/GMM」で線形。初版は単一/2-way FEまで（HDFEは対象外）。
- モジュールは`engine/src/panel/fe_iv.rs`（panel側）。
- 自由度補正はfixest準拠（`df_resid = n - k_structural - df_absorbed`、`panel/fe.rs`の`neffects`と同じ）。
- First-differences IV、RE-IV/EC2SLS、Hausman-Taylorは対象外。

今回改訂する点（#608・#609の調査で判明）:

- **「既存の`iv::two_sls`コアをそのまま再利用する」は成り立たない。** `TwoSlsEstimator`と`compute_first_stage`
  は自由度が`n - k`固定で、吸収した固定効果の数を渡せない（`OlsEstimator`が口を持たない）。
- 代わりに、**第一段階は内生変数ごとに`FeEstimator`で推定する**（`x_endog[j] ~ x_exog + instruments`を
  同じ固定効果で回帰）。自由度・`cov_type`・F統計量が自動的にFE準拠になり、`first_stage()`も
  `FEResults`で返せる。標準誤差は`panel_cov_params`に`X̂`と構造残差を渡す。
- そのため`iv::`から使う計算コアはほぼ無く、panel→ivの依存は最小になる（エラーコードの再利用程度）。
- #372（自由度補正）の内容は本設計の`FeIvEstimator::fit`に最初から入る。

## 1. engine

### 1.1 入力型

```text
panel/common.rs   PanelIndex { entity: GroupCodes, time: Option<TimeKeys>, effect_names: EffectNames }
                  PanelIndex::new(nobs, entity, time)  // 観測数との整合を検証して構築
panel/fe_iv.rs    FeIvInput { y, dep_var_name, x_exog, x_endog, instruments (列スライス Vec<Vec<f64>>),
                              各役割の列名, index: PanelIndex }
                  RoleColumns<'a> { names: Vec<String>, columns: &'a [Vec<f64>] }
                  FeIvInput::from_group_codes(y, dep_var_name, x_exog, x_endog, instruments, entity, time)
```

- `PanelIndex`は`FeInput`・`ReInput`が持つ索引部分（entity・time・効果名）の切り出し。`FeIvInput`が最初に
  採用し、`FeInput`/`ReInput`は別Issue（FEIV実装後）で寄せる。`validate_panel_input`の次元検証は
  列の長さ検証（`validate_column_lengths`）と索引の検証（`validate_panel_index`、`PanelIndex::new`が使う）に分けた
  （検査の順序は従来どおり）。
- 構築の引数は、役割ごとの列と名前を`RoleColumns`にまとめて渡す（引数の数を抑える）。第一段階用の
  `FeInput`への変換（`PanelIndex`の複製を伴う）は#371で足す。
- 定数項は持たない（固定効果が吸収する）。`x_exog`は空でもよい。`x_endog`・`instruments`は非空
  （空は`ROLE_EMPTY`、`build_*`側で検証。IVと同じ）。
- 列は`faer::Mat`ではなく列スライス（`FeInput`と同じ。`WithinTransform::apply`が列単位のため）。

### 1.2 `FeIvEstimator::fit(input, effects, cov_type, confidence_level)`

`effects: PanelEffects`、`cov_type: PanelCovType`（`FeEffects`/`FeCovType`と同じ型）。

1. `ensure_serial()`。2-wayなら`time`の有無を確認（`TwoWayRequiresTime`）。
2. `InsufficientInstruments`（`len(instruments) < len(x_endog)`、カタログの既存code）。
3. singleton検出（`ensure_no_singleton_groups`）、2-wayはバランスパネル検証（`validate_balanced_panel`）。
   先行するFE拡張（5章）でバランスパネル検証は「重複ペアの検査」に置き換わり、singletonの扱いも5.2のとおり
   確定させる。この手順は`FeEstimator`の事前検証と同じ関数を呼ぶ形に保ち、FEIV側で複製しない。
4. `WithinTransform`で`y`・`x_exog`・`x_endog`・`instruments`の全列を変換。
5. 変換後の分散ゼロ検出（`ensure_no_zero_variance_columns`、全役割。`x_exog`・`x_endog`に加え
   `instruments`も共線になるため）、`y`のentity内定数（`OutcomeConstantWithinEntity`）。
6. 自由度検証: `n > k + neffects`（構造式）かつ`n > l + neffects`（第一段階、`l = k_exog + k_instruments`）。
   構造式を先に報告する（IVの`validate_observation_counts`と同じ層分け）。`k = k_exog + k_endog`。
   `neffects`は`FeEstimator`と同じ（1-way: `n_entities`、2-way: `n_entities + n_periods - 1`）。
7. 第一段階: 内生変数ごとに`FeEstimator::fit`（`y=x_endog[j]`、`x=x_exog ++ instruments`、同じ`effects`・
   `cov_type`）。失敗は`stage=FIRST_STAGE`付きで返す（`PanelError::InStage`）。`X̂ = [x̃_exog, x̂̃_endog]`
   （内生列は第一段階の当てはめ値）。
8. 点推定: `β̂ = (X̂'X̂)⁻¹X̂'ỹ`（`least_squares(X̂, ỹ)`の`params`・`xtx_inv`）。
9. 構造残差`e = ỹ - X̃β̂`（`X̃ = [x̃_exog, x̃_endog]`は実際の内生変数。第二段階の残差ではない）。完全フィット
   検査は構造残差で行う（IVと同じ）。
10. `panel_cov_params(cov_type, PanelCovInput { x: X̂, residuals: e, xtx_inv, df_resid, df_model: k + neffects,
    n_slopes: k, entity, structure: FixedEffects { two_way_time } })`。
11. `inference_table`（t分布、`df_inference`）、`wald_f_test(β̂, cov, 0, k, df_inference)`。

`df_resid = n - k - neffects`、`df_model = k + neffects`（`FeEstimator`と同じ定義）。

### 1.3 `cov_type`

FEの実際のセット（Classical / HC1 / HC2 / HC3 / Cluster / DK）。Newey-West HACは持たない（FEにも無い。時系列相関は
DKで扱う）。DKは`panel_driscoll_kraay_cov_params`を`panel_cov_params`経由で使う（#371の決定どおり、素の
`iv::`には追加しない）。未決: 2SLSのHC2/HC3のレバレッジ定義（`X̂`ベースの素のレバレッジ + FEの`1/T_i`補正を暫定）と、
fixestがIVでHC2/HC3を提供するか。fixestはIVでHC2/HC3を提供しない（実測）。**HC2/HC3は他手法と`cov_type`を揃えるため残す**（7章1）。
レバレッジ定義は#379でplmの出力と手計算で確定する。

### 1.4 診断（後続Issueで実装、ここでは取り込み方だけ決める）

- 弱操作変数（#373）: 第一段階の部分F（`SSR_u`・`SSR_r`は変換後の列、分母自由度は`n - l - neffects`）、CD/KP/MOP。
- Sargan（#374）: 構造残差を`Z̃ = [x̃_exog, z̃]`に回帰した`n·R²`相当。`σ̂² = e'e/n`。自由度`len(instruments) - len(x_endog)`。
- Wu-Hausman（#375）: `ỹ ~ [x̃_exog, x̃_endog, v̂]`の補助FE回帰（`v̂`は第一段階の残差）、`v̂`列のWald検定。

### 1.5 エラー

`PanelError`にバリアントを足し、既存のカタログcodeを再利用する（`INSUFFICIENT_INSTRUMENTS`等。同じcodeは
同じhint）。第一段階の失敗は原因のcodeに`stage=FIRST_STAGE`を付ける（`InStage`）。新しいcodeが必要になったら
`rust-style.md`のエラー追加手順に従う。

定数列のエラーは、生データで定数か、固定効果を除くと一定になるかで分ける（5.1）。変換後の分散ゼロ検出
（手順5）は`y`・`x_exog`・`x_endog`・`instruments`の全役割に対し、この2分類で報告する。

## 2. engine_pybind / Python

```python
FEIV(data, y, x_exog, x_endog, instruments, effects, *, options: FEIVOptions | None = None)
```

必須の列は`data`の直後（`api-conventions.md`1章）。クラス名は`FEIV`/`FEIVOptions`/`FEIVResults`
（既存の`FE`・`IV`の命名に合わせる。#377の「FeIvOptions/FeIvResults」表記は直す）。

`FEIVOptions`（全引数キーワード専用）:

| フィールド | 内容 |
|---|---|
| `estimator` | `"2sls"`（既定）/ `"gmm"`（#381で有効化） |
| `cov_type` | 既定`"cluster"`（FEと同じ。クラスターキーは`effects[0]`） |
| `cluster` / `dk_time` / `dk_bandwidth` | FEと同じ（`cov_type`に応じた条件付き必須） |
| `confidence_level` | 既定0.95 |
| `gmm_*` | #381で追加（`IVOptions`と同じ名前・規則） |

`include_intercept`は持たない。`FEIVResults`: FEの係数・標準誤差・推論・`df_*`・`n_groups`・`f_statistic`系・
`dk_bandwidth_used`に、IVの診断（`first_stage()`→`dict[str, FEResults]`、`weak_instrument_f_statistics`、
`overid_*`、`wu_hausman_*`）を足す。`fixed_effects()`は`α_i = ȳ_i - x̄_i'β̂`（`β̂`は構造の係数、`X`は実際の内生変数）。

## 3. 未決事項（実装時に確認）

1. 2SLSのHC2/HC3のレバレッジ定義と、fixestでの参照可否（1.3）。
2. 報告する適合度: `r_squared_within`等（構造残差ベース。IVのR²は意味が限定的）。`log_likelihood`・AIC/BICは出さない案。
3. 第一段階を内生変数ごとに`FeEstimator`で回すと、`x_exog`・`instruments`のwithin変換が`k_endog`回繰り返される。
   問題になる規模なら、変換済みの列を渡す内部経路を足す（性能は#380で計測してから判断）。
4. FE-GMMの重み行列（`classical`/`robust`/`cluster`、`kernel`は`dk_time`で時間順を渡す案）はFE-GMM（#381）で確認する。
5. 変換後に分散ゼロになった`instruments`列のエラー（既存の`ZeroVarianceAfterDemeaning`で列名を示す案）。
   5.1の2分類に合わせて、役割（`x_exog`・`x_endog`・`instruments`）を問わず同じコードで返す。
6. Sarganの`σ̂²`の分母。fixestの`fitstat(m, "sargan")`は`e'e/n`（`n`は観測数、吸収した効果の数は引かない）で、
   手計算と完全に一致する（N=100・T=6で`e'e/(n-N)`だと5.617が4.681になる）。#374は参照と揃えて`e'e/n`にする。
   ただし within変換後の誤差は`n-N`相当の分散しか持たないため、小Tでは漸近論の上で過大になりうる。
   `docs/methods/`に注記する（実装は参照実装に合わせ、差は文書で説明する）。
7. FE-GMMの`robust`重みと`Hansen J`。within変換した誤差は個体内で相関するため、非クラスターの
   重み行列とHansen Jは小Tで妥当でない可能性が高い（未検証）。FE-GMMの既定はクラスター重みにし、
   `robust`は注意書き付きで提供する案。#381で確認する。
8. 参照実装。fixest 0.14.2の`fitstat`はFE付きIVに対して`ivf`・`ivwald`・`sargan`・`wh`（Wu-Hausman）・
   `kpr`（Kleibergen-Paap）・`cd`（Cragg-Donald）を返す。#373〜#375の主参照にでき、
   ivDiagは弱操作変数診断（MOP）の補助にとどまる。#379で各統計量の定義（自由度・分母）を実測して決める。
9. FE-GMMと`iv::gmm`の関係。`GmmEstimator`も自由度が`n - k`固定（古典`σ̂²`・HC1補正・クラスター補正）で、
   吸収した効果の数を渡せない。(a)`gmm_iterate`・`sandwich_covariance`を行列受け取り・自由度引数付きに
   改めて共有するか、(b)FEIV側に別実装を持つかを#381の前に決める。(a)ならCUE・Windmeijer補正を二重に
   書かずに済む。

## 4. 子Issueへの反映

- #371: 「`iv::two_sls`コアの再利用」を本ノート1.2（`FeEstimator`で第一段階・`panel_cov_params`で標準誤差）に直す。
- #372: 1.2の6・10に含まれるため、独立した実装は不要（テスト・文書化のみ）。
- #373〜#375: 1.4の取り込み方に沿う。#373は#363〜#365（5.3の制約付き）の後。#374の`σ̂²`は3章6、
  #375の参照はfixestの`wh`（3章8）。
- #377: クラス名を`FEIV`系に直す。
- #379: 参照は`fixest`の`feols`と`fitstat`（3章8）。HC2/HC3の可否（1.3）もここで決める。
- #381: 入力・within変換・第一段階は本ノートを共有し、点推定をGMMに置き換える。`iv::gmm`との関係は3章9、
  重み行列の既定は3章7。

## 5. 先行する拡張と着手順（2026-11-03の検討）

FEIVの前に、FE・IVの拡張をどこまで済ませるかを整理した。以下のうち「決定」は合意済み、「方針案」「提案」は確認待ち。

### 5.1 決定

- **FEの拡張を先に実装する**: 不均衡パネルの2-way（交互射影）を先に、続けて高次元固定効果（3次元以上）。
  交互射影の核はK個のグループコード配列で書く。不均衡2-wayの実装ではK=2だけを呼び、高次元固定効果で
  `effects`の長さの上限を外す。FEIVはこの変換を使い、2-wayのバランスパネル検証（1.2の手順3）は置き換わる。
- **REの拡張は前提にしない**: 不均衡2-way RE・1-wayの`swamy_arora`の`plm`準拠はFEIVと無関係。ただしFEの
  事前検証や変換が変わるとREの内部FEが継承するため、FEの実装時にREへの影響を確認する。
- **IVの重み付けは前提にしない**: FEIVは`iv::two_sls`のコアを使わない（0章）。重み付きFEIVに要るのは
  FEの重み付け（別Issue）であり、IV側の重み付けではない。ただし5.3の制約Aを守れば、後から足せる。
- **IV本体の弱操作変数診断とC統計量は先に実装する**: #373は#363〜#365の後、FE-GMMの#384は#368・#369の後。
- **定数列のエラーは2分類**: 「生データで定数」と「固定効果を除くと一定」。被説明変数は既存の
  `OUTCOME_CONSTANT`と`OUTCOME_CONSTANT_WITHIN_ENTITY`を使い分け、説明変数（`x_exog`・`x_endog`・
  `instruments`）には生データ版を足す（現状は`REGRESSOR_NO_WITHIN_VARIATION`が両方を兼ねる）。
  コード名と文面はエラー追加手順（`rust-style.md`）で決める。次元別の原因（どの効果の中で一定か）は
  後から`params`に足せるので、初版では出さない。
- **共通化は独立Issueで、FEIVの実装（#371）より前に行う**（5.3）。

### 5.2 グループsingleton（決定）

観測が1つしかないグループは、これまでどおりエラーにする（欠測値を自動除外しない方針と揃える）。
不均衡2-wayの検討で出た「singletonの反復除去」は取り消す（#522・#442の記載は改める）。

- 除去しないので、行の対応（`residuals`の長さ、cluster・DKのキー、`predict`、`fixed_effects()`）が
  入力の行と常に一致する。FEIVは役割列が4種あり、全列で同じ行を落とす仕組みが要らなくなる。
- 理由は統計ではなく方針の一貫性: 除去しても`β̂`と古典SEは変わらない見込みだが、`nobs`が利用者の
  知らないところで変わる点は欠測値と同じ。
- 不均衡パネルでは、除去で新しいsingletonが生まれる（連鎖）。エラー時に、連鎖を含めて何行落ちるかを
  `params`に出し、`hint`でpolarsのフィルタ例を示す案（欠測値のエラーと同型）。
- fixestはOLSでsingletonを既定で除く（`fixef.rm`）。比較フィクスチャはsingletonを含まないデータで作る。
- 次元が増えても規則は同じ（`effects`のどの列でも、水準ごとの観測数が2以上）。

### 5.3 共通化（独立Issue #628、FEIVの前）

2SLS・GMM・FEIVの3つ目の呼び出し元が出るため、`iv/CLAUDE.md`の方針（3つ目が出てから広げる）に沿う。

- **A. 共分散と検定のロジック**: 「独立実装」の範囲を、スコアに依存する部分（HCのレバレッジと小標本補正の
  分母）に限る。古典・HACのカーネル和・Waldの形は`shared/`に寄せる（`HcVariant`の複製も解消）。
  IVのWald統計量の対称化の有無（約1e-7）が出るので、挙動を保つ整理とは別にコミットする。
- **B. 行列を受け取る関数に切り出す**: 2SLSの点推定、Sargan（`σ̂²`の分母を引数にする）、部分F。
  `IvInput`・`OlsEstimator`には依存させない。Wu-Hausmanの拡張回帰は推定器が違うので、列の組み立てだけ共有する。
- **C. #363〜#365は最初から行列受け取りで書く**:
  1. `(Y, X_exog, Z)`を受け取る。重みは√wの事前スケールで足せる。定数項は平均除去ではなく列として偏回帰で落とす。
  2. 吸収自由度`n_absorbed`（既定0）を持つ。CD・KP・部分Fの分母は`n - l - n_absorbed`。
  3. KPのmeatは共有の関数（クラスター・HAC）を使う。FEIVの`cov_type`はDKを含むため、KPでDKをどう扱うか
     （#364の対象はhc・cluster・hac）を#364で決める。

実装順（提案）:

1. 共通化のA（独立Issue。#613の判断）。
2. 共通化のB、IV本体の#363〜#365（Cの制約付き）。B・Cは並行できる。
3. FEの拡張: #522 → 高次元固定効果（#442）。IV側（1・2）とは独立なので並行できる。#522に入る前に、
   5.4（`PanelIndex`の形）を決める（5.2のsingletonは決定済み）。
4. IV本体の残り: #366・#367（バインディングとivDiag照合）、#368・#369（C統計量）。#366・#367は
   #373の公開と照合まで、#368・#369はFE-GMMの#384までに終える。
5. FE-2SLS: #370〜#372、#374、#375 → #373（#363〜#365の後）→ #376〜#380。FEの拡張（3）の後。
6. FE-GMM: #381〜#389（3章9の判断が先）。

### 5.4 FEIV設計への反映（提案）

- **`PanelIndex`をK次元にする**: 現在は`entity`と`time: Option<TimeKeys>`の2次元で、`effects: list[str]`の
  採用と高次元固定効果に合わない。`Vec<GroupCodes>`と列名の形にし、DKの時点キーは効果の次元とは別に持つ
  （`dk_time`は`effects`と独立）。固定効果の復元の基準時点（時間順で最初）は、時間順を持つ次元にだけ要る。
- **変換済み列を渡す内部経路を最初から設計する**: 反復法のためwithin変換が高コストで、第一段階を内生変数ごとに
  回すと変換が繰り返される（3章3は「#380で計測」としていたが前倒し）。収束許容度は4種の列すべてに効く。
  グループ索引・連結成分は列間で共有する。

## 6. 実装前に決める事項（API・テスト・性能）

2026-11-03の検討で洗い出した。「実測」はdevcontainer（fixest 0.14.2・plm 2.6.7）で確かめた。

### 6.1 API・結果

1. **統計量の命名と分布（決定）**: FEは`f_statistic`系（t/F固定）、IVは`wald_*`＋`stat_dist`（t/normal、f/chi2）。
   FEIVは2SLS=t/F、GMM=normal/chi2を両方取るので、IVに合わせて`wald_*`系と`stat_dist: "t" | "normal"`にする案。
   （2章の「FEの`f_statistic`系に足す」は更新が要る。）
2. **GMMのオプション**: `IVOptions`と同名（`gmm_type`・`gmm_weight_type`・`gmm_max_iter`・`gmm_tol`・
   `raise_on_non_convergence`、使わないモードでの指定は`ValidationError`）。決める点は、`gmm_weight_type`の語彙
   （classical/robust/clusterに加え、HAC相当をどう持つか。FEIVに`hac_time`は無く`dk_time`がある）と、
   `cluster`を`cov_type`と重み行列で共用する際の既定（`effects[0]`）。
   **決定**: `gmm_weight_type`の語彙は初版は`classical`・`robust`・`cluster`の3つ。DK型の重み（時点ごとに集計した
   モーメントにカーネルを掛ける）は作らない。IVの`hac`重みは行単位のNewey-Westで`hac_time`が全行で異なる値を
   要するためパネルに使えず、DK型は重み行列のランクが時点数T以下になるため`l`より大きいTを要し、ミクロパネル
   （T=6前後）では実用にならない。必要になれば値を足せる（互換）。**FEIVのGMMの重みの既定は`cluster`
   （`effects[0]`）**。IVの既定（`classical`）と異なる。`classical`の2段階GMMは2SLSと同じ結果になり、
   `robust`は小Tで妥当でない可能性が高い（3章7）ため。`cov_type`の既定が`cluster`であることとも揃う。
3. **C統計量の公開形**: 疑わしい変数を利用者が選ぶので、規約上は事後診断のメソッド（`DiagnosticResult`）で、
   `fit()`では計算しない。メソッドが元データを要するため、`IVResults`が`source_data`を保持するかを#368で決め、
   FEIVも同じ形にする。
   **決定**: 事後診断のメソッドにし、そのために結果が次を保持する。(a)`fit()`に渡したDataFrameへの参照
   （`source_data`。`OLSResult`が同じ仕組みを持つ。参照を増やすだけでコピーしない。使った列だけを`select`した
   ものを持てばpickleが小さくなる）、(b)列の役割と解決済みのオプション、(c)全モデルの重み行列`S_used`
   （l×l）とHansen J（差の検定で同じ`S`を使う場合。#368の判断に依存）。メソッドは呼び出し時にGMMを2回
   推定し直すので、設計行列は保持しない。`fixed_effects()`も同じ`source_data`から必要なときに計算する
   （`FeEstimator`を保持しない）。
4. **R²**: 実測で、fixestの`wr2`（0.233）は**第二段階の残差**ベースで、構造残差ベース（0.746）と異なる。
   本ノート3章2の「構造残差ベース」を採ると、fixestでは照合できず手書きの実装が唯一の参照になる
   （`verification-details.md`の単一リファレンスの例外に載せる）。構造残差ベースで出すか、出さないかを決める。
   `log_likelihood`・AIC/BICは出さない（3章2のとおり）。
   **決定**: 構造残差ベース。IVの`r_squared`が`1 - SSR_構造残差/SST`であること（`two_sls/fit.rs`）と揃う。
   fixestの値（第二段階の残差ベース）は照合に使えず、`verification-details.md`に「定義が異なる」と注記する。
   **決定**: `r_squared_within`・`r_squared_between`・`r_squared_overall`の3つを、`FEResults`と同じ
   フィールド名で出す。IVの`r_squared`は単一フィールド（3分割しない）だが、FEIVは`FEResults`の結果の形を
   引き継ぐパネル推定量なので、FEと同じ3つに揃える。定義は`β̂`に構造の係数、`X`に実際の内生変数を使い、
   within=`1 - e'e/ỹ'ỹ`、between=エンティティ平均に`β̂`を当てはめ、overall=元の`y`・`X`に`β̂`だけを当てはめた残差。
   FEの`fe_r_squared_between`・`_overall`と式は同じで、引数だけが違う。3つとも`linearmodels`の
   `PanelOLS._rsquared(params)`（`params`に任意の係数を渡せる。非公開メソッドなのでフィクスチャ生成側だけで、
   バージョンを固定して使う）で独立に照合できる。値は負や1超になりうる旨をdocstringに書く。`adj_r_squared`は出さない。
5. **`fixed_effects()`**: 実測で、fixestの`fixef`は`mean(y - X β̂)`（`X`は実際の内生変数）に中心化の差を除いて一致
   （差2e-15）。1.2・2章の定義（`α_i = ȳ_i - x̄_i'β̂`）で確定してよい。
6. **`first_stage()`**: `dict[str, FEResults]`にするには、`FeEstimator`から`FEResult`を作る変換が
   `engine_pybind`に要る（IVの`ols_estimator_to_result`に相当）。`FEResult`の`fixed_effects()`は
   `FeEstimator`を保持するので、内生変数の数だけ保持されるメモリと、第一段階側で`fixed_effects()`を許すかを決める。
   IVの第一段階が二重に計算される問題（`engine_pybind/src/iv/CLAUDE.md`）は、FEIVでは最初から避ける
   （第一段階を1回だけ推定し、`X̂`の作成と`first_stage()`の両方に使う）。
   **決定**: 第一段階の結果を作るとき、固定効果の辞書をその場で計算して`LazyFixedEffects::Computed`にする
   （`FeEstimator`を保持しない。辞書は個体数に比例する小さなもの）。残るのは内生変数ごとの残差（8nバイト）と
   スカラー・係数類だけで、`fixed_effects()`も第一段階で使える。`FeEstimator`から`FEResult`を作る変換は
   `engine_pybind`内の`pub(crate)`関数で、Pythonから見える面は増えない（`first_stage()`は既存の公開クラス
   `FEResults`を返す）。
7. **名前と置き場**: `FEIV`/`FEIVOptions`/`FEIVResults`、`python_package/.../panel/fe_iv.py`。docsのnavは
   Panel models配下に置く案（実装・テストが`panel`のため）。
8. **重み**: FEに重みが入る（別Issue）とき、`weight`を`effects`の後ろ（`*`の前）に足せる位置を空けておく
   （追加は互換）。
9. **エラー**: 5.1の2分類のとき、役割（`x_exog`・`x_endog`・`instruments`）を`params`に載せる。個体内で一定の
   操作変数（時間不変）が典型的な誤りなので、`hint`で指摘する。

### 6.2 テスト

1. **配置**: `tests/panel/test_fe_iv_*.py`、`benchmark/panel/`（`datasets.py`にFEIV用のDGP、
   `fixtures/generate_fe_iv_fixtures.py`、`references/`にfixest・plmのアダプタ）。許容誤差は
   `tests/support/_tolerances_panel.py`にFEIVの行を足し、`docs/guide/verification.md`の表にも足す。
2. **FE-2SLSの参照**（実測）:
   - 主: `fixest`（係数、classical/hc1/cluster/dkのSE、F、Sargan・Wu-Hausman・CD・KP・第一段階Fは`fitstat`）。
   - 独立クロスチェック: **`plm(y ~ x2 + x | x2 + z1 + z2, model = "within")`が動く**。係数はfixestと一致、
     classical SEも一致、clusterは`vcovHC(method = "arellano")`が小標本補正で異なる（`type = "sss"`で揃える方針は
     RE/FEのplm照合と同じ）。ノートにある「plm+ivreg」のivregは不要（`ivreg`はパネル非対応）。
   - 小標本のオラクル: `linearmodels.IV2SLS`にエンティティダミーを明示した2SLS（LSDV）。係数・古典SE・
     Sargan・Wu-Hausmanを`n_absorbed`と合わせて照合する。
   - **`ivDiag`と`gmm`はdevcontainerに未導入**（実測）。`Dockerfile`の`install_version()`で追加し、
     `check_r_updates.yml`の対象にする（CLAUDE.md 10章）。CD/KP/Sargan/Wu-Hausmanは`fitstat`でまず足りるので、
     `ivDiag`はMOPの有効F（`eff_F`）に要る。
   - 要調査: fixestの`kpr`（Kleibergen-Paap）が、同じデータで`ivf`=184・`cd`=220.7に対して9.07と桁が違う。
     rk LMかrk Wald Fか、クラスターの扱いかを#364で確認してから参照に使う。
     **#364で確認済み**: `kpr`はiid、または内生変数と操作変数が同数のときだけ実装され、rk LM・rk Wald Fのどちらにも
     一致しない（`iv-spec.md`3.4.2節）。KPの参照にはlinearmodels（FEIVはLSDVの2SLS）を使い、fixestの`kpr`は照合に使わない。
   - 実測（N=100・T=6のデータ）: **plmの`vcovHC(method = "white1", type = "HC0")`は、within IVで構造残差と`X̂`を
     使ったサンドイッチと完全に一致**（比1.000000）。第二段階の残差を使うと0.57〜0.59倍になり一致しない。
     fixestの`vcov = "hetero"`はその`√(n/df_resid)`倍（1.0976 = √(600/498)）で、FEのHC1（`n/df_resid`）と同じ。
     **fixestはIVのHC2・HC3を提供しない**（エラー）。plmはHC0〜HC3を返す。
   - Wu-Hausman: fixestの`wh`は、第一段階の残差`v̂`を足した補助FE回帰の`v̂`のt²と一致（62.797）。iid前提。
3. **FE-GMMの参照**: #388は`plm`＋`gmm::gmm()`を第一候補とするが、`gmm`は未導入。**`momentfit`は導入済みで
   IV GMMのクロスチェックに使っている。** 提案: 主は`linearmodels.IVGMM`を、pandasで自前にdemeanした
   データ（切片なし）に適用し、係数・重み行列・Hansen Jを照合する（SEは自由度補正が違うので補正を揃える）。
   クロスチェックは`momentfit`。
   **決定**: この方針で進める。自前の部分は2つで、(1)within変換（pandasのgroupby平均。不均衡2-wayと高次元は
   小標本のダミー行列への`numpy.linalg.lstsq`による厳密な射影で、本体の交互射影とは別の方法）、
   (2)標準誤差の小標本補正のスカラー倍。係数・重み行列・Hansen J・反復回数は自前処理なしで照合できる。
   (2)は丁度識別での2SLSとの一致（fixestで検証済みのFE-2SLSのSE）で独立に押さえる。この2つはレビューで
   注意深く見る（実装時の`/review-testing`の観点に入れる）。
4. **シナリオ（追加なしで確定）**（`testing-policy.md`の観点に加えてFEIV固有）: 識別（丁度識別・過剰識別1〜3本、内生変数1〜3個、
   `x_exog`が空）、操作変数の強弱（弱い操作変数）、無効な操作変数（Sargan/Hansen Jが棄却）、内生性の有無
   （Wu-Hausman）、時点数T（2・3・6・20。DKはT依存）、クラスターの偏り、1-way・2-way（不均衡は#522後）・
   3-way（#442後）、時間不変の操作変数・説明変数のエラー、完全に当てはまる第一段階。実データはWooldridgeの
   FE-IVの教科書例（`jtrain`を候補とするが、変数の定義を要確認）。
5. **cross_methodへの登録**（実装時の漏れ防止）: `test_call_signatures`（api-conventionsの表に1行）、
   `test_results_docstrings`、`test_pickle`の`SAMPLES`、`test_gil_release`の`FITS`、`test_cov_params`の
   `COV_TYPES_BY_METHOD`/`METHOD_OPTIONS`、`test_validation_order`、`test_keyword_only_arguments`、
   `test_doc_examples`・`test_docstring_examples`・`test_public_docs`。

### 6.3 性能

1. **比較パッケージ**: `linearmodels`にFEIVは無く、pyfixestは性能比較からも外した（`docs/performance/ols.md`）。
   案: (a)`linearmodels`の合成（pandasのgroupby demean＋`IV2SLS`。demeanは計測区間に含める）、
   (b)pyfixestの復活、(c)engine単独。
   **決定**: (c)比較なし（engine単独）。Tobitと同じ（`PerfAdapter.libraries=("engine",)`）。(a)はpandasの
   groupbyが計測の大部分を占め、不均衡2-wayや高次元では参照側に交互射影を手書きする必要も出て恣意的になる。
   (b)はOLSで外した方針と矛盾する。性能ノートに「ライブの相対比較が無い」理由を明記する。
2. **ファイル構成（決定）**: #380・#389は`compare_fe_2sls`・`compare_fe_gmm`の2本だが、`benchmark_performance.yml`の
   matrixは手法単位で、IVはestimatorをmethod軸（variant）で1本にしている。`compare_fe_iv.py`の1本にして
   GMMをmethod軸で測る案（#380・#389の名前を直す）。
3. **軸**: n（T=6固定でNを掃引）、k、内生変数・操作変数の数、cov軸（classical・hc1・hc2・hc3・cluster・dk）、
   method軸（2sls・gmm 2-step・iterated）、2-wayの代表点、不均衡2-way（#522後）、3-way（#442後）。
4. **計測範囲**: 比較なしなので対称性の調整は要らないが、FEIVは診断（Sargan・Wu-Hausman・弱操作変数F等）を
   `fit()`で一括計算する。性能ノートに、計測した時間がこれらを含むことを明記する。
5. **交互射影のコスト**: 1回のFEIVで`y`・`x_exog`・`x_endog`・操作変数の全列を反復変換するので、
   変換済み列を渡す内部経路（5.4）の効果を計測で確かめる。

## 7. 残る未決事項（実装時に迷いやすい点）

1. **HC2/HC3（残す方針で決定）**: fixestはIVで提供しないため、参照はplm（`vcovHC(method = "white1")`。HC0が構造残差と`X̂`の
   サンドイッチに一致することは実測済み）。HC2/HC3のレバレッジがplmで`X̂`ベースかFEの`1/T_i`補正込みかは、
   plmの出力と手計算で#379で確定する。他手法（FE・RE・OLS）と提供する`cov_type`を揃えるため残す。
   plmの定義が本実装の想定と一致しない場合も落とさず、手計算の独立実装を参照にする
   （`verification-details.md`の単一リファレンスの例外に載せる）。HC2/HC3の`1 - h`が1に近づくときの扱いは、IVのWald対称化とともに#628/別Issueで決める。
2. **Wu-Hausmanの参照（決定）**: fixestの`wh`はiid前提で、本実装は`cov_type`に対応させる（1.4）ため、`cov_type`が
   classical以外では一致しない。参照はRスクリプトで、fixestで`v̂`を足した補助FE回帰を同じ`vcov`で回し、
   `v̂`のWaldを取る。分母自由度は補助回帰のもの（`n - k - k_endog - neffects`、IVの`wu_hausman_p_value`の教訓）。
   classicalのときだけ`fitstat(m, "wh")`と直接比べる。
3. **Sarganの`σ̂²`の分母（決定）**: `e'e/n`（fixestの`sargan`と一致を実測）。小Tで過大になる旨は
   `docs/methods/`に注記する。
4. **弱操作変数F（部分F）**: IVと同じく、`cov_type`によらず等分散前提で、分母自由度は`n - l - n_absorbed`。
   FEの既定`cov_type`がclusterでも部分Fはclassicalであることを`docs/methods/`に書く（ロバスト版はKP/MOP）。
   fixestの`ivf`と一致するかは#379で確認する（実測では`ivwald`が`VCOV: IID`と表示）。
5. **DKの参照**: fixestのDKは時点列の指定が要る（`panel.id`）。IVモデルで動くかは#379で確認する。
   plmの`vcovSCC`はwithin IVで値を返す（定義の対応は未確認）。
6. **エンジンの構成**: 現在の`fe_iv.rs`は入力型だけを持つ。`rust-style.md`のファサード＋子モジュールの規約に
   合わせ、#371で`fe_iv/input.rs`へ移し、`fe_iv.rs`はファサードにする。推定量は2SLS用（`FeIvEstimator`）と
   FE-GMM用で別の型とし、共通の読み出し口（`IvEstimate`相当）は#381で足す。
7. **`residuals`**: within変換後の空間での構造残差`ỹ - X̃β̂`（`FEResults.residuals`と同じ空間）。docstringに書く。
8. **pickle**: `source_data`（使った列だけ）を状態に含め、`rebuilt_from`で作り直せる部分は状態に入れない
   （`rust-style.md`）。第一段階の`FEResult`は`Computed`なのでそのままpickleできる。
9. **FE-GMMの重み行列の検証**: `cluster`のクラスター数`G`と操作変数の数`l`（`k_exog + k_instruments`、切片なし）の
   境界は、IVの`InsufficientClustersForWeightMatrix`（過剰識別は`G < l`、丁度識別は`G <= l`）を使い回す。
10. **時間不変の内生変数＋時間不変の操作変数**: Hausman-Taylorが扱う型で、FEIVでは識別されずエラー
    （5.1の分割で「固定効果を除くと一定」）。`hint`でHausman-Taylorが対象外であることを書くかを決める。
11. **ドキュメント**（#378に含める）: `docs/usage`・`docs/api`・`docs/methods`の3ページ（navはPanel models配下）、
    `docs/guide`の表（api-conventions・inference-conventions・validation・verification・performance）、
    READMEの表、`error-codes.md`の再生成、`R²`の定義の差（fixest）を`verification-details.md`に。
12. **R²のbetween・overall**: 6.1の4で決定（FEと同じ3つを出す）。

## 8. Issueの整理（2026-11-03）

- #389は#380に統合してclose（`compare_fe_iv.py`の1本、GMMはmethod軸）。
- #368・#369: C統計量は事後診断のメソッドにし、結果が`source_data`と`S_used`等を保持する方針をコメント済み。
- #371・#372・#377・#380: 本ノートの設計に合わせた更新をコメント済み。
- #373・#374・#375・#379・#388: 参照パッケージ（plmのwithin IV、fixestの`fitstat`、`linearmodels.IVGMM`）の
  記述を直すコメントを投稿済み。

## 9. 進捗（2026-11-07時点）

### 9.1 先行して決める事項は決定済み

5〜7章の決定（グループsingletonはエラー維持、HC2/HC3は残す、R²は3つ出す、GMMの重みの既定は`cluster`、
C統計量は事後診断のメソッド、`first_stage()`は`Computed`変種、性能比較は engine 単独、等）は
確定。Issueへの反映（#368/#369/#371〜#375/#377/#379/#380/#388、#522/#442、#363〜#365、#628）も
投稿済みで、#389は#380に統合してclose済み。

### 9.2 #628（共通化）の進行状況

方針は#628に記録済み（Aは#613の決着、Bは行列受け取りの関数、CはIV診断への制約、GMMコアは狭い継ぎ目で共有）。

- **A 手順1（完了・コミット済み。`91a76a9d`、まだpushしていない）**:
  `shared/covariance.rs`に`HcScale`・`hc_scales`・`hc_meat`を追加。2SLS・GMM・パネル
  （classical・HC・Driscoll-Kraay）の重複を共有関数に置き換え、「独立実装」の方針を、
  スコアに依存する部分（レバレッジと小標本補正の分母）に限った。`shared/wald.rs`から
  `wald_quadratic_form`（対称化込み）を切り出した（OLS・FEが使用）。数値は不変で、全`cov_type`の
  176ケース（悪条件データ含む）がビット一致。engineテスト1142件・pytest全件パス。
  `rust-reviewer`と`/code-review`の指摘は対応済み（対応しなかったもの: 2SLS・GMMのWald複製=手順2、
  HC2/HC3の`1 - h`のガード=別Issue、DKの余分な確保=無視できる）。
- **A 手順2（完了。コミット済み）**: IVとTobitのWaldを共有の対称化版に切り替えた（数値が変わる）。
  - 共有に`wald_chi2_test`を足し、`two_sls/cov.rs`・`gmm/sandwich.rs`・`tobit/statistics.rs`の自前版を削除した。
    Tobitの裾の精度テストは共有側に移した。docs（`iv/CLAUDE.md`・`iv-spec.md`・`tobit-spec.md`・各モジュールdoc）は最終形に更新済み。
  - **影響の測定**（176ケース、手順1の状態との比較）: OLS・FE・REは不変。良条件のIV・Tobitは`wald_*`が約1e-16
    （最終桁）だけ動く。悪条件（傾きの相関が1-5e-9程度）では大きく動き、結果の`cov_params`から有理数で厳密に
    求めた値に対する相対誤差は、旧: 最大約480%（2SLS-HAC）、GMM 最大約58%、Tobit 最大約20% → 新: 1e-8程度。
    `eps=1e-4`のGMM（クラスター重み）の旧`ComputationError` 2ケースは成功する（非対称のまま条件数チェックに
    かかっていたため）。
  - pytest全件（11694件）が許容誤差・フィクスチャを動かさずにパスした。回帰ガードとして
    `tests/cross_method/test_wald_ill_conditioned.py`（2SLS・GMM・Tobitを有理数の厳密解と照合）を追加し、
    旧コードで9件すべて落ちることを確認済み。
  - `rust-reviewer`・正確性レビュー・`/code-review`の指摘は対応済み（見送り: `df_model==0`の分岐を共有側へ寄せる案
    は OLS・FE の挙動に触れるため、別の機会）。結果は#612・#628にコメント済み。
- **B（完了。コミット済み）**: `engine/src/shared/instrumental_variables.rs`に行列を受け取る関数を切り出した。
  - `two_sls_point_estimate`（`X`・`X̂`・`y`→係数・`(X̂'X̂)⁻¹`・構造残差・SSR）、`sargan_test`（`σ̂²`の分母は引数）、
    `partial_f_statistic`（制限なしモデルのSSRと回帰変数の数を受け取り、`n_absorbed`を自由度から引く。
    #628の本文は両方の回帰を関数の中で行う形だったが、第一段階で求めたSSRを再利用する形にした）、
    `wu_hausman_columns`（拡張回帰の列と名前の組み立てだけ）。
  - 2SLSは`second_stage`（`OlsEstimator`）フィールドをやめ、`params`・`param_names`・`dep_var_name`を直接持つ。
    `SecondStageFailed`の発生条件と順序は`validate_second_stage`＋`from_rank_deficient`で維持した
    （回帰変数の有無→信頼水準→特異）。`second_stage`を直接見ていたテスト4か所は書き換えた。
  - 検証: 176ケースがビット一致（弱操作変数Fの値と自由度を追加）。engineテスト・pytest全件パス。
    n=1,000,000・k=5のIV(hc1)は1.06秒→0.92秒（第二段階の不要な標準誤差・F検定・適合度の計算が消えた）。
    FWL不変性（ダミー入り`n_absorbed=0`＝within＋`n_absorbed`）を単体テストで固定した。
  - 残り: Wu-Hausmanの回帰・Waldは呼び出し側の回帰エンジンのまま（FEIVでは`FeEstimator`）。
- **GMMコアの継ぎ目（完了。コミット済み）**: `engine/src/shared/gmm.rs`に`gmm_sandwich_covariance(z, x, s_used, omega)`
  （`B⁻¹(X'ZWΩ̂WZ'X)B⁻¹`。`Ω̂`は受け取る。スケールは`n`で割らない和＋小標本補正込み）、
  `hansen_j_test(z, residuals, s_used, df)`（`IvInput`を外し自由度を受け取る。丁度識別の分岐は呼び出し側）、
  `invert_spd`を切り出した。`Ω̂`の組み立てと小標本補正の分母は`iv/gmm/sandwich.rs`に残る。数値は不変
  （176ケースがビット一致）。`S≠Ω̂`の一般形を別経路の手計算と照合するテストを足した。
  FE-GMMは自前の`Ω̂`（FEのクラスター補正・Driscoll-Kraay・FEのレバレッジ）で同じ関数を呼ぶ。
  なお`cov_params`は丸めで厳密には非対称（従来どおり。Waldは共有版が対称化する）。使う側が対称性を要するなら
  FE-GMMで対称化する。これで#628のA・Bと継ぎ目はすべて完了。

- **#363（Cragg-Donald＋Stock-Yogo、エンジンのみ。完了。コミット済み）**: `shared/weak_instruments.rs`（行列版`cragg_donald(y_endog, x_exog, z, n_absorbed)`）、
  `shared/stock_yogo.rs`（2SLSのバイアス・サイズの2表）、`iv/common/weak_instruments.rs`（`compute_cragg_donald(&IvInput)`）。
  結果は関数を公開するだけで、estimatorにフィールドは足さない（公開は#366）。定義できない入力は`None`。
  linearmodels 7.1の`cragg_donald_f`と相対差1e-15で一致。臨界値はNBER原稿（TWP 284、2002年10月改訂版）の
  Table 1・2をPDFから転記し、`ivreg2`のソースは取得できなかったので全セルの独立な照合はしていない
  （単調性・滑らかさ・既知値で検証）。公開ドキュメント（`docs/methods/`）は#366でPythonに公開するときに書く。

- **#364（Kleibergen-Paap、エンジンのみ。実装済み。未コミット→コミット後に更新）**: `shared/kleibergen_paap.rs`
  （`kleibergen_paap_lm`・`kleibergen_paap_f`。`(Y, X_exog, Z, n_absorbed, ScoreCovariance)`）、
  `iv/common/weak_instruments.rs`（`compute_kleibergen_paap(&IvInput, &CovType)`）。決定事項:
  rk LMとrk Wald Fの両方、`Hc0`〜`Hc3`の4種に対応、DKはFEIV側で`ScoreCovariance`に分岐を足す。
  **linearmodelsの`robust`のFは`Hc1`、`robust`のLMは`Hc0`に対応する**（Fの自由度補正を定義に含める）。
  `λ`の方向に射影した`q−k+1`次元のスコアに共有のmeat関数を使う（`qk×qk`の行列は作らない）。ランク落ちの共分散は
  一般化逆行列を使わず`None`（LMとWald Fは独立に判定。Wald Fはクラスターが1つ多く要る）。linearmodelsと相対差1e-9で一致
  （n=12・n=240）。HC2/HC3・`Hc1`のLM・`n_absorbed`つきは参照が無く、クロネッカー積をそのまま使う独立な計算と照合。
  fixestの`kpr`は参照にならない。`n_absorbed > 0`のHC2/HC3は今は`None`（吸収した効果のレバレッジを含まないため。FE-2SLSでKPを
  hc2/hc3で出すなら、FEのレバレッジ（`1/T_i`込み）を`ScoreCovariance`に渡す形を足す）。

- **#365（MOP有効F、エンジンのみ。単純版は`f725458f`でコミット済み。一般化版は実装済み・未コミット）**: `shared/effective_f.rs`
  （`effective_f`・`tsls_bias_bound`・`patnaik_critical_value`）、`shared/score_covariance.rs`（KPから共分散の選択を取り出して共有）、
  `shared/noncentral_chi_squared.rs`、`iv/common/weak_instruments.rs`（`compute_effective_f`）。決定事項: 内生変数1個だけ
  （2個以上は`None`）、LIMLは実装しない、一般化版（`B_TSLS`の最大化。`y`が要る）は同じIssueで足す（実装済み）。論文のTable 1の
  120セルと同次分散の公表値（8.53・12.27）を再現。別セッションが論文の式だけから書いたPythonオラクル
  （`mop-oracle-spec.md`、`benchmark/iv/references/`）と26ケースで1e-14以内に一致（Rustの回帰テストは代表の17ケースと、誤差がほぼ比例する`Ŵ`の2ケースを1e-9で固定）。
  `Ŵ2`が特異のときの扱い（KPは`None`、有効Fは値を返す）は#366で決める。

### 9.3 push済みのコミット

`1bf01e91`（決定事項）、`91a76a9d`（#628 A 手順1）、`8d8d8dc8`（A 手順2）、`c1f13b84`（進捗）、`1631e728`（#628 B）、
`5075ca28`（GMMコアの継ぎ目。#628はクローズ済み）、続けて#363のコミット（`git log`で確認）。

追加（2026-11-06〜07）: `9581fe26`（#364 KP）、`f725458f`（#365 単純版）、`abcf32f2`（#365 一般化版）、`28f8a375`（オラクル。実装セッションのコミット）。`b7684b3c`（回帰テストの固定）。ここまで**push済み**。#365はクローズ済み。
#364はクローズ済み。#365は一般化版のコミット後も、オラクルのレビューと突き合わせが残るため**まだ開いている**。

### 9.4 この後の順序（5.3の実装順）

1. #363は完了（クローズ）。#364（Kleibergen-Paap）は実装済み。次は#365（MOP）。
   制約（行列受け取り・`n_absorbed`・共有のクラスター/HAC meat）は各issueにコメント済み。
2. #522 → #442（その前に5.4の`PanelIndex`のK次元化を決める。グループsingletonはエラー維持）。
3. IV本体の残り: #366（バインディング）→#367（ivDiag照合）、#368・#369。
4. FE-2SLS（#370〜#372、#374、#375 → #373 → #376〜#380）、続いてFE-GMM（#381〜#388）。

### 9.5 残る論点・引き継ぎ（compact用のメモ）

- **#366で決めること**（#366にコメント済み）:
  - CD統計量の公開の形: `fit()`時のプロパティなら`None`のまま（Wu-Hausman・Sarganと同じ。`fit()`が通った入力では実質`None`にならない）、
    事後診断メソッドならエラー（`Result`＋理由の列挙型＋カタログの`code`）にする。後者なら`compute_cragg_donald`の戻り値を
    `Option`から`Result`へ変える。臨界値が表に無い組み合わせはエラーにせず、統計量だけ返して臨界値の欄を空にする案。
  - 公開ドキュメント（`docs/methods/iv.md`の定義式、出典）はこの公開のときに書く（出典は`iv-spec.md`3.4.1に記載済み）。
- **#366でKPについても決めること**: `compute_kleibergen_paap`は不正な設定（`Cluster`でグループ無し、HACのラグ数や時間順の長さの不正）も
  「定義できない」と同じ`None`に潰す（`fit()`が`cov_type`を先に検証するので、`fit()`後のプロパティなら到達しない）。事後診断メソッドにするなら、
  設定の誤りは`ValidationError`、数値的に定義できないものは別の理由で返す形（CDと同じ`Result`＋理由の列挙型）にそろえる。
  LMとWald Fは独立に`Option`で、`KleibergenPaap`は片方だけ`Some`になりうる。`RankTest`は`iv::common`から再エクスポート済み。
  最小の2つの特異値が重なるときのロバスト版の不安定性は検出していない（既知の限界。`iv-spec.md`3.4.2節）。
- **#366で有効Fについても決めること**: (1) `Ŵ2`が特異のとき、有効Fは値を返す（KPは`None`）。公開時に揃えるか、警告として扱うか。
  (2) `alpha`が不正（(0,1)の外）のときも、`compute_effective_f`は「定義できない」と同じ`None`に潰す。公開側で
  `alpha`を検証して`ValidationError`にするか、エンジンが理由つきで返すか（KP・CDの`None`と同じ論点）。
  (3) 内生変数が2個以上は`None`（原論文は1個。拡張の参照実装が無い）。理由を利用者に示すかどうか。
- **Stock-Yogoの表**: 第二の独立な出典で全セルを照合していない（`ivreg2`のソースはSSLエラーで取得できなかった）。
  単調性・滑らかさ・既知値で検証。独立な出典（`ivreg2`の出力、Stataのマニュアル、他言語の実装）が手に入れば、全セル照合を足す。
- **ルートCLAUDE.mdの3章**: `shared/`の列挙に`instrumental_variables.rs`・`gmm.rs`・`weak_instruments.rs`・`stock_yogo.rs`が載っていない
  （CLAUDE.mdの編集は利用者の承認が要る）。
- **未起票の候補**: HC2/HC3の`1 - h`のガード（#612項目2）。#628のレビューで見送った小さな重複
  （信頼水準の範囲検査が5か所、`partial_f_statistic`の制限モデルの不要な`(X'X)⁻¹`、Sargan・Hansen Jの結果型の二重化、
  `df_model==0`の分岐を共有関数へ寄せる案、`shared/gmm.rs`の`cov_params`の対称化）は、必要になったら起票する。
- **一時スクリプト**（リポジトリ外、`/tmp/claude-1000/-workspaces-econometricsmodels/a59ac130-80bd-4651-8271-18b321bf492f/scratchpad/`）:
  `snapshot.py`（IV/GMM/FE/RE/OLS/Tobit/悪条件176ケース）・`compare_snap.py`・`sy/`（Stock-Yogo表の抽出・検証）。失われたら作り直す。
- **コミット手順**: 署名はホスト端末の確認が要る。署名に失敗したら止まって報告する（再試行・回避はしない）。

### 9.6 #365の現状と次の作業（compact用のメモ。2026-11-07）

**状態**: エンジン側は完了（単純版`f725458f`、一般化版は続くコミット）。engineテスト1266件・clippy・fmtは通っている。pytest全件は未実行
（engineとengine_pybindへの追加のみ）。#365へのコメントとクローズ、pushは未実施（利用者の指示待ち）。

**実装の要点**（詳細は`docs/spec/iv-spec.md`3.4.3、`engine/src/iv/CLAUDE.md`）:
- `shared/effective_f.rs`: `effective_f(Y, X, Z, n_absorbed, ScoreCovariance, alpha, y_dep: Option)`。`y_dep`を渡すと一般化版TSLS
  （`B_TSLS`の最大化）も出す。`β`の格子は`sinh`の2通り（`β₀`と、`tr S1`の最小点`β*`まわりの幅`δ`）。**`β*`まわりの格子は落とさない**
  （誤差がほぼ比例する強い内生性で、鋭い山を取りこぼして`B_TSLS`が9%過小になった）。
- `shared/score_covariance.rs`（KPと共有）、`shared/noncentral_chi_squared.rs`、`iv/common/weak_instruments.rs`（`compute_effective_f`）。
- テストは`effective_f/tests.rs`（ハブ）＋`tests/{bias_bound,critical_values,invariance,reference,undefined}.rs`。`reference.rs`に独立オラクルの代表17ケースを相対差1e-9で固定し、誤差がほぼ比例する`Ŵ`の2ケース（`scale`=1e-2は1e-9、1e-4は1e-7。`Ŵ`への1e-16のノイズで`B_TSLS`が約`1e-18/ratio`動くため）を固定。`scale`=1e-6（`min trS1/trW1`≈1.5e-13）は固定のしきい値1e-12の内側で`None`になるのが仕様（`undefined.rs`で固定）で、オラクルの値は固定しない。オラクルとの突き合わせは26ケースで相対差1e-14以内（実測の最大は約8e-15）。
- 論文のTable 1の120セルと、同次分散の公表値（`τ=0.10`・`α=0.05`で`K=3`が8.53、`K=30`が12.27）を再現。

**独立オラクル（別セッション）**:
- （9.7で更新: 値の表は`benchmark/iv/references/mop_oracle_values.md`へ移った）仕様書は`mop-oracle-spec.md`（コミット済み）。実装は`benchmark/iv/references/mop_oracle.py`・`mop_oracle_cases.py`・
  `mop_ivdiag_compare.py`・`run_ivdiag_eff_f.R`、値の表は`docs/planning/specs/mop-oracle-values.md`、`tests/iv/test_mop_oracle.py`、
  `tests/support/_tolerances_iv.py`、`pyproject.toml`・`uv.lock`（scipy明示）。**これらは別セッションがステージ済みで、コミットはその
  セッションが行う**（こちらのコミットはパス指定で、これらを含めない）。
- **クリーンルームの規則**: オラクルを書くセッションは`weakivtest.ado`（権利が留保されている。Stata原典）を一度も見ていない。
  こちらのセッションは読んでしまったため、`ado`のコピーは削除済み。オラクルのレビューは**コード品質に限り**（規約・重複・テスト構造・
  `benchmark/`の流儀・仕様書8章の確認項目の過不足）、**計算手順（最大化の方法など）の変更提案はしない**。指摘は、ado由来の内容
  （手順・定数・構造）が混ざっていないかをこちらで確認してから、実装セッションへ渡す。レビュー自体はネットワークに出ない
  サブエージェントで行い、adoを取得しない指示を明記する。
- **オラクルのレビューで最優先に確認すること**: 誤差がほぼ比例する強い内生性のケース（`min tr S1 / tr W1`が1e-3〜1e-8）で、オラクルの
  `B_TSLS`の最大化（`beta`軸の粗い格子＋局所探索と、`angle`の周期関数の最大化）が鋭い山を取りこぼしていないか。値の表の26ケースは
  このケースを含まない。こちらの実装は、これで9%過小になる取りこぼしを起こしていた（修正済み）。
- ivDiagとの比較結果（`mop_ivdiag_compare.py`・値の表の末尾）は、こちらではまだ読んでいない。

**#366・#367の前に決めること**: 上の9.5の各項目。有効Fの`Ŵ2`が特異なときの扱い（利用者は「#366で方針を決め、必要ならengine側を修正する」
で合意。レビュー後の意見は、論文が`Ŵ`の正定値を仮定しているので、KPに揃える方向を勧める）、`alpha`の検証の置き場所、
内生変数が2個以上のときの`None`の理由の見せ方。#367ではivDiagの恒久的な導入（`Dockerfile`の`install_version`、
`check_r_updates.yml`）を行う（ivDiagは`eff_F`の統計量だけで臨界値は返さない。lfeの自由度補正の規約が本実装の`n_absorbed`と
合うかを実測する）。

**その他の保留**:
- **READMEの権利に関する注意書き**: 利用者が追加に賛成。文面案は会話で提示済み（"Attribution and Rights"節。連絡先は
  private vulnerability reporting。無条件の削除の約束や「訴訟」の語は避ける）。連絡先の経路（private reportingのみか、メールも併記か）と
  文面の承認待ち。Stock-Yogoの表とMOPのTable 1は、出典つきで転記した公表値。
- **フィクスチャのデータのライセンス**: Wooldridgeデータセット等の外部データの出所とライセンス表記がREADME・ドキュメントに揃っているかの
  確認を提案済み（未着手）。
- **ルートCLAUDE.mdの3章**: `shared/`の列挙に、`instrumental_variables.rs`・`gmm.rs`・`weak_instruments.rs`・`stock_yogo.rs`に加えて
  `kleibergen_paap.rs`・`score_covariance.rs`・`effective_f.rs`・`noncentral_chi_squared.rs`が載っていない（編集は利用者の承認が要る）。
- **次の順序**: #365のオラクルのレビュー→（実装セッションが修正・コミット）→#365のクローズ→#522→#442→#366/#367/#368/#369→FE-2SLS→FE-GMM。
- **一時ファイル**（リポジトリ外、scratchpad）: `mop/mop.pdf`・`mop/mop.txt`（論文のテキスト。公開資料）、`kp_ref.py`・`kp_big.py`（KPの
  linearmodels基準値）、`pylib/`（pypdf）、`rust_dump.txt`（突き合わせ用の一時ダンプ）。失われたら作り直す。

### 9.7 オラクルのレビューの結果と回帰テストの固定（2026-11-07）

- **オラクルのレビュー（完了）**: 最優先の「誤差がほぼ比例する」ケースは問題なし（`min trS1/trW1`が1e-9まで、`β*`中心の対数20万点の
  総当たりと`beta`・`angle`が1e-10以内で一致。1e-11では総当たりより最大7.5e-8（`beta`）・5.8e-9（`angle`）小さい。原因は`trS1`の展開式の
  `trS1`の展開式の桁落ち）。コード品質の指摘は、実装セッションが修正してコミット済み（`28f8a375`。回帰ケース追加・テストの高速化・値の表の移動
  （`benchmark/iv/references/mop_oracle_values.md`）ほか）。#365のオラクル側は終わり。
- **Rustの回帰テストの追加（未コミット）**: `effective_f/tests/reference.rs`に、誤差がほぼ比例する`Ŵ`の2ケース（`scale`=1e-2・1e-4、
  `min trS1/trW1`は1.5e-5・1.5e-9）でオラクルと`B_TSLS`・一般化版の臨界値を照合（許容誤差は1e-9と1e-7）。`scale`=1e-6（≈1.5e-13）は
  `DEGENERATE_BENCHMARK_TOLERANCE`（1e-12）の内側で`None`になる仕様を`undefined.rs`で固定。engineテスト1268件・clippy・fmtは通っている。
- **`/code-review`（7件）の対応**: 実測（`Ŵ`の要素に1e-16の相対ノイズを載せる）で、`B_TSLS`の変化は約`1e-18/ratio`（`ratio`=`min trS1/trW1`。
  1.5e-9で5e-10、1.5e-13で3e-6）。しきい値1e-12は据え置き（`Ŵ`の丸め誤差の影響は1e-6程度で、臨界値に効かない）。対応: ①しきい値の
  コメントと説明文を実測に合わせた（桁落ちを検出するものではなく固定のしきい値）、②固有値は本番と同じ`psd_eigenvalues`の経路、
  ③1e-4のケースの許容誤差を1e-7に緩めた（`sin`/`cos`の1 ulp差の余裕）、④rust-reviewerの指摘で、オラクル照合を`reference.rs`、`None`の判定を`undefined.rs`へ移し、共有ヘルパー（`blocks`・`nearly_proportional_w`）をハブに置いた、
  ⑤固有値の定数を1つにまとめ、`ratio`は記録した値との照合にした。
- **完了**: `b7684b3c`（回帰テストの固定）をコミット・push済み。#365はコメントのうえクローズ済み。次は#522。

### 9.8 #522（不均衡2-way FE）の着手前の調査と決定（2026-11-07）

**調査の要点**
- 現行の2-wayは「entityでquasi-demean→timeでquasi-demean」の2段階で、バランスパネルでだけ厳密。均衡前提は3か所:
  `validate_balanced_panel`、HC2/HC3の2-wayレバレッジ`1/T_i + 1/N_t − 1/n`（`fe_adjustment.rs`）、`fixed_effects()`の2元ANOVA恒等式。
- REの2-wayは内部でFEを呼ぶが、`re/variance.rs`に独立の不均衡ガードがある（RE側は不均衡を拒否したまま。別Issue）。
- 自由度は変えない: 連結成分が2つの不均衡データ（k=1、N=T=8）で、fixest 0.14.2もlinearmodelsも`df_model = k+N+T−1`（=16。真のランクは15）。
- union-find・交互射影のコードはengineに無い（`GroupCodes`のCSR索引は使える）。fixestの2-way呼び出しは既にある（均衡のみ）。
  linearmodelsは不均衡2-wayも扱える（クロスチェックに使う）。不均衡2-wayのDGPは無い。
- 交互射影は有限の許容度で止まるため、分散ゼロ・完全適合・定数yの判定しきい値（`n·EPS·max|x|`など）が合わなくなる。許容度に連動させる。

**利用者の決定**
- **HC2/HC3**: 不均衡2-wayでも対応する（entity+timeダミーのレバレッジをシューア補集合で解く）。小さい方の次元（NかT）を密に解く。
  メモリ`(小さい方の次元)²`に上限を設け、超えたらValidationError（上限は実装時に決める）。
- **連結成分が複数のパネル**: 受け入れる。`df_model = k+N+T−1`のまま。`fixed_effects()`は成分ごとに基準化（時間順で最初の時点をγ=0）。
- **singletonのエラー**: paramsに、連鎖を含めて何行落ちるかを足す。**polarsのフィルタ例をhintに出す案には、利用者が過剰と指摘**。
  連鎖では1回のフィルタで済まないため、hintは機械的な対処（`error-style-guide.md`）にとどめる方向（実装時に文面を決める）。

**実装計画（コミットの単位）**: ①交互射影の核（`shared/`。K個のコード配列、索引・連結成分の共有、Irons–Tuck加速、反復数と収束の有無を返す。
LSDV密解とバランスの閉形式で検証）→②検証の置き換え（重複ペアの検査をソート/ハッシュで。不均衡の拒否をFEから外し、RE専用に文面を直す）→
③FEのfit配線（全ペアそろい・1-wayは閉形式のまま、不均衡だけ交互射影。`FEOptions`に`tol`・`max_iter`。非収束は`ComputationError`。
しきい値を許容度に連動）→④HC2/HC3のレバレッジ→⑤`fixed_effects()`の復元（残差に交互射影、成分ごとの基準化）→
⑥ベンチマーク（DGP: ランダム欠測・疎・連結成分が複数・ほぼ非連結、いずれもsingletonなし。主はfixest、クロスチェックはlinearmodels。
`r_squared_within`はfixestのみ）→⑦ドキュメント（fe-spec・panel-common・validation・verification・CHANGELOG・性能比較）。

**進捗（engine側 ①〜⑤、同日）**: 交互射影の核（`shared/alternating_projections`）、重複ペアの検査、FEのfit配線（`FeAbsorption`の`tol`・`max_iter`、
`fit_with_absorption`）、不均衡2-wayのHC2/HC3レバレッジ（`dummy_leverage`）、`fixed_effects()`の成分ごとの基準化まで実装し、ダミー変数回帰との照合
（係数・残差・古典/HC1/HC2/HC3・`fixed_effects()`・連結成分が複数のパネル）が通っている。rust-reviewer・`/code-review`の指摘（完全適合のしきい値の単位、
収束判定の尺度を第1掃引の後にする、射影の誤差の見積もり`error_bound`をノイズに使う、早期の水準数チェック、`fixed_effects()`は`fit`が保存した効果から組み立てる、
重複検査のO(n)化 ほか）は反映済み。
- **実測**: つながりが細い鎖状のパネルは掃引の回数が鎖の長さの2乗に比例して増える（長さ20で約900、50で約6000、200で約86000）。既定の上限10000で
  `AbsorptionNotConverged`になるのは仕様（`tol`を緩めるか`max_iter`を上げる）。緩い`tol`は分散ゼロ・完全適合の検出も粗くする。
- **残り**: ⑥engine_pybind（`FEOptions`に`tol`・`max_iter`、docstring、pickle、型検査テスト、`error_codes.py`・`docs/guide/error-codes.md`の再生成）、
  Pythonのテスト（`test_unbalanced_two_way_raises`の反転ほか）、ベンチマーク（DGP・fixest主・linearmodelsクロスチェック、`r_squared_within`はfixestのみ）、
  ⑦ドキュメント（`fe-spec.md`・`panel-common.md`・`validation.md`・`verification.md`・`usage/fe.md`・`methods/fe.md`・CHANGELOG・性能比較）、
  singletonのエラーに連鎖の行数を足す件、定数列のエラーの2分類（#522のコメントの付帯項目）。
  Pythonまで終わるまでpushしない（CIが落ちる）。

**進捗（Python側・ベンチマーク・ドキュメント、同日）**: engine側は`f55f4d26`でコミット済み。続けて、engine_pybindの`FEOptions`に`tol`・`max_iter`
（pickle・repr・setter・docstring）、`error_codes.py`・`docs/guide/error-codes.md`の再生成、Pythonテスト（不均衡2-wayの受理、`DUPLICATE_ENTITY_TIME_PAIR`・
`ABSORPTION_NOT_CONVERGED`・`LEVERAGE_DIMENSION_TOO_LARGE`・`tol`/`max_iter`の範囲）、DGPに`unbalanced_sparse`・`unbalanced_disconnected`を追加し、
fixest（全`cov_type`）とlinearmodels（classical・HC1）の凍結フィクスチャで`unbalanced`・上記2シナリオの1-way・2-wayを照合、公開ドキュメント・仕様書・CHANGELOGまで更新。
- **照合結果**: 係数・標準誤差はfixestと最大4.4e-13で一致（連結成分が2つのパネルを含む）。Pythonの全テスト11904件が通る。
- **fixestの落とし穴**: 既定の`fixef.tol=1e-6`だと、疎な不均衡パネルでfixest自身の共分散が約5e-9ずれ、1e-8で比べられない。参照スクリプトで
  `fixef.tol=1e-11`（fixestの下限は2.2e-12）にした。linearmodelsも非連結パネルで`k+N+T−1`の自由度を使うことを実測で確認。
- **残り**: FEの性能比較への不均衡シナリオの追加（`performance/compare_fe.py`）、singletonのエラーに連鎖の行数を足す件、定数列のエラーの2分類
  （#522のコメントの付帯項目。別Issueに分けるか要確認）、`fixed_effects()`のfixest `fixef`との比較（基準化が違うので和`α_i+γ_t`で比べる案）、
  #522のコメントとクローズ、push。

**進捗（#522の付帯項目、同日）**: ①`SINGLETON_GROUP`に`n_dropped_rows`（取り除くことを繰り返したときに落ちる行の総数。2-wayは連鎖を含む。hintはpolarsのフィルタ例を
出さず「連鎖が残るなら繰り返す」まで）、②定数列のエラーの2分類（生の値が全ての行で同じ説明変数は`REGRESSOR_CONSTANT`、同じ被説明変数は`OUTCOME_CONSTANT`、
固定効果を除いた後に分散がなくなる場合は従来どおり`REGRESSOR_NO_WITHIN_VARIATION`・`OUTCOME_CONSTANT_WITHIN_ENTITY`。RE内部のFEも同じ）を実装。
`fixed_effects()`の残差再現テスト（不均衡3シナリオ）も追加。engineテスト1318件・Python全11911件が通る。コミット前。
- **完了**: #522はコミット・push・コメント・クローズ済み（9.9参照）。FE性能比較への不均衡シナリオは別Issueにした（9.9）。

### 9.9 セッションclear用のメモ（2026-11-07 終了時点）

**完了したこと**
- #364（KP）・#365（MOP有効F。オラクルと26ケースで1e-14）・#522（不均衡な2-way FE）が完了・クローズ済み。いずれも`release/v0.10.0`にpush済み。
- #522の最終形: 交互射影（`shared/alternating_projections`）、`FeAbsorption`（`FEOptions.tol`・`max_iter`）、重複ペアの検査、HC2/HC3のダミーのレバレッジ
  （`panel/common/dummy_leverage.rs`）、`fixed_effects()`の成分ごとの基準化、`SINGLETON_GROUP`の`n_dropped_rows`、`REGRESSOR_CONSTANT`・`OUTCOME_CONSTANT`の分離。
  fixest（全`cov_type`）とは最大4.4e-13で一致。Python全11911件・engine 1318件が通る。

**次に進める内容（順序）**
1. **#442（高次元固定効果、3次元以上）**: 着手前に5.4節の`PanelIndex`のK次元化（`Vec<GroupCodes>`＋列名、DKの時点キーは別）を決める。交互射影の核
   （`GroupSystem`）はすでにK個のグループを受け取る。足りないのは、(a)K>2のHC2/HC3（`dummy_leverage`は2次元専用。K次元では密な行列が使えないので、
   拒否するか別法を決める）、(b)`FEOptions`/APIの形（`effects`の列の数の上限、公開APIは別の形にする案がIssueにある）、(c)`fixed_effects()`のK次元の基準化。
   singletonはエラーのまま（`n_dropped_rows`の連鎖計算は2次元まで。K次元に広げる）。
2. **IV本体の残り**: #366（KP・CD・MOPのバインディング。`None`とエラーの設計、`Ŵ2`が特異なときの扱い（KPに揃える案）、`alpha`の検証の置き場所、
   内生変数2個以上のときの理由の見せ方を先に決める）→#367（ivDiagの恒久導入＝`Dockerfile`の`install_version`と`check_r_updates.yml`、lfeの自由度の規約と
   `n_absorbed`の対応を実測）→#368・#369（C統計量）。
3. **FE-2SLS（#370〜#380）→FE-GMM（#381〜#388）**: #522の`GroupSystem`（列間で索引と連結成分を共有）が土台。FEIVは変換済み列を渡す内部経路を最初から設計する（5.4）。

**注意書き**
- コミットはパス指定で行う。署名はホストの端末でパスフレーズの確認が出る。Pythonを触ったら`uv run maturin develop`で入れ直してから`pytest tests`（約6〜7分）。
  エラーコードを足したら`python tools/render_error_catalog.py`で`error_codes.py`・`docs/guide/error-codes.md`を再生成し、`tests/support/_errors.py`の`MIGRATED_CODES`に足す。
- fixestで不均衡パネルを参照にするときは`fixef.tol`を厳しくする（既定の1e-6だと参照値が約5e-9ずれる。下限は2.2e-12）。
- 鎖状のパネル（つながりが細い）は掃引回数が鎖の長さの2乗で増える。既定上限10000で`ABSORPTION_NOT_CONVERGED`になるのは仕様。
- `FE`の`fixed_effects()`は`fit`が保存した列ごとの効果から作る（再射影しない）。
- 別セッションがステージしていたファイルは無い（オラクルは`28f8a375`でコミット済み。値の表は`benchmark/iv/references/mop_oracle_values.md`）。

**保留（利用者の判断待ち・未着手）**
- READMEの「Attribution and Rights」注意書き（連絡先の経路と文面）。フィクスチャのデータ（Wooldridge等）のライセンス確認。
- ルートCLAUDE.md 3章の`shared/`の列挙の更新（`instrumental_variables.rs`・`gmm.rs`・`weak_instruments.rs`・`stock_yogo.rs`・`kleibergen_paap.rs`・
  `score_covariance.rs`・`effective_f.rs`・`noncentral_chi_squared.rs`・`alternating_projections`・`validation`系。編集は承認が要る）。
- `docs/planning/specs/mop-oracle-spec.md`が履歴（`f725458f`）に入っていることの扱い。内容は論文の式とテストデータの定義で、weakivtest.adoの手順・定数は含まない
  （ado名に触れるのは「見ない」という指示の部分だけ）。履歴の書き換えはしない方針を提案済み（要確認。9.9末尾参照）。
- FE性能比較への不均衡シナリオ: `PerfAdapter.build_dataframe(n, k, seed)`が`method`を知らないため、ハーネス（`performance/_perf_harness.py`）の拡張が要る。
  #522には含めず、英語で別Issue #634として起票済み。

### 9.10 #442（高次元固定効果）の方針と実装順（2026-11-08）

**調査の要点**
- #595で`effects: list[str]`は導入済み。公開APIの形は変えず、`effects`の長さの上限を外す。
- `GroupSystem`（`shared/alternating_projections`）はK個のグループを受け取れる。2次元に固定されているのは、呼び出し側（`fe/within.rs`）、`FeInput`の
  `entity_codes`＋`time`、`PanelEffects`（`OneWay`/`TwoWay`）、`PanelDimension`（Entity/Time）、`FixedEffects`の列挙型、`dummy_leverage`、重複ペアの検査、
  `n_dropped_rows`の連鎖計算、pybindの`MAX_EFFECTS=2`（FEとRE共通）。
- fixestの自由度（不均衡・n=272、`firm + year + ind`）: `nparams = k + N1 + (N2−1) + (N3−1)`。`ind`が`firm`にネストして真のランク（48）が下がっても数え方は変わらない
  （52）。2-wayの「連結成分は無視する」と同じ規約なので`df_model = k + Σn_d − (K−1)`。クラスターの小標本補正でネストした次元をKから除く規則
  （`fe_cluster_k_correction`）は`dims`をVecで作る実装のまま広げ、fixestの`ssc(fixef.K="nested")`と数値で照合する。

**利用者の決定**
1. **K≥3のHC2/HC3は対応する。** 最大の次元を消去し、残りの次元の水準数の合計を密に解く（2-wayの`dummy_leverage`の一般化）。上限（`MAX_DENSE_LEVELS`）は流用。
2. **重複の検査はしない（2-wayも含む）。** `effects`はentity/timeの意味を持たず、州×群のように同じ組合せの複数行が正当な使い方がある。重複検査は「全ペアそろい」判定
   （`n == N×T`）の前提だったので、バランスの判定を「`n == N×T`かつ重複なし」に置き換え、バランスのときだけ閉形式、そうでなければ交互射影に回す。
   REの2-wayは同じ判定で「不均衡」として拒否される。`DUPLICATE_ENTITY_TIME_PAIR`は未リリースなのでコードごと削除（`CHANGELOG`の該当記述も直す）。
3. **`fixed_effects()`**: 連結成分ごとに、1つ目の次元は自由、2つ目以降の各次元は基準水準を0にする。基準水準は、2つ目が時間順の最初、3つ目以降は
   水準の辞書順の最初（3つ目以降の列はentityと同じ`Identity`で扱う）。ネストなどで基準化しても分離できない効果があるため、識別されるのは和だけとdocstringに書く。
   fixestの`fixef`との比較も和で行う。
4. **上限**: FEの`effects`は5次元まで、REは2のまま（`MAX_EFFECTS`をモデルごとの値にする。`EFFECTS_TOO_MANY`の`max_columns`で報告）。
5. **収束**: `ABSORPTION_NOT_CONVERGED`は仕様として扱う（既定の`tol=1e-12`・`max_iter=10000`のまま）。

**方針**: K≥3は完全なバランスでも交互射影に統一する（閉形式は1-wayと2-wayのバランスのみ）。singletonは全次元でエラーのまま（`n_dropped_rows`の連鎖はK次元に広げる）。

**実装順（コミットの単位）**
1. K次元のデータモデル（挙動を変えないリファクタ）: `FeInput`/`PanelIndex`を`Vec<GroupCodes>`＋列名にし、DKの時点キーは別に持つ。`PanelDimension`・
   `InsufficientDegreesOfFreedom`・`SingletonGroup`を次元の番号と列名に一般化。REは2次元のまま`from_re_input`で橋渡し。K≤2の既存テストを安全網にする。
2. 検証: 重複検査をバランス判定の条件に置き換え（上記2）、`n_dropped_rows`の連鎖と縮退検査（`two_group_split_is_degenerate`）をK次元に。
3. within変換のK≥3の配線（`GroupSystem::new`にK個。完全適合・分散ゼロのしきい値は`error_bound`に連動）。
4. 自由度（`neffects = Σn_d − (K−1)`）とクラスター補正のK次元化。
5. HC2/HC3のレバレッジのK次元化。
6. `fixed_effects()`のK次元化（`FixedEffects`を`Vec<(列名, 水準と値)>`に一般化、pickleの`FORMAT_VERSION`を上げる）。
7. engine_pybind・Python（`MAX_EFFECTS`のモデル別化、`extract_effects`・`group_counts`・`FEResult.fixed_effects()`、型検査テスト、エラーコードの再生成と`MIGRATED_CODES`）。
8. ベンチマーク: 3次元目のDGP（バランス3-way・不均衡3-way・3つ目が1つ目にネスト・疎・連結成分が複数・同じ組合せの重複行）。主はfixest（`fixef.tol=1e-11`）、
   小標本はLSDV（ダミー変数回帰）をオラクルにする。linearmodelsは`PanelOLS`が3次元以上を持たないため、`AbsorbingLS`の対応を先に実測し、使えなければ単一リファレンスの例外にする。
9. ドキュメント: fe-spec（§5の「不均衡2-wayは未対応」の記述が古い）・panel-common・usage/api/methods/fe・guide（accepted-data・api-conventions・validation・verification）・
   CHANGELOG・`panel/CLAUDE.md`（engine・engine_pybind・python_package）。
10. 性能比較: 3-wayの追加は#634のハーネス拡張と合わせるかを確認する。

Pythonまで終わるまでpushしない。

**進捗（手順1、2026-11-08）**: 実装済み（コミット前）。`PanelDimension`は`effects`の添字の型（`ENTITY`=0・`TIME`=1）、`EffectNames`はVec（既定は`entity`・`time`・`effect3`…。次元の数より多い名前は捨てる）、
`PanelIndex`は`Vec<GroupCodes>`（`new(nobs, groups)`・`effect_codes()`・`n_effects()`）、`FeInput`は`groups: Vec<GroupCodes>`（コンストラクタは当面K≤2。`time_keys()`は`time_periods()`/`time_codes()`に置き換え）、
`InsufficientDegreesOfFreedom`は`n_groups: Vec<usize>`、`absorbed_parameters(&[usize])`（水準数の合計−(K−1)）を追加。`validate_panel_input`/`validate_panel_index`は`&[&GroupCodes]`を取り、各次元の長さ→各次元のキーの順に検査する。
K≤2の挙動は変えていない（engine 1321件・engine_pybind 240件・pytest `tests/panel`＋`tests/cross_method` 8846件が通る）。`fit_fe`はK>2を`debug_assert`で止める（手順3で外す）。
- **手順7で決める**: `PanelIndex::new`・`FeIvInput::from_group_codes`は2つ目が時間順のコードであることをdocでしか保証しない。`engine_pybind`が`TimeKeys`を渡すには`TimeKeys::codes()`が`pub(crate)`のため`GroupCodes`へ変換できない。
  `TimeKeys::into_codes`（`pub`）を足すか、時点の次元専用の引数を設けるかを、`FeInput`のK次元コンストラクタと合わせて決める。`FeInput::from_group_codes`の`time.codes().clone()`（n要素のコピー）も`into_codes`で解消できる。

**進捗（手順2、2026-11-08）**: 実装済み（コミット前）。重複ペアの検査（`ensure_no_duplicate_entity_time_pairs`・`PanelError::DuplicateEntityTimePair`・`DUPLICATE_ENTITY_TIME_PAIR`）を削除し、
`covers_every_combination`を「`n == N×T`かつ重複なし」に置き換えた（`validate_balanced_panel`も同じ判定）。重複のある入力は不均衡として交互射影に回り、ダミー変数回帰と一致する
（engineのLSDV照合・Pythonのnumpy照合・HC2/HC3）。REの2-wayは重複＋行数一致を不均衡として拒否する（`PANEL_NOT_BALANCED_FOR_TWO_WAY`）。
`ensure_no_singleton_groups`・`rows_dropped_by_repeated_removal`（連鎖）・`two_group_split_is_degenerate`は次元のスライスを取る形にし、3次元のテストを足した。
`FeInput`はまだK≤2なので、K≥3の配線は手順3。docs（`validation.md`・`usage/fe.md`・`methods/fe.md`・`fe-spec.md`・CHANGELOG）とエラーコードの生成物を更新済み。
`dummy_leverage`は、重複を許したことで個体内の行数の2乗に比例して遅くなるのを避けるため、個体ごとに時点の出現回数を先に集約してから`M`と`v`を作る
（式は`c_e`を出現回数のベクトルとして書いたままで成り立つ。重複を含む入力でダミー回帰のハット行列と一致）。LSDV照合は、重複あり・行数が`N×T`のまま重複と欠落が相殺する入力で
係数・古典/HC2/HC3の共分散・`fixed_effects()`を確認。Python: 全体実行（`--check-error-coverage`）11912件、再ビルド後の`tests/panel`＋`tests/cross_method` 8847件が通る。

**進捗（手順3、2026-11-08）**: 実装済み（コミット前）。`FeEffects`をFE専用のenum（`OneWay`・`TwoWay`・`MultiWay`）にし（REの`ReEffects`=`PanelEffects`と分離）、`FeInput::from_effect_groups`（K次元の入口）・
`TimeKeys::into_codes`（pybindが時点のコードを渡す経路）を足した。`fit_fe`は吸収する次元のVec（`absorbed_levels`）で分岐し、`MultiWay`は閉形式を使わず交互射影（重複する行も許す）。
自由度（`absorbed_parameters`）・クラスターのK（`fe_cluster_k_correction(levels, ..)`）・退化の検査・`PanelStructure::FixedEffects { levels }`をK次元にした（手順4の大半を含む）。
`fixed_effects()`も`FixedEffects::MultiWay`（次元ごとの(水準, 効果)）に対応し、不均衡な2-wayと同じ`absorbed_effects`で連結成分ごとに基準化する（手順6のengine側を含む）。pybindの`fixed_effects()`のdict化とpickle（`multi_way`タグ）も入れた。
HC2/HC3は3次元以上で`CommonError::Internal`（手順5まで）。`MAX_EFFECTS`は2のままで、Pythonからは3次元以上に到達できない（手順7）。
テスト: LSDV照合（係数・古典/HC1の共分散・残差・`fixed_effects()`の再構成と基準化）、入れ子の第3の効果（係数は2-wayと同じ・自由度は`k+N+T−1+(G−1)`）、クラスター/DK、特異グループ、次元の順序の不変性。
バグ注入（自由度・基準化）で3件が落ちることを確認済み。
レビュー反映: `FeEstimator::n_periods()`は2-wayだけ`Some`（3次元以上は`n_groups()`を使う）。pybindの`n_groups`は`FeEstimator::n_groups()`をそのまま使う（`MAX_EFFECTS`を上げる手順7で次元の数が合うように）。
テストを追加: 分断された3次元・重複する行・全組合せがそろう格子＋無作為な第3の効果・4次元・クラスター標準誤差のLSDV独立照合（個体にネストした効果をKに数えない）・エラー経路
（吸収される説明変数・効果で説明されるy・完全適合・掃引の上限）。**順序の注意**: HC2/HC3は3次元以上で`CommonError::Internal`のままなので、手順5を手順7（`MAX_EFFECTS`を上げる）より先に終える。
fixestで固定した係数・標準誤差の照合は手順8（ベンチマーク）で入れる。


**進捗（手順5、2026-11-08）**: 実装済み。`dummy_leverage`をK次元に一般化した（最大の次元を消去し、残りの次元の水準を1本に並べた
`M = W'W − Σ_g c_g c_g'/n_g`を密に作る。ネスト・連結成分によるランク落ちは列ピボット付きQRで独立な列を選んで解く）。上限3000は
「最大の次元を除いた水準数の合計」。全組合せがそろう2次元は閉形式のまま。`within_transform_multi_way`にも射影前の上限検査を入れた。
`LEVERAGE_DIMENSION_TOO_LARGE`の文面を一般化（ゴールデン・`error-codes.md`・`validation.md`・`fe-spec`更新）。ランク判定を誤ってレバレッジが
`[0, 1]`を外れたときは`MATRIX_NEAR_SINGULAR`にする。テスト: SVDによる総当たりとの一致（3・4次元、ネスト（消去する次元・残りの次元の間）、
重複行、連結成分が複数、タイ・1水準の次元、行数が偏った水準）、LSDVのHC2/HC3との一致、ネストした3つ目の効果が2-wayと一致、上限超過。
レビュー反映: `v = A c_g`を触った水準だけ計算（計算量`Σ_g d_g²`）、古い記述の更新。見送り: fixestとのHC2/HC3の3-wayクロスチェックは手順8で行う。

**進捗（手順7、2026-11-08）**: 実装済み（コミット前）。`MAX_EFFECTS`を`max_effects(model)`にし（FE=5、RE=2。`EFFECTS_TOO_MANY`の`max_columns`はモデル別）、
`extract_effect_groups`（effects[0]=個体の同一性キー、effects[1]=時点キー→`into_codes`、effects[2..]=個体と同じ同一性キー）を追加。`build_fe_input`は
`FeInput::from_effect_groups`と列数で選ぶ`FeEffects`（1/2/3+）。REは`extract_effects`のまま上限2。`fe.py`のdocstring（`effects`・`df_resid`・`df_model`・`n_groups`・
`fixed_effects()`・Raises）を更新。テスト: `tests/panel/test_fe_multi_way.py`（LSDVのnumpyオラクルで係数・古典/HC1/HC2/HC3・重複行・4〜5列・列の順序・キー型・singleton・
吸収される説明変数・クラスターの既定）、`test_effects_argument.py`（RE=3列は拒否、FE=3列は受理、FE=6列は拒否）、`test_validation_order.py`（6列で上限超過）、
`test_pickle.py`（3列の`fixed_effects()`の遅延・往復）。`pytest tests --check-error-coverage`は11930件が通る。ドキュメントは上限の記述だけ直した（本格的な更新は手順9）。
レビュー反映: 古いコメント（`lib.rs`・`fe.rs`・`common.rs`・pybindの`CLAUDE.md`）を更新、上限を`MAX_FE_EFFECTS`・`MAX_RE_EFFECTS`に、3列目の入力検査を
`test_key_columns.py`の`ENTITY_FITS`に追加（null・NaN・非対応dtype・タイムゾーン）、`fixed_effects()`の基準水準のdocstringを実装に合わせた
（3列目以降は文字列にした値の順で、整数列でも数値順にならない）。**手順9に持ち越し**: `docs/methods/fe.md`（`df_model`の一般形・多次元の交互射影・`fixed_effects()`の基準化・
ネストの扱い）、`docs/usage/fe.md`（「at most two columns」が古い。3〜5列の節と例）、`docs/api/fe.md`、`accepted-data.md`（3列目はIdentity扱いで`Boolean`も可）、
`FEOptions.tol`/`max_iter`の説明の「two-way unbalanced」だけという限定。**手順8**: fixestとの照合（`test_fe_multi_way_crosscheck.py`）。

**進捗（手順8、2026-11-08）**: 実装済み（コミット前）。判断（ユーザー確認済み）: シナリオは7種類＋小標本・不均一分散、第2参照は`AbsorbingLS`（classical/hc1のみ、
`debiased=True`、lsmrの打ち切りを1e-14に厳しくして機械精度）、フィクスチャとテストは専用（`fe_multi_way*.json`・`test_fe_multi_way_{reference,crosscheck}.py`）、
**クラスターKは現状維持**（クラスター列が効果の次元そのもののとき、fixestは他の次元の入れ子を調べず`K`が大きくなる。本実装は入れ子を全て数える。`nested`の`cluster_c`は
`K`の差の係数倍として検証し、効果の外の列`region`でfixestと厳密に一致することを確認）。fixestがHC2/HC3で失敗する`nested`は`entity + time`の式を参照にする。
データ: `benchmark/panel/multi_way_datasets.py`。ケース: classical/hc1/hc2/hc3/cluster/cluster_c/cluster_time/cluster_region/dk/dk_bw1。fixestと全シナリオ×全ケースが機械精度で一致
（係数3e-15、標準誤差3e-12、残差6e-12）。バグ注入（HC2/HC3のレバレッジを1%ずらす）で14件が失敗することを確認。`pytest tests --check-error-coverage --check-cov-coverage`が12107件通る。
ドキュメント: `verification.md`・`verification-details.md`・`inference-conventions.md`・`methods/fe.md`（Kの差）・`fe-spec.md`（テスト節）。
**見送り（レビュー指摘）**: 3次元の実データ（`wooldridge`キーは空）、悪条件・高次元k・クラスター数の境界（G=q+1）、`absorption_tol`の検証、`fe_multi_way_true_beta.json`を使うテスト。

**進捗（手順9、2026-11-08）**: 実装済み（コミット前）。公開ページ: `usage/fe.md`（新節「Three or more effects」、列順の節）・`api/fe.md`・`methods/fe.md`（一般形のモデル・`df_model = k + ΣG − (D−1)`・
交互射影・HC2/HC3のレバレッジと3000上限・クラスターKとfixestの差・`fixed_effects()`の基準化）・`guide/{accepted-data,api-conventions,validation,verification,verification-details,inference-conventions}.md`・
`CHANGELOG.md`・`docs/spec/fe-spec.md`・docstring（`fe.py`、`fe.rs`、`lib.rs`）。`/review-docs`を実施（校正2＋校閲2）し、must fix 3件・should fix 全件・一部のnice to haveを反映
（拒否条件の3列対応、検証表の実測値と尺度、fixestの`fixef.tol`が参照スクリプトの設定であること、between R²が列順に依存すること、2クラスター退化は「少なくとも1列」で成立、など）。
エラーカタログの2文面も3列対応に修正（`INSUFFICIENT_DEGREES_OF_FREEDOM`・`OUTCOME_CONSTANT_WITHIN_ENTITY`。ゴールデンと`error-codes.md`を再生成）。
Web確認（ユーザー許可済み）: `xtscc`はHoechle (2007, Stata Journal 7(3))のDriscoll-Kraay実装（確認済み）。fixestの`ssc()`の`fixef.K="nested"`の定義と`G/(G-1)`・`(n-1)/(n-K)`が別の項であること
（公式ページで確認）。reghdfeは「クラスターにネストした固定効果を自由度から除く」（`df_a_nested`）が公式ヘルプにあるが、補正式`G/(G-1)·(n-1)/(n-K)`は読めたページに無く、xtregの補正式は
公式ドキュメントを取得できなかった。そのため`inference-conventions.md`の「fixest・xtreg・reghdfeの利用者が期待する」を、fixestの既定`ssc()`だけを述べる文に直した。
`pytest tests --check-error-coverage --check-cov-coverage`が12107件、engine 1358件・engine_pybind 241件が通る。
**未着手**: 手順10（性能比較に3-wayを足す）。見送ったnice to have: `fe.rs`の`///`の折り返し、`verification.md`の表の行の置き換え（主リファレンスと独立クロスチェックの分離）、
cluster表の2行の統合、`accepted-data.md`の見出し分割。

### 9.11 compact用のメモ（#442、手順9のコミット後。2026-11-08）

**状態**: `release/v0.10.0`。手順1〜9はコミット済み（署名`G`）、**未push**（Pythonまで終わるまでpushしない。CIが落ちるため）、#442はクローズしていない。
コミット（新しい順）: `e99947ef docs: FEの3次元以上の固定効果の使い方・定義・検証を公開ドキュメントに反映する`（手順9）・`c7178a61 test: FEの3次元以上の固定効果をfixestとAbsorbingLSで照合する`（手順8）・
`d13f75f9 feat: FEのeffectsに3〜5列を…`（手順7）・`bd0eb8db feat(engine): FEで3次元以上のHC2/HC3のレバレッジを…`（手順5）・手順3の`4379606c`・手順2の`3421f66d`・手順1。各`docs:`の進捗コミットが別にある。
検証の最終結果: engine 1358件・engine_pybind 241件、`pytest tests --check-error-coverage --check-cov-coverage` 12107件が通る。clippy・fmt・ruffはクリーン。

**できあがったこと（FEは`effects`に1〜5列）**
- engine: K次元の入力（`FeInput::from_effect_groups`）、`FeEffects::{OneWay,TwoWay,MultiWay}`、常に交互射影（3列以上）、`df_model = k + ΣG − (D−1)`、クラスターKの一般化、
  K次元のHC2/HC3（`dummy_leverage`: 最大の次元を消去して残りを密に解く。ピボット付きQRでランク。上限3000は「最大の次元を除いた水準数の合計」）、`FixedEffects::MultiWay`（連結成分ごとに基準化）。
- engine_pybind・Python: `max_effects(model)`（FE=5、RE=2）、`extract_effect_groups`（3列目以降は個体と同じ同一性だけのキー）、`build_fe_input`。
- 検証: `benchmark/panel/multi_way_datasets.py`（9シナリオ）、`fe_multi_way*.json`（fixestが主、`AbsorbingLS`がclassical/hc1のクロスチェック）、`tests/panel/test_fe_multi_way{,_reference,_crosscheck}.py`。
  fixestとの差は2つ（`nested`のHC2/HC3はfixestが失敗するため`entity + time`の式を参照、クラスター列が効果の次元そのものの場合のKは本実装が入れ子を全て数える現状維持）。
- ドキュメント: usage/api/methods/guide/CHANGELOG/spec/docstring。`/review-docs`を実施済み。エラーカタログ2文面を3列対応に直した。

**次の作業: 手順10（性能比較に3-wayを足す）**
- 対象: `performance/compare_fe.py`（FE固有アダプタ、272行）と共通ハーネス`performance/_perf_harness.py`（948行。`PerfAdapter`・`FitContext`・`CovCase`）、開発ノート`docs/performance/fe.md`（日本語。方法論・既知の限界）。
  公開ページ`docs/guide/performance.md`・`performance-results.md`の数値はCIの計測値（`benchmark_performance.yml`）が正で、リリース準備時（`/release`）に再生成する（今は触らない）。
- 現状: Tを6に固定してNをスイープ（`_N_PERIODS_FIXED=6`）、`build_dataframe(n, k, seed)`は`generate_fe_dataset("baseline", ...)`、`default_method="one_way"`・`extra_methods=("two_way",)`
  （代表点1つ: classical・k=5・n=1,000,000）、リファレンスは`linearmodels.PanelOLS`（`entity_effects`・`time_effects`）。
- **linearmodels側の3-way**: `PanelOLS`は2効果まで。`AbsorbingLS`（`absorb`にカテゴリ列を渡す）が使えることは実測済み（手順8）。3-wayの比較相手はこれにする案が自然。
  ただし計測範囲を対称にする点に注意（`PanelResults`は遅延評価。`AbsorbingLS`の結果で何が遅延かを確認する）、`AbsorbingLS`はlsmrの打ち切りで速度が変わる（既定のatol・btol=1e-8で測るか、フィクスチャと同じ1e-14か。**要確認**）。
- **#634（別Issue、英語、OPEN）**: 不均衡2-wayシナリオの追加。`PerfAdapter.build_dataframe(n, k, seed)`が`method`を知らないため、ハーネスの拡張（methodまたは「データの変種」を渡す形）が要る。
  3-wayも同じ制約を受ける（3列目のデータが要る）ので、**#634のハーネス拡張と合わせるかをユーザーに確認する**（9.9・9.10の保留事項）。
  #634の未決: データの渡し方、測る不均衡の形（無作為な間引き。疎・鎖は既定にしない）、HC2/HC3の3000水準上限、`docs/performance/fe.md`と生成ページの更新。
- 確認すべき設計判断（14章。着手前に提示して確認する）: (a) #634と合わせるか、3-wayだけ先にやるか。(b) 3-wayのDGP（`multi_way_datasets.py`の`unbalanced`相当を大きなnで生成するか。三つ目の水準数、Tの扱い）。
  (c) linearmodels側に`AbsorbingLS`を使うか、engine単独にするか、その打ち切り設定。(d) cov軸の対象（classical・hc1が対等、cluster・dkは補正が違う）。(e) 3-wayの代表点（method軸の流用で`three_way`を足す案）。
- 実装後: `performance-reviewer`サブエージェント（`/review-performance`）でレビューする。計測は開発コンテナの単発で、公開ページの数値には使わない。コミットはパス指定。

**注意書き**
- コミットはパス指定。署名はホストの端末でパスフレーズの確認が出る。署名に失敗したら止まって報告（リトライしない）。Pythonを触ったら`uv run maturin develop`で入れ直してから`pytest`。
- 全体のpytestは約6分。エラーコードを足したら`python tools/render_error_catalog.py`・`tests/support/_errors.py`の`MIGRATED_CODES`。
- fixestで不均衡パネルを参照にするときは`fixef.tol`を厳しくする（既定の1e-6だと約5e-9ずれる）。`AbsorbingLS`は`method="lsmr"`、`absorb_options`のatol・btolを1e-14にして機械精度。
- 保留（利用者の判断待ち）: READMEの「Attribution and Rights」注意書き、フィクスチャのデータ（Wooldridge等）のライセンス確認、ルートCLAUDE.md 3章の`shared/`の列挙の更新（編集は承認が要る）、
  `docs/planning/specs/mop-oracle-spec.md`の履歴の扱い。見送ったレビュー指摘: 3次元の実データ、悪条件・高次元k・G=q+1の境界、`absorption_tol`の検証。
- IV本体の残り（#366〜#369）とFE-2SLS/GMM（#370〜#388）は#442の後（9.9参照）。#442のクローズはpush・CIの確認の後。

**手順10の結果（2026-11-08）**: コミット `6784aa17 fix(engine)`・`ade41e2f feat(performance)`（署名`G`、未push）。
- 3-wayのHC2/HC3で、個体×時点×無作為な3つ目の効果（約3,000行以上）が`NearSingularMatrix`になる不具合を計測中に発見して修正（`dummy_leverage`の`M`の引き算の丸め誤差が零固有値をランクの閾値より大きくしていた。TwoSumで補償）。Rust・Pythonの回帰テストを追加。
- 性能比較: method軸に`two_way_unbalanced`・`three_way`（比較相手`AbsorbingLS`の既定の打ち切り、classicalのみ）、cov軸に`hc2_three_way`・`hc3_three_way`（engine単独）。ハーネスは`PerfAdapter.method_dataframes`で、#634の不均衡2-wayも同時に済み。
  `performance-reviewer`の指摘（hc1分岐の削除、数値の修正、公開ページ手書き部分への追記、派生シード、データが良条件である旨）は反映済み。公開ページの表は`/release`の再生成まで更新しない。
- 検証: engine 1360件、`pytest tests --check-error-coverage --check-cov-coverage` 12109件が通る。
- 残り: Pythonまで終わったのでpush（利用者が実行）→CI確認→#442と#634のクローズ。保留事項は上記のとおり。

### 9.12 compact用のメモ（#366のコミット後。2026-11-08）

**状態**: `release/v0.10.0`。#442・#634はクローズ済み（PR #635、`dev`向け、CI全件green。`1689e77c`まで）。その後の**#366は`2c29a259`でコミット済み（署名`G`、未push）**。
#366のコミットはPR #635に積まれていない（pushすればPRに載る。pushは利用者に確認してから）。

**#366で決めたこと（ユーザー決定）**
- 公開の形: `IVResults.cragg_donald()`・`kleibergen_paap()`・`effective_f(*, alpha=0.05)`の**事後診断のメソッド**（`fit()`では計算しない。`fit()`の計算量は増えない）。
  結果が`source_data`（`fit()`に渡した`DataFrame`への参照）・`options`（`IVOptions`の複製）・`x_exog`/`x_endog`/`instruments`の列名を持ち、呼び出し時に`build_iv_input`で`IvInput`を作り直す（2SLSでもGMMでも同じ）。
- `cov_type`は`fit()`のものを使う（メソッドの引数にしない）。CDは等分散で`cov_type`に依存しない。
- 戻り値は診断ごとの専用frozen dataclass（`CraggDonaldResult`・`KleibergenPaapResult`・`EffectiveFResult`。`econometricsmodels.diagnostics`、トップレベルにも公開）。`DiagnosticResult`は継承しない（p値の無い診断があるため）。臨界値は水準をキーにした`dict[float, float]`（`json.dumps`でキーは文字列になる旨をdocstringに明記）。除外操作変数の数は3つとも`n_excluded`。
- 定義できないときは`None`でなく理由つきのエラー: `ALPHA_OUT_OF_RANGE`・`EFFECTIVE_F_ENDOGENOUS_COUNT`（`ValidationError`）、`WEAK_INSTRUMENT_STATISTIC_UNDEFINED`（`ComputationError`、`statistic`に診断名）。
  原因（共線・完全フィット・共分散のランク落ち・レバレッジ1）は1つのコードにまとめ、文面の括弧内に列挙するだけで区別しない。engineの`compute_*`は`Result<_, IvError>`、`shared`の下位関数は`Option`のまま。
- **有効Fは`Ŵ2`のランク落ち（最小固有値が最大の1e-10倍以下）を未定義にした**（KPに揃える。以前は値を返していた）。
- KPはLM・Wald Fを独立に`Optional`で持つ（両方未定義のときだけエラー）。rk LMのp値は`RankTest::p_value()`（`χ²(df)`）。
- 検証: CD・KPはlinearmodelsの`first_stage`（`cragg_donald_f`・`kleibergen_paap`・`kleibergen_paap_f`。**linearmodelsの`robust`のFは本実装の`hc1`、LMは`hc0`に対応**）、有効Fは`benchmark/iv/references/mop_oracle.py`。
  `tests/iv/test_iv_weak_diagnostics.py`（43件）。許容誤差`iv_weak_diagnostics`は1e-10（実測の最大は9e-14）。
- 登録済み: `test_pickle.py`の`TRANSIENT`、`test_gil_release.py`の`ROWS`/`CASES`、`test_keyword_only_arguments.py`の`_RESULT_METHODS`、`tests/support/_errors.py`の`MIGRATED_CODES`。
- ドキュメント: `docs/methods/iv.md`（定義式と出典）、`usage/iv.md`、`api/iv.md`、`guide/{inference-conventions,validation,verification,verification-details,pickling}.md`、`CHANGELOG.md`、`docs/spec/iv-spec.md`3.4.4節、`docs/spec/inference-conventions.md`6章（メソッドにした例外と専用dataclassの理由）。
  **未実施**: `/review-docs`による公開ドキュメントの校閲（実施してよいか利用者に確認）。
- 検証結果: engine 1360件・engine_pybind 241件、`pytest tests --check-error-coverage --check-cov-coverage` 12164件が通る。clippy・fmt・ruffはクリーン。`rust-reviewer`・`python-reviewer`の指摘は反映済み。
- 見送り: `cragg_donald()`が未定義になる経路のPythonテスト（`fit()`が通る入力では作れない。Rustの単体テストで担保）、`alpha`検証を列取り込みより前にする最適化。

**次の作業: #367（ivDiagの恒久導入と参照ベンチマーク）**
- 現状: ivDiag 1.0.8は恒久導入していない。`benchmark/iv/references/mop_ivdiag_compare.py`（一時ディレクトリに`remotes::install_version("ivDiag","1.0.8",lib=<dir>)`して`--r-lib <dir>`で手動実行）と`run_ivdiag_eff_f.R`があり、
  比較結果は`mop_oracle_values.md`の「ivDiag 1.0.8との比較」節。ivDiagが対応するのは`robust`（`hc1`相当）と`cluster`だけ。`eff_F`は有効Fの統計量だけで臨界値は返さない。
- やること（#367のIssue本文）: (1) `.devcontainer/Dockerfile`に`install_version("ivDiag", version="1.0.8", ...)`を足し、`Rscript -e 'library(ivDiag)'`で実際の導入を確認（`install.packages`は失敗が見かけ上成功扱いになる。`install_version`で）。`.github/workflows/check_r_updates.yml`の対象にも足す（新版の週次確認）。
  **devcontainerの再構築は利用者の作業**（再構築が済むまでR側の実行ができない。一時ディレクトリ導入で先に進める案もある）。
  (2) `benchmark/iv/references/`にivDiagのアダプタ。(3) `eff_F`の定式化（FE引数、複数内生変数）の対応状況はソースで確認。(4) fixture（`iv_*`系のJSON、`generate_*`、`cov`の分離規約）とpytest（`tests/iv/`、`/test-new`）。
- 確認すべき設計判断（14章、着手前にユーザーへ）: ivDiagを主参照にするか、独立クロスチェックにとどめるか（有効Fはオラクルが主、ivDiagは`hc1`・`cluster`のみ）。lfeの自由度補正の規約が本実装の`n_absorbed`と合うか（IV本体は`n_absorbed=0`なので今回は関係しない。FEIV #373で効く）。
  CD・KPのR参照（`ivreg`は持たない。ivDiagは持つか、`AER`/`ivpack`/`ivmodel`等の候補）。fixtureをCD/KP/有効Fで分けるか。
- 公開ページの`verification.md`の表（IV弱操作変数診断の行）に「Independent cross-check: none yet」と書いてあるので、#367で埋める（`verification-details.md`の「Weak-instrument diagnostics have one package reference」も更新）。

**その後**: #368（C統計量のengine。同じ重み行列を使うかをここで確定。疑わしい変数を内生/外生で2回GMM推定してHansen Jの差。事後診断のメソッドで、`IVResults`が`source_data`等を保持する仕組みは#366で入った）→#369（バインディング・ラッパー・テスト、#256も参照）→FE-2SLS（#370〜#380）→FE-GMM（#381〜#388）。

**注意書き**
- コミットはパス指定。署名はホストの端末でパスフレーズの確認が出る。署名に失敗したら止まって報告（リトライしない）。Pythonを触ったら`uv run maturin develop`（debugビルド）で入れ直してから`pytest`（releaseビルドだとテスト専用関数が無く`test_error_catalog`が落ちる）。
- 全体のpytestは約6〜8分。エラーコードを足したら`UPDATE_ERROR_CATALOG_GOLDEN=1 cargo test -p engine error_catalog`→`maturin develop`→`python tools/render_error_catalog.py`（ビルド後に実行）→`MIGRATED_CODES`。カタログのmessageは約320文字が上限。
- ルートCLAUDE.mdは、診断を「`DiagnosticResult`で返す」と書いている（2章）。今回の例外は`docs/spec/inference-conventions.md`に書いたが、CLAUDE.mdへの追記は利用者の承認が要る（未確認）。
- 保留（利用者の判断待ち）: READMEの「Attribution and Rights」注意書き、フィクスチャのデータ（Wooldridge等）のライセンス確認、ルートCLAUDE.md 3章の`shared/`の列挙の更新、`mop-oracle-spec.md`の履歴の扱い。
- Web検索は利用者の許可を得てから（#367で他パッケージの挙動を確認するときも）。

### 9.13 #367のコミット後（2026-11-08）

**状態**: `release/v0.10.0`。#367は`79b293b8`（ivDiagの恒久導入・アダプタ・fixture・pytest）、公開docsの校閲対応は`2195c136`、K≤2の有効Fの注意書きは続くコミット。**`79b293b8`以降は未push**（`5a5966e2`までpush済み）。

**#367で決めたこと（ユーザー決定）**
- ivDiagは**独立クロスチェック**（主参照はCD・KPがlinearmodels、有効FがMOPオラクルのまま）。fixtureは診断専用の1ファイル`iv_weak_diagnostics_crosscheck.json`。一時導入用の`mop_ivdiag_compare.py`・`run_ivdiag_eff_f.R`は削除してアダプタ`ivdiag_ref.py`・`run_ivdiag.R`に一本化（`mop_oracle_values.md`のivDiag節はFEありの行の記録として残す）。
- 内生変数1個・`hc1`/`cluster`の範囲だけ比べる。ivDiagの`F.standard`=CD、`F.robust`/`F.cluster`=rk Wald F、`eff_F`（`prec=14`。`ivDiag()`は4桁に丸める）=有効F。rk LM・臨界値・複数内生変数はivDiagに無い。
- 実測の最大相対差: CD 2.1e-14、rk Wald F 2.3e-14、有効F 1.7e-12（実データcard。合成は1e-13以下）。許容は`iv_weak_diagnostics_crosscheck`の`rtol`=1e-10。
- devcontainerの再構築は利用者の作業（Dockerfileは更新済み。再構築までは`R_LIBS=<一時ライブラリ>`でfixtureを再生成できる）。

**K≤2の有効F（ユーザー決定）**: 論文確認の結果、`critical_values=None`にはしない。MOP 2013はNagarバイアスの存在だけを根拠にK≤2でも進める（Lewis-Mertens 2022の記述。MOP原文は未取得）。Stock-YogoはK≤n+1の臨界値を載せない。docs・docstringに「1〜2本のときは`simplified_critical_values`を使う」と注意書きを足した。`classical`でK=2のとき`bias_bound=0`・`critical_values`が全τで同じ値になるのは式どおり。
Stock-Yogoの表は実装の値がNBER TWP 284のTable 1（バイアス）・Table 2（サイズ）と一致。2005年の章との一致は未確認（docsに明記済み）。

**校閲（/review-docs）の結果**: 指摘は全て対応済み（`inference-conventions.md`の手法別の表に診断の行を足す案だけは、診断を扱う表ではないので見送り）。エラーカタログの文面2件も直した。

**次の作業**: #368（C統計量のengine。同じ重み行列を使うかを着手前にユーザーへ確認）→#369→FE-2SLS（#370〜#380）→FE-GMM（#381〜#388）。
**保留（利用者の判断待ち）**: ルートCLAUDE.md 2章への診断の例外の追記、READMEの「Attribution and Rights」、フィクスチャのデータのライセンス確認、CLAUDE.md 3章の`shared/`の列挙、`mop-oracle-spec.md`の履歴の扱い。

### 9.14 #368・#369の完了時点のメモ（2026-11-09）

**状態**: `release/v0.10.0`。`f8c21c58`までpush済み。未push: `e2202d58`（#368 engine）・`2bc64e1c`（#369 バインディング・ラッパー・テスト・docs）。#368・#369はコミット済みでIssueは未クローズ（利用者の確認待ち）。この節を足すコミットも未push。

**決めたこと（ユーザー承認）**: 2SLSも対象（重みは`cov_type`の系統）、`gmm_type`・`cov_type`に依存せず常に2段階、除外操作変数の直交性も検定（`c_stat(*, variables=None, instruments=None)`）、戻り値は`CStatisticResult(DiagnosticResult)`。
**私の判断（要報告）**: 1段階GMM（`gmm_type="one_step"`、重み行列なし）は2SLSと同じく`cov_type`の系統の重み。`COLUMN_NOT_FOUND`は`hint`が`data`の列を指すため、モデルに無い名前は専用の`C_STATISTIC_UNKNOWN_VARIABLE`。外した後の操作変数が足りないときも専用の`C_STATISTIC_TOO_MANY_TESTED_INSTRUMENTS`（`INSUFFICIENT_INSTRUMENTS`は`hint`が「操作変数を足す」）。
**定数項なしの`classical`**: linearmodelsは残差を中心化して`σ̂²`を作るので一致しない（`robust`・`hac`は一致）。エンジンは`fit()`のHansen Jと同じ非中心化のまま。
**検証**: 内生性はlinearmodelsの`c_stat`、直交性・同時・Jは`IVGMM`の部品で組み直した`benchmark/iv/references/linearmodels_c_stat.py`、numpy独立実装`c_statistic_oracle.py`はそれらと一致（`test_c_statistic_oracle.py`）。フィクスチャJSONは無し（ライブ照合）。Rにはこの検定が無い（`momentfit`は`CDtest`・`MOPtest`等はあるがC統計量は無い）。許容は`iv_c_statistic`（1e-10。クラスター数が列数ちょうどのcard地域は1e-8）。
**起票**: #636（2SLSへの頑健な過剰識別検定）。
**未実施**: `/review-docs`による#367・#368・#369の公開docsの校閲（#366の校閲は済み）。Rでの独立クロスチェックを足すか（`momentfit`で重みを指定してJを手組みする案。独立性は限定的）は利用者に確認。
**次の作業**: FE-2SLS（#370〜#380）→FE-GMM（#381〜#388）。着手前に、`n_absorbed`の自由度の扱い（lfeとfixest）と、#374（FEのSargan）の設計をユーザーに確認する。

**ユーザーの決定（9.14の後）**
- 9.14の節の署名コミットと、`e2202d58`・`2bc64e1c`のpushはOK（実施済みなら下の「実施状況」を参照）。
- **#368・#369はcloseしてOK**（英語のコメントを付ける）。**#367〜#369の公開docsを`/review-docs`にかける**（対象: `docs/{usage,api,methods}/iv.md`、`docs/guide/{verification,verification-details,validation,inference-conventions,pickling}.md`のC統計量・ivDiagの変更箇所、`diagnostics.py`の`CStatisticResult`と`iv.py`の`c_stat`のdocstring、`CHANGELOG.md`）。
- **C統計量のRでの独立クロスチェックは足したい**（ユーザー希望）。`momentfit`以外にRで独立に見られるパッケージがあるかを先に調べる。現時点でわかっていること: `momentfit`にC統計量は無い（`CDtest`・`MOPtest`・`SWtest`・`LewMertest`・`hypothesisTest`・`specTest`のみ。`hypothesisTest`はパラメータの制約の検定で、モーメント条件を外す検定ではない）。`ivreg`・`AER`・`plm`（`pgmm`のSarganのみ）にも無い。未確認の候補: `gmm`パッケージ（モーメント関数を自作でき、重み行列を指定できるらしい。**CRAN上の機能はWeb検索で確認する。Web検索は事前に利用者の許可を得る**）。ivreg2（Stata）には`orthog()`があるがこの環境では実行できない。無ければ「`momentfit`（または`gmm`）で重みを指定してJを手組みする」案（独立性は限定的）を、独立性の限界を`verification-details.md`に書いて採用するかを、利用者に確認する。
- **次の順序**: (1) Rのクロスチェックの調査と追加 → (3) 上記のcloseと`/review-docs` → (4) FE-2SLSの前に、`n_absorbed`の自由度の扱い（lfeとfixestの違い。#373）と#374（FEのSargan）の設計を先に実施（設計をユーザーに確認してから）→ FE-2SLS（#370〜#380）→ FE-GMM（#381〜#388）。
- **この後はcompactを行う**（ユーザーが実施）。再開時はこの節を読んで(1)から続ける。

**保留（従来どおり）**: ルートCLAUDE.md 2章への診断の例外の追記、READMEの「Attribution and Rights」、フィクスチャのデータ（Wooldridge等）のライセンス確認、CLAUDE.md 3章の`shared/`の列挙の更新、`mop-oracle-spec.md`の履歴の扱い。
**注意書き**: コミットはパス指定・署名（ホストの端末で確認が出る。失敗したら止まって報告し、リトライは1回まで）。Pythonを触ったら`uv run maturin develop`（debugビルド）で入れ直してから`pytest`。エラーコードを足したら`UPDATE_ERROR_CATALOG_GOLDEN=1 cargo test -p engine error_catalog`→`maturin develop`→`python tools/render_error_catalog.py`→`tests/support/_errors.py`。heredocでRustやドキュメントを書くときはバッククォートをシェルが解釈するので`<<'EOF'`（引用付き）にする。
