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

> **2026-11-13の注記**: 以下の1.1・1.2は#442（K次元化）の前の記述が残る。現在のコードは、`PanelIndex`が`Vec<GroupCodes>`＋効果の列名の
> K次元、`from_group_codes(..., effects: Vec<GroupCodes>)`、効果の種類は`FeEffects`（`OneWay`/`TwoWay`/`MultiWay`）、2-wayは不均衡・重複行も
> 交互射影で受け付けバランスパネルの検証は無い、`cov_type`は`FeCovType`（=`PanelCovType`）。実装は9.18に従う。

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
7. 第一段階: 内生変数ごとに`FeEstimator`の後半（変換済みの列を受け取る内部経路。9.18）で回す（`y=x_endog[j]`、
   `x=x_exog ++ instruments`、同じ`effects`・`cov_type`）。失敗は`stage=FIRST_STAGE`付きで返す（`PanelError::InStage`）。`X̂ = [x̃_exog, x̂̃_endog]`
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
- Sargan（#374）: 構造残差を`Z̃ = [x̃_exog, z̃]`に回帰した`n·R²`相当。`σ̂² = e'e/(n − n_absorbed)`（9.17で`e'e/n`から変更）。自由度`len(instruments) - len(x_endog)`。
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
   手計算と完全に一致する（N=100・T=6で`e'e/(n-N)`だと5.617が4.681になる）。**9.17で変更**: #374は`e'e/(n − n_absorbed)`にする（`e'e/n`は小Tで過剰に棄却するため）。
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
3. **Sarganの`σ̂²`の分母（決定。9.17で変更）**: `e'e/(n − n_absorbed)`（fixestの`sargan`は`e'e/n`で、小Tで過剰に棄却する）。小Tで過大になる旨は
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

### 9.15 C統計量のRクロスチェックを追加（2026-11-08）

**決定（ユーザー承認）**: Rにはパッケージが無いので`momentfit`の部品で組み立てる方式を採用、範囲はシナリオ全体（`NUMERIC_SCENARIOS`・複数内生変数・切片なし・card）。`gmm`パッケージは調べない（既存GMMクロスチェックで「1段階目の重みが単位行列」を理由に不採用済み、Web検索もしていない）。
**実装**: `benchmark/iv/references/run_momentfit_c_stat.R`（`S`・推定・Hansen Jは`momentfit`、`S`の部分行列を重みにする手順とJの差はスクリプト）、`r.py::run_momentfit_c_stat_r`、`benchmark/iv/constants.py`の`C_STAT_*`（検定の定義。生成とテストで共有）、`generate_iv_c_statistic_crosscheck_fixtures.py`（`regenerate_all.py`に登録）、`tests/fixtures/benchmarks/iv_c_statistic_crosscheck.json`、`tests/iv/test_iv_c_statistic_crosscheck.py`（70件）、許容`iv_c_statistic_crosscheck`（rtol 1e-10。実測最大1.0e-11（card）。J_cの0は本実装もRも厳密に0なのでatol無し、cardのクラスターG=9境界も1.5e-12で緩和不要）。
**わかったこと**: `momentfit`の`evalWeights(..., "optimal")`はCLで`vcov()`と一致しない（`run_momentfit.R`のヘッダ2.と同じ件）ので`vcov(model, 2SLS係数)`から重みを作る。クラスターは`cluster=data.frame(...)`か式を渡す。iidの`vcov()`は中心化・n-1の分散なのでclassicalの`S`は`mean(e0^2)*Z'Z/n`を自前で組む。
**docs**: `verification.md`・`verification-details.md`（独立性の限界を明記）・`docs/spec/iv-spec.md` 3.9.1を更新。
**次**: (3) #368・#369のclose（英語コメント）と`/review-docs`、(4) `n_absorbed`（lfe/fixest）と#374の設計確認 → FE-2SLS。

### 9.16 #367〜#369の公開docsの校閲と修正（2026-11-08）

**経過**: #368・#369はclose済み（英語コメント付き）。`/review-docs`（accuracy×2・prose×2）を実施し、指摘は must/should/nice to have をすべて修正（ユーザー承認）。実装との食い違いは無く、直したのはドキュメント側のみ。
**主な修正**: `gmm_type="one_step"`は`cov_type`の系統の重み（methods/usage/api/inference-conventions/CHANGELOG/docstringを統一）、methodsの記号（`m`の衝突→`k_{2B}`・`m_B`・`\mathcal{T}`）、C検定の条件付き性、Wu-Hausmanの誘導文、`verification*.md`（直交性の参照は`IVGMM`の部品で組み直した実装・Rクロスチェックの独立性と範囲・許容誤差の行・`### IV`見出しの統合）、`inference-conventions.md`にC統計量の節を新設、validationのコード補足。
**Web確認（ユーザー許可済み）**: Kinal(1980)の本文は有料で読めず、二次資料（Phillips & Liu-Evans）で「2SLSのモーメントは過剰識別の次数まで存在」→操作変数1本は平均なし・2本は平均あり分散なし。docsは「1本は有限の平均なし、2本は有限の分散なし」に直し、Kinal(1980)を引用に追加。MOP 2013・Stock-Yogo 2005の該当箇所（K≤2の扱い）は確認できなかったので、MOPが「Nagarバイアスで進める」という記述は削除した。Hayashi 2000は3.6節がp.217から（部分集合の検定はp.218、別の部分集合検定がp.232から）まで確認、終了ページは未確認（「pp. 217-221 と p. 232 以降」に変更）。EHS 1988・BSS 2003の書誌は確認済み（EHSの題は`time-series`とハイフン）。
**未了**: 次は(4) `n_absorbed`の自由度（lfe/fixest、#373）と#374（FEのSargan）の設計確認 → FE-2SLS（#370〜#380）。`c372e9a7`とこのコミットは未push。

### 9.17 `n_absorbed`の数え方・FEのSargan・臨界値の小Tでの挙動（2026-11-08）

**状態**: `release/v0.10.0`。未push: `c372e9a7`（C統計量のRクロスチェック）・`a65367d5`（公開docsの校閲の修正）とこの節のコミット。実装は変えていない（調査と実測のみ）。実測スクリプトは作業用ディレクトリのみで、リポジトリには置いていない。

**決定（ユーザー承認）**
1. **`n_absorbed`はfixest流にそろえる**: 数え方は既存の`absorbed_parameters`（水準数の合計 − (効果の数 − 1)。連結成分の数やネストによるランク落ちは引かない）。小標本補正のKは既存の`fe_adjustment`のfixest`K.fixef="nonnested"`の規則を再利用する。`n_absorbed`は`cov_type`で変わるので、共有関数のシグネチャ（単一の数）は変えず、FEIV側が診断ごとに渡す数を決める: 古典の部分F・CD・HC系は全部の効果（`neffects`）、clusterは入れ子の次元を除いたK、DKは全部の効果（fixestの`K.fixef="full"`、`fe-spec.md`の記述どおり）。ドキュメントには、連結成分が複数あるパネルでlfe・ivDiagと1ずれる（lfeは階数を正確に数え、fixestと本実装は1つ多く数える）ことと、入れ子の第3の効果では3者ともランクより多く数えることを書く。
2. **Sargan（#374）の`σ̂²`は`e'e/(n − n_absorbed)`にする**（9.14以前の決定`e'e/n`を変更）。IV本体は`n_absorbed = 0`なので同じ規則で`e'e/n`になる。自由度は`len(instruments) − len(x_endog)`で、丁度識別は`None`（#374の本文どおり）。参照にできるRのパッケージは無い（fixestの`sargan`は`e'e/n`）。手計算の参照（行列版`sargan_test`の分母を引数にする）と、帰無仮説下のサイズを確かめるテストで固める。ドキュメントにはfixestとの差とその理由を書く。
3. **Stock-Yogoの臨界値は、吸収自由度を統計量の自由度に反映すればそのまま使える**（下の実測）。

**実測（再現に必要な条件）**
- 第一段階の部分F（x_exog 1列・操作変数2本・内生変数1個、1-way、N=100・T=6、l=3）: fixestの`ivf1`・lfe・ivDiagの`F.standard`・手計算（`n_absorbed = N`）が84.235（df2=497）で完全一致。`n_absorbed = 0`だと101.18。fixestの`cd`は101.01（`(n − n_endo − n_inst − 1)`でFEも外生変数も引かない）で、`kpr`はFE付きでは別の値になり、どちらも参照にならない。
- 古典の部分F（2-way）: 連結ならfixestとlfeが一致。連結成分が2つのとき、fixestは13.717（df2=112）、lfeは13.840（df2=113）。3-way（効果が入れ子）でfixestとlfeはどちらもKを69に数え、真の階数は65。
- HC1: fixest・lfe・ivDiagが一致し、`K = l + n_absorbed`（F=81.292）。
- cluster: 効果がクラスターに入れ子なら3者が一致し、`K = l + 1`。入れ子でない時点クラスターは`K = l + N`で一致。2-way以上ではfixestとlfeが割れる（個体クラスターで58.51と59.34、時点クラスターで103.46と124.02）。fixestは入れ子の次元を除いて数え、lfeは入れ子の効果をほぼ引かない。本実装はfixest側（FE本体と同じ）。
- **Sargan（帰無仮説下、N=100、3000回、名目5%、df=1）**: 棄却率は`e'e/n`でT=3が10.5%、T=6が7.3%、T=20が5.2%。`e'e/(n−N)`は4.8%・5.1%・4.7%。within残差の二乗和の期待値が`σ²(n − N)`なので、`n`で割ると統計量が`T/(T − 1)`倍になる。
- **臨界値の小T（K2 = 3、内生変数1個、Stock-Yogoのサイズ基準、ρ=0.99、2500回）**: 非心度λを臨界値で`qchisq(0.95, K, ncp = λ)/K`が一致するように置き（10%基準でλ=40.9、20%基準でλ=12.1）、同じλの横断面IV（n=600）とwithin変換（`n_eff = N(T − 1)`が約600のT=2・3・6・20、T=20はN=32）を比べた。部分F（df2 = n − l − N）が臨界値を超える確率は横断面が5.7%（10%基準）・4.9%（20%基準）、withinが5.7〜6.0%・4.7〜5.0%でT=2〜20に依存しない。Waldのサイズ（名目5%）は横断面9.4%・18.6%に対しwithinが10.0〜10.5%・18.2〜19.6%。
  - 吸収自由度を引かない（`n_absorbed = 0`）Fを使うと、10%基準で臨界値を超える確率がT=2で77.5%、T=3で45.6%、T=6で19.2%、T=20で9.2%と膨らむ。**統計量に`n_absorbed`を反映することが条件**。
  - 検証した範囲: 内生変数1個、等分散・正規、K2=3、`σ̂² = e'e/(n − N)`のz検定。KP（頑健）・内生変数が2個以上・小さいNは検証していない。

**保留（ユーザー未決。実装時に確認）**
- 参照の選び方（案）: 内生変数1個の古典・HC1・1-wayクラスターはivDiagとfixestの`ivf1`、2個以上のCD・KPはダミー列を入れたlinearmodels（小さいN、FWL）、入れ子のクラスターと連結成分が複数あるパネルはfixest・手計算。
- `n_absorbed > 0`のHC2/HC3のKP・MOPは今は`None`。FE-2SLSで`hc2`/`hc3`を許すときの診断の扱い（`None`のままか、ダミー込みのレバレッジを計算するか）。
- Sarganは等分散前提で`cov_type`に依らない。頑健な過剰識別検定（#636、2SLSのHansen J）にFE-2SLSを含めるか。
- FE-GMMのHansen J（within変換後の誤差は個体内で相関する。3章7のとおり#381で確認）。

**次**: FE-2SLSの実装（#370〜#372、#374、#375 → #373 → #376〜#380）。#374は`sargan_test`を`σ̂²`の分母つきで使い、#373は#363〜#365の後。

### 9.18 FE-2SLSの#370〜#372の実装計画と決定（2026-11-13）

**範囲**: engineのみ（公開面のpybind・Python・docsは#376以降）。#370（入力型・API設計）は入力型が実装済み（`fe_iv.rs`の`FeIvInput`）のため、
ファサード＋子モジュールへの分割と1章の記述の更新で完了。#372（自由度）は`fit`に最初から入るので#371に含めて同時に完了する。診断（#373〜#375）は含めないが、
第一段階の残差と`X̂`を保持する。

**決定（ユーザー承認）**
1. **変換は列ベースに切り出す**: 先にリファクタ（挙動を変えない）で、`fe/within.rs`の変換を「列の集合と吸収する次元」を受け取る関数にし、`FeInput`版をラッパーに残す。
   続けて`fit_fe`を「検証＋変換」と「変換済みの列を受け取る後半（df・回帰・共分散・推論・R²）」に分ける。FEIVは全列（`y`・`x_exog`・`x_endog`・`instruments`）を
   1回だけ変換し（`noise`も共有）、第一段階と構造式で使う。安全網は既存のFEのテスト。
2. **第一段階の共分散はIV本体と揃える**: 標準誤差が計算できない場合はエラー（`stage=FIRST_STAGE`付き）、それ以外はFを`None`にする。IV本体も、
   第一段階の`l`個の傾きに対してクラスター数が`l`以下でも、標準誤差は計算し第一段階のFだけ`None`にする（`docs/methods/iv.md`）。FEの`panel_cov_params`は
   クラスター数・時点数が検定する傾きの数以下だと（Wald Fのために）弾くので、`PanelCovInput`に「Fを検定するか」を足し、第一段階では検定せず
   （`G > q`・`T > q`の検査を省き、`f_statistic`を`None`にする）。標準誤差に関わる検査（`G >= 2`、2グループ退化、完全適合）は残す。構造式は従来どおり`G <= k`でエラー。
3. **定数列のエラーに`role`パラメータを足す**: `REGRESSOR_CONSTANT`・`REGRESSOR_NO_WITHIN_VARIATION`に`role`（`x`（FE）・`x_exog`・`x_endog`・`instruments`）を足し、
   文面とhintを`{role}`で書く。`y`の`OUTCOME_CONSTANT`・`OUTCOME_CONSTANT_WITHIN_ENTITY`は変えない。FEの既存の挙動・テスト・docsは`role="x"`に直す。
4. **#371にR²（within/between/overall、構造残差ベース）と`fixed_effects()`を含める**。

**実装順（コミット）**: (1)この節 → (2)列ベースの変換への切り出し → (3)`fit_fe`の後半の切り出し → (4)`fe_iv.rs`をファサード＋子モジュールに分割 →
(5)`role`パラメータとエラーの追加 → (6)`FeIvEstimator::fit` → (7)テスト（LSDVの2SLS・fixestとplmの値・手計算のHC2/HC3・性質・cov_type別のdf）→ (8)レビュー・clippy・全テスト → (9)進捗の追記とIssueコメント。

### 9.19 #370〜#372の実装結果（2026-11-13）

**状態**: engineのみ実装済み（`FeIvEstimator::fit`。公開面のpybind・Python・docsは#376以降）。#370（入力型・API設計）・#371（点推定・`cov_type`別SE）・#372（自由度）は
この実装で完了。診断（#373〜#375）は未実装だが、第一段階の推定量（`first_stages()`、変換済みの列・残差）を保持している。
**実装**: 9.18の決定どおり。`fe/within.rs`の`within_transform_columns`（列ベースの変換）、`fe/fit.rs`の`fit_fe_from_within`（`fit_fe`の後半）、`fe/fixed_effects.rs`の`fixed_effects_of`、
`FeEffects::levels`、`PanelCovInput::tests_slopes`、`ColumnRole`（`REGRESSOR_CONSTANT`・`REGRESSOR_NO_WITHIN_VARIATION`に`role`）、`PanelError::{InsufficientInstruments, SecondStageFailed}`、
`fe_iv.rs`（ファサード）・`fe_iv/{input,fit}.rs`。
**検証**: fixest（1-way・2-way×古典・HC1・個体クラスター・時点クラスター、1e-9）、ダミーを明示した2SLS（LSDV。1-way・バランス/不均衡な2-way・3次元×古典・HC1〜HC3・クラスター、
内生変数2個・操作変数3個・外生変数0〜2個、入れ子でないクラスターの`K`、`fixed_effects()`とダミーの係数）。DKはラグ0が時点クラスターと一致。HC2/HC3はLSDVの自己整合のみ（外部の参照は#379でplm・
`ivreg`＋`sandwich`）。`rust-reviewer`の指摘は、規約・テストの不足・perfect fitの誤差の見積もり（`|β̂_j|`×`x`の尺度を足す）を反映した。
**見送り（要判断・後続）**:
- 第一段階が保持するメモリ（内生変数ごとに生の列・変換済みの列・吸収した効果の複製）と、構造式用の`FeInput`の複製は、n=1e6で効く。性能比較（#380）で測ってから、借用・移動に直す。
- DKの第一段階で`T`が`l`以下のとき、DKの共分散がランク落ちするため標準誤差の信頼性が下がる可能性がある（IV本体のクラスターと同じ扱い。判断が要る）。
- `fit_fe_iv`が約300行（第一段階とR²・構造式の組み立てを関数に切り出す案）、`from_group_codes`の名前と列の数の検査が`debug_assert`のみ（`FeInput`と同じ）、proptest未整備。
- テストファイル名（`lsdv`・`structure`は標準のテーマ名に無い）。
**次**: #374（Sargan。`σ̂² = e'e/(n − n_absorbed)`）・#375（Wu-Hausman）→ #373（弱操作変数診断。#363〜#365の後）→ #376〜#380。

### 9.20 compact用のメモ（#370〜#372のclose後。2026-11-13）

**状態**: `release/v0.10.0`は`58be9153`までpush済み（9.19まで）。#370・#371・#372はclose済み（英語コメント付き）。PR #635は未マージ（内容確認後にマージ）。
**次の順序**: #374（Sargan）→ #375（Wu-Hausman）→ #373（弱操作変数診断）→ #376（pybind）→ #377（Pythonラッパー）→ #378（docs）→ #379（参照ベンチマーク）→ #380（性能）→ FE-GMM（#381〜#388）。
どれも着手前に、判断が分かれる点をユーザーに確認する（CLAUDE.md 14章）。実装は`/implement-rust`の流れ（実装→clippy・fmt・テスト→`rust-reviewer`）。

**#374（FEのSargan）の着手メモ**
- 決定済み（9.17）: `σ̂² = e'e/(n − n_absorbed)`（`n_absorbed = neffects`）、自由度`len(instruments) − len(x_endog)`、丁度識別は`None`、`cov_type`に依らず等分散前提。
- 部品: `shared/instrumental_variables.rs`の`sargan_test(z, residuals, sigma2, df)`（`σ̂²`は引数。`Z`は変換後の`[x̃_exog, z̃]`）。`FeIvEstimator`は構造残差（`residuals()`）と第一段階（`first_stages()`。各`within_input()`の`x()`が`[x̃_exog, z̃]`）を持つので、`Z̃`は第一段階の`within_input().x()`から作れる（内生変数が1つ以上ある前提）。
- 公開形は`fit()`時に計算してプロパティ（CLAUDE.md 2章。IV本体のSarganと同じ）。結果の置き場は`FeIvEstimator`のフィールド（`sargan: Option<SarganTest>`）。
- テスト: 参照はRに無い（fixestの`sargan`は`e'e/n`）。手計算（`Z̃`・構造残差・`σ̂²`）と、帰無仮説下のサイズを確かめるテスト（9.17の実測: N=100・T=3/6/20で名目5%）。ダミー込みの2SLSのSargan（`σ̂² = e'e/(n − K)`の`K`はダミー込みの係数の数）と一致するはず（FWL）。
- 要確認: 結果の名前（`overid_*`）、`Hansen J`との関係（#636: 頑健な過剰識別検定をFE-2SLSに含めるか。未決）。

**#375（Wu-Hausman）の着手メモ**: 補助FE回帰`ỹ ~ [x̃_exog, x̃_endog, v̂]`（`v̂`は第一段階の残差。`FeIvEstimator::first_stages()[j].residuals()`）を`cov_type`連動で回し、`v̂`列のWald。`shared/instrumental_variables.rs`の`wu_hausman_columns`は列の組み立てだけ。分母自由度は補助回帰のもの（`n − k − k_endog − neffects`）。参照はfixestの`wh`（古典のときだけ直接比較。それ以外はfixestで`v̂`を足した補助回帰を同じ`vcov`で回す。9.x 7章2）。
**#373の着手メモ**: #363〜#365（IV本体の共有関数。`n_absorbed`を受け取る）の後。`n_absorbed`は診断ごとに渡す数が違う（9.17: 古典の部分F・CD・HC系は`neffects`、クラスターは入れ子を除いたK−k、DKは`neffects`）。KPのHC2/HC3は`n_absorbed > 0`で`None`のまま（FE-2SLSの`hc2`/`hc3`での診断の扱いは要判断）。臨界値はそのまま使う（9.17のMonte Carlo）。

**未決・保留（ユーザー判断）**: DKの第一段階で`T`が`l`以下のときの標準誤差の信頼性（9.19）、#636へのFE-2SLSの追加、HC2/HC3の参照（#379）、FE-GMMのHansen J（3章7）。IV側の保留（ルートCLAUDE.md 2章への診断の例外の追記、READMEの権利の注意書き、フィクスチャのデータのライセンス、3章`shared/`の列挙、`mop-oracle-spec.md`の履歴）は従来どおり。
**注意書き**: コミットはパス指定・署名（ホストの端末で確認。失敗したら止まって報告、リトライは1回まで）。Pythonを触ったら`uv run maturin develop`（debug）で入れ直してから`pytest`。エラーコードを足したら`UPDATE_ERROR_CATALOG_GOLDEN=1 cargo test -p engine error_catalog`→`maturin develop`→`python tools/render_error_catalog.py`→`tests/support/_errors.py`。heredocでバッククォートを含む文章を書くときは`<<'EOF'`。rustfmtが整形すると、あとから文字列置換が当たらなくなるので、置換は件数を`assert`する。

### 9.21 #374（FE-2SLSのSargan）の実装結果（2026-11-13）

**決定（ユーザー承認）**: 結果の名前はIV本体と同じ`sargan_statistic()`・`sargan_p_value()`・`sargan_df()`。FE-2SLSを#636（頑健な過剰識別検定）に含めるかは#374では決めず、#636に残す。

**実装**: `fe_iv/sargan.rs`の`fe_sargan_test`が共有の`sargan_test`を`σ̂² = e'e/(n − neffects)`で呼ぶ。`fit_fe_iv`が構造残差の後に、`Z̃ = [x̃_exog, z̃]`（第一段階の設計行列と同じ列）で計算する。丁度識別と内生変数なしは`None`（IV本体は内生変数なしだと`df = len(instruments)`で検定するので、そこだけ違う。FEIVは第一段階が無い構成を対象にしない）。
**テスト**（`fe_iv/fit/tests/sargan.rs`）: ダミー込みの2SLSでの再計算（1-way・2-way均衡/不均衡・3次元×操作変数・内生変数の数5通り）、分母の手計算、`cov_type`不依存、丁度識別・内生変数なしの`None`、帰無仮説下の棄却率（N=100、1500回。T=3・6で3.5〜6.5%）。変異検査: 分母を`n`にするとT=3の棄却率が11.9%になり3件失敗。
**fixestの確認**: fixest 0.14.2の`feols`のソースで、`stat = n(1 − SSR_r/SSR_null)`（`e'e/n`の尺度）。
**見送り（reviewerのnice to have）**: `Z̃'Z̃`のCholeskyが条件数の悪い入力で`INTERNAL_ERROR`になりうる（IV本体と同じ挙動。実害が出たら`least_squares`に替える）、連結成分が複数ある不均衡2-wayと`n − neffects`が小さい境界のテスト（`/review-testing`で確認）。

### 9.22 #375（FE-2SLSのWu-Hausman）の実装結果（2026-11-13）

**決定**: 名前はIV本体と同じ`wu_hausman_statistic()`・`wu_hausman_p_value()`・`wu_hausman_df()`（#374と同じ方針）。F分布の自由度は`(k_endog, 補助回帰のdf_inference)`（Clusterは`G − 1`、DKは`T − 1`、それ以外は`n − k − k_endog − neffects`）。
**実装**: `fe_iv/wu_hausman.rs`の`fe_wu_hausman_test`。補助FE回帰`ỹ ~ [x̃_exog, x̃_endog, v̂]`を`fit_from_within(.., tests_overall_f = false)`で`cov_type`と同じ共分散で回し、`v̂`の係数のWald F。`v̂`は第一段階の残差（within空間の列）をそのまま使い、`absorbed`は渡さない（固定効果の復元に使わない）。構造式の共分散・F検定の後に呼ぶ（`DegenerateClusterTwoGroups`など構造式のエラーが補助回帰のエラーに隠れないように）。`None`になる条件: 内生変数なし、補助回帰の自由度不足・`v̂`の分散ゼロ・完全適合（交互射影の誤差の見積もりは構造式と同じく`y`と係数倍した各列）・特異・Waldが数値的に特異。それ以外の失敗は`stage=WU_HAUSMAN_REGRESSION`つきで返す。クラスター数・時点数は構造式の`G > k`・`T > k`が`k_endog`以下を除くので、補助回帰で足りなくなることはない。
**参照（fixest 0.14.2、`data.rs`のデータ）**: `fitstat(m, "wh")`は補助回帰の`v̂`のt²（iid）と一致（1-way 4.94841、2-way 3.24120。p値も一致）。hetero（HC1）とcluster（個体）は、同じ補助回帰を`vcov`を変えて回した`v̂`のt²（1-way: 6.30952・5.57364、2-way: 4.42762・3.68746）と1e-9で一致。
**テスト**（`fe_iv/fit/tests/wu_hausman.rs`）: 上のfixest照合、ダミー込みの補助回帰を既存の`lsdv`オラクル（内生変数・操作変数なしの入力で回す）で組み直した再計算（1-way・2-way均衡/不均衡・3次元×古典・HC1〜HC3・個体クラスター×(外生, 内生, 操作変数)4通り、時点クラスター）、DK≡時点クラスター、内生変数なし・完全適合の`None`。
**レビュー（rust-reviewer）への対応**: `fit`の`# Errors`に診断の`None`と`stage=WU_HAUSMAN_REGRESSION`を追記。完全適合の見積もりで、`v̂`列のスケールを`x_endog`の流用から第一段階の係数を使った`|x_endog[j]| + Σ|π_l|·|列_l|`に直した（`abs_max`は構造式と共有）。IV本体との違い（補助回帰は`tests_slopes=false`なので、`G`が`k+1`〜`k+k_endog`でも`v̂`の`k_endog × k_endog`部分行列で検定でき、値を持つ）をdocに明記し、ダミー込みの補助回帰との一致でテストした。補助回帰だけが自由度不足になる境界（3個体×2時点）も`None`でテストした。**限界**: 完全適合の見積もりの式は、無効にしてもテストが通る（補助回帰の`fit_from_within`の判定が先に効く。構造式の同じ式と同様に防御的な保険）。
**見送り（nice to have）**: 補助回帰の`tilde`・残差の複製、補助回帰が捨てるR²・推論表の計算、HC2/HC3の不均衡2-way以上のダミーのレバレッジの再計算（構造式・第一段階・補助回帰で繰り返す。`levels`だけに依るのでキャッシュの余地は#380）、Waldが`NaN`のときに`None`にするかの方針（Sargan・IV本体と合わせて#373で）、`{endog}_first_stage_resid`が利用者の列名と衝突した場合、Wald部分行列が数値的に特異な経路のテスト、`docs/spec/`への集約（実装完了時）。

### 9.23 #373（FE-2SLSの弱操作変数診断）の実装結果（2026-11-13）

**決定（ユーザー承認）**: 公開の形はIV本体と同じ（部分Fは`fit()`時のプロパティ、CD・KP・有効Fは推定後のメソッド。`cov_type`は`fit()`のもの）。**DKのKP・有効Fは実装する**（共有の`ScoreCovariance`に`Dk`を足した）。**`n_absorbed > 0`のHC2/HC3は、ダミー込みのレバレッジを計算する**（`ScoreCovariance::HcAbsorbed`）。Stock-Yogoの臨界値は表をそのまま使う（9.17）。
**実装**: `shared/score_covariance.rs`に`Dk{time, bandwidth}`（時点ごとのスコアの和にBartlett核。小標本の係数は`cluster_correction(T, n, n_params)`）と`HcAbsorbed{variant, leverage}`を追加（`hc_row_scales`・`score_meat`のクロージャが吸収した効果のレバレッジを受け取る）。結果の型（`CraggDonald`・`KleibergenPaap`・`StockYogoCriticalValues`）は`iv::common`から`shared/`へ移し、`iv::common`から再エクスポート（公開パスは同じ）。`PanelError`に既存のカタログのcodeを使う3バリアント（`WeakInstrumentStatisticUndefined`・`EffectiveFEndogenousCount`・`AlphaOutOfRange`）を追加。`fe_iv/weak_instruments.rs`が部分F・CD・KP・有効Fを、第一段階の設計行列から取り出した行列で共有関数に渡す。`n_absorbed`は9.17のとおり診断・`cov_type`ごと（部分F・CD・`Classical`・HC系・DKは`neffects`、`Cluster`は`fe_cluster_k_correction`の`K − l`）。HC2/HC3のレバレッジは`leverage_full(0, levels)`（within変換後の素のレバレッジを0にしたダミー分）。
**テスト**: ダミーを回帰子に入れた（`n_absorbed = 0`）IV本体の`compute_*`との照合（1-way・2-way均衡/不均衡・3次元 × `Classical`・`Hc1`〜`Hc3` × (外生, 内生, 操作変数)3通り。部分F・CD・rk LM・rk Wald F・有効F・Stock-Yogoの臨界値）。クラスターは、時点クラスター（個体FEが入れ子でない）はダミー込み、個体クラスター（入れ子）は個体内で平均を引いた列に切片を足した（`n_absorbed = 1`）IV本体の診断と一致。DK（ラグ0）は時点クラスターと一致（1-way）。共有層は、DKのmeatを手書きのBartlett和と、吸収したレバレッジをダミーを回帰子に入れた行スケールと比較。変異検査（ダミーのレバレッジを0・クラスターの`K`を`neffects`）でテストが落ちることを確認。
**限界**: DKのKP・有効Fには独立な参照が無い（fixestの`kpr`は参照にならない。ivDiagにも無い）。共有層は手書きの定義式との照合とクラスターとの恒等式だけ。HC2/HC3のKP・有効Fの外部参照も無い（IV本体のHC2/HC3と同じく、手計算とダミー込みの再計算）。連結成分が複数ある不均衡なパネルのクラスターは`n_absorbed`の数え方（9.17）。
**レビュー（rust-reviewer）への対応と、実測で分かったこと**:
- **DKでrk LMが自明な値になる（`T ≤ d`、`d = 除外操作変数の数 − 内生変数の数 + 1`）**: 時点を80%に絞ったデータ（T=3、操作変数3本、内生変数1個）で、rk LMがデータに依らない`3.0`・`2.0`を返した。時点ごとのスコアが`d`次元を張るのに足りず、`T = d`・`bandwidth = 0`でLMがちょうど`T`になる（`ξ_t`が正方で、`Σξ_t`の二次形式が定数になる）。rk Wald Fはスコアの和が0なのでランク落ちとして固有値の閾値が`None`にする。→ KPのLMで`Dk`かつ`T ≤ d`を未定義にした（`shared/kleibergen_paap.rs`）。有効Fのスコアは`2q`次元だが使うのは`q × q`の`Ŵ2`だけで、ランク落ちは既存の閾値が未定義にするので、同じ検査は要らない（`score_meat`に入れると過剰）。
- **IV本体のクラスターにも同じ性質がある（既存の挙動。今回は変えていない）**: `compute_kleibergen_paap`で内生変数1個・操作変数3個（`d = 3`）・クラスター3つ（`G = d`）のとき、rk LMはちょうど`3.0`（= `G`）、rk Wald Fは`None`。`G = 4`ではWald Fが1.8e6になる。LMを`G ≤ d`で未定義にするかは、IV本体の挙動を変えるので別に判断（公開前のレビューで）。
- DKの結果は`dk_bandwidth_used`（`fit`が解決した値）で診断に渡る（既定のバンド幅と明示指定が一致、バンド幅を変えると統計量が変わる）ことをテストした。バンド幅>0のDKの独立な参照は無い（meatは手書きのBartlett和との照合のみ）。
- 2-wayのクラスター（個体クラスター→時点をダミー、時点クラスター→個体をダミー。均衡・不均衡）も、個体内（時点内）で平均を引いた列に入れ子でない次元のダミーと切片を足したIV本体の診断と一致（`fe_cluster_k_correction`の「部分的に入れ子」の分岐）。
- `weak_instrument_f_df`の計算はファサードから`weak_instruments.rs`に移した。古くなっていたdoc（`kleibergen_paap.rs`・`effective_f.rs`・`fe_iv.rs`の自由度）を直した。
**見送り（nice to have）**: HC2/HC3のダミーのレバレッジを診断の呼び出しごとに再計算（不均衡な2-way以上は密な行列。`fit`で保持するかは#380）、`matrices()`のコピー、`Hc`と`HcAbsorbed`の並立の整理、非空のデータでの未定義の経路（共線・レバレッジ1）のFE側のテスト、公開側のドキュメント（`n_absorbed`の診断別の違い、DKの`T ≤ d`、HC2/HC3）は#378。

### 9.24 #376（FE-2SLSのpybind）の実装結果（2026-11-13）

**決定（ユーザー承認）**: `FEIVOptions`に`estimator`は今は入れない（GMM実装時にキーワード引数で足す）。エラーの`model`に`"FEIV"`を足す（カタログの`model`は自由な文字列なので変更なし）。#376の範囲は`_lib`（Rust側）と、`_lib`を直接見るテストまで（Pythonラッパー・公開docs・ラッパー経由の横断テストは#377・#378）。6.1の決定（`FEIV`/`FEIVOptions`/`FEIVResults`、第一段階は`dict[str, FEResults]`で固定効果をその場で計算、診断・`fixed_effects()`は`source_data`から再計算、R²は3つ）のとおり。
**実装**: `engine_pybind/src/panel/fe_iv.rs`（`FEIVOptions`・`FEIVResult`・`FeIvColumns`・`build_fe_iv_input`・`fit`）、`lib.rs`（`fit_fe_iv`の登録・`_restore_pickled`）。`panel/fe.rs`は`FEResult`を`Clone`にし（`LazyFixedEffects`が`FeEstimator`を`Arc`で共有）、`fe_result_fields`（借用でFEResultを作る）・`fixed_effects_dict`・`resolve_fe_cov_type`（`CovColumns`）を切り出した。`panel/common.rs`に`PanelModel::FeIv`と`validate_fe_iv_roles`。IVの`diagnostics.rs`に出力型の`from_engine`を足し、FEIVが再利用する。
**検証**: `tests/panel/test_fe_iv_binding.py`（27件）— fixestの係数・標準誤差（1-way・2-way × classical・hc1・cluster）、Wu-Hausman（補助回帰のt²）、ダミーを明示した2SLSのnumpy実装との係数・Sargan・部分F・Cragg-Donald・古典のKP/有効F、第一段階・固定効果・DKのバンド幅・pickle・エラー（役割・hc0・effectsの上限・識別）。`test_pickle.py`の`SAMPLES`に`FEIVOptions`・`FEIVResult`。`tests/iv`・`tests/panel`・`tests/cross_method`は全件通る。
**残り（#377以降）**: Pythonラッパー（`FEIV`・`FEIVResults`・`first_stage`を`dict[str, FEResults]`に、診断のdataclass）、`__init__`・`__all__`、`test_gil_release`・`test_call_signatures`・`test_keyword_only_arguments`・`test_results_docstrings`等の横断テスト、`docs/guide/api-conventions.md`の表、公開docs（#378）。
**レビュー（rust-reviewer）への対応**: テストを`_assertions.assert_close`と`TOLERANCES["fe_iv_binding"]`（`_tolerances_panel.py`。実測の最大相対誤差は1e-10未満）に移し、オラクルをnumpyのダミー変数2SLS（複数の内生変数・任意の次元の固定効果・classical/hc1/hc2/hc3の共分散・Sargan・部分F・固定効果）に一般化した。1-way・2-way・不均衡な2-way・3次元 × 内生変数1個・2個・丁度識別 × hc1〜hc3の係数・共分散・標準誤差・t・p値・信頼区間・Sargan・部分F、within R²・Wald F、第一段階の係数・標準誤差・固定効果、2-wayの固定効果とダミー係数の一致、DK（ラグ0）と時点クラスターの一致、オプションの組合せ・値のエラー、型エラー、`copy`/`deepcopy`/pickle（結果・第一段階・診断）、`fit`後のオプション変更の影響を受けないこと、を確認（89件）。診断・`fixed_effects()`は呼ぶたびに再推定することをdocに書いた。
**見送り・ユーザー判断が要る点**: (1)`first_stage`の持ち方: 現在は第一段階ごとに固定効果をその場で計算した`FEResult`（`Clone`、`Arc`共有）を持ち、`first_stage()`は複製を返す（O(n)の複製）。reviewerの代案は`Py<FEResult>`を持って`clone_ref`で返す形（複製が消え、`FEResult`を`Clone`にしなくてよい）だが、第一段階の`FeEstimator`を保持するか固定効果を先に計算するかは6.1で決定済みなので変えていない。(2)`fe_result_fields`に`placeholder_effects()`を渡して差し替える形（型で守れていない）、(3)`FEIVOptions`・`FEIVResult`の`FEOptions`・`FEResult`との重複（マクロ化）、(4)辞書の順序が非決定的（`HashMap`。IVと同じ流儀。Pythonラッパーで`x_endog`の順に直す）、(5)fixestの参照値のフィクスチャ化（`benchmark/panel/`の生成スクリプト）は#379、(6)ラッパー段階（#377）で`test_gil_release`・`test_call_signatures`・`test_keyword_only_arguments`・`test_validation_order`・`test_results_docstrings`・`api-conventions.md`の表に登録する。

### 9.25 clear用のメモ（#376のコミット後。2026-11-13）

**状態**: `release/v0.10.0`。**未push**（push はユーザーの指示があるまで行わない）: `645d4c68`（9.20のメモ）・`c6282534`（#374 Sargan）・`29387123`（#375 Wu-Hausman）・`8caa96aa`（#373 弱操作変数診断）・`a0b58e89`（#376 pybind）とこのメモのコミット。pushは`58be9153`まで。#374〜#376のIssueはまだ**close していない**（push後、英語コメントつきでユーザー承認を得てclose）。PR #635は未マージ（内容確認後にマージ）。
**実装の結果**: 9.21（#374）・9.22（#375）・9.23（#373。DKのrk LMが`T ≤ d`で自明な値になる件、IV本体のクラスターで`G = d`のLMがちょうど`G`になる既存の性質）・9.24（#376）。
**次の順序**: #377（Pythonラッパー）→ #378（公開docs）→ #379（参照ベンチマーク。fixestの参照値のフィクスチャ化、HC2/HC3の参照、DKのKP・有効Fの外部参照が無いことの扱い）→ #380（性能）→ FE-GMM（#381〜#388）。どれも着手前に、判断が分かれる点をユーザーに確認する（CLAUDE.md 14章）。実装は`/implement-python`・`/implement-rust`の流れ（実装→lint→テスト→reviewer）。

**#377（Pythonラッパー）の着手メモ**
- 置き場: `python_package/econometricsmodels/panel/fe_iv.py`に`FEIV`・`FEIVResults`（`FEIVOptions`は`_lib`から再エクスポート）。`panel/fe.py`の`FE`/`FEResults`（`FEResults`は`:187`、`params_table`は`:411`）・`iv/iv.py`の`IV`/`IVResults`（`_as_dict`は`:729`）・`diagnostics.py`（`CraggDonaldResult`・`KleibergenPaapResult`・`EffectiveFResult`。FEIVも同じdataclassに包む）を踏襲。`__init__.py`の import と`__all__`に足す。`python_package/econometricsmodels/panel/CLAUDE.md`に追記。
- `_lib`の面（9.24）: `fit_fe_iv(data, y, x_exog, x_endog, instruments, effects, options)`、`FEIVResult`のフィールドは`params`〜`dk_bandwidth_used`・`wald_*`（常にF）・`r_squared_within/between/overall`・`weak_instrument_f_statistics`（`HashMap`なので**`x_endog`の順の`dict`に直す**）・`weak_instrument_f_df_num/denom`・`overid_*`・`wu_hausman_*`。メソッド: `first_stage()`（`dict[str, FEResult]`を`x_endog`の順に）・`fixed_effects()`・`cragg_donald()`・`kleibergen_paap()`・`effective_f(alpha)`（出力型はIVと同じ`*Output`。`FEIVResults`のラッパーで`DiagnosticResult`系のdataclassに）。`params_table()`はPython側で足す（`FEResults`と同じ）。
- 引数: `FEIV(data, y, x_exog, x_endog, instruments, effects, *, options)`（`docs/guide/api-conventions.md`の基準。表に`| FEIV | y, x_exog, x_endog, instruments, effects | cluster, dk_time |`を1行足し、effectsの段落・61〜69行目の記述にも入れる）。重みの位置は空けておく（6.1の8）。
- **横断テストへの登録（漏らさない）**: `tests/cross_method/test_gil_release.py`（`FITS`に`"FEIV"`、`ROWS`に`"FEIV"`と`"FEIV.cragg_donald"`等（行数は`PERIODS=10`の倍数）、`CASES`に`_post_case`。メソッドは自動検出されないので明示）、`test_call_signatures.py`（`test_every_estimator_is_checked`の固定の並びに`"FEIV"`）、`test_keyword_only_arguments.py`（`_ESTIMATORS`・`_RESULT_METHODS`に`"FEIVResults": ("cragg_donald","kleibergen_paap","effective_f")`・`FEIVOptions`の位置引数拒否）、`test_results_docstrings.py`（`FEIVResults`のdocstringに「Fields by purpose」で公開メンバーを漏れなく）、`test_validation_order.py`・`test_effects_argument.py`（`MODELS`）・`test_options_repr.py`・`test_option_types.py`・`test_argument_types.py`・`test_option_value_echo.py`・`test_numeric_column_values.py`・`test_data_input_types.py`・`test_key_columns.py`（それぞれ手法の表に足すか確認）、`tests/support/_errors.py`の`MIGRATED_CODES`。`docs/guide/pickling.md:7`・`docs/guide/verification.md`も確認。
- 検査の順序（`test_validation_order.py`）: `INSUFFICIENT_INSTRUMENTS`はデータを見ない件数の検査だが、エンジンの`fit`の中（列の抽出の後）で報告される。IV本体との整合を、ここで固定するときに決める。

**未決・保留（ユーザー判断）**
- `first_stage`の持ち方: 今は第一段階ごとに固定効果を先に計算した`FEResult`（`Clone`、`Arc`共有）を持ち、`first_stage()`は複製を返す。reviewerの代案は`Py<FEResult>`を持って`clone_ref`で返す形（O(n)の複製が消え、`FEResult`を`Clone`にしなくてよい）。6.1で「固定効果を先に計算して`FeEstimator`を保持しない」と決めた点と衝突するので、変えるかはユーザー判断。
- IV本体のクラスターで`G ≤ d`のrk LMが自明な値になる性質を、未定義にするか（IV本体の挙動変更）。
- DKの第一段階で`T`が`l`以下のときの標準誤差の信頼性（9.19）、#636へのFE-2SLSの追加、HC2/HC3・DKのKP・有効Fの外部参照（#379）、FE-GMMのHansen J（3章7）。
- IV側の従来からの保留（ルートCLAUDE.md 2章への診断の例外の追記、READMEの権利の注意書き、フィクスチャのデータのライセンス、3章`shared/`の列挙、`mop-oracle-spec.md`の履歴）。
**注意書き**: コミットはパス指定・署名（ホストの端末で確認。失敗したら止まって報告、リトライは1回まで）。Pythonを触ったら`uv run maturin develop`（debug）で入れ直してから`pytest`（`uv run --no-sync pytest`）。`_lib`を直接見るテストは`tests/panel/test_fe_iv_binding.py`（許容誤差は`TOLERANCES["fe_iv_binding"]`）。エラーコードを足したら`UPDATE_ERROR_CATALOG_GOLDEN=1 cargo test -p engine error_catalog`→`maturin develop`→`python tools/render_error_catalog.py`→`tests/support/_errors.py`。rustfmtが整形すると文字列置換が当たらなくなるので、置換は件数を`assert`する。heredocでバッククォートを含む文章を書くときは`<<'EOF'`。

### 9.26 #377（FE-2SLSのPythonラッパー）の実装結果（2026-11-13）

**実装**: `python_package/econometricsmodels/panel/fe_iv.py`に`FEIV`・`FEIVResults`（`FEIVOptions`は`_lib`の再輸出）、`__init__`・`__all__`。`FEIVResults(raw, x_endog)`は`x_endog`を保持し、`weak_instrument_f_statistics`・`first_stage()`（`dict[str, FEResults]`）を`x_endog`の順に直す（`_lib`は`HashMap`）。診断（`cragg_donald()`・`kleibergen_paap()`・`effective_f(*, alpha)`）の`dataclass`への包みは`_results_utils.py`に移してIVと共有した（IVの挙動は変えていない）。`log_likelihood`・AIC/BICは出さない。
**登録**: 横断テスト（`test_call_signatures`・`test_gil_release`・`test_results_repr`・`test_keyword_only_arguments`・`test_validation_order`・`test_effects_argument`・`test_argument_types`・`test_data_input_types`・`test_numeric_column_values`・`test_key_columns`・`test_cov_params`・`test_stat_dist`・`test_pickle`・`test_option_types`・`test_options_repr`・`test_option_value_echo`）、`tests/panel/test_fe_iv_api.py`（ラッパーの公開面）、`docs/guide/api-conventions.md`の表、`docs/guide/pickling.md`。
**検証順序**: `test_validation_order`では、FEIVは`effects`の列数（4,0）→ `x_endog`の空（4,1）→ 役割の重複（4,2）→ 内の重複（4,3）→ `dk_time`の役割（4,5）。IVの空`x_endog`が`"const"`の検査と同順位（4,6）なのとは違う。`INSUFFICIENT_INSTRUMENTS`（エンジンの`fit`の中、列の抽出の後）の順序は固定していない。
**レビュー（python-reviewer）への対応**: `weak_instrument_f_statistics`の誤解を招く一文の削除、`wald_df_*`が`None`にならない旨、Sarganが`cov_type`に従わない旨、`wu_hausman_df_denom`の定義を追記。
**見送り**: 公開ドキュメント（usage・api・methods・nav）と`docs/guide/verification.md`は#378・#379。`first_stage`の持ち方・IV本体の`G ≤ d`のLMは9.25の未決のまま。診断のラッパー関数の引数`raw`は`Any`（Protocolで型を付ける案は見送り）。

### 9.27 #378（FE-2SLSの公開docs）の実装結果（2026-11-13）

**決定（ユーザー承認）**: methodsページは差分のみ（FE・IVのDefinitionsへリンクし、FEIV固有の式だけ書く）。README・`index.md`・`verification.md`も更新する（検証表は現状を正確に書く: fixest値はテストに埋め込み、第二の参照はダミー変数2SLS。凍結フィクスチャは#379で作り、そのとき書き直す）。fixestとの差は設計ノートで実測済みのものだけ（Sargan・wr2・連結成分が複数のときの自由度）。
**実装**: `docs/usage/fe_iv.md`（自己完結の例。8社×6年、操作変数2本）・`docs/api/fe_iv.md`・`docs/methods/fe_iv.md`、`mkdocs.yml`のnav（Panel modelsのFEとREの間）、`index.md`の表、README（手法一覧・検証表と脚注3）、`guide/inference-conventions.md`（表・Sargan・弱操作変数・Wu-Hausman）、`guide/validation.md`（順序・FEIVの検証・ComputationErrorの表）、`guide/verification.md`・`verification-details.md`（FEIVの節）。`api/fe_iv.md`は`CraggDonaldResult`等を再描画せずIVのページへリンクする（再描画するとautorefsの重複警告が出る）。
**検証**: `mkdocs build`は警告なし（アンカー検証あり）、`tests/cross_method`全件通過、`render_error_catalog.py --check`通過。
**校閲（`/review-docs`）と対応**: 5件のmust fixを直した: (1)DKで`T_d ≤ r`（`r = 除外操作変数の数 − 内生変数の数 + 1`）のとき、rk LMだけでなくrk Wald Fも未定義になり`kleibergen_paap()`・`effective_f()`は`ComputationError`になる（実測。文書は「Wald Fは影響を受けない」と誤っていた）、(2)Sarganは分子だけがダミー込み2SLSと同じで分母は`n − n_abs`（IVは`n`）、(3)fixestとの比較は1-way・2-wayだけ（3次元はダミー変数2SLSのみ）、(4)R²はwithinが1-way・2-way、between・overallが1-wayだけ定義式で検査（他パッケージとの比較ではない。「1を超えうる」は誤りで削除）、(5)`pickling.md`の列挙から`FEIVResults`を外した。should fixは、fixestの`wh`が内生変数複数で同時Wald F、根拠の無い「Stock-Yogoをそのまま使える」「too often」の緩和、ダミー変数2SLSがnumpyとRustの2実装であること、fixestの値は貼り付けで再生成スクリプトが無いこと、FE・REだけを並べていた箇所へのFEIVの追記、IV・FEのusageからの導線、docstring（`wald_df_*`・`weak_instrument_f_df_*`の型を`int`に、`kleibergen_paap()`・`effective_f()`・`wu_hausman_statistic`の記述）。lfeの記述は事実確認できないため削除し、ivDiagは「比較していない」に直した（Web検索はしていない）。DKの`T ≤ r`でrk Wald Fに明示ガードが無い件（ランク判定に任せている）は、ドキュメントを実測に合わせるだけにして、ガード追加は別Issue候補（承認後に起票）。
**未実施**: `docs/guide/performance*.md`と`benchmark_performance.yml`のmatrix（#380の計測後）。`docs/guide/accepted-data.md`はFEIVを表と本文のFE・REの併記に足した。

### 9.28 clear用のメモ（#378のコミット後。2026-11-13。#379以降は別セッション）

**状態**: `release/v0.10.0`。**未push**（pushはユーザーの指示があるまで行わない）: `645d4c68`（9.20のメモ）・`c6282534`（#374 Sargan）・`29387123`（#375 Wu-Hausman）・`8caa96aa`（#373 弱操作変数診断）・`a0b58e89`（#376 pybind）・`aff3a45c`（9.25のメモ）・`0ca0426a`（#377 Pythonラッパー）・`c10f335f`（#378 公開docs）とこのメモのコミット。pushは`58be9153`まで。#374〜#378のIssueはまだ**closeしていない**（push後、英語コメントつきでユーザー承認を得てclose）。PR #635は未マージ（内容確認後にマージ）。
**実装の結果**: 9.21（#374）・9.22（#375）・9.23（#373）・9.24（#376）・9.26（#377）・9.27（#378。校閲の対応を含む）。
**次の順序**: #379（参照ベンチマーク）→ #380（性能）→ FE-GMM（#381〜#388）。どれも着手前に、判断が分かれる点をユーザーに確認する（CLAUDE.md 14章）。

**#379（参照ベンチマーク）の着手メモ**
- 現状の根拠は凍結していない: `tests/panel/test_fe_iv_binding.py`にfixest 0.14.2の値（`set.seed(42)`・小数4桁に丸めたデータ、1-way・2-way × classical・hc1・cluster、Wu-Hausmanの補助回帰のt²）が貼り付けてあり、第二の参照はダミー変数を明示した2SLS（numpy。classical・hc1〜hc3・Sargan・部分F・固定効果）。Rustのエンジンテスト（`engine/src/panel/fe_iv/fit/tests/`の`lsdv.rs`・`sargan.rs`・`wu_hausman.rs`・`weak_instruments.rs`）にも別のRust版のダミー変数2SLSがある。許容誤差は`TOLERANCES["fe_iv_binding"]`（rtol 1e-8・atol 1e-10。実測は2.6e-12（fixest）・3.6e-14（ダミー版））。fixestの生成スクリプトは無い。devcontainerのRにfixest 0.14.2が入っている（lfe・ivDiagは無い）。
- 決める点（9.17の保留）: 参照の選び方（案: 内生変数1個の古典・HC1・1-wayクラスターはivDiagとfixestの`ivf1`、2個以上のCD・KPはダミー列を入れたlinearmodels（小さいN、FWL）、入れ子のクラスターと連結成分が複数のパネルはfixest・手計算）。HC2/HC3の参照（plm・`ivreg`＋`sandwich`）。DKのKP・有効Fには独立な参照が無い（fixestの`kpr`は参照にならない）ことの扱い。fixestの`wr2`はR²の定義が違うので照合に使えない。3次元の固定効果はfixestとの比較が無い（今はダミー版のみ）。
- 完了時に書き換える公開側（今は「凍結していない」と明記している）: `docs/guide/verification.md`（先頭の「Exception: FEIV」の項・参照の表・比較する統計量の表・許容誤差の表・テストデータ）、`docs/guide/verification-details.md`の「FEIV」の節（凍結と書き直す）、`README.md`の検証表と脚注3・「Reproducible and continuous」の例外の注記、`tests/support/_tolerances_panel.py`（`fe_iv_binding`のコメント）。フィクスチャ生成は`benchmark/panel/`に`generate_fe_iv_fixtures.py`（と`references/`のR）を足し、`benchmark/regenerate_all.py`に載せる（`.claude/skills/reference-benchmark/`、`/test-new`の完了条件に`verification.md`が含まれる）。`test_fe_iv_binding.py`の埋め込みの値はフィクスチャに置き換える。`/review-testing`で網羅性を確認する。
- `test_validation_order.py`: `INSUFFICIENT_INSTRUMENTS`（エンジンの`fit`の中、列の抽出の後）の順位を固定していない。IV本体との整合を決めてから足す。

**#380（性能）の着手メモ**: `performance/compare_fe_iv.py`を足し、`benchmark_performance.yml`のmatrixに加える。数値は`/release`でCIの計測値から`docs/guide/performance.md`・`performance-results.md`・`README.md`「Performance」に転記する。9.19の見送り（第一段階が持つメモリ・構造式用`FeInput`の複製、HC2/HC3のダミーのレバレッジの再計算、`first_stage`を`Py<FEResult>`で持つ案）を計測で判断する。`docs/performance/fe_iv.md`（日本語の開発ノート）に方法論を残す。

**未決・保留（ユーザー判断）**
- Issue起票の候補（承認後に英語で起票。関連は1つにまとめる）: DKで`T_d ≤ r`（`r = 除外操作変数の数 − 内生変数の数 + 1`）のときrk LMには明示ガード、rk Wald Fはランク判定任せ（丸め誤差で値を返す余地。公開側は実測の`ComputationError`に合わせてある）。IV本体のクラスターで`G ≤ d`のrk LMがちょうど`G`になる既存の性質。
- `first_stage`の持ち方（`Py<FEResult>`で複製を無くす案。6.1の「固定効果を先に計算する」決定と衝突）、DKの第一段階で`T`が`l`以下のときの標準誤差の信頼性（9.19）、#636へのFE-2SLSの追加、FE-GMMのHansen J（3章7）。
- IV側の従来からの保留（ルートCLAUDE.md 2章への診断の例外の追記、READMEの権利の注意書き、フィクスチャのデータのライセンス、3章`shared/`の列挙、`mop-oracle-spec.md`の履歴）。
**注意書き**: コミットはパス指定・署名（ホストの端末で確認。失敗したら止まって報告、リトライは1回まで）。Pythonを触ったら`uv run maturin develop`（debug）で入れ直してから`uv run --no-sync pytest`。公開ドキュメントを変えたら`/review-docs`（他パッケージ言及のWeb確認は事前に許可を取る）と`mkdocs build -f docs/mkdocs.yml`（警告なしを確認）。heredocでバッククォートを含む文章を書くときは`<<'EOF'`。シェルで標準入力を読むコマンド（`cat >> /dev/null`等）を混ぜない（止まる）。

### 9.29 #379（FE-2SLSの参照ベンチマーク）の実装結果（2026-11-13）

**決定（ユーザー承認）**: 9.28の案どおり（主=fixest、独立=plm・ダミー変数を明示した2SLS・ivDiag）。ivDiagはコンテナに`install_version`で導入（Dockerfileは記載済み＝再構築すれば同じ）。HC2/HC3・DKのKP・有効F・HC2/HC3のKPは手計算の自己整合を参照にして明記する。範囲は合成データ＋`test_fe_iv_binding.py`の埋め込み値の置換（実データは含めない）。コミットはコード → CSV・JSONの順に分ける。
**実装**: `benchmark/panel/fe_iv_datasets.py`（13シナリオ。`freeze.py`が`fe_iv_<シナリオ>.csv`と`fe_iv_true_beta.json`に凍結）、`fe_iv_cases.py`（`fixest_cases()`125・`dummy_cases()`84・`plm_cases()`59・`ivdiag_cases()`61。生成側とテスト側で共有）、`references/`の`run_fixest_iv_benchmark.R`・`dummy_2sls_ref.py`・`run_plm_fe_iv_benchmark.R`（`r.py`にアダプタ）、`fixtures/generate_fe_iv_*`（統計量7本。`*_cov*`は行列）、`regenerate_all.py`。`benchmark/iv/references/run_ivdiag.R`は固定効果を複数列（カンマ区切り）で受けられるようにした（単一列の挙動は同じ）。テストは`tests/panel/test_fe_iv_{reference,dummy_crosscheck,plm_crosscheck,weak_diagnostics_crosscheck}.py`、`tests/support/_fe_iv_helpers.py`、許容誤差は`_tolerances_panel.py`の4キー。`test_fe_iv_binding.py`からfixestの貼り付け値とそのテストを除いた（数値はフィクスチャの照合に移した）。公開側は`verification.md`・`verification-details.md`のFEIV・README・`reference-benchmark`スキルを更新。
**一致の実測**: fixest 125ケースで係数4e-15・標準誤差1e-12・p値6e-12・信頼区間3e-12・Wald F/Wu-Hausman 2e-12・部分F 2e-14・Sargan（`e'e/n`尺度を`(n−n_abs)/n`倍）3e-12・rk Wald F 4e-11。ダミー変数2SLS 84ケースで係数7e-15・標準誤差5e-14・Sargan 1e-12・CD/KP 3e-14・R² 2e-15・有効F（MOP）6e-15。plm 59ケースで標準誤差5e-15。ivDiag 61ケースでCD 5e-15・rk Wald F 1.1e-9・有効F 5e-10。
**分かったこと（参照の定義）**
- fixestの係数は内生変数が先頭（`fit_endog1`）。Rで`x_exog`・`x_endog`の順に直す。
- fixestの`cd`は本実装と一致しない（baselineで206.9対172.7。原因は追っていない）。`kpr`は9.28のとおり使えない。
- **内生変数が1個のとき、rk Wald Fは同じ`vcov`での第一段階の頑健Wald Fに等しい**。fixestの第一段階（`wald(第一段階, keep = 操作変数, vcov = vc)`）を参照にすると、DK・2-way・3-way・不均衡を含む111ケースで4e-11以内で一致した。9.17・9.28の「DKのKPに独立な参照が無い」は、rk Wald F（内生変数1個）については解消した（rk LM・有効F・複数内生変数は従来どおり無い）。
- plm: `vcovHC(white1, HC0)`×`n/(n−k−G)`がHC1、`vcovHC(arellano, HC0)`×`G/(G−1)·(n−1)/(n−k−1)`が個体クラスター、`vcovSCC(HC0, maxlag)`×`T/(T−1)·(n−1)/(n−k−G)`がDKに一致。plmのHC2/HC3はレバレッジに固定効果の分を含まず、本実装と定義が違う（baselineでplmのHC2はplmのHC0の1.007倍、本実装のHC2はplmのHC0の1.10倍）ため使わない。HC2/HC3はnumpyのダミー変数2SLSだけが参照（verification-detailsに記載）。
- linearmodels `IV2SLS`にダミーを入れると、classical・hc1・Sargan（`e'e/n`尺度）・Wu-Hausman（`wooldridge_regression.stat/q`）・部分F（classicalのみ。`diagnostics`は`cov_type`に従う頑健F）・CD・rk LM/rk Wald F が一致。hc1のrk LMはlinearmodelsがHC0基準なので比べない。クラスターはダミーを全て`K`に数えるため、**1-wayの個体FE × 時点クラスター**だけが一致（個体クラスターと2-wayは`K`の数え方が違う）。
- `PanelOLS._rsquared(params)`の返り値の順は`(overall, within, between)`。2-wayのwithinはlinearmodelsが個体のみの別定義なので、ダミー射影の手計算にした。3-wayは`PanelOLS`が扱えずR²の参照なし。
- 有効Fは`mop_oracle.reduced_form`にダミーを外生変数として渡す（`n_absorbed=0`）と、classical・hc1・hc2・hc3・時点クラスターで一致（内生変数1個）。HC2/HC3の有効Fの外部参照はこれ。
**要判断・未解決**
- **ivDiagの2-wayのクラスター頑健Fが本実装と一致しない**（baselineの個体クラスターで160.9対157.5、時点クラスターで348.5対290.9）。fixestの第一段階Wald Fは本実装と一致（157.518・290.873）するので、ズレはivDiag側にある。原因は追っていない。ivDiagの比較は1-wayのclassical・hc1・クラスターと、2-wayのclassical・hc1に限った（`ivdiag_cases()`）。
- 次の候補（未着手）: HC2/HC3のrk Wald Fを、fixestの第一段階（HC2/HC3は非IVのfeolsで提供される）で独立に確かめる。DKのrk LM・有効Fの独立な参照は引き続き無い。
- 実データのフィクスチャは含めていない（9.28のとおり。`jtrain`等は変数の定義の確認が要る）。
- `test_validation_order.py`の`INSUFFICIENT_INSTRUMENTS`の順位は9.28から変わらず未固定。

### 9.30 #379のテスト網羅性レビューへの対応（2026-11-14）

`testing-completeness-reviewer`の指摘のmust fix・should fixを、3段階（各段階でコードとフィクスチャを別コミット）で直した。nice to haveのN3・N4は#655・#656に起票（N3: HC2/HC3のrk Wald Fをfixestの第一段階で確かめる、N4: ivDiagの2-wayクラスターFの不一致の原因）。
**1段階目**（`4580002f`・`fb36db25`）: `tests/panel/test_fe_iv_validation.py`を新設（ValidationError: ロール・列の存在・定数の列・singleton・自由度（第一段階が境界を決める）・クラスター・DK・オプション、ComputationError: 共線・完全フィット・数値的な特異・収束しない射影・弱操作変数統計量の未定義。境界は成功側と対）。`test_fe_iv_binding.py`の検証系は移した。`cov_type`の大文字小文字のテストを追加。フィクスチャから比較しないキー（hc1のrk LM、fixestのSargan生値）を外し、`n_absorbed`を比較する。独立クロスチェックの`atol`を1e-10に締めた。公開ページの誤り（fixestとの一致の桁、ivDiagの範囲、実データなしの行、単一参照の統計量）を直した。
**2段階目**（`895c1e48`・`7cb42ed9`）: `long_panel`の時点ラベルをゼロ埋め（DKの時点列は文字列のバイト順）、DKのバンド幅3・`T-2`を足し（baseline・long_panelの1-way・2-way）、既定式が1になる規模では同一の`dk_bw1`を外して3-wayに足し、推定に使ったバンド幅も照合する。`exogenous_endog`のシードを43にした（Wu-Hausman p値 0.045 → 0.57）。
**3段階目**: シナリオを13から20に増やした（`df1_boundary`・`high_variance`・`moderate_multicollinearity`・`high_condition_number`・`scale_variance_mild`・`many_regressors`・`outlier_regressor`）。外生変数を複数列にするため、追加の列・外れ値は別の乱数生成器（`default_rng([seed, 7])`）から作り、既存シナリオの凍結CSVは変わらない。DKは時点数が傾き係数の数を超えるケースだけ作る。全ケースで参照と機械精度で一致した（fixest 174・ダミー変数2SLS 117・plm 93・ivDiag 92ケース。ダミー変数2SLSの有効Fだけ5e-11と最大）。
**分かったこと**: 自由度1の境界では、fixestの`wald()`が分母自由度を`max(df2, df1+1)`に切り上げるため、部分Fの分母自由度は第一段階の残差自由度を直接使う（統計量は同じ）。MOPオラクルは極小標本（`df1_boundary`）で臨界値計算の途中に`sqrt`の警告を出す（`f_eff`は有限）。
**見送り**: cross_methodの`test_perfect_fit`・`test_collinear_columns`・`test_column_error_context`・`test_wald_ill_conditioned`にFEIVを足す件（新設の`test_fe_iv_validation.py`で同じ`code`と`params`を確認済み）、N2のクラスター不均衡の組み合わせ（2-way・`unbalanced`・`long_panel`）、N6のテスト名・見出し規約の細部。

### 9.31 #380（FE-2SLSの性能計測）の実装結果（2026-11-14）

**決定（ユーザー承認）**: Issueのコメントどおり、`performance/compare_fe_iv.py`の1本・engine単独・`benchmark_performance.yml`のmatrixに`fe_iv`（GMMは未実装のため含めない。実装時に`extra_methods`へ足す）。追加の判断は3点: (1)`k`は傾き係数の総数（`x_exog = k − 内生変数の数`。FEの`k`と揃い、T=6固定でcov軸のdkが`T > k`を満たす。IV本体の`k = x_exog`とは定義が違う）、(2)内生変数・操作変数の数と固定効果の構造は`extra_methods`（method軸）の代表点で測る（`two_way`・`two_way_unbalanced`・`three_way`・`endog3`（内生3・操作変数5）・`inst6`（内生1・操作変数6）、n=1,000,000・k=5・classical）、(3)within変換は単独の計測を置かず、列数を変える軸と交互射影が必要な構造の比から間接的に見る（変換だけを取り出す公開APIが無く、タイミング用の経路は足さない）。
**実装**: `benchmark/panel/fe_iv_datasets.py`の`generate_fe_iv_perf_dataset`（規模を指定できる均衡パネル、numpyの一括生成。凍結しない。n=1,000,000が約0.4秒）、`performance/compare_fe_iv.py`、`render_docs_results.py`の表示順・名前、`benchmark_performance.yml`のmatrix、`docs/performance/fe_iv.md`（日本語の開発ノート。結果の表は置かない）、`docs/guide/performance.md`の手書き部分（比較対象が無いこと・`k`の定義・計測範囲）。`docs/guide/performance-results.md`と先頭のサマリーはリリース準備時にCIの計測値から再生成する（手で書かない）。
**確認**: devcontainerのreleaseビルドで全軸（n=10,000〜1,000,000、k=5・20、cov軸8点、method軸5点）を`--repeats 1`で通した。失敗した軸は無く、n=1,000,000でも完走する。数値は公開しない（CIの計測値が正）。デバッグビルドでも配線を確認し、環境はデバッグビルドへ戻した。
**未実施**: CIでの計測（`workflow_dispatch`かタグ push）、`performance-reviewer`によるレビュー、README「Performance」への転記（リリース時）。9.19の見送り（メモリの複製・HC2/HC3のレバレッジの再計算・`first_stage`の持ち方）の判断はCIの計測値を見てから。
