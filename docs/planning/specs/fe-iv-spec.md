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
panel/fe_iv.rs    FeIvInput { y, x_exog, x_endog, instruments (列スライス Vec<Vec<f64>>),
                              x_exog_names, x_endog_names, instrument_names, dep_var_name, index: PanelIndex }
```

- `PanelIndex`は`FeInput`・`ReInput`が持つ索引部分（entity・time・効果名）の切り出し。`FeIvInput`が最初に
  採用し、`FeInput`/`ReInput`は別Issue（FEIV実装後）で寄せる。`validate_panel_input`の次元検証は
  `PanelIndex`構築と各列の長さ検証に分ける。
- 定数項は持たない（固定効果が吸収する）。`x_exog`は空でもよい。`x_endog`・`instruments`は非空
  （空は`ROLE_EMPTY`、`build_*`側で検証。IVと同じ）。
- 列は`faer::Mat`ではなく列スライス（`FeInput`と同じ。`WithinTransform::apply`が列単位のため）。

### 1.2 `FeIvEstimator::fit(input, effects, cov_type, confidence_level)`

`effects: PanelEffects`、`cov_type: PanelCovType`（`FeEffects`/`FeCovType`と同じ型）。

1. `ensure_serial()`。2-wayなら`time`の有無を確認（`TwoWayRequiresTime`）。
2. `InsufficientInstruments`（`len(instruments) < len(x_endog)`、カタログの既存code）。
3. singleton検出（`ensure_no_singleton_groups`）、2-wayはバランスパネル検証（`validate_balanced_panel`）。
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
fixestがIVでHC2/HC3を提供するか。**#379で参照実装を実測して、残すか落とすかを決める。**

### 1.4 診断（後続Issueで実装、ここでは取り込み方だけ決める）

- 弱操作変数（#373）: 第一段階の部分F（`SSR_u`・`SSR_r`は変換後の列、分母自由度は`n - l - neffects`）、CD/KP/MOP。
- Sargan（#374）: 構造残差を`Z̃ = [x̃_exog, z̃]`に回帰した`n·R²`相当。`σ̂² = e'e/n`。自由度`len(instruments) - len(x_endog)`。
- Wu-Hausman（#375）: `ỹ ~ [x̃_exog, x̃_endog, v̂]`の補助FE回帰（`v̂`は第一段階の残差）、`v̂`列のWald検定。

### 1.5 エラー

`PanelError`にバリアントを足し、既存のカタログcodeを再利用する（`INSUFFICIENT_INSTRUMENTS`等。同じcodeは
同じhint）。第一段階の失敗は原因のcodeに`stage=FIRST_STAGE`を付ける（`InStage`）。新しいcodeが必要になったら
`rust-style.md`のエラー追加手順に従う。

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

## 4. 子Issueへの反映

- #371: 「`iv::two_sls`コアの再利用」を本ノート1.2（`FeEstimator`で第一段階・`panel_cov_params`で標準誤差）に直す。
- #372: 1.2の6・10に含まれるため、独立した実装は不要（テスト・文書化のみ）。
- #373〜#375: 1.4の取り込み方に沿う。
- #377: クラス名を`FEIV`系に直す。
- #381: 入力・within変換・第一段階は本ノートを共有し、点推定をGMMに置き換える。
