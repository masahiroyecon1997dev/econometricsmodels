# FE / RE / IV Issueタスク分解

`docs/planning/specs/panel-api-design.md`（FE/RE共通・FE固有・RE固有）・
`iv-api-design.md`（IV共通差分・IV固有）で確定した設計を、Logit/Probit実装時と同じ粒度
（engine共通基盤→engine型定義→推定コア→標準誤差→適合度統計量→診断メソッド→
engine_pybind境界→python_packageラッパー→ベンチマーク→ドキュメント、という流れ）に
倣ってIssue単位に分解する。

数式・アルゴリズムの細部（Driscoll-Kraayのバンド幅パラメータ設計等）はこの時点では確定させず、
各Issue本文の中で決める。

## 全体方針

- **CLAUDE.md 4章の直近の実装順序（`OLS → WLS → Logit → Probit → Tobit → IV → FE → RE → GLS`）
  に従い、IV → FE → RE の順で着手する**（Tobit完了後）。
- **IVは`engine/src/iv/`という独立系統のため、FE/REとは別の共通基盤（B章）で着手できる**。
  crate横断の共通基盤（A章、t/z検定後処理・`cov_type`パース共通化・複数列ロールバリデーション）
  はIVが最初に必要とするため、IVフェーズの入り口で先に切り出す。
- **FE/REはpanel系統として、独自の共通基盤（C章: `PanelError`・`quasi_demean`・
  `hausman_statistic`）を挟んでからFE→REの順に着手する**。REはFEの内部実装
  （`OlsEstimator`委譲・within回帰残差）に依存するため、FE完了後でないと着手できない
  （`panel-api-design.md`7.4節）。
- 各モデルのIssue数はLogit/Probitと同程度（15〜20件）を目安にする。

---

## A. crate横断共通基盤（IV着手前）

Issue化済み（2026-08-02）:

- [x] **A1. t/z検定後処理のジェネリック関数化** → [#152](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/152)
  `statrs::distribution::ContinuousCDF`をジェネリックに取る関数を`engine`直下（系統をまたぐ
  位置）に切り出す。OLS（t分布）・Logit（z分布）の既存重複を解消しつつ、IVの2SLS（t分布）・
  GMM（z分布）でも使う。詳細: `panel-api-design.md`4.2節
  - 依存: なし
- [x] **A2. engine_pybind: `cov_type`文字列パース＋列抽出ブロックの共通化** → [#153](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/153)
  `ols.rs`/`wls.rs`でほぼ完全一致していた重複を共通関数化する。詳細: `panel-api-design.md`4.2節
  - 依存: なし
- [x] **A3. engine_pybind: `validate_no_duplicate_roles`の複数列ロール対応拡張** → [#154](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/154)
  現状は単一列ロール（`y`/`weight`）のみ対応。IVの`x_exog`/`x_endog`/`instruments`間の重複
  検証に必要。詳細: `panel-api-design.md`4.2節、`iv-api-design.md`4章
  - 依存: なし

## B. IV（2SLS/GMM）

Issue化済み（2026-08-02）:

- [x] **B1. `IvError`型定義** → [#155](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/155)
  `LeastSquaresError`/`MleError`/`PanelError`の前例に倣い、2SLS/GMM共有のエラー型を
  `engine/src/iv/common.rs`に定義。詳細: `iv-api-design.md`4章
  - 依存: なし
- [x] **B2. IVデータ構造定義** → [#156](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/156)
  `IvInput::from_columns`相当。`x_exog`/`x_endog`/`instruments`の保持、`instruments`は
  除外操作変数のみという前提の次元検証。詳細: `iv-api-design.md`1章
  - 依存: B1（#155）
- [x] **B3. 2SLS推定（第一段階＋第二段階）の実装** → [#157](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/157)
  - 依存: B2（#156）
- [x] **B4. GMM共通推定コアの実装（`weight_type`対応）** → [#160](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/160)
  `weight_type`（unadjusted/robust/cluster/kernel）ごとの重み行列で点推定。2SLSをこのコアの
  特殊ケース（`weight_type="unadjusted"`, 1-step）として吸収できるかはB3実装後に判断する
  （無理なら独立実装のままでよい、「無理をしない」という既定方針に従う）。詳細: `iv-api-design.md`6.2節
  - 依存: B2（#156）、B3（#157、共通化の可否判断のため）
- [x] **B5. `gmm_iterations`（1-step/2-step efficient）の実装** → [#165](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/165)
  デフォルト2（efficient two-step）、1で1-step GMM。詳細: `iv-api-design.md`6.2節
  - 依存: B4（#160）
- [x] **B6. cov_type対応（classical/hc0-3/cluster/hac）** → [#166](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/166)
  2SLSのサンドイッチ型分散、GMMは`weight_type`と独立に`cov_type`を選択可能にする。
  詳細: `iv-api-design.md`3.1節・6.7節
  - 依存: A1（#152）, B3（#157）, B4（#160）
- [x] **B7. 第一段階結果（`first_stage`相当データ）の実装** → [#158](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/158)
  内生変数ごとの`x_endog[i] ~ x_exog + instruments`回帰。詳細: `iv-api-design.md`2.2節
  - 依存: B2（#156）
- [x] **B8. 弱操作変数診断（部分F統計量）の実装** → [#163](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/163)
  x_exogを直交化した後の操作変数係数のみを検定する専用計算（`first_stage`のF統計量をそのまま
  使わない）。詳細: `iv-api-design.md`6.4節
  - 依存: B7（#158）
- [x] **B9. 過剰識別検定（Sargan/Hansen J）の実装** → [#167](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/167)
  df = `len(instruments) - len(x_endog)`、丁度識別時は`None`。詳細: `iv-api-design.md`6.5節
  - 依存: B3（#157）, B4（#160）
- [x] **B10. 内生性検定（Wu-Hausman回帰ベース）の実装** → [#164](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/164)
  第一段階残差を構造式に追加回帰する方式（`wooldridge_regression`相当）。`first_stage`の
  残差を再利用。詳細: `iv-api-design.md`6.6節
  - 依存: B7（#158）
- [x] **B11. engine単体テストのカバレッジ確認・不足分を追加** → [#168](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/168)
  - 依存: B5〜B10（#165, #166, #158, #163, #167, #164）
- [x] **B12. engine_pybind: データ抽出・`IvOptions`/`IvResult` pyclass定義** → [#159](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/159)
  - 依存: A3（#154）, B2（#156）
- [x] **B13. engine_pybind: engine呼び出し・エラー変換実装** → [#169](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/169)
  - 依存: B11（#168）, B12（#159）
- [x] **B14. engine_pybind: `first_stage()`メソッド実装** → [#170](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/170)
  戻り値 `dict[str, OlsResults]`（内生変数名キー）。詳細: `iv-api-design.md`2.2節
  - 依存: B13（#169）
- [x] **B15. python_package: IV/IvResultsラッパー実装** → [#161](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/161)
  2SLS/GMM両対応。
  - 依存: B14（#170）
- [x] **B16. tests/api_tests: linearmodels/ivregとの数値照合ベンチマーク作成** → [#171](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/171)
  GMMはPython（linearmodels）単独検証（Rクロスチェック省略、`iv-api-design.md`5.3節）。
  - 依存: B15（#161）
  - **完了済み**: 2SLSのlinearmodelsクロスチェック（`tests/api_tests/test_iv_fixtures.py`、
    コミット`8d84a18`）、GMMのlinearmodelsクロスチェック（`tests/api_tests/test_iv_gmm_fixtures.py`、
    コミット`01aff46`）、ValidationError系のAPI/構造テスト（`tests/api_tests/test_iv.py`、
    コミット`5e38f2f`）。副次的に発見・修正したバグ2件（G=2クラスター境界での
    `has_intercept`混同バグ、GMM `cov_type=Classical`のσ̂²非中心化バグ）は
    `engine/src/iv/CLAUDE.md`参照。
    2SLSの`ivreg`（R）クロスチェック（`tests/api_tests/test_iv_crosscheck.py`、
    `benchmark/iv/fixtures/generate_iv_crosscheck_fixtures.py`、
    `benchmark/iv/run_ivreg_benchmark.R`）も完了。devcontainerのR更新
    （Debian bookworm標準4.2.2→CRAN APTリポジトリ経由で4.6.1系、`.devcontainer/Dockerfile`
    修正済み）後にコンテナ再構築・`ivreg`導入確認（R 4.5.3、`ivreg`利用可能）を行い、
    シナリオ・cov_type構成はlinearmodelsクロスチェック（`generate_iv_fixtures.py`）と揃えた。
    `ivreg`の`summary(diagnostics=TRUE)`は弱操作変数F統計量・Sargan・Wu-Hausmanを
    一括で返すが常にclassical（iid）vcov固定（実測確認済み）のため、weak_instrument_f・
    sarganはcov_typeによらず一律クロスチェック、wu_hausmanはclassical cov_typeのみ
    クロスチェック（hc0/hc1/clusterは既存のlinearmodelsクロスチェックに委ねる、
    ユーザー確認済み）。テスト実装中に2件の許容誤差の細部を確認・確定
    （`test_iv_crosscheck.py`のモジュールdocコメント参照）: small_nシナリオ
    （n=40, hac_lag=3）のみHACの実測乖離が他シナリオ（0.3〜0.8%程度）より大きい
    （SE最大3.8%、F統計量最大8.1%）ため専用の緩めた許容誤差（10%）を設定、
    `f_p_value`がアンダーフローに近い極小値のときは相対誤差ではなく絶対誤差フロア
    （1e-5）で比較する方式に変更。
- [x] **B17. ドキュメント（mkdocs）** → [#162](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/162)
  - 依存: B16（#171）

## C. FE/RE共通基盤（panel系統、IV完了後・FE着手前）

Issue化済み（2026-08-02）:

- [x] **C1. `PanelError`型定義** → [#172](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/172)
  FE/REで共有するエラー型を`engine/src/panel/common.rs`に定義。詳細: `panel-api-design.md`4.4節
  - 依存: なし
- [x] **C2. θパラメータ化した準偏差変換関数`quasi_demean`の実装** → [#173](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/173)
  FE/RE共有。FEは全エンティティに`θ_i=1.0`を渡すことで特殊ケースとして扱う。
  詳細: `panel-api-design.md`7.4節
  - 依存: C1（#172）
- [x] **C3. ハウスマン統計量計算関数`hausman_statistic`の実装** → [#174](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/174)
  FE/RE共有（`(β_FE - β_RE)'[Var(β_FE) - Var(β_RE)]⁻¹(β_FE - β_RE)`）。RE実装時まで実際には
  呼ばれないが、共有関数のため先にまとめて用意する。詳細: `panel-api-design.md`7.3節
  - 依存: C1（#172）

## D. FE

Issue化済み（2026-08-02）:

- [x] **D1. FEデータ構造定義** → [#175](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/175)
  `FeInput::from_columns`相当。`entity`/`time`列の保持。詳細: `panel-api-design.md`1章
  - 依存: C1（#172）
- [x] **D2. within変換の実装** → [#176](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/176)
  1-way: `quasi_demean`のθ=1呼び出し（不均衡パネル対応）。2-way: 閉形式の二重デミーニング
  ＋バランスパネル検証（不均衡ならハードエラー、回避オプションなし）。
  詳細: `panel-api-design.md`6.1節・6.4節
  - 依存: C2（#173）, D1（#175）
- [x] **D3. singleton検出バリデーション** → [#179](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/179)
  entity（常時）・time（2-wayの場合のみ）を対称に検出。詳細: `panel-api-design.md`6.5節
  - 依存: D2（#176）
- [x] **D4. 分散ゼロ説明変数の検出バリデーション** → [#177](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/177)
  デミーニング後の設計行列の列分散チェック。1-way/2-way共通ロジック。
  詳細: `panel-api-design.md`6.7節
  - 依存: D2（#176）
- [x] **D5. `OlsEstimator`への委譲によるパラメータ推定** → [#178](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/178)
  まず委譲を試す緩い方針。うまくいかない場合はFE専用実装に切り替え。
  詳細: `panel-api-design.md`4.3節
  - 依存: D2（#176）, D3（#179）, D4（#177）
- [x] **D6. 自由度調整の実装とdf_resid/df_model反映** → [#180](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/180)
  1-way: `n - n_entities - k`。2-way: `n - n_entities - n_periods + 1 - k`。
  詳細: `panel-api-design.md`6.3節
  - 依存: D5（#178）
- [x] **D7. cov_type対応（classical/hc0-3/cluster）** → [#181](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/181)
  `cov_type`デフォルトを`"cluster"`（entity単位）にする（OLSの`"classical"`から意図的に逸脱）。
  詳細: `panel-api-design.md`3.1節・3.2節
  - 依存: A1（#152）, A2（#153）, D5（#178）
- [x] **D8. Driscoll-Kraay型パネルHACの実装** → [#182](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/182)
  OLSのHAC実装は流用しない別アルゴリズム。詳細: `panel-api-design.md`3.1節
  - 依存: D7（#181）
- [x] **D9. パネル固有R²（within/between/overall）の実装** → [#183](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/183)
  bareの`r_squared`は廃止。詳細: `panel-api-design.md`2.3節
  - 依存: D5（#178）
- [x] **D10. 固定効果自体（α_i）の復元ロジック実装** → [#184](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/184)
  `α_i = ȳ_i - x̄_i'β̂`。within変換時に保持したグループ平均を使う。
  詳細: `panel-api-design.md`6.6節
  - 依存: D5（#178）
- [x] **D11. engine単体テストのカバレッジ確認・不足分を追加** → [#185](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/185)
  - 依存: D6〜D10（#180, #181, #182, #183, #184）
- [x] **D12. engine_pybind: データ抽出・`FeOptions`/`FeResult` pyclass定義** → [#186](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/186)
  - 依存: D1（#175）
- [x] **D13. engine_pybind: engine呼び出し・エラー変換実装** → [#187](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/187)
  - 依存: D11（#185）, D12（#186）
- [x] **D14. engine_pybind: `fixed_effects()`メソッド実装** → [#188](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/188)
  1-way: `dict[str, float]`。2-way: `dict[str, dict[str, float]]`
  （`"entity"`/`"time"`キー）。詳細: `panel-api-design.md`6.6節
  - 依存: D13（#187）
- [x] **D15. python_package: FE/FeResultsラッパー実装** → [#189](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/189)
  - 依存: D14（#188）
- [x] **D16. tests/api_tests: fixest/linearmodelsとの数値照合ベンチマーク作成** → [#190](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/190)
  - 依存: D15（#189）
- [x] **D17. ドキュメント（mkdocs）** → [#191](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/191)
  - 依存: D16（#190）

## E. RE（Dに依存）

Issue化済み（2026-08-02）:

- [x] **E1. REデータ構造定義** → [#192](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/192)
  `ReInput::from_columns`相当。`entity`/`time`列の扱いはFE（D1）を踏襲。
  - 依存: D1（#175）
- [x] **E2. Swamy-Arora分散成分推定（σ_ε²・σ_u²）の実装** → [#193](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/193)
  σ_ε²はFEのwithin回帰残差分散を再利用（RE→FE→`OlsEstimator`の委譲チェーン）。
  詳細: `panel-api-design.md`7.1節
  - 依存: E1（#192）, D5（#178）
- [x] **E3. θ計算・準偏差変換の実装** → [#194](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/194)
  `θ_i = 1 - sqrt(σ_ε² / (T_i・σ_u² + σ_ε²))`。`quasi_demean`（C2）を再利用。
  詳細: `panel-api-design.md`7.2節
  - 依存: C2（#173）, E2（#193）
- [x] **E4. `OlsEstimator`への委譲によるパラメータ推定** → [#195](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/195)
  - 依存: E3（#194）
- [x] **E5. df_resid（`n - k`）・適合度統計量の実装** → [#196](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/196)
  FEの`n - n_entities - k`とは別式。詳細: `panel-api-design.md`7.5節
  - 依存: E4（#195）
- [x] **E6. cov_type対応** → [#197](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/197)
  詳細: `panel-api-design.md`3章（FEと同じ範囲）
  - 依存: A1（#152）, A2（#153）, E4（#195）
- [x] **E7. ハウスマン検定の実装** → [#198](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/198)
  内部でFE推定を呼び出し、`hausman_statistic`（C3）で比較。REの切片は比較対象から除外。
  内部FE推定が失敗した場合はハウスマン関連フィールドを`None`にしてRE本体の結果は返す。
  詳細: `panel-api-design.md`7.3節
  - 依存: C3（#174）, D5（#178）, E4（#195）
- [x] **E8. engine単体テストのカバレッジ確認・不足分を追加** → [#199](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/199)
  - 依存: E5〜E7（#196, #197, #198）
- [x] **E9. engine_pybind: データ抽出・`ReOptions`/`ReResult` pyclass定義** → [#200](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/200)
  - 依存: E1（#192）
- [x] **E10. engine_pybind: engine呼び出し・エラー変換実装** → [#201](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/201)
  - 依存: E8（#199）, E9（#200）
- [x] **E11. python_package: RE/ReResultsラッパー実装** → [#202](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/202)
  - 依存: E10（#201）
- [x] **E12. tests/api_tests: plm/linearmodelsとの数値照合ベンチマーク作成** → [#203](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/203)
  ハウスマン検定は`plm::phtest`のみを参照値とする（Rクロスチェックの例外規定、
  `panel-api-design.md`5.3節）。
  - 依存: E11（#202）
- [x] **E13. ドキュメント（mkdocs）** → [#204](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/204)
  - 依存: E12（#203）

---

## 推奨する着手順序

1. **A1 → A2 → A3**（crate横断共通基盤）
2. **B1 → B2 → B3 → B4 → B5**（IV engineコア。B4着手時に2SLS/GMM共通化の可否を判断）
3. **B6 → B7 → B8 → B9 → B10**（IV 標準誤差・診断3種）
4. **B11 → B13 → B14 → B15 → B16 → B17**（IV テスト・境界・ラッパー・ドキュメント）
5. **C1 → C2 → C3**（panel系統共通基盤）
6. **D1 → D2 → D3 → D4 → D5**（FE engineコア）
7. **D6 → D7 → D8 → D9 → D10**（FE 自由度・SE・R²・固定効果）
8. **D11 → D13 → D14 → D15 → D16 → D17**（FE テスト・境界・ラッパー・ドキュメント）
9. **E1〜E13**（RE。Dに依存するためFE完了後に着手）

IV（A含む）で20 Issue、FE（C含む）で20 Issue、RE（Eのみ）で13 Issue、合計53 Issue。
**全53件Issue化済み**（IV分: #152〜#171、FE/RE分: #172〜#204）。

## 未確定・Issue化時に決める事項

- 2SLSをGMM共通コア（B4）の特殊ケースとして実装できるか、それとも独立実装のままにするかの
  実際の判断（B4着手時、「無理をしない」という既定方針に従う）
- FEの`OlsEstimator`委譲（D5）が実装上うまくいくか、補正が管理可能な範囲に収まるかの判断
  （D5着手時、うまくいかない場合はFE専用実装に切り替え）
- Driscoll-Kraayのバンド幅パラメータの具体的な設計（D8着手時）
- 弱操作変数F統計量のfit()側フィールドの具体的な命名（B8着手時、`iv-api-design.md`6.4節で
  「実装時に確定」としていた）
- python_package側でFE/RE/IVの`Results`クラスに共通基底を設けるか、独立実装のまま重複させる
  か（D15/E11/B15着手時に判断。Logit/Probitでも同じ論点が未確定のまま残っている）
