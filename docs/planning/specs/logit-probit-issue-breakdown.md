# Logit / Probit Issueタスク分解

`docs/planning/specs/nonlinear-api-design.md`・`nonlinear-implementation-notes.md`で確定した設計を、OLS実装時の粒度（`ols-implementation-notes.md`のIssue #2〜#22相当。engine型定義→ソルバー→標準誤差種別ごと→適合度統計量→engine_pybind境界→python_packageラッパー→テストという流れ）に倣ってIssue単位に分解する。

数式（尤度・スコア・Hessianの導出）の細部はこの時点では確定させず、各Issue本文の中で決める。

## 全体方針

- **discrete_choice系統の共通基盤（A）を先に作り、Logit（B）・Probit（C）で使い回す**。OLSは1手法しかなかったため系統共通化の判断が不要だったが、Logit/Probitは`MleError`・ソルバー実行・`cov_type`共通行列演算を共有する設計（`nonlinear-implementation-notes.md`確定済み）のため、先に切り出す
- **Logit→Probitの順で着手**（`nonlinear-api-design.md`1章の優先順位、かつLogitの尤度・勾配・Hessianの方が解析的に単純なため、共通基盤を検証しながら実装する1本目として適している）
- Probit側（C）はLogitで確立したパターンをなぞるため、B・Cはほぼ1:1で対応する。Probit単体でのIssue数は16件（目安15〜20の範囲内）

---

## A. 共通基盤（discrete_choice系統、engine側）

Issue化済み（2026-07-20）:

- [x] **A1. discrete_choice系統の共通エラー型（`MleError`）をthiserrorで定義** → [#51](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/51)
  `nonlinear-implementation-notes.md`のバリアント一覧（`NonConvergence`/`InsufficientObservations`/`InvalidConfidenceLevel`/`InvalidMaxIter`/`MissingClusterColumn`/`InsufficientClusters`/`SingularHessian`/`ComputationFailed`/Tobit専用の`InvalidCensoringBounds`）を実装。Tobit専用分はバリアントだけ先に用意し、実際に使うのはTobit実装時
- [x] **A2. common.rs: ソルバー実行の共通化** → [#52](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/52)
  `method`（`"newton"`/`"bfgs"`/`"lbfgs"`）文字列→argminソルバーへのディスパッチ。設計行列の内部標準化・逆変換、`max_iter`・`tol`（勾配ノルム基準）による収束判定、収束フラグ・反復回数の返却
  - 依存: A1（#51）
- [x] **A3. common.rs: `cov_type`共通行列演算** → [#53](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/53)
  観測情報行列（`-H⁻¹`）/OPG（`(Σsᵢsᵢ')⁻¹`）/サンドイッチ（`H⁻¹(Σsᵢsᵢ')H⁻¹`、HC1補正込み）/クラスター（`H⁻¹(ΣS_gS_g')H⁻¹`、小標本補正込み）の行列演算を、`H`・`scores`を受け取る関数群として実装。モデル固有の尤度計算には依存しない
  - 依存: A1（#51）

## B. Logit

Issue化済み（2026-07-20）:

- [x] **B1. Logitのデータ構造定義** → [#54](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/54)
  `LogitInput::from_columns`（`y`/`x_columns`/`param_names`/`dep_var_name`の保持、次元検証）。OLSの`OlsInput`に相当
- [x] **B2. Logitの尤度・スコア・Hessian実装** → [#55](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/55)
  `LogitProblem`構造体に対するargminの`CostFunction`（負の対数尤度）/`Gradient`/`Hessian`トレイト実装、および観測ごとのスコア行列を返す`scores()`メソッド
  - 依存: B1（#54）
- [x] **B3. Logit: Newton-Raphsonでの最適化・収束判定** → [#56](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/56)
  `LogitEstimator::fit`の骨格。A2のソルバー実行共通化を使い、`method="newton"`（既定）で最適化。`raise_on_non_convergence`の分岐実装
  - 依存: A2（#52）, B2（#55）
- [x] **B4. Logit: BFGS/L-BFGSソルバー対応** → [#57](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/57)
  `method="bfgs"`/`"lbfgs"`の分岐。収束点でのHessian評価（SE用）は`method`に関わらず常に行う
  - 依存: B3（#56）
- [x] **B5. Logit: 観測情報行列（classical/nonrobust）でのSE・z値・p値・信頼区間** → [#58](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/58)
  A3の観測情報行列計算を使い、`std_errors`/`z_stats`/`p_values`/`conf_lower`/`conf_upper`を算出。標準正規分布（statrs `Normal`）を使用
  - 依存: A3（#53）, B3（#56）
- [x] **B6. Logit: OPG（BHHH）・サンドイッチ型（HC0/HC1）でのSE** → [#59](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/59)
  `cov_type="opg"/"hc0"/"hc1"`の分岐。OLSがHC0〜HC3を1 Issueにまとめたのに倣い、3種類まとめて実装
  - 依存: A3（#53）, B5（#58）
- [x] **B7. Logit: クラスターロバストSE** → [#60](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/60)
  `cov_type="cluster"`。クラスターキー未指定時の`MissingClusterColumn`、クラスター数<2の`InsufficientClusters`を含む
  - 依存: A3（#53）, B5（#58）
- [x] **B8. Logit: 適合度統計量** → [#61](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/61)
  `log_likelihood`/`log_likelihood_null`（切片のみモデルの再フィット）/`lr_statistic`/`lr_p_value`（カイ二乗分布）/`pseudo_r_squared`（McFadden）/`aic`/`bic`/`n_obs`/`df_model`/`df_resid`
  - 依存: B3（#56）
- [x] **B9. Logit: 限界効果（`marginal_effects`）** → [#62](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/62)
  `at="overall"`（既定、AME）/`"mean"`/`"median"`。デルタ法標準誤差（`fit()`時の`cov_params`を再利用）
  - 依存: B5（#58）
- [x] **B10. Logit: `predict()` / `pred_table()`** → [#63](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/63)
  予測確率、閾値依存の的中表。コアのReturnには含めない別メソッド
  - 依存: B3（#56）
- [x] **B11. engine単体テストのカバレッジ確認・不足分を追加** → [#64](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/64)
  `cargo-llvm-cov`で計測し、OLS同様「理論上到達不能な防御的エラーパス」の扱い方針に従う
  - 依存: B4〜B10（#57, #58, #59, #60, #61, #62, #63）
- [x] **B12. engine_pybind: データ抽出・`LogitOptions`/`LogitResult` pyclass定義** → [#65](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/65)
  OLSの`column_extraction.rs`（既存、再利用可）を使ったパラメータの受け口。この時点では実計算に接続せず、OLSの`fit_ols`初期実装と同様に一旦打ち切ってもよい
  - 依存: B1（#54、データ構造が固まっていれば着手可、A1〜A3・B2以降と並行して進められる）
- [x] **B13. engine_pybind: engine呼び出し・エラー変換** → [#66](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/66)
  B12の受け口と`LogitEstimator::fit`を接続。`MleError` → `ValidationError`/`ComputationError`変換
  - 依存: B11（#64）, B12（#65）
- [x] **B14. python_package: Logitラッパー（`Logit`/`LogitResults`）実装** → [#67](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/67)
  `discrete_choice/logit.py`。`coef_table()`・`marginal_effects()`・`predict()`・`pred_table()`のPython側ラッパー
  - 依存: B13（#66）
- [x] **B15. tests/api_tests: statsmodels/R glmとの数値照合ベンチマーク作成** → [#68](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/68)
  `/test-new`スキル使用。収束判定`tol`の妥当性検証もここで行う（`nonlinear-implementation-notes.md`の暫定事項）
  - 依存: B14（#67）
- [x] **B16. ドキュメント（mkdocs）** → [#69](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/69)
  - 依存: B15（#68）

## C. Probit（Bと同型、共通基盤流用）

Issue化済み（2026-07-20）:

- [x] **C1. Probitのデータ構造定義**（`ProbitInput::from_columns`） → [#70](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/70) — 依存: なし
- [x] **C2. Probitの尤度・スコア・Hessian実装** → [#71](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/71) — 依存: C1（#70）
- [x] **C3. Probit: Newton-Raphsonでの最適化・収束判定** → [#72](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/72) — 依存: A2（#52）, C2（#71）
- [x] **C4. Probit: BFGS/L-BFGSソルバー対応** → [#73](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/73) — 依存: C3（#72）
- [x] **C5. Probit: 観測情報行列でのSE・z値・p値・信頼区間** → [#74](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/74) — 依存: A3（#53）, C3（#72）
- [x] **C6. Probit: OPG・サンドイッチ型でのSE** → [#75](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/75) — 依存: A3（#53）, C5（#74）
- [x] **C7. Probit: クラスターロバストSE** → [#76](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/76) — 依存: A3（#53）, C5（#74）
- [x] **C8. Probit: 適合度統計量** → [#77](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/77) — 依存: C3（#72）
- [x] **C9. Probit: 限界効果** → [#78](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/78) — 依存: C5（#74）
- [x] **C10. Probit: `predict()` / `pred_table()`** → [#79](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/79) — 依存: C3（#72）
- [x] **C11. engine単体テストのカバレッジ確認** → [#80](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/80) — 依存: C4〜C10（#73, #74, #75, #76, #77, #78, #79）
- [x] **C12. engine_pybind: データ抽出・`ProbitOptions`/`ProbitResult` pyclass定義** → [#81](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/81) — 依存: C1（#70）
- [x] **C13. engine_pybind: engine呼び出し・エラー変換** → [#82](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/82) — 依存: C11（#80）, C12（#81）
- [x] **C14. python_package: Probitラッパー実装** → [#83](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/83) — 依存: C13（#82）
- [x] **C15. tests/api_tests: 数値照合ベンチマーク作成** → [#84](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/84) — 依存: C14（#83）
- [x] **C16. ドキュメント（mkdocs）** → [#85](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/85) — 依存: C15（#84）

---

## 推奨する着手順序

1. **A1 → A2 → A3**（共通基盤）
2. **B1 → B2 →（B12を並行着手可）→ B3 → B4**（Logit engine コア）
3. **B5 → B6 → B7**（Logit 標準誤差）
4. **B8 → B9 → B10**（Logit 適合度統計量・限界効果・predict）
5. **B11 → B13 → B14 → B15 → B16**（Logit テスト・境界・ラッパー・ドキュメント）
6. **C1〜C16**（Probit。Bと同じ順序で進める。共通基盤Aは流用のため再着手不要）

Logit（B、共通基盤Aを含む）で19 Issue、Probit（C）で16 Issue、合計35 Issue。

## 未確定・Issue化時に決める事項

- Tobit専用の`MleError`バリアントの正確な検証条件（A1では枠だけ用意）
- 各モデルの尤度・スコア・Hessianの具体的な数式（B2/C2の中で導出）
- python_package側でLogit/Probitの`Results`クラスに共通基底を設けるか、独立実装のまま重複させるか（B14/C14着手時に判断）
