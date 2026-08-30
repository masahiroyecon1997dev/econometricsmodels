# Probit: パフォーマンス比較（statsmodels）

`Probit(...).fit()`（Rust engine + PyO3）とリファレンス実装 statsmodels（`smf.probit`）の実行時間・メモリ使用量比較の記録。CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けることが目的。

再実行可能なスクリプトは`performance/compare_probit.py`（手法非依存の計測ハーネス`performance/_perf_harness.py` ＋ Probit固有アダプタ、コミット対象）。生の計測結果JSONはコミットしない（`.gitignore`の`docs/spec/_*.json`参照）。

## 計測方法

`docs/spec/ols-performance-notes.md`「最重要の教訓」「計測方法」・`logit-performance-notes.md`と共通（releaseビルド必須・`tracemalloc`不採用・サブプロセス隔離・スレッド数を1に固定・polars→pandas変換は計測区間外・ウォームアップ1回＋`repeats`回の中央値）。Probit固有の点は以下。

- **計測範囲の対称性（Issue #98）**: engine は係数・標準誤差と同じ呼び出しで、対数尤度・**切片のみモデルの対数尤度**・尤度比統計量・そのp値・McFadden擬似R²・AIC・BIC まで常に一括計算する。statsmodels の `ProbitResults` はこれらを遅延評価にしており、特に `llnull` はアクセス時に**切片のみ Probit を別途フィットする**。`_fit_once_statsmodels` は `.fit()` 直後に `llf`/`llnull`/`llr`/`llr_pvalue`/`prsquared`/`aic`/`bic` へ明示アクセスし、engine と同じ処理範囲で計測する。
- **cov_type**: classical と cluster の代表2点。classical/hc0/cluster を n=100,000, k=5 で軽く実測したところ、Logit と同じく cluster が最重だった。cluster の疑似グループ数は 50 固定。`opg` は statsmodels の discrete model がネイティブ非対応（`score_obs` からの手計算になる）で対称計測できないため対象外。
- **method（オプティマイザ）**: engine・statsmodels とも Newton-Raphson（`method="newton"`）で n/k スイープを回す。加えて `bfgs`/`lbfgs` を **method 軸**として代表点1つ（cov_type=classical, k=5, n=100,000）で計測する。
- **スイープ軸**: n軸（k=5固定、n=1,000〜**100,000**）、k軸（n=10,000固定、k=5・20）、method軸（下記）。**n軸は 1,000,000 を含まない**: `generate_binary_choice_dataset("baseline", link="probit")` は k=5 のとき n≥500,000 でΦ(Xβ)の飽和により engine の Probit Hessian が数値的に特異化し fit が失敗する（同じデータで statsmodels は収束する。engine 側の頑健性の課題として `docs/planning/specs/refactoring-candidates.md` 項目45 に記録。Logit（Λ、裾が Φ より厚い）では n=1,000,000, k=5 でも問題は起きない）。

## 結果: n軸（k=5固定）

実行時間（秒、中央値）/ ピークRSS（MB）。devcontainer（12論理コア、シングルスレッド固定）、`repeats=3`、method=newton。

| n | engine | statsmodels |
|---|---|---|
| **classical** | | |
| 1,000 | 0.0018 / 205 | 0.0124 / 210 |
| 10,000 | 0.0114 / 207 | 0.0239 / 215 |
| 100,000 | 0.1190 / 231 | 0.2068 / 241 |
| **cluster** | | |
| 1,000 | 0.0025 / 207 | 0.0111 / 210 |
| 10,000 | 0.0151 / 211 | 0.0283 / 215 |
| 100,000 | 0.1427 / 254 | 0.1671 / 242 |

## 結果: k軸（n=10,000固定）

実行時間（秒、中央値）。method=newton。

| k | engine | statsmodels |
|---|---|---|
| **classical** | | |
| 5 | 0.0098 | 0.0231 |
| 20 | 0.0292 | 0.0512 |
| **cluster** | | |
| 5 | 0.0131 | 0.0238 |
| 20 | 0.0393 | 0.0484 |

## 結果: method軸（cov_type=classical, k=5, n=100,000固定）

実行時間（秒、中央値）。newton は「結果: n軸」classical の n=100,000 行（engine 0.1190 / statsmodels 0.2068）を参照。

| method | engine | statsmodels |
|---|---|---|
| bfgs | 0.9096 | 0.1772 |
| lbfgs | 0.8233 | 0.1594 |

## 考察

- **classical（newton）**: 全n（1,000〜100,000）でengineがstatsmodelsより高速（n=100,000で約1.7倍、0.119s vs 0.207s）。
- **cluster（newton）**: engineが速いが差は小さい（n=100,000で 0.143s vs 0.167s、約1.2倍）。engine の cluster n=100,000 のピークRSSが 254MB と statsmodels（242MB）を上回る唯一の点で、クラスターロバスト共分散の中間行列の持ち方に差がある。
- **Probit は Logit より重い**: 同条件（classical, n=100,000, newton）で Probit engine 0.119s に対し Logit engine 0.051s。標準正規分布の CDF/PDF（Φ/φ）評価がロジスティック（初等関数）より高コスト。statsmodels 側も同傾向（Probit 0.207s vs Logit 0.104s）。
- **Issue #98 の対称化が効く**: Logit と同じく、statsmodels の `llnull`（切片のみモデル）へのアクセスを計測範囲に含めると切片のみ Probit の再フィットが走り、statsmodels の実行時間が増える。engine はこれを常に一括計算しているので、揃えて初めて公平な比較になる。
- **method軸: engine の BFGS/L-BFGS が遅い**: newton（engine 0.12s）に対し bfgs は **0.91s**（約7.6倍）、lbfgs は **0.82s**（約6.9倍）。同じ method の statsmodels（scipy）は bfgs 0.18s・lbfgs 0.16s で newton とほぼ同じ。Logit ほど極端ではないが同傾向で、engine の quasi-Newton 実装に改善余地がある（`refactoring-candidates.md` 項目46）。既定の newton は十分速いため実用上の実害は「newton 以外を選ぶと遅い」という選択上の注意に留まる。
- **kスケーリング（newton）**: classical k=5→20 で engine 約3.0倍 / statsmodels 約2.2倍。Logit ほどではないが engine の伸びがやや急。

## 既知の限界

- **n軸が 100,000 まで**: 上記「計測方法」のとおり、baseline DGP・k=5 では n≥500,000 で engine の Probit fit が Hessian 特異エラーになる（`refactoring-candidates.md` 項目45、engine 側の頑健性の課題として要調査）。このため大規模 n（1,000,000）での Probit の計測値は本ドキュメントに無い。他手法（OLS/WLS/Logit）は n=1,000,000 まで計測している。
- その他は `ols-performance-notes.md`「既知の限界」と共通。特に **engineのマルチスレッド線形代数が多コア機・負荷下で不安定になる問題**（`refactoring-candidates.md` 項目44）のため、本計測はengine・statsmodelsとも1スレッドに固定しており、数値は「シングルスレッドでの計算コア効率」である。

## 再現方法

```bash
uv run maturin develop --release
uv run python -m performance.compare_probit --repeats 3 \
    --output docs/spec/_probit_performance_results.json
uv run python -m performance.render_performance_summary \
    docs/spec/_probit_performance_results.json
```

## 今後の検討事項

- **engineのProbitのHessian特異化**（`refactoring-candidates.md` 項目45）: statsmodels が捌ける大標本条件で engine が失敗する。飽和に強い実装（重みのクリッピング・対数空間での Φ(1−Φ) 計算・Newton の damping/line search 等）の余地を調査する。解消後に n=1,000,000 を n軸に追加して再計測する。
- **engineのBFGS/L-BFGSが遅い**（`refactoring-candidates.md` 項目46）: Logit と共通。newton・statsmodels の同 method 比で遅い。
- **engineのマルチスレッド線形代数の不安定性**（`refactoring-candidates.md` 項目44）: OLSと共通。
- **releaseビルドでの再計測が前提**: 改善見込みの見積もりは、debugビルドの数値（誤り）ではなく本ドキュメントのreleaseビルド数値を基準にすること。
