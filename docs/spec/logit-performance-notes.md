# Logit: パフォーマンス比較（statsmodels）

`Logit(...).fit()`（Rust engine + PyO3）とリファレンス実装 statsmodels（`smf.logit`）の実行時間・メモリ使用量比較の記録。CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けることが目的。

再実行可能なスクリプトは`performance/compare_logit.py`（手法非依存の計測ハーネス`performance/_perf_harness.py` ＋ Logit固有アダプタ、コミット対象）。生の計測結果JSONはコミットしない（`.gitignore`の`docs/spec/_*.json`参照）。

## 計測方法

`docs/spec/ols-performance-notes.md`「最重要の教訓」「計測方法」と共通（releaseビルド必須・`tracemalloc`不採用・サブプロセス隔離・スレッド数を1に固定・polars→pandas変換は計測区間外・ウォームアップ1回＋`repeats`回の中央値）。Logit固有の点は以下。

- **計測範囲の対称性（Issue #98）**: engine は係数・標準誤差と同じ呼び出しで、対数尤度・**切片のみモデルの対数尤度**・尤度比統計量・そのp値・McFadden擬似R²・AIC・BIC まで常に一括計算する。statsmodels はこれらを遅延評価（`cached_value`）にしており、特に `llnull`（切片のみモデルの対数尤度、`llr`/`prsquared` が依存）はアクセス時に**切片のみ Logit を別途フィットする**。`_fit_once_statsmodels` は `.fit()` 直後に `llf`/`llnull`/`llr`/`llr_pvalue`/`prsquared`/`aic`/`bic` へ明示アクセスし、engine と同じ処理範囲で計測する（この対称化により statsmodels 側の計測時間は遅延評価アクセスなしの約2〜3倍になる）。
- **cov_type**: classical と cluster の代表2点。Logit/Probit は OLS/WLS と違い HAC を持たない。classical/hc0/cluster を n=100,000, k=5 で軽く実測したところ cluster が最重だった。cluster の疑似グループ数は 50 固定。
- **`opg` は計測対象外**: statsmodels の discrete model（`Logit.fit`）は `opg` を `cov_type` 引数としてネイティブに受け付けず、`score_obs` からの numpy 手計算になる（`benchmark/nonlinear/references/statsmodels_ref.py`）。engine のネイティブ OPG との比較は「計測対象の処理範囲を対称に揃える」方針に反するため除外する。
- **method（オプティマイザ）**: engine・statsmodels とも Newton-Raphson（`method="newton"`）で n/k スイープを回す。加えて `bfgs`/`lbfgs` を **method 軸**として代表点1つ（cov_type=classical, k=5, n=1,000,000）で計測する（正確性検証〈`test_logit_fixtures.py`〉も newton 主軸、bfgs/lbfgs は代表のみ、という絞り方に合わせる）。
- **スイープ軸**: n軸（k=5固定、n=1,000〜1,000,000）、k軸（n=10,000固定、k=5・20）、method軸（下記）。

## 結果: n軸（k=5固定）

実行時間（秒、中央値）/ ピークRSS（MB）。devcontainer（12論理コア、シングルスレッド固定）、`repeats=3`、method=newton。

| n | engine | statsmodels |
|---|---|---|
| **classical** | | |
| 1,000 | 0.0005 / 205 | 0.0089 / 210 |
| 10,000 | 0.0050 / 207 | 0.0154 / 215 |
| 100,000 | 0.0511 / 231 | 0.1037 / 241 |
| 1,000,000 | 0.6500 / 419 | 1.3665 / 486 |
| **cluster** | | |
| 1,000 | 0.0009 / 206 | 0.0107 / 210 |
| 10,000 | 0.0124 / 210 | 0.0245 / 215 |
| 100,000 | 0.1068 / 254 | 0.1363 / 242 |
| 1,000,000 | 1.1713 / 510 | 1.6376 / 526 |

## 結果: k軸（n=10,000固定）

実行時間（秒、中央値）。method=newton。

| k | engine | statsmodels |
|---|---|---|
| **classical** | | |
| 5 | 0.0066 | 0.0192 |
| 20 | 0.0282 | 0.0415 |
| **cluster** | | |
| 5 | 0.0073 | 0.0205 |
| 20 | 0.0331 | 0.0431 |

## 結果: method軸（cov_type=classical, k=5, n=1,000,000固定）

実行時間（秒、中央値）。newton は「結果: n軸」classical の n=1,000,000 行（engine 0.6500 / statsmodels 1.3665）を参照。

| method | engine | statsmodels |
|---|---|---|
| bfgs | 11.2148 | 1.4660 |
| lbfgs | 23.9116 | 1.4373 |

## 考察

- **classical（newton）**: 全nでengineがstatsmodelsより高速（n=1,000,000で約2.1倍、0.65s vs 1.37s、n=100,000で約2.0倍）。engine は Newton-Raphson の各反復で `k×k` Hessian を構築するため OLS の直接法より計算は重いが、statsmodels 側の Python/patsy/最適化ループのオーバーヘッドの方が大きい。
- **cluster（newton）**: engine が全nで速いが、大 n では差が縮む（n=1,000,000で 1.17s vs 1.64s、約1.4倍）。engine の cluster n=1,000,000 のピークRSSが 510MB と statsmodels（526MB）に迫る。
- **Issue #98 の対称化が効く手法**: statsmodels の `llnull`（切片のみモデル）へのアクセスを計測範囲に含めると、statsmodels の実行時間が遅延評価アクセスなしの約2〜3倍に増える（切片のみ Logit の再フィットが走るため）。engine はこれを常に一括計算しているので、対称に揃えて初めて公平な比較になる（揃えないと engine に不利な非対称計測になっていた）。
- **method軸: engine の BFGS/L-BFGS が極端に遅い**: newton（engine 0.65s）に対し bfgs は **11.21s**（約17倍）、lbfgs は **23.91s**（約37倍）。同じ method の statsmodels（scipy）は bfgs 1.47s・lbfgs 1.44s で newton とほぼ同じ。engine の quasi-Newton 実装（ステップ制御・収束判定・逆Hessian近似の更新）に改善余地があり、`refactoring-candidates.md` 項目46 として記録した。**既定の newton は十分速いため実用上の実害は「newton 以外を選ぶと遅い」という選択上の注意に留まる**。
- **kスケーリング（newton）**: classical k=5→20 で engine 約4.3倍 / statsmodels 約2.2倍。engine の k 方向の伸びがやや急（Newton 各反復の Hessian 構築が `k²` 依存）。絶対値は小さく（k=20 でも 0.03s 台）実用上の問題はないが、傾向として記録する。
- **メモリ**: newton では engine が全点で軽いか同等（n=1,000,000でengine 419〜510MB、statsmodels 486〜526MB）。

## 既知の限界

`ols-performance-notes.md`「既知の限界」と共通。特に **engineのマルチスレッド線形代数が多コア機・負荷下で不安定になる問題**（`docs/planning/specs/refactoring-candidates.md`項目44）のため、本計測はengine・statsmodelsとも1スレッドに固定しており、数値は「シングルスレッドでの計算コア効率」である。計測は開発コンテナ上の1回のスイープ（`repeats=3`の中央値）で、環境ノイズを排除しきれていない。

## 再現方法

```bash
uv run maturin develop --release
uv run python -m performance.compare_logit --repeats 3 \
    --output docs/spec/_logit_performance_results.json
uv run python -m performance.render_performance_summary \
    docs/spec/_logit_performance_results.json
```

## 今後の検討事項

- **engineのBFGS/L-BFGSが遅い**（`refactoring-candidates.md` 項目46）: newton・statsmodels の同 method 比で桁違いに遅い。ステップ制御・収束判定・近似更新の実装を調査する。
- **engineのマルチスレッド線形代数の不安定性**（`refactoring-candidates.md`項目44）: OLSと共通。
- **releaseビルドでの再計測が前提**: 改善見込みの見積もりは、debugビルドの数値（誤り）ではなく本ドキュメントのreleaseビルド数値を基準にすること。
