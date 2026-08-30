# Logit: パフォーマンス比較（statsmodels）

`Logit(...).fit()`（Rust engine + PyO3）とリファレンス実装 statsmodels（`smf.logit`）の実行時間・メモリ使用量比較の記録。CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けることが目的。

再実行可能なスクリプトは`performance/compare_logit.py`（手法非依存の計測ハーネス`performance/_perf_harness.py` ＋ Logit固有アダプタ、コミット対象）。生の計測結果JSONはコミットしない（`.gitignore`の`docs/spec/_*.json`参照）。

## 計測方法

`docs/spec/ols-performance-notes.md`「最重要の教訓」「計測方法」と共通（releaseビルド必須・`tracemalloc`不採用・サブプロセス隔離・スレッド数を1に固定・polars→pandas変換は計測区間外・ウォームアップ1回＋`repeats`回の中央値）。Logit固有の点は以下。

- **計測範囲の対称性（Issue #98）**: engine は係数・標準誤差と同じ呼び出しで、対数尤度・**切片のみモデルの対数尤度**・尤度比統計量・そのp値・McFadden擬似R²・AIC・BIC まで常に一括計算する。statsmodels はこれらを遅延評価（`cached_value`）にしており、特に `llnull`（切片のみモデルの対数尤度、`llr`/`prsquared` が依存）はアクセス時に**切片のみ Logit を別途フィットする**。`_fit_once_statsmodels` は `.fit()` 直後に `llf`/`llnull`/`llr`/`llr_pvalue`/`prsquared`/`aic`/`bic` へ明示アクセスし、engine と同じ処理範囲で計測する（この対称化により statsmodels 側の計測時間は遅延評価アクセスなしの約2〜3倍になる）。
- **cov_type**: classical と cluster の代表2点。Logit/Probit は OLS/WLS と違い HAC を持たない。classical/hc0/cluster を n=100,000, k=5 で軽く実測したところ、engine（classical 0.060s / hc0 0.066s / cluster 0.076s）・statsmodels でも cluster が最重だった。cluster の疑似グループ数は 50 固定。
- **`opg` は計測対象外**: statsmodels の discrete model（`Logit.fit`）は `opg` を `cov_type` 引数としてネイティブに受け付けず、`score_obs` からの numpy 手計算になる（`benchmark/nonlinear/references/statsmodels_ref.py`）。engine のネイティブ OPG との比較は「計測対象の処理範囲を対称に揃える」方針に反するため除外する。
- **オプティマイザ**: engine・statsmodels とも Newton-Raphson（`method="newton"`）。
- **スイープ軸**: n軸（k=5固定、n=1,000〜1,000,000）、k軸（n=10,000固定、k=5・20）。

## 結果: n軸（k=5固定）

実行時間（秒、中央値）/ ピークRSS（MB）。devcontainer（12論理コア、シングルスレッド固定）、`repeats=3`。

| n | engine | statsmodels |
|---|---|---|
| **classical** | | |
| 1,000 | 0.0005 / 205 | 0.0182 / 210 |
| 10,000 | 0.0064 / 207 | 0.0217 / 215 |
| 100,000 | 0.0608 / 231 | 0.1384 / 241 |
| 1,000,000 | 0.8041 / 412 | 1.6803 / 486 |
| **cluster** | | |
| 1,000 | 0.0009 / 207 | 0.0136 / 211 |
| 10,000 | 0.0090 / 210 | 0.0239 / 215 |
| 100,000 | 0.0877 / 242 | 0.1419 / 242 |
| 1,000,000 | 1.0463 / 510 | 1.8889 / 526 |

## 結果: k軸（n=10,000固定）

実行時間（秒、中央値）。

| k | engine | statsmodels |
|---|---|---|
| **classical** | | |
| 5 | 0.0066 | 0.0237 |
| 20 | 0.0294 | 0.0635 |
| **cluster** | | |
| 5 | 0.0095 | 0.0333 |
| 20 | 0.0371 | 0.0548 |

## 考察

- **classical**: 全nでengineがstatsmodelsより高速（n=1,000,000で約2.1倍、0.80s vs 1.68s、n=100,000で約2.3倍）。engine は Newton-Raphson の各反復で `k×k` Hessian を構築するため OLS の直接法より計算は重いが、statsmodels 側の Python/patsy/最適化ループのオーバーヘッドの方が大きい。
- **cluster**: engine が全nで約1.8〜2倍速い（n=1,000,000で 1.05s vs 1.89s）。
- **Issue #98 の対称化が効く手法**: statsmodels の `llnull`（切片のみモデル）へのアクセスを計測範囲に含めると、statsmodels の実行時間が遅延評価アクセスなしの約2〜3倍に増える（切片のみ Logit の再フィットが走るため）。engine はこれを常に一括計算しているので、対称に揃えて初めて公平な比較になる（揃えないと engine に不利な非対称計測になっていた）。
- **メモリはengineが軽いが差は小さい**: n=1,000,000でengine 412〜510MB、statsmodels 486〜526MB。OLS/WLS ほどの開きはない（Logit は反復計算で中間行列を持つため）。
- **kスケーリング**: classical k=5→20 で engine 約4.5倍 / statsmodels 約2.7倍。engine の k 方向の伸びがやや急（Newton 各反復の Hessian 構築が `k²` 依存）。絶対値は小さく（k=20 でも 0.03s 台）実用上の問題はないが、傾向として記録する。

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

- **engineのマルチスレッド線形代数の不安定性**（`refactoring-candidates.md`項目44）: OLSと共通の最優先事項。
- **releaseビルドでの再計測が前提**: 改善見込みの見積もりは、debugビルドの数値（誤り）ではなく本ドキュメントのreleaseビルド数値を基準にすること。
