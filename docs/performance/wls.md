# WLS: パフォーマンス比較（statsmodels）

`WLS(...).fit()`（Rust engine + PyO3）とリファレンス実装 statsmodels（`smf.wls`）の実行時間・メモリ使用量比較の記録。CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けることが目的。

再実行可能なスクリプトは`performance/compare_wls.py`（手法非依存の計測ハーネス`performance/_perf_harness.py` ＋ WLS固有アダプタ、コミット対象）。生の計測結果JSONはコミットしない（`.gitignore`の`docs/performance/results/*.json`参照）。

> **役割分担**: 計測結果の表と性能特性の解釈は、公開ページ（英語。CIの計測値から生成）の[Performance](../guide/performance.md)・[Performance results](../guide/performance-results.md)に置く。このノートは計測方法論・設計判断・既知の限界・今後の検討を記録する。

## 計測方法

`docs/performance/ols.md`「最重要の教訓」「計測方法」と共通（releaseビルド必須・`tracemalloc`不採用・サブプロセス隔離・スレッド数を1に固定・polars→pandas変換は計測区間外・ウォームアップ1回＋`repeats`回の中央値）。WLS固有の点は以下。

- **重みの渡し方**: engine は `WLS(..., weight="weight")`、statsmodels は `smf.wls(..., weights=pandas_df["weight"])`。どちらも analytic weight（分散の逆数に比例、正規化不要）として扱うため、`generate_linear_dataset("baseline")` が返す `weight` 列（0.5〜1.5の一様乱数）をそのまま両方に渡している。
- **計測範囲の対称性**: engine は係数・標準誤差と同じ呼び出しで R²・調整済みR²・対数尤度・AIC・BIC・F統計量・F検定のp値まで常に一括計算する。statsmodels はこれらを遅延評価（`cached_value`）にしているため、`.fit()`直後に該当プロパティへ明示アクセスして計測範囲を揃えている。
- **cov_type**: classical と HAC（Newey-West）の代表2点。classical/hc1/cluster/hac を n=100,000, k=5 で軽く実測したところ、engine（classical 0.0124s / hc1 0.0121s / cluster 0.0199s / hac 0.0238s）・statsmodels（classical 0.0218s / hc1 0.0245s / cluster 0.0264s / hac 0.0650s）とも HAC が最重で、OLS と同じ傾向だった。HACのラグ数は `hac_auto_lag(n)` で両ライブラリに明示指定。
- **スイープ軸**: n軸（k=5固定、n=1,000〜1,000,000）、k軸（n=10,000固定、k=5・20）。

## 既知の限界

`ols.md`「既知の限界」と共通。特に **engineのマルチスレッド線形代数が多コア機・負荷下で不安定になる問題**のため、本計測はengine・statsmodelsとも1スレッドに固定しており、数値は「シングルスレッドでの計算コア効率」である。計測は開発コンテナ上の1回のスイープ（`repeats=3`の中央値）で、環境ノイズを排除しきれていない。

## 再現方法

```bash
uv run maturin develop --release
uv run python -m performance.compare_wls --repeats 3 \
    --output docs/performance/results/wls.json
uv run python -m performance.render_performance_summary \
    docs/performance/results/wls.json
```

## 今後の検討事項

- **engineのマルチスレッド線形代数の不安定性**: OLSと共通の最優先事項。
- **releaseビルドでの再計測が前提**: 改善見込みの見積もりは、debugビルドの数値（誤り）ではなく本ドキュメントのreleaseビルド数値を基準にすること。
