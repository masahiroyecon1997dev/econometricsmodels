# WLS: パフォーマンス比較（statsmodels）

`WLS(...).fit()`（Rust engine + PyO3）とリファレンス実装 statsmodels（`smf.wls`）の実行時間・メモリ使用量比較の記録。CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けることが目的。

再実行可能なスクリプトは`performance/compare_wls.py`（手法非依存の計測ハーネス`performance/_perf_harness.py` ＋ WLS固有アダプタ、コミット対象）。生の計測結果JSONはコミットしない（`.gitignore`の`docs/performance/results/*.json`参照）。

> **役割分担**: 計測結果の表と性能特性の解釈は、公開ページ（英語。CIの計測値から生成）の[Performance](../guide/performance.md)・[Performance results](../guide/performance-results.md)に置く。このノートは計測方法論・設計判断・既知の限界・今後の検討を記録する。

## 計測方法

`docs/performance/ols.md`「最重要の教訓」「計測方法」と共通（releaseビルド必須・`tracemalloc`不採用・サブプロセス隔離・スレッド数を1に固定・polars→pandas変換は計測区間外・ウォームアップ1回＋`repeats`回の中央値）。WLS固有の点は以下。

- **重みの渡し方**: engine は `WLS(..., weight="weight")`、statsmodels は `smf.wls(..., weights=pandas_df["weight"])`。どちらも analytic weight（分散の逆数に比例、正規化不要）として扱うため、`generate_linear_dataset("baseline")` が返す `weight` 列（0.5〜1.5の一様乱数）をそのまま両方に渡している。
- **計測範囲の対称性**: engine は係数・標準誤差と同じ呼び出しで R²・調整済みR²・対数尤度・AIC・BIC・F統計量・F検定のp値まで常に一括計算する。statsmodels はこれらを遅延評価（`cached_value`）にしているため、`.fit()`直後に該当プロパティへ明示アクセスして計測範囲を揃えている。
- **cov_type**: n軸・k軸は既定の classical のみ。標準誤差の種類は **cov軸**（n=100,000・k=5）で全て測る: classical・hc0〜hc3・hac、cluster の 2×2（G=50 / G=n÷10 × クラスターの行が散らばる random / 連続する sorted）。全て statsmodels と比較する。以前は「最も軽い classical と最も重い HAC」の2点だけを実装時の軽い実測で手書き選定していたが、cov_type の順位は実装次第で入れ替わり、選ばなかった HC・cluster の劣化がどこにも現れないため、cov 軸に置き換えた（`.claude/rules/testing-policy.md`「パフォーマンス比較（ベンチマーク）の方法論」）。HACのラグ数は `hac_auto_lag(n)` で両ライブラリに明示指定する。重みは全ケース共通。
- **スイープ軸**: n軸（k=5固定、n=10,000 / 100,000 / 1,000,000）、k軸（n=10,000固定、k=5・20）、cov軸（上記）。n軸・k軸は既定の classical のみ。

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
