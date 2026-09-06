# WLS: パフォーマンス比較（statsmodels）

`WLS(...).fit()`（Rust engine + PyO3）とリファレンス実装 statsmodels（`smf.wls`）の実行時間・メモリ使用量比較の記録。CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けることが目的。

再実行可能なスクリプトは`performance/compare_wls.py`（手法非依存の計測ハーネス`performance/_perf_harness.py` ＋ WLS固有アダプタ、コミット対象）。生の計測結果JSONはコミットしない（`.gitignore`の`docs/performance/results/*.json`参照）。

## 計測方法

`docs/performance/ols.md`「最重要の教訓」「計測方法」と共通（releaseビルド必須・`tracemalloc`不採用・サブプロセス隔離・スレッド数を1に固定・polars→pandas変換は計測区間外・ウォームアップ1回＋`repeats`回の中央値）。WLS固有の点は以下。

- **重みの渡し方**: engine は `WLS(..., weight="weight")`、statsmodels は `smf.wls(..., weights=pandas_df["weight"])`。どちらも analytic weight（分散の逆数に比例、正規化不要）として扱うため、`generate_linear_dataset("baseline")` が返す `weight` 列（0.5〜1.5の一様乱数）をそのまま両方に渡している。
- **計測範囲の対称性（Issue #98）**: engine は係数・標準誤差と同じ呼び出しで R²・調整済みR²・対数尤度・AIC・BIC・F統計量・F検定のp値まで常に一括計算する。statsmodels はこれらを遅延評価（`cached_value`）にしているため、`.fit()`直後に該当プロパティへ明示アクセスして計測範囲を揃えている。
- **cov_type**: classical と HAC（Newey-West）の代表2点。classical/hc1/cluster/hac を n=100,000, k=5 で軽く実測したところ、engine（classical 0.0124s / hc1 0.0121s / cluster 0.0199s / hac 0.0238s）・statsmodels（classical 0.0218s / hc1 0.0245s / cluster 0.0264s / hac 0.0650s）とも HAC が最重で、OLS と同じ傾向だった。HACのラグ数は `hac_auto_lag(n)` で両ライブラリに明示指定。
- **スイープ軸**: n軸（k=5固定、n=1,000〜1,000,000）、k軸（n=10,000固定、k=5・20）。

## 結果: n軸（k=5固定）

実行時間（秒、中央値）/ ピークRSS（MB）。devcontainer（12論理コア、シングルスレッド固定）、`repeats=3`。

| n | engine | statsmodels |
|---|---|---|
| **classical** | | |
| 1,000 | 0.0001 / 157 | 0.0067 / 202 |
| 10,000 | 0.0012 / 158 | 0.0117 / 205 |
| 100,000 | 0.0137 / 183 | 0.0298 / 236 |
| 1,000,000 | 0.1550 / 401 | 0.3373 / 563 |
| **HAC** | | |
| 1,000 | 0.0002 / 157 | 0.0085 / 202 |
| 10,000 | 0.0015 / 158 | 0.0128 / 206 |
| 100,000 | 0.0239 / 187 | 0.0702 / 246 |
| 1,000,000 | 0.2932 / 455 | 1.1948 / 630 |

## 結果: k軸（n=10,000固定）

実行時間（秒、中央値）。

| k | engine | statsmodels |
|---|---|---|
| **classical** | | |
| 5 | 0.0011 | 0.0117 |
| 20 | 0.0037 | 0.0297 |
| **HAC** | | |
| 5 | 0.0020 | 0.0126 |
| 20 | 0.0073 | 0.0449 |

## 考察

- **classical**: 全nでengineがstatsmodelsより高速。n=1,000,000でengineはstatsmodelsの約2.2倍（0.155s vs 0.337s）、n=100,000で約2.2倍速い。小規模nではstatsmodels側のPython/formula/pandasオーバーヘッドが支配的。
- **HAC**: engineが全nで一貫して速く、大きいnほど差が開く（n=1,000,000で約4.1倍、0.293s vs 1.195s）。
- **メモリはengineが一貫して軽い**: 全cov_type・全nでengineのピークRSSが小さい（n=1,000,000でengine 401〜455MB、statsmodels 563〜630MB）。
- **kスケーリング**: k=5→20で、classicalはengine約3.4倍 / statsmodels約2.5倍、HACはengine約3.7倍 / statsmodels約3.6倍。OLSではHACのkスケーリングでengineの伸び（約7.7倍）がリファレンス（約3.3倍）より急という点が残っていたが、WLSではHACでも両者ほぼ同等で、その乖離は見られない。
- **WLS vs OLS**: 同条件のOLS（`ols.md`）と比べ、engineのclassical n=1,000,000は0.139s→0.155sと重み付けの分だけ僅かに重い程度で、全体傾向はOLSと変わらない。

## 既知の限界

`ols.md`「既知の限界」と共通。特に **engineのマルチスレッド線形代数が多コア機・負荷下で不安定になる問題**（`docs/planning/specs/refactoring-candidates.md`項目44）のため、本計測はengine・statsmodelsとも1スレッドに固定しており、数値は「シングルスレッドでの計算コア効率」である。計測は開発コンテナ上の1回のスイープ（`repeats=3`の中央値）で、環境ノイズを排除しきれていない。

## 再現方法

```bash
uv run maturin develop --release
uv run python -m performance.compare_wls --repeats 3 \
    --output docs/performance/results/wls.json
uv run python -m performance.render_performance_summary \
    docs/performance/results/wls.json
```

## 今後の検討事項

- **engineのマルチスレッド線形代数の不安定性**（`refactoring-candidates.md`項目44）: OLSと共通の最優先事項。
- **releaseビルドでの再計測が前提**: 改善見込みの見積もりは、debugビルドの数値（誤り）ではなく本ドキュメントのreleaseビルド数値を基準にすること。
