# IV: パフォーマンス比較（linearmodels）

`IV(...).fit()`（Rust engine + PyO3）とPython製リファレンス実装 linearmodels の実行時間・メモリ使用量比較の記録。CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けることが目的。

再実行可能なスクリプトは`performance/compare_iv.py`（手法非依存の計測ハーネス`performance/_perf_harness.py` ＋ IV固有アダプタ、コミット対象）。生の計測結果JSONはコミットしない（`.gitignore`の`docs/performance/results/*.json`参照）。

比較対象は linearmodels 単体（[検証ページ](../guide/verification.md)の primary reference。`benchmark/iv/references/linearmodels_ref.py`と同じ主リファレンス）。2SLSは`linearmodels.iv.IV2SLS`、GMMは`linearmodels.iv.IVGMM`に対応させる。

> **役割分担**: 計測結果の表と性能特性の解釈は、公開ページ（英語。CIの計測値から生成）の[Performance](../guide/performance.md)・[Performance results](../guide/performance-results.md)に置く。このノートは計測方法論・設計判断・既知の限界・今後の検討を記録する。

## 計測方法

`docs/performance/ols.md`「最重要の教訓」「計測方法」と共通（releaseビルド必須・`tracemalloc`不採用・サブプロセス隔離・スレッド数を1に固定・polars→pandas変換は計測区間外・ウォームアップ1回＋`repeats`回の中央値・HACのラグ数を`hac_auto_lag(n)`で両ライブラリに揃える）。IV固有の点は以下。

- **DGP**: `generate_iv_dataset("baseline", k_endog=1, k_instruments=2)`。過剰識別（`k_instruments > k_endog`）で回し、engine が常に計算する過剰識別検定（Sargan / Hansen J）を計測範囲に確実に含める。n/k スイープの`k`は外生説明変数`x_exog`の本数（列は `y, x1..xk, endog1, z1, z2`）。
- **method（2sls / gmm）**: n/k スイープは既定 method の **2SLS**。**GMM は method 軸**として代表点1つ（cov_type=classical, k=5, n=1,000,000）でのみ計測する。正確性検証（`test_iv_reference.py`）も 2SLS 主軸・GMM は代表シナリオのみ、という絞り方に合わせる。GMM × hac は対象外（下記「既知の限界」）。
- **cov_type**: classical と hac（Newey-West、bartlett kernel）の代表2点。classical/hc1/cluster/hac を n=100,000, k=5 で軽く実測し、OLS/WLS と同じく hac が最重だった（engine 0.129s / linearmodels 0.223s）。engine cov_type ↔ linearmodels `cov_type`/`debiased` の対応は `linearmodels_ref.py` の `_COV_TYPE_MAP` と同じ。`hc2`/`hc3` は linearmodels 側に対応実装が無いため性能比較でも扱わない。
- **計測範囲の対称性**: engine は係数・標準誤差と同じ `.fit()` の中で R²・調整済みR²・F統計量・過剰識別検定・弱操作変数F統計量・Wu-Hausman検定・第一段階回帰まで**常に一括計算**する。linearmodels の `IVResults` はこれらを遅延評価にしており、特に `first_stage.diagnostics` は**第一段階回帰をフル再fitする**（OLS の `rsquared`、Logit の `llnull` と同じ位置づけ）。`_fit_once_linearmodels` は `.fit()` 直後に `params`/`std_errors`/`tstats`/`pvalues`/`rsquared`/`rsquared_adj`/`f_statistic`/`first_stage.diagnostics`（2SLS は加えて `sargan`・`wu_hausman()`、GMM は `j_stat`）へ明示アクセスし、engine と同じ処理範囲で計測する。
- **スイープ軸**: n軸（k=5固定、n=1,000〜1,000,000）、k軸（n=10,000固定、k=5・20）、method軸（下記）。

## 考察（結果表の外に残す、機構・経緯の記録）

- **OLS/WLSより差が大きい理由**: 対称化で linearmodels 側に `first_stage.diagnostics`（第一段階のフル再fit）を含めているため。n=1,000,000 で linearmodels の内訳を実測すると、係数・標準誤差・R²・F統計量・Sargan・Wu-Hausman までで約5.2s、`first_stage.diagnostics` 追加で約16.4s。**この再fitを除いても engine（1.51s）は linearmodels コア（5.2s）の約3.4倍速い**（engine は第一段階を2SLS本体で1回通すだけで弱操作変数Fまで得るため、再fitのコストが実質ゼロ）。
- **メモリはengineが大幅に軽い**: n=1,000,000 で engine 約1.1GB に対し linearmodels 約3.9GB（約3.6倍）。linearmodels は patsy/pandas が構造式・第一段階の設計行列を複数回フルに構築するため、大規模nでメモリが伸びる。engine は Arrow ゼロコピーで polars をそのまま受け取り、内部行列も faer で必要分のみ確保する。
- **method軸: GMM も engine が大幅に速い**: 代表点（classical, k=5, n=1,000,000）で engine 0.49s vs linearmodels 12.20s（約25倍）。engine の GMM（2ステップ）は 2SLS より速い（0.49s vs 1.51s）— 過剰識別が2本と軽く、かつ第一段階診断の再計算が無いため。

## 既知の限界

- **linearmodels の GMM + kernel（hac）が病的に遅い**: `IVGMM` を `weight_type="kernel"` で回すと n=100,000, k=5 で約**40秒**（engine の同条件 0.063s の600倍以上）。n=1,000,000 では数百秒規模になり計測が非現実的なため、**GMM × hac は性能比較の対象から外している**。engine 側の問題ではなく linearmodels の `IVGMM` + kernel weight の実装特性。GMM は classical の代表点のみ計測する。
- **`first_stage.diagnostics` の再fitを計測範囲に含めている**: 上記「計測方法」「考察」のとおり、これは engine が常に計算する弱操作変数診断と処理範囲を揃えるための対称化であり、linearmodels に不利な非対称計測を避けるための措置。再fitを除いた linearmodels コアでも engine が約3倍速いことは「考察」に併記した。
- その他は `ols.md`「既知の限界」と共通。特に **engineのマルチスレッド線形代数が多コア機・負荷下で不安定になる問題**のため、本計測はengine・linearmodelsとも1スレッドに固定しており、数値は「シングルスレッドでの計算コア効率」である。計測は開発コンテナ上の1回のスイープ（`repeats=3`の中央値）で、環境ノイズを排除しきれていない。

## 再現方法

```bash
uv run maturin develop --release
uv run python -m performance.compare_iv --repeats 3 \
    --output docs/performance/results/iv.json
uv run python -m performance.render_performance_summary \
    docs/performance/results/iv.json
```

## 今後の検討事項

- **engineのマルチスレッド線形代数の不安定性**: OLSと共通。
- **kスケーリング**（公開ページの結果表参照）: OLS/Logit と共通の傾向。IV は第一段階＋構造式で行列演算が2段になるぶん、k方向の実装効率を見る価値がある。
- **releaseビルドでの再計測が前提**: 改善見込みの見積もりは、debugビルドの数値（誤り）ではなく本ドキュメントのreleaseビルド数値を基準にすること。
