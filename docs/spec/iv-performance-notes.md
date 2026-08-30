# IV: パフォーマンス比較（linearmodels）

`IV(...).fit()`（Rust engine + PyO3）とPython製リファレンス実装 linearmodels の実行時間・メモリ使用量比較の記録。CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けることが目的。

再実行可能なスクリプトは`performance/compare_iv.py`（手法非依存の計測ハーネス`performance/_perf_harness.py` ＋ IV固有アダプタ、コミット対象）。生の計測結果JSONはコミットしない（`.gitignore`の`docs/spec/_*.json`参照）。

比較対象は linearmodels 単体（README「Verification accuracy」表の primary reference。`benchmark/iv/references/linearmodels_ref.py`と同じ主リファレンス）。2SLSは`linearmodels.iv.IV2SLS`、GMMは`linearmodels.iv.IVGMM`に対応させる。

## 計測方法

`docs/spec/ols-performance-notes.md`「最重要の教訓」「計測方法」と共通（releaseビルド必須・`tracemalloc`不採用・サブプロセス隔離・スレッド数を1に固定・polars→pandas変換は計測区間外・ウォームアップ1回＋`repeats`回の中央値・HACのラグ数を`hac_auto_lag(n)`で両ライブラリに揃える）。IV固有の点は以下。

- **DGP**: `generate_iv_dataset("baseline", k_endog=1, k_instruments=2)`。過剰識別（`k_instruments > k_endog`）で回し、engine が常に計算する過剰識別検定（Sargan / Hansen J）を計測範囲に確実に含める。n/k スイープの`k`は外生説明変数`x_exog`の本数（列は `y, x1..xk, endog1, z1, z2`）。
- **method（2sls / gmm）**: n/k スイープは既定 method の **2SLS**。**GMM は method 軸**として代表点1つ（cov_type=classical, k=5, n=1,000,000）でのみ計測する。正確性検証（`test_iv_fixtures.py`）も 2SLS 主軸・GMM は代表シナリオのみ、という絞り方に合わせる。GMM × hac は対象外（下記「既知の限界」）。
- **cov_type**: classical と hac（Newey-West、bartlett kernel）の代表2点。classical/hc1/cluster/hac を n=100,000, k=5 で軽く実測し、OLS/WLS と同じく hac が最重だった（engine 0.129s / linearmodels 0.223s）。engine cov_type ↔ linearmodels `cov_type`/`debiased` の対応は `linearmodels_ref.py` の `_COV_TYPE_MAP` と同じ。`hc2`/`hc3` は linearmodels 側に対応実装が無いため性能比較でも扱わない。
- **計測範囲の対称性（Issue #98）**: engine は係数・標準誤差と同じ `.fit()` の中で R²・調整済みR²・F統計量・過剰識別検定・弱操作変数F統計量・Wu-Hausman検定・第一段階回帰まで**常に一括計算**する。linearmodels の `IVResults` はこれらを遅延評価にしており、特に `first_stage.diagnostics` は**第一段階回帰をフル再fitする**（OLS の `rsquared`、Logit の `llnull` と同じ位置づけ）。`_fit_once_linearmodels` は `.fit()` 直後に `params`/`std_errors`/`tstats`/`pvalues`/`rsquared`/`rsquared_adj`/`f_statistic`/`first_stage.diagnostics`（2SLS は加えて `sargan`・`wu_hausman()`、GMM は `j_stat`）へ明示アクセスし、engine と同じ処理範囲で計測する。
- **スイープ軸**: n軸（k=5固定、n=1,000〜1,000,000）、k軸（n=10,000固定、k=5・20）、method軸（下記）。

## 結果: n軸（k=5固定）

実行時間（秒、中央値）/ ピークRSS（MB）。devcontainer（12論理コア、シングルスレッド固定）、`repeats=3`、method=2sls。

| n | engine | linearmodels |
|---|---|---|
| **classical** | | |
| 1,000 | 0.0009 / 212 | 0.0403 / 216 |
| 10,000 | 0.0079 / 220 | 0.1326 / 250 |
| 100,000 | 0.0833 / 302 | 1.1774 / 589 |
| 1,000,000 | 1.5143 / 1097 | 17.1606 / 3949 |
| **hac** | | |
| 1,000 | 0.0031 / 212 | 0.0748 / 216 |
| 10,000 | 0.0141 / 221 | 0.1842 / 250 |
| 100,000 | 0.1853 / 310 | 1.5186 / 589 |
| 1,000,000 | 2.3971 / 1158 | 17.3030 / 3949 |

## 結果: k軸（n=10,000固定）

実行時間（秒、中央値）。method=2sls。

| k | engine | linearmodels |
|---|---|---|
| **classical** | | |
| 5 | 0.0082 | 0.3187 |
| 20 | 0.0316 | 0.2783 |
| **hac** | | |
| 5 | 0.0114 | 0.1764 |
| 20 | 0.0441 | 0.3447 |

## 結果: method軸（cov_type=classical, k=5, n=1,000,000固定）

実行時間（秒、中央値）。2sls は「結果: n軸」classical の n=1,000,000 行（engine 1.5143 / linearmodels 17.1606）を参照。

| method | engine | linearmodels |
|---|---|---|
| gmm | 0.4880 | 12.2041 |

## 考察

- **classical（2SLS）**: 全nでengineがlinearmodelsより大幅に高速（n=1,000,000で約11倍、1.51s vs 17.16s、n=100,000で約14倍）。差がOLS/WLS（約2倍）より大きいのは、Issue #98 の対称化で linearmodels 側に `first_stage.diagnostics`（第一段階のフル再fit）を含めているため。n=1,000,000 で linearmodels の内訳を実測すると、係数・標準誤差・R²・F統計量・Sargan・Wu-Hausman までで約5.2s、`first_stage.diagnostics` 追加で約16.4s。**この再fitを除いても engine（1.51s）は linearmodels コア（5.2s）の約3.4倍速い**（engine は第一段階を2SLS本体で1回通すだけで弱操作変数Fまで得るため、再fitのコストが実質ゼロ）。
- **hac（2SLS）**: engineが全nで約7〜9倍速い（n=1,000,000で2.40s vs 17.30s）。linearmodels 側は `first_stage.diagnostics`（classicalで再fit）が支配的なため、classical→hac で linearmodels の総時間はほぼ変わらない（17.16s→17.30s）。engine 側は hac 本体（bartlett kernel の三重ループ）の分だけ増える（1.51s→2.40s）。
- **メモリはengineが大幅に軽い**: n=1,000,000 で engine 約1.1GB に対し linearmodels 約3.9GB（約3.6倍）。linearmodels は patsy/pandas が構造式・第一段階の設計行列を複数回フルに構築するため、大規模nでメモリが伸びる。engine は Arrow ゼロコピーで polars をそのまま受け取り、内部行列も faer で必要分のみ確保する。
- **method軸: GMM も engine が大幅に速い**: 代表点（classical, k=5, n=1,000,000）で engine 0.49s vs linearmodels 12.20s（約25倍）。engine の GMM（2ステップ）は 2SLS より速い（0.49s vs 1.51s）— 過剰識別が2本と軽く、かつ第一段階診断の再計算が無いため。
- **kスケーリング（2SLS）**: classical k=5→20 で engine 約3.9倍（0.0082s→0.0316s）。linearmodels は n=10,000 だと固定オーバーヘッド（formula 解釈・第一段階再fit）が支配的で k=5/20 の差が出ない（0.32s / 0.28s、計測ノイズの範囲）。hac も engine 約3.9倍（0.0114s→0.0441s）で、OLS/Logit と同じく engine の k 方向の伸びがやや急な傾向。絶対値は小さい。

## 既知の限界

- **linearmodels の GMM + kernel（hac）が病的に遅い**: `IVGMM` を `weight_type="kernel"` で回すと n=100,000, k=5 で約**40秒**（engine の同条件 0.063s の600倍以上）。n=1,000,000 では数百秒規模になり計測が非現実的なため、**GMM × hac は性能比較の対象から外している**。engine 側の問題ではなく linearmodels の `IVGMM` + kernel weight の実装特性。GMM は classical の代表点のみ計測する。
- **`first_stage.diagnostics` の再fitを計測範囲に含めている**: 上記「計測方法」「考察」のとおり、これは engine が常に計算する弱操作変数診断と処理範囲を揃えるための対称化（Issue #98）であり、linearmodels に不利な非対称計測を避けるための措置。再fitを除いた linearmodels コアでも engine が約3倍速いことは「考察」に併記した。
- その他は `ols-performance-notes.md`「既知の限界」と共通。特に **engineのマルチスレッド線形代数が多コア機・負荷下で不安定になる問題**（`docs/planning/specs/refactoring-candidates.md`項目44）のため、本計測はengine・linearmodelsとも1スレッドに固定しており、数値は「シングルスレッドでの計算コア効率」である。計測は開発コンテナ上の1回のスイープ（`repeats=3`の中央値）で、環境ノイズを排除しきれていない。

## 再現方法

```bash
uv run maturin develop --release
uv run python -m performance.compare_iv --repeats 3 \
    --output docs/spec/_iv_performance_results.json
uv run python -m performance.render_performance_summary \
    docs/spec/_iv_performance_results.json
```

## 今後の検討事項

- **engineのマルチスレッド線形代数の不安定性**（`refactoring-candidates.md`項目44）: OLSと共通。
- **kスケーリング**（上記「考察」参照）: OLS/Logit と共通の傾向。IV は第一段階＋構造式で行列演算が2段になるぶん、k方向の実装効率を見る価値がある。
- **releaseビルドでの再計測が前提**: 改善見込みの見積もりは、debugビルドの数値（誤り）ではなく本ドキュメントのreleaseビルド数値を基準にすること。
