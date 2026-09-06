# Tobit: パフォーマンス比較（py4etrics）

`Tobit(...).fit()`（Rust engine + PyO3）とリファレンス実装 py4etrics（`statsmodels.GenericLikelihoodModel` ベースの Tobit）の実行時間・メモリ使用量比較の記録。CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けることが目的。

再実行可能なスクリプトは`performance/compare_tobit.py`（手法非依存の計測ハーネス`performance/_perf_harness.py` ＋ Tobit固有アダプタ、コミット対象）。生の計測結果JSONはコミットしない（`.gitignore`の`docs/performance/results/*.json`参照）。

## 計測方法

`docs/performance/ols.md`「最重要の教訓」「計測方法」・`probit.md`と共通（releaseビルド必須・`tracemalloc`不採用・サブプロセス隔離・スレッド数を1に固定・polars→pandas変換は計測区間外・ウォームアップ1回＋`repeats`回の中央値）。Tobit固有の点は以下。

- **リファレンスが py4etrics（R ではない）**: Tobit の正確性検証の主リファレンスは R `AER::tobit`（＝`survreg`）だが、R は共通ハーネスのインプロセス計測モデル（`fit_once(ctx)` を計測ループ内で呼ぶ）に乗らず、`benchmark_performance.yml` への R 導入も要る。statsmodels にネイティブ Tobit は無い。**py4etrics** は `statsmodels.GenericLikelihoodModel` ベースの Tobit を pure Python でパッケージ化したもので、インプロセス計測できる。係数・σ・対数尤度が engine と ~1e-9 で一致することを実機確認済み（`moderate_censoring`, n=1,000〜100,000, classical/cluster, newton）。正式な数値照合は従来どおり R ベースの `tests/nonlinear/test_tobit_reference.py`（`AER::tobit`）・`test_tobit_crosscheck.py`（`censReg`）が担い、ここでは性能の相対傾向のみを見る。
- **核心の非対称: 解析的微分 vs 数値微分**: engine は Tobit の対数尤度のスコア・ヘッシアンを解析式で Rust 実装する。py4etrics（`GenericLikelihoodModel`）は有限差分でスコア・ヘッシアンを数値近似する。したがって本比較の大きな差は「Rust vs Python」だけでなく「手で導出した解析的微分 vs 汎用の数値微分」の寄与を含む。Tobit のように尤度が閉形式で微分できる手法で解析的実装がどれだけ効くかを示す計測でもある。
- **計測範囲の対称性**: engine は係数・標準誤差と同じ呼び出しで対数尤度・AIC・BIC・全体 Wald 統計量まで常に一括計算する。py4etrics（statsmodels）の統計量は遅延評価（`llf` はアクセス時に `loglike` を再計算、`aic`/`bic` はプロパティ）なので、`_fit_once_py4etrics` は `.fit()` 直後に `llf`/`aic`/`bic` へ明示アクセスして揃える。全体 Wald 検定・限界効果は py4etrics 側が自動計算しないため対称化の対象外（Logit/Probit の `llnull` と同じ整理）。
- **cov_type**: classical と cluster の代表2点。`opg`/`hc0`/`hc1` は省略（Logit/Probit と同じ絞り方）。cluster の疑似グループ数は 50 固定。
- **k 軸は engine 単独**（`PerfAdapter.k_sweep_libraries=("engine",)`）: py4etrics の数値微分ヘッシアンは k に対してコストが崖状に悪化し、k=5 は約1.5秒だが k>=8 で事実上フリーズする（n=10,000 でも実機確認）。k 軸のスケーリング比較にならないため engine のみ回す（n 軸・method 軸では py4etrics を比較対象に使う）。
- **method（オプティマイザ）**: engine・py4etrics とも Newton-Raphson（`method="newton"`）で n/k スイープを回す。加えて `lbfgs` を **method 軸**として代表点1つ（cov_type=classical, k=5, n=100,000）で計測する。**`bfgs` は計測対象外**: engine の Tobit BFGS 経路は n>=10,000 で `MoreThuenteLineSearch: NaN or Inf` により発散する（#292）。解消後に戻す。
- **quasi-Newton の劣化ガード**: `compare_tobit.py` の `check_report`（`_check_method_ratios`）が engine の `lbfgs/newton` 実行時間比を計算し、5x を超えたら job summary に `> [!WARNING]` を出す（CI failure にはしない。実時間の絶対値ではなく同一ジョブ内の比なので共有ランナーの速度差に影響されない。#285 と同系統の劣化の早期検知）。
- **打ち切りシナリオ**: `moderate_censoring`（左打ち切り ~35%、`benchmark/nonlinear/datasets.py` の `_TOBIT_SCENARIO_CONFIG`。Tobit テストの `BASELINE_SCENARIO` と同じ）。潜在回帰の誤差 SD（真の σ）は `_TOBIT_ERROR_SD`。
- **スイープ軸**: n軸（k=5固定、n=1,000〜**100,000**、engine/py4etrics）、k軸（n=10,000固定、k=5・20、engine のみ）、method軸（上記）。**n軸は 1,000,000 を含まない**（下記「既知の限界」）。

計測環境: devcontainer（12論理コア、シングルスレッド固定）、`repeats=3`、seed=42、scenario=moderate_censoring、release build。下表は 2026-09-06 のローカル実測（CI の job summary は毎タグ push で更新され、共有ランナーのため数値はぶれる）。

## 結果: n軸（k=5固定）

実行時間（秒、中央値）/ ピークRSS（MB）。method=newton。

### classical

| n | engine | py4etrics | 比 |
|---|---|---|---|
| 1,000 | 0.0011s / 204MB | 0.2212s / 232MB | ~200x |
| 10,000 | 0.0114s / 207MB | 1.1331s / 236MB | ~99x |
| 100,000 | 0.1400s / 238MB | 12.7498s / 263MB | ~91x |

### cluster

| n | engine | py4etrics | 比 |
|---|---|---|---|
| 1,000 | 0.0019s / 205MB | 0.3189s / 232MB | ~168x |
| 10,000 | 0.0159s / 209MB | 1.5056s / 237MB | ~95x |
| 100,000 | 0.1858s / 260MB | 14.5347s / 269MB | ~78x |

## 結果: k軸（n=10,000固定、engine のみ）

実行時間（秒、中央値）。method=newton。

| k | engine classical | engine cluster |
|---|---|---|
| 5 | 0.0133s | 0.0155s |
| 20 | 0.0522s | 0.0617s |

## 結果: method軸（cov_type=classical, k=5, n=100,000固定）

実行時間（秒、中央値）。newton は「結果: n軸」classical の n=100,000 行を参照。

| method | engine | py4etrics |
|---|---|---|
| newton | 0.1400s | 12.7498s |
| lbfgs | 0.5016s | 2.6436s |

engine の `lbfgs/newton` 比は約 3.6x で、`_check_method_ratios` の想定上限 5x 以内のため WARNING は出ていない。

## 考察

- **classical（newton）**: 全 n で engine が py4etrics より圧倒的に速い（n=100,000 で **約91倍**、0.140s vs 12.75s）。比が n とともに 200x → 91x に縮むのは engine が劣化しているのではなく、**py4etrics 側の固定オーバーヘッド**（statsmodels モデル構築で ~0.2秒）が n とともに相対的に薄まるため。engine 自体の n スケーリングは概ね線形（1,000→10,000 で ~10.4x、10,000→100,000 で ~12.3x、いずれも 10倍のデータに対し概ね線形）。
- **cluster（newton）**: 同傾向（n=100,000 で約78倍）。engine の cluster は 10,000→100,000 で ~11.7x で classical（~12.3x）とほぼ同じ伸び。ピーク RSS は engine 260MB vs py4etrics 269MB で同等。
- **解析的微分 vs 数値微分の寄与**: この ~80〜200倍差の主因は Rust 化だけでなく、engine が Tobit 対数尤度のスコア・ヘッシアンを**解析式**で持つのに対し、py4etrics（`GenericLikelihoodModel`）が**有限差分**で近似すること。k を増やすと py4etrics の数値ヘッシアンは O(k²) 回の対数尤度評価を要し、k=5→8 で約1.5秒→150秒超に崖状に悪化する（k 軸を engine 単独にした理由）。
- **k スケーリング（engine, newton）**: classical k=5→20（k 4倍）で 0.0133s→0.0522s（~3.9x）、cluster も 0.0155s→0.0617s（~4.0x）。k 方向は概ね線形〜やや緩く、健全。
- **method軸**: engine の lbfgs（0.502s）は newton（0.140s）の **約3.6倍**。probit の #285（newton 比 ~7倍）ほど極端ではないが同系統の遅さで、quasi-Newton 実装に改善余地がある。py4etrics の lbfgs（2.64s）は自身の newton（12.75s）より速い（数値ヘッシアンが不要なため）。**bfgs は engine が n>=10,000 で発散する（#292）ため計測対象外**。
- **改善余地**: engine の絶対性能は n=100,000 で 0.14〜0.19秒と実用上問題ないが、(a) quasi-Newton（lbfgs 3.6x・bfgs 発散 #292）、(b) newton の大標本での Hessian 特異（#291）が engine 側の Tobit MLE の継続課題。cluster 経路は classical とほぼ同じ伸びで、現時点で特段の懸念はない。

## 既知の限界

- **n軸が 100,000 まで**: 2つの理由による。
    1. py4etrics は数値微分ゆえ大 n で極端に遅い（newton・n=100,000・k=5 で約13秒。engine は約0.14秒）。n=1,000,000 は分オーダーで、両者の比較として非現実的。
    2. engine 側も乱数 β の `moderate_censoring` DGP で大 n・特定 seed の際に `ComputationError: the Hessian is singular and cannot be inverted` になる（seed 依存。seed=42 は n=500,000 まで成功・n=1,000,000 で失敗。seed=1 は n=200,000 で既に失敗。**Issue #291**、Probit の #284 と同系統の engine 側頑健性の課題）。
- **k軸が engine 単独・k は 20 まで**: py4etrics の数値微分ヘッシアンが k>=8 で破綻するため k 軸は engine のみ（「計測方法」参照）。
- **method軸に bfgs を含まない**: engine の Tobit BFGS 経路が n>=10,000 で発散する（**Issue #292**）。#292 解消後に `compare_tobit.py` の `extra_methods` へ戻す。
- その他は `ols.md`「既知の限界」と共通。特に **engineのマルチスレッド線形代数が多コア機・負荷下で不安定になる問題**（#283）のため、本計測はengine・py4etricsとも1スレッドに固定しており、数値は「シングルスレッドでの計算コア効率」である。

## 再現方法

```bash
uv run maturin develop --release
uv run python -m performance.compare_tobit --repeats 3 \
    --output docs/performance/results/tobit.json
uv run python -m performance.render_performance_summary \
    docs/performance/results/tobit.json
```

## 今後の検討事項

- **engineのTobitのHessian特異化**（#291）: py4etrics/statsmodels が捌ける大標本条件で engine が失敗する（seed 依存）。解析的ヘッシアン構築・Newton ソルバの頑健化の余地を調査する。解消後に n=1,000,000 を n軸に追加して再計測する。
- **engineのTobit BFGSが発散する**（#292）: n>=10,000 で `MoreThuenteLineSearch: NaN or Inf`。解消後に method軸へ bfgs を戻す。
- **engineのquasi-Newton（L-BFGS）が遅い**（#285）: Logit/Probit と共通。Tobit では lbfgs/newton ~3.6x（probit の ~7x よりは軽い）。`_check_method_ratios` が 5x 超で job summary に警告する。
- **engineのマルチスレッド線形代数の不安定性**（#283）: OLSと共通。
- **releaseビルドでの再計測が前提**: 改善見込みの見積もりは、debugビルドの数値（誤り）ではなく本ドキュメントのreleaseビルド数値を基準にすること。
- **py4etrics の保守状況**: 最終リリース 2024-01、依存ピン無し。`statsmodels==0.14.6` 固定なので現状問題ないが、statsmodels を上げる際は py4etrics の動作確認とセットで行う（`pyproject.toml` の `benchmark` グループのコメント参照）。
