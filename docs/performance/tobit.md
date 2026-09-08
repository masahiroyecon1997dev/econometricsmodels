# Tobit: パフォーマンス比較

`Tobit(...).fit()`（Rust engine + PyO3）の実行時間・ピークメモリの記録。CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けることが目的。

他手法ページ（`ols.md` 等）と違い **engine 単独の計測**で、リファレンス実装との相対比較は行わない。engine の Tobit が軸ごとに「時間がかかりすぎていないか」（絶対値・スケーリングの推移）を観察する。正確性の数値照合は R ベースの `tests/nonlinear/test_tobit_reference.py`（`AER::tobit`）・`test_tobit_crosscheck.py`（`censReg`）が担う。

再実行可能なスクリプトは `performance/compare_tobit.py`（手法非依存の計測ハーネス `performance/_perf_harness.py` ＋ Tobit 固有アダプタ、コミット対象）。生の計測結果 JSON はコミットしない（`.gitignore` の `docs/performance/results/*.json` 参照）。

> [!NOTE]
> **py4etrics 撤去（2026-09）**: 2026-09 以前はリファレンス実装として py4etrics（`statsmodels.GenericLikelihoodModel` ベースの Tobit）を n 軸・method 軸に並べていた。statsmodels 0.15.0 更新に合わせ、(1) py4etrics が実質メンテ停止（最終リリース 2024-01、依存ピン無し）で statsmodels 0.15.0 では import 不能、(2) 数値微分ヘッシアンのため k>=8 で事実上フリーズし k 軸は元々 engine 単独だった、(3)「engine の解析的スコア／ヘッシアン vs statsmodels の有限差分」という交絡を含み Rust 化そのものの寄与を単独では取り出せない、の3点から撤去した。撤去前の最終クロス比較スナップショット（statsmodels 0.14.6 / py4etrics 0.1.9、2026-09-06〜07 実測）は末尾「[アーカイブ] py4etrics 比較スナップショット」に凍結してある。engine 列の数値は engine 側が変わらない限りそのまま最新値。
>
> このため **Tobit について「CLAUDE.md 1章 の Rust 高速化を裏付けるライブの相対比較」は他5手法と非対称で、定量根拠は 2026-09 の凍結スナップショットに依存する**。かつその ~80〜200x は「Rust vs Python」に「解析的スコア／ヘッシアン vs 有限差分」の寄与が混ざった値で、**クリーンな Rust 化単独の高速化率として引用しない**こと（アーカイブ節「考察（当時）」参照）。将来インプロセス計測できる保守されたネイティブ Tobit 実装が現れた場合はライブ比較を再検討する。

## 計測方法（現行）

`docs/performance/ols.md`「最重要の教訓」「計測方法」・`probit.md` と共通（release ビルド必須・`tracemalloc` 不採用・サブプロセス隔離・スレッド数を1に固定・ウォームアップ1回＋`repeats` 回の中央値）。Tobit 固有の点は以下。

- **engine 単独**（`PerfAdapter.libraries=("engine",)`）。インプロセス計測できるリファレンス実装が無い（主リファレンスの R `AER::tobit` は共通ハーネスのインプロセス計測に乗らず、statsmodels にネイティブ Tobit も無く、代替の py4etrics は上記のとおり撤去）。レポートは相対比較ではなく engine の絶対値・スケーリングの推移になる。
- **cov_type**: classical と cluster の代表2点。`opg`/`hc0`/`hc1` は省略（Logit/Probit と同じ絞り方）。n=100,000・k=5 で全 cov_type を実測して確認済み（classical 0.138s < opg 0.148s < hc1 0.150s < hc0 0.154s < cluster 0.162s、newton）。cluster の疑似グループ数は 50 固定。
- **打ち切りシナリオ**: `moderate_censoring`（左打ち切り ~35%、`benchmark/nonlinear/datasets.py` の `_TOBIT_SCENARIO_CONFIG`。Tobit テストの `BASELINE_SCENARIO` と同じ）。潜在回帰の誤差 SD（真の σ）は `_TOBIT_ERROR_SD`。
- **スイープ軸**:
    - n 軸（k=5 固定、newton）: classical / cluster とも n=1,000〜**100,000**。加えて **classical のみ n=200,000 / 1,000,000**（`n_sweep_engine_only`、seed=42 のみ。全 cov_type で大 n を回すと CI 時間がかさむため classical に絞る）。n=1,000,000 が Issue #291 の再現点で修正の回帰ガード、n=200,000 は追加のスケーリング点（下記「n 軸の大標本ガード」）。
    - k 軸（n=10,000 固定、newton）: k=5・20、classical / cluster。
    - method 軸: `lbfgs` を代表点1つ（classical・k=5・n=100,000）で計測。**`bfgs` は除外**（engine の Tobit BFGS が n>=10,000 で `MoreThuenteLineSearch: NaN or Inf` 発散、#292。解消後に `extra_methods` へ戻す）。
- **quasi-Newton の劣化ガード**: `compare_tobit.py` の `check_report`（`_check_method_ratios`）が engine の `lbfgs/newton` 実行時間比を計算し、5x を超えたら job summary に `> [!WARNING]` を出す（CI failure にはしない。実時間の絶対値ではなく同一ジョブ内の比なので共有ランナーの速度差に影響されない。#285 と同系統の劣化の早期検知）。

## n 軸の大標本ガード（classical のみ n=200,000 / 1,000,000、seed=42）

- かつて engine は乱数 β の `moderate_censoring` DGP で `ComputationError: the Hessian is singular and cannot be inverted` になっていた（seed 依存。seed=42 は n=500,000 まで成功・n=1,000,000 で失敗、seed=1 は n=200,000 で失敗。**Issue #291**、Probit の #284 と同系統）。`d797f9b` / `5b79ffe` で `FaerNewton` に停滞収束判定を入れて解消済み。
- **n=1,000,000（seed=42）が #291 の再現点**で、再発を benchmark ジョブの失敗（`_run_isolated` は `check=True`）として捕捉する回帰ガード。実 Tobit 経路を大標本で通す唯一の自動チェック（`engine/src/nonlinear/common.rs` の単体テストは停滞判定ロジックの模擬）。n=1,000,000 は `n_iter=11` で真の β と ~1e-3、σ≈1.001、logLik≈-1.03e6 の正しい最適点へ収束することを確認済み（停滞判定の早期打ち切りではない）。
- **n=200,000 は n スケーリングのデータ点＋安価な早期警告**。guard が回す seed=42 では #291 を再現しない（破綻していたのは seed=1）。
- **この guard の限界**（深いカバレッジは「今後の検討事項」の凍結フィクスチャ + `AER::tobit` に委ねる）: (1) **単一 seed（42）**——`run_cli` の `--seed` はレポート全体で1つのため、#291 が示した seed 感度は検査できない。(2) **捕捉できるのは再発時の「例外」のみ**——#291 の修正が持ち込みうる silently-wrong な収束（非最適点で収束宣言）は、finite な結果さえ返れば `check=True` を通る。(3) **発火はリリース単位**——`benchmark_performance.yml` はタグ push（`v*`）+ `workflow_dispatch` のみで、solver 回帰がリリースブランチにマージされても次のタグまで捕捉されない。

## 結果（engine 単独）

- **正の出力**: `benchmark_performance.yml` がタグ push（`v*`）ごとに `compare_tobit.py` → `render_performance_summary.py` を回した job summary。共有ランナーのため数値はぶれる（`ols.md`「計測方法」と同じ前提）。
- **代表値**: 下記「[アーカイブ]」節の各表の **engine 列**（n=1,000〜1,000,000、k=5・20、newton / lbfgs）。py4etrics 撤去は engine 側の計算経路に影響しないため、engine 側が変わらない限りこれらが現行の代表値。要点: n=100,000・classical・newton で ~0.15s / ~244MB、n スケーリングは概ね線形、lbfgs は newton の ~3.2x（`_check_method_ratios` の上限 5x 内）、n=1,000,000 も `n_iter=11` で正しい最適点へ収束（#291 解消）。
- **1スレッド固定の解釈**: engine のマルチスレッド線形代数が多コア機・負荷下で不安定になる問題（#283）のため本計測は1スレッドに固定しており、数値は「シングルスレッドでの計算コア効率」。多コアでの実利用の性能特性とは別軸（`ols.md`「既知の限界」と共通）。

## 再現方法

```bash
uv run maturin develop --release
uv run python -m performance.compare_tobit --repeats 3 \
    --output docs/performance/results/tobit.json
uv run python -m performance.render_performance_summary \
    docs/performance/results/tobit.json
```

## 今後の検討事項

- **~~engineのTobitのHessian特異化~~（#291、解消済み）**: `d797f9b` / `5b79ffe` で `FaerNewton` の収束判定に停滞検出（`RegularizedStep::NoProgress` + 勾配停滞 + 目標近傍 + コストHessian正定値）を追加。n 軸に engine 単独の n=200,000 / 1,000,000 行を追加済み（回帰ガード）。残る派生検討: (a) `NEWTON_STALL_GRAD_FACTOR` の絶対閾値を Newton 減少量 `√(gᵀH⁻¹g)` ベースのスケール不変な基準に置き換える（`docs/planning/specs/nonlinear-implementation-notes.md`）、(b) 大 n の凍結フィクスチャ + `AER::tobit` 数値クロスチェック（正確性は scipy 数値微分MLE との照合で logLik 相対誤差 2.6e-13 を確認済みのため優先度は低い）。
- **engineのTobit BFGSが発散する**（#292）: n>=10,000 で `MoreThuenteLineSearch: NaN or Inf`。解消後に method軸へ bfgs を戻す。
- **engineのquasi-Newton（L-BFGS）が遅い**（#285）: Logit/Probit と共通。Tobit では lbfgs/newton ~3x（probit の ~7x よりは軽い）。`_check_method_ratios` が 5x 超で job summary に警告する。
- **engineのマルチスレッド線形代数の不安定性**（#283）: OLSと共通。
- **releaseビルドでの再計測が前提**: 改善見込みの見積もりは、debugビルドの数値（誤り）ではなく本ドキュメントのreleaseビルド数値を基準にすること。
- **Tobit のライブ性能リファレンス**: 現状は engine 単独。将来 statsmodels がネイティブ Tobit を持つ、またはインプロセス計測できる保守された実装が現れた場合は n 軸・method 軸のリファレンスとして再検討する（py4etrics 撤去の経緯は冒頭ノート）。

---

## [アーカイブ] py4etrics 比較スナップショット（statsmodels 0.14.6 / py4etrics 0.1.9）

以下は py4etrics を撤去する前（2026-09-06〜07 実測）の engine vs py4etrics クロス比較の凍結記録。**py4etrics は撤去済み**（冒頭ノート）で、これらの表・考察は当時のもの。engine 列の数値は engine 側が変わらない限り現行値としても読める。

### 計測方法（当時）

`docs/performance/ols.md`「最重要の教訓」「計測方法」・`probit.md`と共通（releaseビルド必須・`tracemalloc`不採用・サブプロセス隔離・スレッド数を1に固定・polars→pandas変換は計測区間外・ウォームアップ1回＋`repeats`回の中央値）。Tobit固有の点は以下。

- **リファレンスが py4etrics（R ではない）**: Tobit の正確性検証の主リファレンスは R `AER::tobit`（＝`survreg`）だが、R は共通ハーネスのインプロセス計測モデル（`fit_once(ctx)` を計測ループ内で呼ぶ）に乗らず、`benchmark_performance.yml` への R 導入も要る。statsmodels にネイティブ Tobit は無い。**py4etrics** は `statsmodels.GenericLikelihoodModel` ベースの Tobit を pure Python でパッケージ化したもので、インプロセス計測できる。係数・σ・対数尤度が engine と ~1e-9 で一致することを実機確認済み（`moderate_censoring`, n=1,000〜100,000, classical/cluster, newton）。正式な数値照合は従来どおり R ベースの `tests/nonlinear/test_tobit_reference.py`（`AER::tobit`）・`test_tobit_crosscheck.py`（`censReg`）が担い、ここでは性能の相対傾向のみを見る。
- **核心の非対称: 解析的微分 vs 数値微分**: engine は Tobit の対数尤度のスコア・ヘッシアンを解析式で Rust 実装する。py4etrics（`GenericLikelihoodModel`）は有限差分でスコア・ヘッシアンを数値近似する。したがって本比較の大きな差は「Rust vs Python」だけでなく「手で導出した解析的微分 vs 汎用の数値微分」の寄与を含む。Tobit のように尤度が閉形式で微分できる手法で解析的実装がどれだけ効くかを示す計測でもある。
- **計測範囲の対称性**: engine は係数・標準誤差と同じ呼び出しで対数尤度・AIC・BIC・全体 Wald 統計量まで常に一括計算する。py4etrics（statsmodels）の統計量は遅延評価（`llf` はアクセス時に `loglike` を再計算、`aic`/`bic` はプロパティ）なので、`_fit_once_py4etrics` は `.fit()` 直後に `llf`/`aic`/`bic` へ明示アクセスして揃える。全体 Wald 検定・限界効果は py4etrics 側が自動計算しないため対称化の対象外（Logit/Probit の `llnull` と同じ整理）。
- **cov_type**: classical と cluster の代表2点。`opg`/`hc0`/`hc1` は省略（Logit/Probit と同じ絞り方）。cluster の疑似グループ数は 50 固定。
- **k 軸は engine 単独**（`PerfAdapter.k_sweep_libraries=("engine",)`）: py4etrics の数値微分ヘッシアンは k に対してコストが崖状に悪化し、k=5 は約1.5秒だが k>=8 で事実上フリーズする（n=10,000 でも実機確認）。k 軸のスケーリング比較にならないため engine のみ回す（n 軸・method 軸では py4etrics を比較対象に使う）。
- **method（オプティマイザ）**: engine・py4etrics とも Newton-Raphson（`method="newton"`）で n/k スイープを回す。加えて `lbfgs` を **method 軸**として代表点1つ（cov_type=classical, k=5, n=100,000）で計測する。**`bfgs` は計測対象外**: engine の Tobit BFGS 経路は n>=10,000 で `MoreThuenteLineSearch: NaN or Inf` により発散する（#292）。解消後に戻す。
- **quasi-Newton の劣化ガード**: `compare_tobit.py` の `check_report`（`_check_method_ratios`）が engine の `lbfgs/newton` 実行時間比を計算し、5x を超えたら job summary に `> [!WARNING]` を出す（CI failure にはしない。実時間の絶対値ではなく同一ジョブ内の比なので共有ランナーの速度差に影響されない。#285 と同系統の劣化の早期検知）。
- **打ち切りシナリオ**: `moderate_censoring`（左打ち切り ~35%、`benchmark/nonlinear/datasets.py` の `_TOBIT_SCENARIO_CONFIG`。Tobit テストの `BASELINE_SCENARIO` と同じ）。潜在回帰の誤差 SD（真の σ）は `_TOBIT_ERROR_SD`。
- **スイープ軸**: n軸（k=5固定。engine/py4etrics 比較は n=1,000〜**100,000**、加えて **engine 単独で n=200,000 / 1,000,000**（`n_sweep_engine_only`、cov_type=classical・newton・seed=42 のみ。n=1,000,000 が Issue #291 の再現点で修正の回帰ガード、n=200,000 は追加のスケーリング点。下記「既知の限界」参照）、k軸（n=10,000固定、k=5・20、engine のみ）、method軸（上記）。

計測環境: devcontainer（12論理コア、シングルスレッド固定）、`repeats=3`、seed=42、scenario=moderate_censoring、release build。n軸 n≤100,000・k軸・method軸は 2026-09-06、engine 単独の n=200,000 / 1,000,000 行は 2026-09-07 のローカル実測（CI の job summary は毎タグ push で更新され、共有ランナーのため数値はぶれる）。

### 結果: n軸（k=5固定）

実行時間（秒、中央値）/ ピークRSS（MB）。method=newton。

#### classical

| n | engine | py4etrics | 比 |
|---|---|---|---|
| 1,000 | 0.0011s / 204MB | 0.2143s / 232MB | ~195x |
| 10,000 | 0.0136s / 207MB | 1.1099s / 236MB | ~82x |
| 100,000 | 0.1538s / 244MB | 12.0527s / 263MB | ~78x |
| 200,000 | 0.2530s / 229MB | -（engine 単独） | - |
| 1,000,000 | 5.9000s / 461MB | -（engine 単独） | - |

engine 単独の 200,000 / 1,000,000 行は Issue #291 の回帰ガード（下記「既知の限界」）。n=1,000,000 は `n_iter=11` で真の β と ~1e-3、σ≈1.001、logLik≈-1.03e6 の正しい最適点へ収束することを確認済み（停滞判定の早期打ち切りではない）。

#### cluster

| n | engine | py4etrics | 比 |
|---|---|---|---|
| 1,000 | 0.0017s / 205MB | 0.3027s / 232MB | ~178x |
| 10,000 | 0.0160s / 208MB | 1.5054s / 237MB | ~94x |
| 100,000 | 0.1635s / 260MB | 13.8665s / 269MB | ~85x |

### 結果: k軸（n=10,000固定、engine のみ）

実行時間（秒、中央値）。method=newton。

| k | engine classical | engine cluster |
|---|---|---|
| 5 | 0.0177s | 0.0153s |
| 20 | 0.0536s | 0.0592s |

### 結果: method軸（cov_type=classical, k=5, n=100,000固定）

実行時間（秒、中央値）。newton は「結果: n軸」classical の n=100,000 行を参照。

| method | engine | py4etrics |
|---|---|---|
| newton | 0.1538s | 12.0527s |
| lbfgs | 0.4935s | 2.5189s |

engine の `lbfgs/newton` 比は約 3.2x で、`_check_method_ratios` の想定上限 5x 以内のため WARNING は出ていない。

### 考察（当時）

数値は 0.01〜0.02 秒台の点で 10〜30% の run 間ばらつきがあり、以下の倍率は概数。傾向（engine が ~80〜200 倍速い・スケーリングは概ね線形）は run をまたいで安定している。

- **classical（newton）**: 全 n で engine が py4etrics より圧倒的に速い（n=100,000 で **約78倍**、0.154s vs 12.05s）。比が n とともに ~195x → ~78x に縮むのは engine が劣化しているのではなく、**py4etrics 側の固定オーバーヘッド**（statsmodels モデル構築で ~0.2秒）が n とともに相対的に薄まるため。engine 自体の n スケーリングは概ね線形（10倍のデータに対し 1,000→10,000 / 10,000→100,000 とも ~11〜12x）。
- **cluster（newton）**: 同傾向（n=100,000 で約85倍）。engine の cluster は 10,000→100,000 で ~10x で classical とほぼ同じ伸び。ピーク RSS は engine 260MB vs py4etrics 269MB で同等。
- **解析的微分 vs 数値微分の寄与**: この ~80〜200倍差の主因は Rust 化だけでなく、engine が Tobit 対数尤度のスコア・ヘッシアンを**解析式**で持つのに対し、py4etrics（`GenericLikelihoodModel`）が**有限差分**で近似すること。k を増やすと py4etrics の数値ヘッシアンは O(k²) 回の対数尤度評価を要し、k=5→8 で約1.5秒→150秒超に崖状に悪化する（k 軸を engine 単独にした理由）。
- **k スケーリング（engine, newton）**: classical k=5→20（k 4倍）で 0.0177s→0.0536s（~3.0x）、cluster も 0.0153s→0.0592s（~3.9x）。k 方向は 4倍のパラメータ増に対し 3〜4倍で、概ね線形。
- **method軸**: engine の lbfgs（0.494s）は newton（0.154s）の **約3.2倍**。probit の #285（newton 比 ~7倍）ほど極端ではないが同系統の遅さで、quasi-Newton 実装に改善余地がある。py4etrics の lbfgs（2.52s）は自身の newton（12.05s）より速い（数値ヘッシアンが不要なため）。**bfgs は engine が n>=10,000 で発散する（#292）ため計測対象外**。
- **改善余地**: engine の絶対性能は n=100,000 で 0.15〜0.16秒と実用上問題ない。newton の大標本での Hessian 特異（#291）は `d797f9b` / `5b79ffe`（`FaerNewton` の停滞収束判定）で解消済みで、engine 単独 n=1,000,000（seed=42）の行がその回帰ガード（下記「既知の限界」に限界つき）。継続課題は quasi-Newton（lbfgs ~3.2x・bfgs 発散 #292）のみ。cluster 経路は classical とほぼ同じ伸びで、現時点で特段の懸念はない。

### 既知の限界（当時）

- **engine vs py4etrics の比較は n=100,000 まで**: py4etrics は数値微分ゆえ大 n で極端に遅い（newton・n=100,000・k=5 で約13秒。engine は約0.14秒）。n=1,000,000 は分オーダーで、両者の比較として非現実的。
- **engine 単独 n=200,000 / 1,000,000 の位置づけ**（`n_sweep_engine_only`、cov_type=classical・newton・seed=42）: 現行の「n 軸の大標本ガード」節に移設。
- **k軸が engine 単独・k は 20 まで**: py4etrics の数値微分ヘッシアンが k>=8 で破綻するため k 軸は engine のみ（「計測方法（当時）」参照）。
- **method軸に bfgs を含まない**: engine の Tobit BFGS 経路が n>=10,000 で発散する（**Issue #292**）。#292 解消後に `compare_tobit.py` の `extra_methods` へ戻す。
- その他は `ols.md`「既知の限界」と共通。特に **engineのマルチスレッド線形代数が多コア機・負荷下で不安定になる問題**（#283）のため、本計測はengine・py4etricsとも1スレッドに固定しており、数値は「シングルスレッドでの計算コア効率」である。
