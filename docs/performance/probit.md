# Probit: パフォーマンス比較（statsmodels）

`Probit(...).fit()`（Rust engine + PyO3）とリファレンス実装 statsmodels（`smf.probit`）の実行時間・メモリ使用量比較の記録。CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けることが目的。

再実行可能なスクリプトは`performance/compare_probit.py`（手法非依存の計測ハーネス`performance/_perf_harness.py` ＋ Probit固有アダプタ、コミット対象）。生の計測結果JSONはコミットしない（`.gitignore`の`docs/performance/results/*.json`参照）。

## 計測方法

`docs/performance/ols.md`「最重要の教訓」「計測方法」・`logit.md`と共通（releaseビルド必須・`tracemalloc`不採用・サブプロセス隔離・スレッド数を1に固定・polars→pandas変換は計測区間外・ウォームアップ1回＋`repeats`回の中央値）。Probit固有の点は以下。

- **計測範囲の対称性**: engine は係数・標準誤差と同じ呼び出しで、対数尤度・**切片のみモデルの対数尤度**・尤度比統計量・そのp値・McFadden擬似R²・AIC・BIC まで常に一括計算する。statsmodels の `ProbitResults` はこれらを遅延評価にしており、特に `llnull` はアクセス時に**切片のみ Probit を別途フィットする**。`_fit_once_statsmodels` は `.fit()` 直後に `llf`/`llnull`/`llr`/`llr_pvalue`/`prsquared`/`aic`/`bic` へ明示アクセスし、engine と同じ処理範囲で計測する。
- **cov_type**: classical と cluster の代表2点。classical/hc0/cluster を n=100,000, k=5 で軽く実測したところ、Logit と同じく cluster が最重だった。cluster の疑似グループ数は 50 固定。`opg` は statsmodels の discrete model がネイティブ非対応（`score_obs` からの手計算になる）で対称計測できないため対象外。
- **method（オプティマイザ）**: engine・statsmodels とも Newton-Raphson（`method="newton"`）で n/k スイープを回す。加えて `bfgs`/`lbfgs` を **method 軸**として代表点1つ（cov_type=classical, k=5, n=100,000）で計測する。
- **スイープ軸**: n軸（k=5固定、n=1,000〜**100,000**、全library・cov_type）＋`n_sweep_engine_only=(200_000, 1_000_000)`（classical・engine単独）、k軸（n=10,000固定、k=5・20）、method軸（下記）。**下表の全library・cov_typeでの一括計測には1,000,000を含まない**（statsmodels側も含めたフル計測はコスト上まだ実施していない）。旧来 `generate_binary_choice_dataset("baseline", link="probit")` はk=5のときn≥500,000でΦ(Xβ)の飽和によりengineのProbit Hessianが数値的に特異化しfitが失敗する問題があったが、warm start統一・`FaerNewton`停滞収束判定により解消済み（2026-09-12実測確認、`generate_binary_choice_dataset("baseline", link="probit", n=1_000_000, k=5, seed=42)`でstatsmodelsと対数尤度・係数とも相対誤差1e-11で一致）。回帰検知は`n_sweep_engine_only`が担う（詳細は`compare_probit.py`docstring「n軸の大標本点」参照）。

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
- **計測範囲の対称化が効く**: Logit と同じく、statsmodels の `llnull`（切片のみモデル）へのアクセスを計測範囲に含めると切片のみ Probit の再フィットが走り、statsmodels の実行時間が増える。engine はこれを常に一括計算しているので、揃えて初めて公平な比較になる。
- **method軸: engine の BFGS/L-BFGS が遅い**: newton（engine 0.12s）に対し bfgs は **0.91s**（約7.6倍）、lbfgs は **0.82s**（約6.9倍）。同じ method の statsmodels（scipy）は bfgs 0.18s・lbfgs 0.16s で newton とほぼ同じ。Logit ほど極端ではないが同傾向で、engine の quasi-Newton 実装に改善余地がある。既定の newton は十分速いため実用上の実害は「newton 以外を選ぶと遅い」という選択上の注意に留まる。
  - **`tol`の観測数`n`正規化（2026-09-12）の影響**: `bfgs`/`lbfgs`の`tol`を`n`で正規化する変更（詳細は[`logit.md`](./logit.md)「考察」参照）はProbitにも共通で適用される。単体ワーカーでのスポット計測（`performance.compare_probit --worker`、cov_type=classical・k=5・n=100,000、`repeats=3`の中央値）: **bfgs 0.24s**（旧0.91sから約3.8倍改善）・**lbfgs 0.27s**（旧0.82sから約3.0倍改善）・参考として**newton 0.17s**（既定`tol`は`newton`のみ変更なしのため実質不変）。この`n`（100,000）ではlogit（n=1,000,000）と異なりlbfgsにも改善が見られた——lbfgsの改善幅が`n`に依存する理由は未調査（別Issue）。上表（旧`FaerBfgs`自前実装のみの状態、正規化前）は当時の記録として残す。
  - **Issue #343（`FaerLbfgs`自前実装への置き換え、後退を確認）**: `Method::Lbfgs`をargmin組み込みLBFGSから自前実装`FaerLbfgs`（`FaerBfgs`と同型のself-scaling初期化を適用、詳細は`engine/src/nonlinear/CLAUDE.md`参照）に置き換えたところ、logit（n=1,000,000）・tobit（n=100,000）は大幅改善した一方、**Probit（n=100,000）は`lbfgs`が0.27s→0.55sへ後退**した（単体ワーカーでの実測、同条件）。原因は`FaerLbfgs::init`の1回目line search初期ステップ幅`alpha0=min(1,1/‖g₀‖)`にあると切り分け済み: `g₀`（n個の観測スコアの総和）のノルムが`n=100,000`で`≈1.3e4`と大きいため`alpha0≈7.5e-5`という過度に小さい初期ステップになり、反復回数が旧実装比7→12へ増加する。`n_obs`で正規化する対策案（`min(1,n_obs/‖g₀‖)`）も試したが、Issue #344のTobit退化ケースが再発する副作用が判明し不採用とした。Issue #344の本来の目的（Tobitの暴走ケース解消）を優先し、この後退は既知の未解決課題として次セッションに持ち越した（ユーザー確認済み、詳細は`engine/src/nonlinear/CLAUDE.md`「`FaerLbfgs`」セクション参照）。
- **kスケーリング（newton）**: classical k=5→20 で engine 約3.0倍 / statsmodels 約2.2倍。Logit ほどではないが engine の伸びがやや急。

## 既知の限界

- **全library・cov_typeでの n軸一括計測は 100,000 まで**: 旧来 baseline DGP・k=5 では n≥500,000 で engine の Probit fit が Hessian 特異エラーになっていた問題は解消済み（上記「計測方法」参照）。回帰検知用の`n_sweep_engine_only`（classical・engine単独、n=200,000/1,000,000）は追加済みだが、statsmodelsも含めた大規模n（1,000,000）でのフル計測値はコスト上まだ本ドキュメントに反映していない。他手法（OLS/WLS/Logit）は n=1,000,000 まで計測している。
- その他は `ols.md`「既知の限界」と共通。特に **engineのマルチスレッド線形代数が多コア機・負荷下で不安定になる問題**のため、本計測はengine・statsmodelsとも1スレッドに固定しており、数値は「シングルスレッドでの計算コア効率」である。

## 再現方法

```bash
uv run maturin develop --release
uv run python -m performance.compare_probit --repeats 3 \
    --output docs/performance/results/probit.json
uv run python -m performance.render_performance_summary \
    docs/performance/results/probit.json
```

## 今後の検討事項

- **engineのProbitのHessian特異化（解消済み）**: 上記「計測方法」参照。statsmodelsも含めたn=1,000,000でのフル計測は次回のreleaseビルド再計測時に n軸へ追加する。
- **Hessianの重み計算のU_CLAMP/z不整合（Issue #316、未解決）**: #284の調査時に発見。`ProbitProblem::hessian`の`w=λᵢ(λᵢ+zᵢ)`がクランプ済み`λᵢ`と生の`zᵢ`を混在させており、悪条件データ・BFGS/L-BFGS経路では理論上まだ負の重みを生みうる（`docs/spec/probit-spec.md`4章参照）。
- **engineのL-BFGSが`n=100,000`で旧実装比後退している（Issue #343、未解決）**: 上記「method軸」の考察参照。`FaerLbfgs::init`の初期ステップ幅`alpha0=min(1,1/‖g₀‖)`が`n`が大きいと過度に小さくなる問題で、`n_obs`正規化はTobitの別ケースを壊すため不採用。より良い初期ステップ幅の設計を次セッションで検討する。
- **engineのBFGSが遅い**: Logit と共通。newton・statsmodels の同 method 比で遅い。
- **engineのマルチスレッド線形代数の不安定性**: OLSと共通。
- **releaseビルドでの再計測が前提**: 改善見込みの見積もりは、debugビルドの数値（誤り）ではなく本ドキュメントのreleaseビルド数値を基準にすること。
