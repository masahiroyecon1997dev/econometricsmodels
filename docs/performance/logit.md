# Logit: パフォーマンス比較（statsmodels）

`Logit(...).fit()`（Rust engine + PyO3）とリファレンス実装 statsmodels（`smf.logit`）の実行時間・メモリ使用量比較の記録。CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けることが目的。

再実行可能なスクリプトは`performance/compare_logit.py`（手法非依存の計測ハーネス`performance/_perf_harness.py` ＋ Logit固有アダプタ、コミット対象）。生の計測結果JSONはコミットしない（`.gitignore`の`docs/performance/results/*.json`参照）。

> **役割分担**: 計測結果の表と性能特性の解釈は、公開ページ（英語。CIの計測値から生成）の[Performance](../guide/performance.md)・[Performance results](../guide/performance-results.md)に置く。このノートは計測方法論・設計判断・既知の限界・今後の検討を記録する。

## 計測方法

`docs/performance/ols.md`「最重要の教訓」「計測方法」と共通（releaseビルド必須・`tracemalloc`不採用・サブプロセス隔離・スレッド数を1に固定・polars→pandas変換は計測区間外・ウォームアップ1回＋`repeats`回の中央値）。Logit固有の点は以下。

- **計測範囲の対称性**: engine は係数・標準誤差と同じ呼び出しで、対数尤度・**切片のみモデルの対数尤度**・尤度比統計量・そのp値・McFadden擬似R²・AIC・BIC まで常に一括計算する。statsmodels はこれらを遅延評価（`cached_value`）にしており、特に `llnull`（切片のみモデルの対数尤度、`llr`/`prsquared` が依存）はアクセス時に**切片のみ Logit を別途フィットする**。`_fit_once_statsmodels` は `.fit()` 直後に `llf`/`llnull`/`llr`/`llr_pvalue`/`prsquared`/`aic`/`bic` へ明示アクセスし、engine と同じ処理範囲で計測する（この対称化により statsmodels 側の計測時間は遅延評価アクセスなしの約2〜3倍になる）。
- **cov_type**: classical と cluster の代表2点。Logit/Probit は OLS/WLS と違い HAC を持たない。classical/hc0/cluster を n=100,000, k=5 で軽く実測したところ cluster が最重だった。cluster の疑似グループ数は 50 固定。
- **`opg` は計測対象外**: statsmodels の discrete model（`Logit.fit`）は `opg` を `cov_type` 引数としてネイティブに受け付けず、`score_obs` からの numpy 手計算になる（`benchmark/nonlinear/references/statsmodels_ref.py`）。engine のネイティブ OPG との比較は「計測対象の処理範囲を対称に揃える」方針に反するため除外する。
- **method（オプティマイザ）**: engine・statsmodels とも Newton-Raphson（engineは`solver="newton"`、比較対象は`method="newton"`）で n/k スイープを回す。加えて `bfgs`/`lbfgs` を **method 軸**として代表点1つ（cov_type=classical, k=5, n=1,000,000）で計測する（正確性検証〈`test_logit_reference.py`〉も newton 主軸、bfgs/lbfgs は代表のみ、という絞り方に合わせる）。
- **スイープ軸**: n軸（k=5固定、n=1,000〜1,000,000）、k軸（n=10,000固定、k=5・20）、method軸（下記）。

## 考察（結果表の外に残す、機構・経緯の記録）

- **計測範囲の対称化が効く手法**: statsmodels の `llnull`（切片のみモデル）へのアクセスを計測範囲に含めると、statsmodels の実行時間が遅延評価アクセスなしの約2〜3倍に増える（切片のみ Logit の再フィットが走るため）。engine はこれを常に一括計算しているので、対称に揃えて初めて公平な比較になる（揃えないと engine に不利な非対称計測になっていた）。
- **method軸: engine の BFGS/L-BFGS が遅かった（経緯、解消済み）**: 初期の計測（devcontainer、n=1,000,000、classical、k=5）では newton（0.65s）に対し bfgs は 11.21s、lbfgs は 23.91s と大幅に遅かった。同じ method の statsmodels（scipy）は bfgs 1.47s・lbfgs 1.44s で newton とほぼ同じ。engine の quasi-Newton 実装（ステップ制御・収束判定・逆Hessian近似の更新）に改善余地があった。**既定の newton は十分速いため実用上の実害は「newton 以外を選ぶと遅い」という選択上の注意に留まる**。
  - **LPMベース warm start への変更（2026-09-08）の影響と解消（2026-10-11）**: ゼロベクトル初期値から warm start（`ols_based_initial_params`）へ変更した際の before/after 実測（`git stash` A/B、cov_type=classical・k=5・n=1,000,000・`repeats=3` のスポット計測）で、**newton は −10〜14% 高速化**した一方、**bfgs は 9.81s → 13.09s（+33%）と悪化**した（この値は後述の`tol`正規化・停滞検出の導入前の旧実装のもの）。その後の改善で bfgs の絶対値は約0.6sまで縮んだが、warm start では反復数だけが増える相互作用は残っていた（n=100,000で8→13反復。line searchの評価回数は増えておらず原因ではない）。真因は、1回目のステップ長が`min(1,1/‖g₀‖)`により初期点によらず常に`‖s₀‖=1`になること（ゼロ起点には適切だが、最適点に近いwarm startでは過大で、最初のsecantペアが遠い領域の曲率を拾い、γが真の逆曲率から外れる）。**warm start点で`H`を1回評価して`H⁻¹`を初期逆Hessianにする**ことで解消した（Logit n=100,000は13→6反復、公式ハーネスでbfgs 0.052s・newton 0.055s、n=1,000,000はbfgs 0.565s・newton 0.573s）。`H`が正定値でない場合は従来の初期化に戻す（詳細は`docs/spec/nonlinear-common.md`のBFGS節）。1回目のステップ長を`c/‖g₀‖`の`c`で振る案は、ケースごとに最良の`c`が異なり（反復数は非単調）、単一の値では全ケースを改善できなかったため不採用。
  - **`FaerBfgs`自前実装への置き換え（2026-09-12）の影響**: `argmin`組み込みBFGS（`argmin::solver::quasinewton::BFGS`）を、Nocedal & Wright *Numerical Optimization* 6.1節のself-scaling初期化（1回目の反復でline searchが実際に受理したステップから`γ=(y₀ᵀs₀)/(y₀ᵀy₀)`を計算し初期逆Hessianをスケーリングする、`argmin`自身にはコメントアウトされ機能していない形でしか存在しない）と、1回目の反復専用のline search初期ステップ幅調整（`min(1,1/‖g₀‖)`）を組み込んだ自前実装`FaerBfgs`に置き換えた。公式ハーネス（`performance.compare_logit --worker`、cov_type=classical・k=5・n=1,000,000・`repeats=3`の中央値、devcontainer・1スレッド固定）での再計測: **bfgs 14.97s**（反復回数22→17に減少）・参考として**newton 0.74s**・**lbfgs 18.23s**（`Method::Lbfgs`は今回変更していないため warm start 変更後の実測値の再確認に相当）。`argmin`組み込みBFGS（warm start 変更後、この対応着手前の状態）はアドホック計測で約16.0s（`repeats=5`の中央値）だったため、**bfgs はこの対応で約6〜7%の改善**にとどまり、newton・statsmodels（bfgs 1.47s）とは依然として1桁以上の差が残る。**LBFGSは対象外**: `argmin`のLBFGS実装は`s`/`y`履歴・初期`γ`を外部から注入する公開APIが無く、同じ手法を適用できないため今回は見送った（`engine/src/nonlinear/CLAUDE.md`参照）。
  - **`tol`の観測数`n`正規化（2026-09-12）の続報**: 上記`FaerBfgs`自前実装後もbfgsが遅い根本原因は、`tol`の意味論そのもの（総和勾配に対する絶対閾値で`n`スケールしない、Tobit/Probitと同根）にあると判明。`bfgs`/`lbfgs`のみ`tol`を観測数`n`で正規化した「観測あたり平均勾配」基準に変更し（`newton`は絶対閾値のまま）、既定値も`method`依存にした（`newton=1e-6`・`bfgs`/`lbfgs`=`1e-8`、詳細は`docs/spec/logit-spec.md`3.2節）。公式ハーネスでの再計測（同条件）: **bfgs 0.73s**（14.97sから約20倍改善、statsmodels（1.90s）より高速化）。**lbfgs 19.73s**（18.23sからほぼ不変）——同じ正規化後の実効閾値でも`bfgs`（6反復）と`lbfgs`（17反復）で収束に必要な反復数が大きく異なるため、`bfgs`に効いた既定値の正規化だけでは`lbfgs`は速くならない。`newton`にも同じ正規化を適用する案は検証したが、`near_separation`等の`RTOL=1e-8`精度検証テストが28件失敗したため不採用（`newton`は非正規化のまま維持）。`lbfgs`固有のより緩い既定値の検討は別Issueとする。
  - **`FaerLbfgs`自前実装への置き換え（本項目の結論）**: `argmin`組み込みLBFGS（`argmin::solver::quasinewton::LBFGS`）を`FaerBfgs`と同型のパターンで自前実装`FaerLbfgs`に置き換えた。真因は`FaerBfgs`と全く同じ「1回目の反復専用のline search初期ステップ幅`min(1,1/‖g₀‖)`」という制御点で、argmin組み込み`LBFGS`は2回目以降のself-scaling（`γ`計算）は内部で行っていたが、この制御点だけが公開APIに無かった。公式ハーネスでの再計測（同条件）: **lbfgs 0.62s**（19.73sから**約32倍改善**）。`bfgs`（0.518s）・`newton`（0.539s）とほぼ同オーダーまで縮まり、statsmodels（bfgs 1.47s・lbfgs 1.44s）も上回る水準になった。詳細（two-loop recursionの実装・line searchへの明示的`max_iters`追加等）は`engine/src/nonlinear/CLAUDE.md`「`method="lbfgs"`はargmin組み込みソルバーを使わず自前実装`FaerLbfgs`」参照。
  - **line search停滞検出の導入（2026-09-26）**: Probitで残っていたbfgs/lbfgsの遅さ（収束点近傍のline searchの空回り、[`probit.md`](./probit.md)「考察」参照）への対処。Logitは勾配基準がたまたま丸め誤差の床より先に満たされていたため影響は小さく、再計測（同条件）でbfgs 0.60s・lbfgs 0.62s（statsmodels bfgs 1.61s・lbfgs 1.52s）と実質不変。

## 既知の限界

`ols.md`「既知の限界」と共通。特に **engineのマルチスレッド線形代数が多コア機・負荷下で不安定になる問題**のため、本計測はengine・statsmodelsとも1スレッドに固定しており、数値は「シングルスレッドでの計算コア効率」である。計測は開発コンテナ上の1回のスイープ（`repeats=3`の中央値）で、環境ノイズを排除しきれていない。

## 再現方法

```bash
uv run maturin develop --release
uv run python -m performance.compare_logit --repeats 3 \
    --output docs/performance/results/logit.json
uv run python -m performance.render_performance_summary \
    docs/performance/results/logit.json
```

## 今後の検討事項

- **engineのL-BFGSが遅い問題は解消済み**: `bfgs`は自前実装`FaerBfgs`＋`tol`の観測数`n`正規化により newton・statsmodels と同オーダーまで解決したが、`lbfgs`は同じ正規化を適用してもほぼ改善しなかった（同じ実効閾値でも収束に必要な反復数が`bfgs`よりずっと多いため）。`Method::Lbfgs`をargmin組み込みLBFGSから自前実装`FaerLbfgs`に置き換え、`FaerBfgs`と同型のself-scaling初期化を適用したところ19.73s→0.62s（約32倍）に改善し解消した（詳細は上記「method軸」の考察参照）。
- **engineのマルチスレッド線形代数の不安定性**: OLSと共通。
- **releaseビルドでの再計測が前提**: 改善見込みの見積もりは、debugビルドの数値（誤り）ではなく本ドキュメントのreleaseビルド数値を基準にすること。
