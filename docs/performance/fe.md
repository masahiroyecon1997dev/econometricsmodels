# FE: パフォーマンス比較（linearmodels）

`FE(...).fit()`（Rust engine + PyO3）とPython製リファレンス実装 linearmodels の実行時間・メモリ使用量比較の記録。CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けることが目的。

再実行可能なスクリプトは`performance/compare_fe.py`（手法非依存の計測ハーネス`performance/_perf_harness.py` ＋ FE固有アダプタ、コミット対象）。生の計測結果JSONはコミットしない（`.gitignore`の`docs/performance/results/*.json`参照）。

比較対象は linearmodels 単体（README「Verification accuracy」表の primary reference。`benchmark/panel/references/linearmodels_ref.py`と同じ主リファレンス）。`linearmodels.panel.PanelOLS`（`entity_effects=True`、2-wayなら`time_effects=True`）に対応させる。

## 計測方法

`docs/performance/ols.md`「最重要の教訓」「計測方法」と共通（releaseビルド必須・`tracemalloc`不採用・サブプロセス隔離・スレッド数を1に固定・ウォームアップ1回＋`repeats`回の中央値）。FE固有の点は以下。

- **N×T分解（サンプルサイズ軸のスイープ方針）**: パネルはサンプルサイズがentity数(N)×時点数(T)の2軸に分解されるが、ハーネスの`build_dataframe(n, k, seed)`は単一の`n`しか持たない。**T（時点数）を`6`に固定し、Nをスイープする**（ミクロパネル——企業・個人パネルでN大・T小が典型——を想定した設計。`benchmark/panel/datasets.py`のbaselineシナリオの既定値と同値）。`n_entities = n // 6`の整数除算により、実際の観測数は要求した`n`と若干ずれる（例: n=1,000 → 実際は996観測）。
- **DGP**: `generate_fe_dataset("baseline", n_entities=n//6, n_periods=6, k=k, seed=seed)`。
- **cov_type**: classicalとhac（Driscoll-Kraay、bartlett kernel）の代表2点。classical/hc1/hc2/hc3/cluster/hacをn_entities=20,000・n_periods=6・k=5で実測した結果、hacが最重量（中央値91.6ms、classicalの82.3msに対し+11%）だったため、OLS/WLS/IVと同じ組み合わせを採用した。ただしT=6・バンド幅2という小さい時点数ではDK計算自体のコストがOLS本体に対して無視できる規模で、n軸スイープ（下記「結果: n軸」）ではclassicalとhacの差がほぼ測定誤差の範囲に収まる（後述「考察」参照）。DKのバンド幅は時点数`T`ベース（`hac_auto_lag(6)=2`）であり、ハーネスの`ctx.hac_lags`（総観測数`n`ベース、Newey-West用）とは基準が異なるため使わず、モジュール定数として別途計算しengine・linearmodels双方に渡す。
- **2-way固定効果（method軸の流用）**: 2-way FE（entity+time）はIV/Logit/Probitの`extra_methods`仕組みを流用し、代表点1つ（cov_type=classical・k=5・n=1,000,000）だけ追加計測する（`default_method="one_way"`, `extra_methods=("two_way",)`）。
- **k軸はclassicalのみ**（`k_sweep_cov_types=("classical",)`）: k=20・hacをn_periods=6で実測するとF検定用の共分散行列の部分行列がほぼ特異になり`ComputationError`で失敗する（T=6に対しk=20は次元過多。詳細は下記「既知の限界」）。
- **MultiIndex構築は計測区間の外**: `linearmodels.panel.PanelOLS`は呼び出し前に`MultiIndex(entity, time)`の構築が必須だが、engineはentity/timeをプレーン列として受け取るためこの手順が不要。MultiIndex構築も実務上は一度きりのデータ準備処理であるため、`PerfAdapter.build_pandas_df`（`_perf_harness.py`への拡張）でウォームアップ前に1回だけ構築し計測ループの外に置く（polars→pandas変換と同じ扱い）。
- **計測範囲の対称性**: engine（`FeEstimator::fit`）は係数・標準誤差と同じ`.fit()`の中でパネル固有R²（within/between/overall）・F統計量まで常に一括計算する。linearmodelsの`PanelResults`は遅延評価プロパティのため、`.fit()`直後に`params`/`std_errors`/`tstats`/`pvalues`/`rsquared_within`/`rsquared_between`/`rsquared_overall`/`f_statistic_robust.stat`へ明示アクセスして確定させる（`f_statistic_robust`はcov_typeに連動する版。`f_statistic`は常にhomoskedastic固定でengineの値と対応しない）。`aic`/`bic`はlinearmodelsが提供しないため対称性を取る対象に含めない。
- **スイープ軸**: n軸（k=5固定、n=1,000〜1,000,000）、k軸（n=10,000固定、k=5・20、classicalのみ）、method軸（下記）。

## 結果: n軸（k=5固定）

実行時間（秒、中央値）/ ピークRSS（MB）。devcontainer（12論理コア、シングルスレッド固定）、`repeats=3`、method=one_way。

| n | engine | linearmodels |
|---|---|---|
| **classical** | | |
| 1,000 | 0.0006 / 212 | 0.0219 / 217 |
| 10,000 | 0.0059 / 213 | 0.0592 / 229 |
| 100,000 | 0.1016 / 250 | 0.3334 / 304 |
| 1,000,000 | 1.5764 / 564 | 3.7959 / 933 |
| **hac** | | |
| 1,000 | 0.0009 / 211 | 0.0290 / 218 |
| 10,000 | 0.0075 / 214 | 0.0618 / 229 |
| 100,000 | 0.0993 / 255 | 0.3789 / 305 |
| 1,000,000 | 1.5815 / 620 | 3.3175 / 931 |

## 結果: k軸（n=10,000固定、classicalのみ）

実行時間（秒、中央値）。method=one_way。

| k | engine | linearmodels |
|---|---|---|
| 5 | 0.0083 | 0.0502 |
| 20 | 0.0213 | 0.0935 |

## 結果: method軸（cov_type=classical, k=5, n=1,000,000固定）

実行時間（秒、中央値）。one_way は「結果: n軸」classicalのn=1,000,000行（engine 1.5764 / linearmodels 3.7959）を参照。

| method | engine | linearmodels |
|---|---|---|
| two_way | 2.0579 | 4.9157 |

## 考察

- **classical**: 全nでengineがlinearmodelsより高速（n=1,000,000で約2.4倍、1.58s vs 3.80s、n=100,000で約3.3倍）。
- **hac（Driscoll-Kraay）**: engineが全nで約2.1〜4.4倍速い。engine自身のclassical→hacの増分はごくわずか（n=1,000,000で1.5764s→1.5815s）——T=6・バンド幅2という小さい時点数では、DK計算（バンド幅内の時点ペアの和）自体のコストがOLS本体の計算に対して無視できる規模のため。linearmodels側も同様に増分が小さい（3.7959s→3.3175sはむしろ減少しており、repeats=3の測定ノイズの範囲内と考えられる）。
- **2-way固定効果**: one-way→two-wayでengineは約1.3倍（1.5764s→2.0579s）、linearmodelsも約1.3倍（3.7959s→4.9157s）と同程度の相対増加。engineの優位（約2.4倍）はtwo-wayでも維持される。
- **メモリはengineが一貫して軽い**: n=1,000,000でengine約564〜655MBに対しlinearmodels約931〜1076MB（約1.6〜1.7倍）。
- **kスケーリング**: classical k=5→20でengine約2.6倍（0.0083s→0.0213s）、linearmodels約1.9倍（0.0502s→0.0935s）。engineのk方向の伸びがやや急な傾向はOLS/IVと同様。

## 既知の限界

- **k=20・hacはengineで`ComputationError`になるため性能比較から除外**: T=6（時点数）に対しk=20（説明変数数）は次元過多で、F検定用の共分散行列の部分行列がほぼ特異になり「coefficient covariance submatrix for the F-test is near-singular」で失敗する（DKの`S`行列はバンド幅内の時点ペアからの寄与の和のため、実効ランクが時点数`T`に制約される）。RE（`re.md`）は同じk=20・hacで問題なく成功する——RE自身のF統計量はFEの部分行列反転とは異なる定義（変換済みyの単純平均を基準にしたSST/SSR比較）を使うため、この特異性の影響を受けない。k軸はclassicalのみで計測している。
- その他は`ols.md`「既知の限界」と共通。計測は開発コンテナ上の1回のスイープ（`repeats=3`の中央値）で、環境ノイズを排除しきれていない。

## 再現方法

```bash
uv run maturin develop --release
uv run python -m performance.compare_fe --repeats 3 \
    --output docs/performance/results/fe.json
uv run python -m performance.render_performance_summary \
    docs/performance/results/fe.json
```

## 今後の検討事項

- **T（時点数）を大きくした場合のDKスケーリング**: 本計測はT=6固定（ミクロパネル想定）のため、DKのバンド幅・時点方向の計算量が支配的になる大T・小Nのケース（マクロパネル）でのスケーリングは未計測。必要になった時点で別途T軸のスイープを検討する。
- **releaseビルドでの再計測が前提**: 改善見込みの見積もりは、debugビルドの数値（誤り）ではなく本ドキュメントのreleaseビルド数値を基準にすること。
