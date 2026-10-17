# FE: パフォーマンス比較（linearmodels）

`FE(...).fit()`（Rust engine + PyO3）とPython製リファレンス実装 linearmodels の実行時間・メモリ使用量比較の記録。CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けることが目的。

再実行可能なスクリプトは`performance/compare_fe.py`（手法非依存の計測ハーネス`performance/_perf_harness.py` ＋ FE固有アダプタ、コミット対象）。生の計測結果JSONはコミットしない（`.gitignore`の`docs/performance/results/*.json`参照）。

比較対象は linearmodels 単体（[検証ページ](../guide/verification.md)の primary reference。`benchmark/panel/references/linearmodels_ref.py`と同じ主リファレンス）。`linearmodels.panel.PanelOLS`（`entity_effects=True`、2-wayなら`time_effects=True`）に対応させる。

> **役割分担**: 計測結果の表と性能特性の解釈は、公開ページ（英語。CIの計測値から生成）の[Performance](../guide/performance.md)・[Performance results](../guide/performance-results.md)に置く。このノートは計測方法論・設計判断・既知の限界・今後の検討を記録する。

## 計測方法

`docs/performance/ols.md`「最重要の教訓」「計測方法」と共通（releaseビルド必須・`tracemalloc`不採用・サブプロセス隔離・スレッド数を1に固定・ウォームアップ1回＋`repeats`回の中央値）。FE固有の点は以下。

- **N×T分解（サンプルサイズ軸のスイープ方針）**: パネルはサンプルサイズがentity数(N)×時点数(T)の2軸に分解されるが、ハーネスの`build_dataframe(n, k, seed)`は単一の`n`しか持たない。**T（時点数）を`6`に固定し、Nをスイープする**（ミクロパネル——企業・個人パネルでN大・T小が典型——を想定した設計。`benchmark/panel/datasets.py`のbaselineシナリオの既定値と同値）。`n_entities = n // 6`の整数除算により、実際の観測数は要求した`n`と若干ずれる（例: n=1,000 → 実際は996観測）。
- **DGP**: `generate_fe_dataset("baseline", n_entities=n//6, n_periods=6, k=k, seed=seed)`。
- **cov_type**: n軸・k軸は既定の classical のみ。標準誤差の種類は **cov軸**（n=100,000・k=5）で全て測る: classical・hc1・hc2・hc3・cluster・dk（`hc0` は FE が対応しない）。`hc2`/`hc3` は linearmodels に対応が無いため engine 単独。cluster は entity クラスター（G=N=n÷T、データは entity 順）の1ケースで、G・並びの 2×2 は使わない（パネルのクラスターは entity で決まり、G を変えると N×T の構成が変わるため）。linearmodels 側は `cluster_entity=True`。以前は classical と hac(dk) の2点を実装時の実測で手書き選定していた（dk は T=6・バンド幅2という小さい時点数では DK 計算自体のコストが OLS 本体に対して無視できる規模で、classical との差はほぼ測定誤差の範囲だった）。DKのバンド幅は時点数`T`ベース（`hac_auto_lag(6)=2`）であり、ハーネスの`ctx.hac_lags`（総観測数`n`ベース）は使わない。
- **2-way固定効果（method軸の流用）**: 2-way FE（entity+time）はIV/Logit/Probitの`extra_methods`仕組みを流用し、代表点1つ（既定の classical・k=5・n=1,000,000）だけ追加計測する（`default_method="one_way"`, `extra_methods=("two_way",)`）。
- **k軸は既定の classical のみ**: k=20・dkはn_periods=6に対し次元過多で、engineが`ValidationError`で拒否する（詳細は下記「既知の限界」）。cov軸の dk は k=5 < T=6 の境界ちょうどに依存する。
- **MultiIndex構築は計測区間の外**: `linearmodels.panel.PanelOLS`は呼び出し前に`MultiIndex(entity, time)`の構築が必須だが、engineはentity/timeをプレーン列として受け取るためこの手順が不要。MultiIndex構築も実務上は一度きりのデータ準備処理であるため、`PerfAdapter.build_pandas_df`（`_perf_harness.py`への拡張）でウォームアップ前に1回だけ構築し計測ループの外に置く（polars→pandas変換と同じ扱い）。
- **計測範囲の対称性**: engine（`FeEstimator::fit`）は係数・標準誤差と同じ`.fit()`の中でパネル固有R²（within/between/overall）・F統計量まで常に一括計算する。linearmodelsの`PanelResults`は遅延評価プロパティのため、`.fit()`直後に`params`/`std_errors`/`tstats`/`pvalues`/`rsquared_within`/`rsquared_between`/`rsquared_overall`/`f_statistic_robust.stat`へ明示アクセスして確定させる（`f_statistic_robust`はcov_typeに連動する版。`f_statistic`は常にhomoskedastic固定でengineの値と対応しない）。`aic`/`bic`はlinearmodelsが提供しないため対称性を取る対象に含めない。
- **スイープ軸**: n軸（k=5固定、n=10,000 / 100,000 / 1,000,000）、k軸（n=10,000固定、k=5・20）、cov軸（上記）、method軸（下記）。n軸・k軸は既定の classical のみ。

## 既知の限界

- **k=20・dkはengineで`ValidationError`になるため k=5 でのみ測る**: DKの`S`行列は時点ごとのスコアの外積（とそのラグ項）の和で、正規方程式によりスコアの和がゼロになるためrankが`T-1`以下。T=6（時点数）に対しk=20（説明変数数）ではF検定の部分行列が構造的に特異になり、engineは`fit()`冒頭で`PanelError::InsufficientDkPeriodsForInference`（`ValidationError`）を返す。RE（`re.md`）もハウスマン検定が同じ制約を受ける（`re.md`「既知の限界」参照）。k軸は classical のみ、cov軸の dk は k=5 で測っている。
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
