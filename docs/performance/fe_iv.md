# FEIV: パフォーマンス計測

`FEIV(...).fit()`（Rust engine + PyO3）の実行時間・ピークメモリの記録。CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けることが目的。

他手法ページ（`ols.md` 等）と違い **engine 単独の計測**で、リファレンス実装との相対比較は行わない（`tobit.md` と同じ）。engine の FEIV が軸ごとに「時間がかかりすぎていないか」（絶対値・スケーリングの推移）を観察する。正確性の数値照合は `tests/panel/test_fe_iv_*.py`（fixest・linearmodels・plm・ivDiag）が担う。

再実行可能なスクリプトは `performance/compare_fe_iv.py`（手法非依存の計測ハーネス `performance/_perf_harness.py` ＋ FEIV 固有アダプタ、コミット対象）。生の計測結果 JSON はコミットしない（`.gitignore` の `docs/performance/results/*.json` 参照）。

> **役割分担**: 計測結果の表と性能特性の解釈は、公開ページ（英語。CIの計測値から生成）の[Performance](../guide/performance.md)・[Performance results](../guide/performance-results.md)に置く。このノートは計測方法論・設計判断・既知の限界・今後の検討を記録する。

## 計測方法

`docs/performance/ols.md`「最重要の教訓」「計測方法」・`fe.md` と共通（release ビルド必須・`tracemalloc` 不採用・サブプロセス隔離・スレッド数を1に固定・ウォームアップ1回＋`repeats` 回の中央値）。FEIV 固有の点は以下。

### 比較対象を置かない理由（engine 単独）

`PerfAdapter.libraries=("engine",)`。理由:

- **linearmodels に固定効果つきのIVが無い**。`PanelOLS`・`AbsorbingLS` は内生変数を取れない。
- **`IV2SLS` にダミー変数を入れる形（LSDV）**は、固定効果を除く反復の代わりに個体数分のダミーの行列を作る別のアルゴリズムで、n が大きいと計測できない。
- **pandas の groupby で個体内平均を引いてから `IV2SLS` に渡す合成**は、計測の大部分が pandas の変換になり、不均衡な2-wayや3次元では参照側に交互射影を手書きする必要も出て、比較が恣意的になる。
- **pyfixest** は性能比較からも外した（`ols.md`）。**fixest（R）**は共通ハーネスのインプロセス計測に乗らない。

### 計測範囲（診断を含む）

`FEIV.fit()` は係数・標準誤差に加えて、Sargan（過剰識別のとき）・Wu-Hausman・弱操作変数の部分F・パネルR²・全体のF検定を**常に一括計算する**。計測した時間はこれらを含む。Cragg-Donald・Kleibergen-Paap・有効F は事後診断のメソッド（`fit()` では計算しない）なので含まない。

### N×T 分解と `k` の定義

`fe.md` と同じく T=6 に固定し N をスイープする（`n_entities = n // 6`。実際の観測数は要求した `n` と若干ずれる）。

**`k` は傾き係数の総数**（`x_exog` + `x_endog`、`x_exog = k − 内生変数の数`）で、IV 本体（`compare_iv.py`、`k = x_exog` の数）とは定義が違う。FE の `k` と同じ規模で比べられ、DK が要求する「時点数 T > 傾き係数の数」が cov 軸（k=5）で成り立つため（`k = x_exog の数` だと傾き係数が6個になり T=6 では `ValidationError`）。k 軸の k=20 は classical だけなので DK の制約に触れない。`compare_fe_iv.py` は、設定変更でこの前提が崩れたら import 時にエラーにする。

### データ

`benchmark/panel/fe_iv_datasets.py` の `generate_fe_iv_perf_dataset`（凍結しない、numpy の一括生成）。外生変数・操作変数・内生変数が個体効果と相関し、構造誤差と第一段階誤差が相関する（`generate_fe_iv_dataset` と同じ構造）。既定は内生変数1個・操作変数2個（過剰識別。Sargan が計測に入る）。操作変数 `l` は内生変数 `l % k_endog` に強く効き、複数の内生変数でも全て識別される。

### スイープ軸

- **n 軸**（k=5 固定、classical）: n=10,000 / 100,000 / 1,000,000。
- **k 軸**（n=10,000 固定、classical）: k=5・20。
- **cov 軸**（n=100,000・k=5、1-way）: `classical`・`hc1`・`hc2`・`hc3`・`cluster`（個体）・`dk`（バンド幅は engine の既定式）。全て engine のみ。3次元の `hc2`・`hc3` は `hc2_three_way`・`hc3_three_way` として足す。
- **method 軸**（n=1,000,000・k=5・classical の代表点1つずつ。`fe.md` と同じく `extra_methods` の仕組みを流用）:

| method | `effects` | 内生変数 / 操作変数 | データ |
|---|---|---|---|
| `one_way`（既定） | entity | 1 / 2 | 均衡（T=6） |
| `two_way` | entity, time | 1 / 2 | 均衡 |
| `two_way_unbalanced` | entity, time | 1 / 2 | 均衡の各行を確率0.7で残す（個体は最低2行） |
| `three_way` | entity, time, c | 1 / 2 | 均衡の各行に3つ目の効果 `c`（50水準）を割り振る |
| `endog3` | entity | 3 / 5 | 均衡 |
| `inst6` | entity | 1 / 6 | 均衡 |

### 繰り返しの within 変換の見方

FEIV は `y`・`x_exog`・`x_endog`・操作変数の全列に同じ within 変換をかける。不均衡な2-way・3次元では交互射影の反復が支配的な費用になりうる。**変換だけを取り出して測る公開 API は無い**ので、単独の計測は置かない（Rust 側にタイミング用の経路を足すと API 面が増える）。代わりに、変換する列の数を変える軸（k 軸・`endog3`・`inst6`）と、交互射影が必要な構造（`two_way_unbalanced`・`three_way`。均衡な `two_way` は閉形式）の `one_way` に対する比から、変換の費用を読む。

## 結果

計測値は公開ページの[Performance results](../guide/performance-results.md)を参照（`benchmark_performance.yml` のCI計測値から生成。リリース準備時に `/release` で更新する）。

- **ピークRSS**: 構造式と第一段階が、変換済みの列・残差・吸収した効果を保持するため、n が大きいときのピークRSSは FE より大きい。CI の計測（`benchmark_performance.yml` を `workflow_dispatch` で実行）では、n=1,000,000 の `one_way` で FE の約1.4倍、`two_way`・`three_way` で約1.5倍（`fit()` 中の複製を減らす前は、どれも約1.7倍）。複製を減らした前後の同じ点で、FEIV のピークRSSは約20〜25%下がった。n が小さい点は、FE の計測側にリファレンスの読み込み分が含まれるため、FEIV と比べられない（比較は n=1,000,000 の点に限る）。生データ（約8列×8バイト×n）に対する倍率の内訳は測っていない。

## 再現方法

```bash
uv run maturin develop --release
uv run python -m performance.compare_fe_iv --repeats 3 \
    --output docs/performance/results/fe_iv.json
uv run python -m performance.render_performance_summary \
    docs/performance/results/fe_iv.json
```

## 既知の限界

- **1スレッド固定**: 本計測は1スレッドに固定しており、数値は「シングルスレッドでの計算コア効率」。多コアでの実利用の性能特性とは別軸（`ols.md`「既知の限界」と共通）。
- **診断を含む計測範囲**: `fit()` が常に計算する診断（Sargan・Wu-Hausman・部分F・R²・全体F）を含む。診断を除いた係数・標準誤差だけの時間は測っていない。
- **不均衡・3次元は実行行数がずれる**: `two_way_unbalanced` は均衡の各行を確率0.7で残すため、行数は要求した `n` の約70%。`three_way` の3つ目の効果は無作為に割り振るので連結度が高い好条件で、入れ子や水準の多い効果は測っていない。どちらも同じ表の均衡の行と時間を比べられない。
- **DK は T > k が前提**: 時点数 T=6 に対し傾き係数が `k=5` までしか DK を測れない。k 軸（k=20）は classical だけ。
- **リファレンスが無い**: engine 単独で、他実装との相対値は出さない。正確性は別のテストが担う（上記「比較対象を置かない理由」）。
- **within 変換だけの計測は無い**: 公開 API が無いため、列数を変える軸と交互射影が要る構造の比から間接的に読む（「繰り返しの within 変換の見方」）。

## 今後の検討事項

- **メモリ**: 保持する状態の設計（残り）: 第一段階が内生変数ごとに生の列・変換済みの列（within変換後の設計行列）・吸収した効果を持ち、構造式用の `FeInput` と推定量の入力も生の列を持つため、`fit()` を終えた後も、生データの数倍の列が残る。`fit()` の途中の一時的な複製は減らした（第一段階・構造式・Wu-Hausman の補助回帰への列の受け渡しを複製せず移動し、点推定の行列・共分散の行列・within 変換した列を使い終わった時点で解放した）。engine 単独の割り当て計測（n=1,000,000・T=6・`classical`、生の列は約8列で約61MB）で、`fit()` 中のピークは約670MBの増加から約425MBの増加に下がった（内生変数3個・操作変数5個では約1,110MBから約810MBに）。`fit()` 後の保持は変わっていない。残りは、保持する状態を軽くする案（`first_stage()` を呼んだときに作り直す、入力を共有する）で、公開面の変更を伴うため別に判断する。HC2/HC3 の吸収した効果のレバレッジを診断の呼び出しごとに再計算する点も同様。
- **FE-GMM**: 未実装。実装したとき `extra_methods` に推定量（2段階GMM・反復GMM）を足し、重み行列の費用を見る。
- **ライブの相対比較**: 保守された固定効果つきIVのPython実装が現れ、インプロセスで計測できるようになれば再検討する。
- **releaseビルドでの再計測が前提**: 改善見込みの見積もりは、debugビルドの数値（誤り）ではなくCIのreleaseビルドの数値を基準にすること。
