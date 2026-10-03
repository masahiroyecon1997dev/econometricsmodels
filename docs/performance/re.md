# RE: パフォーマンス比較（linearmodels）

`RE(...).fit()`（Rust engine + PyO3）とPython製リファレンス実装 linearmodels の実行時間・メモリ使用量比較の記録。CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けることが目的。

再実行可能なスクリプトは`performance/compare_re.py`（手法非依存の計測ハーネス`performance/_perf_harness.py` ＋ RE固有アダプタ、コミット対象）。生の計測結果JSONはコミットしない（`.gitignore`の`docs/performance/results/*.json`参照）。

比較対象は linearmodels 単体（[検証ページ](../guide/verification.md)の primary reference。`benchmark/panel/references/linearmodels_ref.py`と同じ主リファレンス）。`linearmodels.panel.RandomEffects`に対応させる。

> **役割分担**: 計測結果の表と性能特性の解釈は、公開ページ（英語。CIの計測値から生成）の[Performance](../guide/performance.md)・[Performance results](../guide/performance-results.md)に置く。このノートは計測方法論・設計判断・既知の限界・今後の検討を記録する。

## 計測方法

`docs/performance/ols.md`「最重要の教訓」「計測方法」、`docs/performance/fe.md`「N×T分解」「MultiIndex構築は計測区間の外」と共通（T=6固定・Nをスイープ、MultiIndex構築は`PerfAdapter.build_pandas_df`で計測区間の外）。RE固有の点は以下。

- **DGP**: REはFE用の合成データセットをそのまま再利用する既存方針（`benchmark/panel/references/linearmodels_ref.py`のモジュールdocstring「RE専用の合成データセット・凍結コードは追加していない」）に倣い、`generate_fe_dataset`を直接使う。
- **`RandomEffects`の切片**: `linearmodels.RandomEffects`は明示的な定数列が無いと切片を推定しないため、`build_pandas_df`でMultiIndex構築に加え`pdf["const"] = 1.0`を追加する（計測区間の外）。
- **cov_type**: classicalとhacの代表2点。classical/hc1/hc2/hc3/cluster/hacをn_entities=16,666・n_periods=6・k=5で実測した結果、hacが最重量（engine 0.4151s、classicalの0.2692sに対し+54%）だったため、FEと同じ組み合わせを採用した。
- **`time`**: `REOptions.time`は`cov_type="dk"`のHAC時系列順序専用（ハウスマン検定は常に1-way比較で影響を受けない）。本スクリプトは`cov_type="dk"`のときのみ`time`を渡す。
- **2-way軸なし**: REはv1で2-wayをスコープ外にしているため（`extra_methods=()`）、FEと異なりmethod軸は無い。
- **計測範囲の対称性**: engine（`ReEstimator::fit`）は係数・標準誤差と同じ`.fit()`の中でパネル固有R²・F統計量まで常に一括計算する。linearmodelsの`PanelResults`は遅延評価プロパティのため、`.fit()`直後に`params`/`std_errors`/`tstats`/`pvalues`/`rsquared_within`/`rsquared_between`/`rsquared_overall`/`f_statistic.stat`へ明示アクセスして確定させる（`f_statistic`はcov_type非依存のhomoskedastic固定——FEの`f_statistic_robust`とは異なり、engineの`ReEstimator`のF統計量自体がcov_typeに連動しない独自定義のため）。`aic`/`bic`はlinearmodelsが提供しないため対称性を取る対象に含めない。
- **スイープ軸**: n軸（k=5固定、n=1,000〜1,000,000）、k軸（n=10,000固定、k=5・20、classical/hac両方）。

## 考察（結果表の外に残す、機構・経緯の記録）

- **FEより重い理由**: RE自体はFEと異なりSwamy-Arora分散成分推定（内部1-way FE＋between回帰）とハウスマン検定用の内部FE呼び出しを含むため、FE単体（`fe.md`）よりも絶対値は大きい。
- **hacがclassicalより重い理由**: 「内部ハウスマン用FEが1-way→2-wayに変わる」交絡（上記「計測方法」参照）を含むぶん、FE単体のclassical→hac増分（ほぼゼロ）より明確に大きい。
- **devcontainerで観測された、n=1,000,000でlinearmodelsのclassicalがhacより遅いという直感に反する結果**: devcontainerの2回のフルスイープでいずれも同じ傾向だった（classical 8.25s/10.21s、hac 4.95s/5.80s。CIの計測では再現していない）。原因は`linearmodels.RandomEffects`の`cov_type="unadjusted"`（`HomoskedasticCovariance`）と`"kernel"`（`DriscollKraay`)の内部実装の違いに起因すると見られ、本ドキュメントの範囲では特定していない（`linearmodels`側の実装詳細のため、engine側の問題ではない）。

## 既知の限界

- **k=20・dkはengineで`ComputationError`になる（未解決）**: ハウスマン検定を補助回帰版にした変更（FE/REの`cov_type="hac"`を`dk`へ改名した後）以降、n_periods=6に対しk=20では補助回帰（傾き`2k`本）のDriscoll-Kraay共分散部分行列がほぼ特異になり、`fit()`が「Hausman test auxiliary regression failed」で失敗する。`compare_re.py`のk軸は`cov_types`（classical・dk）を両方回すため、このk=20・dkの点でjobが失敗する（FEのk軸はclassicalのみで回避済み）。公開ページには、この点を欠損として明記して載せている。
- **n=1,000,000での run-to-run 分散が大きい**: 2回のフルスイープでlinearmodelsのclassical/hacの大小関係が変わらないことは確認したが、絶対値は±20%程度変動した（classical 8.25s/10.21s）。計測は開発コンテナ上の少数回のスイープ（`repeats=3`の中央値）であり、大標本での環境ノイズ（メモリ確保・GC等）を排除しきれていない。
- その他は`ols.md`「既知の限界」と共通。

## 再現方法

```bash
uv run maturin develop --release
uv run python -m performance.compare_re --repeats 3 \
    --output docs/performance/results/re.json
uv run python -m performance.render_performance_summary \
    docs/performance/results/re.json
```

## 今後の検討事項

- **n=1,000,000でのlinearmodels classical/hacの逆転現象の原因調査**: 上記「考察」参照。`linearmodels`側の内部実装（`HomoskedasticCovariance` vs `DriscollKraay`）の違いを深掘りする価値があるが、engine側の性能には影響しないため優先度は低い。
- **ハウスマン内部FE構造変化の交絡を除いた計測**: `REOptions.time`を分離できるAPI変更（`FEOptions.hac_time`のような独立フィールド）が将来入れば、cov_type単体の計測に切り替えられる。現状のAPI設計を変更する動機としては優先度が低い。
- **releaseビルドでの再計測が前提**: 改善見込みの見積もりは、debugビルドの数値（誤り）ではなく本ドキュメントのreleaseビルド数値を基準にすること。
