# RE: パフォーマンス比較（linearmodels）

`RE(...).fit()`（Rust engine + PyO3）とPython製リファレンス実装 linearmodels の実行時間・メモリ使用量比較の記録。CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けることが目的。

再実行可能なスクリプトは`performance/compare_re.py`（手法非依存の計測ハーネス`performance/_perf_harness.py` ＋ RE固有アダプタ、コミット対象）。生の計測結果JSONはコミットしない（`.gitignore`の`docs/performance/results/*.json`参照）。

比較対象は linearmodels 単体（README「Verification accuracy」表の primary reference。`benchmark/panel/references/linearmodels_ref.py`と同じ主リファレンス）。`linearmodels.panel.RandomEffects`に対応させる。

## 計測方法

`docs/performance/ols.md`「最重要の教訓」「計測方法」、`docs/performance/fe.md`「N×T分解」「MultiIndex構築は計測区間の外」と共通（T=6固定・Nをスイープ、MultiIndex構築は`PerfAdapter.build_pandas_df`で計測区間の外）。RE固有の点は以下。

- **DGP**: REはFE用の合成データセットをそのまま再利用する既存方針（`benchmark/panel/references/linearmodels_ref.py`のモジュールdocstring「RE専用の合成データセット・凍結コードは追加していない」）に倣い、`generate_fe_dataset`を直接使う。
- **`RandomEffects`の切片**: `linearmodels.RandomEffects`は明示的な定数列が無いと切片を推定しないため、`build_pandas_df`でMultiIndex構築に加え`pdf["const"] = 1.0`を追加する（計測区間の外）。
- **cov_type**: classicalとhacの代表2点。classical/hc1/hc2/hc3/cluster/hacをn_entities=16,666・n_periods=6・k=5で実測した結果、hacが最重量（engine 0.4151s、classicalの0.2692sに対し+54%）だったため、FEと同じ組み合わせを採用した。
- **既知の限界: cov_type間でハウスマン内部FEの構造も変わる**: `REOptions.time`は「HACの時系列順序」と「ハウスマン検定用の内部FE呼び出しの1-way/2-way選択（`Some`なら2-way、`None`なら1-way）」を兼ねる（`engine_pybind/src/panel/re.rs`）。本スクリプトは`cov_type="hac"`のときのみ`time`を渡すため、classicalとhacの計測差には「cov_type自体の計算コスト差」に加え「内部ハウスマン用FEが1-way→2-wayに変わることによる追加コスト」が混入する（RE自身のAPI設計上不可避な交絡）。
- **2-way軸なし**: REはv1で2-wayをスコープ外にしているため（`extra_methods=()`）、FEと異なりmethod軸は無い。
- **計測範囲の対称性**: engine（`ReEstimator::fit`）は係数・標準誤差と同じ`.fit()`の中でパネル固有R²・F統計量まで常に一括計算する。linearmodelsの`PanelResults`は遅延評価プロパティのため、`.fit()`直後に`params`/`std_errors`/`tstats`/`pvalues`/`rsquared_within`/`rsquared_between`/`rsquared_overall`/`f_statistic.stat`へ明示アクセスして確定させる（`f_statistic`はcov_type非依存のhomoskedastic固定——FEの`f_statistic_robust`とは異なり、engineの`ReEstimator`のF統計量自体がcov_typeに連動しない独自定義のため）。`aic`/`bic`はlinearmodelsが提供しないため対称性を取る対象に含めない。
- **スイープ軸**: n軸（k=5固定、n=1,000〜1,000,000）、k軸（n=10,000固定、k=5・20、classical/hac両方）。

## 結果: n軸（k=5固定）

実行時間（秒、中央値）/ ピークRSS（MB）。devcontainer（12論理コア、シングルスレッド固定）、`repeats=3`。

| n | engine | linearmodels |
|---|---|---|
| **classical** | | |
| 1,000 | 0.0017 / 212 | 0.0320 / 217 |
| 10,000 | 0.0171 / 215 | 0.0630 / 231 |
| 100,000 | 0.2160 / 270 | 0.4877 / 333 |
| 1,000,000 | 4.2359 / 779 | 8.2486 / 1242 |
| **hac** | | |
| 1,000 | 0.0028 / 211 | 0.0348 / 217 |
| 10,000 | 0.0315 / 218 | 0.0883 / 231 |
| 100,000 | 0.4453 / 299 | 0.5129 / 332 |
| 1,000,000 | 6.9812 / 1079 | 4.9498 / 1242 |

## 結果: k軸（n=10,000固定）

実行時間（秒、中央値）。

| k | engine | linearmodels |
|---|---|---|
| **classical** | | |
| 5 | 0.0166 | 0.0765 |
| 20 | 0.0434 | 0.1104 |
| **hac** | | |
| 5 | 0.0293 | 0.0624 |
| 20 | 0.0648 | 0.0910 |

## 考察

- **classical**: 全nでengineがlinearmodelsより高速（n=1,000,000で約1.9倍、4.24s vs 8.25s）。RE自体はFEと異なりSwamy-Arora分散成分推定（内部1-way FE＋between回帰）とハウスマン検定用の内部FE呼び出しを含むため、FE単体（`fe.md`）よりも絶対値は大きい（n=1,000,000でFE classical 1.58s、RE classical 4.24s）。
- **hac**: engineはn=1,000,000でclassicalの約1.6倍（4.24s→6.98s）——「内部ハウスマン用FEが1-way→2-wayに変わる」交絡（上記「計測方法」参照）を含むぶん、FE単体のclassical→hac増分（ほぼゼロ）より明確に大きい。
- **n=1,000,000でlinearmodelsのclassicalがhacより遅いという直感に反する結果**: 2回のフルスイープでいずれも同じ傾向を確認した（classical 8.25s/10.21s、hac 4.95s/5.80s）。単発ノイズではなく再現する挙動だが、原因は`linearmodels.RandomEffects`の`cov_type="unadjusted"`（`HomoskedasticCovariance`）と`"kernel"`（`DriscollKraay`)の内部実装の違いに起因すると見られ、本ドキュメントの範囲では特定していない（`linearmodels`側の実装詳細のため、engine側の問題ではない）。n=1,000,000でのrun-to-run分散が他のn・他手法より大きい点に注意（下記「既知の限界」）。
- **メモリはengineが軽い**: n=1,000,000でengine約779〜1079MBに対しlinearmodels約1242MB。
- **kスケーリング**: classical k=5→20でengine約2.6倍（0.0166s→0.0434s）、linearmodels約1.4倍（0.0765s→0.1104s）。hacも同様の傾向（engine約2.2倍、linearmodels約1.5倍）。RE自身のk=20・hacはFEと異なり`ComputationError`にならず正常に計測できる（下記「既知の限界」、`fe.md`参照）。

## 既知の限界

- **n=1,000,000での run-to-run 分散が大きい**: 2回のフルスイープでlinearmodelsのclassical/hacの大小関係が変わらないことは確認したが、絶対値は±20%程度変動した（classical 8.25s/10.21s）。計測は開発コンテナ上の少数回のスイープ（`repeats=3`の中央値）であり、大標本での環境ノイズ（メモリ確保・GC等）を排除しきれていない。
- **cov_type間の比較に「ハウスマン内部FEの構造変化」という交絡が混入する**（上記「計測方法」参照）。RE自身のAPI設計（`REOptions.time`がHAC時系列順序とハウスマン内部FEの1-way/2-way選択を兼ねる）に起因する不可避な交絡であり、回避策は無い。
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
- **ハウスマン内部FE構造変化の交絡を除いた計測**: `REOptions.time`を分離できるAPI変更（`FEOptions.time_col`のような独立フィールド）が将来入れば、cov_type単体の計測に切り替えられる。現状のAPI設計を変更する動機としては優先度が低い。
- **releaseビルドでの再計測が前提**: 改善見込みの見積もりは、debugビルドの数値（誤り）ではなく本ドキュメントのreleaseビルド数値を基準にすること。
