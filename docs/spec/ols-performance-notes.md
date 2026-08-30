# OLS: パフォーマンス比較（statsmodels）

`OLS(...).fit()`（Rust engine + PyO3）とPython製リファレンス実装 statsmodels の実行時間・メモリ使用量比較の記録。CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けることが目的。

再実行可能なスクリプトは`performance/compare_ols.py`（手法非依存の計測ハーネス`performance/_perf_harness.py` ＋ OLS固有アダプタ、コミット対象）。生の計測結果JSONはコミットしない（`.gitignore`の`docs/spec/_*.json`参照。実行環境依存で再現性が低いため）。

比較対象を statsmodels 単体に絞っている（README「Verification accuracy」表の primary reference を性能比較でも踏襲、Issue #250）。以前は pyfixest とも比較していたが、正確性検証に使わない実装を性能比較のためだけに依存させる意味が薄いため廃止した（過去の pyfixest 込みの数値は git 履歴の本ファイル旧版を参照）。

## 最重要の教訓: engineは必ずreleaseビルドで計測する

最初のフルスイープを`uv run maturin develop`（デフォルト、debugビルド）のまま実行したところ、**HACがn=1,000,000で224秒経過しても完了せず強制終了**するなど、engineがリファレンス実装より10〜140倍遅いという結果になった。

原因を`.so`ファイルサイズで確認したところ、debugビルドは**約900MB**、`uv run maturin develop --release`後は**約33MB**だった。releaseビルドで再計測すると、classical/HACとも全nでstatsmodelsより高速という、debugビルド時とは全く逆の結論になった。

**この罠を再発させないため、`_perf_harness.py`の`_worker()`は起動時に`_lib`の`.so`ファイルサイズを確認し、debugビルドの疑いがある場合は標準エラー出力に警告を出す**（`_warn_if_debug_build()`、閾値200MB）。今後このスクリプトを実行する前は必ず`uv run maturin develop --release`を実行すること。

## 計測方法

- **計測対象**: `OLS(...).fit()`全体（Python API呼び出し、Arrow変換・PyO3オーバーヘッド込みのエンドツーエンド。実際のユーザー体験に近い値にするため）
- **cov_type**: classical と HAC（Newey-West）の代表2点のみ。`.claude/rules/testing-policy.md`「パフォーマンス比較（ベンチマーク）の方法論」に従い、最も軽いものと最も計算コストの重いものの2点で足りるため（HC1/cluster は classical と同傾向のため省く）。HACのラグ数は両ライブラリで明示的に揃える（`hac_auto_lag(n) = 4*(n/100)^(2/9)`、engineの自動選択式と同じ）
- **計測範囲の対称性（Issue #98）**: engine は係数・標準誤差と同じ呼び出しの中で R²・調整済みR²・対数尤度・AIC・BIC・F統計量・F検定のp値まで**常に一括計算**する。statsmodels はこれらを遅延評価（`cached_value`）にしているため、`.fit()`直後に該当プロパティへ明示アクセスして「フルセットの適合度統計量込み」で計測範囲を揃えている
- **スレッド数を1に固定**: engine（faer/rayon）・statsmodels（numpy/BLAS）とも`RAYON_NUM_THREADS=1`等でシングルスレッドに固定する（`_perf_harness._SINGLE_THREAD_ENV`）。理由は下記「既知の限界」参照
- **実行時間**: ウォームアップ1回 + `repeats`回実行（今回`repeats=3`）の中央値（`time.perf_counter()`）
- **メモリ**: プロセス単位のピークRSS（`resource.getrusage(RUSAGE_SELF).ru_maxrss`）。`tracemalloc`はRust内部（faerの行列確保等）やnumpyバッファのようなネイティブメモリ確保を捕捉できないことを実測で確認したため不採用（設計行列だけで80MB相当のケースで3.8KB程度しか検知しなかった）
- **サブプロセス隔離**: 1計測点＝1サブプロセス。同一プロセス内で連続測定するとアロケータが解放済みメモリを保持したままになり後続の計測のRSSが汚染されるため
- **変換コスト除外**: polars→pandas変換（`df.to_pandas()`）は計測区間の外で1回だけ実施し、各試行で使い回す（リファレンス実装本体の処理ではないため）。engine側はArrowゼロコピーでpolarsをそのまま渡す
- **スイープ軸**: n軸（k=5固定、n=1,000〜1,000,000）、k軸（n=10,000固定、k=5・20）

## 結果: n軸（k=5固定）

実行時間（秒、中央値）/ ピークRSS（MB）。devcontainer（12論理コア、シングルスレッド固定）、`repeats=3`。

| n | engine | statsmodels |
|---|---|---|
| **classical** | | |
| 1,000 | 0.0001 / 158 | 0.0062 / 203 |
| 10,000 | 0.0011 / 160 | 0.0080 / 205 |
| 100,000 | 0.0123 / 183 | 0.0281 / 238 |
| 1,000,000 | 0.1388 / 402 | 0.2675 / 518 |
| **HAC** | | |
| 1,000 | 0.0001 / 158 | 0.0074 / 204 |
| 10,000 | 0.0016 / 160 | 0.0110 / 207 |
| 100,000 | 0.0218 / 192 | 0.0653 / 243 |
| 1,000,000 | 0.3593 / 440 | 1.1911 / 612 |

## 結果: k軸（n=10,000固定）

実行時間（秒、中央値）。

| k | engine | statsmodels |
|---|---|---|
| **classical** | | |
| 5 | 0.0014 | 0.0115 |
| 20 | 0.0043 | 0.0294 |
| **HAC** | | |
| 5 | 0.0016 | 0.0108 |
| 20 | 0.0123 | 0.0361 |

## 考察

- **classical**: 全nでengineがstatsmodelsより高速。n=1,000,000でengineはstatsmodelsの約1.9倍（0.139s vs 0.268s）、n=100,000で約2.3倍速い。小規模nではstatsmodels側のPython/formula/pandasオーバーヘッドが支配的で、engineは1ミリ秒未満に収まる。適合度統計量一式の計算を両者に含めた対称な計測（Issue #98）でもこの差は変わらない。
- **HAC**: engineが全nで一貫して約3倍速い（n=1,000,000で0.36s vs 1.19s、n=100,000で0.022s vs 0.065s）。以前の pyfixest 込みの計測では n=1,000,000 で3者ほぼ互角だったが、シングルスレッド固定・対称計測にした結果、engine優位がはっきり出た。
- **HACのkスケーリングに気になる点が残る**: n=10,000固定でk=5→20に増やすと、engineは0.0016s→0.0123s（**約7.7倍**）、statsmodelsは0.0108s→0.0361s（**約3.3倍**）。classical（engine約3.1倍 / statsmodels約2.6倍）と比べ、HACだけengineのk方向の伸びがリファレンスより急。Newey-West計算のk方向の計算量・実装に余地がある可能性（下記「今後の検討事項」）。
- **メモリはengineが一貫して軽い**: 全cov_type・全nでengineのピークRSSが小さい（n=1,000,000でengine 402〜440MB、statsmodels 518〜612MB）。小〜中規模nではengineが約158〜192MB、statsmodelsが約203〜243MBで、statsmodels側のimport一式（pandas等）の基礎コストの差が出ている。
- **小規模n（1,000）でPyO3/Arrow変換オーバーヘッドは支配的にならない**: classicalでengine 0.0001s（1ミリ秒未満）とstatsmodelsより桁で高速。「小規模では変換オーバーヘッドが支配的で遅くなる」という当初の懸念は当てはまらなかった。

## 既知の限界

- **engineのマルチスレッド線形代数が多コア機・負荷下で不安定**: スレッド数を制限しないと、classical n=1,000,000 でengineの実行時間がシングルスレッド時の約0.15秒から**3〜4秒**（試行ごとに0.6〜4.5秒とばらつく）に膨れ上がる現象を実測（2026-08-30、12論理コアのdevcontainer）。faerは0.24.4のまま・`engine/src/linear/ols.rs`は2026-08-09以降変更が無いためコードリグレッションではなく、線形代数バックエンドのスレッドプールが負荷下で競合する挙動の問題。比較対象のstatsmodels（numpy/OpenBLAS）は同条件でも安定して劣化しなかった。**本計測はengine・statsmodelsとも1スレッドに固定してこの要因を切り離しているため、数値は「シングルスレッドでの計算コア効率」であり、多コアでのマルチスレッド高速化は反映していない**。engine側の挙動そのものは別途調査する（`docs/planning/specs/refactoring-candidates.md`項目44）。
- 計測は開発コンテナ（devcontainer）上の1回のスイープ（`repeats=3`の中央値）。環境ノイズ・実行順序の影響を排除しきれていない。CI（`benchmark_performance.yml`、`ubuntu-latest`）でも同じスクリプトを回すが、共有ランナーのため数値は参考値。

## 再現方法

```bash
# 1. releaseビルドを確認（上記「最重要の教訓」参照）
uv run maturin develop --release

# 2. フルスイープを実行（リポジトリルートから。スレッド数固定はハーネスが自動で行う）
uv run python -m performance.compare_ols --repeats 3 \
    --output docs/spec/_ols_performance_results.json

# 3. Markdown整形（任意）
uv run python -m performance.render_performance_summary \
    docs/spec/_ols_performance_results.json
```

## 今後の検討事項

- **engineのマルチスレッド線形代数の不安定性**（上記「既知の限界」、`refactoring-candidates.md`項目44）: 最優先。OLSの設計行列はtall-skinny（n大・k小）で、skinny行列のQR/Gram構築を多スレッドに分割するとスレッドプールのオーバーヘッド・メモリ帯域競合が支配的になりやすい。問題サイズに応じてシングルスレッドに留める閾値、または明示的なスレッド数上限の導入を検討する。WSL2固有かネイティブ多コアLinuxでも再現するかの切り分けも要る。
- **HACのkスケーリング**（上記「考察」参照）: engineのNewey-West計算（`hac_cov_params`）のk方向の計算量・実装を見る価値がある。
- **releaseビルドでの再計測が前提**: 改善見込みの見積もりは、debugビルドの数値（誤り）ではなく本ドキュメントのreleaseビルド数値を基準にすること。
