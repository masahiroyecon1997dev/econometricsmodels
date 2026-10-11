# OLS: パフォーマンス比較（statsmodels）

`OLS(...).fit()`（Rust engine + PyO3）とPython製リファレンス実装 statsmodels の実行時間・メモリ使用量比較の記録。CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けることが目的。

再実行可能なスクリプトは`performance/compare_ols.py`（手法非依存の計測ハーネス`performance/_perf_harness.py` ＋ OLS固有アダプタ、コミット対象）。生の計測結果JSONはコミットしない（`.gitignore`の`docs/performance/results/*.json`参照。実行環境依存で再現性が低いため）。

比較対象を statsmodels 単体に絞っている（[検証ページ](../guide/verification.md)の primary reference を性能比較でも踏襲）。以前は pyfixest とも比較していたが、正確性検証に使わない実装を性能比較のためだけに依存させる意味が薄いため廃止した（過去の pyfixest 込みの数値は git 履歴の本ファイル旧版を参照）。

> **役割分担**: 計測結果の表と性能特性の解釈は、公開ページ（英語。CIの計測値から生成）の[Performance](../guide/performance.md)・[Performance results](../guide/performance-results.md)に置く。このノートは計測方法論・設計判断・既知の限界・今後の検討を記録する。

## 最重要の教訓: engineは必ずreleaseビルドで計測する

最初のフルスイープを`uv run maturin develop`（デフォルト、debugビルド）のまま実行したところ、**HACがn=1,000,000で224秒経過しても完了せず強制終了**するなど、engineがリファレンス実装より10〜140倍遅いという結果になった。

原因を`.so`ファイルサイズで確認したところ、debugビルドは**約900MB**、`uv run maturin develop --release`後は**約33MB**だった。releaseビルドで再計測すると、classical/HACとも全nでstatsmodelsより高速という、debugビルド時とは全く逆の結論になった。

**この罠を再発させないため、`_perf_harness.py`の`_worker()`は起動時に`_lib`の`.so`ファイルサイズを確認し、debugビルドの疑いがある場合は標準エラー出力に警告を出す**（`_warn_if_debug_build()`、閾値200MB）。今後このスクリプトを実行する前は必ず`uv run maturin develop --release`を実行すること。

## 計測方法

- **計測対象**: `OLS(...).fit()`全体（Python API呼び出し、Arrow変換・PyO3オーバーヘッド込みのエンドツーエンド。実際のユーザー体験に近い値にするため）
- **cov_type**: classical と HAC（Newey-West）の代表2点のみ。`.claude/rules/testing-policy.md`「パフォーマンス比較（ベンチマーク）の方法論」に従い、最も軽いものと最も計算コストの重いものの2点で足りるため（HC1/cluster は classical と同傾向のため省く）。HACのラグ数は両ライブラリで明示的に揃える（`hac_auto_lag(n) = 4*(n/100)^(2/9)`、engineの自動選択式と同じ）
- **計測範囲の対称性**: engine は係数・標準誤差と同じ呼び出しの中で R²・調整済みR²・対数尤度・AIC・BIC・F統計量・F検定のp値まで**常に一括計算**する。statsmodels はこれらを遅延評価（`cached_value`）にしているため、`.fit()`直後に該当プロパティへ明示アクセスして「フルセットの適合度統計量込み」で計測範囲を揃えている
- **スレッド数を1に固定**: engine・statsmodels（numpy/BLAS）とも`RAYON_NUM_THREADS=1`等でシングルスレッドに固定する（`_perf_harness._SINGLE_THREAD_ENV`）。engine側は faer のグローバル並列度を常時`Par::Seq`にしたため実質的に冗長だが、リファレンス実装と対称にする（両者とも逐次）目的で維持している。この選択がユーザー実環境（マルチスレッドBLAS）とどう乖離するかは下記「マルチスレッド環境での挙動」参照
- **実行時間**: ウォームアップ1回 + `repeats`回実行（今回`repeats=3`）の中央値（`time.perf_counter()`）
- **メモリ**: プロセス単位のピークRSS（`resource.getrusage(RUSAGE_SELF).ru_maxrss`）。`tracemalloc`はRust内部（faerの行列確保等）やnumpyバッファのようなネイティブメモリ確保を捕捉できないことを実測で確認したため不採用（設計行列だけで80MB相当のケースで3.8KB程度しか検知しなかった）
- **サブプロセス隔離**: 1計測点＝1サブプロセス。同一プロセス内で連続測定するとアロケータが解放済みメモリを保持したままになり後続の計測のRSSが汚染されるため
- **変換コスト除外**: polars→pandas変換（`df.to_pandas()`）は計測区間の外で1回だけ実施し、各試行で使い回す（リファレンス実装本体の処理ではないため）。engine側はArrowゼロコピーでpolarsをそのまま渡す
- **スイープ軸**: n軸（k=5固定、n=1,000〜1,000,000）、k軸（n=10,000固定、k=5・20）

## 既知の限界

- **engineのマルチスレッド線形代数が多コア機・負荷下で不安定だった（対応済み）**: スレッド数を制限しないと、classical n=1,000,000 でengineの実行時間がシングルスレッド時の約0.13秒から、全コア並列＋背景CPU負荷下では**中央値24.9秒**（無負荷でも中央値0.24秒・単発スパイク1.0秒）に膨れ上がる現象を実測。faerが`rayon` feature既定ONでグローバル並列度が`Par::Rayon(0)`（全コア）のまま、tall-skinnyな設計行列のQR/Gramを暗黙並列化していたことが原因（コードリグレッションではない）。**対策**: `engine::shared::parallelism::ensure_serial()`でfaerのグローバル並列度を常時`Par::Seq`に固定（全`Estimator::fit()`冒頭＋`#[pymodule]`初期化）。対応後は全コア並列＋負荷下でも0.39秒、無負荷で0.14秒（分散1/14）に安定。並列化は今後、実測で有効な箇所のみ`Par::Rayon`を明示opt-inする方針（`.claude/rules/rust-style.md`「パフォーマンス」・`engine/src/linear/CLAUDE.md`）。tall-skinny OLS では暗黙の全コア並列化が高速化しないのは OpenBLAS も同様で、statsmodels 側も1スレッドの方が速い（下記「マルチスレッド環境での挙動」）。WSL2固有か native 多コア Linux でも同程度かは未検証（傾向自体はスレッドプールのオーバーサブスクリプションの一般的挙動でOS問わず出るはず）。
- 計測は開発コンテナ（devcontainer）上の1回のスイープ（`repeats=3`の中央値）。環境ノイズ・実行順序の影響を排除しきれていない。CI（`benchmark_performance.yml`、`ubuntu-latest`）でも同じスクリプトを回すが、共有ランナーのため数値は参考値。

## マルチスレッド環境での挙動（1スレッド固定の妥当性）

本計測は engine・statsmodels とも1スレッドに固定している。「ユーザーの実環境では statsmodels
の BLAS がマルチスレッドで動くので、この比較は実環境の姿を表さないのでは」という疑問に対する
実測（devcontainer、12論理コア、numpy=scipy-openblas 0.3.34、n=1,000,000、classical、
`repeats=10〜12`の中央値）:

| k | ライブラリ | threads=1 | threads=無制限 |
|---|---|---|---|
| 5 | statsmodels | 0.31s（pstdev 0.01） | **0.45s**（pstdev 0.09） |
| 5 | engine | 0.16s | 0.14s |
| 20 | statsmodels | 1.82s | 1.70s |
| 20 | engine | 0.79s | 0.65s |
| 50 | statsmodels | 5.61s | 4.69s |
| 50 | engine | 2.71s | 2.60s |

（HAC・k=5 も同傾向: statsmodels は threads=1 の 1.27s に対し無制限で 1.99s と悪化）

読み取れること:

- **k が小さい領域（＝多数の観測・少数の回帰変数という計量経済学の標準ケース、n軸スイープの
  `k=5`）では、BLAS のマルチスレッド化は statsmodels をむしろ 40〜55% 遅くし、試行間分散も
  数倍に増やす**。tall-skinny な `X'X`（`1e6×5 → 5×5` の縮約）は演算密度が低くメモリ帯域律速で、
  スレッドプールのオーバーサブスクリプション・帯域競合がオーバーヘッドとして支配的になるため
  （engine が過去に踏んだのと同じ現象が OpenBLAS でも起きる）。つまり **1スレッド固定は
  statsmodels を不利にしているのではなく、statsmodels の最良ケースを見せている**（実マルチコア
  環境のユーザーはむしろ遅い側を体験する）。
- **`k=20`（k軸スイープの上限）で statsmodels のマルチスレッド化がほぼ均衡、`k=50` で初めて
  明確な恩恵（約16%高速）** を受ける。ただし engine はどの k でも線形代数がマルチスレッドで
  高速化しない設計（`Par::Seq`）にもかかわらず、`k=50` でも engine が statsmodels の
  マルチスレッド値の 1.8 倍速い。engine の相対優位は statsmodels のスレッド設定によらず頑健。
- したがって「マルチスレッドで statsmodels が engine を脅かす」状況は、少なくとも本パッケージが
  想定する問題形状（`k` が数個〜数十）では観測されない。1スレッド固定は「計算コア効率の
  apples-to-apples 比較」として妥当であり、実環境のユーザー体験を engine 有利に歪めてもいない。

（engine 側の `threads=無制限` がわずかに速いのは faer の並列化ではなく、`OLS()` ラッパーの
polars 前処理が `POLARS_MAX_THREADS` 無制限で並列化されるため。線形代数本体は `Par::Seq`。）

## 再現方法

```bash
# 1. releaseビルドを確認（上記「最重要の教訓」参照）
uv run maturin develop --release

# 2. フルスイープを実行（リポジトリルートから。スレッド数固定はハーネスが自動で行う）
uv run python -m performance.compare_ols --repeats 3 \
    --output docs/performance/results/ols.json

# 3. Markdown整形（任意）
uv run python -m performance.render_performance_summary \
    docs/performance/results/ols.json
```

## 今後の検討事項

- **engineのマルチスレッド線形代数の不安定性（対応済み）**: 上記「既知の限界」参照。faerのグローバル並列度を`Par::Seq`固定にして解消した。残課題は「WSL2固有か native 多コア Linux でも同程度か」の切り分け（優先度低）。
- **releaseビルドでの再計測が前提**: 改善見込みの見積もりは、debugビルドの数値（誤り）ではなく本ドキュメントのreleaseビルド数値を基準にすること。
