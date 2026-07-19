---
name: reference-benchmark
description: statsmodels（主）・Rパッケージ（クロスチェック）・pyfixest（FE系）と、合成データセット/Wooldridgeデータセットを使って推定手法のベンチマーク値を生成し、tests/api_tests/fixtures/benchmarks/にJSONとして固定する。新しい推定手法のテスト作成（/test-new）の一部として使用する。
argument-hint: "[手法名]"
allowed-tools: Read, Write, Bash(python3:*), Bash(Rscript:*), Bash(pytest:*)
---

# リファレンスベンチマーク生成

対象の推定手法について、リファレンス実装でベンチマーク値を生成し、`tests/api_tests/fixtures/benchmarks/`にJSONとして固定する（他の目的のフィクスチャと混在させないためのサブディレクトリ）。詳細な方針は `.claude/rules/testing-policy.md` を参照。

このスキル自体はスクリプトを持たない。実際のコードは `benchmark/` ディレクトリ（リポジトリ直下、`tests/`とは別）に実プロジェクトコードとして置く。理由: これらのスクリプトはテストの実行コードではなく、テストが使うベンチマーク値を生成するツールであり、Rなど別ランタイムに依存するため`tests/`とはライフサイクルが異なる。`.claude/skills/`側に複製すると二重管理になるため、単一ソースとして`benchmark/`のみに置く。

## リファレンス実装の役割分担

- **statsmodels**: 主リファレンス。classical/HC0-3/cluster/HAC、AIC/BIC/log-likelihood、ロバストWald検定まで一貫して対応
- **R（lm + sandwich/lmtest）**: 独立実装によるクロスチェック。新しいcov_type追加時はstatsmodelsとRの一致を先に確認してからフィクスチャを固定する
- **pyfixest**: OLSの正確性検証には使わない（Issue #27。HC2/HC3にpyfixest自身の実装バグによる系統的乖離があるため、詳細は下記「既知の差異」参照）。性能比較専用。固定効果が絡むPhase4（FE/RE）以降での採否はその時点で個別に判断する

## `benchmark/` ディレクトリの構成（動作確認済み）

- `benchmark/generate_synthetic_datasets.py`: 合成データセット生成。`SCENARIOS`に7種類のバリエーション（baseline, small_n, high_variance, heteroskedastic, autocorrelated, moderate_multicollinearity, perfect_multicollinearity）を実装済み。
- `benchmark/load_wooldridge.py`: Wooldridgeデータセットをpolars DataFrameとして読み込む（`pip install wooldridge`が必要）。
- `benchmark/run_statsmodels_benchmark.py`: 主リファレンス。classical/HC0-3/cluster/HACすべて動作検証済み（このSKILL更新時に実行確認）。1回呼べば1ケース分の結果を返す汎用ツール。
- `benchmark/fixtures/generate_<手法名>_fixtures.py`: 対象手法の全シナリオ×全オプションを回し、`tests/api_tests/fixtures/benchmarks/<手法名>.json`へ書き出す専用スクリプト。生成スクリプト（`benchmark/`側）と生成物（`tests/`側）を分けている（`testing-policy.md`「ベンチマーク値のフィクスチャ化」参照）。`generate_ols_fixtures.py`（statsmodels主リファレンス）・`generate_ols_crosscheck_fixtures.py`（R/pyfixestクロスチェック、Issue #18）が実装例。
- `benchmark/run_pyfixest_benchmark.py`: OLS/WLS（`--weights`指定）で動作確認済み。Phase4以降で主に使用。OLSでは`vcov`引数でHC1-3/cluster(`{"CRV1": col}`)/HAC(`"NW"`)も指定可能だが、正式なクロスチェックはRを使う方針（下記参照）。
- `benchmark/run_r_benchmark.R`: fixest/plm/ivreg、および`lm`（base R + sandwich/lmtestによるOLS標準誤差クロスチェック、classical/HC0-3/cluster/HAC対応）でベンチマーク値をJSON出力する。devcontainerに`fixest`/`sandwich`/`lmtest`/`jsonlite`が導入済みであることを確認し、`lm`分岐はIssue #18で動作検証済み（plm/ivreg分岐は引き続き未検証）。
  - 注意: `read.csv()`はデフォルトで列名を`make.names()`により書き換える（例: `_group`→`X_group`）。クラスター列等を渡す場合は影響を受けるため、本スクリプトは`check.names = FALSE`を指定している。
- **pyfixestのHC2/HC3に関する既知の差異**: fixest（R）本体のソース（`vcov_hc2_hc3_internal`）を確認したところ、HC2/HC3にはssc（`n/(n-k)`の小標本補正）を一切適用しない設計だった。一方pyfixest（Python、v0.60.0時点）はHC1/HC2/HC3を同一分岐で扱っており、HC1用の`N/(N-k)`補正をHC2/HC3にも誤って適用している（`sqrt(N/(N-k))`がSEに掛かり、nが小さいほど乖離が拡大する。例: n=20, k=4で約11.8%）。**fixestの仕様ではなくpyfixest自身の実装バグ**であり、OLSの正確性検証からは除外し性能比較専用とする（Issue #27）。詳細は`docs/planning/specs/ols-implementation-notes.md`「クロスチェックの役割分担見直し」参照。

## 手順

1. `$ARGUMENTS`（手法名）に応じて、対象の全cov_type/オプションの組み合わせで`run_statsmodels_benchmark.py`を実行する。
2. 新しいcov_typeを初めて使う場合は、Rの`lm`+`sandwich`/`lmtest`（`run_r_benchmark.R`の`lm`分岐）でも同じ組み合わせを計算し、statsmodelsと一致することを確認する。一致しない場合は既定値（自由度補正等）の違いを疑って調査する。fixest/pyfixestは実装系統がfixestと同一のため、独立実装によるクロスチェックとしては使わない（補助的な確認に留める）。
3. `benchmark/generate_synthetic_datasets.py`の7シナリオで対象手法を実行する。ただし完全な多重共線性シナリオは数値比較の対象外（想定エラーの発生確認のみ、`testing-policy.md`「テストの3系統」参照）。
4. Wooldridge等の実データセットでも同様に確認する（`load_wooldridge.py`の`SUGGESTED_DATASETS`は候補であり、実際に使うデータセットは手法実装時に個別に確認する）。
5. 生成した結果を`tests/api_tests/fixtures/benchmarks/`にJSONとして保存する（`_meta`フィールドにリファレンス実装・バージョン・生成コマンドが含まれることを確認する）。
6. テストコード自体の作成・実行は `/test-new` `/test-run` に引き継ぐ。このスキルはベンチマーク値の生成・フィクスチャ化に留める。

## 既知の未確定事項

- `load_wooldridge.py`の`SUGGESTED_DATASETS`は候補に過ぎない。手法ごとに実際に使うデータセットは個別に相談して確定する。
- `run_r_benchmark.R`の`plm`/`ivreg`分岐は引き続き未検証。特に`plm`のindex指定（individual/time列）は手法・データセットごとに調整が必要。
- フィクスチャJSONの正式なディレクトリ構成・命名規則は、実際に`tests/api_tests/`を作る際に確定する。
