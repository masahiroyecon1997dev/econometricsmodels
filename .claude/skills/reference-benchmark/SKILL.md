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
- **R（lm + sandwich/lmtest）**: 独立実装によるクロスチェック。新しい統計量・cov_type追加時はstatsmodelsとRの一致を先に確認してからフィクスチャを固定する。対象は係数・標準誤差に限らない。R²・AIC・BIC・対数尤度・F統計量・F検定p値等、公開する統計量は全てcrosscheckする（`testing-policy.md`「リファレンス実装」参照）
- **pyfixest**: OLSの正確性検証には使わない（HC2/HC3にpyfixest自身の実装バグによる系統的乖離があるため、詳細は下記「既知の差異」参照）。性能比較専用。固定効果が絡むPhase4（FE/RE）以降での採否はその時点で個別に判断する

## `benchmark/` ディレクトリの構成（動作確認済み）

`engine`/`engine_pybind`と同じ系統（family）単位でディレクトリを分けている（`linear`=OLS/WLS/GLS、`panel`=FE/RE、`iv`=IV、`discrete_choice`等は未着手）。系統をまたいで使う汎用ツールのみ`benchmark/`直下に置く。

- `benchmark/generate_synthetic_datasets.py`（系統非依存、root）: 合成データセット生成。`SCENARIOS`に7種類のバリエーション（baseline, small_n, high_variance, heteroskedastic, autocorrelated, moderate_multicollinearity, perfect_multicollinearity）を実装済み。
- `benchmark/load_wooldridge.py`（系統非依存、root）: Wooldridgeデータセットをpolars DataFrameとして読み込む（`pip install wooldridge`が必要）。
- `benchmark/freeze_datasets.py`（系統非依存、root）: 上記2つを使って生成した入力データを`tests/api_tests/fixtures/benchmarks/data/`にCSVとして**固定**するスクリプト。フィクスチャ生成・pytest実行時は、このCSVを読むだけでよい（ジェネレータを直接呼ばない）。理由: ジェネレータ側のコードが将来変わっても、既に固定したフィクスチャJSONの期待値と無言で不整合にならないようにするため。新しいシナリオ・データセットを追加した場合のみ、このスクリプトを再実行してCSVを更新する（フィクスチャJSON同様、自動追従はしない）。
- `benchmark/<系統>/run_statsmodels_benchmark.py`: 主リファレンス。1回呼べば1ケース分の結果を返す汎用ツール。`linear`系統では`--weight-col`指定でWLS（`smf.wls`）にも対応。
- `benchmark/<系統>/fixtures/generate_<手法名>_fixtures.py`: 対象手法の全シナリオ×全オプションを回し、`tests/api_tests/fixtures/benchmarks/<手法名>.json`へ書き出す専用スクリプト。生成スクリプト（`benchmark/`側）と生成物（`tests/`側）を分けている（`testing-policy.md`「ベンチマーク値のフィクスチャ化」参照）。`linear/fixtures/generate_ols_fixtures.py`（statsmodels主リファレンス）・`generate_ols_crosscheck_fixtures.py`（Rクロスチェック）が実装例。
- `benchmark/linear/run_pyfixest_benchmark.py`: OLS/WLS（`--weights`指定）で動作確認済み。Phase4以降で主に使用。OLSでは`vcov`引数でHC1-3/cluster(`{"CRV1": col}`)/HAC(`"NW"`)も指定可能だが、正式なクロスチェックはRを使う方針（下記参照）。
- Rスクリプトはパッケージ単位で1ファイルに分けている（旧`run_r_benchmark.R`の単一ディスパッチャから分割）:
  - `benchmark/linear/run_lm_crosscheck_benchmark.R`: base R `lm` + sandwich/lmtestによるOLS/WLS標準誤差クロスチェック（classical/HC0-3/cluster/HAC対応、`weights`引数でWLSにも対応）。動作検証済み、正式なクロスチェックとして使用中。
  - `benchmark/linear/run_fixest_benchmark.R`: fixestパッケージ。未検証（現状どのフィクスチャ生成スクリプトからも呼ばれていない）。
  - `benchmark/panel/run_plm_benchmark.R`: plmパッケージ。未検証（Phase4着手時に確認）。
  - `benchmark/iv/run_ivreg_benchmark.R`: ivreg/AERパッケージ。未検証（Phase3着手時に確認）。
  - 注意: `read.csv()`はデフォルトで列名を`make.names()`により書き換える（例: `_group`→`X_group`）。クラスター列等を渡す場合は影響を受けるため、各スクリプトとも`check.names = FALSE`を指定している。
- **pyfixestのHC2/HC3に関する既知の差異**: fixest（R）本体のソース（`vcov_hc2_hc3_internal`）を確認したところ、HC2/HC3にはssc（`n/(n-k)`の小標本補正）を一切適用しない設計だった。一方pyfixest（Python、v0.60.0時点）はHC1/HC2/HC3を同一分岐で扱っており、HC1用の`N/(N-k)`補正をHC2/HC3にも誤って適用している（`sqrt(N/(N-k))`がSEに掛かり、nが小さいほど乖離が拡大する。例: n=20, k=4で約11.8%）。**fixestの仕様ではなくpyfixest自身の実装バグ**であり、OLSの正確性検証からは除外し性能比較専用とする。詳細は`docs/planning/specs/ols-implementation-notes.md`「8. テスト」参照。

## 手順

1. 対象手法が新しい合成シナリオを必要とする場合は、`benchmark/generate_synthetic_datasets.py`を更新した上で`benchmark/freeze_datasets.py`を再実行し、`tests/api_tests/fixtures/benchmarks/data/`のCSVを更新する。既存シナリオを使う場合はこの手順は不要（既に固定済みのCSVを読むだけでよい）。Wooldridgeデータセットはこの固定化の対象外（下記「既知の未確定事項」参照）で、`load_wooldridge.py`経由で都度ロードする。
2. `$ARGUMENTS`（手法名）に応じて、対象の全cov_type/オプションの組み合わせで`benchmark/<系統>/run_statsmodels_benchmark.py`を実行する（固定済みCSVを読む）。
3. 新しい統計量・cov_typeを初めて使う場合は、Rの`lm`+`sandwich`/`lmtest`（`benchmark/<系統>/run_lm_crosscheck_benchmark.R`）でも同じ組み合わせを計算し、statsmodelsと一致することを確認する。一致しない場合は既定値（自由度補正等）の違いを疑って調査する。fixest/pyfixestは実装系統がfixestと同一のため、独立実装によるクロスチェックとしては使わない（補助的な確認に留める）。
   - AIC/BICはR標準の`AIC()`/`BIC()`関数をそのまま使わない。残差分散を1パラメータとして追加でカウントする慣習（k+1）のため、本実装・statsmodels（回帰係数の数kのみ使用）とはAICがちょうど2、BICが`log(n)`だけ系統的にずれる。本実装と同じ式（`-2*loglik + 2*k`等）で手計算した値と比較する（`run_lm_crosscheck_benchmark.R`が実装例）。
4. `benchmark/generate_synthetic_datasets.py`の7シナリオ（固定済みCSV経由）で対象手法を実行する。ただし完全な多重共線性シナリオは数値比較の対象外（想定エラーの発生確認のみ、`testing-policy.md`「テストの3系統」参照）。境界値・悪条件（`n=k+1`、極端なスケール差、高条件数）やクラスター系の不均衡・境界値ケースも対象手法に応じて検討する（`testing-policy.md`「テスト用データセット」参照）。
5. Wooldridge等の実データセットでも同様に確認する（`load_wooldridge.py`の`SUGGESTED_DATASETS`は候補であり、実際に使うデータセットは手法実装時に個別に確認する）。
6. 生成した結果を`tests/api_tests/fixtures/benchmarks/`にJSONとして保存する（`_meta`フィールドにリファレンス実装・バージョン・生成コマンドが含まれることを確認する）。
7. テストコード自体の作成・実行は `/test-new` `/test-run` に引き継ぐ。このスキルはベンチマーク値の生成・フィクスチャ化に留める。pytest側は合成データについては`tests/api_tests/fixtures/benchmarks/data/`の固定CSVを直接読む（ジェネレータを呼ばない）ため`wooldridge`パッケージは不要だが、Wooldridge実データのクロスチェックテストは`pytest.importorskip("wooldridge")`で任意扱いにする（下記参照）。

## 既知の未確定事項

- `load_wooldridge.py`の`SUGGESTED_DATASETS`は候補に過ぎない。手法ごとに実際に使うデータセットは個別に相談して確定する。
- **Wooldridgeデータセットは`freeze_datasets.py`での固定化対象外**（`wooldridge`パッケージ自体はMITライセンスだが、同梱データの著作権が原典の教科書側にある可能性があり、フィルタ後の部分集合であってもリポジトリにCSVとして再配布してよいか未確認のため。ユーザー確認済み）。合成データセットとは異なり、Wooldridgeデータは`load_wooldridge.py`経由で都度ロードし続ける。ライセンスが明確になれば固定化を再検討する。
- `run_plm_benchmark.R`/`run_ivreg_benchmark.R`は引き続き未検証。特に`plm`のindex指定（individual/time列）は手法・データセットごとに調整が必要。
- フィクスチャJSONの正式なディレクトリ構成・命名規則は、実際に`tests/api_tests/`を作る際に確定する。
