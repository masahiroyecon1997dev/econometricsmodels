---
name: reference-benchmark
description: pyfixestまたはRパッケージ（fixest, plm, ivreg等）と、合成データセット/Wooldridgeデータセットを使って推定手法のベンチマーク値を生成する。新しい推定手法のテスト作成（/test-new）の一部として使用する。
argument-hint: "[手法名]"
allowed-tools: Read, Write, Bash(python3:*), Bash(Rscript:*), Bash(pytest:*)
---

# リファレンスベンチマーク生成

対象の推定手法について、pyfixestまたはRパッケージでベンチマーク値（係数・標準誤差）を生成し、`tests/api_tests/`のテストで使える形にする。詳細な方針は `.claude/rules/testing-policy.md` を参照。

このスキル自体はスクリプトを持たない。実際のコードは `benchmark/` ディレクトリ（リポジトリ直下、`tests/`とは別）に実プロジェクトコードとして置く。理由: これらのスクリプトはテストの実行コードではなく、テストが使うベンチマーク値を生成するツールであり、Rなど別ランタイムに依存するため`tests/`とはライフサイクルが異なる。`.claude/skills/`側に複製すると二重管理になるため、単一ソースとして`benchmark/`のみに置く。

## `benchmark/` ディレクトリの構成（動作確認済み・pyfixest/wooldridge部分はこのSKILL作成時に実行検証済み）

- `benchmark/generate_synthetic_datasets.py`: 合成データセット生成。`SCENARIOS`に7種類のバリエーション（baseline, small_n, high_variance, heteroskedastic, autocorrelated, moderate_multicollinearity, perfect_multicollinearity）を実装済み。
- `benchmark/load_wooldridge.py`: Wooldridgeデータセットをpolars DataFrameとして読み込む（`pip install wooldridge`が必要）。
- `benchmark/run_pyfixest_benchmark.py`: 上記2つを使ってpyfixestでベンチマーク値をJSON出力する。OLS/WLS（`--weights`指定）で動作確認済み。
- `benchmark/run_r_benchmark.R`: fixest/plm/ivregでベンチマーク値をJSON出力する **未検証の初版**（開発環境で動作確認が必要）。

## 手順

1. `$ARGUMENTS`（手法名）に応じて、リファレンス実装を選定する（pyfixestで対応可能ならpyfixest、対応できない場合はRパッケージ。`.claude/rules/testing-policy.md`参照）。
2. `benchmark/generate_synthetic_datasets.py`の7シナリオ全てで対象手法を実行し、リファレンス実装との統計量（係数・標準誤差）が一致することを確認する。
3. Wooldridge等の実データセットでも同様に確認する（`load_wooldridge.py`の`SUGGESTED_DATASETS`は候補であり、実際に使うデータセットは手法実装時に個別に確認する）。
4. 対象手法が持つ全オプションの組み合わせについても、同様に一致を確認する。
5. 使用したデータセット・コード・リファレンス実装のバージョン情報を記録する（再現可能な形で残す）。
6. 生成したベンチマーク値を `tests/api_tests/` のテストコードに組み込む（許容誤差は`testing-policy.md`の基準に従う）。
7. テストコード自体の作成・実行は `/test-new` `/test-run` に引き継ぐ。このスキルはベンチマーク値の生成に留める。

## 既知の未確定事項

- `load_wooldridge.py`の`SUGGESTED_DATASETS`は候補に過ぎない。手法ごとに実際に使うデータセットは個別に相談して確定する。
- `run_r_benchmark.R`は未検証。特に`plm`のindex指定（individual/time列）は手法・データセットごとに調整が必要。
