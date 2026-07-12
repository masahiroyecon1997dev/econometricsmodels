---
description: pyfixestまたはRパッケージ（fixest, plm, AER, ivreg等）を使って推定手法のベンチマーク値を生成する。新しい推定手法のテスト作成（/test-new）の一部として使用する。
argument-hint: [手法名]
allowed-tools: Read, Write, Bash(pytest:*), Bash(Rscript:*)
---

# リファレンスベンチマーク生成

対象の推定手法について、pyfixestまたはRパッケージでベンチマーク値（係数・標準誤差等）を生成し、`tests/api_tests/`のテストで使える形にする。

## 前提スクリプト

- `scripts/run_pyfixest_benchmark.py.template`: pyfixestでのベンチマーク生成雛形
- `scripts/run_r_benchmark.R.template`: R（fixest, plm, AER, ivreg等）でのベンチマーク生成雛形

> **TODO（ユーザー提供待ち）**: 上記2ファイルは現時点でプレースホルダーです。実際に使うデータセット（サンプルデータの所在・生成方法）、Rの実行環境（ローカルRか、Dockerか等）、出力形式（CSV/JSON等でテストに埋め込む形）をユーザーから提供してもらい次第、このSKILL.mdと合わせて更新する。

## 手順

1. `$ARGUMENTS`（手法名）に応じて、リファレンス実装を選定する（pyfixestかRパッケージか。`.claude/rules/testing-policy.md`参照）。
2. 選定したリファレンス実装でベンチマーク値を生成する。
3. 使用したデータセット・コード・リファレンス実装のバージョン情報を記録する（再現可能な形で残す）。
4. 生成したベンチマーク値を `tests/api_tests/` のテストコードに組み込む（許容誤差は`testing-policy.md`の基準に従う）。
5. テストコード自体の作成・実行は `/test-new` `/test-run` に引き継ぐ。このスキルはベンチマーク値の生成に留める。
