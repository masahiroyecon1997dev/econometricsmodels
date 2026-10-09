---
name: test-new
description: 新しい推定手法について、リファレンス実装とのベンチマーク作成とテストコード作成を行う
argument-hint: "[対象の推定手法名]"
allowed-tools: Read, Write, Edit, Bash(pytest:*), Bash(Rscript:*)
---

# 新規手法のテスト作成

対応するCLAUDE.mdの方針: 7章（テスト方針）

## 対象手法

$ARGUMENTS

## 手順

1. **リファレンス実装の選定**
   - 対象手法に応じて、主リファレンス（**statsmodels** / **linearmodels**）と、独立クロスチェック用の**Rパッケージ**（fixest, plm, AER, ivreg, momentfit等）を選定する。pyfixestは精度検証には使わない（`.claude/rules/testing-policy.md`「リファレンス実装」参照）。
   - 選定理由をユーザーに提示する。

2. **ベンチマーク値の作成**
   - リファレンス実装で対象手法を実行し、期待される推定値（係数・標準誤差等）を算出する。
   - 使用したデータセット・コード・バージョン情報を記録し、再現可能な形で残す。

3. **テストコードの作成**
   - `tests/<系統>/`（`linear`/`nonlinear`/`iv`。`benchmark/` と同じ grain）に
     `test_<手法>*.py` を作成する。共有物（`conftest.py`・`_assertions.py`・
     `_helpers.py`・`_tolerances.py`）は `tests/` 直下（裸importは
     `pyproject.toml` の `pythonpath = [".", "tests"]` で解決）。既存系統に当て
     はまらない新系統は、`benchmark/` 側のディレクトリ名に合わせて新設する。
   - 許容誤差は **相対誤差1e-8を基本方針** とする。ただし、計算方法自体がリファレンス実装と異なる手法（例: FEにおけるHausman検定など）は、その旨をコメントで明記した上で個別の許容誤差を設定する。

4. **engine単体テストの確認**
   - 対応する純粋ロジックの単体テスト（対象ソースファイル内の`#[cfg(test)] mod tests`、`cargo test -p engine`）が必要か確認し、なければ作成を提案する（`.claude/rules/rust-style.md`「テスト」参照）。

5. **公開ページへの反映**
   - `docs/guide/verification.md`（英語・mkdocsのnav掲載）に、対象手法の行を追加・更新する: 手法×リファレンス表、「What is compared」表、許容誤差の表（`tests/_tolerances.py`の実測値と一致させる）、実データ表。単一リファレンスなど独立クロスチェックの無い統計量は「Not compared against a second reference」に書く。
   - 手法固有のバリデーション（新しい列引数・オプションの検証）や`ComputationError`の原因を追加した場合は、`docs/guide/validation.md`の分類表（Method-specific／`ComputationError`）に反映する。
   - `docs/guide/inference-conventions.md`の手法別の表に1行追加する（`docs/spec/inference-conventions.md`には重複させない）。
   - 許容誤差・リファレンスの食い違い（spec・コード・公開ページ）に気づいたら独自判断で埋めず、先にユーザーへ確認する（CLAUDE.md 14章）。

## 完了条件

- リファレンス実装（statsmodels/linearmodels/R）との比較テストが`tests/`に存在する
- 許容誤差とその根拠がコードコメントに明記されている
- `docs/guide/verification.md`と`docs/guide/inference-conventions.md`に対象手法が反映されている
- 性能比較（`performance/compare_<method>.py`）を追加した場合は、`benchmark_performance.yml`のmatrixに手法を足す（公開ページ`docs/guide/performance-results.md`は次回リリース時にCIのartifactから自動生成されるため手書きしない。`docs/guide/performance.md`の手書き部分に新手法固有の注意があれば追記する）
