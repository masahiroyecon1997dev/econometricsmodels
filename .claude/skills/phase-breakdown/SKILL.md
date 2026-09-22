---
name: phase-breakdown
description: 対象フェーズを実装可能なタスク単位に分解する（高い粒度の計画コマンド）
argument-hint: "[対象フェーズ（例: Phase 1）]"
allowed-tools: Read, Write, Grep, Glob, Bash(gh issue list:*), Bash(gh issue create:*)
---

# フェーズタスク分解

対応するCLAUDE.mdの方針: 4章（実装フェーズと進め方）、12章（今後の検討事項）

## 対象フェーズ

$ARGUMENTS

## 手順

1. `docs/plan.md` 4章から対象フェーズに含まれる手法を確認する。
2. 各手法について、以下を基本単位としてタスクに分解する。
   - `engine`側の実装
   - `engine_pybind`のバインディング
   - `python_package`のラッパー実装
   - ドキュメント（mkdocs）
   - pyfixest/R比較テスト（`/test-new`で対応）
   - 性能比較スクリプト（`performance/compare_<手法>.py` + `docs/performance/<手法>.md`。既存手法の`compare_<method>.py`をテンプレートに、主リファレンス実装との実行時間・メモリ比較を追加する。`.claude/rules/testing-policy.md`「パフォーマンス比較（ベンチマーク）の方法論」参照。正確性検証テストとセットではなく独立したタスクとして分解する——数値照合が済んでいても性能比較が漏れがちなため）
3. 手法間・タスク間の依存関係（共通基盤を先に作る必要があるか等）を整理する。
4. タスク一覧をMarkdown形式（チェックボックス）で提示する。`docs/planning/specs/`への保存も検討する。
5. 推奨する着手順序を提示する。
6. GitHub Issue化を希望する場合のみ、タスク一覧を提示した上でユーザーの明示的な確認を得てから `gh issue create` で作成する。

## 出力形式

- 手法ごとにグルーピングされたタスク一覧（依存関係を明記）
- 推奨する着手順序
