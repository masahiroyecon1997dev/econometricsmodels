# CLAUDE.md

このファイルは、Claude Code がこのリポジトリで作業する際に毎回参照する前提知識です。
言語別の詳細なコーディング規約・テスト方針は `.claude/rules/`（パス指定で自動ロード）に、定型作業は `.claude/commands/` に、コードレビューは `.claude/agents/` のサブエージェントに分離している。重複記載を避けるため、このファイルには全体像・非交渉事項・各詳細ファイルへの参照のみを置く。

## 1. プロジェクト概要

| 項目 | 内容 |
|---|---|
| 名称 | econometricsmodels |
| 目的 | 統計・計量経済学の分析手法を提供するPython API |
| 用途 | 自作の分析GUIアプリ「economicon」のエンジンとして使用 |
| 技術スタック | Rust + PyO3（Python拡張） |
| 線形代数クレート | faer（pure Rust、システムBLAS/LAPACK非依存） |
| ライセンス | MIT License |
| 公開先 | PyPI |
| 開発体制 | 基本一人開発（Claudeと二人三脚）。Git運用の詳細は5章参照 |

## 2. 絶対に守るべき設計方針（非交渉事項）

以下はユーザーが明示的に決定した設計方針であり、**Claudeが「使いやすさ」等の理由で自己判断により逸脱・変更を提案してはならない**。

- **データ入力はpolarsのみ**。pandas等の他形式は受け付けない。
- **Arrowのゼロコピー**でRust側にデータを渡す。コピーによるメモリ・速度のロスを避ける。
- **formula文字列パース方式（`y ~ x1 + x2`）は不採用**。
  - 被説明変数`y`は **単一の列名（`str`）渡し**、説明変数`x`は **List渡し**（例: `y="y_col", x=["x1", "x2"]`）
    - `y`をList型にしない理由: Phase1〜6（VAR等の一部時系列手法を除く）でyは常に1変数であり、`list[str]`だと「長さ1であること」を全推定関数が実行時検証する必要が生じる。将来的に真に多変量なyが必要な手法（VAR等）が出てきた場合は、その手法だけ`y: list[str]`にする。
  - 推定オプションは **オブジェクト（設定用クラス／構造体）渡し**
  - 理由: スクリプト・プログラムからの呼び出しやすさ（型補完、バリデーション、動的組み立て）を優先するため。
- 計算コアはRustで実装し高速化。Python側はPyO3バインディングとして薄く保つ。

これらの変更が必要と思われる場合も、まず提案として提示し、ユーザーの明示的な承認を得てから実装すること。

## 3. リポジトリ構成

```
econometricsmodels/
├── Cargo.toml                  # Workspaceルート
├── pyproject.toml              # maturinの設定（engine_pybindをビルド対象にする）
│
├── .devcontainer/               # Rust + Python 3.12 環境（中身は未確定、TBD）
│   ├── devcontainer.json
│   └── Dockerfile
│
├── engine/                       # 純粋Rustの計算心臓部（PyO3非依存）
│   └── src/{lib.rs, ols.rs, fe.rs, ...}
│
├── engine_pybind/                # PyO3の薄いバインディング層
│   └── src/lib.rs                # #[pymodule] を定義し engine の関数を呼ぶ
│
├── python_package/econometricsmodels/
│   ├── __init__.py               # engine_pybindからのインポート、Polarsラッパー
│   └── py.typed
│
├── tests/
│   ├── engine_tests/             # cargo test（純粋ロジック）
│   └── api_tests/                 # pytest（pyfixest / R実装との答え合わせ）
│
├── benchmark/                     # テスト用データセット生成・リファレンス実装（pyfixest/R）でのベンチマーク値生成スクリプト
│                                   # tests/とは別ライフサイクル（Rランタイム依存、随時実行するツール）。詳細は.claude/skills/reference-benchmark/
│
├── docs/                          # MkDocs（GitHub Pages公開）
│   ├── mkdocs.yml
│   └── planning/                  # plan.md・仕様書（詳細は10章）
│
└── .github/workflows/
    ├── ci_engine.yml               # cargo test / clippy / fmt（engine/配下トリガー）
    ├── ci_python.yml               # pytest / Ruff（python_package/ engine_pybind/ 配下トリガー）
    ├── cd_release.yml              # maturin-actionでのマルチOSホイールビルド・PyPI公開
    └── cd_docs.yml                  # mkdocs → GitHub Pages
```

`engine`と`engine_pybind`を分離しているのは、推定ロジックをPyO3非依存で`cargo test`できるようにするため。手法追加時は基本的に`engine`配下にmoduleを足すだけでよい。

## 4. 実装フェーズと進め方

「基礎から積み上げる」順に段階実装する。**一度に全フェーズ／全手法を実装しない**。フェーズ・タスク単位に細分化して、1つずつ完了させてから次に進む。

1. Phase 1（基礎回帰）: OLS, 区分回帰, WLS
2. Phase 2（一般化・離散選択）: GLS, Logit, Probit, Tobit
3. Phase 3（操作変数）: IV（2SLS, GMM）
4. Phase 4（パネルデータ）: FE（固定効果）, RE（変量効果）
5. Phase 5（因果推論）: DID, RDD
6. Phase 6（IO手法）: ロジット, Nested Logit, Random Coefficient Logit, シングルエージェントモデル, 静学ゲーム, 動学ゲーム
7. Phase 7（後回し・時系列）: ARCH, GARCH, VAR

**現在のステータス: 全フェーズ未着手。次はリポジトリ雛形の作成（Phase 1のタスク分解の詳細は13章参照）。**

**直近の実装順序（Phase横断で決定済み）**: OLS → WLS → IV → Probit / Logit。Phaseのグルーピング（上記1〜7）は分類上のものであり、実際の着手順序はこの直近の並びを優先する。

## 5. Git運用

- **コミットメッセージ**: Conventional Commits（`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `ci:` 等）
- **ブランチ戦略**: feature branch + PR必須を基本とし、機能追加はフェーズ単位でブランチを切る（例: `phase1-ols`, `phase1-wls`）
- **マージ**: CIがgreenであることに加え、内容を確認してからmergeする（自動セルフマージはしない）

## 6. コーディング規約

詳細は `.claude/rules/rust-style.md`（engine/engine_pybind配下で自動ロード）、`.claude/rules/python-style.md`（python_package配下で自動ロード）を参照。要点: Rustはthiserror+PyErr変換・unwrap/expect回避、Pythonは型ヒント＋Googleスタイルdocstring必須・Ruff line-length=79。

## 7. テスト方針

詳細は `.claude/rules/testing-policy.md`（tests配下で自動ロード）を参照。要点: pyfixest/Rとの数値比較で検証、許容誤差は相対誤差1e-8を基本（手法により例外あり）、engine_tests/api_testsに分離。

## 8. ビルド・テスト・Lintコマンド（想定。リポジトリ雛形作成後に確定）

```bash
# Rust側
cargo test                # engine のユニットテスト
cargo clippy --all-targets -- -D warnings
cargo fmt --check

# Python側（maturinでビルド後）
maturin develop
pytest tests/api_tests
ruff check .
ruff format --check .
```

## 9. バージョニング・CI/CD

- SemVer（`X.Y.Z`）。Z=バグ修正/性能改善、Y=機能追加、X=破壊的変更。
- **例外**: `0.x.x`のプレリリース期間中は、`Y`の変更でも破壊的変更を許容する。
- CIはengine（Rust）とpython側でワークフローファイルを分離（`ci_engine.yml` / `ci_python.yml`）。それぞれ対応するパス配下の変更のみでトリガーし、無駄な実行を防ぐ。
- マルチプラットフォーム（Linux/macOS/Windows）向けwheelビルド・配布は`cd_release.yml`（maturin-action想定）。
- mkdocsドキュメントは`cd_docs.yml`でGitHub Pagesに自動デプロイ。

## 10. ドキュメント運用

- **mkdocs** + **GitHub Pages**。GitHub Actionsでビルド・デプロイを自動化。
- `plan.md`や仕様書などの内部ドキュメントも`docs/planning/`配下に格納する。mkdocsのnavには含めない（非公開ナビゲーション）が、リポジトリ自体がMITでpublicなため、**ソースとしては誰でも閲覧可能**という前提で運用する（ユーザー確認済み）。

## 11. 開発環境

- `.devcontainer/`（`devcontainer.json` / `Dockerfile` / `docker-compose.yml`）で開発環境を統一。
- ベースイメージ: `python:3.14-slim-bookworm`。Rust（stable、clippy/rustfmt/llvm-tools）、uv、R（fixest/plm/ivreg/jsonlite、`benchmark/`のベンチマーク生成用）を導入済み。
- Claude Code CLIはdevcontainer.jsonの`ghcr.io/anthropics/devcontainer-features/claude-code`featureで導入（Dockerfile側での重複インストールはしない）。`gh`（GitHub CLI）は`ghcr.io/devcontainers/features/github-cli`featureで導入（`/cicd`等のコマンドが前提とするため）。
- **トークン消費を抑えるための除外設定**: `.claude/settings.json`の`permissions.deny`/`ask`で、lockファイル・`target/`・`.venv/`・ベンチマークのフィクスチャJSON・GitHub Copilot用設定（`.github/agents/` `.github/instructions/`、メンテナンスが最新に追いついていない可能性があるため）等を除外している。
- 詳細は`.claude/settings.json`を参照。

## 12. 対象プラットフォーム・Pythonバージョン

- OS: Linux（manylinux）, macOS（Apple Silicon / Intel）, Windows
- Python: **3.12以上**。CIでのビルド・テスト対象は **3.12 / 3.13 / 3.14** の3バージョン。開発環境（devcontainer）は3.14を使用。

## 13. 今後の検討事項（未確定）

- IO手法（動学ゲーム等）で必要になる数値最適化ライブラリの選定（argmin, ipopt-rs等を比較検討予定。線形代数はfaerで決定済み、これは別途MLE等の数値最適化用）
- Phase 1（OLS等）の詳細タスク分解
- `estimator-scaffold`スキル（engine/engine_pybind/python_packageの配線パターンのテンプレート化）: 手法によって内部実装が大きく異なるため、OLS実装後に実コードから抽出する形で作成する。今は作らない。
- `econometrics-notes`スキル（手法ごとの数式・実装ノウハウ資料）: 着手する手法ごとにその都度作成する。今は作らない。

## 14. 関連ファイル

- 方針書: `docs/planning/plan.md`（本リポジトリの正式な方針ドキュメント）
- 引き継ぎメモ: `docs/planning/handover-1.md`（方針決定までの経緯）、`docs/planning/handover-2.md`（Claude Code設定〜Phase1 OLSのIssue化までの経緯）
- Phase1 OLSのタスク: `docs/planning/create_phase1_ols_issues.sh` で作成したGitHub Issue群を参照
