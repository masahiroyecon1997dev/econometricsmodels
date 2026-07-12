# CLAUDE.md

このファイルは、Claude Code がこのリポジトリで作業する際に毎回参照する前提知識です。
詳細な実装ルール（言語別コーディング規約の細則等）や定型作業は今後 `.claude/rules/` `.claude/commands/` `.claude/skills/` に分離予定ですが、現時点ではこのファイルに集約しています。

## 1. プロジェクト概要

| 項目 | 内容 |
|---|---|
| 名称 | econometricsmodels |
| 目的 | 統計・計量経済学の分析手法を提供するPython API |
| 用途 | 自作の分析GUIアプリ「economicon」のエンジンとして使用 |
| 技術スタック | Rust + PyO3（Python拡張） |
| ライセンス | MIT License |
| 公開先 | PyPI |
| 開発体制 | 基本一人開発（Claudeと二人三脚）。Git運用の詳細は5章参照 |

## 2. 絶対に守るべき設計方針（非交渉事項）

以下はユーザーが明示的に決定した設計方針であり、**Claudeが「使いやすさ」等の理由で自己判断により逸脱・変更を提案してはならない**。

- **データ入力はpolarsのみ**。pandas等の他形式は受け付けない。
- **Arrowのゼロコピー**でRust側にデータを渡す。コピーによるメモリ・速度のロスを避ける。
- **formula文字列パース方式（`y ~ x1 + x2`）は不採用**。
  - 変数は **List渡し**（例: `y=["y_col"], x=["x1", "x2"]`）
  - 推定オプションは **オブジェクト（設定用クラス／構造体）渡し**
  - 理由: スクリプト・プログラムからの呼び出しやすさ（型補完、バリデーション、動的組み立て）を優先するため。
- 計算コアはRustで実装し高速化。Python側はPyO3バインディングとして薄く保つ。

これらの変更が必要と思われる場合も、まず提案として提示し、ユーザーの明示的な承認を得てから実装すること。

## 3. リポジトリ構成

```
econometricsmodels/
├── Cargo.toml                  # Workspaceルート
├── pyproject.toml              # maturinの設定（engine_pybindをビルド対象にする）
├── .claudeignore
│
├── .devcontainer/               # Rust + Python 3.14 環境（中身は未確定、TBD）
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

## 5. Git運用

- **コミットメッセージ**: Conventional Commits（`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `ci:` 等）
- **ブランチ戦略**: feature branch + PR必須を基本とし、機能追加はフェーズ単位でブランチを切る（例: `phase1-ols`, `phase1-wls`）
- **マージ**: CIがgreenであることに加え、内容を確認してからmergeする（自動セルフマージはしない）

## 6. コーディング規約

### Rust（engine / engine_pybind）
- エラーハンドリングは **thiserror** で独自エラー型を定義し、`engine_pybind`（PyO3境界）で `PyErr` に変換する。`unwrap`/`expect`はプロトタイプ段階を除き避ける。
- clippy / fmt はCI（`ci_engine.yml`）で強制。警告ゼロを基準とする（具体的なlintレベルは雛形作成時に`clippy.toml`等で確定）。

### Python（python_package）
- **型ヒント必須**。
- **docstringはGoogleスタイルで必須**（全public関数・クラス）。
- Ruffの **line-length は79（PEP8標準）**。ルールセットの詳細は雛形作成時に`pyproject.toml`で確定。

## 7. テスト方針

- 既存の検証済みパッケージとの数値比較でパラメータ推定値の正しさを検証する。
  - **pyfixest**（Python、固定効果推定等）
  - **Rの各種パッケージ**（fixest, plm, AER, ivreg等、手法に応じて選定）
- 許容誤差は **相対誤差 1e-8程度（厳密）を基本方針** とする。ただし、計算方法自体が実装間で異なる手法（例: FEにおけるHausman検定など）は、手法ごとに個別の許容誤差・比較方法を例外として設定する。
- テストは `engine_tests`（cargo test、純粋ロジック）と `api_tests`（pytest、リファレンス実装との答え合わせ）に分離。
- 各推定手法の実装時に、対応するリファレンス実装との比較テストを必ずセットで用意する。

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

- `.devcontainer/`でRust + Python 3.12環境を統一する予定。
- 中身は未確定（詳細は13章参照）。

## 12. 対象プラットフォーム・Pythonバージョン

- OS: Linux（manylinux）, macOS（Apple Silicon / Intel）, Windows
- Python: **3.12以上**

## 13. 今後の検討事項（未確定）

- IO手法（動学ゲーム等）で必要になる数値最適化ライブラリの選定（argmin, ipopt-rs等を比較検討予定）
- `.devcontainer`の詳細な中身
- Phase 1（OLS等）の詳細タスク分解

## 14. 関連ファイル

- 方針書: `docs/planning/plan.md`（本リポジトリの正式な方針ドキュメント。より詳細な経緯・議論は別途引き継ぎメモを参照）