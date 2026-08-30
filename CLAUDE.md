# CLAUDE.md

このファイルは、Claude Code がこのリポジトリで作業する際に毎回参照する前提知識です。
言語別の詳細なコーディング規約・テスト方針は `.claude/rules/`（パス指定で自動ロード）に、定型作業は `.claude/skills/` に、コードレビューは `.claude/agents/` のサブエージェントに分離している。重複記載を避けるため、このファイルには全体像・非交渉事項・各詳細ファイルへの参照のみを置く。

手法固有の実装ノウハウ（設計判断の理由・既知の落とし穴等）は、対応する `engine/src/<系統>/CLAUDE.md` 等のネストCLAUDE.md（該当ディレクトリ配下のファイルを読み書きしたときだけ自動ロード）に置く。現状は `linear`（OLS/WLS）系統が`engine`/`engine_pybind`/`python_package`の3箇所、`nonlinear`（Logit/Probit）系統が`engine_pybind`/`python_package`の2箇所（`engine/src/nonlinear/`はまだ未作成）で作成済み。他系統・未作成箇所は実装着手時にその都度作成する（4章参照）。

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
├── .devcontainer/               # Rust + Python 3.14 環境（詳細は10章）
│   ├── devcontainer.json
│   ├── docker-compose.yml
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
├── tests/                          # pytest（pyfixest / R実装との答え合わせ）
│
├── benchmark/                     # テスト用フィクスチャ生成ツール（Pythonパッケージ。pytestが収集時にimportする）
│   ├── common/                    # 系統横断の共通ヘルパー（DGP・データIO・リファレンス呼び出し・CLI）
│   ├── linear/ nonlinear/ iv/ panel/  # 系統ごと: datasets.py（DGP＋凍結）・references/（リファレンス実装アダプタ＋.R）・fixtures/（generate_*_fixtures.py）
│   └── regenerate_all.py          # 合成データCSV＋全フィクスチャJSONの一括再生成。詳細は.claude/skills/reference-benchmark/
│
├── performance/                   # リファレンス実装との性能比較（benchmark_performance.ymlから実行。pytestとは無関係、statsmodels/linearmodels依存）
│
├── docs/                          # MkDocs（GitHub Pages公開）
│   ├── mkdocs.yml
│   └── planning/                  # plan.md・仕様書（詳細は9章）
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

フェーズ構成・手法の割り当ては `docs/plan.md` 4章を正本とする（このファイルには複製しない）。現在の実装状況・直近の着手順序はgit logおよびGitHub Issueを参照する。

**新しい手法の実装に着手する前に**、既存の類似手法（系統内、無ければ直近で実装した手法）の実装を`Explore`エージェント（読み取り専用の探索用サブエージェント）で調査してから着手する。設計判断・実装パターンを毎回メインセッションでファイルを読み込んで再発見するコストを避けるため。調査結果は各系統のネストCLAUDE.md（1章参照）に集約されているため、まずそちらを確認し、記載が無い・古い場合にのみ既存コードの探索に切り替える。

## 5. Git運用

- **コミットメッセージ**: Conventional Commits（`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `ci:` 等）
- **ブランチ戦略**: feature branch + PR必須を基本とし、機能追加はフェーズ単位でブランチを切る（例: `phase1-ols`, `phase1-wls`）
- **マージ**: CIがgreenであることに加え、内容を確認してからmergeする（自動セルフマージはしない）

## 6. コーディング規約

詳細は `.claude/rules/rust-style.md`（engine/engine_pybind配下で自動ロード）、`.claude/rules/python-style.md`（python_package配下で自動ロード）を参照。要点: Rustはthiserror+PyErr変換・unwrap/expect回避、Pythonは型ヒント＋Googleスタイルdocstring必須・Ruff line-length=79。

## 7. テスト方針

詳細は `.claude/rules/testing-policy.md`（tests配下で自動ロード）を参照。要点: pyfixest/Rとの数値比較で検証、許容誤差は相対誤差1e-8を基本（手法により例外あり）、engineの単体テストはソース内`mod tests`、`tests/`はpytestに分離。

## 8. バージョニング・CI/CD

- SemVer（`X.Y.Z`）。Z=バグ修正/性能改善、Y=機能追加、X=破壊的変更。
- **例外**: `0.x.x`のプレリリース期間中は、`Y`の変更でも破壊的変更を許容する。
- CIはengine（Rust）とpython側でワークフローファイルを分離（`ci_engine.yml` / `ci_python.yml`）。それぞれ対応するパス配下の変更のみでトリガーし、無駄な実行を防ぐ。
- マルチプラットフォーム（Linux/macOS/Windows）向けwheelビルド・配布は`cd_release.yml`（maturin-action想定）。
- mkdocsドキュメントは`cd_docs.yml`でGitHub Pagesに自動デプロイ。

## 9. ドキュメント運用

- **mkdocs** + **GitHub Pages**。GitHub Actionsでビルド・デプロイを自動化。
- `plan.md`や仕様書などの内部ドキュメントも`docs/planning/`配下に格納する。mkdocsのnavには含めない（非公開ナビゲーション）が、リポジトリ自体がMITでpublicなため、**ソースとしては誰でも閲覧可能**という前提で運用する（ユーザー確認済み）。

## 10. 開発環境

- `.devcontainer/`（`devcontainer.json` / `Dockerfile` / `docker-compose.yml`）で開発環境を統一。
- ベースイメージ: `python:3.14-slim-bookworm`。Rust（stable、clippy/rustfmt/llvm-tools）、uv、R（fixest/plm/ivreg/jsonlite、`benchmark/`のベンチマーク生成用）を導入済み。**旧経緯**: `ivreg`は当初`Dockerfile`が`install.packages()`でインストールを試みていたが実際には失敗し導入されていなかった（Issue #171で発覚。依存先`car`→`MatrixModels`が`Matrix>=1.6.0`（→R>=4.4）を要求するが、Debian bookworm標準のr-baseは4.2.2固定でこれを満たせなかった。`install.packages()`はベクタの一部が失敗してもRUNコマンド自体は成功扱いになるため、ビルドは通ってしまいこの状態に気づきにくかった）。CRAN公式のDebian向けAPTリポジトリ（`bookworm-cran40`、実体は最新のRリリースを追従）を追加してR 4.6.1系に更新し解消した。IVのRクロスチェック（`ivreg`）に着手する際は、コンテナ再構築後に`ivreg`が実際に導入されているか（`Rscript -e 'library(ivreg)'`等）を確認してから進める。
- Claude Code CLIはdevcontainer.jsonの`ghcr.io/anthropics/devcontainer-features/claude-code`featureで導入（Dockerfile側での重複インストールはしない）。`gh`（GitHub CLI）は`ghcr.io/devcontainers/features/github-cli`featureで導入（`/cicd`等のコマンドが前提とするため）。
- **トークン消費を抑えるための除外設定**: `.claude/settings.json`の`permissions.deny`/`ask`で、lockファイル・`target/`・`.venv/`・ベンチマークのフィクスチャJSON・GitHub Copilot用設定（`.github/agents/` `.github/instructions/`、メンテナンスが最新に追いついていない可能性があるため）等を除外している。
- 詳細は`.claude/settings.json`を参照。

## 11. 対象プラットフォーム・Pythonバージョン

- OS: Linux（manylinux）, macOS（Apple Silicon / Intel）, Windows
- Python: **3.12以上**。CIでのビルド・テスト対象は **3.12 / 3.13 / 3.14** の3バージョン。開発環境（devcontainer）は3.14を使用。

## 12. 今後の検討事項（未確定）

- IO手法（動学ゲーム等）で必要になる数値最適化ライブラリの選定（argmin, ipopt-rs等を比較検討予定。線形代数はfaerで決定済み、これは別途MLE等の数値最適化用）
- 並列化クレート`rayon`の採用: 現時点では未導入。候補箇所・採用判断基準（実測してから決める方針）は`.claude/rules/rust-style.md`「パフォーマンス」節に記載済み。パフォーマンス検討時は都度この基準に照らして採用可否を判断する。
- 実装手法が増えてきた段階で、配線パターンのテンプレート化や手法ごとの実装ノウハウ資料化など、スキルとして切り出す余地がないか随時検討する。

## 13. 関連ファイル

- 方針書: `docs/plan.md`（本リポジトリの正式な方針ドキュメント。実装フェーズ・手法の割り当てもここが正本）
- 仕様書: `docs/spec/`（実装済みの手法ごとの数式・API仕様の正本。method非依存のCI/CD・セキュリティ運用ノートも
  ここに置く、例: `ci-cd-notes.md`）、`docs/planning/specs/`（実装途中の手法の設計ノート・実装ノート）。
  ある手法の実装が完了したら、その手法の仕様書は`docs/planning/specs/`から`docs/spec/`へ集約する
  （経緯は削除し理由のみ簡潔に記載、1ファイルにまとめる）。

## 14. 実装・テスト・ベンチマーク作成・仕様検討時の確認方針

- 実装・テストコード作成・ベンチマーク作成・仕様検討のいずれの段階でも、判断が分かれる点や設計上の選択肢に気づいたら、**独自判断で埋めずに先にユーザーへ確認する**。
- 特に以下のような場面で確認が必要になりやすい。
  - 参照実装・パラメータ・許容誤差の選定に複数の妥当な候補がある
  - 検証範囲（網羅的に検証するか代表ケースのみか等）が既存の方針から自明に決まらない
  - 既存ドキュメント・issueの記述と、実装時に判明した事実が食い違う
- 疑問点は着手前の計画段階でまとめて確認する。実装中に新たに判明した場合は、その都度確認してから進める（まとめて後から確認する方式は取らない）。

## 15. コンテキスト管理

- **セッション運用**: 異なる手法・異なるフェーズの作業は1つの長いセッションに混在させず、タスクの区切りで`/clear`する。
- **compaction時に保持すべき情報**: 会話が要約される場合、少なくとも以下は要約後も残す。
  - 変更・作成したファイルの一覧
  - 直前に実行した、またはこれから実行する予定のテスト・Lintコマンドとその結果
  - ユーザーとの間で未解決のまま残っている疑問点・確認待ちの判断（14章）
- **調査・大量ファイル読み込みを伴うタスク**（既存実装の調査、複数ファイルにまたがる横断検索等）は`Explore`エージェント等のサブエージェントに委譲し、メインセッションの文脈を汚さない（4章参照）。
- **ベンチマーク/検証スクリプトの出力**: 標準出力には要約（pass/fail・主要指標の差分等）またはファイル書き出し完了メッセージのみを出し、生の実行結果（フルの回帰結果・データフレーム全体等）を垂れ流さない（既存の`benchmark/`配下のスクリプトは実装済み、新規追加時も踏襲する）。
