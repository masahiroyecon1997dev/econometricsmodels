# CLAUDE.md

このファイルは、Claude Code がこのリポジトリで作業する際に毎回参照する前提知識です。
言語別の詳細なコーディング規約・テスト方針は `.claude/rules/`（パス指定で自動ロード）に、定型作業は `.claude/skills/` に、コードレビューは `.claude/agents/` のサブエージェントに分離している。重複記載を避けるため、このファイルには全体像・非交渉事項・各詳細ファイルへの参照のみを置く。

手法固有の実装ノウハウ（設計判断の理由・既知の落とし穴等）は、対応する `engine/src/<系統>/CLAUDE.md` 等のネストCLAUDE.md（該当ディレクトリ配下のファイルを読み書きしたときだけ自動ロード）に置く。現状は `linear`（OLS/WLS）系統が`engine`/`engine_pybind`/`python_package`の3箇所、`nonlinear`（Logit/Probit）系統が`engine_pybind`/`python_package`の2箇所（`engine/src/nonlinear/`はまだ未作成）で作成済み。他系統・未作成箇所は実装着手時にその都度作成する（4章参照）。

## 1. プロジェクト概要

| 項目 | 内容 |
|---|---|
| 名称 | econometricsmodels |
| 目的 | 統計・計量経済学の分析手法を提供するPython API |
| 用途 | スクリプト・アプリケーションから呼び出して使う分析エンジン |
| 技術スタック | Rust + PyO3（Python拡張） |
| 線形代数クレート | faer（pure Rust、システムBLAS/LAPACK非依存） |
| ライセンス | MIT License |
| 公開先 | PyPI |
| 開発体制 | 基本一人開発。Git運用の詳細は5章参照 |

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
- **検定・診断の公開形**: 推定量そのものの妥当性・識別に属する検定（Sargan/J・Hausman・全体F/Wald/LR等）は`fit()`時に計算してプロパティで公開し、推定後の事後診断（White・Breusch-Godfrey等、利用者の選択が入るもの）は検定ごとの独立メソッドとして公開する（事後診断は`fit()`で自動計算しない）。事後診断メソッドの返り値は検定共通のfrozen dataclass（`DiagnosticResult`: `statistic`/`p_value`/`df`/`df_denom`/`distribution`＋`to_dict()`。検定固有の項目は継承型に足す。例: `OLSResults.white_test()`の`WhiteTestResult`）で、ドット記法で読めJSON化は`to_dict()`で行う。キーが利用者の列名で決まる結果（`params`等）と、表形式の`coef_table()`/`marginal_effects()`は`dict`・`list[dict]`のまま、`predict()`は`list[float]`。詳細は`docs/spec/inference-conventions.md`6章。

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
│   └── src/
│       ├── lib.rs
│       ├── linear/ nonlinear/ iv/ panel/           # 系統別。手法は最初1ファイル、肥大化したらディレクトリ（linear/ols/ 等）
│       └── shared/                                 # 複数の系統が共有するコード。error.rs・inference.rs・linear_algebra.rs・parallelism.rs・validation.rs・design_matrix.rs、計算部品（共分散・Wald検定・最小二乗・適合度・クラスターのグループ化）。各系統の common.rs とは別
│
├── engine_pybind/                # PyO3の薄いバインディング層
│   └── src/lib.rs                # #[pymodule] を定義し engine の関数を呼ぶ
│
├── python_package/econometricsmodels/
│   ├── __init__.py               # engine_pybindからのインポート、Polarsラッパー
│   └── py.typed
│
├── tests/                          # pytest（statsmodels/linearmodels / R実装との答え合わせ）
│   ├── conftest.py _assertions.py _helpers.py _tolerances.py  # 共有（系統によらず全テストが使う）
│   ├── linear/ nonlinear/ iv/ panel/  # 系統別サブディレクトリ（benchmark/ と同じ grain）。test_<手法>*.py（Tobitはnonlinear/）
│   └── fixtures/benchmarks/        # 固定CSV＋リファレンスJSON（コミット済み成果物）
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
│   ├── guide/                     # 利用者向けの英語の公開ページ（navに掲載。受け付けるデータ・バリデーション・推論の慣習・検証・性能。詳細は13章）
│   ├── spec/                      # 実装済み手法の数式・API仕様の正本（詳細は13章）
│   ├── performance/               # 手法別の性能比較の開発ノート（日本語。計測方法論・既知の限界。結果表は公開ページ側。results/にJSON）
│   └── planning/                  # 実装途中の設計ノート（詳細は9章。着手中の手法が無いときは存在しない）
│
└── .github/workflows/
    ├── ci_engine.yml               # cargo test / clippy / fmt（pushはengine/配下のみ、PRは常に実行）
    ├── ci_python.yml               # pytest / Ruff（pushはpython_package/ engine_pybind/ 等の配下のみ、PRは常に実行）
    ├── cd_release.yml              # maturin-actionでのマルチOSホイールビルド・PyPI公開
    └── cd_docs.yml                  # mkdocs → GitHub Pages
```

`engine`と`engine_pybind`を分離しているのは、推定ロジックをPyO3非依存で`cargo test`できるようにするため。手法追加時は基本的に`engine`配下にmoduleを足すだけでよい。

## 4. 実装フェーズと進め方

「基礎から積み上げる」順に段階実装する。**一度に全フェーズ／全手法を実装しない**。フェーズ・タスク単位に細分化して、1つずつ完了させてから次に進む。

フェーズ構成・手法の割り当て・着手順序はGitHub Issueを正本とする（このファイルには複製しない）。現在の実装状況はgit logも参照する。

**新しい手法の実装に着手する前に**、既存の類似手法（系統内、無ければ直近で実装した手法）の実装を`Explore`エージェント（読み取り専用の探索用サブエージェント）で調査してから着手する。設計判断・実装パターンを毎回メインセッションでファイルを読み込んで再発見するコストを避けるため。調査結果は各系統のネストCLAUDE.md（1章参照）に集約されているため、まずそちらを確認し、記載が無い・古い場合にのみ既存コードの探索に切り替える。

## 5. Git運用

個人開発のため、リリース単位のブランチに作業を直接コミットする運用とする。

- **コミットメッセージ**: Conventional Commits（`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `ci:` 等）
- **ブランチ構成**:
  - `main`: 公開済み（PyPIリリース済み）の状態。タグ付けの対象。
  - `dev`: mainへマージする前の検証用。リリース作業中に発生した緊急のバグ・脆弱性対応の受け皿、およびDependabotの更新PRの向け先。
  - `release/vX.Y.Z`: 次リリース用の作業ブランチ。リリースごとに作成し、機能実装・単発のfix/CI変更・ドキュメント更新を含む通常の作業はここに直接コミットする（手法・フェーズ単位のfeature branchは切らない）。
- **緊急対応**: 脆弱性等で`release/*`と切り離して`dev`へ先に反映したい場合のみ、`fix/<内容>`等の短命ブランチを切り、`dev`へのPRでマージする。`dev`に入った変更は`release/*`へ取り込む（`git merge origin/dev`）。
- **保護設定**: `main`と`dev`へはローカルからの直pushを禁止し、pull request経由のみとする（ブランチ保護）。
- **リリースの流れ**: `release/vX.Y.Z` → `dev`（PR）→ `main`（PR）→ タグpush（詳細は`.claude/skills/release-publish/SKILL.md`）。
- **マージ**: CIがgreenであることに加え、内容を確認してからmergeする（自動セルフマージはしない）。
- **GitHub Issue**: リポジトリがpublicなため、英語で記述する（README・MkDocsと同様の理由）。対象はIssue本文のみで、セッション内の会話・コミットメッセージは対象外（引き続き日本語）。
- **リファクタリング・テスト拡充の候補**: コード解説や実装の過程で気づいた候補は、メモファイルに溜めず、ユーザーの承認を得てGitHub Issueに直接起票する（関連する項目は1つのIssueにまとめる。既存Issueで扱える場合はコメントで追記する）。

## 6. コーディング規約

詳細は `.claude/rules/rust-style.md`（engine/engine_pybind配下で自動ロード）、`.claude/rules/python-style.md`（python_package配下で自動ロード）を参照。要点: Rustはthiserror+PyErr変換・unwrap/expect回避、Pythonは型ヒント＋Googleスタイルdocstring必須・Ruff line-length=79。

**コメント・ドキュメントでの参照方針**: GitHub Issue番号、および`docs/planning/specs/`配下（各手法の設計ノート・進捗記録等、項目の追加・変更・削除が起こりうる内部管理ドキュメント。13章参照）への参照は、コード（Rust/Python問わず）・`Cargo.toml`等の設定ファイル・仕様書・各CLAUDE.md（ネストCLAUDE.md含む）のコメント/説明文に書き込まない。git log/GitHub側で常に追跡可能な経緯を重複記録する必然性がなく、かつ内部ドキュメントは変更・削除されうるためリンク切れ・文脈不明のノイズになる。一方、`docs/spec/`配下（実装済み手法の正式仕様書、13章参照）の節への参照は、今後も同期すべき生きた契約であるため許可する（例:「詳細は`docs/spec/ols-spec.md`「テスト」参照」）。過去形の由来説明からIssue番号を削除する際は、そこに書かれている理由（なぜそう実装したか）の文章は残す。複数Issueにまたがる経緯で、その変遷自体が非自明な価値を持つ場合はIssue番号を使わず平易な文章で要約し、単なる経緯の記録に過ぎない場合は削除する。

## 7. テスト方針

詳細は `.claude/rules/testing-policy.md`（tests配下で自動ロード）を参照。要点: statsmodels/linearmodels/Rとの数値比較で検証（pyfixestは性能比較専用）、許容誤差は相対誤差1e-8を基本（手法により例外あり）、engineの単体テストはソース内`mod tests`、`tests/`はpytestに分離。

## 8. バージョニング・CI/CD

- SemVer（`X.Y.Z`）。Z=バグ修正/性能改善、Y=機能追加、X=破壊的変更。
- **例外**: `0.x.x`のプレリリース期間中は、`Y`の変更でも破壊的変更を許容する。
- CIはengine（Rust）とpython側でワークフローファイルを分離（`ci_engine.yml` / `ci_python.yml`）。`main`へのpushは対応するパス配下の変更のみでトリガーし、無駄な実行を防ぐ。`pull_request`はpathsフィルタを付けず常に実行する（必須ステータスチェックとして登録しており、フィルタがあるとパスに触れないPRでマージ不能になるため）。
- マルチプラットフォーム（Linux/macOS/Windows）向けwheelビルド・配布は`cd_release.yml`（maturin-action想定）。
- mkdocsドキュメントは`cd_docs.yml`でGitHub Pagesに自動デプロイ。

## 9. ドキュメント運用

- **mkdocs** + **GitHub Pages**。GitHub Actionsでビルド・デプロイを自動化。
- 仕様書などの内部ドキュメントも`docs/`配下（実装途中の設計ノートは`docs/planning/`）に格納する。mkdocsのnavには含めない（非公開ナビゲーション）が、リポジトリ自体がMITでpublicなため、**ソースとしては誰でも閲覧可能**という前提で運用する（ユーザー確認済み）。

## 10. 開発環境

- `.devcontainer/`（`devcontainer.json` / `Dockerfile` / `docker-compose.yml`）で開発環境を統一。
- ベースイメージ: `python:3.14-slim-bookworm`。Rust（stable、clippy/rustfmt/llvm-tools）、uv、R（fixest/plm/ivreg/AER/censReg/marginaleffects等、`benchmark/`のベンチマーク生成用。Rパッケージは`Dockerfile`で`remotes::install_version()`によりバージョンをピン留めし、新版の有無は`check_r_updates.yml`が週次で確認してIssueで通知する）を導入済み。**旧経緯**: `ivreg`は当初`Dockerfile`が`install.packages()`でインストールを試みていたが実際には失敗し導入されていなかった（依存先`car`→`MatrixModels`が`Matrix>=1.6.0`（→R>=4.4）を要求するが、Debian bookworm標準のr-baseは4.2.2固定でこれを満たせなかった。`install.packages()`はベクタの一部が失敗してもRUNコマンド自体は成功扱いになるため、ビルドは通ってしまいこの状態に気づきにくかった）。CRAN公式のDebian向けAPTリポジトリ（`bookworm-cran40`、実体は最新のRリリースを追従）を追加してR 4.x系（執筆時点で4.5.3）に更新し解消した。現在は`install_version()`で導入済みで、IVのRクロスチェックに使っている。Rパッケージの追加時は`install.packages()`ではなく`install_version()`を使い、導入失敗をビルドエラーとして検知できるようにする。
- Claude Code CLIはdevcontainer.jsonの`ghcr.io/anthropics/devcontainer-features/claude-code`featureで導入（Dockerfile側での重複インストールはしない）。`gh`（GitHub CLI）は`ghcr.io/devcontainers/features/github-cli`featureで導入（`/cicd`等のコマンドが前提とするため）。
- **トークン消費を抑えるための除外設定**: `.claude/settings.json`の`permissions.deny`/`ask`で、lockファイル・`target/`・`.venv/`・ベンチマークのフィクスチャJSON等を除外している。
- 詳細は`.claude/settings.json`を参照。

## 11. 対象プラットフォーム・Pythonバージョン

- OS: Linux（manylinux / musllinux、x86_64 / aarch64の4種）, macOS（Apple Silicon / Intel）, Windows（x64）
- Python: **3.12以上**。CIでのビルド・テスト対象は **3.12 / 3.13 / 3.14** の3バージョン。開発環境（devcontainer）は3.14を使用。

## 12. 今後の検討事項（未確定）

- IO手法（動学ゲーム等）で必要になる数値最適化ライブラリの選定（argmin, ipopt-rs等を比較検討予定。線形代数はfaerで決定済み、これは別途MLE等の数値最適化用）
- 並列化クレート`rayon`の採用: 現時点では未導入。候補箇所・採用判断基準（実測してから決める方針）は`.claude/rules/rust-style.md`「パフォーマンス」節に記載済み。パフォーマンス検討時は都度この基準に照らして採用可否を判断する。
- 実装手法が増えてきた段階で、配線パターンのテンプレート化や手法ごとの実装ノウハウ資料化など、スキルとして切り出す余地がないか随時検討する。

## 13. 関連ファイル

- 仕様書: `docs/spec/`（実装済みの手法ごとの数式・API仕様の正本。method非依存のCI/CD・セキュリティ運用ノートも
  ここに置く、例: `ci-cd-notes.md`・`inference-conventions.md`）、`docs/planning/specs/`（実装途中の手法の設計ノート・実装ノート）。
  ある手法の実装が完了したら、その手法の仕様書は`docs/planning/specs/`から`docs/spec/`へ集約する
  （経緯は削除し理由のみ簡潔に記載、1ファイルにまとめる）。
- 利用者向け横断ガイド: `docs/guide/`（mkdocsのnavに載せる英語の公開ページ）。現状は`inference-conventions.md`（手法別の検定分布・自由度、R/statsmodels/linearmodelsとの違い、診断統計量の読み方）。手法別の一覧表はこの公開ページを正本とし、`docs/spec/inference-conventions.md`には重複させず選択理由・ベンチマーク上の注意のみを置く。新手法の追加時は公開ページの表に1行追加する。
- 性能比較記録: `docs/performance/<method>.md`（`performance/compare_<method>.py`の計測方法論・設計判断・既知の限界・今後の検討を残す日本語の開発ノート。
  計測結果の表は置かない）。生成JSONは`docs/performance/results/`（`.gitignore`対象）。
- **公開ページ（mkdocs nav掲載・英語）の運用ルール**: 検証と性能は、手法が増えたら公開ページにも反映する。
  - `docs/guide/verification.md`: 手法×リファレンス（主・独立クロスチェック）・比較する統計量・許容誤差（`tests/_tolerances.py`が正）・実データ・単一リファレンスの例外。新手法のテスト作成（`/test-new`）の完了条件に含める。
  - `docs/guide/performance.md`（概要・既知の課題・計測条件は手書き、先頭のサマリー表は生成ブロック）と`docs/guide/performance-results.md`（全表、全体が生成物）: **数値は`benchmark_performance.yml`のCI計測値を正とする**（devcontainerの単発計測は使わない）。リリース準備時（`/release`）にリリースブランチで手動実行し、artifactから`python -m performance.render_docs_results`で再生成する。`README.md`「Performance」節の表・例にも同じ数値を丸めて転記しているため、再生成のたびに合わせて更新する（手順は`/release`）。手法を足すときは`benchmark_performance.yml`のmatrixに加えれば、次回の再生成で自動的にページへ現れる。
  - `docs/guide/accepted-data.md`: 受け付ける入力（polarsの`DataFrame`のみ）・列の役割ごとに許可するdtype・値の変換と精度・引数の型（`TypeError`との分担）。許可するdtypeの表は`engine_pybind/src/column_extraction.rs`のdtype検査が正で、検査を変更したらこの表を同時に更新する（許可・拒否の方針を各手法specに複製しない）。
  - `docs/guide/validation.md`: バリデーションの設計思想（欠損値を自動除外しない理由等）と、`ValidationError`/`ComputationError`が出る状況の分類。手法固有のチェック（新しい列引数・オプションの検証、手法固有の`ComputationError`）を追加したら、該当する分類表に1行足す。欠損値・共線列等の共通方針は各手法specに複製せず、このページを参照する。
  - 検証・性能の結果表は日本語ノートや`docs/spec/`に重複させない（公開ページが正本）。

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
