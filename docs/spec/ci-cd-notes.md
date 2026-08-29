# CI/CD・セキュリティ運用ノート

CI/CDワークフロー構成・既知の脆弱性対応方針。特定の推定手法に限定しない、プロジェクト共通の記録。
バージョニング・ワークフローファイル分割の全体方針はCLAUDE.md 9章を参照。

## CI/CDワークフロー

- **`ci_engine.yml`**（`engine`＝純粋Rustの品質検証、`engine/**`・`Cargo.toml`/`Cargo.lock`・
  ワークフローファイル自体をトリガー）:
  - `test`ジョブ: `cargo fmt -p engine --check` → `cargo clippy -p engine --all-targets -- -D warnings`
    → `cargo test -p engine`。`engine_pybind`は対象外（PyO3非依存で完結させる責務分離のため、
    `engine_pybind`側は`ci_python.yml`が担当）。
  - `audit`ジョブ: workspace全体の`Cargo.lock`を`cargo-audit`で検証する。`rustsec/audit-check`
    アクションは不採用（`cargo audit --json`の出力にANSI制御文字が混ざると`JSON.parse()`が
    失敗する既知の不具合が長期未解決のため）。テキスト出力のまま`cargo audit`を直接実行する。
- **`ci_python.yml`**（`python_package`/`engine_pybind`の品質検証、3ジョブ、
  `python_package/**`・`engine_pybind/**`・`pyproject.toml`・`uv.lock`・`tests/**`を
  トリガー。`engine/**`は含めない）:
  - `test`ジョブ（Python 3.12/3.13/3.14マトリクス）: `uv sync --locked --group test` →
    `uv run maturin develop` → `pytest tests` → `ruff check .` → `ruff format --check .`。
    `engine_pybind`はabi3を使っていないためPythonマイナーバージョンごとに別ビルドが必要。
  - `engine_pybind-lint`ジョブ: `cargo fmt -p engine_pybind --check` →
    `cargo clippy -p engine_pybind --all-targets -- -D warnings`。
  - `pip-audit`ジョブ（Python 3.12固定）: `test`グループのみ対象。
- **`cd_release.yml`**（Linux/macOS/Windows向けwheelビルド、タグpush（`v*`）+
  `workflow_dispatch`のみ。PR毎には回さない）: ビルド対象Pythonは
  `-i python3.12 -i python3.13 -i python3.14`を明示指定（`--find-interpreter`は未サポート
  バージョンまで検出するため不採用）。
- **`cd_docs.yml`**: mkdocsドキュメントのGitHub Pagesへの自動デプロイ。
- **`dependabot.yml`**（`cargo`・`uv`・`github-actions`の3エコシステム）: `"pip"`ではなく
  **`"uv"`エコシステム**を採用（uv専用の`package-ecosystem`。`test`/`benchmark`/`dev`/`docs`
  全依存グループが更新対象になる）。`cargo audit`/`pip-audit`（CI実行時点のロックファイル検証）と
  Dependabot（レジストリの継続監視・PR自動生成）は補完関係で、統合・置き換えはしない。
- **`benchmark_ols.yml`**: `performance/compare_performance.py`の定期実行（タグpush +
  手動実行のみ、フルスイープが数分かかるため毎PR/週次は見送り）。結果整形は
  `performance/render_performance_summary.py`として分離し、
  `>> "$GITHUB_STEP_SUMMARY"`でjob summaryに出力する。リポジトリルートから
  `python -m performance.<...>`で実行する（Initiative A のパッケージ化に伴う）。
- 全ワークフローでアクションをコミットSHAで固定する（サプライチェーン攻撃対策）。

## セキュリティ（既知の脆弱性・非メンテナンス依存）

`cargo audit`が検知する既知の脆弱性は、`.cargo/audit.toml`のignore listで上流待ちとして保持している
（`allow-list`＝無視してよいという判断ではなく、「上流待ちの既知課題でci_engine.ymlをブロックしない」
ための措置。上流の対応バージョンが公開され次第、該当エントリを削除すること）。

- **`quick-xml`（RUSTSEC-2026-0194/0195、severity 7.5 high）**: 経路は
  `polars → polars-error → object_store → quick-xml`。`polars`自体の新バージョン待ち。
  **実際にはビルドに含まれない**（`object_store`のクラウドストレージ機能はオプション依存で
  有効化していない。`cargo build`のログに一度もコンパイルが出現せず、`cargo tree -p object_store`
  も空を返すことを確認済み）。`cargo audit`は機能フラグを考慮せず`Cargo.lock`を丸ごとスキャンする
  ため、実際にコンパイルされない依存でも警告に含まれる。
- **`bincode`/`paste`（unmaintained警告）**: それぞれ`polars`/`faer`待ち。

## 参照

- `.cargo/audit.toml`: 上記ignore listの実体。
- CLAUDE.md 9章: バージョニング・ワークフローファイル分割の全体方針。
