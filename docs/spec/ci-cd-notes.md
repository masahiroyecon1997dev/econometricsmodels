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
    `uv run maturin develop` → **import時間チェック** → `pytest tests` → `ruff check .` →
    `ruff format --check .`。
    `engine_pybind`はabi3を使っていないためPythonマイナーバージョンごとに別ビルドが必要。
    import時間チェック（`python -m performance.check_import_time`）は
    直前の`maturin develop`（デバッグ）拡張をそのまま使い、`import econometricsmodels`
    −`import polars`の差分（min of 10）が50msを超えたらfailさせる（パッケージ健全性
    メトリクスの節・`docs/performance/package-health.md`参照）。
  - `engine_pybind-lint`ジョブ: `cargo fmt -p engine_pybind --check` →
    `cargo clippy -p engine_pybind --all-targets -- -D warnings`。
  - `pip-audit`ジョブ（Python 3.12固定）: `test`グループのみ対象。
- **`cd_release.yml`**（Linux（manylinux/musllinux × x86_64/aarch64の4種）/macOS/Windows向けwheelビルド、タグpush（`v*`）+
  `workflow_dispatch`のみ。PR毎には回さない）: ビルド対象Pythonは
  `-i python3.12 -i python3.13 -i python3.14`を明示指定（`--find-interpreter`は未サポート
  バージョンまで検出するため不採用）。各ビルドジョブは`Build wheels`直後に
  **wheelサイズ記録ステップ**（`python -m performance.measure_wheel_size dist`）
  を持つ。ビルド済みwheelのサイズ（圧縮/展開後/うち`.so`|`.pyd`）を
  ジョブサマリーにMarkdown表で出すだけで、**リリースはゲートしない**。linux
  x86_64 manylinuxジョブのみ`--baseline docs/performance/package-health.md --warn-pct 10`を
  渡し、展開後サイズが同ファイルの最新記録行比+10%超なら`::warning::`を出す
  （failはしない）。記録の追記は手動（タグpushはdetached HEADでCIからの
  auto-commitが脆いため。`docs/performance/package-health.md`参照）。
  `linux`ジョブは4要素のmatrix（manylinux x86_64 / musllinux x86_64 / manylinux aarch64 /
  musllinux aarch64）。aarch64はQEMUではなく**ネイティブARMランナー（`ubuntu-24.04-arm`、
  publicリポジトリは無料）**でビルドする（polars一式を含む大きな`.so`のリリースビルドは
  エミュレーション下では極端に遅くなるため）。musllinuxは`manylinux: musllinux_1_2`。
  `fail-fast: false`で1要素の失敗が他をキャンセルしない。artifact名は
  `wheels-linux-<target>-<libc>`（`publish-pypi`は`wheels-*`パターンで全て回収）。
  追加wheelの動作確認（import・推定のスモークテスト）は行わない（macOS/Windowsも同様にビルドのみ）。
- **`cd_docs.yml`**: mkdocsドキュメントのGitHub Pagesへの自動デプロイ。
- **`dependabot.yml`**（`cargo`・`uv`・`github-actions`・`docker`・`devcontainers`の5エコシステム。
  `docker`は`.devcontainer/Dockerfile`のベースイメージ、`devcontainers`は`devcontainer.json`の
  featureと`devcontainer-lock.json`が対象）: `"pip"`ではなく
  **`"uv"`エコシステム**を採用（uv専用の`package-ecosystem`。`test`/`benchmark`/`dev`/`docs`
  全依存グループが更新対象になる）。`cargo audit`/`pip-audit`（CI実行時点のロックファイル検証）と
  Dependabot（レジストリの継続監視・PR自動生成）は補完関係で、統合・置き換えはしない。
- **`check_r_updates.yml`**（毎週月曜 + `workflow_dispatch`）: DependabotがCRANを扱えないため、
  `.devcontainer/Dockerfile`で`remotes::install_version()`によりピン留めしたRパッケージと
  R本体の更新を確認する（ロジックは標準ライブラリのみの`.github/scripts/check_r_updates.py`）。
  - Rパッケージ: Dockerfileのピン留め版 vs CRAN `src/contrib/PACKAGES`の最新版。版比較は
    `-`と`.`を同値に扱う（Rの規則。`0.6.8`＝`0.6-8`）。
  - R本体: APT（`bookworm-cran40`）の索引に最新版しか載らず、版を固定できない。そのため
    基準を「フィクスチャJSONの`_meta.r_version`の最大値（凍結値の生成に使ったR）」とし、
    APT索引の`r-base`最新版と比較する。
  - 更新があれば固定タイトルのIssue（`dependencies`/`setup`/`testing`ラベル）を自動起票し、
    未クローズの同名Issueがあれば本文を更新する。すべて最新になれば自動クローズする。
    Dependabotの`cooldown`相当の待機は設けていない（通知のみで、更新は人が判断するため）。
  - 更新時の手順: Dockerfileの版を上げ → devcontainer再構築 →
    `python -m benchmark.regenerate_all --fixtures-only` → フィクスチャ差分確認・`pytest tests`。
    メジャー更新は数値が変わりうる（例: marginaleffects 1.0.0はデルタ法の`std_err`の数値微分が
    変わり、旧版との相対差が最大1.8e-3動いた）ため、機械的にマージしない。
- **`benchmark_performance.yml`**: `performance/compare_<method>.py`（手法非依存の
  計測ハーネス `performance/_perf_harness.py`＋手法固有アダプタ）を手法ごとの
  matrixジョブ（`method: [ols, wls, ...]`、`fail-fast: false`）で定期実行
  （タグpush + 手動実行のみ、フルスイープが数分かかるため毎PR/週次は見送り）。
  結果整形は `performance/render_performance_summary.py`として分離し、
  `>> "$GITHUB_STEP_SUMMARY"`でjob summaryに出力する。リポジトリルートから
  `python -m performance.<...>`で実行する（Initiative A のパッケージ化に伴う）。
  手動でのローカル実測サマリーは `docs/performance/<method>.md`に記録する
  （生成JSONは`docs/performance/results/`、`.gitignore`対象）。
- 全ワークフローでアクションをコミットSHAで固定する（サプライチェーン攻撃対策）。

## パッケージ健全性メトリクス（import時間・インストール容量）

`benchmark/`・`performance/`が手法ごとの数値精度・推定速度をカバーするのに対し、
パッケージとしての健全性（`import econometricsmodels`の所要時間、`pip install`時の
容量）をCIで監視する。正本は`docs/performance/package-health.md`
（ベースライン実測値・監視の仕組み・サイズ記録表）。要点のみ以下に再掲する。

- **監視は「絶対値」ではなく「自前の差分」**。import時間の約98%、インストール容量の
  約85%は`polars`/`polars-runtime-32`で、CLAUDE.md 2章で「polarsのみ」と設計確定
  済みのため削減対象外。自前コードの寄与は現状ほぼゼロなので、polars分を相殺した
  差分の回帰だけを見る。
- **import時間**（`performance/check_import_time.py`）: `ci_python.yml`の`test`
  ジョブで毎PR実行・**failゲート**。判定値は`import econometricsmodels`−
  `import polars`（両者サブプロセスmin of 10）。閾値50ms（狙いは100ms級の回帰
  ＝stray import・import時実処理・将来の数値最適化ライブラリのeager import検出）。
  デバッグ/テストビルドで測る（`.so`が大きくdlopenコストは悲観側に出るため
  ガードとして安全側。ベースライン値は「デバッグビルドの上限値」）。
- **wheel/`.so`サイズ**（`performance/measure_wheel_size.py`）: `cd_release.yml`の
  各ビルドジョブで**記録のみ**（ゲートしない）。linux x86_64 manylinuxを代表値として、
  展開後サイズが`package-health.md`の最新記録比+10%超なら`::warning::`（musllinux・aarch64は
  `.so`サイズが異なるため記録のみで、基準は混ぜない）。記録表への
  追記は手動。
- **別issue候補**（本監視のスコープ外）: `[profile.release]`に`strip = true`
  （即−11MB）、`lto`/`codegen-units`/`panic = "abort"`。数値性能への影響を
  `.claude/rules/rust-style.md`「パフォーマンス」節の「実測してから決める」に
  従って計測してから採否判断する。

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
- `docs/performance/package-health.md`: パッケージ健全性メトリクスの正本
  （ベースライン実測値・サイズ記録表）。
- `performance/check_import_time.py` / `performance/measure_wheel_size.py`:
  import時間・wheelサイズの計測スクリプト。
- CLAUDE.md 9章: バージョニング・ワークフローファイル分割の全体方針。
