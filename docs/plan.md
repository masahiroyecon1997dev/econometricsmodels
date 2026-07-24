# econometricsmodels 方針書

## 1. 概要

| 項目 | 内容 |
|---|---|
| 名称 | econometricsmodels |
| 目的 | 統計・計量経済学の分析手法を提供するPython API |
| 用途 | 自作の分析GUIアプリ「economicon」のエンジンとして使用 |
| 技術スタック | Rust + PyO3（Python拡張） |
| ライセンス | MIT License |
| 公開先 | PyPI |

## 2. 技術方針

- **データ入力は polars のみ**に限定する。pandas 等の他形式は受け付けない。
- polars の Arrow メモリレイアウトを利用し、**Arrow のゼロコピー**でRust側にデータを渡す。コピーによるメモリ・速度のロスを避ける。
- 計算コアは Rust で実装し、高速化を図る。Python 側は PyO3 によるバインディング層として薄く保つ。

## 3. API設計思想

- formula指定（`y ~ x1 + x2` のような文字列パース方式）は採用しない。
- 変数は **List で渡す**方式とする（例: `y=["y_col"], x=["x1", "x2"]` のような形）。
- 推定オプションは **オブジェクト（設定用のクラス／構造体）で渡す**方式とする。
- スクリプト・プログラムからの呼び出しやすさ（型補完、バリデーション、動的組み立てのしやすさ）を優先する。

## 4. 実装スコープと優先順位

「基礎から積み上げる」順に段階実装する。一度にすべて実装せず、フェーズ／タスク単位で細分化して進める。

1. **Phase 1（基礎回帰）**: OLS, 区分回帰, WLS
2. **Phase 2（一般化・離散選択）**: GLS, Logit, Probit, Tobit
3. **Phase 3（操作変数）**: IV（2SLS, GMM）
4. **Phase 4（パネルデータ）**: FE（固定効果）, RE（変量効果）
5. **Phase 5（因果推論）**: DID, RDD
6. **Phase 6（IO手法）**: ロジット, Nested Logit, Random Coefficient Logit, シングルエージェントモデル, 静学ゲーム, 動学ゲーム
7. **Phase 7（後回し・時系列）**: ARCH, GARCH, VAR

各フェーズは前段の実装を基盤として進める。フェーズ内のタスク分解は、リポジトリ構成確定後（本方針書の第11章）に着手時にあわせて行う。

## 5. 対象プラットフォーム・Pythonバージョン

- OS: Linux（manylinux）, macOS（Apple Silicon / Intel）, Windows
- Python: **3.12以上**

## 6. ドキュメント

- **mkdocs** で作成する。
- ホスティングは **GitHub Pages**。
- GitHub Actions でビルド・デプロイを自動化する。
- `plan.md` や仕様書などの内部ドキュメントも `docs/` 配下（`docs/planning/`）に格納する。公開ナビゲーションに含めるかは mkdocs 側の nav 設定で個別に制御する。

## 7. テスト方針

- 既存の検証済みパッケージとの数値比較によりパラメータ推定値の正しさを確認する。
  - **pyfixest**（Python、固定効果推定等）
  - **R の各種パッケージ**（例: fixest, plm, AER, ivreg 等、手法に応じて選定）
- 各推定手法の実装時に、対応するリファレンス実装との比較テストをセットで用意する。
- テストは `engine`（Rustロジック）と `python_package`（Python API）で分離する（詳細は第11章）。

## 8. バージョニング規則

SemVer（`X.Y.Z`）に準拠する。

- **Z**: バグ修正・パフォーマンス改善
- **Y**: 機能追加
- **X（メジャー）**: 後方互換性のない破壊的変更
- 例外: `0.x.x` のプレリリース期間中は、`Y` の変更でも破壊的変更を許容する。

## 9. CI/CD

- GitHub Actions を使用する。
- パッケージバージョン管理を行う。
- **CIは engine（Rust）と python_package/engine_pybind（Python）でワークフローファイルを分離**する。
  - `ci_engine.yml`: `cargo test` / `clippy` / `fmt`。`engine/` 配下の変更をトリガーとする。
  - `ci_python.yml`: `pytest` / `Ruff`。`python_package/` `engine_pybind/` 配下の変更をトリガーとする。
  - 分離することで、Rust側とPython側のステータス（特にRuffの結果）を個別に確認でき、無駄な実行も減らせる。
- テストを自動実行し、デグレ（既存機能の劣化）を防止する。
- マルチプラットフォーム（Linux / macOS / Windows）向けの wheel ビルド・配布パイプラインを構築する（`cd_release.yml`、maturin-action想定）。
- mkdocs ドキュメントの GitHub Pages への自動デプロイを行う（`cd_docs.yml`）。

## 10. 公開方針

- ライセンス: MIT License
- 公開先: PyPI

## 11. リポジトリ構成・crate/module分割案

AIによる開発を前提に、役割が名前から直感的にわかるフォルダ名を採用する。`engine`（純粋Rustロジック）と `engine_pybind`（PyO3バインディング層）を分離する構成とする。

```
econometricsmodels/
├── Cargo.toml                  # Workspaceルート
├── pyproject.toml              # maturinの設定（engine_pybindをビルド対象にする）
├── README.md
│
├── .devcontainer/                # 開発コンテナ定義（Rust + Python 3.12 環境を統一）
│   ├── devcontainer.json
│   └── Dockerfile
│
├── engine/                       # 純粋Rustの計算心臓部（PyO3非依存）
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── ols.rs
│       └── fe.rs
│
├── engine_pybind/                # PyO3の薄いバインディング層
│   ├── Cargo.toml                # ※ maturinはここを対象にする
│   └── src/
│       └── lib.rs                # #[pymodule] を定義し engine の関数を呼ぶ
│
├── python_package/                # ユーザーがpip installするPythonの皮
│   └── econometricsmodels/
│       ├── __init__.py            # engine_pybindからのインポート、Polarsラッパーロジック
│       └── py.typed               # 型定義があることを明示する空ファイル
│
├── tests/
│   ├── engine_tests/              # engineの純粋ロジックテスト（cargo test）
│   └── api_tests/                 # pyfixest / R実装との答え合わせテスト（pytest）
│
├── docs/                          # MkDocsドキュメント（GitHub Pages公開）
│   ├── mkdocs.yml
│   └── planning/                  # plan.md・仕様書などの内部ドキュメント
│       ├── plan.md
│       └── specs/
│
└── .github/workflows/
    ├── ci_engine.yml               # Rust: cargo test / clippy / fmt
    ├── ci_python.yml               # Python: pytest / Ruff
    ├── cd_release.yml              # maturin-actionで各OS向けホイールをビルド・PyPI公開
    └── cd_docs.yml                  # mkdocs -> GitHub Pages
```

**分割の理由**:

- `engine` と `engine_pybind` を分けることで、推定ロジックを `cargo test` で高速に単体テストでき、PyO3依存を切り離せる。
- 手法追加時は `engine` 配下に module を足すだけで済み、フェーズ単位の拡張がしやすい。
- CI・フォルダ名の両方で Rust側／Python側の境界が明確になり、AIが開発する際にも役割を誤認しにくい。
- `.devcontainer` により、Rust（stable）+ Python 3.12 + maturin/ruff/pytest 等の開発環境をコンテナで統一する。中身の詳細（拡張機能等）は着手時に別途詰める。

## 12. 今後の検討事項（未確定）

- IO手法（動学ゲーム等）で必要になる数値最適化ライブラリの選定（後日、argmin, ipopt-rs等を比較検討）
- `.devcontainer` の詳細な中身（VSCode拡張機能、追加ツール等）
- Phase 1（OLS等）の詳細タスク分解（本方針書のリポジトリ構成をもとに次段階で着手）