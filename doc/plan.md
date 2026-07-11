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

各フェーズは前段の実装を基盤として進める。フェーズ内のタスク分解は着手時に別途行う。

## 5. 対象プラットフォーム・Pythonバージョン

- OS: Linux（manylinux）, macOS（Apple Silicon / Intel）, Windows
- Python: **3.12以上**

## 6. ドキュメント

- **mkdocs** で作成する。
- ホスティングは **GitHub Pages**。
- GitHub Actions でビルド・デプロイを自動化する。

## 7. テスト方針

- 既存の検証済みパッケージとの数値比較によりパラメータ推定値の正しさを確認する。
  - **pyfixest**（Python、固定効果推定等）
  - **R の各種パッケージ**（例: fixest, plm, AER, ivreg 等、手法に応じて選定）
- 各推定手法の実装時に、対応するリファレンス実装との比較テストをセットで用意する。

## 8. バージョニング規則

SemVer（`X.Y.Z`）に準拠する。

- **Z**: バグ修正・パフォーマンス改善
- **Y**: 機能追加
- **X（メジャー）**: 後方互換性のない破壊的変更
- 例外: `0.x.x` のプレリリース期間中は、`Y` の変更でも破壊的変更を許容する。

## 9. CI/CD

- GitHub Actions を使用する。
- パッケージバージョン管理を行う。
- テストを自動実行し、デグレ（既存機能の劣化）を防止する。
- マルチプラットフォーム（Linux / macOS / Windows）向けの wheel ビルド・配布パイプラインを構築する（maturin想定）。
- mkdocs ドキュメントの GitHub Pages への自動デプロイを含む。

## 10. 公開方針

- ライセンス: MIT License
- 公開先: PyPI

## 11. リポジトリ構成・crate/module分割案（一時的）

maturin の mixed Rust/Python レイアウトをベースに、**推定ロジック本体（core）と PyO3バインディング層を分離**する構成を推奨する。

```
econometricsmodels/
├── Cargo.toml                 # workspaceルート
├── crates/
│   ├── core/                  # 純粋Rustの推定ロジック（PyO3非依存、Arrow直接操作）
│   │   ├── src/
│   │   │   ├── ols.rs
│   │   │   ├── wls.rs
│   │   │   ├── ...            # 手法ごとにmodule分割（Phase単位でディレクトリ化も可）
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   └── pybindings/            # PyO3バインディング層（coreを呼び出す薄い層）
│       ├── src/
│       │   └── lib.rs
│       └── Cargo.toml
├── python/
│   └── econometricsmodels/
│       ├── __init__.py
│       └── *.pyi              # 型スタブ
├── tests/
│   ├── rust/                  # cargo testによるcore単体テスト
│   └── python/                 # pyfixest / R比較テスト
├── docs/                      # mkdocs
│   └── mkdocs.yml
├── .github/workflows/
│   ├── ci.yml                  # test + lint
│   ├── release.yml             # maturin build & PyPI publish
│   └── docs.yml                 # mkdocs -> GitHub Pages
├── pyproject.toml
└── README.md
```

**理由**:

- core と pybindings を分けることで、推定ロジックを `cargo test` で高速に単体テストでき、PyO3依存を切り離せる。
- 手法追加時は core 配下に module を足すだけで済み、フェーズ単位の拡張がしやすい。
- Rust側ロジックを将来的に他言語バインディング（例: WASM等）へ展開する余地も残る。

## 12. 今後の検討事項（未確定）

- IO手法（動学ゲーム等）で必要になる数値最適化ライブラリの選定（後日、argmin, ipopt-rs等を比較検討）
- Phase 1（OLS等）の詳細タスク分解（本方針書のリポジトリ構成案をもとに次段階で着手）