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

- **R / statsmodels 風**のインターフェースを採用する。
  - formula指定（例: `y ~ x1 + x2`）による変数指定を基本とする。
  - `fit()` 実行後、`summary()` で係数・標準誤差・検定統計量などを表示する。
- 計量経済学ユーザーに馴染みやすいAPIを優先し、sklearn風の `fit/predict/score` インターフェースは今回のスコープでは採用しない（必要になれば別途検討）。

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

## 5. 対象プラットフォーム

- Linux（manylinux）
- macOS（Apple Silicon / Intel）
- Windows

対応Pythonバージョンの範囲は別途確定する（未決定事項として下記「今後の検討事項」に記載）。

## 6. テスト方針

- 既存の検証済みパッケージとの数値比較によりパラメータ推定値の正しさを確認する。
  - **pyfixest**（Python、固定効果推定等）
  - **R の各種パッケージ**（例: fixest, plm, AER, ivreg 等、手法に応じて選定）
- 各推定手法の実装時に、対応するリファレンス実装との比較テストをセットで用意する。

## 7. CI/CD

- GitHub Actions を使用する。
- パッケージバージョン管理を行う。
- テストを自動実行し、デグレ（既存機能の劣化）を防止する。
- マルチプラットフォーム（Linux / macOS / Windows）向けの wheel ビルド・配布パイプラインを構築する（maturin想定）。

## 8. 公開方針

- ライセンス: MIT License
- 公開先: PyPI

## 9. 今後の検討事項（未確定）

- 対応Pythonバージョンの範囲（例: 3.9以上など）
- ドキュメント整備の方法（Sphinx / mkdocs 等）とホスティング先
- リポジトリ構成・crate/moduleの分割案
- Phase 1（OLS等）着手時の具体的なタスク分解
- バージョニング規則（SemVer等）とリリースフロー
- IO手法（動学ゲーム等）で必要になる数値最適化ライブラリの選定