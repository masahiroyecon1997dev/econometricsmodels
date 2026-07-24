# Changelog

このプロジェクトの変更点はこのファイルに記録します。
フォーマットは [Keep a Changelog](https://keepachangelog.com/ja/1.1.0/) に、バージョニングは [Semantic Versioning](https://semver.org/lang/ja/) に準拠します（`0.x.x`のプレリリース期間中は、マイナーバージョンの変更でも破壊的変更を許容します。CLAUDE.md 9章参照）。

## [Unreleased]

初回リリース（0.1.0）に向けた変更点。Phase 1（基礎回帰）のうち OLS（最小二乗法）のみ実装済みです。

### Added

- OLS推定（classical / HC0-HC3 ロバスト標準誤差 / クラスター標準誤差 / HAC（Newey-West）標準誤差）
- 決定係数（R² / 自由度調整済みR²）、対数尤度、AIC、BIC、Wald F検定
- polars DataFrameを入力とするPython API（`OLS` / `OLSOptions` / `OlsResults`）
- Rust製計算コア（`engine`）とPyO3バインディング（`engine_pybind`）

[Unreleased]: https://github.com/masahiroyecon1997dev/econometricsmodels/commits/main
