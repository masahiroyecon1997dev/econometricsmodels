# Changelog

このプロジェクトの変更点はこのファイルに記録します。
フォーマットは [Keep a Changelog](https://keepachangelog.com/ja/1.1.0/) に、バージョニングは [Semantic Versioning](https://semver.org/lang/ja/) に準拠します（`0.x.x`のプレリリース期間中は、マイナーバージョンの変更でも破壊的変更を許容します。CLAUDE.md 9章参照）。

## [Unreleased]

## [0.2.0] - 2026-07-25

Phase 1（基礎回帰）に WLS（加重最小二乗法）を追加しました。

### Added

- WLS推定（`WLS` / `WlsResults`）。重み列は`y`/`x`と同列のトップレベル引数`weight`で指定する（analytic weight、正規化不要）
- WLSもOLSと同じ標準誤差オプション（classical / HC0-HC3 / クラスター / HAC）に対応
- mkdocsにWLSのAPIリファレンス・使用例を追加

### Changed

- OLSの決定係数・対数尤度・AIC・BIC・F統計量・F検定p値も、主リファレンス（statsmodels）に加えR独立実装でクロスチェックするようにした（従来は係数・標準誤差のみ）

### Fixed

- WLSの決定係数（R² / 自由度調整済みR²）・対数尤度・AIC・BICが、重みが一様でない場合に系統的に誤っていた不具合を修正
- クラスターロバスト標準誤差の計算が、実行プロセスごとに非決定的（内部のグループ集約がHashMapの反復順序に依存）だった不具合を修正（OLS/WLS共通）

## [0.1.0] - 2026-07-24

初回リリース。Phase 1（基礎回帰）のうち OLS（最小二乗法）のみ実装済みです。

### Added

- OLS推定（classical / HC0-HC3 ロバスト標準誤差 / クラスター標準誤差 / HAC（Newey-West）標準誤差）
- 決定係数（R² / 自由度調整済みR²）、対数尤度、AIC、BIC、Wald F検定
- polars DataFrameを入力とするPython API（`OLS` / `OLSOptions` / `OlsResults`）
- Rust製計算コア（`engine`）とPyO3バインディング（`engine_pybind`）

[Unreleased]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/masahiroyecon1997dev/econometricsmodels/releases/tag/v0.1.0
