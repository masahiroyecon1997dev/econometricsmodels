---
paths:
  - engine/**/*
  - engine_pybind/**/*
---

# Rustコーディング規約（engine / engine_pybind）

このファイルは `engine/` `engine_pybind/` 配下で作業する際に自動的に読み込まれる。CLAUDE.mdの非交渉事項（2章）を前提とした、Rust実装の詳細規約。

## 責務分離

- `engine`: 純粋Rustロジック。PyO3に依存しない。
- `engine_pybind`: PyO3の薄いバインディング層。`#[pymodule]`を定義し`engine`の関数を呼ぶだけに留める。計算ロジックを`engine_pybind`に書かない。

## 推定量構造体の設計（全手法共通）

- 各推定量（`OlsEstimator`等）のフィールドは **private** にする。`pub`にして構築後も自由に書き換えられる状態にしない。
- 理由: コンストラクタ（`new()`等）でのバリデーション（次元・欠損値・観測数等）を通過した後にフィールドが書き換えられると、そのバリデーションの意味がなくなる。フィールドをprivateにし、必要な値はgetter経由で公開することで、「バリデーション済みの状態」を構築後も保証する。

## 線形代数

- **faer**（pure Rust）を使用する。システムのBLAS/LAPACKには依存しない。`ndarray-linalg`等、システムBLAS/LAPACKを要求するクレートは使わない。
- Cargo.tomlでfaerのバージョンを明示的に固定する（**0.24.4**で確定。APIが変わることがあるため、アップグレード時は要動作確認）。
- 設計行列が特異になりうるケース（完全な多重共線性等）は、`col_piv_qr`等のpivotありの分解を使い、数値的に検出・処理する。
- 特異性判定の閾値は**絶対閾値ではなく相対閾値**を使う（データのスケールに依存しないようにするため）。具体的な閾値の式は実装時に判断してよい。

## Python境界でのデータ受け渡し（engine_pybind）

- polars DataFrameの受け取りには**pyo3-polars**（`PyDataFrame`）を使う。
- **既知のリスク**: `pyo3-polars`の単体リポジトリ（`pola-rs/pyo3-polars`）は2025年7月にアーカイブ済みで、本体`pola-rs/polars`リポジトリに統合されている。crates.io版`pyo3-polars`とpolars本体リポジトリ内のバージョンにズレがあり、`pyo3`自体のバージョンとの組み合わせでビルドが失敗する事例が報告されている（2026年1月時点）。**リポジトリ雛形作成時、本格的な実装に入る前に`pyo3-polars`込みで一度`cargo build`が通ることを確認しておくこと**。詰まる場合は、`pyo3-polars`を経由せずpolars本体のArrow C Data Interface相当の機能を薄く自前で使う代替案を検討する。
- `engine`はpolars/PyO3を一切知らない設計を維持する（責務分離の原則通り）。`engine_pybind`が polars DataFrame → `faer::Mat<f64>` の変換を担う。
- 変換時、列ごとに`Vec<f64>`へ抽出してから`faer::Mat`に詰め直す（2回のコピーが発生する）。より少ないコピー（polarsの`&[f64]`を直接借用してengine側で詰める等）も技術的には可能だが、`engine`の関数シグネチャにライフタイムが入り込み「pure Rustロジック、polars非依存」の独立性が損なわれるため、**採用しない**。このプロジェクトの想定データ規模では、この2回のコピーのコストはQR分解本体（O(n×k²)）に対して無視できるレベルであるため。
- 欠損値（null）は`.rechunk()` + `null_count()`チェックで検出し、常にエラーとする（`testing-policy.md`・`ols-implementation-notes.md`の欠損値方針を参照）。サンプルの自動除外は行わない。

## エラーハンドリング

- 独自エラー型は **thiserror** で定義する。
- `engine_pybind`（PyO3境界）で、thiserrorのエラー型を`PyErr`に変換する。
- `unwrap` / `expect` はプロトタイプ段階を除き避ける。避けられない場合は理由をコメントで明記する。
- **Python側の例外はカテゴリ別に分ける**（全て`PyValueError`にまとめない）。最低限、以下の2階層を`pyo3::create_exception!`で定義し、`engine_pybind`側でエラー変換時にどちらに属するか判断して使い分ける。
  - `ValidationError`: 入力・パラメータが不正（次元不一致、欠損値、観測数不足、クラスター数不足、`confidence_level`の範囲外等）。Python側の`ValueError`も継承させ、素の`except ValueError`でも捕まえられるようにする。
  - `ComputationError`: 計算過程で発覚した問題（特異行列等）。Python側の`RuntimeError`を継承させる。
  - メモリ不足等は専用の例外クラスを設けない（Rust/Pythonのメモリ確保失敗は通常このレイヤーで綺麗に変換できるものではないため）。

## Lint / フォーマット

- `cargo clippy --all-targets -- -D warnings` で警告ゼロを基準とする。
- `cargo fmt --check` でフォーマット崩れを許容しない。
- 具体的なlintレベルの追加設定（`clippy.toml`等）はリポジトリ雛形作成時に確定する。

## パフォーマンス

- Arrowのゼロコピー原則（CLAUDE.md 2章）を壊す不要な`clone`・コピーを避ける。
- 大きなデータに対する計算量・メモリ使用量に注意する。

## テスト

- 対応する純粋ロジックの単体テストは `tests/engine_tests/`（`cargo test`）に置く。
- 許容誤差等のテスト方針の詳細は `testing-policy.md` を参照。
