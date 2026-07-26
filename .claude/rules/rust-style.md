---
paths:
  - engine/**/*
  - engine_pybind/**/*
---

# Rustコーディング規約（engine / engine_pybind）

このファイルは `engine/` `engine_pybind/` 配下で作業する際に自動的に読み込まれる。CLAUDE.mdの非交渉事項（2章）を前提とした、Rust実装の詳細規約。

## ファイル・ディレクトリ構成（手法が増える前提）

`engine` `engine_pybind`ともに、今後20〜30手法規模に増える前提で以下の構成にする。

- **系統（統計的に近い手法のグループ）＝ディレクトリ**。命名衝突を避け、「この手法はどこにあるか」を一意にする。
  - `linear/`: OLS, WLS, GLS, 区分回帰
  - `nonlinear/`: Logit, Probit, Tobit（MLEベースの非線形モデル。被説明変数が連続なTobitも含むため「discrete_choice」ではなく推定方式で命名）
  - `iv/`: 2SLS, GMM
  - `panel/`: FE, RE
  - `causal/`: DID, RDD
  - `io/`: IO構造推定（Nested Logit, Random Coefficient Logit, 単一エージェントモデル, 静学/動学ゲーム等）
  - `time_series/`: ARCH, GARCH, VAR（Phase7）
  - 系統名・手法の割り当ては上記を初期案とし、実装時に見直してよい（例: Phase2の"Logit"とPhase6の"Logit"は別系統ディレクトリに属するため衝突しない）
- **手法＝最初は1ファイル**（例: `linear/ols.rs`）。ファイルが肥大化したら`ols/`ディレクトリに昇格し`mod.rs`+`data.rs`+`options.rs`等に分割する。全手法に最初から複数ファイルを強制しない。
- **系統内で共有するロジック**は`<系統>/common.rs`に置く。
- **全手法で共有するロジック**（DataFrameからの列抽出等、統計手法に依存しない処理）は系統ディレクトリの外、クレート直下（例: `column_extraction.rs`）に置く。
- `engine`と`engine_pybind`で同じ系統名・ディレクトリ構成を揃える（`engine/src/linear/ols.rs` ⇔ `engine_pybind/src/linear/ols.rs`のように対応させる）。

## 言語方針

- **英語にする**: 例外・バリデーションメッセージ（`ValidationError`/`ComputationError`等、Pythonユーザーに表示される文字列）、公開API（`#[pyclass]` / `#[pyfunction]`）に付ける`///`docコメント（PyO3経由でPythonの`__doc__`になり、`help()`やIDE補完でユーザーに見えるため）。
- **`#[pyclass]`には`module`属性を明示する**（例: `#[pyclass(module = "econometricsmodels._lib")]`）。PyO3のデフォルトでは`__module__ == "builtins"`になり、mkdocs（mkdocstrings/griffe）がPython側の再エクスポート（`_lib` → `python_package`側モジュール）を解決できず`AliasResolutionError`でドキュメントビルドが失敗する（詳細は`docs/planning/specs/ols-implementation-notes.md`「9. ドキュメント」参照）。
- **日本語のままでよい**: 非公開関数・非公開型（`#[pyclass]`/`#[pyfunction]`が付いていないもの）の`///`/`//!`コメント、実装の背景説明、TODOコメント等の開発者向けの記述。GitHub Issue・CLAUDE.md・rules等の開発ドキュメントは対象外（日本語のまま）。
- 理由: `econometricsmodels`はeconomicon専用ではなくPyPI公開の独立パッケージであり、Pythonエコシステムの慣習（pandas/numpy/polars等）に合わせる。economicon側はi18nで独自にローカライズするため、例外はクラス（`ValidationError`/`ComputationError`）で分岐する設計になっており、メッセージ文字列の言語はeconomicon側のi18nに機能的な影響を与えない。

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
- `engine`はpolars/PyO3を一切知らない設計を維持する（責務分離の原則通り）。`polars DataFrame → faer::Mat<f64>`の変換は2段階に分かれる。
  1. `engine_pybind`: polars DataFrameから列ごとに`Vec<f64>`へ抽出する（`column_extraction::extract_f64_column`）。
  2. `engine`: 抽出済みの列（`&[f64]`/`&[Vec<f64>]`）から`faer::Mat`を組み立てる（例: `engine::linear::ols::OlsInput::from_columns`）。切片列の自動追加等、設計行列の組み立てに関わるロジックはここに置く（「計算ロジックをengine_pybindに書かない」原則、`docs/planning/specs/ols-api-design.md`参照）。
- 変換時、列ごとに`Vec<f64>`へ抽出してから`faer::Mat`に詰め直す（2回のコピーが発生する）。より少ないコピー（polarsの`&[f64]`を直接借用してengine側で詰める等）も技術的には可能だが、`engine`の関数シグネチャにライフタイムが入り込み「pure Rustロジック、polars非依存」の独立性が損なわれるため、**採用しない**。このプロジェクトの想定データ規模では、この2回のコピーのコストはQR分解本体（O(n×k²)）に対して無視できるレベルであるため。
- 欠損値（null）は`.rechunk()` + `null_count()`チェックで検出し、常にエラーとする（`testing-policy.md`・`ols-implementation-notes.md`の欠損値方針を参照）。サンプルの自動除外は行わない。
- クラスター変数等の「グループの同一性」だけが意味を持つ列は、整数決め打ちにせず文字列として扱う（州名・企業ID等、実務では文字列/カテゴリカルの方が多いため）。

## エラーハンドリング

- 独自エラー型は **thiserror** で定義する。
- `engine_pybind`（PyO3境界）で、thiserrorのエラー型を`PyErr`に変換する。
- `unwrap` / `expect` はプロトタイプ段階を除き避ける。避けられない場合は理由をコメントで明記する。
- **Python側の例外はカテゴリ別に分ける**（全て`PyValueError`にまとめない）。最低限、以下の2階層を`pyo3::create_exception!`で定義し、`engine_pybind`側でエラー変換時にどちらに属するか判断して使い分ける。
  - `ValidationError`: 入力・パラメータが不正（次元不一致、欠損値、観測数不足、クラスター数不足、`confidence_level`の範囲外等）。Python側の`ValueError`も継承させ、素の`except ValueError`でも捕まえられるようにする。
  - `ComputationError`: 計算過程で発覚した問題（特異行列等）。Python側の`RuntimeError`を継承させる。
  - メモリ不足等は専用の例外クラスを設けない（Rust/Pythonのメモリ確保失敗は通常このレイヤーで綺麗に変換できるものではないため）。
- **系統をまたいで同じ意味・同じメッセージのバリアントが複数の系統のエラー型に重複する場合は`engine::error::CommonError`に切り出す**（`DimensionMismatch`/`InsufficientObservations`/`InvalidConfidenceLevel`/`MissingClusterColumn`/`InsufficientClusters`/`ComputationFailed`がOLS/WLSとnonlinearで重複していたことから、Issue #113で導入）。各系統のエラー型（`LeastSquaresError`/`MleError`等）はthiserrorの`#[error(transparent)] Common(#[from] CommonError)`バリアントでこれを包む。`?`演算子は`#[from]`により自動変換されるため、`CommonError`のバリアントを直接構築する箇所（`return Err(...)`等`?`を経由しない箇所）でのみ`.into()`を明示する。系統固有の追加バリアント（`SingularHessian`等）や、将来「意味は同じだが系統固有の追加フィールドが要る」ケースは`CommonError`に含めず、各系統のエラー型に直接定義してよい（`CommonError`を使うかどうかは系統ごとに選べる）。`engine_pybind`側の変換は`engine_pybind/src/errors.rs`の`common_error_to_pyerr`に集約し、各系統の`*_error_to_pyerr`関数はこれに委譲する。

## Lint / フォーマット

- `cargo clippy --all-targets -- -D warnings` で警告ゼロを基準とする。
- `cargo fmt --check` でフォーマット崩れを許容しない。
- 具体的なlintレベルの追加設定（`clippy.toml`等）はリポジトリ雛形作成時に確定する。

## パフォーマンス

- Arrowのゼロコピー原則（CLAUDE.md 2章）を壊す不要な`clone`・コピーを避ける。
- 大きなデータに対する計算量・メモリ使用量に注意する。

## テスト

- 純粋ロジックの単体テストは、対応するソースファイル内の `#[cfg(test)] mod tests`（同じファイルの末尾。`cargo test -p engine`で実行）に置く。`tests/engine_tests/`は現状未使用（対象コードと同じファイルにあることでリファクタリング時の追従漏れを防げるため、OLS実装で一貫してこの方式を採用している）。将来的にモジュール横断の統合テストが必要になった場合のみ`tests/engine_tests/`の使用を検討する。
- 許容誤差等のテスト方針の詳細は `testing-policy.md` を参照。
- **カバレッジの現実的な目標**: `cargo llvm-cov -p engine`で計測する。100%は目指さず、既に検証済みの不変条件（特異性検出済みの行列のCholesky分解、事前検証済みの自由度によるt分布/F分布の構築等）に対する防御的な`Result`化（`unwrap`/`expect`を避けるため`Result`を返すが、実際にはその不変条件により失敗し得ない`map_err`分岐）はカバレッジ対象外として許容する。対象外にする場合は、その箇所のdocコメントに「なぜ理論上到達不能か」を明記すること（`engine::linear::ols`の`xtx_inverse`・`wald_f_test`等を参照）。
