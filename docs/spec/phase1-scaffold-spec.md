# Spec: リポジトリ雛形の作成

`docs/planning/specs/phase1-scaffold.md` として保存する想定のドキュメント。

## 対象issue

`create_phase1_ols_issues.sh` で作成される最初のissue（`SCAFFOLD`変数、ラベル: `setup`）。

> **リポジトリ雛形の作成（Cargo workspace / engine / engine_pybind / python_package）**
>
> - 以降の全実装の土台となるリポジトリ構造を作る。Phase1(OLS)に限らず、以降のPhase全てが依存する。
> - ルート `Cargo.toml`（workspace）を作成し、`engine`, `engine_pybind` をメンバーとして登録
> - `engine/`, `engine_pybind/` それぞれに空のcrateを作成（`cargo new --lib`相当）
> - `pyproject.toml` を作成し、maturinをビルドバックエンドに設定（`engine_pybind`をビルド対象にする）
> - `python_package/econometricsmodels/__init__.py`, `python_package/econometricsmodels/py.typed` を作成
> - `tests/engine_tests/`, `tests/api_tests/` の空ディレクトリを作成（`.gitkeep`等）
> - `.github/workflows/` ディレクトリのみ作成（中身は別途）
>
> 参照: CLAUDE.md 3章（リポジトリ構成）
>
> 完了条件: `cargo build`（空クレートで）が成功する / `maturin develop` が成功する

## これまでの記録（意思決定の経緯）

チャット上でのやり取りを通じて、以下の点をユーザーに確認・決定した。

| 論点 | 決定内容 | 備考 |
|---|---|---|
| 作業場所 | このセッション（Claude Code外のチャット環境）はGitHubへの直接push権限を持たないため、ファイル内容の提示のみを行う方針とした | 実際の配置・`git init`・push はユーザー側で実施 |
| Rust edition | `2024` | CLAUDE.mdに明記がなかったため確認 |
| faerのバージョン固定 | `=0.24.4` | crates.ioで`max_stable_version`が`0.24.4`であることを確認済み。rust-style.mdの「Cargo.tomlでfaerのバージョンを明示的に固定する」方針に対応 |

上記に加え、提示時点でこちらの判断で補った点（要確認）:

- `pyo3 = "0.29.0"`、`maturin>=1.14,<2.0`、`thiserror = "2.0.18"` は、いずれもcrates.ioの現時点の最新安定版を採用した。
- `resolver = "3"` を明示指定した。仮想workspace（ルートに`[package]`を持たない構成）では、`edition = "2024"`を指定しても`resolver`の暗黙継承が効かず、`[workspace]`側での明示指定が必須という仕様のため。
- `faer`・`thiserror`のバージョンは、`[workspace.dependencies]`に**先出しで定義のみ**行った。実際に`engine/Cargo.toml`の`[dependencies]`へ追加するのは別issue（`ENGINE_DEPS`: 「OLS: engineクレートに依存追加（faer, thiserror）」）の作業範囲であり、本issueのスコープ外という理解。
- `pyproject.toml`に`readme`フィールドは含めていない。`README.md`が存在しない状態でビルドすると失敗しうるため。リポジトリに`README.md`を用意した時点で追加を推奨。

## 未検証事項

- このサンドボックス環境にはRustツールチェーン（`cargo`/`rustc`）がなく、`rustup`もネットワーク許可ドメイン外のため、**`cargo build` / `maturin develop`は未実行・未検証**。完了条件の充足はdevcontainer上でのユーザー確認が必要。
- `pyo3`の`extension-module`フィーチャーはビルド時にPythonインタプリタを検出する（`pyo3-build-config`経由）。`python3`が`PATH`上にある環境（devcontainerのベースイメージ`python:3.14-slim-bookworm`）であれば動作する想定だが未確認。

## 生成物一覧

```
Cargo.toml                                   # workspaceルート
engine/
├── Cargo.toml
└── src/lib.rs
engine_pybind/
├── Cargo.toml
└── src/lib.rs
pyproject.toml
python_package/econometricsmodels/
├── __init__.py
└── py.typed
tests/engine_tests/.gitkeep
tests/api_tests/.gitkeep
.github/workflows/.gitkeep
```

---

## ファイルごとのプロパティ説明

同じ仕組み・意味のプロパティは初出箇所でのみ説明し、以降は名称のみ記載する（重複説明はしない）。

### `Cargo.toml`（ルート）

```toml
[workspace]
resolver = "3"
members = [...]

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "MIT"
publish = false

[workspace.dependencies]
thiserror = "2.0.18"
faer = "=0.24.4"
```

| プロパティ | 説明 |
|---|---|
| `[workspace].members` | このworkspaceに属するcrate（`engine`, `engine_pybind`）のパス一覧。 |
| `[workspace].resolver` | 依存解決アルゴリズムのバージョン。`"3"`はedition 2024で使われる、Rustバージョン（MSRV）を考慮した解決器。仮想workspace（ルートpackageなし）では自動継承されないため明示が必要。 |
| `[workspace.package]` | ここに書いた値を、各メンバーcrateの`[package]`側で`<フィールド名>.workspace = true`と書くだけで継承できる（値の一元管理）。 |
| `[workspace.package].version` | 各crateのデフォルトバージョン番号。 |
| `[workspace.package].edition` | Rustのeditionの既定値。 |
| `[workspace.package].license` | ライセンス表記の既定値（SPDX形式の文字列）。 |
| `[workspace.package].publish` | `crates.io`等への公開可否の既定値。`false`はこのworkspace全体を「公開しない内部crate」として扱う指定。 |
| `[workspace.dependencies]` | 依存crateのバージョンをworkspace単位で一元管理するテーブル。各メンバーの`[dependencies]`からは`<crate名> = { workspace = true }`の形で参照する（バージョン重複記載を避ける）。 |
| `[workspace.dependencies].thiserror` | 独自エラー型定義用クレートのバージョン固定。 |
| `[workspace.dependencies].faer` | 線形代数クレートのバージョン固定。`"=0.24.4"`と`=`を付けているのは、`^`（キャレット、デフォルト）ではなく完全一致を強制するため（APIが変わりうるとの方針を踏まえた厳格な固定）。 |

### `engine/Cargo.toml`

```toml
[package]
name = "engine"
version.workspace = true
edition.workspace = true
license.workspace = true
publish.workspace = true

[dependencies]
```

| プロパティ | 説明 |
|---|---|
| `[package].name` | crate名。`engine`（ディレクトリ名と一致させている）。 |
| `<フィールド>.workspace = true` | 上記`[workspace.package]`の値を継承する記法（`version`/`edition`/`license`/`publish`共通）。 |
| `[dependencies]` | 現時点では空。faer/thiserrorの追加は別issue（`ENGINE_DEPS`）で行う想定のため。 |

### `engine/src/lib.rs`

crateのエントリポイント。現時点ではdocコメントのみで、実装コードは無い（空クレートとしてビルドが通ることの確認が目的）。

### `engine_pybind/Cargo.toml`

```toml
[package]
name = "engine_pybind"
version.workspace = true
edition.workspace = true
license.workspace = true
publish.workspace = true

[lib]
name = "_lib"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.29.0", features = ["extension-module"] }
engine = { path = "../engine" }
```

`[package]`配下は`engine/Cargo.toml`と同じ仕組みのため説明省略（`name`のみ`engine_pybind`に変更）。

| プロパティ | 説明 |
|---|---|
| `[lib].name` | ビルドされる共有ライブラリ（Pythonから見た拡張モジュール）の名前。`pyproject.toml`側の`module-name = "econometricsmodels._lib"`と対応させ、`econometricsmodels`パッケージ内の`_lib`という名前でインポートされる想定。 |
| `[lib].crate-type` | crateの出力形式。`"cdylib"`はC ABI互換の動的ライブラリを意味し、PythonなどからFFIで読み込むネイティブ拡張のビルドに使う。 |
| `[dependencies].pyo3` | RustとPythonのバインディングを行うクレート。`features = ["extension-module"]`は、Pythonの`libpython`に対して静的リンクせずビルドするためのフィーチャーで、通常のPython拡張モジュール配布（wheel化）で必須。 |
| `[dependencies].engine` | ローカルパス依存。`engine`クレートの関数を呼び出すために参照する（`engine_pybind`は計算ロジックを持たず、`engine`への薄い橋渡しに徹する設計）。 |

### `engine_pybind/src/lib.rs`

```rust
use pyo3::prelude::*;

#[pymodule]
fn _lib(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
```

| プロパティ | 説明 |
|---|---|
| `#[pymodule]` | この関数をPythonから見たモジュール初期化関数として登録するpyo3の属性マクロ。関数名（`_lib`）がPython側のモジュール名になる。 |
| `fn _lib(_m: &Bound<'_, PyModule>) -> PyResult<()>` | モジュールオブジェクト`_m`を受け取り、そこに関数・クラスを`add_function`/`add_class`等で登録していく場所。現時点では何も登録せず`Ok(())`を返すのみ（雛形段階）。 |

### `pyproject.toml`

```toml
[build-system]
requires = ["maturin>=1.14,<2.0"]
build-backend = "maturin"

[project]
name = "econometricsmodels"
version = "0.1.0"
description = "..."
requires-python = ">=3.12"
license = { text = "MIT" }
dependencies = ["polars>=1.0"]
classifiers = [...]

[tool.maturin]
manifest-path = "engine_pybind/Cargo.toml"
module-name = "econometricsmodels._lib"
python-source = "python_package"
features = ["pyo3/extension-module"]
```

| プロパティ | 説明 |
|---|---|
| `[build-system].requires` / `build-backend` | このパッケージをビルドするのに必要なツール（`maturin`）とビルドバックエンドの指定。PEP 517で定められた標準的な書き方。 |
| `[project].name` | PyPI上・`pip install`時のパッケージ名。 |
| `[project].requires-python` | 対応する最低Pythonバージョン。CLAUDE.md 12章の「3.12以上」に対応。 |
| `[project].dependencies` | ランタイム依存。`polars`のみ（データ入力はpolarsのみという非交渉事項に対応）。 |
| `[project].classifiers` | PyPI掲載用のメタデータタグ。対応Pythonバージョン（3.12/3.13/3.14）・ライセンス種別などを機械可読な形で示す。 |
| `[tool.maturin].manifest-path` | ビルド対象とするRust crateの`Cargo.toml`の場所。ワークスペース内で`engine_pybind`だけをビルド対象にするための指定。 |
| `[tool.maturin].module-name` | ビルドされたネイティブ拡張をPython側でどのモジュールパスとして配置するか。`econometricsmodels._lib`とすることで、`engine_pybind/Cargo.toml`の`[lib].name = "_lib"`と対応し、`econometricsmodels`パッケージ配下に`_lib`拡張モジュールが置かれる。 |
| `[tool.maturin].python-source` | Pythonソースのルートディレクトリ。`python_package`配下の`econometricsmodels/`をパッケージとして扱う指定。 |
| `[tool.maturin].features` | ビルド時に有効化するCargoフィーチャー。`engine_pybind/Cargo.toml`側の`pyo3`の`extension-module`フィーチャーと対応。 |

### `python_package/econometricsmodels/__init__.py`

```python
from __future__ import annotations

__all__: list[str] = []

__version__ = "0.1.0"
```

| プロパティ | 説明 |
|---|---|
| `from __future__ import annotations` | 型ヒントの評価を遅延させる指定（Python 3.12以降では実質不要になりつつあるが、型ヒント記法の柔軟性のため付与）。 |
| `__all__` | `from econometricsmodels import *`をした際に公開される名前のリスト。現時点では公開APIが無いため空。 |
| `__version__` | パッケージのバージョン文字列（`pyproject.toml`の`version`と手動で同期させる想定）。 |

### `python_package/econometricsmodels/py.typed`

中身の無い空ファイル。PEP 561のマーカーファイルで、「このパッケージは型ヒント情報を含む（`mypy`等の型チェッカーがこのパッケージの型情報を信頼してよい）」ことを示すためのもの。

### `tests/engine_tests/.gitkeep` / `tests/api_tests/.gitkeep` / `.github/workflows/.gitkeep`

いずれも中身の無い空ファイル。Gitは空ディレクトリ自体を追跡できないため、ディレクトリの存在をコミットに残すための慣習的なプレースホルダー。

---

## 次のアクション（案）

- devcontainer等、実際にRustツールチェーンがある環境で`cargo build`・（`maturin`インストール後の）`maturin develop`を実行し、完了条件を満たすか確認する。
- 完了確認後、`ENGINE_DEPS`（faer/thiserrorの`engine`への実追加）・`PY_DEPS`（polarsの`pyproject.toml`への追加確認）等、SCAFFOLDに依存する後続issueに着手する。
