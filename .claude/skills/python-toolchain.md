# Python ツールチェーン（uv / maturin / ruff / pytest）

## uv — パッケージ管理

```bash
# 依存関係インストール
uv sync

# 開発用依存も含めてインストール
uv sync --dev

# パッケージ追加
uv add <package>

# 仮想環境内でコマンド実行
uv run python -c "import econometricsmodels"
```

## maturin — Rust → Python ビルド

```bash
# 開発用ビルド（editable install 相当）
uv run maturin develop

# リリースビルド
uv run maturin develop --release

# ホイール作成
uv run maturin build --release

# PyPI 公開（CI でのみ実行）
uv run maturin publish
```

`maturin develop` 後は Python から `import econometricsmodels` で動作確認できる。

## ruff — Lint・フォーマット

```bash
# Lint チェック（CI 必須）
uv run ruff check .

# 自動修正
uv run ruff check --fix .

# フォーマット確認（CI 必須）
uv run ruff format --check .

# フォーマット適用
uv run ruff format .
```

## pytest — テスト実行

```bash
# 全テスト
uv run pytest tests/

# 特定ファイル
uv run pytest tests/test_ols.py

# カバレッジ付き実行（80% 以上が目標）
uv run pytest tests/ --cov=econometricsmodels --cov-report=term-missing

# 詳細出力
uv run pytest tests/ -v

# 失敗時に即停止
uv run pytest tests/ -x
```

## CI で使うコマンドセット

```bash
uv sync --dev
uv run maturin develop --release
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ --cov=econometricsmodels --cov-report=xml
```

## よくあるエラーと対処

### `ImportError: cannot import name 'econometricsmodels'`
`maturin develop` を実行していない。ビルド後に再試行する。

### `ruff: No such file or directory`
`uv sync --dev` で開発依存をインストールする。

### `maturin develop` でリンクエラー
Rust ツールチェーンが未インストール、または `stable` でない可能性。
`rustup show` でツールチェーンを確認し、`rustup toolchain install stable` を実行する。
