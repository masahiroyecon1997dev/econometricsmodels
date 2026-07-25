# engine_pybind/src/linear/ 実装ノート（OLS/WLS）

このファイルは `engine_pybind/src/linear/` 配下のファイルを読み書きするときだけ自動ロードされる。設計の背景は`docs/planning/specs/ols-implementation-notes.md`6〜7章・`wls-implementation-notes.md`が正本。ここは差分の索引のみ。

## バージョン固定（変更時は要注意）

`pyo3=0.28.2` / `polars=0.54.4` / `pyo3-polars=0.27.0`（すべて`=`で完全固定、`Cargo.toml`）。`pyo3-polars=0.27.0`が`pyo3="^0.28"`を要求するための組み合わせ（`pyo3 0.28.0`/`0.28.1`はyanked済み）。`pyo3`を上げる場合は対応する`pyo3-polars`の新版公開を待つ必要がある（`pyo3-polars`は2025年7月にpolars本体リポジトリへ統合されアーカイブ済み、`.claude/rules/rust-style.md`「既知のリスク」参照）。Rust側`polars`クレートとPython側`polars`パッケージ（PyPI）はバージョン体系が分離しているため、数字を合わせる必要はない（実際の互換性は`pyo3-polars`の`polars_ffi::version_0`が担保）。

## polars 0.54.4特有の差異（踏んだ罠）

- `ChunkedArray::rechunk()`は`Cow<'_, ChunkedArray<T>>`を返す。`Cow`は`IntoIterator`非実装のため`.into_iter()`ではなく`.iter()`を使う（`Cow`はDerefで透過的に呼べる）。
- pyo3 0.28では`PyObject`型エイリアスがpreludeから削除済み。`Py<PyAny>`を直接使う。
- pyo3 0.28以降、`Clone`実装`#[pyclass]`の`FromPyObject`自動導出はopt-in。Python側インスタンスを引数で受け取るオプション型（`OLSOptions`等）には`#[pyclass(from_py_object)]`を明示する。

## バリデーションの責務分担（`engine`と重複させない）

`engine`は列名を知らないため検知できず、`engine_pybind::fit`側で`ValidationError`として弾く項目（OLS/WLS共通）:

- `y`と`x`に同じ列名が含まれる場合／`x`内の重複列名
- `include_intercept=true`のとき`x`に`"const"`という列名がある場合（自動追加する定数項名と衝突）
- `x`が空リストの場合

`confidence_level`の範囲チェック・`cov_type="cluster"`なのに`cluster_col`未指定、といった`engine`側が既に検知する項目は`engine_pybind`側で重複チェックしない（`OlsError`/`WlsError`のバリアント一覧は`ols-implementation-notes.md`1章の対応表を参照）。

## エラー変換

`OlsError` → `PyErr`は`impl From`ではなく`fn ols_error_to_pyerr(err: OlsError) -> PyErr`という関数として実装する（`OlsError`・`PyErr`ともにこのクレート外定義の型でorphan ruleに抵触するため）。呼び出し側で`.map_err(ols_error_to_pyerr)?`する。WLSも同型のパターンを踏襲する。

## `cov_type`固有の追加列

`cluster_col`/`time_col`の抽出は該当する`cov_type`のときのみ行う。無関係な列を誤って要求してエラーにしないこと。
