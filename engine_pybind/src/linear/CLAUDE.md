# engine_pybind/src/linear/ 実装ノート（OLS/WLS）

このファイルは `engine_pybind/src/linear/` 配下のファイルを読み書きするときだけ自動ロードされる。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景は`docs/spec/ols-spec.md`3.6節（データ受け渡し・エラー変換・バインディング）・3.5節（`augment()`）・3.9〜3.11節（事後診断）・`docs/spec/wls-spec.md`が正本（このファイルはその要約ではなく差分の索引）。他の系統の`engine_pybind`も、ここの「バージョン固定」「polars・pyo3のAPI差異」「列の取り込みと検証」を共通の前提にする。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## バージョン固定とAPIの差異

- **`pyo3`・`polars`・`pyo3-polars`は`=`で完全固定。固定値と組み合わせの理由は`engine_pybind/Cargo.toml`のコメントが正本で、ここに数字を複製しない。`pyo3`を上げるときは対応する`pyo3-polars`の新版を待つ**: `pyo3-polars`は本体リポジトリへ統合されアーカイブ済み（`.claude/rules/rust-style.md`「既知のリスク」）。Rust側`polars`とPyPIの`polars`はバージョン体系が別で、数字を合わせる必要はない（互換性は`polars_ffi::version_0`が担保）。
- **polars・pyo3のAPIの差異**: `ChunkedArray::rechunk()`は`Cow`を返すので`.into_iter()`ではなく`.iter()`。pyo3 0.28以降`PyObject`はpreludeに無いので`Py<PyAny>`を使う。`Clone`な`#[pyclass]`の`FromPyObject`自動導出はopt-inなので、引数で受け取るオプション型には`#[pyclass(from_py_object)]`を付ける。`DataFrame::new`は`(height, columns)`の2引数で、列の追加は`Column::new(name.into(), values)`と`.with_column(column)`。

## 列の取り込みと検証

- **`fit_*`と`predict()`/`augment()`の`new_data`で`PyDataFrame`への全列の変換を直接行わず、`check_dataframe`→`select_columns`で指名された列だけを取り込む。列名を持つオプションを足したら`*Options::data_columns()`に足す**: 未使用の`UInt128`列がpolars-coreのpanicを、`Extension`列が`ComputeError`を起こす。`data_columns()`に足さないと列が取り込まれず`COLUMN_NOT_FOUND`になる。存在しない列・取り込めないdtypeの列は列を引く側が報告するので、検査の順序は列を引く位置で決まる（`tests/cross_method/test_validation_order.py`）。詳細: `docs/spec/ols-spec.md`3.6節。
- **列名が要る検証（`x`が空、`y`/`x`の重複、`x`内の重複、`"const"`の衝突、単一列の役割どうしの重複）は`engine_pybind/src/shared/validation.rs`の関数を呼び、新しい手法でも独自実装しない。`y`/`x`は`validate_common_roles`（WLSは`validate_weighted_roles`）でまとめて呼び、`weight`を別に`validate_no_duplicate_roles`で呼ばない**: 「役割間の重複→`x`内の重複→`"const"`」の順を全手法で揃えている（別に呼ぶと崩れる）。`engine`は列名を知らない。詳細: `docs/spec/ols-spec.md`3.6節。
- **`engine`が既に検知する項目（`confidence_level`の範囲、`cluster`の未指定等）を`engine_pybind`で重複して検査しない。同じDataFrameから抽出した列どうしの行数の不一致も検査しない**: 行数の不一致は到達不能（polarsのDataFrameは全列が同じ長さ）。詳細: 同上。
- **`cov_type`の検査は`parse_cov_kind`（値）→`check_cov_options`（未使用のオプション→`hac`での`hac_time`の欠落）→`resolve_cov_type`（列）の3段で、関数は`&OLSOptions`ではなく個々のフィールド値を取る。`cluster`・`hac_time`は該当する`cov_type`のときだけ抽出する。`cov_type="hac"`では`hac_time`を必須にし（`require_hac_time`）、行順を時間順とみなす経路を足さない**: 無関係な列を要求してエラーにしない。行順の既定は、時系列順でないデータでも時系列順のHACに見える結果を黙って返す。IVの`gmm_weight_type="hac"`も同じ関数を使う。詳細: `docs/spec/ols-spec.md`3.2節「HAC」・3.6節。
- **`WLSOptions`は`OLSOptions`と同じフィールドを独立に持ち、共通の構造体に切り出さない**: PyO3のコンストラクタはフラットなkwargsが前提。詳細: `docs/spec/nonlinear-common.md`10章、`docs/spec/wls-spec.md`1章。

## 結果と事後診断

- **`augment()`は取り込んだ`DataFrame`ではなく、保持している元のオブジェクト（`source_data`）に`append_predicted_column`で予測値の列を足す。列名が既にあれば上書きせず`ValidationError`**: 推定に使わない列（取り込めないdtypeを含む）も出力に残し、データをコピーしない。`OLSResult`の`source_data`は`IVResult.first_stage()`由来の構築経路があるので`Option`（他の結果は常に保持）。詳細: `docs/spec/ols-spec.md`3.5節。
- **`OLSResult`の`#[pymethods]`は`ols/result.rs`に1ブロックだけ置き（`multiple-pymethods`を使わない）、事後診断の本体は`ols/diagnostics.rs`の自由関数にして、メソッドが`guard`で包んで呼ぶ。`guard`と`py.detach`の位置を変えない**: 詳細: `docs/spec/ols-spec.md`3.6節。
- **事後診断は`source_data`と`param_names`から`x`の列だけを取り込み直し、保持している残差と合わせてengineに渡す。`has_intercept`は保持フィールドを使い、`param_names[0] == "const"`から推定しない**: `OLSResult`は推定量を持たない。`include_intercept=false`で`x`に`"const"`列がありうる。`source_data`が無い結果は`ValidationError`。LM/Fはengineが両方計算し、バインディングは`statistic`で選ぶだけ。詳細: `docs/spec/ols-spec.md`3.6節・3.9節。
- **事後診断の引数は`&Bound<PyAny>`で受けて`extract_strict_text`/`extract_strict_int`/`extract_column_list`で型を厳密に検査する（`bool`・`float`は`TypeError`）。`nlags`は`i64`のままengineに渡す。元のデータの有無の検査は型の検査より後。Breusch-Paganの`variables`はモデルの`x`・`y`との重複を検査しない**: 範囲と飽和はengine。モデル外の列や`y`も`Z`にできる仕様。詳細: `docs/spec/ols-spec.md`3.10節・3.11節。
