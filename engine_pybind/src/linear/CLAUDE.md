# engine_pybind/src/linear/ 実装ノート（OLS/WLS）

このファイルは `engine_pybind/src/linear/` 配下のファイルを読み書きするときだけ自動ロードされる。設計の背景は`docs/spec/ols-spec.md`「engine/engine_pybind間のデータ受け渡し・エラー変換」・`docs/spec/wls-spec.md`が正本。ここは差分の索引のみ。

## バージョン固定（変更時は要注意）

`pyo3=0.28.2` / `polars=0.54.4` / `pyo3-polars=0.27.0`（すべて`=`で完全固定、`Cargo.toml`）。`pyo3-polars=0.27.0`が`pyo3="^0.28"`を要求するための組み合わせ（`pyo3 0.28.0`/`0.28.1`はyanked済み）。`pyo3`を上げる場合は対応する`pyo3-polars`の新版公開を待つ必要がある（`pyo3-polars`は2025年7月にpolars本体リポジトリへ統合されアーカイブ済み、`.claude/rules/rust-style.md`「既知のリスク」参照）。Rust側`polars`クレートとPython側`polars`パッケージ（PyPI）はバージョン体系が分離しているため、数字を合わせる必要はない（実際の互換性は`pyo3-polars`の`polars_ffi::version_0`が担保）。

## polars 0.54.4特有の差異（踏んだ罠）

- `ChunkedArray::rechunk()`は`Cow<'_, ChunkedArray<T>>`を返す。`Cow`は`IntoIterator`非実装のため`.into_iter()`ではなく`.iter()`を使う（`Cow`はDerefで透過的に呼べる）。
- pyo3 0.28では`PyObject`型エイリアスがpreludeから削除済み。`Py<PyAny>`を直接使う。
- pyo3 0.28以降、`Clone`実装`#[pyclass]`の`FromPyObject`自動導出はopt-in。Python側インスタンスを引数で受け取るオプション型（`OLSOptions`等）には`#[pyclass(from_py_object)]`を明示する。
- `DataFrame::new`は`(height: usize, columns: Vec<Column>)`という2引数シグネチャ（`Vec<Column>`のみを渡す旧APIではない）。列を追加するには`polars::prelude::Column::new(name.into(), values)`で構築し、既存の`DataFrame`には`.with_column(column)`（`PolarsResult<&mut Self>`）で付加する（`OLSResult::augment()`/`WLSResult::augment()`で使う）。

## `white_test()`（事後診断）の配線

`OLSResult`は`OlsEstimator`を保持しないため、`white_test()`は`predict_for`と同じ経路（`x_column_names` →
`extract_f64_columns`）で`training_data`と`param_names`から`x`を再抽出し、保持している`residuals`と
合わせて`engine::linear::diagnostics::white_test`に渡す。`training_data`が`None`（`IVResult.first_stage()`
由来）なら`augment(new_data=None)`と同じく`ValidationError`。`has_intercept`は`param_names[0] == "const"`で
推定せず保持フィールドを使う（`include_intercept=false`で`x`に`"const"`列がありうるため）。LM/Fの両方を
engineが計算し、`statistic`引数（`StatisticVersion`、大文字小文字を区別しない）で`WhiteTestOutput`の
フィールドを選ぶだけ（`select_statistic`はWhite・BPで共通）。計算ロジックは持たない。

`breusch_godfrey_test(time, nlags, statistic)`も同じ配線（`x`と残差に加え、時間列を
`extract_time_order_ranks`で順位にして`engine`へ渡す。`hac_time`と同じく同値・欠損値・NaNは
`ValidationError`）。引数は型検査を厳密にするため`&Bound<PyAny>`で受け、`extract_strict_text`/
`extract_strict_int`（`bool`・`float`は`TypeError`）を通す。`nlags`は`i64`のまま`engine`に渡し、
`nlags < 1`の検査と巨大値の飽和は`engine`側（`InvalidNlags`・観測数不足）。
`breusch_pagan_test(variables, statistic)`も同じ配線で、`variables`は`Option<&Bound<PyAny>>`として
`extract_column_list`（`list`以外・`str`以外の要素は`TypeError`）で受け、`None`なら
`x_column_names`でモデルの`x`を使う。空リスト・重複は`validate_x_non_empty`/
`validate_no_duplicate_within_role`（ロール名`variables`）で`ValidationError`。モデルの`x`や`y`との
重複は検査しない（モデル外の列・`y`列も`Z`にできる仕様）。`training_data`の有無は`variables`の
型検査より後（型の誤りは学習データが無い結果でも`TypeError`）。

## DataFrameを構築して返す（`augment()`）

`predict()`までは全メソッドが`Vec<f64>`等のフラットな値を返すだけだったが、`augment()`は
`OLSResult`/`WLSResult`が`fit()`時の元`PyDataFrame`を非公開の`training_data`フィールド
（`OLSResult`は`Option<DataFrame>`、`IVResult.first_stage()`という別経路の構築元を持つため。
`WLSResult`はこの経路がなく常に`DataFrame`）として保持し、`new_data=None`時にそれへ予測値の列を
付加して返す設計にした。polarsの列は内部で参照カウント方式のため、`DataFrame`を`clone()`しても
実際のデータはコピーされない（Arrowゼロコピー方針、CLAUDE.md 2章と整合）。列名衝突
（ソースデータに既に`"predicted"`列がある場合）は`validation.rs`の
`validate_no_existing_column`で`ValidationError`にする（黙って上書きしない）。詳細な設計判断は
`docs/spec/ols-spec.md`「augment()」参照。

## バリデーションの責務分担（`engine`と重複させない）

`engine`は列名を知らないため検知できず、`engine_pybind`側で`ValidationError`として弾く項目（OLS/WLS/Logit共通）:

- `y`と`x`に同じ列名が含まれる場合（WLSは`weight`と`y`の重複も。`weight`と`x`の重複は
  許容、`docs/spec/wls-spec.md`「API引数」参照）／`x`内の重複列名
- `include_intercept=true`のとき`x`に`"const"`という列名がある場合（自動追加する定数項名と衝突）
- `x`が空リストの場合

これらは元々OLS/WLS/Logitの`fit`/`build_logit_input`にメッセージ文言まで重複して実装されていたが、`engine_pybind/src/validation.rs`（クレート直下、`column_extraction.rs`と同じ位置づけ）に集約した: `validate_x_non_empty`/`validate_no_duplicate_x`/`validate_no_const_collision`/`validate_no_duplicate_roles`（`y`/`weight`等の単一列名ロール間の重複、`roles: &[(&str, &str)]`のペアリストを受け取る汎用関数。ただし`x`のように複数列を取るロール、例えば将来のIVの`instrument`には未対応、着手時に再検討が要る）。**新しい手法を実装する際も、この4関数を呼び出す形にし、同様のチェックを独自実装しないこと。**

`confidence_level`の範囲チェック・`cov_type="cluster"`なのに`cluster`未指定、といった`engine`側が既に検知する項目は`engine_pybind`側で重複チェックしない（`LeastSquaresError`のバリアント一覧は`ols-spec.md`「engine/engine_pybind間のデータ受け渡し・エラー変換」の対応表を参照）。

行数不一致チェック（`y`/`x`/`weight`/`cluster`/`hac_time`の間で行数が食い違っていないかの検証）は、同一DataFrameから抽出する限り理論上到達不能（polarsのDataFrameは全列同じ長さであることを型の不変条件として強制するため）と判明し、全て削除した。**新しい手法でも同種のチェックを追加しないこと。**

## エラー変換

`engine::linear::common::LeastSquaresError`（OLS/WLS共通のエラー型。元々`OlsError`という名前だったがWLSも含む実態に合わせて改名・`linear/common.rs`に移動）→ `PyErr`は`impl From`ではなく`fn least_squares_error_to_pyerr(err: LeastSquaresError) -> PyErr`という関数として実装する（`LeastSquaresError`・`PyErr`ともにこのクレート外定義の型でorphan ruleに抵触するため）。呼び出し側で`.map_err(least_squares_error_to_pyerr)?`する。WLSも同型のパターンを踏襲する。

## `cov_type`固有の追加列

`cluster`/`hac_time`の抽出は該当する`cov_type`のときのみ行う。無関係な列を誤って要求してエラーにしないこと。**`cov_type="hac"`では`hac_time`が必須**（`require_hac_time`、未指定は`ValidationError`）。行順を時間順とみなす暗黙の既定は置かない（データが時系列順でなくても時系列順のHACに見える結果が黙って返るため）。engineの`CovType::Hac.time_order`（IVの`WeightType::Hac`も）は`Option`ではなく必須の`Vec<f64>`で、`engine_pybind`が`hac_time`の順位を渡す（`None`＝行順の経路はengineにも無い）。IVの`gmm_weight_type="hac"`も同じ関数を使う。

## `WLSOptions`（`OLSOptions`とは独立したpyclass）

`WLS`は当初専用の`WLSOptions`を持たず`OLSOptions`をそのまま再利用していたが（`WLSResult`は元から独立型だったのと非対称だった）、ユーザビリティ向上のため`WLSOptions`を新設した。フィールド構成は`OLSOptions`と完全に同一（`cov_type`/`include_intercept`/`confidence_level`/`cluster`/`hac_lags`/`hac_time`、既定値・意味論とも同じ、`docs/spec/wls-spec.md`「API引数」参照）。このフィールド重複は意図的で共通base構造体には切り出さない（`engine_pybind/src/nonlinear/CLAUDE.md`「`LogitOptions`/`ProbitOptions`/`TobitOptions`のフィールド重複は意図的」節と同じ理由。`IVOptions`が`OLSOptions`と同種のフィールドを独立再定義している既存precedentとの一貫性、PyO3のpyclassコンストラクタがフラットなkwargs surface前提であること）。

`parse_cov_type`（`linear/common.rs`）はこの新設に伴い、`&OLSOptions`ではなく`cov_type: &str, cluster: Option<&str>, hac_lags: Option<i64>, hac_time: Option<&str>`という個々のフィールド値を引数に取る形に一般化した（`nonlinear::common::parse_cov_type`が最初から個々の値を取っているのと同じ設計）。`OLSOptions`/`WLSOptions`どちらの`fit`関数も、呼び出し側で`&options.cov_type`等を展開して渡す。
