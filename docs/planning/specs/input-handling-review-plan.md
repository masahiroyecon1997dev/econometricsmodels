# 入力まわりの挙動見直し 実装計画（Issue #451）

検証ガイド（`docs/guide/validation.md`）の執筆中に見つかった入力処理の不自然な挙動を整理し、直す。
別セッションで作業を引き継げるよう、決定事項・調査結果・実装手順を残す。調査日: 2026-10-04。
作業ブランチは `release/v0.8.0`（CLAUDE.md 5章: feature branchは切らず、ステップごとにコミットする）。
`0.x`のため破壊的変更は許容（CLAUDE.md 8章）。コミットは`feat!:`/`fix:`/`docs:`/`test:`を使い分ける。

## 1. スコープと決定事項

Issue #451 の5項目と、調査で追加で見つかった項目（N1〜N5）。ユーザー確認済みの決定を以下に固定する。

| # | 内容 | 決定 | 対応 |
|---|---|---|---|
| ① | 文字列列が黙ってキャストされる | 数値として使う列は整数・浮動小数・Boolean・Decimalの許可リストにし、それ以外は`ValidationError`（dtype名入り） | 本計画 S2 |
| ② | `LazyFrame`/`pl.Series`のメッセージ | `failed to read`は「実体が`polars.DataFrame`なのに抽出に失敗した」場合だけ。それ以外は`must be a polars.DataFrame, got ...`。`LazyFrame`には`.collect()`の案内を足す | S4 |
| ③ | 1-wayのFEは(entity, time)重複を見ない | 挙動は維持し、`docs/spec/fe-spec.md`に明記する（1-wayにはtime列がなく組の概念がない。`dk_time`併用時の重複はDKで二重に数える） | S7 |
| ④ | 引数の型エラーが組み込み`TypeError` | `TypeError`は維持し、メッセージに引数名・期待する型・実際の型を出す | S5 |
| ⑤ | `ComputationError`にサブクラスがない | 現状維持。「サブタイプ」という文言だけ直す | #452 の該当箇条と同一作業。#452側で実施する（重複させない） |
| N1 | DKの時点順序が辞書順 | 別Issue #453 として起票済み。設計方針は#453で決める | 本計画の対象外 |
| N2 | 小さい整数型などが読めない／Decimal・Int128でpanic | polarsの`dtype-i8`/`dtype-i16`/`dtype-u8`/`dtype-u16`/`dtype-i128`/`dtype-decimal`/`dtype-f16`/`dtype-struct`/`dtype-array`を有効化。使える型を公開ページに明示 | S1, S7 |
| N3 | Date/Datetime/Time/Durationが黙って数値化される | 数値として使う列では拒否。時点を表す列（`time`・`dk_time`・`hac_time`）は拒否しない（N1が直るまで整数化を強いると危険なため） | S2, S3 |
| N4 | グループキー列のNaNが通る | float列のNaN・無限大を`ValidationError`にする | S3 |
| N5-a | `hac_lags=True`が1として通る | 型の誤りとして`TypeError`。他の数値optionにも同じ検証を入れる | S6 |
| N5-b | `hac_lags=2**70`が`OverflowError` | 固定の上限は設けない。既存の範囲検査（`hac_lags`は`[0, n)`）に到達させて`ValidationError`にする | S6 |
| N5-c | `x`にtuple/`pl.Series`が通る | `list`以外は`TypeError`（④のヘルパーで同時に実装） | S5 |
| N5-d | 2^53超の整数の精度 | 検証ではなくデータの性質として、新しい公開ページ「Accepted data」に記載 | S7 |

原則（`validation.md`にも書く）: 型の誤り（`bool`を整数として渡す、`x`が`list`でない）は`TypeError`、値の誤り（範囲外、NaN、巨大な整数）は`ValidationError`。

## 2. 調査結果（現状の挙動）

### 2.1 数値列の抽出

- `engine_pybind/src/column_extraction.rs`の`extract_f64_column`が全列を無条件に`cast(Float64)`する。
  文字列は`"1e2"`が通り、`" 2 "`・`"abc"`はnullになり「missing value」と報告される。`"inf"`/`"nan"`は「non-finite」。
- `Date`は日数、`Duration`/`Datetime`はマイクロ秒として通る。`Boolean`は0/1。
- `cast`が失敗する（`List`・`Binary`・`Categorical`・`Enum`）場合だけ`COLUMN_NOT_CASTABLE_TO_NUMERIC`相当のメッセージになる。
- 2^53を超える`Int64`/`UInt64`は丸められる。

### 2.2 dtypeのfeature不足（N2）

`engine_pybind/Cargo.toml`の`polars = "=0.55.2"`にfeatureが指定されていない。そのため`PyDataFrame`への変換が次のように失敗する。

- `Int8`/`Int16`/`UInt8`/`UInt16`/`Float16`/`Array`/`Struct`: `failed to read 'data' as a polars.DataFrame: ... cannot create series from Int8`。**推定に使わない列に含まれていても失敗する。** `df.to_dummies()`は`UInt8`を返すので、ダミー変数を渡すだけで失敗する。
- `Decimal`/`Int128`: Rustのpanic（`pyo3_runtime.PanicException`）。`BaseException`継承なので`except Exception`で捕まらない。
- `Int32`/`UInt32`/`UInt64`/`Time`/`Date`/`Duration`/`List`/`Binary`/`Categorical`/`Enum`/`Null`/`Boolean`/`String`は変換自体は通る。
- feature名は`polars`クレート側にある（`pyo3-polars 0.28.0`には`dtype-array`/`dtype-decimal`/`dtype-struct`等の一部しかない。`dtype-i8`等はpolars直接）。

### 2.3 DataFrame抽出（②）

`extract_dataframe`は`type_name.starts_with("polars.")`で分岐する。`LazyFrame`・`pl.Series`はこの分岐に入り、`failed to read ... 'get_columns'`という内部実装が漏れたメッセージになる。

### 2.4 グループキー（N4）

`extract_group_key_column`は`String`にキャストして使う。`null`は拒否されるが、float列のNaNは`"NaN"`という1グループとして通る。`List`は拒否される。`Date`は通る。

### 2.5 引数の型（④・N5）

- `fit_*`の引数は`x: Vec<String>`等をPyO3が直接変換する。`x="x1"`は`Can't extract 'str' to 'Vec'`、`y=["y"]`は`'list' object is not an instance of 'str'`で、引数名が出ない。
- `x`は`tuple`・`pl.Series`も通る。`set`・ジェネレータ・`dict_keys`は`TypeError`。
- option（`#[pyclass]`、`i64`/`f64`フィールド）:
  `hac_lags=True`・`max_iter=True`・`dk_bandwidth=True`は1として通る。`tol=True`は1.0として通る。
  `2**70`は`OverflowError`（`hac_lags`・`max_iter`・`dk_bandwidth`）。
  `tol=NaN`は`ComputationError: failed to converge`になる（入力の問題なのに計算エラー扱い）。
  `hac_lags=-1`・`dk_bandwidth=-1`・`max_iter=-1`は構築時には通る（`fit()`の範囲検査で扱われるはず。未確認なので S6 で確認する）。
  `bool`型フィールド（`include_intercept`等）は既に厳密（`1`や`"yes"`は`TypeError`）。

対象のoptionフィールド: `confidence_level`（全手法）、`hac_lags`（OLS・WLS・IV）、`max_iter`・`tol`（Logit・Probit・Tobit）、`lower`・`upper`（Tobit）、`gmm_max_iter`・`gmm_tol`（IV）、`dk_bandwidth`（FE・RE）。

### 2.6 既存テストへの影響

文字列列やDate列を使う既存テストは少ない（`tests/nonlinear/test_tobit.py:1280`のクラスター用文字列列、`benchmark/linear/fixtures/generate_wls_fixtures.py:391`のみ。どちらもキー用途なので許可される）。`x`にtupleを渡すテストは無い。

## 3. 設計

### 3.1 列の役割とdtypeの許可表（`column_extraction.rs`）

| 役割 | 該当する引数 | 許可するdtype |
|---|---|---|
| 数値として使う | `y`、`x`、`weight`、IVの`x_exog`/`x_endog`/`instruments`、`dk`以外の数値列 | 整数（Int8〜Int128、UInt8〜UInt64）、浮動小数（Float16/32/64）、Boolean、Decimal |
| 順序を持つキー | FE/REの`time`・`dk_time` | 整数、浮動小数（有限のみ）、Date、Datetime、文字列、Categorical/Enum |
| 順序を持つ数値キー | OLS/WLS/IVの`hac_time` | 数値用のdtype、Date、Datetime（数値化して順序だけに使う。確認済み: エンジンの`time_ordering`は並べ替えの添字を返すだけで、時点の間隔は使わない） |
| 同一性だけのキー | `entity`、`cluster` | 整数、浮動小数（有限のみ）、文字列、Categorical/Enum、Boolean、Date |

- 許可外は`ValidationError`。メッセージは列名・実際のdtype・許可する種類を含める（例: `column 'x1' has dtype String, which cannot be used as a numeric column; cast it to a numeric dtype first`）。メッセージ文言は`tests/_error_messages.py`の定数に追加し、`COLUMN_NOT_CASTABLE_TO_NUMERIC`は到達しなくなるので削除する（`cast`の`map_err`は防御用に残すかは実装時に判断）。
- 数値として使う列は`Date`/`Datetime`/`Time`/`Duration`/`String`/`Categorical`/`Enum`/`Binary`/`List`/`Array`/`Struct`/`Null`/`Object`を拒否する。
- 順序を持つキーについて: 整数の時点は#453が直るまで辞書順になる（現状維持。誤りを広げない）。Date・ISO形式のDatetimeは今でも正しく並ぶ。

### 3.2 DataFrameの抽出（`extract_dataframe`）

`type_name`が`polars.dataframe.`で始まるときだけ`failed to read`にする。それ以外は`must be a polars.DataFrame, got {type_name}`。`type_name`が`polars.lazyframe.`で始まるときだけ末尾に`; call .collect() first`を足す。N2の対応後、`failed to read`は実質到達しにくくなるので、`tests/_error_messages.py`の「再現できない」というコメントを現状に合わせて直す。

### 3.3 引数の型検証（④・N5-c）

`lib.rs`の`fit_*`が`x`等を`Bound<PyAny>`で受け、共通ヘルパー（`validation.rs`か`column_extraction.rs`に置く）で検証する。

- `extract_column_list(param_name, &Bound<PyAny>) -> PyResult<Vec<String>>`: `list`でなければ`TypeError`（`'x' must be a list of column names (e.g. x=["x1"]), got str`）。要素に`str`以外があれば`TypeError`（`'x[1]' must be a str, got int`）。
- `y`・`entity`・`weight`等の`str`引数にも同型のヘルパー（`extract_column_name`）を使い、メッセージの形式を揃える。
- 対象: `fit_ols`・`fit_wls`・`fit_logit`・`fit_probit`・`fit_tobit`・`fit_iv`（`x_exog`・`x_endog`・`instruments`）・`fit_fe`・`fit_re`。
- `predict`/`augment`等、列名リストを受けるメソッドがあれば同じヘルパーに寄せる（**要確認**）。
- 例外クラスは`TypeError`のまま。`TypeError`は`fit()`の入口で出る（構築時ではない）。この点は現状の`validation.md`の記述と同じ。

### 3.4 数値optionの検証（N5-a・N5-b）

PyO3の`i64`/`f64`抽出は`bool`を通し、巨大な整数は`OverflowError`にする。次の方式を第一候補とする。

- 厳密な整数用の新しい型（例: `StrictInt(i64)`）に`FromPyObject`を実装する。`bool`は`TypeError`。`i64`に収まらない巨大な整数は、符号に応じて`i64::MAX`/`i64::MIN`に飽和させる。これにより、既存の範囲検査（`hac_lags`の`[0, n)`、`max_iter`の正値、`dk_bandwidth`の範囲）に到達し`ValidationError`になる。固定の上限は設けない。
- 厳密な浮動小数用の型（例: `StrictFloat(f64)`）は、`bool`を`TypeError`にする。整数は受け付ける。
- `IntoPyObject`も実装し、`#[pyo3(get, set)]`のゲッター・セッターとコンストラクタが同じ検証を通るようにする。型の導入が`#[pyclass]`の`get`/`set`と相性が悪ければ、`#[setter]`を手書きする方式に切り替える（**要確認**）。
- `tol`・`gmm_tol`・`lower`・`upper`のNaNは`ValidationError`にする。`engine/src/nonlinear/common.rs`の`validate_tol`は`tol <= 0`だけを見ているため、NaNが通る。`!(tol > 0.0)`と有限性の検査に改める（`InvalidTol`）。`gmm_tol`・Tobitの`lower`/`upper`（`InvalidCensoringBounds`）も同じ観点で確認する。
- `hac_lags=-1`・`max_iter=-1`・`dk_bandwidth=-1`が`fit()`でどう扱われるかを実測し、`ValidationError`になっていなければ直す。

### 3.5 グループキーのNaN（N4）

`extract_group_key_column`で、キャスト前にdtypeがfloatなら非有限値（NaN・無限大）を検査し、`ValidationError`にする。メッセージは数値列の`COLUMN_HAS_NON_FINITE_VALUE`に揃える（列名・値・行番号）。

## 4. 実装ステップ

各ステップの完了時に`cargo test`/`pytest`/`ruff`を通し、コミットする。

### S1. dtype feature有効化（N2）

1. `engine_pybind/Cargo.toml`の`polars`に`features = ["dtype-i8", "dtype-i16", "dtype-u8", "dtype-u16", "dtype-i128", "dtype-decimal", "dtype-f16", "dtype-struct", "dtype-array"]`を追加する。`pyo3-polars`側にも必要なfeature（`dtype-decimal`・`dtype-struct`・`dtype-array`）があれば追加する。コメントにfeatureを足した理由を書く（Issue番号は書かない。CLAUDE.md 6章）。
2. `maturin develop --release`でビルドし、wheelサイズとビルド時間を変更前後で記録する（結果をコミットメッセージかリリースメモに残す）。
3. 調査用スクリプト（`probe3.py`相当）で、全dtypeの「未使用列」が`fit()`を壊さないこと、`UInt8`の`to_dummies`列が推定に使えること、`Decimal`/`Int128`がpanicしないことを確認する。
4. pytestを追加する（`tests/test_input_dtypes.py`。全手法共通の入力なので`tests/`直下。OLSを代表に、`FE`でキー列、`Logit`で`y`のBoolean）。
   - 未使用の各dtype列があっても推定できる。
   - `Int8`/`Int16`/`UInt8`/`UInt16`/`Float16`/`Decimal`/`Int128`が`x`・`y`として使える（係数が同値の`Float64`列と一致）。
   - 使った場合に拒否されるdtype（`Struct`/`Array`）は、S2で`ValidationError`になることを確認する。
5. コミット: `feat: polarsの数値dtype（Int8/16・UInt8/16・Float16・Int128・Decimal）とStruct/Arrayを読めるようにする`

### S2. 数値列のdtype許可リスト（①・N3）

1. `column_extraction.rs`に数値用のdtype検査を追加し、`extract_f64_column`のキャスト前に呼ぶ。
2. メッセージ定数を`tests/_error_messages.py`に追加し、不要になった`COLUMN_NOT_CASTABLE_TO_NUMERIC`を整理する。
3. テストの更新と追加:
   - `tests/linear/test_ols_validation.py`の`test_non_numeric_dtype_raises`は、新しいメッセージ（`String`列がdtypeエラーになる）に合わせて書き換える。docstringの「missing valueの経路を通る」という説明も直す。
   - `tests/test_input_dtypes.py`に、`String`（数値風の`"1.0"`を含む）・`Categorical`・`Enum`・`Date`・`Datetime`・`Time`・`Duration`・`Binary`・`List`・`Array`・`Struct`・`Null`が、`y`・`x`で`ValidationError`になるテストを足す（全手法の代表としてOLS、`weight`でWLS、`instruments`でIV）。
   - Logitの`y`がBooleanで推定できる（`True`=1）ことをpositiveテストにする。
4. コミット: `feat!: 数値として使う列のdtypeを許可リストで検査し、文字列・日付・時刻列を拒否する`

### S3. キー列のdtype検査とNaN拒否（N3・N4）

1. `extract_group_key_column`に、順序を持つキー・同一性だけのキーの許可表（3.1）に沿ったdtype検査を足す。呼び出し側でキーの種類を渡す引数を追加するか、2つの関数に分けるかは実装時に判断する（呼び出し元は`linear/common.rs`・`iv/common.rs`・`nonlinear/common.rs`・`panel/fe.rs`・`panel/re.rs`）。
2. floatキーの非有限値を拒否する（3.5）。
3. `hac_time`の抽出（`linear/common.rs`ほか）が`Date`/`Datetime`を許可しつつ他を拒否するようにする。許可する前に、HACが時点の間隔を使わないことをengineのコードで確認する。
4. テスト: `entity`・`cluster`・`time`・`dk_time`・`hac_time`それぞれについて、許可するdtype（正常）と拒否するdtype・NaN・無限大（`ValidationError`）を、FE（`tests/panel/test_fe_validation.py`）・OLS（`tests/linear/test_ols_validation.py`）などの既存の検証テストに合わせて置く。
5. コミット: `feat!: キー列のdtypeを検査し、float列のNaN・無限大を拒否する`

### S4. DataFrame抽出のメッセージ（②）

1. `extract_dataframe`の分岐を3.2のとおり直す。
2. `tests/linear/test_ols_validation.py`（`data`/`new_data`にpandasを渡すテストの隣）に`LazyFrame`（`.collect()`の案内）と`pl.Series`のテストを足す。定数は`tests/_error_messages.py`に追加する。`new_data`（`predict`）でも同じ文言になることを1件確認する。
3. `tests/_error_messages.py`の`DATAFRAME_EXTRACTION_FAILED`のコメントを現状に合わせる。
4. コミット: `fix: LazyFrameやSeriesを渡したときのエラーを分かりやすくする`

### S5. 列名引数の型検証（④・N5-c）

1. 3.3のヘルパーを実装し、`lib.rs`の`fit_*`に適用する。
2. テスト: 各手法の代表（OLS、WLS、IV、FE）で、`x="x1"`・`x=("x1",)`・`x=pl.Series([...])`・`x=[1]`・`y=["y"]`・`y=None`が`TypeError`になり、メッセージに引数名が含まれることを確認する。`x`がlistのとき従来どおり動くことも確認する。
3. 既存の型エラー系テスト（`TypeError`を期待するもの）のメッセージ照合を新しい文言に合わせる。
4. コミット: `feat!: 列名引数の型が不正なときTypeErrorに引数名を含め、listのみを受け付ける`

### S6. 数値optionの検証（N5-a・N5-b）

1. 3.4の型（または`#[setter]`手書き）を導入し、2.5に挙げた全フィールドに適用する。
2. `validate_tol`ほか、NaNを見逃す検査をengineで直す（`cargo test -p engine`にNaNの単体テストを足す）。
3. 負の値（`hac_lags=-1`・`max_iter=-1`・`dk_bandwidth=-1`）の挙動を実測し、`ValidationError`にする。
4. テスト（`tests/<系統>/test_*_validation.py`、option検証は系統ごとの既存ファイルに足す）:
   - `bool`を渡すと`TypeError`（`hac_lags`・`max_iter`・`dk_bandwidth`・`gmm_max_iter`・`confidence_level`・`tol`・`gmm_tol`・Tobitの`lower`/`upper`）。
   - `2**70`と`-(2**70)`で`ValidationError`（`OverflowError`にならない）。
   - `tol`・`gmm_tol`がNaNで`ValidationError`。
5. コミット: `feat!: 数値optionでboolをTypeErrorにし、巨大な整数やNaNをValidationErrorにする`

### S7. ドキュメント（①〜N5、③、N2）

1. **新しい公開ページ`docs/guide/accepted-data.md`（英語）**を作り、`docs/mkdocs.yml`のnavに載せる。内容:
   - 受け付けるのはpolarsの`DataFrame`のみ。`LazyFrame`は`.collect()`が必要。
   - 列の役割（3.1の表）ごとに許可するdtypeと拒否するdtype、dtypeごとの変換（Booleanは0/1、Decimalと`Int128`・`UInt64`・`Int64`はf64に変換されること）。
   - 精度の注意: 2^53を超える整数、Decimal、Int128は丸められる。
   - 引数の型（`x`は`list[str]`のみ）と、型の誤りは`TypeError`、値の誤りは`ValidationError`という原則。
   - null・NaN・無限大は拒否（詳細は`validation.md`へのリンク）。
   - 時点列の注意（整数の時点がDKで辞書順になる件は#453が直るまで明記する。直ったら削除）。
2. `docs/guide/validation.md`を更新する: 「Data type」「Missing values」の行と末尾の「Data types」の箇条書きを新ページへの参照に置き換える。`TypeError`の説明を原則に合わせる。`ValidationError`の表に「サポートされない列のdtype」「float列キーのNaN・無限大」「`DataFrame`でない入力（`LazyFrame`は`.collect()`）」を足す。数値optionの`bool`は`TypeError`、巨大な整数・NaN・負値は`ValidationError`と書く。
3. `docs/spec/fe-spec.md`（③）: 1-wayは(entity, time)の重複・不均衡を許容すること、`dk_time`併用時の重複はDKで二重に数えることを書く。
4. `docs/spec/`の各手法仕様書のうち、文字列キャスト・dtype・`TypeError`に触れている箇所を探して直す（`grep`で洗い出す）。
5. CLAUDE.md 13章の公開ページ運用ルールに`docs/guide/accepted-data.md`の項目を足す（dtypeの許可表を変更したら更新する、と書く）。
6. Python側の`Raises:`docstring（`python_package/econometricsmodels/`の各`fit()`）に`TypeError`を追記する（引数の型が不正なとき）。
7. ⑤: #452の「サブタイプ」の箇条と同じ作業で、`tests/nonlinear/_binary_choice_checks.py`・`docs/spec/*`の文言を直す。#452側でまとめて実施する。
8. コミット: `docs: 受け付けるデータのページを追加し、検証ガイドと仕様書を新しい入力検証に合わせる`

### S8. 総合確認

- `cargo test --workspace`・`cargo clippy --all-targets -- -D warnings`・`cargo fmt --check`・`pytest`（全体）・`ruff check`/`ruff format --check`。
- `mkdocs build --strict`（navと内部リンク）。
- レビュー: `rust-reviewer`（S1・S3・S5・S6）、`python-reviewer`（docstring）、`testing-completeness-reviewer`（追加テストの網羅性）。
- 性能への影響: dtype検査は列ごとに1回のみ。`benchmark_performance.yml`の計測値に影響が出ない想定だが、列抽出を触るため`performance/`の手元計測で大きな変化がないことを確認する。
- wheelサイズ・ビルド時間の変化（S1）をリリースメモに残す。
- #451を閉じる前に、残項目（⑤は#452、N1は#453）への参照をコメントに残す。

## 4.1 進捗（2026-10-04時点、`release/v0.8.0`にコミット済み）

- S1〜S6は実装・コミット済み。S7（ドキュメント）は実装済みで、コミットはこのファイルと同じコミットに含まれる。S8のうち自動検査（pytest・cargo test・clippy・fmt・ruff・`mkdocs build --strict`）は通過。残りはレビュー用エージェントによる確認と性能の手元計測。
- 実装中に決めた細部:
  - `Null`型（全値が欠損の列）は、dtypeエラーにせず欠損値エラーで報告する（より分かりやすいため）。
  - dtype名はPythonの`pl.String`等と同じ呼び名（`String`・`Date`・`Boolean`・`Decimal`等、内部パラメータは含めない）。
  - `extract_dataframe`は型名の接頭辞ではなく`isinstance`で判定する（サブクラスも本物のDataFrameとして扱う）。
  - 数値option: `max_iter=2**70`のように大きな正の値は有効（`i64`の上限に丸められ、収束すれば通常どおり返る）。負の巨大な値は範囲検査で`ValidationError`。`tol`はNaN・無限大も`ValidationError`にした（engineの`validate_tol`を修正）。
  - Tobitの`lower`/`upper`、`gmm_tol`は既にNaNを弾いていたため、engine側の変更は`validate_tol`のみ。
  - 数値optionの検証は`engine_pybind/src/option_values.rs`（`from_py_with`と手書きsetter）に集約。
- 実測: リリースビルドの拡張モジュールは34.6MB→43.5MB（gzip後6.4MB→7.7MB）。`Cargo.lock`に推移的な依存が追加された（`ahash`・`float-cmp`ほか）ため、リリース前に依存の確認（`cargo deny`・ライセンス一覧`THIRD-PARTY-LICENSES.html`の更新）を行う。

## 5. 未解決・要確認（実装中にその都度ユーザーへ確認する。CLAUDE.md 14章）

- ~~`hac_time`にDate/Datetimeを許可してよいか~~: 確認済み、許可した（S2）。
- 数値option用の型を導入する方式と、`#[setter]`手書き方式のどちらが`#[pyclass]`と整合するか。
- 順序を持つキー・同一性だけのキーの関数を分けるか、引数で切り替えるか。
- Decimal・Int128・Float16を許可したときの精度の注意文の表現。
- `predict`/`augment`等、列名リストを受ける他のメソッドの有無と対応。

## 6. 本計画の対象外

- N1（DKの時点順序）: #453で設計から決める。解決後、`accepted-data.md`の注意書きと`FEOptions`/`REOptions`のdocstringを更新する。
- `ComputationError`のサブクラス化: 利用ケースが出てから検討する。
- 整数の精度（2^53超）の実行時検出: ドキュメントのみ。
