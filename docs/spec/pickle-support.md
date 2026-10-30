# pickle・`copy`対応（結果・オプション）

`multiprocessing`・`joblib`・キャッシュ・Celeryに載せられるよう、利用者が保持する`#[pyclass]`（`*Options`・`*Result`）を
`pickle`・`copy.copy`・`copy.deepcopy`に対応させる。例外クラスは`errors.rs`の`__reduce__`で先に対応済み
（`error-handling.md`）。実装は`engine_pybind/src/shared/pickle.rs`。

## 方式

- 各クラスの`__reduce__`が`(_lib._restore_pickled, (種別名, 形式の版, 状態のdict))`を返す。状態は
  フィールド名→値のdict。復元は`_lib`のモジュール関数`_restore_pickled`に集約する（結果クラスは`#[new]`を
  持たないので、`__getstate__`/`__setstate__`は使えない）。
- フィールド一覧は`impl_pickle_state!`に1回書く。分割代入と構造体リテラルに展開されるため、フィールドを足して
  一覧に足し忘れるとコンパイルエラーになる。種別名は`PickleState::KIND`（クラス名）で照合するので、`match`に
  綴りを別に書かない。
- 純Pythonのデータに落として持つ案は採らない。`predict`・`augment`・事後診断が`_lib`のオブジェクト上にあり、
  全フィールドの二重管理になるため。

## `source_data`を含める

`fit()`に渡した`polars.DataFrame`（`source_data`）は状態に含める。往復で挙動が変わらないことを優先した:
含めないと、復元後の`augment(new_data=None)`と事後診断（White・Breusch-Pagan・Breusch-Godfrey）が
`TRAINING_DATA_NOT_RETAINED`になる。

- コスト: pickleのサイズは元のDataFrame（推定に使わない列を含む全列）以上になる。結果だけが要る用途では
  `params_table()`等の値を持ち出す。軽量化の手段は別途検討する。
- `copy.copy`はDataFrameを共有し、`copy.deepcopy`は複製する。
- DataFrameのpickle形式はpolars自身が決める。polarsの版をまたぐ互換性は`PICKLE_FORMAT_MISMATCH`の対象外。
- 復元時は`polars.DataFrame`であることを検査する（別の型の値が入った状態から、後の`augment()`で
  `AttributeError`になる結果を作らない）。`OLSResult`は`IVResults.first_stage()`由来で元のデータを持たない
  （`None`）ことがあり、その場合は`None`のまま往復する。

## `Vec<f64>`は`bytes`

残差・係数などの`Vec<f64>`は、要素ごとにPythonの`float`を作らず、リトルエンディアンの`bytes`にして状態に入れる
（観測数に比例するフィールドが`list`だとpickleの時間とサイズの大半を占めるため。n=1,000,000のOLSで
`dumps`は`list`版の約3分の1）。長さが8の倍数でない`bytes`は拒否する。

## 版と互換性

pickleは永続化形式ではない。パッケージの版をまたぐ互換性は保証しない。

- 状態に形式の版（`FORMAT_VERSION`）を持たせ、復元時に一致を確かめる。一致しない、種別が未知、キーの集合が
  違う、値の型が違う、引数の型が違う、はいずれも`ValidationError`の`PICKLE_FORMAT_MISMATCH`にする。
- フィールドの追加・削除はキー集合の検査で検出される。フィールドの意味や表現を変えて既存のpickleを読めなく
  するときは、`FORMAT_VERSION`を上げる。版は全種別共通の1つ。
- 復元は値の意味（`cov_type`の綴り、配列の長さの整合）を検査しない。型が合えば構造体を直接組み立てる
  （`*Options`のコンストラクタを通さない）。pickleは信頼できる入力にだけ使う前提で、不正な値のオプションは
  `fit()`が通常どおり検査して`ValidationError`にする。
- `message`は`object`（種別名）だけを持つ。原因（版・種別・キー・型）の区別は載せない。

## `guard`を使わない

`__reduce__`は値のコピーと変換だけでpanicしうる処理を持たないため、`error-handling.md`の方針どおり
`guard`で包まない（`_restore_pickled`は`#[pyfunction]`なので包む）。

## 対象の範囲と今後

- 実装済み: 全`*Options`、`OLSResult`・`WLSResult`・`REResult`、`MarginalEffectsResult`、
  `CensoringFitCategoryResult`・`CensoringFitCheckResult`。
- 未対応: `LogitResult`・`ProbitResult`・`TobitResult`・`FEResult`・`IVResult`。`engine`の推定量
  （`LogitEstimator`・`FeEstimator`・`OlsEstimator`等）を非公開フィールドで持つ。推定量はフィールドが
  非公開で`Clone`もserdeも無く、設計行列・共分散行列・グループコード等を保持する。方式は別途設計する
  （推定量の検証済みの状態を保証する原則とどう両立するか、再推定の計算コスト、pickleサイズ）。
- `tests/cross_method/test_pickle.py`の`PENDING`が未対応クラスの一覧で、対応したら外して`SAMPLES`に足す。
