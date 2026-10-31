# pickle・`copy`対応（結果・オプション）

`multiprocessing`・`joblib`・キャッシュ・Celeryに載せられるよう、利用者が保持する`#[pyclass]`（`*Options`・`*Result`）を
`pickle`・`copy.copy`・`copy.deepcopy`に対応させる。例外クラスは`errors.rs`の`__reduce__`で先に対応済み
（`error-handling.md`）。実装は`engine_pybind/src/shared/pickle.rs`。
利用者向けの説明（使い方・サイズ・版の制約）は公開ページ`docs/guide/pickling.md`が正本で、ここには設計判断だけを置く。

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

## estimatorを持つ結果（Logit・Probit・Tobit・FE・IV）

結果クラスが`engine`の推定量（`LogitEstimator`・`FeEstimator`・`OlsEstimator`等）を持つと、フィールドが非公開で
`Clone`もserdeも無いため、そのままは運べない。メソッドが実際に読む状態は小さいので、推定量そのものではなく
その状態だけを持つ形にした。

- **Logit・Probit・Tobit**: `engine`に最小の状態型（`LogitFitted`・`ProbitFitted`・`TobitFitted`）を設け、
  `predict`・`pred_table`・`marginal_effects`・`censoring_fit_check`の実装をそこへ移した（推定量は委譲するだけ）。
  状態は入力データ（`y`・`x`・係数名・切片の有無。Tobitは打ち切り境界も）・係数・分散共分散行列（Tobitは`σ`も）。
  標準誤差や適合度統計量は結果クラスに値のコピーがある。結果は`fit`の末尾で`estimator.into_fitted()`を受け取り、
  推定量全体は持たない。「推定量は検証済みの状態を保証する」原則とは、フィールドを非公開のまま、構築経路を`fit`と
  `from_parts`の2つに限り、`from_parts`が次元の整合（行数・列数・係数名・`σ`が正の有限値）を検査することで
  両立させる。最適性（収束していること）は再検証できないので、検査しない。
  - **`y`・`x`は状態に入れない**。`y`・`x`は`source_data`の列から作れる（`fit`が`source_data`の列を数値の列として
    取り出して組み立てたもの）ので、状態にはその列名（`x`の列名・`y`の列名・切片の有無）と係数・分散共分散行列
    （Tobitは境界と`σ`も）だけを入れる。復元は`source_data`を先に復元し、`fit`と同じ列抽出
    （`extract_f64_column(s)`）と`*Input::from_columns`（Tobitの境界と`y`の範囲の検査も同じ）で`y`・`x`を作り直して
    `from_parts`に渡す。`fit`と同じ関数を通すので、作り直した値は元と同じになる（再推定はしない）。列を読む前に`fit()`と同じ構造検査（`x`が空でない、重複がない、`y`が`x`に含まれない、切片と`const`が衝突しない。
    Tobitは`sigma`との衝突も）と列数が係数の数と合うことを確かめる（改ざんされた列名から巨大な行列を確保しないため。
    `x`の列は`source_data`にある相異なる列に限られるので、確保量は`source_data`の大きさで抑えられる）。列が無い、
    数値でない、欠損値がある、0行のときは`PICKLE_FORMAT_MISMATCH`。この「あるフィールドを別のフィールドから復元する」関係は
    `impl_pickle_state!`の`rebuilt_from(source_data) { fitted }`（`PickleFieldFromSource`）で表し、元のフィールドを
    先に復元する。状態の`source_data`を差し替えれば作り直す値も変わる（`source_data`の値を検査する手段は無い。
    信頼できる入力にだけ使う前提は変わらない）。
- **FE**: 固定効果（`FixedEffects`、エンティティ数に比例する辞書）を状態に持つ。`fit()`では計算せず、初回の
  `fixed_effects()`かpickle時に推定量から計算して保持する（使わない利用者の`fit()`の速度を変えないため）。復元した
  結果は推定量を持たず、計算済みの固定効果だけを持つ。
- **IV**: 第一段階の`OlsEstimator`（設計行列を持つ）を、`fit`で`OLSResult`へ変換して保持する。`first_stage()`は
  `OLSResult::clone_ref`で複製して返し、pickleでは各`OLSResult`がそのまま状態に入る。設計行列を手放すので、
  メモリは減る。

却下した案: 復元時に`source_data`から再推定する（毎回の推定コストがかかる。FE・REの結果と、IVの`first_stage()`
由来の結果は元のデータを持たず使えない）、serdeを`engine`に導入する（依存が増え、非公開フィールドの扱いが複雑）、
推定量の全フィールドを復元する（結果クラスに値のコピーがあり二重管理になる）。

サイズと時間: Logit・Probit・Tobitの`y`・`x`は状態に入れず復元時に作り直すので、pickleは`source_data`（元のDataFrame）と
ほぼ同じ大きさ（係数と分散共分散行列の分だけ大きい）。n=1,000,000・k=5で、Logit・Probit・Tobitのpickleは元のDataFrame（56 MB）と
同じ（`dumps`・`loads`とも0.1秒未満。リリースビルド。この計測のDataFrameは`x`5列・`y`2列の7列で56 MB）。復元は`fit`の列抽出と同じ処理を1回行う（`copy.copy`も同じ）。
最初は`y`・`x`を状態に直接入れていて、n=1,000,000・k=5のLogitでpickleは約104 MB（元のDataFrame 48 MB、設計行列 48 MB、`y` 8 MB）だった（元のDataFrameの約2倍。当時の計測のDataFrameは6列で、今回の計測とは列数が違う）。

FEは、`pickle`・`copy`のたびに固定効果を計算する（まだ計算していなければ。エンティティ数に比例する）。利用者が`fixed_effects()`を一度も呼んでいなくても起きる。IVは、`fit`のたびに内生変数の数だけ第一段階を`OLSResult`へ変換する（残差と近似値の`Vec`を作る）。設計行列を手放せるので、メモリは減る。

結果側の`param_names`と`fitted`の係数名は二重に持つが、`predict()`等の列の取り出しは`fitted`の名前を使うので、食い違った状態から添字の範囲外は起きない。

## 対象の範囲

全ての`*Options`・`*Result`と、`MarginalEffectsResult`・`CensoringFitCategoryResult`・`CensoringFitCheckResult`が対象。
`tests/cross_method/test_pickle.py`の`SAMPLES`が対象クラスの一覧で、`_lib`の`#[pyclass]`が`SAMPLES`か内部の型
（`TRANSIENT`）のどちらにも無いと落ちる。
