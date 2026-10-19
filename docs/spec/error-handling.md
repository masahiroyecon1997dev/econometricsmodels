# 例外の構造（`ValidationError` / `ComputationError`）

利用者向けの一覧は`docs/guide/error-codes.md`（`code`・`params`・英語テンプレートの表。カタログから生成）、
状況の分類は`docs/guide/validation.md`。ここは設計の理由と、エラーを足すときの手順を残す。
文面の規則は`error-style-guide.md`。

## 1. 目的

- このパッケージを使うアプリ（GUI・BIツールのバックエンド・バッチ）が、エラーを自前で多言語化・文言カスタマイズできるようにする。
- このパッケージ自身のメッセージは英語のまま。翻訳はパッケージに持ち込まない。
- 文字列のパースを不要にする（`code`と`params`で分岐する）。

非目標: 翻訳カタログの提供、警告（warnings）の追加。

## 2. 公開する属性

| 属性 | 型 | 意味 |
|---|---|---|
| `code` | `str` | 状況を表す安定した識別子（UPPER_SNAKE_CASE） |
| `message` | `str` | 問題の記述（英語、1〜2文）。利用者の値と限界を含む |
| `hint` | `str \| None` | 対処法（英語、1文）。機械的な対処のみ |
| `params` | `dict` | `message`/`hint`に埋め込まれた値。型は`int`/`float`/`str`/`bool`/`None`/`list[str]`のみ |

- `str(e)`は`message`と`hint`を空白1つでつないだ全文。`e.args == (str(e),)`。素の`except ValueError`・`str(e)`・`e.args[0]`を使う既存コードは変わらない。
- 公開コンストラクタは`(code, message, hint=None, params=None)`。pickle・`copy`・`multiprocessing`に対応する（`__reduce__`を自前で定義し、`__dict__`も引き継ぐ）。
- `ErrorCode`（`StrEnum`）で全`code`を定数として公開する。`e.code`は`str`のまま（`e.code == ErrorCode.X`も`e.code == "X"`も成り立つ）。翻訳テーブルの抜けを`list(ErrorCode)`で検査できる。
- 型の誤り（引数・オプション・`data`）は構造化せず、組み込みの`TypeError`にする。Pythonの慣習に合わせるため。文面だけ同じスタイルガイドに従う。

## 3. 不変条件（カタログ契約）

> 同じ`code`なら、例外クラス・`message`のテンプレート・`hint`のテンプレート・`params`のキー集合が常に同じ。

- `hint`が状況で変わるなら別の`code`にする（`hint`の有無も`code`で決まる）。
- これにより「`code`＋`params`から文面を再構成できる」ことが保証され、アプリ側の翻訳テンプレートが書ける。
- 分類（`ValidationError`/`ComputationError`）も`code`の属性。分類の原則そのものの議論は別扱いで、移行時は現行のクラスを保った。

## 4. `code`と`params`の規約

- フラット（階層なし）、状況を表す名詞句。**名前は「何の値か＋どういう状況か」**。汎用名は対象が本当に多岐にわたる状況（`OPTION_UNKNOWN_VALUE`・`OPTION_UNUSED`）に限り、範囲違反は対象ごとに分ける（`MAX_ITER_OUT_OF_RANGE`等。下限・上限の側は分けない）。
- Rustのバリアント名やファイル構造に依存させない（内部のリファクタで`code`を変えないため）。
- 構造ごとに`code`を増やさず、エンジンが計算した量（`min_obs`・`min_clusters`・`n_effects`等）を`params`で渡す。
- リリース済みの`code`は変えない・再利用しない。追加は後方互換。削除・意味変更・`params`キーの削除や型変更は破壊的変更。`params`キーの追加は非破壊。
- `params`のキー名は利用者が見る名前に揃える（`argument`/`option`/`column`/`value`/`n_obs`/`n_params`/`n_clusters`/`n_slopes`/`row`）。内部記号（`g`/`q`/`l`/`t`）は使わない。行番号は0始まり（メッセージに`(0-based)`と明記する）。
- 文字列値は、利用者のデータ（列名・値）、APIの識別子（引数名・オプション値・列挙値）、式（`condition`）に限る。英語の句は入れない（`INTERNAL_ERROR`の`detail`だけ例外）。
- `params`は5個まで（`stage`とその補助値は数えない）。超えそうなら`code`を分ける。
- しきい値は計算済みの値で渡し、条件で文が増減するテンプレートは作らない。

## 5. 入れ子の回帰（`stage`）

第一段階・within回帰・補助回帰などの中で起きたエラーを、`FIRST_STAGE_FAILED`のようなラッパーの`code`にすると、クラスが内側に追従し3章の不変条件と矛盾する。そのため、**原因の`code`をそのまま送出し、文脈を`params["stage"]`（と補助値）で付ける**。`message`は`stage`ごとの前置き（`In the first stage regression for 'x1': `）に原因のテンプレートを続ける。

- `stage`を付けられる`code`はカタログの`stage_allowed`で決める（観測数不足・自由度不足・特異行列など、入れ子の回帰で起きうるもの）。`confidence_level`の範囲違反のように、入れ子の回帰と無関係な`code`には付けない。
- エンジン側では、IVは`FirstStageFailed`等、パネルは`PanelError::InStage{stage, source}`で包み、`report()`が`stage`を足す。RE内部のwithin回帰（σ_ε²の推定）は、自由度不足・最小二乗の失敗・F検定の失敗のみ`VARIANCE_COMPONENT_WITHIN_REGRESSION`で包み、データの事実を述べる`SINGLETON_GROUP`と`PANEL_NOT_BALANCED_FOR_TWO_WAY`は包まない。
- 診断検定の補助回帰は、検定名を`test`（`WHITE`/`BREUSCH_PAGAN`/`BREUSCH_GODFREY`）で`stage`の補助値に持つ。

## 6. 実装

### engine（PyO3非依存）

- `engine/src/shared/error_catalog.rs`の`CATALOG`（Rustの`const`）が正本。1件は`ErrorSpec{code, class, params, message, hint, stage_allowed}`。JSON等の外部ファイルは使わない（実行時にパース・差し替えされる余地を作らず、コンパイル時に型検査できるため）。`message`/`hint`は`{name}`だけを置換できるテンプレートで、描画は`ErrorReport`が行う。
- 各エラー型は`engine::shared::error::ErrorInfo::report() -> ErrorReport`（`code`＋`params`）を実装する。`match`は全バリアントを列挙し（`_`で受けない）、足し忘れをコンパイルエラーにする。エラー型の`Display`にはカタログの文面を書かない（二重管理を避ける）。
- 描画: `Int`はそのまま、`Float`はRustの`{:?}`相当（`1.0`、`NaN`、`inf`）、`None`は`None`。文字列の引用符はテンプレート側に書く。
- 到達不能な状態は`INTERNAL_ERROR`（`detail`に元の文）。

### engine_pybind

- `#[pyclass(extends=PyValueError / PyRuntimeError)]`のマクロで`ValidationError`/`ComputationError`を生成する（属性定義の重複を避けるため）。`module="econometricsmodels._lib"`。
- 変換は`to_pyerr(&err)`の1関数。各系統の`*_error_to_pyerr`はこれを呼ぶだけ。`engine_pybind`発の入力不正（`ColumnExtractionError`・オプション検査）も同じ組み立て関数を通す。
- `_lib.error_catalog()`でカタログをPythonへ渡す（生成スクリプトとテストが読む）。`_lib.raise_error_for_test(code, params)`で任意の`code`を送出できる（カタログ全体の描画検査用）。

### 生成物

`python tools/render_error_catalog.py`が`python_package/econometricsmodels/error_codes.py`（`ErrorCode`）と`docs/guide/error-codes.md`の生成ブロックを作る。`--check`で再生成との差分を検出し、CI（`ci_python.yml`）で実行する。説明文はブロックの外に手書きする。生成にした理由は、手書き＋検査だとmdの表をパースする検査コードが要り、`ErrorCode`の生成のためにどのみち作るスクリプトより重くなるため。

## 7. テスト

- Rust: カタログの検査（テンプレートの`{name}`が`params`に存在する、文面規則、ゴールデン）は`error_catalog.rs`のテストと`error_catalog_golden.txt`。エラー型ごとの`report()`の単体テスト。ゴールデンの更新は`UPDATE_ERROR_CATALOG_GOLDEN=1 cargo test -p engine error_catalog`。
- `INTERNAL_ERROR`は、本物の`polars.DataFrame`の読み取りに失敗したときにPythonからも送出できる（`get_columns`が失敗するサブクラスで構成する。`tests/cross_method/test_data_input_types.py`）。
- Python: `tests/support/_errors.py`の`raises_error(code, **params)`で`code`・クラス・`params`のキーと型・`str(e)`/`args`を検査する。文面は照合しない（言い回しを直してもテストが壊れないように）。
- 全`code`の分類: カタログの`code`は`MIGRATED_CODES`（実際のエラー経路からの送出をテストで保証）か`UNREACHABLE_CODES`（Pythonから構成できない。理由つき）のどちらかに入る（`tests/cross_method/test_error_catalog.py`）。この分類の検査は通常のテスト。`pytest tests --check-error-coverage`は全体実行の終了時に、`MIGRATED_CODES`の全`code`が1度は送出されたこと、`UNREACHABLE_CODES`の`code`が送出されていないこと（送出できるようになったら`MIGRATED_CODES`へ移す）を検査する（絞り込み実行では無効）。
- カタログ全体は`_lib.raise_error_for_test`で、Rustの描画とテンプレートからの再構成が一致することを検査する。

## 8. 既知の制限

- `UNREACHABLE_CODES`の`code`（`BREUSCH_GODFREY_MATRIX_TOO_LARGE`・`VARIANCE_COMPONENTS_MOMENTS_SINGULAR`・`LINE_SEARCH_FAILED`・`SEPARATION_CHECK_FAILED`）と、`MATRIX_NEAR_SINGULAR`の`matrix`のうち`DESIGN_GRAM`・`HESSIAN`・`OPG`・`GMM_WEIGHT`は、実データから再現できていない（防御として残している）。再現できる入力が見つかったら`MIGRATED_CODES`へ移す。
- 分類（どの状況を`ValidationError`にし、どれを`ComputationError`にするか）の原則の見直しは、このシリーズの範囲外。移行時は現行のクラスを保った。
