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
- 分類（`ValidationError`/`ComputationError`）も`code`の属性。原則は3.1章。

## 3.1 分類の原則（`ValidationError`か`ComputationError`か）

- **`ValidationError`**: 引数・オプション・データだけで決まる検査。個数・範囲・重複・パネルの構造・列の単純な統計量。数値ルーチンは要らない。
- **`ComputationError`**: 数値ルーチン（行列の分解・逆行列、最適化、分布関数の評価、行列の確保など資源を要する処理）の実行中か実行後に分かるもの。推定値に依存する条件も含む。実装側の失敗（`INTERNAL_ERROR`）もここ。
- **クラスは検査の性質で決め、実装されている位置では決めない**。検査を`fit()`の前の方へ移しても（クラスター数不足がその例）、クラスは変わらない。
- **新しい`code`のクラスを決める問い**: 「推定を走らせずに、入力を見るだけで分かるか」。分かれば`ValidationError`、分からなければ`ComputationError`。設計行列のランクのように、分解などの数値ルーチンが要るものは「分からない」側に数える。
- クラスは粗い区別（「入力かオプションを直す」と「このデータではこのモデルを推定できない」）にとどめ、それ以上は`code`で分岐する。`ComputationError`の`code`にも、共線な列を除く・変数をリスケールするなど、入力を直す`hint`を持つものはある。1つの`code`は常に同じクラス（3章）なので、ラッパーのエラーは作らず、原因の`code`のクラスがそのまま例外のクラスになる（5章）。
- 利用者向けの説明は`docs/guide/validation.md`。

境界にある`code`と、現行のクラスの理由:

| `code` | クラス | 理由 |
|---|---|---|
| `REGRESSOR_NO_WITHIN_VARIATION`（FE） | `ValidationError` | 個体内変換後の列の単純な統計量。他の手法の`DESIGN_MATRIX_SINGULAR`は分解（ランクの判定）で分かるので`ComputationError` |
| `AMEMIYA_REQUIRES_WITHIN_VARIATION` | `ValidationError` | 同上 |
| `DK_TWO_PERIODS_DEGENERATE`・`CLUSTER_TWO_GROUPS_DEGENERATE` | `ValidationError` | パネルの構造だけで決まる |
| `NO_UNCENSORED_OBSERVATIONS`（Tobit） | `ValidationError` | 打ち切り境界と`y`の個数だけで決まる |
| `DIAGNOSTIC_VARIABLES_CONSTANT` | `ValidationError` | 診断検定の変数がすべて定数かは、列の統計量だけで決まる |
| `DIAGNOSTIC_VARIABLE_DEGENERATE` | `ComputationError` | 列は定数でないのに、二乗や標準化の計算が桁あふれ（非有限）になる。数値計算上の失敗で、列の統計量では決まらない（同じ大きさの定数列は`DIAGNOSTIC_VARIABLES_CONSTANT`） |
| `DIAGNOSTIC_STATISTIC_UNDEFINED` | `ComputationError` | 補助回帰の結果（残差・あてはめ）で決まる |
| `SEPARATION_DETECTED`・`SEPARATION_CHECK_FAILED`（Logit/Probit） | `ComputationError` | 判定に線形計画法（数値ルーチン）が要る。対処はデータの変更 |
| `UNCENSORED_INTERVAL_TOO_NARROW`（Tobit） | `ComputationError` | 推定した`sigma`に依存する |
| `NON_POSITIVE_IDIOSYNCRATIC_VARIANCE`・`VARIANCE_COMPONENTS_MOMENTS_SINGULAR`（RE） | `ComputationError` | 推定した分散成分・モーメント方程式に依存する |
| `NOT_CONVERGED`・`LINE_SEARCH_FAILED` | `ComputationError` | 最適化の結果 |
| `BREUSCH_GODFREY_MATRIX_TOO_LARGE` | `ComputationError` | 行列を確保できるかで決まる（資源の問題。入力の大きさだけでは決まらない） |

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

## 5.1 共線な列の報告（`DESIGN_MATRIX_SINGULAR`の`columns`）

設計行列が特異（完全な多重共線性）のとき、`params["columns"]`に**1つの線形従属に関わる列の名前**（設計の並び順）を入れる。

- 設計の列を並び順に直交化し、それまでの独立な列の線形結合で表せる**最初の列**`x_j`を見つける。`x_j`をそれらの列で表したときに係数が無視できない列（相対`1e-7`）と`x_j`を合わせた集合を報告する。前の独立な列で表す表現は一意なので、`x_j`について集合は一意に決まる。
- **どれを落とすかは決めない**。集合のどの列を落としても、その従属は解消する。「後ろの列を落とす」という規則は報告の選び方であって、利用者への指示ではない。
- **従属が複数あっても最初の1組だけを報告する**（仕様）。`params`は平坦な`list[str]`までしか持てず、複数組を構造のまま表せない。1組を直して再実行すると次の組が出る。
- 切片`const`も設計の列なので、関わる場合は含まれる（定数列と切片、ダミー変数の罠）。Logit/Probit/Tobitは最適化のために列を標準化（切片ありでは平均も引く）するが、列の特定は標準化前の設計の列で行うので、他の手法と同じ報告になる。ゼロ列、定数列が2つ、非有限値を含む列は、それぞれ1列または2列の従属になる。ゼロ列では`columns`が1列だけになる（専用の`code`は設けない。定数列が2つ以上あるかの判定など、別の検査を抱え込まないため）。
- 列名は、その回帰の設計の列名。IVの第二段階では予測値の列が内生変数名で出る。REの内部のwithin回帰は時間変化のある列だけを持つため、その名前が出る。補助回帰（Breusch-Godfreyのラグ）は元の列名を持たず`x[j]`・`residual_lag_l`という位置名で出る。White・Breusch-Paganの補助回帰は列を標準化（中心化）してからあてはめるので、切片との共線性（ダミー変数の罠）は切片を含まない形で出る。
- **探索は失敗しない**。診断が本来のエラーを隠さないよう`Result`を返さず、必ず空でない集合を返す。順序つきの探索で見つからなければ列ピボットQRが小さいと判定した列に、それも無ければ全列にフォールバックする。数値的に境界の入力では集合が近似になりうる。順序つきの探索の許容誤差（相対`1e-7`、Rの`lm`と同じ）はQRの特異判定より緩いため、完全な従属とは別に、ほぼ共線な列の組（例: 平均に比べて変動が極小の列）が先に見つかると、その組が報告されることがある。このとき、報告された列を直して再実行しても、次の従属が出る。
- 計算は失敗経路でだけ走る。直交化は最初の従属の位置`j`で打ち切るので`O(n·j²)`で、元のQRと同じ次数。検出は`engine/src/shared/linear_algebra.rs`の`collinear_columns`で、系統に依存しない（名前を知らない）。呼び出し側が自分の設計の列名に直す。ランク落ちの判定（`RankDeficient`）は列を持たず、列名が要る呼び出し側（`LeastSquaresError::from_rank_deficient`・`checked_design_matrix_qr`）だけが`collinear_columns`を呼ぶ。エラーを捨てる経路（IVのWu-Hausman拡張回帰、Newton法のHessian）では計算しない。

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

### panicの境界変換（`guard`）

- Rustのpanicは、そのままだとPyO3が`pyo3_runtime.PanicException`にする。これは`BaseException`の子で、`except Exception`で捕まらず、アプリ（GUI・BIツールのバックエンド・バッチ）のワーカーが落ちうる。
- `engine_pybind/src/shared/errors.rs`の`guard`が`catch_unwind`でpanicを捕まえ、`INTERNAL_ERROR`（`ComputationError`）にする。`detail`は`panic: `に続けてpanicの文言（最初の空でない行、200文字まで。超えたら` (truncated)`を付ける。文字列でないpayloadは固定の文言、空なら`(no message)`）。場所（ファイルと行）は既定のpanicフックがstderrに出すので`detail`には入れない。
- 境界の`#[pyfunction]`（非公開の`error_catalog`・`raise_error_for_test`を含む）と、処理を持つ`#[pymethods]`（`fit_*`・`predict`・`augment`・診断検定・`marginal_effects`等）の本体を`guard(|| { ... })`で包む。オプション・例外クラスのコンストラクタ・`__init__`・`__str__`・`__reduce__`・setter・`__repr__`・`get_all`のgetterは対象外（panicしうる処理を持たない）。包むのは本体だけで、戻り値のPythonオブジェクトへの変換はPyO3が`guard`の外で行う。**新しい境界の関数を足したら包む**（強制するテストは置かず、手順とレビューで担保する）。
- 正常系のコストはない（`catch_unwind`は正常終了のときは何もしない）。`AssertUnwindSafe`が妥当なのは、panicした呼び出しの結果を使わず、境界の関数が呼び出しをまたぐ可変な状態を持たないため。
- 前提は`panic = "unwind"`（ワークスペースの`[profile.release]`で明示）。`abort`にすると捕まえられずプロセスごと落ちるので変えない（`engine_pybind`は`cfg(panic = "abort")`でビルドエラーにする）。捕まえられないのは、abort・スタックオーバーフロー・メモリ不足・panic中のpanic。外部クレートの内部状態（例えばpyo3-polarsの初期化）は保証しない。
- 理由コメント付きの`expect`/`assert`は残す。`guard`が最後の受け皿になる（上に挙げた捕まえられないものを除く）。
- テストの入口は非公開の`_lib.panic_for_test(kind)`（`raise_error_for_test`と同じ扱いで、`python_package`からは再エクスポートしない）。本番の境界と同じ`guard`を通る。

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
