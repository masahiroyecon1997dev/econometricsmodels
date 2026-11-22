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
| `params` | `dict` | `message`/`hint`に埋め込まれた値。型は`int`/`float`/`str`/`bool`/`None`/`list[str]`のみ。`json.dumps`でそのままJSONにできる（非有限の`float`は文字列。4章） |

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
| `OUTCOME_CONSTANT`・`OUTCOME_CONSTANT_WITHIN_ENTITY` | `ValidationError` | 被説明変数の値だけで決まる（全行が同じ値、個体内変換後に分散が残らない）。推定を走らせずに分かる |
| `UNIT_LEVERAGE_OBSERVATION` | `ComputationError` | レバレッジは設計行列の分解を経て初めて分かる（入力を見るだけでは決まらない）。HC2・HC3の行スケール`1/(1-h)`が定義できない行を報告する（`PERFECT_FIT`が全体、これは1観測の完全フィット） |
| `UNIT_LEVERAGE_SLOPE_COVARIANCE` | `ComputationError` | 傾きのロバスト共分散が特異になる原因が、レバレッジ1の行だと分かったとき。全体F/Waldが失敗してから残差・レバレッジを調べて分かる（入力だけでは決まらない） |
| `PERFECT_FIT` | `ComputationError` | 推定した残差で決まる（残差平方和が総平方和の`k·ε`倍以下）。残差は推定してみないと分からない |
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
- `params`は7個まで（`stage`とその補助値は数えない）。超えそうなら`code`を分ける。7個にしているのは、列の特定（`column`・`argument`・`data_argument`の3つ）に、位置と値（`row`・`value`）と、その`code`固有の量（`COLUMN_NON_FINITE`の全件数`count`が1つ、`OUTCOME_OUTSIDE_CENSORING_BOUNDS`の境界`lower`・`upper`が2つ）を足すため。`error_catalog.rs`の`MAX_PARAMS`が検査する。
- **`data`の列の問題を報告する`code`は、列を同じ形で特定する**: `column`（1列）または`columns`（1つの引数の複数列。`ColumnList`）、`argument`（その列を渡した引数名）、`data_argument`（`data`・`new_data`）の3つを必ず持つ。`row`（0始まりの行番号）を持つ`code`は、1列の`column`・`argument`・`data_argument`も持つ（どの列の行かを示す）。`count`は1列の中の該当する値の数で、`row`・`value`は最初の1件（`COLUMN_NON_FINITE`）。1つの引数の列リストを見るだけの検査（`ROLE_DUPLICATE_COLUMN`・`ROLE_COLUMN_OVERLAP`）、出力の列名（`AUGMENT_COLUMN_EXISTS`）、設計行列の列名（`DESIGN_MATRIX_SINGULAR`の`columns`。`const`のように`data`に無い名前を含む）は`data`の列の問題ではないので、`data_argument`を持たない。例外は`error_catalog.rs`の`NO_DATA_COLUMN_TRIPLE`に理由つきで列挙し、`codes_about_a_data_column_name_the_column_the_same_way`が検査する。`role`は引数名とは別の概念（`C_STATISTIC_UNKNOWN_VARIABLE`の、対象が内生変数か操作変数か）にだけ使い、引数名は`argument`と呼ぶ。
- **非有限の`float`は文字列**: `NaN`・`inf`・`-inf`は`params`では`"nan"`・`"inf"`・`"-inf"`の文字列にする（`float('nan')`は`json.dumps`で不正なJSONの`NaN`になるため）。有限なら`float`のまま。変換は`engine_pybind`の`param_to_py`の1か所で、`engine::shared::error_catalog::non_finite_label`が表記を決める。文に埋め込む表記も同じ（`nan`・`inf`・`-inf`）にして、`params`の値をそのままテンプレートに入れればメッセージが再現できるようにする。新しい`code`が`Float`を足しても自動で守られる。`tests/support/_errors.py`の`raises_error`が、毎回`json.dumps(params, allow_nan=False)`が通ること、`float`は有限の`float`か3つの文字列のどれかであることを検査する。
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
- 列名は、その回帰の設計の列名。IVの第二段階では予測値の列が内生変数名で出る（設計の列名は`x_exog`と`x_endog`の名前）。REの内部のwithin回帰は時間変化のある列だけを持つため、その名前が出る。補助回帰（Breusch-Godfreyのラグ）は元の列名を持たず`x[j]`・`residual_lag_l`という位置名で出る。White・Breusch-Paganの補助回帰は列を標準化（中心化）してからあてはめるので、切片との共線性（ダミー変数の罠）は切片を含まない形で出る。
- **探索は失敗しない**。診断が本来のエラーを隠さないよう`Result`を返さず、必ず空でない集合を返す。順序つきの探索で見つからなければ列ピボットQRが小さいと判定した列に、それも無ければ全列にフォールバックする。数値的に境界の入力では集合が近似になりうる。順序つきの探索の許容誤差（相対`1e-7`、Rの`lm`と同じ）はQRの特異判定より緩いため、完全な従属とは別に、ほぼ共線な列の組（例: 平均に比べて変動が極小の列）が先に見つかると、その組が報告されることがある。このとき、報告された列を直して再実行しても、次の従属が出る。
- 計算は失敗経路でだけ走る。直交化は最初の従属の位置`j`で打ち切るので`O(n·j²)`で、元のQRと同じ次数。検出は`engine/src/shared/linear_algebra.rs`の`collinear_columns`で、系統に依存しない（名前を知らない）。呼び出し側が自分の設計の列名に直す。ランク落ちの判定（`RankDeficient`）は列を持たず、列名が要る呼び出し側（`LeastSquaresError::from_rank_deficient`・`ols_based_initial_params`）だけが`collinear_columns`を呼ぶ。エラーを捨てる経路（IVのWu-Hausman拡張回帰、Newton法のHessian）では計算しない。

## 5.2 engineの検査順（公開ガイドのstep 8）

`fit()`のengineの検査は、全手法で**設定値 → データの大きさに依る検査**の順に並べる（公開側の記述は`docs/guide/validation.md`の「Order in which errors are reported」step 8）。設定値は入力のデータを見ずに判定でき、データを見る検査（singleton・定数列・within変換・観測数・クラスター数）より先に報告しても、他の検査の相対順を変えない。手法ごとに検査を呼ぶ関数が違うため、新しい手法・検査を足すときは次の順に合わせる。

1. `confidence_level`（`(0, 1)`。NaNを含む）
2. `max_iter` → `tol`（MLE・FEとFEIVの交互射影）。GMMは続けて`gmm_max_iter` → `gmm_tol`（`gmm_type="iterated"`）
3. `y`の値・打ち切り境界（Logit・Probit・Tobit）
4. 観測数・自由度、クラスター数・時点数、定数のy、特異行列

理由と実装上の注意:

- **同じ検査を複数箇所に重ねてよいが、順序は先頭の1か所が決める**。IVは`validate_iv_fit_inputs`（`confidence_level`・`gmm_type`の反復設定を引数に取る）、FE・RE・FEIVは`panel::common::validate_confidence_level`をそれぞれの`fit`の先頭で呼ぶ。FEIVの第一段階やREの内部FEのように、入れ子の回帰が`confidence_level`をもう一度検査する箇所は、先頭で検査済みなので通らない（残してあるのは`fit_fe_from_within`などを単独で呼ぶ経路への備え）。
- `confidence_level`の範囲違反は`stage_allowed=false`の`code`なので、`WithinRegressionFailed`・`SecondStageFailed`のような回帰の包みで返しても`stage`は付かず、Pythonから見える`code`・`params`は包みの有無によらず同じ。
- `max_iter`は全手法で`i64`のままengineに渡し、負値もengineが範囲違反にする（pybindで`usize`へ変換しない）。GMMの`gmm_max_iter`もMLEの`max_iter`と同じ。
- Tobitの打ち切り境界は`TobitInput::from_columns`ではなく、`fit()`が上の順で呼ぶ`validate_censoring_bounds`で検査する（`from_columns`は次元だけ）。
- 行数が足りない入力と範囲外の`confidence_level`が重なる場合の順序は、`tests/cross_method/test_validation_order.py`の`test_confidence_level_is_reported_before_the_observation_count`が全手法で固定する。故障の組合せ（`FAULTS`の段階7）が`confidence_level` → `max_iter` → `tol` → `gmm_max_iter` → `gmm_tol` → `y`の値・境界の順位を固定する。

## 5.3 同じ`code`の該当列を全て報告する（`REGRESSOR_CONSTANT`・`REGRESSOR_NO_WITHIN_VARIATION`）

利用者が1回の実行で全ての該当列を直せるよう、`params["columns"]`（`ColumnList`）に**同じ`argument`の該当列を全て**、引数の並び順で入れる。

- 対象は、1つの`code`の中で、入力の列を走査するだけで分かる検査（数値ルーチンを使わない）に限る。別の`code`を1つのエラーにまとめない。列名のリストの長さは利用者が渡した列数で有界なので、件数の上限も`count`も持たない（`count`は`len(columns)`と同じ）。
- 引数（`argument`）は1つの文字列なので横断しない。FE-IVは`x_exog`・`x_endog`・`instruments`の順に調べ、最初に該当列がある役割だけを報告する（直すと次の役割が出る）。`REGRESSOR_CONSTANT`（生の値が定数）は`REGRESSOR_NO_WITHIN_VARIATION`（固定効果を除いた後）より先に、重い変換の前に調べる。
- `column`（単数）は`columns`に置き換え、`role`は`argument`に改名した（`data_argument`も足した）。同じ事実を2つのキーに持たせない。
- 位置を持つ検査は列を横断しない。`COLUMN_NON_FINITE`・`COLUMN_CONTAINS_NULL`は1列のままで、その列の全件数`count`を持つ。`COLUMN_NON_FINITE`の`row`・`value`は最初の1件で、`count`が1より大きいとき「最初の」と文面で示す。列を横断すると、`row`がどの列の位置か分からなくなり、`argument`も1つに決まらない。
- 全列の走査は、すでに終わっているwithin変換と同じ次数の費用で、失敗経路でだけ走る。

## 6. 実装

### engine（PyO3非依存）

- `engine/src/shared/error_catalog.rs`の`CATALOG`（Rustの`const`）が正本。1件は`ErrorSpec{code, class, params, message, hint, stage_allowed}`。JSON等の外部ファイルは使わない（実行時にパース・差し替えされる余地を作らず、コンパイル時に型検査できるため）。`message`/`hint`は`{name}`だけを置換できるテンプレートで、描画は`ErrorReport`が行う。
- 各エラー型は`engine::shared::error::ErrorInfo::report() -> ErrorReport`（`code`＋`params`）を実装する。`match`は全バリアントを列挙し（`_`で受けない）、足し忘れをコンパイルエラーにする。エラー型の`Display`にはカタログの文面を書かない（二重管理を避ける）。
- 描画: `Int`はそのまま、`Float`はRustの`{:?}`相当（`1.0`）で非有限は`nan`・`inf`・`-inf`、`None`は`None`。文字列の引用符はテンプレート側に書く。
- 到達不能な状態は`INTERNAL_ERROR`（`detail`に元の文）。

### engine_pybind

- `#[pyclass(extends=PyValueError / PyRuntimeError)]`のマクロで`ValidationError`/`ComputationError`を生成する（属性定義の重複を避けるため）。`module="econometricsmodels._lib"`。
- 変換は`to_pyerr(&err)`の1関数。各系統の`*_error_to_pyerr`はこれを呼ぶだけ。`engine_pybind`発の入力不正（`ColumnExtractionError`・オプション検査）も同じ組み立て関数を通す。
- `_lib.error_catalog()`でカタログをPythonへ渡す（生成スクリプトとテストが読む）。`_lib.raise_error_for_test(code, params)`で任意の`code`を送出できる（カタログ全体の描画検査用）。`raise_error_for_test`と`panic_for_test`は`cfg(debug_assertions)`で**デバッグビルドにだけ**含める。CIのテスト・`maturin develop`・`cargo test`はデバッグビルドなので使えるが、配布するwheel（`--release`）には入らない。`maturin develop --release`（性能計測用）で入れた環境では存在しないので、その後に`pytest`を走らせる前に`maturin develop`（デバッグ）で入れ直す（`uv run`・`uv sync`が拡張をreleaseで再ビルドした場合も同じ）。入れ直さずに走らせると、`tests/conftest.py`が`AttributeError`を検出して、それを使うテストを入れ直し方つきの失敗にし、ヘッダーに警告を出す（スキップはしない。スキップは件数が出るだけで、エラー`code`を送出するテストが黙って動かなくなるため）。CIは常にデバッグビルドなので失敗せず、`--check-no-skips`がスキップを1件でも検出すれば実行全体を失敗にする。

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

- `UNREACHABLE_CODES`の`code`（`BREUSCH_GODFREY_MATRIX_TOO_LARGE`・`VARIANCE_COMPONENTS_MOMENTS_SINGULAR`・`LINE_SEARCH_FAILED`・`SEPARATION_CHECK_FAILED`）と、`MATRIX_NEAR_SINGULAR`の`matrix`のうち`HESSIAN`・`OPG`・`GMM_WEIGHT`は、実データから再現できていない（防御として残している）。`DESIGN_GRAM`は、設計行列のQR（`OrthonormalDesign::new_checked`）が、列を揃えた条件数の上限（5e7）を超える設計を弾くときに出る（`tests/cross_method/test_collinearity_boundary.py`）。再現できる入力が見つかったら`MIGRATED_CODES`へ移す。
