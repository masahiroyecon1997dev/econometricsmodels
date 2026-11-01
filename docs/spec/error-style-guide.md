# エラーメッセージ スタイルガイド

`error_catalog.rs`の`message`・`hint`と、構造化しない`TypeError`の文面の規則。構造・カタログの仕組みは
`error-handling.md`。規則の一部（記号・非ASCII・内部語・競合パッケージ名・`hint`の形など）は
`engine/src/shared/error_catalog.rs`のテストが強制する。

対象: カタログの`message`/`hint`のテンプレート、`TypeError`の文面（構造化しないが同じ規則に従う）。

## 1. 構成

| 部分 | 規則 |
|---|---|
| `message` | 何が問題か（利用者の値と限界つき）＋なぜ。1〜2文。文末にピリオド |
| `hint` | どうすればよいか。命令形の1文（代替案は`, or`でつなぐ）。文末にピリオド。無い`code`もある |
| 全文 | `message`と`hint`を空白1つでつなぐ（`str(e)`） |
| 目安の長さ | 全文で300文字程度まで。それを超えるなら説明を公開ドキュメントへ回す |
| `stage`の前置き | `In the first stage regression for 'x1': `に続けて、`message`をそのまま（先頭は大文字のまま）続ける【既定の案X】 |

## 2. 文体

- 事実を述べる平叙文。`Error:`・`Invalid`・`failed to`・`please`・`!`を使わない。
- 主語は問題の対象（`Column 'x1' ...`・`hac_lags must be ...`）。「あなた」を主語にしない。
- 識別子（引数名・オプション名）は綴りのまま書き、文頭でも大文字にしない（`max_iter must be ...`）。
- 受動態は避ける。「〜できない」は理由とセットで書く（`cannot be put in time order because the values are tied`）。
- 内部の実装語を書かない: `Hessian`・`OPG`・`bread`・`simplex`・`GroupCodes`・Rustの型名・`faer`・`argmin`・`statrs`。
  必要な場合は利用者の言葉に言い換える（`a matrix needed to compute the covariance`）。`INTERNAL_ERROR`の`detail`だけは例外。
- 競合パッケージ名（statsmodels・linearmodels・fixest等）を書かない（拒否の理由は公開ドキュメントに書く）。
- `n`・`k`・`g`・`q`・`l`・`t`のような記号を書かない。単語で書く（`observations`・`coefficients`・`clusters`）。
- ASCIIだけを使う（`×`・`²`・`σ`・`X̂`を書かない。`R-squared`・`sigma`・`x`）。引用符は半角。

## 3. 引用符と記号（何を何で囲むか）

| 対象 | 書き方 | 例 |
|---|---|---|
| 引数名・オプション名・メソッド名 | 囲まずそのまま。メソッドは`()`を付ける | `cov_type`、`hac_time`、`white_test()` |
| API が受け付ける文字列値・列挙値 | 二重引用符 | `cov_type="cluster"`、`"hc1"` |
| 利用者のデータ（列名・グループの値） | 単一引用符 | `Column 'x1'`、`entity '3'` |
| 数値 | そのまま | `at least 2`、`got 0.5` |
| 式（`condition`） | そのまま（Python式風） | `estimator="gmm" and gmm_type="iterated"` |
| コード例 | 囲まずそのまま（バッククォートを使わない） | `df.with_row_index("t")`、`x=["x1"]` |

- バッククォートは使わない（端末・ログ・翻訳ツールで崩れやすい）。
- `None`・`True`・`False`は綴りのまま。

## 4. 用語

| 使う | 使わない | 備考 |
|---|---|---|
| observations | rows, samples, data points | 行の位置を指すときだけ`row` |
| regressors | independent variables, predictors, covariates, features | `x`の列のこと |
| coefficients | parameters, betas | 引数・オプションのことは`argument`/`option` |
| outcome | dependent variable, target | 引数のことは`y` |
| missing value (null) | NA, empty | polarsの`null`。`NaN`は`non-finite`として別に扱う |
| non-finite value (NaN or infinite) | invalid number | |
| design matrix | X matrix | |
| time periods | time points, waves | |
| entities | individuals, units, groups | `effects`の先頭の列（個体）のこと |
| clusters | groups | `cluster`引数のこと。パネルの個体は`entities` |
| instruments | IVs, excluded instruments | |
| fixed-effect parameters | dummies | |
| option / argument | parameter | `*Options`クラスのフィールドは`option`、呼び出しの引数は`argument`。名前だけで書けるときは名詞を省く |

## 5. 数値と値の描画

- `Int`: そのまま。`Float`: Rustの`{:?}`相当（`1.0`は`1.0`、`1.5`は`1.5`）、`NaN`は`NaN`、無限大は`inf`/`-inf`。`None`は`None`。
- 行番号は0始まりで、最初の言及に`(0-based)`を付ける。公開ドキュメントにRの1始まりとの違いを書く。
- 比較の言い方: `at least {min}`（以上）、`more than {n}`（超える）、`from {min} to {max}`（両端を含む範囲）、`less than`（未満）。
  「greater than or equal to」「<=」を書かない。
- しきい値は計算済みの値を使い、不等式を書かない（`at least {min_obs} are needed`、`n > k`を書かない）。
- リスト: `StrList`（API の値）は`"a", "b", "c"`（二重引用符、`, `で連結）、`ColumnList`（列名）は`'a', 'b'`（単一引用符）。「or」の文法を持たない。
- 複数形: 個数が決まらない名詞は`(s)`を使ってよい（`missing value(s)`）。言い換えで避けられるなら避ける。個数の`params`が1になりうる箇所（`{n_params}`・`{n_obs}`・`{n_iter}`・`{n_instruments}`の直後の名詞）は`(s)`にする（`1 coefficients`にしない）。下限が2以上の個数（`min_clusters`等）には付けない。
- 単位・割合: `1%`のように書く。

## 6. `hint`の書き方

- 動詞で始める命令形: `Use`・`Pass`・`Set`・`Remove`・`Add`・`Rename`・`Drop`・`Check`・`Rescale`・`Try`。
- 機械的な対処だけ: オプションを変える、データを足す、列を除く、列名を直す、前処理する。
  `cov_type`の良し悪し、モデルの当てはまり、識別の妥当性についての助言は書かない（このパッケージは推定量の良し悪しを助言しない方針のため）。
- 選択肢は多くて3つ。`Use ..., or ...`の形。具体名を書く選択肢は、`Use one of:`で`allowed`の値を示す場合に限る（`such as "hc1"`のような個別の推奨は書かない）。
- 回避策のコード例は短く（`df.with_row_index("t")`）、polarsのAPIは公開されたものだけ。
- 同じ`code`は常に同じ`hint`（3章の不変条件）。状況で対処が変わるなら`code`を分ける。
- 原因を断定できないときは`Check ...`で確認を促す（`Check the spelling against data.columns.`）。

## 7. 翻訳しやすさ（アプリ側で多言語化する人のために）

- 文を断片の連結で作らない。可変部分は全て`params`で渡す（テンプレートは1文の中で完結する）。
- `params`に英語の句を入れない（識別子・データ・式だけ）。
- 慣用句・比喩・省略（`e.g.`・`i.e.`・`etc.`）を避ける。例は`for example`で導く。
- 一つの`message`に複数の事実を詰めない。複雑なら`hint`か公開ドキュメントへ分ける。
- 数を単語に書かない（`two`ではなく`2`）。

## 8. 例

良い:
- `Column 'x1' (given as x) in data contains 3 missing value(s) (null). Missing values are never dropped automatically. Drop or impute them first, for example with drop_nulls().`
- `cov_type="cluster" needs at least 4 clusters, but the cluster column has 3. Use a cluster column with more distinct values, or set a different cov_type.`

悪い（と直し方）:
- `computation failed: failed to invert X'ZWZ'X (GMM bread matrix)` → 内部語・接頭辞。`A matrix needed to compute the estimates or their covariance is numerically singular. Rescale the variables so that they are on similar scales, or remove nearly collinear regressors.`
- `bandwidth must be in the range [0, t): got 7, t=6` → 記号`t`・区間記法・引数名なし。`dk_bandwidth must be from 0 to 5, one less than the number of unique time periods in dk_time (6), got 7.`
- `Please remove the \`const\` column.` → `please`・バッククォート。`Rename the column, or set include_intercept=False if it already is your constant.`

## 9. 新しいメッセージを足すときのチェックリスト

1. 既存の`code`で表せないか（`message`・`hint`・クラスが同じなら再利用）。クラスは「推定を走らせずに、入力を見るだけで分かるか」で決める（分かれば`ValidationError`、分からなければ`ComputationError`。分解などの数値ルーチンが要るものは後者。`error-handling.md`3.1章）。
2. 名前は「何の値か＋どういう状況か」（汎用名は対象が多岐にわたるときだけ）。
3. 利用者の値と限界が文に入っているか。記号を使っていないか。
4. `hint`は機械的な対処か。統計的な助言を含んでいないか。
5. 内部語・競合パッケージ名・非ASCII文字が入っていないか。
6. `params`は5個までか。文字列値は識別子・データ・式だけか。
7. カタログに追加し、`ErrorCode`と公開ドキュメントを生成し直し、`code`の送出テストを書いたか。

## 確認結果

1. 【決定】引用符の規則（3章）: 識別子は囲まず、API値は二重引用符、利用者データは単一引用符、バッククォートは使わない。
2. 【決定】`stage`の前置き後は大文字のまま（1章）。
3. 【決定】`option`は`*Options`クラスのフィールド、`argument`は呼び出しの引数（4章）。
4. 【決定】拒否の理由として「参照実装が無いため」を、`message`にも公開ドキュメントにも書かない。参照実装の有無は検証の都合で、
   利用者が取れる対処でも、結果の妥当性についての根拠でもないため。理由を書けるのは、論文などで明確に裏付けられる統計的な事実が
   ある場合に限る。それが無い間は、`hint`は選べる値を示すだけにする。統計的な助言をしない方針は堅持する。
5. 【決定】`hint`は1文（6章）。説明は`message`に移す。
