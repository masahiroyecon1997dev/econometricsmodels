# 引数とOptionsの置き場所（全手法共通）

手法ごとの引数の置き場所は、この章の基準に従う。各手法のspecは基準を複製せず、ここへリンクする。
公開側の説明（英語）は`docs/guide/api-conventions.md`。引数の名前と置き場所の一覧表は公開ページが正本で、
ここには選択理由だけを置く。

## 1. 基準

**必須の列は引数、任意または条件付き必須の設定は`*Options`のフィールド。**

- 必須の列引数は「モデルを定義する入力」で、`data`の直後に並べ、デフォルト値を持たない
  （`y`・`x`・`weight`・`x_exog`・`x_endog`・`instruments`・`entity`）。
- `cluster`・`hac_time`・`dk_time`・`time`は列名を取るが、任意または`cov_type`等に応じた条件付き必須
  なので`Options`に置く。「列名だから引数」にはしない。
- `options`は末尾のキーワード専用引数（デフォルト`None`）。デフォルト値のある引数はすべて`*`の後ろに置く
  （`tests/cross_method/test_keyword_only_arguments.py`が検査する）。

判定は「必須か任意か」の1軸だけにする。以前は「推定方法の設定かモデルを規定するデータか」という
別の軸の説明が混在し、同じ列でも軸によって置き場所が変わって見えたため、機械的に判定できる
必須／任意に統一した。

## 2. 命名

- 1列は`str`で単数形（`y`・`weight`・`entity`・`cluster`・`hac_time`）。
- 複数列は`list[str]`で、`x`・`x_<役割>`・複数形（`x`・`x_exog`・`x_endog`・`instruments`）。
- 列名を取る引数に`_col`等のサフィックスは付けない。

## 3. 名前を変えないと決めたもの

- **IVの`x_exog`/`x_endog`/`instruments`**: 外生と内生の区別が手法の本質で、`x`1つでは区別が隠れる。
  linearmodelsの`IV2SLS(dependent, exog, endog, instruments)`・Stataの`ivregress`と対応する語彙。
  詳細は`iv-spec.md`1.1節。
- **WLSの`weight`**: 1列（`str`）なので単数形で、`y`・`cluster`と同じ規則。`weights`と入力した利用者には
  Pythonの`TypeError`が`Did you mean 'weight'?`と提案する。別名の併用は、1つの概念に名前が2つできるため
  受け付けない。詳細は`wls-spec.md`1章。
- **未知の引数名に手法固有のヒントを出さない**: `__init__`に分岐が増えて、Python側を薄く保つ方針
  （CLAUDE.md 2章）に反するため。

## 4. テスト

`tests/cross_method/test_call_signatures.py`が全推定クラス（`<名前>Options`と`<名前>Results`を持つ
公開クラス）の署名を検査する。

- 先頭が`data: pl.DataFrame`・`y: str`、`options`が最後でキーワード専用・デフォルト`None`。
- `y`と`options`の間は必須の列引数だけで、`str`か`list[str]`、命名は2章に従う。
- 必須の列引数と同名のフィールドを`*Options`が持たない。
- 必須の列引数の名前が公開ガイドの表に載っている（手法追加時のガイドの更新漏れの検出）。

引数名の閉じたリストは持たない。新しい名前は規則に沿えば通り、表に書き足す作業だけがテストで強制される。
