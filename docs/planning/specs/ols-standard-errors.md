# OLS 標準誤差の技術仕様

GitHub Issue #3「OLS: 標準誤差の技術仕様確定（classical / HCロバスト / HAC）」の完了条件
（実装対象の標準誤差の種類・計算式・HACのラグ選択方法の確定）を満たすための設計まとめ。
API・オプションの全体設計は[`ols-api-design.md`](./ols-api-design.md)を参照。

**ステータス**: 技術仕様確定済み。実装は別issue
（#9 classical / #10 HCロバスト / #11 HAC）で行う。

## 1. classical（実装対象）

$$
\widehat{\mathrm{Var}}(\hat\beta) = \hat\sigma^2 (X^\top X)^{-1}, \qquad
\hat\sigma^2 = \frac{\hat\varepsilon^\top \hat\varepsilon}{n-k}
$$

- 既定値（`cov_type`未指定時のデフォルト）。[`ols-api-design.md`](./ols-api-design.md)で確定済み。

## 2. HCロバスト（実装対象: HC0〜HC3 の4種類すべて）

$$
\widehat{\mathrm{Var}}_{HC}(\hat\beta) = (X^\top X)^{-1} \hat\Psi (X^\top X)^{-1}
$$

| タイプ | $\hat\Psi$ の定義 |
|---|---|
| HC0 | $\sum_i \hat\varepsilon_i^2\, x_i x_i^\top$ |
| HC1 | $\dfrac{n}{n-k}\cdot \text{HC0}$ |
| HC2 | $\sum_i \dfrac{\hat\varepsilon_i^2}{1-h_{ii}}\, x_i x_i^\top$ |
| HC3 | $\sum_i \dfrac{\hat\varepsilon_i^2}{(1-h_{ii})^2}\, x_i x_i^\top$ |

$h_{ii} = x_i^\top (X^\top X)^{-1} x_i$（レバレッジ）。

4種類すべてを実装する。理由: 既に`OLSOptions.cov_type`の設計（[`ols-api-design.md`](./ols-api-design.md)3章）で
`"hc0"`〜`"hc3"`を受理する前提になっており、`.claude/rules/testing-policy.md`の不均一分散シナリオも
種別ごとの網羅比較を求めているため、一部のみ実装する理由がない。

## 3. HAC（Newey-West、実装対象）

### 3.1 推定量本体

Bartlettカーネルを用いた標準的なNewey-West (1987) サンドイッチ推定量を採用する
（statsmodelsの`cov_type="HAC"`の既定カーネルもBartlettであり、主リファレンスと一致させるため）。

$$
\widehat{\mathrm{Var}}_{HAC}(\hat\beta) = (X^\top X)^{-1}\, \hat S \,(X^\top X)^{-1}, \qquad
\hat S = \hat S_0 + \sum_{l=1}^{L} w_l \left(\hat S_l + \hat S_l^\top\right)
$$

$$
\hat S_l = \sum_{t=l+1}^{n} \hat\varepsilon_t \hat\varepsilon_{t-l}\, x_t x_{t-l}^\top,
\qquad w_l = 1 - \frac{l}{L+1} \quad \text{(Bartlett重み)}
$$

- $L$: ラグ数（バンド幅）。3.2節で確定。
- クラスターとHACは同時指定不可（`cov_type`は単一選択のまま。組み合わせは将来検討）。
- **正規化・スケーリングの厳密な係数は、実装issue（#11）着手時にstatsmodelsのソース
  （`statsmodels.stats.sandwich_covariance.cov_hac`相当）と突き合わせてビット単位で確認すること**。
  `testing-policy.md`の相対誤差1e-8方針を満たすには、$\hat S$の正規化定数を含めて完全一致させる必要があるため、
  本ドキュメントの式は「採用するカーネル・関数形」の確定であり、実装時の最終ソースはstatsmodels自体とする。

### 3.2 ラグ（バンド幅）選択方法: 固定ラグ＋経験則デフォルト

`OLSOptions`に`hac_lags: Optional[int]`を追加する。

| 指定 | 挙動 |
|---|---|
| `hac_lags=L`（整数指定） | 固定ラグ$L$を使用。statsmodelsの`cov_kwds={"maxlags": L}`と完全に同じ意味論 |
| `hac_lags=None`（未指定、デフォルト） | 経験則により自動計算: $L = \left\lfloor 4 \left(\dfrac{n}{100}\right)^{2/9} \right\rfloor$ |

- 完全なデータ依存の自動バンド幅選択（Newey & West 1994の、AR(1)近似に基づく最適バンド幅選択アルゴリズム）は
  **今回は実装しない**。理由: 主リファレンスのstatsmodelsに同等機能がなく、`maxlags`をユーザー指定必須とする
  設計のため、データ依存アルゴリズムを実装しても数値ベンチマークの照合手段がない。**将来の拡張候補として
  この節に記録するに留める**（着手する場合は、参照実装を新たに用意する必要がある）。
- 経験則デフォルトの式 $L=\lfloor 4(n/100)^{2/9}\rfloor$ はEViews等でも既定バンド幅として使われる、
  データに依存しない決定的な式である。ベンチマーク生成スクリプト（`benchmark/`）側でも同じ式で$L$を計算し、
  statsmodelsに`maxlags=L`として明示的に渡すことで、`hac_lags=None`のケースも1e-8精度で照合可能にする。
- `hac_lags`に負の整数、または`n`以上の値が指定された場合は`ValidationError`とする
  （具体的な妥当性条件は実装時に確定。目安: `0 <= hac_lags < n`）。
- `cov_type != "hac"`のとき、`hac_lags`は`cluster_col`と同様に無視する（エラーにしない）。

### 3.3 時間順序の扱い: `time_col`（OLS共通オプションとして追加）

`OLSOptions`に`time_col: Optional[str]`を追加する。

| 指定 | 挙動 |
|---|---|
| `time_col=None`（デフォルト） | `data`の行順をそのまま時系列順とみなす |
| `time_col="period"`等 | 指定列の値で昇順ソートしてからHACのラグ付き自己共分散を計算する |

- `time_col`は`extract_f64_column`と同様にf64にキャスト可能な列を要求する（整数の期間番号、UNIX時刻等を想定）。
  日付型（`Date`/`Datetime`）の直接サポートはPhase1のスコープ外とし、呼び出し側で数値表現に変換して渡す前提とする。
- `cov_type != "hac"`のとき、`time_col`は`cluster_col`/`hac_lags`と同様に無視する。

**設計上の位置づけ（`OLSOptions`固有かどうか）**: `time_col`はHAC以外の文脈（Phase4のFE/RE、Phase7の時系列手法）
でも将来必要になりうる概念だが、現時点では**`OLSOptions`固有のフィールドとして追加**し、共通オプション構造体への
切り出しは行わない。理由:
- Phase4のパネル手法は「entity + time」の2軸構造を前提とし、単一の`time_col`フラグより豊富な情報
  （エンティティ列との組を持つパネル構造）が必要になる可能性が高く、今の時点で共通形を先取りして決めると
  手戻りのリスクがある
- `engine_pybind/src/linear/mod.rs`のコメントが明記する既存方針（「この系統で共有するロジックが出てきたら
  common.rsを追加する。YAGNI」）と一貫させる
- フィールド名を`time_col`としておくことで、後で複数手法にまたがる共通構造体へ昇格させる際も
  機械的なリネームで済み、設計の再検討は最小限になる

## 4. `OLSOptions`への影響（[`ols-api-design.md`](./ols-api-design.md)への追記事項）

本issueでの決定に伴い、`OLSOptions`に以下の2フィールドを追加する（`ols-api-design.md`側の表も更新済み）。

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `hac_lags` | `int \| None` | `None` | `cov_type="hac"`のときのラグ数。`None`なら経験則で自動計算（3.2節） |
| `time_col` | `str \| None` | `None` | `cov_type="hac"`のときの時間順序列。`None`なら行順をそのまま使用（3.3節） |

`cov_type`が受理する文字列に`"hac"`を追加する（`engine_pybind/src/linear/ols.rs`の`CovType` enumにも
`Hac`バリアントを追加する想定。現状のenumにはまだ存在せず、実装issue #11で追加する）。

## 5. 参考: クラスター標準誤差（本issueのスコープ外・既存決定の再掲）

Issue #3のタイトル・内容（classical / HCロバスト / HAC）にクラスターは含まれないため、本issueでの
新規決定事項ではない。`cluster_col`オプション自体は[`ols-api-design.md`](./ols-api-design.md)（Issue #2）で
既に確定済み。計算式は標準的なStata方式の小標本補正（statsmodelsの`cov_type="cluster"`既定と一致）を
参考として記載する。

$$
\widehat{\mathrm{Var}}_{CL}(\hat\beta) = (X^\top X)^{-1}
\left(\sum_{g=1}^{G} X_g^\top \hat\varepsilon_g \hat\varepsilon_g^\top X_g\right)
(X^\top X)^{-1} \cdot \frac{G}{G-1}\cdot\frac{n-1}{n-k}
$$

**実装issue（#22）で確定した追加事項**: 上記の小標本補正は常に適用し、無効化するオプションは設けない
（`OLSOptions`に対応するフィールドを追加しない）。また、t検定・信頼区間・F検定の自由度は
`cov_type="cluster"`のときだけ`n-k`ではなく**`G-1`（クラスター数-1）**を使う
（statsmodelsの既定`df_correction=True`と一致させる、計量経済学の標準的な慣行。
標準誤差の値自体は自由度に依存しないため変わらないが、p値・信頼区間・F検定のp値はGが小さいほど
大きく変わる）。詳細は[`ols-implementation-notes.md`](./ols-implementation-notes.md)
「クラスター標準誤差」参照。
