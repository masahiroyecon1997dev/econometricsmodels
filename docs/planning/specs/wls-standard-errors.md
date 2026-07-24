# WLS 標準誤差・適合度統計量の技術仕様

classical / HCロバスト（HC0〜HC3） / HAC（Newey-West） / cluster の重み付き版の計算式、および
適合度統計量の重み付き定義のまとめ。API・オプションの全体設計は
[`wls-api-design.md`](./wls-api-design.md)を参照。

**ステータス**: 設計提案中（Issue #34）。[`wls-api-design.md`](./wls-api-design.md)と同様、
2026-07-24時点のOLS実装（`engine/src/linear/ols.rs`）を前提にしている。

## 0. 結論: 新しい計算式の導出は不要

[`wls-api-design.md`](./wls-api-design.md) 4.1節・4.4節で確定した通り、WLSは
`X, y`を行ごとに`sqrt(w_i)`倍した`X̃, ỹ`（切片列を含む）を作り、**既存のOLS計算式
（`OlsEstimator::fit`）をそのまま**適用する。

$$
\tilde{x}_i = \sqrt{w_i}\, x_i, \qquad \tilde{y}_i = \sqrt{w_i}\, y_i, \qquad
\tilde{\varepsilon}_i = \tilde{y}_i - \tilde{x}_i^\top \hat\beta = \sqrt{w_i}\, \hat\varepsilon_i
$$

（$\hat\varepsilon_i = y_i - x_i^\top\hat\beta$は元スケールの残差）。以下の各節は、
「この変換後データにOLSの計算式をそのまま当てはめると、教科書的なWLSの重み付き公式と
一致する」ことを明示的に確認する内容であり、独自に新しい式を導出するものではない。
本issueの主な作業は、この前提の確認と、statsmodels `WLS`（内部的に同じ変換方式を採用）との
ベンチマーク照合（Issue #43）である。

## 1. classical（実装対象）

$$
\widehat{\mathrm{Var}}(\hat\beta) = \hat\sigma^2 (\tilde X^\top \tilde X)^{-1}
= \hat\sigma^2 (X^\top W X)^{-1}, \qquad
\hat\sigma^2 = \frac{\tilde\varepsilon^\top \tilde\varepsilon}{n-k}
= \frac{\sum_i w_i \hat\varepsilon_i^2}{n-k}
$$

$\tilde X^\top \tilde X = X^\top W X$（$W = \mathrm{diag}(w_1, \dots, w_n)$）であり、
これは教科書的なWLSのclassical分散公式そのもの。既定値（`cov_type`未指定時）もOLSと同じ
classicalとする（[`wls-api-design.md`](./wls-api-design.md) 7章）。

## 2. HCロバスト（実装対象: HC0〜HC3 の4種類すべて）

$$
\widehat{\mathrm{Var}}_{HC}(\hat\beta) = (\tilde X^\top \tilde X)^{-1} \hat\Psi (\tilde X^\top \tilde X)^{-1}
= (X^\top W X)^{-1} \hat\Psi (X^\top W X)^{-1}
$$

$\hat\Psi$は変換後の残差・設計行列（$\tilde\varepsilon_i, \tilde x_i$）に対する
[`ols-standard-errors.md`](./ols-standard-errors.md) 2章と同じ式で計算する。
$\tilde\varepsilon_i^2 = w_i \hat\varepsilon_i^2$、$\tilde x_i \tilde x_i^\top = w_i\, x_i x_i^\top$
なので、例えばHC0は

$$
\hat\Psi_{HC0} = \sum_i \tilde\varepsilon_i^2\, \tilde x_i \tilde x_i^\top
= \sum_i w_i^2\, \hat\varepsilon_i^2\, x_i x_i^\top
$$

となる（$w_i$が2乗で効くのは、残差と設計行列の両方に$\sqrt{w_i}$がかかっているため）。これは
「分析用重みで一次補正した上で、残る不均一分散に対してHeteroskedasticity-robustな補正を
さらに掛ける」という標準的な構成であり、statsmodelsの`WLS(...).fit(cov_type="HC0")`等が
内部的に計算しているものと同一である（`WLS`は`wexog`/`wresid`にHC系の式をそのまま適用する実装）。

HC2/HC3で使うレバレッジ$\tilde h_{ii}$も、変換後の設計行列$\tilde X$に対して計算する
（$\tilde h_{ii} = \tilde x_i^\top (\tilde X^\top \tilde X)^{-1} \tilde x_i$）。元の$X$に対する
レバレッジとは異なる値になる点に注意（`engine`側の実装は`OlsEstimator::fit`の既存コードを
そのまま再利用するため、意識的な実装作業は不要。この節は「それが正しい」ことの確認）。

4種類すべてを実装する理由はOLSと同じ（[`ols-standard-errors.md`](./ols-standard-errors.md) 2章）。

## 3. HAC（Newey-West、実装対象）

$$
\widehat{\mathrm{Var}}_{HAC}(\hat\beta) = (\tilde X^\top \tilde X)^{-1}\, \hat S \,(\tilde X^\top \tilde X)^{-1}
$$

$\hat S$は[`ols-standard-errors.md`](./ols-standard-errors.md) 3.1節と同じ式（Bartlettカーネル）を、
変換後の残差・設計行列（$\tilde\varepsilon_i, \tilde x_i$）に適用して計算する。

- **ラグ選択（3.2節相当）**: 経験則 $L = \lfloor 4(n/100)^{2/9}\rfloor$ は観測数`n`のみに依存し、
  重みには依存しない。WLSでも同じ式・同じ`hac_lags`オプションをそのまま使う。
- **時間順序（3.3節相当）**: `time_col`による時系列順序も、元の（重み変換前の）観測の時系列的な
  前後関係を表すものであり、重み付けの影響を受けない。ラグ付き自己共分散
  $\hat S_l = \sum_t \tilde\varepsilon_t \tilde\varepsilon_{t-l}\, \tilde x_t \tilde x_{t-l}^\top$
  の計算順序として、変換前と同じ`time_order`をそのまま使う。
- 重み付き回帰にHACを組み合わせるユースケース自体は実務上まれだが、`cov_type`は`y`/`x`/`weight`と
  独立した設定軸であるため、組み合わせ自体を禁止する理由はない。statsmodelsの`WLS`も同じ組み合わせを
  制限なく受け付ける。

## 4. cluster（実装対象）

$$
\widehat{\mathrm{Var}}_{CL}(\hat\beta) = (\tilde X^\top \tilde X)^{-1}
\left(\sum_{g=1}^{G} \tilde X_g^\top \tilde\varepsilon_g \tilde\varepsilon_g^\top \tilde X_g\right)
(\tilde X^\top \tilde X)^{-1} \cdot \frac{G}{G-1}\cdot\frac{n-1}{n-k}
$$

**クラスターのグループ分け自体（`cluster_col`が指す値によるグルーピング）は、重み変換の影響を
受けない**。各観測が属するクラスターは「その観測が何者か」という属性であり、重みで観測の値
（$y_i, x_i$）をスケーリングしても、グループの所属関係は変わらないため。グループ内で合計する対象
（$\tilde\varepsilon_i \tilde x_i$）だけが変換後の値になる。

小標本補正（$G/(G-1) \cdot (n-1)/(n-k)$）・自由度（t検定・信頼区間・F検定で`G-1`を使う）もOLSと
同じ（[`ols-standard-errors.md`](./ols-standard-errors.md) 5章）。$G$（クラスター数）・$n$（観測数）
は重みに依存しない値であり、変わらない。

## 5. 適合度統計量の重み付き定義

$$
R^2 = 1 - \frac{\tilde{\mathrm{SSR}}}{\tilde{\mathrm{SST}}}, \qquad
\tilde{\mathrm{SSR}} = \sum_i \tilde\varepsilon_i^2 = \sum_i w_i \hat\varepsilon_i^2, \qquad
\tilde{\mathrm{SST}} = \begin{cases}
\sum_i (\tilde y_i - \bar{\tilde y})^2 & \text{（切片ありのとき、centered）} \\
\sum_i \tilde y_i^2 & \text{（切片なしのとき、uncentered）}
\end{cases}
$$

- $\bar{\tilde y} = \frac{1}{n}\sum_i \tilde y_i = \frac{1}{n}\sum_i \sqrt{w_i}\, y_i$
  （**変換後の$\tilde y$の単純平均**であり、元の$y$の加重平均ではない点に注意）。
- 調整済み$R^2$・対数尤度・AIC・BIC・F統計量（ロバストWald検定への切替含む）は
  [`ols-api-design.md`](./ols-api-design.md) 6章・[`ols-standard-errors.md`](./ols-standard-errors.md)
  の式に、変換後の$\tilde{\mathrm{SSR}}$・$\tilde X$・$n$・$k$をそのまま代入したものと一致する。
- **statsmodelsとの整合性**: `RegressionResults.centered_tss`は、モデルが`wendog`
  （重み変換後のendog）を持つ場合はそちらを使って計算される。`WLS`は`wendog`を持つため、
  statsmodelsの`.rsquared`も上記と同じ「変換後データのTSS/SSR」ベースの定義になっている。
  したがって独自定義ではなく、**主リファレンスと同じ定義**である。

## 6. `OLSOptions`への影響

[`wls-api-design.md`](./wls-api-design.md)で確定した通り、WLS専用のOptions型は新設せず
`OLSOptions`をそのまま使う。本ドキュメントで確認した計算式はすべて`OLSOptions`の既存フィールド
（`cov_type` / `cluster_col` / `hac_lags` / `time_col`）の意味論をそのまま踏襲しており、
フィールドの追加・変更は不要。

## 7. 未確定・後続issueで扱う事項

- 上記の各計算式がstatsmodels `WLS`のベンチマーク値と相対誤差1e-8で一致することの実証
  → Issue #43（ベンチマーク作成）、Issue #44（tests/api_tests作成）
- HC0の$\hat\Psi$で$w_i$が2乗で効く点など、直感的に分かりにくい箇所はテストのdocコメントに
  計算根拠を明記する（`ols.rs`の既存テストのコメント密度に合わせる）
