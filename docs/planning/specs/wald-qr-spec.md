# 全体Wald/F統計量のQR経路 設計ノート

高相関な傾きで、robust共分散（`hc*`・`cluster`・`hac`・`dk`）の全体Wald/F統計量の精度が落ち、`MATRIX_NEAR_SINGULAR`で止まる問題（#658）の設計。実装途中の設計ノートで、完了したら`docs/spec/`の該当手法の仕様書へ集約する（CLAUDE.md 13章）。単位変更で止まる問題は#647で解消済み（相関スケール化）で、本ノートはその続き。

## 1. 原因

サンドイッチ共分散`V = B M B`（`B = (X'X)⁻¹`）を作り、その傾き部分行列を反転して統計量を求めている。`B`は条件数が`cond(X)²`の行列の明示的な逆行列で、傾きの相関が`1 - 1e-11`程度を超えると`V`が桁落ちで数値的に正定値でなくなる（classicalは`σ²B`で桁落ちしにくく、`1 - 1e-15`程度まで通る）。1回目の逆行列（`B`）で失った精度は、2回目の反転（`V⁻¹`）で取り戻せない。

## 2. 方法

設計行列を、切片を先頭に置いた**ピボットなしQR** `X = QR`に分解する。`β = R⁻¹Q'y`、`V = R⁻¹ M̃ R⁻ᵀ`で、`M̃`は直交座標のミート（`x_i`を`Q`の`i`行目`q_i`に置き換えたもの）。

- 傾きの統計量: `W = b̃_S' M̃_SS⁻¹ b̃_S`。`b̃ = Q'y = Rβ`、添字`S`は末尾`q`成分。`R⁻¹`が上三角なので、末尾`q`個の係数は`b̃`の末尾`q`成分だけで決まる（`β_S = (R⁻¹)_SS b̃_S`、`V_SS = (R⁻¹)_SS M̃_SS (R⁻¹)_SSᵀ`）。
- 全手法で検定対象が末尾`q`列になっている（切片は先頭、FE・FE-IVは切片なし、REのHausmanとWu-Hausmanは末尾`k`列）。
- `M̃`は既存のmeat構築部品（`hc_cov_params`・`cluster_cov_params`・`hac_cov_params`・パネルの`panel_cov_params`）に`x = Q`・`xtx_inv = 単位行列`を渡すだけで得られる。小標本補正（HC1の`n/(n-k)`・`cluster_correction`・DKの補正）はスカラー倍なのでそのまま効く。HC2/HC3のレバレッジは`‖q_i‖²`になる。
- `wald_quadratic_form(params, cov_params, ...)`は`params = b̃`、`cov_params = M̃`を渡せばそのまま使える（相関スケール化と悪条件の判定も`M̃_SS`に対して働く）。新しいWald関数は要らない。
- 数値の確認（numpy、60個体×8期・切片＋2傾き・cluster、厳密な有理数演算との相対誤差）: `1 - corr`が`5e-13`で現行は非正定値、QR経路は1e-11。

## 3. 決定事項

1. **`cov_params`は`V = R⁻¹ M̃ R⁻ᵀ`で作る（B案）。** 1回の`M̃`計算で統計量と標準誤差の両方を得る（旧経路のmeat計算は消える）。旧経路との標準誤差の差を、完全共線の1件を除く凍結データで測った（numpy。classical・hc1・hc0・clusterのうち該当するもの）: OLS13件で最大`9e-11`（`cond(X) = 1e9`の`scale_variance`。他は`1e-13`以下）、FE36件で`6e-13`、FE-IV20件で`4e-14`、IV23件で`1e-12`。共分散の要素でも最大`2e-12`。参照実装との照合（1e-8）に対して十分小さく、パネル・IVも切り替えて問題ない。
2. **コミットは2段にする。** (1) `M̃`から統計量を求めるだけを配線（標準誤差は不変）、(2) `V`を`M̃`から作る形へ切り替え。差が出たときに`bisect`で切り分けられる。パネル・IVで差が大きければ、そのファミリーは(1)で止める（上の測定では該当なし）。
3. **対象手法**: OLS・WLS・IV 2SLS・FE・RE・FE-IV（Hausman・Wu-Hausmanを含む）。IV GMM・FE-GMM・Tobitは設計行列のQRを持たず、breadが別の形（`Z'X`側、Hessian側）なので別設計とし、現行の`cov_params`経路（相関スケール化済み）に残す。
4. **QRは2本持つ。** 既存の列ピボットQR（ランク判定・係数）は変えず、ピボットなしQRを足す。時間とメモリを測り、重ければ`R`の列ピボットQR（`k×k`）でランク判定を代用する案を検討する。
5. 失敗の契約は維持する: 本体のWald失敗は`FTestFailed`等で`fit()`全体が失敗、Wu-Hausmanは`None`、REのHausmanは`HausmanTestFailed`。

## 4. メモリ

`Q`（n×k）は共分散計算の窓の中だけで作り、直後に破棄する。HCの`hc_meat`はスコア行列（n×k）を作るので、`Q`と重ならないよう、行スケールを`Q`に直接かけて消費する形を検討する。FE-IVは`x_hat`の早期解放の最適化済みなので、`x_hat`の寿命を延ばさず、`x_hat`から`Q`を作る時点で`M̃`まで計算する。`performance/`の計測でFE-IVのピークRSSを前後比較する。

## 5. 段取り

| # | 内容 |
|---|---|
| P0 | 共有部品`shared/orthonormal.rs`（ピボットなしQR、`R⁻¹M̃R⁻ᵀ`）と固定値テスト（厳密値は有理数演算）。パネル・IVの凍結データで旧経路との標準誤差の差を測る |
| P1 | OLS・WLS（`fit_ols`、`wald_test_last_columns`は保持した小行列から末尾`q`を取る） |
| P2 | IV 2SLS（第一段階はOLS経由で自動、Wu-Hausman） |
| P3 | パネル（`panel_cov_params`、HC2/3のレバレッジ、RE・FE-IVのHausman・Wu-Hausman） |
| P4 | 限界を固定していたテストの置換、`docs/guide/validation.md`・各`CLAUDE.md`・`CHANGELOG.md`の更新、時間・メモリの計測、#535への影響の確認、GMM・FE-GMM・Tobitの別Issue |
