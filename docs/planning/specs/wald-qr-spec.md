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

| # | 内容 | 状況 |
|---|---|---|
| P0 | 共有部品`shared/orthonormal.rs`（ピボットなしQR、`R⁻¹M̃R⁻ᵀ`）と固定値テスト（厳密値は有理数演算）。パネル・IVの凍結データで旧経路との標準誤差の差を測る | 完了・コミット済み（`0d033ecb`） |
| P1 | OLS・WLS（`fit_ols`、`wald_test_last_columns`は保持した小行列から末尾`q`を取る） | 完了・コミット済み（`77a34955`） |
| P2 | IV 2SLS（第一段階はOLS経由で自動、Wu-Hausman） | 完了・コミット済み（`77a34955`） |
| P3 | パネル（`panel_cov_params`、HC2/3のレバレッジ、RE・FE-IVのHausman・Wu-Hausman） | 完了・コミット済み（`77a34955`） |
| P4 | テスト置換・文書更新・時間とピークRSSの計測・QRの統合・条件数のガード・精度と境界のテスト | 完了・コミット済み（`b34c0eb4`・`f13fc094`・`5f640b03`・`cf30c291`）。**#535・別Issueは未着手** |

## 6. 進捗（2026-11-14時点）

作業場所: git worktree `/workspaces/econometricsmodels-wald`（ブランチ`feat/wald-qr-statistic`、`release/v0.10.0`の`de8a6c8c`起点）。メインの作業ツリー`/workspaces/econometricsmodels`には並行作業の変更があるので触らない。

コミット済み（署名済み）:

- `0d033ecb` 共有部品と本ノート
- `0f2fa48f` OLS・WLSの全体Wald/Fを直交座標で計算（標準誤差は旧経路のまま）

未コミット（ワークツリー内。`git status`で確認できる）:

- IV 2SLS・FE・FE-IV（Wu-Hausmanを含む）・RE（本体とHausmanのcluster・DK経路）を、直交座標で共分散と全体Fを求める形に変更。OLSを含む全てで`V = R⁻¹M̃R⁻ᵀ`から標準誤差を作る（B）
  - `shared/orthonormal.rs`: `coefficients`を削除、`coefficient_covariance`は対称化、`WaldCoordinates`（`b̃`と`M̃`）を追加、`#![allow(dead_code)]`を外した
  - `OlsEstimator::wald`・`FeEstimator::wald`（`b̃`と`M̃`を保持。`wald_test_last_columns`とFE-IVのWu-Hausmanが使う）
  - `TwoSlsPointEstimate.xtx_hat_inv`を削除（読む箇所が無くなった）
  - REのHausmanのcluster・DK経路は補助回帰の`Q`から`M̃`を作る。classical・hc*はOLSの補助回帰（`wald_test_last_columns`）経由
- テスト: FEの「サンドイッチの限界」を固定していたテストを、有理数の厳密値との一致検証へ置換（`tests/cross_method/test_unit_invariance.py`、厳密値は`tests/support/_exact_wald.py`の`df_resid`・`cluster_k`引数）。`test_wald_ill_conditioned.py`の2SLS（cluster・hac）の2件は、Bの形で`V`と統計量が整合して通る
- 文書: `CHANGELOG.md`、`docs/guide/validation.md`、`docs/spec/ols-spec.md`、`engine/src/linear/CLAUDE.md`（直交座標の項を追加）、`engine/src/panel/CLAUDE.md`、`shared/wald.rs`のdocコメント

検証結果（上の未コミットを含む状態）: `cargo test -p engine` 1550件、`cargo clippy --all-targets -- -D warnings`、`ruff check`・`ruff format --check`、`pytest tests --check-error-coverage` 14594件、すべて通過。

IV・パネルでAの形のコミットを作らなかった理由: `test_wald_ill_conditioned.py`の2SLS（cluster・hac）は、返された`cov_params`から厳密な二次形式を求めて統計量と比べる。Aの形（統計量は`M̃`、`cov_params`は旧経路）では、高相関で旧`V`が不正確なため一致せず、緑のコミットにならない。そのためIV・パネルとOLSのBは1コミットにまとめる（OLSのAの形は`0f2fa48f`として分離済み）。

実測の記録:

- 旧経路との標準誤差の差: 完全共線の1件を除く79件で最大`2e-12`（OLSの`scale_variance`だけ`9e-11`）
- FE・FE-IV・OLSの高相関（`1 - 5e-13`・`1 - 5e-15`）で、有理数の厳密値との相対誤差は`1e-11`〜`1e-9`
- classicalで`eps=1e-9`（`1 - 5e-17`）だったデータは、`MATRIX_NEAR_SINGULAR`にならず推定できるようになった（`least_squares`の`DESIGN_GRAM`は`eps ≤ 1e-8`では残る。下の未決事項）

### 計測と、QR 1本への統合・条件数のガード（2026-11-15）

最初の実装（`least_squares`を残して2本目のQRを足した形）は、OLS・FE・IV・FE-IV・REが変更前より+14〜70%遅かった（n=1M・k=6のOLSのclassicalで、増分のほぼ全てが2本目のQR。`(X'X)⁻¹`は全体の約3%）。そこで、全手法をピボットなしQR 1本（`OrthonormalDesign::new_checked`）に統合した: 係数`R⁻¹Q'y`・残差`y - Qb̃`・共分散・全体Fを同じQRから求め、ランク落ちは`R`（k×k）の列ピボットQRで判定する。`least_squares`は本体から外し、テストの独立な参照実装（`#[cfg(test)]`）にした。IV・FE-IVは`two_sls_point_estimate`が`OrthonormalDesign`を受け取り、FE-IVは`X̂`を`Q`生成の直後に解放して`X̃`をその後に作る（同時に持つ行列の数は変わらない）。REのbetween・プール回帰、`partial_f_statistic`の制限モデルも同じ関数を通す。

QRのままでは`cond(X)`が`1e8`〜`1e15`の設計も推定でき、残差が非ゼロのとき係数が静かに大きくずれる（傾きの差が相対`1e-13`で係数が数十〜数百%、標準誤差単位で約0.3）。そこで`new_checked`に、列を単位ノルムに揃えた`R`の条件数（k×kの特異値分解）が`MAX_EQUILIBRATED_CONDITION`（5e7）を超える設計を`MATRIX_NEAR_SINGULAR`（`DESIGN_GRAM`）で弾くガードを入れた。値は、変更前の`X'X`のCholeskyの成否の境界（列を揃えた条件数で2.8e7まで成功・9.2e7から失敗）に置いた暫定値。**生の条件数でなく列を揃えた条件数にする**のは、Choleskyの成否が列の一様な拡大縮小に依らず、列の単位（1e-6〜1e6）で境界が動かないようにするため。緩めるのは破壊的でなく、厳しくするのは破壊的。

変更前の挙動で不自然だった点も、これで解消した: (1) OLS・WLS・IVが条件数2e7まで通るのにFE・RE・FE-IVは2e6で止まる、(2) 同じ設計で`COEFFICIENT_COVARIANCE`と`DESIGN_GRAM`がepsごとに入れ替わる、(3) FE-IVで`INTERNAL_ERROR`（部分F検定の制限モデル）。480データセット×5手法（OLS・FE・RE・IV・FE-IV）で、結果の内訳が全手法で同じ（成功200・`DESIGN_GRAM`170・`DESIGN_MATRIX_SINGULAR`110）、`INTERNAL_ERROR`は0件。条件数が約1e12を超えると、faerのピボットなしQRが崩れ（最小ピボットが約1e-10以下で、後ろに独立な列があると`R`の最終行が0になる）、`DESIGN_MATRIX_SINGULAR`になることがある（ガードの閾値5e7より遥か外側で、受け入れる設計は変わらない）。ガードの範囲では逆行列誤差は1e-14程度。

変更前（`de8a6c8c`）との比較（release、3回の中央値。同じ変更前の値が計測間で最大25%ずれるので、約±15%以内は有意でない）:

| 手法 | 時間 | ピークRSS |
|---|---|---|
| OLS | −17〜−34% | +0.4〜14%（n=1M classicalは+13.6%。`Q`の48MB分） |
| FE | −6〜−20% | +0.7〜9% |
| RE | −10〜+12%（ノイズ内） | +1.5〜6.6% |
| IV | −22〜−27% | −0.3〜+1.8% |
| FE-IV | −12〜−26% | −0.4〜+5.1%（n=1M classicalは+5.1%） |

REの最終回帰と、Hausmanのcluster・DK経路は、`fit_auxiliary`（QR 1回）の後に同じ設計行列を`OrthonormalDesign::new`でもう一度分解している（`OlsEstimator`が`R`・`Q`を保持しないため）。計測では時間が変更前以下（clusterの+12%はノイズの範囲）なので、整理は見送った。

### 再レビュー後の追加（段階5、2026-11-15）

再レビュー（`rust-reviewer`・`testing-completeness-reviewer`）の指摘に対応した。

- **ランク判定も列を揃えた`R`で行う**（`cd58e20c`）。生の`R`の相対閾値`max|R_ii|`だと、共線な2列を1e-12倍するだけで`DESIGN_MATRIX_SINGULAR`として先に弾かれ、条件数のガードの不変性が活きなかった（変更前から同じ挙動）。列を揃えた行列を1度作り、ランク判定と条件数の両方を行う。共線な2列の単位を1e±12倍しても、6手法で境界が動かない。
- 失敗の契約（FE・FE-IVの全体F検定の`FTestFailed`、FE-IVのWu-Hausmanの`None`、REのHausmanの`HausmanTestFailed`）は、データから厳密に特異な共分散を作ると特異性の判定が丸め誤差の大きさで決まり不安定になるので、写像を小さな関数に切り出し、手組みの零行列の`M̃`で決定的にテストする（`panel/common/overall_wald.rs`・`panel/fe_iv/wu_hausman.rs`・`panel/re/hausman/tests/undefined.rs`）。`g = q+1`のクラスターのテスト（`test_unit_invariance.py`・`ols/cov_params/tests/cluster.rs`）は、固有値が約1e-17という丸め誤差級の特異性に依存したままで、堅牢な代替を作れなかった。
- **REの別名窓**: Hausman補助回帰とbetween回帰は相対残差1e-7未満の列を別名として落とす（Rの`lm`流。相関ランダム効果のための設計で、変更前から同じ）。傾きの相対差が約3e-7以下で`hausman_df`が2から1に減り、統計量と`θ`が変わる。条件数のガード（5e7）とは別の判定。既存の設計として残し、`validation.md`に明記し、挙動をテストで固定した（`test_near_collinear_reparametrization.py`）。
- **精度の検証**: 傾き2列を`(x1, (x2-x1)/2^k)`に厳密に置き換えた設計（引き算と2の冪での割り算は正確）との一致を、RE・hac・dk・Hausman・Wu-Hausmanで確かめる（実測の最大差7.3e-9。変更前の実装は同じ設計で最大39%ずれた）。有理数の厳密値との照合は、OLS・WLS・IV・FE・FE-IVの全`cov_type`のWald・係数・標準誤差（ロバスト標準誤差を含む）に広げた。変更前のRE（classical）のHausmanは、傾きの相対差1e-6で厳密値から1.1%ずれていた。
- IV GMM・Logit・Probit・Tobitは、ガードの対象外（GMMは共分散を作って反転する従来の経路で、近共線の設計に対して2SLSより厳しく止まる。Logit・Probit・TobitはHessianの境界）。FE-GMMはPythonのAPIに公開されていない。
- `RTOL_WALD_CARD_CLUSTER`は2SLSの特例を外した（相対差が8.5e-10・6.8e-11に改善。基本の許容誤差1e-8で比べる）。GMMの2つだけ`5e-8`を残した。

## 7. 解決済みの事項

`least_squares`の`(X'X)⁻¹`（`DESIGN_GRAM`）の扱い: 本体から`least_squares`を外し、`new_checked`の条件数のガードで同じ受け入れ範囲を保つ（上の「計測と、QR 1本への統合・条件数のガード」）。

## 8. 次の手順（`/clear`後の再開用）

1. 本ノートと`git -C /workspaces/econometricsmodels-wald log --oneline -6`・`git status`で状態を確認する。ワークツリーの`.venv`にはdebugビルドの拡張が入っている（`uv run --no-sync pytest`で使う）。最新のコミットは段階5cのもの（`git log`で確認）。検証の結果: `cargo test -p engine` 1567件、`pytest tests --check-error-coverage` 14899件、clippy・fmt・ruff通過。
2. 必要なら最終確認のレビュー（`rust-reviewer`・`testing-completeness-reviewer`）。メインの作業ツリーとは別のgit worktreeなので、サブエージェントには差分ファイル（`git diff de8a6c8c..HEAD`をスクラッチパッドへ書き出したもの）と、ワークツリーの絶対パスで読むよう指示する。再レビューの指摘は段階5で対応済み（上の「再レビュー後の追加」）。未対応は、`g = q+1`のクラスターのテスト（丸め誤差級の特異性に依存し、他OSで合否が変わりうる）。
3. `release/v0.10.0`へ取り込む: 先に捨てる一時ワークツリーで`git cherry-pick -n`し、`cargo test -p engine`が通ることを確かめる（`release/v0.10.0`は先へ進んでいる）。その後、メインの作業ツリー（`release/v0.10.0`をチェックアウト済み）で`git cherry-pick <コミット>...`（署名の確認がホストに出る）。並行作業の未コミット変更とファイルが重ならないことを`git status`で確認する。pushとIssueのクローズは、ユーザーが指示してから。
4. 後片付け（ユーザーの確認後）: `git worktree remove --force /workspaces/econometricsmodels-wald`、`git worktree remove --force /workspaces/econometricsmodels-base`、`git branch -D feat/wald-qr-statistic`。取り込み後は内容が同じであることを`git diff`で確認してから消す。
5. #658の受け入れ条件の確認と、別Issueの起票（ユーザーの承認後。英語）: (a) IV GMM・FE-GMM・Tobitの直交座標の経路（設計行列のQRを持たず、breadが`Z'X`側・Hessian側なので別設計）、(b) #535（IV `card`のcluster共分散の精度）への影響の確認。IV 2SLSのcluster共分散は`V = R⁻¹M̃R⁻ᵀ`になったので、`RTOL_WALD_CARD_CLUSTER`（`tests/support/_tolerances_iv.py`、現状`5e-8`。`tests/iv/test_iv_reference.py`の`card`のcluster照合が使う）を測り直し、`1e-8`へ戻せるかを#535に報告する。
6. 完了したら、本ノートの内容を`docs/spec/`の該当手法の仕様書（`ols-spec.md`・`fe-spec.md`・`re-spec.md`・`iv-spec.md`）へ、経緯を削って理由だけにして集約し、本ノートを削除する（CLAUDE.md 13章）。

## 9. 段階6の計画（最終確認レビューの指摘への対応。2026-11-15、未着手）

状態: ブランチ`feat/wald-qr-statistic`のHEADは`a1a87dbe`（段階5c）。`cargo test -p engine` 1567件・`pytest tests --check-error-coverage` 14899件・fmt・clippy・ruff通過。最終確認レビュー（`rust-reviewer`・`testing-completeness-reviewer`）の結果は**must fixなし**。以下はshould fix・nice to haveで、ユーザーが方針を承認済み（速度の記述は(a)）。コミットは6a・6b・6cの3つに分け、各段階の終わりでコミットして止まる（署名の確認がホストの端末に出ると事前に伝える。署名に失敗したら報告して止まり、リトライはユーザーの指示後）。

### 6a（実装・検証済み、コミット待ち。2026-11-15）: 文書・エラーカタログ・古い記述（公開ページは英語。`docs/spec/`・`.claude/`・`CLAUDE.md`への参照を入れない）

1. **公開ページの記述が実測・テストの裏付けを超えている箇所の訂正**（`docs/guide/validation.md`の「Near-singular matrices」行（302行付近）と92行付近、`CHANGELOG.md`、`docs/guide/verification.md`・`verification-details.md`）:
   - 「IV GMM・FE-IV GMM・Tobitはロバスト共分散だとclassicalより弱い依存で止まる」は誤り。GMMは`cov_type`に依らず、条件数が約2e5で止まる（IV-GMMのスイープ: classical・hc1・clusterで同じepsで止まる）。Tobitは未確認なので触れない。
   - 「`1 - 5e-15`を全`cov_type`でテスト済み」「厳密値と1e-8より良く一致」→「約1e-8」（実測はWald 2.6e-9・係数5.8e-9・標準誤差2.4e-9、許容誤差は3e-8・6e-8・3e-8）。REのテストは`eps ≤ 1e-6`まで、hac・dkは再パラメータ化のオラクルのみ（OLS・IV・FE・FE-IV）であることを正確に書く。
   - 「`1e-12`から`1e12`の間でテスト」→「`1e±6`と`1e±12`でテスト（共線な対を同じ倍率で）」。
   - REの別名窓: 「約3e-7以下」→「約4e-8〜3e-7（それ未満は条件数のガードで止まる）」。「設計行列」→FE・REでは「within/準偏差化した後の設計行列」（REの境界は他の手法より少し外側: seed 7のデータでOLS 3.72e-8・RE 4.51e-8）。
   - `card`×clusterのWald p値は約1.8e-9で絶対許容誤差のフロア1e-8より小さく、実質検査されていない（ivregとの相対差1.5e-8）。`verification.md`の2SLS行の「Waldとそのp値は8.5e-10」は統計量だけの値。p値はフロア未満であることを書くか、相対誤差だけで比べる（Sargan・Wu-Hausmanのp値と同じ扱い。許容誤差は約5e-8が要る）。
   - **速度の記述は(a)**: OLS（15〜35%）とIV（20〜25%）だけ数値を残し、FE・FE-IVは「同等かやや速い」（測定のばらつき約±15%の内側の数値は主張しない。性能の数値の正はCIの計測値、CLAUDE.md 13章）。ピークメモリ「OLSの100万行で最大約14%増」は実測（13.6%）どおり。
   - `verification.md`に、厳密値（有理数）・再パラメータ化の検証を短く足す（許容誤差3e-8・6e-8・1e-7。手法×リファレンスの表の外）。
   - 「`1e12`を超えると`DESIGN_MATRIX_SINGULAR`になることがある」は維持する（`rust-reviewer`は1e15付近のはずと指摘したが、実測はeps=1e-12・条件数約1e12で約30%のケースで`DESIGN_MATRIX_SINGULAR`）。
2. **エラーカタログの助言の書き分け**（実施: 「同じ`code`は常に同じ`hint`」の規則（error-style-guide 6章）があるため、`code`は分けず、助言を「ほぼ共線な回帰変数を除く・まとめる、またはスケールを揃える」の1文にした。`DESIGN_GRAM`専用の助言が要るなら`code`を分ける設計判断になるので別途確認する）: `MATRIX_NEAR_SINGULAR`の助言（`engine/src/shared/error_catalog_golden.txt:38`、`error.rs:221`の`NearSingularMatrix`のdoc）は「スケールを揃えるか、ほぼ共線な回帰変数を除く」だが、`DESIGN_GRAM`のガードは列の単位を変えても合否が動かない。`DESIGN_GRAM`のときは「単位を変えても解消しない。ほぼ共線な回帰変数を除く・まとめる」と書き分ける。`tools/render_error_catalog.py`で`docs/guide/error-codes.md`を再生成し、`docs/spec/error-style-guide.md`・`error-handling.md`の手順に従う。
3. **古い記述（`least_squares`・列ピボットQR・`X'X`のCholesky前提）の更新**: `engine/src/panel/re/variance.rs:105-106,222-223`、`panel/re.rs:22`、`panel/common.rs:43`、`panel/common/transform.rs:22`、`shared/covariance.rs:20`、`iv/two_sls/sargan.rs:31`、`shared/orthonormal.rs`のモジュールdoc（18-19行。`new`の呼び出し側が先に通すのは`OlsEstimator`経由の`new_checked`）、`docs/spec/ols-spec.md`（90-92行付近。「他の推定量は`X`の列ピボットQRで求める従来の形」は誤りで、2SLS・FE・RE・FE-IVも`new_checked`。GMM・Logit・Probit・Tobitだけが従来の形）。`tests/support/_tolerances_iv.py`の`RTOL_WALD_CARD_CLUSTER_GMM`のコメントに残る2SLSの数値（6.7e-11・8.5e-10）を除く。`shared/least_squares.rs`のファイル名と実態（残差平方和とテスト専用の参照実装）の不一致は、必要ならリネームを提案する（nice to have）。

### 6b（実装・検証済み、コミット待ち。2026-11-15。4: REの厳密値は返された`θ`（全個体で同じ）で準偏差化した設計の有理数の最小二乗で、`eps`は1e-5〜6e-8の5つの`cov_type`、dkは有限性のスモークテスト。5: 操作変数側は`zz`・`za`・`dd`の3構成。6: RE境界は約4.6e-8、片側スケール・単位不変・WLS hacを追加。7: 整数データの`g=q+1`置き換え済み）: テスト（`tests/cross_method/`）

4. **REのeps=1e-7（別名窓の中）の検査**: 今は`hausman_df`のみ。`res.theta`が公開されている（`python_package/econometricsmodels/panel/re.py:395`）ので、`θ`を有理数にして準偏差化し、`exact_inference`で最終回帰のF・係数・標準誤差を照合する。最低限、6つの`cov_type`が`eps = 1e-7`・`6e-8`でも有限の統計量を返すスモークテスト。
5. **操作変数側の近共線**（IV・FE-IV）: `z1≈z2`・`z3≈a`で、第一段階の`DESIGN_GRAM`（`stage=FIRST_STAGE`）と、`1e-13`での`DESIGN_MATRIX_SINGULAR`。第二段階の`X̂`のガード（`SecondStageFailed`）が`PERFECT_FIT`より先に出る構成には`d`にノイズが要る。
6. **境界テストの整理**（`test_collinearity_boundary.py`）: RE・FEは設計行列そのものでなくwithin/準偏差化後で判定されるので、境界がずれる（RE 4.51e-8。`_NEAR_INSIDE = 5e-8`の余裕がREだけ約11%、他は約35%。docstringの「全手法で1.2〜1.8倍」はREに当てはまらない）。共線な対の片方だけを1e±6・1e±12倍するケース、単位を変えても統計量が変わらないこと、WLSのhacの再パラメータ化を足す。再パラメータ化のオラクルのdocstring「丸めを増やさない」はOLS・IVだけに当てはまる（FE・RE・FE-IVは個体平均の丸めで約1e-9の摂動。許容誤差1e-7は妥当）。
7. **`g = q+1`のクラスターのテストの置き換え**（`tests/cross_method/test_unit_invariance.py:148`・`engine/src/linear/ols/cov_params/tests/cluster.rs:251`）: 現状のデータは共分散の厳密な行列式が+7e-16で厳密には特異でなく、合否が丸め誤差で決まる（他OSで保証されない）。レビューが検証した置き換え: **OLSの残差が4クラスターのうち厳密に2つにしか乗らない設計**（`g=4 > q=3`で事前検証は通る）。`x1, x2, x3`を乱数で作り、クラスター0・1の行に制限した`X'`の零空間の2本の基底ベクトルを取り、`y = Xβ + r`とする。すると`M̃`のランクが高々1で、残りの固有値は約1e-32（閾値より10桁以上小さい）。Pythonのseed 0〜7の8件すべてで`MATRIX_NEAR_SINGULAR`（`matrix=COEFFICIENT_COVARIANCE`）が出た。Rust側は整数データと整数の`r`で`y`をf64で厳密にする。これでPythonから再現できる唯一の`COEFFICIENT_COVARIANCE`の経路が堅牢に残る（`tests/support/_errors.py`のコメント・`validation.md`が主張している）。段階単位の`overall_f_test`の単体テストは残す。

### 6c: Rustの細部（nice to have）

8. `decompose`で`R`の全要素が有限という判定を足す（NaN列の扱いが「faerの列ピボットQRがNaNの対角にNaNを残す」ことに依存しないように。k²で安価）。`equilibrate_columns`の`norm`を`(r/largest)/sqrt(sum)`の形に（`largest`が`f64::MAX`近傍でのオーバーフロー回避）。`condition_number`の`singular_values[0]`・`[len-1]`を`first()`/`last()`で`INFINITY`に落とす形に。`shared/orthonormal.rs`のテスト間の空行（434-435行付近）。
9. IV 2SLSで`x_structural`の作成を`new_checked`の後に移す（`iv/two_sls/fit.rs:73-80`。FE-IVと同じ方針で、`new_checked`中のピークをn×k分減らす）。
10. 大きなn（1e4〜1e5）の完全共線（重複列・ダミー変数トラップ）が`Columns`になるテスト。ランク判定の閾値は等化した`R`に対する絶対`k·ε`で、Householder QRの丸めで完全重複列の`R_jj`が閾値を超えると`DESIGN_GRAM`になり、列名つきの`DESIGN_MATRIX_SINGULAR`にならない可能性がある（旧`checked_col_piv_qr(X)`も同じ式で退行ではない可能性が高いが、数値は未確認。先に実測し、`DESIGN_GRAM`になるなら、挙動を決める必要があるのでユーザーに確認する）。

### 手順

各段階: 実装 → `cargo fmt`・`cargo clippy --all-targets -- -D warnings`・`cargo test -p engine`・`ruff check/format --check tests`・`uv run --no-sync maturin develop`（debugビルドに反映）・`pytest tests --check-error-coverage`（約8分。xdistは入っていないので`-n`は使わない）→ ユーザーに結果を報告し、署名付きコミット。公開ページを変えたら`tests/cross_method/test_public_docs.py`・`test_doc_examples.py`・`test_docstring_examples.py`を通す。6aの後は`/review-docs`（公開ドキュメントの校閲）を、ユーザーの意向を確認してから検討する（変更しただけでは自動実行しない。CLAUDE.md 13章）。

段階6のあと、8章の手順（`release/v0.10.0`への取り込み、後片付け、#658・#535・別Issue、`docs/spec/`への集約）に進む。

守ること（CLAUDE.md）: コード・仕様書・`CLAUDE.md`のコメントにIssue番号・`docs/planning/specs/`への参照を書かない（6章）。判断が分かれる点は独自判断で埋めずにユーザーへ確認する（14章）。メインの作業ツリーの並行作業には触れない（`git worktree`で作業する）。
