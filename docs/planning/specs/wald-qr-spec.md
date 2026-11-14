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
| P1 | OLS・WLS（`fit_ols`、`wald_test_last_columns`は保持した小行列から末尾`q`を取る） | 完了。統計量だけの形は`0f2fa48f`、標準誤差も`M̃`から作る形（B）は未コミット |
| P2 | IV 2SLS（第一段階はOLS経由で自動、Wu-Hausman） | 完了（B、未コミット） |
| P3 | パネル（`panel_cov_params`、HC2/3のレバレッジ、RE・FE-IVのHausman・Wu-Hausman） | 完了（B、未コミット） |
| P4 | 限界を固定していたテストの置換、`docs/guide/validation.md`・各`CLAUDE.md`・`CHANGELOG.md`の更新、時間・メモリの計測、#535への影響の確認、GMM・FE-GMM・Tobitの別Issue | テスト置換・文書更新・時間とピークRSSの計測は完了（未コミット）。**#535・別Issueは未着手** |

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

### 計測結果と、OLSのQR統合（2026-11-14）

n=1M・k=6のOLS（classical）のステップ別（release、ms）: 列ピボットQR 140〜144、`solve_lstsq` 9、残差 9、`(X'X)⁻¹` 5、追加したピボットなしQR 87〜95、`Q`の生成 24、`Q'y` 3、meatは前後で同じ。増加分のほぼ全てが2本目のQRで、`(X'X)⁻¹`は全体の約3%。

そこで**OLSだけ**を、ピボットなしQR 1本から係数（`R⁻¹b̃`）・残差・共分散・全体Fを求め、ランク落ちは`R`（k×k）の列ピボットQR（`OrthonormalDesign::new_checked`、閾値は従来と同じ）で判定する形にした（ユーザー承認済み）。`cargo test -p engine` 1550件・`pytest tests --check-error-coverage` 14594件・clippy・fmtは通過。OLSは`least_squares`を使わないので`DESIGN_GRAM`は出なくなった。

変更前（`de8a6c8c`）との比較（3回の中央値。同じ変更前の値が計測間で最大25%ずれるので、約±15%以内は有意でない）:

| 手法 | 時間 | ピークRSS |
|---|---|---|
| OLS | −19〜−29% | +4〜14%（n=1M classicalは+14%。`Q`の48MB分） |
| IV（第一段階がOLS） | −2〜−5% | +0.4〜7% |
| RE | −1〜+22%（ノイズ内） | +2〜7% |
| FE | +25〜41%（未対応） | +1.5〜7% |
| FE-IV | +21〜27%（未対応） | +2〜10% |

次: FE・FE-IV（と、REの分散成分）へ同じ統合を広げるかをユーザーと決める（OLSのみで進め、有効なら広げる方針）。FE-IVはピークRSSの最適化済みなので、`x_hat`の寿命を延ばさないこと。

## 7. 未決事項（ユーザーに確認する。CLAUDE.md 14章）

**`least_squares`の`(X'X)⁻¹`（`DESIGN_GRAM`）をどうするか。** `shared/least_squares.rs`の`least_squares`は今も`X'X`のCholeskyで`(X'X)⁻¹`を求める（OLS・FE・IV・FE-IVでは使わなくなった）。この失敗（`RankDeficient::GramMatrix`→`MATRIX_NEAR_SINGULAR`の`DESIGN_GRAM`）が、`cond(X) ≳ 1e8`（傾きの差が相対`1e-8`以下）の設計を止める。選択肢:

- (a) 今のまま。未使用の`O(nk²)`の計算が残る（性能への影響は計測で見る）。受け付ける準共線の程度は変わらない
- (b) OLS・FE・IV・FE-IV用に、`(X'X)⁻¹`を作らない関数（係数と残差だけ）を足して切り替える。`cond(X)`が`1e8`〜`1e15`の設計を、QRで推定する範囲が広がる（係数の精度は`cond(X)·ε`で劣化する。Rの`lm`は相対`1e-7`で列を落とす）。`least_squares`の他の呼び出し元（弱操作変数診断・REの分散成分）は変えない
- (c) 閾値を別途決める

呼び出し元: OLS`linear/ols/fit.rs`、FE`panel/fe/fit.rs`、`shared/instrumental_variables.rs`の`two_sls_point_estimate`（IV・FE-IV）。他は`shared/weak_instruments.rs`・`panel/re/variance*.rs`が`xtx_inv`を使う。

## 8. 次の手順（`/clear`後の再開用）

1. 本ノートと`git -C /workspaces/econometricsmodels-wald log --oneline -3`・`git status`で状態を確認する。ワークツリーの`.venv`にはdebugビルドの拡張が入っている（`uv run --no-sync pytest`で使う）。
2. **時間・ピークメモリの計測（変更前後）**
   - 変更前: `/workspaces/econometricsmodels-base`（`de8a6c8c`のワークツリー）。release拡張は、同じ`.venv`のmaturinでビルドする: `cd /workspaces/econometricsmodels-base && /workspaces/econometricsmodels-wald/.venv/bin/maturin build --release -o <出力先>`。ホイールを`unzip`して`econometricsmodels/_lib*.so`を取り出す（前回のものはスクラッチパッド`/tmp/claude-1000/-workspaces-econometricsmodels/*/scratchpad/baseline_so/`にあるが、セッション固有なので無ければ再ビルド）
   - 変更後: `cd /workspaces/econometricsmodels-wald && uv run --no-sync maturin develop --release` の後、`python_package/econometricsmodels/`ディレクトリを丸ごと別の場所へコピーする。変更前は同じディレクトリのコピーに変更前の`.so`を入れる。計測後は`uv run --no-sync maturin develop`でdebugビルドに戻す（pytestに必要）
   - 実行: ワークツリーで`PYTHONPATH=<コピー先> RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 uv run --no-sync python -m performance.compare_<手法> --worker --library engine --cov-case <ラベル> --n <N> --k 5`。変更前後を交互に3回ずつ、時間の中央値とピークRSSを比べる（JSONのキーは1回実行して確認する）
   - 計測点: OLS（`classical`のn=1,000,000、`hc1`・`hc3`・`cluster_g50_random`のn=100,000）、FE（`classical`のn=1,000,000、`hc1`・`cluster`・`dk`のn=100,000）、RE（`classical`のn=1,000,000、`hc1`・`cluster`のn=100,000）、IV（`classical`のn=1,000,000、`hc1`・`cluster_g50_random`のn=100,000）、**FE-IV（`classical`のn=1,000,000が最重要。ピークメモリを減らす最適化`e1aa4635`の直後なので、ピークRSSが増えていないか）**・`hc1`・`cluster`・`hc3`のn=100,000
   - 判断: 時間の増加が大きい（目安として`classical`で+20%超）、またはFE-IVのピークRSSが増える場合は、`least_squares`の未使用の`(X'X)⁻¹`の除去（上の未決事項(b)）、HCのmeatで`Q`を行スケールして消費する形（`hc_meat`が作るn×kのスコア行列と`Q`が重ならないように）、ピボットなしQRと列ピボットQRの統合（`R`の列ピボットQRでランク判定）を検討する。結果は`docs/performance/`の該当ノートに残す必要は無い（計測値の正は`benchmark_performance.yml`のCI。CLAUDE.md 13章）が、判断の根拠としてユーザーへ報告する
3. 未決事項（`DESIGN_GRAM`）をユーザーに確認し、決まった方針を実装する。
4. コミット（署名が走るため、事前に「ホストの端末でパスフレーズの確認が出る」と伝える。`timeout`は120秒以上。失敗したら報告して止まる。`--no-gpg-sign`は使わない。`.claude/rules/commit-signing.md`）。未コミット分は1コミットにまとめる。メッセージ案: `feat(engine): IV・FE・RE・FE-IVの共分散と全体Wald/Fを直交座標で求め、高相関な傾きでも止まらないようにする`
5. レビュー: `rust-reviewer`と`testing-completeness-reviewer`（サブエージェント）に通し、指摘を反映する（#647のときと同じ手順）。
6. `release/v0.10.0`へ取り込む: 先に捨てる一時ワークツリーで`git cherry-pick -n`し、`cargo test -p engine`が通ることを確かめる（`release/v0.10.0`は先へ進んでいる）。その後、メインの作業ツリー（`release/v0.10.0`をチェックアウト済み）で`git cherry-pick <コミット>...`（署名の確認がホストに出る）。並行作業の未コミット変更とファイルが重ならないことを`git status`で確認する。pushとIssueのクローズは、ユーザーが指示してから。
7. 後片付け（ユーザーの確認後）: `git worktree remove --force /workspaces/econometricsmodels-wald`、`git worktree remove --force /workspaces/econometricsmodels-base`、`git branch -D feat/wald-qr-statistic`。取り込み後は内容が同じであることを`git diff`で確認してから消す。
8. #658の受け入れ条件の確認と、別Issueの起票（ユーザーの承認後。英語）: (a) IV GMM・FE-GMM・Tobitの直交座標の経路（設計行列のQRを持たず、breadが`Z'X`側・Hessian側なので別設計）、(b) #535（IV `card`のcluster共分散の精度）への影響の確認。IV 2SLSのcluster共分散は`V = R⁻¹M̃R⁻ᵀ`になったので、`RTOL_WALD_CARD_CLUSTER`（`tests/support/_tolerances_iv.py`、現状`5e-8`。`tests/iv/test_iv_reference.py`の`card`のcluster照合が使う）を測り直し、`1e-8`へ戻せるかを#535に報告する。
9. 完了したら、本ノートの内容を`docs/spec/`の該当手法の仕様書（`ols-spec.md`・`fe-spec.md`・`re-spec.md`・`iv-spec.md`）へ、経緯を削って理由だけにして集約し、本ノートを削除する（CLAUDE.md 13章）。

守ること（CLAUDE.md）: コード・仕様書・`CLAUDE.md`のコメントにIssue番号・`docs/planning/specs/`への参照を書かない（6章）。判断が分かれる点は独自判断で埋めずにユーザーへ確認する（14章）。メインの作業ツリーの並行作業には触れない（`git worktree`で作業する）。
