# engine/src/linear/ 実装ノート（OLS/WLS）

このファイルは `engine/src/linear/` 配下のファイルを読み書きするときだけ自動ロードされる。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景・数式の導出は`docs/spec/ols-spec.md`・`docs/spec/wls-spec.md`が正本（このファイルはその要約ではなく差分の索引）。

## 踏んだ罠（再発防止）

- **クラスターのグループ化は`BTreeMap`、`HashMap`は使わない**: `HashMap`はプロセスごとのハッシュシードで反復順序が変わり、浮動小数点加算の非結合性により`fit()`を複数回呼ぶと標準誤差が1 ULP程度ぶれる非決定性バグを起こす（WLSで発覚、`fit_cluster_std_errors_are_deterministic_across_repeated_fits`で固定。詳細は`ols-spec.md`「標準誤差」クラスターの節）。クラスター系の実装を今後増やす場合も同じ罠がある。
- **`WlsEstimator`のR²・調整済みR²・log_likelihood（→AIC/BIC）は、変換後データに対する単純なOLS計算をそのまま使ってはいけない**。重み付き平均TSS・変換のヤコビアン補正項（`+0.5*Σlog(w_i)`）が必要（`wls-spec.md`「適合度統計量」）。`weighted_fit_statistics`関数（`WlsEstimator`層）で計算し直している。将来、重み付け系の手法（GLS等）を追加する際も同種の見落としに注意する。
- **係数計算は列ピボットQR（`col_piv_qr` + `solve_lstsq`）、`X'Xβ=X'y`をCholeskyで解く方式は使わない**（`X'X`の条件数が2乗になり不利、QRなら特異性検出と計算を同時に行える）。ただし`xtx_inverse`（標準誤差計算用の`(X'X)⁻¹`）は`X'X`自体のCholesky分解で求めている（QR分解の`R`因子から導出する案は実測で高速化しないことを確認済み、詳細は`docs/performance/ols.md`）。
- **クラスター数`G`と傾き係数の数`q`の関係（G≤qで構造的に特異）**: クラスターロバスト共分散`Ŝ`はG個のランク1行列の和なので`rank(Ŝ)≤G`。`wald_f_test`が使う`q×q`部分行列がG<qで特異になり`fit()`全体が`ComputationError`になる。「クラスタ数境界の成功パス」のテストを書くときはqをG以下に保つ。他のクラスターロバストSEを持つ手法（IV等）でも同じ制約が当てはまる。**追記（Tobit実装時に実測確認、Issue #220）**: `G=q`ちょうど（境界そのもの、`G<q`の厳密な意味では特異にならないはずの場合）でも、データの配置次第では部分行列が実際に特異になりうることを確認した（`engine/src/nonlinear/tobit.rs`の`fit_returns_computation_failed_when_wald_submatrix_is_singular_for_cluster_with_g_equals_q`）。`rank(Ŝ)≤G`は上限を与えるだけで`G`ちょうどの階数を保証しないため理論的には当然だが、「境界の成功パスのテストを書くときはqをG以下に保つ」は`q<G`の厳密不等号で読むこと（`q=G`は安全側に倒れるとは限らない）。
- **HACの行列積は`Par::Seq`を明示指定**: `k×k`という小さい出力サイズではfaerの既定並列実行のディスパッチオーバーヘッドが計算本体を上回り、素朴な並列化は逐次より遅くなる（実測n=10,000,k=2で6倍悪化）。この罠は「小さい行列の頻繁な積」全般に当てはまるため、他手法で同種の計算を書くときも並列化の要否を実測してから決める。
- **HAC以外は`n-k`、`cov_type=Cluster`のときだけ検定の自由度を`G-1`に切り替える**（`df_resid`自体、つまりσ̂²・調整済みR²・AIC/BICの計算に使う自由度は影響を受けず常に`n-k`のまま）。
- **相対閾値との比較（`diag <= threshold`）だけではNaNをすり抜ける**: `ensure_full_rank`（`col_piv_qr`のR対角成分の特異性判定）は、設計行列全体が完全にゼロ（`include_intercept=false`かつ全説明変数列がゼロ）だと、`col_piv_qr`が列選択時の0除算によりR対角成分にNaNを生成しうる（faer 0.24.4で実機確認済み）。NaNとの比較は常に`false`になるため、`diag.is_nan() || diag <= threshold`という形で明示的にNaNもチェックする必要がある（`nonlinear::common::newton_step`の同型の罠を先に踏んで修正済みだった、`nonlinear-implementation-notes.md`参照）。相対閾値で特異性を判定する他の箇所（IV等、将来col_piv_qrを使う手法）でも同じ罠がある。
- **非ピボットCholesky（`Llt`）のL因子対角成分は、設計行列の列間スケール差に起因する数値的なほぼ特異性を検出できない**（`ensure_full_rank`と同じ相対閾値の発想をL因子対角成分に適用しても検出できないことを実測確認済み）。`SelfAdjointEigen`による固有値ベースの判定が必要で、`engine::linear_algebra::ensure_well_conditioned_symmetric_matrix`として系統をまたいで共有している。詳細な導出・具体例は`ols-spec.md`「適合度統計量」参照。他手法で対称正定値行列の条件数チェックが必要になった場合も、非ピボットCholeskyの対角成分に頼らずこの共有関数を使う。
- **`OlsEstimator`は`cov_params`/`df_inference`を非公開フィールドとして保持する（Issue #164）**: 元々は`fit()`内のローカル変数として使い切っていたが、IV系統（`engine::iv::two_sls`）のWu-Hausman検定が「構造式に第一段階残差を追加回帰し、追加した係数だけのジョイントWald検定を行う」ために、`OlsEstimator::wald_test_last_columns(q)`（設計行列の**末尾q列**に対応する係数のロバストWald検定、既存の`wald_f_test`——切片を除く全傾き係数が対象——の対象列を一般化したもの）を新設する必要があり、その内部実装に`cov_params`/`df_inference`が要る。`OLSResult`（Python公開）には引き続き含めない（`ols-spec.md`「結果構造体」）。他系統がOLSの「一部の列だけを検定したい」ケースに直面したら、まずこのメソッドの再利用を検討する（サンドイッチ計算自体を複製しない）。

## 全手法共通ルールの再掲（見落とし防止）

- 推定量構造体（`OlsEstimator`/`WlsEstimator`等）のフィールドはprivate（`.claude/rules/rust-style.md`）。
- 特異性判定は相対閾値（`k * f64::EPSILON * |R[0,0]|`）。絶対閾値は不採用。
- `engine`はpolars/PyO3を知らない。列名の重複チェック等、列名が要る検証は`engine_pybind`側の責務（`engine_pybind/src/linear/CLAUDE.md`参照）。
