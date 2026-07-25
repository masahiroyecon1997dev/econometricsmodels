# engine/src/linear/ 実装ノート（OLS/WLS）

このファイルは `engine/src/linear/` 配下のファイルを読み書きするときだけ自動ロードされる。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景・数式の導出は`docs/planning/specs/ols-implementation-notes.md`・`wls-implementation-notes.md`・`ols-standard-errors.md`・`wls-standard-errors.md`が正本（このファイルはその要約ではなく差分の索引）。

## 踏んだ罠（再発防止）

- **クラスターのグループ化は`BTreeMap`、`HashMap`は使わない**: `HashMap`はプロセスごとのハッシュシードで反復順序が変わり、浮動小数点加算の非結合性により`fit()`を複数回呼ぶと標準誤差が1 ULP程度ぶれる非決定性バグを起こす（WLS Issue #44で発覚、`fit_cluster_std_errors_are_deterministic_across_repeated_fits`で固定）。クラスター系の実装を今後増やす場合も同じ罠がある。
- **`WlsEstimator`のR²・調整済みR²・log_likelihood（→AIC/BIC）は、変換後データに対する単純なOLS計算をそのまま使ってはいけない**。重み付き平均TSS・変換のヤコビアン補正項（`+0.5*Σlog(w_i)`）が必要（`wls-standard-errors.md`5章）。`weighted_fit_statistics`関数（`WlsEstimator`層）で計算し直している。将来、重み付け系の手法（GLS等）を追加する際も同種の見落としに注意する。
- **係数計算は列ピボットQR（`col_piv_qr` + `solve_lstsq`）、`X'Xβ=X'y`をCholeskyで解く方式は使わない**（`X'X`の条件数が2乗になり不利、QRなら特異性検出と計算を同時に行える）。ただし`xtx_inverse`（標準誤差計算用の`(X'X)⁻¹`）は`X'X`自体のCholesky分解で求めている（QR分解の`R`因子から導出する案は実測で高速化しないことを確認済み、`ols-implementation-notes.md`11章）。
- **クラスター数`G`と傾き係数の数`q`の関係（G≤qで構造的に特異）**: クラスターロバスト共分散`Ŝ`はG個のランク1行列の和なので`rank(Ŝ)≤G`。`wald_f_test`が使う`q×q`部分行列がG<qで特異になり`fit()`全体が`ComputationError`になる。「クラスタ数境界の成功パス」のテストを書くときはqをG以下に保つ。他のクラスターロバストSEを持つ手法（IV等）でも同じ制約が当てはまる。
- **HACの行列積は`Par::Seq`を明示指定**: `k×k`という小さい出力サイズではfaerの既定並列実行のディスパッチオーバーヘッドが計算本体を上回り、素朴な並列化は逐次より遅くなる（実測n=10,000,k=2で6倍悪化）。この罠は「小さい行列の頻繁な積」全般に当てはまるため、他手法で同種の計算を書くときも並列化の要否を実測してから決める。
- **HAC以外は`n-k`、`cov_type=Cluster`のときだけ検定の自由度を`G-1`に切り替える**（`df_resid`自体、つまりσ̂²・調整済みR²・AIC/BICの計算に使う自由度は影響を受けず常に`n-k`のまま）。

## 全手法共通ルールの再掲（見落とし防止）

- 推定量構造体（`OlsEstimator`/`WlsEstimator`等）のフィールドはprivate（`.claude/rules/rust-style.md`）。
- 特異性判定は相対閾値（`k * f64::EPSILON * |R[0,0]|`）。絶対閾値は不採用。
- `engine`はpolars/PyO3を知らない。列名の重複チェック等、列名が要る検証は`engine_pybind`側の責務（`engine_pybind/src/linear/CLAUDE.md`参照）。
