# engine/src/nonlinear/ 実装ノート（Logit/Probit/Tobit）

このファイルは `engine/src/nonlinear/` 配下のファイルを読み書きするときだけ自動ロードされる。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景・数式の導出は `docs/spec/logit-spec.md` / `docs/spec/probit-spec.md` / `docs/spec/tobit-spec.md` が正本（このファイルはその要約ではなく差分の索引）。

## 踏んだ罠（再発防止）

- **`ComputationError: the Hessian is singular and cannot be inverted` は「Hessianが特異」を意味するとは限らない（Issue #291、未修正・原因調査のみ完了）**: `common.rs` の `regularized_newton_step` は `MAX_LM_ATTEMPTS`（40）回の LM ステップ試行で `candidate_cost < cost`（狭義）を満たす候補を1つも見つけられなかった場合、無条件で `Err(MleError::SingularHessian)` を返す（`#[error(...)]` 文字列がそのまま `ComputationError` のメッセージになる）。この「コスト減少ステップが見つからない」状態は、**反復点が既にコスト関数の浮動小数点の底に到達しているのに収束判定が発火しない**ときにも成立する。
  - **収束判定側の根因**: `FaerNewton::terminate` の判定は `l2_norm(gradient) < self.tol`（既定 `tol=1e-6`）の**絶対しきい値のみ**。勾配は n 個の観測スコアの総和なので、収束点近傍で桁落ちが支配的になり、達成可能な勾配L2ノルムの「床」がデータ依存で `1e-6〜1e-5` オーダーに張り付く。`tol` は n でスケールしないため大標本では床がしきい値を超え、収束判定が永久に発火しない。相対勾配・`|Δcost|`・`‖Δθ‖` による副次的な停止条件は無い。
  - **失敗分類側の根因**: `regularized_newton_step` は「ステップノルムがパラメータスケール比で機械精度未満（＝既に最適点）」と「H が真に特異／降下方向が取れない」を区別せず、両方を `SingularHessian` に潰している（関数末尾のコメントが「再現データが見つかったらテスト追加」と予告していた経路そのもの）。
  - **切り分けの決め手**: 同一データで `tol` を `1e-5` 以上に緩める / `method="lbfgs"` にすると収束し、`observed_information_cov_params`（= `-H` の Cholesky 逆行列）も成功する。つまり収束点の `-H` は良条件で、メッセージは誤り。`cov_type="opg"` でも同じ `SingularHessian` が出る（opg 経路は `neg_hessian_inverse` を呼ばない）ことから、失敗は SE 計算ではなく**最適化ループ内**で発生している。
  - **再現**: `benchmark/nonlinear/datasets.py` の `generate_censored_regression_dataset("moderate_censoring", n, k=5, seed)` を `Tobit(...).fit()`（cov_type=classical 既定）。`n=200_000, seed=1` / `n=1_000_000, seed=42` 等で発生、`n<=150_000` や `seed` 次第では通る。py4etrics（`statsmodels.GenericLikelihoodModel` ベース、数値微分＋独自の収束基準）は同条件で収束するため engine 側の頑健性の問題。
  - **共有コードの問題であること**: 該当は `common.rs` の `FaerNewton`（`terminate` / `next_iter`）と `regularized_newton_step` で、Logit/Probit と完全に共有。Tobit で顕在化するのは、Tobit の `(β, logσ)` 尤度だけが不定符号領域で LM ラダーに実際に入る（＝`regularized_newton_step` を本気で使う）ことと、Tobit が大標本ベンチを持つため。Logit/Probit は大域凹＋分離ノルムチェックがあり同経路に入りにくいだけで、共有コードとしては同じ弱点を持つ。#284（Probit 大標本で Hessian 特異）と同系統。
  - **修正方針（着手時）**: 収束判定の頑健化（勾配しきい値の n スケール化 or 相対化＋`|Δcost|`/`‖Δθ‖` ベースの副次停止条件）と、`regularized_newton_step` の失敗分類（ステップが実質ゼロなら「収束」として現パラメータを返し `SingularHessian` にしない。`newton_step` が毎回 `SingularHessian` を返した場合のみ真の特異扱い）を**同時に**行う。検証は #291 完了条件どおり `n=1_000_000 seed=42` / `n=200_000 seed=1` で py4etrics・`AER::tobit` と係数・σ・logLik 一致、既存 Tobit テスト（小標本・悪条件）非リグレッション。
