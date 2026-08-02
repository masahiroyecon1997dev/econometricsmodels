# Probit 内部実装ノート（数式・実装判断）

`docs/planning/specs/`配下。`nonlinear-api-design.md`・`nonlinear-implementation-notes.md`（nonlinear系統共通の設計・実装判断）とは別に、**Probit固有の数式導出・実装判断**をまとめる。`logit-implementation-notes.md`と同じ位置づけ。

## データ構造（Issue #70で実装済み）

`engine/src/nonlinear/probit.rs`の`ProbitInput::from_columns`。`engine::nonlinear::logit::LogitInput::from_columns`と同型（フィールド構成・切片列自動追加・次元検証ロジックとも同一）。`MleError`（`nonlinear/common.rs`）をそのまま再利用し、Probit固有のエラーバリアントは追加していない。

`y`が{0.0, 1.0}の二値であることの検証は、`LogitInput`と同様このIssueのスコープ外（次元検証のみ）。尤度・スコア・Hessianを実装する後続Issueで`validate_binary_y`（`nonlinear/common.rs`、Logit/Probit共通）による検証を追加する予定。

現時点で`ProbitInput`は`LogitInput`と完全に同型（フィールド・ロジックとも差分なし）。後続Issue（尤度・スコア・Hessian）でリンク関数がロジスティック関数`Λ(z)`から標準正規分布のCDF`Φ(z)`・PDF`φ(z)`に置き換わるが、これは`ProbitProblem`（`LogitProblem`相当）側の計算ロジックの差であり、`ProbitInput`自体の構造には影響しない見込み。仮に構造にも差分が必要と判明した場合は、この節を更新すること。
