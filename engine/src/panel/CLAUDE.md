# engine/src/panel/ 実装ノート（FE/RE）

このファイルは `engine/src/panel/` 配下のファイルを読み書きするときだけ自動ロードされる。設計の背景・数式の正本は `docs/planning/specs/panel-api-design.md`（FE/RE論点はすべて確定済み）。ここに書くのは「削除するとClaudeが同じ判断を再発見するはめになる」レベルの実装上の決定事項・罠のみ。

## 実装済み（現状）

- `common.rs`: `PanelError`（FE/RE共有エラー型、Issue #172）＋ `PanelDimension` enum ＋ `quasi_demean_column`（θパラメータ化した準偏差変換、Issue #173）＋ `hausman_statistic`（古典的ハウスマン検定の統計量、Issue #174）。
- `fe.rs` / `re.rs` は未着手。いずれも呼び出し側は未実装で、`quasi_demean_column`/`hausman_statistic` は `#[cfg(test)] mod tests` からのみ呼ばれる（`pub fn` なので `dead_code` にはならない）。

## 設計上の決定（再発見コスト削減）

- **`engine` はpolars非依存。`panel-api-design.md` 6.1節の「polarsの`group_by`で実装」は`engine_pybind`層／抽出後配列の話**。`engine`側のグループ集約は、クラスター列と同じく `entity: &[String]`（長さ`n`、行はパネル観測順）を受け取り、`BTreeMap` で集約する（`engine/src/linear/CLAUDE.md`「クラスターのグループ化は`BTreeMap`」——パネル系のグループ集約もこの方針で揃える）。
- **`quasi_demean_column` は列単位の関数**（`y`/`x`をまとめて受けない）。呼び出し側（FE/REの`fit()`）が `y` と `x` の各列にループ適用し、変換後の列を `OlsEstimator::fit` に渡す（WLSがsqrt(w)変換データをOLSへ委譲するのと同型、4.3節・7.4節）。列ごとに独立な変換のため単体テストが単純になる。
- **θの渡し方は `&BTreeMap<String, f64>`（エンティティID→θ_i）**。観測単位の`&[f64]`や「エンティティ順の`&[f64]`」にしない（順序規約をFE/REと共有する必要が生じ、ミスの余地が残るため）。**FEはこの関数の `θ_i = 1.0`（全エンティティ）の特殊ケース**として扱う（`panel-api-design.md` 7.4節）。REは `θ_i = 1 - sqrt(σ_ε² / (T_i·σ_u² + σ_ε²))`（7.2節）。
- **`quasi_demean_column` はグループ平均（`ȳ_i.`）を返さない**（Issue #173スコープ外）。`fixed_effects()` の `α_i = ȳ_i - x̄_i'β̂` 復元（6.6節）・σ_ε²再利用（7.4節）で平均の保持が必要になったら、そのFE/RE実装issueでこの関数を拡張する。
- **契約違反（`entity`と列の長さ不一致・`theta`のキー欠け）は`assert!`/`expect`でpanic**（`Result`を返さない）。ユーザー入力起因ではなく`engine_pybind`〜`engine`間の内部契約違反のため、`validate_cluster_groups`の`assert_eq!`と同じ扱い。`hausman_statistic`のshape契約（`beta_fe`/`beta_re`同長・`cov`が`k×k`）も同様に`assert!`。

- **`hausman_statistic` の設計（Issue #174、7.3節）**:
  - 入力は `beta_*: &[f64]` / `cov_*: &[Vec<f64>]`（`newton_step` と同じ形。RE呼び出し側が `Mat` から一度変換）。戻り値 `Result<(stat, df, p_value), CommonError>`。
  - **比較対象のalign（重なるスロープ係数のみ、REの切片・時間不変変数を除外）は呼び出し側（RE実装）の責務**。この関数は渡された `k` 個をそのまま使う。
  - 差行列 `Var(β_FE) - Var(β_RE)` は対称だが有限標本で非正定値になりうるため、Choleskyではなく `col_piv_qr` + `solve_lstsq`（`newton_step` と同じ相対閾値・NaN明示チェックの特異性検出）。
  - **統計量が負でもそのまま返す**（`plm::phtest` と同じ。`p_value` は `1 - χ²_df.cdf(stat)` で `stat<=0` なら `1.0`）。差行列が数値的に特異なときだけ `CommonError::ComputationFailed`。
  - v1は classical Hausman のみ（`cov_type` 非依存）。robust版は将来issue。

## PanelError（Issue #172で確定済みの方針、変更時は要確認）

- FE/REで共有（`FeError`/`ReError`は作らない）。`CommonError` を `#[error(transparent)] Common(#[from] CommonError)` で包む（`LeastSquaresError`/`MleError`/`IvError` と同じ）。
- `WithinRegressionFailed { #[source] source: LeastSquaresError }` は `#[from]` を使わず明示的に `.map_err` で包む（`CommonError` が2経路で `PanelError` になる曖昧さを避けるため。`IvError::FirstStageFailed` と同じ判断）。
- RE固有バリアント（between回帰の自由度不足、Hausman統計量の非正定値ケース等）は未定義。RE実装issueで計算コードを書く過程で追加する（`common.rs` モジュールdocコメントの「追加候補」参照）。
