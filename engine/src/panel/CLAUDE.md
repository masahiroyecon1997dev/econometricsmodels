# engine/src/panel/ 実装ノート（FE/RE）

このファイルは `engine/src/panel/` 配下のファイルを読み書きするときだけ自動ロードされる。設計の背景・数式の正本は `docs/planning/specs/panel-api-design.md`（FE/RE論点はすべて確定済み）。ここに書くのは「削除するとClaudeが同じ判断を再発見するはめになる」レベルの実装上の決定事項・罠のみ。

## 実装済み（現状）

- `common.rs`: `PanelError`（FE/RE共有エラー型、Issue #172。`entity`/`time`の長さ不一致用`IdentifierDimensionMismatch`をIssue #175で追加）＋ `PanelDimension` enum ＋ `quasi_demean_column`（θパラメータ化した準偏差変換、Issue #173）＋ `hausman_statistic`（古典的ハウスマン検定の統計量、Issue #174）。
- `fe.rs`: `FeInput`（入力データ型、Issue #175）。`y: Vec<f64>` / `x: Vec<Vec<f64>>` / `entity: Vec<String>` / `time: Option<Vec<String>>`等を保持する薄い入れ物で、`OlsInput`/`IvInput`と異なり`faer::Mat`は組み立てない（`quasi_demean_column`が列単位・`&[f64]`で動く設計のため、`fit()`実装（Issue #178）が生の列をそのまま渡せる。詳細は`fe.rs`モジュールdoc）。`from_columns`は次元検証（`y`↔各`x`列・`y`↔`entity`・`y`↔`time`）のみ行い、within変換・singleton検出（#179）・分散ゼロ検証（#177）・バランスパネル検証・自由度調整（#180）・`OlsEstimator`への委譲（#178）はいずれも後続issueで`fe.rs`に追加する。
- `re.rs` は未着手。`fit()`本体（FE/RE共通）が未実装のため、`quasi_demean_column`/`hausman_statistic` は現時点でも `#[cfg(test)] mod tests` からのみ呼ばれる（`pub fn` なので `dead_code` にはならない）。

## faerのグローバル並列度（Issue #283）

- FE/RE の `fit()` エントリを実装するときは、**冒頭で `crate::parallelism::ensure_serial()` を呼ぶこと**（faer のグローバル並列度を `Par::Seq` に固定。OLS/WLS/Logit/Probit/Tobit/2SLS/GMM の各 `fit()` と同じ）。理由・背景は `engine/src/linear/CLAUDE.md`「faerのグローバル並列度」と `.claude/rules/rust-style.md`「パフォーマンス」節を参照。FE は within 変換後に `OlsEstimator::fit` へ委譲するため OLS 側の呼び出しでも一応担保されるが、`cargo test -p engine` で `PanelEstimator::fit` を直接叩く経路との統一のため各 `fit()` からも呼ぶ。回帰ガード（`fit_pins_faer_global_parallelism_to_seq`）も他系統に倣って1本入れる。

## 設計上の決定（再発見コスト削減）

- **`engine` はpolars非依存。`panel-api-design.md` 6.1節の「polarsの`group_by`で実装」は`engine_pybind`層／抽出後配列の話**。`engine`側のグループ集約は、クラスター列と同じく `entity: &[String]`（長さ`n`、行はパネル観測順）を受け取る。
- **`quasi_demean_column` の内部集約は `HashMap` でよい（`cluster_cov_params` の `BTreeMap` 必須とは別）**。あるエンティティの和は観測順（入力行の固定順）に積まれ、各行の出力はそのエンティティの和だけに依存する。エンティティ「間」をまたぐ加算（`Σ_g S_g S_g'` のようにグループ順序が浮動小数点丸めに効く演算）が無いため反復順序非依存でビット単位決定的。`HashMap` で集約・引き当てが O(n log G) → O(n)。**クラスターロバスト分散等、グループ間加算があるパネル系の実装を今後書くときは `engine/src/linear/CLAUDE.md` の `BTreeMap` 方針に戻ること**。
- **`quasi_demean_column` は列単位の関数**（`y`/`x`をまとめて受けない）。呼び出し側（FE/REの`fit()`）が `y` と `x` の各列にループ適用し、変換後の列を `OlsEstimator::fit` に渡す（WLSがsqrt(w)変換データをOLSへ委譲するのと同型、4.3節・7.4節）。列ごとに独立な変換のため単体テストが単純になる。
- **θの渡し方は `&BTreeMap<String, f64>`（エンティティID→θ_i）**。観測単位の`&[f64]`や「エンティティ順の`&[f64]`」にしない（順序規約をFE/REと共有する必要が生じ、ミスの余地が残るため）。**FEはこの関数の `θ_i = 1.0`（全エンティティ）の特殊ケース**として扱う（`panel-api-design.md` 7.4節）。REは `θ_i = 1 - sqrt(σ_ε² / (T_i·σ_u² + σ_ε²))`（7.2節）。
- **`quasi_demean_column` はグループ平均（`ȳ_i.`）を返さない**（Issue #173スコープ外）。`fixed_effects()` の `α_i = ȳ_i - x̄_i'β̂` 復元（6.6節）・σ_ε²再利用（7.4節）で平均の保持が必要になったら、そのFE/RE実装issueでこの関数を拡張する。
- **契約違反（`entity`と列の長さ不一致・`theta`のキー欠け）は`assert!`/`expect`でpanic**（`Result`を返さない）。ユーザー入力起因ではなく`engine_pybind`〜`engine`間の内部契約違反のため、`validate_cluster_groups`の`assert_eq!`と同じ扱い。`hausman_statistic`のshape契約（`beta_fe`/`beta_re`同長・`cov`が`k×k`）も同様に`assert!`。

- **`hausman_statistic` の設計（Issue #174、7.3節）**:
  - 入力は `beta_*: &[f64]` / `cov_*: &[Vec<f64>]`（`newton_step` と同じ形。RE呼び出し側が `Mat` から一度変換）。戻り値 `Result<(stat, df, p_value), CommonError>`。
  - **比較対象のalign（重なるスロープ係数のみ、REの切片・時間不変変数を除外）は呼び出し側（RE実装）の責務**。この関数は渡された `k` 個をそのまま使う。
  - 差行列 `Var(β_FE) - Var(β_RE)` は対称だが有限標本で非正定値になりうるため、Choleskyではなく `col_piv_qr` + `solve_lstsq`（`newton_step` と同じ相対閾値・NaN明示チェックの特異性検出）。
  - **統計量が負でもそのまま返す**（`plm::phtest` と同じ。`stat<=0` なら `p_value == 1.0`）。差行列が数値的に特異なときだけ `CommonError::ComputationFailed`。
  - **p値は `chi2.sf(stat)`（`1.0 - chi2.cdf(stat)` ではない）**。大きい `stat` で `cdf ≈ 1` になり小さいp値の相対精度が失われるのを避けるため（`sf` は正則化上側不完全ガンマを直接計算。`stat<=0` でも `1.0` を返すので負統計量の挙動は不変）。**`iv/gmm.rs` の Hansen J・Wald系は今も `1.0 - chi2.cdf` のまま**——一括で `sf` へ移行するかは別issue（このズレは意図的な暫定）。
  - `df` には常に `k` を使う。差行列の実効ランクが `k` 未満のとき `stat` と `df` に不整合が生じうる（`plm::phtest` も同じ制約）。
  - v1は classical Hausman のみ（`cov_type` 非依存）。robust版は将来issue。

## PanelError（Issue #172で確定済みの方針、変更時は要確認）

- FE/REで共有（`FeError`/`ReError`は作らない）。`CommonError` を `#[error(transparent)] Common(#[from] CommonError)` で包む（`LeastSquaresError`/`MleError`/`IvError` と同じ）。
- `WithinRegressionFailed { #[source] source: LeastSquaresError }` は `#[from]` を使わず明示的に `.map_err` で包む（`CommonError` が2経路で `PanelError` になる曖昧さを避けるため。`IvError::FirstStageFailed` と同じ判断）。
- RE固有バリアント（between回帰の自由度不足、Hausman統計量の非正定値ケース等）は未定義。RE実装issueで計算コードを書く過程で追加する（`common.rs` モジュールdocコメントの「追加候補」参照）。
