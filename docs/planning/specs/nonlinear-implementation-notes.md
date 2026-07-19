# 非線形モデル（MLEベース） 内部実装ノート（パラメータ設計以外）

`docs/planning/specs/`配下。非線形モデルのAPI・オプション設計（`nonlinear-api-design.md`）とは別に、**パラメータ以外の内部実装で決めたこと・まだ決まっていないこと**をまとめる。OLSの`ols-implementation-notes.md`と同じ位置づけ。実装issue（engine関連）着手時に必ず参照すること。

**現状**: Logit/Probit/Tobitともに実装未着手（Issue化前）。本ノートはOLSのように「Issue #Nで実装済み」を積み上げていく形ではなく、設計段階で先に決まった内部実装レベルの事項を記録する。実装着手後は同じ形式（`Issue #Nで実装済み`）で追記していく。

## 確定事項

### エラー型: discrete_choice系統で共有（`MleError`）

- `engine/src/discrete_choice/common.rs`に**Logit/Probit/Tobit共有のエラー型**（仮称`MleError`）を1つ定義し、3手法の`fit()`がこれを使う。OLSの「1手法1エラー型（`OlsError`）」パターンをそのまま横展開せず、共有型にする
- 理由: `raise_on_non_convergence`未収束・観測数不足・`confidence_level`範囲外等、3手法でほぼ共通のバリアントが多く、手法ごとに別エラー型にすると同じ意味のバリアントが3箇所に重複する。`.claude/rules/rust-style.md`「系統内で共有するロジックは`<系統>/common.rs`に置く」という既存ルールに合致する
- Tobit固有のバリアント（打ち切り境界の検証等）は、別のenumに分離せず**同じ`MleError`にバリアントとして追加する**（Logit/Probitの`fit()`はそのバリアントを構築しないだけで、型を分ける必要はない。OLSの`CovType::Hac`/`CovType::Cluster`がフィールド付きバリアントとして共存しているのと同じ考え方）

現時点で想定されるバリアント（`nonlinear-api-design.md`で確定済みの仕様から逆算。実装issue着手時に確定させる）:

| バリアント | 対応するPython例外 | 由来 |
|---|---|---|
| `NonConvergence { n_iter: usize }` | `ComputationError` | 3章: `raise_on_non_convergence=true`かつ`max_iter`到達時 |
| `InsufficientObservations { n: usize, k: usize }` | `ValidationError` | OLSの`InsufficientObservations`と同型 |
| `InvalidConfidenceLevel { confidence_level: f64 }` | `ValidationError` | OLSと同型 |
| `InvalidMaxIter { max_iter: i64 }` | `ValidationError` | `max_iter <= 0`等 |
| `MissingClusterColumn` | `ValidationError` | `cov_type="cluster"`なのにクラスターキー未指定。OLSと同型 |
| `InsufficientClusters { g: usize }` | `ValidationError` | OLSと同型 |
| `SingularHessian` | `ComputationError` | 収束点のHessianが特異で、観測情報行列（`cov_type="classical"`既定）の逆行列が計算できない |
| `ComputationFailed(String)` | `ComputationError` | 正規分布のCDF計算失敗等、OLSの`ComputationFailed`と同型 |
| `InvalidCensoringBounds { lower: Option<f64>, upper: Option<f64> }`（Tobit専用） | `ValidationError` | 下限≧上限等の不正な指定 |

### 標準誤差の技術仕様（`cov_type`の計算式）

`nonlinear-api-design.md`4章で確定した5種類（`classical`/`nonrobust`/`opg`/`hc0`/`hc1`/`cluster`）の具体式。`s_i`を観測`i`のスコアベクトル（対数尤度の1階微分）、`H`を収束点で評価した対数尤度のHessianとする。

- **観測情報行列**（`"classical"` / `"nonrobust"`、既定）: `Σ = -H⁻¹`
- **OPG/BHHH**（`"opg"`）: `Σ = (Σᵢ sᵢsᵢ')⁻¹`
- **サンドイッチ**（`"hc0"`）: `Σ = H⁻¹ (Σᵢ sᵢsᵢ') H⁻¹`（misspecification-robust、quasi-MLEサンドイッチ）
- **サンドイッチ+小標本補正**（`"hc1"`）: `hc0`の`Σ`に`n/(n-k)`を乗じる（OLSのHC1と同じ発想）
- **クラスター**（`"cluster"`）: `Σ = H⁻¹ (Σ_g S_g S_g') H⁻¹`、`S_g = Σ_{i∈g} sᵢ`（OLSの`cluster_cov_params`と同型の導出）

`H`・`s_i`はモデルごとに異なるが、上記5つの行列演算自体（`-H⁻¹`、外積和、サンドイッチ積）はLogit/Probit/Tobitで完全に共通のため、`discrete_choice/common.rs`に共通関数として実装する候補（各モデル側は`s_i`・`H`を渡すだけでよい設計を目指す。engine内のtrait設計自体は未確定、後述）。

**未確定**: クラスターの小標本補正（OLSは`G/(G-1) * (n-1)/(n-k)`を常に適用し無効化オプションを設けない方針）をMLEでも同じ規約にするかは、この会話でまだ決めていない。実装issue着手時に確認する。

### 検定分布: 標準正規分布

- `nonlinear-api-design.md`5章で確定した「z検定」の実装詳細。MLEの漸近理論`θ̂ ~ N(θ, Σ)`に基づき、`z = θ̂ⱼ / se(θ̂ⱼ)`、p値は標準正規分布の両側確率、信頼区間は標準正規分布の臨界値（例: 95%信頼区間なら`z_crit ≈ 1.959964`）を使う
- OLSの`StudentsT`（statrs）に対応する形で、**`statrs::distribution::Normal`**を使う

### 収束判定の実装フロー

1. ソルバー（`method`＝`newton`/`bfgs`/`lbfgs`）を`max_iter`を上限に実行し、収束フラグ・反復回数を得る
2. `raise_on_non_convergence=true`（既定）かつ未収束 → `MleError::NonConvergence { n_iter }`を返す（`fit()`はここで打ち切り、パラメータ等は返さない）
3. それ以外（収束した、または`raise_on_non_convergence=false`） → 結果を返す。`converged`/`n_iter`フィールドに実際の値を格納する

### engineのモジュール構成

`.claude/rules/rust-style.md`の既存規約（系統＝ディレクトリ、手法＝最初は1ファイル）通り、`engine/src/discrete_choice/{common.rs, logit.rs, probit.rs, tobit.rs}`とする。`common.rs`には`MleError`（上記）と、`cov_type`の共通行列演算（観測情報行列/OPG/サンドイッチ/クラスター）を置く候補。`engine_pybind`側も同じ系統名で対応させる（`engine_pybind/src/discrete_choice/{logit,probit,tobit}.rs`）。

## 未確定（実装issue着手時、または追加相談が必要）

- **engine内のMLE共通のtrait設計**: argminの`CostFunction`/`Gradient`/`Hessian`トレイトをどう実装するか（モデルごとに直接実装するか、`common.rs`に薄い共通ラッパーを噛ませるか）、観測ごとのスコアベクトルをどう保持・再利用するか（`cov_type`計算で必要）。`nonlinear-api-design.md`8章のmaxLikアーキテクチャを参考に、Logit/Probit着手時に確定させる
- **収束判定の具体的な`tol`**: 勾配ノルム等、収束とみなす閾値のデフォルト値
- **クラスター標準誤差の小標本補正**: MLEでもOLSと同じ規約（常に適用）にするか
- **Tobit固有エラーの正確な検証条件**: 下限<上限の検証、両側打ち切り時の整合性チェック等の詳細
- **モデルごとの尤度・勾配・Hessian導出**（数式そのもの）: Logit/Probit/Tobitそれぞれ着手時に別ノート（`logit-implementation-notes.md`等、または本ファイルへの追記）を作成
