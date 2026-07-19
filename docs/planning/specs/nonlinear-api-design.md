# 非線形モデル（MLEベース）設計検討メモ

対象: Logit / Probit / Tobit / 多項ロジット・プロビット / 順序ロジット・プロビット
関連: CLAUDE.md 2章（非交渉事項）、4章（実装フェーズ）、13章（今後の検討事項）

---

## 1. 実装対象・着手順序

- **優先順位**: Logit / Probit（最優先）→ Tobit → 多項・順序モデル（難易度が易しいものから）
- 多項・順序モデルはCLAUDE.md 4章のPhase分類に明示的に含まれていないため、フェーズ再整理が必要になる可能性がある（未対応）

---

## 2. 数値最適化ライブラリ

### 決定: `argmin`

- **ライセンス**: Apache-2.0 / MIT のデュアルライセンス（MIT採用で問題なし）
- `argmin-math`経由で**faerバックエンドに公式対応**しており、CLAUDE.mdの「faer採用・システムBLAS/LAPACK非依存」方針と整合する
- `CostFunction` / `Gradient` / `Hessian`トレイトでモデルごとに尤度・勾配・Hessianを実装する方式

### 比較検討した候補と却下理由

| 候補 | 却下・留保理由 |
|---|---|
| `ipopt-rs` | システムのIpopt本体（+ BLAS/LAPACK系）に依存。マルチOS wheel配布（`cd_release.yml`）とfaer方針の両方に反する |
| `cobyla` | 微分不要（COBYLA法）で収束が遅く精度も劣る。補助的選択肢に留める |
| `basin` | argmin類似・faer統合済みの新興クレートだが、実績・エコシステム成熟度でargminに劣るため見送り |

### Tobitの打ち切り構造について

境界制約付き最適化（ipopt-rsが得意とする領域）が必要に見えるが、Tobitの打ち切りは標準的には**尤度関数自体で表現**でき、制約なし最適化（argminのL-BFGS/Newton-CG等）で対応可能と判断。

---

## 3. 共通API設計

- OLSと同様、**説明変数・被説明変数はList渡し**
- モデル固有オプションは**オブジェクト（構造体）渡し**
  - Tobit: 打ち切り方向（左/右/両側）・下限/上限値
  - 多項モデル: 参照カテゴリの指定
  - 順序モデル: 閾値パラメータ数

---

## 4. 他パッケージの実装調査（アーキテクチャ・数値比較の参考用）

### ソルバー

| パッケージ/モデル | デフォルトソルバー |
|---|---|
| statsmodels Logit/Probit | Newton-Raphson（`method='newton'`）。他にbfgs/lbfgs/nm/cg等も選択可 |
| statsmodels GLM | IRLS（Fisher scoring）。canonical linkではNewton-Raphsonと数学的に一致 |
| R `glm()` | IRLS / Fisher scoring（発散しそうな場合はstep-halving） |
| R `MASS::polr`（順序） | `optim()`のBFGS（多次元）。1次元はNelder-Mead |
| R `survival::survreg`（`AER::tobit`の内部エンジン） | Newton-Raphson系の独自最適化 |

### 初期値

- statsmodels: デフォルトはゼロベクトル（`start_params=None`）。一部モデルはOLS結果を初期値に使う例もある
- R `glm`: family-specific（経験比率ベース等）から自動計算
- R `polr`: 初期値自動探索が失敗しやすい既知の問題があり、ユーザーに`start`引数指定を促す設計

### 標準誤差

- statsmodels: デフォルトで最適化後に**負の逆Hessian（観測情報行列）**を計算。ロバスト（サンドイッチ型）は`cov_type`引数で選択可
- R `polr`: 観測情報行列を数値近似で計算
- R `glm`: **期待情報行列**（Fisher information、IRLSの副産物）ベース
- canonical linkのLogit/Probitでは観測情報=期待情報が理論上一致するが、Tobit・順序モデルでは差が出る可能性あり

### アーキテクチャの参考: R `maxLik`パッケージ

- モデル別実装ではなく**汎用MLEエンジンとして設計**されたパッケージ。`censReg`（Tobit）、`mlogit`系（多項ロジット）等がこれをエンジン層として利用しており、「共通engine + モデル固有のloglike/gradient/hessian」という今回作りたい構造そのものに近い
- **2層構造**: 最適化層（NR/BHHH/BFGS/NM/SANNを統一インターフェースで切替）＋ MLE専用の便利層（標準誤差抽出、最終Hessian計算方法の選択）
- **ソルバー選択ロジック**: 解析的Hessianあり→Newton-Raphson、勾配のみ→BHHH/BFGS、どちらもなし→NM/SANN
- **BHHH法**: 情報行列の等価性を利用し、各観測のスコア（尤度勾配）の外積和をHessian近似として使う。解析的Hessian導出が不要になる
- **`finalHessian`パターン**: 「最適化に使ったソルバー」と「標準誤差算出に使う情報行列の種類」を分離できる設計

---

## 5. engine共通設計への示唆（暫定方針）

1. **ソルバー**: デフォルトはNewton-CGまたはL-BFGS。解析的Hessianが書きやすいモデル（Logit/Probit）はNewton系、複雑なモデルはL-BFGSという使い分けも検討
2. **収束判定基準**: モデル横断で共通化（勾配ノルムベースを軸に検討）
3. **初期値**: オブジェクト経由で`Option<Vec<f64>>`としてユーザー指定可能にし、`None`ならゼロベクトルをデフォルトにする（statsmodels方式）
4. **標準誤差**: 観測情報行列（Hessianの逆行列）を基本とする
5. **将来のBHHH/サンドイッチ型SE対応を見据え**、argminの`Gradient`トレイト実装で**観測ごとのスコアベクトル**を返せる設計にしておくことを検討（合計勾配だけでなく個々のスコアも扱えるようにする）

---

## 6. 未決定・次に詰めるべき論点

- [ ] engine内のMLE共通の型・構造（`CostFunction`/`Gradient`/`Hessian`トレイトの実装パターン、観測ごとスコアの扱いを含む）の具体設計
- [ ] 収束判定基準の具体的なデフォルト値
- [ ] 標準誤差の技術仕様の最終確定（観測情報行列 vs 期待情報行列 vs サンドイッチ型の対応範囲）
- [ ] モデル固有仕様（Logit/Probit → Tobit → 多項/順序の順）
- [ ] リファレンス実装・テスト比較ライブラリの最終選定
  - Logit/Probit: statsmodels、R `glm`
  - Tobit: R `AER::tobit`（`survival::survreg`ベース）
  - 多項ロジット: R `nnet::multinom`、statsmodels `MNLogit`
  - 順序ロジット/プロビット: R `MASS::polr`