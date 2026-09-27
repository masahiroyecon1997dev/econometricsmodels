# 検定分布・診断統計量の運用ノート

特定の推定手法に限定しない、検定分布・診断統計量の選択に関する手法横断の記録。各手法の詳細な数式は個別のspec（`docs/spec/`配下の`ols-spec.md`・`iv-spec.md`・`nonlinear-common.md`等）を正本とし、ここではそれらの決定を一覧化し、選択の理由と他の統計ソフトウェアとの違いをまとめる。

## 1. 検定分布（t/F分布 vs z/カイ二乗分布）

| 手法 | 検定分布 | 自由度 |
|---|---|---|
| OLS | t分布（`test_stats`/`f_statistic`） | `n - k`（`cov_type="cluster"`は`G-1`） |
| WLS | t分布 | `n - k`（OLSへの委譲実装のため同じ） |
| 2SLS | t分布 | `df_resid`（`cov_type="cluster"`は`G-1`） |
| GMM | z分布・カイ二乗（`test_stats`/`f_statistic`） | なし（漸近正規性のみに依拠） |
| Logit / Probit | z分布・カイ二乗（`test_stats`、`lr_statistic`） | なし（漸近正規性のみに依拠） |
| Tobit | 未確定（正式spec未作成、実装中。MLEベースのためLogit/Probitと同じz分布を継承する見込みだが本ドキュメントでは保留） | - |

**結果への公開**: 全手法のResultsは、統計量を共通の名前`test_stats`（`coef_table()`のキーは`test_stat`）で返し、その分布を`stat_dist`（`"t"`/`"normal"`）、t分布の自由度を`stat_df`（正規分布は`None`）で示す。`stat_df`は実際にp値・信頼区間に使った自由度で、`df_resid`と一致するとは限らない（OLS/WLS/2SLSの`cov_type="cluster"`は`G-1`）。engine側は`engine::inference::StatDist::{T { df }, Normal}`で「t分布なら自由度がある、正規分布ならない」関係を型で保証する。`marginal_effects()`は常に正規分布で、返り値が`list[dict]`のため行ごとの`stat_dist`は持たない（docstringに明記）。

**検定統計量の自由度の公開**: p値を利用者が再計算できるよう、全検定統計量の自由度を公開する。F型は`f_df_num`/`f_df_denom`（OLS/WLS/FE/RE）、IV全体検定は`wald_df_num`/`wald_df_denom`（`wald_dist`が`"f"`/`"chi2"`、χ²のときは分母`None`）、Wu-Hausmanは`wu_hausman_df_num`/`wu_hausman_df_denom`、弱操作変数F統計量は`weak_instrument_f_df_num`/`weak_instrument_f_df_denom`、χ²型は`lr_df`（Logit/Probit）・`wald_df`（Tobit、`wald_dist="chi2"`）・`overid_df`（IV）・`hausman_df`（RE）。分母の自由度は`df_resid`と一致するとは限らない（OLS/WLS/2SLSの`cov_type="cluster"`は`G-1`）。統計量がNaN/`None`のときは自由度も`None`。名前から分布が一意に決まらない`wald_*`（IV・Tobit）だけ`*_dist`を持つ。

**OLS/WLS/2SLSがt分布を使う理由**: 古典的仮定（誤差項が正規分布に従う等）の下では、係数の標準化統計量`(β̂-β)/ŝe`が**有限標本で厳密に**t分布に従う（コクランの定理）。この結果はサンプルサイズによらず成り立つ厳密な理論であり、`cov_type`（classical/HC系/cluster/hac）によらず一貫してt分布を採用する（`ols-spec.md`30行目、`iv-spec.md`3.2節、`panel-common.md`3.3節で同じ判断を踏襲）。

**GMM/Logit/Probitがz分布を使う理由**: GMMの理論的正当化（Hansen 1982）およびMLEの漸近理論は、いずれもサンプルサイズが無限大に近づくときの漸近正規性のみに依拠しており、OLSの`n-k`に相当する自然な自由度・有限標本での厳密な分布の閉形式が存在しない。t分布を使うことは、存在しない有限標本の理論的裏付けを偽って主張することになるため、素直に漸近論が保証するz分布・カイ二乗分布を採用する（`iv-spec.md`3.2節、`nonlinear-common.md`4章）。statsmodels/R glmがいずれもz検定を標準とすることとも一致する。

## 2. 他の統計ソフトウェアの既定値との違い

- **statsmodels**: `cov_type`が`"nonrobust"`（classical相当）以外だと既定で`use_t=False`（正規分布）を使う。本プロジェクトは全`cov_type`でt分布に統一しているため、ベンチマーク照合時は`use_t=True`を明示指定する必要がある（`ols-spec.md`70-72行目）。
- **linearmodels**: `debiased`という別軸の引数でt/F分布とz/カイ二乗分布が切り替わる（`fit(debiased=False)`が既定でz/カイ二乗、`True`でt/F）。本プロジェクトの2SLSは`cov_type`によらず常にt/F、GMMは常にz/カイ二乗という一貫した設計のため、ベンチマーク生成時は`coef`/`se`のみ`linearmodels`から借り、検定統計量（`test_stats`/`p_values`/`conf_int`/`f_statistic`）は自前でt分布・F分布を使って計算し直している（`benchmark/iv/references/linearmodels_ref.py`参照）。
- **R**（`sandwich`/`lmtest`、`glm`）: 個別の許容誤差・既定値の違いは各手法のspec（`ols-spec.md`のクラスター小標本補正等）を参照。

## 3. Stock-Yogoの弱操作変数F統計量

- v1スコープでは、内生変数ごとの**生の部分F統計量のみ**を返す（`weak_instrument_f_statistics`）。Stock-Yogoの臨界値テーブルとの照合（弱操作変数かどうかの合否判定）は、テーブルが経験的なシミュレーション値でクローズドフォームでないため実装コストが高く、v1では実装しない（`iv-spec.md`3.4節）。目安として一般に10前後がよく引用される閾値だが、本プロジェクトはこの判定自体を提供せず、利用者側の解釈に委ねる。
- 複数内生変数の同時弱操作変数診断（Cragg-Donald統計量）も同様の理由でv1スコープ外（`iv-spec.md`3.4節）。複数内生変数（`k_endog>=2`）シナリオが実際にサポートされた後もこの判断を維持するかは再検討中。

## 4. 過剰識別検定（Sargan/Hansen J）の`cov_type`依存性

- **Sargan検定（2SLS）**: 常に古典的（等分散前提）な計算式`e'Z(Z'Z)⁻¹Z'e/σ̂²`を使い、`cov_type`には依存しない。定義自体が等分散前提の検定であるため（`engine/src/iv/CLAUDE.md`）。
- **Hansen J検定（GMM）**: 点推定に使った重み行列`S`をそのまま使うのが定義そのものであり、`gmm_weight_type`に連動する。
- **不均一分散・クラスター等に頑健な過剰識別検定が必要な場合は、2SLSではなくGMM（Hansen J）を使う**、という役割分担が設計方針（実装時にユーザー確認済み）。`linearmodels`には2SLSの枠組みのままでも頑健な過剰識別検定（`wooldridge_overid`、スコア検定形式）が存在するが、本プロジェクトでは採用していない。GMMへの切り替えで頑健な検定が可能なため、2SLS側への追加実装は現時点で見送っている。
