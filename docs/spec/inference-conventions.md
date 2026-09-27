# 検定分布・診断統計量の運用ノート（内部用）

特定の推定手法に限定しない、検定分布・診断統計量の選択に関する手法横断の設計記録。**利用者向けの一覧表（手法別の検定分布・自由度、R/statsmodels/linearmodelsの既定値との違い、診断統計量の読み方）は公開ドキュメント`docs/guide/inference-conventions.md`（mkdocsのnavに掲載）を正本とし、ここには重複させない**。本ファイルは、なぜそう決めたか・ベンチマーク照合上の注意・不採用にした案など、公開ページに載せない内部事情を扱う。各手法の詳細な数式は個別のspec（`ols-spec.md`・`iv-spec.md`・`nonlinear-common.md`・`panel-common.md`等）を正本とする。

新しい推定手法を追加したときは、公開ページの表に1行追加し、検定分布の選択理由が既存と異なる場合のみここに追記する。

## 1. 検定分布の選択理由

**OLS/WLS/2SLS/FE/REがt分布を使う理由**: 古典的仮定（誤差項が正規分布に従う等）の下では、係数の標準化統計量`(β̂-β)/ŝe`が**有限標本で厳密に**t分布に従う（コクランの定理）。この結果はサンプルサイズによらず成り立つ厳密な理論であり、`cov_type`（classical/HC系/cluster/hac）によらず一貫してt分布を採用する（`ols-spec.md`3.2節、`iv-spec.md`3.2節、`panel-common.md`3.3節で同じ判断を踏襲）。`cov_type`によって分布を切り替えない理由は、標準誤差の種類を変えただけで参照分布まで暗黙に変わるのを避けるため。

**GMM/Logit/Probit/Tobitがz分布を使う理由**: GMMの理論的正当化（Hansen 1982）およびMLEの漸近理論は、いずれもサンプルサイズが無限大に近づくときの漸近正規性のみに依拠しており、OLSの`n-k`に相当する自然な自由度・有限標本での厳密な分布の閉形式が存在しない。t分布を使うことは、存在しない有限標本の理論的裏付けを偽って主張することになるため、素直に漸近論が保証するz分布・カイ二乗分布を採用する（`iv-spec.md`3.2節、`nonlinear-common.md`4章）。Tobitも同じMLEベースのため同様に標準正規分布・Wald χ²（`tobit-spec.md`）。

**クラスター時の自由度**: OLS/WLS/2SLSの`cov_type="cluster"`はp値・信頼区間の自由度を`G-1`にする（`ols-spec.md`3.2節）。FE/REは`cov_type`によらず常にパネル調整済みの`df_resid`（`fe-spec.md`3.2節、`re-spec.md`3.3節）。このため、`fixest::feols(cluster=)`（`G-1`）とはFEのcluster時にp値が一致しない（OLSは一致する）。FE側を`G-1`に揃えるかは未決（現状は`df_resid`統一を維持）。

**engine側の型**: `engine::inference::StatDist::{T { df }, Normal}`で「t分布なら自由度がある、正規分布ならない」関係を型で保証する。Python側の`stat_dist`/`stat_df`はその写像。`marginal_effects()`は常に正規分布で、返り値が`list[dict]`のため行ごとの`stat_dist`は持たない（docstringに明記）。

**`stat_df`と`df_resid`の関係**: `stat_df`は実際にp値・信頼区間に使った自由度で、`df_resid`と一致するとは限らない（上記のcluster）。統計量がNaN/`None`のときは自由度も`None`。名前から分布が一意に決まらない`wald_*`（IV・Tobit）だけ`*_dist`を持つ。

## 2. ベンチマーク照合上の注意（他パッケージの既定値との違いの内部側）

利用者向けの違いは公開ページを参照。ここでは照合の実務上の扱いのみ記す。

- **statsmodels**: `cov_type`が`"nonrobust"`以外だと既定で`use_t=False`になる。ベンチマーク照合時は`use_t=True`を明示指定する（`ols-spec.md`3.2節）。
- **linearmodels（IV）**: `debiased`で分布が切り替わる（`debiased=False`が既定でz/χ²）。2SLSのベンチマーク生成では`coef`/`se`のみ`linearmodels`から借り、`test_stats`/`p_values`/`conf_int`/`f_statistic`は自前でt/F分布を使って計算し直している（`benchmark/iv/references/linearmodels_ref.py`のモジュールdocstringが正本）。
- **R**: `benchmark/*/references/*.R`が各パッケージの出力をJSON化する。公開ページに書いたR各パッケージの分布は、公開ページ作成時にスクリプトで実測して確認した（sandwich 3.1.3・lmtest 0.9.40・fixest 0.14.2・plm 2.6.7・ivreg 0.6.8・AER 1.2.15、R 4.5.3）。パッケージのメジャー更新時は再確認する。特に`plm`は`summary()`の列名が`t-value`でもp値は正規分布から計算される（`within`/`random`とも実測で確認）ため、列名から分布を推測しない。

## 3. Stock-Yogoの弱操作変数F統計量（v1スコープの判断）

- v1スコープでは内生変数ごとの生の部分F統計量のみを返す。Stock-Yogoの臨界値テーブルは経験的なシミュレーション値でクローズドフォームでなく実装コストが高いため、照合（合否判定）は実装しない（`iv-spec.md`3.4節）。
- 複数内生変数の同時弱操作変数診断（Cragg-Donald統計量）も同様の理由でv1スコープ外。複数内生変数（`k_endog>=2`）シナリオが実際にサポートされた後もこの判断を維持するかは再検討中。
- 部分F統計量が`cov_type`に依存しない理由（Stock-Yogoの臨界値表が等分散前提でキャリブレーションされている、`OlsEstimator`が`cov_params`全体を公開していない）は`engine/src/iv/CLAUDE.md`参照。

## 4. 過剰識別検定（Sargan/Hansen J）の設計判断

- Sargan検定（2SLS）が`cov_type`に依存しないのは、定義自体が等分散前提の検定であるため（`engine/src/iv/CLAUDE.md`）。Hansen J検定（GMM）は点推定に使った重み行列`S`をそのまま使うのが定義そのもので、`gmm_weight_type`に連動する。
- **頑健な過剰識別検定が必要な場合はGMM（Hansen J）を使う**、という役割分担が設計方針（実装時にユーザー確認済み）。
- **不採用にした案**: `linearmodels`には2SLSの枠組みのままでも頑健な過剰識別検定（`wooldridge_overid`、スコア検定形式）が存在するが採用していない。GMMへの切り替えで頑健な検定が可能なため、2SLS側への追加実装は見送った。

## 5. Hausman/Wu-Hausman検定

- RE内蔵のHausman検定は回帰ベース（補助回帰）版でclassicalのみ（`cov_type`非依存、比較は常に1-way）。補助回帰にrobust共分散を使うrobust版は将来課題（`re-spec.md`3.7節）。
- IVのWu-Hausman検定は`cov_type`に対応させる。ただし`hac`のみ`linearmodels`の`wooldridge_regression`と一致せず原因未特定のため`None`にする（`iv-spec.md`3.6節）。
