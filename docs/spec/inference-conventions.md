# 検定分布・診断統計量の運用ノート（内部用）

特定の推定手法に限定しない、検定分布・診断統計量の選択に関する手法横断の設計記録。**利用者向けの一覧表（手法別の検定分布・自由度、R/statsmodels/linearmodelsの既定値との違い、診断統計量の読み方）は公開ドキュメント`docs/guide/inference-conventions.md`（mkdocsのnavに掲載）を正本とし、ここには重複させない**。本ファイルは、なぜそう決めたか・ベンチマーク照合上の注意・不採用にした案など、公開ページに載せない内部事情を扱う。各手法の詳細な数式は個別のspec（`ols-spec.md`・`iv-spec.md`・`nonlinear-common.md`・`panel-common.md`等）を正本とする。

新しい推定手法を追加したときは、公開ページの表に1行追加し、検定分布の選択理由が既存と異なる場合のみここに追記する。

## 1. 検定分布の選択理由

**OLS/WLS/2SLS/FE/REがt分布を使う理由**: 古典的仮定（誤差項が正規分布に従う等）の下では、係数の標準化統計量`(β̂-β)/ŝe`が**有限標本で厳密に**t分布に従う（コクランの定理）。この結果はサンプルサイズによらず成り立つ厳密な理論であり、`cov_type`（classical/HC系/cluster/hac）によらず一貫してt分布を採用する（`ols-spec.md`3.2節、`iv-spec.md`3.2節、`panel-common.md`3.3節で同じ判断を踏襲）。`cov_type`によって分布を切り替えない理由は、標準誤差の種類を変えただけで参照分布まで暗黙に変わるのを避けるため。

**GMM/Logit/Probit/Tobitがz分布を使う理由**: GMMの理論的正当化（Hansen 1982）およびMLEの漸近理論は、いずれもサンプルサイズが無限大に近づくときの漸近正規性のみに依拠しており、OLSの`n-k`に相当する自然な自由度・有限標本での厳密な分布の閉形式が存在しない。t分布を使うことは、存在しない有限標本の理論的裏付けを偽って主張することになるため、素直に漸近論が保証するz分布・カイ二乗分布を採用する（`iv-spec.md`3.2節、`nonlinear-common.md`4章）。Tobitも同じMLEベースのため同様に標準正規分布・Wald χ²（`tobit-spec.md`）。

**クラスター時の自由度**: OLS/WLS/2SLSの`cov_type="cluster"`はp値・信頼区間の自由度を`G-1`にする（`ols-spec.md`3.2節）。FE/REも`cov_type="cluster"`では`G-1`（`G`は`cluster`列のクラスター数。既定のentityでなくても同様）、`cov_type="dk"`では`t_periods-1`、それ以外は`df_resid`（`fe-spec.md`3.2節、`re-spec.md`3.3節）。`fixest::feols(cluster=)`の`ssc()`既定`t.df="min"`と一致する。FE/REのF統計量も同じ`cov_type`別の共分散・自由度のWald検定のため、`f_df_denom`は`stat_df`と一致する（`fe-spec.md`3.2節、`re-spec.md`3.5節）。

**engine側の型**: `engine::inference::StatDist::{T { df }, Normal}`で「t分布なら自由度がある、正規分布ならない」関係を型で保証する。Python側の`stat_dist`/`stat_df`はその写像。`marginal_effects()`は常に正規分布で、返り値が`list[dict]`のため行ごとの`stat_dist`は持たない（docstringに明記）。

**`stat_df`と`df_resid`の関係**: `stat_df`は実際にp値・信頼区間に使った自由度で、`df_resid`と一致するとは限らない（上記のcluster）。統計量がNaN/`None`のときは自由度も`None`。名前から分布が一意に決まらない`wald_*`（IV・Tobit）だけ`*_dist`を持つ。

## 2. ベンチマーク照合上の注意（他パッケージの既定値との違いの内部側）

利用者向けの違いは公開ページを参照。ここでは照合の実務上の扱いのみ記す。

- **statsmodels**: `cov_type`が`"nonrobust"`以外だと既定で`use_t=False`になる。ベンチマーク照合時は`use_t=True`を明示指定する（`ols-spec.md`3.2節）。
- **linearmodels（IV）**: `debiased`で分布が切り替わる（`debiased=False`が既定でz/χ²）。2SLSのベンチマーク生成では`coef`/`se`のみ`linearmodels`から借り、`test_stats`/`p_values`/`conf_int`/`f_statistic`は自前でt/F分布を使って計算し直している（`benchmark/iv/references/linearmodels_ref.py`のモジュールdocstringが正本）。
- **R**: `benchmark/*/references/*.R`が各パッケージの出力をJSON化する。公開ページに書いたR各パッケージの分布は、公開ページ作成時にスクリプトで実測して確認した（sandwich 3.1.3・lmtest 0.9.40・fixest 0.14.2・plm 2.6.7・ivreg 0.6.8・AER 1.2.17、R 4.5.3）。パッケージのメジャー更新時は再確認する。特に`plm`は`summary()`の列名が`t-value`でもp値は正規分布から計算される（`within`/`random`とも実測で確認）ため、列名から分布を推測しない。

## 3. Stock-Yogoの弱操作変数F統計量（v1スコープの判断）

- v1スコープでは内生変数ごとの生の部分F統計量のみを返す。Stock-Yogoの臨界値テーブルは経験的なシミュレーション値でクローズドフォームでなく実装コストが高いため、照合（合否判定）は実装しない（`iv-spec.md`3.4節）。
- 複数内生変数の同時弱操作変数診断（Cragg-Donald統計量）も同様の理由でv1スコープ外。複数内生変数（`k_endog>=2`）シナリオが実際にサポートされた後もこの判断を維持するかは再検討中。
- 部分F統計量が`cov_type`に依存しない理由（Stock-Yogoの臨界値表が等分散前提でキャリブレーションされている、`OlsEstimator`が`cov_params`全体を公開していない）は`engine/src/iv/CLAUDE.md`参照。

## 4. 過剰識別検定（Sargan/Hansen J）の設計判断

- Sargan検定（2SLS）が`cov_type`に依存しないのは、定義自体が等分散前提の検定であるため（`engine/src/iv/CLAUDE.md`）。Hansen J検定（GMM）は点推定に使った重み行列`S`をそのまま使うのが定義そのもので、`gmm_weight_type`に連動する。
- **頑健な過剰識別検定が必要な場合はGMM（Hansen J）を使う**、という役割分担が設計方針（実装時にユーザー確認済み）。
- **不採用にした案**: `linearmodels`には2SLSの枠組みのままでも頑健な過剰識別検定（`wooldridge_overid`、スコア検定形式）が存在するが採用していない。GMMへの切り替えで頑健な検定が可能なため、2SLS側への追加実装は見送った。

## 5. Hausman/Wu-Hausman検定

- RE内蔵のHausman検定は回帰ベース（補助回帰）版で、Wald検定の共分散をRE本体の`cov_type`に連動させる（既定のclusterならrobust Hausman、比較は常に1-way）。補助回帰の小標本補正は`OlsEstimator`（Stata・R型）でRE本体（linearmodels型）と混在する。`plm::phtest(method = "aux", vcov = ...)`と一致を確認済み（`re-spec.md`3.7節）。
- IVのWu-Hausman検定は`cov_type`に対応させる。ただし`hac`のみ`linearmodels`の`wooldridge_regression`と一致せず原因未特定のため`None`にする（`iv-spec.md`3.6節）。

## 6. 検定・診断の公開形（プロパティかメソッドか）

新しい検定・診断統計量を足すときの置き場所の規則（ユーザー決定済み）。既存の公開形はこの規則の帰結であり、場当たりに決めたものではない。

- **推定量そのものの妥当性・識別に属する検定は、`fit()`時に計算してプロパティで公開する。** 自由パラメータを持たず（`fit()`のオプションだけで決まり）、推定量を選ぶ・使う時点で必ず目にする検定が対象。例: IVのSargan/Hansen J・Wu-Hausman・弱操作変数F（Cragg-Donald/Kleibergen-Paapも同区分）、REのHausman、全体F/Wald/LR。
- **推定後の事後診断は、検定ごとの独立したメソッドとして公開する。** 検定のバリエーション、ラグ次数、時間順の指定、補助回帰の設計など利用者の選択が入るもの。例: White、Breusch-Godfrey、Breusch-Pagan、今後のRESET/Jarque-Bera。
- **事後診断を`fit()`で自動計算しない。** 「検定してから頑健SEを選ぶ」手順を誘発するため。頑健SEは`cov_type`で最初から選ぶ設計と整合しない。
- **検定ごとにメソッドを分ける。** 引数が検定ごとに異なり（例: BGは`time`・`nlags`、Whiteは引数なし）、検定ごとのメソッドなら型補完とバリデーションがそのまま効くため（`y`を`str`単独にするのと同じ、呼び出しやすさ優先の理由）。全検定を束ねる`diagnostics()`は**当面作らない**（一覧性が必要になった時点で薄い集約メソッドとして足す余地は残す。確定した不採用ではない）。
- **公開形の揃え方**: プロパティは`<name>_statistic`/`_p_value`/`_df`の三つ組。事後診断メソッドは、同じ意味のフィールド（`statistic`/`p_value`/`df`/`distribution`）を持つ検定共通のfrozen dataclassを返し、`to_dict()`でJSON向けの`dict`にできる。理由: `fit_result.params`のように結果は属性（ドット記法）で読むのが基本で、診断だけ添字アクセスだとAPIが不揃いになる。レポートへの埋め込み（`f"{res.p_value:.3f}"`）でも読みやすい。JSON化は`dataclasses.asdict`相当で足りる。検定固有の追加項目は継承した型に足す。型名は`DiagnosticResult`（検定共通）と、検定固有の項目を持つ継承型（例: `WhiteTestResult`、置き場所は`econometricsmodels.diagnostics`、トップレベルにも公開）。`distribution`は`stat_dist`と同じく`Literal`の文字列（`"chi2"`/`"f"`）。LMとF版のように同じ検定の版が複数ある場合は、結果を2つ返さず`statistic`引数（`Literal["lm", "f"]`）で選ばせて型を単純に保つ。一方、キーが利用者の列名で決まる結果（`params`/`std_errors`/`test_stats`/`p_values`/`conf_int`）と、表形式で`pl.DataFrame`/`json.dumps`にそのまま渡す`coef_table()`/`marginal_effects()`（`list[dict]`）は`dict`のまま、`predict()`は`list[float]`（`residuals`と同じ形）とする。
- **メソッドが`X`等を必要とするとき**: `OLSResult`は`X`を保持せず、`augment()`と同様に`training_data`と`param_names`から再抽出する。`training_data`が無い結果（IVの`first_stage()`が返す`OLSResults`）では`ValidationError`にする。
- **時間順が必要な検定（Breusch-Godfrey等）は時間列を必須引数とし、行順を時間順とみなす暗黙の既定を置かない**（`hac_time`と同じ理由、`ols-spec.md`の`hac_time`の項）。パッケージは警告を出さない方針のため、横断面データへの誤適用はこの必須引数で防ぐ。
