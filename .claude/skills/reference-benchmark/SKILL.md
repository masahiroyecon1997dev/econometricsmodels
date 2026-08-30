---
name: reference-benchmark
description: statsmodels（主）・Rパッケージ（クロスチェック）・pyfixest（FE系）と、合成データセット/Wooldridgeデータセットを使って推定手法のベンチマーク値を生成し、tests/fixtures/benchmarks/にJSONとして固定する。新しい推定手法のテスト作成（/test-new）の一部として使用する。
argument-hint: "[手法名]"
allowed-tools: Read, Write, Bash(python3:*), Bash(Rscript:*), Bash(pytest:*)
---

# リファレンスベンチマーク生成

対象の推定手法について、リファレンス実装でベンチマーク値を生成し、`tests/fixtures/benchmarks/`にJSONとして固定する（他の目的のフィクスチャと混在させないためのサブディレクトリ）。詳細な方針は `.claude/rules/testing-policy.md` を参照。

このスキル自体はスクリプトを持たない。実際のコードは `benchmark/` ディレクトリ（リポジトリ直下、`tests/`とは別）に実プロジェクトコードとして置く。理由: これらのスクリプトはテストの実行コードではなく、テストが使うベンチマーク値を生成するツールであり、Rなど別ランタイムに依存するため`tests/`とはライフサイクルが異なる。`.claude/skills/`側に複製すると二重管理になるため、単一ソースとして`benchmark/`のみに置く。

## リファレンス実装の役割分担

- **statsmodels**: 主リファレンス。classical/HC0-3/cluster/HAC、AIC/BIC/log-likelihood、ロバストWald検定まで一貫して対応
- **R（lm + sandwich/lmtest）**: 独立実装によるクロスチェック。新しい統計量・cov_type追加時はstatsmodelsとRの一致を先に確認してからフィクスチャを固定する。対象は係数・標準誤差に限らない。R²・AIC・BIC・対数尤度・F統計量・F検定p値等、公開する統計量は全てcrosscheckする（`testing-policy.md`「リファレンス実装」参照）
- **pyfixest**: OLSの正確性検証には使わない（HC2/HC3にpyfixest自身の実装バグによる系統的乖離があるため、詳細は`testing-policy.md`「リファレンス実装」参照）。性能比較専用。固定効果が絡むPhase4（FE/RE）以降での採否はその時点で個別に判断する
- **statsmodels discrete model（Logit等）固有の既知の欠落**（Probit実装時も要確認）: `cov_type="hc1"`はOLS/GLMと異なり小標本補正が未実装でHC0と同一値になる（Rが正リファレンス）。`cov_type="opg"`はネイティブ非対応（"cov_type not recognized"）で`model.score_obs(params)`から手計算が必要、かつ`opg`の限界効果（`get_margeff()`）はfit済み結果へのcov_params事後上書きが効かないためstatsmodels側では算出不可（R `marginaleffects`の`vcov=`引数を使う）。詳細は`docs/spec/logit-spec.md`「テスト」参照

## `benchmark/` ディレクトリの構成（動作確認済み）

`benchmark/` は `__init__.py` を持つ Python パッケージ（Initiative A、`docs/planning/specs/benchmark-restructure-design.md`）。import はすべてドット表記（`from benchmark.common import ...` 等）、スクリプト実行は**リポジトリルートから `python -m benchmark.<...>`**（各ディレクトリへ `cd` して `python foo.py` は不可）。

`engine`/`engine_pybind`と同じ系統（family）単位でディレクトリを分けている（`linear`=OLS/WLS/GLS、`panel`=FE/RE、`iv`=IV、`nonlinear`=Logit/Probit/Tobit等）。系統をまたいで使う汎用ヘルパーは`benchmark/common/`に集約。1手法は(a)データセット層＝`datasets.py`、(b)リファレンスアダプタ層＝`references/`、(c)フィクスチャドライバ層＝`fixtures/generate_*_fixtures.py`の3つに分ける。

- `benchmark/common/`（系統をまたぐ共通ヘルパー）: 旧 `benchmark/_common.py` / `_dgp_constants.py` / `_common.R` / `load_wooldridge.py` を関心事ごとに分割したパッケージ。
  - `datasets_io.py`: `DATA_DIR`・`BENCHMARKS_DIR`・`load_frozen_dataset`（`{prefix}_{scenario}.csv`＋`{prefix}_true_beta.json`読み込み）・`freeze_scenarios`（freezeの共通ループ）・`run_freeze_cli`（各系統`freeze.py`の`__main__`）。
  - `dgp.py`: `imbalanced_cluster_groups`（クラスターロバストSE確認用の不均衡グループ生成、全系統共通）・`hac_auto_lag`（HAC自動ラグ選択式）・`linear_predictor`・`correlated_design_matrix`・`apply_perfect_multicollinearity`・`validate_choice`・`preview_dataset`（`datasets.py`単体実行時のシナリオ1件プレビュー）。
  - `dgp_constants.py`: 3系統のDGPで共通の誤差項・スケール倍率の定数。
  - `constants.py`: `SYNTHETIC_FORMULA`・`MROZ_FORMULA`・`WEIGHT_COLUMN_NAME`。
  - `driver.py`: `run_fixture_cli`（11個の`generate_*_fixtures.py`で一字一句同じだった`__main__`＝引数パース→`build_fixtures()`→JSON書き出し）。
  - `reference/extract.py`: `extract_coef_se`（`model.params`/`model.bse`→`{"coef":..., "se":...}`）。
  - `reference/r.py`: `run_r`（`Rscript`をsubprocess起動）・`normalize_names`（R出力のパラメータ名正規化・キー順固定）。5系統の`_run_r`/`_normalize_names`コピーの共通化。
  - `load_wooldridge.py`: Wooldridgeデータセットをpolars DataFrameとして読む（`pip install wooldridge`が必要）。
  - `_common.R`: `coeftest()`からの係数・標準誤差抽出とロバストWald F検定（`linear/references/run_lm_crosscheck.R`と`iv/references/run_ivreg.R`で同一だった後処理）。各`.R`が`commandArgs()`の`--file=`から自ディレクトリを特定して`source()`する。
- `benchmark/<系統>/datasets.py`（(a) データセット層）: 合成データセットのDGP（`generate_*` 関数・`SCENARIOS`）とCSV凍結（`freeze.py`から呼ばれる）。`linear`は7シナリオ（baseline, small_n, high_variance, heteroskedastic, autocorrelated, moderate_multicollinearity, perfect_multicollinearity）、`nonlinear`は`generate_binary_choice_dataset(scenario, link="logit"|"probit", ...)`（`link`引数でLogit/Probit共用）、`iv`は識別構造（操作変数×内生変数×構造誤差）を組み込んだ専用シナリオ。
- `benchmark/<系統>/freeze.py`: 各系統の合成データセットを`tests/fixtures/benchmarks/data/`にCSVとして**固定**する。フィクスチャ生成・pytest実行時はこのCSVを読むだけ（ジェネレータを直接呼ばない）。理由: ジェネレータ側のコードが将来変わっても、固定済みフィクスチャJSONの期待値と無言で不整合にならないようにするため。
- `benchmark/regenerate_all.py`（root）: 合成データCSV＋全フィクスチャJSONの一括再生成オーケストレータ（3系統の`freeze.py`＋11個の`generate_*_fixtures.py`を`python -m`で順に実行）。`--datasets-only`（CSVのみ、Rscript不要）／`--fixtures-only`。新しいシナリオ・データセットを追加した場合のみ再実行する（自動追従はしない）。
- `performance/`（リポジトリ直下、`benchmark/`の外）: リファレンス実装との**性能比較**（正確性検証とは別軸、`testing-policy.md`「パフォーマンス比較（ベンチマーク）の方法論」参照）。`compare_performance.py`（`.github/workflows/benchmark_ols.yml`から定期実行）・`render_performance_summary.py`（結果JSON→job summary用Markdown整形）。現状はOLS専用実装（`LIBRARIES`/`COV_TYPES`/回帰式がハードコード）。pyfixest依存・pytest無関係で性質が違うため分離している（`benchmark-restructure-design.md` D5）。
- `benchmark/<系統>/references/`（(b) リファレンスアダプタ層）: 「凍結df＋spec＋cov_type→結果dict」に純化したアダプタと、それが呼ぶ`.R`本体。
  - `statsmodels_ref.py`（`linear`/`nonlinear`、statsmodelsが主リファレンスの系統）: 主リファレンス。1回呼べば1ケース分の結果を返す。`linear`は`--weight-col`指定でWLS（`smf.wls`）にも対応、`nonlinear`は`--model logit`/`--model probit`で切り替える。**ライブラリ名（`statsmodels`）と同名にすると`sys.path`経由で衝突するため`_ref`サフィックス付き**（旧`run_statsmodels_benchmark.py`同名衝突バグの再発防止、`refactoring-issue231-progress.md`）。
  - `linearmodels_ref.py`（`iv`、主リファレンスがlinearmodelsのため）: 2SLSは`run()`、GMMは`run_gmm()`（同ファイル）。
  - `r.py`（各系統）: `common/reference/r.py`の`run_r`/`normalize_names`を呼ぶ薄い系統別ラッパー（`run_lm_r`/`run_glm_r`/`run_ivreg_r`）。
- `benchmark/<系統>/fixtures/generate_<手法名>_fixtures.py`（(c) フィクスチャドライバ層）: 対象手法の全シナリオ×全オプション＋特殊ケースを回し、`tests/fixtures/benchmarks/<手法名>.json`へ書き出す。生成スクリプト（`benchmark/`側）と生成物（`tests/`側）を分けている（`testing-policy.md`「ベンチマーク値のフィクスチャ化」参照）。`__main__`は`run_fixture_cli`、`coef`/`se`抽出は`extract_coef_se`、Rクロスチェック呼び出しは`references/r.py`の薄いラッパー経由で共通化。`linear/fixtures/generate_ols_fixtures.py`（statsmodels主リファレンス）・`generate_ols_crosscheck_fixtures.py`（Rクロスチェック）が実装例。手法ごとに別ファイル（`generate_logit_fixtures.py`/`generate_probit_fixtures.py`等）。
- Rスクリプトはパッケージ単位で1ファイルに分けている（旧`run_r_benchmark.R`の単一ディスパッチャから分割）。cov_type→vcov分岐自体（lmはHC0-3・weight対応、ivregはHC0-1のみ対応等の差分がある）は共通化していない。
  - `benchmark/linear/references/run_lm_crosscheck.R`: base R `lm` + sandwich/lmtestによるOLS/WLS標準誤差クロスチェック（classical/HC0-3/cluster/HAC対応、`weights`引数でWLSにも対応）。動作検証済み、正式なクロスチェックとして使用中。
  - `benchmark/linear/references/run_lm_predict_crosscheck.R`: `predict()`のクロスチェック用。
  - `benchmark/panel/run_plm_benchmark.R`: plmパッケージ。未検証（Phase4着手時に確認）。
  - `benchmark/iv/references/run_ivreg.R`: ivreg/AERパッケージによる2SLSクロスチェック。動作検証済み、正式なクロスチェックとして使用中。
  - `benchmark/nonlinear/references/run_glm_crosscheck.R`: base R `glm` + sandwich（HC0/HC1/cluster）+ 手計算OPG（`sandwich::estfun()`のスコア寄与から`Σ=(Σsᵢsᵢ')⁻¹`）+ `marginaleffects`パッケージ（限界効果、`vcov=`引数でカスタム共分散行列を直接渡す）によるLogit/Probitクロスチェック。第4引数`link`（`logit`/`probit`）で切り替える。動作検証済み、正式なクロスチェックとして使用中。**注意（重要、Probit追加時に発覚）**: `classical`/`hc0`/`hc1`/`cluster`は`glm()`の既定`vcov()`/`vcovHC()`/`vcovCL()`（IRLS/Fisher scoringの期待情報行列ベース）をそのまま使わず、本実装と同じ解析式（`λᵢ(λᵢ+zᵢ)`等）で観測情報行列を手計算したものを`sandwich(bread.=...)`に渡す。Logit（binomial族の正準リンク）は期待情報行列と観測情報行列が理論上一致するため影響が無いが、Probit（非正準リンク）は一致せず、素の`vcov()`を使うと最大約8%の乖離が生じることが実測で発覚した（詳細は`docs/spec/probit-spec.md`参照）。**注意**: `marginaleffects::datagrid()`/`slopes(newdata="mean"|"median")`のショートカット文字列は、整数のみの数値列を`FUN_integer`（既定`round(mean(x))`）で丸めてしまい、本実装・statsmodelsの「生の標本平均・中央値」の定義とずれる。`datagrid(FUN_numeric=mean, FUN_integer=mean)`のように両方明示すること（`docs/spec/logit-spec.md`「テスト」参照）。
  - 注意: `read.csv()`はデフォルトで列名を`make.names()`により書き換える（例: `_group`→`X_group`）。クラスター列等を渡す場合は影響を受けるため、各スクリプトとも`check.names = FALSE`を指定している。
- **pyfixestのHC2/HC3に関する既知の差異**: pyfixest自身の実装バグ（HC1用の小標本補正をHC2/HC3にも誤って適用）による系統的乖離があり、OLSの正確性検証からは除外し性能比較専用とする。詳細は`testing-policy.md`「リファレンス実装」・`docs/spec/ols-spec.md`「テスト」参照。

## 手順

1. 対象手法が新しい合成シナリオを必要とする場合は、対象系統の`benchmark/<系統>/datasets.py`を更新した上で`python -m benchmark.<系統>.freeze`（または全系統まとめて`python -m benchmark.regenerate_all --datasets-only`）を再実行し、`tests/fixtures/benchmarks/data/`のCSVを更新する。既存シナリオを使う場合はこの手順は不要（既に固定済みのCSVを読むだけでよい）。Wooldridgeデータセットはこの固定化の対象外（下記「既知の未確定事項」参照）で、`load_wooldridge.py`経由で都度ロードする。
2. `$ARGUMENTS`（手法名）に応じて、対象の全cov_type/オプションの組み合わせで`benchmark/<系統>/references/statsmodels_ref.py`（IVは`benchmark/iv/references/linearmodels_ref.py`）を実行する（固定済みCSVを読む）。
3. 新しい統計量・cov_typeを初めて使う場合は、Rの`lm`+`sandwich`/`lmtest`（`benchmark/linear/references/run_lm_crosscheck.R`）でも同じ組み合わせを計算し、statsmodelsと一致することを確認する。一致しない場合は既定値（自由度補正等）の違いを疑って調査する。fixest/pyfixestは実装系統がfixestと同一のため、独立実装によるクロスチェックとしては使わない（補助的な確認に留める）。
   - AIC/BICはR標準の`AIC()`/`BIC()`関数をそのまま使わない。残差分散を1パラメータとして追加でカウントする慣習（k+1）のため、本実装・statsmodels（回帰係数の数kのみ使用）とはAICがちょうど2、BICが`log(n)`だけ系統的にずれる。本実装と同じ式（`-2*loglik + 2*k`等）で手計算した値と比較する（`benchmark/linear/references/run_lm_crosscheck.R`が実装例）。
4. `benchmark/linear/datasets.py`の7シナリオ（固定済みCSV経由）で対象手法を実行する。ただし完全な多重共線性シナリオは数値比較の対象外（想定エラーの発生確認のみ、`testing-policy.md`「テストの3系統」参照）。境界値・悪条件（`n=k+1`、極端なスケール差、高条件数）やクラスター系の不均衡・境界値ケースも対象手法に応じて検討する（`testing-policy.md`「テスト用データセット」参照）。
5. Wooldridge等の実データセットでも同様に確認する（実際に使うデータセットは手法実装時に個別に検討し、選定理由・変数構成を`docs/spec/<手法名>-spec.md`「テスト」節に明記する）。
6. 生成した結果を`tests/fixtures/benchmarks/`にJSONとして保存する（`_meta`フィールドにリファレンス実装・バージョン・生成コマンドが含まれることを確認する）。
7. テストコード自体の作成・実行は `/test-new` `/test-run` に引き継ぐ。このスキルはベンチマーク値の生成・フィクスチャ化に留める。pytest側は合成データについては`tests/fixtures/benchmarks/data/`の固定CSVを直接読む（ジェネレータを呼ばない）ため`wooldridge`パッケージは不要だが、Wooldridge実データのクロスチェックテストは`pytest.importorskip("wooldridge")`で任意扱いにする（下記参照）。

## 既知の未確定事項

- **Wooldridgeデータセットは`benchmark/<系統>/freeze.py`での固定化対象外**（`wooldridge`パッケージ自体はMITライセンスだが、同梱データの著作権が原典の教科書側にある可能性があり、フィルタ後の部分集合であってもリポジトリにCSVとして再配布してよいか未確認のため。ユーザー確認済み）。合成データセットとは異なり、Wooldridgeデータは`load_wooldridge.py`経由で都度ロードし続ける。ライセンスが明確になれば固定化を再検討する。
- `benchmark/panel/run_plm_benchmark.R`は引き続き未検証。特に`plm`のindex指定（individual/time列）は手法・データセットごとに調整が必要。
- フィクスチャJSONの正式なディレクトリ構成・命名規則は、実際に`tests/`を作る際に確定する。
