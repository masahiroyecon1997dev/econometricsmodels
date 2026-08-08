---
name: reference-benchmark
description: statsmodels（主）・Rパッケージ（クロスチェック）・pyfixest（FE系）と、合成データセット/Wooldridgeデータセットを使って推定手法のベンチマーク値を生成し、tests/api_tests/fixtures/benchmarks/にJSONとして固定する。新しい推定手法のテスト作成（/test-new）の一部として使用する。
argument-hint: "[手法名]"
allowed-tools: Read, Write, Bash(python3:*), Bash(Rscript:*), Bash(pytest:*)
---

# リファレンスベンチマーク生成

対象の推定手法について、リファレンス実装でベンチマーク値を生成し、`tests/api_tests/fixtures/benchmarks/`にJSONとして固定する（他の目的のフィクスチャと混在させないためのサブディレクトリ）。詳細な方針は `.claude/rules/testing-policy.md` を参照。

このスキル自体はスクリプトを持たない。実際のコードは `benchmark/` ディレクトリ（リポジトリ直下、`tests/`とは別）に実プロジェクトコードとして置く。理由: これらのスクリプトはテストの実行コードではなく、テストが使うベンチマーク値を生成するツールであり、Rなど別ランタイムに依存するため`tests/`とはライフサイクルが異なる。`.claude/skills/`側に複製すると二重管理になるため、単一ソースとして`benchmark/`のみに置く。

## リファレンス実装の役割分担

- **statsmodels**: 主リファレンス。classical/HC0-3/cluster/HAC、AIC/BIC/log-likelihood、ロバストWald検定まで一貫して対応
- **R（lm + sandwich/lmtest）**: 独立実装によるクロスチェック。新しい統計量・cov_type追加時はstatsmodelsとRの一致を先に確認してからフィクスチャを固定する。対象は係数・標準誤差に限らない。R²・AIC・BIC・対数尤度・F統計量・F検定p値等、公開する統計量は全てcrosscheckする（`testing-policy.md`「リファレンス実装」参照）
- **pyfixest**: OLSの正確性検証には使わない（HC2/HC3にpyfixest自身の実装バグによる系統的乖離があるため、詳細は`testing-policy.md`「リファレンス実装」参照）。性能比較専用。固定効果が絡むPhase4（FE/RE）以降での採否はその時点で個別に判断する
- **statsmodels discrete model（Logit等）固有の既知の欠落**（Issue #68で発覚、Probit実装時も要確認）: `cov_type="hc1"`はOLS/GLMと異なり小標本補正が未実装でHC0と同一値になる（Rが正リファレンス）。`cov_type="opg"`はネイティブ非対応（"cov_type not recognized"）で`model.score_obs(params)`から手計算が必要、かつ`opg`の限界効果（`get_margeff()`）はfit済み結果へのcov_params事後上書きが効かないためstatsmodels側では算出不可（R `marginaleffects`の`vcov=`引数を使う）。詳細は`docs/planning/specs/logit-implementation-notes.md`「`cov_type="hc1"`はstatsmodelsのdiscrete modelで未実装と判明」参照

## `benchmark/` ディレクトリの構成（動作確認済み）

`engine`/`engine_pybind`と同じ系統（family）単位でディレクトリを分けている（`linear`=OLS/WLS/GLS、`panel`=FE/RE、`iv`=IV、`nonlinear`=Logit（Issue #68で着手済み）/Probit（Issue #84で着手済み）/Tobit等）。系統をまたいで使う汎用ツールのみ`benchmark/`直下に置く。

- `benchmark/generate_synthetic_datasets.py`（系統非依存、root）: 合成データセット生成。`SCENARIOS`に7種類のバリエーション（baseline, small_n, high_variance, heteroskedastic, autocorrelated, moderate_multicollinearity, perfect_multicollinearity）を実装済み。
- `benchmark/nonlinear/generate_binary_choice_datasets.py`: 2値選択モデル（Logit/Probit）専用の合成データ生成。`generate_binary_choice_dataset(scenario, link="logit"|"probit", ...)`（元は`generate_logit_datasets.py`としてLogit専用だったが、Probit追加時に`link`引数で一般化した。シナリオ・X生成ロジックはOLSの`generate_synthetic_datasets.py`と別設計、`generate_logit_dataset`/`generate_probit_dataset`という名前付きエイリアスも提供）。
- `benchmark/load_wooldridge.py`（系統非依存、root）: Wooldridgeデータセットをpolars DataFrameとして読み込む（`pip install wooldridge`が必要）。
- `benchmark/freeze_datasets.py`（系統非依存、root）: 上記を使って生成した入力データを`tests/api_tests/fixtures/benchmarks/data/`にCSVとして**固定**するスクリプト。フィクスチャ生成・pytest実行時は、このCSVを読むだけでよい（ジェネレータを直接呼ばない）。理由: ジェネレータ側のコードが将来変わっても、既に固定したフィクスチャJSONの期待値と無言で不整合にならないようにするため。新しいシナリオ・データセットを追加した場合のみ、このスクリプトを再実行してCSVを更新する（フィクスチャJSON同様、自動追従はしない）。
- `benchmark/<系統>/run_statsmodels_benchmark.py`: 主リファレンス。1回呼べば1ケース分の結果を返す汎用ツール。`linear`系統では`--weight-col`指定でWLS（`smf.wls`）にも対応、`nonlinear`系統では`--model logit`/`--model probit`でLogit/Probitを切り替える（Issue #84で一般化）。
- `benchmark/<系統>/fixtures/generate_<手法名>_fixtures.py`: 対象手法の全シナリオ×全オプションを回し、`tests/api_tests/fixtures/benchmarks/<手法名>.json`へ書き出す専用スクリプト。生成スクリプト（`benchmark/`側）と生成物（`tests/`側）を分けている（`testing-policy.md`「ベンチマーク値のフィクスチャ化」参照）。`linear/fixtures/generate_ols_fixtures.py`（statsmodels主リファレンス）・`generate_ols_crosscheck_fixtures.py`（Rクロスチェック）が実装例。フィクスチャ生成スクリプト自体は手法ごとに分ける（`run_statsmodels_benchmark.py`のような汎用ツールとは異なる粒度、`generate_logit_fixtures.py`/`generate_probit_fixtures.py`のように別ファイル）。
- `benchmark/linear/run_pyfixest_benchmark.py`: OLS/WLS（`--weights`指定）で動作確認済み。Phase4以降で主に使用。OLSでは`vcov`引数でHC1-3/cluster(`{"CRV1": col}`)/HAC(`"NW"`)も指定可能だが、正式なクロスチェックはRを使う方針（下記参照）。
- Rスクリプトはパッケージ単位で1ファイルに分けている（旧`run_r_benchmark.R`の単一ディスパッチャから分割）:
  - `benchmark/linear/run_lm_crosscheck_benchmark.R`: base R `lm` + sandwich/lmtestによるOLS/WLS標準誤差クロスチェック（classical/HC0-3/cluster/HAC対応、`weights`引数でWLSにも対応）。動作検証済み、正式なクロスチェックとして使用中。
  - `benchmark/linear/run_fixest_benchmark.R`: fixestパッケージ。未検証（現状どのフィクスチャ生成スクリプトからも呼ばれていない）。
  - `benchmark/panel/run_plm_benchmark.R`: plmパッケージ。未検証（Phase4着手時に確認）。
  - `benchmark/iv/run_ivreg_benchmark.R`: ivreg/AERパッケージ。未検証（Phase3着手時に確認）。
  - `benchmark/nonlinear/run_glm_crosscheck_benchmark.R`: base R `glm` + sandwich（HC0/HC1/cluster）+ 手計算OPG（`sandwich::estfun()`のスコア寄与から`Σ=(Σsᵢsᵢ')⁻¹`）+ `marginaleffects`パッケージ（限界効果、`vcov=`引数でカスタム共分散行列を直接渡す）によるLogit/Probitクロスチェック。第4引数`link`（`logit`/`probit`）で切り替える（Issue #84で一般化）。動作検証済み、正式なクロスチェックとして使用中（Logit: Issue #68、Probit: Issue #84）。**注意（重要、Probit追加時に発覚）**: `classical`/`hc0`/`hc1`/`cluster`は`glm()`の既定`vcov()`/`vcovHC()`/`vcovCL()`（IRLS/Fisher scoringの期待情報行列ベース）をそのまま使わず、本実装と同じ解析式（`λᵢ(λᵢ+zᵢ)`等）で観測情報行列を手計算したものを`sandwich(bread.=...)`に渡す。Logit（binomial族の正準リンク）は期待情報行列と観測情報行列が理論上一致するため影響が無いが、Probit（非正準リンク）は一致せず、素の`vcov()`を使うと最大約8%の乖離が生じることが実測で発覚した（詳細は`docs/planning/specs/probit-implementation-notes.md`参照）。**注意**: `marginaleffects::datagrid()`/`slopes(newdata="mean"|"median")`のショートカット文字列は、整数のみの数値列を`FUN_integer`（既定`round(mean(x))`）で丸めてしまい、本実装・statsmodelsの「生の標本平均・中央値」の定義とずれる。`datagrid(FUN_numeric=mean, FUN_integer=mean)`のように両方明示すること（`docs/planning/specs/logit-implementation-notes.md`「R側の限界効果リファレンス」参照）。
  - 注意: `read.csv()`はデフォルトで列名を`make.names()`により書き換える（例: `_group`→`X_group`）。クラスター列等を渡す場合は影響を受けるため、各スクリプトとも`check.names = FALSE`を指定している。
- **pyfixestのHC2/HC3に関する既知の差異**: pyfixest自身の実装バグ（HC1用の小標本補正をHC2/HC3にも誤って適用）による系統的乖離があり、OLSの正確性検証からは除外し性能比較専用とする。詳細は`testing-policy.md`「リファレンス実装」・`docs/spec/ols-spec.md`「テスト」参照。

## 手順

1. 対象手法が新しい合成シナリオを必要とする場合は、`benchmark/generate_synthetic_datasets.py`を更新した上で`benchmark/freeze_datasets.py`を再実行し、`tests/api_tests/fixtures/benchmarks/data/`のCSVを更新する。既存シナリオを使う場合はこの手順は不要（既に固定済みのCSVを読むだけでよい）。Wooldridgeデータセットはこの固定化の対象外（下記「既知の未確定事項」参照）で、`load_wooldridge.py`経由で都度ロードする。
2. `$ARGUMENTS`（手法名）に応じて、対象の全cov_type/オプションの組み合わせで`benchmark/<系統>/run_statsmodels_benchmark.py`を実行する（固定済みCSVを読む）。
3. 新しい統計量・cov_typeを初めて使う場合は、Rの`lm`+`sandwich`/`lmtest`（`benchmark/<系統>/run_lm_crosscheck_benchmark.R`）でも同じ組み合わせを計算し、statsmodelsと一致することを確認する。一致しない場合は既定値（自由度補正等）の違いを疑って調査する。fixest/pyfixestは実装系統がfixestと同一のため、独立実装によるクロスチェックとしては使わない（補助的な確認に留める）。
   - AIC/BICはR標準の`AIC()`/`BIC()`関数をそのまま使わない。残差分散を1パラメータとして追加でカウントする慣習（k+1）のため、本実装・statsmodels（回帰係数の数kのみ使用）とはAICがちょうど2、BICが`log(n)`だけ系統的にずれる。本実装と同じ式（`-2*loglik + 2*k`等）で手計算した値と比較する（`run_lm_crosscheck_benchmark.R`が実装例）。
4. `benchmark/generate_synthetic_datasets.py`の7シナリオ（固定済みCSV経由）で対象手法を実行する。ただし完全な多重共線性シナリオは数値比較の対象外（想定エラーの発生確認のみ、`testing-policy.md`「テストの3系統」参照）。境界値・悪条件（`n=k+1`、極端なスケール差、高条件数）やクラスター系の不均衡・境界値ケースも対象手法に応じて検討する（`testing-policy.md`「テスト用データセット」参照）。
5. Wooldridge等の実データセットでも同様に確認する（`load_wooldridge.py`の`SUGGESTED_DATASETS`は候補であり、実際に使うデータセットは手法実装時に個別に確認する）。
6. 生成した結果を`tests/api_tests/fixtures/benchmarks/`にJSONとして保存する（`_meta`フィールドにリファレンス実装・バージョン・生成コマンドが含まれることを確認する）。
7. テストコード自体の作成・実行は `/test-new` `/test-run` に引き継ぐ。このスキルはベンチマーク値の生成・フィクスチャ化に留める。pytest側は合成データについては`tests/api_tests/fixtures/benchmarks/data/`の固定CSVを直接読む（ジェネレータを呼ばない）ため`wooldridge`パッケージは不要だが、Wooldridge実データのクロスチェックテストは`pytest.importorskip("wooldridge")`で任意扱いにする（下記参照）。

## 既知の未確定事項

- `load_wooldridge.py`の`SUGGESTED_DATASETS`は候補に過ぎない。手法ごとに実際に使うデータセットは個別に相談して確定する。
- **Wooldridgeデータセットは`freeze_datasets.py`での固定化対象外**（`wooldridge`パッケージ自体はMITライセンスだが、同梱データの著作権が原典の教科書側にある可能性があり、フィルタ後の部分集合であってもリポジトリにCSVとして再配布してよいか未確認のため。ユーザー確認済み）。合成データセットとは異なり、Wooldridgeデータは`load_wooldridge.py`経由で都度ロードし続ける。ライセンスが明確になれば固定化を再検討する。
- `run_plm_benchmark.R`/`run_ivreg_benchmark.R`は引き続き未検証。特に`plm`のindex指定（individual/time列）は手法・データセットごとに調整が必要。
- フィクスチャJSONの正式なディレクトリ構成・命名規則は、実際に`tests/api_tests/`を作る際に確定する。
