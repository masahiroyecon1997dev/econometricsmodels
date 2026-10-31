---
paths:
  - tests/**/*
  - benchmark/**/*
---

# テスト方針（tests/）

このファイルは `tests/` 配下で作業する際に自動的に読み込まれる。

## コメント運用

Issue番号・内部管理ドキュメント（`docs/planning/specs/`配下の設計ノート等）への参照可否は、CLAUDE.md 6章「コメント・ドキュメントでの参照方針」に準じる（`tests`/`benchmark`固有の追加規約はない）。

## テストの分離

- `engine`の純粋ロジックの単体テストは、対象コードと同じファイル内の`#[cfg(test)] mod tests`に置く（`cargo test -p engine`）。
- `tests/`: リファレンス実装との答え合わせテスト（`pytest`）。系統別サブディレクトリ（`tests/linear/`・`tests/nonlinear/`・`tests/iv/`・`tests/panel/`。`benchmark/` と同じ grain）に手法別ファイル `test_<手法>*.py` を置く。全手法を横断するテスト（入力の型・オプションの型・`stat_dist`/自由度の整合等）は `tests/cross_method/` に置く。テストではないヘルパー（共有の `_assertions.py`・`_helpers.py`・`_tolerances.py`・`_errors.py`・`_error_expectations.py` 等と、系統別の `_ols_helpers.py`・`_binary_choice_checks.py` 等）は `tests/support/` 一箇所に集約する。`conftest.py` は `tests/` 直下（共有フィクスチャ）、各系統ディレクトリ（系統固有のフィクスチャ）、`tests/cross_method/`（横断テスト用のパネルデータ `panel_df` 等。60行の小さなデータは `_dtype_helpers.make_frame`）に置く。どのディレクトリからでも `from _helpers import ...` 等の裸importが通るのは `pyproject.toml` の `pythonpath = [".", "tests/support"]` による。

## リファレンス実装

既存の検証済みパッケージとの数値比較で、パラメータ推定値の正しさを検証する。統計量ごとに主リファレンスを決め、独立実装によるクロスチェックも行う。

- **statsmodels**（Python）: OLS系（classical/HC0-3/cluster/HAC、AIC/BIC/log-likelihood、ロバストWald検定まで一貫して対応）の**主リファレンス**。動作確認済み（`benchmark/<系統>/references/statsmodels_ref.py`）。
- **R（`lm` + `sandwich`/`lmtest`パッケージ）**: 独立実装によるクロスチェック用。statsmodelsと計算方法の慣習が異なる可能性がある（自由度補正の既定値等）ため、**新しい統計量・cov_typeを追加するたびに、まずstatsmodelsとRが一致することを確認してからフィクスチャを固定する**。両者が一致しない場合は、既定値の違いを疑って調査する。
  - **クロスチェックの対象は係数・標準誤差に限らない**。R²・調整済みR²・AIC・BIC・対数尤度・F統計量・F検定p値等、推定結果として公開する統計量は全て独立実装でもcrosscheckする。一部の統計量だけRクロスチェックを省略し主リファレンス（statsmodels）のみに頼ると、主リファレンス自体のバグやプロジェクト固有の実装差に気づけない。
  - 慣習差の実例: RのAIC()/BIC()標準関数は残差分散を推定パラメータ1個として追加でカウントする（k+1）が、statsmodels・本実装は回帰係数の数kのみを使う。この場合はR標準関数をそのまま使わず、本実装と同じ計算式（`-2*loglik + 2*k`等）で独立に計算した値と比較すること（Rの値をそのまま使うと定数分ずれた「見せかけの不一致」になる）。
  - **クロスチェックスクリプト内で「本実装と同じ計算式」を手計算する箇所は、独立検証としての効力が薄れることに注意する**（本実装とクロスチェックスクリプトを同じ開発者が書くため、共通の誤解があれば両方に同じ形で紛れ込みうる）。参照実装のパッケージが対象の統計量をネイティブに計算しない、または慣習が異なり素の値をそのまま使えない場合（上記AIC/BIC、ロバストWald検定、非正準リンクでの観測情報行列Hessian等）に限り手計算はやむを得ないが、その場合は以下の優先順位で独立性を確保すること。
    1. **他の独立したパッケージ**（既存の主リファレンス・クロスチェック用パッケージに限らない）がその統計量をネイティブに計算していないか確認し、あれば手計算より優先してそちらを使う。
    2. 独立実装が見つからず手計算が避けられない場合は、その数式自体を**formula非依存の手段**（`numDeriv::hessian()`/`numDeriv::grad()`等の数値微分）で別途検証する（実例:
       `benchmark/nonlinear/references/run_glm_crosscheck.R`の`observed_bread()`が
       Probitの観測情報行列Hessianを本実装と同じ解析式で手計算しているが、
       `numDeriv::hessian()`による数値微分Hessianとの一致を別途確認済み）。
    3. 1・2のどちらも現実的でない場合は、その旨と理由をクロスチェックスクリプトのコメントに明記し、独立性が限定的であることを認識した上で使う。
  - 上記の対策が特に重要になるのは、**主リファレンス自体がクロスチェックと同系統（例: 両方ともR実装）で、独立した第三者実装による三角測量が効かない手法**（例: Tobitは主リファレンスがR `AER::tobit`、クロスチェックもR `censReg`。両者は別実装なのでクロスチェックとして成立するが、Python参照が無く外部（R以外）からの確認が効かない）。statsmodels等の独立した主リファレンスが別途存在する手法（OLS/WLS/Logit/Probit/IV等）では、本実装とRクロスチェックに共通の誤解があっても、独立に実装された主リファレンスとは一致しない可能性が高く、実質的な三角測量として機能する。
- **pyfixest**（Python）: OLSの正確性検証には使わない。fixest（R）本体のソースを確認したところ、pyfixest（Python、v0.60.0時点）のHC2/HC3標準誤差はfixestの設計ではなく**pyfixest自身の実装バグ**（HC1用の`N/(N-k)`小標本補正をHC2/HC3にも誤って適用）により系統的に乖離することが判明したため。性能比較（実行時間・メモリ）でのみ使う。FE/REの正確性検証にも使っていない（linearmodelsとR `fixest`/`plm`を使う。詳細は`docs/guide/verification.md`参照）。

## テスト用データセット

各推定手法は、以下の観点で網羅的にテストする。**全てのオプションの組み合わせで、リファレンス実装と統計量が一致することを確認する**。

1. **合成データセット**（`benchmark/<系統>/datasets.py`で生成し、`benchmark/<系統>/freeze.py`（または全系統まとめて`benchmark/regenerate_all.py`）で`tests/fixtures/benchmarks/data/`にCSVとして固定したものを使用。フィクスチャ生成・pytest実行時ともにこのCSVを読み、ジェネレータを直接呼ばない）。以下のバリエーションを持たせる。
   - サンプル数が少ない
   - 分散が大きい
   - 不均一分散（heteroskedasticity）
   - 自己相関（autocorrelation）: 主にHAC実装の検証用。他のcov_typeでは一般的な頑健性確認以上の意味は持たない
   - ある程度の多重共線性
   - 完全な多重共線性（設計行列が特異になるケース）: **リファレンス実装との数値比較はしない**（多くのリファレンス実装は冗長な列を自動的に落として続行するため、本プロジェクトの「エラーで止める」方針と挙動が異なる）。想定した例外（`ComputationError`）が発生することのみを確認する。
   - **境界値・悪条件（ill-conditioned）**: 自由度がちょうど1（`n = k+1`）での成功パス、変数間のスケールが極端に異なる場合（例: 一部の変数が10⁶オーダー、別の変数が10⁻³オーダー）、特異ではないが条件数が非常に大きい設計行列（相関0.99超等）。数値計算ライブラリ（faer等）依存部分の将来のリグレッションを検知するために有効。
   - **高次元（説明変数の数が多い）**: 列数依存のバグ（ループ境界・インデックス誤り等）や、`k`が大きい場合の数値的挙動を検証する。真の係数は列取り違えバグを検出しやすいよう間隔を空けて生成する。MLE系（Logit/Probit）は`k`が増えると線形予測子の分散が増え分離を誘発しやすくなるため、OLS/WLSより係数を小さく較正する必要がある点に注意（`benchmark/nonlinear/datasets.py`の`many_regressors`実装ノート参照）。
   - **説明変数の外れ値・裾の重い分布**: 少数の観測（実装例では5%）だけが極端な値を持つ設計行列（高レバレッジ行）での数値的頑健性を検証する。Tukeyの汚染混合モデル（`(1-p)*N(0,1) + p*N(0,scale²)`）等で実現する。高次元シナリオと異なり、少数の観測のみが極端な場合はMLE系でも分離を誘発しにくい（`benchmark/linear`・`benchmark/nonlinear`の`outlier_regressor`実装ノート参照）。
2. **実データセット**: Wooldridgeデータセット等の教科書的実データ。真の係数と比較できないため、リファレンス実装との一致のみで検証する。
3. **グループ/クラスター単位の分散推定（クラスターロバストSE等）を持つ手法**: 均等サイズの疑似グループだけでなく、不均衡なグループサイズ（例: `[2, 3, 5, 10, 30, 50]`のように偏らせる）、グループ数が境界値に近いケース（最小グループ数ちょうど、その外側の失敗パス）、実データでのグループ列も検証する。均等サイズの疑似グループのみのテストは、実務でバグが出やすい分布の偏りを見逃す。
   - **境界値ケースでは「クラスタ数G」と「傾き係数の数q（`k - k_constant`）」の関係に注意する**（OLSの実装過程で判明し`ValidationError`に統一した）。クラスターロバスト共分散はクラスター寄与スコアの総和がゼロ（正規方程式・MLEの一次条件）のため`rank(Ŝ) ≤ G - 1`であり、ロバストWald/F検定が使う`q×q`部分行列は`G <= q`のとき構造的に特異になる。`G`・`q`は入力だけから判定できるため、`fit()`冒頭で`ValidationError`（`InsufficientClustersForInference`）として弾く（OLS/WLS/Tobit/Logit/Probit/IV横断）。「クラスタ数境界の成功パス」を作る際は`G > q`（**厳密不等号**、`G = q`ちょうども特異）を保つ。`ValidationError`パスのテストは`G <= q`で書く。`G > q`かつ悪条件（極端なスケール差等）で`q×q`部分行列が数値的にほぼ特異になるケースは`ComputationError`がbackstop（別テスト1本）。詳細は`docs/spec/ols-spec.md`「標準誤差」のクラスター参照。

## テストの3系統

1. **成功パス**: 数値比較（フィクスチャJSONとの一致、下記参照）
2. **`ValidationError`パス**: 入力起因のエラー（欠損値、列の重複、観測数不足、`confidence_level`範囲外等）。想定した`code`と`params`で例外が発生することを`tests/support/_errors.py`の`raises_error(code, **params)`で確認する（数値比較はしない。文面は照合しない: 文面はRust側の単体テストと`tests/cross_method/test_error_catalog.py`が固定する）。引数・オプションの型違いは構造化されない`TypeError`で、文面を`_error_expectations.py`のテンプレートで照合する
3. **`ComputationError`パス**: 計算過程で発覚する問題（完全な多重共線性等）。同じく`raises_error`で`code`と`params`を確認する。カタログの`code`はすべて`MIGRATED_CODES`（実際のエラー経路から送出されることをテストで保証済み）か`UNREACHABLE_CODES`（Pythonから構成できない`code`。理由を添える）のどちらかに入れる（通常のテストが検査する）。全体実行（`pytest tests --check-error-coverage`）は、`MIGRATED_CODES`の全`code`が1度は送出されること、`UNREACHABLE_CODES`の`code`が送出されていないことを検査する

## テストファイルの構成規約

手法ごとのテストは、次の4種類のファイルに分ける（`tests/<系統>/test_<手法>_*.py`）。手法が増えても同じ構成・命名に揃える。

- **ファイル = 結果別**:
  - `_api.py`: 成功パスの構造・結果型・オプション反映。
  - `_validation.py`: `ValidationError`/`ComputationError`の境界を確認するテスト。入力・オプションを弾くテストは、拒否する別名の確認（`cov_type="nonrobust"`の拒否等）も含めてこちらに置く。境界の反対側の成功パス（分離と誤検出しない、境界ちょうどで収束する等）も、境界のテストと対にして同じ見出しの下に置く（離すと、境界のどちら側を何が守っているかが読み取れなくなるため）。
  - `_reference.py`: 主リファレンス（statsmodels/linearmodels/R `AER`等）との数値照合。
  - `_crosscheck.py`: 独立実装（R）との数値照合。
- **見出し = 機能別**（`# ── predict() ──`のようなセクション見出し）。見出しの文言と並び順は次のとおりに揃える（該当する機能がある手法のみ）。見出しの罫線は表示幅72桁にそろえる。1ファイルが大きくなっても、見出しの軸（機能別）を結果別に変えない。
  - `_api.py`: 「成功パス・結果型」「API構造」「オプションの反映」「predict()」「augment()」「pred_table()」「censoring_fit_check()」「marginal_effects()」の順。手法固有の見出し（`fixed_effects()`・ハウスマン検定・OLSとの不変条件・`hac_lags_used`等）はこれらの後ろに置く。
    - 手法固有の見出しの並びは、手法固有の機能（`fixed_effects()`・ハウスマン検定・OLSとの不変条件・IVの`GMM（estimator="gmm"）`等）→「統計的健全性チェック（クラスターSE）」→ HAC/DKの実使用値（「HACの実使用ラグ数（hac_lags_used）」「Driscoll-Kraayバンド幅の実使用値（dk_bandwidth_used）」）の順。HAC/DK/GMMのテストを「成功パス・結果型」に置かない。
    - 見出し内の並びは、手法間で共通のテストを先頭に同じ順で置き、手法固有のテストを後ろに置く（同名のテストが手法ごとに別の位置にあると、抜けに気づけないため）。「成功パス・結果型」は`test_fit_succeeds_and_returns_<手法>_results`→`test_default_options_use_*`→その他。「API構造」は`param_names`系→`params_table_structure`→`conf_int_structure`→`params_std_errors_test_stats_p_values_share_keys`→`n_obs`系→その他。「オプションの反映」は`test_cov_type_is_case_insensitive`→`test_confidence_level_changes_interval_width`→`test_include_intercept_false_*`→HACのオプション→solver等のその他のオプション。
  - `_validation.py`: 「ValidationError（入力データ）」「ValidationError（predict()のnew_data）」「ValidationError（augment()）」「ValidationError（オプション）」「ValidationError（marginal_effects()）」「ComputationError」「例外クラスの継承」の順。入力データを主題で分ける場合は「ValidationError（入力データ: 重み列）」のように`入力データ:`に続けて主題を書く。「使われないオプション」は「ValidationError（オプション）」の下に置く。
  - `_reference.py`/`_crosscheck.py`: 「凍結フィクスチャとの数値照合（合成データ）」「凍結フィクスチャとの数値照合（オプション）」「凍結フィクスチャとの数値照合（実データ）」の順（FE/REのように実データを特定する場合は「（実データ: Wooldridge wagepan）」と書いてよい）。凍結フィクスチャを使わない照合（ライブ statsmodels 等）はその後ろに「ライブ <リファレンス> との照合」として置く。見出しが1つだけのファイルにも付ける。
    - 合成データ節の並びは、ベースライン → `predict`（あれば）→ クラスター（`cluster` → `cluster_imbalanced` → `cluster_g2`/`g3` → `cluster_ill_conditioned`）→ HAC → 手法固有のシナリオ（`multi_endog`・`df1`・`weight_in_x`等）。オプション節は`no_intercept`/`include_intercept_false` → `confidence_level` → solver/method（クラスター・HAC付きの組み合わせは単独の直後）。
    - テスト名は`test_<シナリオ>_matches_<参照名>`。ベースライン（合成データの標準シナリオ）はシナリオを省き`test_matches_<参照名>`とする（`synthetic_`は付けない。FEの1-way/2-wayは`test_matches_<参照名>_one_way`/`_two_way`）。切片なしのオプションは`no_intercept`、solverの指定は`solver`（`method`にしない）で表す。
    - FE/REの「境界値」節は合成データ節の後・実データ節の前に置く。ハウスマン検定などの手法固有の主題の節は実データ節の後ろに置く。
  - `tests/cross_method/`の横断テスト: 見出しは論点別（例: 既定の`cov_type`、`cov_type`連動、不均衡パネル）。見出しの中は手法の順（linear → nonlinear → iv → panel、linearはOLS → WLS、nonlinearはLogit/Probit → Tobit、panelはFE → RE）に並べ、どの手法のどの論点が欠けているかが一覧で分かるようにする。補助関数（`_fe`・`_re`・`_iv`等）は先頭にまとめる。
- **主題別ファイル**: 上記4種類に収まらない大きな主題（White検定などの事後診断、二元配置、Mundlak、第一段階回帰、GMM等）は、`test_<手法>_<主題>.py`として別ファイルにしてよい。その中でも、数値照合と構造・エラーの区別は上のファイル分割に倣う。
- **`include_intercept=False`の置き場**: `_api.py`には構造・配線（`param_names`に`const`が出ない、`df_model`、収束）だけを置き、statsmodels/R/linearmodelsとの数値照合は凍結フィクスチャを使う`_reference.py`/`_crosscheck.py`に置く。`predict()`/`augment()`のように独立したAPI面のスモーク級照合に限り、`_api.py`でライブstatsmodels照合を行ってよい（各`_api.py`のモジュールdoc参照）。
- **文字列オプション・文字列引数の大文字小文字**: 大文字小文字を区別しない文字列のオプション（`cov_type`・`solver`・`estimator`・`gmm_weight_type`・`gmm_type`・`variance_method`）とメソッドの文字列引数（`marginal_effects`の`at`・`target`、`predict`/`augment`の`target`、診断検定の`statistic`）は、各手法が受け付ける全ての値について、`tests/support/_helpers.py`の`CASE_SPELLINGS`（小文字・全大文字・先頭のみ大文字・大小交互）×値の網羅で`test_<名前>_is_case_insensitive`（引数は`test_marginal_effects_target_is_case_insensitive`のように対象のメソッド名を含めてよい）に集約する。オプションは`_assertions.assert_option_case_insensitive`（`res.<オプション>`が正規形（小文字）で返ることと、全ての綴りで正規形と同一の推定結果になること。内部で`assert_same_estimates`を使う）、メソッド引数は`_assertions.assert_argument_case_insensitive`（戻り値が正規形のときと等しいこと）を使い、ラベルだけを見る別テストは作らない。クラスターが必要な値（`cluster`）は専用データを使う引数にして同じテストに含める。新しい文字列のオプション・引数を足したら、同じ形のテストを足す。
- **共通の命名**: 結果型は`test_fit_succeeds_and_returns_<手法>_results`、件数・被説明変数名は`test_n_obs_and_dep_var_name`（FE/REは`n_entities`も確認するため`test_n_obs_dep_var_name_and_n_entities`）。
- **solverの収束**: `_api.py`のsolver別テストは`converged`のみを確認する（最尤解との一致は`_reference.py`が厳しい許容誤差で担う）。
- **数値の比較**: `assert abs(...) < tol`や`pytest.approx`を直書きせず、`_assertions.assert_close`（`label`を渡せる）を使う。許容誤差のリテラルを直書きせず、系統別の`_tolerances_<系統>.py`（`linear`・`nonlinear`・`iv`・`panel`）に実測の最大差とマージンの根拠をコメントして置き、`_tolerances.py`が結合して公開する`TOLERANCES`から引いて`functools.partial`で束縛して使う（キーが系統間で重複するとimport時に`ValueError`）（本実装内の不変条件の比較は`<手法>_api`のキー）。
- **docstring**: 現在の挙動と理由だけを書く。「従来は〜だった」「〜に統合した」のような変更履歴、レビュー指摘・ユーザー確認の記録は書かない（経緯はgit logが持つ）。回帰ガードとして「過去にこの入力で〜が起きた」ことが、そのテストの存在理由を説明する場合に限り、理由として残してよい。シナリオの件数は書かず、`NUMERIC_SCENARIOS`のように定数名で参照する。

## engine（Rust）のカバレッジ方針

- `cargo-llvm-cov`で実測するが、**100%は目指さない**。事前に検証済みの不変条件により理論上到達不能な
  防御的エラーパス（`Result`型で返しているが、実際には浮動小数点の丸め誤差以外では失敗し得ないはず、
  という性質のもの）は、コード側のdocコメントで理由を説明した上で未カバーのまま受け入れる。
  `cargo-llvm-cov`の除外マーカーは導入しない。
- `assert!`マクロのメッセージ引数（アサーション失敗時のみ評価される）が「missed lines」に出るのは、
  分析ツールの誤検知であり実際のギャップではない。
- 理論上到達不能と判断した経路が、後から実際に到達すると判明した場合（数値的にほぼ特異な行列の検出等）
  は、テストを追加してカバーする（「理論上到達不能」というラベル自体を疑い直す）。

## ベンチマーク値のフィクスチャ化

- リファレンス実装（statsmodels/R）の実行結果は、CIで毎回再計算せず、**JSONファイルとして`tests/fixtures/benchmarks/`にコミットし固定する**。ベンチマーク由来のフィクスチャを`fixtures/`直下ではなく`benchmarks/`サブディレクトリに分けているのは、他の目的のフィクスチャ（ベンチマーク比較以外でテストが読み込みたいデータ）と混在させないため。
- 理由: CIを軽く保つため（Rパッケージのインストールはビルドが重い）。またリファレンス実装側のバージョンアップで数値が変わり、意図せずテストが壊れることを防ぐため。
- フィクスチャの命名: 手法・シナリオ・cov_type等が分かる形にする（例: `ols.json`に、シナリオ×cov_typeをネストして持たせる。実例は`tests/fixtures/benchmarks/ols.json`参照）。
  - **最上位は全フィクスチャで`synthetic`・`wooldridge`・`_meta`の3キーに統一する**。`synthetic`は合成データのシナリオと、それを元にした`solver`・`no_intercept`等の特殊ケース、`wooldridge`は実データ（`wage1`・`mroz`・`card`・`wagepan`等）。その下はシナリオ×cov_typeの順にネストする（cov_typeのエントリは、リファレンス実装の結果をそのまま置く。補助値の`hac_lag`は結果と同じ階層に置く）。統計量のキーは公開APIの名前（`params`・`std_errors`・`test_stats`・`p_values`・`conf_int`）に揃える。生成スクリプトは平坦に組み立て、末尾で`benchmark.common.group_by_dataset_source`が二層に振り分ける。
- **係数の分散共分散行列（`cov_params`）は統計量のJSONに入れず、別のJSON（`<手法>_cov.json`、クロスチェックは`<手法>_cov_crosscheck.json`）に分ける**。行列は行・列とも係数名をキーにした入れ子のdictで、統計量のJSONと同じシナリオ×cov_typeの木に同じ位置の葉として並べる。2つのJSONは同じ`build_fixtures()`から`benchmark/common/cov_fixtures.py`で分けて書くので、キーはずれない。テストは`tests/support/_fixture_cov.py`の`load_fixtures_with_cov`で合流させて読み、葉の`sqrt(diag)`が`std_errors`と一致すること・葉の過不足がないことを読み込み時に確認する。行列の要素は`|ours - ref| <= rtol * sqrt(V_ii * V_jj)`で比べる（非対角成分が0に近くても要素自身の相対誤差に依存しない。絶対誤差のフロアは置かない）。
- **フィクスチャを生成するスクリプトと、生成されたJSON自体は別の場所に置く**。生成スクリプトは`benchmark/<系統>/fixtures/generate_<手法名>_fixtures.py`（コード、`benchmark/`側で管理。系統ディレクトリ構成は`.claude/skills/reference-benchmark/SKILL.md`参照）、生成物は`tests/fixtures/benchmarks/<手法名>.json`（データ、テスト側で管理）。
- **合成データセット自体も同様にCSVとしてtests側（`tests/fixtures/benchmarks/data/`）に固定する**（`benchmark/<系統>/freeze.py`で生成）。理由: ジェネレータ側のコードが将来変わっても、既に固定したフィクスチャJSONの期待値と無言で不整合にならないようにするため。
  - **Wooldridgeデータセットはこの固定化の対象外**とする（`wooldridge`パッケージ自体はMITライセンスだが、同梱される実データの著作権は原典の教科書側にある可能性があり、フィルタ後の部分集合であってもMITライセンスの本リポジトリにCSVとして再配布してよいか未確認のため。ユーザー確認済み）。Wooldridgeデータは引き続き`load_wooldridge.py`経由で都度ロードする。`wooldridge`パッケージ自体はtest依存グループに含め標準CIで常にインストールする（実データはパッケージのwheel内に留まりリポジトリへ再配布するわけではないため、上記の再配布可否の制約とは無関係）。`pytest.importorskip("wooldridge")`は、test依存グループを経由しない実行環境向けの防御的フォールバックとして残す。
- 各フィクスチャJSONには`_meta`フィールドを含め、少なくとも以下を記録する。
  - `generated_at`: 生成日時（ISO 8601）
  - リファレンス実装のバージョン（例: `statsmodels_version`）
  - 使用したリファレンス実装の種類（`primary_reference`等）
  - （該当する場合）計算方法に関する注記
- フィクスチャは一度固定したら手動更新が前提（自動追従しない）。リファレンス実装側の既知のバグ修正等で意図的に更新する場合は、その理由をコミットメッセージに残す。
- **フィクスチャ生成に使うリファレンス実装（statsmodels等）は、`pyproject.toml`のdev dependenciesにバージョン固定して記載する**（例: `statsmodels==0.14.6`のように`==`で固定。範囲指定にしない）。理由: 生成スクリプトを再実行した際に、フィクスチャ生成時と異なるバージョンで異なる数値が出て混乱するのを防ぐため。

## 許容誤差

- **相対誤差1e-8程度（厳密）を基本方針** とする。
- 計算方法自体がリファレンス実装間で異なりうる手法（例: パネルのHausman検定等）は、以下の手順を取る。
  1. **基準パッケージを1つ決める**（本実装が採用する計算方法・慣習と一致するもの）。基準パッケージとは相対誤差1e-8の厳密比較を行う。
  2. **他のパッケージはクロスチェック用**とし、許容誤差を緩めて比較する（大きく乖離していないかの確認が目的で、厳密一致は期待しない）。
  3. **採用した計算方法と、どのパッケージが同じ計算方法かをテストコードのコメント・関連ドキュメントに明記する**（`docs/planning/specs/`配下の該当手法のノート、無ければ`docs/spec/ols-spec.md`同様の形式で新規作成）。
- **同じクロスチェック用パッケージでも、統計量・cov_typeごとに実測乖離が大きく異なる場合は、許容誤差を分けてよい**（一律に緩めると、本来検出できるはずのバグを見逃す）。分ける場合は実測値（最大相対誤差）に基づいて具体的な数値を決める。例: OLS/WLS/IVのRクロスチェック（`tests/linear/test_ols_crosscheck.py`等）はHACを含む全cov_typeで機械精度一致（実測1e-14程度）のため基本方針の1e-8を適用する。HACは当初0.4〜11%の乖離があり緩めていたが、原因がRの`NeweyWest(adjust = TRUE)`が掛ける`n/(n-k)`というスカラー1つだと分かり、R側を`adjust = FALSE`にして揃えた。**緩める前に、乖離が1つの慣習差（スカラー倍・自由度の数え方等）で説明できるか調べ、参照側の設定で揃えられるなら揃える**（説明できないまま実測値にマージンを載せない）。

## パフォーマンス比較（ベンチマーク）の方法論

- 実行時間・メモリ使用量を他ライブラリと比較する際は、**計測対象の処理範囲を対称に揃える**。比較対象ライブラリが遅延評価（lazy property）で結果を計算する設計の場合、計測コードでその結果に明示的にアクセスしないと、そのライブラリの計算コストが計測から漏れて不公平な比較になる（例: statsmodelsの`RegressionResults`は`rsquared`/`aic`/`bic`等をアクセス時に初めて計算する`cached_value`として実装している）。
- ベンチマーク作成時は、比較対象の実装がどのタイミングで何を計算するか（即時評価か遅延評価か）を事前に確認し、本実装（常に一括計算）と同じ処理範囲になるよう計測コードを書く。
- **入力形式の変換コストは計測区間の外に置く**。本パッケージの入力はpolarsだが、リファレンス実装（statsmodels/linearmodels）はpandas入力を前提とするため`polars.DataFrame.to_pandas()`が必要になる。これはリファレンス実装本体の処理ではなく変換コストなので、ウォームアップ・計測ループの外で1回だけ実施し、変換済みオブジェクトを各試行で使い回す（`performance/_perf_harness.py`の`_worker()`）。engine側はArrowゼロコピーでpolarsをそのまま渡すため変換は発生しない。
- **線形代数バックエンドのスレッド数を両者で揃える（1に固定する）**。engine（faer/rayon）とリファレンス実装（numpy/BLAS）でスレッドプールの挙動が異なり、多コア機・負荷下では競合で実行時間が単一スレッド時の数十倍に膨れ上がって計測が不安定になることを実測で確認した（OLSのclassical n=1,000,000で約0.15秒→3〜4秒）。ワーカーサブプロセスの環境変数（`RAYON_NUM_THREADS`/`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`等）で両者を1スレッドに固定し、「計算コアの効率」の比較をスレッドプール挙動の環境差から切り離す（`performance/_perf_harness.py`の`_SINGLE_THREAD_ENV`）。
- **性能比較は、既定の標準誤差でn軸・k軸を回し、標準誤差の種類（`cov_type`）はcov軸で全て測る**。正確性検証のようにcov_type×オプションの全組み合わせは網羅しないが、「最も重いcov_type」を実装時の軽い実測で手書き選定する方式は採らない（実装が変わるたびに順位が入れ替わり、選ばなかったcov_typeの劣化がどこにも現れないため）。
  - **n軸・k軸**: 既定の標準誤差（classical）だけ。n軸は10,000 / 100,000 / 1,000,000（リファレンスが大nで計測不能な手法は上限を下げ、engine単独の大標本点を足す）、k軸は5・20。
  - **cov軸**: 手法が対応する標準誤差を全て、固定のn（100,000）・k=5で測る（`PerfAdapter.cov_cases`）。clusterはクラスター数G（50 / n÷10）とクラスターの行の並び（random / sorted）の2×2で測る。パネル（FE/RE）のclusterはentity固定の1ケース。リファレンスに同等の標準誤差が無いもの（linearmodelsのHC2/HC3、statsmodels discrete modelの`opg`・`hc1`）は`CovCase.libraries`でengine単独にする。新しいcov_typeを手法に足したら、cov軸のケースも足す。
  - **method軸**: solver・estimatorの違いは代表点1つ。標準誤差と組になる変種（IVのGMMの重み行列・反復）はcov軸の`CovCase.method`/`variant`で測る。
  - 手法ごとに性能比較を1回は作成することは必須。公開ページ（`docs/guide/performance.md`）のサマリーは、既定の標準誤差の表と、cov軸の速度比マトリクス（手法×標準誤差の種類）を機械的に生成する。

## 必須事項

- 各推定手法の実装時に、対応するリファレンス実装との比較テストを必ずセットで用意する（実装だけでマージしない）。
- ベンチマーク値を生成した際は、使用したデータセット・コード・リファレンス実装のバージョン情報を記録し、再現可能な形で残す。

## property-basedテスト（`proptest`）

`engine`クレートに`dev-dependencies`として導入済み（OLSで先行導入し、現在はWLS・Logit・Probit・Tobit・IV・パネル（FE/RE）の各系統にも展開している）。固定seed・固定シナリオでの値の一致確認（`#[cfg(test)] mod tests`の通常の単体テスト）だけでは「そのseedでは偶然問題が顕在化しないケース」を拾えない構造的な穴があるため、各手法が満たすべき不変条件を個別の値比較より高い網羅性で検証する目的で使う。

- 対象ファイルの`#[cfg(test)] mod tests`内に`mod proptests`としてネストする（`engine/src/linear/ols/estimator.rs`参照）。通常の固定値単体テストと同じファイル・同じ`mod tests`に置くことで、リファクタリング時の追従漏れを防ぐ既存方針（`rust-style.md`「テスト」）を踏襲する。
- ランダムに生成する設計行列は、`SingularMatrix`にならない範囲（フルランク保証）に制約する。実務上は「独立な連続一様分布からのサンプリング + `n`に十分なマージンを持たせる（`n >> k`）」で事実上フルランクになるため、追加で`prop_assume!(result.is_ok())`を安全弁として使えば十分（`ols.rs`の`ols_case_strategy`参照）。
- 許容誤差は固定フィクスチャ比較（`RTOL=1e-8`）より緩める（乱数生成でデータのスケールが揃わないため、相対誤差ベース+絶対誤差フロアで`RTOL=1e-6`程度を目安にする）。
- **プロパティの有効性は、既存のテストでは検出できない意図的なバグをコードに一時的に注入し、プロパティが実際に失敗することを確認してから元に戻す**、という形で検証する（全4プロパティについて実施・記録済み。「プロパティを書いただけで一度も落ちたことがない」状態は、プロパティ自体が無意味な可能性を否定できないため）。
- 失敗時に生成される`proptest-regressions/`配下の回帰シードファイルは、実際にプロパティが検出した本物の失敗ケースのみコミットする（バグ注入によるテスト目的の失敗等、意図的に発生させた失敗のシードはコミット前に削除する）。
