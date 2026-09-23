# テストシナリオ網羅性 候補メモ

コード解説（`/explain-code`スキル等）や通常の実装作業の過程で気づいた、
テストシナリオ・データセットバリエーションの過不足を随時記録する場所。
`refactoring-candidates.md`（コード側の重複・デッドコード等）と対になる、
**シナリオ設計側**の未整理メモ。

ここに溜まった項目は、着手時に`.claude/rules/testing-policy.md`「テスト用データセット」の
方針に沿うか確認した上で、Issue化するか`/review-testing`（`testing-completeness-reviewer`）の
確認対象に含めるかを都度ユーザーが判断する。

## 記録フォーマット

各項目は以下を含める。

- **対象**: 系統・ファイルパス
- **内容**: 何が気になったか
- **気づいた経緯**: どの作業中に気づいたか（日付）
- **状態**: 未対応 / 対応済み（対応したIssue・PR等） / 対応不要と判断（理由）

---

## 一覧

### 6. Logit: `SEPARATION_PARAM_NORM_THRESHOLD`の多変量モデル（k大）での誤検知リスクが未検証

- **対象**: [engine/src/nonlinear/logit.rs](../../../engine/src/nonlinear/logit.rs)、
  [docs/spec/logit-spec.md](../../spec/logit-spec.md)4章
- **内容**: `SeparationSuspected`検出に使う標準化パラメータのL2ノルムは、
  `k`が増えるほど各成分が中程度でも合計が大きくなりやすく、真に分離して
  いないケースでの誤検知リスクが理論上あるが、実測での検証は無い。
  検出に使う量（標準化パラメータのL2ノルム）と実際にアンダーフローを
  引き起こす量（線形予測子の最大絶対値）は相関的な関係に過ぎず数学的に
  保証された関係ではないため、特定の1列のみが分離に寄与するケース等で
  検出漏れがありうる点も未検証。
- **気づいた経緯**: 実装時（`docs/spec/logit-spec.md`4章に記載済み）。
  2026-08-15、Issue #231フェーズ4のテスト拡充作業に伴い本メモへ転記・集約。
- **状態**: Issue化済み（[#321](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/321)）。
  2026-09-13に実測で検証したところ、当初想定していた「k大で穏やかな係数が
  積み重なる」ケースではなく、**強い多重共線性**が真の誤検知メカニズムだと
  判明した。列が無相関なら`norm(β_std)²≈Var(線形予測子)`が近似的に成り立ち
  閾値は妥当に機能するが、列が強く相関していると係数が符号反対に大きく
  振れて線形予測子への寄与がほぼ打ち消し合いながらノルムだけが膨張する
  （古典的な多重共線性の症状）ため前提が崩れる。実際にLogitで、列相関を
  ほぼ1に近づけた（`noise_sd=0.0001`）が真の線形予測子は穏やかな非分離
  ロジスティックモデルというデータで、statsmodelsは正常に有限MLEへ収束する
  一方（`beta≈[-409.8, 410.9]`, `SE≈250.7`）、本実装はnewton/bfgs/lbfgs
  全methodで`SeparationSuspected`を誤検知することを確認した（逆算した
  標準化パラメータノルム≈580、閾値100の約5.8倍）。Issue #317（小標本境界
  での検出漏れ、閾値が緩すぎる方向）とは逆方向の問題。

### 15. IV: 複数内生変数対応後もCragg-Donald統計量をv1スコープ外のままにしてよいか（設計判断候補）

- **対象**: `iv-spec.md`3.4節（弱操作変数診断）・`engine/src/iv/two_sls.rs`の
  `partial_f_statistic`（内生変数ごとの単変量部分F統計量のみ実装済み）
- **内容**: ユーザー提案（2026-08-16）。`iv-spec.md`3.4節は「複数内生変数の同時検定
  （Cragg-Donald統計量等）も...v1スコープ外とし、各内生変数ごとの部分F統計量のみ返す」と
  確定していたが、この判断がされた時点では複数内生変数（`k_endog>=2`）のシナリオ自体が
  まだ実装されていなかった可能性がある。その後Issue #231フェーズ4で`multi_endog`シナリオ
  （`benchmark/iv/fixtures/generate_iv_fixtures.py`）が実際に追加され、複数内生変数の
  ケースが実運用でテストされるようになった。各内生変数ごとの部分F統計量だけでは、複数の
  内生変数が絡む「操作変数群全体としての多変量的な弱さ」を検出できない場合がある。
- **Claudeの所感**: 複数内生変数のサポートが実際に進んだ今、v1時点の判断を見直す価値が
  あるかは再検討に値する。Issue化済み（[#247](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/247)）。
- **気づいた経緯**: 2026-08-16、`benchmark/iv/references/linearmodels_ref.py`解説後のユーザー質問。
- **状態**: 未対応（[#247](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/247)で再検討中）

### 16. GMM: C統計量（difference-in-Hansen統計量）による内生性検定が無い（新規機能候補）

- **対象**: `engine/src/iv/gmm.rs`（Wu-Hausman相当の検定が未実装）・
  `benchmark/iv/references/linearmodels_ref.py`の`run_gmm()`
- **内容**: ユーザー提案（2026-08-16）。GMMには2SLSの`wu_hausman_statistic`に相当する
  内生性検定が現状無い（`engine/src/iv/CLAUDE.md`「Wu-Hausman検定はGMMには存在しない」参照）。
  GMMの枠組みで内生性を検定する標準的な手法として**C統計量**（difference-in-Hansen統計量、
  疑わしい変数を内生扱い/外生扱いした2つのモデルのHansen J統計量の差を`χ²`検定する手法、
  Stataの`ivreg2`で実装済み）がある。古典的なWu-Hausman検定は分散の差が半正定値という
  前提が不均一分散・クラスター等のロバスト共分散の下で破綻しうるのに対し、C統計量は
  GMMの重み行列を通じて自然にロバスト対応できるため、GMMではむしろこちらの方が理論的に
  筋が良い。
- **Claudeの所感**: 2つのGMM推定（内生扱い・外生扱い）の重み行列をどう揃えるか等の
  設計判断が必要で、`gmm.rs`側の新規実装が要る（ベンチマークのみでは完結しない）。
  Issue化済み（[#249](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/249)）。
- **気づいた経緯**: 2026-08-16、`generate_iv_gmm_fixtures.py`解説後のユーザー提案。
- **状態**: 未対応（[#249](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/249)で検討中）

### 18. OLS: `gpa2`を`mroz`に置き換え、実データでの線形確率モデル（LPM）検証を追加する案

- **対象**: `benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py`の
  `build_wooldridge_fixtures()`（`wage1`/`gpa2`の2データセット）
- **内容**: ユーザー提案（2026-08-16）。実際に確認したところ、`gpa2`には
  クラスターケースが無く（`if name == "wage1": ...`のみ）、`wage1`との違いは
  「別の連続値`y`の実データでもう一度係数・標準誤差が一致するか確認する」程度で
  独自の検証価値が薄い。一方`mroz`（Logit側で既に使用中、`y=inlf`が0/1）を
  OLSで使えば**線形確率モデル（LPM）の実データ版**という、`wage1`/`gpa2`の
  どちらとも異なる新しい検証内容になる。項目2（合成データでのLPMシナリオ、
  優先度低いと判断済み）とは異なり、既存データセット`mroz`を使い回せるため
  対応コストが低い。
- **Claudeの所感**: `gpa2`を`mroz`に置き換える（`wage1`は地域クラスターの検証役
  として残す）方向は筋が通ると考える。
- **気づいた経緯**: 2026-08-16、`generate_logit_crosscheck_fixtures.py`解説後の
  ユーザー提案。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 19. OLS/WLS: HACの実データクロスチェックが存在しない

- **対象**: `benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py`・
  `generate_wls_crosscheck_fixtures.py`の`WOOLDRIDGE_COV_TYPES`/`hc_types`
  （いずれも`hac`を含まない）
- **内容**: ユーザー指摘（2026-08-16）。OLS/WLSの実データ（`wage1`/`gpa2`/
  `401ksubs`）はいずれも横断面データ（時系列順が無い）のため、`hac`が意図的に
  除外されている（コメント「HACは時系列順の無いクロスセクションデータのため
  対象外」）。nonlinear（Logit/Probit）が構造的に`heteroskedastic`/
  `autocorrelated`シナリオ自体を持たない（既存項目10、設計上の一貫した仕様）のとは
  別の話で、**OLS/WLSはHAC自体をサポートしているのに実データでは一度も
  検証されていない**、という純粋なギャップ。Wooldridgeパッケージに適した時系列
  データ（例: `prminwge`等）を探す必要があり対応コストはやや高い。
- **Claudeの所感**: 優先度は中程度。合成データの`autocorrelated`シナリオで
  HAC自体は検証済みのため、実データでの検証が無くても致命的ではないが、
  「実データでの一致確認」という観点では抜けている。
- **気づいた経緯**: 2026-08-16、`generate_logit_crosscheck_fixtures.py`解説後の
  ユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 20. IV: クラスター時のWu-Hausman検定p値ズレの「根本原因」説明が自動テストで裏付けられていない

- **対象**: `tests/iv/test_iv_crosscheck.py`（`check_wu_hausman_p_value=False`で
  clusterのp値比較自体をスキップしている）・
  [benchmark/iv/references/run_ivreg.R:33-41](../../../benchmark/iv/references/run_ivreg.R#L33-L41)
  （コメントで「G-1で計算するとRのstatisticから本実装のp値が再現できることを
  確認済み」と記載）
- **内容**: ユーザー指摘（2026-08-16）。「統計量は一致するがp値は一致しない」
  こと自体は`test_iv_crosscheck.py`が`check_wu_hausman_p_value=False`でp値比較を
  スキップしつつ統計量は比較する形で自動テストされている。しかし「なぜズレるか」
  （Rのivdiagが常に`n-k`をF分布の分母自由度に使うのに対し、本実装は`G-1`を使う
  ため）という**根本原因の説明**自体は、コメントに「確認済み」とあるだけで、
  それを裏付ける自動テストが無い（一度きりの手動確認が記録として残っているのみ）。
- **Claudeの所感**: Rの`statistic`値を使い、`scipy.stats.f.cdf`等で`G-1`自由度の
  p値を独立計算し、本実装のp値と一致することを確認する専用テストを追加すれば、
  この根本原因の説明を将来にわたって保証できる。
- **気づいた経緯**: 2026-08-16、`benchmark/iv/references/run_ivreg.R`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 21. IV(GMM): RクロスチェックがivregのGMM非対応で省略されている件を再検討する

- **対象**: `docs/spec/iv-spec.md`4章（「GMMのRクロスチェック
  省略（例外規定）」）・`benchmark/iv/fixtures/generate_iv_gmm_fixtures.py`
  （`linearmodels`との照合のみ、Rクロスチェックなし）
- **内容**: ユーザー指摘（2026-08-16）。GMM（Hansen J検定含む）は`linearmodels`
  との数値照合はされているが、独立実装によるRクロスチェックが無い（`ivreg`が
  GMMに非対応なため）。`gmm`パッケージ（Pierre Chaussé作）等、`ivreg`以外に
  GMMを実装しているRパッケージが無いか再調査する価値がある。
  Issue化済み（[#256](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/256)）。
- **気づいた経緯**: 2026-08-16、`benchmark/iv/references/run_ivreg.R`解説後のユーザー指摘
  （C統計量Issue #249と関連するが別の論点として指摘）。
- **状態**: 未対応（[#256](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/256)で検討中）

### 22. Wooldridge実データを使う全テストが標準CI（`ci_python.yml`）で無条件にskipされ、skip自体が検出されない

- **対象**: [pyproject.toml:67-73](../../../pyproject.toml#L67-L73)（`wooldridge==0.5.0`が
  `benchmark`依存グループにあり`test`グループには無い）・
  [.github/workflows/ci_python.yml:44-51](../../../.github/workflows/ci_python.yml#L44-L51)
  （`uv sync --locked --group test`→`pytest tests`、`benchmark`グループは
  インストールしない）・[tests/_helpers.py:89](../../../tests/_helpers.py#L89)
  （`pytest.importorskip("wooldridge")`）
- **内容**: ユーザー指摘（2026-08-22）。`wooldridge`パッケージは`test`依存
  グループではなく`benchmark`依存グループにのみ含まれているため、標準CI
  ワークフロー（`ci_python.yml`、push/PR時に毎回走る）は`wooldridge`を
  インストールしない。このため`tests/_helpers.py`の`wooldridge_loader`/
  `load_wooldridge_dataset`を使う全てのWooldridge実データテスト
  （`test_ols_crosscheck.py`のwage1/gpa2、`test_wls_*.py`の401ksubs、
  `test_logit_*.py`/`test_probit_*.py`のmroz、`test_iv_*.py`のcard等、多数）は、
  `pytest.importorskip("wooldridge")`により**標準CIでは常にskipされる**。
  `pyproject.toml`のコメントには「`wooldridge`パッケージ自体はMITライセンス
  だが、同梱される実データの著作権は原典教科書側にある可能性があり、
  再配布してよいか未確認のため都度ロードする」という意図的な設計判断が
  書かれているが、その代償として実データクロスチェックが標準CIでは一度も
  実行されないという副作用が生じている。
  加えて`ci_python.yml`の`pytest`ステップは`-rs`（skip理由の一覧表示）や
  skip数のしきい値チェックを設定しておらず、pytestのデフォルト出力
  （サマリー行に`N skipped`と出るのみ）に頼っているため、Wooldridge関連の
  skipが増減してもCIログを注意深く読まない限り気づけない。
- **Claudeの所感**: 実データの再配布可否が未確認という制約自体は`benchmark/`
  freeze対象外の判断（`testing-policy.md`）と整合しており妥当だが、
  「CIで実行されないテストがある」という事実そのものが常時可視化されていない
  点は改善の余地がある。対応案としては、(a) CIワークフローで`pytest`に
  `-rs`を付けてskip理由を必ずログへ出す、(b) skip件数が既知の想定値
  （Wooldridge関連テストの件数）と一致することを確認するステップを足す、
  (c) 別途`wooldridge`込みの任意ジョブ（`workflow_dispatch`等）を用意し
  定期的に実行する、等が考えられるが、いずれもユーザー判断が必要。
- **気づいた経緯**: 2026-08-22、`tests/_helpers.py`解説後のユーザー指摘
  （`sys.path.insert`最小化の相談に付随して、CI側でWooldridgeテストが
  実行されない可能性を懸念）。
- **状態**: 未対応（対応方針・優先度はユーザー判断待ち）

### 23. `logit_crosscheck`/`probit_crosscheck`の基本`rtol`（2e-4）だけ、他の全エントリと違い実測根拠のコメントが無い

- **対象**: [tests/_tolerances.py:87-89](../../../tests/_tolerances.py#L87-L89)
  （`"logit_crosscheck": {"rtol": 2e-4, "atol": 1e-8, ...}`）・
  [tests/_tolerances.py:100-102](../../../tests/_tolerances.py#L100-L102)
  （`"probit_crosscheck": {"rtol": 2e-4, "atol": 1e-8, ...}`）
- **内容**: ユーザー指摘（2026-08-22）を受けてファイル全体を確認したところ、
  `TOLERANCES`辞書の他の全エントリ（`rtol_hac`・`atol_p_value`・
  `rtol_margeff_se`・`rtol_near_separation_conf_int`・`rtol_mroz_cluster`等）は
  いずれも「実測最大◯◯（具体的な数値）にマージンを載せた」という形の
  コメントが付いているが、`logit_crosscheck`/`probit_crosscheck`の**基本**
  `rtol=2e-4`・`atol=1e-8`にだけ、なぜこの値なのかを示す実測根拠のコメントが
  無い（`ols_crosscheck`等の基本`rtol_strict`/`atol`はブロック先頭のコメントで
  「機械精度一致（実測1e-14程度）」という根拠が示されているのと対照的）。
- **Claudeの所感**: `testing-policy.md`「許容誤差」の方針
  （「実測値（最大相対誤差）に基づいて具体的な数値を決める」）に沿うなら、
  この基本値についても実測根拠が本来必要なはず。値自体は恐らく実装時に
  実測した上で決めたと推測されるが、コメントとして残っていないため、
  今の状態では「本当に実測に基づく値か」「たまたま通っている緩すぎる値では
  ないか」を後から検証できない。一度実測し直し、コメントとして残すことを
  推奨する。
- **気づいた経緯**: 2026-08-22、`tests/_tolerances.py`解説後のユーザー指摘。
- **状態**: 未対応（実測・コメント追記の要否はユーザー判断待ち）

### 24. Logitのmrozクラスターcrosscheckテストだけ、Probitと違い専用の緩めた許容誤差を使っていない（数値ノイズの有無が未検証）

- **対象**: [tests/nonlinear/test_logit_crosscheck.py:233-245](../../../tests/nonlinear/test_logit_crosscheck.py#L233-L245)
  （`test_mroz_cluster_matches_r_glm`、`rtol`指定無しで基本値2e-4のまま）と
  対比した[tests/nonlinear/test_probit_crosscheck.py:243-264](../../../tests/nonlinear/test_probit_crosscheck.py#L243-L264)
  （同名テストで`RTOL_MROZ_CLUSTER = TOLERANCES["probit_crosscheck"]["rtol_mroz_cluster"]`
  = 2e-3を明示的に使用）
- **内容**: ユーザー依頼（2026-08-22）で確認。`tests/_tolerances.py`の
  `probit_crosscheck`には「Wooldridge mrozのクラスターロバストSE
  （cluster_col="city"、G=2）は合成データのクラスターケースより数値ノイズが
  大きい（実測最大相対誤差~1.1e-3、const）」という専用エントリ
  `rtol_mroz_cluster`があり、Probit側のテストはこれを明示的に使っている。
  一方Logit側の同名テスト（`test_mroz_cluster_matches_r_glm`）は`_assert_dict_close`
  を`rtol`指定無しで呼んでおり（Logit版の`_assert_dict_close`は`atol`しか
  引数に取らず`rtol`は`_assert_close`のデフォルト値=基本の2e-4に固定される
  実装になっている）、Probitと同じ現象（G=2という境界的なクラスタ数＋実データ
  特有のノイズ）が起きているはずのケースで専用の緩和が無い。
- **Claudeの所感**: 2つの可能性がある。(a) Logitでは実際にこの数値ノイズが
  起きておらず基本の2e-4で余裕を持って通っている（Probit固有の現象、
  リンク関数の違いによる数値的な性質の差）、(b) 誰もLogit側でこのケースの
  実測乖離を測っておらず、たまたま2e-4以内に収まっているだけで検証されて
  いない。項目23（基本rtolの実測根拠が無い）とも関連するため、実測して
  どちらか確認するのが望ましい。
- **気づいた経緯**: 2026-08-22、ユーザー依頼により`test_logit_crosscheck.py`/
  `test_probit_crosscheck.py`を突き合わせて確認。
- **状態**: 未対応（実測確認の要否はユーザー判断待ち）

### 25. `conftest.py`の`dataset`が説明変数2個・同分布のため、係数・標準誤差の列対応（順序）バグを検出しづらい

- **対象**: [tests/conftest.py:16-30](../../../tests/conftest.py#L16-L30)
  （`dataset`フィクスチャ、`x1`/`x2`とも`rng.normal(0.0, 1.0, n)`で同一分布）
- **内容**: ユーザー指摘（2026-08-22）。`test_ols.py`等の構造テストは
  `zip(["const", "x1", "x2"], sm_res.params)`のように名前と位置を対応付けて
  比較しており、真の係数（`x1=2.0`, `x2=-0.5`）が異なるため現状の2変数・
  このシードでは列の入れ替わりバグを検出できると考えられる。しかし
  (1) 説明変数が2個しかないため「入れ替わり」パターンが1通りしかなく、
  たまたま推定値が近くなる悪いseedを引くリスクをゼロにできない、
  (2) `x3`が実は`x5`の列に入っていた、のような**より複雑な列対応バグ**
  （3変数以上でしか起こりえないクラスのバグ）は原理的に検出できない、
  という2つの構造的な穴がある。
- **Claudeの所感**: 説明変数を7〜8個に増やし、かつ真の係数を意図的に
  バラけさせる（隣接する値が偶然近くならないようにする）ことで検出力が
  上がると考える。ただし`conftest.py`の`dataset`を直接拡張するか、
  `refactoring-candidates-2.md`項目6（データ生成ライフサイクルを`benchmark/`に
  揃えるか）とセットで`benchmark/`側に切り出すかは設計判断が要るため、
  着手前にユーザー確認が必要。
- **気づいた経緯**: 2026-08-22、`tests/linear/test_ols.py`解説後のユーザー指摘。
- **状態**: 未対応（設計判断待ち、`refactoring-candidates-2.md`項目6と関連）

### 27. `include_intercept=False`・`confidence_level`オプションの効果が、frozen JSON数値照合（fixturesパイプライン）で検証されていない

- **対象**: [benchmark/linear/datasets.py](../../../benchmark/linear/datasets.py)・
  [benchmark/linear/fixtures/generate_ols_fixtures.py](../../../benchmark/linear/fixtures/generate_ols_fixtures.py)
  （どちらにも`include_intercept`・`confidence_level`という文字列が0件）
- **内容**: ユーザー指摘（2026-08-22）を受けて確認。`OLSOptions`の主要な
  フィールドのうち、`include_intercept=False`（切片なし回帰）と
  `confidence_level`（既定0.95以外の信頼水準）は、`tests/linear/test_ols.py`内の
  即席データによる簡易statsmodels比較でのみ検証されており、
  `test_ols_fixtures.py`のfrozen JSON数値照合パイプラインには一度も
  登場しない。なお`conf_int`自体（既定95%信頼区間の値）は
  [tests/linear/test_ols_fixtures.py:85-87](../../../tests/linear/test_ols_fixtures.py#L85-L87)
  で既に数値照合済み（冗長ではなく既存カバレッジ）だが、
  `confidence_level`を変更したときの効果は
  [tests/linear/test_ols.py:353-374](../../../tests/linear/test_ols.py#L353-L374)
  `test_confidence_level_changes_interval_width`が相対比較
  （狭くなる/広くなる）のみで、具体的な数値の正しさまでは見ていない。
  `test_predict_new_data_without_intercept_matches_statsmodels`
  （[tests/linear/test_ols.py:570-588](../../../tests/linear/test_ols.py#L570-L588)）も同様に
  即席データのみでの検証。
- **Claudeの所感**: `testing-policy.md`が要求する「全てのオプションの組み合わせで
  リファレンス実装と統計量が一致することを確認する」の対象漏れだと考える。
  `include_intercept=False`のシナリオを`benchmark/linear/datasets.py`に追加し、
  `generate_ols_fixtures.py`側でcov_type全種と組み合わせて数値照合すれば、
  `refactoring-candidates-2.md`項目52（`test_ols.py`の役割の非対称性）の
  解消（`test_ols.py`から簡易数値比較を削る）の前提条件にもなる。
- **気づいた経緯**: 2026-08-22、`tests/linear/test_ols.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、`refactoring-candidates-2.md`
  項目52と関連）

### 29. クラスターロバストSEが、どの検証層でも`baseline`シナリオでしか数値比較されていない（悪条件・境界シナリオとの組み合わせが未検証）

- **対象**: [benchmark/linear/fixtures/generate_ols_fixtures.py:76-92](../../../benchmark/linear/fixtures/generate_ols_fixtures.py#L76-L92)
  （`if scenario == "baseline":`ブロック内でのみクラスターケースを生成）、
  `tests/linear/test_ols_fixtures.py`のクラスター系4テスト（`scenario`の
  `parametrize`無し、`synthetic_baseline.csv`/`synthetic_baseline_k1.csv`
  固定）、`tests/linear/test_ols_crosscheck.py`の同名クラスター系テスト（同じく
  `scenario`の`parametrize`無し）、`engine/src/linear/ols.rs`のクラスター
  単体テスト（`fit_computes_cluster_std_errors_...`等、リファレンス実装との
  数値比較を伴わない純粋ロジック検証のみ）
- **内容**: ユーザー指摘（2026-08-23）。「クラスターロバストSEは
  シナリオ依存ではなくグルーピングの動作確認が目的」という設計コメント
  （[generate_ols_fixtures.py:76](../../../benchmark/linear/fixtures/generate_ols_fixtures.py#L76)）
  に基づき、クラスター系テストは`baseline`（良条件・標準的なn）以外の
  シナリオでは一度も数値照合されていないことを、Python fixtures層・R
  crosscheck層・Rust単体テスト層の3層全てで確認した。しかしクラスター
  ロバスト共分散`Ŝ=(X'X)⁻¹(Σ_g X_g'e_ge_g'X_g)(X'X)⁻¹`は`(X'X)⁻¹`を
  他のcov_type（classical/HC0-3/HAC）と共有しており、`high_condition_number`
  （悪条件設計行列）や`baseline_df1`（自由度1境界）のような、他のcov_typeでは
  全シナリオで検証している悪条件・境界ケースとクラスターの組み合わせでの
  数値的挙動は未検証のまま。
- **Claudeの所感**: 「クラスターSEの計算式自体はシナリオに依存しない」という
  設計コメントの主張は、疑似グループの割り当て方（均等/不均衡/G境界）に
  関しては正しいが、「シナリオ由来の設計行列の条件（悪条件・自由度境界等）が
  クラスター計算の数値安定性に影響しないか」までは検証していない別の論点。
  `engine/src/linear/CLAUDE.md`に記録されている「G=qちょうどの境界でも
  データの配置次第では特異になりうる」という既知の罠（Tobit実装時に実測発覚）
  を踏まえると、悪条件シナリオ×クラスターの組み合わせで同様の未知の
  数値的落とし穴が無いとは言い切れない。最低限`high_condition_number`または
  `moderate_multicollinearity`のいずれか1シナリオでクラスターケースを
  追加し、数値照合できることを確認するのが妥当と考える。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_ols_fixtures.py`解説中の
  ユーザー指摘（「clusterに関してはシナリオごとで検証する必要はないのか、
  精度漏れの可能性が残ることは避けたい」）を受けて3層を確認。
- **状態**: 対応済み（OLS、2026-09-21）。`high_condition_number`・
  `moderate_multicollinearity`の両シナリオ（「いずれか1シナリオ」という
  所感に対し、より手厚くする方針でユーザー確認の上、両方追加）に、
  均等な疑似グループ（行番号%10）のみのクラスターケースを追加した。
  `benchmark/linear/fixtures/generate_ols_fixtures.py`の`_run_cluster_case`が
  `scenario`引数を取れるよう拡張、`generate_ols_crosscheck_fixtures.py`にも
  同様の`CLUSTER_ILL_CONDITIONED_SCENARIOS`定数と分岐を追加。
  `tests/linear/test_ols_reference.py::test_cluster_ill_conditioned_matches_
  statsmodels`・`tests/linear/test_ols_crosscheck.py::test_cluster_ill_
  conditioned_matches_r`を追加し、Python fixtures層・Rクロスチェック層の
  両方で悪条件・多重共線性シナリオとクラスターの組み合わせが数値的に
  問題なく計算できることを確認した（Rust単体テスト層はリファレンス実装との
  数値比較を目的としないため対象外のまま）。`tests/`配下1675件全通過・
  Ruffクリーンを確認済み。WLS側（`generate_wls_fixtures.py`等）は同じ
  ギャップが存在するが、ユーザー判断によりこの場では対応せず
  [Issue #351](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/351)
  として切り出した（他手法〔IV/Logit/Probit等〕への横展開要否も同Issueで
  検討）。2026-09-21、項目17対応時に`testing-completeness-reviewer`が項目28と
  合わせて再指摘（predict()同様、クラスター系の検証網羅性を先に手厚くした
  Rクロスチェック側に主リファレンス側を追いつかせる、という同型の対応が
  必要という指摘）。

### 30. `time_col`が存在しない列名を指した場合の`ValidationError`テストが無い（`cluster_col`には対になるテストがある）

- **対象**: [tests/linear/test_ols.py:166-173](../../../tests/linear/test_ols.py#L166-L173)
  （`test_cluster_col_nonexistent_column_raises`、`cluster_col`が存在しない
  列を指す場合の専用テスト）と対比した、`time_col`に対する同種テストの不在。
  実装は[engine_pybind/src/linear/common.rs:99-107](../../../engine_pybind/src/linear/common.rs#L99-L107)
  （`cov_type="hac"`のとき`time_col`を`extract_f64_column`で抽出）。
- **内容**: ユーザー依頼（2026-08-23）を受けて確認。実装自体は正しく
  動作する（実機確認済み: `OLSOptions(cov_type="hac", hac_lags=1,
  time_col="does_not_exist")`で`ValidationError("column 'does_not_exist'
  does not exist in the data")`が発生）。しかしこれを確認する
  Pythonテストが`test_ols.py`に無い。`cluster_col`側には
  `testing-completeness-reviewer指摘、Issue #231フェーズ4`という経緯で
  追加された専用テストがあるのに、`time_col`には対になるテストが
  追加されていない非対称な状態。
- **Claudeの所感**: 実装は正しいため緊急度は低いが、`cluster_col`と
  `time_col`は同じ「`cov_type`固有の追加列」という位置づけ
  （`engine_pybind/src/linear/CLAUDE.md`「`cov_type`固有の追加列」参照）で
  あり、片方だけテストがあるのは網羅性の観点で片手落ち。
  `test_cluster_col_nonexistent_column_raises`と同じパターンで数行
  追加すれば埋められる。
- **気づいた経緯**: 2026-08-23、ユーザー依頼により`test_ols.py`の
  バリデーション網羅性を確認中に発見。
- **状態**: 未対応（着手要否はユーザー判断待ち、修正は保留）

### 31. `fit()`本体（`y`/`x`列）でNaN・無限大を含む場合のテストが無い（`predict()`側にはある）

- **対象**: [tests/linear/test_ols.py:181-184](../../../tests/linear/test_ols.py#L181-L184)
  （`test_null_values_raise`、null値のみ）と対比した
  [tests/linear/test_ols.py:651-660](../../../tests/linear/test_ols.py#L651-L660)
  （`test_predict_null_or_non_finite_values_raise`、`predict()`の`new_data`は
  nullと`float("inf")`の両方をテスト済み）。実装は
  [engine_pybind/src/column_extraction.rs:65-72](../../../engine_pybind/src/column_extraction.rs#L65-L72)
  （`extract_f64_column`、コメント「polarsの`null_count()`はNaN/無限大を
  検出しない...別途スキャンする必要がある」の通り、null検証とNaN/Inf検証は
  別ロジック）。
- **内容**: ユーザー依頼（2026-08-23）を受けて`test_ols.py`のバリデーション
  網羅性を確認中に発見。`fit()`が受け取る`y`/`x`列（学習データ本体）は
  null値のテストのみで、NaN・無限大（`float("inf")`/`float("nan")`）を
  含む場合のテストが無い。同じ`extract_f64_column`関数を使う`predict()`の
  `new_data`側には両方のテストがあるのと非対称。
- **Claudeの所感**: null検証とNaN/Inf検証は`extract_f64_column`内で
  別々のスキャン（`null_count()`とその後の`is_finite()`ループ）のため、
  片方だけ通っても他方が壊れていることに気づけない構造。`predict()`側に
  ある`test_predict_null_or_non_finite_values_raise`と対になる
  `fit()`側のテストを追加するのが妥当。
- **気づいた経緯**: 2026-08-23、ユーザー依頼により`test_ols.py`の
  バリデーション網羅性を確認中に発見。
- **状態**: 対応済み（OLS、2026-09-23）。`tests/linear/test_ols_validation.py`の
  `test_null_values_raise`・`test_non_finite_values_raise`に`x1`列の
  null・NaN・無限大ケースを追加し、`y`列側と対称に、かつ`predict()`側の
  `test_predict_null_or_non_finite_values_raise`と同じ範囲まで検証する
  ようにした。37件全通過・Ruffクリーンを確認済み。WLS側（項目34）には
  同型のギャップが残っている。

### 32. `y`列自体が存在しない場合・`cluster_col`にNull値を含む場合の専用テストが無い（低優先度、同一コードパスの既存テストで実質カバー済み）

- **対象**: [tests/linear/test_ols.py:176-178](../../../tests/linear/test_ols.py#L176-L178)
  （`test_missing_column_raises`、`x=["x1", "nonexistent"]`のみで`y`側の
  欠落は未テスト）／`cluster_col`のNull値ケース（テスト無し）
- **内容**: ユーザー依頼（2026-08-23）を受けたバリデーション網羅性確認の
  副産物。(1) `y`が存在しない列名の場合の専用テストが無く、`x`が存在しない
  ケースのみテストされている。(2) `cluster_col`がNull値を含む場合の専用
  テストも無い。
- **Claudeの所感**: いずれも`extract_f64_column`/`extract_group_key_column`
  という共有関数の同じ分岐（「列が存在しない」「Nullを含む」）を通るため、
  `x`側・「列が存在しない」ケースで既に間接的に検証されており、バグを
  見逃すリスクは項目30・31より低いと判断する。優先度は低い。
- **気づいた経緯**: 2026-08-23、ユーザー依頼により`test_ols.py`の
  バリデーション網羅性を確認中に発見。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち、修正は保留）。
  2026-09-21、項目17対応のレビューで`testing-completeness-reviewer`が
  項目31と合わせて再指摘（`fit()`側バリデーションの非対称パターンの一例として）。

### 34. `test_wls_validation.py`にもOLSと同型のバリデーション抜けがある（`y`列自体の欠落・`fit()`本体のNaN/無限大・空文字列の列名）

- **対象**: [tests/linear/test_wls_validation.py:175](../../../tests/linear/test_wls_validation.py#L175)
  （`test_missing_column_raises`、`x`側のみ`x=["x1", "nonexistent"]`、`y`側の
  欠落は未テスト）・[tests/linear/test_wls_validation.py:199](../../../tests/linear/test_wls_validation.py#L199)
  （`test_null_values_raise`）・[tests/linear/test_wls_validation.py:222](../../../tests/linear/test_wls_validation.py#L222)
  （`test_non_finite_values_raise`、対応済み後の現状）。当時の対象ファイル名は
  `test_wls.py`だったが、その後`test_wls_validation.py`等に分割された
  （項目自体の内容は変わらない）。
- **内容**: ユーザー依頼（2026-08-23）を受けて`test_wls.py`のバリデーション
  網羅性を確認したところ、項目30〜32（`test_ols.py`）と同型の抜けが存在した。
  (1) `y`が存在しない列名の場合の専用テストが無い。(2) `fit()`本体の`y`/`x`
  列でNaN・無限大を含む場合のテストが無い（`weight`列は既に分割済みで
  対照的）。(3) `y=""`/`weight=""`/`cluster_col=""`（空文字列）の専用テストが
  無い（実機確認では`y=""`は`ValidationError("column '' does not exist in
  the data")`として正しく動作しており、優先度は低い）。
- **Claudeの所感**: (2)は`testing-completeness-reviewer`のレビュー観点に
  追加した「列引数ごとのバリデーション3点セット」で今後拾えるはずだが、
  既存分としては未対応のまま残っている。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_wls.py`解説後のユーザー指摘を
  受けた確認。
- **状態**: (2)は対応済み（WLS、2026-09-23）。項目31のOLS対応と同じ形で、
  `test_null_values_raise`に`x1`列のnullケースを追加し、新規
  `test_non_finite_values_raise`（`y`/`x1`×NaN/無限大の4ケース）を追加した。
  42件全通過・Ruffクリーンを確認済み。(1)・(3)は未対応のまま
  （優先度低、着手要否はユーザー判断待ち。OLS側の項目32も同じ状態で
  対称性は保たれている）。

### 36. WLSのHACクロスチェックで、statsmodels側とR側が異なるラグ値でNewey-West公式を検証しており、同一設定が両方の独立実装から検証されていない

- **対象**: [tests/linear/test_wls_fixtures.py:70](../../../tests/linear/test_wls_fixtures.py#L70)
  （`HAC_LAG_IN_FIXTURE = 1`という固定値、statsmodels側）と
  [tests/linear/test_wls_crosscheck.py:205-206](../../../tests/linear/test_wls_crosscheck.py#L205-L206)
  （`entry["hac_lag"]`という本実装の自動選択ラグ、R側）
- **内容**: ユーザー指摘（2026-08-23、「Newey-West公式の実装自体が正しいか
  確認するなら、statsmodels側でも同様のことを行い、Rクロスチェックと
  対応する形にしないとクロスチェックが成立しなくならないか」）。
  statsmodels側は恣意的な固定値`1`、Rクロスチェック側は本実装の自動選択
  ラグ値（`autocorrelated`シナリオでは`1`より大きい値になる）を使っており、
  それぞれ異なるラグでNewey-West公式を検証している。結果として
  「ラグ=1」の設定はstatsmodelsのみが、「ラグ=自動選択値」の設定はRのみが
  検証しており、**同一の設定が両方の独立実装（三角測量）から検証されている
  わけではない**。
- **Claudeの所感**: 実害としては「ラグ依存のバグ」があった場合に一方の
  テストでしか拾えない可能性がある、という理論上のリスク。一方で見方を
  変えれば検証しているラグのバリエーションが広がっているとも言える。
  Tobit等、既に一部の統計量でR単独のクロスチェックに頼ることを許容している
  前例があるため、今回のケースも許容範囲というユーザー判断に同意する。
  対応するなら「statsmodels側でも自動選択ラグを使う」または「R側にも
  固定ラグ=1のケースを追加する」のどちらかで同一設定を両実装から検証する
  形に揃えられる。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_wls_crosscheck.py`解説後の
  ユーザー指摘。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 37. `test_marginal_effects_default_excludes_intercept`が部分集合チェック（`<=`）で、余分なキーが混入しても検出できない

- **対象**: [tests/nonlinear/test_logit.py:170-185](../../../tests/nonlinear/test_logit.py#L170-L185)
  （`assert expected_keys <= set(row.keys())`）
- **内容**: ユーザー指摘（2026-08-23）を受けて確認。`marginal_effects()`が
  返す行の実際のキー数（Rust側`MarginalEffectsResult`のフィールド数）は
  `expected_keys`と完全に一致しており（7個ずつ）、`<=`（部分集合）より
  `==`（完全一致）の方が厳密で、かつ現状の実装と矛盾しない。
- **Claudeの所感**: `==`に変えるだけの小さい修正で、将来意図しないキーが
  追加された場合の検出力が上がる。実施しやすい部類。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 38. `test_marginal_effects_at_is_case_insensitive`が`"overall"`のみ検証しており`"mean"`/`"median"`の大文字小文字非依存性は未検証

- **対象**: [tests/nonlinear/test_logit.py:200-204](../../../tests/nonlinear/test_logit.py#L200-L204)
  （`at="OVERALL"`と`at="overall"`の比較のみ、`pytest.mark.parametrize`化
  されていない）
- **内容**: ユーザー指摘（2026-08-23）を受けて確認。`at`は`"overall"`/
  `"mean"`/`"median"`の3値を取るオプションだが、大文字小文字非依存性の
  確認は`"overall"`のみで、`"mean"`/`"median"`側は未検証。
- **Claudeの所感**: `@pytest.mark.parametrize("at", ["mean", "median",
  "overall"])`化すれば3値とも同じテストでカバーできる。実施しやすい。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 39. `test_marginal_effects_confidence_level_out_of_range_raises`が`1.5`のみ検証しており、`fit()`本体側のような境界値（`0.0`・負値）が未検証

- **対象**: [tests/nonlinear/test_logit.py:213-218](../../../tests/nonlinear/test_logit.py#L213-L218)
  （`confidence_level=1.5`のみ）と、対比した
  [tests/nonlinear/test_logit.py:291-303](../../../tests/nonlinear/test_logit.py#L291-L303)
  （`test_invalid_confidence_level_raises`、`fit()`側は`[1.5, 0.0, -0.1]`を
  `pytest.mark.parametrize`で検証済み）
- **内容**: ユーザー指摘（2026-08-23、「こういう系統は-1とかでも検証して
  いなかった？」）を受けて確認。同じ「`confidence_level`が(0,1)範囲外」
  という検証観点が`fit()`側（`LogitOptions.confidence_level`）では境界値
  `0.0`・負値`-0.1`まで含めてparametrize済みだが、`marginal_effects()`側
  は`1.5`（上限超過）のみで下限側（`0.0`・負値）が未検証という非対称が
  ある。
- **Claudeの所感**: `fit()`側と同じ`[1.5, 0.0, -0.1]`にparametrize化すれば
  対称になる。実施しやすい部類。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 40. Logitにも項目32（`y`列自体が存在しない場合の専用テストが無い）と同型の抜けがある

- **対象**: [tests/nonlinear/test_logit.py:247-249](../../../tests/nonlinear/test_logit.py#L247-L249)
  （`test_missing_column_raises`、`x=["does_not_exist"]`のみで`y`側の
  欠落は未テスト。OLSの項目32と同じ非対称）
- **内容**: ユーザー指摘（2026-08-23、「yのdoes_not_exist列検証がない。
  同様にテストの抜けがないか確認してほしい」）を受けて確認。OLSの項目32
  と全く同じパターンがLogitにもそのまま存在する。あわせて確認した結果、
  以下は新規の抜けとしては該当しなかった（項目32と同じ理由＝共有関数の
  同じ分岐が`x`側で間接的に検証済み、または元々方針上意図的に未実施）。
  - `x`列単体でのnull値専用テストは無い（`test_null_values_raise`は`y`側
    のみnullにしている）が、`x`/`y`とも`extract_f64_column`の同じ分岐を
    通るため項目32と同じ理由でリスクは低いと判断。
  - NaN/無限大の専用テストが無いのは、既存分（OLS/Logit/Probit）は
    そのままにする、というユーザー既定方針（2026-08-23、`test_wls.py`
    解説時点）通りの想定内の状態であり新規の抜けではない。
  - エラーメッセージの内容検証が無い点は、`tests/`全体に共通する既知の
    抜け（項目26、`test_ols.py`解説時に発見済み）がLogitでも同様に
    再現しているのみで、Logit固有の新規項目ではない。
- **Claudeの所感**: `y`側の`test_missing_column_raises`相当のテストを
  追加する程度の小さい対応で足りる。優先度は項目32と同程度（低）で
  良いと考える。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit.py`解説後のユーザー指摘。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 41. `method`（bfgs/lbfgs）と`cov_type`・シナリオ・クラスターの組み合わせが検証されていない

- **対象**: [tests/nonlinear/test_logit_fixtures.py:142-152](../../../tests/nonlinear/test_logit_fixtures.py#L142-L152)
  （`test_matches_statsmodels`、`method`は既定〔newton〕固定で
  `cov_type`×シナリオを網羅）と
  [tests/nonlinear/test_logit_fixtures.py:197-216](../../../tests/nonlinear/test_logit_fixtures.py#L197-L216)
  （`test_method_matches_statsmodels`、`method`はbfgs/lbfgsを網羅するが
  `cov_type="classical"`・`scenario="baseline"`に固定）
- **内容**: ユーザー指摘（2026-08-23、「`test_matches_statsmodels`に
  `test_method_matches_statsmodels`のメソッド照合を含めたほうがいいので
  は？モデルごとのcov_type、シナリオ検証が漏れていると思う。加えて
  methodごとのクラスタ検証も漏れていない？」）を受けて確認。`method`は
  `classical`×`baseline`の1点でしか他のcov_type・シナリオ・クラスター
  ケースと掛け合わされておらず、「`method=bfgs`×`cov_type=hc0`」
  「`method=lbfgs`×`scenario=near_separation`」「`method=bfgs`×
  クラスターロバストSE」等の組み合わせは`tests/`のどこにも存在しない。
  `testing-policy.md`「テストの3系統」・レビュー観点3（オプションの
  組み合わせで未検証の組が無いか）に該当する抜け。
- **Claudeの所感**: 全組み合わせ（6シナリオ×3cov_type×3method×クラスター
  3種）を網羅すると組み合わせ爆発になるため、代表的な組み合わせ
  （例: 最も難しいシナリオ`near_separation`×bfgs/lbfgs、クラスター×
  bfgs/lbfgsを1ケースずつ）に絞って追加するのが現実的と考える。
  `test_matches_statsmodels`に`method`を第3の`parametrize`として
  丸ごと含める案は組み合わせ数が9倍（3method×3cov_type×6scenario）に
  膨らみCI時間が増えるため、代表ケースのみの追加が良いと考える。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit_fixtures.py`解説後の
  ユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）。
  - **2026-09-13追記（項目4クローズ時の派生調査で判明した具体例）**: Tobitで
    `method`が`raise_on_non_convergence=False`の挙動に実際に大きく影響する
    ケースを実測で確認した。`x1~Uniform(-2,2), x2~Uniform(-1,1),
    y*=-1.0+0.3·x1+0.2·x2+N(0,1)`を左打ち切り（打ち切り率84.5%、n=200,
    seed=7）で`max_iter=1, raise_on_non_convergence=False`にすると:
    - `method="newton"`/`"bfgs"`: 全cov_type（classical/opg/hc0/hc1/cluster）
      で例外なく成功（`sigma`はそれぞれ0.372/0.640で1.7倍程度の差）。
    - `method="lbfgs"`: `cov_type="opg"`のみ成功（`sigma=1.53`）、
      `classical`/`hc0`/`hc1`/`cluster`は`MleError::SingularHessian`
      （`ComputationError`）を送出。
    - これは**バグではなく仕様通り**（`TobitEstimator::fit`のdocコメントに
      「収束点（または`raise_on_non_convergence=false`時の打ち切り点）の
      Hessianが特異なら`SingularHessian`」と明記済み。`raise_on_non_
      convergence`が抑制するのは`NonConvergence`のみで、打ち切り点の
      Hessian特異性チェックとは独立した別のエラー経路のため）。
    - 軽度な打ち切りのbaselineシナリオ（項目4のクローズ時に実測）では
      newton/bfgs/lbfgs間で`sigma`・標準誤差ともほぼ一致しており、
      method依存の挙動差は「打ち切りが重い等の悪条件シナリオ」で
      顕在化しやすいと考えられる。
    - Logit/Probitでは同じ実測（baselineシナリオ）でmethod間の差は
      ほぼ無く、Tobitの`(β, logσ)`尤度が大域凹でない構造
      （`engine/src/nonlinear/CLAUDE.md`参照）に起因する可能性が高い。
  - **項目4クローズ時に判明したもう1つの積み残し**: `engine/src/nonlinear/`側の
    Rust単体テスト（`fit_returns_unconverged_result_without_raising_when_
    raise_on_non_convergence_is_false`等、Logit/Probit/Tobit）も
    `CovType::Classical`固定のままで、Python側（`tests/nonlinear/`）で
    項目4により埋めたのと同じcov_type網羅ギャップがRustエンジン層にも
    対称的に残っている。

### 42. `test_logit_crosscheck.py`の`_check_margeff`が`z`/`p_value`/`conf_low`/`conf_high`を検証していない（フィクスチャには既に存在するデータ）

- **対象**: [tests/nonlinear/test_logit_crosscheck.py:105-118](../../../tests/nonlinear/test_logit_crosscheck.py#L105-L118)
  （ローカル`_check_margeff`、`dydx`/`std_err`のみ検証）と対比した
  [tests/_assertions.py:59-113](../../../tests/_assertions.py#L59-L113)
  （共通`check_margeff`、`dydx`/`std_err`/`z`/`p_value`/`conf_low`/
  `conf_high`の6項目を検証）
- **内容**: ユーザー指摘（2026-08-23、「`_check_result`にて`_check_margeff`
  がrefになければ漏れるのでこれも危険な気がする」という質問を受けて
  `_check_margeff`の中身自体を精査）。`tests/fixtures/benchmarks/
  logit_crosscheck.json`を実際に確認したところ、`margeff`エントリには
  `dydx`/`se`だけでなく`z`/`p_value`/`conf_low`/`conf_high`も全て
  含まれていた（R`marginaleffects`パッケージの出力をそのまま記録済み）。
  つまり**データは既に存在するのに、このテストは6項目中2項目
  （`dydx`/`std_err`）しか検証しておらず、`z`/`p_value`/`conf_low`/
  `conf_high`はRとの一致を一度も確認していない**。同じ`marginal_effects()`
  の限界効果検証でも、statsmodels側（`test_logit_fixtures.py`、
  `_assertions.py`の`check_margeff`経由）は6項目全て検証しているため、
  RクロスチェックだけがStatsmodels比較より検証範囲が狭いという非対称が
  ある。
- **Claudeの所感**: `testing-policy.md`レビュー観点1（クロスチェックの
  対象は係数・標準誤差に限らず公開する統計量は全て検証する）に反する
  明確な抜け。フィクスチャデータは既に揃っているため、`_assertions.py`の
  `check_margeff`をこのファイルでも使う形に統一すれば（項目91と合わせて
  対応）、コードを増やさずにこの抜けも同時に解消できる。
- **気づいた経緯**: 2026-08-24、`tests/nonlinear/test_logit_crosscheck.py`解説後の
  ユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目91と合わせて対応可能）

### 43. `mroz`実データのクラスターロバストSEテスト（`test_logit_fixtures.py`/`test_logit_crosscheck.py`双方）が`coef`/`se`のみ検証しており、フィクスチャに既に存在する`z_stats`/`p_values`/`conf_int`/適合度統計量/`margeff`を検証していない（synthetic疑似クラスタ側は生成物と一致しており対象外）

- **対象**: [tests/nonlinear/test_logit_fixtures.py:287-299](../../../tests/nonlinear/test_logit_fixtures.py#L287-L299)
  （`test_mroz_cluster_matches_statsmodels`）、
  [tests/nonlinear/test_logit_crosscheck.py:223-235](../../../tests/nonlinear/test_logit_crosscheck.py#L223-L235)
  （`test_mroz_cluster_matches_r_glm`）。いずれも`_assert_dict_close
  (res.params, ...)`・`_assert_dict_close(res.std_errors, ...)`の2行のみ。
  Probit側の対応するテスト（`test_probit_fixtures.py::test_mroz_cluster_
  matches_statsmodels`）にも同型の抜けがある。
- **内容**: ユーザー指摘（2026-08-23、「`test_mroz_cluster_matches_r_glm`
  でz値やp値、他のパラメータの検証が抜けていないか」）を受けてフィクスチャ
  JSONの実際の中身を確認。**mroz実データのクラスターエントリ
  （`logit.json`の`mroz.cluster`・`logit_crosscheck.json`の
  `wooldridge.mroz.cluster.r`）にはstatsmodels側・R側どちらも
  `coef`/`se`以外の`z_stats`/`p_values`/`conf_int`/`log_likelihood`/`aic`/
  `bic`/`lr_statistic`/`lr_p_value`/`pseudo_r_squared`/`margeff`が
  既に生成・記録済み**（`_check_result`が使う完全なフィールドセットと
  同一）であることを確認した。にもかかわらずテストは`coef`/`se`の
  2項目しか検証していない。`city`は正当な実カテゴリ変数であり、
  synthetic疑似グループのような「統計的意味が無いので動作確認だけで
  十分」という理由付けは当てはまらない。
- **追記（2026-08-24、`test_probit_fixtures.py`解説時の再確認で判明・
  当初の記載を訂正）**: 当初はsynthetic疑似クラスタ（`baseline`/
  `cluster_imbalanced`/`cluster_g2`）のフィクスチャエントリにも同様に
  フルの統計量が存在すると記載していたが、**statsmodels側フィクスチャ
  （`logit.json`/`probit.json`）を実際に確認したところ、synthetic疑似
  クラスタのエントリは`coef`/`se`/`_meta`のみで、フルの統計量は
  生成されていなかった**（`test_wls_fixtures.py`解説時に確認した
  「疑似グループは動作確認用に留める」という既存方針とテストが実際に
  一致している）。一方**Rクロスチェック側フィクスチャ
  （`logit_crosscheck.json`/`probit_crosscheck.json`）はsynthetic疑似
  クラスタでもフルの統計量を生成していた**ため、生成スクリプト間で
  「疑似クラスタでどこまで統計量を生成するか」の方針が食い違っている
  （Rクロスチェック側の生成スクリプトが、後から`_check_result`と
  同じ関数を疑似クラスタ用にも流用した結果と推測される）。この
  生成スクリプト間の不統一自体は実害が無い（無駄に多く生成している
  だけで欠落ではない）ため、この項目のタイトル・対象からは
  synthetic疑似クラスタ関連の指摘を除外し、**mroz実データのクラスター
  ケースのみ**に絞った。
- **Claudeの所感**: `test_mroz_cluster_matches_statsmodels`/
  `test_mroz_cluster_matches_r_glm`（Logit・Probit計4テスト）は
  `_check_result`ベースの完全な検証に切り替える価値が高い。データは
  既に存在するためフィクスチャ再生成は不要で、テストコード側の変更
  のみで対応可能。synthetic疑似クラスタ側は現状のフィクスチャ生成物と
  テストの検証範囲が一致しているため対応不要と判断する。
- **追記（2026-08-24、ユーザー提案「synthetic疑似クラスタもlogit.json/
  probit.json側に全統計量を追加し、他cov_typeと同様に検証すべきでは」を
  受けて実機検証）**: `cov_type`を`classical`→`cluster`に変えて実際に
  比較したところ、`params`・`log_likelihood`・`aic`は完全に同じ値のまま
  だった（`se`・`z_stats`等のみ変化）。これは統計的に当然の性質で、
  `cov_type`は「係数をどう推定するか」ではなく「推定済みの係数の標準誤差を
  どう計算するか」のみを決めるオプションのため、**`se`に依存しない
  統計量（`params`本体・`log_likelihood`/`aic`/`bic`/`lr_statistic`/
  `lr_p_value`/`pseudo_r_squared`/`pred_table`/限界効果の`dydx`）は
  cov_typeによらず不変**（他cov_typeで既に検証済みの値と同じものを
  再確認するだけで新しいバグ検出力はほぼ無い）。一方**`se`に依存する
  統計量（`z_stats`/`p_values`/`conf_int`・限界効果の`std_err`/`z`/
  `p_value`/`conf_low`/`conf_high`）はcov_type固有の値になるため、
  synthetic疑似クラスタであっても追加検証する価値がある**（`se`から
  検定統計量への変換にcov_type固有のバグがあるケースを拾える）。
  よってsynthetic疑似クラスタについては、フィクスチャ自体は既存の`run()`
  関数で全統計量を一括生成しつつ（`se`非依存分だけ絞り込む特別扱いは
  実装コストに見合わない）、**テスト側は`z_stats`/`p_values`/`conf_int`
  （＋限界効果）に絞って検証を追加する**のが効率的だと判断する。
- **気づいた経緯**: 2026-08-24、`tests/nonlinear/test_logit_crosscheck.py`解説後の
  ユーザー指摘、`tests/nonlinear/test_probit_fixtures.py`解説時にフィクスチャの
  実際の中身を再確認し記載を訂正、さらにユーザー提案を受けた実機検証で
  「`se`非依存の統計量は再検証不要・`se`依存の統計量のみ追加検証すべき」
  という基準を追記。
- **状態**: 未対応（着手要否はユーザー判断待ち。対象は(1)mroz実データ版
  4テスト〔Logit fixtures/crosscheck・Probit fixtures/crosscheck〕は
  `_check_result`ベースの完全な検証に、(2)synthetic疑似クラスタ版
  （8テスト）は`z_stats`/`p_values`/`conf_int`〔＋限界効果〕のみの
  追加検証に、それぞれ切り替える）

### 44. 項目41が`tests/nonlinear/test_probit_fixtures.py`にも同様に該当する（一括注記）

- **対象**: [tests/nonlinear/test_probit_fixtures.py](../../../tests/nonlinear/test_probit_fixtures.py)
  全体（`test_matches_statsmodels`はmethod既定固定・`test_method_matches_
  statsmodels`はcov_type="classical"固定という同じ構造）
- **内容**: `tests/nonlinear/test_probit_fixtures.py`解説時、コード部分が
  `test_logit_fixtures.py`と完全に同一（項目95・96参照）であることを
  確認したため、項目41（`method`〔bfgs/lbfgs〕と`cov_type`・シナリオ・
  クラスターの組み合わせが未検証）がそのまま該当する。
- **Claudeの所感**: 対応する場合は項目41と同じ方針（全組み合わせでは
  なく代表的な組み合わせに絞って追加）をLogit/Probit両方にまとめて
  適用するのが効率的。
- **気づいた経緯**: 2026-08-24、`tests/nonlinear/test_probit_fixtures.py`解説時に
  確認。
- **状態**: 未対応（項目41への追記の代わりにこの1項目に集約、着手要否は
  ユーザー判断待ち）

### 45. Logit/Probitの実データクラスターロバストSEテストが`mroz`の`city`（G=2）のみで、より現実的な多数クラスタ（数十件規模）での実データ検証が無い——`apple`データセット（`state`、G=49）の採用が決定済み

- **対象**: [tests/nonlinear/test_logit_fixtures.py:287-299](../../../tests/nonlinear/test_logit_fixtures.py#L287-L299)・
  [tests/nonlinear/test_probit_fixtures.py:276-288](../../../tests/nonlinear/test_probit_fixtures.py#L276-L288)・
  両crosscheckファイルの`test_mroz_cluster_matches_*`（`cluster_col="city"`、
  G=2）
- **内容**: ユーザー指摘（2026-08-24、「mrozデータのcity変数ってクラスター
  の実データ検証としてはあまりよくないかもしれない（2値なため）」）。
  `city`はG=2ちょうどのため、統計的には既存の合成データ境界値テスト
  （`test_cluster_g2_matches_statsmodels`等）を実データでなぞっているに
  過ぎず、「実データで多数の小〜中規模グループが自然に存在する」という
  `testing-policy.md`「実データでのグループ列も検証する」の趣旨を
  十分満たさない。`wooldridge`パッケージ内を調査し、`apple`データセット
  （n=660、`state`＝居住州で49州、グループサイズ1〜66・平均13.5の不均衡な
  分布）を候補として提示し、**ユーザーが採用を決定**（2026-08-24）。
  `y`は既存列ではなく`ecolbs > 0`（エコラベル付きりんごを購入したか）
  という派生変数が必要（`ecolbs`という購入量列からの二値化。Wooldridge
  教科書に載っている定番の二値変数ではないが、経済学的には自然な
  「購入するか否か」の意思決定モデル）。
- **Claudeの所感（ユーザー追加コメント含む）**: 実装上の制約として、
  `testing-policy.md`「フィクスチャ化」の方針でWooldridgeデータセットは
  CSV固定の対象外（再配布ライセンス未確認のため`load_wooldridge_dataset`
  経由で都度ロードする方針）のため、`ecolbs > 0`という派生列を
  `benchmark/`側であらかじめCSVに焼き込むことができない。**列追加は
  テスト実行時（`tests/`側の`load_wooldridge_dataset("apple")`呼び出し後、
  `.with_columns()`等でその場で派生列を作る）で行う必要がある**
  （WLSの401ksubsで年齢分位ビンを`_add_age_bin`により都度生成していた
  パターン、`benchmark/linear/fixtures/generate_wls_fixtures.py`参照、と
  同じ設計になる見込み）。フィクスチャ生成スクリプト
  （`generate_logit_fixtures.py`等）側でも同様にその場で派生列を作って
  from `run()`に渡す必要がある。
- **気づいた経緯**: 2026-08-24、`tests/nonlinear/test_probit_crosscheck.py`解説後の
  ユーザー指摘・データセット調査・ユーザーによる採用決定。
- **状態**: 採用決定・実装は未着手（この場では記録のみ。実施時は
  Logit/Probit両方の`test_<method>_fixtures.py`/
  `test_<method>_crosscheck.py`・対応する`generate_*_fixtures.py`/
  `generate_*_crosscheck_fixtures.py`が対象になる見込み）

### 48. `test_missing_column_raises`が`x_exog`のみを検証しており、`y`/`x_endog`/`instruments`側の存在しない列名が未検証

- **対象**: [tests/test_iv.py:639-647](../../../tests/test_iv.py#L639-L647)
  （`x_exog=["nonexistent"]`のみ）。対比として
  [tests/test_iv.py:650-667](../../../tests/test_iv.py#L650-L667)の
  `test_null_values_raise`/`test_non_numeric_dtype_raises`は
  `@pytest.mark.parametrize("bad_col", ["y", "x1", "endog1", "z1"])`で
  4つの役割すべてを網羅済み。
- **内容**: ユーザー指摘（2026-08-30、「x_exogが対象だが、y, x_endog,
  instrumentsでの存在しない列のチェックがない気がする」）を受けて実機
  確認した。`y`/`x_endog`/`instruments`いずれに存在しない列名を渡しても
  `ValidationError: column 'does_not_exist' does not exist in the data`
  になることを確認済み（実装側は正しく動作している、純粋なテスト
  カバレッジの抜け）。`test_null_values_raise`等は既に4役割を
  parametrizeしているのに対し、`test_missing_column_raises`だけ
  `x_exog`単独のままという非対称。
- **Claudeの所感**: `test_null_values_raise`と同じ
  `@pytest.mark.parametrize("bad_col", ["y", "x1", "endog1", "z1"])`の
  形に揃えれば解消できる、実施しやすい部類。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時のユーザー指摘、
  実機検証で確認。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 49. `overid_statistic`/`wu_hausman_statistic`系のテストが「`None`でないこと」の確認のみで、値の妥当性（正であること等）を確認していない

- **対象**: [tests/test_iv.py:288-333](../../../tests/test_iv.py#L288-L333)
  （`test_overid_statistic_present_when_over_identified`・
  `test_overid_statistic_present_for_gmm_hansen_j`・
  `test_wu_hausman_is_not_none_for_2sls`等、すべて`is not None`のみ）
- **内容**: ユーザー指摘（2026-08-30、「`test_overid_statistic_present_
  for_gmm_hansen_j`だが、他の検証では`!=0`や`>0`を使っているので統一した
  ほうが良いか？」）を受けて確認。`test_weak_instrument_f_statistics_
  keyed_by_endog_name`は`> 0.0`、`test_cluster_g2_boundary_succeeds_
  when_x_exog_is_empty`は`!= 0.0`を使うのに対し、過剰識別検定・
  Wu-Hausman検定系は一貫して`is not None`のみで、統計量自体が退化値
  （厳密に`0.0`）になっていないかは確認していない。Sargan/Hansen J・
  Wu-Hausman統計量はいずれもカイ二乗/F分布に従うワルド型検定統計量で
  理論上`0`以上、実務的にはほぼ確実に`>0`になるはずの値であり、
  `engine/src/iv/CLAUDE.md`に記録されている過去の実バグ（Hansen J統計量を
  誤って`/n`していたバグ）のように「統計量が異常に小さい値になる」
  という失敗モードは`is not None`だけでは検出できない。
- **Claudeの所感**: `> 0.0`程度の軽い妥当性チェックを追加する価値はある
  （数値の正確性自体は`test_iv_fixtures.py`/`test_iv_gmm_fixtures.py`の
  役割だが、「退化した0近傍の値でないこと」という安価なサニティチェックは
  構造テスト側で足しても役割分担を壊さないと考える）。p値についても
  `(0, 1)`の範囲チェックを追加できる。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 50. `include_intercept=False`の数値照合（linearmodelsとの一致確認）がIVにだけ存在しない

- **対象**: `tests/test_iv_fixtures.py`・`tests/test_iv_gmm_fixtures.py`・
  `tests/test_iv_crosscheck.py`全体（`grep`で`include_intercept`が
  1件もヒットしない）。対比として`tests/test_ols.py`
  （`test_include_intercept_false_matches_statsmodels`・
  `test_include_intercept_false_matches_statsmodels_robust_cov_types`）、
  `tests/test_wls_fixtures.py`・`tests/test_logit_fixtures.py`・
  `tests/test_probit_fixtures.py`（いずれも同名の
  `test_include_intercept_false_matches_statsmodels`が存在）。
- **内容**: ユーザー指摘（2026-08-30、「OLS, WLS, Logit, Probitで
  intercept=falseの数値一致のテストはなかった気がする」）を受けて
  `grep`で確認したところ、**ユーザーの記憶とは逆に、OLS/WLS/Logit/Probit
  にはすべて`include_intercept=False`の数値照合テストが既に存在した**
  （この点はユーザーの記憶違いだったため訂正する）。一方、**IVは
  `tests/test_iv.py`（構造テスト）に`test_include_intercept_false_omits_
  const`があるのみで、`test_iv_fixtures.py`/`test_iv_gmm_fixtures.py`
  （linearmodelsとの数値照合）/`test_iv_crosscheck.py`のいずれにも
  `include_intercept=False`のケースが1つも存在しない**——2SLS・GMM
  どちらの`method`についても、`include_intercept=False`が実際に
  linearmodelsと一致した値を返すことは一度も検証されていない。
- **Claudeの所感**: これは他手法とIVの間の実在する非対称であり、
  IVが唯一の抜けなので優先度は他項目より高いと考える。特に
  `first_stage()`が絡む分`include_intercept`の伝播経路がOLS単体より
  複雑（項目51参照）なため、数値照合による裏付けの価値は高い。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時のユーザー指摘、
  `grep`で確認。
- **状態**: 未対応（着手要否はユーザー判断待ち、優先度は本ファイル中で
  比較的高い）

### 51. `include_intercept=False`時に`first_stage()`側にも切片が正しく伝播しているかが未検証

- **対象**: [tests/test_iv.py:406-412](../../../tests/test_iv.py#L406-L412)
  （`test_include_intercept_false_omits_const`、トップレベルの
  `res.param_names`のみ確認、`res.first_stage()`は未確認）
- **内容**: ユーザー指摘（2026-08-30、「`first_stage`からは`const`が
  抜かれていないことや...第一段階回帰の`has_intercept=false`のままでは
  なく、`has_intercept=true`になっているかを確認したほうがよいかも
  しれない（実際はIVのオプションで`intercept=false`にしたら
  `first_stage`はどっちになる）」）を受けて実機確認した。
  `IVOptions(include_intercept=False)`でfitした結果、`first_stage()
  ["endog1"].param_names`は`['x1', 'z1', 'z2']`（`const`を含まない）、
  `r_squared`は`OLS(y="endog1", x=["x1","z1","z2"],
  options=OLSOptions(include_intercept=False))`の直接fitと**完全一致**
  （`0.2748969914858259`）した——つまり**`first_stage()`は正しく
  `include_intercept=False`を継承している**ことを確認した
  （`engine/src/iv/CLAUDE.md`に記録されているG=2バグ修正
  〔`without_baked_in_intercept`が`input.has_intercept()`をそのまま
  第一段階に渡す設計〕が効いている、実装は正しい）。
- **Claudeの所感**: 実装は正しく動作していたが、これをロックインする
  テストは存在しない——過去に類似の`has_intercept`取り違えバグが
  実際に発生した箇所（G=2境界バグ、`refactoring-candidates.md`ではなく
  `engine/src/iv/CLAUDE.md`参照）だけに、リグレッション防止の価値が
  高いと考える。項目50（linearmodels数値照合の欠落）とあわせて、
  `include_intercept=False`×`first_stage()`の組み合わせをテストに
  追加する価値がある。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時のユーザー指摘、
  実機検証で確認（現状は正しく動作していることを確認済み）。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目50と合わせて検討）

### 53. `cov_type="cluster"`の大文字小文字非依存性（`"CLUSTER"`等）がリポジトリ全体で未検証

- **対象**: `tests/test_iv.py`の`test_cov_type_is_case_insensitive`
  （`grep`で`"CLUSTER"`/`"Cluster"`が0件）。`tests/test_ols.py`等、
  他手法の同名テストも同様に`cluster`の大文字小文字バリエーションを
  含まない。
- **内容**: `tests/test_iv.py`解説時（項目4、`refactoring-candidates-3.md`）
  の調査中に発見。`cov_type`の大文字小文字非依存性テストは`classical`/
  `hc0`〜`hc3`/`hac`/`nonrobust`は網羅しているが、`cluster`だけは
  常に小文字の`"cluster"`のみで使われており、`"CLUSTER"`のような表記が
  正しく`"cluster"`に正規化されることは一度も検証されていない。
  `cluster_col`付きデータセットが必要という事情（`refactoring-
  candidates-3.md`項目4参照）から、他のcov_typeと同じ
  parametrize済みテストに単純に含められなかったための漏れと推測される。
- **Claudeの所感**: 実装のパース関数が他のcov_typeと同じ正規化経路を
  通っているなら実害は低いと考えられるが、`cluster`は`cluster_col`の
  存在確認等の追加分岐があるため、他のcov_typeと全く同じコードパスとは
  限らない。専用の1テストを足す程度の軽い対応で埋められる。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時の調査中に発見。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 54. 過剰識別検定（Sargan/Hansen J）が実際に棄却される（p値が小さい）シナリオが1つも存在しない

- **対象**: `tests/fixtures/benchmarks/iv.json`・`tests/fixtures/benchmarks/
  iv_gmm.json`全体（`sargan_p_value`/`hansen_j_p_value`の最小値を
  実機で確認したところ、2SLS側は`card`実データの`0.103`が最小、GMM側も
  同水準で、5%はおろか10%水準でも棄却されるケースが1つも無い）
- **内容**: ユーザー指摘（2026-08-30、「シナリオとして過剰識別検定に
  引っかかるシナリオってあるか？ないなら作ったほうが良いか？」）を受けて
  全フィクスチャの`sargan_p_value`/`hansen_j_p_value`を機械的に走査し
  確認した。ユーザーの見立て通り、現在の9つの合成データシナリオ・実データ
  （`card`）のいずれも「操作変数が妥当」という帰無仮説を棄却しない
  （p値が小さくとも0.10程度）。これは各シナリオのDGP
  （`benchmark/iv/datasets.py`）が「操作変数は真に外生」という前提で
  設計されているため当然の結果ではあるが、**「過剰識別検定という機能
  自体が、実際に統計的有意な結果を返せることは一度も確認されていない」**
  という意味でのカバレッジの穴ではある。
- **Claudeの所感**: 妥当な指摘だと考える。操作変数の1本を意図的に構造
  誤差項と相関させた「無効な操作変数を含む」DGPシナリオを追加すれば、
  Sargan/Hansen J検定が実際に小さいp値を返すことを確認でき、検定の
  実装（`two_sls.rs`/`gmm.rs`）が「棄却すべき場面で正しく棄却する」側の
  挙動まで検証できる。ただし新規シナリオの追加は`benchmark/iv/
  datasets.py`・`generate_iv_fixtures.py`・`generate_iv_gmm_fixtures.py`・
  `generate_iv_crosscheck_fixtures.py`全てのフィクスチャ再生成を伴う
  ため、着手時期はユーザー判断が必要。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_fixtures.py`解説時の
  ユーザー指摘、フィクスチャJSONの実機走査で確認。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 55. `first_stage()`が返す数値が、統計モデル・R・engine単体テストのいずれとも一度も照合されていない——過去に実際に発生したバグの実例を踏まえると優先度は高いと考える

- **対象**: `tests/test_iv.py`・`tests/test_iv_fixtures.py`・
  `tests/test_iv_gmm_fixtures.py`・`tests/test_iv_crosscheck.py`全体
  （`grep -n "first_stage"`でヒットする箇所は全て構造確認
  〔`test_iv.py::test_first_stage_structure`、`param_names`の集合一致の
  みで数値は見ない〕か、「OLSのテストで検証済みだから省略する」という
  docstring上の説明のみ）
- **内容**: ユーザー指摘（2026-08-30、「第一段階の結果の検証がされて
  いないのでは？」）を受けて確認したところ、指摘の通り**`first_stage()`
  が返す`OLSResults`の実際の数値（`params`/`r_squared`/`std_errors`等）を
  外部リファレンス（statsmodels/linearmodels/R）と照合するテストは
  1つも存在しない**ことを確認した。`test_iv_fixtures.py`のモジュール
  docstringは「`first_stage()`は通常のOLS回帰の結果をそのまま返すだけ
  で、`test_ols_fixtures.py`が既にOLSの数値一致を検証済みのため」と
  省略理由を説明しているが、この理屈には**穴がある**。`test_ols_
  fixtures.py`が検証しているのは`OlsEstimator::fit`という計算ロジック
  自体の正しさであり、**`first_stage()`がその計算ロジックに正しい
  設計行列（`x_exog ++ instruments`、正しい`include_intercept`）を
  渡しているか**という**IV固有の配線（グルー）コードの正しさ**は
  別問題である。実際、`engine/src/iv/CLAUDE.md`に記録されている
  過去のバグ（`compute_first_stage`が`has_intercept`を常に`false`で
  呼んでいたことによる`k_constant`取り違え）は、まさにこの配線コードの
  バグであり、**`first_stage().r_squared`が実際に静かに間違った値
  （フィクスチャ作成中の手動比較で発覚: `0.430`ではなく正しくは
  `0.338`）を返していた**という実例がある。このバグは自動テストでは
  なくベンチマーク作成中の手動比較で偶然発見されたものであり、もし
  同種のバグが再発しても、現状のテストスイートには検知する手段が
  無い。
- **Claudeの所感**: ユーザー指摘に強く同意する。過去に実際に発生した
  バグの実例がある箇所だけに、他の項目より優先度を高く設定すべきと
  考える。対応案としては、`test_iv_fixtures.py`または`test_iv_
  crosscheck.py`に「`first_stage()['endog1']`の`params`/`r_squared`等が、
  同じデータで直接`OLS(y=x_endog名, x=x_exog+instruments)`をfitした
  結果と一致する」という比較テストを追加する（新規フィクスチャ生成は
  不要、既存の`OLSResults`同士の比較で足りる）のが最も手軽。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_fixtures.py`解説時の
  ユーザー指摘、`grep`で確認。
- **状態**: 未対応（**優先度高**、着手要否はユーザー判断待ち）

### 56. `test_multi_endog_matches_linearmodels`が`cov_type`のみをparametrizeしており、DGPシナリオ軸（弱操作変数・不均一分散等）との組み合わせが無い

- **対象**: [tests/test_iv_fixtures.py:247-268](../../../tests/test_iv_fixtures.py#L247-L268)
  （`x_endog=["endog1", "endog2"]`固定で`iv_baseline_multi_endog.csv`
  1つのみ、`cov_type`のみparametrize）、
  [tests/test_iv_gmm_fixtures.py:231-256](../../../tests/test_iv_gmm_fixtures.py#L231-L256)
  （GMM版も同様に`cov_type`のみ）
  対比: [tests/test_iv_fixtures.py:158-174](../../../tests/test_iv_fixtures.py#L158-L174)
  （`test_matches_linearmodels`、単一内生変数側は`scenario`×`cov_type`の
  直積で9シナリオを網羅）
- **内容**: ユーザー指摘（2026-08-30、「`test_multi_endog_matches_
  linearmodels`は`cov_type`だけでなく、シナリオの組み合わせも検証した
  ほうが良いと思うがどうか？」）を受けて確認した。単一内生変数の
  `test_matches_linearmodels`は9シナリオ（弱操作変数・小標本・
  不均一分散・自己相関・多重共線性等）×`cov_type`を網羅しているのに
  対し、複数内生変数（`x_endog`が2つ）のケースは`iv_baseline_multi_
  endog.csv`という単一の「素直な」データセットでしか検証されておらず、
  「複数内生変数」と「弱操作変数」・「不均一分散」等の**組み合わせ**は
  一度も検証されていない。特に弱操作変数×複数内生変数は、
  `weak_instrument_f_statistics`が内生変数ごとの辞書であることを踏まえると
  実務上重要な組み合わせだと考えられる。
- **Claudeの所感**: 妥当な指摘だが、実施コストは相応に大きい。単一
  内生変数の9シナリオと同じ密度で複数内生変数版を用意すると、DGP・
  固定CSV・フィクスチャ生成の全てを9パターン分新たに用意する必要が
  あり、規模が大きい。全シナリオではなく「弱操作変数」「不均一分散」
  等、複数内生変数との相互作用が特に懸念される2〜3シナリオに絞って
  追加するのが費用対効果が良いと考える。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_fixtures.py`解説時の
  ユーザー指摘。
- **状態**: 未対応（着手要否・対象シナリオの絞り込みはユーザー判断待ち）

### 57. 多重共線性のテストが`x_exog`内部のみで、`instruments`間・`instruments`×`x_exog`・`x_endog`×`x_exog`（第二段階）の組み合わせが未検証

- **対象**: [benchmark/iv/datasets.py:23-27](../../../benchmark/iv/datasets.py#L23-L27)
  （「`moderate_multicollinearity`/`high_condition_number`/
  `perfect_multicollinearity`/`scale_variance`は`x_exog`側の列間
  relationshipを操作する設計...instrumentsやx_endogには適用しない」、
  設計上明記された制限）
- **内容**: ユーザー指摘（2026-08-30、「多重共線性に関してx_exogと
  x_endog, instrumentsとx_exog, instrumentsとx_endogのすべての組み合わせ
  を考える必要があるのでは？」）を受けて確認した。現在の多重共線性系
  シナリオは設計文書に明記されている通り一貫して`x_exog`列間のみを
  操作しており、これは意図的な制限（OLSの`generate_linear_dataset`と
  同じ発想の流用）であって見落としではない。ただし統計的には、IVは
  `x_exog`だけでなく2段階の設計行列を持つため、多重共線性が問題になる
  経路は少なくとも2種類ある。
  1. **第一段階の設計行列（`x_exog ++ instruments`）の特異性**:
     `instruments`同士が強く相関している、または`instruments`が
     `x_exog`と強く相関しているケース。現状`tests/iv/test_iv_validation.py`の
     `test_perfect_multicollinearity_raises_computation_error`
     （旧 `test_singular_first_stage_design_matrix_raises_computation_error`。
     `refactoring-candidates-2.md` 項目54 で固定 CSV 版へ一本化）
     は`x_exog`内部の完全共線性のみで再現しており、`instruments`側の
     共線性は未検証（ただし第一段階の設計行列としては同じ
     `x_exog ++ instruments`の列空間に属するため、実装上のコードパスは
     `x_exog`内部の共線性と共通の可能性が高い）。
  2. **第二段階の設計行列（`x_exog` + `X̂`〔内生変数の予測値〕）の
     特異性**: `x_exog`自体は健全でも、内生変数の予測値`X̂`がたまたま
     `x_exog`と強い共線性を持つケース。これは第一段階とは異なる
     コードパス（`second_stage_input`、`engine/src/iv/CLAUDE.md`
     参照）であり、現状は完全に未検証。
- **Claudeの所感**: (1)は実装上のコードパスがおそらく共通のため優先度は
  低いが、(2)は第一段階とは独立した失敗経路であり、`engine/src/iv/
  CLAUDE.md`に記録されている過去のバグ修正がこの`second_stage_input`
  周りだったことを踏まえると、検証する価値がある。ただし「内生変数の
  予測値がたまたま`x_exog`と共線的になる」DGPを意図的に構築するのは、
  単純な列操作（`x2 = 2*x1`のような）より設計が難しい（第一段階の
  係数を逆算する必要がある）。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_fixtures.py`解説時の
  ユーザー指摘。
- **状態**: 未対応（優先度は(2)のみ中、(1)は低。着手要否はユーザー
  判断待ち）

### 58. クラスターロバストSEのテスト群（`test_cluster_matches_linearmodels`等）が`coef`/`se`のみの検証で、他の統計量への影響が未確認

- **対象**: [tests/test_iv_fixtures.py:177-244](../../../tests/test_iv_fixtures.py#L177-L244)
  （`test_cluster_matches_linearmodels`・
  `test_cluster_imbalanced_matches_linearmodels`・
  `test_cluster_g2_matches_linearmodels`、いずれも`coef`/`se`のみ
  `_assert_dict_close`で検証、`_check_result`は使わない）
- **内容**: ユーザー指摘（2026-08-30、「`test_cluster_matches_
  linearmodels`はフィクスチャ自体に他の統計量も載せてすべて検証する
  （t値とp値も影響を受ける＆その他の統計量も一致をみることで保守的に
  したい）。imbalanced/g2も同様」）。Logit/Probitの解説で見た
  coverage項目43・45と同種の論点がIVにもそのまま当てはまる。`se`が
  変われば`t_stats`/`p_values`/`conf_int`も連動して変わるはずだが、
  現状はそれらを検証していない。
- **Claudeの所感**: 妥当な指摘。ただしLogit/Probitの項目43で整理した
  通り、`se`に依存しない統計量（`r_squared`・`f_statistic`本体等、
  クラスター化によって値が変わらないもの）まで一律に追加するのは
  冗長なので、`se`から連動して変わる統計量（`t_stats`/`p_values`/
  `conf_int`、必要なら`f_p_value`）に絞って`_check_result`相当の
  検証に寄せるのが効率的だと考える。フィクスチャ生成
  （`generate_iv_fixtures.py`）側で`coef`/`se`しか記録していない
  ため、対応にはフィクスチャの再生成が必要。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_fixtures.py`解説時の
  ユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、フィクスチャ再生成を
  伴う）

### 59. `weight_type`×`cov_type`の「両方とも非デフォルトで異なる種類」という組み合わせが`kernel`×`hac`以外に無い

- **対象**: [tests/test_iv_gmm_fixtures.py:259-278](../../../tests/test_iv_gmm_fixtures.py#L259-L278)
  （`test_kernel_hac_matches_linearmodels`が唯一の該当例）
- **内容**: ユーザー指摘（2026-08-30、「問題が起きやすい組み合わせだけ
  別途やっておくと検出力が上がりそうなのだが」）を受けて確認。詳細な
  調査経緯・実装の設計上のリスク評価は`refactoring-candidates-3.md`
  項目23参照（実装は`weight_type`/`cov_type`が一致する場合・しない場合を
  分岐させない一般形サンドイッチのため構造的リスクは低いと判断したが、
  それを裏付けるテストが`kernel`×`hac`1点のみというのは心もとない）。
- **Claudeの所感**: `weight_type="cluster"`×`cov_type="hac"`
  （またはその逆の`weight_type="kernel"`×`cov_type="cluster"`）を
  もう1〜2パターン`baseline`シナリオで追加するのが、全組み合わせ
  網羅（24通り）よりも費用対効果が良いと考える。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_gmm_fixtures.py`解説時の
  ユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 60. GMMで`include_intercept=False`の数値照合（linearmodels）・`first_stage()`への伝播確認が無い（2SLS版項目50・51と同種、GMMも同じ配線コードを共有するため同程度に重要）

- **対象**: `tests/test_iv_gmm_fixtures.py`全体（`grep`で
  `include_intercept`が0件）
- **内容**: ユーザー指摘（2026-08-30、「GMMのconstがfalseの場合が無い。
  GMMにfirst_stageの概念がないから問題にならない？」）を受けて実機
  確認したところ、GMMも2SLSと同じ`compute_first_stage`配線コードを
  共有しており`first_stage()`を持つ（`include_intercept=False`でも
  正しく動作することは確認済み、詳細は`refactoring-candidates-3.md`
  項目24参照）。項目50・51の2SLS版と同じ理由でGMM側にも数値照合
  テストが無い。
- **Claudeの所感**: 項目50・51と同時に対応するのが効率的。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_gmm_fixtures.py`解説時の
  ユーザー指摘、実機検証で確認。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目50・51と合わせて
  検討）

### 61. `test_iv_gmm_fixtures.py`（GMMのlinearmodels主リファレンス照合）に実データセット（Wooldridge `card`）での検証が無い——ドキュメント上も明示的に決定された事項ではない

- **対象**: `tests/test_iv_gmm_fixtures.py`全体（`grep`で`card`/
  `wooldridge`が0件）。対比: `iv-spec.md`4章（当時: 220-227行目）
  （「5.5 実データセット」節、`test_iv_fixtures.py`〔linearmodels〕・
  `test_iv_crosscheck.py`〔ivreg〕の両方でCard実データをクロスチェック
  すると明記されているが、GMMについては「5.3節の方針によりRクロス
  チェック省略のため実データセットでの**Rクロスチェック**も対象外」
  としか書かれておらず、`linearmodels`側〔Python、`test_iv_gmm_
  fixtures.py`〕でのGMM実データ検証を省略してよいかどうかは明記
  されていない）
- **内容**: ユーザー指摘（2026-08-31、「`test_iv_gmm_fixtures.py`では
  実データでの検証が抜けているのでは？」）を受けて確認したところ、
  指摘の通り`test_iv_gmm_fixtures.py`には実データ検証が1つも無い。
  重要なのは、これが`refactoring-candidates-3.md`項目26（`test_iv_
  crosscheck.py`のGMM省略）とは**性質が異なる**という点——項目26は
  「Rクロスチェックの省略」という明確に文書化された決定だが、本項目は
  「`linearmodels`主リファレンスでのGMM実データ検証」の話であり、
  当時の設計ドキュメント5.5節の文言（「Rクロスチェックも対象外」）を素直に
  読むと、Rクロスチェックの省略についてのみ言及しており、`linearmodels`
  側（Python）の実データ検証を省略してよいという決定までは読み取れない。
  `testing-policy.md`「テスト用データセット」2.は「実データセットでの
  検証」を各推定手法に一律に求めており、GMMも例外という明記はない。
- **Claudeの所感**: CLAUDE.md 14章が求める「既存ドキュメント・issueの
  記述と、実装時に判明した事実が食い違う」に近いケースだと考える。
  意図的に省略したのか、単に`ivreg`がGMM非対応という制約から連想して
  Python側の実データ検証まで芋づる式に見送られてしまったのかが、
  ドキュメントからは判別できない。`GmmEstimator`はWu-Hausman検定を
  実装しない・`first_stage()`は共有、という制約はあるが、`params`/
  `std_errors`/`weak_instrument_f_statistics`/`overid_statistic`
  （Hansen J）等、GMMでも実データで検証する価値のある統計量は多い
  ため、追加する方向を推奨する。
- **気づいた経緯**: 2026-08-31、`tests/test_iv_crosscheck.py`解説時の
  ユーザー指摘、当時の設計ドキュメント5.5節の文言を精査して確認。
- **状態**: 未対応（**要ユーザー判断**: 意図的な省略だったか確認した
  上で、追加するならフィクスチャ生成〔`generate_iv_gmm_fixtures.py`〕を
  伴う）

### 62. `censoring_fit_check()`の上側打ち切り（`"upper"`カテゴリ）が一度も検証されていない

- **対象**: [tests/nonlinear/test_tobit.py:168-186](../../../tests/nonlinear/test_tobit.py#L168-L186)
  （`test_censoring_fit_check_structure`・`test_censoring_fit_check_
  omits_upper_when_upper_is_none`、いずれも既定〔左打ち切りのみ〕の
  `censored_dataset`しか使わない）と対比した
  [tests/nonlinear/test_tobit.py:398-412](../../../tests/nonlinear/test_tobit.py#L398-L412)
  （`test_supports_right_censoring_only`、`upper=5.0`のデータはあるが
  `censoring_fit_check()`を呼んでいない）
- **内容**: ユーザー指摘（2026-08-31、「`test_censoring_fit_check_
  structure`で上側打ち切りがチェックできていないのが気になる」）を
  受けて確認した。指摘の通り、`censoring_fit_check()`が返しうる
  3カテゴリ（`"lower"`/`"uncensored"`/`"upper"`）のうち`"upper"`は
  一度もテストされていない。右打ち切りのみのデータ（`upper=5.0`）は
  `test_supports_right_censoring_only`で既に用意されているが、
  そちらは`res.lower`/`res.upper`の構造確認のみで`censoring_fit_
  check()`自体を呼んでいない。
- **Claudeの所感**: `test_supports_right_censoring_only`のデータで
  `censoring_fit_check()`を呼び、`{"upper", "uncensored"}`が返る
  ことを確認するテストを追加すれば埋められる、実施しやすい部類。
  理想的には両側打ち切り（`lower`/`upper`両方指定）のケースで
  3カテゴリ全てが返ることを確認するテストもあるとさらに良い。
- **気づいた経緯**: 2026-08-31、`tests/nonlinear/test_tobit.py`解説時のユーザー
  指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 63. 項目32・39・40がTobitにも同様に該当する（一括注記）

- **対象**: [tests/nonlinear/test_tobit.py](../../../tests/nonlinear/test_tobit.py)全体
- **内容**: ユーザー指摘（2026-08-31、「yの列が存在しない場合の検証が
  されていない」・「`test_null_values_raise`でyは検証されているが
  xの場合がない」・「`test_non_numeric_dtype_raises`もxの場合がない」・
  「`test_marginal_effects_confidence_level_out_of_range_raises`は
  `1.5`のみを検証している」）を受けて確認した。個別に項目を複製すると
  項目数が倍増するため、該当箇所を1項目にまとめて記録する。
  - **項目32**（OLSの`y`列欠落専用テストが無い）: Tobitの`test_
    missing_column_raises`も`x=["does_not_exist"]`のみで、`y="does_
    not_exist"`のテストは無く該当する。
  - **項目32の追記**（`x`側のnull値専用テストが無いのはリスク低い
    という判断）: Tobitの`test_null_values_raise`も`y`列のみ`None`に
    しており`x`側は未検証だが、`y`/`x`とも`extract_f64_column`の
    同じ分岐を通ると考えられるため、項目32・40と同じ理由でリスクは
    低いと判断する。`test_non_numeric_dtype_raises`（`x`側非検証）
    も同じ理由が当てはまると考える。
  - **項目39**（`marginal_effects`の`confidence_level`境界値検証が
    `1.5`のみ）: Tobitの`test_marginal_effects_confidence_level_out_
    of_range_raises`も`1.5`のみで、`fit()`側の`test_invalid_
    confidence_level_raises`（`[1.5, 0.0, -0.1]`をparametrize済み）
    との非対称が同様に存在する。
- **Claudeの所感**: 対応する場合は、Logit/Probit/Tobit3手法をまとめて
  一度に対応するのが効率的（同じ`ValidationError`パス・同じ検証観点
  のため）。
- **気づいた経緯**: 2026-08-31、`tests/nonlinear/test_tobit.py`解説時のユーザー
  指摘。
- **状態**: 未対応（項目32・39・40と統合して対応するのが効率的、
  着手要否はユーザー判断待ち）。2026-09-23追記: 項目75の調査で
  `test_null_values_raise`/`test_non_finite_values_raise`の`x`側
  （2番目の箇条書き）は現状のコードで既に`x1`をカバーしており
  解消済みと判明（v0.6.0リリース時点、2026-09-06、Tobit実装時から
  既に対称に実装されていた）。`y`列欠落専用テスト（項目32相当）・
  `marginal_effects`の`confidence_level`境界値parametrize（項目39相当）は
  未対応のまま残っている。

### 64. HACの自動ラグ選択式（`L = floor(4*(n/100)^(2/9))`）がRust実装とPython実装（`benchmark/common/dgp.py`の`hac_auto_lag`）で一致することを直接検証するテストが無い（IV 2SLS/GMMは特に、Rust内の自己整合性チェックのみ）

- **対象**: [engine/src/linear/ols.rs:1601](../../../engine/src/linear/ols.rs#L1601)
  （`fit_computes_hac_std_errors_with_auto_lags`）・
  [engine/src/iv/two_sls.rs:1733](../../../engine/src/iv/two_sls.rs#L1733)
  （`fit_computes_hac_std_errors_with_auto_lags_matching_explicit_lags`）・
  [engine/src/iv/gmm.rs:2255](../../../engine/src/iv/gmm.rs#L2255)
  （`fit_with_kernel_weight_type_and_auto_lags_matches_explicit_lags_two`）、
  および`tests/linear/test_ols_api.py`・`test_wls_api.py`・`tests/iv/test_iv_api.py`の
  各`test_hac_auto_lags_runs_and_returns_finite_std_errors`
- **内容**: HAC自動ラグ選択式（`hac_lags=None`時に`L = floor(4*(n/100)^(2/9))`で
  自動計算）は、Rust側（`ols.rs`/`two_sls.rs`/`gmm.rs`各所の`resolve_hac_lags`、
  独立実装3箇所）とPython側（`benchmark/common/dgp.py`の`hac_auto_lag`、
  ベンチマーク・性能比較・R/linearmodelsクロスチェックのフィクスチャ生成が
  この1関数を単一の定義元として使う）の両方に存在するが、**この2つの独立実装が
  同じ値を返すことを直接比較する自動テストが無い**。
  - OLS側の`fit_computes_hac_std_errors_with_auto_lags`（`n=5`→`L=2`）は
    唯一、期待値がstatsmodelsに`maxlags=2`を明示指定して独立に計算・検算した
    値だとコメントに明記されており、実質的にクロス言語で直接検証された
    唯一の例（ただし`n=5`の1点のみ）。
  - IV 2SLS/GMM側の該当テストは、`lags=None`と`lags=Some(2)`という
    **Rust内の2つのパスの自己整合性**を確認しているだけで、
    Python側`hac_auto_lag(8)`の値と突き合わせてはいない。
  - `tests/iv/test_iv_reference.py`等の大規模フィクスチャ照合テスト
    （`cov_type="hac"`を含めて`rtol=1e-8`で一律検証、緩和分岐なしを確認済み）が
    実質的に間接検証として機能してはいる（`n=500`・`n=3`・実データ`card`の
    複数`n`で通過）。HACのSEはラグ数が変われば通常は明確に数値が変わるため、
    式が食い違ったまま複数の異なる`n`で偶然一致し続ける可能性は低いと考えられるが、
    **確率的な傍証であり証明ではない**。
  - `OLSResult`/`IVResult`（`engine_pybind/src/linear/ols.rs:136-160`等）は
    `cov_type`文字列のみをエコーバックし、実際に解決されたラグ数自体は
    結果オブジェクトのどこにも露出していないため、外部から直接確認する
    手段が現状無い（`refactoring-candidates.md`系ではなくAPI追加の話のため、
    別途Issue化を検討、`.claude/skills/refactor/SKILL.md`「観点5」参照）。
- **Claudeの所感**: 各手法の`test_hac_auto_lags_runs_and_returns_finite_std_errors`
  （OLS/WLS/IV 2SLS/IV GMMの4箇所）に、`benchmark.common.hac_auto_lag(n)`を
  明示的に`hac_lags`へ渡した結果と、`hac_lags=None`（自動）の結果を同一
  データセットで比較し、標準誤差が厳密一致することを確認するテストを追記すれば、
  結果にラグ数を露出させなくても直接検証できる（IV 2SLS/GMMのRust内蔵
  テストと同じ発想をPython側で行い、実際にPython⇔Rustの式一致を検証する形に
  拡張する）。
- **気づいた経緯**: 2026-09-05、`HAC_MAXLAGS`（`benchmark/linear/constants.py`）の
  設計を巡るユーザーとの議論中に、IV側の自動ラグ選択式の一致がどこまで
  検証されているかを確認した過程で判明。
- **状態**: 未対応（記録のみ、着手要否はユーザー判断待ち）。ラグ数を結果に
  含める案は別途**Issue #282**として発行済み（これが実現すれば、本項目の
  直接クロス言語検証テストも結果を介して書けるようになる）。

### 65. IV: クラスター数`G<=q`の構造方程式向け事前チェック（Issue #289）が、Python APIからは実質到達不能

- **対象**: [engine/src/iv/two_sls.rs:176-192](../../../engine/src/iv/two_sls.rs#L176-L192)
  （`TwoSlsEstimator::fit`冒頭、`compute_first_stage`呼び出しより前に構造方程式の
  `q`で`validate_cluster_count_covers_slopes`を呼ぶ設計。`gmm.rs`も同型）・
  [engine_pybind/src/iv/common.rs:635-707](../../../engine_pybind/src/iv/common.rs#L635-L707)
  （`fit`関数、`TwoSlsEstimator::fit`/`GmmEstimator::fit`を呼ぶより**前**に、弱操作
  変数診断（`weak_instrument_f_statistics`）のため`compute_first_stage`を
  `method`によらず無条件で呼んでいる）
- **内容**: 項目26（`ValidationError`メッセージ内容の検証）の実装中に、
  `test_iv_validation.py::test_cluster_count_at_most_slopes_raises_validation_error`
  へ`match=`を追加する過程で発覚。`engine/src/iv/CLAUDE.md`・
  `two_sls.rs`のdocコメント・当該テストのdocstringは、いずれも「クラスター数
  `G`が構造方程式の傾き係数の数`q`以下の場合、`fit()`冒頭で構造方程式の`q`を
  使った`CommonError::InsufficientClustersForInference`を**第一段階回帰の
  `FirstStageFailed`ラップより前に**返す」という設計・実装意図を記載しており、
  対応する`engine`側のRust単体テスト
  （`fit_returns_validation_error_when_cluster_count_at_most_slopes`）もこの
  前提で書かれ、実際に（`TwoSlsEstimator::fit`を直接呼べば）その通りに動く。
  しかし**Python API（`IV(...).fit()`）経由では、この事前チェックに到達する前に
  必ず`engine_pybind::fit()`が呼ぶ`compute_first_stage`（弱操作変数診断専用）が
  先に失敗する**。第一段階回帰自身も内部で同型の`G<=q`チェックを持つが、
  識別条件（`len(instruments) >= len(x_endog)`）上、第一段階の`q`
  （`x_exog`+`instruments`の傾き数）は常に構造方程式の`q`
  （`x_exog`+`x_endog`の傾き数）以上になるため、**構造方程式側の条件
  `G<=q_structural`が成立する場面では、第一段階側の条件`G<=q_firststage`
  （`q_firststage>=q_structural`）も必ず同時に成立し、先に発火する**
  （`compute_first_stage`が`TwoSlsEstimator::fit`/`GmmEstimator::fit`より前に
  呼ばれるため）。この2つのチェックはロジックとしては同値の状況を検出できて
  いる（型は正しく`ValidationError`のまま）ため実害は限定的だが、以下の
  ズレがある。
  1. **メッセージの`g`・`q`が構造方程式のものと食い違う**（第一段階回帰の
     `x_exog`/`instruments`基準の`q`になる。過剰識別の場合ほど構造方程式の
     `q`より大きくなる）。
  2. **`FirstStageFailed`にラップされる**ため、「構造方程式そのものが弾かれた」
     という直接的なメッセージにならず、「診断計算（第一段階回帰）が失敗した」
     という体裁になる。
  3. 理論上、構造方程式は`G>q_structural`で安全なのに、過剰識別度が高く
     第一段階の`q_firststage`が`G`を上回るケースでは、実際には安全な構造推定
     まで拒否される可能性がある（今回の調査では逆方向——構造方程式側が
     弾かれるはずの状況が必ず先に発火する側——のみ実測確認し、この逆方向
     ケースは理論的な指摘に留まる。実測は別途要）。
  4. `engine`側の`TwoSlsEstimator::fit`/`GmmEstimator::fit`冒頭の事前チェック
     （Issue #289で追加）は、Python APIからは事実上デッドコード。
- **Claudeの所感**: `engine_pybind::fit()`に、`InsufficientInstruments`
  （識別の順序条件、`compute_first_stage`より前に既にチェック済み、
  `engine_pybind/src/iv/common.rs:647-656`）と同じパターンで、構造方程式の`q`を
  使った`G<=q`チェックを`compute_first_stage`呼び出しより前に複製すれば
  解消できると考える（`engine`側の値をそのまま再利用できるかは要確認）。
  スコープはOLS/WLS等より小さいが、`engine_pybind`側のロジック追加になるため
  `refactor`スキルの範囲外。
- **気づいた経緯**: 2026-09-11、項目26（`ValidationError`メッセージ内容の
  検証追加）の実装中に、`test_cluster_count_at_most_slopes_raises_validation_error`
  の期待メッセージが実測と食い違うことから発覚。
- **状態**: 未対応（ユーザー判断により記録のみ、修正は別Issue・別セッションで
  検討）。今回追加したテスト自体は実際の挙動（`FirstStageFailed`ラップ・
  第一段階の`q`）に合わせて`match=`を設定済み（`tests/iv/test_iv_validation.py`）。

### 66. Logit/Probit: `SeparationSuspected`検出が小標本境界（`n=k+1`）でほとんど機能しない（閾値100.0のスケール不整合）

- **対象**: [engine/src/nonlinear/logit.rs](../../../engine/src/nonlinear/logit.rs)・
  [docs/spec/logit-spec.md](../../spec/logit-spec.md)3.2節・4章
  （`SEPARATION_PARAM_NORM_THRESHOLD=100.0`）
- **内容**: nonlinear系統の自由度1境界ケース（`n=k+1`）に凍結データが無いことの
  要否検証中に発見（対応済み・クローズ済みの旧項目）。
  `generate_binary_choice_dataset("baseline", link="logit", n=5, k=3, seed=0..499)`
  （`n=k+1`、engine側`k=4`）で`Logit(...).fit()`を実測したところ、
  `converged=True`のまま完全分離（予測確率が0/1の浮動小数点極値に張り付く）に
  陥ったケースが415/500件（83.0%）あり、そのうち`SeparationSuspected`で
  実際に例外になったのは62件（12.4%）のみだった。デフォルトseed=42でも
  再現（`params`が最大±45、`std_errors`が最大4147等の明らかに異常な値でも
  `converged=True`のまま返る）。項目6（`SEPARATION_PARAM_NORM_THRESHOLD`の
  多変量モデル・k大での誤検知リスク）とは逆方向（今回は閾値が緩すぎて
  見逃す）の問題で、閾値`100.0`が`n=200, k=3`の単一データセットでの実測較正値
  （`docs/spec/logit-spec.md`3.2節）であり小標本でのスケール不整合を検証して
  いなかったことに起因すると推測される。
- **Claudeの所感**: Issue化して`engine`側の閾値見直し（`n`依存の基準にする等）を
  検討する価値があると考える。Probit側は`nonlinear/common.rs`の`run_solver`を
  共有するため同種の限界を持つ可能性が高いが未検証。
- **気づいた経緯**: 2026-09-13、自由度1境界ケースの要否検証作業中に発見。
- **状態**: Issue化済み（[#317](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/317)）。

### 68. Tobit: proptestが左打ち切りのみで、右打ち切り・両側打ち切りはproperty-basedでは未カバー

- **対象**: `engine/src/nonlinear/tobit.rs`の`mod proptests`
- **内容**: OLS/WLS/Logit/Probitに続くproptest拡張（2026-09-13実装）で、
  Tobitにも`score_is_near_zero_at_converged_params`/
  `coefficients_and_se_are_invariant_to_column_order`/
  `hc0_std_errors_are_at_most_hc1_std_errors`の3プロパティを追加したが、
  ケース生成は左打ち切り（`lower=0.0`固定、`upper`は打ち切りなし）のみを
  対象にしている。`censored_contribution`の`direction=-1.0`（右打ち切り）
  分岐や、左右が混在するデータセットは、この3プロパティでは一度も経由され
  ない。固定フィクスチャ（`benchmark/nonlinear/datasets.py`の
  `TOBIT_SCENARIOS`の`right_censoring`/`interval_censoring`、
  `tests/nonlinear/test_tobit*.py`）では既に数値照合済みのため「未検証」
  ではないが、property-basedテストの強み（多数のランダム構成での不変条件
  検証）がこの分岐には及んでいない。
- **Claudeの所感**: rust-reviewerからshould fix指摘として上がったが、
  Logit/Probitの拡張とIV系統への拡張を優先し、今回は見送りとする方が
  作業のペースとして適切と考える。対応する場合は、3プロパティを
  `lower`/`upper`をランダムに持たせる形に拡張する（または右打ち切り専用の
  ケース戦略を追加する）案が考えられる。
- **気づいた経緯**: 2026-09-13、Tobit proptest追加のrust-reviewerレビュー中に
  指摘。
- **状態**: 未対応（ユーザー確認済み、今回は見送りと決定）。

### 69. IV: `many_regressors`（高k）・`outlier_regressor`（外れ値）シナリオが未追加

- **対象**: `benchmark/iv/datasets.py`（合成データセット生成）
- **内容**: 旧項目2（高次元シナリオ、2026-09-13クローズ）・旧項目67
  （外れ値・裾の重い分布シナリオ、2026-09-13クローズ）はいずれもOLS/WLS/
  Logit/Probit/Tobitの5手法には対応済みだが、IV（2SLS/GMM）には
  `many_regressors`・`outlier_regressor`のいずれも追加されていない
  （`benchmark/iv/datasets.py`の`SCENARIOS`に該当エントリなし）。IVは
  内生変数`x_endog`・操作変数`instruments`・構造誤差と第一段階誤差の相関
  という他手法に無い構造を持つため、単純な移植ではなく次の設計判断が
  必要になると考えられる。
  - **高kサブ項目**: `x_exog`（外生説明変数）側だけを増やすのか、
    `instruments`側も増やすのか（過剰識別度合いが変わる）を決める必要が
    ある。
  - **外れ値サブ項目**: 汚染をどの列に適用するか（`x_exog`のみか、
    `x_endog`・`instruments`にも適用するか）で、除外制約・関連性の
    識別前提が崩れないかの検討が必要になりうる。
- **Claudeの所感**: IV固有の設計判断が伴うため、着手前に既存のIV実装
  （`docs/spec/iv-spec.md`4章・`benchmark/iv/datasets.py`の
  既存シナリオ設計）を確認し、どの列に何を適用するかをユーザーに確認して
  から実装する方針が良いと考える。
- **気づいた経緯**: 2026-09-13、旧項目2のクローズ内容を確認する過程で
  IVが対象外だったことに気づいた。
- **状態**: 未対応。

### 70. Logit/Probit: `SeparationSuspected`の近傍分離テストが両極端（明確に発火／明確に安全）のみで、閾値に近い境界ケースがピン留めされていない

- **対象**: [engine/src/nonlinear/logit.rs](../../../engine/src/nonlinear/logit.rs)・
  [engine/src/nonlinear/probit.rs](../../../engine/src/nonlinear/probit.rs)の
  `fit_returns_separation_suspected_error_for_near_separation_data`・
  `fit_converges_normally_for_mild_near_separation_data_across_all_methods`
- **内容**: rust-reviewerの指摘（項目10のProbit回帰テスト追加時のレビュー、
  2026-09-13）。両手法とも、近傍分離データの回帰テストは「明確に`SeparationSuspected`が
  発火するケース」（logit: `beta1=100`・probit: `beta1=50`、標準化パラメータノルムが
  閾値100を大きく上回る）と「明確に正常収束するケース」（両手法とも`beta1=20`、ノルムが
  閾値に対して大きな余裕を持つ）の両極端のみを固定しており、閾値100に対して数%程度の
  マージンしかない境界付近（項目10の調査で実測したprobit `norm≈93.3`・logit
  `norm≈89.0`相当）は回帰テストとしてピン留めされていない。将来、最適化経路や依存
  クレートの変更でこの安全マージンがじわじわ縮む・広がるような回帰が起きても、現状の
  テストでは検知できない可能性がある。Logit側にも同型の構造的なギャップが元々あり、
  今回のProbit側追加に固有の劣化ではない。
- **Claudeの所感**: 境界に近い`beta1`（ノルムが90台になる値）を追加でピン留めする
  価値はあると考えるが、`beta1`とノルムの対応は実測で較正し直す必要があり、かつ
  「境界に近い」こと自体がテストの意図であるため、将来の実装変更でこのテストが
  falseになった場合に「意図的な閾値調整」なのか「望まない回帰」なのかの切り分けが
  難しくなる可能性がある。着手前にこの点をどう扱うか（許容範囲を持たせる、コメントで
  明記する等）をユーザーに確認したい。
- **気づいた経緯**: 2026-09-13、項目10（Probitの`SEPARATION_PARAM_NORM_THRESHOLD`較正
  検証）のrust-reviewerレビュー中。
- **状態**: 未対応（要否・優先度はユーザー判断待ち）

### 71. OLS/WLSのクラスターSEフィクスチャ生成（`_run_cluster_case`）が、patsy由来の切片名"Intercept"を"const"へ正規化していない（同ファイル内の他cov_typeと不整合）

- **対象**: [benchmark/linear/fixtures/generate_ols_fixtures.py](../../../benchmark/linear/fixtures/generate_ols_fixtures.py)の`_run_cluster_case`
  （`extract_coef_se(model)`をそのまま返す）・
  [benchmark/linear/fixtures/generate_wls_fixtures.py](../../../benchmark/linear/fixtures/generate_wls_fixtures.py)の同名関数（同じパターン）
- **内容**: testing-completeness-reviewerの指摘（項目13・33のOLS実データ追加レビュー、
  2026-09-13）。`statsmodels_ref.py`の`run()`は`normalize_names(raw, stat_key="t_stats")`で
  patsyの切片名"Intercept"を本実装の"const"へ正規化しているが、`generate_ols_fixtures.py`・
  `generate_wls_fixtures.py`双方の`_run_cluster_case`（baselineシナリオの疑似グループ
  クラスターケース専用ヘルパー、`smf.ols`/`smf.wls`を直接呼ぶ）はこの正規化を経由せず
  `extract_coef_se(model)`をそのまま返すため、同じフィクスチャJSON内で
  `classical`等（"const"）と`cluster`系（"Intercept"）のキー名規則が食い違っている。
  `normalize_names`は`t_stats`/`p_values`/`conf_int`の存在を前提とする設計のため
  （`coef`/`se`のみの`_run_cluster_case`の返り値にはそのまま適用できない）、項目13の
  実装で新規追加した`_run_wage1_region_cluster_case`（OLS、wage1の実データクラスター
  ケース）ではcoef/seのみを直接畳む形で個別に対応済みだが、既存の`_run_cluster_case`
  （OLS/WLS双方、baseline/cluster_imbalanced/cluster_g2が対象）は未対応のまま。
  `tests/_assertions.py`の`assert_dict_close`が既定で`rename=rename_intercept`を持つため
  実害（テスト失敗）は無い。
- **Claudeの所感**: 実害が無いため優先度は低いが、フィクスチャの一貫性という観点では
  `_run_cluster_case`側にも同じ正規化（coef/seのみを直接畳む形、`_run_wage1_region_
  cluster_case`と同じ書き方）を適用するのが妥当。OLS/WLS両方に同型の修正が必要。
- **気づいた経緯**: 2026-09-13、項目13・33（OLS実データのstatsmodels側追加）の
  testing-completeness-reviewerレビュー。
- **状態**: 未対応（実害無しのため優先度低、着手要否はユーザー判断待ち）

### 73. OLS: `test_scale_variance_raises_computation_error`のcov_typeパラメトライズに`cluster`が含まれておらず、docstringの「全cov_typeでbackstop」という主張が未検証

- **対象**: `tests/linear/test_ols_validation.py`の
  `test_cluster_count_at_most_slopes_raises_validation_error`のdocstring
  （「`G>q`でも悪条件で数値的にほぼ特異なケースは`test_scale_variance_raises_
  computation_error`がbackstop」と明記）と、実際の
  `test_scale_variance_raises_computation_error`の実装
  （`@pytest.mark.parametrize("cov_type", COV_TYPES)`、
  `COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3", "hac"]`で
  `cluster`を含まない）
- **内容**: `testing-completeness-reviewer`の指摘（2026-09-21、項目29の
  レビュー中）。docstringは「全cov_typeでbackstopされる」と主張しているが、
  実装上`cov_type="cluster"`はこの`ComputationError`backstopテストで
  一度も実行されていない。手動で`scale_variance`データセット＋
  `cov_type="cluster"`（`G=10>q=3`）を実行したところ実際には
  `ComputationError`が正しく発生することを確認できたが、これは自動テストで
  検証されておらず、docstringの主張と実装が食い違っている状態。
  項目29でクラスター×悪条件シナリオの成功パス側を拡充したのに対し、
  こちらは同じ組み合わせのエラーパス側（`ComputationError`backstop）の
  対称漏れであり、項目29と直接関連する。
- **Claudeの所感**: `test_scale_variance_raises_computation_error`の
  `cov_type`パラメトライズに`cluster`を追加する形が自然だが、`cluster`は
  `cluster_col`パラメータが別途必要なため、既存の`COV_TYPES`パラメトライズに
  単純に含めることはできず、別テスト（または条件分岐）が必要になる。
- **気づいた経緯**: 2026-09-21、項目29（クラスターロバストSEの悪条件・
  多重共線性シナリオとの組み合わせ追加）対応の`testing-completeness-reviewer`
  レビューで発見。
- **状態**: 対応済み（2026-09-21）。`tests/linear/test_ols_validation.py`に
  `test_scale_variance_cluster_raises_computation_error`を専用テストとして
  追加（`cluster_col`が必要なため既存の`COV_TYPES`パラメトライズには
  含めず、均等な疑似グループ`G=10>q=3`で`ComputationError`が発生することを
  確認）。`test_cluster_count_at_most_slopes_raises_validation_error`の
  docstringも新テスト名を指すよう更新した。`tests/`配下1676件全通過・
  Ruffクリーンを確認済み。

### 74. WLS/IVにも項目73と同型の構造的ギャップがある（`test_scale_variance_raises_computation_error`のcov_typeパラメトライズに`cluster`が無い。ただしOLSと異なりdocstringの虚偽記載は伴わない）

- **対象**: `tests/linear/test_wls_validation.py`（`COV_TYPES`は
  `generate_wls_fixtures.py`由来、`classical/hc0/hc1/hc2/hc3/hac`のみで
  `cluster`を含まない）、`tests/iv/test_iv_validation.py`
  （`COV_TYPES = ["classical", "hc0", "hc1", "hac"]`をファイル内で独自定義、
  同じく`cluster`を含まない）。いずれも`test_scale_variance_raises_
  computation_error`相当のテストに`cluster`専用backstopが無い。
- **内容**: `testing-completeness-reviewer`の指摘（2026-09-21、項目73対応の
  レビュー中）。項目73と全く同型の構造（`cov_type="cluster"`は
  `cluster_col`が別途必要なため既存の`COV_TYPES`パラメトライズに単純に
  含められず、backstopテストが存在しない）がWLS・IVにも現存する。
  ただしOLSの元の問題（docstringが「全cov_typeでbackstop」と誤って主張して
  いた）とは異なり、WLS・IVの該当docstring（
  `test_cluster_count_at_most_slopes_raises_validation_error`相当）は
  そのような虚偽の主張をしていないため、**虚偽記載ではなく単なる未検証
  カバレッジの欠落**（重要度はOLSのケースより一段低い）。
  対照的に`tests/panel/test_fe_validation.py`・`tests/panel/test_re_
  validation.py`は`COV_TYPES`に`cluster`を含めた上で`cluster_col`省略時に
  entityへフォールバックする実装特性を利用しており、既にこのギャップを
  回避できていることを実行確認済み（`cluster`含む5ケース全通過）。
- **Claudeの所感**: 項目73と同じ形（専用テスト追加、`cluster_col`は
  `with_cluster_groups`等の既存ヘルパーでG十分大きく設定）で対応できる。
  IVは`COV_TYPES`がファイル内独自定義なので、まず`cluster`を含むかどうか
  含め既存の`ValidationError`側テスト（クラスタ数境界）の構成を確認してから
  着手するのが安全。
- **気づいた経緯**: 2026-09-21、項目73（OLSの`cluster`×`scale_variance`
  backstopテスト追加）対応の`testing-completeness-reviewer`レビューで発見。
- **状態**: 対応済み（2026-09-21）。WLS・IVそれぞれに項目73と同型の専用
  テストを追加した。`tests/linear/test_wls_validation.py::test_scale_
  variance_cluster_raises_computation_error`（`with_cluster_groups`で
  `G=10>q=3`）、`tests/iv/test_iv_validation.py::test_scale_variance_
  cluster_raises_computation_error`（`G=10`、第一段階回帰の`q=4`
  〔`x_exog`2列+`instruments`2列〕より十分大きい値。実装前にPythonから
  手動実行し、想定通り第一段階回帰の`ComputationError`
  〔`FirstStageFailed`〕が発生し、クラスタ数不足の`ValidationError`
  〔`test_cluster_count_at_most_slopes_raises_validation_error`が別途
  確認済みの経路〕とは区別できることを確認済み）。`tests/`配下1682件
  全通過・Ruffクリーンを確認済み。

  **作業中の余談（このドキュメントの経緯として記録）**: 対応中に、
  このセッションのgit作業ディレクトリが（本セッションの外側で並行して
  動いていた別セッションにより）`release/v0.7.0`から`release/v0.8.0`へ
  切り替わっていたことが判明した。`release/v0.7.0`は既に
  `chore(release): v0.7.0`としてリリース済みで、項目17・28・29・72・73の
  作業内容はすべて引き継がれていることを確認した上で、ユーザー確認の上、
  本項目は現在チェックアウトされている`release/v0.8.0`側にコミットする
  方針とした（このコミット自体が本項目の変更に含まれる）。

### 75. 項目31（OLSのfit()でx列のNaN/無限大検証テストが無い）と同型のギャップをLogit/Probit/IVにも適用する（2026-08-23時点の既定方針を上書き）

- **対象**: `tests/nonlinear/_binary_choice_checks.py`の`check_null_values_raise`
  （`y`側のみ）・`tests/iv/test_iv_validation.py`の`test_null_values_raise`
  （`y`/`x1`/`endog1`/`z1`全列parametrize済みだがNaN/無限大側のテストが丸ごと
  無い）。
- **内容**: ユーザー依頼（2026-09-23）で、項目31（OLS）・項目34（WLS）対応後に
  Probit/Logit/Tobit/IV/Panelを横断確認した。
  - **Tobit**: ギャップ無し。`test_tobit.py`の`test_null_values_raise`/
    `test_non_finite_values_raise`が`y`/`x1`×null/NaN/無限大を既に対称に
    網羅済み（v0.6.0リリース時点、項目63の追記参照）。
  - **Panel（FE/RE）**: ギャップ無し。`test_numeric_column_non_finite_
    values_raise`が`bad_col=["y","x1"]`×`{NaN,inf}`を全組み合わせ
    parametrizeしており、OLS/WLSより網羅的（先に実装済みだった）。
  - **Logit/Probit**: 項目31と同型のギャップを確認。`check_null_values_raise`
    は`y`列のみで`x1`側が未検証、かつNaN/無限大の専用テストが丸ごと無い
    （`predict()`側の`check_predict_null_or_non_finite_values_raise`は
    `x1`のnull/infを既にカバー済みで非対称）。
  - **IV**: 別の形のギャップ。`test_null_values_raise`は`y`/`x_exog`/
    `x_endog`/`instruments`の全列をparametrizeで既にカバーしていたが、
    NaN/無限大側のテストがどの列についても存在しなかった（IVには
    `predict()`自体が無いため、比較対象となる非対称は無い）。
  - 項目40の「NaN/無限大の専用テストが無いのは、既存分（OLS/Logit/Probit）は
    そのままにする、というユーザー既定方針（2026-08-23）」との整合性を
    ユーザーに確認した上で、今回OLS/WLSに適用したのと同じ方針変更として
    Logit/Probit/IVにも適用する判断を得た。
- **Claudeの所感**: `_binary_choice_checks.py`で共通化されているため
  Logit/Probitは1箇所の修正で両方に反映される。IVは`predict()`が無い分、
  修正のスコープはOLS/WLSより単純（`fit()`側のみ）。
- **気づいた経緯**: 2026-09-23、項目31（OLS）・項目34（WLS）対応後の
  ユーザー依頼によるPhase横断確認。
- **状態**: 対応済み（2026-09-23）。`_binary_choice_checks.py`の
  `check_null_values_raise`に`x1`列のnullケースを追加し、新規
  `check_non_finite_values_raise`（`y`/`x1`×NaN/無限大の4ケース）を追加、
  `test_logit_validation.py`・`test_probit_validation.py`双方に薄い
  ラッパーを追加した。`test_iv_validation.py`には新規
  `test_non_finite_values_raise`（`bad_col=["y","x1","endog1","z1"]`×
  `{NaN,inf}`の8ケース）を追加した。`tests/`配下1693件全通過・Ruffクリーンを
  確認済み。

