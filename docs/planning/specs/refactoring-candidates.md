# リファクタリング候補メモ

コード解説（`/explain-code`スキル等）や通常の実装作業の過程で気づいた、
リファクタリングの余地がある箇所を随時記録する場所。

`refactoring-issue231-progress.md`との違い: あちらは
[#231](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/231)
としてスコープ・フェーズを確定させた上で実施する計画書だが、こちらは
Issue化する前の**気づいた時点での未整理のメモ**を溜める場所。ここに溜まった
項目は、着手時にIssue化するか`refactor`スキルの対象範囲として指定するかを
都度ユーザーが判断する。

## 記録フォーマット

各項目は以下を含める。

- **対象**: ファイルパス・行
- **内容**: 何が気になったか
- **気づいた経緯**: どの作業中に気づいたか（日付）
- **状態**: 未対応 / 対応済み（対応したIssue・PR等） / 対応不要と判断（理由）

**完了項目の扱い（2026-08-22運用ルール、2026-08-30更新）**: 「対応済み」「対応不要と
判断」「調査の上やらないと判断」のいずれになった項目もこのファイルから削除する
（コード自体とgit logが原本）。削除した記録は`refactoring-issue231-progress.md`の
「随時対応ログ」に残す——「対応済み」は要点（対応内容・コミットハッシュ）のみ、
却下・見送り判断は同じ提案の調査をやり直さずに済むよう根拠も含めて残す。番号は
削除後も詰め直さない（欠番があっても他項目からの「項目N」表記自体は維持できる）。
ただし**番号を維持しても参照先の内容は消える**ため、削除対象の項目を他の項目が
「項目N」で参照している場合は、削除前にその参照側へ必要な文脈（何の話か・結論）を
埋め込み、削除後も参照側だけで自己完結するようにする。

---

## 一覧

※ 項目12・13・15・16・18〜21・24〜29・31〜35・39 は、`benchmark/` の構造変更（Initiative A）が上位計画として吸収したため削除した（対応の進捗・吸収項目の扱いは`refactoring-issue231-progress.md`「Initiative A」節）。番号は詰め直さない（記録フォーマット節参照）。

### 44. engine（faer/rayon）のマルチスレッド線形代数が、多コア機・負荷下でシングルスレッド時の20倍以上遅くなり不安定になる

- **対象**: `engine/src/linear/ols.rs`の`OlsEstimator::fit`（faer経由の
  QR分解・Gram行列構築）。他の線形代数を伴う手法（WLS/IV/GMM等）も同傾向の
  可能性。
- **内容**: 2026-08-30、Issue #250/#98のOLS性能比較の再計測中に発覚。
  12論理コアのdevcontainer（WSL2）上で`OLS(...).fit()`（classical, n=1,000,000,
  k=5）を計測すると、スレッド数無制限では実行時間が3〜4秒かつ試行ごとに
  0.6〜4.5秒と大きくばらつく。同じ条件で`RAYON_NUM_THREADS=1`等を設定して
  シングルスレッドに固定すると**0.14〜0.18秒で安定**し、statsmodels
  （0.33〜0.49秒）より2〜3倍速いという期待どおりの結果になる。
  faerは0.24.4のまま・`ols.rs`は2026-08-09以降変更なしのため、コード
  リグレッションではなく、スレッドプールが負荷下で競合する挙動の問題。
  比較対象のstatsmodels（numpy/OpenBLAS）は同条件でも安定して劣化しなかった。
- **ユーザーの懸念（2026-08-30）**: ベンチマークは`OLS(...).fit()`をそのまま
  呼んでおり、これは実利用と同じ経路。多コア機のユーザーが大規模データで
  `fit()`すると同じ現象が起きうる（ベンチマークで先に見つかったのは幸い）。
- **想定される調査の方向**: OLSの設計行列はtall-skinny（n大・k小）で、skinny
  行列のQR/Gram構築を多スレッドに分割してもスレッドプールのオーバーヘッドと
  メモリ帯域競合が支配的になりやすい。問題サイズに応じてシングルスレッドに
  留める閾値、あるいは明示的なスレッド数上限の導入を検討する。WSL2固有の
  スケジューラ挙動か、ネイティブの多コアLinuxでも再現するかの切り分けも要る。
  `.claude/rules/rust-style.md`「パフォーマンス」節のrayon採用は「実測してから
  決める」方針であり、本件はその実測データ点。
- **状態**: 未対応（engineのロジック挙動に関わるためリファクタリングの範囲外。
  別途Issue化して調査する。性能比較ハーネス側は`_SINGLE_THREAD_ENV`で
  スレッド数を1に固定し、この現象を計測から切り離す対応を実施済み＝
  `de0b4a7`の後続コミット）

### 45. engineのProbitが、statsmodelsが収束できる大標本条件でHessian特異エラーを出す

- **対象**: `engine/src/nonlinear/`のProbitのHessian（観測情報行列）構築・
  Newtonソルバ（`nonlinear/common.rs`の`run_solver`共有部分を含む）。
- **内容**: 2026-08-30、Probitの性能比較（#253）実装中に発覚。
  `generate_binary_choice_dataset("baseline", link="probit", n=1_000_000,
  k=5, seed=42)`を`Probit(...).fit()`すると
  `ComputationError: the Hessian is singular and cannot be inverted`。
  **同じデータで statsmodels の `smf.probit(...).fit()` は収束し Hessian も
  反転できる（1.87秒、engineは即エラー）**。n=100,000では engine も通り、
  k=3なら n=1,000,000 でも engine が通る。Φはロジスティック分布のΛより裾が
  薄く、n・kが大きいとXβの分散増大でΦ(Xβ)が0/1に飽和する観測が増え、
  Probitの観測情報行列の重み`φ(xβ)²/[Φ(xβ)(1−Φ(xβ))]`がアンダーフローして
  Hessianが数値的に特異化するのが原因と推測。Logit（Λ）では
  n=1,000,000, k=5 でも問題は起きない。
- **ユーザー指摘（2026-08-30）**: statsmodelsが同条件を捌けている以上、
  engine側に不具合の可能性がある。飽和に強い実装（重みのクリッピング・
  対数空間でのΦ(1−Φ)計算・Newtonのdamping/line search・step halving等、
  statsmodelsが持つ頑健化）の余地がないか調査する価値がある。
- **状態**: 未対応（engineのロジック挙動に関わるためリファクタリングの範囲外。
  別途Issue化して調査する。性能比較側はProbitのn軸上限を100,000に制限して
  回避＝`compare_probit.py`の`PROBIT_ADAPTER.n_sweep`、#253のコミット）

### 46. engineのLogit/ProbitのBFGS/L-BFGSがNewton・statsmodelsの同methodより桁違いに遅い

- **対象**: `engine/src/nonlinear/`のLogit/Probitの最適化ソルバのうち
  `method="bfgs"`/`"lbfgs"`の経路（`nonlinear/common.rs`の`run_solver`の
  quasi-Newton分岐）。
- **内容**: 2026-08-30、Logit/Probitの性能比較にmethod軸（#252/#253の追補）を
  足したところ発覚。classical・k=5・代表nで newton と bfgs/lbfgs を実測した
  結果（シングルスレッド、`repeats=3`中央値）:
  - **Logit（n=1,000,000）**: newton engine 0.65秒 / bfgs engine **11.21秒** /
    lbfgs engine **23.91秒**。同じ method の statsmodels は bfgs 1.47秒・
    lbfgs 1.44秒で newton（1.37秒）とほぼ同じ。engine の bfgs は newton 比
    約17倍、lbfgs は約37倍、statsmodels 比でも bfgs 約7.6倍・lbfgs 約16.6倍遅い。
  - **Probit（n=100,000）**: newton engine 0.12秒 / bfgs engine **0.91秒** /
    lbfgs engine **0.82秒**。statsmodels は bfgs 0.18秒・lbfgs 0.16秒。
    engine の quasi-Newton は newton 比 約7倍、statsmodels 比 約5倍遅い。
- **推測される原因**: quasi-Newton は1反復あたり勾配のみ（O(nk)）で Newton の
  Hessian 構築（O(nk²)）より軽いはずなので、(a) 収束が極端に遅く反復回数が
  膨れている、(b) line search（ステップ幅探索）が非効率で1反復あたり多数の
  関数評価をしている、(c) 逆Hessian近似の更新や保持がナイーブ、のいずれか。
  statsmodels（scipy の `fmin_bfgs`/`fmin_l_bfgs_b`）は大標本でも数回反復で
  収束しており、engine 側のステップ制御・収束判定・近似更新の実装に
  改善余地がある。
- **状態**: 未対応（engineの最適化ロジックに関わるためリファクタリングの
  範囲外。別途Issue化して調査する。性能比較側は method 軸を代表点1つで
  計測して現状を記録する（`compare_logit.py`/`compare_probit.py`の
  `extra_methods`、#252/#253追補コミット）。既定の newton は実用上十分速い
  ため、当面の実害は「newton 以外を選ぶと遅い」という選択上の注意に留まる）
