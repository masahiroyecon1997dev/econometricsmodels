# 直交座標での共分散・全体Wald/F統計量（仕様書）

OLS・WLS・IV（2SLS・GMM）・FE・RE・FE-IV（2SLS・GMM）・Tobitが、係数の共分散と全体Wald/F統計量（RE・FE-IVのHausman・Wu-Hausmanを含む）を、QRの因子`R`の直交座標で計算する理由・構成・受け入れ境界・テスト方針。手法ごとの式は各手法の仕様書にあり、ここは共通の設計だけを書く。式の定義は`docs/methods/`が正本。

## 1. 原因と方法

サンドイッチ共分散`V = B M B`（`B = (X'X)⁻¹`）を作ってからその傾き部分行列を反転すると、`B`が条件数`cond(X)²`の行列の明示的な逆行列で、傾きの相関が`1 - 1e-11`程度を超えると`V`が桁落ちで数値的に正定値でなくなる（classicalは`σ²B`で桁落ちしにくく`1 - 1e-15`程度まで通る）。1回目の逆行列で失った精度は2回目の反転で取り戻せない。

`X = QR`（切片を先頭に置いたピボットなしHouseholder QR）とすれば`β = R⁻¹Q'y`、`V = R⁻¹ M̃ R⁻ᵀ`。`M̃`は`x_i`を`Q`の`i`行目`q_i`に置き換えた直交座標のミートで、`Q'Q = I`なので条件数は`Ω`の条件数で決まり`X`の条件数に依らない。

- **統計量**: 末尾`q`個の傾きのWaldは`β_S'V_SS⁻¹β_S = b̃_S'M̃_SS⁻¹b̃_S`（`b̃ = Q'y = Rβ`）。`R⁻¹`が上三角なので末尾`q`個の係数は`b̃`の末尾`q`成分だけで決まる。`V`を作らず`M̃`の末尾ブロックだけを反転する。
- **ピボットなしにする理由**: 列を並べ替えると「検定対象が末尾`q`成分」の性質が崩れる。全手法で切片が先頭・検定対象が末尾（FE・FE-IVは切片なし、REのHausman・Wu-Hausmanは末尾`k`列、Tobitは`[切片, logσ, 傾き]`）。
- **ミートの再利用**: `M̃`は既存のmeat構築部品（`hc_cov_params`・`cluster_cov_params`・`hac_cov_params`・パネルの`panel_cov_params`）に`x = Q`・`xtx_inv = 単位行列`を渡すだけで得られる。小標本補正はスカラー倍なのでそのまま効き、HC2/HC3のレバレッジは`‖q_i‖²`になる。
- **`wald_quadratic_form`**: `params = b̃`、`cov_params = M̃`を渡す（相関スケール化と悪条件の判定も`M̃_SS`に対して働く）。Wald用の別関数は持たない。
- **標準誤差**: `cov_params`も`V = R⁻¹M̃R⁻ᵀ`で作る（1回の`M̃`計算で統計量と標準誤差の両方）。旧経路との標準誤差の差は、完全共線の1件を除く凍結データ79件で最大`2e-12`（OLSの`cond(X) = 1e9`の1件だけ`9e-11`）で、参照実装との照合（`1e-8`）に対して十分小さい。

## 2. 共有部品（`engine/src/shared/orthonormal.rs`）

- `OrthonormalDesign`（`new`・`new_checked`・`checked_r_factor`）: 設計行列のQR。係数`R⁻¹b̃`・残差`y - Qb̃`・`V`・全体Fを同じQRから求める。ランク落ちは`R`（k×k）の列ピボットQRで判定するので、n×kの列ピボットQRを別に走らせない。`Q`は共分散計算の窓の中だけで作り、直後に破棄する。
- `WaldCoordinates`（`b̃`と`M̃`）: 推定量が保持し、末尾`q`列の検定にも使う。
- `covariance_from_factor`・`solve_upper`: `V = R⁻¹M̃R⁻ᵀ`と`R⁻¹b`を三角行列の後退代入で求める（`R⁻¹`を明示的に作らない）。Tobitが`Q`の座標の共分散・係数を標準化空間に戻すのにも使う。

## 3. 受け入れ境界

### 3.1 設計行列（OLS・WLS・IV・FE・RE・FE-IV・Tobit・GMMの`Ĝ`）

`new_checked`は、`R`の列を単位ノルムに揃えた行列に対して次を順に行う。(1) 列ピボットQRのランク判定（相対`k·ε`）、(2) 条件数が`RANK_DEFICIENT_CONDITION`（`1e12`）以内、(3) `MAX_EQUILIBRATED_CONDITION`（`5e7`）以内。(1)(2)の失敗は`RankDeficient::Columns`（列名つきの`DESIGN_MATRIX_SINGULAR`）、(3)は`GramMatrix`（`MATRIX_NEAR_SINGULAR`、`matrix=DESIGN_GRAM`）。

- **`5e7`の根拠**: QRなら`cond(X)`が`1e8`〜`1e15`の設計も推定できてしまうが、残差が非ゼロのとき係数が静かに大きくずれる（傾きの差が相対`1e-13`で係数が数十〜数百%）。そこで、従来の`X'X`のCholeskyの成否の境界（列を揃えた条件数で`2.8e7`まで成功・`9.2e7`から失敗）の間に置いた。緩めるのは破壊的でなく、厳しくするのは破壊的。
- **列を揃えた条件数で判定する理由**: Choleskyの成否は列の一様な拡大縮小に依らない。生の条件数・生の`R`の相対閾値だと、列の単位（`1e±6`・`1e±12`）で合否が動く（共線な2列を`1e-12`倍するだけでランク落ちとして先に弾かれた）。ランク判定も条件数も同じ「列を揃えた`R`」で行う。
- **`1e12`でランク落ちに揃える理由**: 完全な共線（重複列・ダミー変数トラップ）の条件数は`1e14`〜`1e16`で、(1)の閾値の境界に来てどちらに転ぶか決まらず、`DESIGN_GRAM`（列名なし）になって原因が分からなくなる。`1e12`を超えたら列名つきの報告に揃える。`1e12`付近は列の順序で境が動くので、どちらかであって内部エラーにならないことを固定している。
- **手法に依らない境界**: FE・REはwithin・準偏差化の後の設計で、IVは第一段階の設計と第二段階の`X̂`でそれぞれ判定する（`stage`で段階が分かる）。FE・REの境界は設計行列そのものよりわずかにずれる。
- **操作変数・重みの側**: 第一段階の設計は`DESIGN_GRAM`（`stage=FIRST_STAGE`）。第一段階を持たないGMM（FE-IV GMM）は`Z`と重みの失敗が`GMM_WEIGHT`、`Ĝ`の失敗が`GMM_BREAD`（列名は出ない）。

### 3.2 Tobit

標準化した設計行列を`new_checked`で分解して3.1節と同じ判定（`5e7`・`1e12`）を行う。`cov_type`によらず同じ（`opg`も）。完全な共線は`SingularDesignMatrix`（列名つき。列の特定は標準化前の設計で行う）、条件数の超過は`DESIGN_GRAM`。`-H`・スコアの側に別の限界は無い（`σ→0`に退化するデータなどで`-H`が特異なら、従来どおり`SingularHessian`・`SingularOpgMatrix`）。

**`Q`の座標で解く理由**: `-H`は`X'WX`の形で条件数が設計行列の2乗になる。`X`の座標で`H`の各要素を倍精度で組むと、その丸めだけで統計量が条件数に比例してずれる（傾きの差`1e-5`で`1e-6`、`1e-7`では`1e-2`〜全く合わない）。`-H`のコレスキー因子の座標で共分散を作っても、この`H`の丸めは残り、コレスキー分解は条件数が`1e15`を超えても成功して誤った因子を返すので、受け入れ範囲を決める根拠にもならなかった。`Q = X̃R⁻¹`の座標では`Q'Q`が定数倍の単位行列なので`H`の条件数は共線性に依らず（固定した点で約`4`）、誤差は設計行列の`cond(X)`の1乗（QRの精度）になる。ニュートン法はパラメータの線形変換に対して不変で、厳密な演算では同じ反復になる（LM型の正則化`λI`と`bfgs`・`lbfgs`の初期逆Hessianだけは座標に依存する）。

## 4. 手法別の構成

### 4.1 OLS・WLS・IV 2SLS・FE・RE・FE-IV

全手法をピボットなしQR 1本（`OrthonormalDesign::new_checked`）に統合した。係数・残差・共分散・全体Fを同じQRから求める（2本目のQRを足す案は、n=100万・k=6のOLSで増分のほぼ全てが2本目のQRで+14〜70%遅かったので採らなかった）。IV・FE-IVは`two_sls_point_estimate`が`OrthonormalDesign`を受け取る。FE-IVは`X̂`を`Q`生成の直後に解放し、`x_hat`の寿命を延ばさない。REのbetween・プール回帰、`partial_f_statistic`の制限モデルも同じ関数を通す。HCの`hc_meat`はスコア行列（n×k）を作るので、`Q`と重ならないよう行スケールを`Q`に直接かけて消費する。

- **失敗の契約**: 本体のWald失敗は`FTestFailed`等で`fit()`全体が失敗、Wu-Hausmanは`None`、REのHausmanは`HausmanTestFailed`。この写像は小さな関数（`panel/common/overall_wald.rs`・`panel/fe_iv/`のWu-Hausman・`panel/re/hausman/`）に切り出し、手組みの零行列の`M̃`で決定的にテストする（データから厳密に特異な共分散を作ると、特異性の判定が丸め誤差の大きさで決まり不安定になる）。
- **特異の原因の名指し**（`shared::wald::name_unit_leverage_cause`）: 全体Wald/F（RE・FE-IVの補助回帰を含む）が`NearSingularMatrix`（`COEFFICIENT_COVARIANCE`）で失敗したときだけ、`Q`が生きている間にレバレッジ`‖q_i‖²`（FE・FE-IVは吸収した効果込みの`leverage_full`。HC2/HC3と同じ値）を求め、`1 - h <= UNIT_LEVERAGE_TOLERANCE`の行があれば`UnitLeverageSlopeCovariance`（`UNIT_LEVERAGE_SLOPE_COVARIANCE`）に置き換える。そのために全体検定を`design`を手放す前に試す順にした。成功する推定はレバレッジを求めず、結果も変わらない。レバレッジ1の行そのものを拒否しない理由: 切片のあるOLS・WLS・IVでは、行を選ぶ方向が切片を含み傾きの空間の外なので、傾きのブロックは特異にならない（`h_ii = 1`でも標準誤差・全体Fは正しく計算できる）。切片のないモデルとFE・FE-IVではダミーそのものが傾きの空間に入り、スコア`e_i x_i = 0`で分散が0になる。IV GMM・FE-GMMの全体Waldには配線していない（重み行列が先に特異になる症状が別にあり、その`h`は`Z`のレバレッジで、残差が0とは限らないため）。
- **REの別名窓**: REのHausman補助回帰とbetween回帰は、相対残差`1e-7`未満の列を別名として落とす（Rの`lm`流。相関ランダム効果のための設計）。傾きの相対差が約`4e-8`〜`3e-7`で`hausman_df`が落ち、再パラメータ化の不変性は成り立たない。条件数のガード（`5e7`）とは別の判定で、不変性は窓の外（`1e-5`・`1e-6`）で確かめ、窓の中の挙動は別のテストで固定する。
- **REの最終回帰とHausmanのcluster・DK経路**は、`fit_auxiliary`の後に同じ設計行列を`OrthonormalDesign::new`でもう一度分解する（`OlsEstimator`が`R`・`Q`を保持しないため）。時間が変更前以下なので整理は見送った。

### 4.2 IV GMM・FE-IV GMM（`engine/src/shared/gmm.rs`・`gmm_iteration.rs`・`gmm_weight.rs`）

GMMを**白色化した操作変数の空間でのOLS**として解く。重み`S = LL'`で`Z̃ = ZL⁻ᵀ`、`Ĝ = Z̃'X = QR`、`r = Z̃'y`、`β̂ = R⁻¹Q'r`、`Ω̃ = L⁻¹Ω̂L⁻ᵀ`（`Z̃`から直接組む。列の線形変換で和が不変なので`Ω̂`を作ってから変換するのと同じ。HC2/HC3のレバレッジも不変）、全体Waldは`b̃_S'M̃_SS⁻¹b̃_S`（OLSと同じ形）。

- **白色化の経路**（`WhitenedInstruments`）: 初期推定と古典的重みは`Z`のQR。ロバスト・クラスター・HACは`S`を明示的に作らず、`D S D`（対角で揃えた`S`）の条件数が`WEIGHT_CHOLESKY_MAX_CONDITION = 1e6`以下ならCholesky、超えたら`S = A'A`の`A`のQR（`A`はロバストなら`ê_i z_i`の行、クラスターならクラスターのスコア、HACなら窓のスコア和の行`(n+L)×l/√(L+1)`。`Σ_t w_t w_t' = Σ_h (L+1-h)Γ_h`が恒等式）。`S`を明示的に作ると条件数が2乗になって精度を失う。
- **Hansen J**: `‖Z̃'ê‖²/c`（`c`は初期推定・古典的重みの重みのスケール`σ̂²`。FEは`S`に`n/(n - neffects)`を掛ける）。`S`を明示的に作る式は`1e-6`の精度を失った。C統計量は`gmm_point_estimate`経由で同じ白色化の経路を使う。
- **ガード**: `Ĝ`を列で揃えた条件数が`5e7`・`1e12`（`GMM_BREAD`）、`Z`・重みの失敗は`GMM_WEIGHT`。IV GMMはPythonでは第一段階が先に走るので`DESIGN_GRAM`・`DESIGN_MATRIX_SINGULAR`（列名つき）が先に出る。
- **クラスター重みの限界**: クラスター重みの因子`A`は設計より条件数が大きくなりうる（8クラスター・操作変数6本の`eps = 1e-7`で受け入れ範囲の内側でも`GMM_WEIGHT`）。`S`のCholeskyの時代と同じ`S`の条件数による限界で、文書化とテストで固定している（厳密値は傾きの差`1e-5`・`1e-6`のみ）。
- **HAC重みの限界**: 2段階推定の重みの弱い方向が窓和でさらに小さく、入力の丸め（相対`1e-15`）だけで`eps = 1e-6`のWaldが`1e-6`動く（Cholesky経路でもQR経路でも桁まで同じで、QR経路の利得は測れていない）。厳密値は`eps = 1e-5`のみ。HAC腕のQR経路は、条件数が`1e6`を超えたときだけ追加メモリを使う。

### 4.3 Tobit（`engine/src/nonlinear/tobit/coordinates.rs`）

`DesignCoordinates`: 標準化した設計行列を`X̃ = QR`に分解し、`Q`を`√n`に近い2の冪`s`倍した`sQ`で最適化する（パラメータは`b = (R/s)β̃`と`logσ̃`）。`TobitProblem`は`x = sQ`を渡すだけで変更が無い。

- **`Q`を2の冪倍する理由**: 標準化した列のノルムは`√n`で、`Q`は単位ノルムなので、そのままだと勾配が`√n`分小さくなり、`tol`（勾配のノルムとの比較。`bfgs`・`lbfgs`は`tol·n`）が`X̃`の座標より`√n`倍ゆるむ。`√n`に近い2の冪を掛けて勾配の大きさを合わせる。2の冪は丸めなしで掛かる。
- **初期値**: `b₀ = (sQ)'y/s²`（OLS）、`σ`は残差の標準偏差（完全な当てはまりでは`σ = 1`）。列ピボットQRを別に走らせず、ランク判定を兼ねる（時間は変更前以下）。
- **共分散**: `cov_params_for_type`をそのまま`Q`の座標の`H`・スコアに適用（良条件）し、`R⁻¹V_bR⁻ᵀ`で標準化空間`(β̃, logσ̃)`に戻して、`TobitScaling::destandardize_cov_params`で元のスケールへ。
- **全体Wald**: `wald_chi2_test(b, V_b, k_constant, df_model)`。`R`が上三角で先頭が切片・末尾が傾きなので、末尾の傾きの統計量は`b_S'V_b,SS⁻¹b_S`で、元のスケールの`β_S'V_SS⁻¹β_S`と同じ値（標準化は傾きを対角行列で変換するだけ）。`σ`は検定範囲の外（末尾）。
- **精度**: 再パラメータ化との相対差は傾きの差`1e-5`〜`5e-8`でWald・`σ`・`σ`のSEが`3.5e-10`以下、傾きの係数が`2e-9`、標準誤差が`1e-9`。同じ`H`・`S`の有理数の厳密値との差は`1e-8`未満（`engine`の単体テスト）。

## 5. テスト方針

- **有理数の厳密値**（`tests/support/_exact_wald.py`、`tests/cross_method/test_near_collinear_exact.py`）: 同じデータから全体Wald・係数（標準誤差単位）・標準誤差・Hansen Jの絶対値を`fractions`で再計算し、classical・hc0〜hc3・clusterと、GMMの1段階・2段階（classical・robust・cluster・HAC重み）で比べる。傾きの差`1e-5`〜`1e-7`（条件数`2.8e5`〜`2.8e7`）。許容誤差はWald `3e-8`・係数`6e-8`・標準誤差`3e-8`（実測の最大は`2.6e-9`・`5.8e-9`・`2.4e-9`、GMMは`1.6e-9`・`2.5e-9`・`1.4e-9`）。REは返された`θ`で準偏差化した設計の有理数の最小二乗で照合する（`θ`が無理数）。反復GMMは有理数の桁が膨らむので再パラメータ化のみ。
- **実データのOLS・WLSのクラスター**（`tests/support/_exact_cluster.py`、`tests/linear/test_ols_reference.py`・`test_wls_reference.py`の末尾の節）: Wooldridge `card`（9地域）・`wage1`（4地域）の、スケール・相関の大きい設計（`expersq`・`tenursq`、経験年数の4次多項式`card_poly`）で、係数・標準誤差・全体Fを`fractions`の厳密値と比べる。WLSは重みを`(1+ダミー)²`にして`√w`倍のOLSとして厳密に求める（`√w`が厳密に表せる重みだけ）。許容誤差は`1e-10`（実測の最大は係数`2.1e-12`・標準誤差`1.0e-13`・F`2.9e-12`）。`X'X`を明示的に反転する経路を倍精度のnumpyで再現した実測では、`card_poly`の全体Fが相対`3e-8`（OLS）・`1e-9`（WLS）ずれ、検出できるのはほぼこのケースだけ（WLSの余裕は小さい）。`card`・`wage1`はその経路でも`2e-12`以下で、精度の確認にとどまる。置き場所は、厳密値のテストが`cross_method/`にある他と違い、手法別の`_reference.py`（OLS・WLSの実データのクラスターは手法別のファイルに置く判断）。
- **再パラメータ化**（`test_near_collinear_reparametrization.py`）: 傾き2列を`(x1, (x2-x1)/2^k)`に置き換える（OLS・IVは丸めを増やさない。FE・RE・FE-IVは個体平均で約`1e-9`の摂動）。Wald/F・Hausman・Wu-Hausman・Hansen Jが`1e-7`以内。厳密値が作れないhac・dk・反復GMM・REの全体Fの精度を確かめる。
- **境界**（`test_collinearity_boundary.py`）: 内側は推定でき、外側は`DESIGN_GRAM`で止まる。共線な対の単位を`1e±6`・`1e±12`倍しても、片方だけ倍しても動かない。完全な共線（ダミー変数トラップ・重複列）はn=`1e3`〜`1e6`でも列名つきの`DESIGN_MATRIX_SINGULAR`。
- **Tobit**（`test_near_collinear_tobit.py`）: 境界・単位・再パラメータ化（切片なし・ソルバーを含む）と、同じ`H`・`S`の有理数の厳密値（`engine`の`coordinates.rs`の単体テスト）。

## 6. 性能

QR 1本への統合後、変更前との比較は、OLS・IVが速く（OLS 15〜35%・IV 20〜25%）、FE・FE-IVは同等かやや速い（測定のばらつき約±15%の内側は主張しない）。ピークメモリはOLSのn=100万で最大約14%増（`Q`の分）。GMMはn=100万で時間+10〜30%・ピークメモリ+10%。数値の正は`benchmark_performance.yml`のCI計測値。

## 7. 未対応・限界

- GMMの時間・メモリの最適化。
- GMMのDK・反復GMMは再パラメータ化のみ、準共線のC統計量の専用テストは無い。
- `g = q+1`のクラスターのように共分散が構造的に特異になるケースは、事前検証（`validate_cluster_count_covers_slopes`）が先に弾く。残る`COEFFICIENT_COVARIANCE`は、残差が一部のクラスターにしか乗らない設計で再現でき、丸め誤差級の特異性に依存しないテストで固定している。
- 名前の衝突: within後の`Z̃`と白色化した`Z̃`が同じ記号。
