//! REの入力データ型（`ReInput`、Issue #192）。
//!
//! `engine`はpolars/PyO3を知らない（`.claude/rules/rust-style.md`「責務分離」）。
//! `engine_pybind`がpolars DataFrameから`y`/`x`/`entity`/`time`を列ごとに抽出し、
//! それらの列を本モジュールの`ReInput::from_columns`に渡す（`FeInput::from_columns`
//! （`fe.rs`、Issue #175）と同型の設計）。
//!
//! `ReInput`自体は準偏差変換前の生データを保持するだけの入れ物であり、`FeInput`と
//! 同じ理由（`quasi_demean_column`が`&[f64]`の列単位で動く設計のため）で`faer::Mat`は
//! 組み立てない（`docs/planning/specs/panel-api-design.md`7.4節）。
//!
//! `time`フィールドの扱いは`FeInput`をそのまま踏襲する（Issue #192のスコープ、
//! `panel-api-design.md`1章）が、RE自身の準偏差変換（7.2節）は**entity方向のみ**
//! （2-way REはv1スコープ外）で`time`を使わない。`ReInput`が`time`を保持する理由は、
//! `RE.fit()`が内部でFE推定を実行してハウスマン検定の比較対象を得る際
//! （2.4節）、「`entity`/`time`/`x`はRE呼び出し時と同一の指定を使う」ため——つまり
//! `ReInput`から`FeInput`相当のデータを組み立て直す際に、`time`を`ReInput`が
//! 既に保持していれば再抽出が不要になる（`ReOptions.time`、1.1節）。この内部FE呼び出し
//! ロジック自体は本Issueのスコープ外（後続issue、タスクコード#195以降）。
//!
//! ## Swamy-Arora分散成分推定（`swamy_arora_variance_components`、Issue #193、7.1節）
//!
//! σ_ε²（idiosyncratic variance）は内部1-way FE推定（`FeEstimator`）のwithin回帰残差を
//! 再利用し、σ_u²（individual variance）はbetween回帰（エンティティ平均への
//! `OlsEstimator::fit(include_intercept=true)`）から求める（RE→FE/OLS→
//! `OlsEstimator`という7.4節の委譲チェーン）。分母（自由度）は`panel-api-design.md`
//! 7.1節の式を手で組み立てず、FE/OLS委譲先が実際に使った`df_resid`相当の値
//! （`FeEstimator::df_resid()`・`OlsInput::nobs()-k()`）をそのまま再利用する——
//! 7.1節の式は`linearmodels`ソースの`nvar`（切片を含む列数）表記をそのまま転記した
//! ものであり、このプロジェクトの`k`規約（傾き係数のみ）では委譲先の値を使えば
//! 自動的に一致する（詳細な導出・数値検証は`swamy_arora_variance_components`関数doc
//! 参照、ユーザー確認済み・2026-09-13）。
//!
//! ## θ計算・準偏差変換（`quasi_demean_transform`、Issue #194、7.2節）
//!
//! `θ_i = 1 - sqrt(σ_ε² / (T_i・σ_u² + σ_ε²))`（`compute_theta`、7.2節の式そのまま）を
//! エンティティごとに計算し、`quasi_demean_column`（`common.rs`、Issue #173）を`y`・
//! 各`x`列に適用する。REはentity方向のみ（2-way REはv1スコープ外、7.2節）のため
//! 不均衡パネルも無条件でサポートする——`T_i`（エンティティごとの観測数）を直接使う
//! この式は教科書レベルで不均衡対応済みで、FEの2-wayのような反復アルゴリズムは
//! 不要（7.2節）。`sigma2_eps`/`sigma2_u`は`swamy_arora_variance_components`の戻り値を
//! そのまま渡す想定だが、この関数自体はその依存を持たない（テストで独立に検証できる
//! ようにするため。`quasi_demean_column`が`θ`の値域を検証しないのと同じ設計）。
//!
//! ## `OlsEstimator`への委譲（`ReEstimator`、Issue #195、7.4節）
//!
//! `ReEstimator::fit`は「Swamy-Arora分散成分推定（`swamy_arora_variance_components`、
//! Issue #193）→θ計算・準偏差変換（`quasi_demean_transform`、Issue #194）→
//! `OlsEstimator::fit`への委譲」の順にパイプラインを実行する（`FeEstimator::fit`と
//! 同型のパターン）。
//!
//! **REはFEと異なり切片を持つ**（モデル`y_it = β0 + x_it'β + u_i + ε_it`、2.1節）ため、
//! 委譲前に切片項の復元が必要になる。単純に`OlsInput::from_columns`の
//! `include_intercept=true`は使えない——それだと自動追加される定数列が
//! 変換されない生の`1.0`のままになってしまう。正しくは、**すべて`1.0`の列を`y`・`x`と
//! 同じ`theta`で`quasi_demean_column`した列**（`(1 - θ_i)`、エンティティごとに異なる）を
//! 明示的に組み立て、それを設計行列の先頭に加えた上で`include_intercept=false`で
//! `OlsEstimator::fit`に渡す（`linearmodels.RandomEffects.fit()`のソースで、`exog`に
//! 含まれる定数列自体も他の説明変数と同じ`quasi_demean`処理を受けていることを確認
//! 済み。手動データでの数値完全一致で検証済み、詳細は`ReEstimator::fit`関数doc参照）。
//! `param_names`は`OlsInput::from_columns_impl`の自動`"const"`命名と同じ並び
//! （`["const", x_names...]`）に揃える。
//!
//! `swamy_arora_variance_components`・`compute_theta`・`quasi_demean_transform`は
//! これで非テストコードからの呼び出しが生まれたため、`pub`から`pub(crate)`に格下げした
//! （Issue #194の実装ノート・rust-reviewer指摘の通り）。

use std::collections::BTreeMap;

use crate::error::CommonError;
use crate::linear::ols::{CovType, OlsEstimator, OlsInput};
use crate::panel::common::{PanelDimension, PanelError, group_indices_by_key, quasi_demean_column};
use crate::panel::fe::{FeCovType, FeEffects, FeEstimator, FeInput};

/// REの被説明変数・説明変数・パネル識別子を保持する入力データ。
///
/// 準偏差変換前の生データを保持するだけの入れ物（`Mat`を組み立てない理由はモジュール
/// doc参照）。フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」）。
/// `from_columns`で構築した後はgetter経由でのみアクセスする。
#[derive(Debug)]
pub struct ReInput {
    /// 被説明変数（長さ`n`、行はパネルの観測順）。
    y: Vec<f64>,
    /// 説明変数（各列は長さ`n`）。準偏差変換前の生の値。
    x: Vec<Vec<f64>>,
    /// 説明変数名。`x`の列と対応する。
    x_names: Vec<String>,
    /// 各行のエンティティID（長さ`n`）。
    entity: Vec<String>,
    /// 各行の時点ID（長さ`n`）。RE自身の準偏差変換では使わない（モジュールdoc参照）が、
    /// 内部FE呼び出し（ハウスマン検定用）に`entity`と同一の扱いで保持する。
    time: Option<Vec<String>>,
    /// 被説明変数名。
    dep_var_name: String,
}

impl ReInput {
    /// 列ごとの`Vec<f64>`/`Vec<String>`（`engine_pybind`がpolars DataFrameから抽出済み）
    /// から`ReInput`を組み立てる。`FeInput::from_columns`と同一の次元検証を行う。
    ///
    /// # Errors
    /// - いずれかの`x_columns`の長さが`y`と一致しない場合は
    ///   `PanelError::Common(CommonError::DimensionMismatch)`
    /// - `entity`の長さが`y`と一致しない場合は
    ///   `PanelError::IdentifierDimensionMismatch { dimension: PanelDimension::Entity, .. }`
    /// - `time`が`Some`で、その長さが`y`と一致しない場合は
    ///   `PanelError::IdentifierDimensionMismatch { dimension: PanelDimension::Time, .. }`
    ///
    /// 準偏差変換の実施・θ計算・分散成分推定・ハウスマン検定は行わない
    /// （いずれも別issueで`fit()`側が担う、`panel-api-design.md`7章）。
    ///
    /// # パニックについて
    /// `x_names.len() != x_columns.len()`の場合は`debug_assert!`でパニックする
    /// （`FeInput::from_columns`と同じ理由: 呼び出し側`engine_pybind`の実装バグでしか
    /// 起こり得ない内部契約であり、実データに起因する`ValidationError`とは性質が異なる）。
    pub fn from_columns(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        entity: &[String],
        time: Option<&[String]>,
        dep_var_name: String,
    ) -> Result<Self, PanelError> {
        debug_assert_eq!(
            x_columns.len(),
            x_names.len(),
            "x_columns and x_names must have the same length"
        );

        for col in x_columns {
            if col.len() != y.len() {
                return Err(CommonError::DimensionMismatch {
                    y_rows: y.len(),
                    x_rows: col.len(),
                }
                .into());
            }
        }

        if entity.len() != y.len() {
            return Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Entity,
                y_rows: y.len(),
                other_rows: entity.len(),
            });
        }

        if let Some(time) = time
            && time.len() != y.len()
        {
            return Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                y_rows: y.len(),
                other_rows: time.len(),
            });
        }

        Ok(Self {
            y: y.to_vec(),
            x: x_columns.to_vec(),
            x_names,
            entity: entity.to_vec(),
            time: time.map(|t| t.to_vec()),
            dep_var_name,
        })
    }

    /// 被説明変数（長さ`n`）。
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    /// 説明変数（各列は長さ`n`）。
    pub fn x(&self) -> &[Vec<f64>] {
        &self.x
    }

    /// 説明変数名。`x()`の列と対応する。
    pub fn x_names(&self) -> &[String] {
        &self.x_names
    }

    /// 各行のエンティティID（長さ`n`）。
    pub fn entity(&self) -> &[String] {
        &self.entity
    }

    /// 各行の時点ID（長さ`n`）。未指定なら`None`。
    pub fn time(&self) -> Option<&[String]> {
        self.time.as_deref()
    }

    /// 被説明変数名。
    pub fn dep_var_name(&self) -> &str {
        &self.dep_var_name
    }

    /// 観測数 n
    pub fn nobs(&self) -> usize {
        self.y.len()
    }
}

/// エンティティ平均（between回帰用）。`group_indices_by_key`（`common.rs`、Issue #193で
/// FE/RE共有に移設）でエンティティを集計し、`y`/各`x`列のエンティティごとの単純平均と、
/// 各エンティティの観測数`T_i`（7.1節の調和平均`t_bar`計算にも使うため、二重集計を避けて
/// ここで一緒に返す）を返す。
///
/// 戻り値の各`Vec`はエンティティのユニークID辞書順（`group_indices_by_key`のキー順）で
/// 揃っている。entity IDの文字列自体は返さない（between回帰・`t_bar`計算のどちらも
/// 数値だけで足りるため）。
fn entity_means(
    y: &[f64],
    x: &[Vec<f64>],
    entity: &[String],
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let groups = group_indices_by_key(entity);
    let k = x.len();
    let mut y_means = Vec::with_capacity(groups.len());
    let mut x_means: Vec<Vec<f64>> = vec![Vec::with_capacity(groups.len()); k];
    let mut t = Vec::with_capacity(groups.len());
    for indices in groups.values() {
        let t_i = indices.len() as f64;
        y_means.push(indices.iter().map(|&i| y[i]).sum::<f64>() / t_i);
        for (j, x_means_j) in x_means.iter_mut().enumerate() {
            x_means_j.push(indices.iter().map(|&i| x[j][i]).sum::<f64>() / t_i);
        }
        t.push(t_i);
    }
    (y_means, x_means, t)
}

/// Swamy-Arora法で分散成分（σ_ε²・σ_u²）を推定する（Issue #193、7.1節）。
///
/// - **σ_ε²（idiosyncratic variance）**: 内部で1-way FE推定
///   （`FeEstimator::fit`、`FeCovType::Classical`固定——`cov_type`は残差そのものには
///   影響しないため）を呼び、そのwithin回帰残差平方和とFE自身の`df_resid()`
///   （`n - k - n_entities`）から`SSR / df_resid`として求める（7.4節「σ_ε²の推定は
///   FEのwithin回帰の残差分散をそのまま利用する」、RE→FE→`OlsEstimator`の委譲チェーン）。
/// - **σ_u²（individual variance）**: between回帰（エンティティ平均への
///   `OlsEstimator::fit(include_intercept=true)`。REは切片を持つためFEと異なり
///   between回帰にも切片が要る）のSSRと、その`df_resid`
///   （`OlsInput::nobs() - OlsInput::k()`、`k()`は切片込みの設計行列の列数）から、
///   調和平均`t_bar = n_entities / Σ(1/T_i)`を使う標準式
///   `max(0, ssr/df_resid - σ_ε²/t_bar)`で求める。
///
/// **`k`規約についての注記（ユーザー確認済み、2026-09-13）**: `panel-api-design.md`
/// 7.1節に書かれている式（σ_ε²分母`n-k-n_entities+1`、σ_u²分母`n_entities-k`）は、
/// `linearmodels`ソースの`nvar`（切片を含む列数）表記をそのまま転記したものである。
/// このプロジェクトのFE/OLSの`k`規約（傾き係数のみ、切片を含まない）では、分母の
/// 「+1」「-1」を式に手で足し引きする必要はない——FE/OLS双方の委譲先が返す実際の
/// `df_resid`相当の値（`FeEstimator::df_resid()`・`OlsInput::nobs()-k()`）をそのまま
/// 使えば自動的に一致する（`linearmodels`との数値完全一致を複数の乱数・手動データで
/// 実地検証済み）。このためFE/OLSへの委譲を経ず`n`・`n_entities`・`k`から直接式を
/// 組み立てる実装はしない（委譲先の状態を信頼できるソースとして再利用する）。
///
/// # Errors
/// - 内部FE推定が失敗した場合（singleton・分散ゼロ説明変数・自由度不足等）は、その
///   `PanelError`（FE用バリアント）をそのまま伝播する。
/// - between回帰が失敗した場合（エンティティ数が説明変数の数以下等）は
///   `PanelError::BetweenRegressionFailed`。
///
pub(crate) fn swamy_arora_variance_components(
    input: &ReInput,
    confidence_level: f64,
) -> Result<(f64, f64), PanelError> {
    // faerのグローバル並列度をPar::Seqに固定する（Issue #283、`crate::parallelism`。
    // 委譲先の`FeEstimator::fit`/`OlsEstimator::fit`自身も呼ぶが、`cargo test -p engine`で
    // この関数を直接叩く経路との統一のためここでも呼ぶ、`engine/src/panel/CLAUDE.md`
    // 「faerのグローバル並列度」参照）。
    crate::parallelism::ensure_serial();

    // σ_ε²: 内部1-way FE推定のwithin回帰残差を再利用する（7.4節）。
    let fe_input = FeInput::from_columns(
        input.y(),
        input.x(),
        input.x_names().to_vec(),
        input.entity(),
        None,
        input.dep_var_name().to_string(),
    )
    .expect(
        "ReInput::from_columns already validated the same dimension contract \
         (y/x/entity lengths) that FeInput::from_columns requires",
    );
    let fe = FeEstimator::fit(
        fe_input,
        FeEffects::OneWay,
        FeCovType::Classical,
        confidence_level,
    )?;

    let fe_residuals = fe.estimator().residuals();
    let ssr_within: f64 = (0..fe_residuals.nrows())
        .map(|i| {
            let r = *fe_residuals.get(i, 0);
            r * r
        })
        .sum();
    let sigma2_eps = ssr_within / fe.df_resid() as f64;

    // σ_u²: between回帰（エンティティ平均、切片あり）。
    let (y_means, x_means, t) = entity_means(input.y(), input.x(), input.entity());
    let n_entities = y_means.len();

    let between_input = OlsInput::from_columns(
        &y_means,
        &x_means,
        input.x_names().to_vec(),
        true,
        input.dep_var_name().to_string(),
    )
    .map_err(|source| PanelError::BetweenRegressionFailed { source })?;
    let between = OlsEstimator::fit(between_input, CovType::Classical, confidence_level)
        .map_err(|source| PanelError::BetweenRegressionFailed { source })?;

    let between_residuals = between.residuals();
    let ssr_between: f64 = (0..between_residuals.nrows())
        .map(|i| {
            let r = *between_residuals.get(i, 0);
            r * r
        })
        .sum();
    let df_resid_between = between.input().nobs() - between.input().k();

    let t_bar = n_entities as f64 / t.iter().map(|t_i| 1.0 / t_i).sum::<f64>();
    let sigma2_u = (ssr_between / df_resid_between as f64 - sigma2_eps / t_bar).max(0.0);

    Ok((sigma2_eps, sigma2_u))
}

/// θ（準偏差変換の重み）を計算する（Issue #194、7.2節）。
///
/// `θ_i = 1 - sqrt(σ_ε² / (T_i・σ_u² + σ_ε²))`。`T_i`はエンティティ`i`の観測数
/// （`group_indices_by_key`で集計する）。不均衡パネルもこの式で無条件にサポートする
/// （`T_i`が式に直接入るため、教科書レベルで不均衡対応済み。7.2節）。
///
/// `σ_ε²`/`σ_u²`の値域は検証しない（`quasi_demean_column`が`θ`の値域を検証しないのと
/// 同じ設計判断——呼び出し側が`swamy_arora_variance_components`の戻り値を渡す限り
/// `σ_ε²>0`・`σ_u²>=0`は保証されるが、この関数自体はその前提を強制しない）。
fn compute_theta(entity: &[String], sigma2_eps: f64, sigma2_u: f64) -> BTreeMap<String, f64> {
    group_indices_by_key(entity)
        .into_iter()
        .map(|(id, indices)| {
            let t_i = indices.len() as f64;
            let theta_i = 1.0 - (sigma2_eps / (t_i * sigma2_u + sigma2_eps)).sqrt();
            (id.to_string(), theta_i)
        })
        .collect()
}

/// θ計算・準偏差変換（Issue #194、7.2節・7.4節）。`compute_theta`で求めたθを
/// `quasi_demean_column`で`y`・各`x`列に適用する（FEの`within_transform_one_way`と
/// 同型のパターン——FEはθ=1固定、REはエンティティごとに異なるθを使う点だけが異なる）。
///
/// 戻り値は`(theta, y_transformed, x_transformed)`。`theta`も返す理由: Issue #195で
/// `OlsEstimator::fit(include_intercept=false)`への委譲時、切片項を復元するために
/// 定数列（すべて1.0）にも同じ`theta`で`quasi_demean_column`を適用する必要があり
/// （REは切片を持つためFEと異なりこの復元が要る、7.4節）、`theta`の再計算を避けるため
/// 呼び出し側に渡しておく。
pub(crate) fn quasi_demean_transform(
    input: &ReInput,
    sigma2_eps: f64,
    sigma2_u: f64,
) -> (BTreeMap<String, f64>, Vec<f64>, Vec<Vec<f64>>) {
    let theta = compute_theta(input.entity(), sigma2_eps, sigma2_u);
    let y = quasi_demean_column(input.y(), input.entity(), &theta);
    let x = input
        .x()
        .iter()
        .map(|col| quasi_demean_column(col, input.entity(), &theta))
        .collect();
    (theta, y, x)
}

/// REの推定結果。Swamy-Arora分散成分推定（Issue #193）→θ計算・準偏差変換
/// （Issue #194）→`OlsEstimator::fit`への委譲（Issue #195）というパイプラインで
/// `θ変換済み`データの係数推定（`β̂`）のみをスコープとする（モジュールdoc
/// 「`OlsEstimator`への委譲」参照。`FeEstimator`が#178時点で係数推定のみを
/// スコープにしたのと同型——標準誤差等のFE/RE固有の再計算は別issue）。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」）。
#[derive(Debug)]
pub struct ReEstimator {
    input: ReInput,
    estimator: OlsEstimator,
}

impl ReEstimator {
    /// `input`からSwamy-Arora分散成分（σ_ε²・σ_u²）を推定し、θ計算・準偏差変換した
    /// `y`・`x`（切片復元用に同じθで変換した定数列を含む）を`OlsEstimator::fit`に
    /// 委譲してREを推定する。
    ///
    /// パイプライン: `swamy_arora_variance_components`（Issue #193）→
    /// `quasi_demean_transform`（Issue #194）→ 定数列の準偏差変換・設計行列への追加
    /// （モジュールdoc参照）→ `OlsEstimator::fit`への委譲（`include_intercept=false`固定。
    /// 変換済みデータに既に切片相当の列を含めているため、FE同様これ以上の自動追加は
    /// 不要）。
    ///
    /// `OlsEstimator::fit`自体は`CovType::Classical`固定で呼ぶ（`cov_type`対応は
    /// 別issue、`FeEstimator::fit`が委譲先を常に`Classical`で呼ぶのと同じ理由——
    /// `β̂`・残差の取得のみが目的で、`cov_type`ごとの標準誤差は将来REが独自に
    /// 計算し直す設計になる見込みのため）。
    ///
    /// # Errors
    /// - Swamy-Arora分散成分推定が失敗した場合（内部FE推定のsingleton検出・分散ゼロ・
    ///   自由度不足、またはbetween回帰の失敗）は、その`PanelError`をそのまま伝播する。
    /// - 準偏差変換済みデータへの委譲が失敗した場合（観測数不足・特異行列等）は
    ///   `PanelError::QuasiDemeanedRegressionFailed`。
    pub fn fit(input: ReInput, confidence_level: f64) -> Result<Self, PanelError> {
        // faerのグローバル並列度をPar::Seqに固定する（Issue #283、`crate::parallelism`。
        // 委譲先の`FeEstimator::fit`/`OlsEstimator::fit`自身も呼ぶが、`cargo test -p engine`
        // で`ReEstimator::fit`を直接叩く経路との統一のためここでも呼ぶ、
        // `engine/src/panel/CLAUDE.md`「faerのグローバル並列度」参照）。
        crate::parallelism::ensure_serial();

        let (sigma2_eps, sigma2_u) = swamy_arora_variance_components(&input, confidence_level)?;
        let (theta, y, x) = quasi_demean_transform(&input, sigma2_eps, sigma2_u);

        // 切片復元用の定数列（すべて1.0）を、y/xと同じthetaで準偏差変換する
        // （モジュールdoc「`OlsEstimator`への委譲」参照。`OlsInput::from_columns`の
        // `include_intercept=true`は使えない——それだと変換されない生の`1.0`列に
        // なってしまう）。
        let const_column = vec![1.0; input.nobs()];
        let const_transformed = quasi_demean_column(&const_column, input.entity(), &theta);

        let mut x_all = Vec::with_capacity(x.len() + 1);
        x_all.push(const_transformed);
        x_all.extend(x);

        let mut param_names = Vec::with_capacity(input.x_names().len() + 1);
        param_names.push("const".to_string());
        param_names.extend(input.x_names().iter().cloned());

        // `OlsInput::from_columns`が返しうる`LeastSquaresError::Common(DimensionMismatch)`は
        // ここでは理論上到達不能: `y`/`x_all`はどちらも`quasi_demean_column`が
        // `input.y()`/`input.x()`（`ReInput::from_columns`が既に同じ長さであることを
        // 検証済み）と`const_column`（`input.nobs()`で長さを揃えている）から1対1で
        // 生成した同じ長さの列であり、この関数内で長さがずれる操作をしていない
        // （`FeEstimator::fit`の同種のコメントと同じ判断）。
        let ols_input = OlsInput::from_columns(
            &y,
            &x_all,
            param_names,
            false,
            input.dep_var_name().to_string(),
        )
        .map_err(|source| PanelError::QuasiDemeanedRegressionFailed { source })?;
        let estimator = OlsEstimator::fit(ols_input, CovType::Classical, confidence_level)
            .map_err(|source| PanelError::QuasiDemeanedRegressionFailed { source })?;

        Ok(Self { input, estimator })
    }

    /// 準偏差変換前の入力データ。
    pub fn input(&self) -> &ReInput {
        &self.input
    }

    /// 準偏差変換済みデータに対する`OlsEstimator`本体。`params()`の先頭が切片
    /// （`param_names()[0] == "const"`）、以降が`input().x_names()`と同じ並びの
    /// 傾き係数。
    ///
    /// **係数（`params()`）は正しいRE推定量だが、標準誤差・t値・p値・信頼区間・
    /// F統計量・調整済みR²・AIC/BICはこの時点では正しくない**（モジュールdoc参照。
    /// RE固有の自由度・`cov_type`補正が別issueで入るまでの暫定値、`FeEstimator`の
    /// #178時点と同じ扱い）。
    pub fn estimator(&self) -> &OlsEstimator {
        &self.estimator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn from_columns_builds_input_without_time() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let x1 = vec![10.0, 20.0, 30.0, 40.0];
        let entity = strings(&["a", "a", "b", "b"]);

        let input = ReInput::from_columns(
            &y,
            std::slice::from_ref(&x1),
            vec!["x1".to_string()],
            &entity,
            None,
            "y".to_string(),
        )
        .unwrap();

        assert_eq!(input.y(), &y);
        assert_eq!(input.x(), &[x1]);
        assert_eq!(input.x_names(), &["x1".to_string()]);
        assert_eq!(input.entity(), entity.as_slice());
        assert_eq!(input.time(), None);
        assert_eq!(input.dep_var_name(), "y");
        assert_eq!(input.nobs(), 4);
    }

    #[test]
    fn from_columns_builds_input_with_time() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["2020", "2021", "2020", "2021"]);

        let input = ReInput::from_columns(
            &y,
            &[vec![1.0, 1.0, 1.0, 1.0]],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".to_string(),
        )
        .unwrap();

        assert_eq!(input.time(), Some(time.as_slice()));
    }

    #[test]
    fn from_columns_with_no_regressors_succeeds() {
        // OLSと異なりREは説明変数0個でも`ReInput`自体は構築できる（推定可能性の検証は
        // `fit()`側の責務、`FeInput`と同じ層分け）。
        let y = [1.0, 2.0];
        let entity = strings(&["a", "b"]);

        let input = ReInput::from_columns(&y, &[], vec![], &entity, None, "y".to_string()).unwrap();

        assert!(input.x().is_empty());
        assert!(input.x_names().is_empty());
    }

    #[test]
    fn from_columns_returns_dimension_mismatch_on_mismatched_x_column_length() {
        let y = [1.0, 2.0, 3.0];
        let x1 = vec![10.0, 20.0]; // yより短い
        let entity = strings(&["a", "b", "c"]);

        let result = ReInput::from_columns(
            &y,
            &[x1],
            vec!["x1".to_string()],
            &entity,
            None,
            "y".to_string(),
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::Common(CommonError::DimensionMismatch {
                y_rows: 3,
                x_rows: 2,
            })
        );
    }

    #[test]
    fn from_columns_returns_identifier_dimension_mismatch_for_entity() {
        let y = [1.0, 2.0, 3.0];
        let entity = strings(&["a", "b"]); // yより短い

        let result = ReInput::from_columns(&y, &[], vec![], &entity, None, "y".to_string());

        assert_eq!(
            result.unwrap_err(),
            PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Entity,
                y_rows: 3,
                other_rows: 2,
            }
        );
    }

    #[test]
    fn from_columns_returns_identifier_dimension_mismatch_for_time() {
        let y = [1.0, 2.0, 3.0];
        let entity = strings(&["a", "b", "c"]);
        let time = strings(&["2020", "2021"]); // yより短い

        let result = ReInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".to_string());

        assert_eq!(
            result.unwrap_err(),
            PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                y_rows: 3,
                other_rows: 2,
            }
        );
    }

    #[test]
    fn from_columns_with_zero_observations_succeeds() {
        // n=0（y/entity/timeすべて空）でも次元は一致しているため`ReInput`自体の構築は
        // 成功する（`FeInput`の同名テストと同じ境界値の意図、
        // `.claude/rules/testing-policy.md`「境界値・悪条件」）。
        let input =
            ReInput::from_columns(&[], &[], vec![], &[], Some(&[]), "y".to_string()).unwrap();

        assert_eq!(input.nobs(), 0);
        assert_eq!(input.time(), Some([].as_slice()));
    }

    #[test]
    #[should_panic(expected = "x_columns and x_names must have the same length")]
    fn from_columns_panics_on_mismatched_names_arity() {
        let y = [1.0, 2.0];
        let entity = strings(&["a", "b"]);
        let _ = ReInput::from_columns(
            &y,
            &[vec![1.0, 2.0]],
            vec![], // x_columnsは1列だがx_namesは0個
            &entity,
            None,
            "y".to_string(),
        );
    }

    // ── swamy_arora_variance_components ─────────────────────────────────────

    #[test]
    fn swamy_arora_variance_components_matches_linearmodels_reference() {
        // 手動データ（不均衡パネル、entity a: T=3, b: T=2, c: T=2）。
        // `linearmodels.RandomEffects`（Python）で実地検証済みの値と比較する
        // （`RandomEffects(y, [const, x1]).fit().variance_decomposition`）。
        // between回帰は切片+傾き1個の2パラメータのため、`n_entities > k+1`（3章参照）を
        // 満たすには最低3エンティティが必要（2エンティティだと`neffects-nvar=0`で
        // 除算不能になることを`linearmodels`自身でも実地確認済み）。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let (sigma2_eps, sigma2_u) = swamy_arora_variance_components(&input, 0.95).unwrap();

        assert!(
            (sigma2_eps - 1.132_352_941_176_471_5).abs() < 1e-9,
            "sigma2_eps = {sigma2_eps}"
        );
        assert!(
            (sigma2_u - 6.537_912_784_161_284).abs() < 1e-9,
            "sigma2_u = {sigma2_u}"
        );
    }

    #[test]
    fn swamy_arora_variance_components_clips_negative_sigma2_u_to_zero() {
        // entity間のy平均のばらつきが、within回帰から推定したσ_ε²/t_barに対して
        // 十分小さいため、素朴な式ではσ_u²が負になるが`max(0, ...)`で0にクリップされる
        // （`linearmodels`と同じ挙動、7.3節のハウスマン統計量の負値と同型の
        // 「有限標本でのPSD仮定崩れ」）。エンティティ間のx1平均はわずかに異なる値にし、
        // between回帰の設計行列が特異にならないようにする。`linearmodels`で実地検証済み
        // （`variance_decomposition["Effects"] == 0.0`）。
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c"]);
        let x1 = vec![1.0, 2.0, 3.0, 1.2, 2.1, 2.9, 0.9, 2.2, 3.1];
        let y = [2.0, 4.0, 6.0, 2.3, 4.1, 5.9, 1.8, 4.3, 6.2];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let (sigma2_eps, sigma2_u) = swamy_arora_variance_components(&input, 0.95).unwrap();

        assert!(
            (sigma2_eps - 0.005_868_778_280_543_04).abs() < 1e-9,
            "sigma2_eps = {sigma2_eps}"
        );
        assert_eq!(sigma2_u, 0.0);
    }

    #[test]
    fn swamy_arora_variance_components_propagates_fe_singleton_error() {
        // entity "c"は1観測のみ（singleton）。内部FE推定（1-way）の
        // `PanelError::SingletonGroup`がそのまま伝播することを確認する。
        let entity = strings(&["a", "a", "c"]);
        let x1 = vec![1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 3.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = swamy_arora_variance_components(&input, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn swamy_arora_variance_components_returns_between_regression_failed_when_entities_are_insufficient()
     {
        // between回帰は切片+傾き1個で2パラメータ。エンティティ数が2つだけだと
        // `n_entities <= k`（`OlsEstimator::fit`自身の`n<=k`検証）で失敗する。
        // singletonにならないよう各エンティティは2観測以上にする。
        let entity = strings(&["a", "a", "b", "b"]);
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 2.0, 3.0, 5.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = swamy_arora_variance_components(&input, 0.95);

        assert!(matches!(
            result,
            Err(PanelError::BetweenRegressionFailed { .. })
        ));
    }

    // ── compute_theta / quasi_demean_transform ──────────────────────────────

    #[test]
    fn compute_theta_matches_reference_formula_for_unbalanced_panel() {
        // entity a: T=3, b: T=2, c: T=1（不均衡パネル）。`linearmodels`のθ計算式
        // （`RandomEffects.fit()`内の`theta = 1 - sqrt(sigma2_e/(t*sigma2_u+sigma2_e))`）と
        // 数値完全一致することを実地検証済みの値（`swamy_arora_variance_components`の
        // 参照テストと同じσ_ε²・σ_u²を使う）。
        let entity = strings(&["a", "a", "a", "b", "b", "c"]);
        let sigma2_eps = 0.064_516_129_032_258_03;
        let sigma2_u = 7.429_453_144_752_303;

        let theta = compute_theta(&entity, sigma2_eps, sigma2_u);

        assert!((theta["a"] - 0.946_276_110_063_922_5).abs() < 1e-12);
        assert!((theta["b"] - 0.934_249_367_520_359).abs() < 1e-12);
        assert!((theta["c"] - 0.907_214_909_247_309_4).abs() < 1e-12);
    }

    #[test]
    fn compute_theta_matches_reference_formula_for_balanced_panel() {
        // バランスパネル（全エンティティT=2）、σ_ε²=σ_u²=1.0という単純な数値で
        // θ = 1 - sqrt(1/3)を確認する（手計算で検算可能な境界値）。
        let entity = strings(&["a", "a", "b", "b"]);

        let theta = compute_theta(&entity, 1.0, 1.0);

        let expected = 1.0 - (1.0_f64 / 3.0).sqrt();
        assert!((theta["a"] - expected).abs() < 1e-12);
        assert!((theta["b"] - expected).abs() < 1e-12);
    }

    #[test]
    fn compute_theta_is_zero_when_sigma2_u_is_zero() {
        // σ_u²=0（individual varianceが無い、REがpooled OLSに退化するケース）では
        // θ_i = 1 - sqrt(σ_ε²/σ_ε²) = 0 になり、`quasi_demean_column`が実質的に
        // 何も変換しない（プーリングOLSと同じ設計行列になる、7.2節・
        // `quasi_demean_column_with_theta_zero_is_identity`と対応する不変条件）。
        // rust-reviewer指摘（Issue #194）: この退化ケースをフィット実装（#195）より前に
        // 固定しておく。
        let entity = strings(&["a", "a", "b", "b", "b"]);

        let theta = compute_theta(&entity, 2.5, 0.0);

        assert_eq!(theta["a"], 0.0);
        assert_eq!(theta["b"], 0.0);
    }

    #[test]
    fn quasi_demean_transform_applies_computed_theta_to_y_and_x() {
        // `compute_theta_matches_reference_formula_for_unbalanced_panel`と同じデータ・
        // σ_ε²・σ_u²。`quasi_demean_column`自体の正しさは`common.rs`側で既に検証済み
        // のため、ここでは「θの計算結果が正しくy/xの各列に適用されているか」の配線を
        // 確認する。期待値はPythonで手計算した参照値。
        let entity = strings(&["a", "a", "a", "b", "b", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();
        let sigma2_eps = 0.064_516_129_032_258_03;
        let sigma2_u = 7.429_453_144_752_303;

        let (theta, y_t, x_t) = quasi_demean_transform(&input, sigma2_eps, sigma2_u);

        assert!((theta["a"] - 0.946_276_110_063_922_5).abs() < 1e-12);

        let expected_y = [
            -1.415_955_180_298_305_5,
            -0.415_955_180_298_305_47,
            2.584_044_819_701_694_5,
            0.058_880_376_076_948_51,
            1.058_880_376_076_948_5,
            0.556_710_544_516_143_1,
        ];
        for (actual, expected) in y_t.iter().zip(expected_y.iter()) {
            assert!((actual - expected).abs() < 1e-9, "{actual} vs {expected}");
        }

        let expected_x1 = [
            -1.207_977_590_149_152_7,
            -0.207_977_590_149_152_74,
            1.792_022_409_850_847_3,
            -0.335_623_418_800_897_5,
            0.664_376_581_199_102_5,
            0.463_925_453_763_453_2,
        ];
        for (actual, expected) in x_t[0].iter().zip(expected_x1.iter()) {
            assert!((actual - expected).abs() < 1e-9, "{actual} vs {expected}");
        }
    }

    #[test]
    fn quasi_demean_transform_supports_unbalanced_panel_without_error() {
        // REはentity方向のみ（7.2節）のため、FEの2-wayと異なりバランスパネルを
        // 要求しない。singletonエンティティ（T=1）を含む不均衡パネルでも
        // （`swamy_arora_variance_components`と違い内部でFE推定を呼ばないため）
        // エラーにならず変換できることを確認する。
        let entity = strings(&["a", "a", "a", "b", "b", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let (theta, y_t, x_t) = quasi_demean_transform(&input, 0.1, 1.0);

        assert_eq!(theta.len(), 3);
        assert_eq!(y_t.len(), 6);
        assert_eq!(x_t[0].len(), 6);
    }

    // ── ReEstimator::fit ─────────────────────────────────────────────────

    #[test]
    fn re_estimator_fit_matches_linearmodels_reference() {
        // `swamy_arora_variance_components_matches_linearmodels_reference`と同じ
        // データ（entity a: T=3, b: T=2, c: T=2）。`linearmodels.RandomEffects`
        // （Python、`exog=[const, x1]`）で実地検証済みの`params`・`resids`と比較する。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let re = ReEstimator::fit(input, 0.95).unwrap();

        assert_eq!(
            re.estimator().input().param_names(),
            &["const".to_string(), "x1".to_string()]
        );
        assert!((*re.estimator().params().get(0, 0) - 2.224_977_596_254_189_6).abs() < 1e-9);
        assert!((*re.estimator().params().get(1, 0) - 1.400_245_546_585_47).abs() < 1e-9);

        let expected_resids = [
            0.007_456_619_017_130_906,
            -0.392_788_927_568_339_16,
            -0.193_280_020_739_278_4,
            0.983_357_826_800_407_3,
            0.583_112_280_214_937_3,
            -1.843_693_205_551_097,
            0.756_061_247_863_432_8,
        ];
        for (i, expected) in expected_resids.iter().enumerate() {
            let actual = *re.estimator().residuals().get(i, 0);
            assert!((actual - expected).abs() < 1e-9, "{actual} vs {expected}");
        }
    }

    #[test]
    fn re_estimator_fit_propagates_fe_singleton_error_from_variance_components() {
        // entity "c"は1観測のみ（singleton）。`swamy_arora_variance_components`内部の
        // FE推定が`PanelError::SingletonGroup`を返し、そのまま伝播することを確認する
        // （`swamy_arora_variance_components_propagates_fe_singleton_error`と同じ配線）。
        let entity = strings(&["a", "a", "c"]);
        let x1 = vec![1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 3.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = ReEstimator::fit(input, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn re_estimator_fit_propagates_between_regression_failed_error() {
        // `swamy_arora_variance_components_returns_between_regression_failed_when_
        // entities_are_insufficient`と同じデータ（n_entities=2, k=1で between回帰が
        // 特異になる）。`ReEstimator::fit`が分散成分推定の失敗をそのまま伝播することを
        // 確認する。
        let entity = strings(&["a", "a", "b", "b"]);
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 2.0, 3.0, 5.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = ReEstimator::fit(input, 0.95);

        assert!(matches!(
            result,
            Err(PanelError::BetweenRegressionFailed { .. })
        ));
    }

    #[test]
    fn re_estimator_fit_propagates_invalid_confidence_level_error() {
        // `confidence_level`の範囲チェック（`(0, 1)`の範囲外）は、
        // `swamy_arora_variance_components`内部の最初の委譲（`FeEstimator::fit`）が
        // 真っ先に検証するため、`QuasiDemeanedRegressionFailed`ではなく
        // `WithinRegressionFailed`として伝播する（`FeEstimator::fit`自身の
        // `fe_estimator_fit_propagates_invalid_confidence_level_error`と同型）。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = ReEstimator::fit(input, 1.5);

        assert_eq!(
            result.unwrap_err(),
            PanelError::WithinRegressionFailed {
                source: crate::linear::common::LeastSquaresError::Common(
                    CommonError::InvalidConfidenceLevel {
                        confidence_level: 1.5,
                    }
                ),
            }
        );
    }
}
