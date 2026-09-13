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

use crate::error::CommonError;
use crate::linear::ols::{CovType, OlsEstimator, OlsInput};
use crate::panel::common::{PanelDimension, PanelError, group_indices_by_key};
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
/// **可視性について**: 本来はRE内部専用（`ReEstimator::fit`、Issue #195）の実装詳細で
/// `pub(crate)`が適切だが、Issue #195着手前の現時点では非テストコードからの呼び出しが
/// 無く`pub(crate)`だと`dead_code`警告になる。`quasi_demean_column`・`hausman_statistic`
/// （`common.rs`、Issue #173・#174）も同じ理由で`pub`にした前例に倣う。
pub fn swamy_arora_variance_components(
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
}
