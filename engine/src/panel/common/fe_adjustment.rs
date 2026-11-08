//! FE固有の補正（フルレバレッジ・fixest流のK・2グループ退化の検査）。

use super::dummy_leverage::dummy_leverage;
use super::error::PanelError;
use super::validation::covers_every_combination;
use crate::shared::group_codes::GroupCodes;

/// 固定効果の実効パラメータ数`neffects`（`fe-spec.md`3.2節）。`n_groups`は`effects`の列ごとの水準数
/// （`effects`の順）で、水準数の合計から、2つ目以降の効果が1つずつ重複させる定数項ぶん
/// （`n_groups.len() - 1`）を引く。1-wayは`n_groups[0]`、2-wayは`N + T - 1`。fixestと同じ数え方で、
/// 連結成分の数やネストによるランク落ちは引かない。
pub(crate) fn absorbed_parameters(n_groups: &[usize]) -> usize {
    n_groups
        .iter()
        .sum::<usize>()
        .saturating_sub(n_groups.len().saturating_sub(1))
}

/// LSDV相当のフルレバレッジ`h_ii_full`（HC2/HC3用、`fe.rs`のモジュールdoc「`cov_type`対応」の
/// 導出参照）。分割回帰（Frisch-Waugh-Lovell）のレバレッジ分解則により、固定効果ダミーを
/// 明示的に含めた設計行列でのレバレッジは、ダミーのみの回帰のレバレッジ（`h_dummy`）と
/// within変換後のレバレッジ（`h_within`）の和になる（fixestの`vcov="HC2"`/`"HC3"`と数値一致を
/// 1-way・2-way双方で確認済み）。
///
/// `h_dummy`は、1-wayが`1/T_i`、全組合せがそろうバランスの2-wayが`1/T_i + 1/N_t − 1/n`（グループサイズ
/// `T_i`/`N_t`はコードの`counts()`から引く）。組合せが欠ける不均衡の2-wayは閉形式が無く、水準の少ない
/// 方の効果の水準数の密な行列を解く（`dummy_leverage`）。
///
/// # Errors
/// 不均衡の2-wayで、水準の少ない方の効果の水準数が上限を超えるとき
/// `PanelError::LeverageDimensionTooLarge`。
pub(super) fn leverage_full(
    h_within: &[f64],
    entity: &GroupCodes,
    time: Option<&GroupCodes>,
    n: usize,
) -> Result<Vec<f64>, PanelError> {
    let entity_codes = entity.codes();
    let entity_sizes = entity.counts();
    Ok(match time {
        None => (0..n)
            .map(|i| 1.0 / (entity_sizes[entity_codes[i]] as f64) + h_within[i])
            .collect(),
        // 全組合せが過不足なく1回ずつ現れる（`covers_every_combination`）ときだけ閉形式を使う。
        Some(time) if covers_every_combination(entity, time) => {
            let time_codes = time.codes();
            let time_sizes = time.counts();
            (0..n)
                .map(|i| {
                    1.0 / (entity_sizes[entity_codes[i]] as f64)
                        + 1.0 / (time_sizes[time_codes[i]] as f64)
                        - 1.0 / (n as f64)
                        + h_within[i]
                })
                .collect()
        }
        Some(time) => dummy_leverage(entity, time)?
            .into_iter()
            .zip(h_within)
            .map(|(dummy, within)| dummy + within)
            .collect(),
    })
}

/// FE次元（`entity`または`time`）の各値が`cluster`上でちょうど1つの値にしか対応しないか
/// （＝`cluster`がその次元と同じか、それを包含するより粗い分割か）を判定する
/// （fixestの`ssc_compute_K`が言う「FEがクラスター変数にネストしている」の定義。
/// `fe_cluster_k_correction`のdocコメント参照）。元は`entity_nested_within_cluster`
/// という1-way専用の名前だったが、2-way FEでtime次元にも同じ判定を適用する必要が
/// あるため汎用化した（ロジック自体は無変更）。
fn fixef_dimension_nested_within_cluster(dim: &GroupCodes, cluster: &GroupCodes) -> bool {
    // FE次元の各水準が最初に対応したクラスターコード（未出現は`usize::MAX`）。
    let mut mapping = vec![usize::MAX; dim.n_groups()];
    for (&d, &c) in dim.codes().iter().zip(cluster.codes()) {
        if mapping[d] == usize::MAX {
            mapping[d] = c;
        } else if mapping[d] != c {
            return false;
        }
    }
    true
}

/// fixestの`ssc_compute_K`（既定`K.fixef="nonnested"`・`K.exact=FALSE`）を移植した、
/// cluster小標本補正の`K`（`panel_cluster_cov_params`の`(n-1)/(n-K)`の`K`）計算。
///
/// fixest 0.14.2のRソース（`Rscript -e 'cat(deparse(fixest:::ssc_compute_K), sep="\n")'`）と
/// 実地数値実験（devcontainer内のfixest 0.14.2、実装時）で確定した式:
///
/// FE次元（1-wayは`entity`のみ、2-wayは`entity`+`time`）それぞれについて、その次元が
/// クラスター変数に「ネスト」している（＝各水準がクラスターの単一の値にしか対応しない、
/// `fixef_dimension_nested_within_cluster`で判定）かを調べ、ネストした次元の生の水準数
/// （`n_entities`/`n_periods`、`df_model`に含まれる冗長性補正前の値）の合計を
/// `nested_size_sum`、ネストした次元の数を`count_nested`、FE次元の総数を`m`（1か2）とする:
///
/// - `count_nested == 0`（どの次元もネストしていない、Stataの`xtreg,fe`型）:
///   `K = df_model`（固定効果ダミーをフルカウント）
/// - `count_nested == m`（全次元がネスト。1-way・cluster=entityがこの既定ケース）:
///   `K = df_model - nested_size_sum + m`
/// - それ以外（2-way FEで一部の次元だけネスト）:
///   `K = df_model - (nested_size_sum - count_nested)`
///
/// 最後にfixest自身の安全弁`K = max(K, k + 1)`（`k`は傾き係数の数。Rソースの
/// `K = max(K, length(object$coefficients) + 1)`）を適用する——fixestのRソースを
/// 忠実に移植する方針（1章）のため実装しているが、**このプロジェクトの`FeEstimator::fit`が
/// 保証する前提（`n_entities>=1`・2-wayなら`n_periods>=1`、および`df_model = k + neffects`の
/// 定義）の下では、3分岐のどのケースでもこのフロアは実質的にno-op（`raw_k`が既に
/// `k+1`以上）であることを代数的に確認済み**:
/// - `count_nested == m`（全次元ネスト）: `nested_size_sum`はネストした全次元の生サイズの
///   合計で、`df_model`の`neffects`部分もちょうど同じ次元から`Σsize - (m-1)`として
///   構成される（`neffects`の定義、`fe.rs`モジュールdoc「自由度とF統計量」参照）ため、
///   `K = df_model - nested_size_sum + m = k + (Σsize - (m-1)) - Σsize + m = k + 1`が
///   **恒等的に**成り立つ（1-way・2-way両次元ネストのどちらでも）。フロアと厳密に一致する
///   だけで、フロアが値を持ち上げる場面ではない。
/// - 部分ネスト・ネストなし（2-way限定）: ネストしていない側の次元の生サイズが
///   `df_model`にそのまま残るため、`K`はその生サイズの分だけ`k+1`を上回る
///   （`n_entities`/`n_periods`はいずれもパネルとして成立する以上`>=1`、実務上は
///   ほぼ常に`>=2`）。
///
/// 上記のため、`fit()`経由の統合テストではこのフロアの分岐（`raw_k < k+1`になる入力）を
/// 実際には構成できない。フロア自体の検証は`fe_cluster_k_correction`を直接呼ぶ
/// ユニットテスト（`fe_cluster_k_correction_floor_is_a_no_op_under_realistic_inputs`）で、
/// 上記の恒等式そのものを回帰ガードする（`.claude/rules/rust-style.md`「テスト」の
/// 「理論上到達不能な経路は`Result`化しdocで理由を明記すればカバレッジ対象外でよい」
/// 方針と同型——ここでは`Result`ではなく`usize`の恒等式だが、同じ考え方で
/// 「なぜ到達しないか」を明記する）。
///
/// 具体例（実地検証済み）: 1-way FE（`n_entities=20`）・`cluster=entity`（既定）・
/// `k=2`なら`df_model=22`・`nested_size_sum=20`・`count_nested=m=1`で`K=22-20+1=3`
/// （`k+1=3`と一致、フロアはno-op）。
///
/// `two_way_time`は2-wayの時点コード（1-wayは`None`）。FE次元は1-wayがentityのみ、
/// 2-wayがentityとtime（水準数は`time.n_groups()`）。
pub(super) fn fe_cluster_k_correction(
    entity: &GroupCodes,
    two_way_time: Option<&GroupCodes>,
    n_entities: usize,
    df_model: usize,
    k: usize,
    cluster: &GroupCodes,
) -> usize {
    let mut dims = vec![(
        fixef_dimension_nested_within_cluster(entity, cluster),
        n_entities,
    )];
    if let Some(time) = two_way_time {
        dims.push((
            fixef_dimension_nested_within_cluster(time, cluster),
            time.n_groups(),
        ));
    }
    let m = dims.len();
    let count_nested = dims.iter().filter(|(nested, _)| *nested).count();
    let nested_size_sum: usize = dims
        .iter()
        .filter(|(nested, _)| *nested)
        .map(|(_, size)| size)
        .sum();

    let raw_k: i64 = if count_nested == 0 {
        df_model as i64
    } else if count_nested == m {
        df_model as i64 - nested_size_sum as i64 + m as i64
    } else {
        df_model as i64 - (nested_size_sum as i64 - count_nested as i64)
    };
    raw_k.max((k + 1) as i64) as usize
}

/// `levels`の全水準がちょうど2観測を持ち、その2観測の`groups`ラベルが異なるか
/// （`groups`のユニーク数が2の前提で呼ぶ。このとき各水準が2グループに1観測ずつ）。
/// 1水準でも3観測以上・同じグループに2観測・1観測があれば`false`。
///
/// `levels.nobs() == groups.nobs()`は呼び出し側の契約（`FeInput`のコードと、
/// `validate_cluster_groups`済みのクラスター列のコード、または`fit()`のDkアームで
/// 長さを検証済みのDK時点列のコード）。
fn every_level_splits_once_across_two_groups(levels: &GroupCodes, groups: &GroupCodes) -> bool {
    debug_assert_eq!(
        levels.nobs(),
        groups.nobs(),
        "levels and groups must have the same length (caller contract)"
    );
    // 水準ごとの（最初の観測のグループコード、観測数）。`levels`のコードは全水準が
    // 少なくとも1回現れるため、最後の`all`は未出現の水準を考慮しなくてよい。
    let mut first_group = vec![usize::MAX; levels.n_groups()];
    let mut count = vec![0_usize; levels.n_groups()];
    for (&level, &group) in levels.codes().iter().zip(groups.codes()) {
        match count[level] {
            0 => {
                first_group[level] = group;
                count[level] = 1;
            }
            seen if seen >= 2 || first_group[level] == group => return false,
            _ => count[level] += 1,
        }
    }
    count.iter().all(|&c| c == 2)
}

/// ユニーク数2の`groups`（Clusterのクラスター列・Dkの時点列）について、within変換後の
/// グループスコアが恒等的にゼロになるか（`PanelError::DegenerateDkTwoPeriods`/
/// `DegenerateClusterTwoGroups`の判定）。
///
/// within変換は吸収した各FE次元の水準内で和をゼロにする。ある次元の全水準がちょうど2観測で
/// 2グループに1つずつ分かれると、各水準の2観測で`x̃`・`ẽ`が符号反転し、両グループに同じ寄与が
/// 入ってスコアが等しくなる。正規方程式でスコアの和はゼロのため、両方ゼロになる。entity方向
/// （2時点のパネルをtimeでクラスタリング等）だけでなく、2-wayではtime方向（エンティティ2つのパネルを
/// entityでクラスタリング——Clusterの既定——等）も、高次元ではどの次元も同じ構造になる。
///
/// `levels`は吸収した全ての効果の次元（`effects`の順。1-wayは1つ）。
pub(super) fn two_group_split_is_degenerate(levels: &[&GroupCodes], groups: &GroupCodes) -> bool {
    levels
        .iter()
        .any(|level| every_level_splits_once_across_two_groups(level, groups))
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;

    #[test]
    fn absorbed_parameters_subtracts_one_shared_constant_per_extra_effect() {
        assert_eq!(absorbed_parameters(&[]), 0);
        assert_eq!(absorbed_parameters(&[5]), 5);
        assert_eq!(absorbed_parameters(&[5, 2]), 6);
        assert_eq!(absorbed_parameters(&[40, 8, 5]), 51);
        // 水準が1つずつの効果は、定数項と重複するぶんだけを残す。
        assert_eq!(absorbed_parameters(&[1, 1, 1]), 1);
    }

    #[test]
    fn leverage_full_one_way_uses_each_rows_entity_size_for_unbalanced_unordered_rows() {
        // 行がエンティティ順に並んでおらず、T_i（a=3・b=2・c=1）が不均衡。各行は自分の
        // エンティティの観測数`T_i`で引かれなければならない（行番号やコード順で引くと外れる）。
        let entity = codes(&["b", "a", "c", "a", "b", "a"]);
        let h_within = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let sizes = [2.0, 3.0, 1.0, 3.0, 2.0, 3.0];

        let h_full = leverage_full(&h_within, &entity, None, 6).unwrap();

        for i in 0..6 {
            assert!(
                (h_full[i] - (1.0 / sizes[i] + h_within[i])).abs() < 1e-15,
                "row {i}"
            );
        }
    }

    #[test]
    fn leverage_full_two_way_uses_the_closed_form_for_a_balanced_panel_with_unordered_rows() {
        // 3個体×2時点の全組合せ（行は並べ替えてある）。`1/T_i + 1/N_t - 1/n + h`。
        let entity = codes(&["b", "a", "c", "a", "b", "c"]);
        let time = codes(&["t2", "t1", "t1", "t2", "t1", "t2"]);
        let h_within = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

        let h_full = leverage_full(&h_within, &entity, Some(&time), 6).unwrap();

        for i in 0..6 {
            let expected = 1.0 / 2.0 + 1.0 / 3.0 - 1.0 / 6.0 + h_within[i];
            assert!((h_full[i] - expected).abs() < 1e-15, "row {i}");
        }
    }

    #[test]
    fn leverage_full_two_way_adds_the_dummy_leverage_for_an_unbalanced_panel_with_unordered_rows() {
        // (a,t1)(a,t2)(a,t3)(b,t1)(b,t2)(c,t2)(c,t3) を並べ替えた不均衡パネル。
        let entity = codes(&["b", "a", "c", "a", "b", "a", "c"]);
        let time = codes(&["t1", "t1", "t2", "t3", "t2", "t2", "t3"]);
        let h_within = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];

        let h_full = leverage_full(&h_within, &entity, Some(&time), 7).unwrap();
        let dummy = dummy_leverage(&entity, &time).unwrap();

        for i in 0..7 {
            assert!(
                (h_full[i] - (dummy[i] + h_within[i])).abs() < 1e-15,
                "row {i}"
            );
        }
    }

    #[test]
    fn fixef_dimension_nested_within_cluster_true_for_default_entity_grouping() {
        let entity = codes(&["a", "a", "b", "b"]);
        assert!(fixef_dimension_nested_within_cluster(&entity, &entity));
    }

    #[test]
    fn fixef_dimension_nested_within_cluster_true_for_coarser_grouping() {
        // stateはentityより粗い分割（a,b→east、c,d→west）で、各entityは単一のstateに
        // 属するため「nested」と判定されるべき（fixestの実測でも同じ挙動を確認済み、
        // `fe.rs`のモジュールdoc参照）。
        let entity = codes(&["a", "a", "b", "b", "c", "c", "d", "d"]);
        let state = codes(&[
            "east", "east", "east", "east", "west", "west", "west", "west",
        ]);
        assert!(fixef_dimension_nested_within_cluster(&entity, &state));
    }

    #[test]
    fn fixef_dimension_nested_within_cluster_false_when_an_entity_spans_multiple_clusters() {
        // entity "a" が異なる2つのクラスター（"1"と"2"）にまたがるため、nestedではない。
        let entity = codes(&["a", "a", "b", "b"]);
        let cluster = codes(&["1", "2", "1", "2"]);
        assert!(!fixef_dimension_nested_within_cluster(&entity, &cluster));
    }

    #[test]
    fn fe_cluster_k_correction_floor_is_a_no_op_under_realistic_inputs() {
        // `fit()`が保証する前提（`df_model = k + neffects`）の下では、フロア
        // `K = max(K, k+1)`は常にno-op（`fe_cluster_k_correction`関数docの代数的
        // 導出参照）。1-way全ネストの現実的な入力で
        // `K`がちょうど`k+1`に一致することを確認する。
        let entity = codes(&["a", "a", "b", "b"]);
        let k = 1;
        let n_entities = 2;
        let df_model = k + n_entities; // 実際のFeEstimator::fitと同じneffects=n_entitiesの関係
        let k_correction = fe_cluster_k_correction(&entity, None, n_entities, df_model, k, &entity);
        assert_eq!(k_correction, k + 1);
    }

    #[test]
    fn fe_cluster_k_correction_floor_engages_for_inconsistent_inputs() {
        // フロア自体（`K = max(K, k+1)`）を直接検証する。`fit()`経由では`df_model`と
        // `n_entities`が常に整合しているため到達不能な組み合わせ（`fe_cluster_k_
        // correction`関数docの代数的導出参照）を、この関数を直接呼ぶことで意図的に
        // 構成する: `df_model=2`・`n_entities=2`（1-way全ネスト）だと、整合していれば
        // `df_model`は`k+n_entities=k+2`になるはずのところを`df_model=2`（`k=1`なら
        // `k+n_entities=3`のはず）に矛盾させ、`raw_k = df_model - nested_size_sum + 1
        // = 2 - 2 + 1 = 1`が`k+1=2`を下回る状況を作る。
        let entity = codes(&["a", "a", "b", "b"]);
        let k = 1;
        let n_entities = 2;
        let df_model = 2; // 本来のneffects=n_entities=2との整合を意図的に崩す
        let k_correction = fe_cluster_k_correction(&entity, None, n_entities, df_model, k, &entity);
        assert_eq!(
            k_correction,
            k + 1,
            "floor should clamp raw_k=1 up to k+1=2"
        );
    }

    #[test]
    fn every_level_splits_once_across_two_groups_detects_pattern() {
        let entity = codes(&["a", "a", "b", "b"]);
        assert!(every_level_splits_once_across_two_groups(
            &entity,
            &codes(&["1", "2", "2", "1"])
        ));
        // 同じグループに2観測。
        assert!(!every_level_splits_once_across_two_groups(
            &entity,
            &codes(&["1", "1", "1", "2"])
        ));
        // 3観測のエンティティ。
        assert!(!every_level_splits_once_across_two_groups(
            &codes(&["a", "a", "a", "b", "b"]),
            &codes(&["1", "2", "1", "1", "2"])
        ));
        // 1観測のエンティティ（singletonは通常`fit()`が先に弾く）。
        assert!(!every_level_splits_once_across_two_groups(
            &codes(&["a", "a", "b"]),
            &codes(&["1", "2", "1"])
        ));
    }

    #[test]
    fn two_group_split_is_degenerate_checks_every_absorbed_dimension() {
        let groups = codes(&["1", "1", "2", "2"]);
        // 1つ目の次元では各水準の2観測が同じグループに入るので、単独では退化しない。
        let first = codes(&["a", "a", "b", "b"]);
        // 2つ目の次元では各水準の2観測が2グループに1つずつ分かれる。
        let second = codes(&["p", "q", "p", "q"]);
        assert!(!two_group_split_is_degenerate(&[&first], &groups));
        assert!(two_group_split_is_degenerate(&[&second], &groups));
        assert!(two_group_split_is_degenerate(&[&first, &second], &groups));
        // 3つ目の次元だけが退化させる場合も見つける。
        assert!(two_group_split_is_degenerate(
            &[&first, &first, &second],
            &groups
        ));
        assert!(!two_group_split_is_degenerate(&[], &groups));
    }
}
