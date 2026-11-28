//! 負の分散の0への切り詰めのテスト。

use crate::panel::re::ReCovType;
use crate::panel::re::ReEstimator;
use crate::panel::re::input::ReEffects;
use crate::panel::re::input::ReInput;
use crate::panel::re::test_support::*;
use crate::panel::re::variance::VarianceComponents;
use crate::panel::re::variance::swamy_arora_variance_components;

#[test]
fn swamy_arora_variance_components_clips_negative_sigma2_u_to_zero() {
    // entity間のy平均のばらつきが、within回帰から推定したσ_ε²/Tに対して
    // 十分小さいため、素朴な式ではσ_u²が負になるが`max(0, ...)`で0にクリップされる
    // （`linearmodels`・`plm`と同じ挙動、`re-spec.md`3.7節のハウスマン統計量の負値と同型の
    // 「有限標本でのPSD仮定崩れ」）。エンティティ間のx1平均はわずかに異なる値にし、
    // between回帰の設計行列が特異にならないようにする。`linearmodels`で実地検証済み
    // （`variance_decomposition["Effects"] == 0.0`。バランスパネルなので式は同じ）。
    let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c"]);
    let x1 = vec![1.0, 2.0, 3.0, 1.2, 2.1, 2.9, 0.9, 2.2, 3.1];
    let y = [2.0, 4.0, 6.0, 2.3, 4.1, 5.9, 1.8, 4.3, 6.2];
    let input = ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();

    let VarianceComponents {
        sigma2_eps,
        sigma2_u,
        ..
    } = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();

    assert!(
        (sigma2_eps - 0.005_868_778_280_543_04).abs() < 1e-9,
        "sigma2_eps = {sigma2_eps}"
    );
    assert_eq!(sigma2_u, 0.0);
}

#[test]
fn one_way_exposed_variance_components_are_clipped_at_zero() {
    // 個体効果の無いDGPでは負の推定が0に切り上げられ、公開される`σ_u²`は0、`θ_i`はすべて0
    // （変換なし、pooled OLSに一致）。
    let c = two_way_columns();
    let y: Vec<f64> =
        c.x1.iter()
            .enumerate()
            .map(|(r, v)| 2.0 * v + 0.1 * ((r * 7) % 5) as f64)
            .collect();
    let input = ReInput::from_columns(
        &y,
        std::slice::from_ref(&c.x1),
        strings(&["x1"]),
        &c.entity,
        None,
        "y".into(),
    )
    .unwrap();
    let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
    assert_eq!(re.sigma2_u(), 0.0);
    assert!(re.theta_by_entity().unwrap().iter().all(|&t| t == 0.0));
}

#[test]
fn exposed_variance_components_are_clipped_at_zero() {
    // 負の推定値は0に切り上げた値（θの計算に使った値）を返す。グループ効果が無いDGPでは
    // 両方0になり、θもすべて0（変換なし）。
    let c = two_way_columns();
    let y: Vec<f64> =
        c.x1.iter()
            .enumerate()
            .map(|(r, v)| 2.0 * v + 0.1 * ((r * 7) % 5) as f64)
            .collect();
    let input = ReInput::from_columns(
        &y,
        std::slice::from_ref(&c.x1),
        strings(&["x1"]),
        &c.entity,
        Some(&c.time),
        "y".into(),
    )
    .unwrap();
    let re = ReEstimator::fit(input, ReEffects::TwoWay, ReCovType::Classical, 0.95).unwrap();
    assert_eq!(re.sigma2_u(), 0.0);
    assert_eq!(re.sigma2_lambda(), Some(0.0));
    assert_eq!(re.two_way_theta(), Some((0.0, 0.0, 0.0)));
}

#[test]
fn swamy_arora_variance_components_clips_negative_sigma2_u_on_an_unbalanced_panel_like_plm() {
    // 上と同じデータから最後の行を落とした不均衡パネル（T_i={3, 3, 2}）。トレース形の分子
    // `Q_id − σ_ε²(N − p)`が負（Rで約-0.0052、分母`O − Σ T_i h_i`は約2.99）になり、0に切り上げる。
    // R: `plm::ercomp(y ~ x1, method = "swar")$sigma2`は(0.0073145631067961878, 0)で、
    // `plm(model = "random")`の係数はプールドOLSと一致する。
    let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c"]);
    let x1 = vec![1.0, 2.0, 3.0, 1.2, 2.1, 2.9, 0.9, 2.2];
    let y = [2.0, 4.0, 6.0, 2.3, 4.1, 5.9, 1.8, 4.3];
    let input = ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
        .unwrap();

    let vc = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();
    assert_rel(
        vc.sigma2_eps,
        0.007_314_563_106_796_188,
        1e-10,
        "sigma2_eps",
    );
    assert_eq!(vc.sigma2_u, 0.0);

    let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
    assert!(re.theta_by_entity().unwrap().iter().all(|&t| t == 0.0));
    assert_rel(
        *re.estimator().params().get(0, 0),
        -0.075_396_611_992_471_61,
        1e-10,
        "const",
    );
    assert_rel(
        *re.estimator().params().get(1, 0),
        2.026_351_169_669_265_8,
        1e-10,
        "x1",
    );
}
