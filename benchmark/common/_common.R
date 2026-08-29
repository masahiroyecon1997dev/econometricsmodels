# run_lm_crosscheck_benchmark.R（OLS/WLS）とrun_ivreg_benchmark.R（IV/2SLS）で
# 完全に同一だった後処理を集約する共通ヘルパー。source()で読み込んで使う。
#
# cov_type→vcov分岐自体（lmはHC0-3・weight対応、ivregはHC0-1のみ対応、等の
# 実質的な差分がある）は共通化しない。coeftest()からの係数・標準誤差抽出と、
# ロバストWald F検定の2ブロックのみを対象にする。

# 係数・標準誤差・t値・p値をcoeftest()から抽出し、名前付きベクトルとして返す。
extract_coef_se <- function(model, vc, df_inference) {
  ct <- coeftest(model, vcov = vc, df = df_inference)
  coefs <- ct[, 1]
  ses <- ct[, 2]
  t_stats <- ct[, 3]
  p_values <- ct[, 4]
  names(coefs) <- rownames(ct)
  names(ses) <- rownames(ct)
  names(t_stats) <- rownames(ct)
  names(p_values) <- rownames(ct)
  list(coefs = coefs, ses = ses, t_stats = t_stats, p_values = p_values)
}

# ロバストWald F検定（本実装のwald_f_testと同じ定義: 傾き係数の同時共分散
# 部分行列を使うWald検定）。傾き係数の同時共分散部分行列が数値的に特異な場合、
# solve()は"system is computationally singular"としてエラーを投げる。本実装
# （engine::linear::ols::wald_f_test / engine::iv::two_sls側の同等ロジック）も
# 固有値分解による相対閾値判定で同様のケースをComputationErrorとして検出する
# ため、この関数の呼び出し元（各generate_*_crosscheck_fixtures.py）はそのような
# ケースをNUMERIC_SCENARIOSから除外し、両実装が計算不能で一致することのみ
# 確認する（perfect_multicollinearityと同じ方針）。
wald_f_test <- function(model, vc, df_inference) {
  coef_names <- names(coef(model))
  slope_idx <- which(coef_names != "(Intercept)")
  beta_slopes <- coef(model)[slope_idx]
  df_model <- length(slope_idx)
  v_slopes <- vc[slope_idx, slope_idx, drop = FALSE]
  wald <- as.numeric(t(beta_slopes) %*% solve(v_slopes) %*% beta_slopes)
  f_statistic <- wald / df_model
  f_p_value <- 1 - pf(f_statistic, df_model, df_inference)
  list(f_statistic = f_statistic, f_p_value = f_p_value)
}
