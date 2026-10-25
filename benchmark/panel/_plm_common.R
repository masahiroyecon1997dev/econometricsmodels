# パネルのplm系Rスクリプト（`run_plm_benchmark.R`・`run_plm_hausman_benchmark.R`・
# `run_plm_hausman_tilde_benchmark.R`と`references/`の`run_plm_re_*`）が共有する部品。
# `source()`で読み込んで使う（`plm`・`sandwich`は呼び出し側で読み込む）。
#
# FEの`references/run_plm_fe_benchmark.R`は、fixest/Stata型の小標本補正を別の式で
# 再現するためvcovの対応が異なり、`plm_t_inference()`だけを使う。

# cov_typeに対応するplmのvcov関数（plmモデルを受け取り共分散行列を返す）。
#
# - classical: `vcov`
# - hc1/hc2/hc3: `vcovHC(method = "white1", type = "HC1"/"HC2"/"HC3")`
# - cluster（entity）: `vcovHC(method = "arellano", type = "sss")`
#   （`G/(G-1) * (nT-1)/(nT-k)`。plmはgroup/timeしかクラスターにできない）
# - dk: `vcovSCC(maxlag, type = "sss")`（`T/(T-1) * (nT-1)/(nT-k)`補正、Bartlett重み）
plm_vcov_function <- function(cov_type, maxlag = NA) {
  if (cov_type == "classical") {
    vcov
  } else if (cov_type %in% c("hc1", "hc2", "hc3")) {
    hc_type <- toupper(cov_type)
    function(x) vcovHC(x, method = "white1", type = hc_type)
  } else if (cov_type == "cluster") {
    function(x) vcovHC(x, method = "arellano", type = "sss")
  } else if (cov_type == "dk") {
    if (is.na(maxlag)) {
      stop("dk requires maxlag")
    }
    function(x) vcovSCC(x, maxlag = maxlag, type = "sss")
  } else {
    stop(paste("unknown cov_type:", cov_type))
  }
}

# t検定の自由度。classical・hc*は`df.residual`、clusterは`G-1`、dkは`T-1`
# （fixestの`ssc()`既定と同じ規約）。
plm_t_df <- function(cov_type, df_resid, n_groups, n_periods) {
  switch(
    cov_type,
    cluster = n_groups - 1,
    dk = n_periods - 1,
    df_resid
  )
}

# 係数と標準誤差から、本実装と同じt分布の式でt統計量・p値・95%信頼区間を作る
# （plmの既定はz検定のため。95%信頼区間固定は`run_fixest_benchmark.R`の
# `confint()`既定と揃える）。
plm_t_inference <- function(coefs, ses, t_df) {
  test_stats <- coefs / ses
  crit <- qt(0.975, df = t_df)
  list(
    test_stats = test_stats,
    p_values = 2 * pt(-abs(test_stats), df = t_df),
    conf_lower = coefs - crit * ses,
    conf_upper = coefs + crit * ses
  )
}
