# R 参照実装でテストフィクスチャを作る

## 目的

Rust 実装の数値精度を検証するため、R の参照実装（`lm`, `glm`, `AER`, `plm` 等）で「正解値」を生成し、CSV として保存する。

## ディレクトリ構成

```
tests/
  r_reference/        # R スクリプト（.R ファイル）
    01_ols_basic.R
    02_iv_2sls.R
  fixtures/           # R スクリプトの出力 CSV
    01_ols_basic_coefficients.csv
    01_ols_basic_data.csv
```

## R スクリプトのテンプレート

```r
# tests/r_reference/01_ols_basic.R
set.seed(42)  # 再現性のためシードを固定

n <- 1000
x1 <- rnorm(n)
x2 <- rnorm(n)
y  <- 2.0 * x1 - 1.5 * x2 + rnorm(n, sd = 0.5)
df <- data.frame(y = y, x1 = x1, x2 = x2)

# 参照実装で推定
fit <- lm(y ~ x1 + x2, data = df)

# 入力データを保存
write.csv(df, "tests/fixtures/01_ols_basic_data.csv", row.names = FALSE)

# 推定結果を保存
result <- data.frame(
  term     = names(coef(fit)),
  estimate = coef(fit),
  std_err  = summary(fit)$coefficients[, "Std. Error"],
  t_value  = summary(fit)$coefficients[, "t value"],
  p_value  = summary(fit)$coefficients[, "Pr(>|t|)"]
)
write.csv(result, "tests/fixtures/01_ols_basic_coefficients.csv", row.names = FALSE)
```

## 主要 R パッケージ対応表

| 推定量 | R パッケージ・関数 |
|-------|----------------|
| OLS | `stats::lm` |
| Logit / Probit | `stats::glm(family = binomial(...))` |
| Tobit | `AER::tobit` |
| Heckman | `sampleSelection::heckit` |
| IV/2SLS | `AER::ivreg` |
| FE パネル | `plm::plm(model = "within")` |
| RE パネル | `plm::plm(model = "random")` |
| クラスター SE | `sandwich::vcovCL` + `lmtest::coeftest` |

## pytest 側でのフィクスチャ読み込み

```python
import polars as pl
import pytest

@pytest.fixture
def ols_data():
    return pl.read_csv("tests/fixtures/01_ols_basic_data.csv")

@pytest.fixture
def ols_expected():
    return pl.read_csv("tests/fixtures/01_ols_basic_coefficients.csv")

def test_ols_coefficients(ols_data, ols_expected):
    result = OLS(ols_data, "y", ["x1", "x2"]).fit()
    for row in ols_expected.iter_rows(named=True):
        actual = result.params[row["term"]]
        assert abs(actual - row["estimate"]) / abs(row["estimate"]) < 1e-7
```
