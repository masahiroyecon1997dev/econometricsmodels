# econometricsmodels

A Python API providing statistical and econometric analysis methods. It is primarily intended for use as the analysis engine for [economicon](https://github.com/masahiroyecon1997dev/economicon), a GUI application for data analysis, and its design prioritizes ease of embedding from scripts and programs (type completion, validation, dynamic construction).

- The computational core is implemented in **Rust** and thinly bound to Python via **PyO3**.
- Data input is restricted to **polars** DataFrames only, passed to the Rust side via **Arrow zero-copy**.
- Formula-string parsing (e.g. `y ~ x1 + x2`) is not used. The dependent variable is passed as a single column name (`str`), independent variables as a list of column names (`list[str]`), and estimation options as an instance of a dedicated class.

## Installation

```bash
pip install econometricsmodels
```

## Supported methods

Currently implemented: OLS (Ordinary Least Squares), WLS (Weighted Least Squares), and Logit (binary logistic regression). See [Getting Started](getting-started.md) for usage, and the API Reference ([OLS](api/ols.md) / [WLS](api/wls.md) / [Logit](api/logit.md)) for detailed options and return values.

IV (2SLS/GMM), Probit, and others will be added in the future.
