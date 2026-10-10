"""Top-level package for `econometricsmodels`, a programmatic Python API
for econometric analysis.

Exposes a Python API that accepts polars DataFrames, as a thin wrapper
around the native extension (`econometricsmodels._lib`) built by
`engine_pybind`.
"""

from __future__ import annotations

from ._lib import ComputationError, ValidationError
from .diagnostics import (
    BreuschGodfreyTestResult,
    DiagnosticResult,
    WhiteTestResult,
)
from .iv.iv import IV, IVOptions, IVResults
from .linear.ols import OLS, OLSOptions, OLSResults
from .linear.wls import WLS, WLSOptions, WLSResults
from .nonlinear.logit import Logit, LogitOptions, LogitResults
from .nonlinear.probit import Probit, ProbitOptions, ProbitResults
from .nonlinear.tobit import Tobit, TobitOptions, TobitResults
from .panel.fe import FE, FEOptions, FEResults
from .panel.re import RE, REOptions, REResults

__all__ = [
    "FE",
    "IV",
    "OLS",
    "RE",
    "WLS",
    "BreuschGodfreyTestResult",
    "ComputationError",
    "DiagnosticResult",
    "FEOptions",
    "FEResults",
    "IVOptions",
    "IVResults",
    "Logit",
    "LogitOptions",
    "LogitResults",
    "OLSOptions",
    "OLSResults",
    "Probit",
    "ProbitOptions",
    "ProbitResults",
    "REOptions",
    "REResults",
    "Tobit",
    "TobitOptions",
    "TobitResults",
    "ValidationError",
    "WLSOptions",
    "WLSResults",
    "WhiteTestResult",
]

__version__ = "0.8.0"
