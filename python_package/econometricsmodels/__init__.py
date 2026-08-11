"""Top-level package for `econometricsmodels`, the analysis engine for economicon.

Exposes a Python API that accepts polars DataFrames, as a thin wrapper
around the native extension (`econometricsmodels._lib`) built by
`engine_pybind`.
"""

from __future__ import annotations

from ._lib import ComputationError, ValidationError
from .linear.ols import OLS, OLSOptions, OlsResults
from .linear.wls import WLS, WlsResults
from .nonlinear.logit import Logit, LogitOptions, LogitResults
from .nonlinear.probit import Probit, ProbitOptions, ProbitResults

__all__ = [
    "OLS",
    "WLS",
    "ComputationError",
    "Logit",
    "LogitOptions",
    "LogitResults",
    "OLSOptions",
    "OlsResults",
    "Probit",
    "ProbitOptions",
    "ProbitResults",
    "ValidationError",
    "WlsResults",
]

__version__ = "0.4.0"
