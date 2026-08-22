"""Top-level package for `econometricsmodels`, the analysis engine for economicon.

Exposes a Python API that accepts polars DataFrames, as a thin wrapper
around the native extension (`econometricsmodels._lib`) built by
`engine_pybind`.
"""

from __future__ import annotations

from ._lib import ComputationError, ValidationError
from .iv.iv import IV, IvOptions, IvResults
from .linear.ols import OLS, OLSOptions, OlsResults
from .linear.wls import WLS, WlsResults
from .nonlinear.logit import Logit, LogitOptions, LogitResults
from .nonlinear.probit import Probit, ProbitOptions, ProbitResults
from .nonlinear.tobit import Tobit, TobitOptions, TobitResults

__all__ = [
    "IV",
    "OLS",
    "WLS",
    "ComputationError",
    "IvOptions",
    "IvResults",
    "Logit",
    "LogitOptions",
    "LogitResults",
    "OLSOptions",
    "OlsResults",
    "Probit",
    "ProbitOptions",
    "ProbitResults",
    "Tobit",
    "TobitOptions",
    "TobitResults",
    "ValidationError",
    "WlsResults",
]

__version__ = "0.5.0"
