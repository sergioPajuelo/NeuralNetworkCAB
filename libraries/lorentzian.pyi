from typing import Any
import numpy as np


def lorentzian(f: "np.ndarray[np.float64]", ac: float, dt: float, phi: float,
               r: float, kappa: float, dphi: float, fr: float) -> "np.ndarray[np.complex128]":
    ...
