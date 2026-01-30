import numpy as np
from .constants import MIN_POLY_ORDER, MAX_POLY_ORDER

def polysynth(
    frequency         : np.ndarray,
    trace             : np.ndarray,
    poly_order        : int | None = None,
    poly_coeff_scale  : float = 0.05,
    noiser            : bool = False
    ) -> np.ndarray:
    
    frequency = np.asarray(frequency, dtype=np.float64)
    trace     = np.asarray(trace, dtype=np.complex128)
    
    f0   = float(np.mean(frequency))
    span = float(np.ptp(frequency)) + 1e-12
    x    = (frequency - f0) / (span / 2.0)  # ~[-1,1]

    a = _random_poly_response(
        x,
        poly_order       = poly_order,
        poly_coeff_scale = poly_coeff_scale,
    )
    
    if noiser: trace = _noiser(trace)
    
    return a * trace, a

def _random_poly_response(
    x                : np.ndarray,
    poly_order       : int | None = None,
    poly_coeff_scale : float = 0.05,
    ) -> np.ndarray:
    
    if poly_order is None:
        order = int(np.random.randint(MIN_POLY_ORDER, MAX_POLY_ORDER + 1))
    else:
        order = poly_order

    # growing order polynomial: c0 + c1 x + c2 x^2 ...
    coef = np.random.normal(0.0, poly_coeff_scale, size=order + 1)
    coef[0] = 0.0

    # evaluate
    p = np.polynomial.polynomial.polyval(x, coef)

    p = np.exp(p - p.mean())

    return p

def _noiser(
        trace   : np.ndarray,
        sigma   : float = 2,
        ):

    amplitude_noise = np.random.normal(0, sigma, len(trace))
    phase_noise = np.random.normal(0, sigma, len(trace))
    
    amplitude = abs(trace)
    phase     = np.unwrap(np.atan2(trace.imag, trace.real))
    
    amplitude += amplitude_noise
    phase     += phase_noise
    
    return amplitude * np.exp(1j * phase)
    
    