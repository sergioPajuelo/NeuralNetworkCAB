#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 12:50:47 2025

Optimized version of lorentzian function using Cython implementation

@author: SuperTech
"""

# lorentzian.pyx

#cython: language_level=3
import numpy as np
cimport numpy as np
from libc.math cimport cos, sin
cimport cython


def lorentzian(np.ndarray[np.float64_t, ndim=1] f,
               double ac, double dt, double phi,
               double r, double kappa, double dphi, double fr):
    """
    Parameters
    ----------
    r : double
        Coupling ratio, kappa_c / kappa, with 0 < r < 1
    kappa : double
        Total linewidth (kappa_i + kappa_c)
    """
    cdef int i, N = f.shape[0]
    cdef np.ndarray[np.complex128_t, ndim=1] out = np.empty(N, dtype=np.complex128)
    cdef double df
    cdef complex denom, phase, term

    for i in range(N):
        df = f[i] - fr
        denom = 1j * df + kappa
        phase = np.exp(1j * (f[i] * dt + phi))
        term = 1 - r * kappa * np.exp(1j * dphi) / denom
        out[i] = ac * phase * term

    return out

