#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lorentzian: Cython-accelerated Lorentzian resonator model.
ParameterLimits: Parmeter class for lorentzian_generator
polysynth: generates a synthetic polynomial base and applies it to trace
"""

from .lorentzian import lorentzian as lorentzian_cy

from .constants import ParameterLimits, MAX_LENGTH, MIN_LENGTH

from ._polysynth import polysynth

__all__ = ["lorentzian_cy", "ParameterLimits", "polysynth",
           "MAX_LENGTH", "MIN_LENGTH"]