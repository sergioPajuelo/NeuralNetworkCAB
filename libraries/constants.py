#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Constants library

"""

import numpy as np


from typing       import Final, Dict
from numpy.random import uniform,randint

MIN_LENGTH  =  1000
MAX_LENGTH  = 20000


class ParameterLimits:
    
    """
    Constant class with parameter limits used in the Lorentzian generator.
    """
    
    LEVEL_LOWER_LIMIT     : Final[float] = 0.01      # No units
    LEVEL_UPPER_LIMIT     : Final[float] = 2.0       # No units
    DELAY_LOWER_LIMIT     : Final[float] = -1e-7     # s
    DELAY_UPPER_LIMIT     : Final[float] = 0.0       # s
    ZERO_LOWER_LIMIT      : Final[float] = -np.pi    # rad
    ZERO_UPPER_LIMIT      : Final[float] =  np.pi    # rad
    LOSSES_LOWER_LIMIT    : Final[float] = 1e2       # Hz
    LOSSES_UPPER_LIMIT    : Final[float] = 1e5       # Hz
    COUPLING_LOWER_LIMIT  : Final[float] = 1e4       # Hz
    COUPLING_UPPER_LIMIT  : Final[float] = 1e5       # Hz
    FANO_LOWER_LIMIT      : Final[float] = -np.pi/4  # rad
    FANO_UPPER_LIMIT      : Final[float] =  np.pi/4  # rad
    RESONANCE_LOWER_LIMIT : Final[float] = 3.0e8     # Hz
    RESONANCE_UPPER_LIMIT : Final[float] = 1.0e9     # Hz
    
    """  ======================================= """
    
    # SWEEP_FACTOR indicates how many times kappa is covered by frequency sweep
    SWEEP_FACTOR_LOWER_LIMIT : Final[int] = 25
    SWEEP_FACTOR_UPPER_LIMIT : Final[int] = 401
    
    
    @classmethod
    def sample(cls) -> Dict[str, float]:
        
        kappai = cls._randomize(cls.LOSSES_LOWER_LIMIT, cls.LOSSES_UPPER_LIMIT, exp = True)
        kappac = cls._randomize(cls.COUPLING_LOWER_LIMIT, cls.COUPLING_UPPER_LIMIT, exp = True)
        kappa  = kappai + kappac
        return {
            'ac'      : cls._randomize(cls.LEVEL_LOWER_LIMIT, cls.LEVEL_UPPER_LIMIT, exp = True),
            'dt'      : cls._randomize(cls.DELAY_LOWER_LIMIT, cls.DELAY_UPPER_LIMIT),
            'phi'     : cls._randomize(cls.ZERO_LOWER_LIMIT, cls.ZERO_UPPER_LIMIT),
            'r'       : kappac / kappa,
            'kappai'  : kappai,
            'kappac'  : kappac,
            'kappa'   : kappa,
            'dphi'    : cls._randomize(cls.FANO_LOWER_LIMIT, cls.FANO_UPPER_LIMIT),
            'fr'      : cls._randomize(cls.RESONANCE_LOWER_LIMIT, cls.RESONANCE_UPPER_LIMIT),
            
            'sweep_factor' : cls.sweep_factor(),
        }
        
    
    @classmethod
    def sweep_factor(cls) -> int:
        return randint(cls.SWEEP_FACTOR_LOWER_LIMIT, cls.SWEEP_FACTOR_UPPER_LIMIT)
    
    
    @classmethod
    def help(cls) -> None:
        """
        Print a human-readable explanation of each parameter name.
        """
        print("\nParameterLimits — meaning of each constant\n")

        print("LEVEL_*")
        print("  Overall amplitude scaling of the resonator response |S21|.")
        print(f"  Range: [{cls.LEVEL_LOWER_LIMIT}, {cls.LEVEL_UPPER_LIMIT}] (dimensionless)\n")

        print("DELAY_*")
        print("  Global electrical / cable delay applied to the complex response.")
        print(f"  Range: [{cls.DELAY_LOWER_LIMIT}, {cls.DELAY_UPPER_LIMIT}] s\n")

        print("LOSSES_*")
        print("  Internal losses of the resonator (κ_i).")
        print(f"  Range: [{cls.LOSSES_LOWER_LIMIT}, {cls.LOSSES_UPPER_LIMIT}] Hz\n")
        
        print("COUPLING_*")
        print("  External coupling rate (κ_c).")
        print(f"  Range: [{cls.COUPLING_LOWER_LIMIT}, {cls.COUPLING_UPPER_LIMIT}] Hz\n")

        print("FANO_*")
        print("  Fano-like asymmetry / complex phase rotation of the resonance.")
        print(f"  Range: [{cls.FANO_LOWER_LIMIT}, {cls.FANO_UPPER_LIMIT}] rad\n")

        print("RESONANCE_*")
        print("  Resonance frequency f_r of the resonator.")
        print(f"  Range: [{cls.RESONANCE_LOWER_LIMIT}, {cls.RESONANCE_UPPER_LIMIT}] Hz\n")

    @staticmethod
    def _randomize(low: float, high: float, exp: bool = False) -> float:
        """
        Class helper to randomize between limits
        """
        if exp:
            return float(np.exp(uniform(np.log(low), np.log(high))))
        else:
            return float(uniform(low, high))
        

