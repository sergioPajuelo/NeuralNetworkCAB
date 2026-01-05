import numpy as np
from lorentzian import lorentzian as lorentzian_cy  
#from trace import Trace
import matplotlib.pyplot as plt
import numpy as np
from sctlib.analysis import Trace

def _random_poly_response(x: np.ndarray,
                          deg_min: int = 1,
                          deg_max: int = 5,
                          coeff_scale: float = 0.25,
                          ensure_positive: bool = False) -> np.ndarray:
    """
    Devuelve a(f) como polinomio real aleatorio evaluado en x (x normalizado ~ [-1,1]).
    - Randomiza grado
    - Randomiza coeficientes
    """
    deg = int(np.random.randint(deg_min, deg_max + 1))
    coef = np.random.normal(0.0, coeff_scale, size=deg + 1)
    coef[0] = 1.0   # baseline alrededor de 1

    a = np.zeros_like(x, dtype=np.float64)
    xp = np.ones_like(x, dtype=np.float64)
    for k in range(deg + 1):
        a += coef[k] * xp
        xp *= x     # x^k --> x^(k+1)

    # Asegura amp > 0
    if ensure_positive:
        a = a - np.min(a)
        a = 0.2 + a  
    
    a = a / (np.mean(np.abs(a)) + 1e-12)

    return a


def _apply_system_response(f: np.ndarray,
                           s: np.ndarray,
                           poly_deg_range: tuple[int, int] = (1, 5),
                           poly_coeff_scale: float = 0.25,
                           dt_sys_range: tuple[float, float] = (-2e-9, 2e-9),
                           phi0_sys_range: tuple[float, float] = (-np.pi, np.pi)) -> np.ndarray:
    """
    Multiplica s(f) por H(f)=a(f)*exp(i*(dt_sys*f + phi0_sys)).
    """
    f = f.astype(np.float64)
    f0 = float(np.mean(f))
    span = float(np.ptp(f)) + 1e-12     # max(f) - min(f) + 1e-12
    x = (f - f0) / (span / 2.0)     # normaliza en [-1, 1]

    a = _random_poly_response(
        x,
        deg_min=poly_deg_range[0],
        deg_max=poly_deg_range[1],
        coeff_scale=poly_coeff_scale,
        ensure_positive=False,
    )

    dt_sys = float(np.random.uniform(*dt_sys_range))
    phi0_sys = float(np.random.uniform(*phi0_sys_range))

    phase = dt_sys * f + phi0_sys
    H = a * np.exp(1j * phase)
    return s * H

def lorentzian_generator(
    n_samples: int,
    cavity_params: dict,
    kc_limits: tuple[float, float],
    frequency_points: int = 2000,
    noise_std_signal: float | tuple[float, float] = 0.0,
):
    """
    Genera un dataset sintético usando la lorentziana de Cython y la clase Trace.

    Parámetros
    ----------
    n_samples : int
        Número de trazas sintéticas a generar.
    cavity_params : dict
        Parámetros fijos de la cavidad (excepto kc). Ejemplo:
        {
            "ac": 1.0,
            "dt": 0.0,
            "phi": 0.0,
            "dphi": 0.0,
            "kappai": 1.0e6,   # kappa interna
            "fr": 7.5e9,       # frecuencia de resonancia
        }
    kc_limits : (kc_min, kc_max)
        Rango uniforme de kappa_c (acoplo) a muestrear.
    frequency_limits : (f_min, f_max)
        Rango de frecuencias del barrido (Hz).
    frequency_points : int
        Número de puntos de frecuencia por traza.
    noise_std_signal : float
        σ del ruido gaussiano añadido a la señal (real e imag) para las
        mediciones sintéticas.

    Devuelve
    --------
    f : ndarray, shape (F,)
        Vector de frecuencias.
    X_meas : ndarray, shape (N, 2F)
        Dataset de mediciones sintéticas (con ruido):
        [Re(S21_meas) || Im(S21_meas)].
    X_clean : ndarray, shape (N, 2F)
        Dataset de lorentzianas teóricas (sin ruido) para cada muestra:
        [Re(S21_clean) || Im(S21_clean)].
    kc_true : ndarray, shape (N,)
        Valores verdaderos de kappa_c usados para cada traza.
    traces : list[Trace]
        Lista de objetos Trace correspondientes a las mediciones sintéticas.
    """


    kappai_true = np.zeros(n_samples)
    log_lo, log_hi = np.log(kc_limits[0]), np.log(kc_limits[1])
    kc_true = np.exp(np.random.uniform(log_lo, log_hi, size=n_samples))

    X_meas  = np.empty((n_samples, 2 * frequency_points), dtype=np.float32)
    X_clean = np.empty((n_samples, 2 * frequency_points), dtype=np.float32)

    #traces: list[Trace] = []

    for i, kc in enumerate(kc_true):

        # Cavity params
        ac = float(np.exp(np.random.uniform(
            np.log(cavity_params["ac"][0]),
            np.log(cavity_params["ac"][1])
        )))

        dt = float(np.random.uniform(*cavity_params["dt"]))

        fr = float(np.random.uniform(*cavity_params["fr"]))

        dphi = float(np.random.uniform(*cavity_params["dphi"]))

        kappai = float(np.exp(np.random.uniform(
            np.log(cavity_params["kappai"][0]),
            np.log(kc_limits[1] * 5)
        )))
        kappai_true[i] = kappai

        phi  = float(np.random.uniform(*cavity_params["phi"]))

        kc = float(kc)
        kappa = kappai + kc   
        r = kc / kappa        # ratio de acoplo

        delta_f_max = 10 * kc_limits[1] + kappai
        f = np.linspace(
            fr - delta_f_max,
            fr + delta_f_max,
            frequency_points,
            dtype=np.float64,
        )

        # Lorentzian 
        s_clean = lorentzian_cy(f, ac, dt, phi, r, kappa, dphi, fr)

        s_clean = _apply_system_response(
            f, s_clean,
            poly_deg_range=(1, 5),
            poly_coeff_scale=np.random.uniform(0.02, 0.06),      
            dt_sys_range=(-4e-9, 4e-9), 
            phi0_sys_range=(-np.pi, np.pi),
        )

        c0 = (np.random.normal(0.0, 0.05) + 1j*np.random.normal(0.0, 0.05))
        s_clean = s_clean + c0
        eps = np.random.uniform(-0.03, 0.03)  # 3% de imbalance
        I = s_clean.real * (1 + eps)
        Q = s_clean.imag * (1 - eps)
        s_clean = I + 1j*Q

        s_meas = s_clean.copy()

        if isinstance(noise_std_signal, tuple):
            sig = float(np.random.uniform(noise_std_signal[0], noise_std_signal[1]))
        else:
            sig = float(noise_std_signal)

        if sig > 0.0:
            noise_real = np.random.normal(0.0, sig, size=frequency_points)
            noise_imag = np.random.normal(0.0, sig, size=frequency_points)
            s_meas += noise_real + 1j * noise_imag

        # Combinamos partes real e imaginaria
        X_clean[i, :frequency_points] = s_clean.real.astype(np.float32)
        X_clean[i, frequency_points:] = s_clean.imag.astype(np.float32)

        X_meas[i, :frequency_points] = s_meas.real.astype(np.float32)
        X_meas[i, frequency_points:] = s_meas.imag.astype(np.float32)

        # Trace con la medición sintética
        """ tr = Trace(
            frequency=f.copy(),
            trace=s_meas.astype(np.complex128),
            trace_type="frequency sweep",
            mod="iq",
            model="parallel",
        )
        traces.append(tr) """

    return f, X_meas, X_clean, kc_true.astype(np.float32), kappai_true


if __name__ == "__main__":
    cavity_params = {
        "ac":     (0.3, 1.8),          
        "dt":     (0, 1e-9),        
        "phi":    (-np.pi, np.pi),      
        "dphi":   (-np.pi/4, np.pi/4),      
        "kappai": (1e4, 1e6),         
        "fr":     (7.30e11 - 2e9, 7.50e11 + 2e9)
    }

    kc_limits = (1e4, 1e5)

    f, X_meas, X_clean, kc_true, kappai_true = lorentzian_generator(
        n_samples=3,
        cavity_params=cavity_params,
        kc_limits=kc_limits,
        frequency_points=5000,     
        noise_std_signal=0.0,
    )

    i = 2
    F = f.shape[0]

    re = X_meas[i, :F]
    im = X_meas[i, F:]
    mag = np.sqrt((re**2 + im**2))

    f_GHz = f * 1e-9

    plt.figure()
    plt.plot(f_GHz, re, label="Re(S21)")
    plt.plot(f_GHz, im, label="Im(S21)")
    plt.plot(f_GHz, mag, label="|S21|", linestyle="--")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title(f"Lorentzian clean (kc = {kc_true[i]:.2e})")
    plt.legend()

    ax = plt.gca()
    ax.tick_params(direction='in', which='both')

    plt.show()  

    F = f.shape[0]

    I = X_meas[i, :F]      # Parte real → I
    Q = X_meas[i, F:]      # Parte imaginaria → Q

    plt.figure()
    plt.plot(I, Q, label="IQ trajectory")
    plt.scatter(I[0], Q[0], color='green', label="Start", zorder=3)
    plt.scatter(I[-1], Q[-1], color='red', label="End", zorder=3)

    plt.xlabel("I (Re{S21})")
    plt.ylabel("Q (Im{S21})")
    plt.title(f"IQ plot (kc = {kc_true[i]:.2e})")
    plt.legend()
    plt.axis("equal")   

    ax = plt.gca()
    ax.tick_params(direction='in', which='both')

    plt.show() 

    