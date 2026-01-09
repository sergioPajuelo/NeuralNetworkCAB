import numpy as np
from lorentzian import lorentzian as lorentzian_cy
import matplotlib.pyplot as plt
from sctlib.analysis import Trace
from sctlib.analysis.trace.support._one_shot_fit import _guess_phase_delay, polyfit_baseline, guess_kappa, guess_amplitude

def padder_optimum(
    trace,
    *,
    max_F: int,
    kappa_guessed: float,
    order: int,
    interval: tuple[float, float] | None = None,
    # extensión de frecuencia
    keep_df: bool = True,
    df_override: float | None = None,
    return_debug: bool = False,
):
    """
    Padding "optimum":
      1) Calcula p(f) con polyfit_baseline y phase_delay con _guess_phase_delay
      2) Extiende f hasta max_F
      3) Rellena s desde Fi..max_F usando la prolongación del padder:
            padder_ext(f) = p_ext(f) * exp(j*(delay*f + phi0))
         y una base constante s_base.

    Returns
    -------
    f_pad : (max_F,)
    I_pad : (max_F,)
    Q_pad : (max_F,)
    mask  : (max_F,)  (1.0 real, 0.0 padding)
    """

    f = np.asarray(trace.frequency, dtype=np.float64)
    s = np.asarray(trace.trace, dtype=np.complex128)
    Fi = f.size

    if Fi < 3:
        raise ValueError("Trace demasiado corta para padder_optimum (Fi < 3).")
    if Fi > max_F:
        raise ValueError(f"Fi ({Fi}) > max_F ({max_F}).")
    if kappa_guessed is None:
        raise ValueError("Debes pasar kappa_guessed para polyfit_baseline.")

    # Phase delay
    delay_guessed, delayAtZero_guessed = _guess_phase_delay(trace)
    phase_delay = np.exp(1j * (delay_guessed * f + delayAtZero_guessed))

    if interval is None:
        raise ValueError("Debes pasar interval=(f_min, f_max) en Hz (fuera del dip).")

    amplitude_base, p = polyfit_baseline(
        trace,
        kappa_guessed,
        order=int(order),
        interval=float(interval),   
        save=False,
        verbose=False,
        plot=False,
        standalone=False,
    )

    poly = np.asarray(amplitude_base, dtype=np.float64)

    padder_real = poly * phase_delay  

    # Padding
    f_pad = np.empty(max_F, dtype=np.float64)
    f_pad[:Fi] = f

    if max_F > Fi:
        if df_override is not None:
            df = float(df_override)
        else:
            if not keep_df:
                # si no quieres df, por defecto uso el último df
                df = float(f[-1] - f[-2])
            else:
                # df medio 
                df = float(np.mean(np.diff(f)))
        f_pad[Fi:] = f[-1] + df * np.arange(1, (max_F - Fi) + 1, dtype=np.float64)

    # Phase delay sobre f_pad
    phase_delay_ext = np.exp(1j * (delay_guessed * f_pad + delayAtZero_guessed))  # (max_F,)

    # Extiendo p(f)
    poly_ext = np.empty_like(f_pad, dtype=np.float64)
    poly_ext[:Fi] = poly
    poly_ext[Fi:] = poly[-1]   # baseline constante

    padder_ext = poly_ext * phase_delay_ext

    s_corr = np.empty(max_F, dtype=np.complex128)
    s_corr[:Fi] = s

    # base constante en el "espacio sin padder" (para que continúe suave)
    eps = 1e-12
    denom = padder_real[-1]
    if np.abs(denom) < eps:
        # fallback: si el padder es ~0, uso el propio valor final como base
        s_base = s[-1]
        s_corr[Fi:] = s_base
    else:
        s_base = s[-1] / denom
        s_corr[Fi:] = s_base * padder_ext[Fi:]

    # ---- 7) Salida I/Q y máscara ----
    I_pad = s_corr.real.astype(np.float64)
    Q_pad = s_corr.imag.astype(np.float64)

    mask = np.zeros(max_F, dtype=np.float32)
    mask[:Fi] = 1.0

    if return_debug:
        debug = dict(
            Fi=Fi,
            delay_guessed=float(delay_guessed),
            delayAtZero_guessed=float(delayAtZero_guessed),
            kappa_guessed=float(kappa_guessed),
            poly_real=poly,
            poly_ext=poly_ext,
            padder_real=padder_real,
            padder_ext=padder_ext,
        )
        return f_pad, I_pad, Q_pad, mask, debug

    return f_pad, I_pad, Q_pad, mask



def padder_1db_tail(
    f, I, Q,
    max_F,
    db_threshold=1.0,
    noise_std=0.0,
):
    """
    Padding físico desde el final de la traza.
    
    Parámetros
    ----------
    f, I, Q : arrays (Fi,)
        Traza real
    max_F : int
        Longitud final deseada
    db_threshold : float
        Umbral en dB (por defecto 1 dB)
    noise_std : float
        Ruido gaussiano opcional en el padding

    Returns
    -------
    f_pad, I_pad, Q_pad : arrays (max_F,)
    mask : (max_F,)  -> 1 real, 0 padding
    """
    Fi = len(f)
    assert Fi <= max_F

    amp = np.sqrt(I**2 + Q**2)

    # umbral = 1 dB
    amp_ref = amp[-1]
    amp_min = amp_ref * 10 ** (-db_threshold / 20)

    idx = Fi - 1
    while idx > 0 and amp[idx] > amp_min:
        idx -= 1

    # tramo válido para padding
    I_tail = I[idx:Fi]
    Q_tail = Q[idx:Fi]
    f_tail = f[idx:Fi]

    pad_len = max_F - Fi

    if pad_len > 0:
        # repetir el tramo final
        rep = int(np.ceil(pad_len / len(I_tail)))

        I_pad_tail = np.tile(I_tail, rep)[:pad_len]
        Q_pad_tail = np.tile(Q_tail, rep)[:pad_len]

        if noise_std > 0:
            I_pad_tail += np.random.normal(0, noise_std, pad_len)
            Q_pad_tail += np.random.normal(0, noise_std, pad_len)

        # freq extrapolada
        df = f[-1] - f[-2] if Fi > 1 else 1.0
        f_pad_tail = f[-1] + df * np.arange(1, pad_len + 1)

        I_out = np.concatenate([I, I_pad_tail])
        Q_out = np.concatenate([Q, Q_pad_tail])
        f_out = np.concatenate([f, f_pad_tail])

    else:
        I_out, Q_out, f_out = I, Q, f

    # Mask
    mask = np.zeros(max_F, dtype=np.float32)
    mask[:Fi] = 1.0

    return f_out, I_out, Q_out, mask


def _random_poly_response(x: np.ndarray,
                          deg_min: int = 1,
                          deg_max: int = 5,
                          coeff_scale: float = 0.25
                          ) -> np.ndarray:
        
    deg = int(np.random.randint(deg_min, deg_max + 1))
    coef = np.random.normal(0.0, coeff_scale, size=deg + 1)
    coef[0] = 0.0

    a = np.zeros_like(x, dtype=np.float64)
    xp = np.ones_like(x, dtype=np.float64)
    for k in range(deg + 1):
        a += coef[k] * xp
        xp *= x
    
    # In this way, the polynomial is always positive (NEW)
    a = np.exp(a)
    
    # if ensure_positive:
    #     a = a - np.min(a)
    #     a = 0.2 + a
        
    a = a / (np.mean(np.abs(a)) + 1e-12)
    return a


def _apply_random_poly_to_magnitude_only(
    f: np.ndarray,
    s: np.ndarray,
    poly_deg_range: int,
    poly_coeff_scale: float = 0.25,
) -> np.ndarray:
    
    f = f.astype(np.float64)

    mag = np.abs(s)
    phase = np.unwrap(np.angle(s))  

    # eje normalizado [-1,1] para el polinomio
    f0 = float(np.mean(f))
    span = float(np.ptp(f)) + 1e-12
    x = (f - f0) / (span / 2.0)

    a = _random_poly_response(
        x,
        deg_min=1,
        deg_max=5,
        coeff_scale=poly_coeff_scale,
    )

    mag2 = mag * a

    return mag2 * np.exp(1j * phase)


def lorentzian_generator(
    n_samples: int,
    cavity_params: dict,
    kc_limits: tuple[float, float],
    frequency_points=(2000, 5000, 6000, 10000, 15000, 20000),
    noise_std_signal: float | tuple[float, float] = 0.0,
    pad_db_threshold: float = 1.0,
    pad_noise_std: float = 1e-4,
):
    """
      - Genera cada muestra con un Fi aleatorio (de frequency_points).
      - Devuelve X_meas y X_clean con DIMENSIÓN FIJA: 2*max_F (padding físico).
      - Devuelve también F (frecuencias padded), F_len y mask.
    """
    frequency_points = np.asarray(frequency_points, dtype=int)
    max_F = int(frequency_points.max())

    kappai_true = np.zeros(n_samples, dtype=np.float32)

    log_lo, log_hi = np.log(kc_limits[0]), np.log(kc_limits[1])
    kc_true = np.exp(np.random.uniform(log_lo, log_hi, size=n_samples)).astype(np.float32)

    X_meas  = np.zeros((n_samples, 2 * max_F), dtype=np.float32)
    X_clean = np.zeros((n_samples, 2 * max_F), dtype=np.float32)

    F = np.zeros((n_samples, max_F), dtype=np.float64)
    F_len = np.zeros(n_samples, dtype=np.int32)
    mask = np.zeros((n_samples, max_F), dtype=np.float32)

    freqs = np.array(frequency_points, dtype=int)
    p = np.array([0.05, 0.05, 0.10, 0.25, 0.25, 0.30], dtype=float)  
    p = p / p.sum()

    for i, kc in enumerate(kc_true):
        Fi = int(np.random.choice(freqs, p=p))
        F_len[i] = Fi
        mask[i, :Fi] = 1.0

        ac = float(np.exp(np.random.uniform(np.log(cavity_params["ac"][0]),
                                            np.log(cavity_params["ac"][1]))))
        dt = float(np.random.uniform(*cavity_params["dt"]))
        fr = float(np.random.uniform(*cavity_params["fr"]))
        dphi = float(np.random.uniform(*cavity_params["dphi"]))

        kappai = float(np.exp(np.random.uniform(np.log(cavity_params["kappai"][0]),
                                                np.log(cavity_params["kappai"][1]))))
        kappai_true[i] = kappai

        phi = float(np.random.uniform(*cavity_params["phi"]))

        kc_f = float(kc)
        kappa = kappai + kc_f
        r = kc_f / kappa

        nRange = np.random.uniform(100, 500)
        delta_f_max = nRange * kappa

        f_i = np.linspace(fr - delta_f_max, fr + delta_f_max, Fi, dtype=np.float64)

        s0 = lorentzian_cy(f_i, ac, dt, phi, r, kappa, dphi, fr)

        poly_deg_range=np.random.choice(range(1, 5+1))
        s_clean = _apply_random_poly_to_magnitude_only(
            f_i, s0,
            poly_deg_range=poly_deg_range,
            poly_coeff_scale=np.random.uniform(0.02, 0.06),
        )

        c0 = (np.random.normal(0.0, 0.05) + 1j*np.random.normal(0.0, 0.05))
        s_clean = s_clean + c0

        eps = np.random.uniform(-0.03, 0.03)
        I_clean = s_clean.real * (1 + eps)
        Q_clean = s_clean.imag * (1 - eps)
        s_clean = I_clean + 1j * Q_clean

        s_meas = s_clean.copy()

        if isinstance(noise_std_signal, tuple):
            sig = float(np.random.uniform(noise_std_signal[0], noise_std_signal[1]))
        else:
            sig = float(noise_std_signal)

        if sig > 0.0:
            s_meas = s_meas + (
                np.random.normal(0.0, sig, size=Fi) +
                1j*np.random.normal(0.0, sig, size=Fi)
            )

        trace_clean = Trace(frequency=f_i, trace=s_clean)
        trace_meas  = Trace(frequency=f_i, trace=s_meas)

        f_pad, I_clean_pad, Q_clean_pad, _ = padder_optimum(
            trace_clean,
            max_F=max_F,
            kappa_guessed=kappa,
            order=2,
            interval=20.0,
        )

        f_pad, I_meas_pad, Q_meas_pad, _ = padder_optimum(
            trace_meas,
            max_F=max_F,
            kappa_guessed=kappa,
            order=2,
            interval=20.0,
        )

        """ f_pad, I_clean_pad, Q_clean_pad, _mask_clean = padder_1db_tail(
            f_i,
            s_clean.real.astype(np.float64),
            s_clean.imag.astype(np.float64),
            max_F=max_F,
            db_threshold=pad_db_threshold,
            noise_std=pad_noise_std,
        ) 

        f_pad2, I_meas_pad, Q_meas_pad, _mask_meas = padder_1db_tail(
            f_i,
            s_meas.real.astype(np.float64),
            s_meas.imag.astype(np.float64),
            max_F=max_F,
            db_threshold=pad_db_threshold,
            noise_std=pad_noise_std,
        ) """

        F[i, :] = f_pad

        X_clean[i, :max_F] = I_clean_pad.astype(np.float32)
        X_clean[i, max_F:2*max_F] = Q_clean_pad.astype(np.float32)

        X_meas[i, :max_F] = I_meas_pad.astype(np.float32)
        X_meas[i, max_F:2*max_F] = Q_meas_pad.astype(np.float32)

    return F, X_meas, X_clean, kc_true, kappai_true, F_len, mask


if __name__ == "__main__":
    cavity_params = {
        "ac"     : (0.3, 1.8),
        "dt"     : (-1e-7, 0),
        "phi"    : (-np.pi, np.pi),
        "dphi"   : (-np.pi/4, np.pi/4),
        "kappai" : (1e2, 1e5), 
        "fr"     : (7.30e8 - 2e6, 7.50e8 + 2e6)
    }

    kc_limits = (1e4, 1e5)


    F, X_meas, X_clean, kc_true, kappai_true, F_len, mask = lorentzian_generator(
        n_samples=3,
        cavity_params=cavity_params,
        kc_limits=kc_limits,
        frequency_points=[2000, 5000, 6000, 10000, 15000, 20000],
        noise_std_signal=0.0,
    )

    i=2
    max_F = F.shape[1]

    re = X_meas[i, :max_F]
    im = X_meas[i, max_F:2*max_F]
    f_pad = np.arange(max_F)

    mag = np.sqrt(re**2 + im**2)

    plt.figure()
    plt.plot(f_pad, mag)
    plt.axvline(F_len[i], color="r", linestyle="--", label="end real data")
    plt.legend()
    plt.title("Trace with padding (what NN sees)")
    plt.show()

    i = 2
    Fi = F_len[i]       
    f = F[i, :Fi] 
    max_F = F.shape[1]
    re = X_meas[i, :Fi]
    im = X_meas[i, max_F:max_F + Fi]

    mag = np.sqrt(re**2 + im**2)
    phase = np.unwrap(np.arctan2(im, re))

    f_GHz = f * 1e-9

    fig, ax = plt.subplots(2, 1, dpi=300, figsize=(12, 7), constrained_layout=True, sharex=True)
    ax[0].plot(f_GHz, mag, linestyle="--")
    ax[1].plot(f_GHz, phase)
    ax[1].set_xlabel("Frequency [GHz]")
    ax[0].set_ylabel("Amplitude")
    ax[1].set_ylabel("Phase [rad]")
    ax[0].set_title(f"Magnitude (kc = {kc_true[i]:.2e})")
    ax[1].set_title(f"Phase (kc = {kc_true[i]:.2e})")
    ax[0].tick_params(direction='in', which='both')
    ax[1].tick_params(direction='in', which='both')
    plt.show()

    I = X_meas[i, :Fi]
    Q = X_meas[i, max_F:max_F + Fi]

    plt.figure()
    plt.plot(I, Q, label="IQ trajectory")
    plt.scatter(I[0], Q[0], label="Start", zorder=3)
    plt.scatter(I[-1], Q[-1], label="End", zorder=3)
    plt.xlabel("I (Re{S21})")
    plt.ylabel("Q (Im{S21})")
    plt.title(f"IQ plot (kc = {kc_true[i]:.2e})")
    plt.legend()
    plt.axis("equal")
    ax = plt.gca()
    ax.tick_params(direction='in', which='both')
    plt.show()


    

    mag = np.sqrt(I**2 + Q**2)        
    f_GHz = f * 1e-9

    fig, ax = plt.subplots(1, 1, dpi=300, figsize=(9.7, 4.0), constrained_layout=True)

    ax.plot(f_GHz, I, label="Re(S21)")
    ax.plot(f_GHz, Q, label="Im(S21)")
    ax.plot(f_GHz, mag, label="|S21|", linestyle="--")

    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Lorentzian clean (kc = {kc_true[i]:.2e})")
    ax.legend(frameon=False)

    ax.tick_params(direction="in", which="both", top=True, right=True)
    plt.show()

