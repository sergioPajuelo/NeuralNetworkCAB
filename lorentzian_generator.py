import numpy as np
from lorentzian import lorentzian as lorentzian_cy
import matplotlib.pyplot as plt

def _random_poly_response(x: np.ndarray,
                          deg_min: int = 1,
                          deg_max: int = 5,
                          coeff_scale: float = 0.25
                          ) -> np.ndarray:
    
    # Removed ensure_positive. Magnitude always positive.
    
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
    poly_deg_range: tuple[int, int] = (1, 5),
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
        deg_min=poly_deg_range[0],
        deg_max=poly_deg_range[1],
        coeff_scale=poly_coeff_scale,
    )

    mag2 = mag * a

    return mag2 * np.exp(1j * phase)


def lorentzian_generator(
    n_samples: int,
    cavity_params: dict,
    kc_limits: tuple[float, float],
    frequency_points: tuple[int] = [2000, 5000, 10000, 15000, 20000],
    noise_std_signal: float | tuple[float, float] = 0.0,
):
    kappai_true = np.zeros(n_samples)
    log_lo, log_hi = np.log(kc_limits[0]), np.log(kc_limits[1])
    kc_true = np.exp(np.random.uniform(log_lo, log_hi, size=n_samples))

    frequency_points = int(np.random.choice(frequency_points))
    X_meas  = np.empty((n_samples, 2 * frequency_points), dtype=np.float32)
    X_clean = np.empty((n_samples, 2 * frequency_points), dtype=np.float32)

    for i, kc in enumerate(kc_true):
        
        print(i)

        
        ac = float(np.exp(np.random.uniform(np.log(cavity_params["ac"][0]),
                                            np.log(cavity_params["ac"][1]))))
        dt = float(np.random.uniform(*cavity_params["dt"]))
        fr = float(np.random.uniform(*cavity_params["fr"]))
        dphi = float(np.random.uniform(*cavity_params["dphi"]))

        kappai = float(np.exp(np.random.uniform(np.log(cavity_params["kappai"][0]),
                                                np.log(cavity_params["kappai"][1]))))
        kappai_true[i] = kappai

        phi = float(np.random.uniform(*cavity_params["phi"]))

        kc = float(kc)
        kappa = kappai + kc
        r = kc / kappa
        
        print('ac', ac)
        print('kappac', kc)
        print('kappai', kappai)
        print('dt', dt)
        print('phi0', phi)
        print('r', r)
        print('dphi', dphi)
        print('fr', fr)
        
        # NEW: Changed ka_limits[1] with kappa. The actual value.
        nRange = np.random.uniform(100, 500)
        delta_f_max = nRange * kappa
        f = np.linspace(fr - delta_f_max, fr + delta_f_max, frequency_points, dtype=np.float64)

        s0 = lorentzian_cy(f, ac, dt, phi, r, kappa, dphi, fr)

        s_clean = _apply_random_poly_to_magnitude_only(
            f, s0,
            poly_deg_range=(1, 5),
            poly_coeff_scale=np.random.uniform(0.02, 0.06),
        )

        c0 = (np.random.normal(0.0, 0.05) + 1j*np.random.normal(0.0, 0.05))
        s_clean = s_clean + c0

        eps = np.random.uniform(-0.03, 0.03)
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

        X_clean[i, :frequency_points] = s_clean.real.astype(np.float32)
        X_clean[i, frequency_points:] = s_clean.imag.astype(np.float32)
        X_meas[i, :frequency_points]  = s_meas.real.astype(np.float32)
        X_meas[i, frequency_points:]  = s_meas.imag.astype(np.float32)

    return f, X_meas, X_clean, kc_true.astype(np.float32), kappai_true


if __name__ == "__main__":
    cavity_params = {
        "ac"     : (0.3, 1.8),
        "dt"     : (-1e-7, 0),
        "phi"    : (-np.pi, np.pi),
        "dphi"   : (-np.pi/4, np.pi/4),
        "kappai" : (1e2, 1e5), # NEW: changed parameters to a more physical range.
        "fr"     : (7.30e8 - 2e6, 7.50e8 + 2e6)
    }

    kc_limits = (1e4, 1e5)

    f, X_meas, X_clean, kc_true, kappai_true = lorentzian_generator(
        n_samples=3,
        cavity_params=cavity_params,
        kc_limits=kc_limits,
        frequency_points=[2000, 5000, 10000, 15000, 20000],
        noise_std_signal=0.0,
    )

    i = 2
    F = f.shape[0]
    re = X_meas[i, :F]
    im = X_meas[i, F:]
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

    I = X_meas[i, :F]
    Q = X_meas[i, F:]

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


    I = X_meas[i, :F]
    Q = X_meas[i, F:]

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

