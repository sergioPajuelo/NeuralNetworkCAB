import os
import numpy as np

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None  


from pathlib import Path
from typing import Sequence

# from lorentzian import lorentzian as lorentzian_cy
from libraries import lorentzian_cy, ParameterLimits, polysynth 
from libraries import MAX_LENGTH, MIN_LENGTH
from sctlib.analysis import Trace
from sctlib.analysis.trace.support._one_shot_fit import (
                                                    _guess_phase_delay,
                                                    guess_kappa,
                                                    guess_amplitude
                                                    )
from sctlib.tools import plot as uplt

PATH = os.path.dirname(__file__)

def padder_optimum(
    trace,
    *,
    max_F: int,
    keep_df: bool = True,
    df_override: float | None = None,
    return_debug: bool = False,
    noise: float = 0.0,
    order: int = 2,
):
    """
    Padding usando el poly del fit de baseline_polyfit (si existe).
    Si poly=None, hace fallback extrapolando baseline con derivadas (1ª + 2ª)
    estimadas en el extremo derecho (últimos 3 puntos).

    baseline_ext = poly(f_pad)  (si poly existe)
    baseline_ext = extrapolación local (si poly None)

    s_pad = s_base * baseline_ext * exp(1j*(delay*f + phi0))
    """

    f = np.asarray(trace.frequency, dtype=np.float64)
    s = np.asarray(trace.trace, dtype=np.complex128)
    Fi = f.size

    if Fi < 4:
        raise ValueError("Trace demasiado corta para padder_optimum (necesito Fi >= 4).")
    if Fi > max_F:
        raise ValueError(f"Fi ({Fi}) > max_F ({max_F}).")

    # 1) eje de frecuencia extendido
    if keep_df:
        df = float(f[-1] - f[-2])
    else:
        if df_override is None:
            raise ValueError("df_override debe darse si keep_df=False")
        df = float(df_override)

    pad_len = max_F - Fi
    f_pad = np.empty(max_F, dtype=np.float64)
    f_pad[:Fi] = f
    if pad_len > 0:
        f_pad[Fi:] = f[-1] + df * np.arange(1, pad_len + 1, dtype=np.float64)

    # 2) fase (delay + phi0)
    delay_guessed, phi0_guessed = _guess_phase_delay(trace)

    # 3) baseline y poly del fit
    baseline, poly = trace.baseline_polyfit(scale=20.0, order=order, plot=False)
    baseline = np.asarray(baseline, dtype=np.float64)


    if poly is not None:
        # Caso bueno: usar el poly (evaluado en Hz)
        baseline_ext = np.asarray(poly(f_pad), dtype=np.float64)
        used_fallback = False
    else:
        # Fallback
        used_fallback = True

        x0 = f[-1]
        x = np.array([f[-3] - x0, f[-2] - x0, f[-1] - x0], dtype=np.float64)  
        y = np.array([baseline[-3], baseline[-2], baseline[-1]], dtype=np.float64)

        
        A = np.vstack([x**2, x, np.ones_like(x)]).T
        try:
            a2, a1, a0 = np.linalg.solve(A, y)
        except np.linalg.LinAlgError:
            dx = float(f[-1] - f[-2])
            m0 = (baseline[-1] - baseline[-2]) / (dx + 1e-30)
            a2 = 0.0
            a1 = m0
            a0 = baseline[-1]

        baseline_ext = np.empty(max_F, dtype=np.float64)
        baseline_ext[:Fi] = baseline

        if pad_len > 0:
            x_pad = (f_pad[Fi:] - x0)  # >= 0
            baseline_ext[Fi:] = a2 * (x_pad**2) + a1 * x_pad + a0

        # Evitar valores negativos/raros (seguro mínimo)
        # (si no lo quieres, quita estas 2 líneas)
        bmin = max(1e-12, 0.05 * float(np.median(np.abs(baseline[-10:])) + 1e-30))
        baseline_ext = np.maximum(baseline_ext, bmin)
    
    # 5) fase extendida
    phase_delay_ext = np.exp(1j * (delay_guessed * f_pad + phi0_guessed))

    # 6) continuidad en el último punto real
    denom = baseline[-1] * np.exp(1j * (delay_guessed * f[-1] + phi0_guessed))
    s_base = s[-1] if np.abs(denom) < 1e-12 else (s[-1] / denom)

    # 7) salida
    s_corr = np.empty(max_F, dtype=np.complex128)
    s_corr[:Fi] = s

    if pad_len > 0:
        s_corr[Fi:] = s_base * baseline_ext[Fi:] * phase_delay_ext[Fi:]

        if np.abs(s_corr[Fi]) > 1e-15:
            s_corr[Fi:] *= s_corr[Fi - 1] / s_corr[Fi]

        if noise > 0.0:
            s_corr[Fi:] += (
                np.random.normal(0.0, noise, size=pad_len)
                + 1j * np.random.normal(0.0, noise, size=pad_len)
            )

    I_pad = s_corr.real.astype(np.float64)
    Q_pad = s_corr.imag.astype(np.float64)

    mask = np.zeros(max_F, dtype=np.float32)
    mask[:Fi] = 1.0

    if return_debug:
        debug = dict(
            Fi=Fi,
            df=float(df),
            delay_guessed=float(delay_guessed),
            phi0_guessed=float(phi0_guessed),
            s_base=s_base,
            poly_order=int(getattr(poly, "order", -1)) if poly is not None else -1,
            used_fallback=bool(used_fallback),
        )
        return f_pad, I_pad, Q_pad, mask, debug

    return f_pad, I_pad, Q_pad, mask


def _kc_tag(kc: float) -> str:
    # Ej: 1.234e+05 -> 1p234e05
    s = f"{kc:.6e}"          
    s = s.replace("+", "")   # e+05 -> e05
    s = s.replace(".", "p")  # 1.234 -> 1p234
    return s


def lorentzian_generator(
    n_samples            : int,
    noise_std_signal     : float | tuple[float, float] = 0.0,
    save_debug_dataset   : bool = False,
    save_dir             : str = "Dataset",
    debug_dpi            : int = 300,
):
    
    """
      - Genera cada muestra con un Fi aleatorio (de frequency_points).
      - Devuelve X_meas y X_clean con DIMENSIÓN FIJA: 2*max_F (padding físico).
      - Devuelve también F (frecuencias padded), F_len y mask.
    """
    
    out_dir = Path(os.path.join(PATH, save_dir))
    sample_dir = out_dir 
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    if save_debug_dataset:
        out_dir.mkdir(parents=True, exist_ok=True)

    kappai_true = np.zeros(n_samples, dtype=np.float32)
    kappac_true = np.zeros(n_samples, dtype=np.float32)

    
    X_meas  = np.zeros((n_samples, 2 * MAX_LENGTH), dtype=np.float32)
    X_clean = np.zeros((n_samples, 2 * MAX_LENGTH), dtype=np.float32)
    
    F_len = np.zeros( n_samples, dtype=np.int32)
    F     = np.zeros((n_samples, MAX_LENGTH), dtype=np.float64)
    mask  = np.zeros((n_samples, MAX_LENGTH), dtype=np.float32)


    progress_marks = {25, 50, 75, 100}
    printed_marks = set()

    for index in range(n_samples):
        percent = int(100 * (index + 1) / n_samples)

        for m in sorted(progress_marks):
            if percent >= m and m not in printed_marks:
                print(f"{m}% de la generación completado")
                printed_marks.add(m)
            
        # Choose one random length for the trace data, stores the length in an
        # auxiliar array F_len, and asigns mask info.
        trace_length           = np.random.randint(MIN_LENGTH, MAX_LENGTH + 1)
        F_len[index]           = trace_length
        mask[index, :trace_length] = 1.0
        
        # Asign values to lorentzian variables
        params = ParameterLimits.sample()
        sweep_factor = params['sweep_factor']
        
        kappac_true[index] = params['kappac']
        kappai_true[index] = params['kappai']

        delta_f_max = sweep_factor * params['kappa']
        
        # Generate the frequency array
        f_i = np.linspace(params['fr'] - delta_f_max, 
                          params['fr'] + delta_f_max, 
                          trace_length, dtype=np.float64)


        s0 = lorentzian_cy(f_i, 
                           params['ac'],
                           params['dt'],
                           params['phi'],
                           params['r'],
                           params['kappa'],
                           params['dphi'],
                           params['fr']
                           )
        
        y = s0
        x = f_i
        # Calculate step sizes between consecutive points
        dy = np.diff(y)  # Complex differences
        step_sizes = np.abs(dy)  # Magnitude of change
        
        # Find large jumps
        threshold = 10 * np.median(step_sizes)  # Adaptive threshold
        large_jumps = np.where(step_sizes > threshold)[0]
        
        print(f"Trace #{index}: Large jumps at indices: {large_jumps}")
        print(f"Corresponding x values: {x[large_jumps] if len(large_jumps) > 0 else 'None'}")
        
        # Zoom in around the jump
        if len(large_jumps) > 0:
            jump_idx = large_jumps[int(len(large_jumps)/2)]
            window = 5  # Points to show before/after
            
            print(f"\nData around jump at index {jump_idx}:")
            print("Index | x value | Real | Imag | Magnitude")
            print("-" * 50)
            
            for i in range(max(0, jump_idx-window), min(len(y), jump_idx+window+2)):
                print(f"{i:5d} | {x[i]:8.4f} | {y.real[i]:8.4f} | {y.imag[i]:8.4f} | {np.abs(y[i]):8.4f}")
                
        if np.isnan((abs(s0))).any():
            print(f"WARNING, trace number {index + 1} presents NaN discontinuities")
            print(s0)
            
        if np.isinf((abs(s0))).any():
            print(f"WARNING, trace number {index + 1} presents Inf discontinuities")
            print(s0)
        
        if (~np.isfinite(s0.real)).any() or (~np.isfinite(s0.imag)).any():
            print("BAD: non-finite in complex components")
            
        print('Frequency length', len(f_i))    
        print('Trace length', len(s0))

            
        sorted_indices = np.argsort(f_i)
        
        poly_deg_range=np.random.choice(range(1, 5+1))
        
        s_clean, poly = polysynth(
            f_i, s0,
            poly_order = poly_deg_range,
            poly_coeff_scale=np.random.uniform(0.02, 0.06),
        )
        
        # # Synthetic IQ imbalance(I don't want to use it for the moment).
        # eps = np.random.uniform(-0.003, 0.003)
        # I_clean = s_clean.real * (1 + eps)
        # Q_clean = s_clean.imag * (1 - eps)
        # s_clean = I_clean + 1j * Q_clean

        trace_clean = Trace(frequency=f_i, trace=s_clean)

        try: 
            f_pad, I_clean_pad, Q_clean_pad, _ = padder_optimum(
                trace_clean,
                max_F=MAX_LENGTH,
                order=poly_deg_range,
            )
            
        except Exception as e:
            fig, ax = plt.subplots(2,2)
            ax[1,0].plot(s0.real, s0.imag)
            ax[1,0].axis('equal')
            ax[0,0].plot(f_i, abs(s0))
            ax[0,1].plot(f_i, np.unwrap(np.atan2(s0.imag, s0.real)))
            ax[0,0].tick_params(which = 'both', direction = 'in')
            ax[0,1].tick_params(which = 'both', direction = 'in')
            ax[1,0].tick_params(which = 'both', direction = 'in')
            ax[1,1].tick_params(which = 'both', direction = 'in')
            plt.show()
            print(f'Error while padding trace {index+1}.\nReason: {e}')
            continue

        s_clean_pad = I_clean_pad.astype(np.float64) + 1j * Q_clean_pad.astype(np.float64)
        s_meas_pad = s_clean_pad.copy()

        if isinstance(noise_std_signal, tuple):
            sig = float(np.random.uniform(noise_std_signal[0], noise_std_signal[1]))
        else:
            sig = float(noise_std_signal)

        F[index, :] = f_pad

        X_clean[index, :MAX_LENGTH] = s_clean_pad.real.astype(np.float32)
        X_clean[index, MAX_LENGTH:2*MAX_LENGTH] = s_clean_pad.imag.astype(np.float32)

        X_meas[index, :MAX_LENGTH] = s_meas_pad.real.astype(np.float32)
        X_meas[index, MAX_LENGTH:2*MAX_LENGTH] = s_meas_pad.imag.astype(np.float32)


        if save_debug_dataset:
            file_name = f"sample_{index:03d}"
 
            tag = _kc_tag(float(kappac_true[index]))

            Fi_local = int(F_len[index])

            # Para plots vs frecuencia 
            f_plot, f_unit, _ = uplt.guess_magnitude_order(F[index, :Fi_local], 'Hz')
                        
            I_real = X_meas[index, :Fi_local]
            Q_real = X_meas[index, MAX_LENGTH:MAX_LENGTH + Fi_local]
            amp_real = np.sqrt(I_real**2 + Q_real**2)
            phase_real = np.unwrap(np.arctan2(Q_real, I_real))

            # Para IQ: lo que ve la NN (Fmax completo)
            I_full = X_meas[index, :MAX_LENGTH]
            Q_full = X_meas[index, MAX_LENGTH:2*MAX_LENGTH]
            
            
            fig, ax = plt.subplots(2, 2, dpi = debug_dpi, constrained_layout=True)
            fig.suptitle(f"kappac = {kappac_true[index]:.2e} Hz, kappai = {kappai_true[index]:.2e} Hz")
            ax[0,0].plot(f_plot[sorted_indices], amp_real[sorted_indices], linestyle="-")
            ax[0,0].tick_params(direction='in', which='both')
            ax[0,1].plot(f_plot[sorted_indices], phase_real[sorted_indices], linestyle="-")
            ax[0,1].tick_params(direction='in', which='both')
            ax[1,0].plot(I_real[sorted_indices], Q_real[sorted_indices], linestyle="-")
            ax[1,0].tick_params(direction='in', which='both')
            ax[1,0].axis('equal')
            ax[1,1].plot(f_plot[sorted_indices], abs(s0[sorted_indices]), linestyle="-", color='royalblue')
            ax[1,1].plot(f_plot, abs(params['ac'] * poly), linestyle="--", color='tomato')
            ax[1,1].tick_params(direction='in', which='both')
            fig.savefig(sample_dir / file_name)
            plt.close()
            

            # Params_{kc}.dat
            params_path = sample_dir / f"{file_name}_params.dat"
            with open(params_path, "w", encoding="utf-8") as fp:
                fp.write(f"i={index}\n")
                fp.write(f"Fi={Fi_local}\n")
                fp.write(f"max_F={MAX_LENGTH}\n")
                fp.write(f"kc_true={float(kappac_true[index])}\n")
                fp.write(f"kappai={float(kappai_true[index])}\n")
                fp.write(f"ac={params['ac']}\n")
                fp.write(f"dt={params['dt']}\n")
                fp.write(f"fr={params['fr']}\n")
                fp.write(f"dphi={params['dphi']}\n")
                fp.write(f"phi={params['phi']}\n")
                fp.write(f"poly_deg_range={int(poly_deg_range)}\n")
                # fp.write(f"eps={eps}\n")
                fp.write(f"nRange={float(sweep_factor)}\n")
                fp.write(f"delta_f_max={float(delta_f_max)}\n")
                fp.write(f"kappa={params['kappa']}\n")
                fp.write(f"r={params['r']}\n")
                fp.write(f"noise_std_signal_used={float(sig)}\n")

    return F, X_meas, X_clean, kappac_true, kappai_true, F_len, mask



if __name__ == "__main__":
    
    """ F, X_meas, X_clean, kc_true, kappai_true, F_len, mask = lorentzian_generator(
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


    I_full = X_meas[i, :max_F]
    Q_full = X_meas[i, max_F:2*max_F]

    plt.figure()
    plt.plot(I_full, Q_full, label="IQ trajectory (full F)")
    plt.scatter(I_full[0], Q_full[0], label="Start", zorder=3)
    plt.scatter(
        I_full[Fi-1], Q_full[Fi-1],
        label="End real data", zorder=3
    )
    plt.xlabel("I (Re{S21})")
    plt.ylabel("Q (Im{S21})")
    plt.title(f"IQ plot FULL (kc = {kc_true[i]:.2e})")
    plt.legend()
    plt.axis("equal")

    ax = plt.gca()
    ax.tick_params(direction="in", which="both")
    plt.show()


    
    I = X_meas[i, :Fi]
    Q = X_meas[i, max_F:max_F + Fi]
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
    plt.show() """

    F, X_meas, X_clean, kc_true, kappai_true, F_len, mask = lorentzian_generator(
        n_samples          = 100,
        noise_std_signal   = (0.0, 0.01),
        save_debug_dataset = True,
        save_dir           = "Dataset_demo",
    )

