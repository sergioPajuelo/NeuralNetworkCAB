from sctlib.analysis import Trace
import torch
""" trace = Trace()
trace.load_trace(source='cab')
results = trace.do_fit(baseline=(3, 0.7), mode="one-shot", verbose=True)
print(results["one-shot"].final) """

import numpy as np

def inspect_trace_centering(dat_path):
    data = np.genfromtxt(dat_path, comments="#")

    if data.ndim == 1:
        data = data.reshape(1, -1)
    data = data[:, :3]

    f = data[:, 0].astype(np.float64)
    I = data[:, 1].astype(np.float64)
    Q = data[:, 2].astype(np.float64)

    m = np.isfinite(f) & np.isfinite(I) & np.isfinite(Q)
    f, I, Q = f[m], I[m], Q[m]

    if f.size < 2:
        print("Muy pocos puntos válidos tras filtrar NaNs.")
        return

    order = np.argsort(f)
    f, I, Q = f[order], I[order], Q[order]

    amp = np.sqrt(I**2 + Q**2)

    # notch = mínimo de amplitud
    idx_min = int(np.argmin(amp))
    N = len(f)

    print("\n--- Trace centering inspection ---")
    print(f"N points (valid): {N}")
    print(f"f_min      = {f[0]:.6e} Hz")
    print(f"f_max      = {f[-1]:.6e} Hz")
    print(f"f_notch    = {f[idx_min]:.6e} Hz")
    print(f"index_min  = {idx_min}")
    print(f"center_idx = {N//2}")
    print(f"offset     = {idx_min - N//2} points")
    print(f"offset %   = {(idx_min - N//2)/N*100:.2f}%")

inspect_trace_centering("Experimental_Validation_Dataset/Line1_CrossPolCheck_LER3_IQmeas_PDUT-86.0dBm_T100.0mK_Tbb4.5K_0.73345GHzto0.73695GHz_Rate1.0e+06_Sample1000000_BW1.0Hz_Navg1_RF0dBm_LO0dBm.dat")


""" import numpy as np
import matplotlib.pyplot as plt

def load_dat_real_imag(path: str):
    # 1) Intento rápido: saltar 1 línea (cabecera)
    try:
        data = np.loadtxt(path, skiprows=1)
    except ValueError:
        # 2) Alternativa robusta: genfromtxt ignora líneas no numéricas
        data = np.genfromtxt(path, skip_header=1)

    # Si hubiera columnas extra, nos quedamos con las 3 primeras
    data = data[:, :3]

    f  = data[:, 0].astype(np.float64)
    re = data[:, 1].astype(np.float64)
    im = data[:, 2].astype(np.float64)

    return f, re, im


# ---- CARGA DEL .dat ----
path = r"src/sctlib/analysis/trace/NeuralNetwork/Real_traces/Line4_DAS_PRIMA_LER8_VNAmeas_PDUT-99.0dBm_T20.0mK_0.8492GHzto0.8502GHz_IF20Hz.dat"
f, re, im = load_dat_real_imag(path)

# ---- MAGNITUD ----
mag = np.abs(re + 1j * im)

# ---- PLOT ----
plt.figure()
plt.plot(f, re, label="Re(S21)")
plt.plot(f, im, label="Im(S21)")
plt.plot(f, mag, "--", label="|S21|")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title("Measured trace (.dat)")
plt.legend()

ax = plt.gca()
ax.tick_params(direction="in", which="both")

plt.show() """