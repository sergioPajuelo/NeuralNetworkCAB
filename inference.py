import io
import numpy as np
import torch
import matplotlib.pyplot as plt

from lorentzian import lorentzian as lorentzian_cy
from network import Net
from sctlib.analysis import Trace


def load_trained_model(model_path: str) -> tuple[Net, dict]:
    ckpt = torch.load(model_path, map_location="cpu")

    net = Net(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
        n_units=int(ckpt["n_units"]),
        epochs=1,
        lr=1e-3,
    )
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net, ckpt


def load_iq_from_dat(dat_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        data = np.loadtxt(dat_path, comments="#", delimiter=None)
    except Exception:
        data = np.genfromtxt(dat_path, comments="#", delimiter=None)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        raise ValueError(f"Esperaba al menos 3 columnas (freq, I, Q). Tengo {data.shape[1]} en {dat_path}")

    f = data[:, 0].astype(np.float64)
    I = data[:, 1].astype(np.float64)
    Q = data[:, 2].astype(np.float64)

    m = np.isfinite(f) & np.isfinite(I) & np.isfinite(Q)
    f, I, Q = f[m], I[m], Q[m]

    if f.size < 2:
        raise ValueError("El fichero tiene muy pocos puntos válidos.")
    order = np.argsort(f)
    return f[order], I[order], Q[order]



def resample_iq_to_npoints(f: np.ndarray, I: np.ndarray, Q: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Re-muestrea I y Q a n_points uniformemente en frecuencia usando interpolación lineal.
    """
    fmin, fmax = float(f[0]), float(f[-1])
    f_new = np.linspace(fmin, fmax, n_points, dtype=np.float64)

    I_new = np.interp(f_new, f, I).astype(np.float64)
    Q_new = np.interp(f_new, f, Q).astype(np.float64)
    return f_new, I_new, Q_new



def build_nn_input_from_dat_iq(dat_path: str, input_dim: int) -> np.ndarray:
    """
    Construye el input para la NN como [I || Q].
    """
    if input_dim % 2 != 0:
        raise ValueError("input_dim debe ser par (I||Q).")
    n_points = input_dim // 2

    f, I, Q = load_iq_from_dat(dat_path)
    _, I_rs, Q_rs = resample_iq_to_npoints(f, I, Q, n_points)

    X = np.concatenate([I_rs, Q_rs], axis=0).astype(np.float32)[None, :]
    return X

def predict_kc_nn(net: Net, X: np.ndarray) -> float:
    with torch.no_grad():
        y_pred = net.predict(X)  
    return float(np.exp(np.asarray(y_pred).reshape(-1)[0]))


def main():
    MODEL_PATH = "kc_predictor.pt"
    DAT_PATH   = r"Real_traces/Line1_TiAl_CrossPolCheck_LER3_IQmeas_PDUT-84.0dBm_T140.0mK_Tbb5.0K_0.7323GHzto0.7363GHz_Rate1.0e+06_Sample1000000_BW1.0Hz_Navg1_RF0dBm_LO0dBm.dat"

    """ trace = Trace()
    trace.load_trace(source='cab', path=DAT_PATH)
    results = trace.do_fit(baseline=(3, 0.7), mode="one-shot", verbose=False)
    kc_oneshot = float(results["one-shot"].final["kappac"])

    net, ckpt = load_trained_model(MODEL_PATH)
    base = results["one-shot"].base
    #trace.remove_complex_baseline()
    X_real = build_nn_input_from_dat_iq(DAT_PATH, input_dim=int(ckpt["input_dim"]))


    kc_nn = predict_kc_nn(net, X_real)

    print(f"-> Kc pred (NN, .dat real)   = {kc_nn:.6e}")
    print(f"-> Kc one-shot (fit)         = {kc_oneshot:.6e}")
    print(f"-> Relative error (NN vs fit)= {abs(kc_nn-kc_oneshot)/kc_oneshot:.3%}") """

    # Experimental (.dat)
    f_exp, I_exp, Q_exp = load_iq_from_dat(DAT_PATH)
    s_exp = I_exp + 1j * Q_exp
    amp_exp = np.abs(s_exp)

    # One-shot  
    trace = Trace()
    trace.load_trace(source="cab", path=DAT_PATH)
    results = trace.do_fit(baseline=(3, 0.7), mode="one-shot", verbose=False)

    fit = results["one-shot"].final
    kc_oneshot = float(fit["kappac"])
    
    a     = float(fit["a"])
    dt    = float(fit["dt"])
    phi0  = float(fit["phi0"])      
    rc    = float(fit["rc"])        
    dphi  = float(fit["fano"])      
    fr    = float(fit["resonance"])

    amp_fit = np.abs(results["one-shot"].tprx)

    # Neural Network (kc) 
    net, ckpt = load_trained_model(MODEL_PATH)
    X_real = build_nn_input_from_dat_iq(DAT_PATH, input_dim=int(ckpt["input_dim"]))
    kc_nn = predict_kc_nn(net, X_real)

    # Para reconstruir la curva NN: mantengo kappa_i del one-shot y cambio kappa_c
    kappai = float(fit["kappai"])
    kappa_nn = kappai + kc_nn
    rc_nn = kc_nn / kappa_nn if kappa_nn > 0 else rc

    s_nn = lorentzian_cy(
        f_exp.astype(np.float64),
        a, dt, phi0,
        rc_nn, kappa_nn, dphi, fr
    )
    amp_nn = np.abs(s_nn)

    # Plot
    fig, ax = plt.subplots(
        1, 1,
        dpi=300,
        figsize=(9.7, 4.8),
        constrained_layout=True
    )

    # --- Experimental: scatter ---
    ax.scatter(
        f_exp,
        amp_exp,
        s=6,
        color="black",
        alpha=0.6,
        label="Experimental",
        zorder=3
    )

    # --- One-shot: línea ---
    n = min(len(f_exp), len(amp_fit))
    ax.plot(
        f_exp[:n],
        np.asarray(amp_fit)[:n].real,
        label="One-shot",
        linewidth=2.8,
        linestyle="--",     # ← clave
        color="gold",
        zorder=4
    )

    # --- Neural Network: línea ---
    ax.plot(
        f_exp,
        amp_nn,
        label="Neural Network",
        linewidth=1.8,
        color="tomato",
        zorder=1
    )

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|S21|")
    ax.set_title("Amplitude: Experimental vs One-shot vs Neural Network")

    ax.legend(frameon=False)

    # Texto con kappac
    txt = (
        f"One-shot κc = {kc_oneshot:.3e}\n"
        f"NN κc = {kc_nn:.3e}"
    )

    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85,
            linewidth=0.8
        )
    )

    ax.tick_params(
        direction="in",
        which="both",
        top=True,
        right=True
    )

    plt.show()

    print(f"-> Kc pred (NN, .dat real)   = {kc_nn:.6e}")
    print(f"-> Kc one-shot (fit)         = {kc_oneshot:.6e}")
    print(f"-> Relative error (NN vs fit)= {abs(kc_nn-kc_oneshot)/kc_oneshot:.3%}")
    

if __name__ == "__main__":
    main()

