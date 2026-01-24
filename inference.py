import traceback
import matplotlib
matplotlib.use("Agg")   # ← NO ventanas (quitar si inference individual)

import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None  

import io
import numpy as np
import torch
from pathlib import Path

from lorentzian import lorentzian as lorentzian_cy
from network import Net
from sctlib.analysis import Trace
from lorentzian_generator import padder_optimum


def load_trained_model(model_path: str) -> tuple[Net, dict]:
    ckpt = torch.load(model_path, map_location="cpu")

    net = Net(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt.get("output_dim", 1)),
        n_units=int(ckpt["n_units"]),
        epochs=1,
        lr=1e-3,
        conv_channels=int(ckpt.get("conv_channels", 64)),
        kernel_size=int(ckpt.get("kernel_size", 9)),
        dropout=float(ckpt.get("dropout", 0.10)),
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



def build_nn_input_from_dat_iqm(trace: Trace, dat_path: str, input_dim: int) -> np.ndarray:
    """
    Input para la NN como [I || Q || M] (igual que fine-tuning).
    """
    if input_dim % 3 != 0:
        raise ValueError("input_dim debe ser múltiplo de 3 (I||Q||M).")

    max_F = input_dim // 3

    # trace ya viene cargado con trace.load_trace(...)
    f_pad, I_pad, Q_pad, M = padder_optimum(trace, max_F=max_F)

    I_pad = np.asarray(I_pad, dtype=np.float32)
    Q_pad = np.asarray(Q_pad, dtype=np.float32)
    M     = np.asarray(M,     dtype=np.float32)

    iq = np.concatenate([I_pad, Q_pad], axis=0)  # (2*max_F,)
    mu = iq.mean()
    std = iq.std() + 1e-8
    iq = (iq - mu) / std

    X = np.concatenate([iq, M], axis=0).astype(np.float32)[None, :]  # (1, 3*max_F)
    return X

def predict_kc_nn(net: Net, X: np.ndarray) -> float:
    with torch.no_grad():
        y_pred = net.predict(X)  
    return float(np.exp(np.asarray(y_pred).reshape(-1)[0]))


def main():
    MODEL_BASE = "kc_predictor.pt"
    MODEL_FT   = "kc_predictor_finetuned.pt"

    DATASET_DIR = Path("Experimental_Validation_Dataset")

    dat_files = sorted(DATASET_DIR.glob("*.dat"))

    if len(dat_files) == 0:
        print("No se encontraron ficheros .dat en Experimental_Validation_Dataset")
        return

    net_base, ckpt_base = load_trained_model(MODEL_BASE)
    net_ft,   ckpt_ft   = load_trained_model(MODEL_FT)

    input_dim_base = int(ckpt_base["input_dim"])
    input_dim_ft   = int(ckpt_ft["input_dim"])
    assert input_dim_base == input_dim_ft, "Los input_dim no coinciden"
    input_dim = input_dim_base

    rel_errors_base = []
    rel_errors_ft = []
    pct_errors_base = []
    pct_errors_ft   = []
    freq_errors=[]

    for dat_path in dat_files[:100]:
        try:
            f_raw, _, _ = load_iq_from_dat(dat_path)
            Fi = len(f_raw)
            # --- One-shot ---
            trace = Trace()
            trace.load_trace(source="CAB", path=str(dat_path))
            f_trace,trace_I,trace_Q,_=padder_optimum(trace, max_F=20000)
            trace.trace= trace_I + 1j*trace_Q
            trace.frequency = f_trace
            results = trace.do_fit(mode="one-shot", baseline=(3, 0.7), verbose=False)
            plt.close("all")
            fit = results["one-shot"].final
            kc_oneshot = float(fit["kappac"])

            # --- NN ---
            X_real = build_nn_input_from_dat_iqm(trace, str(dat_path), input_dim=input_dim)

            kc_base = predict_kc_nn(net_base, X_real)
            kc_ft   = predict_kc_nn(net_ft,   X_real)

            if kc_oneshot > 0:
                err_base = abs(np.log(kc_base) - np.log(kc_oneshot))
                err_ft   = abs(np.log(kc_ft)   - np.log(kc_oneshot))
                pct_base = abs(kc_base - kc_oneshot) / kc_oneshot * 100.0
                pct_ft   = abs(kc_ft   - kc_oneshot) / kc_oneshot * 100.0
                rel_errors_base.append(err_base)
                rel_errors_ft.append(err_ft)
                pct_errors_base.append(pct_base)
                pct_errors_ft.append(pct_ft)

        except Exception:
            print(Exception)

    if len(rel_errors_base) == 0:
        print("No se pudo calcular el error (todos los ficheros fallaron).")
        return

    rel_errors_base = np.asarray(rel_errors_base)
    rel_errors_ft   = np.asarray(rel_errors_ft)
    pct_errors_base = np.asarray(pct_errors_base, dtype=np.float64)
    pct_errors_ft   = np.asarray(pct_errors_ft, dtype=np.float64)

    print(f"Archivos evaluados: {rel_errors_base.size}")
    print(f"BASE mean |Δlog kc|: {rel_errors_base.mean():.6f}  std: {rel_errors_base.std(ddof=1):.6f}")
    print(f"FT   mean |Δlog kc|: {rel_errors_ft.mean():.6f}  std: {rel_errors_ft.std(ddof=1):.6f}")

    print("\n--- % distancia al one-shot (|kc_pred - kc_one| / kc_one) ---")
    print(f"BASE mean %err: {pct_errors_base.mean():.2f}%  std: {pct_errors_base.std(ddof=1):.2f}%")
    print(f"FT   mean %err: {pct_errors_ft.mean():.2f}%  std: {pct_errors_ft.std(ddof=1):.2f}%")

    print("\n--- Mejora entre modelo base y fine-tuned ---")
    improv = (rel_errors_base.mean() - rel_errors_ft.mean()) / (rel_errors_base.mean() + 1e-12)
    print(f"Mejora relativa (menor es mejor): {improv:.2%}")
    

    """ DAT_PATH = dat_files[5]

    # Experimental (.dat)
    f_exp, I_exp, Q_exp = load_iq_from_dat(DAT_PATH)
    s_exp = I_exp + 1j * Q_exp
    amp_exp = np.abs(s_exp)

    # One-shot  
    trace = Trace()
    trace.load_trace(source="CAB", path=str(DAT_PATH))
    f_trace,trace_I,trace_Q,_=padder_optimum(trace, max_F=20000)
    trace.trace= trace_I + 1j*trace_Q
    trace.frequency = f_trace
    results = trace.do_fit(mode="one-shot", baseline=(3, 0.7), verbose=False)

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
    X_real = build_nn_input_from_dat_iq(trace, str(DAT_PATH), input_dim=int(ckpt["input_dim"]))
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

    print(f"Kc one-shot: {kc_oneshot:.3}")
    print(f"Kc (NN): {kc_nn:.3}")
    print(f"Error: {abs(kc_nn - kc_oneshot) / kc_oneshot if kc_oneshot > 0 else 0.0:.3%}") """


if __name__ == "__main__":
    main()

