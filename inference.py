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
from libraries.constants import GLOBAL_F_SCALE



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


def masked_mean_std_iq(X_iq, M, max_F, eps=1e-8):
    I = X_iq[:, :max_F]
    Q = X_iq[:, max_F:2*max_F]

    w = M.astype(np.float32)
    denom = np.sum(w, axis=1, keepdims=True) + eps

    muI = np.sum(I * w, axis=1, keepdims=True) / denom
    muQ = np.sum(Q * w, axis=1, keepdims=True) / denom

    varI = np.sum(((I - muI) ** 2) * w, axis=1, keepdims=True) / denom
    varQ = np.sum(((Q - muQ) ** 2) * w, axis=1, keepdims=True) / denom

    stdI = np.sqrt(varI + eps)
    stdQ = np.sqrt(varQ + eps)

    I_n = (I - muI) / stdI
    Q_n = (Q - muQ) / stdQ

    return np.concatenate([I_n, Q_n], axis=1).astype(np.float32) 

def build_f_norm_fixed(F, M, eps=1e-12, scale_val=GLOBAL_F_SCALE):
    """
    F: (N, max_F) padded
    M: (N, max_F) mask 0/1
    - Centra cada traza respecto a su media (mu) en zona válida (mask)
    - Escala SIEMPRE con una constante global (scale_val), preservando unidades
    """
    F = F.astype(np.float64, copy=False)
    M = M.astype(np.float32, copy=False)

    denom = np.sum(M, axis=1, keepdims=True) + eps
    mu = np.sum(F * M, axis=1, keepdims=True) / denom

    Fc = (F - mu) * M  # padding -> 0
    F_norm = (Fc / float(scale_val)).astype(np.float32)
    return F_norm

def diagnostic_span_correlation(ok_paths, pct_errors_base):
    """
    Calcula el span de frecuencia de cada archivo y lo correlaciona con el error.
    """
    spans = []
    
    print("\n--- Iniciando diagnóstico de Correlación Error vs Span ---")
    
    for dat_path in ok_paths:
        try:
            # Usamos skip_header=1 para saltar la línea 'Freq(Hz) I Q'
            # o mejor aún, reaprovechamos tu función que ya maneja esto:
            f, _, _ = load_iq_from_dat(str(dat_path))
            f_span = f.max() - f.min()
            spans.append(f_span)
        except Exception as e:
            print(f"Error procesando {dat_path}: {e}")
            continue
    
    spans = np.array(spans)
    errors = np.array(pct_errors_base)

    if len(spans) != len(errors):
        # Ajustamos por si algún archivo falló al cargar
        errors = errors[:len(spans)]

    # --- Generación del Gráfico ---
    plt.figure(figsize=(10, 6), dpi=150)
    plt.scatter(spans / 1e6, errors, color='blue', alpha=0.6, edgecolors='k', label="Datos Reales")
    
    if len(spans) > 1:
        # Ajuste polinómico para ver la tendencia
        z = np.polyfit(spans / 1e6, errors, 1)
        p = np.poly1d(z)
        plt.plot(spans / 1e6, p(spans / 1e6), "r--", alpha=0.8, label="Tendencia")

    plt.title("Diagnóstico: ¿El error depende del Span (Zoom)?")
    plt.xlabel("Span de Frecuencia [MHz]")
    plt.ylabel("Error Relativo (%)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig("diagnostico_error_vs_span.png")
    print(f"Gráfico guardado en: diagnostico_error_vs_span.png")
    
    correlation = np.corrcoef(spans, errors)[0, 1]
    print(f"Coeficiente de correlación de Pearson: {correlation:.4f}")
    
    if correlation < -0.4:
        print("\n[CONFIRMADO]: Existe una correlación negativa clara.")
        print("A menor span (más zoom), mayor es el error. La red no entiende la escala.")
    else:
        print("\n[ANÁLISIS]: La correlación no es determinante.")



def build_nn_input_from_dat(dat_path: str, input_dim: int):
    """
    Input NN: [IQ_norm | MASK | F_norm]
    EXACTAMENTE igual que en training
    """
    if input_dim % 4 != 0:
        raise ValueError("input_dim debe ser 4*max_F")

    max_F = input_dim // 4

    f, I, Q = load_iq_from_dat(dat_path)
    Fi = len(f)

    if Fi > max_F:
        f = f[:max_F]
        I = I[:max_F]
        Q = Q[:max_F]
        Fi = max_F

    # Padding
    I_pad = np.zeros(max_F, dtype=np.float32)
    Q_pad = np.zeros(max_F, dtype=np.float32)
    F_pad = np.zeros(max_F, dtype=np.float64)
    M     = np.zeros(max_F, dtype=np.float32)

    I_pad[:Fi] = I
    Q_pad[:Fi] = Q
    F_pad[:Fi] = f
    M[:Fi]     = 1.0

    # ---- Normalización IQ (MISMA que training) ----
    X_iq = np.concatenate([I_pad, Q_pad])[None, :]
    M_   = M[None, :]

    X_iq = masked_mean_std_iq(X_iq, M_, max_F)
    X_iq[:, :max_F] *= M_
    X_iq[:, max_F:] *= M_

    # ---- F como canal ----
    F_norm = build_f_norm_fixed(F_pad[None, :], M_)

    # ---- X final ----
    X = np.concatenate([
        X_iq,
        M_,
        F_norm
    ], axis=1).astype(np.float32)

    return X

def predict_kc_nn(net: Net, X: np.ndarray) -> float:
    with torch.no_grad():
        y_pred = net.predict(X)
    return float(np.exp(np.asarray(y_pred).reshape(-1)[0]))


def get_real_span_from_dat(dat_path):
    """
    Calcula el span real de frecuencia (Hz) de una traza experimental (.dat),
    ignorando cabeceras y usando SOLO los puntos reales.
    Compatible con archivos con headers tipo:
      - líneas que empiezan por '#'
      - filas no numéricas
    """
    try:
        # genfromtxt ignora automáticamente líneas no numéricas
        data = np.genfromtxt(dat_path, comments="#")

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 1:
            raise ValueError("No hay columna de frecuencia")

        f = data[:, 0].astype(np.float64)
        f = f[np.isfinite(f)]

        if f.size < 2:
            raise ValueError("Muy pocos puntos válidos")

        return float(f.max() - f.min())

    except Exception as e:
        raise RuntimeError(f"Error calculando span en {dat_path}: {e}")

def main():
    MODEL_BASE = "kc_predictor.pt"      # MODELO NORMAL (BASE)
    MODEL_FT   = "kc_predictor_finetuned.pt"    #MODELO CON FINE-TUNING

    DATASET_DIR = Path("Experimental_Validation_Dataset")

    dat_files = sorted(DATASET_DIR.glob("*.dat"))

    if len(dat_files) == 0:
        print("No se encontraron ficheros .dat en Experimental_Validation_Dataset")
        return

    net_base, ckpt_base = load_trained_model(MODEL_BASE)
    #net_ft,   ckpt_ft   = load_trained_model(MODEL_FT)

    input_dim_base = int(ckpt_base["input_dim"])
    #input_dim_ft   = int(ckpt_ft["input_dim"])
    #assert input_dim_base == input_dim_ft, "Los input_dim no coinciden"
    input_dim = input_dim_base

    rel_errors_base = []
    rel_errors_ft = []
    pct_errors_base = []
    pct_errors_ft   = []
    ok_paths = []
    kc_oneshot_list = []
    spans = []

    for dat_path in dat_files[140:195]:

        try:
            # --- One-shot ---
            trace = Trace()
            trace.load_trace(source="CAB", path=str(dat_path))
            results = trace.do_fit(mode="one-shot", baseline=(3, 0.7), verbose=False)
            plt.close("all")
            fit = results["one-shot"].final
            kc_oneshot = float(fit["kappac"])

            # --- FILTRO: solo considerar one-shot en rango ---
            if (not np.isfinite(kc_oneshot)) or (kc_oneshot < 1e4) or (kc_oneshot > 1e5):
                continue

            kc_oneshot_list.append(kc_oneshot)

            # --- NN ---
            X_real = build_nn_input_from_dat(str(dat_path), input_dim)
            kc_base = predict_kc_nn(net_base, X_real)

            # --- Errores ---
            err_base = abs(np.log(kc_base) - np.log(kc_oneshot))
            pct_base = abs(kc_base - kc_oneshot) / kc_oneshot * 100.0

            rel_errors_base.append(err_base)
            ok_paths.append(dat_path)
            pct_errors_base.append(pct_base)
            #pct_errors_ft.append(pct_ft)

        except Exception:
            print(Exception)

        try:
            span = get_real_span_from_dat(dat_path)
            spans.append(span)
            print(f"{dat_path.name}: {span/1e6:.3f} MHz")
        except Exception as e:
            print(e)

    if len(rel_errors_base) == 0:
        print("No se pudo calcular el error (todos los ficheros fallaron).")
        return

    rel_errors_base = np.asarray(rel_errors_base)
    rel_errors_ft   = np.asarray(rel_errors_ft)
    pct_errors_base = np.asarray(pct_errors_base, dtype=np.float64)
    pct_errors_ft   = np.asarray(pct_errors_ft, dtype=np.float64)

    print(f"Archivos evaluados: {rel_errors_base.size}")
    print(f"BASE mean |Δlog kc|: {rel_errors_base.mean():.6f}  std: {rel_errors_base.std(ddof=1):.6f}")
    #print(f"FT   mean |Δlog kc|: {rel_errors_ft.mean():.6f}  std: {rel_errors_ft.std(ddof=1):.6f}")

    print("\n--- % distancia al one-shot (|kc_pred - kc_one| / kc_one) ---")
    print(f"BASE mean %err: {pct_errors_base.mean():.2f}%  std: {pct_errors_base.std(ddof=1):.2f}%")
    #print(f"FT   mean %err: {pct_errors_ft.mean():.2f}%  std: {pct_errors_ft.std(ddof=1):.2f}%")
    print(f"BASE pct_errors: {pct_errors_base}")
    print(f"BASE mean |Δlog kc|: {rel_errors_base}")

    kc_oneshot_arr = np.asarray(kc_oneshot_list)
    print("\n--- Kc one-shot stats (real data) ---")
    print(f"min   kc_one: {kc_oneshot_arr.min():.3e}")
    print(f"median kc_one: {np.median(kc_oneshot_arr):.3e}")
    print(f"max   kc_one: {kc_oneshot_arr.max():.3e}")

    outside = np.mean(
        (kc_oneshot_arr < 1e4) | (kc_oneshot_arr > 1e5)
    ) * 100.0
    print(f"% fuera de [1e4, 1e5]: {outside:.1f}%")

    #print("\n--- Mejora entre modelo base y fine-tuned ---")
    #improv = (rel_errors_base.mean() - rel_errors_ft.mean()) / (rel_errors_base.mean() + 1e-12)
    #print(f"Mejora relativa (menor es mejor): {improv:.2%}")
    

    top_5_idx = np.argsort(pct_errors_base)[-5:][::-1]
    
    print("\n--- Generando plots de los 5 casos de mayor discrepancia ---")
    
    for rank, idx in enumerate(top_5_idx):
        dat_path = ok_paths[idx]
        
        f_exp, I_exp, Q_exp = load_iq_from_dat(str(dat_path))
        amp_exp = np.sqrt(I_exp**2 + Q_exp**2)
        
        trace = Trace()
        trace.load_trace(source="CAB", path=str(dat_path))
        results = trace.do_fit(mode="one-shot", baseline=(3, 0.7), verbose=False)
        fit = results["one-shot"].final
        
        kc_oneshot = float(fit["kappac"])
        X_input = build_nn_input_from_dat(str(dat_path), input_dim)
        kc_nn = predict_kc_nn(net_base, X_input)

        # Reconstrucción de la curva NN para validación visual
        # Usamos los parámetros del one-shot (fr, phi, etc.) pero sustituimos Kc
        kappai = float(fit["kappai"])
        kappa_nn = kappai + kc_nn
        rc_nn = kc_nn / kappa_nn if kappa_nn > 0 else float(fit["rc"])

        s_nn_curve = lorentzian_cy(
            f_exp.astype(np.float64),
            float(fit["a"]), float(fit["dt"]), float(fit["phi0"]),
            rc_nn, kappa_nn, float(fit["fano"]), float(fit["resonance"])
        )
        amp_nn = np.abs(s_nn_curve)
        amp_oneshot = np.abs(results["one-shot"].tprx)

        # Plotting

        n = min(len(f_exp), len(amp_oneshot))
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        ax.scatter(f_exp, amp_exp, s=10, color="black", alpha=0.3, label="Data (Exp)")
        ax.plot(f_exp[:n], amp_oneshot[:n], 'g--', label="One-shot", lw=2)
        ax.plot(f_exp, amp_nn, 'r-', label=f"NN (Kc={kc_nn:.2e})", lw=2)
        
        ax.set_title(f"Discrepancia Rank {rank+1} - Archivo: {dat_path.name}")
        ax.set_xlabel("Frecuencia [Hz]")
        ax.set_ylabel("Amplitud")
        ax.legend()
        
        # Guardar imagen 
        plt.savefig(f"discrepancia_top_{rank+1}.png")
        plt.close()
        print(f"Guardado: discrepancia_top_{rank+1}.png (Error: {pct_errors_base[idx]:.2f}%)")

    spans = np.array(spans)
    print("\n--- Estadísticas span real ---")
    print(f"min    : {spans.min()/1e6:.3f} MHz")
    print(f"median : {np.median(spans)/1e6:.3f} MHz")
    print(f"max    : {spans.max()/1e6:.3f} MHz")
    diagnostic_span_correlation(ok_paths, pct_errors_base)

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
