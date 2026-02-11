import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None  
from lorentzian_generator import lorentzian_generator
from network import Net
import torch
import random
import warnings
from numpy.polynomial.polyutils import RankWarning
warnings.simplefilter("ignore", RankWarning)
import torch.nn as nn
from libraries.constants import GLOBAL_F_SCALE, ParameterLimits, N_ONESHOT_COMP
from sctlib.analysis import Trace
from matplotlib.colors import Colormap, Normalize



def masked_mean_std_iq(X_iq, M, max_F, eps=1e-8):
    # X_iq: (N, 2*max_F)   [I|Q]
    # M:    (N, max_F)     0/1
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

    X_iq_n = np.concatenate([I_n, Q_n], axis=1).astype(np.float32)
    return X_iq_n 

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

def kc_one_shot_from_arrays(f_hz: np.ndarray, I: np.ndarray, Q: np.ndarray) -> float:
    """
    Devuelve kc_os (kappac) de one-shot. Lanza excepción si falla.
    """
    tr = Trace(frequency=f_hz.astype(np.float64), trace=(I.astype(np.float64) + 1j * Q.astype(np.float64)))
    results = tr.do_fit(mode="one-shot", baseline=(3, 0.7), verbose=False)
    fit = results["one-shot"].final
    return float(fit["kappac"])


def main():    


    F, X_meas, X_clean, kc_true, kappai_true, F_len, mask, dfs, dts = lorentzian_generator(
        n_samples=30000,
        noise_std_signal=(0.0001, 0.0004),
    )

    

    y = np.log(kc_true).reshape(-1, 1).astype(np.float32)

    y = np.clip(
        y,
        np.log(ParameterLimits.COUPLING_LOWER_LIMIT),
        np.log(ParameterLimits.COUPLING_UPPER_LIMIT)
    )

    max_F = mask.shape[1]  
    X_iq = X_meas.astype(np.float32)          
    X_m  = mask.astype(np.float32)
    F_norm = build_f_norm_fixed(F, X_m)



    
    N = X_iq.shape[0] 
    idx = np.random.permutation(N)
    split = int(0.8 * N)

    train_idx = idx[:split]
    test_idx  = idx[split:]

    df_eff = (dfs * (F_len - 1)).reshape(-1, 1).astype(np.float32)
    df_log = np.log(df_eff)

    df_log_mean = df_log[train_idx].mean()
    df_log_std  = df_log[train_idx].std()

    df_log_norm = (df_log - df_log_mean) / (df_log_std + 1e-8)

    X = np.concatenate([X_iq, X_m, F_norm, df_log_norm], axis=1).astype(np.float32)


    kappai_train = kappai_true[train_idx]
    kappai_test  = kappai_true[test_idx]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    iq_dim = 2 * max_F
    m_dim  = max_F

    # ---- TRAIN ----
    X_train_iq = X_train[:, :iq_dim].copy()
    X_train_m  = X_train[:, iq_dim:iq_dim + m_dim].copy()

    # Split I/Q
    I_tr = X_train_iq[:, :max_F]
    Q_tr = X_train_iq[:, max_F:2*max_F]

    # Normalización estable: z-score enmascarado (por muestra)
    X_train_iq = masked_mean_std_iq(X_train_iq, X_train_m, max_F)

    X_train_iq[:, :max_F] *= X_train_m
    X_train_iq[:, max_F:2*max_F] *= X_train_m

    # Recupera F_norm de X_train (estaba ya concatenado en X)
    X_train_f = X_train[:, iq_dim + m_dim : iq_dim + 2*m_dim].copy()
    X_train_df = X_train[:, -1:].copy()

    # X final: [IQ_norm, mask, F_norm, df_eff]
    X_train = np.concatenate(
        [X_train_iq, X_train_m, X_train_f, X_train_df],
        axis=1
    ).astype(np.float32)

    # ---- TEST ----
    X_test_iq = X_test[:, :iq_dim].copy()
    X_test_m  = X_test[:, iq_dim:iq_dim + m_dim].copy()

    X_test_iq = masked_mean_std_iq(X_test_iq, X_test_m, max_F)

    X_test_iq[:, :max_F] *= X_test_m
    X_test_iq[:, max_F:2*max_F] *= X_test_m


    # Recupera F_norm de X_test (estaba ya concatenado en X)
    X_test_f = X_test[:, iq_dim + m_dim : iq_dim + 2*m_dim].copy()
    X_test_df = X_test[:, -1:].copy()


    # X final: [IQ_norm, mask, F_norm, df_eff]
    X_test = np.concatenate(
        [X_test_iq, X_test_m, X_test_f, X_test_df],
        axis=1
    ).astype(np.float32)

    net = Net(
        input_dim=X_train.shape[1],
        output_dim=1,
        n_units=256,
        epochs=500,
        lr=1e-3,
        loss=nn.HuberLoss(delta=0.2)
    )

    losses = net.fit(X_train, y_train, batch_size=64)

    y_pred_train = net.predict(X_train, batch_size=256)
    y_pred_test  = net.predict(X_test,  batch_size=256)

    kc_pred_train = np.exp(y_pred_train).flatten()
    kc_true_train = np.exp(y_train).flatten()

    kc_pred_test = np.exp(y_pred_test).flatten()
    kc_true_test = np.exp(y_test).flatten()

    kc_os_test = np.full_like(kc_true_test, np.nan, dtype=np.float64)

    for i, idx_test in enumerate(test_idx[:N_ONESHOT_COMP]):
        try:
            f_hz = F[idx_test, :F_len[idx_test]]
            I = X_meas[idx_test, :max_F][:F_len[idx_test]]
            Q = X_meas[idx_test, max_F:2*max_F][:F_len[idx_test]]

            kc_os_test[i] = kc_one_shot_from_arrays(f_hz, I, Q)
        except Exception:
            continue

    valid = np.isfinite(kc_os_test)

    kc_nn_v   = kc_pred_test[valid]
    kc_os_v   = kc_os_test[valid]
    kc_true_v = kc_true_test[valid]

    err_nn_true = np.abs(kc_nn_v - kc_true_v) / kc_true_v
    err_os_true = np.abs(kc_os_v - kc_true_v) / kc_true_v

    ratio_true_train = kc_true_train / kappai_train
    ratio_true_test  = kc_true_test  / kappai_test

    log_err = np.abs(y_pred_test - y_test)
    print("Mean |Δlog(Kc)|:", log_err.mean())

    model_path = "kc_predictor.pt"
    torch.save({
        "model_state_dict": net.state_dict(),
        "input_dim": X_train.shape[1],
        "output_dim": 1,
        "n_units": net.n_units,
        "conv_channels": net.conv_channels,
        "kernel_size": net.kernel_size,
        "dropout": net.dropout,
        "n_channels": 4,
        "max_F": int(max_F),
        "uses_f_norm": True,
        "df_log_mean": float(df_log_mean),
        "df_log_std": float(df_log_std),
    }, model_path)

    print(f"Model saved to {model_path}")

    # Training loss curve
    plt.figure()
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=200)

    """ # Kc true vs predicted (train and test)
    plt.figure()
    plt.scatter(kc_true_train, kc_pred_train, s=10, alpha=0.5, label="Train")
    plt.scatter(kc_true_test, kc_pred_test, s=10, alpha=0.5, label="Test")

    lo = min(kc_true.min(), kc_pred_train.min(), kc_pred_test.min())
    hi = max(kc_true.max(), kc_pred_train.max(), kc_pred_test.max())
    plt.plot([lo, hi], [lo, hi])

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Kc true")
    plt.ylabel("Kc predicted")
    plt.title("Parity plot (Kc)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("parity_kc_train_test.png", dpi=200) """


    plt.figure()
    plt.scatter(ratio_true_train, kc_pred_train, s=10, alpha=0.5, label="Train")
    plt.scatter(ratio_true_test,  kc_pred_test,  s=10, alpha=0.5, label="Test")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Kc / Ki (true)")
    plt.ylabel("Kc predicted")
    plt.title("Kc predicted vs Kc/Ki")

    plt.legend()
    plt.tight_layout()
    plt.savefig("kc_pred_vs_ratio_kc_ki.png", dpi=200)

    rel_test = (kc_pred_test - kc_true_test) / kc_true_test
    # 4) Residuals vs Kc_true (sesgo)
    plt.figure()
    plt.scatter(kc_true_test, rel_test, s=10, alpha=0.6)
    plt.axhline(0.0)
    plt.xscale("log")
    plt.xlabel("Kc true (test)")
    plt.ylabel("Relative error")
    plt.title("Residuals vs Kc (test)")
    plt.tight_layout()
    plt.savefig("residuals_vs_kc_test.png", dpi=200)

    # 5) NN vs One-shot error (coloreado por dt)

    dt_test = dts[test_idx].astype(np.float64)
    dt_valid = dt_test[valid]
    dt_valid = np.abs(dt_valid)

    eps = 1e-12
    err_os_plot = np.maximum(err_os_true, eps)
    err_nn_plot = np.maximum(err_nn_true, eps)

    valid2 = (
        np.isfinite(err_os_plot) &
        np.isfinite(err_nn_plot) &
        np.isfinite(dt_valid) &
        (dt_valid > 0)
    )

    x = err_os_plot[valid2]
    y = err_nn_plot[valid2]
    p = dt_valid[valid2]

    cmap = matplotlib.colormaps["viridis"]
    assert isinstance(cmap, Colormap)

    norm = Normalize(
        vmin=np.nanpercentile(p, 2),
        vmax=np.nanpercentile(p, 98),
        clip=True
    )

    plt.figure(figsize=(6.4, 5.4))
    sc = plt.scatter(
        x,
        y,
        c=p,
        cmap=cmap,
        norm=norm,
        s=12,
        alpha=0.60,
        edgecolors="none",
    )

    lims = [float(min(x.min(), y.min())),
            float(max(x.max(), y.max()))]

    plt.plot(lims, lims, "k--", lw=1)

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r"Relative error $|K_c^{OS} - K_c^{true}| / K_c^{true}$")
    plt.ylabel(r"Relative error $|K_c^{NN} - K_c^{true}| / K_c^{true}$")
    plt.title("NN vs One-shot error (colored by $|dt|$)")

    cbar = plt.colorbar(sc)
    cbar.set_label(r"$|dt|$ (s)")

    plt.tight_layout()
    plt.savefig("nn_vs_oneshot_error_vs_true_colored_dt.pdf", dpi=350)
    plt.close()

    # 6) Error (test) vs dt

    dt_test = dts[test_idx].astype(np.float64)
    x_dt = np.abs(dt_test)
    y_err = np.abs(rel_test.astype(np.float64))

    valid = np.isfinite(x_dt) & (x_dt > 0) & np.isfinite(y_err)
    x, y = x_dt[valid], y_err[valid]

    order = np.argsort(x)
    x, y = x[order], y[order]

    n_bins = 18
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    centers = np.sqrt(bins[:-1] * bins[1:])
    med, p16, p84 = [np.full(n_bins, np.nan) for _ in range(3)]

    for k in range(n_bins):
        m = (x >= bins[k]) & (x < bins[k + 1])
        if np.any(m):
            med[k] = np.median(y[m])
            p16[k] = np.percentile(y[m], 16)
            p84[k] = np.percentile(y[m], 84)

    ok = np.isfinite(med)

    # Estética
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "text.usetex": False  
    })

    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)

    ax.scatter(x, y, s=6, alpha=0.15, color='tab:gray', edgecolors='none', 
            rasterized=True, label='Test samples')

    ax.fill_between(centers[ok], p16[ok], p84[ok], color='royalblue', 
                    alpha=0.25, label='16th–84th percentile')
    ax.plot(centers[ok], med[ok], color='darkblue', lw=2.0, 
            marker='o', markersize=4, label='Median error')

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"Time step $|dt|$ (s)")
    ax.set_ylabel(r"Relative error $\epsilon = |K_{c}^{NN} - K_{c}^{true}| / K_{c}^{true}$")

    ax.grid(True, which="major", ls="-", alpha=0.3)
    ax.grid(True, which="minor", ls=":", alpha=0.15)
    ax.tick_params(which="both", top=True, right=True)

    ax.legend(frameon=True, loc='upper left', fontsize=10).get_frame().set_linewidth(0.5)

    plt.savefig("nn_error_vs_dt_test.pdf", dpi=400)
    plt.show()

    print("Saved plots:")
    print(" - training_loss.png")
    print(" - parity_kc_train_test.png")
    print(" - residuals_vs_kc_test.png")
    print(" - nn_vs_oneshot_error_diagnostic.png")
    print(" - nn_error_vs_dt_test.pdf")





if __name__ == "__main__":
    main()

