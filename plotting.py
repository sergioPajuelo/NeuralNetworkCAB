# plotting.py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
from numpy.polynomial.polyutils import RankWarning
warnings.simplefilter("ignore", RankWarning)

import torch
from matplotlib.colors import Colormap, Normalize

from lorentzian_generator import lorentzian_generator
from network import Net
from libraries.constants import GLOBAL_F_SCALE, ParameterLimits, N_ONESHOT_COMP
from sctlib.analysis import Trace


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

    X_iq_n = np.concatenate([I_n, Q_n], axis=1).astype(np.float32)
    return X_iq_n


def build_f_norm_fixed(F, M, eps=1e-12, scale_val=GLOBAL_F_SCALE):
    F = F.astype(np.float64, copy=False)
    M = M.astype(np.float32, copy=False)

    denom = np.sum(M, axis=1, keepdims=True) + eps
    mu = np.sum(F * M, axis=1, keepdims=True) / denom

    Fc = (F - mu) * M
    F_norm = (Fc / float(scale_val)).astype(np.float32)
    return F_norm


def kc_one_shot_from_arrays(f_hz: np.ndarray, I: np.ndarray, Q: np.ndarray) -> float:
    tr = Trace(
        frequency=f_hz.astype(np.float64),
        trace=(I.astype(np.float64) + 1j * Q.astype(np.float64)),
    )
    results = tr.do_fit(mode="one-shot", baseline=(3, 0.7), verbose=False)
    fit = results["one-shot"].final
    return float(fit["kappac"])


def _apply_plot6_style_rcparams():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "text.usetex": False
    })


def _plot_param_vs_error_same_style(x_param, y_err, x_label, out_pdf, n_bins=18):
    x_param = np.abs(np.asarray(x_param, dtype=np.float64))
    y_err = np.abs(np.asarray(y_err, dtype=np.float64))

    valid = np.isfinite(x_param) & (x_param > 0) & np.isfinite(y_err)
    x, y = x_param[valid], y_err[valid]

    order = np.argsort(x)
    x, y = x[order], y[order]

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

    _apply_plot6_style_rcparams()
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)

    ax.scatter(
        x, y,
        s=6, alpha=0.15,
        color="tab:gray",
        edgecolors="none",
        rasterized=True,
        label="Test samples"
    )

    ax.fill_between(
        centers[ok], p16[ok], p84[ok],
        color="royalblue", alpha=0.25,
        label="16th–84th percentile"
    )
    ax.plot(
        centers[ok], med[ok],
        color="darkblue", lw=2.0,
        marker="o", markersize=4,
        label="Median error"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"Relative error $\epsilon = |K_{c}^{NN} - K_{c}^{true}| / K_{c}^{true}$")

    ax.grid(True, which="major", ls="-", alpha=0.3)
    ax.grid(True, which="minor", ls=":", alpha=0.15)
    ax.tick_params(which="both", top=True, right=True)

    ax.legend(frameon=True, loc="upper left", fontsize=10).get_frame().set_linewidth(0.5)

    plt.savefig(out_pdf, dpi=400)
    plt.close()


def _plot_nn_vs_os_colored_by_param(err_os_true, err_nn_true, param, param_label, out_pdf):
    err_os_true = np.asarray(err_os_true, dtype=np.float64)
    err_nn_true = np.asarray(err_nn_true, dtype=np.float64)
    param = np.asarray(param, dtype=np.float64)

    eps = 1e-12
    err_os_true = np.maximum(err_os_true, eps)
    err_nn_true = np.maximum(err_nn_true, eps)

    valid = np.isfinite(err_os_true) & np.isfinite(err_nn_true) & np.isfinite(param) & (param != 0)
    x = err_os_true[valid]
    y = err_nn_true[valid]
    p = param[valid]

    cmap = matplotlib.colormaps["viridis"]
    assert isinstance(cmap, Colormap)

    norm = Normalize(
        vmin=np.nanpercentile(p, 2),
        vmax=np.nanpercentile(p, 98),
        clip=True
    )

    plt.figure(figsize=(6.4, 5.4))
    sc = plt.scatter(
        x, y,
        c=p,
        cmap=cmap,
        norm=norm,
        s=12,
        alpha=0.60,
        edgecolors="none",
        rasterized=True
    )

    lims = [float(min(x.min(), y.min())), float(max(x.max(), y.max()))]
    plt.plot(lims, lims, "k--", lw=1)

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r"Relative error $|K_c^{OS} - K_c^{true}| / K_c^{true}$")
    plt.ylabel(r"Relative error $|K_c^{NN} - K_c^{true}| / K_c^{true}$")
    plt.title("NN vs One-shot error (both vs true)")

    cbar = plt.colorbar(sc)
    cbar.set_label(param_label)

    plt.tight_layout()
    plt.savefig(out_pdf, dpi=350)
    plt.close()


def _select_param(param_name: str, param_array):
    """
    Entrada:
      - param_name: uno de {"ac","dt","phi","r","kappa","dphi","fr"}
      - param_array: array (N,) con los valores de ese parámetro (en el mismo orden que tus muestras)

    Salida:
      - values: np.ndarray float64 (N,)
      - label: str (latex-ready)
    """
    name = str(param_name).lower().strip()
    p = np.asarray(param_array, dtype=np.float64).ravel()

    if name == "ac":
        return np.abs(p), r"$a_c$"

    if name == "dt":
        return np.abs(p), r"$|dt|$ (s)"

    if name == "phi":
        return p, r"$\phi$ (rad)"

    if name == "r":
        return np.abs(p), r"$r$"

    if name == "kappa":
        return np.abs(p), r"$\kappa$ (Hz)"

    if name == "dphi":
        return p, r"$\Delta\phi$ (rad)"

    if name == "fr":
        return np.abs(p), r"$f_r$ (Hz)"

    raise ValueError(
        "param_name no válido. Usa únicamente:\n"
        "ac, dt, phi, r, kappa, dphi, fr"
    )



def generate_and_plot(
    *,
    model_path: str = "kc_predictor.pt",
    n_samples: int = 1000,
    noise_std_signal=(0.0001, 0.0004),
    param_name: str = "dt",
    out_prefix: str = "test_",
):
    # 1) Genera dataset sintético
    F, X_meas, X_clean, kc_true, kappai_true, F_len, mask, dfs, dts = lorentzian_generator(
        n_samples=n_samples,
        noise_std_signal=noise_std_signal,
    )

    max_F = mask.shape[1]
    X_iq = X_meas.astype(np.float32)
    X_m = mask.astype(np.float32)
    F_norm = build_f_norm_fixed(F, X_m)

    df_eff = (dfs * (F_len - 1)).reshape(-1, 1).astype(np.float32)
    df_log = np.log(np.maximum(df_eff, 1e-12))
    df_log_mean = float(df_log.mean())
    df_log_std = float(df_log.std() + 1e-8)
    df_log_norm = (df_log - df_log_mean) / df_log_std

    X_iq_n = masked_mean_std_iq(X_iq, X_m, max_F)
    X_iq_n[:, :max_F] *= X_m
    X_iq_n[:, max_F:2*max_F] *= X_m

    X = np.concatenate([X_iq_n, X_m, F_norm, df_log_norm], axis=1).astype(np.float32)

    ckpt = torch.load(model_path, map_location="cpu")
    net = Net(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
        n_units=int(ckpt["n_units"]),
    )
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    with torch.no_grad():
        y_pred = net.predict(X, batch_size=512)  

    kc_nn = np.exp(y_pred).flatten()
    kc_true = np.asarray(kc_true, dtype=np.float64).flatten()

    kc_os = np.full_like(kc_true, np.nan, dtype=np.float64)
    idx_eval = np.random.permutation(len(kc_true))[: int(N_ONESHOT_COMP)]

    for idx in idx_eval:
        try:
            L = int(F_len[idx])
            f_hz = F[idx, :L]
            I = X_meas[idx, :max_F][:L]
            Q = X_meas[idx, max_F:2*max_F][:L]
            kc_os[idx] = kc_one_shot_from_arrays(f_hz, I, Q)
        except Exception:
            continue

    valid = np.isfinite(kc_os) & np.isfinite(kc_nn) & np.isfinite(kc_true) & (kc_true > 0) & (kc_os > 0) & (kc_nn > 0)

    kc_nn_v = kc_nn[valid]
    kc_os_v = kc_os[valid]
    kc_true_v = kc_true[valid]

    param_all, param_label = _select_param(param_name, dts)
    param_v = np.asarray(param_all, dtype=np.float64).flatten()[valid]

    err_nn_true = np.abs(kc_nn_v - kc_true_v) / kc_true_v
    err_os_true = np.abs(kc_os_v - kc_true_v) / kc_true_v

    rel_nn = (kc_nn_v - kc_true_v) / kc_true_v

    out_pdf_A = f"{out_prefix}nn_error_vs_{param_name}.pdf"
    out_pdf_B = f"{out_prefix}nn_vs_os_error_colored_by_{param_name}.pdf"

    _plot_param_vs_error_same_style(
        x_param=param_v,
        y_err=np.abs(rel_nn),
        x_label=param_label,
        out_pdf=out_pdf_A,
        n_bins=18
    )

    _plot_nn_vs_os_colored_by_param(
        err_os_true=err_os_true,
        err_nn_true=err_nn_true,
        param=param_v,
        param_label=param_label,
        out_pdf=out_pdf_B
    )

    print("Saved PDFs:")
    print(" -", out_pdf_A)
    print(" -", out_pdf_B)


if __name__ == "__main__":
    generate_and_plot(param_name="dt")
