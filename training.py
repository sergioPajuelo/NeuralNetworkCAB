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


def main():    

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
        n_samples=15000,
        cavity_params=cavity_params,
        kc_limits=kc_limits,
        frequency_points=[2000, 5000, 6000, 10000, 15000, 20000],     
        noise_std_signal=(0.001, 0.03),
    )

    

    y = np.log(kc_true).reshape(-1, 1).astype(np.float32)

    max_F = mask.shape[1]  
    X_iq = X_meas.astype(np.float32)          
    X_m  = mask.astype(np.float32)

    #Frequency channel normalised
    M = X_m.astype(np.float32)            # (N, max_F)
    Fhz = F.astype(np.float32)            # (N, max_F)

    eps = 1e-8
    den = M.sum(axis=1, keepdims=True) + eps

    f_center = (Fhz * M).sum(axis=1, keepdims=True) / den

    F_masked = np.where(M > 0, Fhz, np.nan)
    f_min = np.nanmin(F_masked, axis=1, keepdims=True)
    f_max = np.nanmax(F_masked, axis=1, keepdims=True)
    span = np.maximum((f_max - f_min).astype(np.float32), 1.0)

    # normalized frequency - [-1, 1] 
    f_norm = (Fhz - f_center) / (0.5 * span)

    f_norm = (f_norm * M).astype(np.float32)

    X = np.concatenate([X_iq, X_m, f_norm], axis=1).astype(np.float32)

    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))

    train_idx = idx[:split]
    test_idx  = idx[split:]

    kappai_train = kappai_true[train_idx]
    kappai_test  = kappai_true[test_idx]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    iq_dim = 2 * max_F
    m_dim  = max_F
    f_dim  = max_F

    X_train_iq = X_train[:, :iq_dim].copy()
    X_train_m  = X_train[:, iq_dim:iq_dim + m_dim].copy()                 # (N, max_F)
    X_train_f  = X_train[:, iq_dim + m_dim:iq_dim + m_dim + f_dim].copy() # (N, max_F)

    X_train_iq[:, :max_F]          *= X_train_m
    X_train_iq[:, max_F:2*max_F]   *= X_train_m

    X_train_iq = masked_mean_std_iq(X_train_iq, X_train_m, max_F)

    X_train_iq[:, :max_F]          *= X_train_m
    X_train_iq[:, max_F:2*max_F]   *= X_train_m

    X_train_f *= X_train_m
    X_train = np.concatenate([X_train_iq, X_train_m, X_train_f], axis=1).astype(np.float32)

    X_test_iq = X_test[:, :iq_dim].copy()
    X_test_m  = X_test[:, iq_dim:iq_dim + m_dim].copy()
    X_test_f  = X_test[:, iq_dim + m_dim:iq_dim + m_dim + f_dim].copy()

    X_test_iq[:, :max_F]        *= X_test_m
    X_test_iq[:, max_F:2*max_F] *= X_test_m

    X_test_iq = masked_mean_std_iq(X_test_iq, X_test_m, max_F)

    X_test_iq[:, :max_F]        *= X_test_m
    X_test_iq[:, max_F:2*max_F] *= X_test_m

    X_test_f *= X_test_m

    X_test = np.concatenate([X_test_iq, X_test_m, X_test_f], axis=1).astype(np.float32)

    net = Net(
        input_dim=X_train.shape[1],
        output_dim=1,
        n_units=256,
        epochs=500,
        lr=1e-3,
        loss=nn.HuberLoss(delta=0.5)
    )

    losses = net.fit(X_train, y_train, batch_size=64)  

    y_pred_train = net.predict(X_train, batch_size=256)
    y_pred_test  = net.predict(X_test,  batch_size=256)

    kc_pred_train = np.exp(y_pred_train).flatten()
    kc_true_train = np.exp(y_train).flatten()

    kc_pred_test = np.exp(y_pred_test).flatten()
    kc_true_test = np.exp(y_test).flatten()

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
        "kc_limits": kc_limits,
        "conv_channels": net.conv_channels,
        "kernel_size": net.kernel_size,
        "dropout": net.dropout,
        "n_channels": 4,
        "max_F": int(max_F),
        "uses_f_norm": True,
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

    print("Saved plots:")
    print(" - training_loss.png")
    print(" - parity_kc_train_test.png")
    print(" - residuals_vs_kc_test.png")



if __name__ == "__main__":
    main()

