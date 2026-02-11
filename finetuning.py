import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network import Net
from sctlib.analysis import Trace
from libraries.constants import GLOBAL_F_SCALE



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ckpt(model_path: str):
    ckpt = torch.load(model_path, map_location="cpu")
    net = Net(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt.get("output_dim", 1)),
        n_units=int(ckpt.get("n_units", 256)),
        epochs=1,
        lr=float(ckpt.get("lr", 1e-3)),
        conv_channels=int(ckpt.get("conv_channels", 64)),
        kernel_size=int(ckpt.get("kernel_size", 9)),
        dropout=float(ckpt.get("dropout", 0.10)),
        loss=None,  
    )
    net.load_state_dict(ckpt["model_state_dict"])
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
    F = F.astype(np.float64, copy=False)
    M = M.astype(np.float32, copy=False)

    denom = np.sum(M, axis=1, keepdims=True) + eps
    mu = np.sum(F * M, axis=1, keepdims=True) / denom

    Fc = (F - mu) * M
    return (Fc / float(scale_val)).astype(np.float32)


def build_nn_input_from_dat(dat_path: str, ckpt: dict) -> np.ndarray:
    """
    Input NN: [IQ_norm | MASK | F_norm | df_log_norm]
    EXACTAMENTE igual que en training.py
    """
    input_dim = int(ckpt["input_dim"])

    if (input_dim - 1) % 4 != 0:
        raise ValueError("input_dim debe ser 4*max_F + 1 (df escalar al final).")

    max_F = (input_dim - 1) // 4

    # --- cargar datos reales ---
    f, I, Q = load_iq_from_dat(dat_path)
    Fi = len(f)

    if Fi > max_F:
        f = f[:max_F]
        I = I[:max_F]
        Q = Q[:max_F]
        Fi = max_F

    # --- df_eff como en training: span en Hz ---
    span_hz = float(f[-1] - f[0])
    if not np.isfinite(span_hz) or span_hz <= 0:
        raise ValueError(f"Span inválido (<=0) en {dat_path}: {span_hz}")

    df_log = np.log(span_hz).astype(np.float32)

    if "df_log_mean" not in ckpt or "df_log_std" not in ckpt:
        raise KeyError("El checkpoint no contiene df_log_mean/df_log_std. Re-entrena o guarda esos campos.")

    df_log_mean = float(ckpt["df_log_mean"])
    df_log_std  = float(ckpt["df_log_std"])
    df_log_norm = (df_log - df_log_mean) / (df_log_std + 1e-8)
    df_log_norm = np.array([[df_log_norm]], dtype=np.float32)  # (1,1)

    # --- Padding ---
    I_pad = np.zeros(max_F, dtype=np.float32)
    Q_pad = np.zeros(max_F, dtype=np.float32)
    F_pad = np.zeros(max_F, dtype=np.float64)
    M     = np.zeros(max_F, dtype=np.float32)

    I_pad[:Fi] = I
    Q_pad[:Fi] = Q
    F_pad[:Fi] = f
    M[:Fi]     = 1.0

    # --- Normalización IQ (igual que training) ---
    X_iq = np.concatenate([I_pad, Q_pad])[None, :]  # (1, 2*max_F)
    M_   = M[None, :]                               # (1, max_F)

    X_iq = masked_mean_std_iq(X_iq, M_, max_F)
    X_iq[:, :max_F] *= M_
    X_iq[:, max_F:] *= M_

    # --- F como canal (igual que training) ---
    F_norm = build_f_norm_fixed(F_pad[None, :], M_)  # (1, max_F)

    # --- X final: [IQ_norm | mask | F_norm | df_log_norm] ---
    X = np.concatenate([X_iq, M_, F_norm, df_log_norm], axis=1).astype(np.float32)

    if X.shape[1] != input_dim:
        raise RuntimeError(f"Input dim mismatch: construido {X.shape[1]} vs ckpt {input_dim}")

    return X


def make_experimental_dataset(dataset_dir: Path, ckpt: dict, *, baseline=(3,0.7), limit_files=None):

    dat_files = list(dataset_dir.glob("*.dat"))

    rng = np.random.default_rng(seed=42)  
    rng.shuffle(dat_files)

    if limit_files is not None:
        dat_files = dat_files[: int(limit_files)]

    X_list = []
    y_list = []
    ok = 0
    fail = 0

    for p in dat_files:
        try:
            X = build_nn_input_from_dat(str(p), ckpt)   # (1, input_dim)
            X = X.reshape(-1)

            # One-shot 
            trace = Trace()
            trace.load_trace(source="CAB", path=str(p))
            results = trace.do_fit(mode="one-shot", baseline=baseline, verbose=False)
            fit = results["one-shot"].final
            kc = float(fit["kappac"])

            if (not np.isfinite(kc)) or (kc <= 0.0) or (kc < 1e4) or (kc > 1e5):
                raise ValueError(f"kc one-shot inválido: {kc}")
            
            y = np.log(kc).astype(np.float32)

            X_list.append(X)
            y_list.append(y)
            ok += 1

        except Exception as e:
            fail += 1
            print(f"[FAIL] {p.name}: {e}")
            continue

    if ok == 0:
        raise RuntimeError(f"No se pudo construir dataset. ok=0, fail={fail}")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32).reshape(-1, 1)

    print(f"[dataset] ok={ok}, fail={fail}, X={X.shape}, y={y.shape}")
    return X, y


def freeze_backbone(net: Net, freeze: bool):
    """
    Congela stem+resblocks; deja FC entrenable.
    """
    backbone = ["stem", "block1", "block2", "block3"]
    for name, module in net.named_children():
        if name in backbone:
            for p in module.parameters():
                p.requires_grad = (not freeze)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", type=str, default="kc_predictor.pt")
    ap.add_argument("--dataset_dir", type=str, default="Experimental_Validation_Dataset")
    ap.add_argument("--out_ckpt", type=str, default="kc_predictor_finetuned.pt")

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--loss", type=str, default="huber", choices=["mse", "huber"])
    ap.add_argument("--huber_delta", type=float, default=0.5)

    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--limit_files", type=int, default=400)
    ap.add_argument("--val_split", type=float, default=0.15)

    # baseline one-shot
    ap.add_argument("--baseline_order", type=int, default=3)
    ap.add_argument("--baseline_scale", type=float, default=0.7)

    args = ap.parse_args()

    net, ckpt = load_ckpt(args.base_ckpt)
    max_F = (ckpt["input_dim"] - 1) // 4


    # Config fine-tune
    net.epochs = int(args.epochs)
    net.lr = float(args.lr)
    net.weight_decay = float(args.weight_decay)

    if args.loss == "mse":
        net.loss = nn.MSELoss()
    else:
        net.loss = nn.HuberLoss(delta=float(args.huber_delta))

    freeze_backbone(net, freeze=bool(args.freeze_backbone))

    # Dataset experimental (pseudo-label = one-shot)
    X, y = make_experimental_dataset(Path(args.dataset_dir), ckpt, baseline=(int(args.baseline_order), float(args.baseline_scale)), limit_files=int(args.limit_files) if args.limit_files is not None else None)


    # Split train/val
    N = len(X)
    idx = np.random.permutation(N)
    n_val = max(1, int(N * float(args.val_split)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[val_idx], y[val_idx]

    print(f"[split] train={len(X_tr)}, val={len(X_va)}")

    # Entrenamiento 
    losses = net.fit(X_tr, y_tr, batch_size=int(args.batch_size), shuffle=True)

    yhat = net.predict(X_va, batch_size=256).reshape(-1)
    ytrue = y_va.reshape(-1)
    mae = float(np.mean(np.abs(yhat - ytrue)))
    print(f"[val] MAE(|Δlog kc|) = {mae:.6f}")

    # Guardar modelo fine-tuned
    out = {
        "model_state_dict": net.state_dict(),
        "input_dim": int(ckpt["input_dim"]),
        "output_dim": int(ckpt.get("output_dim", 1)),
        "n_units": int(ckpt.get("n_units", 256)),
        "kc_limits": ckpt.get("kc_limits", None),
        "conv_channels": int(ckpt.get("conv_channels", 64)),
        "kernel_size": int(ckpt.get("kernel_size", 9)),
        "dropout": float(ckpt.get("dropout", 0.10)),
        "df_log_mean": float(ckpt["df_log_mean"]),
        "df_log_std":  float(ckpt["df_log_std"]),
        "n_channels":  int(ckpt.get("n_channels", 4)),
        "max_F":       int(ckpt.get("max_F", (ckpt["input_dim"]-1)//4)),
        "uses_f_norm": True,
        "finetune": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "loss": args.loss,
            "huber_delta": float(args.huber_delta),
            "freeze_backbone": bool(args.freeze_backbone),
            "dataset_dir": str(args.dataset_dir),
            "limit_files": int(args.limit_files),
            "baseline": (int(args.baseline_order), float(args.baseline_scale)),
            "val_mae_logkc": mae,
        },
    }
    torch.save(out, args.out_ckpt)
    print(f"[save] {args.out_ckpt}")

    # Plot de losses
    plt.figure()
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Fine-tuning loss")
    plt.tight_layout()
    plt.savefig("finetune_loss.png", dpi=200)
    print("[save] finetune_loss.png")


if __name__ == "__main__":
    main()
