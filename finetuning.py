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
from lorentzian_generator import padder_optimum


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


def build_X_from_trace(trace: Trace, max_F: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve:
      X: (3*max_F,) = [I_pad||Q_pad||M]
      M: (max_F,)
    Nota: normalización igual que training.py:
      z-score por muestra SOLO en (I||Q), y la mask se concatena sin normalizar.
    """
    f_pad, I_pad, Q_pad, M = padder_optimum(trace, max_F=max_F)

    I_pad = np.asarray(I_pad, dtype=np.float32)
    Q_pad = np.asarray(Q_pad, dtype=np.float32)
    M = np.asarray(M, dtype=np.float32)

    iq = np.concatenate([I_pad, Q_pad], axis=0)  # (2*max_F,)
    mu = iq.mean()
    std = iq.std() + 1e-8
    iq = (iq - mu) / std

    X = np.concatenate([iq, M], axis=0).astype(np.float32)  # (3*max_F,)
    return X, M


def make_experimental_dataset(
    dataset_dir: Path,
    max_F: int,
    *,
    baseline=(3, 0.7),
    limit_files: int | None = None,
):
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
            trace = Trace()
            trace.load_trace(source="CAB", path=str(p))

            # Padding + mask (para que el input sea consistente con la Net)
            X, _M = build_X_from_trace(trace, max_F=max_F)

            # One-shot 
            f_pad, I_pad, Q_pad, _M2 = padder_optimum(trace, max_F=max_F)
            trace.trace = np.asarray(I_pad) + 1j * np.asarray(Q_pad)
            trace.frequency = np.asarray(f_pad)

            results = trace.do_fit(mode="one-shot", baseline=baseline, verbose=False)
            fit = results["one-shot"].final
            kc = float(fit["kappac"])


            y = np.log(kc).astype(np.float32)

            X_list.append(X)
            y_list.append(y)
            ok += 1

        except Exception:
            fail += 1
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
    max_F = int(ckpt["input_dim"]) // 3

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
    X, y = make_experimental_dataset(
        Path(args.dataset_dir),
        max_F=max_F,
        baseline=(int(args.baseline_order), float(args.baseline_scale)),
        limit_files=int(args.limit_files) if args.limit_files is not None else None,
    )

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
