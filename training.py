import numpy as np
import matplotlib.pyplot as plt
from lorentzian_generator import lorentzian_generator
from network import Net
import torch
from trace import Trace

def main():    

    cavity_params = {
        "ac":     (0.3, 1.8),          
        "dt":     (0, 1e-9),        
        "phi":    (-np.pi, np.pi),      
        "dphi":   (-np.pi/4, np.pi/4),      
        "kappai": (1e4, 1e6),         
        "fr": (7.30e11 - 2e9, 7.50e11 + 2e9)
    }

    kc_limits = (1e4, 1e5)

    f, X_meas, X_clean, kc_true = lorentzian_generator(
        n_samples=30000,
        cavity_params=cavity_params,
        kc_limits=kc_limits,
        frequency_points=5000,     
        noise_std_signal=(0.001, 0.05),
    )
    

    y = np.log(kc_true).reshape(-1, 1).astype(np.float32)
    X = X_meas.astype(np.float32)

    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))

    train_idx = idx[:split]
    test_idx  = idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    net = Net(
        input_dim=X_train.shape[1],
        output_dim=1,
        n_units=128,
        epochs=2000,
        lr=1e-3,
    )

    losses = net.fit(X_train, y_train)  

    y_pred_train = net.predict(X_train)
    y_pred_test = net.predict(X_test)

    kc_pred_train = np.exp(y_pred_train).flatten()
    kc_true_train = np.exp(y_train).flatten()

    kc_pred_test = np.exp(y_pred_test).flatten()
    kc_true_test = np.exp(y_test).flatten()

    mae_test = np.mean(np.abs(kc_pred_test - kc_true_test))
    rel_err_test = np.mean(np.abs(kc_pred_test - kc_true_test) / kc_true_test)

    print(f"MAE Kc (test): {mae_test:.3e}")
    print(f"Relative error (test): {rel_err_test:.3%}")

    model_path = "kc_predictor.pt"
    torch.save({
        "model_state_dict": net.state_dict(),
        "input_dim": X_train.shape[1],
        "output_dim": 1,
        "n_units": net.n_units,
        "kc_limits": kc_limits,
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

    # Kc true vs predicted (train and test)
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
    plt.savefig("parity_kc_train_test.png", dpi=200)

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

