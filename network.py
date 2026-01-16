import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def np_to_th(x):
    """
    Convierte np.ndarray -> torch.Tensor en DEVICE.
    Mantiene shape (N, -1) para compatibilidad; el reshape a (N,2,F) lo hace el forward().
    """
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float32).to(DEVICE).reshape(n_samples, -1)


class Net(nn.Module):
    """
    Arquitectura mejorada (multi-escala):
      3x [Conv1D + BN + ReLU] con downsampling (stride=2 en conv2/conv3)
      + GlobalAveragePooling
      + 3*(Linear + ReLU)
      + Linear(out)

    Input esperado:
      - X con shape (N, 2*F) donde la primera mitad es I y la segunda mitad Q
      - Internamente se reinterpreta como (N, C=2, L=F)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=128,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=1e-3,
        conv_channels=64,
        kernel_size=9,
        dropout=0.10,
    ) -> None:
        super().__init__()

        if input_dim % 2 != 0:
            raise ValueError("input_dim debe ser par (I || Q).")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.F = self.input_dim // 2

        self.epochs = int(epochs)
        self.loss = loss
        self.lr = float(lr)
        self.n_units = int(n_units)

        k = int(kernel_size)
        if k < 1 or k % 2 == 0:
            raise ValueError("kernel_size debe ser impar y >= 1 para padding 'same' simple.")
        padding = k // 2

        # Elegimos pirámide de canales a partir de conv_channels (compatibilidad con tu API actual)
        c3 = int(conv_channels)
        c2 = max(8, c3 // 2)
        c1 = max(8, c3 // 4)

        self.act = nn.ReLU()

        # Bloque conv 1 (sin downsample)
        self.conv1 = nn.Conv1d(2, c1, kernel_size=k, padding=padding, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(c1)

        # Bloque conv 2 (downsample)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=k, padding=padding, stride=2, bias=False)
        self.bn2 = nn.BatchNorm1d(c2)

        # Bloque conv 3 (downsample)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=k, padding=padding, stride=2, bias=False)
        self.bn3 = nn.BatchNorm1d(c3)

        # Para compatibilidad con tu training.py que guarda net.conv.out_channels / kernel_size
        # (esto apuntará a la última conv).
        self.conv = self.conv3

        # Global Average Pooling: (N, C, L) -> (N, C, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.drop = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()

        # 3*(Linear + ReLU)
        self.fc1 = nn.Linear(c3, self.n_units)
        self.fc2 = nn.Linear(self.n_units, self.n_units)
        self.fc3 = nn.Linear(self.n_units, self.n_units)

        # salida
        self.out = nn.Linear(self.n_units, self.output_dim)

    def forward(self, x):
        # x: (N, 2F) -> (N, 2, F)
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input shape incompatible: esperaba (N, {self.input_dim}), tengo (N, {x.shape[1]})."
            )

        I = x[:, : self.F]
        Q = x[:, self.F :]
        x = torch.stack([I, Q], dim=1)  # (N, 2, F)

        # Conv blocks
        h = self.act(self.bn1(self.conv1(x)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))

        # GAP
        h = self.gap(h).squeeze(-1)  # (N, c3)

        # FC head
        h = self.drop(self.act(self.fc1(h)))
        h = self.drop(self.act(self.fc2(h)))
        h = self.drop(self.act(self.fc3(h)))

        return self.out(h)

    def fit(self, X, y, batch_size=64, shuffle=True):
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32))

        ds = TensorDataset(Xt, yt)
        dl = DataLoader(
            ds,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

        optimiser = optim.Adam(self.parameters(), lr=self.lr)
        self.train()

        losses = []
        log_every = max(1, self.epochs // 10)

        for ep in range(self.epochs):
            running = 0.0
            n = 0

            for xb, yb in dl:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                optimiser.zero_grad(set_to_none=True)
                out = self.forward(xb)
                loss = self.loss(yb, out)
                loss.backward()
                optimiser.step()

                running += float(loss.item()) * xb.size(0)
                n += xb.size(0)

            epoch_loss = running / max(1, n)
            losses.append(epoch_loss)

            if ep % log_every == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {epoch_loss:.6f}")

        return losses

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            out = self.forward(np_to_th(X))
        return out.detach().cpu().numpy()


class NetDiscovery(Net):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=128,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=1e-3,
        conv_channels=64,
        kernel_size=9,
        dropout=0.10,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            n_units=n_units,
            epochs=epochs,
            loss=loss,
            lr=lr,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Mantengo tu parámetro extra por compatibilidad con lo que ya tenías
        self.r = nn.Parameter(data=torch.tensor([0.0], device=DEVICE))
