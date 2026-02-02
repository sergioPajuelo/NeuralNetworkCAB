import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def np_to_th(x):
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float32).to(DEVICE).reshape(n_samples, -1)


class ResidualBlock1D(nn.Module):
    """
    Residual block 1D con:
      - conv -> BN -> ReLU -> (Drop) -> conv -> BN
      - skip projection si cambia canales o si stride != 1
    """
    def __init__(self, c_in, c_out, k, *, stride=1, dilation=1, dropout=0.0):
        super().__init__()
        if k < 1 or k % 2 == 0:
            raise ValueError("kernel_size debe ser impar y >= 1.")

        # "same-ish" padding para mantener centrado; con stride>1 habrá downsample igualmente
        pad = (k // 2) * dilation

        self.conv1 = nn.Conv1d(
            c_in, c_out, kernel_size=k, stride=stride, padding=pad, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(c_out)

        self.conv2 = nn.Conv1d(
            c_out, c_out, kernel_size=k, stride=1, padding=pad, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(c_out)

        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()

        need_proj = (c_in != c_out) or (stride != 1)
        self.proj = (
            nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(c_out),
            )
            if need_proj
            else nn.Identity()
        )

    def forward(self, x):
        skip = self.proj(x)
        h = self.act(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        h = self.drop(h)
        return self.act(h + skip)


class Net(nn.Module):
    """
    Net mejorada:
      - Stride para comprimir F (mejor invariancia y menos sobreajuste a padding)
      - Residuales con proyección y stride
      - Pooling Avg + Max concatenado
      - MSELoss (por defecto)
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=256,
        epochs=400,
        loss=None,                 # si None -> MSE
        lr=2e-3,
        weight_decay=1e-4,         # ayuda bastante a generalizar
        conv_channels=64,
        kernel_size=9,
        dropout=0.10,
    ) -> None:
        super().__init__()

        if input_dim % 3 != 0:
            raise ValueError("input_dim debe ser múltiplo de 3 (I || Q || mask).")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.F = input_dim // 3

        self.epochs = int(epochs)
        self.loss = nn.MSELoss() if loss is None else loss
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.n_units = int(n_units)
        self.conv_channels = int(conv_channels)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)

        k = int(kernel_size)

        # Pirámide de canales (más capacidad según bajas resolución)
        c1 = max(16, int(conv_channels // 2))
        c2 = max(32, int(conv_channels))
        c3 = max(64, int(conv_channels * 2))
        c4 = max(64, int(conv_channels * 2))

        # Stem con stride=2: baja F rápido y aprende patrones locales robustos
        self.stem = nn.Sequential(
            nn.Conv1d(3, c1, kernel_size=k, stride=2, padding=(k // 2), bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
        )

        # Bloques: metemos más stride para multi-escala
        self.block1 = ResidualBlock1D(c1, c2, k, stride=1, dilation=1, dropout=dropout)
        self.block2 = ResidualBlock1D(c2, c3, k, stride=2, dilation=1, dropout=dropout)
        self.block3 = ResidualBlock1D(c3, c4, k, stride=2, dilation=2, dropout=dropout)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)

        self.drop = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

        # concat avg+max => 2*c4
        self.fc1 = nn.Linear(2 * c4 + 1, self.n_units)
        self.fc2 = nn.Linear(self.n_units, self.n_units)
        self.fc3 = nn.Linear(self.n_units, self.n_units)
        self.out = nn.Linear(self.n_units, self.output_dim)

    def forward(self, x, df_scalar):
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)

        if x.shape[1] != self.input_dim:
            raise ValueError(f"Input shape incompatible: esperaba (N, {self.input_dim}), tengo (N, {x.shape[1]}).")

        I = x[:, :self.F]
        Q = x[:, self.F:2*self.F]
        M = x[:, 2*self.F:3*self.F]
        
        x_img = torch.stack([I*M, Q*M, M], dim=1)  # (N, 3, F)

        h = self.stem(x_img)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)

        m = M.unsqueeze(1)  # (N,1,F)
        m = torch.nn.functional.max_pool1d(m, kernel_size=2, stride=2)  # -> ~F/2 (stem)
        m = torch.nn.functional.max_pool1d(m, kernel_size=2, stride=2)  # -> ~F/4 (block2)
        m = torch.nn.functional.max_pool1d(m, kernel_size=2, stride=2)  # -> ~F/8 (block3)

        if m.shape[-1] != h.shape[-1]:
            L = min(m.shape[-1], h.shape[-1])
            m = m[..., :L]
            h = h[..., :L]

        hm = h * m
        denom = m.sum(dim=-1).clamp_min(1.0)         
        h_avg = hm.sum(dim=-1) / denom                


        neg_inf = torch.finfo(h.dtype).min
        h_masked = h.masked_fill(m == 0, neg_inf)     # (N,c4,L)
        h_max = h_masked.max(dim=-1).values           # (N,c4)

        h_combined = torch.cat([h_avg, h_max, df_scalar], dim=1)          # (N, 2*c4)

        h = self.drop(self.act(self.fc1(h_combined)))
        h = self.drop(self.act(self.fc2(h)))
        h = self.drop(self.act(self.fc3(h)))

        return self.out(h)

    def fit(self, X, dfs, y, batch_size=64, shuffle=True): # Añadimos 'dfs'
        self.to(DEVICE)
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        DFt = torch.from_numpy(np.asarray(dfs, dtype=np.float32)) # Nuevo
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32))

        ds = TensorDataset(Xt, DFt, yt) # Ahora son 3 elementos
        dl = DataLoader(ds, batch_size=int(batch_size), shuffle=bool(shuffle))

        optimiser = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(10, self.epochs))

        self.train()
        losses = []
        log_every = max(1, self.epochs // 10)

        for ep in range(self.epochs):
            running = 0.0   
            n = 0
            for xb, dfb, yb in dl: 
                xb, dfb, yb = xb.to(DEVICE), dfb.to(DEVICE), yb.to(DEVICE)
                dfb = dfb.view(-1, 1)
                optimiser.zero_grad()
                out = self.forward(xb, dfb)
                loss = self.loss(out, yb)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimiser.step()

                running += float(loss.item()) * xb.size(0)
                n += xb.size(0)

            scheduler.step()

            epoch_loss = running / max(1, n)
            losses.append(epoch_loss)

            if ep % log_every == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {epoch_loss:.6f}")

        return losses

    def predict(self, X, dfs, batch_size: int = 256): # Añadimos 'dfs'
        self.to(DEVICE)
        self.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[i:i+batch_size]).to(DEVICE).float()
                dfb = torch.from_numpy(dfs[i:i+batch_size]).to(DEVICE).float()
                dfb = dfb.view(-1, 1)
                out = self.forward(xb, dfb)
                preds.append(out.cpu())
        return torch.cat(preds, dim=0).numpy()


