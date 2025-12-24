import torch
import torch.nn as nn
import torch.optim as optim

# ======================================================
# FROZEN DEEP-SVDD HYPERPARAMETERS
# ======================================================
SVDD_NU = 0.05
SVDD_EPOCHS = 30
SVDD_LR = 1e-3
SVDD_WEIGHT_DECAY = 1e-4

H1 = 64
H2 = 32
NEG_SLOPE = 0.1


# ======================================================
# MODEL
# ======================================================
class DeepSVDD(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, H1, bias=False),
            nn.LeakyReLU(NEG_SLOPE),
            nn.Linear(H1, H2, bias=False),
            nn.LeakyReLU(NEG_SLOPE),
            nn.Linear(H2, input_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


# ======================================================
# TRAINING (SOFT-BOUNDARY SVDD)
# ======================================================
def train_deepsvdd(X_train, device):
    """
    X_train: numpy array (N, D) – real-only features (already scaled)
    Returns: trained model, center, threshold tau
    """
    model = DeepSVDD(X_train.shape[1]).to(device)
    X = torch.tensor(X_train, dtype=torch.float32).to(device)

    # --- center + radius initialization ---
    with torch.no_grad():
        center = model(X).mean(0)
        dist = torch.sum((model(X) - center) ** 2, dim=1)
        R = torch.sqrt(torch.quantile(dist, 1 - SVDD_NU))

    optimizer = optim.Adam(
        model.parameters(),
        lr=SVDD_LR,
        weight_decay=SVDD_WEIGHT_DECAY
    )

    model.train()
    for _ in range(SVDD_EPOCHS):
        optimizer.zero_grad()

        z = model(X)
        dist = torch.sum((z - center) ** 2, dim=1)

        loss = R**2 + (1 / SVDD_NU) * torch.mean(
            torch.clamp(dist - R**2, min=0)
        )

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            dist = torch.sum((model(X) - center) ** 2, dim=1)
            R = torch.sqrt(torch.quantile(dist, 1 - SVDD_NU))

    tau = R.item() ** 2
    model.eval()
    return model, center, tau


# ======================================================
# SCORING
# ======================================================
def svdd_scores(model, center, X, device):
    """
    X: numpy array (N, D)
    Returns squared distance to SVDD center
    """
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32).to(device)
        Z = model(X)
        return torch.sum((Z - center) ** 2, dim=1).cpu().numpy()
