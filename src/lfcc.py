import os
import random
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from spafe.features.lfcc import lfcc

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, DetCurveDisplay
import umap
import matplotlib.pyplot as plt
import plotly.express as px

# ======================================================
# REPRODUCIBILITY
# ======================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ======================================================
# PATHS (PROTOCOL IN spectral/, CODE IN spectral/src/)
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # spectral/src
PROJECT_DIR = os.path.dirname(BASE_DIR)                   # spectral

PROTOCOL_CSV  = os.path.join(PROJECT_DIR, "oc_protocol_eval1000.csv")
PROTOCOL_ROOT = "/data/FF_V2/FF_V2"
REAL_ROOT     = "/nfs/turbo/umd-hafiz/issf_server_data/famousfigures"
SPEAKER       = "Donald_Trump"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = os.path.join(BASE_DIR, "results/lfcc")
MODEL_DIR = os.path.join(BASE_DIR, "results/models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ======================================================
# LFCC (FROZEN)
# ======================================================
SR = 16000
DURATION = 10.0
N_FFT = 512
N_FILTERS = 60
N_LFCC = 20

# ======================================================
# DeepSVDD (FROZEN)
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
# HELPERS
# ======================================================
def resolve_path(p):
    return p.replace(PROTOCOL_ROOT, REAL_ROOT).replace("/Original/", "/-/")

def label_from_path(p):
    return "Real" if "/Original/" in p else "Fake"

def attack_from_path(p):
    if "/Original/" in p:
        return "Real"
    token = f"/{SPEAKER}/"
    return p.split(token, 1)[1].split("/", 1)[0]

# ======================================================
# LFCC EXTRACTION
# ======================================================
def extract_lfcc_vector(p):
    y, _ = librosa.load(resolve_path(p), sr=SR, duration=DURATION)
    feats = lfcc(
        sig=y,
        fs=SR,
        num_ceps=N_LFCC,
        nfilts=N_FILTERS,
        nfft=N_FFT
    )
    return np.mean(feats, axis=0)

def build_matrix(paths):
    return np.vstack([
        extract_lfcc_vector(p)
        for p in tqdm(paths, desc="LFCC")
    ])

# ======================================================
# METRICS
# ======================================================
def compute_eer(sr, sf):
    y = np.concatenate([np.zeros_like(sr), np.ones_like(sf)])
    s = np.concatenate([sr, sf])
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return fpr[idx], thr[idx], fpr, tpr

# ======================================================
# TRAINING
# ======================================================
def train_deepsvdd(X_train):
    model = DeepSVDD(X_train.shape[1]).to(DEVICE)
    X = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        center = model(X).mean(0)
        dist = torch.sum((model(X) - center) ** 2, dim=1)
        R = torch.sqrt(torch.quantile(dist, 1 - SVDD_NU))

    opt = optim.Adam(
        model.parameters(),
        lr=SVDD_LR,
        weight_decay=SVDD_WEIGHT_DECAY
    )

    model.train()
    for _ in range(SVDD_EPOCHS):
        opt.zero_grad()
        z = model(X)
        dist = torch.sum((z - center) ** 2, dim=1)
        loss = R**2 + (1 / SVDD_NU) * torch.mean(
            torch.clamp(dist - R**2, min=0)
        )
        loss.backward()
        opt.step()

        with torch.no_grad():
            dist = torch.sum((model(X) - center) ** 2, dim=1)
            R = torch.sqrt(torch.quantile(dist, 1 - SVDD_NU))

    tau = R.item() ** 2
    model.eval()
    return model, center, tau

# ======================================================
# MAIN
# ======================================================
def main():
    df = pd.read_csv(PROTOCOL_CSV)
    df["label"] = df.audiofilepath.apply(label_from_path)
    df["attack"] = df.audiofilepath.apply(attack_from_path)

    real_train = df[(df.label == "Real") & (df.split == "train")]
    eval_df = df[df.split == "eval"]
    real_eval = eval_df[eval_df.label == "Real"]
    fake_eval = eval_df[eval_df.label == "Fake"]

    # ---------- FEATURES ----------
    Xtr = build_matrix(real_train.audiofilepath.values)
    Xr  = build_matrix(real_eval.audiofilepath.values)
    Xf  = build_matrix(fake_eval.audiofilepath.values)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xr  = scaler.transform(Xr)
    Xf  = scaler.transform(Xf)

    # ---------- TRAIN ----------
    model, center, tau = train_deepsvdd(Xtr)

    with torch.no_grad():
        sr = torch.sum(
            (model(torch.tensor(Xr, dtype=torch.float32).to(DEVICE)) - center) ** 2,
            dim=1
        ).cpu().numpy()

        sf = torch.sum(
            (model(torch.tensor(Xf, dtype=torch.float32).to(DEVICE)) - center) ** 2,
            dim=1
        ).cpu().numpy()

    FRR = np.mean(sr > tau)
    FAR = np.mean(sf <= tau)
    eer, _, fpr, tpr = compute_eer(sr, sf)
    auc = roc_auc_score(
        np.concatenate([np.zeros_like(sr), np.ones_like(sf)]),
        np.concatenate([sr, sf])
    )

    print("\n=== LFCC + DeepSVDD ===")
    print(f"Dim: {Xtr.shape[1]}")
    print(f"FRR: {FRR:.4f} | FAR: {FAR:.4f}")
    print(f"EER: {eer:.4f} | AUC: {auc:.4f}")

    # ---------- SAVE MODEL ----------
    MODEL_PATH = os.path.join(MODEL_DIR, "lfcc_deepsvdd_trump.pt")
    torch.save({
        "net_state_dict": model.state_dict(),
        "center": center.detach().cpu(),
        "radius": torch.tensor(np.sqrt(tau)),
        "scaler": scaler
    }, MODEL_PATH)

    print(f"[SAVED] LFCC DeepSVDD → {MODEL_PATH}")

    # ---------- VISUALS ----------
    plt.figure()
    plt.hist(sr, bins=50, alpha=0.6, label="Real")
    plt.hist(sf, bins=50, alpha=0.6, label="Fake")
    plt.axvline(tau, color="red", linestyle="--")
    plt.legend()
    plt.title("LFCC – SVDD Score Distribution")
    plt.savefig(os.path.join(OUT_DIR, "score_distribution.png"))
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC – LFCC")
    plt.savefig(os.path.join(OUT_DIR, "roc.png"))
    plt.close()

    DetCurveDisplay(fpr=fpr, fnr=1 - tpr).plot()
    plt.title("DET – LFCC")
    plt.savefig(os.path.join(OUT_DIR, "det.png"))
    plt.close()

    print("\nSaved all LFCC outputs to:", OUT_DIR)

# ======================================================
if __name__ == "__main__":
    main()
